#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_v10_final_optimized.py — 资源管理优化版
---------------------------------------------------------------------------
1. 单点下载：将 CNN 权重下载逻辑移至 main 函数，避免子进程冲突。
2. 线程隔离：每个子进程根据 CPU 总核数自动分配线程，防止争抢。
3. 高精滑动窗口：1024 窗口 + 64 重叠，保证长序列拼接处的精度。
4. 官方清洗：集成 Heinzinger 的标准氨基酸过滤，确保推理稳定性。
5. 核心保留：Decoder/CNN 切换、精确 Embedding Slicing ([:, 1:])。
"""

import argparse
import time
import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel, set_seed
from urllib import request

set_seed(42)

# ---------------------------
# 常量与配置
# ---------------------------
SS_MAPPING = {i: c for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
CNN_WEIGHTS_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"
CNN_LOCAL_PATH = Path.cwd() / "cnn_chkpnt" / "model.pt"

# ---------------------------
# CNN 架构 (Heinzinger 原版)
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        return self.classifier(x).squeeze(dim=-1)

# 全局变量（子进程内部使用）
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_PREDICTOR = None
_GLOBAL_IS_3DI = None
_GLOBAL_ENCODER_ONLY = None

# ---------------------------
# 下载逻辑（仅主进程调用一次）
# ---------------------------
def ensure_cnn_weights():
    if not CNN_LOCAL_PATH.exists():
        print(f"📥 CNN weights not found. Downloading from {CNN_WEIGHTS_URL}...")
        CNN_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 增加 User-Agent 避免某些环境下载失败
        req = request.Request(CNN_WEIGHTS_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with request.urlopen(req) as response, open(CNN_LOCAL_PATH, 'wb') as f:
            f.write(response.read())
        print("✅ Download complete.")
    else:
        print("✅ CNN weights already exist. Skipping download.")

# ---------------------------
# Worker 初始化逻辑
# ---------------------------
def init_worker(model_dir, is_3Di, encoder_only, threads_per_proc):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_IS_3DI, _GLOBAL_ENCODER_ONLY
    
    torch.set_num_threads(threads_per_proc)
    device = torch.device('cpu')
    
    # 子进程只负责从本地路径加载
    tokenizer = T5Tokenizer.from_pretrained(model_dir, local_files_only=True)
    if encoder_only:
        model = T5EncoderModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        predictor = CNN().to(device).eval()
        # 加载本地权重，不再检查下载
        predictor.load_state_dict(torch.load(CNN_LOCAL_PATH, map_location='cpu')["state_dict"])
        _GLOBAL_PREDICTOR = predictor
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device).eval()

    _GLOBAL_MODEL, _GLOBAL_TOKENIZER = model, tokenizer
    _GLOBAL_IS_3DI, _GLOBAL_ENCODER_ONLY = is_3Di, encoder_only

# ---------------------------
# 推理核心与滑动窗口
# ---------------------------
def translate_core(seq):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PREDICTOR, _GLOBAL_IS_3DI, _GLOBAL_ENCODER_ONLY
    
    # 官方清洗逻辑
    clean_seq = "".join([aa if aa in STANDARD_AA else 'X' for aa in seq.upper()])
    prefix = "<fold2AA>" if _GLOBAL_IS_3DI else "<AA2fold>"
    encoded = _GLOBAL_TOKENIZER(prefix + " " + " ".join(list(clean_seq)), return_tensors='pt').to('cpu')

    with torch.no_grad():
        if _GLOBAL_ENCODER_ONLY:
            emb_full = _GLOBAL_MODEL(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
            # 精确 Slicing：剔除 BOS，对齐残基特征
            residue_emb = emb_full[:, 1:] 
            prediction = _GLOBAL_PREDICTOR(residue_emb)
            probs = F.softmax(prediction, dim=1)
            max_p, max_idx = torch.max(probs, dim=1)
            
            out_seq = "".join([SS_MAPPING[i] for i in max_idx[0].numpy()])
            avg_conf = float(max_p.mean().item())
            return out_seq, avg_conf, residue_emb.squeeze(0)
        else:
            output = _GLOBAL_MODEL.generate(**encoded, max_length=len(clean_seq)+1, do_sample=False)
            decoded = _GLOBAL_TOKENIZER.decode(output[0], skip_special_tokens=True)
            return "".join(decoded.split()), 1.0, None

def smart_process(seq_id, seq):
    L = len(seq)
    window, overlap = 1024, 64
    
    if L <= window:
        o, c, e = translate_core(seq)
        return {"id": seq_id, "out": o, "conf": c, "emb": e, "len": L}
    
    parts, confs, embs = [], [], []
    for start in range(0, L, window - overlap):
        end = min(L, start + window)
        o, c, e = translate_core(seq[start:end])
        if not parts:
            parts.append(o)
            if e is not None: embs.append(e)
        else:
            parts.append(o[overlap:])
            if e is not None: embs.append(e[overlap:])
        confs.append(c)
        if end >= L: break
        
    return {"id": seq_id, "out": "".join(parts), "conf": sum(confs)/len(confs), "emb": torch.cat(embs) if embs else None, "len": L}

def worker_fn(args_tuple):
    sid, seq, out_dir = args_tuple
    res = smart_process(sid, seq)
    Path(out_dir).joinpath(f"{sid}_3di.txt").write_text(f">{sid}\n{res['out']}\n")
    
    emb_file = ""
    if res["emb"] is not None:
        emb_path = Path(out_dir) / f"{sid}_embedding.pt"
        torch.save(res["emb"].half(), emb_path)
        emb_file = emb_path.name
        
    return {"seq_id": sid, "seq_len": res["len"], "avg_conf": res["conf"], "emb_file": emb_file}

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--encoder_only', type=int, default=1)
    parser.add_argument('--is_3Di', type=int, default=0)
    args = parser.parse_args()

    # 1. 资源计算与下载（单点执行）
    if args.encoder_only:
        ensure_cnn_weights()
    
    threads_per_proc = max(1, cpu_count() // args.nproc)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 读取数据
    seqs = {}
    with open(args.input, 'r') as f:
        curr = ""
        for line in f:
            if line.startswith(">"): curr = line[1:].split()[0]; seqs[curr] = ""
            else: seqs[curr] += line.strip()

    print(f"🚀 Integrated Mode: nproc={args.nproc}, threads/proc={threads_per_proc}")
    
    # 3. 启动进程池
    job_args = [(sid, seq, args.output) for sid, seq in seqs.items()]
    with Pool(processes=args.nproc, initializer=init_worker, 
              initargs=(args.model, bool(args.is_3Di), bool(args.encoder_only), threads_per_proc)) as pool:
        results = list(tqdm(pool.imap_unordered(worker_fn, job_args), total=len(job_args)))

    # 4. 保存 Summary
    with open(out_dir / "summary.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["seq_id", "seq_len", "avg_conf", "emb_file"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Success. Output: {args.output}")

if __name__ == "__main__":
    main()