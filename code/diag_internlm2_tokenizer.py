#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InternLM2 reward 模型 tokenizer 加载诊断脚本。
用法（从项目根目录）：python code/diag_internlm2_tokenizer.py
"""
import sys

MODEL_DIR = "code/model/internlm2-1_8b-reward"

print("=" * 60)
print("环境版本")
print("=" * 60)
import transformers
print(f"transformers : {transformers.__version__}")
try:
    import sentencepiece as spm
    print(f"sentencepiece: {spm.__version__}")
except ImportError:
    print("sentencepiece: 未安装")
    spm = None
try:
    import google.protobuf
    print(f"protobuf     : {google.protobuf.__version__}")
except ImportError:
    print("protobuf     : 未安装")

print()
print("=" * 60)
print("① sentencepiece 直接加载 tokenizer.model")
print("=" * 60)
if spm is not None:
    try:
        sp = spm.SentencePieceProcessor(model_file=MODEL_DIR + "/tokenizer.model")
        print(f"[OK] vocab size = {sp.get_piece_size()}")
        print(f"     piece('<s>')        = {sp.piece_to_id('<s>')}")
        print(f"     piece('<|reward|>') = {sp.piece_to_id('<|reward|>')}")
        print(f"     encode('你好')      = {sp.encode('你好')}")
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        print("       -> 大概率是 sentencepiece/protobuf 版本不兼容")
else:
    print("[SKIP] 未安装 sentencepiece")

print()
print("=" * 60)
print("② AutoTokenizer(use_fast=False) 完整加载")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    print(f"[OK] 返回类型 = {type(tok).__name__}, vocab_size = {getattr(tok, 'vocab_size', None)}")
    print(f"     apply_chat_template 可用: {hasattr(tok, 'apply_chat_template')}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")

print()
print("=" * 60)
print("③ AutoTokenizer 默认(fast) 完整加载")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"[OK] 返回类型 = {type(tok).__name__}, vocab_size = {getattr(tok, 'vocab_size', None)}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {str(e)[:200]}")
