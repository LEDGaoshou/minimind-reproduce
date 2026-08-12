#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据清洗脚本 —— 编码修复 + 垃圾行过滤（不覆盖原始文件）

背景：
    data/sft_t2t_mini.jsonl 中混有 UTF-8 被误用 GBK 解码产生的乱码（mojibake）行，
    例如 "涓€涓�帰闄╅槦" 实际是 "一个探险队" 的字节被错误解码的结果。
    这类错误大多可逆：把乱码字符串按 GBK 编码回字节，再按 UTF-8 解码即可还原原文。

处理逻辑（逐行流式，内存友好）：
    1. JSON 解析失败的行 → 丢弃（记数）
    2. 对每个字符串字段（content / reasoning_content / tools / tool_calls 等）：
       - 尝试 GBK 反转修复（中文数据主因），失败再试 latin-1/cp1252 反转（"â€" 型欧洲乱码）
       - 修复结果需通过校验（含可读字符、中文密度不降、无替换符 �），防止误伤正常文本
    3. 样本级判定：修复后任一字段仍含替换符 � 或无可读字符（CJK/英文字母）→ 丢弃
    4. 其余样本以原结构写出（ensure_ascii=False，保持中文可读）

用法：
    python code/clean_sft_data.py \
        --input data/sft_t2t_mini.jsonl \
        --output data/sft_t2t_mini_clean.jsonl

默认输出 data/sft_t2t_mini_clean.jsonl，不会修改原始文件。
"""
import argparse
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _cjk_ratio(s):
    """中文字符占比，用于修复结果的合理性校验。"""
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(s)


def _readable_chars(s):
    """是否含可读字符（CJK 或英文字母）。"""
    return any(
        "\u4e00" <= ch <= "\u9fff" or (ch.isascii() and ch.isalpha()) for ch in s
    )


def fix_gbk_mojibake(s):
    """尝试反转 UTF-8 -> GBK 误解码（中文乱码主因）。

    返回修复后的字符串；修复不可信时返回 None。
    """
    if not s:
        return None
    try:
        fixed = s.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    # 校验：必须有实质变化、无替换符、含可读字符、中文密度不降（防误反转正常文本）
    if fixed == s or "\ufffd" in fixed:
        return None
    if not _readable_chars(fixed):
        return None
    if _cjk_ratio(fixed) + 1e-9 < _cjk_ratio(s):
        return None
    return fixed


def fix_latin1_mojibake(s):
    """尝试反转 UTF-8 -> latin-1/cp1252 误解码（"â€" 型，欧洲数据常见）。"""
    if not s or ("â€" not in s and "Ã" not in s):
        return None
    try:
        fixed = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if fixed == s or "\ufffd" in fixed:
        return None
    if not _readable_chars(fixed):
        return None
    return fixed


def clean_value(v):
    """递归遍历 JSON 树，对每个字符串字段尝试乱码修复。"""
    if isinstance(v, str):
        for fixer in (fix_gbk_mojibake, fix_latin1_mojibake):
            fixed = fixer(v)
            if fixed:
                return fixed
        return v
    if isinstance(v, list):
        return [clean_value(x) for x in v]
    if isinstance(v, dict):
        return {k: clean_value(x) for k, x in v.items()}
    return v


def value_readable(v):
    """递归检查：所有字符串均可读（非空时含 CJK/英文字母且无替换符）。"""
    if isinstance(v, str):
        if not v:
            return True
        if "\ufffd" in v:
            return False
        return _readable_chars(v)
    if isinstance(v, list):
        return all(value_readable(x) for x in v)
    if isinstance(v, dict):
        return all(value_readable(x) for x in v.values())
    return True


def main():
    parser = argparse.ArgumentParser(
        description="清洗 SFT 数据（编码修复 + 垃圾行过滤），不覆盖原始文件"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(DATA_DIR, "sft_t2t_mini.jsonl"),
        help="原始数据路径（默认 data/sft_t2t_mini.jsonl）",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "sft_t2t_mini_clean.jsonl"),
        help="清洗结果输出路径（默认 data/sft_t2t_mini_clean.jsonl）",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("错误：输出路径不能与输入路径相同（本脚本不覆盖原始文件）", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在：{args.input}", file=sys.stderr)
        sys.exit(1)

    total = fixed = dropped = json_errors = 0
    with open(args.input, encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            if line_no % 100000 == 0:
                print(
                    f"...已处理 {line_no:,} 行（保留 {total - dropped - json_errors:,}，"
                    f"丢弃 {dropped:,}，JSON错误 {json_errors:,}）",
                    file=sys.stderr,
                )
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            cleaned = clean_value(sample)
            if cleaned != sample:
                fixed += 1
            if not value_readable(cleaned):
                dropped += 1
                continue
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    kept = total - dropped - json_errors
    print("=" * 60)
    print(f"输入   : {args.input}")
    print(f"输出   : {args.output}")
    print(f"总样本 : {total:,}")
    print(f"保留   : {kept:,}（{kept / max(total, 1) * 100:.2f}%）")
    print(f"编码修复: {fixed:,}（含乱码被反转还原的行）")
    print(f"丢弃   : {dropped:,}（无法修复/无可读内容）")
    print(f"JSON错误: {json_errors:,}")


if __name__ == "__main__":
    main()
