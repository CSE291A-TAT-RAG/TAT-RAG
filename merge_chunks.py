#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import zipfile
from typing import Dict, Any, Optional, List


def get_token_count_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    tokens = meta.get("tokens")
    if tokens is None:
        return None
    try:
        return int(tokens)
    except (ValueError, TypeError):
        return None


def get_token_count(chunk: Dict[str, Any]) -> int:
    """优先用 metadata['tokens']，没有就用简单分词数估。"""
    meta = chunk.get("metadata", {}) or {}
    tokens = get_token_count_from_meta(meta)
    if tokens is not None:
        return tokens
    return len(chunk.get("text", "").split())


def can_merge(prev_chunk: Dict[str, Any], curr_chunk: Dict[str, Any]) -> bool:
    """
    判断两个 chunk 是否允许合并：
    - doc_id 相同
    - page 相同
    不去强制限制 section_path，避免把表格拆得太碎，
    但在 metadata 里记录所有参与的 section_path。
    """
    prev_meta = prev_chunk.get("metadata", {}) or {}
    curr_meta = curr_chunk.get("metadata", {}) or {}

    if prev_meta.get("doc_id") != curr_meta.get("doc_id"):
        return False
    if prev_meta.get("page") != curr_meta.get("page"):
        return False

    return True


def merge_two_chunks(
    base_chunk: Dict[str, Any],
    new_chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """
    把 new_chunk 合并到 base_chunk 里，返回更新后的 base_chunk。
    """
    base_text = base_chunk.get("text", "") or ""
    new_text = new_chunk.get("text", "") or ""

    # 文本之间留一个空行，避免粘成一行
    if base_text.endswith("\n"):
        sep = ""
    else:
        sep = "\n\n"
    base_chunk["text"] = base_text + sep + new_text

    base_meta = base_chunk.setdefault("metadata", {})
    new_meta = new_chunk.get("metadata", {}) or {}

    # 更新 tokens
    base_tokens = get_token_count(base_chunk)
    new_tokens = get_token_count(new_chunk)
    base_meta["tokens"] = base_tokens + new_tokens

    # 记录合并过来的 hash
    base_hash = base_meta.get("hash")
    new_hash = new_meta.get("hash")

    if "merged_hashes" not in base_meta:
        merged: List[str] = []
        if isinstance(base_hash, str):
            merged.append(base_hash)
        base_meta["merged_hashes"] = merged

    if isinstance(new_hash, str):
        base_meta["merged_hashes"].append(new_hash)

    # 记录合并的 order（方便 debug）
    base_order = base_meta.get("order")
    new_order = new_meta.get("order")
    if "merged_orders" not in base_meta:
        orders: List[int] = []
        if isinstance(base_order, int):
            orders.append(base_order)
        base_meta["merged_orders"] = orders
    if isinstance(new_order, int):
        base_meta["merged_orders"].append(new_order)

    # type 合并信息：type_merged 记录所有出现过的类型
    type_set = set()

    def add_type(t):
        if not t:
            return
        if isinstance(t, str):
            for part in t.split("+"):
                part = part.strip()
                if part:
                    type_set.add(part)
        elif isinstance(t, list):
            for part in t:
                if isinstance(part, str) and part.strip():
                    type_set.add(part.strip())

    add_type(base_meta.get("type"))
    add_type(new_meta.get("type"))

    if len(type_set) > 1:
        base_meta["type_merged"] = sorted(type_set)

    # section_path 合并信息：section_paths_merged 记录所有出现过的 section_path
    sec_set = set()
    base_sec = base_meta.get("section_path")
    new_sec = new_meta.get("section_path")
    if base_sec is not None:
        sec_set.add(str(base_sec))
    if new_sec is not None:
        sec_set.add(str(new_sec))
    if len(sec_set) > 1:
        base_meta["section_paths_merged"] = sorted(sec_set)

    return base_chunk


def merge_chunks(
    input_zip: str,
    input_jsonl: str,
    output_jsonl: str,
    min_tokens: int = 40,
    small_chunk_tokens: int = 15,
    max_tokens: int = 350,
) -> None:
    """
    从 zip 里的 jsonl 读入 chunk，合并短文本 chunk，并写出新的 jsonl。
    """
    with zipfile.ZipFile(input_zip, "r") as zf:
        with zf.open(input_jsonl, "r") as f_in, open(
            output_jsonl, "w", encoding="utf-8"
        ) as f_out:

            current_chunk: Optional[Dict[str, Any]] = None

            for raw_line in f_in:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                chunk = json.loads(line)

                if current_chunk is None:
                    current_chunk = chunk
                    continue

                # 判断是否可以和 current_chunk 合并（doc_id + page）
                if not can_merge(current_chunk, chunk):
                    # 边界：直接输出 current_chunk，换成新的
                    json.dump(current_chunk, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    current_chunk = chunk
                    continue

                curr_tokens = get_token_count(chunk)
                cur_tokens = get_token_count(current_chunk)

                # 如果当前已经太长了，直接 flush，再从新 chunk 开始
                if cur_tokens >= max_tokens:
                    json.dump(current_chunk, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    current_chunk = chunk
                    continue

                # 合并条件：
                # 1) current_chunk 还没到 min_tokens
                #    或 2) 新 chunk 非常短（small_chunk_tokens 以下）
                if cur_tokens < min_tokens or curr_tokens < small_chunk_tokens:
                    current_chunk = merge_two_chunks(current_chunk, chunk)
                else:
                    # 两边都不短，那就保持独立
                    json.dump(current_chunk, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    current_chunk = chunk

            # 文件结束别忘了最后一个
            if current_chunk is not None:
                json.dump(current_chunk, f_out, ensure_ascii=False)
                f_out.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge short RAG chunks from chunks_all.jsonl inside a zip file."
    )
    parser.add_argument(
        "--input_zip",
        type=str,
        default="chunks_all.zip",
        help="Input zip file containing the jsonl (default: chunks_all.zip)",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="chunks_all.jsonl",
        help="Name of the jsonl file inside the zip (default: chunks_all.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunks_all_merged.jsonl",
        help="Output jsonl file path (default: chunks_all_merged.jsonl)",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=40,
        help="Target minimum tokens for a merged chunk (default: 40)",
    )
    parser.add_argument(
        "--small_chunk_tokens",
        type=int,
        default=20,
        help="Threshold under which a chunk is considered small and will be merged into neighbor (default: 20)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=350,
        help="Maximum tokens allowed for a merged chunk (default: 350)",
    )

    args = parser.parse_args()

    merge_chunks(
        input_zip=args.input_zip,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output,
        min_tokens=args.min_tokens,
        small_chunk_tokens=args.small_chunk_tokens,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
