#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse PDFs from tat_docs.zip -> JSONL chunks (paragraph-level).

Input:
  - ./tat_docs.zip  (contains ./tat_docs/ with 170+ PDFs)
Output:
  - ./phase1/chunks.jsonl  (one JSON per line)

Each chunk includes:
  - chunk_id: str  (deterministic id: {doc_stem}_p{page}_c{para_index})
  - doc_name: str  (PDF filename)
  - doc_path: str  (relative path under ./tat_docs)
  - page: int      (1-based)
  - para_index: int (1-based within the page)
  - text: str
  - char_count: int
"""

import os
import re
import sys
import json
import zipfile
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as e:
    print("[ERROR] PyMuPDF not installed. Please run: pip install pymupdf")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent      # project root (has tat_docs.zip)
PHASE1_DIR = ROOT / "phase1"
ZIP_PATH = ROOT / "tat_docs.zip"
DOCS_DIR = ROOT / "tat_docs"
OUT_JSONL = PHASE1_DIR / "chunks.jsonl"


def ensure_unzipped(zip_path: Path, extract_dir: Path):
    """Unzip tat_docs.zip to ./tat_docs if needed."""
    if extract_dir.exists() and any(extract_dir.rglob("*.pdf")):
        print(f"[INFO] Using existing folder: {extract_dir}")
        return
    if not zip_path.exists():
        print(f"[ERROR] Zip not found: {zip_path}")
        sys.exit(1)
    print(f"[INFO] Unzipping {zip_path} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ROOT)
    print("[INFO] Unzip done.")


def normalize_block_text(text: str) -> str:
    """
    Normalize raw block text:
    - Convert CR to LF
    - Trim trailing spaces
    - Collapse 3+ newlines to 2 (to keep paragraph boundaries)
    - Strip outer whitespace
    """
    if not text:
        return ""
    t = text.replace("\r", "\n")
    # strip trailing spaces on each line
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    # collapse excessive blank lines to max 2 newlines (i.e., one blank line)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_into_paragraphs(text: str) -> list:
    """
    Split block text into paragraphs by blank lines.
    If a block is a single paragraph, returns [text].
    """
    if not text:
        return []
    # Split on one or more blank lines
    paras = re.split(r"\n\s*\n", text)
    # Clean each paragraph; drop very short noise
    cleaned = [p.strip() for p in paras if p and p.strip()]
    return cleaned


def iter_pdf_paragraphs(pdf_path: Path):
    """
    Yield (page_no, paragraph_texts) pairs where paragraph_texts is an ordered list.
    We use page.get_text('blocks') to respect PDF's text blocks, then split blocks by blank lines.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] Failed to open PDF: {pdf_path} ({e})")
        return

    if doc.is_encrypted:
        try:
            doc.authenticate("")

        except Exception:
            print(f"[WARN] Skipping encrypted PDF (cannot open): {pdf_path}")
            return

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        # Each block: (x0, y0, x1, y1, "text", block_no, ...); we sort by y, then x.
        try:
            blocks = page.get_text("blocks")
        except Exception as e:
            print(f"[WARN] get_text('blocks') failed on {pdf_path} page {page_idx+1}: {e}")
            continue

        # sort reading order
        blocks = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))

        paragraphs = []
        for b in blocks:
            if len(b) < 5:
                continue
            raw = b[4] or ""
            raw = normalize_block_text(raw)
            if not raw:
                continue
            # further split by blank lines (if the block has multiple paragraphs)
            paras = split_into_paragraphs(raw)
            paragraphs.extend(paras)

        yield (page_idx + 1, paragraphs)

    doc.close()


def main():
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    ensure_unzipped(ZIP_PATH, DOCS_DIR)

    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDFs found under {DOCS_DIR}")
        sys.exit(1)

    out_f = OUT_JSONL.open("w", encoding="utf-8")
    total_docs = 0
    total_pages = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        total_docs += 1
        rel_path = pdf_path.relative_to(DOCS_DIR)
        doc_name = pdf_path.name
        doc_stem = pdf_path.stem

        for page_no, paras in iter_pdf_paragraphs(pdf_path):
            total_pages += 1
            para_index = 0
            for para in paras:
                para_index += 1
                # drop super short noise paragraphs (e.g., 1-2 chars)
                if len(para.strip()) < 3:
                    continue

                chunk_id = f"{doc_stem}_p{page_no}_c{para_index}"
                record = {
                    "id": total_chunks + 1,
                    "chunk_id": chunk_id,
                    "doc_name": doc_name,
                    "page": page_no,
                    "para_index": para_index,
                    "text": para,
                    "char_count": len(para),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    out_f.close()
    print("\n[SUMMARY]")
    print(f"  PDFs processed : {total_docs}")
    print(f"  Pages visited  : {total_pages}")
    print(f"  Chunks written : {total_chunks}")
    print(f"  Output         : {OUT_JSONL}")


if __name__ == "__main__":
    main()
