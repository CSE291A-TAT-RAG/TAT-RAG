from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re, hashlib

import fitz
import tabula
import tiktoken

import json


TABLES_ENABLED = True                # [NEW-FAST] 是否启用表格抽取通道（整体开关）

CSV_PATH = "./csvs/"

def tok_len(s: str) -> int: return max(1, len(s)//4)

_enc = tiktoken.get_encoding("cl100k_base")
def tok_split(s: str, n: int) -> List[str]:
    ids = _enc.encode(s)
    return [_enc.decode(ids[i:i+n]) for i in range(0, len(ids), n)]

def md5(s: str) -> str: return hashlib.md5(s.encode("utf-8")).hexdigest()

@dataclass
class ParagraphBlock:
    text: str
    page: int
    section_path: str

@dataclass
class TableBlock:
    csv_file_name: str
    page: int
    table_id_for_page: int

HEADING_PATTERNS = [
    r"^\s*Table\s+of\s+Contents\b.*",
    r"^\s*Item\s+\d+[A-Z]?\b.*",
    r"^\s*Management[’']?s\s+Discussion\b.*",
    r"^\s*Consolidated\s+(Statements?|Statement)\b.*",
    r"^\s*Notes?\s+to\s+Consolidated\s+Financial\s+Statements\b.*",
    r"^\s*Report\s+of\s+Independent\b.*",
    r"^\s*Exhibit\s+Index\b.*",
    r"^\s*Signatures\b.*",
]

HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.I)

# -------------------------------------------------------------------
# ### CHANGE 1: 多列页面处理（列检测 + 列内排序 + 段落切分）
# -------------------------------------------------------------------

def _normalize_block_text(text: str) -> str:  # [NEW]
    """Normalize block text for robust paragraph splitting."""
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _split_into_paragraphs(text: str) -> List[str]:  # [NEW]
    """Split text by blank lines into paragraphs (trim short noise)."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p and p.strip()]

def _detect_columns(blocks: List[Tuple[float,float,float,float,str]], page_w: float) -> Optional[float]:  # [NEW]
    """
    Detect a single vertical gutter (2-column layout). Return x position of gutter if found, else None.
    """
    if not blocks or len(blocks) < 8:  # too few blocks -> likely single column
        return None
    mids = sorted([((b[0]+b[2])/2.0) for b in blocks])  # mid-x
    gaps = []
    for i in range(1, len(mids)):
        gaps.append((mids[i]-mids[i-1], (mids[i-1]+mids[i])/2.0))  # (gap_size, gap_center)
    if not gaps:
        return None
    gaps.sort(reverse=True, key=lambda x: x[0])
    best_gap, gutter_x = gaps[0]
    if best_gap < 0.08 * page_w:  # threshold: 8% page width
        return None
    # ensure both sides have enough blocks
    left_ct = sum(1 for b in blocks if (b[0]+b[2])/2.0 < gutter_x)
    right_ct = len(blocks) - left_ct
    if left_ct >= 3 and right_ct >= 3:
        return gutter_x
    return None

def _assign_blocks_to_columns(blocks: List[Tuple[float,float,float,float,str]], gutter_x: float) -> Tuple[List, List]:  # [NEW]
    """Split blocks into (left_col_blocks, right_col_blocks) by gutter."""
    left, right = [], []
    for b in blocks:
        midx = (b[0]+b[2])/2.0
        (left if midx < gutter_x else right).append(b)
    # Sort inside each column by (y0, x0)
    left.sort(key=lambda b: (round(b[1],2), round(b[0],2)))
    right.sort(key=lambda b: (round(b[1],2), round(b[0],2)))
    return left, right

def _blocks_to_paragraphs(blocks: List[Tuple[float,float,float,float,str]]) -> List[str]:  # [NEW]
    """Concatenate blocks in order, then split into paragraphs by blank lines."""
    paras: List[str] = []
    for b in blocks:
        if len(b) < 5:
            continue
        raw = _normalize_block_text(b[4])
        if not raw:
            continue
        parts = _split_into_paragraphs(raw)
        paras.extend(parts)
    return paras

def extract_paragraphs_with_sections(pdf_path: str) -> List[ParagraphBlock]:
    blocks: List[ParagraphBlock] = []
    doc = fitz.open(pdf_path)
    current_section = "None"
    for pno in range(len(doc)):
        page = doc[pno]

        # 改为 blocks 模式以支持列处理
        try:                                                                               # [MOD-FAST]
            bks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
        except Exception:
            # fallback to raw text if blocks fail (rare)
            text = page.get_text("text")
            chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
            for c in chunks:
                lines = c.splitlines()
                head = lines[0].strip() if lines else ""
                if HEADING_RE.match(head):
                    current_section = head
                blocks.append(ParagraphBlock(text=c, page=pno+1, section_path=current_section))
            continue

        # 过滤空文本块
        bks = [b for b in bks if len(b) >= 5 and (b[4] or "").strip()]

        # 检测两栏
        page_w = page.rect.width
        gutter_x = _detect_columns(bks, page_w)

        # 根据是否两栏分别构建段落
        if gutter_x is not None:
            left, right = _assign_blocks_to_columns(bks, gutter_x)
            ordered_columns = [left, right]
        else:
            # 单栏：整体按 y,x 排序
            bks.sort(key=lambda b: (round(b[1],2), round(b[0],2)))
            ordered_columns = [bks]

        # 逐列产出段落，确保“只在同一列内合并”，避免 aabb 拼接
        for col_blocks in ordered_columns:
            paras = _blocks_to_paragraphs(col_blocks)
            for c in paras:
                if not c.strip():
                    continue
                head = c.splitlines()[0].strip() if c else ""
                if HEADING_RE.match(head):
                    current_section = head
                blocks.append(ParagraphBlock(text=c, page=pno+1, section_path=current_section))

    doc.close()
    return blocks

# -------------------------------------------------------------------
# ### table extraction
# -------------------------------------------------------------------

def extract_tables(pdf_path: str, doc_id: str) -> List[TableBlock]:
    tbls: List[TableBlock] = []

    doc = fitz.open(pdf_path)
    tbls = []
    for pno in range(len(doc)):  # 0-based
        dfs = tabula.read_pdf(pdf_path, pages=pno+1, multiple_tables=True, stream=True)
        if not dfs:
            continue
        for df_i, df in enumerate(dfs):
            fname = f"{doc_id}_page_{pno}_table_{df_i}.csv"
            path = Path(CSV_PATH) / fname

            df.to_csv(path, compression="infer")
            tbls.append(TableBlock(
                csv_file_name=str(path),
                page=pno+1,
                table_id_for_page=df_i,
            ))
        
    return tbls
    

# -------------------------------------------------------------------
# chunking
# -------------------------------------------------------------------

def chunk_paragraph(text: str, size: int = 640, overlap: int = 96) -> List[str]:
    if tok_len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    parts = tok_split(text, step)
    res = []
    prev_tail = ""
    for i, p in enumerate(parts):
        cur = (prev_tail + p) if i > 0 else p
        if tok_len(cur) > size:
            cur = tok_split(cur, size)[0]
        res.append(cur)
        tail_tokens = tok_split(cur, max(1, tok_len(cur)-overlap))
        prev_tail = tail_tokens[-1] if tail_tokens else ""
    return res

def build_chunks_for_financial_report(
    pdf_path: str,
    doc_meta: Dict,
    para_size: int = 640,
    para_overlap: int = 96,
) -> List[Dict]:
    out: List[Dict] = []
    doc_id = doc_meta.get("doc_id") or md5(pdf_path)

    # 使用新的多列安全段落抽取
    paras = extract_paragraphs_with_sections(pdf_path)

    order = 0
    for pb in paras:
        splits = chunk_paragraph(pb.text, para_size, para_overlap)
        for j, s in enumerate(splits):
            md = {
                **doc_meta,
                "doc_id": doc_id,
                "type": "paragraph",
                "section_path": pb.section_path,
                "page": pb.page,
                "order": order,
                "inner_index": j,
                "tokens": tok_len(s),
            }
            md["hash"] = md5(f"{doc_id}|p|{pb.page}|{order}|{j}|{s[:80]}")
            out.append({"text": s, "metadata": md})
        order += 1

    if TABLES_ENABLED:
        tables = extract_tables(pdf_path, doc_id)
        for t_idx, tb in enumerate(tables):
            md = {
                **doc_meta,
                "doc_id": doc_id,
                "type": "table",
                "page": tb.page,
                "inner_index": tb.table_id_for_page,
                "table_id": f"T{t_idx}",
            }
            md["hash"] = md5(f"{doc_id}|t|{tb.page}|{tb.table_id_for_page}|{t_idx}")
            out.append({"text": tb.csv_file_name, "metadata": md})

    return out

# example usage
if __name__ == "__main__":
    Path(CSV_PATH).mkdir(parents=True, exist_ok=True)

    # set data folder path
    data_folder = Path("../tat_docs_filtered/")
    pdf_files = sorted(data_folder.glob("*.pdf"))

    exclude_file_path = "./not_included.txt"
    exclude_files = set()

    if Path(exclude_file_path).exists():
        with open(exclude_file_path, 'r') as f:
            exclude_files = {f"{line.strip()}.pdf" for line in f if line.strip()}

    filtered_pdf_files = [f for f in pdf_files if f.name not in exclude_files]

    total_chunks = []

    print(f"[INFO] Found {len(pdf_files)} PDF files under {data_folder}")
    print(f"[INFO] Excluded {len(pdf_files) - len(filtered_pdf_files)} PDF files")
    print(f"[INFO] Remaining {len(filtered_pdf_files)} PDF files to process")

    for idx, f in enumerate(filtered_pdf_files, start=1):
        doc_stem = f.stem
        print(f"[{idx}/{len(filtered_pdf_files)}] Processing {doc_stem} ...")
        try:
            chunks = build_chunks_for_financial_report(
                str(f),
                {"doc_id": doc_stem}   # 自动使用文件名作为 doc_id
            )
            total_chunks.extend(chunks)
        except Exception as e:
            print("\033[91m[WARN] Failed on {f}: {e}\033[0m".format(f=f, e=e))
            with open("./log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Error processing {f}: {e}\n")
            continue

    print("\n[SUMMARY]")
    print(f"  Total PDFs processed : {len(pdf_files)}")
    print(f"  Total chunks created  : {len(total_chunks)}")

    out_path = Path("chunks_all.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for ck in total_chunks:
            fout.write(json.dumps(ck, ensure_ascii=False) + "\n")
    print(f"  Output written to: {out_path.resolve()}")
