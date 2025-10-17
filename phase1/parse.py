from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re, hashlib

import fitz
import pdfplumber

# ---------- 性能相关开关 ----------
FAST_MODE = True                     # [NEW-FAST] 快速模式：只在疑似有表格的页做表格抽取
TABLES_ENABLED = True                # [NEW-FAST] 是否启用表格抽取通道（整体开关）
TABLE_FLAVOR = "stream"              # [NEW-FAST] Camelot 优先使用 stream（更快更鲁棒）
TRY_LATTICE_FALLBACK = True          # [NEW-FAST] 对候选页 stream 无果时，再尝试 lattice 少量页
CALC_TOKENS_STRICT = False           # [NEW-FAST] 是否用 tiktoken 精确计数；关掉用近似法更快

# for table extraction
try:
    import camelot  # for tables (lattice/stream)
    _has_camelot = True
except Exception:
    _has_camelot = False

# [保持禁用 tabula]
# try:
#     import tabula  # requires Java
#     _has_tabula = True
# except Exception:
#     _has_tabula = False

# tokenizer
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def tok_len(s: str) -> int: return len(_enc.encode(s))
    def tok_split(s: str, n: int) -> List[str]:
        ids = _enc.encode(s)
        return [_enc.decode(ids[i:i+n]) for i in range(0, len(ids), n)]
except Exception:
    def tok_len(s: str) -> int: return max(1, len(s)//4)
    def tok_split(s: str, n: int) -> List[str]:
        step = n*4
        return [s[i:i+step] for i in range(0, len(s), step)]

# [NEW-FAST] 更快的 token 计数包装（允许近似法）
def tok_count(s: str) -> int:
    if CALC_TOKENS_STRICT:
        return tok_len(s)
    # 近似：字数/4（与上面 fallback 保持一致），大幅提速
    return max(1, len(s) // 4)

def md5(s: str) -> str: return hashlib.md5(s.encode("utf-8")).hexdigest()

@dataclass
class ParagraphBlock:
    text: str
    page: int
    section_path: str

@dataclass
class TableBlock:
    rows: List[List[str]]  # rows[0] is table header
    page: int
    title: str
    currency: Optional[str] = None
    unit: Optional[str] = None

HEADING_PATTERNS = [
    r"^Item\s+1A?\b.*",  # Item 1 / 1A Risk Factors
    r"^Item\s+7\b.*",    # Item 7 MD&A
    r"^Item\s+7A\b.*",
    r"^Item\s+8\b.*",    # Financial Statements and Supplementary Data
    r"^Management.?s Discussion.*",
    r"^Consolidated\s+(Statements?|Balance Sheets?|Cash Flows?).*",
    r"^Notes?\s+to\s+Consolidated\s+Financial\s+Statements.*",
    r"^Risk\s+Factors.*",
] # TODO
HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.I)

CURRENCY_HINT = re.compile(r"\b(USD|US\$|\$|HKD|EUR|GBP|RMB|CNY)\b")
UNIT_HINT = re.compile(r"\b(thousands|millions|billions|’000|000s)\b", re.I)

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
    current_section = "Front"
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
# ### CHANGE 2: 表格抽取增强 + 提速（候选页检测 + stream优先 + 少量lattice兜底）
# -------------------------------------------------------------------

def _rows_clean(rows: List[List[str]]) -> List[List[str]]:  # [NEW]
    """Basic cleanup for rows: join multiline cells, strip whitespace, drop all-empty rows."""
    out = []
    for row in rows:
        if row is None:
            continue
        clean = [((c or "").replace("\n", " ").strip()) for c in row]
        if any(cell for cell in clean):
            out.append(clean)
    return out

# [NEW-FAST] 轻量级“是否疑似有表格”页级检测
TABLE_TEXT_HINT = re.compile(
    r"\b(Consolidated|Balance\s+Sheets?|Statements?|Cash\s+Flows?|Total|Net|%|\$)\b",
    re.I
)

def _page_has_table_hint(plumber_page: "pdfplumber.page.Page") -> bool:  # [NEW-FAST]
    """
    启发式：
      - 线/矩形/曲线对象较多，或者
      - 文本包含典型表头关键词/货币符号/百分号
    命中任一则认为该页“疑似有表格”。
    """
    try:
        line_ct = len(getattr(plumber_page, "lines", []) or [])
        rect_ct = len(getattr(plumber_page, "rects", []) or [])
        curve_ct = len(getattr(plumber_page, "curves", []) or [])
        if line_ct + rect_ct + curve_ct >= 8:  # 阈值可调
            return True
        txt = plumber_page.extract_text() or ""
        if TABLE_TEXT_HINT.search(txt):
            return True
    except Exception:
        return False
    return False

def extract_tables(pdf_path: str) -> List[TableBlock]:
    if not TABLES_ENABLED:                            # [NEW-FAST]
        return []                                     # [NEW-FAST]

    tbls: List[TableBlock] = []

    # 先用 pdfplumber 做“页级候选检测”，只对候选页跑后续重抽取
    candidate_pages: List[int] = []                   # [NEW-FAST]
    with pdfplumber.open(pdf_path) as pdf:           # [NEW-FAST]
        for pno, page in enumerate(pdf.pages, start=1):
            if (not FAST_MODE) or _page_has_table_hint(page):
                candidate_pages.append(pno)

        # 如果没有候选页，直接返回（大幅提速）
        if not candidate_pages:                       # [NEW-FAST]
            return []                                 # [NEW-FAST]

        pages_str = ",".join(map(str, candidate_pages))

        # 1) Camelot stream 优先（仅候选页）
        if _has_camelot:                              # [NEW-FAST]
            try:
                t_stream = camelot.read_pdf(pdf_path, flavor=TABLE_FLAVOR, pages=pages_str)
                for t in t_stream:
                    rows = [list(map(str, r)) for r in t.df.values.tolist()]
                    rows = _rows_clean(rows)
                    if rows:
                        title = rows[0][0].strip() if rows[0] and len(rows[0][0]) < 120 else "Table"
                        meta = " ".join(sum(rows[:3], []))[:500]
                        currency = (CURRENCY_HINT.search(meta).group(0) if CURRENCY_HINT.search(meta) else None)
                        unit = (UNIT_HINT.search(meta).group(0) if UNIT_HINT.search(meta) else None)
                        page_no = getattr(t, "page", 0) or 0
                        page_no = int(page_no) if page_no else candidate_pages[0]
                        tbls.append(TableBlock(rows=rows, page=page_no, title=title, currency=currency, unit=unit))
            except Exception:
                pass

            # 1b) 对“少量候选页”再尝试 lattice 兜底（仅当 stream 几乎没结果时）
            if TRY_LATTICE_FALLBACK and len(tbls) == 0:   # [NEW-FAST]
                try:
                    t_lat = camelot.read_pdf(pdf_path, flavor="lattice", pages=pages_str)
                    for t in t_lat:
                        rows = [list(map(str, r)) for r in t.df.values.tolist()]
                        rows = _rows_clean(rows)
                        if rows:
                            title = rows[0][0].strip() if rows[0] and len(rows[0][0]) < 120 else "Table"
                            meta = " ".join(sum(rows[:3], []))[:500]
                            currency = (CURRENCY_HINT.search(meta).group(0) if CURRENCY_HINT.search(meta) else None)
                            unit = (UNIT_HINT.search(meta).group(0) if UNIT_HINT.search(meta) else None)
                            page_no = getattr(t, "page", 0) or 0
                            page_no = int(page_no) if page_no else candidate_pages[0]
                            tbls.append(TableBlock(rows=rows, page=page_no, title=title, currency=currency, unit=unit))
                except Exception:
                    pass

        # 2) pdfplumber 作为补充（仅候选页），即使 Camelot 有结果也可以补充一些未抓到的表
        try:
            for pno in candidate_pages:
                page = pdf.pages[pno-1]
                tables = page.extract_tables()
                for rows in tables:
                    rows = _rows_clean(rows)
                    if rows:
                        meta = " ".join(sum(rows[:2], []))[:500]
                        currency = (CURRENCY_HINT.search(meta).group(0) if CURRENCY_HINT.search(meta) else None)
                        unit = (UNIT_HINT.search(meta).group(0) if UNIT_HINT.search(meta) else None)
                        tbls.append(TableBlock(rows=rows, page=pno, title="Table", currency=currency, unit=unit))
        except Exception:
            pass

    return tbls

# -------------------------------------------------------------------
# 原有逻辑保留：chunk（文本与表格）
# -------------------------------------------------------------------

def chunk_paragraph(text: str, size: int = 640, overlap: int = 96) -> List[str]:
    if tok_count(text) <= size:               # [MOD-FAST] tok_len -> tok_count（可近似）
        return [text]
    step = max(1, size - overlap)
    parts = tok_split(text, step)
    res = []
    prev_tail = ""
    for i, p in enumerate(parts):
        cur = (prev_tail + p) if i > 0 else p
        if tok_count(cur) > size:             # [MOD-FAST]
            cur = tok_split(cur, size)[0]
        res.append(cur)
        tail_tokens = tok_split(cur, max(1, tok_count(cur)-overlap))  # [MOD-FAST]
        prev_tail = tail_tokens[-1] if tail_tokens else ""
    return res

def chunk_table_by_rows(rows: List[List[str]], window: int = 4) -> List[List[List[str]]]:
    if not rows:
        return []
    header = rows[0]
    data = rows[1:]
    if not data:
        return [rows]
    chunks = []
    for i in range(0, len(data), window):
        block = [header] + data[i:i+window]
        chunks.append(block)
    return chunks

def build_chunks_for_financial_report(
    pdf_path: str,
    doc_meta: Dict,
    para_size: int = 640,
    para_overlap: int = 96,
    table_window: int = 4
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
                "tokens": tok_count(s),     # [MOD-FAST]
            }
            md["hash"] = md5(f"{doc_id}|p|{pb.page}|{order}|{j}|{s[:80]}")
            out.append({"text": s, "metadata": md})
        order += 1

    tables = extract_tables(pdf_path)
    for t_idx, tb in enumerate(tables):
        t_chunks = chunk_table_by_rows(tb.rows, window=table_window)
        for k, block in enumerate(t_chunks):
            tsv = "\n".join(["\t".join(row) for row in block])
            md = {
                **doc_meta,
                "doc_id": doc_id,
                "type": "table",
                "section_path": f"{tb.title}",
                "page": tb.page,
                "table_id": f"T{t_idx}",
                "table_chunk": k,
                "headers": block[0],
                "currency": tb.currency,
                "unit": tb.unit,
                "tokens": tok_count(tsv),  # [MOD-FAST]
            }
            md["headers_hash"] = md5("|".join(md["headers"]))
            md["hash"] = md5(f"{doc_id}|t|{tb.page}|{t_idx}|{k}|{tsv[:80]}")
            out.append({"text": tsv, "metadata": md})
    return out

# example usage (可保留/可移除)
if __name__ == "__main__":
    from pathlib import Path
    import json
    # set data folder path
    data_folder = Path("./tat_docs")
    pdf_files = sorted(data_folder.glob("*.pdf")) 
    total_chunks = []

    print(f"[INFO] Found {len(pdf_files)} PDF files under {data_folder}")
    for idx, f in enumerate(pdf_files, start=1):
        doc_stem = f.stem
        print(f"[{idx}/{len(pdf_files)}] Processing {doc_stem} ...")
        try:
            chunks = build_chunks_for_financial_report(
                str(f),
                {"doc_id": doc_stem}   # 自动使用文件名作为 doc_id
            )
            total_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Failed on {f}: {e}")
            continue

    print("\n[SUMMARY]")
    print(f"  Total PDFs processed : {len(pdf_files)}")
    print(f"  Total chunks created  : {len(total_chunks)}")

    out_path = Path("chunks_all.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for ck in total_chunks:
            fout.write(json.dumps(ck, ensure_ascii=False) + "\n")
    print(f"  Output written to: {out_path.resolve()}")
