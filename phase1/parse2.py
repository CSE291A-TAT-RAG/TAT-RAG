import json
import re
import hashlib
import pathlib
from collections import Counter
from typing import List, Dict, Any, Tuple
import pymupdf4llm
import fitz  # PyMuPDF
import tabula
import pandas as pd


CONFIG = {
    "min_chunk_length": 50,
    "max_chunk_tokens": 256,
    "chunk_overlap": 64, # usually be 5-10% of the max_chunk_tokens
    
    "input_dir": "../tat_docs_test/",
    "output_file": "chunks_all.jsonl",
    "exclude_file": "./not_included.txt",
    "csv_output_dir": "./csvs/",
    
    "TABLES_ENABLED": True,
    
    "detect_header_lines": 3,
    "detect_footer_lines": 3,
    "noise_threshold_ratio": 0.15
}

IMPORTANT_SECTIONS = [
    r"^\s*Item\s+\d+[A-Z]?\b.*",
    r"^\s*Management[’']?s\s+Discussion\b.*",
    r"^\s*Consolidated\s+(Statements?|Statement)\b.*",
    r"^\s*Notes?\s+to\s+Consolidated\s+Financial\s+Statements\b.*",
    r"^\s*Report\s+of\s+Independent\b.*",
    r"^\s*Exhibit\s+Index\b.*",
    r"^\s*Signatures\b.*",
]

GENERIC_NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                           
    r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",      
    r"^\s*Table\s+of\s+Contents\s*$",         
    r"^\s*Form\s+10-K\s*$",                   
    r"^\s*\\s*",                    
    r"^\s*_{3,}\s*$"                          
]

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def remove_markdown_format(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'!\[(.*?)\]\([^\)]+\)', r'', text)
    text = text.replace(r'\_', '_').replace(r'\*', '*')
    
    return text.strip()


class AutoHeaderFooterDetector:
    def __init__(self, md_pages: List[Dict], threshold_ratio: float = 0.15):
        self.noise_signatures = set()
        self._learn_patterns(md_pages, threshold_ratio)

    def _normalize(self, text: str) -> str:
        text = re.sub(r'\d+', '<NUM>', text)
        return text.strip().lower()

    def _learn_patterns(self, md_pages: List[Dict], threshold_ratio: float):
        total_pages = len(md_pages)
        if total_pages == 0: return
        line_counter = Counter()
        for page in md_pages:
            text = page['text']
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines: continue
            head = lines[:CONFIG["detect_header_lines"]]
            tail = lines[-CONFIG["detect_footer_lines"]:] if len(lines) > CONFIG["detect_header_lines"] else []
            for line in head + tail:
                line_counter[self._normalize(line)] += 1
        
        threshold_count = max(2, total_pages * threshold_ratio)
        for sig, count in line_counter.items():
            if count >= threshold_count:
                self.noise_signatures.add(sig)

    def is_noise(self, line: str) -> bool:
        if not line.strip(): return False
        return self._normalize(line) in self.noise_signatures

def clean_markdown_content(text: str, auto_detector: AutoHeaderFooterDetector) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()
        is_noise = False
        for pattern in GENERIC_NOISE_PATTERNS:
            if re.match(pattern, line_str, re.IGNORECASE):
                is_noise = True
                break
        if not is_noise and auto_detector.is_noise(line_str):
            is_noise = True
        if not is_noise:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def get_overlap_buffer(buffer_lines: List[Tuple[str, int]], overlap_tokens: int) -> List[Tuple[str, int]]:
    if not buffer_lines: return []
    overlap_buffer = []
    current_tokens = 0
    for line_text, page_num in reversed(buffer_lines):
        line_tokens = estimate_tokens(line_text)
        overlap_buffer.insert(0, (line_text, page_num))
        current_tokens += line_tokens
        if current_tokens >= overlap_tokens:
            break
    return overlap_buffer


def process_text_chunks(pdf_path: pathlib.Path, doc_id: str, start_order: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    chunks = []
    try:
        md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, write_images=False)
    except Exception as e:
        print(f"[Text Error] {pdf_path.name}: {e}")
        return [], start_order

    detector = AutoHeaderFooterDetector(md_pages, threshold_ratio=CONFIG["noise_threshold_ratio"])

    current_section_meta = {
        "doc_id": doc_id, 
        "section_path": "General"
    }
    
    buffer_lines: List[Tuple[str, int]] = []
    buffer_token_count = 0
    global_order = start_order

    for page_data in md_pages:
        page_num = page_data['metadata']['page']
        raw_text = page_data['text']
        
        clean_text = clean_markdown_content(raw_text, detector)
        if not clean_text: continue

        lines = clean_text.split('\n')
        
        for line in lines:
            header_match = re.match(r'^(#{1,3})\s+(.*)', line)
            
            if header_match:
                if buffer_lines:
                    chunk_text = "\n".join([t[0] for t in buffer_lines]).strip()
                    chunk_pages = sorted(list(set(t[1] for t in buffer_lines)))
                    
                    if len(chunk_text) > CONFIG["min_chunk_length"]:
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **current_section_meta,
                                "pages": chunk_pages,
                                "page": chunk_pages[0],
                                "tokens": estimate_tokens(chunk_text),
                                "order": global_order,
                                "type": "text",
                                "hash": md5(f"{doc_id}|text|{global_order}|{chunk_text[:50]}")
                            }
                        })
                        global_order += 1
                
                buffer_lines = []
                buffer_token_count = 0
                
                raw_header = header_match.group(2).strip()
                
                clean_header_line = remove_markdown_format(raw_header)
                
                buffer_lines.append((clean_header_line, page_num))
                buffer_token_count += estimate_tokens(clean_header_line)
                
                for pattern in IMPORTANT_SECTIONS:
                    if re.search(pattern, raw_header, re.IGNORECASE):
                        current_section_meta["section_path"] = clean_header_line
                        break
            
            else:
                if not line.strip().startswith('|'):
                    clean_line = remove_markdown_format(line)
                else:
                    clean_line = remove_markdown_format(line) 
                
                if not clean_line: continue

                line_tokens = estimate_tokens(clean_line)
                buffer_lines.append((clean_line, page_num))
                buffer_token_count += line_tokens
                
                if buffer_token_count > CONFIG["max_chunk_tokens"] and not line.strip().startswith("|"):
                    chunk_text = "\n".join([t[0] for t in buffer_lines]).strip()
                    chunk_pages = sorted(list(set(t[1] for t in buffer_lines)))
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **current_section_meta,
                            "pages": chunk_pages,
                            "page": chunk_pages[0],
                            "tokens": estimate_tokens(chunk_text),
                            "order": global_order,
                            "type": "text",
                            "hash": md5(f"{doc_id}|text|{global_order}|{chunk_text[:50]}")
                        }
                    })
                    global_order += 1
                    
                    overlap_buffer = get_overlap_buffer(buffer_lines, CONFIG["chunk_overlap"])
                    buffer_lines = overlap_buffer
                    buffer_token_count = sum(estimate_tokens(t[0]) for t in buffer_lines)

    if buffer_lines:
        chunk_text = "\n".join([t[0] for t in buffer_lines]).strip()
        chunk_pages = sorted(list(set(t[1] for t in buffer_lines)))
        if len(chunk_text) > CONFIG["min_chunk_length"]:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **current_section_meta,
                    "pages": chunk_pages,
                    "page": chunk_pages[0],
                    "tokens": estimate_tokens(chunk_text),
                    "order": global_order,
                    "type": "text",
                    "hash": md5(f"{doc_id}|text|{global_order}|{chunk_text[:50]}")
                }
            })
            global_order += 1

    return chunks, global_order

def process_table_chunks(pdf_path: pathlib.Path, doc_id: str, start_order: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    if not CONFIG["TABLES_ENABLED"]:
        return [], start_order

    table_chunks = []
    global_order = start_order
    pathlib.Path(CONFIG["csv_output_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"  -> Extracting tables with Tabula...")
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()

        for pno in range(total_pages):
            page_dfs = tabula.read_pdf(str(pdf_path), pages=pno+1, multiple_tables=True, stream=True, silent=True)
            if not page_dfs: continue
            
            for t_idx, df in enumerate(page_dfs):
                if len(df) < 2: continue

                fname = f"{doc_id}_page_{pno}_table_{t_idx}.csv"
                save_path = pathlib.Path(CONFIG["csv_output_dir"]) / fname
                df.to_csv(save_path, index=False)
                
                meta = {
                    "doc_id": doc_id,
                    "type": "table",
                    "page": pno + 1,
                    "pages": [pno + 1],
                    "order": global_order,
                    "table_id": f"T{t_idx}",
                    "section_path": "None",
                    "tokens": 0,
                    "hash": md5(f"{doc_id}|table|{pno}|{t_idx}")
                }
                
                table_chunks.append({
                    "text": str(save_path),
                    "metadata": meta
                })
                global_order += 1

    except Exception as e:
        print(f"  [Table Error] Failed to extract tables: {e}")
        return [], global_order

    return table_chunks, global_order


def main():
    data_folder = pathlib.Path(CONFIG["input_dir"])
    pdf_files = sorted(data_folder.glob("*.pdf"))
    
    exclude_files = set()
    if pathlib.Path(CONFIG["exclude_file"]).exists():
        with open(CONFIG["exclude_file"], 'r') as f:
            exclude_files = {line.strip() for line in f if line.strip()}
    
    filtered_pdf_files = [f for f in pdf_files if f.name not in exclude_files]
    
    print(f"Found {len(filtered_pdf_files)} PDFs. Mode: Plain Text Integrated.")
    print(f"Config: MaxTokens={CONFIG['max_chunk_tokens']}, Overlap={CONFIG['chunk_overlap']}")
    
    total_count = 0
    
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f_out:
        for idx, pdf_file in enumerate(filtered_pdf_files):
            print(f"[{idx+1}/{len(filtered_pdf_files)}] {pdf_file.name}...")
            doc_id = pdf_file.stem
            
            text_chunks, next_order = process_text_chunks(pdf_file, doc_id, start_order=0)
            print(f"  -> Text Chunks: {len(text_chunks)}")
            
            table_chunks, final_order = process_table_chunks(pdf_file, doc_id, start_order=next_order)
            print(f"  -> Table Chunks: {len(table_chunks)}")
            
            all_chunks = text_chunks + table_chunks
            all_chunks.sort(key=lambda x: (x['metadata'].get('page', 0), x['metadata']['order']))

            for chunk in all_chunks:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
            total_count += len(all_chunks)

    print(f"\nDone! Total chunks: {total_count}")
    print(f"Output saved to: {CONFIG['output_file']}")

if __name__ == "__main__":
    main()