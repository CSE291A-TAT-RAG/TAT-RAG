## RAG 流程（傻瓜版）

1) 把 PDF 切成块（scripts/chunk_revise_.py）
- 输入：`/app/tat_docs_test/*.pdf`，输出：文本块到 `data/chunks_all.jsonl`，表格到 `data/csvs/`.
- 分块：段落最少 80 字符；超过 ~400 token 会切分，重叠 80 字符。
- 技术栈：PyMuPDF 取段落/版面，Tabula 抽表为 CSV。token 统计用 `cl100k_base`。
- 表格：Tabula 输出 CSV，路径写在 JSONL 的 `text`/`table_path`，并记录 `table_title/table_id/page`。章节名归一化：Item X → `item_x`（含 1a/7a/10 等）。

2) 写入向量库（src/ingestion.py）
- 跑：`python main.py ingest data/chunks_all.jsonl`（在容器内）。
- 元数据：补 `company/fiscal_year/source`；内容前加 “Document/Section/Page/Type” 头，方便检索。
- 表格：容器路径 `/app/data/...` 会映射为本地 `data/...` 读取 CSV，读不到也会保留 `table_path`。
- 建索引：BM25 建在 `content`；keyword 索引建在 `metadata.doc_id/company/fiscal_year`。

3) 检索（src/retrieval.py）
- top_k=20 直接作为检索条数；Hybrid 开时用 dense+BM25，否则 dense。
- 路由过滤：风险/MD&A/财报/内控/业务/董事会 等关键词 → section 过滤，命中不足时优先同 doc 补回。
- 分数加权：table 基础 1.5x；company/source/doc_id 1.4x；section 1.1x；table 标题/行列/年度 1.2x；命中财务关键词再额外 boost。
- 表格补全：表格太少时，从同 doc_id 追加若干跨页表再 rerank，提升表格 recall。
- 多公司均衡：query 同时提到多个公司名时，为每个命中公司保留至少几条候选再 rerank，避免单一公司霸榜。
- Rerank：开启时 rerank 后再截断到 top_k。

4) 评测与金标准
- 跑评测（容器内）：`python scripts/evaluate_retrieval.py --golden-path requests/requests.json --top-k 20 --k-values 1 3 5 10 --save-details /app/output/retrieval_details.json`
