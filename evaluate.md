# Retrieval Evaluation Metrics (TAT-RAG)

This document explains the retrieval metrics emitted by `scripts/evaluate_retrieval.py` after the coverage-focused changes.

## Matching rule
- Normalize text: lowercase and collapse whitespace.
- A retrieved chunk covers a gold evidence span if either:
  - Exact/substring match after normalization, or
  - `SequenceMatcher` similarity >= `--fuzzy-threshold` (default `0.7`).
- Coverage is computed per query using only the top-k retrieved chunks for each metric's k.

## Metrics
- Precision@5: Relevant chunks in top 5 / 5.
- EvidenceRecall@K (K=3,10): Unique gold evidences covered by top-K chunks / total gold evidences (micro).
- PerQueryCoverage@K (K=3,10): For each query, (covered gold evidences in top-K) / (total gold evidences); then averaged across queries (macro).
- FullCoverageRate@K (K=3,10): Share of queries where all gold evidences are covered by at least one chunk in the top-K.
- MAP: Mean Average Precision across queries (uses all retrieved ranks).
- MRR: Mean Reciprocal Rank of the first relevant chunk.
- HitRate@10: Fraction of queries with at least one relevant chunk in the top 10.

## CLI tips
Example run (defaults to fuzzy threshold 0.7):
```bash
docker-compose exec rag-app python scripts/evaluate_retrieval.py \
  --golden-path requests/requests.json \
  --top-k 10 --k-values 1 3 5 10 \
  --save-details //app/output/retrieval_details.json
```

Adjust fuzzy matching if needed:
```bash
--fuzzy-threshold 0.8  # stricter overlap requirement
```
