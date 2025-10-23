"""Smoke test for Gemini embedding models via the google-generativeai SDK."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import requests

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None


ENV_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

DEFAULT_TEXTS: tuple[str, ...] = (
    "Gemini embedding quick test.",
    "Second sample sentence to verify vector shape.",
)


def embed_texts(
    model: str,
    task_type: str | None,
    texts: Sequence[str],
    api_key: str | None,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
        if genai is not None:
            payload = {"model": model, "content": text}
            if task_type:
                payload["task_type"] = task_type
            response = genai.embed_content(**payload)
            if isinstance(response, dict):
                vector = response.get("embedding")
                debug_payload: Any = response
            else:
                vector = getattr(response, "embedding", None)
                debug_payload = response.to_dict() if hasattr(response, "to_dict") else {"raw": str(response)}
        else:
            if not api_key:
                raise SystemExit("Provide an API key via --api-key or GEMINI_API_KEY / GOOGLE_API_KEY.")
            model_path = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:embedContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            body: dict[str, Any] = {
                "content": {
                    "parts": [
                        {"text": text},
                    ]
                }
            }
            if task_type:
                body["taskType"] = task_type
            response = requests.post(url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            debug_payload = response.json()
            vector_obj: Any = debug_payload.get("embedding")
            vector = vector_obj.get("values") if isinstance(vector_obj, dict) else vector_obj
        if not isinstance(vector, list):
            raise RuntimeError(f"Unexpected response payload: {json.dumps(debug_payload, indent=2)}")
        embeddings.append(vector)
    return embeddings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the Gemini embedding API and dump vector stats.")
    parser.add_argument(
        "--api-key",
        default=ENV_API_KEY,
        help="API key with access to the Gemini API (falls back to GEMINI_API_KEY / GOOGLE_API_KEY env vars).",
    )
    parser.add_argument(
        "--model",
        default="models/text-embedding-004",
        help="Gemini embedding model name.",
    )
    parser.add_argument(
        "--task-type",
        default="retrieval_document",
        help="Optional task_type hint (e.g. retrieval_query, retrieval_document, semantic_similarity).",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full raw embeddings JSON payload instead of the truncated summary.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=DEFAULT_TEXTS,
        help="Texts to embed. If omitted, a couple of sample snippets are used.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Provide an API key via --api-key or GEMINI_API_KEY / GOOGLE_API_KEY.")

    if genai is not None:
        genai.configure(api_key=args.api_key)

    vectors = embed_texts(args.model, args.task_type, args.texts, args.api_key)

    if args.show_json:
        print(json.dumps({"vectors": vectors}, indent=2))
        return 0

    for idx, (text, vector) in enumerate(zip(args.texts, vectors), start=1):
        head = ", ".join(f"{value:.4f}" for value in vector[:5])
        print(f"[{idx}] len={len(vector)} first5=[{head}] text={text[:60]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
