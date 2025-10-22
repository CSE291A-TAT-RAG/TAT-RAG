"""Hardcoded smoke test for amazon.titan-embed-text-v2:0 via Bedrock."""

import json
from typing import List

import boto3

# ---------------------------------------------------------------------------
# Configuration: fill in the values you want to test.
# Leave credentials empty to use the default AWS credential chain (env vars,
# shared config, instance/role credentials, etc.).
# ---------------------------------------------------------------------------
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "ASIAXNAJRSS4EEZTLLMH"
AWS_SECRET_ACCESS_KEY = "ZOZ3tg6ixMjm3iQYphtQ6+KfWDOe7iQX4BCN6VEp"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEHMaCXVzLXdlc3QtMiJHMEUCIQCHSt3uFOHVhxc74epKpmem0+8CGKZcd3xyN6hmxpK/LAIgZ0QVDz/2b3m7nhmVpRB5yaxGEiH9fatKK00sKaSyzkkqlAIILBACGgw1MDg5NzM2NTExMjgiDKCvMWLpdyyvP9qQxCrxAYH+quYshOGKUOrg+QOrbzaMcjhzT1ChcF0NXuUiWc0cmGCl9+Npoc/F3HNPmkwZNTCQdsgPXUcWexHNv6xY1O9w0Q6EEj5/2+pteyc+vsrV6j3jwXydH6SXwnUud3V1K2jeHldsSPMzPtaSIjg1GSYj3FMSdM+kjNrVoCg/Sxflo6XzL79j8lkilXouRrdVevF9nWUwgpJXxMRm5lWtaHA+V+MzIMD5MU1BMtmTVXE0i+L7gRZkx9JKbtrkhmFVFeUjVne5FJTX3PLTvNnmWsD7yPr22RgXeMxGFCuwFrRvF686KaUNT1vcTUFiKtCyzQMwiOfixwY6nQEsR5IF9PsqXo34miS+yrNF/UWvXkDFPfZX8fpgXY4kBR0KR4+a6Yy1xL4gEwruepv9r9DmTRnADLk5WvxSSBzUE9y2SD1g7fVJc8ZEVSLdZIyWkrXmrjI9k2hSdMfgIvK/WPTd0H7Fo04Y3czHNoNxSdAQHEDEoR+Uy/XDfhu/wGswlD2aGtoqGKTf/NiefH4o46UP3Lo/SkxGld+W"  # optional
AWS_PROFILE = ""

TITAN_MODEL_ID = "amazon.titan-embed-text-v1"
TITAN_DIMENSIONS: int | None = None  # e.g. 256, 1024 or None for default

TEXT_SNIPPETS: List[str] = [
    "Amazon Bedrock Titan embedding quick test.",
    "Second sample sentence to verify vector shape.",
]


def embed_texts(
    client,
    texts: List[str],
    model_id: str,
    dimensions: int | None,
) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for text in texts:
        payload = {"inputText": text}
        if dimensions:
            payload["dimensions"] = dimensions
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(response["body"].read())
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected response payload: {data}")
        embeddings.append(embedding)
    return embeddings


def main() -> int:
    if not TEXT_SNIPPETS:
        raise SystemExit("TEXT_SNIPPETS is empty; add some sample inputs.")

    session_kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        session_kwargs["profile_name"] = AWS_PROFILE
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs.update(
            {
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            }
        )
        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

    session = boto3.Session(**session_kwargs)
    client = session.client("bedrock-runtime")

    vectors = embed_texts(client, TEXT_SNIPPETS, TITAN_MODEL_ID, TITAN_DIMENSIONS)

    for idx, (text, vec) in enumerate(zip(TEXT_SNIPPETS, vectors), start=1):
        head = ", ".join(f"{value:.4f}" for value in vec[:5])
        print(f"[{idx}] len={len(vec)} first5=[{head}] text={text[:60]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
