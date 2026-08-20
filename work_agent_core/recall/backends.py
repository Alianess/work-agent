"""向量与精排后端：硅基流动云端，本地服务作为显式覆盖。

密钥只从环境变量读，源码里没有默认值。参考实现 V5_dev 把 key 写死在源码里当默认，
这个仓库要公开，那条路不能走。

两个后端都可缺席：拿不到 embedding 就退回纯词法召回，拿不到 rerank 就用 RRF 的
名次。**降级要安静且可见**——检索结果里说明这次是不是降级过，而不是悄悄变差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import json
import os
import urllib.error
import urllib.request


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "Pro/BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# reranker 实测约 1000 字符内安全，截到 900 留余量。超限会让整批回退。
RERANK_MAX_CHARS = 900
EMBED_BATCH_SIZE = 16
REQUEST_TIMEOUT_SECONDS = 60


class RecallBackendError(RuntimeError):
    """后端不可用。调用方据此降级，不是让一次检索失败。"""


def _post(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    # 空 ProxyHandler：绕开环境代理变量，和 llm.py 的做法一致。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RecallBackendError(f"HTTP {error.code}: {detail}") from error
    except Exception as error:
        raise RecallBackendError(str(error)) from error


@dataclass
class SiliconFlowEmbedding:
    api_key: str = ""
    base_url: str = SILICONFLOW_BASE_URL
    model: str = DEFAULT_EMBEDDING_MODEL
    timeout: int = REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "SiliconFlowEmbedding":
        return cls(
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url=os.getenv("RECALL_EMBEDDING_BASE_URL", SILICONFLOW_BASE_URL),
            model=os.getenv("RECALL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.available:
            raise RecallBackendError("未配置 SILICONFLOW_API_KEY，向量召回不可用")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [str(text or " ") for text in texts[start : start + EMBED_BATCH_SIZE]]
            payload = _post(
                f"{self.base_url.rstrip('/')}/embeddings",
                {"model": self.model, "input": batch},
                self.api_key,
                self.timeout,
            )
            data = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
            if len(data) != len(batch):
                raise RecallBackendError(
                    f"embedding 返回条数不符：请求 {len(batch)}，返回 {len(data)}"
                )
            vectors.extend([float(value) for value in item["embedding"]] for item in data)
        return vectors


@dataclass
class SiliconFlowRerank:
    api_key: str = ""
    base_url: str = SILICONFLOW_BASE_URL
    model: str = DEFAULT_RERANK_MODEL
    timeout: int = REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "SiliconFlowRerank":
        return cls(
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url=os.getenv("RECALL_RERANK_BASE_URL", SILICONFLOW_BASE_URL),
            model=os.getenv("RECALL_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def rank(self, query: str, documents: Sequence[str], *, top_n: int = 8) -> list[tuple[int, float]]:
        """返回 [(原始下标, 分数)]，已按分数降序。"""

        if not documents:
            return []
        if not self.available:
            raise RecallBackendError("未配置 SILICONFLOW_API_KEY，精排不可用")
        payload = _post(
            f"{self.base_url.rstrip('/')}/rerank",
            {
                "model": self.model,
                "query": query[:RERANK_MAX_CHARS],
                "documents": [str(item or " ")[:RERANK_MAX_CHARS] for item in documents],
                "top_n": min(int(top_n), len(documents)),
            },
            self.api_key,
            self.timeout,
        )
        results = payload.get("results") or []
        ranked = [
            (int(item.get("index", 0)), float(item.get("relevance_score", 0.0)))
            for item in results
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def normalize(vector: Sequence[float]) -> list[float]:
    total = sum(value * value for value in vector) ** 0.5
    if total <= 0:
        return [0.0] * len(vector)
    return [value / total for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
