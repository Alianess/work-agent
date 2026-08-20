from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Protocol
import os
import threading

import numpy as np


DEFAULT_MLX_MODEL_PATH = Path.home() / "llm_model" / "mlx" / "bge-m3-6bit"
DEFAULT_MLX_MODEL_ID = "bge-m3-mlx-6bit"
DEFAULT_MAX_LENGTH = 512
DEFAULT_INFERENCE_BATCH_SIZE = 8
_MLX_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MLX_INFERENCE_LOCK = threading.RLock()


class RetrievalBackendError(RuntimeError):
    pass


class RetrievalBackend(Protocol):
    enabled: bool
    embedding_model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class MlxRetrievalBackend:
    """Lazy, in-process BGE-M3 embeddings for Apple Silicon.

    Search owns indexing, project/account isolation and RRF. This backend owns
    only model loading and dense-vector inference. Keeping that boundary small
    lets callers retain one stable retrieval API without another local daemon.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_path: str | Path = DEFAULT_MLX_MODEL_PATH,
        embedding_model: str = DEFAULT_MLX_MODEL_ID,
        max_length: int = DEFAULT_MAX_LENGTH,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = Path(model_path).expanduser()
        self.embedding_model = str(embedding_model or DEFAULT_MLX_MODEL_ID)
        self.max_length = max(64, min(int(max_length), 8192))
        self.inference_batch_size = max(1, min(int(inference_batch_size), 32))

    @classmethod
    def from_env(cls) -> "MlxRetrievalBackend":
        return cls(
            enabled=env_flag("WORK_AGENT_HISTORY_RAG_ENABLED", False),
            model_path=os.getenv(
                "WORK_AGENT_HISTORY_MLX_MODEL_PATH",
                str(DEFAULT_MLX_MODEL_PATH),
            ),
            embedding_model=os.getenv(
                "WORK_AGENT_HISTORY_EMBEDDING_MODEL",
                DEFAULT_MLX_MODEL_ID,
            ),
            max_length=int(
                os.getenv(
                    "WORK_AGENT_HISTORY_MLX_MAX_LENGTH",
                    str(DEFAULT_MAX_LENGTH),
                )
            ),
            inference_batch_size=int(
                os.getenv(
                    "WORK_AGENT_HISTORY_MLX_BATCH_SIZE",
                    str(DEFAULT_INFERENCE_BATCH_SIZE),
                )
            ),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            raise RetrievalBackendError("MLX embedding disabled")
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return []
        if any(not text for text in clean_texts):
            raise RetrievalBackendError("MLX embedding input contains empty text")

        with _MLX_INFERENCE_LOCK:
            model, tokenizer, generate, mx = self._ensure_loaded()
            vectors: list[list[float]] = []
            for start in range(0, len(clean_texts), self.inference_batch_size):
                batch = clean_texts[start : start + self.inference_batch_size]
                try:
                    output = generate(
                        model,
                        tokenizer,
                        texts=batch,
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                    )
                    embeddings = output.text_embeds
                    mx.eval(embeddings)
                    array = np.asarray(embeddings, dtype=np.float32)
                except Exception as error:
                    raise RetrievalBackendError(
                        f"MLX embedding inference failed: {type(error).__name__}: {error}"
                    ) from error
                if array.ndim != 2 or array.shape[0] != len(batch):
                    raise RetrievalBackendError(
                        "MLX embedding output shape mismatch: "
                        f"expected_rows={len(batch)} actual={array.shape}"
                    )
                vectors.extend(array.tolist())
            return vectors

    def _ensure_loaded(self) -> tuple[Any, Any, Any, Any]:
        cache_key = str(self.model_path.resolve())
        cached = _MLX_MODEL_CACHE.get(cache_key)
        if cached is not None:
            from mlx_embeddings import generate
            import mlx.core as mx

            return cached[0], cached[1], generate, mx
        if not self.model_path.is_dir():
            raise RetrievalBackendError(
                f"MLX embedding model not found: {self.model_path}"
            )
        try:
            from mlx_embeddings import generate, load
            import mlx.core as mx

            model, tokenizer = load(str(self.model_path))
        except Exception as error:
            raise RetrievalBackendError(
                f"MLX embedding model load failed: {type(error).__name__}: {error}"
            ) from error
        _MLX_MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer, generate, mx


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# 中文分词：FTS5 的 unicode61 不切中文，所以索引期把中文展开成 n-gram 词串
# ---------------------------------------------------------------------------

MAX_QUERY_TERMS = 24

TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_.+-]*|[0-9]+(?:\.[0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
CJK_STOP_TERMS = {
    "之前", "我们", "你们", "他们", "这个", "那个", "一下", "什么", "怎么",
    "聊天", "历史", "提到", "说过", "讨论", "回想", "记得", "内容", "相关", "当时",
}


def cjk_ngrams(value: str, *, include_unigrams: bool) -> list[str]:
    if len(value) == 1:
        return [value]
    terms = [value[index : index + 2] for index in range(len(value) - 1)]
    if include_unigrams:
        terms.extend(value)
    return terms


def index_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in TOKEN_PATTERN.findall(str(text or "")):
        token = raw.lower()
        if CJK_PATTERN.fullmatch(token):
            terms.extend(cjk_ngrams(token, include_unigrams=True))
        elif len(token) >= 2 or token.isdigit():
            terms.append(token)
    return dedupe(terms)


def extract_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    cleaned = str(text or "")
    for stop_term in sorted(CJK_STOP_TERMS, key=len, reverse=True):
        cleaned = cleaned.replace(stop_term, " ")
    for raw in TOKEN_PATTERN.findall(cleaned):
        token = raw.lower()
        if CJK_PATTERN.fullmatch(token):
            grams = cjk_ngrams(token, include_unigrams=len(token) == 1)
            terms.extend(term for term in grams if term not in CJK_STOP_TERMS)
        elif len(token) >= 2 or token.isdigit():
            terms.append(token)
    unique = dedupe(terms)
    if len(unique) <= MAX_QUERY_TERMS:
        return unique
    ranked = sorted(enumerate(unique), key=lambda item: (-len(item[1]), item[0]))[:MAX_QUERY_TERMS]
    keep = {index for index, _ in ranked}
    return [term for index, term in enumerate(unique) if index in keep]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
