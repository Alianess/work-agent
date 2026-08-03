from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
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
