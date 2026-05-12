"""
encoder.py — Batched, thread-safe CodeBERT encoder for AISearch.

Encodes natural-language queries and code snippets into 768-dim vectors
using microsoft/codebert-base. A thread pool allows multiple concurrent
encoding requests without blocking the FastAPI event loop.
"""

from __future__ import annotations
import threading
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "microsoft/codebert-base"


class CodeBERTEncoder:
    """
    Thread-safe singleton wrapper around CodeBERT.

    The model is loaded once and shared across threads via a lock.
    For high-throughput serving, run multiple encoder replicas instead.
    """

    _instance: "CodeBERTEncoder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model_lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModel.from_pretrained(_MODEL_NAME).to(self.device)
        self.model.eval()

    @classmethod
    def get(cls) -> "CodeBERTEncoder":
        """Return the singleton encoder (lazy init)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def encode(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """
        Encode a batch of texts → float32 array of shape (N, 768).
        Uses [CLS] token representation as the sentence embedding.
        """
        with self._model_lock:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # [CLS] token is at position 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            return cls_embeddings.cpu().numpy().astype(np.float32)
