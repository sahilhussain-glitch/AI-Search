"""
faiss_index.py — IVF-PQ FAISS index for AISearch.

Wraps Facebook's FAISS library to provide:
  - fast approximate nearest-neighbour search over 768-dim CodeBERT vectors
  - sub-linear search time via Inverted File (IVF) coarse quantizer
  - memory compression via Product Quantization (PQ)
  - serialisation to disk for persistent indexes
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Install faiss: pip install faiss-cpu  (or faiss-gpu for GPU support)")


class FAISSIndex:
    """
    IVF-PQ index that maps a vector query to the nearest code snippet IDs.

    Parameters
    ----------
    dim          : vector dimensionality (768 for CodeBERT)
    n_lists      : number of Voronoi cells (coarse quantizer clusters)
    n_probe      : cells to visit at query time — trade recall vs speed
    pq_subvecs   : number of sub-vectors for Product Quantization
    """

    def __init__(
        self,
        dim: int = 768,
        n_lists: int = 256,
        n_probe: int = 32,
        pq_subvecs: int = 64,
    ):
        self.dim = dim
        self.n_probe = n_probe
        quantizer = faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIVFPQ(quantizer, dim, n_lists, pq_subvecs, 8)
        self._index.nprobe = n_probe
        self._id_map: List[str] = []   # row_number → snippet_id
        self._trained = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def train(self, vectors: np.ndarray) -> None:
        """Train the coarse quantizer + PQ codebooks on a representative sample."""
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self._index.train(vectors)
        self._trained = True

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add normalised vectors with corresponding snippet IDs."""
        if not self._trained:
            raise RuntimeError("Call train() before add()")
        assert len(vectors) == len(ids)
        vectors = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._id_map.extend(ids)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Return the top_k nearest snippet IDs and their cosine similarity scores.
        """
        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = self._index.search(q, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue   # FAISS returns -1 for empty slots
            # IVF-PQ uses L2 on normalised vectors → cosine similarity = 1 - d/2
            score = float(1.0 - dist / 2.0)
            results.append((self._id_map[idx], score))
        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Persist index and id_map to *directory*."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "id_map.json"), "w") as f:
            json.dump(self._id_map, f)

    @classmethod
    def load(cls, directory: str, n_probe: int = 32) -> "FAISSIndex":
        """Load a previously saved index."""
        obj = cls.__new__(cls)
        obj._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        obj._index.nprobe = n_probe
        obj.n_probe = n_probe
        obj.dim = obj._index.d
        obj._trained = True
        with open(os.path.join(directory, "id_map.json")) as f:
            obj._id_map = json.load(f)
        return obj

    def __len__(self) -> int:
        return self._index.ntotal
