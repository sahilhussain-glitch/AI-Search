"""
main.py — FastAPI application for AISearch.
"""

from __future__ import annotations
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from embeddings.encoder import CodeBERTEncoder
from indexing.faiss_index import FAISSIndex

INDEX_DIR = "index/"

app = FastAPI(title="AISearch", version="1.0.0", description="Semantic Code Search Engine")

# Load index once at startup
_index: FAISSIndex | None = None


@app.on_event("startup")
async def startup():
    global _index
    try:
        _index = FAISSIndex.load(INDEX_DIR)
        print(f"Loaded index with {len(_index):,} snippets")
    except FileNotFoundError:
        print("No index found — run scripts/index_corpus.py first")


# ── Response models ────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    id: str
    score: float
    language: str
    snippet: str
    repo: str
    file: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Natural-language search query"),
    lang: Optional[str] = Query(None, description="Filter by language"),
    top_k: int = Query(5, ge=1, le=50),
):
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    t0 = time.perf_counter()
    encoder = CodeBERTEncoder.get()
    query_vec = encoder.encode([q])[0]
    raw_results = _index.search(query_vec, top_k=top_k * 3)  # over-fetch for lang filter

    # Filter by language if requested
    results: List[SearchResult] = []
    # (In a real system, metadata is stored in a side-table keyed by snippet ID)
    for snippet_id, score in raw_results:
        meta = _lookup_metadata(snippet_id)
        if lang and meta["language"] != lang:
            continue
        results.append(SearchResult(
            id=snippet_id,
            score=round(score, 4),
            language=meta["language"],
            snippet=meta["snippet"],
            repo=meta["repo"],
            file=meta["file"],
        ))
        if len(results) >= top_k:
            break

    latency_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=q, results=results, latency_ms=round(latency_ms, 2))


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": len(_index) if _index else 0}


def _lookup_metadata(snippet_id: str) -> dict:
    """Stub — replace with a real metadata store (SQLite / Redis)."""
    return {
        "language": "python",
        "snippet": f"# snippet {snippet_id}",
        "repo": "example/repo",
        "file": "main.py",
    }
