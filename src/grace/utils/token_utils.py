"""Shared helpers for token accounting and similarity scoring.

These functions replace the previously missing `estimate_tokens`
and `calculate_relevance` helpers that multiple modules import.
They rely only on NumPy (already a project dependency)."""

from __future__ import annotations

from typing import Sequence
import numpy as np

TOKEN_RATIO = 4  # ≈ characters per token for GPT‑style BPE


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: 1 token ≈ 4 characters (fallback)."""
    return max(1, len(text) // TOKEN_RATIO)


def calculate_relevance(query_emb: Sequence[float], doc_emb: Sequence[float]) -> float:
    """Cosine similarity in the closed interval [-1, 1]."""
    q = np.asarray(query_emb, dtype=np.float32)
    d = np.asarray(doc_emb, dtype=np.float32)
    denom = (np.linalg.norm(q) * np.linalg.norm(d)) + 1e-9
    return float(np.dot(q, d) / denom)
