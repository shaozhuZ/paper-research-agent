"""Retrieval metrics.

A chunk is identified by (filename, chunk_id). `retrieved` is the ranked list
the search returned, best first. `gold` is the set of chunks that contain the
answer. For questions with no answer in the corpus, gold is empty and the
metrics don't apply, so they return None.
"""
from __future__ import annotations

ChunkKey = tuple[str, int]


def recall_at_k(retrieved: list[ChunkKey], gold: set[ChunkKey], k: int) -> float | None:
    """Fraction of gold chunks that appear in the top k results.

    retrieved = [A, B, C, D, E], gold = {C}     -> k=1: 0.0, k=3: 1.0
    retrieved = [A, B, C, D, E], gold = {A, E}  -> k=3: 0.5, k=5: 1.0
    """
    if not gold:
        return None
    top_k = set(retrieved[:k])
    found = top_k & gold        # & 是交集：两边都有的元素
    return len(found) / len(gold)


def reciprocal_rank(retrieved: list[ChunkKey], gold: set[ChunkKey]) -> float | None:
    """1 / (rank of the first gold chunk), ranks start at 1. 0.0 if none was found.

    retrieved = [A, B, C, D, E], gold = {C} -> 1/3
    """
    if not gold:
        return None

    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in gold:
            return 1 / rank
    return 0.0
def mean(values: list[float | None]) -> float | None:
    """Average that ignores None (questions the metric doesn't apply to)."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
