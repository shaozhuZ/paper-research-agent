import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import mean, recall_at_k, reciprocal_rank  # noqa: E402

A, B, C, D, E = [("paper.pdf", i) for i in range(5)]
RANKED = [A, B, C, D, E]


@pytest.mark.parametrize("gold,k,expected", [
    ({C}, 1, 0.0),
    ({C}, 3, 1.0),
    ({C}, 5, 1.0),
    ({A, E}, 3, 0.5),
    ({A, E}, 5, 1.0),
    ({("other.pdf", 0)}, 5, 0.0),
])
def test_recall_at_k(gold, k, expected):
    assert recall_at_k(RANKED, gold, k) == pytest.approx(expected)


def test_recall_k_larger_than_results():
    assert recall_at_k([A, B], {B}, 10) == 1.0


def test_recall_counts_each_gold_chunk_once():
    # the same chunk showing up twice must not count as finding two gold chunks
    assert recall_at_k([A, A, C], {A, B}, 3) == 0.5


def test_same_chunk_id_in_another_paper_is_not_a_hit():
    assert recall_at_k([("other.pdf", 2)], {C}, 1) == 0.0


@pytest.mark.parametrize("gold,expected", [
    ({A}, 1.0),
    ({C}, 1 / 3),
    ({B, D}, 0.5),
    ({("other.pdf", 0)}, 0.0),
])
def test_reciprocal_rank(gold, expected):
    assert reciprocal_rank(RANKED, gold) == pytest.approx(expected)


def test_empty_gold_is_not_applicable():
    assert recall_at_k(RANKED, set(), 5) is None
    assert reciprocal_rank(RANKED, set()) is None


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None
