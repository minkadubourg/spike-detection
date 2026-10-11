import numpy as np
import pytest

from spikedet import match


def test_perfect_match():
    m = match([10, 50, 90], [10, 50, 90], tol=2)
    assert (m.tp, m.fp, m.fn) == (3, 0, 0) and m.f1 == 1


def test_one_detection_cannot_claim_two_reference_events():
    # greedy "any reference within the window" would count both references as found
    m = match([100], [99, 101], tol=3)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)


def test_assignment_maximises_matches_where_greedy_nearest_fails():
    # nearest-first would pair 10<->11 and leave 8 and 13 unmatched
    m = match([10, 13], [8, 11], tol=3)
    assert m.tp == 2


def test_ties_prefer_the_smaller_offset():
    m = match([100], [97, 101], tol=5)
    assert m.offsets.tolist() == [-1]


def test_shift_aligns_a_reference_with_a_different_timing_convention():
    det = np.array([100, 200, 300])
    ref = det + 10                                    # reference marks 10 samples after the trough
    assert match(det, ref, tol=2).tp == 0
    m = match(det, ref, tol=2, shift=10)
    assert m.tp == 3 and np.all(m.offsets == 0)


@pytest.mark.parametrize("d,r,expected", [([], [], (0, 0, 0)), ([5], [], (0, 1, 0)), ([], [5], (0, 0, 1))])
def test_empty_inputs(d, r, expected):
    m = match(d, r, tol=3)
    assert (m.tp, m.fp, m.fn) == expected


def test_matching_is_order_independent():
    rng = np.random.default_rng(0)
    r = np.sort(rng.choice(100000, 300, replace=False))
    d = r + rng.integers(-4, 5, r.size)
    d = np.concatenate([d, rng.choice(100000, 50)])
    a = match(d, r, tol=5)
    perm = rng.permutation(d.size)
    b = match(d[perm], r[::-1], tol=5)
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
    assert a.tp + a.fn == r.size and a.tp + a.fp == d.size
