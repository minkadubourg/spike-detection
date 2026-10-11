"""Scoring detections against reference events.

One-to-one matching: each reference event is matched to at most one detection
and vice versa, choosing the pairing that maximises the number of matches and,
among those, minimises the total time offset (an assignment problem). Greedy
"nearest within a window" matching lets one detection claim two reference
events, which inflates recall on bursts and overlaps.

The same function scores against ground truth (synthetic data) and against
another detector (real data, no ground truth); in the latter case the numbers
measure agreement, not accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    matched_detected: np.ndarray      # indices into `detected`
    matched_reference: np.ndarray     # indices into `reference`
    offsets: np.ndarray               # detected - reference, in samples, for matched pairs

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else float("nan")

    def as_dict(self):
        return dict(tp=self.tp, fp=self.fp, fn=self.fn, precision=self.precision,
                    recall=self.recall, f1=self.f1,
                    median_offset=float(np.median(self.offsets)) if len(self.offsets) else float("nan"))


def match(detected, reference, tol: int, shift: int = 0) -> Score:
    """Match detected to reference sample indices within ±tol samples.

    `shift` is added to the detected indices before matching: it accounts for
    a known convention difference (e.g. a reference detector that marks a fixed
    number of samples after the trough), without changing the detector itself.
    """
    d = np.asarray(detected, np.int64) + shift
    r = np.asarray(reference, np.int64)
    if len(d) == 0 or len(r) == 0:
        return Score(0, len(d), len(r), np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    # candidate pairs within tolerance (sorted arrays -> sweep)
    order_d, order_r = np.argsort(d), np.argsort(r)
    ds, rs = d[order_d], r[order_r]
    pi, pj = [], []
    for i, x in enumerate(ds):
        lo = np.searchsorted(rs, x - tol, side="left")
        hi = np.searchsorted(rs, x + tol, side="right")
        pi.extend([i] * (hi - lo))
        pj.extend(range(lo, hi))
    if not pi:
        return Score(0, len(d), len(r), np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    pi, pj = np.array(pi), np.array(pj)
    # solve each connected cluster of candidate pairs separately (small problems)
    nd = len(ds)
    graph = coo_matrix((np.ones(len(pi)), (pi, nd + pj)), shape=(nd + len(rs),) * 2)
    _, label = connected_components(graph, directed=False)
    md, mr = [], []
    for lab in np.unique(label[pi]):
        sel = label[pi] == lab
        ui, uj = np.unique(pi[sel]), np.unique(pj[sel])
        cost = np.full((len(ui), len(uj)), 1e6)
        cost[np.searchsorted(ui, pi[sel]), np.searchsorted(uj, pj[sel])] = np.abs(ds[pi[sel]] - rs[pj[sel]])
        a, b = linear_sum_assignment(cost)
        ok = cost[a, b] < 1e6
        md.extend(ui[a[ok]])
        mr.extend(uj[b[ok]])
    md, mr = order_d[np.array(md, int)], order_r[np.array(mr, int)]
    tp = len(md)
    return Score(tp, len(d) - tp, len(r) - tp, md, mr, (d[md] - r[mr]).astype(float))
