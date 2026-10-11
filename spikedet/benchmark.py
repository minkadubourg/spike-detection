"""Failure-mode benchmark: run a detector configuration on every scenario and
score accepted events against ground truth (one-to-one, ±1 ms).

    python -m spikedet.benchmark            # prints the table, writes results/benchmark.csv
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import scenarios as S
from .config import DetectorConfig
from .detector import detect, detect_multichannel
from .evaluate import match
from .events import accepted
from .streaming import detect_streaming

PROFILES = {
    "offline (zero-phase)": dict(config=DetectorConfig(), mode="offline"),
    "streaming (causal)": dict(config=DetectorConfig(causal=True, refractory_ms=1.5, noise_calibration_s=0.25),
                               mode="streaming"),
}


def run_scenario(sc: S.Scenario, config: DetectorConfig, mode: str = "offline", tol_ms: float = 1.0) -> dict:
    tol = max(1, int(round(tol_ms * 1e-3 * sc.fs)))
    if np.ndim(sc.voltage) == 2:
        ev = detect_multichannel(sc.voltage, sc.fs, config)
        rows = [match(accepted(ev[ev.channel == c])["sample"], t, tol) for c, t in enumerate(sc.truth)]
        tp, fp, fn = (sum(getattr(r, k) for r in rows) for k in ("tp", "fp", "fn"))
        offsets = np.concatenate([r.offsets for r in rows])
        n_events, n_truth = len(ev), sum(len(t) for t in sc.truth)
    else:
        if mode == "streaming":
            chunk = sc.extra.get("chunk", 1000)
            ev = detect_streaming(sc.voltage, sc.fs, chunk, config)
        else:
            ev = detect(sc.voltage, sc.fs, config=config)
        r = match(accepted(ev)["sample"], sc.truth, tol)
        tp, fp, fn, offsets = r.tp, r.fp, r.fn, r.offsets
        n_events, n_truth = len(ev), len(sc.truth)
    row = dict(scenario=sc.name, description=sc.description, truth=n_truth, detected_events=n_events,
               tp=tp, fp=fp, fn=fn,
               precision=tp / (tp + fp) if tp + fp else np.nan,
               recall=tp / (tp + fn) if tp + fn else np.nan,
               median_offset_samples=float(np.median(offsets)) if len(offsets) else np.nan)
    # flag-level checks where the scenario has the information
    if "artifacts" in sc.extra:
        flagged = ev.loc[ev.artifact, "sample"].to_numpy()
        w = int(0.005 * sc.fs)
        row["artifacts_flagged"] = int(sum(np.any(np.abs(flagged - a) <= w) for a in sc.extra["artifacts"]))
        row["artifacts_total"] = len(sc.extra["artifacts"])
    if "shared_artifact" in sc.extra:
        common = ev.loc[ev.common_artifact, "sample"].to_numpy()
        row["artifacts_flagged"] = int(np.any(np.abs(common - sc.extra["shared_artifact"]) <= int(0.005 * sc.fs)))
        row["artifacts_total"] = 1
    if sc.name.startswith("12_edge"):
        row["edge_flagged"] = int(ev.edge.sum())
    if sc.name == "12_clipped":
        row["clipped_flagged"] = int(accepted(ev).clipped.sum())
    if "straddle" in sc.extra:
        acc = accepted(ev)["sample"].to_numpy()
        row["straddling_spike_found"] = bool(np.any(np.abs(acc - sc.extra["straddle"]) <= tol))
    return row


def all_scenarios():
    return [*S.all_single_channel()[:8], S.multichannel(), *S.all_single_channel()[8:]]


def run(profiles=PROFILES) -> pd.DataFrame:
    rows = []
    for name, p in profiles.items():
        for sc in all_scenarios():
            if p["mode"] == "streaming" and np.ndim(sc.voltage) == 2:
                continue
            rows.append({"profile": name, **run_scenario(sc, p["config"], p["mode"])})
    return pd.DataFrame(rows)


def main():
    df = run()
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False, float_format="%.3f")
    cols = ["profile", "scenario", "truth", "tp", "fp", "fn", "precision", "recall"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
