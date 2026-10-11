"""Gallery of the benchmark scenarios: a short raw excerpt of each, with true
spikes (ticks) and the detector's accepted (blue) and flagged (grey) events.

    python analyses/plot_scenarios.py      -> figures/scenarios.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spikedet import detect  # noqa: E402
from spikedet import scenarios as S  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"

PANELS = [  # scenario, centre (sample or None = first true spike), window in ms
    (S.clean, None, 60), (S.low_snr, None, 60), (S.drift, None, 400), (S.mixed_amplitudes, None, 120),
    (S.bursts, "burst", 40), (S.overlaps, "collision", 12), (S.artifacts, "artifact", 30),
    (S.line_noise, None, 60), (lambda: S.edge_cases()[2], 0, 8), (lambda: S.edge_cases()[4], None, 60),
    (lambda: S.edge_cases()[5], None, 8), (S.streaming, "straddle", 12),
]


def main():
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(13, 7))
    for ax, (make, centre, win_ms) in zip(axes.flat, PANELS):
        sc = make()
        x, fs = sc.voltage, sc.fs
        if centre is None:
            c = int(sc.truth[min(2, len(sc.truth) - 1)])
        elif centre == "burst":
            d = np.diff(sc.truth)
            c = int(sc.truth[np.argmin(d)])
        elif centre == "collision":
            c = int(sc.extra["colliding"][0])
        elif centre == "artifact":
            c = int(sc.extra["artifacts"][0])
        elif centre == "straddle":
            c = int(sc.extra["straddle"])
        else:
            c = int(centre)
        h = int(win_ms * 1e-3 * fs / 2)
        lo, hi = max(0, c - h), min(len(x), c + h)
        t = (np.arange(lo, hi) - c) / fs * 1e3
        ax.plot(t, x[lo:hi], color="0.35", lw=0.6)
        ev = detect(x, fs)
        ev = ev[(ev["sample"] >= lo) & (ev["sample"] < hi)]
        y0 = np.nanmin(x[lo:hi])
        for _, r in ev.iterrows():
            ax.plot((r["sample"] - c) / fs * 1e3, y0 * 1.12, "^", ms=5,
                    color="#3b6ea8" if r.accepted else "0.65")
        tt = sc.truth[(sc.truth >= lo) & (sc.truth < hi)]
        ax.vlines((tt - c) / fs * 1e3, y0 * 1.02, y0 * 1.08, color="#b23a48", lw=1.2)
        if "chunk" in sc.extra:
            for b in range((lo // sc.extra["chunk"] + 1) * sc.extra["chunk"], hi, sc.extra["chunk"]):
                ax.axvline((b - c) / fs * 1e3, color="#d08c3c", ls="--", lw=0.8)
        ax.set_title(sc.name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("ms", fontsize=7)
    axes[0, 0].set_ylabel("µV", fontsize=8)
    fig.suptitle("Benchmark scenarios — red ticks: ground truth; triangles: detector events "
                 "(blue accepted, grey flagged)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "scenarios.png", dpi=140)


if __name__ == "__main__":
    main()
