"""Compare the detector with a reference detector on a real recording.

The recording (~12 s of single-channel broadband intracortical signal at
19.6 kHz, with a reference detector's output as a boolean column) is not
redistributable, so it is not in this repository. Point the script at a CSV
with columns time_s, broadband_uv, spikes_bool:

    python analyses/reference_comparison.py --data path/to/recording.csv

There is no ground truth here: the reference is another detector. The
comparison measures agreement, and every disagreement is inspected on the
signal rather than counted as an error of either side.

Outputs: results/reference_*.csv|json, figures/reference_*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spikedet import DetectorConfig, detect, match  # noqa: E402
from spikedet.detector import preprocess  # noqa: E402
from spikedet.events import accepted  # noqa: E402
from spikedet.signal import bandpass, infer_fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
C = {"shared": "#3b6ea8", "detector only": "#d08c3c", "reference only": "#b23a48", "artifact": "#888888"}


def snippets(f, idx, half):
    idx = [i for i in idx if half <= i < len(f) - half]
    return np.array([f[i - half:i + half + 1] for i in idx]).reshape(len(idx), 2 * half + 1)


def snr_db(f, events, fs):
    """Median event power within ±1 ms over mean noise power outside ±2.5 ms of any event."""
    e, x = int(round(1e-3 * fs)), int(round(2.5e-3 * fs))
    p_event = np.median([np.mean(f[max(0, i - e):i + e + 1] ** 2) for i in events])
    mask = np.ones(len(f), bool)
    for i in events:
        mask[max(0, i - x):i + x + 1] = False
    return 10 * np.log10(p_event / np.mean(f[mask] ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    t, x = df["time_s"].to_numpy(), df["broadband_uv"].to_numpy()
    ref = np.flatnonzero(df["spikes_bool"].to_numpy())
    fs = infer_fs(t)
    cfg = DetectorConfig()
    _, f, thr, _ = preprocess(x, fs, cfg)
    ev = detect(x, time_s=t, config=cfg)
    acc = accepted(ev)

    # 1) timing convention of the reference: offsets of matched pairs within ±1 ms
    loose = match(acc["sample"], ref, int(round(1e-3 * fs)))
    offsets = -loose.offsets                              # reference - trough, samples
    shift = int(np.median(offsets))

    # 2) agreement after removing that convention, tolerance ±0.5 ms
    m = match(acc["sample"], ref, int(round(0.5e-3 * fs)), shift=shift)
    only_det = np.setdiff1d(np.arange(len(acc)), m.matched_detected)
    only_ref = np.setdiff1d(np.arange(len(ref)), m.matched_reference)

    art = ev.loc[ev.artifact, "sample"].to_numpy()
    near_art = lambda s: bool(len(art)) and np.min(np.abs(art - s)) <= int(0.01 * fs)
    rows = []
    for i in m.matched_detected:
        rows.append(("shared", acc["sample"].iloc[i]))
    for i in only_det:
        rows.append(("detector only", acc["sample"].iloc[i]))
    for j in only_ref:
        # the trough the reference event refers to
        s = ref[j] - shift
        lo = max(0, s - 5)
        rows.append(("reference only", lo + int(np.argmin(f[lo:s + 6]))))
    table = pd.DataFrame(rows, columns=["category", "sample"]).sort_values("sample").reset_index(drop=True)
    table["time_s"] = t[table["sample"]]
    table["trough_uv"] = f[table["sample"]]
    table["trough_over_threshold"] = table["trough_uv"] / thr[table["sample"]]
    table["within_10ms_of_flagged_artifact"] = [near_art(s) for s in table["sample"]]
    table.to_csv(RES / "reference_events.csv", index=False, float_format="%.4f")

    # 3) exploration: filter settings, scored on fixed event sets (no tuning to the reference)
    sweep = []
    fixed = table.loc[table.category != "reference only", "sample"].to_numpy()
    for name, band, order in [("order 2", (300, 3500), 2), ("order 4", (300, 3500), 4), ("order 6", (300, 3500), 6),
                              ("low cut 150", (150, 3500), 4), ("low cut 500", (500, 3500), 4),
                              ("high cut 2000", (300, 2000), 4), ("high cut 5000", (300, 5000), 4)]:
        g = bandpass(x, fs, band, order)
        c = DetectorConfig(band_hz=band, filter_order=order)
        a = accepted(detect(x, fs, config=c))
        mm = match(a["sample"], ref, int(round(0.5e-3 * fs)), shift=shift)
        sweep.append(dict(setting=name, snr_db=snr_db(g, fixed, fs), detected=len(a),
                          shared=mm.tp, detector_only=mm.fp, reference_only=mm.fn))
    pd.DataFrame(sweep).to_csv(RES / "reference_filter_sweep.csv", index=False, float_format="%.2f")

    noise_windows = [float(-v / cfg.k) for v in np.unique(
        __import__("spikedet").signal.threshold_trace(f, fs, cfg.k, window_s=2.0))]
    summary = dict(
        fs_hz=fs, duration_s=float(t[-1] - t[0]), reference_events=int(len(ref)),
        detector_events_total=int(len(ev)), detector_accepted=int(len(acc)),
        flagged_artifact=int(ev.artifact.sum()), threshold_uv=float(thr[0]),
        noise_sigma_uv=float(-thr[0] / cfg.k),
        noise_sigma_per_2s_window_uv=[round(v, 2) for v in noise_windows],
        reference_offset_samples=dict(zip(*[a.tolist() for a in np.unique(offsets, return_counts=True)])),
        reference_offset_ms=shift / fs * 1e3,
        shared=int(m.tp), detector_only=int(m.fp), reference_only=int(m.fn),
    )
    (RES / "reference_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---------------- figures
    half = int(round(1.5e-3 * fs))
    ms = (np.arange(-half, half + 1)) / fs * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, cat in zip(axes, ["shared", "detector only", "reference only"]):
        idx = table.loc[table.category == cat, "sample"].to_numpy()
        S = snippets(f, idx, half)
        for s in S:
            ax.plot(ms, s, color=C[cat], alpha=0.35, lw=0.8)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axvline(shift / fs * 1e3, color="0.3", lw=0.8, ls="--")
        ax.axhline(thr[0], color="0.6", lw=0.8, ls=":")
        ax.set_title(f"{cat} (n = {len(idx)})", fontsize=10)
        ax.set_xlabel("time from trough (ms)")
    axes[0].set_ylabel("band-passed signal (µV)")
    axes[0].text(shift / fs * 1e3 + 0.05, axes[0].get_ylim()[0] * 0.9, "reference\nmark", fontsize=7, color="0.3")
    fig.tight_layout()
    fig.savefig(FIG / "reference_waveforms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3))
    vals, counts = np.unique(offsets, return_counts=True)
    ax.bar(vals / fs * 1e3, counts, width=0.8 / fs * 1e3, color=C["shared"])
    ax.set_xlabel("reference mark − detected trough (ms)")
    ax.set_ylabel("events")
    ax.set_title("Reference timing convention", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "reference_offsets.png", dpi=150)
    plt.close(fig)

    # artifact train
    sel = (t >= t[0] + 0.7) & (t <= t[0] + 2.5)
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    axes[0].plot(t[sel], x[sel], color="0.4", lw=0.5)
    axes[0].set_ylabel("broadband (µV)")
    axes[1].plot(t[sel], f[sel], color="0.4", lw=0.5)
    axes[1].axhline(thr[0], color="0.6", ls=":", lw=0.8)
    for _, r in ev[(ev.time_s >= t[sel][0]) & (ev.time_s <= t[sel][-1])].iterrows():
        axes[1].plot(r.time_s, r.amplitude_uv, "v", ms=4,
                     color=C["artifact"] if r.artifact else C["shared"])
    axes[1].set_ylabel("band-passed (µV)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("Periodic artifact train: events flagged as artifacts (grey) and accepted (blue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "reference_artifact_train.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print(table.groupby(["category", "within_10ms_of_flagged_artifact"]).size())
    print(table[table.category != "shared"].to_string(index=False))
    print(pd.DataFrame(sweep).to_string(index=False))


if __name__ == "__main__":
    main()
