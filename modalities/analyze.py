"""Analyses of six other electrophysiology data types, each scored against
the ground truth of modalities/generate.py.

    python modalities/generate.py && python modalities/analyze.py

Writes results/modalities.json and figures/modality_*.png. Every detection
step is data-driven; ground-truth columns are read only to score.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage, signal  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "modalities" / "data", ROOT / "figures", ROOT / "results"


def mad(x):
    x = np.asarray(x, float)
    return np.median(np.abs(x - np.median(x))) / 0.6745


def scores(pred, true):
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return dict(precision=round(float(p), 3), recall=round(float(r), 3),
                f1=round(float(2 * p * r / (p + r)), 3) if p + r else np.nan)


# ------------------------------------------------------------------ 02 LFP
def lfp():
    df = pd.read_csv(DATA / "02_lfp.csv")
    fs = 1 / np.median(np.diff(df.time_s))
    x = df.lfp_uv.to_numpy()

    # line noise: notch the fundamental and the harmonic seen in the PSD
    y = x.copy()
    for f0 in (60, 120):
        b, a = signal.iirnotch(f0, 30, fs=fs)
        y = signal.filtfilt(b, a, y)
    f, p_raw = signal.welch(x, fs, nperseg=4096)
    _, p_cln = signal.welch(y, fs, nperseg=4096)
    line_drop_db = {f"{f0} Hz": round(float(10 * np.log10(p_raw[np.argmin(abs(f - f0))] / p_cln[np.argmin(abs(f - f0))])), 1)
                    for f0 in (60, 120)}

    # artifacts: data-driven (local variance far above the robust level), not the label
    b, a = signal.butter(4, 150, btype="high", fs=fs)
    hf = signal.filtfilt(b, a, x)
    env = ndimage.uniform_filter1d(hf ** 2, int(0.05 * fs)) ** 0.5
    art = env > np.median(env) + 8 * mad(env)
    art = ndimage.binary_dilation(art, iterations=int(0.1 * fs))
    art_score = scores(art, df.is_artifact)
    idx = np.arange(len(y))
    y_clean = y.copy()
    y_clean[art] = np.interp(idx[art], idx[~art], y[~art])

    # gamma bursts: band-pass, Hilbert envelope, smoothing, hysteresis on robust levels
    b, a = signal.butter(4, [30, 50], btype="band", fs=fs)
    gamma = signal.filtfilt(b, a, y_clean)
    genv = np.abs(signal.hilbert(gamma))
    b, a = signal.butter(4, 10, fs=fs)
    genv = signal.filtfilt(b, a, genv)
    lo, hi = np.median(genv) + 1.5 * mad(genv), np.median(genv) + 4 * mad(genv)
    lab, n = ndimage.label(genv > lo)
    keep = ndimage.maximum(genv, lab, index=np.arange(1, n + 1)) > hi
    burst = keep[lab - 1] & (lab > 0)
    burst_score = scores(burst[~art], df.gamma_burst_active.to_numpy()[~art])

    # phase-amplitude coupling: Tort modulation index, against time-shifted surrogates
    b, a = signal.butter(3, [4, 8], btype="band", fs=fs)
    phase = np.angle(signal.hilbert(signal.filtfilt(b, a, y_clean)))
    amp = np.abs(signal.hilbert(gamma))
    edges = np.linspace(-np.pi, np.pi, 19)

    def modulation_index(ph, am):
        m = np.array([am[(ph >= edges[i]) & (ph < edges[i + 1])].mean() for i in range(18)])
        pdist = m / m.sum()
        return float((np.log(18) + np.sum(pdist * np.log(pdist))) / np.log(18)), pdist

    # coupling is only expected while gamma is present: compute it within the detected bursts
    ph_b, am_b = phase[burst & ~art], amp[burst & ~art]
    mi, pdist = modulation_index(ph_b, am_b)
    mi_all, _ = modulation_index(phase, amp)
    rng = np.random.default_rng(0)
    surr = [modulation_index(ph_b, np.roll(am_b, rng.integers(int(0.2 * fs), len(am_b) - int(0.2 * fs))))[0]
            for _ in range(200)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    sel = slice(int(20 * fs), int(40 * fs))
    t = df.time_s.to_numpy()
    axes[0].plot(t[sel], genv[sel], color="0.3", lw=0.8, label="gamma envelope")
    axes[0].fill_between(t[sel], 0, genv[sel].max(), where=df.gamma_burst_active.to_numpy()[sel] > 0,
                         color="#3b6ea8", alpha=0.2, label="true bursts")
    axes[0].fill_between(t[sel], 0, genv[sel].max() * 0.5, where=burst[sel], color="#d08c3c", alpha=0.5,
                         label="detected")
    axes[0].set_xlabel("time (s)"); axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Gamma bursts (hysteresis on the Hilbert envelope)", fontsize=9)
    c = (edges[:-1] + edges[1:]) / 2
    axes[1].bar(np.degrees(c), pdist, width=18, color="#3b6ea8")
    axes[1].set_xlabel("theta phase (deg)"); axes[1].set_ylabel("normalised gamma amplitude")
    axes[1].set_title(f"Coupling within bursts: MI = {mi:.4f} (surrogate 99th pct {np.percentile(surr, 99):.4f})",
                      fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "modality_lfp.png", dpi=150); plt.close(fig)

    return dict(line_noise_attenuation_db=line_drop_db, artifact_detection=art_score,
                gamma_bursts=burst_score, pac_modulation_index_in_bursts=round(mi, 4),
                pac_modulation_index_whole_recording=round(mi_all, 4),
                pac_surrogate_p=float((np.sum(np.array(surr) >= mi) + 1) / (len(surr) + 1)),
                preferred_theta_phase_deg=round(float(np.degrees(c[np.argmax(pdist)])), 0))


# ------------------------------------------------------------------ 03 spike trains
def spike_trains():
    df = pd.read_csv(DATA / "03_spike_train.csv")
    trials = pd.read_csv(DATA / "03_trial_events.csv").trial_time_s.to_numpy()
    a = np.sort(df.loc[df.unit == "A", "spike_time_s"].to_numpy())
    bdf = df[df.unit == "B"].sort_values("spike_time_s")
    b = bdf.spike_time_s.to_numpy()

    isi_b = np.diff(b, prepend=-np.inf)
    flagged = isi_b < 0.002
    violations = scores(flagged, bdf.is_injected_violation.to_numpy() == 1)

    aligned = np.concatenate([a - tr for tr in trials])
    base = ((aligned >= -0.2) & (aligned < 0)).sum() / len(trials) / 0.2
    evoked = ((aligned >= 0) & (aligned < 0.2)).sum() / len(trials) / 0.2

    def ccg_zero(x, y, w=0.002):
        return int(sum(np.sum(np.abs(x - t) <= w) for t in y))

    observed = ccg_zero(a, b)
    rng = np.random.default_rng(0)
    jitter = [ccg_zero(a, b + rng.uniform(-0.025, 0.025, b.size)) for _ in range(200)]

    burst_frac = float(np.mean(np.diff(b) < 0.01))
    return dict(rate_hz=dict(A=round(len(a) / 60, 2), B=round(len(b) / 60, 2)),
                unit_B_fraction_isi_below_10ms=round(burst_frac, 3),
                refractory_violations_unit_B=violations,
                psth_unit_A_hz=dict(baseline=round(base, 1), evoked_0_200ms=round(evoked, 1)),
                synchrony_pairs_within_2ms=observed,
                synchrony_jitter_mean=round(float(np.mean(jitter)), 1),
                synchrony_jitter_p=float((np.sum(np.array(jitter) >= observed) + 1) / 201))


# ------------------------------------------------------------------ 04 snippets
def snippets():
    df = pd.read_csv(DATA / "04_spike_snippets.csv")
    cols = [c for c in df.columns if c.startswith("sample_")]
    W = df[cols].to_numpy()
    fs = 20000

    # artifact rejection from shape: an action potential has a sharp trough;
    # count samples below half the trough (width) and the trough's share of the peak-to-peak
    trough = W.min(1)
    width = (W <= trough[:, None] / 2).sum(1) / fs * 1e3
    is_art = (width > 0.6) | (W.max(1) > -trough)
    art_score = scores(is_art, df.is_artifact == 1)

    keep = ~is_art
    X = PCA(3).fit_transform(W[keep])           # PCA on waveforms in µV, not per-sample z-scores
    sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)) for k in range(2, 7)}
    k = max(sil, key=sil.get)
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    truth = df.true_unit_id.to_numpy()[keep]
    ok = truth >= 0
    ari = adjusted_rand_score(truth[ok], lab[ok])
    ari3 = adjusted_rand_score(truth[ok], KMeans(3, n_init=10, random_state=0).fit_predict(X)[ok])

    t = df.time_s.to_numpy()[keep]
    halves = {}
    for name, m in (("first half", t < np.median(t)), ("second half", t >= np.median(t))):
        Xh = PCA(3).fit_transform(W[keep][m])
        halves[name] = round(adjusted_rand_score(truth[m][truth[m] >= 0],
                                                 KMeans(3, n_init=10, random_state=0).fit_predict(Xh)[truth[m] >= 0]), 3)
    return dict(artifact_rejection=art_score, silhouette_best_k=int(k), ari_best_k=round(ari, 3),
                ari_k3=round(ari3, 3), ari_k3_by_half=halves)


# ------------------------------------------------------------------ 05 binned counts
def binned():
    df = pd.read_csv(DATA / "05_binned_counts.csv")
    chans = [c for c in df.columns if c.startswith("ch")]
    st = pd.DataFrame({"mean": df[chans].mean(), "var": df[chans].var()})
    st["fano"] = st["var"] / st["mean"]
    nan = df.ch5.isna()
    gaps = int(((nan != nan.shift()) & nan).sum())
    half = df.time_s < df.time_s.max() / 2
    coarse = df.ch1.iloc[: len(df) // 4 * 4].to_numpy().reshape(-1, 4).sum(1)
    return dict(dead=st.index[st["mean"] == 0].tolist(),
                overdispersed=st.index[st.fano > 1.5].tolist(),
                fano=st.fano.round(2).dropna().to_dict(),
                ch5_nan_gaps=gaps, ch5_nan_seconds=float(nan.sum() * 0.05),
                ch3_rate_hz=dict(first_half=round(df.loc[half, "ch3"].mean() / 0.05, 1),
                                 second_half=round(df.loc[~half, "ch3"].mean() / 0.05, 1)),
                ch1_fano_50ms_vs_200ms=[round(float(st.fano["ch1"]), 2), round(float(coarse.var() / coarse.mean()), 2)])


# ------------------------------------------------------------------ 06 decoding
def decoding():
    df = pd.read_csv(DATA / "06_neural_behavior.csv")
    df[["vx", "vy"]] = df[["vx", "vy"]].interpolate(limit=2, limit_direction="both")
    chans = [c for c in df.columns if c.startswith("ch")]
    dt = 0.02

    def design(lag_bins, history=3):
        cols = {f"{c}_{h}": df[c].shift(lag_bins + h) for c in chans for h in range(history)}
        return pd.concat([pd.DataFrame(cols), df[["time_s", "vx", "vy", "epoch"]]], axis=1).dropna()

    def blocked_cv(d, feats, folds=5, gap_s=0.5):
        """Contiguous folds over the task epochs, with a gap on each side of the test fold."""
        task = d[d.epoch == "task"].reset_index(drop=True)
        edges = np.linspace(0, len(task), folds + 1).astype(int)
        yt, yp, yb = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            t0, t1 = task.time_s.iloc[a], task.time_s.iloc[b - 1]
            tr = (task.time_s < t0 - gap_s) | (task.time_s > t1 + gap_s)
            te = np.zeros(len(task), bool)
            te[a:b] = True
            m = Ridge(1.0).fit(task.loc[tr, feats], task.loc[tr, ["vx", "vy"]])
            yt.append(task.loc[te, ["vx", "vy"]].to_numpy())
            yp.append(m.predict(task.loc[te, feats]))
            yb.append(np.broadcast_to(task.loc[tr, ["vx", "vy"]].mean().to_numpy(), yt[-1].shape))
        yt, yp, yb = map(np.concatenate, (yt, yp, yb))
        return r2_score(yt, yp), r2_score(yt, yb)

    sweep = {}
    for lag in range(0, 11):
        d = design(lag)
        feats = [c for c in d.columns if c.startswith("ch")]
        sweep[lag * dt * 1e3] = blocked_cv(d, feats)[0]
    best = max(sweep, key=sweep.get)
    d = design(int(round(best / 1e3 / dt)))
    feats = [c for c in d.columns if c.startswith("ch")]
    cv_r2, cv_baseline = blocked_cv(d, feats)

    # the two task blocks cover different movement directions (the direction is a random walk)
    angle = np.degrees(np.arctan2(df.vy, df.vx))
    block1, block2 = (df.epoch == "task") & (df.time_s < 35), (df.epoch == "task") & (df.time_s >= 40)
    coverage = {name: [round(float(np.percentile(angle[m].dropna(), q)), 0) for q in (5, 95)]
                for name, m in (("block 1", block1), ("block 2", block2))}

    train = (d.epoch == "task") & (d.time_s < 35)
    m = Ridge(1.0).fit(d.loc[train, feats], d.loc[train, ["vx", "vy"]])
    base = DummyRegressor().fit(d.loc[train, feats], d.loc[train, ["vx", "vy"]])
    out = {}
    for name, sel in (("block 2 (train on block 1)", (d.epoch == "task") & (d.time_s >= 40)),
                      ("rest", d.epoch == "rest"), ("out of distribution", d.epoch == "ood")):
        yt, yp = d.loc[sel, ["vx", "vy"]].to_numpy(), m.predict(d.loc[sel, feats])
        out[name] = dict(r2=round(r2_score(yt, yp), 3),
                         r2_mean_baseline=round(r2_score(yt, base.predict(d.loc[sel, feats])), 3),
                         speed_true=round(float(np.linalg.norm(yt, axis=1).mean()), 1),
                         speed_pred=round(float(np.linalg.norm(yp, axis=1).mean()), 1))

    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.plot(list(sweep), list(sweep.values()), "o-", color="#3b6ea8")
    ax.axvline(100, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("neural lead (ms)"); ax.set_ylabel("blocked cross-validated R²")
    ax.set_title("Decoder lag sweep (true lead 100 ms)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "modality_decoder_lag.png", dpi=150); plt.close(fig)
    return dict(best_lead_ms=best, lag_sweep_cv_r2={int(k): round(v, 3) for k, v in sweep.items()},
                blocked_cv=dict(r2=round(cv_r2, 3), r2_mean_baseline=round(cv_baseline, 3)),
                direction_range_deg_5_95=coverage, **out)


# ------------------------------------------------------------------ 07 multichannel QC
def multichannel():
    df = pd.read_csv(DATA / "07_multichannel.csv")
    meta = pd.read_csv(DATA / "07_channels_metadata.csv").set_index("channel_id")
    chs = [c for c in df.columns if c.startswith("ch")]
    fs = 20000
    b, a = signal.butter(4, [300, 3000], btype="band", fs=fs)
    X = signal.filtfilt(b, a, df[chs].to_numpy(), axis=0)
    noise = np.array([mad(X[:, i]) for i in range(len(chs))])
    med = np.median(noise)
    status = np.where(noise < 0.3 * med, "dead", np.where(noise > 2 * med, "noisy", "healthy")).astype(object)
    half = len(X) // 2
    ratio = np.array([mad(X[half:, i]) / mad(X[:half, i]) for i in range(len(chs))])
    status[(ratio > 2) & (status == "healthy")] = "degrading"

    # crosstalk: spike-triggered average of every channel on each channel's own spikes
    ctalk = {}
    for i in range(len(chs)):
        if status[i] != "healthy":
            continue
        ev = signal.find_peaks(-X[:, i], height=4.5 * noise[i], distance=int(0.001 * fs))[0]
        ev = ev[(ev > 30) & (ev < len(X) - 30)]
        if len(ev) < 5:
            continue
        sta = np.array([[X[e - 30:e + 30, j] for e in ev] for j in range(len(chs))]).mean(1)
        for j in range(len(chs)):
            if j != i and status[j] == "healthy":
                r = np.corrcoef(sta[i], sta[j])[0, 1]
                gain = np.ptp(sta[j]) / np.ptp(sta[i])
                if r > 0.8 and 0.15 < gain < 1:          # an attenuated copy, not the source
                    ctalk[j] = dict(source=i, shape_r=round(float(r), 2), gain=round(float(gain), 2))
    final = status.copy()
    for j in ctalk:
        final[j] = "healthy_with_crosstalk"

    # common-mode removal: correlation between channels below 300 Hz, before vs after median referencing
    raw = df[chs].to_numpy()
    good = [i for i in range(len(chs)) if status[i] == "healthy"]
    car = raw - np.median(raw[:, good], axis=1, keepdims=True)
    b, a = signal.butter(4, 300, fs=fs)
    def mean_corr(Y):
        Z = signal.filtfilt(b, a, Y[:, good], axis=0)
        C = np.corrcoef(Z.T)
        return float(C[np.triu_indices(len(good), 1)].mean())

    table = pd.DataFrame({"metadata": meta.metadata_status, "data_driven": final,
                          "truth": meta.true_status, "noise_uv": noise.round(1),
                          "noise_ratio_2nd_1st_half": ratio.round(2)}, index=meta.index)
    table.to_csv(RES / "modality_channel_qc.csv")
    return dict(channels_correct=int((table.data_driven == table.truth).sum()), channels=len(table),
                metadata_correct=int((table.metadata == table.truth).sum()),
                metadata_disagreements=table.index[table.metadata != table.data_driven].tolist(),
                crosstalk=ctalk,
                low_freq_interchannel_corr=dict(before=round(mean_corr(raw), 3), after_median_ref=round(mean_corr(car), 3)))


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = {"02_lfp": lfp(), "03_spike_trains": spike_trains(), "04_snippets": snippets(),
           "05_binned_counts": binned(), "06_decoding": decoding(), "07_multichannel": multichannel()}
    (RES / "modalities.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
