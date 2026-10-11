"""
Synthetic datasets for six other kinds of electrophysiology data, each with
ground truth for several problems at once:

  02 LFP              theta, gamma bursts coupled to theta phase, line noise, movement artifacts
  03 spike trains     tonic + evoked unit, bursty unit, injected refractory violations, synchrony
  04 spike snippets   three units (one drifting), rare unit, non-AP artifacts
  05 binned counts    Poisson, overdispersed, silent, non-stationary, dead and gappy channels
  06 neural+behaviour cosine-tuned channels leading velocity by 100 ms, NaN dropouts, saturation
  07 multichannel     dead/noisy/degrading channels, wrong metadata, crosstalk, common-mode drift

The scenarios were specified by the author; this generator was written with an
AI assistant. Every dataset uses a fixed seed.

Run: python modalities/generate.py   (writes modalities/data/)
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(__file__))
from _spikelib import make_ap_waveform, white_noise, random_walk_drift, embed_spike, min_isi_indices

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows, {len(df.columns)} cols)")


# ================================================================ 2. LFP
def gen_lfp():
    fs = 1000
    dur = 120
    n = int(fs * dur)
    t = np.arange(n) / fs
    rng = np.random.default_rng(2)

    # background LFP-like noise: bandpass-filtered white noise, 1-100Hz
    raw_noise = rng.normal(0, 1, n)
    b, a = butter(4, [1 / (fs / 2), 100 / (fs / 2)], btype="band")
    background = filtfilt(b, a, raw_noise) * 8.0

    # continuous theta rhythm (6Hz)
    theta_phase = 2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi)
    theta = 25 * np.sin(theta_phase)

    # non-stationary gamma bursts (40Hz), amplitude-modulated by theta phase
    # (phase-amplitude coupling) only during 5 burst windows
    burst_centers = np.linspace(15, 105, 5)
    burst_half_width = 2.0  # seconds
    burst_envelope = np.zeros(n)
    for c in burst_centers:
        w = 0.5 * (1 + np.cos(np.pi * np.clip((t - c) / burst_half_width, -1, 1)))
        w[np.abs(t - c) > burst_half_width] = 0
        burst_envelope = np.maximum(burst_envelope, w)
    gamma_amp = 3 + 15 * burst_envelope  # baseline 3uV, up to 18uV in bursts
    preferred_phase = 0.6
    pac_gain = 0.4 + 0.6 * burst_envelope  # coupling strength also ramps up in-burst
    gamma_amp_eff = gamma_amp * (1 - pac_gain + pac_gain * (0.5 + 0.5 * np.cos(theta_phase - preferred_phase)))
    gamma = gamma_amp_eff * np.sin(2 * np.pi * 40 * t + rng.uniform(0, 2 * np.pi))

    # line noise
    line = 15 * np.sin(2 * np.pi * 60 * t) + 4 * np.sin(2 * np.pi * 120 * t)

    # movement artifacts: 2 short broadband bursts
    artifact_centers = [38.0, 91.5]
    artifact_half_width = 0.3
    artifact = np.zeros(n)
    is_artifact = np.zeros(n, dtype=int)
    for c in artifact_centers:
        mask = np.abs(t - c) <= artifact_half_width
        win = 0.5 * (1 - np.cos(2 * np.pi * (np.where(mask)[0] - np.where(mask)[0][0]) / mask.sum()))
        artifact[mask] = rng.normal(0, 80, mask.sum()) * win
        is_artifact[mask] = 1

    lfp = background + theta + gamma + line + artifact
    gamma_burst_active = (burst_envelope > 0.5).astype(int)

    df = pd.DataFrame({
        "sample_index": np.arange(n),
        "time_s": t,
        "lfp_uv": lfp.astype(np.float32),
        "gamma_burst_active": gamma_burst_active,
        "is_artifact": is_artifact,
    })
    save(df, "02_lfp.csv")
    print("  -> fs=1000Hz. Embedded: theta(6Hz)+gamma(40Hz) bursts w/ phase-amplitude "
          "coupling (non-stationary), 60/120Hz line noise, 2 movement artifacts, background noise.")


# ================================================================ 3. Spike trains
def make_renewal_train(rate_hz, duration, refractory_s, rng):
    mean_isi = 1.0 / rate_hz
    times = []
    t = 0.0
    while t < duration:
        isi = refractory_s + rng.exponential(max(mean_isi - refractory_s, 1e-4))
        t += isi
        if t < duration:
            times.append(t)
    return np.array(times)


def gen_spike_trains():
    rng = np.random.default_rng(3)
    duration = 60.0

    # trial events
    n_trials = 15
    trial_times = np.sort(rng.uniform(2, duration - 2, n_trials))

    # unit A: tonic ~5Hz baseline + evoked burst (~15Hz for 200ms) after each trial
    a_baseline = make_renewal_train(5.0, duration, 0.002, rng)
    a_evoked = []
    for tt in trial_times:
        n_eb = rng.poisson(15.0 * 0.2)
        a_evoked.extend(tt + rng.uniform(0, 0.2, n_eb))
    a_times = np.sort(np.concatenate([a_baseline, a_evoked]))
    # greedy refractory prune
    keep = [a_times[0]] if len(a_times) else []
    for x in a_times[1:]:
        if x - keep[-1] >= 0.002:
            keep.append(x)
    a_times = np.array(keep)

    # unit B: sparse tonic ~1Hz baseline + periodic bursts (3-6 spikes, 4-8ms ISI)
    b_baseline = make_renewal_train(1.0, duration, 0.003, rng)
    b_bursts = []
    burst_start = rng.uniform(1, 3)
    while burst_start < duration - 1:
        n_sp = rng.integers(3, 7)
        tt = burst_start
        for _ in range(n_sp):
            b_bursts.append(tt)
            tt += rng.uniform(0.004, 0.008)
        burst_start += rng.uniform(2, 4)
    b_times = np.sort(np.concatenate([b_baseline, np.array(b_bursts)]))
    keep = [b_times[0]] if len(b_times) else []
    for x in b_times[1:]:
        if x - keep[-1] >= 0.003:
            keep.append(x)
    b_times = np.array(keep)

    # inject 4 refractory-violation duplicates into unit B (deliberate data-quality flaw)
    violation_flags_b = np.zeros(len(b_times), dtype=int)
    dup_source_idx = rng.choice(len(b_times), size=4, replace=False)
    dup_times = b_times[dup_source_idx] + rng.uniform(0.0003, 0.0006, 4)
    b_times_full = np.concatenate([b_times, dup_times])
    violation_flags = np.concatenate([violation_flags_b, np.ones(4, dtype=int)])
    order = np.argsort(b_times_full)
    b_times_full = b_times_full[order]
    violation_flags = violation_flags[order]

    # inject ~10 near-synchronous companion B spikes around random A spikes (cross-corr peak)
    sync_source = rng.choice(a_times, size=10, replace=False)
    sync_times = sync_source + rng.uniform(-0.002, 0.002, 10)
    b_times_full = np.concatenate([b_times_full, sync_times])
    violation_flags = np.concatenate([violation_flags, np.zeros(10, dtype=int)])
    order = np.argsort(b_times_full)
    b_times_full = b_times_full[order]
    violation_flags = violation_flags[order]

    df = pd.DataFrame({
        "unit": ["A"] * len(a_times) + ["B"] * len(b_times_full),
        "spike_time_s": np.concatenate([a_times, b_times_full]),
        "is_injected_violation": np.concatenate([np.zeros(len(a_times), dtype=int), violation_flags]),
    }).sort_values(["unit", "spike_time_s"]).reset_index(drop=True)
    save(df, "03_spike_train.csv")

    df_events = pd.DataFrame({"trial_time_s": trial_times})
    save(df_events, "03_trial_events.csv")
    print(f"  -> Unit A: tonic + event-locked evoked response ({n_trials} trials). "
          f"Unit B: bursty + 4 injected refractory violations + ~10 synchronous spikes with A.")


# ================================================================ 4. Spike waveforms/snippets
def gen_snippets():
    fs = 20000
    window_ms = 3
    window_len = int(window_ms / 1000 * fs)
    rng = np.random.default_rng(4)
    rows = []

    def make_snippet(amp, dur_ms, asymmetry, noise_std, jitter_max=2):
        w, pk = make_ap_waveform(fs, amp=amp, dur_ms=dur_ms, asymmetry=asymmetry)
        buf = white_noise(window_len + 20, noise_std, seed=rng.integers(1e9))
        center = window_len // 2 + rng.integers(-jitter_max, jitter_max + 1)
        embed_spike(buf, center + 10, w, pk)
        return buf[10:10 + window_len]

    def make_artifact_snippet(noise_std=8):
        buf = white_noise(window_len, noise_std, seed=rng.integers(1e9))
        width = window_len
        tt = np.linspace(-1, 1, width)
        amp = rng.choice([1, -1]) * rng.uniform(250, 400)
        buf += amp * np.exp(-tt ** 2 / (2 * 0.5 ** 2))
        return buf

    # unit 1: drifts in amplitude across the session (electrode micromotion)
    n1 = 60
    times1 = np.sort(rng.uniform(0, 60, n1))
    amps1 = np.linspace(90, 60, n1)  # decreasing amplitude over time
    for tt, a in zip(times1, amps1):
        snip = make_snippet(a, 1.6, 0.35, noise_std=5)
        rows.append((tt, 0, 0, snip))

    # unit 2: similar shape/amplitude range to unit 1 -> hard to separate
    n2 = 60
    times2 = np.sort(rng.uniform(0, 60, n2))
    for tt in times2:
        snip = make_snippet(55, 1.5, 0.4, noise_std=5)
        rows.append((tt, 1, 0, snip))

    # unit 3: rare, distinct, large amplitude
    n3 = 10
    times3 = np.sort(rng.uniform(0, 60, n3))
    for tt in times3:
        snip = make_snippet(150, 1.8, 0.6, noise_std=5)
        rows.append((tt, 2, 0, snip))

    # artifacts mixed in
    n4 = 10
    times4 = np.sort(rng.uniform(0, 60, n4))
    for tt in times4:
        snip = make_artifact_snippet()
        rows.append((tt, -1, 1, snip))

    rows.sort(key=lambda r: r[0])
    data = {"snippet_id": np.arange(len(rows)),
            "time_s": [r[0] for r in rows],
            "true_unit_id": [r[1] for r in rows],
            "is_artifact": [r[2] for r in rows]}
    samples = np.array([r[3] for r in rows])
    for i in range(window_len):
        data[f"sample_{i}"] = samples[:, i]
    df = pd.DataFrame(data)
    save(df, "04_spike_snippets.csv")
    print(f"  -> fs=20000Hz, {window_len}-sample (3ms) windows. Unit 0 drifts 90->60uV across "
          f"the session, unit 1 overlaps unit 0's later amplitude range, unit 2 is rare+distinct, "
          f"10 artifact snippets mixed in (true_unit_id=-1).")


# ================================================================ 5. Binned spike counts
def gen_binned_counts():
    rng = np.random.default_rng(5)
    bin_size = 0.05
    duration = 120.0
    n_bins = int(duration / bin_size)
    t = np.arange(n_bins) * bin_size

    ch0 = rng.poisson(0.5, n_bins)  # normal tonic ~10Hz

    # overdispersed/bursty: doubly-stochastic Poisson (Gamma-mixed)
    lam1 = rng.gamma(shape=0.3, scale=0.5 / 0.3, size=n_bins)
    ch1 = rng.poisson(lam1)

    ch2 = rng.poisson(0.01, n_bins)  # sparse, near-silent

    # non-stationary: ramps from 5Hz to 20Hz over the session
    ramp = np.clip((t - 50) / 20, 0, 1)  # transition centered ~60s
    lam3 = 0.25 + 0.75 * ramp
    ch3 = rng.poisson(lam3)

    ch4 = np.zeros(n_bins, dtype=int)  # dead channel

    ch5 = rng.poisson(0.5, n_bins).astype(float)
    drop_start = int(40 / bin_size)
    drop_len = int(2 / bin_size)
    ch5[drop_start:drop_start + drop_len] = np.nan  # dropped stretch

    df = pd.DataFrame({
        "bin_index": np.arange(n_bins), "time_s": t,
        "ch0": ch0, "ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4, "ch5": ch5,
    })
    save(df, "05_binned_counts.csv")
    print("  -> bin_size=50ms, 120s. ch0=normal Poisson, ch1=overdispersed/bursty, "
          "ch2=near-silent, ch3=non-stationary rate (ramps up), ch4=dead (all-zero), "
          "ch5=normal but with a 2s dropped (NaN) stretch. Aggregate bins yourself to "
          "explore bin-size trade-offs.")


# ================================================================ 6. Neural + behavioral
def gen_neural_behavior():
    rng = np.random.default_rng(6)
    bin_size = 0.02
    duration = 90.0
    n = int(duration / bin_size)
    t = np.arange(n) * bin_size

    epochs = np.array(["task"] * n, dtype=object)
    def set_epoch(lo, hi, label):
        epochs[(t >= lo) & (t < hi)] = label
    set_epoch(0, 5, "rest")
    set_epoch(35, 40, "rest")
    set_epoch(65, 70, "rest")
    set_epoch(70, 90, "ood")

    # smooth pursuit direction/speed for task epochs
    accel = rng.normal(0, 1, n)
    direction = np.cumsum(accel) * 0.02
    speed_raw = np.abs(np.convolve(rng.normal(0, 1, n), np.ones(25) / 25, mode="same"))
    speed = 40 + 120 * (speed_raw / (speed_raw.max() + 1e-9))

    vx_true = speed * np.cos(direction)
    vy_true = speed * np.sin(direction)

    rest_mask = np.isin(epochs, ["rest"])
    vx_true[rest_mask] = 0.0
    vy_true[rest_mask] = 0.0

    ood_mask = epochs == "ood"
    f_rot = 0.5
    ood_dir = 2 * np.pi * f_rot * (t - 70)
    vx_true[ood_mask] = 250 * np.cos(ood_dir[ood_mask])
    vy_true[ood_mask] = 250 * np.sin(ood_dir[ood_mask])

    # true neural-to-behavior lag: neural activity leads behavior by 100ms
    lag_bins = int(0.1 / bin_size)
    vx_enc = np.roll(vx_true, -lag_bins)
    vy_enc = np.roll(vy_true, -lag_bins)
    vx_enc[-lag_bins:] = vx_enc[-lag_bins - 1]
    vy_enc[-lag_bins:] = vy_enc[-lag_bins - 1]
    speed_enc = np.sqrt(vx_enc ** 2 + vy_enc ** 2)
    dir_enc = np.arctan2(vy_enc, vx_enc)

    n_ch = 4
    lam_base = 0.05
    gain = 0.012
    speed_cap = 180.0
    counts = np.zeros((n, n_ch), dtype=int)
    for k in range(n_ch):
        pref = k * (2 * np.pi / n_ch)
        tuning = np.clip(np.cos(dir_enc - pref), 0, None)
        speed_clipped = np.minimum(speed_enc, speed_cap)
        lam = lam_base + gain * speed_clipped * tuning
        counts[:, k] = rng.poisson(np.clip(lam, 0, None))

    # measurement noise + missing behavior samples
    vx_obs = vx_true + rng.normal(0, 5, n)
    vy_obs = vy_true + rng.normal(0, 5, n)
    drop_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    vx_obs[drop_idx] = np.nan
    vy_obs[drop_idx] = np.nan

    df = pd.DataFrame({
        "bin_index": np.arange(n), "time_s": t,
        "ch0": counts[:, 0], "ch1": counts[:, 1], "ch2": counts[:, 2], "ch3": counts[:, 3],
        "vx": vx_obs, "vy": vy_obs, "epoch": epochs,
    })
    save(df, "06_neural_behavior.csv")
    print("  -> bin_size=20ms, 90s. 4 cosine-tuned channels encode velocity ~100ms *before* "
          "it happens (true lag to recover). rest epochs = zero true velocity (sanity check). "
          "ood epoch (70-90s) = fast sustained rotation whose speed exceeds each channel's "
          "tuning saturation cap -> a linear decoder trained on task epochs should degrade "
          "there. ~3% of vx/vy rows are NaN (dropped tracking).")


# ================================================================ 7. Multichannel + metadata
def gen_multichannel():
    fs = 20000
    dur = 3.0
    n = int(fs * dur)
    rng = np.random.default_rng(7)

    shared = random_walk_drift(n, step_std=0.03, seed=7, smooth_win=501)
    shared -= shared.mean()
    shared *= (15.0 / (np.abs(shared).max() + 1e-9))

    voltage = np.zeros((8, n), dtype=np.float32)
    true_status = {}

    def add_unit(ch, amp, noise_std, n_spikes, seg=(0, n)):
        trace = np.zeros(n, dtype=np.float32)
        trace[seg[0]:seg[1]] += white_noise(seg[1] - seg[0], noise_std, seed=int(rng.integers(1e9)))
        w, pk = make_ap_waveform(fs, amp=amp)
        spikes = min_isi_indices(seg[1] - seg[0], n_spikes, int(0.02 * fs), seed=int(rng.integers(1e9))) + seg[0]
        for s in spikes:
            embed_spike(trace, s, w, pk)
        return trace, spikes

    for ch, amp in zip([0, 1, 2], [70, 75, 65]):
        trace, _ = add_unit(ch, amp, 6.0, 12)
        voltage[ch] = trace
        true_status[ch] = "healthy"

    voltage[3] = white_noise(n, 1.0, seed=103)  # dead
    true_status[3] = "dead"

    trace4, _ = add_unit(4, 25, 40.0, 12)  # noisy/high-impedance, spikes barely visible
    voltage[4] = trace4
    true_status[4] = "noisy"

    half = n // 2
    trace5a, _ = add_unit(5, 70, 6.0, 6, seg=(0, half))
    trace5b = np.zeros(n, dtype=np.float32)
    trace5b[half:] = white_noise(n - half, 35.0, seed=105)  # degrades: noisy, no spikes, 2nd half
    voltage[5] = trace5a + trace5b
    true_status[5] = "degrading"

    trace6, spikes6_clean_idx = add_unit(6, 80, 6.0, 10)
    clean6 = np.zeros(n, dtype=np.float32)
    w, pk = make_ap_waveform(fs, amp=80)
    for s in spikes6_clean_idx:
        embed_spike(clean6, s, w, pk)
    voltage[6] = trace6
    voltage[7] = 0.3 * clean6 + white_noise(n, 6.0, seed=107)  # crosstalk from ch6
    true_status[6] = "healthy"
    true_status[7] = "healthy_with_crosstalk"

    voltage += shared[np.newaxis, :]  # shared common-mode artifact on every channel

    cols = {"sample_index": np.arange(n), "time_s": np.arange(n) / fs}
    for ch in range(8):
        cols[f"ch{ch}_uv"] = voltage[ch]
    df = pd.DataFrame(cols)
    save(df, "07_multichannel.csv")

    xy = {0: (0, 0), 1: (50, 0), 2: (100, 0), 3: (150, 0),
          4: (0, 50), 5: (50, 50), 6: (100, 50), 7: (150, 50)}
    impedance = {0: 280, 1: 310, 2: 265, 3: 9200, 4: 2200, 5: 300, 6: 280, 7: 260}
    metadata_status = {0: "healthy", 1: "healthy", 2: "healthy", 3: "dead",
                        4: "healthy", 5: "healthy", 6: "healthy", 7: "healthy"}  # ch4 wrong; ch5 stale
    meta = pd.DataFrame({
        "channel_id": list(range(8)),
        "x_um": [xy[c][0] for c in range(8)],
        "y_um": [xy[c][1] for c in range(8)],
        "impedance_kohm": [impedance[c] for c in range(8)],
        "metadata_status": [metadata_status[c] for c in range(8)],
        "true_status": [true_status[c] for c in range(8)],
    })
    save(meta, "07_channels_metadata.csv")
    print("  -> fs=20000Hz, 3s, 8 channels. ch3=dead, ch4=noisy (metadata wrongly says "
          "'healthy' despite a suspiciously high impedance), ch5=degrades mid-recording "
          "(metadata is stale/static, can't show this), ch6/ch7=adjacent electrodes with "
          "crosstalk. Shared common-mode drift on all channels (try common-average "
          "referencing). true_status is for self-checking only -- don't read it before you "
          "do your own per-channel assessment.")


if __name__ == "__main__":
    gen_lfp()
    gen_spike_trains()
    gen_snippets()
    gen_binned_counts()
    gen_neural_behavior()
    gen_multichannel()
    print("done")
