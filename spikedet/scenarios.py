"""Synthetic recordings with ground truth, one per failure mode.

Each scenario isolates one way spike detection goes wrong on real
extracellular data (low SNR, drift, mixed amplitudes, bursts, collisions,
artifacts, mains hum, shared artifacts across channels, sampling rate,
streaming, and edge cases). Every scenario is rebuilt from a fixed seed, so the
benchmark needs no data files.

Units: microvolts, Hz, samples. Ground truth is the sample of each spike's
negative peak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    name: str
    description: str
    voltage: np.ndarray                 # (n,) or (channels, n)
    fs: float
    truth: np.ndarray | list            # spike samples (list of arrays for multichannel)
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- building blocks
def ap_waveform(fs, amp=60.0, dur_ms=1.6, asymmetry=0.35):
    """Biphasic extracellular action potential: sharp trough, slower smaller rebound.
    Returns (waveform, index of the trough)."""
    n = max(5, int(round(dur_ms / 1000 * fs)))
    t = np.linspace(-1, 1, n)
    w = -np.exp(-((t + 0.15) ** 2) / (2 * 0.12 ** 2)) + asymmetry * np.exp(-((t - 0.35) ** 2) / (2 * 0.25 ** 2))
    w = amp * w / np.abs(w).max()
    return w, int(np.argmin(w))


def white_noise(n, std, seed):
    return np.random.default_rng(seed).normal(0, std, n)


def embed(trace, center, w, peak):
    """Add waveform w so that its trough lands on `center`; truncates at the edges."""
    start = center - peak
    s0, s1 = max(0, start), min(len(trace), start + len(w))
    if s1 > s0:
        trace[s0:s1] += w[s0 - start:s1 - start]


def spaced_times(n, count, min_gap, seed, margin=200, avoid=()):
    """`count` random samples at least `min_gap` apart (and from `avoid`)."""
    rng = np.random.default_rng(seed)
    out = list(avoid)
    picked = []
    for _ in range(count * 500):
        if len(picked) == count:
            break
        c = int(rng.integers(margin, n - margin))
        if all(abs(c - x) >= min_gap for x in out):
            out.append(c)
            picked.append(c)
    return np.array(sorted(picked), dtype=np.int64)


def _simple(name, desc, fs, dur_s, noise, amp, count, gap_s, seed, **kw):
    n = int(fs * dur_s)
    x = white_noise(n, noise, seed)
    w, pk = ap_waveform(fs, amp, **kw)
    t = spaced_times(n, count, int(gap_s * fs), seed)
    for s in t:
        embed(x, s, w, pk)
    return Scenario(name, desc, x, fs, t)


# ----------------------------------------------------------------- scenarios
def clean():
    return _simple("01_clean", "12 isolated 80 µV spikes in 3 µV noise", 30000, 1.0, 3.0, 80.0, 12, 0.02, 1)


def low_snr():
    return _simple("02_low_snr", "40 µV spikes in 15 µV noise (peak 2.7 × broadband noise σ)", 30000, 1.5, 15.0, 40.0, 25, 0.015, 2)


def drift():
    fs, n = 30000, 60000
    x = white_noise(n, 6.0, 3)
    walk = np.cumsum(np.random.default_rng(3).normal(0, 0.05, n))
    walk = np.convolve(walk, np.ones(1001) / 1001, mode="same")
    x += walk - walk.mean()
    w, pk = ap_waveform(fs, 70.0)
    t = spaced_times(n, 20, 600, 3)
    for s in t:
        embed(x, s, w, pk)
    return Scenario("03_drift", "slow random-walk baseline wander under 70 µV spikes", x, fs, t)


def mixed_amplitudes():
    fs, n = 30000, 60000
    x = white_noise(n, 6.0, 4)
    t = spaced_times(n, 30, 450, 4)
    unit = np.random.default_rng(4).integers(0, 3, len(t))
    for s, u in zip(t, unit):
        embed(x, s, *ap_waveform(fs, (25.0, 70.0, 160.0)[u]))
    return Scenario("04_mixed_amplitudes", "three units of 25, 70 and 160 µV on one channel", x, fs, t,
                    {"unit": unit})


def bursts():
    """Isolated spikes plus bursts with 3-6 ms intervals. Burst spikes are kept
    at least 3 ms from every other spike (the original generator did not
    check, and produced pairs 1 and 6 samples apart, which no neuron fires)."""
    fs, n = 30000, 60000
    x = white_noise(n, 5.0, 5)
    w, pk = ap_waveform(fs, 90.0)
    rng = np.random.default_rng(5)
    t = list(spaced_times(n, 10, 900, 5, margin=500))
    min_gap = int(0.003 * fs)
    for _ in range(4):
        for _attempt in range(100):
            start = int(rng.integers(500, n - 3000))
            burst = np.cumsum(np.concatenate([[start], rng.integers(int(0.003 * fs), int(0.006 * fs), 3)]))
            if all(abs(b - s) >= min_gap for b in burst for s in t):
                t.extend(int(b) for b in burst)
                break
    t = np.array(sorted(t), dtype=np.int64)
    for s in t:
        embed(x, s, w, pk)
    return Scenario("05_bursts", "isolated spikes and 4-spike bursts (3-6 ms intervals)", x, fs, t)


def overlaps():
    fs, n = 30000, 45000
    x = white_noise(n, 5.0, 6)
    w1, p1 = ap_waveform(fs, 90.0, 1.6)
    w2, p2 = ap_waveform(fs, 60.0, 1.2, asymmetry=0.55)
    u1 = spaced_times(n, 10, 900, 6, margin=500)
    off = np.random.default_rng(6).integers(int(0.0002 * fs), int(0.0008 * fs), len(u1) // 2)
    u2 = np.concatenate([u1[:len(off)] + off, spaced_times(n, 8, 900, 60, margin=500)])
    for s in u1:
        embed(x, s, w1, p1)
    for s in u2:
        embed(x, s, w2, p2)
    t = np.array(sorted(set(u1.tolist()) | set(u2.tolist())), dtype=np.int64)
    return Scenario("06_overlaps", "two units; half of unit 2 collides 0.2-0.8 ms after unit 1", x, fs, t,
                    {"colliding": u1[:len(off)] + off})


def artifacts():
    fs, n = 30000, 60000
    x = white_noise(n, 6.0, 7)
    w, pk = ap_waveform(fs, 70.0)
    t = spaced_times(n, 20, 600, 7)
    for s in t:
        embed(x, s, w, pk)
    rng = np.random.default_rng(7)
    centers = rng.integers(1000, n - 1000, 4)
    width = int(0.008 * fs)
    for c in centers:
        amp = rng.choice([1, -1]) * rng.uniform(300, 500)
        shape = amp * np.exp(-np.linspace(-1, 1, width) ** 2 / (2 * 0.5 ** 2))
        s0 = int(c - width // 2)
        a, b = max(0, s0), min(n, s0 + width)
        x[a:b] += shape[a - s0:b - s0]
    return Scenario("07_artifacts", "20 spikes plus 4 broad 300-500 µV artifacts of either polarity", x, fs, t,
                    {"artifacts": np.sort(centers)})


def line_noise():
    fs, n = 30000, 60000
    x = white_noise(n, 6.0, 8)
    tt = np.arange(n) / fs
    x += 35.0 * np.sin(2 * np.pi * 60 * tt) + 8.0 * np.sin(2 * np.pi * 180 * tt)
    w, pk = ap_waveform(fs, 90.0)
    t = spaced_times(n, 20, 600, 8)
    for s in t:
        embed(x, s, w, pk)
    return Scenario("08_line_noise", "60 Hz (35 µV) and 180 Hz (8 µV) hum under 90 µV spikes", x, fs, t)


def multichannel():
    fs, n, nch = 30000, 45000, 4
    V = np.zeros((nch, n))
    truth = []
    for ch in range(nch):
        V[ch] = white_noise(n, 6.0, 100 + ch)
        w, pk = ap_waveform(fs, 60.0 + 20.0 * ch)
        t = spaced_times(n, 15, 600, 100 + ch)
        for s in t:
            embed(V[ch], s, w, pk)
        truth.append(t)
    c = int(np.random.default_rng(9).integers(2000, n - 2000))
    width = int(0.006 * fs)
    shape = 400.0 * np.exp(-np.linspace(-1, 1, width) ** 2 / (2 * 0.5 ** 2))
    V[:, c - width // 2:c - width // 2 + width] += shape
    return Scenario("09_multichannel", "4 channels, independent units, one artifact on all channels", V, fs, truth,
                    {"shared_artifact": c})


def sampling_rate(fs):
    return _simple(f"10_fs_{int(fs / 1000)}k", f"same recipe at {fs / 1000:g} kHz", fs, 1.5, 6.0, 80.0, 15, 0.02, 10)


def streaming(chunk=1000):
    """Like the clean case, with a spike placed across a chunk boundary
    (the original generator computed the shift but never applied it)."""
    fs, n = 30000, 45000
    x = white_noise(n, 8.0, 11)
    w, pk = ap_waveform(fs, 75.0)
    straddle = 4 * chunk + 3          # trough 3 samples after a boundary, waveform starts before it
    t = spaced_times(n, 17, 600, 11, avoid=[straddle])
    t = np.array(sorted(list(t) + [straddle]), dtype=np.int64)
    for s in t:
        embed(x, s, w, pk)
    return Scenario("11_streaming", "consumed in 1000-sample chunks; one spike straddles a boundary", x, fs, t,
                    {"chunk": chunk, "straddle": straddle})


def edge_cases():
    fs = 30000
    w, pk = ap_waveform(fs, 90.0)
    out = []
    out.append(Scenario("12_empty", "zero samples", np.zeros(0), fs, np.zeros(0, int)))
    out.append(Scenario("12_all_noise", "noise only", white_noise(15000, 6.0, 120), fs, np.zeros(0, int)))
    x = white_noise(9000, 5.0, 121); embed(x, 2, w, pk)
    out.append(Scenario("12_edge_start", "spike truncated by the start of the recording", x, fs, np.array([2])))
    x = white_noise(9000, 5.0, 122); embed(x, 9000 - 3, w, pk)
    out.append(Scenario("12_edge_end", "spike truncated by the end of the recording", x, fs, np.array([8997])))
    x = white_noise(9000, 5.0, 123)
    t = spaced_times(9000, 6, 600, 123)
    for s in t:
        embed(x, s, w, pk)
    gaps = np.random.default_rng(124).choice(9000, 50, replace=False)
    x[gaps] = np.nan
    out.append(Scenario("12_nans", "6 spikes, 50 dropped (NaN) samples", x, fs, t, {"nan": np.sort(gaps)}))
    x = white_noise(9000, 5.0, 125)
    t = spaced_times(9000, 6, 600, 125)
    wb, pb = ap_waveform(fs, 200.0)
    for s in t:
        embed(x, s, wb, pb)
    out.append(Scenario("12_clipped", "200 µV spikes clipped at ±120 µV", np.clip(x, -120, 120), fs, t))
    return out


def all_single_channel():
    return [clean(), low_snr(), drift(), mixed_amplitudes(), bursts(), overlaps(), artifacts(), line_noise(),
            sampling_rate(20000), sampling_rate(30000), streaming(), *edge_cases()]
