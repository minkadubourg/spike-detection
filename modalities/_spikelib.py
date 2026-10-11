"""
Waveform and noise helpers for modalities/generate.py (written with an AI
assistant, as was the generator).

Units are arbitrary but chosen to feel like real extracellular recordings:
  - voltage in "uV" (microvolts) -- baseline noise ~ a few uV, spikes ~ tens to
    low hundreds of uV, matching typical extracellular action potential sizes.
  - fs in Hz (sampling rate), typically 20-30 kHz for spike-band data.

Every dataset is saved as a .npz with at least:
  - voltage           : float32 array (or list of raw arrays for multichannel)
  - fs                 : sampling rate (Hz)
  - true_spike_indices : int array of ground-truth spike sample indices (rising
                          edge / negative-peak alignment -- see waveform doc below)
  - description        : short string describing the case and what it's testing
"""

import numpy as np

RNG_GLOBAL_NOTE = "Each dataset uses its own fixed seed for reproducibility."


def make_ap_waveform(fs, amp=60.0, dur_ms=1.6, asymmetry=0.35):
    """
    A biphasic extracellular action-potential-like waveform:
    sharp negative deflection followed by a smaller, slower positive rebound.

    Returns a 1D array. The array's own biggest (negative) deflection sits at
    the index returned as `peak_idx`, which is what "true_spike_indices"
    refers to when this waveform is embedded.
    """
    n = int(round(dur_ms / 1000 * fs))
    if n < 5:
        n = 5
    t = np.linspace(-1, 1, n)
    neg = -np.exp(-((t + 0.15) ** 2) / (2 * 0.12 ** 2))
    pos = asymmetry * np.exp(-((t - 0.35) ** 2) / (2 * 0.25 ** 2))
    w = neg + pos
    w = w / np.max(np.abs(w))  # normalize so the negative peak has magnitude 1
    w = amp * w
    peak_idx = int(np.argmin(w))  # index of the negative peak within the waveform
    return w.astype(np.float32), peak_idx


def white_noise(n_samples, std, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, std, n_samples).astype(np.float32)


def random_walk_drift(n_samples, step_std, seed, smooth_win=201):
    """Slow baseline wander via a smoothed random walk."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, step_std, n_samples))
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        walk = np.convolve(walk, kernel, mode="same")
    return walk.astype(np.float32)


def line_noise(n_samples, fs, freq, amp, seed, phase_jitter=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    phase = rng.uniform(0, 2 * np.pi) if phase_jitter else 0.0
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def embed_spike(trace, center_idx, waveform, peak_idx):
    """
    Add `waveform` into `trace` such that waveform's peak_idx lands at
    center_idx in the trace. Handles truncation at the array boundaries
    (useful for deliberately testing edge-of-array spikes).
    """
    n = len(trace)
    start = center_idx - peak_idx
    end = start + len(waveform)
    w_start = 0
    w_end = len(waveform)
    if start < 0:
        w_start = -start
        start = 0
    if end > n:
        w_end = len(waveform) - (end - n)
        end = n
    if w_end <= w_start or start >= end:
        return  # nothing left to embed (fully off-trace)
    trace[start:end] += waveform[w_start:w_end]


def min_isi_indices(n_samples, n_spikes, min_gap, seed, margin=200):
    """Random-ish but well-separated spike center indices."""
    rng = np.random.default_rng(seed)
    indices = []
    attempts = 0
    lo, hi = margin, n_samples - margin
    while len(indices) < n_spikes and attempts < n_spikes * 200:
        cand = rng.integers(lo, hi)
        if all(abs(cand - x) >= min_gap for x in indices):
            indices.append(int(cand))
        attempts += 1
    return np.array(sorted(indices), dtype=np.int64)
