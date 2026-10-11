"""Causal, chunk-by-chunk spike detection.

The detector keeps the filter state between chunks (`sosfilt` with `zi`), so
filtering a recording in chunks gives exactly the same samples as filtering it
in one go. Event flags depend only on a bounded neighbourhood of each event
(`detector.context_samples`), so an event is emitted once that much signal has
arrived after it: the output is identical to the offline detector run in causal
mode, whatever the chunk size, with a fixed latency.

The noise level is estimated once, on the first `noise_calibration_s` seconds;
nothing is emitted before that.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.signal import sosfilt

from . import signal as sig
from .config import DetectorConfig
from .detector import context_samples, events_from_block
from .events import empty_events


class StreamingDetector:
    def __init__(self, fs: float, config: DetectorConfig | None = None, channel: int = 0):
        cfg = config or DetectorConfig()
        if cfg.noise_calibration_s is None:
            cfg = replace(cfg, noise_calibration_s=1.0)
        if cfg.noise_window_s is not None:
            raise ValueError("rolling noise windows are offline-only; use noise_calibration_s")
        self.cfg = replace(cfg, causal=True)
        self.fs, self.channel = fs, channel
        self.sos = sig.design_bandpass(fs, cfg.band_hz, cfg.filter_order)
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.last_valid = 0.0
        self.context = context_samples(self.cfg, fs)
        self.calibration = max(2, int(round(self.cfg.noise_calibration_s * fs)))
        self.threshold = None
        self.n_seen = 0
        self.buf_start = 0
        self.emitted_until = 0
        self.raw = np.zeros(0)
        self.f = np.zeros(0)
        self.nan = np.zeros(0, bool)

    @property
    def latency_s(self) -> float:
        return self.context / self.fs

    def process(self, chunk) -> pd.DataFrame:
        """Feed the next block of raw samples; returns the events that became final."""
        chunk = np.asarray(chunk, float)
        if len(chunk) == 0:
            return empty_events()
        rep, nan = sig.repair_gaps(chunk, causal=True, last_valid=self.last_valid)
        valid = np.flatnonzero(~nan)
        if len(valid):
            self.last_valid = chunk[valid[-1]]
        filt, self.zi = sosfilt(self.sos, rep, zi=self.zi)
        self.raw = np.concatenate([self.raw, rep])
        self.f = np.concatenate([self.f, filt])
        self.nan = np.concatenate([self.nan, nan])
        self.n_seen += len(chunk)

        if self.threshold is None:
            if self.n_seen < self.calibration:
                return empty_events()
            self.threshold = float(sig.threshold_trace(self.f[:self.calibration], self.fs, self.cfg.k)[0])
        return self._emit(self.n_seen - self.context, final=False)

    def flush(self) -> pd.DataFrame:
        """End of recording: emit everything left, with end-of-recording edge flags."""
        if self.threshold is None:
            if self.n_seen < 2:
                return empty_events()
            self.threshold = float(sig.threshold_trace(self.f[:self.calibration], self.fs, self.cfg.k)[0])
        return self._emit(self.n_seen, final=True)

    def _emit(self, until: int, final: bool) -> pd.DataFrame:
        if until <= self.emitted_until:
            return empty_events()
        ev = events_from_block(self.f, self.threshold, self.fs, self.cfg, raw=self.raw, nan=self.nan,
                               base=self.buf_start, n_total=self.n_seen if final else None,
                               starts_above=self.buf_start == 0, channel=self.channel)
        ev = ev[(ev["sample"] >= self.emitted_until) & (ev["sample"] < until)].reset_index(drop=True)
        self.emitted_until = until
        # drop what no future event can depend on
        keep_from = max(self.buf_start, until - self.context)
        cut = keep_from - self.buf_start
        self.raw, self.f, self.nan = self.raw[cut:], self.f[cut:], self.nan[cut:]
        self.buf_start = keep_from
        return ev


def detect_streaming(voltage, fs, chunk: int, config: DetectorConfig | None = None) -> pd.DataFrame:
    """Run the streaming detector over a recording in fixed-size chunks."""
    det = StreamingDetector(fs, config)
    parts = [det.process(voltage[i:i + chunk]) for i in range(0, len(voltage), chunk)]
    parts.append(det.flush())
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else empty_events()
