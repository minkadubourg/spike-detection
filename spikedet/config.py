"""Detector configuration. Every duration is in milliseconds or seconds, never in
samples, so one configuration works at any sampling rate."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DetectorConfig:
    # band-pass (spike band)
    band_hz: tuple[float, float] = (300.0, 3500.0)
    filter_order: int = 4
    causal: bool = False                 # False: zero-phase (offline); True: causal (streaming-compatible)

    # threshold: median - k * sigma, sigma = MAD / 0.6745
    k: float = 4.5
    noise_window_s: float | None = None  # per-window noise estimate (follows slow noise changes)
    noise_calibration_s: float | None = None  # estimate noise on the first seconds only (streaming-compatible)

    # event alignment
    align_window_ms: float = 1.0         # search for the trough this long after each crossing

    # quality flags
    refractory_ms: float = 1.0           # within this, keep the deeper trough
    ringing_ms: float = 3.0              # within this, drop troughs ringing_ratio times smaller than a
    ringing_ratio: float = 10.0          # neighbour (band-pass ringing around very large spikes)
    artifact_window_ms: float = 1.0      # polarity test: positive peak bigger than the trough in ±window
    polarity_test: bool | None = None    # None: on for zero-phase filtering, off for causal (a causal
                                         # band-pass gives spikes a large positive overshoot)
    max_width_ms: float = 1.0            # time continuously below threshold; wider is not an action potential
    excursion_context_ms: float = 5.0    # slow-excursion test: look this far around the event in the raw signal
    excursion_baseline_ms: float = 20.0  # raw signal minus its rolling median over this window
    excursion_ratio: float = 3.0         # artifact if a broad raw excursion is this many times the event
    max_amplitude_uv: float | None = None  # optional absolute ceiling (e.g. 1000 µV)
    clip_level_uv: float | None = None   # amplifier rail; events touching it are flagged

    def to_dict(self) -> dict:
        return asdict(self)
