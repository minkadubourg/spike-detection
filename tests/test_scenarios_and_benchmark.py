"""The benchmark as a regression test: each failure mode has an expected
performance, and known limitations are written down rather than hidden."""
import numpy as np
import pytest

from spikedet import scenarios as S
from spikedet.benchmark import PROFILES, all_scenarios, run_scenario

# (min recall, max false positives) for the default offline profile
OFFLINE = {
    "01_clean": (1.0, 0), "02_low_snr": (0.75, 1),   # peak SNR 2.7: misses are expected at k = 4.5
    "03_drift": (1.0, 1), "04_mixed_amplitudes": (1.0, 1), "05_bursts": (1.0, 0),
    "06_overlaps": (0.7, 1),                          # limitation: collisions need template matching
    "07_artifacts": (1.0, 1), "08_line_noise": (1.0, 0), "09_multichannel": (1.0, 0),
    "10_fs_20k": (1.0, 1), "10_fs_30k": (1.0, 0), "11_streaming": (1.0, 0),
    "12_empty": (None, 0), "12_all_noise": (None, 0), "12_edge_start": (1.0, 0), "12_edge_end": (1.0, 0),
    "12_nans": (1.0, 0), "12_clipped": (1.0, 0),
}
STREAMING = dict(OFFLINE, **{
    "02_low_snr": (0.1, 1),            # limitation: a causal band-pass halves the trough-to-noise ratio
    "04_mixed_amplitudes": (0.9, 1),   # same cause, on the 25 µV unit
})


def _check(row, expected):
    min_recall, max_fp = expected[row["scenario"]]
    assert row["fp"] <= max_fp, row
    if min_recall is not None:
        assert row["recall"] >= min_recall, row


@pytest.mark.parametrize("sc", all_scenarios(), ids=lambda s: s.name)
def test_offline_benchmark(sc):
    _check(run_scenario(sc, **PROFILES["offline (zero-phase)"]), OFFLINE)


@pytest.mark.parametrize("sc", [s for s in all_scenarios() if np.ndim(s.voltage) == 1], ids=lambda s: s.name)
def test_streaming_benchmark(sc):
    _check(run_scenario(sc, **PROFILES["streaming (causal)"]), STREAMING)


def test_every_artifact_is_flagged():
    for sc in (S.artifacts(), S.multichannel()):
        row = run_scenario(sc, **PROFILES["offline (zero-phase)"])
        assert row["artifacts_flagged"] == row["artifacts_total"]


# ---------------------------------------------------------------- scenario sanity
def test_scenarios_are_deterministic():
    a, b = S.low_snr(), S.low_snr()
    assert np.array_equal(a.voltage, b.voltage) and np.array_equal(a.truth, b.truth)


def test_ground_truth_respects_a_refractory_period():
    """Regression: the first version of the bursts generator placed true spikes
    1 and 6 samples apart."""
    for sc in all_scenarios():
        for t in (sc.truth if isinstance(sc.truth, list) else [sc.truth]):
            if sc.name != "06_overlaps" and len(t) > 1:
                assert np.diff(np.sort(t)).min() >= 0.003 * sc.fs, sc.name


def test_streaming_scenario_has_a_spike_across_a_chunk_boundary():
    """Regression: the first version computed the shift but never applied it."""
    sc = S.streaming()
    w, pk = S.ap_waveform(sc.fs, 75.0)
    start, end = sc.extra["straddle"] - pk, sc.extra["straddle"] - pk + len(w)
    assert start // sc.extra["chunk"] != (end - 1) // sc.extra["chunk"]


def test_ground_truth_lies_inside_the_recording():
    for sc in all_scenarios():
        n = np.shape(sc.voltage)[-1]
        for t in (sc.truth if isinstance(sc.truth, list) else [sc.truth]):
            assert np.all((t >= 0) & (t < max(n, 1)))
