import numpy as np
import pytest

from spikedet import DetectorConfig, detect, match
from spikedet import scenarios as S
from spikedet.events import accepted
from spikedet.streaming import StreamingDetector, detect_streaming

CFG = DetectorConfig(causal=True, noise_calibration_s=0.25, refractory_ms=1.5)
COLUMNS = ["sample", "amplitude_uv", "threshold_uv", "width_ms", "artifact", "edge",
           "dropped_sample", "refractory_duplicate", "accepted"]


def assert_same(a, b):
    assert len(a) == len(b)
    for c in COLUMNS:
        if a[c].dtype == float:
            assert np.allclose(a[c], b[c], rtol=1e-9, atol=1e-9), c
        else:
            assert a[c].tolist() == b[c].tolist(), c


@pytest.mark.parametrize("chunk", [1, 37, 1000, 4096, 10 ** 6])
def test_streaming_reproduces_offline_causal_detection_for_any_chunk_size(chunk):
    sc = S.streaming()
    x = sc.voltage if chunk > 1 else sc.voltage[:12000]
    assert_same(detect_streaming(x, sc.fs, chunk, CFG), detect(x, sc.fs, config=CFG))


@pytest.mark.parametrize("make", [S.artifacts, S.bursts, lambda: S.edge_cases()[4], lambda: S.edge_cases()[3]])
def test_equivalence_holds_on_artifacts_bursts_gaps_and_edges(make):
    sc = make()
    assert_same(detect_streaming(sc.voltage, sc.fs, 777, CFG), detect(sc.voltage, sc.fs, config=CFG))


def test_spike_across_a_chunk_boundary_is_found_once():
    sc = S.streaming()
    ev = accepted(detect_streaming(sc.voltage, sc.fs, sc.extra["chunk"], CFG))
    near = ev[np.abs(ev["sample"] - sc.extra["straddle"]) <= 30]
    assert len(near) == 1
    assert match(ev["sample"], sc.truth, 30).recall == 1


def test_events_are_emitted_with_bounded_latency():
    sc = S.streaming()
    det = StreamingDetector(sc.fs, CFG)
    lag = []
    for i in range(0, len(sc.voltage), 500):
        out = det.process(sc.voltage[i:i + 500])
        end = min(i + 500, len(sc.voltage))
        lag += (end - out["sample"]).tolist()
    assert max(lag) <= det.context + 500 + int(0.25 * sc.fs)
    assert det.latency_s < 0.05


def test_memory_stays_bounded():
    sc = S.streaming()
    det = StreamingDetector(sc.fs, CFG)
    for i in range(0, len(sc.voltage), 1000):
        det.process(sc.voltage[i:i + 1000])
    assert len(det.f) <= det.context + 1000 + int(0.25 * sc.fs)


def test_short_input_flushes_cleanly():
    det = StreamingDetector(30000, CFG)
    assert det.process(np.zeros(10)).empty
    assert det.flush().empty


def test_rolling_noise_windows_are_rejected():
    with pytest.raises(ValueError):
        StreamingDetector(30000, DetectorConfig(noise_window_s=1.0))
