import numpy as np
import pandas as pd
import pytest

from spikedet import DetectorConfig, detect, detect_multichannel, match
from spikedet import scenarios as S
from spikedet.events import SCHEMA, accepted, empty_events

FS = 30000


def recording(fs=FS, dur=1.0, amp=80.0, noise=4.0, times_s=(0.1, 0.35, 0.6, 0.85), seed=0):
    n = int(fs * dur)
    x = np.random.default_rng(seed).normal(0, noise, n)
    w, pk = S.ap_waveform(fs, amp)
    truth = np.array([int(round(t * fs)) for t in times_s])
    for s in truth:
        S.embed(x, s, w, pk)
    return x, truth


# ---------------------------------------------------------------- output schema
@pytest.mark.parametrize("x", [np.zeros(0), np.full(3000, np.nan), np.zeros(5),
                               np.random.default_rng(0).normal(0, 1, 30000)])
def test_schema_is_identical_for_empty_degenerate_and_noise_inputs(x):
    ev = detect(x, FS)
    assert list(ev.columns) == list(SCHEMA)
    assert ev.dtypes.astype(str).to_dict() == SCHEMA


def test_accepted_on_an_empty_table_keeps_columns():
    out = accepted(empty_events())
    assert len(out) == 0 and list(out.columns) == list(SCHEMA)


def test_fs_or_timestamps_required():
    with pytest.raises(ValueError):
        detect(np.zeros(100))


def test_timestamps_are_used_for_event_times():
    x, truth = recording()
    t = 373.4 + np.arange(len(x)) / FS
    ev = accepted(detect(x, time_s=t))
    assert np.allclose(ev.time_s, t[ev["sample"]])


# ---------------------------------------------------------------- invariances
def test_same_events_in_seconds_at_20_and_30_khz():
    out = []
    for fs in (20000, 30000):
        x, truth = recording(fs=fs)
        ev = accepted(detect(x, fs))
        assert match(ev["sample"], truth, tol=1).tp == len(truth)
        out.append(ev.time_s.to_numpy())
    assert np.allclose(out[0], out[1], atol=1 / 20000)


def test_scaling_the_signal_does_not_change_the_events():
    x, _ = recording()
    a, b = detect(x, FS), detect(37.0 * x, FS)
    assert a["sample"].tolist() == b["sample"].tolist()
    assert np.allclose(37.0 * a.amplitude_uv, b.amplitude_uv)


def test_shifting_the_recording_shifts_the_events():
    x, _ = recording()
    a = accepted(detect(x, FS))
    b = accepted(detect(np.concatenate([np.random.default_rng(9).normal(0, 4, 3000), x]), FS))
    assert (b["sample"].to_numpy() - 3000).tolist() == a["sample"].tolist()


def test_trough_index_is_the_minimum_of_the_filtered_signal():
    from spikedet.detector import preprocess
    x, truth = recording()
    ev = accepted(detect(x, FS))
    _, f, _, _ = preprocess(x, FS, DetectorConfig())
    for s in ev["sample"]:
        assert f[s] == f[s - 3:s + 4].min()
    assert np.all(np.abs(ev.subsample_offset) <= 0.5)


# ---------------------------------------------------------------- flags
def test_refractory_keeps_the_deeper_of_two_close_troughs():
    x, _ = recording(times_s=(0.5,))
    w, pk = S.ap_waveform(FS, 160.0)
    S.embed(x, int(0.5 * FS) + 15, w, pk)           # a deeper trough 0.5 ms later
    ev = detect(x, FS, config=DetectorConfig(align_window_ms=0.3))
    acc = accepted(ev)
    assert len(acc) == 1 and acc.amplitude_uv.iloc[0] == ev.amplitude_uv.min()


def test_broad_artifacts_of_either_polarity_are_flagged_not_dropped():
    sc = S.artifacts()
    ev = detect(sc.voltage, sc.fs)
    leaked = 0
    for c in sc.extra["artifacts"]:
        near = ev[np.abs(ev["sample"] - c) <= 0.006 * sc.fs]
        assert near.artifact.any()                    # flagged and kept in the table
        leaked += int((~near.artifact).sum())
    assert leaked <= 1                                # one filter lobe at an artifact edge still passes


def test_spike_right_after_an_artifact_is_kept():
    sc = S.artifacts()
    ev = accepted(detect(sc.voltage, sc.fs))
    assert match(ev["sample"], sc.truth, 30).fn == 0


def test_truncated_edge_spikes_are_kept_and_flagged():
    for sc in S.edge_cases()[2:4]:
        ev = accepted(detect(sc.voltage, sc.fs))
        m = match(ev["sample"], sc.truth, 30)
        assert m.tp == 1 and ev.iloc[m.matched_detected].edge.all()


def test_events_on_dropped_samples_are_rejected():
    x, _ = recording()
    from spikedet.detector import preprocess
    ev = accepted(detect(x, FS))
    s = int(ev["sample"].iloc[0])
    y = x.copy()
    y[s - 1:s + 2] = np.nan
    ev2 = detect(y, FS)
    assert ev2.loc[np.abs(ev2["sample"] - s) <= 2, "dropped_sample"].any()


def test_clipped_spikes_are_detected_and_flagged():
    sc = S.edge_cases()[5]
    ev = accepted(detect(sc.voltage, sc.fs))
    assert match(ev["sample"], sc.truth, 30).tp == 6 and ev.clipped.sum() == 6


def test_shared_artifact_is_flagged_on_every_channel_and_spikes_are_not():
    sc = S.multichannel()
    ev = detect_multichannel(sc.voltage, sc.fs)
    c = sc.extra["shared_artifact"]
    near = ev[np.abs(ev["sample"] - c) <= 0.005 * sc.fs]
    assert set(near.channel) == {0, 1, 2, 3} and near.common_artifact.all()
    for ch, truth in enumerate(sc.truth):
        acc = accepted(ev[ev.channel == ch])
        assert match(acc["sample"], truth, 30).tp == len(truth)
    assert not ev[~ev.common_artifact & ev.accepted].empty


def test_absolute_ceiling_is_optional():
    x, _ = recording(amp=500.0)
    assert accepted(detect(x, FS)).shape[0] == 4
    assert accepted(detect(x, FS, config=DetectorConfig(max_amplitude_uv=200))).shape[0] == 0
