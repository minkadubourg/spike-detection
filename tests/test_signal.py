import numpy as np
import pytest

from spikedet import signal as sg


def test_infer_fs_is_robust_to_jitter_and_a_gap():
    t = np.arange(10000) / 19608.0
    t = t + np.random.default_rng(0).normal(0, 1e-7, t.size)
    t[5000:] += 0.01                                  # one dropped stretch
    assert sg.infer_fs(t) == pytest.approx(19608, rel=1e-3)


def test_infer_fs_needs_two_samples():
    with pytest.raises(ValueError):
        sg.infer_fs([0.0])


def test_offline_gap_repair_interpolates_and_reports_the_mask():
    x = np.array([0.0, np.nan, np.nan, 3.0])
    out, nan = sg.repair_gaps(x)
    assert out.tolist() == [0, 1, 2, 3] and nan.tolist() == [False, True, True, False]


def test_causal_gap_repair_holds_the_last_value_across_blocks():
    out, _ = sg.repair_gaps(np.array([np.nan, 1.0, np.nan]), causal=True, last_valid=5.0)
    assert out.tolist() == [5.0, 1.0, 1.0]


def test_all_nan_gives_zeros():
    out, nan = sg.repair_gaps(np.full(5, np.nan))
    assert np.all(out == 0) and nan.all()


def test_mad_sigma_estimates_gaussian_sigma_and_ignores_outliers():
    x = np.random.default_rng(1).normal(0, 7.0, 200000)
    x[:1000] = 1e4                                   # 0.5 % huge outliers
    assert sg.mad_sigma(x) == pytest.approx(7.0, rel=0.02)
    assert np.std(x) > 100                           # what a std-based threshold would use


def test_rolling_threshold_follows_a_noise_step():
    rng = np.random.default_rng(2)
    fs = 1000
    x = np.concatenate([rng.normal(0, 1, 5000), rng.normal(0, 5, 5000)])
    thr = sg.threshold_trace(x, fs, k=4, window_s=1.0)
    assert thr[1000] == pytest.approx(-4, rel=0.1)
    assert thr[9000] == pytest.approx(-20, rel=0.1)
    glob = sg.threshold_trace(x, fs, k=4)
    assert np.unique(glob).size == 1


def test_calibration_threshold_uses_only_the_first_seconds():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 10, 9000)])
    thr = sg.threshold_trace(x, 1000, k=4, calibration_s=1.0)
    assert np.allclose(thr, thr[0]) and thr[0] == pytest.approx(-4, rel=0.15)


def test_rolling_threshold_merges_a_short_last_window():
    x = np.random.default_rng(4).normal(0, 1, 2300)
    thr = sg.threshold_trace(x, 1000, k=4, window_s=1.0)
    assert thr[1000] == thr[2299]                    # 300-sample tail merged with the second window


def test_crossings_include_a_recording_that_starts_below_threshold():
    x = np.array([-5.0, -1, 0, -5, -5, 0])
    assert sg.crossings(x, -2).tolist() == [0, 3]
    assert sg.crossings(x, -2, starts_above=False).tolist() == [3]


def test_alignment_returns_the_true_trough_and_a_subsample_offset():
    # parabola with its minimum at 50.3
    i = np.arange(100.0)
    x = 0.1 * (i - 50.3) ** 2 - 40
    troughs, offsets = sg.align_to_trough(x, np.array([45]), window=20)
    assert troughs[0] == 50
    assert offsets[0] == pytest.approx(0.3, abs=1e-9)


def test_alignment_searches_up_to_the_last_sample():
    x = np.zeros(10)
    x[-1] = -5
    troughs, _ = sg.align_to_trough(x, np.array([8]), window=5)
    assert troughs[0] == 9


def test_width_is_bounded_by_the_cap():
    x = np.full(1000, -10.0)
    assert sg.width_below(x, np.array([500]), -5, cap=20)[0] == 41


def test_causal_and_zero_phase_filters_have_the_same_magnitude_response():
    fs = 30000
    x = np.random.default_rng(5).normal(0, 1, 2 ** 16)
    a, b = sg.bandpass(x, fs, causal=False), sg.bandpass(x, fs, causal=True)
    # zero-phase applies the filter twice: power ratio squared, so compare in-band power loosely
    assert np.std(a) == pytest.approx(np.std(b), rel=0.2)


def test_band_is_capped_below_nyquist():
    sos = sg.design_bandpass(8000, (300, 5000), 4)  # 5 kHz > Nyquist of 4 kHz
    assert np.isfinite(sos).all()
