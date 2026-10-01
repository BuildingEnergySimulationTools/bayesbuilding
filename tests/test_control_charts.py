import numpy as np
import pandas as pd
import pytest

from bayesbuilding.control_charts import (
    _contiguous_true_runs,
    cusum_control_stats,
    ewma_control_stats,
    flatten_predictive_samples,
    plot_ewma_chart,
    standardized_residuals,
    xbar_control_stats,
)


def test_flatten_predictive_samples_collapses_chain_and_draw():
    arr_3d = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (chain, draw, time)
    flat = flatten_predictive_samples(arr_3d)
    assert flat.shape == (6, 4)
    np.testing.assert_array_equal(flat, arr_3d.reshape(6, 4))

    arr_2d = np.arange(5 * 4).reshape(5, 4)  # already (samples, time)
    flat_2d = flatten_predictive_samples(arr_2d)
    assert flat_2d.shape == (5, 4)
    np.testing.assert_array_equal(flat_2d, arr_2d)


def test_xbar_control_stats_residual_and_out_of_control_flag():
    mu_draws = np.tile([100.0, 200.0, 300.0], (4, 1))  # 4 draws, same mu each day
    sigma_draws = np.full((4, 3), 2.0)
    y_true = np.array([100.0, 210.0, 300.0])  # day 1: residual=10, others: 0

    stats = xbar_control_stats(y_true, mu_draws, sigma_draws, L=1.96)

    np.testing.assert_allclose(stats["mu_hat"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(stats["sigma_hat"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(stats["residual"], [0.0, 10.0, 0.0])
    np.testing.assert_allclose(stats["ucl"], [1.96 * 2.0] * 3)
    np.testing.assert_allclose(stats["lcl"], [-1.96 * 2.0] * 3)
    np.testing.assert_array_equal(stats["out_of_control"], [False, True, False])


def test_xbar_control_stats_limits_follow_heteroscedastic_sigma():
    # A model with a heteroscedastic noise term (e.g. heating_cp_occ_rad)
    # should produce limits that widen/narrow with it, not a single flat
    # sigma pooled over the whole period.
    n_draws, n_days = 50, 3
    mu_draws = np.zeros((n_draws, n_days))
    sigma_draws = np.tile([1.0, 5.0, 1.0], (n_draws, 1))
    y_true = np.zeros(n_days)

    stats = xbar_control_stats(y_true, mu_draws, sigma_draws)

    assert stats["ucl"][1] > stats["ucl"][0]
    assert stats["ucl"][1] > stats["ucl"][2]
    np.testing.assert_allclose(stats["ucl"][0], stats["ucl"][2])


def test_ewma_control_stats_matches_manual_recursion():
    mu_draws = np.zeros((1, 3))
    sigma_draws = np.ones((1, 3))
    y_true = np.array([2.0, 0.0, 0.0])  # residual = [2, 0, 0], sigma_hat = 1
    lam = 0.5

    stats = ewma_control_stats(y_true, mu_draws, sigma_draws, lam=lam, L=1.0)

    z0 = lam * 2.0
    var0 = lam**2 * 1.0**2
    z1 = lam * 0.0 + (1 - lam) * z0
    var1 = lam**2 * 1.0**2 + (1 - lam) ** 2 * var0
    z2 = lam * 0.0 + (1 - lam) * z1
    var2 = lam**2 * 1.0**2 + (1 - lam) ** 2 * var1

    np.testing.assert_allclose(stats["ewma"], [z0, z1, z2])
    np.testing.assert_allclose(stats["ucl"], np.sqrt([var0, var1, var2]))
    np.testing.assert_allclose(stats["lcl"], -np.sqrt([var0, var1, var2]))


def test_ewma_control_stats_limits_widen_then_plateau():
    # Classic EWMA control-limit shape: tight near the first sample, widening
    # towards the constant-sigma asymptote L*sigma*sqrt(lam/(2-lam)).
    n_days = 200
    mu_draws = np.zeros((10, n_days))
    sigma_draws = np.full((10, n_days), 3.0)
    y_true = np.zeros(n_days)
    lam, L = 0.2, 1.96

    stats = ewma_control_stats(y_true, mu_draws, sigma_draws, lam=lam, L=L)

    asymptotic_ucl = L * 3.0 * np.sqrt(lam / (2 - lam))
    assert stats["ucl"][0] < stats["ucl"][-1]
    assert stats["ucl"][-1] == pytest.approx(asymptotic_ucl, rel=1e-3)


def test_ewma_control_stats_default_alarm_matches_out_of_control():
    # n_reset=None (default) must reproduce the old, memory-less behaviour
    # exactly: alarm == the instantaneous out_of_control test, no resets ever.
    mu_draws = np.zeros((1, 5))
    sigma_draws = np.ones((1, 5))
    y_true = np.array([0.0, 3.0, 0.0, -3.0, 0.0])

    stats = ewma_control_stats(y_true, mu_draws, sigma_draws, lam=0.5, L=1.0)

    np.testing.assert_array_equal(stats["alarm"], stats["out_of_control"])
    assert not stats["reset_points"].any()


def test_ewma_control_stats_confirms_alarm_then_resets_after_transient_event():
    # residual jumps to 5 for 3 steps (enough to confirm an alarm after
    # n_confirm=2 consecutive out-of-limit EWMA steps), then drops back to 0
    # for good -- enough consecutive in-control-error steps (n_reset=2) to
    # confirm the event has genuinely cleared and reset the EWMA to 0.
    mu_draws = np.zeros((1, 8))
    sigma_draws = np.ones((1, 8))
    y_true = np.array([0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    lam, L = 0.5, 1.0

    stats = ewma_control_stats(
        y_true,
        mu_draws,
        sigma_draws,
        lam=lam,
        L=L,
        n_confirm=2,
        n_reset=2,
        reset_epsilon=0.5,
    )

    expected_alarm = [False, False, True, True, True, False, False, False]
    np.testing.assert_array_equal(stats["alarm"], expected_alarm)
    np.testing.assert_array_equal(stats["out_of_control"], expected_alarm)

    expected_reset = [False, False, False, False, False, True, False, False]
    np.testing.assert_array_equal(stats["reset_points"], expected_reset)

    # at the reset step, the EWMA and its variance recursion restart exactly
    # as they would at a fresh first point (z <- 0, var <- lam**2*sigma**2).
    assert stats["ewma"][5] == 0.0
    assert stats["ucl"][5] == pytest.approx(lam * L)


def test_ewma_control_stats_persistent_drift_never_resets():
    # same alarm-triggering ramp, but the residual stays elevated forever
    # afterwards -- a real sustained drift must never get reset away.
    mu_draws = np.zeros((1, 8))
    sigma_draws = np.ones((1, 8))
    y_true = np.array([0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    stats = ewma_control_stats(
        y_true,
        mu_draws,
        sigma_draws,
        lam=0.5,
        L=1.0,
        n_confirm=2,
        n_reset=2,
        reset_epsilon=0.5,
    )

    assert stats["alarm"][2:].all()
    assert not stats["reset_points"].any()


def test_ewma_control_stats_short_excess_never_confirms_alarm():
    # n_confirm larger than the whole series: no run of consecutive
    # out-of-limit EWMA steps can ever reach it, so the alarm never confirms
    # regardless of the spike -- a short blip shouldn't be over-eagerly
    # flagged as a confirmed alarm.
    mu_draws = np.zeros((1, 5))
    sigma_draws = np.ones((1, 5))
    y_true = np.array([0.0, 5.0, 0.0, 0.0, 0.0])

    stats = ewma_control_stats(
        y_true,
        mu_draws,
        sigma_draws,
        lam=0.5,
        L=1.0,
        n_confirm=10,
        n_reset=2,
        reset_epsilon=0.5,
    )

    assert not stats["alarm"].any()
    assert not stats["out_of_control"].any()
    assert not stats["reset_points"].any()


def test_cusum_control_stats_accumulates_a_sustained_upward_drift():
    # z = residual/sigma_hat = 1.0 every day, k=0.5 -> cusum_pos grows by 0.5
    # each day, cusum_neg stays clamped at 0 (never goes negative).
    mu_draws = np.zeros((1, 4))
    sigma_draws = np.ones((1, 4))
    y_true = np.array([1.0, 1.0, 1.0, 1.0])

    stats = cusum_control_stats(y_true, mu_draws, sigma_draws, k=0.5, h=5.0)

    np.testing.assert_allclose(stats["cusum_pos"], [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(stats["cusum_neg"], [0.0, 0.0, 0.0, 0.0])
    assert not stats["out_of_control"].any()


def test_cusum_control_stats_accumulates_a_sustained_downward_drift():
    mu_draws = np.zeros((1, 4))
    sigma_draws = np.ones((1, 4))
    y_true = np.array([-1.0, -1.0, -1.0, -1.0])

    stats = cusum_control_stats(y_true, mu_draws, sigma_draws, k=0.5, h=5.0)

    np.testing.assert_allclose(stats["cusum_pos"], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(stats["cusum_neg"], [0.5, 1.0, 1.5, 2.0])
    assert not stats["out_of_control"].any()


def test_cusum_control_stats_flags_out_of_control_once_h_is_exceeded():
    # z=2.0, k=0.5 -> cusum_pos grows by 1.5/day: 1.5, 3.0, 4.5, 6.0
    # -> crosses h=5 on day 4.
    mu_draws = np.zeros((1, 4))
    sigma_draws = np.ones((1, 4))
    y_true = np.array([2.0, 2.0, 2.0, 2.0])

    stats = cusum_control_stats(y_true, mu_draws, sigma_draws, k=0.5, h=5.0)

    np.testing.assert_array_equal(stats["out_of_control"], [False, False, False, True])


def test_standardized_residuals_matches_residual_over_sigma_hat():
    mu_draws = np.tile([100.0, 200.0, 300.0], (4, 1))  # 4 draws, same mu each day
    sigma_draws = np.full((4, 3), 2.0)
    y_true = np.array([100.0, 210.0, 294.0])  # residual = [0, 10, -6]

    z = standardized_residuals(y_true, mu_draws, sigma_draws)

    np.testing.assert_allclose(z, [0.0, 5.0, -3.0])


def test_standardized_residuals_follows_heteroscedastic_sigma():
    # Same absolute residual, but a day with a wider sigma should read as a
    # smaller z -- sigma_hat isn't pooled into a single flat scale.
    mu_draws = np.zeros((10, 2))
    sigma_draws = np.tile([1.0, 5.0], (10, 1))
    y_true = np.array([2.0, 2.0])

    z = standardized_residuals(y_true, mu_draws, sigma_draws)

    assert z[0] == pytest.approx(2.0)
    assert z[1] == pytest.approx(0.4)


def test_contiguous_true_runs_finds_each_block():
    mask = np.array([False, True, True, False, False, True, False])
    assert _contiguous_true_runs(mask) == [(1, 2), (5, 5)]


def test_plot_ewma_chart_without_reset_args_has_no_alarm_shading_or_markers():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    measure = pd.Series([0.0, 5.0, 5.0, 5.0, 0.0], index=index)
    mu_draws = np.zeros((1, 5))
    sigma_draws = np.ones((1, 5))

    fig = plot_ewma_chart(measure, mu_draws, sigma_draws, lam=0.5, L=1.0)

    assert len(fig.layout.shapes) == 1  # just the y=0 center line, no alarm vrect
    assert not any(trace.name == "reset EWMA" for trace in fig.data)


def test_plot_ewma_chart_with_reset_args_shades_alarm_and_marks_resets():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    measure = pd.Series([0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0], index=index)
    mu_draws = np.zeros((1, 8))
    sigma_draws = np.ones((1, 8))

    fig = plot_ewma_chart(
        measure,
        mu_draws,
        sigma_draws,
        lam=0.5,
        L=1.0,
        n_confirm=2,
        n_reset=2,
        reset_epsilon=0.5,
    )

    # the y=0 center line, plus one shaded vrect for the confirmed alarm run
    assert len(fig.layout.shapes) == 2
    reset_trace = next(trace for trace in fig.data if trace.name == "reset EWMA")
    assert list(reset_trace.x) == [index[5]]
