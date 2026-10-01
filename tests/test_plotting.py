import numpy as np
import pandas as pd
import pytest
import xarray

from bayesbuilding.plotting import (
    _boolean_blocks,
    _flatten_chains,
    get_cumulative_quantiles,
    plot_cumulative_energy_hdi,
    plot_cumulative_gap_hdi,
    time_series_hdi,
    time_series_hdi_comparison,
)


class TestFlattenChains:
    def test_2d_passthrough(self):
        arr = np.random.rand(10, 5)
        assert np.array_equal(_flatten_chains(arr), arr)

    def test_3d_reshape(self):
        arr = np.random.rand(4, 10, 5)  # chain, draw, time
        assert _flatten_chains(arr).shape == (40, 5)

    def test_xarray_input(self):
        arr = xarray.DataArray(np.random.rand(2, 3, 4), dims=("chain", "draw", "date"))
        flat = _flatten_chains(arr)
        assert isinstance(flat, np.ndarray)
        assert flat.shape == (6, 4)


class TestGetCumulativeQuantiles:
    def test_constant_per_draw_exact_cumsum(self):
        # 3 draws, each constant across 4 timesteps: values 1, 2, 3
        samples = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=float
        )
        q = get_cumulative_quantiles(samples, lower_q=0.0, upper_q=1.0)
        t = np.array([1, 2, 3, 4])
        np.testing.assert_allclose(q[0, :], 1 * t)  # min draw
        np.testing.assert_allclose(q[1, :], 2 * t)  # median draw
        np.testing.assert_allclose(q[2, :], 3 * t)  # max draw

    def test_monotonic_non_decreasing_for_nonnegative_draws(self):
        rng = np.random.default_rng(0)
        samples = rng.random((200, 30))  # non-negative -> cumsum must be non-decreasing
        q = get_cumulative_quantiles(samples)
        for row in q:
            assert np.all(np.diff(row) >= -1e-9)

    def test_2d_and_3d_equivalent(self):
        rng = np.random.default_rng(1)
        chain_draw = rng.random((3, 50, 10))
        flat = chain_draw.reshape(-1, 10)
        np.testing.assert_allclose(
            get_cumulative_quantiles(chain_draw), get_cumulative_quantiles(flat)
        )


class TestPlotCumulativeEnergyHdi:
    def _sample_data(self):
        rng = np.random.default_rng(2)
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        measure = pd.Series(rng.random(10) * 100, index=index)
        prediction = rng.random((2, 100, 10)) * 100  # chain, draw, time
        return measure, prediction

    def test_plotly_backend(self):
        measure, prediction = self._sample_data()
        fig = plot_cumulative_energy_hdi(measure, prediction, backend="plotly")
        assert fig is not None

    def test_matplotlib_backend(self, tmp_path):
        measure, prediction = self._sample_data()
        fig = plot_cumulative_energy_hdi(
            measure,
            prediction,
            backend="matplotlib",
            image_path=tmp_path / "cumulative.png",
        )
        assert fig is not None
        assert (tmp_path / "cumulative.png").exists()

    def test_invalid_backend_raises(self):
        measure, prediction = self._sample_data()
        try:
            plot_cumulative_energy_hdi(measure, prediction, backend="bogus")
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestPlotCumulativeGapHdi:
    def _sample_data(self):
        rng = np.random.default_rng(2)
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        measure = pd.Series(rng.random(10) * 100, index=index)
        prediction = rng.random((2, 100, 10)) * 100  # chain, draw, time
        return measure, prediction

    def test_plotly_backend(self):
        measure, prediction = self._sample_data()
        fig = plot_cumulative_gap_hdi(measure, prediction, backend="plotly")
        assert fig is not None

    def test_matplotlib_backend(self, tmp_path):
        measure, prediction = self._sample_data()
        fig = plot_cumulative_gap_hdi(
            measure,
            prediction,
            backend="matplotlib",
            image_path=tmp_path / "gap.png",
        )
        assert fig is not None
        assert (tmp_path / "gap.png").exists()

    def test_invalid_backend_raises(self):
        measure, prediction = self._sample_data()
        try:
            plot_cumulative_gap_hdi(measure, prediction, backend="bogus")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_gap_bounds_are_reversed_cumulative_quantiles(self):
        # gap = measure_cum - pred_cum is a decreasing transform of pred_cum,
        # so the gap's [low, up] band must equal measure_cum minus the
        # cumulative prediction's [up, low] quantiles (bounds swapped) -- this
        # is what keeps this function's "coverage" numerically identical to
        # plot_cumulative_energy_hdi's.
        measure, prediction = self._sample_data()
        pred_low, pred_med, pred_up = get_cumulative_quantiles(prediction)
        measure_cum = measure.cumsum().to_numpy()

        expected_gap_low = measure_cum - pred_up
        expected_gap_med = measure_cum - pred_med
        expected_gap_up = measure_cum - pred_low

        fig = plot_cumulative_gap_hdi(measure, prediction, backend="plotly")
        band_low = np.array(fig.data[1].y)
        gap_med = np.array(fig.data[2].y)
        band_up = np.array(fig.data[0].y)

        np.testing.assert_allclose(band_low, expected_gap_low)
        np.testing.assert_allclose(gap_med, expected_gap_med)
        np.testing.assert_allclose(band_up, expected_gap_up)


class TestBooleanBlocks:
    def _daily_index(self, n=10):
        return pd.date_range("2023-01-01", periods=n, freq="D")

    def test_all_false_returns_no_blocks(self):
        index = self._daily_index()
        mask = np.zeros(10, dtype=bool)
        assert _boolean_blocks(index, mask) == []

    def test_single_run_becomes_one_block(self):
        index = self._daily_index()
        mask = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        blocks = _boolean_blocks(index, mask)
        assert blocks == [(index[2], index[4])]

    def test_two_runs_become_two_blocks(self):
        index = self._daily_index()
        mask = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
        blocks = _boolean_blocks(index, mask)
        assert blocks == [(index[0], index[1]), (index[5], index[6])]

    def test_isolated_point_is_padded_by_half_median_step(self):
        index = self._daily_index()
        mask = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        [(start, end)] = _boolean_blocks(index, mask)
        half_step = pd.Timedelta("12h")
        assert start == index[3] - half_step
        assert end == index[3] + half_step


class TestTimeSeriesHdiStateOverlay:
    def _sample_data(self):
        rng = np.random.default_rng(3)
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        measure = pd.Series(rng.random(10) * 100, index=index)
        prediction = rng.random((2, 100, 10)) * 100  # chain, draw, time
        state = pd.Series([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], index=index)
        return measure, prediction, state

    def test_plotly_backend_with_state(self):
        measure, prediction, state = self._sample_data()
        fig = time_series_hdi(
            measure, prediction, state_ts=state, backend="plotly"
        )
        assert fig is not None

    def test_matplotlib_backend_with_state(self, tmp_path):
        measure, prediction, state = self._sample_data()
        fig = time_series_hdi(
            measure,
            prediction,
            state_ts=state,
            backend="matplotlib",
            image_path=tmp_path / "state.png",
        )
        assert fig is not None
        assert (tmp_path / "state.png").exists()

    def test_without_state_ts_still_works(self):
        measure, prediction, _ = self._sample_data()
        fig = time_series_hdi(measure, prediction, backend="plotly")
        assert fig is not None


class TestTimeSeriesHdiComparison:
    def _sample_data(self):
        rng = np.random.default_rng(4)
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        measure = pd.Series(rng.random(10) * 100, index=index)
        predictions = {
            "modele_a": rng.random((2, 100, 10)) * 100,  # chain, draw, time
            "modele_b": rng.random((2, 100, 10)) * 100,
        }
        state = pd.Series([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], index=index)
        return measure, predictions, state

    def test_plotly_backend_trace_count(self):
        measure, predictions, _ = self._sample_data()
        fig = time_series_hdi_comparison(measure, predictions, backend="plotly")
        assert fig is not None
        # 1 observed + 2 traces per model (median line + HDI band), no state
        assert len(fig.data) == 1 + 2 * len(predictions)

    def test_matplotlib_backend_line_count(self, tmp_path):
        measure, predictions, _ = self._sample_data()
        fig = time_series_hdi_comparison(
            measure,
            predictions,
            backend="matplotlib",
            image_path=tmp_path / "comparison.png",
        )
        assert fig is not None
        assert (tmp_path / "comparison.png").exists()
        ax = fig.axes[0]
        # one median line per model (fill_between doesn't add to ax.lines)
        assert len(ax.lines) == len(predictions)

    def test_plotly_backend_with_state_shading(self):
        measure, predictions, state = self._sample_data()
        fig = time_series_hdi_comparison(
            measure, predictions, state_ts=state, backend="plotly"
        )
        assert len(fig.layout.shapes) >= 1
        assert any(trace.name == "État" for trace in fig.data)

    def test_matplotlib_backend_with_state_shading(self):
        measure, predictions, state = self._sample_data()
        fig = time_series_hdi_comparison(
            measure, predictions, state_ts=state, backend="matplotlib"
        )
        ax = fig.axes[0]
        assert len(ax.patches) >= 1

    def test_without_state_ts_still_works(self):
        measure, predictions, _ = self._sample_data()
        fig = time_series_hdi_comparison(measure, predictions, backend="plotly")
        assert fig is not None
        assert len(fig.layout.shapes) == 0

    def test_custom_colors_dict(self):
        measure, predictions, _ = self._sample_data()
        fig = time_series_hdi_comparison(
            measure,
            predictions,
            backend="plotly",
            colors={"modele_a": "black", "modele_b": "purple"},
        )
        median_traces = [t for t in fig.data if "Médiane" in (t.name or "")]
        assert {t.line.color for t in median_traces} == {"black", "purple"}

    def test_empty_predictions_raises(self):
        measure, _, _ = self._sample_data()
        with pytest.raises(ValueError):
            time_series_hdi_comparison(measure, {}, backend="plotly")

    def test_invalid_backend_raises(self):
        measure, predictions, _ = self._sample_data()
        with pytest.raises(ValueError):
            time_series_hdi_comparison(measure, predictions, backend="bokeh")
