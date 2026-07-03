import numpy as np
import pandas as pd
import xarray

from bayesbuilding.plotting import (
    _flatten_chains,
    get_cumulative_quantiles,
    plot_cumulative_energy_hdi,
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
