import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from bayesbuilding.models import season_cp_heating_es
from bayesbuilding.plotting import time_series_hdi, changepoint_graph
from bayesbuilding.wrapper import PymcWrapper, _resample_samples

IMAGE_TEST_PATH = Path(tempfile.mkdtemp()) / "image.png"


class TestResampleSamples:
    def test_sums_samples_and_y_into_weekly_bins(self):
        index = pd.date_range("2023-01-02", periods=14, freq="D")  # Mon -> 2 full weeks
        flattened_trace = np.vstack(
            [np.ones(14), 2 * np.ones(14)]
        )  # 2 samples, constant per day
        y = pd.Series(np.arange(1, 15, dtype=float), index=index)

        resampled_trace, resampled_y = _resample_samples(
            flattened_trace, y, resample_rule="W"
        )

        assert resampled_trace.shape == (2, 2)
        np.testing.assert_allclose(resampled_trace[0], [7.0, 7.0])
        np.testing.assert_allclose(resampled_trace[1], [14.0, 14.0])
        np.testing.assert_allclose(resampled_y.to_numpy(), [28.0, 77.0])


class TestWrapper:
    def test_pymc_wrapper(self, tmp_path):
        data = pd.DataFrame(
            {
                "Text": [
                    7.149576,
                    6.622098,
                    11.563192,
                    13.513255,
                    17.170108,
                    22.034521,
                    21.532960,
                    22.664882,
                    21.892324,
                    17.477688,
                    11.189259,
                    8.383155,
                    7.207740,
                    10.597761,
                ]
            },
            index=pd.date_range("2023-01", freq="ME", periods=14),
        )

        true_g = 50  # kWh/°C
        true_base = 50  # kWh
        true_tau = 14  # °C
        true_sigma = 10

        np.random.seed(42)
        noise = np.random.randn(14)
        noise_g = true_sigma * noise + true_g
        noise_base = true_sigma * noise + true_base

        data["heating"] = noise_g * np.maximum(true_tau - data["Text"], 0) + noise_base

        data_train = data.loc["2023", :]
        data_test = data.loc["2024", :]

        test_model = PymcWrapper(
            model_function=season_cp_heating_es,
            priors_dict={
                "g": (pm.Normal, dict(name="g", mu=40, sigma=5)),
                "tau": (pm.Normal, dict(name="tau", mu=12, sigma=1)),
                "base": (pm.Normal, dict(name="base", mu=30, sigma=5)),
                "sigma": (pm.Normal, dict(name="sigma", mu=12, sigma=1)),
            },
        )

        test_model.sample_prior(
            samples=4000, x=data_train[["Text"]], sample_kwargs={"random_seed": 42}
        )

        assert round(test_model.get_summary(group="prior").loc["g", "mean"], 1) == 40.1
        assert round(test_model.get_summary(group="prior").loc["g", "sd"], 1) == 5.0

        test_model.sample(
            x=data_train[["Text"]],
            y=data_train["heating"],
            draws=4000,
            sample_kwargs={"random_seed": 42},
        )

        assert (
            round(test_model.get_summary(group="sampling").loc["g", "mean"], 1) == 47.6
        )
        assert round(test_model.get_summary(group="sampling").loc["g", "sd"], 1) == 1.9

        score_res = test_model.score(
            x=data_test[["Text"]],
            y=data_test["heating"],
            sample_kwargs={"random_seed": 42},
        )

        score_res = {key: round(val, 2) for key, val in score_res.items()}
        assert score_res == {"mean_score": 0.76, "sd_score": 0.07}

        test_model.get_loo_score()

        # === test save / load ===
        test_model.save_model(Path(tmp_path))


        new_model = PymcWrapper()
        new_model.load_model(Path(tmp_path))

        # === test plots ===
        test_model.sample_posterior_predictive(data[["Text"]])

        test_model.plot_dist_comparison()

        time_series_hdi(
            measure_ts=data["heating"],
            prediction=test_model.traces["posterior"].posterior_predictive[
                "observations"
            ],
            title="Posterior model accuracy",
            y_label="Energy consumption [kWh]",
            backend="plotly",
            image_path=IMAGE_TEST_PATH,
        )
        time_series_hdi(
            measure_ts=data["heating"],
            prediction=test_model.traces["posterior"].posterior_predictive[
                "observations"
            ],
            title="Posterior model accuracy",
            y_label="Energy consumption [kWh]",
            backend="matplotlib",
        )

        changepoint_graph(
            data["Text"],
            data["heating"],
            test_model.traces["posterior"].posterior_predictive["observations"],
            backend="plotly",
            x_label="text",
            y_label="heating",
            title="test",
        )

        changepoint_graph(
            data["Text"],
            data["heating"],
            test_model.traces["posterior"].posterior_predictive["observations"],
            backend="matplotlib",
            x_label="text",
            y_label="heating",
            title="test",
        )

    def test_score_with_weekly_resample_rule(self):
        index = pd.date_range("2023-01-02", periods=28, freq="D")  # Mon -> 4 weeks
        data = pd.DataFrame({"Text": np.linspace(0, 20, 28)}, index=index)

        true_g, true_base, true_tau, true_sigma = 50, 50, 14, 5
        np.random.seed(0)
        noise = true_sigma * np.random.randn(28)
        data["heating"] = (
            true_g * np.maximum(true_tau - data["Text"], 0) + true_base + noise
        )

        test_model = PymcWrapper(
            model_function=season_cp_heating_es,
            priors_dict={
                "g": (pm.Normal, dict(name="g", mu=40, sigma=5)),
                "tau": (pm.Normal, dict(name="tau", mu=12, sigma=1)),
                "base": (pm.Normal, dict(name="base", mu=30, sigma=5)),
                "sigma": (pm.Normal, dict(name="sigma", mu=12, sigma=1)),
            },
        )
        test_model.sample(
            x=data[["Text"]],
            y=data["heating"],
            draws=500,
            tune=500,
            chains=2,
            sample_kwargs={"random_seed": 42},
        )

        score_daily = test_model.score(
            x=data[["Text"]], y=data["heating"], sample_kwargs={"random_seed": 42}
        )
        score_weekly = test_model.score(
            x=data[["Text"]],
            y=data["heating"],
            sample_kwargs={"random_seed": 42},
            resample_rule="W",
        )

        assert set(score_daily) == {"mean_score", "sd_score"}
        assert set(score_weekly) == {"mean_score", "sd_score"}
        # Weekly aggregation averages out day-to-day noise -> tighter score spread.
        assert score_weekly["sd_score"] < score_daily["sd_score"]
