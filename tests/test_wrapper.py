import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from bayesbuilding.models import season_cp_heating_es
from bayesbuilding.plotting import time_series_hdi, changepoint_graph
from bayesbuilding.wrapper import PymcWrapper, _resample_samples

IMAGE_TEST_PATH = Path(tempfile.mkdtemp()) / "image.png"


def _toy_model_with_sigma(x, variables_dict):
    """Minimal model_function exercising the (mu, extras) tuple contract: a
    model-computed heteroscedastic "sigma" -- a genuinely new quantity (not a
    prior named "sigma"), so PymcWrapper.build_model exposes it as its own
    Deterministic."""
    driver = x[:, 0]
    base = variables_dict["base"]
    g = variables_dict["g"]
    s0 = variables_dict["s0"]
    alpha = variables_dict["alpha"]
    heat = pm.math.maximum(g * driver, 0)
    return base + heat, {"sigma": s0 + alpha * heat}


def _toy_model_with_sigma_and_nu(x, variables_dict):
    """Minimal model_function for a likelihood needing more than mu/sigma
    (e.g. StudentT's nu): every likelihood param must come back via extras,
    even a plain prior forwarded unchanged."""
    driver = x[:, 0]
    base = variables_dict["base"]
    g = variables_dict["g"]
    sigma = variables_dict["sigma"]
    nu = variables_dict["nu"]
    return base + g * driver, {"sigma": sigma, "nu": nu}


def _season_cp_heating_es_by_state(x, variable_dict):
    """Like season_cp_heating_es, but with a per-category sigma: the model
    itself indexes variables_dict["sigma"] by a state column and returns it
    explicitly via extras (replaces the old sigma_change_point_idx wrapper
    mechanism)."""
    t_ext = x[:, 0]
    state = x[:, 1].astype("int32")
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]
    sigma = variable_dict["sigma"][state]

    consumption = g * pm.math.maximum(tau - t_ext, 0)
    return consumption + baseline, {"sigma": sigma}


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

    def test_sigma_change_point_idx_indexes_sigma_by_category(self):
        """Regression coverage for a per-category sigma: the model_function
        indexes variables_dict["sigma"] itself and returns it via extras
        (replaces the old sigma_change_point_idx wrapper parameter)."""
        index = pd.date_range("2023-01-02", periods=20, freq="D")
        text = np.linspace(0, 20, 20)
        state = (text > 10).astype(float)  # raw categorical feature column

        true_g, true_base, true_tau = 50, 50, 14
        sigma_by_state = np.array([2.0, 20.0])
        np.random.seed(1)
        noise = np.random.randn(20) * sigma_by_state[state.astype(int)]
        heating = true_g * np.maximum(true_tau - text, 0) + true_base + noise
        data = pd.DataFrame({"Text": text, "state": state, "heating": heating}, index=index)

        test_model = PymcWrapper(
            model_function=_season_cp_heating_es_by_state,
            priors_dict={
                "g": (pm.Normal, dict(name="g", mu=40, sigma=5)),
                "tau": (pm.Normal, dict(name="tau", mu=12, sigma=1)),
                "base": (pm.Normal, dict(name="base", mu=30, sigma=5)),
                "sigma": (pm.Normal, dict(name="sigma", mu=10, sigma=2, shape=2)),
            },
        )
        test_model.sample(
            x=data[["Text", "state"]],
            y=data["heating"],
            draws=200,
            tune=200,
            chains=1,
            sample_kwargs={"random_seed": 42, "cores": 1},
        )

        assert test_model.traces["sampling"].posterior["sigma"].shape[-1] == 2


class TestExtrasAndModelDefinedSigma:
    def test_model_computed_sigma_extra_is_used_as_likelihood_scale(self):
        """A model_function can compute its own heteroscedastic sigma and
        return it under the reserved "sigma" extras key -- no separate
        sigma_function/priors_dict["sigma"] needed."""
        index = pd.date_range("2023-01-02", periods=20, freq="D")
        driver = np.linspace(0, 10, 20)
        data = pd.DataFrame({"driver": driver}, index=index)

        true_base, true_g = 10, 2
        np.random.seed(0)
        data["y"] = (
            true_base + true_g * driver + np.random.randn(20) * (1 + 0.5 * driver)
        )

        test_model = PymcWrapper(
            model_function=_toy_model_with_sigma,
            priors_dict={
                "base": (pm.Normal, dict(name="base", mu=10, sigma=5)),
                "g": (pm.Normal, dict(name="g", mu=2, sigma=1)),
                "s0": (pm.HalfNormal, dict(name="s0", sigma=5)),
                "alpha": (pm.HalfNormal, dict(name="alpha", sigma=1)),
            },
        )

        test_model.sample(
            x=data[["driver"]],
            y=data["y"],
            draws=200,
            tune=200,
            chains=1,
            sample_kwargs={"random_seed": 42, "cores": 1},
        )

        # "sigma" (the reserved extra, used as the likelihood scale) is
        # registered as a pm.Deterministic and lands in the posterior trace,
        # one value per observation -- not a single shared scalar.
        sigma_post = test_model.traces["sampling"].posterior["sigma"]
        assert sigma_post.shape[-1] == 20
        assert float(sigma_post.to_numpy().std(axis=-1).mean()) > 0
        # "sigma" is the reserved likelihood-scale key, not a plain extra.
        assert "sigma" not in test_model._extras_dict

    def test_model_function_bare_tensor_return_valid_when_no_likelihood_params(self):
        """A model_function returning a bare tensor (no extras tuple) is only
        valid when likelihood_params is empty -- here pm.Normal falls back to
        its own default sigma=1, with no priors_dict["sigma"] involved."""
        index = pd.date_range("2023-01-02", periods=10, freq="D")
        data = pd.DataFrame({"Text": np.linspace(0, 20, 10)}, index=index)
        data["heating"] = 50 * np.maximum(14 - data["Text"], 0) + 50

        def _bare_tensor_model(x, variable_dict):
            t_ext = x[:, 0]
            g = variable_dict["g"]
            tau = variable_dict["tau"]
            baseline = variable_dict["base"]
            return baseline + g * pm.math.maximum(tau - t_ext, 0)

        test_model = PymcWrapper(
            model_function=_bare_tensor_model,
            priors_dict={
                "g": (pm.Normal, dict(name="g", mu=40, sigma=5)),
                "tau": (pm.Normal, dict(name="tau", mu=12, sigma=1)),
                "base": (pm.Normal, dict(name="base", mu=30, sigma=5)),
            },
            likelihood_params=[],
        )
        test_model.sample(
            x=data[["Text"]],
            y=data["heating"],
            draws=200,
            tune=200,
            chains=1,
            sample_kwargs={"random_seed": 42, "cores": 1},
        )
        assert test_model._extras_dict == {}

    def test_missing_likelihood_param_raises_value_error(self):
        """A model_function that omits a required likelihood_params key from
        its extras dict fails fast with a clear error, instead of a KeyError
        on some guessed variables_dict lookup."""

        def _model_missing_sigma(x, variable_dict):
            return variable_dict["base"], {}

        with pytest.raises(ValueError, match="sigma"):
            PymcWrapper(
                model_function=_model_missing_sigma,
                priors_dict={
                    "base": (pm.Normal, dict(name="base", mu=0, sigma=1)),
                    "sigma": (pm.HalfNormal, dict(name="sigma", sigma=1)),
                },
            )

    def test_studentt_likelihood_uses_extra_nu_param(self):
        """likelihood/likelihood_params let a wrapper use a distribution family
        other than Normal, with model_function supplying every extra kwarg
        (here sigma and nu for a Student-T) explicitly via extras."""
        index = pd.date_range("2023-01-02", periods=20, freq="D")
        driver = np.linspace(0, 10, 20)
        data = pd.DataFrame({"driver": driver}, index=index)

        true_base, true_g, true_sigma = 10, 2, 1.5
        np.random.seed(3)
        data["y"] = (
            true_base
            + true_g * driver
            + true_sigma * np.random.standard_t(5, size=20)
        )

        test_model = PymcWrapper(
            model_function=_toy_model_with_sigma_and_nu,
            priors_dict={
                "base": (pm.Normal, dict(name="base", mu=10, sigma=5)),
                "g": (pm.Normal, dict(name="g", mu=2, sigma=1)),
                "sigma": (pm.HalfNormal, dict(name="sigma", sigma=5)),
                "nu": (pm.Gamma, dict(name="nu", alpha=2, beta=0.1)),
            },
            likelihood=pm.StudentT,
            likelihood_params=["sigma", "nu"],
        )
        test_model.sample(
            x=data[["driver"]],
            y=data["y"],
            draws=200,
            tune=200,
            chains=1,
            sample_kwargs={"random_seed": 42, "cores": 1},
        )

        assert "nu" in test_model.traces["sampling"].posterior
        assert "sigma" in test_model.traces["sampling"].posterior
