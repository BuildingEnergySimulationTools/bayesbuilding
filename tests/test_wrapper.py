import pandas as pd
import numpy as np
import pymc as pm

from bayesbuilding.wrapper import PymcWrapper
from bayesbuilding.models import season_cp_heating_es

class TestWrapper:
    def test_pymc_wrapper(self):
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

        assert round(
            test_model.get_summary(group="sampling").loc["g", "mean"], 1) == 47.6
        assert round(test_model.get_summary(group="sampling").loc["g", "sd"], 1) == 1.9

        score_res = test_model.score(
            x=data_test[["Text"]], y=data_test["heating"],
            sample_kwargs={"random_seed": 42}
        )

        score_res = {key: round(val, 2) for key, val in score_res.items()}
        assert score_res == {"mean_score": 0.76, "sd_score": 0.07}

        test_model.get_loo_score()
