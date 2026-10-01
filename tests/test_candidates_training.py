import json
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from bayesbuilding.candidates import (
    BayesConfig,
    CandidateConfig,
    build_priors_dict,
    posterior_as_priors,
)
from bayesbuilding.formula import FormulaModel
from bayesbuilding.legacy_formulas import LEGACY_FORMULAS, legacy_to_formula
from bayesbuilding.training import (
    build_candidate_xy,
    cv_rmse,
    flatten_regime_vars,
    mae,
    nmbe,
    train_candidates,
)
from bayesbuilding.wrapper import PymcWrapper
from tests.test_formula_legacy_equivalence import PRIOR_SHAPES

PRIORS_DT = {
    "g": {"dist": "Normal", "kwargs": {"mu": 100.0, "sigma": 10.0}},
    "tau": {"dist": "Normal", "kwargs": {"mu": 0.0, "sigma": 1.0}},
    "base": {"dist": "Normal", "kwargs": {"mu": 10.0, "sigma": 1.0}},
    "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 1.0}},
}

CONFIG_PAYLOAD = {
    "config_id": 1,
    "feature_pipe": {"resample": [["Resample", {"rule": "d"}]]},
    "target": "chauffage__Wh__Zone__FULL",
    "candidates": [
        {
            "name": "dt_change_point",
            "model": {"mu": "g*max(dt - tau, 0) + base", "sigma": "sigma", "lower": 0.0},
            "inputs": {"dt": "dt__C__Zone__FULL"},
            "priors": PRIORS_DT,
            "prior_draws": 500,
            "draws": 1000,
            "tune": 500,
            "chains": 2,
            "random_seed": 42,
        },
    ],
    "unused_candidates": [
        {
            "name": "text_change_point",
            "model": {"mu": "g*max(tau - text, 0) + base", "sigma": "sigma"},
            "inputs": {"text": "Text__C"},
            "priors": PRIORS_DT,
        },
    ],
}


class TestConfig:
    def test_from_json_round_trip(self, tmp_path):
        path = tmp_path / "bayes_config.json"
        path.write_text(json.dumps(CONFIG_PAYLOAD))

        config = BayesConfig.from_json(path)

        assert config.config_id == 1
        dt_candidate = config.candidates[0]
        assert isinstance(dt_candidate.model, FormulaModel)
        assert dt_candidate.model.inputs == ("dt",)
        assert dt_candidate.draws == 1000
        assert config.unused_candidates[0].draws == 4000  # default
        assert config.get_candidate("text_change_point").model.inputs == ("text",)

        config.to_json(tmp_path / "copy.json")
        assert BayesConfig.from_json(tmp_path / "copy.json").to_dict() == config.to_dict()

    def test_unused_candidates_is_optional(self):
        payload = {k: v for k, v in CONFIG_PAYLOAD.items() if k != "unused_candidates"}
        assert BayesConfig.from_dict(payload).unused_candidates == []

    def test_legacy_model_name_is_translated_with_a_warning(self):
        with pytest.warns(DeprecationWarning, match="legacy model name"):
            candidate = CandidateConfig(
                name="legacy",
                model="season_cp_heating_es_dt",
                inputs={"dt": "dt__C"},
                priors=PRIORS_DT,
            )
        assert candidate.model.mu == LEGACY_FORMULAS["season_cp_heating_es_dt"]["mu"]
        assert candidate.model.lower == 0.0

    def test_candidate_errors_name_the_candidate(self):
        with pytest.raises(ValueError, match="'bad'.*indices"):
            CandidateConfig(
                name="bad",
                model={"mu": "g[k]*dt", "sigma": "sigma"},
                inputs={},
                priors={"g": {}, "sigma": {}, "k": {}},
            )

    def test_build_priors_dict_injects_name_and_resolves_distribution(self):
        priors = build_priors_dict(PRIORS_DT)
        assert priors["g"][0] is pm.Normal
        assert priors["g"][1] == {"name": "g", "mu": 100.0, "sigma": 10.0}


class TestBuildCandidateXY:
    def test_selects_and_orders_columns_driver_first(self):
        df = pd.DataFrame(
            {
                "target": [1.0, 2.0],
                "occ_col": [0.0, 1.0],
                "dt_col": [3.0, 4.0],
                "rad_col": [5.0, 6.0],
                "unrelated": [7.0, 8.0],
            }
        )
        candidate = CandidateConfig(
            name="dt_occ",
            model={"mu": "g[occ]*dt - fs*rad", "sigma": "s0[occ]"},
            inputs={"occ": "occ_col", "dt": "dt_col", "rad": "rad_col"},
            priors={"g": {}, "fs": {}, "s0": {}},
        )
        x, y = build_candidate_xy(df, "target", candidate)
        assert list(x.columns) == ["dt_col", "rad_col", "occ_col"]
        assert y.tolist() == [1.0, 2.0]

    def test_raises_on_missing_input_mapping(self):
        df = pd.DataFrame({"target": [1.0], "dt_col": [2.0]})
        candidate = CandidateConfig(
            name="incomplete",
            model={"mu": "g*max(dt - tau, 0) + base", "sigma": "sigma"},
            inputs={},
            priors=PRIORS_DT,
        )
        with pytest.raises(KeyError, match="incomplete"):
            build_candidate_xy(df, "target", candidate)


@pytest.mark.parametrize("name", sorted(LEGACY_FORMULAS))
def test_every_legacy_formula_builds_a_truncated_normal_wrapper(name):
    """What fit_candidate builds, minus the (expensive) sampling."""
    priors = {
        var: {"dist": "HalfNormal", "kwargs": {"sigma": 1.0, "shape": shape or None}}
        for var, shape in PRIOR_SHAPES[name].items()
    }
    for spec in priors.values():
        if spec["kwargs"]["shape"] is None:
            del spec["kwargs"]["shape"]
    candidate = CandidateConfig(
        name=name, model=legacy_to_formula(name), inputs={}, priors=priors
    )
    wrapper = PymcWrapper(
        model_function=candidate.model,
        priors_dict=build_priors_dict(priors),
        likelihood=pm.TruncatedNormal,
        likelihood_params=["sigma", "lower"],
    )
    assert "mu" in {rv.name for rv in wrapper.model.deterministics}


def test_posterior_as_priors_does_not_inject_mu_for_half_normal():
    """A HalfNormal (only a 'sigma' kwarg) must not come back with a spurious
    'mu' -- pm.HalfNormal doesn't accept one."""
    candidate = CandidateConfig(
        name="c",
        model={"mu": "g*dt", "sigma": "s0"},
        inputs={},
        priors={
            "g": {"dist": "Normal", "kwargs": {"mu": 100.0, "sigma": 10.0}},
            "s0": {"dist": "HalfNormal", "kwargs": {"sigma": 3000.0}},
        },
    )
    fake_posterior = xr.Dataset(
        {
            "g": (("chain", "draw"), [[95.0, 105.0, 100.0]]),
            "s0": (("chain", "draw"), [[1.0, 2.0, 3.0]]),
        }
    )

    class FakeTrace:
        posterior = fake_posterior

    class FakeWrapper:
        traces = {"sampling": FakeTrace()}

    updated = posterior_as_priors(candidate, FakeWrapper())

    assert set(updated["g"]["kwargs"]) == {"mu", "sigma"}
    assert set(updated["s0"]["kwargs"]) == {"sigma"}


def test_flatten_regime_vars_explodes_shape_2_regime_variable():
    group = xr.Dataset(
        {
            "sigma": (("chain", "draw"), [[1.0, 2.0, 3.0]]),
            "g": (("chain", "draw", "g_dim_0"), [[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]]),
        }
    )
    flattened = flatten_regime_vars(group, ["sigma", "g", "does_not_exist"])
    assert set(flattened) == {"sigma", "g0", "g1"}
    np.testing.assert_allclose(flattened["g1"], [20.0, 21.0, 22.0])


def test_metrics_match_corrai_definitions():
    # Values from corrai.base.metrics docstrings (y_pred, y_true order there).
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    assert nmbe(y_true, y_pred) == pytest.approx(1.6666666666666667)
    assert cv_rmse(y_true, y_pred) == pytest.approx(6.123724356957945)
    assert mae(y_true, y_pred) == pytest.approx(10.0)


def test_train_candidates_fits_scores_and_records_failures(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    dt = rng.uniform(0, 15, 70)
    df = pd.DataFrame(
        {"dt__C": dt, "energy": 4.0 * np.maximum(dt - 3, 0) + 10 + rng.normal(0, 1, 70)},
        index=idx,
    )
    priors = {
        "g": {"dist": "HalfNormal", "kwargs": {"sigma": 10.0}},
        "tau": {"dist": "Normal", "kwargs": {"mu": 3.0, "sigma": 2.0}},
        "base": {"dist": "Normal", "kwargs": {"mu": 10.0, "sigma": 5.0}},
        "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 3.0}},
    }
    small = {"prior_draws": 20, "draws": 60, "tune": 60, "chains": 1, "random_seed": 0}
    config = BayesConfig.from_dict(
        {
            "target": "energy",
            "candidates": [
                {
                    "name": "cp",
                    "model": {"mu": "g*max(dt - tau, 0) + base", "sigma": "sigma"},
                    "inputs": {"dt": "dt__C"},
                    "priors": priors,
                    **small,
                },
                {
                    "name": "broken",
                    "model": {"mu": "g*max(dt - tau, 0) + base", "sigma": "sigma"},
                    "inputs": {"dt": "missing_column"},
                    "priors": priors,
                    **small,
                },
            ],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrappers, comparison, errors = train_candidates(config, df, tmp_path)

    assert list(wrappers) == ["cp"]
    assert list(errors) == ["broken"]
    assert list(comparison.index) == ["cp"]
    for col in ("elpd_loo", "elpd_loo_se", "r2_mean_d", "cv_rmse_sd_ME", "mae_mean_W"):
        assert col in comparison.columns
    assert (tmp_path / "comparison_train.csv").exists()
    assert (tmp_path / "cp" / "trace" / "config.json").exists()
