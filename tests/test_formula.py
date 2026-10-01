import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from bayesbuilding.formula import FormulaError, FormulaModel
from bayesbuilding.wrapper import PymcWrapper


class TestParsing:
    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo')",
            "g.__class__",
            "eval('1')",
            "(lambda: 1)()",
            "g if dt else tau",
            "g < dt",
            "g[dt*2]",
            "g[0.5]",
            "max(g)",
            "max(g, dt, tau)",
            "max(a=g, b=dt)",
            "'text'",
            "[g, dt]",
            "g @ dt",
            "max",
            "g +",
        ],
    )
    def test_rejects_constructs_outside_the_grammar(self, expr):
        with pytest.raises(FormulaError):
            FormulaModel(mu=expr, sigma="sigma")

    def test_rejects_reserved_extras(self):
        with pytest.raises(FormulaError, match="Reserved"):
            FormulaModel(mu="g*dt", sigma="sigma", extras={"sigma": "g"})

    def test_symbols_in_order_of_first_appearance(self):
        model = FormulaModel(
            mu="g[occ]*dt - fs*rad + max(tau - dt, 0)",
            sigma="s0[occ]",
            extras={"heat": "g[occ]*dt + rad2"},
        )
        assert model.symbols == ["g", "occ", "dt", "fs", "rad", "tau", "s0", "rad2"]
        assert model.categorical_inputs == {"occ"}

    def test_bind_derives_inputs_from_prior_names(self):
        model = FormulaModel(mu="g[occ]*dt - fs*rad", sigma="s0[occ]")
        model.bind(["g", "fs", "s0"])
        assert model.inputs == ("dt", "rad", "occ")

    def test_switch_condition_goes_after_drivers(self):
        model = FormulaModel(
            mu="switch(is_heating, base[0] + g*max(tau - text, 0) - fs*rad, base[1])",
            sigma="sigma",
        ).bind(["base", "g", "tau", "fs", "sigma"])
        assert model.inputs == ("text", "rad", "is_heating")
        assert model.categorical_inputs == set()

    def test_bind_keeps_explicit_input_order(self):
        model = FormulaModel(mu="g*dt + h*rad", sigma="s", inputs=["rad", "dt"])
        assert model.bind(["g", "h", "s"]).inputs == ("rad", "dt")

    def test_bind_rejects_explicit_inputs_missing_a_symbol(self):
        model = FormulaModel(mu="g*dt + h*rad", sigma="s", inputs=["dt"])
        with pytest.raises(FormulaError, match="do not match"):
            model.bind(["g", "h", "s"])

    def test_multi_index_categorical(self):
        model = FormulaModel(mu="g[occ, 1]*dt + base[is_heating, period]", sigma="s")
        model.bind(["g", "base", "s"])
        assert model.categorical_inputs == {"occ", "is_heating", "period"}

    def test_bind_warns_on_unused_prior(self):
        with pytest.warns(UserWarning, match="not used"):
            FormulaModel(mu="g*dt", sigma="sigma").bind(["g", "sigma", "tau"])

    def test_bind_rejects_index_that_is_a_prior(self):
        with pytest.raises(FormulaError, match="indices"):
            FormulaModel(mu="g[k]*dt", sigma="sigma").bind(["g", "k", "sigma"])

    def test_explicit_inputs_must_appear(self):
        with pytest.raises(FormulaError, match="do not appear"):
            FormulaModel(mu="g*dt", sigma="sigma", inputs=["dt", "rad"])

    def test_dict_round_trip(self):
        model = FormulaModel(
            mu="g*dt", sigma="sigma", lower=None, extras={"heat": "g*dt"}
        ).bind(["g", "sigma"])
        spec = json.loads(json.dumps(model.to_dict()))
        clone = FormulaModel.from_dict(spec)
        assert clone.to_dict() == model.to_dict()
        assert clone.inputs == ("dt",)


class TestEvaluation:
    def test_evaluates_indexing_and_functions(self):
        x = np.array([[10.0, 0.0], [2.0, 1.0]])  # columns: text, occ
        variables = {
            "g": pt.as_tensor_variable(np.array([1.0, 3.0])),
            "tau": pt.as_tensor_variable(np.array([15.0, 5.0])),
            "s0": pt.as_tensor_variable(np.array([0.5, 2.0])),
        }
        model = FormulaModel(
            mu="g[occ]*max(tau[occ] - text, 0)",
            sigma="s0[occ]",
            extras={"heat": "sqrt(abs(-4.0))"},
        ).bind(variables)
        mu, extras = model(pt.as_tensor_variable(x), variables)
        np.testing.assert_allclose(mu.eval(), [5.0, 9.0])
        np.testing.assert_allclose(extras["sigma"].eval(), [0.5, 2.0])
        assert float(extras["lower"].eval()) == 0.0
        assert float(extras["heat"].eval()) == 2.0

    def test_no_lower_when_none(self):
        model = FormulaModel(mu="g*dt", sigma="sigma", lower=None)
        variables = {"g": pt.constant(1.0), "sigma": pt.constant(1.0)}
        _, extras = model(pt.as_tensor_variable(np.ones((3, 1))), variables)
        assert "lower" not in extras

    def test_unbound_model_derives_inputs_at_call(self):
        model = FormulaModel(mu="g*dt + base", sigma="sigma")
        variables = {k: pt.constant(2.0) for k in ("g", "base", "sigma")}
        mu, _ = model(pt.as_tensor_variable(np.array([[1.0], [3.0]])), variables)
        np.testing.assert_allclose(mu.eval(), [4.0, 8.0])


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    text = rng.uniform(-5, 20, 60)
    occ = rng.integers(0, 2, 60).astype(float)
    y = 3.0 * np.maximum(15 - text, 0) + np.where(occ == 1, 20, 10)
    y = y + rng.normal(0, 1, 60)
    x = pd.DataFrame({"text": text, "occ": occ}, index=idx)
    return x, pd.Series(np.maximum(y, 0.1), index=idx, name="energy")


def test_wrapper_save_load_round_trip_with_formula(toy_data):
    x, y = toy_data
    priors = {
        "g": (pm.HalfNormal, {"name": "g", "sigma": 5}),
        "tau": (pm.Normal, {"name": "tau", "mu": 15, "sigma": 3}),
        "base": (pm.Normal, {"name": "base", "mu": 15, "sigma": 10, "shape": 2}),
        "sigma": (pm.HalfNormal, {"name": "sigma", "sigma": 3}),
    }
    model = FormulaModel(mu="g*max(tau - text, 0) + base[occ]", sigma="sigma").bind(
        priors
    )
    wrapper = PymcWrapper(
        model_function=model,
        priors_dict=priors,
        likelihood=pm.TruncatedNormal,
        likelihood_params=["sigma", "lower"],
    )
    wrapper.sample(
        x=x[list(model.inputs)],
        y=y,
        draws=50,
        tune=50,
        chains=1,
        sample_kwargs={"random_seed": 0, "cores": 1, "progressbar": False},
    )
    path = Path(tempfile.mkdtemp()) / "trace"
    wrapper.save_model(path)

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["model_function"] is None
    assert config["model_spec"]["mu"] == "g*max(tau - text, 0) + base[occ]"

    loaded = PymcWrapper()
    loaded.load_model(path)
    assert isinstance(loaded.model_function, FormulaModel)
    assert loaded.model_function.inputs == ("text", "occ")  # driver first
    loaded.sample_posterior_predictive(x=x[list(model.inputs)])
    assert "observations" in loaded.traces["posterior"].posterior_predictive
