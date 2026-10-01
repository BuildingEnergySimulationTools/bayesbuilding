<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/bayesbuilding/main/logo_bayes_building.svg" alt="BayesBuilding" width="200"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/bayesbuilding?label=pypi%20package)](https://pypi.org/project/bayesbuilding/)
[![Static Badge](https://img.shields.io/badge/python-3.12-blue)](https://pypi.org/project/bayesbuilding/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/BuildingEnergySimulationTools/bayesbuilding/actions/workflows/build.yaml/badge.svg)](https://github.com/BuildingEnergySimulationTools/bayesbuilding/actions)

# Bayes Building

A Bayesian approach to HVAC and building energy modeling, built on top of
[PyMC](https://www.pymc.io/) and [ArviZ](https://python.arviz.org/).

BayesBuilding wraps the common workflow of fitting a physics-inspired regression model
(energy signatures, change-point models, PV production models, ...) to measured building
data: define priors, sample the prior and posterior distributions, score the model on
held-out data, and visualize the results, without writing PyMC boilerplate for every
project.

## Features

- `PymcWrapper`: a thin wrapper around a PyMC model that handles prior/posterior
  sampling, scoring, LOO cross-validation, and saving/loading fitted models to disk.
- Forward models written as formulas (`bayesbuilding.formula.FormulaModel`), e.g.
  `"g[occ]*max(tau - text, 0) - fs*rad + base[occ]"`: no Python function to write for
  a new model variant. (`bayesbuilding.models` holds the older hand-written functions,
  frozen and kept only to reload traces saved with them.)
- JSON candidate configs (`bayesbuilding.candidates`) and a training loop that fits,
  scores (LOO, R2, NMBE, CV(RMSE), MAE at daily/weekly/monthly resolution) and ranks
  candidates (`bayesbuilding.training.train_candidates`).
- Control charts on residuals (`bayesbuilding.control_charts`): X-bar, EWMA (with alarm
  confirmation/reset) and CUSUM, with limits following the model's own sigma.
- Plotting helpers (`bayesbuilding.plotting`) for prior/posterior comparison, HDI time
  series plots, and change-point diagnostic plots, with both `matplotlib` and `plotly`
  backends.

## Installation

```bash
pip install bayesbuilding
```

Requires Python >= 3.12. See `pyproject.toml` for the full list of dependencies
(PyMC, ArviZ, xarray, pandas, numpy, matplotlib, plotly, seaborn).

## Quickstart

```python
import numpy as np
import pandas as pd
import pymc as pm

from bayesbuilding.formula import FormulaModel
from bayesbuilding.wrapper import PymcWrapper

# Monthly external temperature and heating consumption
data = pd.DataFrame(
    {"Text": [7.1, 6.6, 11.6, 13.5, 17.2, 22.0, 21.5, 22.7, 21.9, 17.5, 11.2, 8.4]},
    index=pd.date_range("2023-01", freq="ME", periods=12),
)
data["heating"] = 50 * np.maximum(14 - data["Text"], 0) + 50 + np.random.randn(12) * 5

# Define the model and priors for a seasonal change-point energy signature.
# Symbols that are not priors are inputs, i.e. columns of x. The likelihood
# declared in the FormulaModel is the one the wrapper uses.
model = PymcWrapper(
    model_function=FormulaModel(
        likelihood="TruncatedNormal",
        mu="g*max(tau - Text, 0) + base",
        params={"sigma": "sigma", "lower": 0.0},
    ),
    priors_dict={
        "g": (pm.Normal, dict(name="g", mu=40, sigma=5)),
        "tau": (pm.Normal, dict(name="tau", mu=12, sigma=1)),
        "base": (pm.Normal, dict(name="base", mu=30, sigma=5)),
        "sigma": (pm.Normal, dict(name="sigma", mu=12, sigma=1)),
    },
)

# Sample the prior, then fit the model on data
model.sample_prior(samples=2000, x=data[["Text"]])
model.sample(x=data[["Text"]], y=data["heating"], draws=2000)

print(model.get_summary(group="sampling"))
print(model.get_loo_score())

# Save / reload a fitted model
model.save_model("my_model")
reloaded = PymcWrapper()
reloaded.load_model("my_model")
```

## Formula grammar

- Operators `+ - * / **`, numeric constants, and the functions `max(a, b)`,
  `min(a, b)`, `switch(cond, a, b)`, `sigmoid`, `sqrt`, `exp`, `log`, `abs`.
- `g[occ]` / `g[occ, 1]` index a vector/matrix prior by a categorical input.
- Input order (the columns of `x`): continuous drivers first, then categorical inputs,
  each in order of first appearance.

Expressions are parsed against a whitelist, never `eval`-ed.

## Likelihood

A model declares the distribution of the observations explicitly:

- `likelihood`: the name of a PyMC distribution (`"Normal"`, `"TruncatedNormal"`,
  `"StudentT"`, ...).
- `mu`: the formula of its mean.
- `params`: every other argument of that distribution, each a formula or a number.
  `sigma` may be heteroscedastic, e.g.
  `"sqrt(s0[occ]**2 + (s1*sigmoid((tau - text)/1.5))**2)"`. Parameter names are
  checked against the distribution's signature.

`TruncatedNormal`'s `lower` is the truncation bound: observations are `Normal(mu, sigma)`
restricted to `[lower, +inf)` and renormalized, since an energy consumption cannot be
negative. It matters when `mu` gets close to 0 (summer, mid-season): a plain `Normal`
would put some probability on negative consumptions. If the data contain many exact
zeros (heating off), a censored likelihood describes them better than a truncated one.

The older form `{"mu": ..., "sigma": ..., "lower": 0.0}` is still accepted: it means a
`TruncatedNormal`, or a `Normal` when `lower` is `null`.

## From a candidate dict to a wrapper

A candidate model is a JSON-serializable dict. `inputs` maps each input symbol of the
formulas to a column of your data:

```python
import numpy as np
import pandas as pd

from bayesbuilding.candidates import CandidateConfig
from bayesbuilding.training import build_candidate_xy

candidate_dict = {
    "name": "dt_occ",
    "model": {
        "likelihood": "TruncatedNormal",
        "mu": "g[occ]*dt - fs[occ]*rad",
        "params": {"sigma": "s0[occ]", "lower": 0.0},
    },
    "inputs": {"dt": "dt__C", "rad": "GHI__W/m2", "occ": "occupation"},
    "priors": {
        "g": {"dist": "HalfNormal", "kwargs": {"sigma": 500, "shape": 2}},
        "fs": {"dist": "HalfNormal", "kwargs": {"sigma": 5, "shape": 2}},
        "s0": {"dist": "HalfNormal", "kwargs": {"sigma": 500, "shape": 2}},
    },
    "draws": 1000,
    "tune": 1000,
}

# Daily data: indoor/outdoor temperature difference, solar radiation, occupation
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "dt__C": rng.uniform(0, 20, 120),
        "GHI__W/m2": rng.uniform(0, 300, 120),
        "occupation": rng.integers(0, 2, 120),
    },
    index=pd.date_range("2024-01-01", freq="D", periods=120),
)
df["heating"] = np.maximum(
    np.where(df["occupation"] == 1, 300, 150) * df["dt__C"] - 2 * df["GHI__W/m2"], 0
) + rng.normal(0, 100, 120).clip(0)

candidate = CandidateConfig(**candidate_dict)
wrapper = candidate.build_wrapper()  # a PymcWrapper with a TruncatedNormal likelihood
x, y = build_candidate_xy(df, "heating", candidate)  # columns in the formula's order
wrapper.sample(x=x, y=y, draws=candidate.draws, tune=candidate.tune)
print(wrapper.get_summary(group="sampling"))
```

A whole config (several candidates sharing a target) is loaded with
`BayesConfig.from_json(path)`, and `bayesbuilding.training.train_candidates` fits, scores
and ranks all its candidates.

See `tests/test_wrapper.py` for a complete end-to-end example, including scoring on held-out
data and plotting predictions with `bayesbuilding.plotting.time_series_hdi` and
`changepoint_graph`.