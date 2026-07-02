<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/bayesbuilding/main/logo_bayes_building.svg" alt="BayesBuilding" width="200"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/bayesbuilding?label=pypi%20package)](https://pypi.org/project/bayesbuilding/)
[![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.11-blue)](https://pypi.org/project/bayesbuilding/)
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
- A library of ready-to-use `bayesbuilding.models` functions covering common building
  energy patterns: seasonal change-point energy signatures, heating/cooling with
  occupation change points, solar-radiation-augmented models, artificial lighting, and
  PV panel production (constant efficiency and NOCT models).
- Plotting helpers (`bayesbuilding.plotting`) for prior/posterior comparison, HDI time
  series plots, and change-point diagnostic plots, with both `matplotlib` and `plotly`
  backends.

## Installation

```bash
pip install bayesbuilding
```

Requires Python >= 3.10. See `pyproject.toml` for the full list of dependencies
(PyMC, ArviZ, xarray, pandas, numpy, matplotlib, plotly, seaborn).

## Quickstart

```python
import numpy as np
import pandas as pd
import pymc as pm

from bayesbuilding.models import season_cp_heating_es
from bayesbuilding.wrapper import PymcWrapper

# Monthly external temperature and heating consumption
data = pd.DataFrame(
    {"Text": [7.1, 6.6, 11.6, 13.5, 17.2, 22.0, 21.5, 22.7, 21.9, 17.5, 11.2, 8.4]},
    index=pd.date_range("2023-01", freq="ME", periods=12),
)
data["heating"] = 50 * np.maximum(14 - data["Text"], 0) + 50 + np.random.randn(12) * 5

# Define the model and priors for a seasonal change-point energy signature:
# heating = g * max(tau - Text, 0) + base
model = PymcWrapper(
    model_function=season_cp_heating_es,
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

See `bayesbuilding/models.py` for the full list of built-in model functions and
`tests/test_wrapper.py` for a complete end-to-end example, including scoring on held-out
data and plotting predictions with `bayesbuilding.plotting.time_series_hdi` and
`changepoint_graph`.