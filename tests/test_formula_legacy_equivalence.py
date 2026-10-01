import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest

import bayesbuilding.models as mods
from bayesbuilding.formula import FormulaModel
from bayesbuilding.legacy_formulas import (
    LEGACY_FORMULAS,
    LEGACY_INPUTS,
    legacy_to_formula,
)

N = 50
N_OCC, N_PERIOD, N_ROOM = 2, 3, 3

CATEGORICAL_LEVELS = {"occ": N_OCC, "is_heating": 2, "period": N_PERIOD}

# Shape of each prior, per legacy model (scalar when absent).
PRIOR_SHAPES = {
    "season_cp_heating_es_dt": {"g": (), "tau": (), "base": (), "sigma": ()},
    "season_cp_heating_es": {"g": (), "tau": (), "base": (), "sigma": ()},
    "heating_es_dju": {"g": (), "base": (), "sigma": ()},
    "season_cp_occ_cp_heating_es": {
        k: (N_OCC,) for k in ("g", "tau", "baseline", "sigma")
    },
    "season_cp_occ_cp_es_dt": {k: (N_OCC,) for k in ("g", "tau", "baseline", "sigma")},
    "heating_es_dju_rad": {"g": (), "fs": (), "base": (), "sigma": ()},
    "season_cp_heating_es_rad": {
        "g": (),
        "fs": (),
        "tau": (),
        "base": (2,),
        "sigma": (),
    },
    "season_cp_heating_es_setback": {"g": (), "tau": (), "base": (2,), "sigma": ()},
    "heating_es_dju_rad_occ": {
        "g": (N_OCC,),
        "fs": (N_OCC,),
        "base": (N_OCC,),
        "sigma": (),
    },
    "heating_es_dju_rad_occ_setback": {"g": (), "fs": (), "base": (2,), "sigma": ()},
    "heating_cp_occ_rad": {
        **{k: (N_OCC,) for k in ("base", "g", "tau", "fs", "s0")},
        "s1": (),
    },
    "season_cp_heating_es_rad_g_by_period": {
        "g": (N_PERIOD,),
        "fs": (),
        "tau": (),
        "base": (2,),
        "sigma": (),
    },
    "season_cp_heating_es_rad_tau_by_period": {
        "g": (),
        "fs": (),
        "tau": (N_PERIOD,),
        "base": (2,),
        "sigma": (),
    },
    "season_cp_heating_es_rad_base_by_period": {
        "g": (),
        "fs": (),
        "tau": (),
        "base": (2, N_PERIOD),
        "sigma": (),
    },
    "season_cp_heating_es_rad_g_tau_by_period": {
        "g": (N_PERIOD,),
        "fs": (),
        "tau": (N_PERIOD,),
        "base": (2,),
        "sigma": (),
    },
}
_DT = {k: (N_OCC,) for k in ("g", "fs", "alpha", "s0")}
PRIOR_SHAPES.update(
    {
        "heating_dt_occ_rad": _DT,
        "heating_dt_occ_rad_multiroom": {
            "g": (N_OCC, N_ROOM),
            "fs": (N_OCC, N_ROOM),
            "alpha": (N_OCC,),
            "s0": (N_OCC,),
        },
        "heating_dt_occ_rad_lag": {**_DT, "h": (N_OCC,)},
        "heating_dt_occ_rad_DTdt": {**_DT, "c": ()},
        "heating_dt_occ_rad_DTdt_lags": {**_DT, "c0": (), "rho": ()},
        "heating_dt_occ_rad_DTdt_wall": {**_DT, "C": (), "beta": ()},
        "heating_dt_occ_rad_DTdt_wall_Ci": {**_DT, "C": (), "beta": (), "Ci": ()},
        "heating_dt_occ_rad_DTdt_wall_Ci_radlag": {
            **_DT,
            "C": (),
            "beta": (),
            "Ci": (),
            "fm": (N_OCC,),
        },
    }
)


def test_tables_cover_the_same_models():
    assert set(LEGACY_FORMULAS) == set(LEGACY_INPUTS) == set(PRIOR_SHAPES)


@pytest.mark.parametrize("name", sorted(LEGACY_FORMULAS))
def test_formula_matches_legacy_model(name):
    rng = np.random.default_rng(0)
    columns = {}
    for col in LEGACY_INPUTS[name]:
        if col in CATEGORICAL_LEVELS:
            columns[col] = rng.integers(0, CATEGORICAL_LEVELS[col], N).astype(float)
        else:
            columns[col] = rng.normal(5.0, 3.0, N)
    df = pd.DataFrame(columns)
    variables = {
        k: pt.as_tensor_variable(rng.uniform(0.1, 2.0, shape))
        for k, shape in PRIOR_SHAPES[name].items()
    }

    legacy_mu, legacy_extras = getattr(mods, name)(
        pt.as_tensor_variable(df[list(LEGACY_INPUTS[name])].to_numpy()), variables
    )

    formula = FormulaModel.from_dict(legacy_to_formula(name)).bind(variables)
    assert set(formula.inputs) == set(LEGACY_INPUTS[name])
    # the main driver comes first (scripts plot against x.iloc[:, 0])
    assert formula.inputs[0] not in CATEGORICAL_LEVELS
    mu, extras = formula(
        pt.as_tensor_variable(df[list(formula.inputs)].to_numpy()), variables
    )

    np.testing.assert_allclose(mu.eval(), legacy_mu.eval(), rtol=1e-10)
    np.testing.assert_allclose(
        np.broadcast_to(extras["sigma"].eval(), N),
        np.broadcast_to(legacy_extras["sigma"].eval(), N),
        rtol=1e-10,
    )
    assert float(extras["lower"].eval()) == float(legacy_extras["lower"].eval())
