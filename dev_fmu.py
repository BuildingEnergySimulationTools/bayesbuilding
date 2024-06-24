import pytensor
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from corrai.fmu import ModelicaFmuModel
from bayesbuilding.custom import ModelicaFMULogLike, RefLogLike
from pathlib import Path

# %%
WORKDIR = Path(r"C:\Users\bdurandestebe\Documents\42_MBLABS\Bayesian_FMUs")


# %%
def my_model(variable_dict, x):
    m = variable_dict["m.k"]
    c = variable_dict["c.k"]
    return m * x + c


# %%
my_model_fmu = ModelicaFmuModel(
    WORKDIR / "lienear_fmu_combi_in_mo.fmu",
    simulation_options={
        "startTime": 0,
        "stopTime": 9,
        "stepSize": 1,
        "solver": "CVode",
        "outputInterval": 1,
        "tolerance": 1e-6,
        "fmi_type": "ModelExchange",
    },
    output_list=["add.y"],
)

# %%
my_model_fmu.fmu_instance
# %%

my_model_fmu.model_description

# %%
import cloudpickle

try:
    # Attempt to pickle a specific object
    cloudpickle.dumps(my_model_fmu)
except ValueError as e:
    print(f"Error pickling object: {e}")

# %%
my_model_fmu.simulate()

# %%
df = pd.DataFrame(
    {"x": [1.0, 2.0, 3.0, 4.0]},
    index=pd.date_range("2020-01-01 00:00:00", freq="s", periods=4),
)

# df = pd.DataFrame(
#     {"x": [1.0, 2.0, 3.0, 4.0]},
# )

# %%
# set up our data
N = 10  # number of data points
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

variables_dict = {"m.k": mtrue, "c.k": ctrue, "sigma": 1.0}

truemodel = my_model(variables_dict, x)

# %%
# my_model_fmu.simulate(parameter_dict={"m.k": mtrue, "c.k": ctrue}, x=pd.DataFrame(x))

# %%
my_model_fmu.simulation_options
# %%
# make data
rng = np.random.default_rng(716743)
data = variables_dict["sigma"] * rng.normal(size=N) + truemodel

# %%
# # create our Op
loglike_op = ModelicaFMULogLike(my_model_fmu)
# test_out = loglike_op(*tuple(variables_dict.values()), x=x, data=data)
#
# loglike_op = RefLogLike(my_model)
# pytensor.dprint(test_out, print_type=True)


def custom_dist_loglike(data, *dist_params):
    # data, or observed is always passed as the first input of CustomDist
    variables = dist_params[:-1]
    return loglike_op(*variables, x=dist_params[-1], data=data)


# %%
# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    variable_dict = {
        "mtrue": pm.Uniform("mtrue", lower=-10.0, upper=10.0, initval=mtrue),
        "ctrue": pm.Uniform("ctrue", lower=-10.0, upper=10.0, initval=ctrue),
        "sigma": pm.Normal("sigma", mu=1.0, sigma=0.05, initval=1.0),
    }

    loglike_op.variables_keys = variables_dict.keys()
    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood",
        *(tuple(variable_dict.values()) + (x,)),
        observed=data,
        logp=custom_dist_loglike,
    )

# %%
# ip = no_grad_model.initial_point()

# %%
# no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip)


# %%
# with no_grad_model:
#     prior = pm.sample_prior_predictive(100)

# %%
with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(draws=10, tune=5, random_seed=42, cores=1)

# %%
# plot the traces
az.plot_trace(idata_no_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)])
plt.show()

# %%
