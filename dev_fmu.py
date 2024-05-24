import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


# %%
def my_model(variable_dict, x):
    m = variable_dict["mtrue"]
    c = variable_dict["ctrue"]
    return m * x + c


# %%
class LogLike(Op):
    def __init__(self, model_function):
        self.model_function = model_function
        self.variables_keys = None

    def make_node(self, *variables, x, data) -> Apply:
        # Convert inputs to tensor variables
        variable_list = [pt.as_tensor(val) for val in variables]
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)

        inputs = [*variable_list, x, data]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [data.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(
        self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]
    ) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        variables = inputs[:-2]
        x, data = inputs[-2:]  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = self._loglike(*variables, x=x, data=data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

    def _loglike(self, *variable_list, x, data):
        # We fail explicitly if inputs are not numerical types for the sake of this tutorial
        # As defined, my_loglike would actually work fine with PyTensor variables!
        for param in variable_list + (x, data):
            if not isinstance(param, (float, np.ndarray)):
                raise TypeError(f"Invalid input type to loglike: {type(param)}")
        variable_dict = {
            key: val for key, val in zip(self.variables_keys, variable_list)
        }

        model = self.model_function(variable_dict, x)
        return (
            -0.5 * ((data - model) / variable_dict["sigma"]) ** 2
            - np.log(np.sqrt(2 * np.pi))
            - np.log(variable_dict["sigma"])
        )


# %%
# set up our data
N = 10  # number of data points
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

variables_dict = {"mtrue": mtrue, "ctrue": ctrue, "sigma": 1.0}

truemodel = my_model(variables_dict, x)

# make data
rng = np.random.default_rng(716743)
data = variables_dict["sigma"] * rng.normal(size=N) + truemodel

# create our Op
loglike_op = LogLike(my_model)
test_out = loglike_op(*tuple(variables_dict.values()), x=x, data=data)

pytensor.dprint(test_out, print_type=True)


# %%
def custom_dist_loglike(data, *dist_params):
    # data, or observed is always passed as the first input of CustomDist
    print(dist_params)
    variables = dist_params[:-1]
    return loglike_op(*variables, x=dist_params[-1], data=data)


# %%
# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    variable_dict = {
        "mtrue": pm.Uniform("mtrue", lower=-10.0, upper=10.0, initval=mtrue),
        "ctrue": pm.Uniform("ctrue", lower=-10.0, upper=10.0, initval=ctrue),
        "sigma": pm.Normal("sigma", mu=1.0, sigma=0.1, initval=1.0),
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
ip = no_grad_model.initial_point()

# %%
no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip)

# %%
try:
    no_grad_model.compile_dlogp()
except Exception as exc:
    print(type(exc))

# %%
with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(3000, tune=1000)

# %%
# plot the traces
az.plot_trace(idata_no_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)])
