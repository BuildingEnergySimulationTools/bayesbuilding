import json
import os
from collections.abc import Callable
from pathlib import Path
import bayesbuilding.models as mods
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from xarray import DataTree


def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    return 1 - numerator / denominator


def _resample_samples(flattened_trace: np.ndarray, y: pd.Series, resample_rule: str):
    """Sum each (samples, time) posterior predictive sample path, and `y`, into
    `resample_rule` bins (e.g. "W"). Aggregation happens on daily sample paths
    rather than on the driving variable before re-evaluating the model, since a
    nonlinear forward model (e.g. a change-point's max(driving_var - tau, 0)) isn't
    equivalent under the two orderings."""
    resampled_trace = (
        pd.DataFrame(flattened_trace.T, index=y.index)
        .resample(resample_rule)
        .sum()
        .to_numpy()
        .T
    )
    resampled_y = y.resample(resample_rule).sum()
    return resampled_trace, resampled_y


def custom_serializer(obj):
    if callable(obj) and hasattr(obj, "__name__"):
        return obj.__name__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class PymcWrapper:
    """
    A class simplifying the creation of probabilistic models using PyMC3.

    This class abstracts some complexities of PyMC while enforcing a structured
     framework:
    - Features are provided as pandas DataFrames.
    - Only a single target variable, provided as a pandas Series, is allowed.
    - The model is defined by a model function, which returns the mean ('mu') of
        the likelihood, plus, explicitly, every other parameter the chosen
        likelihood distribution needs (e.g. 'sigma').
    - The likelihood distribution is configurable (`likelihood`), defaulting to
        Normal. The likelihood function is used for prediction, with the
        variable named "observations".

    Parameters:
    -----------
    model_function : Callable
        A function defining the PyMC model. It takes two arguments: 'x' and
        'variable_dict'. 'x' is a 2D array derived from the features DataFrame.
        Use standard indexing as DataFrame column names cannot be used within
        the function's scope.
        'variable_dict' is a dictionary where keys are variable names and values
        are PyMC variables.
        It must return a tuple ``(mu, extras)``: ``mu`` is the model's mean
        output, and ``extras`` is a dict[str, tensor] that must contain, under
        its own name, every parameter listed in `likelihood_params` (e.g.
        ``{"sigma": ...}``) -- there is no implicit fallback to `variables_dict`
        for these: `model_function` must build and return them explicitly, even
        when that's just ``variables_dict["sigma"]`` unchanged. This is also
        where a model computes its own (possibly heteroscedastic or per-category)
        noise term, from quantities only the model itself has access to (e.g.
        `sigma = s0 + alpha * heat`, or `sigma = variables_dict["sigma"][state]`
        for a change-point-indexed sigma -- see
        `bayesbuilding.models.heating_cp_occ_rad`). ``extras`` may also contain
        further named quantities not required by `likelihood_params`, exposed as
        extra `pm.Deterministic`s for diagnostics (e.g. an internal 'heat' term).
        A bare tensor return (no tuple) is only valid when `likelihood_params`
        is empty.
    priors_dict : dict[str:(Callable, dict)]
        A dictionary containing prior distributions for model parameters.
        Keys are parameter names, and values are tuples where the first element is
        a PyMC callable defining the prior distribution, and the second element is
        a dictionary of parameters for the prior distribution.
        These are only the raw materials `model_function` has available in
        `variables_dict`; they are never looked up by the wrapper itself to
        build the likelihood -- `model_function` decides explicitly what feeds
        `mu` and each of `likelihood_params` (see above).
    likelihood : Callable, default pm.Normal
        The PyMC distribution class used to build the "observations" likelihood.
        Must accept a `mu` kwarg, plus whatever names are listed in
        `likelihood_params`.
    likelihood_params : list[str], optional
        Names of the likelihood's kwargs, beyond `mu`, that `model_function`
        must return in its `extras` dict (e.g. `["sigma"]`, the default, or
        `["sigma", "nu"]` for a Student-T likelihood).

    Attributes:
    -----------
    var_names : list
        Names of the model variables.
    features_names : list
        Names of the features used in the model. Only available after performing
        prior prediction or sampling.
    target_name : str
        Name of the target variable. Only available after sampling.
    trace : InferenceData
        PyMC trace containing samples.
    likelihood : Callable
        The PyMC distribution class used for the likelihood.
    likelihood_params : list[str]
        Names of the likelihood's extra kwargs sourced from `model_function`'s
        `extras`.
    _observations : pm.Distribution
        PyMC distribution defining the model's likelihood.
    _data_dict : dict
        Dictionary containing PyMC Data objects.
    _variables_dict : dict
        Dictionary containing PyMC3 variables for the model parameters.

    Methods:
    --------
    __repr__():
        Returns a string representation of the model's variables, features, and target.
    sample_prior():
        Samples from the prior predictive distribution.
    sample():
        Samples to obtain posterior distributions using the NUTS algorithm.
    sample_posterior_predictive():
        Samples from the posterior predictive distribution.
    get_summary():
        Returns a summary of the posterior distribution. Use the 'group' argument to
        select among 'prior', 'prior_predictive', 'posterior', or
        'posterior_predictive'.
    get_loo_score():
        Computes the leave-one-out cross-validation (LOO) score.
    save_model(Path):
        Save the model to the required path. It will contain 3 traces
        and a json file.
    load_model(Path):
        Load the model traces and json file contained in the desired path and build
        the pymc model.
        If the loaded model function is not in bayesbuilding.model, it must
        be provided separately and the pymc model must be build using the build_model()
        method.
    plot_dist_comparison(var_names):
        Plot compare prior and posterior distributions of variables and observations
        var_names arguments filter the values to display. default is self.var_names
    """

    def __init__(
        self,
        model_function: Callable = None,
        priors_dict: dict[str:(Callable, dict)] = None,
        likelihood: Callable = pm.Normal,
        likelihood_params: list[str] = None,
    ):
        self.model_function = model_function
        self.priors_dict = priors_dict
        self.features_names = None
        self.target_name = None
        self.var_names = None
        self.model = None
        self.likelihood = likelihood
        self.likelihood_params = (
            likelihood_params if likelihood_params is not None else ["sigma"]
        )
        self.traces = {
            "prior": az.InferenceData(),
            "sampling": az.InferenceData(),
            "posterior": az.InferenceData(),
        }
        self._x_train = None
        self._y_train = None
        self._observations = None
        self._data_dict = None
        self._variables_dict = None
        self._extras_dict = None

        self.build_model()

    def __repr__(self):
        string_out = """"""
        string_out += """=== Variables names : \n"""
        if self.var_names is not None:
            for var in self.var_names:
                string_out += f"""- {var} \n"""
        string_out += """\n"""

        string_out += """=== Features names: \n"""
        if self.features_names is not None:
            for feat in self.features_names:
                string_out += f"""- {feat} \n"""
        string_out += """\n"""

        string_out += """=== Target name: \n"""
        if self.target_name is not None:
            string_out += f"""- {self.target_name} \n"""

        return string_out

    def save_model(self, dir_path: Path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for name, traces in self.traces.items():
            traces.to_netcdf((dir_path / f"{name}.nc").as_posix())

        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            to_dump = {
                "model_function": self.model_function,
                "priors_dict": self.priors_dict,
                "likelihood": self.likelihood,
                "likelihood_params": self.likelihood_params,
                "features_names": self.features_names,
                "target_name": self.target_name,
            }
            json.dump(
                to_dump, f, ensure_ascii=False, default=custom_serializer, indent=4
            )

    def load_model(self, dir_path: Path):
        if not os.path.exists(dir_path):
            raise ValueError(f"Provided dir_path : {dir_path} not found")

        for name, _ in self.traces.items():
            self.traces[name] = az.from_netcdf((dir_path / f"{name}.nc").as_posix())

        with open(dir_path / "config.json", encoding="utf-8") as f:
            config_dict = json.load(f)

        for attr, value in config_dict.items():
            setattr(self, attr, value)

        for val in self.priors_dict.values():
            val[0] = getattr(pm, val[0])

        self.likelihood = getattr(pm, self.likelihood)

        if self.model_function is not None:
            try:
                self.model_function = getattr(mods, self.model_function)
            except AttributeError:
                warnings.warn(
                    f"Model function {self.model_function} not found in"
                    f"bayesbuilding.models. Load a model function before running"
                    f"build_model() method"
                )
                self.model_function = None

        self.build_model()

    def build_model(self):
        if self.model_function is not None and self.priors_dict is not None:
            self.model = pm.Model()
            with self.model:
                # Set empty data objects
                self._data_dict = {
                    "x": pm.Data(
                        name="x", value=np.array([[]]), dims=["date", "features"]
                    ),
                    "y": pm.Data(name="y", value=np.array([]), dims=["target"]),
                }

                # Set priors
                self._variables_dict = {
                    name: val[0](**val[1]) for name, val in self.priors_dict.items()
                }

                model_func_output = self.model_function(
                    self._data_dict["x"], self._variables_dict
                )
                if isinstance(model_func_output, tuple):
                    mu_expr, extras = model_func_output
                else:
                    mu_expr, extras = model_func_output, {}

                mu = pm.Deterministic(name="mu", var=mu_expr)

                # Every name in likelihood_params must be provided explicitly by
                # model_function via extras -- no implicit fallback to
                # variables_dict (see class docstring).
                missing = [
                    name for name in self.likelihood_params if name not in extras
                ]
                if missing:
                    raise ValueError(
                        f"model_function must return {missing} in its extras "
                        f"dict for likelihood={self.likelihood.__name__} "
                        f"(got extras keys: {list(extras)})"
                    )

                resolved_params = {}
                for name in self.likelihood_params:
                    value = extras.pop(name)
                    if name in self._variables_dict:
                        # `name` already names a prior/model variable (e.g.
                        # model_function just forwards or indexes
                        # variables_dict["sigma"]) -- reuse that expression
                        # as-is rather than registering a second
                        # pm.Deterministic under the same name, which PyMC
                        # would reject as a duplicate.
                        resolved_params[name] = value
                    else:
                        # A genuinely new quantity (e.g. a heteroscedastic
                        # sigma computed from other priors): expose it as its
                        # own Deterministic so it lands in the trace.
                        resolved_params[name] = pm.Deterministic(name=name, var=value)
                # Remaining extras are diagnostics only, not consumed by the
                # likelihood (e.g. an internal 'heat' term).
                self._extras_dict = {
                    name: pm.Deterministic(name=name, var=expr)
                    for name, expr in extras.items()
                }

                self._observations = self.likelihood(
                    name="observations",
                    mu=mu,
                    observed=self._data_dict["y"],
                    shape=self._data_dict["x"].shape[0],
                    **resolved_params,
                )
            self.var_names = list(self.priors_dict.keys())

    def sample_prior(
        self,
        samples: int = 500,
        x: pd.DataFrame = None,
        var_names=None,
        sample_kwargs=None,
    ):
        if sample_kwargs is None:
            sample_kwargs = {}
        with self.model:
            if x is not None:
                pm.set_data({"x": x})
                self.features_names = list(x.columns)

            self.traces["prior"] = pm.sample_prior_predictive(
                draws=samples, var_names=var_names, **sample_kwargs
            )

    def sample(
        self,
        y: pd.Series,
        x: pd.DataFrame = None,
        draws: int = 1000,
        tune: int = 1000,
        chains=None,
        sample_kwargs=None,
    ):
        if sample_kwargs is None:
            sample_kwargs = {}

        with self.model:
            if x is not None:
                pm.set_data({"x": x})
                self.features_names = list(x.columns)

            pm.set_data({"y": y})
            self.target_name = y.name

            self.traces["sampling"] = pm.sample(
                draws=draws, tune=tune, chains=chains, **sample_kwargs
            )

        self.sample_posterior_predictive()

        self._x_train = x
        self._y_train = y

    def sample_posterior_predictive(self, x: pd.DataFrame = None, sample_kwargs=None):
        if self.traces["sampling"] is None:
            raise ValueError(
                "No posterior trace available. Perform sampling before"
                "sampling from posterior"
            )

        if sample_kwargs is None:
            sample_kwargs = {}

        with self.model:
            if x is not None:
                pm.set_data({"x": x})

            self.traces["posterior"] = pm.sample_posterior_predictive(
                trace=self.traces["sampling"],
                var_names=["observations"],
                **sample_kwargs,
            )

    # Maps a self.traces key to the arviz-internal group name it actually
    # holds the samples under (see sample_prior/sample/sample_posterior_predictive).
    _TRACE_TO_ARVIZ_GROUP = {
        "prior": "prior",
        "sampling": "posterior",
        "posterior": "posterior_predictive",
    }

    def get_summary(
        self, group: str = None, var_names=None, filter_vars=None, summary_kwargs=None
    ):
        if group not in self.traces.keys():
            raise ValueError(
                f"Unknown group {group} choose one of {self.traces.keys()}"
            )
        # round_to="none" keeps the returned columns numeric (arviz's default
        # "auto" rounding formats them as display strings instead), overridable
        # via summary_kwargs.
        merged_summary_kwargs = {"round_to": "none", **(summary_kwargs or {})}

        return az.summary(
            data=self.traces[group],
            var_names=var_names,
            filter_vars=filter_vars,
            group=self._TRACE_TO_ARVIZ_GROUP[group],
            **merged_summary_kwargs,
        )

    def get_loo_score(self, loo_kwargs=None):
        if self.traces["sampling"] is None:
            raise ValueError(
                "Sampling trace is not available, perform sampling "
                "before checking loo score"
            )
        if loo_kwargs is None:
            loo_kwargs = {}
        with self.model:
            pm.set_data({"x": self._x_train, "y": self._y_train})
            pm.compute_log_likelihood(self.traces["sampling"])
        return az.loo(data=(self.traces["sampling"]), **loo_kwargs)

    def score(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        score_function: Callable = r2_score,
        sample_kwargs: dict = None,
        resample_rule: str = None,
    ):
        """Score a fitted model's out-of-sample posterior predictive against `y`.

        If `resample_rule` is given (e.g. "W"), each posterior predictive daily
        sample path and `y` are summed into that resolution's bins before scoring,
        instead of resampling `x` and re-evaluating the model -- the forward model
        may be nonlinear in the driving variable (e.g. a change-point's
        max(driving_var - tau, 0)), so the two are not equivalent.
        """
        if sample_kwargs is None:
            sample_kwargs = {}

        self.sample_posterior_predictive(x=x, sample_kwargs=sample_kwargs)
        post_trace = self.traces["posterior"].posterior_predictive["observations"]
        flattened_trace = np.array(post_trace).reshape(-1, post_trace.shape[-1])

        if resample_rule is not None:
            flattened_trace, y = _resample_samples(flattened_trace, y, resample_rule)

        scores_array = np.array(
            [score_function(y, sample) for sample in flattened_trace]
        )
        return {"mean_score": np.mean(scores_array), "sd_score": np.std(scores_array)}

    def plot_dist_comparison(self, var_names: list[str] = None, plot_dist_kwargs=None):
        if var_names is None:
            var_names = self.var_names
        if plot_dist_kwargs is None:
            plot_dist_kwargs = {}
        temp = DataTree()
        temp["posterior"] = self.traces["sampling"]["posterior"]
        temp["prior"] = self.traces["prior"]["prior"]
        return az.plot_prior_posterior(temp, var_names=var_names, **plot_dist_kwargs)
