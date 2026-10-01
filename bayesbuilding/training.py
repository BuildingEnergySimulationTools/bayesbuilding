"""Fit, score and compare the candidates of a :class:`~bayesbuilding.candidates.
BayesConfig`."""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from bayesbuilding.candidates import BayesConfig, CandidateConfig, build_priors_dict
from bayesbuilding.wrapper import PymcWrapper, r2_score, resample_samples


def nmbe(y_true, y_pred) -> float:
    """Normalized Mean Bias Error [%] (same definition as corrai's)."""
    return float(np.sum(y_pred - y_true) / np.sum(y_true) * 100)


def cv_rmse(y_true, y_pred) -> float:
    """Coefficient of Variation of the RMSE [%] (same definition as corrai's,
    i.e. ASHRAE Guideline 14's, with n - 1 degrees of freedom)."""
    n = len(y_true)
    return float(
        np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1)) / np.mean(y_true) * 100
    )


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


METRICS = {"r2": r2_score, "nmbe": nmbe, "cv_rmse": cv_rmse, "mae": mae}


def build_candidate_xy(
    df: pd.DataFrame, target: str, candidate: CandidateConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Select the columns of ``df`` a candidate needs, in its formula's input order.

    Raises KeyError early (naming the candidate and the missing keys) if
    ``candidate.inputs`` doesn't map every input symbol of the formula, instead
    of letting PyMC fail opaquely further down.
    """
    required = candidate.model.inputs
    missing = [k for k in required if k not in candidate.inputs]
    if missing:
        raise KeyError(
            f"Candidate {candidate.name!r}: missing input mapping for {missing}"
        )
    x = df[[candidate.inputs[k] for k in required]]
    y = df[target]
    return x, y


def fit_candidate(
    candidate: CandidateConfig,
    x: pd.DataFrame,
    y: pd.Series,
    artifact_path: Path | None,
    priors: dict = None,
) -> PymcWrapper:
    """Build, sample (prior + posterior) and persist a PymcWrapper for one candidate.

    The likelihood is a Normal truncated at ``candidate.model.lower`` (an energy
    consumption cannot be negative), or a plain Normal when ``lower`` is None.

    ``priors`` defaults to ``candidate.priors`` (the config's declared priors).
    Passing a different priors spec (e.g. from :func:`~bayesbuilding.candidates.
    posterior_as_priors`) lets the same candidate be re-conditioned on a
    different period, starting from an already-informed prior instead of the
    config's original one -- a sequential Bayesian update.

    The wrapper is saved to ``artifact_path / "trace"`` unless ``artifact_path``
    is None.
    """
    if candidate.model.lower is None:
        likelihood, likelihood_params = pm.Normal, ["sigma"]
    else:
        likelihood, likelihood_params = pm.TruncatedNormal, ["sigma", "lower"]
    wrapper = PymcWrapper(
        model_function=candidate.model,
        priors_dict=build_priors_dict(
            priors if priors is not None else candidate.priors
        ),
        likelihood=likelihood,
        likelihood_params=likelihood_params,
    )
    wrapper.sample_prior(
        samples=candidate.prior_draws,
        x=x,
        sample_kwargs={"random_seed": candidate.random_seed},
    )
    sample_kwargs = {"random_seed": candidate.random_seed}
    if sys.platform == "win32":
        # PyMC's multi-chain sampling multiprocesses via "spawn" on Windows,
        # which re-imports the caller's top-level code in each child process --
        # this crashes scripts written to run cell-by-cell without an
        # `if __name__ == "__main__":` guard. Run chains sequentially instead.
        sample_kwargs["cores"] = 1
    wrapper.sample(
        x=x,
        y=y,
        draws=candidate.draws,
        tune=candidate.tune,
        chains=candidate.chains,
        sample_kwargs=sample_kwargs,
    )
    if artifact_path is not None:
        wrapper.save_model(Path(artifact_path) / "trace")
    return wrapper


def sample_mu_and_observations(
    wrapper: PymcWrapper, x: pd.DataFrame
) -> az.InferenceData:
    """Posterior predictive of the deterministic ``mu``, the likelihood's noise
    scale ``sigma`` and the noisy ``observations`` for arbitrary ``x``, in one
    call.

    ``PymcWrapper.sample_posterior_predictive`` only samples ``observations``;
    ``mu``/``sigma`` are needed to separate parametric from noise uncertainty
    and to derive control-chart limits from the model's own (possibly
    heteroscedastic) noise scale (see :mod:`bayesbuilding.control_charts`).
    Does not mutate ``wrapper.traces["posterior"]``.
    """
    with wrapper.model:
        pm.set_data({"x": x})
        return pm.sample_posterior_predictive(
            trace=wrapper.traces["sampling"], var_names=["mu", "sigma", "observations"]
        )


def flatten_regime_vars(
    group: xr.Dataset, var_names: list[str]
) -> dict[str, np.ndarray]:
    """Flatten each named variable of a posterior/prior xarray group to a 1D
    array of draws, ready for e.g. a KDE.

    A variable indexed by a two-level regime -- shape 2 on a ``"{var}_dim_0"``
    dim, e.g. ``g[occ]`` -- is exploded into ``"{var}0"``/``"{var}1"``. A
    ``var_names`` entry absent from ``group`` (e.g. a candidate whose priors
    don't cover every var asked for) is silently skipped rather than raising.
    """
    flattened = {}
    for name in var_names:
        if name not in group.data_vars:
            continue
        da = group[name]
        dim = f"{name}_dim_0"
        if dim in da.dims and da.sizes[dim] == 2:
            for i in (0, 1):
                flattened[f"{name}{i}"] = da.isel({dim: i}).to_numpy().reshape(-1)
        else:
            flattened[name] = da.to_numpy().reshape(-1)
    return flattened


def evaluate_candidate(
    wrapper: PymcWrapper, x_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """LOO (elpd, se) and scores of a fitted candidate on ``(x_test, y_test)``.

    R2/NMBE/CV(RMSE)/MAE are computed both at native (daily) resolution and
    aggregated to calendar weeks/months ("_d"/"_W"/"_ME" suffixes), as the mean
    and std over posterior predictive sample paths. Weekly/monthly scores sum
    each daily sample path (and y_test) into bins before scoring, rather than
    resampling the driving variable and re-evaluating the model -- a nonlinear
    forward model (e.g. max(driving_var - tau, 0)) isn't equivalent under the
    two orderings. LOO is intrinsically pointwise (daily) and computed on the
    training data.

    The posterior predictive is sampled once and reused for every resolution
    and metric.
    """
    loo = wrapper.get_loo_score()
    results = {
        "elpd_loo": float(loo.elpd),
        "elpd_loo_se": float(loo.se),
    }

    wrapper.sample_posterior_predictive(x=x_test)
    post_trace = wrapper.traces["posterior"].posterior_predictive["observations"]
    flattened_trace_full = np.array(post_trace).reshape(-1, post_trace.shape[-1])

    for suffix, resample_rule in (("d", None), ("W", "W"), ("ME", "ME")):
        if resample_rule is None:
            flattened_trace, y_eval = flattened_trace_full, y_test
        else:
            flattened_trace, y_eval = resample_samples(
                flattened_trace_full, y_test, resample_rule
            )
        y_eval = np.asarray(y_eval)
        for name, metric_func in METRICS.items():
            scores_array = np.array(
                [metric_func(y_eval, sample) for sample in flattened_trace]
            )
            results[f"{name}_mean_{suffix}"] = float(np.mean(scores_array))
            results[f"{name}_sd_{suffix}"] = float(np.std(scores_array))

    return results


def compare_candidates(results: dict[str, dict]) -> pd.DataFrame:
    """Assemble a comparison table across candidates, best elpd_loo first."""
    return pd.DataFrame(results).T.sort_values("elpd_loo", ascending=False)


def train_candidates(
    config: BayesConfig,
    df: pd.DataFrame,
    out_dir: Path | str,
    df_test: pd.DataFrame | None = None,
    candidates: list[CandidateConfig] | None = None,
    comparison_filename: str | None = "comparison_train.csv",
) -> tuple[dict[str, PymcWrapper], pd.DataFrame, dict[str, str]]:
    """Fit and score every candidate, then build the comparison table.

    Each candidate is fitted on ``df`` and saved to ``out_dir/<name>/trace``,
    then scored on ``df_test`` (in sample on ``df`` when None). A candidate
    that fails (sampler crash, missing input column...) is recorded in the
    returned errors and skipped instead of aborting the loop.

    :param candidates: subset to train, defaults to ``config.candidates``.
    :param comparison_filename: file written in ``out_dir`` with the comparison
        table, or None to write nothing.
    :return: ``(wrappers, comparison, errors)``: fitted wrappers by name, the
        :func:`compare_candidates` table, and error messages by name.
    """
    out_dir = Path(out_dir)
    candidates = config.candidates if candidates is None else candidates
    wrappers, results, errors = {}, {}, {}
    for candidate in candidates:
        print(f"=== candidate: {candidate.name} ===")
        try:
            x, y = build_candidate_xy(df, config.target, candidate)
            artifact_path = out_dir / candidate.name
            artifact_path.mkdir(parents=True, exist_ok=True)
            wrapper = fit_candidate(candidate, x, y, artifact_path)
            if df_test is None:
                x_eval, y_eval = x, y
            else:
                x_eval, y_eval = build_candidate_xy(df_test, config.target, candidate)
            results[candidate.name] = evaluate_candidate(wrapper, x_eval, y_eval)
            wrappers[candidate.name] = wrapper
        except Exception as exc:
            print(f"  [failed: {exc}]")
            errors[candidate.name] = str(exc)

    comparison = compare_candidates(results) if results else pd.DataFrame()
    if comparison_filename is not None and not comparison.empty:
        comparison.to_csv(out_dir / comparison_filename)
    return wrappers, comparison, errors
