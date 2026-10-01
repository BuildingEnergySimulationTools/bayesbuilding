"""JSON configuration of candidate models to fit and compare.

A config file looks like::

    {
      "config_id": 69370,
      "target": "heating_energy",
      "feature_pipe": {...},
      "candidates": [
        {
          "name": "dt_occ",
          "model": {
              "likelihood": "TruncatedNormal",
              "mu": "g[occ]*dt - fs[occ]*rad",
              "params": {"sigma": "s0[occ]", "lower": 0.0}
          },
          "inputs": {"dt": "<column>", "rad": "<column>", "occ": "<column>"},
          "priors": {
              "g": {"dist": "HalfNormal", "kwargs": {"sigma": 10, "shape": 2}},
              ...
          },
          "draws": 4000, "tune": 1000
        }
      ],
      "unused_candidates": []
    }

``model`` is a :class:`~bayesbuilding.formula.FormulaModel` spec: the
likelihood distribution, the formula of its mean ``mu`` and its other
parameters ``params`` (formulas or numbers). ``inputs`` maps
each input symbol of the formula to a real column name, since column names vary
from one site to the next. ``feature_pipe`` is an opaque dict left to the caller
(e.g. a Tide pipe computing the columns once for every candidate).
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pymc as pm

from bayesbuilding.formula import FormulaModel
from bayesbuilding.legacy_formulas import legacy_to_formula
from bayesbuilding.wrapper import PymcWrapper


def build_priors_dict(priors: dict) -> dict:
    """Turn a JSON-serializable priors spec into PymcWrapper's priors_dict.

    Input  : {"g": {"dist": "Normal", "kwargs": {"mu": 100.0, "sigma": 10.0}}, ...}
    Output : {"g": (pm.Normal, {"name": "g", "mu": 100.0, "sigma": 10.0}), ...}
    """
    return {
        name: (getattr(pm, spec["dist"]), {"name": name, **spec["kwargs"]})
        for name, spec in priors.items()
    }


@dataclass
class CandidateConfig:
    """One forward model to fit and compare, within a BayesConfig.

    ``model`` is given as a formula spec dict (or a FormulaModel) and is stored
    as a FormulaModel bound to the prior names, so ``model.inputs`` is the
    ordered list of input symbols. A legacy model name (a string) is still
    accepted and translated with :func:`~bayesbuilding.legacy_formulas.
    legacy_to_formula`, with a DeprecationWarning.
    """

    name: str
    model: dict | FormulaModel
    inputs: dict[str, str]
    priors: dict
    prior_draws: int = 4000
    draws: int = 4000
    tune: int = 1000
    chains: int | None = None
    random_seed: int | None = None

    def __post_init__(self):
        if isinstance(self.model, str):
            warnings.warn(
                f"Candidate {self.name!r}: legacy model name {self.model!r}, "
                "replace it by its formula (see legacy_to_formula)",
                DeprecationWarning,
                stacklevel=3,
            )
            self.model = legacy_to_formula(self.model)
        if isinstance(self.model, dict):
            self.model = FormulaModel.from_dict(self.model)
        try:
            self.model.bind(self.priors)
        except ValueError as e:
            raise ValueError(f"Candidate {self.name!r}: {e}") from e

    def build_wrapper(self, priors: dict = None) -> PymcWrapper:
        """An unsampled PymcWrapper for this candidate, with the likelihood
        declared in ``model``. ``priors`` (a priors spec, e.g. from
        :func:`posterior_as_priors`) defaults to ``self.priors``."""
        return PymcWrapper(
            model_function=self.model,
            priors_dict=build_priors_dict(
                priors if priors is not None else self.priors
            ),
        )

    def to_dict(self) -> dict:
        spec = self.model.to_dict()
        spec.pop("inputs", None)  # derived from the priors on load
        return {
            "name": self.name,
            "model": spec,
            "inputs": dict(self.inputs),
            "priors": self.priors,
            "prior_draws": self.prior_draws,
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "random_seed": self.random_seed,
        }


@dataclass
class BayesConfig:
    """A set of candidate models sharing the same target and feature pipe."""

    target: str
    candidates: list[CandidateConfig]
    config_id: int | str | None = None
    feature_pipe: dict = field(default_factory=dict)
    unused_candidates: list[CandidateConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "BayesConfig":
        raw = dict(raw)
        candidates = [CandidateConfig(**c) for c in raw.pop("candidates")]
        unused = [CandidateConfig(**c) for c in raw.pop("unused_candidates", [])]
        return cls(candidates=candidates, unused_candidates=unused, **raw)

    @classmethod
    def from_json(cls, path: Path | str) -> "BayesConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "target": self.target,
            "feature_pipe": self.feature_pipe,
            "candidates": [c.to_dict() for c in self.candidates],
            "unused_candidates": [c.to_dict() for c in self.unused_candidates],
        }

    def to_json(self, path: Path | str):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_candidate(self, name: str) -> CandidateConfig:
        for candidate in self.candidates + self.unused_candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No candidate named {name!r}")


def posterior_as_priors(candidate: CandidateConfig, wrapper) -> dict:
    """Turn a fitted wrapper's parameter posterior into a new priors spec.

    For each of ``candidate.priors``' variables, keeps the original prior's
    distribution family (and any kwarg other than ``mu``/``sigma``, e.g.
    ``TruncatedNormal``'s ``lower``) but replaces whichever of ``mu``/``sigma``
    the original spec already declared with the posterior sample's empirical
    mean/std -- i.e. a moment-matched parametric refit of "today's posterior"
    usable as "tomorrow's prior" (see :func:`~bayesbuilding.training.
    fit_candidate`'s ``priors`` argument). This is a standard approximation for
    sequential Bayesian updating: PyMC priors must be given as a parametric
    family, so the posterior's exact (possibly non-Gaussian) shape isn't
    preserved, only its first two moments. Only kwargs already present in the
    original spec are touched -- a one-parameter family like ``HalfNormal``
    (kwarg ``sigma`` only, no ``mu``) doesn't get a spurious ``mu`` injected,
    which would break the next ``fit_candidate(..., priors=updated)`` call.
    """
    posterior = wrapper.traces["sampling"].posterior
    updated = {}
    for name, spec in candidate.priors.items():
        samples = posterior[name].to_numpy().reshape(-1)
        moments = {}
        if "mu" in spec["kwargs"]:
            moments["mu"] = float(samples.mean())
        if "sigma" in spec["kwargs"]:
            moments["sigma"] = float(samples.std())
        updated[name] = {
            "dist": spec["dist"],
            "kwargs": {**spec["kwargs"], **moments},
        }
    return updated
