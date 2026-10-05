"""Statistical process control charts on a fitted candidate's residuals.

X-bar (Individuals), EWMA and CUSUM statistics computed from the posterior
predictive of ``mu`` and ``sigma`` (see :func:`bayesbuilding.training.
sample_mu_and_observations`), so control limits follow the model's own --
possibly heteroscedastic -- noise scale. The ``*_control_stats`` functions take
plain numpy/pandas inputs (draws already flattened across chains) so they can be
unit-tested without running any MCMC; the ``plot_*_chart`` functions draw them
with plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go


def flatten_predictive_samples(draws) -> np.ndarray:
    """(chain, draw, time) or already-(samples, time), ndarray or
    xarray.DataArray -> (samples, time) ndarray.

    Same positional convention as ``bayesbuilding.plotting._flatten_chains``:
    never indexes by a dim name (PyMC/arviz auto-names an undeclared-``dims``
    axis like "observations_dim_0", which carries no semantic meaning), just
    collapses every leading axis into one samples axis positionally.
    """
    arr = np.asarray(draws)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def xbar_control_stats(
    y_true: pd.Series, mu_draws, sigma_draws, L: float = 1.96
) -> dict:
    """Classical Individuals/X control-chart statistic and limits for the
    residual ``mesure - modèle``, with the model's own (possibly
    heteroscedastic) noise scale standing in for the usual historical
    process sigma.

    ``mu_draws``/``sigma_draws`` are the posterior predictive of the
    deterministic mean and the likelihood's noise scale (e.g. from
    ``bayesbuilding.training.sample_mu_and_observations`` -- NOT the noisy
    ``observations`` themselves, which would double-count noise: once in the
    residual, once in the limits). Reduced across draws to one point
    estimate per day: ``mu_hat = median(mu)``, ``sigma_hat = mean(sigma)``.

    ``residual[t] = y_true[t] - mu_hat[t]`` is then a single trajectory
    (not one per draw) -- comparing a measurement to a *known*/calibrated
    target and a *known* noise scale is exactly the classical control-chart
    setup, unlike a credible band that would re-propagate the model's own
    parameter uncertainty into the limits on every point. ``sigma_hat`` is
    per-day (not pooled into one scalar for the whole period): for a model
    with a heteroscedastic noise term (e.g. a FormulaModel `sigma` formula
    ``"sqrt(s0[occ]**2 + (s1*sigmoid((tau[occ] - text)/1.5))**2)"`` growing
    with heating activity), the limits should genuinely narrow/widen with it
    instead of being forced flat.

    Returns a dict with ``residual``, ``mu_hat``, ``sigma_hat``, ``ucl``
    (``L*sigma_hat``), ``lcl`` (``-ucl``), and ``out_of_control`` (bool array,
    ``residual`` outside ``[lcl, ucl]``).
    """
    mu_hat = np.median(flatten_predictive_samples(mu_draws), axis=0)
    sigma_hat = np.mean(flatten_predictive_samples(sigma_draws), axis=0)
    y = np.asarray(y_true, dtype=float)
    residual = y - mu_hat

    ucl = L * sigma_hat
    lcl = -ucl
    out_of_control = (residual > ucl) | (residual < lcl)

    return {
        "residual": residual,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control": out_of_control,
    }


def standardized_residuals(y_true: pd.Series, mu_draws, sigma_draws) -> np.ndarray:
    """Per-day standardized residual (z-score) ``(y_true - mu_ref) / sig_ref``,
    where ``mu_ref``/``sig_ref`` are the same per-day ``mu_hat``/``sigma_hat``
    point estimates computed by :func:`xbar_control_stats` -- reused rather
    than recomputed, so both stay in sync -- but rescaled to a unitless z
    instead of the raw ``residual``/``ucl``/``lcl`` that function returns.

    Meant as an M&V follow-up diagnostic once a candidate is calibrated on a
    train period and applied forward to a verification/monitoring period: a
    bimodal histogram of ``z`` signals a mixture of regimes rather than a
    simple level shift; a scatter of ``z`` against an explanatory variable
    (e.g. outdoor temperature) that clusters away from zero at one end
    signals a localized, unmodeled effect; a run of consecutive out-of-range
    days in ``z`` over time signals a dated event (drift, fault) rather than
    i.i.d. noise.
    """
    stats = xbar_control_stats(y_true, mu_draws, sigma_draws)
    return stats["residual"] / stats["sigma_hat"]


def ewma_control_stats(
    y_true: pd.Series,
    mu_draws,
    sigma_draws,
    lam: float = 0.2,
    L: float = 1.96,
    n_confirm: int = 3,
    n_reset: int = None,
    reset_epsilon: float = 0.5,
) -> dict:
    """Classical EWMA control-chart statistic and limits on the same
    residual as :func:`xbar_control_stats` (one trajectory, not a per-draw
    ensemble).

    The EWMA recursion ``Z[0] = lam*residual[0]``, ``Z[t] = lam*residual[t] +
    (1-lam)*Z[t-1]`` is applied once, to the single residual trajectory.
    Its limits use the standard Montgomery formula generalized to a
    time-varying noise scale ``sigma_hat[t]`` (valid here because, unlike an
    ensemble of posterior draws, day-to-day residuals against a *fixed*
    ``mu_hat`` are independent given that estimate -- no shared per-draw
    parameters correlating them across days):
    ``Var(Z[t]) = lam**2 * sum_{j<=t} (1-lam)**(2*(t-j)) * sigma_hat[j]**2``,
    computed via the equivalent recursion
    ``Var(Z[t]) = lam**2*sigma_hat[t]**2 + (1-lam)**2*Var(Z[t-1])``. This
    reproduces the classic control-chart look (limits starting tight near
    the first sample and widening towards an asymptote) instead of a
    per-day credible band.

    ``out_of_control`` is, by default (``n_reset=None``), the purely
    instantaneous test ``ewma[t]`` outside ``[lcl[t], ucl[t]]`` -- no memory
    of past points. That's a poor fit for a transient event (the error rises
    then drops abruptly back to 0): the EWMA's own inertia keeps it pinned
    outside the limits for several steps after the anomaly is already gone,
    while a plain "N steps out -> reset" would risk instead masking a real,
    sustained drift by prematurely re-labelling it "back to normal".

    Passing ``n_reset`` (not None) switches on a 3-state alternative instead,
    layered on top of the same ``ewma`` trajectory:

    - NORMAL: default state.
    - ALARME (confirmed): reached once ``ewma`` sits outside its limits for
      ``n_confirm`` consecutive steps in a row (rules out a single noisy
      point).
    - Once in ALARME, the *raw* residual (not the EWMA) is watched instead:
      ``|residual[t]| < reset_epsilon * sigma_hat[t]`` for ``n_reset``
      consecutive steps confirms the anomaly has genuinely cleared, and
      resets the EWMA memory (``ewma[t] <- 0``, its variance recursion
      restarted at ``lam**2*sigma_hat[t]**2`` as if ``t`` were a fresh first
      point) before returning to NORMAL. If the residual never settles back
      under the threshold, ALARME simply persists to the end of the series
      -- a sustained drift never gets reset away.

    ``reset_epsilon`` is expressed in multiples of ``sigma_hat[t]`` (like
    ``L`` here, and ``k``/``h`` for :func:`cusum_control_stats`), so it
    stays meaningful across sites/candidates with different residual scales.
    ``n_confirm`` is only used when ``n_reset`` is given.

    Returns a dict with ``ewma``, ``ucl``, ``lcl``, ``out_of_control`` (bool
    array -- aliases ``alarm`` when the reset logic is active, otherwise the
    instantaneous test described above), ``alarm`` (the persistent confirmed-
    alarm state; equals ``out_of_control`` when ``n_reset`` is None),
    ``reset_points`` (bool array, True at the exact step a reset fired --
    all False when ``n_reset`` is None), plus the underlying
    ``residual``/``sigma_hat`` from :func:`xbar_control_stats`.
    """
    xbar = xbar_control_stats(y_true, mu_draws, sigma_draws, L=L)
    residual, sigma_hat = xbar["residual"], xbar["sigma_hat"]
    n = residual.shape[0]
    reset_active = n_reset is not None

    ewma = np.empty(n)
    var_z = np.empty(n)
    alarm = np.zeros(n, dtype=bool)
    reset_points = np.zeros(n, dtype=bool)

    ewma[0] = lam * residual[0]
    var_z[0] = lam**2 * sigma_hat[0] ** 2

    state = "normal"
    consec_out = 0
    consec_normal_err = 0

    def _out_of_limits(z, v):
        limit = L * np.sqrt(v)
        return z > limit or z < -limit

    if reset_active:
        consec_out = 1 if _out_of_limits(ewma[0], var_z[0]) else 0
        if consec_out >= n_confirm:
            state = "alarm"
        alarm[0] = state == "alarm"

    for t in range(1, n):
        ewma[t] = lam * residual[t] + (1 - lam) * ewma[t - 1]
        var_z[t] = lam**2 * sigma_hat[t] ** 2 + (1 - lam) ** 2 * var_z[t - 1]

        if reset_active:
            if state == "normal":
                out_t = _out_of_limits(ewma[t], var_z[t])
                consec_out = consec_out + 1 if out_t else 0
                if consec_out >= n_confirm:
                    state = "alarm"
                    consec_normal_err = 0
            else:  # state == "alarm"
                if abs(residual[t]) < reset_epsilon * sigma_hat[t]:
                    consec_normal_err += 1
                else:
                    consec_normal_err = 0
                if consec_normal_err >= n_reset:
                    state = "normal"
                    consec_out = 0
                    ewma[t] = 0.0
                    var_z[t] = lam**2 * sigma_hat[t] ** 2
                    reset_points[t] = True
            alarm[t] = state == "alarm"

    ucl = L * np.sqrt(var_z)
    lcl = -ucl

    if reset_active:
        out_of_control = alarm
    else:
        out_of_control = (ewma > ucl) | (ewma < lcl)
        alarm = out_of_control

    return {
        "ewma": ewma,
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control": out_of_control,
        "alarm": alarm,
        "reset_points": reset_points,
        "residual": residual,
        "sigma_hat": sigma_hat,
    }


def cusum_control_stats(
    y_true: pd.Series,
    mu_draws,
    sigma_draws,
    k: float = 0.5,
    h: float = 5.0,
    L: float = 1.96,
) -> dict:
    """Classical two-sided tabular CUSUM (Page) on the standardized residual
    ``z = residual / sigma_hat`` -- same ``mu_hat``/``sigma_hat`` point
    estimates as :func:`xbar_control_stats`/:func:`ewma_control_stats` (one
    trajectory, not a per-draw ensemble).

    ``k`` (the reference/slack value, standard default 0.5 sigma -- half the
    smallest shift worth detecting) and ``h`` (the decision interval,
    standard default 4-5 sigma) are both already in standardized units, so
    the decision limits are the constant ``h``/``-h`` regardless of any
    heteroscedasticity in ``sigma_hat`` -- unlike xbar/ewma's limits, which
    track ``sigma_hat`` directly in the residual's own (unstandardized)
    units.

    Running sums (reset at 0, never negative)::

        cusum_pos[t] = max(0, cusum_pos[t-1] + z[t] - k)
        cusum_neg[t] = max(0, cusum_neg[t-1] - z[t] - k)

    Out of control the first time either sum exceeds ``h`` -- ``cusum_pos``
    signals a sustained upward shift, ``cusum_neg`` a sustained downward one.
    A CUSUM is more sensitive than an X or EWMA chart to a small, sustained
    drift (as opposed to a single large outlier), since it accumulates
    evidence across days rather than resetting each step.

    Returns a dict with ``cusum_pos``, ``cusum_neg``, ``h``,
    ``out_of_control`` (bool array, either sum > ``h``), plus the underlying
    ``residual``/``sigma_hat`` from :func:`xbar_control_stats`.
    """
    xbar = xbar_control_stats(y_true, mu_draws, sigma_draws, L=L)
    residual, sigma_hat = xbar["residual"], xbar["sigma_hat"]
    z = residual / sigma_hat
    n = z.shape[0]

    cusum_pos = np.empty(n)
    cusum_neg = np.empty(n)
    prev_pos = prev_neg = 0.0
    for t in range(n):
        prev_pos = max(0.0, prev_pos + z[t] - k)
        prev_neg = max(0.0, prev_neg - z[t] - k)
        cusum_pos[t] = prev_pos
        cusum_neg[t] = prev_neg

    out_of_control = (cusum_pos > h) | (cusum_neg > h)

    return {
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "h": h,
        "out_of_control": out_of_control,
        "residual": residual,
        "sigma_hat": sigma_hat,
    }


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple]:
    """Positions ``(start_i, end_i)`` (inclusive) of each contiguous run of
    ``True`` in ``mask``. Simpler than
    ``bayesbuilding.plotting._boolean_blocks`` -- no single-point padding is
    needed here, since a confirmed EWMA alarm run is, by construction, at
    least ``n_confirm`` steps long (see
    ``ewma_control_stats``).
    """
    runs = []
    run_start = None
    for i, val in enumerate(mask):
        if val and run_start is None:
            run_start = i
        elif not val and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(mask) - 1))
    return runs


def _plot_control_chart(
    index,
    stat: np.ndarray,
    ucl: np.ndarray,
    lcl: np.ndarray,
    out_of_control: np.ndarray,
    y_label: str,
    title: str,
    series_name: str,
    alarm: np.ndarray = None,
    reset_points: np.ndarray = None,
) -> go.Figure:
    """Shared renderer for :func:`plot_xbar_chart`/:func:`plot_ewma_chart`,
    in the classical SPC style (Minitab et al.): a single trajectory (not a
    per-draw ensemble/credible band), a solid center line at 0, solid red
    UCL/LCL curves (flat for a homoscedastic sigma, following the model's
    own heteroscedasticity otherwise -- see
    ``xbar_control_stats``/``ewma_control_stats``),
    and red markers directly on the trajectory's own out-of-limits points --
    the point genuinely sits outside ``[lcl, ucl]`` there, so it visibly
    pokes past the red lines instead of a marker that's always contained
    inside its own band by construction.

    ``alarm``/``reset_points`` are optional (only :func:`plot_ewma_chart`'s
    3-state reset logic passes them, see
    ``ewma_control_stats``) -- ``alarm``'s ``True``
    runs are shaded pale red in the background (the *confirmed*, persistent
    alarm period, as opposed to ``out_of_control``'s pointwise markers), and
    ``reset_points`` gets its own green marker where the EWMA was reset back
    to 0 after the error genuinely returned to normal.
    """
    index = np.asarray(index)
    fig = go.Figure()

    if alarm is not None:
        for start_i, end_i in _contiguous_true_runs(alarm):
            fig.add_vrect(
                x0=index[start_i],
                x1=index[end_i],
                fillcolor="rgba(239,85,59,0.12)",
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(x=index, y=ucl, mode="lines", line=dict(color="#EF553B", width=1.5))
    )
    fig.add_trace(
        go.Scatter(x=index, y=lcl, mode="lines", line=dict(color="#EF553B", width=1.5))
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=stat,
            mode="lines+markers",
            line=dict(color="#2a78d6"),
            marker=dict(size=5),
            name=series_name,
        )
    )
    if out_of_control.any():
        fig.add_trace(
            go.Scatter(
                x=index[out_of_control],
                y=stat[out_of_control],
                mode="markers",
                marker=dict(
                    color="#EF553B", size=10, symbol="circle-open", line=dict(width=2)
                ),
                name="hors contrôle",
            )
        )
    if reset_points is not None and reset_points.any():
        fig.add_trace(
            go.Scatter(
                x=index[reset_points],
                y=stat[reset_points],
                mode="markers",
                marker=dict(
                    color="#2ca02c", size=10, symbol="circle-open", line=dict(width=2)
                ),
                name="reset EWMA",
            )
        )

    fig.add_hline(y=0, line_color="black")
    fig.add_annotation(
        x=index[-1],
        y=ucl[-1],
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font=dict(color="#EF553B"),
        text=f"LCS={ucl[-1]:.3g}",
    )
    fig.add_annotation(
        x=index[-1],
        y=lcl[-1],
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font=dict(color="#EF553B"),
        text=f"LCI={lcl[-1]:.3g}",
    )

    n_out = int(out_of_control.sum())
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        text=(
            f"Points hors contrôle : {n_out}/{len(stat)} "
            f"({n_out / len(stat) * 100:.1f}%)"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        yaxis_title=y_label,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def plot_xbar_chart(
    measure_ts: pd.Series,
    mu_prediction,
    sigma_prediction,
    L: float = 1.96,
    y_label: str = None,
    title: str = None,
) -> go.Figure:
    """Carte de contrôle X (individus) classique sur le résidu (mesure -
    modèle), centrée en 0.

    ``mu_prediction``/``sigma_prediction`` sont la posterior predictive de la
    moyenne déterministe et de l'échelle de bruit du modèle (voir
    ``bayesbuilding.training.sample_mu_and_observations``) -- PAS les observations
    bruitées elles-mêmes, sinon le bruit serait compté deux fois (une fois
    dans le résidu, une fois dans les limites). Les limites viennent du
    sigma du modèle (voir ``xbar_control_stats``),
    éventuellement variable dans le temps si le modèle est hétéroscédastique
    -- pas une bande de crédibilité recalculée à chaque point.
    """
    stats = xbar_control_stats(measure_ts, mu_prediction, sigma_prediction, L=L)
    return _plot_control_chart(
        measure_ts.index,
        stats["residual"],
        stats["ucl"],
        stats["lcl"],
        stats["out_of_control"],
        y_label,
        title or "Carte de contrôle X (résidu)",
        "résidu (mesure - modèle)",
    )


def plot_ewma_chart(
    measure_ts: pd.Series,
    mu_prediction,
    sigma_prediction,
    lam: float = 0.2,
    L: float = 1.96,
    n_confirm: int = 3,
    n_reset: int = None,
    reset_epsilon: float = 0.5,
    y_label: str = None,
    title: str = None,
) -> go.Figure:
    """Carte de contrôle EWMA classique sur le résidu (mesure - modèle),
    centrée en 0 -- même résidu que :func:`plot_xbar_chart`, lissé par la
    récursion EWMA, avec des limites qui s'élargissent depuis le premier
    point jusqu'à une asymptote (voir
    ``ewma_control_stats``), comme une carte EWMA
    classique (Minitab et al.).

    ``n_confirm``/``n_reset``/``reset_epsilon`` sont transmis tels quels à
    :func:`bayesbuilding.control_charts.ewma_control_stats` -- par défaut
    (``n_reset=None``) rien ne change par rapport à la carte EWMA classique.
    En passant ``n_reset``, la logique à 3 états (NORMAL / ALARME confirmée /
    retour à la normale) s'active : la période d'alarme confirmée est alors
    ombrée en rouge pâle en arrière-plan (au lieu des simples marqueurs
    ponctuels de ``out_of_control``), et chaque reset de l'EWMA (l'erreur
    brute est réellement revenue sous ``reset_epsilon`` sigma) est marqué
    d'un cercle vert.
    """
    stats = ewma_control_stats(
        measure_ts,
        mu_prediction,
        sigma_prediction,
        lam=lam,
        L=L,
        n_confirm=n_confirm,
        n_reset=n_reset,
        reset_epsilon=reset_epsilon,
    )
    return _plot_control_chart(
        measure_ts.index,
        stats["ewma"],
        stats["ucl"],
        stats["lcl"],
        stats["out_of_control"],
        y_label,
        title or f"Carte de contrôle EWMA (résidu, λ={lam})",
        "EWMA résidu",
        alarm=stats["alarm"] if n_reset is not None else None,
        reset_points=stats["reset_points"] if n_reset is not None else None,
    )


def plot_cusum_chart(
    measure_ts: pd.Series,
    mu_prediction,
    sigma_prediction,
    k: float = 0.5,
    h: float = 5.0,
    L: float = 1.96,
    y_label: str = None,
    title: str = None,
) -> go.Figure:
    """Carte de contrôle CUSUM tabulaire classique sur le résidu standardisé
    (mesure - modèle)/sigma -- plus sensible qu'une carte X ou EWMA à une
    petite dérive soutenue (voir
    ``cusum_control_stats``).

    Tracée comme ``cusum_pos - cusum_neg`` : au plus une des deux sommes est
    active à la fois en pratique (l'autre est retombée à 0 par construction),
    donc cette différence signée reproduit la même trajectoire "centrée en 0,
    limites symétriques" que les cartes X/EWMA, via le même renderer partagé
    (:func:`_plot_control_chart`) -- ``h``/``-h`` sont des limites constantes
    (déjà en unités sigma), contrairement aux limites de
    :func:`plot_xbar_chart`/:func:`plot_ewma_chart` qui suivent
    ``sigma_hat`` dans les unités brutes du résidu.
    """
    stats = cusum_control_stats(
        measure_ts, mu_prediction, sigma_prediction, k=k, h=h, L=L
    )
    signed = stats["cusum_pos"] - stats["cusum_neg"]
    h_line = np.full_like(signed, stats["h"], dtype=float)
    return _plot_control_chart(
        measure_ts.index,
        signed,
        h_line,
        -h_line,
        stats["out_of_control"],
        y_label,
        title or f"Carte de contrôle CUSUM (résidu standardisé, k={k}, h={h})",
        "CUSUM (C+ - C-)",
    )
