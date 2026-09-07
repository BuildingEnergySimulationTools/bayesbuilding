import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import xarray
from plotly.subplots import make_subplots
from pathlib import Path

sns.set_style("whitegrid")


def _flatten_chains(prediction):
    """Convert an xarray DataArray to ndarray and flatten (chain, draw, time) into
    (samples, time). No-op if `prediction` is already 2D."""
    if isinstance(prediction, xarray.DataArray):
        prediction = np.array(prediction)

    if prediction.ndim > 2:  # Assume it's because we have several chains
        prediction = prediction.reshape((-1, prediction.shape[-1]))

    return prediction


def get_quantiles(prediction, lower_q, upper_q, lower_cut, upper_cut):
    prediction = _flatten_chains(prediction)

    prediction_q = np.quantile(
        prediction,
        q=[lower_q, 0.5, upper_q],
        axis=0,
    )
    if lower_cut is not None:
        prediction_q[prediction_q < lower_cut] = lower_cut

    if upper_cut is not None:
        prediction_q[prediction_q < upper_cut] = upper_cut

    return prediction_q


def get_cumulative_quantiles(prediction, lower_q=0.025, upper_q=0.975):
    """
    Quantiles of the cumulative sum of posterior predictive sample paths.

    Unlike get_quantiles (quantiles of each per-timestep marginal independently),
    this cumsum's each sample's full time path first, then takes quantiles across
    samples. This preserves the within-draw correlation across time induced by
    shared parameter values (e.g. a high-`g` draw stays high for every day), which
    a per-timestep sigma*sqrt(n) propagation cannot capture. No parametric
    autocorrelation assumption or inflation constant is needed.

    Parameters:
    - prediction (np.ndarray | xarray.DataArray): predictions, shape (samples, time)
      or (chain, draw, time).
    - lower_q (float): lower cumulative quantile (default 0.025).
    - upper_q (float): upper cumulative quantile (default 0.975).

    Returns:
    - np.ndarray of shape (3, time): stacked [lower, median, upper] cumulative
      quantiles, matching get_quantiles's output convention.
    """
    samples = _flatten_chains(prediction)
    cum_samples = np.cumsum(samples, axis=1)
    return np.quantile(cum_samples, q=[lower_q, 0.5, upper_q], axis=0)


def time_series_hdi(
    measure_ts: pd.Series,
    prediction: np.ndarray | xarray.DataArray,
    y_label: str = None,
    title: str = None,
    lower_q=0.025,
    upper_q=0.975,
    upper_cut=None,
    lower_cut=None,
    image_path: Path = None,
    figsize: tuple = (10, 6),
    backend: str = "plotly",
):
    """
    Visualise actual measure time series  and probabilist model prediction.
    Measure are plotted as scatter points, predictions are plotted using a surface
    bounded by lower and upper quantiles around the median.

    Parameters:
    - measure_ts (pd.Series): The observed time series data.
    - prediction (np.ndarray): Array containing predictions, potentially from multiple
    chains.
    - y_label (str): Label for the y-axis of the plot.
    - title (str): Title for the plot.
    - lower_q (float): Lower quantile for the high-density interval (default is 0.025).
    - upper_q (float): Upper quantile for the high-density interval (default is 0.975).
    - upper_cut (float): Upper bound for the high-density interval (optional).
    - lower_cut (float): Lower bound for the high-density interval (optional).
    - backend (str): switch between a matplotlib or a plotly render

    Returns:
    - None: The function displays the plot.

    """
    pridiction_q = get_quantiles(prediction, lower_q, upper_q, lower_cut, upper_cut)
    d_data = measure_ts.to_frame()
    d_data["pred_low"] = pridiction_q[0, :]
    d_data["pred_med"] = pridiction_q[1, :]
    d_data["pred_up"] = pridiction_q[2, :]

    if backend == "plotly":
        fig = make_subplots()
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data.iloc[:, 0],
                mode="markers",
                name="Observed",
                marker=dict(color="green", size=10),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d %H:%M}<br>"
                    + (y_label or "Value")
                    + ": %{y}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_med"],
                mode="lines",
                name="Predicted Median",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_low"],
                mode="lines",
                fill=None,
                line=dict(color="orange"),
                name=f"Quantile {lower_q}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_up"],
                mode="lines",
                fill="tonexty",
                line=dict(color="orange"),
                name=f"Quantile {upper_q}",
            )
        )
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_label)
        return fig

    elif backend == "matplotlib":
        plt.figure(figsize=figsize)
        plt.scatter(
            d_data.index,
            d_data.iloc[:, 0],
            color="green",
            label="Observed",
            alpha=0.5,
        )

        plt.plot(
            d_data.index,
            d_data["pred_med"],
            color="orange",
            label="Predicted Median",
        )

        plt.fill_between(
            d_data.index,
            d_data["pred_low"],
            d_data["pred_up"],
            color="orange",
            alpha=0.1,
        )

        plt.ylabel(y_label)
        plt.title(title)
        if image_path is not None:
            plt.savefig(image_path, format="png", bbox_inches="tight")
        return plt.gcf()

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )


def plot_cumulative_energy_hdi(
    measure_ts: pd.Series,
    prediction: np.ndarray | xarray.DataArray,
    y_label: str = None,
    title: str = None,
    lower_q=0.025,
    upper_q=0.975,
    image_path: Path = None,
    figsize: tuple = (10, 6),
    backend: str = "plotly",
):
    """
    Visualise cumulative measured energy against the cumulative posterior predictive
    distribution. Each posterior predictive sample path is cumsum'd over time before
    quantiles are taken across samples (see get_cumulative_quantiles), so the band
    reflects the actual correlation structure of the fitted model instead of a
    parametric sigma*sqrt(n) (or hand-tuned autocorrelation-inflated) approximation.

    Parameters:
    - measure_ts (pd.Series): The observed time series data.
    - prediction (np.ndarray | xarray.DataArray): Posterior predictive samples,
      potentially from multiple chains.
    - y_label (str): Label for the y-axis of the plot.
    - title (str): Title for the plot.
    - lower_q (float): Lower cumulative quantile (default is 0.025).
    - upper_q (float): Upper cumulative quantile (default is 0.975).
    - backend (str): switch between a matplotlib or a plotly render

    Returns:
    - The rendered figure.
    """
    prediction_cq = get_cumulative_quantiles(prediction, lower_q, upper_q)

    d_data = measure_ts.cumsum().to_frame(name="measure_cum")
    d_data["pred_low"] = prediction_cq[0, :]
    d_data["pred_med"] = prediction_cq[1, :]
    d_data["pred_up"] = prediction_cq[2, :]

    final_gap = d_data["pred_med"].iloc[-1] - d_data["measure_cum"].iloc[-1]
    final_half_width = (d_data["pred_up"].iloc[-1] - d_data["pred_low"].iloc[-1]) / 2
    coverage = (
        (d_data["measure_cum"] >= d_data["pred_low"])
        & (d_data["measure_cum"] <= d_data["pred_up"])
    ).mean() * 100

    daily_median = np.median(_flatten_chains(prediction), axis=0)
    nmbe_percent = (
        100 * (daily_median - measure_ts.values).sum() / measure_ts.values.sum()
    )

    if backend == "plotly":
        fig = make_subplots()
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_up"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name=f"Predicted [{lower_q}, {upper_q}]",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["pred_med"],
                mode="lines",
                name="Predicted cumulative median",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["measure_cum"],
                mode="lines",
                name="Measured cumulative",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d %H:%M}<br>"
                    + (y_label or "Measured cumulative")
                    + ": %{y}<extra></extra>"
                ),
            )
        )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=(
                f"Final gap = {final_gap:.0f}<br>"
                f"NMBE = {nmbe_percent:.1f}%<br>"
                f"Final band half-width = {final_half_width:.0f}<br>"
                f"Coverage = {coverage:.1f}%"
            ),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            hovermode="x unified",
        )
        return fig

    elif backend == "matplotlib":
        plt.figure(figsize=figsize)
        plt.plot(
            d_data.index,
            d_data["measure_cum"],
            color="green",
            label="Measured cumulative",
        )
        plt.plot(
            d_data.index,
            d_data["pred_med"],
            color="orange",
            label="Predicted cumulative median",
        )
        plt.fill_between(
            d_data.index,
            d_data["pred_low"],
            d_data["pred_up"],
            color="orange",
            alpha=0.1,
            label=f"Predicted [{lower_q}, {upper_q}]",
        )

        text = (
            f"Final gap = {final_gap:.0f}\n"
            f"NMBE = {nmbe_percent:.1f}%\n"
            f"Final band half-width = {final_half_width:.0f}\n"
            f"Coverage = {coverage:.1f}%"
        )
        plt.gca().text(
            0.02,
            0.98,
            text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        if image_path is not None:
            plt.savefig(image_path, format="png", bbox_inches="tight")
        return plt.gcf()

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )


def plot_cumulative_gap_hdi(
    measure_ts: pd.Series,
    prediction: np.ndarray | xarray.DataArray,
    y_label: str = None,
    title: str = None,
    lower_q=0.025,
    upper_q=0.975,
    image_path: Path = None,
    figsize: tuple = (10, 6),
    backend: str = "plotly",
):
    """
    Visualise the cumulative gap (measured - predicted) over time, instead of
    the two growing cumulative curves side by side (see
    plot_cumulative_energy_hdi). Both cumulative curves grow with the period
    total, so a persistent bias of only a few % can be invisible against the
    y-axis scale needed to show the full total. Centering on the gap removes
    that shared growing scale: a drift or an offset shows up directly as a
    curve moving away from (or wandering around) the zero line.

    The gap's credible band reuses the SAME cumulative posterior predictive
    quantiles as plot_cumulative_energy_hdi (get_cumulative_quantiles) rather
    than recomputing anything: since the measurement is a fixed (non-random)
    series, gap = measure_cum - pred_cum is, per draw, a monotonically
    DEcreasing affine transform of pred_cum, so its quantiles are the
    prediction's cumulative quantiles subtracted from measure_cum with the
    lower/upper bounds swapped. This keeps the two plots numerically
    consistent: this function's "coverage" (fraction of time steps where the
    gap band contains 0) equals plot_cumulative_energy_hdi's coverage
    (fraction of time steps where the cumulative band contains measure_cum).

    Parameters: same as plot_cumulative_energy_hdi.

    Returns:
    - The rendered figure.
    """
    prediction_cq = get_cumulative_quantiles(prediction, lower_q, upper_q)
    measure_cum = measure_ts.cumsum().to_numpy()

    d_data = pd.DataFrame(index=measure_ts.index)
    d_data["gap_low"] = measure_cum - prediction_cq[2, :]
    d_data["gap_med"] = measure_cum - prediction_cq[1, :]
    d_data["gap_up"] = measure_cum - prediction_cq[0, :]

    final_gap = d_data["gap_med"].iloc[-1]
    final_half_width = (d_data["gap_up"].iloc[-1] - d_data["gap_low"].iloc[-1]) / 2
    coverage = (
        (d_data["gap_low"] <= 0) & (d_data["gap_up"] >= 0)
    ).mean() * 100

    daily_median = np.median(_flatten_chains(prediction), axis=0)
    nmbe_percent = (
        100 * (daily_median - measure_ts.values).sum() / measure_ts.values.sum()
    )

    if backend == "plotly":
        fig = make_subplots()
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["gap_up"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["gap_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name=f"Gap [{lower_q}, {upper_q}]",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_data.index,
                y=d_data["gap_med"],
                mode="lines",
                name="Gap median (measure - predicted)",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d %H:%M}<br>"
                    + (y_label or "Gap")
                    + ": %{y}<extra></extra>"
                ),
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=(
                f"Final gap = {final_gap:.0f}<br>"
                f"NMBE = {nmbe_percent:.1f}%<br>"
                f"Final band half-width = {final_half_width:.0f}<br>"
                f"Coverage = {coverage:.1f}%"
            ),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=(f"{y_label} (measure - predicted)" if y_label else "Gap"),
            hovermode="x unified",
        )
        return fig

    elif backend == "matplotlib":
        plt.figure(figsize=figsize)
        plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Mesure (référence)")
        plt.plot(
            d_data.index,
            d_data["gap_med"],
            color="orange",
            label="Gap median (measure - predicted)",
        )
        plt.fill_between(
            d_data.index,
            d_data["gap_low"],
            d_data["gap_up"],
            color="orange",
            alpha=0.1,
            label=f"Gap [{lower_q}, {upper_q}]",
        )

        text = (
            f"Final gap = {final_gap:.0f}\n"
            f"NMBE = {nmbe_percent:.1f}%\n"
            f"Final band half-width = {final_half_width:.0f}\n"
            f"Coverage = {coverage:.1f}%"
        )
        plt.gca().text(
            0.02,
            0.98,
            text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.ylabel(f"{y_label} (measure - predicted)" if y_label else "Gap")
        plt.title(title)
        plt.legend()
        if image_path is not None:
            plt.savefig(image_path, format="png", bbox_inches="tight")
        return plt.gcf()

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )


def parity_plot(
    datasets: dict[str, tuple[pd.Series, np.ndarray | xarray.DataArray]],
    lower_q: float = 0.025,
    upper_q: float = 0.975,
    upper_cut=None,
    lower_cut=None,
    y_label: str = None,
    title: str = None,
    image_path: Path = None,
    figsize: tuple = (7, 7),
    backend: str = "plotly",
):
    """
    Parity plot (predicted vs. observed), typically one entry per split (train,
    test). Unlike a point-estimate model's parity plot, a Bayesian one must show
    per-observation uncertainty rather than a bare scatter of point predictions:
    each point is plotted at the posterior predictive median with an error bar
    spanning its own [lower_q, upper_q] credible interval, and what matters is
    calibration -- whether the measurement actually falls inside that interval --
    not merely how close the median sits to the 1:1 line. Points whose measurement
    falls outside their own credible interval are outlined in red. Per-dataset
    coverage, NMBE and CV(RMSE) (on medians) are annotated; a test split with much
    worse coverage/NMBE than train signals overfitting to the training period.

    Parameters:
    - datasets (dict): mapping of split name (e.g. "train", "test") to a tuple
      (measure_ts, prediction), same shapes as accepted by time_series_hdi.
    - lower_q, upper_q (float): credible interval bounds (default 0.025, 0.975).
    - y_label, title (str): axis label / title.
    - backend (str): 'plotly' or 'matplotlib'.

    Returns: the rendered figure.
    """
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    prepared = {}
    stats_lines = []
    bounds = []

    for name, (measure_ts, prediction) in datasets.items():
        pred_low, pred_med, pred_up = get_quantiles(
            prediction, lower_q, upper_q, lower_cut, upper_cut
        )
        measured = measure_ts.to_numpy()
        covered = (measured >= pred_low) & (measured <= pred_up)
        coverage = covered.mean() * 100
        nmbe = 100 * (pred_med - measured).sum() / measured.sum()
        cv_rmse = 100 * np.sqrt(np.mean((pred_med - measured) ** 2)) / measured.mean()

        prepared[name] = dict(
            measured=measured,
            pred_med=pred_med,
            pred_low=pred_low,
            pred_up=pred_up,
            covered=covered,
            index=measure_ts.index,
        )
        stats_lines.append(
            f"{name}: coverage={coverage:.0f}%, NMBE={nmbe:.1f}%, "
            f"CV(RMSE)={cv_rmse:.1f}%"
        )
        bounds.extend([measured.min(), measured.max(), pred_low.min(), pred_up.max()])

    lims = (min(bounds), max(bounds))

    if backend == "plotly":
        fig = make_subplots()
        fig.add_trace(
            go.Scatter(
                x=list(lims),
                y=list(lims),
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="1:1",
                hoverinfo="skip",
            )
        )
        for (name, d), color in zip(prepared.items(), palette):
            edge_color = ["red" if not c else "rgba(0,0,0,0)" for c in d["covered"]]
            hovertemplate = (
                "Mesuré: %{x}<br>Prédit (médiane): %{y}<br>"
                "Date: %{customdata}<extra></extra>"
            )
            customdata = (
                d["index"].strftime("%Y-%m-%d %H:%M")
                if isinstance(d["index"], pd.DatetimeIndex)
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=d["measured"],
                    y=d["pred_med"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=color,
                        size=8,
                        line=dict(width=1.5, color=edge_color),
                    ),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=d["pred_up"] - d["pred_med"],
                        arrayminus=d["pred_med"] - d["pred_low"],
                        thickness=1,
                        width=0,
                        color=color,
                    ),
                    hovertemplate=hovertemplate,
                    customdata=customdata,
                )
            )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text="<br>".join(stats_lines),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Mesuré" + (f" ({y_label})" if y_label else ""),
            yaxis_title="Prédit (médiane)" + (f" ({y_label})" if y_label else ""),
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        return fig

    elif backend == "matplotlib":
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lims, lims, color="black", linestyle="--", label="1:1", zorder=1)
        for (name, d), color in zip(prepared.items(), palette):
            yerr = np.vstack(
                [d["pred_med"] - d["pred_low"], d["pred_up"] - d["pred_med"]]
            )
            edgecolors = np.where(d["covered"], "none", "red")
            ax.errorbar(
                d["measured"],
                d["pred_med"],
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                alpha=0.5,
                zorder=2,
            )
            ax.scatter(
                d["measured"],
                d["pred_med"],
                color=color,
                edgecolors=edgecolors,
                linewidths=1.5,
                s=40,
                label=name,
                alpha=0.8,
                zorder=3,
            )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Mesuré" + (f" ({y_label})" if y_label else ""))
        ax.set_ylabel("Prédit (médiane)" + (f" ({y_label})" if y_label else ""))
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.legend(loc="lower right")
        if image_path is not None:
            plt.savefig(image_path, format="png", bbox_inches="tight")
        return fig

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )


def residual_cusum_plot(
    datasets: dict[str, tuple[pd.Series, np.ndarray | xarray.DataArray]],
    y_label: str = None,
    title: str = None,
    image_path: Path = None,
    figsize: tuple = (10, 7),
    n_permutations: int = 500,
    envelope_q: tuple = (0.025, 0.975),
    random_state: int = None,
):
    """
    Two-panel bias diagnostic across one or more chronologically ordered splits
    (typically train then test, in that order): per-observation residual
    (measured - posterior predictive median) over calendar time, and its
    cumulative sum carried over across splits (NOT reset to 0 at each split
    boundary). A kink or slope change in the CUSUM right at a split boundary
    signals a level shift in the bias between periods (e.g. a physical change in
    the building, a sensor recalibration) rather than plain day-to-day noise --
    a single unbroken slope throughout instead shows one consistent bias rate.
    Each split's mean residual is drawn as a dashed horizontal reference line on
    the residual panel, making an additive offset between splits directly
    readable.

    A CUSUM of purely random, unbiased residuals is already a random walk that
    wanders away from 0 on its own -- curvature alone is not evidence of a real
    pattern. Each split's CUSUM is therefore also compared against a null
    envelope built by randomly permuting that split's own residuals
    `n_permutations` times and recomputing the CUSUM: permuting preserves the
    split's total (mean bias) but destroys any temporal order, so the resulting
    [envelope_q[0], envelope_q[1]] band is the range of paths a purely random
    day-to-day ordering of those SAME residuals would produce. If the real CUSUM
    stays inside the band, the wiggle isn't distinguishable from noise. If it
    breaks out, that excursion is a real, non-random pattern worth investigating
    in the raw data.

    Parameters:
    - datasets (dict): mapping of split name, IN CHRONOLOGICAL ORDER, to a tuple
      (measure_ts, prediction) -- same shapes as accepted by time_series_hdi. The
      CUSUM's running total carries over from one entry to the next in dict
      iteration order, so pass splits in the order they occur in time.
    - y_label, title (str): axis label / title.
    - image_path (Path, optional): save path.
    - n_permutations (int): number of random shuffles used to build the null
      envelope (default 500).
    - envelope_q (tuple): lower/upper quantile of the null envelope (default
      (0.025, 0.975)).
    - random_state (int, optional): seed for the permutation RNG.

    Returns: the rendered matplotlib figure.
    """
    rng = np.random.default_rng(random_state)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, (ax_res, ax_cusum) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    running_total = 0.0
    for (name, (measure_ts, prediction)), color in zip(datasets.items(), palette):
        pred_med = get_quantiles(prediction, 0.025, 0.975, None, None)[1]
        residual = measure_ts.to_numpy() - pred_med
        mean_res = residual.mean()

        ax_res.plot(
            measure_ts.index, residual, "o-", color=color, ms=3, lw=0.7, label=name
        )
        ax_res.axhline(
            mean_res,
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{name} moyenne = {mean_res:.0f}",
        )

        permuted = np.array(
            [rng.permutation(residual) for _ in range(n_permutations)]
        )
        null_cusum = running_total + np.cumsum(permuted, axis=1)
        lower_env, upper_env = np.quantile(null_cusum, envelope_q, axis=0)
        ax_cusum.fill_between(
            measure_ts.index,
            lower_env,
            upper_env,
            color=color,
            alpha=0.15,
            label=f"{name} enveloppe nulle "
            f"[{envelope_q[0]:.1%}, {envelope_q[1]:.1%}]",
        )

        cusum = running_total + np.cumsum(residual)
        ax_cusum.plot(measure_ts.index, cusum, color=color, label=name)
        running_total = cusum[-1]

    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.set_ylabel("Résidu (mesuré - prédit)" + (f" ({y_label})" if y_label else ""))
    ax_res.legend(fontsize=8, ncol=2)
    ax_res.set_title(title)

    ax_cusum.set_ylabel("Résidu cumulé" + (f" ({y_label})" if y_label else ""))
    ax_cusum.axhline(0, color="black", linewidth=0.8)
    ax_cusum.legend(fontsize=8)
    ax_cusum.set_xlabel("Date")

    plt.tight_layout()
    if image_path is not None:
        plt.savefig(image_path, format="png", bbox_inches="tight")
    return fig


def plot_period_total_bars(
    measure_ts: pd.Series,
    predictions: dict[str, np.ndarray | xarray.DataArray],
    y_label: str = None,
    title: str = None,
    lower_q: float = 0.025,
    upper_q: float = 0.975,
    measure_color: str = "#EF553B",
) -> go.Figure:
    """
    Bar chart comparing the measured period total against each model's mean
    predicted period total, with an asymmetric error bar showing the
    [lower_q, upper_q] CI from the posterior predictive draws (see
    get_total_stats). First bar is the measured total (measure_color, no CI);
    one further bar per entry in `predictions`.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Mesure"],
            y=[float(measure_ts.sum())],
            name="Mesure",
            marker_color=measure_color,
        )
    )
    for name, prediction in predictions.items():
        stats = get_total_stats(prediction, lower_q, upper_q)
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[stats["mean"]],
                name=name,
                error_y=dict(
                    type="data",
                    array=[stats["upper"] - stats["mean"]],
                    arrayminus=[stats["mean"] - stats["lower"]],
                    visible=True,
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        yaxis_title=y_label,
        showlegend=False,
    )
    return fig


def changepoint_graph(
    x_variable: pd.Series,
    y_measure: pd.Series,
    prediction: np.ndarray,
    changepoint_periods: pd.Series = None,
    lower_q=0.025,
    upper_q=0.975,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    upper_cut=None,
    lower_cut=None,
    image_path: Path = None,
    backend: str = "plotly",
):
    """
    Visualise target data, measures and prediction as a function of an independent
    variable. Adapted to change point model. Prediction is displayed as a surface
    bounded by the lower an upper quartiles. A surface is drawn for each changepoint
    period.

    Parameters:
    - x_variable (pd.Series): The independent variable data.
    - y_measure (pd.Series): The dependent variable data.
    - prediction (np.ndarray): Array containing predictions.
    - changepoint_periods (pd.Series, optional): Series containing changepoint periods
        (default is None, only one period is considered).
    - lower_q (float): Lower quantile for the high-density interval (default is 0.025).
    - upper_q (float): Upper quantile for the high-density interval (default is 0.975).
    - x_label (str, optional): Label for the x-axis (default is None).
    - y_label (str, optional): Label for the y-axis (default is None).
    - title (str, optional): Title of the plot (default is None).
    - upper_cut (float, optional): Upper bound for the high-density interval
        (default is None).
    - lower_cut (float, optional): Lower bound for the high-density interval
        (default is None).
    - backend (str, optional): Backend for plotting, choose either 'plotly' or
        'matplotlib' (default is 'plotly').

    Returns:
    - None: The function displays the plot.

    Raises:
    - ValueError: If the specified backend is not 'plotly' or 'matplotlib'.
    """

    if changepoint_periods is None:
        changepoint_periods = np.zeros(x_variable.shape[0])

    prediction_q = get_quantiles(prediction, lower_q, upper_q, lower_cut, upper_cut)
    d_data = pd.concat([x_variable, y_measure], axis=1)
    d_data["pred_low"] = prediction_q[0, :]
    d_data["pred_med"] = prediction_q[1, :]
    d_data["pred_up"] = prediction_q[2, :]

    x_name = x_variable.name
    y_name = y_measure.name

    d_data.sort_values(x_name, inplace=True)

    color_list = ["blue", "red", "orange", "green"]
    mask_list = []
    for period in set(changepoint_periods):
        mask_list.append(changepoint_periods == period)

    if backend == "plotly":
        fig = make_subplots()
        if isinstance(d_data.index, pd.DatetimeIndex):
            observed_hovertemplate = (
                (x_label or x_name) + ": %{x}<br>"
                + (y_label or y_name) + ": %{y}<br>"
                "Date: %{customdata}<extra></extra>"
            )
            observed_customdata = d_data.index.strftime("%Y-%m-%d %H:%M")
        else:
            observed_hovertemplate = None
            observed_customdata = None
        fig.add_trace(
            go.Scatter(
                x=d_data[x_name],
                y=d_data[y_name],
                mode="markers",
                marker=dict(color=changepoint_periods, colorscale="Bluered", size=10),
                name="Observed",
                hovertemplate=observed_hovertemplate,
                customdata=observed_customdata,
            )
        )
        for mask, color in zip(mask_list, color_list):
            fig.add_trace(
                go.Scatter(
                    x=d_data.loc[mask, x_name],
                    y=d_data.loc[mask, "pred_med"],
                    mode="lines",
                    line=dict(color=color),
                    name="Predicted Median",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=d_data.loc[mask, x_name],
                    y=d_data.loc[mask, "pred_low"],
                    mode="lines",
                    fill=None,
                    line=dict(color=color),
                    name="Lower Bound",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=d_data.loc[mask, x_name],
                    y=d_data.loc[mask, "pred_up"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(color=color),
                    name="Upper Bound",
                )
            )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        return fig

    elif backend == "matplotlib":
        plt.figure(figsize=(10, 6))
        plt.scatter(
            d_data[x_name],
            d_data[y_name],
            c=changepoint_periods,
            cmap="coolwarm",
            label="Observed",
            alpha=0.5,
        )

        for mask, color in zip(mask_list, color_list):
            plt.plot(
                d_data.loc[mask, x_name],
                d_data.loc[mask, "pred_med"],
                color=color,
                label="Predicted Median",
            )

            plt.fill_between(
                d_data.loc[mask, x_name],
                d_data.loc[mask, "pred_low"],
                d_data.loc[mask, "pred_up"],
                color=color,
                alpha=0.1,
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if image_path is not None:
            plt.savefig(image_path, format="png", bbox_inches="tight")
        return plt.gcf()

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )


def time_series_bar_plot(
    measure: pd.Series,
    prediction: np.ndarray | xarray.DataArray,
    lower_q=0.025,
    upper_q=0.975,
    upper_cut=None,
    lower_cut=None,
    bar_width=0.35,
    title: str = None,
    y_label: str = None,
    image_path: Path = None,
):
    """
    Generate a time series bar plot comparing a measure against a prediction with
    error bars representing quantiles.

    Args:
        measure (pd.Series): The measured values as a Pandas Series.
        prediction (np.ndarray or xarray.DataArray):
            The predicted values or data array.
        lower_q (float, optional): The lower quantile for the error bars.
        Defaults to 0.025.
        upper_q (float, optional): The upper quantile for the error bars.
        Defaults to 0.975.
        upper_cut (float, optional): Upper limit for outliers. Defaults to None.
        lower_cut (float, optional): Lower limit for outliers. Defaults to None.
        bar_width (float, optional): Width of each bar. Defaults to 0.35.
        title (str, optional): Title for the plot. Defaults to None.
        y_label (str, optional): Label for the y-axis. Defaults to None.
    """

    lower_q, med, upper_q = get_quantiles(
        prediction, lower_q, upper_q, lower_cut, upper_cut
    )

    index = range(len(measure.index))

    plt.figure(figsize=(6, 5))
    plt.bar(
        [i - bar_width / 2 for i in index],
        med,
        width=bar_width,
        yerr=[med - lower_q, upper_q - med],
        capsize=10,
        label="Modèle",
    )
    plt.bar(
        [i + bar_width / 2 for i in index], measure, width=bar_width, label="Mesure"
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(index, measure.index)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.legend()
    if image_path is not None:
        plt.savefig(image_path, format="png", bbox_inches="tight")
    return plt.gcf()


def compare_bars(
    measure: float,
    prediction: np.ndarray | xarray.DataArray,
    lower_q: float = 0.025,
    upper_q: float = 0.975,
    upper_cut: int | float = None,
    lower_cut: int | float = None,
    title: str = None,
    y_label: str = None,
    measure_label: str = "Measure",
    prediction_label: str = "Model",
    image_path: Path = None,
    font_size: int = 12,
):
    """
    Compare a measure against a prediction using bar plots with error bars
    representing quantiles.

    Args:
        measure (float): The measured value to compare against the prediction.
        prediction (np.ndarray or xarray.DataArray):
            The predicted values or data array.
        lower_q (float, optional): The lower quantile for the error bars.
        Defaults to 0.025.
        upper_q (float, optional): The upper quantile for the error bars.
        Defaults to 0.975.
        upper_cut (float, optional): Upper limit for outliers. Defaults to None.
        lower_cut (float, optional): Lower limit for outliers. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        y_label (str, optional): Label for the y-axis. Defaults to None.
        measure_label (str, optional): Label for the measure. Defaults to "Measure".
        prediction_label (str, optional): Label for the prediction. Defaults to "Model".
        image_path (Path, optional): Saving path to png image
        font_size (Int, optional): Fontsize for all figure text. Default 12
    """

    lower_q, med, upper_q = get_quantiles(
        prediction, lower_q, upper_q, lower_cut, upper_cut
    )

    plt.figure(figsize=(6, 5))
    plt.bar(
        0,
        med,
        yerr=[[med - lower_q], [upper_q - med]],
        capsize=10,
        label=prediction_label,
    )
    plt.bar(1, measure, label=measure_label)
    plt.title(title, fontsize=font_size)  # Set font size for title
    plt.ylabel(y_label, fontsize=font_size)  # Set font size for y-label
    plt.xticks(
        [0, 1], [prediction_label, measure_label], fontsize=font_size
    )  # Set font size for x-ticks
    plt.yticks(fontsize=font_size)  # Set font size for y-ticks

    if image_path is not None:
        plt.savefig(image_path, format="png", bbox_inches="tight")

    return plt.gcf()
