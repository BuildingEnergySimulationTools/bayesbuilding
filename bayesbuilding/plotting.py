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


def get_quantiles(prediction, lower_q, upper_q, lower_cut, upper_cut):
    if isinstance(prediction, xarray.core.dataarray.DataArray):
        prediction = np.array(prediction)

    if prediction.ndim > 2:  # Assume it's because we have several chains
        prediction = prediction.reshape((-1, prediction.shape[2]))

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


def time_series_hdi(
    measure_ts: pd.Series,
    prediction: np.ndarray | xarray.core.dataarray.DataArray,
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
        fig.add_trace(
            go.Scatter(
                x=d_data[x_name],
                y=d_data[y_name],
                mode="markers",
                marker=dict(color=changepoint_periods, colorscale="Bluered", size=10),
                name="Observed",
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
    prediction: np.ndarray | xarray.core.dataarray.DataArray,
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
        prediction (np.ndarray or xarray.core.dataarray.DataArray):
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
    prediction: np.ndarray | xarray.core.dataarray.DataArray,
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
        prediction (np.ndarray or xarray.core.dataarray.DataArray):
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
