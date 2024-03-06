import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import xarray
from plotly.subplots import make_subplots


def time_series_hdi(
    measure_ts: pd.Series,
    prediction: np.ndarray | xarray.core.dataarray.DataArray,
    y_label: str = None,
    title: str = None,
    lower_q=0.025,
    upper_q=0.975,
    upper_cut=None,
    lower_cut=None,
    backend: str = "plotly",
):
    """
    Visualise actual measure and probabilist model prediction.
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
    if isinstance(prediction, xarray.core.dataarray.DataArray):
        prediction = np.array(prediction)

    if prediction.ndim > 2:  # Assume it's because we have several chains
        prediction = prediction.reshape((-1, prediction.shape[2]))

    predictive_q = np.quantile(
        prediction,
        q=[lower_q, 0.5, upper_q],
        axis=0,
    )
    if lower_cut is not None:
        predictive_q[predictive_q < lower_cut] = lower_cut

    if upper_cut is not None:
        predictive_q[predictive_q < upper_cut] = upper_cut

    d_data = measure_ts.to_frame()
    d_data["pred_low"] = predictive_q[0, :]
    d_data["pred_med"] = predictive_q[1, :]
    d_data["pred_up"] = predictive_q[2, :]

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
        fig.show()
    elif backend == "matplotlib":
        plt.figure(figsize=(10, 6))
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
        plt.show()

    else:
        raise ValueError(
            f"{backend} is an invalid backennd argument, choose one of"
            f"'plotly' or 'matplotlib"
        )
