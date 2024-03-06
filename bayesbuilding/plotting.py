import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import xarray
from plotly.subplots import make_subplots


def _prepare_quantiles(prediction, lower_q, upper_q, lower_cut, upper_cut):
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
    pridiction_q = _prepare_quantiles(
        prediction, lower_q, upper_q, lower_cut, upper_cut
    )
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

    prediction_q = _prepare_quantiles(
        prediction, lower_q, upper_q, lower_cut, upper_cut
    )
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
        fig.show()

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
        plt.show()

    else:
        raise ValueError(
            f"{backend} is an invalid backend argument, choose one of"
            f"'plotly' or 'matplotlib"
        )
