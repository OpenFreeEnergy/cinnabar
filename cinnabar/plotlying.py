from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from . import stats


def plot_bar(
    df: pd.DataFrame,
    ddg_cols: str,
    error_cols: str,
    exp_col: str = "exp",
    exp_error_col: str = "dexp",
    name_col: str = "edge",
    title: str = "",
    filename: Optional[str] = None,
):
    """
    Creates a plotly barplot. It takes a pandas.Dataframe df as input and plots
    horizontal bars grouping the values in the rows together. The columns which
    will be used are specified by ddg_cols (DDG values),
    error_cols (corresponding errors), exp_col (column with exp. values),
    exp_error_col (column with exp. errors) and name_col (column which will be
    used as y axis tick labels).
    """

    # create color palette
    colors = sns.color_palette(palette="bright")

    num_edges = df.shape[0]
    num_bars_per_edge = len(ddg_cols)
    height = 20 * (num_bars_per_edge + 0.3) * num_edges
    exp_size = height / num_edges / 2.0
    alim = (
        np.max(
            np.fabs(df.loc[:, ddg_cols + [exp_col]].values)
            + np.fabs(df.loc[:, error_cols + [exp_error_col]].values)
        )
        * 1.05
    )

    fig = go.Figure()

    # add data
    for i, (col, ecol) in enumerate(zip(ddg_cols, error_cols)):
        fig.add_trace(
            go.Bar(
                x=df.loc[:, col].values,
                y=df[name_col].values,
                error_x=dict(
                    type="data",  # value of error bar given in data coordinates
                    array=df.loc[:, ecol].values,
                    visible=True,
                ),
                name=col,
                marker=dict(color=f"rgba{colors[i]}", line=None),
                orientation="h",
            )
        )

    if exp_col is not None:
        fig.add_trace(
            go.Scatter(
                x=df.loc[:, exp_col].values,
                y=df[name_col].values,
                name="experiment",
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    color="black",
                    size=exp_size,
                    line_width=4,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.loc[:, exp_col].values - df.loc[:, exp_error_col].values,
                y=df[name_col].values,
                name="ExpErrors1",
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    color="black",
                    size=exp_size,
                    line_width=2,
                ),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.loc[:, exp_col].values + df.loc[:, exp_error_col].values,
                y=df[name_col].values,
                name="ExpErrors2",
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    color="black",
                    size=exp_size,
                    line_width=2,
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=r"$\Delta\Delta G\, \mathrm{[kcal\,mol^{-1}]}$",
            titlefont_size=16,
            tickfont_size=14,
            range=(-alim, alim),
        ),
        yaxis=dict(
            title="Edge",
            titlefont_size=16,
            tickfont_size=14,
            range=(-0.5, num_edges - 0.5),
        ),
        width=800,
        height=height,
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
            font_size=16,
        ),
        barmode="group",
        bargap=0.3,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.0,  # gap between bars of the same location coordinate.
    )

    if filename is None:
        fig.show()
    elif filename.find(".html"):
        fig.write_html(filename)
    else:
        fig.write_image(filename)


def _master_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
    method_name: str = "",
    target_name: str = "",
    plot_type: str = "",
    guidelines: bool = True,
    origins: bool = True,
    statistics: list = ["RMSE", "MUE"],
    filename: Optional[str] = None,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
):
    nsamples = len(x)
    ax_min = min(min(x), min(y)) - 0.5
    ax_max = max(max(x), max(y)) + 0.5

    fig = go.Figure()

    # x = 0 and y = 0 axes through origin
    if origins:
        # x=0
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[ax_min, ax_max],
                line_color="black",
                mode="lines",
                showlegend=False,
            )
        )
        # y =0
        fig.add_trace(
            go.Scatter(
                x=[ax_min, ax_max],
                y=[0, 0],
                line_color="black",
                mode="lines",
                showlegend=False,
            )
        )
    if guidelines:
        small_dist = 0.5
        fig.add_trace(
            go.Scatter(
                x=[ax_min, ax_max, ax_max, ax_min],
                y=[
                    ax_min + 2.0 * small_dist,
                    ax_max + 2.0 * small_dist,
                    ax_max - 2.0 * small_dist,
                    ax_min - 2.0 * small_dist,
                ],
                name="1 kcal/mol margin",
                hoveron="points+fills",
                hoverinfo="name",
                fill="toself",
                mode="lines",
                line_width=0,
                fillcolor="rgba(0, 0, 0, 0.2)",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[ax_min, ax_max, ax_max, ax_min],
                y=[
                    ax_min + small_dist,
                    ax_max + small_dist,
                    ax_max - small_dist,
                    ax_min - small_dist,
                ],
                name=".5 kcal/mol margin",
                hoveron="points+fills",
                hoverinfo="name",
                fill="toself",
                mode="lines",
                line_width=0,
                fillcolor="rgba(0, 0, 0, 0.2)",
                showlegend=False,
            )
        )

    # diagonal
    fig.add_trace(
        go.Scatter(
            x=[ax_min, ax_max],
            y=[ax_min, ax_max],
            line_color="black",
            mode="lines",
            showlegend=False,
        )
    )

    # 2.372 kcal / mol = 4 RT
    clr = np.abs(x - y) / 2.372

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{target_name},{method_name}",
            marker=dict(symbol="circle", color=clr, colorscale="BlueRed"),
            error_x=dict(
                type="data",  # value of error bar given in data coordinates
                array=xerr,
                visible=True,
            ),
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=yerr,
                visible=True,
            ),
            showlegend=False,
        )
    )

    # stats and title
    string = []
    if statistic_type not in ['mle', 'mean']:
        raise ValueError(f"Unknown statistic type {statistic_type}")
    for statistic in statistics:
        bss = stats.bootstrap_statistic(x,
                                        y,
                                        xerr,
                                        yerr,
                                        statistic=statistic,
                                        include_true_uncertainty=bootstrap_x_uncertainty,
                                        include_pred_uncertainty=bootstrap_y_uncertainty)
        string.append(
            f"{statistic + ':':5s}{bss[statistic_type]:5.2f} [95%: {bss['low']:5.2f}, {bss['high']:5.2f}]"
        )
    stats_string = "<br>".join(string)

    long_title = f"{title}<br>{target_name} (N = {nsamples})<br>{stats_string}"

    # figure layout
    fig.update_layout(
        title=dict(
            text=long_title,
            font_family="monospace",
            x=0.0,
            y=0.95,
            font_size=14,
        ),
        xaxis=dict(
            title=f"Experimental {plot_type} [kcal mol<sup>-1</sup>]",
            titlefont_size=14,
            tickfont_size=12,
            range=(ax_min, ax_max),
        ),
        yaxis=dict(
            title=f"Calculated {plot_type} {method_name} [kcal mol<sup>-1</sup>]",
            titlefont_size=14,
            tickfont_size=12,
            range=(ax_min, ax_max),
        ),
        width=400,
        height=400
        #         legend=dict(
        #             x=1.0,
        #             y=1.0,
        #             bgcolor='rgba(255, 255, 255, 0)',
        #             bordercolor='rgba(255, 255, 255, 0)',
        #             font_size=12
        #         )
    )

    if filename is None:
        fig.show()
    elif filename.find(".html") > 0:
        fig.write_html(filename)
    else:
        fig.write_image(filename)
