import itertools
from typing import Union
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
from . import plotlying, stats


def _master_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xerr: Union[list, None] = None,
    yerr: Union[list, None] = None,
    method_name: str = "",
    target_name: str = "",
    quantity: str = r"$\Delta \Delta$ G",
    xlabel: str = "Experimental",
    ylabel: str = "Calculated",
    units: str = r"$\mathrm{kcal\,mol^{-1}}$",
    guidelines: bool = True,
    origins: bool = True,
    color: Union[str, None] = None,
    statistics: list = ["RMSE", "MUE"],
    filename: Union[str, None] = None,
    centralizing: bool = True,
    shift: float = 0.0,
    figsize: float = 3.25,
):
    """Handles the aesthetics of the plots in one place.

    Parameters
    ----------
    x : list
        Values to plot on the x axis
    y : list
        Values to plot on the y axis
    title : string, default = ''
        Title for the plot
    xerr : list , default = None
        Error bars for x values
    yerr : list , default = None
        Error bars for y values
    method_name : string, optional
        name of method associated with results, e.g. 'perses'
    target_name : string, optional
        name of system for results, e.g. 'Thrombin'
    quantity : str, default = '$\Delta \Delta$ G'
        metric that is being plotted
    xlabel : str, default = 'Experimental'
        label for xaxis
    ylabel : str, default = 'Calculated'
        label for yaxis
    units : str, default = r'$\mathrm{kcal\,mol^{-1}}$'
        string value of units to label axis
    guidelines : bool, default = True
        toggles plotting of grey 0.5 and 1 kcal/mol error zone
    origins : bool, default = True
        toggles plotting of x and y axis
    color : str, default = None
        if None, will be coloured according to distance from unity
    statistics : list(str), default = ['RMSE',  'MUE']
        list of statistics to calculate and report on the plot
    filename : str, default = None
        filename for plot
    centralizing : bool, default = True
        ofset the free energies
    shift : float, default = 0.
        shift both the x and y axis by a constant
    figsize : float, default = 3.25
        size of figure for matplotlib

    Returns
    -------

    """
    nsamples = len(x)
    # aesthetics
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["font.size"] = 12

    fig = plt.figure(figsize=(figsize, figsize))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    plt.xlabel(f"{xlabel} {quantity} / " + units)
    plt.ylabel(f"{ylabel} {quantity} / " + units)

    ax_min = min(min(x), min(y)) - 0.5
    ax_max = max(max(x), max(y)) + 0.5
    scale = [ax_min, ax_max]

    plt.xlim(scale)
    plt.ylim(scale)

    # plots x-axis and y-axis
    if origins:
        plt.plot([0, 0], scale, "gray")
        plt.plot(scale, [0, 0], "gray")

    # plots x=y line
    plt.plot(scale, scale, "k:")
    if guidelines:
        small_dist = 0.5
        # plots grey region around x=y line
        plt.fill_between(
            scale,
            [ax_min - small_dist, ax_max - small_dist],
            [ax_min + small_dist, ax_max + small_dist],
            color="grey",
            alpha=0.2,
        )
        plt.fill_between(
            scale,
            [ax_min - small_dist * 2, ax_max - small_dist * 2],
            [ax_min + small_dist * 2, ax_max + small_dist * 2],
            color="grey",
            alpha=0.2,
        )
    # actual plotting
    cm = plt.get_cmap("coolwarm")

    if color is None:
        color = np.abs(x - y)
        # 2.372 kcal / mol = 4 RT
        color = cm(color / 2.372)
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color="gray",
        linewidth=0.0,
        elinewidth=2.0,
        zorder=1,
    )
    plt.scatter(x, y, color=color, s=10, marker="o", zorder=2)

    # stats and title
    statistics_string = ""
    for statistic in statistics:
        s = stats.bootstrap_statistic(x, y, xerr, yerr, statistic=statistic)
        string = f"{statistic}:   {s['mle']:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}] " + "\n"
        statistics_string += string

    long_title = f"{title} \n {target_name} (N = {nsamples}) \n {statistics_string}"

    plt.title(
        long_title,
        fontsize=12,
        loc="right",
        horizontalalignment="right",
        family="monospace",
    )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches="tight")
    return fig


def plot_DDGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    map_positive: bool = False,
    filename: Union[list, None] = None,
    symmetrise: bool = False,
    plotly: bool = False,
    **kwargs,
):
    """Function to plot relative free energies

    Parameters
    ----------
    graph : nx.DiGraph
        graph object with relative free energy edges
    method_name : string, optional
        name of method associated with results, e.g. 'perses'
    target_name : string, optional
        name of system for results, e.g. 'Thrombin'
    title : string, default = ''
        Title for the plot
    map_positive : bool, default=False
        whether to map all DDGs to the positive x values.
        this is an aesthetic choice
    filename : str, default = None
        filename for plot
    symmetrise : bool, default = False
        whether to plot each datapoint twice, both
        positive and negative

    Returns
    -------

    """

    assert (
        int(symmetrise) + int(map_positive) != 2
    ), "Symmetrise and map_positive cannot both be True in the same plot"

    # data
    x = [x[2]["exp_DDG"] for x in graph.edges(data=True)]
    y = [x[2]["calc_DDG"] for x in graph.edges(data=True)]

    if symmetrise:
        x_data = np.append(x, [-i for i in x])
        y_data = np.append(y, [-i for i in y])
    elif map_positive:
        x_data = []
        y_data = []
        for i, j in zip(x, y):
            if i < 0:
                x_data.append(-i)
                y_data.append(-j)
            else:
                x_data.append(i)
                y_data.append(j)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
    else:
        x_data = np.asarray(x)
        y_data = np.asarray(y)

    xerr = np.asarray([x[2]["exp_dDDG"] for x in graph.edges(data=True)])
    yerr = np.asarray([x[2]["calc_dDDG"] for x in graph.edges(data=True)])
    if symmetrise:
        xerr = np.append(xerr, xerr)
        yerr = np.append(yerr, yerr)

    if plotly:
        plotlying._master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            filename=filename,
            plot_type="ΔΔG",
            title=title,
            method_name=method_name,
            target_name=target_name,
            **kwargs,
        )
    else:
        _master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            filename=filename,
            title=title,
            method_name=method_name,
            target_name=target_name,
            **kwargs,
        )

    return


def plot_DGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Union[str, None] = None,
    plotly: bool = False,
    centralizing: bool = True,
    shift: float = 0.0,
    **kwargs,
):
    """Function to plot absolute free energies.

    Parameters
    ----------
    graph : nx.DiGraph
        graph object with relative free energy edges
    method_name : string, optional
        name of method associated with results, e.g. 'perses'
    target_name : string, optional
        name of system for results, e.g. 'Thrombin'
    title : string, default = ''
        Title for the plot
    filename : str, default = None
        filename for plot

    Returns
    -------

    """

    # data
    x_data = np.asarray([node[1]["exp_DG"] for node in graph.nodes(data=True)])
    y_data = np.asarray([node[1]["calc_DG"] for node in graph.nodes(data=True)])
    xerr = np.asarray([node[1]["exp_dDG"] for node in graph.nodes(data=True)])
    yerr = np.asarray([node[1]["calc_dDG"] for node in graph.nodes(data=True)])

    # centralising
    # this should be replaced by providing one experimental result
    if centralizing == True:
        x_data = x_data - np.mean(x_data) + shift
        y_data = y_data - np.mean(y_data) + shift

    if plotly:
        plotlying._master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            origins=False,
            statistics=["RMSE", "MUE", "R2", "rho"],
            plot_type="ΔG",
            title=title,
            method_name=method_name,
            target_name=target_name,
            filename=filename,
            **kwargs,
        )
    else:
        _master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            origins=False,
            statistics=["RMSE", "MUE", "R2", "rho"],
            quantity=rf"$\Delta$ G",
            title=title,
            method_name=method_name,
            target_name=target_name,
            filename=filename,
            **kwargs,
        )

    return


def plot_all_DDGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Union[str, None] = None,
    plotly: bool = False,
    **kwargs,
):
    """Plots relative free energies between all ligands, which is calculated from
    the differences between all the absolute free energies. This data is different to `plot_DGs`

    Parameters
    ----------
    graph : nx.DiGraph
        graph object with relative free energy edges
    method_name : string, optional
        name of method associated with results, e.g. 'perses'
    target_name : string, optional
        name of system for results, e.g. 'Thrombin'
    title : string, default = ''
        Title for the plot
    filename : str, default = None
        filename for plot
    plotly : bool, default = True
        whether to use plotly for the plotting

    Returns
    -------

    """

    nodes = graph.nodes(data=True)

    x_abs = np.asarray([node[1]["exp_DG"] for node in nodes])
    y_abs = np.asarray([node[1]["calc_DG"] for node in nodes])
    xabserr = np.asarray([node[1]["exp_dDG"] for node in nodes])
    yabserr = np.asarray([node[1]["calc_dDG"] for node in nodes])
    # do all to plot_all
    x_data = []
    y_data = []
    xerr = []
    yerr = []
    for a, b in itertools.combinations(range(len(x_abs)), 2):
        x = x_abs[a] - x_abs[b]
        x_data.append(x)
        x_data.append(-x)
        err = (xabserr[a] ** 2 + xabserr[b] ** 2) ** 0.5
        xerr.append(err)
        xerr.append(err)
        y = y_abs[a] - y_abs[b]
        y_data.append(y)
        y_data.append(-y)
        err = (yabserr[a] ** 2 + yabserr[b] ** 2) ** 0.5
        yerr.append(err)
        yerr.append(err)
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if plotly:
        plotlying._master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            title=title,
            method_name=method_name,
            plot_type="ΔΔG",
            filename=filename,
            target_name=target_name,
            **kwargs,
        )

    else:
        _master_plot(
            x_data,
            y_data,
            xerr=xerr,
            yerr=yerr,
            title=title,
            method_name=method_name,
            filename=filename,
            target_name=target_name,
            shift=shift,
            **kwargs,
        )
