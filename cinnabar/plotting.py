import itertools
from typing import Union, Optional
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
from adjustText import adjust_text
from . import plotlying, stats


def _master_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
    method_name: str = "",
    target_name: str = "",
    quantity: str = r"$\Delta \Delta$ G",
    xlabel: str = "Experimental",
    ylabel: str = "Calculated",
    units: str = r"$\mathrm{kcal\,mol^{-1}}$",
    guidelines: bool = True,
    origins: bool = True,
    color: Optional[str] = None,
    statistics: list = ["RMSE", "MUE"],
    filename: Optional[str] = None,
    centralizing: bool = True,
    shift: float = 0.0,
    figsize: float = 3.25,
    dpi: Union[float, str] = "figure",
    data_labels: list = [],
    axis_padding: float = 0.5,
    xy_lim: list = [],
    font_sizes: dict = {"title": 12, "labels": 9, "other": 12},
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
    scatter_kwargs: dict = {"s": 20, "marker": "o"},
):
    """Handles the aesthetics of the plots in one place.

    Parameters
    ----------
    x : np.ndarray
        Values to plot on the x axis
    y : np.ndarray
        Values to plot on the y axis
    title : string, default = ''
        Title for the plot
    xerr : np.ndarray , default = None
        Error bars for x values
    yerr : np.ndarray , default = None
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
    dpi : float or 'figure', default 'figure'
        the resolution in dots per inch
        if 'figure', uses the figure's dpi value (this behavior is copied from
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
    data_labels : list of str, default []
        list of labels for each data point
    axis_padding : float, default = 0.5
        padding to add to maximum axis value and subtract from the minimum axis value
    xy_lim : list, default []
        contains the minimium and maximum values to use for the x and y axes. if specified, axis_padding is ignored
    font_sizes : dict, default {"title": 12, "labels": 9, "other": 12}
        font sizes to use for the title ("title"), the data labels ("labels"), and the rest of the plot ("other")
    bootstrap_x_uncertainty : bool, default False
        whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        the type of statistic to use, either 'mle' (i.e. sample statistic)
        or 'mean' (i.e. bootstrapped mean statistic)
    scatter_kwargs : dict, default {"s": 20, "marker": "o"}
        arguments to control plt.scatter()

    Returns
    -------

    """
    nsamples = len(x)
    # aesthetics
    plt.rcParams["xtick.labelsize"] = font_sizes["other"]
    plt.rcParams["ytick.labelsize"] = font_sizes["other"]
    plt.rcParams["font.size"] = font_sizes["other"]

    fig = plt.figure(figsize=(figsize, figsize))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    plt.xlabel(f"{xlabel} {quantity} ({units})")
    plt.ylabel(f"{ylabel} {quantity} ({units})")

    if xy_lim:
        ax_min, ax_max = xy_lim
        scale = xy_lim
    else:
        ax_min = min(min(x), min(y)) - axis_padding
        ax_max = max(max(x), max(y)) + axis_padding
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
    plt.scatter(x, y, color=color, zorder=2, edgecolors='dimgrey', linewidths=0.7, **scatter_kwargs)

    # Label points
    texts = []
    for i, label in enumerate(data_labels):
        texts.append(plt.text(x[i] + 0.03, y[i] + 0.03, label, fontsize=font_sizes["labels"]))
    adjust_text(texts)

    # stats and title
    statistics_string = ""
    if statistic_type not in ['mle', 'mean']:
        raise ValueError(f"Unknown statistic type {statistic_type}")
    for statistic in statistics:
        s = stats.bootstrap_statistic(x,
                                      y,
                                      xerr,
                                      yerr,
                                      statistic=statistic,
                                      include_true_uncertainty=bootstrap_x_uncertainty,
                                      include_pred_uncertainty=bootstrap_y_uncertainty)
        string = f"{statistic}:   {s[statistic_type]:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}] " + "\n"
        statistics_string += string

    long_title = f"{title} \n {target_name} (N = {nsamples}) \n {statistics_string}"

    plt.title(
        long_title,
        fontsize=font_sizes["title"],
        loc="right",
        horizontalalignment="right",
        family="monospace",
    )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    return fig


def plot_DDGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    map_positive: bool = False,
    filename: Optional[str] = None,
    symmetrise: bool = False,
    plotly: bool = False,
    data_label_type: str = None,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
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
    plotly : bool, default = False
        whether to use plotly to generate the plot
    data_label_type : str or None, default = None
        type of data label to add to each edge

        if ``None`` data labels will not be added

        if ``'small-molecule'`` edge labels will be ``f"{node_A_name}→{node_B_name}"``.

        if ``'protein-mutation'`` edge labels will given as single letter amino
        acid codes separated by the mutated residue index (eg. ``"Y29A"``)

        If both node names start with ``"-"``, the negative sign will be factored
        out (eg. ``"-(Y29A)"`` or ``"-(benzene→toluene)"``).

    bootstrap_x_uncertainty : bool, default False
        whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        the type of statistic to use, either 'mle' (i.e. sample statistic)
        or 'mean' (i.e. bootstrapped mean statistic)

    Returns
    -------
    Nothing
    """

    assert (
        int(symmetrise) + int(map_positive) != 2
    ), "Symmetrise and map_positive cannot both be True in the same plot"

    # data
    x = [x[2]["exp_DDG"] for x in graph.edges(data=True)]
    y = [x[2]["calc_DDG"] for x in graph.edges(data=True)]
    xerr = np.asarray([x[2]["exp_dDDG"] for x in graph.edges(data=True)])
    yerr = np.asarray([x[2]["calc_dDDG"] for x in graph.edges(data=True)])

    # labels
    data_labels = []
    if data_label_type:
        node_names = {node_id: node_data.get("name", node_id) for node_id, node_data in graph.nodes(data=True)}
        data_labels = []
        for node_A, node_B, edge_data in graph.edges(data=True):
            node_A_name = node_names[node_A]
            node_B_name = node_names[node_B]
            if (
                "-" == node_A_name[0] and "-" == node_B_name[0]
            ):  # If the node names both start with "-", handle the negative sign properly in the label
                if data_label_type == "small-molecule":
                    data_labels.append(f"-({node_A_name[1:]}→{node_B_name[1:]})")
                elif data_label_type == "protein-mutation":
                    data_labels.append(f"-({node_A_name[1:]}{node_B_name[1]})")
                else:
                    raise Exception(
                        "data_label_type unsupported. supported types: 'small-molecule' and 'protein-mutation'"
                    )
            else:
                if data_label_type == "small-molecule":
                    data_labels.append(f"{node_A_name}→{node_B_name}")
                elif data_label_type == "protein-mutation":
                    data_labels.append(f"{node_A_name}{node_B_name[0]}")
                else:
                    raise Exception(
                        "data_label_type unsupported. supported types: 'small-molecule' and 'protein-mutation'"
                    )

    if symmetrise:
        x_data = np.append(x, [-i for i in x])
        y_data = np.append(y, [-i for i in y])
        xerr = np.append(xerr, xerr)
        yerr = np.append(yerr, yerr)
        data_labels = np.append(data_labels, data_labels)
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
            data_labels=data_labels,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
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
            data_labels=data_labels,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
            **kwargs,
        )


def plot_DGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Optional[str] = None,
    plotly: bool = False,
    data_label_type: str = None,
    centralizing: bool = True,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
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
    plotly : bool, default = False
        whether to use plotly to generate the plot
    data_label_type : str or None, default = None
        type of data label to add to each edge

        if ``None`` data labels will not be added

        if ``'small-molecule'`` labels will be ``f"{node_name}"``.
        
    bootstrap_x_uncertainty : bool, default False
        whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        the type of statistic to use, either 'mle' (i.e. sample statistic)
        or 'mean' (i.e. bootstrapped mean statistic)

    Returns
    -------

    """

    # data
    x_data = np.asarray([node[1]["exp_DG"] for node in graph.nodes(data=True)])
    y_data = np.asarray([node[1]["calc_DG"] for node in graph.nodes(data=True)])
    xerr = np.asarray([node[1]["exp_dDG"] for node in graph.nodes(data=True)])
    yerr = np.asarray([node[1]["calc_dDG"] for node in graph.nodes(data=True)])

    # labels
    data_labels = []
    if data_label_type:
        if data_label_type == "small-molecule":
            data_labels = [node_data.get("name", node_id) for node_id, node_data in graph.nodes(data=True)]
        else:
            raise Exception("data_label_type unsupported. supported types: 'small-molecule'")
         
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
            data_labels=data_labels,
            filename=filename,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
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
            data_labels=data_labels,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
            **kwargs,
        )


def plot_all_DDGs(
    graph: nx.DiGraph,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Optional[str] = None,
    plotly: bool = False,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
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
    shift : float, default = 0.
        shift both the x and y axis by a constant
    bootstrap_x_uncertainty : bool, default False
        whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        the type of statistic to use, either 'mle' (i.e. sample statistic)
        or 'mean' (i.e. bootstrapped mean statistic)

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
    x_data_ = np.array(x_data)
    y_data_ = np.array(y_data)
    xerr_ = np.array(xerr)
    yerr_ = np.array(yerr)

    if plotly:
        plotlying._master_plot(
            x_data_,
            y_data_,
            xerr=xerr_,
            yerr=yerr_,
            title=title,
            method_name=method_name,
            plot_type="ΔΔG",
            filename=filename,
            target_name=target_name,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
            **kwargs,
        )

    else:
        _master_plot(
            x_data_,
            y_data_,
            xerr=xerr_,
            yerr=yerr_,
            title=title,
            method_name=method_name,
            filename=filename,
            target_name=target_name,
            shift=shift,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
            **kwargs,
        )
