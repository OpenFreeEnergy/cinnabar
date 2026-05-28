import itertools
from typing import Any, Literal

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import seaborn as sns
from adjustText import adjust_text

from cinnabar import plotlying, stats
from cinnabar.femap import FEMap


def _master_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
    method_name: str = "",
    target_name: str = "",
    quantity: str = r"$\Delta\Delta$G",
    xlabel: str = "Experimental",
    ylabel: str = "Calculated",
    units: str = r"$\mathrm{kcal\,mol^{-1}}$",
    guidelines: bool = True,
    origins: bool = True,
    color: str | None = None,
    statistics: list[Literal["RMSE", "NRMSE", "MUE", "RAE", "R2", "rho", "KTAU", "PI"]] | None = None,
    filename: str | None = None,
    centralizing: bool = True,
    shift: float = 0.0,
    figsize: float = 3.25,
    dpi: float | str = "figure",
    data_labels: list[str] | None = None,
    axis_padding: float = 0.5,
    xy_lim: tuple[float, float] | None = None,
    font_sizes: dict[str, int] | None = None,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: Literal["mle", "mean"] = "mle",
    scatter_kwargs: dict[str, float | int | str] | None = None,
):
    r"""Handles the aesthetics of the plots in one place.

    Parameters
    ----------
    x : np.ndarray
        Values to plot on the x-axis
    y : np.ndarray
        Values to plot on the y-axis
    title : string, default ""
        Title for the plot
    xerr : np.ndarray | None , default None
        Error bars for x values if available.
    yerr : np.ndarray | None , default None
        Error bars for y values if available.
    method_name : string, default ""
        Name of method associated with results, e.g. "openfe".
    target_name : string, default ""
        Name of system for results, e.g. "Thrombin".
    quantity : str, default r"$\Delta\Delta$G"
        Metric that is being plotted, which will be included in the axis labels.
    xlabel : str, default "Experimental"
        Main abel for the x-axis.
    ylabel : str, default "Calculated"
        Main label for the y-axis.
    units : str, default r"$\mathrm{kcal\,mol^{-1}}$"
        String value of units to label axis with.
    guidelines : bool, default True
        Toggles plotting of grey 0.5 and 1 kcal/mol error zone.
    origins : bool, default True
        Toggles plotting of x and y-axis.
    color : str, default None
        The name of the colour scheme for the scatter plots. If None, will be coloured according to distance from unity
    statistics : list[{'RMSE', 'NRMSE', 'MUE', 'RAE', 'R2', 'rho', 'KTAU', 'PI'}] | None, default None
        List of statistics to calculate and report on the plot, if None "RMSE" and "MUE" will be reported.
    filename : str | None, default None
        Filename for plot, if not provided the plot will be displayed.
    centralizing : bool, default True
        Offset the free energies ``True`` or report raw values ``False``.
    shift : float, default 0.
        Shift both the x and y-axis by a constant.
    figsize : float, default 3.25
        Size of figure for matplotlib.
    dpi : float or 'figure', default 'figure'
        The resolution in dots per inch
        if 'figure', uses the figure's dpi value (this behavior is copied from
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
    data_labels : list[str] | None, default None
        List of labels for each data point.
    axis_padding : float, default 0.5
        Padding to add to maximum axis value and subtract from the minimum axis value.
    xy_lim : tuple[float, float] | None, default None
        Contains the minimum and maximum values to use for the x and y-axis. if specified, ``axis_padding`` is ignored.
    font_sizes : dict[str, int] | None, default None
        Font sizes to use for the title ("title"), the data labels ("labels"), and the rest of the plot ("other").
        Defaults to {"title": 12, "labels": 9, "other": 12}
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping.
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping.
    statistic_type : {"mle", "mean"}, default "mle"
        The type of statistic to use, either "mle" (i.e. sample statistic) or "mean" (i.e. bootstrapped mean statistic)
    scatter_kwargs : dict[str, float | int | str] | None, default None
        Arguments to control plt.scatter(), these will override the default cinnabar settings.

    Returns
    -------

    """
    nsamples = len(x)

    # aesthetics
    if font_sizes is None:
        font_sizes = {"title": 12, "labels": 9, "other": 12}

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
        scale = (ax_min, ax_max)

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
        _color = np.abs(x - y)
        # 2.372 kcal / mol = 4 RT
        _color = cm(_color / 2.372)
    else:
        _color = color

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
    # add our cinnabar preset settings to the scatter_kwargs to make sure they do not clash
    # scatter kwargs will override the default settings
    default_kwargs: dict[str, Any] = {
        "color": _color,
        "zorder": 2,
        "edgecolors": "dimgrey",
        "linewidths": 0.7,
        "s": 20,
        "marker": "o",
    }
    if scatter_kwargs is not None:
        default_kwargs.update(scatter_kwargs)
    plt.scatter(x, y, **default_kwargs)

    # Label points
    if data_labels is not None:
        texts = []
        for i, label in enumerate(data_labels):
            texts.append(plt.text(x[i] + 0.03, y[i] + 0.03, label, fontsize=font_sizes["labels"]))
        adjust_text(texts)

    # stats and title
    statistics_string = ""
    if statistic_type not in ["mle", "mean"]:
        raise ValueError(f"Unknown statistic type {statistic_type}")

    # set default statistics values if not provided
    if statistics is None:
        statistics = ["RMSE", "MUE"]

    for statistic in statistics:
        s = stats.bootstrap_statistic(
            x,
            y,
            xerr,
            yerr,
            statistic=statistic,
            include_true_uncertainty=bootstrap_x_uncertainty,
            include_pred_uncertainty=bootstrap_y_uncertainty,
        )
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
    femap: FEMap,
    source: str,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    map_positive: bool = False,
    filename: str | None = None,
    symmetrise: bool = False,
    plotly: bool = False,
    data_label_type: Literal["small-molecule", "protein-mutation"] | None = None,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: Literal["mle", "mean"] = "mle",
    **kwargs,
):
    """Function to plot relative free energies

    Parameters
    ----------
    femap : FEMap
        FEMap object with relative and absolute free energy edges
    source : str
        The source label of the computational relative free energies to plot against experiment.
        This must match the ``source`` field used when the calculations were added to the
        ``FEMap``.  For data loaded via ``FEMap.from_csv``
        (or added without an explicit source), pass ``source=""``.
    method_name : string, default ""
        Name of method associated with results, e.g. "openfe" by default an empty string.
    target_name : string, default ""
        Name of system for results, e.g. "Thrombin".
    title : string, default ""
        Title for the plot.
    map_positive : bool, default False
        Whether to map all DDGs to the positive x values.
        This is an aesthetic choice
    filename : str | None, default None
        Filename for plot, if None the plot will be displayed.
    symmetrise : bool, default False
        Whether to plot each datapoint twice, both positive and negative values.
    plotly : bool, default False
        Whether to use plotly to generate the plot.
    data_label_type : {"small-molecule", "protein-mutation"} | None, default None
        Type of data label to add to each edge

        if ``None`` data labels will not be added

        if ``'small-molecule'`` edge labels will be ``f"{node_A_name}→{node_B_name}"``.

        if ``'protein-mutation'`` edge labels will given as single letter amino
        acid codes separated by the mutated residue index (eg. ``"Y29A"``)

        If both node names start with ``"-"``, the negative sign will be factored
        out (eg. ``"-(Y29A)"`` or ``"-(benzene→toluene)"``).

        currently unsupported for plotly-generated plots

        .. TODO: implement data labeling for the case where plotly=True

    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping.
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping.
    statistic_type : {"mle", "mean"}, default "mle"
        The type of statistic to use, either "mle" (i.e. sample statistic) or "mean" (i.e. bootstrapped mean statistic).

    Returns
    -------
    Nothing
    """

    assert int(symmetrise) + int(map_positive) != 2, "Symmetrise and map_positive cannot both be True in the same plot"

    if data_label_type:
        assert not plotly, "We currently do not support data labeling for plotly-generated plots"

    # load the data using the internal dataframes
    rel_df = femap.get_relative_dataframe()
    comp_mask = rel_df["computational"]
    all_comp_sources = rel_df.loc[comp_mask, "source"].unique().tolist()

    if not all_comp_sources:
        raise ValueError("The FEMap contains no computational edges.")

    if source not in all_comp_sources:
        raise ValueError(f"Source {source} is not a valid source, available sources: {all_comp_sources}")

    # get the comp data from the computational source
    comp_data = rel_df[comp_mask & (rel_df["source"] == source)]
    # get the experimental data
    exp_data = rel_df[~comp_mask].rename(
        columns={"DDG (kcal/mol)": "DDG_exp", "uncertainty (kcal/mol)": "uncertainty_exp"}
    )

    # merge to align the data and drop any values missing an experimental data point
    merged = comp_data.merge(exp_data, how="left", on=["labelA", "labelB"])
    merged = merged.dropna(subset=["DDG_exp"])

    # extract the required data
    x = merged["DDG_exp"].to_numpy(copy=True)
    y = merged["DDG (kcal/mol)"].to_numpy(copy=True)
    xerr = merged["uncertainty_exp"].to_numpy(copy=True)
    yerr = merged["uncertainty (kcal/mol)"].to_numpy(copy=True)

    # labels
    data_labels: list[str] = []
    if data_label_type:
        for _, row in merged.iterrows():
            node_a_name = row["labelA"]
            node_b_name = row["labelB"]
            if node_a_name.startswith("-") and node_b_name.startswith("-"):
                # factor out "-" if both start with it
                node_a_name = node_a_name[1:]
                node_b_name = node_b_name[1:]
                prefix = "-"
            else:
                prefix = ""
            if data_label_type == "small-molecule":
                data_labels.append(f"{prefix}({node_a_name}→{node_b_name})")
            elif data_label_type == "protein-mutation":
                data_labels.append(f"{prefix}({node_a_name}{node_b_name[0]})")
            else:
                raise ValueError(
                    "data_label_type unsupported. supported types: 'small-molecule' and 'protein-mutation'"
                )

    if symmetrise:
        x_data = np.append(x, [-i for i in x])
        y_data = np.append(y, [-i for i in y])
        xerr = np.append(xerr, xerr)
        yerr = np.append(yerr, yerr)
        # double the data labels list
        data_labels = data_labels + data_labels

    elif map_positive:
        x_data_map = []
        y_data_map = []
        for i, j in zip(x, y):
            if i < 0:
                x_data_map.append(-i)
                y_data_map.append(-j)
            else:
                x_data_map.append(i)
                y_data_map.append(j)
        x_data = np.asarray(x_data_map)
        y_data = np.asarray(y_data_map)
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
    femap: FEMap,
    source: str,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: str | None = None,
    plotly: bool = False,
    centralizing: bool = True,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: Literal["mle", "mean"] = "mle",
    **kwargs,
):
    """Function to plot absolute free energies.

    Parameters
    ----------
    femap : FEMap
        FEMap object with absolute free energies to plot.
    source : str
        The name of the source label of the computational absolute values, if absolute values are generated with the MLE estimator this should be "MLE".
    method_name : string, default ""
        Name of method associated with results, e.g. "openfe" by default an empty string.
    target_name : string, default ""
        Name of system for results, e.g. "Thrombin" by default an empty string.
    title : string, default ""
        Title for the plot.
    filename : str | None, default None
        Filename for plot if None the plot will be displayed.
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping.
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping.
    statistic_type : {"mle", "mean"}, default "mle"
        The type of statistic to use, either "mle" (i.e. sample statistic) or "mean" (i.e. bootstrapped mean statistic).

    Returns
    -------

    """

    # extract the data from the internal dataframes
    df = femap.get_absolute_dataframe()
    comp_mask = df["computational"]
    all_comp_sources = df.loc[comp_mask, "source"].unique().tolist()

    if not all_comp_sources:
        raise ValueError(
            f"The FEMap contains no computed absolute values. "
            "Call generate_absolute_values() first or add calculated absolute measurements directly."
        )

    if source not in all_comp_sources:
        raise ValueError(f"Source {source} is not a valid source, available sources: {all_comp_sources}")

    # get the comp data from the computational source
    comp_data = df[comp_mask & (df["source"] == source)]
    # get the experimental data
    exp_data = df[~comp_mask].rename(columns={"DG (kcal/mol)": "DG_exp", "uncertainty (kcal/mol)": "uncertainty_exp"})

    # merge to align the data and drop any values missing an experimental data point
    merged = comp_data.merge(exp_data, how="left", on=["label"])
    merged = merged.dropna(subset=["DG_exp"])

    # extract the required data
    x_data = merged["DG_exp"].to_numpy(copy=True)
    y_data = merged["DG (kcal/mol)"].to_numpy(copy=True)
    xerr = merged["uncertainty_exp"].to_numpy(copy=True)
    yerr = merged["uncertainty (kcal/mol)"].to_numpy(copy=True)

    # centralising
    # this should be replaced by providing one experimental result
    if centralizing:
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
            quantity=r"$\Delta$G",
            title=title,
            method_name=method_name,
            target_name=target_name,
            filename=filename,
            bootstrap_x_uncertainty=bootstrap_x_uncertainty,
            bootstrap_y_uncertainty=bootstrap_y_uncertainty,
            statistic_type=statistic_type,
            **kwargs,
        )


def plot_all_DDGs(
    femap: FEMap,
    source: str,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: str | None = None,
    plotly: bool = False,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: Literal["mle", "mean"] = "mle",
    **kwargs,
):
    """Plots relative free energies between all ligands, which is calculated from
    the differences between all the absolute free energies. This data is different to ``plot_DGs``.

    Parameters
    ----------
    femap : FEMap
        FEMap object with absolute free energies to plot.
    source : str
        The name of the source label of the computational absolute values, if absolute values are generated with the MLE estimator this should be "MLE".
    method_name : string, default ""
        Name of method associated with results, e.g. "openfe" by default an empty string.
    target_name : string, default ""
        Name of system for results, e.g. "Thrombin" by default an empty string.
    title : string, default ""
        Title for the plot.
    filename : str | None, default None
        Filename for plot if None the plot will be displayed.
    plotly : bool, default True
        Whether to use plotly for the plotting.
    shift : float, default 0.
        Shift both the x and y-axis by a constant.
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping.
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping.
    statistic_type : {"mle", "mean"}, default "mle"
        The type of statistic to use, either "mle" (i.e. sample statistic) or "mean" (i.e. bootstrapped mean statistic).

    Returns
    -------

    """
    # use the internal dataframes to get the pairwise differences
    rel_df = femap.get_all_to_all_relative_dataframe(symmetrical=True)
    comp_mask = rel_df["computational"]
    all_comp_sources = rel_df.loc[comp_mask, "source"].unique().tolist()

    if not all_comp_sources:
        raise ValueError(
            f"The FEMap contains no computed absolute values which are need to obtain the all-to-all pairwise DDGs. "
            "Call generate_absolute_values() first or add calculated absolute measurements directly."
        )

    if source not in all_comp_sources:
        raise ValueError(f"Source {source} is not a valid source, available sources: {all_comp_sources}")

    # get the comp data from the computational source
    comp_data = rel_df[comp_mask & (rel_df["source"] == source)]
    # get the experimental data
    exp_data = rel_df[~comp_mask].rename(
        columns={"DDG (kcal/mol)": "DDG_exp", "uncertainty (kcal/mol)": "uncertainty_exp"}
    )

    # merge to align the data and drop any values missing an experimental data point
    merged = comp_data.merge(exp_data, how="left", on=["labelA", "labelB"])
    merged = merged.dropna(subset=["DDG_exp"])

    # extract the required data
    x_data = merged["DDG_exp"].to_numpy(copy=True)
    y_data = merged["DDG (kcal/mol)"].to_numpy(copy=True)
    xerr = merged["uncertainty_exp"].to_numpy(copy=True)
    yerr = merged["uncertainty (kcal/mol)"].to_numpy(copy=True)

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


def ecdf_plot(
    datasets: dict[str, np.ndarray],
    title: str | None = "ECDF of Absolute Errors",
    xlabel: str = "Pairwise",
    quantity: str = r"$|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$",
    units: str = r"$\mathrm{kcal\,mol^{-1}}$",
    ylabel: str = "Cumulative Probability",
    figsize: float | tuple[float, float] = 4,
    colors: list[str] | None = None,
    ecdf_kwargs: dict[str, Any] | None = None,
    filename: str | None = None,
    nbootstraps: int = 1_000,
    ci: float = 0.95,
) -> plt.Figure:
    r"""
    Plot ECDFs for one or more datasets. Where the dataset is a flat array of absolute errors.

    Parameters
    ----------
    datasets : dict[str, np.ndarray]
        A dictionary where keys are dataset labels and values are the data arrays.
    title: str | None, default "ECDF of Absolute Errors"
        Title for the plot. If None, no title is set.
    xlabel : str, default "Absolute Error"
        Label for the x-axis.
    quantity : str, default r"$\Delta\Delta$G"
        Metric that is being plotted.
    units : str, default r"$\mathrm{kcal\,mol^{-1}}$"
        Units of the metric being plotted.
    ylabel : str, default "Cumulative Probability"
        Label for the y-axis.
    figsize : float | tuple[float, float], default 4
        Size of the figure.
    colors : list[str] | None, default None
        List of colors for each dataset. If None, default colors are used.
    ecdf_kwargs : dict, default None
        Additional keyword arguments to pass to seaborn.ecdfplot.
    filename : str | None, default None
        If provided, the plot will be saved to this filename.
    nbootstraps : int, default = 1_000
        Number of bootstraps to perform for estimating confidence intervals.
    ci : float, default = 0.95
        Confidence level for the confidence intervals (e.g., 0.95 for 95% confidence intervals).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the ECDF plot which can be edited further.
    """
    if ecdf_kwargs is None:
        ecdf_kwargs = {}

    if not datasets:
        raise ValueError("At least one dataset is required to plot an ECDF.")

    # make sure 0 < ci < 1
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")

    if not isinstance(figsize, tuple):
        figsize = (figsize, figsize)
    fig, axs = plt.subplots(figsize=figsize)

    # make the default ecdf_kwargs for the plot
    default_kwargs = {
        "ax": axs,
        "linewidth": 2,
    }
    default_kwargs.update(ecdf_kwargs)

    if colors is None:
        # get the default colors
        colors = sns.color_palette(None, n_colors=len(datasets))

    # Iterate over the dictionary to plot ECDFs
    for i, (label, data) in enumerate(datasets.items()):
        # Pick a color for the dataset if specified
        color = colors[i]
        default_kwargs["color"] = color

        # we first need to sort the data so its in the same order if CIs are plotted
        data = np.sort(data)
        sns.ecdfplot(data, label=label, **default_kwargs)

        # estimate an error bounds via bootsrapping over the data if the number of bootstraps is > 0
        if nbootstraps > 0:
            boot_ecdfs = []
            for _ in range(nbootstraps):
                sample = np.random.choice(data, size=len(data), replace=True)
                # calculate the ECDF for this sample at each point in the true sample data
                # searchsorted returns an array of indices in sample where the values in data would fit in
                # by dividing by the length of the sample we get the fraction of sample points that are less than or
                # equal to each point in the data, which is the ECDF value at that point
                boot_ecdf = np.searchsorted(np.sort(sample), data, side="right") / len(sample)
                boot_ecdfs.append(boot_ecdf)

            # calculate the confidence interval based on the user input
            low_percentile = (1.0 - ci) / 2.0 * 100
            high_percentile = 100 - low_percentile
            lower = np.percentile(boot_ecdfs, low_percentile, axis=0)
            upper = np.percentile(boot_ecdfs, high_percentile, axis=0)
            # now plot a shaded region between the confidence intervals
            plt.fill_between(data, lower, upper, alpha=0.2, color=color)

    if title is not None:
        plt.title(title, fontsize=14)

    plt.xlabel(f"{xlabel} {quantity} ({units})", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(left=0)
    plt.legend()
    # add gridlines to help identify 1, 2 kcal/mol errors
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    return fig


def ecdf_plot_DDGs(
    graphs: list[FEMap | nx.MultiDiGraph],
    labels: list[str],
    title: str | None = "ECDF of Edgewise Absolute Errors",
    filename: str | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot ECDF of absolute errors for edgewise relative free energies in a graph.

    Parameters
    ----------
    graphs: list[FEMap | nx.MultiDiGraph]
        A list of graph objects with relative free energy edges.
    labels: list[str]
        A list of labels corresponding to each graph, these will be used in the legend.
    title : str | None, default "ECDF of Absolute Errors"
        Title for the plot. If None, no title is set.
    filename : str | None, default None
        If provided, the plot will be saved to this filename.
    **kwargs
        Additional keyword arguments to pass to `ecdf_plot`.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the ECDF plot which can be edited further.

    Notes
    -----
    We assume that the graphs have edges with 'calc_DDG' and 'exp_DDG' attributes. If any edges are missing an experimental value,
    they will be skipped in the absolute error calculation.

    Raises
    ------
    ValueError
        If any edges are missing a calculated DDG value.
    """
    # extract the edgewise absolute errors for each graph
    datasets = {}
    for graph, label in zip(graphs, labels):
        # handle the case where a FEMap is provided
        if isinstance(graph, FEMap):
            graph = graph.to_legacy_graph()

        # if the experimental value is missing, add a nan so we can filter it out
        x = np.array([x[2].get("exp_DDG", np.nan) for x in graph.edges(data=True)])
        y = np.array([x[2].get("calc_DDG", np.nan) for x in graph.edges(data=True)])
        # if any calculated values are missing raise an error
        if np.any(np.isnan(y)) or y.size == 0:
            raise ValueError(
                f"Graph with label {label} has edges with missing calculated DDG values, which should be stored as `calc_DDG`."
            )
        # filter out edges with missing experimental values
        mask = ~np.isnan(x)
        x = x[mask]
        y = y[mask]
        abs_errors = np.abs(y - x)
        datasets[label] = abs_errors

    fig = ecdf_plot(
        datasets,
        title=title,
        filename=filename,
        xlabel="Edgewise",
        **kwargs,
    )
    return fig


def ecdf_plot_DGs(
    graphs: list[FEMap | nx.MultiDiGraph],
    labels: list[str],
    title: str | None = "ECDF of Nodewise Absolute Errors",
    filename: str | None = None,
    centralizing: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Plot ECDF of absolute errors for nodewise absolute free energies in a graph.

    Parameters
    ----------
    graphs: list[FEMap | nx.MultiDiGraph]
        A list of graph objects with relative free energy edges.
    labels: list[str]
        A list of labels corresponding to each graph, these will be used in the legend.
    title : str | None, default "ECDF of Absolute Errors"
        Title for the plot. If None, no title is set.
    filename : str | None, default None
        If provided, the plot will be saved to this filename.
    centralizing : bool, default True
        whether to center both calculated and experimental values around zero before calculating absolute errors.
    **kwargs
        Additional keyword arguments to pass to `ecdf_plot`.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the ECDF plot which can be edited further.

    Notes
    -----
    We assume that the graphs have nodes with 'calc_DG' and 'exp_DG' attributes. The absolute errors are calculated after centering both
    calculated and experimental values around zero.

    Raises
    ------
    ValueError
        If any nodes are missing a calculated DG value.
    """
    # extract the nodewise absolute errors for each graph
    datasets = {}
    for graph, label in zip(graphs, labels):
        # handle the case where a FEMap is provided
        if isinstance(graph, FEMap):
            graph = graph.to_legacy_graph()

        # if the experimental value is missing, add a nan so we can filter it out
        x = np.array([node[1].get("exp_DG", np.nan) for node in graph.nodes(data=True)])
        y = np.array([node[1].get("calc_DG", np.nan) for node in graph.nodes(data=True)])
        # if any nodes are missing calculated values raise an error
        if np.any(np.isnan(y)) or y.size == 0:
            raise ValueError(
                f"Graph with label {label} has nodes with missing calculated DG values, which should be stored as `calc_DG`."
            )
        # filter out nodes with missing experimental values
        mask = ~np.isnan(x)
        x = x[mask]
        y = y[mask]
        # we need to shift the arrays to both be centered around zero if requested
        if centralizing:
            x -= np.mean(x)
            y -= np.mean(y)
        abs_errors = np.abs(y - x)
        datasets[label] = abs_errors

    fig = ecdf_plot(
        datasets,
        title=title,
        xlabel="Nodewise",
        quantity=r"$|\Delta$G$_{calc} - \Delta$G$_{exp}|$",
        filename=filename,
        **kwargs,
    )
    return fig


def ecdf_plot_all_DDGs(
    graphs: list[FEMap | nx.MultiDiGraph],
    labels: list[str],
    title: str | None = "ECDF of Pairwise (all-to-all) Absolute Errors",
    filename: str | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot ECDF of absolute errors for all-to-all relative free energies calculated from absolute free energies in a graph.

    Parameters
    ----------
    graphs: list[FEMap | nx.MultiDiGraph]
        A list of graph objects with relative free energy edges.
    labels: list[str]
        A list of labels corresponding to each graph, these will be used in the legend.
    title : str | None, default "ECDF of Absolute Errors"
        Title for the plot. If None, no title is set.
    filename : str | None, default None
        If provided, the plot will be saved to this filename.
    **kwargs
        Additional keyword arguments to pass to `ecdf_plot`.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the ECDF plot which can be edited further.

    Notes
    -----
    We assume that the graphs have nodes with 'calc_DG' and 'exp_DG' attributes. If any nodes are missing an experimental value,
    they will be skipped in the absolute error calculation.

    Raises
    ------
    ValueError
        If any nodes are missing a calculated DG value.
    """
    # extract the all-to-all absolute errors for each graph
    datasets = {}
    for graph, label in zip(graphs, labels):
        # handle the case where a FEMap is provided
        if isinstance(graph, FEMap):
            graph = graph.to_legacy_graph()

        nodes = graph.nodes(data=True)

        # if the experimental value is missing, add a nan so we can filter it out
        exp = np.array([node[1].get("exp_DG", np.nan) for node in nodes])
        calc = np.array([node[1].get("calc_DG", np.nan) for node in nodes])
        # if any nodes are missing calculated values raise an error
        if np.any(np.isnan(calc)) or calc.size == 0:
            raise ValueError(
                f"Graph with label {label} has nodes with missing calculated DG values, which should be stored as `calc_DG`."
            )
        # filter out nodes with missing experimental values
        mask = ~np.isnan(exp)
        exp = exp[mask]
        calc = calc[mask]
        # do all to plot_all we are taking the abs error so we only need the error once per pair
        errors = []
        for a, b in itertools.combinations(range(len(calc)), 2):
            # transform a -> b has a DDG of calc[b] - calc[a]
            calc_ddg = calc[b] - calc[a]
            exp_ddg = exp[b] - exp[a]
            errors.append(calc_ddg - exp_ddg)

        datasets[label] = np.abs(errors)

    fig = ecdf_plot(
        datasets,
        title=title,
        xlabel="Pairwise",
        filename=filename,
        **kwargs,
    )
    return fig
