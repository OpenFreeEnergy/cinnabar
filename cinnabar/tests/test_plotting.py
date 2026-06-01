import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import pytest
from openff.units import unit

from cinnabar import FEMap, plotting
from cinnabar.measurements import ReferenceState


@pytest.fixture(scope="function")
def show_called(monkeypatch):
    """Fixture to mock plt.show() and track if it was called."""
    called = {}

    def mock_show():
        called["show"] = True

    monkeypatch.setattr(plt, "show", mock_show)
    return called


def test_plot_ddgs_to_file(tmp_path, fe_map):
    output_file = tmp_path / "ddg_plot.png"
    _ = plotting.plot_DDGs(fe_map, source="", filename=output_file)
    assert output_file.exists()


def test_plot_ddgs_show(fe_map, show_called):
    _ = plotting.plot_DDGs(fe_map, source="", filename=None)
    assert "show" in show_called


def test_plot_dgs_to_file(tmp_path, fe_map):
    output_file = tmp_path / "dg_plot.png"
    fe_map.generate_absolute_values()
    _ = plotting.plot_DGs(fe_map, source="MLE", filename=output_file)
    assert output_file.exists()


def test_plot_dgs_show(fe_map, show_called):
    fe_map.generate_absolute_values()
    _ = plotting.plot_DGs(fe_map, source="MLE", filename=None)
    assert "show" in show_called


def test_plot_all_ddgs_to_file(tmp_path, fe_map):
    output_file = tmp_path / "all_ddg_plot.png"
    fe_map.generate_absolute_values()
    _ = plotting.plot_all_DDGs(fe_map, source="MLE", filename=output_file)
    assert output_file.exists()


def test_plot_all_ddgs_show(fe_map, show_called):
    fe_map.generate_absolute_values()
    _ = plotting.plot_all_DDGs(fe_map, source="MLE", filename=None)
    assert "show" in show_called


def test_plot_ddgs_symm_and_map(fe_map):
    """Test that plotting DDGs with both symmetrise and map raises an error."""
    with pytest.raises(AssertionError):
        _ = plotting.plot_DDGs(fe_map, source="", symmetrise=True, map_positive=True)


def test_plot_ddgs_data_labels_and_plotly(fe_map):
    """Test that data labels and plotly backend raise an error."""
    with pytest.raises(AssertionError):
        _ = plotting.plot_DDGs(fe_map, source="", plotly=True, data_label_type="small-molecule")


@pytest.mark.parametrize("data_label_type", ["small-molecule", "protein-mutation"])
def test_plot_ddgs_data_labels(fe_map, data_label_type, show_called):
    """Test that data labels can be set."""

    _ = plotting.plot_DDGs(fe_map, source="", data_label_type=data_label_type)
    assert "show" in show_called


@pytest.mark.parametrize("data_label_type", ["small-molecule", "protein-mutation"])
def test_plot_ddgs_negative_data_labels(show_called, data_label_type):
    """Test that negative data labels work."""

    fe_map = FEMap()
    fe_map.add_relative_calculation(
        labelA="-ligand1",
        labelB="-ligand2",
        value=2.5 * unit.kilocalorie_per_mole,
        uncertainty=0.5 * unit.kilocalorie_per_mole,
        source="test",
    )
    fe_map.add_experimental_measurement(
        label="-ligand1",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
        source="test",
    )
    fe_map.add_experimental_measurement(
        label="-ligand2",
        value=1.5 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
        source="test",
    )
    _ = plotting.plot_DDGs(fe_map, source="test", data_label_type=data_label_type, map_positive=False)
    assert "show" in show_called


def test_plot_ddgs_negative_bad_labels():
    """Test that negative data labels with bad label type raise an error."""
    fe_map = FEMap()
    fe_map.add_relative_calculation(
        labelA="-ligand1",
        labelB="-ligand2",
        value=2.5 * unit.kilocalorie_per_mole,
        uncertainty=0.5 * unit.kilocalorie_per_mole,
    )
    fe_map.add_experimental_measurement(
        label="-ligand1",
        value=-1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
    )
    fe_map.add_experimental_measurement(
        label="-ligand2",
        value=1.5 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
    )
    with pytest.raises(ValueError, match="data_label_type unsupported. supported types:"):
        _ = plotting.plot_DDGs(fe_map, source="", data_label_type="invalid-label-type")


def test_plot_ddgs_bad_labels(fe_map):
    """Test that bad data labels raise an error."""
    with pytest.raises(ValueError, match="data_label_type unsupported. supported types:"):
        _ = plotting.plot_DDGs(fe_map, source="", data_label_type="invalid-label-type")


def test_plot_ddgs_symmetrise(fe_map, show_called):
    """Test that symmetrise option works."""
    _ = plotting.plot_DDGs(fe_map, source="", symmetrise=True)
    assert "show" in show_called


def test_plot_ddgs_map_positive(fe_map, show_called):
    """Test that map_positive option works."""
    _ = plotting.plot_DDGs(fe_map, source="", map_positive=True)
    assert "show" in show_called


def test_plot_ddgs_no_results():
    """Test that trying to plot nothing raises a clear error"""
    fe_map = FEMap()
    with pytest.raises(ValueError, match="The FEMap contains no computational edges."):
        plotting.plot_DDGs(fe_map, source="")


@pytest.mark.parametrize(
    "plot_func",
    [
        pytest.param(plotting.plot_DDGs, id="plot_DDGs"),
        pytest.param(plotting.plot_DGs, id="plot_DGs"),
        pytest.param(plotting.plot_all_DDGs, id="plot_all_DDGs"),
    ],
)
def test_pair_plot_bad_source(fe_map, plot_func):
    """Test that trying to plot with a bad source raises an error."""
    fe_map.generate_absolute_values()
    with pytest.raises(ValueError, match="Source bad_source is not a valid source"):
        plot_func(fe_map, source="bad_source")


@pytest.mark.parametrize(
    "plot_func",
    [
        pytest.param(plotting.plot_all_DDGs, id="plot_all_DDGs"),
        pytest.param(plotting.plot_DGs, id="plot_DGs"),
    ],
)
def test_pair_plots_no_absolute_values(fe_map, plot_func):
    """Test that trying to plot without absolute values raises an error for DG and pairwise DDG plots."""
    with pytest.raises(ValueError, match="The FEMap contains no computed absolute values"):
        plot_func(fe_map, source="")


def test_plot_dgs_centralising(fe_map, show_called):
    """Test that centralising option works."""
    fe_map.generate_absolute_values()
    _ = plotting.plot_DGs(fe_map, source="MLE", centralizing=True)
    assert "show" in show_called


def test_master_plot_bad_statistic_type(example_data_mle):
    """Test that bad statistic in master plot raises an error."""

    x_data, y_data, xerr, yerr = example_data_mle
    with pytest.raises(ValueError, match="Unknown statistic type bad_stat"):
        _ = plotting._master_plot(
            x_data,
            y_data,
            statistic_type="bad_stat",
        )


def test_master_plot_xy_lim(example_data_mle, show_called):
    """Test that x and y limits are set correctly in master plot."""

    x_data, y_data, xerr, yerr = example_data_mle
    lims = [-10, 10]
    fig = plotting._master_plot(x_data, y_data, filename=None, xy_lim=lims)
    # inspect the figure axes to check axis limits
    axes = fig.get_axes()
    assert axes[0].get_xlim() == tuple(lims)
    assert axes[0].get_ylim() == tuple(lims)
    assert "show" in show_called


def test_master_plot_axis_labels(example_data_mle, show_called):
    """Test that axis labels are set correctly in master plot."""

    x_data, y_data, xerr, yerr = example_data_mle
    x_label = "True Values"
    y_label = "Predicted Values"
    quantity = "DG"
    units = "kcal/mol"
    fig = plotting._master_plot(
        x_data,
        y_data,
        filename=None,
        xlabel=x_label,
        ylabel=y_label,
        quantity=quantity,
        units=units,
    )
    # inspect the figure axes to check axis labels
    axes = fig.get_axes()
    assert axes[0].get_xlabel() == f"{x_label} {quantity} ({units})"
    assert axes[0].get_ylabel() == f"{y_label} {quantity} ({units})"
    assert "show" in show_called


def test_master_plot_stats(example_data_mle, show_called):
    """Test that statistics are included in the master plot title."""

    x_data, y_data, xerr, yerr = example_data_mle
    title = "Test Plot"
    target_name = "Test Target"
    # add some non-default statistics
    statistics = ["RMSE", "MUE", "rho", "KTAU"]
    fig = plotting._master_plot(
        x_data,
        y_data,
        filename=None,
        title=title,
        target_name=target_name,
        statistics=statistics,
    )
    # inspect the figure title to check for statistics
    title_text = fig.axes[0].get_title(loc="right")
    for stat in statistics:
        assert stat in title_text
    assert "show" in show_called


def test_master_plot_clashing_scatter_kwargs(example_data_mle, show_called):
    """Test that clashing scatter_kwargs will work with the scatter kwargs taking precedence."""

    x_data, y_data, xerr, yerr = example_data_mle
    fig = plotting._master_plot(
        x_data,
        y_data,
        filename=None,
        scatter_kwargs={"edgecolors": "black", "s": 20, "marker": "o"},
    )
    # inspect the scatter plot to check for edgecolors
    scatter = fig.axes[0].collections[-1]  # the scatter is plotted first so it's the last collection
    assert tuple(scatter.get_edgecolors()[0][0:3]) == (0.0, 0.0, 0.0)  # black in RGB
    assert "show" in show_called


def test_plot_ecdf_ddgs(fe_map, tmp_path):
    """Test ECDF DDG plotting function."""
    output_file = tmp_path / "test_ecdf_ddgs.png"
    fig = plotting.ecdf_plot_DDGs(fe_map, filename=output_file.as_posix())
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Edgewise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    assert axes.get_title() == "ECDF of Edgewise Absolute Errors"
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_ddgs_missing_data(tmp_path, ecdf_femap_missing_exp_data):
    """Test ECDF DDG plotting function with missing experimental data (one edge skipped)."""
    output_file = tmp_path / "test_ecdf_ddgs_missing_data.png"
    fig = plotting.ecdf_plot_DDGs(ecdf_femap_missing_exp_data, filename=output_file.as_posix())
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Edgewise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_ddgs_no_computational_edges():
    """Test that a FEMap with no computational edges raises a clear error."""
    fe = FEMap()
    fe.add_experimental_measurement(
        label="ligand1",
        value=-7.0 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
    )
    with pytest.raises(ValueError, match="no computational edges"):
        plotting.ecdf_plot_DDGs(fe)


def test_plot_ecdf_ddgs_invalid_source(fe_map):
    """Test that a requested source not present in the FEMap raises a clear error."""
    with pytest.raises(ValueError, match="No computational edges found for source 'nonexistent'"):
        plotting.ecdf_plot_DDGs(fe_map, sources=["nonexistent"])


@pytest.mark.parametrize(
    "plot_func",
    [
        pytest.param(plotting.ecdf_plot_DDGs, id="ecdf_plot_DDGs"),
        pytest.param(plotting.ecdf_plot_DGs, id="ecdf_plot_DGs"),
        pytest.param(plotting.ecdf_plot_all_DDGs, id="ecdf_plot_all_DDGs"),
    ],
)
def test_plot_ecdf_mismatched_sources_labels(fe_map, plot_func):
    """Test that mismatched sources/labels lengths raise a clear error."""
    with pytest.raises(ValueError, match="must have the same length"):
        # required for the DG and all-to-all DDG plots
        fe_map.generate_absolute_values()
        plot_func(fe_map, sources=[""], labels=["A", "B"])


def test_plot_ecdf_ddgs_inconsistent_sources(fe_map):
    """Test that inconsistent results across sources raise a clear error."""
    fe_map.add_relative_calculation(
        labelA="CAT-13b",
        labelB="CAT-17g",
        value=-2.0 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
        source="Method B",
    )
    with pytest.raises(ValueError, match="Inconsistent number of computational edges across sources"):
        plotting.ecdf_plot_DDGs(fe_map)


def test_plot_ecdf_all_ddgs(fe_map, tmp_path):
    """Test ECDF All DDG plotting function."""
    output_file = tmp_path / "test_ecdf_all_ddgs.png"
    fe_map.generate_absolute_values()
    fig = plotting.ecdf_plot_all_DDGs(fe_map, filename=output_file.as_posix())
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Pairwise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    assert axes.get_title() == "ECDF of Pairwise (all-to-all) Absolute Errors"
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_all_ddgs_missing_data(tmp_path, ecdf_femap_missing_exp_data):
    """Test ECDF All DDG plotting function with missing experimental data."""
    output_file = tmp_path / "test_ecdf_all_ddgs_missing_data.png"
    ecdf_femap_missing_exp_data.generate_absolute_values()
    fig = plotting.ecdf_plot_all_DDGs(ecdf_femap_missing_exp_data, filename=output_file.as_posix())
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Pairwise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    # make sure the file was created
    assert output_file.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        pytest.param(plotting.ecdf_plot_all_DDGs, id="ecdf_plot_all_DDGs"),
        pytest.param(plotting.ecdf_plot_DGs, id="ecdf_plot_DGs"),
    ],
)
def test_plot_ecdf_no_absolute_values(fe_map, plot_func):
    """Test that calling ecdf_plot_all_DDGs without generating absolute values raises a clear error."""
    with pytest.raises(ValueError, match="generate_absolute_values"):
        plot_func(fe_map)


@pytest.mark.parametrize(
    "plot_func, expected",
    [
        pytest.param(
            plotting.ecdf_plot_DGs, "No computed absolute values found for source 'Method C'.", id="ecdf_plot_DGs"
        ),
        pytest.param(
            plotting.ecdf_plot_all_DDGs, "No computational edges found for source 'Method C'.", id="ecdf_plot_all_DDGs"
        ),
    ],
)
def test_plot_ecdf_missing_absolute_values(fe_map, plot_func, expected):
    """Test that the DG and all-to-all DDG plots raise a clear error with missing absolute values."""
    fe_map.generate_absolute_values()
    with pytest.raises(ValueError, match=expected):
        plot_func(fe_map, sources=["Method C"])


@pytest.mark.parametrize(
    "plot_func",
    [
        pytest.param(plotting.ecdf_plot_DGs, id="ecdf_plot_DGs"),
        pytest.param(plotting.ecdf_plot_all_DDGs, id="ecdf_plot_all_DDGs"),
    ],
)
def test_plot_ecdf_inconsistent_sources(fe_map, plot_func):
    """Test that inconsistent results across sources raise a clear error for the DG and all-to-all DDG plots."""
    fe_map.generate_absolute_values()
    fe_map.add_absolute_calculation(
        label="CAT-13b",
        value=-8 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
        source="Method C",
    )
    fe_map.add_absolute_calculation(
        label="CAT-17g",
        value=-9 * unit.kilocalorie_per_mole,
        uncertainty=0.3 * unit.kilocalorie_per_mole,
        source="Method C",
    )
    with pytest.raises(ValueError, match="Inconsistent number of computational"):
        plot_func(fe_map)


@pytest.mark.parametrize("centralising, xlim", [(True, (2.1, 2.2)), (False, (11.5, 11.6))])
def test_plot_ecdf_dgs(fe_map, tmp_path, centralising, xlim):
    """Test ECDF DG plotting function with and without centralizing."""
    fe_map.generate_absolute_values()
    output_file = tmp_path / "test_ecdf_dgs.png"
    fig = plotting.ecdf_plot_DGs(fe_map, filename=output_file.as_posix(), centralizing=centralising)
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert axes.get_xlabel() == r"Nodewise $|\Delta$G$_{calc} - \Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    assert axes.get_title() == "ECDF of Nodewise Absolute Errors"
    assert xlim[0] <= axes.get_xlim()[1] <= xlim[1]
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_ddgs_multiple_sources(fe_map, tmp_path):
    """Test ECDF DDG plotting with two computational sources in a single FEMap."""
    output_file = tmp_path / "test_ecdf_ddgs_multiple.png"
    # add a second source (Method B) with small noise on the same edges
    rng = np.random.default_rng(seed=42)
    for m in list(fe_map):
        if m.computational:
            if isinstance(m.labelA, ReferenceState):
                continue
            noise = rng.normal(0, 0.3) * m.DG.u
            fe_map.add_relative_calculation(
                labelA=m.labelA,
                labelB=m.labelB,
                value=m.DG + noise,
                uncertainty=m.uncertainty,
                source="Method B",
            )
    fig = plotting.ecdf_plot_DDGs(
        fe_map,
        sources=["", "Method B"],
        labels=["Method A", "Method B"],
        filename=output_file.as_posix(),
    )
    assert fig is not None
    # legend should contain both labels
    legend_texts = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
    assert "Method A" in legend_texts
    assert "Method B" in legend_texts
    assert output_file.exists()


def test_plot_ecdf_no_datasets():
    with pytest.raises(ValueError, match="At least one dataset is required to plot an ECDF."):
        plotting.ecdf_plot({})


def test_plot_ecdf_bad_ci(fe_map):
    with pytest.raises(ValueError, match="ci must be between 0 and 1."):
        plotting.ecdf_plot(fe_map, ci=1.1)


def test_plot_ecdf_colors(fe_map, tmp_path):
    """Test ECDF plotting function with custom colors."""
    output_file = tmp_path / "test_ecdf_colors.png"
    fig = plotting.ecdf_plot_DDGs(fe_map, colors=["#FF5733"], filename=output_file.as_posix())
    assert fig is not None
    # check the file was created
    assert output_file.exists()
    # check that the line color matches the specified color
    line = fig.get_axes()[0].lines[0]
    assert line.get_color() == "#FF5733"


def test_plot_cycle_closure(fe_map, tmp_path):
    output_file = tmp_path / "cycle_closure.png"
    fig = plotting.plot_cycle_closure(fe_map, filename=str(output_file))
    assert fig is not None
    axes = fig.get_axes()[0]
    assert axes.get_xlabel() == r"Cycle closure (kcal mol$^{-1}$)"
    assert axes.get_ylabel() == "Count"
    assert output_file.exists()


def test_plot_cycle_closure_show(fe_map, show_called):
    _ = plotting.plot_cycle_closure(fe_map, filename=None)
    assert "show" in show_called


def test_plot_cycle_closure_no_cycles_no_plot(tmp_path):
    fe = FEMap()
    fe.add_relative_calculation(
        "A",
        "B",
        value=1.0 * unit.kilocalorie_per_mole,
        uncertainty=0.1 * unit.kilocalorie_per_mole,
    )
    assert fe.get_cycle_closure_dataframe().empty
    output_file = tmp_path / "cycle_closure.png"
    with pytest.raises(ValueError, match="The FEMap does not contain cycles"):
        plotting.plot_cycle_closure(fe, filename=str(output_file))
    assert not output_file.exists()


def test_plot_cycle_closure_invalid_source(fe_map, tmp_path):
    output_file = tmp_path / "cycle_closure.png"
    with pytest.raises(ValueError, match="No cycles found for sources"):
        plotting.plot_cycle_closure(fe_map, filename=str(output_file), sources=["nonexistent_source"])
    assert not output_file.exists()


def test_plot_cycle_closure_multiple_sources(perfect_cycle, imperfect_cycle, tmp_path):
    fe = perfect_cycle + imperfect_cycle
    output_file = tmp_path / "cycle_closure_multiple.png"
    fig = plotting.plot_cycle_closure(fe, filename=str(output_file))
    assert fig is not None
    axes = fig.get_axes()[0]
    assert axes.get_xlabel() == r"Cycle closure (kcal mol$^{-1}$)"
    assert axes.get_ylabel() == "Count"
    legend_texts = [t.get_text() for t in axes.get_legend().get_texts()]
    assert "method_a" in legend_texts
    assert "method_b" in legend_texts
    assert output_file.exists()
