from cinnabar import plotting
import pytest
import matplotlib.pylab as plt

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
    _ = plotting.plot_DDGs(fe_map.to_legacy_graph(), filename=output_file)
    assert output_file.exists()


def test_plot_ddgs_show(fe_map, show_called):
    _ = plotting.plot_DDGs(fe_map.to_legacy_graph(), filename=None)
    assert "show" in show_called

def test_plot_dgs_to_file(tmp_path, fe_map):
    output_file = tmp_path / "dg_plot.png"
    _ = plotting.plot_DGs(fe_map.to_legacy_graph(), filename=output_file)
    assert output_file.exists()


def test_plot_dgs_show(fe_map, show_called):
    _ = plotting.plot_DGs(fe_map.to_legacy_graph(), filename=None)
    assert "show" in show_called


def test_plot_all_ddgs_to_file(tmp_path, fe_map):
    output_file = tmp_path / "all_ddg_plot.png"
    _ = plotting.plot_all_DDGs(fe_map.to_legacy_graph(), filename=output_file)
    assert output_file.exists()


def test_plot_all_ddgs_show(fe_map, show_called):

    _ = plotting.plot_all_DDGs(fe_map.to_legacy_graph(), filename=None)
    assert "show" in show_called


def test_plot_ddgs_symm_and_map(fe_map):
    """Test that plotting DDGs with both symmetrise and map raises an error."""
    graph = fe_map.to_legacy_graph()
    with pytest.raises(AssertionError):
        _ = plotting.plot_DDGs(graph, symmetrise=True, map_positive=True)


def test_plot_ddgs_data_labels_and_plotly(fe_map):
    """Test that data labels and plotly backend raise an error."""
    graph = fe_map.to_legacy_graph()
    with pytest.raises(AssertionError):
        _ = plotting.plot_DDGs(graph, plotly=True, data_label_type="small-molecule")


@pytest.mark.parametrize("data_label_type", ["small-molecule", "protein-mutation"])
def test_plot_ddgs_data_labels(fe_map, data_label_type, show_called):
    """Test that data labels can be set."""

    graph = fe_map.to_legacy_graph()
    _ = plotting.plot_DDGs(graph, data_label_type=data_label_type)
    assert "show" in show_called


def test_plot_ddgs_bad_labels(fe_map):
    """Test that bad data labels raise an error."""
    graph = fe_map.to_legacy_graph()
    with pytest.raises(Exception, match="data_label_type unsupported. supported types:"):
        _ = plotting.plot_DDGs(graph, data_label_type="invalid-label-type")


def test_plot_ddgs_symmetrise(fe_map, show_called):
    """Test that symmetrise option works."""

    graph = fe_map.to_legacy_graph()
    _ = plotting.plot_DDGs(graph, symmetrise=True)
    assert "show" in show_called


def test_plot_ddgs_map_positive(fe_map, show_called):
    """Test that map_positive option works."""

    graph = fe_map.to_legacy_graph()
    _ = plotting.plot_DDGs(graph, map_positive=True)
    assert "show" in show_called


def test_plot_dgs_centralising(fe_map, show_called):
    """Test that centralising option works."""

    graph = fe_map.to_legacy_graph()
    _ = plotting.plot_DGs(graph, centralizing=True)
    assert "show" in show_called


def test_master_plot_bad_statistic_type(example_data):
    """Test that bad statistic in master plot raises an error."""

    x_data, y_data, xerr, yerr = example_data
    with pytest.raises(ValueError, match="Unknown statistic type bad_stat"):
        _ = plotting._master_plot(
            x_data,
            y_data,
            statistic_type="bad_stat",
        )


def test_master_plot_xy_lim(example_data, show_called):
    """Test that x and y limits are set correctly in master plot."""

    x_data, y_data, xerr, yerr = example_data
    lims = [-10, 10]
    fig = plotting._master_plot(
        x_data,
        y_data,
        filename=None,
        xy_lim=lims
    )
    # inspect the figure axes to check axis limits
    axes = fig.get_axes()
    assert axes[0].get_xlim() == tuple(lims)
    assert axes[0].get_ylim() == tuple(lims)
    assert "show" in show_called


def test_master_plot_axis_labels(example_data, show_called):
    """Test that axis labels are set correctly in master plot."""

    x_data, y_data, xerr, yerr = example_data
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
    assert axes[0].get_xlabel() == f"{x_label} {quantity} {units}"
    assert axes[0].get_ylabel() == f"{y_label} {quantity} {units}"
    assert "show" in show_called


def test_master_plot_stats(example_data, show_called):
    """Test that statistics are included in the master plot title."""

    x_data, y_data, xerr, yerr = example_data
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


def test_master_plot_clashing_scatter_kwargs(example_data, show_called):
    """Test that clashing scatter_kwargs will work with the scatter kwargs taking precedence."""

    x_data, y_data, xerr, yerr = example_data
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