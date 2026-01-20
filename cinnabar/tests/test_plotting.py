import networkx as nx
import numpy as np
import pytest

from cinnabar import FEMap, plotting


def test_plot_ecdf_ddgs(fe_map, tmp_path):
    """Test ECDF DDG plotting function."""
    output_file = tmp_path / "test_ecdf_ddgs.png"
    fig = plotting.ecdf_plot_DDGs([fe_map], labels=["Test FE Map"], filename=output_file.as_posix())
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
    """Test ECDF DDG plotting function with missing experimental data."""
    output_file = tmp_path / "test_ecdf_ddgs_missing_data.png"
    fig = plotting.ecdf_plot_DDGs(
        [ecdf_femap_missing_exp_data], labels=["FE Map with Missing Data"], filename=output_file.as_posix()
    )
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Edgewise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_all_ddgs(fe_map, tmp_path):
    """Test ECDF All DDG plotting function."""
    output_file = tmp_path / "test_ecdf_all_ddgs.png"
    fig = plotting.ecdf_plot_all_DDGs([fe_map], labels=["Test FE Map"], filename=output_file.as_posix())
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
    fig = plotting.ecdf_plot_all_DDGs(
        [ecdf_femap_missing_exp_data], labels=["FE Map with Missing Data"], filename=output_file.as_posix()
    )
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert (
        axes.get_xlabel() == r"Pairwise $|\Delta\Delta$G$_{calc} - \Delta\Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    )
    # make sure the file was created
    assert output_file.exists()


@pytest.mark.parametrize("centralising, xlim", [(True, (2.1, 2.2)), (False, (11.5, 11.6))])
def test_plot_ecdf_dgs(fe_map, tmp_path, centralising, xlim):
    """Test ECDF DG plotting function with and without centralizing."""
    output_file = tmp_path / "test_ecdf_dgs.png"
    fig = plotting.ecdf_plot_DGs([fe_map], labels=["Test FE Map"], filename=output_file.as_posix(), centralizing=centralising)
    assert fig is not None
    # check the axis are labeled correctly
    axes = fig.get_axes()[0]
    assert axes.get_ylabel() == "Cumulative Probability"
    assert axes.get_xlabel() == r"Nodewise $|\Delta$G$_{calc} - \Delta$G$_{exp}|$ ($\mathrm{kcal\,mol^{-1}}$)"
    assert axes.get_title() == "ECDF of Nodewise Absolute Errors"
    assert xlim[0] <= axes.get_xlim()[1] <= xlim[1]
    # make sure the file was created
    assert output_file.exists()


def test_plot_ecdf_ddgs_multiple(fe_map, tmp_path):
    """Test ECDF DDG plotting function with multiple FE maps."""
    output_file = tmp_path / "test_ecdf_ddgs_multiple.png"
    # Create a second FE map for testing
    # add gaussian noise to the first map to create a second map
    graph = nx.MultiDiGraph()
    for a, b, data in fe_map.to_networkx().edges(data=True):
        new_data = data.copy()
        if data["source"] != "reverse" and data["computational"]:
            # add noise to the result
            new_result = data["DG"] + np.random.normal(0, data["uncertainty"].m) * data["DG"].u
            new_data["DG"] = new_result
            graph.add_edge(a, b, **new_data)
            # and add the reverse edge
            rev_data = new_data.copy()
            rev_data["source"] = "reverse"
            rev_data["DG"] = -new_data["DG"]
            graph.add_edge(b, a, **rev_data)
        else:
            graph.add_edge(a, b, **data)
    fe_map_2 = FEMap.from_networkx(graph)
    fig = plotting.ecdf_plot_DDGs([fe_map, fe_map_2], labels=["FE Map 1", "FE Map 2"], filename=output_file.as_posix())
    assert fig is not None
    # check the file was created
    assert output_file.exists()


def test_plot_ecdf_no_datasets():
    with pytest.raises(ValueError, match="At least one dataset is required to plot an ECDF."):
        plotting.ecdf_plot({})


def test_plot_ecdf_colors(fe_map, tmp_path):
    """Test ECDF plotting function with custom colors."""
    output_file = tmp_path / "test_ecdf_colors.png"
    fig = plotting.ecdf_plot_DDGs([fe_map], labels=["Test FE Map"], colors=["#FF5733"], filename=output_file.as_posix())
    assert fig is not None
    # check the file was created
    assert output_file.exists()
    # check that the line color matches the specified color
    line = fig.get_axes()[0].lines[0]
    assert line.get_color() == "#FF5733"
