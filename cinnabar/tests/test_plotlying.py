import plotly.graph_objects as go
import pytest

from cinnabar import plotlying, plotting


def test_plot_ddgs_plotly(tmp_path, fe_map):
    output_file = tmp_path / "ddg_plot.html"
    _ = plotting.plot_DDGs(fe_map.to_legacy_graph(), filename=output_file.as_posix(), plotly=True)
    assert output_file.exists()


def test_plot_dgs_plotly(tmp_path, fe_map):
    output_file = tmp_path / "dg_plot.html"
    _ = plotting.plot_DGs(fe_map.to_legacy_graph(), filename=output_file.as_posix(), plotly=True)
    assert output_file.exists()


def test_plot_all_ddgs_plotly(tmp_path, fe_map):
    output_file = tmp_path / "all_ddg_plot.html"
    _ = plotting.plot_all_DDGs(fe_map.to_legacy_graph(), filename=output_file.as_posix(), plotly=True)
    assert output_file.exists()


def test_master_plot_bad_statistic_type(example_data):
    """Test that bad statistic in master plot raises an error."""

    x_data, y_data, xerr, yerr = example_data
    with pytest.raises(ValueError, match="Unknown statistic type bad_stat"):
        _ = plotlying._master_plot(
            x_data,
            y_data,
            statistic_type="bad_stat",
        )


def test_master_plot_show(example_data, monkeypatch):
    """Test that master plot shows when filename is None."""

    x_data, y_data, xerr, yerr = example_data
    called = {}

    def mock_show(self):
        called["show"] = True

    monkeypatch.setattr(go.Figure, "show", mock_show)

    _ = plotlying._master_plot(
        x_data,
        y_data,
        filename=None,
    )
    assert "show" in called
