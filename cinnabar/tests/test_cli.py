import pytest
import subprocess


def test_cli_smoke():
    # check that running the help command doesn't explode
    p = subprocess.run(['cinnabar', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize('mode', ['all', 'ddg', 'dg', 'all ddg', 'network'])
def test_cli_plot(mode, example_csv):
    p = subprocess.run(['cinnabar', example_csv, '--plot', mode],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert p.returncode == 0, p.stderr
