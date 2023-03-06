import pytest
import subprocess


def test_cli_smoke():
    # check that running the help command doesn't explode
    p = subprocess.call(['cinnabar', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert p == 0

