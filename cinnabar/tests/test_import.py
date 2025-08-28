"""
Import tests
"""

import sys

import pytest

import cinnabar


def test_cinnabar_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cinnabar" in sys.modules
