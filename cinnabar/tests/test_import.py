"""
Import tests
"""

import cinnabar
import pytest
import sys


def test_cinnabar_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cinnabar" in sys.modules
