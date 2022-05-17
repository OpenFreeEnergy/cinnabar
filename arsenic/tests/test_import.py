"""
Import tests
"""

import arsenic
import pytest
import sys


def test_arsenic_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "arsenic" in sys.modules
