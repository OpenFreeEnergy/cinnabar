"""
Unit and regression test for the arsenic package.
"""

# Import package, test suite, and other packages as needed
import arsenic
import pytest
import sys

def test_arsenic_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "arsenic" in sys.modules
