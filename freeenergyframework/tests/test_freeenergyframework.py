"""
Unit and regression test for the freeenergyframework package.
"""

# Import package, test suite, and other packages as needed
import freeenergyframework
import pytest
import sys

def test_freeenergyframework_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "freeenergyframework" in sys.modules
