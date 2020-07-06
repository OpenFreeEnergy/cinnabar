"""
Unit and regression test for the beryllium package.
"""

# Import package, test suite, and other packages as needed
import beryllium
import pytest
import sys

def test_beryllium_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "beryllium" in sys.modules
