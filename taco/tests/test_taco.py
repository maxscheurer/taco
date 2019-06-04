"""
Unit and regression test for the taco package.
"""

# Import package, test suite, and other packages as needed
import taco
import pytest
import sys

def test_taco_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "taco" in sys.modules
