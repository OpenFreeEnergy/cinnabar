import pytest

from cinnabar._due import due

pytest.importorskip("duecredit")


def test_duecredit_mle():
    """Make sure duecredit is captured when the stats module is used"""
    mle_key = ("cinnabar.stats.mle", "10.1021/acs.jcim.9b00528")
    assert due.citations[mle_key].cites_module
