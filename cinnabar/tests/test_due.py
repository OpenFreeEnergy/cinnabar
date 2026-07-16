import pytest

from cinnabar._due import due

pytest.importorskip("duecredit")


def test_duecredit_mle():
    """Make sure duecredit is captured when the stats module is used"""
    mle_key_xu = ("cinnabar.estimators.MLEEstimator.mle", "10.1021/acs.jcim.9b00528")
    mle_key_kenney = ("cinnabar.estimators.MLEEstimator.mle", "Kenney2023Biophysical")
    assert due.citations[mle_key_xu].cites_module
    assert due.citations[mle_key_kenney].cites_module
