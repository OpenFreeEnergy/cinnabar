import pytest
from cinnabar.estimators import MLEEstimator, _build_graph_from_measurements


def test_weakly_connected():
    estimator = MLEEstimator()
    assert not estimator._check_weakly_connected([])


def test_no_measurements():
    with pytest.raises(ValueError, match="No measurements provided"):
        _build_graph_from_measurements([])
