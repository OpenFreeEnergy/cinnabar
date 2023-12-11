import abc

from .. import FEMap


class BaseEstimator(abc.ABC):
    """base class for implementing custom estimators"""
    @abc.abstractmethod
    def estimate(self, prior: FEMap) -> FEMap:
        pass
