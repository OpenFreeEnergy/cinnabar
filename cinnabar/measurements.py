from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from typing import Hashable


class GroundState:
    """A label indicating a reference point to which absolute measurements are relative

    E.g. an absolute measurement for "LigandA" is defined as::

    >>> m = Measurement(labelA=GroundState(), labelB='LigandA', ...)

    A ``GroundState`` can have a label, which is used to differentiate it to other absolute measurements that
    might be relative to a different reference point.
    """
    label: str

    def __init__(self, label: str = ""):
        """
        Parameters
        ----------
        label: str, optional
          label for this reference point.  Defaults to "", which is treated as "true zero"
        """
        self.label = label

    def is_true_ground(self) -> bool:
        return not self.label

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.label == self.label

    def __hash__(self):
        return hash(self.label)


class Measurement(DefaultModel):
    """The free energy difference of moving from A to B"""
    labelA: Hashable
    labelB: Hashable
    DG: FloatQuantity['kilocalorie_per_mole']
    uncertainty: FloatQuantity['kilocalorie_per_mole']
    computational: bool
    source: str = ""
