from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from typing import Hashable, Optional
import uuid


class GroundState:
    """A label indicating a reference point to which absolute measurements are relative

    E.g. an absolute measurement for "LigandA" is defined as::

    >>> m = Measurement(labelA=GroundState(label='foo'), labelB='LigandA', ...)

    A ``GroundState`` has a label, which is used to differentiate it to other absolute measurements that
    might be relative to a different reference point.  If not label is given on creating the GroundState
    then one is randomly generated.  This means that two GroundState objects created independently
    aren't considered equal::

    >>> g1 = GroundState()
    >>> g2 = GroundState()
    >>> assert g1 != g2


    The ``TrueGround`` class is used to denote the reference point of "true" Ground.
    """
    label: str

    def __init__(self, label: Optional[str] = None):
        """
        Parameters
        ----------
        label: str, optional
          label for this reference point.  If no label is given, a unique label will randomly be
          assigned
        """
        if label is None:
            label = str(uuid.uuid4())
        self.label = label

    def is_true_ground(self) -> bool:
        return not self.label

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.label == self.label

    def __hash__(self):
        return hash(self.label)


class TrueGround(GroundState):
    """A reference to a hypothetical True Ground state"""
    _instance = None
    label: str

    def __new__(cls, *args, **kwargs):
        # singleton pattern
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        super().__init__(label='')


class Measurement(DefaultModel):
    """The free energy difference of moving from A to B"""
    labelA: Hashable
    labelB: Hashable
    DG: FloatQuantity['kilocalorie_per_mole']
    uncertainty: FloatQuantity['kilocalorie_per_mole']
    computational: bool
    source: str = ""
