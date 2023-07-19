from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
from typing import Hashable


class ReferenceState:
    """A label indicating a reference point to which absolute measurements are relative

    E.g. an absolute measurement for "LigandA" is defined as::

    >>> m = Measurement(labelA=ReferenceState(), labelB='LigandA',
    ...                 DG=2.4 * unit.kilocalorie_per_mol,
    ...                 uncertainty=0.2 * unit.kilocalorie_per_mol,
    ...                 source='gromacs')

    A ``ReferenceState`` optionally has a label, which is used to differentiate
    it to other absolute measurements that might be relative to a different
    reference point.  E.g. MLE measurements are against an arbitrary reference
    state that must be linked to the reference point of experiments.
    """
    label: str

    def __init__(self, label: str = ""):
        """
        Parameters
        ----------
        label: str, optional
          label for this reference point.  If no label is given, an empty string
          is used, signifying the "true zero" reference point.
        """
        self.label = label

    def is_true_ground(self) -> bool:
        return not self.label

    def __repr__(self):
        if self.is_true_ground():
            return "<ReferenceState Zero>"
        else:
            return f"<ReferenceState ({self.label})>"

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

    @classmethod
    def from_experiment(cls,
                        pIC50: float,
                        label: str,
                        source: str = '',
                        ):
        """Create Measurement from experimental data

        Parameters
        ----------
        """
        return cls(labelA=ReferenceState(),
                   labelB=label,
                   DG=1.0 * unit.kilocalorie_per_mol,
                   uncertainty=0.0 * unit.kilocalorie_per_mol,
                   computational=False,
                   source=source,
                   )
