from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
from typing import Hashable
import math
import warnings


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
                        IC50: float,
                        label: str,
                        uncertainty: float = 0,
                        source: str = '',
                        temperature: float = 298,
                        ):
        """Create Measurement from experimental data

        Parameters
        ----------
        IC50: float
          experimental IC50 value in unit nM
        label: str, optional
          label for this data point.
        uncertainty: float, optional
          uncertainty of the experimental value in nM, default is zero if no uncertainty is provided
        source: str, optional
          source of experimental measurement
        temperature: float
          temperature in K at which the experimental measurement was carried out. By default: 298 K
        """
        R = 1.9872042586408 * 0.001 * unit.kilocalorie_per_mole / unit.kelvin # Gas constant in kcal/mol/K
        IC50_upper = IC50 * unit.nanomolar + uncertainty * unit.nanomolar
        IC50_lower = IC50 * unit.nanomolar - uncertainty * unit.nanomolar
        if IC50 > 0:
            # Check if IC50 potentially given in M instead of nM.
            if 0 < IC50 < 10**-6:
                wmsg = ("IC50 < 1 femtomolar. Check if the IC50 was given in unit nM or potentially M")
                warnings.warn(wmsg)
            # dG = RT ln IC50
            DG = R * temperature * unit.kelvin * math.log(IC50 * unit.nanomolar/ 10 ** 9 * unit.nanomolar)
        else:
            raise ValueError(
                "IC50 value cannot be zero or negative. Check if dG value was provided instead of IC50."
            )
        #Convert IC50 uncertainty into dG uncertainty
        if uncertainty >= 0:
            uncertainty_DG = 0.5 * R * temperature * unit.kelvin * math.log(IC50_upper / IC50_lower)
        else:
            raise ValueError(
                "Uncertainty cannot be negative. Check input."
            )
        return cls(labelA=ReferenceState(),
                   labelB=label,
                   DG=DG,
                   uncertainty=uncertainty_DG,
                   computational=False,
                   source=source,
                   )
