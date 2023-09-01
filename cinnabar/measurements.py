from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
from typing import Hashable
import math


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
    temperature: FloatQuantity['kelvin'] = 298.15 * unit.kelvin
    computational: bool
    source: str = ""

    @classmethod
    def from_experiment(cls,
                        label: str | Hashable,
                        Ki: unit.Quantity,
                        uncertainty: unit.Quantity = 0 * unit.nanomolar,
                        *,
                        source: str = '',
                        temperature: unit.Quantity = 298.15 * unit.kelvin,
                        ):
        """Create Measurement from experimental data

        Parameters
        ----------
        label: str | Hashable
            label for this data point.
        Ki: unit.Quantity
            experimental Ki value
            ex.: 500 * unit.nanomolar OR 0.5 * unit.micromolar
        uncertainty: unit.Quantity
            uncertainty of the experimental value
            default is zero if no uncertainty is provided (0 * unit.nanomolar)
        source: str, optional
            source of experimental measurement
        temperature: unit.Quantity
            temperature in K at which the experimental measurement was carried out.
            By default: 298 K (298.15 * unit.kelvin)
        """
        if Ki > 0 * unit.molar:
            DG = (unit.molar_gas_constant * temperature.to(unit.kelvin)
                  * math.log( Ki / unit.molar)).to(unit.kilocalorie_per_mole)
        else:
            raise ValueError(
                "Ki value cannot be zero or negative. "
                "Check if dG value was provided instead of Ki."
            )
        # Convert Ki uncertainty into dG uncertainty: RT * uncertainty/Ki
        # https://physics.stackexchange.com/questions/95254/the-error-of-the-natural-logarithm
        if uncertainty >= 0 * unit.molar:
            uncertainty_DG = (unit.molar_gas_constant * temperature.to(unit.kelvin)
                              * uncertainty / Ki).to(unit.kilocalorie_per_mole)
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
                   temperature=temperature,
                   )
