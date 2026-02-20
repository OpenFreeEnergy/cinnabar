"""
Measurements
============

Contains the :class:`Measurement` class which is used to define a single free energy difference,
as well as the :class:`ReferenceState` class which denotes the end point for absolute measurements.

"""
from dataclasses import dataclass
import math
from typing import Hashable, Union

from openff.units import unit


class ReferenceState:
    """A label indicating a reference point to which absolute measurements are relative

    A ``ReferenceState`` optionally has a label, which is used to differentiate
    it to other absolute measurements that might be relative to a different
    reference point.  E.g. MLE estimations are against an arbitrary reference
    state that can be linked to the reference point of experiments.
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
        """If this ReferenceState is the zero point of all other measurements"""
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


@dataclass(frozen=True)
class Measurement:
    """The free energy difference of moving from A to B

    All quantities are accompanied by units, to prevent mix-ups associated with
    kcal and kJ.  This is done via the `openff.units` package::

      >>> m = Measurement(labelA='LigandA', labelB='LigandB',
      ...                 DG=2.4 * unit.kilocalorie_per_mol,
      ...                 uncertainty=0.2 * unit.kilocalorie_per_mol,
      ...                 computational=True,
      ...                 source='gromacs')

    Alternatively strings are automatically coerced into quantities, making this
    equivalent to above::

      >>> m = Measurement(labelA='LigandA', labelB='LigandB',
      ...                 DG='2.4 kcal/mol',
      ...                 uncertainty='0.2 kcal/mol',
      ...                 computational=True,
      ...                 source='gromacs')

    Where a measurement is "absolute" then a `ReferenceState` can be used as the
    label at one end of the measurement.  I.e. it is relative to a reference
    ground state. E.g. an absolute measurement for "LigandA" is defined as::

      >>> m = Measurement(labelA=ReferenceState(), labelB='LigandA',
      ...                 DG=-11.2 * unit.kilocalorie_per_mol,
      ...                 uncertainty=0.3 * unit.kilocalorie_per_mol,
      ...                 computational=False)
    """

    labelA: Hashable
    """Label of state A, e.g. a ligand name or any hashable Python object"""
    labelB: Hashable
    """Label of state B"""
    DG: unit.Quantity
    """The free energy difference of moving from A to B in kcal/mol"""
    uncertainty: unit.Quantity
    """The uncertainty of the DG measurement in kcal/mol"""
    computational: bool
    """If this measurement is computationally based (or experimental)"""
    source: str = ""
    """An arbitrary label to group measurements from a common source"""
    temperature: unit.Quantity = 298.15 * unit.kelvin
    """Temperature that the measurement was taken at in K. By default: 298 K (298.15 * unit.kelvin)"""

    def __init__(self, labelA: Hashable, labelB: Hashable, DG: unit.Quantity, uncertainty: unit.Quantity, computational: bool, source: str = "", temperature: unit.Quantity = 298.15 * unit.kelvin):
        """
        Initialize a Measurement object converting all quantities to the correct default units.
        """
        object.__setattr__(self, "labelA", labelA)
        object.__setattr__(self, "labelB", labelB)
        object.__setattr__(self, "DG", DG.to(unit.kilocalorie_per_mole))
        object.__setattr__(self, "uncertainty", uncertainty.to(unit.kilocalorie_per_mole))
        object.__setattr__(self, "computational", computational)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "temperature", temperature.to(unit.kelvin))


    @classmethod
    def from_experiment(
        cls,
        label: Union[str, Hashable],
        Ki: unit.Quantity,
        uncertainty: unit.Quantity = 0 * unit.nanomolar,
        *,
        source: str = "",
        temperature: unit.Quantity = 298.15 * unit.kelvin,
    ):
        """Shortcut to create a Measurement from experimental data

        Can perform conversion from Ki values to kcal/mol values.

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
        temperature: unit.Quantity, optional
            temperature in K at which the experimental measurement was carried out.
            By default: 298 K (298.15 * unit.kelvin)
        """
        if Ki > 0 * unit.molar:
            DG = (unit.molar_gas_constant * temperature.to(unit.kelvin) * math.log(Ki / unit.molar)).to(
                unit.kilocalorie_per_mole
            )
        else:
            raise ValueError("Ki value cannot be zero or negative. Check if dG value was provided instead of Ki.")
        # Convert Ki uncertainty into dG uncertainty: RT * uncertainty/Ki
        # https://physics.stackexchange.com/questions/95254/the-error-of-the-natural-logarithm
        if uncertainty >= 0 * unit.molar:
            uncertainty_DG = (unit.molar_gas_constant * temperature.to(unit.kelvin) * uncertainty / Ki).to(
                unit.kilocalorie_per_mole
            )
        else:
            raise ValueError("Uncertainty cannot be negative. Check input.")
        return cls(
            labelA=ReferenceState(),
            labelB=label,
            DG=DG,
            uncertainty=uncertainty_DG,
            computational=False,
            source=source,
            temperature=temperature,
        )
