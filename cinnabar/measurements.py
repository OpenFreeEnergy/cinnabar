from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity


class RelativeMeasurement(DefaultModel):
    """The free energy difference of moving from A to B"""
    labelA: str
    labelB: str
    DDG: FloatQuantity['kilocalorie_per_mole']
    uncertainty: FloatQuantity['kilocalorie_per_mole']
    computational: bool
    source: str = ""


class AbsoluteMeasurement(DefaultModel):
    label: str
    DG: FloatQuantity['kilocalorie_per_mole']
    uncertainty: FloatQuantity['kilocalorie_per_mole']
    computational: bool
    source: str = ""
