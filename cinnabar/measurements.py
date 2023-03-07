from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity


class RelativeMeasurement(DefaultModel):
    """The free energy difference of moving from A to B"""
    labelA: str
    labelB: str
    DDG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool
    source: str = ""


class AbsoluteMeasurement(DefaultModel):
    label: str
    DG: FloatQuantity['kilojoule_per_mole']
    uncertainty: FloatQuantity['kilojoule_per_mole']
    computational: bool
    source: str = ""
