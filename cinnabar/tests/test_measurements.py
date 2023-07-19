import cinnabar
from openff.units import unit


def test_ground():
    g1 = cinnabar.ReferenceState()
    g2 = cinnabar.ReferenceState()
    g3 = cinnabar.ReferenceState(label='MLE')
    g4 = cinnabar.ReferenceState(label='MLE')

    assert g1 == g2
    assert g1 != g3
    assert g3 == g4


def test_measurement_temp():
    m = cinnabar.Measurement(
        labelA='foo',
        labelB='bar',
        DG=2.0 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
        computational=True,
    )

    assert m.temperature == 298.15 * unit.kelvin
