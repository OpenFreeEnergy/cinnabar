import cinnabar
import pytest
from openff.units import unit


def test_ground():
    g1 = cinnabar.ReferenceState()
    g2 = cinnabar.ReferenceState()
    g3 = cinnabar.ReferenceState(label='MLE')
    g4 = cinnabar.ReferenceState(label='MLE')

    assert g1 == g2
    assert g1 != g3
    assert g3 == g4

def test_Ki_to_DG():
    Ki = 100 * unit.nanomolar
    uncertainty = 10 * unit.nanomolar
    dG = -9.55 * unit.kilocalorie_per_mole
    dG_uncertainty = 0.059 * unit.kilocalorie_per_mole
    label = 'lig'
    Ki_to_DG = cinnabar.Measurement.from_experiment(Ki, label, uncertainty)

    assert pytest.approx(dG, 0.001) == Ki_to_DG.DG
    assert pytest.approx(dG_uncertainty, 0.01) == Ki_to_DG.uncertainty