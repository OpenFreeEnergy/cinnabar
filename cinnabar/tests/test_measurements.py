import pytest
from openff.units import unit

import cinnabar


def test_ground():
    g1 = cinnabar.ReferenceState()
    g2 = cinnabar.ReferenceState()
    g3 = cinnabar.ReferenceState(label="MLE")
    g4 = cinnabar.ReferenceState(label="MLE")

    assert g1 == g2
    assert g1 != g3
    assert g3 == g4


def test_measurement_hash():
    m1 = cinnabar.Measurement(
        labelA="foo",
        labelB="bar",
        DG=0.1 * unit.kilocalorie_per_mole,
        uncertainty=0.01 * unit.kilocalorie_per_mole,
        computational=True,
    )
    m1a = cinnabar.Measurement(
        labelA="foo",
        labelB="bar",
        DG=0.1 * unit.kilocalorie_per_mole,
        uncertainty=0.01 * unit.kilocalorie_per_mole,
        computational=True,
    )
    m2 = cinnabar.Measurement(
        labelA="foo",
        labelB="bar",
        DG=0.11 * unit.kilocalorie_per_mole,
        uncertainty=0.01 * unit.kilocalorie_per_mole,
        computational=False,
    )

    thing = set([m1, m1a, m2])

    assert len(thing) == 2
    assert m1 in thing
    assert m2 in thing


@pytest.mark.parametrize(
    "Ki,uncertainty,dG,dG_uncertainty,label,temp",
    [
        [
            100 * unit.nanomolar,
            10 * unit.nanomolar,
            -9.55 * unit.kilocalorie_per_mole,
            0.059 * unit.kilocalorie_per_mole,
            "lig",
            298.15 * unit.kelvin,
        ],
        [
            0.1 * unit.micromolar,
            0.01 * unit.micromolar,
            -9.55 * unit.kilocalorie_per_mole,
            0.059 * unit.kilocalorie_per_mole,
            "lig",
            298.15 * unit.kelvin,
        ],
        [
            100 * unit.nanomolar,
            10 * unit.nanomolar,
            -10.57 * unit.kilocalorie_per_mole,
            0.066 * unit.kilocalorie_per_mole,
            "lig",
            330 * unit.kelvin,
        ],
    ],
)
def test_Ki_to_DG(Ki, uncertainty, dG, dG_uncertainty, label, temp):
    Ki_to_DG = cinnabar.Measurement.from_experiment(label, Ki, uncertainty, source="", temperature=temp)

    assert Ki_to_DG.DG.units == unit.kilocalorie_per_mole
    assert Ki_to_DG.uncertainty.units == unit.kilocalorie_per_mole
    assert pytest.approx(dG, 0.001) == Ki_to_DG.DG
    assert pytest.approx(dG_uncertainty, 0.01) == Ki_to_DG.uncertainty


def test_negative_Ki():
    with pytest.raises(
        ValueError,
        match=r"Ki value cannot be zero or negative. Check if "
        r"dG value was provided instead of Ki.",
    ):
        cinnabar.Measurement.from_experiment("Test Label", -100 * unit.nanomolar)


def test_negative_uncertainty():
    with pytest.raises(
        ValueError,
        match=r"Uncertainty cannot be negative. "
        r"Check input.",
    ):
        cinnabar.Measurement.from_experiment("Test Label", 100 * unit.nanomolar, -10 * unit.nanomolar)


def test_measurement_temp():
    m = cinnabar.Measurement(
        labelA="foo",
        labelB="bar",
        DG=2.0 * unit.kilocalorie_per_mole,
        uncertainty=0.2 * unit.kilocalorie_per_mole,
        computational=True,
    )

    assert m.temperature == 298.15 * unit.kelvin


@pytest.mark.parametrize(
    "label, expected",
    [
        ("lig_a", False),
        ("", True),
    ],
)
def test_is_true_groud(label, expected):
    m = cinnabar.ReferenceState(label=label)
    assert m.is_true_ground() is expected


@pytest.mark.parametrize(
    "label, expected",
    [
        ("MLE", "<ReferenceState (MLE)>"),
        ("", "<ReferenceState Zero>"),
    ],
)
def test_ref_state_repr(label, expected):
    rs = cinnabar.ReferenceState(label=label)
    assert repr(rs) == expected
