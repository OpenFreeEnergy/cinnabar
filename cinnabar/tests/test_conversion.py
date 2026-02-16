import pytest
from openff.units import unit

from cinnabar import conversion


@pytest.mark.parametrize(
    "value, uncertainty, temp, output_type, expected_value, expected_uncertainty",
    [
        # self transforms but changing the temp
        (-10.0, 0.1, 300.0, "dg", -10.0, 0.1),
        (-10.0, 0.1, 298.15, "dg", -10.0, 0.1),
        (-10.0, None, 298.15, "dg", -10.0, None),
        # to ki with changing temp
        (-10.0, 0.1, 298.15, "ki", 46.77, 7.89),
        (-10.0, None, 300, "ki", 51.9, None),
        # to ic50 with changing temp
        (-10.0, 0.1, 298.15, "ic50", 46.77, 7.89),
        (-10.0, None, 300, "ic50", 51.9, None),
        # to pic50 with changing temp
        (-10.0, 0.1, 298.15, "pic50", 7.33, 0.07),
        (-10.0, None, 300, "pic50", 7.28, None),
    ],
)
def test_convert_value_from_dg(value, uncertainty, temp, output_type, expected_value, expected_uncertainty):
    kclm = unit("kcal / mole")
    value *= kclm
    if uncertainty is not None:
        uncertainty *= kclm

    converted_value, converted_error = conversion.convert_observable(
        value=value,
        original_type="dg",
        final_type=output_type,
        uncertainty=uncertainty,
        temperature=temp * unit.kelvin,
    )
    assert pytest.approx(converted_value.m) == expected_value
    if expected_uncertainty is None:
        assert converted_error is None
    else:
        assert pytest.approx(converted_error.m) == expected_uncertainty


@pytest.mark.parametrize(
    "value, uncertainty, temp, output_type, expected_value, expected_uncertainty",
    [
        # self transforms but changing the temp
        (42.0, 2.5, 300.0, "ki", 42.0, 2.5),
        (42.0, 2.5, 298.15, "ki", 42.0, 2.5),
        (42.0, None, 298.15, "ki", 42.0, None),
        # to dg with changing temp this is the reverse transformation of the first test
        (46.77, 7.89, 298.15, "dg", -10.0, 0.1),
        (51.9, None, 300, "dg", -10.0, None),
        # to ic50 with changing temp
        (46.77, 7.89, 298.15, "ic50", 46.77, 7.89),
        (46.77, None, 300, "ic50", 46.77, None),
        # to pic50 with changing temp
        (46.77, 7.89, 298.15, "pic50", 7.33, 0.07),
        (51.9, None, 300, "pic50", 7.28, None),
    ],
)
def test_convert_value_from_ki(value, uncertainty, temp, output_type, expected_value, expected_uncertainty):
    nm = unit("nanomolar")
    value *= nm
    if uncertainty is not None:
        uncertainty *= nm

    converted_value, converted_error = conversion.convert_observable(
        value=value,
        original_type="ki",
        final_type=output_type,
        uncertainty=uncertainty,
        temperature=temp * unit.kelvin,
    )
    assert pytest.approx(converted_value.m) == expected_value
    if expected_uncertainty is None:
        assert converted_error is None
    else:
        assert pytest.approx(converted_error.m) == expected_uncertainty


@pytest.mark.parametrize(
    "value, uncertainty, temp, output_type, expected_value, expected_uncertainty",
    [
        # self transforms but changing the temp
        (42.0, 2.5, 300.0, "ic50", 42.0, 2.5),
        (42.0, 2.5, 298.15, "ic50", 42.0, 2.5),
        (42.0, None, 298.15, "ic50", 42.0, None),
        # to dg with changing temp this is the reverse transformation of the first test
        (46.77, 7.89, 298.15, "dg", -10.0, 0.1),
        (51.9, None, 300, "dg", -10.0, None),
        # to ki with changing temp
        (46.77, 7.89, 298.15, "ki", 46.77, 7.89),
        (46.77, None, 300, "ki", 46.77, None),
        # to pic50 with changing temp
        (46.77, 7.89, 298.15, "pic50", 7.33, 0.07),
        (51.9, None, 300, "pic50", 7.28, None),
    ],
)
def test_convert_value_from_ic50(value, uncertainty, temp, output_type, expected_value, expected_uncertainty):
    nm = unit("nanomolar")
    value *= nm
    if uncertainty is not None:
        uncertainty *= nm

    converted_value, converted_error = conversion.convert_observable(
        value=value,
        original_type="ic50",
        final_type=output_type,
        uncertainty=uncertainty,
        temperature=temp * unit.kelvin,
    )
    assert pytest.approx(converted_value.m) == expected_value
    if expected_uncertainty is None:
        assert converted_error is None
    else:
        assert pytest.approx(converted_error.m) == expected_uncertainty


@pytest.mark.parametrize(
    "value, uncertainty, temp, output_type, expected_value, expected_uncertainty",
    [
        # self transforms but changing the temp
        (7.33, 0.07, 300.0, "pic50", 7.33, 0.07),
        (7.33, 0.07, 298.15, "pic50", 7.33, 0.07),
        (7.33, None, 298.15, "pic50", 7.33, None),
        # to dg with changing temp this is the reverse transformation of the first test
        (7.33, 0.07, 298.15, "dg", -10.0, 0.1),
        (7.28, None, 300, "dg", -10.0, None),
        # to ki with changing temp
        (7.33, 0.07, 298.15, "ki", 46.77, 7.54),
        (7.33, None, 300, "ki", 46.77, None),
        # to ic50 with changing temp
        (7.33, 0.07, 298.15, "ic50", 46.77, 7.54),
        (7.33, None, 300, "ic50", 46.77, None),
    ],
)
def test_convert_value_from_pic50(value, uncertainty, temp, output_type, expected_value, expected_uncertainty):
    converted_value, converted_error = conversion.convert_observable(
        value=value,
        original_type="pic50",
        final_type=output_type,
        uncertainty=uncertainty,
        temperature=temp * unit.kelvin,
    )
    assert pytest.approx(converted_value.m, abs=0.1) == expected_value
    if expected_uncertainty is None:
        assert converted_error is None
    else:
        assert pytest.approx(converted_error.m) == expected_uncertainty


def test_convert_observable_invalid_types():
    with pytest.raises(ValueError, match="Unknown original_type: invalid"):
        conversion.convert_observable(
            value=1.0,
            original_type="invalid",
            final_type="dg",
            uncertainty=None,
            temperature=298.15 * unit.kelvin,
        )

    with pytest.raises(ValueError, match="Unknown final_type: invalid"):
        conversion.convert_observable(
            value=1.0,
            original_type="dg",
            final_type="invalid",
            uncertainty=None,
            temperature=298.15 * unit.kelvin,
        )
