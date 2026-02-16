"""
Conversion functions taken from
<https://raw.githubusercontent.com/openforcefield/protein-ligand-benchmark/refs/tags/0.2.1/plbenchmark/utils.py>
"""
from openff.units import unit
from typing import Literal, get_args
import numpy as np

# Single source of truth for the observable types we support
OBSERVABLE_TYPES = Literal["dg", "ki", "ic50", "pic50"]


def convert_observable(
        value: unit.Quantity,
        original_type: OBSERVABLE_TYPES,
        final_type: OBSERVABLE_TYPES,
        uncertainty: None | unit.Quantity = None,
        temperature: unit.Quantity = 300.0 * unit.kelvin
) -> tuple[unit.Quantity, unit.Quantity | None]:
    """
    Converts an affinity value into another derived quantity,
    including automatic error propagation if error is provided.

    Parameters
    ----------
    value : unit.Quantity
        Numerical value of the original observable with units.
    original_type : str
        Code for the original observable. Can be `dg`, `ki`, `ic50`, `pic50`.
    final_type : str
        Code for the desired derived quantity. Can be `dg`, `ki`, `ic50`, `pic50`.
    uncertainty : unit.Quantity, optional
        The uncertainty/error in the original observable with units, should always be positive.
    temperature : unit.Quantity, optional
        Temperature in kelvin for conversions involving dG. Default is 300 K.


    Notes
    -----
    - The function uses the molar gas constant (R) and the provided temperature to perform conversions involving dG.
    - If the original value is below a threshold (e.g., 1e-15 M for Ki/IC50), the function will return a default value to avoid numerical issues.
    - The function rounds the converted value and uncertainty to 2 decimal places if the type changes to reflect the typical precision of such measurements.
    - The following default units are used for the output based on the final_type:
        - `dg`: kilocalories per mole
        - `ki`: nanomolar
        - `ic50`: nanomolar
        - `pic50`: unitless (logarithmic scale)


    Returns
    -------
    converted_value : unit.Quantity
        The converted value in the desired units.
    converted_error : unit.Quantity or None
        The propagated error in the converted value, or None if no error was provided.

    Examples
    --------
    >>> from openff.units import unit
    >>> from cinnabar.conversion import convert_observable
    >>> value = -10.0 * unit("kilocalories / mole")
    >>> uncertainty = 0.1 * unit("kilocalories / mole")
    >>> convert_observable(value, "dg", "pic50", uncertainty)
    (7.33, 0.07)
    >>> convert_observable(value, "dg", "ki", uncertainty)
    (46.77 nanomolar, 7.89 nanomolar)

    Raises
    ------
    ValueError
        If the original_type or final_type is not recognized.
    """
    # calculate kT for the given temperature, this will be used in the conversions involving dG
    k_bt = unit.molar_gas_constant * temperature

    # Validate input types
    valid_types = get_args(OBSERVABLE_TYPES)
    if original_type not in valid_types:
        raise ValueError(
            f"Unknown original_type: {original_type}. "
            f"Must be one of: {', '.join(valid_types)}"
        )
    if final_type not in valid_types:
        raise ValueError(
            f"Unknown final_type: {final_type}. "
            f"Must be one of: {', '.join(valid_types)}"
        )

    # store the conversion functions by (original_type, final_type) in a dictionary for easy lookup
    converters = {
        # dG conversions
        ("dg", "dg"): lambda v, u: (v, u),
        ("dg", "ki"): lambda v, e: _convert_dg_to_ki(v, e, k_bt),
        ("dg", "ic50"): lambda v, e: _convert_dg_to_ic50(v, e, k_bt),
        ("dg", "pic50"): lambda v, e: _convert_dg_to_pic50(v, e, k_bt),

        # Ki conversions
        ("ki", "dg"): lambda v, e: _convert_ki_to_dg(v, e, k_bt),
        ("ki", "ki"): lambda v, e: (v, e),
        ("ki", "ic50"): lambda v, e: (v, e),
        ("ki", "pic50"): lambda v, e: _convert_ki_to_pic50(v, e),

        # IC50 conversions
        ("ic50", "dg"): lambda v, e: _convert_ic50_to_dg(v, e, k_bt),
        ("ic50" ,"ki"): lambda v, e: (v, e),
        ("ic50", "ic50"): lambda v, e: (v, e),
        ("ic50", "pic50"): lambda v, e: _convert_ic50_to_pic50(v, e),

        # pIC50 conversions
        ("pic50", "dg"): lambda v, e: _convert_pic50_to_dg(v, e, k_bt),
        ("pic50", "ki"): lambda v, e: _convert_pic50_to_ki(v, e),
        ("pic50", "ic50"): lambda v, e: _convert_pic50_to_ic50(v, e),
        ("pic50", "pic50"): lambda v, e: (v, e),
    }

    # get the converter function based on the original and final types
    converter = converters[(original_type, final_type)]

    # the uncertainty should always be positive, so we take the absolute value if it is provided
    if uncertainty is not None:
        uncertainty = abs(uncertainty)

    # if the input is in pic50, add dimensionless units for consistent conversions
    if original_type == "pic50":
        value = value * unit("")
        if uncertainty is not None:
            uncertainty = uncertainty * unit("")

    # do the conversion and error propagation
    converted_value, converted_uncertainty = converter(value, uncertainty)

    # get the units for the output type and convert the value and uncertainty to those units
    default_units = {
        "dg": unit("kilocalories / mole"),
        "ki": unit("nanomolar"),
        "ic50": unit("nanomolar"),
        "pic50": unit("")
    }
    out_unit = default_units[final_type]
    converted_value = converted_value.to(out_unit)
    if converted_uncertainty is not None:
        converted_uncertainty = converted_uncertainty.to(out_unit)

    # conversions add uncertainty, so we round to 2 decimal places if the type changes
    if original_type != final_type:
        converted_value = converted_value.round(2)
        if converted_uncertainty is not None:
             converted_uncertainty = converted_uncertainty.round(2)

    return converted_value, converted_uncertainty


# dG conversion functions
def _convert_dg_to_ki(value, error, k_bt):
    """Convert dG to Ki with error propagation.

    Note
    ----
    - we do not take the negative of the value here because the conversion formula already accounts for it (Ki = exp(dG/RT))
    this is different from the reference implementation in the PLB.
    """
    result = np.exp(value / k_bt) * unit.molar
    error_result = None
    if error is not None:
        # Error propagation: e_ki = 1/RT * exp(dG/RT) * e_dG
        error_result = 1.0 / k_bt * np.exp(value / k_bt) * error * unit.molar
    return result, error_result


def _convert_dg_to_ic50(value, error, k_bt):
    """Convert dG to IC50 with error propagation."""
    result = np.exp(value / k_bt) * unit.molar
    error_result = None
    if error is not None:
        # Error propagation: same as Ki
        error_result = 1.0 / k_bt * np.exp(value / k_bt) * error * unit.molar
    return result, error_result


def _convert_dg_to_pic50(value, error, k_bt):
    """Convert dG to pIC50 with error propagation."""
    result = -value / (k_bt * np.log(10))
    error_result = None
    if error is not None:
        # Error propagation: e_pic50 = 1/(RT*ln(10)) * e_dG
        error_result = 1.0 / (k_bt * np.log(10)) * error
    return result, error_result


# Ki conversion functions
def _convert_ki_to_dg(value, error, k_bt):
    """Convert Ki to dG with error propagation."""
    # set the default if we are below the threshold limit
    result = 0.0
    error_result = None
    if value > 1e-15 * unit.molar:
        result = k_bt * np.log(value / unit.molar)

    if error is not None:
        # Error propagation: e_dG = RT / Ki * e_Ki
        error_result = k_bt / value * error
    return result, error_result


def _convert_ki_to_pic50(value, error):
    """Convert Ki to pIC50 with error propagation."""
    # set the default result if we are below the threshold limit
    result = -1e15
    error_result = None
    if value > 1e-15 * unit("molar"):
        result = -np.log(value / unit.molar) / np.log(10)
    # Error propagation: e_pic50 = 1/(Ki*ln(10)) * e_Ki
    if (value * np.log(10)) < 1e-15 * unit("molar") and error is not None:
        error_result =  1e15
    elif error is not None:
        error_result = 1 / (value * np.log(10)) * error
    return result, error_result


# IC50 conversion functions
def _convert_ic50_to_dg(value, error, k_bt):
    """Convert IC50 to dG with error propagation."""
    # set the default if we are below the threshold limit
    result = 0.0 * unit("kilocalories / mole")
    error_result = None
    if value > 1e-15 * unit("molar"):
        result = k_bt * np.log(value.to("molar") / unit.molar)
    # Error propagation: e_dG = RT / IC50 * e_IC50
    if error is not None:
        error_result = k_bt / value * error
    return result, error_result


def _convert_ic50_to_pic50(value, error):
    """Convert IC50 to pIC50 with error propagation."""
    # set the default result if we are below the threshold limit
    result = -1e15
    error_result = None
    if value > 1e-15 * unit("molar"):
        result = -np.log(value / unit.molar) / np.log(10)
    # Error propagation: e_pic50 = 1/(IC50*ln(10)) * e_IC50
    if (value * np.log(10)) < 1e-15 * unit("molar") and error is not None:
        error_result = 1e15
    elif error is not None:
        error_result = 1 / (value * np.log(10)) * error
    return result, error_result


# pIC50 conversion functions
def _convert_pic50_to_dg(value, error, k_bt):
    """Convert pIC50 to dG with error propagation."""
    error_result = None
    result = -1 * k_bt * value * np.log(10)
    if error is not None:
        # Error propagation: e_dG = RT * ln(10) * e_pIC50
        error_result = k_bt * np.log(10) * error
    return result, error_result


def _convert_pic50_to_ki(value, error):
    """Convert pIC50 to Ki with error propagation."""
    error_result = None
    result = 10 ** (-value) * unit("molar")
    if error is not None:
        # Error propagation: e_Ki = ln(10) * 10^(-pIC50) * e_pIC50
        error_result = np.log(10) * 10 ** (-value) * error * unit("molar")
    return result, error_result


def _convert_pic50_to_ic50(value, error):
    """Convert pIC50 to IC50 with error propagation."""
    error_result = None
    result = 10 ** (-value) * unit("molar")
    if error is not None:
        # Error propagation: e_IC50 = ln(10) * 10^(-pIC50) * e_pIC50
        error_result = np.log(10) * 10 ** (-value) * error * unit("molar")
    return result, error_result
