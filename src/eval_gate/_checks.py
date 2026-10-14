import math


def check_finite_non_negative(value: float, name: str) -> None:
    """Check that a value is finite and greater than or equal to zero.

    Intended for standard errors, standard deviations, variances and margins, where zero is a legitimate
    (if degenerate) value but negative, infinite or ``NaN`` values indicate an upstream error.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the parameter, used in the error message.

    Raises
    ------
    ValueError
        If ``value`` is negative, infinite or ``NaN``.
    """
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"'{name}' must be finite and non-negative, got {value}.")


def check_in_open_interval(value: float, name: str, low: float, high: float) -> None:
    """Check that a value lies strictly inside an open interval.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the parameter, used in the error message so the caller knows which argument is invalid.
    low : float
        Lower bound of the interval (exclusive).
    high : float
        Upper bound of the interval (exclusive).

    Raises
    ------
    ValueError
        If ``value`` is not in ``(low, high)``. ``NaN`` is rejected as well.
    """
    if not low < value < high:
        raise ValueError(f"'{name}' must be in ({low}, {high}), got {value}.")


def check_probability(value: float, name: str) -> None:
    """Check that a value is a probability in the open interval ``(0, 1)``.

    Intended for confidence levels, significance levels and statistical power, where
    the boundary values 0 and 1 are degenerate and therefore not allowed.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the parameter, used in the error message.
    """
    check_in_open_interval(value, name, 0.0, 1.0)
