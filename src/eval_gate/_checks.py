import math
import numpy as np


def check_finite(value: float, name: str) -> None:
    """Check that a value is finite (neither infinite nor ``NaN``).

    The sign is not constrained. Use ``check_finite_non_negative`` when the value
    must also be greater than or equal to zero.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the parameter, used in the error message.

    Raises
    ------
    ValueError
        If ``value`` is infinite or ``NaN``.

    See Also
    --------
    check_all_finite : Same check for every element of an array.
    """
    if not math.isfinite(value):
        raise ValueError(f"'{name}' must be finite, got {value}.")


def check_all_finite(values: np.ndarray, name: str) -> None:
    """Check that every element of an array is finite (no ``inf`` and no ``NaN``).

    Meant to run before any statistic is computed from the array.

    Parameters
    ----------
    values : np.ndarray
        Array (or array-like) to check, of any shape. An empty array passes.
    name : str
        Name of the parameter, used in the error message.

    Raises
    ------
    ValueError
        If any element is infinite or ``NaN``.

    See Also
    --------
    check_finite : Same check for a single scalar parameter.
    """
    array = np.asarray(values)
    nb_non_finite = int(array.size - np.count_nonzero(np.isfinite(array)))
    if nb_non_finite:
        raise ValueError(
            f"'{name}' must contain only finite values, got {nb_non_finite} non-finite value(s) out of {array.size}."
        )


def check_non_negative(value: float, name: str) -> None:
    """Check that a value is greater than or equal to zero.

    Infinity is allowed; combine with ``check_finite`` (or use
    ``check_finite_non_negative``) to exclude it.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name of the parameter, used in the error message.

    Raises
    ------
    ValueError
        If ``value`` is negative or ``NaN``.
    """
    if not value >= 0.0:
        raise ValueError(f"'{name}' must be non-negative, got {value}.")


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
        If ``value`` is infinite, ``NaN`` or negative.
    """
    check_finite(value, name)
    check_non_negative(value, name)


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


def check_min_units(n: int, what: str, minimum: int) -> None:
    """Check that there are enough units to compute a statistic.

    Parameters
    ----------
    n : int
        Number of units available.
    what : str
        Statistic being computed, used in the error message (for example ``"the standard error"``).
    minimum : int
        Minimum number of units required.

    Raises
    ------
    ValueError
        If ``n < minimum``.
    """
    if n < minimum:
        raise ValueError(f"At least {minimum} units are required to compute {what}, got {n}.")
