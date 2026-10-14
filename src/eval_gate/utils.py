import math
import numpy as np
from typing import NamedTuple


class MeanSE(NamedTuple):
    """Mean and standard error of the mean, with the number of units behind them.

    Attributes
    ----------
    mean : float
        Mean of the per-unit values.
    se : float
        Standard error of the mean, ``std(ddof=1) / sqrt(nb_units)``.
    nb_units : int
        Number of units.
    """
    mean: float
    se: float
    nb_units: int


def unit_means(outcomes: np.ndarray) -> np.ndarray:
    """Return one value per unit, averaging replicates within each unit.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcomes of shape ``(n,)`` or ``(n, k)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``outcomes`` is not 1D or 2D.
    """
    x = np.asarray(outcomes, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise ValueError(f"Expected a 1D or 2D array, got shape {x.shape}.")


def mean_and_se(outcomes: np.ndarray) -> MeanSE:
    """Compute the mean and unit-clustered standard error of the mean.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcomes of shape ``(n,)`` or ``(n, k)``; replicates are averaged within each unit.

    Returns
    -------
    MeanSE
        Mean, standard error and number of units.

    Raises
    ------
    ValueError
        If ``outcomes`` has fewer than 2 units.
    """
    u = unit_means(outcomes)
    n = u.shape[0]
    if n < 2:
        raise ValueError(f"At least 2 units are required to compute the standard error, got {n}.")
    return MeanSE(float(u.mean()), float(u.std(ddof=1) / math.sqrt(n)), n)


def _format_pairs_of_arrays(a_in: np.ndarray, b_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and format a pair of arrays for paired statistical analysis.

    Coerces inputs to float arrays, promotes 1D arrays to 2D column vectors, and checks
    that the shapes match.

    Parameters
    ----------
    a_in : np.ndarray
        Reference (baseline) measurements.
    b_in : np.ndarray
        Candidate measurements.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The formatted ``(a, b)``, both of shape ``(n, k)``.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    a_out = np.asarray(a_in, dtype=float)
    b_out = np.asarray(b_in, dtype=float)
    if a_out.ndim == 1:
        a_out = a_out[:, np.newaxis]
    if b_out.ndim == 1:
        b_out = b_out[:, np.newaxis]
    if a_out.shape != b_out.shape:
        raise ValueError(f"Shape mismatch: {a_out.shape} vs {b_out.shape}.")
    return a_out, b_out


def _paired_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-unit paired differences (candidate - reference).

    Replicates are averaged within each unit before taking the difference.

    Parameters
    ----------
    a : np.ndarray
        Reference (baseline) measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    b : np.ndarray
        Candidate measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.

    Returns
    -------
    np.ndarray
        Per-unit differences ``b - a``, of shape ``(nb_units,)``.
    """
    a_formatted, b_formatted = _format_pairs_of_arrays(a, b)
    return b_formatted.mean(axis=1) - a_formatted.mean(axis=1)


def paired_std(a, b):
    return float(np.std(_paired_differences(a, b), ddof=1))
