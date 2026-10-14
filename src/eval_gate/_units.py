import math
import numpy as np
from typing import NamedTuple

from ._checks import check_min_units


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
    """
    u = unit_means(outcomes)
    n = u.shape[0]
    check_min_units(n, what="the standard error", minimum=2)
    return MeanSE(float(u.mean()), float(u.std(ddof=1) / math.sqrt(n)), n)


def paired_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-unit paired differences (candidate - reference).

    Replicates are averaged within each unit before taking the difference.

    Parameters
    ----------
    a : np.ndarray
        Reference measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    b : np.ndarray
        Candidate measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.

    Returns
    -------
    np.ndarray
        Per-unit differences ``b - a``, of shape ``(nb_units,)``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have a different number of units.
    """
    a_units = unit_means(a)
    b_units = unit_means(b)
    if a_units.shape != b_units.shape:
        raise ValueError(f"Number of units differs: {a_units.shape[0]} vs {b_units.shape[0]}.")
    return b_units - a_units


def paired_std(a: np.ndarray, b: np.ndarray) -> float:
    """Standard deviation of the per-unit paired differences ``b - a``.

    Typically estimated from pilot data and passed as ``std`` to the power and
    sample-size functions.

    Parameters
    ----------
    a : np.ndarray
        Reference measurements, shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    b : np.ndarray
        Candidate measurements, same number of units as ``a``. The number of
        replicates may differ.

    Returns
    -------
    float
        Sample standard deviation (``ddof=1``) of the per-unit differences, after
        averaging replicates within each unit.

    Notes
    -----
    Replicates are averaged within each unit first, so the result depends on the
    number of replicates in the pilot. Reuse it only for a study that plans a
    similar number of replicates per unit.
    """
    differences = paired_differences(a, b)
    check_min_units(differences.shape[0], what="the standard deviation", minimum=2)
    return float(differences.std(ddof=1))
