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


def paired_differences(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Compute per-unit paired differences ``candidate - reference``.

    Replicates are averaged within each unit before taking the difference.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    candidate : np.ndarray
        Candidate measurements, same number of units as ``reference``. The number of replicates may differ.

    Returns
    -------
    np.ndarray
        Per-unit differences ``candidate - reference``, of shape ``(nb_units,)``.

    Raises
    ------
    ValueError
        If ``reference`` and ``candidate`` have a different number of units.
    """
    reference_units = unit_means(reference)
    candidate_units = unit_means(candidate)
    if reference_units.shape != candidate_units.shape:
        raise ValueError(f"Number of units differs: {reference_units.shape[0]} vs {candidate_units.shape[0]}.")
    return candidate_units - reference_units


def paired_std(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Standard deviation of the per-unit paired differences ``candidate - reference``.

    Typically estimated from pilot data and passed as ``std`` to the power and
    sample-size functions.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements, shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    candidate : np.ndarray
        Candidate measurements, same number of units as ``reference``. The number of replicates may differ.

    Returns
    -------
    float
        Sample standard deviation (``ddof=1``) of the per-unit differences, after averaging replicates within each unit.

    Notes
    -----
    Replicates are averaged within each unit first, so the result depends on the number of replicates in the pilot.
    Reuse it only for a study that plans a similar number of replicates per unit.
    """
    differences = paired_differences(reference, candidate)
    check_min_units(differences.shape[0], what="the standard deviation", minimum=2)
    return float(differences.std(ddof=1))
