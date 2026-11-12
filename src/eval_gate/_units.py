import math
import numpy as np
from typing import NamedTuple

from ._checks import check_min_units, check_all_finite


class MeanSE(NamedTuple):
    """Mean and standard error of the mean, with the number of units behind them.

    Attributes
    ----------
    mean : float
        Mean of the per-unit values.
    se : float
        Standard error of the mean, ``std(ddof=1) / sqrt(n_units)``.
    n_units : int
        Number of units.
    """
    mean: float
    se: float
    n_units: int


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
    check_all_finite(u, "outcomes")

    n = u.shape[0]
    check_min_units(n, what="the standard error", minimum=2)

    return MeanSE(float(u.mean()), float(u.std(ddof=1) / math.sqrt(n)), n)


def paired_differences(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Compute per-unit paired differences ``candidate - reference``.

    Replicates are averaged within each unit before taking the difference.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements of shape ``(n_units,)`` or ``(n_units, n_replicates)``.
    candidate : np.ndarray
        Candidate measurements, same number of units as ``reference``. The number of replicates may differ.

    Returns
    -------
    np.ndarray
        Per-unit differences ``candidate - reference``, of shape ``(n_units,)``.

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

    Typically estimated from pilot data and passed as ``std`` to the power and sample-size functions for paired tests.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements, shape ``(n_units,)`` or ``(n_units, n_replicates)``.
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


def unpaired_std(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Pooled standard deviation of two independent groups.

    Typically estimated from pilot data and passed as ``std`` to the power and sample-size functions for unpaired tests.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements, shape ``(n_units,)`` or ``(n_units, n_replicates)``.
    candidate : np.ndarray
        Candidate measurements. The number of units and replicates may differ.

    Returns
    -------
    float
        Pooled sample standard deviation of the two groups, after averaging replicates within each unit.

    Notes
    -----
    Replicates are averaged within each unit first. The groups may have different numbers of units and replicates.
    """
    reference_units = unit_means(reference)
    candidate_units = unit_means(candidate)

    n_units_reference = reference_units.shape[0]
    n_units_candidate = candidate_units.shape[0]

    check_min_units(n_units_reference, what="the standard deviation", minimum=2)
    check_min_units(n_units_candidate, what="the standard deviation", minimum=2)

    sum_of_squares_reference = float(np.sum((reference_units - reference_units.mean()) ** 2))
    sum_of_squares_candidate = float(np.sum((candidate_units - candidate_units.mean()) ** 2))
    df = n_units_reference + n_units_candidate - 2

    return float(math.sqrt((sum_of_squares_reference + sum_of_squares_candidate) / df))


class UnpairedDifference(NamedTuple):
    """Difference between two independent group means, with its standard error.

    Attributes
    ----------
    estimate : float
        ``mean(candidate) - mean(reference)``.
    se : float
        Welch standard error, ``sqrt(se_reference**2 + se_candidate**2)``; the group variances are not assumed equal.
    n_units_reference : int
        Number of units in the reference group.
    n_units_candidate : int
        Number of units in the candidate group.
    """
    estimate: float
    se: float
    n_units_reference: int
    n_units_candidate: int


def unpaired_difference(reference: np.ndarray, candidate: np.ndarray) -> UnpairedDifference:
    """Compute the difference of two independent group means (candidate - reference) and its standard error.

    Each group is summarized by :func:`mean_and_se` (replicates averaged within each unit). The groups may have a
    different number of units.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements, shape ``(n_units,)`` or ``(n_units, n_replicates)``.
    candidate : np.ndarray
        Candidate measurements. The number of units and replicates may differ.

    Returns
    -------
    UnpairedDifference
        Difference, Welch standard error, and the number of units in each group.
    """
    r = mean_and_se(reference)
    c = mean_and_se(candidate)
    return UnpairedDifference(
        estimate=c.mean - r.mean,
        se=math.hypot(r.se, c.se),
        n_units_reference=r.n_units,
        n_units_candidate=c.n_units,
    )
