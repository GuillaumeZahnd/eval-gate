import math
import numpy as np
from typing import NamedTuple


class MeanSE(NamedTuple):
    mean: float
    se: float
    nb_units: int


def unit_means(outcomes: np.ndarray) -> np.ndarray:
    """Return one value per unit, averaging replicates of shape ``(n, k)`` to ``(n,)``."""
    x = np.asarray(outcomes, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise ValueError(f"Expected a 1D or 2D array, got shape {x.shape}.")


def mean_and_se(outcomes: np.ndarray) -> MeanSE:
    """Mean and unit-clustered standard error of the mean, requiring at least 2 units."""
    u = unit_means(outcomes)
    n = u.shape[0]
    if n < 2:
        raise ValueError(f"At least 2 units are required to compute the standard error, got {n}.")
    return MeanSE(float(u.mean()), float(u.std(ddof=1) / math.sqrt(n)), n)


def _format_pairs_of_arrays(a_in: np.ndarray, b_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and format a_out pair of arrays for paired statistical analysis.

    Coerces inputs to float arrays, promotes 1D arrays to 2D column vectors, and verifies that shapes match.
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
    """Compute per-unit paired differences (candidate B minus baseline A).

    Averages any multi-replicate dimensions within each unit before computing the difference.

    Parameters
    ----------
    a : np.ndarray
        Baseline measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.
    b : np.ndarray
        Candidate measurements of shape ``(nb_units,)`` or ``(nb_units, nb_replicates)``.

    Returns
    -------
    np.ndarray
        A 1D array of shape ``(nb_units,)`` containing the per-unit differences.
    """
    a_formatted, b_formatted = _format_pairs_of_arrays(a, b)
    return b_formatted.mean(axis=1) - a_formatted.mean(axis=1)
