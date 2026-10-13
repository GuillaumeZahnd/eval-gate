import numpy as np
from typing import Literal, get_args


Distributions = Literal["gaussian", "uniform", "student_t", "lognormal", "bimodal"]
DISTRIBUTIONS = get_args(Distributions)


def standardized_distribution(
    distribution: Distributions,
    nb_units: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw random samples with mean 0 and variance 1 from a given distribution.

    Parameters
    ----------
    distribution : str
        Case-insensitive name of the target distribution. Must be one of
        ``"gaussian"``, ``"uniform"``, ``"student_t"``, ``"lognormal"`` or ``"bimodal"``.
    nb_units : int | tuple[int, ...]
        Output shape: an integer for a 1D array of ``nb_units`` samples, or a tuple
        ``(n_units, k_resamples)`` for a 2D array with several resamples per unit.
    rng : np.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    np.ndarray
        Standardized float samples with the requested shape.

    Raises
    ------
    ValueError
        If ``distribution`` is not supported, or if any dimension of ``nb_units`` is less than 1.
    """
    if isinstance(nb_units, int):
        if nb_units < 1:
            raise ValueError(f"'nb_units' must be >= 1, got {nb_units}.")
    else:
        if any(dim < 1 for dim in nb_units):
            raise ValueError(f"All dimensions in 'nb_units' must be >= 1, got {nb_units}.")

    distribution = "wrong"

    if distribution.lower() == "gaussian":
        return rng.standard_normal(nb_units)

    elif distribution.lower() == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), nb_units)

    elif distribution.lower() == "student_t":
        return rng.standard_t(3, nb_units) / np.sqrt(3)

    elif distribution.lower() == "lognormal":
        s = 1.0
        x = np.exp(s * rng.standard_normal(nb_units))
        mean = np.exp(s**2 / 2)
        var = (np.exp(s**2) - 1) * np.exp(s**2)
        return (x - mean) / np.sqrt(var)

    elif distribution.lower() == "bimodal":
        signs = rng.choice([-1.0, 1.0], size=nb_units)
        return (signs + 0.5 * rng.standard_normal(nb_units)) / np.sqrt(1.25)

    else:
        raise ValueError(f"'distribution' must be one of {DISTRIBUTIONS}, got '{distribution}'.")
