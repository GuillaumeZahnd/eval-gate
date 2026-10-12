import numpy as np
from typing import Literal, get_args


Distributions = Literal["gaussian", "uniform", "student_t", "lognormal", "bimodal"]
DISTRIBUTIONS = get_args(Distributions)


def standardized_distribution(
    distribution: str,
    nb_units: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw random samples with a mean of 0 and a variance of 1
    from a specified probability distribution.

    Parameters
    ----------
    distribution : Distributions
        The target distribution type. Must be one of:
        'gaussian', 'uniform', 'student_t', 'lognormal', or 'bimodal'.
    nb_units : int | tuple[int, ...]
        Output shape: can be an integer for 1D arrays of `nb_units`,
        or a tuple `(n_units, k_resamples)` for 2D arrays with multiple resamples per unit.
    rng : np.random.Generator
        The NumPy random number generator instance used for sampling.

    Returns
    -------
    np.ndarray
            An array of standardized float samples matching the requested shape.

    Raises
    ------
    ValueError
        If the provided `distribution` string is not supported.
    """

    if distribution == "gaussian":
        return rng.standard_normal(nb_units)

    elif distribution == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), nb_units)

    elif distribution == "student_t":
        return rng.standard_t(3, nb_units) / np.sqrt(3)

    elif distribution == "lognormal":
        s = 1.0
        x = np.exp(s * rng.standard_normal(nb_units))
        mean = np.exp(s**2 / 2)
        var = (np.exp(s**2) - 1) * np.exp(s**2)
        return (x - mean) / np.sqrt(var)

    elif distribution == "bimodal":
        signs = rng.choice([-1.0, 1.0], size=nb_units)
        return (signs + 0.5 * rng.standard_normal(nb_units)) / np.sqrt(1.25)

    else:
        raise ValueError(f"'distribution' must be one of {DISTRIBUTIONS}, got '{distribution}'.")
