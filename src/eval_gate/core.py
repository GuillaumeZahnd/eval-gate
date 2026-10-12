import math
from typing import Literal, get_args
from scipy.stats import norm


AlternativeHypothesis = Literal["two_sided", "greater", "less"]
ALTERNATIVE_HYPOTHESES = get_args(AlternativeHypothesis)


def confidence_bounds(
    alternative_hypothesis: str,
    estimate: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculates normal-approximation confidence bounds for a scalar estimate.

    Parameters
    ----------
    alternative_hypothesis : str
        The type of confidence bound. Must be one of:
        - 'two_sided': Standard (1 - alpha) two-sided confidence interval.
        - 'greater': One-sided lower bound (from estimate minus margin to infinity).
        - 'less': One-sided upper bound (from negative infinity to estimate plus margin).
    estimate : float
        The point estimate value.
    se : float
        The standard error of the estimate.
    alpha : float, default=0.05
        Significance level (e.g., 0.05 for a 95% confidence interval).

    Returns
    -------
    tuple[float, float]
        A tuple containing the lower and upper bounds: `(lower_bound, upper_bound)`.

    Raises
    ------
    ValueError
        If the provided `alternative_hypothesis` string is not supported.
    """
    if alternative_hypothesis == "two_sided":
        half = float(norm.ppf(1 - alpha / 2) * se)
        return float(estimate - half), float(estimate + half)

    elif alternative_hypothesis == "greater":
        half = float(norm.ppf(1 - alpha) * se)
        return float(estimate - half), math.inf

    elif alternative_hypothesis == "less":
        half = float(norm.ppf(1 - alpha) * se)
        return -math.inf, float(estimate + half)

    else:
        raise ValueError(
            f"'alternative_hypothesis' must be one of {ALTERNATIVE_HYPOTHESES}, got '{alternative_hypothesis}'."
        )
