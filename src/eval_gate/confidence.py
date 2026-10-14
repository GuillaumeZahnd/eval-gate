import math
from scipy.stats import norm

from .core import AlternativeHypothesis, get_spec
from ._checks import check_probability, check_finite_non_negative


def confidence_bounds(
    alternative_hypothesis: AlternativeHypothesis,
    estimate: float,
    se: float,
    alpha: float = 0.05,
    *,
    margin: float = 0.0,
) -> tuple[float, float]:
    """Compute normal-approximation confidence bounds dual to the test at level ``alpha``.

    Rejecting the null at level ``alpha`` is equivalent to the bounds excluding the null region:

    - ``"two_sided"``: symmetric ``(1 - alpha)`` interval.
    - ``"superiority"`` / ``"non_inferiority"``: lower bound at confidence ``1 - alpha``; the upper bound is ``+inf``.
    - ``"equivalence"``: the ``(1 - 2 * alpha)`` two-sided interval (TOST duality).

    Parameters
    ----------
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis the bounds are dual to.
    estimate : float
        Point estimate (e.g., a paired mean difference, Candidate - Reference; or a plain mean).
    se : float
        Standard error of the estimate.
    alpha : float, default=0.05
        Significance level of the dual test, in ``(0, 1)``.
    margin : float, default=0.0
        Acceptability margin. Required to be positive for ``"non_inferiority"`` and ``"equivalence"``.
        Validated, not used in the computation.

    Returns
    -------
    tuple[float, float]
        ``(lower bound, upper bound)``
    """
    check_probability(alpha, "alpha")
    check_finite_non_negative(se, "se")

    spec = get_spec(alternative_hypothesis)

    # Validated only so invalid margins fail identically to "paired_p_value"; the bounds themselves do not depend on it.
    spec.margin_for(margin)

    match spec.direction:
        case "two_sided":
            half = float(norm.ppf(1 - alpha / 2)) * se
            return estimate - half, estimate + half
        case "greater":
            full = float(norm.ppf(1 - alpha)) * se
            return estimate - full, math.inf
        case "equivalence":
            half = float(norm.ppf(1 - alpha)) * se
            return estimate - half, estimate + half
