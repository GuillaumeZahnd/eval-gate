from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from .core import AlternativeHypothesis, get_spec
from ._units import paired_differences, mean_and_se
from ._checks import check_probability


@dataclass(frozen=True)
class PValueResult:
    """Result of a paired p-value analysis.

    Attributes
    ----------
    p_value : float
        Probability, under the null, of a result at least as favourable to the alternative as the one observed.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis that was tested.
    margin : float
        Effective margin used in the test; ``0.0`` when the hypothesis has no margin.
    estimate : float
        Observed mean paired difference, ``mean(b - a)`` (candidate - reference).
    se : float
        Standard error of the estimate, ``std(ddof=1) / sqrt(nb_units)``.
    nb_units : int
        Number of paired units.
    alpha : float
        Significance level of the decision. It does not affect ``p_value``.
    """
    p_value: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    estimate: float
    se: float
    nb_units: int
    alpha: float

    @property
    def reject(self) -> bool:
        """Whether the null hypothesis is rejected at level ``alpha``."""
        return self.p_value < self.alpha


def paired_p_value(
    a: np.ndarray,
    b: np.ndarray,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> PValueResult:
    """Compute the p-value for a paired mean difference under a normal approximation.

    The tested difference is ``Δ = mean(b - a)`` (candidate - reference).

    Parameters
    ----------
    a : np.ndarray
        Reference (baseline) measurements of shape ``(n,)`` or ``(n, k)``.
    b : np.ndarray
        Candidate measurements of shape ``(n,)`` or ``(n, k)``.
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis being tested; the null hypothesis of each is:

        - ``"two_sided"``: the true difference is 0.
        - ``"superiority"``: the true difference is at most 0.
        - ``"non_inferiority"``: the true difference is at most ``-margin``.
        - ``"equivalence"``: the true difference is at least ``margin`` in
          absolute value (two one-sided tests; the larger p-value is returned).
    margin : float, default=0.0
        Largest acceptable difference. Required to be positive for
        ``"non_inferiority"`` and ``"equivalence"``, and ignored otherwise.
    alpha : float, default=0.05
        Significance level used for the decision, in ``(0, 1)``. It is stored in the result
        and does not affect the p-value.

    Returns
    -------
    PValueResult
        The p-value together with the hypothesis, margin, estimate, standard error and number of units.

    Raises
    ------
    ValueError
        If the standard error is zero.
    """
    check_probability(alpha, "alpha")

    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)

    diff, se, nb_units = mean_and_se(paired_differences(a, b))

    if se == 0.0:
        raise ValueError("All paired differences are identical, so the standard error is 0.")

    def greater(null: float) -> float:  # P(T >= t) under diff = null
        return float(norm.sf((diff - null) / se))

    def lesser(null: float) -> float:  # P(T <= t) under diff = null
        return float(norm.cdf((diff - null) / se))

    match spec.direction:
        case "two_sided":
            p = 2 * min(greater(0.0), lesser(0.0))
        case "greater":
            p = greater(-m)
        case "equivalence":
            p = max(greater(-m), lesser(m))

    return PValueResult(
        p_value=min(p, 1.0),
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        estimate=diff,
        se=se,
        nb_units=nb_units,
        alpha=alpha,
    )
