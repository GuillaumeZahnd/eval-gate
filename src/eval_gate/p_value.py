import numpy as np
from scipy.stats import norm

from .core import AlternativeHypothesis, get_spec
from .utils import _paired_differences, mean_and_se


def paired_p_value(
    a: np.ndarray,
    b: np.ndarray,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    margin: float = 0.0,
) -> float:
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

    Returns
    -------
    float
        Probability, under the null, of a result at least as favourable to the alternative as the one observed.

    Raises
    ------
    ValueError
        If the standard error is zero.
    """
    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)

    d = _paired_differences(a, b)
    diff, se, _ = mean_and_se(d)

    if se == 0.0:
        raise ValueError("All paired differences are identical, so the standard error is 0.")

    def greater(b: float) -> float:  # P(T >= t) under diff = b
        return float(norm.sf((diff - b) / se))

    def lesser(b: float) -> float:  # P(T <= t) under diff = b
        return float(norm.cdf((diff - b) / se))

    match spec.direction:
        case "two_sided":
            return 2 * min(greater(0.0), lesser(0.0))
        case "greater":
            return greater(-m)
        case "equivalence":
            return max(greater(-m), lesser(m))
