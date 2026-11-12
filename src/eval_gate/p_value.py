from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from .core import AlternativeHypothesis, Direction, get_spec
from ._units import paired_differences, unpaired_difference, mean_and_se
from ._checks import check_in_open_interval_0_1, check_se_is_non_zero


def _normal_p_value(direction: Direction, estimate: float, se: float, margin: float) -> float:
    """Return the normal-approximation p-value of ``estimate`` for a test direction (no validation)."""

    def greater(null: float) -> float:  # P(T >= t) under difference = null
        return float(norm.sf((estimate - null) / se))

    def lesser(null: float) -> float:  # P(T <= t) under difference = null
        return float(norm.cdf((estimate - null) / se))

    match direction:
        case "two_sided":
            p = 2 * min(greater(0.0), lesser(0.0))
        case "greater":
            p = greater(-margin)
        case "equivalence":
            p = max(greater(-margin), lesser(margin))
    return min(p, 1.0)


@dataclass(frozen=True)
class PairedPValueResult:
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
        Standard error of the estimate, ``std(ddof=1) / sqrt(n_units)``.
    n_units : int
        Number of paired units.
    alpha : float
        Significance level of the decision. It does not affect ``p_value``.
    """
    p_value: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    estimate: float
    se: float
    n_units: int
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
) -> PairedPValueResult:
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
    PairedPValueResult
        The p-value together with the hypothesis, margin, estimate, standard error and number of units.

    Raises
    ------
    ValueError
        If the standard error is zero.

    Notes
    -----
    The p-value is the probability, if the null hypothesis holds, of a result at least as favorable to the
    alternative as the one observed. It is not the probability that the null hypothesis is true.

    For ``"two_sided"`` and ``"superiority"``, a small p-value shows that a difference is detectable,
    not that it is large: report an effect size alongside it.
    """
    check_in_open_interval_0_1(alpha, "alpha")

    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)

    diff, se, n_units = mean_and_se(paired_differences(a, b))

    check_se_is_non_zero(se)

    p_value = _normal_p_value(
        direction=spec.direction,
        estimate=diff,
        se=se,
        margin=m,
    )

    return PairedPValueResult(
        p_value=p_value,
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        estimate=diff,
        se=se,
        n_units=n_units,
        alpha=alpha,
    )


@dataclass(frozen=True)
class UnpairedPValueResult:
    """Result of an unpaired (Welch) p-value analysis.

    Attributes
    ----------
    p_value : float
        Probability, under the null, of a result at least as favourable to the alternative as the one observed.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis that was tested.
    margin : float
        Effective margin used in the test; ``0.0`` when the hypothesis has no margin.
    estimate : float
        Observed difference of means, ``mean(candidate) - mean(reference)``.
    se : float
        Welch standard error of the estimate, ``sqrt(s_reference**2 / n_reference + s_candidate**2 / n_candidate)``.
    n_units_reference : int
        Number of units in the reference group.
    n_units_candidate : int
        Number of units in the candidate group.
    alpha : float
        Significance level of the decision. It does not affect ``p_value``.
    """
    p_value: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    estimate: float
    se: float
    n_units_reference: int
    n_units_candidate: int
    alpha: float

    @property
    def reject(self) -> bool:
        """Whether the null hypothesis is rejected at level ``alpha``."""
        return self.p_value < self.alpha


def unpaired_p_value(
    a: np.ndarray,
    b: np.ndarray,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> UnpairedPValueResult:
    """Compute the p-value for a difference of means between two independent groups (Welch, normal approximation).

    The tested difference is ``Δ = mean(candidate) - mean(reference)``. The groups contain different units, may have
    different sizes, and are not assumed to have equal variances.

    Parameters
    ----------
    a : np.ndarray
        Reference (baseline) measurements of shape ``(n_reference,)`` or ``(n_reference, k)``.
    b : np.ndarray
        Candidate measurements of shape ``(n_candidate,)`` or ``(n_candidate, k)``.
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
    UnpairedPValueResult
        The p-value together with the hypothesis, margin, estimate, standard error and group sizes.

    Raises
    ------
    ValueError
        If a group has fewer than 2 units, or if the standard error is zero.

    Notes
    -----
    The p-value is the probability, if the null hypothesis holds, of a result at least as favorable to the
    alternative as the one observed. It is not the probability that the null hypothesis is true.

    For ``"two_sided"`` and ``"superiority"``, a small p-value shows that a difference is detectable,
    not that it is large: report an effect size alongside it.
    """
    check_in_open_interval_0_1(alpha, "alpha")

    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)

    difference = unpaired_difference(a, b)

    check_se_is_non_zero(difference.se)

    p_value = _normal_p_value(
        direction=spec.direction,
        estimate=difference.estimate,
        se=difference.se,
        margin=m,
    )

    return UnpairedPValueResult(
        p_value=p_value,
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        estimate=difference.estimate,
        se=difference.se,
        n_units_reference=difference.n_units_reference,
        n_units_candidate=difference.n_units_candidate,
        alpha=alpha,
    )
