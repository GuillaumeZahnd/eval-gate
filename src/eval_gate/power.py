import math
from dataclasses import asdict, dataclass
from scipy.stats import norm

from .core import AlternativeHypothesis, Direction, get_spec
from ._checks import check_probability, check_finite_non_negative, check_finite


_MAX_NB_UNITS = 10**9  # Upper limit of the search for the required number of units.


@dataclass(frozen=True)
class PowerAtResult:
    """Power of a paired comparison for a given number of units, with its design conditions.

    Attributes
    ----------
    nb_units : int
        Number of paired units.
    power : float
        Probability that the test rejects the null hypothesis with ``nb_units``. It equals ``alpha``
        when ``effect`` sits on the boundary of the null.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis to be tested.
    margin : float
        Effective margin; ``0.0`` when the hypothesis has no margin.
    effect : float
        Assumed true mean paired difference (candidate - reference).
    std : float
        Assumed standard deviation of the per-unit paired differences.
    alpha : float
        Significance level of the test.
    """

    nb_units: int
    power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


@dataclass(frozen=True)
class PowerResult:
    """Result of a sample-size analysis for a paired comparison, with its design conditions.

    Attributes
    ----------
    nb_units : int
        Smallest number of paired units reaching ``target_power``.
    power : float
        Power achieved with ``nb_units``; at least ``target_power``.
    target_power : float
        Power the design was required to reach.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis to be tested.
    margin : float
        Effective margin; ``0.0`` when the hypothesis has no margin.
    effect : float
        Assumed true mean paired difference (candidate - reference).
    std : float
        Assumed standard deviation of the per-unit paired differences.
    alpha : float
        Significance level of the test.
    """

    nb_units: int
    power: float
    target_power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


def _validate_inputs(effect: float, std: float, alpha: float) -> None:
    """Validate the numeric inputs shared by the power functions."""
    check_probability(alpha, "alpha")
    check_finite_non_negative(std, "std")
    check_finite(effect, "effect")


def _validate_effect(name: str, direction: Direction, effect: float, margin: float) -> None:
    """Validate that the assumed effect lies in the alternative hypothesis.

    Outside it, the power cannot exceed ``alpha``, so no sample size can reach a higher target.

    Raises
    ------
    ValueError
        If ``effect`` lies in the null region of the hypothesis.
    """
    match direction:
        case "two_sided":
            ok = effect != 0.0
            target_region = "nonzero"
        case "greater":
            ok = effect > -margin
            target_region = f"greater than {-margin:g}"
        case "equivalence":
            ok = abs(effect) < margin
            target_region = f"strictly within ±{margin:g}"
    if not ok:
        raise ValueError(f"For '{name}', 'effect' must be {target_region}, got {effect}.")


def _power(
    direction: Direction,
    nb_units: float,
    effect: float,
    std: float,
    margin: float,
    alpha: float
) -> float:
    """Return the power of the normal-approximation test for a known ``std`` (no validation)."""
    scale = math.sqrt(nb_units) / std  # converts a difference into standard-error units
    match direction:
        case "two_sided":
            z = float(norm.ppf(1 - alpha / 2))
            shift = effect * scale
            return float(norm.cdf(shift - z) + norm.cdf(-shift - z))
        case "greater":
            z = float(norm.ppf(1 - alpha))
            return float(norm.cdf((effect + margin) * scale - z))
        case "equivalence":
            z = float(norm.ppf(1 - alpha))
            lower = float(norm.cdf((effect + margin) * scale - z))
            upper = float(norm.cdf((margin - effect) * scale - z))
            return max(0.0, lower + upper - 1.0)


def paired_power(
    nb_units: int,
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> PowerAtResult:
    """Compute the power of ``paired_p_value`` for a given number of units.

    The test is the normal approximation used by ``paired_p_value``, with ``std`` taken as known.

    Parameters
    ----------
    nb_units : int
        Number of paired units, at least 2.
    effect : float
        Assumed true mean paired difference (candidate - reference).
    std : float
        Assumed standard deviation of the per-unit paired differences (replicates averaged within each unit).
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis to be tested.
    margin : float, default=0.0
        Largest acceptable difference. Required to be positive for
        ``"non_inferiority"`` and ``"equivalence"``, and ignored otherwise.
    alpha : float, default=0.05
        Significance level of the test, in ``(0, 1)``.

    Returns
    -------
    PowerAtResult
        The power together with the number of units and the design conditions. The power
        equals ``alpha`` when ``effect`` sits on the boundary of the null.

    Raises
    ------
    ValueError
        If ``nb_units`` is less than 2.
    """
    if nb_units < 2:
        raise ValueError(f"At least 2 units are required, got {nb_units}.")

    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)
    _validate_inputs(effect, std, alpha)

    return PowerAtResult(
        nb_units=nb_units,
        power=_power(spec.direction, nb_units, effect, std, m, alpha),
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        effect=effect,
        std=std,
        alpha=alpha,
    )


def required_nb_units(
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    power: float = 0.8,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> PowerResult:
    """Compute the smallest number of paired units reaching a target power.

    Parameters
    ----------
    effect : float
        Assumed true mean paired difference (candidate - reference). It must lie in the alternative hypothesis,
        e.g. ``effect > 0`` for ``"superiority"`` or ``|effect| < margin`` for ``"equivalence"``. For
        ``"non_inferiority"`` and ``"equivalence"``, ``0.0`` is a common planning value.
    std : float
        Assumed standard deviation of the per-unit paired differences (replicates averaged within each unit).
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis to be tested.
    power : float, default=0.8
        Target probability of rejecting the null hypothesis, in ``(0, 1)``.
    margin : float, default=0.0
        Largest acceptable difference. Required to be positive for ``"non_inferiority"`` and ``"equivalence"``,
        and ignored otherwise.
    alpha : float, default=0.05
        Significance level of the test, in ``(0, 1)``.

    Returns
    -------
    PowerResult
        The required number of units, the power it achieves, and the design conditions.

    Raises
    ------
    ValueError
        If more than ``10**9`` units would be needed.
    """
    check_probability(power, "power")
    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)
    _validate_inputs(effect, std, alpha)
    _validate_effect(spec.name, spec.direction, effect, m)

    def reaches_target(n: int) -> bool:
        return _power(spec.direction, n, effect, std, m, alpha) >= power

    # Power increases with the number of units: double until the target is met, then bisect.
    lo = 2
    hi = 2
    while not reaches_target(hi):
        lo = hi
        hi = hi * 2
        if hi > _MAX_NB_UNITS:
            raise ValueError(f"More than {_MAX_NB_UNITS} units would be needed; check 'effect' and 'std'.")
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches_target(mid):
            hi = mid
        else:
            lo = mid + 1

    achieved = paired_power(
        nb_units=hi,
        effect=effect,
        std=std,
        alternative_hypothesis=alternative_hypothesis,
        margin=margin,
        alpha=alpha,
    )
    return PowerResult(target_power=power, **asdict(achieved))
