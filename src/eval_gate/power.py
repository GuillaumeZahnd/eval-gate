import math
from dataclasses import asdict, dataclass
from scipy.stats import norm

from .core import AlternativeHypothesis, Direction, get_spec
from ._checks import check_in_open_interval_0_1, check_finite_strictly_positive, check_finite


_MAX_N_UNITS = 10**9  # Upper limit of the search for the required number of units.


@dataclass(frozen=True)
class PairedPowerResult:
    """Result of a sample-size analysis for a paired comparison, with its design conditions.

    Attributes
    ----------
    n_units : int
        Smallest number of paired units reaching ``target_power``.
    power : float
        Power achieved with ``n_units``; at least ``target_power``.
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

    n_units: int
    power: float
    target_power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


@dataclass(frozen=True)
class UnpairedPowerResult:
    """Result of a sample-size analysis for an unpaired comparison, with its design conditions.

    Attributes
    ----------
    n_units_reference : int
        Smallest number of reference units reaching ``target_power`` (balanced design).
    n_units_candidate : int
        Smallest number of candidate units reaching ``target_power`` (balanced design).
    power : float
        Power achieved with these sample sizes; at least ``target_power``.
    target_power : float
        Power the design was required to reach.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis to be tested.
    margin : float
        Effective margin; ``0.0`` when the hypothesis has no margin.
    effect : float
        Assumed true mean difference (candidate - reference).
    std : float
        Assumed pooled standard deviation of the two groups.
    alpha : float
        Significance level of the test.
    """

    n_units_reference: int
    n_units_candidate: int
    power: float
    target_power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


@dataclass(frozen=True)
class PairedPowerAtResult:
    """Power of a paired comparison for a given number of units, with its design conditions.

    Attributes
    ----------
    n_units : int
        Number of paired units.
    power : float
        Probability that the test rejects the null hypothesis with ``n_units``.
        It is at most ``alpha`` when ``effect`` sits on the boundary of the null.
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

    n_units: int
    power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


@dataclass(frozen=True)
class UnpairedPowerAtResult:
    """Power of an unpaired comparison for a given sample size per group, with its design conditions.

    Attributes
    ----------
    n_units_reference : int
        Number of units in the reference group.
    n_units_candidate : int
        Number of units in the candidate group.
    power : float
        Probability that the test rejects the null hypothesis.
    alternative_hypothesis : AlternativeHypothesis
        The alternative hypothesis to be tested.
    margin : float
        Effective margin; ``0.0`` when the hypothesis has no margin.
    effect : float
        Assumed true mean difference (candidate - reference).
    std : float
        Assumed pooled standard deviation of the two groups.
    alpha : float
        Significance level of the test.
    """

    n_units_reference: int
    n_units_candidate: int
    power: float
    alternative_hypothesis: AlternativeHypothesis
    margin: float
    effect: float
    std: float
    alpha: float


def paired_power(
    n_units: int,
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> PairedPowerAtResult:
    """Compute the power of ``paired_p_value`` for a given number of units.

    The test is the normal approximation used by ``paired_p_value``, with ``std`` taken as known.

    Parameters
    ----------
    n_units : int
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
    PairedPowerAtResult
        The power together with the number of units and the design conditions. The power
        equals ``alpha`` when ``effect`` sits on the boundary of the null.

    Raises
    ------
    ValueError
        If ``n_units`` is less than 2.

    Notes
    -----
    Power is the probability that the test rejects the null hypothesis if the assumed difference holds. The
    calculation uses a normal approximation with the standard deviation taken as known; if the true
    standard deviation is larger than assumed, or is estimated from a small sample, the actual power can be lower.
    """
    if n_units < 2:
        raise ValueError(f"At least 2 units are required, got {n_units}.")

    spec, m = _prepare_power_params(effect, std, alternative_hypothesis, margin, alpha)

    return PairedPowerAtResult(
        n_units=n_units,
        power=_compute_paired_power(spec.direction, n_units, effect, std, m, alpha),
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        effect=effect,
        std=std,
        alpha=alpha,
    )


def unpaired_power(
    n_units_reference: int,
    n_units_candidate: int,
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sidèd" if False else "two_sided",
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> UnpairedPowerAtResult:
    """Compute the power of ``unpaired_p_value`` for given group sample sizes.

    The test is the normal approximation used by ``unpaired_p_value``, with ``std`` taken as the known pooled standard deviation.

    Parameters
    ----------
    n_units_reference : int
        Number of units in the reference group, at least 2.
    n_units_candidate : int
        Number of units in the candidate group, at least 2.
    effect : float
        Assumed true mean difference (candidate - reference).
    std : float
        Assumed pooled standard deviation across the two independent groups.
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis to be tested.
    margin : float, default=0.0
        Largest acceptable difference. Required to be positive for
        ``"non_inferiority"`` and ``"equivalence"``, and ignored otherwise.
    alpha : float, default=0.05
        Significance level of the test, in ``(0, 1)``.

    Returns
    -------
    UnpairedPowerAtResult
        The power together with group sample sizes and design conditions. The power
        equals ``alpha`` when ``effect`` sits on the boundary of the null.

    Raises
    ------
    ValueError
        If either group has fewer than 2 units, or if group sizes are unbalanced during power evaluation.

    Notes
    -----
    Power is the probability that the test rejects the null hypothesis if the assumed difference holds. The
    calculation uses a normal approximation with the standard deviation taken as known; if the true
    standard deviation is larger than assumed, or is estimated from a small sample, the actual power can be lower.
    """
    if n_units_reference < 2 or n_units_candidate < 2:
        raise ValueError(
            f"At least 2 units are required in each group, got reference={n_units_reference}, candidate={n_units_candidate}."
        )
    if n_units_reference != n_units_candidate:
        raise ValueError("Power evaluation currently requires balanced group sizes (n_units_reference == n_units_candidate).")

    spec, m = _prepare_power_params(effect, std, alternative_hypothesis, margin, alpha)

    power_val = _compute_unpaired_power(spec.direction, n_units_reference, effect, std, m, alpha)

    return UnpairedPowerAtResult(
        n_units_reference=n_units_reference,
        n_units_candidate=n_units_candidate,
        power=power_val,
        alternative_hypothesis=alternative_hypothesis,
        margin=m,
        effect=effect,
        std=std,
        alpha=alpha,
    )


def required_paired_n_units(
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    power: float = 0.8,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> PairedPowerResult:
    """Compute the smallest number of paired units reaching a target power.

    Parameters
    ----------
    effect : float
        Assumed true mean paired difference (candidate - reference). It must lie in the alternative hypothesis.
    std : float
        Assumed standard deviation of the per-unit paired differences.
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis to be tested.
    power : float, default=0.8
        Target probability of rejecting the null hypothesis, in ``(0, 1)``.
    margin : float, default=0.0
        Largest acceptable difference.
    alpha : float, default=0.05
        Significance level of the test, in ``(0, 1)``.

    Returns
    -------
    PairedPowerResult
        The required number of paired units, the power it achieves, and design conditions.

    Notes
    -----
    Power is the probability that the test rejects the null hypothesis if the assumed difference holds. The
    calculation uses a normal approximation with the standard deviation taken as known; if the true
    standard deviation is larger than assumed, or is estimated from a small sample, the actual power can be lower.
    """
    hi = _find_required_sample_size(
        effect, std, alternative_hypothesis, power, margin, alpha, _compute_paired_power
    )
    achieved = paired_power(
        n_units=hi,
        effect=effect,
        std=std,
        alternative_hypothesis=alternative_hypothesis,
        margin=margin,
        alpha=alpha,
    )
    return PairedPowerResult(target_power=power, **asdict(achieved))


def required_unpaired_n_units(
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
    *,
    power: float = 0.8,
    margin: float = 0.0,
    alpha: float = 0.05,
) -> UnpairedPowerResult:
    """Compute the smallest number of units per group in an unpaired design reaching a target power.

    Parameters
    ----------
    effect : float
        Assumed true mean difference (candidate - reference). It must lie in the alternative hypothesis.
    std : float
        Assumed pooled standard deviation across the two independent groups.
    alternative_hypothesis : AlternativeHypothesis, default="two_sided"
        The alternative hypothesis to be tested.
    power : float, default=0.8
        Target probability of rejecting the null hypothesis, in ``(0, 1)``.
    margin : float, default=0.0
        Largest acceptable difference.
    alpha : float, default=0.05
        Significance level of the test, in ``(0, 1)``.

    Returns
    -------
    UnpairedPowerResult
        The required sample size per group, the power achieved, and design conditions.

    Notes
    -----
    Power is the probability that the test rejects the null hypothesis if the assumed difference holds. The
    calculation uses a normal approximation with the standard deviation taken as known; if the true
    standard deviation is larger than assumed, or is estimated from a small sample, the actual power can be lower.
    """
    hi = _find_required_sample_size(
        effect, std, alternative_hypothesis, power, margin, alpha, _compute_unpaired_power
    )
    achieved = unpaired_power(
        n_units_reference=hi,
        n_units_candidate=hi,
        effect=effect,
        std=std,
        alternative_hypothesis=alternative_hypothesis,
        margin=margin,
        alpha=alpha,
    )
    return UnpairedPowerResult(target_power=power, **asdict(achieved))


def _find_required_n(reaches_target, max_units: int = _MAX_N_UNITS) -> int:
    """Shared bisection and exponential growth search for required sample size."""
    lo = 2
    hi = 2
    while not reaches_target(hi):
        lo = hi
        hi = min(hi * 2, max_units)
        if hi == max_units:
            raise ValueError(f"More than {max_units} units would be needed; check 'effect' and 'std'.")
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches_target(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


def _find_required_sample_size(
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis,
    power: float,
    margin: float,
    alpha: float,
    power_evaluator,
) -> int:
    """Validate inputs, resolve specification/margin, and find the required sample size via bisection."""
    check_in_open_interval_0_1(power, "power")
    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)
    _validate_inputs(effect, std, alpha)
    _validate_effect(spec.name, spec.direction, effect, m)

    def reaches_target(n: int) -> bool:
        return power_evaluator(spec.direction, n, effect, std, m, alpha) >= power

    return _find_required_n(reaches_target)


def _validate_inputs(effect: float, std: float, alpha: float) -> None:
    """Validate the numeric inputs shared by the power functions."""
    check_in_open_interval_0_1(alpha, "alpha")
    check_finite_strictly_positive(std, "std")
    check_finite(effect, "effect")


def _validate_effect(name: str, direction: Direction, effect: float, margin: float) -> None:
    """Validate that the assumed effect lies in the alternative hypothesis."""
    match direction:
        case "two_sided":
            ok = effect != 0.0
            target_region = "nonzero"
        case "greater":
            ok = effect > -margin
            target_region = f"greater than {0.0 - margin:g}"
        case "equivalence":
            ok = abs(effect) < margin
            target_region = f"strictly within ±{margin:g}"
    if not ok:
        raise ValueError(f"For '{name}', 'effect' must be {target_region}, got {effect}.")


def _core_power(
    direction: Direction,
    scale: float,
    effect: float,
    margin: float,
    alpha: float,
) -> float:
    """Core normal-approximation power calculation given a standard-error scale."""
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


def _compute_paired_power(
    direction: Direction,
    n_units: float,
    effect: float,
    std: float,
    margin: float,
    alpha: float,
) -> float:
    """Return the power of the paired normal-approximation test for a known ``std`` (no validation)."""
    scale = math.sqrt(n_units) / std  # converts a difference into standard-error units
    return _core_power(direction, scale, effect, margin, alpha)


def _compute_unpaired_power(
    direction: Direction,
    n_units: float,
    effect: float,
    std: float,
    margin: float,
    alpha: float,
) -> float:
    """Internal math helper: return the power of the unpaired normal-approximation test."""
    scale = math.sqrt(n_units / 2) / std
    return _core_power(direction, scale, effect, margin, alpha)


def _prepare_power_params(
    effect: float,
    std: float,
    alternative_hypothesis: AlternativeHypothesis,
    margin: float,
    alpha: float,
) -> tuple:
    """Shared validation and specification resolution for power functions."""
    spec = get_spec(alternative_hypothesis)
    m = spec.margin_for(margin)
    _validate_inputs(effect, std, alpha)
    return spec, m
