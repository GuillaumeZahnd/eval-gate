from collections.abc import Sequence

from .sem import SEMResult
from .p_value import PairedPValueResult, UnpairedPValueResult
from .power import (
    PairedPowerResult,
    UnpairedPowerResult,
    PairedPowerAtResult,
    UnpairedPowerAtResult,
)
from .core import AlternativeHypothesis
from .effect_size import PairedEffectResult, UnpairedEffectResult, magnitude_label


SMALL_N = 30  # Below this many units, the normal approximation can be noticeably optimistic


def _pct(x: float) -> str:
    """Format a proportion as a percentage, e.g. ``0.95`` -> ``"95%"``, ``0.999`` -> ``"99.9%"``."""
    return f"{round(x * 100, 6):g}%"


def _fmt_p(p: float) -> str:
    """Format a p-value for prose, e.g. ``0.0123`` -> ``"p = 0.012"``, ``1e-6`` -> ``"p < 0.001"``."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _report(result: str, verdict: str, caveats: Sequence[str] = ()) -> str:
    """Join the three blocks of an interpretation: result, verdict and, only when there are any, caveats."""
    return "\n".join([f"[RESULT] {result}", f"[VERDICT] {verdict}", *(f"[CAVEAT] {c}" for c in caveats)])


def _caveat_small_n(n: int, unit: str, consequence: str, *, smaller_group: bool = False) -> list[str]:
    """Return the small-sample caveat as a list of zero or one line (empty when ``n`` is at least ``SMALL_N``)."""
    if n >= SMALL_N:
        return []
    subject = f"The smaller group has only {n} {unit}" if smaller_group else f"Only {n} {unit}"
    return [f"{subject} (less than {SMALL_N}): {consequence}."]

# ----------------------------------------------------------------
# Standard error of the mean
# ----------------------------------------------------------------

def interpret_sem(
    result: SEMResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``compute_sem`` result.

    Parameters
    ----------
    result : SEMResult
        The result object returned by ``compute_sem``.
    metric : str, default="mean outcome"
        What the mean measures, e.g. ``"mean accuracy"`` or ``"mean systolic pressure"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for numbers; ``".1%"`` prints proportions as percentages.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    level = _pct(result.confidence)
    n = result.n_units
    ci_lower = num(result.ci_lower)
    ci_upper = num(result.ci_upper)

    # Result: the numbers and the inputs it was computed from
    numbers = f"The {metric} is estimated at {num(result.mean)} (SEM {num(result.sem)}, {n} {unit})."

    # Verdict: what the numbers achieve
    match result.alternative_hypothesis:
        case "two_sided":
            verdict = f"With {level} confidence, the true {metric} lies between {ci_lower} and {ci_upper}."
        case "superiority":
            verdict = f"With {level} confidence, the true {metric} is at least {ci_lower}."

    # Caveats: small-sample (only when triggered), identical outcome (only when triggered)
    caveats = _caveat_small_n(n, unit, "the normal approximation can make the interval somewhat too narrow")
    if result.sem == 0:
        caveats.append(
            f"All {unit} gave the same outcome, so the interval has zero width "
            "(this reflects the sample, not certainty about the population)."
        )

    return _report(numbers, verdict, caveats)

# ----------------------------------------------------------------
# p-value
# ----------------------------------------------------------------

def _interpret_p_value(
    result: PairedPValueResult | UnpairedPValueResult,
    *,
    metric: str,
    fmt: str,
    sizes: str,
    smallest: int,
    smaller_group: bool,
) -> str:
    """Return a plain-language reading of a ``p_value`` result, shared by the paired and unpaired versions.

    Parameters
    ----------
    result : PairedPValueResult or UnpairedPValueResult
        The p-value result to read.
    metric : str
        What is being compared, e.g. ``"mean accuracy"``.
    fmt : str
        Format spec for the margin, estimate and standard error.
    sizes : str
        Sample size(s) shown next to the estimate, e.g. ``"12 units"``.
    smallest : int
        Size of the smallest sample behind the test, compared with ``SMALL_N``.
    smaller_group : bool
        Whether ``smallest`` is the smaller of two independent groups (unpaired test) or the number
        of paired units (paired test); it only changes the wording of the small-sample caveat.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    m = num(result.margin)

    # Verdict: what the numbers achieve
    match result.alternative_hypothesis:
        case "two_sided":
            direction = "higher" if result.estimate > 0 else "lower"
            if_rejected = f"Difference established. The candidate's {metric} is {direction} than the reference's."
            if_not_rejected = (
                "Difference not established. This does not show that the two are equal, "
                "only that the data do not rule out no difference."
            )
        case "superiority":
            if_rejected = f"Superiority established. The candidate's {metric} is higher than the reference's."
            if_not_rejected = (
                "Superiority not established. The candidate may still be higher, "
                "but the data are not strong enough to say so."
            )
        case "non_inferiority":
            if_rejected = (
                f"Non-inferiority established. A drop of {m} or more is ruled out; "
                "the candidate may be equal or higher."
            )
            if_not_rejected = (
                f"Non-inferiority not established. A drop of {m} or more cannot be ruled out; "
                "this does not show that the candidate is lower."
            )
        case "equivalence":
            if_rejected = f"Equivalence established. The difference between the candidate and the reference is within ±{m}."
            if_not_rejected = (
                f"Equivalence not established. A difference of {m} or more in at least one "
                "direction cannot be ruled out."
            )
    verdict = if_rejected if result.reject else if_not_rejected

    # Result: the numbers and the inputs it was computed from
    numbers = (
        f"{_fmt_p(result.p_value)} (significance level {result.alpha:g}). "
        f"Observed difference (candidate - reference): {num(result.estimate)} (SE {num(result.se)}, {sizes})."
    )

    # Caveats: small-sample (only when triggered)
    caveats = _caveat_small_n(
        smallest,
        "units",
        "the normal approximation can make the p-value somewhat too small",
        smaller_group=smaller_group,
    )

    return _report(numbers, verdict, caveats)


def interpret_paired_p_value(
    result: PairedPValueResult,
    *,
    metric: str = "mean outcome",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``paired_p_value`` result.

    Parameters
    ----------
    result : PairedPValueResult
        The result object returned by ``paired_p_value``. It carries the hypothesis, the margin, the significance
        level, the estimate, and the number of units used in the test.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    fmt : str, default=".3f"
        Format spec for the margin, estimate, and standard error; ``".1%"`` prints proportions as percentages.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n = result.n_units
    return _interpret_p_value(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n} units",
        smallest=n,
        smaller_group=False,
    )


def interpret_unpaired_p_value(
    result: UnpairedPValueResult,
    *,
    metric: str = "mean outcome",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``unpaired_p_value`` result.

    Parameters
    ----------
    result : UnpairedPValueResult
        The result object returned by ``unpaired_p_value``. It carries the hypothesis, the margin, the significance
        level, the estimate and the size of each group.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    fmt : str, default=".3f"
        Format spec for the margin, estimate, and standard error; ``".1%"`` prints proportions as percentages.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n_reference = result.n_units_reference
    n_candidate = result.n_units_candidate
    smallest = min(n_reference, n_candidate)
    return _interpret_p_value(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate units",
        smallest=smallest,
        smaller_group=True,
    )

# ----------------------------------------------------------------
# Power analysis (required n)
# ----------------------------------------------------------------

def _power_goal(alternative_hypothesis: AlternativeHypothesis, metric: str, margin: str) -> str:
    """Return what a rejected null hypothesis shows, e.g. ``"show that the candidate's mean outcome is higher"``."""
    match alternative_hypothesis:
        case "two_sided":
            return f"detect a difference in {metric}"
        case "superiority":
            return f"show that the candidate's {metric} is higher"
        case "non_inferiority":
            return f"show that the candidate's {metric} is not lower than the reference's by {margin} or more"
        case "equivalence":
            return f"show that the difference in {metric} is within ±{margin}"


def _interpret_required_n(
    result: PairedPowerResult | UnpairedPowerResult,
    *,
    metric: str,
    unit: str,
    fmt: str,
    sizes: str,
    std_label: str,
    smallest: int,
    smaller_group: bool,
) -> str:
    """Return a plain-language reading of a ``required_n_units`` result, shared by the paired and unpaired versions.

    Parameters
    ----------
    result : PairedPowerResult or UnpairedPowerResult
        The sample-size result to read.
    metric : str
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str
        Format spec for the effect, margin and standard deviation.
    sizes : str
        The required sample size(s), e.g. ``"134 units"`` or ``"200 reference and 150 candidate units"``.
    std_label : str
        Name of the assumed variability, e.g. ``"standard deviation of the per-unit differences"``.
    smallest : int
        Size of the smallest sample, compared with ``SMALL_N``.
    smaller_group : bool
        Whether ``smallest`` is the smaller of two independent groups; it only changes the caveat wording.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    power = _pct(result.target_power)
    goal = _power_goal(result.alternative_hypothesis, metric, num(result.margin))

    # Result: the numbers and the inputs it was computed from
    numbers = (
        f"{sizes} needed for at least {power} power at significance level {result.alpha:g}. "
        f"Assumed true difference (candidate - reference): {num(result.effect)}; "
        f"assumed {std_label}: {num(result.std)} (standardized effect {result.effect / result.std:.2f})."
    )

    # Verdict: what the numbers achieve
    verdict = f"With {sizes}, the test would {goal} at least {power} of the time under these assumptions."

    # Caveats: small-sample (only when triggered), planning (always)
    caveats = _caveat_small_n(
        smallest,
        unit,
        "the normal approximation can make this number somewhat too low; treat it as a lower bound",
        smaller_group=smaller_group,
    )
    caveats.append(
        "The number is only as reliable as the assumed difference and standard deviation; "
        "a larger standard deviation needs more units."
    )

    return _report(numbers, verdict, caveats)


def interpret_paired_power(
    result: PairedPowerResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``required_n_units`` result.

    Parameters
    ----------
    result : PairedPowerResult
        The result object returned by ``required_n_units``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the effect, margin and standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized effect always uses two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n = result.n_units
    return _interpret_required_n(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n} {unit}",
        std_label="standard deviation of the per-unit differences",
        smallest=n,
        smaller_group=False,
    )


def interpret_unpaired_power(
    result: UnpairedPowerResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of an unpaired ``required_n_units`` result.

    Parameters
    ----------
    result : UnpairedPowerResult
        The result object returned by the unpaired ``required_n_units``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the effect, margin and standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized effect always uses two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n_reference = result.n_units_reference
    n_candidate = result.n_units_candidate
    return _interpret_required_n(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate {unit}",
        std_label="pooled standard deviation",
        smallest=min(n_reference, n_candidate),
        smaller_group=True,
    )

# ----------------------------------------------------------------
# Power analysis (power at)
# ----------------------------------------------------------------

def _power_status(alternative_hypothesis: AlternativeHypothesis, effect: float, margin: float, power: float) -> str:
    """Classify a power result as ``"not_a_power"``, ``"adequate"``, ``"underpowered"`` or ``"badly_underpowered"``.

    ``"not_a_power"`` means the assumed ``effect`` lies in the null region of the hypothesis, where the
    rejection probability cannot exceed ``alpha``. Otherwise the power is adequate from 80%, underpowered
    from 50% to 80%, and badly underpowered below 50%.
    """
    match alternative_hypothesis:
        case "two_sided":
            in_null = effect == 0
        case "superiority" | "non_inferiority":
            in_null = effect <= -margin
        case "equivalence":
            in_null = abs(effect) >= margin

    if in_null:
        return "not_a_power"
    if power >= 0.8:
        return "adequate"
    if power >= 0.5:
        return "underpowered"
    return "badly_underpowered"


def _interpret_power_at(
    result: PairedPowerAtResult | UnpairedPowerAtResult,
    *,
    metric: str,
    unit: str,
    fmt: str,
    sizes: str,
    std_label: str,
    smallest: int,
    smaller_group: bool,
) -> str:
    """Return a plain-language reading of a ``power_at`` result, shared by the paired and unpaired versions.

    Parameters
    ----------
    result : PairedPowerAtResult or UnpairedPowerAtResult
        The power result to read.
    metric : str
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str
        Format spec for the effect, margin and standard deviation.
    sizes : str
        The sample size(s) the power refers to, e.g. ``"200 units"`` or ``"200 reference and 150 candidate units"``.
    std_label : str
        Name of the assumed variability, e.g. ``"standard deviation of the per-unit differences"``.
    smallest : int
        Size of the smallest sample, compared with ``SMALL_N``.
    smaller_group : bool
        Whether ``smallest`` is the smaller of two independent groups; it only changes the caveat wording.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    power = result.power
    goal = _power_goal(result.alternative_hypothesis, metric, num(result.margin))

    # Result: the numbers and the inputs it was computed from
    numbers = (
        f"Power {power:.1%} with {sizes} at significance level {result.alpha:g}. "
        f"Assumed true difference (candidate - reference): {num(result.effect)}; "
        f"assumed {std_label}: {num(result.std)} (standardized effect {result.effect / result.std:.2f})."
    )

    # Verdict: what the numbers achieve
    match _power_status(result.alternative_hypothesis, result.effect, result.margin, power):
        case "adequate":
            verdict = (
                f"Adequate. Under these assumptions, the test would {goal} {power:.1%} of the time "
                "(at least the usual 80%)."
            )
        case "underpowered":
            verdict = (
                f"Underpowered. Under these assumptions, the test would fail to {goal} {1 - power:.1%} of the time "
                "(the usual target is 80% power)."
            )
        case "badly_underpowered":
            verdict = (
                f"Badly underpowered. Under these assumptions, the test would fail to {goal} {1 - power:.1%} "
                "of the time, more often than it succeeds."
            )
        case "not_a_power":
            verdict = (
                "Not a power. The assumed difference lies in the null region of the hypothesis, "
                f"so the test rejects at most {result.alpha:g} of the time."
            )

    # Caveats: small-sample (only when triggered), planning (always)
    caveats = _caveat_small_n(
        smallest,
        unit,
        "the normal approximation can make the power somewhat overestimated",
        smaller_group=smaller_group,
    )
    caveats.append("The power is only as reliable as the assumed difference and standard deviation.")

    return _report(numbers, verdict, caveats)


def interpret_paired_power_at(
    result: PairedPowerAtResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``paired_power`` result.

    Parameters
    ----------
    result : PairedPowerAtResult
        The result object returned by ``paired_power``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the effect, margin and standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized effect always uses two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n = result.n_units
    return _interpret_power_at(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n} {unit}",
        std_label="standard deviation of the per-unit differences",
        smallest=n,
        smaller_group=False,
    )


def interpret_unpaired_power_at(
    result: UnpairedPowerAtResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of an ``unpaired_power`` result.

    Parameters
    ----------
    result : UnpairedPowerAtResult
        The result object returned by ``unpaired_power``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the effect, margin and standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized effect always uses two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n_reference = result.n_units_reference
    n_candidate = result.n_units_candidate
    return _interpret_power_at(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate {unit}",
        std_label="pooled standard deviation",
        smallest=min(n_reference, n_candidate),
        smaller_group=True,
    )

# ----------------------------------------------------------------
# Effect size
# ----------------------------------------------------------------

def _interpret_effect_size(
    result: PairedEffectResult | UnpairedEffectResult,
    *,
    metric: str,
    unit: str,
    fmt: str,
    sizes: str,
    smallest: int,
    smaller_group: bool,
) -> str:
    """Return a plain-language reading of a ``compute_effect_size`` result, shared by the paired and unpaired versions.

    The text has a ``[RESULT]`` line (raw difference, Hedges' g and its interval), a ``[VERDICT]`` line, and
    ``[CAVEAT]`` lines when triggered. The verdict is one of six outcomes, decided by where the interval lies
    relative to zero and to the negligible band (``|g|`` below 0.2):

    - difference established, with the range of plausible sizes;
    - difference established, but its size is unclear and may be negligible;
    - difference established, but negligible;
    - no meaningful difference (effects beyond negligible ruled out, not a formal equivalence test);
    - inconclusive, with one direction beyond negligible ruled out;
    - inconclusive.

    Parameters
    ----------
    result : PairedEffectResult or UnpairedEffectResult
        The effect size result to read.
    metric : str
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str
        Format spec for the raw difference. The standardized values always use two decimals.
    sizes : str
        Sample size(s) shown next to the raw difference, e.g. ``"12 units"``.
    smallest : int
        Size of the smallest sample, compared with ``SMALL_N``.
    smaller_group : bool
        Whether ``smallest`` is the smaller of two independent groups; it only changes the caveat wording.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    g = result.hedges_g
    lower = result.ci_lower
    upper = result.ci_upper
    direction = "higher" if g > 0 else "lower"
    change = "increase" if g > 0 else "decrease"
    level = _pct(1.0 - result.alpha)

    # Result: the numbers and the inputs it was computed from
    numbers = (
        f"Mean difference (candidate - reference): {num(result.mean_difference)} ({sizes}). "
        f"Hedges' g = {g:.2f}, {level} CI [{lower:.2f}, {upper:.2f}]."
    )

    # Classification: where the interval lies relative to zero and to the negligible band
    zero_inside = lower <= 0 <= upper
    lower_label = magnitude_label(lower)
    upper_label = magnitude_label(upper)
    # When zero lies outside the interval, g has the sign of the whole interval, so the end
    # nearest to zero is the lower one if g > 0 and the upper one otherwise.
    near_label, far_label = (lower_label, upper_label) if g > 0 else (upper_label, lower_label)
    # Number of interval ends that are negligible (0, 1 or 2). With zero outside the interval, a single
    # negligible end is necessarily the nearest one.
    n_negligible_bounds = (lower_label == "negligible") + (upper_label == "negligible")

    if zero_inside:
        case = ("inconclusive", "inconclusive_one_side", "no_meaningful_difference")[n_negligible_bounds]
    else:
        case = ("established", "established_unclear_size", "established_negligible")[n_negligible_bounds]

    # Verdict: what the numbers achieve
    claim = f"The candidate's {metric} is {direction} than the reference's"
    match case:
        case "established":
            if near_label == far_label:
                plausible = f"all plausible effects are {far_label} {change}s"
            else:
                plausible = f"plausible effects range from a {near_label} to a {far_label} {change}"
            verdict = f"Difference established. {claim}; {plausible}."
        case "established_unclear_size":
            verdict = (
                f"Difference established. {claim}, but its size is unclear: "
                f"plausible effects range from a negligible to a {far_label} {change}."
            )
        case "established_negligible":
            verdict = f"Difference established. {claim}, but all plausible effects are negligible."
        case "no_meaningful_difference":
            verdict = "No meaningful difference. Effects beyond negligible are ruled out in both directions."
        case "inconclusive_one_side":
            if lower_label == "negligible":
                ruled_out, possible, possible_label = "A decrease", "an increase", upper_label
            else:
                ruled_out, possible, possible_label = "An increase", "a decrease", lower_label
            verdict = (
                f"Inconclusive. {ruled_out} beyond negligible is ruled out, "
                f"but {possible} up to a {possible_label} effect is not."
            )
        case "inconclusive":
            verdict = f"Inconclusive. Plausible effects range from a {lower_label} decrease to a {upper_label} increase."

    # Caveats: small-sample (only when triggered), no meaningful difference (only when triggered)
    caveats = []
    if case == "no_meaningful_difference":
        caveats.append("This is not a formal equivalence test.")
    caveats.extend(
        _caveat_small_n(
            smallest, unit, "the normal approximation can make the interval somewhat too narrow", smaller_group=smaller_group
        )
    )

    return _report(numbers, verdict, caveats)


def interpret_paired_effect_size(
    result: PairedEffectResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``compute_paired_effect_size`` result.

    Parameters
    ----------
    result : PairedEffectResult
        The result object returned by ``compute_paired_effect_size``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the raw difference; ``".1%"`` prints proportions as percentages.
        The standardized values always use two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n = result.n_units
    return _interpret_effect_size(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n} {unit}",
        smallest=n,
        smaller_group=False,
    )


def interpret_unpaired_effect_size(
    result: UnpairedEffectResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``compute_unpaired_effect_size`` result.

    Parameters
    ----------
    result : UnpairedEffectResult
        The result object returned by ``compute_unpaired_effect_size``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the raw difference; ``".1%"`` prints proportions as percentages.
        The standardized values always use two decimals.

    Returns
    -------
    str
        The ``[RESULT]`` and ``[VERDICT]`` lines, followed by one ``[CAVEAT]`` line per caveat.
    """
    n_reference = result.n_units_reference
    n_candidate = result.n_units_candidate
    return _interpret_effect_size(
        result,
        metric=metric,
        unit=unit,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate {unit}",
        smallest=min(n_reference, n_candidate),
        smaller_group=True,
    )
