from .sem import SEMResult
from .p_value import PairedPValueResult, UnpairedPValueResult
from .power import PowerResult, PowerAtResult
from .effect_size import PairedEffectResult, UnpairedEffectResult, magnitude_label


SMALL_N = 30  # Below this many units, the normal approximation can be noticeably optimistic


def _pct(x: float) -> str:
    """Format a proportion as a percentage, e.g. ``0.95`` -> ``"95%"``, ``0.999`` -> ``"99.9%"``."""
    return f"{round(x * 100, 6):g}%"


def _fmt_p(p: float) -> str:
    """Format a p-value for prose, e.g. ``0.0123`` -> ``"p = 0.012"``, ``1e-6`` -> ``"p < 0.001"``."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def interpret_sem(
    result: SEMResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a one-paragraph plain-language reading of a ``compute_sem`` result.

    The paragraph states the estimate and its standard error, what the confidence
    interval says about the true mean, and caveats when the sample is degenerate
    (zero standard error) or small (fewer than ``SMALL_N`` units).

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
        A one-paragraph interpretive summary.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    level = _pct(result.confidence)
    n = result.nb_units
    alternative_hypothesis = result.alternative_hypothesis
    ci_lower = num(result.ci_lower)
    ci_upper = num(result.ci_upper)

    # [RESULT]
    head = f"[RESULT] The {metric} is estimated at {num(result.mean)} (SEM {num(result.sem)}, {n} {unit})."

    # [VERDICT]
    if alternative_hypothesis == "two_sided":
        verdict = f"With {level} confidence, the true {metric} lies between {ci_lower} and {ci_upper}."
    elif alternative_hypothesis == "superiority":
        verdict = f"With {level} confidence, the true {metric} is at least {ci_lower}."
    verdict = f"[VERDICT] {verdict}"

    # [CAVEAT]
    caveats = []
    if result.sem == 0:
        caveats.append(
            "[CAVEAT] "
            f"All {unit} gave the same outcome, so the interval has zero width "
            "(this reflects the sample, not certainty about the population)."
        )
    if n < SMALL_N:
        caveats.append(
            "[CAVEAT] "
            f"Only {n} {unit} (less than {SMALL_N}): the normal approximation can make the interval somewhat too narrow."
        )

    return " ".join([head, verdict, *caveats])


def _interpret_p_value(
    result: PairedPValueResult | UnpairedPValueResult,
    *,
    metric: str,
    fmt: str,
    sizes: str,
    smallest: int,
    smallest_label: str,
) -> str:
    """Return the one-paragraph reading shared by the paired and unpaired p-value interpretations.

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
    smallest_label : str
        How the smallest sample is named in the small-sample caveat, e.g. ``"Only 12 units"``.

    Returns
    -------
    str
        A one-paragraph interpretive summary.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    m = num(result.margin)
    significant = result.reject

    match result.alternative_hypothesis:
        case "two_sided":
            null = "there were no difference between the candidate and the reference"
            direction = "higher" if result.estimate > 0 else "lower"
            if_rejected = f"The candidate's {metric} differs from the reference's: it is {direction}."
            if_not_rejected = (
                "A difference cannot be established. This does not show that the two are equal, "
                "only that the data do not rule out no difference."
            )
        case "superiority":
            null = "the candidate were no higher than the reference"
            if_rejected = f"There is evidence that the candidate's {metric} is higher than the reference's."
            if_not_rejected = (
                "Superiority is not established. The candidate may still be higher, "
                "but the data are not strong enough to say so."
            )
        case "non_inferiority":
            null = f"the candidate were lower than the reference by {m} or more"
            if_rejected = (
                f"A drop of {m} or more is ruled out. The candidate may be equal or higher; "
                "this test does not distinguish between the two."
            )
            if_not_rejected = (
                f"Non-inferiority is not established: a drop of {m} or more cannot be ruled out. "
                "This does not show that the candidate is lower."
            )
        case "equivalence":
            null = f"the true difference were {m} or more in either direction"
            if_rejected = f"The difference between the candidate and the reference is within ±{m}."
            if_not_rejected = (
                f"Equivalence is not established: a difference of {m} or more in at least one "
                "direction cannot be ruled out."
            )

    head = (
        "[RESULT] "
        f"{_fmt_p(result.p_value)} (significance level {result.alpha:g}): "
        f"the null hypothesis is {'rejected' if significant else 'not rejected'}."
    )
    conclusion = (
        "[CONCLUSION] "
        f"{if_rejected if significant else if_not_rejected}"
    )
    about = (
        "[ABOUT] "
        f"This p-value is the probability, if {null}, of a result at least as favourable "
        "to the alternative as the one observed."
    )

    note_parts = [
        f"The observed difference (candidate - reference) is {num(result.estimate)} "
        f"(SE {num(result.se)}, {sizes})."
    ]
    if significant and result.alternative_hypothesis in ("two_sided", "superiority"):
        note_parts.append("A small p-value shows that a difference is detectable, not that it is large.")
    if smallest < SMALL_N:
        note_parts.append(
            f"{smallest_label} (less than {SMALL_N}): "
            "the normal approximation can make the p-value somewhat too small."
        )
    notes = "[NOTES] " + " ".join(note_parts)

    return "\n".join([head, conclusion, about, notes])


def interpret_paired_p_value(
    result: PairedPValueResult,
    *,
    metric: str = "mean outcome",
    fmt: str = ".3f",
) -> str:
    """Return a one-paragraph plain-language reading of a ``paired_p_value`` result.

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
        A one-paragraph interpretive summary.
    """
    n = result.nb_units
    return _interpret_p_value(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n} units",
        smallest=n,
        smallest_label=f"Only {n} units",
    )


def interpret_unpaired_p_value(
    result: UnpairedPValueResult,
    *,
    metric: str = "mean outcome",
    fmt: str = ".3f",
) -> str:
    """Return a one-paragraph plain-language reading of an ``unpaired_p_value`` result.

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
        A one-paragraph interpretive summary.
    """
    n_reference = result.nb_units_reference
    n_candidate = result.nb_units_candidate
    smallest = min(n_reference, n_candidate)
    return _interpret_p_value(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate units",
        smallest=smallest,
        smallest_label=f"The smaller group has only {smallest} units",
    )


def interpret_power(
    result: PowerResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``required_nb_units`` result.

    The text has one line per element:

    1. The required number of units, with the goal it serves and the criteria it meets
       (target power and significance level).
    2. The assumptions behind it: the true difference and the standard deviation of the
       per-unit differences, with the resulting standardized effect.
    3. A caveat when the number is small (fewer than ``SMALL_N`` units).
    4. "About the test", a sentence describing the method and its limits.

    Parameters
    ----------
    result : PowerResult
        The result object returned by ``required_nb_units``.
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
        A short interpretive text.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    n = result.nb_units
    margin = num(result.margin)

    # Line 1: the number, its goal, and the criteria it meets
    match result.alternative_hypothesis:
        case "two_sided":
            goal = f"detect a difference in {metric}"
        case "superiority":
            goal = f"show that the candidate's {metric} is higher"
        case "non_inferiority":
            goal = f"show that the candidate's {metric} is not lower than the reference's by {margin} or more"
        case "equivalence":
            goal = f"show that the difference in {metric} is within ±{margin}"

    head = (
        f"{n} {unit} are needed to {goal}, "
        f"with at least {_pct(result.target_power)} power at significance level {result.alpha:g}."
    )

    # Line 2: the assumptions
    assumptions = (
        f"Assumptions: a true difference (candidate - reference) of {num(result.effect)} "
        f"and a standard deviation of the per-unit differences of {num(result.std)} "
        f"(standardized effect {result.effect / result.std:.2f})."
    )

    # Line 3: caveat when the number is small
    small_n = []
    if n < SMALL_N:
        small_n.append(
            f"Only {n} {unit} (less than {SMALL_N}): the number is approximate and should be treated as a lower bound."
        )

    # Line 4: About the test
    about = (
        "About the test: power is the probability that the test rejects the null hypothesis if the assumed "
        "difference holds. It uses a normal approximation with the standard deviation taken as known; "
        "if the true standard deviation is larger than assumed, or is estimated from a small sample, "
        "the actual power can be lower."
    )

    return "\n".join([head, assumptions, *small_n, about])


def interpret_power_at(
    result: PowerAtResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``paired_power`` result.

    The text has one line per element:

    1. The power reached with the given number of units, and the goal it refers to.
    2. The conclusion, one of four outcomes decided by the power and by where the assumed
       difference lies:

       - adequate (power of at least 80%);
       - underpowered (50% to 80%);
       - badly underpowered (below 50%);
       - not a power (the assumed difference lies in the null region of the hypothesis).

    3. The assumptions behind it: the true difference and the standard deviation of the
       per-unit differences, with the resulting standardized effect.
    4. A caveat when the sample is small (fewer than ``SMALL_N`` units).
    5. "About the test", a sentence describing the method and its limits.

    Parameters
    ----------
    result : PowerAtResult
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
        A short interpretive text, with its lines separated by newlines.
    """

    def num(x: float) -> str:
        return format(x, fmt)

    n = result.nb_units
    power = result.power
    margin = num(result.margin)

    # Line 1: the power and its goal
    match result.alternative_hypothesis:
        case "two_sided":
            goal = f"detect a difference in {metric}"
        case "superiority":
            goal = f"show that the candidate's {metric} is higher"
        case "non_inferiority":
            goal = f"show that the candidate's {metric} is not lower than the reference's by {margin} or more"
        case "equivalence":
            goal = f"show that the difference in {metric} is within ±{margin}"

    head = f"With {n} {unit}, the test has {power:.1%} power to {goal}, at significance level {result.alpha:g}."

    # Classification: is the assumed difference in the null region, and how high is the power
    match result.alternative_hypothesis:
        case "two_sided":
            in_null = result.effect == 0
        case "superiority" | "non_inferiority":
            in_null = result.effect <= -result.margin
        case "equivalence":
            in_null = abs(result.effect) >= result.margin

    if in_null:
        case = "not_a_power"
    elif power >= 0.8:
        case = "adequate"
    elif power >= 0.5:
        case = "underpowered"
    else:
        case = "badly_underpowered"

    # Line 2: conclusion
    match case:
        case "adequate":
            verdict = f"adequate. The assumed difference would be detected {power:.1%} of the time (at least the usual 80%)."
        case "underpowered":
            verdict = (
                f"underpowered. The assumed difference would be missed {1 - power:.1%} of the time "
                "(the usual target is 80% power)."
            )
        case "badly_underpowered":
            verdict = (
                f"badly underpowered. The assumed difference is more likely to be missed than detected "
                f"(missed {1 - power:.1%} of the time)."
            )
        case "not_a_power":
            verdict = (
                "not a power. The assumed difference lies in the null region of the hypothesis, "
                f"so the test rejects at most {result.alpha:g} of the time."
            )
    conclusion = f"Conclusion: {verdict}"

    # Line 3: assumptions
    assumptions = (
        f"Assumptions: a true difference (candidate - reference) of {num(result.effect)} "
        f"and a standard deviation of the per-unit differences of {num(result.std)} "
        f"(standardized effect {result.effect / result.std:.2f})."
    )

    # Line 4: caveat when the number is small
    small_n = []
    if n < SMALL_N:
        small_n.append(
            f"Only {n} {unit} (less than {SMALL_N}): the power is approximate and can be somewhat overestimated."
        )

    # Line 5: About the test
    about = (
        "About the test: power is the probability that the test rejects the null hypothesis if the assumed "
        "difference holds. It uses a normal approximation with the standard deviation taken as known; "
        "if the true standard deviation is larger than assumed, or is estimated from a small sample, "
        "the actual power can be lower."
    )

    return "\n".join([head, conclusion, assumptions, *small_n, about])


def _interpret_effect_size(
    result: PairedEffectResult | UnpairedEffectResult,
    *,
    metric: str,
    fmt: str,
    sizes: str,
    smallest: int,
    smallest_label: str,
) -> str:
    """Return the short plain-language reading shared by paired and unpaired effect size interpretations."""

    def num(x: float) -> str:
        return format(x, fmt)

    g = result.hedges_g
    lower = result.ci_lower
    upper = result.ci_upper
    direction = "higher" if g > 0 else "lower"
    change = "increase" if g > 0 else "decrease"
    level = _pct(1.0 - result.alpha)

    # [RESULT]
    if result.mean_difference == 0:
        difference = "does not differ from the reference's on average"
    else:
        difference = f"is {num(abs(result.mean_difference))} {direction} than the reference's on average"

    head = (
        "[RESULT] "
        f"The candidate's {metric} {difference} ({sizes}), a {result.magnitude} standardized effect "
        f"(Hedges' g={g:.2f}, {level} CI [{lower:.2f}, {upper:.2f}])."
    )

    # Classification: where the interval lies relative to zero and to the negligible band
    zero_inside = lower <= 0 <= upper
    lower_label = magnitude_label(lower)
    upper_label = magnitude_label(upper)
    near_label, far_label = (lower_label, upper_label) if g > 0 else (upper_label, lower_label)
    n_negligible_bounds = (lower_label == "negligible") + (upper_label == "negligible")

    if zero_inside:
        case = ("inconclusive", "inconclusive_one_side", "no_meaningful_difference")[n_negligible_bounds]
    else:
        case = ("established", "established_unclear_size", "established_negligible")[n_negligible_bounds]

    # [VERDICT]
    match case:
        case "established":
            verdict = f"Difference established. The candidate is {direction}, by at least a {near_label} effect."
        case "established_unclear_size":
            verdict = (
                f"Difference established. The candidate is {direction}, but the effect size is unclear and may be negligible."
            )
        case "established_negligible":
            verdict = f"Difference established. The candidate is {direction}, but the effect is negligible."
        case "no_meaningful_difference":
            verdict = (
                "No meaningful difference. Effects beyond negligible are ruled out in both directions "
                "(this is not a formal equivalence test)."
            )
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
            verdict = "Inconclusive. The data are compatible with both positive and negative effects."
    verdict = f"[VERDICT] {verdict}"

    if n_negligible_bounds == 2:
        plausible = "All plausible effects are negligible."
    elif zero_inside:
        plausible = f"Plausible effects range from a {lower_label} decrease to a {upper_label} increase."
    elif near_label == far_label:
        plausible = f"All plausible effects are {far_label} {change}s."
    else:
        plausible = f"Plausible effects range from a {near_label} to a {far_label} {change}."

    if zero_inside:
        zero = f"Zero lies inside the {level} interval, so 'no difference' cannot be ruled out."
    else:
        zero = f"Zero lies outside the {level} interval, so 'no difference' is ruled out."

    # [CAVEAT]
    small_n = []
    if smallest < SMALL_N:
        small_n.append(
            "[CAVEAT] "
            f"{smallest_label} (less than {SMALL_N}): the interval is approximate and can be somewhat too narrow."
        )

    return "\n".join([head, f"{verdict} {plausible} {zero}", *small_n])


def interpret_paired_effect_size(
    result: PairedEffectResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``compute_paired_effect_size`` result.

    About the test: Hedges' g is the mean paired difference divided by the standard deviation of the
    per-unit differences, then corrected for small-sample bias; its interval uses a normal approximation,
    and the size labels are Cohen's rules of thumb (0.2, 0.5, 0.8).

    Parameters
    ----------
    result : PairedEffectResult
        The result object returned by ``compute_paired_effect_size``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the raw difference and its standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized values always use two decimals.

    Returns
    -------
    str
        A short plain-language reading of a ``compute_paired_effect_size`` result.
    """
    n = result.nb_units

    def num(x: float) -> str:
        return format(x, fmt)

    return _interpret_effect_size(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n} {unit}",
        smallest=n,
        smallest_label=f"Only {n} {unit}",
    )


def interpret_unpaired_effect_size(
    result: UnpairedEffectResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """Return a plain-language reading of a ``compute_unpaired_effect_size`` result.

    About the test: Hedges' g is the mean difference divided by the pooled standard deviation,
    then corrected for small-sample bias; its interval uses a normal approximation, and the size
    labels are Cohen's rules of thumb (0.2, 0.5, 0.8).

    Parameters
    ----------
    result : UnpairedEffectResult
        The result object returned by ``compute_unpaired_effect_size``.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    unit : str, default="units"
        What one unit is, e.g. ``"questions"``, ``"patients"`` or ``"matches"``.
    fmt : str, default=".3f"
        Format spec for the raw difference and its pooled standard deviation; ``".1%"`` prints
        proportions as percentages. The standardized values always use two decimals.

    Returns
    -------
    str
        A short plain-language reading of a ``compute_unpaired_effect_size`` result.
    """
    n_reference = result.nb_units_reference
    n_candidate = result.nb_units_candidate
    smallest = min(n_reference, n_candidate)

    def num(x: float) -> str:
        return format(x, fmt)

    return _interpret_effect_size(
        result,
        metric=metric,
        fmt=fmt,
        sizes=f"{n_reference} reference and {n_candidate} candidate {unit}",
        smallest=smallest,
        smallest_label=f"The smaller group has only {smallest} {unit}",
    )
