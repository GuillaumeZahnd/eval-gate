from .sem import SEMResult
from .p_value import PValueResult


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

    head = f"The {metric} is estimated at {num(result.mean)} (SEM {num(result.sem)}, {n} {unit})."

    if alternative_hypothesis == "two_sided":
        body = f"With {level} confidence, the true {metric} lies between {ci_lower} and {ci_upper}."
    elif alternative_hypothesis == "superiority":
        body = f"With {level} confidence, the true {metric} is at least {ci_lower}."

    notes = []
    if result.sem == 0:
        notes.append(
            f"All {unit} gave the same outcome, so the interval has zero width "
            "(this reflects the sample, not certainty about the population)."
        )
    if n < SMALL_N:
        notes.append(
            f"Only {n} {unit} (less than {SMALL_N}): the normal approximation can make the interval somewhat too narrow."
        )
    return " ".join([head, body, *notes])


def interpret_p_value(
    result: PValueResult,
    *,
    metric: str = "mean outcome",
    fmt: str = ".3f",
) -> str:
    """Return a one-paragraph plain-language reading of a ``paired_p_value`` result.

    The paragraph opens with the p-value and the decision at the result's significance
    level, then explains what the p-value means under the null hypothesis, what can
    (and cannot) be concluded, the observed difference, and a caveat when the sample
    is small.

    Parameters
    ----------
    result : PValueResult
        The result object returned by ``paired_p_value``. It carries the hypothesis,
        the margin, the significance level, the estimate and the number of units
        used in the test.
    metric : str, default="mean outcome"
        What is being compared, e.g. ``"mean accuracy"``.
    fmt : str, default=".3f"
        Format spec for the margin, estimate and standard error; ``".1%"`` prints
        proportions as percentages.

    Returns
    -------
    str
        A one-paragraph interpretive summary.
    """
    m = format(result.margin, fmt)
    significant = result.reject

    match result.alternative_hypothesis:
        case "two_sided":
            null = "there were no difference between the candidate and the reference"
            direction = "higher" if result.estimate > 0 else "lower"
            if_rejected = f"The candidate's {metric} differs from the reference's: it is {direction}."
            if_not = (
                "A difference cannot be established. This does not show that the two are equal, "
                "only that the data do not rule out no difference."
            )
        case "superiority":
            null = "the candidate were no better than the reference"
            if_rejected = f"There is evidence that the candidate's {metric} is higher than the reference's."
            if_not = (
                "Superiority is not established. The candidate may still be better, "
                "but the data are not strong enough to say so."
            )
        case "non_inferiority":
            null = f"the candidate were worse than the reference by {m} or more"
            if_rejected = (
                f"A drop of {m} or more is ruled out. The candidate may be equal or better; "
                "this test does not distinguish between the two."
            )
            if_not = (
                f"Non-inferiority is not established: a drop of {m} or more cannot be ruled out. "
                "This does not show that the candidate is worse."
            )
        case "equivalence":
            null = f"the true difference were {m} or more in either direction"
            if_rejected = f"The difference between the candidate and the reference is within ±{m}."
            if_not = (
                f"Equivalence is not established: a difference of {m} or more in at least one "
                "direction cannot be ruled out."
            )

    head = (
        f"{_fmt_p(result.p_value)} (significance level {result.alpha:g}): "
        f"the null hypothesis is {'rejected' if significant else 'not rejected'}."
    )
    meaning = (
        f"This p-value is the probability, if {null}, of a result at least as favourable "
        "to the alternative as the one observed."
    )
    conclusion = if_rejected if significant else if_not

    notes = [
        f"The observed difference (candidate - reference) is {format(result.estimate, fmt)} "
        f"(SE {format(result.se, fmt)}, {result.nb_units} units)."
    ]
    if significant:
        notes.append("A small p-value shows that a difference is detectable, not that it is large.")
    if result.nb_units < SMALL_N:
        notes.append(
            f"Only {result.nb_units} units (less than {SMALL_N}): "
            "the normal approximation can make the p-value somewhat too small."
        )
    return " ".join([head, meaning, conclusion, *notes])
