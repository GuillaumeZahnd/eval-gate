from .sem import SEMResult
from .core import ALTERNATIVE_HYPOTHESES


SMALL_N = 30  # Below this many units, the normal approximation can be noticeably optimistic


def _pct(x: float) -> str:
    return f"{round(x * 100, 6):g}%"  # 0.95 -> "95%", 0.999 -> "99.9%"


def interpret_sem(
    result: SEMResult,
    *,
    metric: str = "mean outcome",
    unit: str = "units",
    fmt: str = ".3f",
) -> str:
    """One-paragraph reading of a `compute_sem` result.

    Parameters
    ----------
    result : SEMResult
        The result object from `compute_sem`.
    metric : str, default="mean outcome"
        What the mean measures, e.g. "mean accuracy" or "mean systolic pressure".
    unit : str, default="units"
        What one unit is, e.g. "questions", "patients", "matches".
    fmt : str, default=".3f"
        Format spec for numbers; ".1%" prints proportions as percentages.

    Returns
    -------
    str
        A clear, one-paragraph interpretive summary.
    """
    num = lambda x: format(x, fmt)
    level = _pct(result.confidence)
    n = result.nb_units
    alternative_hypothesis = result.alternative_hypothesis
    ci_lower = num(result.ci_lower)
    ci_upper = num(result.ci_upper)

    head = f"The {metric} is estimated at {num(result.mean)} (SEM {num(result.sem)}, {n} {unit})."

    if alternative_hypothesis == "two_sided":
        body = (f"With {level} confidence, the true {metric} lies between {ci_lower} and {ci_upper}.")

    elif alternative_hypothesis == "greater":
        body = f"With {level} confidence, the true {metric} is at least {ci_lower}."

    elif alternative_hypothesis == "less":
        body = f"With {level} confidence, the true {metric} is at most {ci_upper}."

    else:
        raise ValueError(
            f"'alternative_hypothesis' must be one of {ALTERNATIVE_HYPOTHESES}, got '{alternative_hypothesis}'."
        )

    notes = []
    if result.sem == 0:
        notes.append(
            f"All {unit} gave the same outcome, so the interval has zero width "
            "(this reflects the sample, not certainty about the population)."
        )
    if n < SMALL_N:
        notes.append(
            f"Only {n} {unit} (less than {SMALL_N}: the normal approximation can make the interval somewhat too narrow."
        )
    return " ".join([head, body, *notes])
