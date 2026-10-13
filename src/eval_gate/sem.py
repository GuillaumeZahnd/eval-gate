from dataclasses import dataclass
from typing import Literal, get_args
import numpy as np

from .confidence import confidence_bounds
from .utils import mean_and_se

# Intervals that make sense for a plain mean: there is no reference and no margin.
SEMAlternative = Literal["two_sided", "superiority"]
SEM_ALTERNATIVES: tuple[str, ...] = get_args(SEMAlternative)


@dataclass(frozen=True)
class SEMResult:
    mean: float  # Mean of the per-unit outcomes.
    sem: float  # Standard error of the mean, ``std(ddof=1) / sqrt(nb_units)``.
    ci_lower: float  # Lower bound of the confidence interval.
    ci_upper: float  # Upper bound of the confidence interval (``+inf`` when alternative_hypothesis is ``"superiority"``).
    confidence: float  # Confidence level.
    alternative_hypothesis: SEMAlternative  # Sidedness of the interval (``"two_sided"`` or ``"superiority"``).
    nb_units: int  # Number of units.


def compute_sem(
    outcomes: np.ndarray,
    confidence: float = 0.95,
    alternative_hypothesis: SEMAlternative = "two_sided",
) -> SEMResult:
    """
    Standard error of the mean (SEM) and a confidence interval for the mean.

    Args:
        outcomes : np.ndarray
            Measured outcomes, one entry per unit. Either:
            - shape ``(nb_units,)``: one outcome per unit.
            - shape ``(nb_units, nb_replicates_per_unit)``:  In this case, replicates are first averaged within each
              unit so the SEM is clustered at the unit level, then these unit means are treated as independent draws.
        confidence : float, default=0.95
            Confidence level of the interval, strictly between 0 and 1.
        alternative_hypothesis : {"two_sided", "superiority"}, default="two_sided"
            Sidedness of the interval. The SEM itself is unaffected.
            - ``"two_sided"``: symmetric interval at level ``confidence``.
              Use it to report an estimate with error bars.
            - ``"superiority"``: one-sided lower bound, ``ci_upper`` is ``+inf``.
              Use it for a claim of the form "the true mean is at least x".
            ``"non_inferiority"`` and ``"equivalence"`` are not accepted: they are defined
            relative to a reference and a margin, which a single mean does not have.

    Returns:
        SEMResult

    Raises:
        ValueError
            If ``confidence`` is not in ``(0, 1)``, if ``alternative_hypothesis`` is not
            ``"two_sided"`` or ``"superiority"``, if ``outcomes`` is not 1D or 2D, or if it
            has fewer than 2 units.
    """
    if alternative_hypothesis not in SEM_ALTERNATIVES:
        raise ValueError(
            f"'alternative_hypothesis' must be one of {SEM_ALTERNATIVES} for a single mean, "
            f"got {alternative_hypothesis!r}."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    mean, sem, nb_units = mean_and_se(outcomes)

    ci_lower, ci_upper = confidence_bounds(
        alternative_hypothesis=alternative_hypothesis,
        estimate=mean,
        se=sem,
        alpha=1.0 - confidence,
    )

    return SEMResult(
        mean=mean,
        sem=sem,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        alternative_hypothesis=alternative_hypothesis,
        nb_units=nb_units,
    )
