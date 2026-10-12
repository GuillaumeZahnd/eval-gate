import math
from dataclasses import dataclass
import numpy as np

from .core import AlternativeHypothesis, confidence_bounds


@dataclass(frozen=True)
class SEMResult:
    mean: float  # Mean of the per-unit outcomes.
    sem: float  # Standard error of the mean, ``std(ddof=1) / sqrt(nb_units)``.
    ci_lower: float  # Lower bound of the condifence interval (infinite when alternative_hypothesis is ``"less"``).
    ci_upper: float  # Upper bound of the condifence interval (infinite when alternative_hypothesis is ``"greater"``).
    confidence: float  # Confidence level.
    alternative_hypothesis: AlternativeHypothesis  # Sideness (``two_sided``, ``greater``, or ``less``).
    nb_units: int  # Number of units.


def compute_sem(
    outcomes: np.ndarray,
    confidence: float = 0.95,
    alternative_hypothesis: AlternativeHypothesis = "two_sided",
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
        alternative_hypothesis : {"two_sided", "greater", "less"}, default="two_sided"
            Sidedness of the interval. The SEM itself is unaffected.
            - ``"two_sided"``: symmetric interval at level ``confidence``.
              Use it to report an estimate with error bars.
            - ``"greater"``: one-sided lower bound, ``ci_upper`` is ``+inf``.
              Use it for a claim of the form "the true mean is at least x".
            - ``"less"``: one-sided upper bound, ``ci_lower`` is ``-inf``.
               Use it for a claim of the form "the true mean is at most x".

    Returns:
        SEMResult dictionary.

    Raises:
        ValueError
            If ``confidence`` is not in ``(0, 1)``, if ``alternative_hypothesis`` is unknown,
            if ``outcomes`` is not 1D or 2D, or if it has fewer than 2 units.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    outcomes = np.asarray(outcomes, dtype=np.float64)

    nb_units = outcomes.shape[0]
    if nb_units < 2:
        raise ValueError(f"At least 2 units are required to compute the standard error of the mean, got {nb_units}.")

    # Collapse replicates to one outcome per unit (M, K) -> (M,)
    if outcomes.ndim == 2:
        unit_means = outcomes.mean(axis=1)
    elif outcomes.ndim == 1:
        unit_means = outcomes
    else:
        raise ValueError(f"Expected outcomes to be a 1D or 2D array, got shape {outcomes.shape}.")

    mean = float(np.mean(unit_means))
    std = float(np.std(unit_means, ddof=1))
    sem = std / math.sqrt(nb_units)
    alpha = 1.0 - confidence

    ci_lower, ci_upper = confidence_bounds(
        alternative_hypothesis=alternative_hypothesis,
        estimate=mean,
        se=sem,
        alpha=alpha
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
