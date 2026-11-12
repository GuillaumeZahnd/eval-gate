import math
from dataclasses import dataclass
from typing import Literal, NamedTuple
import numpy as np

from ._units import paired_differences, unit_means
from .confidence import confidence_bounds
from ._checks import (
    check_all_finite,
    check_finite_strictly_positive,
    check_min_units,
    check_in_open_interval_0_1
)


EffectMagnitude = Literal["negligible", "small", "medium", "large"]

# Upper bounds of each label, following Cohen's rough conventions for standardized mean differences.
_MAGNITUDE_THRESHOLDS: tuple[tuple[float, EffectMagnitude], ...] = (
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
)

# Hedges' correction needs df = n - 1 >= 2: for df = 1 the correction factor is exactly 0.
_MIN_UNITS = 3


def magnitude_label(size: float) -> EffectMagnitude:
    """Rough verbal label for a standardized effect: negligible, small, medium or large.

    Uses Cohen's conventions (0.2, 0.5 and 0.8) on ``abs(size)``. They are generic rules of thumb
    from the social sciences, not statements about practical importance: whether a given effect matters
    depends on the application, and should be judged in the raw units of ``mean_difference``.
    """
    size = abs(size)
    for upper, label in _MAGNITUDE_THRESHOLDS:
        if size < upper:
            return label
    return "large"


@dataclass(frozen=True)
class EffectResult:
    """Standardized effect size with its confidence interval (common part of paired and unpaired results).

    Attributes
    ----------
    cohens_d : float
        Standardized effect size (Cohen's ``d``): ``mean_difference`` divided by the standardizer
        (the standard deviation of the paired differences, or the pooled standard deviation).
        Positive means the candidate is higher than the reference. Inflated in small samples.
    hedges_g : float
        Small-sample bias-corrected effect size (Hedges' g). Prefer it to ``cohens_d`` as the headline
        estimate; the confidence interval refers to it.
    ci_lower : float
        Lower bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    ci_upper : float
        Upper bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    mean_difference : float
        The raw (unstandardized) effect ``mean(candidate) - mean(reference)``, in the units of the measurement.
    alpha : float
        Significance level of the confidence interval.
    """

    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    mean_difference: float
    alpha: float

    @property
    def magnitude(self) -> EffectMagnitude:
        """Rough verbal label for ``|hedges_g|``, using Cohen's convention: negligible, small, medium or large."""
        return magnitude_label(self.hedges_g)


@dataclass(frozen=True)
class PairedEffectResult(EffectResult):
    """Result of a standardized effect size analysis for paired data.

    Inherits the attributes of :class:`EffectResult`; here ``cohens_d`` is Cohen's ``d_z``,
    ``mean(candidate - reference) / std(candidate - reference)``.

    Attributes
    ----------
    std_difference : float
        Standard deviation (``ddof=1``) of the per-unit paired differences, used as the standardizer.
    n_units : int
        Number of paired units analyzed.
    """

    std_difference: float
    n_units: int


@dataclass(frozen=True)
class UnpairedEffectResult(EffectResult):
    """Result of a standardized effect size analysis for unpaired (independent groups) data.

    Inherits the attributes of :class:`EffectResult`; here the standardizer is the pooled standard deviation.

    Attributes
    ----------
    pooled_std : float
        Pooled sample standard deviation across both groups, used as the standardizer.
    n_units_reference : int
        Number of units in the reference group.
    n_units_candidate : int
        Number of units in the candidate group.
    """

    pooled_std: float
    n_units_reference: int
    n_units_candidate: int


def _hedges_correction(df: int) -> float:
    """Exact small-sample bias correction factor ``J(df)`` for a standardized mean difference."""
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1) / 2.0)) / math.sqrt(df / 2.0)


class _StandardizedEffect(NamedTuple):
    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    standardizer: float


def _standardized_effect(
    mean_difference: float,
    samples: tuple[np.ndarray, ...],
    alpha: float,
    std_name: str = "std",
) -> _StandardizedEffect:
    """Standardize a mean difference and compute Hedges' g with its confidence interval.

    Covers one sample of paired differences, and two independent groups.

    Parameters
    ----------
    mean_difference : float
        The raw mean difference, ``mean(candidate - reference)``.
    samples : tuple[np.ndarray, ...]
        One 1D array (the paired differences) or two (reference and candidate units).
    alpha : float
        Significance level of the confidence interval, in ``(0, 1)``.
    std_name : str, default="std"
        Name of the standardizer in the error message when it is zero.

    Returns
    -------
    _StandardizedEffect
        Cohen's d, Hedges' g, the interval for g, and the standardizer (the standard deviation of the
        differences, or the pooled standard deviation).

    Raises
    ------
    ValueError
        If every sample is constant, so the standardizer is zero.
    """
    sizes = [s.shape[0] for s in samples]
    n_total = sum(sizes)
    df = n_total - len(samples)

    # Exactly 0 when every sample is constant: a floating-point mean would leave a ~1e-17 residual.
    if all(np.ptp(s) == 0 for s in samples):
        standardizer = 0.0
    else:
        standardizer = math.sqrt(sum(float(np.sum((s - s.mean()) ** 2)) for s in samples) / df)
    check_finite_strictly_positive(standardizer, std_name)

    # Cohen's d: the mean difference expressed in standard deviations, so effects can be compared across metrics.
    cohens_d = mean_difference / standardizer

    # Approximate standard error of d: the first term is the uncertainty on the mean difference, the second the
    # uncertainty on the standard deviation, which grows with the effect.
    se_d = math.sqrt(sum(1.0 / n for n in sizes) + cohens_d**2 / (2.0 * n_total))

    correction = _hedges_correction(df)
    hedges_g = cohens_d * correction
    ci_lower, ci_upper = confidence_bounds("two_sided", hedges_g, correction * se_d, alpha=alpha)

    return _StandardizedEffect(cohens_d, hedges_g, ci_lower, ci_upper, standardizer)


def compute_paired_effect_size(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    alpha: float = 0.05,
) -> PairedEffectResult:
    """Compute standardized effect sizes (Cohen's d and Hedges' g) for paired measurements.

    The effect size measures the magnitude of the mean paired difference relative to
    the variability of the differences: ``mean(candidate - reference) / std(candidate - reference)``.
    Because it divides by the spread of the differences rather than the spread of the measurements,
    strongly correlated pairs give larger values than the usual unpaired Cohen's d, so do not compare
    the two directly.

    Cohen's d overestimates the true effect in small samples. Hedges' g removes that bias by
    multiplying d by a correction factor below 1 (about 0.80 with 5 units, 0.91 with 10, 0.96 with
    20 and 0.99 with 100):

    - For large samples (about 20 units or more), the factor is close to 1, so ``cohens_d`` and
      ``hedges_g`` are nearly identical.
    - For small samples (fewer than about 20 units), the factor is meaningfully below 1:
      ``hedges_g`` is the more accurate, unbiased estimate of the true effect, while ``cohens_d``
      remains inflated.

    Report ``hedges_g`` as the headline value. The confidence interval is therefore given for it.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements of shape ``(n_units,)`` or ``(n_units, n_replicates)``.
    candidate : np.ndarray
        Candidate measurements, same number of units as ``reference``. The number of replicates may differ.
    alpha : float, default=0.05
        Significance level for the confidence interval, in ``(0, 1)``.

    Returns
    -------
    PairedEffectResult
        Cohen's d, Hedges' g, the confidence interval for Hedges' g, the raw mean difference and
        the standard deviation behind them, and the analysis parameters.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if the inputs have a different number of units or fewer
        than 3 units, if the data are not finite, or if all paired differences are identical
        (the standardized effect is then undefined).

    Notes
    -----
    The interval is centred on ``hedges_g`` and uses the large-sample normal approximation
    ``se(g) = J * sqrt(1/n + d**2 / (2n))``, where ``J`` is the correction factor that turns ``d``
    into ``g`` and ``n`` is the number of units (Hedges and Olkin). The correction removes the bias of
    the point estimate, not the approximation error of the interval: for small ``n`` the interval is
    still approximate, and an exact interval based on the non-central t distribution is more accurate there.
    """
    check_in_open_interval_0_1(alpha, "alpha")

    differences = paired_differences(reference, candidate)
    check_all_finite(differences, "paired differences")

    n_units = differences.shape[0]
    check_min_units(n_units, "the effect size", minimum=_MIN_UNITS)

    mean_difference = float(differences.mean())
    effect = _standardized_effect(mean_difference, (differences,), alpha, "std of paired differences")

    return PairedEffectResult(
        cohens_d=effect.cohens_d,
        hedges_g=effect.hedges_g,
        ci_lower=effect.ci_lower,
        ci_upper=effect.ci_upper,
        mean_difference=mean_difference,
        std_difference=effect.standardizer,
        n_units=n_units,
        alpha=alpha,
    )


def compute_unpaired_effect_size(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    alpha: float = 0.05,
) -> UnpairedEffectResult:
    """Compute standardized effect sizes (Cohen's d and Hedges' g) for independent groups (unpaired measurements).

    The effect size measures the magnitude of the difference between two independent group means relative to
    the pooled sample standard deviation: ``(mean(candidate) - mean(reference)) / pooled_std``.

    Cohen's d overestimates the true effect in small samples. Hedges' g removes that bias by
    multiplying d by a correction factor based on the total degrees of freedom (``n_units_reference + n_units_candidate - 2``).

    Report ``hedges_g`` as the headline value. The confidence interval is therefore given for it.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements of shape ``(n_units_reference,)`` or ``(n_units_reference, n_replicates)``.
    candidate : np.ndarray
        Candidate measurements of shape ``(n_units_candidate,)`` or ``(n_units_candidate, n_replicates)``.
        The number of replicates may differ from ``reference``.
    alpha : float, default=0.05
        Significance level for the confidence interval, in ``(0, 1)``.

    Returns
    -------
    UnpairedEffectResult
        Cohen's d, Hedges' g, the confidence interval for Hedges' g, the raw mean difference,
        the pooled standard deviation, group sizes, and the analysis parameters.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if either group has fewer than 2 units, if the data are not
        finite, or if both groups are constant (the pooled standard deviation is then zero and the
        standardized effect is undefined).

    Notes
    -----
    The interval is centred on ``hedges_g`` and uses the large-sample normal approximation
    ``se(g) = J * sqrt(1/n_1 + 1/n_2 + d**2 / (2N))``, where ``J`` is the correction factor,
    ``N = n_1 + n_2`` is the total number of units, and ``df = N - 2`` (Hedges and Olkin).
    For small groups the interval is still approximate.
    """
    check_in_open_interval_0_1(alpha, "alpha")

    reference_units = unit_means(reference)
    candidate_units = unit_means(candidate)
    check_all_finite(reference_units, "reference measurements")
    check_all_finite(candidate_units, "candidate measurements")

    n_units_reference = reference_units.shape[0]
    n_units_candidate = candidate_units.shape[0]
    check_min_units(n_units_reference, "the reference group for the effect size", minimum=2)
    check_min_units(n_units_candidate, "the candidate group for the effect size", minimum=2)

    mean_difference = float(candidate_units.mean() - reference_units.mean())
    effect = _standardized_effect(mean_difference, (reference_units, candidate_units), alpha, "pooled std")

    return UnpairedEffectResult(
        cohens_d=effect.cohens_d,
        hedges_g=effect.hedges_g,
        ci_lower=effect.ci_lower,
        ci_upper=effect.ci_upper,
        mean_difference=mean_difference,
        pooled_std=effect.standardizer,
        n_units_reference=n_units_reference,
        n_units_candidate=n_units_candidate,
        alpha=alpha,
    )
