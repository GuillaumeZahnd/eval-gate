import math
from dataclasses import dataclass
from typing import Literal
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
class PairedEffectResult:
    """Result of a standardized effect size analysis for paired data.

    Attributes
    ----------
    cohens_d : float
        Standardized paired effect size (Cohen's ``d_z``): ``mean(candidate - reference) / std(candidate - reference)``.
        Positive means the candidate is higher than the reference. Inflated in small samples.
    hedges_g : float
        Small-sample bias-corrected effect size (Hedges' g). Prefer it to ``cohens_d`` as the headline
        estimate; the confidence interval refers to it.
    ci_lower : float
        Lower bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    ci_upper : float
        Upper bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    mean_difference : float
        The raw (unstandardized) effect ``mean(candidate - reference)``, in the units of the measurement.
    std_difference : float
        Standard deviation (``ddof=1``) of the per-unit paired differences, used as the standardizer.
    n_units : int
        Number of paired units analyzed.
    alpha : float
        Significance level of the confidence interval.
    """

    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    mean_difference: float
    std_difference: float
    n_units: int
    alpha: float

    @property
    def magnitude(self) -> EffectMagnitude:
        """Rough verbal label for ``|hedges_g|``, using Cohen's convention: negligible, small, medium or large."""
        return magnitude_label(self.hedges_g)


@dataclass(frozen=True)
class UnpairedEffectResult:
    """Result of a standardized effect size analysis for unpaired (independent groups) data.

    Attributes
    ----------
    cohens_d : float
        Standardized effect size (Cohen's ``d``): ``(mean(candidate) - mean(reference)) / pooled_std``.
        Positive means the candidate is higher than the reference. Inflated in small samples.
    hedges_g : float
        Small-sample bias-corrected effect size (Hedges' g) using the pooled standard deviation.
        Prefer it to ``cohens_d`` as the headline estimate; the confidence interval refers to it.
    ci_lower : float
        Lower bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    ci_upper : float
        Upper bound of the confidence interval for ``hedges_g`` at level ``1 - alpha``.
    mean_difference : float
        The raw (unstandardized) effect ``mean(candidate) - mean(reference)``, in the units of the measurement.
    pooled_std : float
        Pooled sample standard deviation across both groups, used as the standardizer.
    n_units_reference : int
        Number of units in the reference group.
    n_units_candidate : int
        Number of units in the candidate group.
    alpha : float
        Significance level of the confidence interval.
    """

    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    mean_difference: float
    pooled_std: float
    n_units_reference: int
    n_units_candidate: int
    alpha: float

    @property
    def magnitude(self) -> EffectMagnitude:
        """Rough verbal label for ``|hedges_g|``, using Cohen's convention: negligible, small, medium or large."""
        return magnitude_label(self.hedges_g)


def _hedges_correction(df: int) -> float:
    """Exact small-sample bias correction factor ``J(df)`` for a standardized mean difference."""
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1) / 2.0)) / math.sqrt(df / 2.0)


def _compute_effect_result(
    cohens_d: float,
    df_correction: int,
    se_d: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Compute Hedges' g, its standard error, and confidence bounds from Cohen's d and se_d.

    Parameters
    ----------
    cohens_d : float
        Standardized Cohen's d effect size.
    df_correction : int
        Degrees of freedom used for the small-sample bias correction (e.g., ``n_units - 1``
        for paired data or ``n_1 + n_2 - 2`` for unpaired data).
    se_d : float
        Approximate standard error of Cohen's d.
    alpha : float
        Significance level for the confidence interval, in ``(0, 1)``.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:
        - Hedges' g corrected effect size.
        - Lower bound of the confidence interval for Hedges' g.
        - Upper bound of the confidence interval for Hedges' g.
    """
    correction = _hedges_correction(df_correction)
    hedges_g = cohens_d * correction
    se_g = correction * se_d
    ci_lower, ci_upper = confidence_bounds("two_sided", hedges_g, se_g, alpha=alpha)
    return hedges_g, ci_lower, ci_upper


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
        Reference (baseline) measurements of shape ``(n_units,)`` or ``(n_units, nb_replicates)``.
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
    into ``g`` (Hedges and Olkin). The correction removes the bias of the point estimate, not the
    approximation error of the interval: for small ``n`` the interval is still approximate, and an
    exact interval based on the non-central t distribution is more accurate there.
    """
    check_in_open_interval_0_1(alpha, "alpha")

    differences = paired_differences(reference, candidate)
    check_all_finite(differences, "paired differences")

    n_units = differences.shape[0]
    check_min_units(n_units, "the effect size", minimum=_MIN_UNITS)

    mean_difference = float(differences.mean())

    std_difference = float(differences.std(ddof=1))
    check_finite_strictly_positive(std_difference, "std")

    # Cohen's d: the mean difference expressed in standard deviations of the differences, so effects
    # can be compared across metrics (0.5 means the average gain is half a typical per-unit spread).
    cohens_d = mean_difference / std_difference

    # Approximate standard error of Cohen's d: the 1/n term is the uncertainty on the mean difference,
    # the d**2 / (2n) term is the uncertainty on the standard deviation, which grows with the effect.
    se_d = math.sqrt(1.0 / n_units + cohens_d**2 / (2.0 * n_units))

    hedges_g, ci_lower, ci_upper = _compute_effect_result(
        cohens_d=cohens_d,
        df_correction=n_units - 1,
        se_d=se_d,
        alpha=alpha,
    )

    return PairedEffectResult(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_difference=mean_difference,
        std_difference=std_difference,
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
    multiplying d by a correction factor based on the total degrees of freedom (``n_units_reference + n_candidate - 2``).

    Report ``hedges_g`` as the headline value. The confidence interval is therefore given for it.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) measurements of shape ``(nb_reference,)`` or ``(nb_reference, k)``.
    candidate : np.ndarray
        Candidate measurements of shape ``(nb_candidate,)`` or ``(nb_candidate, k)``. The number of replicates
        may differ from ``reference``.
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
        If ``alpha`` is not in ``(0, 1)``, if either group has fewer than 2 units, if the total degrees
        of freedom is less than 1, if the data are not finite, or if all values across both groups are
        identical (the pooled standard deviation is then zero and the standardized effect is undefined).

    Notes
    -----
    The interval is centred on ``hedges_g`` and uses the large-sample normal approximation
    ``se(g) = J * sqrt(1/n_1 + 1/n_2 + d**2 / (2 * df))``, where ``J`` is the correction factor
    and ``df = n_1 + n_2 - 2``.
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

    df = n_units_reference + n_units_candidate - 2
    if df < 1:
        raise ValueError("At least 3 total units across both groups are required to compute the unpaired effect size.")

    mean_reference = float(reference_units.mean())
    mean_candidate = float(candidate_units.mean())
    mean_difference = mean_candidate - mean_reference

    sum_of_squares_reference = float(np.sum((reference_units - mean_reference) ** 2))
    sum_of_squares_candidate = float(np.sum((candidate_units - mean_candidate) ** 2))

    pooled_std = math.sqrt((sum_of_squares_reference + sum_of_squares_candidate) / df)
    check_finite_strictly_positive(pooled_std, "pooled std")

    cohens_d = mean_difference / pooled_std
    se_d = math.sqrt(1.0 / n_units_reference + 1.0 / n_units_candidate + cohens_d**2 / (2.0 * df))

    hedges_g, ci_lower, ci_upper = _compute_effect_result(
        cohens_d=cohens_d,
        df_correction=df,
        se_d=se_d,
        alpha=alpha,
    )

    return UnpairedEffectResult(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_difference=mean_difference,
        pooled_std=pooled_std,
        n_units_reference=n_units_reference,
        n_units_candidate=n_units_candidate,
        alpha=alpha,
    )
