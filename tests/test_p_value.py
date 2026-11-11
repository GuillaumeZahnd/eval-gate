import math
import numpy as np
import pytest
from scipy.stats import norm

from eval_gate.p_value import (
    PairedPValueResult,
    UnpairedPValueResult,
    _normal_p_value,
    paired_p_value,
    unpaired_p_value,
)


Z_95 = float(norm.ppf(0.95))
Z_975 = float(norm.ppf(0.975))

# Paired data: the differences are 1..5, so their mean is 3 and the standard error is sqrt(2.5 / 5) = sqrt(0.5).
PAIRED_REFERENCE = np.zeros(5)
PAIRED_CANDIDATE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
PAIRED_ESTIMATE = 3.0
PAIRED_SE = math.sqrt(0.5)

# Unpaired data: reference has mean 3, variance 2.5, n = 5; candidate has mean 5, variance 20 / 3, n = 4.
# The Welch standard error is sqrt(2.5 / 5 + (20 / 3) / 4) = sqrt(13 / 6); a pooled one would be 1.389.
UNPAIRED_REFERENCE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
UNPAIRED_CANDIDATE = np.array([2.0, 4.0, 6.0, 8.0])
UNPAIRED_ESTIMATE = 2.0
UNPAIRED_SE = math.sqrt(13 / 6)

# (alternative, margin, true difference): each sits exactly on the boundary of its null hypothesis.
NULL_BOUNDARIES = [
    ("two_sided", 0.0, 0.0),
    ("superiority", 0.0, 0.0),
    ("non_inferiority", 0.5, -0.5),
    ("equivalence", 0.5, 0.5),
]
NB_EXPERIMENTS = 2000


class TestNormalPValue:
    def test_two_sided_at_the_critical_value(self):
        assert _normal_p_value("two_sided", Z_975, 1.0, 0.0) == pytest.approx(0.05)

    def test_two_sided_is_symmetric(self):
        assert _normal_p_value("two_sided", 1.3, 1.0, 0.0) == pytest.approx(_normal_p_value("two_sided", -1.3, 1.0, 0.0))

    def test_two_sided_zero_estimate_gives_one(self):
        assert _normal_p_value("two_sided", 0.0, 1.0, 0.0) == 1.0

    def test_greater_at_the_critical_value(self):
        assert _normal_p_value("greater", Z_95, 1.0, 0.0) == pytest.approx(0.05)

    def test_greater_is_one_sided(self):
        assert _normal_p_value("greater", -Z_95, 1.0, 0.0) == pytest.approx(0.95)

    def test_greater_margin_shifts_the_null(self):
        # The null is "difference <= -margin", so an estimate of 0 sits Z_95 standard errors above it.
        assert _normal_p_value("greater", 0.0, 2.0, 2.0 * Z_95) == pytest.approx(0.05)

    def test_equivalence_takes_the_larger_one_sided_p_value(self):
        # Estimate 0.5, margin 1: the p-values are sf(1.5) = 0.067 and cdf(-0.5) = 0.309.
        assert _normal_p_value("equivalence", 0.5, 1.0, 1.0) == pytest.approx(norm.cdf(-0.5))

    def test_equivalence_is_symmetric(self):
        assert _normal_p_value("equivalence", 0.5, 1.0, 1.0) == pytest.approx(_normal_p_value("equivalence", -0.5, 1.0, 1.0))

    def test_equivalence_at_zero_estimate(self):
        assert _normal_p_value("equivalence", 0.0, 1.0, Z_95) == pytest.approx(0.05)

    def test_only_the_ratio_estimate_over_se_matters(self):
        assert _normal_p_value("two_sided", 2.0, 2.0, 0.0) == pytest.approx(_normal_p_value("two_sided", 1.0, 1.0, 0.0))

    def test_p_value_decreases_as_the_estimate_grows(self):
        values = [_normal_p_value("greater", estimate, 1.0, 0.0) for estimate in (-1.0, 0.0, 1.0, 2.0)]
        assert values == sorted(values, reverse=True)


class TestPairedPValue:
    def test_estimate_se_and_nb_units(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE)
        assert result.estimate == pytest.approx(PAIRED_ESTIMATE)
        assert result.se == pytest.approx(PAIRED_SE)
        assert result.nb_units == 5

    def test_two_sided_p_value(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE)
        assert result.p_value == pytest.approx(2 * norm.sf(PAIRED_ESTIMATE / PAIRED_SE))

    def test_superiority_p_value(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "superiority")
        assert result.p_value == pytest.approx(norm.sf(PAIRED_ESTIMATE / PAIRED_SE))

    def test_non_inferiority_p_value(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "non_inferiority", margin=1.0)
        assert result.p_value == pytest.approx(norm.sf((PAIRED_ESTIMATE + 1.0) / PAIRED_SE))
        assert result.margin == 1.0

    def test_equivalence_rejected_when_the_margin_is_wide(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "equivalence", margin=5.0)
        assert result.p_value == pytest.approx(norm.cdf((PAIRED_ESTIMATE - 5.0) / PAIRED_SE))
        assert result.reject

    def test_equivalence_not_rejected_when_the_margin_is_narrow(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "equivalence", margin=2.0)
        assert result.p_value == pytest.approx(norm.cdf((PAIRED_ESTIMATE - 2.0) / PAIRED_SE))
        assert not result.reject

    def test_difference_is_candidate_minus_reference(self):
        flipped = paired_p_value(PAIRED_CANDIDATE, PAIRED_REFERENCE)
        assert flipped.estimate == pytest.approx(-PAIRED_ESTIMATE)
        assert flipped.p_value == pytest.approx(paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE).p_value)

    def test_superiority_p_value_is_large_when_the_candidate_is_lower(self):
        result = paired_p_value(PAIRED_CANDIDATE, PAIRED_REFERENCE, "superiority")
        assert result.p_value > 0.99

    def test_margin_is_ignored_without_a_margin_hypothesis(self):
        with_margin = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "superiority", margin=5.0)
        without = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "superiority")
        assert with_margin.margin == 0.0
        assert with_margin.p_value == without.p_value

    def test_alpha_does_not_change_the_p_value(self):
        strict = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, alpha=0.001)
        lenient = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, alpha=0.2)
        assert strict.p_value == lenient.p_value
        assert strict.alpha == 0.001
        assert lenient.alpha == 0.2

    def test_result_echoes_the_inputs(self):
        result = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE, "non_inferiority", margin=1.0, alpha=0.1)
        assert result.alternative_hypothesis == "non_inferiority"
        assert result.alpha == 0.1

    def test_replicates_are_averaged_within_units(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(30, 4))
        candidate = rng.normal(loc=0.3, size=(30, 2))  # the number of replicates may differ
        assert paired_p_value(reference, candidate) == paired_p_value(reference.mean(axis=1), candidate.mean(axis=1))

    def test_invariant_to_a_common_shift_and_rescaling(self):
        base = paired_p_value(PAIRED_REFERENCE, PAIRED_CANDIDATE)
        shifted = paired_p_value(PAIRED_REFERENCE + 100.0, PAIRED_CANDIDATE + 100.0)
        rescaled = paired_p_value(PAIRED_REFERENCE * 10.0, PAIRED_CANDIDATE * 10.0)
        assert shifted.p_value == pytest.approx(base.p_value)
        assert rescaled.p_value == pytest.approx(base.p_value)


class TestUnpairedPValue:
    def test_estimate_se_and_group_sizes(self):
        result = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE)
        assert result.estimate == pytest.approx(UNPAIRED_ESTIMATE)
        assert result.se == pytest.approx(UNPAIRED_SE)
        assert result.nb_units_reference == 5
        assert result.nb_units_candidate == 4

    def test_standard_error_is_welch_not_pooled(self):
        pooled_variance = (4 * 2.5 + 3 * (20 / 3)) / 7
        pooled_se = math.sqrt(pooled_variance * (1 / 5 + 1 / 4))
        result = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE)
        assert result.se != pytest.approx(pooled_se)

    @pytest.mark.parametrize(
        ("alternative", "margin", "expected"),
        [
            ("two_sided", 0.0, 2 * norm.sf(UNPAIRED_ESTIMATE / UNPAIRED_SE)),
            ("superiority", 0.0, norm.sf(UNPAIRED_ESTIMATE / UNPAIRED_SE)),
            ("non_inferiority", 1.0, norm.sf((UNPAIRED_ESTIMATE + 1.0) / UNPAIRED_SE)),
            ("equivalence", 5.0, norm.cdf((UNPAIRED_ESTIMATE - 5.0) / UNPAIRED_SE)),
        ],
    )
    def test_p_value_for_each_alternative(self, alternative, margin, expected):
        result = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE, alternative, margin=margin)
        assert result.p_value == pytest.approx(expected)

    def test_difference_is_candidate_minus_reference(self):
        flipped = unpaired_p_value(UNPAIRED_CANDIDATE, UNPAIRED_REFERENCE)
        original = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE)
        assert flipped.estimate == pytest.approx(-UNPAIRED_ESTIMATE)
        assert flipped.se == pytest.approx(original.se)
        assert flipped.p_value == pytest.approx(original.p_value)

    def test_one_group_may_be_constant(self):
        result = unpaired_p_value(np.full(6, 3.0), UNPAIRED_CANDIDATE)
        assert result.se > 0.0

    def test_groups_may_have_equal_sizes(self):
        result = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_REFERENCE + 1.0)
        assert result.nb_units_reference == result.nb_units_candidate == 5

    def test_replicates_are_averaged_within_units(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(40, 3))
        candidate = rng.normal(loc=0.5, size=(25, 5))  # different units and different replicates
        assert unpaired_p_value(reference, candidate) == unpaired_p_value(
            reference.mean(axis=1), candidate.mean(axis=1)
        )

    def test_margin_is_ignored_without_a_margin_hypothesis(self):
        result = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE, "superiority", margin=5.0)
        assert result.margin == 0.0

    def test_alpha_does_not_change_the_p_value(self):
        strict = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE, alpha=0.001)
        lenient = unpaired_p_value(UNPAIRED_REFERENCE, UNPAIRED_CANDIDATE, alpha=0.2)
        assert strict.p_value == lenient.p_value


class TestResultDecision:
    @staticmethod
    def _paired(p_value: float, alpha: float) -> PairedPValueResult:
        return PairedPValueResult(
            p_value=p_value,
            alternative_hypothesis="two_sided",
            margin=0.0,
            estimate=1.0,
            se=1.0,
            nb_units=10,
            alpha=alpha,
        )

    @staticmethod
    def _unpaired(p_value: float, alpha: float) -> UnpairedPValueResult:
        return UnpairedPValueResult(
            p_value=p_value,
            alternative_hypothesis="two_sided",
            margin=0.0,
            estimate=1.0,
            se=1.0,
            nb_units_reference=10,
            nb_units_candidate=12,
            alpha=alpha,
        )

    @pytest.mark.parametrize("build", ["_paired", "_unpaired"])
    def test_reject_is_a_strict_comparison(self, build):
        make = getattr(self, build)
        assert make(0.03, 0.05).reject
        assert not make(0.03, 0.01).reject
        assert not make(0.05, 0.05).reject


class TestTypeIErrorRate:
    @pytest.mark.parametrize(("alternative", "margin", "true_difference"), NULL_BOUNDARIES)
    def test_paired_rejection_rate_on_the_null_boundary_is_alpha(self, alternative, margin, true_difference):
        rng = np.random.default_rng(0)
        rejected = 0
        for _ in range(NB_EXPERIMENTS):
            reference = rng.normal(size=100)
            candidate = reference + rng.normal(true_difference, 1.0, size=100)
            rejected += paired_p_value(reference, candidate, alternative, margin=margin).reject
        assert rejected / NB_EXPERIMENTS == pytest.approx(0.05, abs=0.02)

    @pytest.mark.parametrize(("alternative", "margin", "true_difference"), NULL_BOUNDARIES)
    def test_unpaired_rejection_rate_on_the_null_boundary_is_alpha(self, alternative, margin, true_difference):
        # Unequal sizes and unequal variances: the Welch standard error must keep the rate at alpha.
        rng = np.random.default_rng(0)
        rejected = 0
        for _ in range(NB_EXPERIMENTS):
            reference = rng.normal(0.0, 1.0, size=150)
            candidate = rng.normal(true_difference, 2.0, size=100)
            rejected += unpaired_p_value(reference, candidate, alternative, margin=margin).reject
        assert rejected / NB_EXPERIMENTS == pytest.approx(0.05, abs=0.02)


class TestInputValidation:
    @pytest.mark.parametrize("function", [paired_p_value, unpaired_p_value])
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_alpha_outside_open_unit_interval_raises(self, function, alpha):
        with pytest.raises(ValueError, match="'alpha'"):
            function(PAIRED_REFERENCE, PAIRED_CANDIDATE, alpha=alpha)

    @pytest.mark.parametrize("function", [paired_p_value, unpaired_p_value])
    @pytest.mark.parametrize("alternative", ["non_inferiority", "equivalence"])
    @pytest.mark.parametrize("margin", [0.0, -1.0])
    def test_margin_must_be_positive_when_required(self, function, alternative, margin):
        with pytest.raises(ValueError, match="margin"):
            function(PAIRED_REFERENCE, PAIRED_CANDIDATE, alternative, margin=margin)

    def test_paired_units_must_match(self):
        with pytest.raises(ValueError, match="Number of units differs"):
            paired_p_value(np.zeros(5), np.ones(6))

    def test_paired_identical_differences_raise(self):
        with pytest.raises(ValueError):
            paired_p_value(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))

    def test_unpaired_identical_values_in_both_groups_raise(self):
        with pytest.raises(ValueError):
            unpaired_p_value(np.full(5, 1.0), np.full(4, 2.0))

    @pytest.mark.parametrize("function", [paired_p_value, unpaired_p_value])
    def test_fewer_than_two_units_raises(self, function):
        with pytest.raises(ValueError, match="At least 2 units"):
            function(np.array([1.0]), np.array([2.0]))

    @pytest.mark.parametrize("function", [paired_p_value, unpaired_p_value])
    def test_non_finite_values_raise(self, function):
        candidate = PAIRED_CANDIDATE.copy()
        candidate[1] = np.nan
        with pytest.raises(ValueError):
            function(PAIRED_REFERENCE, candidate)

    def test_unpaired_group_with_one_unit_raises(self):
        with pytest.raises(ValueError, match="At least 2 units"):
            unpaired_p_value(UNPAIRED_REFERENCE, np.array([1.0]))
