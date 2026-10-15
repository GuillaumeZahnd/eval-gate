import math

import numpy as np
import pytest

from eval_gate.effect_size import PairedEffectResult, compute_paired_effect_size

Z_975 = 1.959963984540054  # norm.ppf(0.975)
J_DF4 = math.sqrt(2.0 / math.pi)  # exact Hedges correction for df=4 (n=5)
J_DF2 = 1.0 / math.sqrt(math.pi)  # exact Hedges correction for df=2 (n=3)

A = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
B = A + np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # differences 1..5: mean 3, std sqrt(2.5)
D = 3.0 / math.sqrt(2.5)


def make_result(effect: float) -> PairedEffectResult:
    return PairedEffectResult(
        cohens_d=effect,
        hedges_g=effect,
        ci_lower=effect - 1.0,
        ci_upper=effect + 1.0,
        mean_difference=1.0,
        std_difference=1.0,
        nb_units=10,
        alpha=0.05,
    )


class TestKnownValues:
    def test_cohens_d_and_its_components(self):
        result = compute_paired_effect_size(A, B)
        assert result.cohens_d == pytest.approx(D)
        assert result.mean_difference == pytest.approx(3.0)
        assert result.std_difference == pytest.approx(math.sqrt(2.5))
        assert result.nb_units == 5
        assert result.alpha == 0.05

    def test_hedges_g_applies_exact_correction(self):
        assert compute_paired_effect_size(A, B).hedges_g == pytest.approx(D * J_DF4)

    def test_three_units_case_with_hand_computed_values(self):
        # Differences 1, 2, 3: mean 2, std 1, so d = 2 and se(d) = sqrt(1/3 + 4/6) = 1.
        # g = J * d and se(g) = J * se(d), with J = 1 / sqrt(pi) for 2 degrees of freedom.
        result = compute_paired_effect_size(A[:3], B[:3])
        assert result.cohens_d == pytest.approx(2.0)
        assert result.hedges_g == pytest.approx(2.0 * J_DF2)
        assert result.ci_lower == pytest.approx(J_DF2 * (2.0 - Z_975))
        assert result.ci_upper == pytest.approx(J_DF2 * (2.0 + Z_975))

    def test_confidence_interval_is_centred_on_hedges_g(self):
        se_g = J_DF4 * math.sqrt(1 / 5 + D**2 / 10)
        g = D * J_DF4
        result = compute_paired_effect_size(A, B)
        assert result.ci_lower == pytest.approx(g - Z_975 * se_g)
        assert result.ci_upper == pytest.approx(g + Z_975 * se_g)

    def test_correction_shrinks_towards_zero_and_vanishes_for_large_n(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)
        b = a + rng.normal(loc=0.3, size=2000)
        result = compute_paired_effect_size(a, b)
        assert abs(result.hedges_g) < abs(result.cohens_d)
        assert result.hedges_g == pytest.approx(result.cohens_d, rel=1e-3)

    @pytest.mark.parametrize("nb_units, factor", [(5, 0.80), (10, 0.91), (20, 0.96), (100, 0.99)])
    def test_correction_factor_matches_the_values_quoted_in_the_docstring(self, nb_units, factor):
        candidate = np.arange(1.0, nb_units + 1.0)
        result = compute_paired_effect_size(np.zeros(nb_units), candidate)
        assert result.hedges_g / result.cohens_d == pytest.approx(factor, abs=0.005)


class TestProperties:
    def test_equals_mean_over_std_of_differences(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=30), rng.normal(loc=0.5, size=30)
        diffs = b - a
        result = compute_paired_effect_size(a, b)
        assert result.cohens_d == pytest.approx(diffs.mean() / diffs.std(ddof=1))

    def test_relation_with_paired_t_statistic(self):
        # d * sqrt(n) is the paired t statistic.
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=25), rng.normal(loc=0.4, size=25)
        diffs = b - a
        t = diffs.mean() / (diffs.std(ddof=1) / math.sqrt(25))
        assert compute_paired_effect_size(a, b).cohens_d * math.sqrt(25) == pytest.approx(t)

    def test_sign_follows_candidate_minus_reference(self):
        forward = compute_paired_effect_size(A, B)
        backward = compute_paired_effect_size(B, A)
        assert forward.cohens_d > 0 > backward.cohens_d
        assert backward.cohens_d == pytest.approx(-forward.cohens_d)
        assert backward.hedges_g == pytest.approx(-forward.hedges_g)
        assert backward.ci_lower == pytest.approx(-forward.ci_upper)
        assert backward.ci_upper == pytest.approx(-forward.ci_lower)

    @pytest.mark.parametrize("scale", [0.001, 3.0, 1e6])
    def test_invariant_to_rescaling_the_metric(self, scale):
        base = compute_paired_effect_size(A, B)
        scaled = compute_paired_effect_size(A * scale, B * scale)
        assert scaled.cohens_d == pytest.approx(base.cohens_d)
        assert scaled.hedges_g == pytest.approx(base.hedges_g)
        assert scaled.ci_lower == pytest.approx(base.ci_lower)
        assert scaled.mean_difference == pytest.approx(base.mean_difference * scale)
        assert scaled.std_difference == pytest.approx(base.std_difference * scale)

    def test_invariant_to_shifting_both_conditions(self):
        base = compute_paired_effect_size(A, B)
        shifted = compute_paired_effect_size(A + 100.0, B + 100.0)
        assert shifted.cohens_d == pytest.approx(base.cohens_d)

    def test_replicates_are_averaged_within_units(self):
        a2 = np.column_stack([A - 1.0, A + 1.0])  # unit means equal A
        b2 = np.column_stack([B - 2.0, B + 2.0])  # unit means equal B
        assert compute_paired_effect_size(a2, b2) == compute_paired_effect_size(A, B)

    def test_number_of_replicates_may_differ_between_conditions(self):
        b3 = np.column_stack([B - 1.0, B, B + 1.0])
        result = compute_paired_effect_size(A, b3)
        assert result.cohens_d == pytest.approx(compute_paired_effect_size(A, B).cohens_d)

    def test_interval_contains_hedges_g(self):
        result = compute_paired_effect_size(A, B)
        assert result.ci_lower < result.hedges_g < result.ci_upper

    def test_smaller_alpha_gives_wider_interval(self):
        narrow = compute_paired_effect_size(A, B, alpha=0.10)
        wide = compute_paired_effect_size(A, B, alpha=0.01)
        assert wide.ci_lower < narrow.ci_lower
        assert wide.ci_upper > narrow.ci_upper

    def test_interval_narrows_with_more_units(self):
        rng = np.random.default_rng(3)
        small_a, large_a = rng.normal(size=20), rng.normal(size=500)
        small = compute_paired_effect_size(small_a, small_a + rng.normal(loc=0.5, size=20))
        large = compute_paired_effect_size(large_a, large_a + rng.normal(loc=0.5, size=500))
        assert (large.ci_upper - large.ci_lower) < (small.ci_upper - small.ci_lower)


class TestMagnitude:
    @pytest.mark.parametrize(
        "effect, label",
        [
            (0.0, "negligible"),
            (0.19, "negligible"),
            (0.2, "small"),
            (0.49, "small"),
            (0.5, "medium"),
            (0.79, "medium"),
            (0.8, "large"),
            (2.5, "large"),
            (-0.1, "negligible"),  # the label ignores the sign
            (-0.6, "medium"),
            (-1.0, "large"),
        ],
    )
    def test_labels(self, effect, label):
        assert make_result(effect).magnitude == label

    def test_label_is_based_on_hedges_g_not_cohens_d(self):
        result = PairedEffectResult(
            cohens_d=0.55,  # would be "medium"
            hedges_g=0.45,  # is "small"
            ci_lower=0.0,
            ci_upper=1.0,
            mean_difference=1.0,
            std_difference=1.0,
            nb_units=5,
            alpha=0.05,
        )
        assert result.magnitude == "small"


class TestInputValidation:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            compute_paired_effect_size(A, B, alpha=alpha)

    def test_identical_differences_raise(self):
        with pytest.raises(ValueError, match="identical"):
            compute_paired_effect_size(A, A + 1.0)

    def test_identical_arrays_raise(self):
        with pytest.raises(ValueError, match="identical"):
            compute_paired_effect_size(A, A)

    @pytest.mark.parametrize("nb_units", [1, 2])
    def test_fewer_than_three_units_raise(self, nb_units):
        # With 2 units the Hedges correction factor is exactly 0, which would silently give g = 0.
        with pytest.raises(ValueError, match="At least 3 units"):
            compute_paired_effect_size(A[:nb_units], B[:nb_units])

    def test_different_number_of_units_raises(self):
        with pytest.raises(ValueError, match="Number of units differs"):
            compute_paired_effect_size(A, B[:4])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("condition", ["reference", "candidate"])
    def test_non_finite_data_raises_without_numpy_warnings(self, bad, condition, recwarn):
        reference, candidate = A.copy(), B.copy()
        (reference if condition == "reference" else candidate)[0] = bad
        with pytest.raises(ValueError, match="must contain only finite values"):
            compute_paired_effect_size(reference, candidate)
        assert len(recwarn) == 0  # no RuntimeWarning leaking out of NumPy

    def test_alpha_is_keyword_only(self):
        with pytest.raises(TypeError):
            compute_paired_effect_size(A, B, 0.05)
