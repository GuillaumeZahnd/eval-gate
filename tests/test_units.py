import math
import re

import numpy as np
import pytest

from eval_gate._units import (
    MeanSE,
    UnpairedDifference,
    mean_and_se,
    paired_differences,
    paired_std,
    unit_means,
    unpaired_difference,
    unpaired_std,
)

A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # mean 3, std(ddof=1) sqrt(2.5)
B = A + np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # B - A = 1..5


def replicates(values: np.ndarray, k: int) -> np.ndarray:
    """Return ``k`` replicates per unit whose per-unit mean is exactly ``values`` (integer-valued inputs)."""
    offsets = np.arange(k) - (k - 1) / 2.0  # symmetric offsets summing to zero
    return values[:, None] + offsets[None, :]


class TestUnitMeans:
    def test_1d_is_returned_as_float_array(self):
        result = unit_means(np.array([1, 2, 3]))
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_accepts_a_list(self):
        np.testing.assert_array_equal(unit_means([1.0, 2.0]), [1.0, 2.0])

    def test_2d_averages_replicates_within_each_unit(self):
        result = unit_means(np.array([[1.0, 3.0], [2.0, 4.0], [10.0, 20.0]]))
        np.testing.assert_allclose(result, [2.0, 3.0, 15.0])
        assert result.shape == (3,)

    def test_single_replicate_column_equals_1d(self):
        np.testing.assert_array_equal(unit_means(A[:, None]), A)

    @pytest.mark.parametrize("shape", [(), (2, 2, 2)])
    def test_other_dimensions_raise(self, shape):
        data = np.zeros(shape)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Expected a 1D or 2D array, got shape {shape}."),
        ):
            unit_means(data)


class TestNamedTuples:
    def test_mean_se_fields_and_unpacking(self):
        mean, se, n_units = MeanSE(1.0, 0.5, 4)
        assert (mean, se, n_units) == (1.0, 0.5, 4)
        assert MeanSE._fields == ("mean", "se", "n_units")

    def test_unpaired_difference_fields(self):
        assert UnpairedDifference._fields == (
            "estimate",
            "se",
            "n_units_reference",
            "n_units_candidate",
        )

    def test_are_immutable(self):
        with pytest.raises(AttributeError):
            MeanSE(1.0, 0.5, 4).mean = 2.0  # type: ignore[misc]


class TestMeanAndSE:
    def test_known_values(self):
        result = mean_and_se(A)
        assert result.mean == pytest.approx(3.0)
        assert result.se == pytest.approx(math.sqrt(2.5) / math.sqrt(5))
        assert result.n_units == 5

    def test_two_units_is_the_minimum(self):
        # [1, 3]: std(ddof=1) = sqrt(2), so se = sqrt(2) / sqrt(2) = 1.
        result = mean_and_se(np.array([1.0, 3.0]))
        assert result.mean == pytest.approx(2.0)
        assert result.se == pytest.approx(1.0)
        assert result.n_units == 2

    def test_matches_numpy_on_random_data(self):
        x = np.random.default_rng(0).normal(loc=2.0, scale=3.0, size=50)
        result = mean_and_se(x)
        assert result.mean == pytest.approx(x.mean())
        assert result.se == pytest.approx(x.std(ddof=1) / math.sqrt(50))

    def test_result_types(self):
        result = mean_and_se(A)
        assert isinstance(result, MeanSE)
        assert isinstance(result.mean, float)
        assert isinstance(result.se, float)
        assert isinstance(result.n_units, int)

    def test_replicates_are_averaged_within_units(self):
        assert mean_and_se(replicates(A, 2)) == mean_and_se(A)

    def test_number_of_replicates_does_not_change_the_result(self):
        assert mean_and_se(replicates(A, 3)) == mean_and_se(A)

    def test_n_units_counts_units_not_observations(self):
        assert mean_and_se(replicates(A, 4)).n_units == 5

    def test_shift_moves_the_mean_only(self):
        base, shifted = mean_and_se(A), mean_and_se(A + 100.0)
        assert shifted.mean == pytest.approx(base.mean + 100.0)
        assert shifted.se == pytest.approx(base.se)

    def test_scaling_scales_mean_and_se(self):
        base, scaled = mean_and_se(A), mean_and_se(A * 1000.0)
        assert scaled.mean == pytest.approx(base.mean * 1000.0)
        assert scaled.se == pytest.approx(base.se * 1000.0)

    @pytest.mark.parametrize("n", [2, 3, 7, 100])
    def test_identical_values_give_exactly_zero_se(self, n):
        # 0.1 is not exactly representable: a floating-point mean would leave a ~1e-17 residual in the std.
        result = mean_and_se(np.full(n, 0.1))
        assert result.se == 0.0
        assert result.mean == 0.1
        assert result.n_units == n

    def test_identical_unit_means_from_replicates_give_exactly_zero_se(self):
        result = mean_and_se(np.full((4, 3), 0.1))
        assert result.se == 0.0

    @pytest.mark.parametrize(
        "data", [np.array([1.0]), np.empty(0), np.array([[1.0, 2.0, 3.0]])]
    )
    def test_fewer_than_two_units_raises(self, data):
        with pytest.raises(
            ValueError,
            match="At least 2 units are required to compute the standard error",
        ):
            mean_and_se(data)

    def test_error_message_reports_the_number_of_units(self):
        expected = "At least 2 units are required to compute the standard error, got 1."
        with pytest.raises(ValueError, match=re.escape(expected)):
            mean_and_se(np.array([1.0]))

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_raises_without_numpy_warnings(self, bad, recwarn):
        x = A.copy()
        x[2] = bad
        with pytest.raises(
            ValueError, match="'outcomes' must contain only finite values"
        ):
            mean_and_se(x)
        assert len(recwarn) == 0

    def test_non_finite_replicate_is_detected(self):
        x = replicates(A, 2)
        x[0, 1] = math.nan
        with pytest.raises(ValueError, match="must contain only finite values"):
            mean_and_se(x)

    def test_3d_input_raises(self):
        with pytest.raises(ValueError, match="Expected a 1D or 2D array"):
            mean_and_se(np.zeros((2, 2, 2)))


class TestPairedDifferences:
    def test_is_candidate_minus_reference(self):
        np.testing.assert_allclose(paired_differences(A, B), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_swapping_arguments_flips_the_sign(self):
        np.testing.assert_allclose(paired_differences(B, A), -paired_differences(A, B))

    def test_output_shape_is_one_value_per_unit(self):
        assert paired_differences(replicates(A, 3), replicates(B, 2)).shape == (5,)

    def test_replicates_are_averaged_within_units(self):
        np.testing.assert_allclose(
            paired_differences(replicates(A, 2), replicates(B, 3)),
            paired_differences(A, B),
        )

    def test_mixed_1d_and_2d_inputs(self):
        np.testing.assert_allclose(
            paired_differences(A, replicates(B, 4)), paired_differences(A, B)
        )

    def test_different_number_of_units_raises(self):
        with pytest.raises(
            ValueError, match=re.escape("Number of units differs: 5 vs 4.")
        ):
            paired_differences(A, B[:4])

    def test_different_number_of_units_with_replicates_raises(self):
        with pytest.raises(
            ValueError, match=re.escape("Number of units differs: 5 vs 4.")
        ):
            paired_differences(replicates(A, 2), B[:4])


class TestPairedStd:
    def test_known_value(self):
        # Differences 1..5 have std(ddof=1) = sqrt(2.5).
        assert paired_std(A, B) == pytest.approx(math.sqrt(2.5))

    def test_matches_numpy_on_random_data(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=40), rng.normal(loc=0.5, size=40)
        assert paired_std(a, b) == pytest.approx((b - a).std(ddof=1))

    def test_symmetric_in_the_two_conditions(self):
        assert paired_std(A, B) == pytest.approx(paired_std(B, A))

    def test_ignores_a_constant_offset(self):
        assert paired_std(A, A + 7.0) == 0.0

    def test_replicates_are_averaged_within_units(self):
        assert paired_std(replicates(A, 2), replicates(B, 3)) == pytest.approx(
            paired_std(A, B)
        )

    def test_returns_a_float(self):
        assert isinstance(paired_std(A, B), float)

    def test_one_unit_raises(self):
        expected = (
            "At least 2 units are required to compute the standard deviation, got 1."
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            paired_std(A[:1], B[:1])

    def test_different_number_of_units_raises(self):
        with pytest.raises(ValueError, match="Number of units differs"):
            paired_std(A, B[:3])


class TestUnpairedStd:
    def test_known_value_equal_sizes(self):
        # Sums of squares 2 and 8, df = 3 + 3 - 2 = 4: sqrt(10 / 4).
        reference = np.array([1.0, 2.0, 3.0])
        candidate = np.array([10.0, 12.0, 14.0])
        assert unpaired_std(reference, candidate) == pytest.approx(math.sqrt(2.5))

    def test_known_value_unequal_sizes(self):
        # Sums of squares 10 (5 units) and 2 (2 units), df = 5: sqrt(12 / 5).
        assert unpaired_std(A, np.array([0.0, 2.0])) == pytest.approx(
            math.sqrt(12.0 / 5.0)
        )

    def test_matches_the_weighted_average_of_variances(self):
        rng = np.random.default_rng(2)
        r, c = rng.normal(scale=2.0, size=60), rng.normal(loc=1.0, scale=5.0, size=25)
        expected = math.sqrt(
            ((len(r) - 1) * r.var(ddof=1) + (len(c) - 1) * c.var(ddof=1))
            / (len(r) + len(c) - 2)
        )
        assert unpaired_std(r, c) == pytest.approx(expected)

    def test_equal_group_stds_give_that_std(self):
        r = np.array([1.0, 2.0, 3.0, 4.0])
        assert unpaired_std(r, r + 50.0) == pytest.approx(r.std(ddof=1))

    def test_ignores_the_difference_in_means(self):
        assert unpaired_std(A, B[:4]) == pytest.approx(unpaired_std(A, B[:4] + 1000.0))

    def test_symmetric_in_the_two_groups(self):
        assert unpaired_std(A, B[:3]) == pytest.approx(unpaired_std(B[:3], A))

    def test_replicates_are_averaged_within_units(self):
        assert unpaired_std(replicates(A, 2), replicates(B[:3], 3)) == pytest.approx(
            unpaired_std(A, B[:3])
        )

    def test_constant_groups_give_zero(self):
        assert unpaired_std(np.full(3, 1.0), np.full(4, 2.0)) == 0.0

    def test_returns_a_float(self):
        assert isinstance(unpaired_std(A, B), float)

    @pytest.mark.parametrize("which", ["reference", "candidate"])
    def test_group_with_fewer_than_two_units_raises(self, which):
        reference, candidate = (A[:1], B) if which == "reference" else (A, B[:1])
        expected = (
            "At least 2 units are required to compute the standard deviation, got 1."
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            unpaired_std(reference, candidate)


class TestUnpairedDifference:
    REFERENCE = np.array([1.0, 2.0, 3.0])  # mean 2, var 1, se^2 = 1/3
    CANDIDATE = np.array([10.0, 12.0, 14.0, 16.0])  # mean 13, var 20/3, se^2 = 5/3

    def test_known_values(self):
        result = unpaired_difference(self.REFERENCE, self.CANDIDATE)
        assert result.estimate == pytest.approx(11.0)
        assert result.se == pytest.approx(math.sqrt(2.0))  # sqrt(1/3 + 5/3)
        assert result.n_units_reference == 3
        assert result.n_units_candidate == 4

    def test_is_candidate_minus_reference(self):
        result = unpaired_difference(self.REFERENCE, self.CANDIDATE)
        assert result.estimate == pytest.approx(
            self.CANDIDATE.mean() - self.REFERENCE.mean()
        )

    def test_se_combines_the_group_standard_errors(self):
        rng = np.random.default_rng(3)
        r, c = rng.normal(size=80), rng.normal(loc=0.3, scale=2.0, size=30)
        expected = math.hypot(
            r.std(ddof=1) / math.sqrt(80), c.std(ddof=1) / math.sqrt(30)
        )
        assert unpaired_difference(r, c).se == pytest.approx(expected)

    def test_se_is_welch_not_pooled(self):
        result = unpaired_difference(self.REFERENCE, self.CANDIDATE)
        pooled_var = (2 * 1.0 + 3 * (20.0 / 3.0)) / 5.0
        pooled_se = math.sqrt(pooled_var * (1 / 3 + 1 / 4))
        assert result.se != pytest.approx(pooled_se)

    def test_swapping_groups_negates_the_estimate_and_keeps_the_se(self):
        forward = unpaired_difference(self.REFERENCE, self.CANDIDATE)
        backward = unpaired_difference(self.CANDIDATE, self.REFERENCE)
        assert backward.estimate == pytest.approx(-forward.estimate)
        assert backward.se == pytest.approx(forward.se)
        assert backward.n_units_reference == forward.n_units_candidate
        assert backward.n_units_candidate == forward.n_units_reference

    def test_replicates_are_averaged_within_units(self):
        with_replicates = unpaired_difference(replicates(A, 2), replicates(B[:3], 3))
        assert with_replicates == unpaired_difference(A, B[:3])

    def test_groups_may_have_different_numbers_of_units(self):
        result = unpaired_difference(A, B[:2])
        assert (result.n_units_reference, result.n_units_candidate) == (5, 2)

    def test_two_constant_groups_give_exactly_zero_se(self):
        result = unpaired_difference(np.full(3, 0.1), np.full(4, 0.2))
        assert result.se == 0.0
        assert result.estimate == pytest.approx(0.1)

    def test_one_constant_group_keeps_the_other_groups_uncertainty(self):
        result = unpaired_difference(np.full(3, 1.0), self.CANDIDATE)
        assert result.se == pytest.approx(mean_and_se(self.CANDIDATE).se)

    def test_result_type(self):
        result = unpaired_difference(A, B)
        assert isinstance(result, UnpairedDifference)
        estimate, se, n_reference, n_candidate = result
        assert (estimate, se, n_reference, n_candidate) == tuple(result)

    @pytest.mark.parametrize("which", ["reference", "candidate"])
    def test_group_with_fewer_than_two_units_raises(self, which):
        reference, candidate = (A[:1], B) if which == "reference" else (A, B[:1])
        with pytest.raises(
            ValueError,
            match="At least 2 units are required to compute the standard error",
        ):
            unpaired_difference(reference, candidate)

    @pytest.mark.parametrize("which", ["reference", "candidate"])
    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_data_raises(self, which, bad):
        reference, candidate = A.copy(), B.copy()
        (reference if which == "reference" else candidate)[0] = bad
        with pytest.raises(ValueError, match="must contain only finite values"):
            unpaired_difference(reference, candidate)
