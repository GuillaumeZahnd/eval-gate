import math

import numpy as np
import pytest
from scipy.stats import norm

from eval_gate.confidence import confidence_bounds

Z_975 = 1.959963984540054  # norm.ppf(0.975)
Z_95 = 1.6448536269514722  # norm.ppf(0.95)

HYPOTHESES = ["two_sided", "superiority", "non_inferiority", "equivalence"]
NEEDS_MARGIN = ["non_inferiority", "equivalence"]
NO_MARGIN = ["two_sided", "superiority"]


def margin_for(hypothesis: str) -> float:
    return 0.5 if hypothesis in NEEDS_MARGIN else 0.0


class TestKnownValues:
    def test_two_sided_is_symmetric_with_z_975(self):
        lower, upper = confidence_bounds("two_sided", 1.0, 0.5)
        assert lower == pytest.approx(1.0 - Z_975 * 0.5)
        assert upper == pytest.approx(1.0 + Z_975 * 0.5)

    @pytest.mark.parametrize("hypothesis", ["superiority", "non_inferiority"])
    def test_one_sided_has_a_lower_bound_with_z_95_and_no_upper_bound(self, hypothesis):
        lower, upper = confidence_bounds(
            hypothesis, 1.0, 0.5, margin=margin_for(hypothesis)
        )
        assert lower == pytest.approx(1.0 - Z_95 * 0.5)
        assert upper == math.inf

    def test_equivalence_is_the_two_sided_interval_at_2_alpha(self):
        lower, upper = confidence_bounds("equivalence", 1.0, 0.5, margin=0.5)
        assert lower == pytest.approx(1.0 - Z_95 * 0.5)
        assert upper == pytest.approx(1.0 + Z_95 * 0.5)

    def test_alpha_changes_the_critical_value(self):
        lower, upper = confidence_bounds("two_sided", 0.0, 1.0, alpha=0.10)
        assert (lower, upper) == pytest.approx((-Z_95, Z_95))

    def test_alpha_defaults_to_0_05(self):
        assert confidence_bounds("two_sided", 2.0, 1.0) == confidence_bounds(
            "two_sided", 2.0, 1.0, 0.05
        )

    def test_negative_estimate(self):
        lower, upper = confidence_bounds("two_sided", -3.0, 1.0)
        assert lower == pytest.approx(-3.0 - Z_975)
        assert upper == pytest.approx(-3.0 + Z_975)

    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_returns_a_pair_of_floats(self, hypothesis):
        result = confidence_bounds(hypothesis, 1, 2, margin=margin_for(hypothesis))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(bound, float) for bound in result)


class TestRelationsBetweenHypotheses:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_equivalence_equals_two_sided_at_twice_alpha(self, alpha):
        equivalence = confidence_bounds("equivalence", 0.3, 0.8, alpha, margin=1.0)
        two_sided = confidence_bounds("two_sided", 0.3, 0.8, 2 * alpha)
        assert equivalence == pytest.approx(two_sided)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_one_sided_lower_bound_equals_two_sided_lower_at_twice_alpha(self, alpha):
        one_sided = confidence_bounds("superiority", 0.3, 0.8, alpha)
        two_sided = confidence_bounds("two_sided", 0.3, 0.8, 2 * alpha)
        assert one_sided[0] == pytest.approx(two_sided[0])

    def test_superiority_and_non_inferiority_give_the_same_bounds(self):
        assert confidence_bounds("superiority", 0.3, 0.8) == confidence_bounds(
            "non_inferiority", 0.3, 0.8, margin=0.4
        )

    def test_equivalence_interval_is_narrower_than_two_sided(self):
        eq_lower, eq_upper = confidence_bounds("equivalence", 0.0, 1.0, margin=1.0)
        ts_lower, ts_upper = confidence_bounds("two_sided", 0.0, 1.0)
        assert ts_lower < eq_lower < eq_upper < ts_upper

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [0.1, 0.5, 7.0])
    def test_margin_does_not_change_the_bounds(self, hypothesis, margin):
        assert confidence_bounds(
            hypothesis, 0.3, 0.8, margin=margin
        ) == confidence_bounds(hypothesis, 0.3, 0.8, margin=0.5)


class TestProperties:
    @pytest.mark.parametrize("hypothesis", ["two_sided", "equivalence"])
    def test_two_sided_intervals_are_centred_on_the_estimate(self, hypothesis):
        lower, upper = confidence_bounds(
            hypothesis, 2.5, 0.7, margin=margin_for(hypothesis)
        )
        assert (lower + upper) / 2 == pytest.approx(2.5)

    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_smaller_alpha_gives_a_lower_lower_bound(self, hypothesis):
        margin = margin_for(hypothesis)
        loose = confidence_bounds(hypothesis, 0.0, 1.0, alpha=0.10, margin=margin)
        strict = confidence_bounds(hypothesis, 0.0, 1.0, alpha=0.01, margin=margin)
        assert strict[0] < loose[0]
        assert strict[1] >= loose[1]

    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_larger_se_gives_a_lower_lower_bound(self, hypothesis):
        margin = margin_for(hypothesis)
        small = confidence_bounds(hypothesis, 0.0, 0.5, margin=margin)
        large = confidence_bounds(hypothesis, 0.0, 2.0, margin=margin)
        assert large[0] < small[0]
        assert large[1] >= small[1]

    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_shifting_the_estimate_shifts_the_bounds(self, hypothesis):
        margin = margin_for(hypothesis)
        base = confidence_bounds(hypothesis, 0.0, 1.0, margin=margin)
        shifted = confidence_bounds(hypothesis, 10.0, 1.0, margin=margin)
        assert shifted[0] == pytest.approx(base[0] + 10.0)
        if math.isfinite(base[1]):
            assert shifted[1] == pytest.approx(base[1] + 10.0)

    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_zero_se_collapses_to_the_estimate(self, hypothesis):
        lower, upper = confidence_bounds(
            hypothesis, 1.5, 0.0, margin=margin_for(hypothesis)
        )
        assert lower == 1.5
        assert upper == (
            math.inf if hypothesis in ("superiority", "non_inferiority") else 1.5
        )


def _rejects(
    hypothesis: str, estimate: float, se: float, margin: float, alpha: float
) -> bool:
    """Decision of the corresponding normal-approximation test, computed independently."""
    match hypothesis:
        case "two_sided":
            p = 2 * norm.sf(abs(estimate) / se)
        case "superiority":
            p = norm.sf(estimate / se)
        case "non_inferiority":
            p = norm.sf((estimate + margin) / se)
        case _:
            p = max(
                norm.sf((estimate + margin) / se), norm.cdf((estimate - margin) / se)
            )
    return bool(p < alpha)


def _excludes_null(hypothesis: str, lower: float, upper: float, margin: float) -> bool:
    match hypothesis:
        case "two_sided":
            return lower > 0.0 or upper < 0.0
        case "superiority":
            return lower > 0.0
        case "non_inferiority":
            return lower > -margin
        case _:
            return lower > -margin and upper < margin


class TestDualityWithTheTest:
    @pytest.mark.parametrize("hypothesis", HYPOTHESES)
    def test_bounds_exclude_the_null_exactly_when_the_test_rejects(self, hypothesis):
        rng = np.random.default_rng(0)
        for _ in range(300):
            estimate = float(rng.normal(0.0, 2.0))
            se = float(rng.uniform(0.1, 2.0))
            margin = float(rng.uniform(0.1, 3.0)) if hypothesis in NEEDS_MARGIN else 0.0
            alpha = float(rng.choice([0.01, 0.05, 0.10]))

            lower, upper = confidence_bounds(
                hypothesis, estimate, se, alpha, margin=margin
            )

            assert _excludes_null(hypothesis, lower, upper, margin) == _rejects(
                hypothesis, estimate, se, margin, alpha
            )


class TestCoverage:
    N_REPS = 4000
    MU = 0.7
    SE = 1.3

    def _draws(self):
        return np.random.default_rng(0).normal(self.MU, self.SE, self.N_REPS)

    def test_two_sided_covers_the_true_value_with_probability_1_minus_alpha(self):
        covered = [
            lo <= self.MU <= hi
            for lo, hi in (
                confidence_bounds("two_sided", float(x), self.SE) for x in self._draws()
            )
        ]
        assert np.mean(covered) == pytest.approx(0.95, abs=0.025)

    def test_one_sided_lower_bound_covers_with_probability_1_minus_alpha(self):
        covered = [
            confidence_bounds("superiority", float(x), self.SE)[0] <= self.MU
            for x in self._draws()
        ]
        assert np.mean(covered) == pytest.approx(0.95, abs=0.025)

    def test_equivalence_interval_covers_with_probability_1_minus_2_alpha(self):
        covered = [
            lo <= self.MU <= hi
            for lo, hi in (
                confidence_bounds("equivalence", float(x), self.SE, margin=1.0)
                for x in self._draws()
            )
        ]
        assert np.mean(covered) == pytest.approx(0.90, abs=0.025)


class TestInputValidation:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="'alpha' must be in"):
            confidence_bounds("two_sided", 0.0, 1.0, alpha)

    @pytest.mark.parametrize("se", [math.nan, math.inf, -math.inf])
    def test_non_finite_se_raises(self, se):
        with pytest.raises(ValueError, match="'se' must be finite"):
            confidence_bounds("two_sided", 0.0, se)

    @pytest.mark.parametrize("se", [-1.0, -1e-12])
    def test_negative_se_raises(self, se):
        with pytest.raises(ValueError, match="'se' must be non-negative"):
            confidence_bounds("two_sided", 0.0, se)

    @pytest.mark.parametrize("bad", ["unknown", "Two_Sided", "", None])
    def test_unknown_alternative_hypothesis_raises(self, bad):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            confidence_bounds(bad, 0.0, 1.0)

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_margin_raises_when_required(self, hypothesis, margin):
        with pytest.raises(ValueError, match="'margin' must be"):
            confidence_bounds(hypothesis, 0.0, 1.0, margin=margin)

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    def test_missing_margin_raises_when_required(self, hypothesis):
        with pytest.raises(ValueError, match="'margin' must be"):
            confidence_bounds(hypothesis, 0.0, 1.0)

    @pytest.mark.parametrize("hypothesis", NO_MARGIN)
    @pytest.mark.parametrize("margin", [-1.0, math.nan, math.inf, 5.0])
    def test_margin_is_ignored_when_not_required(self, hypothesis, margin):
        assert confidence_bounds(
            hypothesis, 0.3, 0.8, margin=margin
        ) == confidence_bounds(hypothesis, 0.3, 0.8)

    def test_zero_se_is_allowed(self):
        assert confidence_bounds("two_sided", 1.0, 0.0) == (1.0, 1.0)

    def test_margin_is_keyword_only(self):
        with pytest.raises(TypeError):
            confidence_bounds("equivalence", 0.0, 1.0, 0.05, 0.5)  # type: ignore[misc]
