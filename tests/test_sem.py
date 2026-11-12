import math
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from scipy.stats import norm

from eval_gate.sem import SEM_ALTERNATIVES, compute_sem


# 1..5 has mean 3 and std(ddof=1) = sqrt(2.5), so the standard error is sqrt(2.5 / 5) = sqrt(0.5).
OUTCOMES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
MEAN = 3.0
SEM = math.sqrt(0.5)


class TestEstimates:
    def test_mean_sem_and_n_units(self):
        result = compute_sem(OUTCOMES)
        assert result.mean == pytest.approx(MEAN)
        assert result.sem == pytest.approx(SEM)
        assert result.n_units == 5

    def test_two_units_is_the_minimum(self):
        # std(ddof=1) of [1, 3] is sqrt(2), so the standard error is sqrt(2) / sqrt(2) = 1.
        result = compute_sem(np.array([1.0, 3.0]))
        assert result.mean == pytest.approx(2.0)
        assert result.sem == pytest.approx(1.0)

    def test_sem_does_not_depend_on_the_alternative(self):
        two_sided = compute_sem(OUTCOMES, alternative_hypothesis="two_sided")
        superiority = compute_sem(OUTCOMES, alternative_hypothesis="superiority")
        assert two_sided.sem == superiority.sem

    def test_affine_transformation(self):
        scale = 3.0
        shift = -10.0
        base = compute_sem(OUTCOMES)
        moved = compute_sem(scale * OUTCOMES + shift)
        assert moved.mean == pytest.approx(scale * base.mean + shift)
        assert moved.sem == pytest.approx(scale * base.sem)
        assert moved.ci_lower == pytest.approx(scale * base.ci_lower + shift)
        assert moved.ci_upper == pytest.approx(scale * base.ci_upper + shift)

    def test_identical_outcomes_give_a_zero_width_interval(self):
        result = compute_sem(np.full(5, 2.0))
        assert result.sem == 0.0
        assert result.ci_lower == 2.0
        assert result.ci_upper == 2.0


class TestReplicates:
    def test_replicates_are_averaged_within_units(self):
        y = np.random.default_rng(0).normal(size=(20, 4))
        assert compute_sem(y) == compute_sem(y.mean(axis=1))

    def test_n_units_counts_units_not_measurements(self):
        y = np.arange(7.0)[:, None] + np.zeros((7, 3))
        assert compute_sem(y).n_units == 7


class TestInterval:
    @pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
    def test_two_sided_is_a_symmetric_normal_interval(self, confidence):
        result = compute_sem(OUTCOMES, confidence=confidence)
        half = norm.ppf(1 - (1 - confidence) / 2) * SEM
        assert result.ci_lower == pytest.approx(MEAN - half)
        assert result.ci_upper == pytest.approx(MEAN + half)

    @pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
    def test_superiority_is_a_one_sided_lower_bound(self, confidence):
        result = compute_sem(OUTCOMES, confidence=confidence, alternative_hypothesis="superiority")
        assert result.ci_lower == pytest.approx(MEAN - norm.ppf(confidence) * SEM)
        assert result.ci_upper == math.inf

    def test_one_sided_bound_is_tighter_than_the_two_sided_one(self):
        two_sided = compute_sem(OUTCOMES)
        superiority = compute_sem(OUTCOMES, alternative_hypothesis="superiority")
        assert superiority.ci_lower > two_sided.ci_lower

    def test_higher_confidence_widens_the_interval(self):
        narrow = compute_sem(OUTCOMES, confidence=0.9)
        wide = compute_sem(OUTCOMES, confidence=0.99)
        assert wide.ci_lower < narrow.ci_lower
        assert wide.ci_upper > narrow.ci_upper

    def test_zero_width_interval_with_superiority(self):
        result = compute_sem(np.full(5, 2.0), alternative_hypothesis="superiority")
        assert result.ci_lower == 2.0
        assert result.ci_upper == math.inf


class TestCoverage:
    @pytest.mark.parametrize("alternative", ["two_sided", "superiority"])
    def test_interval_covers_the_true_mean_at_the_nominal_rate(self, alternative):
        rng = np.random.default_rng(0)
        true_mean, n_experiments = 2.0, 2000
        covered = 0
        for _ in range(n_experiments):
            sample = rng.normal(true_mean, 1.0, size=100)
            result = compute_sem(sample, alternative_hypothesis=alternative)
            covered += result.ci_lower <= true_mean <= result.ci_upper
        assert covered / n_experiments == pytest.approx(0.95, abs=0.02)


class TestResult:
    def test_defaults(self):
        result = compute_sem(OUTCOMES)
        assert result.confidence == 0.95
        assert result.alternative_hypothesis == "two_sided"

    def test_result_echoes_the_inputs(self):
        result = compute_sem(OUTCOMES, confidence=0.9, alternative_hypothesis="superiority")
        assert result.confidence == 0.9
        assert result.alternative_hypothesis == "superiority"
        assert result.n_units == 5

    def test_result_is_frozen(self):
        result = compute_sem(OUTCOMES)
        with pytest.raises(FrozenInstanceError):
            result.mean = 0.0

    def test_allowed_alternatives(self):
        assert SEM_ALTERNATIVES == ("two_sided", "superiority")

    def test_options_are_keyword_only(self):
        with pytest.raises(TypeError):
            compute_sem(OUTCOMES, 0.9)


class TestInputValidation:
    @pytest.mark.parametrize("alternative", ["non_inferiority", "equivalence", "greater", ""])
    def test_unsupported_alternative_raises(self, alternative):
        with pytest.raises(ValueError, match="'alternative_hypothesis'"):
            compute_sem(OUTCOMES, alternative_hypothesis=alternative)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_confidence_outside_open_unit_interval_raises(self, confidence):
        with pytest.raises(ValueError, match="'confidence'"):
            compute_sem(OUTCOMES, confidence=confidence)

    @pytest.mark.parametrize("outcomes", [np.array([1.0]), np.array([]), np.ones((1, 3))])
    def test_fewer_than_two_units_raises(self, outcomes):
        with pytest.raises(ValueError, match="At least 2 units"):
            compute_sem(outcomes)

    @pytest.mark.parametrize("outcomes", [np.zeros((2, 2, 2)), np.array(1.0)])
    def test_not_1d_or_2d_raises(self, outcomes):
        with pytest.raises(ValueError, match="Expected a 1D or 2D array"):
            compute_sem(outcomes)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_outcomes_raise(self, bad, recwarn):
        outcomes = OUTCOMES.copy()
        outcomes[2] = bad
        with pytest.raises(ValueError):
            compute_sem(outcomes)
        assert not recwarn.list  # no numpy warning escapes before the check

    def test_non_finite_replicate_raises(self):
        outcomes = np.ones((5, 3))
        outcomes[1, 2] = np.nan
        with pytest.raises(ValueError):
            compute_sem(outcomes)
