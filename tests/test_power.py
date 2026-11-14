import dataclasses
import itertools
import math
import re

import numpy as np
import pytest
from scipy.stats import norm

from eval_gate.power import (
    PairedPowerResult,
    PairedRequiredNResult,
    UnpairedPowerResult,
    UnpairedRequiredNResult,
    _find_required_n,
    paired_power,
    required_paired_n_units,
    required_unpaired_n_units,
    unpaired_power,
)

# (hypothesis, effect, margin) with std = 1. The effect lies in the alternative hypothesis.
CASES = [
    ("two_sided", 0.3, 0.0),
    ("superiority", 0.3, 0.0),
    ("non_inferiority", 0.1, 0.3),
    ("equivalence", 0.05, 0.4),
]
NEEDS_MARGIN = ["non_inferiority", "equivalence"]
NO_MARGIN = ["two_sided", "superiority"]


def expected_power(hypothesis, scale, effect, margin, alpha=0.05):
    """Independent normal-approximation power; ``scale`` converts a difference into standard-error units."""
    if hypothesis == "two_sided":
        z = norm.ppf(1 - alpha / 2)
        return norm.cdf(effect * scale - z) + norm.cdf(-effect * scale - z)
    z = norm.ppf(1 - alpha)
    if hypothesis in ("superiority", "non_inferiority"):
        return norm.cdf((effect + margin) * scale - z)
    lower = norm.cdf((effect + margin) * scale - z)
    upper = norm.cdf((margin - effect) * scale - z)
    return max(0.0, lower + upper - 1.0)


def simulated_power(
    hypothesis, n, effect, std, margin, alpha=0.05, reps=20_000, seed=0
):
    """Monte Carlo power of the z-test with known std, written from the test definitions."""
    se = std / math.sqrt(n)
    estimate = np.random.default_rng(seed).normal(effect, se, reps)
    z_one, z_two = norm.ppf(1 - alpha), norm.ppf(1 - alpha / 2)
    match hypothesis:
        case "two_sided":
            reject = np.abs(estimate) / se > z_two
        case "superiority":
            reject = estimate / se > z_one
        case "non_inferiority":
            reject = (estimate + margin) / se > z_one
        case _:
            reject = ((estimate + margin) / se > z_one) & (
                (margin - estimate) / se > z_one
            )
    return float(reject.mean())


def closed_form_n(effect_plus_margin, std, alpha, power, ratio_term=1.0):
    """Exact required n for one-sided ('greater') tests: power(n) >= target iff sqrt(n) >= (za + zb) * ..."""
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return math.ceil((z * std / effect_plus_margin) ** 2 * ratio_term)


class TestPairedPower:
    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    @pytest.mark.parametrize("n_units", [10, 50])
    def test_matches_the_normal_approximation(
        self, hypothesis, effect, margin, n_units
    ):
        result = paired_power(n_units, effect, 1.0, hypothesis, margin=margin)
        expected = expected_power(hypothesis, math.sqrt(n_units), effect, margin)
        assert result.power == pytest.approx(expected)

    def test_readme_equivalence_example(self):
        result = paired_power(400, 0.05, 0.698, "equivalence", margin=0.15)
        assert result.power == pytest.approx(0.889, abs=5e-4)

    @pytest.mark.parametrize(
        ("hypothesis", "n_units", "effect", "std", "margin"),
        [
            ("two_sided", 30, 0.5, 2.0, 0.0),
            ("superiority", 30, 0.5, 2.0, 0.0),
            ("non_inferiority", 30, 0.2, 2.0, 0.6),
            ("equivalence", 30, 0.1, 2.0, 1.2),
        ],
    )
    def test_agrees_with_a_simulation_of_the_test(
        self, hypothesis, n_units, effect, std, margin
    ):
        power = paired_power(n_units, effect, std, hypothesis, margin=margin).power
        assert power == pytest.approx(
            simulated_power(hypothesis, n_units, effect, std, margin), abs=0.02
        )

    @pytest.mark.parametrize(
        ("hypothesis", "effect", "margin"),
        [
            ("two_sided", 0.0, 0.0),
            ("superiority", 0.0, 0.0),
            ("non_inferiority", -0.3, 0.3),
        ],
    )
    def test_power_equals_alpha_on_the_boundary_of_the_null(
        self, hypothesis, effect, margin
    ):
        result = paired_power(40, effect, 1.0, hypothesis, margin=margin, alpha=0.05)
        assert result.power == pytest.approx(0.05)

    @pytest.mark.parametrize("effect", [0.3, -0.3])
    def test_equivalence_power_is_at_most_alpha_on_the_boundary(self, effect):
        assert (
            paired_power(40, effect, 1.0, "equivalence", margin=0.3).power
            <= 0.05 + 1e-12
        )

    @pytest.mark.parametrize("hypothesis", ["two_sided", "equivalence"])
    def test_symmetric_in_the_sign_of_the_effect(self, hypothesis):
        margin = 0.4 if hypothesis == "equivalence" else 0.0
        positive = paired_power(30, 0.2, 1.0, hypothesis, margin=margin).power
        negative = paired_power(30, -0.2, 1.0, hypothesis, margin=margin).power
        assert positive == pytest.approx(negative)

    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    def test_power_increases_with_the_number_of_units(self, hypothesis, effect, margin):
        powers = [
            paired_power(n, effect, 1.0, hypothesis, margin=margin).power
            for n in (5, 10, 20, 50, 100)
        ]
        assert all(later >= earlier for earlier, later in itertools.pairwise(powers))
        assert powers[-1] > powers[0]

    @pytest.mark.parametrize("hypothesis", ["two_sided", "superiority"])
    def test_power_increases_with_the_effect(self, hypothesis):
        powers = [
            paired_power(30, e, 1.0, hypothesis).power for e in (0.1, 0.2, 0.4, 0.8)
        ]
        assert powers == sorted(powers)
        assert powers[-1] > powers[0]

    def test_equivalence_power_decreases_as_the_effect_nears_the_margin(self):
        powers = [
            paired_power(100, e, 1.0, "equivalence", margin=0.4).power
            for e in (0.0, 0.1, 0.2, 0.3)
        ]
        assert powers == sorted(powers, reverse=True)

    def test_power_decreases_with_a_larger_std(self):
        assert (
            paired_power(30, 0.3, 2.0, "superiority").power
            < paired_power(30, 0.3, 1.0, "superiority").power
        )

    def test_smaller_alpha_gives_less_power(self):
        loose = paired_power(30, 0.3, 1.0, "superiority", alpha=0.10).power
        strict = paired_power(30, 0.3, 1.0, "superiority", alpha=0.01).power
        assert strict < loose

    def test_equivalence_power_is_clamped_at_zero(self):
        # With 2 units and a tiny margin the two one-sided powers sum to less than 1.
        assert paired_power(2, 0.0, 1.0, "equivalence", margin=0.01).power == 0.0

    def test_power_is_a_probability(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            for hypothesis, _, margin in CASES:
                n = int(rng.integers(2, 500))
                effect = float(rng.normal(0.0, 0.5))
                if hypothesis == "equivalence":
                    margin = float(rng.uniform(0.1, 1.0))
                power = paired_power(
                    n, effect, float(rng.uniform(0.2, 3.0)), hypothesis, margin=margin
                ).power
                assert 0.0 <= power <= 1.0

    @pytest.mark.parametrize("hypothesis", NO_MARGIN)
    def test_margin_is_ignored_when_not_required(self, hypothesis):
        with_margin = paired_power(20, 0.3, 1.0, hypothesis, margin=5.0)
        without = paired_power(20, 0.3, 1.0, hypothesis)
        assert with_margin == without
        assert with_margin.margin == 0.0

    def test_result_holds_the_design_conditions(self):
        result = paired_power(25, 0.1, 2.0, "non_inferiority", margin=0.3, alpha=0.1)
        assert isinstance(result, PairedPowerResult)
        assert result.n_units == 25
        assert result.effect == 0.1
        assert result.std == 2.0
        assert result.alternative_hypothesis == "non_inferiority"
        assert result.margin == 0.3
        assert result.alpha == 0.1

    def test_default_hypothesis_is_two_sided(self):
        assert paired_power(20, 0.3, 1.0).alternative_hypothesis == "two_sided"

    def test_two_units_is_the_minimum(self):
        assert 0.0 <= paired_power(2, 0.3, 1.0).power <= 1.0

    def test_effect_outside_the_alternative_is_allowed(self):
        # Power at the null (or beyond) is a legitimate question, unlike for the sample-size search.
        assert paired_power(30, 0.0, 1.0, "superiority").power == pytest.approx(0.05)
        assert paired_power(30, -0.5, 1.0, "superiority").power < 0.05


class TestPairedPowerValidation:
    @pytest.mark.parametrize("n_units", [1, 0, -3, math.nan])
    def test_fewer_than_two_units_raises(self, n_units):
        expected = f"At least 2 units are required to compute the power, got {n_units}."
        with pytest.raises(ValueError, match=re.escape(expected)):
            paired_power(n_units, 0.3, 1.0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="'alpha' must be in"):
            paired_power(20, 0.3, 1.0, alpha=alpha)

    @pytest.mark.parametrize("std", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_std_raises(self, std):
        with pytest.raises(ValueError, match="'std' must be"):
            paired_power(20, 0.3, std)

    @pytest.mark.parametrize("effect", [math.nan, math.inf, -math.inf])
    def test_non_finite_effect_raises(self, effect):
        with pytest.raises(ValueError, match="'effect' must be finite"):
            paired_power(20, effect, 1.0)

    def test_unknown_hypothesis_raises(self):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            paired_power(20, 0.3, 1.0, "unknown")

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_or_missing_margin_raises_when_required(self, hypothesis, margin):
        with pytest.raises(ValueError, match="'margin' must be"):
            paired_power(20, 0.1, 1.0, hypothesis, margin=margin)

    def test_margin_and_alpha_are_keyword_only(self):
        with pytest.raises(TypeError):
            paired_power(20, 0.3, 1.0, "two_sided", 0.05)  # type: ignore[misc]


class TestUnpairedPower:
    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    @pytest.mark.parametrize(("n_reference", "n_candidate"), [(30, 30), (60, 20)])
    def test_matches_the_normal_approximation(
        self, hypothesis, effect, margin, n_reference, n_candidate
    ):
        result = unpaired_power(
            n_reference, n_candidate, effect, 1.0, hypothesis, margin=margin
        )
        scale = 1.0 / math.sqrt(1.0 / n_reference + 1.0 / n_candidate)
        assert result.power == pytest.approx(
            expected_power(hypothesis, scale, effect, margin)
        )

    def test_readme_superiority_example(self):
        result = unpaired_power(300, 100, 0.05, 3.708, "superiority")
        assert result.power == pytest.approx(0.063, abs=5e-4)

    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    def test_balanced_groups_equal_a_paired_design_with_half_the_units(
        self, hypothesis, effect, margin
    ):
        unpaired = unpaired_power(
            100, 100, effect, 1.0, hypothesis, margin=margin
        ).power
        paired = paired_power(50, effect, 1.0, hypothesis, margin=margin).power
        assert unpaired == pytest.approx(paired)

    def test_swapping_group_sizes_gives_the_same_power(self):
        assert unpaired_power(80, 20, 0.3, 1.0, "superiority").power == pytest.approx(
            unpaired_power(20, 80, 0.3, 1.0, "superiority").power
        )

    @pytest.mark.parametrize("which", ["reference", "candidate"])
    def test_more_units_in_either_group_gives_more_power(self, which):
        base = unpaired_power(30, 30, 0.3, 1.0, "superiority").power
        more = (
            unpaired_power(60, 30, 0.3, 1.0, "superiority")
            if which == "reference"
            else unpaired_power(30, 60, 0.3, 1.0, "superiority")
        ).power
        assert more > base

    @pytest.mark.parametrize(
        ("hypothesis", "effect", "margin"),
        [
            ("two_sided", 0.0, 0.0),
            ("superiority", 0.0, 0.0),
            ("non_inferiority", -0.3, 0.3),
        ],
    )
    def test_power_equals_alpha_on_the_boundary_of_the_null(
        self, hypothesis, effect, margin
    ):
        result = unpaired_power(40, 25, effect, 1.0, hypothesis, margin=margin)
        assert result.power == pytest.approx(0.05)

    @pytest.mark.parametrize("hypothesis", NO_MARGIN)
    def test_margin_is_ignored_when_not_required(self, hypothesis):
        with_margin = unpaired_power(20, 30, 0.3, 1.0, hypothesis, margin=5.0)
        assert with_margin == unpaired_power(20, 30, 0.3, 1.0, hypothesis)
        assert with_margin.margin == 0.0

    def test_result_holds_the_design_conditions(self):
        result = unpaired_power(25, 40, 0.1, 2.0, "equivalence", margin=0.5, alpha=0.1)
        assert isinstance(result, UnpairedPowerResult)
        assert (result.n_units_reference, result.n_units_candidate) == (25, 40)
        assert result.effect == 0.1
        assert result.std == 2.0
        assert result.alternative_hypothesis == "equivalence"
        assert result.margin == 0.5
        assert result.alpha == 0.1

    @pytest.mark.parametrize(
        ("n_reference", "n_candidate", "group", "got"),
        [
            (1, 10, "reference", 1),
            (10, 1, "candidate", 1),
            (0, 10, "reference", 0),
            (10, -2, "candidate", -2),
            (math.nan, 10, "reference", math.nan),
            (10, math.nan, "candidate", math.nan),
        ],
    )
    def test_group_with_fewer_than_two_units_raises(
        self, n_reference, n_candidate, group, got
    ):
        expected = f"At least 2 units are required to compute the power ({group} group), got {got}."
        with pytest.raises(ValueError, match=re.escape(expected)):
            unpaired_power(n_reference, n_candidate, 0.3, 1.0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="'alpha' must be in"):
            unpaired_power(20, 20, 0.3, 1.0, alpha=alpha)

    @pytest.mark.parametrize("std", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_std_raises(self, std):
        with pytest.raises(ValueError, match="'std' must be"):
            unpaired_power(20, 20, 0.3, std)

    def test_non_finite_effect_raises(self):
        with pytest.raises(ValueError, match="'effect' must be finite"):
            unpaired_power(20, 20, math.nan, 1.0)

    def test_unknown_hypothesis_raises(self):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            unpaired_power(20, 20, 0.3, 1.0, "unknown")

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    def test_missing_margin_raises_when_required(self, hypothesis):
        with pytest.raises(ValueError, match="'margin' must be"):
            unpaired_power(20, 20, 0.1, 1.0, hypothesis)


class TestRequiredPairedNUnits:
    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    def test_n_is_the_smallest_reaching_the_target_power(
        self, hypothesis, effect, margin
    ):
        result = required_paired_n_units(effect, 1.0, hypothesis, margin=margin)
        assert (
            paired_power(result.n_units, effect, 1.0, hypothesis, margin=margin).power
            >= 0.8
        )
        assert (
            result.n_units == 2
            or paired_power(
                result.n_units - 1, effect, 1.0, hypothesis, margin=margin
            ).power
            < 0.8
        )

    @pytest.mark.parametrize(
        ("effect", "margin", "hypothesis"),
        [
            (0.3, 0.0, "superiority"),
            (0.1, 0.3, "non_inferiority"),
            (0.5, 0.0, "superiority"),
        ],
    )
    @pytest.mark.parametrize(
        ("power", "alpha"), [(0.8, 0.05), (0.9, 0.05), (0.8, 0.01)]
    )
    def test_matches_the_closed_form_for_one_sided_tests(
        self, effect, margin, hypothesis, power, alpha
    ):
        result = required_paired_n_units(
            effect, 1.0, hypothesis, power=power, margin=margin, alpha=alpha
        )
        assert result.n_units == closed_form_n(effect + margin, 1.0, alpha, power)

    def test_readme_example(self):
        result = required_paired_n_units(
            0.15, 0.696, "superiority", power=0.8, alpha=0.05
        )
        assert result.n_units == 134

    def test_minimum_is_two_units(self):
        assert required_paired_n_units(100.0, 1.0, "superiority").n_units == 2

    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    def test_result_is_consistent_with_paired_power(self, hypothesis, effect, margin):
        result = required_paired_n_units(
            effect, 1.0, hypothesis, power=0.9, margin=margin, alpha=0.1
        )
        achieved = paired_power(
            result.n_units, effect, 1.0, hypothesis, margin=margin, alpha=0.1
        )
        assert result.power == achieved.power
        assert result.power >= result.target_power == 0.9

    def test_result_holds_the_design_conditions(self):
        result = required_paired_n_units(
            0.1, 2.0, "non_inferiority", power=0.9, margin=0.3, alpha=0.1
        )
        assert isinstance(result, PairedRequiredNResult)
        assert result.effect == 0.1
        assert result.std == 2.0
        assert result.alternative_hypothesis == "non_inferiority"
        assert result.margin == 0.3
        assert result.alpha == 0.1
        assert result.target_power == 0.9

    def test_two_sided_is_symmetric_in_the_sign_of_the_effect(self):
        assert (
            required_paired_n_units(0.3, 1.0).n_units
            == required_paired_n_units(-0.3, 1.0).n_units
        )

    def test_doubling_the_std_roughly_quadruples_n(self):
        base = required_paired_n_units(0.15, 0.7, "superiority").n_units
        doubled = required_paired_n_units(0.15, 1.4, "superiority").n_units
        assert doubled / base == pytest.approx(4.0, rel=0.05)

    def test_n_decreases_with_a_larger_effect(self):
        assert (
            required_paired_n_units(0.6, 1.0, "superiority").n_units
            < required_paired_n_units(0.3, 1.0, "superiority").n_units
        )

    def test_n_increases_with_the_target_power(self):
        assert (
            required_paired_n_units(0.3, 1.0, power=0.9).n_units
            > required_paired_n_units(0.3, 1.0, power=0.7).n_units
        )

    def test_n_increases_with_a_smaller_alpha(self):
        assert (
            required_paired_n_units(0.3, 1.0, alpha=0.01).n_units
            > required_paired_n_units(0.3, 1.0, alpha=0.10).n_units
        )

    def test_default_power_is_0_8(self):
        assert required_paired_n_units(0.3, 1.0).target_power == 0.8

    @pytest.mark.parametrize("power", [0.0, 1.0, -0.1, 1.5, math.nan])
    def test_invalid_power_raises(self, power):
        with pytest.raises(ValueError, match="'power' must be in"):
            required_paired_n_units(0.3, 1.0, power=power)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="'alpha' must be in"):
            required_paired_n_units(0.3, 1.0, alpha=alpha)

    @pytest.mark.parametrize("std", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_std_raises(self, std):
        with pytest.raises(ValueError, match="'std' must be"):
            required_paired_n_units(0.3, std)

    @pytest.mark.parametrize("effect", [math.nan, math.inf])
    def test_non_finite_effect_raises(self, effect):
        with pytest.raises(ValueError, match="'effect' must be finite"):
            required_paired_n_units(effect, 1.0)

    def test_unknown_hypothesis_raises(self):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            required_paired_n_units(0.3, 1.0, "unknown")

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    def test_missing_margin_raises_when_required(self, hypothesis):
        with pytest.raises(ValueError, match="'margin' must be"):
            required_paired_n_units(0.1, 1.0, hypothesis)

    def test_margin_power_and_alpha_are_keyword_only(self):
        with pytest.raises(TypeError):
            required_paired_n_units(0.3, 1.0, "two_sided", 0.8)  # type: ignore[misc]

    def test_unreachable_target_raises(self):
        with pytest.raises(
            ValueError, match="More than 1000000000 units would be needed"
        ):
            required_paired_n_units(1e-9, 1.0, "superiority")


class TestEffectMustLieInTheAlternative:
    @pytest.mark.parametrize(
        ("hypothesis", "effect", "margin", "region"),
        [
            ("two_sided", 0.0, 0.0, "nonzero"),
            ("superiority", 0.0, 0.0, "greater than 0"),
            ("superiority", -0.1, 0.0, "greater than 0"),
            ("non_inferiority", -0.5, 0.5, "greater than -0.5"),
            ("non_inferiority", -0.6, 0.5, "greater than -0.5"),
            ("equivalence", 0.5, 0.5, "strictly within ±0.5"),
            ("equivalence", -0.5, 0.5, "strictly within ±0.5"),
            ("equivalence", 0.7, 0.5, "strictly within ±0.5"),
        ],
    )
    @pytest.mark.parametrize("design", ["paired", "unpaired"])
    def test_effect_outside_the_alternative_raises(
        self, design, hypothesis, effect, margin, region
    ):
        expected = f"For '{hypothesis}', 'effect' must be {region}, got {effect}."
        required = (
            required_paired_n_units if design == "paired" else required_unpaired_n_units
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            required(effect, 1.0, hypothesis, margin=margin)

    @pytest.mark.parametrize("design", ["paired", "unpaired"])
    def test_non_inferiority_accepts_a_negative_effect_inside_the_margin(self, design):
        required = (
            required_paired_n_units if design == "paired" else required_unpaired_n_units
        )
        assert required(-0.2, 1.0, "non_inferiority", margin=0.5) is not None


class TestRequiredUnpairedNUnits:
    @pytest.mark.parametrize("ratio", [1.0, 2.0])
    @pytest.mark.parametrize(
        ("effect", "margin", "hypothesis"),
        [(0.3, 0.0, "superiority"), (0.1, 0.3, "non_inferiority")],
    )
    def test_matches_the_closed_form_for_one_sided_tests(
        self, ratio, effect, margin, hypothesis
    ):
        result = required_unpaired_n_units(
            effect, 1.0, hypothesis, margin=margin, allocation_ratio=ratio
        )
        expected = closed_form_n(
            effect + margin, 1.0, 0.05, 0.8, ratio_term=1.0 + 1.0 / ratio
        )
        assert result.n_units_reference == expected
        assert result.n_units_candidate == round(expected * ratio)

    @pytest.mark.parametrize(("hypothesis", "effect", "margin"), CASES)
    @pytest.mark.parametrize("ratio", [1.0, 2.0, 0.5])
    def test_sizes_are_the_smallest_reaching_the_target_power(
        self, hypothesis, effect, margin, ratio
    ):
        def candidate_size(n_reference: int) -> int:
            return max(2, round(n_reference * ratio))

        result = required_unpaired_n_units(
            effect, 1.0, hypothesis, margin=margin, allocation_ratio=ratio
        )
        n_reference = result.n_units_reference

        assert result.n_units_candidate == candidate_size(n_reference)
        assert result.power >= 0.8
        if n_reference > 2:
            below = unpaired_power(
                n_reference - 1,
                candidate_size(n_reference - 1),
                effect,
                1.0,
                hypothesis,
                margin=margin,
            )
            assert below.power < 0.8

    def test_balanced_design_by_default(self):
        result = required_unpaired_n_units(0.3, 1.0, "superiority")
        assert result.n_units_candidate == result.n_units_reference

    def test_candidate_group_follows_the_allocation_ratio(self):
        result = required_unpaired_n_units(
            0.3, 1.0, "superiority", allocation_ratio=3.0
        )
        assert result.n_units_candidate == 3 * result.n_units_reference

    def test_unbalanced_design_needs_more_units_in_total(self):
        balanced = required_unpaired_n_units(0.3, 1.0, "superiority")
        skewed = required_unpaired_n_units(
            0.3, 1.0, "superiority", allocation_ratio=4.0
        )
        assert (
            skewed.n_units_reference + skewed.n_units_candidate
            > balanced.n_units_reference + balanced.n_units_candidate
        )

    def test_result_is_consistent_with_unpaired_power(self):
        result = required_unpaired_n_units(
            0.1, 2.0, "equivalence", power=0.9, margin=0.5, alpha=0.1
        )
        achieved = unpaired_power(
            result.n_units_reference,
            result.n_units_candidate,
            0.1,
            2.0,
            "equivalence",
            margin=0.5,
            alpha=0.1,
        )
        assert result.power == achieved.power

    def test_result_holds_the_design_conditions(self):
        result = required_unpaired_n_units(
            0.1, 2.0, "equivalence", power=0.9, margin=0.5, alpha=0.1
        )
        assert isinstance(result, UnpairedRequiredNResult)
        assert result.effect == 0.1
        assert result.std == 2.0
        assert result.alternative_hypothesis == "equivalence"
        assert result.margin == 0.5
        assert result.alpha == 0.1
        assert result.target_power == 0.9

    def test_minimum_is_two_units_per_group(self):
        result = required_unpaired_n_units(100.0, 1.0, "superiority")
        assert (result.n_units_reference, result.n_units_candidate) == (2, 2)

    def test_tiny_allocation_ratio_keeps_at_least_two_candidate_units(self):
        result = required_unpaired_n_units(
            100.0, 1.0, "superiority", allocation_ratio=0.01
        )
        assert result.n_units_candidate >= 2

    def test_larger_std_needs_more_units(self):
        small = required_unpaired_n_units(0.3, 1.0, "superiority").n_units_reference
        large = required_unpaired_n_units(0.3, 2.0, "superiority").n_units_reference
        assert large / small == pytest.approx(4.0, rel=0.05)

    @pytest.mark.parametrize("ratio", [0, -1.0, -0.5])
    def test_non_positive_allocation_ratio_raises(self, ratio):
        expected = f"'allocation_ratio' must be strictly positive, got {ratio}."
        with pytest.raises(ValueError, match=re.escape(expected)):
            required_unpaired_n_units(0.3, 1.0, allocation_ratio=ratio)

    @pytest.mark.parametrize("ratio", [math.nan, math.inf])
    def test_non_finite_allocation_ratio_raises(self, ratio):
        with pytest.raises(ValueError, match="'allocation_ratio' must be finite"):
            required_unpaired_n_units(0.3, 1.0, allocation_ratio=ratio)

    @pytest.mark.parametrize("power", [0.0, 1.0, math.nan])
    def test_invalid_power_raises(self, power):
        with pytest.raises(ValueError, match="'power' must be in"):
            required_unpaired_n_units(0.3, 1.0, power=power)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, math.nan])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="'alpha' must be in"):
            required_unpaired_n_units(0.3, 1.0, alpha=alpha)

    @pytest.mark.parametrize("std", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_std_raises(self, std):
        with pytest.raises(ValueError, match="'std' must be"):
            required_unpaired_n_units(0.3, std)

    def test_non_finite_effect_raises(self):
        with pytest.raises(ValueError, match="'effect' must be finite"):
            required_unpaired_n_units(math.nan, 1.0)

    def test_unknown_hypothesis_raises(self):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            required_unpaired_n_units(0.3, 1.0, "unknown")

    @pytest.mark.parametrize("hypothesis", NEEDS_MARGIN)
    def test_missing_margin_raises_when_required(self, hypothesis):
        with pytest.raises(ValueError, match="'margin' must be"):
            required_unpaired_n_units(0.1, 1.0, hypothesis)

    def test_unreachable_target_raises(self):
        with pytest.raises(
            ValueError, match="More than 1000000000 units would be needed"
        ):
            required_unpaired_n_units(1e-9, 1.0, "superiority")


class TestFindRequiredN:
    def test_returns_the_smallest_n_reaching_the_target(self):
        assert _find_required_n(lambda n: n >= 37) == 37

    @pytest.mark.parametrize("threshold", [2, 3, 17, 64, 65, 1000])
    def test_finds_any_threshold(self, threshold):
        assert _find_required_n(lambda n: n >= threshold) == threshold

    def test_minimum_is_two(self):
        assert _find_required_n(lambda n: True) == 2

    def test_never_reaching_the_target_raises(self):
        with pytest.raises(ValueError, match="More than 100 units would be needed"):
            _find_required_n(lambda n: False, max_units=100)

    def test_the_cap_itself_is_a_valid_answer(self):
        # The search must evaluate ``max_units`` before giving up.
        assert _find_required_n(lambda n: n >= 100, max_units=100) == 100


class TestResultTypes:
    @pytest.mark.parametrize(
        ("cls", "fields"),
        [
            (
                PairedRequiredNResult,
                [
                    "n_units",
                    "power",
                    "target_power",
                    "alternative_hypothesis",
                    "margin",
                    "effect",
                    "std",
                    "alpha",
                ],
            ),
            (
                UnpairedRequiredNResult,
                [
                    "n_units_reference",
                    "n_units_candidate",
                    "power",
                    "target_power",
                    "alternative_hypothesis",
                    "margin",
                    "effect",
                    "std",
                    "alpha",
                ],
            ),
            (
                PairedPowerResult,
                [
                    "n_units",
                    "power",
                    "alternative_hypothesis",
                    "margin",
                    "effect",
                    "std",
                    "alpha",
                ],
            ),
            (
                UnpairedPowerResult,
                [
                    "n_units_reference",
                    "n_units_candidate",
                    "power",
                    "alternative_hypothesis",
                    "margin",
                    "effect",
                    "std",
                    "alpha",
                ],
            ),
        ],
    )
    def test_field_names(self, cls, fields):
        assert [field.name for field in dataclasses.fields(cls)] == fields

    def test_results_are_frozen(self):
        result = paired_power(20, 0.3, 1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.power = 1.0  # type: ignore[misc]

    def test_required_n_results_carry_the_power_result_fields(self):
        required = required_paired_n_units(0.3, 1.0, "superiority")
        achieved = paired_power(required.n_units, 0.3, 1.0, "superiority")
        for field in dataclasses.fields(PairedPowerResult):
            assert getattr(required, field.name) == getattr(achieved, field.name)
