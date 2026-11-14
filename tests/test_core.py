import dataclasses
import math
import re
from typing import get_args

import pytest

from eval_gate import core
from eval_gate.core import (
    ALTERNATIVE_HYPOTHESES,
    AlternativeHypothesis,
    Direction,
    HypothesisSpec,
    get_spec,
)

# (name, direction, needs_margin) for each alternative hypothesis.
EXPECTED_SPECS = [
    ("two_sided", "two_sided", False),
    ("superiority", "greater", False),
    ("non_inferiority", "greater", True),
    ("equivalence", "equivalence", True),
]
NO_MARGIN = [name for name, _, needs in EXPECTED_SPECS if not needs]
NEEDS_MARGIN = [name for name, _, needs in EXPECTED_SPECS if needs]


class TestAlternativeHypotheses:
    def test_exposes_the_four_hypotheses(self):
        assert ALTERNATIVE_HYPOTHESES == (
            "two_sided",
            "superiority",
            "non_inferiority",
            "equivalence",
        )

    def test_matches_the_literal_type(self):
        assert ALTERNATIVE_HYPOTHESES == get_args(AlternativeHypothesis)

    def test_no_duplicates(self):
        assert len(set(ALTERNATIVE_HYPOTHESES)) == len(ALTERNATIVE_HYPOTHESES)

    def test_spec_table_covers_exactly_the_hypotheses(self):
        # Invariant also enforced at import time by a RuntimeError.
        assert set(core._SPECS) == set(ALTERNATIVE_HYPOTHESES)

    def test_every_direction_is_a_valid_direction(self):
        valid = set(get_args(Direction))
        assert {get_spec(name).direction for name in ALTERNATIVE_HYPOTHESES} <= valid


class TestGetSpec:
    @pytest.mark.parametrize(("name", "direction", "needs_margin"), EXPECTED_SPECS)
    def test_returns_the_expected_spec(self, name, direction, needs_margin):
        spec = get_spec(name)
        assert isinstance(spec, HypothesisSpec)
        assert spec.name == name
        assert spec.direction == direction
        assert spec.needs_margin is needs_margin

    @pytest.mark.parametrize("name", ALTERNATIVE_HYPOTHESES)
    def test_every_listed_hypothesis_resolves(self, name):
        assert get_spec(name).name == name

    def test_returns_the_same_object_each_time(self):
        assert get_spec("equivalence") is get_spec("equivalence")

    def test_superiority_and_non_inferiority_share_a_direction(self):
        assert get_spec("superiority").direction == get_spec("non_inferiority").direction

    @pytest.mark.parametrize(
        "bad",
        [
            "unknown",
            "",
            " superiority",
            "Superiority",
            "TWO_SIDED",
            "two-sided",
            "non-inferiority",
        ],
    )
    def test_unknown_name_raises(self, bad):
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            get_spec(bad)

    @pytest.mark.parametrize("bad", [None, 1, 1.5, ["superiority"], ("superiority",), {"a": 1}])
    def test_non_string_raises_value_error(self, bad):
        # Unhashable inputs (list, dict) and non-str hashables must not leak KeyError or TypeError.
        with pytest.raises(ValueError, match="'alternative_hypothesis' must be one of"):
            get_spec(bad)

    def test_error_message_lists_options_and_shows_the_bad_value(self):
        expected = (
            f"'alternative_hypothesis' must be one of {ALTERNATIVE_HYPOTHESES}, got 'nope'."
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            get_spec("nope")

    def test_error_has_no_chained_cause(self):
        # raised with ``from None``: the internal KeyError is hidden.
        with pytest.raises(ValueError) as exc_info:
            get_spec("nope")
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


class TestMarginForNoMargin:
    @pytest.mark.parametrize("name", NO_MARGIN)
    @pytest.mark.parametrize("margin", [0.0, 0.5, 3, -1.0, math.nan, math.inf])
    def test_returns_zero_and_ignores_the_margin(self, name, margin):
        # No validation either: an unused margin is never an error.
        result = get_spec(name).margin_for(margin)
        assert result == 0.0
        assert isinstance(result, float)


class TestMarginForRequired:
    @pytest.mark.parametrize("name", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [0.1, 1.0, 2.5, 1e-12])
    def test_returns_the_margin(self, name, margin):
        assert get_spec(name).margin_for(margin) == margin

    @pytest.mark.parametrize("name", NEEDS_MARGIN)
    def test_integer_margin_is_returned_as_float(self, name):
        result = get_spec(name).margin_for(2)
        assert result == 2.0
        assert isinstance(result, float)

    @pytest.mark.parametrize("name", NEEDS_MARGIN)
    @pytest.mark.parametrize(
        "margin", [0.0, -0.0, -1.0, -1e-12, math.nan, math.inf, -math.inf]
    )
    def test_non_positive_or_non_finite_margin_raises(self, name, margin):
        with pytest.raises(ValueError, match="'margin' must be"):
            get_spec(name).margin_for(margin)

    @pytest.mark.parametrize("name", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [math.nan, math.inf, -math.inf])
    def test_non_finite_margin_message(self, name, margin):
        expected = f"'margin' must be finite, got {margin}."
        with pytest.raises(ValueError, match=re.escape(expected)):
            get_spec(name).margin_for(margin)

    @pytest.mark.parametrize("name", NEEDS_MARGIN)
    @pytest.mark.parametrize("margin", [0.0, -1.0])
    def test_non_positive_margin_message(self, name, margin):
        expected = f"'margin' must be strictly positive, got {margin}."
        with pytest.raises(ValueError, match=re.escape(expected)):
            get_spec(name).margin_for(margin)


class TestHypothesisSpec:
    def test_needs_margin_defaults_to_false(self):
        assert HypothesisSpec("custom", "two_sided").needs_margin is False

    def test_is_frozen(self):
        spec = get_spec("superiority")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.needs_margin = True  # type: ignore[misc]

    def test_equal_specs_compare_equal_and_hash_alike(self):
        a = HypothesisSpec("equivalence", "equivalence", needs_margin=True)
        assert a == get_spec("equivalence")
        assert hash(a) == hash(get_spec("equivalence"))

    def test_margin_for_works_on_a_directly_built_spec(self):
        spec = HypothesisSpec("custom", "greater", needs_margin=True)
        assert spec.margin_for(0.25) == 0.25
        with pytest.raises(ValueError, match="'margin' must be"):
            spec.margin_for(0.0)
