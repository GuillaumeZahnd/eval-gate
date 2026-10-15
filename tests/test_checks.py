import math
import re
import pytest
import numpy as np

from eval_gate._checks import (
    check_in_open_interval,
    check_probability,
    check_min_units,
    check_finite,
    check_non_negative,    
    check_finite_non_negative,
    check_all_finite,
)


# ----------------------------------------------------------------
class TestCheckInOpenInterval:
    @pytest.mark.parametrize(
        "value, low, high",
        [
            (0.5, 0.0, 1.0),
            (1e-12, 0.0, 1.0),
            (1.0 - 1e-12, 0.0, 1.0),
            (-0.5, -1.0, 0.0),
            (5, 0, 10),
            (0.0, -1.0, 1.0),
        ],
    )
    def test_accepts_values_strictly_inside(self, value, low, high):
        assert check_in_open_interval(value, "x", low, high) is None

    @pytest.mark.parametrize(
        "value, low, high",
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (-0.1, 0.0, 1.0),
            (1.5, 0.0, 1.0),
            (-math.inf, 0.0, 1.0),
            (math.inf, 0.0, 1.0),
            (-1.0, -1.0, 0.0),
            (10, 0, 10),
        ],
    )
    def test_rejects_boundaries_and_outside(self, value, low, high):
        with pytest.raises(ValueError):
            check_in_open_interval(value, "x", low, high)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="nan"):
            check_in_open_interval(float("nan"), "x", 0.0, 1.0)

    def test_error_message_contains_name_bounds_and_value(self):
        with pytest.raises(ValueError) as excinfo:
            check_in_open_interval(2.5, "my_param", 0.0, 1.0)
        assert str(excinfo.value) == "'my_param' must be in (0.0, 1.0), got 2.5."

    def test_error_message_uses_custom_bounds(self):
        with pytest.raises(ValueError, match=re.escape("(-3, 3)")):
            check_in_open_interval(3, "x", -3, 3)

    def test_empty_interval_rejects_everything(self):
        with pytest.raises(ValueError):
            check_in_open_interval(0.5, "x", 1.0, 0.0)
        with pytest.raises(ValueError):
            check_in_open_interval(0.5, "x", 0.5, 0.5)


# ----------------------------------------------------------------
class TestCheckProbability:
    @pytest.mark.parametrize("value", [1e-12, 0.01, 0.05, 0.5, 0.95, 0.999, 1.0 - 1e-12])
    def test_accepts_values_in_open_unit_interval(self, value):
        assert check_probability(value, "confidence") is None

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, 95, -math.inf, math.inf, float("nan")])
    def test_rejects_values_outside_open_unit_interval(self, value):
        with pytest.raises(ValueError):
            check_probability(value, "confidence")

    @pytest.mark.parametrize("name", ["confidence", "alpha", "power"])
    def test_error_message_contains_parameter_name(self, name):
        with pytest.raises(ValueError, match=rf"^'{name}' must be in"):
            check_probability(1.5, name)

    def test_error_message_format(self):
        with pytest.raises(ValueError) as excinfo:
            check_probability(1.5, "confidence")
        assert str(excinfo.value) == "'confidence' must be in (0.0, 1.0), got 1.5."


# ----------------------------------------------------------------
class TestCheckFiniteNonNegative:
    @pytest.mark.parametrize("value", [0.0, -0.0, 1e-300, 0.5, 1.0, 1e300, 3, 0, np.float64(0.2), np.float32(0.2)])
    def test_accepts_finite_non_negative(self, value):
        assert check_finite_non_negative(value, "se") is None

    @pytest.mark.parametrize("value", [-1e-12, -0.5, -1.0, -1e300, -3, np.float64(-0.2)])
    def test_rejects_negative(self, value):
        with pytest.raises(ValueError):
            check_finite_non_negative(value, "se")

    @pytest.mark.parametrize("value", [math.inf, -math.inf, np.inf, -np.inf])
    def test_rejects_infinite(self, value):
        with pytest.raises(ValueError):
            check_finite_non_negative(value, "se")

    @pytest.mark.parametrize("value", [math.nan, np.nan, np.float64("nan")])
    def test_rejects_nan(self, value):
        # NaN fails `isfinite`; the explicit test documents that this is intended.
        with pytest.raises(ValueError, match="nan"):
            check_finite_non_negative(value, "se")

    @pytest.mark.parametrize("name", ["se", "sem", "std"])
    def test_error_message_contains_parameter_name(self, name):
        with pytest.raises(ValueError, match=rf"^'{name}' must be"):
            check_finite_non_negative(-1.0, name)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (-1.5, "'se' must be non-negative, got -1.5."),
            (math.inf, "'se' must be finite, got inf."),
            (-math.inf, "'se' must be finite, got -inf."),
            (math.nan, "'se' must be finite, got nan."),
        ],
    )
    def test_error_message_format(self, value, expected):
        with pytest.raises(ValueError) as excinfo:
            check_finite_non_negative(value, "se")
        assert str(excinfo.value) == expected

    def test_raises_exactly_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            check_finite_non_negative(-1.0, "se")
        assert type(excinfo.value) is ValueError


# ----------------------------------------------------------------
class TestCheckMinUnits:
    @pytest.mark.parametrize("n", [2, 3, 1000])
    def test_accepts_enough_units(self, n):
        assert check_min_units(n, "the standard error", minimum=2) is None

    @pytest.mark.parametrize("n", [0, 1])
    def test_rejects_too_few_units(self, n):
        with pytest.raises(ValueError):
            check_min_units(n, "the standard error", minimum=2)

    def test_custom_minimum(self):
        assert check_min_units(5, "x", minimum=5) is None
        with pytest.raises(ValueError):
            check_min_units(4, "x", minimum=5)

    def test_error_message_format(self):
        with pytest.raises(ValueError) as excinfo:
            check_min_units(1, "the standard deviation", minimum=2)
        assert str(excinfo.value) == (
            "At least 2 units are required to compute the standard deviation, got 1."
        )


# ----------------------------------------------------------------
class TestCheckNonNegative:
    @pytest.mark.parametrize("value", [0.0, -0.0, 1e-300, 1.0, 1e300, 0, 3, math.inf])
    def test_accepts_non_negative_including_inf(self, value):
        assert check_non_negative(value, "x") is None

    @pytest.mark.parametrize("value", [-1e-12, -1.0, -1e300, -math.inf, -3])
    def test_rejects_negative(self, value):
        with pytest.raises(ValueError):
            check_non_negative(value, "x")

    def test_rejects_nan(self):
        # `nan < 0.0` is False, so a naive `if value < 0.0` would let NaN through.
        with pytest.raises(ValueError, match="nan"):
            check_non_negative(math.nan, "x")

    def test_error_message_format(self):
        with pytest.raises(ValueError) as excinfo:
            check_non_negative(-2.0, "x")
        assert str(excinfo.value) == "'x' must be non-negative, got -2.0."
        
        
# ----------------------------------------------------------------
class TestCheckFinite:
    @pytest.mark.parametrize(
        "value",
        [0.0, -0.0, 0.2, -0.2, 1e300, -1e300, 1e-300, 0, 3, -3, np.float64(0.2), np.float32(-0.2)],
    )
    def test_accepts_finite_values(self, value):
        assert check_finite(value, "effect") is None

    @pytest.mark.parametrize("value", [math.inf, -math.inf, np.inf, -np.inf])
    def test_rejects_infinite(self, value):
        with pytest.raises(ValueError):
            check_finite(value, "effect")

    @pytest.mark.parametrize("value", [math.nan, np.nan, np.float64("nan")])
    def test_rejects_nan(self, value):
        with pytest.raises(ValueError, match="nan"):
            check_finite(value, "effect")

    @pytest.mark.parametrize(
        "value, shown", [(math.inf, "inf"), (-math.inf, "-inf"), (math.nan, "nan")]
    )
    def test_error_message_format(self, value, shown):
        with pytest.raises(ValueError) as excinfo:
            check_finite(value, "effect")
        assert str(excinfo.value) == f"'effect' must be finite, got {shown}."

    def test_negative_values_are_allowed(self):
        assert check_finite(-5.0, "effect") is None
        

# ----------------------------------------------------------------
class TestCheckAllFinite:
    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0, -1.5, 2.5, 1e300, -1e300]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2D
            np.array([1, 2, 3]),  # integers are always finite
            np.array([]),  # nothing to reject
            [1.0, 2.0, 3.0],  # array-likes are accepted
            np.float32([0.5, 1.5]),
        ],
    )
    def test_accepts_finite_arrays(self, values):
        assert check_all_finite(values, "x") is None
 
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_a_single_non_finite_value(self, bad):
        with pytest.raises(ValueError):
            check_all_finite(np.array([1.0, bad, 3.0]), "x")
 
    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_rejects_wherever_the_bad_value_is(self, position):
        values = np.array([1.0, 2.0, 3.0])
        values[position] = math.nan
        with pytest.raises(ValueError):
            check_all_finite(values, "x")
 
    def test_rejects_non_finite_value_in_2d_array(self):
        values = np.array([[1.0, 2.0], [3.0, math.inf]])
        with pytest.raises(ValueError):
            check_all_finite(values, "x")
 
    def test_rejects_when_every_value_is_non_finite(self):
        with pytest.raises(ValueError):
            check_all_finite(np.array([math.nan, math.inf, -math.inf]), "x")
 
    @pytest.mark.parametrize("name", ["differences", "reference", "paired differences"])
    def test_error_message_contains_parameter_name(self, name):
        with pytest.raises(ValueError, match=rf"^'{name}' must contain only finite values"):
            check_all_finite(np.array([1.0, math.nan]), name)
 
    @pytest.mark.parametrize(
        "values, nb_bad, size",
        [
            (np.array([1.0, math.nan, 3.0]), 1, 3),
            (np.array([math.nan, math.inf, -math.inf, 4.0]), 3, 4),
            (np.array([[1.0, math.inf], [math.nan, 4.0]]), 2, 4),
        ],
    )
    def test_error_message_format_reports_how_many_values_are_bad(self, values, nb_bad, size):
        with pytest.raises(ValueError) as excinfo:
            check_all_finite(values, "x")
        assert str(excinfo.value) == (
            f"'x' must contain only finite values, got {nb_bad} non-finite value(s) out of {size}."
        )
 
    def test_does_not_emit_numpy_warnings(self, recwarn):
        with pytest.raises(ValueError):
            check_all_finite(np.array([math.inf, -math.inf, math.nan]), "x")
        assert len(recwarn) == 0
 
    def test_does_not_modify_its_input(self):
        values = np.array([1.0, 2.0, 3.0])
        check_all_finite(values, "x")
        assert np.array_equal(values, [1.0, 2.0, 3.0])
 
    def test_raises_exactly_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            check_all_finite(np.array([math.nan]), "x")
        assert type(excinfo.value) is ValueError
        
