import math
import re
import pytest
import numpy as np

from eval_gate._checks import (
    check_in_open_interval,
    check_probability,
    check_finite_non_negative,
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
        with pytest.raises(ValueError, match=rf"^'{name}' must be finite and non-negative"):
            check_finite_non_negative(-1.0, name)

    @pytest.mark.parametrize("value, shown", [(-1.5, "-1.5"), (math.inf, "inf"), (-math.inf, "-inf"), (math.nan, "nan")])
    def test_error_message_format(self, value, shown):
        with pytest.raises(ValueError) as excinfo:
            check_finite_non_negative(value, "se")
        assert str(excinfo.value) == f"'se' must be finite and non-negative, got {shown}."

    def test_raises_exactly_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            check_finite_non_negative(-1.0, "se")
        assert type(excinfo.value) is ValueError

