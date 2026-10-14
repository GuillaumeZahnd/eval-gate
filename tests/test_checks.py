import math
import re
import pytest

from eval_gate._checks import (
    check_in_open_interval,
    check_probability,
)


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
        assert str(excinfo.value) == "my_param must be in (0.0, 1.0), got 2.5."

    def test_error_message_uses_custom_bounds(self):
        with pytest.raises(ValueError, match=re.escape("(-3, 3)")):
            check_in_open_interval(3, "x", -3, 3)

    def test_empty_interval_rejects_everything(self):
        with pytest.raises(ValueError):
            check_in_open_interval(0.5, "x", 1.0, 0.0)
        with pytest.raises(ValueError):
            check_in_open_interval(0.5, "x", 0.5, 0.5)


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
        with pytest.raises(ValueError, match=rf"^{name} must be in"):
            check_probability(1.5, name)

    def test_error_message_format(self):
        with pytest.raises(ValueError) as excinfo:
            check_probability(1.5, "confidence")
        assert str(excinfo.value) == "confidence must be in (0.0, 1.0), got 1.5."

