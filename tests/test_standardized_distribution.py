import numpy as np
import pytest

from eval_gate import standardized_distribution, DISTRIBUTIONS


def test_distributions_constant():
    """Verify that the available distributions constant matches the expected set of supported types."""
    assert set(DISTRIBUTIONS) == {"gaussian", "uniform", "student_t", "lognormal", "bimodal"}


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_standardized_distribution_moments(distribution: str) -> None:
    """Verify that the generated distributions approximate mean=0 and variance=1."""
    samples = standardized_distribution(distribution, 50_000, np.random.default_rng(0))
    assert np.isclose(np.mean(samples), 0.0, atol=0.1)
    assert np.isclose(np.var(samples), 1.0, atol=0.1)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_multidimensional_shape(distribution):
    """Verify that generated distributions correctly handle 1D and 2D shape specifications."""
    assert standardized_distribution(distribution, (50, 10), np.random.default_rng(0)).shape == (50, 10)
    assert standardized_distribution(distribution, (50,), np.random.default_rng(0)).shape == (50,)
    assert standardized_distribution(distribution, 50, np.random.default_rng(0)).shape == (50,)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_dtype(distribution):
    """Verify that generated sample arrays use a floating-point data type."""
    out = standardized_distribution(distribution, (50, 10), np.random.default_rng(0))
    assert np.issubdtype(out.dtype, np.floating)


@pytest.mark.parametrize("unsupported", ["cauchy", ""])
def test_unsupported_distribution_raises(unsupported):
    """Verify that requesting an unsupported distribution name raises a ValueError."""
    with pytest.raises(ValueError, match="'distribution' must be one of"):
        standardized_distribution(unsupported, 10, np.random.default_rng(0))


@pytest.mark.parametrize("invalid_units", [0, -1, (0, 5), (5, -1), (-2,)])
def test_invalid_nb_units_raises(invalid_units):
    """Verify that passing non-positive values or shapes for units raises a ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        standardized_distribution("gaussian", invalid_units, rng)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_reproducible_with_seed(distribution):
    """Verify that identical random seeds produce identical outputs and distinct seeds produce different outputs."""
    a = standardized_distribution(distribution, 100, np.random.default_rng(1))
    b = standardized_distribution(distribution, 100, np.random.default_rng(2))
    c = standardized_distribution(distribution, 100, np.random.default_rng(1))
    np.testing.assert_array_equal(a, c)
    assert not np.array_equal(a, b)
