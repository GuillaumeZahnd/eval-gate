import numpy as np
import pytest


from eval_gate import standardized_distribution, DISTRIBUTIONS


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_standardized_distribution_moments(distribution: str) -> None:
    """Verifies that generated distributions approximate mean=0 and variance=1."""
    rng = np.random.default_rng(0)
    n = 50_000
    samples = standardized_distribution(
        distribution=distribution,
        nb_units=n,
        rng=rng
    )
    assert samples.shape == (n,)
    assert np.isclose(np.mean(samples), 0.0, atol=0.1)
    assert np.isclose(np.var(samples), 1.0, atol=0.1)


def test_standardized_distribution_multidimensional() -> None:
    """Verifies support for multi-dimensional shapes (e.g., 2D resamples)."""
    rng = np.random.default_rng(0)
    samples = standardized_distribution(
        distribution="gaussian",
        nb_units=(50, 10),
        rng=rng
    )
    assert samples.shape == (50, 10)
