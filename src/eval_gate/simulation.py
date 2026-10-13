from dataclasses import dataclass
import numpy as np

from .standardized_distribution import standardized_distribution, Distributions


@dataclass(frozen=True)
class Simulation:
    """Synthetic nested evaluation data with its ground-truth parameters.

    Attributes
    ----------
    y : np.ndarray
        Measurements, of shape ``(n, m)``.
    mu_i : np.ndarray
        True mean of each unit, of shape ``(n,)``.
    grand_mean : float
        True population mean.
    var_between : float
        True variance of the unit means.
    var_within : float
        True average within-unit variance.
    """
    y: np.ndarray
    mu_i: np.ndarray
    grand_mean: float
    var_between: float
    var_within: float

    def __post_init__(self) -> None:
        # Enforce true immutability on the underlying NumPy arrays
        self.y.flags.writeable = False
        self.mu_i.flags.writeable = False

    @property
    def true_se(self) -> float:
        """Standard deviation of the grand mean across repeated experiments."""
        n, m = self.y.shape
        return float(np.sqrt((self.var_between + self.var_within / m) / n))


def simulate(
    n: int,
    m: int,
    *,
    distribution_between: Distributions = "gaussian",
    distribution_within: Distributions = "gaussian",
    mu: float = 0.0,
    std_between: float = 1.0,
    std_within: float = 1.0,
    heteroscedastic: bool = False,
    seed: int | None = None,
) -> Simulation:
    """Generate synthetic nested evaluation data with controllable variance structures.
        
    Parameters
    ----------
    n : int
        Number of units.
    m : int
        Number of resamples per unit.
    distribution_between : str, default="gaussian"
        Distribution type for between-unit means.
    distribution_within : str, default="gaussian"
        Distribution type for within-unit noise.
    mu : float, default=0.0
        Population grand mean.
    std_between : float, default=1.0
        Standard deviation between units.
    std_within : float, default=1.0
        Standard deviation within units.
    heteroscedastic : bool, default=False
        If True, assigns varying noise levels across units using a Gamma distribution.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    Simulation
        A frozen simulation data container.     
    """
    rng = np.random.default_rng(seed)

    # Generate standardized unit means and scale by std_between
    mu_i = mu + std_between * standardized_distribution(distribution_between, n, rng)

    std_i = np.full(n, std_within)
    if heteroscedastic:
        # Variance factor w ~ Gamma(2, 0.5) has mean 1, preserving average variance
        std_i = std_within * np.sqrt(rng.gamma(2.0, 0.5, size=n))

    # Generate 2D measurements matrix of shape (n, m)
    noise = standardized_distribution(distribution_within, (n, m), rng)
    y = mu_i[:, None] + std_i[:, None] * noise

    return Simulation(
        y=y,
        mu_i=mu_i,
        grand_mean=mu,
        var_between=std_between**2,
        var_within=std_within**2,
    )
