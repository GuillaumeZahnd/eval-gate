"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- n_units (n): number of paired units.
- std: variability of the data, here the standard deviation of the per-unit paired differences.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    simulate,
    paired_std,
    paired_power,
    interpret_paired_power_at
)


if __name__ == "__main__":

    population_size = 400
    n_resample = 20

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size,
        m=n_resample,
        distribution_between="gaussian",
        mu=1.0,
        std_between=1.0,
        seed=0,
    )

    candidate = simulate(
        n=population_size,
        m=n_resample,
        distribution_between="lognormal",
        mu=1.0,
        std_between=0.5,
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = paired_std(reference.y, candidate.y)

    # Question: what power does a study have?
    # Fixed: effect, margin, n_units, std, alpha. Solved for: power.
    result = paired_power(
        n_units=population_size,  # given: number of units available
        effect=0.05,  # assumed: the true difference
        margin=0.15,  # chosen: largest difference still considered equivalent
        std=std,  # estimated from pilot data
        alternative_hypothesis="equivalence",  # chosen
        alpha=0.05,  # chosen: significance level
    )

    print(interpret_paired_power_at(result))
