"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- population_size (n): number of paired units.
- std: variability of the data, here the standard deviation of the per-unit paired differences.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    interpret_paired_required_n,
    paired_std,
    required_paired_n_units,
    simulate,
)

if __name__ == "__main__":
    population_size = 200  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        mu=1.0,  # population grand mean
        std_between=1.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size,
        m=n_measurements_per_unit,
        distribution_between="lognormal",  # distribution of the values across units, within a single sample
        mu=1.15,  # population grand mean
        std_between=2.0,  # standard deviation across units
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = paired_std(reference.y, candidate.y)

    # Question: how many units are needed to detect an effect?
    # Fixed: effect, std, alpha, power. Solved for: population_size.
    result = required_paired_n_units(
        effect=0.15,  # assumed: the smallest true difference worth detecting
        std=std,  # estimated from pilot data
        alternative_hypothesis="superiority",  # chosen
        power=0.8,  # chosen: target power
        alpha=0.05,  # chosen: significance level
    )

    print(interpret_paired_required_n(result))
