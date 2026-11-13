"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- population_size_reference, population_size_candidate: number of units in each independent group.
- std: variability of the data, here the pooled standard deviation of the two independent groups.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    interpret_unpaired_power,
    simulate,
    unpaired_power,
    unpaired_std,
)

if __name__ == "__main__":
    population_size_reference = 300  # number of units
    population_size_candidate = 100  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size_reference,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        mu=5.0,  # population grand mean
        std_between=4.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size_candidate,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        mu=5.3,  # population grand mean
        std_between=0.5,  # standard deviation across units
        seed=1,
    )

    # Variability: pooled standard deviation of the two independent groups.
    std = unpaired_std(reference.y, candidate.y)

    # Question: what power does a study have?
    # Fixed: effect, population_size, std, alpha. Solved for: power.
    result = unpaired_power(
        n_units_reference=population_size_reference,  # given: number of reference units available
        n_units_candidate=population_size_candidate,  # given: number of candidate units available
        effect=0.05,  # assumed: the true difference
        std=std,  # estimated from pilot data
        alternative_hypothesis="superiority",  # chosen
        alpha=0.05,  # chosen: significance level
    )

    print(interpret_unpaired_power(result))
