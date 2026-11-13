"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- population_size_reference, population_size_candidate: number of units in each independent group.
- std: pooled standard deviation of the two independent groups.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    interpret_unpaired_required_n,
    required_unpaired_n_units,
    simulate,
    unpaired_std,
)

if __name__ == "__main__":
    population_size_reference = 3_000  # number of units
    population_size_candidate = 6_000  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size_reference,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        mu=0.0,  # population grand mean
        std_between=4.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size_candidate,
        m=n_measurements_per_unit,
        distribution_between="lognormal",  # distribution of the values across units, within a single sample
        mu=0.2,  # population grand mean
        std_between=6.0,  # standard deviation across units
        seed=1,
    )

    # Variability: pooled standard deviation of the two independent groups.
    std = unpaired_std(reference.y, candidate.y)

    # Question: how many units are needed to detect an effect?
    # Fixed: effect, margin, std, alpha, power, allocation_ratio. Solved for: population_size_reference and population_size_candidate.
    result = required_unpaired_n_units(
        effect=0.1,  # assumed: the smallest true difference worth detecting
        std=std,  # estimated from pilot data
        alternative_hypothesis="equivalence",  # chosen: alternative hypothesis
        power=0.8,  # chosen: target power
        alpha=0.05,  # chosen: significance level
        margin=0.5,  # chosen: largest difference still considered equivalent
        allocation_ratio=1.0,  # chosen: ratio of candidate units to reference units
    )

    print(interpret_unpaired_required_n(result))
