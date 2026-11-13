"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- population_size (n): number of paired units.
- std: variability of the data, here the standard deviation of the per-unit paired differences.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import interpret_paired_power, paired_power, paired_std, simulate

if __name__ == "__main__":
    population_size = 400  # number of units
    n_measurements_per_unit = 20  # number of measurements per unit (resample)

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        distribution_within="gaussian",  # distribution of the repeated measurements (resamples)  
        mu=1.0,  # population grand mean
        std_between=1.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size,
        m=n_measurements_per_unit,
        distribution_between="lognormal",  # distribution of the values across units, within a single sample
        distribution_within="gaussian",  # distribution of the repeated measurements (resamples)  
        mu=1.0,  # population grand mean
        std_between=0.5,  # standard deviation across units
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = paired_std(reference.y, candidate.y)

    # Question: what power does a study have?
    # Fixed: effect, margin, population_size, std, alpha. Solved for: power.
    result = paired_power(
        n_units=population_size,  # given: number of units available
        effect=0.05,  # assumed: the true difference
        margin=0.15,  # chosen: largest difference still considered equivalent
        std=std,  # estimated from pilot data
        alternative_hypothesis="equivalence",  # chosen
        alpha=0.05,  # chosen: significance level
    )

    print(interpret_paired_power(result))
