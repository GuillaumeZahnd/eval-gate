"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- n_units: number of units in each independent group.
- std: pooled standard deviation of the two independent groups.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    simulate,
    unpaired_std,
    required_unpaired_n_units,
    interpret_unpaired_power,
)


if __name__ == "__main__":

    n_units_reference = 3_000
    n_units_candidate = 6_000
    n_resample = 1

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=n_units_reference,
        m=n_resample,
        distribution_between="gaussian",
        mu=0.0,
        std_between=4.0,
        seed=0,
    )

    candidate = simulate(
        n=n_units_candidate,
        m=n_resample,
        distribution_between="lognormal",
        mu=0.2,
        std_between=6.0,
        seed=0,
    )

    # Variability: pooled standard deviation of the two independent groups.
    std = unpaired_std(reference.y, candidate.y)

    # Question: how many units are needed to detect an effect?
    # Fixed: effect, margin, std, alpha, power, allocation_ratio. Solved for: n_units_reference and n_units_candidate.
    result = required_unpaired_n_units(
        effect=0.1,  # assumed: the smallest true difference worth detecting
        std=std,  # estimated from pilot data
        alternative_hypothesis="equivalence",  # chosen: alternative hypothesis
        power=0.8,  # chosen: target power
        alpha=0.05,  # chosen: significance level
        margin=0.5,  # chosen: largest difference still considered equivalent
        allocation_ratio=1.0,  # chosen: ratio of candidate units to reference units
    )

    print(interpret_unpaired_power(result))
