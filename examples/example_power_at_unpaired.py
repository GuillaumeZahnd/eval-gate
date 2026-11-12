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
    unpaired_std,
    unpaired_power,
    interpret_unpaired_power_at
)


if __name__ == "__main__":

    n_units_reference = 300
    n_units_candidate = 100
    n_resample = 1

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=n_units_reference,
        m=n_resample,
        distribution_between="gaussian",
        mu=5.0,
        std_between=4.0,
        seed=0,
    )

    candidate = simulate(
        n=n_units_candidate,
        m=n_resample,
        distribution_between="gaussian",
        mu=5.3,
        std_between=0.5,
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = unpaired_std(reference.y, candidate.y)


    # Question: what power does a study have?
    # Fixed: effect, margin, n_units, std, alpha. Solved for: power.
    result = unpaired_power(
        n_units_reference=n_units_reference,  # given: number of reference units available
        n_units_candidate=n_units_candidate,  # given: number of candidate units available     
        effect=0.05,  # assumed: the true difference
        margin=0.15,  # chosen: largest difference still considered equivalent
        std=std,  # estimated from pilot data
        alternative_hypothesis="superiority",  # chosen
        alpha=0.05,  # chosen: significance level
    )

    print(interpret_unpaired_power_at(result))
