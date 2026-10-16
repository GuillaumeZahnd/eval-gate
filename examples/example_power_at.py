"""
Power analysis links five interdependent quantities: fixing any four determines the fifth.

- effect (delta): magnitude of the true difference to detect, delta = mu_candidate - mu_reference.
- nb_units (n): number of paired units.
- std: variability of the data, here the standard deviation of the per-unit paired differences.
- alpha: significance level, the false-positive rate (probability of rejecting the null when it is true).
- power (1 - beta): the true-positive rate (probability of rejecting the null when the effect is real).
"""

from eval_gate import (
    simulate,
    paired_std,
    paired_power,
    interpret_power_at
)    


if __name__ == "__main__":

    population_size = 200
    nb_resample = 1
    alternative_hypothesis = "equivalence"

    # Pilot data, used only to estimate the variability.
    reference = simulate(
        n=population_size,
        m=nb_resample,
        distribution_between="gaussian",
        mu=1.0,
        std_between=1.0,
        seed=0,
    )

    candidate = simulate(
        n=population_size,
        m=nb_resample,
        distribution_between="lognormal",
        mu=1.0,
        std_between=0.5,
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = paired_std(reference.y, candidate.y)

    # Question: what power does a study of the pilot's size (population_size units) have?
    # Fixed: effect, nb_units, std, alpha (projections, assumed before running the experiment, not measured from data).
    # Solved for: power.
    achieved = paired_power(
        nb_units=population_size,
        effect=0.05,
        margin=0.15,
        std=std,
        alternative_hypothesis=alternative_hypothesis,
        alpha=0.05,
    )
    print(interpret_power_at(achieved))
