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
    required_nb_units,
    interpret_power,
)    


if __name__ == "__main__":

    population_size = 200
    nb_resample = 1
    alternative_hypothesis = "superiority"

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
        mu=1.15,
        std_between=2.0,
        seed=0,
    )

    # Variability: standard deviation of the per-unit paired differences.
    std = paired_std(reference.y, candidate.y)

    # Question: how many units are needed to detect an effect of 0.2 at a significance level of 0.05 with 80% power?
    # Fixed: effect, std, alpha, power (projections, assumed before running the experiment, not measured from data).
    # Solved for: nb_units.
    design = required_nb_units(
        effect=0.4,
        std=std,
        alternative_hypothesis=alternative_hypothesis,
        power=0.8,
        alpha=0.05,
    )

    print(interpret_power(design))

