from eval_gate import simulate, compute_paired_effect_size, interpret_paired_effect_size


if __name__ == "__main__":

    population_size = 25
    n_resample = 1

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
        mu=1.25,
        std_between=2.0,
        seed=0,
    )

    result = compute_paired_effect_size(
        reference=reference.y,
        candidate=candidate.y,
        alpha=0.05,  # significance level of the decision
    )

    print(interpret_paired_effect_size(result))

