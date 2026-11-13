from eval_gate import compute_paired_effect_size, interpret_paired_effect_size, simulate

if __name__ == "__main__":
    population_size = 25  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

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
        mu=1.25,  # population grand mean
        std_between=2.0,  # standard deviation across units
        seed=0,
    )

    result = compute_paired_effect_size(
        reference=reference.y,
        candidate=candidate.y,
        alpha=0.05,  # significance level of the confidence interval
    )

    print(interpret_paired_effect_size(result))
