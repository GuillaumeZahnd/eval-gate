from eval_gate import interpret_paired_p_value, paired_p_value, simulate

if __name__ == "__main__":
    population_size = 200  # number of units
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
        mu=1.15,  # population grand mean
        std_between=2.0,  # standard deviation across units
        seed=0,
    )

    result = paired_p_value(
        reference=reference.y,
        candidate=candidate.y,
        alternative_hypothesis="non_inferiority",  # alternative hypothesis being tested
        margin=0.1,  # largest acceptable drop
        alpha=0.05,  # significance level of the decision
    )

    print(interpret_paired_p_value(result))
