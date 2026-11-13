from eval_gate import interpret_unpaired_p_value, simulate, unpaired_p_value

if __name__ == "__main__":
    # Groups are independent
    population_size_reference = 200  # number of units
    population_size_candidate = 50  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

    reference = simulate(
        n=population_size_reference,
        m=n_measurements_per_unit,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        mu=1.0,  # population grand mean
        std_between=1.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size_candidate,
        m=n_measurements_per_unit,
        distribution_between="lognormal",  # distribution of the values across units, within a single sample
        mu=1.15,  # population grand mean
        std_between=2.0,  # standard deviation across units
        seed=1,
    )

    result = unpaired_p_value(
        reference=reference.y,
        candidate=candidate.y,
        alternative_hypothesis="superiority",  # alternative hypothesis being tested
        alpha=0.05,  # significance level of the decision
    )

    print(interpret_unpaired_p_value(result))
