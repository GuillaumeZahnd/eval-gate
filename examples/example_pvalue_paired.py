from eval_gate import simulate, paired_p_value, interpret_paired_p_value


if __name__ == "__main__":

    population_size = 200
    alternative_hypothesis = "superiority"
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
        mu=1.15,
        std_between=2.0,
        seed=0,
    )

    result = paired_p_value(
        a=reference.y,
        b=candidate.y,
        alternative_hypothesis=alternative_hypothesis,
    )

    print(interpret_paired_p_value(result))
