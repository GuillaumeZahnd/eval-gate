from eval_gate import simulate, paired_p_value


if __name__ == "__main__":

    population_size = 200
    nb_resample = 1

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

    p_value = paired_p_value(
        a=reference.y,
        b=candidate.y,
        alternative_hypothesis="superiority",
    )

    print(p_value)
