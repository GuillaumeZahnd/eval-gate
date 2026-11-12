from eval_gate import simulate, compute_sem, interpret_sem


if __name__ == "__main__":

    population_size = 500
    n_resample = 1

    measurements = simulate(
        n=population_size,
        m=n_resample,
        seed=0,
    )

    result = compute_sem(
        outcomes=measurements.y,
        confidence=0.95,  # confidence level of the interval
        alternative_hypothesis="two_sided"  # sidedness of the interval
    )

    print(interpret_sem(result))
