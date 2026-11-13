from eval_gate import compute_sem, interpret_sem, simulate

if __name__ == "__main__":
    population_size = 500  # number of units
    n_measurements_per_unit = 1  # number of measurements per unit (resample)

    measurements = simulate(
        n=population_size,
        m=n_measurements_per_unit,
        seed=0,
    )

    result = compute_sem(
        outcomes=measurements.y,
        confidence=0.95,  # confidence level of the interval
        alternative_hypothesis="two_sided",  # sidedness of the interval
    )

    print(interpret_sem(result))
