from eval_gate import simulate, compute_sem


if __name__ == "__main__":

    population_size = 500
    nb_resample = 10

    measurements = simulate(
        n=population_size,
        m=nb_resample,
    )

    sem_results = compute_sem(
        outcomes=measurements.y,
    )

