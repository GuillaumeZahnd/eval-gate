from eval_gate import simulate, compute_sem, interpret_sem


if __name__ == "__main__":

    population_size = 500
    nb_resample = 1

    measurements = simulate(
        n=population_size,
        m=nb_resample,
        seed=0,
    )

    result = compute_sem(
        outcomes=measurements.y,
    )

    print(interpret_sem(result))
    
    from plot_simu import plot_simulation
    plot_simulation(measurements)
