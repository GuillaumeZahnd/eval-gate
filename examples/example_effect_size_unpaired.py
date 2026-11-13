from eval_gate import (
    compute_unpaired_effect_size,
    interpret_unpaired_effect_size,
    simulate,
)

if __name__ == "__main__":
    # Groups are independent
    population_size_reference = 2_000  # number of units
    population_size_candidate = 700  # number of units
    n_measurements_per_unit_reference = 1  # number of measurements per unit (resample)
    n_measurements_per_unit_candidate = 3  # number of measurements per unit (resample)

    reference = simulate(
        n=population_size_reference,
        m=n_measurements_per_unit_reference,
        distribution_between="student_t",  # distribution of the values across units, within a single sample
        mu=80.0,  # population grand mean
        std_between=5.0,  # standard deviation across units
        seed=0,
    )

    candidate = simulate(
        n=population_size_candidate,
        m=n_measurements_per_unit_candidate,
        distribution_between="gaussian",  # distribution of the values across units, within a single sample
        distribution_within="gaussian",  # distribution of the repeated measurements (resamples)  
        mu=83.0,  # population grand mean
        std_between=2.0,  # standard deviation across units
        std_within=1.0,  # standard deviation across repeated measurements 
        seed=1,
    )

    result = compute_unpaired_effect_size(
        reference=reference.y,
        candidate=candidate.y,
        alpha=0.005,  # significance level of the confidence interval (99.5% here)
    )

    print(interpret_unpaired_effect_size(result))
