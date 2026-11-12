from eval_gate import (
    compute_unpaired_effect_size,
    interpret_unpaired_effect_size,
    simulate,
)

if __name__ == "__main__":
    # Groups are independent
    n_units_reference = 2_000
    n_units_candidate = 700
    n_resample_reference = 1
    n_resample_candidate = 3

    reference = simulate(
        n=n_units_reference,
        m=n_resample_reference,
        distribution_between="student_t",
        mu=80.0,
        std_between=5.0,
        seed=0,
    )

    candidate = simulate(
        n=n_units_candidate,
        m=n_resample_candidate,
        distribution_between="gaussian",
        mu=83.0,
        std_between=2.0,
        std_within=1.0,
        seed=0,
    )

    result = compute_unpaired_effect_size(
        reference=reference.y,
        candidate=candidate.y,
        alpha=0.005,  # significance level of the decision
    )

    print(interpret_unpaired_effect_size(result))
