from eval_gate import interpret_unpaired_p_value, simulate, unpaired_p_value

if __name__ == "__main__":
    # Groups are independent
    n_units_reference = 200
    n_units_candidate = 50

    n_resample = 1

    alternative_hypothesis = "superiority"

    reference = simulate(
        n=n_units_reference,
        m=n_resample,
        distribution_between="gaussian",
        mu=1.0,
        std_between=1.0,
        seed=0,
    )

    candidate = simulate(
        n=n_units_candidate,
        m=n_resample,
        distribution_between="lognormal",
        mu=1.15,
        std_between=2.0,
        seed=1,
    )

    result = unpaired_p_value(
        a=reference.y,
        b=candidate.y,
        alternative_hypothesis="superiority",  # alternative hypothesis being tested
        margin=1.0,  # minimum required increase
        alpha=0.05,  # significance level of the decision
    )

    print(interpret_unpaired_p_value(result))
