from eval_gate import simulate, unpaired_p_value, interpret_unpaired_p_value


if __name__ == "__main__":

    # Groups are independent
    n_units_reference = 200
    n_units_candidate = 50
    
    n_resample = 1    
    
    alternative_hypothesis = "non_inferiority"

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
        alternative_hypothesis=alternative_hypothesis,
        margin=1.0,
    )

    print(interpret_unpaired_p_value(result))
