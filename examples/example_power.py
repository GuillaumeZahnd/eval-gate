import math

from eval_gate import simulate, paired_power, required_nb_units, paired_std


if __name__ == "__main__":

    population_size = 200
    nb_resample = 1

    # ------------------------------------------------------------------
    # Pilot data: used only to estimate the standard deviation of the paired differences.
    # ------------------------------------------------------------------
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

    std = paired_std(reference.y, candidate.y)
    print(f"Pilot std of paired differences: {std:.3f}\n")

    # ------------------------------------------------------------------
    # 1. Sample size: how many units are needed to detect the planned effect?
    #    The PowerResult remembers the whole design (hypothesis, margin, effect, std, alpha).
    # ------------------------------------------------------------------
    design = required_nb_units(
        effect=0.2,
        std=paired_std(reference.y, candidate.y),
        alternative_hypothesis="superiority",
        power=0.8,
        alpha=0.05,
    )
    print(design)

    # Reuse the design conditions for every follow-up question, so the hypothesis,
    # margin, std and alpha can never drift away from the ones the design was made for.
    conditions = {
        "effect": design.effect,
        "std": design.std,
        "alternative_hypothesis": design.alternative_hypothesis,
        "margin": design.margin,
        "alpha": design.alpha,
    }

    # ------------------------------------------------------------------
    # 2. Power: the inverse question. With a given number of units, what is the
    #    probability of detecting the effect? paired_power returns a PowerAtResult,
    #    which carries its own design conditions, so it is self-describing too.
    #    Round trip: at the required number of units, power must match the design.
    # ------------------------------------------------------------------
    at_required = paired_power(nb_units=design.nb_units, **conditions)
    assert math.isclose(at_required.power, design.power)
    print(f"\n{at_required}")

    # ------------------------------------------------------------------
    # 3. Power as a function of the number of units, for the planned effect.
    # ------------------------------------------------------------------
    print(f"\nPower for effect={design.effect} ({design.alternative_hypothesis}, alpha={design.alpha}):")
    for nb_units in (50, 100, 200, 400, 800):
        result = paired_power(nb_units=nb_units, **conditions)
        print(f"  nb_units={result.nb_units:>4}: {result.power:.3f}")

    # ------------------------------------------------------------------
    # 4. Sensitivity: what if the true effect is smaller than planned?
    #    Only the effect changes; everything else comes from the design.
    # ------------------------------------------------------------------
    print(f"\nPower at nb_units={design.nb_units} if the true effect is smaller:")
    for smaller_effect in (0.1, 0.15, design.effect):
        result = paired_power(nb_units=design.nb_units, **{**conditions, "effect": smaller_effect})
        print(f"  effect={result.effect:<5}: {result.power:.3f}")
