# Hypothesis-first decision gating

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://guillaumezahnd.github.io/eval-gate/)

## Examples

### Estimating a mean with a confidence interval

```python
from eval_gate import compute_sem, interpret_sem

# outcomes: array of shape (500,)
result = compute_sem(
  outcomes,
  confidence=0.95,  # confidence level of the interval
  alternative_hypothesis="two_sided"  # sidedness of the interval
)

print(interpret_sem(result))
```

> The mean outcome is estimated at -0.031 (SEM 0.048, 500 units). With 95% confidence, the true mean outcome lies between -0.125 and 0.063.

### Comparing a candidate to a reference (paired)

```python
from eval_gate import paired_p_value, interpret_p_value

# reference, candidate: arrays of shape (200,), one paired measurement per unit
result = paired_p_value(
    reference,
    candidate,
    alternative_hypothesis="non_inferiority",  # alternative hypothesis being tested
    margin=0.1,  # largest acceptable drop
    alpha=0.05,  # significance level of the decision
)

print(interpret_p_value(result))
```

> p = 0.002 (significance level 0.05): the null hypothesis is rejected. This p-value is the probability, if the candidate were worse than the reference by 0.100 or more, of a result at least as favourable to the alternative as the one observed. A drop of 0.100 or more is ruled out. The candidate may be equal or better; this test does not distinguish between the two. The observed difference (candidate - reference) is 0.044 (SE 0.049, 200 units). A small p-value shows that a difference is detectable, not that it is large.

### Estimating the paired effect size

```python
from eval_gate compute_paired_effect_size, interpret_effect_size

# reference, candidate: arrays of shape (1000,), one paired measurement per unit
result = compute_paired_effect_size(
    reference,
    candidate,
    alpha=0.05,  # significance level of the decision
)

print(interpret_effect_size(result))
```

> The candidate's mean outcome is 0.184 higher than the reference's on average (1000 units), a negligible standardized effect (Hedges' g = 0.17, 95% CI [0.11, 0.23]).
>
> Conclusion: difference established — the candidate is higher, but its size is unclear and may be negligible.
>
> Zero lies outside the 95% interval, so 'no difference' is ruled out. Plausible effects range from a negligible to a small increase.
>
> About the test: Hedges' g is the mean paired difference (0.184) divided by the standard deviation of the per-unit differences (1.081), then corrected for small-sample bias; its interval uses a normal approximation, and the size labels are Cohen's rules of thumb (0.2, 0.5, 0.8).

### Estimating the required number of units

```python
from eval_gate required_nb_units, interpret_power

# Question: how many units are needed to detect an effect?
# Fixed: effect, std, alpha, power. Solved for: nb_units.
design = required_nb_units(
    effect=0.15,                           # assumed: the smallest true difference worth detecting
    std=std,                               # estimated from pilot data
    alternative_hypothesis="superiority",  # chosen: alternative hypothesis
    power=0.8,                             # chosen: target power
    alpha=0.05,                            # chosen: significance level
)

print(interpret_power(design))
```

> 134 units are needed to show that the candidate's mean outcome is higher, with at least 80% power at significance level 0.05.
>
> Assumptions: a true difference (candidate - reference) of 0.150 and a standard deviation of the per-unit differences of 0.696 (standardized effect 0.22).

### Estimating the power level

```python
from eval_gate paired_power, interpret_power_at

# Question: what power does a study have?
# Fixed: effect, margin, nb_units, std, alpha. Solved for: power.
achieved = paired_power(
    nb_units=population_size,  # given: number of units available
    effect=0.05,                           # assumed: the true difference
    margin=0.15,                           # chosen: largest difference still considered equivalent
    std=std,                               # estimated from pilot data
    alternative_hypothesis="equivalence",  # chosen: alternative hypothesis
    alpha=0.05,                            # chosen: significance level
)

print(interpret_power_at(achieved))
```

> With 200 units, the test has 65.6% power to show that the difference in mean outcome is within ±0.150, at significance level 0.05.
>
> Conclusion: underpowered. The assumed difference would be missed 34.4% of the time (the usual target is 80% power).
>
> Assumptions: a true difference (candidate - reference) of 0.050 and a standard deviation of the per-unit differences of 0.685 (standardized effect 0.07).

