# Hypothesis-first decision gating

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://guillaumezahnd.github.io/eval-gate/)

## Examples

### Estimating a mean with a confidence interval

TODO

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

> [RESULT] The mean outcome is estimated at -0.096 (SEM 0.062, 500 units).
>
> [VERDICT] With 95% confidence, the true mean outcome lies between -0.218 and 0.025.


### Comparing a candidate to a reference (paired)

TODO

```python
from eval_gate import paired_p_value, interpret_paired__value

# reference, candidate: arrays of shape (200,), one paired measurement per unit
result = paired_p_value(
    reference,
    candidate,
    alternative_hypothesis="non_inferiority",  # alternative hypothesis being tested
    margin=0.1,  # largest acceptable drop
    alpha=0.05,  # significance level of the decision
)

print(interpret_paired_p_value(result))
```

> [RESULT] p = 0.002 (significance level 0.05). Observed difference (candidate - reference): 0.044 (SE 0.049, 200 units).
>
> [VERDICT] Non-inferiority established. A drop of 0.100 or more is ruled out; the candidate may be equal or higher.

### Comparing a candidate to a reference (unpaired)

TODO

```python
from eval_gate import unpaired_p_value, interpret_unpaired__value

# reference: array of shape (200,), one measurement per unit
# candidate: array of shape (50,), one measurement per unit
result = unpaired_p_value(
    reference,
    candidate,
    alternative_hypothesis="superiority",  # alternative hypothesis being tested
    margin=1.0,  # minimum required increase
    alpha=0.05,  # significance level of the decision
)

print(interpret_unpaired_p_value(result))
```

> [RESULT] p = 0.669 (significance level 0.05). Observed difference (candidate - reference): -0.110 (SE 0.253, 200 reference and 50 candidate units).
> 
> [VERDICT] Superiority not established. The candidate may still be higher, but the data are not strong enough to say so.

### Estimating a paired effect size

TODO

```python
from eval_gate import compute_paired_effect_size, interpret_paired_effect_size

# reference, candidate: arrays of shape (25,), one paired measurement per unit
result = compute_paired_effect_size(
    reference,
    candidate,
    alpha=0.05,  # significance level of the decision
)

print(interpret_paired_effect_size(result))
```

> [RESULT] Mean difference (candidate - reference): -0.011 (25 units). Hedges' g = -0.02, 95% CI [-0.40, 0.36].
>
> [VERDICT] Inconclusive. Plausible effects range from a small decrease to a small increase.
>
> [CAVEAT] Only 25 units (less than 30): the normal approximation can make the interval somewhat too narrow.


### Estimating an unpaired effect size

TODO

```python
from eval_gate import compute_unpaired_effect_size, interpret_unpaired_effect_size

# reference: array of shape (2_000,), one measurement per unit
# candidate: array of shape (700,), three measurements per unit
result = compute_paired_effect_size(
    reference,
    candidate,
    alpha=0.005,  # significance level of the decision
)

print(interpret_unpaired_effect_size(result))
```

> [RESULT] Mean difference (candidate - reference): 2.875 (2000 reference and 700 candidate units). Hedges' g = 0.70, 99.5% CI [0.58, 0.83].
>
> [VERDICT] Difference established. The candidate's mean outcome is higher than the reference's; plausible effects range from a medium to a large increase.

### Estimating the required number of units (paired)

TODO

```python
from eval_gate import paired_std, required_paired_n_units, interpret_paired_power

# reference, candidate: arrays of shape (200,), one paired measurement per unit
std = paired_std(reference, candidate)

# Question: how many units are needed to detect an effect?
# Fixed: effect, std, alpha, power. Solved for: nb_units.
result = required_nb_units(
    effect=0.15,                           # assumed: the smallest true difference worth detecting
    std=std,                               # estimated from pilot data
    alternative_hypothesis="superiority",  # chosen: alternative hypothesis
    power=0.8,                             # chosen: target power
    alpha=0.05,                            # chosen: significance level
)

print(interpret_paired_power(result))
```

> [RESULT] 134 units needed for at least 80% power at significance level 0.05. Assumed true difference (candidate - reference): 0.150; assumed standard deviation of the per-unit differences: 0.696 (standardized effect 0.22).
>
> [VERDICT] With 134 units, the test would show that the candidate's mean outcome is higher at least 80% of the time under these assumptions.
>
> [CAVEAT] The number is only as reliable as the assumed difference and standard deviation; a larger standard deviation needs more units.

### Estimating the required number of units (unpaired)

TODO

```python
from eval_gate import unpaired_std, required_unpaired_n_units, interpret_unpaired_power

# reference: array of shape (3_000,), one measurement per unit
# candidate: array of shape (6_000,), one measurements per unit
std = unpaired_std(reference, candidate)

# Question: how many units are needed to detect an effect?
# Fixed: effect, std, alpha, power. Solved for: nb_units.
result = required_unpaired_n_units(
    effect=0.1,                            # assumed: the smallest true difference worth detecting
    std=std,                               # estimated from pilot data
    alternative_hypothesis="equivalence",  # chosen: alternative hypothesis
    power=0.8,                             # chosen: target power
    alpha=0.05,                            # chosen: significance level
    margin=0.5,                            # TODO
)

print(interpret_unpaired_power(result))
```

> [RESULT] 2212 reference and 2212 candidate units needed for at least 80% power at significance level 0.05. Assumed true difference (candidate - reference): 0.100; assumed pooled standard deviation: 5.234 (standardized effect 0.02).
>
> [VERDICT] With 2212 reference and 2212 candidate units, the test would show that the difference in mean outcome is within ±0.500 at least 80% of the time under these assumptions.
>
> [CAVEAT] The number is only as reliable as the assumed difference and standard deviation; a larger standard deviation needs more units.

### Estimating the power level (paired)

TODO

```python
from eval_gate paired_std, import paired_power, interpret_paired_power_at

# reference, candidate: arrays of shape (200,), 20 paired measurement per unit
std = paired_std(reference, candidate)

# Question: what power does a study have?
# Fixed: effect, margin, nb_units, std, alpha. Solved for: power.
result = paired_power(
    nb_units=population_size,  # given: number of units available
    effect=0.05,                           # assumed: the true difference
    margin=0.15,                           # chosen: largest difference still considered equivalent
    std=std,                               # estimated from pilot data
    alternative_hypothesis="equivalence",  # chosen: alternative hypothesis
    alpha=0.05,                            # chosen: significance level
)

print(interpret_paired_power_at(result))
```

> [RESULT] Power 88.9% with 400 units at significance level 0.05. Assumed true difference (candidate - reference): 0.050; assumed standard deviation of the per-unit differences: 0.698 (standardized effect 0.07).
>
> [VERDICT] Adequate. Under these assumptions, the test would show that the difference in mean outcome is within ±0.150 88.9% of the time (at least the usual 80%).
>
> [CAVEAT] The power is only as reliable as the assumed difference and standard deviation.

### Estimating the power level (unpaired)

TODO

```python
from eval_gate import unpaired_std, unpaired_power, interpret_unpaired_power_at

# reference: array of shape (300,), one measurement per unit
# candidate: array of shape (600,), one measurements per unit
std = unpaired_std(reference, candidate)

# Question: what power does a study have?
# Fixed: effect, margin, nb_units, std, alpha. Solved for: power.
result = unpaired_power(
    n_units_reference=n_units_reference,  # given: number of reference units available
    n_units_candidate=n_units_candidate,  # given: number of candidate units available
    effect=0.05,  # assumed: the true difference
    margin=0.15,  # chosen: largest difference still considered equivalent
    std=std,  # estimated from pilot data
    alternative_hypothesis="superiority",  # chosen
    alpha=0.05,  # chosen: significance level
)

print(interpret_unpaired_power_at(result))
```

> [RESULT] Power 6.3% with 300 reference and 100 candidate units at significance level 0.05. Assumed true difference (candidate - reference): 0.050; assumed pooled standard deviation: 3.710 (standardized effect 0.01).
>
> [VERDICT] Badly underpowered. Under these assumptions, the test would fail to show that the candidate's mean outcome is higher 93.7% of the time, more often than it succeeds.
>
> [CAVEAT] The power is only as reliable as the assumed difference and standard deviation.

## 📖 Documentation

This project uses [MkDocs](https://www.mkdocs.org/) with `mkdocstrings` for API reference generation. Documentation is automatically built and published to GitHub Pages via GitHub Actions whenever changes are merged into the `main` branch.

### Local development

To build and preview the documentation locally:

1. **Clone the repository and sync dependencies** (including the docs group):

```sh
git clone [https://github.com/GuillaumeZahnd/eval-gate](https://github.com/GuillaumeZahnd/eval-gate)
cd eval-gate
uv sync --group docs
```

2. Run the live-reload development server:

```sh
uv run mkdocs serve
```

3. Open `http://127.0.0.1:8000` in your browser.

4. Verify the build (runs in strict mode to catch broken links or invalid docstrings):

```sh
uv run mkdocs build --strict
```

## Writing docstrings

When adding or updating Python functions, please use NumPy-style docstrings so they render properly in the API reference:

```python
def example_function(parameter: type) -> return_type:
    """Short summary of the function.

    Parameters
    ----------
    parameter : type
        Description of the parameter.

    Returns
    -------
    return_type
        Description of the return value.
    """
```
