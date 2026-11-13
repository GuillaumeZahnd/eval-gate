# Hypothesis-first decision gating

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://guillaumezahnd.github.io/eval-gate/)

## Example: Estimating the required number of units in a paired test

Computes how many paired units a study needs for the test to detect a given true difference with a chosen probability (the power).

```python
# example_required_n_paired.py
from eval_gate import paired_std, required_paired_n_units, interpret_paired_required_n

# reference, candidate: arrays of shape (200,), one paired measurement per unit
std = paired_std(reference, candidate)

# Question: how many units are needed to detect an effect?
# Fixed: effect, std, alpha, power. Solved for: n_units.
result = required_paired_n_units(
    effect=0.15,  # assumed: the smallest true difference worth detecting
    std=std,  # estimated from pilot data
    alternative_hypothesis="superiority",  # chosen: alternative hypothesis
    power=0.8,  # chosen: target power
    alpha=0.05,  # chosen: significance level
)

print(interpret_paired_required_n(result))
```

> [RESULT] 134 units needed for at least 80% power at significance level 0.05. Assumed true difference (candidate - reference): 0.150; assumed standard deviation of the per-unit differences: 0.696 (standardized effect 0.22).
>
> [VERDICT] With 134 units, the test would show that the candidate's mean outcome is higher at least 80% of the time under these assumptions.
>
> [CAVEAT] The number is only as reliable as the assumed difference and standard deviation; a larger standard deviation needs more units.

👉 See the [`examples/`](https://github.com/GuillaumeZahnd/eval-gate/tree/main/examples) folder for every test: mean and confidence interval, p-values, effect sizes, required number of units and power, each in paired and unpaired form.

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
