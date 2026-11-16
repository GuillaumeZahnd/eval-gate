# eval-gate: Hypothesis-first decision gating

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://guillaumezahnd.github.io/eval-gate/)

An open-source Python library for statistically rigorous comparison of a candidate against a reference. Every numerical result comes with a plain-language interpretation. Declare your hypothesis first, then let the data speak.

The library answers the following questions:

- **Is the candidate different from, higher than, not lower than (within a margin), or equivalent to (within a margin) the reference?** P-value for each hypothesis.
- **How large is the difference?** Effect size (Hedges' g) with its confidence interval.
- **How precise is my estimate?** Standard error of the mean (SEM) with its confidence interval.
- **How many units do I need?** Smallest number of units that detects a given true difference with a chosen probability (sample size).
- **How likely am I to detect a difference?** Probability that the test detects a given true difference with the number of units available (power analysis).

Every comparison comes in two versions:

- **Paired**: the same units are measured under both the reference and the candidate.
- **Unpaired**: the reference and the candidate are measured on independent units.

## Example: Estimating the required number of units in a paired test

Computes how many paired units a study needs for the test to detect a given true difference with a chosen probability (the power).

```python
"""example_required_n_paired.py"""

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

👉 See the [`examples/`](https://github.com/GuillaumeZahnd/eval-gate/tree/main/examples) folder for every statistical test: mean and confidence interval, p-values, effect sizes, required number of units and power, each in paired and unpaired form.

## Installation

### Prerequisites

- Python 3.12 or later.
- Dependencies, installed automatically: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [scikit-learn](https://scikit-learn.org/).

### Install

Install from GitHub with `pip`:

```sh
pip install git+https://github.com/GuillaumeZahnd/eval-gate
```

or with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/GuillaumeZahnd/eval-gate
```

## Documentation

The full API reference is available at [guillaumezahnd.github.io/eval-gate](https://guillaumezahnd.github.io/eval-gate/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the docstring conventions, and how to build the documentation.
