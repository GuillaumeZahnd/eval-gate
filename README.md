# Hypothesis-first decision gating

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://guillaumezahnd.github.io/eval-gate/)

## Examples

### Estimating a mean with a confidence interval

```python
import numpy as np
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
