from dataclasses import dataclass
from typing import Literal, get_args


# Alternative hypotheses; comments give the null (Δ = true mean(Candidate - Reference), m = margin > 0).
AlternativeHypothesis = Literal[
    "two_sided",        # H0: Δ = 0
    "superiority",      # H0: Δ ≤ 0
    "non_inferiority",  # H0: Δ ≤ -m
    "equivalence",      # H0: |Δ| ≥ m
]
ALTERNATIVE_HYPOTHESES: tuple[str, ...] = get_args(AlternativeHypothesis)

# Geometry of the test; hypotheses that differ only by margin share a direction.
Direction = Literal["two_sided", "greater", "equivalence"]

@dataclass(frozen=True)
class HypothesisSpec:
    name: str
    direction: Direction
    needs_margin: bool = False

    def margin_for(self, margin: float) -> float:
        """Effective margin: 0 if unused, validated positive if required."""
        if not self.needs_margin:
            return 0.0
        if not margin > 0.0:
            raise ValueError(f"'{self.name}' requires a positive 'margin', got {margin}.")
        return float(margin)


_SPECS = {s.name: s for s in (
    HypothesisSpec("two_sided", "two_sided"),
    HypothesisSpec("superiority", "greater"),
    HypothesisSpec("non_inferiority", "greater", needs_margin=True),
    HypothesisSpec("equivalence", "equivalence", needs_margin=True),
)}

if set(_SPECS) != set(ALTERNATIVE_HYPOTHESES):  # fail fast if the table drifts from the Literal
    raise RuntimeError("HypothesisSpec table is out of sync with AlternativeHypothesis.")


def get_spec(alternative_hypothesis: str) -> HypothesisSpec:
    """Return the spec for ``alternative_hypothesis`` or raise ``ValueError``."""
    try:
        return _SPECS[alternative_hypothesis]
    except (KeyError, TypeError):
        raise ValueError(
            f"'alternative_hypothesis' must be one of {ALTERNATIVE_HYPOTHESES}, got {alternative_hypothesis!r}."
        ) from None
