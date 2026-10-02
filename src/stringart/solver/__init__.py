from dataclasses import dataclass, field


@dataclass
class SolveResult:
    sequence: list[int]
    scores: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
