# constants_enclosures.py
"""
Analytic constants & basic enclosures used for heuristic/pretentious layers.
These are *soft* enclosures; for rigorous work tighten using certified computations.

- Euler–Mascheroni γ in [0.57721, 0.57722] (crude but safe for our purposes here).
- Harmonic number bounds via integrals: H_n ∈ [log(n+1), 1 + log n].
- Meissel–Mertens (prime) constant B1 ≈ 0.2614972128... (soft interval [0.26,0.27]).
"""

import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

gamma = Interval(0.57721, 0.57722)
B1_prime = Interval(0.26, 0.27)  # soft

def Hn_bounds(n: int) -> Interval:
    if n <= 0:
        return Interval(0.0, 0.0)
    lo = math.log(n+1.0)
    hi = 1.0 + math.log(n)
    return Interval(lo, hi)

def loglogx_enclosure(x: float) -> Interval:
    assert x > 1.0
    val = math.log(math.log(x))
    # simple outward pad
    return Interval(val - 1e-12, val + 1e-12)
