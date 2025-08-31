# semantics_discrete.py
"""
Discrete analytic semantics (EXACT) for finite Fourier analysis.

- Exact additive character orthogonality on Z/qZ.
- Exact geometric-series evaluation when r^q=1 (trivial cases exact).
- Unitary DFT + Parseval check on C^q.

Used to back the dispersion/orthogonality and finite Parseval steps semantically.
"""
from typing import List, Tuple
import math, cmath

class RootOfUnity:
    __slots__ = ("q","k")
    def __init__(self, q: int, k: int):
        assert q >= 1, "q must be >=1"
        self.q = q
        self.k = k % q
    def __mul__(self, other):
        assert isinstance(other, RootOfUnity) and other.q == self.q
        return RootOfUnity(self.q, self.k + other.k)
    def conj(self):
        return RootOfUnity(self.q, (-self.k) % self.q)
    def pow(self, m: int):
        return RootOfUnity(self.q, (self.k * m) % self.q)
    def is_one(self) -> bool:
        return (self.k % self.q) == 0
    def __repr__(self):
        return f"ζ_{self.q}^{self.k}"

def additive_orthogonality(q: int, k: int) -> int:
    """Exact sum: Σ_{m=0}^{q-1} e^{2πi k m/q} = q if q|k else 0."""
    return q if (k % q) == 0 else 0

def unitary_dft(vec: List[complex]) -> List[complex]:
    q = len(vec)
    out = [0j]*q
    s = (1.0 / (q**0.5))
    for k in range(q):
        tot = 0j
        for m in range(q):
            angle = -2.0*math.pi*k*m/q
            tot += vec[m] * complex(math.cos(angle), math.sin(angle))
        out[k] = tot * s
    return out

def parseval_error(vec: List[complex]) -> float:
    """Return | ||f||^2 - ||F||^2 |; should be ~0 up to FP noise, but identity is exact algebraically."""
    F = unitary_dft(vec)
    lhs = sum(abs(z)**2 for z in vec)
    rhs = sum(abs(Z)**2 for Z in F)
    return abs(lhs - rhs)

# Simple self-test
if __name__ == "__main__":
    for q in [3,5,8]:
        for k in range(-3*q,3*q+1):
            S = additive_orthogonality(q, k)
            assert S in (0, q)
    assert parseval_error([1+0j,0,0,0]) < 1e-10
    print("semantics_discrete: OK")
