# semantics_ext.py
"""
Extended semantic checks and rigorous-leaning enclosures.

Provides:
- Interval-based integral enclosures via partitioning.
- Semantic inequality checkers: Cauchy–Schwarz, Chebyshev-style tail (empirical witness),
  and Bilinear dispersion (geometric-series magnitude vs x^{1/2}(PQ)^{1/2}(log x)^A).
"""

from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
import math

# Basic interval as in semantics_measure
@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    def width(self) -> float: return self.hi - self.lo

def I(a: float, b: float) -> Interval:
    lo, hi = (a,b) if a<=b else (b,a)
    return Interval(lo, hi)

def min_max_on_interval(f: Callable[[float], float], a: float, b: float, critical_points: List[float]=None) -> Tuple[float,float]:
    """
    Compute conservative min/max by sampling endpoints + any provided critical points.
    This is not fully rigorous for arbitrary f; users should provide critical points
    where possible (e.g., sin/cos). We add a small grid to avoid missing features.
    """
    pts = [a,b]
    if critical_points:
        for x in critical_points:
            if a <= x <= b:
                pts.append(x)
    # light uniform sampling for safety
    G = 8
    h = (b-a)/G if b>a else 0.0
    for k in range(1, G):
        pts.append(a + k*h)
    vals = [f(x) for x in pts]
    return min(vals), max(vals)

def integrate_enclosure(f: Callable[[float], float], a: float, b: float, n: int,
                        crit: Callable[[float,float], List[float]] = None) -> Interval:
    """
    Enclose ∫_a^b f(x) dx by subdividing [a,b] into n slices and bounding f on each slice
    with min/max sampling (plus user-supplied critical points if provided).
    """
    if n <= 0: n = 1
    h = (b-a)/n
    lo_sum = 0.0
    hi_sum = 0.0
    for i in range(n):
        L = a + i*h
        R = L + h
        cps = crit(L,R) if crit else None
        m, M = min_max_on_interval(f, L, R, cps)
        lo_sum += m*h
        hi_sum += M*h
    # small outward pad
    pad = 1e-12 * abs(hi_sum) + 1e-12
    return Interval(lo_sum - pad, hi_sum + pad)

# --------- Semantic inequality checkers ---------

def check_cauchy_schwarz(vec1: List[complex], vec2: List[complex]) -> bool:
    """Verify |<v1,v2>|^2 ≤ ||v1||^2 ||v2||^2 numerically (semantic inequality)."""
    import cmath
    dot = sum((a.conjugate()*b for a,b in zip(vec1, vec2)))
    lhs = abs(dot)**2
    n1 = sum((abs(a)**2 for a in vec1))
    n2 = sum((abs(b)**2 for b in vec2))
    return lhs <= (n1 + 1e-12) * (n2 + 1e-12)

def check_chebyshev_tail(values: List[float], T: float, allowed_fraction: float) -> bool:
    """
    Empirical witness for Chebyshev-type tail: fraction(|value|>T) ≤ allowed_fraction.
    Callers should set T according to theory (e.g., H^{1/2}(log N)^C) and allowed_fraction small.
    """
    if len(values)==0: return True
    bad = sum(1 for v in values if abs(v) > T)
    frac = bad / len(values)
    return frac <= allowed_fraction + 1e-9

def geom_sum_abs(M: int, denom: int, c: int=1) -> float:
    """|sum_{m=1}^M e(2π i c m / denom)| = |sin(π c M/den) / sin(π c/den)|, careful if denominator divides c."""
    num = abs(math.sin(math.pi * c * M / denom))
    den = abs(math.sin(math.pi * c / denom))
    if den == 0.0:
        return float(M)
    return num/den

def primes_in_range(lo: int, hi: int) -> List[int]:
    if hi <= 1 or hi <= lo: return []
    sieve = [True]*(hi+1)
    sieve[0]=sieve[1]=False
    r = int(hi**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            for j in range(p*p, hi+1, p):
                sieve[j]=False
    return [p for p in range(max(2,lo+1), hi+1) if sieve[p]]

def check_bilinear_dispersion_bound(x: int, P: int, Q: int, A: float=2.0, c: int=1, slack: float=1.0) -> Dict:
    """
    Compute S = sum_{p in (P,2P]} sum_{q in (Q,2Q]} |sum_{m <= x/(pq)} e(c m / (pq))|
    and compare to slack * x^{1/2} (PQ)^{1/2} (log x)^A.
    Returns dict with ratio and boolean 'ok'.
    """
    ps = primes_in_range(P, 2*P)
    qs = primes_in_range(Q, 2*Q)
    S = 0.0
    for p in ps:
        for q in qs:
            M = x // (p*q)
            if M <= 0: continue
            S += geom_sum_abs(int(M), p*q, c)
    bench = (x**0.5) * ((P*Q)**0.5) * (math.log(x)**A) if x>1 else 1.0
    ratio = S / bench if bench>0 else float('inf')
    return {"S": S, "bench": bench, "ratio": ratio, "ok": (ratio <= slack)}
