# semantics_measure.py
"""
Continuous analytic semantics (initial build with rigorous-leaning enclosures).

- Interval arithmetic for +,-,*,/,exp,log,sin,cos with outward padding.
- Exact min/max for sin/cos on intervals using critical points.
- Exact Fourier orthogonality on [0,1] for exponentials.
- Simple-function integral for step functions.

This is enough to start backing Parseval/orthogonality and basic integral claims.
"""

from dataclasses import dataclass
from typing import Tuple, List, Callable
import math

PAD = 1e-15  # outward safety pad for float rounding (conservative)

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    def width(self) -> float: return self.hi - self.lo
    def pad(self, eps: float = PAD) -> 'Interval':
        return Interval(self.lo - eps - eps*abs(self.lo), self.hi + eps + eps*abs(self.hi))

def I(a: float, b: float) -> Interval:
    lo, hi = (a,b) if a <= b else (b,a)
    return Interval(lo, hi)

def Iadd(x: Interval, y: Interval) -> Interval:
    return Interval(x.lo + y.lo, x.hi + y.hi).pad()

def Isub(x: Interval, y: Interval) -> Interval:
    return Interval(x.lo - y.hi, x.hi - y.lo).pad()

def Imul(x: Interval, y: Interval) -> Interval:
    cands = [x.lo*y.lo, x.lo*y.hi, x.hi*y.lo, x.hi*y.hi]
    return Interval(min(cands), max(cands)).pad()

def Irecip(x: Interval) -> Interval:
    assert not (x.lo <= 0.0 <= x.hi), "Interval crosses 0; cannot invert."
    cands = [1.0/x.lo, 1.0/x.hi]
    lo, hi = min(cands), max(cands)
    return Interval(lo, hi).pad()

def Idiv(x: Interval, y: Interval) -> Interval:
    return Imul(x, Irecip(y)).pad()

def Iexp(x: Interval) -> Interval:
    return Interval(math.exp(x.lo), math.exp(x.hi)).pad()

def Ilog(x: Interval) -> Interval:
    assert x.lo > 0.0, "log requires positive interval"
    return Interval(math.log(x.lo), math.log(x.hi)).pad()

def _sin_cos_extrema(f: str, a: float, b: float) -> Tuple[float,float]:
    """
    Exact min/max of sin or cos on [a,b] using critical points.
    """
    import math
    if f == 'sin':
        # derivative cos=0 at x = pi/2 + k*pi
        def val(x): return math.sin(x)
        crit = []
        k_start = math.ceil((a - math.pi/2.0)/math.pi)
        k_end   = math.floor((b - math.pi/2.0)/math.pi)
        for k in range(int(k_start), int(k_end)+1):
            x = math.pi/2.0 + k*math.pi
            if a <= x <= b: crit.append(x)
    else:
        # derivative -sin=0 at x = k*pi
        def val(x): return math.cos(x)
        crit = []
        k_start = math.ceil(a/math.pi)
        k_end   = math.floor(b/math.pi)
        for k in range(int(k_start), int(k_end)+1):
            x = k*math.pi
            if a <= x <= b: crit.append(x)
    pts = [a,b] + crit
    vals = [val(t) for t in pts]
    return min(vals), max(vals)

def Isin(x: Interval) -> Interval:
    lo, hi = _sin_cos_extrema('sin', x.lo, x.hi)
    return Interval(lo, hi).pad()

def Icos(x: Interval) -> Interval:
    lo, hi = _sin_cos_extrema('cos', x.lo, x.hi)
    return Interval(lo, hi).pad()

# Simple step-function integral on [0,1]
@dataclass
class SimpleFn:
    parts: List[Tuple[Interval, float]]  # value on each subinterval
def integrate_simple(f: SimpleFn) -> float:
    total = 0.0
    for I, c in f.parts:
        total += c * max(0.0, I.width())
    return total

# Exact Fourier orthogonality on [0,1]:
def fourier_orthogonality(n: int, m: int) -> complex:
    """
    ∫_0^1 e^{2π i n x} \overline{e^{2π i m x}} dx
    = ∫_0^1 e^{2π i (n-m) x} dx = 1_{n=m}.
    """
    k = n - m
    if k == 0: return 1.0 + 0j
    # integral of exp(2π i k x) from 0 to 1 = (e^{2π i k}-1)/(2π i k)=0 exactly since k∈Z\{0}
    return 0.0 + 0j

if __name__ == "__main__":
    # quick checks
    assert fourier_orthogonality(3,3) == 1.0+0j
    assert fourier_orthogonality(3,4) == 0.0+0j
    # trig enclosures sanity
    r = Isin(I(0, math.pi))
    assert r.lo <= 0.0 <= r.hi and r.hi >= 1.0 - 1e-12
    print("semantics_measure: OK")
