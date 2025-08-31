# rigorous_interval.py
"""
Rigorous-leaning interval arithmetic using directed rounding (float nextafter).
For monotone functions (exp/log) we round outward; for sin/cos we provide both
exact-extrema enclosures and Taylor-series remainder enclosures on small radii.
"""

import math
from dataclasses import dataclass

def down(x: float) -> float:
    return math.nextafter(x, -math.inf)

def up(x: float) -> float:
    return math.nextafter(x, math.inf)

@dataclass(frozen=True)
class I:
    lo: float
    hi: float
    def width(self) -> float: return self.hi - self.lo
    def __repr__(self): return f"[{self.lo}, {self.hi}]"

# Basic ops
def i_add(a: I, b: I) -> I:
    return I(down(a.lo + b.lo), up(a.hi + b.hi))

def i_sub(a: I, b: I) -> I:
    return I(down(a.lo - b.hi), up(a.hi - b.lo))

def i_mul(a: I, b: I) -> I:
    cands = [a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi]
    return I(down(min(cands)), up(max(cands)))

def i_recip(a: I) -> I:
    assert not (a.lo <= 0.0 <= a.hi), "interval crosses zero"
    vals = [1.0/a.lo, 1.0/a.hi]
    return I(down(min(vals)), up(max(vals)))

def i_div(a: I, b: I) -> I:
    return i_mul(a, i_recip(b))

# Monotone funcs
def i_exp(a: I) -> I:
    return I(down(math.exp(a.lo)), up(math.exp(a.hi)))

def i_log(a: I) -> I:
    assert a.lo > 0.0, "log needs positive interval"
    return I(down(math.log(a.lo)), up(math.log(a.hi)))

# Exact extrema for sin/cos using critical points
def _sin_cos_extrema(func: str, lo: float, hi: float):
    import math
    pts = [lo, hi]
    if func == "sin":
        k0 = math.ceil((lo - math.pi/2)/math.pi)
        k1 = math.floor((hi - math.pi/2)/math.pi)
        for k in range(int(k0), int(k1)+1):
            pts.append(math.pi/2 + k*math.pi)
        f = math.sin
    else:
        k0 = math.ceil(lo/math.pi)
        k1 = math.floor(hi/math.pi)
        for k in range(int(k0), int(k1)+1):
            pts.append(k*math.pi)
        f = math.cos
    vals = [f(x) for x in pts]
    return min(vals), max(vals)

def i_sin_ext(a: I) -> I:
    lo, hi = _sin_cos_extrema("sin", a.lo, a.hi)
    return I(down(lo), up(hi))

def i_cos_ext(a: I) -> I:
    lo, hi = _sin_cos_extrema("cos", a.lo, a.hi)
    return I(down(lo), up(hi))

# Taylor-based enclosure around center c with radius r (assumes |x-c| ≤ r ≤ 1 for tightness)
def _taylor_bounds_sin(c: float, r: float):
    import math
    # sin(x) = sin c + cos c*(x-c) - sin c*(x-c)^2/2 - cos c*(x-c)^3/6 + R4
    sc = math.sin(c); cc = math.cos(c)
    # bound |R4| ≤ |x-c|^4/24 since |sin^{(4)}| = |sin| ≤ 1
    M4 = 1.0
    def poly(dx): return sc + cc*dx - sc*(dx**2)/2 - cc*(dx**3)/6
    P_lo = poly(-r); P_hi = poly(r)
    R = (r**4)/24.0
    return (down(min(P_lo,P_hi) - R), up(max(P_lo,P_hi) + R))

def _taylor_bounds_cos(c: float, r: float):
    import math
    # cos(x) = cos c - sin c*(x-c) - cos c*(x-c)^2/2 + sin c*(x-c)^3/6 + R4
    sc = math.sin(c); cc = math.cos(c)
    M4 = 1.0
    def poly(dx): return cc - sc*dx - cc*(dx**2)/2 + sc*(dx**3)/6
    P_lo = poly(-r); P_hi = poly(r)
    R = (r**4)/24.0
    return (down(min(P_lo,P_hi) - R), up(max(P_lo,P_hi) + R))

def i_sin_series(a: I) -> I:
    c = 0.5*(a.lo + a.hi)
    r = 0.5*(a.hi - a.lo)
    lo, hi = _taylor_bounds_sin(c, r)
    return I(lo, hi)

def i_cos_series(a: I) -> I:
    c = 0.5*(a.lo + a.hi)
    r = 0.5*(a.hi - a.lo)
    lo, hi = _taylor_bounds_cos(c, r)
    return I(lo, hi)

# Minimal self-test
if __name__ == "__main__":
    x = I(0.0, 1.0)
    y = I(2.0, 3.0)
    assert i_add(x,y).lo <= 3.0 <= i_add(x,y).hi
    s_ext = i_sin_ext(I(0.0, math.pi))
    assert s_ext.lo <= 0.0 and s_ext.hi >= 1.0
    print("rigorous_interval: OK")
