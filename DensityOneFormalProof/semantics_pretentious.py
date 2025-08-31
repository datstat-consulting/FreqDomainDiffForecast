# semantics_pretentious.py
"""
Pretentious distance and Halász-style mean value evaluation for completely multiplicative f with |f|≤1.

This module computes:
- D(f, n^{it}; X)^2 = sum_{p≤X} (1 - Re(f(p) p^{-it}))/p
- A Halász-style bound evaluator for |(1/X) sum_{n≤X} f(n)| using a grid search in t.

For λ, f(p)=-1, so D^2(t) = sum_{p≤X} (1 + cos(t log p))/p ≈ 2 log log X.
This makes the bound tiny, matching cancellation heuristics.
"""

import math
from typing import Callable, List, Dict

def sieve_primes(n: int) -> List[int]:
    if n < 2: return []
    sieve = bytearray(b"\x01")*(n+1)
    sieve[0]=sieve[1]=0
    r = int(n**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:n+1:step] = b"\x00"*(((n - start)//step)+1)
    return [i for i in range(2, n+1) if sieve[i]]

def pretentious_distance_sq(f_on_prime: Callable[[int], complex], X: int, t: float) -> float:
    """
    D^2 = sum_{p≤X} (1 - Re(f(p) * p^{-it}))/p
    """
    ps = sieve_primes(X)
    acc = 0.0
    for p in ps:
        term = 1.0 - ( (f_on_prime(p) * complex(math.cos(-t*math.log(p)), math.sin(-t*math.log(p)))).real )
        acc += term / p
    return acc

def halasz_mean_value_bound(f_on_prime: Callable[[int], complex], X: int, t_grid: List[float]) -> Dict:
    """
    Evaluate a Halász-style envelope:
        B = min_{t in grid} exp(-D(f,n^{it};X)^2) + 1/sqrt(log X)
    (Up to constants; this is a *semantic evaluator*, not a theorem prover.)
    """
    if X < 3:
        return {"best_t": 0.0, "D2": 0.0, "bound": 1.0}
    best = None
    best_t = 0.0
    for t in t_grid:
        D2 = pretentious_distance_sq(f_on_prime, X, t)
        val = math.exp(-D2) + 1.0/(max(1.0, math.log(X))**0.5)
        if (best is None) or (val < best):
            best = val
            best_t = t
            best_D2 = D2
    return {"best_t": best_t, "D2": best_D2, "bound": best}

# Convenience: λ on primes
def liouville_on_prime(p: int) -> complex:
    return -1.0 + 0j

if __name__ == "__main__":
    X = 200000
    grid = [k*0.2 for k in range(-25, 26)]
    out = halasz_mean_value_bound(liouville_on_prime, X, grid)
    print("Halasz-style bound at X=", X, out)
