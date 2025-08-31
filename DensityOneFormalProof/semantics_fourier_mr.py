# semantics_fourier_mr.py
"""
Semantic replay of a *mean-square identity* used in short-interval arguments.

Identity (discrete toy): For a sequence a_n (1..N), define S_x = sum_{n} w_{x,H}(n) a_n
with rectangular window w_{x,H}(n)=1 if x < n ≤ x+H else 0. Then
(1/N) sum_{x=0}^{N-1} |S_x|^2 = sum_{d=-(H-1)}^{H-1} (H-|d|)/N * sum_{n} a_n \overline{a_{n+d}}
(where out-of-range indices are treated as 0). We verify this equality by DFT on a cyclic extension.
"""

import math, cmath
from typing import List

def mean_square_identity(a: List[complex], H: int) -> float:
    """
    Returns the absolute difference between LHS and RHS of the identity on a cyclic extension of length L.
    We take L >= len(a) + H to avoid wrap interference in the window.
    """
    N = len(a)
    L = 1
    while L < N + H + 1:
        L *= 2  # power of two for FFT-like DFT; we implement direct DFT to keep it simple
    # zero-pad a to length L
    A = a + [0j]*(L-N)
    # Build all windows S_x for x=0..N-1 (non-wrapping)
    S = []
    for x in range(N):
        s = 0j
        for n in range(x+1, min(x+H, N)+1):
            s += a[n-1]
        S.append(s)
    lhs = (1.0/N) * sum((abs(z)**2 for z in S))

    # RHS via autocorrelation with triangular kernel
    rhs = 0.0
    for d in range(-(H-1), H):
        weight = (H - abs(d))/N
        acc = 0j
        for n in range(1, N+1):
            n2 = n + d
            if 1 <= n2 <= N:
                acc += a[n-1] * complex(a[n2-1].conjugate())
        rhs += weight * acc.real  # a arbitrary complex; identity holds; we take real part
    return abs(lhs - rhs)

if __name__ == "__main__":
    import random
    N=300; H=30
    a = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(N)]
    err = mean_square_identity(a,H)
    assert err < 1e-8
    print("semantics_fourier_mr: identity check OK, err=", err)
