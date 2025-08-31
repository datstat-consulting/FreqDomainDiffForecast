# semantics_mr_witness.py
"""
Numerical MR inequality witness for λ:
We compute the average short-interval mean square
    (1/N) ∑_{x≤N} |∑_{x<n≤x+H} λ(n)|^2
and compare it to H / (log N)^A for modest N (semantic witness only).
"""

import math
from typing import List, Dict

def sieve_lambda(N: int) -> List[int]:
    spf = [0]*(N+1)
    for i in range(2, N+1):
        if spf[i]==0:
            spf[i]=i
            if i*i <= N:
                for j in range(i*i, N+1, i):
                    if spf[j]==0: spf[j]=i
    lam = [0]*(N+1); lam[1]=1
    for x in range(2, N+1):
        p=spf[x]; y=x//p; k=1
        while y%p==0:
            y//=p; k+=1
        lam[x] = lam[y] * ((-1)**k)
    return lam

def average_short_interval_msq_lambda(N: int, H: int) -> float:
    lam = sieve_lambda(N+H+5)
    # compute block sums and average of squares for x=0..N-1 (window (x, x+H])
    total = 0.0
    for x in range(N):
        s = 0
        L = x+1
        R = min(x+H, N)
        if L<=R:
            for n in range(L, R+1):
                s += lam[n]
        total += (s*s)
    return total / N

def mr_bound_witness(N: int, H: int, A: float=2.0) -> Dict:
    lhs = average_short_interval_msq_lambda(N, H)
    rhs = H / (max(2.0, math.log(N))**A)
    return {"lhs": lhs, "rhs": rhs, "ratio": lhs/max(1e-9, rhs), "ok": lhs <= rhs}

if __name__ == "__main__":
    N = 200000; H = int(N**0.5); A=2.0
    out = mr_bound_witness(N,H,A)
    print("MR witness:", out)
