# numerics.py
"""
Numerics that mirror each step of the density-one Liouville pair-correlation proof.
This file is **not** part of the formal proof checker — it illustrates scales/phenomena
used in the argument and produces CSVs and plots for your paper.

What it shows (matches paper sections):
1) Square–divisor identity sanity check (Lemma: λ(n)=Σ_{d^2|n} μ(n/d^2)).
2) Type I (MR + Chebyshev + blocks): H = N^s (default s=1/2), threshold T = H^{1/2} (log N)^C.
   - For each b ≤ N^{1/2−η}, compute T_b = Σ_{m≤M} λ(m)λ(bm+h), M≈N/b.
   - Partition into blocks of length H; count blocks with |block sum| > T (exceptional).
   - Compare |T_b| to the theoretical scale N^{3/4}/b · (log N)^C.
3) Type I H-scaling: vary s in [s_min, s_max], compare max block sums to H^{1/2}(log N)^C.
4) Type II/III toy dispersion (geometric-series magnitude) vs x^{1/2}(PQ)^{1/2}(log x)^A.

Usage:
    python numerics.py --N 200000 --h 1 --eta 0.10 --C 2.0 --s 0.5 --outdir out --run-all
    # or individual parts:
    python numerics.py --N 200000 --h 1 --eta 0.10 --C 2.0 --s 0.5 --typeI
    python numerics.py --N 200000 --h 1 --C 2.0 --sweep-H
    python numerics.py --dispersion

Outputs (saved to --outdir, default: current dir):
  CSVs: typeI_block_stats_*.csv, typeI_H_scaling_*.csv, typeII_dispersion_toy_*.csv
  Plots: plot_typeI_ratio_vs_b.png, plot_typeI_excfrac_vs_b.png, plot_typeI_H_scaling.png, plot_typeII_toy_ratio.png
"""

import math, argparse, os
from math import log
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# ----------------------------
# Basic number-theoretic sieves
# ----------------------------

def sieve_spf(n: int) -> List[int]:
    """Return smallest prime factor for every 0..n (spf[0]=spf[1]=0)."""
    spf = [0]*(n+1)
    for i in range(2, n+1):
        if spf[i] == 0:
            spf[i] = i
            if i*i <= n:
                step = i
                start = i*i
                for j in range(start, n+1, step):
                    if spf[j] == 0:
                        spf[j] = i
    return spf

def sieve_lambda_mu(n: int) -> Tuple[List[int], List[int]]:
    """Compute Liouville lambda(n)=(-1)^Ω(n) and Möbius μ(n) for n<=N using spf sieve."""
    spf = sieve_spf(n)
    lam = [0]*(n+1)
    mu  = [0]*(n+1)
    lam[1] = 1
    mu[1]  = 1
    for x in range(2, n+1):
        p = spf[x]
        y = x // p
        k = 1
        while y % p == 0:
            y //= p
            k += 1
        lam[x] = lam[y] * ((-1)**k)
        mu[x] = 0 if k >= 2 else -mu[y]
    return lam, mu

def lambda_via_square_divisor(n: int, mu: List[int]) -> int:
    """Compute λ(n) by Σ_{d^2|n} μ(n/d^2)."""
    s = 0
    d = 1
    while d*d <= n:
        if n % (d*d) == 0:
            s += mu[n // (d*d)]
        d += 1
    return s

# ----------------------------
# Type I experiments
# ----------------------------

def typeI_block_stats(N: int, h: int = 1, eta: float = 0.1, C: float = 2.0, s: float = 0.5):
    """
    For b ≤ N^(1/2 - eta):
       - a_m = λ(m)λ(bm+h), M≈N/b
       - Blocks of length H=⌊N^s⌋
       - Threshold T = H^(1/2)*(log N)^C
       - Count exceptional blocks (|block sum|>T) and compute T_b
       - Compare |T_b| against N^(3/4)/b * (log N)^C (when s=1/2)
    Returns dict with per-b rows and exception stats.
    """
    H = max(1, int(N**s))
    T = (H**0.5) * (log(N)**C)
    bmax = max(1, int(N**(0.5 - eta)))

    lam, mu = sieve_lambda_mu(N + abs(h) + 5)
    rows = []
    total_blocks = 0
    total_exc = 0

    for b in range(1, bmax+1):
        M = N // b
        seq = [0]*(M+1)  # 1-indexed
        ok = True
        for m in range(1, M+1):
            idx = b*m + h
            if idx <= 0 or idx >= len(lam):
                ok = False
                break
            seq[m] = lam[m] * lam[idx]
        if not ok:
            continue

        ps = [0]*(M+1)
        for i in range(1, M+1):
            ps[i] = ps[i-1] + seq[i]

        R = (M + H - 1)//H  # ceil
        exc = 0
        max_block_abs = 0.0
        for j in range(R):
            L = j*H + 1
            Rr = min((j+1)*H, M)
            ssum = ps[Rr] - ps[L-1]
            if abs(ssum) > max_block_abs:
                max_block_abs = abs(ssum)
            if abs(ssum) > T:
                exc += 1

        total_blocks += R
        total_exc += exc
        T_b = ps[M]
        scale = (N**0.75)/(b) * (log(N)**C) if abs(s-0.5) < 1e-12 else None
        ratio = (abs(T_b)/scale) if scale else None
        rows.append({
            "b": b, "M": M, "R_blocks": R, "T_b": T_b,
            "scale_N34_over_b_logC_if_s_half": scale,
            "ratio_if_s_half": ratio,
            "exc_blocks": exc, "exc_fraction": exc/max(1,R),
            "max_block_abs": max_block_abs, "H": H, "T": T
        })

    return {
        "N": N, "h": h, "eta": eta, "C": C, "s": s,
        "H": H, "T": T, "bmax": bmax, "rows": rows,
        "total_blocks": total_blocks,
        "total_exceptions": total_exc,
        "global_exception_fraction": (total_exc / max(1,total_blocks))
    }

def typeI_H_scaling(N: int, b: int = 1, h: int = 1, C: float = 2.0, s_min: float=0.30, s_max: float=0.70, s_step: float=0.05):
    """
    Fix b and vary H = ⌊N^s⌋, s in [s_min, s_max]. For each H compute the max block sum of a_m
    and compare to H^(1/2)(log N)^C.
    Returns a list of dicts with s, H, max_block_abs, predicted_T, ratio.
    """
    lam, mu = sieve_lambda_mu(N + abs(h) + 5)
    M = N // b
    seq = [0]*(M+1)
    for m in range(1, M+1):
        idx = b*m + h
        if idx <= 0 or idx >= len(lam):
            raise ValueError("Index out of range; increase sieve limit.")
        seq[m] = lam[m] * lam[idx]
    ps = [0]*(M+1)
    for i in range(1, M+1):
        ps[i] = ps[i-1] + seq[i]

    out = []
    k = 0
    s = s_min
    while s <= s_max + 1e-12:
        H = max(1, int(N**s))
        R = (M + H - 1)//H
        max_block_abs = 0.0
        for j in range(R):
            L = j*H + 1
            Rr = min((j+1)*H, M)
            ssum = ps[Rr] - ps[L-1]
            if abs(ssum) > max_block_abs:
                max_block_abs = abs(ssum)
        T = (H**0.5) * (log(N)**C)
        out.append({
            "s": round(s,3), "H": H,
            "max_block_abs": max_block_abs,
            "predicted_T": T,
            "ratio": max_block_abs / max(1.0, T)
        })
        k += 1
        s = s_min + k*s_step
    return out

# ----------------------------
# Type II/III toy dispersion
# ----------------------------

def primes_in_range(lo: int, hi: int) -> List[int]:
    """Return primes in (lo, hi] via simple sieve."""
    if hi <= 1 or hi <= lo:
        return []
    sieve = [True]*(hi+1)
    sieve[0]=sieve[1]=False
    r = int(hi**0.5)
    for p in range(2, r+1):
        if sieve[p]:
            for j in range(p*p, hi+1, p):
                sieve[j]=False
    return [p for p in range(max(2,lo+1), hi+1) if sieve[p]]

def _geom_sum_abs(M: int, denom: int, c: int=1) -> float:
    """
    |sum_{m=1}^M e(2π i c m / denom)| simplifies to |sin(π c M / denom) / sin(π c / denom)|,
    with care if denom | c → value = M.
    """
    import math
    num = abs(math.sin(math.pi * c * M / denom))
    den = abs(math.sin(math.pi * c / denom))
    if den == 0.0:
        return float(M)
    return num / den

def typeII_bilinear_dispersion(x: int, P: int, Q: int, A: float = 2.0, c: int = 1) -> Dict:
    ps = primes_in_range(P, 2*P)
    qs = primes_in_range(Q, 2*Q)
    S = 0.0
    for p in ps:
        for q in qs:
            M = x // (p*q)
            if M <= 0: 
                continue
            S += _geom_sum_abs(int(M), p*q, c)
    bench = (x**0.5) * ((P*Q)**0.5) * (log(x)**A) if x>1 else 1.0
    return {
        "x": x, "P": P, "Q": Q, "A": A, "c": c,
        "num_pairs": len(ps)*len(qs),
        "sum_abs_geom": S,
        "benchmark_x12_PQ12_logA": bench,
        "ratio": S / bench if bench>0 else float('inf')
    }

# ----------------------------
# Plot helpers (matplotlib; one plot per figure; no custom colors)
# ----------------------------

def _save_plot(fig, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("Saved plot:", path)
    return path

def plot_typeI_ratio_vs_b(rows: List[Dict], outdir: str):
    import matplotlib.pyplot as plt
    xs = [r["b"] for r in rows if r["ratio_if_s_half"] is not None]
    ys = [r["ratio_if_s_half"] for r in rows if r["ratio_if_s_half"] is not None]
    fig = plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="-")
    plt.xlabel("b")
    plt.ylabel("ratio = |T_b| / (N^{3/4}/b · (log N)^C)")
    plt.title("Type I: envelope check (s=1/2)")
    return _save_plot(fig, outdir, "plot_typeI_ratio_vs_b.png")

def plot_typeI_excfrac_vs_b(rows: List[Dict], outdir: str):
    import matplotlib.pyplot as plt
    xs = [r["b"] for r in rows]
    ys = [r["exc_fraction"] for r in rows]
    fig = plt.figure()
    plt.plot(xs, ys, marker="s", linestyle="-")
    plt.xlabel("b")
    plt.ylabel("exceptional block fraction")
    plt.title("Type I: exceptional blocks per b")
    return _save_plot(fig, outdir, "plot_typeI_excfrac_vs_b.png")

def plot_typeI_H_scaling(hrows: List[Dict], outdir: str):
    import matplotlib.pyplot as plt
    Hs = [r["H"] for r in hrows]
    maxs = [r["max_block_abs"] for r in hrows]
    preds = [r["predicted_T"] for r in hrows]
    fig = plt.figure()
    plt.plot(Hs, maxs, marker="o", linestyle="-", label="max_block_abs")
    plt.plot(Hs, preds, marker="^", linestyle="--", label="H^{1/2}(log N)^C")
    plt.xlabel("H")
    plt.ylabel("magnitude")
    plt.title("Type I: H-scaling (max block vs prediction)")
    plt.legend()
    return _save_plot(fig, outdir, "plot_typeI_H_scaling.png")

def plot_typeII_ratios(results: List[Dict], outdir: str):
    import matplotlib.pyplot as plt
    xs = list(range(len(results)))
    ys = [r["ratio"] for r in results]
    fig = plt.figure()
    plt.bar([str(i) for i in xs], ys)
    plt.xlabel("scenario index")
    plt.ylabel("ratio = S / (x^{1/2}(PQ)^{1/2}(log x)^A)")
    plt.title("Type II/III toy: dispersion vs benchmark")
    return _save_plot(fig, outdir, "plot_typeII_toy_ratio.png")

# ----------------------------
# CSV helpers
# ----------------------------

def write_csv_rows(rows: List[Dict], outdir: str, basename: str):
    import csv
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, basename)
    if not rows:
        open(path,"w").write("")
        return path
    headers = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in headers})
    print("Saved CSV:", path)
    return path

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200000)
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--eta", type=float, default=0.10)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--s", type=float, default=0.50, help="H=N^s for Type I block stats")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--typeI", action="store_true", help="Run Type I block stats")
    ap.add_argument("--sweep-H", action="store_true", help="Run H-scaling sweep (Type I)")
    ap.add_argument("--dispersion", action="store_true", help="Run Type II/III toy dispersion")
    ap.add_argument("--run-all", action="store_true", help="Run all demos")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_all or args.typeI:
        out = typeI_block_stats(N=args.N, h=args.h, eta=args.eta, C=args.C, s=args.s)
        rows = out["rows"]
        csv_name = f"typeI_block_stats_N{args.N}_h{args.h}_eta{args.eta}_C{args.C}_s{args.s}_{ts}.csv"
        csv_path = write_csv_rows(rows, args.outdir, csv_name)
        p1 = plot_typeI_ratio_vs_b(rows, args.outdir)
        p2 = plot_typeI_excfrac_vs_b(rows, args.outdir)
        print(f"[Type I] global_exception_fraction = {out['global_exception_fraction']:.6f} ; H={out['H']} ; T≈{out['T']:.3f}; bmax={out['bmax']}")
        print("Files:", csv_path, p1, p2)

    if args.run_all or args.sweep_H:
        hrows = typeI_H_scaling(N=args.N, b=1, h=args.h, C=args.C)
        csv_name = f"typeI_H_scaling_N{args.N}_b1_h{args.h}_C{args.C}_{ts}.csv"
        csv_path = write_csv_rows(hrows, args.outdir, csv_name)
        p3 = plot_typeI_H_scaling(hrows, args.outdir)
        print("Files:", csv_path, p3)

    if args.run_all or args.dispersion:
        configs = [
            {"x": max(100000, args.N//2), "P": 41, "Q": 59, "A": args.C, "c": 1},
            {"x": max(300000, args.N),   "P": 97, "Q": 127, "A": args.C, "c": 1},
        ]
        results = [typeII_bilinear_dispersion(**cfg) for cfg in configs]
        csv_name = f"typeII_dispersion_toy_{ts}.csv"
        csv_path = write_csv_rows(results, args.outdir, csv_name)
        p4 = plot_typeII_ratios(results, args.outdir)
        print("Files:", csv_path, p4)

if __name__ == "__main__":
    import datetime
    main()
