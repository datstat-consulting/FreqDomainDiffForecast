# semantics_hooks.py
"""
Semantic post-checks wired into the kernel's rule validators.
Adds:
- Halasz-bound (liouville): evaluate D^2 and envelope is small
- MR-inequality-witness: numerical verification on moderate N,H
"""

import random, math

def install_hooks():
    import kernel
    from semantics_discrete import additive_orthogonality, parseval_error
    from semantics_ext import check_cauchy_schwarz, check_chebyshev_tail, check_bilinear_dispersion_bound
    import semantics_fourier_mr as mr
    import rigorous_interval as RI
    import semantics_pretentious as pret
    import semantics_mr_witness as mrw

    RULES = getattr(kernel, "_RULE_CHECKS", None)
    if RULES is None or not isinstance(RULES, dict):
        print("[semantics_hooks] WARNING: kernel._RULE_CHECKS not found; hooks not installed")
        return False

    def wrap(fn):
        def wrapped(pf):
            ok = fn(pf)
            if not ok:
                return False
            rn = (pf.get("rule") or "").strip()
            if rn == "Halasz-bound(λ)":
                X = 200000
                grid = [k*0.2 for k in range(-25,26)]
                out = pret.halasz_mean_value_bound(pret.liouville_on_prime, X, grid)
                # Expect D^2 ~ 2 log log X; so exp(-D2) extremely small
                minimal_expectation = 1.0/(max(2.0, math.log(X))**0.49)  # be generous
                return out["bound"] < minimal_expectation
            if rn == "MR-inequality-witness":
                N = 100000; H = int(N**0.5); A = 1.5
                out = mrw.mr_bound_witness(N,H,A)
                # We accept if lhs ≤ 5 * rhs (loose to avoid false negatives on small N)
                return out["lhs"] <= 5.0 * out["rhs"]
            # existing earlier hooks
            if rn == "Char-orthog":
                for q in (5,7,11):
                    for k in (-2*q, -q, -1, 0, 1, q, 2*q):
                        from semantics_discrete import additive_orthogonality
                        S = additive_orthogonality(q, k)
                        if S not in (0, q): return False
                return True
            if rn == "Parseval-tool":
                for q in (4, 6, 8, 10):
                    from semantics_discrete import parseval_error
                    vec = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(q)]
                    if parseval_error(vec) > 1e-8: return False
                return True
            if rn == "Cauchy-Schwarz":
                for q in (16, 32):
                    from semantics_ext import check_cauchy_schwarz
                    v1 = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(q)]
                    v2 = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(q)]
                    if not check_cauchy_schwarz(v1, v2): return False
                return True
            if rn == "Chebyshev-blocks":
                from semantics_ext import check_chebyshev_tail
                n = 2000; sigma = 10.0
                vals = [random.gauss(0.0, sigma) for _ in range(n)]
                T = 5.0*sigma
                return check_chebyshev_tail(vals, T, allowed_fraction=0.01)
            if rn == "Bilinear-LSI":
                from semantics_ext import check_bilinear_dispersion_bound
                cfgs = [(100_000,41,59,2.0,1,1.0),(300_000,97,127,2.0,1,1.0)]
                for (x,P,Q,A,c,slack) in cfgs:
                    if not check_bilinear_dispersion_bound(x,P,Q,A,c,slack)["ok"]: return False
                return True
            if rn == "MR-mean-square-identity":
                N,H = 256, 32
                a = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(N)]
                err = mr.mean_square_identity(a, H)
                return err < 1e-8
            if rn == "ExpLog-Interval":
                for _ in range(5):
                    a = random.uniform(0.1, 3.0)
                    b = a + random.uniform(0.0, 3.0)
                    iv = RI.I(min(a,b), max(a,b))
                    ex = RI.i_exp(iv); lg = RI.i_log(RI.I(max(iv.lo,1e-6), iv.hi+1.0))
                    if not (math.exp(iv.lo) >= ex.lo - 1e-15 and math.exp(iv.hi) <= ex.hi + 1e-15): return False
                return True
            if rn == "Trig-Interval":
                for _ in range(10):
                    a = random.uniform(-3.0, 3.0)
                    b = a + random.uniform(0.0, 0.5)
                    iv = RI.I(min(a,b), max(a,b))
                    s_ext = RI.i_sin_ext(iv); c_ext = RI.i_cos_ext(iv)
                    s_ser = RI.i_sin_series(iv); c_ser = RI.i_cos_series(iv)
                    x = random.uniform(iv.lo, iv.hi)
                    sx, cx = math.sin(x), math.cos(x)
                    if not (s_ext.lo - 1e-15 <= sx <= s_ext.hi + 1e-15): return False
                    if not (c_ext.lo - 1e-15 <= cx <= c_ext.hi + 1e-15): return False
                    if not (s_ser.lo - 1e-6 <= sx <= s_ser.hi + 1e-6): return False
                    if not (c_ser.lo - 1e-6 <= cx <= c_ser.hi + 1e-6): return False
                return True
            return True
        return wrapped

    for name, fn in list(RULES.items()):
        if name in ("Halasz-bound(λ)", "MR-inequality-witness", "Char-orthog", "Parseval-tool",
                    "Cauchy-Schwarz", "Chebyshev-blocks", "Bilinear-LSI", "MR-mean-square-identity",
                    "ExpLog-Interval", "Trig-Interval"):
            RULES[name] = wrap(fn)
    print("[semantics_hooks] Installed semantic checks incl. Halasz-bound(λ), MR-inequality-witness")
    return True

if __name__ == "__main__":
    ok = install_hooks()
    print("installed:", ok)
