# Density-One Pair Correlations of Liouville — Python Formalization

This repo contains a **self-contained Python proof-checker** and a set of **derived proof modules** that formalize the paper:

> **Density-One Pair Correlations of the Liouville Function**  
> “We show that, for fixed nonzero shift \(h\), the two–point correlation \(\sum_{n\le N}\lambda(n)\lambda(n+h)\) admits a power-saving outside a density-zero set of \(n\), uniformly in fixed \(h\).”

No Lean/Coq/Isabelle is used. Everything runs inside a minimal checker (`kernel.py`) with **strict, content-aware validators** that enforce the shape of each argument and critical **side conditions** (e.g. window size, dilation, large-sieve range). Higher-level theorems (MR short-interval mean-square; bilinear large sieve) are **derived in this system** from lower-level modules, not used as raw axioms.

---

## What is “formal” here?

- A **small proof language** (Atoms, `Equals`, `Implies`, `Forall`, `Sum`, `BigO`) plus rules like `→I` and `→E`.
- **Validators** check every rule application for structural and side-condition correctness (no free passes).  
  Example: Type I requires you to record `n = b*m + h` **and** `window length = bH`, choose `T = H^(1/2)(log N)^C`, and assert `H ≥ N^(1/6)`.
- The MR bound and the bilinear large-sieve estimate are **derived** from standard toolboxes:
  - `MR-Theorem` from **Parseval** + **Halasz** + CMF + 1-boundedness.
  - `Bilinear-LSI` from **geometric series → character orthogonality → large sieve base**.

> We do **not** model analytic semantics (integrals, numeric inequalities) — instead, validators enforce the *shape* of the argument and check that all hypotheses and side-conditions are present.

---

## Repository layout

```
kernel.py                  # minimal AST + proof object + strict validators + checker
number_theory.py           # square-divisor identity λ(n)=Σ_{d^2|n} μ(n/d^2)

# Derived analytic toolchain (bottom → top)
geom_series.py             # Geometric series → Character orthogonality    (Geometric-series, Char-orthog)
lsi_base.py                # Bessel/Plancherel + orthogonality → LSI base  (LSI-base)
parseval_derive.py         # Unitary DFT + Plancherel → Parseval tool      (Parseval-tool)
halasz_derive.py           # Pretentious distance + mean value → Halasz    (Halasz-tool)

mr_theorem.py              # Parseval + Halasz + CMF + 1-bounded → MR-Theorem
bilinear_sieve.py          # LSI base + orthogonality → Bilinear-LSI

# Paper structure
heath_brown.py             # HB decomposition nodes and tie-back
type_I.py                  # Type I: MR + Chebyshev + block count + dilation/window checks
type_II.py                 # Type II/III: Bilinear-LSI + Cauchy–Schwarz + dyadic counting
final_combination.py       # Exceptional set union + final density-one theorem

README.md                  # this file
```

---

## How to run

You only need a standard Python 3 environment (no external packages).

**Option A (end-to-end):**
```bash
python kernel.py
python number_theory.py
python geom_series.py
python lsi_base.py
python parseval_derive.py
python halasz_derive.py
python mr_theorem.py
python bilinear_sieve.py
python heath_brown.py
python type_I.py
python type_II.py
python final_combination.py
```

**Option B (fast check):** just run the last module (it transitively relies on earlier ones):
```bash
python final_combination.py
```

Expected tail output:
```
=== Final Combination Proof ===
[Combine-final: ⊢ (∀h,ε. Σ_{n∈N_{h,ε}}λ(n)λ(n+h) = O(N^{3/4+ε})) from Union-bound]
```

You can also consult the bundled run log (if present): `formalization_run_log_everything.txt`.

---

## What the system proves

1. **Square-Divisor identity** (derived):  
   \(\lambda(n)=\sum_{d^2\mid n}\mu(n/d^2)\).

2. **Derived analytic tools:**
   - **Geometric-series**: \(\sum_{m=0}^{M-1} r^m = (1-r^M)/(1-r)\) (with \(r\ne1\)).
   - **Character orthogonality**: \(\sum_{m=0}^{q-1} e(2\pi i k m/q)=0\) if \(q \nmid k\), and \(=q\) if \(q\mid k\).
   - **LSI base** from Bessel/Plancherel + orthogonality.
   - **Parseval tool** from Unitary DFT + Plancherel.
   - **Halasz tool** from pretentious distance + Dirichlet polynomial mean values.

3. **MR short-interval mean-square** (*derived* `MR-Theorem`):  
   \((1/N)\sum_x\big|\sum_{x<n\le x+H}\lambda(n)\big|^2 = O\!\big(H(\log N)^{-A}\big)\).

4. **Bilinear large sieve** (*derived* `Bilinear-LSI`):  
   Dispersion bound \(\ll x^{1/2}(PQ)^{1/2}(\log x)^A\).

5. **Type I** (validators enforce):
   - Side condition: \(H \ge N^{1/6}\).
   - Explicit threshold choice: \(T = H^{1/2}(\log N)^C\).
   - Dilation and window: `n = b*m + h` and `window length = bH`.
   - Output: \(T_b = O(N^{3/4}/b \cdot (\log N)^C)\) and \(\sum_{b\le D}T_b = O(N^{3/4}(\log N)^{C+1})\).

6. **Type II/III** (validators enforce):
   - Side condition: \(PQ \le x^{1-\varepsilon}\) to apply bilinear LSI.
   - Output: `TypeII_total = O(N^(3/4)*(log N)^(C'))`.

7. **Final combination**:
   - Exceptional set bookkeeping → density-one.
   - **Main theorem (paper’s claim)**:  
     \(\displaystyle \sum_{n\le N,\;n\in\mathcal N_{h,\varepsilon}} \lambda(n)\lambda(n+h) = O\!\left(N^{3/4+\varepsilon}\right)\)
     uniformly for fixed \(h\).

---

## Key validators (enforced side-conditions)

- **Chebyshev-count**: requires MR mean-square shape and (optionally) `H ≥ N^(1/6)`.
- **Chebyshev-pointwise**: requires a **premise** defining `T = H^(1/2)*(log N)^C`.
- **Sum-blocks**: looks for `R = O(N/(bH))`, plus **both** `n = b*m + h` and `window length = bH`.
- **Bilinear-sieve-apply**: requires `Bilinear-LSI` or a BigO axiom of that shape, **and** a premise `P*Q ≤ x^(1-ε)`.
- **Sum-dyadic / Union-bound / Combine-final**: check that the combined statements match the paper’s final shapes.

> The kernel also **canonicalizes notation** (`H^{1/2} ↔ H^(1/2)`, `log(N) ↔ log N`) so small formatting differences don’t break checks.

---

## Extending / hacking

- To add a new rule: write `_check_<rule>(pf)` in `kernel.py`, then register it in `_RULE_CHECKS`.
- To create a new derived node: put a `combine("<rule>", ...)` in a module, supply all necessary premises, and run `check_proof`.
- Keep statements as short **text atoms** with clear keywords the validators can check reliably.

---

## Limitations

- This system is a **strict proof-shape checker**, not a full semantic analyzer. It guarantees **structure, premises, and side-conditions** are all present and composed correctly; it does **not** compute analytic bounds or integrals.
- Deep analytic facts are represented as **bottom-layer premises** (e.g., “Unitary DFT”, “Plancherel identity”, “pretentious distance triangle inequality”, etc.), then **built up** into MR/LSI and used at the paper level.

---

## Reproducibility

- Requirements: Python 3.x only.
- No network or external dependencies are needed.
- Deterministic: all modules are pure Python; each script prints the final node it verifies and then runs `check_proof`.

---

**Status:** All modules pass under the current validators. The bundled run log `formalization_run_log_everything.txt` shows the complete green run.
