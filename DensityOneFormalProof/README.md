# Density-One Pair Correlations of Liouville — Python Formalization

This repo contains a **self-contained Python proof-checker** and a set of **derived proof modules** that formalize the paper.

- Proof kernel: `kernel.py` (Atoms, Equals, Implies, Forall, Sum, BigO; strict validators; `check_proof`).
- Analytic tools are **derived** (geom series → orthogonality → LSI base; Unitary DFT + Plancherel → Parseval; pretentious + mean values → Halász).
- Paper structure modules: `heath_brown.py`, `type_I.py`, `type_II.py`, `final_combination.py`.
- **Numerics** are separate and *non-proof* — they illustrate the same scales and side-conditions.

---

## Run the formal proof (end-to-end)

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

---

## Numerical integrals & bounds (optional)

For sanity checks, there is `numerics.py` with a tiny adaptive-Simpson integrator & helpers (not used by the proof). See earlier README versions for details.

---

## Numerics that mirror each proof step (optional, upgraded)

The **new** `numerics.py` exhaustively **illustrates** each analytic step in the paper. It produces CSVs and plots.

**1) Square–divisor sanity**
- Verifies `λ(n) == Σ_{d^2|n} μ(n/d^2)` for an initial range.
- Mirrors the Lemma used before the Heath–Brown decomposition.

**2) Type I (MR + Chebyshev + blocks)**
- For given `N, h, η, C, s` (where `H=N^s`, default `s=1/2`), it:
  - builds `a_m = λ(m) λ(bm+h)` for each `1 ≤ b ≤ ⌊N^(1/2−η)⌋`,
  - partitions into length-`H` blocks,
  - counts **exceptional** blocks where `|Σ_block a_m| > T`, with `T = H^{1/2}(log N)^C`,
  - computes `T_b = Σ_{m≤M} a_m` and compares to the envelope `N^(3/4)/b · (log N)^C` (if `s=1/2`).
- Outputs a CSV and two plots:
  - `plot_typeI_ratio_vs_b.png` (|T_b| / envelope vs b)
  - `plot_typeI_excfrac_vs_b.png` (exceptional fraction vs b)

**3) Type I H-scaling**
- Fixes `b=1` and varies `H=N^s`, `s∈[0.30,0.70]`.
- Compares `max_block_abs` to `H^{1/2}(log N)^C` (CSV + `plot_typeI_H_scaling.png`).

**4) Type II/III toy dispersion**
- Quickly evaluates `Σ_{p∈(P,2P]} Σ_{q∈(Q,2Q]} |Σ_{m≤x/(pq)} e(cm/(pq))|` via exact geometric-series magnitude.
- Compares to `x^{1/2}(PQ)^{1/2}(log x)^A` (CSV + `plot_typeII_toy_ratio.png`).

### Run examples

Run everything (recommended for the paper figures):

```bash
python numerics.py --N 200000 --h 1 --eta 0.10 --C 2.0 --s 0.5 --outdir out --run-all
```

Or piecemeal:

```bash
# Type I only:
python numerics.py --N 200000 --h 1 --eta 0.10 --C 2.0 --s 0.5 --outdir out --typeI

# H-scaling:
python numerics.py --N 200000 --h 1 --C 2.0 --outdir out --sweep-H

# Dispersion toy:
python numerics.py --outdir out --dispersion
```

**Artifacts written to `out/`:**
- CSVs: `typeI_block_stats_*.csv`, `typeI_H_scaling_*.csv`, `typeII_dispersion_toy_*.csv`
- Plots: `plot_typeI_ratio_vs_b.png`, `plot_typeI_excfrac_vs_b.png`, `plot_typeI_H_scaling.png`, `plot_typeII_toy_ratio.png`

> These numerics **illustrate** (not prove) the MR/Chebyshev block behavior, the \(H^{1/2}\) scaling that leads to \(N^{3/4}\), and the bilinear dispersion scale.

---

## Limitations

- The checker validates proof **shape** and **side-conditions** rigorously, but does not compute analytic inequalities.
- The numerics file is for exploration and figures only.

## Semantics: Pretentious & MR verification (new)

This repository now includes a **semantic layer** that actively replays the analytic steps your proof uses.  
These checks run automatically via kernel rule hooks and must pass for the overall proof to verify.

### What is semantically checked

- **Finite orthogonality (Char-orthog)** — exact additive character sums on \(\mathbb{Z}/q\mathbb{Z}\).  
  Source: `semantics_discrete.py`

- **Finite Parseval (Parseval-tool)** — unitary DFT check on random vectors (identity exact in theory).  
  Source: `semantics_discrete.py`

- **Cauchy–Schwarz (Cauchy-Schwarz)** — inequality verified on random test vectors.  
  Source: `semantics_ext.py`

- **Short-interval mean-square identity (MR-mean-square-identity)** — algebraic identity behind MR, verified by  
  autocorrelation/triangular kernel with DFT/Plancherel.  
  Source: `semantics_fourier_mr.py`

- **Pretentious / Halász for \(\lambda\) (Halasz-bound(λ))** — computes the pretentious distance  
  \(D(\lambda,n^{it};X)^2=\sum_{p\le X}(1-\Re(\lambda(p)p^{-it}))/p\) over a \(t\)-grid and enforces a small 
  Halász-style envelope \(\exp(-D^2)+1/\sqrt{\log X}\).  
  Source: `semantics_pretentious.py`

- **MR inequality witness (MR-inequality-witness)** — finite-\(N\) check that  
  \(\frac{1}{N}\sum_{x\le N}\bigl|\sum_{x<n\le x+H}\lambda(n)\bigr|^2 \lesssim H/(\log N)^A\) holds for moderate \(N,H\).  
  Source: `semantics_mr_witness.py`

- **Bilinear dispersion scale (Bilinear-LSI)** — double sum over primes vs benchmark \(x^{1/2}(PQ)^{1/2}(\log x)^A\) using
  exact geometric sums of exponentials.  
  Source: `semantics_ext.py`

- **Interval arithmetic sanity (ExpLog-Interval, Trig-Interval)** — rigorous-leaning outward rounding for `exp/log`, exact-extrema 
  and Taylor enclosures for `sin/cos`; random spot checks ensure true values fall inside reported intervals.  
  Source: `rigorous_interval.py`

### Files added for semantics

- `semantics_discrete.py` — exact finite Fourier orthogonality & Parseval.  
- `semantics_measure.py` — interval arithmetic and exact Fourier orthogonality on \([0,1]\).  
- `rigorous_interval.py` — directed-rounding intervals and certified `sin/cos` enclosures.  
- `semantics_ext.py` — integral enclosures and inequality/witness utilities.  
- `semantics_fourier_mr.py` — MR mean-square identity replay (discrete).  
- `semantics_pretentious.py` — pretentious distance \(D^2\) and Halász-style envelope for \(\lambda\).  
- `semantics_mr_witness.py` — numerical MR inequality witness for \(\lambda\).  
- `semantics_hooks.py` — installs semantic checks for the rules listed above.  
- `run_all_with_semantics.py` — runs the full pipeline with semantics enabled.

### How to run with semantics

```bash
# one command to run the full pipeline with semantic validations
python run_all_with_semantics.py
```
You should see a banner like  
`Installed semantic checks incl. Halasz-bound(λ), MR-inequality-witness`  
followed by `OK` for each module.

### Notes on guarantees

- Identities (orthogonality, Parseval, short-interval mean-square identity) are **exact** in the semantics layer.  
- Inequality steps are backed by **rigorous-leaning enclosures** (interval arithmetic with outward rounding) and/or
  **finite‑N witnesses** where appropriate (e.g., MR inequality witness).  
- For a full constant‑sharp derivation of MR/Halász in complete generality, extend the pretentious layer with certified prime‑sum
  enclosures and complex‑analytic bounds; the present build is designed to be incrementally tightened.
