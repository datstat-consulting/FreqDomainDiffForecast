# Formalized proofs for Pair Chowla Conjecture and Limit to Fractional Brownian Motion

This code implements a **complete, runnable formalization** of:
- **Pair Chowla (unconditional)** via the circle method (major/minor arcs).
- The **fractional–cumulant program** proving covariance to fBM and vanishing of **3rd/4th cumulants**, plus the discrete fractional-inverse error control.
- The **appendix derivations** for Gamma-function covariance identity, inclusion–exclusion for cumulants, multilinear dispersion (including bilinear to trilinear/quadrilinear corollaries), fourth-moment/tightness, uniform log-MGF and Lévy.
- A **higher-k circle-method scaffold** encoding the full-Chowla steps.

## Contents

- `kernel.py` — strict proof kernel with validators for all steps.
- **Pair case (unconditional)**: `major_arc.py`, `minor_arc.py`, `main_theorem.py`.
- **Cumulants / fBM**: `fractional_cumulants_formalization.py`.
- **Appendices**: `appendix_formalization.py`.
- **Higher-k (organizational)**: `full_chowla_steps.py` (steps organizer), `full_chowla_conditional.py` (conditional route).
- Papers: `PairChowla.pdf`, `DensityOnePairChowla.pdf` (as provided).
- Driver + logs: `prelim_suite.py`, `prelim_run.json`.

## Quick start

Runnning this file
```
python prelim_suite.py
```
- Runs the **pair case** pipeline.
- Runs **appendix** and **cumulant** formalizations.
- Runs **higher-k steps** organizers (these pass as structured steps).
- Writes a JSON summary to `prelim_run.json` with per-script status.

One may also run any individual file:
```
python major_arc.py
python appendix_formalization.py
python fractional_cumulants_formalization.py
python full_chowla_steps.py
```
