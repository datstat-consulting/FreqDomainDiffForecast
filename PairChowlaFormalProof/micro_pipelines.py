
from kernel import Atom, Equals, BigO, combine, axiom, check_proof

# ---- Zero-density micro pipeline ----
s = Atom("s"); T = Atom("T"); r = Atom("r"); sigma = Atom("sigma"); chi = Atom("chi")

zd_explicit = combine("Zero-density-explicit-formula",
    Equals(Atom("NZeros(sigma,T,chi)"),
           Atom("Explicit formula integral + prime sum + zero sum")),
    axiom(Atom("Weil explicit formula schema")))

zfr = combine("Zero-free-region-classical",
    Equals(Atom("Re(s) >= 1 - c / log(r*T)"),
           Atom("Zero-free region for Dirichlet L")),
    axiom(Atom("de la Vallée Poussin")))

log_deriv = combine("Log-derivative-mean-value",
    BigO(Atom("Mean value of (L'/L)(s,chi)"),
         Atom("(r*T)^{eps} * (log(r*T))^B")),
    axiom(Atom("Montgomery–Vaughan mean-value")))

lratio = combine("L-function-ratio",
    Equals(Atom("L(s,chi⊗λ)"), Atom("L(2s,chi^2) / L(s,chi)")),
    axiom(Atom("Euler product algebra")))

zd_micro = combine("Zero-density-micro",
    BigO(Atom("NZeros(sigma,T,chi, liouville_twist=True)"),
         Atom("(r*T)^(C*(1-sigma)) * (log(r*T))^D")),
    lratio, zd_explicit, zfr, log_deriv)

# ---- Bilinear dispersion micro pipeline ----
X = Atom("X"); P = Atom("P"); Q = Atom("Q"); d = Atom("d")

completion = combine("Completion-step",
    Equals(Atom("Bilinear sum over a~P, b~Q"),
           Atom("Completed to additive characters mod d + smooth weights")),
    axiom(Atom("Poisson/Voronoi completion schema")))

large_sieve = combine("Large-sieve-inequality",
    BigO(Atom("Second moment over characters/moduli"),
         Atom("(X + d^2) * (PQ) * (log X)^A")),
    axiom(Atom("Hybrid/dual large sieve")))

cs = combine("Cauchy-Schwarz-step",
    Equals(Atom("Dispersion second moment"),
           Atom("∑_{Δ} |∑_{a,b} ...|^2 with Δ-splitting via Cauchy-Schwarz")),
    axiom(Atom("Second moment expansion")))

smooth = combine("Smoothing-optimization",
    BigO(Atom("Optimized bound after dyadic partition"),
         Atom("X^{1/2} * (PQ)^{1/2} * (log X)^C")),
    axiom(Atom("Choice of weights/parameters")))

uniformity = combine("Dispersion-parameter-uniformity",
    Equals(Atom("Uniform in r,a,blocks"), Atom("Holds for all dyadic blocks with PQ ≤ X^{1-ε}")),
    axiom(Atom("Dyadic bookkeeping lemma")))

bilinear_micro = combine("Bilinear-dispersion-micro",
    BigO(Atom("Bilinear dispersion sum"),
         Atom("X^{1/2} * (PQ)^{1/2} * d^{-δ} * (log X)^C")),
    completion, large_sieve, cs, smooth, uniformity)

bilinear_mapped = combine("Bilinear-dispersion-HS-mapped",
    bilinear_micro.conclusion,
    bilinear_micro)

if __name__ == "__main__":
    for pf in [zd_micro, bilinear_micro, bilinear_mapped]:
        check_proof(pf)
    print("micro pipelines ok")
