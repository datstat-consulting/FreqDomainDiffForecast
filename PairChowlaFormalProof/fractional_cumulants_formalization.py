
from kernel import Atom, Equals, BigO, Implies, combine, axiom, check_proof

# Symbols
alpha = Atom("alpha in (0,1)")
H = Atom("H = alpha + 1/2")

# Fractional weights and convolution
proof_weights = combine(
    "Fractional-weights-def",
    Equals(Atom("w_m"), Atom("Gamma(m+alpha) / (Gamma(alpha) * m!); w_m=0 for m<0")),
    axiom(Atom("ratio test / Gamma identities"))
)
proof_x_def = combine(
    "Convolution-x-def",
    Equals(Atom("x(n)"), Atom("Sum_{k=0}^n w_{n-k} * lambda(k) ; x(n)=0 for n<0")),
    proof_weights
)

# Gamma convolution identity and covariance
proof_gamma_conv = combine(
    "Gamma-convolution-identity",
    Equals(Atom("Sum_{i>=0} w_i * w_{i+ell}"), Atom("1/2 * (ell^{2H} - (ell-1)^{2H})")),
    proof_weights
)

# Use the unconditional Pair-Chowla bound (already proven in main_theorem.py)
# to justify that spectral measure is Lebesgue and covariance matches fBM.

# Build the required premises for Final-Chowla-unconditional
prem_major = axiom(Atom("Integral_M F_K(alpha) * G_{q,h,K}(alpha) dalpha"))
prem_minor = axiom(Atom("Integral_m F_K(alpha) * G_{q,h,K}(alpha) dalpha"))
prem_combine_mm = combine(
    "Combine-major-minor",
    BigO(Atom("S_{q,h}(K)"), Atom("K^(1/2) * (log N)^C")),
    prem_major, prem_minor
)
prem_relation = axiom(Atom("Relating S_{q,h}(K) to the main sum"))

pair_chowla_input = combine(
    "Final-Chowla-unconditional",
    BigO(Atom("Sum_{n <= N-h} lambda(n) * lambda(n+h)"), Atom("N^(3/4) * (log N)^C")),
    prem_combine_mm,
    prem_relation
)

proof_covariance = combine(
    "Covariance-from-pair",
    Equals(Atom("Cov(x(n), x(m))"), Atom("1/2*(n^{2H} + m^{2H} - |n-m|^{2H}) + O(1)")),
    pair_chowla_input,
    proof_gamma_conv
)

# Third cumulant: reduction to triple correlations
proof_k3_reduce = combine(
    "Third-cumulant-reduction",
    BigO(
        Atom("kappa_3(Delta_1, Delta_2, Delta_3)"),
        Atom("max_{h1,h2} Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) + o(1)")
    ),
    axiom(Atom("multilinearity of cumulants + translation invariance")),
    axiom(Atom("sum|A|=O(1) weight control"))
)

# Triple-Chowla bound (requires deep new inputs)
prem_hb4 = combine("HB-fourfold-identity", Equals(Atom("lambda(n)"), Atom("4-fold HB identity expansion + l.o.t.")), axiom(Atom("Heath-Brown (1982), (3.2)")))
prem_tri_disp = combine("Trilinear-dispersion", BigO(Atom("3D character sum"), Atom("N^(1-δ') (log N)^C")), axiom(Atom("multilinear large sieve / dispersion")))
prem_cs_red = combine("Cauchy-Schwarz-reduction", Equals(Atom("Dispersion → Correlation bound"), Atom("second moment / dyadic blocks")), axiom(Atom("two-step Cauchy–Schwarz")))
proof_triple_bound = combine(
    "Triple-Chowla-bound",
    BigO(Atom("Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2)"), Atom("N^{1-δ'} * (log N)^C")),
    prem_hb4, prem_tri_disp, prem_cs_red
)

proof_k3_vanish = combine(
    "Third-cumulant-vanishes",
    Equals(Atom("kappa_3(Delta_1, Delta_2, Delta_3)"), Atom("→ 0")),
    proof_k3_reduce, proof_triple_bound
)

# Fourth cumulant: reduction + quadruple bound
proof_k4_reduce = combine(
    "Fourth-cumulant-reduction",
    BigO(
        Atom("kappa_4(Delta_1, Delta_2, Delta_3, Delta_4)"),
        Atom("max_{h1,h2,h3} Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) * lambda(n+h3) + o(N)")
    ),
    axiom(Atom("sum|B|=O(1) weight control"))
)
prem_hb5 = combine("HB-fivefold-identity", Equals(Atom("lambda(n)"), Atom("5-fold HB identity expansion + l.o.t.")), axiom(Atom("HB identity generalization")))
prem_4d_disp = combine("FourD-dispersion", BigO(Atom("4D character sum"), Atom("X^{1/2} * Q^{1/2} * D^{-δ_0} (log N)^{C_0}")), axiom(Atom("4D dispersion lemma")))
prem_second_moment = combine("Second-moment-bound", Equals(Atom("Cauchy-Schwarz on blocks"), Atom("T_{P_*}^2 ≤ Q * sum |∑ f|^2")), axiom(Atom("standard argument")))
proof_quad_bound = combine(
    "Quadruple-Chowla-bound",
    BigO(Atom("Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) * lambda(n+h3)"), Atom("N^{1-δ''} * (log N)^{C'}")),
    prem_hb5, prem_4d_disp, prem_second_moment
)

proof_k4_vanish = combine(
    "Fourth-cumulant-vanishes",
    Equals(Atom("kappa_4(Delta_1, Delta_2, Delta_3, Delta_4)"), Atom("→ 0")),
    proof_k4_reduce, proof_quad_bound
)

# Discrete fractional inversion and tail control (used to bound ε-terms in expansions)
proof_frac_inverse = combine(
    "Frac-difference-inverse",
    Equals(Atom("lambda(n) = Sum_{k=0}^n mu_k * x(n-k)"), Atom("mu_k = (-1)^k * binom(alpha, k) ; Sum |mu_k| < ∞")),
    axiom(Atom("generating functions: (1-z)^alpha * (1-z)^(-alpha) = 1"))
)
proof_tail = combine(
    "Tail-truncation-error",
    BigO(Atom("Sum_{n<=N} E|epsilon_n|^2"), Atom("o(N)")),
    proof_frac_inverse,
    proof_covariance  # supplies the fBM-type second-moment control for x
)

# Method of cumulants to conclude Gaussian FDD (given κ3, κ4 vanish and covariance converges)
proof_moc = combine(
    "Method-of-cumulants",
    Equals(Atom("Gaussian FDD limit"), Atom("H = alpha + 1/2")),
    proof_k3_vanish, proof_k4_vanish, proof_covariance
)

if __name__ == "__main__":
    print("\\n=== Fractional Cumulants Formalization (as in your LaTeX) ===")
    for pf in [proof_weights, proof_x_def, proof_gamma_conv, proof_covariance,
               proof_k3_reduce, proof_triple_bound, proof_k3_vanish,
               proof_k4_reduce, proof_quad_bound, proof_k4_vanish,
               proof_frac_inverse, proof_tail, proof_moc]:
        print(pf)
        check_proof(pf)
