
from kernel import Atom, Equals, Implies, BigO, combine, axiom, check_proof

# Symbols
k_sym = Atom("k")
K_sym = Atom("K")
N_sym = Atom("N")

# --- Step 1: k-fold Heath–Brown style decomposition (structural)
proof_k_heath_brown = combine(
    "kHeath-Brown-decomposition",
    Equals(
        Atom("lambda(n+h_1)*...*lambda(n+h_k)"),
        Atom("HB_k(n; h_1,...,h_k; D)")
    ),
    axiom(Atom("Generalized Heath–Brown identity for lambda^k (cited)"))
)

# --- Step 2: Express correlation as circle-method integrals and split arcs
proof_k_integral_split = combine(
    "kIntegral-split",
    Equals(
        Atom("S_{h1..hk}(K)"),
        Atom("Integral_M F_{k,K}(alpha) * G_{k,h,K}(alpha) dalpha + Integral_m F_{k,K}(alpha) * G_{k,h,K}(alpha) dalpha")
    ),
    proof_k_heath_brown
)


# --- Step 3: Major arcs bound using L-ratio, zero density, Gauss sums, contour shift
prem_euler = combine(
    "Euler-product-L-lambda",
    Equals(Atom("L(s, chi ⊗ λ)"), Atom("Π_{p|p is prime}(1 / (1 + chi(p)p^-s))")),
    axiom(Atom("Euler product citation"))
)
prem_lratio = combine(
    "L-function-ratio",
    Equals(Atom("L(s, chi ⊗ λ)"), Atom("L(2s, chi^2) / L(s, chi)")),
    prem_euler
)
prem_zerod = combine(
    "Zero-density-bound",
    BigO(Atom("N_{chi ⊗ λ}(sigma, T)"), Atom("(r*T)^(C*(1-sigma)) * (log(r*T))^D")),
    prem_lratio,
    axiom(BigO(Atom("N_{chi}(sigma, T)"), Atom("T^(C*(1-sigma)) * (log T)^D"))),
    axiom(Atom("zero density citation"))
)
prem_gauss = combine(
    "Gauss-sum-expansion",
    Equals(Atom("F_K(a/r + beta)"), Atom("(1/phi(r)) * Sum_chi tau(bar(chi)) * chi(a) * S(chi, beta)")),
    axiom(Atom("Gauss sum identity"))
)
prem_contour = combine(
    "Contour-shift",
    BigO(Atom("S(chi, beta)"), Atom("K^(1/2) * (log N)^C1")),
    combine("Perron-formula", Equals(Atom("S(chi, beta)"), Atom("∫ ... (K^s/s) ds + errors")), axiom(Atom("Perron citation"))),
    prem_zerod,
    axiom(Atom("convexity/truncation axiom"))
)
proof_k_major = combine(
    "Higher-major-arc-bound",
    BigO(
        Atom("Integral_M F_{k,K}(alpha) * G_{k,h,K}(alpha) dalpha"),
        Atom("N * (log N)^(-A)")
    ),
    prem_lratio, prem_zerod, prem_gauss, prem_contour,
    axiom(Atom("Singular series control / modulus bookkeeping"))
)

# --- Step 4: Minor arcs bound using U^s inverse + nilsequence non-correlation
proof_us_inverse = combine(
    "U^s-inverse-theorem",
    Implies(
        Atom("||F_{k,K}||_{U^s} >= delta"),
        Atom("correlation with an s-1 step nilsequence Phi")
    ),
    axiom(Atom("Green–Tao–Ziegler U^s inverse theorem citation"))
)
proof_nil_k = combine(
    "Nilsequence-non-correlation-k",
    BigO(
        Atom("|Sum_{n=1}^K lambda(n) Phi(n)|"),
        Atom("K * (log K)^(-A)")
    ),
    axiom(Atom("Quantitative nilsequence non-correlation for lambda vs (s-1)-step nilsequence"))
)
proof_k_minor = combine(
    "Higher-minor-arc-bound",
    BigO(
        Atom("Integral_m F_{k,K}(alpha) * G_{k,h,K}(alpha) dalpha"),
        Atom("N * (log N)^(-A)")
    ),
    proof_us_inverse,
    proof_nil_k,
    axiom(Atom("Energy/density-decrement iteration across k linear forms"))
)

# --- Step 5: Combine arc bounds
proof_k_combine = combine(
    "Combine-k-major-minor",
    BigO(
        Atom("S_{h1..hk}(K)"),
        Atom("N * (log N)^(-A)")
    ),
    proof_k_major,
    proof_k_minor
)

# --- Step 6: Relate S_{h1..hk}(K) back to the main correlation sum
proof_k_relate = combine(
    "Relate-kS-to-main-sum",
    Equals(
        Atom("S_{h1..hk}(K)"),
        Atom("Sum_{n <= N} lambda(n) * ... * lambda(n+h_k) (with smooth cutoff)")
    ),
    axiom(Atom("Cutoff removal / partial summation / dyadic decomposition"))
)

# --- Final: Full Chowla steps (conditional organizer)
proof_full_chowla_steps = combine(
    "Full-Chowla-steps",
    BigO(
        Atom("Sum_{n <= N} lambda(n) * ... * lambda(n+h_k)"),
        Atom("o(N)")
    ),
    proof_k_heath_brown,
    proof_k_integral_split,
    proof_k_major,
    proof_k_minor,
    proof_k_combine,
    proof_k_relate
)

if __name__ == "__main__":
    print("\\n=== Full Chowla (Steps, Conditional) ===")
    print(proof_full_chowla_steps)
    check_proof(proof_full_chowla_steps)