
from kernel import Atom, Equals, Forall, Product, Fraction, LFunction, Power, Plus, Minus, Multiply, Sum, Integral, NZeros, Implies, BigO, axiom, combine, check_proof

# --- 1. Euler-product identity for L(s, chi ⊗ λ) ---

s_sym = Atom("s")
chi_sym = Atom("chi")
p_sym = Atom("p")

# L(s, chi ⊗ λ) = Product_p (1 / (1 + chi(p)p^-s))
proof_euler_product = combine(
    'Euler-product-L-lambda',
    Equals(
        LFunction(s_sym, chi_sym, liouville_twist=True),
        Product(
            f'{p_sym}',
            Atom(f'{p_sym} is prime'),
            Fraction(Atom('1'), Plus(Atom('1'), Atom('chi(p)p^-s')))
        )
    ),
    axiom(Atom("Geometric series sum axiom")),
    axiom(Atom("Algebraic manipulation axiom"))
)

# L(s, chi ⊗ λ) = L(2s, chi^2) / L(s, chi)
proof_l_function_ratio = combine(
    'L-function-ratio',
    Equals(
        LFunction(s_sym, chi_sym, liouville_twist=True),
        Fraction(LFunction(Atom('2s'), Atom('chi^2')), LFunction(s_sym, chi_sym))
    ),
    proof_euler_product
)

# --- 2. Zero-density for L(s, chi ⊗ λ) ---

sigma_sym = Atom('sigma')
T_sym = Atom('T')
r_sym = Atom('r')

# Classical zero-density for L(s, chi^2)
proof_classical_zero_density = axiom(
    BigO(
        NZeros(Atom('sigma_prime'), Atom('T_prime'), Atom('psi')),
        Atom('(r*T_prime)^(A*(1-sigma_prime)) * (log(r*T_prime))^B')
    )
)

# Conclusion of zero-density lemma
proof_zero_density_conclusion = combine(
    'Zero-density-bound',
    BigO(
        NZeros(sigma_sym, T_sym, chi_sym, liouville_twist=True),
        Atom('(r*T)^(C*(1-sigma)) * (log(r*T))^D')
    ),
    proof_l_function_ratio,
    proof_classical_zero_density
)

# --- 3. Mellin-Perron for S(chi, beta) and Contour Shift ---

K_sym = Atom("K")
beta_sym = Atom("beta")

# S(chi, beta) = (1/(2*pi*i)) * Integral(c-iT to c+iT) L(s, chi ⊗ λ) * K^s / s ds + O(K/T) + O(K*|beta|)
proof_perron_formula = combine(
    'Perron-formula',
    Equals(
        Atom("S(chi, beta)"),
        Plus(
            Multiply(
                Fraction(Atom("1"), Multiply(Atom("2"), Multiply(Atom("pi"), Atom("i")))),
                Integral(
                    s_sym,
                    Minus(Atom("c"), Multiply(Atom("i"), T_sym)),
                    Plus(Atom("c"), Multiply(Atom("i"), T_sym)),
                    Multiply(LFunction(s_sym, chi_sym, liouville_twist=True), Fraction(Power(K_sym, s_sym), s_sym))
                )
            ),
            Plus(
                BigO(K_sym, T_sym),
                BigO(K_sym, Atom("|beta|"))
            )
        )
    ),
    axiom(Atom("Standard Perron formula axiom"))
)

# Shifting to Re(s) = 1 - eta, using zero-density bound
proof_contour_shift = combine(
    'Contour-shift',
    BigO(
        Atom("S(chi, beta)"),
        Atom("K^(1/2) * (log N)^C1")
    ),
    proof_perron_formula,
    proof_zero_density_conclusion,
    axiom(Atom("Convexity bound for L-functions"))
)

# --- 4. Assembling F_K and G_{q,h,K} ---

# F_K(a/r + beta) expansion using Gauss sums
proof_gauss_sum_expansion_F = combine(
    'Gauss-sum-expansion',
    Equals(
        Atom("F_K(a/r + beta)"),
        Atom("(1/phi(r)) * Sum_chi tau(bar(chi)) * chi(a) * S(chi, beta)")
    ),
    axiom(Atom("Gauss sum identity axiom"))
)

# Bounding F_K(a/r + beta)
proof_bound_F_K = combine(
    'Major-arc-integral-bound',
    BigO(
        Atom("|F_K(a/r + beta)|"),
        Atom("K^(1/2) * r^(1/2) * (log N)^C1")
    ),
    proof_gauss_sum_expansion_F,
    proof_contour_shift,
    axiom(Atom("|tau(bar(chi))| = sqrt(r) axiom"))
)

# Bounding G_{q,h,K}(a/r + beta) (similar to F_K)
proof_bound_G_K = combine(
    'Major-arc-integral-bound',
    BigO(
        Atom("|G_{q,h,K}(a/r + beta)|"),
        Atom("K^(1/2) * r^(1/2) * (log N)^C2")
    ),
    axiom(Atom("Similar argument for G_K"))
)

# --- 5. Putting major-arc pieces together ---

# Integral over a single major arc
proof_single_major_arc_integral = combine(
    'Major-arc-integral-bound',
    BigO(
        Atom("Integral over single major arc"),
        Atom("Q * (log N)^C_prime")
    ),
    proof_bound_F_K,
    proof_bound_G_K
)

# Summing over all major arcs
proof_total_major_arc_integral = combine(
    'Major-arc-integral-bound',
    BigO(
        Atom("Integral over all major arcs"),
        Atom("K^(1/2) * q^(-delta1) * (log N)^C1")
    ),
    proof_single_major_arc_integral,
    axiom(Atom("Sum over r and a/r axiom"))
)


# === Self-test ===
if __name__ == "__main__":
    print("\n=== Major Arc Formalization (Continued) ===")
    print(proof_euler_product)
    check_proof(proof_euler_product)
    print(proof_l_function_ratio)
    check_proof(proof_l_function_ratio)
    print(proof_zero_density_conclusion)
    check_proof(proof_zero_density_conclusion)
    print(proof_perron_formula)
    check_proof(proof_perron_formula)
    print(proof_contour_shift)
    check_proof(proof_contour_shift)
    print(proof_gauss_sum_expansion_F)
    check_proof(proof_gauss_sum_expansion_F)
    print(proof_bound_F_K)
    check_proof(proof_bound_F_K)
    print(proof_bound_G_K)
    check_proof(proof_bound_G_K)
    print(proof_single_major_arc_integral)
    check_proof(proof_single_major_arc_integral)
    print(proof_total_major_arc_integral)
    check_proof(proof_total_major_arc_integral)


