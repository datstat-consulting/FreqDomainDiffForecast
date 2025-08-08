# number_theory.py
# Formalization of the square‐divisor identity using the kernel.

from kernel import Atom, Equals, Forall, axiom, combine, check_proof

# --- Function symbols for μ, λ, and the square-divisor sum ---

def μ_sym(n: str) -> Atom:
    """Symbolic Möbius function atom."""
    return Atom(f"μ({n})")

def λ_sym(n: str) -> Atom:
    """Symbolic Liouville function atom."""
    return Atom(f"λ({n})")

def sum_d2_sym(n: str) -> Atom:
    """Symbolic ∑_{d^2|n} μ(n/d^2) atom."""
    return Atom(f"Σ[d^2|{n}]μ({n}//d^2)")

# --- 1. Prime‐power identity axiom ---
# ∀ p,e: λ(p^e) = Σ_{d^2|p^e} μ(p^e/d^2)

proof_pp = axiom(
    Forall("p,e",
        Equals(
            λ_sym("p^e"),
            sum_d2_sym("p^e")
        )
    )
)

# --- 2. Multiplicativity axioms ---
# Mult(λ) and Mult(sum_d2)

proof_mult_lambda = axiom(Atom("Mult(λ)"))
proof_mult_sumd2  = axiom(Atom("Mult(sum_d2)"))

# --- 3. Multiplicative extension ---
# From prime‐power case + multiplicativity, derive the general identity:

proof_square_divisor = combine(
    "mult_ext",
    Forall("n",
        Equals(
            λ_sym("n"),
            sum_d2_sym("n")
        )
    ),
    proof_pp,
    proof_mult_lambda,
    proof_mult_sumd2
)

# === 4. Self‐test ===
if __name__ == "__main__":
    print("=== Square‐Divisor Identity Proof ===")
    print(proof_square_divisor)
    check_proof(proof_square_divisor)
