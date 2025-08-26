# type_I.py
# Formalization of the Type I (short‐interval) subproof
# in the density‐one Pair Chowla theorem.

from kernel import Atom, Equals, BigO, Implies, axiom, combine, check_proof

# 1. Matomäki–Radziwiłł mean‐square lemma
proof_ms = axiom(
    BigO(
        Atom("(1/N)Σ_{x=1}^{N-H}|Σ_{n=x+1}^{x+H}λ(n)|²"),
        Atom("H*(log N)^(-A)")
    )
)

# 2. Chebyshev count: size of bad set
proof_cheb_count = combine(
    "Chebyshev-count",
    BigO(
        Atom("|BadSet|"),
        Atom("N/T² * H*(log N)^(-A)")
    ),
    proof_ms
)

# 3. Pointwise Chebyshev: outside BadSet the block sum is small
proof_cheb_point = combine(
    "Chebyshev-pointwise",
    Implies(
        Atom("x ∉ BadSet"),
        Atom("|Σ_{n∈I_j}λ(n)λ(bn+h)| ≤ T")
    ),
    proof_cheb_count
)

# 4. Block‐count axiom: number of blocks R = O(N/(bH))
proof_block_count = axiom(
    Equals(
        Atom("R"),
        Atom("O(N/(bH))")
    )
)

# 5. Summation over blocks: T_b = O(N^(3/4)/b * (log N)^C)
proof_sum_blocks = combine(
    "Sum-blocks",
    BigO(
        Atom("T_b"),
        Atom("N^(3/4)/b*(log N)^C")
    ),
    proof_block_count,
    proof_cheb_point
)

# 6. Summation over b: Σ_{b≤D} T_b = O(N^(3/4)*(log N)^(C+1))
proof_sum_b = combine(
    "Sum-over-b",
    BigO(
        Atom("Σ_{b≤D}T_b"),
        Atom("N^(3/4)*(log N)^(C+1)")
    ),
    proof_sum_blocks
)

# === Self‐test ===
if __name__ == "__main__":
    print("=== Type I Subproof Formalization ===")
    print(proof_sum_b)
    check_proof(proof_sum_b)