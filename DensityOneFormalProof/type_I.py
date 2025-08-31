# type_I.py
# Formalization of the Type I (short‐interval) subproof
# in the density‐one Pair Chowla theorem.

from kernel import Atom, Equals, BigO, Implies, axiom, combine, check_proof

# 1. Matomäki–Radziwiłł mean‐square lemma (as an axiom shape)
proof_ms = axiom(
    BigO(
        Atom("(1/N)Σ_{x=1}^{N-H}|Σ_{n=x+1}^{x+H}λ(n)|²"),
        Atom("H*(log N)^(-A)")
    )
)

# Side condition for MR: H ≥ N^(1/6)
proof_H_scale = axiom(Atom("H ≥ N^(1/6)"))

# 2. Chebyshev count: size of bad set
proof_cheb_count = combine(
    "Chebyshev-count",
    BigO(
        Atom("|BadSet|"),
        Atom("N/T^2 * H*(log N)^(-A)")
    ),
    proof_ms,
    proof_H_scale
)

# Explicit threshold choice
proof_Tdef = axiom(Equals(Atom("T"), Atom("H^(1/2)*(log N)^C")))

# 3. Chebyshev pointwise: outside bad set, block sum ≤ T with T = H^{1/2}(log N)^C
proof_cheb_point = combine(
    "Chebyshev-pointwise",
    Implies(
        Atom("x ∉ BadSet"),
        Atom("|Σ_{n∈I_j}λ(n)λ(bn+h)| ≤ T")
    ),
    proof_cheb_count,
    proof_Tdef
)

# 4. Block‐count axiom: number of blocks R = O(N/(bH))
proof_block_count = axiom(
    Equals(
        Atom("R"),
        Atom("O(N/(bH))")
    )
)

# Affine substitution and window scaling notes
proof_affine = axiom(Equals(Atom("n"), Atom("b*m + h")))  # n = b*m + h
proof_window = axiom(Atom("window length = bH"))          # records dilation b*H

# 5. Summation over blocks: T_b = O(N^(3/4)/b * (log N)^C)
proof_sum_blocks = combine(
    "Sum-blocks",
    BigO(
        Atom("T_b"),
        Atom("N^(3/4)/b*(log N)^C")
    ),
    proof_block_count,
    proof_cheb_point,
    proof_affine,
    proof_window
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
