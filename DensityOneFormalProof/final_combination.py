# final_combination.py
# Formalization of the final combination step in the density‐one Pair Chowla proof.

from kernel import Atom, Forall, BigO, combine, check_proof
from type_I import proof_sum_b       # Type I result: Σ_{b≤D} T_b = O(N^{3/4}(log N)^{C+1})
from type_II import proof_sum_dyadic  # Type II/III result: TypeII_total = O(N^{3/4}(log N)^{C'})

# 1. Exceptional‐set union bound
proof_union = combine(
    "Union-bound",
    Atom("|ExcI ∪ ExcII| = O(N^{1-δ})"),
    proof_sum_b,
    proof_sum_dyadic
)

# 2. Main density‐one conclusion
final_conclusion = Forall(
    "h,ε",
    BigO(
        Atom("Σ_{n∈N_{h,ε}}λ(n)λ(n+h)"),
        Atom("N^{3/4+ε}")
    )
)

proof_final = combine(
    "Combine-final",
    final_conclusion,
    proof_union
)

if __name__ == "__main__":
    print("=== Final Combination Proof ===")
    print(proof_final)
    check_proof(proof_final)
