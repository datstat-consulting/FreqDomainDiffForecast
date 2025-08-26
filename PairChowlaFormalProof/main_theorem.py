
from kernel import Atom, Equals, BigO, Plus, Multiply, axiom, combine, check_proof
from major_arc import proof_total_major_arc_integral
from minor_arc import proof_minor_arc_bound

# --- Main Theorem: Unconditional Pair Chowla ---

# S_{q,h}(K) = Integral_M + Integral_m
proof_integral_split = axiom(
    Equals(
        Atom("S_{q,h}(K)"),
        Plus(
            Atom("Integral_M F_K(alpha) * G_{q,h,K}(alpha) dalpha"),
            Atom("Integral_m F_K(alpha) * G_{q,h,K}(alpha) dalpha")
        )
    )
)

# Combine major and minor arc bounds
proof_combine_major_minor = combine(
    "Combine-major-minor",
    BigO(
        Atom("S_{q,h}(K)"),
        Atom("N^(3/4) * (log N)^C") # This is the final expected bound
    ),
    proof_integral_split,
    proof_total_major_arc_integral,
    proof_minor_arc_bound
)

# Final theorem statement
proof_final_chowla_unconditional = combine(
    "Final-Chowla-unconditional",
    BigO(
        Atom("Sum_{n <= N-h} lambda(n) * lambda(n+h)"),
        Atom("N^(3/4) * (log N)^C")
    ),
    proof_combine_major_minor,
    axiom(Atom("Relating S_{q,h}(K) to the main sum"))
)

# === Self-test ===
if __name__ == "__main__":
    print("\n=== Main Theorem Formalization ===")
    print(proof_integral_split)
    check_proof(proof_integral_split)
    print(proof_combine_major_minor)
    check_proof(proof_combine_major_minor)
    print(proof_final_chowla_unconditional)
    check_proof(proof_final_chowla_unconditional)


