
from kernel import Atom, BigO, combine, check_proof
from fractional_derivatives_chowla import proof_third_cumulant_vanishing, proof_fourth_cumulant_vanishing, proof_chowla_from_cumulants

# Conditional: if the cumulant-vanishing and implication steps hold, then full Chowla follows.
proof_full_chowla_conditional = combine(
    "Full-Chowla-Conditional",
    BigO(Atom("Sum_{n <= N} lambda(n) * lambda(n+h)"), Atom("o(N)")),
    proof_third_cumulant_vanishing,
    proof_fourth_cumulant_vanishing,
    proof_chowla_from_cumulants
)

if __name__ == "__main__":
    print("\n=== Full Chowla (Conditional) ===")
    print(proof_full_chowla_conditional)
    check_proof(proof_full_chowla_conditional)
