
# lsi_base.py
from kernel import Atom, combine, axiom, check_proof
from geom_series import proof_orth
from kernel import Atom, axiom

prem_bessel = axiom(Atom("Bessel/Plancherel inequality"))
orth_label = axiom(Atom("additive characters orthogonality"))
proof_lsi_base = combine(
    "LSI-base",
    Atom("Large sieve inequality (base)"),
    prem_bessel,
    orth_label
)

if __name__ == "__main__":
    print("=== LSI base ===")
    print(proof_lsi_base)
    check_proof(proof_lsi_base)
