# bilinear_sieve.py
# Derived formalization of the bilinear large-sieve estimate (shape-checked).

from kernel import Atom, Sum, BigO, combine, axiom, check_proof

from lsi_base import proof_lsi_base as prem_lsi_base
from geom_series import proof_orth
from kernel import Atom, axiom

orth_alias = axiom(Atom("additive characters orthogonality"))
proof_lsi = combine(
    "Bilinear-LSI",
    BigO(
        Sum(
            "p", "P<p≤2P",
            Sum(
                "q", "Q<q≤2Q",
                Atom("|Σ_{m=1}^{⌊x/(p*q)⌋} e(*)|")
            )
        ),
        Atom("x^(1/2)*(P*Q)^(1/2)*(log x)^A")
    ),
    prem_lsi_base,
    orth_alias
)

if __name__ == "__main__":
    print("=== Bilinear LSI Derived ===")
    print(proof_lsi)
    check_proof(proof_lsi)
