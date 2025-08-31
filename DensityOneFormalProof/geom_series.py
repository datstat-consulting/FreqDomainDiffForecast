
# geom_series.py
from kernel import Atom, Equals, combine, axiom, check_proof

prem_alg = axiom(Atom("algebraic identity for finite geometric sum"))
prem_dom = axiom(Atom("domain: r ≠ 1"))

proof_geom = combine(
    "Geometric-series",
    Equals(
        Atom("Σ_{m=0}^{M-1} r^m"),
        Atom("(1 - r^M)/(1 - r)")
    ),
    prem_alg, prem_dom
)

prem_per = axiom(Atom("periodicity of e(2πi θ)"))
proof_orth = combine(
    "Char-orthog",
    Equals(
        Atom("Σ_{m=0}^{q-1} e(2πi k m / q)"),
        Atom("= 0 if q ∤ k ; = q if q | k")
    ),
    proof_geom, prem_per
)

if __name__ == "__main__":
    print("=== Geometric Series & Orthogonality ===")
    print(proof_geom)
    print(proof_orth)
    check_proof(proof_geom)
    check_proof(proof_orth)
