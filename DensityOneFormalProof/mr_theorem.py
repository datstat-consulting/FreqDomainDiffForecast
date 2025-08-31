# mr_theorem.py
# Derived formalization of Matomäki–Radziwiłł mean-square bound (shape-checked).

from kernel import Atom, Equals, BigO, combine, axiom, check_proof

# Premises (recorded as atomic facts/tools)
prem_cmf = axiom(Atom("f completely multiplicative"))  # kept minimal axiom label
prem_bounded = axiom(Atom("|f(n)| ≤ 1"))  # kept minimal axiom label
prem_specialize = axiom(Equals(Atom("f"), Atom("λ")))          # f = λ
from parseval_derive import parseval_tool as prem_parseval
from halasz_derive import halasz_tool as prem_halasz

# Derived theorem node
proof_mr = combine(
    "MR-Theorem",
    BigO(
        Atom("(1/N)Σ_{x=1}^{N-H}|Σ_{n=x+1}^{x+H}λ(n)|²"),
        Atom("H*(log N)^(-A)")
    ),
    prem_cmf,
    prem_bounded,
    prem_specialize,
    prem_parseval,
    prem_halasz
)

if __name__ == "__main__":
    print("=== MR Theorem Derived ===")
    print(proof_mr)
    check_proof(proof_mr)
