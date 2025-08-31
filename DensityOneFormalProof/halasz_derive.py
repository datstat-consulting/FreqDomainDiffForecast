
# halasz_derive.py
from kernel import Atom, combine, axiom, check_proof

prem_pret = axiom(Atom("pretentious distance triangle inequality"))
prem_mvd = axiom(Atom("mean value bound for Dirichlet polynomials"))
halasz_tool = combine(
    "Halasz-tool",
    Atom("Halász tool available"),
    prem_pret, prem_mvd
)

if __name__ == "__main__":
    print("=== Halasz tool ===")
    print(halasz_tool)
    check_proof(halasz_tool)
