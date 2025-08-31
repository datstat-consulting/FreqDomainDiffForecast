
# parseval_derive.py
from kernel import Atom, combine, axiom, check_proof

prem_unitary = axiom(Atom("Unitary DFT"))
prem_planch = axiom(Atom("Plancherel identity"))
parseval_tool = combine(
    "Parseval-tool",
    Atom("Parseval tool available"),
    prem_unitary, prem_planch
)

if __name__ == "__main__":
    print("=== Parseval tool ===")
    print(parseval_tool)
    check_proof(parseval_tool)
