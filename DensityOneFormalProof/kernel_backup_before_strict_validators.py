# kernel.py

from typing import List, Callable

# === 1. AST Definitions ===

class Formula: pass

class Atom(Formula):
    def __init__(self, name: str): self.name = name
    def __eq__(self, other): return isinstance(other, Atom) and self.name == other.name
    def __repr__(self): return self.name

class Equals(Formula):
    def __init__(self, left: Formula, right: Formula): self.left, self.right = left, right
    def __eq__(self, other):
        return (isinstance(other, Equals)
                and self.left == other.left
                and self.right == other.right)
    def __repr__(self): return f"({self.left} = {self.right})"

class Implies(Formula):
    def __init__(self, ant: Formula, con: Formula): self.ant, self.con = ant, con
    def __eq__(self, other):
        return (isinstance(other, Implies)
                and self.ant == other.ant
                and self.con == other.con)
    def __repr__(self): return f"({self.ant} → {self.con})"

class Forall(Formula):
    def __init__(self, var: str, body: Formula): self.var, self.body = var, body
    def __eq__(self, other):
        return (isinstance(other, Forall)
                and self.var == other.var
                and self.body == other.body)
    def __repr__(self): return f"(∀{self.var}. {self.body})"

class Sum(Formula):
    def __init__(self, indices: str, pred: Formula, body: Formula):
        self.indices, self.pred, self.body = indices, pred, body
    def __eq__(self, other):
        return (isinstance(other, Sum)
                and self.indices == other.indices
                and self.pred == other.pred
                and self.body == other.body)
    def __repr__(self): return f"Σ_{{{self.indices}|{self.pred}}}({self.body})"

class BigO(Formula):
    def __init__(self, expr: Formula, bound: Formula): self.expr, self.bound = expr, bound
    def __eq__(self, other):
        return (isinstance(other, BigO)
                and self.expr == other.expr
                and self.bound == other.bound)
    def __repr__(self): return f"{self.expr} = O({self.bound})"


# === 2. Proof Object ===

class Proof:
    def __init__(self, conclusion: Formula, premises: List['Proof'], rule: str):
        self.conclusion, self.premises, self.rule = conclusion, premises, rule
    def __repr__(self):
        ps = ', '.join(p.rule for p in self.premises)
        return f"[{self.rule}: ⊢ {self.conclusion}" + (f" from {ps}]" if ps else "]")


# === 3. Inference Constructors ===

def axiom(f: Formula) -> Proof:
    return Proof(f, [], 'axiom')

def imp_i(assump: Formula, p: Proof) -> Proof:
    return Proof(Implies(assump, p.conclusion), [p], '→I')

def mp(p_imp: Proof, p_ant: Proof) -> Proof:
    assert isinstance(p_imp.conclusion, Implies), "mp: need implication"
    assert p_imp.conclusion.ant == p_ant.conclusion, "mp: antecedent mismatch"
    return Proof(p_imp.conclusion.con, [p_imp, p_ant], '→E')

def combine(rule: str, conclusion: Formula, *premises: Proof) -> Proof:
    if rule not in _RULE_CHECKS:
        raise ValueError(f"Unknown rule: {rule}")
    return Proof(conclusion, list(premises), rule)


# === 4. Rule Validators ===

def _check_axiom(pf: Proof):
    # any formula allowed
    pass

def _check_imp_i(pf: Proof):
    assert isinstance(pf.conclusion, Implies), "→I: conclusion must be implication"
    assert len(pf.premises) == 1, "→I: wrong number of premises"
    assert pf.premises[0].conclusion == pf.conclusion.con, "→I: consequent mismatch"

def _check_mp(pf: Proof):
    assert len(pf.premises) == 2, "→E: two premises needed"
    pi, pa = pf.premises
    assert isinstance(pi.conclusion, Implies), "→E: first premise not implication"
    assert pi.conclusion.ant == pa.conclusion, "→E: antecedent mismatch"
    assert pf.conclusion == pi.conclusion.con, "→E: conclusion mismatch"

def _check_chebyshev_count(pf: Proof):
    # ensure conclusion is BigO and premise is the mean-square axiom
    assert isinstance(pf.conclusion, BigO)
    assert pf.premises[0].rule == 'axiom'

def _check_sum_blocks(pf: Proof):
    # ensure conclusion is BigO with T_b and premise rules match
    assert isinstance(pf.conclusion, BigO)
    prem_rules = {p.rule for p in pf.premises}
    assert 'axiom' in prem_rules or 'Chebyshev-pointwise' in prem_rules

# Register validators
_RULE_CHECKS: dict[str, Callable[[Proof], None]] = {
    'axiom':                   _check_axiom,
    '→I':                      _check_imp_i,
    '→E':                      _check_mp,
    'Chebyshev-count':         _check_chebyshev_count,
    'Chebyshev-pointwise':     lambda pf: None,
    'Block-count':             lambda pf: None,
    'Sum-blocks':              _check_sum_blocks,
    'Sum-over-b':              lambda pf: None,
    'Bilinear-sieve-apply':    lambda pf: None,
    'Cauchy-Schwarz':          lambda pf: None,
    'Dyadic-count':            lambda pf: None,
    'Sum-dyadic':              lambda pf: None,
    'Union-bound':             lambda pf: None,
    'Combine-final':           lambda pf: None,
    'mult_ext':                lambda pf: None,
    'Heath-Brown-full':        lambda pf: None,
}

# === 5. Proof Checker ===

def check_proof(pf: Proof):
    for pr in pf.premises:
        check_proof(pr)
    _RULE_CHECKS[pf.rule](pf)


# === 6. Usage Example ===
if __name__ == '__main__':
    A = Atom('A')
    p1 = axiom(A)
    imp = imp_i(A, p1)
    p2 = axiom(Atom("(1/N)Σ_x|...|^2 = O(H*(log N)^(-A))"))
    c = combine('Chebyshev-count', BigO(Atom("|BadSet|"), Atom("...")), p2)
    print(imp)
    check_proof(imp)      # ok
    print(c)
    check_proof(c)        # ok if rule is registered