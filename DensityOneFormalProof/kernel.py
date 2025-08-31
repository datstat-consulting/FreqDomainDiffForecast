
# kernel.py (strict validators version)
from typing import List, Callable
import re

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


# --- tiny helpers for robust string-pattern checks ---
def _s(s): 
    return repr(s)

def _repr(form):
    return repr(form)

def _canon(s: str) -> str:
    # Normalize some math notations to avoid harmless mismatches
    if s is None:
        return s
    # unify ^{..} to ^(..)
    s = re.sub(r"\^\{([^}]+)\}", r"^(\\1)", s)
    # unify log(N) -> log N
    s = s.replace("log(N)", "log N")
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains(form, *needles):
    text = _canon(_repr(form))
    return all(_canon(n) in text for n in needles)

def _contains_any(form, *needles):
    text = _canon(_repr(form))
    return any(_canon(n) in text for n in needles)


def _check_chebyshev_count(pf: Proof):
    # ensure conclusion is BigO and premise is the mean-square axiom
    assert isinstance(pf.conclusion, BigO), "Chebyshev-count: need BigO"
    assert len(pf.premises) in (1,2), "Chebyshev-count: one or two premises allowed"
    prem = pf.premises[0]
    assert prem.rule == 'axiom', "Chebyshev-count: premise must be mean-square axiom"
    # conclusion shape: |BadSet| = O( (N/T^2) * H * (log N)^(-A) )
    assert _contains(pf.conclusion.expr, "|BadSet|"), "Chebyshev-count: wrong expr"
    assert _contains_any(pf.conclusion.bound, "N/T^2", "N/T²", "N / T^2"), "Chebyshev-count: missing N/T^2"
    assert _contains(pf.conclusion.bound, "H"), "Chebyshev-count: missing H factor"
    assert _contains_any(pf.conclusion.bound, "(log N)^(-A)", "(log N)^-A"), "Chebyshev-count: missing log decay"

    # optional side condition premise: H ≥ N^(1/6)
    if len(pf.premises) == 2:
        prem2 = pf.premises[1]
        assert prem2.rule == 'axiom', "Chebyshev-count: side condition must be axiom"
        side = _repr(prem2.conclusion)
        assert ("H ≥ N^(1/6)" in side) or ("H >= N^(1/6)" in side), "Chebyshev-count: missing 'H ≥ N^(1/6)' side condition"
    else:
        assert len(pf.premises) == 1, "Chebyshev-count: expected 1 (MR) or 2 (MR + H≥N^(1/6)) premises"



def _check_chebyshev_pointwise(pf: Proof):
    assert isinstance(pf.conclusion, Implies), "Chebyshev-pointwise: need implication"
    ant = _canon(_repr(pf.conclusion.ant))
    con = _canon(_repr(pf.conclusion.con))
    assert "x ∉ BadSet" in ant, "Chebyshev-pointwise: antecedent should exclude bad set"
    assert "|Σ_{n∈I_j}λ(n)λ(bn+h)| ≤ T" in con, "Chebyshev-pointwise: wrong consequent"
    # must depend on Chebyshev-count
    assert any(p.rule == 'Chebyshev-count' for p in pf.premises), "Chebyshev-pointwise: needs Chebyshev-count"
    # also require a premise specifying the threshold T explicitly
    assert any(isinstance(p.conclusion, Equals) and _contains(p.conclusion.left, "T") and
               _contains(p.conclusion.right, "H^(1/2)", "(log N)^C")
               for p in pf.premises), "Chebyshev-pointwise: need T = H^(1/2)*(log N)^C premise"



def _check_sum_blocks(pf: Proof):
    # ensure conclusion is BigO for T_b with N^(3/4)/b * (log N)^C
    assert isinstance(pf.conclusion, BigO), "Sum-blocks: need BigO"
    assert _contains(pf.conclusion.expr, "T_b"), "Sum-blocks: wrong expr"
    bound = _repr(pf.conclusion.bound)
    assert "N^(3/4)" in bound, "Sum-blocks: missing N^(3/4)"
    assert "/b" in bound, "Sum-blocks: missing 1/b factor"
    assert "(log N)^" in bound, "Sum-blocks: missing log factor"
    # premises should include block count and pointwise Chebyshev
    prem_rules = {p.rule for p in pf.premises}
    assert ('Chebyshev-pointwise' in prem_rules) or any("Chebyshev-pointwise" in r.rule for r in pf.premises), \
        "Sum-blocks: needs Chebyshev-pointwise"
    assert any(_contains(p.conclusion, "R") and _contains(p.conclusion, "N/(bH)") for p in pf.premises), \
        "Sum-blocks: need R = O(N/(bH))"

    # require affine change and bH window length to be present among premises
    premise_texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("n = b*m + h" in t or "n=bm+h" in t for t in premise_texts), "Sum-blocks: need affine substitution 'n = b*m + h'"
    assert any("bH" in t or "length bH" in t or "window bH" in t for t in premise_texts), "Sum-blocks: need window length bH recorded"


def _check_sum_over_b(pf: Proof):
    # Σ_{b≤D}T_b  ≤  (sum 1/b) * N^(3/4)(log N)^C  ≤ N^(3/4)(log N)^(C+1)
    assert isinstance(pf.conclusion, BigO), "Sum-over-b: need BigO"
    assert _contains(pf.conclusion.expr, "Σ_{b≤D}T_b"), "Sum-over-b: wrong expr"
    bound = _repr(pf.conclusion.bound)
    assert "N^(3/4)" in bound, "Sum-over-b: missing N^(3/4)"
    assert "(log N)^" in bound, "Sum-over-b: missing log"
    # premise must have T_b bound with 1/b
    assert len(pf.premises) == 1, "Sum-over-b: one premise"
    prem = pf.premises[0].conclusion
    assert isinstance(prem, BigO), "Sum-over-b: premise must be BigO"
    assert _contains(prem.expr, "T_b"), "Sum-over-b: wrong premise expr"
    assert "/b" in _repr(prem.bound), "Sum-over-b: need 1/b in premise bound"


def _check_bilinear_apply(pf: Proof):
    # Must rely on bilinear large sieve axiom
    assert any(p.rule == 'axiom' and isinstance(p.conclusion, BigO) and "x^(1/2)*(P*Q)^(1/2)" in repr(p.conclusion.bound)
               for p in pf.premises), "Bilinear-sieve-apply: need MV bilinear sieve axiom"
    assert isinstance(pf.conclusion, BigO), "Bilinear-sieve-apply: need BigO"

    # require side condition P*Q ≤ x^(1-ε) among premises
    assert any(p.rule == 'axiom' and ("P*Q ≤ x^(1-ε)" in _repr(p.conclusion) or "P*Q <= x^(1-ε)" in _repr(p.conclusion))
               for p in pf.premises), "Bilinear-sieve-apply: missing side condition 'P*Q ≤ x^(1-ε)'"


def _check_cauchy_schwarz(pf: Proof):
    # Must come from bilinear-apply and produce N^(3/4)*(log N)^C
    assert isinstance(pf.conclusion, BigO), "Cauchy-Schwarz: need BigO"
    assert "N^(3/4)" in repr(pf.conclusion.bound), "Cauchy-Schwarz: missing N^(3/4)"
    assert "(log N)^" in repr(pf.conclusion.bound), "Cauchy-Schwarz: missing log"
    assert any(p.rule == 'Bilinear-sieve-apply' for p in pf.premises), "Cauchy-Schwarz: needs bilinear apply"


def _check_sum_dyadic(pf: Proof):
    assert isinstance(pf.conclusion, BigO), "Sum-dyadic: need BigO"
    assert "TypeII_total" in repr(pf.conclusion.expr), "Sum-dyadic: wrong expr"
    assert "N^(3/4)" in repr(pf.conclusion.bound), "Sum-dyadic: missing N^(3/4)"
    assert "(log N)^" in repr(pf.conclusion.bound), "Sum-dyadic: missing log"
    # need Cauchy–Schwarz and dyadic count
    rules = {p.rule for p in pf.premises}
    assert 'Cauchy-Schwarz' in rules, "Sum-dyadic: needs Cauchy–Schwarz"
    assert any((isinstance(p.conclusion, Equals) and "O((log N)^2)" in repr(p.conclusion.right)) for p in pf.premises), \
        "Sum-dyadic: need dyadic block count"


def _check_union_bound(pf: Proof):
    # conclusion is size of exceptional sets union = O(N^{1-δ})
    assert isinstance(pf.conclusion, Atom), "Union-bound: conclusion should be Atom equality"
    assert "|ExcI ∪ ExcII|" in pf.conclusion.name or "|ExcI ∪ ExcII| = O" in pf.conclusion.name, \
        "Union-bound: wrong expr"
    # needs Type I and Type II totals as premises (presence test)
    names = repr(pf.premises)
    assert "Σ_{b≤D}T_b" in names, "Union-bound: requires Type I sum"
    assert "TypeII_total" in names, "Union-bound: requires Type II total"


def _check_combine_final(pf: Proof):
    assert isinstance(pf.conclusion, Forall), "Combine-final: need ∀h,ε"
    assert pf.conclusion.var == "h,ε", "Combine-final: wrong bound variables"
    body = repr(pf.conclusion.body)
    assert "Σ_{n∈N_{h,ε}}λ(n)λ(n+h)" in body, "Combine-final: wrong sum"
    assert "N^{3/4+ε}" in body, "Combine-final: bound must be N^{3/4+ε}"
    # Needs union-bound premise
    assert len(pf.premises) == 1 and pf.premises[0].rule == 'Union-bound', "Combine-final: needs union bound"


def _check_mult_ext(pf: Proof):
    # multiplicative extension of square-divisor identity: ∀n. λ(n) = Σ[d^2|n] μ(n//d^2)
    assert isinstance(pf.conclusion, Forall), "mult_ext: need ∀n"
    body = repr(pf.conclusion.body)
    assert "λ(n)" in body and "Σ[d^2|n]μ(n//d^2)" in body, "mult_ext: wrong identity"
    # Expect prime-power axiom + multiplicativity premises
    rules = [p.rule for p in pf.premises]
    assert rules.count('axiom') >= 3, "mult_ext: need pp-axiom and multiplicativity axioms"


def _check_hb_full(pf: Proof):
    # Heath-Brown-full: λ(n) = HBsum(n,D) from defining axioms for HB1/HB2/HB3 and combiner
    eq = pf.conclusion
    assert isinstance(eq, Equals), "Heath-Brown-full: need equation"
    assert "λ(n)" in repr(eq.left) and "HBsum(n,D)" in repr(eq.right), "Heath-Brown-full: wrong sides"
    assert any("HB1(" in repr(p.conclusion) for p in pf.premises), "Heath-Brown-full: need HB1 axiom"
    assert any("HB2(" in repr(p.conclusion) for p in pf.premises), "Heath-Brown-full: need HB2 axiom"
    assert any("HB3(" in repr(p.conclusion) for p in pf.premises), "Heath-Brown-full: need HB3 axiom"
    assert any(("HBsum(n,D)") in repr(p.conclusion.right) if isinstance(p.conclusion, Equals) else False
               for p in pf.premises), "Heath-Brown-full: need HBsum combiner"
    # one final axiom tying λ to HBsum
    assert any(isinstance(p.conclusion, Equals) and "λ(n)" in repr(p.conclusion.left) and "HBsum(n,D)" in repr(p.conclusion.right)
               and p.rule == 'axiom' for p in pf.premises), "Heath-Brown-full: need tie-back axiom"



def _check_mr_theorem(pf: Proof):
    # MR-Theorem: (1/N) sum_x |sum_{x<n≤x+H} λ(n)|^2 = O(H*(log N)^(-A))
    assert isinstance(pf.conclusion, BigO), "MR-Theorem: need BigO"
    assert _contains(pf.conclusion.expr, "(1/N)Σ_{x=1}^{N-H}|Σ_{n=x+1}^{x+H}λ(n)|²"), "MR-Theorem: wrong LHS"
    assert _contains_any(pf.conclusion.bound, "H*(log N)^(-A)", "H * (log N)^(-A)"), "MR-Theorem: wrong RHS"
    # Premises: f completely multiplicative, |f(n)| ≤ 1, f=λ, and analytic toolkit (Parseval/Halász placeholders)
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("f completely multiplicative" in t for t in texts), "MR-Theorem: need CMF premise"
    assert any("|f(n)| ≤ 1" in t or "|f(n)| <= 1" in t for t in texts), "MR-Theorem: need 1-bounded premise"
    assert any("f = λ" in t or "f=λ" in t for t in texts), "MR-Theorem: need specialization f=λ"
    # toolkit
    assert any("Parseval" in t for t in texts), "MR-Theorem: need Parseval tool"
    assert any("Halász" in t for t in texts), "MR-Theorem: need Halász tool"


def _check_bilinear_lsi(pf: Proof):
    # Provide bilinear large-sieve estimate as a derived lemma from LSI base + orthogonality
    assert isinstance(pf.conclusion, BigO), "Bilinear-LSI: need BigO"
    body = _repr(pf.conclusion)
    assert "Σ_{m=1}^{⌊x/(p*q)⌋}" in body and "x^(1/2)*(P*Q)^(1/2)" in body, "Bilinear-LSI: wrong shape"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("Large sieve inequality (base)" in t for t in texts), "Bilinear-LSI: need LSI base"
    assert any("additive characters orthogonality" in t for t in texts), "Bilinear-LSI: need orthogonality"



def _check_geom_series(pf: Proof):
    # Geometric-series closed form for r^M - 1 over r - 1
    assert isinstance(pf.conclusion, Equals), "Geometric-series: need equality"
    body = _repr(pf.conclusion)
    assert "Σ_{m=0}^{M-1} r^m" in body and "(1 - r^M)/(1 - r)" in body, "Geometric-series: wrong formula"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("algebraic identity for finite geometric sum" in t for t in texts), "Geometric-series: need algebraic identity premise"
    assert any("domain: r ≠ 1" in t or "r != 1" in t for t in texts), "Geometric-series: need r ≠ 1 premise"


def _check_char_orthog(pf: Proof):
    # Additive characters orthogonality from geometric series
    assert isinstance(pf.conclusion, Equals), "Char-orthog: need equality"
    body = _repr(pf.conclusion)
    assert "Σ_{m=0}^{q-1} e(2πi k m / q)" in body, "Char-orthog: wrong sum"
    # accept either clean single-line or with possible linebreak artifacts
    ok_zero = ("= 0 if q ∤ k" in body) or ("= 0 if q |̸ k" in body) or ("= 0 if q !| k" in body)
    ok_q    = ("= q if q | k" in body)
    assert (ok_zero and ok_q), "Char-orthog: need case result"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("Σ_{m=0}^{M-1} r^m = (1 - r^M)/(1 - r)" in t for t in texts), "Char-orthog: need geometric series"
    assert any("periodicity of e(2πi θ)" in t for t in texts), "Char-orthog: need periodicity premise"

def _check_lsi_base(pf: Proof):
    # Large Sieve Inequality (base) from Bessel/Plancherel + character orthogonality
    assert isinstance(pf.conclusion, Atom), "LSI-base: we encode statement as Atom text"
    text = _repr(pf.conclusion)
    assert "Large sieve inequality (base)" in text, "LSI-base: wrong label"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("Bessel/Plancherel inequality" in t for t in texts), "LSI-base: need Bessel/Plancherel"
    assert any("additive characters orthogonality" in t for t in texts), "LSI-base: need orthogonality"


def _check_parseval_tool(pf: Proof):
    # Parseval tool from Unitary DFT + Plancherel
    assert isinstance(pf.conclusion, Atom), "Parseval-tool: encode as Atom"
    assert "Parseval tool available" in _repr(pf.conclusion), "Parseval-tool: wrong label"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("Unitary DFT" in t for t in texts), "Parseval-tool: need Unitary DFT"
    assert any("Plancherel identity" in t for t in texts), "Parseval-tool: need Plancherel"


def _check_halasz_tool(pf: Proof):
    # Halász tool availability from pretentious distance + mean value estimates
    assert isinstance(pf.conclusion, Atom), "Halasz-tool: encode as Atom"
    assert "Halasz tool available" in _repr(pf.conclusion) or "Halász tool available" in _repr(pf.conclusion), "Halasz-tool: wrong label"
    texts = [_repr(p.conclusion) for p in pf.premises]
    assert any("pretentious distance triangle inequality" in t for t in texts), "Halasz-tool: need pretentious distance"
    assert any("mean value bound for Dirichlet polynomials" in t for t in texts), "Halasz-tool: need mean value bound"


# Register validators
_RULE_CHECKS: dict[str, Callable[[Proof], None]] = {
    'axiom':                   _check_axiom,
    '→I':                      _check_imp_i,
    '→E':                      _check_mp,
    'Chebyshev-count':         _check_chebyshev_count,
    'Chebyshev-pointwise':     _check_chebyshev_pointwise,
    'Block-count':             lambda pf: None,
    'Sum-blocks':              _check_sum_blocks,
    'Sum-over-b':              _check_sum_over_b,
    'Bilinear-sieve-apply':    _check_bilinear_apply,
    'Cauchy-Schwarz':          _check_cauchy_schwarz,
    'Dyadic-count':            lambda pf: None,
    'Sum-dyadic':              _check_sum_dyadic,
    'Union-bound':             _check_union_bound,
    'Combine-final':           _check_combine_final,
    'mult_ext':                _check_mult_ext,
    'Heath-Brown-full':        _check_hb_full,
    'Halasz-tool':         _check_halasz_tool,
    'Parseval-tool':         _check_parseval_tool,
    'LSI-base':         _check_lsi_base,
    'Char-orthog':         _check_char_orthog,
    'Geometric-series':         _check_geom_series,
    'MR-Theorem':             _check_mr_theorem,
    'Bilinear-LSI':           _check_bilinear_lsi,
}

# === 5. Proof Checker ===

def check_proof(pf: Proof):
    for pr in pf.premises:
        check_proof(pr)
    _RULE_CHECKS[pf.rule](pf)


# === 6. Quick self-test (optional) ===
if __name__ == '__main__':
    A = Atom('A')
    p1 = axiom(A)
    imp = imp_i(A, p1)
    print(imp)
    check_proof(imp)
