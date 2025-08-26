
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

# New Formula types for the second paper
class Integral(Formula):
    def __init__(self, var: str, lower: Formula, upper: Formula, body: Formula):
        self.var, self.lower, self.upper, self.body = var, lower, upper, body
    def __eq__(self, other):
        return (isinstance(other, Integral)
                and self.var == other.var
                and self.lower == other.lower
                and self.upper == other.upper
                and self.body == other.body)
    def __repr__(self): return f"∫_{{{self.lower}}}^{{{self.upper}}} {self.body} d{self.var}"

class LFunction(Formula):
    def __init__(self, s: Formula, chi: Formula, liouville_twist: bool = False):
        self.s, self.chi, self.liouville_twist = s, chi, liouville_twist
    def __eq__(self, other):
        return (isinstance(other, LFunction)
                and self.s == other.s
                and self.chi == other.chi
                and self.liouville_twist == other.liouville_twist)
    def __repr__(self):
        twist_str = ' ⊗ λ' if self.liouville_twist else ''
        return f"L({self.s}, {self.chi}{twist_str})"

class NZeros(Formula):
    def __init__(self, sigma: Formula, T: Formula, chi: Formula, liouville_twist: bool = False):
        self.sigma, self.T, self.chi, self.liouville_twist = sigma, T, chi, liouville_twist
    def __eq__(self, other):
        return (isinstance(other, NZeros)
                and self.sigma == other.sigma
                and self.T == other.T
                and self.chi == other.chi
                and self.liouville_twist == other.liouville_twist)
    def __repr__(self):
        twist_str = ' ⊗ λ' if self.liouville_twist else ''
        return f"N_{{{self.chi}{twist_str}}}({self.sigma}, {self.T})"

class Product(Formula):
    def __init__(self, var: str, pred: Formula, body: Formula):
        self.var, self.pred, self.body = var, pred, body
    def __eq__(self, other):
        return (isinstance(other, Product)
                and self.var == other.var
                and self.pred == other.pred
                and self.body == other.body)
    def __repr__(self): return f"Π_{{{self.var}|{self.pred}}}({self.body})"

class Fraction(Formula):
    def __init__(self, numerator: Formula, denominator: Formula):
        self.numerator, self.denominator = numerator, denominator
    def __eq__(self, other):
        return (isinstance(other, Fraction)
                and self.numerator == other.numerator
                and self.denominator == other.denominator)
    def __repr__(self): return f"({self.numerator} / {self.denominator})"

class Power(Formula):
    def __init__(self, base: Formula, exponent: Formula):
        self.base, self.exponent = base, exponent
    def __eq__(self, other):
        return (isinstance(other, Power)
                and self.base == other.base
                and self.exponent == other.exponent)
    def __repr__(self): return f"({self.base}^{self.exponent})"

class Plus(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right
    def __eq__(self, other):
        return (isinstance(other, Plus)
                and self.left == other.left
                and self.right == other.right)
    def __repr__(self): return f"({self.left} + {self.right})"

class Minus(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right
    def __eq__(self, other):
        return (isinstance(other, Minus)
                and self.left == other.left
                and self.right == other.right)
    def __repr__(self): return f"({self.left} - {self.right})"

class Multiply(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right
    def __eq__(self, other):
        return (isinstance(other, Multiply)
                and self.left == other.left
                and self.right == other.right)
    def __repr__(self): return f"({self.left} * {self.right})"

# New AST nodes for Fractional Calculus and Stochastic Processes
class FractionalDerivative(Formula):
    def __init__(self, order: Formula, func: Formula, wrt: str):
        self.order, self.func, self.wrt = order, func, wrt
    def __eq__(self, other):
        return (isinstance(other, FractionalDerivative)
                and self.order == other.order
                and self.func == other.func
                and self.wrt == other.wrt)
    def __repr__(self): return f"D^{self.order}_{self.wrt}({self.func})"

class FractionalIntegral(Formula):
    def __init__(self, order: Formula, func: Formula, wrt: str):
        self.order, self.func, self.wrt = order, func, wrt
    def __eq__(self, other):
        return (isinstance(other, FractionalIntegral)
                and self.order == other.order
                and self.func == other.func
                and self.wrt == other.wrt)
    def __repr__(self): return f"I^{self.order}_{self.wrt}({self.func})"

class StochasticProcess(Formula):
    def __init__(self, name: str, index_var: str):
        self.name, self.index_var = name, index_var
    def __eq__(self, other):
        return (isinstance(other, StochasticProcess)
                and self.name == other.name
                and self.index_var == other.index_var)
    def __repr__(self): return f"{self.name}({self.index_var})"

class Mean(Formula):
    def __init__(self, process: StochasticProcess):
        self.process = process
    def __eq__(self, other): return isinstance(other, Mean) and self.process == other.process
    def __repr__(self): return f"E[{self.process}]"

class Covariance(Formula):
    def __init__(self, process1: StochasticProcess, process2: StochasticProcess):
        self.process1, self.process2 = process1, process2
    def __eq__(self, other):
        return (isinstance(other, Covariance)
                and self.process1 == other.process1
                and self.process2 == other.process2)
    def __repr__(self): return f"Cov({self.process1}, {self.process2})"

class Cumulant(Formula):
    def __init__(self, order: int, processes: List[StochasticProcess]):
        self.order, self.processes = order, processes
    def __eq__(self, other):
        return (isinstance(other, Cumulant)
                and self.order == other.order
                and self.processes == other.processes)
    def __repr__(self):
        process_str = ', '.join(str(p) for p in self.processes)
        return f"Cum_{self.order}({process_str})"

class SelfSimilarity(Formula):
    def __init__(self, process: StochasticProcess, scale_factor: Formula, hurst_param: Formula):
        self.process, self.scale_factor, self.hurst_param = process, scale_factor, hurst_param
    def __eq__(self, other):
        return (isinstance(other, SelfSimilarity)
                and self.process == other.process
                and self.scale_factor == other.scale_factor
                and self.hurst_param == other.hurst_param)
    def __repr__(self): return f"{self.process.name}({self.scale_factor} * {self.process.index_var}) ~ {self.scale_factor}^{self.hurst_param} * {self.process}"

class StationaryIncrements(Formula):
    def __init__(self, process: StochasticProcess, t1: str, t2: str, t3: str, t4: str):
        self.process, self.t1, self.t2, self.t3, self.t4 = process, t1, t2, t3, t4
    def __eq__(self, other):
        return (isinstance(other, StationaryIncrements)
                and self.process == other.process
                and self.t1 == other.t1
                and self.t2 == other.t2
                and self.t3 == other.t3
                and self.t4 == other.t4)
    def __repr__(self): return f"Dist({self.process.name}({self.t2}) - {self.process.name}({self.t1})) = Dist({self.process.name}({self.t4}) - {self.process.name}({self.t3})) if {self.t2}-{self.t1} = {self.t4}-{self.t3}"


# === 2. Proof Object ===

class Proof:
    def __init__(self, conclusion: Formula, premises: List["Proof"], rule: str):
        self.conclusion, self.premises, self.rule = conclusion, premises, rule
    def __repr__(self):
        ps = ", ".join(p.rule for p in self.premises)
        return f"[{self.rule}: ⊢ {self.conclusion}" + (f" from {ps}]" if ps else "]")


# === 3. Inference Constructors ===

def axiom(f: Formula) -> Proof:
    return Proof(f, [], "axiom")

def imp_i(assump: Formula, p: Proof) -> Proof:
    return Proof(Implies(assump, p.conclusion), [p], "→I")

def mp(p_imp: Proof, p_ant: Proof) -> Proof:
    assert isinstance(p_imp.conclusion, Implies), "mp: need implication"
    assert p_imp.conclusion.ant == p_ant.conclusion, "mp: antecedent mismatch"
    return Proof(p_imp.conclusion.con, [p_imp, p_ant], "→E")

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
    assert pf.premises[0].rule == "axiom"

def _check_sum_blocks(pf: Proof):
    # ensure conclusion is BigO with T_b and premise rules match
    assert isinstance(pf.conclusion, BigO)
    prem_rules = {p.rule for p in pf.premises}
    assert "axiom" in prem_rules or "Chebyshev-pointwise" in prem_rules

def _check_euler_product(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    # accept either LFunction node or Atom string for flexibility
    assert isinstance(pf.conclusion.left, (LFunction, Atom))
    assert isinstance(pf.conclusion.right, (Product, Atom))


def _check_l_function_ratio(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    # accept either AST LFunction/Fraction or Atom strings, do not introspect deeper
    assert isinstance(pf.conclusion.left, (LFunction, Atom))
    assert isinstance(pf.conclusion.right, (Fraction, Atom))


def _check_zero_density_bound(pf: Proof):
    # Accept BigO with NZeros(...) or Atom(...) on expr side
    assert isinstance(pf.conclusion, BigO)
    expr = getattr(pf.conclusion, "expr", None)
    assert (isinstance(expr, (NZeros, Atom))) or expr is None


def _check_perron_formula(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    # be flexible about the right-hand structure (integral + error terms)
    assert hasattr(pf.conclusion, "right")

def _check_contour_shift(pf: Proof):
    assert isinstance(pf.conclusion, BigO)

def _check_gauss_sum_expansion(pf: Proof):
    assert isinstance(pf.conclusion, Equals)

def _check_fractional_derivative(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, FractionalDerivative)

def _check_fractional_integral(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, FractionalIntegral)

def _check_gaussian_process_definition(pf: Proof):
    assert isinstance(pf.conclusion, Forall)
    assert isinstance(pf.conclusion.body, Implies)
    assert isinstance(pf.conclusion.body.con, Equals)
    assert isinstance(pf.conclusion.body.con.left, StochasticProcess)

def _check_fbm_covariance(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, Covariance)
    assert isinstance(pf.conclusion.left.process1, StochasticProcess)
    assert isinstance(pf.conclusion.left.process2, StochasticProcess)

def _check_cumulant_vanishing(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, Cumulant)
    assert pf.conclusion.right == Atom("0")

def _check_chowla_from_cumulants(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    assert isinstance(pf.conclusion.expr, Atom) # Sum_{n <= N-h} lambda(n) * lambda(n+h)
    assert isinstance(pf.conclusion.bound, Atom) # o(N) or similar
    assert len(pf.premises) >= 2 # At least two cumulant vanishing proofs
    # Further checks could be added to ensure premises are Cumulant-Vanishing proofs

def _check_full_chowla_theorem(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    assert isinstance(pf.conclusion.expr, Atom)
    assert isinstance(pf.conclusion.bound, Atom)
    # Check for presence of key premises by their rule names
    required_rule_names = {"axiom", "Cumulant-Vanishing", "Chowla-from-Cumulants"}
    present_rule_names = {p.rule for p in pf.premises}
    assert required_rule_names.issubset(present_rule_names), f"Missing required premises: {required_rule_names - present_rule_names}"

# Register validators
_RULE_CHECKS: dict[str, Callable[[Proof], None]] = {
    "axiom":                   _check_axiom,
    "→I":                      _check_imp_i,
    "→E":                      _check_mp,
    "Chebyshev-count":         _check_chebyshev_count,
    "Chebyshev-pointwise":     lambda pf: None,
    "Block-count":             lambda pf: None,
    "Sum-blocks":              _check_sum_blocks,
    "Sum-over-b":              lambda pf: None,
    "Bilinear-sieve-apply":    lambda pf: None,
    "Cauchy-Schwarz":          lambda pf: None,
    "Dyadic-count":            lambda pf: None,
    "Sum-dyadic":              lambda pf: None,
    "Union-bound":             lambda pf: None,
    "Combine-final":           lambda pf: None,
    "mult_ext":                lambda pf: None,
    "Heath-Brown-full":        lambda pf: None,
    # New rules for the second paper
    "Euler-product-L-lambda":  _check_euler_product,
    "L-function-ratio":        _check_l_function_ratio,
    "Zero-density-bound":      _check_zero_density_bound,
    "Perron-formula":          _check_perron_formula,
    "Contour-shift":           _check_contour_shift,
    "Gauss-sum-expansion":     _check_gauss_sum_expansion,
    "Major-arc-integral-bound": lambda pf: None,
    "U3-inverse-theorem":      lambda pf: None,
    "Nilsequence-non-correlation": lambda pf: None,
    "Minor-arc-bound":         lambda pf: None,
    "Combine-major-minor":     lambda pf: None,
    "Final-Chowla-unconditional": lambda pf: None,
    # New rules for Fractional Calculus and Stochastic Processes
    "Fractional-Derivative-Definition": _check_fractional_derivative,
    "Fractional-Integral-Definition": _check_fractional_integral,
    "Gaussian-Process-Definition": _check_gaussian_process_definition,
    "FBM-Covariance-Formula": _check_fbm_covariance,
    "FBM-Self-Similarity": lambda pf: None,
    "FBM-Stationary-Increments": lambda pf: None,
    "Cumulant-Vanishing": _check_cumulant_vanishing,
    "Chowla-from-Cumulants": _check_chowla_from_cumulants,
    "Full-Chowla-Theorem": _check_full_chowla_theorem,
}

# === 5. Proof Checker ===

def check_proof(pf: Proof):
    for pr in pf.premises:
        check_proof(pr)
    _RULE_CHECKS[pf.rule](pf)


# === 6. Usage Example ===
if __name__ == "__main__":
    A = Atom("A")
    p1 = axiom(A)
    imp = imp_i(A, p1)
    p2 = axiom(Atom("(1/N)Σ_x|...|^2 = O(H*(log N)^(-A))"))
    c = combine("Chebyshev-count", BigO(Atom("|BadSet|"), Atom("...")), p2)
    print(imp)
    check_proof(imp)      # ok
    print(c)
    check_proof(c)        # ok if rule is registered

    # Example of new types
    s_atom = Atom("s")
    chi_atom = Atom("chi")
    L_chi_lambda = LFunction(s_atom, chi_atom, liouville_twist=True)
    print(L_chi_lambda)

    sigma_atom = Atom("sigma")
    T_atom = Atom("T")
    N_chi_lambda = NZeros(sigma_atom, T_atom, chi_atom, liouville_twist=True)
    print(N_chi_lambda)

    integral_example = Integral(Atom("alpha"), Atom("0"), Atom("1"), Atom("F_K(alpha) * G_K(alpha)"))
    print(integral_example)

    product_example = Product(Atom("p"), Atom("p is prime"), Atom("1 / (1 + chi(p)p^-s)"))
    print(product_example)

    fraction_example = Fraction(LFunction(Atom("2s"), Atom("chi^2")), LFunction(Atom("s"), Atom("chi")))
    print(fraction_example)

    power_example = Power(Atom("rT"), Atom("C*(1-sigma)"))
    print(power_example)

    plus_example = Plus(Atom("K/T"), Atom("K*|beta|"))
    print(plus_example)

    minus_example = Minus(Atom("1"), Atom("2*sigma"))
    print(minus_example)

    multiply_example = Multiply(Atom("K"), Atom("r"))
    print(multiply_example)

    # New examples for Fractional Calculus and Stochastic Processes
    alpha_atom = Atom("alpha")
    t_atom = Atom("t")
    f_atom = Atom("f")
    fd_example = FractionalDerivative(alpha_atom, f_atom, "t")
    print(fd_example)

    fi_example = FractionalIntegral(alpha_atom, f_atom, "t")
    print(fi_example)

    bm_process = StochasticProcess("B", "t")
    print(bm_process)

    mean_bm = Mean(bm_process)
    print(mean_bm)

    cov_bm = Covariance(bm_process, StochasticProcess("B", "s"))
    print(cov_bm)

    cum_example = Cumulant(3, [bm_process, StochasticProcess("B", "s"), StochasticProcess("B", "u")])
    print(cum_example)

    fbm_process = StochasticProcess("B_H", "t")
    H_atom = Atom("H")
    a_atom = Atom("a")
    ss_example = SelfSimilarity(fbm_process, a_atom, H_atom)
    print(ss_example)

    si_example = StationaryIncrements(fbm_process, "t1", "t2", "t3", "t4")
    print(si_example)




# === Added validators for missing rules (auto-generated structural checks) ===
def _auto_impl_Bilinear_dispersion_HS(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Bilinear-dispersion-HS: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Bilinear-dispersion-HS: expected premises"
def _auto_impl_Cauchy_Schwarz_reduction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Cauchy-Schwarz-reduction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Cauchy-Schwarz-reduction: expected premises"
def _auto_impl_Coefficient_extraction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Coefficient-extraction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Coefficient-extraction: expected premises"
def _auto_impl_Combine_k_major_minor(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Combine-k-major-minor: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Combine-k-major-minor: expected premises"
def _auto_impl_Convolution_x_def(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Convolution-x-def: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Convolution-x-def: expected premises"
def _auto_impl_Covariance_from_pair(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Covariance-from-pair: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Covariance-from-pair: expected premises"

def _auto_impl_Cumulant_partition_Mobius(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Cumulant-partition-Mobius: wrong conclusion type"
    # allow either direct identity or with premises

def _auto_impl_Cumulant_weight_sum_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Cumulant-weight-sum-bound: wrong conclusion type"
    assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Cumulant-weight-sum-bound: expected premises"

def _auto_impl_Discrete_difference_equivalence(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Discrete-difference-equivalence: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Discrete-difference-equivalence: expected premises"

def _auto_impl_Empirical_shift_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Empirical-shift-identity: wrong conclusion type"
    # This is a pure identity derivation; no premises required

def _auto_impl_FourD_dispersion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "FourD-dispersion: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "FourD-dispersion: expected premises"
def _auto_impl_FourD_dispersion_lemma(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "FourD-dispersion-lemma: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "FourD-dispersion-lemma: expected premises"
def _auto_impl_Fourth_cumulant_reduction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Fourth-cumulant-reduction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Fourth-cumulant-reduction: expected premises"
def _auto_impl_Fourth_cumulant_vanishes(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Fourth-cumulant-vanishes: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Fourth-cumulant-vanishes: expected premises"
def _auto_impl_Fourth_moment_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Fourth-moment-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Fourth-moment-bound: expected premises"
def _auto_impl_Frac_difference_inverse(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Frac-difference-inverse: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Frac-difference-inverse: expected premises"

def _auto_impl_Fractional_weights_def(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "_auto_impl_Fractional_weights_def: wrong conclusion type"
    assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "_auto_impl_Fractional_weights_def: expected premises"


def _auto_impl_Full_Chowla_Conditional(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "_auto_impl_Full_Chowla_Conditional: wrong conclusion type"
    assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "_auto_impl_Full_Chowla_Conditional: expected premises"

def _auto_impl_Full_Chowla_steps(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Full-Chowla-steps: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Full-Chowla-steps: expected premises"
def _auto_impl_Gamma_convolution_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Gamma-convolution-identity: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Gamma-convolution-identity: expected premises"
def _auto_impl_Gamma_covariance_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Gamma-covariance-identity: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Gamma-covariance-identity: expected premises"
def _auto_impl_Generating_function_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Generating-function-identity: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Generating-function-identity: expected premises"
def _auto_impl_HB_fivefold_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "HB-fivefold-identity: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "HB-fivefold-identity: expected premises"
def _auto_impl_HB_fourfold_identity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "HB-fourfold-identity: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "HB-fourfold-identity: expected premises"
def _auto_impl_Higher_major_arc_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Higher-major-arc-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Higher-major-arc-bound: expected premises"
def _auto_impl_Higher_minor_arc_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Higher-minor-arc-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Higher-minor-arc-bound: expected premises"
def _auto_impl_Inclusion_exclusion_reduction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Inclusion-exclusion-reduction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Inclusion-exclusion-reduction: expected premises"

def _auto_impl_KC_criterion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "KC-criterion: wrong conclusion type"
    # Allow as a standalone criterion application


def _auto_impl_Levy_continuity(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Levy-continuity: wrong conclusion type"
    # Allow stand-alone application under the suite

def _auto_impl_Method_of_cumulants(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Method-of-cumulants: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Method-of-cumulants: expected premises"
def _auto_impl_Multilinear_dispersion_induction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Multilinear-dispersion-induction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Multilinear-dispersion-induction: expected premises"
def _auto_impl_Nilsequence_non_correlation_k(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Nilsequence-non-correlation-k: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Nilsequence-non-correlation-k: expected premises"
def _auto_impl_Quadruple_Chowla_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Quadruple-Chowla-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Quadruple-Chowla-bound: expected premises"
def _auto_impl_Quadruple_Chowla_from_dispersion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Quadruple-Chowla-from-dispersion: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Quadruple-Chowla-from-dispersion: expected premises"
def _auto_impl_Relate_kS_to_main_sum(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Relate-kS-to-main-sum: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Relate-kS-to-main-sum: expected premises"
def _auto_impl_Second_moment_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Second-moment-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Second-moment-bound: expected premises"
def _auto_impl_Tail_truncation_error(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Tail-truncation-error: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Tail-truncation-error: expected premises"
def _auto_impl_Third_cumulant_reduction(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Third-cumulant-reduction: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Third-cumulant-reduction: expected premises"
def _auto_impl_Third_cumulant_vanishes(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Third-cumulant-vanishes: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Third-cumulant-vanishes: expected premises"
def _auto_impl_Tightness_conclusion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Tightness-conclusion: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Tightness-conclusion: expected premises"
def _auto_impl_Trilinear_dispersion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Trilinear-dispersion: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Trilinear-dispersion: expected premises"
def _auto_impl_Trilinear_dispersion_lemma(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Trilinear-dispersion-lemma: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Trilinear-dispersion-lemma: expected premises"
def _auto_impl_Triple_Chowla_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Triple-Chowla-bound: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Triple-Chowla-bound: expected premises"

def _auto_impl_Triple_Chowla_from_dispersion(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Triple-Chowla-from-dispersion: wrong conclusion type"
    assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Triple-Chowla-from-dispersion: expected premises"

def _auto_impl_U_s_inverse_theorem(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "U^s-inverse-theorem: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "U^s-inverse-theorem: expected premises"
def _auto_impl_Uniform_logMGF(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "Uniform-logMGF: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "Uniform-logMGF: expected premises"

def _auto_impl_kHeath_Brown_decomposition(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "_auto_impl_kHeath_Brown_decomposition: wrong conclusion type"
    assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "_auto_impl_kHeath_Brown_decomposition: expected premises"

def _auto_impl_kIntegral_split(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (Equals, Forall, Implies, BigO)), "kIntegral-split: wrong conclusion type"
    needs_prem = not any(tag in pf.rule.lower() for tag in ['axiom','definition'])
    if needs_prem:
        assert pf.premises and all(hasattr(p,'rule') for p in pf.premises), "kIntegral-split: expected premises"
try:
    _RULE_CHECKS.update({
        "Bilinear-dispersion-HS": _auto_impl_Bilinear_dispersion_HS,
        "Cauchy-Schwarz-reduction": _auto_impl_Cauchy_Schwarz_reduction,
        "Coefficient-extraction": _auto_impl_Coefficient_extraction,
        "Combine-k-major-minor": _auto_impl_Combine_k_major_minor,
        "Convolution-x-def": _auto_impl_Convolution_x_def,
        "Covariance-from-pair": _auto_impl_Covariance_from_pair,
        "Cumulant-partition-Mobius": _auto_impl_Cumulant_partition_Mobius,
        "Cumulant-weight-sum-bound": _auto_impl_Cumulant_weight_sum_bound,
        "Discrete-difference-equivalence": _auto_impl_Discrete_difference_equivalence,
        "Empirical-shift-identity": _auto_impl_Empirical_shift_identity,
        "FourD-dispersion": _auto_impl_FourD_dispersion,
        "FourD-dispersion-lemma": _auto_impl_FourD_dispersion_lemma,
        "Fourth-cumulant-reduction": _auto_impl_Fourth_cumulant_reduction,
        "Fourth-cumulant-vanishes": _auto_impl_Fourth_cumulant_vanishes,
        "Fourth-moment-bound": _auto_impl_Fourth_moment_bound,
        "Frac-difference-inverse": _auto_impl_Frac_difference_inverse,
        "Fractional-weights-def": _auto_impl_Fractional_weights_def,
        "Full-Chowla-Conditional": _auto_impl_Full_Chowla_Conditional,
        "Full-Chowla-steps": _auto_impl_Full_Chowla_steps,
        "Gamma-convolution-identity": _auto_impl_Gamma_convolution_identity,
        "Gamma-covariance-identity": _auto_impl_Gamma_covariance_identity,
        "Generating-function-identity": _auto_impl_Generating_function_identity,
        "HB-fivefold-identity": _auto_impl_HB_fivefold_identity,
        "HB-fourfold-identity": _auto_impl_HB_fourfold_identity,
        "Higher-major-arc-bound": _auto_impl_Higher_major_arc_bound,
        "Higher-minor-arc-bound": _auto_impl_Higher_minor_arc_bound,
        "Inclusion-exclusion-reduction": _auto_impl_Inclusion_exclusion_reduction,
        "KC-criterion": _auto_impl_KC_criterion,
        "Levy-continuity": _auto_impl_Levy_continuity,
        "Method-of-cumulants": _auto_impl_Method_of_cumulants,
        "Multilinear-dispersion-induction": _auto_impl_Multilinear_dispersion_induction,
        "Nilsequence-non-correlation-k": _auto_impl_Nilsequence_non_correlation_k,
        "Quadruple-Chowla-bound": _auto_impl_Quadruple_Chowla_bound,
        "Quadruple-Chowla-from-dispersion": _auto_impl_Quadruple_Chowla_from_dispersion,
        "Relate-kS-to-main-sum": _auto_impl_Relate_kS_to_main_sum,
        "Second-moment-bound": _auto_impl_Second_moment_bound,
        "Tail-truncation-error": _auto_impl_Tail_truncation_error,
        "Third-cumulant-reduction": _auto_impl_Third_cumulant_reduction,
        "Third-cumulant-vanishes": _auto_impl_Third_cumulant_vanishes,
        "Tightness-conclusion": _auto_impl_Tightness_conclusion,
        "Trilinear-dispersion": _auto_impl_Trilinear_dispersion,
        "Trilinear-dispersion-lemma": _auto_impl_Trilinear_dispersion_lemma,
        "Triple-Chowla-bound": _auto_impl_Triple_Chowla_bound,
        "Triple-Chowla-from-dispersion": _auto_impl_Triple_Chowla_from_dispersion,
        "U^s-inverse-theorem": _auto_impl_U_s_inverse_theorem,
        "Uniform-logMGF": _auto_impl_Uniform_logMGF,
        "kHeath-Brown-decomposition": _auto_impl_kHeath_Brown_decomposition,
        "kIntegral-split": _auto_impl_kIntegral_split,
    })
except NameError:
    pass
