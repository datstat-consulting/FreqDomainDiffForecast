
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
    assert isinstance(pf.conclusion.left, LFunction)
    assert isinstance(pf.conclusion.right, Product)

def _check_l_function_ratio(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, LFunction)
    assert isinstance(pf.conclusion.right, Fraction)
    assert isinstance(pf.conclusion.right.numerator, LFunction)
    assert isinstance(pf.conclusion.right.denominator, LFunction)

def _check_zero_density_bound(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    assert isinstance(pf.conclusion.expr, NZeros)

def _check_perron_formula(pf: Proof):
    assert isinstance(pf.conclusion, Equals)
    assert isinstance(pf.conclusion.left, Atom) # S(chi, beta)
    assert isinstance(pf.conclusion.right, Plus)
    assert isinstance(pf.conclusion.right.left, Multiply)
    assert isinstance(pf.conclusion.right.left.left, Fraction)
    assert isinstance(pf.conclusion.right.left.right, Integral)

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


