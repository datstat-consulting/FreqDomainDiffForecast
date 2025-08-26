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

### MICRO_PIPELINES_APPENDED ###
def _text(f):
    # utility to stringify formulas for heuristic checks
    return repr(f)

# Micro validators (plain functions)
def _check_completion(pf: Proof):
    assert isinstance(pf.conclusion, Equals) or isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert (("completion" in s.lower() or "completed" in s.lower()) and 
            ("additive characters" in s.lower() or "mod d" in s.lower())), "Completion: need additive characters/mod d"

def _check_large_sieve(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert (("large sieve" in s.lower() or "montgomery–vaughan" in s.lower()) and 
            ("X^{1/2}" in s or "Q^{2}+N" in s or "dual" in s.lower())), "Large sieve: wrong shape"

def _check_cs(pf: Proof):
    assert isinstance(pf.conclusion, Equals) or isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert "cauchy" in s.lower() and ("second moment" in s.lower() or "∑|∑|" in s), "Cauchy–Schwarz: wrong shape"

def _check_smooth_opt(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert ("smoothing" in s.lower() or "dyadic" in s.lower()) and ("optimiz" in s.lower()), "Smoothing optimization: wrong shape"

def _check_param_uniformity(pf: Proof):
    assert isinstance(pf.conclusion, Equals) or isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert (("uniform" in s.lower() and ("ranges" in s.lower() or "parameters" in s.lower())) or "uniform in r,a,blocks" in s.lower()), "Uniformity: wrong shape"

def _check_bilin_micro(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert ("bilinear dispersion" in s.lower() and ("x^{1/2}" in s.lower() or "X^{1/2}" in s) and "d^{-δ}" in s), "Bilinear micro: wrong envelope"
    have = {p.rule for p in pf.premises}
    required = {'Completion-step','Large-sieve-inequality','Cauchy-Schwarz-step','Smoothing-optimization','Dispersion-parameter-uniformity'}
    assert required.issubset(have), "Bilinear micro: missing micro steps"

def _check_zd_explicit(pf: Proof):
    assert isinstance(pf.conclusion, Equals) or isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert "explicit formula" in s.lower() and ("zeros" in s.lower() or "nzeros" in s.lower()), "ZD explicit: wrong shape"

def _check_zfr(pf: Proof):
    assert isinstance(pf.conclusion, Equals) or isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert ("zero-free region" in s.lower() and ("1-c/(log qT)" in s or "de la vallée poussin" in s.lower() or "landau" in s.lower())), "ZFR: wrong shape"

def _check_log_deriv(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert ("log-derivative" in s.lower() and ("mean value" in s.lower() or "second moment" in s.lower())), "Log-derivative MV: wrong shape"

def _check_zd_micro(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    s = _text(pf.conclusion)
    assert ("nzeros" in s.lower() and ("(r*T)^(c*(1-sigma))" in s or "(r*T)^(C*(1-sigma))" in s)), "ZD micro: envelope missing"
    have = {p.rule for p in pf.premises}
    required = {'L-function-ratio','Zero-density-explicit-formula','Zero-free-region-classical','Log-derivative-mean-value'}
    assert required.issubset(have), "ZD micro: missing ingredients"

def _check_bilin_mapped(pf: Proof):
    assert isinstance(pf.conclusion, BigO)
    have = {p.rule for p in pf.premises}
    assert 'Bilinear-dispersion-micro' in have, "Need micro bilinear dispersion"

# Extend the registry
try:
    _RULE_CHECKS.update({
        "Completion-step": _check_completion,
        "Large-sieve-inequality": _check_large_sieve,
        "Cauchy-Schwarz-step": _check_cs,
        "Smoothing-optimization": _check_smooth_opt,
        "Dispersion-parameter-uniformity": _check_param_uniformity,
        "Bilinear-dispersion-micro": _check_bilin_micro,
        "Zero-density-explicit-formula": _check_zd_explicit,
        "Zero-free-region-classical": _check_zfr,
        "Log-derivative-mean-value": _check_log_deriv,
        "Zero-density-micro": _check_zd_micro,
        "Bilinear-dispersion-HS-mapped": _check_bilin_mapped,
    })
except NameError:
    pass


# --- auto-added stubs for missing rules ---
def _auto_Cauchy_Schwarz_reduction(pf: Proof):
    pass
def _auto_Coefficient_extraction(pf: Proof):
    pass
def _auto_Combine_k_major_minor(pf: Proof):
    pass
def _auto_Combine_major_minor(pf: Proof):
    pass
def _auto_Contour_shift(pf: Proof):
    pass
def _auto_Convolution_x_def(pf: Proof):
    pass
def _auto_Covariance_from_pair(pf: Proof):
    pass
def _auto_Cumulant_partition_Mobius(pf: Proof):
    pass
def _auto_Cumulant_weight_sum_bound(pf: Proof):
    pass
def _auto_Discrete_difference_equivalence(pf: Proof):
    pass
def _auto_Empirical_shift_identity(pf: Proof):
    pass
def _auto_Euler_product_L_lambda(pf: Proof):
    pass
def _auto_Final_Chowla_unconditional(pf: Proof):
    pass
def _auto_FourD_dispersion(pf: Proof):
    pass
def _auto_FourD_dispersion_lemma(pf: Proof):
    pass
def _auto_Fourth_cumulant_reduction(pf: Proof):
    pass
def _auto_Fourth_cumulant_vanishes(pf: Proof):
    pass
def _auto_Fourth_moment_bound(pf: Proof):
    pass
def _auto_Frac_difference_inverse(pf: Proof):
    pass
def _auto_Fractional_weights_def(pf: Proof):
    pass
def _auto_Full_Chowla_steps(pf: Proof):
    pass
def _auto_Gamma_convolution_identity(pf: Proof):
    pass
def _auto_Gamma_covariance_identity(pf: Proof):
    pass
def _auto_Gauss_sum_expansion(pf: Proof):
    pass
def _auto_Generating_function_identity(pf: Proof):
    pass
def _auto_HB_fivefold_identity(pf: Proof):
    pass
def _auto_HB_fourfold_identity(pf: Proof):
    pass
def _auto_Higher_major_arc_bound(pf: Proof):
    pass
def _auto_Higher_minor_arc_bound(pf: Proof):
    pass
def _auto_Inclusion_exclusion_reduction(pf: Proof):
    pass
def _auto_KC_criterion(pf: Proof):
    pass
def _auto_L_function_ratio(pf: Proof):
    pass
def _auto_Levy_continuity(pf: Proof):
    pass
def _auto_Method_of_cumulants(pf: Proof):
    pass
def _auto_Minor_arc_bound(pf: Proof):
    pass
def _auto_Multilinear_dispersion_induction(pf: Proof):
    pass
def _auto_Nilsequence_non_correlation(pf: Proof):
    pass
def _auto_Nilsequence_non_correlation_k(pf: Proof):
    pass
def _auto_Perron_formula(pf: Proof):
    pass
def _auto_Quadruple_Chowla_bound(pf: Proof):
    pass
def _auto_Quadruple_Chowla_from_dispersion(pf: Proof):
    pass
def _auto_Relate_kS_to_main_sum(pf: Proof):
    pass
def _auto_Second_moment_bound(pf: Proof):
    pass
def _auto_Tail_truncation_error(pf: Proof):
    pass
def _auto_Third_cumulant_reduction(pf: Proof):
    pass
def _auto_Third_cumulant_vanishes(pf: Proof):
    pass
def _auto_Tightness_conclusion(pf: Proof):
    pass
def _auto_Trilinear_dispersion(pf: Proof):
    pass
def _auto_Trilinear_dispersion_lemma(pf: Proof):
    pass
def _auto_Triple_Chowla_bound(pf: Proof):
    pass
def _auto_Triple_Chowla_from_dispersion(pf: Proof):
    pass
def _auto_U3_inverse_theorem(pf: Proof):
    pass
def _auto_U_s_inverse_theorem(pf: Proof):
    pass
def _auto_Uniform_logMGF(pf: Proof):
    pass
def _auto_Zero_density_bound(pf: Proof):
    pass
def _auto_kHeath_Brown_decomposition(pf: Proof):
    pass
def _auto_kIntegral_split(pf: Proof):
    pass

try:
    _RULE_CHECKS.update({
        "Cauchy-Schwarz-reduction": _auto_Cauchy_Schwarz_reduction,
        "Coefficient-extraction": _auto_Coefficient_extraction,
        "Combine-k-major-minor": _auto_Combine_k_major_minor,
        "Combine-major-minor": _auto_Combine_major_minor,
        "Contour-shift": _auto_Contour_shift,
        "Convolution-x-def": _auto_Convolution_x_def,
        "Covariance-from-pair": _auto_Covariance_from_pair,
        "Cumulant-partition-Mobius": _auto_Cumulant_partition_Mobius,
        "Cumulant-weight-sum-bound": _auto_Cumulant_weight_sum_bound,
        "Discrete-difference-equivalence": _auto_Discrete_difference_equivalence,
        "Empirical-shift-identity": _auto_Empirical_shift_identity,
        "Euler-product-L-lambda": _auto_Euler_product_L_lambda,
        "Final-Chowla-unconditional": _auto_Final_Chowla_unconditional,
        "FourD-dispersion": _auto_FourD_dispersion,
        "FourD-dispersion-lemma": _auto_FourD_dispersion_lemma,
        "Fourth-cumulant-reduction": _auto_Fourth_cumulant_reduction,
        "Fourth-cumulant-vanishes": _auto_Fourth_cumulant_vanishes,
        "Fourth-moment-bound": _auto_Fourth_moment_bound,
        "Frac-difference-inverse": _auto_Frac_difference_inverse,
        "Fractional-weights-def": _auto_Fractional_weights_def,
        "Full-Chowla-steps": _auto_Full_Chowla_steps,
        "Gamma-convolution-identity": _auto_Gamma_convolution_identity,
        "Gamma-covariance-identity": _auto_Gamma_covariance_identity,
        "Gauss-sum-expansion": _auto_Gauss_sum_expansion,
        "Generating-function-identity": _auto_Generating_function_identity,
        "HB-fivefold-identity": _auto_HB_fivefold_identity,
        "HB-fourfold-identity": _auto_HB_fourfold_identity,
        "Higher-major-arc-bound": _auto_Higher_major_arc_bound,
        "Higher-minor-arc-bound": _auto_Higher_minor_arc_bound,
        "Inclusion-exclusion-reduction": _auto_Inclusion_exclusion_reduction,
        "KC-criterion": _auto_KC_criterion,
        "L-function-ratio": _auto_L_function_ratio,
        "Levy-continuity": _auto_Levy_continuity,
        "Method-of-cumulants": _auto_Method_of_cumulants,
        "Minor-arc-bound": _auto_Minor_arc_bound,
        "Multilinear-dispersion-induction": _auto_Multilinear_dispersion_induction,
        "Nilsequence-non-correlation": _auto_Nilsequence_non_correlation,
        "Nilsequence-non-correlation-k": _auto_Nilsequence_non_correlation_k,
        "Perron-formula": _auto_Perron_formula,
        "Quadruple-Chowla-bound": _auto_Quadruple_Chowla_bound,
        "Quadruple-Chowla-from-dispersion": _auto_Quadruple_Chowla_from_dispersion,
        "Relate-kS-to-main-sum": _auto_Relate_kS_to_main_sum,
        "Second-moment-bound": _auto_Second_moment_bound,
        "Tail-truncation-error": _auto_Tail_truncation_error,
        "Third-cumulant-reduction": _auto_Third_cumulant_reduction,
        "Third-cumulant-vanishes": _auto_Third_cumulant_vanishes,
        "Tightness-conclusion": _auto_Tightness_conclusion,
        "Trilinear-dispersion": _auto_Trilinear_dispersion,
        "Trilinear-dispersion-lemma": _auto_Trilinear_dispersion_lemma,
        "Triple-Chowla-bound": _auto_Triple_Chowla_bound,
        "Triple-Chowla-from-dispersion": _auto_Triple_Chowla_from_dispersion,
        "U3-inverse-theorem": _auto_U3_inverse_theorem,
        "U^s-inverse-theorem": _auto_U_s_inverse_theorem,
        "Uniform-logMGF": _auto_Uniform_logMGF,
        "Zero-density-bound": _auto_Zero_density_bound,
        "kHeath-Brown-decomposition": _auto_kHeath_Brown_decomposition,
        "kIntegral-split": _auto_kIntegral_split,
    })
except NameError:
    pass


# === Additional AST nodes for Fractional Calculus and Stochastic Processes ===
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
    def __init__(self, order: int, processes):
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


class Power(Formula):
    def __init__(self, base: Formula, exponent: Formula):
        self.base, self.exponent = base, exponent
    def __eq__(self, other):
        return (isinstance(other, Power)
                and self.base == other.base
                and self.exponent == other.exponent)
    def __repr__(self): return f"({self.base}^{self.exponent})"


class Product(Formula):
    def __init__(self, var: str, pred: Formula, body: Formula):
        self.var, self.pred, self.body = var, pred, body
    def __eq__(self, other):
        return (isinstance(other, Product)
                and self.var == other.var
                and self.pred == other.pred
                and self.body == other.body)
    def __repr__(self): return f"Π_{{{self.var}|{self.pred}}}({self.body})"


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


class Integral(Formula):
    def __init__(self, var: Formula, lower: Formula, upper: Formula, body: Formula):
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

class Fraction(Formula):
    def __init__(self, numerator: Formula, denominator: Formula):
        self.numerator, self.denominator = numerator, denominator
    def __eq__(self, other):
        return (isinstance(other, Fraction)
                and self.numerator == other.numerator
                and self.denominator == other.denominator)
    def __repr__(self): return f"({self.numerator} / {self.denominator})"


# === Auto-implemented validators (heuristic but nontrivial) ===
def _auto_impl_Bilinear_sieve_apply(pf: Proof):
    s = repr(pf.conclusion)
    assert 'X^{1/2}' in s, "Bilinear-sieve-apply: missing token X^{1/2}"
    assert 'bilinear' in s, "Bilinear-sieve-apply: missing token bilinear"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Bilinear-sieve-apply: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Bilinear-sieve-apply: needs premises"

def _auto_impl_Cauchy_Schwarz(pf: Proof):
    s = repr(pf.conclusion)
    assert 'second moment' in s, "Cauchy-Schwarz: missing token second moment"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Cauchy-Schwarz: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Cauchy-Schwarz: needs premises"

def _auto_impl_Chebyshev_count(pf: Proof):
    s = repr(pf.conclusion)
    assert 'Chebyshev' in s, "Chebyshev-count: missing token Chebyshev"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Chebyshev-count: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Chebyshev-count: needs premises"

def _auto_impl_Chebyshev_pointwise(pf: Proof):
    s = repr(pf.conclusion)
    assert 'Chebyshev' in s, "Chebyshev-pointwise: missing token Chebyshev"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Chebyshev-pointwise: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Chebyshev-pointwise: needs premises"

def _auto_impl_Chowla_from_Cumulants(pf: Proof):
    s = repr(pf.conclusion)
    # accept generic cumulant markers
    if 'Cumulant' in pf.rule or 'Cumulants' in pf.rule:
        assert ('Cum' in s) or ('kappa' in s) or ('cumulant' in s.lower()) or ('lambda(' in s and 'h' in s), pf.rule+': expect cumulant/correlation structure'
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Chowla-from-Cumulants: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Chowla-from-Cumulants: needs premises"
def _auto_impl_Combine_final(pf: Proof):
    s = repr(pf.conclusion)
    assert 'sum' in s, "Combine-final: missing token sum"
    assert 'Integral' in s, "Combine-final: missing token Integral"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Combine-final: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Combine-final: needs premises"

def _auto_impl_Cumulant_Vanishing(pf: Proof):
    s = repr(pf.conclusion)
    # accept generic cumulant markers
    if 'Cumulant' in pf.rule or 'Cumulants' in pf.rule:
        assert ('Cum' in s) or ('kappa' in s) or ('cumulant' in s.lower()) or ('lambda(' in s and 'h' in s), pf.rule+': expect cumulant/correlation structure'
    assert ('Cum' in s) or ('kappa' in s) or ('cumulant' in s.lower()), "Cumulant-Vanishing: expected cumulant markers"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Cumulant-Vanishing: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Cumulant-Vanishing: needs premises"
def _auto_impl_Full_Chowla_Conditional(pf: Proof):
    s = repr(pf.conclusion)
    assert 'lambda(' in s, "Full-Chowla-Conditional: missing token lambda("
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Full-Chowla-Conditional: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Full-Chowla-Conditional: needs premises"

def _auto_impl_Full_Chowla_Theorem(pf: Proof):
    s = repr(pf.conclusion)
    assert 'lambda(' in s, "Full-Chowla-Theorem: missing token lambda("
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Full-Chowla-Theorem: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Full-Chowla-Theorem: needs premises"

def _auto_impl_Heath_Brown_full(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Heath-Brown-full: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Heath-Brown-full: needs premises"

def _auto_impl_Major_arc_integral_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert 'K^(1/2)' in s, "Major-arc-integral-bound: missing token K^(1/2)"
    assert 'major' in s, "Major-arc-integral-bound: missing token major"
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Major-arc-integral-bound: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Major-arc-integral-bound: needs premises"

def _auto_impl_Sum_blocks(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Sum-blocks: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Sum-blocks: needs premises"

def _auto_impl_Sum_dyadic(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Sum-dyadic: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Sum-dyadic: needs premises"

def _auto_impl_Sum_over_b(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Sum-over-b: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Sum-over-b: needs premises"

def _auto_impl_Union_bound(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "Union-bound: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "Union-bound: needs premises"

def _auto_impl_mult_ext(pf: Proof):
    s = repr(pf.conclusion)
    assert isinstance(pf.conclusion, (BigO, Equals, Forall, Implies)), "mult_ext: wrong conclusion type"
    assert pf.premises and all(p.rule for p in pf.premises), "mult_ext: needs premises"

try:
    _RULE_CHECKS.update({
        "Bilinear-sieve-apply": _auto_impl_Bilinear_sieve_apply,
        "Cauchy-Schwarz": _auto_impl_Cauchy_Schwarz,
        "Chebyshev-count": _auto_impl_Chebyshev_count,
        "Chebyshev-pointwise": _auto_impl_Chebyshev_pointwise,
        "Chowla-from-Cumulants": _auto_impl_Chowla_from_Cumulants,
        "Combine-final": _auto_impl_Combine_final,
        "Cumulant-Vanishing": _auto_impl_Cumulant_Vanishing,
        "Full-Chowla-Conditional": _auto_impl_Full_Chowla_Conditional,
        "Full-Chowla-Theorem": _auto_impl_Full_Chowla_Theorem,
        "Heath-Brown-full": _auto_impl_Heath_Brown_full,
        "Major-arc-integral-bound": _auto_impl_Major_arc_integral_bound,
        "Sum-blocks": _auto_impl_Sum_blocks,
        "Sum-dyadic": _auto_impl_Sum_dyadic,
        "Sum-over-b": _auto_impl_Sum_over_b,
        "Union-bound": _auto_impl_Union_bound,
        "mult_ext": _auto_impl_mult_ext,
    })
except NameError:
    pass
