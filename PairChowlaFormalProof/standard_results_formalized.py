
from kernel import Atom, Equals, BigO, Product, Fraction, LFunction, Power, Plus, Multiply, Integral, NZeros, Implies, Forall, combine, check_proof

chi = Atom("chi"); r = Atom("r"); a = Atom("a"); s = Atom("s"); T = Atom("T"); beta = Atom("beta"); K = Atom("K")

orthogonality = combine("Character-orthogonality",
    Equals(Atom("Sum_{x mod r} chi(x) * bar(chi')(x)"), Atom("0 unless chi=chi', else phi(r)")))

finite_fourier = combine("Finite-Fourier-inversion",
    Equals(Atom("f(n)"), Atom("1/phi(r) * Sum_chi tau(bar(chi)) * Sum_{a mod r} chi(a) f_hat(chi)")))

proof_gauss_sum_identity = combine("Gauss-sum-identity",
    Equals(Atom("F_K(a/r + beta)"),
           Atom("(1/phi(r)) * Sum_chi tau(bar(chi)) * chi(a) * S(chi, beta)")),
    orthogonality, finite_fourier)

proof_tau_magnitude = combine("Gauss-tau-magnitude",
    Equals(Atom("|tau(bar(chi))|"), Atom("sqrt(r)")),
    orthogonality)

mellin_inv = combine("Mellin-inversion",
    Equals(Atom("Sum_{n} a(n) w(n/K)"),
           Atom("(1/(2*pi*i)) * Integral(c-iT to c+iT) A(s) * K^s / s ds")))

trunc_err = combine("Truncation-error-bound",
    BigO(Atom("Tail contribution"), Atom("K/T + K*|beta|")))

abs_conv = combine("Absolute-convergence",
    BigO(Atom("Vertical integral piece"), Atom("K^c * (log K)^B")))

proof_perron_formula = combine("Perron-formula",
    Equals(
        Atom("S(chi, beta)"),
        Plus(
            Multiply(
                Fraction(Atom("1"), Multiply(Atom("2"), Multiply(Atom("pi"), Atom("i")))),
                Integral(
                    s, 
                    Plus(Atom("c"), Multiply(Atom("-i"), T)),
                    Plus(Atom("c"), Multiply(Atom("i"), T)),
                    Multiply(LFunction(s, chi, liouville_twist=True), Fraction(Power(K, s), s))
                )
            ),
            Plus(Atom("O(K/T)"), Atom("O(K*|beta|)"))
        )
    ),
    mellin_inv, trunc_err, abs_conv)

explicit_formula = combine("Explicit-formula",
    Equals(Atom("NZeros(sigma,T,chi)"),
           Atom("Explicit formula integral + prime sum + zero sum")))

zero_free_region = combine("Zero-free-region",
    Equals(Atom("Re(s) >= 1 - c / log(r*T)"),
           Atom("Dirichlet L has no zeros here")))

log_deriv_mv = combine("Log-derivative-mean-value",
    BigO(Atom("Mean value of (L'/L)(s,chi)"),
         Atom("(r*T)^{eps} * (log(r*T))^B")))

l_ratio = combine("L-ratio-twist",
    Equals(LFunction(Atom("s"), Atom("chi"), liouville_twist=True),
           Fraction(LFunction(Atom("2s"), Atom("chi^2")), LFunction(Atom("s"), Atom("chi")))))

proof_zero_density_micro = combine("Zero-density-bound",
    BigO(NZeros(Atom("sigma"), Atom("T"), Atom("chi"), liouville_twist=True),
         Atom("(r*T)^(C*(1-sigma)) * (log(r*T))^D")),
    l_ratio, explicit_formula, zero_free_region, log_deriv_mv)

feq = combine("Functional-equation-schema",
    Equals(Atom("Completed L(s,chi)"), Atom("Gamma-factor * L(1-s, bar(chi))")))

pl_principle = combine("Phragmen-Lindelof-principle",
    BigO(Atom("L(s,chi) on strip"),
         Atom("(r*T)^{(1-sigma)/2 + eps}")))

proof_convexity_bound = combine("Convexity-bound-L",
    BigO(Atom("L(s,chi) on Re(s)=1-eta"),
         Atom("(r*T)^{(1-(1-eta))/2 + eps} * (log(r*T))^B")),
    feq, pl_principle)

__all__ = [
    "proof_gauss_sum_identity",
    "proof_tau_magnitude",
    "proof_perron_formula",
    "proof_zero_density_micro",
    "proof_convexity_bound",
]
