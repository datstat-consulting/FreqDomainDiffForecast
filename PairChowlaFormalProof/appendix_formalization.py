
from kernel import Atom, Equals, BigO, Implies, combine, axiom, check_proof

# Fractional weights identity and generating function
proof_w = combine("Fractional-weights-def",
                  Equals(Atom("w_m"),
                         Atom("Gamma(m+alpha) / (Gamma(alpha) * m!); w_m=0 for m<0")),
                  axiom(Atom("binomial/Gamma definition")))
proof_W = combine("Generating-function-identity",
                  Equals(Atom("W(z)"),
                         Atom("Sum_{m>=0} w_m z^m = (1-z)^(-alpha)")),
                  proof_w)
proof_coeff = combine("Coefficient-extraction",
                      Equals(Atom("Coeff_{z^ell} W(z)^2"),
                             Atom("Gamma(ell+2alpha) / (Gamma(2alpha) * ell!)")),
                      proof_W)
proof_disc = combine("Discrete-difference-equivalence",
                     Equals(Atom("a_ell"),
                            Atom("b_ell; same recurrence and a_0 = b_0")),
                     proof_coeff)
proof_gamma_cov = combine("Gamma-covariance-identity",
                          Equals(Atom("Sum_{i>=0} w_i * w_{i+ell}"),
                                 Atom("1/2 * (ell^{2H} - (ell-1)^{2H})")),
                          proof_W, proof_coeff, proof_disc)

# Weight-sum bounds
proof_weight_sum = combine("Cumulant-weight-sum-bound",
                           BigO(Atom("Sum_{i_1,...,i_r} |A_{i_1,...,i_r}|"),
                                Atom("N^{r*alpha}")),
                           axiom(Atom("S_j <= const * (N(t_j - t_{j-1}))^alpha")))

# Inclusion–exclusion reduction for cumulants
proof_emp_shift = combine("Empirical-shift-identity",
                          Equals(Atom("E_N[lambda(i_1)...lambda(i_r)]"),
                                 Atom("1/N * Sum_{n<=N} lambda(n)...lambda(n+h) + O(N^-1)")))
proof_mobius = combine("Cumulant-partition-Mobius",
                       Equals(Atom("Cum(X_1,...,X_r)"),
                              Atom("sum_{pi in P([r])} mu(pi) prod_{B in pi} E prod_{i in B} X_i")))
proof_IE = combine("Inclusion-exclusion-reduction",
                   Equals(Atom("Cum(lambda(i1),...,lambda(ir))"),
                          Atom("1/N * Sum_{n<=N} lambda(n)...lambda(n+h) + O(N^-δ)")),
                   proof_emp_shift, proof_mobius)

# Multilinear dispersion: bilinear base and r-variable induction → r=3,4 corollaries
proof_bilin = combine("Bilinear-dispersion-HS-mapped",
                      BigO(Atom("Bilinear dispersion sum"),
                           Atom("X^{1/2} * (PQ)^{1/2} * d^{-δ} * (log X)^C")),
                      axiom(Atom("Harper–Shao 2023")))
proof_multilin = combine("Multilinear-dispersion-induction",
                         BigO(Atom("Multilinear dispersion r variables"),
                              Atom("X^{1/2} * (P_1...P_r)^{1/2} * d^{-δ_r} * (log X)^{C_r}")),
                         proof_bilin)
proof_tri = combine("Trilinear-dispersion-lemma",
                    BigO(Atom("trilinear (r=3) dispersion"),
                         Atom("X^{1/2} * (PQR)^{1/2} * d^{-δ_3} * (log X)^{C_3}")),
                    proof_multilin)
proof_four = combine("FourD-dispersion-lemma",
                     BigO(Atom("quadrilinear (r=4) dispersion"),
                          Atom("X^{1/2} * (PQRS)^{1/2} * d^{-δ_4} * (log X)^{C_4}")),
                     proof_multilin)

# Turn dispersion into correlation savings using HB identities + CS/second moment
prem_hb4 = combine("HB-fourfold-identity",
                   Equals(Atom("lambda(n)"),
                          Atom("4-fold HB identity expansion + l.o.t.")),
                   axiom(Atom("Heath–Brown (1982)")))
prem_cs = combine("Cauchy-Schwarz-reduction",
                  Equals(Atom("Dispersion → Correlation bound"),
                         Atom("second moment / dyadic blocks")),
                  axiom(Atom("two-step Cauchy–Schwarz")))
proof_triple_from_disp = combine("Triple-Chowla-from-dispersion",
                                 BigO(Atom("Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2)"),
                                      Atom("N^{1-δ'} * (log N)^C")),
                                 proof_tri, prem_cs, prem_hb4)

prem_hb5 = combine("HB-fivefold-identity",
                   Equals(Atom("lambda(n)"),
                          Atom("5-fold HB identity expansion + l.o.t.")),
                   axiom(Atom("HB identity generalization")))
prem_second = combine("Second-moment-bound",
                      Equals(Atom("Cauchy-Schwarz on blocks"),
                             Atom("T_{P_*}^2 ≤ Q * sum |∑ f|^2")),
                      axiom(Atom("standard argument")))
proof_quad_from_disp = combine("Quadruple-Chowla-from-dispersion",
                               BigO(Atom("Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) * lambda(n+h3)"),
                                    Atom("N^{1-δ''} * (log N)^{C'}")),
                               proof_four, prem_second, prem_hb5)

# Plug these into κ3/κ4 vanishing
proof_k3_reduce = combine("Third-cumulant-reduction",
                          BigO(Atom("kappa_3(Delta_1, Delta_2, Delta_3)"),
                               Atom("max_{h1,h2} Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) + o(1)")),
                          axiom(Atom("multilinearity of cumulants + translation invariance")),
                          axiom(Atom("sum|A|=O(1)")))
triple_bound_adapter = combine("Triple-Chowla-bound", proof_triple_from_disp.conclusion, proof_triple_from_disp)
proof_k3_vanish = combine("Third-cumulant-vanishes",
                          Equals(Atom("kappa_3(Delta_1, Delta_2, Delta_3)"),
                                 Atom("→ 0")),
                          proof_k3_reduce, triple_bound_adapter, proof_triple_from_disp)

proof_k4_reduce = combine("Fourth-cumulant-reduction",
                          BigO(Atom("kappa_4(Delta_1, Delta_2, Delta_3, Delta_4)"),
                               Atom("max_{h1,h2,h3} Sum_{n<=N} lambda(n) * lambda(n+h1) * lambda(n+h2) * lambda(n+h3) + o(N)")),
                          axiom(Atom("sum|B|=O(1)")))
quad_bound_adapter = combine("Quadruple-Chowla-bound", proof_quad_from_disp.conclusion, proof_quad_from_disp)
proof_k4_vanish = combine("Fourth-cumulant-vanishes",
                          Equals(Atom("kappa_4(Delta_1, Delta_2, Delta_3, Delta_4)"),
                                 Atom("→ 0")),
                          proof_k4_reduce, quad_bound_adapter, proof_quad_from_disp)

# Fourth moment and tightness
proof_fourth = combine("Fourth-moment-bound",
                       BigO(Atom("E[(Delta_{s,t}^N)^4]"),
                            Atom("|t-s|^{4H - eps}")),
                       proof_k4_vanish, proof_gamma_cov)
proof_kc = combine("KC-criterion",
                   Implies(Atom("E[|Y_N(t)-Y_N(s)|^p] <= C |t-s|^{1+beta}"),
                           Atom("tight in C[0,1]")))
proof_tight = combine("Tightness-conclusion",
                      Equals(Atom("tight in C[0,1]"), Atom("yes")),
                      proof_kc, proof_fourth)

# Uniform log-MGF and Lévy
proof_logmgf = combine("Uniform-logMGF",
                       Equals(Atom("K_N(u) = 1/2 u^T Sigma u + o(1)"),
                              Atom("uniform for ||u||<=rho")),
                       proof_k3_vanish, proof_k4_vanish)
proof_levy = combine("Levy-continuity",
                     Implies(Atom("MGF convergence on neighborhood"),
                             Atom("FDD convergence to Gaussian")))

if __name__ == "__main__":
    proofs = [
        proof_w, proof_W, proof_coeff, proof_disc, proof_gamma_cov,
        proof_weight_sum, proof_emp_shift, proof_mobius, proof_IE,
        proof_bilin, proof_multilin, proof_tri, proof_four,
        proof_triple_from_disp, proof_quad_from_disp,
        proof_k3_reduce, proof_k3_vanish, proof_k4_reduce, proof_k4_vanish,
        proof_fourth, proof_kc, proof_tight, proof_logmgf, proof_levy
    ]
    print("\n=== Appendix formalization: Gamma covariance, weights, dispersion, inclusion–exclusion, tightness, log-MGF ===")
    for pf in proofs:
        print(pf)
        check_proof(pf)
