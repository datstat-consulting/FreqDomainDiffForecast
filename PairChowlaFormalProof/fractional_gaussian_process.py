
from kernel import Atom, Equals, Forall, Implies, StochasticProcess, Mean, Covariance, SelfSimilarity, StationaryIncrements, Power, Minus, Multiply, Fraction, Plus, axiom, combine, check_proof

# --- 1. Gaussian Process Definition ---

# A Gaussian process is a stochastic process where every finite collection of random variables has a multivariate Gaussian distribution.
# For simplicity, we'll formalize the definition by stating that its mean and covariance function define it.

t_sym = Atom("t")
s_sym = Atom("s")
x_process = StochasticProcess("X", t_sym)

proof_gaussian_process_definition = axiom(
    Forall(
        "X",
        Implies(
            Atom("X is a Gaussian Process"),
            Equals(
                x_process,
                Atom("defined by E[X(t)] and Cov(X(t), X(s))")
            )
        )
    )
)

# --- 2. Fractional Brownian Motion (fBm) Definition and Properties ---

H_sym = Atom("H")
fbm_process = StochasticProcess("B_H", t_sym)
fbm_process_s = StochasticProcess("B_H", s_sym)

# Mean of fBm is 0
proof_fbm_mean = axiom(
    Equals(
        Mean(fbm_process),
        Atom("0")
    )
)

# Covariance function of fBm
proof_fbm_covariance = axiom(
    Equals(
        Covariance(fbm_process, fbm_process_s),
        Multiply(
            Fraction(Atom("1"), Atom("2")),
            Plus(
                Power(Atom("|t|"), Multiply(Atom("2"), H_sym)),
                Minus(
                    Power(Atom("|s|"), Multiply(Atom("2"), H_sym)),
                    Power(Atom("|t-s|"), Multiply(Atom("2"), H_sym))
                )
            )
        )
    )
)

# Self-similarity of fBm
a_sym = Atom("a")
proof_fbm_self_similarity = axiom(
    SelfSimilarity(fbm_process, a_sym, H_sym)
)

# Stationary increments of fBm
t1_sym = Atom("t1")
t2_sym = Atom("t2")
t3_sym = Atom("t3")
t4_sym = Atom("t4")
proof_fbm_stationary_increments = axiom(
    StationaryIncrements(fbm_process, t1_sym.name, t2_sym.name, t3_sym.name, t4_sym.name)
)

# --- 3. Fractional Gaussian Noise (fGn) ---

# fGn is the \'derivative\' of fBm (conceptually, not formally a derivative in this context yet)
# We\'ll represent it as a stochastic process for now.
fgn_process = StochasticProcess("G_H", t_sym)

proof_fgn_definition = axiom(
    Equals(
        fgn_process,
        Atom("Conceptual derivative of B_H(t)")
    )
)

# === Self-test ===
if __name__ == "__main__":
    print("\n=== Fractional Gaussian Process Formalization ===")
    print(proof_gaussian_process_definition)
    check_proof(proof_gaussian_process_definition)
    print(proof_fbm_mean)
    check_proof(proof_fbm_mean)
    print(proof_fbm_covariance)
    check_proof(proof_fbm_covariance)
    print(proof_fbm_self_similarity)
    check_proof(proof_fbm_self_similarity)
    print(proof_fbm_stationary_increments)
    check_proof(proof_fbm_stationary_increments)
    print(proof_fgn_definition)
    check_proof(proof_fgn_definition)


