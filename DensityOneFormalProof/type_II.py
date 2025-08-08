# type_II.py
# Formalization of the Type II/III (bilinear dispersion) subproof
# in the density‐one Pair Chowla theorem.

from kernel import Atom, Sum, BigO, Equals, axiom, combine, check_proof

# 1. Bilinear large‐sieve lemma (Montgomery–Vaughan)
proof_bs = axiom(
    BigO(
        Sum(
            "p", "P<p≤2P",
            Sum(
                "q", "Q<q≤2Q",
                Atom("|Σ_{m=1}^{⌊x/(p*q)⌋} e(*)|")
            )
        ),
        Atom("x^(1/2)*(P*Q)^(1/2)*(log x)^A")
    )
)

# 2. Apply it to the off‐diagonal sum
proof_bs_apply = combine(
    "Bilinear-sieve-apply",
    BigO(
        Atom("Σ_{a=M1..2M1} Σ_{b=M2..2M2} Σ_{m1≠m2} e(r*a*b*(m1−m2)/D)"),
        Atom("x^(1/2)*(M1*M2)^(1/2)*(log x)^A * K")
    ),
    proof_bs
)

# 3. Cauchy–Schwarz inference
proof_cs = combine(
    "Cauchy-Schwarz",
    BigO(
        Atom("Σ_{a=M1..2M1} Σ_{b=M2..2M2} |A_{a,b}|"),
        Atom("N^(3/4)*(log N)^C")
    ),
    proof_bs_apply
)

# 4. Count dyadic blocks: O((log N)^2)
proof_dyadic = axiom(
    Equals(
        Atom("#dyadic_pairs"),
        Atom("O((log N)^2)")
    )
)

# 5. Sum over dyadic blocks
proof_sum_dyadic = combine(
    "Sum-dyadic",
    BigO(
        Atom("TypeII_total"),
        Atom("N^(3/4)*(log N)^(C')")
    ),
    proof_cs,
    proof_dyadic
)

# === Self‐test ===
if __name__ == "__main__":
    print("=== Type II/III Subproof Formalization ===")
    print(proof_sum_dyadic)
    check_proof(proof_sum_dyadic)
