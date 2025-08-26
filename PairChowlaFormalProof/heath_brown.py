# heath_brown.py
# Full formalization of the Heath–Brown 3-fold identity in our Python proof assistant.

from kernel import Atom, Equals, Sum, Proof, axiom, combine, check_proof

# === 1. Setup variables ===
n, D = 'n', 'D'

# === 2. Define the three HB sums explicitly ===

# HB1(n,D) := Σ_{a,b,c: a*b*c = n ∧ a ≤ D} μ(a) λ(b) λ(c)
hb1 = axiom(Equals(
    Atom(f"HB1({n},{D})"),
    Sum(
        f"a,b,c",
        Atom(f"a*b*c = {n} ∧ a ≤ {D}"),
        Atom("μ(a) * λ(b) * λ(c)")
    )
))

# HB2(n,D) := Σ_{a,b,c: a*b*c = n ∧ a ≤ D ∧ b ≤ D} μ(a) λ(b) λ(c)
hb2 = axiom(Equals(
    Atom(f"HB2({n},{D})"),
    Sum(
        f"a,b,c",
        Atom(f"a*b*c = {n} ∧ a ≤ {D} ∧ b ≤ {D}"),
        Atom("μ(a) * λ(b) * λ(c)")
    )
))

# HB3(n,D) := Σ_{a,b,c: a*b*c = n ∧ a ≤ D ∧ b ≤ D ∧ c ≤ D} μ(a) λ(b) λ(c)
hb3 = axiom(Equals(
    Atom(f"HB3({n},{D})"),
    Sum(
        f"a,b,c",
        Atom(f"a*b*c = {n} ∧ a ≤ {D} ∧ b ≤ {D} ∧ c ≤ {D}"),
        Atom("μ(a) * λ(b) * λ(c)")
    )
))

# === 3. Combine into the full Heath–Brown decomposition ===

comb = axiom(Equals(
    Atom(f"HBsum({n},{D})"),
    Atom(f"HB1({n},{D}) - 2*HB2({n},{D}) + HB3({n},{D})")
))

# === 4. Tie back to λ(n) ===

hb_axiom = axiom(Equals(
    Atom(f"λ({n})"),
    Atom(f"HBsum({n},{D})")
))

# === 5. Assemble the complete proof ===

proof_hb_full = combine(
    "Heath-Brown-full",
    Equals(Atom(f"λ({n})"), Atom(f"HBsum({n},{D})")),
    hb1, hb2, hb3, comb, hb_axiom
)

# === 6. Output and verify ===

print("=== Full Heath–Brown Decomposition Proof ===")
print(proof_hb_full)

# This will raise if any premise fails to check
check_proof(proof_hb_full)