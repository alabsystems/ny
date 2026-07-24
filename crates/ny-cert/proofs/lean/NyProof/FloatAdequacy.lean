/-
  FloatAdequacy.lean — `R_float ⊑ R_real` for the residual float (non-cert) relaxation paths.

  CONTEXT.  NY's emitted certificates are EXACT: f32 weights lift losslessly to ℚ (n/2^k)
  and the entire CROWN/relaxation math runs in exact ℚ, so for emitted certs R_float =
  R_real *by construction* (SPEC.md "Float adequacy"; `proofs/tcb.json` row
  `float_adequacy`, status closed-by-construction).  This file is therefore NOT needed for
  emitted-cert soundness.  It de-TCBs the RESIDUAL float paths — the fast f64 cut-DISCOVERY
  path (`crown_deep.rs:551`, demoted to discovery-only) and the non-cert ny-propagate
  verifier — which compute relaxations in floating point with DIRECTED OUTWARD ROUNDING.

  MODEL.  The float-rounding hardware behavior is the named TCB, abstracted as two ℚ→ℚ
  directed-rounding operators bounded by the identity (`dn q ≤ q ≤ up q`).  Every per-op
  adequacy lemma is then PROVEN from those bounds by linear monotonicity (`linarith`).  The
  resulting trust base is exactly the two rounding bounds plus Lean's standard
  `[propext, Classical.choice, Quot.sound]` — reported by `#print axioms`; no `sorry`.

  STATUS.  VALIDATED 2026-06-28 — `lake build` clean against Mathlib v4.30.0 (toolchain
  leanprover/lean4:v4.30.0); `#print axioms` on the per-op lemmas reports exactly
  `[propext, Classical.choice, Quot.sound]` (no `sorry`, no extra axioms — the float TCB is
  the explicit `Round` hypotheses `dn_le`/`le_up`, not an axiom).  These theorems are
  intentionally NOT in `cite-map.json`: the soundness-critical cert path is exact and does
  not depend on them; they ground only the non-cert / discovery TCB rows.
-/
import Mathlib

namespace FloatAdequacy

/-- A directed-rounding model: `dn` rounds toward −∞, `up` toward +∞.  The two bounds are
    the named float TCB (the hardware IEEE-754 directed-rounding behavior). -/
structure Round where
  dn : ℚ → ℚ
  up : ℚ → ℚ
  dn_le : ∀ q, dn q ≤ q
  le_up : ∀ q, q ≤ up q

variable (R : Round)

/-- The core adequacy fact: outward rounding of an interval contains it — every point of
    the exact interval `[lo, hi]` lies in the rounded interval `[dn lo, up hi]`. -/
theorem interval_outward_contains (lo hi x : ℚ) (hx : lo ≤ x ∧ x ≤ hi) :
    R.dn lo ≤ x ∧ x ≤ R.up hi := by
  obtain ⟨h1, h2⟩ := hx
  have hd := R.dn_le lo
  have hu := R.le_up hi
  exact ⟨by linarith, by linarith⟩

/-- Affine lower bound, outward-rounded, is sound: if the exact value clears the exact
    lower bound `lo`, it clears the rounded bound `dn lo`. -/
theorem affine_lower_adequate (lo v : ℚ) (h : lo ≤ v) : R.dn lo ≤ v := by
  have := R.dn_le lo; linarith

/-- Affine upper bound, outward-rounded, is sound. -/
theorem affine_upper_adequate (hi v : ℚ) (h : v ≤ hi) : v ≤ R.up hi := by
  have := R.le_up hi; linarith

/-- Box adequacy: an exact box `[l, u]` constraint survives outward rounding. -/
theorem box_adequate (l u x : ℚ) (h : l ≤ x ∧ x ≤ u) :
    R.dn l ≤ x ∧ x ≤ R.up u :=
  interval_outward_contains R l u x h

/-- ReLU upper-chord adequacy: the chord `y ≤ λ·x + μ` with `λ, μ` outward-rounded (both
    `up`, for `x ≥ 0`) still dominates the exact chord. -/
theorem relu_chord_upper_adequate (lam mu x y : ℚ)
    (hxy : y ≤ lam * x + mu) (hx : 0 ≤ x) :
    y ≤ R.up lam * x + R.up mu := by
  have hl := R.le_up lam
  have hm := R.le_up mu
  have hmul : lam * x ≤ R.up lam * x := mul_le_mul_of_nonneg_right hl hx
  linarith

/-- The adequacy binding `R_float ⊑ R_real`, at the certificate level: if the
    outward-rounded upper bound certifies safety (`up exact_hi ≤ 0`), the EXACT upper bound
    is safe (`exact_hi ≤ 0`).  So a float-verified verdict implies the real verdict — the
    float relaxation is a sound over-approximation of the real one. -/
theorem float_bound_implies_real (exact_hi : ℚ) (hfloat : R.up exact_hi ≤ 0) :
    exact_hi ≤ 0 := by
  have := R.le_up exact_hi; linarith

end FloatAdequacy

-- Trust base check: only the rounding model (passed as hypotheses) + Lean's standard
-- axioms; must report `[propext, Classical.choice, Quot.sound]` (no `sorry`, no extra axioms).
#print axioms FloatAdequacy.float_bound_implies_real
#print axioms FloatAdequacy.interval_outward_contains
#print axioms FloatAdequacy.relu_chord_upper_adequate
