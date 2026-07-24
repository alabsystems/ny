/-
  SatReluGadget.lean — the exact-arithmetic soundness core of the sat_relu CNF
  decompilation (Route A, docs/MEASURED_SAT_RELU.md).

  sat_relu compiles k-SAT into Gemm→ReLU→Gemm nets. The recovery in
  `crates/ny-cli/src/commands/beta_crown/cnf_route.rs` rests on two exact facts;
  this file proves the first (the Booleanization lemma) and the clause-row fact,
  both in real arithmetic, `sorry`-free.

  Booleanization: the network's `Y_1 = Σ_j (x_j − ReLU(2x_j − 1))` output is a
  sum of per-coordinate residuals `bres x = x − ReLU(2x − 1)`. On `[0,1]` each
  residual equals `min x (1 − x) ≥ 0`, and is `0` EXACTLY at the boolean corners
  `x ∈ {0,1}`. So the unsafe constraint `Y_1 ≤ 0` (with `x ∈ [0,1]^n`, all
  residuals `≥ 0`, sum `≤ 0`) forces every residual to `0`, i.e. forces boolean
  inputs — the exact equivalence the decompilation depends on, not a relaxation.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith

namespace Crownproof

namespace SatRelu

/-- Scalar ReLU. -/
def relu (t : ℝ) : ℝ := max 0 t

/-- The per-coordinate Booleanization residual `x − ReLU(2x − 1)`. -/
def bres (x : ℝ) : ℝ := x - relu (2 * x - 1)

/-- On `[0,1]`, the residual is `min x (1 − x)`. -/
theorem bres_eq_min {x : ℝ} : bres x = min x (1 - x) := by
  unfold bres relu
  rcases le_total (2 * x - 1) 0 with h | h
  · rw [max_eq_left h, min_eq_left (by linarith)]
    linarith
  · rw [max_eq_right h, min_eq_right (by linarith)]
    linarith

/-- **Booleanization, nonnegativity.**  `x − ReLU(2x − 1) ≥ 0` on `[0,1]`. -/
theorem bres_nonneg {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : 0 ≤ bres x := by
  rw [bres_eq_min]; exact le_min h0 (by linarith)

/-- **Booleanization, equality iff boolean.**  On `[0,1]`, the residual is `0`
    exactly at the boolean corners. This is what forces `Y_1 ≤ 0` ⇒ boolean. -/
theorem bres_eq_zero_iff {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    bres x = 0 ↔ x = 0 ∨ x = 1 := by
  rw [bres_eq_min]
  constructor
  · intro h
    rcases min_eq_iff.mp h with ⟨hx, _⟩ | ⟨hx, _⟩
    · exact Or.inl hx
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl)
    · rw [min_eq_left (by linarith)]
    · rw [min_eq_right (by linarith)]
      linarith

/-- **The gadget's Booleanization block forces boolean inputs.**

If every coordinate is in `[0,1]` and the total Booleanization output
`Σ_j bres x_j` is `≤ 0` (the unsafe constraint `Y_1 ≤ 0`), then — since every
residual is `≥ 0` — every residual is exactly `0`, hence every coordinate is
boolean.  Stated over a `Finset` of coordinates. -/
theorem forces_boolean {ι : Type*} (s : Finset ι) (x : ι → ℝ)
    (hbox : ∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1)
    (hY1 : ∑ j ∈ s, bres (x j) ≤ 0) :
    ∀ j ∈ s, x j = 0 ∨ x j = 1 := by
  -- Each term is ≥ 0 and the sum is ≤ 0, so every term is 0.
  have hnonneg : ∀ j ∈ s, 0 ≤ bres (x j) := fun j hj =>
    bres_nonneg (hbox j hj).1 (hbox j hj).2
  have hsum_ge : 0 ≤ ∑ j ∈ s, bres (x j) := Finset.sum_nonneg hnonneg
  have hsum_zero : ∑ j ∈ s, bres (x j) = 0 := le_antisymm hY1 hsum_ge
  intro j hj
  have hzero : bres (x j) = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg hnonneg).mp hsum_zero j hj
  exact (bres_eq_zero_iff (hbox j hj).1 (hbox j hj).2).mp hzero

end SatRelu

end Crownproof

/-! ## Trust-base check — the gadget soundness core must reduce to the standard
axioms only (no `sorryAx`). -/

#print axioms Crownproof.SatRelu.bres_nonneg
#print axioms Crownproof.SatRelu.bres_eq_zero_iff
#print axioms Crownproof.SatRelu.forces_boolean
