/-
Copyright 2026 Andrew Yates
SPDX-License-Identifier: Apache-2.0

Exact-real centered normalization ranges.

The direction bound is derived from finite-sum Cauchy--Schwarz. The sharper
coordinate bound is derived by applying Cauchy--Schwarz to every other
coordinate and using the zero-sum identity; it is not a range hypothesis.

The first frozen version reached the pinned official Lean 4.30/Mathlib
overlay but exposed two proof elaboration errors. This successor repairs
the theorem reference and addition ordering without changing any statement
or assumption. Its local queries still require a fresh kernel validation.
Existing NyProof and AxiomAudit registrations are unchanged.

All values are real numbers. These statements do not establish coefficient
bit import, directed floating arithmetic, the actual source graph/epsilon,
certificate publication, a compiled Rust correspondence, or a work/memory/
deadline theorem. Those remain separate implementation proof obligations.
-/
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace NyProof.CenteredNormalization

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The squared Euclidean norm written as an explicit finite sum. -/
noncomputable def squaredNorm (x : ι → ℝ) : ℝ := ∑ i, x i ^ 2

/-- The Euclidean norm of the same finite coordinate vector. -/
noncomputable def l2Norm (x : ι → ℝ) : ℝ := Real.sqrt (squaredNorm x)

/-- The original real coordinate dot product. -/
noncomputable def dot (a z : ι → ℝ) : ℝ := ∑ i, a i * z i

/-- The real arithmetic mean; positivity is required by centering theorems. -/
noncomputable def mean (x : ι → ℝ) : ℝ := (∑ i, x i) / Fintype.card ι

/-- Subtract the same mean from every coordinate. -/
noncomputable def center (x : ι → ℝ) (i : ι) : ℝ := x i - mean x

theorem squaredNorm_nonneg (x : ι → ℝ) : 0 ≤ squaredNorm x := by
  exact Finset.sum_nonneg (fun i _ => sq_nonneg (x i))

/-- Cauchy--Schwarz supplies the squared direction inequality. -/
theorem dot_sq_le (a z : ι → ℝ) :
    dot a z ^ 2 ≤ squaredNorm a * squaredNorm z := by
  exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ a z

/-- Positive epsilon gives a strictly positive normalization denominator. -/
theorem denominator_pos {epsilon energy dimension : ℝ}
    (he : 0 < epsilon) (hE : 0 ≤ energy) (hd : 0 < dimension) :
    0 < Real.sqrt (epsilon + energy / dimension) := by
  exact Real.sqrt_pos.2 (add_pos_of_pos_of_nonneg he (div_nonneg hE hd.le))

/-- An algebraic adapter from a *derived squared inequality*, not an assumed
normalization range. The direction and coordinate theorems derive its premise
from the actual finite sums below. -/
theorem divided_abs_bound_of_sq {x energy dimension rangeSq epsilon : ℝ}
    (he : 0 < epsilon) (hE : 0 ≤ energy) (hd : 0 < dimension)
    (hR : 0 ≤ rangeSq) (hsq : dimension * x ^ 2 ≤ rangeSq * energy) :
    |x| / Real.sqrt (epsilon + energy / dimension) ≤ Real.sqrt rangeSq := by
  have harg : 0 < epsilon + energy / dimension :=
    add_pos_of_pos_of_nonneg he (div_nonneg hE hd.le)
  have hden : 0 < Real.sqrt (epsilon + energy / dimension) := Real.sqrt_pos.2 harg
  have hx : x ^ 2 ≤ rangeSq * (energy / dimension) := by
    calc
      x ^ 2 ≤ rangeSq * energy / dimension :=
        (le_div_iff₀ hd).2 (by nlinarith [hsq])
      _ = rangeSq * (energy / dimension) := by ring
  have hproduct :
      (Real.sqrt rangeSq * Real.sqrt (epsilon + energy / dimension)) ^ 2 =
        rangeSq * (epsilon + energy / dimension) := by
    rw [mul_pow, Real.sq_sqrt hR, Real.sq_sqrt harg.le]
  have hxsq : x ^ 2 ≤
      (Real.sqrt rangeSq * Real.sqrt (epsilon + energy / dimension)) ^ 2 := by
    rw [hproduct]
    nlinarith [mul_nonneg hR he.le]
  have habs : |x| ≤ Real.sqrt rangeSq * Real.sqrt (epsilon + energy / dimension) :=
    abs_le_of_sq_le_sq hxsq (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
  exact (div_le_iff₀ hden).2 habs

/-- Exact-real direction range. It actually holds without centering; when z
is centered it is the radial LayerNorm bound used by the streamed head.
No norm or desired normalization inequality is supplied as a premise. -/
theorem normalized_direction_bound (a z : ι → ℝ) {epsilon dimension : ℝ}
    (he : 0 < epsilon) (hd : 0 < dimension) :
    |dot a z| / Real.sqrt (epsilon + squaredNorm z / dimension) ≤
      Real.sqrt dimension * l2Norm a := by
  have ha := squaredNorm_nonneg a
  have hz := squaredNorm_nonneg z
  have hscaled : dimension * dot a z ^ 2 ≤
      (dimension * squaredNorm a) * squaredNorm z := by
    calc
      dimension * dot a z ^ 2 ≤ dimension * (squaredNorm a * squaredNorm z) :=
        mul_le_mul_of_nonneg_left (dot_sq_le a z) hd.le
      _ = (dimension * squaredNorm a) * squaredNorm z := by ring
  have h := divided_abs_bound_of_sq he hz hd (mul_nonneg hd.le ha) hscaled
  simpa only [Real.sqrt_mul hd.le, l2Norm] using h

/-- The finite coordinate specialization uses the true dimension, not a
caller-supplied shape pretending to be the dimension. -/
theorem normalized_direction_bound_card (a z : ι → ℝ) {epsilon : ℝ}
    (he : 0 < epsilon) (hd : 0 < Fintype.card ι) :
    |dot a z| / Real.sqrt (epsilon + squaredNorm z / Fintype.card ι) ≤
      Real.sqrt (Fintype.card ι) * l2Norm a := by
  exact normalized_direction_bound a z he (by exact_mod_cast hd)

theorem center_sum_zero (x : ι → ℝ) (hd : 0 < Fintype.card ι) :
    (∑ i, center x i) = 0 := by
  have hreal : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  simp only [center, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul, mean]
  field_simp [hreal]
  <;> ring

/-- Projecting the direction or projecting the source vector are the same
exact dot product. This is an identity, independent of numerical rounding. -/
theorem projected_dot_eq (a x : ι → ℝ) : dot (center a) x = dot a (center x) := by
  unfold dot center
  simp only [sub_mul, mul_sub, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum, ← Finset.sum_mul]
  unfold mean
  ring

theorem dot_center_right_of_sum_zero (a x : ι → ℝ) (ha : (∑ i, a i) = 0) :
    dot a (center x) = dot a x := by
  unfold dot center
  simp only [mul_sub, Finset.sum_sub_distrib]
  rw [← Finset.sum_mul, ha, zero_mul, sub_zero]

/-- A projected direction evaluated on the original input has the centered
source numerator. Its norm can therefore be paid/bounded after projection. -/
theorem projected_normalized_direction_bound (a x : ι → ℝ) {epsilon : ℝ}
    (he : 0 < epsilon) (hd : 0 < Fintype.card ι) :
    |dot (center a) x| / Real.sqrt (epsilon + squaredNorm (center x) / Fintype.card ι) ≤
      Real.sqrt (Fintype.card ι) * l2Norm (center a) := by
  rw [← dot_center_right_of_sum_zero (center a) x (center_sum_zero a hd)]
  exact normalized_direction_bound_card (center a) (center x) he hd

/-- Derived sharper coordinate inequality. The other coordinates have sum
-z_k. Applying finite-sum Cauchy--Schwarz to those coordinates gives
z_k²≤(h-1)Σ_{j≠k}z_j², hence h z_k²≤(h-1)Σ_j z_j². -/
theorem centered_coordinate_sq_bound (z : ι → ℝ) (k : ι)
    (hz : (∑ i, z i) = 0) :
    (Fintype.card ι : ℝ) * z k ^ 2 ≤
      ((Fintype.card ι : ℝ) - 1) * squaredNorm z := by
  have hd : 0 < Fintype.card ι := Fintype.card_pos_iff.mpr ⟨k⟩
  have hsplit : z k + (∑ i ∈ Finset.univ.erase k, z i) = ∑ i, z i :=
    Finset.add_sum_erase Finset.univ z (Finset.mem_univ k)
  have hsplitSq : z k ^ 2 + (∑ i ∈ Finset.univ.erase k, z i ^ 2) = squaredNorm z :=
    Finset.add_sum_erase Finset.univ (fun i => z i ^ 2) (Finset.mem_univ k)
  have hrest : (∑ i ∈ Finset.univ.erase k, z i) = -z k := by
    linarith [hsplit, hz]
  have hrestSq : (∑ i ∈ Finset.univ.erase k, z i ^ 2) = squaredNorm z - z k ^ 2 := by
    linarith [hsplitSq]
  have hcard : ((Finset.univ.erase k).card : ℝ) = (Fintype.card ι : ℝ) - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ k), Finset.card_univ,
      Nat.cast_sub (Nat.one_le_of_lt hd), Nat.cast_one]
  have hother := _root_.sq_sum_le_card_mul_sum_sq (s := Finset.univ.erase k) (f := z)
  rw [hrest, hrestSq, hcard] at hother
  nlinarith [hother]

/-- Every coordinate of a zero-sum vector has normalized magnitude at most
sqrt(h-1), including h=1 where the bound is exactly zero. -/
theorem normalized_coordinate_bound (z : ι → ℝ) (k : ι) {epsilon : ℝ}
    (he : 0 < epsilon) (hz : (∑ i, z i) = 0) :
    |z k| / Real.sqrt (epsilon + squaredNorm z / Fintype.card ι) ≤
      Real.sqrt ((Fintype.card ι : ℝ) - 1) := by
  have hd : 0 < Fintype.card ι := Fintype.card_pos_iff.mpr ⟨k⟩
  have hdreal : 0 < (Fintype.card ι : ℝ) := by exact_mod_cast hd
  have hrange : 0 ≤ (Fintype.card ι : ℝ) - 1 := by
    have hcard : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Nat.one_le_of_lt hd
    linarith
  exact divided_abs_bound_of_sq he (squaredNorm_nonneg z) hdreal hrange
    (centered_coordinate_sq_bound z k hz)

/-- Mean centering supplies the zero-sum premise of the coordinate theorem. -/
theorem normalized_centered_coordinate_bound (x : ι → ℝ) (k : ι) {epsilon : ℝ}
    (he : 0 < epsilon) :
    |center x k| / Real.sqrt (epsilon + squaredNorm (center x) / Fintype.card ι) ≤
      Real.sqrt ((Fintype.card ι : ℝ) - 1) := by
  have hd : 0 < Fintype.card ι := Fintype.card_pos_iff.mpr ⟨k⟩
  exact normalized_coordinate_bound (center x) k he (center_sum_zero x hd)

/-- Explicit one-coordinate control: centered normalization is zero. -/
theorem center_fin_one_zero (x : Fin 1 → ℝ) : center x = fun _ => 0 := by
  funext i
  have hi : i = (0 : Fin 1) := Subsingleton.elim _ _
  subst i
  simp [center, mean]

/-- A scalar beta/bias term is retained separately from the direction range. -/
theorem normalized_direction_with_constant (a z : ι → ℝ)
    {epsilon dimension constant : ℝ} (he : 0 < epsilon) (hd : 0 < dimension) :
    |dot a z / Real.sqrt (epsilon + squaredNorm z / dimension) + constant| ≤
      Real.sqrt dimension * l2Norm a + |constant| := by
  have hden := denominator_pos he (squaredNorm_nonneg z) hd
  calc
    _ ≤ |dot a z / Real.sqrt (epsilon + squaredNorm z / dimension)| + |constant| :=
      abs_add_le _ _
    _ = |dot a z| / Real.sqrt (epsilon + squaredNorm z / dimension) + |constant| := by
      rw [abs_div, abs_of_pos hden]
    _ ≤ Real.sqrt dimension * l2Norm a + |constant| :=
      add_le_add (normalized_direction_bound a z he hd) (le_refl |constant|)

#print axioms squaredNorm_nonneg
#print axioms dot_sq_le
#print axioms denominator_pos
#print axioms divided_abs_bound_of_sq
#print axioms normalized_direction_bound
#print axioms normalized_direction_bound_card
#print axioms center_sum_zero
#print axioms projected_dot_eq
#print axioms dot_center_right_of_sum_zero
#print axioms projected_normalized_direction_bound
#print axioms centered_coordinate_sq_bound
#print axioms normalized_coordinate_bound
#print axioms normalized_centered_coordinate_bound
#print axioms center_fin_one_zero
#print axioms normalized_direction_with_constant

end NyProof.CenteredNormalization
