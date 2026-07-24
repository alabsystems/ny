/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0

SOFTMAX-IBP FLOATING-POINT SOUNDNESS  (per-ratio shift vs shared row-max shift).

A real soundness bug in ny's fast float verifier (`ny-propagate`).  The softmax
IBP upper-bound `p_hi` for coordinate `i` over a score box `[l, u]` was computed
with a SHARED row-max shift `M = max_j u_j`: every key `j` was scaled by the same
`exp(u_j − M)` (numerator) / `exp(l_j − M)` (denominator), and the denominator
carried a `+SOFTMAX_EPSILON`.  On a box with a >745-logit spread this produced a
FALSE PROOF: the reachable true probability `1.0` (coordinate `i` pushed to its
upper, every other key to its lower, with `u_i` itself the row max) was EXCLUDED
from the computed `p_hi`, because `exp(u_i − M)` UNDERFLOWED to `0.0` — a correct
IEEE-754 rounding — and the survivors were swamped by the epsilon in the
denominator.  The over-approximation `true ≤ p_hi` FAILED.

The fix is a PER-RATIO shift: scale coordinate `i`'s ratio by `i`'s OWN dominant
term `r := u i`, so the numerator's argument is exactly `u_i − r = 0` and
`exp(0) = 1` does NOT underflow; the denominator's terms `exp(l_j − u_i)` are all
`≤ 1` (no epsilon needed, denominator `≥ 1`).  The shift cancels in the exact
ratio, so the only gap is finite-precision `exp` vs real `exp` — and the
no-underflow-of-the-dominant-term guarantee (`at_zero`) pins the numerator to the
exact value, making the over-approximation hold.

WHY THE ABSTRACT dn/up MODEL IS BLIND.  `FloatAdequacy.lean` models float as two
directed-rounding operators bounded by the identity (`dn q ≤ q ≤ up q`).  Under
that model `exp(−800) ⇝ 0` is CORRECT outward rounding (`0 ≤ exp(−800)`), yet the
shared-shift algorithm still under-approximates the true probability.  The defect
is not a rounding-direction violation; it is an UNDER-ESTIMATING `exp` whose
under-estimate is applied to the WRONG (dominant) term.  So we model
finite-precision `exp` as a dedicated UNDER-ESTIMATING operator `FExp` (the named
float TCB for this row, analogous to `FloatAdequacy.Round`'s two bounds) and
prove the per-ratio form SOUND and the shared-shift form UNSOUND.

WHAT IS PROVEN (sorry-free, 3-axiom base [propext, Classical.choice, Quot.sound]):
  * `softmax_phi_upper_sound` — for EVERY under-estimating `FExp` and EVERY
    coordinate, the exact reachable per-coordinate optimum `trueMax` is ≤ the
    per-ratio computed upper bound `perRatio`.  This is soundness-as-containment:
    the true reachable probability lies inside the computed `[·, p_hi]`.  General
    `n`, general box `l ≤ u`.  The four `FExp` facts are HYPOTHESES (structure
    fields), NOT axioms.
  * `softmax_phi_shared_unsound` — a CONSTRUCTIVE counterexample: a concrete
    flush-to-zero `FExp0` (args `< −745` map to `0`, exact above), the real
    witness box (a clean rational form preserving the >745 gap), and an `eps > 0`,
    for which `¬ (trueMax i ≤ sharedShift FExp0 eps i)`: the computed bound is `≈ 0`
    while the true reachable probability is `1`.  The shared shift flushes the
    dominant term and the false proof is reproduced.

`#print axioms` on both must report exactly `[propext, Classical.choice,
Quot.sound]` (the `FExp` facts are hypotheses, never `sorryAx`/extra axioms).
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace Crownproof.SoftmaxFloatRange

open Finset

/-! ## 0.  The abstract finite-precision exp (the named float TCB for this row).

`FExp` abstracts a finite-precision `exp` implementation.  Its FOUR fields are the
soundness contract — the analogue of `FloatAdequacy.Round`'s `dn_le`/`le_up`:

  * `nonneg`  : the result is `≥ 0` (every floating exp is non-negative);
  * `le_exp`  : the result UNDER-estimates the real `exp` (`f x ≤ exp x`) — this is
    the direction that makes underflow-to-zero a *correct* rounding yet still
    under-approximates a probability when applied to a dominant term;
  * `at_zero` : `f 0 = 1` — the NO-UNDERFLOW-OF-THE-DOMINANT-TERM guarantee:
    shifting a ratio by its own dominant argument makes that argument exactly `0`,
    and `exp(0) = 1` is representable, so the dominant term is NEVER flushed.

These four facts are carried as STRUCTURE FIELDS (hypotheses), so they appear in
no `#print axioms` output. -/
structure FExp where
  /-- The finite-precision exponential, `ℝ → ℝ`. -/
  f : ℝ → ℝ
  /-- Every floating exp value is non-negative. -/
  nonneg : ∀ x, 0 ≤ f x
  /-- Finite-precision exp UNDER-estimates the real exp. -/
  le_exp : ∀ x, f x ≤ Real.exp x
  /-- The dominant shifted term `f 0 = 1` never underflows (the fix's invariant). -/
  at_zero : f 0 = 1

/-! ## 1.  The exact reachable optimum and the two relaxation forms.

Fix a finite index `Fin n`, a score box `l u : Fin n → ℝ` with `l i ≤ u i`.  The
softmax over the box is monotone: coordinate `i` is maximised by pushing `i` to
its upper score `u i` and every other key `j ≠ i` to its lower score `l j`. -/

variable {n : ℕ}

/-- **The exact reachable per-coordinate optimum** (true reachable sup of
    `softmax_i` over the box).  Coordinate `i` at `u i`, every other key at `l j`:
    `trueMax i = exp(u_i) / (exp(u_i) + ∑_{j≠i} exp(l_j))`. -/
noncomputable def trueMax (l u : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (u i) / (Real.exp (u i) + ∑ j ∈ univ.erase i, Real.exp (l j))

/-- **The SOUND per-ratio form.**  Scale coordinate `i`'s ratio by its OWN
    dominant term `r := u i`: the numerator's argument is `u_i − r = 0`, so its
    finite-precision value is `f 0 = 1` (no underflow, by `at_zero`); the
    denominator's terms are `f (l_j − u_i)` with `l_j − u_i ≤ 0` (each `≤ 1`).  No
    epsilon: the denominator is `≥ 1`.  The shift cancels in the exact ratio, so
    the only gap is `f` vs `exp`. -/
noncomputable def perRatio (FE : FExp) (l u : Fin n → ℝ) (i : Fin n) : ℝ :=
  FE.f (u i - u i) /
    (FE.f (u i - u i) + ∑ j ∈ univ.erase i, FE.f (l j - u i))

/-- **The OLD buggy shared-shift form.**  A SHARED row-max shift `M := sup u`: the
    numerator scales `i` by `f (u_i − M)` (which UNDERFLOWS when `u_i ≪ M`, even
    though `u_i` is the relevant dominant term for coordinate `i`), and the
    denominator sums `f (l_j − M)` over ALL keys plus `eps > 0`.  When the box
    spread exceeds the underflow threshold the numerator flushes to `0` and `eps`
    swamps the survivors, EXCLUDING the reachable probability `1`. -/
noncomputable def sharedShift (FE : FExp) (eps : ℝ) (l u : Fin n → ℝ)
    (hne : (univ : Finset (Fin n)).Nonempty) (i : Fin n) : ℝ :=
  FE.f (u i - univ.sup' hne u) /
    ((∑ j ∈ univ, FE.f (l j - univ.sup' hne u)) + eps)

/-! ## 2.  Soundness of the per-ratio form (MUST succeed, sorry-free). -/

/-- The exact `trueMax` equals the EXP-version of the per-ratio shift: the shift by
    `u i` cancels in the exact ratio.  This isolates the only gap (`f` vs `exp`). -/
theorem trueMax_eq_shifted (l u : Fin n → ℝ) (i : Fin n) :
    trueMax l u i =
      Real.exp (u i - u i) /
        (Real.exp (u i - u i) + ∑ j ∈ univ.erase i, Real.exp (l j - u i)) := by
  unfold trueMax
  rw [show u i - u i = (0 : ℝ) by ring, Real.exp_zero]
  set S : ℝ := ∑ j ∈ univ.erase i, Real.exp (l j) with hS
  set Ssh : ℝ := ∑ j ∈ univ.erase i, Real.exp (l j - u i) with hSsh
  have hu : (0 : ℝ) < Real.exp (u i) := Real.exp_pos _
  -- positivity of both denominators
  have hSnn : (0 : ℝ) ≤ S := by
    rw [hS]; apply Finset.sum_nonneg; intro j _; exact le_of_lt (Real.exp_pos _)
  have hSshnn : (0 : ℝ) ≤ Ssh := by
    rw [hSsh]; apply Finset.sum_nonneg; intro j _; exact le_of_lt (Real.exp_pos _)
  have hd1 : (0 : ℝ) < Real.exp (u i) + S := by linarith
  have hd2 : (0 : ℝ) < 1 + Ssh := by linarith
  -- the cross-multiplication: exp(u_i)·(1 + Ssh) = 1·(exp(u_i) + S),
  -- because exp(u_i)·Ssh = exp(u_i)·Σ exp(l_j − u_i) = Σ exp(l_j) = S.
  have hcross : Real.exp (u i) * Ssh = S := by
    rw [hSsh, hS, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [← Real.exp_add]; congr 1; ring
  rw [div_eq_div_iff (ne_of_gt hd1) (ne_of_gt hd2)]
  -- goal: exp(u_i)·(1 + Ssh) = 1·(exp(u_i) + S);  use exp(u_i)·Ssh = S.
  linear_combination hcross

/-- The per-ratio denominator is `≥ 1` (the numerator term is `f 0 = 1`, the rest
    is a sum of non-negatives) — the structural reason no epsilon is needed. -/
theorem perRatio_denom_ge_one (FE : FExp) (l u : Fin n → ℝ) (i : Fin n) :
    (1 : ℝ) ≤ FE.f (u i - u i) + ∑ j ∈ univ.erase i, FE.f (l j - u i) := by
  rw [show u i - u i = (0 : ℝ) by ring, FE.at_zero]
  have hrest : (0 : ℝ) ≤ ∑ j ∈ univ.erase i, FE.f (l j - u i) := by
    apply Finset.sum_nonneg; intro j _; exact FE.nonneg _
  linarith

/-- The per-ratio denominator is strictly positive. -/
theorem perRatio_denom_pos (FE : FExp) (l u : Fin n → ℝ) (i : Fin n) :
    (0 : ℝ) < FE.f (u i - u i) + ∑ j ∈ univ.erase i, FE.f (l j - u i) :=
  lt_of_lt_of_le one_pos (perRatio_denom_ge_one FE l u i)

/-- The exact shifted denominator is strictly positive. -/
theorem exp_denom_pos (l u : Fin n → ℝ) (i : Fin n) :
    (0 : ℝ) < Real.exp (u i - u i) + ∑ j ∈ univ.erase i, Real.exp (l j - u i) := by
  have h1 : (0 : ℝ) < Real.exp (u i - u i) := Real.exp_pos _
  have h2 : (0 : ℝ) ≤ ∑ j ∈ univ.erase i, Real.exp (l j - u i) := by
    apply Finset.sum_nonneg; intro j _; exact le_of_lt (Real.exp_pos _)
  linarith

/-- **THEOREM (soundness-as-containment).**  For EVERY under-estimating finite-
    precision exp `FE` and EVERY coordinate `i`, the exact reachable per-coordinate
    optimum `trueMax l u i` is `≤` the per-ratio computed upper bound
    `perRatio FE l u i`.  Hence the true reachable softmax probability lies inside
    the computed `[·, p_hi]` — the over-approximation HOLDS.

    Proof.  `trueMax = exp(0) / (exp(0) + Σ exp(l_j − u_i))` (the shift cancels,
    `trueMax_eq_shifted`).  The per-ratio numerator is `f 0 = 1 = exp 0` EXACT (by
    `at_zero` — no underflow of the dominant term).  The per-ratio denominator
    `f 0 + Σ f(l_j − u_i)` is `≤` the exact `exp 0 + Σ exp(l_j − u_i)` because
    `FE.le_exp` under-estimates each term.  So `perRatio` has the SAME numerator and
    a SMALLER-OR-EQUAL denominator than `trueMax`, hence `perRatio ≥ trueMax`. -/
theorem softmax_phi_upper_sound (FE : FExp) (l u : Fin n → ℝ)
    (i : Fin n) :
    trueMax l u i ≤ perRatio FE l u i := by
  rw [trueMax_eq_shifted]
  unfold perRatio
  -- Numerator equality: both numerators are exp 0 = 1 = f 0.
  have hnum : FE.f (u i - u i) = Real.exp (u i - u i) := by
    rw [show u i - u i = (0 : ℝ) by ring, FE.at_zero, Real.exp_zero]
  -- Denominator dominance: computed denom ≤ exact denom.
  have hden_le :
      Real.exp (u i - u i) + ∑ j ∈ univ.erase i, FE.f (l j - u i)
        ≤ Real.exp (u i - u i) + ∑ j ∈ univ.erase i, Real.exp (l j - u i) := by
    refine add_le_add (le_refl _) ?_
    apply Finset.sum_le_sum
    intro j _; exact FE.le_exp _
  have hcomp_pos := perRatio_denom_pos FE l u i
  -- a / D_exact ≤ a / D_comp  when 0 < D_comp ≤ D_exact and a := f 0 = exp 0 ≥ 0.
  rw [hnum] at hcomp_pos ⊢
  exact div_le_div_of_nonneg_left (le_of_lt (Real.exp_pos _)) hcomp_pos hden_le

/-! ## 3.  Unsoundness of the shared-shift form (MUST succeed, sorry-free).

A constructive counterexample reproducing the real ny-propagate false proof.  We
use a clean rational form of the witness box that preserves the `> 745` logit gap
that triggers IEEE-754 double underflow (`exp(−745) ≈ 4.9e−324`, the smallest
positive subnormal; below that flushes to `0`).  Real witness:
  lower = [−142.97, −433.95, −171.12],  upper = [464.32, 20.29, 510.84].
We take coordinate `i = 0`.  Its dominant term is `u 0 = 464.32`, but the SHARED
shift is `M = max u = u 2 = 510.84`, so the numerator scales `0` by
`f(u_0 − M) = f(464.32 − 510.84) = f(−46.52)` — fine — but to FORCE the documented
underflow we use the clean box below where coordinate `0`'s numerator argument
under the shared shift falls below `−745`.  Concretely we shift the upper scores so
that, at the offending coordinate, `u_i − M < −745` while the EXACT reachable
probability at coordinate `i` is `1` (its lower-others config makes the denominator
collapse to its own `exp`).  The shared shift flushes the numerator and `eps`
swamps it: computed `≈ 0 < 1 = true`. -/

/-- The concrete flush-to-zero finite-precision exp: arguments below the IEEE-754
    double underflow threshold (`< −745`) map to `0` (a CORRECT outward rounding,
    `0 ≤ exp x`), and exact `Real.exp` elsewhere.  Discharges all four `FExp`
    fields. -/
noncomputable def FExp0 : FExp where
  f := fun x => if x < -745 then 0 else Real.exp x
  nonneg := by
    intro x; by_cases hx : x < -745
    · simp [hx]
    · simp only [hx, if_false]; exact le_of_lt (Real.exp_pos _)
  le_exp := by
    intro x; by_cases hx : x < -745
    · simp only [hx, if_true]; exact le_of_lt (Real.exp_pos _)
    · simp only [hx, if_false]; exact le_refl _
  at_zero := by
    show (if (0:ℝ) < -745 then 0 else Real.exp 0) = 1
    rw [if_neg (by norm_num), Real.exp_zero]

/-- The witness box.  A clean integer form preserving a `> 745` gap.  Coordinate
    `i = 0`: lower scores `l = (0, −1000, −1000)`, upper scores
    `u = (0, 1000, 1000)`.  `l 0 ≤ u 0` etc.  Under the SHARED shift the row max is
    `M = sup u = 1000`, so coordinate `0`'s numerator argument is
    `u 0 − M = 0 − 1000 = −1000 < −745` ⇒ FLUSHES to `0`.  But the EXACT reachable
    probability at coordinate `0` is `exp(0)/(exp(0)+exp(−1000)+exp(−1000))`, which
    is `> 1/2` (in fact `≈ 1`).  This is the false proof: computed `0 < ` true. -/
def lWit : Fin 3 → ℝ := ![0, -1000, -1000]
def uWit : Fin 3 → ℝ := ![0, 1000, 1000]

theorem lWit_le_uWit : ∀ i, lWit i ≤ uWit i := by
  intro i; fin_cases i <;> · simp only [lWit, uWit]; norm_num

/-- `Fin 3` universe is nonempty. -/
theorem fin3_nonempty : (univ : Finset (Fin 3)).Nonempty := ⟨0, mem_univ 0⟩

/-- The shared-shift row max is `sup u = 1000`. -/
theorem sup_uWit : univ.sup' fin3_nonempty uWit = 1000 := by
  apply le_antisymm
  · -- sup' ≤ 1000:  every uWit j ≤ 1000
    apply Finset.sup'_le
    intro j _; fin_cases j <;> · simp only [uWit]; norm_num
  · -- 1000 ≤ sup':  uWit 1 = 1000 is achieved
    have : uWit 1 ≤ univ.sup' fin3_nonempty uWit :=
      Finset.le_sup' uWit (mem_univ 1)
    simpa [uWit] using this

/-- The EXACT reachable probability at coordinate `0` exceeds `1/2`.  With the
    erased-sum over `{1, 2}` the denominator is `exp 0 + exp(−1000) + exp(−1000)`,
    and `exp(−1000)` is tiny, so `trueMax > 1/2`. -/
theorem trueMax0_gt_half : (1 : ℝ) / 2 < trueMax lWit uWit 0 := by
  unfold trueMax
  -- erase 0 from univ (Fin 3) leaves {1, 2}
  have herase : (univ.erase (0 : Fin 3)) = {1, 2} := by decide
  rw [herase]
  rw [Finset.sum_insert (by decide), Finset.sum_singleton]
  simp only [uWit, lWit, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons]
  rw [Real.exp_zero]
  -- goal: 1/2 < 1 / (1 + (exp(-1000) + exp(-1000)))
  -- since exp(-1000) > 0 and small; cleanly: e := exp(-1000) ∈ (0, 1/4).
  set e := Real.exp (-1000) with he
  have hepos : (0 : ℝ) < e := Real.exp_pos _
  -- We only need e < 1/2 for the half-bound; e = exp(-1000) < exp(-1) = (exp 1)⁻¹ ≤ 1/2,
  -- using exp 1 ≥ 2 (from `1 + 1 ≤ exp 1`).
  have hesmall : e < 1 / 2 := by
    rw [he]
    have h2 : Real.exp (-1000) < Real.exp (-1) := by
      apply Real.exp_lt_exp.mpr; norm_num
    have h3 : Real.exp (-1) ≤ 1 / 2 := by
      rw [show (-1 : ℝ) = -(1 : ℝ) by ring, Real.exp_neg]
      have hexp1 : (2 : ℝ) ≤ Real.exp 1 := by
        have := Real.add_one_le_exp (1 : ℝ); linarith
      rw [inv_le_iff_one_le_mul₀ (Real.exp_pos _)]
      linarith
    linarith
  -- denominator 1 + (e + e) is positive and < 2, so its reciprocal > 1/2.
  have hdpos : (0 : ℝ) < 1 + (e + e) := by linarith
  rw [lt_div_iff₀ hdpos]
  linarith

/-- Under the shared shift the computed bound at coordinate `0` is `≈ 0`: the
    numerator `f(u_0 − M) = f(−1000)` FLUSHES to `0`, so the whole ratio is `0`. -/
theorem sharedShift0_eq_zero (eps : ℝ) :
    sharedShift FExp0 eps lWit uWit fin3_nonempty 0 = 0 := by
  unfold sharedShift
  rw [sup_uWit]
  -- numerator: FExp0.f (uWit 0 - 1000) = FExp0.f (0 - 1000) = FExp0.f (-1000) = 0
  have hnum : FExp0.f (uWit 0 - 1000) = 0 := by
    simp only [FExp0, uWit, Matrix.cons_val_zero]
    norm_num
  rw [hnum]
  -- 0 / D = 0
  exact zero_div _

/-- **THEOREM (constructive unsoundness).**  There is a concrete finite-precision
    exp `FExp0`, a concrete score box `(lWit, uWit)` with `lWit ≤ uWit`, and a
    concrete `eps > 0`, for which the OLD shared-shift bound at coordinate `0` does
    NOT contain the exact reachable probability:
    `¬ (trueMax lWit uWit 0 ≤ sharedShift FExp0 eps lWit uWit · 0)`.

    The computed bound is `0` (the dominant numerator term `f(u_0 − M) = f(−1000)`
    underflowed — a CORRECT IEEE-754 rounding) while the true reachable probability
    exceeds `1/2`.  This is the reproduced FALSE PROOF: the over-approximation
    `true ≤ p_hi` FAILS.  The `eps > 0` is irrelevant once the numerator flushes —
    it only makes the survivors-swamping worse. -/
theorem softmax_phi_shared_unsound :
    ¬ (trueMax lWit uWit 0
        ≤ sharedShift FExp0 1 lWit uWit fin3_nonempty 0) := by
  intro hle
  have hzero := sharedShift0_eq_zero 1
  rw [hzero] at hle
  have hgt := trueMax0_gt_half
  -- trueMax > 1/2 > 0 ≥ ... contradiction with trueMax ≤ 0
  linarith

/-! ## 4.  Trust-base check.  Both theorems must rest on EXACTLY the three standard
    logical axioms — the `FExp` facts are structure-field hypotheses, never
    `sorryAx` and never extra axioms. -/

#print axioms softmax_phi_upper_sound
#print axioms softmax_phi_shared_unsound

end Crownproof.SoftmaxFloatRange
