/-
  AlphaGradientDefect.lean — why ny's α-CROWN ascent adds exactly 0.000000, proved.

  Context: `crates/ny-propagate/src/network/graph_alpha/backward/gradients.rs`.
  The α-CROWN warmup on cifar100 spends ~22% of the scored budget and, on the
  diagnostic row `CIFAR100_resnet_medium_prop_idx_7704_sidx_3701`, reports
  `best_impr = 0.000e0` at every iteration while `best_lower_sum` never leaves its
  CROWN initializer (−3564.689453).  Measured 2026-08-03: sweeping `lr_alpha` over
  0.25 / 0.05 / 0.01 leaves `best_lower_sum` BIT-IDENTICAL at −3564.689453, which no
  step-size explanation can produce.  α-β-CROWN proves the same row at the ROOT
  (`best_l = +0.00814`, zero BaB, ~10-17 s) with a working ascent.

  The shipped local rule (`gradients.rs:91,115,119`) is

      for each unstable neuron i  (guard `l ≥ 0 ∨ u ≤ 0 → continue`, so `l < 0 < u`)
        g_i := Σ_{j : A[j,i] > 0}  A[j,i] * l_i

  and the Adam step (`propagate_dag/alpha_update.rs:60-69`) is

      m ← β₁m + (1−β₁)(−g);  v ← β₂v + (1−β₂)(−g)²;  a ← clamp(a − lr·m̂/(√v̂+ε), 0, 1)

  This file proves the three facts that together explain the observation:

  * **T1 `local_rule_nonpos`** — the rule is SIGN-DEFINITE: `g_i ≤ 0` for every
    unstable neuron, every objective, every iteration.  It is not a gradient, it is a
    constant-sign field.
  * **T2 `adam_step_nonincreasing`** — consequently the Adam update never INCREASES
    any α.  The ascent is a monotone descent to the `α = 0` clamp, whatever the true
    objective is.  Directional information is absent, not merely noisy — which is
    exactly why the lr sweep moved nothing.
  * **T3 `local_rule_sign_can_be_wrong`** — the true derivative (envelope theorem,
    `∂bound/∂α_i = A[j,i]·ĥ_i(x*)` with `ĥ_i(x*) ∈ [l_i,u_i]` the pre-activation at the
    concretization argmin) is STRICTLY POSITIVE whenever that binding value is
    positive, while the rule is simultaneously strictly negative.  The rule does not
    approximate the derivative there; it has the opposite sign.

  And the reason this is safe to fix, which is the practical point:

  * **T4 `alpha_sound_regardless`** — for EVERY `α ∈ [0,1]` the lower envelope
    `α·z ≤ relu z` holds, via `Crownproof.relu_lower`.  So no choice of α, however
    the optimizer arrives at it, can produce an unsound bound.  Fixing the gradient
    is a BOUND-QUALITY change with ZERO false-`unsat` exposure — the one direction
    that costs −150.  This is the formal content of the source comment
    "gradients are non-soundness-critical" (`gradients.rs:31`).

  Everything is over `ℚ` (a `LinearOrderedField`), matching `Crownproof.Basic`.
  Reals would add no content: every statement is ordered-field reasoning.

  NOT proved here, and deliberately: that the envelope-theorem rule is the correct
  derivative of ny's concretized bound.  That needs the full CROWN concretization
  semantics.  T3 assumes only that the true derivative has the form `A[j,i]·h` for
  some `h` in the neuron's own range — which is what the repo's own finite-difference
  oracle (`backward/true_grad_oracle_tests.rs`) measures, reporting the local rule
  >10× wrong with at least one sign flip, and what `gradients.rs:21-31` states in prose.
-/

import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Linarith
import Mathlib.Algebra.Order.BigOperators.Group.List
import Crownproof.Basic

namespace NyProof.AlphaGradientDefect

open Crownproof

/-! ## T1. The shipped rule is sign-definite. -/

/--
One term of the shipped accumulation, `gradients.rs:119`: `grad_i += a_ji * l`,
reached only under the guard `a_ji > 0` (`:115`) with `l < 0` forced by the
unstable-neuron guard `l ≥ 0 ∨ u ≤ 0 → continue` (`:91`).

Every such term is non-positive.
-/
theorem local_rule_term_nonpos (a l : ℚ) (ha : 0 < a) (hl : l < 0) :
    a * l ≤ 0 :=
  mul_nonpos_of_nonneg_of_nonpos (le_of_lt ha) (le_of_lt hl)

/--
**T1.** The accumulated local "gradient" is non-positive for every unstable neuron.

`terms` is the list of admitted coefficients `A[j,i]` (those passing the `a_ji > 0`
guard at `gradients.rs:115`); `l` is the neuron's pre-activation lower bound, strictly
negative for an unstable neuron. The rule computes `(Σ terms) * l`.

There is no hypothesis relating `terms` to the objective, because the conclusion does
not need one: the sign is fixed by the guards alone.
-/
theorem local_rule_nonpos (terms : List ℚ) (l : ℚ)
    (hpos : ∀ a ∈ terms, 0 < a) (hl : l < 0) :
    (terms.sum) * l ≤ 0 := by
  have hsum : 0 ≤ terms.sum :=
    List.sum_nonneg fun a ha => le_of_lt (hpos a ha)
  exact mul_nonpos_of_nonneg_of_nonpos hsum (le_of_lt hl)

/-! ## T2. Hence the ascent can only move α down. -/

/--
**T2.** The Adam update never increases α.

`alpha_update.rs:60-69` computes `neg_g = -g`, accumulates `m` and `v` from it, and
applies `a ← a - lr * m̂ / (√v̂ + ε)`.  With `g ≤ 0` (T1) the sign of `neg_g` is
non-negative, so — for the non-negative `m̂` that a non-negative history produces, any
`lr > 0`, and the strictly positive denominator `√v̂ + ε` — the step SUBTRACTS a
non-negative quantity.

Stated on the step itself (`step = lr * m̂ / denom`), which is what the code applies,
so the conclusion holds for the update as written rather than for an idealisation of it.
-/
theorem adam_step_nonincreasing
    (a lr mhat denom : ℚ)
    (hlr : 0 < lr) (hm : 0 ≤ mhat) (hden : 0 < denom) :
    a - lr * mhat / denom ≤ a := by
  have hstep : 0 ≤ lr * mhat / denom :=
    div_nonneg (mul_nonneg (le_of_lt hlr) hm) (le_of_lt hden)
  linarith

/--
**T2'.** Clamping does not rescue it: a non-increasing step stays non-increasing after
the `a.clamp(0,1)` at `alpha_update.rs:69`, provided the pre-step value was already in
range (it is: α is initialised in `[0,1]` and every update re-clamps).

So across iterations α is monotonically non-increasing for every unstable neuron —
a descent to the `0` clamp, independent of the objective.  This is the formal statement
of the observed `best_impr = 0.000e0`.
-/
theorem clamped_step_nonincreasing
    (a lr mhat denom : ℚ)
    (ha0 : 0 ≤ a) (_ha1 : a ≤ 1)
    (hlr : 0 < lr) (hm : 0 ≤ mhat) (hden : 0 < denom) :
    max 0 (min 1 (a - lr * mhat / denom)) ≤ a := by
  have hstep := adam_step_nonincreasing a lr mhat denom hlr hm hden
  have hmin : min 1 (a - lr * mhat / denom) ≤ a :=
    le_trans (min_le_right _ _) hstep
  exact max_le ha0 hmin

/-! ## T3. The rule's sign is not merely imprecise — it can be opposite. -/

/--
**T3.** Where the binding pre-activation value at the concretization argmin is
POSITIVE, the true derivative `A[j,i] * h` is strictly positive while the shipped rule
`A[j,i] * l` is strictly negative.

`h` is `ĥ_i(x*)`, the neuron's own pre-activation evaluated at the argmin of the FINAL
row's concretization — it ranges over `[l, u]` and equals `l` only when `x*` happens to
minimise this particular neuron.  On a 40-layer ResNet with 99 objectives that
coincidence is the exception, not the rule.

The two conclusions together say the rule is not a bounded-error approximation of the
derivative: on this region it points the other way, so following it is anti-ascent.
-/
theorem local_rule_sign_can_be_wrong (a l u h : ℚ)
    (ha : 0 < a) (hl : l < 0) (_hu : 0 < u)
    (_hmem : l ≤ h ∧ h ≤ u) (hhpos : 0 < h) :
    0 < a * h ∧ a * l < 0 := by
  refine ⟨mul_pos ha hhpos, ?_⟩
  exact mul_neg_of_pos_of_neg ha hl

/--
**T3'.** The two disagree on a concrete witness, so this is not vacuous.
`A = 1`, `l = -1`, `u = 1`, `h = 1/2`: true derivative `+1/2`, shipped rule `-1`.
-/
example : (0:ℚ) < 1 * (1/2) ∧ (1:ℚ) * (-1) < 0 := by norm_num

/-! ## T4. Why fixing it cannot cost soundness. -/

/--
**T4.** For every `α ∈ [0,1]` and every pre-activation `z`, the lower envelope holds.

This is `Crownproof.relu_lower` applied verbatim; it is restated here to make the
dependency explicit at the point where the argument is used.

Consequence for the engineering decision: the α optimizer selects a POINT in `[0,1]`,
and soundness holds at every such point.  Therefore replacing the gradient — with the
envelope-theorem rule, with autodiff, or with anything else — cannot introduce a
false `unsat`.  It can only change WHICH sound bound is obtained, i.e. bound quality.
That is what makes this defect safe to fix, and it is why the fix does not require the
soundness campaign that a relaxation change would.
-/
theorem alpha_sound_regardless (alpha z : ℚ) (h0 : 0 ≤ alpha) (h1 : alpha ≤ 1) :
    alpha * z ≤ relu z :=
  relu_lower alpha z h0 h1

/--
**T4'.** In particular the degenerate endpoint the shipped ascent drives toward,
`α = 0`, is sound — which is why this defect has never produced a wrong verdict.
It is pure dead weight: ~22% of the scored budget spent reaching a valid but
maximally loose envelope.
-/
theorem alpha_zero_sound (z : ℚ) : (0:ℚ) * z ≤ relu z :=
  alpha_sound_regardless 0 z le_rfl zero_le_one

end NyProof.AlphaGradientDefect

#print axioms NyProof.AlphaGradientDefect.local_rule_nonpos
#print axioms NyProof.AlphaGradientDefect.adam_step_nonincreasing
#print axioms NyProof.AlphaGradientDefect.clamped_step_nonincreasing
#print axioms NyProof.AlphaGradientDefect.local_rule_sign_can_be_wrong
#print axioms NyProof.AlphaGradientDefect.alpha_sound_regardless
#print axioms NyProof.AlphaGradientDefect.alpha_zero_sound
