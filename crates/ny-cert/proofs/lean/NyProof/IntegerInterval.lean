/-
  IntegerInterval.lean — exact integer/lattice interval reasoning for binarized nets.

  # Why this file exists

  `SignFusion.lean` proves that the ONNX idiom `Sign → Add(c) → Sign` (with `0 < |c| < 1`)
  fuses to a TWO-valued step `hs`/`hs'`, and that value-branching such a node is sound.
  What it does NOT fix is the *bound* side: on the traffic BNNs the pre-activation of the
  second convolution is EXACTLY `0` on whole regions, and ny's sound OUTWARD rounding widens
  `[0,0]` to `[-ε, +ε]`, which straddles the breakpoint, so the unit is unstable forever and
  BaB cannot close it.

  The measured structure of those nets (see the header table below) says that all such
  quantities are INTEGERS — in fact even integers.  Over the integers, an interval carries
  strictly more information than over the reals:

  * a real bound `[-0.3, 0.7]` on an integral quantity is exactly `{0}` (`intTighten`);
  * a real bound `[0.2, 0.7]` on an integral quantity is EMPTY, i.e. the branch is
    refuted for free (`intTighten_infeasible`);
  * stability tests become exact integer comparisons, with no `u = 0` boundary hazard
    (`hs_stable_neg_int` / `hs_stable_pos_int`, and their optimality);
  * on a `d`-lattice (`d = 2` for these nets, `d = 4` for many output margins) a certified
    lower bound of `−d + δ` already PROVES `margin ≥ 0` (`lattice_lower_forces_nonneg`)
    — the operational lever, since ny's best margin bounds stall in the hundreds.

  Nothing here is a trade-off: `ℤ ⊆ ℝ` and exact arithmetic is trivially an enclosure, so
  every theorem is a strict tightening of the corresponding real-interval fact.  The
  soundness direction (`intTighten_sound`, `lattice_tighten_sound`) is what a verifier must
  cite; the tightness/optimality direction (`intTighten_optimal`, `hs_stable_neg_int_sharp`)
  is what says the tightening cannot be improved, so an implementation that computes these
  bounds is not leaving anything on the table.

  # HONEST SCOPE — where the theory does and does not apply  (measured, not assumed)

  The vnnlib specs declare `(declare-const X_i Real)`, so the admissible set is a
  CONTINUOUS box that merely happens to have integer corners.  Therefore:

  * **Layer 1 is NOT integral.**  `∑ ±1 · x_i` over real `x_i` is not an integer.
    `box_contains_nonintegral` is the machine-checked refutation: any box of positive
    width contains a non-integral point, so no integrality hypothesis is available at the
    first convolution.  This was confirmed on the real nets (ORT, 8 random real points:
    the first `Conv` output was non-integral in 8/8 draws).
  * **Everything from the first `Sign` onward IS integral.**  `hs`/`hs'` are two-valued
    (`SignFusion.hs_two_valued`) irrespective of their argument, so `pm1_weighted_sum`
    applies from layer 2 on with no integrality hypothesis on the input at all.  This is
    exactly where the blocker lives, so the layer-1 loss is not fatal.
  * **`MaxPool` preserves it** (`isIntegral_max`), and **`BatchNormalization` /
    per-channel-scaled convolutions do not** — but they are order-preserving affine maps
    with a positive scale, and `affine_sign_iff_int` reduces the `Sign` after them to an
    exact integer comparison anyway, with `affine_ne_zero_of_threshold_not_int` showing
    the `u = 0` hazard is vacuous whenever the threshold is not an integer.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Archimedean
import Mathlib.Data.Int.GCD
import Mathlib.Algebra.Order.Floor.Defs
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Algebra.Order.Floor.Div
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import NyProof.SignFusion

namespace Crownproof

namespace IntegerInterval

open Crownproof.SignFusion

/-! ## §0  Integrality and lattice membership

`IsIntegral v` is the hypothesis a verifier discharges structurally (from `±1` weights and
`±1` activations); `OnLattice d v` is its refinement to `v ∈ d·ℤ`, which is what the traffic
nets actually satisfy (`d = 2`, because every fan-in is even). -/

/-- `v` is an integer, viewed inside `ℝ`. -/
def IsIntegral (v : ℝ) : Prop := ∃ m : ℤ, v = (m : ℝ)

/-- `v` lies on the lattice `d·ℤ`. -/
def OnLattice (d : ℤ) (v : ℝ) : Prop := ∃ k : ℤ, v = ((d * k : ℤ) : ℝ)

theorem isIntegral_intCast (m : ℤ) : IsIntegral (m : ℝ) := ⟨m, rfl⟩

theorem isIntegral_zero : IsIntegral 0 := ⟨0, by norm_num⟩

theorem isIntegral_one : IsIntegral 1 := ⟨1, by norm_num⟩

theorem isIntegral_neg_one : IsIntegral (-1) := ⟨-1, by norm_num⟩

theorem OnLattice.isIntegral {d : ℤ} {v : ℝ} (h : OnLattice d v) : IsIntegral v := by
  obtain ⟨k, hk⟩ := h; exact ⟨d * k, hk⟩

theorem onLattice_one {v : ℝ} (h : IsIntegral v) : OnLattice 1 v := by
  obtain ⟨m, hm⟩ := h; exact ⟨m, by simpa using hm⟩

/-- Membership in `d·ℤ` is closed under addition (used to propagate the even-parity
    invariant through the layer-wise sums). -/
theorem OnLattice.add {d : ℤ} {x y : ℝ} (hx : OnLattice d x) (hy : OnLattice d y) :
    OnLattice d (x + y) := by
  obtain ⟨j, hj⟩ := hx; obtain ⟨k, hk⟩ := hy
  exact ⟨j + k, by rw [hj, hk]; push_cast; ring⟩

theorem OnLattice.sub {d : ℤ} {x y : ℝ} (hx : OnLattice d x) (hy : OnLattice d y) :
    OnLattice d (x - y) := by
  obtain ⟨j, hj⟩ := hx; obtain ⟨k, hk⟩ := hy
  exact ⟨j - k, by rw [hj, hk]; push_cast; ring⟩

theorem OnLattice.neg {d : ℤ} {x : ℝ} (hx : OnLattice d x) : OnLattice d (-x) := by
  obtain ⟨k, hk⟩ := hx
  exact ⟨-k, by rw [hk]; push_cast; ring⟩

/-! ## §1  INTEGER TIGHTENING is sound, is an enclosure, and is optimal

This is claim (2) of the theory: a real bound `[l,u]` on an integral quantity may be
replaced by `[⌈l⌉, ⌊u⌋]` for free.  Three separate facts are needed, and a verifier that
cites only the first is citing the wrong thing:

* `intTighten_sound` — the tightened interval still CONTAINS `v` (soundness / enclosure of
  the true value);
* `intTighten_contracts` — the tightened interval is a SUBSET of the original (so it is a
  tightening, not a widening: this is the direction that would be violated by a buggy
  round-the-wrong-way);
* `intTighten_optimal` — the endpoints are attained by integers, so no strictly smaller
  interval with the same guarantee exists. -/

/-- **INTEGER TIGHTENING (soundness).**  An integral `v` known to lie in `[l,u]` lies in
    `[⌈l⌉, ⌊u⌋]`. -/
theorem intTighten_sound {v l u : ℝ} (hv : IsIntegral v) (hl : l ≤ v) (hu : v ≤ u) :
    ((⌈l⌉ : ℤ) : ℝ) ≤ v ∧ v ≤ ((⌊u⌋ : ℤ) : ℝ) := by
  obtain ⟨m, rfl⟩ := hv
  constructor
  · exact_mod_cast Int.ceil_le.mpr hl
  · exact_mod_cast Int.le_floor.mpr hu

/-- **INTEGER TIGHTENING is an enclosure**: `[⌈l⌉, ⌊u⌋] ⊆ [l, u]`.  Together with
    `intTighten_sound` this is the full "sound by construction, and tighter" claim. -/
theorem intTighten_contracts (l u : ℝ) :
    l ≤ ((⌈l⌉ : ℤ) : ℝ) ∧ ((⌊u⌋ : ℤ) : ℝ) ≤ u :=
  ⟨Int.le_ceil l, Int.floor_le u⟩

/-- Set form of the previous two: the tightened box is sandwiched, so replacing `[l,u]` by
    `[⌈l⌉,⌊u⌋]` loses no integral point and gains no real one. -/
theorem intTighten_set_eq (l u : ℝ) :
    {v : ℝ | IsIntegral v ∧ l ≤ v ∧ v ≤ u}
      = {v : ℝ | IsIntegral v ∧ ((⌈l⌉ : ℤ) : ℝ) ≤ v ∧ v ≤ ((⌊u⌋ : ℤ) : ℝ)} := by
  ext v
  constructor
  · rintro ⟨hv, hl, hu⟩
    exact ⟨hv, (intTighten_sound hv hl hu).1, (intTighten_sound hv hl hu).2⟩
  · rintro ⟨hv, hl, hu⟩
    exact ⟨hv, le_trans (Int.le_ceil l) hl, le_trans hu (Int.floor_le u)⟩

/-- **Tightening is OPTIMAL**: both endpoints of `[⌈l⌉, ⌊u⌋]` are themselves integral and,
    when the tightened interval is nonempty, are attained.  No smaller interval is sound. -/
theorem intTighten_optimal {l u : ℝ} (h : (⌈l⌉ : ℤ) ≤ ⌊u⌋) :
    IsIntegral ((⌈l⌉ : ℤ) : ℝ) ∧ l ≤ ((⌈l⌉ : ℤ) : ℝ) ∧ ((⌈l⌉ : ℤ) : ℝ) ≤ u ∧
    IsIntegral ((⌊u⌋ : ℤ) : ℝ) ∧ l ≤ ((⌊u⌋ : ℤ) : ℝ) ∧ ((⌊u⌋ : ℤ) : ℝ) ≤ u := by
  have h1 : ((⌈l⌉ : ℤ) : ℝ) ≤ ((⌊u⌋ : ℤ) : ℝ) := by exact_mod_cast h
  refine ⟨isIntegral_intCast _, Int.le_ceil l, ?_, isIntegral_intCast _, ?_, Int.floor_le u⟩
  · exact le_trans h1 (Int.floor_le u)
  · exact le_trans (Int.le_ceil l) h1

/-- **Free refutation.**  If the tightened interval is empty (`⌊u⌋ < ⌈l⌉`) then NO integral
    value lies in `[l,u]` — the branch is closed with no search at all.  This is the
    integer analogue of an infeasible LP relaxation and has no real-interval counterpart:
    `[0.2, 0.7]` is nonempty over `ℝ` and empty over `ℤ`. -/
theorem intTighten_infeasible {l u : ℝ} (h : (⌊u⌋ : ℤ) < ⌈l⌉) :
    ¬ ∃ v : ℝ, IsIntegral v ∧ l ≤ v ∧ v ≤ u := by
  rintro ⟨v, hv, hl, hu⟩
  obtain ⟨m, rfl⟩ := hv
  have h1 : (⌈l⌉ : ℤ) ≤ m := Int.ceil_le.mpr hl
  have h2 : m ≤ (⌊u⌋ : ℤ) := Int.le_floor.mpr hu
  omega

/-- Tightening is idempotent: applying it to already-integral endpoints is a no-op, so a
    fixed-point iteration terminates immediately. -/
theorem intTighten_idem (a b : ℤ) :
    ⌈((a : ℤ) : ℝ)⌉ = a ∧ ⌊((b : ℤ) : ℝ)⌋ = b :=
  ⟨Int.ceil_intCast a, Int.floor_intCast b⟩

/-! ## §2  LATTICE tightening, and the decisive margin lemma

The traffic nets satisfy the stronger `OnLattice 2` (every fan-in is even, every weight is
`±1`, every activation is `±1`), and many output-margin pairs satisfy `OnLattice 4`.  The
general-`d` statements below specialise to those. -/

/-- **LATTICE TIGHTENING (soundness).**  If `v ∈ d·ℤ` with `d > 0` and `l ≤ v ≤ u`, then
    `d·⌈l/d⌉ ≤ v ≤ d·⌊u/d⌋`.  For `d = 1` this is `intTighten_sound`. -/
theorem lattice_tighten_sound {d : ℤ} (hd : 0 < d) {v l u : ℝ}
    (hv : OnLattice d v) (hl : l ≤ v) (hu : v ≤ u) :
    ((d * ⌈l / (d : ℝ)⌉ : ℤ) : ℝ) ≤ v ∧ v ≤ ((d * ⌊u / (d : ℝ)⌋ : ℤ) : ℝ) := by
  obtain ⟨k, rfl⟩ := hv
  have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  have hlk : (⌈l / (d : ℝ)⌉ : ℤ) ≤ k := by
    refine Int.ceil_le.mpr ?_
    rw [div_le_iff₀ hdR]
    have : l ≤ ((d * k : ℤ) : ℝ) := hl
    push_cast at this ⊢
    linarith
  have huk : k ≤ (⌊u / (d : ℝ)⌋ : ℤ) := by
    refine Int.le_floor.mpr ?_
    rw [le_div_iff₀ hdR]
    have : ((d * k : ℤ) : ℝ) ≤ u := hu
    push_cast at this ⊢
    linarith
  constructor
  · have : (d * ⌈l / (d : ℝ)⌉ : ℤ) ≤ d * k := by
      exact Int.mul_le_mul_of_nonneg_left hlk (le_of_lt hd)
    exact_mod_cast this
  · have : (d * k : ℤ) ≤ d * ⌊u / (d : ℝ)⌋ := by
      exact Int.mul_le_mul_of_nonneg_left huk (le_of_lt hd)
    exact_mod_cast this

/-- **THE MARGIN LEMMA (division-free, and the operational lever).**  If the output margin
    lies on `d·ℤ` and a verifier certifies the weak lower bound `−d < l ≤ margin`, then
    `0 ≤ margin` — the property is PROVED.  With `d = 4` a certified bound of `−3.9` already
    decides a row that a real-valued verifier would still call unknown. -/
theorem lattice_lower_forces_nonneg {d : ℤ} (hd : 0 < d) {v l : ℝ}
    (hv : OnLattice d v) (hl : l ≤ v) (hgap : -(d : ℝ) < l) : 0 ≤ v := by
  obtain ⟨k, rfl⟩ := hv
  have h1 : -(d : ℝ) < ((d * k : ℤ) : ℝ) := lt_of_lt_of_le hgap hl
  have h3 : (0 : ℤ) ≤ k := by
    by_contra hk
    have hk1 : k ≤ -1 := by omega
    have hz : d * k ≤ d * (-1) := Int.mul_le_mul_of_nonneg_left hk1 (le_of_lt hd)
    have hcast : ((d * k : ℤ) : ℝ) ≤ ((d * (-1) : ℤ) : ℝ) := by exact_mod_cast hz
    push_cast at hcast h1
    linarith
  have h4 : (0 : ℤ) ≤ d * k := mul_nonneg (le_of_lt hd) h3
  exact_mod_cast h4

/-- **The gap is SHARP.**  With `l = −d` exactly, the conclusion fails: `v = −d` is on the
    lattice and satisfies `l ≤ v`, yet `v < 0`.  So the strict `−d < l` cannot be weakened,
    and an implementation must not use `≤` here. -/
theorem lattice_lower_forces_nonneg_sharp (d : ℤ) (hd : 0 < d) :
    ∃ v l : ℝ, OnLattice d v ∧ l ≤ v ∧ -(d : ℝ) ≤ l ∧ ¬ (0 ≤ v) := by
  refine ⟨((d * (-1) : ℤ) : ℝ), -(d : ℝ), ⟨-1, rfl⟩, by push_cast; linarith, le_refl _, ?_⟩
  have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  push_cast
  linarith

/-- Mirror for upper bounds: a certified `v ≤ u < d` on a `d`-lattice quantity gives
    `v ≤ 0`. -/
theorem lattice_upper_forces_nonpos {d : ℤ} (hd : 0 < d) {v u : ℝ}
    (hv : OnLattice d v) (hu : v ≤ u) (hgap : u < (d : ℝ)) : v ≤ 0 := by
  have h := lattice_lower_forces_nonneg hd hv.neg (neg_le_neg hu) (by linarith)
  linarith

/-! ## §3  Integer interval arithmetic for `+`, `−` and `±1`-weighted sums is EXACT

Claim (1) of the theory.  "Exact" is given its only meaningful formal content: the
interval computed from integer endpoints is simultaneously an ENCLOSURE of the true image
and ATTAINED at both ends, so no outward widening is required — in contrast with the float
path, where each operation must round outward and the certified-error channel accumulates. -/

/-- Integrality is closed under `+`. -/
theorem isIntegral_add {x y : ℝ} (hx : IsIntegral x) (hy : IsIntegral y) :
    IsIntegral (x + y) := by
  obtain ⟨a, rfl⟩ := hx; obtain ⟨b, rfl⟩ := hy
  exact ⟨a + b, by push_cast; ring⟩

/-- Integrality is closed under `−`. -/
theorem isIntegral_sub {x y : ℝ} (hx : IsIntegral x) (hy : IsIntegral y) :
    IsIntegral (x - y) := by
  obtain ⟨a, rfl⟩ := hx; obtain ⟨b, rfl⟩ := hy
  exact ⟨a - b, by push_cast; ring⟩

theorem isIntegral_neg {x : ℝ} (hx : IsIntegral x) : IsIntegral (-x) := by
  obtain ⟨a, rfl⟩ := hx; exact ⟨-a, by push_cast; ring⟩

/-- Integrality is closed under `*` (needed for the `±1`-weight product). -/
theorem isIntegral_mul {x y : ℝ} (hx : IsIntegral x) (hy : IsIntegral y) :
    IsIntegral (x * y) := by
  obtain ⟨a, rfl⟩ := hx; obtain ⟨b, rfl⟩ := hy
  exact ⟨a * b, by push_cast; ring⟩

/-- `MaxPool` preserves integrality (it is a selection, not an arithmetic operation). -/
theorem isIntegral_max {x y : ℝ} (hx : IsIntegral x) (hy : IsIntegral y) :
    IsIntegral (max x y) := by
  rcases le_total x y with h | h
  · rwa [max_eq_right h]
  · rwa [max_eq_left h]

/-- `MaxPool` also preserves the `d`-lattice — needed for nets 2 and 3, where a `MaxPool`
    sits between the `±1` convolution and the `BatchNormalization`, so the quantity reaching
    the affine layer is still on `2ℤ`. -/
theorem onLattice_max {d : ℤ} {x y : ℝ} (hx : OnLattice d x) (hy : OnLattice d y) :
    OnLattice d (max x y) := by
  rcases le_total x y with h | h
  · rwa [max_eq_right h]
  · rwa [max_eq_left h]

/-- **Interval addition on integers is an ENCLOSURE.** -/
theorem int_interval_add_sound {x y a b c e : ℝ}
    (hxa : a ≤ x) (hxb : x ≤ b) (hyc : c ≤ y) (hye : y ≤ e) :
    a + c ≤ x + y ∧ x + y ≤ b + e :=
  ⟨add_le_add hxa hyc, add_le_add hxb hye⟩

/-- **…and it is EXACT**: both endpoints are attained by integral operands lying in the
    operand intervals, so `[a+c, b+e]` cannot be shrunk.  (Stated with integer endpoints,
    which is the case the BNN pipeline produces.) -/
theorem int_interval_add_exact (a b c e : ℤ) (hab : a ≤ b) (hce : c ≤ e) :
    (∃ x y : ℝ, IsIntegral x ∧ IsIntegral y ∧ (a : ℝ) ≤ x ∧ x ≤ b ∧ (c : ℝ) ≤ y ∧ y ≤ e ∧
        x + y = ((a + c : ℤ) : ℝ)) ∧
    (∃ x y : ℝ, IsIntegral x ∧ IsIntegral y ∧ (a : ℝ) ≤ x ∧ x ≤ b ∧ (c : ℝ) ≤ y ∧ y ≤ e ∧
        x + y = ((b + e : ℤ) : ℝ)) := by
  have hab' : ((a : ℤ) : ℝ) ≤ ((b : ℤ) : ℝ) := by exact_mod_cast hab
  have hce' : ((c : ℤ) : ℝ) ≤ ((e : ℤ) : ℝ) := by exact_mod_cast hce
  refine ⟨⟨(a : ℝ), (c : ℝ), isIntegral_intCast a, isIntegral_intCast c, le_refl _, hab',
      le_refl _, hce', by push_cast; ring⟩,
    ⟨(b : ℝ), (e : ℝ), isIntegral_intCast b, isIntegral_intCast e, hab', le_refl _,
      hce', le_refl _, by push_cast; ring⟩⟩

/-- **Interval subtraction on integers is an ENCLOSURE** (with the operand order the
    negate-and-swap convention requires — the site of a previously confirmed false-proof
    in ny's `Sub` handling). -/
theorem int_interval_sub_sound {x y a b c e : ℝ}
    (hxa : a ≤ x) (hxb : x ≤ b) (hyc : c ≤ y) (hye : y ≤ e) :
    a - e ≤ x - y ∧ x - y ≤ b - c :=
  ⟨sub_le_sub hxa hye, sub_le_sub hxb hyc⟩

/-! ### `±1`-weighted sums

The core structural fact.  `f i` ranges over the pointwise products `w i * s i` of a `±1`
weight and a `±1` activation, so `f i ∈ {−1, +1}` and no integrality hypothesis is needed
on the layer's INPUT — which is exactly why the argument survives the continuous input box
from layer 2 onward. -/

/-- A product of two `±1` values is `±1`. -/
theorem pm1_mul {w s : ℝ} (hw : w = 1 ∨ w = -1) (hs : s = 1 ∨ s = -1) :
    w * s = 1 ∨ w * s = -1 := by
  rcases hw with rfl | rfl <;> rcases hs with rfl | rfl <;> norm_num

/-- **`±1`-WEIGHTED SUMS ARE EXACTLY INTEGRAL**, with the two extra facts a bound engine
    wants: the sum is bounded by the fan-in, and it has the PARITY of the fan-in (so an
    even fan-in — every layer of every traffic net — forces `OnLattice 2`). -/
theorem pm1_weighted_sum {ι : Type*} (s : Finset ι) (f : ι → ℝ)
    (hf : ∀ i ∈ s, f i = 1 ∨ f i = -1) :
    ∃ m : ℤ, (∑ i ∈ s, f i) = (m : ℝ) ∧ |m| ≤ (s.card : ℤ) ∧ (2 : ℤ) ∣ ((s.card : ℤ) - m) := by
  classical
  have hg : ∀ i ∈ s, ∃ g : ℤ, f i = (g : ℝ) ∧ (g = 1 ∨ g = -1) := by
    intro i hi
    rcases hf i hi with h | h
    · exact ⟨1, by rw [h]; norm_num, Or.inl rfl⟩
    · exact ⟨-1, by rw [h]; norm_num, Or.inr rfl⟩
  choose! g hgf hgpm using hg
  refine ⟨∑ i ∈ s, g i, ?_, ?_, ?_⟩
  · push_cast
    exact Finset.sum_congr rfl fun i hi => hgf i hi
  · calc |∑ i ∈ s, g i| ≤ ∑ i ∈ s, |g i| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ _i ∈ s, (1 : ℤ) := by
          refine Finset.sum_congr rfl fun i hi => ?_
          rcases hgpm i hi with h | h <;> rw [h] <;> decide
      _ = (s.card : ℤ) := by simp
  · have hrw : (s.card : ℤ) - ∑ i ∈ s, g i = ∑ i ∈ s, (1 - g i) := by
      rw [Finset.sum_sub_distrib]; simp
    rw [hrw]
    refine Finset.dvd_sum fun i hi => ?_
    rcases hgpm i hi with h | h <;> rw [h] <;> decide

/-- Specialisation actually used by the nets: an EVEN fan-in of `±1` terms lands on `2ℤ`. -/
theorem pm1_weighted_sum_even {ι : Type*} (s : Finset ι) (f : ι → ℝ)
    (hf : ∀ i ∈ s, f i = 1 ∨ f i = -1) (hcard : (2 : ℤ) ∣ (s.card : ℤ)) :
    OnLattice 2 (∑ i ∈ s, f i) ∧
      -(s.card : ℝ) ≤ (∑ i ∈ s, f i) ∧ (∑ i ∈ s, f i) ≤ (s.card : ℝ) := by
  obtain ⟨m, hm, habs, hpar⟩ := pm1_weighted_sum s f hf
  obtain ⟨hlo, hhi⟩ := abs_le.mp habs
  refine ⟨?_, ?_, ?_⟩
  · obtain ⟨t, ht⟩ : (2 : ℤ) ∣ m := (dvd_sub_right hcard).mp hpar
    exact ⟨t, by rw [hm, ht]⟩
  · rw [hm]; exact_mod_cast hlo
  · rw [hm]; exact_mod_cast hhi

/-- The interval `[−n, n]` produced by interval arithmetic on a `±1`-weighted sum of fan-in
    `n` is EXACT: it is attained at `s = w` and at `s = −w`.  So the enclosure is not merely
    sound, it is the true range — there is nothing for a tighter relaxation to recover. -/
theorem pm1_range_exact {ι : Type*} (s : Finset ι) (w : ι → ℝ)
    (hw : ∀ i ∈ s, w i = 1 ∨ w i = -1) :
    (∑ i ∈ s, w i * w i) = (s.card : ℝ) ∧ (∑ i ∈ s, w i * (-w i)) = -(s.card : ℝ) := by
  have h1 : ∀ i ∈ s, w i * w i = 1 := by
    intro i hi; rcases hw i hi with h | h <;> rw [h] <;> norm_num
  constructor
  · rw [Finset.sum_congr rfl h1]; simp
  · have h2 : ∀ i ∈ s, w i * (-w i) = -1 := by
      intro i hi; rcases hw i hi with h | h <;> rw [h] <;> norm_num
    rw [Finset.sum_congr rfl h2]; simp

/-! ## §4  EXACT STABILITY on integers — and it is ASYMMETRIC

`SignFusion.hs_fix_neg_on_closed_unsound` proves the real-valued stability tests for a fused
node with `hs 0 = +1` must be `u < 0 ⇒ −1` (STRICT) and `0 ≤ l ⇒ +1`.  Over the integers the
STRICT/NON-STRICT distinction becomes a difference of one lattice step, and the two tests
below are the exact integer forms.  They are strictly weaker hypotheses than the real ones —
`⌊u⌋ ≤ −1` fires whenever `u < 0`, and `0 ≤ ⌈l⌉` fires whenever `−1 < l`, not merely when
`0 ≤ l` — so more units become stable, which is the point.

The asymmetry is preserved verbatim: `⌊u⌋ ≤ −1` on the negative side (an integer strictly
below `0`), `0 ≤ ⌈l⌉` on the positive side (an integer at or above `0`).  Collapsing the
first to the symmetric `⌊u⌋ ≤ 0` is refuted by `hs_stable_neg_int_unsound`. -/

/-- **EXACT STABILITY, `+1` side.**  For an INTEGRAL pre-activation, `0 ≤ ⌈l⌉` suffices —
    strictly weaker than the real test `0 ≤ l`. -/
theorem hs_stable_pos_int {z l : ℝ} (hz : IsIntegral z) (hl : l ≤ z) (h : (0 : ℤ) ≤ ⌈l⌉) :
    hs z = 1 := by
  obtain ⟨m, rfl⟩ := hz
  have h1 : (⌈l⌉ : ℤ) ≤ m := Int.ceil_le.mpr hl
  have h2 : (0 : ℤ) ≤ m := le_trans h h1
  exact hs_nonneg (by exact_mod_cast h2)

/-- **EXACT STABILITY, `−1` side — STRICT, matching `hs_fix_neg_on_closed_unsound`.**
    The integer test is `⌊u⌋ ≤ −1`, i.e. an integer STRICTLY below `0`. -/
theorem hs_stable_neg_int {z u : ℝ} (hz : IsIntegral z) (hu : z ≤ u) (h : (⌊u⌋ : ℤ) ≤ -1) :
    hs z = -1 := by
  obtain ⟨m, rfl⟩ := hz
  have h1 : m ≤ (⌊u⌋ : ℤ) := Int.le_floor.mpr hu
  have h2 : m ≤ -1 := le_trans h1 h
  have h3 : ((m : ℤ) : ℝ) < 0 := by
    have : ((m : ℤ) : ℝ) ≤ -1 := by exact_mod_cast h2
    linarith
  exact hs_neg h3

/-- **REFUTATION of the symmetric integer test.**  `⌊u⌋ ≤ 0 ⇒ −1` is UNSOUND: `z = 0` is
    integral, satisfies `z ≤ 0` and `⌊(0:ℝ)⌋ ≤ 0`, yet `hs 0 = +1`.  This is the integer
    image of `SignFusion.hs_fix_neg_on_closed_unsound`, and of the symmetric predicate
    `l >= 0.0 || u <= 0.0` at `beta_crown/nonlinear_branching/scoring.rs:44`: porting that
    predicate to the integer lane WITHOUT restoring the strictness reintroduces exactly the
    same false-`unsat` source. -/
theorem hs_stable_neg_int_unsound :
    ∃ (z u : ℝ), IsIntegral z ∧ z ≤ u ∧ (⌊u⌋ : ℤ) ≤ 0 ∧ hs z ≠ -1 := by
  refine ⟨0, 0, isIntegral_zero, le_refl _, ?_, ?_⟩
  · simp
  · rw [hs_nonneg (le_refl (0 : ℝ))]; norm_num

/-- **The `−1` test is SHARP.**  If `⌊u⌋ ≥ 0` then some integral `z ≤ u` has `hs z = +1`,
    so `⌊u⌋ ≤ −1` cannot be weakened at all: it is the exact stability boundary. -/
theorem hs_stable_neg_int_sharp {u : ℝ} (h : (0 : ℤ) ≤ ⌊u⌋) :
    ∃ z : ℝ, IsIntegral z ∧ z ≤ u ∧ hs z = 1 := by
  refine ⟨0, isIntegral_zero, ?_, hs_nonneg (le_refl (0 : ℝ))⟩
  have : ((0 : ℤ) : ℝ) ≤ ((⌊u⌋ : ℤ) : ℝ) := by exact_mod_cast h
  have h2 : ((⌊u⌋ : ℤ) : ℝ) ≤ u := Int.floor_le u
  push_cast at this
  linarith

/-- **The `+1` test is SHARP.**  If `⌈l⌉ ≤ −1` then some integral `z ≥ l` has `hs z = −1`. -/
theorem hs_stable_pos_int_sharp {l : ℝ} (h : (⌈l⌉ : ℤ) ≤ -1) :
    ∃ z : ℝ, IsIntegral z ∧ l ≤ z ∧ hs z = -1 := by
  refine ⟨((-1 : ℤ) : ℝ), isIntegral_intCast _, ?_, ?_⟩
  · have h1 : l ≤ ((⌈l⌉ : ℤ) : ℝ) := Int.le_ceil l
    have h2 : ((⌈l⌉ : ℤ) : ℝ) ≤ ((-1 : ℤ) : ℝ) := by exact_mod_cast h
    linarith
  · exact hs_neg (by norm_num)

/-- **EXACT DECIDABILITY.**  On a point interval (`l = u = z`, integral) the two stability
    tests are EXHAUSTIVE: one of them always fires.  Over the reals this fails precisely at
    the breakpoint, which is the `u = 0` hazard; over the integers there is no boundary case
    left, because `0` is itself an admissible integer and lands in the `+1` test. -/
theorem hs_stable_int_complete {z : ℝ} (hz : IsIntegral z) :
    ((0 : ℤ) ≤ ⌈z⌉ ∧ hs z = 1) ∨ ((⌊z⌋ : ℤ) ≤ -1 ∧ hs z = -1) := by
  obtain ⟨m, rfl⟩ := hz
  by_cases h : (0 : ℤ) ≤ m
  · left
    refine ⟨by rw [Int.ceil_intCast]; exact h, hs_nonneg (by exact_mod_cast h)⟩
  · right
    have h1 : m ≤ -1 := by omega
    refine ⟨by rw [Int.floor_intCast]; exact h1, hs_neg ?_⟩
    have : ((m : ℤ) : ℝ) ≤ -1 := by exact_mod_cast h1
    linarith

/-- Mirror for the other fusion polarity (`hs' 0 = −1`): the strictness moves to the `+1`
    side, so the integer tests are `1 ≤ ⌈l⌉` and `⌊u⌋ ≤ 0`. -/
theorem hs'_stable_pos_int {z l : ℝ} (hz : IsIntegral z) (hl : l ≤ z) (h : (1 : ℤ) ≤ ⌈l⌉) :
    hs' z = 1 := by
  obtain ⟨m, rfl⟩ := hz
  have h1 : (⌈l⌉ : ℤ) ≤ m := Int.ceil_le.mpr hl
  have h2 : (1 : ℤ) ≤ m := le_trans h h1
  have : (0 : ℝ) < ((m : ℤ) : ℝ) := by
    have : ((1 : ℤ) : ℝ) ≤ ((m : ℤ) : ℝ) := by exact_mod_cast h2
    push_cast at this; linarith
  exact hs'_pos this

theorem hs'_stable_neg_int {z u : ℝ} (hz : IsIntegral z) (hu : z ≤ u) (h : (⌊u⌋ : ℤ) ≤ 0) :
    hs' z = -1 := by
  obtain ⟨m, rfl⟩ := hz
  have h1 : m ≤ (⌊u⌋ : ℤ) := Int.le_floor.mpr hu
  have h2 : m ≤ 0 := le_trans h1 h
  exact hs'_nonpos (by exact_mod_cast h2)

/-- Mirror refutation: the symmetric `1 ≤ ⌈l⌉` relaxed to `0 ≤ ⌈l⌉` is unsound for `hs'`. -/
theorem hs'_stable_pos_int_unsound :
    ∃ (z l : ℝ), IsIntegral z ∧ l ≤ z ∧ (0 : ℤ) ≤ ⌈l⌉ ∧ hs' z ≠ 1 := by
  refine ⟨0, 0, isIntegral_zero, le_refl _, ?_, ?_⟩
  · simp
  · rw [hs'_nonpos (le_refl (0 : ℝ))]; norm_num

/-! ## §5  Monotone affine layers (BatchNormalization, per-channel-scaled convolutions)

Nets 2 and 3 interleave `BatchNormalization` (and one convolution whose weights are a
per-channel scale times a `±1` kernel) between the `Sign` layers.  These BREAK integrality —
the measured effective scales are irrational-looking floats — but they are affine with a
STRICTLY POSITIVE scale (measured: `gamma = 1` and `var > 0` in every channel of every BN of
both nets), hence order-preserving, so the `Sign` after them still reduces to an EXACT
integer comparison against a single threshold. -/

/-- **EXACT THRESHOLD REDUCTION.**  For `a > 0` and integral `z`, the sign of `a·z + b` is
    decided by an integer comparison against `⌈−b/a⌉`.  No relaxation, no rounding. -/
theorem affine_sign_iff_int {a b : ℝ} (ha : 0 < a) (m : ℤ) :
    0 ≤ a * (m : ℝ) + b ↔ (⌈-b / a⌉ : ℤ) ≤ m := by
  rw [Int.ceil_le, div_le_iff₀ ha]
  constructor <;> intro h <;> linarith [h]

/-- Consequently the fused node after a positive-scale affine layer is decided exactly:
    a single integer comparison replaces the whole relaxation. -/
theorem hs_affine_int {a b : ℝ} (ha : 0 < a) (m : ℤ) :
    hs (a * (m : ℝ) + b) = 1 ↔ (⌈-b / a⌉ : ℤ) ≤ m := by
  constructor
  · intro h
    rw [← affine_sign_iff_int ha]
    by_contra hc
    rw [hs_neg (not_le.mp hc)] at h
    norm_num at h
  · intro h
    exact hs_nonneg ((affine_sign_iff_int ha m).mpr h)

/-- **THE `u = 0` HAZARD IS VACUOUS AFTER A POSITIVE-SCALE AFFINE LAYER whose threshold is
    not an integer.**  Measured on the real artifacts: every BN channel of nets 2 and 3 has
    `var + eps` a non-square rational and `beta ≠ 0`, so `−b/a` is IRRATIONAL and this
    hypothesis holds unconditionally there. -/
theorem affine_ne_zero_of_threshold_not_int {a b : ℝ} (ha : 0 < a)
    (hthr : ¬ IsIntegral (-b / a)) {z : ℝ} (hz : IsIntegral z) : a * z + b ≠ 0 := by
  obtain ⟨m, rfl⟩ := hz
  intro h
  exact hthr ⟨m, by field_simp; linarith⟩

/-- Order-preservation, the reason a positive-scale affine layer cannot destroy a decided
    stability verdict even though it destroys integrality. -/
theorem affine_strictMono {a : ℝ} (ha : 0 < a) (b : ℝ) {x y : ℝ} (h : x ≤ y) :
    a * x + b ≤ a * y + b := by nlinarith

/-! ## §6  HONEST SCOPE — the continuous input box, stated as a theorem

The vnnlib files declare `Real` inputs.  The following two theorems delimit exactly where
the integer theory starts. -/

/-- **LAYER 1 IS NOT INTEGRAL.**  Any box of positive width — every traffic instance has
    width `≥ 1` in at least one coordinate — contains a non-integral point, so no
    integrality hypothesis is available at the first convolution even though the box corners
    are integers.  This is the critical caveat: `intTighten_sound` MUST NOT be applied to a
    layer-1 pre-activation. -/
theorem box_contains_nonintegral {l u : ℝ} (h : l < u) :
    ∃ x : ℝ, l ≤ x ∧ x ≤ u ∧ ¬ IsIntegral x := by
  -- two points of the box less than `1` apart cannot both be integers
  set w : ℝ := min (u - l) 1 with hwdef
  have hw0 : 0 < w := lt_min (by linarith) one_pos
  have hwu : w ≤ u - l := min_le_left _ _
  have hw1 : w ≤ 1 := min_le_right _ _
  by_cases h1 : IsIntegral (l + w / 3)
  · refine ⟨l + 2 * w / 3, by linarith, by linarith, ?_⟩
    rintro ⟨m2, hm2⟩
    obtain ⟨m1, hm1⟩ := h1
    have hd : ((m2 : ℤ) : ℝ) - ((m1 : ℤ) : ℝ) = w / 3 := by rw [← hm1, ← hm2]; ring
    have hpos : (0 : ℤ) < m2 - m1 := by
      have : (0 : ℝ) < ((m2 : ℤ) : ℝ) - ((m1 : ℤ) : ℝ) := by rw [hd]; linarith
      have h2 : (0 : ℝ) < ((m2 - m1 : ℤ) : ℝ) := by push_cast; linarith
      exact_mod_cast h2
    have hone : (1 : ℝ) ≤ ((m2 : ℤ) : ℝ) - ((m1 : ℤ) : ℝ) := by
      have : (1 : ℤ) ≤ m2 - m1 := hpos
      have h2 : (1 : ℝ) ≤ ((m2 - m1 : ℤ) : ℝ) := by exact_mod_cast this
      push_cast at h2; linarith
    rw [hd] at hone
    linarith
  · exact ⟨l + w / 3, by linarith, by linarith, h1⟩

/-- **LAYER 2 ONWARD IS INTEGRAL REGARDLESS.**  The fused node is two-valued for EVERY real
    argument, so the next layer's `±1`-weighted sum is integral with no hypothesis on the
    input box at all.  This is why the layer-1 loss above is not fatal: the blocker (a
    second-layer pre-activation pinned at `0`) lives strictly after the first `Sign`. -/
theorem post_sign_layer_integral {ι : Type*} (s : Finset ι) (w : ι → ℝ) (pre : ι → ℝ)
    (hw : ∀ i ∈ s, w i = 1 ∨ w i = -1) :
    ∃ m : ℤ, (∑ i ∈ s, w i * hs (pre i)) = (m : ℝ) ∧ |m| ≤ (s.card : ℤ) ∧
      (2 : ℤ) ∣ ((s.card : ℤ) - m) :=
  pm1_weighted_sum s (fun i => w i * hs (pre i))
    (fun i hi => pm1_mul (hw i hi) (hs_two_valued (pre i)).symm)

/-- The same for the even-fan-in case that every traffic layer actually has: the second
    convolution's pre-activation is on `2ℤ` over the WHOLE continuous box.  Instantiating
    `s.card = 64` (net 1, `16·2·2`) gives the exact statement the blocker needs. -/
theorem post_sign_layer_even {ι : Type*} (s : Finset ι) (w : ι → ℝ) (pre : ι → ℝ)
    (hw : ∀ i ∈ s, w i = 1 ∨ w i = -1) (hcard : (2 : ℤ) ∣ (s.card : ℤ)) :
    OnLattice 2 (∑ i ∈ s, w i * hs (pre i)) :=
  (pm1_weighted_sum_even s (fun i => w i * hs (pre i))
    (fun i hi => pm1_mul (hw i hi) (hs_two_valued (pre i)).symm) hcard).1

/-- **THE SPEC IS OVER SOFTMAX, AND THAT IS FINE.**  Every traffic net terminates in
    `Softmax` and the vnnlib constraints are `Y_i ≥ Y_j` on the SOFTMAX outputs, which are
    not integral.  But softmax divides by a common positive denominator and `exp` is
    strictly monotone, so the comparison is EXACTLY the pre-softmax logit comparison — which
    is where the `2ℤ` lattice lives.  Without this step the margin lemma would not reach
    the property. -/
theorem softmax_cmp_iff {S : ℝ} (hS : 0 < S) (zi zj : ℝ) :
    Real.exp zi / S ≤ Real.exp zj / S ↔ zi ≤ zj := by
  rw [div_le_div_iff_of_pos_right hS, Real.exp_le_exp]

/-- **THE BLOCKER, RESOLVED.**  A second-layer pre-activation that a real-valued engine can
    only bound as `[−ε, +ε]` (straddling the breakpoint, hence unstable forever) is on `2ℤ`,
    so ANY bound with `|·| < 2` pins it to exactly `0` — and `0` is decided, by
    `hs_stable_int_complete`, to `+1` with no boundary hazard. -/
theorem straddling_bound_pins_zero {v l u : ℝ} (hv : OnLattice 2 v)
    (hl : l ≤ v) (hu : v ≤ u) (hlow : -2 < l) (hhigh : u < 2) : v = 0 ∧ hs v = 1 := by
  have h1 : 0 ≤ v := lattice_lower_forces_nonneg (by norm_num) hv hl (by exact_mod_cast hlow)
  have h2 : v ≤ 0 := lattice_upper_forces_nonpos (by norm_num) hv hu (by exact_mod_cast hhigh)
  have hz : v = 0 := le_antisymm h2 h1
  exact ⟨hz, by rw [hz]; exact hs_nonneg (le_refl (0 : ℝ))⟩

end IntegerInterval

end Crownproof
