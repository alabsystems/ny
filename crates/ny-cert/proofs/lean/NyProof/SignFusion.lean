/-
  SignFusion.lean — the two lemmas underpinning the fused-`Sign` (binarized-net) route.

  Design context: `docs/SIGN_COMPOSITE_FUSION_DESIGN_2026-07-27.md`.  The traffic /
  BNN nets encode each activation as ONNX `Sign → Add(c) → Sign` with `c = 0.1`.
  ONNX `Sign` is THREE-valued (`sgn 0 = 0`), which is why ny models it faithfully today
  (`layers/misc/piecewise_constant.rs`: `l > 0 ⇒ 1`, `u < 0 ⇒ −1`, both STRICT) and why
  branch-at-0 on a raw `Sign` is not a decision procedure.  This file proves:

  * **(L1) Sign-pair fusion.** For every real `z` and every `0 < c < 1`,
    `sgn (sgn z + c) = hs z` where `hs` is the TWO-valued step (`hs 0 = +1`);
    and the mirror, for `−1 < c < 0`, `sgn (sgn z + c) = hs' z` (`hs' 0 = −1`).
    Plus the GUARDS: the rewrite is FALSE for `c = 0`, for `1 ≤ c`, and for `c ≤ −1`
    — exactly the `0 < |c| < 1` side condition the build-time pass must check — and
    a rounding-robust version showing the fusion survives any sign-faithful float
    arithmetic (so the f32 `Add` cannot break it).  Caveat, worth stating because ny has
    been bitten by it before (the ConvTranspose DAZ fail-open): under FLUSH-TO-ZERO /
    DENORMALS-ARE-ZERO arithmetic a SUBNORMAL `c` is flushed to `0` and the fusion becomes
    exactly the refuted `c = 0` case, so the build-time guard should demand a normal `c`,
    not merely `0 < |c| < 1`.  For the shipped `c = 0.1` this is free.

  * **(L2) Value-branch coverage.** The condition under which splitting a two-valued
    node into the two CLOSED children `{z ≥ 0} ↦ +1` and `{z ≤ 0} ↦ −1` is sound for
    BaB even though the second child's fixed value is WRONG at `z = 0`.

  # L2 — the honest hypothesis list (READ THIS)

  The informal argument is: (i) the regions cover, (ii) child A's fixed value is correct
  on all of `{z ≥ 0}`, (iii) BaB reports "holds" only if every child reports "holds".
  **Those three clauses are NOT sufficient** — `three_clause_split_unsound` below is a
  machine-checked counterexample: they permit child B's fixed value to be wrong
  *everywhere*, not just at `z = 0`, and then both children can report "holds" while the
  property fails.  The missing clause is the one the informal proof silently uses:

      (iv) child B's fixed value is correct on `B_region \ A_region`
           (equivalently: the sets on which each child's fixed value is CORRECT — not
            merely the regions — must cover).

  With (iv) the argument goes through; that is `branch_sound_two`.  The general form is
  `branch_sound_of_agreement_cover`: a family of children, each carrying a `region` (what
  its verdict speaks about) and an `agree` set (where its value-fixed system agrees with
  the true one), is sound as soon as `agree ⊆ region` per child and the `agree` sets cover.

  For the fused node (iv) is discharged automatically and for EITHER polarity, because at
  the split point the true node takes one of the two fixed values, so at least one child
  is correct there: `twoValued_closed_split_sound`.  For the RAW three-valued ONNX `Sign`
  it is NOT discharged — `sgn 0 = 0` is neither child's fixed value — and
  `sgn_closed_split_unsound` machine-checks that the same split really is unsound there.
  That is precisely the defect fusion removes.

  Three further hypotheses fall out of the proof and are implementation obligations:

  * **The split point must be the node's breakpoint.** `offcenter_closed_split_unsound`:
    value-fixing children at any `p < 0` is unsound.  (Input-box splits at arbitrary
    points are unaffected — they fix no node value.)
  * **The breakpoint must belong to the child whose fixed value matches `hs 0`.**
    `plus_child_must_contain_breakpoint`: with `hs 0 = +1`, separating the halves as
    `[l, 0]` / `[next(0), u]` — or otherwise opening the `+1` child at `0` — is unsound.
    Closed/closed `[l, 0]` / `[0, u]` is the safe shape.  Mirrored for `hs'`.
  * **A stability shortcut is NOT a branch and gets no sibling.** `hs` is constant `−1`
    only on the OPEN `{z < 0}` and constant `+1` on the CLOSED `{0 ≤ z}`, so for a fused
    node with `hs 0 = +1` the sound stability tests are `u < 0 ⇒ −1` (STRICT) and
    `0 ≤ l ⇒ +1`; `u ≤ 0 ⇒ −1` is unsound at `u = 0`
    (`hs_fix_neg_on_closed_unsound`).  Mirrored for `hs'`.  Note that
    `beta_crown/nonlinear_branching/scoring.rs:44` uses the SYMMETRIC test
    `l >= 0.0 || u <= 0.0` to mark a unit stable; that test is fine as a branching
    *skip* only if the value assigned when it fires respects the strict threshold above.

  Depth: `FrontierSound` + `frontierSound_root` / `frontierSound_refine` /
  `frontierSound_verdict` carry the argument to arbitrary BaB depth (the invariant is
  "the frontier's AGREE sets cover the input space", and each refinement must cover its
  parent's AGREE set, not merely its region), and `frontierSound_refine_sign` discharges
  the refinement obligations for a fused-sign split at any leaf.

  Scope: everything here is about the SAFETY direction ("property holds").  A
  counterexample produced inside child B is not licensed by these theorems — B's system
  differs from the true one at `z = 0` — so falsification witnesses must still be
  replayed on the unfused network (ny already validates SAT witnesses independently).
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace Crownproof

namespace SignFusion

/-! ## §1  The three sign functions, defined explicitly

`sgn` is ONNX's THREE-valued `Sign`; `hs`/`hs'` are the two TWO-valued steps the fused
node can compute.  Nothing is imported from a library sign function: the ONNX convention
`sgn 0 = 0` is the whole point of L1 and must not be assumed away. -/

/-- ONNX `Sign`: three-valued, `sgn 0 = 0`. -/
noncomputable def sgn (z : ℝ) : ℝ := if z < 0 then -1 else if z = 0 then 0 else 1

/-- The two-valued step with `hs 0 = +1` (the fusion target for `0 < c < 1`). -/
noncomputable def hs (z : ℝ) : ℝ := if z < 0 then -1 else 1

/-- The two-valued step with `hs' 0 = −1` (the fusion target for `−1 < c < 0`). -/
noncomputable def hs' (z : ℝ) : ℝ := if 0 < z then 1 else -1

theorem sgn_neg {z : ℝ} (h : z < 0) : sgn z = -1 := by
  unfold sgn; rw [if_pos h]

theorem sgn_zero : sgn (0 : ℝ) = 0 := by
  unfold sgn; rw [if_neg (lt_irrefl (0 : ℝ)), if_pos rfl]

theorem sgn_pos {z : ℝ} (h : 0 < z) : sgn z = 1 := by
  unfold sgn; rw [if_neg (not_lt.mpr h.le), if_neg h.ne']

/-- ONNX's `Sign` is three-valued — the fact `hs`/`hs'` must absorb. -/
theorem sgn_three_valued (z : ℝ) : sgn z = -1 ∨ sgn z = 0 ∨ sgn z = 1 := by
  rcases lt_trichotomy z 0 with h | h | h
  · exact Or.inl (sgn_neg h)
  · subst h; exact Or.inr (Or.inl sgn_zero)
  · exact Or.inr (Or.inr (sgn_pos h))

theorem hs_neg {z : ℝ} (h : z < 0) : hs z = -1 := by
  unfold hs; rw [if_pos h]

theorem hs_nonneg {z : ℝ} (h : 0 ≤ z) : hs z = 1 := by
  unfold hs; rw [if_neg (not_lt.mpr h)]

theorem hs_two_valued (z : ℝ) : hs z = -1 ∨ hs z = 1 := by
  rcases lt_trichotomy z 0 with h | h | h
  · exact Or.inl (hs_neg h)
  · subst h; exact Or.inr (hs_nonneg (le_refl (0 : ℝ)))
  · exact Or.inr (hs_nonneg h.le)

theorem hs'_pos {z : ℝ} (h : 0 < z) : hs' z = 1 := by
  unfold hs'; rw [if_pos h]

theorem hs'_nonpos {z : ℝ} (h : z ≤ 0) : hs' z = -1 := by
  unfold hs'; rw [if_neg (not_lt.mpr h)]

theorem hs'_two_valued (z : ℝ) : hs' z = -1 ∨ hs' z = 1 := by
  rcases lt_trichotomy z 0 with h | h | h
  · exact Or.inl (hs'_nonpos h.le)
  · subst h; exact Or.inl (hs'_nonpos (le_refl (0 : ℝ)))
  · exact Or.inr (hs'_pos h)

/-! ## §2  L1 — the Sign-pair fusion lemma -/

/-- **L1.**  `Sign → Add(c) → Sign` with `0 < c < 1` computes the TWO-valued step
    `hs` (`hs 0 = +1`).  The ONNX `sgn 0 = 0` case is absorbed by the `+c`. -/
theorem sign_pair_fusion {c : ℝ} (hc0 : 0 < c) (hc1 : c < 1) (z : ℝ) :
    sgn (sgn z + c) = hs z := by
  rcases lt_trichotomy z 0 with h | h | h
  · rw [sgn_neg h, hs_neg h]
    exact sgn_neg (by linarith)
  · subst h
    rw [sgn_zero, hs_nonneg (le_refl (0 : ℝ)), zero_add]
    exact sgn_pos hc0
  · rw [sgn_pos h, hs_nonneg h.le]
    exact sgn_pos (by linarith)

/-- **L1, mirror polarity.**  With `−1 < c < 0` the pair computes `hs'` (`hs' 0 = −1`). -/
theorem sign_pair_fusion_neg {c : ℝ} (hc0 : -1 < c) (hc1 : c < 0) (z : ℝ) :
    sgn (sgn z + c) = hs' z := by
  rcases lt_trichotomy z 0 with h | h | h
  · rw [sgn_neg h, hs'_nonpos h.le]
    exact sgn_neg (by linarith)
  · subst h
    rw [sgn_zero, hs'_nonpos (le_refl (0 : ℝ)), zero_add]
    exact sgn_neg hc1
  · rw [sgn_pos h, hs'_pos h]
    exact sgn_pos (by linarith)

/-- **L1 is a graph rewrite.**  Whatever the rest of the network `N` does with the
    activation, replacing the pair by the fused node changes nothing. -/
theorem sign_pair_fusion_rewrite {Out : Type*} (N : ℝ → ℝ → Out)
    {c : ℝ} (hc0 : 0 < c) (hc1 : c < 1) (z : ℝ) :
    N (sgn (sgn z + c)) z = N (hs z) z := by
  rw [sign_pair_fusion hc0 hc1]

/-- **L1 under rounding.**  The fusion needs no exact arithmetic for the interior `Add`:
    ANY sign-faithful rounding `ρ` (one that maps positives to positives and negatives to
    negatives) leaves the identity intact.  Round-to-nearest f32 is sign-faithful, and the
    three exact intermediate values `c−1, c, c+1` are bounded away from `0` by `min c (1−c)`
    — for the shipped `c = 0.1` that is `0.1`, some 10^36 ULPs from the subnormal range —
    so no f32 `Add` can flip the outer `Sign`. -/
theorem sign_pair_fusion_of_signFaithful (ρ : ℝ → ℝ)
    (hpos : ∀ x : ℝ, 0 < x → 0 < ρ x) (hneg : ∀ x : ℝ, x < 0 → ρ x < 0)
    {c : ℝ} (hc0 : 0 < c) (hc1 : c < 1) (z : ℝ) :
    sgn (ρ (sgn z + c)) = hs z := by
  rcases lt_trichotomy z 0 with h | h | h
  · rw [sgn_neg h, hs_neg h]
    exact sgn_neg (hneg _ (by linarith))
  · subst h
    rw [sgn_zero, hs_nonneg (le_refl (0 : ℝ)), zero_add]
    exact sgn_pos (hpos _ hc0)
  · rw [sgn_pos h, hs_nonneg h.le]
    exact sgn_pos (hpos _ (by linarith))

/-! ### Guards: the side condition `0 < |c| < 1` is exactly right -/

/-- Guard.  For `1 ≤ c` the rewrite is FALSE (witness `z = −1`). -/
theorem sign_pair_fusion_fails_of_one_le {c : ℝ} (hc : 1 ≤ c) :
    sgn (sgn (-1 : ℝ) + c) ≠ hs (-1 : ℝ) := by
  rw [sgn_neg (by norm_num : (-1 : ℝ) < 0), hs_neg (by norm_num : (-1 : ℝ) < 0)]
  unfold sgn
  rw [if_neg (by linarith : ¬ ((-1 : ℝ) + c < 0))]
  split_ifs <;> norm_num

/-- Guard.  For `c = 0` the rewrite is FALSE (witness `z = 0`): with no offset the
    three-valued `sgn 0 = 0` survives. -/
theorem sign_pair_fusion_fails_of_zero_const :
    sgn (sgn (0 : ℝ) + 0) ≠ hs (0 : ℝ) := by
  have h1 : sgn (sgn (0 : ℝ) + 0) = 0 := by rw [sgn_zero, add_zero, sgn_zero]
  have h2 : hs (0 : ℝ) = 1 := hs_nonneg (le_refl (0 : ℝ))
  rw [h1, h2]; norm_num

/-- Guard.  For `c ≤ −1` the mirror rewrite is FALSE (witness `z = 1`). -/
theorem sign_pair_fusion_fails_of_le_neg_one {c : ℝ} (hc : c ≤ -1) :
    sgn (sgn (1 : ℝ) + c) ≠ hs' (1 : ℝ) := by
  have h1 : sgn (1 : ℝ) = 1 := sgn_pos (by norm_num)
  have h2 : hs' (1 : ℝ) = 1 := hs'_pos (by norm_num)
  have h3 : sgn ((1 : ℝ) + c) ≤ 0 := by
    rcases lt_trichotomy ((1 : ℝ) + c) 0 with h | h | h
    · rw [sgn_neg h]; norm_num
    · rw [h, sgn_zero]
    · exfalso; linarith
  rw [h1, h2]
  intro hcon
  rw [hcon] at h3
  norm_num at h3

/-! ## §3  L2 — value-branch coverage

A BaB child is modelled by three data: the `region` its verdict speaks about, the set
`agree` on which its value-FIXED system coincides with the true system, and the fixed
system itself.  For a `hs` node split at `0`, the `+1` child has `region = agree = {z ≥ 0}`
while the `−1` child has `region = {z ≤ 0}` but only `agree = {z < 0}`. -/

section ValueBranch

variable {Z Out : Type*}

/-- **L2, general form.**  If every child's verdict is sound on its own region, every
    child's fixed system agrees with the true system on its `agree` set, each `agree` set
    lies inside its own region, and the `agree` sets COVER the input space, then the
    conjunction of the children's verdicts soundly implies the property. -/
theorem branch_sound_of_agreement_cover {ι : Type*}
    (F : Z → Out) (P : Out → Prop)
    (R A : ι → Z → Prop) (G : ι → Z → Out)
    (hwf : ∀ i z, A i z → R i z)
    (hfaithful : ∀ i z, A i z → G i z = F z)
    (hcover : ∀ z, ∃ i, A i z)
    (hverdict : ∀ i z, R i z → P (G i z)) :
    ∀ z, P (F z) := by
  intro z
  obtain ⟨i, hz⟩ := hcover z
  have h := hverdict i z (hwf i z hz)
  rwa [hfaithful i z hz] at h

/-- **L2 in the shape the implementation assumes**, with the missing clause made
    explicit.  Clauses (i) `hcover`, (ii) `hA`, (iii) `hdA ∧ hdB` are the three the
    informal argument lists; `hB` — *child B's fixed value is correct wherever child A's
    region does not reach* — is the FOURTH, and it is indispensable (see
    `three_clause_split_unsound`). -/
theorem branch_sound_two
    (F GA GB : Z → Out) (P : Out → Prop) (RA RB : Z → Prop)
    (hcover : ∀ z, RA z ∨ RB z)
    (hA : ∀ z, RA z → GA z = F z)
    (hB : ∀ z, RB z → ¬ RA z → GB z = F z)
    (hdA : ∀ z, RA z → P (GA z))
    (hdB : ∀ z, RB z → P (GB z)) :
    ∀ z, P (F z) := by
  intro z
  by_cases hz : RA z
  · have h := hdA z hz
    rwa [hA z hz] at h
  · rcases hcover z with h | h
    · exact absurd h hz
    · have h' := hdB z h
      rwa [hB z h hz] at h'

/-- **REFUTATION of the informal three-clause argument.**  Regions cover, child A's fixed
    value is correct on all of `A_region`, and BOTH children report "holds" — yet the
    property FAILS.  (`F = id`, `P y := 0 ≤ y`, `A_region = {z ≥ 0}` with `GA = F`,
    `B_region = {z ≤ 0}` with `GB ≡ 0`; the verdict is wrong at `z = −1`.)  Nothing in
    clauses (i)–(iii) constrains child B's fixed value, so it may be wrong everywhere,
    not only at the shared boundary. -/
theorem three_clause_split_unsound :
    ∃ (F GA GB : ℝ → ℝ) (P : ℝ → Prop) (RA RB : ℝ → Prop),
      (∀ z, RA z ∨ RB z) ∧
      (∀ z, RA z → GA z = F z) ∧
      (∀ z, RA z → P (GA z)) ∧
      (∀ z, RB z → P (GB z)) ∧
      ¬ (∀ z, P (F z)) := by
  refine ⟨fun z => z, fun z => z, fun _ => 0, fun y => 0 ≤ y,
          fun z => 0 ≤ z, fun z => z ≤ 0, ?_, ?_, ?_, ?_, ?_⟩
  · intro z; exact le_total 0 z
  · intro z _; rfl
  · intro z hz; exact hz
  · intro z _; exact le_refl (0 : ℝ)
  · intro h
    have h0 : (0 : ℝ) ≤ -1 := h (-1)
    norm_num at h0

end ValueBranch

/-! ### The fused node discharges the fourth clause — for either polarity -/

/-- **The punchline for L2.**  For ANY two-valued node `g` (value `vneg` strictly below
    the breakpoint, `vpos` strictly above, and *anything in `{vneg, vpos}`* at the
    breakpoint), the CLOSED/CLOSED split with fixed values `vpos` / `vneg` is sound.  The
    boundary point is always covered by whichever child happens to be right there, and
    since `g 0` is one of the two fixed values, one always is. -/
theorem twoValued_closed_split_sound {V Out : Type*} {vneg vpos : V}
    (g : ℝ → V)
    (hneg : ∀ z : ℝ, z < 0 → g z = vneg)
    (hpos : ∀ z : ℝ, 0 < z → g z = vpos)
    (hzero : g 0 = vneg ∨ g 0 = vpos)
    (N : V → ℝ → Out) (P : Out → Prop)
    (hplus : ∀ z : ℝ, 0 ≤ z → P (N vpos z))
    (hminus : ∀ z : ℝ, z ≤ 0 → P (N vneg z)) :
    ∀ z : ℝ, P (N (g z) z) := by
  intro z
  rcases lt_trichotomy z 0 with h | h | h
  · rw [hneg z h]; exact hminus z h.le
  · subst h
    rcases hzero with h0 | h0
    · rw [h0]; exact hminus 0 (le_refl (0 : ℝ))
    · rw [h0]; exact hplus 0 (le_refl (0 : ℝ))
  · rw [hpos z h]; exact hplus z h.le

/-- **L2, instantiated at the fused node `hs` (`hs 0 = +1`).**  Checking the two closed
    children `{z ≥ 0} ↦ +1` and `{z ≤ 0} ↦ −1` soundly certifies the property, even
    though the `−1` child's fixed value is wrong at `z = 0`. -/
theorem hs_closed_split_sound {Out : Type*} (N : ℝ → ℝ → Out) (P : Out → Prop)
    (hplus : ∀ z : ℝ, 0 ≤ z → P (N 1 z))
    (hminus : ∀ z : ℝ, z ≤ 0 → P (N (-1) z)) :
    ∀ z : ℝ, P (N (hs z) z) :=
  twoValued_closed_split_sound hs (fun _ h => hs_neg h) (fun _ h => hs_nonneg h.le)
    (Or.inr (hs_nonneg (le_refl (0 : ℝ)))) N P hplus hminus

/-- **L2 for the mirror polarity `hs'` (`hs' 0 = −1`).**  The SAME split, with the same
    fixed values, is sound — now it is the `+1` child that is wrong at `0`. -/
theorem hs'_closed_split_sound {Out : Type*} (N : ℝ → ℝ → Out) (P : Out → Prop)
    (hplus : ∀ z : ℝ, 0 ≤ z → P (N 1 z))
    (hminus : ∀ z : ℝ, z ≤ 0 → P (N (-1) z)) :
    ∀ z : ℝ, P (N (hs' z) z) :=
  twoValued_closed_split_sound hs' (fun _ h => hs'_nonpos h.le) (fun _ h => hs'_pos h)
    (Or.inl (hs'_nonpos (le_refl (0 : ℝ)))) N P hplus hminus

/-- **REFUTATION for the UNFUSED node.**  The very same closed/closed split applied to the
    raw THREE-valued ONNX `Sign` is UNSOUND: `sgn 0 = 0` is neither child's fixed value,
    so the boundary point is covered by no child.  (`N v _ := v * v`, `P y := y = 1`: both
    children report `1 = 1`, but the true network outputs `0` at `z = 0`.)  This is
    exactly the completeness/soundness defect that motivates the fusion — branch-at-0 on a
    faithful `Sign` may not fix the value at all. -/
theorem sgn_closed_split_unsound :
    ∃ (N : ℝ → ℝ → ℝ) (P : ℝ → Prop),
      (∀ z : ℝ, 0 ≤ z → P (N 1 z)) ∧
      (∀ z : ℝ, z ≤ 0 → P (N (-1) z)) ∧
      ¬ (∀ z : ℝ, P (N (sgn z) z)) := by
  refine ⟨fun v _ => v * v, fun y => y = 1, ?_, ?_, ?_⟩
  · intro z _; norm_num
  · intro z _; norm_num
  · intro h
    have h0 : sgn 0 * sgn 0 = 1 := h 0
    rw [sgn_zero] at h0
    norm_num at h0

/-- **The split point must be the node's breakpoint.**  Value-fixing children at any
    `p < 0` is unsound: the strip `(p, 0)` is claimed only by the `+1` child, where the
    node really is `−1`.  (Splitting the input BOX at an arbitrary point is unaffected —
    that fixes no node value.) -/
theorem offcenter_closed_split_unsound {p : ℝ} (hp : p < 0) :
    ∃ (N : ℝ → ℝ → ℝ × ℝ) (P : ℝ × ℝ → Prop),
      (∀ z : ℝ, p ≤ z → P (N 1 z)) ∧
      (∀ z : ℝ, z ≤ p → P (N (-1) z)) ∧
      ¬ (∀ z : ℝ, P (N (hs z) z)) := by
  refine ⟨fun v z => (v, z), fun y => y.1 = 1 ∨ y.2 ≤ p, ?_, ?_, ?_⟩
  · intro z _; exact Or.inl rfl
  · intro z hz; exact Or.inr hz
  · intro h
    have h0 : hs (p / 2) = 1 ∨ p / 2 ≤ p := h (p / 2)
    rw [hs_neg (by linarith : p / 2 < 0)] at h0
    rcases h0 with h1 | h1
    · norm_num at h1
    · linarith

/-- **The breakpoint must be given to the `+1` child.**  Both children may be closed, or
    the `−1` child may be half-open — but the child whose fixed value matches `hs 0 = +1`
    MUST contain `z = 0`.  An engine that separates the halves as `[l, 0]` / `[next(0), u]`
    (or that opens the `+1` child) is unsound at the breakpoint.  Mirrored for `hs'`, where
    the breakpoint must go to the `−1` child. -/
theorem plus_child_must_contain_breakpoint :
    ∃ (N : ℝ → ℝ → ℝ × ℝ) (P : ℝ × ℝ → Prop),
      (∀ z : ℝ, 0 < z → P (N 1 z)) ∧
      (∀ z : ℝ, z ≤ 0 → P (N (-1) z)) ∧
      ¬ (∀ z : ℝ, P (N (hs z) z)) := by
  refine ⟨fun v z => (v, z), fun y => y.1 = -1 ∨ 0 < y.2, ?_, ?_, ?_⟩
  · intro z hz; exact Or.inr hz
  · intro z _; exact Or.inl rfl
  · intro h
    have h0 : hs 0 = -1 ∨ (0 : ℝ) < 0 := h 0
    rw [hs_nonneg (le_refl (0 : ℝ))] at h0
    rcases h0 with h1 | h1
    · norm_num at h1
    · exact absurd h1 (lt_irrefl (0 : ℝ))

/-! ### Stability shortcuts are NOT branches: the thresholds must be asymmetric

A "stable unit" shortcut fixes a node's value from its interval bounds WITHOUT creating a
sibling, so the coverage argument above does not apply and the fixed value must be correct
on the whole region.  For `hs` that means the negative test must be STRICT. -/

/-- The `+1` side needs no strictness: `hs` really is constant `+1` on the CLOSED `{0 ≤ z}`,
    so `0 ≤ l ⇒ +1` is sound. -/
theorem hs_fix_pos_on_closed_sound {Out : Type*} (N : ℝ → ℝ → Out) (P : Out → Prop)
    (h : ∀ z : ℝ, 0 ≤ z → P (N 1 z)) :
    ∀ z : ℝ, 0 ≤ z → P (N (hs z) z) := by
  intro z hz; rw [hs_nonneg hz]; exact h z hz

/-- The `−1` side is sound only on the OPEN `{z < 0}`, i.e. only under `u < 0`. -/
theorem hs_fix_neg_on_open_sound {Out : Type*} (N : ℝ → ℝ → Out) (P : Out → Prop)
    (h : ∀ z : ℝ, z < 0 → P (N (-1) z)) :
    ∀ z : ℝ, z < 0 → P (N (hs z) z) := by
  intro z hz; rw [hs_neg hz]; exact h z hz

/-- **REFUTATION.**  Fixing the fused node to `−1` on the CLOSED `{z ≤ 0}` with no sibling
    — i.e. a stability test `u ≤ 0 ⇒ −1` firing at `u = 0` — is UNSOUND, because
    `hs 0 = +1`.  ny's ReLU/Sign stability predicate is the symmetric
    `l >= 0.0 || u <= 0.0`; reused verbatim for a fused node it admits `u = 0`. -/
theorem hs_fix_neg_on_closed_unsound :
    ∃ (N : ℝ → ℝ → ℝ) (P : ℝ → Prop),
      (∀ z : ℝ, z ≤ 0 → P (N (-1) z)) ∧
      ¬ (∀ z : ℝ, z ≤ 0 → P (N (hs z) z)) := by
  refine ⟨fun v _ => v, fun y => y = -1, ?_, ?_⟩
  · intro z _; rfl
  · intro h
    have h0 : hs 0 = -1 := h 0 (le_refl (0 : ℝ))
    rw [hs_nonneg (le_refl (0 : ℝ))] at h0
    norm_num at h0

/-- Mirror: for `hs'` (`hs' 0 = −1`) the strictness moves to the `+1` side. -/
theorem hs'_fix_neg_on_closed_sound {Out : Type*} (N : ℝ → ℝ → Out) (P : Out → Prop)
    (h : ∀ z : ℝ, z ≤ 0 → P (N (-1) z)) :
    ∀ z : ℝ, z ≤ 0 → P (N (hs' z) z) := by
  intro z hz; rw [hs'_nonpos hz]; exact h z hz

/-- Mirror refutation: `0 ≤ l ⇒ +1` is unsound for `hs'` at `l = 0`. -/
theorem hs'_fix_pos_on_closed_unsound :
    ∃ (N : ℝ → ℝ → ℝ) (P : ℝ → Prop),
      (∀ z : ℝ, 0 ≤ z → P (N 1 z)) ∧
      ¬ (∀ z : ℝ, 0 ≤ z → P (N (hs' z) z)) := by
  refine ⟨fun v _ => v, fun y => y = 1, ?_, ?_⟩
  · intro z _; rfl
  · intro h
    have h0 : hs' 0 = 1 := h 0 (le_refl (0 : ℝ))
    rw [hs'_nonpos (le_refl (0 : ℝ))] at h0
    norm_num at h0

/-! ## §4  Arbitrary depth — the BaB frontier invariant

One split is not a proof about BaB; a search tree is.  The invariant that must be
maintained is that the frontier's AGREE sets cover the input space (and each child's agree
set lies inside its parent's agree set, NOT merely inside its parent's region). -/

section Frontier

variable {Z Out : Type*}

/-- The BaB frontier invariant: agree ⊆ region per leaf, each leaf's value-fixed system
    agrees with the true system on its agree set, and the agree sets cover. -/
def FrontierSound {ι : Type*} (F : Z → Out) (R A : ι → Z → Prop) (G : ι → Z → Out) : Prop :=
  (∀ i z, A i z → R i z) ∧ (∀ i z, A i z → G i z = F z) ∧ (∀ z, ∃ i, A i z)

/-- The root frontier (one leaf, the whole space, the unmodified network) is sound. -/
theorem frontierSound_root (F : Z → Out) :
    FrontierSound (ι := Unit) F (fun _ _ => True) (fun _ _ => True) (fun _ => F) :=
  ⟨fun _ _ _ => trivial, fun _ _ _ => rfl, fun _ => ⟨(), trivial⟩⟩

/-- A sound frontier all of whose leaves report "holds" certifies the property. -/
theorem frontierSound_verdict {ι : Type*} {F : Z → Out} {R A : ι → Z → Prop}
    {G : ι → Z → Out} (h : FrontierSound F R A G) (P : Out → Prop)
    (hverdict : ∀ i z, R i z → P (G i z)) : ∀ z, P (F z) :=
  branch_sound_of_agreement_cover F P R A G h.1 h.2.1 h.2.2 hverdict

/-- **Refinement preserves the invariant** — hence, by iteration, BaB is sound at any
    depth.  Note the two hypotheses that are easy to get wrong: a child's agree set must
    sit inside its PARENT'S AGREE set (`hsub`), and the children's agree sets must cover
    the parent's AGREE set (`hcover`) — the parent's *region* is not enough.  Leaves that
    are not being split are handled by taking their children to be copies of themselves. -/
theorem frontierSound_refine {ι κ : Type*}
    {F : Z → Out} {R A : ι → Z → Prop} {G : ι → Z → Out}
    (h : FrontierSound F R A G)
    (R' A' : ι → κ → Z → Prop) (G' : ι → κ → Z → Out)
    (hwf : ∀ i k z, A' i k z → R' i k z)
    (hsub : ∀ i k z, A' i k z → A i z)
    (hfaithful : ∀ i k z, A' i k z → G' i k z = G i z)
    (hcover : ∀ i z, A i z → ∃ k, A' i k z) :
    FrontierSound F (fun p : ι × κ => R' p.1 p.2) (fun p : ι × κ => A' p.1 p.2)
      (fun p : ι × κ => G' p.1 p.2) := by
  refine ⟨fun p z hz => hwf p.1 p.2 z hz, ?_, ?_⟩
  · intro p z hz
    show G' p.1 p.2 z = F z
    rw [hfaithful p.1 p.2 z hz]
    exact h.2.1 p.1 z (hsub p.1 p.2 z hz)
  · intro z
    obtain ⟨i, hi⟩ := h.2.2 z
    obtain ⟨k, hk⟩ := hcover i z hi
    exact ⟨(i, k), hk⟩

end Frontier

/-! ### The fused-sign split, as a frontier refinement -/

section SignSplit

variable {Z Out : Type*}

/-- The two closed children of a fused-sign split: `true` is `{ζ ≥ 0}`, `false` is
    `{ζ ≤ 0}`.  Both are CLOSED — a float engine cannot represent an open half. -/
def signRegion (R : Z → Prop) (ζ : Z → ℝ) : Bool → Z → Prop
  | true,  z => R z ∧ 0 ≤ ζ z
  | false, z => R z ∧ ζ z ≤ 0

/-- Where each child's FIXED value is correct: `{ζ ≥ 0}` for `+1`, but only the OPEN
    `{ζ < 0}` for `−1`.  This is strictly smaller than the `false` child's region. -/
def signAgree (A : Z → Prop) (ζ : Z → ℝ) : Bool → Z → Prop
  | true,  z => A z ∧ 0 ≤ ζ z
  | false, z => A z ∧ ζ z < 0

/-- The child systems: the node's output is replaced by the constant `+1` / `−1`. -/
def signFixed (N : ℝ → Z → Out) : Bool → Z → Out
  | true,  z => N 1 z
  | false, z => N (-1) z

theorem signAgree_imp_region {R A : Z → Prop} {ζ : Z → ℝ}
    (hRA : ∀ z, A z → R z) (b : Bool) (z : Z) (h : signAgree A ζ b z) :
    signRegion R ζ b z := by
  cases b with
  | true => exact ⟨hRA z h.1, h.2⟩
  | false => exact ⟨hRA z h.1, le_of_lt h.2⟩

theorem signAgree_imp_parent {A : Z → Prop} {ζ : Z → ℝ} (b : Bool) (z : Z)
    (h : signAgree A ζ b z) : A z := by
  cases b with
  | true => exact h.1
  | false => exact h.1

theorem signFixed_faithful {A : Z → Prop} (N : ℝ → Z → Out) (ζ : Z → ℝ) (b : Bool) (z : Z)
    (h : signAgree A ζ b z) : signFixed N b z = N (hs (ζ z)) z := by
  cases b with
  | true =>
      show N 1 z = N (hs (ζ z)) z
      rw [hs_nonneg h.2]
  | false =>
      show N (-1) z = N (hs (ζ z)) z
      rw [hs_neg h.2]

/-- The children's agree sets cover the parent's AGREE set — the clause that makes the
    boundary point `ζ z = 0` safe (it lands in the `+1` child, where `hs` really is `+1`). -/
theorem signAgree_cover {A : Z → Prop} (ζ : Z → ℝ) (z : Z) (h : A z) :
    ∃ b, signAgree A ζ b z := by
  rcases lt_trichotomy (ζ z) 0 with hz | hz | hz
  · exact ⟨false, h, hz⟩
  · exact ⟨true, h, le_of_eq hz.symm⟩
  · exact ⟨true, h, le_of_lt hz⟩

/-- **A fused-sign split at every leaf preserves the frontier invariant.**  `hnode` is the
    modelling hypothesis: the leaf's system routes through a fused two-valued node with
    pre-activation `ζ i`.  Combined with `frontierSound_root`, `frontierSound_refine` and
    `frontierSound_verdict`, this is the full BaB soundness statement for the Sign-BaB
    path at arbitrary depth. -/
theorem frontierSound_refine_sign {ι : Type*}
    {F : Z → Out} {R A : ι → Z → Prop} {G : ι → Z → Out}
    (h : FrontierSound F R A G)
    (N : ι → ℝ → Z → Out) (ζ : ι → Z → ℝ)
    (hnode : ∀ i z, G i z = N i (hs (ζ i z)) z) :
    FrontierSound F
      (fun p : ι × Bool => signRegion (R p.1) (ζ p.1) p.2)
      (fun p : ι × Bool => signAgree (A p.1) (ζ p.1) p.2)
      (fun p : ι × Bool => signFixed (N p.1) p.2) :=
  frontierSound_refine h
    (fun i b z => signRegion (R i) (ζ i) b z)
    (fun i b z => signAgree (A i) (ζ i) b z)
    (fun i b z => signFixed (N i) b z)
    (fun i b z hz => signAgree_imp_region (h.1 i) b z hz)
    (fun i b z hz => signAgree_imp_parent b z hz)
    (fun i b z hz => by
      show signFixed (N i) b z = G i z
      rw [signFixed_faithful (N i) (ζ i) b z hz, hnode i z])
    (fun i z hz => signAgree_cover (ζ i) z hz)

end SignSplit

end SignFusion

end Crownproof
