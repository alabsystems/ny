/-
  AristotleLemmas.lean — lemmas proven by the Aristotle autonomous prover,
  vendored VERBATIM (proof bodies unaltered) and re-verified locally: built
  against THIS project's pinned Lean/Mathlib (v4.30.0), grep-checked sorry-free,
  and axiom-audited via AxiomAudit.lean.  Provenance per theorem below.

  Discipline note (why the paranoia): Aristotle downloads taken mid-run can be
  `sorry` stubs — only a post-completion download carries the proof.  Never mark
  an Aristotle result proved without a local sorry-grep + `#print axioms`.

  1. `analytic_zeros_finite`:
     a real-analytic function on an open set containing `[a,b]`, not identically
     zero there, has FINITELY many zeros in `[a,b]`.  This is the mathematical
     core discharging the MeanValueForm/MeanValueChain "finitely many
     breakpoints" hypothesis: each ReLU pre-activation along the MVF segment is
     (per analytic piece) either identically zero — the branch-fixed case,
     `piecewise_multivariate_centered_form_branch_fixed` — or has finitely many
     zeros, so a finite monotone partition of `[0,1]` exists.  (The remaining
     prose step, composing per-node analyticity through the DAG, stays with the
     Rust module docs.)

  2. `pow2_compose_envelope`:
     linear-in-`x` upper/lower envelope bounds for `x⁴` on `[l,u]` by composing
     the quadratic secant/tangent envelopes — the composed-envelope step that
     unblocks the geo_conform torus quartic tier previously refused as
     `NestedSquareNotYetCertifiable`.  Key SOS
     identities (in the proof): `x⁴ − (t_l+t_u)x² + t_l t_u = (x²−t_l)(x²−t_u) ≤ 0`,
     `(l+u)x − lu − x² = (x−l)(u−x) ≥ 0`, and
     `x⁴ − (4cdx − 2cd² − c²) = (x²−c)² + 2c(x−d)² ≥ 0`.
-/
import Mathlib

set_option maxHeartbeats 8000000 in
/-- **Finitely many zeros of a nonzero analytic function on a compact interval.**
If `f` is real-analytic on an open set `U ⊇ [a,b]` and not identically zero on
`[a,b]`, then `{x ∈ [a,b] | f x = 0}` is finite.  (Hypotheses `a < b` and
`IsOpen U` are kept as stated; the proof does not need them.)  Proof: an infinite
zero set in the compact interval has an accumulation point; the identity theorem
(`AnalyticOnNhd.eqOn_zero_of_preconnected_of_frequently_eq_zero`) then forces
`f ≡ 0` on `[a,b]`, contradiction. -/
theorem Crownproof.analytic_zeros_finite
    (f : ℝ → ℝ) (a b : ℝ) (hab : a < b)
    (U : Set ℝ) (hU : IsOpen U) (hsub : Set.Icc a b ⊆ U)
    (hf : AnalyticOnNhd ℝ f U)
    (hne : ∃ x ∈ Set.Icc a b, f x ≠ 0) :
    {x ∈ Set.Icc a b | f x = 0}.Finite := by
  -- By contradiction, assume the set of zeros is infinite.
  by_contra h_inf;
  -- Apply `Set.Infinite.exists_accPt_of_subset_isCompact` to obtain a point `z₀ ∈ Set.Icc a b` with `AccPt z₀ (Filter.principal S)`.
  obtain ⟨z₀, hz₀⟩ : ∃ z₀ ∈ Set.Icc a b, AccPt z₀ (Filter.principal {x ∈ Set.Icc a b | f x = 0}) := by
    apply_rules [ Set.Infinite.exists_accPt_of_subset_isCompact ];
    · exact CompactIccSpace.isCompact_Icc;
    · exact fun x hx => hx.1;
  -- Since $z₀$ is an accumulation point of the set of zeros of $f$ in $[a, b]$, we have $f(z₀) = 0$ and $f$ has infinitely many zeros in any neighborhood of $z₀$.
  have h_inf_zeros : ∃ᶠ z in nhdsWithin z₀ {z₀}ᶜ, f z = 0 := by
    rw [ accPt_iff_frequently ] at hz₀;
    rw [ Filter.frequently_iff ] at *;
    intro U hU; rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.mp hU with ⟨ V, hV₁, hV₂ ⟩ ; rcases hz₀.2 hV₁ with ⟨ x, hx₁, hx₂, hx₃ ⟩ ; use x; aesop;
  -- Since $f$ is real-analytic on $[a, b]$, by the identity theorem for analytic functions, $f$ must be identically zero on $[a, b]$.
  have h_id : ∀ x ∈ Set.Icc a b, f x = 0 := by
    apply AnalyticOnNhd.eqOn_zero_of_preconnected_of_frequently_eq_zero;
    exacts [ hf.mono hsub, isPreconnected_Icc, hz₀.1, h_inf_zeros ];
  grind

set_option maxHeartbeats 8000000 in
/-- **Composed quartic envelope (`pow2_compose`).**  For `x ∈ [l,u]`, with
`t_u = max l² u²` and `t_l = 0` when the interval straddles `0` (else
`min l² u²`): (a) `x⁴ ≤ (t_l+t_u)((l+u)x − lu) − t_l·t_u` and (b) for every
`c ≥ 0`, `d`: `x⁴ ≥ 4cdx − 2cd² − c²`.  Linear-in-`x` bounds for the nested
square, sound for the torus quartic residual.  (`hlu` kept as stated; implied.) -/
theorem Crownproof.pow2_compose_envelope
    (l u x : ℝ) (hlu : l ≤ u) (hlx : l ≤ x) (hxu : x ≤ u)
    (t_u : ℝ) (htu : t_u = max (l^2) (u^2))
    (t_l : ℝ) (htl : t_l = if l ≤ 0 ∧ 0 ≤ u then 0 else min (l^2) (u^2)) :
    (x^4 ≤ (t_l + t_u) * ((l+u)*x - l*u) - t_l * t_u) ∧
    (∀ c d : ℝ, 0 ≤ c → x^4 ≥ 4*c*d*x - 2*c*d^2 - c^2) := by
  refine ⟨?_, ?_⟩
  · -- upper bound
    have hxsq_le : x^2 ≤ t_u := by
      rw [htu]
      rcases le_total x 0 with hx | hx
      · exact le_trans (by nlinarith) (le_max_left _ _)
      · exact le_trans (by nlinarith) (le_max_right _ _)
    have htl_le : t_l ≤ x^2 := by
      rw [htl]
      split
      · positivity
      · next h =>
        push_neg at h
        rcases lt_or_ge 0 l with hl | hl
        · -- l > 0, whole interval positive, min = l^2 ≤ x^2
          have : min (l^2) (u^2) ≤ l^2 := min_le_left _ _
          nlinarith
        · -- l ≤ 0, so u < 0 from h
          have hu : u < 0 := h hl
          have : min (l^2) (u^2) ≤ u^2 := min_le_right _ _
          nlinarith
    have htl_nonneg : 0 ≤ t_l := by
      rw [htl]; split
      · exact le_refl 0
      · exact le_min (by positivity) (by positivity)
    have htu_nonneg : 0 ≤ t_u := by rw [htu]; positivity
    have h1 : x^4 ≤ (t_l + t_u) * x^2 - t_l * t_u := by
      nlinarith [mul_nonneg (sub_nonneg.mpr htl_le) (sub_nonneg.mpr hxsq_le)]
    have h2 : x^2 ≤ (l+u)*x - l*u := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hlx) (sub_nonneg.mpr hxu)]
    nlinarith [mul_le_mul_of_nonneg_left h2 (by linarith : (0:ℝ) ≤ t_l + t_u)]
  · -- lower bound
    intro c d hc
    nlinarith [sq_nonneg (x^2 - c), mul_nonneg hc (sq_nonneg (x - d))]

/-! ## Batch 2 (2026-07-09)

  3. `ReluPiecewise.relu_analytic_piecewise`:
     for `z` real-analytic on a neighborhood of `[0,1]`, `relu ∘ z`
     is piecewise `z`-or-`0` over a FINITE MONOTONE PARTITION `0 = t 0 ≤ … ≤
     t k = 1` — EXACTLY the partition object
     `piecewise_multivariate_centered_form(_branch_fixed)` consumes.  Together
     with batch 1's `analytic_zeros_finite` this discharges the per-activation
     step of the MVF "finitely many breakpoints" hypothesis; the remaining
     prose is only the DAG induction (each pre-activation along the segment is
     analytic per phase-fixed piece).  Support lemmas (`finite_zeros`,
     `sign_const`, `emb_first/last`) are namespaced to avoid corpus collisions.

  4. `farkas_refutation_sound`:
     nonnegative rational multipliers with zero column sums and negative
     constant sum refute a system of non-strict rational linear inequalities —
     the `la_generic` Alethe-leaf soundness core for checking AY's exported
     Alethe certificates in Clean, in exactly AY's Fin-indexed shape.
-/

namespace Crownproof.ReluPiecewise

/-- If `z` is analytic on a neighborhood of `[0, 1]` and is not identically zero there, then its
zero set inside `[0, 1]` is finite. -/
lemma finite_zeros (z : ℝ → ℝ) (hz : AnalyticOnNhd ℝ z (Set.Icc 0 1))
    (hnz : ¬ Set.EqOn z 0 (Set.Icc 0 1)) :
    (Set.Icc (0:ℝ) 1 ∩ {x | z x = 0}).Finite := by
  by_contra hinf
  rw [Set.not_finite] at hinf
  obtain ⟨x, hx, hacc⟩ := hinf.exists_accPt_of_subset_isCompact isCompact_Icc
    (Set.inter_subset_left)
  apply hnz
  apply hz.eqOn_zero_of_preconnected_of_frequently_eq_zero (convex_Icc 0 1).isPreconnected hx
  rw [accPt_iff_frequently] at hacc
  rw [nhdsWithin, Filter.frequently_inf_principal]
  exact hacc.mono (fun y hy => ⟨hy.1, hy.2.2⟩)

/-- A continuous function without zeros on an open interval has constant sign there. -/
lemma sign_const (f : ℝ → ℝ) (a b : ℝ) (hc : ContinuousOn f (Set.Ioo a b))
    (hnz : ∀ x ∈ Set.Ioo a b, f x ≠ 0) :
    (∀ x ∈ Set.Ioo a b, 0 ≤ f x) ∨ (∀ x ∈ Set.Ioo a b, f x ≤ 0) := by
  by_contra h
  push_neg at h
  obtain ⟨⟨x1, hx1, hx1neg⟩, ⟨x2, hx2, hx2pos⟩⟩ := h
  rcases le_total x1 x2 with hle | hle
  · have hsub : Set.Icc x1 x2 ⊆ Set.Ioo a b := by
      intro y hy; exact ⟨lt_of_lt_of_le hx1.1 hy.1, lt_of_le_of_lt hy.2 hx2.2⟩
    have hcc : ContinuousOn f (Set.Icc x1 x2) := hc.mono hsub
    have hiv := intermediate_value_Icc hle hcc
    have h0 : (0:ℝ) ∈ Set.Icc (f x1) (f x2) := ⟨le_of_lt hx1neg, le_of_lt hx2pos⟩
    obtain ⟨c, hc', hc0⟩ := hiv h0
    exact hnz c (hsub hc') hc0
  · have hsub : Set.Icc x2 x1 ⊆ Set.Ioo a b := by
      intro y hy; exact ⟨lt_of_lt_of_le hx2.1 hy.1, lt_of_le_of_lt hy.2 hx1.2⟩
    have hcc : ContinuousOn f (Set.Icc x2 x1) := hc.mono hsub
    have hiv := intermediate_value_Icc' hle hcc
    have h0 : (0:ℝ) ∈ Set.Icc (f x1) (f x2) := ⟨le_of_lt hx1neg, le_of_lt hx2pos⟩
    obtain ⟨c, hc', hc0⟩ := hiv h0
    exact hnz c (hsub hc') hc0

/-- The first element of the monotone enumeration of a finite set equals its minimum. -/
lemma emb_first (F : Finset ℝ) (n : ℕ) (h : F.card = n) (hn : 0 < n) (m : ℝ)
    (hm : m ∈ F) (hle : ∀ x ∈ F, m ≤ x) : F.orderEmbOfFin h ⟨0, hn⟩ = m := by
  have hmem : F.orderEmbOfFin h ⟨0, hn⟩ ∈ F := F.orderEmbOfFin_mem h _
  have hge : m ≤ F.orderEmbOfFin h ⟨0, hn⟩ := hle _ hmem
  have hr : m ∈ Set.range (F.orderEmbOfFin h) := by rw [Finset.range_orderEmbOfFin]; exact hm
  obtain ⟨i, hi⟩ := hr
  have hle2 : F.orderEmbOfFin h ⟨0, hn⟩ ≤ F.orderEmbOfFin h i :=
    (F.orderEmbOfFin h).monotone (by simp [Fin.le_def])
  rw [hi] at hle2
  exact le_antisymm hle2 hge

/-- The last element of the monotone enumeration of a finite set equals its maximum. -/
lemma emb_last (F : Finset ℝ) (n : ℕ) (h : F.card = n) (hn : 0 < n) (M : ℝ)
    (hM : M ∈ F) (hge : ∀ x ∈ F, x ≤ M) : F.orderEmbOfFin h ⟨n-1, by omega⟩ = M := by
  have hmem : F.orderEmbOfFin h ⟨n-1, by omega⟩ ∈ F := F.orderEmbOfFin_mem h _
  have hle : F.orderEmbOfFin h ⟨n-1, by omega⟩ ≤ M := hge _ hmem
  have hr : M ∈ Set.range (F.orderEmbOfFin h) := by rw [Finset.range_orderEmbOfFin]; exact hM
  obtain ⟨i, hi⟩ := hr
  have hle2 : F.orderEmbOfFin h i ≤ F.orderEmbOfFin h ⟨n-1, by omega⟩ :=
    (F.orderEmbOfFin h).monotone (by simp only [Fin.le_def]; omega)
  rw [hi] at hle2
  exact le_antisymm hle hle2

set_option maxHeartbeats 8000000 in
/-- **Piecewise structure of `relu ∘ z` for a real-analytic `z`** — the finite
monotone partition consumed by `piecewise_multivariate_centered_form`. -/
theorem relu_analytic_piecewise (z : ℝ → ℝ)
    (hz : AnalyticOnNhd ℝ z (Set.Icc 0 1)) :
    ∃ (k : ℕ) (t : ℕ → ℝ),
      Monotone t ∧ t 0 = 0 ∧ t k = 1 ∧
      ∀ j < k, (∀ x ∈ Set.Ioo (t j) (t (j+1)), max (z x) 0 = z x) ∨
               (∀ x ∈ Set.Ioo (t j) (t (j+1)), max (z x) 0 = 0) := by
  by_cases hnz : Set.EqOn z 0 (Set.Icc 0 1)
  · -- `z` is identically zero on `[0, 1]`: a single piece works.
    refine ⟨1, fun j => min (j:ℝ) 1, ?_, ?_, ?_, ?_⟩
    · intro a b hab
      exact min_le_min (by exact_mod_cast Nat.cast_le.mpr hab) le_rfl
    · norm_num
    · norm_num
    · intro j hj
      interval_cases j
      left
      intro x hx
      simp only [Nat.cast_zero, Nat.reduceAdd, Nat.cast_one] at hx
      rw [min_eq_left (by norm_num : (0:ℝ) ≤ 1), min_self] at hx
      have hx01 : x ∈ Set.Icc (0:ℝ) 1 := ⟨le_of_lt hx.1, le_of_lt hx.2⟩
      rw [hnz hx01]; simp
  · -- `z` has finitely many zeros; use them (with `0` and `1`) as partition points.
    have hfin := finite_zeros z hz hnz
    set F : Finset ℝ := insert 0 (insert 1 hfin.toFinset) with hFdef
    have h0F : (0:ℝ) ∈ F := by simp [hFdef]
    have h1F : (1:ℝ) ∈ F := by simp [hFdef]
    have hFsub : ∀ x ∈ F, x ∈ Set.Icc (0:ℝ) 1 := by
      intro x hx
      rw [hFdef] at hx
      simp only [Finset.mem_insert, Set.Finite.mem_toFinset, Set.mem_inter_iff] at hx
      rcases hx with h | h | h
      · subst h; norm_num
      · subst h; norm_num
      · exact h.1
    have hcard : F.card = F.card := rfl
    set n := F.card with hn
    have hn2 : 2 ≤ n := by
      have hsub : ({0, 1} : Finset ℝ) ⊆ F := by
        intro x hx; simp only [Finset.mem_insert, Finset.mem_singleton] at hx
        rcases hx with h|h <;> subst h <;> assumption
      have hle := Finset.card_le_card hsub
      have hpc : ({0,1}:Finset ℝ).card = 2 := Finset.card_pair (by norm_num)
      omega
    have hn0 : 0 < n := by omega
    set e := F.orderEmbOfFin hcard with he
    have he0 : e ⟨0, hn0⟩ = 0 := emb_first F n hcard hn0 0 h0F (fun x hx => (hFsub x hx).1)
    have heN : e ⟨n-1, by omega⟩ = 1 := emb_last F n hcard hn0 1 h1F (fun x hx => (hFsub x hx).2)
    refine ⟨n-1, fun j => e ⟨min j (n-1), by omega⟩, ?_, ?_, ?_, ?_⟩
    · intro a b hab
      apply e.monotone
      simp only [Fin.mk_le_mk]; omega
    · show e ⟨min 0 (n-1), _⟩ = 0
      have hh : (⟨min 0 (n-1), by omega⟩ : Fin n) = ⟨0, hn0⟩ := by apply Fin.ext; simp
      rw [hh]; exact he0
    · show e ⟨min (n-1) (n-1), _⟩ = 1
      have hh : (⟨min (n-1) (n-1), by omega⟩ : Fin n) = ⟨n-1, by omega⟩ := by apply Fin.ext; simp
      rw [hh]; exact heN
    · intro j hj
      have hjn : min j (n-1) = j := by omega
      have hj1n : min (j+1) (n-1) = j+1 := by omega
      simp only [hjn, hj1n]
      set a := e ⟨j, by omega⟩ with hae
      set b := e ⟨j+1, by omega⟩ with hbe
      have ha0 : (0:ℝ) ≤ a := by
        rw [hae, ← he0]; exact e.monotone (by simp only [Fin.mk_le_mk]; omega)
      have hb1 : b ≤ 1 := by
        rw [hbe, ← heN]; exact e.monotone (by simp only [Fin.mk_le_mk]; omega)
      have hnozero : ∀ x ∈ Set.Ioo a b, z x ≠ 0 := by
        intro x hx hzx
        have hx01 : x ∈ Set.Icc (0:ℝ) 1 :=
          ⟨le_of_lt (lt_of_le_of_lt ha0 hx.1), le_of_lt (lt_of_lt_of_le hx.2 hb1)⟩
        have hxF : x ∈ F := by
          rw [hFdef]; simp only [Finset.mem_insert, Set.Finite.mem_toFinset, Set.mem_inter_iff]
          right; right; exact ⟨hx01, hzx⟩
        have hxr : x ∈ Set.range e := by
          rw [he, Finset.range_orderEmbOfFin F hcard]; exact hxF
        obtain ⟨i, hi⟩ := hxr
        rw [← hi] at hx
        have h1 := hx.1
        have h2 := hx.2
        rw [hae, e.lt_iff_lt] at h1
        rw [hbe, e.lt_iff_lt] at h2
        simp only [Fin.lt_def] at h1 h2
        omega
      have hcont : ContinuousOn z (Set.Ioo a b) := by
        apply (hz.continuousOn).mono
        intro x hx
        exact ⟨le_of_lt (lt_of_le_of_lt ha0 hx.1), le_of_lt (lt_of_lt_of_le hx.2 hb1)⟩
      rcases sign_const z a b hcont hnozero with hpos | hneg
      · left; intro x hx; exact max_eq_left (hpos x hx)
      · right; intro x hx; exact max_eq_right (hneg x hx)

end Crownproof.ReluPiecewise

/-- **Soundness of a Farkas-certificate refutation** for non-strict rational
linear inequalities — the `la_generic` Alethe-leaf core (AY→Clean trust loop). -/
theorem Crownproof.farkas_refutation_sound
    (n m : ℕ) (A : Fin m → Fin n → ℚ) (b : Fin m → ℚ) (lam : Fin m → ℚ)
    (hlam : ∀ i, 0 ≤ lam i)
    (hcol : ∀ j, ∑ i, lam i * A i j = 0)
    (hb : ∑ i, lam i * b i < 0) :
    ¬ ∃ x : Fin n → ℚ, ∀ i, ∑ j, A i j * x j ≤ b i := by
  rintro ⟨x, hx⟩
  have hstep : ∀ i, lam i * (∑ j, A i j * x j) ≤ lam i * b i := by
    intro i
    exact mul_le_mul_of_nonneg_left (hx i) (hlam i)
  have hsum : ∑ i, lam i * (∑ j, A i j * x j) ≤ ∑ i, lam i * b i :=
    Finset.sum_le_sum (fun i _ => hstep i)
  have hzero : ∑ i, lam i * (∑ j, A i j * x j) = 0 := by
    have : ∑ i, lam i * (∑ j, A i j * x j)
        = ∑ j, (∑ i, lam i * A i j) * x j := by
      rw [Finset.sum_congr rfl (fun i _ => Finset.mul_sum _ _ _)]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i _
      ring
    rw [this]
    simp [hcol]
  rw [hzero] at hsum
  linarith

/-! ## Batch 3 (2026-07-09)

  5. `FarkasStrict.farkas_la_generic_unsat`:
     the FULL-GENERALITY `la_generic` rule — a Farkas certificate
     (nonnegative rational multipliers, vanishing column sums, negative combined
     bound OR zero bound with a strictly-used strict constraint) refutes a MIXED
     system of strict and non-strict rational linear inequalities.  Extends
     batch 2's non-strict `farkas_refutation_sound` to exactly the rule shape
     AY's Alethe `la_generic` leaves use (AY→Clean trust loop, roadmap 13).
-/

namespace Crownproof.FarkasStrict

/-- Whether an assignment `x` satisfies constraint `i`: strict constraints use `<`,
non-strict constraints use `≤`. -/
def Satisfies (n m : ℕ) (A : Fin m → Fin n → ℚ) (b : Fin m → ℚ)
    (strict : Fin m → Bool) (x : Fin n → ℚ) (i : Fin m) : Prop :=
  if strict i then ∑ j, A i j * x j < b i else ∑ j, A i j * x j ≤ b i

/-
Soundness of the Farkas-certificate refutation (`la_generic`) for a mixed
system of strict and non-strict rational linear inequalities. If there are
nonnegative multipliers `lam` such that the combined constraint coefficients
vanish and the combined bound is negative (or zero with a strictly-used strict
constraint), then the system is unsatisfiable.
-/
theorem farkas_la_generic_unsat
    (n m : ℕ) (A : Fin m → Fin n → ℚ) (b : Fin m → ℚ) (strict : Fin m → Bool)
    (lam : Fin m → ℚ) (hlam : ∀ i, 0 ≤ lam i)
    (hcoeff : ∀ j, ∑ i, lam i * A i j = 0)
    (hbound : (∑ i, lam i * b i < 0) ∨
      (∑ i, lam i * b i = 0 ∧ ∃ i, strict i = true ∧ 0 < lam i)) :
    ¬ ∃ x : Fin n → ℚ, ∀ i, Satisfies n m A b strict x i := by
  intro ⟨ x, hx ⟩;
  obtain hbound | ⟨ hbound, i, hi, hi' ⟩ := hbound;
  · -- Since $\sum_{i} \lambda_i A_{ij} x_j = \sum_{j} A_{ij} \sum_{i} \lambda_i x_j = 0$, we can simplify the inequality.
    have h_simplified : ∑ i, lam i * (∑ j, A i j * x j) ≤ ∑ i, lam i * b i := by
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ( hx i |> fun h => by unfold Satisfies at h; split_ifs at h <;> linarith ) ( hlam i );
    contrapose! h_simplified;
    rw [ show ∑ i, lam i * ∑ j, A i j * x j = ∑ j, ∑ i, lam i * A i j * x j by rw [ Finset.sum_comm ] ; exact Finset.sum_congr rfl fun _ _ => by rw [ Finset.mul_sum _ _ _ ] ; exact Finset.sum_congr rfl fun _ _ => by ring ] ; simp_all +decide [ ← Finset.sum_mul ];
  · -- For each $i$, we have $lam i * (∑ j, A i j * x j) ≤ lam i * b i$.
    have h_ineq : ∀ i, lam i * (∑ j, A i j * x j) ≤ lam i * b i := by
      intro i; specialize hx i; unfold Satisfies at hx; split_ifs at hx <;> nlinarith [ hlam i ] ;
    -- For the strict constraint $i$, we have $lam i * (∑ j, A i j * x j) < lam i * b i$.
    have h_strict : lam i * (∑ j, A i j * x j) < lam i * b i := by
      exact mul_lt_mul_of_pos_left ( hx i |> fun h => by unfold Satisfies at h; aesop ) hi';
    -- Summing over all $i$, we get $\sum_{i} lam i * (∑ j, A i j * x j) < \sum_{i} lam i * b i$.
    have h_sum_ineq : ∑ i, lam i * (∑ j, A i j * x j) < ∑ i, lam i * b i := by
      exact Finset.sum_lt_sum ( fun i _ => h_ineq i ) ⟨ i, Finset.mem_univ i, h_strict ⟩;
    -- By Fubini's theorem, we can interchange the order of summation.
    have h_fubini : ∑ i, lam i * (∑ j, A i j * x j) = ∑ j, (∑ i, lam i * A i j) * x j := by
      simpa only [ mul_assoc, Finset.mul_sum _ _ _, Finset.sum_mul ] using Finset.sum_comm;
    aesop

end Crownproof.FarkasStrict

/-! ## Batch 3b (2026-07-09)

  6. `PiecewiseAnalytic.{piecewiseAnalytic_linear_comb, piecewiseAnalytic_relu}`:
     the class of piecewise-analytic
     functions on [0,1] (finite monotone partition, analytic representative per
     closed piece — the `PA` predicate) is CLOSED under linear combination and
     under ReLU composition.  This is the DAG-induction engine for the MVF
     "finitely many breakpoints" hypothesis: pre-activations along the segment
     start affine (analytic), stay `PA` under affine combination (a) and ReLU (b),
     so every network pre-activation admits the finite partition that
     `relu_analytic_piecewise` / `piecewise_multivariate_centered_form` consume.
     Architecture: an equivalent covering characterization `PAcov` (finite
     breakpoint set) with `pa_to_cov`/`cov_to_pa` via `Finset.orderEmbOfFin`;
     internal support lemmas namespaced (incl. a local `analytic_zeros_finite`
     distinct from batch 1's top-level one).
-/

namespace Crownproof.PiecewiseAnalytic

open Set
open scoped Classical

def PA (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℕ) (t : ℕ → ℝ) (g : ℕ → ℝ → ℝ),
    t 0 = 0 ∧ t k = 1 ∧ (∀ j < k, t j ≤ t (j + 1)) ∧
    (∀ j < k, AnalyticOnNhd ℝ (g j) (Set.Icc (t j) (t (j + 1))) ∧
              Set.EqOn f (g j) (Set.Icc (t j) (t (j + 1))))

/-- A convenient equivalent characterisation: there is a finite set `S` of breakpoints, containing
`0` and `1` and contained in `[0,1]`, such that on any closed subinterval of `[0,1]` that contains
no breakpoint in its interior, `f` agrees with an analytic function. -/
def PAcov (f : ℝ → ℝ) (S : Finset ℝ) : Prop :=
  (0 : ℝ) ∈ S ∧ (1 : ℝ) ∈ S ∧ (↑S ⊆ Set.Icc (0 : ℝ) 1) ∧
  ∀ a b : ℝ, 0 ≤ a → a ≤ b → b ≤ 1 → (Set.Ioo a b ∩ (↑S : Set ℝ) = ∅) →
    ∃ g : ℝ → ℝ, AnalyticOnNhd ℝ g (Set.Icc a b) ∧ Set.EqOn f g (Set.Icc a b)

/-
From consecutive monotonicity we get full monotonicity up to `k`.
-/
lemma mono_of_consec {t : ℕ → ℝ} {k : ℕ} (h : ∀ j < k, t j ≤ t (j + 1)) :
    ∀ i j, i ≤ j → j ≤ k → t i ≤ t j := by
  intro i j hij hjk; induction hij <;> simp_all +decide [ Nat.succ_le_iff ] ;
  grind

/-
Given a monotone partition and a subinterval `[a,b] ⊆ [0,1]` whose interior contains no
breakpoint, there is a piece `j` with `t j ≤ a` and `b ≤ t (j+1)`.
-/
lemma exists_piece {k : ℕ} {t : ℕ → ℝ} (ht0 : t 0 = 0) (htk : t k = 1)
    {a b : ℝ} (ha : 0 ≤ a) (hb : b ≤ 1)
    (hin : ∀ i, i ≤ k → ¬ (a < t i ∧ t i < b)) :
    ∃ j, j < k ∧ t j ≤ a ∧ b ≤ t (j + 1) := by
  -- Consider the set of indices {i | i < k ∧ t i ≤ a}. It is nonempty: i = 0 works since t 0 = 0 ≤ a (from ha) and 0 < k. It is bounded by k. Let j be its maximum (use `Nat.findGreatest (fun i => t i ≤ a) (k-1)` or extract a maximal element of a nonempty finite set; concretely, let j be the greatest natural < k with t i ≤ a).
  obtain ⟨j, hj₁, hj₂⟩ : ∃ j < k, t j ≤ a ∧ ∀ i < k, t i ≤ a → i ≤ j := by
    obtain ⟨j, hj₁, hj₂⟩ : ∃ j < k, t j ≤ a := by
      grind;
    exact ⟨ Finset.max' ( Finset.filter ( fun i => t i ≤ a ) ( Finset.range k ) ) ⟨ j, by aesop ⟩, Finset.mem_range.mp ( Finset.mem_filter.mp ( Finset.max'_mem ( Finset.filter ( fun i => t i ≤ a ) ( Finset.range k ) ) ⟨ j, by aesop ⟩ ) |>.1 ), Finset.mem_filter.mp ( Finset.max'_mem ( Finset.filter ( fun i => t i ≤ a ) ( Finset.range k ) ) ⟨ j, by aesop ⟩ ) |>.2, fun i hi hi' => Finset.le_max' _ _ ( by aesop ) ⟩;
  grind

/-
The faithful definition implies the covering characterisation.
-/
lemma pa_to_cov {f : ℝ → ℝ} (h : PA f) : ∃ S : Finset ℝ, PAcov f S := by
  obtain ⟨ k, t, g, ht0, htk, hmono, hpiece ⟩ := h;
  refine' ⟨ Finset.image t ( Finset.range ( k + 1 ) ), _, _, _, _ ⟩ <;> simp_all +decide [ Finset.subset_iff ];
  · exact ⟨ 0, Nat.zero_le _, ht0 ⟩;
  · exact ⟨ k, le_rfl, htk ⟩;
  · intro j hj; have := mono_of_consec hmono 0 j; have := mono_of_consec hmono j k; aesop;
  · intro a b ha hb h1 h2
    obtain ⟨j, hjk, hja, hjb⟩ := exists_piece ht0 htk ha h1 (fun i hi => by
      exact fun h => h2.subset ⟨ h, i, Nat.lt_succ_of_le hi, rfl ⟩);
    exact ⟨ g j, hpiece j hjk |>.1.mono ( Set.Icc_subset_Icc hja hjb ), hpiece j hjk |>.2.mono ( Set.Icc_subset_Icc hja hjb ) ⟩

/-
The covering characterisation implies the faithful definition.
-/
lemma cov_to_pa {f : ℝ → ℝ} {S : Finset ℝ} (h : PAcov f S) : PA f := by
  -- Let n := S.card and e := S.orderEmbOfFin (rfl : S.card = n) : Fin n ↪o ℝ.
  obtain ⟨n, hn, e, he⟩ : ∃ n, S.card = n ∧ ∃ e : Fin n ↪o ℝ, Set.range e = ↑S := by
    refine' ⟨ S.card, rfl, _ ⟩;
    refine' ⟨ S.orderEmbOfFin rfl, _ ⟩;
    exact Set.ext fun x => by simp +decide [ Finset.mem_coe, Finset.mem_range ] ;
  rcases n with ( _ | n ) <;> simp_all +decide [ PAcov ];
  -- Define the partition function t : ℕ → ℝ := fun i => if hi : i < n + 1 then e ⟨i, hi⟩ else 0.
  set t : ℕ → ℝ := fun i => if hi : i < n + 1 then e ⟨i, hi⟩ else 0 with ht_def
  have ht0 : t 0 = 0 := by
    simp_all +decide [ Set.ext_iff ];
    have h_min : ∀ x ∈ S, e 0 ≤ x := by
      exact fun x hx => by obtain ⟨ y, rfl ⟩ := he x |>.2 hx; exact e.monotone ( Nat.zero_le _ ) ;
    exact le_antisymm ( h_min 0 h.1 ) ( h.2.2.1 ( he _ |>.1 ⟨ 0, rfl ⟩ ) |>.1 )
  have htk : t n = 1 := by
    -- Since $1 \in S$, there exists some $i$ such that $e i = 1$.
    obtain ⟨i, hi⟩ : ∃ i : Fin (n + 1), e i = 1 := by
      exact he.symm.subset h.2.1;
    -- Since $e$ is strictly monotone, $i$ must be the maximum element in $Fin (n + 1)$, which is $n$.
    have hi_max : i = Fin.last n := by
      exact le_antisymm ( Fin.le_last _ ) ( not_lt.mp fun contra => by linarith [ e.lt_iff_lt.2 contra, Set.mem_Icc.mp ( h.2.2.1 <| he ▸ Set.mem_range_self <| Fin.last n ) ] )
    aesop
  have ht_mono : ∀ j < n, t j ≤ t (j + 1) := by
    simp +zetaDelta at *;
    intro j hj; split_ifs <;> simp_all +decide [ Fin.le_iff_val_le_val, Fin.val_add ] ;
    linarith
  have ht_no_breakpoint : ∀ j < n, Set.Ioo (t j) (t (j + 1)) ∩ (S : Set ℝ) = ∅ := by
    simp_all +decide [ Set.ext_iff ];
    intro j hj x hx₁ hx₂ hx₃; obtain ⟨ y, rfl ⟩ := he x |>.2 hx₃; split_ifs at hx₁ <;> norm_num at *;
    · exact hx₁.not_ge ( Nat.le_of_lt_succ hx₂ );
    · lia;
  -- For each j < n, choose g' j analytic from hcov.
  have hg'_exists : ∀ j < n, ∃ g' : ℝ → ℝ, AnalyticOnNhd ℝ g' (Set.Icc (t j) (t (j + 1))) ∧ Set.EqOn f g' (Set.Icc (t j) (t (j + 1))) := by
    intro j hj
    have h_interval : 0 ≤ t j ∧ t j ≤ t (j + 1) ∧ t (j + 1) ≤ 1 := by
      simp +zetaDelta at *;
      split_ifs <;> simp_all +decide [ le_of_lt ];
      exact ⟨ h.2.2.1 ( he.subset <| Set.mem_range_self _ ) |>.1, h.2.2.1 ( he.subset <| Set.mem_range_self _ ) |>.2 ⟩;
    exact h.2.2.2 _ _ h_interval.1 h_interval.2.1 h_interval.2.2 ( ht_no_breakpoint j hj );
  choose! g' hg' using hg'_exists;
  exact ⟨ n, t, g', ht0, htk, ht_mono, hg' ⟩

/-
An analytic function on a compact interval that is not identically zero there has only finitely
many zeros in the interval.
-/
lemma analytic_zeros_finite {g : ℝ → ℝ} {c d : ℝ}
    (hg : AnalyticOnNhd ℝ g (Set.Icc c d)) (hne : ¬ Set.EqOn g 0 (Set.Icc c d)) :
    {x ∈ Set.Icc c d | g x = 0}.Finite := by
  have h_finite_zeroes : ∀ x ∈ Set.Icc c d, g x = 0 → ∃ ε > 0, ∀ y ∈ Set.Icc c d, |y - x| < ε → g y = 0 → y = x := by
    intro x hx hx'; have := hg x hx;
    have := this.eventually_eq_zero_or_eventually_ne_zero;
    rcases this with h|h;
    · apply_rules [ hg.eqOn_zero_of_preconnected_of_eventuallyEq_zero ];
      exact isPreconnected_Icc;
    · rw [ eventually_nhdsWithin_iff ] at h;
      rcases Metric.mem_nhds_iff.mp h with ⟨ ε, εpos, hε ⟩ ; use ε, εpos; intro y hy hy' hy''; specialize hε hy'; aesop;
  -- By the properties of compactness and isolated zeros, the set of zeros of $g$ in $[c, d]$ is finite.
  have h_compact : IsCompact {x ∈ Set.Icc c d | g x = 0} := by
    exact CompactIccSpace.isCompact_Icc.of_isClosed_subset ( hg.continuousOn.preimage_isClosed_of_isClosed isClosed_Icc isClosed_singleton ) fun x hx => hx.1
  have h_isolated : ∀ x ∈ {x ∈ Set.Icc c d | g x = 0}, ∃ ε > 0, ∀ y ∈ {x ∈ Set.Icc c d | g x = 0}, abs (y - x) < ε → y = x := by
    exact fun x hx => by obtain ⟨ ε, ε_pos, hε ⟩ := h_finite_zeroes x hx.1 hx.2; exact ⟨ ε, ε_pos, fun y hy hy' => hε y hy.1 hy' hy.2 ⟩ ;
  have h_finite : Set.Finite {x ∈ Set.Icc c d | g x = 0} := by
    have h_discrete : DiscreteTopology {x ∈ Set.Icc c d | g x = 0} := by
      refine' discreteTopology_iff_isOpen_singleton.mpr _;
      intro x; specialize h_isolated x x.2; rcases h_isolated with ⟨ ε, ε_pos, hε ⟩ ; exact Metric.isOpen_iff.mpr fun y hy => by
        refine ⟨ ε, ε_pos, fun z hz => ?_ ⟩
        have hyx : y = x := Set.mem_singleton_iff.mp hy
        have hdz : dist (z : ℝ) (x : ℝ) < ε := by
          rw [← hyx, ← Subtype.dist_eq]
          exact Metric.mem_ball.mp hz
        exact Set.mem_singleton_iff.mpr
          (Subtype.ext (hε (z : ℝ) z.2 (by simpa [Real.dist_eq] using hdz)))
    exact h_compact.finite (isDiscrete_iff_discreteTopology.mpr h_discrete)
  exact h_finite

/-
A continuous function on `[c,d]` with no zero in the open interval has constant sign there.
-/
lemma const_sign {g : ℝ → ℝ} {c d : ℝ} (hcd : c ≤ d) (hcont : ContinuousOn g (Set.Icc c d))
    (hno : ∀ x ∈ Set.Ioo c d, g x ≠ 0) :
    (∀ x ∈ Set.Icc c d, 0 ≤ g x) ∨ (∀ x ∈ Set.Icc c d, g x ≤ 0) := by
  by_cases hcd : c < d;
  · -- Since $g$ is continuous on $[c,d]$ and has no zeros in $(c,d)$, $g$ must be either strictly positive or strictly negative on $(c,d)$.
    have h_sign : (∀ x ∈ Set.Ioo c d, 0 < g x) ∨ (∀ x ∈ Set.Ioo c d, g x < 0) := by
      have h_sign : IsConnected (g '' Set.Ioo c d) := by
        exact ⟨ Set.Nonempty.image _ ⟨ c + ( d - c ) / 2, ⟨ by linarith, by linarith ⟩ ⟩, isPreconnected_Ioo.image _ <| hcont.mono <| Set.Ioo_subset_Icc_self ⟩;
      contrapose! hno;
      exact h_sign.Icc_subset ( Set.mem_image_of_mem _ hno.1.choose_spec.1 ) ( Set.mem_image_of_mem _ hno.2.choose_spec.1 ) ⟨ hno.1.choose_spec.2, hno.2.choose_spec.2 ⟩;
    cases' h_sign with h_sign h_sign;
    · have h_cont_pos : Filter.Tendsto g (nhdsWithin c (Set.Ioi c)) (nhds (g c)) ∧ Filter.Tendsto g (nhdsWithin d (Set.Iio d)) (nhds (g d)) := by
        have := hcont c ( Set.left_mem_Icc.mpr <| by linarith ) ; have := hcont d ( Set.right_mem_Icc.mpr <| by linarith ) ; simp_all +decide [ ContinuousWithinAt ] ;
        exact ⟨ Filter.Tendsto.mono_left ‹_› ( nhdsWithin_mono _ <| Set.Ioi_subset_Ici_self ), Filter.Tendsto.mono_left ‹_› ( nhdsWithin_mono _ <| Set.Iio_subset_Iic_self ) ⟩;
      have h_cont_pos : 0 ≤ g c ∧ 0 ≤ g d := by
        exact ⟨ le_of_tendsto_of_tendsto tendsto_const_nhds h_cont_pos.1 ( Filter.eventually_of_mem ( Ioo_mem_nhdsGT_of_mem ⟨ le_rfl, hcd ⟩ ) fun x hx => le_of_lt ( h_sign x hx ) ), le_of_tendsto_of_tendsto tendsto_const_nhds h_cont_pos.2 ( Filter.eventually_of_mem ( Ioo_mem_nhdsLT_of_mem ⟨ hcd, le_rfl ⟩ ) fun x hx => le_of_lt ( h_sign x hx ) ) ⟩;
      exact Or.inl fun x hx => if hx' : x = c then hx'.symm ▸ h_cont_pos.1 else if hx'' : x = d then hx''.symm ▸ h_cont_pos.2 else h_sign x ⟨ lt_of_le_of_ne hx.1 ( Ne.symm hx' ), lt_of_le_of_ne hx.2 hx'' ⟩ |> le_of_lt;
    · -- Since $g$ is continuous on $[c,d]$ and has no zeros in $(c,d)$, $g$ must be either strictly positive or strictly negative on $(c,d)$. Hence, $g(c) \leq 0$ and $g(d) \leq 0$.
      have h_endpoints : g c ≤ 0 ∧ g d ≤ 0 := by
        constructor <;> contrapose! h_sign;
        · have := Metric.continuousOn_iff.mp hcont c ( Set.left_mem_Icc.mpr ‹_› );
          obtain ⟨ δ, δ_pos, H ⟩ := this _ h_sign;
          exact ⟨ c + Min.min δ ( d - c ) / 2, ⟨ by linarith [ lt_min δ_pos ( sub_pos.mpr hcd ) ], by linarith [ min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ⟩, by linarith [ abs_lt.mp ( H ( c + Min.min δ ( d - c ) / 2 ) ⟨ by linarith [ lt_min δ_pos ( sub_pos.mpr hcd ) ], by linarith [ min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ⟩ ( by rw [ dist_eq_norm ] ; exact abs_lt.mpr ⟨ by linarith [ lt_min δ_pos ( sub_pos.mpr hcd ) ], by linarith [ min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ⟩ ) ) ] ⟩;
        · -- Since $g$ is continuous on $[c,d]$ and $g(d) > 0$, there exists a $\delta > 0$ such that $g(x) > 0$ for all $x \in (d - \delta, d]$.
          obtain ⟨δ, hδ_pos, hδ⟩ : ∃ δ > 0, ∀ x ∈ Set.Icc c d, abs (x - d) < δ → 0 < g x := by
            have := Metric.continuousOn_iff.mp hcont d ⟨ by linarith, by linarith ⟩;
            exact Exists.elim ( this _ h_sign ) fun δ hδ => ⟨ δ, hδ.1, fun x hx hx' => by linarith [ abs_lt.mp ( hδ.2 x hx hx' ) ] ⟩;
          exact ⟨ d - Min.min δ ( d - c ) / 2, ⟨ by linarith [ lt_min hδ_pos ( sub_pos.mpr hcd ), min_le_left δ ( d - c ), min_le_right δ ( d - c ) ], by linarith [ lt_min hδ_pos ( sub_pos.mpr hcd ), min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ⟩, le_of_lt ( hδ _ ⟨ by linarith [ lt_min hδ_pos ( sub_pos.mpr hcd ), min_le_left δ ( d - c ), min_le_right δ ( d - c ) ], by linarith [ lt_min hδ_pos ( sub_pos.mpr hcd ), min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ⟩ ( by rw [ abs_of_nonpos ] <;> linarith [ lt_min hδ_pos ( sub_pos.mpr hcd ), min_le_left δ ( d - c ), min_le_right δ ( d - c ) ] ) ) ⟩;
      exact Or.inr fun x hx => if hx' : x = c then hx'.symm ▸ h_endpoints.1 else if hx'' : x = d then hx''.symm ▸ h_endpoints.2 else h_sign x ⟨ lt_of_le_of_ne hx.1 ( Ne.symm hx' ), lt_of_le_of_ne hx.2 hx'' ⟩ |> le_of_lt;
  · grind +revert

/-
Closure under linear combinations, at the level of the covering characterisation.
-/
lemma pacov_linear {f₁ f₂ : ℝ → ℝ} {S₁ S₂ : Finset ℝ} (a b : ℝ)
    (h₁ : PAcov f₁ S₁) (h₂ : PAcov f₂ S₂) :
    PAcov (fun x => a * f₁ x + b * f₂ x) (S₁ ∪ S₂) := by
  refine' ⟨ _, _, _, _ ⟩ <;> norm_num [ h₁.1, h₁.2, h₂.1, h₂.2 ];
  intro c d hc hd hcd h; have := h₁.2.2.2 c d hc hd hcd; have := h₂.2.2.2 c d hc hd hcd; simp_all +decide [ Set.inter_union_distrib_left ] ;
  rcases ‹∃ g, AnalyticOnNhd ℝ g ( Icc c d ) ∧ EqOn f₁ g ( Icc c d ) › with ⟨ g₁, hg₁, hg₁' ⟩ ; rcases ‹∃ g, AnalyticOnNhd ℝ g ( Icc c d ) ∧ EqOn f₂ g ( Icc c d ) › with ⟨ g₂, hg₂, hg₂' ⟩ ; exact ⟨ fun x => a * g₁ x + b * g₂ x, by exact AnalyticOnNhd.add ( AnalyticOnNhd.mul ( analyticOnNhd_const ) hg₁ ) ( AnalyticOnNhd.mul ( analyticOnNhd_const ) hg₂ ), fun x hx => by simp +decide [ hg₁' hx, hg₂' hx ] ⟩ ;

/-
Closure under `ReLU`, at the level of the covering characterisation, starting from the faithful
definition of the input.
-/
lemma pacov_relu {f : ℝ → ℝ} (h : PA f) :
    ∃ S : Finset ℝ, PAcov (fun x => max (f x) 0) S := by
  revert h;
  intro h
  obtain ⟨k, t, g, ht0, htk, hmono, hpiece⟩ := h
  set B := Finset.image t (Finset.range (k+1)) with hB_def
  set Zf : ℕ → Finset ℝ := fun j => if hh : ({x ∈ (Set.Icc (t j) (t (j + 1))) | (g j) x = 0}).Finite then hh.toFinset else ∅ with hZf_def
  set S := B ∪ Finset.biUnion (Finset.range k) Zf with hS_def
  use S;
  refine' ⟨ _, _, _, _ ⟩;
  · exact Finset.mem_union_left _ ( Finset.mem_image.mpr ⟨ 0, Finset.mem_range.mpr ( Nat.succ_pos _ ), ht0 ⟩ );
  · exact Finset.mem_union_left _ ( Finset.mem_image.mpr ⟨ k, Finset.mem_range.mpr ( Nat.lt_succ_self _ ), htk ⟩ );
  · -- For any $i \leq k$, we have $0 \leq t i \leq 1$.
    have h_bounds : ∀ i ≤ k, 0 ≤ t i ∧ t i ≤ 1 := by
      intro i hi; exact ⟨ by linarith [ mono_of_consec hmono 0 i ( by linarith ) hi ], by linarith [ mono_of_consec hmono i k ( by linarith ) ( by linarith ) ] ⟩ ;
    simp_all +decide [ Finset.subset_iff ];
    refine' ⟨ fun i hi => h_bounds i ( Nat.le_of_lt_succ hi ), fun i hi => _ ⟩ ; split_ifs <;> simp_all +decide [ Set.subset_def ];
    exact fun x hx₁ hx₂ hx₃ => ⟨ by linarith [ h_bounds i ( by linarith ) ], by linarith [ h_bounds ( i + 1 ) ( by linarith ) ] ⟩;
  · intro a b ha hb hb' hAS
    obtain ⟨j, hjk, hja, hjb⟩ := exists_piece ht0 htk ha hb' (by
    intro i hi; contrapose! hAS; simp_all +decide [ Set.ext_iff ] ;
    exact ⟨ t i, ⟨ hAS.1, hAS.2 ⟩, Or.inl ⟨ i, Nat.lt_succ_of_le hi, rfl ⟩ ⟩);
    by_cases hz : Set.EqOn (g j) 0 (Set.Icc (t j) (t (j + 1)));
    · use fun _ => 0;
      exact ⟨ analyticOnNhd_const, fun x hx => by simp +decide [ hpiece j hjk |>.2 ( show x ∈ Set.Icc ( t j ) ( t ( j + 1 ) ) from ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ ), hz ( show x ∈ Set.Icc ( t j ) ( t ( j + 1 ) ) from ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ ) ] ⟩;
    · have hfin : ({x ∈ Set.Icc (t j) (t (j + 1)) | g j x = 0}).Finite := analytic_zeros_finite (hpiece j hjk).left hz;
      have hno : ∀ x ∈ Set.Ioo a b, g j x ≠ 0 := by
        intro x hx hgx
        have hxZf : x ∈ Zf j := by
          simp [Zf, hfin];
          split_ifs ; simp_all +decide [ Set.ext_iff ];
          · constructor <;> linarith;
          · exact False.elim <| ‹¬Set.Finite { x | ( t j ≤ x ∧ x ≤ t ( j + 1 ) ) ∧ g j x = 0 } › hfin;
        exact hAS.subset ⟨ hx, Finset.mem_union_right _ <| Finset.mem_biUnion.mpr ⟨ j, Finset.mem_range.mpr hjk, hxZf ⟩ ⟩;
      have hcont : ContinuousOn (g j) (Set.Icc a b) := by
        exact hpiece j hjk |>.1.continuousOn.mono ( Set.Icc_subset_Icc ( by linarith ) ( by linarith ) );
      obtain hsign | hsign := const_sign hb hcont hno;
      · use g j;
        exact ⟨ hpiece j hjk |>.1.mono ( Set.Icc_subset_Icc hja hjb ), fun x hx => by simp +decide [ hpiece j hjk |>.2 ( Set.Icc_subset_Icc hja hjb hx ), hsign x hx ] ⟩;
      · use fun _ => 0;
        exact ⟨ analyticOnNhd_const, fun x hx => max_eq_right <| by linarith [ hsign x hx, hpiece j hjk |>.2 <| show x ∈ Icc ( t j ) ( t ( j + 1 ) ) from ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ ] ⟩

/-- **(a)** Piecewise-analytic functions are closed under linear combinations. -/
theorem piecewiseAnalytic_linear_comb (f₁ f₂ : ℝ → ℝ) (a b : ℝ)
    (h₁ : PA f₁) (h₂ : PA f₂) : PA (fun x => a * f₁ x + b * f₂ x) := by
  obtain ⟨S₁, hS₁⟩ := pa_to_cov h₁
  obtain ⟨S₂, hS₂⟩ := pa_to_cov h₂
  exact cov_to_pa (pacov_linear a b hS₁ hS₂)

/-- **(b)** Piecewise-analytic functions are closed under composition with `ReLU`, i.e.
`x ↦ max (f x) 0` is piecewise-analytic whenever `f` is. -/
theorem piecewiseAnalytic_relu (f : ℝ → ℝ) (h : PA f) : PA (fun x => max (f x) 0) := by
  obtain ⟨S, hS⟩ := pacov_relu h
  exact cov_to_pa hS


end Crownproof.PiecewiseAnalytic

/-! ## Batch 4 (2026-07-09)

  7. `Resolution.PropResolution.{resolution_sound, refutation_sound}`:
     soundness of propositional resolution and
     of resolution REFUTATIONS (a derivation from formula F ending in the empty
     clause ⇒ F unsatisfiable).  This is the refutation-import core for roadmap
     item 12's remaining half: composed with `SatRelu.unsat_implies_safe`, a
     checked propositional refutation of the recovered CNF yields the kernel
     statement "the sat_relu property holds" — leaving only artifact parsing
     (LRAT/DRAT → this derivation shape) as plumbing.  Vendored with the FULL
     original header per the batch-3b lesson; `open scoped Classical` retained.
-/

namespace Crownproof.Resolution

open scoped Classical

/-!
# Soundness of propositional resolution

We model propositional CNF as described in `STATEMENT.md`:

* a *literal* is a pair `(v, p) : ℕ × Bool` of a variable index and a polarity;
* an *assignment* is a function `σ : ℕ → Bool`;
* a literal `(v, p)` is satisfied by `σ` when `σ v = p`;
* a *clause* is a finite list of literals, satisfied when at least one literal is satisfied;
* a *formula* is a finite list of clauses, satisfied when every clause is satisfied.

The resolvent of `C` and `D` on `v` removes every occurrence of `(v, true)` from `C`
and every occurrence of `(v, false)` from `D`, and appends the results.
-/

namespace PropResolution

abbrev Literal := ℕ × Bool
abbrev Assignment := ℕ → Bool
abbrev Clause := List Literal
abbrev Formula := List Clause

/-- A literal `(v, p)` is satisfied by `σ` when `σ v = p`. -/
def satLit (σ : Assignment) (l : Literal) : Prop := σ l.1 = l.2

/-- A clause is satisfied when at least one of its literals is satisfied. -/
def satClause (σ : Assignment) (C : Clause) : Prop := ∃ l ∈ C, satLit σ l

/-- A formula is satisfied when every clause in it is satisfied. -/
def satFormula (σ : Assignment) (F : Formula) : Prop := ∀ C ∈ F, satClause σ C

/-- The resolvent of `C` and `D` on the variable `v`. -/
def resolvent (C D : Clause) (v : ℕ) : Clause :=
  (C.filter (fun l => decide (l ≠ (v, true)))) ++ (D.filter (fun l => decide (l ≠ (v, false))))

/-- **Part (a): Resolution soundness.**
If `σ` satisfies both `C` and `D`, and `C` contains `(v, true)` and `D` contains `(v, false)`,
then `σ` satisfies the resolvent of `C` and `D` on `v`.

(The hypotheses `hvC : (v, true) ∈ C` and `hvD : (v, false) ∈ D` are stated as required by
part (a), but turn out not to be needed by the proof.) -/
theorem resolution_sound (σ : Assignment) (C D : Clause) (v : ℕ)
    (hC : satClause σ C) (hD : satClause σ D)
    (hvC : (v, true) ∈ C) (hvD : (v, false) ∈ D) :
    satClause σ (resolvent C D v) := by
  cases' em ( σ v = true ) with hv hv <;> simp_all +decide [ satClause, satLit, resolvent ]; all_goals grind

/-- A resolution derivation from `F`: each clause `R[i]` is the resolvent on some variable `v`
of two parent clauses `P` and `Q`, each of which is drawn from `F` or from an earlier
clause `R[j]` (`j < i`), with the complementary literals `(v, true) ∈ P` and `(v, false) ∈ Q`. -/
def Derivation (F : Formula) (R : List Clause) : Prop :=
  ∀ i (hi : i < R.length), ∃ (v : ℕ) (P Q : Clause),
    P ∈ F ++ R.take i ∧ Q ∈ F ++ R.take i ∧
    (v, true) ∈ P ∧ (v, false) ∈ Q ∧ R.get ⟨i, hi⟩ = resolvent P Q v

/-
Every clause of a derivation from `F` is satisfied by any assignment satisfying `F`.
-/
theorem derivation_sat (σ : Assignment) (F : Formula) (R : List Clause)
    (hF : satFormula σ F) (hR : Derivation F R) :
    ∀ i (hi : i < R.length), satClause σ (R.get ⟨i, hi⟩) := by
  intro i hi; induction' i using Nat.strong_induction_on with i ih; rcases hR i hi with ⟨ v, P, Q, hP, hQ, hPv, hQv, hRi ⟩ ; simp_all +decide ;
  apply resolution_sound; all_goals generalize_proofs at *;
  · cases' hP with hP hP <;> simp_all +decide [ List.mem_iff_getElem ];
    · obtain ⟨ i, hi, rfl ⟩ := hP; exact hF _ ( by simp ) ;
    · rcases hP with ⟨ j, hj, rfl ⟩ ; exact ih j hj.1 hj.2;
  · rcases hQ with ( hQ | hQ );
    · exact hF Q hQ;
    · obtain ⟨ k, hk ⟩ := List.mem_iff_getElem.mp hQ; aesop;
  · assumption;
  · assumption

/-
**Part (b): Refutation soundness.**
If a resolution derivation from `F` ends in the empty clause, then `F` is unsatisfiable.
-/
theorem refutation_sound (F : Formula) (R : List Clause)
    (hR : Derivation F R) (hlast : R.getLast? = some []) :
    ¬ ∃ σ, satFormula σ F := by
  by_contra h_eval
  obtain ⟨σ, hσ⟩ := h_eval;
  obtain ⟨ k, hk ⟩ := List.mem_iff_getElem.mp ( List.mem_of_mem_getLast? hlast );
  obtain ⟨ hk₁, hk₂ ⟩ := hk; specialize hσ; have := derivation_sat σ F R hσ hR k hk₁; simp_all +decide [ satClause ] ;

end PropResolution

end Crownproof.Resolution

/-! ## Batch 5 (2026-07-10)

  8. `RupImport.RUP.rup_sound`:
     soundness of REVERSE UNIT PROPAGATION — if clause `C` is RUP with respect
     to formula `F` (negating `C` and unit-propagating clauses of `F` reaches a
     conflict), then `F` entails `C`.  This is the exact per-step rule of the
     LRAT/DRUP artifact format, closing the gap between `refutation_sound`
     (plain resolution) and what `ay check drat` actually verifies: the full
     certified-sweep chain is now LRAT step → `rup_sound` → CNF-unsat →
     `SatRelu.unsat_implies_safe` → property holds, with only artifact PARSING
     left as trusted plumbing.  Vendored with the full original header.
-/

namespace Crownproof.RupImport

open scoped Classical

namespace RUP

/-- A literal is a pair of a variable index and a Boolean polarity. -/
abbrev Lit := ℕ × Bool

/-- The negation of a literal flips its polarity. -/
def negLit (l : Lit) : Lit := (l.1, !l.2)

/-- A clause is a finite list of literals. -/
abbrev Clause := List Lit

/-- A formula is a finite list of clauses. -/
abbrev Formula := List Clause

/-- An assignment is a function from variable indices to Booleans. -/
abbrev Assign := ℕ → Bool

/-- A literal `(v, p)` is satisfied when `σ v = p`. -/
def satLit (σ : Assign) (l : Lit) : Prop := σ l.1 = l.2

/-- A clause is satisfied when some literal in it is satisfied. -/
def satClause (σ : Assign) (C : Clause) : Prop := ∃ l ∈ C, satLit σ l

/-- A formula is satisfied when every clause is satisfied. -/
def satFormula (σ : Assign) (F : Formula) : Prop := ∀ C ∈ F, satClause σ C

/-- A clause `C ∈ F` propagates the literal `l` under the asserted list `L` if
`l ∈ C`, every other literal of `C` has its negation in `L`, and neither `l` nor
its negation is in `L`. -/
def Propagates (F : Formula) (L : List Lit) (C : Clause) (l : Lit) : Prop :=
  C ∈ F ∧ l ∈ C ∧ (∀ l' ∈ C, l' ≠ l → negLit l' ∈ L) ∧ l ∉ L ∧ negLit l ∉ L

/-- A single unit-propagation step extends `L` with a literal propagated by some
clause of `F` under `L`. -/
def Step (F : Formula) (L L' : List Lit) : Prop :=
  ∃ C l, Propagates F L C l ∧ L' = l :: L

/-- `C` is RUP with respect to `F` if, starting from the negations of all literals
of `C`, a finite sequence of unit-propagation steps ends in a conflict: some
clause of `F` has all its literals' negations in the final asserted list. -/
def IsRUP (F : Formula) (C : Clause) : Prop :=
  ∃ Lk, Relation.ReflTransGen (Step F) (C.map negLit) Lk ∧
        ∃ D ∈ F, ∀ l' ∈ D, negLit l' ∈ Lk

/-
If `σ` satisfies the negation of `l'`, then it does not satisfy `l'`.
-/
lemma not_satLit_of_satLit_negLit (σ : Assign) (l' : Lit)
    (h : satLit σ (negLit l')) : ¬ satLit σ l' := by
      unfold satLit negLit at *; aesop;

/-
One unit-propagation step preserves the invariant that `σ` satisfies every
asserted literal, provided `σ` satisfies `F`.
-/
lemma Step.satLit_preserved (F : Formula) (σ : Assign) (hF : satFormula σ F)
    {L L' : List Lit} (hstep : Step F L L')
    (hL : ∀ l ∈ L, satLit σ l) : ∀ l ∈ L', satLit σ l := by
      grind +locals

/-
The invariant that `σ` satisfies every asserted literal is maintained along
the whole propagation sequence.
-/
lemma rup_invariant (F : Formula) (σ : Assign) (hF : satFormula σ F)
    {L₀ Lk : List Lit} (hstar : Relation.ReflTransGen (Step F) L₀ Lk)
    (hbase : ∀ l ∈ L₀, satLit σ l) : ∀ l ∈ Lk, satLit σ l := by
      induction hstar;
      · assumption;
      · exact Step.satLit_preserved F σ hF ‹_› ‹_›

/-
**RUP soundness**: if `C` is RUP with respect to `F`, then every assignment
that satisfies `F` also satisfies `C`.
-/
theorem rup_sound (F : Formula) (C : Clause) (h : IsRUP F C) :
    ∀ σ, satFormula σ F → satClause σ C := by
      intro σ hσ
      obtain ⟨Lk, hLk₁, D, hD₁, hD₂⟩ := h
      by_contra h_not_sat;
      have h_base : ∀ l ∈ C.map negLit, satLit σ l := by
        simp_all +decide [ satClause ];
        simp_all +decide [ satLit, negLit ];
        grind;
      have := rup_invariant F σ hσ hLk₁ h_base;
      exact absurd ( hσ D hD₁ ) ( by rintro ⟨ l', hl', hl'' ⟩ ; exact not_satLit_of_satLit_negLit σ l' ( this _ ( hD₂ _ hl' ) ) hl'' )

end RUP

end Crownproof.RupImport

/-! ## Trust-base check -/

#print axioms Crownproof.analytic_zeros_finite
#print axioms Crownproof.pow2_compose_envelope
#print axioms Crownproof.ReluPiecewise.relu_analytic_piecewise
#print axioms Crownproof.farkas_refutation_sound
#print axioms Crownproof.FarkasStrict.farkas_la_generic_unsat
#print axioms Crownproof.PiecewiseAnalytic.piecewiseAnalytic_linear_comb
#print axioms Crownproof.PiecewiseAnalytic.piecewiseAnalytic_relu
#print axioms Crownproof.Resolution.PropResolution.resolution_sound
#print axioms Crownproof.Resolution.PropResolution.derivation_sat
#print axioms Crownproof.Resolution.PropResolution.refutation_sound
#print axioms Crownproof.RupImport.RUP.rup_sound
