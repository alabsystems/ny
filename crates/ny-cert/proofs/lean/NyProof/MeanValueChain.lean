/-
  Multivariate CHAIN STEP of NY's f64 mean-value / centered-form bounds
  (`ny-propagate/src/network/graph_ibp_f64_mvf.rs` — the segment-walk
  argument in its module docs), building on the 1-D core in
  `Crownproof.MeanValueForm`.

  The Rust module walks, for a box `B` with midpoint `m` and any `x ∈ B`,
  the segment `γ(t) = m + t•(x − m)` and studies the scalar shadow
  `g(t) = f(γ(t))`.  Its soundness argument has three parts:

    (1) the multivariate chain rule `g'(t) = Σ_i ∂_i f(γ(t)) · (x_i − m_i)`
        on each differentiability piece,
    (2) each branch-fixed partial `∂_i f` is enclosed coordinatewise in
        `[dl i, du i]` over the WHOLE box, so `g'(t)` lies in the fixed
        interval `Σ_i hull(dl i·h i, du i·h i)` with `h = x − m`,
    (3) the piecewise MVT telescope on `[0, 1]`.

  WHAT IS NOW DISCHARGED (vs v1 = `MeanValueForm.lean`, where the whole
  derivative enclosure `deriv g ∈ [lo, hi]` was a HYPOTHESIS):

    * `segment_deriv` — the multivariate chain step (1): if `f` has Fréchet
      derivative `Df` at `γ(t)`, then `g` has scalar derivative
      `Σ_i (x i − m i) · Df (Pi.single i 1)` at `t`.  The directional
      derivative decomposes into coordinate partials via the canonical
      basis (`pi_eq_sum_univ'` + linearity of `Df`).
    * `sum_partials_mem_hull` — the coordinatewise hull step (2): partials
      enclosed per coordinate give the summed per-coordinate corner hulls
      `[Σ_i min(dl i·h i, du i·h i), Σ_i max(dl i·h i, du i·h i)]` — the
      real-arithmetic content of the Rust accumulation
      `Σ_i interval_mul(D_i, [lo_i − m_i, hi_i − m_i])`.
    * `multivariate_centered_form` (+ `_of_convex`) — headline, single
      smooth piece: `f` Fréchet-differentiable along the segment with
      partials enclosed on `s` ⇒
      `f x − f m ∈ [Σ_i min(dl i·h i, du i·h i), Σ_i max(dl i·h i, du i·h i)]`.
    * `piecewise_multivariate_centered_form` — the FULL Rust shape: `f`
      only piecewise-differentiable ALONG THE SEGMENT (finite monotone
      partition `0 = t 0 ≤ … ≤ t k = 1`, Fréchet differentiability at
      `γ`-points of each OPEN piece), scalar shadow continuous on `[0, 1]`,
      partials enclosed on `s` ⇒ same conclusion, via the v1
      `piecewise_mvt_telescope`.
    * `piecewise_multivariate_centered_form_branch_fixed` — closes the gap
      found by adversarial review of the theorem above: on a DEGENERATE
      piece where a ReLU pre-activation is IDENTICALLY ZERO along the
      segment, `f` itself is not Fréchet-differentiable there, yet the
      Rust soundness argument (`graph_ibp_f64_mvf.rs`, the branch-fixed
      case analysis) admits such pieces.  Here each piece `j` carries its
      own SMOOTH branch-fixed extension `F j` (Fréchet-differentiable on
      all of `s` with partials enclosed as before) that agrees with `f` on
      the CLOSED piece along the segment (`hagree`) — this is exactly the
      Rust invariant "branch-fixed pieces agree with `f` along the
      segment".  NO continuity of the scalar shadow and NO
      differentiability of `f` itself are assumed: agreement at the shared
      partition endpoints glues the telescope.  The plain piecewise
      theorem remains for the generic (nondegenerate) case.

  HONEST SCOPE / WHAT REMAINS:

    * FINITELY many breakpoints along the segment is still a HYPOTHESIS
      (the partition `t : ℕ → ℝ` is given).  The Rust prose discharges it
      by piecewise analyticity of the supported DAG (each ReLU argument has
      finitely many zeros per analytic piece, or is identically zero there).
    * Continuity of the scalar shadow on `[0, 1]` is a hypothesis of the
      plain piecewise theorem — the Rust fail-closed op gate guarantees it
      by rejecting every discontinuous op (Trunc, ArgMax, CompareTensor,
      ScatterND).  (In the single-piece theorem it is DERIVED from
      differentiability; the branch-fixed variant needs no continuity
      hypothesis at all — endpoint agreement glues the telescope.)
    * That each interval forward-mode AD rule (Mul/Div/Sigmoid/ReLU/
      MatMul/… with outward f64 rounding) encloses its true branch-fixed
      partial `Df v (Pi.single i 1) ∈ [dl i, du i]` over the whole box is
      the per-op obligation, argued op-by-op in the Rust module docs; here
      it is the hypothesis `hpart`.
    * Everything is over ℝ: f64 outward rounding, sectioned centering of
      ulp-narrow axes, and the final intersection with the zeroth-order
      interval remain Rust-side concerns.
    * SECTIONED-WALK INSTANTIATION (adversarial-review note): the Rust walk
      seeds derivative channels only for the SEEDED axes (`axis_is_seeded`),
      while `hpart` here quantifies over every `i : ι`.  The theorems
      instantiate to the sectioned walk by choosing `ι` := the seeded axes
      and absorbing the pinned (unseeded) coordinates into `f` itself —
      then `hpart` demands exactly the enclosures Rust computes.  (Pinning
      unseeded coordinates via `h i = 0` alone would still demand finite
      enclosures Rust never computes; instantiate by restriction, not by
      zeroing.)

  Sorry-free; trust base reported by `#print axioms` at the bottom.
-/
import NyProof.MeanValueForm
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Convex.Star
import Mathlib.Algebra.BigOperators.Pi

namespace Crownproof

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-! ### The chain step: directional derivative along the segment -/

/-- A continuous linear functional on `ι → ℝ` applied to a direction `v`
decomposes into coordinate partials along the canonical basis:
`Df v = Σ_i v i · Df (Pi.single i 1)`. -/
theorem fderiv_apply_eq_sum_partials (Df : (ι → ℝ) →L[ℝ] ℝ) (v : ι → ℝ) :
    Df v = ∑ i, v i * Df (Pi.single i 1) := by
  have hv : v = ∑ i, v i • Pi.single (M := fun _ => ℝ) i 1 := pi_eq_sum_univ' v
  calc Df v = Df (∑ i, v i • Pi.single i 1) := by rw [← hv]
    _ = ∑ i, v i * Df (Pi.single i 1) := by
        rw [map_sum]
        exact Finset.sum_congr rfl fun i _ => by rw [map_smul, smul_eq_mul]

/-- **The chain step** (Rust soundness step 2, chain-rule half): if `f` has
Fréchet derivative `Df` at the segment point `γ t = m + t•(x − m)`, then the
scalar shadow `g s = f (m + s•(x − m))` has scalar derivative
`Σ_i (x i − m i) · Df (Pi.single i 1)` at `t` — i.e. `g' = Σ_i ∂_i f · h_i`
with direction `h = x − m` and partials `∂_i f = Df (Pi.single i 1)`. -/
theorem segment_deriv {f : (ι → ℝ) → ℝ} {Df : (ι → ℝ) →L[ℝ] ℝ} {x m : ι → ℝ}
    {t : ℝ} (hf : HasFDerivAt f Df (m + t • (x - m))) :
    HasDerivAt (fun s : ℝ => f (m + s • (x - m)))
      (∑ i, (x i - m i) * Df (Pi.single i 1)) t := by
  have hγ : HasDerivAt (fun s : ℝ => m + s • (x - m)) (x - m) t := by
    simpa using ((hasDerivAt_id t).smul_const (x - m)).const_add m
  have h := hf.comp_hasDerivAt t hγ
  have hd : Df (x - m) = ∑ i, (x i - m i) * Df (Pi.single i 1) := by
    rw [fderiv_apply_eq_sum_partials]
    exact Finset.sum_congr rfl fun i _ => by rw [Pi.sub_apply]
  rw [hd] at h
  exact h

/-! ### The coordinatewise hull step -/

omit [DecidableEq ι] in
/-- **Summed per-coordinate corner hulls** (Rust soundness step 2, enclosure
half): partials enclosed coordinatewise, `d i ∈ [dl i, du i]`, and a fixed
direction `h` give
`Σ_i d i·h i ∈ [Σ_i min(dl i·h i, du i·h i), Σ_i max(dl i·h i, du i·h i)]` —
the real-arithmetic content of the accumulation
`Σ_i interval_mul(D_i, x_i − m_i)`. -/
theorem sum_partials_mem_hull {dl du d h : ι → ℝ}
    (hd : ∀ i, d i ∈ Set.Icc (dl i) (du i)) :
    (∑ i, d i * h i) ∈
      Set.Icc (∑ i, min (dl i * h i) (du i * h i))
              (∑ i, max (dl i * h i) (du i * h i)) :=
  ⟨Finset.sum_le_sum fun i _ => (mul_left_mem_endpoint_hull (hd i)).1,
   Finset.sum_le_sum fun i _ => (mul_left_mem_endpoint_hull (hd i)).2⟩

/-! ### Multivariate centered form, single smooth piece -/

/-- **Multivariate centered form** (single smooth piece).  If the segment
`γ t = m + t•(x − m)`, `t ∈ [0, 1]`, stays in `s`; `f` has Fréchet
derivative `Df v` at every `v ∈ s`; and every partial is enclosed
coordinatewise, `Df v (Pi.single i 1) ∈ [dl i, du i]` on `s`; then

    f x − f m ∈ [Σ_i min(dl i·(x i − m i), du i·(x i − m i)),
                 Σ_i max(dl i·(x i − m i), du i·(x i − m i))].

Continuity of the scalar shadow is DERIVED from differentiability here. -/
theorem multivariate_centered_form {f : (ι → ℝ) → ℝ}
    {Df : (ι → ℝ) → (ι → ℝ) →L[ℝ] ℝ} {s : Set (ι → ℝ)} {x m dl du : ι → ℝ}
    (hseg : ∀ t ∈ Set.Icc (0 : ℝ) 1, m + t • (x - m) ∈ s)
    (hdiff : ∀ v ∈ s, HasFDerivAt f (Df v) v)
    (hpart : ∀ v ∈ s, ∀ i, Df v (Pi.single i 1) ∈ Set.Icc (dl i) (du i)) :
    f x - f m ∈
      Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
              (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
  set g : ℝ → ℝ := fun u => f (m + u • (x - m)) with hgdef
  have hderiv : ∀ u ∈ Set.Icc (0 : ℝ) 1,
      HasDerivAt g (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1)) u :=
    fun u hu => segment_deriv (hdiff _ (hseg u hu))
  have hmem : ∀ u ∈ Set.Icc (0 : ℝ) 1,
      (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1)) ∈
        Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
                (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
    intro u hu
    have hcomm : (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1))
        = ∑ i, Df (m + u • (x - m)) (Pi.single i 1) * (x i - m i) :=
      Finset.sum_congr rfl fun i _ => mul_comm _ _
    rw [hcomm]
    exact sum_partials_mem_hull fun i => hpart _ (hseg u hu) i
  have hIoo : Set.Ioo (0 : ℝ) 1 ⊆ Set.Icc 0 1 := Set.Ioo_subset_Icc_self
  have hcont : ContinuousOn g (Set.Icc 0 1) :=
    fun u hu => (hderiv u hu).continuousAt.continuousWithinAt
  have hdiff' : ∀ y ∈ Set.Ioo (0 : ℝ) 1, DifferentiableAt ℝ g y :=
    fun y hy => (hderiv y (hIoo hy)).differentiableAt
  have hderiv' : ∀ y ∈ Set.Ioo (0 : ℝ) 1, deriv g y ∈
      Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
              (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
    intro y hy
    rw [(hderiv y (hIoo hy)).deriv]
    exact hmem y (hIoo hy)
  have h10 := mvt_piece_bound (by norm_num : (0 : ℝ) ≤ 1) hcont hdiff' hderiv'
  have hg1 : g 1 = f x := by
    simp only [hgdef, one_smul]
    congr 1
    abel
  have hg0 : g 0 = f m := by simp [hgdef]
  rw [hg1, hg0] at h10
  simpa using h10

/-- Convex corollary: any convex `s` containing `x` and `m` satisfies the
segment hypothesis of `multivariate_centered_form` (a box `B` is convex —
this is the exact geometry of the Rust walk). -/
theorem multivariate_centered_form_of_convex {f : (ι → ℝ) → ℝ}
    {Df : (ι → ℝ) → (ι → ℝ) →L[ℝ] ℝ} {s : Set (ι → ℝ)} {x m dl du : ι → ℝ}
    (hs : Convex ℝ s) (hx : x ∈ s) (hm : m ∈ s)
    (hdiff : ∀ v ∈ s, HasFDerivAt f (Df v) v)
    (hpart : ∀ v ∈ s, ∀ i, Df v (Pi.single i 1) ∈ Set.Icc (dl i) (du i)) :
    f x - f m ∈
      Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
              (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) :=
  multivariate_centered_form
    (fun _ ht => (hs.starConvex hm).add_smul_sub_mem hx ht.1 ht.2) hdiff hpart

/-! ### Multivariate centered form, piecewise along the segment -/

/-- **Piecewise multivariate centered form** — the FULL shape of the Rust
argument.  `f` need only be Fréchet-differentiable at the `γ`-points of the
OPEN pieces of a finite monotone partition `0 = t 0 ≤ … ≤ t k = 1` of the
segment parameter (breakpoints = the ReLU branch flips along the segment);
the scalar shadow `u ↦ f (m + u•(x − m))` must be continuous on `[0, 1]`
(guaranteed in Rust by the fail-closed op gate); the partials are enclosed
coordinatewise on `s` as before.  Conclusion identical to the single-piece
form: the per-piece chain step lands the derivative in the fixed summed
corner hull, and the v1 `piecewise_mvt_telescope` telescopes. -/
theorem piecewise_multivariate_centered_form {f : (ι → ℝ) → ℝ}
    {Df : (ι → ℝ) → (ι → ℝ) →L[ℝ] ℝ} {s : Set (ι → ℝ)} {x m dl du : ι → ℝ}
    {k : ℕ} {t : ℕ → ℝ}
    (ht0 : t 0 = 0) (htk : t k = 1)
    (hmono : ∀ j, j < k → t j ≤ t (j + 1))
    (hseg : ∀ u ∈ Set.Icc (0 : ℝ) 1, m + u • (x - m) ∈ s)
    (hcont : ContinuousOn (fun u : ℝ => f (m + u • (x - m))) (Set.Icc 0 1))
    (hdiff : ∀ j, j < k → ∀ u ∈ Set.Ioo (t j) (t (j + 1)),
      HasFDerivAt f (Df (m + u • (x - m))) (m + u • (x - m)))
    (hpart : ∀ v ∈ s, ∀ i, Df v (Pi.single i 1) ∈ Set.Icc (dl i) (du i)) :
    f x - f m ∈
      Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
              (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
  set g : ℝ → ℝ := fun u => f (m + u • (x - m)) with hgdef
  -- the partition points stay in [0, 1]
  have hchain : ∀ a b : ℕ, a ≤ b → b ≤ k → t a ≤ t b := by
    intro a b hab hbk
    induction b with
    | zero => exact le_of_eq (by rw [Nat.le_zero.mp hab])
    | succ n ih =>
      rcases Nat.lt_or_ge a (n + 1) with h | h
      · exact le_trans (ih (Nat.lt_succ_iff.mp h) (Nat.le_of_succ_le hbk))
          (hmono n hbk)
      · rw [le_antisymm hab h]
  have ht_mem : ∀ j, j ≤ k → t j ∈ Set.Icc (0 : ℝ) 1 := fun j hj =>
    ⟨ht0 ▸ hchain 0 j (Nat.zero_le j) hj, htk ▸ hchain j k hj le_rfl⟩
  have hsub : ∀ j, j < k → Set.Ioo (t j) (t (j + 1)) ⊆ Set.Icc (0 : ℝ) 1 := by
    intro j hj u hu
    exact ⟨le_trans (ht_mem j (le_of_lt hj)).1 (le_of_lt hu.1),
      le_trans (le_of_lt hu.2) (ht_mem (j + 1) hj).2⟩
  have hsubIcc : ∀ j, j < k → Set.Icc (t j) (t (j + 1)) ⊆ Set.Icc (0 : ℝ) 1 :=
    fun j hj => Set.Icc_subset_Icc (ht_mem j (le_of_lt hj)).1 (ht_mem (j + 1) hj).2
  -- per-piece chain step: derivative of the shadow, enclosed in the fixed hull
  have hderiv : ∀ j, j < k → ∀ u ∈ Set.Ioo (t j) (t (j + 1)),
      HasDerivAt g (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1)) u :=
    fun j hj u hu => segment_deriv (hdiff j hj u hu)
  have hmem : ∀ u ∈ Set.Icc (0 : ℝ) 1,
      (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1)) ∈
        Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
                (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
    intro u hu
    have hcomm : (∑ i, (x i - m i) * Df (m + u • (x - m)) (Pi.single i 1))
        = ∑ i, Df (m + u • (x - m)) (Pi.single i 1) * (x i - m i) :=
      Finset.sum_congr rfl fun i _ => mul_comm _ _
    rw [hcomm]
    exact sum_partials_mem_hull fun i => hpart _ (hseg u hu) i
  -- telescope on [0, 1]
  have h10 := piecewise_mvt_telescope (f := g) (a := 0) (b := 1)
    ht0 htk hmono
    (fun j hj => hcont.mono (hsubIcc j hj))
    (fun j hj y hy => (hderiv j hj y hy).differentiableAt)
    (fun j hj y hy => by
      rw [(hderiv j hj y hy).deriv]
      exact hmem y (hsub j hj hy))
  have hg1 : g 1 = f x := by
    simp only [hgdef, one_smul]
    congr 1
    abel
  have hg0 : g 0 = f m := by simp [hgdef]
  rw [hg1, hg0] at h10
  simpa using h10

/-! ### Multivariate centered form, branch-fixed piecewise extensions -/

/-- **Branch-fixed piecewise multivariate centered form** — closes the
adversarial-review gap in `piecewise_multivariate_centered_form`: on a
degenerate piece where a ReLU pre-activation is IDENTICALLY ZERO along the
segment, `f` itself is not Fréchet-differentiable, but the branch-fixed
extension is smooth and agrees with `f` on the closed piece.  So each piece
`j < k` of the monotone partition `0 = t 0 ≤ … ≤ t k = 1` carries its own
smooth extension `F j`, Fréchet-differentiable EVERYWHERE on `s` with
partials enclosed coordinatewise in `[dl i, du i]`, and `hagree` says `F j`
agrees with `f` at the `γ`-points of the CLOSED piece
`[t j, t (j+1)]` — exactly the Rust "branch-fixed pieces agree with `f`
along the segment" invariant.  No continuity of the scalar shadow and no
differentiability of `f` itself are assumed: agreement at the shared
partition endpoints glues the telescope
`f x − f m = Σ_j (f (γ (t (j+1))) − f (γ (t j)))`, each summand is handled
by `multivariate_centered_form` for `F j` on the reparametrized
sub-segment, and the per-piece corner hulls scale by the piece widths
`t (j+1) − t j`, which sum to `1`.  Conclusion identical to the other
forms. -/
theorem piecewise_multivariate_centered_form_branch_fixed {f : (ι → ℝ) → ℝ}
    {F : ℕ → (ι → ℝ) → ℝ} {DF : ℕ → (ι → ℝ) → (ι → ℝ) →L[ℝ] ℝ}
    {s : Set (ι → ℝ)} {x m dl du : ι → ℝ} {k : ℕ} {t : ℕ → ℝ}
    (ht0 : t 0 = 0) (htk : t k = 1)
    (hmono : ∀ j, j < k → t j ≤ t (j + 1))
    (hseg : ∀ u ∈ Set.Icc (0 : ℝ) 1, m + u • (x - m) ∈ s)
    (hdiff : ∀ j, j < k → ∀ v ∈ s, HasFDerivAt (F j) (DF j v) v)
    (hpart : ∀ j, j < k → ∀ v ∈ s, ∀ i,
      DF j v (Pi.single i 1) ∈ Set.Icc (dl i) (du i))
    (hagree : ∀ j, j < k → ∀ u ∈ Set.Icc (t j) (t (j + 1)),
      f (m + u • (x - m)) = F j (m + u • (x - m))) :
    f x - f m ∈
      Set.Icc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
              (∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
  -- the partition points stay in [0, 1]
  have hchain : ∀ a b : ℕ, a ≤ b → b ≤ k → t a ≤ t b := by
    intro a b hab hbk
    induction b with
    | zero => exact le_of_eq (by rw [Nat.le_zero.mp hab])
    | succ n ih =>
      rcases Nat.lt_or_ge a (n + 1) with h | h
      · exact le_trans (ih (Nat.lt_succ_iff.mp h) (Nat.le_of_succ_le hbk))
          (hmono n hbk)
      · rw [le_antisymm hab h]
  have ht_mem : ∀ j, j ≤ k → t j ∈ Set.Icc (0 : ℝ) 1 := fun j hj =>
    ⟨ht0 ▸ hchain 0 j (Nat.zero_le j) hj, htk ▸ hchain j k hj le_rfl⟩
  -- per-piece enclosure via the branch-fixed extension, scaled by the width
  have hpiece : ∀ j, j < k →
      f (m + t (j + 1) • (x - m)) - f (m + t j • (x - m)) ∈
        Set.Icc
          ((t (j + 1) - t j) *
            ∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
          ((t (j + 1) - t j) *
            ∑ i, max (dl i * (x i - m i)) (du i * (x i - m i))) := by
    intro j hj
    have hc : (0 : ℝ) ≤ t (j + 1) - t j := sub_nonneg.mpr (hmono j hj)
    -- the sub-segment reparametrizes into the master segment
    have hrepar : ∀ u : ℝ,
        (m + t j • (x - m)) +
            u • ((m + t (j + 1) • (x - m)) - (m + t j • (x - m)))
          = m + (t j + u * (t (j + 1) - t j)) • (x - m) := by
      intro u
      funext i
      simp only [Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul]
      ring
    have hseg' : ∀ u ∈ Set.Icc (0 : ℝ) 1,
        (m + t j • (x - m)) +
            u • ((m + t (j + 1) • (x - m)) - (m + t j • (x - m))) ∈ s := by
      intro u hu
      rw [hrepar u]
      have h1 : 0 ≤ u * (t (j + 1) - t j) := mul_nonneg hu.1 hc
      have h2 : u * (t (j + 1) - t j) ≤ t (j + 1) - t j :=
        mul_le_of_le_one_left hc hu.2
      exact hseg _ ⟨by linarith [(ht_mem j (le_of_lt hj)).1],
        by linarith [(ht_mem (j + 1) hj).2]⟩
    have henc := multivariate_centered_form
      (x := m + t (j + 1) • (x - m)) (m := m + t j • (x - m))
      hseg' (hdiff j hj) (hpart j hj)
    -- coordinates of the sub-segment direction scale by the piece width
    have hcoord : ∀ i : ι,
        (m + t (j + 1) • (x - m)) i - (m + t j • (x - m)) i
          = (t (j + 1) - t j) * (x i - m i) := by
      intro i
      simp only [Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul]
      ring
    have hsum_min :
        (∑ i, min (dl i * ((m + t (j + 1) • (x - m)) i - (m + t j • (x - m)) i))
                  (du i * ((m + t (j + 1) • (x - m)) i - (m + t j • (x - m)) i)))
          = (t (j + 1) - t j) *
              ∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hcoord i,
        show dl i * ((t (j + 1) - t j) * (x i - m i))
          = (t (j + 1) - t j) * (dl i * (x i - m i)) by ring,
        show du i * ((t (j + 1) - t j) * (x i - m i))
          = (t (j + 1) - t j) * (du i * (x i - m i)) by ring,
        mul_min_of_nonneg _ _ hc]
    have hsum_max :
        (∑ i, max (dl i * ((m + t (j + 1) • (x - m)) i - (m + t j • (x - m)) i))
                  (du i * ((m + t (j + 1) • (x - m)) i - (m + t j • (x - m)) i)))
          = (t (j + 1) - t j) *
              ∑ i, max (dl i * (x i - m i)) (du i * (x i - m i)) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hcoord i,
        show dl i * ((t (j + 1) - t j) * (x i - m i))
          = (t (j + 1) - t j) * (dl i * (x i - m i)) by ring,
        show du i * ((t (j + 1) - t j) * (x i - m i))
          = (t (j + 1) - t j) * (du i * (x i - m i)) by ring,
        mul_max_of_nonneg _ _ hc]
    -- the branch-fixed extension agrees with f at the CLOSED endpoints
    have hfx : f (m + t (j + 1) • (x - m)) = F j (m + t (j + 1) • (x - m)) :=
      hagree j hj (t (j + 1)) ⟨hmono j hj, le_rfl⟩
    have hfm : f (m + t j • (x - m)) = F j (m + t j • (x - m)) :=
      hagree j hj (t j) ⟨le_rfl, hmono j hj⟩
    rw [hsum_min, hsum_max] at henc
    rw [hfx, hfm]
    exact henc
  -- telescoping identities: widths sum to 1, differences to f x − f m
  have hwidth : (∑ j ∈ Finset.range k, (t (j + 1) - t j)) = 1 := by
    rw [Finset.sum_range_sub t k, ht0, htk, sub_zero]
  have hend1 : m + (1 : ℝ) • (x - m) = x := by
    rw [one_smul]; abel
  have hend0 : m + (0 : ℝ) • (x - m) = m := by
    rw [zero_smul, add_zero]
  have htele :
      (∑ j ∈ Finset.range k,
        (f (m + t (j + 1) • (x - m)) - f (m + t j • (x - m)))) = f x - f m := by
    have h := Finset.sum_range_sub (fun j => f (m + t j • (x - m))) k
    simp only [ht0, htk, hend1, hend0] at h
    exact h
  refine ⟨?_, ?_⟩
  · calc (∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)))
        = ∑ j ∈ Finset.range k, (t (j + 1) - t j) *
            ∑ i, min (dl i * (x i - m i)) (du i * (x i - m i)) := by
          rw [← Finset.sum_mul, hwidth, one_mul]
      _ ≤ ∑ j ∈ Finset.range k,
            (f (m + t (j + 1) • (x - m)) - f (m + t j • (x - m))) :=
          Finset.sum_le_sum fun j hj => (hpiece j (Finset.mem_range.mp hj)).1
      _ = f x - f m := htele
  · calc f x - f m
        = ∑ j ∈ Finset.range k,
            (f (m + t (j + 1) • (x - m)) - f (m + t j • (x - m))) := htele.symm
      _ ≤ ∑ j ∈ Finset.range k, (t (j + 1) - t j) *
            ∑ i, max (dl i * (x i - m i)) (du i * (x i - m i)) :=
          Finset.sum_le_sum fun j hj => (hpiece j (Finset.mem_range.mp hj)).2
      _ = ∑ i, max (dl i * (x i - m i)) (du i * (x i - m i)) := by
          rw [← Finset.sum_mul, hwidth, one_mul]

end Crownproof

#print axioms Crownproof.fderiv_apply_eq_sum_partials
#print axioms Crownproof.segment_deriv
#print axioms Crownproof.sum_partials_mem_hull
#print axioms Crownproof.multivariate_centered_form
#print axioms Crownproof.multivariate_centered_form_of_convex
#print axioms Crownproof.piecewise_multivariate_centered_form
#print axioms Crownproof.piecewise_multivariate_centered_form_branch_fixed
