/-
  Soundness CORE of NY's f64 mean-value / centered-form bounds
  (`ny-propagate/src/network/graph_ibp_f64_mvf.rs`, commit 8aafe56b —
  roadmap item 10: "MVF soundness → unlock nn4sys mscn").

  The Rust module certifies, for one output element `f` of a supported DAG
  over a box `B` with midpoint `m`:

      f(x) ∈ f(m) + Σ_i D_i · (x_i − m_i)      for every x ∈ B,

  by the piecewise mean-value/telescope argument in its module docs: along the
  segment `γ(t) = m + t(x − m)` the scalar function `g(t) = f(γ(t))` is
  continuous and piecewise differentiable with `g'(t)` enclosed in a fixed
  interval `[lo, hi] := Σ_i D_i·(x_i − m_i)` on every open piece; the per-piece
  MVT telescopes to `g(1) − g(0) ∈ [lo, hi]` (the pieces' widths are
  nonnegative and sum to 1, and an interval is convex).

  WHAT IS PROVED HERE (all over ℝ, one real variable — the scalar shadow
  `g : ℝ → ℝ` of the segment walk):

    * `interval_convex_combination_mem` — the 1-D containment step:
      `d ∈ [lo, hi]`, `h ∈ [0, w]` ⇒ `d·h ∈ [min (lo·w) 0, max (hi·w) 0]`
      (the per-piece term `g'(ξ_j)·(t_{j+1} − t_j)` with weight in `[0, w]`).
    * `interval_convex_combination_sum_mem` — a convex combination
      (nonnegative weights summing to 1) of values in `[lo, hi]` stays in
      `[lo, hi]` — "convexity of the interval T" in the Rust docs.
    * `interval_mul_mem` — rectangle-corner hull: `d ∈ [dl, du]`,
      `h ∈ [hl, hu]` ⇒ `d·h` lies between the min and max of the four corner
      products.  This is the real-arithmetic fact under the Rust accumulation
      step `interval_mul(d_lo, d_hi, lo_i − m_i, hi_i − m_i)`.
    * `relu_hull_mul_mem` — the ReLU straddling rule: a branch multiplier
      `s ∈ [0, 1]` (in the code: `s ∈ {0, 1}`) times `d ∈ [dl, du]` lies in
      `[min dl 0, max du 0]` — the hull `hull(0·d, 1·d)`.
    * `mvt_piece_bound` — single-piece two-sided MVT: `f` continuous on
      `[c, d]`, differentiable on `(c, d)` with `deriv f ∈ [lo, hi]` there ⇒
      `f d − f c ∈ [lo·(d − c), hi·(d − c)]`  (mathlib's
      `Convex.mul_sub_le_image_sub_of_le_deriv` / `…image_sub_le_mul_sub_of_deriv_le`).
    * `piecewise_mvt_telescope` — the telescoped composition over a finite
      monotone partition `a = t 0 ≤ t 1 ≤ … ≤ t k = b`:
      `f b − f a ∈ [lo·(b − a), hi·(b − a)]`.
    * `piecewise_mvt_telescope_subinterval` — the same increment bound between
      ANY two points `u ≤ v` of `[a, b]` (the partition clipped to `[u, v]`);
      this is the form the segment walk needs, since `x` is arbitrary in `B`.
    * `centered_form_enclosure` — the centered-form corollary: with any
      center `m ∈ [a, b]` and any `x ∈ [a, b]`,
      `f x ∈ [f m + min₄, f m + max₄]` where `min₄`/`max₄` are the corner
      hull of `[lo, hi]·[a − m, b − m]` — exactly the shape
      `f(m) ⊕ D·[lo − m, hi − m]` the Rust cell accumulates.

  HONEST SCOPE / GAPS (what graph_ibp_f64_mvf.rs assumes beyond this file):

    * ONE real variable.  The Rust argument is about `g(t) = f(γ(t))` along a
      segment in ℝⁿ; the multivariate chain-rule step
      `g'(t) = Σ_i (∂f_branch/∂x_i)(γ(t))·(x_i − m_i)` and the claim that the
      interval forward-mode AD rules (Mul/Div/Sigmoid/ReLU/MatMul/… with
      outward f64 rounding) enclose every branch-fixed partial over the whole
      box are NOT formalized here.  Here the derivative enclosure
      `deriv f ∈ [lo, hi]` on each open piece is a hypothesis.
    * The PARTITION IS GIVEN.  The Rust prose derives the existence of
      finitely many breakpoints from piecewise analyticity of the DAG
      (finitely many ReLU-argument zeros per piece, or an identically-zero
      argument).  Here the finite monotone partition `t : ℕ → ℝ` is a
      hypothesis; degenerate pieces (`t j = t (j+1)`) are allowed.
    * Differentiability is required two-sided on each OPEN piece and `f` must
      be continuous on the CLOSED interval — matching the Rust fail-closed
      contract, which rejects every discontinuous op (Trunc, ArgMax,
      CompareTensor, ScatterND).
    * Everything is over ℝ.  The f64 outward rounding (next_down/next_up,
      Higham gamma_n widening), the sectioned centering of ulp-narrow axes
      (`f(m_S; x_T)` enclosed by a partially-pinned cell forward), and the
      final intersection with the zeroth-order interval are Rust-side
      concerns, out of scope for this file.

  Sorry-free; trust base reported by `#print axioms` at the bottom.
-/
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

namespace Crownproof

/-! ### 1-D containment steps (interval arithmetic over ℝ) -/

/-- For a fixed left factor `c`, the linear image of `h ∈ [hl, hu]` lies in
the hull of the endpoint products.  (Linearity in the interval argument —
used to bound `lo·(x − m)` and `hi·(x − m)` by corner products.) -/
theorem mul_right_mem_endpoint_hull {c hl hu h : ℝ} (hh : h ∈ Set.Icc hl hu) :
    c * h ∈ Set.Icc (min (c * hl) (c * hu)) (max (c * hl) (c * hu)) := by
  obtain ⟨h1, h2⟩ := hh
  rcases le_total 0 c with hc | hc
  · exact ⟨le_trans (min_le_left _ _) (mul_le_mul_of_nonneg_left h1 hc),
      le_trans (mul_le_mul_of_nonneg_left h2 hc) (le_max_right _ _)⟩
  · exact ⟨le_trans (min_le_right _ _) (mul_le_mul_of_nonpos_left h2 hc),
      le_trans (mul_le_mul_of_nonpos_left h1 hc) (le_max_left _ _)⟩

/-- For a fixed right factor `c`, the linear image of `d ∈ [dl, du]` lies in
the hull of the endpoint products. -/
theorem mul_left_mem_endpoint_hull {dl du c d : ℝ} (hd : d ∈ Set.Icc dl du) :
    d * c ∈ Set.Icc (min (dl * c) (du * c)) (max (dl * c) (du * c)) := by
  obtain ⟨h1, h2⟩ := hd
  rcases le_total 0 c with hc | hc
  · exact ⟨le_trans (min_le_left _ _) (mul_le_mul_of_nonneg_right h1 hc),
      le_trans (mul_le_mul_of_nonneg_right h2 hc) (le_max_right _ _)⟩
  · exact ⟨le_trans (min_le_right _ _) (mul_le_mul_of_nonpos_right h2 hc),
      le_trans (mul_le_mul_of_nonpos_right h1 hc) (le_max_left _ _)⟩

/-- Rectangle-corner hull soundness of interval multiplication: for
`d ∈ [dl, du]` and `h ∈ [hl, hu]`, the product `d·h` lies between the min and
max of the four corner products.  This is the exact real-arithmetic content
of the Rust `interval_mul` used in the centered-form accumulation
`Σ_i D_i · [lo_i − m_i, hi_i − m_i]` (graph_ibp_f64_mvf.rs, `interval_mul`
plus 1-ulp outward widening — the widening only enlarges this hull). -/
theorem interval_mul_mem {dl du hl hu d h : ℝ}
    (hd : d ∈ Set.Icc dl du) (hh : h ∈ Set.Icc hl hu) :
    d * h ∈ Set.Icc (min (min (dl * hl) (dl * hu)) (min (du * hl) (du * hu)))
                    (max (max (dl * hl) (dl * hu)) (max (du * hl) (du * hu))) := by
  have hmid := mul_right_mem_endpoint_hull (c := d) hh
  have hlft := mul_left_mem_endpoint_hull (c := hl) hd
  have hrgt := mul_left_mem_endpoint_hull (c := hu) hd
  constructor
  · have c1 : min (min (dl * hl) (dl * hu)) (min (du * hl) (du * hu))
        ≤ min (dl * hl) (du * hl) :=
      le_min (le_trans (min_le_left _ _) (min_le_left _ _))
        (le_trans (min_le_right _ _) (min_le_left _ _))
    have c2 : min (min (dl * hl) (dl * hu)) (min (du * hl) (du * hu))
        ≤ min (dl * hu) (du * hu) :=
      le_min (le_trans (min_le_left _ _) (min_le_right _ _))
        (le_trans (min_le_right _ _) (min_le_right _ _))
    exact le_trans (le_min (le_trans c1 hlft.1) (le_trans c2 hrgt.1)) hmid.1
  · have c1 : max (dl * hl) (du * hl)
        ≤ max (max (dl * hl) (dl * hu)) (max (du * hl) (du * hu)) :=
      max_le (le_trans (le_max_left _ _) (le_max_left _ _))
        (le_trans (le_max_left _ _) (le_max_right _ _))
    have c2 : max (dl * hu) (du * hu)
        ≤ max (max (dl * hl) (dl * hu)) (max (du * hl) (du * hu)) :=
      max_le (le_trans (le_max_right _ _) (le_max_left _ _))
        (le_trans (le_max_right _ _) (le_max_right _ _))
    exact le_trans hmid.2 (max_le (le_trans hlft.2 c1) (le_trans hrgt.2 c2))

/-- 1-D containment step of the telescoped mean value form: a derivative value
`d ∈ [lo, hi]` times a nonnegative piece weight `h ∈ [0, w]` lands in
`[min (lo·w) 0, max (hi·w) 0]`.  This is the per-piece term
`g'(ξ_j) · (t_{j+1} − t_j)` of the telescope, contained in the hull of
`[lo, hi] · [0, w]`. -/
theorem interval_convex_combination_mem {lo hi w d h : ℝ}
    (hd : d ∈ Set.Icc lo hi) (hh : h ∈ Set.Icc 0 w) :
    d * h ∈ Set.Icc (min (lo * w) 0) (max (hi * w) 0) := by
  obtain ⟨hd1, hd2⟩ := hd
  obtain ⟨hh0, hhw⟩ := hh
  have hw : (0 : ℝ) ≤ w := le_trans hh0 hhw
  constructor
  · rcases le_total 0 d with hdp | hdn
    · exact le_trans (min_le_right _ _) (mul_nonneg hdp hh0)
    · calc min (lo * w) 0 ≤ lo * w := min_le_left _ _
        _ ≤ d * w := mul_le_mul_of_nonneg_right hd1 hw
        _ ≤ d * h := mul_le_mul_of_nonpos_left hhw hdn
  · rcases le_total 0 d with hdp | hdn
    · calc d * h ≤ d * w := mul_le_mul_of_nonneg_left hhw hdp
        _ ≤ hi * w := mul_le_mul_of_nonneg_right hd2 hw
        _ ≤ max (hi * w) 0 := le_max_left _ _
    · have hdh : d * h ≤ 0 := by nlinarith
      exact le_trans hdh (le_max_right _ _)

/-- "Convexity of the interval `T`" (Rust soundness step 3): a convex
combination — nonnegative weights `w j` summing to 1 — of values
`c j ∈ [lo, hi]` stays in `[lo, hi]`. -/
theorem interval_convex_combination_sum_mem {k : ℕ} {w c : ℕ → ℝ} {lo hi : ℝ}
    (hw0 : ∀ j, j < k → 0 ≤ w j)
    (hw1 : ∑ j ∈ Finset.range k, w j = 1)
    (hc : ∀ j, j < k → c j ∈ Set.Icc lo hi) :
    (∑ j ∈ Finset.range k, w j * c j) ∈ Set.Icc lo hi := by
  have hlo : lo = ∑ j ∈ Finset.range k, w j * lo := by
    rw [← Finset.sum_mul, hw1, one_mul]
  constructor
  · calc lo = ∑ j ∈ Finset.range k, w j * lo := hlo
      _ ≤ ∑ j ∈ Finset.range k, w j * c j :=
        Finset.sum_le_sum fun j hj =>
          mul_le_mul_of_nonneg_left (hc j (Finset.mem_range.mp hj)).1
            (hw0 j (Finset.mem_range.mp hj))
  · calc (∑ j ∈ Finset.range k, w j * c j)
        ≤ ∑ j ∈ Finset.range k, w j * hi :=
        Finset.sum_le_sum fun j hj =>
          mul_le_mul_of_nonneg_left (hc j (Finset.mem_range.mp hj)).2
            (hw0 j (Finset.mem_range.mp hj))
      _ = hi := by rw [← Finset.sum_mul, hw1, one_mul]

/-- ReLU straddling rule of the derivative channel: on a box where the ReLU
argument straddles zero, the branch multiplier is `s ∈ {0, 1} ⊆ [0, 1]` on
each piece, so the contribution `s·d` with `d ∈ [dl, du]` is contained in
`hull(0·d, 1·d) = [min dl 0, max du 0]` — the Rust rule
`*l = l.min(0.0); *h = h.max(0.0)`. -/
theorem relu_hull_mul_mem {dl du d s : ℝ}
    (hd : d ∈ Set.Icc dl du) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
    s * d ∈ Set.Icc (min dl 0) (max du 0) := by
  obtain ⟨hd1, hd2⟩ := hd
  obtain ⟨hs0, hs1⟩ := hs
  constructor
  · rcases le_total 0 d with hdp | hdn
    · exact le_trans (min_le_right _ _) (mul_nonneg hs0 hdp)
    · calc min dl 0 ≤ dl := min_le_left _ _
        _ ≤ d := hd1
        _ ≤ s * d := by nlinarith
  · rcases le_total 0 d with hdp | hdn
    · calc s * d ≤ d := by nlinarith
        _ ≤ du := hd2
        _ ≤ max du 0 := le_max_left _ _
    · have hsd : s * d ≤ 0 := by nlinarith
      exact le_trans hsd (le_max_right _ _)

/-! ### Piecewise mean value telescope -/

/-- Two-sided mean value bound on ONE piece: `f` continuous on `[c, d]`,
differentiable on the open piece `(c, d)` with derivative in `[lo, hi]`
there, gives `f d − f c ∈ [lo·(d − c), hi·(d − c)]`.  Degenerate pieces
(`c = d`) are allowed (both sides are `0`). -/
theorem mvt_piece_bound {f : ℝ → ℝ} {c d lo hi : ℝ} (hcd : c ≤ d)
    (hcont : ContinuousOn f (Set.Icc c d))
    (hdiff : ∀ y ∈ Set.Ioo c d, DifferentiableAt ℝ f y)
    (hderiv : ∀ y ∈ Set.Ioo c d, deriv f y ∈ Set.Icc lo hi) :
    f d - f c ∈ Set.Icc (lo * (d - c)) (hi * (d - c)) := by
  have hdiff' : DifferentiableOn ℝ f (interior (Set.Icc c d)) := by
    rw [interior_Icc]
    exact fun y hy => (hdiff y hy).differentiableWithinAt
  have hmc : c ∈ Set.Icc c d := Set.left_mem_Icc.mpr hcd
  have hmd : d ∈ Set.Icc c d := Set.right_mem_Icc.mpr hcd
  constructor
  · exact (convex_Icc c d).mul_sub_le_image_sub_of_le_deriv hcont hdiff'
      (fun y hy => (hderiv y (by rwa [interior_Icc] at hy)).1) c hmc d hmd hcd
  · exact (convex_Icc c d).image_sub_le_mul_sub_of_deriv_le hcont hdiff'
      (fun y hy => (hderiv y (by rwa [interior_Icc] at hy)).2) c hmc d hmd hcd

/-- **Piecewise MVT telescope** (Rust soundness steps 1+3).  Let
`a = t 0 ≤ t 1 ≤ … ≤ t k = b` be a finite monotone partition (degenerate
pieces allowed), `f` continuous on each closed piece and differentiable on
each open piece with derivative in the FIXED interval `[lo, hi]` on every
piece.  Then `f b − f a ∈ [lo·(b − a), hi·(b − a)]`: the per-piece MVT bounds
telescope, because the piece widths are nonnegative and sum to `b − a`. -/
theorem piecewise_mvt_telescope {f : ℝ → ℝ} {a b lo hi : ℝ} {k : ℕ} {t : ℕ → ℝ}
    (ht0 : t 0 = a) (htk : t k = b)
    (hmono : ∀ j, j < k → t j ≤ t (j + 1))
    (hcont : ∀ j, j < k → ContinuousOn f (Set.Icc (t j) (t (j + 1))))
    (hdiff : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), DifferentiableAt ℝ f y)
    (hderiv : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), deriv f y ∈ Set.Icc lo hi) :
    f b - f a ∈ Set.Icc (lo * (b - a)) (hi * (b - a)) := by
  have hpiece : ∀ j ∈ Finset.range k,
      f (t (j + 1)) - f (t j) ∈
        Set.Icc (lo * (t (j + 1) - t j)) (hi * (t (j + 1) - t j)) := by
    intro j hj
    have hjk := Finset.mem_range.mp hj
    exact mvt_piece_bound (hmono j hjk) (hcont j hjk) (hdiff j hjk) (hderiv j hjk)
  have hsum : ∑ j ∈ Finset.range k, (f (t (j + 1)) - f (t j)) = f b - f a := by
    rw [Finset.sum_range_sub (fun j => f (t j)) k, ht0, htk]
  have hwidth : ∑ j ∈ Finset.range k, (t (j + 1) - t j) = b - a := by
    rw [Finset.sum_range_sub t k, ht0, htk]
  have hlosum : lo * (b - a) = ∑ j ∈ Finset.range k, lo * (t (j + 1) - t j) := by
    rw [← Finset.mul_sum, hwidth]
  constructor
  · calc lo * (b - a) = ∑ j ∈ Finset.range k, lo * (t (j + 1) - t j) := hlosum
      _ ≤ ∑ j ∈ Finset.range k, (f (t (j + 1)) - f (t j)) :=
        Finset.sum_le_sum fun j hj => (hpiece j hj).1
      _ = f b - f a := hsum
  · calc f b - f a = ∑ j ∈ Finset.range k, (f (t (j + 1)) - f (t j)) := hsum.symm
      _ ≤ ∑ j ∈ Finset.range k, hi * (t (j + 1) - t j) :=
        Finset.sum_le_sum fun j hj => (hpiece j hj).2
      _ = hi * (b - a) := by rw [← Finset.mul_sum, hwidth]

/-- Piecewise MVT increment bound between ANY two points `u ≤ v` of `[a, b]`
(the segment walk needs `m` to arbitrary `x`, not endpoint to endpoint):
the partition is clipped to `[u, v]` via `j ↦ min (max (t j) u) v`, whose
open pieces embed in the original open pieces, so the telescope applies.
Continuity is required on all of `[a, b]` here (that is what the Rust
fail-closed op gate guarantees). -/
theorem piecewise_mvt_telescope_subinterval {f : ℝ → ℝ} {a b lo hi u v : ℝ}
    {k : ℕ} {t : ℕ → ℝ}
    (ht0 : t 0 = a) (htk : t k = b)
    (hmono : ∀ j, j < k → t j ≤ t (j + 1))
    (hcont : ContinuousOn f (Set.Icc a b))
    (hdiff : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), DifferentiableAt ℝ f y)
    (hderiv : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), deriv f y ∈ Set.Icc lo hi)
    (hu : u ∈ Set.Icc a b) (hv : v ∈ Set.Icc a b) (huv : u ≤ v) :
    f v - f u ∈ Set.Icc (lo * (v - u)) (hi * (v - u)) := by
  set c : ℕ → ℝ := fun j => min (max (t j) u) v with hcdef
  have hcmem : ∀ j, c j ∈ Set.Icc u v := fun j =>
    ⟨le_min (le_max_right _ _) huv, min_le_right _ _⟩
  have hc0 : c 0 = u := by
    simp only [hcdef, ht0, max_eq_right hu.1, min_eq_left huv]
  have hck : c k = v := by
    simp only [hcdef, htk, max_eq_left hu.2, min_eq_right hv.2]
  have hcmono : ∀ j, j < k → c j ≤ c (j + 1) := fun j hj =>
    min_le_min (max_le_max (hmono j hj) le_rfl) le_rfl
  have hsub : ∀ j, j < k → Set.Ioo (c j) (c (j + 1)) ⊆ Set.Ioo (t j) (t (j + 1)) := by
    intro j hj y hy
    obtain ⟨hy1, hy2⟩ := hy
    have hyv : y < v := lt_of_lt_of_le hy2 (min_le_right _ _)
    have hyu : u < y := lt_of_le_of_lt (hcmem j).1 hy1
    constructor
    · rcases min_lt_iff.mp hy1 with hlt | hlt
      · exact lt_of_le_of_lt (le_max_left _ _) hlt
      · exact absurd hlt (lt_asymm hyv)
    · have hym : y < max (t (j + 1)) u := lt_of_lt_of_le hy2 (min_le_left _ _)
      rcases lt_max_iff.mp hym with hlt | hlt
      · exact hlt
      · exact absurd hlt (lt_asymm hyu)
  have hIccsub : ∀ j, Set.Icc (c j) (c (j + 1)) ⊆ Set.Icc a b := fun j =>
    Set.Icc_subset_Icc (le_trans hu.1 (hcmem j).1) (le_trans (hcmem (j + 1)).2 hv.2)
  exact piecewise_mvt_telescope hc0 hck hcmono
    (fun j _ => hcont.mono (hIccsub j))
    (fun j hj y hy => hdiff j hj y (hsub j hj hy))
    (fun j hj y hy => hderiv j hj y (hsub j hj hy))

/-! ### Centered-form corollary -/

/-- **Centered-form enclosure** — the shape the Rust cell uses
(`f(m) ⊕ D·[lo − m, hi − m]`, one seeded axis, over ℝ).  Under the piecewise
hypotheses of the telescope (partition of `[a, b]`, continuity on `[a, b]`,
derivative in `[lo, hi]` on every open piece), for ANY center `m ∈ [a, b]`
and ANY `x ∈ [a, b]`:

    f x ∈ [ f m + min₄ , f m + max₄ ]

where `min₄`/`max₄` are the corner hull of `[lo, hi] · [a − m, b − m]`.
Since `a − m ≤ 0 ≤ b − m`, this interval always contains `f m` itself.
Any interior point is a valid center — exactness of the midpoint is not
required (matching the Rust comment on `mid.clamp`). -/
theorem centered_form_enclosure {f : ℝ → ℝ} {a b lo hi m x : ℝ}
    {k : ℕ} {t : ℕ → ℝ}
    (ht0 : t 0 = a) (htk : t k = b)
    (hmono : ∀ j, j < k → t j ≤ t (j + 1))
    (hcont : ContinuousOn f (Set.Icc a b))
    (hdiff : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), DifferentiableAt ℝ f y)
    (hderiv : ∀ j, j < k → ∀ y ∈ Set.Ioo (t j) (t (j + 1)), deriv f y ∈ Set.Icc lo hi)
    (hm : m ∈ Set.Icc a b) (hx : x ∈ Set.Icc a b) :
    f x ∈ Set.Icc
      (f m + min (min (lo * (a - m)) (lo * (b - m)))
                 (min (hi * (a - m)) (hi * (b - m))))
      (f m + max (max (lo * (a - m)) (lo * (b - m)))
                 (max (hi * (a - m)) (hi * (b - m)))) := by
  have hxm : x - m ∈ Set.Icc (a - m) (b - m) :=
    ⟨sub_le_sub_right hx.1 m, sub_le_sub_right hx.2 m⟩
  rcases le_total m x with hmx | hxm2
  · have h := piecewise_mvt_telescope_subinterval ht0 htk hmono hcont hdiff hderiv
      hm hx hmx
    have hlo := (mul_right_mem_endpoint_hull (c := lo) hxm).1
    have hhi := (mul_right_mem_endpoint_hull (c := hi) hxm).2
    constructor
    · have h1 := h.1
      have h2 : min (min (lo * (a - m)) (lo * (b - m)))
          (min (hi * (a - m)) (hi * (b - m))) ≤ lo * (x - m) :=
        le_trans (min_le_left _ _) hlo
      linarith
    · have h1 := h.2
      have h2 : hi * (x - m) ≤ max (max (lo * (a - m)) (lo * (b - m)))
          (max (hi * (a - m)) (hi * (b - m))) :=
        le_trans hhi (le_max_right _ _)
      linarith
  · have h := piecewise_mvt_telescope_subinterval ht0 htk hmono hcont hdiff hderiv
      hx hm hxm2
    have e1 : lo * (m - x) = -(lo * (x - m)) := by ring
    have e2 : hi * (m - x) = -(hi * (x - m)) := by ring
    have hlo := (mul_right_mem_endpoint_hull (c := lo) hxm).2
    have hhi := (mul_right_mem_endpoint_hull (c := hi) hxm).1
    constructor
    · have h1 := h.2
      have h2 : min (min (lo * (a - m)) (lo * (b - m)))
          (min (hi * (a - m)) (hi * (b - m))) ≤ hi * (x - m) :=
        le_trans (min_le_right _ _) hhi
      linarith
    · have h1 := h.1
      have h2 : lo * (x - m) ≤ max (max (lo * (a - m)) (lo * (b - m)))
          (max (hi * (a - m)) (hi * (b - m))) :=
        le_trans hlo (le_max_left _ _)
      linarith

end Crownproof

#print axioms Crownproof.mul_right_mem_endpoint_hull
#print axioms Crownproof.mul_left_mem_endpoint_hull
#print axioms Crownproof.interval_mul_mem
#print axioms Crownproof.interval_convex_combination_mem
#print axioms Crownproof.interval_convex_combination_sum_mem
#print axioms Crownproof.relu_hull_mul_mem
#print axioms Crownproof.mvt_piece_bound
#print axioms Crownproof.piecewise_mvt_telescope
#print axioms Crownproof.piecewise_mvt_telescope_subinterval
#print axioms Crownproof.centered_form_enclosure
