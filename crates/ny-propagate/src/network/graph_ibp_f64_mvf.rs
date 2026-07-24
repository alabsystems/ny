// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound f64 FIRST-ORDER (mean-value / centered form) bounds over a
//! [`GraphNetwork`] via interval forward-mode derivatives (#f64-mvf).
//!
//! # Why (nn4sys mscn `_dual` multi-axis plateau clauses)
//!
//! The zeroth-order f64 interval forward ([`GraphNetwork::propagate_ibp_f64_cell`])
//! has an interval-DEPENDENCY excess that shrinks only LINEARLY with the input
//! box width. The mscn `_dual` band-plateau clauses (2-3 sweep axes, band
//! margins ~1e-5) therefore need ~35k zeroth-order leaves each — ~10x every
//! official time budget (measured, commit 196720ef). The centered form
//!
//! ```text
//! f(x) ∈ f(m) + Σ_i D_i · (x_i − m_i),     m = box midpoint,
//! ```
//!
//! with `D_i` an interval enclosure of `∂f/∂x_i` over the WHOLE box, has
//! excess `O(width²)` — quadratic — because `f(m)` is a point evaluation
//! (ulp-tight via the cell forward) and the `D_i` themselves converge linearly
//! to the true point derivatives, making the residual term second order. On a
//! flat plateau (`D_i ≈ 0`) it decides a clause near the ROOT box.
//!
//! # Soundness argument (mean value theorem for the piecewise-analytic DAG)
//!
//! Fix a box `B` (the value channel's input intervals), a point `m ∈ B`, and
//! any `x ∈ B`. Let `γ(t) = m + t·(x − m)` for `t ∈ [0, 1]` and
//! `g(t) = f(γ(t))` for one output element `f`.
//!
//! 1. *Piecewise analyticity.* Every supported op is real-analytic (Linear,
//!    MatMul, Mul, Add, Sub, AddConstant, MulConstant, ReduceSum, Sigmoid,
//!    Div with a divisor that is sign-definite over `B` — enforced by the
//!    value channel, which FAILS the whole walk otherwise), an exact index
//!    movement (Reshape, Slice, Squeeze, Unsqueeze, Transpose, Concat,
//!    Gather-with-constant-indices — linear, hence analytic), or ReLU. By
//!    induction over the finitely many DAG nodes, every node value along the
//!    segment is piecewise analytic in `t`: analytic ops preserve piecewise
//!    analyticity, and `relu(u)` with `u` piecewise analytic subdivides each
//!    piece at the (finitely many) zeros of `u` — or `u ≡ 0` on a piece and
//!    `relu(u) ≡ 0` there. So there are `0 = t_0 < … < t_N = 1` such that on
//!    each open piece `(t_j, t_{j+1})` every node value is analytic and every
//!    ReLU argument has constant sign (or is identically zero).
//!
//! 2. *Per-piece chain rule is enclosed.* On one piece, `f` coincides with
//!    the smooth function obtained by fixing every ReLU to its active branch,
//!    so the classical chain rule applies: `g'(t) = Σ_i (∂f_branch/∂x_i)(γ(t))
//!    · (x_i − m_i)`. The interval forward-mode rules below compute, at every
//!    node, an interval enclosing that branch-fixed partial derivative at
//!    every point of `B` (hence at `γ(t) ∈ B`):
//!    - linear ops map derivatives linearly (same Higham-widened kernels as
//!      the value channel, bias excluded);
//!    - `Mul`: `(ab)' = a'b + ab'` with interval products of the (sound) value
//!      enclosures — inclusion-monotone;
//!    - `Div`: `(a/b)' = (a'b − ab')/b²`, `b` sign-definite so `b² > 0`; the
//!      independent-product hull of `b·b` contains `{b² : b ∈ [b_l, b_u]}`;
//!    - `Sigmoid`: `σ' = σ(1−σ)` evaluated on the sound sigmoid VALUE
//!      enclosure, intersected with σ's global derivative range `[0, 1/4]`;
//!    - `ReLU`: value `> 0` on `B` ⇒ multiplier 1 (identity on `B`); value
//!      `≤ 0` on `B` ⇒ `relu ≡ const 0` on `B` ⇒ derivative 0; straddling ⇒
//!      the branch multiplier is 0 or 1 on each piece, so the contribution is
//!      contained in `hull(0·d, 1·d) = [min(d_l, 0), max(d_u, 0)]`.
//!    Every non-exact rule is widened outward exactly like the value channel
//!    (1 ulp per elementwise endpoint, Higham `gamma_n` for dot products), so
//!    each `D_i` encloses the branch-fixed real partial on ALL of `B`, for
//!    EVERY branch-fixing that is active somewhere on `B`. Hence
//!    `g'(t) ∈ T := Σ_i D_i · (x_i − m_i)` (a fixed interval) on every piece.
//!
//! 3. *Mean value composition.* `g` is continuous on `[0, 1]` (all supported
//!    ops are continuous; discontinuous layers — Trunc, ArgMax, CompareTensor,
//!    ScatterND — are REJECTED, fail-closed) and differentiable on each open
//!    piece, so the classical MVT per piece gives `g(t_{j+1}) − g(t_j) =
//!    g'(ξ_j)·(t_{j+1} − t_j)` with `g'(ξ_j) ∈ T`. Summing telescopically with
//!    the nonnegative weights `t_{j+1} − t_j` (total 1) and using convexity of
//!    the interval `T`: `f(x) − f(m) = g(1) − g(0) ∈ T`.
//!
//! Therefore `f(x) ∈ f(m) + Σ_i D_i·(x_i − m_i)` for every `x ∈ B`, where
//! `f(m)` is enclosed by the (ulp-tight) point-box cell forward and the sum is
//! evaluated in outward-rounded f64 interval arithmetic with
//! `(x_i − m_i) ∈ [lo_i − m_i, hi_i − m_i]` (outward-rounded, containing 0).
//! The final enclosure is intersected with the zeroth-order interval — the
//! intersection of two sound enclosures is sound, and an EMPTY intersection
//! (impossible for two sound enclosures of a nonempty image) fails the walk
//! rather than certifying anything.
//!
//! ## Sectioned centering: ulp-narrow axes are absorbed, not seeded
//!
//! The screen's box tightening rounds per-clause f64 bounds OUTWARD to f32,
//! so nominal point axes arrive 1-2 f32-ulps wide — an mscn clause box has
//! hundreds of such axes but only 1-3 REAL sweep axes. Seeding every
//! ulp-wide axis would cost one derivative channel each for no measurable
//! tightening, so only axes wider than [`ABSORB_REL_WIDTH`] (relative to the
//! axis magnitude) are seeded; the rest stay as intervals in the CENTER
//! evaluation. Soundness (sectioned MVT): fix `x ∈ B` and move ONLY the
//! seeded coordinates `S` along `γ(t) = (m_S + t(x_S − m_S); x_T)` with the
//! unseeded coordinates `x_T` held fixed. The argument above applies verbatim
//! to this segment (it stays inside `B`, and `D_i` enclose the partials over
//! ALL of `B`), giving `f(x) ∈ f(m_S; x_T) + Σ_{i∈S} D_i·(x_i − m_i)`, and
//! `f(m_S; x_T)` for EVERY `x_T` is enclosed by the cell forward over the box
//! with seeded axes pinned to `m` and unseeded axes left at their (narrow)
//! intervals. The absorbed axes contribute zeroth-order (linear) width — at
//! ulp scale that is orders of magnitude below the 1e-5 mscn band margins.
//!
//! # Fail-closed contract
//!
//! [`GraphNetwork::supports_ibp_f64_centered`] admits ONLY the op set above.
//! Everything else — including all Lipschitz-but-unimplemented ops (Clip,
//! Min/Max, MaxPool2d, Resize, Conv2d) and every discontinuous op (Trunc,
//! ArgMax, CompareTensor, ScatterND), for which a mean-value form would be
//! UNSOUND — makes the gate return `false` and the walk return `Err`. Any
//! error leaves the caller with the zeroth-order bound only.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use crate::layers::Layer;

use super::core::graph::{GraphNetwork, GraphNode, NETWORK_INPUT};
use super::graph_ibp_f64_batch::{eval_linear_stacked_prepared, F64WeightCache};
use super::graph_ibp_f64_cell::{
    broadcast_binary, eval_concat, eval_linear_with_bias, eval_matmul, eval_node, eval_reduce_sum,
    interval_mul, stable_sigmoid_f64, widen1, widen_down_n, widen_up_n, Interval64,
    TRANSCENDENTAL_ULPS,
};

/// Maximum boxes per batched centered-form chunk (#f64-batch-boxes): the
/// batched mean-value walk holds one `Channels` (value + k derivative
/// tensors) per live node per box, so an unbounded wave of 2048-wide mscn
/// boxes would hold GBs; 96 boxes keep the stacked Linear rows far above
/// the fast-kernel gate at a bounded footprint (measured ~hundreds of MB
/// live at the 2048-wide mscn dual shapes: ~(1+k)·2 row-tensors per box
/// per ~4 live nodes) while amortizing the per-chunk walk overhead
/// (HashMaps, rayon dispatch, weight-cache lookups) over 3x more boxes
/// than the original 32.
const MAX_CENTERED_BATCH: usize = 96;

/// Boxes per batched MONO-CORNER cell walk (#mono-corner × #f64-batch-boxes):
/// corner walks carry ONE value channel (no derivatives), so they afford a
/// larger chunk than the (1+k)-channel centered walk at the same footprint.
const MONO_CORNER_CELL_BATCH: usize = 256;

/// Value interval plus one derivative interval per seed axis, all over the
/// same box. `derivs[s]` has the node's output shape and encloses
/// `∂ node / ∂ input[seed_s]` at every point of the box (in the branch-fixed
/// sense of the module-level soundness argument).
struct Channels {
    value: Interval64,
    derivs: Vec<Interval64>,
}

/// Axes at least this wide RELATIVE to their magnitude (`max(1, |lo|, |hi|)`)
/// are seeded with a derivative channel; narrower axes are absorbed into the
/// center evaluation (see "Sectioned centering" in the module docs). 1e-6 is
/// ~8 f32 ulps at unit scale: comfortably above the outward f64→f32 box
/// rounding that widens nominal point axes (1-2 ulps), and the absorbed
/// zeroth-order contribution `|∂f/∂x_i|·width` stays orders of magnitude
/// below the 1e-5 mscn band margins for any plausible derivative size.
const ABSORB_REL_WIDTH: f64 = 1e-6;

/// Whether one axis `[lo, hi]` is wide enough to deserve a derivative seed.
#[inline]
fn axis_is_seeded(lo: f64, hi: f64) -> bool {
    hi - lo > ABSORB_REL_WIDTH * lo.abs().max(hi.abs()).max(1.0)
}

/// Cap on distinct per-output-element derivative SIGN PATTERNS the
/// mono-corner refinement (#mono-corner) will evaluate — each pattern costs
/// two extra cell walks (min corner + max corner). The nn4sys outputs this
/// exists for have 1-2 elements (1-2 patterns); exceeding the cap returns no
/// mono bound (fail-open: the caller keeps the centered bound).
const MONO_MAX_PATTERNS: usize = 8;

/// Result of the fused zeroth + centered + monotonicity-corner walk
/// (#mono-corner; [`GraphNetwork::propagate_ibp_f64_centered_mono`]).
pub struct CenteredMono {
    /// Zeroth-order value channel — bit-identical to
    /// [`GraphNetwork::propagate_ibp_f64_cell`] over the same box.
    pub value: Interval64,
    /// Centered (mean-value) bound intersected with `value` — bit-identical
    /// to [`GraphNetwork::propagate_ibp_f64_centered_with_value`]'s second
    /// element.
    pub centered: Interval64,
    /// Monotonicity-corner bound intersected with `centered`, present when
    /// at least one (output element, seeded axis) pair certified a
    /// derivative sign. `None` = nothing certified / pattern cap exceeded /
    /// corner walk failed — callers keep `centered` (fail-open, never less
    /// tight than before).
    pub mono: Option<Interval64>,
    /// Number of seeded input axes (derivative channels).
    pub seeded_axes: usize,
    /// (output element, seeded axis) pairs whose derivative enclosure
    /// certified a constant sign over the whole box, and the total pair
    /// count — the measured certification rate is `certified_pairs /
    /// total_pairs`.
    pub certified_pairs: usize,
    pub total_pairs: usize,
    /// True when EVERY pair certified: the mono bound is the exact range up
    /// to corner-eval rounding and absorbed ulp-narrow-axis width.
    pub all_certified: bool,
}

/// Number of axes the centered form would seed for this (f32) box — the
/// screen's cost gate MUST use this so it agrees with the seeding rule in
/// [`GraphNetwork::propagate_ibp_f64_centered`] (f32→f64 is exact).
pub fn centered_seed_axes_f32(lo: &[f32], hi: &[f32]) -> usize {
    lo.iter()
        .zip(hi.iter())
        .filter(|(&l, &h)| axis_is_seeded(f64::from(l), f64::from(h)))
        .count()
}

/// Sound real interval quotient `[al, ah] / [bl, bh]`; errors (fail-closed)
/// when the divisor interval contains zero. Mirrors the value channel's Div.
fn interval_div(al: f64, ah: f64, bl: f64, bh: f64) -> Result<(f64, f64)> {
    if bl <= 0.0 && bh >= 0.0 {
        return Err(NyError::InvalidSpec(
            "f64 mvf: divisor interval contains zero".to_string(),
        ));
    }
    let candidates = [al / bl, al / bh, ah / bl, ah / bh];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    Ok((lo, hi))
}

impl GraphNetwork {
    /// Whether every output-ancestor node is in the op set supported by the
    /// centered-form walk ([`Self::propagate_ibp_f64_centered`]). KEEP IN
    /// SYNC with `eval_node_derivs`: anything not provably enclosable by the
    /// mean-value argument (see module docs) must return `false`.
    pub fn supports_ibp_f64_centered(&self) -> bool {
        match self.output_ancestors() {
            Ok(needed) => needed.iter().all(|name| {
                self.node(name)
                    .is_some_and(|node| centered_supports_layer(node.layer()))
            }),
            Err(_) => false,
        }
    }

    /// Sound f64 FIRST-ORDER output enclosure over the input box: the
    /// mean-value/centered form `f(m) ⊕ Σ_i D_i·[lo_i − m_i, hi_i − m_i]`
    /// over the SEEDED axes (width above [`ABSORB_REL_WIDTH`]; narrower axes
    /// are absorbed into the center evaluation — sectioned MVT, see module
    /// docs), intersected with the zeroth-order interval (both sound ⇒
    /// intersection sound). A box with no seedable axis degenerates to the
    /// zeroth-order cell forward. Fails closed (`UnsupportedOp`/
    /// `InvalidSpec`) on unsupported ops, non-finite boxes, or an
    /// (impossible-if-sound) empty intersection.
    ///
    /// Cost is ~`(k + 2)` cell-forward walks for `k` seeded axes — callers
    /// should gate on a small [`centered_seed_axes_f32`] (the mscn `_dual`
    /// clause boxes have 1-3 sweep axes).
    pub fn propagate_ibp_f64_centered(&self, input: &Interval64) -> Result<Interval64> {
        self.propagate_ibp_f64_centered_with_value(input)
            .map(|(_, centered)| centered)
    }

    /// [`Self::propagate_ibp_f64_centered`] returning ALSO the walk's
    /// zeroth-order VALUE channel as the first tuple element (#f64-fused-walk).
    ///
    /// The value channel applies EXACTLY the zeroth-order cell rules
    /// ([`eval_node`], same `exec_order` walk over the same output-ancestor
    /// set), so it is BIT-IDENTICAL to a separate
    /// [`GraphNetwork::propagate_ibp_f64_cell`] over the same box (gate test
    /// `centered_with_value_matches_separate_walks`). Callers that need both
    /// the zeroth-order and the centered bound (the box-refinement screen
    /// bounds every node with the pair) pay ONE mean-value walk instead of a
    /// cell walk PLUS a centered walk that re-derives the identical value
    /// channel internally: 2+k value-equivalent passes per node instead of
    /// 3+k for k seeded axes.
    pub fn propagate_ibp_f64_centered_with_value(
        &self,
        input: &Interval64,
    ) -> Result<(Interval64, Interval64)> {
        self.centered_walk_full(input, false)
            .map(|out| (out.value, out.centered))
    }

    /// Fused zeroth + centered + MONOTONICITY-CORNER walk (#mono-corner):
    /// everything [`Self::propagate_ibp_f64_centered_with_value`] computes
    /// (bit-identical `value` and `centered`), plus — when the interval
    /// forward-mode derivative channels certify a constant sign for some
    /// (output element, seeded axis) pairs — a corner-pinned bound that is
    /// EXACT (up to sound eval rounding and absorbed ulp-narrow axes) on
    /// fully-certified elements.
    ///
    /// # Soundness (corner bound)
    ///
    /// Fix an output element `f = f_j`, the box `B`, and the derivative
    /// enclosures `D_s ⊇ {∂f_branch/∂x_s (p) : p ∈ B, branch active at p}`
    /// established by the module-level mean-value argument. Call seeded axis
    /// `s` POSITIVE-certified when `D_s ⊆ [0, ∞)` and NEGATIVE-certified
    /// when `D_s ⊆ (−∞, 0]` (weak inequalities suffice: certification
    /// claims monotone non-decreasing/non-increasing, not strict). Build
    /// the MIN corner box `B⁻` by pinning every POSITIVE axis to its lower
    /// endpoint and every NEGATIVE axis to its upper endpoint, leaving
    /// mixed-sign seeded axes AND unseeded (ulp-narrow) axes at their full
    /// intervals; `B⁺` pins the opposite endpoints.
    ///
    /// For any `x ∈ B`, apply the sectioned mean-value argument (module
    /// docs) moving ONLY the certified coordinates `S` from their pinned
    /// corner values `c_S` to `x_S`, holding every other coordinate at
    /// `x`'s value — the segment stays in `B`, so
    /// `f(x) − f(c_S; x_T) ∈ Σ_{s∈S} D_s·(x_s − c_s)`. Each term has
    /// `D_s ⊆ [0,∞)` with `x_s − lo_s ≥ 0`, or `D_s ⊆ (−∞,0]` with
    /// `x_s − hi_s ≤ 0` — every product set lies in `[0, ∞)`, so
    /// `f(x) ≥ f(c_S; x_T)`. The point `(c_S; x_T)` lies in `B⁻` for every
    /// `x`, and the (sound, outward-rounded) cell forward over `B⁻`
    /// encloses `f` at every point of `B⁻`, hence
    /// `min_B f ≥ lower(cell(B⁻))`. Symmetrically `max_B f ≤
    /// upper(cell(B⁺))`. Elements sharing one sign pattern share the two
    /// corner walks. The final mono interval is intersected with `centered`
    /// (both sound ⇒ intersection sound; empty ⇒ fail closed, `Err`).
    ///
    /// Fail-open contract: any reason NOT to produce the corner bound
    /// (nothing certified, more than [`MONO_MAX_PATTERNS`] sign patterns, a
    /// failed corner walk) yields `mono: None` with `value`/`centered`
    /// untouched — the caller is never worse off than the centered form.
    pub fn propagate_ibp_f64_centered_mono(&self, input: &Interval64) -> Result<CenteredMono> {
        self.centered_walk_full(input, true)
    }

    /// Shared implementation of the centered walk, optionally extended with
    /// the monotonicity-corner refinement (#mono-corner).
    fn centered_walk_full(&self, input: &Interval64, want_mono: bool) -> Result<CenteredMono> {
        let lo_std = input.lower.as_standard_layout();
        let hi_std = input.upper.as_standard_layout();
        let (lo, hi) = match (lo_std.as_slice(), hi_std.as_slice()) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => {
                return Err(NyError::InvalidSpec(
                    "f64 mvf: input box not contiguous".to_string(),
                ))
            }
        };
        for (&l, &h) in lo.iter().zip(hi.iter()) {
            if !(l.is_finite() && h.is_finite() && l <= h) {
                return Err(NyError::InvalidSpec(
                    "f64 mvf: input box must be finite and ordered".to_string(),
                ));
            }
        }
        // Seed only meaningfully wide axes; ulp-narrow axes (outward f64→f32
        // box rounding of nominal point axes) are absorbed into the center
        // evaluation — see "Sectioned centering" in the module docs.
        let seeds: Vec<usize> = (0..lo.len())
            .filter(|&i| axis_is_seeded(lo[i], hi[i]))
            .collect();
        if seeds.is_empty() {
            // Nothing to center over: the zeroth-order forward is already
            // (near-)point-tight on this box.
            let cell = self.propagate_ibp_f64_cell(input)?;
            return Ok(CenteredMono {
                value: cell.clone(),
                centered: cell,
                mono: None,
                seeded_axes: 0,
                certified_pairs: 0,
                total_pairs: 0,
                all_certified: false,
            });
        }

        // Interval forward-mode AD over the box: value channel + one
        // derivative channel per seeded axis.
        let mv = self.propagate_ibp_f64_mean_value(input, &seeds)?;

        // Center m: the f64 midpoint clamped into the box (ANY interior point
        // is a valid center for the MVT — exactness of m is not required).
        let shape = input.lower.shape().to_vec();
        let mut mid = lo.to_vec();
        for &s in &seeds {
            // Kept verbatim: raw f64 bounds may exceed f64::MAX/2, where
            // f64::midpoint rounds differently; the clamped ±inf center must
            // keep collapsing to the box endpoint.
            #[allow(clippy::manual_midpoint)]
            let m = 0.5 * (lo[s] + hi[s]);
            mid[s] = m.clamp(lo[s], hi[s]);
        }
        // Center box: seeded axes pinned to m, unseeded axes kept at their
        // (narrow) intervals — encloses f(m_S; x_T) for EVERY x_T (sectioned
        // MVT). For an all-seeded box this is exactly the point forward.
        let mut center_lo = lo.to_vec();
        let mut center_hi = hi.to_vec();
        for &s in &seeds {
            center_lo[s] = mid[s];
            center_hi[s] = mid[s];
        }
        let center_box = Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(&shape), center_lo)
                .map_err(|e| NyError::InvalidSpec(format!("f64 mvf: center tensor: {e}")))?,
            upper: ArrayD::from_shape_vec(IxDyn(&shape), center_hi)
                .map_err(|e| NyError::InvalidSpec(format!("f64 mvf: center tensor: {e}")))?,
        };
        let point = self.propagate_ibp_f64_cell(&center_box)?;

        let centered = combine_centered(lo, hi, &mid, &seeds, &mv, &point)?;

        // Monotonicity-corner refinement (#mono-corner): classify the sign
        // of every (output element, seeded axis) derivative enclosure, and
        // bound sign-certified elements by sound cell walks over their
        // pinned corner boxes (see `propagate_ibp_f64_centered_mono` for
        // the enclosure argument). Everything below is fail-open: `mono`
        // stays `None` unless a corner bound was soundly produced.
        let mut certified_pairs = 0usize;
        let mut total_pairs = 0usize;
        let mut mono: Option<Interval64> = None;
        if want_mono {
            if let Some((m, cert, total)) =
                self.mono_corner_bound(lo, hi, &shape, &seeds, &mv, &centered)?
            {
                certified_pairs = cert;
                total_pairs = total;
                mono = Some(m);
            }
        }
        let all_certified = total_pairs > 0 && certified_pairs == total_pairs;
        Ok(CenteredMono {
            value: mv.value,
            centered,
            mono,
            seeded_axes: seeds.len(),
            certified_pairs,
            total_pairs,
            all_certified,
        })
    }

    /// The corner-pinning step of the mono-corner refinement: classify
    /// derivative signs, group output elements by sign pattern, run the two
    /// sound corner cell walks per pattern, and intersect with `centered`.
    ///
    /// Returns `Ok(None)` when no pair certified or the pattern cap was
    /// exceeded (fail-open); `Err` ONLY on an empty mono∩centered
    /// intersection (fail-closed: two sound enclosures cannot be disjoint,
    /// so certify nothing). A failed corner walk also fails open to `None`.
    ///
    /// Shares [`classify_sign_groups`] / [`corner_boxes_for_pattern`] /
    /// [`write_corner_endpoints`] / [`intersect_mono_with_centered`] with
    /// the batched multi-box lane so the two cannot drift.
    #[allow(clippy::type_complexity)]
    fn mono_corner_bound(
        &self,
        lo: &[f64],
        hi: &[f64],
        shape: &[usize],
        seeds: &[usize],
        mv: &Channels,
        centered: &Interval64,
    ) -> Result<Option<(Interval64, usize, usize)>> {
        let out_len: usize = centered.lower.len();
        let Some(sg) = classify_sign_groups(mv, seeds.len(), out_len) else {
            return Ok(None); // nothing certified / cap / layout — fail open
        };

        // Corner walks per pattern; elements not covered by any pattern
        // keep ±inf (the intersection with `centered` leaves them there).
        let mut m_lo = vec![f64::NEG_INFINITY; out_len];
        let mut m_hi = vec![f64::INFINITY; out_len];
        for (pat, elems) in &sg.groups {
            let (min_box, max_box) = corner_boxes_for_pattern(lo, hi, shape, seeds, pat)?;
            let (min_out, max_out) = match (
                self.propagate_ibp_f64_cell(&min_box),
                self.propagate_ibp_f64_cell(&max_box),
            ) {
                (Ok(a), Ok(b)) => (a, b),
                _ => return Ok(None), // corner walk failed — fail open
            };
            if !write_corner_endpoints(&min_out, &max_out, elems, &mut m_lo, &mut m_hi) {
                return Ok(None); // fail open
            }
        }

        let Some(mono) = intersect_mono_with_centered(&mut m_lo, &mut m_hi, centered)? else {
            return Ok(None); // non-contiguous centered — fail open
        };
        Ok(Some((mono, sg.certified_pairs, sg.total_pairs)))
    }

    /// Batched multi-box centered form (#f64-batch-boxes × #f64-mvf): the
    /// mean-value/centered bound of [`Self::propagate_ibp_f64_centered`] for
    /// W INDEPENDENT boxes, evaluated so that every Linear sees a FAT
    /// stacked interval GEMM (boxes × derivative channels × rows) instead of
    /// W·(k+2) thin ones — the Rump kernel (#f64-blas-gemm) fires at mscn's
    /// per-box shapes only through this stacking.
    ///
    /// Boxes are grouped by their (identical) seeded-axis set — the seeding
    /// rule is per box, so different boxes may legitimately seed different
    /// axes — and each group is processed in chunks of
    /// [`MAX_CENTERED_BATCH`] boxes (memory: one `Channels` per node per
    /// box). Groups with NO seedable axis degenerate to the batched
    /// zeroth-order forward, mirroring the per-box entry.
    ///
    /// Per box, the result is bit-identical to the per-box entry whenever
    /// kernel selection agrees (thin stacks), and a sound containing
    /// superset when the stacked shapes promote the fast kernel (every
    /// downstream combination step is inclusion-monotone, and the final
    /// intersection with the box's own zeroth-order value channel keeps it
    /// a refinement). Cross-box isolation is structural: stacked Linear
    /// rows never mix, everything else evaluates per box.
    ///
    /// Fail-closed: ANY error fails the WHOLE call; callers must fall back
    /// to per-box [`Self::propagate_ibp_f64_centered`] walks.
    pub fn propagate_ibp_f64_centered_cells(
        &self,
        inputs: &[Interval64],
    ) -> Result<Vec<Interval64>> {
        self.propagate_ibp_f64_centered_cells_cached(inputs, None)
    }

    /// [`Self::propagate_ibp_f64_centered_cells`] with an optional
    /// prepared-weight cache ([`Self::build_f64_weight_cache`]) —
    /// bit-identical results, skips the per-call f64 weight conversion.
    pub fn propagate_ibp_f64_centered_cells_cached(
        &self,
        inputs: &[Interval64],
        weights: Option<&F64WeightCache>,
    ) -> Result<Vec<Interval64>> {
        Ok(self
            .propagate_ibp_f64_centered_mono_cells_cached(inputs, false, weights)?
            .into_iter()
            .map(|o| o.centered)
            .collect())
    }

    /// Batched multi-box FUSED zeroth + centered + mono walk
    /// (#f64-fused-walk × #f64-batch-boxes × #mono-corner): one
    /// [`CenteredMono`] per input box, from chunked multi-box walks that
    /// stack every Linear into fat interval GEMMs.
    ///
    /// Per box:
    /// - `value` is the zeroth-order channel of the batched mean-value walk
    ///   — BIT-IDENTICAL to [`Self::propagate_ibp_f64_cells_cached`] over
    ///   the same chunk (identical stacked Linear evaluations, identical
    ///   per-box op rules; gate `batched_fused_value_matches_batched_cells`)
    ///   — so callers that previously ran a separate batched zeroth pass
    ///   plus this centered pass can DROP the zeroth pass;
    /// - `centered` is bit-identical to
    ///   [`Self::propagate_ibp_f64_centered_cells_cached`]'s result;
    /// - `mono` (when `want_mono`, #mono-corner) applies the per-box
    ///   mono-corner refinement with the PATTERN CORNER WALKS STACKED
    ///   across the whole chunk into batched cell walks (~2 extra batched
    ///   walks per chunk instead of 2·W thin per-box walks). Soundness is
    ///   the per-box argument verbatim: sign certification from the (sound,
    ///   possibly batch-widened — certification only ever FAILS more often,
    ///   fail-open) derivative channels, corners evaluated by the sound
    ///   batched cell walk, intersected with `centered`. Per box the result
    ///   equals the per-box [`Self::propagate_ibp_f64_centered_mono`] `mono`
    ///   whenever kernel selection agrees, and a sound containing superset
    ///   otherwise (gate `batched_mono_matches_per_box_when_kernels_agree`).
    ///
    /// Fail-closed: ANY error fails the WHOLE call (callers fall back to
    /// per-box walks); a failed/declined mono stage fails OPEN per box
    /// (`mono: None`) exactly like the per-box lane.
    pub fn propagate_ibp_f64_centered_mono_cells_cached(
        &self,
        inputs: &[Interval64],
        want_mono: bool,
        weights: Option<&F64WeightCache>,
    ) -> Result<Vec<CenteredMono>> {
        let w = inputs.len();
        if w == 0 {
            return Ok(Vec::new());
        }
        let in_shape = inputs[0].lower.shape().to_vec();
        // Group boxes by identical seed set (order-insensitive by
        // construction: seeds are collected in ascending axis order).
        let mut groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for (b, x) in inputs.iter().enumerate() {
            if x.lower.shape() != in_shape.as_slice() || x.upper.shape() != in_shape.as_slice() {
                return Err(NyError::InvalidSpec(
                    "f64 mvf batch: input boxes must share one shape".to_string(),
                ));
            }
            let lo_std = x.lower.as_standard_layout();
            let hi_std = x.upper.as_standard_layout();
            let (lo, hi) = match (lo_std.as_slice(), hi_std.as_slice()) {
                (Some(lo), Some(hi)) => (lo, hi),
                _ => {
                    return Err(NyError::InvalidSpec(
                        "f64 mvf batch: input box not contiguous".to_string(),
                    ))
                }
            };
            for (&l, &h) in lo.iter().zip(hi.iter()) {
                if !(l.is_finite() && h.is_finite() && l <= h) {
                    return Err(NyError::InvalidSpec(
                        "f64 mvf batch: input box must be finite and ordered".to_string(),
                    ));
                }
            }
            let seeds: Vec<usize> = (0..lo.len())
                .filter(|&i| axis_is_seeded(lo[i], hi[i]))
                .collect();
            groups.entry(seeds).or_default().push(b);
        }

        let mut out: Vec<Option<CenteredMono>> = (0..w).map(|_| None).collect();
        for (seeds, idxs) in groups {
            for chunk in idxs.chunks(MAX_CENTERED_BATCH) {
                let boxes: Vec<Interval64> = chunk.iter().map(|&b| inputs[b].clone()).collect();
                let results = if seeds.is_empty() {
                    // No seedable axis: the zeroth-order forward is already
                    // (near-)point-tight — same degeneration as the per-box
                    // entry, batched.
                    self.propagate_ibp_f64_cells_cached(&boxes, weights)?
                        .into_iter()
                        .map(|cell| CenteredMono {
                            value: cell.clone(),
                            centered: cell,
                            mono: None,
                            seeded_axes: 0,
                            certified_pairs: 0,
                            total_pairs: 0,
                            all_certified: false,
                        })
                        .collect()
                } else {
                    self.centered_mono_cells_group(&boxes, &seeds, want_mono, weights)?
                };
                if results.len() != chunk.len() {
                    return Err(NyError::InvalidSpec(
                        "f64 mvf batch: group result count mismatch".to_string(),
                    ));
                }
                for (&b, r) in chunk.iter().zip(results) {
                    out[b] = Some(r);
                }
            }
        }
        out.into_iter()
            .map(|o| {
                o.ok_or_else(|| {
                    NyError::InvalidSpec("f64 mvf batch: box left unbounded".to_string())
                })
            })
            .collect()
    }

    /// One seed-group chunk of the batched fused walk: batched mean-value
    /// walk, batched center-point walk, the per-box combination shared
    /// bit-for-bit with the per-box entry ([`combine_centered`]), then —
    /// when `want_mono` — the mono-corner stage with all pattern corner
    /// boxes of the chunk stacked into batched cell walks.
    fn centered_mono_cells_group(
        &self,
        boxes: &[Interval64],
        seeds: &[usize],
        want_mono: bool,
        weights: Option<&F64WeightCache>,
    ) -> Result<Vec<CenteredMono>> {
        let mv = self.propagate_mean_value_batched(boxes, seeds, weights)?;

        // Per-box centers and center boxes (seeded axes pinned to the box's
        // OWN midpoint; unseeded axes keep their narrow intervals —
        // sectioned MVT, module docs).
        let shape = boxes[0].lower.shape().to_vec();
        let mut los: Vec<Vec<f64>> = Vec::with_capacity(boxes.len());
        let mut his: Vec<Vec<f64>> = Vec::with_capacity(boxes.len());
        let mut mids: Vec<Vec<f64>> = Vec::with_capacity(boxes.len());
        let mut center_boxes: Vec<Interval64> = Vec::with_capacity(boxes.len());
        for x in boxes {
            let lo_std = x.lower.as_standard_layout();
            let hi_std = x.upper.as_standard_layout();
            let (lo, hi) = match (lo_std.as_slice(), hi_std.as_slice()) {
                (Some(lo), Some(hi)) => (lo, hi),
                _ => {
                    return Err(NyError::InvalidSpec(
                        "f64 mvf batch: input box not contiguous".to_string(),
                    ))
                }
            };
            let mut mid = lo.to_vec();
            for &s in seeds {
                // Kept verbatim: raw f64 bounds may exceed f64::MAX/2, where
                // f64::midpoint rounds differently; the clamped ±inf center must
                // keep collapsing to the box endpoint.
                #[allow(clippy::manual_midpoint)]
                let m = 0.5 * (lo[s] + hi[s]);
                mid[s] = m.clamp(lo[s], hi[s]);
            }
            let mut center_lo = lo.to_vec();
            let mut center_hi = hi.to_vec();
            for &s in seeds {
                center_lo[s] = mid[s];
                center_hi[s] = mid[s];
            }
            center_boxes.push(Interval64 {
                lower: ArrayD::from_shape_vec(IxDyn(&shape), center_lo).map_err(|e| {
                    NyError::InvalidSpec(format!("f64 mvf batch: center tensor: {e}"))
                })?,
                upper: ArrayD::from_shape_vec(IxDyn(&shape), center_hi).map_err(|e| {
                    NyError::InvalidSpec(format!("f64 mvf batch: center tensor: {e}"))
                })?,
            });
            los.push(lo.to_vec());
            his.push(hi.to_vec());
            mids.push(mid);
        }
        let points = self.propagate_ibp_f64_cells_cached(&center_boxes, weights)?;

        let centereds: Vec<Interval64> = (0..boxes.len())
            .map(|b| combine_centered(&los[b], &his[b], &mids[b], seeds, &mv[b], &points[b]))
            .collect::<Result<Vec<_>>>()?;

        // Mono-corner stage (#mono-corner × #f64-batch-boxes): classify per
        // box, stack every pattern's two corner boxes across the chunk, and
        // bound them in chunked batched cell walks. Everything here is
        // fail-open per box except the (bug-tripwire) empty mono∩centered
        // intersection, which fails the whole call closed — mirroring the
        // per-box lane.
        let mut monos: Vec<Option<Interval64>> = vec![None; boxes.len()];
        let mut stats: Vec<(usize, usize)> = vec![(0, 0); boxes.len()];
        if want_mono {
            struct BoxPlan {
                box_idx: usize,
                /// Patterns in deterministic order; walk slots
                /// `first_walk + 2p` (min) and `first_walk + 2p + 1` (max).
                groups: Vec<(Vec<u8>, Vec<usize>)>,
                first_walk: usize,
                certified_pairs: usize,
                total_pairs: usize,
            }
            let mut plans: Vec<BoxPlan> = Vec::new();
            let mut corner_boxes: Vec<Interval64> = Vec::new();
            for b in 0..boxes.len() {
                let out_len = centereds[b].lower.len();
                let Some(sg) = classify_sign_groups(&mv[b], seeds.len(), out_len) else {
                    continue; // fail open: this box keeps mono = None
                };
                let mut groups: Vec<(Vec<u8>, Vec<usize>)> = sg.groups.into_iter().collect();
                groups.sort_by(|a, b| a.0.cmp(&b.0));
                let first_walk = corner_boxes.len();
                for (pat, _) in &groups {
                    let (min_box, max_box) =
                        corner_boxes_for_pattern(&los[b], &his[b], &shape, seeds, pat)?;
                    corner_boxes.push(min_box);
                    corner_boxes.push(max_box);
                }
                plans.push(BoxPlan {
                    box_idx: b,
                    groups,
                    first_walk,
                    certified_pairs: sg.certified_pairs,
                    total_pairs: sg.total_pairs,
                });
            }

            let mut walk_outs: Vec<Option<Interval64>> = vec![None; corner_boxes.len()];
            for (chunk_idx, chunk) in corner_boxes.chunks(MONO_CORNER_CELL_BATCH).enumerate() {
                let base = chunk_idx * MONO_CORNER_CELL_BATCH;
                if let Ok(outs) = self.propagate_ibp_f64_cells_cached(chunk, weights) {
                    for (off, out) in outs.into_iter().enumerate() {
                        walk_outs[base + off] = Some(out);
                    }
                }
                // Err: those corner walks stay None — their boxes fail open.
            }

            for plan in plans {
                let b = plan.box_idx;
                let out_len = centereds[b].lower.len();
                let mut m_lo = vec![f64::NEG_INFINITY; out_len];
                let mut m_hi = vec![f64::INFINITY; out_len];
                let mut ok = true;
                for (p, (_, elems)) in plan.groups.iter().enumerate() {
                    let (min_out, max_out) = match (
                        walk_outs[plan.first_walk + 2 * p].as_ref(),
                        walk_outs[plan.first_walk + 2 * p + 1].as_ref(),
                    ) {
                        (Some(a), Some(c)) => (a, c),
                        _ => {
                            ok = false; // corner walk failed — fail open
                            break;
                        }
                    };
                    if !write_corner_endpoints(min_out, max_out, elems, &mut m_lo, &mut m_hi) {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                if let Some(mono) =
                    intersect_mono_with_centered(&mut m_lo, &mut m_hi, &centereds[b])?
                {
                    monos[b] = Some(mono);
                    stats[b] = (plan.certified_pairs, plan.total_pairs);
                }
            }
        }

        let mut results = Vec::with_capacity(boxes.len());
        for (b, (mv_b, centered)) in mv.into_iter().zip(centereds).enumerate() {
            let (certified_pairs, total_pairs) = stats[b];
            results.push(CenteredMono {
                value: mv_b.value,
                centered,
                mono: monos[b].take(),
                seeded_axes: seeds.len(),
                certified_pairs,
                total_pairs,
                all_certified: total_pairs > 0 && certified_pairs == total_pairs,
            });
        }
        Ok(results)
    }

    /// Batched interval forward-mode walk: per node, one [`Channels`] per
    /// box. Non-Linear ops run the EXACT per-box rules ([`eval_node`] /
    /// [`eval_node_derivs`]) in parallel across boxes; Linear stacks the
    /// value channels (W·rows) and ALL derivative channels (W·k·rows,
    /// box-major channel-minor) into two fat interval GEMMs through the
    /// shared row-independent stacking helper.
    fn propagate_mean_value_batched(
        &self,
        inputs: &[Interval64],
        seeds: &[usize],
        weights: Option<&F64WeightCache>,
    ) -> Result<Vec<Channels>> {
        let w = inputs.len();
        let n_seeds = seeds.len();
        let needed = self.output_ancestors()?;
        let input_entries: Vec<Channels> = inputs
            .iter()
            .map(|x| build_input_entry(x, seeds))
            .collect::<Result<Vec<_>>>()?;

        // Consumer refcounts (with multiplicity) for eviction — W Channels
        // per live node is the walk's memory footprint.
        let mut remaining: HashMap<&str, usize> = HashMap::new();
        for name in &needed {
            let node = self.node(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 mvf batch: missing node '{name}'"))
            })?;
            for input in node.inputs() {
                if input != NETWORK_INPUT {
                    *remaining.entry(input.as_str()).or_insert(0) += 1;
                }
            }
        }
        *remaining.entry(self.output_name()).or_insert(0) += 1;

        let mut cache: HashMap<&str, Vec<Channels>> = HashMap::new();
        for node_name in self.exec_order()? {
            if !needed.contains(node_name.as_str()) {
                continue;
            }
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 mvf batch: missing node '{node_name}'"))
            })?;

            let outs: Vec<Channels> = match node.layer() {
                Layer::Linear(linear) => {
                    let input_name = node.inputs().first().ok_or_else(|| {
                        NyError::InvalidSpec("f64 mvf batch: Linear missing its input".to_string())
                    })?;
                    let entries: Vec<&Channels> = if input_name == NETWORK_INPUT {
                        input_entries.iter().collect()
                    } else {
                        cache
                            .get(input_name.as_str())
                            .map(|v| v.iter().collect())
                            .ok_or_else(|| {
                                NyError::InvalidSpec(format!(
                                    "f64 mvf batch: '{input_name}' not computed"
                                ))
                            })?
                    };
                    let prepared = weights.and_then(|c| c.get(node_name.as_str()));
                    // Value channels: W stacked rows, bias included.
                    let value_inputs: Vec<Interval64> =
                        entries.iter().map(|e| e.value.clone()).collect();
                    let values =
                        eval_linear_stacked_prepared(linear, &value_inputs, true, prepared)?;
                    // Derivative channels: W·k stacked rows (box-major,
                    // channel-minor), bias EXCLUDED (constants have zero
                    // derivative).
                    let mut deriv_inputs: Vec<Interval64> = Vec::with_capacity(w * n_seeds);
                    for e in &entries {
                        if e.derivs.len() != n_seeds {
                            return Err(NyError::InvalidSpec(
                                "f64 mvf batch: derivative channel count diverged".to_string(),
                            ));
                        }
                        deriv_inputs.extend(e.derivs.iter().cloned());
                    }
                    let mut deriv_outs = if deriv_inputs.is_empty() {
                        Vec::new()
                    } else {
                        eval_linear_stacked_prepared(linear, &deriv_inputs, false, prepared)?
                    };
                    // Reassemble per box.
                    let mut outs = Vec::with_capacity(w);
                    for (b, value) in values.into_iter().enumerate() {
                        let derivs: Vec<Interval64> =
                            deriv_outs.drain(..n_seeds.min(deriv_outs.len())).collect();
                        if derivs.len() != n_seeds {
                            return Err(NyError::InvalidSpec(format!(
                                "f64 mvf batch: box {b} derivative unstack mismatch"
                            )));
                        }
                        outs.push(Channels { value, derivs });
                    }
                    outs
                }
                _ => {
                    let cache_ref = &cache;
                    let entries_ref = &input_entries;
                    (0..w)
                        .into_par_iter()
                        .map(|b| {
                            let resolve_value = |name: &str| -> Result<Interval64> {
                                if name == NETWORK_INPUT {
                                    return Ok(inputs[b].clone());
                                }
                                cache_ref
                                    .get(name)
                                    .and_then(|v| v.get(b))
                                    .map(|e| e.value.clone())
                                    .ok_or_else(|| {
                                        NyError::InvalidSpec(format!(
                                            "f64 mvf batch: '{name}' not computed"
                                        ))
                                    })
                            };
                            let value = eval_node(node.layer(), node, &resolve_value)?;
                            let entry = |name: &str| -> Result<&Channels> {
                                if name == NETWORK_INPUT {
                                    return Ok(&entries_ref[b]);
                                }
                                cache_ref.get(name).and_then(|v| v.get(b)).ok_or_else(|| {
                                    NyError::InvalidSpec(format!(
                                        "f64 mvf batch: '{name}' not computed"
                                    ))
                                })
                            };
                            let derivs = eval_node_derivs(node, n_seeds, &entry)?;
                            Ok(Channels { value, derivs })
                        })
                        .collect::<Result<Vec<Channels>>>()?
                }
            };
            if outs.len() != w {
                return Err(NyError::InvalidSpec(format!(
                    "f64 mvf batch: node '{node_name}' produced {} results for {w} boxes",
                    outs.len()
                )));
            }
            for e in &outs {
                if e.derivs
                    .iter()
                    .any(|d| d.lower.shape() != e.value.lower.shape())
                {
                    return Err(NyError::InvalidSpec(format!(
                        "f64 mvf batch: derivative shape diverged from value at '{node_name}'"
                    )));
                }
            }

            for input in node.inputs() {
                if input == NETWORK_INPUT {
                    continue;
                }
                if let Some(count) = remaining.get_mut(input.as_str()) {
                    *count -= 1;
                    if *count == 0 {
                        cache.remove(input.as_str());
                    }
                }
            }
            cache.insert(node.name(), outs);
        }
        cache.remove(self.output_name()).ok_or_else(|| {
            NyError::InvalidSpec("f64 mvf batch: output node not computed".to_string())
        })
    }

    /// Interval forward-mode walk: per node, the value interval (identical to
    /// the zeroth-order cell forward) plus one derivative interval per seed
    /// axis. `seeds` are flat (standard-layout) indices into the input.
    fn propagate_ibp_f64_mean_value(
        &self,
        input: &Interval64,
        seeds: &[usize],
    ) -> Result<Channels> {
        let needed = self.output_ancestors()?;
        let input_entry = build_input_entry(input, seeds)?;

        let mut cache: HashMap<&str, Channels> = HashMap::new();
        for node_name in self.exec_order()? {
            if !needed.contains(node_name.as_str()) {
                continue;
            }
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 mvf: missing node '{node_name}'"))
            })?;
            // Value channel: EXACTLY the zeroth-order cell rules.
            let resolve_value = |name: &str| -> Result<Interval64> {
                if name == NETWORK_INPUT {
                    return Ok(input.clone());
                }
                cache
                    .get(name)
                    .map(|e| e.value.clone())
                    .ok_or_else(|| NyError::InvalidSpec(format!("f64 mvf: '{name}' not computed")))
            };
            let value = eval_node(node.layer(), node, &resolve_value)?;
            let entry = |name: &str| -> Result<&Channels> {
                if name == NETWORK_INPUT {
                    return Ok(&input_entry);
                }
                cache
                    .get(name)
                    .ok_or_else(|| NyError::InvalidSpec(format!("f64 mvf: '{name}' not computed")))
            };
            let derivs = eval_node_derivs(node, seeds.len(), &entry)?;
            if derivs
                .iter()
                .any(|d| d.lower.shape() != value.lower.shape())
            {
                return Err(NyError::InvalidSpec(format!(
                    "f64 mvf: derivative shape diverged from value at '{node_name}'"
                )));
            }
            cache.insert(node.name(), Channels { value, derivs });
        }
        cache
            .remove(self.output_name())
            .ok_or_else(|| NyError::InvalidSpec("f64 mvf: output node not computed".to_string()))
    }
}

/// Output-element sign-pattern classification of one box's derivative
/// channels (#mono-corner), shared by the per-box and batched lanes.
struct SignGroups {
    /// Sign pattern (one [`POS`]/[`NEG`]/[`MIXED`] byte per seeded axis) →
    /// the output elements carrying it. Only elements with at least one
    /// certified axis appear.
    groups: HashMap<Vec<u8>, Vec<usize>>,
    certified_pairs: usize,
    total_pairs: usize,
}

/// Sign-pattern codes: POSITIVE-certified (min at lo), NEGATIVE-certified
/// (min at hi), mixed (axis stays an interval in the corner walks).
const POS: u8 = 0;
const NEG: u8 = 1;
const MIXED: u8 = 2;

/// Classify the sign of every (output element, seeded axis) derivative
/// enclosure and group elements by their full pattern. `None` = fail open
/// (non-contiguous/mismatched derivative channels, nothing certified, or
/// more than [`MONO_MAX_PATTERNS`] distinct patterns). NaN endpoints fail
/// both weak sign tests and land on `MIXED` — fail-open by construction.
fn classify_sign_groups(mv: &Channels, n_seeds: usize, out_len: usize) -> Option<SignGroups> {
    // Per-seed derivative slices (standard layout, one entry per output
    // element).
    let d_lo_std: Vec<_> = mv
        .derivs
        .iter()
        .map(|d| d.lower.as_standard_layout())
        .collect();
    let d_hi_std: Vec<_> = mv
        .derivs
        .iter()
        .map(|d| d.upper.as_standard_layout())
        .collect();
    let mut d_lo: Vec<&[f64]> = Vec::with_capacity(n_seeds);
    let mut d_hi: Vec<&[f64]> = Vec::with_capacity(n_seeds);
    for (l, h) in d_lo_std.iter().zip(d_hi_std.iter()) {
        match (l.as_slice(), h.as_slice()) {
            (Some(l), Some(h)) => {
                d_lo.push(l);
                d_hi.push(h);
            }
            _ => return None, // non-contiguous derivative — fail open
        }
    }
    if d_lo.len() != n_seeds || d_lo.iter().any(|s| s.len() != out_len) {
        return None; // shape mismatch — fail open
    }

    let mut groups: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
    let mut certified_pairs = 0usize;
    let total_pairs = out_len * n_seeds;
    for j in 0..out_len {
        let mut pat = Vec::with_capacity(n_seeds);
        let mut any = false;
        for k in 0..n_seeds {
            if d_lo[k][j] >= 0.0 {
                pat.push(POS);
                any = true;
                certified_pairs += 1;
            } else if d_hi[k][j] <= 0.0 {
                pat.push(NEG);
                any = true;
                certified_pairs += 1;
            } else {
                pat.push(MIXED);
            }
        }
        if any {
            groups.entry(pat).or_default().push(j);
        }
    }
    if groups.is_empty() || groups.len() > MONO_MAX_PATTERNS {
        return None; // nothing certified / too many patterns — fail open
    }
    Some(SignGroups {
        groups,
        certified_pairs,
        total_pairs,
    })
}

/// The MIN and MAX corner boxes of one sign pattern: POSITIVE axes pin to
/// (lo, hi) respectively, NEGATIVE axes to (hi, lo), mixed/unseeded axes
/// keep their full intervals (see `propagate_ibp_f64_centered_mono` for the
/// enclosure argument).
fn corner_boxes_for_pattern(
    lo: &[f64],
    hi: &[f64],
    shape: &[usize],
    seeds: &[usize],
    pat: &[u8],
) -> Result<(Interval64, Interval64)> {
    let mut min_lo = lo.to_vec();
    let mut min_hi = hi.to_vec();
    let mut max_lo = lo.to_vec();
    let mut max_hi = hi.to_vec();
    for (k, &s) in seeds.iter().enumerate() {
        match pat[k] {
            POS => {
                // Non-decreasing: min at lo_s, max at hi_s.
                min_hi[s] = lo[s];
                max_lo[s] = hi[s];
            }
            NEG => {
                // Non-increasing: min at hi_s, max at lo_s.
                min_lo[s] = hi[s];
                max_hi[s] = lo[s];
            }
            _ => {} // mixed: keep the full interval
        }
    }
    let make_box = |l: Vec<f64>, h: Vec<f64>| -> Result<Interval64> {
        Ok(Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(shape), l)
                .map_err(|e| NyError::InvalidSpec(format!("f64 mono: corner tensor: {e}")))?,
            upper: ArrayD::from_shape_vec(IxDyn(shape), h)
                .map_err(|e| NyError::InvalidSpec(format!("f64 mono: corner tensor: {e}")))?,
        })
    };
    Ok((make_box(min_lo, min_hi)?, make_box(max_lo, max_hi)?))
}

/// Write one pattern's corner-walk endpoints into the per-element
/// accumulators: `m_lo[j] = lower(min walk)`, `m_hi[j] = upper(max walk)`
/// for the pattern's elements. `false` = layout/shape surprise (fail open).
fn write_corner_endpoints(
    min_out: &Interval64,
    max_out: &Interval64,
    elems: &[usize],
    m_lo: &mut [f64],
    m_hi: &mut [f64],
) -> bool {
    let out_len = m_lo.len();
    let min_std = min_out.lower.as_standard_layout();
    let max_std = max_out.upper.as_standard_layout();
    let (min_s, max_s) = match (min_std.as_slice(), max_std.as_slice()) {
        (Some(a), Some(b)) if a.len() == out_len && b.len() == out_len => (a, b),
        _ => return false,
    };
    for &j in elems {
        m_lo[j] = min_s[j];
        m_hi[j] = max_s[j];
    }
    true
}

/// Intersect per-element corner endpoints with the centered bound: both
/// sound, so the intersection is sound and MUST be nonempty (NaN corner
/// endpoints lose the max/min and fall back to the centered endpoint —
/// conservative). `Err` on an empty intersection (fail closed); `Ok(None)`
/// on a non-contiguous centered tensor (fail open).
fn intersect_mono_with_centered(
    m_lo: &mut [f64],
    m_hi: &mut [f64],
    centered: &Interval64,
) -> Result<Option<Interval64>> {
    let c_lo_std = centered.lower.as_standard_layout();
    let c_hi_std = centered.upper.as_standard_layout();
    let (c_lo, c_hi) = match (c_lo_std.as_slice(), c_hi_std.as_slice()) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(None),
    };
    for j in 0..m_lo.len() {
        let l = m_lo[j].max(c_lo[j]);
        let h = m_hi[j].min(c_hi[j]);
        // Negated form is deliberate: NaN endpoints make `l <= h` false, so a
        // NaN intersection FAILS CLOSED here; `l > h` would certify it.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(l <= h) {
            return Err(NyError::InvalidSpec(
                "f64 mono: corner/centered intersection empty — failing closed".to_string(),
            ));
        }
        m_lo[j] = l;
        m_hi[j] = h;
    }
    let out_shape = centered.lower.shape().to_vec();
    Ok(Some(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), m_lo.to_vec())
            .map_err(|e| NyError::InvalidSpec(format!("f64 mono: out lower: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), m_hi.to_vec())
            .map_err(|e| NyError::InvalidSpec(format!("f64 mono: out upper: {e}")))?,
    }))
}

/// Final centered-form combination, shared BIT-FOR-BIT by the per-box entry
/// and the batched multi-box entry: accumulate
/// `point ⊕ Σ_i D_i · [lo_i − m_i, hi_i − m_i]` outward-rounded, then
/// intersect with the zeroth-order value channel (both sound ⇒ intersection
/// sound; empty ⇒ fail closed).
fn combine_centered(
    lo: &[f64],
    hi: &[f64],
    mid: &[f64],
    seeds: &[usize],
    mv: &Channels,
    point: &Interval64,
) -> Result<Interval64> {
    let out_shape = mv.value.lower.shape().to_vec();
    if point.lower.shape() != out_shape.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: out_shape,
            got: point.lower.shape().to_vec(),
        });
    }
    let mut c_lo: Vec<f64> = point.lower.iter().copied().collect();
    let mut c_hi: Vec<f64> = point.upper.iter().copied().collect();

    // Accumulate Σ_i D_i · [lo_i − m_i, hi_i − m_i], outward-rounded.
    for (k, &s) in seeds.iter().enumerate() {
        // (x_s − m_s) ∈ [lo_s − m_s, hi_s − m_s]: endpoints outward, and
        // pinned to contain 0 (m is inside the box).
        let dl = (lo[s] - mid[s]).next_down().min(0.0);
        let dh = (hi[s] - mid[s]).next_up().max(0.0);
        let d_lo_std = mv.derivs[k].lower.as_standard_layout();
        let d_hi_std = mv.derivs[k].upper.as_standard_layout();
        let (d_lo, d_hi) = match (d_lo_std.as_slice(), d_hi_std.as_slice()) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(NyError::InvalidSpec(
                    "f64 mvf: derivative channel not contiguous".to_string(),
                ))
            }
        };
        if d_lo.len() != c_lo.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![c_lo.len()],
                got: vec![d_lo.len()],
            });
        }
        for j in 0..c_lo.len() {
            let (tl, th) = interval_mul(d_lo[j], d_hi[j], dl, dh);
            let (tl, th) = widen1(tl, th);
            c_lo[j] = (c_lo[j] + tl).next_down();
            c_hi[j] = (c_hi[j] + th).next_up();
        }
    }

    // Intersect with the zeroth-order enclosure (value channel of the
    // same walk — identical to propagate_ibp_f64_cell): both are sound,
    // so the intersection is sound and MUST be nonempty. NaN centered
    // endpoints (inf·0 upstream) lose the max/min and fall back to the
    // zeroth-order endpoint — conservative.
    let z_lo: Vec<f64> = mv.value.lower.iter().copied().collect();
    let z_hi: Vec<f64> = mv.value.upper.iter().copied().collect();
    for j in 0..c_lo.len() {
        let l = c_lo[j].max(z_lo[j]);
        let h = c_hi[j].min(z_hi[j]);
        // Negated form is deliberate: NaN endpoints make `l <= h` false, so a
        // NaN intersection FAILS CLOSED here; `l > h` would certify it.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(l <= h) {
            // Empty (or NaN) intersection: refuse to certify anything.
            return Err(NyError::InvalidSpec(
                "f64 mvf: centered/zeroth intersection empty — failing closed".to_string(),
            ));
        }
        c_lo[j] = l;
        c_hi[j] = h;
    }

    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), c_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 mvf: out lower: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), c_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 mvf: out upper: {e}")))?,
    })
}

/// Static layer-support predicate for the centered form. KEEP IN SYNC with
/// `eval_node_derivs`. Deliberately EXCLUDED (fail-closed):
/// - discontinuous ops (Trunc, ArgMax, CompareTensor, ScatterND) — the mean
///   value theorem does not hold across a jump, a centered form would be
///   UNSOUND;
/// - Lipschitz ops without an implemented derivative rule (Clip, MinBinary,
///   MaxBinary, MaxPool2d, Resize, Conv2d) — provable via hull rules but not
///   needed for the nn4sys mscn op set this exists for.
fn centered_supports_layer(layer: &Layer) -> bool {
    match layer {
        Layer::Reshape(_)
        | Layer::Slice(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Transpose(_)
        | Layer::Concat(_)
        | Layer::ReLU(_)
        | Layer::Sigmoid(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::MulBinary(_)
        | Layer::Div(_)
        | Layer::AddConstant(_)
        | Layer::MulConstant(_)
        | Layer::ReduceSum(_)
        | Layer::Linear(_)
        | Layer::MatMul(_) => true,
        Layer::Gather(gather) => gather.constant_indices().is_some(),
        _ => false,
    }
}

/// Seed channels for the network input: derivative of `input[q]` w.r.t. seed
/// axis `p` is the point interval `1` at `q == p`, `0` elsewhere.
fn build_input_entry(input: &Interval64, seeds: &[usize]) -> Result<Channels> {
    let shape = input.lower.shape().to_vec();
    let n: usize = shape.iter().product();
    let mut derivs = Vec::with_capacity(seeds.len());
    for &s in seeds {
        if s >= n {
            return Err(NyError::InvalidSpec(format!(
                "f64 mvf: seed axis {s} out of range for input of {n} elements"
            )));
        }
        let mut arr = ArrayD::zeros(IxDyn(&shape));
        let slice = arr.as_slice_mut().ok_or_else(|| {
            NyError::InvalidSpec("f64 mvf: seed tensor not contiguous".to_string())
        })?;
        slice[s] = 1.0;
        derivs.push(Interval64::point(arr));
    }
    Ok(Channels {
        value: input.clone(),
        derivs,
    })
}

/// Forward-mode derivative rules (one interval per seed axis). See the
/// module-level soundness argument for why each rule encloses the
/// branch-fixed real partial derivative at every point of the box.
///
/// `entry` resolves a node name (or [`NETWORK_INPUT`]) to its channels —
/// the per-box walk resolves from its flat cache, the batched multi-box
/// walk (#f64-batch-boxes) from one box's lane of its per-node vectors.
fn eval_node_derivs<'c>(
    node: &GraphNode,
    n_seeds: usize,
    entry: &dyn Fn(&str) -> Result<&'c Channels>,
) -> Result<Vec<Interval64>> {
    let unary = || -> Result<&Channels> {
        node.inputs()
            .first()
            .map(|n| entry(n))
            .ok_or_else(|| NyError::InvalidSpec("f64 mvf: missing unary input".to_string()))?
    };
    let binary = || -> Result<(&Channels, &Channels)> {
        match (node.inputs().first(), node.inputs().get(1)) {
            (Some(a), Some(b)) => Ok((entry(a)?, entry(b)?)),
            _ => Err(NyError::InvalidSpec(
                "f64 mvf: missing binary inputs".to_string(),
            )),
        }
    };
    let layer = node.layer();

    match layer {
        // ---- exact index movement: the identical op applies channel-wise ----
        Layer::Reshape(_)
        | Layer::Slice(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Transpose(_)
        | Layer::Gather(_) => (0..n_seeds)
            .map(|s| {
                let resolve =
                    |name: &str| -> Result<Interval64> { entry(name).map(|e| e.derivs[s].clone()) };
                eval_node(layer, node, &resolve)
            })
            .collect(),

        // Concat: graph inputs contribute their derivative channels; embedded
        // CONSTANT slots have zero derivative (constants do not depend on x).
        Layer::Concat(concat) => (0..n_seeds)
            .map(|s| {
                let mut parts: Vec<Interval64> = Vec::new();
                if let Some(ref slots) = concat.constant_inputs {
                    let mut graph_idx = 0usize;
                    for slot in slots {
                        match slot {
                            Some(constant) => parts.push(Interval64::point(ArrayD::zeros(IxDyn(
                                constant.lower().shape(),
                            )))),
                            None => {
                                let name = node.inputs().get(graph_idx).ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "f64 mvf: Concat ran out of graph inputs".to_string(),
                                    )
                                })?;
                                graph_idx += 1;
                                parts.push(entry(name)?.derivs[s].clone());
                            }
                        }
                    }
                } else {
                    for name in node.inputs() {
                        parts.push(entry(name)?.derivs[s].clone());
                    }
                }
                eval_concat(concat, &parts)
            })
            .collect(),

        // ---- ReLU: branch multiplier {0, 1}, hull at a possible crossing ----
        Layer::ReLU(_) => {
            let e = unary()?;
            let vl = &e.value.lower;
            let vu = &e.value.upper;
            e.derivs
                .iter()
                .map(|d| {
                    let mut lo = d.lower.clone();
                    let mut hi = d.upper.clone();
                    ndarray::Zip::from(&mut lo)
                        .and(&mut hi)
                        .and(vl)
                        .and(vu)
                        .for_each(|l, h, &v_lo, &v_hi| {
                            if v_lo >= 0.0 {
                                // relu ≡ identity on the box: multiplier 1.
                            } else if v_hi <= 0.0 {
                                // relu ≡ 0 on the box: derivative 0.
                                *l = 0.0;
                                *h = 0.0;
                            } else {
                                // Possible crossing: hull of {0·d, 1·d}.
                                *l = l.min(0.0);
                                *h = h.max(0.0);
                            }
                        });
                    Ok(Interval64 {
                        lower: lo,
                        upper: hi,
                    })
                })
                .collect()
        }

        // ---- Sigmoid: σ' = σ(1−σ) on the sound σ VALUE enclosure, ∩ [0, ¼] --
        Layer::Sigmoid(_) => {
            let e = unary()?;
            let vl = &e.value.lower;
            let vu = &e.value.upper;
            e.derivs
                .iter()
                .map(|d| {
                    let mut lo = d.lower.clone();
                    let mut hi = d.upper.clone();
                    ndarray::Zip::from(&mut lo)
                        .and(&mut hi)
                        .and(vl)
                        .and(vu)
                        .for_each(|l, h, &v_lo, &v_hi| {
                            // Sound σ(v) enclosure (same rule as the value channel).
                            let s_lo = widen_down_n(stable_sigmoid_f64(v_lo), TRANSCENDENTAL_ULPS)
                                .max(0.0);
                            let s_hi =
                                widen_up_n(stable_sigmoid_f64(v_hi), TRANSCENDENTAL_ULPS).min(1.0);
                            // 1 − σ, outward, clamped to its true range [0, 1].
                            let om_lo = (1.0 - s_hi).next_down().max(0.0);
                            let om_hi = (1.0 - s_lo).next_up().min(1.0);
                            // σ' enclosure ∩ its global range [0, 1/4].
                            let (m_lo, m_hi) = interval_mul(s_lo, s_hi, om_lo, om_hi);
                            let (m_lo, m_hi) = widen1(m_lo, m_hi);
                            let (m_lo, m_hi) = (m_lo.max(0.0), m_hi.min(0.25));
                            let (tl, th) = interval_mul(m_lo, m_hi, *l, *h);
                            let (tl, th) = widen1(tl, th);
                            *l = tl;
                            *h = th;
                        });
                    Ok(Interval64 {
                        lower: lo,
                        upper: hi,
                    })
                })
                .collect()
        }

        // ---- linear combinations: derivatives add / subtract ----------------
        Layer::Add(_) => {
            let (a, b) = binary()?;
            (0..n_seeds)
                .map(|s| {
                    broadcast_binary(&a.derivs[s], &b.derivs[s], false, |al, ah, bl, bh| {
                        Ok((al + bl, ah + bh))
                    })
                })
                .collect()
        }
        Layer::Sub(_) => {
            let (a, b) = binary()?;
            (0..n_seeds)
                .map(|s| {
                    broadcast_binary(&a.derivs[s], &b.derivs[s], false, |al, ah, bl, bh| {
                        Ok((al - bh, ah - bl))
                    })
                })
                .collect()
        }
        // d/dx (v + c) = v' exactly; the exact 0-add broadcasts v' to the
        // node's (possibly constant-broadcast) output shape without widening.
        Layer::AddConstant(add) => {
            let x = unary()?;
            let zero = Interval64::point(ArrayD::zeros(IxDyn(add.constant().shape())));
            x.derivs
                .iter()
                .map(|d| broadcast_binary(d, &zero, true, |al, ah, bl, bh| Ok((al + bl, ah + bh))))
                .collect()
        }
        Layer::MulConstant(mul) => {
            let x = unary()?;
            let c = Interval64::from_f32(mul.constant(), mul.constant());
            x.derivs
                .iter()
                .map(|d| {
                    broadcast_binary(d, &c, false, |al, ah, bl, bh| {
                        Ok(interval_mul(al, ah, bl, bh))
                    })
                })
                .collect()
        }

        // ---- product / quotient rules ---------------------------------------
        Layer::MulBinary(_) => {
            let (a, b) = binary()?;
            (0..n_seeds)
                .map(|s| {
                    let da_b =
                        broadcast_binary(&a.derivs[s], &b.value, false, |al, ah, bl, bh| {
                            Ok(interval_mul(al, ah, bl, bh))
                        })?;
                    let a_db =
                        broadcast_binary(&a.value, &b.derivs[s], false, |al, ah, bl, bh| {
                            Ok(interval_mul(al, ah, bl, bh))
                        })?;
                    broadcast_binary(&da_b, &a_db, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
                })
                .collect()
        }
        Layer::Div(_) => {
            let (a, b) = binary()?;
            // b is sign-definite over the box: the value channel (evaluated
            // BEFORE derivatives for this node) already failed the walk
            // otherwise. b² via the independent-product hull ⊇ {b²}.
            let b_sq = broadcast_binary(&b.value, &b.value, false, |al, ah, bl, bh| {
                Ok(interval_mul(al, ah, bl, bh))
            })?;
            (0..n_seeds)
                .map(|s| {
                    let da_b =
                        broadcast_binary(&a.derivs[s], &b.value, false, |al, ah, bl, bh| {
                            Ok(interval_mul(al, ah, bl, bh))
                        })?;
                    let a_db =
                        broadcast_binary(&a.value, &b.derivs[s], false, |al, ah, bl, bh| {
                            Ok(interval_mul(al, ah, bl, bh))
                        })?;
                    let num = broadcast_binary(&da_b, &a_db, false, |al, ah, bl, bh| {
                        Ok((al - bh, ah - bl))
                    })?;
                    broadcast_binary(&num, &b_sq, false, |al, ah, bl, bh| {
                        interval_div(al, ah, bl, bh)
                    })
                })
                .collect()
        }

        // ---- reductions / dot products: derivatives map linearly ------------
        Layer::ReduceSum(reduce) => {
            let x = unary()?;
            x.derivs
                .iter()
                .map(|d| eval_reduce_sum(reduce, d))
                .collect()
        }
        Layer::Linear(linear) => {
            let x = unary()?;
            x.derivs
                .iter()
                .map(|d| eval_linear_with_bias(linear, d, false))
                .collect()
        }
        Layer::MatMul(matmul) => {
            let (a, b) = binary()?;
            // d(s·a@b) = s·(da@b) + s·(a@db): eval_matmul applies the layer's
            // optional constant scale to each term, distributing correctly.
            (0..n_seeds)
                .map(|s| {
                    let da_b = eval_matmul(matmul, &a.derivs[s], &b.value)?;
                    let a_db = eval_matmul(matmul, &a.value, &b.derivs[s])?;
                    broadcast_binary(&da_b, &a_db, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
                })
                .collect()
        }

        other => Err(NyError::UnsupportedOp(format!(
            "f64 mvf: no sound derivative rule for layer {} — failing closed",
            other.layer_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{
        AddConstantLayer, AddLayer, DivLayer, LinearLayer, MatMulLayer, MulBinaryLayer,
        MulConstantLayer, ReLULayer, ReduceSumLayer, SigmoidLayer, SliceLayer, SubLayer,
        TruncLayer,
    };
    use ndarray::{arr1, arr2};

    /// Deterministic pseudo-random stream (xorshift) — no extra dev-dep.
    struct Rng(u64);
    impl Rng {
        fn next_unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn box64(lo: &[f32], hi: &[f32]) -> Interval64 {
        Interval64::from_f32(&arr1(lo).into_dyn(), &arr1(hi).into_dyn())
    }

    /// mscn-shaped DAG (mirrors the cell-forward test DAG): input [4]
    /// --Linear(4->3)--> ReLU --Mul(slice of input)--> Add(slice)
    /// --ReduceSum--> Div(by 2.5 + ReduceSum(relu)) --> Sigmoid.
    fn build_mscn_like_graph() -> GraphNetwork {
        let w = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ]);
        let b = arr1(&[0.125f32, -0.25, 0.5]);
        let linear = LinearLayer::new(w, Some(b)).unwrap();
        let slice = SliceLayer::new(0, 0, 3);

        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin".to_string()],
        ));
        g.add_node(GraphNode::from_input("head", Layer::Slice(slice)));
        g.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "relu",
            "head",
        ));
        g.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "mul",
            "head",
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["add".to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu_sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "denom",
            Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
                IxDyn(&[1]),
                2.5f32,
            ))),
            vec!["relu_sum".to_string()],
        ));
        g.add_node(GraphNode::binary(
            "div",
            Layer::Div(DivLayer),
            "sum",
            "denom",
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["div".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// Plain f64 concrete forward of the same DAG — the sample oracle.
    fn mscn_like_concrete(x: &[f64; 4]) -> f64 {
        let w = [
            [0.5f64, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ];
        let b = [0.125f64, -0.25, 0.5];
        let mut relu = [0.0f64; 3];
        for o in 0..3 {
            let mut s = b[o];
            for j in 0..4 {
                s += w[o][j] * x[j];
            }
            relu[o] = s.max(0.0);
        }
        let head = [x[0], x[1], x[2]];
        let mut sum = 0.0f64;
        let mut relu_sum = 0.0f64;
        for i in 0..3 {
            sum += relu[i] * head[i] + head[i];
            relu_sum += relu[i];
        }
        let div = sum / (2.5 + relu_sum);
        stable_sigmoid_f64(div)
    }

    /// Task test (a) ENCLOSURE: on random boxes over the mscn-like DAG
    /// (Linear/ReLU/Mul/Add/ReduceSum/Div/Sigmoid/Slice, incl. ReLU-crossing
    /// boxes), 2000 sampled concrete forwards per box plus all 16 corners lie
    /// inside the centered-form bound, and the bound refines the zeroth-order
    /// interval (never widens it).
    #[test]
    fn centered_encloses_samples_on_random_boxes_mscn_dag() {
        let g = build_mscn_like_graph();
        assert!(g.supports_ibp_f64_centered(), "test DAG must be supported");
        let mut rng = Rng(0xA5A5_5A5A_1234_9876);
        for round in 0..8 {
            let mut lo = [0.0f32; 4];
            let mut hi = [0.0f32; 4];
            for i in 0..4 {
                let c = (rng.next_unit() * 2.0 - 1.0) as f32; // in [-1, 1]
                let r = (rng.next_unit() * 0.6) as f32; // radius in [0, 0.6]
                lo[i] = c - r;
                hi[i] = c + r;
            }
            let input = box64(&lo, &hi);
            let centered = g.propagate_ibp_f64_centered(&input).unwrap();
            let zeroth = g.propagate_ibp_f64_cell(&input).unwrap();
            let (c_l, c_u) = (centered.lower[[0]], centered.upper[[0]]);
            let (z_l, z_u) = (zeroth.lower[[0]], zeroth.upper[[0]]);
            assert!(c_l <= c_u, "round {round}: malformed interval");
            assert!(
                c_l >= z_l && c_u <= z_u,
                "round {round}: centered [{c_l}, {c_u}] must refine zeroth [{z_l}, {z_u}]"
            );

            for _ in 0..2000 {
                let mut x = [0.0f64; 4];
                for i in 0..4 {
                    let (l, h) = (f64::from(lo[i]), f64::from(hi[i]));
                    x[i] = l + (h - l) * rng.next_unit();
                }
                let y = mscn_like_concrete(&x);
                assert!(
                    c_l <= y && y <= c_u,
                    "round {round}: sample {y} escapes centered [{c_l}, {c_u}] at x={x:?}"
                );
            }
            for mask in 0..16u32 {
                let mut x = [0.0f64; 4];
                for i in 0..4 {
                    x[i] = if mask & (1 << i) != 0 {
                        f64::from(hi[i])
                    } else {
                        f64::from(lo[i])
                    };
                }
                let y = mscn_like_concrete(&x);
                assert!(
                    c_l <= y && y <= c_u,
                    "round {round}: corner {y} escapes centered [{c_l}, {c_u}]"
                );
            }
        }
    }

    /// ENCLOSURE on a live-x-live MatMul DAG (both matmul operands perturbed):
    /// exercises the bilinear product rule da@b + a@db.
    #[test]
    fn centered_encloses_samples_matmul_dag() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "mm",
            Layer::MatMul(MatMulLayer::new(false, None)),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["mm".to_string()],
        ));
        g.add_node(GraphNode::new(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            vec!["relu".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Add(AddLayer),
            vec!["mul".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("out");
        assert!(g.supports_ibp_f64_centered());

        let lo = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5f32, 0.25, -0.75, 0.5]).unwrap();
        let hi = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5f32, 1.0, 0.25, 1.25]).unwrap();
        let input = Interval64::from_f32(&lo, &hi);
        let out = g.propagate_ibp_f64_centered(&input).unwrap();

        let mut rng = Rng(0xDEADBEEFCAFEF00D);
        for _ in 0..2000 {
            let mut x = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let l = f64::from(lo[[i, j]]);
                    let h = f64::from(hi[[i, j]]);
                    x[i][j] = l + (h - l) * rng.next_unit();
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    let mm = x[i][0] * x[0][j] + x[i][1] * x[1][j];
                    let y = mm.max(0.0) * x[i][j] + x[i][j];
                    let (l, u) = (out.lower[[i, j]], out.upper[[i, j]]);
                    assert!(
                        l <= y && y <= u,
                        "sample {y} escapes [{l}, {u}] at [{i},{j}]"
                    );
                }
            }
        }
    }

    /// Monotone MLP for the mono-corner tests: 2 inputs -> ReLU(W1 x + b1)
    /// -> w2 · h with ALL weights nonnegative — every partial derivative
    /// is >= 0 everywhere, so both dims must certify POSITIVE on any box,
    /// and the corner bound must equal the true range [f(lo), f(hi)] up to
    /// rounding.
    fn build_monotone_mlp() -> GraphNetwork {
        let w1 = arr2(&[[0.5f32, 1.25], [2.0, 0.25], [0.75, 1.5]]);
        let b1 = arr1(&[-0.5f32, 0.25, -1.0]);
        let w2 = arr2(&[[1.0f32, 0.5, 2.0]]);
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.set_output("out");
        g
    }

    fn monotone_mlp_concrete(x: &[f64; 2]) -> f64 {
        let w1 = [[0.5f64, 1.25], [2.0, 0.25], [0.75, 1.5]];
        let b1 = [-0.5f64, 0.25, -1.0];
        let w2 = [1.0f64, 0.5, 2.0];
        let mut acc = 0.0;
        for o in 0..3 {
            let h = (w1[o][0] * x[0] + w1[o][1] * x[1] + b1[o]).max(0.0);
            acc += w2[o] * h;
        }
        acc
    }

    /// Mission enclosure test (a), monotone side: on a truly monotone net
    /// the mono-corner lane certifies (on boxes where no straddling ReLU's
    /// exactly-zero derivative low gets pushed negative by the sound Higham
    /// widening of a following Linear — a straddling box may conservatively
    /// FAIL certification, which is the fail-open contract, never a
    /// soundness issue), its bound encloses a dense sample of the true
    /// image, and on fully-certified boxes it matches the true corner range
    /// to ~1e-12.
    #[test]
    fn mono_corner_certifies_and_is_exact_on_monotone_mlp() {
        let g = build_monotone_mlp();
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        let mut certified_rounds = 0usize;
        for round in 0..8 {
            let mut lo = [0.0f32; 2];
            let mut hi = [0.0f32; 2];
            for i in 0..2 {
                let c = (rng.next_unit() * 2.0 - 1.0) as f32;
                let r = (rng.next_unit() * 0.5 + 0.05) as f32;
                lo[i] = c - r;
                hi[i] = c + r;
            }
            let input = box64(&lo, &hi);
            let out = g.propagate_ibp_f64_centered_mono(&input).unwrap();
            assert_eq!(out.seeded_axes, 2, "round {round}: both axes seed");
            let Some(mono) = out.mono else {
                // Conservative non-certification (straddling ReLU + outward
                // rounding) — allowed; the exactness claim is checked on the
                // rounds that do certify, and at least one must.
                continue;
            };
            let (m_l, m_u) = (mono.lower[[0]], mono.upper[[0]]);
            let (c_l, c_u) = (out.centered.lower[[0]], out.centered.upper[[0]]);
            assert!(
                m_l >= c_l && m_u <= c_u,
                "round {round}: mono [{m_l}, {m_u}] must refine centered [{c_l}, {c_u}]"
            );
            if out.all_certified {
                certified_rounds += 1;
                // Exactness: the true range is [f(lo), f(hi)] by monotonicity.
                let t_min = monotone_mlp_concrete(&[f64::from(lo[0]), f64::from(lo[1])]);
                let t_max = monotone_mlp_concrete(&[f64::from(hi[0]), f64::from(hi[1])]);
                assert!(
                    (m_l - t_min).abs() <= 1e-12 && (m_u - t_max).abs() <= 1e-12,
                    "round {round}: mono [{m_l}, {m_u}] vs true [{t_min}, {t_max}]"
                );
            }
            // Enclosure against a dense sample of the true image.
            for _ in 0..2000 {
                let x = [
                    f64::from(lo[0]) + (f64::from(hi[0]) - f64::from(lo[0])) * rng.next_unit(),
                    f64::from(lo[1]) + (f64::from(hi[1]) - f64::from(lo[1])) * rng.next_unit(),
                ];
                let y = monotone_mlp_concrete(&x);
                assert!(
                    m_l <= y && y <= m_u,
                    "round {round}: sample {y} escapes mono [{m_l}, {m_u}] at {x:?}"
                );
            }
        }
        assert!(
            certified_rounds >= 1,
            "at least one random box must fully certify (got {certified_rounds})"
        );

        // Deterministic sign-definite box (all pre-activations positive:
        // W1 x + b1 with x >= 1 is componentwise > 0): certification and
        // corner exactness must BOTH hold here.
        let input = box64(&[1.0, 1.0], &[1.5, 2.0]);
        let out = g.propagate_ibp_f64_centered_mono(&input).unwrap();
        assert_eq!((out.certified_pairs, out.total_pairs), (2, 2));
        assert!(out.all_certified);
        let mono = out.mono.expect("sign-definite monotone box must certify");
        let (m_l, m_u) = (mono.lower[[0]], mono.upper[[0]]);
        let t_min = monotone_mlp_concrete(&[1.0, 1.0]);
        let t_max = monotone_mlp_concrete(&[1.5, 2.0]);
        assert!(
            (m_l - t_min).abs() <= 1e-12 && (m_u - t_max).abs() <= 1e-12,
            "mono [{m_l}, {m_u}] vs true [{t_min}, {t_max}]"
        );
    }

    /// Mission enclosure test (a), NON-monotone side: f(x) = relu(x) +
    /// relu(-x) (a V shape) has mixed-sign derivative on any zero-straddling
    /// box — the lane must NOT certify that dim (fail-open), and whatever
    /// bound it returns must still enclose dense samples.
    #[test]
    fn mono_corner_refuses_non_monotone_dim() {
        let w1 = arr2(&[[1.0f32], [-1.0]]);
        let w2 = arr2(&[[1.0f32, 1.0]]);
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(LinearLayer::new(w1, None).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.set_output("out");

        // Zero-straddling box: derivative is -1 left of 0, +1 right — mixed.
        let input = box64(&[-0.5], &[0.75]);
        let out = g.propagate_ibp_f64_centered_mono(&input).unwrap();
        assert_eq!(out.seeded_axes, 1);
        assert_eq!(
            out.certified_pairs, 0,
            "V-shaped net must NOT certify the straddling dim"
        );
        assert!(out.mono.is_none(), "no certification => no mono bound");

        // Sign-definite box: derivative is exactly +1 — certifies, and the
        // corner bound is the exact range.
        let input = box64(&[0.25], &[0.75]);
        let out = g.propagate_ibp_f64_centered_mono(&input).unwrap();
        assert_eq!((out.certified_pairs, out.total_pairs), (1, 1));
        let mono = out.mono.expect("sign-definite box must certify");
        let (m_l, m_u) = (mono.lower[[0]], mono.upper[[0]]);
        assert!(
            (m_l - 0.25).abs() <= 1e-12 && (m_u - 0.75).abs() <= 1e-12,
            "mono [{m_l}, {m_u}] vs true [0.25, 0.75]"
        );
    }

    /// Hybrid (mixed + certified dims): f(x0, x1) = relu(x0) + relu(-x0) +
    /// x1 over a box straddling x0 = 0 — x0 is mixed (stays an interval in
    /// the corner walks), x1 certifies POSITIVE. The mono bound must
    /// enclose dense samples, refine the centered bound, and match the
    /// dimension-reduced exact range (max over |x0| hull at x1 endpoints).
    #[test]
    fn mono_corner_hybrid_reduces_dimension_soundly() {
        let w1 = arr2(&[[1.0f32, 0.0], [-1.0, 0.0], [0.0, 1.0]]);
        let w2 = arr2(&[[1.0f32, 1.0, 1.0]]);
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(LinearLayer::new(w1, None).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.set_output("out");
        // NOTE: relu(x1)? no — w1 row 3 feeds x1 through ReLU. Keep the box
        // x1-positive so relu(x1) = x1 exactly and the true range is easy.
        let (x0l, x0u, x1l, x1u) = (-0.5f32, 0.25f32, 0.5f32, 1.0f32);
        let input = box64(&[x0l, x1l], &[x0u, x1u]);
        let out = g.propagate_ibp_f64_centered_mono(&input).unwrap();
        assert_eq!(out.total_pairs, 2);
        assert_eq!(
            out.certified_pairs, 1,
            "exactly the x1 dim certifies (x0 straddles zero)"
        );
        assert!(!out.all_certified);
        let mono = out.mono.expect("partial certification still refines");
        let (m_l, m_u) = (mono.lower[[0]], mono.upper[[0]]);
        // True range: |x0| + x1 with |x0| in [0, 0.5], x1 in [0.5, 1].
        let (t_min, t_max) = (0.5f64, 1.5f64);
        assert!(
            m_l <= t_min + 1e-12 && m_u >= t_max - 1e-12,
            "mono [{m_l}, {m_u}] must enclose true [{t_min}, {t_max}]"
        );
        let mut rng = Rng(0xFEED_FACE_0123_4567);
        for _ in 0..2000 {
            let x0 = f64::from(x0l) + (f64::from(x0u) - f64::from(x0l)) * rng.next_unit();
            let x1 = f64::from(x1l) + (f64::from(x1u) - f64::from(x1l)) * rng.next_unit();
            let y = x0.max(0.0) + (-x0).max(0.0) + x1.max(0.0);
            assert!(
                m_l <= y && y <= m_u,
                "sample {y} escapes mono [{m_l}, {m_u}] at ({x0}, {x1})"
            );
        }
        // Dimension reduction: the x1 pinning must beat the zeroth-order
        // interval's x1 contribution. Zeroth: relu-sum treats x1's [0.5, 1]
        // dependently... just assert refinement of centered.
        let (c_l, c_u) = (out.centered.lower[[0]], out.centered.upper[[0]]);
        assert!(m_l >= c_l && m_u <= c_u, "mono must refine centered");
    }

    /// Task test (b) QUADRATIC SHRINK: on f(x0, x1) = x0*x1 - x0 over
    /// [1, 1+w]^2 the true range is [0, (1+w)w] (width w + w^2), the
    /// zeroth-order interval is [-w, 2w + w^2] (excess EXACTLY 2w — linear),
    /// and the centered form's excess is EXACTLY w^2 — quadratic. Halving the
    /// box must therefore halve the zeroth-order excess (~2x) but quarter the
    /// centered excess (~4x).
    #[test]
    fn centered_excess_shrinks_quadratically() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "x0",
            Layer::Slice(SliceLayer::new(0, 0, 1)),
        ));
        g.add_node(GraphNode::from_input(
            "x1",
            Layer::Slice(SliceLayer::new(0, 1, 2)),
        ));
        g.add_node(GraphNode::binary(
            "prod",
            Layer::MulBinary(MulBinaryLayer),
            "x0",
            "x1",
        ));
        g.add_node(GraphNode::binary("out", Layer::Sub(SubLayer), "prod", "x0"));
        g.set_output("out");
        assert!(g.supports_ibp_f64_centered());

        let excesses = |w: f32| -> (f64, f64) {
            let input = box64(&[1.0, 1.0], &[1.0 + w, 1.0 + w]);
            let zeroth = g.propagate_ibp_f64_cell(&input).unwrap();
            let centered = g.propagate_ibp_f64_centered(&input).unwrap();
            let w64 = f64::from(w);
            let true_width = (1.0 + w64) * w64; // range [0, (1+w)w]
            let ez = (zeroth.upper[[0]] - zeroth.lower[[0]]) - true_width;
            let ec = (centered.upper[[0]] - centered.lower[[0]]) - true_width;
            assert!(ez > 0.0 && ec > 0.0, "excesses must be positive");
            (ez, ec)
        };

        let w = 1.0f32 / 64.0;
        let (ez_full, ec_full) = excesses(w);
        let (ez_half, ec_half) = excesses(w / 2.0);

        let zeroth_ratio = ez_full / ez_half;
        let centered_ratio = ec_full / ec_half;
        eprintln!(
            "quadratic-shrink: zeroth excess {ez_full:.3e} -> {ez_half:.3e} (ratio {zeroth_ratio:.3}), \
             centered excess {ec_full:.3e} -> {ec_half:.3e} (ratio {centered_ratio:.3})"
        );
        assert!(
            (1.7..=2.3).contains(&zeroth_ratio),
            "zeroth-order excess should shrink ~linearly (~2x per halving), got {zeroth_ratio}"
        );
        assert!(
            (3.3..=4.7).contains(&centered_ratio),
            "centered excess should shrink ~quadratically (~4x per halving), got {centered_ratio}"
        );
        assert!(
            ec_full < ez_full / 20.0,
            "centered excess {ec_full} should be far below zeroth excess {ez_full}"
        );
    }

    /// ReLU-crossing box: the derivative hull rule must keep the centered
    /// form enclosing (f = relu(x) on [-1, 2]; range [0, 2]).
    #[test]
    fn centered_relu_crossing_still_encloses() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("r", Layer::ReLU(ReLULayer)));
        g.set_output("r");
        let input = box64(&[-1.0], &[2.0]);
        let out = g.propagate_ibp_f64_centered(&input).unwrap();
        assert!(out.lower[[0]] <= 0.0, "must cover relu min 0");
        assert!(out.upper[[0]] >= 2.0, "must cover relu max 2");
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        for _ in 0..2000 {
            let x = -1.0 + 3.0 * rng.next_unit();
            let y = x.max(0.0);
            assert!(out.lower[[0]] <= y && y <= out.upper[[0]]);
        }
    }

    /// MulConstant derivative: f = 0.25·sigmoid(x0 - x1) — the screen test
    /// plateau head. Sampled enclosure + tighter-than-zeroth on a wide box.
    #[test]
    fn centered_encloses_scaled_sigmoid_of_difference() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "x0",
            Layer::Slice(SliceLayer::new(0, 0, 1)),
        ));
        g.add_node(GraphNode::from_input(
            "x1",
            Layer::Slice(SliceLayer::new(0, 1, 2)),
        ));
        g.add_node(GraphNode::binary("d", Layer::Sub(SubLayer), "x0", "x1"));
        g.add_node(GraphNode::new(
            "s",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["d".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::MulConstant(MulConstantLayer::new(ArrayD::from_elem(
                IxDyn(&[1]),
                0.25f32,
            ))),
            vec!["s".to_string()],
        ));
        g.set_output("out");
        assert!(g.supports_ibp_f64_centered());

        let input = box64(&[0.5, 0.5], &[1.5, 1.5]);
        let out = g.propagate_ibp_f64_centered(&input).unwrap();
        let mut rng = Rng(0xFEED_F00D_0BAD_CAFE);
        for _ in 0..2000 {
            let x0 = 0.5 + rng.next_unit();
            let x1 = 0.5 + rng.next_unit();
            let y = 0.25 * stable_sigmoid_f64(x0 - x1);
            assert!(
                out.lower[[0]] <= y && y <= out.upper[[0]],
                "sample {y} escapes [{}, {}]",
                out.lower[[0]],
                out.upper[[0]]
            );
        }
        // True range: 0.25·[σ(-1), σ(1)]; the centered upper must include it.
        let true_max = 0.25 * stable_sigmoid_f64(1.0);
        assert!(out.upper[[0]] >= true_max);
    }

    /// Point box degenerates to the (ulp-tight) zeroth-order cell forward.
    #[test]
    fn centered_point_box_degenerates_to_cell() {
        let g = build_mscn_like_graph();
        let p = [0.375f32, 0.625, -0.5, 0.25];
        let input = box64(&p, &p);
        let centered = g.propagate_ibp_f64_centered(&input).unwrap();
        let cell = g.propagate_ibp_f64_cell(&input).unwrap();
        assert_eq!(centered.lower, cell.lower);
        assert_eq!(centered.upper, cell.upper);
    }

    /// Discontinuous op (Trunc): the MVT does not hold across a jump — the
    /// gate must reject it and the walk must fail closed.
    #[test]
    fn centered_fails_closed_on_discontinuous_op() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("t", Layer::Trunc(TruncLayer)));
        g.set_output("t");
        assert!(!g.supports_ibp_f64_centered());
        let input = box64(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(g.propagate_ibp_f64_centered(&input).is_err());
    }

    /// Batched centered form (#f64-batch-boxes), same-kernel regime: W = 3
    /// boxes (mixed seed sets, so the internal grouping is exercised) on the
    /// mscn-like DAG keep every stacked Linear below the fast-kernel row
    /// gate — every box's batched result must be BIT-IDENTICAL to its
    /// per-box centered walk.
    #[test]
    fn centered_cells_equal_per_box_when_kernels_agree() {
        let g = build_mscn_like_graph();
        // Box 0/1: axes 1 and 3 wide (same seed set); box 2: axis 2 wide.
        let boxes = vec![
            box64(&[0.3, -0.2, 0.5, 0.1], &[0.3, 0.2, 0.5, 0.4]),
            box64(&[-0.1, 0.0, 0.25, -0.5], &[-0.1, 0.3, 0.25, -0.1]),
            box64(&[0.2, 0.4, -0.6, 0.7], &[0.2, 0.4, -0.2, 0.7]),
        ];
        let batched = g.propagate_ibp_f64_centered_cells(&boxes).expect("batched");
        for (b, x) in boxes.iter().enumerate() {
            let single = g.propagate_ibp_f64_centered(x).expect("per-box");
            let eq = |a: &ArrayD<f64>, c: &ArrayD<f64>| {
                a.iter()
                    .map(|v| v.to_bits())
                    .eq(c.iter().map(|v| v.to_bits()))
            };
            assert!(
                eq(&batched[b].lower, &single.lower) && eq(&batched[b].upper, &single.upper),
                "box {b}: batched centered diverged from per-box (same-kernel regime)"
            );
        }
    }

    /// Batched centered form, cross-box contamination: widening ONE box's
    /// sweep axis must leave every other box's batched result bit-identical
    /// (results depend only on the box's own input).
    #[test]
    fn centered_cells_mutation_leaves_other_boxes_bit_identical() {
        let g = build_mscn_like_graph();
        let mk = |w: f32| box64(&[0.3, -0.2, 0.5, 0.1], &[0.3, -0.2 + w, 0.5, 0.1 + 0.3]);
        let mut boxes = vec![mk(0.4), mk(0.5), mk(0.6)];
        let before = g.propagate_ibp_f64_centered_cells(&boxes).expect("batched");
        boxes[1] = mk(0.55); // same seed set, different width
        let after = g.propagate_ibp_f64_centered_cells(&boxes).expect("batched");
        for b in [0usize, 2] {
            assert_eq!(
                before[b]
                    .lower
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                after[b]
                    .lower
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "box {b}: lower changed when only box 1 was mutated"
            );
            assert_eq!(
                before[b]
                    .upper
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                after[b]
                    .upper
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "box {b}: upper changed when only box 1 was mutated"
            );
        }
        assert_ne!(
            before[1]
                .upper
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after[1]
                .upper
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "box 1 was widened — its result must change"
        );
    }

    /// Batched centered form, fat regime: a wide-Linear DAG with W = 16
    /// two-sweep-axis boxes promotes the stacked value/derivative Linears to
    /// the Rump kernel. Per box the batched result must CONTAIN the per-box
    /// centered result (kernel containment + inclusion-monotone combination)
    /// and every sampled concrete forward (soundness).
    #[test]
    fn centered_cells_fat_path_contains_per_box_and_samples() {
        let (rows, dim) = (8usize, 16usize);
        let mut rng = Rng(0x5EED_BA7C_4ED0_0001);
        let mut w1 = ndarray::Array2::<f32>::zeros((dim, dim));
        for v in w1.iter_mut() {
            *v = (rng.next_unit() * 0.4 - 0.2) as f32;
        }
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["lin2".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["sum".to_string()],
        ));
        g.set_output("out");
        assert!(g.supports_ibp_f64_centered());

        let w = 16usize;
        let n = rows * dim;
        let boxes: Vec<Interval64> = (0..w)
            .map(|_| {
                let mut lo: Vec<f64> = (0..n).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
                let mut hi = lo.clone();
                // Two sweep axes (0 and 7), everything else point.
                for &s in &[0usize, 7] {
                    lo[s] -= 0.2;
                    hi[s] += 0.2;
                }
                Interval64 {
                    lower: ArrayD::from_shape_vec(IxDyn(&[rows, dim]), lo).unwrap(),
                    upper: ArrayD::from_shape_vec(IxDyn(&[rows, dim]), hi).unwrap(),
                }
            })
            .collect();

        let batched = g.propagate_ibp_f64_centered_cells(&boxes).expect("batched");
        // The prepared-weight cache must not change a single bit.
        let cache = g.build_f64_weight_cache();
        let cached = g
            .propagate_ibp_f64_centered_cells_cached(&boxes, Some(&cache))
            .expect("cached");
        for (b, c) in cached.iter().enumerate() {
            assert!(
                c.lower
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(batched[b].lower.iter().map(|v| v.to_bits()))
                    && c.upper
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(batched[b].upper.iter().map(|v| v.to_bits())),
                "box {b}: prepared-weight cache changed the centered result"
            );
        }
        for (b, x) in boxes.iter().enumerate() {
            let single = g.propagate_ibp_f64_centered(x).expect("per-box");
            for ((bl, bu), (sl, su)) in batched[b]
                .lower
                .iter()
                .zip(batched[b].upper.iter())
                .zip(single.lower.iter().zip(single.upper.iter()))
            {
                assert!(
                    bl <= sl && bu >= su,
                    "box {b}: batched [{bl}, {bu}] does not contain per-box [{sl}, {su}]"
                );
            }
            // Sampled soundness against ulp-tight point walks.
            for _ in 0..20 {
                let mut sample = x.lower.clone();
                for (s, (&l, &h)) in sample.iter_mut().zip(x.lower.iter().zip(x.upper.iter())) {
                    *s = l + (h - l) * rng.next_unit();
                }
                let y = g
                    .propagate_ibp_f64_cell(&Interval64::point(sample))
                    .expect("point walk");
                for ((&yl, &yu), (&bl, &bu)) in y
                    .lower
                    .iter()
                    .zip(y.upper.iter())
                    .zip(batched[b].lower.iter().zip(batched[b].upper.iter()))
                {
                    // Bit-identical containment probe: f64::midpoint rounds differently past f64::MAX/2.
                    #[allow(clippy::manual_midpoint)]
                    let mid = 0.5 * (yl + yu);
                    assert!(
                        bl <= mid && mid <= bu,
                        "box {b}: sampled forward {mid} escapes batched [{bl}, {bu}]"
                    );
                }
            }
        }
    }

    /// Fused-walk gate (#f64-fused-walk): `propagate_ibp_f64_centered_with_value`
    /// must return a value channel BIT-IDENTICAL to a separate
    /// `propagate_ibp_f64_cell` walk and a centered interval BIT-IDENTICAL to
    /// `propagate_ibp_f64_centered`, across random boxes (mixed seeded /
    /// absorbed axes), an all-point box (seeds-empty degeneration), and a
    /// wide multi-axis box.
    #[test]
    fn centered_with_value_matches_separate_walks() {
        let g = build_mscn_like_graph();
        let bits = |x: &Interval64| -> (Vec<u64>, Vec<u64>) {
            (
                x.lower.iter().map(|v| v.to_bits()).collect(),
                x.upper.iter().map(|v| v.to_bits()).collect(),
            )
        };
        let mut rng = Rng(0xF05E_D0A1_C0DE_0001);
        let mut cases: Vec<Interval64> = Vec::new();
        for _ in 0..12 {
            let mut lo = [0.0f32; 4];
            let mut hi = [0.0f32; 4];
            for i in 0..4 {
                let c = (rng.next_unit() * 2.0 - 1.0) as f32;
                // Mix of point axes (absorbed) and wide axes (seeded).
                let r = if rng.next_unit() < 0.5 {
                    0.0
                } else {
                    (rng.next_unit() * 0.5) as f32
                };
                lo[i] = c - r;
                hi[i] = c + r;
            }
            cases.push(box64(&lo, &hi));
        }
        // All-point box: the seeds-empty degeneration to the cell forward.
        cases.push(box64(&[0.25, -0.5, 0.75, 0.0], &[0.25, -0.5, 0.75, 0.0]));
        // All-wide box.
        cases.push(box64(&[-0.4, -0.3, -0.2, -0.1], &[0.4, 0.3, 0.2, 0.1]));

        for (i, input) in cases.iter().enumerate() {
            let (value, fused_centered) = g
                .propagate_ibp_f64_centered_with_value(input)
                .expect("fused walk");
            let cell = g.propagate_ibp_f64_cell(input).expect("cell walk");
            let centered = g.propagate_ibp_f64_centered(input).expect("centered walk");
            assert_eq!(
                bits(&value),
                bits(&cell),
                "case {i}: fused value channel diverged from the cell walk"
            );
            assert_eq!(
                bits(&fused_centered),
                bits(&centered),
                "case {i}: fused centered diverged from the centered walk"
            );
        }
    }

    /// Div derivative straddling-zero divisor fails closed via the value
    /// channel (the derivative rule is never reached with an unsafe divisor).
    #[test]
    fn centered_div_straddling_zero_fails_closed() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "d",
            Layer::Div(DivLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("d");
        assert!(g.supports_ibp_f64_centered());
        let input = box64(&[-1.0, -1.0], &[1.0, 1.0]);
        assert!(g.propagate_ibp_f64_centered(&input).is_err());
    }

    /// Monotone MLP with FAT Linears (nonnegative weights, stacked shapes
    /// promoting the Rump kernel at moderate W) for the batched fused-walk
    /// gates.
    fn build_fat_monotone_mlp(rng: &mut Rng) -> GraphNetwork {
        let (inp, hid) = (32usize, 64usize);
        let mut w1 = ndarray::Array2::<f32>::zeros((hid, inp));
        for v in w1.iter_mut() {
            *v = (rng.next_unit() * 0.2 + 0.01) as f32; // strictly positive
        }
        let mut w2 = ndarray::Array2::<f32>::zeros((1, hid));
        for v in w2.iter_mut() {
            *v = (rng.next_unit() * 0.3 + 0.01) as f32;
        }
        let mut b1 = ndarray::Array1::<f32>::zeros(hid);
        for v in b1.iter_mut() {
            *v = (rng.next_unit() * 0.5 - 0.25) as f32;
        }
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// Random box over [inp] with `n_sweep` wide axes (the rest point).
    fn fat_mlp_box(rng: &mut Rng, n_sweep: usize) -> Interval64 {
        let inp = 32usize;
        let mut lo: Vec<f64> = (0..inp).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let mut hi = lo.clone();
        for s in 0..n_sweep {
            let axis = s * 7 % inp;
            lo[axis] -= 0.1 + rng.next_unit() * 0.2;
            hi[axis] += 0.1 + rng.next_unit() * 0.2;
        }
        Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(&[inp]), lo).unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(&[inp]), hi).unwrap(),
        }
    }

    /// D2 gate (#f64-fused-walk × #f64-batch-boxes): the batched fused
    /// walk's VALUE channel is bit-identical to the batched zeroth-order
    /// walk (`propagate_ibp_f64_cells_cached`) over the same chunk, and its
    /// centered channel is bit-identical to the batched centered entry —
    /// at BOTH thin (scalar-kernel) and fat (Rump-promoted) stacked shapes.
    /// This is what lets the screen drop its separate zeroth chunk for
    /// fused-lane boxes.
    #[test]
    fn batched_fused_value_matches_batched_cells() {
        let mut rng = Rng(0xD2_5EED_0001);
        let g = build_fat_monotone_mlp(&mut rng);
        let cache = g.build_f64_weight_cache();
        let bits = |x: &Interval64| -> (Vec<u64>, Vec<u64>) {
            (
                x.lower.iter().map(|v| v.to_bits()).collect(),
                x.upper.iter().map(|v| v.to_bits()).collect(),
            )
        };
        for &w in &[3usize, 40] {
            // rows = 1 per box: m = W stacked rows; W = 3 stays on the scalar
            // kernel, W = 40 promotes the Rump kernel (vol = 40·32·64 > 32768).
            let boxes: Vec<Interval64> = (0..w).map(|_| fat_mlp_box(&mut rng, 2)).collect();
            let fused = g
                .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, Some(&cache))
                .expect("fused batched walk");
            let cells = g
                .propagate_ibp_f64_cells_cached(&boxes, Some(&cache))
                .expect("batched cell walk");
            let centered = g
                .propagate_ibp_f64_centered_cells_cached(&boxes, Some(&cache))
                .expect("batched centered walk");
            for b in 0..w {
                assert_eq!(
                    bits(&fused[b].value),
                    bits(&cells[b]),
                    "W={w} box {b}: fused value diverged from batched cells"
                );
                assert_eq!(
                    bits(&fused[b].centered),
                    bits(&centered[b]),
                    "W={w} box {b}: fused centered diverged from batched centered"
                );
            }
        }
    }

    /// D5 gate (#mono-corner × #f64-batch-boxes), same-kernel regime: with
    /// thin stacked shapes (all kernels scalar) the batched fused walk's
    /// value/centered/mono and certification stats are BIT-IDENTICAL to the
    /// per-box `propagate_ibp_f64_centered_mono` walks.
    #[test]
    fn batched_mono_matches_per_box_when_kernels_agree() {
        let mut rng = Rng(0xD5_5EED_0002);
        let g = build_monotone_mlp();
        let bits = |x: &Interval64| -> (Vec<u64>, Vec<u64>) {
            (
                x.lower.iter().map(|v| v.to_bits()).collect(),
                x.upper.iter().map(|v| v.to_bits()).collect(),
            )
        };
        // 3 boxes, rows = 1, in = 2: stacked m = 3 < FAST_GEMM_MIN_ROWS and
        // volume tiny — every Linear stays on the scalar kernel in both lanes.
        let boxes: Vec<Interval64> = (0..3)
            .map(|_| {
                let c0 = rng.next_unit() as f32 * 2.0 - 1.0;
                let c1 = rng.next_unit() as f32 * 2.0 - 1.0;
                box64(&[c0 - 0.2, c1 - 0.3], &[c0 + 0.25, c1 + 0.15])
            })
            .collect();
        let batched = g
            .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, None)
            .expect("batched fused walk");
        for (b, x) in boxes.iter().enumerate() {
            let single = g.propagate_ibp_f64_centered_mono(x).expect("per-box walk");
            assert_eq!(
                bits(&batched[b].value),
                bits(&single.value),
                "box {b} value"
            );
            assert_eq!(
                bits(&batched[b].centered),
                bits(&single.centered),
                "box {b} centered"
            );
            assert_eq!(
                (
                    batched[b].seeded_axes,
                    batched[b].certified_pairs,
                    batched[b].total_pairs,
                    batched[b].all_certified
                ),
                (
                    single.seeded_axes,
                    single.certified_pairs,
                    single.total_pairs,
                    single.all_certified
                ),
                "box {b} certification stats"
            );
            match (&batched[b].mono, &single.mono) {
                (None, None) => {}
                (Some(bm), Some(sm)) => {
                    assert_eq!(bits(bm), bits(sm), "box {b} mono");
                }
                (bm, sm) => panic!(
                    "box {b}: mono presence diverged (batched {:?} vs per-box {:?})",
                    bm.is_some(),
                    sm.is_some()
                ),
            }
        }
    }

    /// D5 gate, fat (Rump-promoted) regime: per box the batched mono bound
    /// (when produced) CONTAINS the per-box mono bound (scalar corner walks
    /// over tighter corner boxes — inclusion-monotone rules + kernel
    /// containment), refines the batched centered bound, and encloses a
    /// dense sample of true forwards. On the strictly-monotone net with
    /// sign-definite boxes the fat lane must certify at least one box.
    #[test]
    fn batched_mono_fat_path_contains_per_box_and_samples() {
        let mut rng = Rng(0xD5_5EED_0003);
        let g = build_fat_monotone_mlp(&mut rng);
        let cache = g.build_f64_weight_cache();
        let w = 40usize;
        // Positive boxes: all l1 pre-activations > 0 for most boxes, so the
        // derivative signs certify cleanly.
        let boxes: Vec<Interval64> = (0..w)
            .map(|_| {
                let inp = 32usize;
                let mut lo: Vec<f64> = (0..inp).map(|_| rng.next_unit() + 1.0).collect();
                let mut hi = lo.clone();
                for &axis in &[0usize, 7, 21] {
                    lo[axis] -= 0.15;
                    hi[axis] += 0.2;
                }
                Interval64 {
                    lower: ArrayD::from_shape_vec(IxDyn(&[inp]), lo).unwrap(),
                    upper: ArrayD::from_shape_vec(IxDyn(&[inp]), hi).unwrap(),
                }
            })
            .collect();
        let batched = g
            .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, Some(&cache))
            .expect("batched fused walk");
        let mut certified_boxes = 0usize;
        for (b, x) in boxes.iter().enumerate() {
            let out = &batched[b];
            // mono refines centered.
            if let Some(m) = &out.mono {
                certified_boxes += 1;
                for ((&ml, &mh), (&cl, &ch)) in m
                    .lower
                    .iter()
                    .zip(m.upper.iter())
                    .zip(out.centered.lower.iter().zip(out.centered.upper.iter()))
                {
                    assert!(
                        ml >= cl && mh <= ch,
                        "box {b}: mono [{ml}, {mh}] must refine centered [{cl}, {ch}]"
                    );
                }
                // Containment of the per-box mono when both certified.
                let single = g.propagate_ibp_f64_centered_mono(x).expect("per-box walk");
                if let Some(sm) = &single.mono {
                    for ((&ml, &mh), (&sl, &sh)) in m
                        .lower
                        .iter()
                        .zip(m.upper.iter())
                        .zip(sm.lower.iter().zip(sm.upper.iter()))
                    {
                        assert!(
                            ml <= sl && mh >= sh,
                            "box {b}: batched mono [{ml}, {mh}] does not contain \
                             per-box mono [{sl}, {sh}]"
                        );
                    }
                }
                // Dense-sample enclosure via ulp-tight point walks.
                for _ in 0..50 {
                    let mut sample = x.lower.clone();
                    for (s, (&l, &h)) in sample.iter_mut().zip(x.lower.iter().zip(x.upper.iter())) {
                        *s = l + (h - l) * rng.next_unit();
                    }
                    let y = g
                        .propagate_ibp_f64_cell(&Interval64::point(sample))
                        .expect("point walk");
                    let mid = f64::midpoint(y.lower[[0]], y.upper[[0]]);
                    assert!(
                        m.lower[[0]] <= mid && mid <= m.upper[[0]],
                        "box {b}: sample {mid} escapes batched mono [{}, {}]",
                        m.lower[[0]],
                        m.upper[[0]]
                    );
                }
            }
        }
        assert!(
            certified_boxes >= 1,
            "sign-definite monotone boxes must certify at least once (got {certified_boxes})"
        );
    }

    /// Cross-box isolation of the batched fused walk (mono included):
    /// mutating ONE box leaves every other box's value/centered/mono
    /// BIT-IDENTICAL (the stacked corner walks are per-box isolated and
    /// their results do not depend on stack position).
    #[test]
    fn batched_mono_mutation_leaves_other_boxes_bit_identical() {
        let mut rng = Rng(0xD5_5EED_0004);
        let g = build_fat_monotone_mlp(&mut rng);
        let mut boxes: Vec<Interval64> = (0..20).map(|_| fat_mlp_box(&mut rng, 2)).collect();
        let bits = |x: &Interval64| -> (Vec<u64>, Vec<u64>) {
            (
                x.lower.iter().map(|v| v.to_bits()).collect(),
                x.upper.iter().map(|v| v.to_bits()).collect(),
            )
        };
        let before = g
            .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, None)
            .expect("batched fused walk");
        boxes[1] = Interval64 {
            lower: boxes[1].lower.mapv(|v| v - 0.05),
            upper: boxes[1].upper.mapv(|v| v + 0.05),
        };
        let after = g
            .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, None)
            .expect("batched fused walk");
        for b in 0..boxes.len() {
            if b == 1 {
                assert_ne!(
                    bits(&before[b].value),
                    bits(&after[b].value),
                    "box 1 was widened — its value must change"
                );
                continue;
            }
            assert_eq!(
                bits(&before[b].value),
                bits(&after[b].value),
                "box {b} value"
            );
            assert_eq!(
                bits(&before[b].centered),
                bits(&after[b].centered),
                "box {b} centered"
            );
            match (&before[b].mono, &after[b].mono) {
                (None, None) => {}
                (Some(x), Some(y)) => assert_eq!(bits(x), bits(y), "box {b} mono"),
                _ => panic!("box {b}: mono presence changed when only box 1 was mutated"),
            }
        }
    }

    /// ADVERSARIAL D5 differential (multi-pattern regime): mixed-sign MLPs
    /// whose output elements carry SEVERAL DISTINCT sign patterns per box —
    /// the `first_walk + 2p` corner-walk indexing of the batched mono stage
    /// is only exercised with >= 2 patterns (the monotone-net gates above
    /// always classify a single all-POS pattern). Checks, per random box:
    /// - thin (same-kernel) shapes: batched value/centered/mono and stats
    ///   BIT-IDENTICAL to the per-box walk (any pattern/walk-slot mixup
    ///   diverges immediately);
    /// - fat (Rump-promoted) shapes: batched mono CONTAINS the per-box mono
    ///   and refines centered;
    /// - both: 40 random interior points per box (ulp-tight point walks)
    ///   stay inside the mono bound — the direct no-false-tightening probe.
    #[test]
    fn adv_batched_mono_multi_pattern_differential() {
        let bits = |x: &Interval64| -> (Vec<u64>, Vec<u64>) {
            (
                x.lower.iter().map(|v| v.to_bits()).collect(),
                x.upper.iter().map(|v| v.to_bits()).collect(),
            )
        };
        for (seed, inp, hid, out, fat) in [
            (0xADD5_0001u64, 6usize, 24usize, 4usize, false),
            (0xADD5_0002, 6, 24, 4, false),
            (0xADD5_0003, 16, 512, 4, true),
        ] {
            let mut rng = Rng(seed);
            // Mixed-sign weights, biased positive: many derivative signs
            // certify, negative columns give NEG, straddling ReLUs give
            // MIXED — several patterns per box across `out` elements.
            let mut w1 = ndarray::Array2::<f32>::zeros((hid, inp));
            for (i, v) in w1.iter_mut().enumerate() {
                let m = if i % 5 == 0 { -1.0 } else { 1.0 };
                *v = (m * (0.05 + rng.next_unit() * 0.6)) as f32;
            }
            let mut b1 = ndarray::Array1::<f32>::zeros(hid);
            for v in b1.iter_mut() {
                *v = (rng.next_unit() * 0.4 - 0.2) as f32;
            }
            let mut w2 = ndarray::Array2::<f32>::zeros((out, hid));
            for (i, v) in w2.iter_mut().enumerate() {
                let m = if (i / hid) % 2 == 0 { 1.0 } else { -1.0 };
                *v = (m * (0.02 + rng.next_unit() * 0.4)) as f32;
            }
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input(
                "l1",
                Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
            ));
            g.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer),
                vec!["l1".to_string()],
            ));
            g.add_node(GraphNode::new(
                "out",
                Layer::Linear(LinearLayer::new(w2, None).unwrap()),
                vec!["relu".to_string()],
            ));
            g.set_output("out");
            let cache = g.build_f64_weight_cache();

            let w = 24usize;
            let boxes: Vec<Interval64> = (0..w)
                .map(|_| {
                    let mut lo: Vec<f64> = (0..inp).map(|_| rng.next_unit() * 1.6 - 0.3).collect();
                    let mut hi = lo.clone();
                    for &axis in &[0usize, 3] {
                        lo[axis] -= 0.05 + rng.next_unit() * 0.25;
                        hi[axis] += 0.05 + rng.next_unit() * 0.25;
                    }
                    Interval64 {
                        lower: ArrayD::from_shape_vec(IxDyn(&[inp]), lo).unwrap(),
                        upper: ArrayD::from_shape_vec(IxDyn(&[inp]), hi).unwrap(),
                    }
                })
                .collect();

            // The adversarial premise must hold: at least one box classifies
            // >= 2 distinct sign patterns (multi-corner-walk indexing runs).
            let mut multi_pattern_boxes = 0usize;
            for x in &boxes {
                let lo_s = x.lower.as_slice().unwrap();
                let hi_s = x.upper.as_slice().unwrap();
                let seeds: Vec<usize> = (0..lo_s.len())
                    .filter(|&i| axis_is_seeded(lo_s[i], hi_s[i]))
                    .collect();
                if seeds.is_empty() {
                    continue;
                }
                let mv = g.propagate_ibp_f64_mean_value(x, &seeds).unwrap();
                if let Some(sg) = classify_sign_groups(&mv, seeds.len(), out) {
                    if sg.groups.len() >= 2 {
                        multi_pattern_boxes += 1;
                    }
                }
            }
            assert!(
                multi_pattern_boxes >= 3,
                "seed {seed:#x}: differential is toothless — only \
                 {multi_pattern_boxes} multi-pattern boxes"
            );

            let batched = g
                .propagate_ibp_f64_centered_mono_cells_cached(&boxes, true, Some(&cache))
                .expect("batched fused walk");
            let mut mono_produced = 0usize;
            for (b, x) in boxes.iter().enumerate() {
                let single = g.propagate_ibp_f64_centered_mono(x).expect("per-box walk");
                if !fat {
                    // Same-kernel regime: everything bit-identical.
                    assert_eq!(
                        bits(&batched[b].value),
                        bits(&single.value),
                        "box {b} value"
                    );
                    assert_eq!(
                        bits(&batched[b].centered),
                        bits(&single.centered),
                        "box {b} centered"
                    );
                    match (&batched[b].mono, &single.mono) {
                        (None, None) => {}
                        (Some(bm), Some(sm)) => assert_eq!(bits(bm), bits(sm), "box {b} mono"),
                        (bm, sm) => panic!(
                            "box {b}: mono presence diverged (batched {} vs per-box {})",
                            bm.is_some(),
                            sm.is_some()
                        ),
                    }
                    assert_eq!(
                        (batched[b].certified_pairs, batched[b].total_pairs),
                        (single.certified_pairs, single.total_pairs),
                        "box {b} stats"
                    );
                } else if let (Some(bm), Some(sm)) = (&batched[b].mono, &single.mono) {
                    // Fat regime: batched mono must CONTAIN the per-box mono.
                    for ((&ml, &mh), (&sl, &sh)) in bm
                        .lower
                        .iter()
                        .zip(bm.upper.iter())
                        .zip(sm.lower.iter().zip(sm.upper.iter()))
                    {
                        assert!(
                            ml <= sl && mh >= sh,
                            "box {b}: batched mono [{ml}, {mh}] excludes per-box [{sl}, {sh}]"
                        );
                    }
                }
                // Direct no-false-tightening probe: interior samples stay
                // inside the batched mono bound on every output element.
                if let Some(m) = &batched[b].mono {
                    mono_produced += 1;
                    for ((&ml, &mh), (&cl, &ch)) in m.lower.iter().zip(m.upper.iter()).zip(
                        batched[b]
                            .centered
                            .lower
                            .iter()
                            .zip(batched[b].centered.upper.iter()),
                    ) {
                        assert!(ml >= cl && mh <= ch, "box {b}: mono must refine centered");
                    }
                    for t in 0..40 {
                        let mut sample = x.lower.clone();
                        for (s, (&l, &h)) in
                            sample.iter_mut().zip(x.lower.iter().zip(x.upper.iter()))
                        {
                            *s = l + (h - l) * rng.next_unit();
                        }
                        let y = g
                            .propagate_ibp_f64_cell(&Interval64::point(sample))
                            .expect("point walk");
                        for j in 0..out {
                            let mid = f64::midpoint(y.lower[[j]], y.upper[[j]]);
                            assert!(
                                m.lower[[j]] <= mid && mid <= m.upper[[j]],
                                "box {b} sample {t} elem {j}: {mid} escapes mono \
                                 [{}, {}] (seed {seed:#x})",
                                m.lower[[j]],
                                m.upper[[j]]
                            );
                        }
                    }
                }
            }
            assert!(
                mono_produced >= 3,
                "seed {seed:#x}: only {mono_produced} mono bounds produced — differential \
                 lost its subject"
            );
        }
    }
}
