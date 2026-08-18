// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound multi-neuron (k-ReLU) relaxation — INCREMENT 1 (k = 2).
//!
//! Implements the soundness-first 2-ReLU convex-hull relaxation of
//! `docs/MULTI_NEURON_RELAXATION_DESIGN.md` and mirrors the *validated* numpy
//! oracle `validate_hull.py` (kept alongside this module, with its fixture
//! generator `gen_fixture.py`). The pipeline is:
//!
//! 1. [`Octahedron2`] — a sound octahedral over-approximation `P ⊇ Z` of the
//!    reachable pre-activation set (§1.2, Invariant P1).
//! 2. [`arrangement_lifted_vertices`] / [`proposed_hull_normals`] — floating
//!    geometry that proposes useful stored-`f32` directions.  Its tolerances and
//!    deduplication carry no verdict authority.
//! 3. [`ExactRelu2Support`] — exact rational maximization of each stored normal
//!    over `P` intersected with all four ReLU orthants, followed by verified
//!    upward conversion of the exact support to a finite `f32` RHS.
//! 4. [`MultiNeuronConstraint`] / [`MultiNeuronPool`] — an accepted group facet with a
//!    `β_c ≥ 0` Adam multiplier, mirroring `GraphCuttingPlane` verbatim.
//! 5. `LinearBounds::apply_group_facet_contribution` (in `bounds/linear.rs`) —
//!    the A-matrix injection (§2.2), generalizing `apply_beta_split_to_column`.
//!
//! # The soundness backbone (why every stored half-space is a superset)
//!
//! For **any** finite stored direction `a ∈ f32⁴` (facet normal or not), ReLU is
//! affine on each of its four closed orthants.  The support over bounded `P` is
//! therefore the maximum of four linear programs, attained at a cell vertex.
//! [`ExactRelu2Support`] enumerates those vertices and evaluates feasibility and
//! support in exact rational arithmetic, then verifies that the stored `f32` RHS
//! is at least that exact maximum.  Consequently proposal geometry governs only
//! *tightness*: an imperfect normal is still safe once certified, and a failed
//! proposal is dropped.  [`hull_facets`] remains a legacy research helper whose
//! RHS is outward only relative to its supplied tolerance-derived vertices.

use crate::beta_crown::bab_cuts::{CutKind, CutMetadata};
use crate::bounds::LinearBounds;
use ny_core::{NyError, Result as NyResult};
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::Instant;

pub mod producer;
pub use producer::{
    combined_row_octahedron, combined_row_octahedron_with_deadline, combined_rows_octahedra,
    combined_rows_octahedra_with_deadline,
};

// The historical authority entry stays hard-disabled. M1 exposes only the
// private, call-local observation orchestrator below; the evidence-bearing
// carrier remains private and cannot become an ambient/raw-facet API or verdict
// surface.
#[allow(dead_code)]
mod certified_cut_authority;
#[allow(dead_code)]
mod certified_cut_m2_shadow;
#[allow(dead_code)]
mod certified_cut_shadow;
pub(crate) use certified_cut_m2_shadow::production_resident_cut_m2_projected_enabled;
pub(crate) use certified_cut_shadow::{
    production_resident_cut_shadow_enabled, run_production_resident_cut_shadow,
    ProductionResidentCutShadowRequest,
};

pub mod root_inject;

mod support_exact;
pub use support_exact::{ExactRelu2FacetCertificate, ExactRelu2Support};

/// Row-major (C-order) flat index of the neuron at `(c, h, w)` in a conv output
/// tensor of shape `[C, H, W]` — SOUNDNESS-CRITICAL for conv-group facets
/// (increment 3, item 2).
///
/// NY flattens tensors row-major (`BoundedTensor::flatten`, and the whole
/// `layers/` stack — see the `per_channel`/`prelu` "row-major, channel-major"
/// invariants), so the flattened pre-activation column of a conv node is
/// `flat = c·(H·W) + h·W + w`. The producer's spec rows index that flat column;
/// a wrong mapping would bound the WRONG neuron pair (an unsound facet), so this
/// mapping is validated end-to-end against a live NY conv backward by
/// `chw_indexing_selects_intended_conv_neurons` in `tests.rs`.
#[inline]
pub fn chw_to_flat(c: usize, h: usize, w: usize, height: usize, width: usize) -> usize {
    c * (height * width) + h * width + w
}

// ===========================================================================
// §1.2  Sound octahedral producer  P ⊇ Z   (k = 2)
// ===========================================================================

/// A sound octahedral over-approximation `P ⊇ Z` of the reachable
/// pre-activation set of a 2-neuron group.
///
/// `P = { x ∈ ℝ² : l_i ≤ x_i ≤ u_i, s_lo ≤ x_1+x_2 ≤ s_hi,
///                 d_lo ≤ x_1−x_2 ≤ d_hi }`.
///
/// The eight bounds are, in production, produced by the same sound
/// outward-rounded primitive that produces the per-neuron `l_i,u_i` today
/// (Invariant P1). [`Octahedron2::from_affine`] is the test/standalone producer
/// for an affine pre-activation map `x = W u + t` over a box — it mirrors
/// `octahedral_P` in `validate_hull.py`.
#[derive(Clone, Debug, PartialEq)]
pub struct Octahedron2 {
    pub l1: f64,
    pub u1: f64,
    pub l2: f64,
    pub u2: f64,
    pub s_lo: f64,
    pub s_hi: f64,
    pub d_lo: f64,
    pub d_hi: f64,
}

/// Exact interval range of an affine form `f(u) = w·u + t` over the box
/// `[lo, hi]` (each coordinate independent). Matches the oracle's `lin_range`.
fn affine_range(w: &[f64], t: f64, lo: &[f64], hi: &[f64]) -> (f64, f64) {
    let mut range_lo = t;
    let mut range_hi = t;
    for j in 0..w.len() {
        let a = w[j] * lo[j];
        let b = w[j] * hi[j];
        range_lo += a.min(b);
        range_hi += a.max(b);
    }
    (range_lo, range_hi)
}

impl Octahedron2 {
    /// Sound octahedral `P` for an affine pre-activation map `x_i = w_i·u + t_i`
    /// over the input box `[u_lo, u_hi]` (Invariant P1 — outward-rounded).
    ///
    /// Each octahedral bound is an interval range of a linear objective over the
    /// box (`x_1`, `x_2`, `x_1±x_2`); every *upper* bound is nudged up with
    /// `next_up_f32` and every *lower* bound down with `next_down_f32` so the
    /// stored `P` certifiably contains the true reachable set `Z` even after the
    /// f64 range accumulation. `w1`/`w2` must have the same length as
    /// `u_lo`/`u_hi`.
    pub fn from_affine(
        w1: &[f64],
        w2: &[f64],
        t1: f64,
        t2: f64,
        u_lo: &[f64],
        u_hi: &[f64],
    ) -> Self {
        assert_eq!(w1.len(), w2.len());
        assert_eq!(w1.len(), u_lo.len());
        assert_eq!(w1.len(), u_hi.len());
        let out_up = |x: f64| next_up_f32(x as f32) as f64;
        let out_dn = |x: f64| next_down_f32(x as f32) as f64;

        let (l1, u1) = affine_range(w1, t1, u_lo, u_hi);
        let (l2, u2) = affine_range(w2, t2, u_lo, u_hi);
        let sum_w: Vec<f64> = w1.iter().zip(w2).map(|(a, b)| a + b).collect();
        let dif_w: Vec<f64> = w1.iter().zip(w2).map(|(a, b)| a - b).collect();
        let (s_lo, s_hi) = affine_range(&sum_w, t1 + t2, u_lo, u_hi);
        let (d_lo, d_hi) = affine_range(&dif_w, t1 - t2, u_lo, u_hi);
        Self {
            l1: out_dn(l1),
            u1: out_up(u1),
            l2: out_dn(l2),
            u2: out_up(u2),
            s_lo: out_dn(s_lo),
            s_hi: out_up(s_hi),
            d_lo: out_dn(d_lo),
            d_hi: out_up(d_hi),
        }
    }

    /// Build `P` directly from already-sound (outward-rounded) octahedral bounds
    /// — the production path, where the eight bounds come from NY's CROWN/IBP
    /// bound machinery. The caller guarantees Invariant P1.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bounds(
        l1: f64,
        u1: f64,
        l2: f64,
        u2: f64,
        s_lo: f64,
        s_hi: f64,
        d_lo: f64,
        d_hi: f64,
    ) -> Self {
        Self {
            l1,
            u1,
            l2,
            u2,
            s_lo,
            s_hi,
            d_lo,
            d_hi,
        }
    }

    /// Both neurons unstable (`l_i < 0 < u_i`) — the only case increment 1 emits
    /// a coupling facet for. Stable neurons collapse to the single-neuron
    /// identity/zero facets NY already applies.
    pub fn both_unstable(&self) -> bool {
        self.l1 < 0.0 && 0.0 < self.u1 && self.l2 < 0.0 && 0.0 < self.u2
    }

    /// §3 group-selection score: how hard `P` clips the box corners the
    /// independent-triangle product must cover but the joint hull discards
    /// (Theorem 2(iv)). It is the total corner slack the octahedral sum/difference
    /// bounds remove from the box:
    ///
    /// `score = max(0, (u1+u2) − s_hi) + max(0, s_lo − (l1+l2))`
    ///       `+ max(0, (u1−l2) − d_hi) + max(0, d_lo − (l1−u2))`.
    ///
    /// A large score ⇒ `P ⊊ box` strongly ⇒ Theorem 2(iv) strict tightening; a
    /// zero score ⇒ `P = box` (Lemma 1, no coupling gain). The selection heuristic
    /// (§3.2) sorts both-unstable candidate pairs by this and keeps the top-N.
    /// Never negative (the octahedral bounds are never looser than the box, by
    /// Invariant P1's outward-rounded sub/sum objectives over the same `U`).
    pub fn excluded_corner_score(&self) -> f64 {
        let sum_hi_gap = ((self.u1 + self.u2) - self.s_hi).max(0.0);
        let sum_lo_gap = (self.s_lo - (self.l1 + self.l2)).max(0.0);
        let dif_hi_gap = ((self.u1 - self.l2) - self.d_hi).max(0.0);
        let dif_lo_gap = (self.d_lo - (self.l1 - self.u2)).max(0.0);
        sum_hi_gap + sum_lo_gap + dif_hi_gap + dif_lo_gap
    }

    /// The eight half-space rows `A x ≤ b` of `P`, `x = (x_1, x_2)`.
    /// Row order matches `octahedral_P` in `validate_hull.py`:
    /// `x1≤u1, −x1≤−l1, x2≤u2, −x2≤−l2, x1+x2≤s_hi, −(x1+x2)≤−s_lo,
    ///  x1−x2≤d_hi, −(x1−x2)≤−d_lo`.
    fn constraints(&self) -> ([[f64; 2]; 8], [f64; 8]) {
        let a = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ];
        let b = [
            self.u1, -self.l1, self.u2, -self.l2, self.s_hi, -self.s_lo, self.d_hi, -self.d_lo,
        ];
        (a, b)
    }
}

// ===========================================================================
// §1.3(iii)  Lifted arrangement vertices  V
// ===========================================================================

/// Vertices of the 2D polytope `{x : A x ≤ b}` by pairwise-constraint
/// intersection + feasibility filtering. Mirrors the oracle's
/// `poly_vertices_2d` (including its `1e-9` feasibility slack and `1e-7` dedup).
fn poly_vertices_2d(a: &[[f64; 2]], b: &[f64]) -> Vec<[f64; 2]> {
    let m = a.len();
    let mut verts: Vec<[f64; 2]> = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            let det = a[i][0] * a[j][1] - a[i][1] * a[j][0];
            if det.abs() < 1e-12 {
                continue;
            }
            // Solve [a_i; a_j] p = [b_i; b_j].
            let px = (b[i] * a[j][1] - b[j] * a[i][1]) / det;
            let py = (a[i][0] * b[j] - a[j][0] * b[i]) / det;
            // Feasibility: A p ≤ b + 1e-9.
            let mut feasible = true;
            for k in 0..m {
                if a[k][0] * px + a[k][1] * py > b[k] + 1e-9 {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                verts.push([px, py]);
            }
        }
    }
    dedup_rows(&verts, 1e-7)
}

/// Lifted arrangement vertices of `P` cut by `{x_1=0}, {x_2=0}` and lifted with
/// the true ReLU on each orthant cell (Theorem 2(iii)). Mirrors the oracle's
/// `arrangement_lifted_vertices`. Returns rows `(x_1, x_2, y_1, y_2)`.
pub fn arrangement_lifted_vertices(p: &Octahedron2) -> Vec<[f64; 4]> {
    let (base_a, base_b) = p.constraints();
    let mut lifted: Vec<[f64; 4]> = Vec::new();
    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            // orthant cell: s_i·x_i ≥ 0  ->  −s_i·x_i ≤ 0
            let mut a: Vec<[f64; 2]> = base_a.to_vec();
            let mut b: Vec<f64> = base_b.to_vec();
            a.push([-s1, 0.0]);
            b.push(0.0);
            a.push([0.0, -s2]);
            b.push(0.0);
            for x in poly_vertices_2d(&a, &b) {
                let y1 = if s1 > 0.0 { x[0] } else { 0.0 };
                let y2 = if s2 > 0.0 { x[1] } else { 0.0 };
                lifted.push([x[0], x[1], y1, y2]);
            }
        }
    }
    dedup_rows4(&lifted, 1e-7)
}

fn dedup_rows(rows: &[[f64; 2]], atol: f64) -> Vec<[f64; 2]> {
    let mut keep: Vec<[f64; 2]> = Vec::new();
    for &v in rows {
        if !keep
            .iter()
            .any(|w| (v[0] - w[0]).abs() <= atol && (v[1] - w[1]).abs() <= atol)
        {
            keep.push(v);
        }
    }
    keep
}

fn dedup_rows4(rows: &[[f64; 4]], atol: f64) -> Vec<[f64; 4]> {
    let mut keep: Vec<[f64; 4]> = Vec::new();
    for &v in rows {
        if !keep
            .iter()
            .any(|w| (0..4).all(|k| (v[k] - w[k]).abs() <= atol))
        {
            keep.push(v);
        }
    }
    keep
}

// ===========================================================================
// §1.3 / §4  Facet enumeration in ℝ⁴ with certified-outward RHS
// ===========================================================================

/// A stored group facet `a·(x_1,x_2,y_1,y_2) ≤ b` for a 2-ReLU group.
///
/// `a` and `b` are stored in **f32** for alignment with `LinearBounds`.  Verdict
/// authority requires `b` to come from [`ExactRelu2Support::certify_normal`],
/// which proves `a·w ≤ b` for every `w=(x,ReLU(x))`, `x∈P`.  The legacy
/// [`hull_facets`] constructor establishes the inequality only over the supplied
/// floating vertex slice and is research-only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Facet {
    pub a: [f32; 4],
    pub b: f32,
}

fn check_exact_facet_deadline(deadline: Option<Instant>, stage: &str) -> NyResult<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "exact k=2 facet production: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

impl Facet {
    /// Residual `a·w − b` at a point `w = (x_1,x_2,y_1,y_2)` (≤ 0 ⇔ satisfied),
    /// evaluated in f64 for a well-scaled test.
    pub fn residual(&self, w: &[f64; 4]) -> f64 {
        let mut s = -(self.b as f64);
        for k in 0..4 {
            s += self.a[k] as f64 * w[k];
        }
        s
    }

    /// A facet **couples** the two neurons iff it carries a nonzero coefficient
    /// on some variable of neuron 1 (`x_1` or `y_1`) AND on some variable of
    /// neuron 2 (`x_2` or `y_2`). These are the value-add facets; the rest are
    /// the single-neuron triangle facets NY already applies (§1.3.2).
    pub fn is_coupling(&self) -> bool {
        let n1 = self.a[0] != 0.0 || self.a[2] != 0.0;
        let n2 = self.a[1] != 0.0 || self.a[3] != 0.0;
        n1 && n2
    }
}

/// The 4D "generalized cross product": the vector orthogonal to `d1,d2,d3`.
/// Component `i` is `(-1)^i` times the 3×3 determinant of the matrix formed by
/// the columns `{0,1,2,3}\{i}` of the 3×4 matrix `[d1; d2; d3]`.
fn normal4(d1: &[f64; 4], d2: &[f64; 4], d3: &[f64; 4]) -> [f64; 4] {
    let rows = [d1, d2, d3];
    let mut n = [0.0f64; 4];
    for i in 0..4 {
        // columns other than i
        let cols: Vec<usize> = (0..4).filter(|&c| c != i).collect();
        let m = [
            [rows[0][cols[0]], rows[0][cols[1]], rows[0][cols[2]]],
            [rows[1][cols[0]], rows[1][cols[1]], rows[1][cols[2]]],
            [rows[2][cols[0]], rows[2][cols[1]], rows[2][cols[2]]],
        ];
        let det3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        n[i] = if i % 2 == 0 { det3 } else { -det3 };
    }
    n
}

/// Propose facet normals for `conv(V)` by brute force over 4-vertex subsets.
///
/// A hyperplane spanned by 4 affinely-independent vertices is a facet iff all
/// other vertices lie weakly on one side. Degenerate (near-coplanar) subsets
/// give a tiny/non-finite normal and are dropped.  This tolerance-based geometry
/// has **no verdict authority**: each returned stored-`f32` normal is only a
/// proposal.  [`ExactRelu2Support::certify_normal`] independently establishes a
/// valid support half-space without trusting this vertex set, its feasibility
/// slack, or its deduplication.
pub fn proposed_hull_normals(verts: &[[f64; 4]]) -> Vec<[f32; 4]> {
    proposed_hull_normals_with_deadline(verts, None)
        .expect("an unbounded facet proposal cannot exceed a deadline")
}

/// Deadline-aware form of [`proposed_hull_normals`].
///
/// Geometry remains proposal-only. The deadline is polled through the
/// combinatorial four-vertex enumeration so a cut request cannot spend past its
/// private budget before reaching the exact checker.
pub fn proposed_hull_normals_with_deadline(
    verts: &[[f64; 4]],
    deadline: Option<Instant>,
) -> NyResult<Vec<[f32; 4]>> {
    check_exact_facet_deadline(deadline, "before normal enumeration")?;
    let n = verts.len();
    if n < 5 {
        // Lower-dimensional lifted set (weakly-correlated / degenerate pair):
        // no full-dimensional hull, so emit no coupling facet. Sound: fewer
        // half-spaces only enlarges the relaxation (design R7).
        check_exact_facet_deadline(deadline, "before publishing empty normal list")?;
        return Ok(Vec::new());
    }
    let scale = verts
        .iter()
        .flat_map(|v| v.iter())
        .fold(1.0f64, |m, &c| m.max(c.abs()));
    let side_tol = 1e-9 * (1.0 + scale);

    // Collect unique oriented facet directions (f64, unit-normalized).
    let mut dirs: Vec<([f64; 4], f64)> = Vec::new();
    for i0 in 0..n {
        for i1 in (i0 + 1)..n {
            for i2 in (i1 + 1)..n {
                for i3 in (i2 + 1)..n {
                    check_exact_facet_deadline(deadline, "during normal enumeration")?;
                    let p0 = verts[i0];
                    let d1 = sub4(&verts[i1], &p0);
                    let d2 = sub4(&verts[i2], &p0);
                    let d3 = sub4(&verts[i3], &p0);
                    let mut a = normal4(&d1, &d2, &d3);
                    let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
                    if !norm.is_finite() || norm < 1e-9 {
                        continue; // degenerate 4-subset — drop (sound)
                    }
                    for c in &mut a {
                        *c /= norm;
                    }
                    let off = dot4(&a, &p0);
                    // Facet test: every vertex weakly on one side.
                    let mut all_le = true;
                    let mut all_ge = true;
                    for v in verts {
                        let s = dot4(&a, v) - off;
                        if s > side_tol {
                            all_le = false;
                        }
                        if s < -side_tol {
                            all_ge = false;
                        }
                    }
                    if !all_le && !all_ge {
                        continue; // spanning hyperplane, not supporting
                    }
                    // Orient as an upper bound: a·w ≤ off.
                    let (a, off) = if all_le { (a, off) } else { (neg4(&a), -off) };
                    if !dirs.iter().any(|(da, doff)| {
                        (0..4).all(|k| (da[k] - a[k]).abs() <= 1e-6) && (doff - off).abs() <= 1e-6
                    }) {
                        dirs.push((a, off));
                    }
                }
            }
        }
    }

    // The f64 geometry chooses directions only.  Casting fixes the exact normal
    // that a separate checker must certify.
    let mut normals = Vec::with_capacity(dirs.len());
    for (a64, _) in dirs {
        check_exact_facet_deadline(deadline, "during stored-normal conversion")?;
        let a32 = [a64[0] as f32, a64[1] as f32, a64[2] as f32, a64[3] as f32];
        if a32.iter().all(|c| c.is_finite()) {
            normals.push(a32);
        }
    }
    check_exact_facet_deadline(deadline, "after normal enumeration")?;
    Ok(normals)
}

/// Legacy research facet producer over a supplied lifted vertex set.
///
/// The RHS is outward relative to the supplied `verts`, but the vertex producer
/// itself uses floating feasibility tolerances and tolerance-based deduplication.
/// Therefore this function is not an authority certificate for all of `P`; all
/// verdict gates consuming it remain hard-quarantined.  New authority work must
/// treat [`proposed_hull_normals`] as proposals and use
/// [`ExactRelu2Support::certify_normal`] for the RHS.
pub fn hull_facets(verts: &[[f64; 4]]) -> Vec<Facet> {
    proposed_hull_normals(verts)
        .into_iter()
        .map(|a32| {
            let mut max_dot = f64::NEG_INFINITY;
            for v in verts {
                let mut s = 0.0f64;
                for k in 0..4 {
                    s += a32[k] as f64 * v[k];
                }
                if s > max_dot {
                    max_dot = s;
                }
            }
            let b = next_up_f32(max_dot as f32);
            Facet { a: a32, b }
        })
        .collect()
}

fn sub4(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
fn neg4(a: &[f64; 4]) -> [f64; 4] {
    [-a[0], -a[1], -a[2], -a[3]]
}
fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Legacy research producer: `P` → tolerance-derived lifted `V` → coupling
/// facets.  Returns only facets that couple the two neurons.  Its RHS is not an
/// authority certificate for all of `P`; use [`certified_coupling_facets_exact`]
/// for the exact-support repair seam.  Empty when the pair is not both-unstable
/// or the lifted set is degenerate.
pub fn coupling_facets(p: &Octahedron2) -> Vec<Facet> {
    if !p.both_unstable() {
        return Vec::new();
    }
    let verts = arrangement_lifted_vertices(p);
    hull_facets(&verts)
        .into_iter()
        .filter(Facet::is_coupling)
        .collect()
}

/// Default-unused exact-support repair path for k=2 coupling facets that
/// preserves exact-checker evidence and its support domain.
///
/// Floating geometry supplies only stored-`f32` normal proposals.  Every RHS is
/// rebuilt by exact maximization over `P` intersected with all four ReLU
/// orthants.  A failed proposal is dropped, which can only weaken the relaxation.
/// This function is intentionally not wired to any production authority gate.
pub fn certified_coupling_facet_certificates_exact(
    p: &Octahedron2,
) -> Vec<ExactRelu2FacetCertificate> {
    certified_coupling_facet_certificates_exact_with_deadline(p, None)
        .expect("an unbounded exact facet check cannot exceed a deadline")
}

/// Deadline-aware form of
/// [`certified_coupling_facet_certificates_exact`].
///
/// The same request-local deadline covers proposal enumeration and the
/// per-normal exact-support checks. Expiration publishes no partial certificate
/// list: the caller receives [`NyError::DeadlineExceeded`] and must skip the
/// candidate.
pub fn certified_coupling_facet_certificates_exact_with_deadline(
    p: &Octahedron2,
    deadline: Option<Instant>,
) -> NyResult<Vec<ExactRelu2FacetCertificate>> {
    check_exact_facet_deadline(deadline, "before support construction")?;
    if !p.both_unstable() {
        check_exact_facet_deadline(deadline, "before publishing empty certificate list")?;
        return Ok(Vec::new());
    }
    let checker = ExactRelu2Support::new(p);
    check_exact_facet_deadline(deadline, "after support construction")?;
    let Some(checker) = checker else {
        check_exact_facet_deadline(deadline, "before publishing empty certificate list")?;
        return Ok(Vec::new());
    };
    let verts = arrangement_lifted_vertices(p);
    check_exact_facet_deadline(deadline, "after lifted-vertex construction")?;
    let normals = proposed_hull_normals_with_deadline(&verts, deadline)?;
    let mut certificates = Vec::with_capacity(normals.len());
    for normal in normals {
        check_exact_facet_deadline(deadline, "before exact support check")?;
        if let Some(certificate) = checker.certify_normal_certificate(normal) {
            if certificate.facet().is_coupling() {
                certificates.push(certificate);
            }
        }
        check_exact_facet_deadline(deadline, "after exact support check")?;
    }
    check_exact_facet_deadline(deadline, "after exact support checks")?;
    Ok(certificates)
}

/// Compatibility projection of
/// [`certified_coupling_facet_certificates_exact`] into raw facets.
///
/// Raw [`Facet`] values deliberately carry no proof identity and remain
/// unsuitable as inputs to a verdict-bearing carrier. Existing dark research
/// callers retain this view; new authority work must consume the certificate
/// function and bind each certificate's support domain to its request.
pub fn certified_coupling_facets_exact(p: &Octahedron2) -> Vec<Facet> {
    certified_coupling_facet_certificates_exact(p)
        .into_iter()
        .map(|certificate| certificate.facet())
        .collect()
}

// ===========================================================================
// §2.3  Group-facet constraint + pool + β_c Adam multiplier
// ===========================================================================

/// Which of a neuron's two variables a term refers to: the ReLU input
/// (pre-activation `x_i`) or output (post-activation `y_i`). Mirrors the design
/// `MnVar`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MnVar {
    PreActivation,
    PostActivation,
}

/// Outcome of the pre-relaxation half of one multi-neuron Lagrangian term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MnInjectOutcome {
    /// This group is inactive at the current node; no mutation or completion is
    /// required.
    Inert,
    /// The group committed its post-activation half. The caller must complete
    /// the pre-activation and price half after the ReLU relaxation.
    Injected,
    /// Validation failed before mutation, so the entire group was omitted.
    Skipped,
}

/// A single term of a multi-neuron group facet — a real coefficient on one
/// neuron's pre- or post-activation variable (design `MnTerm`). Generalizes
/// `GraphCutTerm`, which is post/indicator-only with coefficient ∈ {±1}.
#[derive(Clone, Debug)]
pub struct MnTerm {
    pub node_name: String,
    pub neuron_idx: usize,
    pub var: MnVar,
    pub coefficient: f32,
}

/// A multi-neuron group constraint `Σ a_i x_i + Σ g_i y_i ≤ bias`, valid on the
/// group's reachable set (Invariant MN), with a `β_c ≥ 0` Adam multiplier.
///
/// Mirrors [`GraphCuttingPlane`](crate::beta_crown::bab_cuts::GraphCuttingPlane)
/// verbatim for the multiplier + optimizer state — [`update_beta_adam`] is a
/// copy of `update_lambda_adam` including the `clamp(0, MAX_BETA)` projection,
/// the `β ≥ 0` invariant, and the NaN-guard reset. This is the only thing that
/// keeps the relaxation sound at optimization time (design R6).
///
/// [`update_beta_adam`]: MultiNeuronConstraint::update_beta_adam
#[derive(Debug, Clone)]
pub struct MultiNeuronConstraint {
    pub(crate) terms: Vec<MnTerm>,
    pub(crate) bias: f32,
    pub(crate) beta: f32,
    pub(crate) beta_grad: f32,
    pub(crate) beta_m: f32,
    pub(crate) beta_v: f32,
    /// Eviction/freshness tracking, mirroring `GraphCuttingPlane::metadata`.
    /// Reserved for the per-domain pool lifecycle (design §2.3 / R8) exercised
    /// once the k-ReLU conv-stack producer (increment 2) populates the pool.
    #[allow(dead_code)]
    pub(crate) metadata: CutMetadata,
}

/// Upper clamp on `β_c`, mirroring `GraphCuttingPlane`'s `MAX_LAMBDA`.
const MAX_BETA: f32 = 10.0;

impl MultiNeuronConstraint {
    /// Create a validated group constraint. `beta` must be `≥ 0` and finite;
    /// `bias` and every term coefficient must be finite (mirrors
    /// `GraphCuttingPlane::new`). Optimizer state initialized to zero.
    ///
    /// SOUNDNESS: additionally enforces the SINGLE-ANCHOR invariant — `terms`
    /// must be non-empty and every term must name the SAME ReLU node. That node
    /// is the group's [`anchor`](MultiNeuronConstraint::anchor), the one place
    /// its `−β_c·b_c` price may be charged (see
    /// [`inject_pre_terms_after_relu`](MultiNeuronConstraint::inject_pre_terms_after_relu)).
    pub fn new(terms: Vec<MnTerm>, bias: f32, beta: f32) -> Result<Self, String> {
        if !bias.is_finite() {
            return Err(format!(
                "MultiNeuronConstraint bias must be finite, got {bias}"
            ));
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(format!(
                "MultiNeuronConstraint beta must be finite and >= 0, got {beta}"
            ));
        }
        for (i, t) in terms.iter().enumerate() {
            if !t.coefficient.is_finite() {
                return Err(format!(
                    "MultiNeuronConstraint term[{i}] coefficient must be finite, got {}",
                    t.coefficient
                ));
            }
        }
        // SOUNDNESS — SINGLE-ANCHOR invariant (guards the §2.2 step-3 price).
        //
        // The Lagrangian embedding adds ONE constant `−β_c·b_c` per group to the
        // margin whose LOWER bound is computed, and `inject_pre_terms_after_relu`
        // is where that constant is folded. The backward sweep visits every ReLU
        // node once and offers EVERY pooled group at EVERY node, so the fold must
        // be pinned to a single, unambiguous node — the group's ANCHOR.
        //
        // A group with NO terms has no anchor; a group whose terms straddle two
        // ReLU nodes has two. Either way the "once per group" contract is not
        // expressible, and paying the price more than once with `b_c < 0` RAISES
        // the lower bound above what the Lagrangian justifies — a strict FALSE
        // TIGHTENING (a false VERIFIED). Reject both here so `anchor()` is
        // total and the price is provably once-per-group.
        let Some(anchor) = terms.first().map(|t| t.node_name.clone()) else {
            return Err(
                "MultiNeuronConstraint must carry at least one term (no anchor node)".to_string(),
            );
        };
        if let Some(t) = terms.iter().find(|t| t.node_name != anchor) {
            return Err(format!(
                "MultiNeuronConstraint terms must all live on ONE ReLU node (the group anchor); \
                 got '{anchor}' and '{}'",
                t.node_name
            ));
        }
        Ok(Self {
            terms,
            bias,
            beta,
            beta_grad: 0.0,
            beta_m: 0.0,
            beta_v: 0.0,
            metadata: CutMetadata::new(0, CutKind::Proactive),
        })
    }

    /// Build a group constraint from a geometric [`Facet`] over `(x_1,x_2,y_1,
    /// y_2)` and the two neuron identities. Only nonzero-coefficient terms are
    /// kept.
    pub fn from_facet_for_group(
        facet: &Facet,
        node1: &str,
        idx1: usize,
        node2: &str,
        idx2: usize,
    ) -> Result<Self, String> {
        let mut terms = Vec::new();
        let slots = [
            (node1, idx1, MnVar::PreActivation, facet.a[0]),
            (node2, idx2, MnVar::PreActivation, facet.a[1]),
            (node1, idx1, MnVar::PostActivation, facet.a[2]),
            (node2, idx2, MnVar::PostActivation, facet.a[3]),
        ];
        for (name, idx, var, coeff) in slots {
            if coeff != 0.0 {
                terms.push(MnTerm {
                    node_name: name.to_string(),
                    neuron_idx: idx,
                    var,
                    coefficient: coeff,
                });
            }
        }
        Self::new(terms, facet.b, 0.0)
    }

    /// The single ReLU node every term of this group lives on — the group's
    /// ANCHOR, and the ONLY node at which its `−β_c·b_c` price may be charged.
    ///
    /// Total by construction: [`MultiNeuronConstraint::new`] rejects a term-less
    /// group and a group whose terms straddle two nodes, so the first term's
    /// `node_name` IS the anchor. The `""` arm is unreachable and fails closed
    /// (no real graph node is named `""`, so nothing would ever be charged).
    pub fn anchor(&self) -> &str {
        self.terms
            .first()
            .map(|t| t.node_name.as_str())
            .unwrap_or("")
    }

    pub fn beta(&self) -> f32 {
        self.beta
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    // =======================================================================
    // §2.2  LIVE backward injection — the soundness-critical pre/post routing
    // =======================================================================
    //
    // The Lagrangian embedding (§2.1) adds `+β_c·(a·x + g·y − b)` to the margin
    // whose LOWER bound is computed. As the backward sweep reaches the group's
    // ReLU node, that term splits across the pre- vs post-relaxation carrier —
    // getting this split RIGHT is the soundness-critical step (design R5):
    //
    //   step 1 (BEFORE relaxation): the post-activation term `+β_c·g_i` is added
    //     to the ReLU-OUTPUT column of the incoming carrier `node_lb`, so it
    //     rides `propagate_linear_with_alpha` and is relaxed AS a post-activation
    //     (the relaxation picks the lower facet `y≥αx` or the upper chord by the
    //     net coefficient's sign — exactly like a β-split rides the relaxation).
    //   step 2 (AFTER relaxation): the pre-activation term `+β_c·a_i` is added to
    //     the ReLU-INPUT column of the relaxed carrier `new_lb` — a DIRECT linear
    //     term on `x_i` that bypasses the ReLU, where the single-neuron β-split
    //     adds its `±β` today.
    //   step 3 (AFTER relaxation): the constant `−β_c·b_c` is folded into the
    //     lower bias, rounded OUTWARD.
    //
    // Both carriers touch only `lower_a` / `lower_b` (the verification side); the
    // upper bound is irrelevant to the margin LB and is left unchanged. Every
    // mutation folds its f32 rounding into the certified coeff-err (R4) via the
    // inc1 primitives `add_to_lower_column` / `add_lower_bias_outward`.
    //
    // Convention (ENFORCED by `new`'s single-anchor invariant): for a group at
    // ReLU node `R`, every term carries `node_name == R` and `neuron_idx` = the
    // neuron's flat column in `R` (ReLU is elementwise, so the input and output
    // columns share the index). `R` is the group's ANCHOR — step 3's constant is
    // charged there and NOWHERE else, because the sweep offers every pooled
    // group at every ReLU node.

    /// Validate all terms at the anchor before either half of the Lagrangian
    /// contribution mutates a carrier.
    ///
    /// The coefficient mass and the `−β_c·b_c` price form one indivisible
    /// contribution. An out-of-range term must therefore suppress the whole
    /// group, not silently drop one coefficient while still paying its price.
    fn injectable_at(&self, relu_node: &str, ncols: usize) -> bool {
        self.bias.is_finite()
            && self.anchor() == relu_node
            && self
                .terms
                .iter()
                .all(|term| term.neuron_idx < ncols && term.coefficient.is_finite())
    }

    /// §2.2 step 1 — inject post-activation terms `+β_c·g_i` on the ReLU
    /// output columns before relaxation.
    ///
    /// The product is computed in f64 and rounded down. Since every ReLU output
    /// is non-negative, this is the conservative coefficient direction. The
    /// f32 carrier-addition gap is retained in its certified coefficient-error
    /// matrix. [`MnInjectOutcome::Injected`] obligates the caller to invoke
    /// [`inject_pre_terms_after_relu`](Self::inject_pre_terms_after_relu) after
    /// relaxation; a failed completion degrades the lower carrier before it is
    /// returned to the caller.
    #[must_use]
    pub(crate) fn inject_post_terms_before_relu(
        &self,
        node_lb: &mut LinearBounds,
        relu_node: &str,
        beta: f32,
    ) -> MnInjectOutcome {
        if !beta.is_finite() || beta <= 0.0 || self.anchor() != relu_node {
            return MnInjectOutcome::Inert;
        }
        if !self.injectable_at(relu_node, node_lb.num_inputs()) {
            return MnInjectOutcome::Skipped;
        }
        let has_post = self
            .terms
            .iter()
            .any(|term| term.var == MnVar::PostActivation);
        if has_post {
            node_lb.ensure_lower_coeff_err_tracking();
        }
        for term in &self.terms {
            if term.var != MnVar::PostActivation {
                continue;
            }
            let coeff = ny_core::f64_to_f32_down(
                ny_core::f32_to_f64_exact(beta) * ny_core::f32_to_f64_exact(term.coefficient),
            );
            // A floating-point comparison may classify a subnormal as zero
            // when DAZ is enabled.  In particular, silently dropping a
            // negative post coefficient would increase a lower functional on
            // y >= 0.  Inspect the representation instead.
            if coeff.to_bits() & 0x7fff_ffff == 0 {
                continue;
            }
            if !coeff.is_finite()
                || !node_lb.add_to_lower_column_with_err(term.neuron_idx, coeff, 0.0)
            {
                node_lb.degrade_lower_to_vacuous();
                return MnInjectOutcome::Injected;
            }
        }
        MnInjectOutcome::Injected
    }

    /// §2.2 steps 2+3 — inject the pre-activation terms `+β_c·a_i` on the
    /// ReLU-INPUT columns of the relaxed carrier `new_lb` (direct on `x_i`,
    /// bypassing the ReLU) AND fold the constant `−β_c·b_c` into the lower bias
    /// (outward). Call at the ReLU node's backward step AFTER the relaxation,
    /// before accumulating `new_lb` into the input.
    ///
    /// Call this only to complete an [`MnInjectOutcome::Injected`] result from
    /// the same group, node, and bit-identical `beta`. The pre-activation
    /// product is sign-indefinite, so its f64-to-f32 residual is explicitly
    /// carried as coefficient error. The price `β_c·b_c` is rounded up before
    /// subtraction, ensuring the lower relaxation never pays less than the
    /// certificate requires.
    ///
    /// Returns `false` when completion is impossible and replaces the lower
    /// relaxation with a vacuous one before returning. This internal fail-closed
    /// action keeps direct callers sound even if they only honor `#[must_use]`
    /// diagnostically; the production coordinator repeats it defensively.
    #[must_use]
    pub(crate) fn inject_pre_terms_after_relu(
        &self,
        new_lb: &mut LinearBounds,
        relu_node: &str,
        beta: f32,
    ) -> bool {
        if !beta.is_finite() || beta <= 0.0 || self.anchor() != relu_node {
            return true;
        }
        if !self.injectable_at(relu_node, new_lb.num_inputs()) {
            new_lb.degrade_lower_to_vacuous();
            return false;
        }
        let has_pre = self
            .terms
            .iter()
            .any(|term| term.var == MnVar::PreActivation);
        if has_pre {
            new_lb.ensure_lower_coeff_err_tracking();
        }
        for term in &self.terms {
            if term.var != MnVar::PreActivation {
                continue;
            }
            let exact =
                ny_core::f32_to_f64_exact(beta) * ny_core::f32_to_f64_exact(term.coefficient);
            let coeff = exact as f32;
            if !coeff.is_finite() {
                new_lb.degrade_lower_to_vacuous();
                return false;
            }
            let residual = ny_core::f64_to_f32_up((ny_core::f32_to_f64_exact(coeff) - exact).abs());
            if !residual.is_finite() {
                new_lb.degrade_lower_to_vacuous();
                return false;
            }
            // DAZ may make an arithmetic comparison erase a non-zero
            // subnormal coefficient or residual before the certified column
            // mutation sees it.  Bitwise zero classification preserves the
            // exact binary32 operands in every floating-point environment.
            if coeff.to_bits() & 0x7fff_ffff == 0 && residual.to_bits() & 0x7fff_ffff == 0 {
                continue;
            }
            if !new_lb.add_to_lower_column_with_err(term.neuron_idx, coeff, residual) {
                new_lb.degrade_lower_to_vacuous();
                return false;
            }
        }
        let price = ny_core::f64_to_f32_up(
            ny_core::f32_to_f64_exact(beta) * ny_core::f32_to_f64_exact(self.bias),
        );
        if !price.is_finite() {
            new_lb.degrade_lower_to_vacuous();
            return false;
        }
        new_lb.add_lower_bias_outward(-price);
        true
    }

    pub fn terms(&self) -> &[MnTerm] {
        &self.terms
    }

    /// Set `β_c` with the `≥ 0 && finite` invariant (mirrors
    /// `GraphCuttingPlane::set_lambda`).
    pub fn set_beta(&mut self, value: f32) -> Result<(), String> {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("beta must be finite and >= 0, got {value}"));
        }
        self.beta = value;
        Ok(())
    }

    /// Set the gradient `∂LB/∂β_c` (NaN/Inf reset to 0 to protect Adam moments,
    /// mirrors `set_lambda_grad`).
    pub fn set_beta_grad(&mut self, value: f32) {
        self.beta_grad = if value.is_finite() { value } else { 0.0 };
    }

    pub fn zero_grad(&mut self) {
        self.beta_grad = 0.0;
    }

    /// Adam step on `β_c` — a VERBATIM copy of
    /// `GraphCuttingPlane::update_lambda_adam` (maximize, project to
    /// `[0, MAX_BETA]`, NaN-guard reset). The `β ≥ 0` projection is the soundness
    /// guard R6: a negative multiplier would turn a valid `h_c ≥ 0` into an
    /// unsound subtraction.
    pub fn update_beta_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: usize) {
        let t = t.max(1);
        self.beta_m = beta1 * self.beta_m + (1.0 - beta1) * self.beta_grad;
        self.beta_v = beta2 * self.beta_v + (1.0 - beta2) * self.beta_grad * self.beta_grad;
        let t_f32 = t as f32;
        let m_hat = self.beta_m / (1.0 - beta1.powf(t_f32)).max(f32::EPSILON);
        let v_hat = self.beta_v / (1.0 - beta2.powf(t_f32)).max(f32::EPSILON);
        self.beta += lr * m_hat / (v_hat.sqrt() + epsilon);
        if !self.beta.is_finite() || !self.beta_m.is_finite() || !self.beta_v.is_finite() {
            self.beta = 0.0;
            self.beta_m = 0.0;
            self.beta_v = 0.0;
            self.beta_grad = 0.0;
        } else {
            self.beta = self.beta.clamp(0.0, MAX_BETA);
        }
    }
}

/// A per-domain pool of multi-neuron group constraints, mirroring
/// [`GraphCutPool`](crate::beta_crown::bab_cuts::GraphCutPool) (minimal for
/// increment 1: no eviction, a single group).
#[derive(Debug, Clone, Default)]
pub struct MultiNeuronPool {
    groups: Vec<MultiNeuronConstraint>,
    max_groups: usize,
}

impl MultiNeuronPool {
    pub fn new(max_groups: usize) -> Self {
        Self {
            groups: Vec::new(),
            max_groups,
        }
    }

    pub fn push(&mut self, group: MultiNeuronConstraint) -> bool {
        if self.max_groups != 0 && self.groups.len() >= self.max_groups {
            return false;
        }
        self.groups.push(group);
        true
    }

    pub fn groups(&self) -> &[MultiNeuronConstraint] {
        &self.groups
    }

    pub fn groups_mut(&mut self) -> &mut [MultiNeuronConstraint] {
        &mut self.groups
    }

    pub fn zero_grad(&mut self) {
        for g in &mut self.groups {
            g.zero_grad();
        }
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

#[cfg(test)]
mod tests;
