// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Structure-aware SOUND lower/upper bound for Two-Level-Lattice (TLL) networks.
//!
//! # Why this exists
//!
//! The tllverifybench_2023 benchmark realizes a scalar CPWA function
//!
//! ```text
//!   y(x) = max_j  min_{i in S_j}  L_i(x)
//! ```
//!
//! where `x` is a 2-D input in a box, `L_i(x) = a_i . x + b_i` are `N` AFFINE
//! "local linear functions" (the `linearLayer`), and the wide `minBank*/maxBank*`
//! ReLU banks realize the min/max lattice over selector sets `S_j` decoded from
//! the one-hot `selectionLayer`.
//!
//! NY's GENERIC relaxations (IBP / CROWN / alpha-CROWN) push the min/max encoding
//! `min(a,b)=a-relu(a-b)`, `max(a,b)=a+relu(b-a)` through the wide banks + 12
//! ReLU layers and lose the correlation that every `L_i` is a function of the
//! SAME 2-D `x`. On the `N=56` instance the root output lower bound bottoms out
//! at ~-199 vs a true min of -2.369 - far too loose to clear the -3.04 threshold,
//! so the generic BaB never converges inside budget.
//!
//! # The bound
//!
//! Because the input is only 2-D we bound `y` DIRECTLY from the affine `L_i`,
//! preserving the shared-`x` correlation. Over any sub-box (cell) `C`:
//!
//! ```text
//!   min_{x in C} y(x) = min_x max_j min_{i in S_j} L_i(x)
//!                    >= max_j min_{i in S_j}  min_{x in C} L_i(x)     (max-min <= min-max)
//!                     = max_j min_{i in S_j}  c_i(C)
//! ```
//!
//! where `c_i(C)` = box-min of the affine `L_i` over `C` (EXACT, at a corner).
//! Splitting `[lo,hi]^2` into an `nx x nx` grid and taking the MIN over cells of
//! the per-cell bound converges to the true min from BELOW as the grid refines
//! (the max-min<=min-max slack vanishes as cells shrink). A symmetric derivation
//! gives a sound UPPER bound. All floating-point steps round OUTWARD.
//!
//! # Soundness (MOAT-critical - this bound FEEDS the unsat verdict)
//!
//! - Every affine corner value rounds OUTWARD (lower rounds DOWN, upper rounds UP),
//!   so `lb <= true min` and `ub >= true max` of the REAL-valued network.
//! - A raw-protobuf identity proof checks the sole FLOAT input/output, every
//!   node and edge, all authored FLOAT tensors, the bit-exact selector, and
//!   every algebraic min/max gadget. Finite NY/ORT samples remain diagnostics
//!   only and cannot establish or change the decoded whole-domain identity.
//! - We only ever return `Unsat`, and only when the sound bound strictly clears
//!   the threshold - so a too-LOOSE bound merely forgoes the win (falls through),
//!   never a false verdict.
//!
//! Exact source authentication is wired for the property input box and
//! directional threshold as well as the model. The verdict route remains
//! release-dark while the complete chain undergoes qualification; no rounded
//! property constant can reach the bound or verdict comparison.

use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ny_core::f32_to_f64_exact;
use ny_onnx::{load_onnx_with_config, OnnxLoadConfig};
use ny_propagate::Interval64;

use super::vnncomp::VnncompResult;
use ny_onnx::{CompoundNodePolicy, GraphNetworkOptions};

mod model_identity;
mod property_identity;

/// Direction of the single output threshold constraint (on `Y_0`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ThreshDir {
    /// unsafe region is `Y_0 <= c` (or `< c`): UNSAT iff `min Y_0 > c`.
    Le,
    /// unsafe region is `Y_0 >= c` (or `> c`): UNSAT iff `max Y_0 < c`.
    Ge,
}

/// Decoded TLL structure: the `N` affine functions and the selector sets.
struct TllStructure {
    /// `a[i] = [a_i0, a_i1]` - affine slope of `L_i` (2-D input only).
    a: Vec<[f64; 2]>,
    /// `b[i]` - affine bias of `L_i`.
    b: Vec<f64>,
    /// Selector sets `S_j` (indices into `0..N`).
    groups: Vec<Vec<usize>>,
    /// `true`: `y = max_j min_{i} L_i` (the authenticated realization);
    /// `false`: retained only for symmetric bound-level tests.
    max_of_min: bool,
}

/// Relative outward-rounding slack (>> the ~4u accumulated f64 error for a
/// 2-term affine; u = 2^-53, `EPSILON` = 2^-52).
const OUTWARD: f64 = 32.0 * f64::EPSILON;

/// Bound worst-case work before entering the per-cell lattice reductions. The
/// largest qualified official model has 2,031 group memberships, requiring
/// 831,897,600 membership comparisons at the final 640x640 grid. One billion
/// preserves that route while rejecting adversarial authenticated shapes that
/// could otherwise monopolize the verifier before the generic fallback.
const MAX_GRID_LATTICE_WORK: usize = 1_000_000_000;

/// Both authored model and property sources now have independent raw identity
/// proofs. Publication remains deliberately dark during release qualification;
/// no environment variable may bypass this compile-time gate.
const TLL_PROPERTY_SOURCE_AUTHENTICATED: bool = false;

/// Round `v` DOWN by a slack proportional to `mag`, an upper bound on the
/// magnitude SUM of the addends that produced `v` (#tll-round-slack-harden).
///
/// Under catastrophic cancellation (`|v| << mag`, e.g. ~1e7-magnitude
/// coefficients summing to ~0) the accumulated f64 error scales with `mag`,
/// NOT `|v|` — a `|v|`-relative slack under-covers it. Since `mag >= |fl(v)|`
/// up to rounding, this is strictly MORE outward than the `|v|`-relative
/// slack it generalizes, so it stays sound in every coefficient regime
/// (upgrade of the verifier-flagged benchmark-regime observation to a proof;
/// for tll's O(1) coefficients the extra slack is ~1e-14 against a 0.67
/// verdict margin — no behavior change).
#[inline]
fn round_down_mag(v: f64, mag: f64) -> f64 {
    v - (mag * OUTWARD + f64::MIN_POSITIVE)
}

#[inline]
fn round_up_mag(v: f64, mag: f64) -> f64 {
    v + (mag * OUTWARD + f64::MIN_POSITIVE)
}

/// `|v|`-relative outward rounding — sound ONLY where the f64 error of the
/// producing computation is relative to `|v|` itself: a single product
/// `a * x` (one rounding, error <= u*|a*x|). NOT sound for sums under
/// cancellation — use [`round_down_mag`]/[`round_up_mag`] there.
#[inline]
fn round_down(v: f64) -> f64 {
    round_down_mag(v, v.abs())
}

#[inline]
fn round_up(v: f64) -> f64 {
    round_up_mag(v, v.abs())
}

/// Attempt a structure-aware SOUND unsat verdict for a TLL net.
///
/// Returns `Some(VnncompResult::Unsat)` ONLY when the decoded, self-checked,
/// outward-rounded structure bound STRICTLY clears the property threshold.
/// Returns `None` in every other case (not a TLL net, unsupported property,
/// self-check/enclosure failure, or bound too loose) so the caller falls
/// through to the generic verifier with no behavior change.
pub(crate) fn try_tll_unsat(onnx: &Path, vnnlib: &Path) -> Option<VnncompResult> {
    // Keep publication unreachable during qualification of the complete raw
    // model + exact-decimal property authentication chain.
    if !TLL_PROPERTY_SOURCE_AUTHENTICATED {
        return None;
    }
    try_tll_unsat_authenticated(onnx, vnnlib)
}

/// Complete qualified route kept separate so tests can exercise every proof
/// seam while the public compile-time publication gate remains dark.
fn try_tll_unsat_authenticated(onnx: &Path, vnnlib: &Path) -> Option<VnncompResult> {
    // Cheap STRUCTURAL pre-gate. The property is a small text file; the model
    // may be hundreds of MB, so the spec shape is filtered first and the ONNX
    // is only touched once this looks like the TLL family.
    //
    // This replaces a `file_name().contains("tll")` pre-gate. That gate was an
    // identity check on public benchmark filenames: a TLL network delivered
    // under any other name silently lost the fast path, and a non-TLL network
    // named "tll_*" paid a pointless load. The 2-input/1-output/non-disjunctive
    // shape below is what the decoder actually requires, and it is selective
    // enough to keep the load off unrelated families. The raw model-identity
    // proof below remains the real admission gate and fails closed.
    let authenticated_property = property_identity::authenticate_raw_tll_property(vnnlib)?;
    let spec = authenticated_property.spec();
    if spec.is_disjunction || spec.num_outputs != 1 || spec.num_inputs != 2 {
        return None;
    }
    let (dir, thresh) = authenticated_property.threshold();
    let [box0, box1] = authenticated_property.input_bounds();
    if !(box0.0.is_finite() && box0.1.is_finite() && box1.0.is_finite() && box1.1.is_finite())
        || box0.0 > box0.1
        || box1.0 > box1.1
    {
        return None;
    }
    // Every authored affine coefficient is f32. Keeping endpoints inside the
    // finite f32 range makes edge differences, coefficient products, magnitude
    // sums, and their outward slack finite in f64. Extreme finite f64 boxes
    // could otherwise create NaNs that `min`/`max` reductions silently ignore.
    let finite_input_limit = f32_to_f64_exact(f32::MAX);
    if [box0.0, box0.1, box1.0, box1.1]
        .iter()
        .any(|bound| bound.abs() > finite_input_limit)
    {
        return None;
    }

    // Authenticate the authored protobuf itself. This proves the sole FLOAT
    // input/output, the complete live chain, every edge/op/attribute, exact
    // affine and selector tensors, and every min/max gadget. It neither uses
    // NY's lowering nor samples as proof authority.
    let authenticated_model = model_identity::authenticate_raw_tll_model(onnx)?;
    let tll = authenticated_model.structure();

    // NY and ORT forwards below are diagnostics only. A mismatch can decline
    // the lane, but a match contributes no identity authority.
    let model = load_onnx_with_config(onnx, &OnnxLoadConfig::default()).ok()?;
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .ok()?;

    // Concrete forward (trusted sound f64 cell evaluator) for the self-check.
    let samples = sample_points(box0, box1);
    let forward = concrete_forward(&graph, &samples)?;

    // The raw proof fixes the order as max-of-min. Samples may veto that proof
    // path diagnostically, but must never select or alter its semantics.
    if !self_check(tll, &samples, &forward).0 {
        return None;
    }

    // Independent diagnostic oracle (#tll-ort-oracle): cross-check NY's sample
    // forward through ONNX Runtime. Neither sample engine contributes to the
    // raw algebraic identity proof; either may only veto the structural lane.
    // ORT unavailable or any disagreement beyond tolerance fails closed.
    if !ort_cross_check(onnx, &samples, &forward) {
        return None;
    }

    let sampled_min = forward.iter().copied().fold(f64::INFINITY, f64::min);
    let sampled_max = forward.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Escalating grid: return as soon as the sound bound clears the threshold.
    let enclose_tol = 1e-3 * (1.0 + sampled_max.abs().max(sampled_min.abs()));
    for &nx in &[32usize, 96, 288, 640] {
        let (lb, ub) = tll.box_bounds(box0, box1, nx)?;
        if !(lb.is_finite() && ub.is_finite()) || lb > ub {
            return None;
        }
        // Enclosure gate: a sound bound MUST enclose every sampled forward value
        // (lb <= sampled_min, ub >= sampled_max). A violation means the decode or
        // bound is inconsistent with the real network - fail closed.
        if lb > sampled_min + enclose_tol || ub < sampled_max - enclose_tol {
            return None;
        }
        let proven = match dir {
            ThreshDir::Le => lb > thresh, // min Y_0 >= lb > c  =>  no x with Y_0 <= c
            ThreshDir::Ge => ub < thresh, // max Y_0 <= ub < c  =>  no x with Y_0 >= c
        };
        if proven {
            // Bind publication to the exact bytes whose algebra was proved.
            if !authenticated_model.source_still_matches(onnx)
                || !authenticated_property.source_still_matches(vnnlib)
            {
                return None;
            }
            eprintln!(
                "TLL structure bound: certified UNSAT via {} lattice bound \
                 (dir={dir:?}, thresh={thresh:.6}, lb={lb:.6}, ub={ub:.6}, \
                 grid={nx}x{nx}, sampled_min={sampled_min:.6})",
                if tll.max_of_min {
                    "max-of-min"
                } else {
                    "min-of-max"
                }
            );
            return Some(VnncompResult::Unsat);
        }
    }
    None
}

impl TllStructure {
    /// Reconstructed forward value at a single point (exact f64, no rounding
    /// slack - used only for the self-check comparison).
    fn eval(&self, x0: f64, x1: f64) -> f64 {
        let l: Vec<f64> = (0..self.a.len())
            .map(|i| self.a[i][0] * x0 + self.a[i][1] * x1 + self.b[i])
            .collect();
        if self.max_of_min {
            self.groups
                .iter()
                .map(|g| g.iter().map(|&i| l[i]).fold(f64::INFINITY, f64::min))
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            self.groups
                .iter()
                .map(|g| g.iter().map(|&i| l[i]).fold(f64::NEG_INFINITY, f64::max))
                .fold(f64::INFINITY, f64::min)
        }
    }

    /// SOUND `(lower, upper)` bound of `y` over the box, via an `nx x nx` grid
    /// with OUTWARD-rounded per-cell affine corner minima/maxima.
    ///
    /// Separable pre-computation: the box-min of `a_i . x` over a cell splits
    /// per coordinate, so we tabulate per-axis contributions once (`N x nx`) and
    /// only add per cell.
    fn box_bounds(&self, box0: (f64, f64), box1: (f64, f64), nx: usize) -> Option<(f64, f64)> {
        let n = self.a.len();
        let memberships = self
            .groups
            .iter()
            .try_fold(0usize, |sum, group| sum.checked_add(group.len()))?;
        let cells = nx.checked_mul(nx)?;
        if memberships.checked_mul(cells)? > MAX_GRID_LATTICE_WORK {
            return None;
        }
        let e0 = checked_grid_edges(box0.0, box0.1, nx)?;
        let e1 = checked_grid_edges(box1.0, box1.1, nx)?;

        // Per-axis min/max contribution of a_i * x over each cell slice.
        // xlo[i*nx+ix] = min over [e0[ix],e0[ix+1]] of a_i0 * x  (rounded DOWN)
        // xhi[..]      = max                                     (rounded UP)
        let mut xlo = vec![0.0f64; n * nx];
        let mut xhi = vec![0.0f64; n * nx];
        let mut ylo = vec![0.0f64; n * nx];
        let mut yhi = vec![0.0f64; n * nx];
        for i in 0..n {
            let (a0, a1) = (self.a[i][0], self.a[i][1]);
            for ix in 0..nx {
                let (lo_edge, hi_edge) = (e0[ix], e0[ix + 1]);
                // a0>=0: increasing => min at lo_edge, max at hi_edge.
                // a0<0 : decreasing => min at hi_edge, max at lo_edge.
                let (cmin, cmax) = if a0 >= 0.0 {
                    (a0 * lo_edge, a0 * hi_edge)
                } else {
                    (a0 * hi_edge, a0 * lo_edge)
                };
                xlo[i * nx + ix] = round_down(cmin);
                xhi[i * nx + ix] = round_up(cmax);
            }
            for iy in 0..nx {
                let (lo_edge, hi_edge) = (e1[iy], e1[iy + 1]);
                let (cmin, cmax) = if a1 >= 0.0 {
                    (a1 * lo_edge, a1 * hi_edge)
                } else {
                    (a1 * hi_edge, a1 * lo_edge)
                };
                ylo[i * nx + iy] = round_down(cmin);
                yhi[i * nx + iy] = round_up(cmax);
            }
        }

        let mut global_lb = f64::INFINITY;
        let mut global_ub = f64::NEG_INFINITY;
        let mut cmin_i = vec![0.0f64; n];
        let mut cmax_i = vec![0.0f64; n];
        for ix in 0..nx {
            for iy in 0..nx {
                for i in 0..n {
                    // cell-min of L_i (rounded DOWN); cell-max (rounded UP).
                    // #tll-round-slack-harden: the 3-term SUM can cancel
                    // (|smin| << addend magnitudes), so the outward slack must
                    // scale with the addends' magnitude sum, not |smin|.
                    let xl = xlo[i * nx + ix];
                    let xh = xhi[i * nx + ix];
                    let yl = ylo[i * nx + iy];
                    let yh = yhi[i * nx + iy];
                    let smin = self.b[i] + xl + yl;
                    let smax = self.b[i] + xh + yh;
                    let mag = self.b[i].abs() + xl.abs().max(xh.abs()) + yl.abs().max(yh.abs());
                    cmin_i[i] = round_down_mag(smin, mag);
                    cmax_i[i] = round_up_mag(smax, mag);
                }
                // Per-cell bound. For max-of-min:
                //   LB_cell = max_j min_{i in S_j} cmin_i   (<= min_x y over cell)
                //   UB_cell = max_j min_{i in S_j} cmax_i   (>= max_x y over cell)
                // For the min-of-max realization the outer/inner reductions swap.
                // Reductions over already-outward values are exact (comparisons).
                let (lb_cell, ub_cell) = if self.max_of_min {
                    let mut lb = f64::NEG_INFINITY;
                    let mut ub = f64::NEG_INFINITY;
                    for g in &self.groups {
                        let mut gmin_lo = f64::INFINITY;
                        let mut gmin_hi = f64::INFINITY;
                        for &i in g {
                            gmin_lo = gmin_lo.min(cmin_i[i]);
                            gmin_hi = gmin_hi.min(cmax_i[i]);
                        }
                        lb = lb.max(gmin_lo);
                        ub = ub.max(gmin_hi);
                    }
                    (lb, ub)
                } else {
                    let mut lb = f64::INFINITY;
                    let mut ub = f64::INFINITY;
                    for g in &self.groups {
                        let mut gmax_lo = f64::NEG_INFINITY;
                        let mut gmax_hi = f64::NEG_INFINITY;
                        for &i in g {
                            gmax_lo = gmax_lo.max(cmin_i[i]);
                            gmax_hi = gmax_hi.max(cmax_i[i]);
                        }
                        lb = lb.min(gmax_lo);
                        ub = ub.min(gmax_hi);
                    }
                    (lb, ub)
                };
                global_lb = global_lb.min(lb_cell);
                global_ub = global_ub.max(ub_cell);
            }
        }
        Some((global_lb, global_ub))
    }
}

/// Construct a complete cell partition of `[lo, hi]` and authenticate the
/// floating-point realization before it can drive a proof. Exact endpoints
/// plus finite, contained, nondecreasing interior edges ensure the adjacent
/// closed cells cover the complete outward input interval. Duplicate edges are
/// harmless zero-width cells; a gap, reversal, overflow, or NaN fails closed.
fn checked_grid_edges(lo: f64, hi: f64, nx: usize) -> Option<Vec<f64>> {
    if nx == 0 || !lo.is_finite() || !hi.is_finite() || lo > hi {
        return None;
    }
    let span = hi - lo;
    let denominator = nx as f64;
    if !span.is_finite() || span < 0.0 || !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }

    let capacity = nx.checked_add(1)?;
    let mut edges = Vec::new();
    edges.try_reserve_exact(capacity).ok()?;
    edges.push(lo);
    for k in 1..nx {
        let edge = lo + span * (k as f64) / denominator;
        let previous = *edges.last()?;
        if !edge.is_finite() || edge < lo || edge > hi || edge < previous {
            return None;
        }
        edges.push(edge);
    }
    if hi < *edges.last()? {
        return None;
    }
    edges.push(hi);
    if edges.len() != capacity
        || edges.first().copied()?.to_bits() != lo.to_bits()
        || edges.last().copied()?.to_bits() != hi.to_bits()
    {
        return None;
    }
    Some(edges)
}

/// Sample points for the decode self-check: a dense interior grid spanning the
/// box (includes the corners).
fn sample_points(box0: (f64, f64), box1: (f64, f64)) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    let g = 9usize;
    for i in 0..g {
        for j in 0..g {
            let x0 = box0.0 + (box0.1 - box0.0) * (i as f64) / ((g - 1) as f64);
            let x1 = box1.0 + (box1.1 - box1.0) * (j as f64) / ((g - 1) as f64);
            pts.push([x0, x1]);
        }
    }
    pts
}

/// NY's CONCRETE forward at each sample point, via the TRUSTED sound f64 cell
/// interval evaluator (`propagate_ibp_f64_cell`) - the same evaluator the SAT
/// oracle uses. At a point it returns a ~1e-14-wide enclosure of the true f64
/// output; we take the midpoint. (The generic f32 `propagate_ibp` is a LOOSE
/// IBP relaxation even at a point for these wide min/max banks - unusable as a
/// ground truth here.)
fn concrete_forward(graph: &ny_propagate::GraphNetwork, pts: &[[f64; 2]]) -> Option<Vec<f64>> {
    if !graph.supports_ibp_f64_cell() {
        return None;
    }
    let mut out = Vec::with_capacity(pts.len());
    for p in pts {
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![p[0], p[1]]).ok()?;
        let r = graph.propagate_ibp_f64_cell(&Interval64::point(arr)).ok()?;
        if r.lower.len() != 1 || r.upper.len() != 1 {
            return None;
        }
        let lo = *r.lower.iter().next()?;
        let hi = *r.upper.iter().next()?;
        if !(lo.is_finite() && hi.is_finite()) || lo > hi {
            return None;
        }
        out.push(f64::midpoint(lo, hi));
    }
    Some(out)
}

/// Require the reconstructed function to match NY's forward at every sample.
/// Returns `(passed, max_abs_err)`.
fn self_check(tll: &TllStructure, pts: &[[f64; 2]], forward: &[f64]) -> (bool, f64) {
    if pts.len() != forward.len() || forward.iter().any(|value| !value.is_finite()) {
        return (false, f64::INFINITY);
    }
    let scale = forward.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1.0);
    let tol = 1e-3 * scale;
    let mut max_err = 0.0f64;
    for (p, &f) in pts.iter().zip(forward) {
        max_err = max_err.max((tll.eval(p[0], p[1]) - f).abs());
    }
    (max_err <= tol, max_err)
}

/// Cross-check ny's concrete forward against ONNX Runtime at every sample
/// point (#tll-ort-oracle). The ORT session is committed from the ORIGINAL
/// model bytes via `ny_onnx::diff::OrtForward` (input shape read straight
/// from the protobuf), sharing NOTHING with ny's own graph parse - a
/// systematic misparse cannot fool both sides. Returns `false` (the caller
/// fails closed) when ORT is unavailable, any run fails, the model does not
/// produce exactly one output value, or any output disagrees with `forward`
/// beyond the same relative tolerance the self-check uses.
fn ort_cross_check(onnx: &Path, pts: &[[f64; 2]], forward: &[f64]) -> bool {
    if pts.len() != forward.len() {
        return false;
    }
    let mut ort = match ny_onnx::diff::OrtForward::from_path(onnx, 2) {
        Ok(fwd) => fwd,
        Err(e) => {
            eprintln!(
                "TLL structure bound: ORT oracle unavailable ({e}); \
                 falling back to the generic path"
            );
            return false;
        }
    };
    let scale = forward.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1.0);
    let tol = 1e-3 * scale;
    for (p, &f) in pts.iter().zip(forward) {
        let out = match ort.run(&[p[0] as f32, p[1] as f32]) {
            Ok(out) => out,
            Err(e) => {
                eprintln!(
                    "TLL structure bound: ORT oracle forward failed ({e}); \
                     falling back to the generic path"
                );
                return false;
            }
        };
        if out.len() != 1 {
            return false;
        }
        let o = f32_to_f64_exact(out[0]);
        if !o.is_finite() || (o - f).abs() > tol {
            eprintln!(
                "TLL structure bound: ORT oracle disagrees with ny forward at \
                 ({}, {}): ort={o}, ny={f} (tol={tol:.3e}); falling back to \
                 the generic path",
                p[0], p[1]
            );
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small TLL `y = max_j min_{i in S_j} L_i` and brute-force its
    /// true min/max over the box on a dense grid.
    fn brute(tll: &TllStructure, b0: (f64, f64), b1: (f64, f64), g: usize) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..=g {
            for j in 0..=g {
                let x0 = b0.0 + (b0.1 - b0.0) * (i as f64) / (g as f64);
                let x1 = b1.0 + (b1.1 - b1.0) * (j as f64) / (g as f64);
                let v = tll.eval(x0, x1);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo, hi)
    }

    fn sample() -> TllStructure {
        // Five affine functions of a 2-D input, two overlapping selector groups.
        TllStructure {
            a: vec![
                [1.0, 0.5],
                [-0.7, 1.2],
                [0.3, -0.9],
                [-1.1, -0.4],
                [0.6, 0.6],
            ],
            b: vec![0.2, -0.5, 0.8, 0.1, -0.3],
            groups: vec![vec![0, 1, 2], vec![2, 3, 4], vec![0, 4]],
            max_of_min: true,
        }
    }

    #[test]
    fn verdict_route_requires_source_authenticated_property() {
        // Raw model and exact-decimal property identity are independently
        // established; publication stays dark until release qualification is
        // explicitly completed.
        const { assert!(!TLL_PROPERTY_SOURCE_AUTHENTICATED) };
    }

    #[test]
    fn checked_grid_edges_cover_box_or_fail_closed() {
        let edges =
            checked_grid_edges(-0.1_f64, 0.3_f64.next_up(), 640).expect("finite ordered grid");
        assert_eq!(edges.len(), 641);
        assert_eq!(edges.first().unwrap().to_bits(), (-0.1_f64).to_bits());
        assert_eq!(edges.last().unwrap().to_bits(), 0.3_f64.next_up().to_bits());
        assert!(edges.iter().all(|edge| edge.is_finite()));
        assert!(edges.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(edges
            .iter()
            .all(|&edge| { edge >= -0.1_f64 && edge <= 0.3_f64.next_up() }));

        assert!(checked_grid_edges(0.0, 1.0, 0).is_none());
        assert!(checked_grid_edges(1.0, 0.0, 32).is_none());
        assert!(checked_grid_edges(f64::NAN, 1.0, 32).is_none());
        assert!(checked_grid_edges(-f64::MAX, f64::MAX, 32).is_none());
    }

    #[test]
    fn adversarial_grid_work_fails_closed_before_cell_reduction() {
        let mut tll = sample();
        tll.groups = vec![vec![0; 2_500]];
        assert!(tll.box_bounds((-2.0, 2.0), (-2.0, 2.0), 640).is_none());
    }

    /// Exercise the complete route, including raw source seals, NY/ORT vetoes,
    /// checked grid, and directional threshold, without enabling publication.
    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn requested_real_end_to_end_pair_is_certified() {
        let model = std::env::var("NY_TLL_END_TO_END_MODEL").expect(
            "external-vnncomp TLL end-to-end conformance requires \
             NY_TLL_END_TO_END_MODEL=/path/to/model.onnx",
        );
        let property = std::env::var("NY_TLL_END_TO_END_PROPERTY").expect(
            "external-vnncomp TLL end-to-end conformance requires \
             NY_TLL_END_TO_END_PROPERTY=/path/to/property.vnnlib",
        );
        assert_eq!(
            try_tll_unsat_authenticated(Path::new(&model), Path::new(&property)),
            Some(VnncompResult::Unsat)
        );
    }

    #[test]
    fn bound_is_sound_enclosure_max_of_min() {
        let tll = sample();
        let b0 = (-2.0, 2.0);
        let b1 = (-2.0, 2.0);
        let (bmin, bmax) = brute(&tll, b0, b1, 2000);
        for &nx in &[8usize, 32, 128] {
            let (lb, ub) = tll.box_bounds(b0, b1, nx).expect("valid grid");
            // SOUND: lb <= true min, ub >= true max (small slack for the coarse
            // brute grid missing the exact extremum on the low-nx bounds).
            assert!(lb <= bmin + 1e-9, "nx={nx}: lb={lb} > brute_min={bmin}");
            assert!(ub >= bmax - 1e-9, "nx={nx}: ub={ub} < brute_max={bmax}");
        }
        // Convergence: the finest grid brackets the true min/max tightly.
        let (lb, ub) = tll.box_bounds(b0, b1, 512).expect("valid grid");
        assert!(bmin - lb < 0.05, "lb not tight: {lb} vs {bmin}");
        assert!(ub - bmax < 0.05, "ub not tight: {ub} vs {bmax}");
    }

    #[test]
    fn bound_is_sound_enclosure_min_of_max() {
        let mut tll = sample();
        tll.max_of_min = false;
        let b0 = (-1.5, 2.5);
        let b1 = (-2.0, 1.0);
        let (bmin, bmax) = brute(&tll, b0, b1, 2000);
        for &nx in &[8usize, 32, 128] {
            let (lb, ub) = tll.box_bounds(b0, b1, nx).expect("valid grid");
            assert!(lb <= bmin + 1e-9, "nx={nx}: lb={lb} > brute_min={bmin}");
            assert!(ub >= bmax - 1e-9, "nx={nx}: ub={ub} < brute_max={bmax}");
        }
    }

    #[test]
    fn outward_rounding_is_directed() {
        // round_down strictly below, round_up strictly above (for nonzero v).
        for &v in &[1.0, -3.75, 1e6, -1e-3] {
            assert!(round_down(v) < v);
            assert!(round_up(v) > v);
        }
    }

    /// ORT oracle unavailability (missing/unreadable model) must fail closed:
    /// the fast path falls back to the generic lane, never a verdict.
    #[test]
    fn ort_oracle_missing_model_fails_closed() {
        let pts = [[0.0f64, 0.0], [1.0, 1.0]];
        let fwd = [0.0f64, 2.0];
        assert!(!ort_cross_check(
            Path::new("/nonexistent/no_such_model.onnx"),
            &pts,
            &fwd
        ));
    }

    /// Write a tiny 2-in/1-out ONNX model (`Y = MatMul(X, [[1],[1]]) = x0+x1`)
    /// and return its path (kept alive by the returned tempdir).
    fn write_sum_model() -> (tempfile::TempDir, std::path::PathBuf) {
        use ny_onnx::onnx_proto;
        use prost::Message;

        let dim = |v: i64| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                v,
            )),
        };
        let vinfo = |name: &str, dims: &[i64]| onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: 1, // FLOAT
                    shape: Some(onnx_proto::TensorShapeProto {
                        dim: dims.iter().map(|&v| dim(v)).collect(),
                    }),
                }),
            }),
        };
        let model = onnx_proto::ModelProto {
            ir_version: 8,
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(onnx_proto::GraphProto {
                node: vec![onnx_proto::NodeProto {
                    input: vec!["X".to_string(), "W".to_string()],
                    output: vec!["Y".to_string()],
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    attribute: Vec::new(),
                }],
                name: "sum2".to_string(),
                initializer: vec![onnx_proto::TensorProto {
                    dims: vec![2, 1],
                    data_type: 1, // FLOAT
                    name: "W".to_string(),
                    float_data: vec![1.0, 1.0],
                    ..Default::default()
                }],
                input: vec![vinfo("X", &[1, 2])],
                output: vec![vinfo("Y", &[1, 1])],
                ..Default::default()
            }),
            ..Default::default()
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sum2.onnx");
        std::fs::write(&path, model.encode_to_vec()).expect("write model");
        (dir, path)
    }

    /// The ORT oracle confirms a forward that matches the real model and
    /// rejects one perturbed beyond the tolerance (mismatch fails closed).
    #[test]
    fn ort_oracle_confirms_match_and_rejects_mismatch() {
        let (_dir, path) = write_sum_model();
        let pts = [[0.25f64, 0.5], [1.0, -2.0], [-0.75, 0.125]];
        let fwd: Vec<f64> = pts.iter().map(|p| p[0] + p[1]).collect();
        assert!(
            ort_cross_check(&path, &pts, &fwd),
            "matching forward must pass the ORT oracle"
        );

        let mut wrong = fwd.clone();
        wrong[1] += 1.0; // way beyond the 1e-3-relative gate
        assert!(
            !ort_cross_check(&path, &pts, &wrong),
            "a forward disagreeing with ORT must fail closed"
        );

        // Length mismatch is malformed input - fail closed.
        assert!(!ort_cross_check(&path, &pts, &fwd[..2]));
    }

    #[test]
    fn mag_slack_covers_catastrophic_cancellation() {
        // #tll-round-slack-harden: with ~1e7-magnitude addends cancelling to
        // ~0, the f64 sum error scales with the ADDEND magnitudes; the old
        // |v|-relative slack under-covered it. Assert the mag-scaled slack
        // encloses the EXACT (rational-free: computed via two-f64 splitting)
        // sum at 4000 fuzz points across magnitude regimes up to 1e12.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for scale_exp in [0i32, 3, 7, 9, 12] {
            let scale = 10f64.powi(scale_exp);
            for _ in 0..800 {
                let x = rnd() * scale;
                let y = rnd() * scale;
                // b engineered to cancel the sum to ~0 (the hostile regime).
                let b = -(x + y) + rnd() * 1e-3;
                let v = b + x + y; // the f64-rounded sum under test
                let mag = b.abs() + x.abs() + y.abs();
                // Exact sum via Kahan-style two-sum error extraction.
                let s1 = b + x;
                let e1 = (b - (s1 - x)) + (x - (s1 - (s1 - x)));
                let s2 = s1 + y;
                let e2 = (s1 - (s2 - y)) + (y - (s2 - (s2 - y)));
                let exact = s2 + (e1 + e2); // exact to well below the slack
                assert!(
                    round_down_mag(v, mag) <= exact,
                    "round_down_mag not outward: v={v} mag={mag} exact={exact}"
                );
                assert!(
                    round_up_mag(v, mag) >= exact,
                    "round_up_mag not outward: v={v} mag={mag} exact={exact}"
                );
            }
        }
    }
}
