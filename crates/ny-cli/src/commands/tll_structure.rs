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
//! - The decode (affine `L_i` + selector sets + max-of-min vs min-of-max order) is
//!   SELF-CHECKED at runtime against NY's own concrete forward at many sampled
//!   points; a mismatch fails closed (returns `None`, generic path runs).
//! - An enclosure gate additionally requires `lb <= sampled_min` and
//!   `ub >= sampled_max`; a violated gate fails closed.
//! - We only ever return `Unsat`, and only when the sound bound strictly clears
//!   the threshold - so a too-LOOSE bound merely forgoes the win (falls through),
//!   never a false verdict.
//!
//! Default ON; disable with `NY_TLL_STRUCTURE_BOUND=0`. Only engages when the
//! filename, node-name pattern, shape signature, one-hot selection, and forward
//! self-check ALL match a genuine TLL net - a no-op for every other benchmark.

use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::OutputConstraint;
use ny_onnx::{load_onnx_with_config, OnnxLoadConfig};
use ny_propagate::Interval64;

use super::vnncomp::VnncompResult;
use ny_onnx::{CompoundNodePolicy, GraphNetworkOptions};

/// Direction of the single output threshold constraint (on `Y_0`).
#[derive(Clone, Copy, Debug)]
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
    /// `true`: `y = max_j min_{i} L_i` (the observed tllverifybench realization);
    /// `false`: `y = min_j max_{i} L_i` (checked as a fallback).
    max_of_min: bool,
}

/// Relative outward-rounding slack (>> the ~4u accumulated f64 error for a
/// 2-term affine; u = 2^-53, `EPSILON` = 2^-52).
const OUTWARD: f64 = 32.0 * f64::EPSILON;

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
    // Disable knob (batteries-included default ON).
    if std::env::var("NY_TLL_STRUCTURE_BOUND").ok().as_deref() == Some("0") {
        return None;
    }

    // Cheap pre-gate: only touch the (large) model for TLL-named instances.
    let fname = onnx.file_name()?.to_string_lossy().to_ascii_lowercase();
    if !fname.contains("tll") {
        return None;
    }

    // Parse the property: require a single atomic threshold on Y_0 over a 2-D box.
    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib).ok()?;
    if spec.is_disjunction || spec.num_outputs != 1 || spec.num_inputs != 2 {
        return None;
    }
    let (dir, thresh) = single_output_threshold(&spec)?;
    if spec.input_bounds.len() != 2 {
        return None;
    }
    let box0 = spec.input_bounds[0];
    let box1 = spec.input_bounds[1];
    if !(box0.0.is_finite() && box0.1.is_finite() && box1.0.is_finite() && box1.1.is_finite())
        || box0.0 > box0.1
        || box1.0 > box1.1
    {
        return None;
    }

    // Load ONCE: raw weights (pre-fusion names/initializers intact) for the
    // decode, then build the concrete graph from the SAME model (no re-parse).
    let model = load_onnx_with_config(onnx, &OnnxLoadConfig::default()).ok()?;
    let mut tll = decode_tll(&model.weights)?;
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .ok()?;

    // Concrete forward (trusted sound f64 cell evaluator) for the self-check.
    let samples = sample_points(box0, box1);
    let forward = concrete_forward(&graph, &samples)?;

    // Self-check the decode; auto-detect max-of-min vs min-of-max order.
    // A mismatch on BOTH orders means our reconstruction is not this network -
    // fail closed (fall through to the generic verifier).
    if !self_check(&tll, &samples, &forward).0 {
        tll.max_of_min = !tll.max_of_min;
        if !self_check(&tll, &samples, &forward).0 {
            return None;
        }
    }

    // Independent parse oracle (#tll-ort-oracle): `forward` above derives from
    // the SAME ny ONNX parse as the decode (load_onnx_with_config +
    // to_graph_network), so a systematic misparse could fool the self-check
    // AND the bound together. Cross-check the same sample grid through ONNX
    // Runtime, committed from the ORIGINAL model bytes with the input shape
    // read straight from the protobuf - shares nothing with ny's parse. ORT
    // unavailable or any disagreement beyond the self-check tolerance fails
    // closed: the generic (non-fast-path) lane runs, and no verdict ever
    // rests on the ny parse alone.
    if !ort_cross_check(onnx, &samples, &forward) {
        return None;
    }

    let sampled_min = forward.iter().copied().fold(f64::INFINITY, f64::min);
    let sampled_max = forward.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Escalating grid: return as soon as the sound bound clears the threshold.
    let enclose_tol = 1e-3 * (1.0 + sampled_max.abs().max(sampled_min.abs()));
    for &nx in &[32usize, 96, 288, 640] {
        let (lb, ub) = tll.box_bounds(box0, box1, nx);
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

/// Extract the single atomic output-threshold constraint on `Y_0`, if that is
/// exactly the property.
fn single_output_threshold(spec: &ny_onnx::vnnlib::VnnLibSpec) -> Option<(ThreshDir, f64)> {
    // Prefer the clause representation; fall back to the flat list.
    let atoms: &[OutputConstraint] = if !spec.output_constraint_clauses.is_empty() {
        if spec.output_constraint_clauses.len() != 1 {
            return None;
        }
        &spec.output_constraint_clauses[0]
    } else {
        &spec.output_constraints
    };
    if atoms.len() != 1 {
        return None;
    }
    match atoms[0] {
        OutputConstraint::LessEqConst(0, c) | OutputConstraint::LessThanConst(0, c) => {
            Some((ThreshDir::Le, c))
        }
        OutputConstraint::GreaterEqConst(0, c) | OutputConstraint::GreaterThanConst(0, c) => {
            Some((ThreshDir::Ge, c))
        }
        _ => None,
    }
}

/// Decode `L_i` (linearLayer) and selector sets `S_j` (selectionLayer one-hot)
/// from the raw weight store. Returns `None` unless the full TLL signature -
/// linearLayer + selectionLayer + minBank* + maxBank* with a clean one-hot
/// selection and a 2-D input - is present.
fn decode_tll(weights: &ny_onnx::WeightStore) -> Option<TllStructure> {
    let find = |needle: &str| -> Option<&ArrayD<f32>> {
        weights
            .iter()
            .find(|(k, _)| k.contains(needle))
            .map(|(_, v)| v)
    };
    // Bank signature: both min and max banks must be present.
    if !weights.keys().any(|k| k.contains("minBank"))
        || !weights.keys().any(|k| k.contains("maxBank"))
    {
        return None;
    }

    let lin_w = find("linearLayer/MatMul")?; // shape [in=2, N]
    let lin_b = find("linearLayer/BiasAdd")?; // shape [N]
    let sel_w = find("selectionLayer/MatMul")?; // shape [N, N*SL]

    if lin_w.ndim() != 2 || sel_w.ndim() != 2 || lin_b.ndim() != 1 {
        return None;
    }
    let in_dim = lin_w.shape()[0];
    let n = lin_w.shape()[1];
    if in_dim != 2 || n == 0 || lin_b.shape()[0] != n {
        return None;
    }
    if sel_w.shape()[0] != n {
        return None;
    }
    let ncol = sel_w.shape()[1];
    if ncol == 0 || ncol % n != 0 {
        return None;
    }
    let slots = ncol / n; // slots per group

    // Affine functions L_i(x) = a_i . x + b_i.
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push([lin_w[[0, i]] as f64, lin_w[[1, i]] as f64]);
        b.push(lin_b[[i]] as f64);
    }

    // Decode groups from the one-hot selection matrix. Column c belongs to
    // group `c / slots`; its single ~1.0 row is the selected L-index. A column
    // that is not clean one-hot aborts the decode (fail closed).
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n];
    for c in 0..ncol {
        let mut sel_row: Option<usize> = None;
        for r in 0..n {
            let v = sel_w[[r, c]];
            if v.abs() > 1e-4 {
                if (v - 1.0).abs() > 1e-3 || sel_row.is_some() {
                    return None; // not a clean one-hot column
                }
                sel_row = Some(r);
            }
        }
        let r = sel_row?;
        let j = c / slots;
        let g = &mut groups[j];
        if !g.contains(&r) {
            g.push(r);
        }
    }
    if groups.iter().any(|g| g.is_empty()) {
        return None;
    }

    Some(TllStructure {
        a,
        b,
        groups,
        max_of_min: true,
    })
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
    fn box_bounds(&self, box0: (f64, f64), box1: (f64, f64), nx: usize) -> (f64, f64) {
        let n = self.a.len();
        let edges = |lo: f64, hi: f64| -> Vec<f64> {
            let mut e = Vec::with_capacity(nx + 1);
            for k in 0..=nx {
                if k == 0 {
                    e.push(lo);
                } else if k == nx {
                    e.push(hi);
                } else {
                    e.push(lo + (hi - lo) * (k as f64) / (nx as f64));
                }
            }
            e
        };
        let e0 = edges(box0.0, box0.1);
        let e1 = edges(box1.0, box1.1);

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
        (global_lb, global_ub)
    }
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
        out.push(f64::midpoint(lo, hi));
    }
    Some(out)
}

/// Require the reconstructed function to match NY's forward at every sample.
/// Returns `(passed, max_abs_err)`.
fn self_check(tll: &TllStructure, pts: &[[f64; 2]], forward: &[f64]) -> (bool, f64) {
    if pts.len() != forward.len() {
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
        let o = f64::from(out[0]);
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
    fn bound_is_sound_enclosure_max_of_min() {
        let tll = sample();
        let b0 = (-2.0, 2.0);
        let b1 = (-2.0, 2.0);
        let (bmin, bmax) = brute(&tll, b0, b1, 2000);
        for &nx in &[8usize, 32, 128] {
            let (lb, ub) = tll.box_bounds(b0, b1, nx);
            // SOUND: lb <= true min, ub >= true max (small slack for the coarse
            // brute grid missing the exact extremum on the low-nx bounds).
            assert!(lb <= bmin + 1e-9, "nx={nx}: lb={lb} > brute_min={bmin}");
            assert!(ub >= bmax - 1e-9, "nx={nx}: ub={ub} < brute_max={bmax}");
        }
        // Convergence: the finest grid brackets the true min/max tightly.
        let (lb, ub) = tll.box_bounds(b0, b1, 512);
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
            let (lb, ub) = tll.box_bounds(b0, b1, nx);
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
