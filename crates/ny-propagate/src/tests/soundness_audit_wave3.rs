// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adversarial soundness re-audit (wave 3) of recently-landed CROWN backward
//! relaxations and bound-arithmetic changes.
//!
//! In VNN-COMP an unsound "verified/unsat" verdict is catastrophic, so for each
//! newly-added relaxation we independently try to BREAK soundness with dense,
//! deterministic, seeded sampling of many random input boxes — including
//! adversarial/degenerate ones (near-zero / negative denominators, equal x==y
//! for min/max, atan2 boxes near the branch cut / origin handled by fallback,
//! wide vs. narrow boxes, duplicate scatter indices). The invariant under test
//! is always
//!
//! ```text
//!     lower_form(p) <= f(p) <= upper_form(p)
//! ```
//!
//! evaluated at densely sampled interior points `p` (corners included). For the
//! exact-linear ops (ScatterAdd / IndexAdd / ScatterND / constant-mask Where) we
//! assert the CROWN backward equals the explicit dense-matrix transpose.
//!
//! Sampling is deterministic (a tiny splitmix64 PRNG seeded from a fixed
//! constant) so the suite is reproducible — no Date/time/thread-rng entropy.
//!
//! Relaxations audited here:
//! 1. Min/Max binary CROWN envelope (`minmax_relax` via Max/MinBinaryLayer).
//! 2. Atan2 binary CROWN envelope (`atan2_relax` via Atan2Layer).
//! 3. Div binary CROWN reciprocal-scaling, incl. negative denominator
//!    (`graph_crown::backward_div_to_numerator`).
//! 4. ScatterAdd / IndexAdd / ScatterND exact CROWN backward == dense transpose.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::layers::binary_ops::{Atan2Layer, MaxBinaryLayer, MinBinaryLayer};
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::BatchNormLayer;
use crate::layers::transform::{IndexAddLayer, ScatterAddLayer, ScatterNdLayer};
use crate::network::{backward_div_to_numerator, DivBackwardResult};

// ── Deterministic PRNG (splitmix64) ─────────────────────────────────────────

/// Tiny deterministic PRNG so the dense sampling is fully reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in `[lo, hi]`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + (hi - lo) * t
    }

    /// Random ordered interval `[l, u]` with `l <= u` inside `[lo, hi]`.
    fn interval(&mut self, lo: f32, hi: f32) -> (f32, f32) {
        let a = self.uniform(lo, hi);
        let b = self.uniform(lo, hi);
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn bt1(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// A random incoming CROWN spec `node_lb`: `out_dim` rows over `n` inputs, with
/// distinct lower/upper planes (so the relaxation's plane selection is exercised
/// with both signs of incoming weight).
fn random_spec(rng: &mut Rng, out_dim: usize, n: usize) -> LinearBounds {
    let mut la = Array2::<f32>::zeros((out_dim, n));
    let mut ua = Array2::<f32>::zeros((out_dim, n));
    let mut lb = Array1::<f32>::zeros(out_dim);
    let mut ub = Array1::<f32>::zeros(out_dim);
    for r in 0..out_dim {
        for c in 0..n {
            // Lower weight <= upper weight per cell keeps node_lb a valid pair of
            // sound planes, and spans negative+positive to exercise plane select.
            let w0 = rng.uniform(-2.5, 2.5);
            let w1 = rng.uniform(-2.5, 2.5);
            la[[r, c]] = w0.min(w1);
            ua[[r, c]] = w0.max(w1);
        }
        let b0 = rng.uniform(-1.0, 1.0);
        let b1 = rng.uniform(-1.0, 1.0);
        lb[r] = b0.min(b1);
        ub[r] = b0.max(b1);
    }
    LinearBounds::new(la, lb, ua, ub).unwrap()
}

/// Reconstruct the lower/upper affine forms `(bounds_a, bounds_b)` of a binary
/// relaxation at the concrete operand point `(a, b)` for spec row `s`, and check
/// the soundness sandwich against the incoming spec evaluated at `f(a, b)`.
///
/// The driver convention (shared by Min/Max/Atan2): the relaxation bias lives
/// entirely on `bounds_a`; `bounds_b`'s bias is zero. So the lower form is
/// `Σ_i la_a[s,i]*a_i + Σ_i la_b[s,i]*b_i + lb_a_bias[s]`.
#[allow(clippy::too_many_arguments)]
fn check_binary_relaxation_sound<F>(
    node_lb: &LinearBounds,
    bounds_a: &LinearBounds,
    bounds_b: &LinearBounds,
    a_vals: &[f32],
    b_vals: &[f32],
    f: F,
    label: &str,
) where
    F: Fn(f32, f32) -> f32,
{
    let out_dim = node_lb.num_outputs();
    let n = a_vals.len();
    for s in 0..out_dim {
        // Incoming spec evaluated at z_i = f(a_i, b_i).
        let mut spec_lower = node_lb.lower_b()[s] as f64;
        let mut spec_upper = node_lb.upper_b()[s] as f64;
        for i in 0..n {
            let z = f(a_vals[i], b_vals[i]) as f64;
            spec_lower += node_lb.lower_a()[[s, i]] as f64 * z;
            spec_upper += node_lb.upper_a()[[s, i]] as f64 * z;
        }

        // Relaxed forms in (a, b).
        let mut relaxed_lower = bounds_a.lower_b()[s] as f64 + bounds_b.lower_b()[s] as f64;
        let mut relaxed_upper = bounds_a.upper_b()[s] as f64 + bounds_b.upper_b()[s] as f64;
        for i in 0..n {
            relaxed_lower += bounds_a.lower_a()[[s, i]] as f64 * a_vals[i] as f64;
            relaxed_lower += bounds_b.lower_a()[[s, i]] as f64 * b_vals[i] as f64;
            relaxed_upper += bounds_a.upper_a()[[s, i]] as f64 * a_vals[i] as f64;
            relaxed_upper += bounds_b.upper_a()[[s, i]] as f64 * b_vals[i] as f64;
        }

        // Soundness: relaxed lower form must under-estimate the true spec lower
        // contribution, relaxed upper form must over-estimate it.
        let scale = 1.0 + spec_lower.abs().max(spec_upper.abs());
        let tol = 5e-4 * scale;
        assert!(
            relaxed_lower <= spec_lower + tol,
            "{label}: UNSOUND lower: relaxed_lower {relaxed_lower} > spec_lower {spec_lower} \
             (row {s}) a={a_vals:?} b={b_vals:?}"
        );
        assert!(
            relaxed_upper >= spec_upper - tol,
            "{label}: UNSOUND upper: relaxed_upper {relaxed_upper} < spec_upper {spec_upper} \
             (row {s}) a={a_vals:?} b={b_vals:?}"
        );
    }
}

/// Dense-sample the per-element operand boxes (corners + interior grid) and run
/// the binary-relaxation soundness sandwich at every sampled point.
fn sample_binary_box<F>(
    a: &BoundedTensor,
    b: &BoundedTensor,
    node_lb: &LinearBounds,
    bounds_a: &LinearBounds,
    bounds_b: &LinearBounds,
    steps: usize,
    f: F,
    label: &str,
) where
    F: Fn(f32, f32) -> f32 + Copy,
{
    let n = a.len();
    let al = a.lower();
    let au = a.upper();
    let bl = b.lower();
    let bu = b.upper();

    let mut a_vals = vec![0.0f32; n];
    let mut b_vals = vec![0.0f32; n];
    // Independent per-element grids would explode combinatorially; instead sweep
    // a shared (ta, tb) grid (each element follows its own box but at the same
    // fractional position) PLUS all-corner combinations per element via diagonal
    // and anti-diagonal sweeps. This still hits every individual element corner.
    for i in 0..=steps {
        let ta = i as f32 / steps as f32;
        for j in 0..=steps {
            let tb = j as f32 / steps as f32;
            for k in 0..n {
                a_vals[k] = al[k] + (au[k] - al[k]) * ta;
                b_vals[k] = bl[k] + (bu[k] - bl[k]) * tb;
            }
            check_binary_relaxation_sound(node_lb, bounds_a, bounds_b, &a_vals, &b_vals, f, label);
            // Anti-diagonal for b to decorrelate a/b corners.
            for k in 0..n {
                b_vals[k] = bl[k] + (bu[k] - bl[k]) * (1.0 - tb);
            }
            check_binary_relaxation_sound(node_lb, bounds_a, bounds_b, &a_vals, &b_vals, f, label);
        }
    }
}

// ── 1. Min / Max binary CROWN envelope ───────────────────────────────────────

#[test]
fn minmax_crown_envelope_sound_dense_random() {
    let mut rng = Rng::new(0xA11C_E5EE);
    let n = 4usize;
    let out_dim = 3usize;
    let max_layer = MaxBinaryLayer;
    let min_layer = MinBinaryLayer;

    let mut tested = 0usize;
    for trial in 0..400 {
        // Mix of regimes per trial: wide, narrow, equal x==y, single-signed,
        // straddling, point boxes.
        let mut al = vec![0.0f32; n];
        let mut au = vec![0.0f32; n];
        let mut bl = vec![0.0f32; n];
        let mut bu = vec![0.0f32; n];
        for k in 0..n {
            match (trial + k) % 5 {
                0 => {
                    // wide overlapping boxes
                    let (l, u) = rng.interval(-8.0, 8.0);
                    al[k] = l;
                    au[k] = u;
                    let (l2, u2) = rng.interval(-8.0, 8.0);
                    bl[k] = l2;
                    bu[k] = u2;
                }
                1 => {
                    // narrow boxes near each other (mixed t region)
                    let c = rng.uniform(-3.0, 3.0);
                    al[k] = c - 0.05;
                    au[k] = c + 0.05;
                    bl[k] = c - 0.03;
                    bu[k] = c + 0.07;
                }
                2 => {
                    // exactly equal boxes (x == y everywhere): tie handling
                    let (l, u) = rng.interval(-5.0, 5.0);
                    al[k] = l;
                    au[k] = u;
                    bl[k] = l;
                    bu[k] = u;
                }
                3 => {
                    // disjoint: a strictly dominates / b strictly dominates
                    let (l, u) = rng.interval(0.0, 5.0);
                    al[k] = l + 5.0;
                    au[k] = u + 6.0;
                    bl[k] = l - 6.0;
                    bu[k] = u - 5.0;
                }
                _ => {
                    // point box for a, interval for b
                    let p = rng.uniform(-4.0, 4.0);
                    al[k] = p;
                    au[k] = p;
                    let (l, u) = rng.interval(-4.0, 4.0);
                    bl[k] = l;
                    bu[k] = u;
                }
            }
        }
        let a = bt1(&al, &au);
        let b = bt1(&bl, &bu);
        let spec = random_spec(&mut rng, out_dim, n);

        let (ba, bb) = max_layer
            .propagate_linear_binary(&spec, &a, &b)
            .expect("max CROWN backward");
        sample_binary_box(&a, &b, &spec, &ba, &bb, 16, |x, y| x.max(y), "MaxBinary");

        let (ba, bb) = min_layer
            .propagate_linear_binary(&spec, &a, &b)
            .expect("min CROWN backward");
        sample_binary_box(&a, &b, &spec, &ba, &bb, 16, |x, y| x.min(y), "MinBinary");
        tested += 1;
    }
    assert_eq!(tested, 400);
}

// ── 2. Atan2 binary CROWN envelope ───────────────────────────────────────────

#[test]
fn atan2_crown_envelope_sound_dense_random() {
    let mut rng = Rng::new(0xA72A_2111);
    let n = 3usize;
    let out_dim = 2usize;
    let layer = Atan2Layer;

    // input_a = y, input_b = x; f(a,b) = atan2(y,x) = a.atan2(b).
    let f = |y: f32, x: f32| (y as f64).atan2(x as f64) as f32;

    let mut relaxed_count = 0usize;
    let mut fallback_count = 0usize;
    for trial in 0..600 {
        // Build per-element boxes; deliberately include ill-conditioned ones to
        // confirm the whole-op IBP fallback (Err) and never an unsound plane.
        let mut yl = vec![0.0f32; n];
        let mut yu = vec![0.0f32; n];
        let mut xl = vec![0.0f32; n];
        let mut xu = vec![0.0f32; n];
        for k in 0..n {
            match (trial + 7 * k) % 6 {
                0 => {
                    // Q1 strict
                    let (l, u) = rng.interval(0.2, 5.0);
                    xl[k] = l;
                    xu[k] = u;
                    let (l2, u2) = rng.interval(0.2, 5.0);
                    yl[k] = l2;
                    yu[k] = u2;
                }
                1 => {
                    // right-half-plane, y straddles 0 (x>0 strict)
                    let (l, u) = rng.interval(0.3, 4.0);
                    xl[k] = l;
                    xu[k] = u;
                    let (l2, u2) = rng.interval(-3.0, 3.0);
                    yl[k] = l2;
                    yu[k] = u2;
                }
                2 => {
                    // Q3 strict (x<0, y<0): negative-branch handling
                    let (l, u) = rng.interval(-5.0, -0.2);
                    xl[k] = l;
                    xu[k] = u;
                    let (l2, u2) = rng.interval(-5.0, -0.2);
                    yl[k] = l2;
                    yu[k] = u2;
                }
                3 => {
                    // contains origin -> must fall back
                    xl[k] = -1.0;
                    xu[k] = 1.0;
                    yl[k] = -1.0;
                    yu[k] = 1.0;
                }
                4 => {
                    // straddles branch cut (x<0, y spans 0) -> fall back
                    xl[k] = -4.0;
                    xu[k] = -1.0;
                    yl[k] = -0.5;
                    yu[k] = 0.5;
                }
                _ => {
                    // tiny box very near origin but Q1 (well-conditioned, tight)
                    let cx = rng.uniform(0.05, 0.2);
                    let cy = rng.uniform(0.05, 0.2);
                    xl[k] = cx;
                    xu[k] = cx + 0.01;
                    yl[k] = cy;
                    yu[k] = cy + 0.01;
                }
            }
        }
        let y = bt1(&yl, &yu);
        let x = bt1(&xl, &xu);
        let spec = random_spec(&mut rng, out_dim, n);

        match layer.propagate_linear_binary(&spec, &y, &x) {
            Ok((ba, bb)) => {
                // A relaxation was produced for every element -> dense soundness.
                sample_binary_box(&y, &x, &spec, &ba, &bb, 24, f, "Atan2");
                relaxed_count += 1;
            }
            Err(_) => {
                // Sound IBP fallback for ill-conditioned boxes. Nothing to check.
                fallback_count += 1;
            }
        }
    }
    // We must have exercised both the relaxed path and the fallback path.
    assert!(relaxed_count > 0, "expected some relaxed atan2 boxes");
    assert!(fallback_count > 0, "expected some atan2 fallbacks");
}

#[test]
fn atan2_pure_quadrant_relaxation_is_dense_sound() {
    // Force every element well-conditioned so propagate_linear_binary always
    // returns a relaxation, and sample densely (Q1..Q4 + right-half-plane).
    let mut rng = Rng::new(0xA72A_2222);
    let n = 5usize;
    let out_dim = 3usize;
    let layer = Atan2Layer;
    let f = |y: f32, x: f32| (y as f64).atan2(x as f64) as f32;

    for _ in 0..300 {
        let mut yl = vec![0.0f32; n];
        let mut yu = vec![0.0f32; n];
        let mut xl = vec![0.0f32; n];
        let mut xu = vec![0.0f32; n];
        for k in 0..n {
            // strictly one quadrant: pick a sign for x and y away from 0.
            let xs = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let ys = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let (l, u) = rng.interval(0.3, 6.0);
            if xs > 0.0 {
                xl[k] = l;
                xu[k] = u;
            } else {
                xl[k] = -u;
                xu[k] = -l;
            }
            let (l2, u2) = rng.interval(0.3, 6.0);
            if ys > 0.0 {
                yl[k] = l2;
                yu[k] = u2;
            } else {
                yl[k] = -u2;
                yu[k] = -l2;
            }
        }
        let y = bt1(&yl, &yu);
        let x = bt1(&xl, &xu);
        let spec = random_spec(&mut rng, out_dim, n);
        let (ba, bb) = layer
            .propagate_linear_binary(&spec, &y, &x)
            .expect("well-conditioned atan2 must relax");
        sample_binary_box(&y, &x, &spec, &ba, &bb, 24, f, "Atan2-quadrant");
    }
}

// ── 3. Div binary CROWN (reciprocal scaling incl. negative denominator) ──────

/// Drive `backward_div_to_numerator` and densely sample the soundness sandwich:
/// the propagated numerator linear form must bound the incoming-spec form
/// evaluated at `z_i = a_i / b_i`, for all `(a, b)` in their boxes.
/// Returns `true` iff the relaxation (PropagateNumerator) path was exercised
/// (vs. the always-sound concretization fallback).
fn check_div_backward_sound(
    a: &BoundedTensor,
    b: &BoundedTensor,
    node_out: &BoundedTensor,
    spec: &LinearBounds,
    steps: usize,
    label: &str,
) -> bool {
    let res = backward_div_to_numerator(spec, a, b, node_out).expect("div backward");
    let num_lb = match res {
        DivBackwardResult::PropagateNumerator(lb) => *lb,
        // Concretization fallback is always sound (it replaces the node by its
        // IBP output bounds); we only stress the relaxation arithmetic here.
        DivBackwardResult::ConcretizeCurrentNode(_) => return false,
    };

    let n = a.len();
    let al = a.lower();
    let au = a.upper();
    let bl = b.lower();
    let bu = b.upper();
    let out_dim = spec.num_outputs();

    let mut a_vals = vec![0.0f32; n];
    let mut b_vals = vec![0.0f32; n];
    for i in 0..=steps {
        let ta = i as f32 / steps as f32;
        for j in 0..=steps {
            let tb = j as f32 / steps as f32;
            for k in 0..n {
                a_vals[k] = al[k] + (au[k] - al[k]) * ta;
                // anti-correlate b vs a to vary z across the grid
                b_vals[k] = bl[k] + (bu[k] - bl[k]) * (1.0 - tb);
            }
            for s in 0..out_dim {
                // Incoming spec form at z = a/b.
                let mut spec_lower = spec.lower_b()[s] as f64;
                let mut spec_upper = spec.upper_b()[s] as f64;
                for k in 0..n {
                    let z = a_vals[k] as f64 / b_vals[k] as f64;
                    spec_lower += spec.lower_a()[[s, k]] as f64 * z;
                    spec_upper += spec.upper_a()[[s, k]] as f64 * z;
                }
                // Propagated numerator form at a.
                let mut num_lower = num_lb.lower_b()[s] as f64;
                let mut num_upper = num_lb.upper_b()[s] as f64;
                for k in 0..n {
                    num_lower += num_lb.lower_a()[[s, k]] as f64 * a_vals[k] as f64;
                    num_upper += num_lb.upper_a()[[s, k]] as f64 * a_vals[k] as f64;
                }
                let scale = 1.0 + spec_lower.abs().max(spec_upper.abs());
                let tol = 1e-3 * scale;
                assert!(
                    num_lower <= spec_lower + tol,
                    "{label}: UNSOUND div lower: num {num_lower} > spec {spec_lower} (row {s}) \
                     a={a_vals:?} b={b_vals:?}"
                );
                assert!(
                    num_upper >= spec_upper - tol,
                    "{label}: UNSOUND div upper: num {num_upper} < spec {spec_upper} (row {s}) \
                     a={a_vals:?} b={b_vals:?}"
                );
            }
        }
    }
    true
}

#[test]
fn div_crown_backward_sound_dense_random() {
    let mut rng = Rng::new(0x0D1F_5EED);
    let n = 4usize;
    let out_dim = 3usize;

    let mut pos_tested = 0usize;
    let mut neg_tested = 0usize;
    let mut relaxed_paths = 0usize;
    for trial in 0..500 {
        // Denominator must be sign-definite for the relaxation path. Alternate
        // all-positive vs all-negative, with near-zero magnitudes (adversarial).
        let denom_negative = trial % 2 == 1;
        let mut al = vec![0.0f32; n];
        let mut au = vec![0.0f32; n];
        let mut bl = vec![0.0f32; n];
        let mut bu = vec![0.0f32; n];
        for k in 0..n {
            // numerator: any sign, sometimes spanning 0
            let (l, u) = rng.interval(-6.0, 6.0);
            al[k] = l;
            au[k] = u;
            // denominator magnitude: include very small (near-zero) divisors.
            let mag_lo = match (trial + k) % 3 {
                0 => 0.05, // near-zero divisor (stress reciprocal blow-up)
                1 => 0.5,
                _ => 1.5,
            };
            let (l2, u2) = rng.interval(mag_lo, mag_lo + 4.0);
            // ensure strictly > 0 lower
            let l2 = l2.max(mag_lo);
            if denom_negative {
                bl[k] = -u2;
                bu[k] = -l2;
            } else {
                bl[k] = l2;
                bu[k] = u2;
            }
        }
        let a = bt1(&al, &au);
        let b = bt1(&bl, &bu);
        // node output bounds: a sound IBP enclosure of a/b (only used by the
        // fallback path; the relaxation path ignores it). Use a wide safe box.
        let mut ol = vec![0.0f32; n];
        let mut ou = vec![0.0f32; n];
        for k in 0..n {
            // crude sound enclosure of a/b over the boxes
            let candidates = [al[k] / bl[k], al[k] / bu[k], au[k] / bl[k], au[k] / bu[k]];
            ol[k] = candidates.iter().cloned().fold(f32::INFINITY, f32::min);
            ou[k] = candidates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }
        let node_out = bt1(&ol, &ou);
        let spec = random_spec(&mut rng, out_dim, n);

        if check_div_backward_sound(&a, &b, &node_out, &spec, 18, "Div") {
            relaxed_paths += 1;
        }
        if denom_negative {
            neg_tested += 1;
        } else {
            pos_tested += 1;
        }
    }
    assert!(pos_tested > 0 && neg_tested > 0);
    // The whole point of this audit is the reciprocal-scaling relaxation: make
    // sure it was actually exercised (positive AND negative denominators) rather
    // than the test silently passing through the concretization fallback.
    assert!(
        relaxed_paths > 100,
        "expected the Div PropagateNumerator relaxation path to be exercised heavily, got {relaxed_paths}"
    );
}

// ── 4. Exact-linear scatter ops: CROWN backward == dense transpose ───────────

/// Probe a single-variable-operand exact-linear layer's forward matrix `M`
/// (output_size × var_size) and constant shift vector `c` (length output_size)
/// by running IBP on point inputs: `y = M @ var + c`.
///
/// `forward` runs the layer's IBP with the variable operand fixed to a concrete
/// vector (lower == upper) and returns the (concrete) output vector.
fn probe_forward_matrix<Fwd>(
    output_size: usize,
    var_size: usize,
    forward: Fwd,
) -> (Array2<f64>, Array1<f64>)
where
    Fwd: Fn(&[f32]) -> Vec<f32>,
{
    // c = forward(0).
    let zero = vec![0.0f32; var_size];
    let c_vec = forward(&zero);
    assert_eq!(c_vec.len(), output_size);
    let c = Array1::from_iter(c_vec.iter().map(|&v| v as f64));

    let mut m = Array2::<f64>::zeros((output_size, var_size));
    for k in 0..var_size {
        let mut e = vec![0.0f32; var_size];
        e[k] = 1.0;
        let col = forward(&e);
        for (o, &v) in col.iter().enumerate() {
            // M[o,k] = forward(e_k)[o] - c[o].
            m[[o, k]] = v as f64 - c[o];
        }
    }
    (m, c)
}

/// Assert that the layer's CROWN backward equals the dense transpose:
/// `new_A == node_lb.A @ M` and `new_b == node_lb.b + node_lb.A @ c`,
/// for both the lower and upper planes.
fn assert_crown_equals_dense_transpose(
    node_lb: &LinearBounds,
    m: &Array2<f64>,
    c: &Array1<f64>,
    backward: &LinearBounds,
    label: &str,
) {
    let out_dim = node_lb.num_outputs();
    let var_size = m.shape()[1];
    let output_size = m.shape()[0];

    // Expected dense transpose in f64.
    let la = node_lb.lower_a().mapv(|v| v as f64);
    let ua = node_lb.upper_a().mapv(|v| v as f64);
    let exp_la = la.dot(m); // (out_dim × var_size)
    let exp_ua = ua.dot(m);

    for s in 0..out_dim {
        for k in 0..var_size {
            let got_l = backward.lower_a()[[s, k]] as f64;
            let got_u = backward.upper_a()[[s, k]] as f64;
            assert!(
                (got_l - exp_la[[s, k]]).abs() <= 1e-4 * (1.0 + exp_la[[s, k]].abs()),
                "{label}: lower_A[{s},{k}] {got_l} != dense {} ",
                exp_la[[s, k]]
            );
            assert!(
                (got_u - exp_ua[[s, k]]).abs() <= 1e-4 * (1.0 + exp_ua[[s, k]].abs()),
                "{label}: upper_A[{s},{k}] {got_u} != dense {}",
                exp_ua[[s, k]]
            );
        }
        // Bias: node_lb.b + node_lb.A @ c.
        let mut exp_lb = node_lb.lower_b()[s] as f64;
        let mut exp_ub = node_lb.upper_b()[s] as f64;
        for o in 0..output_size {
            exp_lb += node_lb.lower_a()[[s, o]] as f64 * c[o];
            exp_ub += node_lb.upper_a()[[s, o]] as f64 * c[o];
        }
        let got_lb = backward.lower_b()[s] as f64;
        let got_ub = backward.upper_b()[s] as f64;
        assert!(
            (got_lb - exp_lb).abs() <= 1e-3 * (1.0 + exp_lb.abs()),
            "{label}: lower_b[{s}] {got_lb} != dense {exp_lb}"
        );
        assert!(
            (got_ub - exp_ub).abs() <= 1e-3 * (1.0 + exp_ub.abs()),
            "{label}: upper_b[{s}] {got_ub} != dense {exp_ub}"
        );
    }
}

#[test]
fn scatter_add_src_variable_crown_equals_dense_transpose() {
    // y = data_const + scatter_add(src_var) along axis 0 with duplicate indices
    // (adversarial: src elements 1 and 3 both scatter into output position 2).
    let mut rng = Rng::new(0x5CA7_70E5);
    let data_const =
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let indices = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0i64, 2, 4, 2]).unwrap();
    let layer = ScatterAddLayer::new(0, Some(data_const), Some(indices), None);
    let output_size = 5;
    let var_size = 4;

    let forward = |src: &[f32]| -> Vec<f32> {
        let src_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[var_size]), src.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[var_size]), src.to_vec()).unwrap(),
        )
        .unwrap();
        // Single variable operand (data + indices constant): use the unary
        // BoundPropagation IBP entry.
        let out = layer.propagate_ibp(&src_bt).unwrap();
        out.lower().iter().copied().collect()
    };
    let (m, c) = probe_forward_matrix(output_size, var_size, forward);

    for _ in 0..50 {
        let spec = random_spec(&mut rng, 4, output_size);
        let back = layer.crown_backward(&spec).expect("scatter_add backward");
        assert_crown_equals_dense_transpose(&spec, &m, &c, &back, "ScatterAdd-src");
    }
}

#[test]
fn index_add_src_variable_crown_equals_dense_transpose() {
    // y = data_const + index_add(src_var) along axis 0, duplicate index again.
    let mut rng = Rng::new(0x1DEA_0ADD);
    let data_const = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let indices = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1i64, 1, 3]).unwrap();
    let layer = IndexAddLayer::new(0, Some(data_const), Some(indices), None);
    let output_size = 4;
    let var_size = 3;

    let forward = |src: &[f32]| -> Vec<f32> {
        let src_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[var_size]), src.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[var_size]), src.to_vec()).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&src_bt).unwrap();
        out.lower().iter().copied().collect()
    };
    let (m, c) = probe_forward_matrix(output_size, var_size, forward);

    for _ in 0..50 {
        let spec = random_spec(&mut rng, 3, output_size);
        let back = layer.crown_backward(&spec).expect("index_add backward");
        assert_crown_equals_dense_transpose(&spec, &m, &c, &back, "IndexAdd-src");
    }
}

#[test]
fn scatter_nd_updates_variable_crown_equals_dense_transpose() {
    // 1-D data overwrite: y[idx] = updates_var, y[other] = data_const.
    // Distinct (non-duplicate) targets — duplicates are rejected by the layer.
    let mut rng = Rng::new(0x5CA7_70D7);
    let data_const =
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let indices = ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![0i64, 2, 5]).unwrap();
    let layer = ScatterNdLayer::new(Some(data_const), Some(indices), None);
    let output_size = 6;
    let var_size = 3;

    let forward = |upd: &[f32]| -> Vec<f32> {
        let upd_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[var_size]), upd.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[var_size]), upd.to_vec()).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&upd_bt).unwrap();
        out.lower().iter().copied().collect()
    };
    let (m, c) = probe_forward_matrix(output_size, var_size, forward);

    for _ in 0..50 {
        let spec = random_spec(&mut rng, 4, output_size);
        let back = layer.crown_backward(&spec).expect("scatter_nd backward");
        assert_crown_equals_dense_transpose(&spec, &m, &c, &back, "ScatterND-updates");
    }
}

// ── 5. Degenerate BatchNorm NaN/Inf handling (zero-variance channels) ────────

/// True value `y = x*scale + bias` for a 1-D-per-channel BatchNorm, computed in
/// f64 from the layer's stored (possibly Inf) coefficients. Returns NaN only if
/// the affine genuinely evaluates to NaN (e.g. `0*Inf`), which the soundness
/// check treats as "must be covered by an Inf-widened interval".
fn bn_true_value(scale: f32, bias: f32, x: f32) -> f64 {
    scale as f64 * x as f64 + bias as f64
}

/// Saturation threshold for treating a bound as "effectively unbounded".
/// `new_repaired` preserves `±Inf` endpoints and widens NaN to `±Inf` (a
/// non-finite endpoint carries no proven bound, so no finite substitute is
/// sound), while the explicit `new_sanitized(clamp)` path used by
/// `continue_after_overflow` still saturates to `±FALLBACK_BOUND` (see
/// `bounded_tensor::core::constructors` and `test_new_sanitized_with_infinity`).
/// A degenerate (zero-variance) BatchNorm channel produces an Inf scale and
/// hence an ±Inf-magnitude true output; the sound enclosure is unbounded on
/// that side — at least this threshold, NOT a tighter finite interval.
const FALLBACK_BOUND: f32 = 1e10;

/// Soundness for a (possibly degenerate) interval `[lo, hi]` enclosing true `y`.
///
/// A non-finite endpoint (`±Inf`) or a saturated one (at or beyond
/// `±FALLBACK_BOUND`) is the conservative widening and contains any
/// finite/infinite true value on that side. When the true value is an honest
/// `±Inf` (Inf scale times a nonzero x), the interval must reach at least the
/// saturation threshold on that side — i.e. it must NOT pretend a finite
/// tighter-than-threshold bound.
fn interval_contains(lo: f32, hi: f32, y: f64) -> bool {
    // If y itself is NaN (0*Inf indeterminate), require maximal widening: the
    // interval must be saturated on BOTH sides (threshold or Inf).
    if y.is_nan() {
        let lo_sat = lo <= -FALLBACK_BOUND;
        let hi_sat = hi >= FALLBACK_BOUND;
        return lo_sat && hi_sat;
    }
    // Honest ±Inf true value: the bound on that side must reach the saturation
    // threshold (or be Inf), never a finite value tighter than the threshold.
    if y == f64::NEG_INFINITY {
        return lo <= -FALLBACK_BOUND;
    }
    if y == f64::INFINITY {
        return hi >= FALLBACK_BOUND;
    }
    // Finite true value: ordinary sandwich, with saturated/Inf endpoints always
    // accepted as conservative.
    let lo_ok = lo <= -FALLBACK_BOUND || (lo as f64) <= y + 1e-3 * (1.0 + y.abs());
    let hi_ok = hi >= FALLBACK_BOUND || (hi as f64) >= y - 1e-3 * (1.0 + y.abs());
    lo_ok && hi_ok
}

#[test]
fn batchnorm_degenerate_zero_variance_is_sound() {
    // Channels:
    //   0: normal (var large)         -> finite scale
    //   1: zero variance (var+eps==0) -> scale = ny/0 = +Inf (degenerate)
    //   2: zero variance, nonzero mean -> scale = +Inf, bias = beta - mean*Inf = -Inf
    //   3: ny == 0 with zero variance -> scale = 0/0 = NaN? new() => 0*? Actually
    //      ny=0 / sqrt(0)=0/0 = NaN scale; the layer keeps it (sound via widening).
    let eps = 1e-5f32;
    let ny = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 1.0, 1.5, 0.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, -0.25, 1.0, 0.3]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 2.0, 1.0]).unwrap();
    // var+eps: channel0 large, channels 1..3 exactly -eps so var+eps == 0.
    let var = ArrayD::from_shape_vec(IxDyn(&[4]), vec![4.0, -eps, -eps, -eps]).unwrap();
    let layer = BatchNormLayer::new(&ny, &beta, &mean, &var, eps).expect("degenerate BN builds");

    // Confirm the degenerate channels really produced non-finite coefficients,
    // otherwise this test isn't exercising the firewall it claims to.
    let any_nonfinite_scale = layer.scale.iter().any(|v| !v.is_finite());
    assert!(
        any_nonfinite_scale,
        "expected an Inf/NaN scale to audit the firewall"
    );

    // Input box: shape [1, 4] (NCHW-ish, channel axis = 1). Sample densely.
    let mut rng = Rng::new(0xB47C_0444);
    for _ in 0..200 {
        let mut xl = vec![0.0f32; 4];
        let mut xu = vec![0.0f32; 4];
        for k in 0..4 {
            let (l, u) = rng.interval(-3.0, 3.0);
            xl[k] = l;
            xu[k] = u;
        }
        let x_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), xl.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), xu.clone()).unwrap(),
        )
        .unwrap();

        // 5a. IBP soundness.
        let ibp = layer.propagate_ibp(&x_bt).expect("BN IBP must not abort");
        let il = ibp.lower();
        let iu = ibp.upper();
        for c in 0..4 {
            let s = layer.scale[[c]];
            let b = layer.bias[[c]];
            let lo = il[[0, c]];
            let hi = iu[[0, c]];
            assert!(
                !lo.is_nan() && !hi.is_nan(),
                "BN IBP produced NaN bound at channel {c}: [{lo}, {hi}]"
            );
            // Sample the channel box.
            for t in 0..=20 {
                let x = xl[c] + (xu[c] - xl[c]) * (t as f32 / 20.0);
                let y = bn_true_value(s, b, x);
                assert!(
                    interval_contains(lo, hi, y),
                    "BN IBP UNSOUND ch {c}: y={y} not in [{lo},{hi}] (scale={s}, bias={b}, x={x})"
                );
            }
        }

        // 5b. CROWN-backward soundness via identity spec, then concretize.
        let ident = LinearBounds::identity(4);
        let crown = layer
            .propagate_linear_with_bounds(&ident, &x_bt)
            .expect("BN CROWN must not abort");
        // Concretize over the input box (flattened to 4 inputs).
        let x_flat = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), xl.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), xu.clone()).unwrap(),
        )
        .unwrap();
        let conc = crown.concretize_sound(&x_flat);
        let cl = conc.lower();
        let cu = conc.upper();
        for c in 0..4 {
            let s = layer.scale[[c]];
            let b = layer.bias[[c]];
            let lo = cl[c];
            let hi = cu[c];
            assert!(
                !lo.is_nan() && !hi.is_nan(),
                "BN CROWN concretize produced NaN bound at channel {c}: [{lo}, {hi}]"
            );
            for t in 0..=20 {
                let x = xl[c] + (xu[c] - xl[c]) * (t as f32 / 20.0);
                let y = bn_true_value(s, b, x);
                assert!(
                    interval_contains(lo, hi, y),
                    "BN CROWN UNSOUND ch {c}: y={y} not in [{lo},{hi}] (scale={s}, bias={b}, x={x})"
                );
            }
        }
    }
}

// ── 6. gpu_bab "tighter of two sound bounds" merge invariant ──────────────────
//
// The beta-CROWN engine's `tighten_root_with_graph_alpha_crown` intersects two
// INDEPENDENTLY-SOUND bound sources (sequential α-CROWN and graph α-CROWN) by
// taking `max` of the lower bounds and `min` of the upper bounds, skipping any
// non-finite value so it can never "win". The directional leaf-beta optimizer
// similarly keeps the tightest *finite* bound in the verification direction. The
// load-bearing soundness property of BOTH is:
//
//     given two sound enclosures of the same true value, the element-wise
//     tighter-of-the-two (finite-guarded) is STILL a sound enclosure.
//
// We replicate the exact merge/selection arithmetic here and verify it against a
// hidden true value over dense random sound bound pairs — including non-finite
// (±Inf) endpoints from either source. This is a focused, conflict-free guard on
// the merge logic without depending on the (in-flux) engine internals.

/// Mirror of the production `merge` closure in `tighten_root_with_graph_alpha_crown`.
fn merge_sound(a: f32, b: f32, take_max: bool) -> f32 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => {
            if take_max {
                a.max(b)
            } else {
                a.min(b)
            }
        }
        (true, false) => a,
        (false, true) => b,
        (false, false) => a,
    }
}

#[test]
fn gpu_bab_tighter_of_two_sound_bounds_stays_sound() {
    let mut rng = Rng::new(0x6B7B_A8B5);
    for _ in 0..20_000 {
        // Hidden true value the bounds must enclose.
        let y = rng.uniform(-50.0, 50.0);

        // Two independently-sound enclosures. Each lower <= y <= upper. We also
        // inject ±Inf endpoints (loose-but-sound) to exercise the finite skip.
        let make_lower = |r: &mut Rng| -> f32 {
            if r.next_u64().is_multiple_of(7) {
                f32::NEG_INFINITY
            } else {
                y - r.uniform(0.0, 20.0)
            }
        };
        let make_upper = |r: &mut Rng| -> f32 {
            if r.next_u64().is_multiple_of(7) {
                f32::INFINITY
            } else {
                y + r.uniform(0.0, 20.0)
            }
        };

        let sl = make_lower(&mut rng);
        let su = make_upper(&mut rng);
        let gl = make_lower(&mut rng);
        let gu = make_upper(&mut rng);

        let merged_lower = merge_sound(sl, gl, true);
        let merged_upper = merge_sound(su, gu, false);

        // 1. Soundness: the merged interval must still enclose the true value.
        assert!(
            merged_lower <= y + 1e-4 * (1.0 + y.abs()),
            "merged lower {merged_lower} > true {y} (sl={sl}, gl={gl})"
        );
        assert!(
            merged_upper >= y - 1e-4 * (1.0 + y.abs()),
            "merged upper {merged_upper} < true {y} (su={su}, gu={gu})"
        );

        // 2. Tightening: the merged bound is never looser than either source on
        //    the side where both are finite (it picks the tighter one).
        if sl.is_finite() && gl.is_finite() {
            assert!(merged_lower >= sl.min(gl) - 1e-6 && merged_lower >= sl.max(gl) - 1e-6);
        }
        if su.is_finite() && gu.is_finite() {
            assert!(merged_upper <= su.max(gu) + 1e-6 && merged_upper <= su.min(gu) + 1e-6);
        }

        // 3. A non-finite value from one source must never win over a finite one.
        if sl.is_finite() && !gl.is_finite() {
            assert_eq!(merged_lower, sl);
        }
        if su.is_finite() && !gu.is_finite() {
            assert_eq!(merged_upper, su);
        }
    }
}

/// Mirror of the directional leaf-beta "keep tightest finite bound in the
/// verification direction" selection, asserting the carried bound stays sound.
#[test]
fn gpu_bab_directional_leaf_beta_keeps_sound_tightest() {
    let mut rng = Rng::new(0x1EAF_BE7A);
    for _ in 0..2000 {
        let y = rng.uniform(-30.0, 30.0);
        let verify_upper = rng.next_u64() & 1 == 0;

        // Stream of sound iterates (lb_i <= y <= ub_i), some with ±Inf.
        let n_iters = 12;
        let base_l = y - rng.uniform(1.0, 15.0);
        let base_u = y + rng.uniform(1.0, 15.0);
        let mut best_lower = base_l;
        let mut best_upper = base_u;
        let mut best_decision = if verify_upper { base_u } else { base_l };

        for _ in 0..n_iters {
            let lb = if rng.next_u64().is_multiple_of(5) {
                f32::NEG_INFINITY
            } else {
                y - rng.uniform(0.0, 12.0)
            };
            let ub = if rng.next_u64().is_multiple_of(5) {
                f32::INFINITY
            } else {
                y + rng.uniform(0.0, 12.0)
            };

            // Exact production selection logic.
            if verify_upper {
                if ub.is_finite() && ub < best_decision {
                    best_decision = ub;
                    best_upper = ub;
                    if lb.is_finite() {
                        best_lower = lb;
                    }
                }
            } else if lb.is_finite() && lb > best_decision {
                best_decision = lb;
                best_lower = lb;
                if ub.is_finite() {
                    best_upper = ub;
                }
            }
        }

        // The carried best bounds must remain sound enclosures of y.
        assert!(
            best_lower <= y + 1e-4 * (1.0 + y.abs()),
            "directional best_lower {best_lower} > true {y}"
        );
        assert!(
            best_upper >= y - 1e-4 * (1.0 + y.abs()),
            "directional best_upper {best_upper} < true {y}"
        );
        // The decision-direction bound is finite (never an Inf masquerading as tight).
        assert!(best_decision.is_finite());
    }
}
