// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound linear relaxations for element-wise `max(x, y)` and `min(x, y)`.
//!
//! Both ops are piecewise-linear and convex (max) / concave (min), so they
//! admit exact convex-hull linear envelopes over an axis-aligned input box
//! `[lx, ux] × [ly, uy]`. These envelopes feed the CROWN backward pass exactly
//! like the McCormick envelope for `Mul`.
//!
//! # Derivation (max)
//!
//! `max(x, y) = y + relu(x - y)`. Let `t = x - y`, ranging over
//! `[t_l, t_u] = [lx - uy, ux - ly]`. Then `max(x, y) = y + relu(t)`.
//!
//! * If `t_u <= 0` then `x <= y` on the whole box, so `max = y` exactly.
//! * If `t_l >= 0` then `x >= y` on the whole box, so `max = x` exactly.
//! * Otherwise (`t_l < 0 < t_u`) we use the standard ReLU convex hull:
//!     - Upper: `relu(t) <= s * (t - t_l)` with slope `s = t_u / (t_u - t_l)`,
//!       giving `max(x, y) <= s*x + (1-s)*y - s*t_l`. The slope `s` and
//!       `1-s` are rounded to f32, perturbing the ideal coefficient pair, so the
//!       constant is NOT the ideal `-s*t_l`: instead it is recomputed (and
//!       rounded up) from the worst box corner using the ACTUAL f32 coefficients
//!       (`max_corner_residual`). Because `max(x,y) - (a*x + b*y)` is convex its
//!       maximum over the box is at a corner, so this keeps the plane provably
//!       `>= max` everywhere regardless of the f32 slope perturbation.
//!     - Lower: `relu(t) >= 0` and `relu(t) >= t`, giving the two sound
//!       lower planes `max(x, y) >= y` and `max(x, y) >= x`. Any convex
//!       combination of these two planes is also a sound lower bound because
//!       `max` is convex.
//!
//! # Derivation (min)
//!
//! `min(x, y) = -max(-x, -y)`. We reuse the max envelope on the negated box
//! and negate the resulting planes. Concretely the roles of lower/upper swap.
//!
//! Soundness is verified by dense sampling in the unit tests of `min.rs` /
//! `max.rs` and the proptest soundness suite.

use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::shape::{broadcast_flat_index_map, broadcast_shapes};
use crate::{contiguous_flat_slice, LinearBounds};

/// An affine plane `coeff_x * x + coeff_y * y + c`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plane {
    pub coeff_x: f32,
    pub coeff_y: f32,
    pub c: f32,
}

impl Plane {
    /// Evaluate the plane at `(x, y)` in f64 for tie-break comparisons.
    #[inline]
    fn eval(&self, x: f32, y: f32) -> f64 {
        self.coeff_x as f64 * x as f64 + self.coeff_y as f64 * y as f64 + self.c as f64
    }
}

/// Lower and upper affine envelopes for an element-wise binary op.
#[derive(Clone, Copy, Debug)]
pub(super) struct Envelope {
    /// `z >= lower(x, y)` for all `(x, y)` in the box.
    pub lower: Plane,
    /// `z <= upper(x, y)` for all `(x, y)` in the box.
    pub upper: Plane,
}

#[cfg(test)]
impl Envelope {
    /// Evaluate the lower plane at `(x, y)` (f32). Used by sibling-module
    /// relaxation soundness tests (e.g. `atan2_relax`).
    #[inline]
    pub(super) fn lower_eval(&self, x: f32, y: f32) -> f32 {
        self.lower.eval(x, y) as f32
    }

    /// Evaluate the upper plane at `(x, y)` (f32).
    #[inline]
    pub(super) fn upper_eval(&self, x: f32, y: f32) -> f32 {
        self.upper.eval(x, y) as f32
    }
}

/// Largest value of `f(x, y) - (coeff_x*x + coeff_y*y)` over the four corners
/// of the box `[lx, ux] × [ly, uy]`, evaluated in f64.
///
/// For a convex `f` (e.g. `max`) the linear-corrected residual `f - linear` is
/// convex, so its maximum over the box is attained at a corner. The caller adds
/// (then `next_up_f32`-rounds) this as the upper-plane constant `c`, guaranteeing
/// `coeff_x*x + coeff_y*y + c >= f(x, y)` everywhere on the box regardless of the
/// f32 rounding applied to `coeff_x`/`coeff_y`. Each `coeff as f64 * v as f64`
/// product is exact (both operands are f32-exact-in-f64), so the only rounding is
/// the final f64 subtraction, which the caller's `next_up_f32` absorbs.
#[inline]
fn max_corner_residual(
    coeff_x: f32,
    coeff_y: f32,
    lx: f32,
    ux: f32,
    ly: f32,
    uy: f32,
    f: impl Fn(f32, f32) -> f64,
) -> f64 {
    let cx = coeff_x as f64;
    let cy = coeff_y as f64;
    let mut best = f64::NEG_INFINITY;
    for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
        let resid = f(x, y) - (cx * x as f64 + cy * y as f64);
        if resid > best {
            best = resid;
        }
    }
    best
}

/// Sound linear envelope for `z = max(x, y)` over `[lx, ux] × [ly, uy]`.
///
/// Returns `None` if any bound is non-finite (caller keeps the IBP fallback).
#[inline]
pub(super) fn max_envelope(lx: f32, ux: f32, ly: f32, uy: f32) -> Option<Envelope> {
    if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
        return None;
    }

    // t = x - y ranges over [t_l, t_u].
    let t_l = lx - uy;
    let t_u = ux - ly;

    // Degenerate: x <= y everywhere => max = y.
    if t_u <= 0.0 {
        let p = Plane {
            coeff_x: 0.0,
            coeff_y: 1.0,
            c: 0.0,
        };
        return Some(Envelope { lower: p, upper: p });
    }
    // Degenerate: x >= y everywhere => max = x.
    if t_l >= 0.0 {
        let p = Plane {
            coeff_x: 1.0,
            coeff_y: 0.0,
            c: 0.0,
        };
        return Some(Envelope { lower: p, upper: p });
    }

    // Mixed: t_l < 0 < t_u. Upper = convex-hull (ReLU secant) envelope.
    let denom = (t_u - t_l) as f64;
    // denom = t_u - t_l > 0 because t_u > 0 > t_l here.
    let s = (t_u as f64 / denom) as f32; // slope in (0, 1)
    let coeff_x = s;
    let coeff_y = 1.0 - s; // computed in f32; perturbs the ideal (s, 1-s) pair

    // The mathematical upper plane is `s*x + (1-s)*y - s*t_l`, exact at the two
    // box corners where the ReLU secant touches `relu(t)`. But `coeff_x`/`coeff_y`
    // are the *rounded* f32 coefficients, so the ideal constant `-s*t_l` no longer
    // guarantees the plane sits above `max` at those corners (the audit's sub-1e-4
    // violation). Recompute the constant directly against the ACTUAL f32 coeffs so
    // the bias absorbs any f32 slope perturbation.
    //
    // `g(x,y) = max(x,y) - (coeff_x*x + coeff_y*y)` is convex (max convex, linear
    // part affine), so its maximum over the box is attained at a vertex. Setting
    // `c >= max_corners g` makes `coeff_x*x + coeff_y*y + c >= max(x,y)` on the
    // whole box for ANY coefficient choice. The corner residuals are computed in
    // f64 (each f32->f64 product is exact); `next_up_f32` then guarantees the
    // stored f32 `c` is >= the true f64 residual even after the cast-to-f32 round.
    let c_upper = max_corner_residual(coeff_x, coeff_y, lx, ux, ly, uy, |x, y| x.max(y) as f64);
    let upper = Plane {
        coeff_x,
        coeff_y,
        c: next_up_f32(c_upper as f32),
    };

    // Lower: the better of the two sound planes z >= x and z >= y is chosen at
    // CROWN time via weight-sign selection. Here we expose `z >= x` as the
    // canonical lower plane; the caller may also use `z >= y`. We pick the
    // plane with the larger value at the box midpoint to keep a single tight
    // lower plane (still sound, since both planes are sound everywhere).
    // Bit-identical plane-selection anchors: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let xm = 0.5 * (lx + ux);
    #[allow(clippy::manual_midpoint)]
    let ym = 0.5 * (ly + uy);
    let plane_x = Plane {
        coeff_x: 1.0,
        coeff_y: 0.0,
        c: 0.0,
    };
    let plane_y = Plane {
        coeff_x: 0.0,
        coeff_y: 1.0,
        c: 0.0,
    };
    let lower = if plane_x.eval(xm, ym) >= plane_y.eval(xm, ym) {
        plane_x
    } else {
        plane_y
    };

    Some(Envelope { lower, upper })
}

/// Sound linear envelope for `z = min(x, y)` over `[lx, ux] × [ly, uy]`.
///
/// Uses `min(x, y) = -max(-x, -y)`: build the max envelope on the negated box
/// and negate the planes, swapping lower/upper.
///
/// Returns `None` if any bound is non-finite (caller keeps the IBP fallback).
#[inline]
pub(super) fn min_envelope(lx: f32, ux: f32, ly: f32, uy: f32) -> Option<Envelope> {
    // Negated box: -x in [-ux, -lx], -y in [-uy, -ly].
    let env = max_envelope(-ux, -lx, -uy, -ly)?;
    // min(x,y) = -max(-x,-y). If max(-x,-y) <= U(-x,-y) = a*(-x)+b*(-y)+c
    // then min(x,y) = -max >= -(a*(-x)+b*(-y)+c) = a*x + b*y - c. So the
    // max upper plane becomes the min lower plane (negate constant only;
    // coeffs flip sign twice). Likewise max lower -> min upper.
    let lower = Plane {
        coeff_x: env.upper.coeff_x,
        coeff_y: env.upper.coeff_y,
        c: next_down_f32(-(env.upper.c as f64) as f32),
    };
    let upper = Plane {
        coeff_x: env.lower.coeff_x,
        coeff_y: env.lower.coeff_y,
        c: next_up_f32(-(env.lower.c as f64) as f32),
    };
    Some(Envelope { lower, upper })
}

/// Shared CROWN backward driver for element-wise `max`/`min`.
///
/// `envelope_fn` returns the sound lower/upper affine planes for a single
/// `(x, y)` element box; when it returns `None` for any required element the
/// whole op falls back to IBP (signalled via `UnsupportedOp`).
///
/// Mirrors `MulBinaryLayer::propagate_linear_binary`: broadcast-aware, splits
/// the McCormick-style constant into a separate bias channel carried entirely
/// on `bounds_a` (with `bounds_b` bias zeroed) so the DAG counts it once.
///
/// Weight-sign selection: for the output lower bound, a non-negative incoming
/// weight uses the lower plane of `z`, a negative weight uses the upper plane
/// (and vice versa for the output upper bound). This is the same rule the
/// `Mul` "Middle" relaxation uses and is exactly the requirement that the
/// resulting affine form remains a valid CROWN lower/upper bound.
pub(super) fn propagate_minmax_linear_binary<F>(
    bounds: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    op_name: &str,
    envelope_fn: F,
) -> Result<(LinearBounds, LinearBounds)>
where
    F: Fn(f32, f32, f32, f32) -> Option<Envelope>,
{
    let n = bounds.num_inputs();
    let num_outputs = bounds.num_outputs();
    let n_a = input_a_bounds.len();
    let n_b = input_b_bounds.len();

    let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: input_a_bounds.shape().to_vec(),
            got: input_b_bounds.shape().to_vec(),
        })?;
    let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{op_name} broadcast shape product overflows: {output_shape:?}"
        ))
    })?;
    if broadcast_n != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![broadcast_n],
        });
    }

    let a_idx_map = broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
    let b_idx_map = broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

    let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
    let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
    let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
    let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

    let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
    let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
    let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
    let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
    // f64 bias accumulation to avoid catastrophic cancellation (#2471).
    let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
    let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

    // Precompute per-element envelopes once (independent of output row). If any
    // element has a non-finite box, refuse and let the caller use IBP.
    let mut envelopes: Vec<Envelope> = Vec::with_capacity(n);
    for j in 0..n {
        let a_idx = a_idx_map[j];
        let b_idx = b_idx_map[j];
        let lx = a_lower_flat[a_idx];
        let ux = a_upper_flat[a_idx];
        let ly = b_lower_flat[b_idx];
        let uy = b_upper_flat[b_idx];
        let env = envelope_fn(lx, ux, ly, uy).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "{op_name} CROWN backward requires finite input bounds"
            ))
        })?;
        envelopes.push(env);
    }

    for out_idx in 0..num_outputs {
        let mut const_lower = bounds.lower_b()[out_idx] as f64;
        let mut const_upper = bounds.upper_b()[out_idx] as f64;

        for j in 0..n {
            let w_lower = bounds.lower_a()[[out_idx, j]];
            let w_upper = bounds.upper_a()[[out_idx, j]];
            let a_idx = a_idx_map[j];
            let b_idx = b_idx_map[j];
            let env = envelopes[j];

            // Output lower bound: w >= 0 uses z's lower plane, w < 0 uses upper.
            let pl = if w_lower >= 0.0 { env.lower } else { env.upper };
            lower_a_a[[out_idx, a_idx]] += w_lower * pl.coeff_x;
            lower_a_b[[out_idx, b_idx]] += w_lower * pl.coeff_y;
            const_lower += w_lower as f64 * pl.c as f64;

            // Output upper bound: w >= 0 uses z's upper plane, w < 0 uses lower.
            let pu = if w_upper >= 0.0 { env.upper } else { env.lower };
            upper_a_a[[out_idx, a_idx]] += w_upper * pu.coeff_x;
            upper_a_b[[out_idx, b_idx]] += w_upper * pu.coeff_y;
            const_upper += w_upper as f64 * pu.c as f64;
        }

        lower_b_total[out_idx] = const_lower;
        upper_b_total[out_idx] = const_upper;
    }

    let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
    let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

    let bounds_a =
        LinearBounds::new_or_conservative(lower_a_a, lower_b_f32, upper_a_a, upper_b_f32)?;
    let bounds_b = LinearBounds::new_or_conservative(
        lower_a_b,
        Array1::zeros(num_outputs),
        upper_a_b,
        Array1::zeros(num_outputs),
    )?;

    Ok((bounds_a, bounds_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// One ULP of `z` (the spacing to the next representable f32 above |z|).
    /// Used as the soundness tolerance instead of the old loose `1e-4`.
    fn ulp_of(z: f32) -> f64 {
        let a = z.abs();
        (next_up_f32(a) as f64 - a as f64).max(f64::MIN_POSITIVE)
    }

    /// Dense-sample the box and assert lower(x,y) <= f(x,y) <= upper(x,y) to
    /// within a TIGHT ~1-ULP tolerance (not the old 1e-4 that masked the
    /// sub-1e-4 envelope violation). The envelope planes are evaluated in f64
    /// exactly as the CROWN driver accumulates them, so a 1-ULP slack only
    /// covers the unavoidable final f32->f64 representation of the inputs.
    fn assert_envelope_sound(
        env: Envelope,
        lx: f32,
        ux: f32,
        ly: f32,
        uy: f32,
        f: impl Fn(f32, f32) -> f32,
    ) {
        let steps = 40;
        for i in 0..=steps {
            let x = lx + (ux - lx) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let y = ly + (uy - ly) * (j as f32 / steps as f32);
                let z = f(x, y) as f64;
                let lo = env.lower.eval(x, y);
                let hi = env.upper.eval(x, y);
                let tol = ulp_of(z as f32);
                assert!(
                    lo <= z + tol,
                    "lower {lo} > f {z} (slack {}) at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]",
                    lo - z
                );
                assert!(
                    hi >= z - tol,
                    "upper {hi} < f {z} (slack {}) at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]",
                    z - hi
                );
            }
        }
    }

    fn boxes() -> Vec<(f32, f32, f32, f32)> {
        vec![
            (-3.0, 2.0, -1.0, 4.0),   // mixed / mixed
            (0.0, 5.0, -2.0, 3.0),    // mixed
            (-5.0, -1.0, -4.0, -2.0), // both negative
            (1.0, 6.0, 2.0, 8.0),     // both positive, x mostly below y
            (5.0, 10.0, 0.0, 3.0),    // x dominates (x>=y everywhere)
            (0.0, 1.0, 5.0, 6.0),     // y dominates (x<=y everywhere)
            (-2.0, 2.0, -2.0, 2.0),   // symmetric
            (2.0, 2.0, -1.0, 3.0),    // x is a point
            (0.0, 0.0, 0.0, 0.0),     // degenerate point
        ]
    }

    #[test]
    fn max_envelope_is_sound_dense() {
        for (lx, ux, ly, uy) in boxes() {
            let env = max_envelope(lx, ux, ly, uy).unwrap();
            assert_envelope_sound(env, lx, ux, ly, uy, |x, y| x.max(y));
        }
    }

    #[test]
    fn min_envelope_is_sound_dense() {
        for (lx, ux, ly, uy) in boxes() {
            let env = min_envelope(lx, ux, ly, uy).unwrap();
            assert_envelope_sound(env, lx, ux, ly, uy, |x, y| x.min(y));
        }
    }

    /// Deterministic xorshift RNG returning f64 in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Strict per-corner check: the upper plane must be `>= max` and the lower
    /// plane must be `<= max` at every box corner with NO tolerance at all
    /// (the corners are where the relaxation is tight and the f32 slope
    /// perturbation is worst — this is exactly the envelope-violation the audit
    /// flagged). Evaluated in f64 the way the CROWN driver accumulates planes.
    fn assert_corners_strict(env: Envelope, lx: f32, ux: f32, ly: f32, uy: f32, is_max: bool) {
        for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
            let z = if is_max { x.max(y) } else { x.min(y) } as f64;
            let lo = env.lower.eval(x, y);
            let hi = env.upper.eval(x, y);
            assert!(
                hi >= z,
                "upper {hi} < {z} at corner ({x},{y}) box [{lx},{ux}]x[{ly},{uy}] (violation {})",
                z - hi
            );
            assert!(
                lo <= z,
                "lower {lo} > {z} at corner ({x},{y}) box [{lx},{ux}]x[{ly},{uy}] (violation {})",
                lo - z
            );
        }
    }

    /// Many random boxes biased toward the regime that triggers the f32 slope
    /// perturbation: wide magnitude range, mixed sign (so `t_l < 0 < t_u`), and
    /// large coordinate magnitudes that amplify the `(s, 1-s)` rounding error.
    /// Asserts both the strict-corner property and a dense ~1-ULP interior check
    /// for max AND min.
    #[test]
    fn minmax_envelope_random_boxes_tight_ulp() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut mixed_seen = 0u64;
        for _ in 0..200_000 {
            // Scale spans 0.01 .. 1e6 to stress coefficient-rounding error.
            let scale = 10f64.powf(rng.next() * 8.0 - 2.0);
            let lx = ((rng.next() * 2.0 - 1.0) * scale) as f32;
            let ux = lx + (rng.next() * scale) as f32;
            let ly = ((rng.next() * 2.0 - 1.0) * scale) as f32;
            let uy = ly + (rng.next() * scale) as f32;
            if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
                continue;
            }
            // Count how many boxes land in the mixed (relaxed) regime.
            if (lx - uy) < 0.0 && (ux - ly) > 0.0 {
                mixed_seen += 1;
            }

            let max_env = max_envelope(lx, ux, ly, uy).unwrap();
            let min_env = min_envelope(lx, ux, ly, uy).unwrap();

            // Strict (zero-tolerance) corner soundness: this is the property the
            // old 1e-4 tolerance masked.
            assert_corners_strict(max_env, lx, ux, ly, uy, true);
            assert_corners_strict(min_env, lx, ux, ly, uy, false);

            // Dense interior soundness at ~1 ULP.
            assert_envelope_sound(max_env, lx, ux, ly, uy, |x, y| x.max(y));
            assert_envelope_sound(min_env, lx, ux, ly, uy, |x, y| x.min(y));
        }
        // Sanity: the random generator actually exercises the relaxed branch.
        assert!(
            mixed_seen > 1_000,
            "expected many mixed-regime boxes, saw only {mixed_seen}"
        );
    }

    /// Hand-picked box reproducing the audit's worst observed violation regime:
    /// a tiny upper-x endpoint next to a large negative y endpoint, where the
    /// f32 `(s, 1-s)` pair previously pushed the upper plane below the true max
    /// by ~1.5e-5 at the corner. Asserts the fixed plane is sound there.
    #[test]
    fn upper_plane_sound_at_audit_worst_corner() {
        let (lx, ux, ly, uy) = (
            -286.696_08f32,
            -0.003_692_627f32,
            -374.136_05f32,
            -134.931_73f32,
        );
        let env = max_envelope(lx, ux, ly, uy).unwrap();
        assert_corners_strict(env, lx, ux, ly, uy, true);
        // The specific corner the probe flagged.
        let (x, y) = (ux, ly);
        let z = x.max(y) as f64;
        let hi = env.upper.eval(x, y);
        assert!(
            hi >= z,
            "regression: upper {hi} < max {z} at audit corner ({x},{y}) (violation {})",
            z - hi
        );
    }

    #[test]
    fn non_finite_box_returns_none() {
        assert!(max_envelope(f32::NEG_INFINITY, 1.0, 0.0, 1.0).is_none());
        assert!(min_envelope(0.0, 1.0, 0.0, f32::INFINITY).is_none());
        assert!(max_envelope(0.0, f32::NAN, 0.0, 1.0).is_none());
    }

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    /// Drive the full CROWN backward through an identity spec and confirm the
    /// concretized scalar bounds enclose f at every sampled box corner+interior.
    fn assert_driver_sound(a: &BoundedTensor, b: &BoundedTensor, is_max: bool) {
        let n = a.len();
        // Identity incoming spec: each output picks one element directly.
        let ident = LinearBounds::identity(n);
        let (lb_a, lb_b) = if is_max {
            propagate_minmax_linear_binary(&ident, a, b, "MaxBinary", max_envelope).unwrap()
        } else {
            propagate_minmax_linear_binary(&ident, a, b, "MinBinary", min_envelope).unwrap()
        };

        let al = a.lower();
        let au = a.upper();
        let bl = b.lower();
        let bu = b.upper();
        let steps = 25;
        for k in 0..n {
            // bias lives entirely on lb_a; lb_b bias is zero.
            for i in 0..=steps {
                let x = al[k] + (au[k] - al[k]) * (i as f32 / steps as f32);
                for j in 0..=steps {
                    let y = bl[k] + (bu[k] - bl[k]) * (j as f32 / steps as f32);
                    let z = if is_max { x.max(y) } else { x.min(y) };
                    // Reconstruct the affine lower/upper form for output k.
                    let lo =
                        lb_a.lower_a()[[k, k]] * x + lb_b.lower_a()[[k, k]] * y + lb_a.lower_b()[k];
                    let hi =
                        lb_a.upper_a()[[k, k]] * x + lb_b.upper_a()[[k, k]] * y + lb_a.upper_b()[k];
                    let tol = 1e-3 * (1.0 + z.abs());
                    assert!(lo <= z + tol, "driver lower {lo} > {z} at k={k} ({x},{y})");
                    assert!(hi >= z - tol, "driver upper {hi} < {z} at k={k} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn driver_max_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound(&a, &b, true);
    }

    #[test]
    fn driver_min_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound(&a, &b, false);
    }

    /// Exercise the w < 0 plane-selection branch by using a negated incoming
    /// spec (-I). For output k the lower form must bound -f from below and the
    /// upper form must bound -f from above.
    fn assert_driver_sound_negated(a: &BoundedTensor, b: &BoundedTensor, is_max: bool) {
        let n = a.len();
        let neg_ident = {
            let mut la = Array2::<f32>::zeros((n, n));
            let mut ua = Array2::<f32>::zeros((n, n));
            for k in 0..n {
                la[[k, k]] = -1.0;
                ua[[k, k]] = -1.0;
            }
            LinearBounds::new(la, Array1::zeros(n), ua, Array1::zeros(n)).unwrap()
        };
        let (lb_a, lb_b) = if is_max {
            propagate_minmax_linear_binary(&neg_ident, a, b, "MaxBinary", max_envelope).unwrap()
        } else {
            propagate_minmax_linear_binary(&neg_ident, a, b, "MinBinary", min_envelope).unwrap()
        };
        let (al, au, bl, bu) = (a.lower(), a.upper(), b.lower(), b.upper());
        let steps = 25;
        for k in 0..n {
            for i in 0..=steps {
                let x = al[k] + (au[k] - al[k]) * (i as f32 / steps as f32);
                for j in 0..=steps {
                    let y = bl[k] + (bu[k] - bl[k]) * (j as f32 / steps as f32);
                    let f = if is_max { x.max(y) } else { x.min(y) };
                    let z = -f;
                    let lo =
                        lb_a.lower_a()[[k, k]] * x + lb_b.lower_a()[[k, k]] * y + lb_a.lower_b()[k];
                    let hi =
                        lb_a.upper_a()[[k, k]] * x + lb_b.upper_a()[[k, k]] * y + lb_a.upper_b()[k];
                    let tol = 1e-3 * (1.0 + z.abs());
                    assert!(lo <= z + tol, "neg lower {lo} > {z} at k={k} ({x},{y})");
                    assert!(hi >= z - tol, "neg upper {hi} < {z} at k={k} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn driver_max_negated_spec_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound_negated(&a, &b, true);
    }

    #[test]
    fn driver_min_negated_spec_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound_negated(&a, &b, false);
    }
}
