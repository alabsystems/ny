// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound linear relaxation for the binary op `z = atan2(y, x)`.
//!
//! `atan2(y, x)` is the angle of the point `(x, y)`. It is smooth (`C^infty`)
//! everywhere except at the origin and along the negative-`x` axis (the branch
//! cut, where it jumps by `+-2*pi`). Over an axis-aligned input box
//! `[lx, ux] x [ly, uy]` that stays in a single well-conditioned region we
//! build a sound affine lower/upper envelope; otherwise we refuse and the
//! caller keeps the (sound) IBP fallback.
//!
//! # Well-conditioned region (when we attempt a relaxation)
//!
//! We only relax when the box is strictly bounded away from the origin AND
//! strictly avoids the branch cut, so that `atan2` is `C^1` with a *bounded*
//! gradient over the entire (convex) box. Concretely we require **either**
//!
//! * the box lies strictly in one open quadrant
//!   (`lx > 0 || ux < 0`) and (`ly > 0 || uy < 0`), **or**
//! * the box lies strictly in the open right half plane (`lx > 0`); there
//!   `atan2(y, x) = atan(y / x)` is smooth even when `y` spans `0`.
//!
//! In both cases the box never touches the negative-`x` axis nor the origin,
//! so there is no branch cut and `r^2 = x^2 + y^2 >= r_min^2 > 0`.
//!
//! # Derivation (mean-value enclosure)
//!
//! The partials are
//! `fx = d/dx atan2(y, x) = -y / (x^2 + y^2)` and
//! `fy = d/dy atan2(y, x) =  x / (x^2 + y^2)`.
//!
//! Let `B` be the (convex) box, `(x0, y0)` its center, and `[gxl, gxu]`,
//! `[gyl, gyu]` rigorous interval enclosures of `fx`, `fy` over `B`. Pick the
//! plane coefficients `a = mid(gxl, gxu)`, `b = mid(gyl, gyu)` and define
//!
//! `P(x, y) = a * (x - x0) + b * (y - y0) + atan2(y0, x0)`.
//!
//! Because `B` is convex, the segment from `(x0, y0)` to any `(x, y)` in `B`
//! stays in `B`, so by the (vector) mean-value theorem
//!
//! `f(x, y) - P(x, y)
//!    = integral_0^1 [ (fx(p) - a) (x - x0) + (fy(p) - b) (y - y0) ] dt`
//!
//! with `p` on that segment. Hence
//!
//! `|f(x, y) - P(x, y)| <= dx * hx + dy * hy =: pad`,
//!
//! where `dx = (gxu - gxl) / 2 >= |fx - a|`, `hx = (ux - lx) / 2 >= |x - x0|`,
//! and similarly for `y`. The sound envelope is then
//!
//! `P(x, y) - pad <= atan2(y, x) <= P(x, y) + pad`  for all `(x, y)` in `B`.
//!
//! All intermediate quantities are computed in `f64` with the gradient interval
//! widened outward, `atan2(y0, x0)` and `pad` widened outward, and the final
//! f32 plane constants rounded outward via `next_down_f32` / `next_up_f32`, so
//! the stored planes are true bounds at every point of the box.
//!
//! # Tightening: asymmetric constants via box subdivision
//!
//! Once the plane *slope* `(a, b)` is fixed, the tightest sound lower/upper
//! constants are the true minimum and maximum of the residual
//! `g(x, y) = atan2(y, x) - a*x - b*y` over the box. The single mean-value
//! `pad` above is the symmetric, coarse enclosure `g in [g(x0,y0) - pad,
//! g(x0,y0) + pad]`. We tighten it *without weakening soundness* by splitting
//! the box into a `K x K` grid of sub-boxes and taking, over all sub-boxes, the
//! min of `g(center_s) - pad_s` and the max of `g(center_s) + pad_s`, where on
//! each sub-box `pad_s = dx_s * hx_s + dy_s * hy_s` with the *residual* gradient
//! deviations `dx_s = max(|gxl_s - a|, |gxu_s - a|)` and likewise `dy_s`. Each
//! sub-box enclosure is rigorous (same mean-value argument on the convex
//! sub-box), so the union (min of lowers, max of uppers) rigorously encloses
//! `g` over the whole box. The `K = 1` case reproduces the original `pad`, so
//! the subdivided constants are never wider than before; empirically they are
//! ~2.3x tighter on ml4acopf-scale quadrant boxes. Constants are rounded
//! outward at the end so the stored f32 planes remain true bounds.

use ny_tensor::{next_down_f32, next_up_f32};

use super::minmax_relax::{Envelope, Plane};

/// Rigorous interval enclosure `[lo, hi]` of `-y / (x^2 + y^2)` over the box,
/// returned in f64. Requires the box to stay strictly away from the origin so
/// `x^2 + y^2 > 0` everywhere; the caller guarantees this.
///
/// We use crude-but-sound monotone interval arithmetic: `r2 = x^2 + y^2` has a
/// positive interval `[r2_lo, r2_hi]`, the numerator `-y` has interval
/// `[-uy, -ly]`, and `n / d` over a box where `d > 0` attains its extremes at
/// numerator/denominator corner combinations.
fn grad_interval(num_lo: f64, num_hi: f64, r2_lo: f64, r2_hi: f64) -> (f64, f64) {
    debug_assert!(r2_lo > 0.0);
    // n / d with d in [r2_lo, r2_hi] (>0). For fixed n, n/d is monotone in d;
    // the extreme over the rectangle [num_lo,num_hi] x [r2_lo,r2_hi] is at a
    // corner. Enumerate all four.
    let candidates = [
        num_lo / r2_lo,
        num_lo / r2_hi,
        num_hi / r2_lo,
        num_hi / r2_hi,
    ];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    (lo, hi)
}

/// Compute `r^2 = x^2 + y^2` interval `[lo, hi]` over `[lx, ux] x [ly, uy]`.
///
/// `x^2` over `[lx, ux]`: if the interval straddles 0 the min is 0, else the
/// min is the squared endpoint closest to 0; the max is the squared endpoint
/// farthest from 0. Same for `y^2`. Sum the two squared intervals.
fn r2_interval(lx: f64, ux: f64, ly: f64, uy: f64) -> (f64, f64) {
    let sq = |l: f64, u: f64| -> (f64, f64) {
        if l > 0.0 {
            (l * l, u * u)
        } else if u < 0.0 {
            (u * u, l * l)
        } else {
            // straddles 0
            (0.0, (l.abs().max(u.abs())).powi(2))
        }
    };
    let (xl2, xu2) = sq(lx, ux);
    let (yl2, yu2) = sq(ly, uy);
    (xl2 + yl2, xu2 + yu2)
}

/// Number of sub-intervals per axis used by `residual_enclosure`. The box is
/// split into `SUBDIV x SUBDIV` cells; larger values give tighter constants at
/// a quadratic cost. `4` captures the bulk of the gain (~2.3x tighter than the
/// single mean-value pad) while staying cheap (16 cells/element).
const SUBDIV: usize = 4;

/// Rigorous enclosure `[g_lo, g_hi]` of the residual
/// `g(x, y) = atan2(y, x) - a*x - b*y` over the box `[lx, ux] x [ly, uy]`.
///
/// The plane slope `(a, b)` is fixed by the caller; the tightest sound
/// lower/upper plane constants for that slope are exactly `min g` and `max g`.
/// We bound them rigorously by subdividing into a `SUBDIV x SUBDIV` grid: on
/// each sub-box, the (vector) mean-value theorem on the convex sub-box gives
///
/// `g(x, y) in [g(cs) - pad_s, g(cs) + pad_s]`,  `pad_s = dx_s*hx_s + dy_s*hy_s`,
///
/// where `cs` is the sub-box center, `hx_s, hy_s` its half-widths, and
/// `dx_s >= |fx - a|`, `dy_s >= |fy - b|` are the *residual* gradient
/// deviations from the rigorous sub-box gradient intervals. Taking the min of
/// the lower ends and max of the upper ends over all sub-boxes rigorously
/// encloses `g` over the union (the whole box).
///
/// The caller guarantees `r^2 > 0` on the whole box (well conditioned), hence
/// on every sub-box. All arithmetic is f64; the result is widened outward by
/// the caller before the f32 cast. Returns `None` if any sub-box produces a
/// non-finite quantity (caller falls back to IBP).
fn residual_enclosure(lx: f64, ux: f64, ly: f64, uy: f64, a: f64, b: f64) -> Option<(f64, f64)> {
    let mut g_lo = f64::INFINITY;
    let mut g_hi = f64::NEG_INFINITY;
    let kf = SUBDIV as f64;
    for i in 0..SUBDIV {
        let sx0 = lx + (ux - lx) * (i as f64) / kf;
        let sx1 = lx + (ux - lx) * ((i + 1) as f64) / kf;
        for j in 0..SUBDIV {
            let sy0 = ly + (uy - ly) * (j as f64) / kf;
            let sy1 = ly + (uy - ly) * ((j + 1) as f64) / kf;

            let (r2_lo, r2_hi) = r2_interval(sx0, sx1, sy0, sy1);
            // NaN-aware "not (r2_lo > 0)": TRUE for NaN — `r2_lo <= 0.0` would let a
            // NaN r² pass the positivity guard, so the negated form is load-bearing.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(r2_lo > 0.0) || !r2_hi.is_finite() {
                return None;
            }
            // Residual gradient intervals over the sub-box.
            let (gxl, gxu) = grad_interval(-sy1, -sy0, r2_lo, r2_hi);
            let (gyl, gyu) = grad_interval(sx0, sx1, r2_lo, r2_hi);
            if !gxl.is_finite() || !gxu.is_finite() || !gyl.is_finite() || !gyu.is_finite() {
                return None;
            }
            // dx_s >= |fx - a| over the sub-box; the extreme of |fx - a| on the
            // interval [gxl, gxu] is at one of its endpoints.
            let dx = (gxl - a).abs().max((gxu - a).abs());
            let dy = (gyl - b).abs().max((gyu - b).abs());

            // Bit-identical to `0.5 * (s0 + s1)`: sub-box endpoints derived from finite
            // f32-cast bounds stay on f64::midpoint's non-overflow `(a + b) * 0.5` path.
            let xc = f64::midpoint(sx0, sx1);
            let yc = f64::midpoint(sy0, sy1);
            let hx = 0.5 * (sx1 - sx0);
            let hy = 0.5 * (sy1 - sy0);

            let gc = yc.atan2(xc) - a * xc - b * yc;
            if !gc.is_finite() {
                return None;
            }
            let pad = dx * hx + dy * hy;
            if !pad.is_finite() || pad < 0.0 {
                return None;
            }
            // Widen each sub-box bound outward by a tiny relative margin to
            // absorb f64 rounding in the products/sums above.
            let lo_s = gc - pad * (1.0 + 1e-12) - 1e-12;
            let hi_s = gc + pad * (1.0 + 1e-12) + 1e-12;
            if lo_s < g_lo {
                g_lo = lo_s;
            }
            if hi_s > g_hi {
                g_hi = hi_s;
            }
        }
    }
    if !g_lo.is_finite() || !g_hi.is_finite() || g_lo > g_hi {
        return None;
    }
    Some((g_lo, g_hi))
}

/// Returns `true` when the box `[lx, ux] x [ly, uy]` lies in a region where
/// `atan2(y, x)` is `C^1` with a bounded gradient and no branch cut, so a
/// linear relaxation may be attempted. See module docs for the exact predicate.
fn is_well_conditioned(lx: f32, ux: f32, ly: f32, uy: f32) -> bool {
    if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
        return false;
    }
    // Strictly inside the open right half plane: x > 0 everywhere. atan2 is
    // smooth here even when y spans 0 (no origin, no branch cut).
    if lx > 0.0 {
        return true;
    }
    // Strictly inside one open quadrant: x has a single strict sign AND y has a
    // single strict sign. (lx > 0 already handled; here ux < 0 means x < 0.)
    let x_strict = lx > 0.0 || ux < 0.0;
    let y_strict = ly > 0.0 || uy < 0.0;
    x_strict && y_strict
}

/// Sound linear envelope for `z = atan2(y, x)` over `[lx, ux] x [ly, uy]`.
///
/// Note the argument order: `x` is the FIRST tensor argument here (the
/// "denominator"/real part) and `y` the SECOND, matching the
/// `(coeff_x, coeff_y)` plane convention used across `minmax_relax`. The caller
/// is responsible for mapping its `(input_a = y, input_b = x)` operands onto
/// the right axes.
///
/// Returns `None` (caller keeps IBP) when the box is not well conditioned:
/// non-finite, near the origin, or straddling the branch cut.
pub(super) fn atan2_envelope(lx: f32, ux: f32, ly: f32, uy: f32) -> Option<Envelope> {
    if !is_well_conditioned(lx, ux, ly, uy) {
        return None;
    }

    let lxd = lx as f64;
    let uxd = ux as f64;
    let lyd = ly as f64;
    let uyd = uy as f64;

    // r^2 interval over the box. Well-conditioned => strictly positive.
    let (r2_lo, r2_hi) = r2_interval(lxd, uxd, lyd, uyd);
    // NaN-aware "not (r2_lo > 0)": TRUE for NaN — `r2_lo <= 0.0` would let a
    // NaN r² pass the positivity guard, so the negated form is load-bearing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(r2_lo > 0.0) || !r2_hi.is_finite() {
        return None;
    }

    // Gradient intervals (rigorous):
    //   fx = -y / r^2, numerator -y in [-uy, -ly]
    //   fy =  x / r^2, numerator  x in [ lx,  ux]
    let (gxl, gxu) = grad_interval(-uyd, -lyd, r2_lo, r2_hi);
    let (gyl, gyu) = grad_interval(lxd, uxd, r2_lo, r2_hi);
    if !gxl.is_finite() || !gxu.is_finite() || !gyl.is_finite() || !gyu.is_finite() {
        return None;
    }

    // Plane coefficients = interval midpoints (tightest worst-case deviation).
    // Kept verbatim: the gradient ends are only finite-guarded, so `gxl + gxu` may
    // overflow to ±inf and the non-finite `gc` guard below must keep seeing that
    // (f64::midpoint would fabricate a finite center and change which boxes fall
    // back to IBP).
    #[allow(clippy::manual_midpoint)]
    let a = 0.5 * (gxl + gxu);
    #[allow(clippy::manual_midpoint)]
    let b = 0.5 * (gyl + gyu);

    // For the fixed slope (a, b), the tightest sound lower/upper constants are
    // the true min/max of the residual g(x, y) = atan2(y, x) - a*x - b*y over
    // the box. We bound them rigorously and asymmetrically via box subdivision
    // (see `residual_enclosure`); this is the key tightening over a single
    // symmetric mean-value pad. Both ends are already widened outward inside
    // the helper to absorb f64 rounding.
    let (c_lower, c_upper) = residual_enclosure(lxd, uxd, lyd, uyd, a, b)?;

    let lower = Plane {
        coeff_x: a as f32,
        coeff_y: b as f32,
        c: next_down_f32(c_lower as f32),
    };
    let upper = Plane {
        coeff_x: a as f32,
        coeff_y: b as f32,
        c: next_up_f32(c_upper as f32),
    };

    // The f32 cast of the coefficients perturbs the plane slope; that
    // perturbation times the box half-width is already dominated by `pad`
    // only if the cast error is < the gradient interval half-width. To be
    // safe against the coeff cast we additionally widen the constants by the
    // coefficient cast error times the half-widths.
    let a_cast_err = (a - a as f32 as f64).abs();
    let b_cast_err = (b - b as f32 as f64).abs();
    let cast_pad =
        a_cast_err * (lxd.abs().max(uxd.abs())) + b_cast_err * (lyd.abs().max(uyd.abs()));
    let cast_pad = cast_pad * (1.0 + 1e-12) + 1e-12;

    let lower = Plane {
        c: next_down_f32((lower.c as f64 - cast_pad) as f32),
        ..lower
    };
    let upper = Plane {
        c: next_up_f32((upper.c as f64 + cast_pad) as f32),
        ..upper
    };

    Some(Envelope { lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense-sample the box and assert lower(x,y) <= atan2(y,x) <= upper(x,y).
    fn assert_atan2_envelope_sound(env: Envelope, lx: f32, ux: f32, ly: f32, uy: f32) {
        let steps = 60;
        for i in 0..=steps {
            let x = lx + (ux - lx) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let y = ly + (uy - ly) * (j as f32 / steps as f32);
                let z = (y as f64).atan2(x as f64) as f32;
                let lo = env.lower_eval(x, y);
                let hi = env.upper_eval(x, y);
                assert!(
                    lo <= z,
                    "lower {lo} > atan2 {z} at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]"
                );
                assert!(
                    hi >= z,
                    "upper {hi} < atan2 {z} at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]"
                );
            }
        }
    }

    /// Boxes strictly inside a single quadrant (Q1..Q4) plus right-half-plane.
    fn well_conditioned_boxes() -> Vec<(f32, f32, f32, f32)> {
        vec![
            // Q1: x>0, y>0
            (1.0, 4.0, 1.0, 3.0),
            (0.1, 0.5, 2.0, 5.0),
            (3.0, 3.5, 0.2, 0.4),
            // Q2: x<0, y>0
            (-4.0, -2.0, 1.0, 3.0),
            (-0.5, -0.1, 0.5, 2.0),
            // Q3: x<0, y<0
            (-4.0, -2.0, -3.0, -1.0),
            (-1.0, -0.5, -2.0, -0.5),
            // Q4: x>0, y<0
            (2.0, 5.0, -3.0, -1.0),
            (0.5, 1.0, -0.4, -0.1),
            // Right half plane with y spanning 0 (x>0 strictly).
            (1.0, 3.0, -2.0, 2.0),
            (0.5, 4.0, -1.0, 5.0),
            (2.0, 2.5, -0.1, 0.1),
            // Near-degenerate (point-ish) boxes still well away from origin.
            (3.0, 3.0, 4.0, 4.0),
            (1.0, 1.0, -1.0, 1.0),
        ]
    }

    /// Boxes that MUST fall back (origin / branch cut / non-finite).
    fn ill_conditioned_boxes() -> Vec<(f32, f32, f32, f32)> {
        vec![
            // Contains origin.
            (-1.0, 1.0, -1.0, 1.0),
            // Straddles branch cut (negative x axis, y spans 0).
            (-4.0, -2.0, -0.5, 0.5),
            (-4.0, -2.0, -1.0, 0.0), // y_upper = 0 touches cut
            (-4.0, -2.0, 0.0, 1.0),  // y_lower = 0 touches cut
            // x spans 0 with y single-signed but x touches 0 -> ambiguous /
            // near origin column.
            (-1.0, 1.0, 1.0, 2.0),
            (-2.0, 0.0, 1.0, 2.0), // x_upper = 0 touches y-axis (still ok? no: ux<0 false, lx>0 false) -> fallback
            // Non-finite.
            (f32::NEG_INFINITY, 1.0, 1.0, 2.0),
            (1.0, f32::INFINITY, 1.0, 2.0),
            (1.0, 2.0, f32::NAN, 2.0),
        ]
    }

    #[test]
    fn atan2_envelope_is_sound_dense() {
        for (lx, ux, ly, uy) in well_conditioned_boxes() {
            let env = atan2_envelope(lx, ux, ly, uy)
                .unwrap_or_else(|| panic!("expected relaxation for box [{lx},{ux}]x[{ly},{uy}]"));
            assert_atan2_envelope_sound(env, lx, ux, ly, uy);
        }
    }

    #[test]
    fn ill_conditioned_boxes_fall_back() {
        for (lx, ux, ly, uy) in ill_conditioned_boxes() {
            assert!(
                atan2_envelope(lx, ux, ly, uy).is_none(),
                "expected IBP fallback (None) for box [{lx},{ux}]x[{ly},{uy}]"
            );
        }
    }

    #[test]
    fn directed_rounding_is_outward_on_pi_over_4() {
        // atan2(1,1) = pi/4 is not representable in f32; a point box must still
        // strictly enclose it.
        let env = atan2_envelope(1.0, 1.0, 1.0, 1.0).expect("point box in Q1");
        let exact = std::f64::consts::FRAC_PI_4;
        let lo = env.lower_eval(1.0, 1.0) as f64;
        let hi = env.upper_eval(1.0, 1.0) as f64;
        assert!(lo <= exact, "lower {lo} must be <= pi/4 {exact}");
        assert!(hi >= exact, "upper {hi} must be >= pi/4 {exact}");
    }

    #[test]
    fn tight_box_is_much_tighter_than_ibp_full_range() {
        // A small Q1 box should give a far tighter spread than the IBP range,
        // demonstrating the relaxation actually helps ml4acopf-style boxes.
        let env = atan2_envelope(1.9, 2.1, 0.9, 1.1).expect("small Q1 box");
        // Width of the envelope at the box center.
        let lo = env.lower_eval(2.0, 1.0) as f64;
        let hi = env.upper_eval(2.0, 1.0) as f64;
        assert!(hi - lo < 0.1, "envelope spread {} too wide", hi - lo);
    }

    /// Deterministic LCG so the dense soundness sweep is reproducible without an
    /// `rand` dependency in this crate's tests.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 53 bits -> [0, 1).
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// Generate a strictly-well-conditioned box for the requested region. All
    /// returned boxes satisfy `is_well_conditioned` so `atan2_envelope` must
    /// return `Some`.
    fn random_box(rng: &mut Lcg, region: u8) -> (f32, f32, f32, f32) {
        let pos = |rng: &mut Lcg| 0.1 + 2.9 * rng.next_f64();
        let neg = |rng: &mut Lcg| -(0.1 + 2.9 * rng.next_f64());
        let w = |rng: &mut Lcg| 0.01 + 1.5 * rng.next_f64();
        let (lx, ux, ly, uy): (f64, f64, f64, f64) = match region {
            0 => {
                // Q1: x>0, y>0
                let lx = pos(rng);
                let ly = pos(rng);
                (lx, lx + w(rng), ly, ly + w(rng))
            }
            1 => {
                // Q2: x<0, y>0
                let ux = neg(rng);
                let ly = pos(rng);
                (ux - w(rng), ux, ly, ly + w(rng))
            }
            2 => {
                // Q3: x<0, y<0
                let ux = neg(rng);
                let uy = neg(rng);
                (ux - w(rng), ux, uy - w(rng), uy)
            }
            3 => {
                // Q4: x>0, y<0
                let lx = pos(rng);
                let uy = neg(rng);
                (lx, lx + w(rng), uy - w(rng), uy)
            }
            _ => {
                // Right half plane with y spanning 0 (x>0 strictly).
                let lx = pos(rng);
                let ly = -(0.05 + 2.0 * rng.next_f64());
                let uy = ly + (0.1 + 2.0 * rng.next_f64());
                (lx, lx + w(rng), ly, uy)
            }
        };
        (lx as f32, ux as f32, ly as f32, uy as f32)
    }

    /// Thousands of random boxes per region, thousands of interior samples each:
    /// the asymmetric subdivided envelope MUST sandwich atan2 everywhere.
    #[test]
    fn atan2_envelope_dense_random_soundness() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for region in 0u8..=4 {
            for _ in 0..400 {
                let (lx, ux, ly, uy) = random_box(&mut rng, region);
                let env = match atan2_envelope(lx, ux, ly, uy) {
                    Some(e) => e,
                    // Degenerate near-point box where r^2 underflows; skip.
                    None => continue,
                };
                // Dense interior + boundary grid (41x41 = 1681 points/box).
                let steps = 40;
                for i in 0..=steps {
                    let x = lx + (ux - lx) * (i as f32 / steps as f32);
                    for j in 0..=steps {
                        let y = ly + (uy - ly) * (j as f32 / steps as f32);
                        let z = (y as f64).atan2(x as f64) as f32;
                        let lo = env.lower_eval(x, y);
                        let hi = env.upper_eval(x, y);
                        assert!(
                            lo <= z,
                            "region {region}: lower {lo} > atan2 {z} at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]"
                        );
                        assert!(
                            hi >= z,
                            "region {region}: upper {hi} < atan2 {z} at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]"
                        );
                    }
                }
            }
        }
    }

    /// The single-mean-value symmetric pad (the pre-tightening construction):
    /// reproduces the *old* envelope width so we can prove the subdivided
    /// constants are never wider and are usually meaningfully tighter.
    fn old_pad_width(lx: f32, ux: f32, ly: f32, uy: f32) -> f64 {
        let (lxd, uxd, lyd, uyd) = (lx as f64, ux as f64, ly as f64, uy as f64);
        let (r2_lo, r2_hi) = r2_interval(lxd, uxd, lyd, uyd);
        let (gxl, gxu) = grad_interval(-uyd, -lyd, r2_lo, r2_hi);
        let (gyl, gyu) = grad_interval(lxd, uxd, r2_lo, r2_hi);
        let hx = 0.5 * (uxd - lxd);
        let hy = 0.5 * (uyd - lyd);
        let dx = 0.5 * (gxu - gxl);
        let dy = 0.5 * (gyu - gyl);
        // Old envelope width was 2*pad.
        2.0 * (dx * hx + dy * hy)
    }

    /// Width of the actual (subdivided) envelope: it is slope-shared, so the
    /// constant gap `c_upper - c_lower` is the spread at every point.
    fn new_env_width(env: &Envelope) -> f64 {
        (env.upper.c as f64) - (env.lower.c as f64)
    }

    /// Regression: the subdivided constants must NEVER widen the envelope versus
    /// the old symmetric pad, and on representative quadrant boxes must be
    /// substantially tighter (the whole point of the change).
    #[test]
    fn subdivision_never_wider_and_usually_tighter() {
        let mut rng = Lcg(0xfeed_face_dead_beef);
        let mut sum_ratio = 0.0f64;
        let mut count = 0usize;
        // A tiny slack covers the extra outward widening the subdivided path
        // applies per sub-box (relative 1e-12 + 1e-12 absolute, plus f32 ULP).
        let slack = 1e-5;
        for region in 0u8..=4 {
            for _ in 0..300 {
                let (lx, ux, ly, uy) = random_box(&mut rng, region);
                let env = match atan2_envelope(lx, ux, ly, uy) {
                    Some(e) => e,
                    None => continue,
                };
                let old_w = old_pad_width(lx, ux, ly, uy);
                let new_w = new_env_width(&env);
                assert!(
                    new_w <= old_w + slack + 1e-6 * old_w,
                    "subdivided width {new_w} wider than old pad {old_w} on box [{lx},{ux}]x[{ly},{uy}]"
                );
                if old_w > 1e-6 {
                    sum_ratio += new_w / old_w;
                    count += 1;
                }
            }
        }
        let mean_ratio = sum_ratio / (count as f64);
        // Empirically ~0.4; assert a comfortable < 0.7 to lock in the tightening.
        assert!(
            mean_ratio < 0.7,
            "mean subdivided/old width ratio {mean_ratio} not tight enough (count {count})"
        );
    }
}
