// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{f32_affine_eval_error, f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

use super::LinearRelaxation;

/// Log layer: y = ln(x)
///
/// Element-wise natural logarithm. Requires x > 0.
/// Common in log-softmax and various loss functions.
#[derive(Debug, Clone, Default)]
pub struct LogLayer;

impl LogLayer {
    /// Create a new Log layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for LogLayer {
    /// IBP for Log: y = ln(x)
    ///
    /// Log is monotonically increasing for x > 0, so bounds are straightforward.
    /// Requires strictly positive input bounds for sound results.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Validate input is strictly positive to ensure sound bounds.
        // If input bounds allow non-positive values, log is undefined.
        // NaN-propagating fold: NaN in input must not be silently absorbed — see #2577.
        // NaN.is_nan() || NaN <= 0.0 is false, so we also check explicitly.
        let min_input = input
            .lower()
            .iter()
            .copied()
            .fold(f32::INFINITY, nan_propagating_min);
        if min_input.is_nan() || min_input <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "LogLayer requires strictly positive input, got minimum bound {}",
                min_input
            )));
        }

        // Directed rounding: compute ln in f64 for precision, cast to f32 with
        // next_down/next_up to guarantee lower bounds round DOWN and upper bounds
        // round UP. Raw f32 ln can round either direction. (#1483)
        let lower = input
            .lower()
            .mapv(|x| next_down_f32((x as f64).ln() as f32));
        let upper = input.upper().mapv(|x| next_up_f32((x as f64).ln() as f32));
        BoundedTensor::new(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        LogLayer,
        NyError::UnsupportedOp(
            "Log is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string()
        )
    );
}

/// Linear relaxation for log on interval [l, u].
///
/// log(x) is concave and monotonically increasing (for x > 0).
/// For concave functions: chord is lower bound, tangent is upper bound.
///
/// **Lower bound (chord/secant through endpoints):**
///   slope_l = (log(u) - log(l)) / (u - l)
///   intercept_l = log(l) - slope_l * l
///
/// **Upper bound (tangent at midpoint m):**
///   m = (l + u) / 2
///   slope_u = 1/m                    [derivative of log at m]
///   intercept_u = log(m) - 1         [tangent: y = (1/m)*x + log(m) - 1]
///
/// Reference: alpha-beta-CROWN `BoundLog.bound_relax` in
/// `auto_LiRPA/operators/convex_concave.py:36-45`
///
/// # Precondition
/// Caller must ensure l > 0 and u > 0. This function is only called from
/// CROWN backward propagation where pre-activation bounds should already
/// be validated by `propagate_ibp`.
pub fn log_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard: NaN bounds → return (-inf, +inf) intercepts so CROWN drives bounds to ±inf.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Guard: infinite bounds cause inf/inf = NaN in chord computation.
    // log(x) is only defined for x > 0. log(x) -> -inf as x -> 0+, log(x) -> +inf as x -> +inf.
    // l should always be > 0 (validated by IBP), but guard for robustness.
    if u.is_infinite() {
        if u.is_sign_negative() {
            return LinearRelaxation::nan_fallback();
        }
        // u = +inf: chord slope (log(u) - log(l))/(u - l) -> 0 as u -> inf.
        // Lower: slope 0, intercept = log(l) (constant, sound since log is increasing).
        // Upper: tangent at l, y = (1/l)x + log(l) - 1 (sound since log is concave).
        if l <= 0.0 || !l.is_finite() {
            return LinearRelaxation::nan_fallback();
        }
        let log_l = (l as f64).ln();
        let upper_slope = 1.0 / (l as f64);
        let upper_intercept = log_l - 1.0;
        // On an unbounded positive interval, a finite intercept cannot compensate
        // a downward-rounded slope for every x.  Round the upper slope upward
        // instead.  If it is not representable, use the safe constant +infinity
        // upper bound rather than introducing an infinite coefficient.
        let upper_slope_f32 = next_up_f32(upper_slope as f32);
        if !upper_slope_f32.is_finite() {
            return LinearRelaxation::constant(next_down_f32(log_l as f32), f32::INFINITY);
        }
        return LinearRelaxation::new(
            0.0,
            next_down_f32(log_l as f32),
            upper_slope_f32,
            next_up_f32(upper_intercept as f32),
        );
    }
    // l = -inf or l = +inf shouldn't occur for log (domain x > 0), but handle gracefully.
    if l.is_infinite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // DOMAIN GUARD, not a clamp (#log-epsilon-nonenclosing).
    //
    // This used to be `l = max(l, 1e-10); u = max(u, 1e-10)`, described as a
    // "safety clamp". Raising `l` MOVES THE DOMAIN: for a true `l` below 1e-10 the
    // chord/tangent below are built for [1e-10, u] while the input can reach the
    // real `l`, and since ln(1e-10) = -23.03 while ln(1e-20) = -46.05, the lower
    // line sits ABOVE ln on (l, 1e-10). That is a NON-ENCLOSING lower relaxation,
    // i.e. a false-`unsat` generator — the one direction that costs -150.
    //
    // MEASURED by `envelope_audit_expologpow::audit_log_envelope`: 10,484,626 of
    // 37,861,389 sampled points violate the lower line, worst violation 6.447238e1
    // (8,450,523 f32 ULPs) at l = u = 1e-38; the upper line is clean. Excluding
    // l < 1e-10 leaves 0 of 23,096,113 violating — every violation is inside the
    // clamped region, which is exactly what a domain shift predicts.
    //
    // ln is computed in f64 here and is finite for every positive f32 including
    // subnormals (ln(1e-45) = -103.6), so no clamp is needed to keep it finite.
    // Non-positive endpoints and inverted intervals violate this helper's
    // precondition. Fail open there rather than pretending the domain starts
    // higher than it does.
    //
    // Blast radius when this was found: zero scored rows. No ONNX file in
    // vnncomp2025 or vnncomp2026 contains a Log or LogSoftmax op (0 of 2372
    // checked), so no banked verdict depended on the broken envelope.
    if l <= 0.0 || u <= 0.0 || l > u {
        return LinearRelaxation::nan_fallback();
    }

    // Tiny `l`: use the ENDPOINT-CONSTANT BAND, not a sloped line.
    //
    // Below EPSILON the honest chord is astronomically steep — on [1e-12, 2e-12]
    // its slope is ln(2)/1e-12 ~ 6.9e11 — because that is genuinely how fast ln
    // moves there. Such a line is mathematically sound but numerically explosive:
    // every downstream certified error term scales with |coeff|, so shipping a
    // 6.9e11 coefficient trades a non-enclosing bound for an error blow-up. The
    // old code avoided that by clamping the DOMAIN, which is unsound (see above).
    //
    // The constant band is sound for any 0 < l <= u and has slope 0: ln is
    // increasing, so ln(l) <= ln(x) <= ln(u) for every x in [l, u], and both ends
    // are rounded outward. Same shape the narrow-interval path below uses, and the
    // same shape sqrt.rs uses. It is looser than the chord, which costs nothing
    // here — no ONNX file in vnncomp2025 or vnncomp2026 contains a Log or
    // LogSoftmax op (0 of 2372), so no scored row depends on tightness at 1e-12.
    const EPSILON: f32 = 1e-10;
    if l < EPSILON {
        return LinearRelaxation::new(
            0.0,
            next_down_f32(f64::from(l).ln() as f32),
            0.0,
            next_up_f32(f64::from(u).ln() as f32),
        );
    }

    // Use f64 intermediates to prevent catastrophic cancellation.
    // The chord slope (log(u) - log(l)) / (u - l) is a classic cancellation
    // case when l and u are close: both numerator and denominator approach 0.
    // Same pattern as Exp fix (#1745).
    let l64 = l as f64;
    let u64 = u as f64;

    // Handle narrow intervals with a guard RELATIVE to l (#drift-log-narrow-tiny).
    //
    // The former ABSOLUTE guard (|u - l| < 1e-8) also captured intervals at
    // tiny magnitude and returned the tangent at l for BOTH lines. A tangent
    // lies ABOVE concave ln, so it is never a valid LOWER line: at x = u it
    // overshoots ln(u) by r - ln(1 + r) with r = (u - l)/l — unbounded as
    // l -> 0 (ln's curvature-to-slope ratio |ln''|/ln' = 1/l blows up).
    // Observed: +45.09 on [1e-10, 5e-9] (r ~ 49), +1.1e-5 on
    // [1e-6, 1e-6 + 5e-9].
    //
    // With r < 1e-7 the output range ln(u) - ln(l) <= r is below f32
    // resolution, so the endpoint-constant band
    // [ln(l) rounded down, ln(u) rounded up] (same shape as sqrt.rs's narrow
    // band) is sound for any l <= u — ln is increasing and the rounding is
    // outward — and within ~1e-7 of tight. Wider intervals take the
    // chord/tangent path below, whose chord slope ln(u/l)/(u - l) stays
    // well-conditioned once u - l >= l * 1e-7.
    const NARROW_REL: f64 = 1e-7;
    if u64 - l64 < l64 * NARROW_REL {
        return LinearRelaxation::new(
            0.0,
            next_down_f32(l64.ln() as f32),
            0.0,
            next_up_f32(u64.ln() as f32),
        );
    }

    let log_l = l64.ln();
    let log_u = u64.ln();

    // Lower bound: chord (secant) through (l, log(l)) and (u, log(u)).
    // For concave log, the chord lies below the function.
    let lower_slope = (log_u - log_l) / (u64 - l64);
    let lower_intercept = log_l - lower_slope * l64;

    // Upper bound: tangent line at the parallel-to-chord point.
    // The tangent to a concave function at ANY point d in (l,u) lies above the
    // function (global upper bound). The single TIGHTEST tangent upper line is
    // the one parallel to the chord: its slope equals chord_slope, and the
    // tangent point is where log'(d) = 1/d = chord_slope, i.e. d = 1/chord_slope.
    //
    // Soundness of d in (l,u): by the mean-value theorem chord_slope = log'(c)
    // for some c in (l,u); since log' is strictly monotone (log'' = -1/x^2 < 0),
    // d = 1/chord_slope = c lies in (l,u). The tangent at d is therefore a valid,
    // tighter upper bound than the midpoint tangent.
    //
    // chord_slope here is exactly `lower_slope` computed above for the chord.
    let chord_slope = lower_slope;
    // Guard: only take the tighter tangent when l > 0 (true here after clamp),
    // u finite (true: u.is_infinite() handled earlier), chord_slope finite and
    // strictly positive, and d = 1/chord_slope lies within [l, u]. Otherwise
    // fall back to the midpoint tangent, which is still a sound upper bound.
    let (upper_slope, upper_intercept) = {
        let d = 1.0 / chord_slope;
        if l64 > 0.0 && chord_slope.is_finite() && chord_slope > 0.0 && d >= l64 && d <= u64 {
            // Parallel-to-chord tangent at d: slope = 1/d = chord_slope,
            // intercept = log(d) - 1.
            (chord_slope, d.ln() - 1.0)
        } else {
            // Fallback: tangent at the midpoint m (sound for concave log).
            let m = f64::midpoint(l64, u64);
            (1.0 / m, m.ln() - 1.0)
        }
    };

    // Directed rounding: compensate for f64→f32 truncation in both slope and
    // intercept, matching the pattern in exp_linear_relaxation.
    let max_abs_x = nan_propagating_max(l.abs(), u.abs());
    let lower_slope_f32 = lower_slope as f32;
    let upper_slope_f32 = upper_slope as f32;
    let lower_eval_err =
        f32_affine_eval_error(lower_slope, lower_slope_f32, lower_intercept, max_abs_x);
    let upper_eval_err =
        f32_affine_eval_error(upper_slope, upper_slope_f32, upper_intercept, max_abs_x);
    let lower_intercept_f32 = next_down_f32(f64_to_f32_down(lower_intercept - lower_eval_err));
    let upper_intercept_f32 = next_up_f32(f64_to_f32_up(upper_intercept + upper_eval_err));

    // Extreme positive f32s can require an affine slope or intercept outside
    // f32.  A monotonic endpoint band is less precise but always represents a
    // sound relaxation, and avoids `inf * 0` in downstream concretization.
    if !lower_slope_f32.is_finite()
        || !upper_slope_f32.is_finite()
        || !lower_intercept_f32.is_finite()
        || !upper_intercept_f32.is_finite()
    {
        return LinearRelaxation::constant(next_down_f32(log_l as f32), next_up_f32(log_u as f32));
    }

    LinearRelaxation::new(
        lower_slope_f32,
        lower_intercept_f32,
        upper_slope_f32,
        upper_intercept_f32,
    )
}

/// Domain guard for Log CROWN backward propagation.
///
/// Validates that pre-activation bounds are strictly positive and finite before
/// computing the Log CROWN linear relaxation. Without this guard, non-positive
/// lower bounds get silently clamped to epsilon (line ~110), producing bounds
/// that are NOT sound relative to the original pre-activation interval.
///
/// Analogous to `exp_crown_domain_guard` in `exp.rs:160`.
///
/// Reference: alpha-beta-CROWN `BoundLog.__init__` sets `self.range_l = 1e-6`
/// (`auto_LiRPA/operators/convex_concave.py:28`), which clamps the lower bound
/// in the base class. Our guard rejects instead of clamping, returning
/// `NumericalInstability` so the caller falls back to IBP.
fn log_crown_domain_guard(pre_activation: &BoundedTensor) -> Result<()> {
    // Guard: reject non-positive lower bounds. log(x) is undefined for x <= 0.
    // Clamping to epsilon would produce a relaxation that doesn't contain all
    // possible outputs for the original pre-activation range.
    if pre_activation.lower().iter().any(|&x| x <= 0.0) {
        let min_lower = pre_activation
            .lower()
            .iter()
            .cloned()
            .fold(f32::INFINITY, nan_propagating_min);
        return Err(NyError::NumericalInstability(format!(
            "Log CROWN: non-positive pre-activation lower bound {:.6e}",
            min_lower
        )));
    }

    // Guard: reject non-finite bounds (NaN or ±Inf)
    if pre_activation.lower().iter().any(|x| !x.is_finite())
        || pre_activation.upper().iter().any(|x| !x.is_finite())
    {
        return Err(NyError::NumericalInstability(
            "Log CROWN: non-finite pre-activation bounds".to_string(),
        ));
    }

    Ok(())
}

impl LogLayer {
    impl_elementwise_activation!(
        @inherent_methods
        LogLayer,
        log_linear_relaxation,
        domain_guard: log_crown_domain_guard
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;

    /// Verify that the linear relaxation forms a valid envelope around log(x)
    /// for all x in [l, u]: lower_slope * x + lower_intercept <= log(x) <= upper_slope * x + upper_intercept
    fn assert_log_envelope(l: f32, u: f32) {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = log_linear_relaxation(l, u);
        let samples = if (u - l).abs() < 1e-8 { 1 } else { 100 };
        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let x = l + (u - l) * t;
            let fx = x.ln();
            let lower = ls * x + li;
            let upper = us * x + ui;
            let tol = 1e-4 * fx.abs().max(1.0);
            assert!(
                lower <= fx + tol,
                "log lower envelope violated for [{l}, {u}] at x={x}: lower={lower} > log(x)={fx}"
            );
            assert!(
                upper + tol >= fx,
                "log upper envelope violated for [{l}, {u}] at x={x}: upper={upper} < log(x)={fx}"
            );
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_envelope_basic() {
        let intervals = [
            (1.0, 1.0),    // point
            (0.1, 1.0),    // includes log(x) < 0
            (1.0, 10.0),   // positive log range
            (0.01, 100.0), // wide range
            (0.5, 2.0),    // moderate range crossing log(1)=0
            (1e-3, 1e-1),  // small positive values
            (2.0, 3.0),    // narrow positive
            (0.1, 0.2),    // narrow near zero
        ];
        for (l, u) in intervals {
            assert_log_envelope(l, u);
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_degenerate_point() {
        // Point interval: both bounds should be tight
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = log_linear_relaxation(2.0, 2.0);
        let log_2 = 2.0_f32.ln();
        assert!((ls * 2.0 + li - log_2).abs() < 1e-5);
        assert!((us * 2.0 + ui - log_2).abs() < 1e-5);
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_backward_soundness() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        let l = 0.5_f32;
        let u = 5.0_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.ln();
            assert!(
                la * x + lb <= y + 1e-3,
                "Log CROWN lb violated at x={x}: {} > {y}",
                la * x + lb
            );
            assert!(
                ua * x + ub >= y - 1e-3,
                "Log CROWN ub violated at x={x}: {} < {y}",
                ua * x + ub
            );
        }
    }

    #[test]
    fn test_crown_backward_narrow_interval() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        let l = 1.0_f32;
        let u = 2.0_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.ln();
            assert!(
                result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-4,
                "narrow log lb violated at x={x}"
            );
            assert!(
                result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-4,
                "narrow log ub violated at x={x}"
            );
        }
    }

    #[test]
    fn test_crown_backward_small_positive() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        let l = 0.01_f32;
        let u = 0.1_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.ln();
            assert!(
                result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-2,
                "small positive log lb violated at x={x}"
            );
            assert!(
                result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-2,
                "small positive log ub violated at x={x}"
            );
        }
    }

    #[test]
    fn test_crown_backward_multi_neuron() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        let pre = BoundedTensor::new(
            arr1(&[0.1_f32, 1.0]).into_dyn(),
            arr1(&[1.0_f32, 10.0]).into_dyn(),
        )
        .unwrap();
        let bounds = LinearBounds::identity(2);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for neuron in 0..2 {
            let la = result.lower_a[[neuron, neuron]];
            let lb = result.lower_b[neuron];
            let ua = result.upper_a[[neuron, neuron]];
            let ub = result.upper_b[neuron];
            let lo = pre.lower()[neuron];
            let hi = pre.upper()[neuron];

            for k in 0..=20 {
                let x = lo + (hi - lo) * (k as f32 / 20.0);
                let y = x.ln();
                assert!(
                    la * x + lb <= y + 1e-2,
                    "neuron {neuron} lb violated at x={x}"
                );
                assert!(
                    ua * x + ub >= y - 1e-2,
                    "neuron {neuron} ub violated at x={x}"
                );
            }
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = LogLayer::new();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear(&bounds).is_err(),
            "Log CROWN without pre-activation bounds should fail"
        );
        assert!(layer.requires_pre_activation_bounds());
    }

    // ── Domain guard tests (#2565) ──────────────────────────────────────

    #[test]
    fn test_crown_backward_rejects_non_positive_lower() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        // l = -1.0 is non-positive → domain guard should reject
        let pre =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre);
        assert!(
            result.is_err(),
            "Log CROWN should reject non-positive lower bound"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-positive"),
            "Error should mention non-positive: {err}"
        );
    }

    #[test]
    fn test_crown_backward_rejects_zero_lower() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        // l = 0.0 is non-positive → domain guard should reject
        let pre =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear_with_bounds(&bounds, &pre).is_err(),
            "Log CROWN should reject zero lower bound"
        );
    }

    #[test]
    fn test_crown_backward_rejects_nan_preactivation() {
        use ndarray::{ArrayD, IxDyn};
        let layer = LogLayer::new();
        let l = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
        let pre = BoundedTensor::new_unchecked(l, u).unwrap();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear_with_bounds(&bounds, &pre).is_err(),
            "Log CROWN should reject NaN pre-activation bounds"
        );
    }

    #[test]
    fn test_crown_backward_rejects_inf_preactivation() {
        use ndarray::{ArrayD, IxDyn};
        let layer = LogLayer::new();
        // BoundedTensor::new rejects Inf, so use new_unchecked
        let l = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0_f32]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap();
        let pre = BoundedTensor::new_unchecked(l, u).unwrap();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear_with_bounds(&bounds, &pre).is_err(),
            "Log CROWN should reject infinite pre-activation bounds"
        );
    }

    #[test]
    fn test_crown_backward_accepts_positive_bounds() {
        use ndarray::arr1;
        let layer = LogLayer::new();
        // l = 0.5, u = 5.0 are strictly positive → should succeed
        let pre =
            BoundedTensor::new(arr1(&[0.5_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear_with_bounds(&bounds, &pre).is_ok(),
            "Log CROWN should accept strictly positive bounds"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_chord_below_tangent_above() {
        // For a non-degenerate interval, lower (chord) should be below
        // and upper (tangent) should be above at the midpoint.
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = log_linear_relaxation(1.0, 3.0);
        let m = f32::midpoint(1.0_f32, 3.0_f32);
        let fx = m.ln();
        let lower_at_m = ls * m + li;
        let upper_at_m = us * m + ui;
        // Chord should be below at midpoint (concave function)
        assert!(lower_at_m <= fx + 1e-4, "chord should be below at midpoint");
        // Upper (tangent) must lie above the function at the midpoint.
        assert!(
            upper_at_m + 1e-4 >= fx,
            "tangent should be above at midpoint"
        );
    }

    /// The parallel-to-chord tangent upper bound is strictly tighter than the
    /// old midpoint tangent on a wide interval: its maximum gap to ln(x) over
    /// [l, u] is strictly smaller. Also re-asserts the new upper is a sound
    /// global over-approximation (gap >= 0 at every sample, within rounding).
    #[ntest::timeout(5000)]
    #[test]
    fn log_parallel_chord_tangent_tighter_than_midpoint() {
        // Wide interval so the two tangent points differ noticeably.
        let l = 0.5_f64;
        let u = 50.0_f64;

        // New upper bound: parallel-to-chord tangent (what the code now uses).
        let r = log_linear_relaxation(l as f32, u as f32);
        let new_us = r.upper_slope as f64;
        let new_ui = r.upper_intercept as f64;

        // Old upper bound: midpoint tangent (the previous implementation).
        let m = f64::midpoint(l, u);
        let old_us = 1.0 / m;
        let old_ui = m.ln() - 1.0;

        let mut new_max_gap = f64::NEG_INFINITY;
        let mut old_max_gap = f64::NEG_INFINITY;
        let mut min_new_gap = f64::INFINITY;
        let samples = 2000;
        for i in 0..=samples {
            let x = l + (u - l) * (i as f64 / samples as f64);
            let fx = x.ln();
            let new_gap = (new_us * x + new_ui) - fx;
            let old_gap = (old_us * x + old_ui) - fx;
            new_max_gap = new_max_gap.max(new_gap);
            old_max_gap = old_max_gap.max(old_gap);
            min_new_gap = min_new_gap.min(new_gap);
        }

        // Soundness re-check: new upper never dips below ln (within f32 rounding).
        assert!(
            min_new_gap >= -1e-4,
            "new upper bound dipped below ln: min gap = {min_new_gap}"
        );
        // Tightness: the new (parallel-to-chord) upper has a strictly smaller
        // maximum gap to ln than the old midpoint tangent on this wide interval.
        assert!(
            new_max_gap < old_max_gap,
            "parallel-to-chord upper not tighter: new_max_gap={new_max_gap} \
             >= old_max_gap={old_max_gap}"
        );
    }

    // ── Narrow-interval regression tests (#drift-log-narrow-tiny) ──────
    //
    // The old ABSOLUTE narrow guard (|u - l| < 1e-8) returned the tangent at
    // l for BOTH lines; a tangent lies ABOVE concave ln, so the certified
    // LOWER line overshot ln(u) by r - ln(1 + r), r = (u - l)/l — up to
    // +45.09 on [1e-10, 5e-9]. These tests evaluate the certified f32
    // coefficients in f64 against true ln on a dense grid over the clamped
    // interval, with only a tiny relative slack for the f64 reference's own
    // evaluation noise.

    /// Strict f64 envelope check with the certified f32 coefficients over the
    /// full positive-f32 domain accepted by Log.
    fn assert_log_envelope_strict(l: f32, u: f32) {
        let r = log_linear_relaxation(l, u);
        // Sample the box the relaxation was BUILT FOR, i.e. [l, u].
        //
        // This used to sample `[l.max(1e-10), u.max(1e-10)]`, mirroring the
        // production `l = max(l, 1e-10)` domain clamp. That made the helper agree
        // with the defect instead of catching it: a caller passing [1e-12, 2e-12]
        // received a relaxation valid only on [1e-10, 1e-10], leaving all of
        // [1e-12, 1e-10) unguarded — and this helper never looked there, because it
        // applied the same shift. The independent
        // `envelope_audit_expologpow::audit_log_envelope`, which samples the true
        // box, caught it: 10,484,626 of 37,861,389 points violated the lower line,
        // and excluding l < 1e-10 left 0 of 23,096,113.
        //
        // The clamp is gone from production (#log-epsilon-nonenclosing), so the
        // helper must test the real contract: for all x in [l, u],
        // lower(x) <= ln(x) <= upper(x).
        let lc = f64::from(l);
        let uc = f64::from(u);
        let (ls, li) = (r.lower_slope as f64, r.lower_intercept as f64);
        let (us, ui) = (r.upper_slope as f64, r.upper_intercept as f64);
        let n = 256;
        for i in 0..=n {
            let x = lc + (uc - lc) * (i as f64) / (n as f64);
            let fx = x.ln();
            let slack = 1e-12 * (1.0 + fx.abs());
            let lower = ls * x + li;
            let upper = us * x + ui;
            assert!(
                lower <= fx + slack,
                "log LOWER line above ln on [{l:e}, {u:e}] at x={x:e}: \
                 line={lower:e} ln={fx:e} relax={r:?}"
            );
            assert!(
                upper >= fx - slack,
                "log UPPER line below ln on [{l:e}, {u:e}] at x={x:e}: \
                 line={upper:e} ln={fx:e} relax={r:?}"
            );
            assert!(
                lower <= upper + slack,
                "log lines cross on [{l:e}, {u:e}] at x={x:e}: relax={r:?}"
            );
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_narrow_tiny_magnitude_sound() {
        // Historical unsoundness: on [1e-10, 5e-9] the old narrow path's
        // LOWER line evaluated to +25.974148 at x = u vs ln(u) = -19.113828
        // (unsound by 45.09).
        assert_log_envelope_strict(1e-10, 5e-9);
        let r = log_linear_relaxation(1e-10, 5e-9);
        let at_u = (r.lower_slope as f64) * 5e-9 + (r.lower_intercept as f64);
        let ln_u = 5e-9_f64.ln();
        assert!(
            at_u <= ln_u,
            "lower line at u must not exceed ln(u): {at_u} > {ln_u}"
        );

        // Unsound by 1.1e-5 under the old absolute guard.
        let u = 1e-6_f32 + 5e-9_f32;
        assert_log_envelope_strict(1e-6, u);

        // Below EPSILON: NOT clamped any more (#log-epsilon-nonenclosing). The
        // relaxation must now be sound on the TRUE box, which these assert.
        // Previously l = 1e-12 was silently raised to 1e-10, leaving
        // [1e-12, 1e-10) outside anything the lines guarded.
        assert_log_envelope_strict(1e-12, 5e-9);
        assert_log_envelope_strict(1e-12, 2e-12);
        // Subnormal and near-zero lower bounds: the constant band must still
        // enclose, with no clamp and no astronomically steep chord.
        assert_log_envelope_strict(1e-38, 1e-30);
        assert_log_envelope_strict(f32::MIN_POSITIVE, 1e-20);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_point_interval_constant_band() {
        // l == u exactly: constant band [ln(l) rounded down, ln(l) rounded up].
        for x in [1e-10_f32, 5e-9, 1e-6, 0.5, 1.0, 2.0, 1e8] {
            let r = log_linear_relaxation(x, x);
            assert_eq!(r.lower_slope, 0.0, "point interval lower slope at x={x:e}");
            assert_eq!(r.upper_slope, 0.0, "point interval upper slope at x={x:e}");
            let fx = (x as f64).ln();
            assert!(
                (r.lower_intercept as f64) <= fx,
                "point band lower above ln at x={x:e}: {} > {fx}",
                r.lower_intercept
            );
            assert!(
                (r.upper_intercept as f64) >= fx,
                "point band upper below ln at x={x:e}: {} < {fx}",
                r.upper_intercept
            );
            // Outward rounding costs at most ~2 f32 ulps of ln(x).
            let width = (r.upper_intercept - r.lower_intercept) as f64;
            assert!(
                width <= 4.0 * (f32::EPSILON as f64) * (1.0 + fx.abs()),
                "point band too wide at x={x:e}: {width}"
            );
        }
    }

    /// Broad soundness sweep over a log-spaced grid of (l, width) pairs:
    /// subnormal-adjacent magnitudes,
    /// relative widths straddling the 1e-7 narrow threshold, absolute widths
    /// straddling the old 1e-8 guard, and 1-ulp intervals in every binade.
    #[ntest::timeout(5000)]
    #[test]
    fn log_relaxation_narrow_sweep_sound() {
        let anchors: [f32; 14] = [
            f32::from_bits(1),           // smallest positive subnormal
            f32::from_bits(0x0040_0000), // mid subnormal
            f32::MIN_POSITIVE,
            1e-38,
            1e-30,
            1e-12,
            1e-10,
            1e-8,
            1e-6,
            1e-3,
            1.0,
            2.0,
            1e4,
            1e8,
        ];
        let rel_widths: [f64; 8] = [0.0, 1e-9, 5e-8, 1e-7, 2e-7, 1e-3, 5e-2, 49.0];
        let abs_widths: [f64; 6] = [0.0, 1e-12, 5e-9, 1e-8, 2e-8, 1e-4];
        for &l in &anchors {
            for &w in &rel_widths {
                let u = ((l as f64) * (1.0 + w)) as f32;
                if u >= l && u.is_finite() {
                    assert_log_envelope_strict(l, u);
                }
            }
            for &w in &abs_widths {
                let u = ((l as f64) + w) as f32;
                if u >= l && u.is_finite() {
                    assert_log_envelope_strict(l, u);
                }
            }
            let u = next_up_f32(l);
            if u.is_finite() {
                assert_log_envelope_strict(l, u);
            }
        }
    }
}
