// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::validate::{validate_finite, validate_nonnegative_finite};
use super::LinearRelaxation;
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

// Re-export for test module's `use super::*` — cfg(test) items are used by
// the separate tests.rs file but the linter can't see the cross-file usage.
#[cfg(test)]
#[allow(unused_imports)]
use crate::LinearBounds;

/// HardSigmoid layer: y = max(0, min(1, alpha * x + beta))
///
/// A piecewise linear approximation of sigmoid, more efficient to compute.
/// Default ONNX values: alpha = 0.2, beta = 0.5
#[derive(Debug, Clone)]
pub struct HardSigmoidLayer {
    /// Slope in the linear region (default: 0.2)
    pub(crate) alpha: f32,
    /// Offset in the linear region (default: 0.5)
    pub(crate) beta: f32,
}

#[inline]
fn hard_sigmoid_eval(alpha: f32, beta: f32, x: f32) -> f32 {
    (alpha * x + beta).clamp(0.0, 1.0)
}

#[inline]
fn hard_sigmoid_linear_relaxation(l: f32, u: f32, alpha: f32, beta: f32) -> LinearRelaxation {
    // NaN/Inf guard: full-range constant bounds are always sound for HardSigmoid.
    if l.is_nan() || u.is_nan() || !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, 0.0, 0.0, 1.0);
    }

    // Negative alpha flips monotonicity and inverts the x_low/x_high region
    // ordering the branches below assume — a negative-alpha layer is refused
    // at construction and in IBP, but fail closed here too (defense in depth):
    // the constant [0, 1] envelope is sound for ANY alpha since the
    // HardSigmoid range is contained in [0, 1] by definition.
    if alpha < 0.0 {
        return LinearRelaxation::new(0.0, 0.0, 0.0, 1.0);
    }

    // Degenerate case: alpha=0 makes HardSigmoid a constant y = clip(beta, 0, 1).
    // Guard against division by zero on lines computing x_low and x_high.
    if alpha == 0.0 {
        let c = beta.clamp(0.0, 1.0);
        return LinearRelaxation::new(0.0, c, 0.0, c);
    }

    // Near-degenerate interval: exact value, no relaxation gap.
    // Uses epsilon guard to avoid extreme slopes from division by near-zero (u - l).
    // Pattern: SiLU (relaxation.rs:59), Mish, Softsign, HardSwish, GELU.
    if (u - l).abs() < 1e-8 {
        // SOUNDNESS (false-proof fix): a single eval(l) is NOT a sound constant relaxation —
        // hard_sigmoid is monotone increasing, so f varies over [l,u] by ~α·(u−l) and eval(l)
        // sits below f(u) (a certified upper bound under the true value when that gap exceeds
        // the ULP, e.g. small outputs near the lower kink). Cover the endpoint range with
        // directed outward rounding instead. NOTE: the same narrow-case-constant pattern is in
        // SiLU/Mish/Softsign/HardSwish/GELU (per the comment above) — flagged for audit follow-up.
        let y_l = hard_sigmoid_eval(alpha, beta, l);
        let y_u = hard_sigmoid_eval(alpha, beta, u);
        let lo = next_down_f32(y_l.min(y_u));
        let hi = next_up_f32(y_l.max(y_u));
        return LinearRelaxation::new(0.0, lo, 0.0, hi);
    }

    let x_low = -beta / alpha;
    let x_high = (1.0 - beta) / alpha;
    let yl = hard_sigmoid_eval(alpha, beta, l);
    let yu = hard_sigmoid_eval(alpha, beta, u);

    if u <= x_low {
        // Entirely in y = 0 region
        LinearRelaxation::zero()
    } else if l >= x_high {
        // Entirely in y = 1 region
        LinearRelaxation::new(0.0, 1.0, 0.0, 1.0)
    } else if l >= x_low && u <= x_high {
        // Entirely in linear region: y = alpha*x + beta
        LinearRelaxation::new(alpha, beta, alpha, beta)
    } else if l < x_low && u > x_high {
        // Crosses both boundaries. Directed rounding.
        // Upper: chord from (l, 0) to (x_high, 1). Lower: chord from (x_low, 0) to (u, 1).
        let max_abs = l.abs().max(u.abs()) as f64;
        let su64 = 1.0_f64 / (x_high as f64 - l as f64);
        let su = su64 as f32;
        let su_err = next_up_f32(((su64 - su as f64).abs() * max_abs) as f32);
        let iu = next_up_f32((1.0_f64 - su64 * x_high as f64) as f32 + su_err);
        let sl64 = 1.0_f64 / (u as f64 - x_low as f64);
        let sl = sl64 as f32;
        let sl_err = next_up_f32(((sl64 - sl as f64).abs() * max_abs) as f32);
        let il = next_down_f32((-sl64 * x_low as f64) as f32 - sl_err);
        LinearRelaxation::new(sl, il, su, iu)
    } else if l < x_low {
        // Crosses lower boundary only. Directed rounding.
        let max_abs = l.abs().max(u.abs()) as f64;
        let su64 = yu as f64 / (u as f64 - l as f64);
        let su = su64 as f32;
        let su_err = next_up_f32(((su64 - su as f64).abs() * max_abs) as f32);
        let (ls, li) = if su > 0.5 * alpha {
            (alpha, beta)
        } else {
            (0.0, 0.0)
        };
        let iu = next_up_f32((yu as f64 - su64 * u as f64) as f32 + su_err);
        LinearRelaxation::new(ls, li, su, iu)
    } else {
        // Crosses upper boundary only. Directed rounding.
        let max_abs = l.abs().max(u.abs()) as f64;
        let sl64 = (1.0_f64 - yl as f64) / (u as f64 - l as f64);
        let sl = sl64 as f32;
        let sl_err = next_up_f32(((sl64 - sl as f64).abs() * max_abs) as f32);
        let (us, ui) = if sl > 0.5 * alpha {
            (alpha, beta)
        } else {
            (0.0, 1.0)
        };
        let il = next_down_f32((yl as f64 - sl64 * l as f64) as f32 - sl_err);
        LinearRelaxation::new(sl, il, us, ui)
    }
}

impl HardSigmoidLayer {
    /// Validate and create a new HardSigmoid layer with the given parameters.
    ///
    /// `alpha = 0` is admitted: it makes HardSigmoid the constant function
    /// `y = clamp(beta, 0, 1)`, which both the IBP and CROWN paths handle
    /// exactly. Negative alpha stays refused — it flips monotonicity and the
    /// piecewise-region analysis in both paths assumes an increasing ramp.
    pub fn try_new(alpha: f32, beta: f32) -> Result<Self> {
        Ok(Self {
            alpha: validate_nonnegative_finite(alpha, "HardSigmoidLayer", "alpha")?,
            beta: validate_finite(beta, "HardSigmoidLayer", "beta")?,
        })
    }

    /// Create a new HardSigmoid layer with the given parameters.
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self::try_new(alpha, beta)
            .expect("invariant: HardSigmoidLayer::new requires validated parameters")
    }

    /// Create a HardSigmoid layer with ONNX default parameters (alpha=0.2, beta=0.5).
    pub fn default_params() -> Self {
        Self::new(0.2, 0.5)
    }

    /// Evaluate HardSigmoid at a point: max(0, min(1, alpha * x + beta))
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        (self.alpha * x + self.beta).clamp(0.0, 1.0)
    }
}

impl Default for HardSigmoidLayer {
    fn default() -> Self {
        Self::default_params()
    }
}

impl BoundPropagation for HardSigmoidLayer {
    /// IBP for HardSigmoid: y = max(0, min(1, alpha * x + beta))
    ///
    /// Three regions:
    /// - y = 0 when alpha * x + beta <= 0
    /// - y = alpha * x + beta when 0 < alpha * x + beta < 1
    /// - y = 1 when alpha * x + beta >= 1
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: the endpoint mapping below assumes a monotonically
        // increasing ramp (alpha >= 0). Negative alpha flips monotonicity —
        // f(lower) > f(upper) would invert the bounds — and stays refused
        // (also refused at construction; this covers crate-internal struct
        // literals). alpha = 0 is admitted and handled exactly below.
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "HardSigmoid IBP requires finite non-negative alpha (negative alpha flips monotonicity), got {}",
                self.alpha,
            )));
        }
        if !self.beta.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "HardSigmoid IBP requires finite beta, got {}",
                self.beta,
            )));
        }
        // Guard: NaN input bounds fail closed — NaN propagates through eval
        // (alpha*NaN + beta = NaN, and NaN.clamp is NaN).
        //
        // ±Inf endpoints are NOT rejected (deep-stack IBP overflow saturation,
        // dc9da0fde class): `eval` is limit-correct at infinity for alpha > 0
        // — alpha·(-inf) + beta = -inf clamps to 0 = HardSigmoid(-inf), and
        // alpha·(+inf) + beta = +inf clamps to 1 = HardSigmoid(+inf) — so a
        // saturated endpoint maps to its exact limit and the output stays a
        // finite subset of [0, 1]. (The alpha = 0 constant case is handled
        // separately: 0·(±inf) would be NaN, not a limit.)
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "HardSigmoid IBP: NaN input bounds".to_string(),
            ));
        }

        // alpha = 0: HardSigmoid is the constant y = clamp(beta, 0, 1),
        // independent of x (matches the CROWN relaxation's exact alpha = 0
        // constant handling). Computed directly — the monotone mapping below
        // would evaluate 0·(±inf) = NaN on saturated inputs.
        if self.alpha == 0.0 {
            let c = self.beta.clamp(0.0, 1.0);
            let lower = input.lower().mapv(|_| c);
            let upper = input.upper().mapv(|_| c);
            return BoundedTensor::new(lower, upper);
        }

        // HardSigmoid is monotonically increasing (alpha > 0 established above)
        let lower = input.lower().mapv(|x| self.eval(x));
        let upper = input.upper().mapv(|x| self.eval(x));
        BoundedTensor::new(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        HardSigmoidLayer,
        NyError::InvalidSpec(
            "HardSigmoid CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl HardSigmoidLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        HardSigmoidLayer,
        |layer: &HardSigmoidLayer, l, u| {
            hard_sigmoid_linear_relaxation(l, u, layer.alpha, layer.beta)
        },
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("HardSigmoid", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn audit_hard_sigmoid_relax(l: f32, u: f32, alpha: f32, beta: f32) -> LinearRelaxation {
    hard_sigmoid_linear_relaxation(l, u, alpha, beta)
}
