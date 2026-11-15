// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified enclosure for activation `QuantizeLinear -> DequantizeLinear` pairs.
//!
//! # The function being enclosed
//!
//! The fused QDQ pair computes, in ONNX/onnxruntime FLOAT32 semantics,
//!
//! ```text
//! d = fl32(x / s)                        (one rounded f32 division; some
//!                                         kernels instead compute
//!                                         fl32(x * fl32(1/s)), two roundings)
//! r = round_ties_even(d)                 (exact: integer-valued result)
//! q = saturate(r + zp, qmin, qmax)       (exact integer arithmetic)
//! y = fl32((q - zp) * s)                 (one rounded f32 multiply;
//!                                         |q - zp| <= 255 is exact in f32)
//! ```
//!
//! Writing `k = q - zp = clamp(r, qmin - zp, qmax - zp)`, `LO' = (qmin - zp)*s`,
//! `HI' = (qmax - zp)*s`, `M = max(|LO'|, |HI'|)` and `U = 2^-24` (the f32
//! unit roundoff), the enclosure this layer certifies is:
//!
//! ```text
//! y ∈ [ clamp(x - EPS, LO_dn, HI_dn) - G ,  clamp(x + EPS, LO_up, HI_up) + G ]
//! ```
//!
//! with, computed OUTWARD in f64 and rounded OUTWARD to f32:
//!
//! * `EPS >= s/2 + 8*U*(M + 2s) + tiny_abs` — covers the half-bucket rounding
//!   (ties-to-even at exactly .5 is inclusive in the `<= 1/2` bound), the
//!   division rounding of `x/s` under the divide, reciprocal-multiply, and
//!   f32-zero-point-add kernel models, and the subnormal absolute floor. The `|x|`-proportional division error only matters while
//!   the clamp is NOT saturated, where `|x| <= M + s` (see the bootstrap
//!   argument below), so a CONSTANT `EPS` suffices.
//! * `G >= 4*U*M + tiny_abs` — covers the final dequantization multiply
//!   (`|y - k*s| <= |k*s|*U <= M*U`), with margin.
//! * `LO_dn <= LO' <= LO_up` and `HI_dn <= HI' <= HI_up` — the saturation
//!   edges rounded DOWN for the lower envelope and UP for the upper envelope,
//!   so a saturated `y = HI'*(1 ± U)` stays inside on both sides.
//!
//! Bootstrap bound for the non-saturated region: `x + e1 ∈ [LO', HI']` with
//! `|e1| = |r*s - x| <= s/2 + cU|x| + 256*s*U` (c <= 3) gives
//! `|x| < M + s` up to negligible factors, so `8U(M + 2s)` dominates the
//! error exactly where it is needed; outside that region both the true value
//! and the enclosure saturate to the same (outward-rounded) edge.
//!
//! Both envelope functions are monotone non-decreasing shifted `Clip`s, so:
//!
//! * IBP is `[env_lo(l), env_up(u)]` with directed f32 steps at every add;
//! * the CROWN backward relaxation composes the certified `Clip` relaxation
//!   (`clip_linear_relaxation`, directed rounding) evaluated on the
//!   OUTWARD-shifted interval, with the shift and the `±G` term folded into
//!   the intercepts under f64 directed rounding.
//!
//! Parameters are per-tensor (length 1) or per-last-axis (length = feature
//! count), applied to the flattened tensor by cyclic indexing — the same
//! trailing-broadcast convention as `EpsilonExpandLayer`.
//!
//! The constructor is certified against the exhaustive containment suite in
//! `tests` below: all 256 uint8 buckets x scale/zero-point grids, f32 inputs
//! sampled densely around every bucket edge including the round-half-even
//! ties, under both the divide and reciprocal-multiply kernel models, plus a
//! planted-violation test that shows the harness CAN fail.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{add_up_f32, next_down_f32, next_up_f32, sub_down_f32, BoundedTensor};
use std::borrow::Cow;

use crate::layers::activations::clip_linear_relaxation;
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// f32 unit roundoff.
const UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8; // 2^-24

/// Certified enclosure layer for an activation QuantizeLinear->DequantizeLinear
/// pair (see the module docs for the derivation).
#[derive(Debug, Clone)]
pub struct QdqPerturbationLayer {
    /// Pre-clip additive widening (`EPS`), per parameter slot.
    eps: Array1<f32>,
    /// Post-clip additive widening (`G`), per parameter slot.
    recon: Array1<f32>,
    /// Saturation edges for the LOWER envelope (rounded down).
    lo_dn: Array1<f32>,
    hi_dn: Array1<f32>,
    /// Saturation edges for the UPPER envelope (rounded up).
    lo_up: Array1<f32>,
    hi_up: Array1<f32>,
}

/// One parameter slot of the certified envelope.
#[derive(Clone, Copy)]
struct Slot {
    eps: f32,
    recon: f32,
    lo_dn: f32,
    hi_dn: f32,
    lo_up: f32,
    hi_up: f32,
}

fn f64_to_f32_down(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) > v {
        next_down_f32(f)
    } else {
        f
    }
}

fn f64_to_f32_up(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) < v {
        next_up_f32(f)
    } else {
        f
    }
}

impl QdqPerturbationLayer {
    /// Build the certified envelope for per-slot quantization parameters.
    ///
    /// `scales` and `zero_points` must have equal length (each entry is one
    /// parameter slot: length 1 for per-tensor quantization, feature count for
    /// per-last-axis). `qmin`/`qmax` are the saturation range of the quantized
    /// integer type (uint8: 0..=255, int8: -128..=127).
    pub fn try_certified(
        scales: &[f32],
        zero_points: &[i64],
        qmin: i64,
        qmax: i64,
    ) -> Result<Self> {
        if scales.is_empty() || scales.len() != zero_points.len() {
            return Err(NyError::InvalidSpec(format!(
                "QDQ relaxation requires matching non-empty scale/zero_point slots, got {} and {}",
                scales.len(),
                zero_points.len()
            )));
        }
        if qmin >= qmax || qmin < -(1 << 24) || qmax > (1 << 24) {
            return Err(NyError::InvalidSpec(format!(
                "QDQ relaxation has invalid quantized range [{qmin}, {qmax}]"
            )));
        }
        let n = scales.len();
        let mut eps = Array1::<f32>::zeros(n);
        let mut recon = Array1::<f32>::zeros(n);
        let mut lo_dn = Array1::<f32>::zeros(n);
        let mut hi_dn = Array1::<f32>::zeros(n);
        let mut lo_up = Array1::<f32>::zeros(n);
        let mut hi_up = Array1::<f32>::zeros(n);
        for i in 0..n {
            let scale = scales[i];
            let zp = zero_points[i];
            if !(scale.is_finite() && scale > 0.0) {
                return Err(NyError::InvalidSpec(format!(
                    "QDQ relaxation requires positive finite scale, got {scale}"
                )));
            }
            if zp < qmin || zp > qmax {
                return Err(NyError::InvalidSpec(format!(
                    "QDQ relaxation zero_point {zp} outside quantized range [{qmin}, {qmax}]"
                )));
            }
            let s = f64::from(scale);
            // Exact in f64: |qmin - zp| <= 2^25 and s is an f32.
            let lo_real = (qmin - zp) as f64 * s;
            let hi_real = (qmax - zp) as f64 * s;
            let m = lo_real.abs().max(hi_real.abs());
            // tiny_abs covers subnormal division/multiply results, where the
            // relative rounding model does not hold.
            let tiny_abs = f64::from(f32::MIN_POSITIVE);
            // 8U(m + 2s): covers the divide model (U), the reciprocal-multiply
            // model (2U + U^2), AND kernels that add the zero point in f32
            // BEFORE rounding (an extra (|d| + |zp|)U <= (|x|/s + 256)U term,
            // whose s-scaled form 256*s*U is below 2U*m + 2U*s for every
            // 8-bit range), with margin. Costs ~1e-6 absolute at smart_turn's
            // head scale against a half-bucket of 4.7e-3.
            let eps_real = s / 2.0 + 8.0 * UNIT_ROUNDOFF * (m + 2.0 * s) + tiny_abs;
            let g_real = 4.0 * UNIT_ROUNDOFF * m + tiny_abs;
            // One extra outward ULP step on every published constant absorbs
            // the (much smaller than 1 f32 ULP) f64 evaluation error above.
            eps[i] = next_up_f32(f64_to_f32_up(eps_real));
            recon[i] = next_up_f32(f64_to_f32_up(g_real));
            lo_dn[i] = next_down_f32(f64_to_f32_down(lo_real));
            hi_dn[i] = next_down_f32(f64_to_f32_down(hi_real));
            lo_up[i] = next_up_f32(f64_to_f32_up(lo_real));
            hi_up[i] = next_up_f32(f64_to_f32_up(hi_real));
            if !(eps[i].is_finite()
                && recon[i].is_finite()
                && lo_dn[i].is_finite()
                && hi_up[i].is_finite())
            {
                return Err(NyError::InvalidSpec(format!(
                    "QDQ relaxation envelope overflows f32 for scale {scale}, zero_point {zp}, range [{qmin}, {qmax}]"
                )));
            }
        }
        Ok(Self {
            eps,
            recon,
            lo_dn,
            hi_dn,
            lo_up,
            hi_up,
        })
    }

    /// Number of parameter slots (1 = per-tensor).
    pub fn param_slots(&self) -> usize {
        self.eps.len()
    }

    /// The pre-clip widening radius of slot 0 (telemetry only).
    pub fn epsilon(&self) -> f32 {
        self.eps[0]
    }

    #[inline]
    fn slot(&self, flat_index: usize) -> Slot {
        let j = flat_index % self.eps.len();
        Slot {
            eps: self.eps[j],
            recon: self.recon[j],
            lo_dn: self.lo_dn[j],
            hi_dn: self.hi_dn[j],
            lo_up: self.lo_up[j],
            hi_up: self.hi_up[j],
        }
    }

    /// Lower envelope `clamp(x - EPS, LO_dn, HI_dn) - G`, every step directed
    /// toward -inf.
    #[inline]
    fn env_lo(&self, x: f32, s: &Slot) -> f32 {
        sub_down_f32(sub_down_f32(x, s.eps).clamp(s.lo_dn, s.hi_dn), s.recon)
    }

    /// Upper envelope `clamp(x + EPS, LO_up, HI_up) + G`, every step directed
    /// toward +inf.
    #[inline]
    fn env_up(&self, x: f32, s: &Slot) -> f32 {
        add_up_f32(add_up_f32(x, s.eps).clamp(s.lo_up, s.hi_up), s.recon)
    }

    /// Per-element CROWN relaxation of the envelope on pre-activation `[l, u]`.
    ///
    /// Composes the certified `Clip` relaxation on the OUTWARD-shifted
    /// interval, folding the shift and the `±G` term into the intercepts with
    /// f64 directed rounding:
    ///
    /// * lower: `y >= Clip_[LO_dn,HI_dn](x - EPS) - G >= sl*x + (il - sl*EPS - G)`
    /// * upper: `y <= Clip_[LO_up,HI_up](x + EPS) + G <= su*x + (iu + su*EPS + G)`
    fn relaxation(&self, l: f32, u: f32, flat_index: usize) -> LinearRelaxation {
        let s = self.slot(flat_index);
        if l.is_nan() || u.is_nan() || !l.is_finite() || !u.is_finite() {
            // Constant fallback at the full envelope range: always sound.
            return LinearRelaxation::new(
                0.0,
                sub_down_f32(s.lo_dn, s.recon),
                0.0,
                add_up_f32(s.hi_up, s.recon),
            );
        }
        // Lower envelope: relax Clip on an interval CONTAINING [l-EPS, u-EPS].
        let lo_relax = clip_linear_relaxation(
            sub_down_f32(l, s.eps),
            next_up_f32(u - s.eps),
            s.lo_dn,
            s.hi_dn,
        );
        // Upper envelope: relax Clip on an interval CONTAINING [l+EPS, u+EPS].
        let up_relax = clip_linear_relaxation(
            next_down_f32(l + s.eps),
            add_up_f32(u, s.eps),
            s.lo_up,
            s.hi_up,
        );
        let sl = lo_relax.lower_slope;
        let su = up_relax.upper_slope;
        // Fold the shift into the intercepts in f64, rounded outward. Slopes
        // from the Clip relaxation are within [0, 1] with directed error
        // handling done inside `clip_linear_relaxation`.
        let il = f64_to_f32_down(
            f64::from(lo_relax.lower_intercept)
                - f64::from(sl) * f64::from(s.eps)
                - f64::from(s.recon),
        );
        let iu = f64_to_f32_up(
            f64::from(up_relax.upper_intercept)
                + f64::from(su) * f64::from(s.eps)
                + f64::from(s.recon),
        );
        // One extra outward step absorbs the f64 fold's own rounding.
        LinearRelaxation::new(sl, next_down_f32(il), su, next_up_f32(iu))
    }

    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        crate::layers::common::crown_elementwise_backward_indexed(
            bounds,
            pre_activation,
            |l, u, i| self.relaxation(l, u, i),
        )
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        crate::layers::common::crown_elementwise_backward_batched_indexed(
            bounds,
            pre_activation,
            |l, u, i| self.relaxation(l, u, i),
        )
    }
}

impl BoundPropagation for QdqPerturbationLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // NaN guard: f32::clamp propagates NaN. ±Inf is legitimate (an
        // upstream OpaqueSkip hands [-inf, +inf]) and saturates through the
        // clamp to the finite envelope range — this layer is a taint sink,
        // like Clip.
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "QdqPerturbation IBP: NaN input bounds".to_string(),
            ));
        }
        let shape = input.lower().shape().to_vec();
        let lower_std = input.lower().as_standard_layout();
        let upper_std = input.upper().as_standard_layout();
        let lower: Vec<f32> = lower_std
            .iter()
            .enumerate()
            .map(|(i, &v)| self.env_lo(v, &self.slot(i)))
            .collect();
        let upper: Vec<f32> = upper_std
            .iter()
            .enumerate()
            .map(|(i, &v)| self.env_up(v, &self.slot(i)))
            .collect();
        let lower = ArrayD::from_shape_vec(IxDyn(&shape), lower)
            .map_err(|e| NyError::InvalidSpec(format!("QdqPerturbation IBP reshape: {e}")))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&shape), upper)
            .map_err(|e| NyError::InvalidSpec(format!("QdqPerturbation IBP reshape: {e}")))?;
        // The envelope of a finite input is finite by construction; an
        // infinite input saturates to the finite clip range plus G. NaN
        // cannot arise (guard above), and would still hard-error here.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::InvalidSpec(
            "QdqPerturbation CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        QdqPerturbationLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

#[cfg(test)]
mod tests;
