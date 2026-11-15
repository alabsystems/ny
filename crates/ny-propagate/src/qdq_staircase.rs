// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared-input tightening for the QDQ activation staircase (#qdq-crown-followup).
//!
//! ## The defect this repairs (diagnosed on smart_turn row 11)
//!
//! The graph builder's normalization canonicalization rewrites every
//! decomposed LayerNorm/RmsNorm into primitives whose square is
//! `MulBinary(centered, centered)` — a product of a node WITH ITSELF. Plain
//! interval-product IBP does not know the operands are the same variable, so
//! `[-9, 9] * [-9, 9]` publishes `[-81, 81]` where the true square image is
//! `[0, 81]`. The negative lower bound then flows `variance -> var+eps ->
//! Sqrt` (clamped at 0) `-> Reciprocal` over an interval whose lower endpoint
//! is 0 — and the reciprocal is unbounded there, so `__inv_std` publishes
//! `[..., +inf]` and `__normalized = centered * inv_std` publishes
//! `[-inf, +inf]`. Every decomposed-normalization output in
//! smart-turn-multimodal-cpu.onnx reads `[-inf, +inf]` at IBP root through
//! exactly this chain (119-quantizer probe, 2026-09-01), re-clamped only by
//! the following `QdqPerturbation` envelopes.
//!
//! The erf-GELU staircase (`Div(x, c) -> Erf -> Add(1) -> Mul(x, ·)`,
//! `c ≈ √2`) loses the same correlation at its product: `x` and
//! `1 + erf(x/c)` share `x`, and the factor-wise product of
//! `[-4, 4] x [0, 2]` is `[-8, 8]` where the true image of
//! `x·(1 + erf(x/c))` is `[-0.34, 8]` (global minimum `2·min GELU ≈ -0.34`).
//!
//! ## What this module computes
//!
//! Two per-element tightenings, both INTERSECTIONS with the already-computed
//! plain interval-product bound (shrink-only, mirroring the SwiGLU zonotope
//! intersection at the same dispatch site):
//!
//! - **Same-input square** (`MulBinary(a, a)` by node name): the exact square
//!   image via [`square_interval_bounds`] (outward-rounded; lower is 0 on a
//!   sign-crossing interval, never negative).
//! - **Erf-GELU product** (`Mul(x, Add(Erf(Div(x, c)), 1))` with scalar
//!   constants, `c > 0`): the EXACT identity
//!   `x·(1 + erf(x/c)) = c·√2 · GELU_erf(x·√2/c)` reduces the product to the
//!   reviewed 1-D [`gelu_bound_interval`] (auto_LiRPA's BoundGelu treatment:
//!   endpoint + critical-point evaluation with directed rounding), scaled by
//!   the positive constant `c·√2` under outward f64 rounding. The authored
//!   divisor `c` is used exactly — no assumption that it equals √2.
//!
//! ## Soundness posture
//!
//! CONTAINMENT, never finiteness: each tightening encloses the true image of
//! the shared-input product over the operand box, and the final box is the
//! per-element intersection with the plain product bound, falling back to the
//! plain bound wherever the intersection would invert. DARK by default: armed
//! ONLY through the existing `model.qdq_activation_relaxation` loader gate
//! (canary-only; no shipped preset arms it, pinned by
//! `vnncomp_preset_tests`), the same dark lane whose report-only latch
//! withholds smart_turn verdicts. Default-off keeps every other family
//! bit-identical.

use std::sync::atomic::{AtomicBool, Ordering};

use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::normalization::math_common::square_interval_bounds;
use crate::layers::softmax::{gelu_bound_interval, GeluApproximation};
use crate::layers::Layer;
use crate::network::GraphNetwork;

/// Armed state of the staircase tightening (see module docs). Set at model
/// load from the `model.qdq_activation_relaxation` preset key; never read
/// from the environment.
static QDQ_STAIRCASE_IBP: AtomicBool = AtomicBool::new(false);

/// One-shot engagement observables (eprintln!, not tracing — the vnncomp lane
/// suppresses info!-level events; see `[global_3d_avgpool] armed`).
static SQUARE_ANNOUNCED: AtomicBool = AtomicBool::new(false);
static GELU_ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// Arm or disarm the shared-input staircase tightening. Called by the CLI at
/// model load with the resolved `model.qdq_activation_relaxation` value, so a
/// later in-process load without the gate disarms it again.
pub fn set_qdq_staircase_ibp(enabled: bool) {
    QDQ_STAIRCASE_IBP.store(enabled, Ordering::SeqCst);
}

pub(crate) fn qdq_staircase_ibp_armed() -> bool {
    QDQ_STAIRCASE_IBP.load(Ordering::SeqCst)
}

/// Per-element intersection of two sound enclosures of the same value.
///
/// Falls back to the `base` endpoints wherever directed rounding makes the
/// intersection empty (both inputs are sound, so a strict inversion can only
/// be rounding noise; keeping `base` is always sound).
fn intersect_into(
    base: &BoundedTensor,
    tighter_lower: &[f32],
    tighter_upper: &[f32],
) -> Option<BoundedTensor> {
    let mut lower = base.lower().to_owned();
    let mut upper = base.upper().to_owned();
    for ((lo, up), (&tl, &tu)) in lower
        .iter_mut()
        .zip(upper.iter_mut())
        .zip(tighter_lower.iter().zip(tighter_upper.iter()))
    {
        let new_lo = lo.max(tl);
        let new_up = up.min(tu);
        if new_lo <= new_up {
            *lo = new_lo;
            *up = new_up;
        }
    }
    BoundedTensor::new(lower, upper).ok()
}

/// The scalar constant of a rank-0/len-1 constant tensor.
fn scalar_constant(constant: &ndarray::ArrayD<f32>) -> Option<f32> {
    (constant.len() == 1).then(|| constant.iter().copied().next())?
}

impl GraphNetwork {
    /// Try both shared-input tightenings for a `MulBinary` node (see module
    /// docs). Returns `None` when disarmed, when no motif matches, or when
    /// anything about the match is not exactly as certified — the caller then
    /// keeps the plain interval-product bound unchanged.
    pub(crate) fn try_qdq_staircase_mul_tighten(
        &self,
        name_a: &str,
        name_b: &str,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        plain: &BoundedTensor,
    ) -> Option<BoundedTensor> {
        if !qdq_staircase_ibp_armed() {
            return None;
        }
        if name_a == name_b {
            return self.tighten_same_input_square(input_a, plain);
        }
        // Try the erf-GELU motif with either operand as the carrier `x`.
        if let Some(result) = self.tighten_erf_gelu_product(name_a, name_b, input_a, plain) {
            return Some(result);
        }
        self.tighten_erf_gelu_product(name_b, name_a, input_b, plain)
    }

    /// `MulBinary(a, a)`: the exact square image, elementwise.
    fn tighten_same_input_square(
        &self,
        input: &BoundedTensor,
        plain: &BoundedTensor,
    ) -> Option<BoundedTensor> {
        if input.shape() != plain.shape() {
            return None; // a broadcast square is not the plain x*x motif
        }
        let n = input.lower().len();
        let mut tl = Vec::with_capacity(n);
        let mut tu = Vec::with_capacity(n);
        for (&l, &u) in input.lower().iter().zip(input.upper().iter()) {
            if !l.is_finite() || !u.is_finite() {
                tl.push(f32::NEG_INFINITY);
                tu.push(f32::INFINITY);
                continue;
            }
            let (sl, su) = square_interval_bounds(l, u);
            tl.push(sl);
            tu.push(su);
        }
        if !SQUARE_ANNOUNCED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[qdq_staircase] armed: same-input square IBP engaged \
                 (MulBinary(a, a) uses the exact square image, lower >= 0)"
            );
        }
        intersect_into(plain, &tl, &tu)
    }

    /// `Mul(x, Add(Erf(Div(x, c)), 1))`: the exact 1-D image via
    /// `c·√2 · GELU_erf(x·√2/c)`.
    fn tighten_erf_gelu_product(
        &self,
        x_name: &str,
        gate_name: &str,
        x_bounds: &BoundedTensor,
        plain: &BoundedTensor,
    ) -> Option<BoundedTensor> {
        // gate = AddConstant(scalar exactly 1.0) over Erf over
        // DivConstant(scalar c > 0) over the SAME x node.
        let gate = self.node(gate_name)?;
        let Layer::AddConstant(add) = gate.layer() else {
            return None;
        };
        if scalar_constant(add.constant())? != 1.0 {
            return None;
        }
        let [erf_name] = gate.inputs() else {
            return None;
        };
        let erf_node = self.node(erf_name)?;
        if !matches!(erf_node.layer(), Layer::Erf(_)) {
            return None;
        }
        let [div_name] = erf_node.inputs() else {
            return None;
        };
        let div_node = self.node(div_name)?;
        let Layer::DivConstant(div) = div_node.layer() else {
            return None;
        };
        let c = scalar_constant(div.constant())?;
        if !(c.is_finite() && c > 0.0) {
            return None;
        }
        let [div_input] = div_node.inputs() else {
            return None;
        };
        if div_input != x_name {
            return None;
        }
        if x_bounds.shape() != plain.shape() {
            return None; // broadcasting would change the elementwise identity
        }

        // Exact identity: with v = x·√2/c (c > 0, so the map is increasing),
        //   x·(1 + erf(x/c)) = (c/√2)·v·(1 + erf(v/√2)) = c·√2 · GELU_erf(v).
        // Outward f64 rounding on the substitution and the back-scale keeps
        // the enclosure sound; gelu_bound_interval itself directs its
        // endpoint/critical-point evaluations outward (auto_LiRPA BoundGelu
        // tables port).
        let c64 = c as f64;
        let sqrt2 = std::f64::consts::SQRT_2;
        let scale_in = sqrt2 / c64; // > 0
        let scale_out = c64 * sqrt2; // > 0
        let n = x_bounds.lower().len();
        let mut tl = Vec::with_capacity(n);
        let mut tu = Vec::with_capacity(n);
        for (&l, &u) in x_bounds.lower().iter().zip(x_bounds.upper().iter()) {
            if !l.is_finite() || !u.is_finite() {
                tl.push(f32::NEG_INFINITY);
                tu.push(f32::INFINITY);
                continue;
            }
            let vl = next_down_f32((l as f64 * scale_in) as f32);
            let vu = next_up_f32((u as f64 * scale_in) as f32);
            let (gl, gu) = gelu_bound_interval(vl, vu, GeluApproximation::Erf);
            tl.push(next_down_f32((gl as f64 * scale_out) as f32));
            tu.push(next_up_f32((gu as f64 * scale_out) as f32));
        }
        if !GELU_ANNOUNCED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[qdq_staircase] armed: erf-gelu product IBP engaged \
                 (Mul(x, 1 + erf(x/c)) uses the exact 1-D image c*sqrt2*GELU(x*sqrt2/c))"
            );
        }
        intersect_into(plain, &tl, &tu)
    }
}

#[cfg(test)]
mod tests;
