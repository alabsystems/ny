// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Variable-style AdaIN ternary CROWN backward propagation.
//!
//! For y = g * InstanceNorm(x) + b, the center-point linearization gives:
//!   J_x = Diag(g_c) · J_instnorm(x_c)
//!   J_g = Diag(z_c)  where z_c = InstanceNorm(x_c)
//!   J_b = I
//!   constant = g_c * (z_c - J_instnorm · x_c)
//!
//! Returns `BackwardDispatchResult::Nary` with 3 `LinearBounds` for (x, g, b).
//!
//! Ref: designs/2026-03-18-issue-4142-packet-a-ny-local-ternary.md

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::types::AdaIN1dLayer;
use crate::layers::normalization::crown_common::flatten_preactivation;
use crate::layers::normalization::trait_norm::NormLayer;
use crate::{contiguous_flat_slice, LinearBounds};

/// Return type for ternary CROWN backward: (per-input LinearBounds, bias_lower, bias_upper).
pub(crate) type TernaryCrownResult = (Vec<Option<LinearBounds>>, Array1<f32>, Array1<f32>);

/// Bundled input bounds for ternary IBP margin validation.
struct JointBounds<'a> {
    x_lo: &'a Array1<f32>,
    x_hi: &'a Array1<f32>,
    g_lo: &'a Array1<f32>,
    g_hi: &'a Array1<f32>,
    b_lo: &'a Array1<f32>,
    b_hi: &'a Array1<f32>,
}

/// Center-point linearization artifacts.
struct Linearization {
    j_x: Array2<f32>,
    z_c: Array1<f32>,
    constant: Array1<f32>,
}

impl AdaIN1dLayer {
    /// Ternary CROWN backward for variable-style AdaIN.
    ///
    /// Returns 3 `LinearBounds` (for x, style_gamma, style_beta) plus shared bias.
    /// Each `LinearBounds` has zero biases; all bias goes through the shared channel.
    ///
    /// Uses center-point linearization with IBP-validated margin widening for soundness.
    pub(crate) fn propagate_crown_ternary(
        &self,
        node_lb: &LinearBounds,
        x_bounds: &BoundedTensor,
        g_bounds: &BoundedTensor,
        b_bounds: &BoundedTensor,
    ) -> Result<TernaryCrownResult> {
        let (x_lower, x_upper) = flatten_preactivation(x_bounds)?;
        let (g_lower, g_upper) = flatten_preactivation(g_bounds)?;
        let (b_lower, b_upper) = flatten_preactivation(b_bounds)?;

        let n = x_lower.len();
        let d = node_lb.num_outputs();

        // Non-finite guard (#3259).
        let has_non_finite = [&x_lower, &x_upper, &g_lower, &g_upper, &b_lower, &b_upper]
            .iter()
            .any(|a| a.iter().any(|v| !v.is_finite()));
        if has_non_finite {
            return self.conservative_ternary_result(d, n);
        }

        let lin =
            compute_linearization(&self.instance_norm, &x_lower, &x_upper, &g_lower, &g_upper)?;

        // IBP ternary bounds for validation.
        let ibp_bounds = self.propagate_ibp_ternary(x_bounds, g_bounds, b_bounds)?;
        let ibp_flat = ibp_bounds.flatten();
        let ibp_lo = contiguous_flat_slice(ibp_flat.lower());
        let ibp_hi = contiguous_flat_slice(ibp_flat.upper());

        let joint = JointBounds {
            x_lo: &x_lower,
            x_hi: &x_upper,
            g_lo: &g_lower,
            g_hi: &g_upper,
            b_lo: &b_lower,
            b_hi: &b_upper,
        };
        let (margin_above, margin_below) =
            compute_ibp_validated_margins(n, &lin, &joint, ibp_lo.as_ref(), ibp_hi.as_ref());

        backward_ternary_f64(d, n, node_lb, &lin, &margin_above, &margin_below)
    }

    /// Conservative result: zero A-matrices, ±∞ bias.
    fn conservative_ternary_result(&self, d: usize, n: usize) -> Result<TernaryCrownResult> {
        let zero_lb = LinearBounds::new_or_conservative(
            Array2::zeros((d, n)),
            Array1::zeros(d),
            Array2::zeros((d, n)),
            Array1::zeros(d),
        )?;
        Ok((
            vec![Some(zero_lb.clone()), Some(zero_lb.clone()), Some(zero_lb)],
            Array1::from_elem(d, f32::NEG_INFINITY),
            Array1::from_elem(d, f32::INFINITY),
        ))
    }
}

/// Overflow-safe midpoint: l + (u - l) / 2.
fn safe_midpoint(lower: &Array1<f32>, upper: &Array1<f32>) -> Array1<f32> {
    ndarray::Zip::from(lower)
        .and(upper)
        .map_collect(|&l, &u| l + (u - l) / 2.0)
}

/// Compute center-point linearization artifacts (J_x, z_c, constant).
fn compute_linearization(
    instance_norm: &crate::layers::normalization::InstanceNorm1dLayer,
    x_lo: &Array1<f32>,
    x_hi: &Array1<f32>,
    g_lo: &Array1<f32>,
    g_hi: &Array1<f32>,
) -> Result<Linearization> {
    let x_c = safe_midpoint(x_lo, x_hi);
    let g_c = safe_midpoint(g_lo, g_hi);
    let n = x_c.len();

    let z_c = instance_norm.eval(&x_c)?;
    let j_z = instance_norm.jacobian(&x_c)?;

    if j_z.iter().any(|v| !v.is_finite()) || z_c.iter().any(|v| !v.is_finite()) {
        return Err(NyError::NumericalInstability(
            "AdaIN1d ternary CROWN: non-finite InstanceNorm Jacobian or eval".to_string(),
        ));
    }

    // J_x[i, j] = g_c[i] * J_z[i, j]  (Diag(g_c) · J_z)
    let mut j_x = j_z.clone();
    for i in 0..n {
        let gc = g_c[i];
        for j in 0..n {
            j_x[[i, j]] *= gc;
        }
    }

    // constant = g_c * (z_c - J_z · x_c)
    let jz_xc = j_z.dot(&x_c);
    let constant: Array1<f32> = ndarray::Zip::from(&g_c)
        .and(&z_c)
        .and(&jz_xc)
        .map_collect(|&gc, &zc, &jzxc| gc * (zc - jzxc));

    Ok(Linearization { j_x, z_c, constant })
}

/// Compute IBP-validated margins for the ternary linear approximation.
///
/// Concretizes `y_approx = J_x · x + Diag(z_c) · g + b + constant` over the joint
/// input box and compares with IBP bounds to find the minimum margin needed.
fn compute_ibp_validated_margins(
    n: usize,
    lin: &Linearization,
    joint: &JointBounds<'_>,
    ibp_lo: &[f32],
    ibp_hi: &[f32],
) -> (Array1<f32>, Array1<f32>) {
    let mut margin_above = Array1::from_elem(n, 0.0f32);
    let mut margin_below = Array1::from_elem(n, 0.0f32);

    for i in 0..n {
        let (approx_lo, approx_hi) = concretize_element(i, n, lin, joint);

        // Soundness margins: widen bounds when the center-point linearization
        // is tighter than IBP (potentially unsound due to multilinear cross-terms).
        //
        // margin_below[i] > 0: approx_lo > ibp_lo → push lower DOWN.
        // margin_above[i] > 0: approx_hi < ibp_hi → push upper UP.
        let widen_below = approx_lo - ibp_lo[i] as f64;
        let widen_above = ibp_hi[i] as f64 - approx_hi;

        if widen_below > 0.0 {
            margin_below[i] = next_up_f32(widen_below as f32);
        }
        if widen_above > 0.0 {
            margin_above[i] = next_up_f32(widen_above as f32);
        }
    }

    (margin_above, margin_below)
}

/// Concretize the i-th output of the linear approximation over the joint box.
fn concretize_element(
    i: usize,
    n: usize,
    lin: &Linearization,
    joint: &JointBounds<'_>,
) -> (f64, f64) {
    let mut approx_lo = lin.constant[i] as f64;
    let mut approx_hi = lin.constant[i] as f64;

    for j in 0..n {
        let jxij = lin.j_x[[i, j]] as f64;
        if jxij >= 0.0 {
            approx_lo += jxij * joint.x_lo[j] as f64;
            approx_hi += jxij * joint.x_hi[j] as f64;
        } else {
            approx_lo += jxij * joint.x_hi[j] as f64;
            approx_hi += jxij * joint.x_lo[j] as f64;
        }
    }

    let zci = lin.z_c[i] as f64;
    if zci >= 0.0 {
        approx_lo += zci * joint.g_lo[i] as f64;
        approx_hi += zci * joint.g_hi[i] as f64;
    } else {
        approx_lo += zci * joint.g_hi[i] as f64;
        approx_hi += zci * joint.g_lo[i] as f64;
    }

    approx_lo += joint.b_lo[i] as f64;
    approx_hi += joint.b_hi[i] as f64;

    (approx_lo, approx_hi)
}

/// Backward propagation with f64 accumulation for the ternary surface.
///
/// Produces 3 `LinearBounds` (for x, g, b) plus shared bias.
fn backward_ternary_f64(
    d: usize,
    n: usize,
    node_lb: &LinearBounds,
    lin: &Linearization,
    margin_above: &Array1<f32>,
    margin_below: &Array1<f32>,
) -> Result<TernaryCrownResult> {
    let lower_a = node_lb.lower_a();
    let upper_a = node_lb.upper_a();

    let mut ax_lower = Array2::<f32>::zeros((d, n));
    let mut ax_upper = Array2::<f32>::zeros((d, n));
    let mut ag_lower = Array2::<f32>::zeros((d, n));
    let mut ag_upper = Array2::<f32>::zeros((d, n));
    let ab_lower = lower_a.to_owned();
    let ab_upper = upper_a.to_owned();

    let mut bias_lower = Array1::<f32>::zeros(d);
    let mut bias_upper = Array1::<f32>::zeros(d);

    for row in 0..d {
        let mut bias_lo_acc = node_lb.lower_b()[row] as f64;
        let mut bias_hi_acc = node_lb.upper_b()[row] as f64;

        for j in 0..n {
            let mut ax_lo_j = 0.0f64;
            let mut ax_hi_j = 0.0f64;
            for i in 0..n {
                ax_lo_j += lower_a[[row, i]] as f64 * lin.j_x[[i, j]] as f64;
                ax_hi_j += upper_a[[row, i]] as f64 * lin.j_x[[i, j]] as f64;
            }
            ax_lower[[row, j]] = ax_lo_j as f32;
            ax_upper[[row, j]] = ax_hi_j as f32;

            let zc_j = lin.z_c[j] as f64;
            ag_lower[[row, j]] = (lower_a[[row, j]] as f64 * zc_j) as f32;
            ag_upper[[row, j]] = (upper_a[[row, j]] as f64 * zc_j) as f32;
        }

        // Shared bias = node_lb.b + A · constant ± margin adjustments.
        for i in 0..n {
            let lo_a = lower_a[[row, i]] as f64;
            let hi_a = upper_a[[row, i]] as f64;
            bias_lo_acc += lo_a * lin.constant[i] as f64;
            bias_hi_acc += hi_a * lin.constant[i] as f64;

            // IBP-validated soundness margins: widen the bias.
            let mb = margin_below[i] as f64;
            let ma = margin_above[i] as f64;
            if lo_a >= 0.0 {
                bias_lo_acc -= lo_a * mb;
            } else {
                bias_lo_acc += lo_a * ma; // lo_a < 0, so this subtracts
            }
            if hi_a >= 0.0 {
                bias_hi_acc += hi_a * ma;
            } else {
                bias_hi_acc -= hi_a * mb; // hi_a < 0, so -hi_a * mb > 0
            }
        }

        if !bias_lo_acc.is_finite() || !bias_hi_acc.is_finite() {
            bias_lower[row] = f32::NEG_INFINITY;
            bias_upper[row] = f32::INFINITY;
            for j in 0..n {
                ax_lower[[row, j]] = 0.0;
                ax_upper[[row, j]] = 0.0;
                ag_lower[[row, j]] = 0.0;
                ag_upper[[row, j]] = 0.0;
            }
        } else {
            bias_lower[row] = next_down_f32(bias_lo_acc as f32);
            bias_upper[row] = next_up_f32(bias_hi_acc as f32);
        }
    }

    let lb_x =
        LinearBounds::new_or_conservative(ax_lower, Array1::zeros(d), ax_upper, Array1::zeros(d))?;
    let lb_g =
        LinearBounds::new_or_conservative(ag_lower, Array1::zeros(d), ag_upper, Array1::zeros(d))?;
    let lb_b =
        LinearBounds::new_or_conservative(ab_lower, Array1::zeros(d), ab_upper, Array1::zeros(d))?;

    Ok((
        vec![Some(lb_x), Some(lb_g), Some(lb_b)],
        bias_lower,
        bias_upper,
    ))
}
