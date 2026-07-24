// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_dim_product, checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::shape::broadcast_shapes;
use crate::{contiguous_flat_slice, BatchedLinearBounds, LinearBounds, MulBinaryRelaxationMode};

/// Element-wise multiplication layer for two bounded tensors (e.g., SwiGLU gating).
///
/// For z = x * y where both x and y are bounded, uses IBP-style interval arithmetic:
/// - z_l = min(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
/// - z_u = max(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
#[derive(Debug, Clone)]
pub struct MulBinaryLayer;

/// Bound direction for McCormick envelope plane selection.
///
/// Shared by `mul` (element-wise) and `matmul` (batched bilinear CROWN).
#[derive(Clone, Copy)]
pub(super) enum BoundDir {
    Lower,
    Upper,
}

/// Select the tightest McCormick envelope plane for bilinear z = x·y.
///
/// Returns `(coeff_x, coeff_y, const_term)` for the affine relaxation plane.
///
/// McCormick envelope for z = x·y over box [lx, ux] × [ly, uy]:
/// - Lower planes: L1: ly·x + lx·y − lx·ly,  L2: uy·x + ux·y − ux·uy
/// - Upper planes: U1: uy·x + lx·y − lx·uy,  U2: ly·x + ux·y − ux·ly
///
/// Selection depends on the incoming CROWN weight `w` and bound direction:
/// positive weight wants tight same-direction plane, negative weight flips.
///
/// Reference: McCormick (1976), "Computability of global solutions to factorable
/// nonconvex programs". Used in auto_LiRPA/operators/bivariate.py.
#[inline]
// Justification: McCormick envelope selection requires variable bounds (lx, ux, ly, uy),
// evaluation point (x0, y0), incoming weight (w), and bound direction — 8 parameters
// that map directly to the mathematical formulation (McCormick 1976).
#[allow(clippy::too_many_arguments)]
pub(super) fn select_mccormick_plane(
    lx: f32,
    ux: f32,
    ly: f32,
    uy: f32,
    x0: f32,
    y0: f32,
    w: f32,
    dir: BoundDir,
) -> (f32, f32, f32) {
    // Non-finite guard: if any input bound or evaluation point is NaN or
    // infinite, return conservative trivial bounds rather than propagating
    // NaN coefficients. Infinity is included because McCormick products like
    // `-lx * ly` can produce NaN via `0 * inf` when one bound is zero and
    // another is infinite.
    // Lower → z >= -∞ (trivially true), Upper → z <= +∞ (trivially true).
    // Reference: auto_LiRPA/operators/bivariate.py:161-186 (softmax-only
    // output-level NaN guard sets similar trivial bounds).
    if !lx.is_finite()
        || !ux.is_finite()
        || !ly.is_finite()
        || !uy.is_finite()
        || !x0.is_finite()
        || !y0.is_finite()
    {
        return match dir {
            BoundDir::Lower => (0.0, 0.0, f32::NEG_INFINITY),
            BoundDir::Upper => (0.0, 0.0, f32::INFINITY),
        };
    }

    let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
    let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
    let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
    let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

    match dir {
        BoundDir::Lower => {
            if w >= 0.0 {
                // w * lower(z): choose the larger lower plane at reference point
                if l1.3 >= l2.3 {
                    (l1.0, l1.1, l1.2)
                } else {
                    (l2.0, l2.1, l2.2)
                }
            } else {
                // w * upper(z) for lower bound: choose the smaller upper plane
                if u1.3 <= u2.3 {
                    (u1.0, u1.1, u1.2)
                } else {
                    (u2.0, u2.1, u2.2)
                }
            }
        }
        BoundDir::Upper => {
            if w >= 0.0 {
                // w * upper(z): choose the smaller upper plane at reference point
                if u1.3 <= u2.3 {
                    (u1.0, u1.1, u1.2)
                } else {
                    (u2.0, u2.1, u2.2)
                }
            } else {
                // w * lower(z) for upper bound: choose the larger lower plane
                if l1.3 >= l2.3 {
                    (l1.0, l1.1, l1.2)
                } else {
                    (l2.0, l2.1, l2.2)
                }
            }
        }
    }
}

impl MulBinaryLayer {
    /// Propagate IBP bounds through element-wise multiplication with broadcasting.
    ///
    /// Delegates to `BoundedTensor::mul` for the core interval arithmetic:
    /// `[a,b]*[c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`.
    /// This ensures consistent NaN/Inf semantics matching the alpha-beta-CROWN reference
    /// (`auto_LiRPA/operators/bivariate.py:419-421`).
    ///
    /// Supports NumPy-style broadcasting (e.g., [1, 11, 128] * [1, 11, 1]).
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Fast path: same shape, delegate directly
        if input_a.shape() == input_b.shape() {
            return input_a.mul(input_b);
        }

        // Broadcast to common shape
        let target_shape = broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
            NyError::ShapeMismatch {
                expected: input_a.shape().to_vec(),
                got: input_b.shape().to_vec(),
            }
        })?;

        let a_lower = input_a
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?
            .to_owned();
        let a_upper = input_a
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?
            .to_owned();
        let b_lower = input_b
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?
            .to_owned();
        let b_upper = input_b
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?
            .to_owned();

        // Construct broadcast BoundedTensors (unchecked: intermediate bounds may be infinite).
        // NaN guard: broadcasting cannot introduce NaN, but if upstream bounds
        // contain NaN (from a prior new_unchecked path), catch it here.
        if a_lower.iter().any(|v| v.is_nan())
            || a_upper.iter().any(|v| v.is_nan())
            || b_lower.iter().any(|v| v.is_nan())
            || b_upper.iter().any(|v| v.is_nan())
        {
            return Err(NyError::NumericalInstability(
                "MulBinaryLayer::propagate_ibp_binary: NaN in broadcast bounds".to_string(),
            ));
        }
        let a_broadcast = BoundedTensor::new_allow_infinite(a_lower, a_upper)?;
        let b_broadcast = BoundedTensor::new_allow_infinite(b_lower, b_upper)?;

        a_broadcast.mul(&b_broadcast)
    }

    /// Compute middle relaxation coefficients for MulBinary (z = x * y).
    ///
    /// Uses auto_LiRPA's fixed-middle formulas with interpolation parameter 0.5.
    /// Returns (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) where:
    ///   z >= alpha_l * x + beta_l * y + ny_l  (lower bound)
    ///   z <= alpha_u * x + beta_u * y + ny_u  (upper bound)
    ///
    /// Reference: auto_LiRPA/operators/bivariate.py:MulHelper.interpolated_relaxation
    #[inline]
    fn compute_middle_coefficients(
        x_l: f32,
        x_u: f32,
        y_l: f32,
        y_u: f32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // auto_LiRPA middle interpolation formulas with r=0.5:
        // alpha_l = (y_l - y_u) * 0.5 + y_u = y_mid
        // beta_l  = (x_l - x_u) * 0.5 + x_u = x_mid
        // ny_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u
        // alpha_u = (y_u - y_l) * 0.5 + y_l = y_mid
        // beta_u  = (x_l - x_u) * 0.5 + x_u = x_mid
        // ny_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u

        let alpha_l = (y_l - y_u) * 0.5 + y_u;
        let beta_l = (x_l - x_u) * 0.5 + x_u;
        let ny_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u;

        let alpha_u = (y_u - y_l) * 0.5 + y_l;
        let beta_u = (x_l - x_u) * 0.5 + x_u;
        let ny_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u;

        (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u)
    }

    /// Compute interpolated McCormick coefficients for MulBinary (z = x * y)
    /// with learnable interpolation parameters r_l and r_u in [0, 1].
    ///
    /// Generalizes `compute_middle_coefficients` (which uses r=0.5).
    /// At r_l=0: L2 facet (xu, yu corner). At r_l=1: L1 facet (xl, yl corner).
    /// At r_u=0: U2 facet (xu, yl corner). At r_u=1: U1 facet (xl, yu corner).
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:40-75 (Xu et al., Appendix C,
    /// https://openreview.net/pdf?id=BJxwPJHFwS).
    #[inline]
    fn compute_interpolated_coefficients(
        x_l: f32,
        x_u: f32,
        y_l: f32,
        y_u: f32,
        r_l: f32,
        r_u: f32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // Lower bound interpolation (L2 at r=0, L1 at r=1):
        //   L1: z >= y_l*x + x_l*y - x_l*y_l
        //   L2: z >= y_u*x + x_u*y - x_u*y_u
        let alpha_l = (y_l - y_u) * r_l + y_u;
        let beta_l = (x_l - x_u) * r_l + x_u;
        let ny_l = (y_u * x_u - y_l * x_l) * r_l - y_u * x_u;

        // Upper bound interpolation (U2 at r=0, U1 at r=1):
        //   U1: z <= y_u*x + x_l*y - x_l*y_u
        //   U2: z <= y_l*x + x_u*y - x_u*y_l
        let alpha_u = (y_u - y_l) * r_u + y_l;
        let beta_u = (x_l - x_u) * r_u + x_u;
        let ny_u = (y_l * x_u - y_u * x_l) * r_u - y_l * x_u;

        (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u)
    }

    /// CROWN backward propagation for Mul (z = x * y) using McCormick or Middle relaxation.
    ///
    /// For element-wise multiplication `z[i] = x[i] * y[i]`, McCormick envelope provides
    /// sound linear bounds:
    ///
    /// Lower bounds (take max):
    ///   z ≥ x_l*y + x*y_l - x_l*y_l
    ///   z ≥ x_u*y + x*y_u - x_u*y_u
    ///
    /// Upper bounds (take min):
    ///   z ≤ x_l*y + x*y_u - x_l*y_u
    ///   z ≤ x_u*y + x*y_l - x_u*y_l
    ///
    /// Each bound is linear in (x, y), enabling CROWN backward propagation.
    /// Returns (bounds_for_a, bounds_for_b).
    ///
    /// # Arguments
    /// * `relaxation_mode` - Selects between McCormick (default, selects among 4 facets)
    ///   and Middle (fixed coefficients with interpolation parameter 0.5, matches auto_LiRPA).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        relaxation_mode: MulBinaryRelaxationMode,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let n = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let n_a = input_a_bounds.len();
        let n_b = input_b_bounds.len();

        // Verify shapes are broadcastable to the output dimension (#3499).
        // The forward pass uses NumPy-style broadcasting: [512,5] * [512,1] → [512,5].
        // CROWN backward must reduce coefficients for broadcast inputs by summing
        // along the broadcast dimensions (alpha-beta-CROWN: reduce_broadcast_dims).
        let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_a_bounds.shape().to_vec(),
                got: input_b_bounds.shape().to_vec(),
            })?;
        let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulBinary broadcast shape product overflows: {output_shape:?}"
            ))
        })?;
        if broadcast_n != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n],
                got: vec![broadcast_n],
            });
        }

        // Build broadcast index maps: for each output flat index, which input
        // flat index provides the value? Identity when no broadcast occurs.
        let a_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
        let b_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary")?;

        // Flatten bounds for element-wise indexing.
        // Defensive contiguity: reshape/broadcast can produce non-contiguous
        // views where .as_slice() returns None. Copy only when necessary.
        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        // Coefficient matrices sized to each input's actual dimension.
        // When broadcast, multiple output positions map to the same input element,
        // so coefficients accumulate with += (not =).
        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        // Certified-error accumulators mirroring crown_dense.rs (#matmul-dense-mccormick):
        // each coefficient cell is f32-accumulated with += over the inner `j` axis
        // (broadcast can map every output position to one input cell), so it carries
        // the Higham f32 accumulation error gamma_n * S that MUST reach concretize.
        // S = Sum|w * coeff| is accumulated exactly in f64 (f32*f32 is exact in f64).
        // #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        // Bias accumulation uses f64 to prevent catastrophic cancellation (#2471):
        // McCormick constant terms involve products of input bounds (e.g., -lx*ly, -ux*uy).
        // When accumulated across large inner dimensions (4096+ for SwiGLU), f32 loses
        // significant bits from alternating-sign cancellation.
        // Same pattern as crown_dense.rs and common/mod.rs (#1745).
        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        // Process each output dimension
        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b()[out_idx] as f64;
            let mut const_upper = bounds.upper_b()[out_idx] as f64;

            // For each output element position (element-wise: z[j] = x[a_j] * y[b_j]
            // where a_j/b_j follow broadcast mapping).
            for j in 0..n {
                let w_lower = bounds.lower_a()[[out_idx, j]];
                let w_upper = bounds.upper_a()[[out_idx, j]];

                // Broadcast-aware indexing (#3499): a_idx/b_idx map the output
                // flat index j to the corresponding input element. For non-broadcast
                // inputs these are identity (a_idx == j); for broadcast inputs
                // (e.g., SE block [512,1] → [512,5]) multiple j values map to
                // the same input element.
                let a_idx = a_idx_map[j];
                let b_idx = b_idx_map[j];
                let lx = a_lower_flat[a_idx];
                let ux = a_upper_flat[a_idx];
                let ly = b_lower_flat[b_idx];
                let uy = b_upper_flat[b_idx];

                match relaxation_mode {
                    MulBinaryRelaxationMode::McCormick => {
                        // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                        #[allow(clippy::manual_midpoint)]
                        let x0 = (lx + ux) * 0.5;
                        #[allow(clippy::manual_midpoint)]
                        let y0 = (ly + uy) * 0.5;

                        // Select McCormick plane for lower bound computation
                        let (ax_l, ay_l, c_l) = select_mccormick_plane(
                            lx,
                            ux,
                            ly,
                            uy,
                            x0,
                            y0,
                            w_lower,
                            BoundDir::Lower,
                        );
                        // Accumulate with += for broadcast: multiple output positions
                        // map to the same input element, coefficients sum.
                        lower_a_a[[out_idx, a_idx]] += w_lower * ax_l;
                        lower_a_b[[out_idx, b_idx]] += w_lower * ay_l;
                        lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (ax_l as f64).abs();
                        lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (ay_l as f64).abs();
                        const_lower += w_lower as f64 * c_l as f64;

                        // Select McCormick plane for upper bound computation
                        let (ax_u, ay_u, c_u) = select_mccormick_plane(
                            lx,
                            ux,
                            ly,
                            uy,
                            x0,
                            y0,
                            w_upper,
                            BoundDir::Upper,
                        );
                        upper_a_a[[out_idx, a_idx]] += w_upper * ax_u;
                        upper_a_b[[out_idx, b_idx]] += w_upper * ay_u;
                        upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (ax_u as f64).abs();
                        upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (ay_u as f64).abs();
                        const_upper += w_upper as f64 * c_u as f64;
                    }
                    MulBinaryRelaxationMode::Middle => {
                        // Middle relaxation: fixed coefficients with interpolation r=0.5
                        // z_lower >= alpha_l * x + beta_l * y + ny_l
                        // z_upper <= alpha_u * x + beta_u * y + ny_u
                        let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                            Self::compute_middle_coefficients(lx, ux, ly, uy);

                        // For lower bound accumulation, select based on weight sign:
                        // w >= 0: use lower bound of z -> alpha_l, beta_l, ny_l
                        // w < 0:  use upper bound of z -> alpha_u, beta_u, ny_u
                        if w_lower >= 0.0 {
                            lower_a_a[[out_idx, a_idx]] += w_lower * alpha_l;
                            lower_a_b[[out_idx, b_idx]] += w_lower * beta_l;
                            lower_s_a[[out_idx, a_idx]] +=
                                (w_lower as f64).abs() * (alpha_l as f64).abs();
                            lower_s_b[[out_idx, b_idx]] +=
                                (w_lower as f64).abs() * (beta_l as f64).abs();
                            const_lower += w_lower as f64 * ny_l as f64;
                        } else {
                            lower_a_a[[out_idx, a_idx]] += w_lower * alpha_u;
                            lower_a_b[[out_idx, b_idx]] += w_lower * beta_u;
                            lower_s_a[[out_idx, a_idx]] +=
                                (w_lower as f64).abs() * (alpha_u as f64).abs();
                            lower_s_b[[out_idx, b_idx]] +=
                                (w_lower as f64).abs() * (beta_u as f64).abs();
                            const_lower += w_lower as f64 * ny_u as f64;
                        }

                        // For upper bound accumulation, select based on weight sign:
                        // w >= 0: use upper bound of z -> alpha_u, beta_u, ny_u
                        // w < 0:  use lower bound of z -> alpha_l, beta_l, ny_l
                        if w_upper >= 0.0 {
                            upper_a_a[[out_idx, a_idx]] += w_upper * alpha_u;
                            upper_a_b[[out_idx, b_idx]] += w_upper * beta_u;
                            upper_s_a[[out_idx, a_idx]] +=
                                (w_upper as f64).abs() * (alpha_u as f64).abs();
                            upper_s_b[[out_idx, b_idx]] +=
                                (w_upper as f64).abs() * (beta_u as f64).abs();
                            const_upper += w_upper as f64 * ny_u as f64;
                        } else {
                            upper_a_a[[out_idx, a_idx]] += w_upper * alpha_l;
                            upper_a_b[[out_idx, b_idx]] += w_upper * beta_l;
                            upper_s_a[[out_idx, a_idx]] +=
                                (w_upper as f64).abs() * (alpha_l as f64).abs();
                            upper_s_b[[out_idx, b_idx]] +=
                                (w_upper as f64).abs() * (beta_l as f64).abs();
                            const_upper += w_upper as f64 * ny_l as f64;
                        }
                    }
                }
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        // Apply directed rounding on f64→f32 downcast (#2471):
        // lower bounds round toward -inf, upper bounds round toward +inf.
        // This ensures stored f32 bounds always contain the true mathematical value.
        let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
        let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

        // Certified coefficient error mirroring crown_dense.rs (#matmul-dense-mccormick).
        // Every coefficient cell `[out_idx, idx]` is f32-accumulated with += only inside
        // the inner `for j in 0..n` loop (out_idx is fixed by the outer loop). Each j does
        // at most one += into a given cell, and broadcast index maps can land every one of
        // the n iterations in the SAME cell, so n is a sound UPPER BOUND on the per-cell
        // f32-addition depth for BOTH the a-side and b-side. |stored - exact| <=
        // gamma_n_f32(n) * S (Higham f32 unit roundoff), rounded UP to a sound f32 and
        // carried so concretize penalizes outward.
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(n);
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma * s) as f32));

        // MulBinary returns two affine forms (for x and y) but only one shared
        // McCormick constant term. Keep the full bias on bounds_a and zero
        // bounds_b bias so DAG accumulation counts the constant exactly once (#2520).
        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_f32,
            upper_a_a,
            upper_b_f32,
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            Array1::zeros(num_outputs),
            upper_a_b,
            Array1::zeros(num_outputs),
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// CROWN backward for MulBinary with alpha-parameterized McCormick interpolation.
    ///
    /// When `alphas` is `Some`, uses per-element interpolation parameters
    /// `r_l = alphas[[0, j]]`, `r_u = alphas[[1, j]]` (shape `[2, n]`) to select
    /// between McCormick envelope facets. When `None`, delegates to fixed McCormick.
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:40-75 (Xu et al., Appendix C).
    pub fn propagate_linear_binary_with_alpha(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&Array2<f32>>,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let alphas = match alphas {
            Some(a) => a,
            None => {
                return self.propagate_linear_binary(
                    bounds,
                    input_a_bounds,
                    input_b_bounds,
                    MulBinaryRelaxationMode::McCormick,
                );
            }
        };

        let n = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let n_a = input_a_bounds.len();
        let n_b = input_b_bounds.len();

        if alphas.shape() != [2, n] {
            return Err(NyError::ShapeMismatch {
                expected: vec![2, n],
                got: alphas.shape().to_vec(),
            });
        }

        let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_a_bounds.shape().to_vec(),
                got: input_b_bounds.shape().to_vec(),
            })?;
        let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulBinary alpha broadcast shape product overflows: {output_shape:?}"
            ))
        })?;
        if broadcast_n != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n],
                got: vec![broadcast_n],
            });
        }

        let a_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
        let b_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary alpha")?;

        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        // Certified-error abssum accumulators (#mul-mccormick-alpha), mirroring
        // crown_dense.rs: each coefficient cell is f32-accumulated over the j axis
        // (the only inner loop), so it carries the Higham f32 accumulation error
        // gamma_depth * S that MUST reach concretize. S = Sum|w * coeff| is summed
        // exactly in f64 (f32*f32 is exact in f64). #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, n_b));

        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b()[out_idx] as f64;
            let mut const_upper = bounds.upper_b()[out_idx] as f64;

            for j in 0..n {
                let w_lower = bounds.lower_a()[[out_idx, j]];
                let w_upper = bounds.upper_a()[[out_idx, j]];

                let a_idx = a_idx_map[j];
                let b_idx = b_idx_map[j];

                let lx = a_lower_flat[a_idx];
                let ux = a_upper_flat[a_idx];
                let ly = b_lower_flat[b_idx];
                let uy = b_upper_flat[b_idx];

                // Non-finite guard: same as select_mccormick_plane.
                if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
                    // Trivial bounds: lower → -inf, upper → +inf.
                    if w_lower != 0.0 {
                        const_lower = f64::NEG_INFINITY;
                    }
                    if w_upper != 0.0 {
                        const_upper = f64::INFINITY;
                    }
                    continue;
                }

                let r_l = alphas[[0, j]].clamp(0.0, 1.0);
                let r_u = alphas[[1, j]].clamp(0.0, 1.0);

                let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                    Self::compute_interpolated_coefficients(lx, ux, ly, uy, r_l, r_u);

                // Weight-sign splitting: w >= 0 uses same-direction relaxation,
                // w < 0 uses opposite-direction relaxation.
                if w_lower >= 0.0 {
                    lower_a_a[[out_idx, a_idx]] += w_lower * alpha_l;
                    lower_a_b[[out_idx, b_idx]] += w_lower * beta_l;
                    lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (alpha_l as f64).abs();
                    lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (beta_l as f64).abs();
                    const_lower += w_lower as f64 * ny_l as f64;
                } else {
                    lower_a_a[[out_idx, a_idx]] += w_lower * alpha_u;
                    lower_a_b[[out_idx, b_idx]] += w_lower * beta_u;
                    lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (alpha_u as f64).abs();
                    lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (beta_u as f64).abs();
                    const_lower += w_lower as f64 * ny_u as f64;
                }

                if w_upper >= 0.0 {
                    upper_a_a[[out_idx, a_idx]] += w_upper * alpha_u;
                    upper_a_b[[out_idx, b_idx]] += w_upper * beta_u;
                    upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (alpha_u as f64).abs();
                    upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (beta_u as f64).abs();
                    const_upper += w_upper as f64 * ny_u as f64;
                } else {
                    upper_a_a[[out_idx, a_idx]] += w_upper * alpha_l;
                    upper_a_b[[out_idx, b_idx]] += w_upper * beta_l;
                    upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (alpha_l as f64).abs();
                    upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (beta_l as f64).abs();
                    const_upper += w_upper as f64 * ny_l as f64;
                }
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
        let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

        // Certified coefficient error (#mul-mccormick-alpha), mirroring crown_dense.rs.
        // The only inner loop is `for j in 0..n`; the destination cell column is
        // a_idx = a_idx_map[j] (A side) / b_idx = b_idx_map[j] (B side), with out_idx
        // fixed by the outer loop. The number of f32 += landing in one A-side cell is
        // the broadcast multiplicity of that input element, which NumPy broadcasting
        // replicates uniformly = n / n_a; n.div_ceil(n_a) is a conservative UPPER bound
        // (== n/n_a when n_a | n, never an under-count). B side: n.div_ceil(n_b).
        // |stored - exact| <= gamma_depth_f32 * S (Higham f32 unit roundoff), with
        // S = Sum|w * coeff| accumulated exactly in f64, rounded UP to a sound f32 and
        // carried so concretize penalizes outward. #vnncomp-aw-soundness.
        let depth_a = n.div_ceil(n_a.max(1)).max(1);
        let depth_b = n.div_ceil(n_b.max(1)).max(1);
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(depth_a);
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(depth_b);
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_f32,
            upper_a_a,
            upper_b_f32,
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            Array1::zeros(num_outputs),
            upper_a_b,
            Array1::zeros(num_outputs),
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for MulBinary (z = x * y) with McCormick or Middle relaxation.
    ///
    /// Same as `propagate_linear_binary` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    ///
    /// For element-wise `z[i] = x[i] * y[i]`, uses McCormick relaxation:
    ///   Lower bounds (take max):
    ///     z ≥ x_l·y + x·y_l - x_l·y_l
    ///     z ≥ x_u·y + x·y_u - x_u·y_u
    ///   Upper bounds (take min):
    ///     z ≤ x_l·y + x·y_u - x_l·y_u
    ///     z ≤ x_u·y + x·y_l - x_u·y_l
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    ///
    /// # Arguments
    /// * `relaxation_mode` - Selects between McCormick (default, selects among 4 facets)
    ///   and Middle (fixed coefficients with interpolation parameter 0.5, matches auto_LiRPA).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        relaxation_mode: MulBinaryRelaxationMode,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        // Get shapes
        let a_shape = bounds.lower_a.shape();
        let ndim = a_shape.len();

        // For element-wise multiplication, the last dimension is the "n" (features)
        // BatchedLinearBounds shape: [...batch, out_dim, n]
        // where out_dim == n for element-wise operations (identity CROWN)
        if ndim < 2 {
            return Err(NyError::ShapeMismatch {
                expected: vec![2], // at least 2D
                got: a_shape.to_vec(),
            });
        }

        let n = a_shape[ndim - 1]; // last dimension is input features
        let out_dim = a_shape[ndim - 2]; // second-to-last is output dimension

        // Verify input bounds shapes match
        if input_a_bounds.len() != input_b_bounds.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![input_a_bounds.len()],
                got: vec![input_b_bounds.len()],
            });
        }

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary batched")?;

        // Flatten bounds for element-wise indexing.
        // Defensive contiguity: reshape/broadcast can produce non-contiguous
        // views where .as_slice() returns None. Copy only when necessary.
        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        // Calculate batch size (all dimensions except last two)
        let batch_dims = &a_shape[..ndim - 2];
        let batch_size: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MulBinary batched CROWN: batch dimensions {batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Flatten input arrays to work with them more easily
        let lower_a_flat = contiguous_flat_slice(&bounds.lower_a);
        let upper_a_flat = contiguous_flat_slice(&bounds.upper_a);
        let lower_b_flat = contiguous_flat_slice(&bounds.lower_b);
        let upper_b_flat = contiguous_flat_slice(&bounds.upper_b);

        // Output arrays
        let coeff_len = checked_dim_product(&[batch_size, out_dim, n], "MulBinary batched CROWN")?;
        let bias_len = checked_dim_product(
            &[batch_size, out_dim],
            "MulBinary batched CROWN bias buffers",
        )?;
        let mut lower_a_a = vec![0.0_f32; coeff_len];
        let mut lower_a_b = vec![0.0_f32; coeff_len];
        let mut upper_a_a = vec![0.0_f32; coeff_len];
        let mut upper_a_b = vec![0.0_f32; coeff_len];
        // Certified coefficient-error abssum accumulators (#mul-batched-mccormick).
        // Unlike the broadcast non-batched path (which `+=`-accumulates via index maps),
        // every coeff cell here is written EXACTLY ONCE with `=`: the dest index
        // a_flat_idx = batch_idx*out_dim*n + out_idx*n + j is a bijection of the three
        // enclosing loops (batch_idx, out_idx, j), so no inner reduction axis collides
        // into a cell and the per-cell f32-rounding depth is exactly 1 (a single
        // f32*f32 multiply stored to f32, which still rounds-to-nearest and can land
        // TIGHTER than the true real product). S = |w * coeff| is exact in f64
        // (f32*f32 is exact in f64); err = gamma_1 * S is carried via set_coeff_err so
        // concretize penalizes outward, mirroring crown_dense.rs / crown_batched.rs.
        // #vnncomp-aw-soundness.
        let mut s_lower_a_a = vec![0.0_f64; coeff_len];
        let mut s_lower_a_b = vec![0.0_f64; coeff_len];
        let mut s_upper_a_a = vec![0.0_f64; coeff_len];
        let mut s_upper_a_b = vec![0.0_f64; coeff_len];
        // Bias accumulation in f64 to prevent catastrophic cancellation (#2471).
        // Same rationale as non-batched path above and crown_dense.rs (#1745).
        let mut lower_b_out = vec![0.0_f64; bias_len];
        let mut upper_b_out = vec![0.0_f64; bias_len];

        // Input bounds may or may not have a batch dimension.
        // If input_len == n, the same bounds are broadcast over all batches.
        // If input_len == batch_size * n, each batch has its own bounds slice.
        let input_len = a_lower_flat.len();
        let input_is_batched = input_len != n;
        if input_is_batched && input_len != batch_size * n {
            return Err(NyError::ShapeMismatch {
                expected: vec![batch_size * n],
                got: vec![input_len],
            });
        }

        // Process each batch position
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_dim {
                let out_flat_idx = batch_idx * out_dim + out_idx;
                let mut const_lower = lower_b_flat[out_flat_idx] as f64;
                let mut const_upper = upper_b_flat[out_flat_idx] as f64;

                for j in 0..n {
                    let a_flat_idx = batch_idx * out_dim * n + out_idx * n + j;
                    let w_lower = lower_a_flat[a_flat_idx];
                    let w_upper = upper_a_flat[a_flat_idx];

                    // Index into input bounds: broadcast (reuse) or batch-offset
                    let input_j = if input_is_batched {
                        batch_idx * n + j
                    } else {
                        j
                    };

                    let lx = a_lower_flat[input_j];
                    let ux = a_upper_flat[input_j];
                    let ly = b_lower_flat[input_j];
                    let uy = b_upper_flat[input_j];

                    match relaxation_mode {
                        MulBinaryRelaxationMode::McCormick => {
                            // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                            #[allow(clippy::manual_midpoint)]
                            let x0 = (lx + ux) * 0.5;
                            #[allow(clippy::manual_midpoint)]
                            let y0 = (ly + uy) * 0.5;

                            // Select McCormick plane for lower bound
                            let (ax_l, ay_l, c_l) = select_mccormick_plane(
                                lx,
                                ux,
                                ly,
                                uy,
                                x0,
                                y0,
                                w_lower,
                                BoundDir::Lower,
                            );
                            lower_a_a[a_flat_idx] = w_lower * ax_l;
                            lower_a_b[a_flat_idx] = w_lower * ay_l;
                            s_lower_a_a[a_flat_idx] = (w_lower as f64).abs() * (ax_l as f64).abs();
                            s_lower_a_b[a_flat_idx] = (w_lower as f64).abs() * (ay_l as f64).abs();
                            const_lower += w_lower as f64 * c_l as f64;

                            // Select McCormick plane for upper bound
                            let (ax_u, ay_u, c_u) = select_mccormick_plane(
                                lx,
                                ux,
                                ly,
                                uy,
                                x0,
                                y0,
                                w_upper,
                                BoundDir::Upper,
                            );
                            upper_a_a[a_flat_idx] = w_upper * ax_u;
                            upper_a_b[a_flat_idx] = w_upper * ay_u;
                            s_upper_a_a[a_flat_idx] = (w_upper as f64).abs() * (ax_u as f64).abs();
                            s_upper_a_b[a_flat_idx] = (w_upper as f64).abs() * (ay_u as f64).abs();
                            const_upper += w_upper as f64 * c_u as f64;
                        }
                        MulBinaryRelaxationMode::Middle => {
                            // Middle relaxation: fixed coefficients with interpolation r=0.5
                            let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                                Self::compute_middle_coefficients(lx, ux, ly, uy);

                            // For lower bound accumulation, select based on weight sign
                            if w_lower >= 0.0 {
                                lower_a_a[a_flat_idx] = w_lower * alpha_l;
                                lower_a_b[a_flat_idx] = w_lower * beta_l;
                                s_lower_a_a[a_flat_idx] =
                                    (w_lower as f64).abs() * (alpha_l as f64).abs();
                                s_lower_a_b[a_flat_idx] =
                                    (w_lower as f64).abs() * (beta_l as f64).abs();
                                const_lower += w_lower as f64 * ny_l as f64;
                            } else {
                                lower_a_a[a_flat_idx] = w_lower * alpha_u;
                                lower_a_b[a_flat_idx] = w_lower * beta_u;
                                s_lower_a_a[a_flat_idx] =
                                    (w_lower as f64).abs() * (alpha_u as f64).abs();
                                s_lower_a_b[a_flat_idx] =
                                    (w_lower as f64).abs() * (beta_u as f64).abs();
                                const_lower += w_lower as f64 * ny_u as f64;
                            }

                            // For upper bound accumulation, select based on weight sign
                            if w_upper >= 0.0 {
                                upper_a_a[a_flat_idx] = w_upper * alpha_u;
                                upper_a_b[a_flat_idx] = w_upper * beta_u;
                                s_upper_a_a[a_flat_idx] =
                                    (w_upper as f64).abs() * (alpha_u as f64).abs();
                                s_upper_a_b[a_flat_idx] =
                                    (w_upper as f64).abs() * (beta_u as f64).abs();
                                const_upper += w_upper as f64 * ny_u as f64;
                            } else {
                                upper_a_a[a_flat_idx] = w_upper * alpha_l;
                                upper_a_b[a_flat_idx] = w_upper * beta_l;
                                s_upper_a_a[a_flat_idx] =
                                    (w_upper as f64).abs() * (alpha_l as f64).abs();
                                s_upper_a_b[a_flat_idx] =
                                    (w_upper as f64).abs() * (beta_l as f64).abs();
                                const_upper += w_upper as f64 * ny_l as f64;
                            }
                        }
                    }
                }

                lower_b_out[out_flat_idx] = const_lower;
                upper_b_out[out_flat_idx] = const_upper;
            }
        }

        // Apply directed rounding on f64→f32 downcast (#2471):
        // lower bounds round toward -inf, upper bounds round toward +inf.
        let lower_b_f32: Vec<f32> = lower_b_out
            .iter()
            .map(|&v| next_down_f32(v as f32))
            .collect();
        let upper_b_f32: Vec<f32> = upper_b_out.iter().map(|&v| next_up_f32(v as f32)).collect();

        // Certified coefficient error (#mul-batched-mccormick). Per-cell f32-rounding
        // depth is exactly 1: each coeff cell is written with a single `=` multiply and
        // a_flat_idx = batch_idx*out_dim*n + out_idx*n + j bijects the three enclosing
        // loops (no reduction axis holds a_flat_idx fixed), so gamma_1 upper-bounds the
        // round-to-nearest error of the stored f32 product. err = gamma_1 * S rounded UP,
        // carried via set_coeff_err so concretize penalizes outward (mirrors
        // crown_dense.rs / crown_batched.rs). #vnncomp-aw-soundness.
        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        let mk_err = |s: &[f64]| -> Result<ArrayD<f32>> {
            let v: Vec<f32> = s
                .iter()
                .map(|&sv| next_up_f32((gamma1 * sv) as f32))
                .collect();
            let len = v.len();
            ArrayD::from_shape_vec(IxDyn(a_shape), v).map_err(|_| NyError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: vec![len],
            })
        };
        let lower_a_a_err = mk_err(&s_lower_a_a)?;
        let upper_a_a_err = mk_err(&s_upper_a_a)?;
        let lower_a_b_err = mk_err(&s_lower_a_b)?;
        let upper_a_b_err = mk_err(&s_upper_a_b)?;

        // Reshape outputs to batched form
        let lower_b_bias_shape = IxDyn(bounds.lower_b.shape());
        let upper_b_bias_shape = IxDyn(bounds.upper_b.shape());
        // Phase 4 audit: per-layer McCormick output — catches NaN from upstream via McCormick.
        // Capture vec lengths before from_shape_vec moves them (#3110 pattern).
        let lower_a_a_len = lower_a_a.len();
        let lower_b_f32_len = lower_b_f32.len();
        let upper_a_a_len = upper_a_a.len();
        let upper_b_f32_len = upper_b_f32.len();
        let lower_a_b_len = lower_a_b.len();
        let upper_a_b_len = upper_a_b.len();
        let mut bounds_a = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_a).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![lower_a_a_len],
                }
            })?,
            ArrayD::from_shape_vec(lower_b_bias_shape.clone(), lower_b_f32).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: bounds.lower_b.shape().to_vec(),
                    got: vec![lower_b_f32_len],
                }
            })?,
            ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_a).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![upper_a_a_len],
                }
            })?,
            ArrayD::from_shape_vec(upper_b_bias_shape.clone(), upper_b_f32).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: bounds.upper_b.shape().to_vec(),
                    got: vec![upper_b_f32_len],
                }
            })?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        bounds_a.set_coeff_err(lower_a_a_err, upper_a_a_err);

        let mut bounds_b = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_b).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![lower_a_b_len],
                }
            })?,
            ArrayD::zeros(lower_b_bias_shape),
            ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_b).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![upper_a_b_len],
                }
            })?,
            ArrayD::zeros(upper_b_bias_shape),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        bounds_b.set_coeff_err(lower_a_b_err, upper_a_b_err);

        Ok((bounds_a, bounds_b))
    }
}

#[cfg(test)]
mod tests;
