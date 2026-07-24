// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::matmul::decode_batch_index_into_buf;
use super::BilinearCrownLayer;
use crate::bounds::{nan_propagating_max, nan_propagating_min};

impl BilinearCrownLayer {
    /// N-D one-sided CROWN compose through BilinearCrown (#286).
    ///
    /// When the downstream has N-D structure (last dim = n, not m*n), composes
    /// along the n dimension only:
    /// - Q path: full linear bounds [batch..., m, out, k] preserving tensor structure
    /// - K path: eagerly concretized via IBP bounds → folded into Q bias
    ///
    /// This is strictly tighter than partial CROWN fallback (which concretizes
    /// both Q and K) because Q retains CROWN linear structure through subsequent
    /// layers.
    ///
    /// # McCormick composition math
    ///
    /// For z[i,j] = sum_l Q[i,l]*K^T[l,j] (i.e. transpose_b):
    ///   z[i,j] >= alpha_l[i,j,:] @ Q[i,:] + beta_l[i,j,:] @ K[j,:] + ny_l[i,j]
    ///
    /// With downstream A[batch,i,o,j] (N-D structure, last dim = n):
    ///   Q coeff[batch,i,o,l] = sum_j interval_compose(ds_A[batch,i,o,j], alpha[i,j,l])
    ///   K contrib[batch,i,o] = sum_j sum_l interval_compose(ds_A[batch,i,o,j] * beta[i,j,l]) * K[j,l]
    ///   ny contrib[batch,i,o] = sum_j interval_compose(ds_A[batch,i,o,j], ny[i,j])
    ///
    /// Reference: auto_LiRPA operators/bivariate.py `bound_backward_both_perturbed`
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propagate_nd_one_sided(
        &self,
        downstream: &crate::BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        m: usize,
        n: usize,
        k: usize,
        scale: f32,
        a_batch: &[usize],
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        use crate::bounds::safe_math::{interval_mul_for_bounds, sign_split_compose_for_bounds};

        // Validate downstream has N-D structure
        let ds_a_shape = downstream.lower_a().shape();
        let ds_ndim = ds_a_shape.len();
        if ds_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "BilinearCrown N-D compose: downstream must be >= 2D".to_string(),
            ));
        }
        let ds_last = ds_a_shape[ds_ndim - 1];
        let out_dim = ds_a_shape[ds_ndim - 2];
        let ds_batch = &ds_a_shape[..ds_ndim - 2];

        if ds_last != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n],
                got: vec![ds_last],
            });
        }

        // The downstream batch should include the m dimension (from N-D identity).
        // Expected: [B, H, m] or similar where last batch dim = m.
        let ds_batch_last = ds_batch.last().copied().unwrap_or(0);
        if ds_batch_last != m {
            return Err(NyError::ShapeMismatch {
                expected: vec![m],
                got: vec![ds_batch_last],
            });
        }
        let outer_batch = &ds_batch[..ds_batch.len() - 1];
        let outer_batch_size: usize = checked_shape_product(outer_batch).ok_or_else(|| {
            NyError::InvalidSpec("BilinearCrown N-D compose: outer batch overflow".to_string())
        })?;
        // Empty outer_batch slice yields product=1 (single batch position), which is correct.
        // A zero-valued dimension means zero elements — cannot compose with empty downstream.
        if outer_batch_size == 0 {
            return Err(NyError::InvalidSpec(
                "BilinearCrown N-D compose: zero-element outer batch dimension".to_string(),
            ));
        }

        // Flatten downstream A to [outer_batch, m, out, n] for iteration
        let ds_lower_a = downstream
            .lower_a()
            .view()
            .into_shape_with_order((outer_batch_size, m, out_dim, n))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearCrown N-D compose: reshape ds_lower_a: {e}"
                ))
            })?;
        let ds_upper_a = downstream
            .upper_a()
            .view()
            .into_shape_with_order((outer_batch_size, m, out_dim, n))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearCrown N-D compose: reshape ds_upper_a: {e}"
                ))
            })?;

        // Compute batch-reduced global intervals for McCormick planes
        // (same algorithm as the dense path)
        let batch_size: usize = checked_shape_product(a_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown N-D: batch dims {a_batch:?} overflow",
            ))
        })?;
        if batch_size == 0 {
            return Err(NyError::InvalidSpec(
                "BilinearCrown N-D compose: zero-element input batch".to_string(),
            ));
        }
        let batch_index_len = a_batch.len();
        // Stack-allocated index buffers (#2237 F4).
        assert!(
            batch_index_len + 2 <= 8,
            "BilinearCrown N-D: batch_index_len + 2 exceeds stack buffer"
        );
        let mut a_idx = [0usize; 8];
        let mut b_idx = [0usize; 8];
        let idx_len = batch_index_len + 2;

        // Q output: [outer_batch, m, out, k]
        let mut q_lower_a = ndarray::Array4::<f32>::zeros((outer_batch_size, m, out_dim, k));
        let mut q_upper_a = ndarray::Array4::<f32>::zeros((outer_batch_size, m, out_dim, k));
        // Certified-error abssum accumulators for the Q coefficients
        // (#vnncomp-aw-soundness). Each cell [ob,i,o,l] is f32-accumulated over
        // the j axis (depth = n: the j loop is the ONLY enclosing loop that
        // varies while [ob,i,o,l] stays fixed), so it carries the Higham f32
        // accumulation error gamma_n(n) * S. S = Sum_j |q_prod| accumulated
        // exactly in f64 (each q_prod is already a single f32, so abs as f64 is
        // exact). Mirrors crown_dense.rs lower_s_a / upper_s_a.
        let mut q_lower_s = ndarray::Array4::<f64>::zeros((outer_batch_size, m, out_dim, k));
        let mut q_upper_s = ndarray::Array4::<f64>::zeros((outer_batch_size, m, out_dim, k));
        // Bias: [outer_batch, m, out] — includes K concretized + ny + ds_bias
        let mut bias_lower = ndarray::Array3::<f64>::zeros((outer_batch_size, m, out_dim));
        let mut bias_upper = ndarray::Array3::<f64>::zeros((outer_batch_size, m, out_dim));

        for i in 0..m {
            for j in 0..n {
                // Compute batch-reduced intervals per contraction index l.
                // Each McCormick term z[i,j,l] = q[i,l] * k[l,j] uses its own
                // interval bounds Q[i,l] ∈ [q_l, q_u] and K[j,l] ∈ [k_l, k_u],
                // reduced only across batch positions (not across l).
                // This is tighter than the previous per-(i,j) global approach
                // which reduced over both batch AND l.
                // Reference: auto_LiRPA bivariate.py computes element-wise
                // McCormick coefficients — one per (i,j,l) triple.
                let mut q_l_per_l = vec![f32::INFINITY; k];
                let mut q_u_per_l = vec![f32::NEG_INFINITY; k];
                let mut k_l_per_l = vec![f32::INFINITY; k];
                let mut k_u_per_l = vec![f32::NEG_INFINITY; k];

                for batch_idx in 0..batch_size {
                    decode_batch_index_into_buf(batch_idx, a_batch, &mut a_idx[..batch_index_len])?;
                    b_idx[..batch_index_len].copy_from_slice(&a_idx[..batch_index_len]);
                    a_idx[batch_index_len] = i;
                    a_idx[batch_index_len + 1] = 0;
                    b_idx[batch_index_len] = 0;
                    b_idx[batch_index_len + 1] = 0;
                    if self.transpose_b {
                        b_idx[batch_index_len] = j;
                    } else {
                        b_idx[batch_index_len + 1] = j;
                    }

                    for l in 0..k {
                        a_idx[batch_index_len + 1] = l;
                        let ql = input_a_bounds.lower()[&a_idx[..idx_len]];
                        let qu = input_a_bounds.upper()[&a_idx[..idx_len]];
                        q_l_per_l[l] = nan_propagating_min(q_l_per_l[l], ql);
                        q_u_per_l[l] = nan_propagating_max(q_u_per_l[l], qu);

                        if self.transpose_b {
                            b_idx[batch_index_len + 1] = l;
                        } else {
                            b_idx[batch_index_len] = l;
                        }
                        let kl = input_b_bounds.lower()[&b_idx[..idx_len]];
                        let ku = input_b_bounds.upper()[&b_idx[..idx_len]];
                        k_l_per_l[l] = nan_propagating_min(k_l_per_l[l], kl);
                        k_u_per_l[l] = nan_propagating_max(k_u_per_l[l], ku);
                    }
                }

                // McCormick plane selection per contraction index l
                for l in 0..k {
                    let q_l = q_l_per_l[l];
                    let q_u = q_u_per_l[l];
                    let k_l = k_l_per_l[l];
                    let k_u = k_u_per_l[l];

                    // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let q0 = (q_l + q_u) * 0.5;
                    #[allow(clippy::manual_midpoint)]
                    let k0 = (k_l + k_u) * 0.5;

                    // Lower plane: tighter of L1 (uses k_l, q_l) and L2 (uses k_u, q_u)
                    let l1_val = k_l * q0 + q_l * k0 - q_l * k_l;
                    let l2_val = k_u * q0 + q_u * k0 - q_u * k_u;
                    let (ax_l, ay_l, c_l) = if l1_val >= l2_val {
                        (k_l, q_l, -q_l * k_l)
                    } else {
                        (k_u, q_u, -q_u * k_u)
                    };

                    // Upper plane: tighter of U1 (uses k_u, q_l) and U2 (uses k_l, q_u)
                    let u1_val = k_u * q0 + q_l * k0 - q_l * k_u;
                    let u2_val = k_l * q0 + q_u * k0 - q_u * k_l;
                    let (ax_u, ay_u, c_u) = if u1_val <= u2_val {
                        (k_u, q_l, -q_l * k_u)
                    } else {
                        (k_l, q_u, -q_u * k_l)
                    };

                    // Scale coefficients
                    let alpha_l = scale * ax_l;
                    let alpha_u = scale * ax_u;
                    let beta_l = scale * ay_l;
                    let beta_u = scale * ay_u;
                    let ny_l = scale * c_l;
                    let ny_u = scale * c_u;

                    // Compose with downstream along j dimension
                    for ob in 0..outer_batch_size {
                        for o in 0..out_dim {
                            let ds_l = ds_lower_a[[ob, i, o, j]];
                            let ds_u = ds_upper_a[[ob, i, o, j]];

                            // Q path: sign-split compose(ds_A, [alpha_l, alpha_u]).
                            // ds_l / ds_u are KNOWN downstream coefficients for the
                            // lower / upper bound directions, and (alpha_l, alpha_u)
                            // are the McCormick lower / upper plane Q-slopes. This is
                            // the tight sign-split (same rule as broadcast.rs:174-194),
                            // <= the 4-corner interval product while staying sound.
                            let (q_prod_l, q_prod_u) =
                                sign_split_compose_for_bounds(ds_l, ds_u, alpha_l, alpha_u);
                            q_lower_a[[ob, i, o, l]] += q_prod_l;
                            q_upper_a[[ob, i, o, l]] += q_prod_u;
                            // Sum of absolute per-term coefficients into the same
                            // [ob,i,o,l] cell (f32->f64 abs is exact), the base for
                            // the certified f32-accumulation error (#vnncomp-aw-soundness).
                            q_lower_s[[ob, i, o, l]] += (q_prod_l as f64).abs();
                            q_upper_s[[ob, i, o, l]] += (q_prod_u as f64).abs();

                            // K path: eagerly concretize.
                            // Stage 1: sign-split compose(ds_A, [beta_l, beta_u]) — the
                            // beta plane slopes are known per-direction coefficients,
                            // so this mirrors the broadcast.rs beta path.
                            let (beta_prod_l, beta_prod_u) =
                                sign_split_compose_for_bounds(ds_l, ds_u, beta_l, beta_u);
                            // Stage 2: K[j,l] ∈ [k_l, k_u] is a genuine input INTERVAL,
                            // so the product against [beta_prod_l, beta_prod_u] is a true
                            // interval-interval product (4-corner) — keep it.
                            let (k_conc_l, k_conc_u) =
                                interval_mul_for_bounds(beta_prod_l, beta_prod_u, k_l, k_u);
                            bias_lower[[ob, i, o]] += k_conc_l as f64;
                            bias_upper[[ob, i, o]] += k_conc_u as f64;

                            // Ny path: sign-split compose(ds_A, [ny_l, ny_u]) — each
                            // McCormick term l contributes its own lower / upper plane
                            // bias (varies per l with per-element intervals).
                            let (g_prod_l, g_prod_u) =
                                sign_split_compose_for_bounds(ds_l, ds_u, ny_l, ny_u);
                            bias_lower[[ob, i, o]] += g_prod_l as f64;
                            bias_upper[[ob, i, o]] += g_prod_u as f64;
                        }
                    }
                }
            }
        }

        // Apply directed rounding on Q coefficients
        q_lower_a.mapv_inplace(|v| {
            if v.is_nan() {
                f32::NEG_INFINITY
            } else {
                next_down_f32(v)
            }
        });
        q_upper_a.mapv_inplace(|v| {
            if v.is_nan() {
                f32::INFINITY
            } else {
                next_up_f32(v)
            }
        });

        // Add downstream bias and apply directed rounding
        let ds_lower_b = downstream.lower_b();
        let ds_upper_b = downstream.upper_b();
        let ds_lb_flat = ds_lower_b
            .view()
            .into_shape_with_order((outer_batch_size, m, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape ds_bias: {e}"))
            })?;
        let ds_ub_flat = ds_upper_b
            .view()
            .into_shape_with_order((outer_batch_size, m, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape ds_bias: {e}"))
            })?;

        for ob in 0..outer_batch_size {
            for i in 0..m {
                for o in 0..out_dim {
                    bias_lower[[ob, i, o]] += ds_lb_flat[[ob, i, o]] as f64;
                    bias_upper[[ob, i, o]] += ds_ub_flat[[ob, i, o]] as f64;
                }
            }
        }

        let bias_lower_f32 = bias_lower.mapv(|v| {
            if v.is_nan() {
                f32::NEG_INFINITY
            } else {
                next_down_f32(v as f32)
            }
        });
        let bias_upper_f32 = bias_upper.mapv(|v| {
            if v.is_nan() {
                f32::INFINITY
            } else {
                next_up_f32(v as f32)
            }
        });

        // Reshape to N-D: [outer_batch..., m, out, k] and [outer_batch..., m, out]
        let mut q_a_shape: Vec<usize> = outer_batch.to_vec();
        q_a_shape.extend_from_slice(&[m, out_dim, k]);
        let mut bias_shape: Vec<usize> = outer_batch.to_vec();
        bias_shape.extend_from_slice(&[m, out_dim]);

        let q_la = q_lower_a
            .into_dyn()
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| NyError::InternalError(format!("BilinearCrown N-D: reshape q_la: {e}")))?;
        let q_ua = q_upper_a
            .into_dyn()
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| NyError::InternalError(format!("BilinearCrown N-D: reshape q_ua: {e}")))?;
        let bl = bias_lower_f32
            .into_dyn()
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape bias_l: {e}"))
            })?;
        let bu = bias_upper_f32
            .into_dyn()
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape bias_u: {e}"))
            })?;

        // input_shape is metadata matching the actual input tensor shape at
        // concretization. GELU backward passes it through unchanged, so it must
        // match the network input shape. The A matrix dimensions handle the
        // N-D batch structure separately (batch [outer_batch..., m], out, in=k).
        let q_input_shape = input_a_bounds.shape().to_vec();
        let q_output_shape = downstream.output_shape().to_vec();

        // Certified Q-coefficient error (#vnncomp-aw-soundness): each Q cell
        // [ob,i,o,l] f32-accumulates over the j axis (depth = n), so the f32
        // accumulation error is bounded by gamma_n(n) * S, rounded UP to a sound
        // f32 and carried so concretize penalizes the bound outward. An f32
        // coefficient that came out TIGHTER than its true real value (absorption
        // under-widening) is exactly the false-proof this penalty closes.
        let gamma_depth = crate::layers::linear::crown_single_gamma_n_f32(n);
        let q_lower_err = q_lower_s
            .mapv(|s| next_up_f32((gamma_depth * s) as f32))
            .into_dyn()
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape q_lower_err: {e}"))
            })?;
        let q_upper_err = q_upper_s
            .mapv(|s| next_up_f32((gamma_depth * s) as f32))
            .into_dyn()
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearCrown N-D: reshape q_upper_err: {e}"))
            })?;

        let mut bounds_q = crate::BatchedLinearBounds::new_or_conservative(
            q_la,
            bl,
            q_ua,
            bu,
            q_input_shape,
            q_output_shape.clone(),
        )?;
        bounds_q.set_coeff_err(q_lower_err, q_upper_err);

        // K path: zero bounds (K was concretized into Q bias).
        // The A matrix last dim must equal the product of k_input_shape so that
        // concretization and accumulation shapes are consistent. Since all values
        // are zero, only the shape matters for compatibility.
        let k_input_shape = input_b_bounds.shape().to_vec();
        let k_in_size: usize = checked_shape_product(&k_input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown N-D: K input shape product overflow: {k_input_shape:?}",
            ))
        })?;
        let zero_a = ArrayD::zeros(IxDyn(&[bias_shape.clone(), vec![k_in_size]].concat()));
        let zero_b = ArrayD::zeros(IxDyn(&bias_shape));
        let bounds_k = crate::BatchedLinearBounds::new_or_conservative(
            zero_a.clone(),
            zero_b.clone(),
            zero_a,
            zero_b,
            k_input_shape,
            q_output_shape,
        )?;

        Ok((bounds_q, bounds_k))
    }
}
