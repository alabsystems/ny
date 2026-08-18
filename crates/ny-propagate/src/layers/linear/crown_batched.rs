// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for linear layers.
//!
//! Contains both the N-D batched path (single domain with batch dimensions)
//! and the multi-domain GPU-batched path (multiple domains stacked for one
//! large GEMM call).

use faer::Mat;
use ndarray::{s, Array1, Array2, ArrayD, ArrayView3, IxDyn};
use ny_core::{checked_shape_product, is_crown_coeff_safe, GemmEngine, NyError, Result};
use tracing::debug;

use super::bias::{
    accumulate_bias_f64, add_coeff_err_bias_product_up, add_f64_down, add_f64_up, f32_to_f64_exact,
    finalize_bias_directed, publish_error_up_normal, BiasBlockParams,
};
use super::LinearLayer;
use crate::{contiguous_flat_slice, BatchedLinearBounds};

/// Batched CROWN backward through linear layer y=Wx+b (N-D batch dims).
///
/// Substitution: A@(Wx+b)+c = (A@W)@x + (A@b+c). Operates on last dims only,
/// preserving batch structure. A: [...batch, out_dim, mid_dim], W: [mid_dim, in_dim].
/// Result: [...batch, out_dim, in_dim].
pub(crate) fn propagate_linear_batched(
    layer: &LinearLayer,
    bounds: &BatchedLinearBounds,
) -> Result<BatchedLinearBounds> {
    propagate_linear_batched_maybe_engine(layer, bounds, None)
}

/// Preserve historical fallback for ordinary engines. A bounded host facade's
/// structured memory refusal is terminal because local fallback would allocate
/// the refused buffer; a deadline is terminal only when the engine proves
/// expiry through the cooperative CROWN poll seam.
#[inline]
fn engine_error_is_terminal(error: &NyError, engine: &dyn GemmEngine) -> bool {
    (error.is_cpu_memory_exceeded() && engine.forbids_unbounded_cpu_fallback())
        || (error.is_deadline_exceeded()
            && matches!(
                engine.poll_crown_backward_deadline(),
                Err(error) if error.is_deadline_exceeded()
            ))
}

pub(crate) fn propagate_linear_batched_maybe_engine(
    layer: &LinearLayer,
    bounds: &BatchedLinearBounds,
    engine: Option<&dyn GemmEngine>,
) -> Result<BatchedLinearBounds> {
    debug!("Linear layer batched CROWN backward propagation");

    if engine.is_some_and(|engine| engine.forbids_unbounded_cpu_fallback()) {
        return Err(NyError::UnsupportedOp(
            "bounded Linear batched CROWN has no pollable host implementation".into(),
        ));
    }

    let a_shape = bounds.lower_a.shape();
    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "BatchedLinearBounds must have at least 2 dimensions".to_string(),
        ));
    }

    let out_dim = a_shape[a_shape.len() - 2];
    let mid_dim = a_shape[a_shape.len() - 1];

    if mid_dim != layer.out_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_dim, layer.out_features()],
            got: vec![out_dim, mid_dim],
        });
    }

    let in_dim = layer.in_features();
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let total_batch: usize = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Linear batched CROWN: batch dims {batch_dims:?} overflow"
            ))
        })?
        .max(1);

    // Output A shape: [...batch, out_dim, in_dim]
    let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
    out_a_shape.push(out_dim);
    out_a_shape.push(in_dim);

    // Output b shape: [...batch, out_dim]
    let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
    out_b_shape.push(out_dim);

    // Reshape A to [batch, out_dim, mid_dim] for computation
    let lower_a_3d = bounds
        .lower_a
        .view()
        .into_shape_with_order((total_batch, out_dim, mid_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
    let upper_a_3d = bounds
        .upper_a
        .view()
        .into_shape_with_order((total_batch, out_dim, mid_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

    // Incoming certified error on this bounds object's coefficients (from a prior
    // linear/conv backward composed through earlier layers, #vnncomp-aw-soundness).
    // It MUST be propagated as `Σ_k err_in[i,k]·|W[k,j]|` (per-coefficient) and
    // `Σ_k err_in[i,k]·|bias[k]|` (into the bias), exactly as the scalar
    // `propagate_linear` does. Dropping it makes the batched (β-CROWN/BaB) verdict
    // path's penalty too small → bounds TIGHTER than the proven-sound scalar path.
    let in_lower_err_3d = bounds
        .lower_a_err
        .as_ref()
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, mid_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming lower_a_err".to_string()))?;
    let in_upper_err_3d = bounds
        .upper_a_err
        .as_ref()
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, mid_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming upper_a_err".to_string()))?;

    let total_rows = total_batch * out_dim;
    let (result_lower_flat, result_upper_flat, lower_err_flat, upper_err_flat) =
        if let Some(engine) = engine {
            match compute_batched_linear_coefficients_engine(
                layer,
                lower_a_3d,
                upper_a_3d,
                in_lower_err_3d.as_ref().map(|v| v.view()),
                in_upper_err_3d.as_ref().map(|v| v.view()),
                total_rows,
                mid_dim,
                in_dim,
                engine,
            ) {
                Ok(results) => results,
                Err(err) if engine_error_is_terminal(&err, engine) => {
                    return Err(err);
                }
                Err(err) => {
                    debug!("Linear batched CROWN: GEMM engine failed, falling back to faer: {err}");
                    compute_batched_linear_coefficients_cpu(
                        layer,
                        lower_a_3d,
                        upper_a_3d,
                        in_lower_err_3d.as_ref().map(|v| v.view()),
                        in_upper_err_3d.as_ref().map(|v| v.view()),
                        total_batch,
                        out_dim,
                        mid_dim,
                        in_dim,
                    )?
                }
            }
        } else {
            compute_batched_linear_coefficients_cpu(
                layer,
                lower_a_3d,
                upper_a_3d,
                in_lower_err_3d.as_ref().map(|v| v.view()),
                in_upper_err_3d.as_ref().map(|v| v.view()),
                total_batch,
                out_dim,
                mid_dim,
                in_dim,
            )?
        };

    let mut new_lower_a = Array2::from_shape_vec((total_rows, in_dim), result_lower_flat)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
    let mut new_upper_a = Array2::from_shape_vec((total_rows, in_dim), result_upper_flat)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
    // Certified per-coefficient error (#vnncomp-aw-soundness), same layout as the
    // coefficient matrices.
    let mut new_lower_err = Array2::from_shape_vec((total_rows, in_dim), lower_err_flat)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_err".to_string()))?;
    let mut new_upper_err = Array2::from_shape_vec((total_rows, in_dim), upper_err_flat)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_err".to_string()))?;

    // Track which output rows have non-finite coefficients (#2681).
    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];
    for row in 0..total_rows {
        for col in 0..in_dim {
            if !is_crown_coeff_safe(new_lower_a[[row, col]])
                || !new_lower_err[[row, col]].is_finite()
            {
                lower_nonfinite_rows[row] = true;
            }
            if !is_crown_coeff_safe(new_upper_a[[row, col]])
                || !new_upper_err[[row, col]].is_finite()
            {
                upper_nonfinite_rows[row] = true;
            }
        }
        // A row degraded to ±inf bias is maximally loose; zero both its
        // coefficients AND its certified error so the penalty isn't double-applied.
        if lower_nonfinite_rows[row] {
            for col in 0..in_dim {
                new_lower_a[[row, col]] = 0.0;
                new_lower_err[[row, col]] = 0.0;
            }
        }
        if upper_nonfinite_rows[row] {
            for col in 0..in_dim {
                new_upper_a[[row, col]] = 0.0;
                new_upper_err[[row, col]] = 0.0;
            }
        }
    }

    // #2681/#1932: For rows with non-finite or near-overflow A-matrix coefficients,
    // zero the entire row. Bias override (to ±inf) happens after bias computation.
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "Linear CROWN backward (batched{}): overflow/magnitude in {}/{} lower rows, \
             {}/{} upper rows — falling back to ±inf bias for affected rows (#1932)",
            if engine.is_some() { " GEMM" } else { " faer" },
            lower_affected,
            total_rows,
            upper_affected,
            total_rows
        );
    }

    // Reshape back to [...batch, out_dim, in_dim]
    let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
    let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
    let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
    let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
    let (new_lower_err_vec, _) = new_lower_err.into_raw_vec_and_offset();
    let (new_upper_err_vec, _) = new_upper_err.into_raw_vec_and_offset();
    let new_lower_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_err_vec)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_err".to_string()))?;
    let new_upper_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_err_vec)
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_err".to_string()))?;

    // Compute bias contribution: A @ bias + old_b (#2427: shared helper).
    let (new_lower_b, new_upper_b) = if let Some(ref bias) = layer.bias {
        let lower_b_3d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_3d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        let block = BiasBlockParams {
            num_outputs: out_dim,
            out_features: mid_dim,
            col_offset: 0,
        };

        let mut new_lower_b_f32 = Vec::with_capacity(total_batch * out_dim);
        let mut new_upper_b_f32 = Vec::with_capacity(total_batch * out_dim);

        for b in 0..total_batch {
            let a_lower = lower_a_3d.slice(s![b, .., ..]);
            let a_upper = upper_a_3d.slice(s![b, .., ..]);

            let mut lower_accum = vec![0.0_f64; out_dim];
            let mut upper_accum = vec![0.0_f64; out_dim];
            accumulate_bias_f64(
                &mut (&mut lower_accum[..], &mut upper_accum[..]),
                |i, j| a_lower[[i, j]],
                |i, j| a_upper[[i, j]],
                bias,
                &block,
            );

            // Incoming-error contribution to the bias (#vnncomp-aw-soundness): the
            // bias term is `Σ_k A[i,k]·bias[k]`, so a coefficient uncertainty
            // `err_in[i,k]` widens it by `Σ_k err_in[i,k]·|bias[k]|`. Fold it OUTWARD
            // (lower DOWN, upper UP) BEFORE the directed cast, mirroring the scalar
            // `propagate_linear` bias-error fold.
            if let Some(le) = in_lower_err_3d.as_ref() {
                let e = le.slice(s![b, .., ..]);
                for i in 0..out_dim {
                    let mut acc = 0.0f64;
                    for k in 0..mid_dim {
                        acc = add_coeff_err_bias_product_up(acc, e[[i, k]], bias[k]);
                    }
                    lower_accum[i] = add_f64_down(lower_accum[i], -acc);
                }
            }
            if let Some(ue) = in_upper_err_3d.as_ref() {
                let e = ue.slice(s![b, .., ..]);
                for i in 0..out_dim {
                    let mut acc = 0.0f64;
                    for k in 0..mid_dim {
                        acc = add_coeff_err_bias_product_up(acc, e[[i, k]], bias[k]);
                    }
                    upper_accum[i] = add_f64_up(upper_accum[i], acc);
                }
            }

            let old_lb = lower_b_3d.slice(s![b, ..]);
            let old_ub = upper_b_3d.slice(s![b, ..]);
            let (lb, ub) = finalize_bias_directed(
                &Array1::from(lower_accum),
                &Array1::from(upper_accum),
                &old_lb.to_owned(),
                &old_ub.to_owned(),
            );
            let lb_slice = contiguous_flat_slice(&lb);
            let ub_slice = contiguous_flat_slice(&ub);
            new_lower_b_f32.extend_from_slice(&lb_slice);
            new_upper_b_f32.extend_from_slice(&ub_slice);
        }

        (
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_f32)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_f32)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
        )
    } else {
        (bounds.lower_b.clone(), bounds.upper_b.clone())
    };

    // #2681: Override bias to ±inf for rows with non-finite A-matrix coefficients.
    // The A-coefficients were already zeroed above; now set bias so concretized
    // bounds become [-inf, +inf] for those rows (sound, maximally loose).
    let mut new_lower_b = new_lower_b;
    let mut new_upper_b = new_upper_b;
    if lower_affected > 0 || upper_affected > 0 {
        let lower_b_flat = new_lower_b
            .as_slice_mut()
            .ok_or_else(|| NyError::InternalError("new_lower_b not contiguous".to_string()))?;
        let upper_b_flat = new_upper_b
            .as_slice_mut()
            .ok_or_else(|| NyError::InternalError("new_upper_b not contiguous".to_string()))?;
        for row in 0..total_rows {
            if lower_nonfinite_rows[row] {
                lower_b_flat[row] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[row] {
                upper_b_flat[row] = f32::INFINITY;
            }
        }
    }

    // Update input shape to reflect the linear layer's input dimension
    let mut new_input_shape = bounds.input_shape.clone();
    if let Some(last_dim) = new_input_shape.last_mut() {
        *last_dim = in_dim;
    }

    // CROWN backward NaN firewall (#2812): conservative fallback instead of aborting
    // when matmul produces NaN/Inf. Has own non-finite row handling but this catches
    // any remaining contamination from upstream biases or missed rows.
    let mut result = BatchedLinearBounds::new_or_conservative(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_input_shape,
        bounds.output_shape.clone(),
    )?;
    // Attach the certified coefficient error so the batched (β-CROWN/BaB) verdict
    // path carries the SAME soundness margin as the scalar path. Shapes match the
    // coefficient matrices iff the firewall did not degrade to conservative; the
    // setter no-ops on mismatch (conservative bounds already cover the penalty).
    result.set_coeff_err(new_lower_err, new_upper_err);
    Ok(result)
}

/// CPU batched `A·W` with the SAME f64-accumulated product as the scalar path
/// (#vnncomp-aw-soundness). Returns the point coefficients AND the certified
/// per-coefficient error `γ_n·S + cast_gap` (next_up-rounded), flattened to
/// `[total_rows, in_dim]` row-major. Using `aw_f64_with_abssum` here makes the
/// batched coefficient BIT-IDENTICAL to the scalar `propagate_linear` result
/// (both round the exact real `A·W` from an f64 accumulation), fixing the
/// previous ~1-ULP optimistic divergence of the f32 GEMM batched path.
#[allow(clippy::too_many_arguments)]
fn compute_batched_linear_coefficients_cpu(
    layer: &LinearLayer,
    lower_a_3d: ArrayView3<'_, f32>,
    upper_a_3d: ArrayView3<'_, f32>,
    in_lower_err_3d: Option<ArrayView3<'_, f32>>,
    in_upper_err_3d: Option<ArrayView3<'_, f32>>,
    total_batch: usize,
    out_dim: usize,
    mid_dim: usize,
    in_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use crate::faer_parallelism::mat_mul;
    use crate::layers::linear::crown_single::{
        aw_f64_with_abssum, gamma_n_f64, incoming_error_product,
    };

    let total_rows = total_batch * out_dim;
    let mut result_lower_flat = vec![0.0_f32; total_rows * in_dim];
    let mut result_upper_flat = vec![0.0_f32; total_rows * in_dim];
    let mut lower_err_flat = vec![0.0_f32; total_rows * in_dim];
    let mut upper_err_flat = vec![0.0_f32; total_rows * in_dim];
    let gamma = gamma_n_f64(mid_dim);
    // |W| once (reused per batch): the incoming error propagates as
    // `P[i,j] = Σ_k err_in[i,k]·|W[k,j]|`, mirroring the scalar `propagate_linear`.
    let w_abs = Mat::<f32>::from_fn(mid_dim, in_dim, |k, j| layer.weight_faer()[(k, j)].abs());

    // TEST-ONLY (NY_AW_LEGACY_F32): reproduce the OLD optimistic batched path —
    // round-to-nearest f32 faer GEMM for `A·W` with NO certified error — so the
    // strict batched soundness proptest can confirm it CATCHES the unsoundness
    // (symmetric with the scalar `propagate_linear_cpu` legacy hook). Never set
    // in production; gated on debug builds only.
    let legacy_f32 = cfg!(debug_assertions) && std::env::var("NY_AW_LEGACY_F32").is_ok();

    for batch_idx in 0..total_batch {
        let a_lower = lower_a_3d.slice(s![batch_idx, .., ..]);
        let a_upper = upper_a_3d.slice(s![batch_idx, .., ..]);

        let a_lower_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_lower[[i, j]]);
        let a_upper_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_upper[[i, j]]);

        if legacy_f32 {
            // UNSOUND legacy: f32 GEMM, zero error.
            let ml = mat_mul(&a_lower_faer, layer.weight_faer());
            let mu = mat_mul(&a_upper_faer, layer.weight_faer());
            for out_idx in 0..out_dim {
                let row = batch_idx * out_dim + out_idx;
                for in_idx in 0..in_dim {
                    result_lower_flat[row * in_dim + in_idx] = ml[(out_idx, in_idx)];
                    result_upper_flat[row * in_dim + in_idx] = mu[(out_idx, in_idx)];
                    // err stays 0.0 (no certified penalty) — the bug.
                }
            }
            continue;
        }

        let (lower_a64, lower_s) = aw_f64_with_abssum(&a_lower_faer, layer.weight_faer());
        let (upper_a64, upper_s) = aw_f64_with_abssum(&a_upper_faer, layer.weight_faer());

        // Propagated incoming error P[i,j] = Σ_k err_in[i,k]·|W[k,j]| (per batch).
        let prop_lower = match in_lower_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(&block, 0, mid_dim, &w_abs, None)?)
            }
            None => None,
        };
        let prop_upper = match in_upper_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(&block, 0, mid_dim, &w_abs, None)?)
            }
            None => None,
        };

        for out_idx in 0..out_dim {
            let row = batch_idx * out_dim + out_idx;
            for in_idx in 0..in_dim {
                let l = lower_a64[[out_idx, in_idx]] as f32;
                let u = upper_a64[[out_idx, in_idx]] as f32;
                result_lower_flat[row * in_dim + in_idx] = l;
                result_upper_flat[row * in_dim + in_idx] = u;
                let l_cast_err = (lower_a64[[out_idx, in_idx]] - f32_to_f64_exact(l)).abs();
                let u_cast_err = (upper_a64[[out_idx, in_idx]] - f32_to_f64_exact(u)).abs();
                let l_prop = prop_lower
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(out_idx, in_idx)]));
                let u_prop = prop_upper
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(out_idx, in_idx)]));
                lower_err_flat[row * in_dim + in_idx] = publish_error_up_normal(
                    l_cast_err + gamma * lower_s[[out_idx, in_idx]] + l_prop,
                );
                upper_err_flat[row * in_dim + in_idx] = publish_error_up_normal(
                    u_cast_err + gamma * upper_s[[out_idx, in_idx]] + u_prop,
                );
            }
        }
    }

    Ok((
        result_lower_flat,
        result_upper_flat,
        lower_err_flat,
        upper_err_flat,
    ))
}

/// Engine (GPU) batched `A·W`: the point coefficients come from the f32 GEMM
/// (keeping the accelerated call-count contract), but the certified error is
/// computed independently in f64 with the f32 growth factor `γ_n^{f32}` — sound
/// because the engine result and the f64 `a64` both round the SAME exact real
/// `A·W`, so `γ_n^{f32}·S` covers both the engine↔a64 ULP and the a64↔true gap
/// (same argument as the scalar engine path). Returns
/// `(lower, upper, lower_err, upper_err)` flattened to `[total_rows, in_dim]`.
#[allow(clippy::too_many_arguments)]
fn compute_batched_linear_coefficients_engine(
    layer: &LinearLayer,
    lower_a_3d: ArrayView3<'_, f32>,
    upper_a_3d: ArrayView3<'_, f32>,
    in_lower_err_3d: Option<ArrayView3<'_, f32>>,
    in_upper_err_3d: Option<ArrayView3<'_, f32>>,
    total_rows: usize,
    mid_dim: usize,
    in_dim: usize,
    engine: &dyn GemmEngine,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use crate::layers::linear::crown_single::{
        aw_f64_with_abssum, gamma_n_f32, incoming_error_product,
    };

    let weight_slice = layer
        .weight
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("Linear weight is not contiguous".to_string()))?;
    let lower_stacked: Vec<f32> = lower_a_3d.iter().copied().collect();
    let upper_stacked: Vec<f32> = upper_a_3d.iter().copied().collect();
    let result_lower =
        engine.gemm_f32(total_rows, mid_dim, in_dim, &lower_stacked, weight_slice)?;
    let result_upper =
        engine.gemm_f32(total_rows, mid_dim, in_dim, &upper_stacked, weight_slice)?;

    // Certified error (CPU f64), independent of the GPU point coefficient.
    let gamma = gamma_n_f32(mid_dim);
    let weight_faer = layer.weight_faer();
    // |W| for the incoming-error propagation P[i,j] = Σ_k err_in[i,k]·|W[k,j]|.
    let w_abs = Mat::<f32>::from_fn(mid_dim, in_dim, |k, j| weight_faer[(k, j)].abs());
    let mut lower_err = vec![0.0f32; total_rows * in_dim];
    let mut upper_err = vec![0.0f32; total_rows * in_dim];
    // total_rows = total_batch * out_dim; lower_a_3d is [total_batch, out_dim, mid_dim].
    let tb = lower_a_3d.shape()[0];
    let out_dim = lower_a_3d.shape()[1];
    for batch_idx in 0..tb {
        let a_lower = lower_a_3d.slice(s![batch_idx, .., ..]);
        let a_upper = upper_a_3d.slice(s![batch_idx, .., ..]);
        let a_lower_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_lower[[i, j]]);
        let a_upper_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_upper[[i, j]]);
        let (lower_reference, lower_s) = aw_f64_with_abssum(&a_lower_faer, weight_faer);
        let (upper_reference, upper_s) = aw_f64_with_abssum(&a_upper_faer, weight_faer);
        // Propagated incoming error per batch.
        let prop_lower = match in_lower_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(&block, 0, mid_dim, &w_abs, None)?)
            }
            None => None,
        };
        let prop_upper = match in_upper_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(&block, 0, mid_dim, &w_abs, None)?)
            }
            None => None,
        };
        for out_idx in 0..out_dim {
            let row = batch_idx * out_dim + out_idx;
            for in_idx in 0..in_dim {
                let stored_lower = result_lower[row * in_dim + in_idx];
                let stored_upper = result_upper[row * in_dim + in_idx];
                let lower_gap =
                    (f32_to_f64_exact(stored_lower) - lower_reference[[out_idx, in_idx]]).abs();
                let upper_gap =
                    (f32_to_f64_exact(stored_upper) - upper_reference[[out_idx, in_idx]]).abs();
                let l_prop = prop_lower
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(out_idx, in_idx)]));
                let u_prop = prop_upper
                    .as_ref()
                    .map_or(0.0, |p| f32_to_f64_exact(p[(out_idx, in_idx)]));
                lower_err[row * in_dim + in_idx] = publish_error_up_normal(
                    lower_gap + gamma * lower_s[[out_idx, in_idx]] + l_prop,
                );
                upper_err[row * in_dim + in_idx] = publish_error_up_normal(
                    upper_gap + gamma * upper_s[[out_idx, in_idx]] + u_prop,
                );
            }
        }
    }

    Ok((result_lower, result_upper, lower_err, upper_err))
}
