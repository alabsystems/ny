// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-domain GEMM batching for linear CROWN backward.

use faer::Mat;
use ndarray::{Array1, Array2};
use ny_core::{is_crown_coeff_safe, GemmEngine, NyError, Result};
use tracing::debug;

use super::bias::{
    accumulate_bias_f64, add_coeff_err_bias_product_up, add_f64_down, add_f64_up, f32_to_f64_exact,
    finalize_bias_directed, publish_error_up_normal, BiasBlockParams,
};
use super::crown_single::{aw_f64_with_abssum, gamma_n_f32, incoming_error_product};
use super::layout::resolve_backward_layout;
use super::LinearLayer;
use crate::{contiguous_flat_slice_mut, LinearBounds};

/// Multi-domain GPU-batched CROWN backward propagation.
///
/// Processes N domains' `LinearBounds` through a single Linear layer using one
/// batched GEMM call per bound direction.
///
/// # Soundness (#vnncomp-aw-soundness)
///
/// The point coefficients come from the accelerated f32 GEMM (preserving the
/// batched-call contract), but the certified per-coefficient error is computed
/// INDEPENDENTLY in f64 with the f32 growth factor `γ_n^{f32}` — sound because
/// the engine result and the f64 `a64` both round the SAME exact real `A·W`, so
/// `γ_n^{f32}·S` (where `S = |A|·|W|`, the contraction width is the GEMM `k`)
/// covers the engine↔a64 ULP gap as well as the a64↔true residual (same argument
/// as the scalar engine path in `crown_single.rs` and the N-D batched engine path
/// `compute_batched_linear_coefficients_engine` in `crown_batched.rs`). Any
/// incoming coefficient error on the input bounds is propagated as
/// `Σ_k err_in[i,k]·|W[k,j]|` (per-coefficient) and `Σ_k err_in[i,k]·|bias[k]|`
/// (folded OUTWARD into the bias before the directed cast). Linear is in
/// `propagates_coeff_err()` so the dispatcher TRUSTS this path to carry the error;
/// dropping it makes the β-CROWN/BaB verdict bound TIGHTER than the proven-sound
/// scalar path = a FALSE PROOF.
pub(crate) fn propagate_linear_batched_with_engine(
    layer: &LinearLayer,
    bounds_batch: &[&LinearBounds],
    engine: &dyn GemmEngine,
) -> Result<Vec<LinearBounds>> {
    if bounds_batch.is_empty() {
        return Ok(Vec::new());
    }
    if engine.forbids_unbounded_cpu_fallback() {
        return Err(NyError::UnsupportedOp(
            "bounded multi-domain Linear CROWN has no pollable host implementation".into(),
        ));
    }

    let first = bounds_batch[0];
    let num_outputs = first.num_outputs();
    let bounds_inputs = first.num_inputs();
    let n_domains = bounds_batch.len();

    for (idx, bounds) in bounds_batch.iter().enumerate().skip(1) {
        if bounds.num_outputs() != num_outputs || bounds.num_inputs() != bounds_inputs {
            return Err(NyError::InvalidSpec(format!(
                "Domain {} has shape [{}, {}], expected [{}, {}]",
                idx,
                bounds.num_outputs(),
                bounds.num_inputs(),
                num_outputs,
                bounds_inputs
            )));
        }
    }

    let weight_rows = layer.weight.nrows();
    let in_features = layer.weight.ncols();
    let layout = resolve_backward_layout(num_outputs, bounds_inputs, weight_rows, in_features)?;
    let weight_slice = layer
        .weight
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("Linear weight is not contiguous".to_string()))?;
    let total_stacked_rows = n_domains * num_outputs;

    // Certified-error machinery (#vnncomp-aw-soundness). The contraction width of
    // the `A·W` GEMM is `layout.out_features` (== weight_rows), the GEMM `k`-dim;
    // its f32 accumulation error is `γ_n^{f32}·S`. `|W|` (shape
    // [out_features, in_features]) is reused across positions/domains for both the
    // S-scaling and the incoming-error propagation `Σ_k err_in·|W|`.
    let gamma = gamma_n_f32(layout.out_features);
    let weight_faer = layer.weight_faer();
    let w_abs = Mat::<f32>::from_fn(layout.out_features, in_features, |k, j| {
        weight_faer[(k, j)].abs()
    });

    let mut results: Vec<LinearBounds> = Vec::with_capacity(n_domains);
    // Per-domain certified per-coefficient error accumulators (same layout as the
    // coefficient matrices), filled position-by-position alongside the coefficients.
    let mut lower_err_accum: Vec<Array2<f32>> = (0..n_domains)
        .map(|_| Array2::<f32>::zeros((num_outputs, layout.total_in_features)))
        .collect();
    let mut upper_err_accum: Vec<Array2<f32>> = (0..n_domains)
        .map(|_| Array2::<f32>::zeros((num_outputs, layout.total_in_features)))
        .collect();
    let mut bias_accum: Vec<(Array1<f64>, Array1<f64>)> = (0..n_domains)
        .map(|_| {
            (
                Array1::<f64>::zeros(num_outputs),
                Array1::<f64>::zeros(num_outputs),
            )
        })
        .collect();
    let mut lower_nonfinite: Vec<Vec<bool>> =
        (0..n_domains).map(|_| vec![false; num_outputs]).collect();
    let mut upper_nonfinite: Vec<Vec<bool>> =
        (0..n_domains).map(|_| vec![false; num_outputs]).collect();

    for pos in 0..layout.num_positions {
        let in_start = pos * layout.out_features;
        let out_start = pos * in_features;
        let mut stacked_lower = vec![0.0f32; total_stacked_rows * layout.out_features];
        let mut stacked_upper = vec![0.0f32; total_stacked_rows * layout.out_features];

        for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
            let row_offset = domain_idx * num_outputs;
            for i in 0..num_outputs {
                let dst_row = row_offset + i;
                for j in 0..layout.out_features {
                    stacked_lower[dst_row * layout.out_features + j] =
                        bounds.lower_a()[[i, in_start + j]];
                    stacked_upper[dst_row * layout.out_features + j] =
                        bounds.upper_a()[[i, in_start + j]];
                }
            }
        }

        let result_lower = engine.gemm_f32(
            total_stacked_rows,
            layout.out_features,
            in_features,
            &stacked_lower,
            weight_slice,
        )?;
        let result_upper = engine.gemm_f32(
            total_stacked_rows,
            layout.out_features,
            in_features,
            &stacked_upper,
            weight_slice,
        )?;

        if pos == 0 {
            results.reserve(n_domains);
            for _ in 0..n_domains {
                let new_lower_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
                let new_upper_a = Array2::<f32>::zeros((num_outputs, layout.total_in_features));
                results.push(LinearBounds::new_or_conservative(
                    new_lower_a,
                    Array1::<f32>::zeros(num_outputs),
                    new_upper_a,
                    Array1::<f32>::zeros(num_outputs),
                )?);
            }
        }

        for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
            let row_offset = domain_idx * num_outputs;

            // SOUND certified-error base (#vnncomp-aw-soundness): re-accumulate
            // `S = |A|·|W|` in f64 for THIS domain/position block. f32→f64 widening
            // is exact and f32×f32 fits in 48<53 bits, so `S` is an exact bound on
            // the absolute-product sum; `γ_n^{f32}·S` then bounds the f32 GEMM's
            // accumulation error for the same contraction.
            let a_lower_faer = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.lower_a()[[i, in_start + j]]
            });
            let a_upper_faer = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.upper_a()[[i, in_start + j]]
            });
            let (lower_reference, lower_s) = aw_f64_with_abssum(&a_lower_faer, weight_faer);
            let (upper_reference, upper_s) = aw_f64_with_abssum(&a_upper_faer, weight_faer);

            // Propagated incoming error: P[i,j] = Σ_k err_in[i,in_start+k]·|W[k,j]|.
            let prop_lower = match bounds.lower_a_err() {
                Some(error) => Some(incoming_error_product(
                    error,
                    in_start,
                    layout.out_features,
                    &w_abs,
                    None,
                )?),
                None => None,
            };
            let prop_upper = match bounds.upper_a_err() {
                Some(error) => Some(incoming_error_product(
                    error,
                    in_start,
                    layout.out_features,
                    &w_abs,
                    None,
                )?),
                None => None,
            };

            for i in 0..num_outputs {
                let src_row = row_offset + i;
                for j in 0..in_features {
                    let lower = result_lower[src_row * in_features + j];
                    let upper = result_upper[src_row * in_features + j];
                    let lower_gap = (f32_to_f64_exact(lower) - lower_reference[[i, j]]).abs();
                    let upper_gap = (f32_to_f64_exact(upper) - upper_reference[[i, j]]).abs();
                    let l_prop = prop_lower
                        .as_ref()
                        .map_or(0.0, |p| f32_to_f64_exact(p[(i, j)]));
                    let u_prop = prop_upper
                        .as_ref()
                        .map_or(0.0, |p| f32_to_f64_exact(p[(i, j)]));
                    let l_err =
                        publish_error_up_normal(lower_gap + gamma * lower_s[[i, j]] + l_prop);
                    let u_err =
                        publish_error_up_normal(upper_gap + gamma * upper_s[[i, j]] + u_prop);
                    // A stored coefficient is sound iff BOTH the coefficient and
                    // its certified error stay finite/in-range; otherwise the row
                    // is degraded to ±inf bias (handled below).
                    if is_crown_coeff_safe(lower) && l_err.is_finite() {
                        results[domain_idx].lower_a_mut()[[i, out_start + j]] = lower;
                        lower_err_accum[domain_idx][[i, out_start + j]] = l_err;
                    } else {
                        lower_nonfinite[domain_idx][i] = true;
                    }
                    if is_crown_coeff_safe(upper) && u_err.is_finite() {
                        results[domain_idx].upper_a_mut()[[i, out_start + j]] = upper;
                        upper_err_accum[domain_idx][[i, out_start + j]] = u_err;
                    } else {
                        upper_nonfinite[domain_idx][i] = true;
                    }
                }
            }
        }

        if let Some(ref bias) = layer.bias {
            let block = BiasBlockParams {
                num_outputs,
                out_features: layout.out_features,
                col_offset: in_start,
            };
            for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
                let (ref mut lower_acc, ref mut upper_acc) = bias_accum[domain_idx];
                let lower_slice = contiguous_flat_slice_mut(lower_acc)?;
                let upper_slice = contiguous_flat_slice_mut(upper_acc)?;
                accumulate_bias_f64(
                    &mut (lower_slice, upper_slice),
                    |i, j| bounds.lower_a()[[i, j]],
                    |i, j| bounds.upper_a()[[i, j]],
                    bias,
                    &block,
                );
            }
        }
    }

    // Fold the propagated incoming error into the BIAS too (#vnncomp-aw-soundness):
    // bias_contrib[i] = Σ_j A[i,j]·bias[j], so its certified error is
    // Σ_j err_in[i,j]·|bias[j]| — folded OUTWARD (lower decreases, upper increases)
    // into the f64 accumulators BEFORE the directed cast. Mirrors crown_single.rs.
    if let Some(ref bias) = layer.bias {
        for (domain_idx, bounds) in bounds_batch.iter().enumerate() {
            if let Some(le) = bounds.lower_a_err() {
                for i in 0..num_outputs {
                    let mut e = 0.0f64;
                    for j in 0..bias.len() {
                        e = add_coeff_err_bias_product_up(e, le[[i, j]], bias[j]);
                    }
                    bias_accum[domain_idx].0[i] = add_f64_down(bias_accum[domain_idx].0[i], -e);
                }
            }
            if let Some(ue) = bounds.upper_a_err() {
                for i in 0..num_outputs {
                    let mut e = 0.0f64;
                    for j in 0..bias.len() {
                        e = add_coeff_err_bias_product_up(e, ue[[i, j]], bias[j]);
                    }
                    bias_accum[domain_idx].1[i] = add_f64_up(bias_accum[domain_idx].1[i], e);
                }
            }
        }
    }

    if layer.bias.is_some() {
        for (domain_idx, result) in results.iter_mut().enumerate() {
            let (lower_b, upper_b) = finalize_bias_directed(
                &bias_accum[domain_idx].0,
                &bias_accum[domain_idx].1,
                bounds_batch[domain_idx].lower_b(),
                bounds_batch[domain_idx].upper_b(),
            );
            *result.lower_b_mut() = lower_b;
            *result.upper_b_mut() = upper_b;
        }
    } else {
        for (domain_idx, result) in results.iter_mut().enumerate() {
            *result.lower_b_mut() = bounds_batch[domain_idx].lower_b().clone();
            *result.upper_b_mut() = bounds_batch[domain_idx].upper_b().clone();
        }
    }

    for (domain_idx, result) in results.iter_mut().enumerate() {
        let lower_affected = lower_nonfinite[domain_idx]
            .iter()
            .filter(|&&row| row)
            .count();
        let upper_affected = upper_nonfinite[domain_idx]
            .iter()
            .filter(|&&row| row)
            .count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "Linear CROWN backward (multi-domain GEMM, domain {}): overflow/magnitude \
                 in {}/{} lower rows, {}/{} upper rows — falling back to ±inf bias (#1932)",
                domain_idx, lower_affected, num_outputs, upper_affected, num_outputs
            );
            for i in 0..num_outputs {
                if lower_nonfinite[domain_idx][i] {
                    for j in 0..layout.total_in_features {
                        result.lower_a_mut()[[i, j]] = 0.0;
                        // A row degraded to ±inf bias is maximally loose; zero its
                        // certified error so the penalty is not double-applied.
                        lower_err_accum[domain_idx][[i, j]] = 0.0;
                    }
                    result.lower_b_mut()[i] = f32::NEG_INFINITY;
                }
                if upper_nonfinite[domain_idx][i] {
                    for j in 0..layout.total_in_features {
                        result.upper_a_mut()[[i, j]] = 0.0;
                        upper_err_accum[domain_idx][[i, j]] = 0.0;
                    }
                    result.upper_b_mut()[i] = f32::INFINITY;
                }
            }
        }
        // Attach the certified per-coefficient error so concretize_sound applies
        // the S-scaled penalty (#vnncomp-aw-soundness). Linear declares
        // propagates_coeff_err = true, so the dispatcher relies on this.
        result.set_coeff_err(
            std::mem::take(&mut lower_err_accum[domain_idx]),
            std::mem::take(&mut upper_err_accum[domain_idx]),
        );
    }

    Ok(results)
}
