// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SoA (structure-of-arrays) multi-domain GEMM batching for linear CROWN
//! backward — the Linear arm of the #lsnc-batched-bwd input-split fast lane
//! (design-doc slice S3, `docs/LSNC_BATCH_TENSOR_DESIGN.md`).
//!
//! # Relationship to [`propagate_linear_batched_with_engine`]
//!
//! [`super::crown_batched_multi_domain::propagate_linear_batched_with_engine`]
//! is the scalar reference: it consumes per-domain `LinearBounds`, issues ONE
//! stacked f32 engine GEMM per position over the active domains, then runs the
//! per-domain certified-error machinery (`aw_f64_with_abssum` S-recompute,
//! incoming-error propagation, f64 bias accumulation, directed finalize,
//! non-finite row degrade) SERIALLY on the driver thread.
//!
//! This SoA twin is a faithful transcription that
//! 1. reads the pending coefficient planes directly from contiguous
//!    `[B, R, W]` batch tensors (no per-domain `LinearBounds` allocation),
//! 2. issues the IDENTICAL stacked engine GEMM calls (same stacked row
//!    contents, same `m/k/n`, same engine → identical f32 bits), and
//! 3. runs the per-domain certified-error block in COARSE rayon chunks over
//!    the domain axis. Every per-domain step is domain-independent and calls
//!    the SAME shared leaves (`aw_f64_with_abssum`, `mat_mul`,
//!    `accumulate_bias_f64`, `finalize_bias_directed`, `is_crown_coeff_safe`,
//!    `next_up_f32`) with the same inputs in the same per-domain order, so the
//!    result is BIT-IDENTICAL per domain to the reference.
//!
//! A [`NestedFaerParGuard`] is held inside each worker so the faer f64
//! products (`aw_f64_with_abssum`, the `err_in·|W|` `mat_mul`) run under the
//! SAME faer parallelism policy as the reference's driver-thread calls
//! (`current_par()` would otherwise force `Par::Seq` inside a rayon worker).
//! At the lsnc-lane matrix sizes faer never splits these products, so the
//! blocking — and hence the f64 bits — is identical either way; the guard
//! makes the equality hold by construction for larger shapes too. Parity is
//! pinned by `test_linear_backward_soa_matches_reference` below and the
//! full-pipeline gate test in `tests_soundness.rs`.

use faer::Mat;
use ndarray::Array1;
use ny_core::{is_crown_coeff_safe, GemmEngine, NyError, Result};
use ny_tensor::next_up_f32;
use rayon::prelude::*;
use tracing::debug;

use super::bias::{accumulate_bias_f64, finalize_bias_directed, BiasBlockParams};
use super::crown_single::{aw_f64_with_abssum, gamma_n_f32};
use super::layout::resolve_backward_layout;
use super::LinearLayer;
use crate::faer_parallelism::{mat_mul, NestedFaerParGuard};

/// Borrowed SoA view of the per-node pending linear bounds for a batch of
/// domains. Planes are domain-major row-major: domain `d`'s coefficient block
/// is `plane[d*rows*cols .. (d+1)*rows*cols]`, bias block is
/// `bias[d*rows .. (d+1)*rows]`.
///
/// `*_err_present[d]` mirrors the per-domain `Option` on
/// `LinearBounds::lower_a_err`/`upper_a_err` (`None` = exact marker, I-D5):
/// when `false`, the err plane region for that domain is ignored (it holds
/// zeros). Invariant: a `true` flag requires the corresponding plane to be
/// `Some`.
pub(crate) struct SoaLinearBackwardInput<'a> {
    pub n_domains: usize,
    pub rows: usize,
    pub cols: usize,
    pub active: &'a [bool],
    pub lower_a: &'a [f32],
    pub upper_a: &'a [f32],
    pub lower_b: &'a [f32],
    pub upper_b: &'a [f32],
    pub lower_a_err: Option<&'a [f32]>,
    pub upper_a_err: Option<&'a [f32]>,
    pub lower_err_present: &'a [bool],
    pub upper_err_present: &'a [bool],
}

/// Owned SoA result of the Linear SoA backward. Same layout contract as
/// [`SoaLinearBackwardInput`]; `cols == layout.total_in_features`. The
/// certified coefficient-error planes are ALWAYS attached for active domains
/// (the reference `set_coeff_err`s unconditionally — Linear declares
/// `propagates_coeff_err = true`, I-D5).
pub(crate) struct SoaLinearBackwardOutput {
    pub rows: usize,
    pub cols: usize,
    pub active: Vec<bool>,
    pub lower_a: Vec<f32>,
    pub upper_a: Vec<f32>,
    pub lower_b: Vec<f32>,
    pub upper_b: Vec<f32>,
    pub lower_a_err: Vec<f32>,
    pub upper_a_err: Vec<f32>,
}

/// SoA twin of [`propagate_linear_batched_with_engine`] — see module docs for
/// the bit-parity argument. `min_chunk` is the minimum number of domains per
/// rayon task (coarse chunking; NO per-domain fan-out).
pub(crate) fn propagate_linear_batched_soa(
    layer: &LinearLayer,
    input: &SoaLinearBackwardInput<'_>,
    engine: &dyn GemmEngine,
    min_chunk: usize,
) -> Result<SoaLinearBackwardOutput> {
    let n_domains = input.n_domains;
    let num_outputs = input.rows;
    let bounds_inputs = input.cols;
    debug_assert_eq!(input.active.len(), n_domains);
    debug_assert_eq!(input.lower_a.len(), n_domains * num_outputs * bounds_inputs);
    debug_assert_eq!(input.lower_b.len(), n_domains * num_outputs);
    debug_assert!(input.lower_a_err.is_some() || input.lower_err_present.iter().all(|&p| !p));
    debug_assert!(input.upper_a_err.is_some() || input.upper_err_present.iter().all(|&p| !p));

    let active_domains: Vec<usize> = (0..n_domains).filter(|&d| input.active[d]).collect();

    let weight_rows = layer.weight.nrows();
    let in_features = layer.weight.ncols();
    let layout = resolve_backward_layout(num_outputs, bounds_inputs, weight_rows, in_features)?;
    let weight_slice = layer
        .weight
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("Linear weight is not contiguous".to_string()))?;
    let n_active = active_domains.len();
    let total_stacked_rows = n_active * num_outputs;
    let out_cols = layout.total_in_features;

    let mut out_lower_a = vec![0.0f32; n_domains * num_outputs * out_cols];
    let mut out_upper_a = vec![0.0f32; n_domains * num_outputs * out_cols];
    let mut out_lower_b = vec![0.0f32; n_domains * num_outputs];
    let mut out_upper_b = vec![0.0f32; n_domains * num_outputs];
    let mut out_lower_err = vec![0.0f32; n_domains * num_outputs * out_cols];
    let mut out_upper_err = vec![0.0f32; n_domains * num_outputs * out_cols];
    if n_active == 0 {
        return Ok(SoaLinearBackwardOutput {
            rows: num_outputs,
            cols: out_cols,
            active: input.active.to_vec(),
            lower_a: out_lower_a,
            upper_a: out_upper_a,
            lower_b: out_lower_b,
            upper_b: out_upper_b,
            lower_a_err: out_lower_err,
            upper_a_err: out_upper_err,
        });
    }

    // Certified-error machinery (#vnncomp-aw-soundness) — identical to the
    // reference: contraction width = layout.out_features, f32 growth factor.
    let gamma = gamma_n_f32(layout.out_features);
    let weight_faer = layer.weight_faer();
    let w_abs = Mat::<f32>::from_fn(layout.out_features, in_features, |k, j| {
        weight_faer[(k, j)].abs()
    });

    // Phase 1 (driver thread): the stacked f32 engine GEMMs, one pair per
    // position — IDENTICAL stacked contents and call shapes to the reference
    // (active domains in domain-index order, `num_outputs` rows each).
    let mut gemm_lower: Vec<Vec<f32>> = Vec::with_capacity(layout.num_positions);
    let mut gemm_upper: Vec<Vec<f32>> = Vec::with_capacity(layout.num_positions);
    for pos in 0..layout.num_positions {
        let in_start = pos * layout.out_features;
        let mut stacked_lower = vec![0.0f32; total_stacked_rows * layout.out_features];
        let mut stacked_upper = vec![0.0f32; total_stacked_rows * layout.out_features];
        for (k, &d) in active_domains.iter().enumerate() {
            let row_offset = k * num_outputs;
            let base = d * num_outputs * bounds_inputs;
            for i in 0..num_outputs {
                let src = base + i * bounds_inputs + in_start;
                let dst = (row_offset + i) * layout.out_features;
                stacked_lower[dst..dst + layout.out_features]
                    .copy_from_slice(&input.lower_a[src..src + layout.out_features]);
                stacked_upper[dst..dst + layout.out_features]
                    .copy_from_slice(&input.upper_a[src..src + layout.out_features]);
            }
        }
        gemm_lower.push(engine.gemm_f32(
            total_stacked_rows,
            layout.out_features,
            in_features,
            &stacked_lower,
            weight_slice,
        )?);
        gemm_upper.push(engine.gemm_f32(
            total_stacked_rows,
            layout.out_features,
            in_features,
            &stacked_upper,
            weight_slice,
        )?);
    }

    // Phase 2: per-domain certified-error block, coarse-chunked over domains.
    // Safe disjoint parallelism: the output planes are chunked per domain and
    // zipped, so each worker owns exactly its domain's regions.
    let coeff_len = num_outputs * out_cols;
    let in_coeff_len = num_outputs * bounds_inputs;

    // Stacked-row index of each domain in the active-only GEMM (None =
    // inactive), matching the reference's active-list enumeration order.
    let k_of: Vec<Option<usize>> = {
        let mut k_of = vec![None; n_domains];
        for (k, &d) in active_domains.iter().enumerate() {
            k_of[d] = Some(k);
        }
        k_of
    };

    out_lower_a
        .par_chunks_mut(coeff_len)
        .zip(out_upper_a.par_chunks_mut(coeff_len))
        .zip(out_lower_b.par_chunks_mut(num_outputs))
        .zip(out_upper_b.par_chunks_mut(num_outputs))
        .zip(out_lower_err.par_chunks_mut(coeff_len))
        .zip(out_upper_err.par_chunks_mut(coeff_len))
        .enumerate()
        .with_min_len(min_chunk.max(1))
        .for_each(
            |(
                d,
                (
                    ((((res_lower_a, res_upper_a), res_lower_b), res_upper_b), res_lower_err),
                    res_upper_err,
                ),
            )| {
                let Some(k) = k_of[d] else { return };
                // Match the reference's faer parallelism policy (driver thread):
                // without this, `current_par()` inside a rayon worker forces
                // Par::Seq — same bits at lsnc sizes, but the guard makes the
                // policy equal by construction. See module docs.
                let _faer_par = NestedFaerParGuard::new();

                let in_base = d * in_coeff_len;
                let in_lower_a = &input.lower_a[in_base..in_base + in_coeff_len];
                let in_upper_a = &input.upper_a[in_base..in_base + in_coeff_len];
                let in_lower_err = input.lower_err_present[d].then(|| {
                    &input.lower_a_err.expect("flagged err plane")[in_base..in_base + in_coeff_len]
                });
                let in_upper_err = input.upper_err_present[d].then(|| {
                    &input.upper_a_err.expect("flagged err plane")[in_base..in_base + in_coeff_len]
                });

                let row_offset = k * num_outputs;
                let mut bias_lower_acc = vec![0.0f64; num_outputs];
                let mut bias_upper_acc = vec![0.0f64; num_outputs];
                let mut lower_nonfinite = vec![false; num_outputs];
                let mut upper_nonfinite = vec![false; num_outputs];

                for pos in 0..layout.num_positions {
                    let in_start = pos * layout.out_features;
                    let out_start = pos * in_features;

                    // SOUND certified-error base (#vnncomp-aw-soundness):
                    // re-accumulate S = |A|·|W| in f64 for THIS domain/position
                    // block — identical to the reference (same Mat contents, same
                    // shared `aw_f64_with_abssum`).
                    let a_lower_faer =
                        Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                            in_lower_a[i * bounds_inputs + in_start + j]
                        });
                    let a_upper_faer =
                        Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                            in_upper_a[i * bounds_inputs + in_start + j]
                        });
                    let (_, lower_s) = aw_f64_with_abssum(&a_lower_faer, weight_faer);
                    let (_, upper_s) = aw_f64_with_abssum(&a_upper_faer, weight_faer);

                    // Propagated incoming error: P[i,j] = Σ_k err_in[i,in_start+k]·|W[k,j]|.
                    let prop_lower = in_lower_err.map(|e| {
                        let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, kk| {
                            e[i * bounds_inputs + in_start + kk]
                        });
                        mat_mul(&blk, &w_abs)
                    });
                    let prop_upper = in_upper_err.map(|e| {
                        let blk = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, kk| {
                            e[i * bounds_inputs + in_start + kk]
                        });
                        mat_mul(&blk, &w_abs)
                    });

                    let g_lower = &gemm_lower[pos];
                    let g_upper = &gemm_upper[pos];
                    for i in 0..num_outputs {
                        let src_row = row_offset + i;
                        for j in 0..in_features {
                            let lower = g_lower[src_row * in_features + j];
                            let upper = g_upper[src_row * in_features + j];
                            let l_prop = prop_lower.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                            let u_prop = prop_upper.as_ref().map_or(0.0, |p| p[(i, j)] as f64);
                            let l_err = next_up_f32((gamma * lower_s[[i, j]] + l_prop) as f32);
                            let u_err = next_up_f32((gamma * upper_s[[i, j]] + u_prop) as f32);
                            // A stored coefficient is sound iff BOTH the
                            // coefficient and its certified error stay finite/in-
                            // range; otherwise the row degrades to ±inf bias.
                            if is_crown_coeff_safe(lower) && l_err.is_finite() {
                                res_lower_a[i * out_cols + out_start + j] = lower;
                                res_lower_err[i * out_cols + out_start + j] = l_err;
                            } else {
                                lower_nonfinite[i] = true;
                            }
                            if is_crown_coeff_safe(upper) && u_err.is_finite() {
                                res_upper_a[i * out_cols + out_start + j] = upper;
                                res_upper_err[i * out_cols + out_start + j] = u_err;
                            } else {
                                upper_nonfinite[i] = true;
                            }
                        }
                    }

                    if let Some(ref bias) = layer.bias {
                        let block = BiasBlockParams {
                            num_outputs,
                            out_features: layout.out_features,
                            col_offset: in_start,
                        };
                        accumulate_bias_f64(
                            &mut (bias_lower_acc.as_mut_slice(), bias_upper_acc.as_mut_slice()),
                            |i, j| in_lower_a[i * bounds_inputs + j],
                            |i, j| in_upper_a[i * bounds_inputs + j],
                            bias,
                            &block,
                        );
                    }
                }

                // Fold the propagated incoming error into the BIAS
                // (#vnncomp-aw-soundness): Σ_j err_in[i,j]·|bias[j]|, folded
                // OUTWARD in f64 BEFORE the directed cast — identical to the
                // reference's post-position loop.
                if let Some(ref bias) = layer.bias {
                    if let Some(le) = in_lower_err {
                        for i in 0..num_outputs {
                            let mut e = 0.0f64;
                            for j in 0..bias.len() {
                                e += le[i * bounds_inputs + j] as f64 * (bias[j] as f64).abs();
                            }
                            bias_lower_acc[i] -= e;
                        }
                    }
                    if let Some(ue) = in_upper_err {
                        for i in 0..num_outputs {
                            let mut e = 0.0f64;
                            for j in 0..bias.len() {
                                e += ue[i * bounds_inputs + j] as f64 * (bias[j] as f64).abs();
                            }
                            bias_upper_acc[i] += e;
                        }
                    }
                }

                let bias_base = d * num_outputs;
                if layer.bias.is_some() {
                    let lower_acc = Array1::from(bias_lower_acc);
                    let upper_acc = Array1::from(bias_upper_acc);
                    let old_lower_b = Array1::from_iter(
                        input.lower_b[bias_base..bias_base + num_outputs]
                            .iter()
                            .copied(),
                    );
                    let old_upper_b = Array1::from_iter(
                        input.upper_b[bias_base..bias_base + num_outputs]
                            .iter()
                            .copied(),
                    );
                    let (lb, ub) =
                        finalize_bias_directed(&lower_acc, &upper_acc, &old_lower_b, &old_upper_b);
                    for i in 0..num_outputs {
                        res_lower_b[i] = lb[i];
                        res_upper_b[i] = ub[i];
                    }
                } else {
                    res_lower_b.copy_from_slice(&input.lower_b[bias_base..bias_base + num_outputs]);
                    res_upper_b.copy_from_slice(&input.upper_b[bias_base..bias_base + num_outputs]);
                }

                // Non-finite row degrade: zero coefficients AND their certified
                // error (no double-count), ±inf bias — identical to the reference.
                let lower_affected = lower_nonfinite.iter().filter(|&&row| row).count();
                let upper_affected = upper_nonfinite.iter().filter(|&&row| row).count();
                if lower_affected > 0 || upper_affected > 0 {
                    debug!(
                    "Linear CROWN backward (SoA multi-domain GEMM, domain {}): overflow/magnitude \
                     in {}/{} lower rows, {}/{} upper rows — falling back to ±inf bias (#1932)",
                    d, lower_affected, num_outputs, upper_affected, num_outputs
                );
                    for i in 0..num_outputs {
                        if lower_nonfinite[i] {
                            for j in 0..out_cols {
                                res_lower_a[i * out_cols + j] = 0.0;
                                res_lower_err[i * out_cols + j] = 0.0;
                            }
                            res_lower_b[i] = f32::NEG_INFINITY;
                        }
                        if upper_nonfinite[i] {
                            for j in 0..out_cols {
                                res_upper_a[i * out_cols + j] = 0.0;
                                res_upper_err[i * out_cols + j] = 0.0;
                            }
                            res_upper_b[i] = f32::INFINITY;
                        }
                    }
                }
            },
        );

    Ok(SoaLinearBackwardOutput {
        rows: num_outputs,
        cols: out_cols,
        active: input.active.to_vec(),
        lower_a: out_lower_a,
        upper_a: out_upper_a,
        lower_b: out_lower_b,
        upper_b: out_upper_b,
        lower_a_err: out_lower_err,
        upper_a_err: out_upper_err,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::super::crown_batched_multi_domain::propagate_linear_batched_with_engine;
    use super::*;
    use crate::LinearBounds;
    use ny_core::NaiveCpuGemmEngine;

    /// Deterministic LCG f32 stream with occasional zeros / huge / tiny values.
    fn lcg_stream(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32);
            match (state >> 20) & 0xf {
                0 => 0.0,
                1 => (u - 0.5) * 3.0e37, // huge (exercises coeff-safe degrade)
                2 => (u - 0.5) * 1.0e-30,
                _ => (u - 0.5) * 4.0,
            }
        }
    }

    /// The SoA Linear backward must be BIT-IDENTICAL (coefficients, biases,
    /// AND certified-error matrices) to the per-domain reference
    /// `propagate_linear_batched_with_engine` across: bias / no-bias layers,
    /// err-present / err-absent domains, partial active masks, and huge
    /// coefficients driving the non-finite row degrade. #lsnc-batched-bwd.
    #[ntest::timeout(60000)]
    #[test]
    fn test_linear_backward_soa_matches_reference() {
        let rows = 7usize; // spec rows
        let cols = 8usize; // bounds inputs == weight rows
        let in_features = 6usize;
        let n_domains = 9usize;

        for (case, with_bias) in [(0u64, true), (1u64, false)] {
            let mut next = lcg_stream(0x9E3779B97F4A7C15 ^ case);
            let weight = Array2::from_shape_fn((cols, in_features), |_| next());
            let bias = with_bias.then(|| Array1::from_shape_fn(cols, |_| next()));
            let layer = LinearLayer::new(weight, bias).expect("valid layer");

            // Per-domain LinearBounds fixture: domain 3 inactive; errs on
            // domains {0, 2, 5, 8} (lower+upper), others exact (None).
            let active: Vec<bool> = (0..n_domains).map(|d| d != 3).collect();
            let mut per_domain: Vec<Option<LinearBounds>> = Vec::new();
            for d in 0..n_domains {
                if !active[d] {
                    per_domain.push(None);
                    continue;
                }
                let la = Array2::from_shape_fn((rows, cols), |_| next());
                let ua = Array2::from_shape_fn((rows, cols), |_| next());
                let lb = Array1::from_shape_fn(rows, |_| next());
                let ub = Array1::from_shape_fn(rows, |_| next());
                let mut bounds = LinearBounds {
                    lower_a: la,
                    lower_b: lb,
                    upper_a: ua,
                    upper_b: ub,
                    lower_a_err: None,
                    upper_a_err: None,
                };
                if d % 3 != 1 {
                    let le = Array2::from_shape_fn((rows, cols), |_| next().abs() * 1e-3);
                    let ue = Array2::from_shape_fn((rows, cols), |_| next().abs() * 1e-3);
                    bounds.set_coeff_err(le, ue);
                }
                per_domain.push(Some(bounds));
            }

            // Reference: active-only slice through the per-domain kernel.
            let active_refs: Vec<&LinearBounds> =
                per_domain.iter().filter_map(|b| b.as_ref()).collect();
            let engine = NaiveCpuGemmEngine;
            let reference = propagate_linear_batched_with_engine(&layer, &active_refs, &engine)
                .expect("reference backward");

            // SoA: pack the same fixture into flat planes.
            let coeff_len = rows * cols;
            let mut lower_a = vec![0.0f32; n_domains * coeff_len];
            let mut upper_a = vec![0.0f32; n_domains * coeff_len];
            let mut lower_b = vec![0.0f32; n_domains * rows];
            let mut upper_b = vec![0.0f32; n_domains * rows];
            let mut lower_err = vec![0.0f32; n_domains * coeff_len];
            let mut upper_err = vec![0.0f32; n_domains * coeff_len];
            let mut lower_err_present = vec![false; n_domains];
            let mut upper_err_present = vec![false; n_domains];
            for (d, b) in per_domain.iter().enumerate() {
                let Some(b) = b else { continue };
                lower_a[d * coeff_len..(d + 1) * coeff_len]
                    .copy_from_slice(b.lower_a.as_slice().unwrap());
                upper_a[d * coeff_len..(d + 1) * coeff_len]
                    .copy_from_slice(b.upper_a.as_slice().unwrap());
                lower_b[d * rows..(d + 1) * rows].copy_from_slice(b.lower_b.as_slice().unwrap());
                upper_b[d * rows..(d + 1) * rows].copy_from_slice(b.upper_b.as_slice().unwrap());
                if let Some(e) = b.lower_a_err.as_ref() {
                    lower_err[d * coeff_len..(d + 1) * coeff_len]
                        .copy_from_slice(e.as_slice().unwrap());
                    lower_err_present[d] = true;
                }
                if let Some(e) = b.upper_a_err.as_ref() {
                    upper_err[d * coeff_len..(d + 1) * coeff_len]
                        .copy_from_slice(e.as_slice().unwrap());
                    upper_err_present[d] = true;
                }
            }
            let soa_in = SoaLinearBackwardInput {
                n_domains,
                rows,
                cols,
                active: &active,
                lower_a: &lower_a,
                upper_a: &upper_a,
                lower_b: &lower_b,
                upper_b: &upper_b,
                lower_a_err: Some(&lower_err),
                upper_a_err: Some(&upper_err),
                lower_err_present: &lower_err_present,
                upper_err_present: &upper_err_present,
            };
            // min_chunk = 2 forces multi-task chunking with 9 domains.
            let soa =
                propagate_linear_batched_soa(&layer, &soa_in, &engine, 2).expect("SoA backward");

            assert_eq!(soa.rows, rows);
            let out_cols = soa.cols;
            let mut ref_iter = reference.iter();
            for d in 0..n_domains {
                if !active[d] {
                    assert!(!soa.active[d]);
                    continue;
                }
                let r = ref_iter.next().expect("reference result per active domain");
                let rc = rows * out_cols;
                let la = r.lower_a().as_slice().unwrap();
                let ua = r.upper_a().as_slice().unwrap();
                let le = r
                    .lower_a_err()
                    .expect("reference attaches err")
                    .as_slice()
                    .unwrap();
                let ue = r
                    .upper_a_err()
                    .expect("reference attaches err")
                    .as_slice()
                    .unwrap();
                for i in 0..rc {
                    assert_eq!(
                        soa.lower_a[d * rc + i].to_bits(),
                        la[i].to_bits(),
                        "case {case} domain {d} lower_a[{i}]"
                    );
                    assert_eq!(
                        soa.upper_a[d * rc + i].to_bits(),
                        ua[i].to_bits(),
                        "case {case} domain {d} upper_a[{i}]"
                    );
                    assert_eq!(
                        soa.lower_a_err[d * rc + i].to_bits(),
                        le[i].to_bits(),
                        "case {case} domain {d} lower_a_err[{i}]"
                    );
                    assert_eq!(
                        soa.upper_a_err[d * rc + i].to_bits(),
                        ue[i].to_bits(),
                        "case {case} domain {d} upper_a_err[{i}]"
                    );
                }
                for i in 0..rows {
                    assert_eq!(
                        soa.lower_b[d * rows + i].to_bits(),
                        r.lower_b()[i].to_bits(),
                        "case {case} domain {d} lower_b[{i}]"
                    );
                    assert_eq!(
                        soa.upper_b[d * rows + i].to_bits(),
                        r.upper_b()[i].to_bits(),
                        "case {case} domain {d} upper_b[{i}]"
                    );
                }
            }
            assert!(ref_iter.next().is_none());
        }
    }
}
