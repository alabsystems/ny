// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SoA (structure-of-arrays) multi-domain batching for the linear CROWN
//! backward — the Linear arm of the #lsnc-batched-bwd input-split fast lane
//! (design-doc slice S3, `docs/LSNC_BATCH_TENSOR_DESIGN.md`).
//!
//! # Relationship to the reference Linear arm: DELEGATION, not transcription
//!
//! The reference batched backward's Linear arm
//! (`backward_core::dispatch_node_backward`) calls
//! [`LinearLayer::propagate_linear_pollable_cpu`] once per active domain — the
//! certified f64 `A·W` contraction, published with a `γ_n^{f64}·S`
//! coefficient-error certificate.
//!
//! This lane CALLS THAT SAME FUNCTION, per domain, in the same domain order.
//! It owns only the SoA plumbing: reading each domain's pending coefficient
//! block out of the contiguous `[B, R, W]` batch tensors, dispatching the
//! shared kernel over coarse rayon chunks on the domain axis, and writing the
//! results back into flat output planes. **No arithmetic lives in this file.**
//! Bit-identity with the reference is therefore true by construction rather
//! than by review.
//!
//! # Why (the defect this shape exists to prevent)
//!
//! This module used to be a hand-written *transcription* of
//! [`super::crown_batched_multi_domain::propagate_linear_batched_with_engine`]:
//! a stacked f32 engine GEMM published with a `γ_n^{f32}·S` certificate. When
//! `backward_core.rs`'s Linear arm moved onto the certified f64 contraction
//! (removing a deadline-presence arithmetic split), the transcription did not
//! move with it, and `propagate_linear_batched_with_engine` was left with zero
//! production callers — this lane became a faithful copy of nothing.
//!
//! The observable consequence was that the fast lane published bounds
//! differing from its own reference by up to 8 ULP on
//! `test_input_split_batched_bwd_full_pipeline_bit_identical_lsnc_s3`. The
//! divergence was always in the LOOSER direction (`γ_n^{f32}` is ~2^29 wider
//! than `γ_n^{f64}`, and that term dominates the point-coefficient wobble for
//! every contraction width `n >= 2`), so no proof was ever unsound — but the
//! lane's "strictly a performance transform, never a bound change" contract
//! was false, and the certificate it published was ~2^29 looser than the one
//! the reference could justify.
//!
//! A copy can drift; a call cannot. Do NOT reintroduce a private arithmetic
//! path here — if the Linear stacking is ever worth restoring for throughput,
//! it belongs in the shared kernel where BOTH lanes get it.
//!
//! A [`NestedFaerParGuard`] is held inside each worker so the shared kernel's
//! faer f64 products run under the SAME faer parallelism policy as the
//! reference's driver-thread calls (`current_par()` would otherwise force
//! `Par::Seq` inside a rayon worker). The pollable-CPU contraction is
//! documented row-count- and par-policy-independent (see
//! `crown_single::aw_f64_with_abssum_pollable_cpu_with_chunk_macs`), so the
//! guard makes the equality hold by construction rather than by matrix size.
//!
//! Parity is pinned by `test_linear_backward_soa_matches_reference` below
//! (against the very same per-domain reference call) and by the full-pipeline
//! gate test in `tests_soundness.rs`.

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use rayon::prelude::*;

use super::layout::resolve_backward_layout;
use super::LinearLayer;
use crate::faer_parallelism::NestedFaerParGuard;
use crate::LinearBounds;

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

/// The Linear arm of the #lsnc-batched-bwd SoA fast lane.
///
/// Delegates EVERY domain to the shared reference kernel
/// [`LinearLayer::propagate_linear_pollable_cpu`] — the same call the
/// reference loop (`backward_core::dispatch_node_backward`) makes — so the
/// published coefficients, biases and certified-error planes are bit-identical
/// to the reference by construction. See the module docs for why this is a
/// call and not a copy.
///
/// `min_chunk` is the minimum number of domains per rayon task (coarse
/// chunking; NO per-domain fan-out). The fast lane only ever runs without a
/// deadline (`batched_bwd::try_backward_soa` declines on any finite one), so
/// `None` is passed through as the kernel's deadline — matching the reference,
/// which is handed the same `None`.
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

    let weight_rows = layer.weight.nrows();
    let in_features = layer.weight.ncols();
    let layout = resolve_backward_layout(num_outputs, bounds_inputs, weight_rows, in_features)?;
    let out_cols = layout.total_in_features;
    let coeff_len = num_outputs * out_cols;

    let mut out = SoaLinearBackwardOutput {
        rows: num_outputs,
        cols: out_cols,
        active: input.active.to_vec(),
        lower_a: vec![0.0f32; n_domains * coeff_len],
        upper_a: vec![0.0f32; n_domains * coeff_len],
        lower_b: vec![0.0f32; n_domains * num_outputs],
        upper_b: vec![0.0f32; n_domains * num_outputs],
        lower_a_err: vec![0.0f32; n_domains * coeff_len],
        upper_a_err: vec![0.0f32; n_domains * coeff_len],
    };
    if !input.active.iter().any(|&active| active) {
        return Ok(out);
    }

    // Phase 1 — run the SHARED per-domain reference kernel, coarse-chunked over
    // the domain axis. Domains are independent (each reads only its own pending
    // block and produces only its own result), so chunking changes scheduling
    // and nothing else.
    let per_domain: Vec<Option<Result<LinearBounds>>> = (0..n_domains)
        .into_par_iter()
        .with_min_len(min_chunk.max(1))
        .map(|d| {
            if !input.active[d] {
                return None;
            }
            // Match the reference's faer parallelism policy (driver thread).
            let _faer_par = NestedFaerParGuard::new();
            Some(
                domain_pending_bounds(input, d, num_outputs, bounds_inputs).and_then(|bounds| {
                    layer
                        .propagate_linear_pollable_cpu(&bounds, Some(engine), None)
                        .map(std::borrow::Cow::into_owned)
                }),
            )
        })
        .collect();

    // Phase 2 — scatter into the flat output planes (pure copy, no arithmetic).
    for (d, slot) in per_domain.into_iter().enumerate() {
        let Some(result) = slot else { continue };
        let bounds = result?;
        if bounds.lower_a().dim() != (num_outputs, out_cols) {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, out_cols],
                got: bounds.lower_a().shape().to_vec(),
            });
        }
        let coeff = d * coeff_len..(d + 1) * coeff_len;
        let bias = d * num_outputs..(d + 1) * num_outputs;
        copy_from(&mut out.lower_a[coeff.clone()], bounds.lower_a().iter());
        copy_from(&mut out.upper_a[coeff.clone()], bounds.upper_a().iter());
        copy_from(&mut out.lower_b[bias.clone()], bounds.lower_b().iter());
        copy_from(&mut out.upper_b[bias], bounds.upper_b().iter());
        // `None` on a certified-error plane is the EXACT marker (error 0), which
        // is exactly what the freshly zeroed output plane already holds, so the
        // absent case needs no write. `BatchSlot::from_linear_out` records
        // "present" for every active domain, matching the reference Linear arm,
        // which attaches error unconditionally (`propagates_coeff_err`, I-D5).
        if let Some(err) = bounds.lower_a_err() {
            copy_from(&mut out.lower_a_err[coeff.clone()], err.iter());
        }
        if let Some(err) = bounds.upper_a_err() {
            copy_from(&mut out.upper_a_err[coeff], err.iter());
        }
    }

    Ok(out)
}

fn copy_from<'a>(dst: &mut [f32], src: impl Iterator<Item = &'a f32>) {
    for (slot, &value) in dst.iter_mut().zip(src) {
        *slot = value;
    }
}

/// Materialize domain `d`'s pending [`LinearBounds`] from the SoA planes.
///
/// Struct-literal construction (not `LinearBounds::new`) because the fast lane
/// must hand the shared kernel EXACTLY what the reference loop hands it: the
/// reference reads these objects straight out of `IndexedPendingLinearBounds`
/// with no revalidation, and `new`'s NaN/Inf rejection would turn a row that
/// the kernel degrades in place (zeroed coefficients + ±inf bias, #1932/#2681)
/// into a hard error — a behaviour difference, i.e. the very class of drift
/// this module was rebuilt to make impossible.
fn domain_pending_bounds(
    input: &SoaLinearBackwardInput<'_>,
    d: usize,
    rows: usize,
    cols: usize,
) -> Result<LinearBounds> {
    let rc = rows * cols;
    let base = d * rc;
    let bias_base = d * rows;
    let coeff = |plane: &[f32]| -> Result<Array2<f32>> {
        Array2::from_shape_vec((rows, cols), plane[base..base + rc].to_vec()).map_err(|err| {
            NyError::InvalidSpec(format!("SoA Linear pending block for domain {d}: {err}"))
        })
    };
    let bias = |plane: &[f32]| Array1::from(plane[bias_base..bias_base + rows].to_vec());
    let err = |plane: Option<&[f32]>, present: bool| -> Result<Option<Array2<f32>>> {
        if !present {
            return Ok(None);
        }
        let plane = plane.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "SoA Linear certified-error plane flagged present but absent (domain {d})"
            ))
        })?;
        coeff(plane).map(Some)
    };
    Ok(LinearBounds {
        lower_a: coeff(input.lower_a)?,
        lower_b: bias(input.lower_b),
        upper_a: coeff(input.upper_a)?,
        upper_b: bias(input.upper_b),
        lower_a_err: err(input.lower_a_err, input.lower_err_present[d])?,
        upper_a_err: err(input.upper_a_err, input.upper_err_present[d])?,
    })
}

#[cfg(test)]
mod tests {
    use ny_core::NaiveCpuGemmEngine;

    use super::*;

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
    /// AND certified-error matrices) to THE REFERENCE THE FAST LANE ACTUALLY
    /// MIRRORS — `LinearLayer::propagate_linear_pollable_cpu`, the call
    /// `backward_core::dispatch_node_backward`'s Linear arm makes — across:
    /// bias / no-bias layers, err-present / err-absent domains, partial active
    /// masks, and huge coefficients driving the non-finite row degrade.
    /// #lsnc-batched-bwd.
    ///
    /// This test used to compare against
    /// `crown_batched_multi_domain::propagate_linear_batched_with_engine` (a
    /// stacked f32 engine GEMM with a `γ_n^{f32}·S` certificate). It stayed
    /// GREEN through the entire window in which the fast lane published bounds
    /// up to 8 ULP away from what the batched backward's own Linear arm
    /// published, because that arm had moved to the certified f64 contraction
    /// and this comparand had not — it had zero production callers left. The
    /// comparand is now the live reference, so this test cannot pass while the
    /// lane and its reference disagree. Do not point it back at a function
    /// nothing calls.
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

            // Reference: the shared per-domain kernel, called exactly as the
            // reference batched backward calls it (active domains in ascending
            // domain order, no deadline).
            let active_refs: Vec<&LinearBounds> =
                per_domain.iter().filter_map(|b| b.as_ref()).collect();
            let engine = NaiveCpuGemmEngine;
            let reference: Vec<LinearBounds> = active_refs
                .iter()
                .copied()
                .map(|bounds| {
                    layer
                        .propagate_linear_pollable_cpu(bounds, Some(&engine), None)
                        .expect("reference backward")
                        .into_owned()
                })
                .collect();

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
