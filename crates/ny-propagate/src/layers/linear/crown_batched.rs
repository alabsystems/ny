// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for linear layers.
//!
//! Contains both the N-D batched path (single domain with batch dimensions)
//! and the multi-domain GPU-batched path (multiple domains stacked for one
//! large GEMM call).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

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

/// One uninterrupted faer f64 dispatch on the #pollable-faer batched host
/// path: `rows_per_block·mid_dim·in_dim ≤ 2^26` MACs per block (times two —
/// `A·W` plus `|A|·|W|`), with the bounded facade's deadline polled BETWEEN
/// blocks. MEASURED on the GB10 (faer 0.24.0, release, the exact chunk loop):
/// one block at this quantum is **2.75 ms** on the driver thread
/// (`Par::rayon`) and **~6 ms** in the `Par::Seq` rayon-worker regime
/// (5.8 ms for the scratch-limited 199×12288 wide-head class) — comfortably
/// inside the watchdog's tolerance, while each block is still 2·2^26 MACs so
/// faer's throughput fully survives (call overhead is µs-scale). The
/// in-tree probe `pollable_faer_block_timing_probe` re-pins this on demand.
/// The 64 MiB faer scratch ceiling from the crown_single deadline path
/// applies unchanged (a bounded facade is a MEMORY authority too), so the
/// row count per block is `min(MAC-limited, scratch-limited)`.
pub(crate) const POLLABLE_FAER_BLOCK_MACS: usize = 1 << 26;

/// #pollable-faer gate: `NY_POLLABLE_FAER=1` arms the pollable chunked-faer
/// host implementation for BOUNDED (deadline-facade) batched Linear CROWN.
///
/// Default OFF for the A/B — gate-off is byte-identical to the historical
/// refusal below. The goal is default-ON after measurement: this arm changes
/// NO bound math — it runs the SAME certified f64 `A·W`/`|A|·|W|` pair with
/// the SAME `γ_n·S + cast_gap` publication as the engine-absent host path,
/// chunked on the output-row axis only, which is bit-identical (pinned by
/// `pollable_faer_batched_bit_identical_to_cpu_and_scalar`). It is a pure
/// throughput/coverage restoration for deadline-carrying CROWN: without it
/// every bounded caller is refused here and degrades (partial-CROWN fallback
/// or the facade's scalar per-MAC GEMM — the 6f49a660 deadline-cliff class
/// that cost dist_shift index5291 and cgan row 7 their banked unsats).
///
/// DECLARED lever (ratchet discipline): `ny_levers::decls::pollable_faer::
/// POLLABLE_FAER`, `LeverKind::Text` precisely so the legacy truthy parser
/// below — not an exact-`"1"` Boolean — stays authoritative.
///
/// The gate's unit-tested truthy predicate — the spec of its arming rule,
/// preserved verbatim through the ny-levers migration: armed by any value
/// that trims non-empty and is neither exact `0` nor case-insensitive
/// `false`.
pub(crate) fn pollable_faer_token_arms(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

/// #pollable-faer gate, latched once (hot path).
fn pollable_faer_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        // Keep the legacy truthy predicate in the production path: an
        // environment token decides through it verbatim; only absence falls
        // through to the typed preset cell and then the shipped-dark default.
        ny_levers::gate::resolve_raw_predicate(
            &ny_levers::decls::pollable_faer::POLLABLE_FAER,
            crate::preset_gates::pollable_faer_cell(),
            false,
            pollable_faer_token_arms,
        )
    })
}

/// #pollable-faer telemetry: cumulative row blocks dispatched, deadline polls
/// observed, and typed deadline expiries surfaced by the bounded batched arm.
static POLLABLE_FAER_BLOCKS: AtomicU64 = AtomicU64::new(0);
static POLLABLE_FAER_POLLS: AtomicU64 = AtomicU64::new(0);
static POLLABLE_FAER_EXPIRIES: AtomicU64 = AtomicU64::new(0);

/// Rate-limited (≥5 s between lines) stderr telemetry for the #pollable-faer
/// arm — the lever's own observable move (measurement-parity rule R9: an A/B
/// without `blocks > 0` in the armed arm is a wrong-lane null, not a result).
fn pollable_faer_telemetry_flush() {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    static LAST_PRINT_SECS: AtomicU64 = AtomicU64::new(0);
    const MIN_SECS_BETWEEN_LINES: u64 = 5;

    let epoch = *EPOCH.get_or_init(Instant::now);
    // +1 so the first flush (elapsed 0) still beats the initial sentinel 0.
    let now_secs = epoch.elapsed().as_secs().saturating_add(1);
    let last = LAST_PRINT_SECS.load(Ordering::Relaxed);
    if now_secs.saturating_sub(last) < MIN_SECS_BETWEEN_LINES && last != 0 {
        return;
    }
    if LAST_PRINT_SECS
        .compare_exchange(last, now_secs, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    eprintln!(
        "[pollable-faer] blocks={} polls={} expiries={}",
        POLLABLE_FAER_BLOCKS.load(Ordering::Relaxed),
        POLLABLE_FAER_POLLS.load(Ordering::Relaxed),
        POLLABLE_FAER_EXPIRIES.load(Ordering::Relaxed),
    );
}

/// Observe the bounded facade's deadline once (telemetry-accounted).
///
/// The facade's typed `DeadlineExceeded` propagates unchanged — the identical
/// error shape its own scalar per-MAC GEMM surfaces today — and no partial
/// result is ever published past a failed poll.
#[inline]
fn bounded_poll(engine: &dyn GemmEngine) -> Result<()> {
    POLLABLE_FAER_POLLS.fetch_add(1, Ordering::Relaxed);
    match engine.poll_crown_backward_deadline() {
        Ok(()) => Ok(()),
        Err(error) => {
            if error.is_deadline_exceeded() {
                POLLABLE_FAER_EXPIRIES.fetch_add(1, Ordering::Relaxed);
                pollable_faer_telemetry_flush();
            }
            Err(error)
        }
    }
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
    propagate_linear_batched_maybe_engine_with_pollable(
        layer,
        bounds,
        engine,
        pollable_faer_enabled(),
        POLLABLE_FAER_BLOCK_MACS,
    )
}

/// Testable core of [`propagate_linear_batched_maybe_engine`]. Production
/// always passes (`pollable_faer_enabled()`, [`POLLABLE_FAER_BLOCK_MACS`]);
/// tests pass explicit arms and a tiny block quantum to force multi-block
/// dispatch without touching process environment.
pub(crate) fn propagate_linear_batched_maybe_engine_with_pollable(
    layer: &LinearLayer,
    bounds: &BatchedLinearBounds,
    engine: Option<&dyn GemmEngine>,
    pollable_faer: bool,
    pollable_block_macs: usize,
) -> Result<BatchedLinearBounds> {
    debug!("Linear layer batched CROWN backward propagation");

    // A bounded host facade (deadline authority) forbids the ordinary local
    // CPU fallback. Historically this lane REFUSED outright; the #pollable-faer
    // arm (gate above) instead runs the same certified host computation with
    // the facade's deadline polled between row blocks. Fail closed: a facade
    // without the pollable host-GEMM capability keeps the historical refusal
    // even when the gate is armed.
    let bounded_host: Option<&dyn GemmEngine> =
        engine.filter(|engine| engine.forbids_unbounded_cpu_fallback());
    if bounded_host
        .is_some_and(|engine| !(pollable_faer && engine.provides_deadline_pollable_host_gemm()))
    {
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
        if let Some(bounded) = bounded_host {
            // #pollable-faer bounded arm: the facade's errors are terminal by
            // contract (its deadline refusal must abort, and a local fallback
            // would allocate the very buffers the facade caps), so there is no
            // fallback leg here.
            compute_batched_linear_coefficients_pollable_faer(
                layer,
                lower_a_3d,
                upper_a_3d,
                in_lower_err_3d.as_ref().map(|v| v.view()),
                in_upper_err_3d.as_ref().map(|v| v.view()),
                total_batch,
                out_dim,
                mid_dim,
                in_dim,
                bounded,
                pollable_block_macs,
            )?
        } else if let Some(engine) = engine {
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

    // Track which output rows have non-finite coefficients.
    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];
    // #pollable-faer: the scan and bias folds below are the remaining
    // host-side stretches under a bounded facade; observe its deadline at a
    // coarse cadence (they are linear/quadratic, not GEMM-scale).
    const BOUNDED_SCAN_POLL_ROWS: usize = 1 << 10;
    for row in 0..total_rows {
        if let Some(bounded) = bounded_host {
            if row.is_multiple_of(BOUNDED_SCAN_POLL_ROWS) {
                bounded_poll(bounded)?;
            }
        }
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

    // For rows with non-finite or near-overflow A-matrix coefficients,
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

    // Compute bias contribution: A @ bias + old_b (shared helper).
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
            if let Some(bounded) = bounded_host {
                bounded_poll(bounded)?;
            }
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

    // Override bias to ±inf for rows with non-finite A-matrix coefficients.
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

    // CROWN backward NaN firewall: conservative fallback instead of aborting
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
        aw_f64_with_abssum_for_layer, gamma_n_f64, incoming_error_product,
    };

    let total_rows = total_batch * out_dim;
    let mut result_lower_flat = vec![0.0_f32; total_rows * in_dim];
    let mut result_upper_flat = vec![0.0_f32; total_rows * in_dim];
    let mut lower_err_flat = vec![0.0_f32; total_rows * in_dim];
    let mut upper_err_flat = vec![0.0_f32; total_rows * in_dim];
    let gamma = gamma_n_f64(mid_dim);

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

        // #linear-widened-weight-cache: same operands and same faer arm; the
        // layer's immutable weight is widened to f64 once instead of on every
        // batch of every call, so the published bits are unchanged.
        let (lower_a64, lower_s) = aw_f64_with_abssum_for_layer(layer, &a_lower_faer);
        let (upper_a64, upper_s) = aw_f64_with_abssum_for_layer(layer, &a_upper_faer);

        // Propagated incoming error P[i,j] = Σ_k err_in[i,k]·|W[k,j]| (per batch).
        //
        // #linear-abs-weight-cache: `|W|` (`mid_dim × in_dim`, and `mid_dim ==
        // layer.out_features()` is enforced at the batched entry) comes from
        // the layer's cache instead of being rebuilt once per CALL, and is read
        // only when an incoming error plane exists. `abs` on the same immutable
        // `weight_faer` — identical bits, identical shape.
        let prop_lower = match in_lower_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(
                    &block,
                    0,
                    mid_dim,
                    layer.weight_abs_faer(),
                    None,
                )?)
            }
            None => None,
        };
        let prop_upper = match in_upper_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(
                    &block,
                    0,
                    mid_dim,
                    layer.weight_abs_faer(),
                    None,
                )?)
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

/// Guarded output-buffer allocation for the bounded (#pollable-faer) arm: a
/// bounded host facade is a memory authority as well as a deadline authority,
/// so this lane never allocates past the CPU dense-CROWN budget and surfaces
/// a structured refusal instead of entering the allocator unbounded.
fn pollable_faer_output_buffer(elements: usize) -> Result<Vec<f32>> {
    const SITE: &str = "bounded batched Linear CROWN output";
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let required_bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| NyError::InvalidSpec(format!("{SITE}: byte size overflow")))?;
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    values.resize(elements, 0.0);
    Ok(values)
}

/// Row-blocked incoming-error composition for the bounded (#pollable-faer)
/// arm: `P[i,j] = Σ_k err_in[i,k]·|W[k,j]|` computed by the SAME
/// `incoming_error_product` the unbounded host path uses, split on the output
/// row axis with the facade's deadline polled between blocks.
///
/// BIT-IDENTITY vs the single whole-matrix call: both of
/// `incoming_error_product`'s arms are per-output-row independent — the GEMM
/// lane tiles rows only (full `k` per tile; faer per-element reductions are
/// row-count independent, the `aw_batched_matches_per_domain_bit_for_bit`
/// property) and the scalar stepped arm accumulates each `(i,j)` separately —
/// and the two arms publish identical f32 values for finite inputs (pinned by
/// `incoming_error_gemm_publishes_identical_values`), so a per-block arm
/// choice flip (e.g. a non-finite element confined to one block) cannot
/// change published values either. Pinned for this entry by the batched
/// three-way bit-identity test.
fn incoming_error_product_row_blocked(
    error_block: &Array2<f32>,
    w_abs: &Mat<f32>,
    in_dim: usize,
    block_macs: usize,
    engine: &dyn GemmEngine,
) -> Result<Array2<f32>> {
    use crate::layers::linear::crown_single::incoming_error_product;

    let rows = error_block.nrows();
    let contraction = error_block.ncols();
    let row_macs = contraction.saturating_mul(in_dim).max(1);
    let rows_per_block = (block_macs / row_macs).max(1);
    let mut out = Array2::<f32>::zeros((rows, in_dim));
    let mut row_start = 0usize;
    while row_start < rows {
        bounded_poll(engine)?;
        let row_end = row_start.saturating_add(rows_per_block).min(rows);
        let sub = error_block.slice(s![row_start..row_end, ..]).to_owned();
        let block = incoming_error_product(&sub, 0, contraction, w_abs, None)?;
        out.slice_mut(s![row_start..row_end, ..]).assign(&block);
        POLLABLE_FAER_BLOCKS.fetch_add(1, Ordering::Relaxed);
        row_start = row_end;
    }
    bounded_poll(engine)?;
    Ok(out)
}

/// Bounded-host (#pollable-faer) batched `A·W`: the SAME certified f64
/// computation as [`compute_batched_linear_coefficients_cpu`] — identical
/// operands, the identical faer arm of `aw_f64_with_abssum`, the identical
/// `cast_gap + γ_n·S + prop` publication — with every dense stretch chunked on
/// the output-row axis and the caller facade's deadline polled BETWEEN blocks
/// ([`GemmEngine::poll_crown_backward_deadline`]).
///
/// Results are BIT-IDENTICAL to the unbounded host (faer) path — row-blocking
/// never touches a row's `k`-reduction; see `aw_f64_with_abssum_chunked_poll`.
/// (Above the process-global f64-engine offload threshold the unbounded path
/// may itself take an accelerator route that is certificate-equivalent rather
/// than bit-identical; the order-independent `γ_n·S` certificate covers every
/// route.) Arming this lane changes no bound math — it is a pure
/// throughput/coverage restoration. On expiry the facade's typed
/// `DeadlineExceeded` propagates
/// with no partial results, exactly as the scalar per-MAC paths abort today.
/// Shapes the block budget cannot bound (a single row's `mid_dim·in_dim` over
/// the quantum, the faer scratch ceiling, or the `NY_NAIVE_F64_AW` parity
/// kill-switch) refuse with the historical `UnsupportedOp`, so callers'
/// degradation contract is unchanged for exactly that subset.
#[allow(clippy::too_many_arguments)]
fn compute_batched_linear_coefficients_pollable_faer(
    layer: &LinearLayer,
    lower_a_3d: ArrayView3<'_, f32>,
    upper_a_3d: ArrayView3<'_, f32>,
    in_lower_err_3d: Option<ArrayView3<'_, f32>>,
    in_upper_err_3d: Option<ArrayView3<'_, f32>>,
    total_batch: usize,
    out_dim: usize,
    mid_dim: usize,
    in_dim: usize,
    engine: &dyn GemmEngine,
    block_macs: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use crate::layers::linear::crown_single::{
        aw_f64_with_abssum_chunked_poll_for_layer, gamma_n_f64,
    };

    bounded_poll(engine)?;
    let total_rows = total_batch * out_dim;
    let elements = total_rows.checked_mul(in_dim).ok_or_else(|| {
        NyError::InvalidSpec("bounded batched Linear CROWN output size overflow".to_string())
    })?;
    let mut result_lower_flat = pollable_faer_output_buffer(elements)?;
    bounded_poll(engine)?;
    let mut result_upper_flat = pollable_faer_output_buffer(elements)?;
    bounded_poll(engine)?;
    let mut lower_err_flat = pollable_faer_output_buffer(elements)?;
    bounded_poll(engine)?;
    let mut upper_err_flat = pollable_faer_output_buffer(elements)?;
    bounded_poll(engine)?;

    let gamma = gamma_n_f64(mid_dim);
    bounded_poll(engine)?;

    for batch_idx in 0..total_batch {
        bounded_poll(engine)?;
        let a_lower = lower_a_3d.slice(s![batch_idx, .., ..]);
        let a_upper = upper_a_3d.slice(s![batch_idx, .., ..]);

        let a_lower_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_lower[[i, j]]);
        let a_upper_faer = Mat::<f32>::from_fn(out_dim, mid_dim, |i, j| a_upper[[i, j]]);

        let mut poll = || bounded_poll(engine);
        // #linear-widened-weight-cache: identical chunking, identical polls,
        // identical operands — the per-call `W` / `|W|` widening is taken from
        // the layer instead of rebuilt.
        let Some((lower_a64, lower_s, lower_blocks)) =
            aw_f64_with_abssum_chunked_poll_for_layer(layer, &a_lower_faer, block_macs, &mut poll)?
        else {
            return Err(NyError::UnsupportedOp(
                "bounded Linear batched CROWN: shape exceeds the pollable-faer block budget".into(),
            ));
        };
        let Some((upper_a64, upper_s, upper_blocks)) =
            aw_f64_with_abssum_chunked_poll_for_layer(layer, &a_upper_faer, block_macs, &mut poll)?
        else {
            return Err(NyError::UnsupportedOp(
                "bounded Linear batched CROWN: shape exceeds the pollable-faer block budget".into(),
            ));
        };
        POLLABLE_FAER_BLOCKS.fetch_add((lower_blocks + upper_blocks) as u64, Ordering::Relaxed);

        // Propagated incoming error P[i,j] = Σ_k err_in[i,k]·|W[k,j]| (per batch).
        //
        // #linear-abs-weight-cache: `|W|` from the layer's cache, exactly as
        // the unbounded host path above. POLL CADENCE IS UNCHANGED — the count
        // and order of `bounded_poll` calls is the same, and the one-time fill
        // this may perform on a layer's first error-carrying call is the SAME
        // `mid_dim × in_dim` fill that used to run unconditionally in this
        // function on every call, moved into a neighbouring gap rather than
        // added to one.
        let prop_lower = match in_lower_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product_row_blocked(
                    &block,
                    layer.weight_abs_faer(),
                    in_dim,
                    block_macs,
                    engine,
                )?)
            }
            None => None,
        };
        let prop_upper = match in_upper_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product_row_blocked(
                    &block,
                    layer.weight_abs_faer(),
                    in_dim,
                    block_macs,
                    engine,
                )?)
            }
            None => None,
        };

        // Publication: identical arithmetic and rounding to the unbounded host
        // path (quadratic, one poll per batch above).
        bounded_poll(engine)?;
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
    bounded_poll(engine)?;
    pollable_faer_telemetry_flush();

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
        aw_f64_with_abssum_for_layer, gamma_n_f32, incoming_error_product,
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
        // #linear-widened-weight-cache: this reference product is the same faer
        // arm over the same operands (`layer.weight_faer()`), with the layer's
        // cached f64 widening in place of a per-batch rebuild.
        let (lower_reference, lower_s) = aw_f64_with_abssum_for_layer(layer, &a_lower_faer);
        let (upper_reference, upper_s) = aw_f64_with_abssum_for_layer(layer, &a_upper_faer);
        // Propagated incoming error per batch.
        //
        // #linear-abs-weight-cache: `|W|` from the layer's cache rather than a
        // per-call rebuild, read only when an incoming error plane exists. The
        // engine `gemm_f32` above still takes `layer.weight` as a slice and is
        // untouched.
        let prop_lower = match in_lower_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(
                    &block,
                    0,
                    mid_dim,
                    layer.weight_abs_faer(),
                    None,
                )?)
            }
            None => None,
        };
        let prop_upper = match in_upper_err_3d.as_ref() {
            Some(error) => {
                let block =
                    Array2::from_shape_fn((out_dim, mid_dim), |(i, k)| error[[batch_idx, i, k]]);
                Some(incoming_error_product(
                    &block,
                    0,
                    mid_dim,
                    layer.weight_abs_faer(),
                    None,
                )?)
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

#[cfg(test)]
mod abs_weight_cache_engagement_tests {
    use super::{propagate_linear_batched, BatchedLinearBounds, LinearLayer};
    use ndarray::{arr2, Array1, ArrayD, IxDyn};

    /// `out_features = 3` (the batched `mid_dim`), `in_features = 2`.
    fn fixture_layer() -> LinearLayer {
        LinearLayer::new(
            arr2(&[[1.0_f32, -2.0], [-3.0, 4.0], [0.5, -0.25]]),
            Some(Array1::from_vec(vec![0.125_f32, -0.5, 2.0])),
        )
        .expect("batched abs-cache fixture layer builds")
    }

    /// Coefficients shaped `[batch = 2, out_dim = 2, mid_dim = 3]`.
    fn fixture_bounds() -> BatchedLinearBounds {
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                1.0_f32, -1.0, 2.0, 0.5, 2.0, -1.5, -0.25, 1.25, 0.75, 3.0, -0.5, 1.0,
            ],
        )
        .expect("batched coefficient fixture");
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.25_f32, -0.75, 0.5, -0.25])
            .expect("batched bias fixture");
        BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![2, 3], vec![2, 2])
            .expect("batched bounds fixture builds")
    }

    /// #linear-abs-weight-cache ENGAGEMENT for the batched CPU arm
    /// (`compute_batched_linear_coefficients_cpu`), in BOTH directions: the
    /// layer's `|W|` cell must fill when the incoming bounds carry a certified
    /// error plane, and must stay COLD when they do not — the second half is
    /// the `mid_dim × in_dim` allocation this arm used to build and drop unused
    /// on every error-free call.
    ///
    /// FAILS CLOSED. If a future edit reinstates a local `Mat::from_fn` here,
    /// the `is_some()` assertion fails rather than the change passing silently.
    #[test]
    fn batched_cpu_backward_fills_the_abs_weight_cache_only_with_an_error_plane() {
        let without_error_plane = fixture_layer();
        let plain = fixture_bounds();
        assert!(
            without_error_plane.weight_abs_faer.get().is_none(),
            "fixture layer must start with a COLD |W| cell"
        );
        let _plain_result = propagate_linear_batched(&without_error_plane, &plain)
            .expect("batched backward without error plane");
        assert!(
            without_error_plane.weight_abs_faer.get().is_none(),
            "batched bounds carrying NO certified-error plane must not build |W| at all"
        );

        let with_error_plane = fixture_layer();
        let mut carrying = fixture_bounds();
        carrying.set_coeff_err(
            ArrayD::from_elem(IxDyn(&[2, 2, 3]), 0.125_f32),
            ArrayD::from_elem(IxDyn(&[2, 2, 3]), 0.25_f32),
        );
        assert!(
            carrying.lower_a_err.is_some() && carrying.upper_a_err.is_some(),
            "set_coeff_err must have attached both planes, else the assertion below \
             would pass for the wrong reason"
        );
        assert!(with_error_plane.weight_abs_faer.get().is_none());
        let _carried_result = propagate_linear_batched(&with_error_plane, &carrying)
            .expect("batched backward with error plane");
        assert!(
            with_error_plane.weight_abs_faer.get().is_some(),
            "the batched CPU arm must compose Σ err_in·|W| through the layer's cache"
        );
    }
}
