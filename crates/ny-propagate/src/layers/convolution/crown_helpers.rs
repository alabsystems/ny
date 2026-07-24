// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{s, Array1, Array2, ArrayD};
use ny_core::{is_crown_coeff_safe, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use tracing::{debug, warn};

use crate::LinearBounds;

/// SOUND outward widening for a conv-family IBP FORWARD (#vnncomp-aw-soundness).
///
/// The plain conv/transpose `propagate_ibp` accumulates each output over `macs` products
/// in round-to-nearest f32 (no f64, no directed rounding), so it can deviate from the true
/// value by the Higham bound `γ_macs · Σ_k |W_ok|·|x_k|`, which under cancellation vastly
/// exceeds the generic 1-ULP `round_for_soundness` widening — yielding a box that EXCLUDES
/// the true value (a false-proof on the intermediate-bound / verdict path).
///
/// `y` is the plain f32 forward; `s` is the per-output abssum
/// `S_o = Σ_k |W_ok|·max(|x_l_k|,|x_u_k|)` (same shape as `y`), obtained by running the SAME
/// interval forward with `|kernel|` (so `W+ = |W|`, `W- = 0`) on the degenerate `max(|l|,|u|)`
/// box — this handles 1D/2D/grouped/transpose uniformly. `macs` is an UPPER bound on the
/// per-output f32 accumulation depth (`(in_c/groups)·∏kernel_spatial`).
///
/// Folds the certified error `err_o = up(γ_{macs+2}·S_safe + 2u·|y_o|)` OUTWARD, where
/// `S_safe = S_o · 1/(1−γ_macs) ≥ S_true` corrects the round-to-nearest f32 computation of `S`
/// itself (which can fall short of `S_true` by up to its own `γ_macs` error). `+2` covers the
/// `W+/W-` combine and bias add; `2u·|y|` covers those final roundings. The returned box
/// strictly encloses the true conv output — looser only ⇒ Timeout, never a false proof.
pub(crate) fn higham_widen_ibp(
    y: &BoundedTensor,
    s: &ArrayD<f32>,
    macs: usize,
) -> Result<BoundedTensor> {
    const U: f64 = 1.0 / (1u64 << 24) as f64; // f32 unit roundoff 2^-24
    let k = (macs.saturating_add(2)) as f64;
    let gamma = if k * U < 1.0 {
        (k * U) / (1.0 - k * U)
    } else {
        f64::INFINITY
    };
    let gamma_macs = {
        let m = macs as f64;
        if m * U < 1.0 {
            (m * U) / (1.0 - m * U)
        } else {
            f64::INFINITY
        }
    };
    let s_inflate = if gamma_macs < 1.0 {
        1.0 / (1.0 - gamma_macs)
    } else {
        f64::INFINITY
    };

    let mut lower = y.lower().to_owned();
    let mut upper = y.upper().to_owned();
    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(s)
        .for_each(|lo_o, up_o, &s_o| {
            let mag = (lo_o.abs()).max(up_o.abs()) as f64;
            let s_safe = s_o as f64 * s_inflate;
            let err = next_up_f32((gamma * s_safe + 2.0 * U * mag) as f32);
            if err.is_finite() {
                *lo_o = next_down_f32(*lo_o - err);
                *up_o = next_up_f32(*up_o + err);
            } else {
                *lo_o = f32::NEG_INFINITY;
                *up_o = f32::INFINITY;
            }
        });
    BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
}

/// Guard: reject NaN weights at CROWN backward entry. (#2747)
pub(crate) fn guard_nan_weights(
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    layer_name: &str,
) -> Result<()> {
    if kernel.iter().any(|v| v.is_nan()) {
        warn!("{layer_name} CROWN backward: kernel contains NaN");
        return Err(NyError::NumericalInstability(format!(
            "{layer_name} CROWN backward: kernel contains NaN — corrupted model weights"
        )));
    }
    if let Some(bias) = bias {
        if bias.iter().any(|v| v.is_nan()) {
            warn!("{layer_name} CROWN backward: bias contains NaN");
            return Err(NyError::NumericalInstability(format!(
                "{layer_name} CROWN backward: bias contains NaN — corrupted model weights"
            )));
        }
    }
    Ok(())
}

/// Detect rows with unsafe coefficients and fall back to +/-inf bias. (#2812, #2681)
pub(crate) fn detect_and_fix_nonfinite_rows(
    lower_a: &mut Array2<f32>,
    upper_a: &mut Array2<f32>,
    lower_b: &mut Array1<f32>,
    upper_b: &mut Array1<f32>,
    conv_in_size: usize,
    layer_name: &str,
) -> (usize, usize) {
    debug_assert_eq!(lower_a.ncols(), conv_in_size);
    debug_assert_eq!(upper_a.ncols(), conv_in_size);
    debug_assert_eq!(lower_a.nrows(), upper_a.nrows());
    debug_assert_eq!(lower_a.nrows(), lower_b.len());
    debug_assert_eq!(upper_a.nrows(), upper_b.len());

    let num_outputs = lower_a.nrows();
    let mut lower_affected = 0usize;
    let mut upper_affected = 0usize;

    for row_idx in 0..num_outputs {
        let lower_has_nonfinite = lower_a
            .row(row_idx)
            .iter()
            .any(|v| !is_crown_coeff_safe(*v));
        if lower_has_nonfinite {
            lower_a.row_mut(row_idx).fill(0.0);
            lower_b[row_idx] = f32::NEG_INFINITY;
            lower_affected += 1;
        }

        let upper_has_nonfinite = upper_a
            .row(row_idx)
            .iter()
            .any(|v| !is_crown_coeff_safe(*v));
        if upper_has_nonfinite {
            upper_a.row_mut(row_idx).fill(0.0);
            upper_b[row_idx] = f32::INFINITY;
            upper_affected += 1;
        }
    }

    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "{layer_name} CROWN backward: non-finite A-matrix overflow in {lower_affected}/{num_outputs} lower rows, \
             {upper_affected}/{num_outputs} upper rows — falling back to ±inf bias for affected rows"
        );
    }

    (lower_affected, upper_affected)
}

/// Compute broadcast convolution bias contribution in f64 with directed rounding.
/// (#2812, #1863)
pub(crate) fn compute_conv_bias_f64(
    bounds: &LinearBounds,
    bias: Option<&Array1<f32>>,
    out_c: usize,
    spatial_size: usize,
) -> (Array1<f32>, Array1<f32>) {
    debug_assert_eq!(bounds.num_inputs(), out_c * spatial_size);

    match bias {
        Some(bias) => {
            let num_outputs = bounds.num_outputs();
            let mut lower_bias_contrib = Array1::<f64>::zeros(num_outputs);
            let mut upper_bias_contrib = Array1::<f64>::zeros(num_outputs);

            for row_idx in 0..num_outputs {
                let mut lower_sum = 0.0f64;
                let mut upper_sum = 0.0f64;

                for c in 0..out_c {
                    let spatial_start = c * spatial_size;
                    let spatial_end = spatial_start + spatial_size;

                    let lower_spatial_sum: f64 = bounds
                        .lower_a()
                        .row(row_idx)
                        .slice(s![spatial_start..spatial_end])
                        .iter()
                        .map(|&v| v as f64)
                        .sum();
                    let upper_spatial_sum: f64 = bounds
                        .upper_a()
                        .row(row_idx)
                        .slice(s![spatial_start..spatial_end])
                        .iter()
                        .map(|&v| v as f64)
                        .sum();

                    lower_sum += lower_spatial_sum * (bias[c] as f64);
                    upper_sum += upper_spatial_sum * (bias[c] as f64);
                }

                lower_bias_contrib[row_idx] = lower_sum;
                upper_bias_contrib[row_idx] = upper_sum;
            }

            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_bias_contrib;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_bias_contrib;
            (
                lb_f64.mapv(|x| next_down_f32(x as f32)),
                ub_f64.mapv(|x| next_up_f32(x as f32)),
            )
        }
        None => (bounds.lower_b().clone(), bounds.upper_b().clone()),
    }
}

/// Whether the conv CROWN-backward f64-recomputes the coefficient.
///
/// Currently `true` for ALL contraction widths — the conv backward ALWAYS
/// f64-accumulates the coefficient and certifies `cast_err + γ_n^f64·S` (tight,
/// matching Linear's `aw_f64_with_abssum`). A small-n fast path that keeps the f32
/// GEMM coefficient + the (sound but ~2^29× larger) `γ_n^f32·S` factor was tried
/// to skip the recompute on tiny convs, but its looser certified error pushed
/// CROWN's concretized bounds past the IBP bound on tightness/CROWN⊆IBP tests, so
/// it is disabled. The f64 recompute's `γ_n^f64·S` is sub-ULP, keeping bounds tight
/// AND sound (#vnncomp-aw-soundness). The hook is retained so a future tightness-
/// preserving fast path can be slotted in centrally.
#[inline]
pub(crate) fn conv_should_f64_recompute(_n_contraction: usize) -> bool {
    true
}

/// #wall-deadwork gate — DEFAULT ON since 2026-07-20 (`NY_CONV_SKIP_DEAD_F32=0`
/// is the kill-switch).
///
/// Under `conv_should_f64_recompute` (unconditionally true today) the f32
/// coefficient GEMM pair's A-values are discarded on BOTH downstream paths:
/// recompute success overwrites them with the directed-rounded f64 result, and
/// recompute failure degrades the row to ±inf bias. The pair contributes only
/// buffer allocation and the per-node deadline check, so the skip replaces it
/// with direct allocation plus an explicit deadline check, and runs the two f64
/// recomputes concurrently (each is internally deterministic; the certified
/// error channel is summation-order independent, so the join is bit-safe).
///
/// Flip evidence (ledger 2026-07-19/20): bitwise-identity oracle on the
/// recompute path; expired-deadline and mem-cap oracles; 145/145 conv suite;
/// 226 production wall runs with zero anomalies + 2 banked-unsat guards
/// FASTER (62→54s, 59→53s); ~25% measured on the root-CROWN-intersect phase.
/// Deadline timing is intentionally not byte-identical: an already-expired
/// deadline aborts even on small work the unchunked pair would have finished.
/// Also, a future deadline can expire inside the uninterruptible f64 join after
/// the legacy chunked f32 pair would have polled and aborted. The f64 recompute
/// was already uninterruptible on the legacy path; the skip merely starts it
/// sooner. Either case can affect fallback timing, never bound soundness.
#[inline]
pub(crate) fn conv_skip_dead_f32_enabled() -> bool {
    std::env::var("NY_CONV_SKIP_DEAD_F32").ok().as_deref() != Some("0")
}

/// Build the certified per-coefficient error matrix for a conv backward
/// (#vnncomp-aw-soundness — conv f32-accumulation bug). Shared by the scalar and
/// batched paths.
///
/// Error = `cast_err + γ·S + prop`, with the row-constant over-bound
/// `S[i,p] ≤ row_max(a,i)·‖kernel‖_1` (sub-ULP once multiplied by γ, so the
/// over-bound is harmless) and, for the incoming-error propagation term, either
///
///   - `prop_exact = Some(P)` (#cgan-conv-err-compose): `P` is the incoming error
///     matrix composed through the SAME backward column transform as the
///     coefficients, but with `|kernel|`: `P[i,p] = Σ_j err_in[i,j]·|K_{j→p}|`
///     (computed by the caller as one extra f32 conv/GEMM on non-negative data).
///     This is the EXACT first-order bound `|Σ_j (a±e)_j·K_{j→p} − Σ_j a_j·K_{j→p}|
///     ≤ Σ_j e_j·|K_{j→p}|`; the f32 evaluation of `P` itself is inflated by
///     `(1+γ_{n}^f32)` — sound for any summation order because every summand is
///     non-negative (Higham §4.2: relative error of a non-negative sum ≤ γ_n).
///     Non-finite entries (INF-poisoned incoming rows) stay `+INF` (outward).
///   - `prop_exact = None`: the legacy row-constant over-bound
///     `prop[i] = row_max(err_in,i)·‖kernel‖_1`, applied to every column. Sound
///     but catastrophically loose on real conv stacks: `‖kernel‖_1` sums over the
///     WHOLE kernel (all output channels × all taps) while a single input column
///     only ever receives `fan-out ≪ ‖kernel‖_1` of it, so the certified error
///     grows by ~`‖kernel‖_1 / mean-column-L1` (100–1000×) per conv layer and,
///     after the discharge at the next non-carrier layer, dominated the CROWN
///     width on cGAN-class conv/BN stacks (BN_5 2.05×, Conv_19 404× vs exact).
///
/// Two SOUND γ modes, selected by the caller:
///   - `coeff_f64 = Some(f64-recompute)` → `γ = γ_n^f64` and `cast_err =
///     |f64 − stored_f32|` (the stored coefficient is the directed f32 of the f64
///     recompute). Tight; used on wide contractions.
///   - `coeff_f64 = None` → `γ = γ_n^f32` and `cast_err = 0` (the stored
///     coefficient is the f32 GEMM result, whose error `γ_n^f32·S` is itself the
///     bound). Cheap; used on small contractions where this is already tight.
///
/// `in_a` is the incoming coefficient block and `in_err` the incoming certified
/// error, both flattened to `(rows, mid_dim)`.
pub(crate) fn conv_coeff_err_matrix(
    in_a: &Array2<f32>,
    in_err: Option<&Array2<f32>>,
    stored: &Array2<f32>,
    coeff_f64: Option<&Array2<f64>>,
    kernel_l1: f64,
    n_contraction: usize,
    prop_exact: Option<&Array2<f32>>,
) -> Array2<f32> {
    let nrows = stored.nrows();
    let ncols = stored.ncols();
    let recompute_ok = coeff_f64.is_some_and(|c| c.dim() == (nrows, ncols));
    // f64 sum error when the coefficient is f64-accumulated; otherwise the f32
    // GEMM coefficient's error is the (larger) f32 growth factor. Both sound.
    let gamma = if recompute_ok {
        crate::layers::linear::crown_single_gamma_n_f64(n_contraction)
    } else {
        crate::layers::linear::crown_single_gamma_n_f32(n_contraction)
    };
    let row_max = |a: &Array2<f32>, i: usize| -> f64 {
        let mut m = 0.0f64;
        for k in 0..a.ncols() {
            let v = (a[[i, k]] as f64).abs();
            if v > m {
                m = v;
            }
        }
        m
    };
    // Exact prop path: inflate the f32-evaluated non-negative composition by
    // (1 + γ_{n+2}^f32) to cover its own accumulation rounding (n products, ≤ n−1
    // adds, +2 headroom for the per-product rounding), and only when the shape
    // matches the output block (defensive: mismatch falls back to the row bound).
    let prop_exact = prop_exact.filter(|p| p.dim() == (nrows, ncols));
    let prop_inflate =
        1.0 + crate::layers::linear::crown_single_gamma_n_f32(n_contraction.saturating_add(2));
    let mut err = Array2::<f32>::zeros((nrows, ncols));
    for i in 0..nrows {
        let s = gamma * row_max(in_a, i) * kernel_l1;
        let prop_row = if prop_exact.is_some() {
            0.0
        } else {
            in_err.map_or(0.0, |e| row_max(e, i) * kernel_l1)
        };
        for p in 0..ncols {
            let cast = coeff_f64
                .filter(|_| recompute_ok)
                .map_or(0.0, |c| (c[[i, p]] - stored[[i, p]] as f64).abs());
            let prop = match prop_exact {
                Some(pe) => {
                    let v = pe[[i, p]] as f64;
                    if v.is_finite() {
                        // Sanitize negative garbage outward (should not happen:
                        // non-negative inputs), then inflate.
                        v.max(0.0) * prop_inflate
                    } else {
                        f64::INFINITY
                    }
                }
                None => prop_row,
            };
            err[[i, p]] = next_up_f32((cast + s + prop) as f32);
        }
    }
    err
}

/// Batched-conv alias kept for the batched call sites (identical semantics).
pub(crate) use conv_coeff_err_matrix as batched_conv_coeff_err;
