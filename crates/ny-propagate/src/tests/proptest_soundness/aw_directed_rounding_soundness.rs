// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for three directed-rounding /
//! f64-accumulation fixes (#vnncomp-aw-soundness):
//!
//! (A) MatMul IBP standard path (`binary_ops/matmul/ibp_standard.rs`): the f32
//!     interval dot product was accumulated in f32 with NO directed rounding, so
//!     a wide contraction could store an [lower, upper] interval that does NOT
//!     enclose the true real product → false proof. The fix accumulates in f64
//!     (exact f32→f64-widened products) and directed-rounds OUTWARD at the store.
//!
//! (B) Scatter/IndexAdd const-shift bias fold (`transform/accumulate.rs`): the
//!     `b += A @ const_shift` dot was f32 with no directed rounding. The fix
//!     accumulates in f64 and folds OUTWARD (`next_down_f32` lower, `next_up_f32`
//!     upper).
//!
//! (C) Tile replica-sum coefficient (`transform/tile/mod.rs`): the f32-accumulated
//!     replica sum was certified with the f64 growth factor `γ_reps^f64`
//!     (~2^29× too small for an f32 accumulation). The fix certifies with the f32
//!     factor `γ_reps^f32`.
//!
//! Each test FAILS on the pre-fix code and PASSES after the fix.

use crate::layers::common::BoundPropagation;
use crate::layers::transform::ScatterAddLayer;
use crate::{LinearBounds, MatMulLayer};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

// ---------------------------------------------------------------------------
// (A) MatMul IBP standard path: f64 accumulate + directed-round OUTWARD.
// ---------------------------------------------------------------------------

/// Deterministic xorshift fill in a fine cancelling grid (forces f32 rounding).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    };
    (0..n).map(|_| next()).collect()
}

/// REPRODUCE + VERIFY (A): the MatMul IBP standard path must enclose the true
/// f64-exact dot product for EVERY concrete corner of a WIDE matmul, with ZERO
/// tolerance. On the pre-fix f32-accumulating path the stored interval can be
/// strictly INSIDE the true value (false proof) because the running f32 sum
/// rounds inward; the f64-accumulate + directed-round-outward fix makes the
/// stored f32 interval provably enclose the true real value.
#[test]
fn matmul_ibp_standard_encloses_true_value_zero_tol() {
    // Wide contraction k so the f32 accumulation genuinely rounds and cancels.
    let m = 3usize;
    let k = 256usize;
    let n = 3usize;

    // Point intervals (lower == upper) so the IBP output is exactly the matmul
    // of the two concrete matrices — the tightest case, where any inward
    // rounding of the stored bound is a soundness VIOLATION (no slack).
    let a_vals = fill(m * k, 0x9E3779B97F4A7C15);
    let b_vals = fill(k * n, 0xD1B54A32D192ED03);

    let a_lo = ArrayD::from_shape_vec(IxDyn(&[m, k]), a_vals.clone()).unwrap();
    let b_lo = ArrayD::from_shape_vec(IxDyn(&[k, n]), b_vals.clone()).unwrap();
    let a = BoundedTensor::new(a_lo.clone(), a_lo).unwrap();
    let b = BoundedTensor::new(b_lo.clone(), b_lo).unwrap();

    let layer = MatMulLayer::new(false, None);
    let out = layer.propagate_ibp_binary(&a, &b).unwrap();

    // True real value of each output element: the f64-exact dot (exact f32→f64
    // products, f64 sum ≈ the real value up to a tiny γ_n^f64 residual which is
    // FAR below an f32 ULP of the result and is what next_down/next_up cover).
    let mut violations = 0usize;
    let mut worst_lower_gap = 0.0f64;
    let mut worst_upper_gap = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut true_val = 0.0f64;
            for l in 0..k {
                true_val += (a_vals[i * k + l] as f64) * (b_vals[l * n + j] as f64);
            }
            let lo = out.lower()[[i, j]] as f64;
            let hi = out.upper()[[i, j]] as f64;
            if lo > true_val {
                violations += 1;
                worst_lower_gap = worst_lower_gap.max(lo - true_val);
            }
            if hi < true_val {
                violations += 1;
                worst_upper_gap = worst_upper_gap.max(true_val - hi);
            }
        }
    }

    assert_eq!(
        violations, 0,
        "MatMul IBP standard bound does NOT enclose the true f64 value \
         (worst lower over-shoot={worst_lower_gap:e}, worst upper under-shoot={worst_upper_gap:e}) \
         — f32 accumulation rounded the certified interval INWARD (false proof)."
    );
}

/// REPRODUCE + VERIFY (A) with the negative-scale branch (swaps endpoints): the
/// directed rounding must still be OUTWARD after scaling.
#[test]
fn matmul_ibp_standard_neg_scale_encloses_true_value_zero_tol() {
    let m = 2usize;
    let k = 200usize;
    let n = 2usize;
    let scale = -1.5f32;
    let a_vals = fill(m * k, 0x243F6A8885A308D3);
    let b_vals = fill(k * n, 0x13198A2E03707344);

    let a_lo = ArrayD::from_shape_vec(IxDyn(&[m, k]), a_vals.clone()).unwrap();
    let b_lo = ArrayD::from_shape_vec(IxDyn(&[k, n]), b_vals.clone()).unwrap();
    let a = BoundedTensor::new(a_lo.clone(), a_lo).unwrap();
    let b = BoundedTensor::new(b_lo.clone(), b_lo).unwrap();

    let layer = MatMulLayer::new(false, Some(scale));
    let out = layer.propagate_ibp_binary(&a, &b).unwrap();

    let mut violations = 0usize;
    for i in 0..m {
        for j in 0..n {
            let mut dot = 0.0f64;
            for l in 0..k {
                dot += (a_vals[i * k + l] as f64) * (b_vals[l * n + j] as f64);
            }
            let true_val = dot * (scale as f64);
            let lo = out.lower()[[i, j]] as f64;
            let hi = out.upper()[[i, j]] as f64;
            if lo > true_val || hi < true_val {
                violations += 1;
            }
        }
    }
    assert_eq!(
        violations, 0,
        "MatMul IBP standard (negative scale) bound does not enclose the true value."
    );
}

// ---------------------------------------------------------------------------
// (B) Scatter const-shift bias fold: f64 accumulate + fold OUTWARD.
// ---------------------------------------------------------------------------

/// REPRODUCE + VERIFY (B): the ScatterAdd (src-variable) CROWN backward folds the
/// constant `data` operand into the bias as `b += A @ data_const`. This is a
/// CERTIFIED bound, so the resulting `[lower_b, upper_b]` must enclose the true
/// real value `incoming_b + A @ data_const` for every row, with ZERO tolerance.
/// On the pre-fix f32-accumulating fold a WIDE const-shift can round the bias
/// INWARD (false proof); the f64-accumulate + fold-OUTWARD fix encloses it.
#[test]
fn scatter_const_shift_bias_fold_encloses_true_value_zero_tol() {
    // Wide output_size so the `A @ data_const` bias dot genuinely rounds.
    let output_size = 256usize;
    let num_obj = 4usize;

    // Constant data of length output_size (the const-shift folded into the bias).
    let data_vals = fill(output_size, 0x2545F4914F6CDD1D);
    let data_const = ArrayD::from_shape_vec(IxDyn(&[output_size]), data_vals.clone()).unwrap();

    // ScatterAdd over a length-output_size axis, src is the (single) variable
    // operand; indices scatter src into the output. The src/index sizes don't
    // matter for the BIAS fold (which uses the full data_const), so use a small
    // src and identity-ish indices.
    let src_len = 4usize;
    let indices: Vec<i64> = (0..src_len as i64).collect();
    let layer = ScatterAddLayer::new(
        -1,
        Some(data_const),
        Some(ArrayD::from_shape_vec(IxDyn(&[src_len]), indices).unwrap()),
        None,
    );

    // Dense, cancelling incoming A (NOT identity) so each bias-fold row is a real
    // width-output_size dot product, plus a nonzero incoming bias.
    let a_vals = fill(num_obj * output_size, 0x8A5CD789635D2DFF);
    let lower_a = Array2::from_shape_vec((num_obj, output_size), a_vals.clone()).unwrap();
    let upper_a = lower_a.clone();
    let in_b_vals = fill(num_obj, 0x71526459A1BCDEF0);
    let lower_b = Array1::from_vec(in_b_vals.clone());
    let upper_b = lower_b.clone();
    let incoming = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

    let crown = layer.crown_backward(&incoming).unwrap();
    let out_lo = crown.lower_b();
    let out_hi = crown.upper_b();

    let mut violations = 0usize;
    let mut worst_lower = 0.0f64;
    let mut worst_upper = 0.0f64;
    for row in 0..num_obj {
        let mut true_val = in_b_vals[row] as f64;
        for col in 0..output_size {
            true_val += (a_vals[row * output_size + col] as f64) * (data_vals[col] as f64);
        }
        let lo = out_lo[row] as f64;
        let hi = out_hi[row] as f64;
        if lo > true_val {
            violations += 1;
            worst_lower = worst_lower.max(lo - true_val);
        }
        if hi < true_val {
            violations += 1;
            worst_upper = worst_upper.max(true_val - hi);
        }
    }

    assert_eq!(
        violations, 0,
        "ScatterAdd const-shift bias fold does NOT enclose the true f64 value \
         (worst lower over-shoot={worst_lower:e}, worst upper under-shoot={worst_upper:e}) \
         — f32 bias accumulation rounded the certified bias INWARD (false proof)."
    );
}

// ---------------------------------------------------------------------------
// (C) Tile replica-sum coefficient certified with γ_reps^f32 (not f64).
// ---------------------------------------------------------------------------

/// REPRODUCE + VERIFY (C): the Tile backward coefficient error must cover the
/// true |stored_f32_coeff − f64_recompute| for every coefficient. With the OLD
/// f64 growth factor the certified error UNDER-counts the real f32 replica-sum
/// rounding error (~2^29× too small); the f32-factor fix covers it.
#[test]
fn tile_backward_cert_covers_f32_replica_sum_error_zero_tol() {
    // Tile axis 0 of a length-`n_axis` vector by `reps`, so each input column is
    // the f32 sum of `reps` output columns. Wide reps + cancelling coefficients
    // force the f32 replica sum to round.
    let n_axis = 6usize;
    let reps = 200usize;
    let out_len = n_axis * reps;
    let num_obj = 4usize;

    let mut layer = crate::layers::transform::TileLayer::new(0, reps);
    layer.set_input_shape(vec![n_axis]);

    let a_vals = fill(num_obj * out_len, 0xA5A5A5A5DEADBEEF);
    let a = Array2::from_shape_vec((num_obj, out_len), a_vals).unwrap();
    let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();

    let result = layer.propagate_linear(&spec).unwrap();
    let stored = result.lower_a();
    let cert = result
        .lower_a_err()
        .expect("Tile backward must attach a certified coefficient error");

    // f64-exact replica sum (ground truth real coefficient).
    // For axis-0 tiling of a flat vector with suffix_size = 1: block_size = n_axis,
    // out_block_size = n_axis*reps, so input column i (prefix=0) maps to output
    // columns rep*n_axis + i for rep in 0..reps.
    let mut violations = 0usize;
    let mut worst_ratio = 0.0f64;
    for i in 0..n_axis {
        for row in 0..num_obj {
            let f32_coeff = stored[[row, i]] as f64;
            let mut f64_coeff = 0.0f64;
            for rep in 0..reps {
                f64_coeff += a[[row, rep * n_axis + i]] as f64;
            }
            let true_gap = (f32_coeff - f64_coeff).abs();
            let certified = cert[[row, i]] as f64;
            if certified < true_gap {
                violations += 1;
                if true_gap > 0.0 {
                    worst_ratio = worst_ratio.max(true_gap / certified.max(f64::MIN_POSITIVE));
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Tile backward certified error UNDER-counts the true f32 replica-sum error \
         (worst true_gap/certified ratio={worst_ratio:e}) — f32 accumulation certified \
         with the f64 growth factor (false proof)."
    );
}
