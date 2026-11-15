// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cct-margin-row-admission — the CERTIFIED `Gemm -> BatchNormalization`
//! fold for the margin-row head.
//!
//! # Why
//!
//! `challenging_certified_training_2026`'s CNN7 nets end
//! `Flatten -> Gemm(32768 -> 512) -> BatchNormalization(512) -> Relu -> Gemm`.
//! The lane's compiler admits exactly `Gemm -> Relu -> Gemm` at the tail, so
//! the head BN has to disappear into the first head Gemm. alpha-beta-CROWN's
//! BN merges are NOT a model for this: its `merge_bn` flag
//! (`complete_verifier/onnx_opt.py:420`) fuses only `Conv -> BatchNormalization`
//! (the branch is gated on `node.op_type == "Conv"`), its only Gemm/BN fusions
//! (`merge_gemm_reshape_bn`, `merge_bn_reshape_gemm`) are cgan-shaped, and all
//! of them are plain nearest-rounded numpy arithmetic with NO error term — a
//! merge that is not a sound over-approximation of the network it replaced.
//! (Checked in review 2026-09-24 against `~/field-src/alpha-beta-CROWN`.)
//! This fold is computed in f64
//! and CARRIES its error: every folded weight within [`RHO_FOLD`] of the exact
//! real-arithmetic product (relative to the stored value), every folded bias
//! within its own absolute `bias_err`. The compiler hands both to every head
//! consumer (`engine.rs` seed, both GPU head statics), exactly like the
//! existing `Conv + BatchNormalization` fold.
//!
//! # The exact map
//!
//! ONNX `BatchNormalization` (inference form) on the Gemm output `u = W z + b`:
//!
//! ```text
//!     y_o = s_o * (u_o - mean_o) + beta_o,   s_o = gamma_o / sqrt(var_o + eps)
//!         = (s_o W_o) . z + [ beta_o + s_o (b_o - mean_o) ]
//! ```
//!
//! with `s_o` the REAL value (irrational in general).
//!
//! # The error bound (u = 2^-53, every intermediate NORMAL — checked below)
//!
//! `t = fl(var + eps)`, `q = fl(sqrt(t))`, `a = fl(gamma / q)`: three
//! nearest-rounded ops give `a = s (1 + theta)` with
//! `|theta| <= (1+u)(1-u)^{-3/2} - 1 <= 2.51u`. A folded weight is
//! `w' = fl(a w) = s w (1 + theta)(1 + d)`, so `|w' - s w| <= 3.52u |s w|
//! <= 3.53u |w'|` — [`RHO_FOLD`] = 1e-15 = 9.0u covers it with 2.5x headroom.
//!
//! The bias is `b' = fl( fl(beta - fl(a mean)) + fl(a b) )`, the Conv+BN
//! fold's own operation order. Writing `m1 = fl(a mean)`, `c = fl(a b)`:
//! `|m1 - s mean| <= 4.02u |m1|`, `|c - s b| <= 4.02u |c|`, the two
//! additions contribute `u |fl(beta - m1)| + u |b'|` with
//! `|fl(beta - m1)| <= 1.0001 |b'| + |c|`; summed,
//! `|b' - exact| <= 4.02u |m1| + 5.03u |c| + 2.01u |b'|`, dominated by the
//! Conv+BN arm's `8u (|m1| + |c| + |b'|)`. The `1e-300` floor absorbs any
//! gradual-underflow residue in the bias chain (each such rounding is
//! `<= 2^-1075` absolute).
//!
//! The weight bound is RELATIVE, so it needs normal results: a nonzero
//! product that lands subnormal (or overflows) REFUSES the fold (`None`), and
//! the extractor then declines the whole net — fail-closed, the lane does not
//! run. Weights that are exactly zero, or rows with `gamma == 0` (`s = 0`),
//! fold exactly.
//!
//! `fold_dense_batchnorm_encloses_exact_gemm_bn_rational` (below) checks every
//! one of these claims in exact rational arithmetic, with a negative control
//! that proves the oracle can fail.

use super::rounding::{RHO_FOLD, UNIT};

/// Inference-form BatchNormalization parameters, one entry per feature, as
/// exact f64 values (the ONNX f32 payloads widen exactly).
#[derive(Debug, Clone, Copy)]
pub struct DenseBnParams<'a> {
    /// `scale` (gamma).
    pub gamma: &'a [f64],
    /// `B` (beta).
    pub beta: &'a [f64],
    /// `input_mean`.
    pub mean: &'a [f64],
    /// `input_var`.
    pub var: &'a [f64],
    /// `epsilon` (the f32 attribute widened exactly).
    pub eps: f64,
}

/// A folded head Gemm: `y = weight . z + bias`, row-major `(n_out, n_in)`,
/// with its certified error budget vs the exact `Gemm -> BatchNormalization`.
#[derive(Debug, Clone)]
pub struct DenseBnFold {
    /// Folded weights `fl(a_o * W_oi)`, row-major `(n_out, n_in)`.
    pub weight: Vec<f64>,
    /// Folded bias per row.
    pub bias: Vec<f64>,
    /// Certified ABSOLUTE per-row bias error vs the exact real fold.
    pub bias_err: Vec<f64>,
    /// Certified RELATIVE per-entry weight error (relative to the STORED
    /// folded weight): always [`RHO_FOLD`].
    pub weight_rel_err: f64,
}

/// Fold an inference `BatchNormalization` into the Gemm it consumes.
///
/// `weight` is row-major `(n_out, n_in)` (the extractor's post-`transB`
/// layout), `bias` has `n_out` entries. Returns `None` — never a partial
/// fold — on any shape mismatch, non-finite input or output, `var + eps` that
/// is not a positive normal, or a nonzero product that is not a normal f64
/// (the relative bound would not hold).
pub fn fold_dense_batchnorm(
    weight: &[f64],
    bias: &[f64],
    shape: (usize, usize),
    bn: &DenseBnParams<'_>,
) -> Option<DenseBnFold> {
    // The carried fold budgets assume nearest rounding and gradual
    // underflow. Qualify their producer before any arithmetic: a later
    // consumer check cannot repair a fold made under another thread's mode.
    ny_core::require_f64_interval_proof_environment().ok()?;
    let (n_out, n_in) = shape;
    if n_out == 0
        || n_in == 0
        || weight.len() != n_out.checked_mul(n_in)?
        || bias.len() != n_out
        || bn.gamma.len() != n_out
        || bn.beta.len() != n_out
        || bn.mean.len() != n_out
        || bn.var.len() != n_out
        || !bn.eps.is_finite()
    {
        return None;
    }
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    if !(finite(weight)
        && finite(bias)
        && finite(bn.gamma)
        && finite(bn.beta)
        && finite(bn.mean)
        && finite(bn.var))
    {
        return None;
    }
    // A nonzero exact product must land on a NORMAL f64 for the relative
    // bound to hold; a zero operand makes the product exactly zero.
    let product_ok = |x: f64, y: f64, p: f64| (x == 0.0 || y == 0.0) || p.is_normal();

    let mut out_w = vec![0.0f64; weight.len()];
    let mut out_b = vec![0.0f64; n_out];
    let mut out_e = vec![0.0f64; n_out];
    for o in 0..n_out {
        let t1 = bn.var[o] + bn.eps;
        // Positive AND normal (`is_normal` is false for NaN, zero, subnormal
        // and infinity): the relative error of `var + eps` needs a normal
        // result, and the square root needs a positive one.
        if t1 <= 0.0 || !t1.is_normal() {
            return None;
        }
        let sq = t1.sqrt();
        if !sq.is_normal() {
            return None;
        }
        // Parity with the Conv+BN fold: a = gamma / sqrt(var + eps). A zero
        // gamma folds exactly (s = 0); a nonzero one must stay normal.
        let a = bn.gamma[o] / sq;
        if bn.gamma[o] != 0.0 && !a.is_normal() {
            return None;
        }
        // b' = (beta - a * mean) + a * b, the Conv+BN arm's operation order.
        let m1 = a * bn.mean[o];
        let cb = a * bias[o];
        let b_c = (bn.beta[o] - m1) + cb;
        if !(m1.is_finite() && cb.is_finite() && b_c.is_finite()) {
            return None;
        }
        out_b[o] = b_c;
        // Certified absolute error (module docs): <= 4.02u|m1| + 5.03u|cb|
        // + 2.01u|b'|, dominated by the Conv+BN arm's 8u(...) + 1e-300.
        out_e[o] = 8.0 * UNIT * (m1.abs() + cb.abs() + b_c.abs()) + 1e-300;
        if !out_e[o].is_finite() {
            return None;
        }
        let row = &weight[o * n_in..(o + 1) * n_in];
        let dst = &mut out_w[o * n_in..(o + 1) * n_in];
        for (d, &w) in dst.iter_mut().zip(row) {
            let p = a * w;
            if !p.is_finite() || !product_ok(a, w, p) {
                return None;
            }
            *d = p;
        }
    }
    Some(DenseBnFold {
        weight: out_w,
        bias: out_b,
        bias_err: out_e,
        weight_rel_err: RHO_FOLD,
    })
}

#[cfg(test)]
mod tests;
