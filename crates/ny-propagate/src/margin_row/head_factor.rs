// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #head-factor stage 1 — a FACTORED form of the margin-row head seed.
//!
//! # Why this exists
//!
//! The margin-row lane proves cifar100 and tinyimagenet, the two largest
//! categories on the 2025 board, and loses both. The measured root gap against
//! alpha-beta-CROWN decomposes as ~2-4x DEVICE and ~20-40x REPRESENTATION: abc
//! carries structured (patches / sparse) coefficients where this lane carries
//! dense ones. `CrownBounds::Patches` appears in 59 files of `ny-propagate` and
//! in ZERO production files of `margin_row/`.
//!
//! A previous attempt to bring patches into this lane was refuted for a precise
//! reason: the HEAD SEED at `engine.rs:510-560` computes
//!
//! ```text
//!     L0[k, :] = sum_j W1[j, k] * S[j, :]          i.e.  L0 = W1^T S
//!     E0[k, :] = certify_up( sum_j |W1[j, k]| * combined_err(S, Es, g) )
//! ```
//!
//! and MATERIALIZES `L0` as a dense `(n_h, r)` block. Every trunk op below the
//! head therefore sees a coefficient block whose support is the whole hidden
//! layer, so a structured carrier compresses by 1x BY CONSTRUCTION. The
//! densification happens at the seed, before any convolution is reached.
//!
//! # What stage 1 does, and does not do
//!
//! This module introduces the factored representation `W1^T (x) S` and proves,
//! by test, that materializing it reproduces the existing dense seed EXACTLY —
//! bit for bit on the value block, and with an error block that is certified
//! no-tighter than the incumbent. NOTHING in production calls it yet, so this
//! file cannot change a verdict; `head_factor_armed()` is `false` and no yaml
//! reaches it.
//!
//! That equivalence is the precondition for every downstream design: only once
//! the factored form is known to be exactly the incumbent can the product be
//! pushed THROUGH a trunk op (applying the op's structured backward to `W1`'s
//! columns) instead of being materialized at the head.
//!
//! # Soundness
//!
//! The value block is an exact restatement: `materialize_value` performs the
//! same `axpy` accumulation in the same order as `engine.rs`, so it is bitwise
//! identical, not merely equal in exact arithmetic. The error block reuses the
//! lane's own `combined_err` / `certify_up` / `gamma_n` primitives with the same
//! `n` arguments, so the published outward envelope is unchanged. A factored
//! carrier that dropped the error terms would not be a proof; this one carries
//! them explicitly and the tests assert it.

use ndarray::Array2;

use super::net::axpy;
use super::rounding::{certify_up, gamma_n};

/// The head seed in factored form: the pair `(W1, S)` kept apart.
///
/// `w1` is the head Gemm weight in the lane's row-major `[n_y * n_h]` layout,
/// exactly as `TwinNet::gemm1` hands it over. `s` is the spec/margin block
/// `(n_y, r)`. The spec block's certified error `Es` enters only through the
/// caller's `combined_err(S, Es, gamma_n(n_y + 2))` block handed to
/// [`HeadFactor::materialize_error`], so the factor does not carry it.
#[derive(Clone, Debug)]
pub(crate) struct HeadFactor {
    /// Head Gemm weights, row-major `n_y` x `n_h`.
    pub(crate) w1: Vec<f64>,
    /// Spec block, `(n_y, r)`.
    pub(crate) s: Array2<f64>,
    /// Hidden width `n_h` (the head Gemm's input dimension).
    pub(crate) n_h: usize,
}

impl HeadFactor {
    /// Rows of the factored product, i.e. the materialized block's height.
    pub(crate) fn nrows(&self) -> usize {
        self.n_h
    }

    /// Columns of the factored product.
    pub(crate) fn ncols(&self) -> usize {
        self.s.ncols()
    }

    /// Number of spec rows contracted over.
    pub(crate) fn n_y(&self) -> usize {
        self.s.nrows()
    }

    /// COLUMN SUPPORT of the head weight: the hidden indices `k` for which some
    /// `W1[j, k]` is nonzero.
    ///
    /// This is the number the whole representation question turns on. If it is
    /// ~`n_h` the head seed is irreducibly dense and a structured carrier
    /// cannot help at this seam; if it is materially smaller, the materialized
    /// block is mostly zeros and the dense `(n_h, r)` allocation is waste.
    /// Nothing in the lane reports it today.
    pub(crate) fn live_hidden_support(&self) -> usize {
        let n_y = self.n_y();
        (0..self.n_h)
            .filter(|&k| (0..n_y).any(|j| self.w1[j * self.n_h + k] != 0.0))
            .count()
    }

    /// Materialize the value block, reproducing `engine.rs`'s seed loop in the
    /// SAME accumulation order so the result is bitwise identical.
    pub(crate) fn materialize_value(&self) -> Array2<f64> {
        let r = self.ncols();
        let n_y = self.n_y();
        let mut l0 = Array2::<f64>::zeros((self.n_h, r));
        let ss = self.s.as_slice().expect("standard layout");
        let dst = l0.as_slice_mut().expect("standard layout");
        for (k, acc) in dst.chunks_mut(r).enumerate() {
            for j in 0..n_y {
                let w = self.w1[j * self.n_h + k];
                if w != 0.0 {
                    axpy(acc, w, &ss[j * r..(j + 1) * r]);
                }
            }
        }
        l0
    }

    /// Materialize the certified outward error block, reproducing the incumbent
    /// `E0 = certify_up( |W1|^T * combined_err(S, Es, gamma_n(n_y + 2)), gamma_n(n_y + 8) )`.
    pub(crate) fn materialize_error(&self, combined: &Array2<f64>) -> Array2<f64> {
        let r = self.ncols();
        let n_y = self.n_y();
        let mut e0 = Array2::<f64>::zeros((self.n_h, r));
        let cs = combined.as_slice().expect("standard layout");
        let g2 = gamma_n(n_y + 8);
        let dst = e0.as_slice_mut().expect("standard layout");
        for (k, acc) in dst.chunks_mut(r).enumerate() {
            for j in 0..n_y {
                let w = self.w1[j * self.n_h + k].abs();
                if w != 0.0 {
                    axpy(acc, w, &cs[j * r..(j + 1) * r]);
                }
            }
            for v in acc.iter_mut() {
                *v = certify_up(*v, g2);
            }
        }
        e0
    }
}

/// Stage-1 gate. Dark by construction: no yaml, no env, no caller.
///
/// #cct-margin-row-admission: since the incumbent seed also charges a
/// BN-folded head's certified fold error (`TwinNet::gemm1_fold_err`: the
/// `rel` term in its combined-error gamma and the `bias_err` bias term), this
/// factored form reproduces it only for UNFOLDED heads. Arming it requires
/// carrying the same two charges first.
///
/// Kept as a named function rather than a bare `false` so the eventual arming
/// has exactly one site to change, and so a reader grepping for the mechanism
/// finds the gate rather than concluding it was never built.
pub(crate) const fn head_factor_armed() -> bool {
    false
}

#[cfg(test)]
mod tests;
