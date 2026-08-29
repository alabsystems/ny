// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! 3-D (volumetric) convolution for bound propagation.
//!
//! Video and volumetric models — Qwen3-VL / PaddleOCR-VL 3-D patch embeddings,
//! Smart Turn's video stem, medical volume CNNs — open with a rank-5 `Conv`.
//! Before this module NY had no layer to lower one to: `ny-build`'s ONNX
//! converter classified a rank-5 kernel `UnsupportedOp`, and the trace bridge
//! refused `TraceOp::Conv3d` naming the MISSING LAYER (not missing wiring) as
//! the obstruction.
//!
//! # What this layer does and does not do
//!
//! * **IBP: certified.** [`Conv3dLayer`]'s only interval forward is the
//!   directed-binary64 enclosure in [`ops_ibp_cert`] — see
//!   [`crate::layers::BoundPropagation::propagate_ibp`] on [`Conv3dLayer`] for
//!   why the certificate is the DEFAULT here rather than an opt-in arm beside a
//!   faster f32 path.
//! * **CROWN: fails closed.** No certified rank-3-spatial transposed
//!   convolution exists, so `propagate_linear` returns
//!   [`ny_core::NyError::UnsupportedOp`] and drivers fall back to IBP. Conv3d
//!   is absent from `Layer::propagates_coeff_err` and
//!   `Layer::supports_batched_crown` for the same reason.
//! * **Parameters: stride, padding, dilation, groups** — the same set Conv2d
//!   supports. Everything else is refused by name (see [`Conv3dLayer`]).

mod bound;
mod ops_ibp_cert;
mod types;

#[cfg(test)]
mod tests;

pub use types::Conv3dLayer;
