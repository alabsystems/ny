// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verifier entry points and configuration.
//!
//! Re-exports the curated verification entry points plus the engine types used
//! by engine-aware verification helpers.
//!
//! ```rust
//! use std::sync::Arc;
//! use ny_api::verify::{NaiveCpuGemmEngine, PropagationConfig, PropagationMethod, Verifier};
//!
//! let verifier = Verifier::new_with_engine(
//!     PropagationConfig {
//!         method: PropagationMethod::Crown,
//!         ..Default::default()
//!     },
//!     Arc::new(NaiveCpuGemmEngine),
//! );
//! # let _ = verifier;
//! ```

pub use ny_core::{GemmEngine, NaiveCpuGemmEngine};
pub use ny_propagate::{PropagationConfig, PropagationMethod, Verifier};

/// Relaxation mode for `MulBinary` (elementwise product) bound propagation —
/// the [`PropagationConfig::mul_binary_relaxation`] field type. Both modes are
/// sound linear enclosures of the bilinear term; they differ only in tightness
/// trade-offs (McCormick facets vs. auto_LiRPA-style middle interpolation).
pub use ny_propagate::MulBinaryRelaxationMode;

/// Double-precision (f64) sequential evaluation/propagation helpers for bound
/// tightness measurement and differential testing (Linear/Conv2d/ReLU chains).
///
/// [`convert_network_to_f64`] lifts a sequential network's layers to
/// [`SequentialLayerF64`]; [`evaluate_network_f64`] evaluates them exactly at a
/// concrete point; [`propagate_network_f64`] propagates a
/// [`BoundedTensor64`](crate::BoundedTensor64) box through them under an
/// [`F64PropagationMode`] (IBP or CROWN). These are measurement/reference
/// helpers: the sound f32 verdict path does not depend on them.
pub use ny_propagate::{
    convert_network_to_f64, evaluate_network_f64, propagate_network_f64, F64PropagationMode,
    SequentialLayerF64,
};
