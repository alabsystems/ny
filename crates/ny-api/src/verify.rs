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
