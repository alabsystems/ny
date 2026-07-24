// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Native model loading without ONNX export.
//!
//! This module provides architecture detection and network construction
//! directly from PyTorch/SafeTensors weight files.
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::native::NativeModel;
//!
//! // Auto-detect architecture
//! let model = NativeModel::load("model.pt").unwrap();
//!
//! // Or specify architecture from config
//! // let model = NativeModel::load_with_config("model.pt", config).unwrap();
//! ```

mod builders;
mod config;
mod detect;
mod helpers;
mod hf_config;
#[cfg(test)]
mod hf_config_special_arch_tests;
mod model;
#[cfg(feature = "internal-test-utils")]
pub mod test_support;
mod weights;

/// Native architecture and loader configuration types.
pub use config::{Architecture, ModelConfig};
/// Hugging Face configuration schema used for architecture detection.
pub use hf_config::HfConfig;
/// Native model wrapper loaded directly from weight files.
pub use model::NativeModel;
/// Weight loading helper for native (non-ONNX-exported) model checkpoints.
pub use weights::load_weights;

#[cfg(test)]
mod tests;
