// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound width profiling for neural networks.
//!
//! This module provides detailed analysis of how bound widths propagate
//! through a neural network, helping identify where verification becomes
//! difficult.
//!
//! ## Key Metrics
//!
//! For each layer, we track:
//! - **Input/Output width**: Max bound width at layer boundaries
//! - **Width growth**: Ratio of output to input width (expansion factor)
//! - **Cumulative width**: Total bound expansion from input to this layer
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ny_onnx::profile::{analyze_profile, ProfileConfig};
//!
//! let config = ProfileConfig::default();
//! let result = analyze_profile("model.onnx", &config)?;
//! println!("{}", result.summary());
//! # Ok::<(), ny_onnx::profile::ProfileError>(())
//! ```
//!
//! The legacy `profile_bounds*` family remains available for compatibility.

mod graph;
mod model;
mod stats;
mod types;

use ny_core::truncate_name;
// Re-export stats functions for test access via `use super::*` in #[cfg(test)]
#[cfg(test)]
use stats::{difficulty_score, make_unit_variance_input, median};

pub use graph::{analyze_profile_graph, profile_bounds_graph};
pub use model::{analyze_profile, analyze_profile_model, profile_bounds, profile_bounds_model};
pub use types::{BoundStatus, LayerProfile, ProfileConfig, ProfileError, ProfileResult};

#[cfg(test)]
mod tests;
