// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Static compute-cost analysis for ONNX models.
//!
//! This module estimates per-layer arithmetic cost and activation memory from
//! fully-known tensor shapes. It is intended for engineering-style reports on
//! fixed-shape exports such as Kokoro vocoder ONNX graphs, not as a replacement
//! for numerical bound propagation.

mod layer_metadata;
mod lookup;
mod model;
mod timing;
mod types;

pub use model::estimate_model_cost;
pub use timing::{
    estimate_model_timing, FamilyTimingCalibration, LayerTimingEstimate, TimingEstimate,
    TimingProfile,
};
pub use types::{CostError, CostResult, LayerCost};

#[cfg(test)]
#[cfg_attr(not(feature = "external-avoice"), allow(dead_code, unused_imports))]
mod tests;
#[cfg(test)]
mod tests_missing_shapes;
#[cfg(all(test, feature = "external-avoice"))]
mod tests_real_export_fallback;
