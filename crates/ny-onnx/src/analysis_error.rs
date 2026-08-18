// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared error type for ny-onnx analysis modules (quantize, sensitivity, profile).
//!
//! These three modules previously defined structurally identical error enums.
//! This module consolidates them into a single `AnalysisError` used by all three,
//! reducing duplication and ensuring consistent error handling.
//!
//! `DiffError` is intentionally excluded — it has additional variants specific
//! to the model comparison use case (ORT runtime, IO, NPY, shape mismatch).
//!
//! # Structured payloads (Issue #2295)
//!
//! `LoadError` and `PropagationError` wrap `Box<NyError>` instead of `String`
//! to preserve programmatic error discrimination across the analysis boundary.
//! The bridge conversion (`From<AnalysisError> for NyError`) can recover
//! the original variant instead of a lossy `NyError -> String -> NyError`
//! round-trip.

use ny_core::NyError;
use thiserror::Error;

/// Errors that can occur during analysis operations (quantize, sensitivity, profile).
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Model loading or parsing failed.
    ///
    /// Wraps the original `NyError` structurally so the bridge can recover
    /// the exact variant (e.g., `ModelLoad`, `InvalidSpec`).
    #[error("{context}: load failed: {source}")]
    LoadError {
        context: &'static str,
        #[source]
        source: Box<NyError>,
    },

    /// Bound propagation or network construction failed.
    ///
    /// Wraps the original `NyError` structurally so the bridge can recover
    /// the exact variant.
    #[error("{context}: propagation failed: {source}")]
    PropagationError {
        context: &'static str,
        #[source]
        source: Box<NyError>,
    },

    /// Network has no layers to analyze.
    #[error("{context}: no layers in network")]
    NoLayers { context: &'static str },

    /// Input specification has an invalid shape.
    #[error("{context}: invalid input shape: {detail}")]
    InvalidInputShape {
        context: &'static str,
        detail: String,
    },
}

impl AnalysisError {
    /// Create a load error wrapping a `NyError`.
    pub fn load(context: &'static str, source: NyError) -> Self {
        Self::LoadError {
            context,
            source: Box::new(source),
        }
    }

    /// Create a propagation error wrapping a `NyError`.
    pub fn propagation(context: &'static str, source: NyError) -> Self {
        Self::PropagationError {
            context,
            source: Box::new(source),
        }
    }

    /// Create a propagation error from a plain message (no source `NyError`).
    ///
    /// The message is wrapped as `NyError::InternalError` to preserve the
    /// bridge's mapping of `PropagationError` → `InternalError`.
    pub fn propagation_msg(context: &'static str, msg: impl Into<String>) -> Self {
        Self::PropagationError {
            context,
            source: Box::new(NyError::InternalError(msg.into())),
        }
    }

    /// Create a no-layers error.
    pub fn no_layers(context: &'static str) -> Self {
        Self::NoLayers { context }
    }

    /// Create an invalid-input-shape error.
    pub fn invalid_input_shape(context: &'static str, detail: impl Into<String>) -> Self {
        Self::InvalidInputShape {
            context,
            detail: detail.into(),
        }
    }
}

/// Validate the perturbation radius shared by profile, quantization, and
/// sensitivity analyses.
///
/// `BoundedTensor::from_epsilon` performs this check when an analysis creates
/// its default input. A caller-supplied bounded tensor bypasses that
/// constructor, so validate the configuration independently to keep result
/// normalization and reported metadata meaningful.
pub(crate) fn validate_analysis_epsilon(
    context: &'static str,
    epsilon: f32,
) -> Result<(), AnalysisError> {
    if epsilon >= 0.0 && epsilon.is_finite() {
        return Ok(());
    }

    Err(AnalysisError::propagation(
        context,
        NyError::InvalidSpec(format!(
            "analysis epsilon must be non-negative and finite, got {epsilon}"
        )),
    ))
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn analysis_epsilon_validation_rejects_non_finite_and_negative_values() {
        for epsilon in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_analysis_epsilon("test", epsilon)
                .expect_err("invalid epsilon must fail even with a custom input");
            assert!(err.to_string().contains("epsilon"), "err = {err}");
            assert!(err.to_string().contains("non-negative"), "err = {err}");
        }
    }

    #[test]
    fn analysis_epsilon_validation_accepts_zero_and_positive_values() {
        validate_analysis_epsilon("test", 0.0).expect("zero-radius analysis is valid");
        validate_analysis_epsilon("test", 0.01).expect("positive epsilon is valid");
    }
}
