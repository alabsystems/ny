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
