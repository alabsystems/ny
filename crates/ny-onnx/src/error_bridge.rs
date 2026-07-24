// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bridge conversions from ny-onnx analysis error types to `NyError`.
//!
//! This module provides `From` implementations so that functions returning
//! `ny_core::Result<T>` can use `?` on calls to diff, quantize, profile,
//! and sensitivity analysis without manual `.map_err()`.
//!
//! Mapping policy (per designs/2026-02-11-nyerror-bridge-plan.md):
//! - Loading/parsing failures → `NyError::ModelLoad`
//! - Input/config/schema failures → `NyError::InvalidSpec`
//! - Propagation/internal analysis failures → `NyError::InternalError`
//! - `DiffError` messages prefixed with "diff:" for traceability.
//! - `AnalysisError` messages passed through without prefix (call sites provide context).

use ny_core::NyError;

use crate::analysis_error::AnalysisError;
use crate::diff::DiffError;

impl From<DiffError> for NyError {
    fn from(e: DiffError) -> Self {
        match e {
            DiffError::LoadError(s) => NyError::ModelLoad(format!("diff: {s}")),
            DiffError::OrtUnavailable => {
                NyError::UnsupportedConfiguration("diff: ONNX Runtime support not enabled".into())
            }
            #[cfg(feature = "ort")]
            DiffError::OrtError(e) => NyError::ModelLoad(format!("diff: {e}")),
            DiffError::IoError(e) => NyError::ModelLoad(format!("diff: {e}")),
            DiffError::NpyError(s) => NyError::ModelLoad(format!("diff: npy: {s}")),
            DiffError::InputShapeMismatch { model_a, model_b } => NyError::InvalidSpec(format!(
                "diff: input shape mismatch: model A {model_a:?} vs model B {model_b:?}"
            )),
            DiffError::LayerNotFound(s) => {
                NyError::InvalidSpec(format!("diff: layer not found: {s}"))
            }
            DiffError::NoLayers => NyError::InvalidSpec("diff: no layers to compare".to_string()),
        }
    }
}

/// Consolidated bridge for the shared `AnalysisError` type used by
/// quantize, sensitivity, and profile modules.
///
/// Previously, three structurally identical `From` impls existed (one per
/// module-specific error alias). Now that the three error types are unified
/// into `AnalysisError`, a single impl handles all three.
///
/// With structured payloads (#2295), `LoadError` and `PropagationError` now
/// carry `Box<NyError>` instead of `String`. The bridge recovers the
/// original `NyError` variant, preserving type fidelity across the
/// analysis boundary.
impl From<AnalysisError> for NyError {
    fn from(e: AnalysisError) -> Self {
        match e {
            AnalysisError::LoadError { source, .. } => *source,
            AnalysisError::PropagationError { source, .. } => *source,
            AnalysisError::NoLayers { context } => {
                NyError::InvalidSpec(format!("{context}: no layers in network"))
            }
            AnalysisError::InvalidInputShape { context, detail } => {
                NyError::InvalidSpec(format!("{context}: invalid input shape: {detail}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-contract tests: prove that `?` works for each error type
    // when returning `ny_core::Result<T>`.

    fn diff_to_ny_core_result() -> ny_core::Result<()> {
        let err: Result<(), DiffError> = Err(DiffError::NoLayers);
        err?;
        Ok(())
    }

    fn analysis_to_ny_core_result() -> ny_core::Result<()> {
        let err: Result<(), AnalysisError> = Err(AnalysisError::no_layers("test"));
        err?;
        Ok(())
    }

    #[test]
    fn test_diff_error_question_mark_into_ny_core_result() {
        let result = diff_to_ny_core_result();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("diff:"),
            "Error message should contain module prefix 'diff:', got: {msg}"
        );
    }

    #[test]
    fn test_analysis_error_question_mark_into_ny_core_result() {
        let result = analysis_to_ny_core_result();
        assert!(result.is_err());
        // AnalysisError::NoLayers maps to NyError::InvalidSpec
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("no layers"),
            "Error message should contain 'no layers', got: {msg}"
        );
    }

    #[test]
    fn test_diff_error_variant_mapping() {
        // LoadError → ModelLoad
        let e: NyError = DiffError::LoadError("file not found".into()).into();
        assert!(
            matches!(e, NyError::ModelLoad(_)),
            "DiffError::LoadError should map to NyError::ModelLoad"
        );

        // InputShapeMismatch → InvalidSpec
        let e: NyError = DiffError::InputShapeMismatch {
            model_a: vec![1, 3, 224],
            model_b: vec![1, 3, 256],
        }
        .into();
        assert!(
            matches!(e, NyError::InvalidSpec(_)),
            "DiffError::InputShapeMismatch should map to NyError::InvalidSpec"
        );

        // LayerNotFound → InvalidSpec
        let e: NyError = DiffError::LayerNotFound("conv1".into()).into();
        assert!(
            matches!(e, NyError::InvalidSpec(_)),
            "DiffError::LayerNotFound should map to NyError::InvalidSpec"
        );
    }

    #[test]
    fn test_analysis_error_variant_mapping() {
        // LoadError wrapping ModelLoad → recovers ModelLoad
        let e: NyError = AnalysisError::load("test", NyError::ModelLoad("bad model".into())).into();
        assert!(
            matches!(e, NyError::ModelLoad(_)),
            "LoadError wrapping ModelLoad should recover ModelLoad, got: {e:?}"
        );

        // PropagationError wrapping InternalError → recovers InternalError
        let e: NyError =
            AnalysisError::propagation("test", NyError::InternalError("shape fail".into())).into();
        assert!(
            matches!(e, NyError::InternalError(_)),
            "PropagationError wrapping InternalError should recover InternalError, got: {e:?}"
        );

        // PropagationError wrapping ShapeMismatch → recovers ShapeMismatch (#2295)
        let e: NyError = AnalysisError::propagation(
            "test",
            NyError::ShapeMismatch {
                expected: vec![1, 3],
                got: vec![1, 4],
            },
        )
        .into();
        assert!(
            matches!(e, NyError::ShapeMismatch { .. }),
            "PropagationError wrapping ShapeMismatch should recover ShapeMismatch, got: {e:?}"
        );

        // propagation_msg → wraps as InternalError → recovers InternalError
        let e: NyError = AnalysisError::propagation_msg("test", "node missing").into();
        assert!(
            matches!(e, NyError::InternalError(_)),
            "propagation_msg should produce InternalError, got: {e:?}"
        );

        // InvalidInputShape → InvalidSpec
        let e: NyError = AnalysisError::invalid_input_shape("test", "wrong dims").into();
        assert!(matches!(e, NyError::InvalidSpec(_)));

        // NoLayers → InvalidSpec
        let e: NyError = AnalysisError::no_layers("test").into();
        assert!(matches!(e, NyError::InvalidSpec(_)));
    }
}
