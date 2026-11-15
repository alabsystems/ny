// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-margin-alpha-complex (P8 bank gate): the ONNX Runtime implementation
//! of [`RootAdjudicationOracle`].
//!
//! The containment fence compares a root-terminal multi-objective `Verified`'s
//! certified objective lower bounds against ny's own sound f64 walk, which
//! shares ny's graph with the bounds. This oracle evaluates the ORIGINAL model
//! file with ONNX Runtime — the runtime the trusted sat gate
//! (`confirm_violation_with_ort`) and the organizer use — built straight from
//! the protobuf (`OrtForward::from_path`), never through ny's graph loader. A
//! certified lower bound above the ORT objective value at an in-box sample is
//! then evidence that ny's graph is not the model, and the verdict is
//! downgraded.
//!
//! The session is served from the process-global ORT session cache
//! (`#ort-session-once`), so building a forward per sample costs one cache
//! lookup, and this type holds only the model path: it is trivially `Send +
//! Sync`, as the seam requires.

use std::path::{Path, PathBuf};

use ny_propagate::beta_crown::root_adjudication::RootAdjudicationOracle;

/// ONNX Runtime forward of the original model file (gzip-aware).
pub(super) struct OrtRootAdjudicationOracle {
    model: PathBuf,
}

impl OrtRootAdjudicationOracle {
    pub(super) fn new(model: &Path) -> Self {
        Self {
            model: model.to_path_buf(),
        }
    }
}

impl RootAdjudicationOracle for OrtRootAdjudicationOracle {
    fn name(&self) -> &'static str {
        "ort"
    }

    fn evaluate(&self, flat_input: &[f32]) -> Result<Vec<f32>, String> {
        let mut forward = ny_onnx::diff::OrtForward::from_path(&self.model, flat_input.len())
            .map_err(|error| {
                format!(
                    "building the ONNX Runtime session for {}: {error}",
                    self.model.display()
                )
            })?;
        let output = forward
            .run(flat_input)
            .map_err(|error| format!("ONNX Runtime inference failed: {error}"))?;
        if output.is_empty() {
            return Err("ONNX Runtime returned no outputs".to_string());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing model is an `Err` (which the fence counts and, when the
    /// adjudication is armed, fails CLOSED on) — never a fabricated output.
    #[test]
    fn ort_root_adjudication_oracle_reports_a_missing_model_as_an_error() {
        let oracle = OrtRootAdjudicationOracle::new(Path::new(
            "/nonexistent/vit-margin-alpha-complex/no_such_model.onnx",
        ));
        assert_eq!(oracle.name(), "ort");
        let result = oracle.evaluate(&[0.0_f32; 4]);
        assert!(
            result.is_err(),
            "a missing model must not yield an output: {result:?}"
        );
    }
}
