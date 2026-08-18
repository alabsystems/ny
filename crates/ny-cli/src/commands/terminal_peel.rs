// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Typed receipt and output-coordinate policy for terminal-activation peeling.

use anyhow::Result;
use std::borrow::Cow;

/// Exact terminal activation removed by a joint model/property loader pass.
///
/// A boolean receipt is insufficient at a witness boundary: a peeled Sigmoid
/// can be mapped elementwise back to original-model outputs, while a peeled
/// Softmax-family result cannot be rehydrated without retaining its axis/group
/// semantics.  Keep this type attached to the loaded model until rendering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AppliedTerminalPeel {
    #[default]
    None,
    Softmax,
    LogSoftmax,
    Sigmoid,
}

impl AppliedTerminalPeel {
    pub(crate) fn from_report(report: &ny_onnx::PeelOffReport) -> Result<Self> {
        if !report.peeled {
            if report.layer_type.is_some() {
                anyhow::bail!(
                    "terminal peel reported an activation type without reporting application"
                );
            }
            return Ok(Self::None);
        }
        match report.layer_type.as_ref() {
            Some(ny_core::LayerType::Softmax) => Ok(Self::Softmax),
            Some(ny_core::LayerType::LogSoftmax) => Ok(Self::LogSoftmax),
            Some(ny_core::LayerType::Sigmoid) => Ok(Self::Sigmoid),
            Some(other) => {
                anyhow::bail!("terminal peel reported unsupported applied activation {other:?}")
            }
            None => anyhow::bail!("terminal peel reported applied without an activation type"),
        }
    }

    pub(crate) fn applied(self) -> bool {
        self != Self::None
    }

    pub(crate) fn is_sigmoid(self) -> bool {
        self == Self::Sigmoid
    }

    pub(crate) fn needs_original_output_rehydration(self) -> bool {
        matches!(self, Self::Softmax | Self::LogSoftmax)
    }

    pub(crate) fn activation_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Softmax => "Softmax",
            Self::LogSoftmax => "LogSoftmax",
            Self::Sigmoid => "Sigmoid",
        }
    }

    /// Return witness values in original-model output coordinates when that
    /// transformation is unambiguous from the typed receipt alone.
    pub(crate) fn output_in_original_coordinates(self, output: &[f32]) -> Option<Cow<'_, [f32]>> {
        match self {
            Self::None => Some(Cow::Borrowed(output)),
            Self::Sigmoid => Some(Cow::Owned(
                output
                    .iter()
                    .map(|&z| {
                        if z >= 0.0 {
                            (1.0 / (1.0 + (-f64::from(z)).exp())) as f32
                        } else {
                            let exp_z = f64::from(z).exp();
                            (exp_z / (1.0 + exp_z)) as f32
                        }
                    })
                    .collect(),
            )),
            Self::Softmax | Self::LogSoftmax => None,
        }
    }

    /// Human-readable witness presentation. Softmax-family values remain
    /// useful diagnostics, but the label must make their peeled coordinates
    /// explicit and must never call them original-model `Y` values.
    pub(crate) fn human_witness_output(self, output: &[f32]) -> (&'static str, Cow<'_, [f32]>) {
        match self {
            Self::None => ("Counterexample output", Cow::Borrowed(output)),
            Self::Sigmoid => (
                "Counterexample output",
                self.output_in_original_coordinates(output)
                    .expect("Sigmoid output mapping is defined"),
            ),
            Self::Softmax => (
                "Counterexample preactivation (peeled Softmax; not original-model Y)",
                Cow::Borrowed(output),
            ),
            Self::LogSoftmax => (
                "Counterexample preactivation (peeled LogSoftmax; not original-model Y)",
                Cow::Borrowed(output),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_witness_is_mapped_to_original_output_coordinates() {
        let (label, output) =
            AppliedTerminalPeel::Sigmoid.human_witness_output(&[-1000.0, 0.0, 2.0, 1000.0]);
        assert_eq!(label, "Counterexample output");
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.5);
        assert!(output[2] > 0.88 && output[2] < 0.89);
        assert_eq!(output[3], 1.0);
    }

    #[test]
    fn softmax_family_witness_is_never_presented_as_original_y() {
        for peel in [
            AppliedTerminalPeel::Softmax,
            AppliedTerminalPeel::LogSoftmax,
        ] {
            assert!(peel.output_in_original_coordinates(&[1.0]).is_none());
            let (label, output) = peel.human_witness_output(&[1.0]);
            assert!(label.contains("preactivation"));
            assert!(label.contains("not original-model Y"));
            assert_eq!(&*output, &[1.0]);
        }
    }

    #[test]
    fn inconsistent_loader_receipt_is_rejected() {
        let report = ny_onnx::PeelOffReport {
            peeled: false,
            layer_type: Some(ny_core::LayerType::Softmax),
            reason: Some("inconsistent".to_string()),
        };
        assert!(AppliedTerminalPeel::from_report(&report).is_err());
    }
}
