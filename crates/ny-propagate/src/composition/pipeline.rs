// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pipeline composition for chaining model bounds with exact method tags.
//!
//! Verifies multi-model pipelines (e.g., encoder → decoder → vocoder)
//! by propagating bounds from one model's output to the next model's input.
//! Each stage records its contribution to the certificate chain.
//!
//! Part of #3920.

use ny_core::{NyError, Result, SoundnessProvenance};
use ny_tensor::BoundedTensor;

use super::certificate::{BoundCertificate, BoundProvenance};

fn provenance_strength(provenance: BoundProvenance) -> u8 {
    match provenance {
        BoundProvenance::Ibp => 0,
        BoundProvenance::Crown => 1,
        BoundProvenance::AlphaCrown => 2,
    }
}

/// A single stage in a verification pipeline.
///
/// Each stage preserves the exact input witness used to verify that model plus
/// the resulting per-model certificate.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    input_bounds: BoundedTensor,
    certificate: BoundCertificate,
}

impl PipelineStage {
    /// Input bounds that justified this stage in the composed chain.
    pub fn input_bounds(&self) -> &BoundedTensor {
        &self.input_bounds
    }

    /// The per-model certificate produced from those stage inputs.
    pub fn certificate(&self) -> &BoundCertificate {
        &self.certificate
    }
}

/// Certificate for a complete pipeline verification.
#[derive(Debug, Clone)]
pub struct PipelineCertificate {
    stages: Vec<PipelineStage>,
    overall_provenance: BoundProvenance,
    overall_soundness: SoundnessProvenance,
    final_bounds: BoundedTensor,
}

impl PipelineCertificate {
    /// Per-stage witnesses and certificates, in pipeline order.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Weakest-link provenance across the pipeline.
    pub fn overall_provenance(&self) -> BoundProvenance {
        self.overall_provenance
    }

    /// Merged soundness provenance across all stages.
    pub fn overall_soundness(&self) -> &SoundnessProvenance {
        &self.overall_soundness
    }

    /// Final output bounds from the last stage in the chain.
    pub fn final_bounds(&self) -> &BoundedTensor {
        &self.final_bounds
    }
}

/// Merge soundness provenance across pipeline stages.
///
/// Concatenates heuristics in pipeline order, deduplicates identical entries,
/// and builds the summary through `SoundnessProvenance::from_heuristics`.
fn merge_soundness(stages: &[PipelineStage]) -> SoundnessProvenance {
    let mut heuristics = Vec::new();
    for stage in stages {
        for heuristic in stage.certificate().soundness().heuristics_used() {
            if !heuristics.contains(heuristic) {
                heuristics.push(heuristic.clone());
            }
        }
    }
    SoundnessProvenance::from_heuristics(heuristics)
}

/// Verifier for sequential multi-model pipelines.
///
/// Usage:
/// ```ignore
/// let mut pipeline = PipelineVerifier::new();
/// pipeline.push_stage(encoder_input_bounds, encoder_cert)?;
/// pipeline.push_stage(decoder_input_bounds, decoder_cert)?;
/// let cert = pipeline.finalize()?;
/// ```
#[derive(Default)]
pub struct PipelineVerifier {
    stages: Vec<PipelineStage>,
}

impl PipelineVerifier {
    /// Create a new empty pipeline verifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a completed stage to the pipeline.
    ///
    /// `input_bounds` should come from the perturbation spec (first stage) or
    /// from the next-stage verification domain that contains the previous
    /// stage's output certificate.
    pub fn push_stage(
        &mut self,
        input_bounds: BoundedTensor,
        certificate: BoundCertificate,
    ) -> Result<()> {
        if let Some(previous) = self.stages.last() {
            validate_stage_transition(previous, &input_bounds, certificate.model_id())?;
        }
        self.stages.push(PipelineStage {
            input_bounds,
            certificate,
        });
        Ok(())
    }

    /// Get the output bounds of the last completed stage.
    pub fn last_output(&self) -> Result<&BoundedTensor> {
        self.stages
            .last()
            .map(|stage| stage.certificate().output_bounds())
            .ok_or_else(|| {
                NyError::InvalidConfig(
                    "PipelineVerifier: no completed stages have been added yet".to_string(),
                )
            })
    }

    /// Finalize the pipeline, producing a certificate if all stages are verified
    /// and the chain is sound.
    ///
    /// Only method tags supported by Packet A bound certificates may appear in a
    /// finalized pipeline; unsupported tags (including BetaCrown) are rejected
    /// at `BoundCertificate::try_new` time, before a stage can enter the chain.
    pub fn finalize(self) -> Result<PipelineCertificate> {
        if self.stages.is_empty() {
            return Err(NyError::InvalidConfig(
                "PipelineVerifier: no stages added".to_string(),
            ));
        }

        validate_chain(&self.stages)?;

        let final_bounds = self
            .stages
            .last()
            .map(|stage| stage.certificate().output_bounds().clone())
            .ok_or_else(|| {
                NyError::InvalidConfig("PipelineVerifier: empty pipeline".to_string())
            })?;

        let overall_provenance = self
            .stages
            .iter()
            .min_by_key(|stage| provenance_strength(stage.certificate().provenance()))
            .map(|stage| stage.certificate().provenance())
            .ok_or_else(|| {
                NyError::InternalError(
                    "PipelineVerifier: validated pipeline unexpectedly had no stages".to_string(),
                )
            })?;

        let overall_soundness = merge_soundness(&self.stages);

        Ok(PipelineCertificate {
            stages: self.stages,
            overall_provenance,
            overall_soundness,
            final_bounds,
        })
    }
}

fn validate_chain(stages: &[PipelineStage]) -> Result<()> {
    for window in stages.windows(2) {
        validate_stage_transition(
            &window[0],
            window[1].input_bounds(),
            window[1].certificate().model_id(),
        )?;
    }
    Ok(())
}

fn validate_stage_transition(
    previous: &PipelineStage,
    next_input: &BoundedTensor,
    next_model_id: &str,
) -> Result<()> {
    let prev_output = previous.certificate().output_bounds();

    if prev_output.shape() != next_input.shape() {
        return Err(NyError::ShapeMismatch {
            expected: next_input.shape().to_vec(),
            got: prev_output.shape().to_vec(),
        });
    }

    for (idx, ((&prev_l, &prev_u), (&next_l, &next_u))) in prev_output
        .lower()
        .iter()
        .zip(prev_output.upper().iter())
        .zip(next_input.lower().iter().zip(next_input.upper().iter()))
        .enumerate()
    {
        // NaN in any bound is a propagation failure — reject immediately.
        // IEEE 754 ordered comparisons return false for NaN operands, so
        // the containment check below would silently pass NaN values.
        if prev_l.is_nan() || prev_u.is_nan() || next_l.is_nan() || next_u.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "PipelineVerifier: NaN in stage transition — \
                 stage '{}' output[{idx}] = [{prev_l}, {prev_u}], \
                 stage '{next_model_id}' input[{idx}] = [{next_l}, {next_u}]",
                previous.certificate().model_id(),
            )));
        }
        if prev_l < next_l || prev_u > next_u {
            return Err(NyError::InvalidConfig(format!(
                "PipelineVerifier: stage '{}' output [{}, {}] not contained in stage '{}' input \
                 [{}, {}]",
                previous.certificate().model_id(),
                prev_l,
                prev_u,
                next_model_id,
                next_l,
                next_u
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
