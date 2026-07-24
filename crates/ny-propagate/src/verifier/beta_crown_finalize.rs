// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Beta-CROWN result finalization: counterexample validation, output bound
//! assembly, and BaB status mapping.

use super::Verifier;
use crate::beta_crown::{BabVerificationStatus, BetaCrownResult};
use crate::network::Network;
use crate::types::PropagationMethod;
use ndarray::{ArrayD, IxDyn};
use ny_core::{
    Bound, HeuristicUsed, MethodUsed, NyError, Result, SoundnessProvenance, VerificationResult,
    VerificationSpec,
};
use ny_tensor::BoundedTensor;
use tracing::{debug, warn};

impl Verifier {
    pub(crate) fn finalize_beta_crown_result(
        &self,
        network: &Network,
        input: &BoundedTensor,
        spec: &VerificationSpec,
        result: BetaCrownResult,
        threshold: f32,
    ) -> Result<VerificationResult> {
        // Prefer beta-CROWN's reported bounds, falling back to conservative bounds.
        let fallback_bound = match &result.result {
            BabVerificationStatus::Verified => Bound::new_allow_infinite(threshold, f32::INFINITY),
            BabVerificationStatus::Violated { .. } | BabVerificationStatus::PotentialViolation => {
                Bound::new_allow_infinite(f32::NEG_INFINITY, threshold)
            }
            BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout => {
                Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)
            }
        };

        let mut output_bounds = vec![fallback_bound; spec.output_bounds().len()];
        if let Some(bounds) = &result.output_bounds {
            Self::apply_bab_output_bounds(&mut output_bounds, bounds);
        }

        let mut provenance = crate::soundness::soundness_provenance_for_network(
            network,
            &PropagationMethod::BetaCrown,
        );
        let sqrt_negative_domain_nodes =
            crate::soundness::count_sqrt_negative_domain_network(network, input)?;
        if sqrt_negative_domain_nodes > 0 {
            let mut heuristics = provenance.heuristics_used().to_vec();
            heuristics.push(HeuristicUsed::SqrtNegativeDomain {
                num_nodes: sqrt_negative_domain_nodes,
            });
            provenance = SoundnessProvenance::from_heuristics(heuristics);
        }

        match result.result {
            BabVerificationStatus::Verified => {
                // Per-output validation (#2241): BaB proved min(all_outputs) > min_threshold,
                // but each output must meet its *own* bound. Downgrade to Unknown if any
                // individual output's computed bounds don't satisfy its spec requirement.
                for (computed, required) in output_bounds.iter().zip(spec.output_bounds()) {
                    let lower_gap = if computed.lower() < required.lower() {
                        required.lower() - computed.lower()
                    } else {
                        0.0
                    };
                    let upper_gap = if computed.upper() > required.upper() {
                        computed.upper() - required.upper()
                    } else {
                        0.0
                    };
                    let gap = lower_gap.max(upper_gap);
                    if gap > 0.0 {
                        debug!(
                            "beta-CROWN verified globally but per-output gap={}: \
                             computed=[{}, {}], required=[{}, {}]",
                            gap,
                            computed.lower(),
                            computed.upper(),
                            required.lower(),
                            required.upper()
                        );
                        return Ok(VerificationResult::Unknown {
                            provenance,
                            bounds: output_bounds,
                            reason: ny_core::UnknownReason::BoundsTooLoose { gap: Some(gap) },
                            actual_method: Some(MethodUsed::BetaCrown),
                        });
                    }
                }
                Ok(VerificationResult::Verified {
                    provenance,
                    output_bounds,
                    proof: None,
                    actual_method: Some(MethodUsed::BetaCrown),
                })
            }
            BabVerificationStatus::Violated {
                counterexample,
                output: reported_output,
            } => {
                match self.validate_beta_crown_counterexample(network, input, spec, &counterexample)
                {
                    Ok(Some(actual_output)) => {
                        let mut validated_output_bounds = output_bounds.clone();
                        for (idx, value) in actual_output.iter().copied().enumerate() {
                            if idx >= validated_output_bounds.len() {
                                break;
                            }
                            if value.is_finite() {
                                validated_output_bounds[idx] = Bound::concrete(value);
                            }
                        }
                        let details = ny_core::InformativeCounterexample::new(
                            counterexample.clone(),
                            actual_output.clone(),
                            Some(&validated_output_bounds),
                        );
                        Ok(VerificationResult::Violated {
                            provenance,
                            counterexample,
                            output: actual_output,
                            details: Some(Box::new(details)),
                            actual_method: Some(MethodUsed::BetaCrown),
                        })
                    }
                    Ok(None) => {
                        warn!(
                            reported_output_len = reported_output.len(),
                            counterexample_len = counterexample.len(),
                            "beta-CROWN reported a counterexample that failed concrete \
                         spec validation; downgrading Violated to Unknown"
                        );
                        Ok(VerificationResult::Unknown {
                            provenance,
                            bounds: output_bounds,
                            reason: ny_core::UnknownReason::PotentialViolation,
                            actual_method: Some(MethodUsed::BetaCrown),
                        })
                    }
                    Err(err) => {
                        warn!(
                        ?err,
                        reported_output_len = reported_output.len(),
                        counterexample_len = counterexample.len(),
                        "beta-CROWN counterexample validation errored; downgrading Violated to Unknown"
                    );
                        Ok(VerificationResult::Unknown {
                            provenance,
                            bounds: output_bounds,
                            reason: ny_core::UnknownReason::PotentialViolation,
                            actual_method: Some(MethodUsed::BetaCrown),
                        })
                    }
                }
            }
            BabVerificationStatus::PotentialViolation => Ok(VerificationResult::Unknown {
                provenance,
                bounds: output_bounds,
                reason: ny_core::UnknownReason::PotentialViolation,
                actual_method: Some(MethodUsed::BetaCrown),
            }),
            BabVerificationStatus::Unknown { reason } => Ok(VerificationResult::Unknown {
                provenance,
                bounds: output_bounds,
                reason: ny_core::UnknownReason::from(reason),
                actual_method: Some(MethodUsed::BetaCrown),
            }),
            BabVerificationStatus::Timeout => Ok(VerificationResult::Timeout {
                provenance,
                partial_bounds: Some(output_bounds),
                actual_method: Some(MethodUsed::BetaCrown),
            }),
        }
    }

    pub(crate) fn validate_beta_crown_counterexample(
        &self,
        network: &Network,
        input: &BoundedTensor,
        spec: &VerificationSpec,
        counterexample: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        const TOLERANCE: f32 = 1e-6;

        if counterexample.len() != input.len() {
            return Ok(None);
        }

        for (&value, (&lower, &upper)) in counterexample
            .iter()
            .zip(input.lower().iter().zip(input.upper().iter()))
        {
            if !value.is_finite()
                || !lower.is_finite()
                || !upper.is_finite()
                || value < lower - TOLERANCE
                || value > upper + TOLERANCE
            {
                return Ok(None);
            }
        }

        let candidate = ArrayD::from_shape_vec(IxDyn(input.shape()), counterexample.to_vec())
            .map_err(|err| {
                NyError::InternalError(format!(
                    "beta-CROWN counterexample shape mismatch for input {:?}: {err}",
                    input.shape()
                ))
            })?;
        let concrete = BoundedTensor::concrete(candidate)?;
        let output = network.propagate_ibp(&concrete)?;
        let actual_output: Vec<f32> = output.lower().iter().copied().collect();

        if actual_output.len() < spec.output_bounds().len() {
            return Err(NyError::InvalidSpec(format!(
                "Network produced {} outputs but spec requires {} during beta-CROWN counterexample validation",
                actual_output.len(),
                spec.output_bounds().len(),
            )));
        }

        if actual_output.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }

        let violates_spec = actual_output
            .iter()
            .zip(spec.output_bounds().iter())
            .any(|(&value, required)| value < required.lower() || value > required.upper());

        Ok(violates_spec.then_some(actual_output))
    }

    /// Apply BaB output bounds to the fallback array, skipping NaN entries.
    ///
    /// NaN bounds indicate numerical corruption in CROWN propagation or BaB.
    /// These are logged as warnings rather than silently dropped (#2589).
    pub(super) fn apply_bab_output_bounds(output_bounds: &mut [Bound], bab_bounds: &BoundedTensor) {
        let mut nan_count = 0usize;
        for (idx, (&l, &u)) in bab_bounds
            .lower()
            .iter()
            .zip(bab_bounds.upper().iter())
            .enumerate()
        {
            if idx >= output_bounds.len() {
                break;
            }
            if l.is_nan() || u.is_nan() {
                nan_count += 1;
                tracing::warn!(
                    idx,
                    lower = l,
                    upper = u,
                    "BaB result contains NaN output bounds — numerical corruption \
                     in bound propagation. Falling back to conservative bounds."
                );
            } else if l <= u {
                output_bounds[idx] = Bound::new_allow_infinite(l, u);
            }
        }
        if nan_count > 0 {
            tracing::warn!(
                nan_count,
                total = output_bounds.len(),
                "BaB produced NaN output bounds — downstream results may show \
                 Unknown/Timeout instead of the true verification status"
            );
        }
    }
}
