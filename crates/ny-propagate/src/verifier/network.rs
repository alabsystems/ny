// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network (sequential) verification path.

use super::Verifier;
use crate::beta_crown::BetaCrownVerifier;
use crate::bounds::nan_propagating_min;
use crate::composition::certificate::{BoundCertificate, BoundCertificationResult};
use crate::network::Network;
use crate::types::PropagationMethod;
use ny_core::{
    GemmEngine, HeuristicUsed, MethodUsed, NyError, Result, SoundnessProvenance,
    VerificationResult, VerificationSpec,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::{debug, info};

enum NetworkBoundsResult {
    Completed {
        output_bounds: BoundedTensor,
        actual_method: MethodUsed,
        provenance: SoundnessProvenance,
    },
    Timeout {
        partial_bounds: Option<BoundedTensor>,
        actual_method: MethodUsed,
        provenance: SoundnessProvenance,
    },
}

impl Verifier {
    /// Verify a specification on a Network.
    ///
    /// Propagates input bounds through the network using the configured method
    /// and checks if output bounds satisfy the specification.
    ///
    /// # REQUIRES
    /// - `network` layer input/output dimensions must be compatible (valid network)
    /// - `spec.input_bounds.len()` matches network input dimension
    /// - `spec.output_bounds.len()` matches network output dimension
    /// - Input bounds must be well-formed: `∀b ∈ spec.input_bounds: b.lower <= b.upper`
    ///
    /// # ENSURES
    /// - If `result == Ok(Verified)`, output bounds satisfy spec for all inputs in input_bounds
    /// - If `result == Ok(Violated)`, a counterexample exists within input_bounds
    /// - Result is sound: no false positives for `Verified` status
    pub fn verify(&self, network: &Network, spec: &VerificationSpec) -> Result<VerificationResult> {
        self.verify_with_engine(network, spec, self.engine())
    }

    pub fn verify_with_engine(
        &self,
        network: &Network,
        spec: &VerificationSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<VerificationResult> {
        // Guard: reject empty spec before wasting propagation compute (#2266)
        if spec.output_bounds().is_empty() {
            return Err(NyError::InvalidSpec(
                "empty output_bounds in specification — nothing to verify".to_string(),
            ));
        }

        // Convert input bounds to bounded tensor
        let input_bounds = Self::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;

        if self.config.method == PropagationMethod::BetaCrown {
            return self.verify_beta_crown(network, &input_bounds, spec, engine);
        }

        match self.propagate_network_bounds_with_engine(
            network,
            &input_bounds,
            spec.timeout_ms(),
            engine,
        )? {
            NetworkBoundsResult::Completed {
                output_bounds,
                actual_method,
                provenance,
            } => self.check_spec(
                &output_bounds,
                spec.output_bounds(),
                Some(actual_method),
                provenance,
            ),
            NetworkBoundsResult::Timeout {
                partial_bounds,
                actual_method,
                provenance,
            } => Ok(VerificationResult::Timeout {
                provenance,
                partial_bounds: partial_bounds.as_ref().map(Self::flatten_output_bounds),
                actual_method: Some(actual_method),
            }),
        }
    }

    /// Certify all output bounds for a sequential network without fabricating
    /// an output-constrained verification spec.
    pub fn certify_network_bounds(
        &self,
        model_id: impl Into<String>,
        network: &Network,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
    ) -> Result<BoundCertificationResult> {
        self.certify_network_bounds_with_engine(
            model_id,
            network,
            input_bounds,
            timeout_ms,
            self.engine(),
        )
    }

    fn certify_network_bounds_with_engine(
        &self,
        model_id: impl Into<String>,
        network: &Network,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundCertificationResult> {
        if self.config.method == PropagationMethod::BetaCrown {
            return Err(NyError::UnsupportedOp(
                "certify_network_bounds does not support PropagationMethod::BetaCrown because \
                 the current beta-crown path depends on output constraints"
                    .to_string(),
            ));
        }

        let model_id = model_id.into();
        match self.propagate_network_bounds_with_engine(
            network,
            input_bounds,
            timeout_ms,
            engine,
        )? {
            NetworkBoundsResult::Completed {
                output_bounds,
                actual_method,
                provenance,
            } => Ok(BoundCertificationResult::Certified(
                BoundCertificate::try_new(model_id, output_bounds, actual_method, provenance)?,
            )),
            NetworkBoundsResult::Timeout {
                partial_bounds,
                actual_method,
                provenance,
            } => Ok(BoundCertificationResult::Timeout {
                partial: partial_bounds
                    .map(|bounds| {
                        BoundCertificate::try_new(
                            model_id.clone(),
                            bounds,
                            actual_method.clone(),
                            provenance.clone(),
                        )
                    })
                    .transpose()?,
                actual_method,
                soundness: provenance,
            }),
        }
    }

    fn propagate_network_bounds_with_engine(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<NetworkBoundsResult> {
        let engine = self.resolve_engine(engine);
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        let past_deadline = || deadline.map(|d| Instant::now() >= d).unwrap_or(false);

        info!(
            "Starting verification with {:?}, {} layers, deadline={:?}",
            self.config.method,
            network.layers.len(),
            deadline.map(|d| d.duration_since(Instant::now())),
        );

        let actual_method = self.config.method.method_used();
        let output_bounds = match self.config.method {
            PropagationMethod::Ibp => network.propagate_ibp(input_bounds)?,
            PropagationMethod::Crown => {
                if past_deadline() {
                    info!("Timeout before CROWN backward pass");
                    return Ok(NetworkBoundsResult::Timeout {
                        provenance: SoundnessProvenance::default(),
                        partial_bounds: None,
                        actual_method: MethodUsed::Crown,
                    });
                }
                network.propagate_crown_with_engine_and_deadline(input_bounds, engine, deadline)?
            }
            PropagationMethod::AlphaCrown => {
                if past_deadline() {
                    info!("Timeout before α-CROWN optimization");
                    return Ok(NetworkBoundsResult::Timeout {
                        provenance: SoundnessProvenance::default(),
                        partial_bounds: None,
                        actual_method: MethodUsed::AlphaCrown,
                    });
                }
                let alpha_config = self.alpha_crown_config(deadline);
                network.propagate_alpha_crown_with_config_and_engine(
                    input_bounds,
                    &alpha_config,
                    engine,
                )?
            }
            // SDP-CROWN's ReLU offsets and concretization are only valid over an ℓ2 input ball,
            // and a `VerificationSpec` carries per-element ℓ∞ bounds. The ball of the box's
            // half-width ε covers a strict subset of the box (its corners sit at ℓ2 distance
            // ε√n), and the ball that does contain the box has radius ε√n, over which
            // ‖a‖₂·ε√n >= ‖a‖₁·ε leaves the concretization no tighter than CROWN's. Neither
            // answers a box spec, so refuse rather than certify a region we did not bound.
            PropagationMethod::SdpCrown => {
                return Err(NyError::UnsupportedOp(
                    "SDP-CROWN requires an ℓ2 input ball, but the specification declares an \
                     ℓ∞ input box; use CROWN or α-CROWN instead"
                        .to_string(),
                ));
            }
            PropagationMethod::BetaCrown => {
                return Err(NyError::InternalError(
                    "propagate_network_bounds_with_engine does not support BetaCrown".to_string(),
                ));
            }
        };

        let output_bounds = Self::sanitize_output_bounds(output_bounds)?;
        let mut provenance =
            crate::soundness::soundness_provenance_for_network(network, &self.config.method);
        let sqrt_negative_domain_nodes =
            crate::soundness::count_sqrt_negative_domain_network(network, input_bounds)?;
        if sqrt_negative_domain_nodes > 0 {
            let mut heuristics = provenance.heuristics_used().to_vec();
            heuristics.push(HeuristicUsed::SqrtNegativeDomain {
                num_nodes: sqrt_negative_domain_nodes,
            });
            provenance = SoundnessProvenance::from_heuristics(heuristics);
        }

        Ok(NetworkBoundsResult::Completed {
            output_bounds,
            actual_method,
            provenance,
        })
    }

    /// Verify using β-CROWN with branch-and-bound search.
    ///
    /// β-CROWN handles verification directly (returns VerificationResult)
    /// rather than just computing output bounds.
    fn verify_beta_crown(
        &self,
        network: &Network,
        input: &BoundedTensor,
        spec: &VerificationSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<VerificationResult> {
        debug!("β-CROWN verification");

        if spec.output_bounds().is_empty() {
            return Err(NyError::InvalidSpec(
                "β-CROWN requires at least one output bound".to_string(),
            ));
        }

        // Create β-CROWN verifier with default config
        // The threshold is derived from output specification
        let config = self.beta_crown_config(spec);
        let beta_verifier = match self.engine_arc() {
            Some(stored_engine) => BetaCrownVerifier::new_with_engine(config, stored_engine),
            None => BetaCrownVerifier::new(config),
        };

        // Derive BaB threshold from output bounds (#2241).
        //
        // For multi-output specs, use min(all finite lower bounds) as the BaB
        // threshold. The BaB engine verifies `min(all_outputs) > threshold` via
        // lower_scalar(), so using the minimum ensures the search explores all
        // domains where ANY output might violate its requirement.
        //
        // After BaB, per-output validation (below) checks each output against
        // its specific bound, downgrading Verified → Unknown if needed.
        //
        // Ref: alpha-beta-CROWN uses per-spec thresholds via `rhs` vector
        // (complete_verifier/bab.py). Full per-output BaB is future work.
        let threshold = spec
            .output_bounds()
            .iter()
            .map(|b| b.lower())
            .filter(|l| l.is_finite())
            .fold(f32::INFINITY, nan_propagating_min);
        // If all lower bounds are -inf (no finite lower constraints),
        // use -inf so BaB trivially verifies the lower-bound direction.
        let threshold = if threshold == f32::INFINITY {
            f32::NEG_INFINITY
        } else {
            threshold
        };

        let result = beta_verifier.verify_with_engine(network, input, threshold, engine, None)?;
        self.finalize_beta_crown_result(network, input, spec, result, threshold)
    }
}
