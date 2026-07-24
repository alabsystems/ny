// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph (DAG) verification path with fallback chains.

use super::Verifier;
use crate::composition::certificate::{BoundCertificate, BoundCertificationResult};
use crate::network::GraphNetwork;
use crate::types::{CrownBackwardResult, PropagationMethod};
use ny_core::{
    GemmEngine, HeuristicUsed, MethodUsed, NyError, Result, SoundnessProvenance,
    VerificationResult, VerificationSpec,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::{debug, info};

enum GraphBoundsResult {
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

fn actual_method_for_crown_result(result: &CrownBackwardResult) -> PropagationMethod {
    if result.is_fallback() {
        PropagationMethod::Ibp
    } else {
        PropagationMethod::Crown
    }
}

/// Policy-critical errors that must not degrade to IBP (#3706).
///
/// `SoundnessRefusal` means a layer deliberately refused CROWN for policy
/// reasons (e.g., LayerNorm in `Sound` mode). `InternalError` means a
/// programmer-invariant violation. Both must propagate as hard errors rather
/// than being silently converted into IBP fallback results.
///
/// Reference: #1743 (sequential fix), #3107 (lower graph-CROWN dispatch fix).
fn graph_crown_error_must_propagate(error: &NyError) -> bool {
    matches!(
        error,
        NyError::SoundnessRefusal(_) | NyError::InternalError(_)
    )
}

/// Run the BatchedCROWN → FlatCROWN → IBP degradation cascade.
///
/// Returns the output bounds and the propagation method label recorded at the
/// verifier surface.
///
/// Whole-verifier fallback to IBP always reports `Ibp`. Successful fixed-slope
/// CROWN calls may also report `Ibp` when their returned provenance already
/// reflects a forward-bound fallback. Internal alpha-CROWN degradations that
/// are absorbed before reaching this helper may still remain labeled `Crown`.
/// Checks `deadline` before each fallback attempt (#2987).
pub(super) fn crown_fallback_chain(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    mul_binary_relaxation: crate::types::MulBinaryRelaxationMode,
    deadline: Option<Instant>,
) -> Result<(BoundedTensor, PropagationMethod)> {
    match graph.propagate_crown_batched_with_engine_relaxation_and_deadline(
        input_bounds,
        engine,
        mul_binary_relaxation,
        deadline,
    ) {
        Ok(result) => {
            let method = actual_method_for_crown_result(&result);
            Ok((result.bounds, method))
        }
        Err(e) => {
            // Policy-critical errors must not degrade to IBP (#3706).
            if graph_crown_error_must_propagate(&e) {
                return Err(e);
            }
            debug!("Batched CROWN failed ({}); trying flat CROWN", e);
            // Check deadline before flat CROWN fallback (#2987)
            if deadline.map(|d| Instant::now() >= d).unwrap_or(false) {
                info!("Timeout during CROWN fallback chain after batched CROWN failure");
                let b = graph.propagate_ibp(input_bounds)?;
                return Ok((b, PropagationMethod::Ibp));
            }
            match graph.propagate_crown_with_engine_relaxation_and_deadline(
                input_bounds,
                engine,
                mul_binary_relaxation,
                deadline,
            ) {
                Ok(result) => {
                    let method = actual_method_for_crown_result(&result);
                    Ok((result.bounds, method))
                }
                Err(e2) => {
                    // Policy-critical errors must not degrade to IBP (#3706).
                    if graph_crown_error_must_propagate(&e2) {
                        return Err(e2);
                    }
                    info!("Graph CROWN failed ({}); falling back to IBP", e2);
                    let b = graph.propagate_ibp(input_bounds)?;
                    Ok((b, PropagationMethod::Ibp))
                }
            }
        }
    }
}

impl Verifier {
    /// Verify a specification on a GraphNetwork (DAG-based network with binary ops support).
    ///
    /// This method supports models with binary operations like attention MatMul (Q@K^T)
    /// where both inputs are bounded tensors. Use this for transformer models.
    ///
    /// # Example
    /// ```rust,no_run
    /// // Verify a GraphNetwork (for models with attention):
    /// // let graph = model.to_graph_network().unwrap();
    /// // let result = verifier.verify_graph(&graph, &spec).unwrap();
    /// ```
    pub fn verify_graph(
        &self,
        graph: &GraphNetwork,
        spec: &VerificationSpec,
    ) -> Result<VerificationResult> {
        self.verify_graph_with_engine(graph, spec, self.engine())
    }

    pub fn verify_graph_with_engine(
        &self,
        graph: &GraphNetwork,
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
            return self.verify_graph_beta_crown_with_engine(graph, &input_bounds, spec, engine);
        }

        match self.propagate_graph_bounds_with_engine(
            graph,
            &input_bounds,
            spec.timeout_ms(),
            engine,
        )? {
            GraphBoundsResult::Completed {
                output_bounds,
                actual_method,
                provenance,
            } => self.check_spec(
                &output_bounds,
                spec.output_bounds(),
                Some(actual_method),
                provenance,
            ),
            GraphBoundsResult::Timeout {
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

    /// Certify all output bounds for a graph network without fabricating an
    /// output-constrained verification spec.
    pub fn certify_graph_bounds(
        &self,
        model_id: impl Into<String>,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
    ) -> Result<BoundCertificationResult> {
        self.certify_graph_bounds_with_engine(
            model_id,
            graph,
            input_bounds,
            timeout_ms,
            self.engine(),
        )
    }

    fn certify_graph_bounds_with_engine(
        &self,
        model_id: impl Into<String>,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundCertificationResult> {
        if self.config.method == PropagationMethod::BetaCrown {
            return Err(NyError::UnsupportedOp(
                "certify_graph_bounds does not support PropagationMethod::BetaCrown because \
                 the current graph beta-crown path is still property-serving"
                    .to_string(),
            ));
        }

        let model_id = model_id.into();
        match self.propagate_graph_bounds_with_engine(graph, input_bounds, timeout_ms, engine)? {
            GraphBoundsResult::Completed {
                output_bounds,
                actual_method,
                provenance,
            } => Ok(BoundCertificationResult::Certified(
                BoundCertificate::try_new(model_id, output_bounds, actual_method, provenance)?,
            )),
            GraphBoundsResult::Timeout {
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

    fn propagate_graph_bounds_with_engine(
        &self,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        timeout_ms: Option<u64>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<GraphBoundsResult> {
        let engine = self.resolve_engine(engine);
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        let past_deadline = || deadline.map(|d| Instant::now() >= d).unwrap_or(false);

        info!(
            "Starting graph verification with {:?}, {} nodes, deadline={:?}",
            self.config.method,
            graph.num_nodes(),
            deadline.map(|d| d.duration_since(Instant::now())),
        );

        let mul_binary_relaxation = self.config.mul_binary_relaxation;
        let mut actual_method = self.config.method;
        let output_bounds = match self.config.method {
            PropagationMethod::Ibp => graph.propagate_ibp(input_bounds)?,
            PropagationMethod::Crown => {
                if past_deadline() {
                    info!("Timeout before graph CROWN propagation");
                    return Ok(GraphBoundsResult::Timeout {
                        provenance: SoundnessProvenance::default(),
                        partial_bounds: None,
                        actual_method: MethodUsed::Crown,
                    });
                }
                if mul_binary_relaxation == crate::types::MulBinaryRelaxationMode::default() {
                    let alpha_config = self.alpha_crown_config(deadline);
                    match graph.propagate_alpha_crown_with_config_and_engine(
                        input_bounds,
                        &alpha_config,
                        engine,
                    ) {
                        Ok(bounds) => bounds,
                        Err(error) => {
                            if graph_crown_error_must_propagate(&error) {
                                return Err(error);
                            }
                            debug!(
                                "Optimized graph CROWN failed ({}); trying fallback chain",
                                error
                            );
                            let (bounds, method) = crown_fallback_chain(
                                graph,
                                input_bounds,
                                engine,
                                mul_binary_relaxation,
                                deadline,
                            )?;
                            actual_method = method;
                            bounds
                        }
                    }
                } else {
                    let (bounds, method) = crown_fallback_chain(
                        graph,
                        input_bounds,
                        engine,
                        mul_binary_relaxation,
                        deadline,
                    )?;
                    actual_method = method;
                    bounds
                }
            }
            PropagationMethod::AlphaCrown => {
                if past_deadline() {
                    info!("Timeout before graph α-CROWN optimization");
                    return Ok(GraphBoundsResult::Timeout {
                        provenance: SoundnessProvenance::default(),
                        partial_bounds: None,
                        actual_method: MethodUsed::AlphaCrown,
                    });
                }
                debug!("Using α-CROWN for GraphNetwork");
                let alpha_config = self.alpha_crown_config(deadline);
                match graph.propagate_alpha_crown_with_config_and_engine(
                    input_bounds,
                    &alpha_config,
                    engine,
                ) {
                    Ok(bounds) => bounds,
                    Err(error) => {
                        if graph_crown_error_must_propagate(&error) {
                            return Err(error);
                        }
                        debug!("α-CROWN failed ({}); trying CROWN fallback chain", error);
                        if past_deadline() {
                            info!("Timeout after α-CROWN failure, before CROWN fallback");
                            return Ok(GraphBoundsResult::Timeout {
                                provenance: SoundnessProvenance::default(),
                                partial_bounds: None,
                                actual_method: MethodUsed::AlphaCrown,
                            });
                        }
                        let (bounds, method) = crown_fallback_chain(
                            graph,
                            input_bounds,
                            engine,
                            mul_binary_relaxation,
                            deadline,
                        )?;
                        actual_method = method;
                        bounds
                    }
                }
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
                    "propagate_graph_bounds_with_engine does not support BetaCrown".to_string(),
                ));
            }
        };

        let output_bounds = Self::sanitize_output_bounds(output_bounds)?;
        let mut provenance =
            crate::soundness::soundness_provenance_for_graph(graph, &actual_method);
        let sqrt_negative_domain_nodes =
            crate::soundness::count_sqrt_negative_domain_graph(graph, input_bounds)?;
        if sqrt_negative_domain_nodes > 0 {
            let mut heuristics = provenance.heuristics_used().to_vec();
            heuristics.push(HeuristicUsed::SqrtNegativeDomain {
                num_nodes: sqrt_negative_domain_nodes,
            });
            provenance = SoundnessProvenance::from_heuristics(heuristics);
        }

        Ok(GraphBoundsResult::Completed {
            output_bounds,
            actual_method: actual_method.method_used(),
            provenance,
        })
    }

    fn verify_graph_beta_crown_with_engine(
        &self,
        graph: &GraphNetwork,
        input_bounds: &BoundedTensor,
        spec: &VerificationSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<VerificationResult> {
        let engine = self.resolve_engine(engine);
        let deadline = spec
            .timeout_ms()
            .map(|ms| Instant::now() + Duration::from_millis(ms));
        let past_deadline = || deadline.map(|d| Instant::now() >= d).unwrap_or(false);
        let mul_binary_relaxation = self.config.mul_binary_relaxation;

        debug!("BetaCrown via verify: computing CROWN bounds for all outputs");
        if past_deadline() {
            info!("Timeout before BetaCrown-via-verify α-CROWN");
            return Ok(VerificationResult::Timeout {
                provenance: SoundnessProvenance::default(),
                partial_bounds: None,
                actual_method: Some(MethodUsed::BetaCrown),
            });
        }
        let alpha_config = self.alpha_crown_config(deadline);
        let (output_bounds, actual_method) = match graph
            .propagate_alpha_crown_with_config_and_engine(input_bounds, &alpha_config, engine)
        {
            Ok(bounds) => (bounds, PropagationMethod::AlphaCrown),
            Err(error) => {
                if graph_crown_error_must_propagate(&error) {
                    return Err(error);
                }
                debug!("α-CROWN failed ({}); trying CROWN fallback chain", error);
                if past_deadline() {
                    info!("Timeout after α-CROWN failure in BetaCrown path");
                    return Ok(VerificationResult::Timeout {
                        provenance: SoundnessProvenance::default(),
                        partial_bounds: None,
                        actual_method: Some(MethodUsed::BetaCrown),
                    });
                }
                let (bounds, method) = crown_fallback_chain(
                    graph,
                    input_bounds,
                    engine,
                    mul_binary_relaxation,
                    deadline,
                )?;
                (bounds, method)
            }
        };

        let output_bounds = Self::sanitize_output_bounds(output_bounds)?;
        let mut provenance =
            crate::soundness::soundness_provenance_for_graph(graph, &actual_method);
        let sqrt_negative_domain_nodes =
            crate::soundness::count_sqrt_negative_domain_graph(graph, input_bounds)?;
        if sqrt_negative_domain_nodes > 0 {
            let mut heuristics = provenance.heuristics_used().to_vec();
            heuristics.push(HeuristicUsed::SqrtNegativeDomain {
                num_nodes: sqrt_negative_domain_nodes,
            });
            provenance = SoundnessProvenance::from_heuristics(heuristics);
        }

        self.check_spec(
            &output_bounds,
            spec.output_bounds(),
            Some(actual_method.method_used()),
            provenance,
        )
    }
}
