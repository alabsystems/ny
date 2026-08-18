// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds, PatchesMaterializationPurpose};
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::core::graph::backward_helpers::{
    mask_linear_bounds_columns, where_constant_mask,
};
use crate::network::core::{
    apply_dense_backward_dispatch_result_with_deadline, crown_backward_step_patches,
    CrownStepResult, NETWORK_INPUT,
};
use crate::network::tighten_crown_output_with_provenance_and_deadline;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use ndarray::{Array2, ArrayD, Ix1, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::super::core::GraphNetwork;
use super::helpers::is_softmax_decomposition_mul;
use super::spec_propagation::SpecCrownRequest;
use crate::network::CrownMergeAccumulator;

/// Outcome of the plain Graph-CROWN Patches fast path.
///
/// Deadline expiry is verifier authority: retrying the same node through the
/// Dense path after a cooperative Patches worker expires can consume the rest
/// of the outer budget (and materialize a much larger relation). Other
/// recoverable Patches failures retain the historical Dense retry.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PlainPatchesDispatchOutcome {
    AccumulateToInput,
    IbpFallback(CrownIbpFallbackReason),
    FallThroughDense,
}

/// Prepare a plain DAG-CROWN carrier for a Dense-only boundary.
///
/// The central Patches materializer distinguishes resource refusal from
/// malformed/semantic errors.  Only the former is authority to abandon this
/// CROWN walk for the established sound IBP memory fallback.  `ensure_dense`
/// is transactional, so either the conversion succeeds or the caller retains
/// the exact Patches carrier for diagnostics and policy handling.
pub(super) fn prepare_plain_dense_boundary(
    node_cb: &mut CrownBounds,
    deadline: Option<Instant>,
) -> Result<Option<CrownIbpFallbackReason>> {
    prepare_plain_dense_boundary_for_purpose(
        node_cb,
        deadline,
        PatchesMaterializationPurpose::Other,
    )
}

fn prepare_plain_dense_boundary_for_purpose(
    node_cb: &mut CrownBounds,
    deadline: Option<Instant>,
    purpose: PatchesMaterializationPurpose,
) -> Result<Option<CrownIbpFallbackReason>> {
    match node_cb.ensure_dense_with_deadline_for_purpose(deadline, purpose) {
        Ok(_) => Ok(None),
        Err(NyError::CpuMemoryExceeded { .. }) => {
            Ok(Some(CrownIbpFallbackReason::MemoryBudgetExceeded))
        }
        Err(NyError::DeadlineExceeded(_)) => {
            Ok(Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded))
        }
        Err(error) => Err(error),
    }
}

#[inline]
fn plain_resource_fallback_reason(error: &NyError) -> Option<CrownIbpFallbackReason> {
    match error {
        NyError::CpuMemoryExceeded { .. } => Some(CrownIbpFallbackReason::MemoryBudgetExceeded),
        NyError::DeadlineExceeded(_) => Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded),
        _ => None,
    }
}

#[inline]
pub(super) fn plain_dense_retry_is_authorized(error: &NyError) -> bool {
    matches!(
        error,
        NyError::UnsupportedOp(_) | NyError::UnsupportedConfiguration(_)
    )
}

const PLAIN_FORWARD_CLONE_POLL_STRIDE: usize = 4_096;

#[inline]
fn check_plain_forward_clone_deadline(
    deadline: Option<Instant>,
    phase: &'static str,
) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "GraphNetwork DAG-CROWN: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

/// Fallibly clone an already-collected output enclosure under finite verifier
/// authority.  The retained source and complete staged destination are both
/// charged before allocation; publication is transactional and cooperatively
/// polled across the endpoint copy and invariant validation.
fn clone_plain_forward_bounds_with_deadline(
    forward_bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    const SITE: &str = "GraphNetwork DAG-CROWN forward fallback clone";
    check_plain_forward_clone_deadline(deadline, "before forward fallback clone")?;

    let elements = forward_bounds.len();
    let source_bytes = elements.saturating_mul(2).saturating_mul(size_of::<f32>());
    let required_bytes = source_bytes.saturating_mul(2);
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    let allocation_error = || NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site: SITE,
    };
    let mut lower = Vec::new();
    lower
        .try_reserve_exact(elements)
        .map_err(|_| allocation_error())?;
    let mut upper = Vec::new();
    upper
        .try_reserve_exact(elements)
        .map_err(|_| allocation_error())?;
    let actual_required_bytes = source_bytes.saturating_add(
        lower
            .capacity()
            .saturating_add(upper.capacity())
            .saturating_mul(size_of::<f32>()),
    );
    if actual_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    for (index, (&lower_value, &upper_value)) in forward_bounds
        .lower()
        .iter()
        .zip(forward_bounds.upper().iter())
        .enumerate()
    {
        if index.is_multiple_of(PLAIN_FORWARD_CLONE_POLL_STRIDE) {
            check_plain_forward_clone_deadline(deadline, "while cloning forward fallback")?;
        }
        lower.push(lower_value);
        upper.push(upper_value);
    }
    check_plain_forward_clone_deadline(deadline, "after cloning forward fallback")?;

    let lower = ArrayD::from_shape_vec(IxDyn(forward_bounds.shape()), lower)
        .map_err(|error| NyError::InternalError(format!("{SITE}: lower shape: {error}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(forward_bounds.shape()), upper)
        .map_err(|error| NyError::InternalError(format!("{SITE}: upper shape: {error}")))?;
    BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
        check_plain_forward_clone_deadline(deadline, "validating forward fallback clone")
    })
}

/// Publish the already-collected output enclosure for a finite request.  The
/// no-deadline lane retains the historical whole-network IBP recomputation.
fn plain_forward_fallback(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    forward_bounds: &BoundedTensor,
    deadline: Option<Instant>,
    reason: CrownIbpFallbackReason,
) -> Result<CrownBackwardResult> {
    let bounds = if deadline.is_some() {
        clone_plain_forward_bounds_with_deadline(forward_bounds, deadline)?
    } else {
        graph.propagate_ibp(input)?
    };
    Ok(CrownBackwardResult {
        bounds,
        provenance: BoundsProvenance::ForwardFallback(reason),
    })
}

/// Complete a plain DAG-CROWN resource boundary without erasing semantic
/// failures.  Callers use this only after a fallible materialization/merge:
/// CPU budget/deadline refusal selects the established sound forward fallback;
/// every other error is returned unchanged.
pub(super) fn plain_memory_fallback_or_error(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    forward_bounds: &BoundedTensor,
    deadline: Option<Instant>,
    error: NyError,
) -> Result<CrownBackwardResult> {
    let Some(reason) = plain_resource_fallback_reason(&error) else {
        return Err(error);
    };
    plain_forward_fallback(graph, input, forward_bounds, deadline, reason)
}

#[cfg(test)]
std::thread_local! {
    static PLAIN_PATCHES_DEADLINE_CAPTURE:
        std::cell::RefCell<Option<Vec<Option<Instant>>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) fn with_plain_patches_deadline_capture<T>(
    run: impl FnOnce() -> T,
) -> (T, Vec<Option<Instant>>) {
    PLAIN_PATCHES_DEADLINE_CAPTURE.with(|capture| {
        assert!(
            capture.borrow_mut().replace(Vec::new()).is_none(),
            "plain Patches deadline capture must not be nested"
        );
    });
    let result = run();
    let captured = PLAIN_PATCHES_DEADLINE_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .take()
            .expect("plain Patches deadline capture must be installed")
    });
    (result, captured)
}

/// Run one plain Graph-CROWN Patches step and apply its site-local fallback
/// policy.
///
/// The caller supplies the already-computed node-local deadline. In
/// particular, this must not receive the outer verification deadline: the
/// per-node budget is what prevents one Patches operation from consuming the
/// entire Graph-CROWN pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_plain_patches_or_fallback(
    node_cb: &mut CrownBounds,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    node_deadline: Option<Instant>,
    node_name: &str,
    layer_type: &str,
) -> Result<PlainPatchesDispatchOutcome> {
    let debug_patches = ny_levers::read(&ny_levers::decls::diagnostics::CONV_PATCHES_DEBUG)
        .value
        .as_str()
        .is_some_and(|value| !value.is_empty() && value != "0");
    #[cfg(test)]
    PLAIN_PATCHES_DEADLINE_CAPTURE.with(|capture| {
        if let Some(deadlines) = capture.borrow_mut().as_mut() {
            deadlines.push(node_deadline);
        }
    });

    match crown_backward_step_patches(
        layer,
        node_cb,
        pre_activation,
        engine,
        0, // layer_idx not meaningful in graph
        "DAG-CROWN",
        node_deadline,
    ) {
        Ok(CrownStepResult::Continue) => {
            return Ok(PlainPatchesDispatchOutcome::AccumulateToInput);
        }
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            if debug_patches {
                eprintln!(
                    "[conv-patches-dbg] node={node_name} layer={layer_type} outcome=ibp_fallback reason={:?} details={}",
                    fallback.reason, fallback.details
                );
            }
            debug!(
                "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) \
                 requested IBP fallback: {}; keeping its typed policy",
                node_name, layer_type, fallback.details
            );
            return Ok(PlainPatchesDispatchOutcome::IbpFallback(fallback.reason));
        }
        Err(error @ NyError::CpuMemoryExceeded { .. }) => {
            if debug_patches {
                eprintln!(
                    "[conv-patches-dbg] node={node_name} layer={layer_type} outcome=memory_refusal error={error}"
                );
            }
            debug!(
                "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) hit memory budget guard: {}; using IBP",
                node_name, layer_type, error
            );
            return Ok(PlainPatchesDispatchOutcome::IbpFallback(
                CrownIbpFallbackReason::MemoryBudgetExceeded,
            ));
        }
        Err(error @ NyError::DeadlineExceeded(_)) => {
            if debug_patches {
                eprintln!(
                    "[conv-patches-dbg] node={node_name} layer={layer_type} outcome=deadline_refusal error={error}"
                );
            }
            debug!(
                "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) exhausted its \
                 node-local deadline: {}; using IBP without a Dense retry",
                node_name, layer_type, error
            );
            return Ok(PlainPatchesDispatchOutcome::IbpFallback(
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
            ));
        }
        Err(error) if plain_dense_retry_is_authorized(&error) => {
            if debug_patches {
                eprintln!(
                    "[conv-patches-dbg] node={node_name} layer={layer_type} outcome=dense_retry error={error}"
                );
            }
            debug!(
                "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) is unsupported: {}, \
                 falling back to Dense",
                node_name, layer_type, error
            );
        }
        Err(error) => return Err(error),
    }

    // Only a semantic Unsupported refusal authorizes a Dense retry. Resource
    // authorities and other semantic failures returned above are terminal for
    // this Patches transaction.
    if matches!(node_cb, CrownBounds::Patches(_)) {
        if let Some(reason) = prepare_plain_dense_boundary(node_cb, node_deadline)? {
            if debug_patches {
                eprintln!(
                    "[conv-patches-dbg] node={node_name} layer={layer_type} outcome=dense_boundary_refusal reason={reason:?}"
                );
            }
            debug!(
                "GraphNetwork DAG-CROWN: ensure_dense hit the memory budget at {}; IBP fallback",
                node_name
            );
            return Ok(PlainPatchesDispatchOutcome::IbpFallback(reason));
        }
    }
    Ok(PlainPatchesDispatchOutcome::FallThroughDense)
}

/// Extension trait for CROWN backward propagation on graph networks.
pub(crate) trait GraphNetworkCrownExt {
    fn crown_backward_with_relaxation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor>;

    /// CROWN backward propagation with explicit provenance metadata.
    ///
    /// Returns a [`CrownBackwardResult`] that indicates whether the output bounds
    /// came from actual CROWN backward propagation or were silently replaced with
    /// forward bounds due to invalid CROWN output (NaN/Inf or inverted intervals).
    fn crown_backward_with_relaxation_and_provenance(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<CrownBackwardResult>;

    /// CROWN backward propagation with deadline enforcement (#3398).
    ///
    /// A finite authority uses cooperative workers and publishes an
    /// already-collected sound forward enclosure when a live route declines.
    /// Expiry before a publishable enclosure exists remains a typed deadline
    /// error, so the method never launches fresh no-deadline fallback work.
    fn crown_backward_with_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult>;

    fn crown_backward_with_relaxation_and_deadline_and_truncation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<CrownBackwardResult>;

    /// Like [`Self::crown_backward_with_relaxation_and_deadline_and_truncation`]
    /// but reuses caller-precollected intermediate node bounds instead of
    /// running the internal Step-1 collection (#dedup-root-collections Fix B).
    ///
    /// `precollected_node_bounds` must be a valid enclosure map for the SAME
    /// input box, covering every graph node (any CROWN-IBP / forward-linear /
    /// IBP collection over `input` qualifies; an extra `NETWORK_INPUT` entry
    /// is ignored). When `Some`, the pre-collection deadline gate is also
    /// skipped — the bounds are already paid for, so falling back to vacuous
    /// IBP before even starting the backward pass would discard them. The
    /// per-node deadline checks inside the backward loop remain in force
    /// (sound IBP fallback on true budget exhaustion). Passing `None` is
    /// byte-for-byte the legacy behavior.
    ///
    /// `crown_ibp_tightening_cap`, when present, applies only if Step 1 must
    /// freshly run the per-node CROWN-IBP collector. It is ignored when
    /// `precollected_node_bounds` is present and does not shorten `deadline`.
    fn crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
        precollected_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
        crown_ibp_tightening_cap: Option<Duration>,
    ) -> Result<CrownBackwardResult>;

    fn crown_backward_specs_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor>;

    #[cfg(test)]
    fn crown_backward_specs_linear_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)>;
}

impl GraphNetworkCrownExt for GraphNetwork {
    fn crown_backward_with_relaxation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            None,
            None,
        )
        .map(|result| result.bounds)
    }

    fn crown_backward_with_relaxation_and_provenance(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            None,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            deadline,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline_and_truncation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            input,
            engine,
            mul_binary_relaxation,
            deadline,
            crown_backward_layers,
            None,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
        precollected_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
        crown_ibp_tightening_cap: Option<Duration>,
    ) -> Result<CrownBackwardResult> {
        // Disable the L2/Cauchy–Schwarz lever for the entire fixed-slope CROWN
        // backward scope (this is the single chokepoint all the public
        // `crown_backward_with_relaxation*` variants funnel through, and the
        // historical no-deadline fallbacks below call `propagate_ibp` from
        // inside it).
        // The CROWN-IBP intermediate forward passes collected here skip the
        // per-pass lever work. Sound (lever only tightens); restored on drop.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        if self.nodes.is_empty() {
            return Ok(CrownBackwardResult {
                bounds: if deadline.is_some() {
                    clone_plain_forward_bounds_with_deadline(input, deadline)?
                } else {
                    input.clone()
                },
                provenance: BoundsProvenance::Crown,
            });
        }

        // Get execution order
        let exec_order = self.exec_order()?;
        let plan = self.dispatch_plan()?;

        // Whether this graph family qualifies for CROWN-IBP intermediates —
        // pure function of the graph; also names the provenance label below.
        let use_crown_ibp = self.should_use_crown_ibp_intermediates();

        // Step 1: Bounds at each node for nonlinear relaxations.
        //
        // #dedup-root-collections Fix B: when the caller already holds a valid
        // same-box enclosure map (e.g., the DAG alpha init reference bounds —
        // previously this function re-collected the IDENTICAL map, ~73 s of
        // dead work per root episode on vggnet16_2022), reuse it and skip both
        // the internal collection and the pre-collection deadline gate (that
        // gate only protects the collection cost; discarding paid-for bounds
        // for vacuous IBP would lose the root objective). The per-node
        // deadline checks in the backward loop below remain in force.
        let collected_node_bounds;
        let node_bounds: &std::collections::HashMap<String, BoundedTensor> = if let Some(bounds) =
            precollected_node_bounds
        {
            bounds
        } else {
            // Deadline check before expensive CROWN-IBP collection (#3398).
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Err(NyError::DeadlineExceeded(
                        "GraphNetwork DAG-CROWN: deadline exceeded before forward-bound collection"
                            .into(),
                    ));
                }
            }

            // - CNN-style DAGs: use expensive CROWN-IBP intermediates for much tighter ReLU relaxations.
            // - Transformer-style graphs: use IBP forward bounds (includes transformer-specific tightening).
            let use_per_node_crown_ibp = self.should_collect_per_node_crown_ibp_intermediates();
            // Image forward-linear intermediates (#vnncomp-image-forward-linear):
            // use the same shared policy/flags as spec setup, including the
            // sequential ConvTranspose2d+Conv2d cGAN surface. The cached
            // certified pass is free after its first computation. Fail closed
            // to the existing selection on any collector refusal.
            let forward_linear_bounds = {
                // The cached forward-linear map is an optional tightening.
                // Serving it currently clones the full node-bound map before
                // publication, and that clone is not cooperatively pollable.
                // Keep finite authority on the deadline-aware CROWN-IBP/IBP
                // collection below; preserve the no-deadline cache path.
                if deadline.is_none() && self.should_collect_forward_linear_intermediate_reference()
                {
                    match self.collect_forward_linear_bounds_dag_cached(input, engine, deadline) {
                        Ok(bounds) => {
                            info!(
                                "GraphNetwork DAG-CROWN: forward-linear intermediates \
                                 (image graph, cached)"
                            );
                            Some((*bounds).clone())
                        }
                        Err(
                            error @ (NyError::UnsupportedOp(_)
                            | NyError::UnsupportedConfiguration(_)
                            | NyError::DeadlineExceeded(_)
                            | NyError::ShapeMismatch { .. }
                            | NyError::CpuMemoryExceeded { .. }),
                        ) => {
                            info!(
                                "GraphNetwork DAG-CROWN: forward-linear intermediates unavailable \
                             ({error}); falling back (fail-closed)"
                            );
                            None
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    None
                }
            };
            collected_node_bounds = if let Some(bounds) = forward_linear_bounds {
                bounds
            } else if use_per_node_crown_ibp {
                // Pass deadline to CROWN-IBP collection so the O(N²) per-node backward
                // passes respect the overall verification timeout. Without this, large
                // CNN DAGs (e.g., metaroom 6cnn_ry_49_8, 49 layers) can spend 13+
                // minutes in CROWN-IBP despite a 210s timeout. Fixed in #3397.
                match crown_ibp_tightening_cap {
                    Some(cap) => {
                        self.collect_crown_ibp_bounds_dag_with_status_deadline_and_tightening_cap(
                            input, deadline, engine, cap,
                        )?
                        .bounds
                    }
                    None => {
                        self.collect_crown_ibp_bounds_dag_with_status_and_deadline(
                            input, deadline, engine,
                        )?
                        .bounds
                    }
                }
            } else {
                if use_crown_ibp {
                    info!(
                    "GraphNetwork DAG-CROWN: {} nodes exceeds per-node CROWN-IBP threshold {}, using IBP intermediates for final backward pass",
                    self.nodes.len(),
                    crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD
                );
                }
                // Keep DAG CROWN relaxation intermediates on the scalar IBP path.
                // `collect_node_bounds_with_engine` feeds Linear IBP through the GEMM
                // engine, which accumulates in f32 instead of the scalar path's f64
                // dot products (`layers/linear/ibp.rs`). That precision loss is enough
                // to flip the short-seq talker CROWN canary from Verified to
                // Unknown(BoundsTooLoose) (#4219).
                if deadline.is_some() {
                    self.collect_node_bounds_with_engine_and_deadline(input, None, deadline)?
                } else {
                    self.collect_node_bounds(input)?
                }
            };
            &collected_node_bounds
        };

        // Determine output node and dimension
        let output_node_name = plan.name_of(plan.output_node_idx);
        debug_assert_eq!(plan.index_of(output_node_name), Some(plan.output_node_idx));

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "GraphNetwork DAG-CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Dense identity construction is O(output_dim^2) and has no fallible,
        // cooperatively-polled implementation.  Moreover, the finite shared
        // dispatcher currently declines every Dense legacy kernel.  Publish
        // the already-collected output enclosure before allocating the node
        // lookup/frontier or that seed. Finite Patches-seeded convolution
        // graphs retain their cooperative route, and no-deadline behavior
        // remains unchanged.
        let has_conv2d = plan.has_conv2d;
        let use_patches_seed = output_shape.len() == 3 && has_conv2d && self.use_patches_mode;
        if deadline.is_some() && !use_patches_seed {
            debug!(
                "GraphNetwork DAG-CROWN: finite Dense seed has no cooperative route; using collected forward bounds"
            );
            return plain_forward_fallback(
                self,
                input,
                output_bounds,
                deadline,
                CrownIbpFallbackReason::CrownPropagationError,
            );
        }

        // Pre-build node lookup vector (eliminates self.nodes HashMap lookups in hot loop).
        // Pattern from DAG alpha-CROWN backward/mod.rs:193-200.
        let nodes_by_idx: Vec<&_> = plan
            .exec_order
            .iter()
            .map(|&idx| {
                self.nodes
                    .get(plan.name_of(idx))
                    .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", idx)))
            })
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Initialize linear bounds per node
        // Each node tracks the accumulated linear bounds from all its consumers.
        // Phase 1b (#2613): Use CrownBounds to support Patches mode for CNN DAGs.
        // When the output is 3D spatial with Conv2d layers, start in Patches mode.
        // Accumulation at merge points converts to Dense via ensure_dense().
        let mut node_linear_bounds = CrownMergeAccumulator::new_indexed(exec_order);

        // Output node starts with identity bounds — Patches when spatial + Conv2d
        // and use_patches_mode is enabled. Matrix mode (use_patches_mode=false) forces
        // Dense throughout, matching the reference conv_mode='matrix' policy.
        // Reference: abcrown.py:228-231 — matrix when cuts enabled.
        // #margin-subset-seed (#margin-subset-alpha): when the initial-bounds
        // scope published the spec-referenced OUTPUT indices (single-margin
        // specs on wide heads, e.g. vggnet16 `(>= Y_200 Y_177)` over 1000
        // outputs), seed ONLY the k referenced identity rows. Each row is
        // bit-identical in semantics to its full-width counterpart
        // (row-independence: backward walk, per-row error term, and per-row
        // concretize are all row-local); the k concretized rows are scattered
        // over the output node's sound forward bounds at the exits below.
        // On vggnet16 the full-width seed materializes `[1000 x 401408]` conv
        // coefficient buffers (measured 119 GB anon-RSS, kernel-OOM) for 998
        // rows the verdict never reads; the k-row seed is ~500x smaller.
        // Unpublished scope (every caller outside the single-margin
        // initial-bounds computation) => `None` => byte-identical behavior.
        let margin_subset = if use_patches_seed {
            None
        } else {
            crate::output_margin_seed::margin_subset_indices(output_dim)
        };
        let initial_crown_bounds = if use_patches_seed {
            let (oc, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "GraphNetwork DAG-CROWN: Initializing Patches mode (output {}x{}x{})",
                oc, oh, ow
            );
            let shape = (oc, oh, ow);
            let seed =
                match PatchesLinearBounds::try_identity_with_deadline(shape, shape, deadline, 0) {
                    Ok(seed) => seed,
                    Err(error) => {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        )
                    }
                };
            CrownBounds::Patches(Box::new(seed))
        } else if let Some(indices) = margin_subset.as_deref() {
            info!(
                "GraphNetwork DAG-CROWN: margin-subset OUTPUT seed engaged (k={} of {} rows)",
                indices.len(),
                output_dim
            );
            CrownBounds::Dense(LinearBounds::identity_rows(output_dim, indices))
        } else {
            CrownBounds::Dense(LinearBounds::identity(output_dim))
        };
        node_linear_bounds.insert(output_node_name.to_string(), initial_crown_bounds);

        // Number of rows the seed carries. Every downstream use of the former
        // `output_dim` in this walk denotes the SEED ROW COUNT (zero-coefficient
        // bias blocks, frontier concretization, accumulator hints) — never the
        // output node's width — so shadow it with the row count. Full-width
        // seeds keep `seed_rows == output_dim` (byte-identical).
        let seed_rows = margin_subset.as_deref().map_or(output_dim, <[usize]>::len);
        let output_dim = seed_rows;

        let input_dim = input.len();
        let mut input_accumulated = false;

        // Step 3: Propagate backward through nodes in reverse order.
        // Phase 1b (#2613): CrownBounds-aware dispatch. Single-input nodes in Patches
        // mode use crown_backward_step_patches for Conv2d/BN/activation/pool Patches
        // dispatch. Compatible same-shape Add/Sub residuals duplicate the Patches
        // carrier across their branches; other multi-input or Patches-unsupported
        // layers convert to Dense. Compatible Patches contributions merge natively,
        // while mixed or incompatible carriers promote transactionally to Dense.

        // Per-node deadline budgeting (#3795): give each backward step a fraction of
        // the remaining budget instead of the full global deadline. Without this, a
        // single large Conv2d backward can consume the entire timeout, leaving zero
        // time for BaB domain exploration.
        //
        // Budget policy (matches crown_tighten.rs constants):
        //   per_node = max(remaining / nodes_remaining, remaining * 0.25)
        //   minimum floor = 2.0s (below this, bail to IBP immediately)
        const OUTPUT_CROWN_MAX_BUDGET_FRACTION: f64 = 0.25;
        const OUTPUT_CROWN_MIN_NODE_BUDGET_SECS: f64 = 2.0;
        let total_backward_nodes = plan.node_count();
        let mut backward_steps = 0usize;

        // #iter0-alpha-parity (dark, NY_ITER0_PARITY_TRACE=1, print-only):
        // claim one walk id per backward walk so this baseline fold's per-node
        // lines separate from the α fold's in an interleaved log.
        let parity_trace = crate::iter0_parity_trace::iter0_parity_trace_enabled()
            .then(crate::iter0_parity_trace::next_walk_id);
        // #patches-drop (dark, NY_PATCHES_CARRIER_TRACE=1, print-only): publish
        // this walk's position so a `[patches-drop]` line emitted deep inside
        // the materializer names the node whose carrier densified.
        let carrier_trace = crate::patches_carrier_trace::enabled();

        for (rev_pos, &idx) in plan.reverse_order.iter().enumerate() {
            let node_name = plan.name_of(idx);
            // Deadline enforcement: check before each node's backward pass (#3398).
            // For large models (e.g., relusplitter with 1094s+ overruns), a single
            // backward pass can exceed the entire verification budget. Checking at
            // each node gives O(node_count) granularity. Falls back to IBP which
            // is always sound (just looser). Matches spec_propagation.rs:188-194.
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "GraphNetwork DAG-CROWN: deadline exceeded at node '{}', falling back to IBP",
                        node_name
                    );
                    return plain_forward_fallback(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        CrownIbpFallbackReason::DeadlineExceeded,
                    );
                }
            }

            // Compute per-node deadline for this backward step (#3795).
            let node_deadline = super::backward_node_dispatch::compute_node_deadline(
                deadline,
                rev_pos,
                total_backward_nodes,
                OUTPUT_CROWN_MAX_BUDGET_FRACTION,
                OUTPUT_CROWN_MIN_NODE_BUDGET_SECS,
            );

            // If the overall deadline expires during budget calculation, bail to IBP for
            // the remaining backward pass. Sub-floor node shares keep the global deadline
            // so CROWN LinearBounds are preserved on short-budget tiny graphs (#3881).
            if deadline.is_some() && node_deadline.is_none() {
                info!(
                    "GraphNetwork DAG-CROWN: deadline expired while budgeting '{}' ({}/{} nodes), falling back to IBP",
                    node_name,
                    rev_pos + 1,
                    total_backward_nodes,
                );
                return plain_forward_fallback(
                    self,
                    input,
                    output_bounds,
                    deadline,
                    CrownIbpFallbackReason::DeadlineExceeded,
                );
            }

            if crown_backward_layers.is_some_and(|max_layers| backward_steps >= max_layers) {
                info!(
                    "GraphNetwork DAG-CROWN: truncating backward after {} nodes at frontier '{}'",
                    backward_steps, node_name
                );
                // A finite truncation currently has to inventory and snapshot
                // the entire fan-out frontier. That inventory still performs
                // unbounded map/count/key work before the pollable carrier
                // materializers run. Preserve hard authority by returning the
                // already-collected forward enclosure instead; no frontier is
                // drained or partially published. The legacy truncation
                // concretizer remains exact for no-deadline requests.
                if deadline.is_some() {
                    return plain_forward_fallback(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        CrownIbpFallbackReason::CrownPropagationError,
                    );
                }
                let final_bounds = match self
                    .concretize_crown_frontier_to_network_input_with_deadline(
                        &mut node_linear_bounds,
                        node_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        node_deadline,
                    ) {
                    Ok(bounds) => bounds,
                    Err(error) => {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        )
                    }
                };
                let crown_output =
                    match final_bounds.concretize_sound_with_deadline(input, node_deadline) {
                        Ok(bounds) => bounds,
                        Err(error) => {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            )
                        }
                    };
                // #margin-subset-seed: scatter the k computed rows over the
                // output node's sound forward bounds (full-width no-op).
                let crown_output = match margin_subset.as_deref() {
                    Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                        output_bounds,
                        indices,
                        &crown_output,
                    )?,
                    None => crown_output,
                };
                let crown_output = match crown_output.into_reshape_with_poll(&output_shape, || {
                    if node_deadline.is_some_and(|limit| Instant::now() >= limit) {
                        Err(NyError::DeadlineExceeded(
                            "GraphNetwork DAG-CROWN: node deadline exceeded during truncated reshape"
                                .into(),
                        ))
                    } else {
                        Ok(())
                    }
                }) {
                    Ok(bounds) => bounds,
                    Err(error) => {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        )
                    }
                };
                let label = if use_crown_ibp {
                    "GraphNetwork DAG-CROWN (CROWN-IBP)"
                } else {
                    "GraphNetwork DAG-CROWN"
                };
                let (tightened, provenance) =
                    match tighten_crown_output_with_provenance_and_deadline(
                        crown_output,
                        output_bounds,
                        label,
                        node_deadline,
                    ) {
                        Ok(result) => result,
                        Err(error) => {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            )
                        }
                    };
                if node_deadline.is_some_and(|limit| Instant::now() >= limit) {
                    return plain_memory_fallback_or_error(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        NyError::DeadlineExceeded(
                            "GraphNetwork DAG-CROWN: node deadline exceeded before truncated publication"
                                .into(),
                        ),
                    );
                }
                return Ok(CrownBackwardResult {
                    bounds: tightened,
                    provenance,
                });
            }

            // Direct Vec-indexed node lookup (#4296) — no HashMap access in hot loop.
            let node = nodes_by_idx[idx];

            // Get this node's accumulated CrownBounds via direct index (#4296).
            // We can move it out because reverse-topological traversal guarantees
            // all consumers have already contributed their bounds.
            let mut node_cb = match node_linear_bounds.take_by_idx_with_deadline(idx, node_deadline)
            {
                Ok(Some(cb)) => cb,
                Ok(None) => {
                    debug!(
                        "GraphNetwork DAG-CROWN: node {} has no consumers, skipping",
                        node_name
                    );
                    continue;
                }
                Err(error) => {
                    return plain_memory_fallback_or_error(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        error,
                    )
                }
            };
            backward_steps += 1;
            if let Some(walk) = parity_trace {
                crate::iter0_parity_trace::trace_node(
                    walk,
                    "graph-crown",
                    node_name,
                    node.layer.layer_type(),
                    &node_cb,
                );
            }
            if carrier_trace {
                crate::patches_carrier_trace::enter_node("graph-crown", node_name);
            }

            // Use the plan's first-input route for shared pre-activation logic.
            let first_input_idx = plan.first_input_idx(idx);
            let first_input = plan.name_of(first_input_idx);
            let pre_activation = if plan.is_network_input(first_input_idx) {
                input
            } else {
                node_bounds.get(first_input).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        first_input
                    ))
                })?
            };

            // #3813: Dense→Patches re-entry at unary Conv2d boundaries.
            super::backward_node_dispatch::try_patches_reentry(
                &mut node_cb,
                node,
                node_bounds,
                node_name,
                self.use_patches_mode,
                "GraphNetwork DAG-CROWN",
                node_deadline,
            );

            let is_patches = matches!(&node_cb, CrownBounds::Patches(_));
            // Provenance, not the mere presence of a verifier deadline, opts
            // the eventual Dense dispatch into the strict structured-boundary
            // contract. This becomes true only when this node's live Patches
            // flow actually falls through to a Dense carrier.
            let mut finite_structured_boundary = false;
            debug!(
                "GraphNetwork DAG-CROWN: backward through {} ({}) [{}]",
                node_name,
                node.layer.layer_type(),
                if is_patches { "Patches" } else { "Dense" }
            );

            // === Phase 1b Patches fast-path (#2613) ===
            // For single-input nodes in Patches mode, use the sequential Patches-aware
            // dispatch. This handles Conv2d, BatchNorm, activations (30 types), AvgPool,
            // MaxPool natively in Patches, and terminates to Dense at Linear/Flatten/Reshape.
            // Compatible same-shape Add/Sub nodes bypass this unary arm and use the
            // residual passthrough below, preserving and duplicating their Patches
            // carrier. Other multi-input layers (MulBinary, Where, etc.) densify.
            // Compatible Patches branch contributions can merge natively.
            if is_patches && node.inputs.len() == 1 {
                match dispatch_plain_patches_or_fallback(
                    &mut node_cb,
                    &node.layer,
                    pre_activation,
                    engine,
                    node_deadline,
                    node_name,
                    node.layer.layer_type(),
                )? {
                    PlainPatchesDispatchOutcome::AccumulateToInput => {
                        if let Err(error) = self.accumulate_crown_bounds_to_input_with_deadline(
                            first_input,
                            node_cb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                        continue;
                    }
                    PlainPatchesDispatchOutcome::IbpFallback(reason) => {
                        return plain_forward_fallback(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            reason,
                        );
                    }
                    PlainPatchesDispatchOutcome::FallThroughDense => {
                        finite_structured_boundary = node_deadline.is_some();
                    }
                }
            }

            // === Patches residual passthrough for Add/Sub (#4382) ===
            match crate::network::core::graph::backward_helpers::try_apply_patches_residual_passthrough_with_deadline(
                self,
                node,
                &mut node_cb,
                node_bounds,
                &mut node_linear_bounds,
                output_dim,
                input_dim,
                &mut input_accumulated,
                "DAG-CROWN",
                node_deadline,
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    if let Some(reason) = plain_resource_fallback_reason(&error) {
                        debug!(
                            "GraphNetwork DAG-CROWN: Patches residual merge for {} ({}) hit \
                             the memory budget: {}; using IBP",
                            node_name,
                            node.layer.layer_type(),
                            error
                        );
                        return plain_forward_fallback(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            reason,
                        );
                    }
                    return Err(error);
                }
            }

            // === Dense dispatch ===
            // Convert CrownBounds to LinearBounds for existing dispatch logic.
            // For multi-input nodes or after Patches fallback, this is the main path.
            let materializes_patches = matches!(&node_cb, CrownBounds::Patches(_));
            if let Some(reason) = prepare_plain_dense_boundary(&mut node_cb, node_deadline)? {
                debug!(
                    "GraphNetwork DAG-CROWN: Dense boundary for {} ({}) hit the memory \
                     budget; using IBP",
                    node_name,
                    node.layer.layer_type()
                );
                return plain_forward_fallback(self, input, output_bounds, deadline, reason);
            }
            finite_structured_boundary |= materializes_patches && node_deadline.is_some();
            let CrownBounds::Dense(node_lb) = node_cb else {
                unreachable!("successful Dense-boundary preparation must publish Dense")
            };

            // These site-specific routes bypass the canonical dispatcher and
            // still clone/scan a complete Dense carrier without cooperative
            // receipts. Decline only when this carrier actually crossed a
            // finite Patches boundary. An ordinary deadline-bearing Dense
            // carrier keeps the historical route.
            if finite_structured_boundary
                && matches!(
                    &node.layer,
                    Layer::ReLU(_) | Layer::MulBinary(_) | Layer::Where(_)
                )
            {
                debug!(
                    "GraphNetwork DAG-CROWN: {} '{}' declines its legacy Dense route after a finite Patches boundary; using collected forward bounds",
                    node.layer.layer_type(),
                    node_name,
                );
                return plain_forward_fallback(
                    self,
                    input,
                    output_bounds,
                    deadline,
                    CrownIbpFallbackReason::CrownPropagationError,
                );
            }

            // Handle site-specific layers first (MulBinary, Where, Linear dimension
            // check), then route all other layers through the shared dispatch core
            // (#1949 Step B). This eliminates ~400 LOC of duplicated match arms.

            // === Linear: pre-dispatch dimension check with IBP fallback (#2817) ===
            if super::backward_node_dispatch::linear_dimension_mismatch(node, &node_lb) {
                return plain_forward_fallback(
                    self,
                    input,
                    output_bounds,
                    deadline,
                    CrownIbpFallbackReason::ShapeMismatch,
                );
            }

            // === ReLU: heuristic relaxation via shared dispatch (#3935) ===
            if matches!(&node.layer, Layer::ReLU(_)) {
                use super::backward_node_dispatch::{dispatch_relu_backward, NodeDispatchResult};
                match dispatch_relu_backward(
                    node,
                    &node_lb,
                    pre_activation,
                    node_name,
                    "GraphNetwork DAG-CROWN",
                    None,
                    None,
                )? {
                    NodeDispatchResult::SingleDense(bounds) => {
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            first_input,
                            *bounds,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                    }
                    NodeDispatchResult::IbpFallback(reason) => {
                        return plain_forward_fallback(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            reason,
                        );
                    }
                }
                continue;
            }

            // === MulBinary: site-specific (relaxation mode, softmax decomposition, IBP fallback) ===
            if matches!(&node.layer, Layer::MulBinary(_)) {
                use super::backward_node_dispatch::{
                    dispatch_mul_binary_backward, MulBinaryDispatchCtx, MulBinaryDispatchResult,
                };

                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = self.bounds_ref(input_a_name, input, node_bounds)?;
                let input_b_bounds = self.bounds_ref(input_b_name, input, node_bounds)?;

                let dispatch_ctx = MulBinaryDispatchCtx {
                    node,
                    node_name,
                    node_lb: &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    mul_binary_relaxation,
                    mul_binary_alpha: None,
                    softmax_decomposition: is_softmax_decomposition_mul(self, node),
                    label: "GraphNetwork DAG-CROWN",
                };
                match dispatch_mul_binary_backward(&dispatch_ctx)? {
                    MulBinaryDispatchResult::BinaryDense {
                        bounds_a,
                        bounds_b,
                        bias_lower,
                        bias_upper,
                    } => {
                        if let Err(error) =
                            Self::accumulate_bias_to_network_input_crown_with_deadline(
                                &bias_lower,
                                &bias_upper,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                                node_deadline,
                            )
                        {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            input_a_name,
                            *bounds_a,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            input_b_name,
                            *bounds_b,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                    }
                    MulBinaryDispatchResult::SoftmaxNonFinite => {
                        return plain_forward_fallback(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            CrownIbpFallbackReason::CrownPropagationError,
                        );
                    }
                    MulBinaryDispatchResult::RecoverableError(err) => {
                        debug!(
                            "GraphNetwork DAG-CROWN: MulBinary '{}' {:?} failed ({}), falling back to IBP",
                            node_name, mul_binary_relaxation, err,
                        );
                        return plain_forward_fallback(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            CrownIbpFallbackReason::CrownPropagationError,
                        );
                    }
                }
                continue;
            }

            // === Div: site-specific (positive-denominator reciprocal scaling) ===
            if matches!(&node.layer, Layer::Div(_)) {
                use super::backward_node_dispatch::{
                    backward_div_to_numerator_with_deadline, DivBackwardResult,
                };

                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = self.bounds_ref(input_a_name, input, node_bounds)?;
                let input_b_bounds = self.bounds_ref(input_b_name, input, node_bounds)?;
                let node_output_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Div output bounds for {} not found during DAG-CROWN",
                        node_name
                    ))
                })?;

                let div_result = match backward_div_to_numerator_with_deadline(
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_output_bounds,
                    node_deadline,
                ) {
                    Ok(result) => result,
                    Err(error) => {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        )
                    }
                };
                match div_result {
                    DivBackwardResult::PropagateNumerator(bounds) => {
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            input_a_name,
                            *bounds,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                    }
                    DivBackwardResult::ConcretizeCurrentNode(bias) => {
                        if let Err(error) =
                            Self::accumulate_bias_to_network_input_crown_with_deadline(
                                &bias.lower,
                                &bias.upper,
                                &mut node_linear_bounds,
                                output_dim,
                                input_dim,
                                &mut input_accumulated,
                                node_deadline,
                            )
                        {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                    }
                }
                continue;
            }

            // === Where: site-specific (ternary conditional with concretization) ===
            if let Layer::Where(where_layer) = &node.layer {
                let where_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Where output bounds for {} not found during DAG-CROWN",
                        node_name
                    ))
                })?;

                // === Embedded-constant Where (single `cond` input; both branches
                // are constants). The output is a constant vector w.r.t. the network
                // input — no linear dependence on `cond` — so the EXACT CROWN backward
                // folds the entire output into the bias and routes zero to `cond`.
                // `embedded_constant_select_output` returns the exact per-element
                // select when `cond` is constant (tighter than IBP) and the sound
                // IBP union otherwise. require_ternary_inputs would error here because
                // the node has only 1 input.
                if where_layer.has_embedded_constants() {
                    let cond_input = node.require_unary_input().map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "Where node {} with embedded constants requires 1 input (condition)",
                            node_name
                        ))
                    })?;
                    let cond_bounds = self.bounds_ref(cond_input, input, node_bounds)?;
                    let select = where_layer.embedded_constant_select_output(cond_bounds)?;
                    let concrete =
                        match node_lb.concretize_sound_with_deadline(&select, node_deadline) {
                            Ok(bounds) => bounds,
                            Err(error) => {
                                return plain_memory_fallback_or_error(
                                    self,
                                    input,
                                    output_bounds,
                                    deadline,
                                    error,
                                )
                            }
                        };
                    let (lower_b, upper_b) = concrete.into_parts();
                    let lower_b = lower_b.into_dimensionality::<Ix1>().map_err(|error| {
                        NyError::InternalError(format!(
                            "graph-crown embedded-constant Where lower result was not 1D: {error}"
                        ))
                    })?;
                    let upper_b = upper_b.into_dimensionality::<Ix1>().map_err(|error| {
                        NyError::InternalError(format!(
                            "graph-crown embedded-constant Where upper result was not 1D: {error}"
                        ))
                    })?;
                    if let Err(error) = Self::accumulate_bias_to_network_input_crown_with_deadline(
                        &lower_b,
                        &upper_b,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        node_deadline,
                    ) {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        );
                    }
                    continue;
                }

                let (cond_input, true_input, false_input) = node.require_ternary_inputs()?;
                let cond_bounds = self.bounds_ref(cond_input, input, node_bounds)?;
                let cond_all_true = cond_bounds.lower().iter().all(|&v| v >= 0.5);
                let cond_all_false = cond_bounds.upper().iter().all(|&v| v <= 0.5);

                if cond_all_true {
                    if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                        true_input,
                        node_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        node_deadline,
                    ) {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        );
                    }
                    continue;
                } else if cond_all_false {
                    if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                        false_input,
                        node_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        node_deadline,
                    ) {
                        return plain_memory_fallback_or_error(
                            self,
                            input,
                            output_bounds,
                            deadline,
                            error,
                        );
                    }
                    continue;
                }

                // === Exact per-element select for a bound-independent (constant)
                // condition mask (#Where-const-cond). When the condition is fixed
                // (lower == upper elementwise), Where degenerates to a fixed 0/1
                // mask: output[i] = true_input[i] if mask[i] else false_input[i].
                // This is an EXACT linear transform — route each output column to
                // the correct branch by zeroing the other branch's columns. The
                // generic mixed fallback below would instead concretize the whole
                // tensor (loose IBP), so we prefer this exact split.
                if let Some(mask) = where_constant_mask(cond_bounds) {
                    debug_assert_eq!(mask.len(), node_lb.num_inputs());
                    if mask.len() == node_lb.num_inputs() {
                        let true_lb = mask_linear_bounds_columns(&node_lb, &mask, true);
                        let false_lb = mask_linear_bounds_columns(&node_lb, &mask, false);
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            true_input,
                            true_lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                        if let Err(error) = self.accumulate_dense_bounds_to_input_with_deadline(
                            false_input,
                            false_lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ) {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            );
                        }
                        continue;
                    }
                }

                let concrete =
                    match node_lb.concretize_sound_with_deadline(where_bounds, node_deadline) {
                        Ok(bounds) => bounds,
                        Err(error) => {
                            return plain_memory_fallback_or_error(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                error,
                            )
                        }
                    };
                let (lower_b, upper_b) = concrete.into_parts();
                let lower_b = lower_b.into_dimensionality::<Ix1>().map_err(|error| {
                    NyError::InternalError(format!(
                        "graph-crown Where mixed lower result was not 1D: {error}"
                    ))
                })?;
                let upper_b = upper_b.into_dimensionality::<Ix1>().map_err(|error| {
                    NyError::InternalError(format!(
                        "graph-crown Where mixed upper result was not 1D: {error}"
                    ))
                })?;

                if let Err(error) = Self::accumulate_bias_to_network_input_crown_with_deadline(
                    &lower_b,
                    &upper_b,
                    &mut node_linear_bounds,
                    output_dim,
                    input_dim,
                    &mut input_accumulated,
                    node_deadline,
                ) {
                    return plain_memory_fallback_or_error(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        error,
                    );
                }
                continue;
            }

            // === All other layers: shared dispatch core (#1949 Step B, #3935) ===
            use super::backward_node_dispatch::{
                dispatch_shared_core, SharedDispatchCtx, SharedDispatchResult,
            };
            let shared_ctx = SharedDispatchCtx {
                node,
                node_name,
                node_lb: &node_lb,
                pre_activation,
                network_input: input,
                node_bounds,
                engine,
                node_deadline,
                finite_structured_boundary,
                mul_binary_relaxation,
                label: "GraphNetwork DAG-CROWN",
            };
            let shared_result = match dispatch_shared_core(&shared_ctx) {
                Ok(result) => result,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    debug!(
                        "GraphNetwork DAG-CROWN: dense dispatch for {} ({}) hit memory budget guard: {}; using IBP",
                        node_name,
                        node.layer.layer_type(),
                        error
                    );
                    return plain_forward_fallback(
                        self,
                        input,
                        output_bounds,
                        deadline,
                        CrownIbpFallbackReason::MemoryBudgetExceeded,
                    );
                }
                Err(error) => return Err(error),
            };
            match shared_result {
                SharedDispatchResult::Dispatch(result) => {
                    if let Err(error) = apply_dense_backward_dispatch_result_with_deadline(
                        self,
                        node,
                        first_input,
                        &node_lb,
                        *result,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        "Dispatch",
                        node_deadline,
                    ) {
                        if let Some(reason) = plain_resource_fallback_reason(&error) {
                            debug!(
                                "GraphNetwork DAG-CROWN: dispatch merge for {} ({}) hit the \
                                 memory budget: {}; using IBP",
                                node_name,
                                node.layer.layer_type(),
                                error
                            );
                            return plain_forward_fallback(
                                self,
                                input,
                                output_bounds,
                                deadline,
                                reason,
                            );
                        }
                        return Err(error);
                    }
                }
                SharedDispatchResult::IbpFallback(reason) => {
                    return plain_forward_fallback(self, input, output_bounds, deadline, reason);
                }
            }
        }

        // Step 4: Concretize final bounds.
        // Convert CrownBounds to Dense for concretization.
        let mut final_cb = match node_linear_bounds.take_with_deadline(NETWORK_INPUT, deadline) {
            Ok(Some(bounds)) => bounds,
            Ok(None) => {
                return Err(NyError::InvalidSpec(
                    "No path to network input found".to_string(),
                ));
            }
            Err(error) => {
                return plain_memory_fallback_or_error(self, input, output_bounds, deadline, error)
            }
        };
        if let Some(reason) = prepare_plain_dense_boundary_for_purpose(
            &mut final_cb,
            deadline,
            PatchesMaterializationPurpose::NetworkInputTerminal,
        )? {
            debug!(
                "GraphNetwork DAG-CROWN: final Patches materialization hit its resource authority; \
                 using IBP"
            );
            return plain_forward_fallback(self, input, output_bounds, deadline, reason);
        }
        let CrownBounds::Dense(final_bounds) = final_cb else {
            unreachable!("successful final-bound preparation must publish Dense")
        };

        debug!(
            "GraphNetwork DAG-CROWN: Concretizing {} outputs from {} inputs",
            final_bounds.num_outputs(),
            final_bounds.num_inputs()
        );
        let crown_output = match final_bounds.concretize_sound_with_deadline(input, deadline) {
            Ok(bounds) => bounds,
            Err(error) => {
                return plain_memory_fallback_or_error(self, input, output_bounds, deadline, error)
            }
        };
        // #margin-subset-seed: scatter the k computed rows over the output
        // node's sound forward bounds (full-width no-op). Every row of the
        // scattered result is a valid enclosure; the tighten below intersects
        // with the forward bounds exactly as for a full-width map.
        let crown_output = match margin_subset.as_deref() {
            Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                output_bounds,
                indices,
                &crown_output,
            )?,
            None => crown_output,
        };
        let crown_output = match crown_output.into_reshape_with_poll(&output_shape, || {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                Err(NyError::DeadlineExceeded(
                    "GraphNetwork DAG-CROWN: deadline exceeded during final reshape".into(),
                ))
            } else {
                Ok(())
            }
        }) {
            Ok(bounds) => bounds,
            Err(error) => {
                return plain_memory_fallback_or_error(self, input, output_bounds, deadline, error)
            }
        };

        // Post-concretization tightening with provenance — shared with all CROWN paths (#3043).
        let label = if use_crown_ibp {
            "GraphNetwork DAG-CROWN (CROWN-IBP)"
        } else {
            "GraphNetwork DAG-CROWN"
        };
        let (tightened, provenance) = match tighten_crown_output_with_provenance_and_deadline(
            crown_output,
            output_bounds,
            label,
            deadline,
        ) {
            Ok(result) => result,
            Err(error) => {
                return plain_memory_fallback_or_error(self, input, output_bounds, deadline, error)
            }
        };

        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return plain_memory_fallback_or_error(
                self,
                input,
                output_bounds,
                deadline,
                NyError::DeadlineExceeded(
                    "GraphNetwork DAG-CROWN: deadline exceeded before final publication".into(),
                ),
            );
        }

        Ok(CrownBackwardResult {
            bounds: tightened,
            provenance,
        })
    }

    fn crown_backward_specs_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_relaxation(mul_binary_relaxation)
            .run()
    }

    #[cfg(test)]
    fn crown_backward_specs_linear_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_relaxation(mul_binary_relaxation)
            .run_with_linear()
    }
}
