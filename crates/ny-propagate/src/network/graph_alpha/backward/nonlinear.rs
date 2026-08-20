// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState, LinearBounds};
use crate::invprop::InvpropState;
use crate::layers::{Layer, ReLULayer};
use crate::network::core::{crown_backward_step_patches, CrownStepResult, GraphNetwork, GraphNode};
use crate::network::CrownMergeAccumulator;

use ndarray::Array1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

/// #dag-alpha-patches-expiry: does finite authority refuse the alpha-specific
/// Patches ReLU kernels?
///
/// Graph-lane twin of `patches_step::hard_finite_authority_refuses_patches`
/// (`4d0257ba9`), sharing its lever so the two lanes cannot drift apart.
///
/// * lever OFF (default): returns `deadline.is_some()`, i.e. deadline PRESENCE
///   refuses. Byte-identical to the historical `deadline.is_none()` guard.
/// * lever ON: returns true only on actual EXPIRY, so a scored run with budget
///   remaining keeps its optimized alpha instead of silently reverting to the
///   heuristic slope.
///
/// The trade the original guard was protecting is real and is NOT resolved by
/// arming this: those kernels still do not poll inside allocation/flatten/
/// compose, so an armed run can overrun its deadline inside one. NOTE
/// (2026-08-19): the shared predicate now SHIPS ARMED by default (`=0` kill
/// switch), so this comment records the residual RISK, not the default. The
/// durable fix is to thread the deadline through those phases; this exists to
/// price whether that work is worth doing before doing it.
fn alpha_patches_finite_authority_refuses(deadline: Option<Instant>) -> bool {
    // Delegates to the ONE shared predicate. `deadline_is_hard` is deadline
    // presence at this site, which is what the historical guard tested.
    let refuses = crate::network::core::graph::backward_helpers::patches_finite_authority_refuses(
        deadline.is_some(),
        deadline,
    );
    if deadline.is_some()
        && crate::network::core::sequential::crown::patches_step::expiry_authority_armed()
    {
        // RULE 7 instrumentation, latched once per process. Arming
        // NY_PATCHES_FINITE_EXPIRY for cifar100 previously produced a null whose
        // telemetry was EMPTY IN BOTH ARMS — the sequential-lane lever simply
        // never reached a DAG resnet. A lever with no engagement signal cannot
        // be distinguished from an inert one, and that null cost a full A/B.
        // This line is the difference between "measured negative" and "never ran".
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            eprintln!(
                "[dag-alpha-patches-expiry] ARMED on the graph lane: alpha-specific \
                 Patches ReLU kernels now refuse on EXPIRY, not on deadline presence"
            );
        });
    }
    refuses
}
use tracing::debug;

const INTERMEDIATE_COPY_CHECK_STRIDE: usize = 4096;

fn intermediate_copy_checkpoint(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "DAG alpha-CROWN ReLU intermediate deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

/// Fallibly copy a borrowed pre-activation pair into its flat intermediate
/// representation. The receipt covers the caller's retained payload, the live
/// borrowed pair, both output vectors, and allocator capacity overage.
fn copy_pre_activation_for_intermediate(
    pre_activation: &BoundedTensor,
    retained_base_bytes: usize,
    deadline: Option<Instant>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    const SITE: &str = "graph-alpha ReLU intermediate pre-activation copy";
    intermediate_copy_checkpoint(deadline, "before pre-activation admission")?;

    let len = pre_activation.len();
    let pair_bytes = len.saturating_mul(2).saturating_mul(size_of::<f32>());
    let nominal_required_bytes = retained_base_bytes
        .saturating_add(pair_bytes) // borrowed source pair
        .saturating_add(pair_bytes); // owned flattened pair
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if nominal_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: nominal_required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    let allocation_error = |required_bytes| NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site: SITE,
    };
    let mut lower = Vec::new();
    lower
        .try_reserve_exact(len)
        .map_err(|_| allocation_error(nominal_required_bytes))?;
    let lower_overage = lower
        .capacity()
        .saturating_sub(len)
        .saturating_mul(size_of::<f32>());
    let after_lower_reserve = nominal_required_bytes.saturating_add(lower_overage);
    if after_lower_reserve > budget_bytes {
        return Err(allocation_error(after_lower_reserve));
    }

    let mut upper = Vec::new();
    upper
        .try_reserve_exact(len)
        .map_err(|_| allocation_error(after_lower_reserve))?;
    let upper_overage = upper
        .capacity()
        .saturating_sub(len)
        .saturating_mul(size_of::<f32>());
    let actual_required_bytes = after_lower_reserve.saturating_add(upper_overage);
    if actual_required_bytes > budget_bytes {
        return Err(allocation_error(actual_required_bytes));
    }
    intermediate_copy_checkpoint(deadline, "after pre-activation allocation")?;

    for (index, value) in pre_activation.lower().iter().copied().enumerate() {
        if index.is_multiple_of(INTERMEDIATE_COPY_CHECK_STRIDE) {
            intermediate_copy_checkpoint(deadline, "during lower pre-activation copy")?;
        }
        lower.push(value);
    }
    for (index, value) in pre_activation.upper().iter().copied().enumerate() {
        if index.is_multiple_of(INTERMEDIATE_COPY_CHECK_STRIDE) {
            intermediate_copy_checkpoint(deadline, "during upper pre-activation copy")?;
        }
        upper.push(value);
    }
    intermediate_copy_checkpoint(deadline, "before pre-activation publication")?;
    Ok((Array1::from_vec(lower), Array1::from_vec(upper)))
}

// `ReturnBounds` keeps `BoundedTensor` inline: the value is constructed and
// consumed within one backward step, so a 232-byte move beats a heap
// allocation per nonlinear node on this hot path.
#[allow(clippy::large_enum_variant)]
pub(super) enum NonlinearNodeResult {
    /// Boxed: `CrownBounds` dwarfs the other variants (clippy `large_enum_variant`).
    NotHandled(Box<CrownBounds>),
    Continue,
    ReturnBounds(BoundedTensor),
}

/// Which spec rows this backward walk's carrier rows denote
/// (#spec-axis-alpha, design §2b).
///
/// Only the output-seeded DAG-α walk may consult per-spec δ, and its carrier
/// row `j` names an ORIGINAL output row only through the seed's subset map
/// (`margin_subset_indices`): a k-row subset seed makes `j` a compact carrier
/// index, not a spec id. The reference-refresh walk never constructs
/// [`DagAlphaNodeContext`], so it is isolated structurally — but provenance
/// is still encoded explicitly here rather than inferred, per the external
/// review: a bool would go ambiguous the moment chunked seeds land.
#[derive(Debug, Clone)]
pub(crate) enum AlphaRowScope {
    /// Rows are not output specs (or δ must not apply); read shared α only.
    Shared,
    /// Rows ARE output specs. `subset[j]` maps carrier row `j` to its
    /// original output row; `None` means the identity seed (row `j` IS spec
    /// row `j`).
    OutputSpecs {
        subset: Option<std::sync::Arc<[usize]>>,
    },
}

impl AlphaRowScope {
    /// Original spec row denoted by carrier row `j`, when rows are specs.
    pub(crate) fn spec_row(&self, carrier_row: usize) -> Option<usize> {
        match self {
            Self::Shared => None,
            Self::OutputSpecs { subset: None } => Some(carrier_row),
            Self::OutputSpecs {
                subset: Some(indices),
            } => indices.get(carrier_row).copied(),
        }
    }
}

pub(super) struct DagAlphaNodeContext<'a> {
    pub(super) input: &'a BoundedTensor,
    pub(super) relu_name_to_idx: &'a HashMap<String, usize>,
    pub(super) alpha_state: &'a GraphAlphaState,
    pub(super) invprop_state: Option<&'a InvpropState>,
    pub(super) gradients: &'a mut [Array1<f32>],
    pub(super) gradients_upper: &'a mut [Array1<f32>],
    pub(super) track_gradients: bool,
    pub(super) node_crown_bounds: &'a mut CrownMergeAccumulator,
    pub(super) intermediate: Option<&'a mut GraphAlphaCrownIntermediate>,
    pub(super) output_dim: usize,
    pub(super) input_dim: usize,
    pub(super) input_accumulated: &'a mut bool,
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) deadline: Option<Instant>,
    /// Whether this node's Dense continuation originated as a structured
    /// Patches carrier under finite authority. A plain Dense seed with a
    /// deadline retains the historical entry/post-checked alpha route; only
    /// an actual finite Patches-to-Dense boundary closes the legacy nonlinear
    /// kernels before work.
    pub(super) finite_structured_boundary: bool,
    /// The compact exact-seed caller must never degrade into a raw-identity
    /// full-output CROWN walk: that would defeat its bounded-allocation/OOM
    /// contract before the missing intermediates could be rejected.
    pub(super) forbid_full_output_fallback: bool,
    /// Row provenance for per-spec δ (#spec-axis-alpha).
    pub(super) row_scope: &'a AlphaRowScope,
}

pub(super) fn retry_monotone_shape_mismatch_with_fixed_slope<FAlpha, FFixed>(
    node_name: &str,
    layer_type: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
    alpha_propagate: FAlpha,
    fixed_propagate: FFixed,
) -> Result<LinearBounds>
where
    FAlpha: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
    FFixed: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
{
    match alpha_propagate(node_lb, pre_activation) {
        Ok(bounds) => Ok(bounds),
        Err(NyError::ShapeMismatch { .. }) => {
            debug!(
                "DAG α-CROWN: {layer_type} {node_name} alpha propagation ShapeMismatch, retrying local fixed-slope CROWN"
            );
            fixed_propagate(node_lb, pre_activation)
        }
        Err(err) => Err(err),
    }
}

pub(super) fn handle_nonlinear_node(
    network: &GraphNetwork,
    node_name: &str,
    node: &GraphNode,
    first_input: &str,
    node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    mut context: DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    match &node.layer {
        Layer::ReLU(relu) => handle_relu_node(
            network,
            node_name,
            node,
            first_input,
            node_cb,
            pre_activation,
            relu,
            &mut context,
        ),
        Layer::Sigmoid(sigmoid) => {
            let Some(alpha) = context.alpha_state.monotone_s_shaped_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Sigmoid",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_activation| {
                    sigmoid.propagate_linear_with_alpha(node_lb, pre_activation, alpha)
                },
                |node_lb, pre_activation| {
                    sigmoid.propagate_linear_with_bounds(node_lb, pre_activation)
                },
                &mut context,
            )
        }
        Layer::Tanh(tanh) => {
            let Some(alpha) = context.alpha_state.monotone_s_shaped_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Tanh",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_activation| {
                    tanh.propagate_linear_with_alpha(node_lb, pre_activation, alpha)
                },
                |node_lb, pre_activation| {
                    tanh.propagate_linear_with_bounds(node_lb, pre_activation)
                },
                &mut context,
            )
        }
        Layer::Sqrt(sqrt) => {
            let Some(alpha) = context.alpha_state.sqrt_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Sqrt",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_act| {
                    sqrt.propagate_linear_with_alpha(
                        node_lb,
                        pre_act,
                        &alpha.lower_path_mid,
                        Some(&alpha.upper_path_mid),
                    )
                },
                |node_lb, pre_act| sqrt.propagate_linear_with_bounds(node_lb, pre_act),
                &mut context,
            )
        }
        Layer::Reciprocal(reciprocal) => {
            let Some(alpha) = context.alpha_state.reciprocal_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Reciprocal",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_act| {
                    reciprocal.propagate_linear_with_alpha(
                        node_lb,
                        pre_act,
                        &alpha.lower_path_mid,
                        Some(&alpha.upper_path_mid),
                    )
                },
                |node_lb, pre_act| reciprocal.propagate_linear_with_bounds(node_lb, pre_act),
                &mut context,
            )
        }
        _ => Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb))),
    }
}

// DagAlphaNodeContext already bundles 14 fields; the remaining args are
// per-call inputs that differ across the match arms in handle_nonlinear_node.
#[allow(clippy::too_many_arguments)]
fn handle_relu_node(
    network: &GraphNetwork,
    node_name: &str,
    node: &GraphNode,
    first_input: &str,
    mut node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    relu: &ReLULayer,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    if matches!(&node_cb, CrownBounds::Patches(_)) && node.inputs.len() == 1 {
        // The alpha-specific Patches kernels do not yet carry the absolute
        // graph deadline through their allocation/flatten/compose phases.
        // Under finite authority skip them before inspecting geometry or
        // expanding alpha; the cooperative ordinary Patches dispatcher below
        // handles Anchored ReLU, while other variants take its typed Dense/full
        // CROWN fallback.
        //
        // #dag-alpha-patches-expiry: this is the GRAPH-lane twin of
        // `patches_step.rs:1107` (`4d0257ba9`, sequential lane, bootstrap
        // 37.3s -> 1.4s). Same defect class, and strictly worse here: the
        // sequential site only densified the carrier, whereas skipping this
        // branch also DISCARDS the optimized alpha — the node falls back to the
        // heuristic slope, so the whole root alpha ascent stops affecting the
        // bound at every Patches ReLU.
        //
        // It matters on cifar100 specifically because `patches_step.rs` lives
        // under `network/core/sequential/` and a DAG resnet never reaches it —
        // measured: arming NY_PATCHES_FINITE_EXPIRY alone moves cifar100
        // bootstrap by 0.0s and 0 of 6 verdicts, with the lever's own telemetry
        // empty in both arms. The sequential fix cannot help this benchmark;
        // this site is where the equivalent cost is paid.
        //
        // Same treatment, same lever, same contract: with the lever off the
        // predicate returns `deadline.is_some()` and the condition is
        // byte-identical to the historical `deadline.is_none()`.
        if !alpha_patches_finite_authority_refuses(context.deadline) {
            if let Some(&relu_idx) = context.relu_name_to_idx.get(node_name) {
                if let Some(alpha) = context.alpha_state.alpha(node_name) {
                    if let CrownBounds::Patches(ref pb) = node_cb {
                        // NOTE(#3782): Only lower-path gradient is captured here.
                        // Patches-mode ReLU is single-alpha (one optimizable lower
                        // slope + fixed upper chord). A dual-alpha patches path
                        // would need `propagate_patches_with_alpha` to return a
                        // separate `grad_upper`, which requires a relaxation redesign.
                        // #4404: expand channel-only alpha to full spatial before use.
                        let alpha_expanded = context.alpha_state.expand_alpha(node_name, alpha);
                        let propagated = if context.track_gradients {
                            relu.propagate_patches_with_alpha(pb, pre_activation, &alpha_expanded)
                                .map(|(bounds, gradient)| (bounds, Some(gradient)))
                        } else {
                            relu.propagate_patches_with_alpha_bound_only(
                                pb,
                                pre_activation,
                                &alpha_expanded,
                            )
                            .map(|bounds| (bounds, None))
                        };
                        match propagated {
                            Ok((new_cb, grad)) => {
                                // #4404: reduce gradient back to per-channel if channel-only.
                                if let Some(grad) = grad {
                                    context.gradients[relu_idx] =
                                        context.alpha_state.reduce_gradient(node_name, &grad);
                                }
                                record_patches_relu_intermediate(
                                    node_name,
                                    &node_cb,
                                    pre_activation,
                                    context,
                                )?;
                                return accumulate_crown_result(
                                    network,
                                    first_input,
                                    new_cb,
                                    context,
                                );
                            }
                            Err(e) if e.is_deadline_exceeded() || e.is_cpu_memory_exceeded() => {
                                // Resource authority is graph-wide. A failed alpha
                                // attempt must not turn a deadline or checked-memory
                                // refusal into an unbounded Dense retry.
                                return Err(e);
                            }
                            Err(e) => {
                                debug!(
                                    "DAG α-CROWN: Patches alpha-ReLU failed at {}: {}, \
                                 falling back to Dense alpha",
                                    node_name, e
                                );
                            }
                        }
                    }
                }
            }
        }

        // Patches-native ReLU backward (heuristic slope, no alpha or alpha failed).
        //
        // #1937: record Dense intermediates BEFORE the backward step mutates
        // `node_cb` (the stored A matrix must be the coefficients at the ReLU
        // output, matching the dense/alpha branches). Without this, a ReLU that
        // takes the heuristic branch during the AnalyticChain intermediates
        // pass leaves no A matrix / pre-ReLU bounds, so
        // `compute_graph_chain_rule_gradients` emits a zero-length gradient and
        // `GraphAlphaState::update` skips the node every iteration ("gradient
        // length 0 != alpha length N"). Recording is gradient-only — bounds
        // are untouched and any alpha in [0,1] stays a valid slope — so a
        // a semantic recording failure just leaves the gradient zero, exactly
        // the pre-fix behavior. Resource failures remain authoritative.
        if context.intermediate.is_some() {
            match record_patches_relu_intermediate(node_name, &node_cb, pre_activation, context) {
                Ok(()) => {}
                Err(e) if e.is_deadline_exceeded() || e.is_cpu_memory_exceeded() => {
                    // The capture is optional only for semantic failures. A
                    // finite deadline and the checked memory envelope remain
                    // authoritative and are classified by the outer walk.
                    return Err(e);
                }
                Err(e) => {
                    debug!(
                        "DAG α-CROWN: heuristic patches-ReLU intermediate recording failed at {}: {} \
                         (#1937); gradient for this node stays zero",
                        node_name, e
                    );
                }
            }
        }
        match crown_backward_step_patches(
            &node.layer,
            &mut node_cb,
            pre_activation,
            context.engine,
            0,
            "DAG-α-CROWN",
            context.deadline,
        ) {
            Ok(CrownStepResult::Continue) => {
                return accumulate_crown_result(network, first_input, node_cb, context);
            }
            Ok(CrownStepResult::IbpFallback(fallback)) => {
                if fallback.reason == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded {
                    debug!(
                        "DAG α-CROWN: ReLU Patches dispatch hit memory budget at {}: {}; falling back to CROWN",
                        node_name, fallback.details
                    );
                    return crown_fallback_result(network, context);
                }
                debug!(
                    "DAG α-CROWN: ReLU Patches dispatch failed at {}, converting to Dense",
                    node_name
                );
            }
            Err(e) if e.is_deadline_exceeded() || e.is_cpu_memory_exceeded() => {
                return Err(e);
            }
            Err(_) => {
                debug!(
                    "DAG α-CROWN: ReLU Patches dispatch failed at {}, converting to Dense",
                    node_name
                );
            }
        }
    }

    // The dense alpha/INVPROP ReLU kernels still contain unpollable
    // expansion, flatten, and publication phases. A finite request may use
    // the cooperative Anchored Patches path above, but it must not densify and
    // then enter those legacy kernels under the same absolute authority.
    if context.finite_structured_boundary {
        return crown_fallback_result(network, context);
    }

    if let Some(result) =
        ensure_dense_or_crown_fallback(network, node_name, "ReLU", &mut node_cb, context)?
    {
        return Ok(result);
    }
    let CrownBounds::Dense(node_lb) = node_cb else {
        unreachable!("successful ReLU Dense-boundary preparation must publish Dense")
    };
    let node_lb =
        GraphNetwork::apply_invprop_constraints(node_name, node_lb, context.invprop_state);
    record_dense_relu_intermediate(node_name, &node_lb, pre_activation, context)?;

    if let Some(&relu_idx) = context.relu_name_to_idx.get(node_name) {
        if let Some(alpha) = context.alpha_state.alpha(node_name) {
            let alpha_upper = context.alpha_state.alpha_upper(node_name);
            // #4404: expand channel-only alpha to full spatial before backward.
            let alpha_expanded = context.alpha_state.expand_alpha(node_name, alpha);
            let alpha_upper_expanded =
                alpha_upper.map(|au| context.alpha_state.expand_alpha(node_name, au));
            // #spec-axis-alpha (design §5.2): when this walk's rows are
            // output specs and the node carries δ rows, build the per-node
            // row table ONCE — materialize each slot's clamped `α_base + δ`
            // at base width, expand channel-only alphas exactly like the
            // shared path above, and map carrier rows through the seed's
            // subset to slots. Everything else (no slots, malformed state,
            // Shared scope) leaves `spec_table` empty and the propagate
            // calls bind the identical shared slices as always.
            let mut slot_alphas: Vec<Array1<f32>> = Vec::new();
            let mut slot_of_row: Vec<Option<usize>> = Vec::new();
            if context.alpha_state.has_spec_deltas()
                && !matches!(context.row_scope, AlphaRowScope::Shared)
            {
                if let Some(materialized) = context.alpha_state.materialized_spec_alphas(node_name)
                {
                    slot_alphas = materialized
                        .iter()
                        .map(|slot_alpha| {
                            context
                                .alpha_state
                                .expand_alpha(node_name, slot_alpha)
                                .into_owned()
                        })
                        .collect();
                    slot_of_row = (0..node_lb.num_outputs())
                        .map(|carrier_row| {
                            context
                                .row_scope
                                .spec_row(carrier_row)
                                .and_then(|spec_row| {
                                    context.alpha_state.slot_for_spec_row(spec_row)
                                })
                        })
                        .collect();
                    // A table with no active row is dead weight — drop it so
                    // the fast path stays the byte-identical two-arg call.
                    if slot_of_row.iter().all(Option::is_none) {
                        slot_alphas.clear();
                        slot_of_row.clear();
                    }
                }
            }
            // #3813: Catch ShapeMismatch from ReLU alpha propagation
            // (RSPLITTER models change intermediate dimensions). Fall back
            // to plain CROWN, which is always sound.
            let propagated = if !slot_alphas.is_empty() {
                relu.propagate_linear_with_alpha_spec_rows(
                    &node_lb,
                    pre_activation,
                    &alpha_expanded,
                    alpha_upper_expanded.as_ref(),
                    crate::layers::activations::relu::SpecRowAlphas {
                        slot_of_row: &slot_of_row,
                        slot_alphas: &slot_alphas,
                    },
                    // Slice 2c wires the per-slot gradient buffers through the
                    // optimize loop; the read side stays gradient-silent here.
                    None,
                    context.track_gradients,
                )
                .map(|(bounds, lower, upper)| {
                    if context.track_gradients {
                        (bounds, Some((lower, upper)))
                    } else {
                        (bounds, None)
                    }
                })
            } else if context.track_gradients {
                relu.propagate_linear_with_alpha(
                    &node_lb,
                    pre_activation,
                    &alpha_expanded,
                    alpha_upper_expanded.as_ref(),
                )
                .map(|(bounds, lower, upper)| (bounds, Some((lower, upper))))
            } else {
                relu.propagate_linear_with_alpha_bound_only(
                    &node_lb,
                    pre_activation,
                    &alpha_expanded,
                    alpha_upper_expanded.as_ref(),
                )
                .map(|bounds| (bounds, None))
            };
            match propagated {
                Ok((new_lb, grads)) => {
                    // #4404: reduce per-neuron gradient to per-channel if channel-only.
                    if let Some((grad, grad_upper)) = grads {
                        context.gradients[relu_idx] =
                            context.alpha_state.reduce_gradient(node_name, &grad);
                        context.gradients_upper[relu_idx] =
                            context.alpha_state.reduce_gradient(node_name, &grad_upper);
                    }
                    return accumulate_dense_result(network, first_input, new_lb, context);
                }
                Err(NyError::ShapeMismatch { .. }) => {
                    debug!(
                        "DAG α-CROWN: ReLU {} alpha propagation ShapeMismatch, CROWN fallback",
                        node_name,
                    );
                    return crown_fallback_result(network, context);
                }
                Err(e) => return Err(e),
            }
        }
    }

    // Fallback: propagate without alpha
    // #3813: Catch ShapeMismatch instead of wrapping in InvalidSpec.
    match relu.propagate_linear_with_bounds(&node_lb, pre_activation) {
        Ok(new_lb) => accumulate_dense_result(network, first_input, new_lb, context),
        Err(NyError::ShapeMismatch { .. }) => {
            debug!(
                "DAG α-CROWN: ReLU {} fallback propagation ShapeMismatch, CROWN fallback",
                node_name,
            );
            crown_fallback_result(network, context)
        }
        Err(e) => Err(NyError::InvalidSpec(format!(
            "DAG α-CROWN failed at node '{}' (ReLU): {}",
            node_name, e
        ))),
    }
}

// Same rationale as handle_relu_node: per-call args differ across match arms
// and DagAlphaNodeContext already bundles the shared state.
#[allow(clippy::too_many_arguments)]
fn handle_monotone_node<FAlpha, FFixed>(
    network: &GraphNetwork,
    node_name: &str,
    layer_type: &str,
    first_input: &str,
    mut node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    alpha_propagate: FAlpha,
    fixed_propagate: FFixed,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult>
where
    FAlpha: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
    FFixed: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
{
    // Monotone dense alpha/fixed-slope kernels and the INVPROP fold are not
    // cooperatively pollable yet. Keep finite authority on the already
    // deadline-aware whole-result fallback before materialization or scans.
    if context.finite_structured_boundary {
        return crown_fallback_result(network, context);
    }
    if let Some(result) =
        ensure_dense_or_crown_fallback(network, node_name, layer_type, &mut node_cb, context)?
    {
        return Ok(result);
    }
    let CrownBounds::Dense(node_lb) = node_cb else {
        unreachable!("successful monotone Dense-boundary preparation must publish Dense")
    };
    let node_lb =
        GraphNetwork::apply_invprop_constraints(node_name, node_lb, context.invprop_state);

    // #4118: Catch ShapeMismatch/UnsupportedConfiguration from both
    // alpha AND fixed-slope retry, falling back to plain CROWN instead
    // of propagating the error up to graph-wide IBP fallback.
    match retry_monotone_shape_mismatch_with_fixed_slope(
        node_name,
        layer_type,
        &node_lb,
        pre_activation,
        alpha_propagate,
        fixed_propagate,
    ) {
        Ok(new_lb) => accumulate_dense_result(network, first_input, new_lb, context),
        Err(NyError::ShapeMismatch { .. }) | Err(NyError::UnsupportedConfiguration(_)) => {
            debug!(
                "DAG α-CROWN: {layer_type} {} fixed-slope retry also failed, CROWN fallback",
                node_name,
            );
            crown_fallback_result(network, context)
        }
        Err(e) => Err(e),
    }
}

fn ensure_dense_or_crown_fallback(
    network: &GraphNetwork,
    node_name: &str,
    layer_type: &str,
    node_cb: &mut CrownBounds,
    context: &DagAlphaNodeContext<'_>,
) -> Result<Option<NonlinearNodeResult>> {
    if matches!(node_cb, CrownBounds::Patches(_)) {
        match node_cb.ensure_dense_with_deadline_for_purpose(
            context.deadline,
            PatchesMaterializationPurpose::Other,
        ) {
            Ok(_) => {}
            Err(e) if e.is_deadline_exceeded() || e.is_cpu_memory_exceeded() => return Err(e),
            Err(e) => {
                debug!(
                    "DAG α-CROWN: ensure_dense failed at {layer_type} {}: {}, CROWN fallback",
                    node_name, e
                );
                return Ok(Some(crown_fallback_result(network, context)?));
            }
        }
    }
    Ok(None)
}

pub(super) fn record_patches_relu_intermediate(
    node_name: &str,
    node_cb: &CrownBounds,
    pre_activation: &BoundedTensor,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<()> {
    // Store Dense intermediate for chain-rule gradient
    // computation (#3293). The Patches alpha-ReLU path
    // bypasses Dense intermediate storage. Convert
    // Patches->Dense only for A-matrix storage so
    // compute_graph_chain_rule_gradients gets non-zero
    // gradients for Patches-mode ReLUs.
    // Reference: design doc 2026-03-04-alpha-gradient-
    // patches-alternative.md Approach B.
    let deadline = context.deadline;
    let Some(inter) = context.intermediate.as_deref_mut() else {
        return Ok(());
    };
    let intermediate_bytes = inter.logical_memory_bytes();

    let CrownBounds::Patches(patches) = node_cb else {
        return Err(NyError::InternalError(
            "record_patches_relu_intermediate requires a Patches carrier".into(),
        ));
    };
    // Materialize through the borrowed carrier. Cloning `node_cb` here used to
    // deep-clone every patch/error/anchor buffer infallibly BEFORE `to_dense`
    // could apply its checked memory admission.
    let pre_activation_bytes = pre_activation
        .lower()
        .len()
        .saturating_add(pre_activation.upper().len())
        .saturating_mul(size_of::<f32>());
    let dense_lb = patches.to_dense_with_deadline_and_resident_for_purpose(
        deadline,
        intermediate_bytes.saturating_add(pre_activation_bytes),
        PatchesMaterializationPurpose::Other,
    )?;
    // This Dense value exists only to publish lower A. Move that matrix out and
    // immediately drop upper A, biases, and error receipts instead of cloning a
    // third full coefficient matrix beside the materialized lower/upper pair.
    let (lower_a, _, _, _) = dense_lb.into_parts();
    let retained_bytes = intermediate_bytes
        .saturating_add(patches.memory_bytes())
        .saturating_add(lower_a.len().saturating_mul(size_of::<f32>()));
    let (lower, upper) =
        copy_pre_activation_for_intermediate(pre_activation, retained_bytes, deadline)?;
    // All fallible work completed while both the source carrier and the
    // intermediate maps were untouched.
    inter.a_at_relu.insert(node_name.to_string(), lower_a);
    inter
        .pre_relu_bounds
        .insert(node_name.to_string(), (lower, upper));
    Ok(())
}

pub(super) fn record_dense_relu_intermediate(
    node_name: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<()> {
    // When capturing intermediates, store A matrix and pre-ReLU bounds
    // BEFORE the ReLU is applied (for chain-rule gradients).
    let deadline = context.deadline;
    let Some(inter) = context.intermediate.as_deref_mut() else {
        return Ok(());
    };
    let intermediate_bytes = inter.logical_memory_bytes();
    let pre_activation_bytes = pre_activation
        .lower()
        .len()
        .saturating_add(pre_activation.upper().len())
        .saturating_mul(size_of::<f32>());
    let lower_a = node_lb.try_clone_lower_a_with_deadline(
        deadline,
        intermediate_bytes.saturating_add(pre_activation_bytes),
    )?;
    let retained_bytes = intermediate_bytes
        .saturating_add(node_lb.memory_bytes())
        .saturating_add(lower_a.len().saturating_mul(size_of::<f32>()));
    let (lower, upper) =
        copy_pre_activation_for_intermediate(pre_activation, retained_bytes, deadline)?;

    // Publish both maps only after all checked allocations and deadline polls.
    inter.a_at_relu.insert(node_name.to_string(), lower_a);
    inter
        .pre_relu_bounds
        .insert(node_name.to_string(), (lower, upper));
    Ok(())
}

fn accumulate_crown_result(
    network: &GraphNetwork,
    first_input: &str,
    node_cb: CrownBounds,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    network.accumulate_crown_bounds_to_input_with_deadline(
        first_input,
        node_cb,
        context.node_crown_bounds,
        context.output_dim,
        context.input_dim,
        context.input_accumulated,
        context.deadline,
    )?;
    Ok(NonlinearNodeResult::Continue)
}

fn accumulate_dense_result(
    network: &GraphNetwork,
    first_input: &str,
    node_lb: LinearBounds,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    network.accumulate_dense_bounds_to_input_with_deadline(
        first_input,
        node_lb,
        context.node_crown_bounds,
        context.output_dim,
        context.input_dim,
        context.input_accumulated,
        context.deadline,
    )?;
    Ok(NonlinearNodeResult::Continue)
}

fn crown_fallback_result(
    network: &GraphNetwork,
    context: &DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    let bounds =
        super::full_output_crown_fallback_or_refuse(context.forbid_full_output_fallback, || {
            network
                .propagate_crown_with_engine_and_deadline(
                    context.input,
                    context.engine,
                    context.deadline,
                )
                .map(|result| result.bounds)
        })?;
    Ok(NonlinearNodeResult::ReturnBounds(bounds))
}
