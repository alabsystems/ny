// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::patches_target::patches_dense_fallback_details;
use super::*;
use crate::network::core::try_dense_spatial_patches_reentry;
use crate::network::core::GraphNode;

#[cfg(test)]
pub(super) fn initial_target_crown_bounds(
    graph: &GraphNetwork,
    target_node: &str,
    alpha_state: Option<&GraphAlphaState>,
    relevant_nodes: &[String],
    target_bounds: &BoundedTensor,
    target_contract: &GraphTargetShapeContract,
) -> (bool, CrownBounds) {
    initial_target_crown_bounds_with_override(
        graph,
        target_node,
        alpha_state,
        relevant_nodes,
        target_bounds,
        target_contract,
        false,
    )
}

/// Variant of `initial_target_crown_bounds` that accepts a patches override.
///
/// When `collector_patches_override` is true, the CROWN-IBP collector may use
/// patches mode even when the global `use_patches_mode` is false (matrix mode
/// for BaB cuts). The collector doesn't use cutting planes, so the matrix-mode
/// constraint doesn't apply to it. This avoids 30s+ per-node dense Conv2d
/// backward passes during intermediate bound collection (#3813).
pub(super) fn initial_target_crown_bounds_with_override(
    graph: &GraphNetwork,
    target_node: &str,
    alpha_state: Option<&GraphAlphaState>,
    relevant_nodes: &[String],
    target_bounds: &BoundedTensor,
    target_contract: &GraphTargetShapeContract,
    collector_patches_override: bool,
) -> (bool, CrownBounds) {
    let allow_patches = target_allows_patches_start(
        graph,
        target_node,
        alpha_state,
        relevant_nodes,
        target_bounds,
        collector_patches_override,
    );
    let initial_bounds = if allow_patches && target_bounds.shape().len() == 3 {
        let shape = target_bounds.shape();
        CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (shape[0], shape[1], shape[2]),
            (shape[0], shape[1], shape[2]),
        )))
    } else {
        CrownBounds::Dense(target_contract.identity_linear_bounds())
    };
    (allow_patches, initial_bounds)
}

/// Patches-start predicate shared by `initial_target_crown_bounds_with_override`
/// and the objective-chunked driver, which must decide `allow_patches` WITHOUT
/// materializing the full dense identity seed (#cgan-bn11-chunk): for an
/// over-budget target (e.g. a 28,800-dim BatchNorm) the dense `[dim x dim]`
/// pair is exactly the allocation the chunked route exists to avoid.
///
/// Gate Patches mode with use_patches_mode: matrix mode forces Dense
/// (abcrown.py:228-231). Exception: the CROWN-IBP collector may override this
/// gate because it doesn't use cutting planes — only the BaB backward needs
/// matrix mode for cuts (#3813).
pub(super) fn target_allows_patches_start(
    graph: &GraphNetwork,
    target_node: &str,
    alpha_state: Option<&GraphAlphaState>,
    relevant_nodes: &[String],
    target_bounds: &BoundedTensor,
    collector_patches_override: bool,
) -> bool {
    let patches_mode = graph.use_patches_mode || collector_patches_override;
    // Patches-mode alpha-CROWN (#conv-patches-collect alpha). Historically the
    // patches-start path required `alpha_state.is_none()` because the ReLU
    // patches backward applied only heuristic (non-alpha) relaxation slopes, so
    // an active alpha_state had to force the exact — but memory-blowing — dense
    // path. The per-node lower alpha is now applied in patches representation
    // (`ReLULayer::propagate_patches_with_alpha`, wired into
    // `try_patches_target_step_core`), so an alpha_state no longer forces Dense.
    // Gated behind NY_CONV_PATCHES_COLLECT so gate-OFF is byte-identical: with
    // the env unset, `alpha_state.is_some()` still forces Dense exactly as
    // before.
    let alpha_ok = alpha_state.is_none() || conv_patches_collect_alpha_enabled();
    patches_mode
        && alpha_ok
        && !relevant_nodes.is_empty()
        && graph.crown_ibp_target_can_start_in_patches(target_node, target_bounds)
}

/// `NY_CONV_PATCHES_COLLECT` gate for the patches-mode alpha-CROWN path
/// (#conv-patches-collect alpha). Mirrors the `budget_policy` env check verbatim
/// so the aggregate-budget raise and this gate relaxation share one
/// interpretation of the flag (set and non-`"0"`/non-empty).
pub(super) fn conv_patches_collect_alpha_enabled() -> bool {
    std::env::var_os("NY_CONV_PATCHES_COLLECT").is_some_and(|v| v != "0" && !v.is_empty())
}

pub(super) fn resolve_preactivation<'a>(
    input: &'a BoundedTensor,
    first_input_name: &'a str,
    crown_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
    ibp_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
) -> Result<&'a BoundedTensor> {
    if first_input_name == NETWORK_INPUT {
        Ok(input)
    } else {
        crown_bounds
            .get(first_input_name)
            .or_else(|| ibp_bounds.get(first_input_name))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for {} not found",
                    first_input_name
                ))
            })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_patches_target_step_core(
    graph: &GraphNetwork,
    label: &str,
    node_name: &str,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    first_input_name: &str,
    pre_activation: &BoundedTensor,
    ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
    node_crown_bounds: &mut crate::network::CrownMergeAccumulator,
    target_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    engine: Option<&dyn ny_core::GemmEngine>,
    per_node_deadline: Option<std::time::Instant>,
    collector_patches_override: bool,
    alpha_state: Option<&GraphAlphaState>,
) -> Result<bool> {
    // Gate re-entry with use_patches_mode: matrix mode skips re-entry (abcrown.py:228-231).
    // Exception: collector_patches_override allows re-entry in the CROWN-IBP collector (#3813).
    let patches_mode = graph.use_patches_mode || collector_patches_override;
    try_dense_spatial_patches_reentry(node_cb, node, node_name, ibp_bounds, patches_mode, label);

    if !matches!(node_cb, CrownBounds::Patches(_)) {
        return Ok(false);
    }

    // Patches residual passthrough for Add/Sub (#4382)
    if crate::network::core::graph::backward_helpers::try_apply_patches_residual_passthrough(
        graph,
        node,
        node_cb,
        ibp_bounds,
        node_crown_bounds,
        target_dim,
        input_dim,
        input_accumulated,
        label,
    )? {
        return Ok(true);
    }

    // Patches-mode alpha-CROWN for ReLU (#conv-patches-collect alpha). When an
    // alpha_state carries an optimized lower slope for THIS ReLU node, apply it
    // in patches representation instead of the heuristic
    // `propagate_patches_with_bounds` inside `crown_backward_step_patches`. This
    // is the SAME single-alpha relaxation the dense path takes with
    // `alpha_upper = None`: lower envelope `y >= alpha_i * x`
    // (`alpha_i in [0,1]`, sound for any crossing neuron), upper envelope the
    // fixed chord. Op-level bit-parity with the dense operator is pinned by the
    // `patches_backward_alpha` equivalence tests; the alpha is broadcast to
    // per-neuron with the SAME `expand_alpha` the dense branch uses. Any error
    // (shape mismatch, numerical) degrades SOUNDLY to the non-alpha patches /
    // dense path below — never a tighter-than-true bound. Only entered under the
    // relaxed `target_allows_patches_start` gate (NY_CONV_PATCHES_COLLECT), so
    // gate-OFF never reaches here with a `Some` alpha_state.
    if let Layer::ReLU(relu) = &node.layer {
        if let Some(alpha) = alpha_state.and_then(|s| s.alpha(node_name)) {
            let alpha_expanded = alpha_state
                .map(|s| s.expand_alpha(node_name, alpha))
                .unwrap_or_else(|| alpha.clone());
            // Scope the immutable borrow of `node_cb` (via `pb`) to just the
            // backward call, which returns owned bounds — so the reassignment
            // `*node_cb = result` below does not conflict with the borrow.
            let alpha_result = if let CrownBounds::Patches(pb) = &*node_cb {
                Some(relu.propagate_patches_with_alpha(pb, pre_activation, &alpha_expanded))
            } else {
                None
            };
            match alpha_result {
                Some(Ok((result, _grad))) => {
                    *node_cb = result;
                    graph.accumulate_crown_bounds_to_input(
                        first_input_name,
                        node_cb.clone(),
                        node_crown_bounds,
                        target_dim,
                        input_dim,
                        input_accumulated,
                    )?;
                    return Ok(true);
                }
                Some(Err(err)) if err.is_deadline_exceeded() => return Err(err),
                Some(Err(err)) => {
                    debug!(
                        "{}: Patches ReLU alpha for '{}' failed ({}); \
                         degrading to non-alpha patches/dense",
                        label, node_name, err
                    );
                    // Fall through to the heuristic patches step (still sound).
                }
                None => {}
            }
        }
    }

    match crown_backward_step_patches(
        &node.layer,
        node_cb,
        pre_activation,
        engine,
        0,
        label,
        per_node_deadline,
    ) {
        Ok(CrownStepResult::Continue) => {
            graph.accumulate_crown_bounds_to_input(
                first_input_name,
                node_cb.clone(),
                node_crown_bounds,
                target_dim,
                input_dim,
                input_accumulated,
            )?;
            Ok(true)
        }
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            if fallback.reason == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: patches dispatch at '{node_name}' exceeded memory budget: {}",
                    fallback.details
                )));
            }
            debug!(
                "{}: Patches dispatch for {} ({}) requested Dense fallback: {}",
                label,
                node_name,
                node.layer.layer_type(),
                fallback.details
            );
            if let Some(details) = patches_dense_fallback_details(
                node_cb,
                "graph_alpha::bounds::propagate_crown_to_node_core",
            )? {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: {details}"
                )));
            }
            node_cb.ensure_dense()?;
            Ok(false)
        }
        Err(err) if err.is_deadline_exceeded() => Err(err),
        Err(err) => {
            debug!(
                "{}: Patches dispatch for {} ({}) failed: {}, falling back to Dense",
                label,
                node_name,
                node.layer.layer_type(),
                err
            );
            if let Some(details) = patches_dense_fallback_details(
                node_cb,
                "graph_alpha::bounds::propagate_crown_to_node_core",
            )? {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{label}: {details}"
                )));
            }
            node_cb.ensure_dense()?;
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Conv2dLayer, ReLULayer};
    use ndarray::{arr1, ArrayD, IxDyn};

    #[test]
    fn test_initial_target_crown_bounds_skips_patches_in_matrix_mode_3813() {
        crate::tests::with_crown_dense_budget_mb("1", || {
            let kernel =
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4])
                    .expect("valid conv kernel");
            let conv = Conv2dLayer::with_input_shape(
                kernel,
                Some(arr1(&[0.1_f32])),
                (1, 1),
                (0, 0),
                33,
                33,
            )
            .expect("valid conv");

            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
            graph.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer),
                vec!["conv".into()],
            ));
            graph.set_output("relu");
            graph.set_use_patches_mode(false);

            let target_bounds = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 32, 32]), -0.4_f32),
                ArrayD::from_elem(IxDyn(&[1, 32, 32]), 0.7_f32),
            )
            .expect("valid target bounds");
            let target_contract = GraphTargetShapeContract::from_bounds("relu", &target_bounds);
            let relevant_nodes = vec!["conv".to_string(), "relu".to_string()];

            let (allow_patches, initial_bounds) = initial_target_crown_bounds(
                &graph,
                "relu",
                None,
                &relevant_nodes,
                &target_bounds,
                &target_contract,
            );

            assert!(
                !allow_patches,
                "#3813: matrix conv mode must disable CROWN-IBP target patches starts"
            );
            assert!(
                matches!(initial_bounds, CrownBounds::Dense(_)),
                "#3813: matrix conv mode must seed target backward in Dense mode"
            );
        });
    }
}
