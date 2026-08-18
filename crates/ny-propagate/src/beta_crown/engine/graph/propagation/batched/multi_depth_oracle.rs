// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Host reference oracle for selected intermediate rows at multiple graph depths.
//!
//! This is deliberately an observation/differential surface, not a production
//! tightening route. Every target occupies one logical domain in the existing
//! batched graph-CROWN carrier. All logical domains borrow the same immutable
//! bounds cache and use bit-identical copies of the same input box, so one
//! reverse-topological walk can inject selected identities at heterogeneous
//! graph nodes without ever merging rows belonging to different targets.
//!
//! The multi-domain Linear helper requires every simultaneously active domain
//! to have the same objective-row count. The planner therefore pads shorter
//! selections by repeating their first selected identity row. Rows are
//! independent throughout CROWN propagation; the repeated suffix is discarded
//! before certificate-error discharge and cannot affect a requested prefix.
//!
//! The public-within-crate entry points fix the engine to
//! [`ny_core::NaiveCpuGemmEngine`]. Callers cannot accidentally turn this oracle
//! into an accelerator route. Results are transactional: an invalid target, a
//! dispatch/deadline failure, a missing network-input terminal, or coefficient
//! error that cannot be discharged over the shared input box refuses the whole
//! request.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{s, Array2};
use ny_core::{GemmEngine, NaiveCpuGemmEngine};
use ny_tensor::BoundedTensor;

use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::network::CrownDispatchPlan;
use crate::{GraphNetwork, LinearBounds};

use super::indexed_pending::IndexedPendingLinearBounds;

// Diagnostic-only ceilings. They bound host validation/allocation work without
// prescribing accelerator capacity or production batching policy.
const HOST_MULTI_DEPTH_MAX_GRAPH_NODES: usize = 1 << 16;
const HOST_MULTI_DEPTH_MAX_TARGETS: usize = 256;
const HOST_MULTI_DEPTH_MAX_ROWS_PER_TARGET: usize = 4096;
const HOST_MULTI_DEPTH_MAX_NODE_DIM: usize = 1 << 22;
const HOST_MULTI_DEPTH_MAX_SEED_ELEMENTS: usize = 1 << 24;
const HOST_MULTI_DEPTH_POLL_STRIDE: usize = 4096;

/// One selected-identity target for the host multi-depth oracle.
///
/// `selected_neurons` must be strictly increasing, matching the accelerator
/// plan contract. `target_id` must be unique within one request.
#[derive(Clone, Debug)]
pub(crate) struct HostMultiDepthTarget<'a, TargetId> {
    pub(crate) target_id: TargetId,
    pub(crate) node_name: &'a str,
    pub(crate) selected_neurons: &'a [usize],
}

/// One completed oracle result, associated with the caller's target ID.
#[derive(Debug)]
pub(crate) struct HostMultiDepthBound<TargetId, Bounds> {
    pub(crate) target_id: TargetId,
    pub(crate) bounds: Bounds,
}

#[derive(Debug)]
struct PlannedTarget {
    node_idx: usize,
    node_dim: usize,
    requested_rows: usize,
}

#[inline]
fn past_deadline(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|limit| Instant::now() >= limit)
}

/// Validate every target before the backward walk allocates or dispatches work.
///
/// The returned maximum row count is the common logical-domain row count used
/// by the existing multi-domain Linear implementation.
fn plan_targets<TargetId: Eq + Hash>(
    graph_plan: &CrownDispatchPlan,
    targets: &[HostMultiDepthTarget<'_, TargetId>],
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    deadline: Option<Instant>,
) -> Option<(Vec<PlannedTarget>, usize)> {
    if targets.is_empty() || targets.len() > HOST_MULTI_DEPTH_MAX_TARGETS || past_deadline(deadline)
    {
        return None;
    }

    let mut ids = HashSet::new();
    ids.try_reserve(targets.len()).ok()?;
    let mut planned = Vec::new();
    planned.try_reserve_exact(targets.len()).ok()?;
    let mut max_rows = 0usize;
    for (target_index, target) in targets.iter().enumerate() {
        if past_deadline(deadline)
            || target.selected_neurons.is_empty()
            || target.selected_neurons.len() > HOST_MULTI_DEPTH_MAX_ROWS_PER_TARGET
            || !ids.insert(&target.target_id)
        {
            return None;
        }
        let node_idx = graph_plan.index_of(target.node_name)?;
        if graph_plan.is_network_input(node_idx) {
            // NETWORK_INPUT is not a graph node and therefore has no cache entry.
            // Direct-input selected identities have a separate exact path.
            return None;
        }
        let node_dim = bounds_cache.get(target.node_name)?.len();
        if node_dim == 0 || node_dim > HOST_MULTI_DEPTH_MAX_NODE_DIM {
            return None;
        }
        let mut previous = None;
        for (row_index, &neuron_idx) in target.selected_neurons.iter().enumerate() {
            if (target_index + row_index).is_multiple_of(HOST_MULTI_DEPTH_POLL_STRIDE)
                && past_deadline(deadline)
            {
                return None;
            }
            if neuron_idx >= node_dim
                || u32::try_from(neuron_idx).is_err()
                || previous.is_some_and(|prior| prior >= neuron_idx)
            {
                return None;
            }
            previous = Some(neuron_idx);
        }
        max_rows = max_rows.max(target.selected_neurons.len());
        planned.push(PlannedTarget {
            node_idx,
            node_dim,
            requested_rows: target.selected_neurons.len(),
        });
    }

    // Preflight the largest immediately materialized seed. The ndarray shape
    // constructor must never receive an overflowing logical element count.
    let mut aggregate_seed_elements = 0usize;
    for (index, target) in planned.iter().enumerate() {
        if index.is_multiple_of(64) && past_deadline(deadline) {
            return None;
        }
        aggregate_seed_elements =
            aggregate_seed_elements.checked_add(max_rows.checked_mul(target.node_dim)?)?;
        if aggregate_seed_elements > HOST_MULTI_DEPTH_MAX_SEED_ELEMENTS {
            return None;
        }
    }
    targets.len().checked_mul(max_rows)?;
    (!past_deadline(deadline)).then_some(())?;
    Some((planned, max_rows))
}

/// Retain the requested prefix of a padded logical-domain terminal while
/// preserving its coefficient-error certificate for the subsequent discharge.
fn selected_prefix(bounds: LinearBounds, requested_rows: usize) -> Option<LinearBounds> {
    if requested_rows == 0 || requested_rows > bounds.num_outputs() {
        return None;
    }
    if requested_rows == bounds.num_outputs() {
        return Some(bounds);
    }

    let lower_err = bounds
        .lower_a_err()
        .map(|error| error.slice(s![..requested_rows, ..]).to_owned());
    let upper_err = bounds
        .upper_a_err()
        .map(|error| error.slice(s![..requested_rows, ..]).to_owned());
    let n_inputs = bounds.num_inputs();
    let mut prefix = LinearBounds::new(
        bounds.lower_a().slice(s![..requested_rows, ..]).to_owned(),
        bounds.lower_b().slice(s![..requested_rows]).to_owned(),
        bounds.upper_a().slice(s![..requested_rows, ..]).to_owned(),
        bounds.upper_b().slice(s![..requested_rows]).to_owned(),
    )
    .ok()?;

    match (lower_err, upper_err) {
        (None, None) => {}
        (lower, upper) => prefix.set_coeff_err(
            lower.unwrap_or_else(|| Array2::zeros((requested_rows, n_inputs))),
            upper.unwrap_or_else(|| Array2::zeros((requested_rows, n_inputs))),
        ),
    }
    Some(prefix)
}

/// Implementation seam used only by this module's tests to count the shared
/// host walk. The crate-visible oracle below always supplies the naive CPU
/// engine.
#[allow(clippy::too_many_arguments)]
fn input_relative_with_engine<TargetId>(
    graph: &GraphNetwork,
    targets: &[HostMultiDepthTarget<'_, TargetId>],
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
) -> Option<Vec<HostMultiDepthBound<TargetId, LinearBounds>>>
where
    TargetId: Clone + Eq + Hash,
{
    if graph.nodes.len() > HOST_MULTI_DEPTH_MAX_GRAPH_NODES || past_deadline(deadline) {
        return None;
    }
    let graph_plan = CrownDispatchPlan::build(graph).ok()?;
    if graph_plan.reverse_order.len() > HOST_MULTI_DEPTH_MAX_GRAPH_NODES || past_deadline(deadline)
    {
        return None;
    }
    let (planned, padded_rows) = plan_targets(&graph_plan, targets, bounds_cache, deadline)?;
    let mut peak_frontier_elements = 0usize;
    for (index, name) in graph_plan.idx_to_name.iter().enumerate() {
        if index.is_multiple_of(64) && past_deadline(deadline) {
            return None;
        }
        let node_dim = if name == crate::NETWORK_INPUT {
            constrained_input.len()
        } else {
            bounds_cache.get(name)?.len()
        };
        if node_dim == 0 || node_dim > HOST_MULTI_DEPTH_MAX_NODE_DIM {
            return None;
        }
        peak_frontier_elements = peak_frontier_elements.max(padded_rows.checked_mul(node_dim)?);
    }
    // Four affine matrices (lower/upper centers and possible coefficient-error
    // twins) may coexist at a frontier. Bound the whole carrier, not merely the
    // initially injected identity matrices at their target depths.
    let peak_affine_elements = peak_frontier_elements.checked_mul(4)?;
    if peak_affine_elements > HOST_MULTI_DEPTH_MAX_SEED_ELEMENTS {
        return None;
    }
    let nodes_by_idx = super::build_nodes_by_idx(graph, &graph_plan).ok()?;
    if past_deadline(deadline) {
        return None;
    }
    let n_targets = targets.len();

    let input_clone_elements = n_targets.checked_mul(constrained_input.len())?;
    if input_clone_elements > HOST_MULTI_DEPTH_MAX_SEED_ELEMENTS {
        return None;
    }

    let mut pending = IndexedPendingLinearBounds::new(&graph_plan, n_targets);
    for (domain_idx, (target, plan)) in targets.iter().zip(&planned).enumerate() {
        if past_deadline(deadline) {
            return None;
        }
        let mut padded_selection = Vec::with_capacity(padded_rows);
        padded_selection.extend_from_slice(target.selected_neurons);
        padded_selection.resize(padded_rows, target.selected_neurons[0]);
        let seed = LinearBounds::identity_rows(plan.node_dim, &padded_selection);
        pending.seed_idx(plan.node_idx, domain_idx, seed).ok()?;
    }

    // `dispatch_node_backward` models batch entries as domains and therefore
    // accepts owned input boxes. These are bit-identical copies of one borrowed
    // snapshot; every cache reference below aliases the exact same immutable map.
    let mut constrained_inputs = Vec::new();
    constrained_inputs.try_reserve_exact(n_targets).ok()?;
    for target_index in 0..n_targets {
        if target_index.is_multiple_of(16) && past_deadline(deadline) {
            return None;
        }
        constrained_inputs.push(constrained_input.clone());
    }
    let bounds_caches = vec![bounds_cache; n_targets];
    let beta_states = vec![beta_state; n_targets];
    let alpha_states = vec![alpha_state; n_targets];

    // The only graph traversal in this oracle. A target seeded below another
    // remains dormant until its own index, while higher targets accumulate into
    // the same earlier node in separate logical-domain slots.
    for &idx in &graph_plan.reverse_order {
        if past_deadline(deadline) {
            return None;
        }
        let Some(node_bounds) = pending.take_idx(idx) else {
            continue;
        };
        if !node_bounds.iter().any(Option::is_some) {
            continue;
        }
        super::backward_core::dispatch_node_backward(
            graph_plan.name_of(idx),
            nodes_by_idx[idx],
            node_bounds,
            &constrained_inputs,
            &bounds_caches,
            &beta_states,
            &alpha_states,
            &mut pending,
            n_targets,
            constrained_input.len(),
            engine,
            deadline,
            None,
            false,
        )
        .ok()?;
    }

    if past_deadline(deadline)
        || pending.input_accumulated().len() != n_targets
        || pending.input_accumulated().iter().any(|&done| !done)
    {
        return None;
    }
    let terminals = pending.take_network_input()?;
    if terminals.len() != n_targets {
        return None;
    }

    // Stage every result locally. Nothing escapes unless every target has a
    // terminal and every requested row's coefficient error folds completely.
    let mut completed = Vec::with_capacity(n_targets);
    for ((target, plan), terminal) in targets.iter().zip(&planned).zip(terminals) {
        if past_deadline(deadline) {
            return None;
        }
        let mut bounds = selected_prefix(terminal?, plan.requested_rows)?;
        // Refuse a malformed terminal before the no-deadline eager fold gets a
        // chance to normalize it to conservative rows. The oracle's contract is
        // exact in both deadline modes: valid coefficient error must discharge;
        // malformed coefficients/errors refuse the complete target set.
        if bounds.validate_internal_shapes().is_err() || bounds.validate_no_nan().is_err() {
            return None;
        }
        bounds
            .fold_coeff_err_over_box_eager_with_deadline(constrained_input, deadline)
            .ok()?;
        if bounds.has_coeff_err()
            || bounds.num_outputs() != plan.requested_rows
            || bounds.num_inputs() != constrained_input.len()
            || bounds.validate_internal_shapes().is_err()
            || bounds.validate_no_nan().is_err()
        {
            return None;
        }
        completed.push(HostMultiDepthBound {
            target_id: target.target_id.clone(),
            bounds,
        });
    }
    if past_deadline(deadline) {
        return None;
    }
    Some(completed)
}

/// Run the selected multi-depth CROWN reference on the naive CPU engine and
/// return input-relative affine bounds in request order.
///
/// This function has no production call site and confers no verification
/// authority. It exists to differentially check fused accelerator work against
/// the existing host graph semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_multi_depth_input_relative_bounds<TargetId>(
    graph: &GraphNetwork,
    targets: &[HostMultiDepthTarget<'_, TargetId>],
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    deadline: Option<Instant>,
) -> Option<Vec<HostMultiDepthBound<TargetId, LinearBounds>>>
where
    TargetId: Clone + Eq + Hash,
{
    input_relative_with_engine(
        graph,
        targets,
        bounds_cache,
        constrained_input,
        beta_state,
        alpha_state,
        &NaiveCpuGemmEngine,
        deadline,
    )
}

/// Run the host multi-depth reference and soundly concretize every completed
/// affine result over the same immutable input box.
///
/// The return remains all-or-nothing: even though concretization is staged in
/// target order, no prefix is published if a later target refuses or expires.
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_multi_depth_concretized_bounds<TargetId>(
    graph: &GraphNetwork,
    targets: &[HostMultiDepthTarget<'_, TargetId>],
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    deadline: Option<Instant>,
) -> Option<Vec<HostMultiDepthBound<TargetId, BoundedTensor>>>
where
    TargetId: Clone + Eq + Hash,
{
    let affine = host_multi_depth_input_relative_bounds(
        graph,
        targets,
        bounds_cache,
        constrained_input,
        beta_state,
        alpha_state,
        deadline,
    )?;
    let mut concrete = Vec::with_capacity(affine.len());
    for result in affine {
        if past_deadline(deadline) {
            return None;
        }
        let bounds = result
            .bounds
            .concretize_sound_with_deadline(constrained_input, deadline)
            .ok()?;
        concrete.push(HostMultiDepthBound {
            target_id: result.target_id,
            bounds,
        });
    }
    if past_deadline(deadline) {
        return None;
    }
    Some(concrete)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::{arr1, arr2};
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, Result};

    use super::*;
    use crate::layers::binary_ops::AddLayer;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::{GraphNode, NETWORK_INPUT};

    struct CountingCpuGemm {
        calls: AtomicUsize,
    }

    impl CountingCpuGemm {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl GemmEngine for CountingCpuGemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
    }

    fn bounded(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("valid fixture bounds")
    }

    fn fixture() -> (
        GraphNetwork,
        HashMap<String, Arc<BoundedTensor>>,
        BoundedTensor,
    ) {
        let linear0 = LinearLayer::new(
            arr2(&[[1.0, -0.5], [-1.5, 0.25], [0.75, 2.0]]),
            Some(arr1(&[0.1, -0.2, 0.3])),
        )
        .expect("linear0");
        let linear1 = LinearLayer::new(
            arr2(&[[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]]),
            Some(arr1(&[-0.1, 0.2])),
        )
        .expect("linear1");
        let linear2 = LinearLayer::new(
            arr2(&[[1.0, -0.75], [0.4, 1.5]]),
            Some(arr1(&[0.05, -0.15])),
        )
        .expect("linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear0", Layer::Linear(linear0)));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["linear0".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear1",
            Layer::Linear(linear1),
            vec!["relu0".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu1".into()],
        ));
        graph.set_output("linear2");

        let cache = HashMap::from([
            (
                "linear0".into(),
                Arc::new(bounded(&[-2.0, -2.0, -3.0], &[2.0, 2.0, 3.0])),
            ),
            (
                "relu0".into(),
                Arc::new(bounded(&[0.0, 0.0, 0.0], &[2.0, 2.0, 3.0])),
            ),
            (
                "linear1".into(),
                Arc::new(bounded(&[-4.0, -4.0], &[4.0, 4.0])),
            ),
            ("relu1".into(), Arc::new(bounded(&[0.0, 0.0], &[4.0, 4.0]))),
            (
                "linear2".into(),
                Arc::new(bounded(&[-7.0, -7.0], &[7.0, 7.0])),
            ),
        ]);
        (graph, cache, bounded(&[-1.0, -0.5], &[1.0, 0.75]))
    }

    fn assert_linear_bits_eq(actual: &LinearBounds, expected: &LinearBounds) {
        assert_eq!(actual.lower_a().shape(), expected.lower_a().shape());
        assert_eq!(actual.upper_a().shape(), expected.upper_a().shape());
        assert_eq!(
            actual
                .lower_a()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .lower_a()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .upper_a()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .upper_a()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .lower_b()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .lower_b()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .upper_b()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .upper_b()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dag_merge_is_isolated_per_target_and_matches_scalar_replays() {
        let left = LinearLayer::new(arr2(&[[1.25]]), Some(arr1(&[0.1]))).expect("left");
        let right = LinearLayer::new(arr2(&[[-0.75]]), Some(arr1(&[-0.2]))).expect("right");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("left", Layer::Linear(left)));
        graph.add_node(GraphNode::from_input("right", Layer::Linear(right)));
        graph.add_node(GraphNode::new(
            "sum",
            Layer::Add(AddLayer),
            vec!["left".into(), "right".into()],
        ));
        graph.set_output("sum");
        let cache = HashMap::from([
            ("left".into(), Arc::new(bounded(&[-3.0], &[3.0]))),
            ("right".into(), Arc::new(bounded(&[-3.0], &[3.0]))),
            ("sum".into(), Arc::new(bounded(&[-6.0], &[6.0]))),
        ]);
        let input = bounded(&[-1.0], &[2.0]);
        let row = [0usize];
        let targets = [
            HostMultiDepthTarget {
                target_id: 3u64,
                node_name: "left",
                selected_neurons: &row,
            },
            HostMultiDepthTarget {
                target_id: 5u64,
                node_name: "sum",
                selected_neurons: &row,
            },
        ];
        let engine = CountingCpuGemm::new();
        let actual =
            input_relative_with_engine(&graph, &targets, &cache, &input, None, None, &engine, None)
                .expect("one DAG host walk");

        // right is active for the deep target only; left handles both logical
        // domains in the same two GEMMs. Separate replays would need six calls.
        assert_eq!(engine.calls(), 4);
        for (actual, node) in actual.iter().zip(["left", "sum"]) {
            let scalar = super::super::backward_selected_input_relative_bounds_at_node(
                &graph,
                node,
                &row,
                &cache,
                &input,
                None,
                None,
                &NaiveCpuGemmEngine,
                None,
            )
            .expect("scalar DAG replay");
            assert_linear_bits_eq(&actual.bounds, &scalar);
            assert!(!actual.bounds.has_coeff_err());
        }
    }

    #[test]
    fn heterogeneous_depths_share_one_walk_and_match_scalar_replays() {
        let (graph, cache, input) = fixture();
        let shallow_rows = [0usize, 2];
        let deep_rows = [1usize];
        let targets = [
            HostMultiDepthTarget {
                target_id: 17u64,
                node_name: "linear0",
                selected_neurons: &shallow_rows,
            },
            HostMultiDepthTarget {
                target_id: 29u64,
                node_name: "linear2",
                selected_neurons: &deep_rows,
            },
        ];
        let engine = CountingCpuGemm::new();
        let actual =
            input_relative_with_engine(&graph, &targets, &cache, &input, None, None, &engine, None)
                .expect("multi-depth host walk");

        // Three Linear nodes are visited once each, with one lower and one upper
        // GEMM per node. Separate replays would visit linear0 twice (8 calls).
        assert_eq!(engine.calls(), 6);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].target_id, 17);
        assert_eq!(actual[1].target_id, 29);
        assert!(!actual[0].bounds.has_coeff_err());
        assert!(!actual[1].bounds.has_coeff_err());

        let shallow = super::super::backward_selected_input_relative_bounds_at_node(
            &graph,
            "linear0",
            &shallow_rows,
            &cache,
            &input,
            None,
            None,
            &NaiveCpuGemmEngine,
            None,
        )
        .expect("scalar shallow replay");
        let deep = super::super::backward_selected_input_relative_bounds_at_node(
            &graph,
            "linear2",
            &deep_rows,
            &cache,
            &input,
            None,
            None,
            &NaiveCpuGemmEngine,
            None,
        )
        .expect("scalar deep replay");
        assert_linear_bits_eq(&actual[0].bounds, &shallow);
        assert_linear_bits_eq(&actual[1].bounds, &deep);
    }

    #[test]
    fn invalid_late_target_refuses_before_any_dispatch() {
        let (graph, cache, input) = fixture();
        let valid_rows = [0usize];
        let invalid_rows = [2usize]; // linear2 has dimension 2
        let targets = [
            HostMultiDepthTarget {
                target_id: "valid",
                node_name: "linear0",
                selected_neurons: &valid_rows,
            },
            HostMultiDepthTarget {
                target_id: "invalid",
                node_name: "linear2",
                selected_neurons: &invalid_rows,
            },
        ];
        let engine = CountingCpuGemm::new();
        assert!(input_relative_with_engine(
            &graph, &targets, &cache, &input, None, None, &engine, None,
        )
        .is_none());
        assert_eq!(engine.calls(), 0, "planning must be transactional");
    }

    #[test]
    fn resource_caps_and_expired_deadline_refuse_before_dispatch() {
        let (graph, cache, input) = fixture();
        let row = [0usize];
        let oversized: Vec<_> = (0..=HOST_MULTI_DEPTH_MAX_TARGETS)
            .map(|target_id| HostMultiDepthTarget {
                target_id,
                node_name: "linear0",
                selected_neurons: &row,
            })
            .collect();
        let engine = CountingCpuGemm::new();
        assert!(input_relative_with_engine(
            &graph, &oversized, &cache, &input, None, None, &engine, None,
        )
        .is_none());
        assert_eq!(engine.calls(), 0);

        let one = [HostMultiDepthTarget {
            target_id: 1usize,
            node_name: "linear0",
            selected_neurons: &row,
        }];
        assert!(input_relative_with_engine(
            &graph,
            &one,
            &cache,
            &input,
            None,
            None,
            &engine,
            Some(Instant::now()),
        )
        .is_none());
        assert_eq!(engine.calls(), 0);
    }

    #[test]
    fn undischarged_error_refuses_the_complete_target_set() {
        let (graph, cache, _) = fixture();
        let unbounded_input = BoundedTensor::new_allow_infinite(
            arr1(&[f32::NEG_INFINITY, -0.5]).into_dyn(),
            arr1(&[f32::INFINITY, 0.75]).into_dyn(),
        )
        .expect("valid unbounded input box");
        let shallow_rows = [0usize];
        let deep_rows = [1usize];
        let targets = [
            HostMultiDepthTarget {
                target_id: 1usize,
                node_name: "linear0",
                selected_neurons: &shallow_rows,
            },
            HostMultiDepthTarget {
                target_id: 2usize,
                node_name: "linear2",
                selected_neurons: &deep_rows,
            },
        ];
        assert!(host_multi_depth_input_relative_bounds(
            &graph,
            &targets,
            &cache,
            &unbounded_input,
            None,
            None,
            None,
        )
        .is_none());
    }

    #[test]
    fn concretized_surface_preserves_ids_order_and_sound_fold() {
        let (graph, cache, input) = fixture();
        let shallow_rows = [0usize, 1];
        let deep_rows = [0usize];
        let targets = [
            HostMultiDepthTarget {
                target_id: "shallow",
                node_name: "linear0",
                selected_neurons: &shallow_rows,
            },
            HostMultiDepthTarget {
                target_id: "deep",
                node_name: "linear2",
                selected_neurons: &deep_rows,
            },
        ];
        let affine = host_multi_depth_input_relative_bounds(
            &graph, &targets, &cache, &input, None, None, None,
        )
        .expect("affine oracle");
        let concrete =
            host_multi_depth_concretized_bounds(&graph, &targets, &cache, &input, None, None, None)
                .expect("concrete oracle");
        assert_eq!(concrete[0].target_id, "shallow");
        assert_eq!(concrete[1].target_id, "deep");

        for (affine, concrete) in affine.iter().zip(&concrete) {
            let expected = affine.bounds.concretize_sound(&input);
            assert_eq!(
                concrete
                    .bounds
                    .lower()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .lower()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                concrete
                    .bounds
                    .upper()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .upper()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn network_input_duplicate_ids_and_noncanonical_rows_are_rejected() {
        let (graph, cache, input) = fixture();
        let row = [0usize];
        let direct = [HostMultiDepthTarget {
            target_id: 1usize,
            node_name: NETWORK_INPUT,
            selected_neurons: &row,
        }];
        assert!(host_multi_depth_input_relative_bounds(
            &graph, &direct, &cache, &input, None, None, None,
        )
        .is_none());

        let noncanonical_rows = [1usize, 0];
        let noncanonical = [HostMultiDepthTarget {
            target_id: 9usize,
            node_name: "linear0",
            selected_neurons: &noncanonical_rows,
        }];
        assert!(host_multi_depth_input_relative_bounds(
            &graph,
            &noncanonical,
            &cache,
            &input,
            None,
            None,
            None,
        )
        .is_none());

        let duplicate_ids = [
            HostMultiDepthTarget {
                target_id: 7usize,
                node_name: "linear0",
                selected_neurons: &row,
            },
            HostMultiDepthTarget {
                target_id: 7usize,
                node_name: "linear2",
                selected_neurons: &row,
            },
        ];
        assert!(host_multi_depth_input_relative_bounds(
            &graph,
            &duplicate_ids,
            &cache,
            &input,
            None,
            None,
            None,
        )
        .is_none());
    }
}
