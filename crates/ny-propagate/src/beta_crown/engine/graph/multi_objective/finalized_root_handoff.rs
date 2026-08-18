// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Execution-dark custody for one finalized owned multi-objective root.

use ny_tensor::BoundedTensor;

use crate::{GraphNetwork, OwnedSignNormalizedObjectiveSet};

use super::root::MultiObjectiveRootState;

/// The exact finalized host owners at the owned-objective root boundary.
///
/// This private, non-`Clone` value makes the configured graph, finalized root
/// state (including the original root output enclosure), and sign-normalized
/// property one move-owned unit while retaining a real borrow of the exact
/// caller input. It deliberately exposes no observation, receipt, static
/// composition, provider, plan, phase, or execution authority.
///
/// Today construction is followed immediately by the sole infallible legacy
/// decomposition below. A future fallible admission must instead return this
/// exact value on every clean decline, or on any other outcome whose contract
/// permits legacy fallback. It must never silently drop or reconstruct one of
/// these owners before falling back.
#[must_use]
pub(super) struct ResidentBabFinalizedRootHandoffV1<'input> {
    configured_graph: GraphNetwork,
    input: &'input BoundedTensor,
    root: MultiObjectiveRootState,
    property: OwnedSignNormalizedObjectiveSet,
}

impl<'input> ResidentBabFinalizedRootHandoffV1<'input> {
    /// Seal exact owners without validation, allocation, inspection, or a
    /// fallible boundary.
    #[inline]
    pub(super) fn new(
        configured_graph: GraphNetwork,
        input: &'input BoundedTensor,
        root: MultiObjectiveRootState,
        property: OwnedSignNormalizedObjectiveSet,
    ) -> Self {
        #[cfg(test)]
        HANDOFF_CONSTRUCTIONS.with(|count| count.set(count.get().saturating_add(1)));

        Self {
            configured_graph,
            input,
            root,
            property,
        }
    }

    /// Restore the exact established CPU-loop owners.
    ///
    /// The input is intentionally not returned: consuming this handoff ends its
    /// association borrow, and the verifier already retains the same caller
    /// reference for the unchanged legacy loop.
    #[inline]
    pub(super) fn into_legacy_parts(
        self,
    ) -> (
        GraphNetwork,
        MultiObjectiveRootState,
        OwnedSignNormalizedObjectiveSet,
    ) {
        let Self {
            configured_graph,
            input,
            root,
            property,
        } = self;
        let _ = input;
        (configured_graph, root, property)
    }
}

#[cfg(test)]
std::thread_local! {
    static HANDOFF_CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Take and reset this thread's construction count. Production has no observer.
#[cfg(test)]
pub(super) fn take_handoff_constructions_for_test() -> usize {
    HANDOFF_CONSTRUCTIONS.with(|count| count.replace(0))
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::ptr;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ndarray::{arr1, arr2};
    use ny_tensor::{
        BoundedTensorHostAllocationProvenanceV1, BoundedTensorHostAllocationReceiptV1,
    };

    use super::*;
    use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use crate::{BetaCrownConfig, BetaCrownVerifier};

    use super::super::root::{
        evaluate_root, MultiObjectiveProperty, MultiObjectiveRootEvaluation,
        MultiObjectiveRootOutcome, MultiObjectiveRootRequest,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TensorAllocationIdentity {
        lower_pointer: usize,
        upper_pointer: usize,
        lower_capacity: usize,
        upper_capacity: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct GraphAllocationIdentity {
        scope: crate::beta_crown::bab_cuts::CutFoldScope,
        node_order_pointer: usize,
        node_order_capacity: usize,
        linear_name_pointer: usize,
        linear_weight_pointer: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RootAllocationIdentity {
        initial_output: TensorAllocationIdentity,
        linear_bounds_arc: usize,
        input_bounds_arc: usize,
        objective_bounds_pointer: usize,
        objective_bounds_capacity: usize,
        relu_nodes_pointer: usize,
        relu_nodes_capacity: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ObjectiveAllocationIdentity {
        outer_pointer: usize,
        outer_capacity: usize,
        first_row_pointer: usize,
        first_row_capacity: usize,
        second_row_pointer: usize,
        second_row_capacity: usize,
        threshold_pointer: usize,
        threshold_capacity: usize,
    }

    fn accounted(bounds: &BoundedTensor) -> BoundedTensorHostAllocationReceiptV1<'_> {
        match bounds.host_allocation_provenance_v1() {
            BoundedTensorHostAllocationProvenanceV1::Accounted(receipt) => receipt,
            other => panic!("expected accountable test bounds, got {other:?}"),
        }
    }

    fn tensor_identity(bounds: &BoundedTensor) -> TensorAllocationIdentity {
        let receipt = accounted(bounds);
        TensorAllocationIdentity {
            lower_pointer: bounds.lower().as_ptr() as usize,
            upper_pointer: bounds.upper().as_ptr() as usize,
            lower_capacity: receipt.lower_element_capacity(),
            upper_capacity: receipt.upper_element_capacity(),
        }
    }

    fn graph_identity(graph: &GraphNetwork) -> GraphAllocationIdentity {
        let linear = match graph.node("linear").expect("linear node").layer() {
            Layer::Linear(linear) => linear,
            other => panic!("expected Linear, got {}", other.layer_type()),
        };
        GraphAllocationIdentity {
            scope: graph.cut_fold_scope(),
            node_order_pointer: graph.node_order.as_ptr() as usize,
            node_order_capacity: graph.node_order.capacity(),
            linear_name_pointer: graph.node("linear").unwrap().name().as_ptr() as usize,
            linear_weight_pointer: linear.weight().as_ptr() as usize,
        }
    }

    fn root_identity(root: &MultiObjectiveRootState) -> RootAllocationIdentity {
        RootAllocationIdentity {
            initial_output: tensor_identity(&root.initial_output),
            linear_bounds_arc: Arc::as_ptr(
                root.root_domain
                    .node_bounds
                    .get("linear")
                    .expect("linear root bounds"),
            ) as usize,
            input_bounds_arc: Arc::as_ptr(&root.root_domain.input_bounds) as usize,
            objective_bounds_pointer: root.root_domain.objective_bounds.as_ptr() as usize,
            objective_bounds_capacity: root.root_domain.objective_bounds.capacity(),
            relu_nodes_pointer: root.relu_nodes.as_ptr() as usize,
            relu_nodes_capacity: root.relu_nodes.capacity(),
        }
    }

    fn objective_identity(
        property: &OwnedSignNormalizedObjectiveSet,
    ) -> ObjectiveAllocationIdentity {
        let (outer_pointer, outer_capacity, threshold_pointer, threshold_capacity) =
            property.allocation_custody_for_test();
        ObjectiveAllocationIdentity {
            outer_pointer: outer_pointer as usize,
            outer_capacity,
            first_row_pointer: property.rows()[0].as_ptr() as usize,
            first_row_capacity: property.rows()[0].capacity(),
            second_row_pointer: property.rows()[1].as_ptr() as usize,
            second_row_capacity: property.rows()[1].capacity(),
            threshold_pointer: threshold_pointer as usize,
            threshold_capacity,
        }
    }

    fn vec_with_capacity(values: &[f32], capacity: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(capacity);
        result.extend_from_slice(values);
        result
    }

    fn continuing_fixture() -> (
        GraphNetwork,
        BoundedTensor,
        MultiObjectiveRootState,
        OwnedSignNormalizedObjectiveSet,
    ) {
        let mut source_graph = GraphNetwork::new();
        source_graph.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("linear")),
        ));
        source_graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".to_string()],
        ));
        source_graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("output")),
            vec!["relu".to_string()],
        ));
        source_graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("bounded input");
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            timeout: Duration::from_secs(5),
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_pgd_attack: false,
            enable_cuts: false,
            batch_size: 1,
            ..Default::default()
        });
        let configured_graph = verifier.configured_graph_for_crown(&source_graph);

        let mut rows = Vec::with_capacity(7);
        rows.push(vec_with_capacity(&[1.0], 11));
        rows.push(vec_with_capacity(&[1.0], 13));
        let thresholds = vec_with_capacity(&[0.5, 0.5], 17);
        let property = OwnedSignNormalizedObjectiveSet::new(rows, thresholds);
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let MultiObjectiveRootEvaluation { outcome, property } = evaluate_root(
            MultiObjectiveRootRequest {
                verifier: &verifier,
                graph: &configured_graph,
                input: &input,
                property: MultiObjectiveProperty::owned(property),
                engine: None,
                conjunctive: false,
                deadline: None,
            },
            &mut lifecycle,
        )
        .expect("toy root evaluation");
        let root = match outcome {
            MultiObjectiveRootOutcome::Continue(root) => *root,
            MultiObjectiveRootOutcome::Finished(_) => panic!("fixture must continue after root"),
        };
        let property = match property {
            MultiObjectiveProperty::Owned(property) => property,
            MultiObjectiveProperty::Borrowed { .. } => panic!("fixture lost owned property"),
        };

        (configured_graph, input, root, property)
    }

    #[ntest::timeout(15000)]
    #[test]
    fn handoff_and_legacy_unpack_preserve_every_exact_owner_identity() {
        assert_eq!(take_handoff_constructions_for_test(), 0);
        let (configured_graph, input, root, property) = continuing_fixture();
        let decoy_input = input.clone();
        let expected_graph = graph_identity(&configured_graph);
        let expected_root = root_identity(&root);
        let expected_property = objective_identity(&property);

        let handoff =
            ResidentBabFinalizedRootHandoffV1::new(configured_graph, &input, root, property);
        assert_eq!(take_handoff_constructions_for_test(), 1);
        assert!(ptr::eq(ptr::from_ref(handoff.input), ptr::from_ref(&input)));
        assert!(!ptr::eq(
            ptr::from_ref(handoff.input),
            ptr::from_ref(&decoy_input)
        ));
        assert_eq!(graph_identity(&handoff.configured_graph), expected_graph);
        assert_eq!(root_identity(&handoff.root), expected_root);
        assert_eq!(objective_identity(&handoff.property), expected_property);

        let (configured_graph, root, property) = handoff.into_legacy_parts();
        assert_eq!(graph_identity(&configured_graph), expected_graph);
        assert_eq!(root_identity(&root), expected_root);
        assert_eq!(objective_identity(&property), expected_property);
    }

    #[ntest::timeout(15000)]
    #[test]
    fn unwind_drops_owned_graph_and_root_but_not_borrowed_input() {
        assert_eq!(take_handoff_constructions_for_test(), 0);
        let (configured_graph, input, root, property) = continuing_fixture();
        let graph_scope = Arc::downgrade(&configured_graph.crown_degradation_log_scope);
        let root_bounds = Arc::downgrade(
            root.root_domain
                .node_bounds
                .get("linear")
                .expect("linear root bounds"),
        );
        let input_identity = tensor_identity(&input);

        let unwind = catch_unwind(AssertUnwindSafe(|| {
            let _handoff =
                ResidentBabFinalizedRootHandoffV1::new(configured_graph, &input, root, property);
            panic!("intentional handoff unwind");
        }));

        assert!(unwind.is_err());
        assert_eq!(take_handoff_constructions_for_test(), 1);
        assert!(graph_scope.upgrade().is_none());
        assert!(root_bounds.upgrade().is_none());
        assert_eq!(tensor_identity(&input), input_identity);
    }
}
