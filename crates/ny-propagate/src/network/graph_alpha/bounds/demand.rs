// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Demand-driven intermediate-bound selection for CROWN-IBP (#3775).
//!
//! Computes which graph nodes need CROWN-IBP tightened bounds based on
//! downstream consumer demand. Nodes that no nonlinear consumer requires
//! keep their forward IBP bounds without attempting CROWN backward.
//!
//! Reference: alpha-beta-CROWN `check_prior_bounds` recursively selects
//! nodes needing intermediate bounds. Source: `auto_LiRPA/bound_general.py:923-968`

use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};

use crate::layers::Layer;
use crate::network::core::graph::NETWORK_INPUT;
use crate::network::core::GraphNetwork;

/// Exact, default-dark diagnostic for alpha-beta-CROWN-style sparse
/// intermediate bounds.
///
/// This reuses the established exact diagnostic gate rather than adding
/// another solver-control surface. It only reports the IBP stability profile
/// of demanded targets; it does not select rows or change a bound.
pub(super) const SPARSE_INTERM_DIAG_ENV: &str = "NY_UNSTABLE_COUNT";
pub(super) const SPARSE_RELU_ROWS_ENV: &str = "NY_CROWN_IBP_SPARSE_RELU_ROWS";

/// Sparse objective rows for an intermediate producer whose bound-demanding
/// consumers are all ReLUs.
///
/// Rows are flat indices into `bounds`. Omitted rows retain their sound IBP
/// enclosure; selected rows are the only rows seeded into backward CROWN.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SparseReluRowPlan {
    selected_rows: Vec<usize>,
    total_rows: usize,
}

impl SparseReluRowPlan {
    pub(super) fn selected_rows(&self) -> &[usize] {
        &self.selected_rows
    }

    pub(super) fn selected_len(&self) -> usize {
        self.selected_rows.len()
    }

    pub(super) fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub(super) fn is_all_stable(&self) -> bool {
        self.selected_rows.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ReluStabilityProfile {
    pub(super) active_stable: usize,
    pub(super) inactive_stable: usize,
    pub(super) unstable: usize,
    pub(super) unresolved: usize,
}

impl ReluStabilityProfile {
    pub(super) fn total(self) -> usize {
        self.active_stable + self.inactive_stable + self.unstable + self.unresolved
    }

    pub(super) fn stable(self) -> usize {
        self.active_stable + self.inactive_stable
    }

    /// Rows a future sparse-intermediate implementation must retain.
    ///
    /// Non-finite, inverted, or otherwise unresolved rows are deliberately
    /// retained. Only finite intervals that already prove one ReLU phase are
    /// candidates for omission.
    pub(super) fn retained(self) -> usize {
        self.unstable + self.unresolved
    }
}

pub(super) fn sparse_interm_diag_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn sparse_interm_diag_enabled() -> bool {
    sparse_interm_diag_from_raw(std::env::var(SPARSE_INTERM_DIAG_ENV).ok().as_deref())
}

pub(super) fn sparse_relu_rows_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn sparse_relu_rows_enabled() -> bool {
    sparse_relu_rows_from_raw(std::env::var(SPARSE_RELU_ROWS_ENV).ok().as_deref())
}

/// Classify target rows using the exact ReLU stability predicate.
///
/// A row is omittable only when finite IBP bounds prove it active (`l >= 0`)
/// or inactive (`u <= 0`). Every non-finite or inverted interval stays in the
/// retained set, so the measurement never overstates a safe sparse row set.
pub(super) fn relu_stability_profile(bounds: &BoundedTensor) -> ReluStabilityProfile {
    let mut profile = ReluStabilityProfile::default();
    for (&lower, &upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            profile.unresolved += 1;
        } else if lower >= 0.0 {
            profile.active_stable += 1;
        } else if upper <= 0.0 {
            profile.inactive_stable += 1;
        } else {
            profile.unstable += 1;
        }
    }
    profile
}

/// Count nonlinear consumers that request this producer's bounds.
///
/// Sparse ReLU intermediate tightening is only a direct semantic match when
/// every requesting consumer is a ReLU. Targets also requested by another
/// relaxation surface remain measurement-only until that operator has its own
/// row-selection proof.
pub(super) fn required_consumer_counts(
    graph: &GraphNetwork,
    target_name: &str,
) -> (/* ReLU */ usize, /* other */ usize) {
    let mut relu = 0usize;
    let mut other = 0usize;
    for consumer in graph.nodes.values() {
        for &index in consumer.layer.required_input_bound_indices() {
            if consumer
                .inputs
                .get(index)
                .is_some_and(|input| input == target_name)
            {
                if matches!(&consumer.layer, Layer::ReLU(_)) {
                    relu += 1;
                } else {
                    other += 1;
                }
            }
        }
    }
    (relu, other)
}

/// Build a sparse-row plan for a structurally eligible intermediate target.
///
/// This follows alpha-beta-CROWN's actual default threshold:
/// `unstable_rows <= 0.9 * total_rows`. The `0.9` is a maximum retained-row
/// fraction, not a requirement that 90% of rows already be stable.
///
/// Eligibility deliberately excludes the final target and every producer with
/// a non-ReLU bound-demanding consumer. Keeping those targets dense avoids
/// dropping useful CROWN correlation tightening on an output row merely
/// because that row itself is sign-stable.
pub(super) fn sparse_relu_row_plan_for_target(
    graph: &GraphNetwork,
    target_name: &str,
    output_name: Option<&str>,
    bounds: &BoundedTensor,
) -> Option<SparseReluRowPlan> {
    if !relu_only_intermediate_target_is_eligible(graph, target_name, output_name) {
        return None;
    }
    sparse_relu_row_plan_for_bounds(bounds)
}

/// Structural half of sparse ReLU-row eligibility.
///
/// Kept separate from the 90% row threshold so a target-complete scheduler can
/// choose exactly one proof-relevant producer first, then fail that same target
/// to the established dense/chunked path when its unresolved-row ratio is too
/// high. Conflating those decisions would make "above 90%" silently select a
/// different node.
pub(super) fn relu_only_intermediate_target_is_eligible(
    graph: &GraphNetwork,
    target_name: &str,
    output_name: Option<&str>,
) -> bool {
    if output_name.is_some_and(|output| output == target_name) {
        return false;
    }
    let (relu_consumers, other_consumers) = required_consumer_counts(graph, target_name);
    relu_consumers > 0 && other_consumers == 0
}

fn sparse_relu_row_plan_for_bounds(bounds: &BoundedTensor) -> Option<SparseReluRowPlan> {
    let selected_rows: Vec<usize> = bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .enumerate()
        .filter_map(|(index, (&lower, &upper))| {
            let unresolved = !lower.is_finite() || !upper.is_finite() || lower > upper;
            (unresolved || (lower < 0.0 && upper > 0.0)).then_some(index)
        })
        .collect();
    let total_rows = bounds.len();
    sparse_relu_threshold_allows(selected_rows.len(), total_rows).then_some(SparseReluRowPlan {
        selected_rows,
        total_rows,
    })
}

fn sparse_relu_threshold_allows(selected_rows: usize, total_rows: usize) -> bool {
    selected_rows <= total_rows
        && (selected_rows as u128).saturating_mul(10) <= (total_rows as u128).saturating_mul(9)
}

/// Identify which nodes need CROWN-IBP tightened bounds.
///
/// A node needs tightened bounds if a downstream layer lists that input index
/// in `required_input_bound_indices()` and the producer's IBP bounds are not
/// already concrete (lower == upper). The graph output is always included so
/// CROWN-IBP preserves the existing contract that exact output nodes still run
/// backward CROWN unless a real fallback fires. Network input is excluded —
/// this is about intermediate node selection, not re-tightening the input
/// domain.
pub(crate) fn nodes_requiring_crown_tightening(
    graph: &GraphNetwork,
    exec_order: &[String],
    ibp_bounds: &HashMap<String, BoundedTensor>,
) -> HashSet<String> {
    let mut needs_bounds = HashSet::new();

    let output_name = if graph.output_name().is_empty() {
        exec_order.last().map(String::as_str)
    } else {
        Some(graph.output_name())
    };
    if let Some(output_name) = output_name.filter(|name| *name != NETWORK_INPUT) {
        needs_bounds.insert(output_name.to_string());
    }

    for node_name in exec_order {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
        let required_indices = node.layer.required_input_bound_indices();
        for &idx in required_indices {
            if let Some(input_name) = node.inputs.get(idx) {
                // Skip the network input — not an intermediate to tighten.
                if input_name == NETWORK_INPUT {
                    continue;
                }
                // Skip producers whose bounds are already concrete.
                if let Some(bounds) = ibp_bounds.get(input_name) {
                    if bounds.lower() == bounds.upper() {
                        continue;
                    }
                }
                needs_bounds.insert(input_name.clone());
            }
        }
    }
    needs_bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{ReLULayer, SigmoidLayer};
    use crate::network::core::GraphNode;
    use ndarray::{Array1, ArrayD, IxDyn};

    #[test]
    fn sparse_interm_diag_gate_is_exact_and_default_dark() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!sparse_interm_diag_from_raw(raw));
        }
        assert!(sparse_interm_diag_from_raw(Some("1")));
    }

    #[test]
    fn relu_stability_profile_retains_only_unstable_or_unresolved_rows() {
        let bounds = BoundedTensor::new(
            Array1::from_vec(vec![0.0, 2.0, -3.0, -2.0, -1.0]).into_dyn(),
            Array1::from_vec(vec![0.0, 4.0, -1.0, 5.0, f32::MIN_POSITIVE]).into_dyn(),
        )
        .expect("valid finite bounds");

        let profile = relu_stability_profile(&bounds);
        assert_eq!(
            profile,
            ReluStabilityProfile {
                active_stable: 2,
                inactive_stable: 1,
                unstable: 2,
                unresolved: 0,
            }
        );
        assert_eq!(profile.total(), 5);
        assert_eq!(profile.stable(), 3);
        assert_eq!(profile.retained(), 2);
    }

    #[test]
    fn sparse_relu_rows_gate_is_exact_and_default_dark() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!sparse_relu_rows_from_raw(raw));
        }
        assert!(sparse_relu_rows_from_raw(Some("1")));
    }

    fn bounds_with_crossing_rows(crossing: usize, stable: usize) -> BoundedTensor {
        let mut lower = vec![-1.0; crossing];
        let mut upper = vec![1.0; crossing];
        lower.extend(std::iter::repeat_n(0.0, stable));
        upper.extend(std::iter::repeat_n(2.0, stable));
        BoundedTensor::new(
            Array1::from_vec(lower).into_dyn(),
            Array1::from_vec(upper).into_dyn(),
        )
        .expect("valid bounds")
    }

    #[test]
    fn sparse_relu_row_threshold_matches_upstream_ninety_percent_semantics() {
        let accepted = sparse_relu_row_plan_for_bounds(&bounds_with_crossing_rows(9, 1))
            .expect("9/10 retained rows engage");
        assert_eq!(accepted.selected_len(), 9);
        assert_eq!(accepted.total_rows(), 10);

        assert!(
            sparse_relu_row_plan_for_bounds(&bounds_with_crossing_rows(10, 1)).is_none(),
            "10/11 retained rows exceed the 90% threshold"
        );
        assert!(
            sparse_relu_row_plan_for_bounds(&bounds_with_crossing_rows(10, 0)).is_none(),
            "a fully unstable target stays dense"
        );

        let all_stable = sparse_relu_row_plan_for_bounds(&bounds_with_crossing_rows(0, 10))
            .expect("all-stable target engages as an empty plan");
        assert!(all_stable.is_all_stable());
    }

    #[test]
    fn sparse_relu_rows_retain_strict_crossings_and_unresolved_intervals() {
        let bounds = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(
                IxDyn(&[6]),
                vec![-1.0, 0.0, -2.0, f32::NEG_INFINITY, -3.0, -0.0],
            )
            .expect("shape"),
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0, 2.0, 0.0, 4.0, f32::INFINITY, 0.0])
                .expect("shape"),
        )
        .expect("valid extended-real bounds");
        let plan = sparse_relu_row_plan_for_bounds(&bounds).expect("3/6 rows engage");
        assert_eq!(plan.selected_rows(), &[0, 3, 4]);
    }

    fn relu_consumer_graph(with_sigmoid_consumer: bool) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("producer", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "relu_consumer",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_string()],
        ));
        if with_sigmoid_consumer {
            graph.add_node(GraphNode::new(
                "sigmoid_consumer",
                Layer::Sigmoid(SigmoidLayer::new()),
                vec!["producer".to_string()],
            ));
        }
        graph.set_output("relu_consumer");
        graph
    }

    #[test]
    fn sparse_relu_rows_require_only_relu_consumers_and_exclude_output() {
        let bounds = bounds_with_crossing_rows(2, 8);
        let relu_only = relu_consumer_graph(false);
        assert!(sparse_relu_row_plan_for_target(
            &relu_only,
            "producer",
            Some("relu_consumer"),
            &bounds
        )
        .is_some());
        assert!(
            sparse_relu_row_plan_for_target(&relu_only, "producer", Some("producer"), &bounds)
                .is_none(),
            "the final target must remain dense even when sign-stable"
        );

        let mixed = relu_consumer_graph(true);
        assert!(
            sparse_relu_row_plan_for_target(&mixed, "producer", Some("relu_consumer"), &bounds)
                .is_none(),
            "a non-ReLU bound-demanding consumer forces dense collection"
        );
    }
}
