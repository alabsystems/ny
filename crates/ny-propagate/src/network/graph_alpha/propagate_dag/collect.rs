// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Collection entry point for DAG α-CROWN that exposes `GraphAlphaState`.
//!
//! Part of #4036: the root alpha collection path delegates here when the
//! configured gradient method is not SPSA, so the DAG optimizer's full
//! gradient dispatch (SPSA, FD, Analytic, AnalyticChain) is used.

use crate::bounds::{AlphaCrownConfig, GraphAlphaState};
use crate::network::core::GraphNetwork;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use super::init::DagAlphaInitResult;
use super::{final_alpha_bound_only_enabled, DagAlphaLoopResultUse};

impl GraphNetwork {
    /// α-CROWN for DAG graphs, returning both optimized bounds and the
    /// `GraphAlphaState` for warm-starting BaB per-domain optimization.
    ///
    /// Uses the same gradient dispatch as `propagate_dag_alpha_crown_with_config_and_engine`
    /// (SPSA, FD, Analytic, or AnalyticChain depending on `config.gradient_method`).
    pub(in crate::network::graph_alpha) fn propagate_dag_alpha_crown_collect_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Option<(BoundedTensor, GraphAlphaState)>> {
        self.propagate_dag_alpha_crown_collect_with_engine_and_gate(
            input,
            config,
            engine,
            final_alpha_bound_only_enabled(),
        )
    }

    /// Gate-injected collection core. Collection always marks the returned
    /// state as observable; the explicit gate argument exists so regression
    /// tests can prove enabled/disabled collection identity without mutating
    /// process-global environment variables.
    fn propagate_dag_alpha_crown_collect_with_engine_and_gate(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        final_bound_only_gate: bool,
    ) -> Result<Option<(BoundedTensor, GraphAlphaState)>> {
        let init = match self.init_dag_alpha_state(input, config, engine)? {
            DagAlphaInitResult::EarlyReturn(_bounds) => return Ok(None),
            DagAlphaInitResult::Ready(state) => *state,
        };
        self.dag_alpha_optimize_loop(
            input,
            config,
            engine,
            init,
            final_bound_only_gate,
            DagAlphaLoopResultUse::BoundsAndState,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::{GradientMethod, Optimizer};
    use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::{arr1, arr2, Array1};
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    fn build_collection_state_escape_dag() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "left_linear",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[1.0_f32, -0.4], [0.7, 0.9]]),
                    Some(arr1(&[0.1_f32, -0.2])),
                )
                .expect("left linear layer should construct"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "left_relu",
            Layer::ReLU(ReLULayer),
            vec!["left_linear".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "right_linear",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[-0.6_f32, 1.1], [0.8, -0.5]]),
                    Some(arr1(&[-0.15_f32, 0.05])),
                )
                .expect("right linear layer should construct"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "right_relu",
            Layer::ReLU(ReLULayer),
            vec!["right_linear".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "residual",
            Layer::Add(AddLayer),
            "left_relu",
            "right_relu",
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.2_f32, -0.9]]), Some(arr1(&[0.07_f32])))
                    .expect("output linear layer should construct"),
            ),
            vec!["residual".to_string()],
        ));
        graph.set_output("output");

        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.7]).into_dyn(),
            arr1(&[1.3_f32, 1.1]).into_dyn(),
        )
        .expect("test input should construct");
        (graph, input)
    }

    fn collection_config(iterations: usize) -> AlphaCrownConfig {
        AlphaCrownConfig {
            iterations,
            gradient_method: GradientMethod::AnalyticChain,
            optimizer: Optimizer::Adam,
            learning_rate: 0.1,
            lr_decay: 0.95,
            tolerance: 0.0,
            early_stop_patience: usize::MAX,
            start_save_best: 0.0,
            fix_interm_bounds: false,
            adaptive_skip: false,
            adaptive_skip_pilot: false,
            ..AlphaCrownConfig::default()
        }
    }

    fn f32_bits(values: &Array1<f32>) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn assert_f32_map_bits_eq(
        enabled: &BTreeMap<String, Array1<f32>>,
        disabled: &BTreeMap<String, Array1<f32>>,
        field: &str,
        iterations: usize,
    ) {
        assert_eq!(
            enabled.keys().collect::<Vec<_>>(),
            disabled.keys().collect::<Vec<_>>(),
            "iterations={iterations}: {field} keys changed under the terminal-bound-only gate"
        );
        for (name, disabled_values) in disabled {
            let enabled_values = enabled
                .get(name)
                .unwrap_or_else(|| panic!("iterations={iterations}: missing {field}[{name}]"));
            assert_eq!(
                f32_bits(enabled_values),
                f32_bits(disabled_values),
                "iterations={iterations}: {field}[{name}] bytes changed under the gate"
            );
        }
    }

    fn assert_collection_state_bits_eq(
        enabled: &GraphAlphaState,
        disabled: &GraphAlphaState,
        iterations: usize,
    ) {
        for (field, enabled_map, disabled_map) in [
            ("alphas", &enabled.alphas, &disabled.alphas),
            (
                "alphas_upper",
                &enabled.alphas_upper,
                &disabled.alphas_upper,
            ),
            ("velocity", &enabled.velocity, &disabled.velocity),
            (
                "velocity_upper",
                &enabled.velocity_upper,
                &disabled.velocity_upper,
            ),
            ("adam_m", &enabled.adam_m, &disabled.adam_m),
            ("adam_v", &enabled.adam_v, &disabled.adam_v),
            (
                "adam_m_upper",
                &enabled.adam_m_upper,
                &disabled.adam_m_upper,
            ),
            (
                "adam_v_upper",
                &enabled.adam_v_upper,
                &disabled.adam_v_upper,
            ),
        ] {
            assert_f32_map_bits_eq(enabled_map, disabled_map, field, iterations);
        }
        assert_eq!(
            enabled.unstable_mask, disabled.unstable_mask,
            "iterations={iterations}: unstable masks changed under the gate"
        );
        assert_eq!(
            enabled.spatial_shapes, disabled.spatial_shapes,
            "iterations={iterations}: spatial alpha metadata changed under the gate"
        );
        assert!(
            enabled.monotone_s_shaped_alphas.is_empty()
                && disabled.monotone_s_shaped_alphas.is_empty()
                && enabled.sqrt_alphas.is_empty()
                && disabled.sqrt_alphas.is_empty()
                && enabled.reciprocal_alphas.is_empty()
                && disabled.reciprocal_alphas.is_empty(),
            "dense ReLU fixture should not create extended alpha state"
        );
    }

    fn assert_bound_bits_eq(
        enabled: &BoundedTensor,
        disabled: &BoundedTensor,
        context: &str,
        iterations: usize,
    ) {
        assert_eq!(
            enabled
                .lower()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            disabled
                .lower()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "iterations={iterations}: {context} lower-bound bytes changed under the gate"
        );
        assert_eq!(
            enabled
                .upper()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            disabled
                .upper()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "iterations={iterations}: {context} upper-bound bytes changed under the gate"
        );
    }

    struct CollectionRun {
        output_bounds: BoundedTensor,
        alpha_state: GraphAlphaState,
        reevaluated_bounds: HashMap<String, BoundedTensor>,
    }

    fn run_collection(iterations: usize, gate_enabled: bool) -> CollectionRun {
        let (graph, input) = build_collection_state_escape_dag();
        let config = collection_config(iterations);
        let (output_bounds, alpha_state) = graph
            .propagate_dag_alpha_crown_collect_with_engine_and_gate(
                &input,
                &config,
                None,
                gate_enabled,
            )
            .expect("DAG alpha collection should succeed")
            .expect("unstable ReLU fixture should enter the DAG optimizer");

        // Mirror `try_dag_gradient_dispatch`: the state returned by collection
        // is immediately consumed by a fresh all-node alpha-CROWN evaluation.
        let exec_order = graph.exec_order().expect("test graph should sort");
        let reference_bounds = graph
            .collect_alpha_reference_bounds_with_engine(&input, &config, None, exec_order)
            .expect("reference bounds should collect");
        let reevaluated_bounds = graph
            .collect_crown_bounds_with_alpha(
                &input,
                &reference_bounds,
                &alpha_state,
                None,
                config.deadline,
            )
            .expect("returned alpha state should re-evaluate");

        CollectionRun {
            output_bounds,
            alpha_state,
            reevaluated_bounds,
        }
    }

    #[ntest::timeout(30000)]
    #[test]
    fn terminal_bound_only_gate_preserves_dag_collection_state_and_reevaluation() {
        for iterations in [1usize, 3] {
            let disabled = run_collection(iterations, false);
            let enabled = run_collection(iterations, true);

            assert_bound_bits_eq(
                &enabled.output_bounds,
                &disabled.output_bounds,
                "optimizer output",
                iterations,
            );
            assert_collection_state_bits_eq(
                &enabled.alpha_state,
                &disabled.alpha_state,
                iterations,
            );

            assert_eq!(
                enabled.reevaluated_bounds.keys().collect::<BTreeSet<_>>(),
                disabled.reevaluated_bounds.keys().collect::<BTreeSet<_>>(),
                "iterations={iterations}: re-evaluated node set changed under the gate"
            );
            for (name, disabled_bound) in &disabled.reevaluated_bounds {
                assert_bound_bits_eq(
                    enabled
                        .reevaluated_bounds
                        .get(name)
                        .unwrap_or_else(|| panic!("missing re-evaluated node {name}")),
                    disabled_bound,
                    &format!("re-evaluated node {name}"),
                    iterations,
                );
            }

            assert!(
                disabled
                    .alpha_state
                    .adam_m
                    .values()
                    .flat_map(|values| values.iter())
                    .any(|value| value.abs() > 1e-8),
                "iterations={iterations}: collection must retain the terminal optimizer update"
            );
        }
    }
}
