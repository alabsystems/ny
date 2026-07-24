// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched IBP pre-screen helpers for input-split BaB.

use ndarray::{Array2, Axis};
use ny_core::{checked_dim_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::BetaCrownConfig;
use crate::GraphNetwork;

use super::grouped_semantics::disjunctive_domain_verified;

/// Run a single batched IBP forward pass for a set of child domains and return
/// their per-child verification status.
///
/// This stacks all child inputs along a fresh leading batch axis, performs one
/// graph IBP pass, then applies the spec matrix independently to each batch row.
/// The graph traversal is shared; only the final scalar/objective reduction is
/// per-child. Part of #4353 Packet A.
pub(crate) fn batched_ibp_prescreen(
    graph: &GraphNetwork,
    child_inputs: &[&BoundedTensor],
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: Option<&[usize]>,
    verify_upper_bound: bool,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<bool>> {
    if child_inputs.is_empty() {
        return Ok(Vec::new());
    }

    validate_prescreen_layout(spec_matrix, thresholds, clause_sizes)?;

    let stacked_input = stack_child_input_refs(child_inputs)?;
    let sanitized = run_ibp_forward(graph, &stacked_input, engine)?;

    evaluate_batched_output(
        &sanitized,
        child_inputs.len(),
        spec_matrix,
        thresholds,
        clause_sizes,
        verify_upper_bound,
    )
}

/// Run graph IBP forward and return sanitized output bounds.
pub(super) fn run_ibp_forward(
    graph: &GraphNetwork,
    stacked_input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    // SOUNDNESS GATE: the batched prescreen stacks independent domains on a fresh
    // leading axis. Graphs with absolute-axis operators (Gather/Concat/Reduce/…)
    // would mis-index across that batch axis once it is prepended — at best an
    // out-of-bounds error, at worst silently mixing bounds across domains and
    // marking an unsafe domain "verified". Refuse to batch such graphs; the caller
    // treats `UnsupportedConfiguration` as a skippable enhancement and falls back to
    // per-domain bounding (which runs each domain unbatched and is unaffected). The
    // prescreen is a pure speed optimization, so skipping it never changes a verdict.
    if !graph.is_input_split_batch_stack_safe() {
        return Err(NyError::UnsupportedConfiguration(
            "input-split batched IBP prescreen skipped: graph has absolute-axis operators that are not batch-stack safe".to_string(),
        ));
    }
    let node_bounds = graph.collect_node_bounds_with_engine(stacked_input, engine)?;
    let output_node_name = output_node_name(graph)?;
    let output_bounds = node_bounds.get(&output_node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Output node '{}' not found for batched IBP prescreen",
            output_node_name
        ))
    })?;
    Ok(GraphNetwork::sanitize_bounds_for_fallback(output_bounds))
}

/// Evaluate batched IBP output into per-child verified mask.
/// Shared by both `batched_ibp_prescreen` and `batched_ibp_prescreen_from_flat`.
pub(super) fn evaluate_batched_output(
    sanitized: &BoundedTensor,
    n: usize,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: Option<&[usize]>,
    verify_upper_bound: bool,
) -> Result<Vec<bool>> {
    let output_shape = sanitized.shape();
    if output_shape.is_empty() {
        return Err(NyError::InvalidSpec(
            "batched_ibp_prescreen expected a leading batch dimension on the output".to_string(),
        ));
    }
    if output_shape[0] != n {
        return Err(NyError::InvalidSpec(format!(
            "batched_ibp_prescreen output batch mismatch: expected {}, got shape {:?}",
            n, output_shape
        )));
    }

    let per_child_dim =
        checked_dim_product(&output_shape[1..], "batched_ibp_prescreen output dimension")?;
    if spec_matrix.ncols() != per_child_dim {
        return Err(NyError::InvalidSpec(format!(
            "batched_ibp_prescreen spec width {} does not match per-child output size {}",
            spec_matrix.ncols(),
            per_child_dim
        )));
    }

    let mut verified = Vec::with_capacity(n);
    for batch_idx in 0..n {
        let lower_values: Vec<f32> = sanitized
            .lower()
            .index_axis(Axis(0), batch_idx)
            .iter()
            .copied()
            .collect();
        let upper_values: Vec<f32> = sanitized
            .upper()
            .index_axis(Axis(0), batch_idx)
            .iter()
            .copied()
            .collect();

        // Shares the per-domain fallback's reduction so the batched mask cannot drift
        // from the per-child baseline it replaces, and inherits its outward rounding:
        // a `true` here permanently drops the child from the BaB queue.
        let row_bounds: Vec<(f32, f32)> = spec_matrix
            .rows()
            .into_iter()
            .map(|spec_row| {
                GraphNetwork::spec_row_interval_bounds(spec_row, &lower_values, &upper_values)
            })
            .collect();

        let is_verified = if let Some(sizes) = clause_sizes {
            disjunctive_domain_verified(&row_bounds, thresholds, sizes)
        } else if thresholds.len() == 1 && row_bounds.len() == 1 {
            let (lower, upper) = row_bounds[0];
            BetaCrownConfig::domain_is_verified_for_mode(
                verify_upper_bound,
                lower,
                upper,
                thresholds[0],
            )
        } else {
            conjunctive_multi_obj_verified(&row_bounds, thresholds)
        };
        verified.push(is_verified);
    }

    Ok(verified)
}

fn stack_child_input_refs(child_inputs: &[&BoundedTensor]) -> Result<BoundedTensor> {
    if child_inputs.is_empty() {
        return Err(NyError::InvalidSpec(
            "Cannot stack empty child input list".to_string(),
        ));
    }

    let first_shape = child_inputs[0].shape();
    for input in child_inputs.iter().skip(1) {
        if input.shape() != first_shape {
            return Err(NyError::shape_mismatch(
                first_shape.to_vec(),
                input.shape().to_vec(),
            ));
        }
    }

    let lower_views: Vec<_> = child_inputs
        .iter()
        .map(|input| input.lower().view())
        .collect();
    let upper_views: Vec<_> = child_inputs
        .iter()
        .map(|input| input.upper().view())
        .collect();
    let lower = ndarray::stack(Axis(0), &lower_views)
        .map_err(|error| NyError::InvalidSpec(format!("Stacking failed: {error}")))?;
    let upper = ndarray::stack(Axis(0), &upper_views)
        .map_err(|error| NyError::InvalidSpec(format!("Stacking failed: {error}")))?;

    BoundedTensor::new(lower, upper)
}

fn output_node_name(graph: &GraphNetwork) -> Result<String> {
    if graph.output_node.is_empty() {
        graph
            .exec_order()?
            .last()
            .cloned()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))
    } else {
        Ok(graph.output_node.clone())
    }
}

pub(super) fn validate_prescreen_layout(
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: Option<&[usize]>,
) -> Result<()> {
    if let Some(clause_sizes) = clause_sizes {
        let total_rows = clause_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "batched_ibp_prescreen clause_sizes {:?} overflow usize",
                    clause_sizes
                ))
            })
        })?;
        if total_rows != spec_matrix.nrows() || total_rows != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "batched_ibp_prescreen grouped layout mismatch: {} spec rows, {} thresholds, clause_sizes {:?}",
                spec_matrix.nrows(),
                thresholds.len(),
                clause_sizes
            )));
        }
        return Ok(());
    }

    if thresholds.len() != 1 && thresholds.len() != spec_matrix.nrows() {
        return Err(NyError::InvalidSpec(format!(
            "batched_ibp_prescreen expected either one threshold or one threshold per spec row; got {} thresholds for {} spec rows",
            thresholds.len(),
            spec_matrix.nrows()
        )));
    }
    Ok(())
}

fn conjunctive_multi_obj_verified(obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> bool {
    obj_bounds
        .iter()
        .zip(thresholds.iter())
        .any(|((lower, _upper), &threshold)| lower.is_finite() && *lower > threshold)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use ny_test_utils::CountingGemmEngine;

    use super::*;
    use crate::beta_crown::engine::graph::input_split::grouped_semantics::disjunctive_domain_verified;
    use crate::beta_crown::engine::graph::input_split::shared::{
        extract_obj_bounds, graph_spec_ibp_fallback,
    };
    use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::{GraphNetwork, GraphNode};

    fn build_prescreen_graph_4353() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.1_f32, -0.4_f32], [0.3_f32, 0.8_f32]]), None)
                    .expect("valid linear1"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(
                LinearLayer::new(arr2(&[[0.7_f32, -0.2_f32]]), Some(arr1(&[0.05_f32])))
                    .expect("valid linear2"),
            ),
            vec!["relu1".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    #[test]
    fn test_batched_ibp_prescreen_matches_single_objective_baseline_4353() {
        let graph = build_prescreen_graph_4353();
        let spec_matrix = arr2(&[[1.0_f32]]);
        let threshold = 0.2_f32;
        let child_a = BoundedTensor::new(
            arr1(&[-0.8_f32, -0.5_f32]).into_dyn(),
            arr1(&[0.4_f32, 0.9_f32]).into_dyn(),
        )
        .expect("valid child_a");
        let child_b = BoundedTensor::new(
            arr1(&[-0.2_f32, -0.7_f32]).into_dyn(),
            arr1(&[0.9_f32, 0.3_f32]).into_dyn(),
        )
        .expect("valid child_b");

        let baseline_engine = CountingGemmEngine::new();
        let baseline: Vec<bool> = [&child_a, &child_b]
            .into_iter()
            .map(|child| {
                let (bounds, _) = graph_spec_ibp_fallback(
                    &graph,
                    child,
                    &spec_matrix,
                    Some(&baseline_engine),
                    None,
                )
                .expect("per-child IBP fallback should succeed");
                BetaCrownConfig::domain_is_verified_for_mode(
                    false,
                    bounds.lower_scalar(),
                    bounds.upper_scalar(),
                    threshold,
                )
            })
            .collect();

        let batched_engine = CountingGemmEngine::new();
        let actual = batched_ibp_prescreen(
            &graph,
            &[&child_a, &child_b],
            &spec_matrix,
            &[threshold],
            None,
            false,
            Some(&batched_engine),
        )
        .expect("batched IBP prescreen should succeed");

        assert_eq!(
            actual, baseline,
            "batched single-objective IBP mask changed"
        );
        assert!(
            batched_engine.gemm_calls() < baseline_engine.gemm_calls(),
            "batched prescreen should reduce GEMM dispatches: batched={}, baseline={}",
            batched_engine.gemm_calls(),
            baseline_engine.gemm_calls()
        );
    }

    #[test]
    fn test_batched_ibp_prescreen_matches_disjunctive_baseline_4353() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("hidden linear")),
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
            vec!["hidden".to_string()],
        ));
        graph.set_output("out");

        let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
        let thresholds = [0.0_f32, 0.0_f32];
        let clause_sizes = [1usize, 1usize];
        let child_a = BoundedTensor::new(arr1(&[0.6_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid child_a");
        let child_b = BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid child_b");

        let baseline_engine = CountingGemmEngine::new();
        let baseline: Vec<bool> = [&child_a, &child_b]
            .into_iter()
            .map(|child| {
                let (bounds, _) = graph_spec_ibp_fallback(
                    &graph,
                    child,
                    &spec_matrix,
                    Some(&baseline_engine),
                    None,
                )
                .expect("per-child grouped IBP fallback should succeed");
                let obj_bounds = extract_obj_bounds(&bounds, thresholds.len()).unwrap();
                disjunctive_domain_verified(&obj_bounds, &thresholds, &clause_sizes)
            })
            .collect();

        let batched_engine = CountingGemmEngine::new();
        let actual = batched_ibp_prescreen(
            &graph,
            &[&child_a, &child_b],
            &spec_matrix,
            &thresholds,
            Some(&clause_sizes),
            false,
            Some(&batched_engine),
        )
        .expect("batched grouped IBP prescreen should succeed");

        assert_eq!(actual, baseline, "batched grouped IBP mask changed");
        assert!(
            batched_engine.gemm_calls() < baseline_engine.gemm_calls(),
            "batched grouped prescreen should reduce GEMM dispatches: batched={}, baseline={}",
            batched_engine.gemm_calls(),
            baseline_engine.gemm_calls()
        );
    }
}
