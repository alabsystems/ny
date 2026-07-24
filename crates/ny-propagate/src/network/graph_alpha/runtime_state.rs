// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::resnet_skeleton::ResnetSegmentSkeleton;
use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{GraphAlphaState, MonotoneSShapedGradients, SqrtGradients};
use crate::invprop::InvpropState;
use ndarray::Array1;
use ny_core::{NyError, Result};
use std::collections::{BTreeMap, HashMap};

/// Verifier-local DAG alpha runtime state.
///
/// `GraphAlphaState` remains the reusable, warm-startable alpha payload.
/// This wrapper holds only per-run verifier state and the current ReLU
/// index adapter used by the DAG optimizer loop.
pub(super) struct DagAlphaRuntimeState {
    graph: GraphAlphaState,
    invprop: Option<InvpropState>,
    relu_nodes: Vec<String>,
    relu_name_to_idx: HashMap<String, usize>,
    /// #root-alpha-gpu (A): the warmup extraction skeleton built once per
    /// dag-alpha optimization loop (`NY_ROOT_ALPHA_GPU=1`). `None` (the
    /// default, and always with the gate off) keeps every warmup site on the
    /// legacy per-iteration extraction — the fail-closed path.
    warmup_skeleton: Option<ResnetSegmentSkeleton>,
}

impl DagAlphaRuntimeState {
    pub(super) fn new(
        graph: GraphAlphaState,
        invprop: Option<InvpropState>,
        relu_nodes: Vec<String>,
    ) -> Self {
        let relu_name_to_idx = relu_nodes
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect();
        Self {
            graph,
            invprop,
            relu_nodes,
            relu_name_to_idx,
            warmup_skeleton: None,
        }
    }

    /// #root-alpha-gpu (A): the loop's warmup segment skeleton, if one was
    /// built. Consumers must treat `None` (or any fold refusal) as "use the
    /// legacy extraction" — never as an error.
    #[must_use]
    pub(super) fn warmup_skeleton(&self) -> Option<&ResnetSegmentSkeleton> {
        self.warmup_skeleton.as_ref()
    }

    pub(super) fn set_warmup_skeleton(&mut self, skeleton: Option<ResnetSegmentSkeleton>) {
        self.warmup_skeleton = skeleton;
    }

    #[must_use]
    pub(super) fn graph(&self) -> &GraphAlphaState {
        &self.graph
    }

    pub(super) fn graph_mut(&mut self) -> &mut GraphAlphaState {
        &mut self.graph
    }

    #[must_use]
    pub(super) fn invprop(&self) -> Option<&InvpropState> {
        self.invprop.as_ref()
    }

    pub(super) fn invprop_mut(&mut self) -> Option<&mut InvpropState> {
        self.invprop.as_mut()
    }

    pub(super) fn clip_gammas(&mut self) {
        if let Some(invprop) = self.invprop.as_mut() {
            invprop.clip_all_gammas();
        }
    }

    #[must_use]
    pub(super) fn relu_nodes(&self) -> &[String] {
        &self.relu_nodes
    }

    #[must_use]
    pub(super) fn relu_name_to_idx(&self) -> &HashMap<String, usize> {
        &self.relu_name_to_idx
    }

    #[must_use]
    pub(super) fn relu_name(&self, relu_idx: usize) -> Option<&str> {
        self.relu_nodes.get(relu_idx).map(String::as_str)
    }

    #[must_use]
    pub(super) fn relu_len(&self, relu_idx: usize) -> Option<usize> {
        self.relu_name(relu_idx)
            .and_then(|node_name| self.graph.relu_len(node_name))
    }

    #[must_use]
    pub(super) fn relu_unstable_mask(&self, relu_idx: usize) -> Option<&Array1<bool>> {
        self.relu_name(relu_idx)
            .and_then(|node_name| self.graph.relu_unstable_mask(node_name))
    }

    pub(super) fn relu_alpha_entry(
        &self,
        relu_idx: usize,
        neuron_idx: usize,
    ) -> Result<(f32, f32)> {
        let node_name = self.relu_name(relu_idx).ok_or_else(|| {
            NyError::InternalError(format!("missing ReLU node for index {relu_idx}"))
        })?;
        let (lower, upper) = self.graph.relu_alpha_pair(node_name).ok_or_else(|| {
            NyError::InternalError(format!(
                "missing GraphAlphaState ReLU alpha pair for '{node_name}'"
            ))
        })?;
        if neuron_idx >= lower.len() || neuron_idx >= upper.len() {
            return Err(NyError::InternalError(format!(
                "relu neuron index {neuron_idx} out of bounds for '{node_name}'"
            )));
        }
        Ok((lower[neuron_idx], upper[neuron_idx]))
    }

    pub(super) fn set_relu_alpha_entry(
        &mut self,
        relu_idx: usize,
        neuron_idx: usize,
        lower: f32,
        upper: f32,
    ) -> Result<()> {
        let node_name = self
            .relu_name(relu_idx)
            .ok_or_else(|| {
                NyError::InternalError(format!("missing ReLU node for index {relu_idx}"))
            })?
            .to_string();
        let (lower_path, upper_path) =
            self.graph.relu_alpha_pair_mut(&node_name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "missing GraphAlphaState ReLU alpha pair for '{node_name}'"
                ))
            })?;
        if neuron_idx >= lower_path.len() || neuron_idx >= upper_path.len() {
            return Err(NyError::InternalError(format!(
                "relu neuron index {neuron_idx} out of bounds for '{node_name}'"
            )));
        }
        lower_path[neuron_idx] = lower.clamp(0.0, 1.0);
        upper_path[neuron_idx] = upper.clamp(0.0, 1.0);
        Ok(())
    }

    #[must_use]
    pub(super) fn snapshot_graph(&self) -> GraphAlphaState {
        self.graph.clone()
    }

    /// Consume the runtime and return the owned `GraphAlphaState`.
    ///
    /// Used by collection paths that need the optimized alpha state
    /// without an extra clone.
    pub(super) fn into_graph_alpha_state(self) -> GraphAlphaState {
        self.graph
    }

    pub(super) fn restore_graph(&mut self, snapshot: &GraphAlphaState) {
        self.graph = snapshot.clone();
    }

    pub(super) fn apply_relu_perturbations(
        &mut self,
        perturbations: &[Array1<f32>],
        eps: f32,
    ) -> Result<()> {
        for (relu_idx, perturbation) in perturbations.iter().enumerate() {
            let node_name = self
                .relu_name(relu_idx)
                .ok_or_else(|| {
                    NyError::InternalError(format!(
                        "missing ReLU node for perturbation index {relu_idx}"
                    ))
                })?
                .to_string();
            let mask = self
                .graph
                .relu_unstable_mask(&node_name)
                .ok_or_else(|| {
                    NyError::InternalError(format!("missing ReLU unstable mask for '{node_name}'"))
                })?
                .clone();
            let (lower, upper) = self.graph.relu_alpha_pair_mut(&node_name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "missing GraphAlphaState ReLU alpha pair for '{node_name}'"
                ))
            })?;
            if perturbation.len() != lower.len() || perturbation.len() != upper.len() {
                return Err(NyError::InternalError(format!(
                    "perturbation length {} != alpha length {} for '{}'",
                    perturbation.len(),
                    lower.len(),
                    node_name
                )));
            }
            for neuron_idx in 0..perturbation.len() {
                if mask[neuron_idx] {
                    let delta = eps * perturbation[neuron_idx];
                    lower[neuron_idx] = (lower[neuron_idx] + delta).clamp(0.0, 1.0);
                    upper[neuron_idx] = (upper[neuron_idx] + delta).clamp(0.0, 1.0);
                }
            }
        }
        Ok(())
    }

    pub(super) fn apply_monotone_perturbations(
        &mut self,
        perturbations: &BTreeMap<String, MonotoneSShapedGradients>,
        eps: f32,
    ) -> Result<()> {
        for (node_name, perturbation) in perturbations {
            let alpha = self
                .graph
                .monotone_s_shaped_alpha_mut(node_name)
                .ok_or_else(|| {
                    NyError::InternalError(format!(
                        "missing monotone alpha bundle for '{node_name}'"
                    ))
                })?;
            alpha.apply_perturbation(perturbation, eps);
        }
        Ok(())
    }

    pub(super) fn apply_sqrt_perturbations(
        &mut self,
        perturbations: &BTreeMap<String, SqrtGradients>,
        eps: f32,
    ) -> Result<()> {
        for (node_name, perturbation) in perturbations {
            let alpha = self.graph.sqrt_alpha_mut(node_name).ok_or_else(|| {
                NyError::InternalError(format!("missing sqrt alpha bundle for '{node_name}'"))
            })?;
            alpha.apply_perturbation(perturbation, eps);
        }
        Ok(())
    }

    pub(super) fn apply_reciprocal_perturbations(
        &mut self,
        perturbations: &BTreeMap<String, ReciprocalGradients>,
        eps: f32,
    ) -> Result<()> {
        for (node_name, perturbation) in perturbations {
            let alpha = self.graph.reciprocal_alpha_mut(node_name).ok_or_else(|| {
                NyError::InternalError(format!("missing reciprocal alpha bundle for '{node_name}'"))
            })?;
            alpha.apply_perturbation(perturbation, eps);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invprop::OutputConstraints;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    fn unit_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("test bounds should construct")
    }

    #[test]
    fn test_dag_alpha_runtime_state_snapshot_restores_relu_and_monotone_state() {
        let mut graph = GraphAlphaState::new();
        let relu_bounds = unit_bounds(&[-1.0, -0.5], &[1.0, 0.5]);
        let sigmoid_bounds = unit_bounds(&[-1.0, -0.2], &[0.7, 1.1]);
        graph
            .add_relu_node("relu_a", &relu_bounds, false)
            .expect("relu init should succeed");
        graph
            .add_sigmoid_node("sigmoid_a", &sigmoid_bounds)
            .expect("sigmoid init should succeed");

        let mut runtime = DagAlphaRuntimeState::new(graph, None, vec!["relu_a".to_string()]);
        let snapshot = runtime.snapshot_graph();

        runtime
            .set_relu_alpha_entry(0, 0, 0.25, 0.75)
            .expect("relu alpha update should succeed");
        let mut monotone_perturb = snapshot
            .monotone_s_shaped_alpha("sigmoid_a")
            .expect("snapshot monotone alpha should exist")
            .zeros_gradients();
        monotone_perturb.tp_pos.lower_path.fill(1.0);
        monotone_perturb.tp_both_upper.upper_path.fill(-1.0);
        let monotone_perts = BTreeMap::from([("sigmoid_a".to_string(), monotone_perturb.clone())]);
        runtime
            .apply_monotone_perturbations(&monotone_perts, 0.1)
            .expect("monotone perturbation should succeed");

        let (mutated_lower, mutated_upper) = runtime
            .relu_alpha_entry(0, 0)
            .expect("mutated relu alpha entry should exist");
        assert!(
            (mutated_lower - snapshot.alpha("relu_a").expect("snapshot lower alpha")[0]).abs()
                > 1e-6
                || (mutated_upper
                    - snapshot
                        .alpha_upper("relu_a")
                        .expect("snapshot upper alpha")[0])
                    .abs()
                    > 1e-6,
            "runtime mutation should change the live graph alpha state before restore"
        );

        runtime.restore_graph(&snapshot);

        let (restored_lower, restored_upper) = runtime
            .relu_alpha_entry(0, 0)
            .expect("restored relu alpha entry should exist");
        assert_eq!(
            restored_lower,
            snapshot.alpha("relu_a").expect("snapshot lower alpha")[0]
        );
        assert_eq!(
            restored_upper,
            snapshot
                .alpha_upper("relu_a")
                .expect("snapshot upper alpha")[0]
        );

        let restored_monotone = runtime
            .graph()
            .monotone_s_shaped_alpha("sigmoid_a")
            .expect("restored monotone alpha should exist");
        let snapshot_monotone = snapshot
            .monotone_s_shaped_alpha("sigmoid_a")
            .expect("snapshot monotone alpha should exist");
        assert_eq!(
            restored_monotone.tp_pos.lower_path,
            snapshot_monotone.tp_pos.lower_path
        );
        assert_eq!(
            restored_monotone.tp_both_upper.upper_path,
            snapshot_monotone.tp_both_upper.upper_path
        );
    }

    #[test]
    fn test_dag_alpha_runtime_state_invprop_not_in_graph_snapshot() {
        let mut graph = GraphAlphaState::new();
        let relu_bounds = unit_bounds(&[-1.0], &[1.0]);
        graph
            .add_relu_node("relu_a", &relu_bounds, false)
            .expect("relu init should succeed");

        let constraints =
            OutputConstraints::new(arr2(&[[1.0_f32]]), arr1(&[0.0_f32]), true).unwrap();
        let invprop = InvpropState::new(constraints, 1);
        let mut runtime =
            DagAlphaRuntimeState::new(graph, Some(invprop), vec!["relu_a".to_string()]);

        let snapshot = runtime.snapshot_graph();
        runtime
            .invprop_mut()
            .expect("runtime invprop should exist")
            .mark_infeasible(0)
            .expect("batch 0 should exist");
        runtime
            .set_relu_alpha_entry(0, 0, 0.1, 0.9)
            .expect("relu alpha update should succeed");
        runtime.restore_graph(&snapshot);

        assert!(
            runtime
                .invprop()
                .expect("runtime invprop should exist")
                .is_infeasible(0),
            "graph snapshot restore must not reset verifier-local invprop state"
        );
        let (restored_lower, restored_upper) = runtime
            .relu_alpha_entry(0, 0)
            .expect("restored relu alpha entry should exist");
        assert_eq!(
            restored_lower,
            snapshot.alpha("relu_a").expect("snapshot lower alpha")[0]
        );
        assert_eq!(
            restored_upper,
            snapshot
                .alpha_upper("relu_a")
                .expect("snapshot upper alpha")[0]
        );
    }

    #[test]
    fn test_dag_alpha_runtime_state_relu_index_adapter_matches_node_names() {
        let mut graph = GraphAlphaState::new();
        let relu_a_bounds = unit_bounds(&[-1.0], &[1.0]);
        let relu_b_bounds = unit_bounds(&[-0.3, -0.1], &[0.9, 0.7]);
        graph
            .add_relu_node("relu_a", &relu_a_bounds, false)
            .expect("relu_a init should succeed");
        graph
            .add_relu_node("relu_b", &relu_b_bounds, false)
            .expect("relu_b init should succeed");

        let mut runtime = DagAlphaRuntimeState::new(
            graph,
            None,
            vec!["relu_b".to_string(), "relu_a".to_string()],
        );
        runtime
            .set_relu_alpha_entry(0, 0, 0.2, 0.8)
            .expect("relu_b entry update should succeed");
        runtime
            .set_relu_alpha_entry(1, 0, 0.6, 0.4)
            .expect("relu_a entry update should succeed");

        assert_eq!(runtime.relu_name_to_idx()["relu_b"], 0);
        assert_eq!(runtime.relu_name_to_idx()["relu_a"], 1);
        assert_eq!(runtime.relu_name(0), Some("relu_b"));
        assert_eq!(runtime.relu_name(1), Some("relu_a"));
        assert_eq!(
            runtime.graph().alpha("relu_b").expect("relu_b lower alpha")[0],
            runtime
                .relu_alpha_entry(0, 0)
                .expect("runtime relu_b entry should exist")
                .0
        );
        assert_eq!(
            runtime
                .graph()
                .alpha_upper("relu_b")
                .expect("relu_b upper alpha")[0],
            runtime
                .relu_alpha_entry(0, 0)
                .expect("runtime relu_b entry should exist")
                .1
        );
        assert_eq!(
            runtime.graph().alpha("relu_a").expect("relu_a lower alpha")[0],
            runtime
                .relu_alpha_entry(1, 0)
                .expect("runtime relu_a entry should exist")
                .0
        );
        assert_eq!(
            runtime
                .graph()
                .alpha_upper("relu_a")
                .expect("relu_a upper alpha")[0],
            runtime
                .relu_alpha_entry(1, 0)
                .expect("runtime relu_a entry should exist")
                .1
        );
    }
}
