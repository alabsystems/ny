// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::{AlphaState, GraphAlphaState};
use std::collections::HashMap;

pub(super) fn relu_names_in_alpha_order(
    relu_name_to_idx: &HashMap<String, usize>,
    expected_len: usize,
) -> Option<Vec<String>> {
    let mut relu_names = vec![String::new(); expected_len];
    for (name, &idx) in relu_name_to_idx {
        if idx >= expected_len || !relu_names[idx].is_empty() {
            return None;
        }
        relu_names[idx] = name.clone();
    }

    if relu_names.iter().any(|name| name.is_empty()) {
        None
    } else {
        Some(relu_names)
    }
}

pub(super) fn graph_alpha_state_from_sequential(
    alpha_state: &AlphaState,
    relu_name_to_idx: &HashMap<String, usize>,
) -> Option<GraphAlphaState> {
    let relu_names = relu_names_in_alpha_order(relu_name_to_idx, alpha_state.alphas.len())?;
    let mut graph_alpha_state = GraphAlphaState::new();

    for (relu_idx, node_name) in relu_names.iter().enumerate() {
        graph_alpha_state
            .alphas
            .insert(node_name.clone(), alpha_state.alphas[relu_idx].clone());
        graph_alpha_state.alphas_upper.insert(
            node_name.clone(),
            alpha_state.alphas_upper[relu_idx].clone(),
        );
        graph_alpha_state.unstable_mask.insert(
            node_name.clone(),
            alpha_state.unstable_mask[relu_idx].clone(),
        );
        graph_alpha_state
            .velocity
            .insert(node_name.clone(), alpha_state.velocity[relu_idx].clone());
        graph_alpha_state
            .adam_m
            .insert(node_name.clone(), alpha_state.adam_m[relu_idx].clone());
        graph_alpha_state
            .adam_v
            .insert(node_name.clone(), alpha_state.adam_v[relu_idx].clone());
        graph_alpha_state.velocity_upper.insert(
            node_name.clone(),
            alpha_state.velocity_upper[relu_idx].clone(),
        );
        graph_alpha_state.adam_m_upper.insert(
            node_name.clone(),
            alpha_state.adam_m_upper[relu_idx].clone(),
        );
        graph_alpha_state.adam_v_upper.insert(
            node_name.clone(),
            alpha_state.adam_v_upper[relu_idx].clone(),
        );
    }

    Some(graph_alpha_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn relu_names_in_alpha_order_reconstructs_contiguous_order() {
        let relu_name_to_idx = HashMap::from([
            ("relu_b".to_string(), 1_usize),
            ("relu_a".to_string(), 0_usize),
            ("relu_c".to_string(), 2_usize),
        ]);

        let relu_names = relu_names_in_alpha_order(&relu_name_to_idx, 3).expect("contiguous order");

        assert_eq!(
            relu_names,
            vec![
                "relu_a".to_string(),
                "relu_b".to_string(),
                "relu_c".to_string()
            ]
        );
    }

    #[test]
    fn relu_names_in_alpha_order_rejects_non_contiguous_indices() {
        let relu_name_to_idx = HashMap::from([
            ("relu_a".to_string(), 0_usize),
            ("relu_b".to_string(), 2_usize),
        ]);

        assert!(
            relu_names_in_alpha_order(&relu_name_to_idx, 2).is_none(),
            "missing index 1 must reject the reconstruction instead of fabricating order"
        );
    }
}
