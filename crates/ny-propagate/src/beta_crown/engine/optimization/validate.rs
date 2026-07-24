// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Validation helpers for optimization-bound inputs.

use super::super::BetaCrownVerifier;
use ny_core::Result;
use ny_tensor::BoundedTensor;
use std::sync::Arc;

use crate::beta_crown::branching::SplitHistory;
use crate::Network;

impl BetaCrownVerifier {
    pub(super) fn validate_layer_bounds_len(
        &self,
        network: &Network,
        layer_bounds: &[Arc<BoundedTensor>],
    ) -> Result<()> {
        let expected = network.layers.len();
        if layer_bounds.len() != expected {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "layer_bounds length {} does not match network layers {}",
                layer_bounds.len(),
                expected
            )));
        }
        Ok(())
    }

    pub(super) fn validate_split_history(
        &self,
        network: &Network,
        input: &BoundedTensor,
        layer_bounds: &[Arc<BoundedTensor>],
        history: &SplitHistory,
    ) -> Result<()> {
        for constraint in &history.constraints {
            if constraint.layer_idx >= network.layers.len() {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "constraint layer_idx {} out of range (layers={})",
                    constraint.layer_idx,
                    network.layers.len()
                )));
            }
            let expected_len = if constraint.layer_idx == 0 {
                input.len()
            } else {
                layer_bounds[constraint.layer_idx - 1].len()
            };
            if constraint.neuron_idx >= expected_len {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "constraint neuron_idx {} out of range for layer {} (len={})",
                    constraint.neuron_idx, constraint.layer_idx, expected_len
                )));
            }
        }
        Ok(())
    }

    pub(super) fn output_dim_from_layer_bounds(
        &self,
        layer_bounds: &[Arc<BoundedTensor>],
        context: &str,
    ) -> Result<usize> {
        layer_bounds.last().map(|b| b.len()).ok_or_else(|| {
            ny_core::NyError::InternalError(format!(
                "{context}: layer_bounds empty after validation"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::NeuronConstraint;
    use crate::{Layer, ReLULayer};
    use ndarray::arr1;

    fn make_network(num_layers: usize) -> Network {
        let mut network = Network::new();
        for _ in 0..num_layers {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
        network
    }

    fn make_bounds(lower: &[f32], upper: &[f32]) -> Arc<BoundedTensor> {
        Arc::new(
            BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
                .expect("test bounds must be valid"),
        )
    }

    fn make_constraint(layer_idx: usize, neuron_idx: usize) -> NeuronConstraint {
        NeuronConstraint::new(layer_idx, neuron_idx, true, 0.0)
            .expect("test constraint must be valid")
    }

    #[test]
    fn test_validate_layer_bounds_len_accepts_matching_count() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(2);
        let layer_bounds = vec![
            make_bounds(&[-1.0, -0.5], &[1.0, 0.5]),
            make_bounds(&[-0.25], &[0.25]),
        ];

        verifier
            .validate_layer_bounds_len(&network, &layer_bounds)
            .expect("matching layer bounds must validate");
    }

    #[test]
    fn test_validate_layer_bounds_len_rejects_mismatch() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(2);
        let layer_bounds = vec![make_bounds(&[-1.0], &[1.0])];

        let err = verifier
            .validate_layer_bounds_len(&network, &layer_bounds)
            .expect_err("mismatched layer bounds length must error");
        assert!(
            matches!(
                err,
                ny_core::NyError::InvalidSpec(ref msg)
                    if msg.contains("layer_bounds length 1 does not match network layers 2")
            ),
            "expected layer-bounds mismatch error, got {err:?}"
        );
    }

    #[test]
    fn test_validate_split_history_accepts_input_and_hidden_constraints() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(3);
        let input = BoundedTensor::new(
            arr1(&[-1.0, -1.0, -1.0]).into_dyn(),
            arr1(&[1.0, 1.0, 1.0]).into_dyn(),
        )
        .expect("test input must be valid");
        let layer_bounds = vec![
            make_bounds(&[-1.0, -1.0], &[1.0, 1.0]),
            make_bounds(&[-0.5], &[0.5]),
            make_bounds(&[-0.25], &[0.25]),
        ];
        let mut history = SplitHistory::new();
        history.add_constraint(make_constraint(0, 2));
        history.add_constraint(make_constraint(2, 0));

        verifier
            .validate_split_history(&network, &input, &layer_bounds, &history)
            .expect("in-range constraints must validate");
    }

    #[test]
    fn test_validate_split_history_rejects_layer_index_out_of_range() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(2);
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .expect("test input must be valid");
        let layer_bounds = vec![make_bounds(&[-1.0], &[1.0]), make_bounds(&[-0.5], &[0.5])];
        let history = SplitHistory::new().with_constraint(make_constraint(2, 0));

        let err = verifier
            .validate_split_history(&network, &input, &layer_bounds, &history)
            .expect_err("out-of-range layer_idx must error");
        assert!(
            matches!(
                err,
                ny_core::NyError::InvalidSpec(ref msg)
                    if msg.contains("constraint layer_idx 2 out of range (layers=2)")
            ),
            "expected layer_idx validation error, got {err:?}"
        );
    }

    #[test]
    fn test_validate_split_history_rejects_input_neuron_index_out_of_range() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(1);
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .expect("test input must be valid");
        let layer_bounds = vec![make_bounds(&[-0.5], &[0.5])];
        let history = SplitHistory::new().with_constraint(make_constraint(0, 2));

        let err = verifier
            .validate_split_history(&network, &input, &layer_bounds, &history)
            .expect_err("input neuron index past input width must error");
        assert!(
            matches!(
                err,
                ny_core::NyError::InvalidSpec(ref msg)
                    if msg.contains("constraint neuron_idx 2 out of range for layer 0 (len=2)")
            ),
            "expected input neuron_idx validation error, got {err:?}"
        );
    }

    #[test]
    fn test_validate_split_history_rejects_hidden_neuron_index_out_of_range() {
        let verifier = BetaCrownVerifier::default();
        let network = make_network(2);
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("test input must be valid");
        let layer_bounds = vec![
            make_bounds(&[-1.0, -0.5], &[1.0, 0.5]),
            make_bounds(&[-0.25], &[0.25]),
        ];
        let history = SplitHistory::new().with_constraint(make_constraint(1, 2));

        let err = verifier
            .validate_split_history(&network, &input, &layer_bounds, &history)
            .expect_err("hidden neuron index past previous layer width must error");
        assert!(
            matches!(
                err,
                ny_core::NyError::InvalidSpec(ref msg)
                    if msg.contains("constraint neuron_idx 2 out of range for layer 1 (len=2)")
            ),
            "expected hidden neuron_idx validation error, got {err:?}"
        );
    }

    #[test]
    fn test_output_dim_from_layer_bounds_returns_last_layer_len() {
        let verifier = BetaCrownVerifier::default();
        let layer_bounds = vec![
            make_bounds(&[-1.0, -0.5], &[1.0, 0.5]),
            make_bounds(&[-0.25, 0.0, 0.25], &[0.25, 0.5, 0.75]),
        ];

        let output_dim = verifier
            .output_dim_from_layer_bounds(&layer_bounds, "test output dim")
            .expect("non-empty layer bounds must have an output dim");

        assert_eq!(output_dim, 3);
    }

    #[test]
    fn test_output_dim_from_layer_bounds_rejects_empty_slice() {
        let verifier = BetaCrownVerifier::default();
        let layer_bounds: Vec<Arc<BoundedTensor>> = Vec::new();

        let err = verifier
            .output_dim_from_layer_bounds(&layer_bounds, "compute_bounds")
            .expect_err("empty layer bounds must error");
        assert!(
            matches!(
                err,
                ny_core::NyError::InternalError(ref msg)
                    if msg.contains("compute_bounds: layer_bounds empty after validation")
            ),
            "expected empty layer-bounds internal error, got {err:?}"
        );
    }
}
