// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dead neuron analysis via IBP bounds.
//!
//! Runs IBP forward through a sequential network and classifies neurons
//! at each ReLU activation layer based on their pre-activation bounds:
//! - **Dead**: upper bound <= 0 (always inactive, output is always 0)
//! - **AlwaysActive**: lower bound >= 0 (linear pass-through)
//! - **Constant**: lower ~= upper (output is approximately fixed)
//! - **Unstable**: could be active or inactive (straddles zero)
//!
//! This is the first step toward compiler-style optimization passes that
//! eliminate provably dead or constant neurons from the network.
//!
//! Reference: alpha-beta-CROWN uses pre-activation bounds to determine
//! ReLU stability. See `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/operators/relu.py`.

use crate::layers::Layer;
use crate::network::Network;
use ny_core::Result;
use ny_tensor::BoundedTensor;

/// Classification of a neuron's behavior based on IBP pre-activation bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NeuronStatus {
    /// Upper bound <= 0: ReLU always outputs 0 (dead neuron).
    Dead,
    /// Lower bound >= 0: ReLU is identity (always active).
    AlwaysActive,
    /// |upper - lower| < epsilon: output is approximately constant.
    Constant(f32),
    /// Neuron straddles zero: could be active or inactive.
    Unstable,
}

/// Analysis of a single neuron at a ReLU activation layer.
#[derive(Debug, Clone)]
pub struct NeuronAnalysis {
    /// Index of the ReLU layer in the network.
    pub layer_index: usize,
    /// Flat index of the neuron within the layer output.
    pub neuron_index: usize,
    /// Classification of this neuron's behavior.
    pub status: NeuronStatus,
    /// Pre-activation lower bound.
    pub lower_bound: f32,
    /// Pre-activation upper bound.
    pub upper_bound: f32,
}

/// Result of analyzing all neurons in a sequential network.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Per-neuron analysis entries (only for neurons at ReLU layers).
    pub neurons: Vec<NeuronAnalysis>,
    /// Total number of neurons analyzed (across all ReLU layers).
    pub total_neurons: usize,
    /// Number of dead neurons (upper <= 0).
    pub dead_count: usize,
    /// Number of always-active neurons (lower >= 0).
    pub always_active_count: usize,
    /// Number of approximately-constant neurons.
    pub constant_count: usize,
    /// Number of unstable neurons (straddling zero).
    pub unstable_count: usize,
}

impl AnalysisResult {
    /// Fraction of neurons that are provably stable (dead or always-active).
    #[must_use]
    pub fn stable_fraction(&self) -> f32 {
        if self.total_neurons == 0 {
            return 1.0;
        }
        (self.dead_count + self.always_active_count) as f32 / self.total_neurons as f32
    }
}

/// Default threshold for classifying a neuron as constant.
const DEFAULT_CONSTANT_EPSILON: f32 = 1e-6;

/// Classify a single neuron based on its pre-activation bounds.
///
/// Pre-activation bounds are the bounds of the input to a ReLU layer
/// (i.e., the output of the preceding linear/conv layer).
///
/// Classification logic:
/// - upper <= 0 => Dead (ReLU always outputs 0)
/// - lower >= 0 => AlwaysActive (ReLU is identity)
/// - |upper - lower| < epsilon => Constant (midpoint)
/// - otherwise => Unstable
#[must_use]
fn classify_neuron(lower: f32, upper: f32, epsilon: f32) -> NeuronStatus {
    if upper <= 0.0 {
        NeuronStatus::Dead
    } else if lower >= 0.0 {
        NeuronStatus::AlwaysActive
    } else if (upper - lower).abs() < epsilon {
        // lower < 0 < upper here, so the sum cannot overflow: midpoint == (l+u)/2 exactly.
        NeuronStatus::Constant(f32::midpoint(lower, upper))
    } else {
        NeuronStatus::Unstable
    }
}

/// Analyze a sequential network for dead/constant/always-active neurons.
///
/// Runs IBP forward through the network, then examines the pre-activation
/// bounds at each ReLU layer. The pre-activation bounds for ReLU at layer
/// index `i` are the output bounds of layer `i-1` (the preceding linear layer).
///
/// # Arguments
///
/// * `network` - The sequential network to analyze
/// * `input` - Input bounds (BoundedTensor defining the input domain)
///
/// # Returns
///
/// An `AnalysisResult` with per-neuron classifications and aggregate statistics.
///
/// # Errors
///
/// Returns an error if IBP propagation fails (e.g., shape mismatch).
pub fn analyze_neurons(network: &Network, input: &BoundedTensor) -> Result<AnalysisResult> {
    analyze_neurons_with_epsilon(network, input, DEFAULT_CONSTANT_EPSILON)
}

/// Analyze a sequential network with a custom constant-detection threshold.
///
/// See [`analyze_neurons`] for details. The `epsilon` parameter controls
/// how close lower and upper bounds must be for a neuron to be classified
/// as `Constant`.
pub fn analyze_neurons_with_epsilon(
    network: &Network,
    input: &BoundedTensor,
    epsilon: f32,
) -> Result<AnalysisResult> {
    let layers = network.layers();
    let ibp_bounds = network.collect_ibp_bounds(input)?;

    let mut neurons = Vec::new();
    let mut dead_count = 0usize;
    let mut always_active_count = 0usize;
    let mut constant_count = 0usize;
    let mut unstable_count = 0usize;

    for (layer_idx, layer) in layers.iter().enumerate() {
        // We only analyze ReLU layers.
        if !matches!(layer, Layer::ReLU(_)) {
            continue;
        }

        // Pre-activation bounds = output of the previous layer.
        // For layer 0, the pre-activation bounds are the input.
        let pre_activation = if layer_idx == 0 {
            input
        } else {
            // ibp_bounds[i] = output of layer i, so pre-activation for layer i
            // is the output of layer i-1, which is ibp_bounds[i-1].
            &ibp_bounds[layer_idx - 1]
        };

        let lower_flat = pre_activation.lower().iter().copied().collect::<Vec<_>>();
        let upper_flat = pre_activation.upper().iter().copied().collect::<Vec<_>>();

        for (neuron_idx, (&l, &u)) in lower_flat.iter().zip(upper_flat.iter()).enumerate() {
            let status = classify_neuron(l, u, epsilon);
            match status {
                NeuronStatus::Dead => dead_count += 1,
                NeuronStatus::AlwaysActive => always_active_count += 1,
                NeuronStatus::Constant(_) => constant_count += 1,
                NeuronStatus::Unstable => unstable_count += 1,
            }
            neurons.push(NeuronAnalysis {
                layer_index: layer_idx,
                neuron_index: neuron_idx,
                status,
                lower_bound: l,
                upper_bound: u,
            });
        }
    }

    let total_neurons = dead_count + always_active_count + constant_count + unstable_count;

    Ok(AnalysisResult {
        neurons,
        total_neurons,
        dead_count,
        always_active_count,
        constant_count,
        unstable_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    /// Build a 2-layer network: Linear(2->4) -> ReLU -> Linear(4->1)
    /// with weights chosen so that for input in [-1, 1]^2:
    /// - Neuron 0: pre-activation always positive (always active)
    /// - Neuron 1: pre-activation always negative (dead)
    /// - Neuron 2: pre-activation straddles zero (unstable)
    /// - Neuron 3: pre-activation always positive (always active)
    #[ntest::timeout(10000)]
    #[test]
    fn test_dead_neuron_analysis_basic() {
        let mut network = Network::new();

        // Linear layer: W * x + b
        // For input x in [-1, 1]^2:
        //
        // Neuron 0: w=[2, 0], b=3 => output in [1, 5] (always positive)
        // Neuron 1: w=[0, -2], b=-3 => output in [-5, -1] (always negative)
        // Neuron 2: w=[1, 1], b=0 => output in [-2, 2] (unstable)
        // Neuron 3: w=[0, 0], b=5 => output in [5, 5] (always active, constant-ish)
        let weights = arr2(&[[2.0, 0.0], [0.0, -2.0], [1.0, 1.0], [0.0, 0.0]]);
        let bias = arr1(&[3.0, -3.0, 0.0, 5.0]);
        network.add_layer(Layer::Linear(
            LinearLayer::new(weights, Some(bias)).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 1.0, 1.0, 1.0]]), None).unwrap(),
        ));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();

        assert_eq!(result.total_neurons, 4);
        assert_eq!(result.always_active_count, 2); // neurons 0 and 3
        assert_eq!(result.dead_count, 1); // neuron 1
        assert_eq!(result.unstable_count, 1); // neuron 2
        assert_eq!(result.constant_count, 0);

        // Verify individual neurons
        assert_eq!(result.neurons[0].status, NeuronStatus::AlwaysActive);
        assert_eq!(result.neurons[0].layer_index, 1); // ReLU is layer 1
        assert_eq!(result.neurons[0].neuron_index, 0);

        assert_eq!(result.neurons[1].status, NeuronStatus::Dead);
        assert_eq!(result.neurons[1].neuron_index, 1);

        assert_eq!(result.neurons[2].status, NeuronStatus::Unstable);
        assert_eq!(result.neurons[2].neuron_index, 2);

        assert_eq!(result.neurons[3].status, NeuronStatus::AlwaysActive);
        assert_eq!(result.neurons[3].neuron_index, 3);
    }

    /// Test with a deeper network: Linear -> ReLU -> Linear -> ReLU
    /// The second ReLU's pre-activation bounds depend on the first ReLU's output.
    #[ntest::timeout(10000)]
    #[test]
    fn test_dead_neuron_analysis_deep_network() {
        let mut network = Network::new();

        // Layer 0: Linear(1->2), w=[[10], [-10]], b=[0, 0]
        // For input x in [0.5, 1.0]:
        //   neuron 0: [5, 10] (always positive)
        //   neuron 1: [-10, -5] (always negative)
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[10.0], [-10.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));
        // Layer 1: ReLU -> after ReLU: neuron 0: [5, 10], neuron 1: [0, 0]
        network.add_layer(Layer::ReLU(ReLULayer));

        // Layer 2: Linear(2->2), w=[[1, 1], [-1, -1]], b=[0, 0]
        // Input: [5, 10] x [0, 0] = (neuron0_post_relu, neuron1_post_relu)
        //   neuron 0: 1*[5,10] + 1*[0,0] = [5, 10] (always positive)
        //   neuron 1: -1*[5,10] + -1*[0,0] = [-10, -5] (always negative)
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 1.0], [-1.0, -1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));
        // Layer 3: ReLU
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();

        // 4 total ReLU neurons: 2 at layer 1, 2 at layer 3
        assert_eq!(result.total_neurons, 4);

        // Layer 1 (first ReLU): neuron 0 always active, neuron 1 dead
        let layer1_neurons: Vec<_> = result
            .neurons
            .iter()
            .filter(|n| n.layer_index == 1)
            .collect();
        assert_eq!(layer1_neurons.len(), 2);
        assert_eq!(layer1_neurons[0].status, NeuronStatus::AlwaysActive);
        assert_eq!(layer1_neurons[1].status, NeuronStatus::Dead);

        // Layer 3 (second ReLU): neuron 0 always active, neuron 1 dead
        let layer3_neurons: Vec<_> = result
            .neurons
            .iter()
            .filter(|n| n.layer_index == 3)
            .collect();
        assert_eq!(layer3_neurons.len(), 2);
        assert_eq!(layer3_neurons[0].status, NeuronStatus::AlwaysActive);
        assert_eq!(layer3_neurons[1].status, NeuronStatus::Dead);

        // All neurons are stable
        assert_eq!(result.dead_count, 2);
        assert_eq!(result.always_active_count, 2);
        assert_eq!(result.unstable_count, 0);
        assert!((result.stable_fraction() - 1.0).abs() < f32::EPSILON);
    }

    /// Network with no ReLU layers produces empty analysis.
    #[ntest::timeout(5000)]
    #[test]
    fn test_analysis_no_relu_layers() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None).unwrap(),
        ));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result.total_neurons, 0);
        assert_eq!(result.neurons.len(), 0);
        assert!((result.stable_fraction() - 1.0).abs() < f32::EPSILON);
    }

    /// Test that all-dead neurons are correctly identified.
    #[ntest::timeout(5000)]
    #[test]
    fn test_all_dead_neurons() {
        let mut network = Network::new();

        // Linear: all outputs guaranteed negative for input in [0, 1]
        // w=[-5], b=-1 => output in [-6, -1]
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[-5.0]]), Some(arr1(&[-1.0]))).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result.total_neurons, 1);
        assert_eq!(result.dead_count, 1);
        assert_eq!(result.always_active_count, 0);
        assert_eq!(result.unstable_count, 0);

        // Verify bounds
        assert!(result.neurons[0].upper_bound <= 0.0);
    }

    /// Test that all-active neurons are correctly identified.
    #[ntest::timeout(5000)]
    #[test]
    fn test_all_active_neurons() {
        let mut network = Network::new();

        // Linear: all outputs guaranteed positive for input in [0, 1]
        // w=[5], b=1 => output in [1, 6]
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[5.0]]), Some(arr1(&[1.0]))).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result.total_neurons, 1);
        assert_eq!(result.always_active_count, 1);
        assert_eq!(result.dead_count, 0);
        assert_eq!(result.unstable_count, 0);

        // Verify bounds
        assert!(result.neurons[0].lower_bound >= 0.0);
    }

    /// Test the constant neuron detection with custom epsilon.
    #[ntest::timeout(5000)]
    #[test]
    fn test_constant_neuron_detection() {
        let mut network = Network::new();

        // Linear: w=[0], b=0 => output is always exactly 0
        // Pre-activation bounds: [0, 0] — this is lower >= 0, so AlwaysActive
        // To get Constant classification, we need lower < 0 < upper with tiny width.
        // Use very small negative bias:
        // w=[0.0001], b=-0.00005 => for input in [0, 1]: output in [-0.00005, 0.00005]
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[0.0001]]), Some(arr1(&[-0.00005]))).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();

        // With a large epsilon, this should be classified as Constant
        let result = analyze_neurons_with_epsilon(&network, &input, 1.0).unwrap();
        assert_eq!(result.total_neurons, 1);
        assert_eq!(result.constant_count, 1);
        if let NeuronStatus::Constant(val) = result.neurons[0].status {
            // Midpoint should be approximately 0
            assert!(val.abs() < 0.001);
        } else {
            panic!(
                "Expected Constant status, got {:?}",
                result.neurons[0].status
            );
        }

        // With default epsilon, the width 0.0001 > 1e-6, but also straddles zero => Unstable
        let result_default = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result_default.unstable_count, 1);
    }

    /// Test stable_fraction for a mixed network.
    #[ntest::timeout(5000)]
    #[test]
    fn test_stable_fraction() {
        let mut network = Network::new();

        // 3 neurons: 1 dead, 1 active, 1 unstable
        let weights = arr2(&[[0.0], [0.0], [1.0]]);
        let bias = arr1(&[-1.0, 1.0, 0.0]);
        network.add_layer(Layer::Linear(
            LinearLayer::new(weights, Some(bias)).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result.total_neurons, 3);
        assert_eq!(result.dead_count, 1);
        assert_eq!(result.always_active_count, 1);
        assert_eq!(result.unstable_count, 1);

        // stable = (1 dead + 1 active) / 3 total = 2/3
        let expected = 2.0 / 3.0;
        assert!((result.stable_fraction() - expected).abs() < 1e-6);
    }

    /// ReLU as the first layer: pre-activation bounds come from the input directly.
    #[ntest::timeout(5000)]
    #[test]
    fn test_relu_as_first_layer() {
        let mut network = Network::new();
        network.add_layer(Layer::ReLU(ReLULayer));

        // Input with 3 neurons: one always positive, one always negative, one unstable
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -2.0, -0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, -0.5, 0.5]).unwrap(),
        )
        .unwrap();

        let result = analyze_neurons(&network, &input).unwrap();
        assert_eq!(result.total_neurons, 3);
        assert_eq!(result.always_active_count, 1);
        assert_eq!(result.dead_count, 1);
        assert_eq!(result.unstable_count, 1);
    }
}
