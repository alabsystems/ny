// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness provenance scanning for verification runs.
//!
//! This module implements a coarse-grained preflight scan over a network's layers to surface
//! known heuristic/unsound switches (e.g. sampling-based relaxations). Downstream consumers can
//! treat `SoundnessProvenance.mode == Sound` as a *label* for
//! "no known heuristics were used",
//! not a proof of soundness.

use ny_core::{HeuristicUsed, NyError, Result, SoundnessProvenance};

use ny_tensor::BoundedTensor;

use crate::layers::BoundPropagation;
use crate::{GraphNetwork, Layer, Network, PropagationMethod, NETWORK_INPUT};

fn bounds_has_negative_lower(bounds: &BoundedTensor) -> bool {
    bounds.lower().iter().any(|&v| v < 0.0)
}

pub fn count_sqrt_negative_domain_network(
    network: &Network,
    input: &BoundedTensor,
) -> Result<usize> {
    if !network
        .layers
        .iter()
        .any(|layer| matches!(layer, Layer::Sqrt(_)))
    {
        return Ok(0);
    }
    if network.has_self_attention() {
        return Err(NyError::UnsupportedConfiguration(
            "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
        ));
    }
    let mut count = 0usize;
    let mut current = input.clone();
    for (i, layer) in network.layers.iter().enumerate() {
        if matches!(layer, Layer::Sqrt(_)) && bounds_has_negative_lower(&current) {
            count += 1;
        }
        current = match layer {
            Layer::Sqrt(sqrt) => sqrt.propagate_ibp_lenient(&current),
            _ => layer.propagate_ibp(&current),
        }
        .map_err(|e| NyError::LayerError {
            layer_index: i,
            layer_type: layer.layer_type().to_string(),
            source: Box::new(e),
        })?;
    }
    Ok(count)
}

pub fn count_sqrt_negative_domain_graph(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<usize> {
    if !graph
        .nodes
        .values()
        .any(|node| matches!(node.layer, Layer::Sqrt(_)))
    {
        return Ok(0);
    }
    let node_bounds = graph.collect_node_bounds_allowing_negative_sqrt(input)?;
    count_sqrt_negative_domain_from_bounds(graph, input, &node_bounds)
}

/// Count sqrt nodes receiving negative-domain inputs using pre-computed per-node IBP bounds.
///
/// This avoids redundant IBP propagation when the caller has already computed
/// per-node bounds via [`GraphNetwork::collect_node_bounds`]. Used by external
/// verifier consumers to eliminate O(G) overhead in `run_escalation()`.
pub fn count_sqrt_negative_domain_from_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
) -> Result<usize> {
    if !graph
        .nodes
        .values()
        .any(|node| matches!(node.layer, Layer::Sqrt(_)))
    {
        return Ok(0);
    }
    let mut count = 0usize;
    for node_name in &graph.node_order {
        let node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Node not found during sqrt scan: {}", node_name))
        })?;
        if matches!(node.layer, Layer::Sqrt(_)) {
            let input_name = node.inputs.first().ok_or_else(|| {
                NyError::InvalidSpec(format!("Sqrt node {} has no inputs", node_name))
            })?;
            let input_bounds = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Missing bounds for sqrt input {} in node {}",
                        input_name, node_name
                    ))
                })?
            };
            if bounds_has_negative_lower(input_bounds) {
                count += 1;
            }
        }
    }
    Ok(count)
}

pub fn soundness_provenance_for_network(
    network: &Network,
    actual_method: &PropagationMethod,
) -> SoundnessProvenance {
    soundness_provenance_for_layers(network.layers.iter(), actual_method)
}

pub fn soundness_provenance_for_graph(
    graph: &GraphNetwork,
    actual_method: &PropagationMethod,
) -> SoundnessProvenance {
    let layers = graph
        .node_names()
        .iter()
        .filter_map(|name| graph.node(name))
        .map(|node| &node.layer);
    soundness_provenance_for_layers(layers, actual_method)
}

/// Scan layers for heuristic/unsound switches and produce a soundness provenance report.
///
/// Uses `PropagationMethod` enum instead of string matching to prevent
/// silent mismatches when new variants are added. Ref: #2807.
pub(crate) fn soundness_provenance_for_layers<'a, I>(
    layers: I,
    actual_method: &PropagationMethod,
) -> SoundnessProvenance
where
    I: Iterator<Item = &'a Layer>,
{
    let crown_family = matches!(
        actual_method,
        PropagationMethod::Crown
            | PropagationMethod::AlphaCrown
            | PropagationMethod::BetaCrown
            // SdpCrown here is dormant: every dispatch site (network/graph/parallel)
            // refuses SDP-CROWN over ℓ∞ box specs before provenance is computed, so
            // no SdpCrown verdict reaches this classification today. If an SDP-CROWN
            // execution path is reintroduced (e.g. genuine ℓ2-ball specs), re-review
            // whether its bounds warrant unqualified Sound provenance before relying
            // on this arm.
            | PropagationMethod::SdpCrown
    );

    let mut layernorm_forward_mode_nodes = 0usize;
    let mut rmsnorm_forward_mode_nodes = 0usize;
    let mut groupnorm_forward_mode_nodes = 0usize;
    let mut instancenorm_forward_mode_nodes = 0usize;
    let mut adain_forward_mode_nodes = 0usize;
    let mut layernorm_sampling_nodes = 0usize;
    let mut softmax_sampling_nodes = 0usize;
    let mut causal_softmax_sampling_nodes = 0usize;
    let mut logsoftmax_sampling_nodes = 0usize;
    let mut reduce_extremum_fixed_index_nodes = 0usize;
    // Other sampling-based relaxations (GELU/Sin/Cos) share a coarse-grained marker.
    let mut sampling_relaxations_present = false;

    for layer in layers {
        match layer {
            Layer::LayerNorm(ln) => {
                if ln.forward_mode {
                    layernorm_forward_mode_nodes += 1;
                }
                if ln.crown_mode == crate::layers::LayerNormCrownMode::Sampling {
                    layernorm_sampling_nodes += 1;
                }
            }
            Layer::RmsNorm(rn) => {
                if rn.forward_mode {
                    rmsnorm_forward_mode_nodes += 1;
                }
                if rn.crown_mode == crate::layers::LayerNormCrownMode::Sampling {
                    layernorm_sampling_nodes += 1;
                }
            }
            Layer::InstanceNorm1d(inn) => {
                if inn.forward_mode {
                    instancenorm_forward_mode_nodes += 1;
                }
                if inn.crown_mode == crate::layers::LayerNormCrownMode::Sampling {
                    layernorm_sampling_nodes += 1;
                }
            }
            Layer::GroupNorm(gn) => {
                if gn.forward_mode {
                    groupnorm_forward_mode_nodes += 1;
                }
                if gn.crown_mode == crate::layers::LayerNormCrownMode::Sampling {
                    layernorm_sampling_nodes += 1;
                }
            }
            Layer::AdaIN1d(adain) => {
                if adain.instance_norm.forward_mode {
                    adain_forward_mode_nodes += 1;
                }
                if adain.instance_norm.crown_mode == crate::layers::LayerNormCrownMode::Sampling {
                    layernorm_sampling_nodes += 1;
                }
            }
            Layer::Softmax(softmax) => {
                if !softmax.sound {
                    softmax_sampling_nodes += 1;
                }
            }
            Layer::CausalSoftmax(softmax) => {
                if !softmax.sound {
                    causal_softmax_sampling_nodes += 1;
                }
            }
            Layer::LogSoftmax(ls) => {
                if !ls.sound {
                    logsoftmax_sampling_nodes += 1;
                }
            }
            Layer::ReduceMax(_) | Layer::ReduceMin(_) => {
                reduce_extremum_fixed_index_nodes += 1;
            }
            Layer::GELU(gelu) => {
                // GELU with sound=true uses precomputed tangent tables instead of sampling,
                // so it doesn't trigger the heuristic flag (Erf or Tanh approximation).
                if !gelu.is_sound() {
                    sampling_relaxations_present = true;
                }
            }
            Layer::Sin(sin) => {
                if !sin.sound {
                    sampling_relaxations_present = true;
                }
            }
            Layer::Cos(cos) => {
                if !cos.sound {
                    sampling_relaxations_present = true;
                }
            }
            // Exhaustive listing of layers with no heuristic flags.
            // Adding a new Layer variant will produce a compile error here,
            // forcing explicit consideration of whether it needs heuristic scanning.
            // See: #2475
            Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::ReLU(_)
            | Layer::LeakyReLU(_)
            | Layer::Clip(_)
            | Layer::Elu(_)
            | Layer::Selu(_)
            | Layer::PRelu(_)
            | Layer::HardSigmoid(_)
            | Layer::HardSwish(_)
            | Layer::SiLU(_)
            | Layer::Exp(_)
            | Layer::Log(_)
            | Layer::Celu(_)
            | Layer::Mish(_)
            | Layer::LogSumExp(_)
            | Layer::ThresholdedRelu(_)
            | Layer::Shrink(_)
            | Layer::Softsign(_)
            | Layer::Snake(_)
            | Layer::SelfAttention(_)
            | Layer::BatchNorm(_)
            | Layer::MatMul(_)
            | Layer::MulBinary(_)
            | Layer::Add(_)
            | Layer::Concat(_)
            | Layer::Sub(_)
            | Layer::Div(_)
            | Layer::Atan2(_)
            | Layer::BilinearCrown(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
            | Layer::AddConstant(_)
            | Layer::Transpose(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::MulConstant(_)
            | Layer::Abs(_)
            | Layer::Sqrt(_)
            | Layer::DivConstant(_)
            | Layer::SubConstant(_)
            | Layer::PowConstant(_)
            | Layer::ReduceMean(_)
            | Layer::ReduceSum(_)
            | Layer::CumSum(_)
            | Layer::Topk(_)
            | Layer::ArgMax(_)
            | Layer::ArgMin(_)
            | Layer::ArgSort(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Erf(_)
            | Layer::Softplus(_)
            | Layer::Tan(_)
            | Layer::Arctan(_)
            | Layer::Tile(_)
            | Layer::Gather(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::ScatterNd(_)
            | Layer::Pad(_)
            | Layer::Resize(_)
            | Layer::Slice(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Where(_)
            | Layer::NonZero(_)
            | Layer::Floor(_)
            | Layer::Ceil(_)
            | Layer::Round(_)
            | Layer::Trunc(_)
            | Layer::Sign(_)
            | Layer::Reciprocal(_)
            | Layer::RoPE(_)
            | Layer::SkipMerge(_)
            | Layer::OpaqueSkip(_)
            | Layer::QdqPerturbation(_)
            | Layer::ExpandLikeLastAxis(_)
            | Layer::Compare(_)
            | Layer::CompareTensor(_) => {}
        }
    }

    let mut heuristics_used = Vec::new();
    if layernorm_forward_mode_nodes > 0 {
        heuristics_used.push(HeuristicUsed::LayerNormForwardMode {
            num_nodes: layernorm_forward_mode_nodes,
        });
    }
    if rmsnorm_forward_mode_nodes > 0 {
        heuristics_used.push(HeuristicUsed::RmsNormForwardMode {
            num_nodes: rmsnorm_forward_mode_nodes,
        });
    }
    if groupnorm_forward_mode_nodes > 0 {
        heuristics_used.push(HeuristicUsed::GroupNormForwardMode {
            num_nodes: groupnorm_forward_mode_nodes,
        });
    }
    if instancenorm_forward_mode_nodes > 0 {
        heuristics_used.push(HeuristicUsed::InstanceNormForwardMode {
            num_nodes: instancenorm_forward_mode_nodes,
        });
    }
    if adain_forward_mode_nodes > 0 {
        heuristics_used.push(HeuristicUsed::AdaInForwardMode {
            num_nodes: adain_forward_mode_nodes,
        });
    }
    if layernorm_sampling_nodes > 0 {
        heuristics_used.push(HeuristicUsed::LayerNormCrownSampling {
            num_nodes: layernorm_sampling_nodes,
        });
    }
    if crown_family {
        if softmax_sampling_nodes > 0 {
            heuristics_used.push(HeuristicUsed::SoftmaxCrownSampling {
                num_nodes: softmax_sampling_nodes,
            });
        }
        if causal_softmax_sampling_nodes > 0 {
            heuristics_used.push(HeuristicUsed::CausalSoftmaxCrownSampling {
                num_nodes: causal_softmax_sampling_nodes,
            });
        }
        if logsoftmax_sampling_nodes > 0 {
            heuristics_used.push(HeuristicUsed::LogSoftmaxCrownSampling {
                num_nodes: logsoftmax_sampling_nodes,
            });
        }
        if reduce_extremum_fixed_index_nodes > 0 {
            heuristics_used.push(HeuristicUsed::ReduceExtremumFixedIndex {
                num_nodes: reduce_extremum_fixed_index_nodes,
            });
        }
        if sampling_relaxations_present {
            heuristics_used.push(HeuristicUsed::SamplingBasedNonlinearRelaxations);
        }
    }

    SoundnessProvenance::from_heuristics(heuristics_used)
}

#[cfg(test)]
mod tests {
    use super::{count_sqrt_negative_domain_graph, count_sqrt_negative_domain_network};
    use crate::{
        AdaIN1dLayer, CausalSoftmaxLayer, CosLayer, GraphNetwork, GroupNormLayer,
        InstanceNorm1dLayer, Layer, LayerNormLayer, LinearLayer, LogSoftmaxLayer, Network,
        PropagationMethod, ReduceMaxLayer, ReduceMinLayer, RmsNormLayer, SiLULayer, SinLayer,
        SoftmaxLayer, SqrtLayer,
    };
    use ndarray::{arr1, arr2};
    use ny_core::{HeuristicUsed, VerificationSoundnessMode};
    use ny_tensor::BoundedTensor;

    fn simple_bounds(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
            .expect("bounds tensor")
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_sqrt_negative_domain_network_absent() {
        let mut network = Network::new();
        let weight = arr2(&[[1.0]]);
        let bias = arr1(&[0.0]);
        network.add_layer(Layer::Linear(
            LinearLayer::new(weight, Some(bias)).expect("linear layer"),
        ));
        let input = simple_bounds(-1.0, 1.0);
        let count = count_sqrt_negative_domain_network(&network, &input).expect("sqrt scan");
        assert_eq!(count, 0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_sqrt_negative_domain_network_counts() {
        let mut network = Network::new();
        network.add_layer(Layer::Sqrt(SqrtLayer));
        let input = simple_bounds(-1.0, 1.0);
        let count = count_sqrt_negative_domain_network(&network, &input).expect("sqrt scan");
        assert_eq!(count, 1);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_sqrt_negative_domain_graph_counts() {
        let mut network = Network::new();
        network.add_layer(Layer::Sqrt(SqrtLayer));
        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input = simple_bounds(-1.0, 1.0);
        let count = count_sqrt_negative_domain_graph(&graph, &input).expect("sqrt scan");
        assert_eq!(count, 1);
    }

    /// Test count_sqrt_negative_domain_from_bounds with pre-computed bounds (#3184).
    #[ntest::timeout(5000)]
    #[test]
    fn test_sqrt_negative_domain_from_precomputed_bounds() {
        let mut network = Network::new();
        network.add_layer(Layer::Sqrt(SqrtLayer));
        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input = simple_bounds(-1.0, 1.0);
        let node_bounds = graph.collect_node_bounds(&input).expect("collect bounds");
        let count = super::count_sqrt_negative_domain_from_bounds(&graph, &input, &node_bounds)
            .expect("sqrt scan from bounds");
        assert_eq!(count, 1);
    }

    /// Verify from_bounds returns 0 when no sqrt nodes have negative inputs (#3184).
    #[ntest::timeout(5000)]
    #[test]
    fn test_sqrt_negative_domain_from_bounds_positive_input() {
        let mut network = Network::new();
        network.add_layer(Layer::Sqrt(SqrtLayer));
        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input = simple_bounds(0.5, 2.0);
        let node_bounds = graph.collect_node_bounds(&input).expect("collect bounds");
        let count = super::count_sqrt_negative_domain_from_bounds(&graph, &input, &node_bounds)
            .expect("sqrt scan from bounds");
        assert_eq!(count, 0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_trig_relaxations_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("linear layer"),
        ));
        network.add_layer(Layer::Sin(SinLayer::new()));
        network.add_layer(Layer::Cos(CosLayer::new()));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::SamplingBasedNonlinearRelaxations),
            "expected SamplingBasedNonlinearRelaxations, got {:?}",
            provenance.heuristics_used()
        );
    }

    /// Regression test for #2533: SiLU uses analytical relaxation (chord+tangent),
    /// not sampling. A network containing only SiLU should be classified as Sound.
    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_silu_is_sound() {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("linear layer"),
        ));
        network.add_layer(Layer::SiLU(SiLULayer::new()));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(
            provenance.mode(),
            VerificationSoundnessMode::Sound,
            "SiLU uses analytical relaxation — should be Sound, not Heuristic"
        );
        assert!(
            provenance.heuristics_used().is_empty(),
            "SiLU should not trigger any heuristic flags, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_normalization_forward_mode_split_by_family() {
        let mut network = Network::new();
        network.add_layer(Layer::LayerNorm(
            LayerNormLayer::new_default(3, 1e-5)
                .expect("valid layer norm")
                .with_forward_mode(true),
        ));
        network.add_layer(Layer::RmsNorm(
            RmsNormLayer::new_default(3, 1e-5)
                .expect("valid rms norm")
                .with_forward_mode(true),
        ));
        network.add_layer(Layer::GroupNorm(
            GroupNormLayer::new_default(4, 2, 1e-5)
                .expect("valid group norm")
                .with_forward_mode(true),
        ));
        network.add_layer(Layer::InstanceNorm1d(
            InstanceNorm1dLayer::new_default(2, 1e-5)
                .expect("valid instance norm")
                .with_forward_mode(true),
        ));
        network.add_layer(Layer::AdaIN1d(
            AdaIN1dLayer::new_identity_style(
                InstanceNorm1dLayer::new_default(2, 1e-5).expect("valid inner instance norm"),
            )
            .expect("valid identity-style AdaIN")
            .with_forward_mode(true),
        ));

        let provenance = super::soundness_provenance_for_network(&network, &PropagationMethod::Ibp);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert_eq!(provenance.heuristics_used().len(), 5);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::LayerNormForwardMode { num_nodes: 1 }),
            "expected LayerNormForwardMode, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::RmsNormForwardMode { num_nodes: 1 }),
            "expected RmsNormForwardMode, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::GroupNormForwardMode { num_nodes: 1 }),
            "expected GroupNormForwardMode, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::InstanceNormForwardMode { num_nodes: 1 }),
            "expected InstanceNormForwardMode, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::AdaInForwardMode { num_nodes: 1 }),
            "expected AdaInForwardMode, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_softmax_sampling_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::Softmax(
            SoftmaxLayer::new(-1).with_heuristic_sampling(true),
        ));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::SoftmaxCrownSampling { num_nodes: 1 }),
            "expected SoftmaxCrownSampling, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_causal_softmax_sampling_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::CausalSoftmax(
            CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true),
        ));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::CausalSoftmaxCrownSampling { num_nodes: 1 }),
            "expected CausalSoftmaxCrownSampling, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_logsoftmax_sampling_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::LogSoftmax(
            LogSoftmaxLayer::new(-1).with_heuristic_sampling(true),
        ));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::LogSoftmaxCrownSampling { num_nodes: 1 }),
            "expected LogSoftmaxCrownSampling, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_softmax_defaults_sound() {
        let mut network = Network::new();
        network.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));
        network.add_layer(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)));
        network.add_layer(Layer::LogSoftmax(LogSoftmaxLayer::new(-1)));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
        assert!(
            provenance.heuristics_used().is_empty(),
            "expected no heuristics for default sound softmax/logsoftmax, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_reduce_extremum_fixed_index_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::ReduceMax(ReduceMaxLayer::new(vec![0], true)));
        network.add_layer(Layer::ReduceMin(ReduceMinLayer::new(vec![0], true)));

        let provenance =
            super::soundness_provenance_for_network(&network, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 2 }),
            "expected ReduceExtremumFixedIndex, got {:?}",
            provenance.heuristics_used()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_graph_softmax_defaults_sound() {
        let mut network = Network::new();
        network.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));
        network.add_layer(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)));
        network.add_layer(Layer::LogSoftmax(LogSoftmaxLayer::new(-1)));
        let graph = GraphNetwork::from_sequential(&network).unwrap();

        let provenance = super::soundness_provenance_for_graph(&graph, &PropagationMethod::Crown);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
        assert!(
            provenance.heuristics_used().is_empty(),
            "expected no heuristics for default sound softmax/logsoftmax graph, got {:?}",
            provenance.heuristics_used()
        );
    }

    /// Regression test for #2807: IBP method should skip heuristic scanning
    /// (heuristics are only relevant for CROWN-family methods).
    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_ibp_skips_crown_heuristics() {
        let mut network = Network::new();
        network.add_layer(Layer::Softmax(
            SoftmaxLayer::new(-1).with_heuristic_sampling(true),
        ));
        network.add_layer(Layer::Sin(SinLayer::new()));

        let provenance = super::soundness_provenance_for_network(&network, &PropagationMethod::Ibp);
        // IBP is not in the CROWN family, so CROWN-specific heuristics should not be reported.
        // LayerNorm forward-mode is always reported regardless of method, but softmax/sin
        // sampling heuristics are CROWN-specific.
        assert!(
            !provenance
                .heuristics_used()
                .contains(&HeuristicUsed::SoftmaxCrownSampling { num_nodes: 1 }),
            "IBP should not report SoftmaxCrownSampling, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            !provenance
                .heuristics_used()
                .contains(&HeuristicUsed::SamplingBasedNonlinearRelaxations),
            "IBP should not report SamplingBasedNonlinearRelaxations, got {:?}",
            provenance.heuristics_used()
        );
        assert!(
            !provenance
                .heuristics_used()
                .contains(&HeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 1 }),
            "IBP should not report ReduceExtremumFixedIndex, got {:?}",
            provenance.heuristics_used()
        );
    }

    /// Regression test for #2807: each CROWN variant should detect heuristic flags.
    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_all_crown_variants_detect_heuristics() {
        let mut network = Network::new();
        network.add_layer(Layer::Sin(SinLayer::new()));

        for method in &[
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown,
            PropagationMethod::BetaCrown,
            PropagationMethod::SdpCrown,
        ] {
            let provenance = super::soundness_provenance_for_network(&network, method);
            assert!(
                provenance
                    .heuristics_used()
                    .contains(&HeuristicUsed::SamplingBasedNonlinearRelaxations),
                "{:?} should detect SamplingBasedNonlinearRelaxations, got {:?}",
                method,
                provenance.heuristics_used()
            );
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_soundness_provenance_ibp_skips_reduce_extremum_fixed_index_heuristic() {
        let mut network = Network::new();
        network.add_layer(Layer::ReduceMax(ReduceMaxLayer::new(vec![0], true)));

        let provenance = super::soundness_provenance_for_network(&network, &PropagationMethod::Ibp);
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
        assert!(
            !provenance
                .heuristics_used()
                .contains(&HeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 1 }),
            "IBP should not report ReduceExtremumFixedIndex, got {:?}",
            provenance.heuristics_used()
        );
    }
}
