// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #fl-alpha-composition (consult #8 Days 1-3): α ascent ON TOP of the
//! FL-fixed intermediates as ONE path.
//!
//! Pins the three unit legs of the composition:
//!
//! 1. **FL-fixed intermediates**: on a conv DAG with forward-linear references
//!    admitted, the reference map the α loop consumes IS the cached FL node
//!    map — bit-identical, not a re-collected CROWN-IBP/IBP map — and the
//!    pre-loop reuse gate accepts it (`can_reuse_initial_node_bounds`), so no
//!    incomparable collector is substituted mid-walk. Across iterations the
//!    map only ever tightens (`GraphAlphaReferenceBounds::merge_candidate` is
//!    element-wise intersect-only), which is exactly the
//!    `fix_interm_bounds=true` FL-floor semantics the consult specifies.
//!
//! 2. **Gradient flow**: with FL references as the node map, the binding-row
//!    margin objective selects an unverified row and the true-gradient replay
//!    (`binding_row_true_alpha_grads`) produces finite, not-all-zero ∂bound/∂α
//!    on that row — the ascent has a real direction to move in.
//!
//! 3. **Disarmed = today**: with the typed forward-alpha-surrogate lever OFF
//!    (the default), the certified rebuild refuses (`Ok(None)`) — the
//!    composition is bit-inert until a preset key / resolver rule arms it.

use super::can_reuse_initial_node_bounds;
use super::init::DagAlphaInitResult;
use crate::bounds::{AlphaCrownConfig, AlphaSpecAscent, AlphaSpecEarlyExit};
use crate::layers::{AddLayer, Conv2dLayer, Layer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
use crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource;
use crate::tests::with_default_fl_reference_env;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

const SPATIAL: usize = 8;

/// 3x3 stride-1 same-padding single-channel conv with deterministic
/// mixed-sign weights (same recipe as `residual_patches_tests`).
fn small_conv(seed: f32) -> Conv2dLayer {
    let mut kernel = ArrayD::zeros(IxDyn(&[1, 1, 3, 3]));
    for i in 0..3 {
        for j in 0..3 {
            kernel[[0, 0, i, j]] = seed * (((i * 3 + j) as f32) - 4.0) / 10.0;
        }
    }
    Conv2dLayer::with_input_shape(
        kernel,
        Some(Array1::from_vec(vec![0.01_f32])),
        (1, 1),
        (1, 1),
        SPATIAL,
        SPATIAL,
    )
    .expect("valid fixture conv")
}

/// Minimal conv-DAG (ResNet block): the graph class whose alpha reference
/// collection routes to the certified forward-linear pass.
///
/// input -> conv1 -> relu1 -> conv2 -> add(conv2, conv1) -> relu2 -> conv3
fn build_fl_resnet_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(small_conv(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(small_conv(0.7)),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["conv2".to_string(), "conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv3",
        Layer::Conv2d(small_conv(0.5)),
        vec!["relu2".to_string()],
    ));
    graph.set_output("conv3");

    // Centered at 0 with a wide box so the ReLU pre-activations genuinely
    // cross zero — unstable neurons are what α (and its gradient) act on.
    let center = ArrayD::from_elem(IxDyn(&[1, SPATIAL, SPATIAL]), 0.0_f32);
    let input = BoundedTensor::from_epsilon(center, 0.5).expect("valid input box");
    (graph, input)
}

fn assert_bitwise_eq(name: &str, a: &BoundedTensor, b: &BoundedTensor) {
    assert_eq!(a.shape(), b.shape(), "{name}: shape mismatch");
    for (x, y) in a.lower().iter().zip(b.lower().iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{name}: lower bits differ");
    }
    for (x, y) in a.upper().iter().zip(b.upper().iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{name}: upper bits differ");
    }
}

/// Leg 1: the intermediates the α walk consumes ARE the FL ones.
///
/// `init_dag_alpha_state` must (a) report the ForwardLinear collector, (b)
/// hand the loop a node map bit-identical to the cached certified FL map for
/// every FL node, and (c) pass the pre-loop reuse gate so the initial CROWN
/// bound consumes this map instead of re-collecting CROWN-IBP.
#[ntest::timeout(120000)]
#[test]
fn alpha_init_consumes_the_fl_reference_map_bit_identically() {
    with_default_fl_reference_env(|_| {
        let (graph, input) = build_fl_resnet_fixture();
        let exec_order = graph.exec_order().expect("exec order").to_vec();

        let config = AlphaCrownConfig {
            fix_interm_bounds: true,
            ..AlphaCrownConfig::default()
        };

        // The collector the α init runs.
        let (collected, source) = graph
            .collect_alpha_reference_bounds_with_engine_and_source(
                &input,
                &config,
                None,
                &exec_order,
            )
            .expect("alpha reference collection");
        assert_eq!(
            source,
            AlphaReferenceBoundsSource::ForwardLinear,
            "conv DAG with FL enabled must route the α references to forward-linear"
        );

        // The certified FL map itself (cache warmed by the collection above).
        let fl_map = graph
            .collect_forward_linear_bounds_dag_cached(&input, None, None)
            .expect("certified FL map");
        assert!(!fl_map.is_empty(), "FL map must cover the graph");
        for (node, fl_bounds) in fl_map.iter() {
            let got = collected
                .get(node)
                .unwrap_or_else(|| panic!("α references missing FL node '{node}'"));
            assert_bitwise_eq(node, got, fl_bounds);
        }

        // The init state the loop destructures carries the SAME map + source.
        let init = graph
            .init_dag_alpha_state(&input, &config, None, None)
            .expect("init dag alpha state");
        let DagAlphaInitResult::Ready(state) = init else {
            panic!("fixture has unstable ReLUs: init must be Ready");
        };
        assert_eq!(
            state.node_bounds_source,
            AlphaReferenceBoundsSource::ForwardLinear
        );
        for (node, fl_bounds) in fl_map.iter() {
            let got = state
                .node_bounds
                .get(node)
                .unwrap_or_else(|| panic!("init node map missing FL node '{node}'"));
            assert_bitwise_eq(node, got, fl_bounds);
        }
        assert!(
            state.node_bounds.contains_key(NETWORK_INPUT),
            "init inserts the input box alongside the FL map"
        );

        // Pre-loop reuse gate: Step 1 would independently select
        // forward-linear here, so the FL init map IS the map the initial
        // CROWN bound consumes — no CROWN-IBP recollection is substituted.
        let step1_would_use_forward_linear = graph.has_conv_layers()
            && !graph.is_sequential_graph(&exec_order)
            && GraphNetwork::forward_linear_reference_enabled();
        assert!(step1_would_use_forward_linear);
        assert!(can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            step1_would_use_forward_linear,
        ));
        // And the incomparable-collector guard still refuses the swap when
        // Step 1 would NOT pick forward-linear (fail-closed leg).
        assert!(!can_reuse_initial_node_bounds(
            AlphaReferenceBoundsSource::ForwardLinear,
            false,
        ));
    });
}

/// Leg 2: finite, not-all-zero binding-row α gradients on the FL map.
///
/// Margin fixture: two one-hot rows with thresholds the current bound cannot
/// clear, so both are unverified and `binding_margin_lower_objective` must
/// select the worse one. The true-gradient replay at that row must return
/// finite per-ReLU ∂bound/∂α with at least one nonzero component.
#[ntest::timeout(120000)]
#[test]
fn binding_row_gradients_are_finite_nonzero_on_fl_references() {
    with_default_fl_reference_env(|_| {
        let (graph, input) = build_fl_resnet_fixture();

        let config = AlphaCrownConfig {
            fix_interm_bounds: true,
            ..AlphaCrownConfig::default()
        };
        let DagAlphaInitResult::Ready(state) = graph
            .init_dag_alpha_state(&input, &config, None, None)
            .expect("init dag alpha state")
        else {
            panic!("fixture has unstable ReLUs: init must be Ready");
        };
        assert_eq!(
            state.node_bounds_source,
            AlphaReferenceBoundsSource::ForwardLinear
        );

        // One certified fold over the FL-fixed intermediates, capturing the
        // per-ReLU intermediates the replay differentiates.
        let mut grads_lower: Vec<Array1<f32>> = state
            .relu_nodes
            .iter()
            .map(|(name, _)| {
                let pre = graph
                    .relu_preactivation_bounds(name, &input, &state.node_bounds, "fl-comp-test")
                    .expect("pre-activation bounds");
                Array1::zeros(pre.len())
            })
            .collect();
        let mut grads_upper: Vec<Array1<f32>> =
            grads_lower.iter().map(|g| Array1::zeros(g.len())).collect();
        let (bounds, intermediate) = graph
            .dag_alpha_backward_pass_with_intermediates(
                &input,
                &state.node_bounds,
                &state.exec_order,
                state.output_dim,
                state.input_dim,
                state.runtime.relu_name_to_idx(),
                state.runtime.graph(),
                None,
                &mut grads_lower,
                &mut grads_upper,
                None,
                None,
                None,
                None,
            )
            .expect("certified fold over FL-fixed intermediates");

        // Margin rows: one-hot on outputs 3 and 11, thresholds far above the
        // reachable bound — both unverified by construction.
        let one_hot = |k: usize| {
            let mut row = vec![0.0_f32; state.output_dim];
            row[k] = 1.0;
            row
        };
        let ascent = AlphaSpecAscent::new(vec![
            AlphaSpecEarlyExit {
                objective: one_hot(3),
                threshold: 1.0e3,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: one_hot(11),
                threshold: 1.0e3,
                verify_upper_bound: false,
            },
        ])
        .expect("valid ascent rows");
        let lo: Vec<f32> = bounds.lower().iter().copied().collect();
        let hi: Vec<f32> = bounds.upper().iter().copied().collect();
        let binding = super::binding_margin_lower_objective(&ascent, &lo, &hi)
            .expect("both rows are unverified: a binding row must exist");
        assert!(binding.slack <= 0.0, "binding row must be unverified");
        assert!(binding.lower_objective.iter().all(|v| v.is_finite()));

        // The one-hot margin row IS an output row: replay it.
        let output_row = if binding.row_index == 0 { 3 } else { 11 };
        let replay = graph
            .binding_row_true_alpha_grads(&input, state.runtime.graph(), &intermediate, output_row)
            .expect("binding-row true-gradient replay");

        let mut saw_nonzero = false;
        assert!(!replay.grads.is_empty(), "replay must cover the ReLU nodes");
        for (node, grad) in &replay.grads {
            for &g in grad.iter() {
                assert!(g.is_finite(), "non-finite ∂bound/∂α at '{node}'");
                if g != 0.0 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(
            saw_nonzero,
            "binding-row gradient must be nonzero somewhere: the ascent has a direction"
        );
        assert!(replay.replayed_row_value.is_finite());
    });
}

/// Leg 3 (δ=0 trip-wire): with the typed lever DISARMED (default), the
/// certified alpha-optimized rebuild refuses — the composition adds nothing
/// until a preset key / resolver rule arms it, so disarmed runs are today's
/// runs bit-for-bit on this path.
#[ntest::timeout(120000)]
#[test]
fn disarmed_surrogate_refuses_the_alpha_optimized_rebuild() {
    with_default_fl_reference_env(|_| {
        let (graph, input) = build_fl_resnet_fixture();
        assert!(
            !graph.cgan_forward_alpha_surrogate_enabled(),
            "the forward-alpha surrogate must be default-OFF"
        );
        // Warm the fixed FL cache the optimizer would otherwise want.
        graph
            .collect_forward_linear_bounds_dag_cached(&input, None, None)
            .expect("certified FL map");
        let spec = ndarray::Array2::<f32>::eye(SPATIAL * SPATIAL);
        let refused = graph
            .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
            .expect("disarmed lever must not error");
        assert!(
            refused.is_none(),
            "disarmed lever must refuse the rebuild (Ok(None)) — composition inert"
        );
    });
}
