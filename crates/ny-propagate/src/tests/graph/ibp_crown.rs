// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN-IBP and NaN guard tests for GraphNetwork IBP propagation.
//!
//! Split from `ibp.rs` to keep files under 500 lines (#2633).

use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::NyError;

fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(&u, &l)| u - l)
        .sum()
}

fn linear4(weights: [[f32; 4]; 4], bias: [f32; 4]) -> LinearLayer {
    LinearLayer::new(arr2(&weights), Some(arr1(&bias)))
        .expect("invariant: valid 4x4 linear parameters")
}

fn selected_widths_3775(
    ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
    result: &GraphCrownIbpBoundsResult,
) -> Vec<(&'static str, f32, f32)> {
    // linear0 is first layer (CROWN = IBP), linear_pre_ln is bounded through
    // relu → linear0 where ReLU CROWN can tighten.
    ["linear0", "linear_pre_ln"]
        .iter()
        .map(|producer| {
            let ibp = ibp_bounds
                .get(*producer)
                .expect("selected IBP producer bound missing");
            let crown_ibp = result
                .bounds
                .get(*producer)
                .expect("selected CROWN-IBP producer bound missing");
            (*producer, total_width(ibp), total_width(crown_ibp))
        })
        .collect()
}

fn layernorm_widths_3775(
    ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
    result: &GraphCrownIbpBoundsResult,
) -> (f32, f32) {
    (
        total_width(
            ibp_bounds
                .get("layernorm")
                .expect("layernorm IBP node bound missing"),
        ),
        total_width(
            result
                .bounds
                .get("layernorm")
                .expect("layernorm CROWN-IBP node bound missing"),
        ),
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_dag_status_reports_nonzero_fallback_provenance() {
    tests::with_crown_dense_budget_mb("2048", || {
        let w1 = arr2(&[[1.0_f32, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
        let b1 = arr1(&[0.1_f32, -0.2, 0.3]);

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(
                LinearLayer::new(w1, Some(b1))
                    .expect("invariant: linear layer construction must succeed"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "nonzero",
            Layer::NonZero(NonZeroLayer),
            vec!["lin1".to_string()],
        ));
        graph.set_output("nonzero");

        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("invariant: input bounds must be valid");

        let with_status = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("invariant: DAG CROWN-IBP should fallback to IBP for NonZero");

        assert!(with_status.has_fallbacks());
        assert_eq!(with_status.fallback_count(), 1);
        assert_eq!(
            with_status.provenance_for_node("nonzero"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::CrownPropagationError
            ))
        );
        let event = with_status
            .fallback_events
            .first()
            .expect("invariant: fallback event must exist");
        assert_eq!(event.layer_type, "NonZero");
        assert_eq!(event.reason, CrownIbpFallbackReason::CrownPropagationError);
        assert!(event.details.contains("nonzero"));
    });
}

/// Regression: non-sound IBP must return error on NaN input (#2563).
#[ntest::timeout(10000)]
#[test]
fn test_graph_ibp_nonsound_nan_input_returns_error() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let nan_input = BoundedTensor::new_unchecked(
        arr1(&[1.0_f32, f32::NAN, -1.0]).into_dyn(),
        arr1(&[2.0_f32, f32::NAN, 0.5]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_ibp(&nan_input);
    assert!(
        result.is_err(),
        "non-sound IBP must reject NaN input, got Ok({:?})",
        result.unwrap()
    );
    let err = result.unwrap_err();
    match &err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("NaN"), "error should mention NaN, got: {msg}");
        }
        other => panic!("expected NumericalInstability for NaN input, got: {other:?}"),
    }
}

/// Sound IBP must also reject NaN input (existing behavior, sanity check).
#[ntest::timeout(10000)]
#[test]
fn test_graph_ibp_sound_nan_input_returns_error() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let nan_input = BoundedTensor::new_unchecked(
        arr1(&[1.0_f32, f32::NAN, -1.0]).into_dyn(),
        arr1(&[2.0_f32, f32::NAN, 0.5]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_ibp_sound(&nan_input);
    assert!(
        result.is_err(),
        "sound IBP must reject NaN input, got Ok({:?})",
        result.unwrap()
    );
    let err = result.unwrap_err();
    match &err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("NaN"), "error should mention NaN, got: {msg}");
        }
        other => panic!("expected NumericalInstability for NaN input, got: {other:?}"),
    }
}

/// Build the #3680 extracted-stage reproducer graph.
fn build_extracted_stage_graph_3680() -> (GraphNetwork, BoundedTensor) {
    use crate::layers::{AddLayer, Conv1dLayer, PadLayer, PadMode, SliceLayer};

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "main_slice",
        Layer::Slice(SliceLayer::new(-1, 1, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "main_pad",
        Layer::Pad(PadLayer::new(
            vec![(0, 0), (0, 0), (1, 1)],
            PadMode::Constant(0.0),
        )),
        vec!["main_slice".to_string()],
    ));

    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.45_f32, -0.20])
        .expect("valid Conv1d kernel");
    let conv = Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.05_f32])), 1, 0, 5)
        .expect("valid Conv1d params");
    graph.add_node(GraphNode::new(
        "main_conv",
        Layer::Conv1d(conv),
        vec!["main_pad".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "skip_slice",
        Layer::Slice(SliceLayer::new(-1, 0, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "output_add",
        Layer::Add(AddLayer),
        vec!["main_conv".to_string(), "skip_slice".to_string()],
    ));
    graph.set_output("output_add");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![-1.0_f32, -0.8, -0.4, 0.1, 0.5])
            .expect("valid lower input"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![0.2_f32, 0.4, 0.7, 1.0, 1.3])
            .expect("valid upper input"),
    )
    .expect("valid bounded input");

    (graph, input)
}

/// Build the encoder-prefix section: linear0 → relu → linear_pre_ln.
fn build_encoder_prefix_3775(graph: &mut GraphNetwork) {
    let linear0 = linear4(
        [
            [1.0_f32, -0.2, 0.1, 0.4],
            [0.3, 0.8, -0.5, 0.2],
            [-0.6, 0.4, 0.9, -0.1],
            [0.2, -0.7, 0.3, 1.1],
        ],
        [0.05_f32, -0.03, 0.02, 0.1],
    );
    graph.add_node(GraphNode::from_input("linear0", Layer::Linear(linear0)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear0".to_string()],
    ));
    let linear_pre_ln = linear4(
        [
            [0.6_f32, -0.1, 0.2, 0.3],
            [-0.3, 0.5, 0.4, -0.2],
            [0.1, 0.3, -0.5, 0.6],
            [0.4, -0.4, 0.1, 0.7],
        ],
        [0.02_f32, -0.01, 0.03, -0.02],
    );
    graph.add_node(GraphNode::new(
        "linear_pre_ln",
        Layer::Linear(linear_pre_ln),
        vec!["relu".to_string()],
    ));
}

/// Graph: linear0 → relu → linear_pre_ln → layernorm → linear1 → gelu → linear2.
/// ReLU lets CROWN backward from linear_pre_ln exploit linear relaxation (#3775).
fn build_unary_transformer_graph_3775() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    build_encoder_prefix_3775(&mut graph);

    let layernorm = LayerNormLayer::new_default(4, 1e-5)
        .expect("invariant: valid LayerNorm default parameters");
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(layernorm),
        vec!["linear_pre_ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(linear4(
            [
                [0.5, -0.2, 0.3, 0.1],
                [-0.1, 0.4, 0.2, -0.3],
                [0.3, 0.1, -0.4, 0.5],
                [0.2, -0.3, 0.1, 0.6],
            ],
            [0.0, 0.05, -0.03, 0.02],
        )),
        vec!["layernorm".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::new(GeluApproximation::Tanh)),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear4(
            [
                [0.6, -0.1, 0.3, 0.2],
                [-0.2, 0.5, 0.4, -0.3],
                [0.1, 0.2, -0.7, 0.8],
                [0.3, -0.6, 0.2, 0.4],
            ],
            [0.02, -0.01, 0.0, 0.03],
        )),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), 0.25_f32)
        .expect("invariant: symmetric transformer test interval should construct bounded input");
    (graph, input)
}

/// Regression #3680: CROWN-IBP on extracted-subgraph must not degrade to fallback.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_extracted_graph_no_propagation_error_fallback_3680() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_extracted_stage_graph_3680();

        let ibp = graph
            .propagate_ibp(&input)
            .expect("IBP should succeed on extracted-stage graph");
        assert_eq!(ibp.shape(), &[1, 1, 4], "#3680 IBP output must be [1,1,4]");

        let with_status = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("#3680 CROWN-IBP DAG should succeed on extracted-stage graph");

        let output_provenance = with_status.provenance_for_node("output_add");
        assert_eq!(
            output_provenance,
            Some(BoundsProvenance::Crown),
            "#3680 extracted output must have Crown provenance: {output_provenance:?}"
        );

        let crown_ibp_output = with_status
            .bounds
            .get("output_add")
            .expect("#3680 CROWN-IBP bounds must include output_add");
        assert_eq!(crown_ibp_output.shape(), ibp.shape());

        for (i, (lo, up)) in crown_ibp_output
            .lower()
            .iter()
            .zip(crown_ibp_output.upper().iter())
            .enumerate()
        {
            assert!(
                lo.is_finite() && up.is_finite(),
                "#3680 element {i}: non-finite"
            );
            assert!(lo <= up, "#3680 element {i}: inverted lo={lo} > up={up}");
        }
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_unary_transformer_selected_nodes_tighten_3775() {
    tests::with_crown_dense_budget_mb("2048", || {
        use crate::layers::LayerNormCrownMode;

        let (mut graph, input) = build_unary_transformer_graph_3775();

        assert_eq!(
            graph.set_layernorm_crown_mode(LayerNormCrownMode::Sampling),
            1,
            "#3775 test graph should contain exactly one LayerNorm node"
        );

        let ibp_bounds = graph
            .collect_node_bounds(&input)
            .expect("#3775 unary transformer graph should collect IBP bounds");
        let with_status = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("#3775 unary transformer graph should collect CROWN-IBP bounds");

        assert!(
            graph.should_use_crown_ibp_intermediates(),
            "#3775 unary transformer graph must use CROWN-IBP intermediates"
        );

        for producer in ["linear0", "linear_pre_ln"] {
            assert_eq!(
                with_status.provenance_for_node(producer),
                Some(BoundsProvenance::Crown),
                "#3775 selected producer `{producer}` must stay on the Crown path"
            );
        }

        assert!(
            with_status
                .fallback_events
                .iter()
                .all(|event| event.reason != CrownIbpFallbackReason::DemandDrivenSkip),
            "#3775 demand-driven skips are policy decisions, not fallback events"
        );

        let selected_widths = selected_widths_3775(&ibp_bounds, &with_status);
        let any_tighter = selected_widths
            .iter()
            .any(|(_, ibp_width, crown_width)| *crown_width < *ibp_width - 1e-6);
        let layernorm_widths = layernorm_widths_3775(&ibp_bounds, &with_status);
        assert!(
            any_tighter,
            "#3775 demand-driven CROWN-IBP should strictly tighten at least one selected producer: \
             selected={selected_widths:?}, layernorm={layernorm_widths:?}"
        );
    });
}

/// Build a variable-style AdaIN ternary graph for #4142.
///
/// Graph: _input → add_x (+0.0) ───────────────────┐
///        _input → mul_g (*0.5) → add_g (+1.0) ──── AdaIN(variable) → output
///        _input → add_b (+0.5) ───────────────────┘
fn build_variable_adain_graph_4142() -> (GraphNetwork, BoundedTensor) {
    let (num_channels, time_steps) = (2, 3);
    let scalar = |v: f32| ArrayD::from_elem(IxDyn(&[]), v);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x_pass",
        Layer::AddConstant(AddConstantLayer::new(scalar(0.0))),
    ));
    graph.add_node(GraphNode::from_input(
        "g_scale",
        Layer::MulConstant(MulConstantLayer::new(scalar(0.5))),
    ));
    graph.add_node(GraphNode::new(
        "g_shift",
        Layer::AddConstant(AddConstantLayer::new(scalar(1.0))),
        vec!["g_scale".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "b_shift",
        Layer::AddConstant(AddConstantLayer::new(scalar(0.5))),
    ));

    let inn = InstanceNorm1dLayer::new_default(num_channels, 1e-5)
        .expect("invariant: valid InstanceNorm1d");
    let adain = AdaIN1dLayer::variable_style(inn).expect("invariant: valid variable-style AdaIN");
    assert!(adain.requires_style_inputs());
    assert_eq!(Layer::AdaIN1d(adain.clone()).min_inputs(), 3);

    graph.add_node(GraphNode::new(
        "adain_out",
        Layer::AdaIN1d(adain),
        vec!["x_pass".into(), "g_shift".into(), "b_shift".into()],
    ));
    graph.set_output("adain_out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[num_channels, time_steps]),
            vec![-0.5_f32, 0.2, -0.3, 0.1, -0.4, 0.6],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[num_channels, time_steps]),
            vec![0.5_f32, 1.2, 0.7, 1.1, 0.6, 1.6],
        )
        .unwrap(),
    )
    .unwrap();

    (graph, input)
}

/// Concrete AdaIN eval for one sample: y = g * InstanceNorm(x) + b.
/// Returns the flat output vector.
fn eval_adain_concrete_4142(input: &BoundedTensor, seed: u32) -> Vec<f32> {
    let shape = input.shape();
    let (num_ch, time) = (shape[0], shape[1]);
    let n = num_ch * time;
    let inp_lo = input.lower().as_slice().unwrap();
    let inp_hi = input.upper().as_slice().unwrap();

    let mut x = vec![0.0_f32; n];
    let mut g = vec![0.0_f32; n];
    let mut b = vec![0.0_f32; n];
    for i in 0..n {
        let t = hash_sample(seed, i as u32, 0) as f32 / u32::MAX as f32;
        let s = inp_lo[i] + t * (inp_hi[i] - inp_lo[i]);
        x[i] = s;
        g[i] = s * 0.5 + 1.0;
        b[i] = s + 0.5;
    }

    // InstanceNorm: per-channel normalize.
    let mut z = x.clone();
    for c in 0..num_ch {
        let start = c * time;
        let sl = &x[start..start + time];
        let mean: f32 = sl.iter().sum::<f32>() / time as f32;
        let var: f32 = sl.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / time as f32;
        let std = (var + 1e-5_f32).sqrt();
        for t_idx in 0..time {
            z[start + t_idx] = (sl[t_idx] - mean) / std;
        }
    }
    (0..n).map(|i| g[i] * z[i] + b[i]).collect()
}

fn hash_sample(seed: u32, idx: u32, offset: u32) -> u32 {
    let mut h = seed
        .wrapping_mul(2654435761)
        .wrapping_add(idx.wrapping_mul(2246822519));
    h = h.wrapping_add(offset.wrapping_mul(3266489917));
    h ^= h >> 16;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^= h >> 16;
    h
}

/// #4142 regression: variable-style AdaIN graph IBP must route through the
/// ternary classify path and produce sound bounds.
#[ntest::timeout(10000)]
#[test]
fn test_graph_ibp_variable_style_adain_ternary_dispatch_4142() {
    let (graph, input) = build_variable_adain_graph_4142();

    let result = graph
        .propagate_ibp(&input)
        .expect("#4142: variable-style AdaIN graph IBP must succeed");

    assert_eq!(
        result.shape(),
        input.shape(),
        "output shape must match input"
    );

    let lo = result.lower().as_slice().unwrap();
    let hi = result.upper().as_slice().unwrap();
    for seed in 0..200_u32 {
        let y = eval_adain_concrete_4142(&input, seed);
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                yi >= lo[i] - 1e-4 && yi <= hi[i] + 1e-4,
                "#4142 IBP soundness: seed={seed}, i={i}: y={yi} not in [{}, {}]",
                lo[i],
                hi[i]
            );
        }
    }
}
