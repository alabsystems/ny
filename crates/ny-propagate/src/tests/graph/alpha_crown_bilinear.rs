// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN tests for BilinearCrown nodes (attention Q @ K^T).
//!
//! Verifies that alpha-CROWN with BilinearCrown alpha optimization produces
//! bounds tighter than IBP on attention subgraphs.
//!
//! Part of #3287: Wire BilinearCrown alpha optimization into alpha_crown_loop.

use crate::layers::binary_ops::BilinearCrownLayer;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

/// Sum of (upper - lower) across all output dimensions.
///
/// Panics if any element has inverted or NaN bounds, preventing silent masking.
fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .enumerate()
        .map(|(i, (l, u))| {
            let w = u - l;
            assert!(
                w >= -1e-6 && w.is_finite(),
                "output[{i}]: bad width {w} (l={l}, u={u})"
            );
            w
        })
        .sum()
}

/// Assert alpha-CROWN bounds contain sampled concrete outputs through a graph.
///
/// Samples each input element independently over `[0, 0.5, 1]` (lower/mid/upper),
/// giving 3^n_elements points. For a [2,2] input this is 81 points covering all
/// corner and midpoint combinations — no element pairing blind spots.
fn assert_bilinear_soundness(
    graph: &GraphNetwork,
    bounds: &BoundedTensor,
    input_center: &[f32],
    input_radius: &[f32],
    shape: &[usize],
) {
    let n = input_center.len();
    let levels = [0.0_f32, 0.5, 1.0];
    let n_samples = 3_usize.pow(n as u32); // 3^n combinations
    for sample_idx in 0..n_samples {
        let vals: Vec<f32> = (0..n)
            .map(|dim| {
                let level_idx = (sample_idx / 3_usize.pow(dim as u32)) % 3;
                let t = levels[level_idx];
                input_center[dim] - input_radius[dim] + 2.0 * input_radius[dim] * t
            })
            .collect();
        let point = ArrayD::from_shape_vec(IxDyn(shape), vals).unwrap();
        let point_input = BoundedTensor::concrete(point).unwrap();
        let exact = graph.propagate_ibp(&point_input).unwrap();
        for (idx, ((out_val, &lower), &upper)) in exact
            .lower()
            .iter()
            .zip(bounds.lower().iter())
            .zip(bounds.upper().iter())
            .enumerate()
        {
            assert!(
                *out_val >= lower - 1e-3,
                "BilinearCrown alpha soundness: sample {sample_idx} output[{idx}] \
                 {out_val} < lower {lower}"
            );
            assert!(
                *out_val <= upper + 1e-3,
                "BilinearCrown alpha soundness: sample {sample_idx} output[{idx}] \
                 {out_val} > upper {upper}"
            );
        }
    }
}

/// Assert bounds are finite and non-inverted.
fn assert_finite_non_inverted(bounds: &BoundedTensor) {
    for (i, (&l, &u)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(l.is_finite(), "output[{i}]: lower non-finite");
        assert!(u.is_finite(), "output[{i}]: upper non-finite");
        assert!(l <= u + 1e-6, "output[{i}]: inverted l={l} > u={u}");
    }
}

/// Build attention subgraph: Input -> Linear+ReLU (Q) + Linear+ReLU (K) -> BilinearCrown.
fn build_attention_qk_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    // Q projection: input [2, 2] -> linear [2, 2] -> ReLU
    let w_q = arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]);
    let proj_q = LinearLayer::new(w_q, Some(arr1(&[0.1_f32, -0.2]))).unwrap();
    graph.add_node(GraphNode::from_input("proj_q", Layer::Linear(proj_q)));
    graph.add_node(GraphNode::new(
        "relu_q",
        Layer::ReLU(ReLULayer),
        vec!["proj_q".to_string()],
    ));

    // K projection: input [2, 2] -> linear [2, 2] -> ReLU
    let w_k = arr2(&[[-0.5_f32, 0.7], [0.6, -0.4]]);
    let proj_k = LinearLayer::new(w_k, Some(arr1(&[-0.1_f32, 0.15]))).unwrap();
    graph.add_node(GraphNode::from_input("proj_k", Layer::Linear(proj_k)));
    graph.add_node(GraphNode::new(
        "relu_k",
        Layer::ReLU(ReLULayer),
        vec!["proj_k".to_string()],
    ));

    // BilinearCrown: Q @ K^T with attention scale 1/sqrt(2)
    let bilinear = BilinearCrownLayer::new(true, Some(1.0 / 2.0_f32.sqrt()));
    graph.add_node(GraphNode::binary(
        "qk",
        Layer::BilinearCrown(bilinear),
        "relu_q",
        "relu_k",
    ));
    graph.set_output("qk");
    graph
}

/// Measurement test for #3287: BilinearCrown alpha optimization on attention subgraph.
///
/// Builds: Input -> Linear+ReLU (Q) + Linear+ReLU (K) -> BilinearCrown(Q @ K^T).
/// Alpha-CROWN should optimize both ReLU and BilinearCrown alphas, producing
/// bounds tighter than IBP.
#[ntest::timeout(60000)]
#[test]
fn test_graph_alpha_crown_bilinear_tighter_than_ibp_3287() {
    use crate::bounds::AlphaCrownConfig;

    let graph = build_attention_qk_graph();

    let center = [0.5_f32, 1.0, 1.0, 0.5];
    let radius = [0.5_f32, 0.5, 0.5, 0.5];
    let lower: Vec<f32> = center.iter().zip(&radius).map(|(c, r)| c - r).collect();
    let upper: Vec<f32> = center.iter().zip(&radius).map(|(c, r)| c + r).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper).unwrap(),
    )
    .unwrap();

    let config = AlphaCrownConfig {
        iterations: 50,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    let ibp_width = total_width(&ibp_bounds);
    let alpha_width = total_width(&alpha_bounds);

    assert_bilinear_soundness(&graph, &alpha_bounds, &center, &radius, &[2, 2]);
    assert_finite_non_inverted(&alpha_bounds);

    assert!(
        alpha_width <= ibp_width + 1e-4,
        "#3287: alpha-CROWN width ({alpha_width:.6}) should be <= IBP ({ibp_width:.6})"
    );

    eprintln!(
        "#3287 BilinearCrown alpha: IBP={ibp_width:.6}, alpha={alpha_width:.6}, \
         improvement={:.2}%",
        if ibp_width > 0.0 {
            (1.0 - alpha_width / ibp_width) * 100.0
        } else {
            0.0
        }
    );
}
