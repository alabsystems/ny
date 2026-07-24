// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase 3 measurement: McCormick CROWN vs IBP decision gate (#286).
//!
//! Compares IBP, CROWN, and alpha-CROWN bounds at multiple perturbation radii
//! to determine whether broadcast McCormick architecture produces tighter bounds
//! than IBP for attention BilinearCrown nodes.
//!
//! Reference: designs/2026-03-04-286-attention-bilinear-alternative.md Phase 3
//!
//! Split across two files:
//! - `measurement.rs` — synthetic graph tests (Linear -> BilinearCrown, no ReLU)
//! - `measurement_phase3.rs` — decision gate with ReLU activations (DAG alpha-CROWN)

use std::fmt::Write as _;

use ndarray::{arr1, array, Array2, ArrayD};
use ny_tensor::BoundedTensor;

use crate::bounds::{AlphaCrownConfig, GradientMethod};
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::Layer;
use crate::network::core::graph::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// Single-eps measurement result for decision gate comparison.
struct EpsMeasurement {
    eps: f32,
    ibp_max: f32,
    crown_max: f32,
    alpha_max: f32,
}

/// Build a bilinear graph with the given Q/K weight matrices.
fn build_bilinear_graph_with_weights(
    wq: Array2<f32>,
    bias_q: Option<ndarray::Array1<f32>>,
    wk: Array2<f32>,
    bias_k: Option<ndarray::Array1<f32>>,
    scale: f32,
) -> GraphNetwork {
    let linear_q = LinearLayer::new(wq, bias_q).expect("invariant: valid Q weight matrix");
    let linear_k = LinearLayer::new(wk, bias_k).expect("invariant: valid K weight matrix");
    let bilinear = BilinearCrownLayer::new(false, Some(scale));

    let mut graph = GraphNetwork {
        output_node: "bilinear".to_string(),
        ..GraphNetwork::new()
    };
    for (name, layer) in [
        ("linear_q", Layer::Linear(linear_q)),
        ("linear_k", Layer::Linear(linear_k)),
        ("bilinear", Layer::BilinearCrown(bilinear)),
    ] {
        let inputs = if name == "bilinear" {
            vec!["linear_q".to_string(), "linear_k".to_string()]
        } else {
            vec![NETWORK_INPUT.to_string()]
        };
        graph.nodes.insert(
            name.to_string(),
            GraphNode {
                name: name.to_string(),
                layer,
                inputs,
            },
        );
    }
    graph.node_order = vec![
        "linear_q".to_string(),
        "linear_k".to_string(),
        "bilinear".to_string(),
    ];
    graph
}

/// Compute max bound width from a BoundedTensor.
pub(super) fn max_bound_width(bt: &BoundedTensor) -> f32 {
    bt.lower()
        .iter()
        .zip(bt.upper().iter())
        .map(|(&l, &u)| u - l)
        .fold(0.0_f32, f32::max)
}

/// Assert all bounds in a BoundedTensor are valid (lower <= upper).
pub(super) fn assert_valid_bounds(bt: &BoundedTensor, label: &str, eps: f32) {
    for (l, u) in bt.lower().iter().zip(bt.upper().iter()) {
        assert!(
            l <= u || (*l - *u).abs() < 1e-5,
            "{} bounds invalid at eps={}: l={} > u={}",
            label,
            eps,
            l,
            u
        );
    }
}

/// Run IBP/CROWN/alpha-CROWN comparison at a single eps value.
fn measure_at_eps(
    graph: &GraphNetwork,
    center: &ArrayD<f32>,
    eps: f32,
    alpha_config: &AlphaCrownConfig,
) -> EpsMeasurement {
    let lower = center - eps;
    let upper = center + eps;
    let input =
        BoundedTensor::new(lower, upper).expect("invariant: center +/- eps produces valid bounds");

    let ibp_bounds = graph
        .propagate_ibp(&input)
        .expect("invariant: IBP succeeds on valid bilinear graph");
    let ibp_max = max_bound_width(&ibp_bounds);

    let crown_max = match graph.propagate_crown_batched(&input) {
        Ok(b) => {
            assert_valid_bounds(&b, "CROWN", eps);
            max_bound_width(&b)
        }
        Err(_) => f32::NAN,
    };

    let alpha_max = match graph.alpha_crown_batched_optimize(&input, alpha_config, None) {
        Ok(r) => {
            assert_valid_bounds(&r.bounds, "alpha-CROWN", eps);
            max_bound_width(&r.bounds)
        }
        Err(_) => f32::NAN,
    };

    EpsMeasurement {
        eps,
        ibp_max,
        crown_max,
        alpha_max,
    }
}

/// Format measurement results into a string for test output.
fn format_measurements(label: &str, results: &[EpsMeasurement]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "  Phase 3: {label}");
    for r in results {
        let cr_ratio = if r.ibp_max > 0.0 {
            r.crown_max / r.ibp_max
        } else {
            f32::NAN
        };
        let _ = writeln!(
            out,
            "    eps={:.4} ibp_max={:.6} crown_max={:.6} alpha_max={:.6} crown/ibp={:.4}",
            r.eps, r.ibp_max, r.crown_max, r.alpha_max, cr_ratio
        );
    }
    out
}

fn default_alpha_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        spsa_samples: 3,
        learning_rate: 0.05,
        lr_decay: 0.95,
        gradient_method: GradientMethod::Spsa,
        ..AlphaCrownConfig::default()
    }
}

const EPS_VALUES: [f32; 4] = [0.001, 0.01, 0.05, 0.1];

/// Phase 3: 2x2 identity vs asymmetric weights.
///
/// Decision gate from designs/2026-03-04-286-attention-bilinear-alternative.md.
/// Reference: auto_LiRPA/operators/bivariate.py:39-75
#[ntest::timeout(120000)]
#[test]
fn test_phase3_mccormick_vs_ibp_2x2() {
    let center = array![[0.5_f32, 0.3], [0.2, 0.6]].into_dyn();
    let config = default_alpha_config(10);

    // Identity weights (maximally correlated Q/K)
    let graph_id = build_bilinear_graph_with_weights(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        Some(arr1(&[0.0, 0.0])),
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        Some(arr1(&[0.0, 0.0])),
        1.0,
    );
    let id_results: Vec<_> = EPS_VALUES
        .iter()
        .map(|&e| measure_at_eps(&graph_id, &center, e, &config))
        .collect();
    eprint!("{}", format_measurements("2x2 Identity", &id_results));

    // Asymmetric weights (partially decorrelated Q/K)
    let graph_asym = build_bilinear_graph_with_weights(
        array![[0.8_f32, 0.3], [-0.2, 0.7]],
        Some(arr1(&[0.1, -0.1])),
        array![[0.5_f32, -0.4], [0.6, 0.9]],
        Some(arr1(&[-0.05, 0.05])),
        1.0,
    );
    let asym_results: Vec<_> = EPS_VALUES
        .iter()
        .map(|&e| measure_at_eps(&graph_asym, &center, e, &config))
        .collect();
    eprint!("{}", format_measurements("2x2 Asymmetric", &asym_results));
}

/// Phase 3: 4x4 asymmetric weights — larger graph scaling.
#[ntest::timeout(120000)]
#[test]
fn test_phase3_mccormick_vs_ibp_4x4() {
    let wq = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.8, 0.1, -0.2, 0.3, 0.0, 0.7, 0.4, -0.1, -0.3, 0.2, 0.9, 0.0, 0.1, -0.1, 0.0, 0.6,
        ],
    )
    .expect("invariant: valid 4x4 weight matrix");
    let wk = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.2, 0.1, 0.4, 0.8, -0.1, 0.3, -0.2, 0.1, 0.6, 0.5, 0.3, 0.0, -0.4, 0.7,
        ],
    )
    .expect("invariant: valid 4x4 weight matrix");
    let graph = build_bilinear_graph_with_weights(wq, None, wk, None, 0.5);

    let center = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, 0.3, 0.2, 0.4, 0.1, 0.6, 0.3, 0.2, 0.4, 0.2, 0.5, 0.1, 0.3, 0.4, 0.1, 0.6,
        ],
    )
    .expect("invariant: valid 4x4 center matrix")
    .into_dyn();

    let config = default_alpha_config(15);
    let results: Vec<_> = EPS_VALUES
        .iter()
        .map(|&e| measure_at_eps(&graph, &center, e, &config))
        .collect();
    let report = format_measurements("4x4 Asymmetric", &results);
    let crown_tighter = results.iter().any(|r| r.crown_max < r.ibp_max * 0.999);
    eprint!("{}  decision: crown_tighter={crown_tighter}", report);
}
