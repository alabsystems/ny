// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase 3 decision gate: broadcast+alpha McCormick vs IBP with ReLU activations.
//!
//! Uses the joint DAG alpha-CROWN path that optimizes both ReLU and BilinearCrown
//! alphas simultaneously, matching the real attention pattern.
//!
//! Reference: designs/2026-03-04-286-attention-bilinear-alternative.md Phase 3

use std::fmt::Write as _;

use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::AlphaCrownConfig;
use crate::layers::activations::ReLULayer;
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::Layer;
use crate::network::core::graph::{GraphNetwork, GraphNode};

use super::measurement::{assert_valid_bounds, max_bound_width};

const EPS_VALUES: [f32; 4] = [0.001, 0.01, 0.05, 0.1];

/// Extended measurement result including total width and joint alpha-CROWN.
struct EpsMeasurementFull {
    eps: f32,
    ibp_total: f32,
    ibp_max: f32,
    crown_total: f32,
    crown_max: f32,
    alpha_total: f32,
    alpha_max: f32,
    soundness_ok: bool,
    n_output_elements: usize,
}

/// Build attention subgraph with ReLU: Input -> Linear+ReLU (Q) + Linear+ReLU (K)
/// -> BilinearCrown(Q @ K^T).
fn build_attention_relu_graph(
    wq: Array2<f32>,
    bias_q: Option<ndarray::Array1<f32>>,
    wk: Array2<f32>,
    bias_k: Option<ndarray::Array1<f32>>,
    scale: f32,
) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let proj_q = LinearLayer::new(wq, bias_q).expect("invariant: valid Q weight matrix");
    graph.add_node(GraphNode::from_input("proj_q", Layer::Linear(proj_q)));
    graph.add_node(GraphNode::new(
        "relu_q",
        Layer::ReLU(ReLULayer),
        vec!["proj_q".to_string()],
    ));
    let proj_k = LinearLayer::new(wk, bias_k).expect("invariant: valid K weight matrix");
    graph.add_node(GraphNode::from_input("proj_k", Layer::Linear(proj_k)));
    graph.add_node(GraphNode::new(
        "relu_k",
        Layer::ReLU(ReLULayer),
        vec!["proj_k".to_string()],
    ));
    let bilinear = BilinearCrownLayer::new(true, Some(scale));
    graph.add_node(GraphNode::binary(
        "qk",
        Layer::BilinearCrown(bilinear),
        "relu_q",
        "relu_k",
    ));
    graph.set_output("qk");
    graph
}

/// Compute total bound width (sum of upper - lower across all elements).
fn total_bound_width(bt: &BoundedTensor) -> f32 {
    bt.lower()
        .iter()
        .zip(bt.upper().iter())
        .map(|(&l, &u)| (u - l).max(0.0))
        .sum()
}

/// Verify soundness of bounds by sampling concrete outputs.
/// Samples 3^n_inputs grid points and checks all fall within bounds.
fn verify_soundness_by_sampling(
    graph: &GraphNetwork,
    bounds: &BoundedTensor,
    center: &ArrayD<f32>,
    eps: f32,
    shape: &[usize],
) -> bool {
    let n = center.len();
    if n > 8 {
        return true;
    }
    let levels = [0.0_f32, 0.5, 1.0];
    let n_samples = 3_usize.pow(n as u32);
    let lower_vals: Vec<f32> = center.iter().map(|&c| c - eps).collect();
    for sample_idx in 0..n_samples {
        let vals: Vec<f32> = (0..n)
            .map(|dim| {
                let level_idx = (sample_idx / 3_usize.pow(dim as u32)) % 3;
                lower_vals[dim] + 2.0 * eps * levels[level_idx]
            })
            .collect();
        let point =
            ArrayD::from_shape_vec(IxDyn(shape), vals).expect("invariant: valid sample shape");
        let point_input = BoundedTensor::concrete(point).expect("invariant: valid concrete tensor");
        let exact = match graph.propagate_ibp(&point_input) {
            Ok(b) => b,
            Err(_) => return false,
        };
        for ((out_val, &lower), &upper) in exact
            .lower()
            .iter()
            .zip(bounds.lower().iter())
            .zip(bounds.upper().iter())
        {
            if *out_val < lower - 1e-3 || *out_val > upper + 1e-3 {
                return false;
            }
        }
    }
    true
}

/// Run full IBP/CROWN/joint-alpha-CROWN comparison at a single eps value.
fn measure_full_at_eps(
    graph: &GraphNetwork,
    center: &ArrayD<f32>,
    eps: f32,
    shape: &[usize],
    alpha_config: &AlphaCrownConfig,
) -> EpsMeasurementFull {
    let lower = center - eps;
    let upper = center + eps;
    let input =
        BoundedTensor::new(lower, upper).expect("invariant: center +/- eps produces valid bounds");

    let ibp_bounds = graph
        .propagate_ibp(&input)
        .expect("invariant: IBP succeeds on valid graph");
    let ibp_total = total_bound_width(&ibp_bounds);
    let ibp_max = max_bound_width(&ibp_bounds);
    let n_output = ibp_bounds.lower().len();

    let crown = graph
        .propagate_crown_batched(&input)
        .unwrap_or_else(|error| {
            panic!("CROWN must cover the ReLU-attention fixture at eps={eps}: {error}")
        });
    assert_valid_bounds(&crown, "CROWN", eps);
    let crown_total = total_bound_width(&crown);
    let crown_max = max_bound_width(&crown);

    let alpha = graph
        .propagate_alpha_crown_with_config(&input, alpha_config)
        .unwrap_or_else(|error| {
            panic!("alpha-CROWN must cover the ReLU-attention fixture at eps={eps}: {error}")
        });
    assert_valid_bounds(&alpha, "alpha-CROWN", eps);
    let alpha_total = total_bound_width(&alpha);
    let alpha_max = max_bound_width(&alpha);
    let soundness_ok = verify_soundness_by_sampling(graph, &alpha, center, eps, shape);

    EpsMeasurementFull {
        eps,
        ibp_total,
        ibp_max,
        crown_total,
        crown_max,
        alpha_total,
        alpha_max,
        soundness_ok,
        n_output_elements: n_output,
    }
}

/// Format measurement results into a report string.
fn format_report(label: &str, results: &[EpsMeasurementFull]) -> String {
    let mut out = String::new();
    let n_elem = results.first().map_or(0, |r| r.n_output_elements);
    let _ = writeln!(out, "\n  Phase 3: {label} ({n_elem} output elements)");
    let _ = writeln!(
        out,
        "  {:>7} | {:>12} {:>12} {:>12} | {:>8}",
        "eps", "IBP total", "CROWN total", "alpha total", "alpha/IBP"
    );
    let _ = writeln!(out, "  {}", "-".repeat(70));
    for r in results {
        let pct = if r.ibp_total > 0.0 && r.alpha_total.is_finite() {
            (1.0 - r.alpha_total / r.ibp_total) * 100.0
        } else {
            f32::NAN
        };
        let snd = if r.soundness_ok { "ok" } else { "FAIL" };
        let _ = writeln!(
            out,
            "  {:>7.4} | {:>12.6} {:>12.6} {:>12.6} | {:>+7.2}% [{snd}]",
            r.eps, r.ibp_total, r.crown_total, r.alpha_total, pct
        );
    }
    let _ = writeln!(
        out,
        "  {:>7} | {:>12} {:>12} {:>12} | {:>8}",
        "eps", "IBP max", "CROWN max", "alpha max", "decision"
    );
    let _ = writeln!(out, "  {}", "-".repeat(70));
    for r in results {
        let decision = if r.alpha_max.is_finite() && r.alpha_max < r.ibp_max * 0.999 {
            "TIGHTER"
        } else if r.alpha_max.is_nan() {
            "FAILED"
        } else {
            "~IBP"
        };
        let _ = writeln!(
            out,
            "  {:>7.4} | {:>12.6} {:>12.6} {:>12.6} | {:>8}",
            r.eps, r.ibp_max, r.crown_max, r.alpha_max, decision
        );
    }
    out
}

/// Format a decision summary for one scenario.
fn format_decision(label: &str, results: &[EpsMeasurementFull]) -> String {
    let n_tighter = results
        .iter()
        .filter(|r| r.alpha_total.is_finite() && r.alpha_total < r.ibp_total * 0.999)
        .count();
    let all_sound = results.iter().all(|r| r.soundness_ok);
    let any_fail = results.iter().any(|r| r.alpha_total.is_nan());
    format!(
        "  {label}: tighter at {n_tighter}/{} eps. sound={}, failures={}\n",
        results.len(),
        if all_sound { "pass" } else { "FAIL" },
        if any_fail { "yes" } else { "none" },
    )
}

/// Phase 3 decision gate: broadcast+alpha McCormick vs IBP with ReLU activations.
///
/// Graph: Input [2,2] -> Linear+ReLU (Q) + Linear+ReLU (K) -> BilinearCrown(Q @ K^T)
///
/// Uses joint DAG alpha-CROWN (both ReLU and BilinearCrown alphas).
/// Reference: auto_LiRPA operators/bivariate.py:39-75
/// Reference: auto_LiRPA operators/linear.py:512-585
#[ntest::timeout(120000)]
#[test]
fn test_phase3_decision_gate_relu_attention() {
    let shape = [2, 2];
    let center = arr2(&[[0.5_f32, 1.0], [1.0, 0.5]]).into_dyn();
    let config = AlphaCrownConfig {
        iterations: 30,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let mut report = String::from(
        "\n=== Phase 3 Decision Gate: broadcast+alpha McCormick vs IBP ===\n\
         Graph: Input [2,2] -> Linear+ReLU (Q/K) -> BilinearCrown(Q@K^T)\n",
    );

    // Scenario 1: Asymmetric Q/K weights (decorrelated paths — favorable for McCormick)
    let graph_asym = build_attention_relu_graph(
        arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
        Some(arr1(&[0.1, -0.2])),
        arr2(&[[-0.5_f32, 0.7], [0.6, -0.4]]),
        Some(arr1(&[-0.1, 0.15])),
        1.0 / 2.0_f32.sqrt(),
    );
    let asym_results: Vec<_> = EPS_VALUES
        .iter()
        .map(|&e| measure_full_at_eps(&graph_asym, &center, e, &shape, &config))
        .collect();
    report.push_str(&format_report("Asymmetric Q/K with ReLU", &asym_results));
    report.push_str(&format_decision("Asymmetric", &asym_results));

    // Scenario 2: Similar Q/K weights (correlated paths — unfavorable for McCormick)
    let graph_corr = build_attention_relu_graph(
        arr2(&[[0.7_f32, 0.2], [0.1, 0.8]]),
        Some(arr1(&[0.05, -0.05])),
        arr2(&[[0.65_f32, 0.25], [0.15, 0.75]]),
        Some(arr1(&[0.0, 0.0])),
        1.0 / 2.0_f32.sqrt(),
    );
    let corr_results: Vec<_> = EPS_VALUES
        .iter()
        .map(|&e| measure_full_at_eps(&graph_corr, &center, e, &shape, &config))
        .collect();
    report.push_str(&format_report("Correlated Q/K with ReLU", &corr_results));
    report.push_str(&format_decision("Correlated", &corr_results));

    // Overall decision
    let any_tighter = asym_results
        .iter()
        .chain(corr_results.iter())
        .any(|r| r.alpha_total.is_finite() && r.alpha_total < r.ibp_total * 0.999);
    let all_sound = asym_results
        .iter()
        .chain(corr_results.iter())
        .all(|r| r.soundness_ok);

    report.push_str("\n=== OVERALL DECISION ===\n");
    if any_tighter && all_sound {
        report.push_str("  PASS: broadcast+alpha McCormick beats IBP.\n");
    } else if any_tighter {
        report.push_str("  BLOCKED: tighter but UNSOUND.\n");
    } else {
        report.push_str("  FAIL: does NOT beat IBP. Accept IBP at attention.\n");
    }
    report.push_str("=== END Phase 3 ===\n");

    eprint!("{report}");

    assert!(
        all_sound,
        "Phase 3: alpha-CROWN bounds must be sound (contain all concrete outputs)"
    );
}
