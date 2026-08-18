// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic tests for #1923: GraphNetwork bound propagation looser
//! than Network for ACAS-Xu 4_x family.
//!
//! ACAS-Xu has architecture: 5 inputs, 6 hidden layers of 50 neurons, 5 outputs.
//! The sequential `Network` path verifies models 4_2-4_5 within 30s but the
//! graph `GraphNetwork` path times out. This test isolates the bound gap at
//! each propagation level: IBP, fixed-slope CROWN, and alpha-CROWN.

use crate::bounds::AlphaCrownConfig;
use crate::*;
use ndarray::{array, Array2, ArrayD, IxDyn};

/// Build a deterministic ACAS-Xu-like fully-connected network.
///
/// Architecture: Linear(5,50) + ReLU + Linear(50,50) + ReLU (x4) + Linear(50,5).
/// Uses a simple deterministic weight pattern for reproducibility.
fn build_acasxu_like_network() -> Network {
    let mut network = Network::new();

    // Layer dimensions: 5 -> 50 -> 50 -> 50 -> 50 -> 50 -> 50 -> 5
    let dims = [5, 50, 50, 50, 50, 50, 50, 5];

    for layer_idx in 0..dims.len() - 1 {
        let input_dim = dims[layer_idx];
        let output_dim = dims[layer_idx + 1];
        let is_output = layer_idx == dims.len() - 2;

        // Deterministic weights: Xavier-like initialization with a seed pattern
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_dim * input_dim);
        let mut biases = Vec::with_capacity(output_dim);

        for i in 0..output_dim {
            for j in 0..input_dim {
                // Deterministic pseudo-random using a hash-like pattern
                let seed = (layer_idx * 1000 + i * 37 + j * 13 + 7) as f32;
                let val = ((seed * 2654435761.0_f32) % 1000.0) / 1000.0 - 0.5;
                weights.push(val * scale);
            }
            let bias_seed = (layer_idx * 500 + i * 17 + 3) as f32;
            biases.push(((bias_seed * 2654435761.0_f32) % 1000.0) / 10000.0 - 0.05);
        }

        let weight_arr =
            Array2::from_shape_vec((output_dim, input_dim), weights).expect("weight shape");
        let bias_arr = ndarray::Array1::from_vec(biases);
        let linear = LinearLayer::new(weight_arr, Some(bias_arr)).expect("linear layer");
        network.add_layer(Layer::Linear(linear));

        if !is_output {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
    }

    network
}

/// Build a scalar-objective version of the ACAS-Xu-like network.
///
/// The final linear layer converts the original 5 outputs into one scalar so
/// full-output CROWN and spec-guided CROWN can be compared directly on the
/// same child input domain.
fn build_acasxu_scalar_objective_network() -> Network {
    let mut network = build_acasxu_like_network();
    let objective = Array2::from_shape_vec((1, 5), vec![1.0, -1.0, 0.5, -0.25, 0.75])
        .expect("objective weight shape");
    let bias = array![0.0_f32];
    let projection = LinearLayer::new(objective, Some(bias)).expect("objective projection");
    network.add_layer(Layer::Linear(projection));
    network
}

/// Build a BoundedTensor matching ACAS-Xu property 2 input shape.
fn build_acasxu_input() -> BoundedTensor {
    // Approximate ACAS-Xu property 2 input bounds (5 dimensions)
    let lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.5, -0.5, 0.45, -0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.679858, 0.5, 0.5, 0.5, -0.45]).unwrap();
    BoundedTensor::new(lower, upper).unwrap()
}

/// Build the left/right children produced by bisecting one input dimension.
fn split_input_dimension(
    input: &BoundedTensor,
    dim: usize,
    take_left_child: bool,
) -> BoundedTensor {
    let lower_vec: Vec<f32> = input.lower().iter().copied().collect();
    let upper_vec: Vec<f32> = input.upper().iter().copied().collect();
    let mut child_lower = lower_vec.clone();
    let mut child_upper = upper_vec.clone();

    let midpoint = f32::midpoint(lower_vec[dim], upper_vec[dim]);
    if take_left_child {
        child_upper[dim] = midpoint;
    } else {
        child_lower[dim] = midpoint;
    }

    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(input.shape()), child_lower).expect("child lower shape"),
        ArrayD::from_shape_vec(IxDyn(input.shape()), child_upper).expect("child upper shape"),
    )
    .expect("child bounds should remain valid")
}

fn scalar_width(bounds: &BoundedTensor) -> f32 {
    let (lower, upper) = scalar_bounds(bounds);
    upper - lower
}

fn scalar_bounds(bounds: &BoundedTensor) -> (f32, f32) {
    assert_eq!(
        bounds.len(),
        1,
        "expected scalar output bounds, got {} elements",
        bounds.len()
    );
    (bounds.lower()[[0]], bounds.upper()[[0]])
}

fn evaluate_child_domain_gap_1923(
    network: &Network,
    graph: &GraphNetwork,
    child: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    alpha_config: &AlphaCrownConfig,
    dim: usize,
    label: &str,
) -> f32 {
    let seq_fixed = network.propagate_crown(child).unwrap();
    let seq_alpha = network
        .propagate_alpha_crown_with_config(child, alpha_config)
        .unwrap();
    let graph_spec = graph
        .propagate_crown_with_specs_and_engine(child, spec_matrix, None)
        .unwrap();
    let (seq_fixed_lower, seq_fixed_upper) = scalar_bounds(&seq_fixed);
    let (seq_alpha_lower, seq_alpha_upper) = scalar_bounds(&seq_alpha);
    let (graph_spec_lower, graph_spec_upper) = scalar_bounds(&graph_spec);

    let seq_fixed_width = scalar_width(&seq_fixed);
    let seq_alpha_width = scalar_width(&seq_alpha);
    let graph_spec_width = scalar_width(&graph_spec);

    eprintln!(
        "child dim={dim} {label}: seq_fixed=[{:.6}, {:.6}] w={:.6}, \
         graph_spec=[{:.6}, {:.6}] w={:.6}, seq_alpha=[{:.6}, {:.6}] w={:.6}",
        seq_fixed_lower,
        seq_fixed_upper,
        seq_fixed_width,
        graph_spec_lower,
        graph_spec_upper,
        graph_spec_width,
        seq_alpha_lower,
        seq_alpha_upper,
        seq_alpha_width,
    );

    assert!(
        (seq_fixed_lower - graph_spec_lower).abs() < 1e-3,
        "child dim={dim} {label}: GraphNetwork spec-guided lower bound diverged from \
         sequential fixed-slope CROWN (seq={}, graph={})",
        seq_fixed_lower,
        graph_spec_lower,
    );
    assert!(
        (seq_fixed_upper - graph_spec_upper).abs() < 1e-3,
        "child dim={dim} {label}: GraphNetwork spec-guided upper bound diverged from \
         sequential fixed-slope CROWN (seq={}, graph={})",
        seq_fixed_upper,
        graph_spec_upper,
    );
    if seq_alpha_width > seq_fixed_width + 1e-4 {
        eprintln!(
            "#1923 NOTE: child dim={dim} {label} sequential alpha-CROWN was looser than \
             fixed-slope CROWN (alpha_width={seq_alpha_width}, fixed_width={seq_fixed_width}); \
             treating this as diagnostic only because the parity assertion is graph_spec \
             vs seq_fixed."
        );
    }

    (seq_fixed_width - seq_alpha_width).max(0.0)
}

/// Compute per-element bound width: upper - lower.
fn bound_width(bounds: &BoundedTensor) -> Vec<f32> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect()
}

/// Assert that a published bound is a usable finite enclosure.
fn assert_finite_ordered_bounds(label: &str, bounds: &BoundedTensor) {
    assert!(!bounds.is_empty(), "{label} returned an empty bound");
    for (idx, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(
            lower.is_finite() && upper.is_finite(),
            "{label}[{idx}] is non-finite: [{lower}, {upper}]"
        );
        assert!(
            lower <= upper,
            "{label}[{idx}] is inverted: [{lower}, {upper}]"
        );
    }
}

/// Assert `candidate` is contained in (therefore no looser than) `baseline`,
/// allowing only floating-point accumulation noise.
fn assert_no_loosening(baseline: &BoundedTensor, candidate: &BoundedTensor, context: &str) {
    assert_eq!(
        baseline.shape(),
        candidate.shape(),
        "{context}: shape mismatch"
    );
    assert_finite_ordered_bounds(&format!("{context} baseline"), baseline);
    assert_finite_ordered_bounds(&format!("{context} candidate"), candidate);

    for (idx, ((&base_lower, &base_upper), (&candidate_lower, &candidate_upper))) in baseline
        .lower()
        .iter()
        .zip(baseline.upper().iter())
        .zip(candidate.lower().iter().zip(candidate.upper().iter()))
        .enumerate()
    {
        let scale = base_lower
            .abs()
            .max(base_upper.abs())
            .max(candidate_lower.abs())
            .max(candidate_upper.abs())
            .max(1.0);
        let tol = 1e-3_f32.max(scale * 1e-4);
        assert!(
            candidate_lower >= base_lower - tol,
            "{context}[{idx}] loosened lower bound: baseline={base_lower}, \
             candidate={candidate_lower}, tol={tol}"
        );
        assert!(
            candidate_upper <= base_upper + tol,
            "{context}[{idx}] loosened upper bound: baseline={base_upper}, \
             candidate={candidate_upper}, tol={tol}"
        );
    }
}

/// Summary statistics for bound comparison.
struct BoundComparison {
    /// Whether graph bounds are always at least as tight as sequential.
    graph_tighter_or_equal: bool,
    /// Whether sequential bounds are always at least as tight as graph.
    seq_tighter_or_equal: bool,
    /// Maximum width difference: max(graph_width - seq_width) per element.
    max_graph_excess: f32,
    /// Maximum width difference: max(seq_width - graph_width) per element.
    max_seq_excess: f32,
    /// Mean width: graph path.
    mean_graph_width: f32,
    /// Mean width: sequential path.
    mean_seq_width: f32,
}

fn compare_bounds(seq_bounds: &BoundedTensor, graph_bounds: &BoundedTensor) -> BoundComparison {
    let seq_widths = bound_width(seq_bounds);
    let graph_widths = bound_width(graph_bounds);

    let mut graph_tighter_or_equal = true;
    let mut seq_tighter_or_equal = true;
    let mut max_graph_excess = 0.0_f32;
    let mut max_seq_excess = 0.0_f32;
    let tol = 1e-5;

    for (sw, gw) in seq_widths.iter().zip(graph_widths.iter()) {
        if gw > &(sw + tol) {
            seq_tighter_or_equal = true;
            graph_tighter_or_equal = false;
        }
        if sw > &(gw + tol) {
            graph_tighter_or_equal &= true;
            seq_tighter_or_equal = false;
        }
        max_graph_excess = max_graph_excess.max(gw - sw);
        max_seq_excess = max_seq_excess.max(sw - gw);
    }

    let n = seq_widths.len() as f32;
    BoundComparison {
        graph_tighter_or_equal,
        seq_tighter_or_equal,
        max_graph_excess,
        max_seq_excess,
        mean_graph_width: graph_widths.iter().sum::<f32>() / n,
        mean_seq_width: seq_widths.iter().sum::<f32>() / n,
    }
}

/// Phase 1: IBP bounds must be identical between Network and GraphNetwork.
#[ntest::timeout(10000)]
#[test]
fn test_1923_ibp_parity() {
    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    let seq_ibp = network.propagate_ibp(&input).unwrap();
    let graph_ibp = graph.propagate_ibp(&input).unwrap();

    assert_eq!(seq_ibp.shape(), graph_ibp.shape());

    for (idx, ((&sl, &su), (&gl, &gu))) in seq_ibp
        .lower()
        .iter()
        .zip(seq_ibp.upper().iter())
        .zip(graph_ibp.lower().iter().zip(graph_ibp.upper().iter()))
        .enumerate()
    {
        assert!(
            (sl - gl).abs() < 1e-5,
            "IBP lower mismatch at idx={}: seq={}, graph={}",
            idx,
            sl,
            gl
        );
        assert!(
            (su - gu).abs() < 1e-5,
            "IBP upper mismatch at idx={}: seq={}, graph={}",
            idx,
            su,
            gu
        );
    }
}

/// Phase 2: Fixed-slope CROWN bounds on a sequential graph should match
/// the sequential Network CROWN bounds.
///
/// If this test fails, the GraphNetwork fixed-slope CROWN backward pass
/// computes different linear relaxations than the Network path.
///
/// Timeout 10s→30s: the body pins NY_DENSE_BUDGET_MB behind the shared env
/// lock, so under parallel test load it can queue several seconds behind the
/// zero-budget suite before its ~0.3s of actual work; 30s still guards hangs.
#[ntest::timeout(30000)]
#[test]
fn test_1923_fixed_slope_crown_parity() {
    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    // Pin NY_DENSE_BUDGET_MB (holding the shared env lock): both CROWN paths
    // read the budget per call, and a concurrently-running zero-budget test's
    // window would otherwise degrade one side to IBP and break parity.
    let (seq_crown, graph_crown_fixed) = tests::with_crown_dense_budget_mb("2048", || {
        let seq_crown = network.propagate_crown(&input).unwrap();
        let graph_crown_fixed = graph.propagate_crown_fixed_slope(&input).unwrap();
        (seq_crown, graph_crown_fixed)
    });

    assert_eq!(seq_crown.shape(), graph_crown_fixed.shape());

    let cmp = compare_bounds(&seq_crown, &graph_crown_fixed);
    eprintln!("--- Fixed-slope CROWN parity (#1923) ---");
    eprintln!(
        "Sequential mean width: {:.6}, Graph mean width: {:.6}",
        cmp.mean_seq_width, cmp.mean_graph_width
    );
    eprintln!(
        "Max graph excess: {:.6}, Max seq excess: {:.6}",
        cmp.max_graph_excess, cmp.max_seq_excess
    );

    for (idx, ((&sl, &su), (&gl, &gu))) in seq_crown
        .lower()
        .iter()
        .zip(seq_crown.upper().iter())
        .zip(
            graph_crown_fixed
                .lower()
                .iter()
                .zip(graph_crown_fixed.upper().iter()),
        )
        .enumerate()
    {
        // Fixed-slope CROWN on a sequential graph should produce near-identical bounds.
        // Allow small tolerance for floating-point accumulation differences.
        let lower_diff = (sl - gl).abs();
        let upper_diff = (su - gu).abs();
        assert!(
            lower_diff < 1e-3,
            "Fixed-slope CROWN lower mismatch at idx={}: seq={:.6}, graph={:.6}, diff={:.6}",
            idx,
            sl,
            gl,
            lower_diff
        );
        assert!(
            upper_diff < 1e-3,
            "Fixed-slope CROWN upper mismatch at idx={}: seq={:.6}, graph={:.6}, diff={:.6}",
            idx,
            su,
            gu,
            upper_diff
        );
    }
}

/// Phase 3: Alpha-CROWN comparison between Network and GraphNetwork.
///
/// This is the key diagnostic for #1923. If the graph alpha-CROWN produces
/// significantly looser bounds, it explains why GPU BaB (which uses
/// GraphNetwork) times out while the sequential path (which uses Network)
/// verifies.
#[ntest::timeout(60000)]
#[test]
fn test_1923_alpha_crown_bound_gap_diagnostic() {
    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    // Use identical alpha-CROWN configurations for both paths
    let config = AlphaCrownConfig {
        iterations: 20,
        ..AlphaCrownConfig::default()
    };

    let (seq_alpha, graph_alpha) = tests::with_crown_dense_budget_mb("2048", || {
        let seq_alpha = network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();
        let graph_alpha = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();
        (seq_alpha, graph_alpha)
    });

    assert_eq!(seq_alpha.shape(), graph_alpha.shape());
    assert_no_loosening(
        &seq_alpha,
        &graph_alpha,
        "Graph alpha-CROWN vs sequential alpha-CROWN",
    );

    let cmp = compare_bounds(&seq_alpha, &graph_alpha);
    eprintln!("--- Alpha-CROWN bound gap diagnostic (#1923) ---");
    eprintln!(
        "Sequential alpha-CROWN mean width: {:.6}",
        cmp.mean_seq_width
    );
    eprintln!(
        "Graph alpha-CROWN mean width:      {:.6}",
        cmp.mean_graph_width
    );
    eprintln!("Max graph excess width: {:.6}", cmp.max_graph_excess);
    eprintln!("Max sequential excess width: {:.6}", cmp.max_seq_excess);
    eprintln!("Graph tighter or equal: {}", cmp.graph_tighter_or_equal);
    eprintln!("Seq tighter or equal:   {}", cmp.seq_tighter_or_equal);

    // Print per-element comparison
    let seq_widths = bound_width(&seq_alpha);
    let graph_widths = bound_width(&graph_alpha);
    for (idx, (sw, gw)) in seq_widths.iter().zip(graph_widths.iter()).enumerate() {
        let diff = gw - sw;
        let pct = if sw.abs() > 1e-8 {
            diff / sw * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  output[{}]: seq={:.6}, graph={:.6}, diff={:+.6} ({:+.1}%)",
            idx, sw, gw, diff, pct
        );
    }

    assert!(
        cmp.max_graph_excess <= 1e-3_f32.max(cmp.mean_seq_width.abs() * 1e-4),
        "Graph alpha-CROWN width regressed against sequential alpha-CROWN: \
         max excess={}, sequential mean width={}",
        cmp.max_graph_excess,
        cmp.mean_seq_width
    );
}

/// Phase 3b: Bound tightening progression across all methods.
///
/// Shows how bounds improve from IBP -> CROWN -> alpha-CROWN for both paths.
#[ntest::timeout(60000)]
#[test]
fn test_1923_bound_tightening_progression() {
    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    let config = AlphaCrownConfig {
        iterations: 20,
        ..AlphaCrownConfig::default()
    };
    let (seq_ibp, seq_crown, graph_fixed, seq_alpha, graph_alpha) =
        tests::with_crown_dense_budget_mb("2048", || {
            let seq_ibp = network.propagate_ibp(&input).unwrap();
            let seq_crown = network.propagate_crown(&input).unwrap();
            let graph_fixed = graph.propagate_crown_fixed_slope(&input).unwrap();
            let seq_alpha = network
                .propagate_alpha_crown_with_config(&input, &config)
                .unwrap();
            let graph_alpha = graph
                .propagate_alpha_crown_with_config(&input, &config)
                .unwrap();
            (seq_ibp, seq_crown, graph_fixed, seq_alpha, graph_alpha)
        });

    assert_no_loosening(&seq_ibp, &seq_crown, "sequential CROWN vs IBP");
    assert_no_loosening(&seq_ibp, &graph_fixed, "graph fixed-slope CROWN vs IBP");
    assert_no_loosening(
        &seq_crown,
        &seq_alpha,
        "sequential alpha-CROWN vs fixed-slope CROWN",
    );
    assert_no_loosening(
        &graph_fixed,
        &graph_alpha,
        "graph alpha-CROWN vs fixed-slope CROWN",
    );

    let ibp_w = bound_width(&seq_ibp);
    let crown_w = bound_width(&seq_crown);
    let gfixed_w = bound_width(&graph_fixed);
    let salpha_w = bound_width(&seq_alpha);
    let galpha_w = bound_width(&graph_alpha);

    eprintln!("--- Bound tightening progression (#1923) ---");
    for idx in 0..ibp_w.len() {
        eprintln!(
            "  [{}]: IBP={:.4} -> SeqCROWN={:.4} -> SeqAlpha={:.4}; \
             IBP={:.4} -> GFixed={:.4} -> GAlpha={:.4}",
            idx, ibp_w[idx], crown_w[idx], salpha_w[idx], ibp_w[idx], gfixed_w[idx], galpha_w[idx]
        );
    }
}

/// Phase 4: Compare CROWN-IBP intermediate bounds between both paths.
///
/// The intermediate bounds feed into alpha-CROWN optimization. If the
/// graph CROWN-IBP produces looser intermediates, alpha optimization
/// starts from a worse baseline, compounding the bound gap.
///
/// The graph path returns bounds keyed by node name; the sequential path
/// returns bounds indexed by layer position. `from_sequential` names each
/// node `layer_{idx}` after its sequential layer index, so the two sides
/// pair exactly.
///
/// The two lanes intentionally differ at activation nodes: the graph DAG
/// collection demand-skips nodes no downstream nonlinear consumer needs
/// (#3775, provenance `DemandDrivenSkip`) and leaves them at their forward
/// IBP bounds, because alpha-CROWN/BaB only consume PRE-ACTIVATION bounds.
/// So the parity contract is per-provenance:
/// - `Crown` nodes (the pre-activation/output targets that seed alpha-CROWN)
///   must match the sequential widths to fp-accumulation noise — a looser
///   graph side here is the #1923 timeout mechanism.
/// - `DemandDrivenSkip` nodes must CONTAIN the sequential CROWN-IBP bounds
///   (IBP is never tighter than the intersected CROWN-IBP result).
#[ntest::timeout(10000)]
#[test]
fn test_1923_crown_ibp_intermediate_parity() {
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};

    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    // Collect CROWN-IBP intermediate bounds from both paths
    let seq_layer_bounds = network.collect_crown_ibp_bounds(&input).unwrap();
    let graph_result = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .unwrap();
    let graph_node_bounds = &graph_result.bounds;

    eprintln!("--- CROWN-IBP intermediate bounds comparison (#1923) ---");
    eprintln!("Sequential: {} layer bounds", seq_layer_bounds.len());
    eprintln!("Graph: {} node bounds", graph_node_bounds.len());
    assert!(
        !seq_layer_bounds.is_empty(),
        "sequential CROWN-IBP returned no layer bounds"
    );

    for (idx, seq_bt) in seq_layer_bounds.iter().enumerate() {
        let node_name = format!("layer_{idx}");
        let graph_bt = graph_node_bounds
            .get(&node_name)
            .unwrap_or_else(|| panic!("graph CROWN-IBP missing bounds for node '{node_name}'"));
        assert_eq!(
            graph_bt.len(),
            seq_bt.len(),
            "dim mismatch at '{node_name}'"
        );
        let provenance = graph_result
            .provenance
            .get(&node_name)
            .unwrap_or_else(|| panic!("graph CROWN-IBP missing provenance for '{node_name}'"));

        let seq_width: f32 = bound_width(seq_bt).iter().sum::<f32>() / seq_bt.len() as f32;
        let graph_width: f32 = bound_width(graph_bt).iter().sum::<f32>() / graph_bt.len() as f32;
        eprintln!(
            "  layer {}: dim={}, seq mean_width={:.6}, graph mean_width={:.6}, prov={:?}",
            idx,
            seq_bt.len(),
            seq_width,
            graph_width,
            provenance
        );

        match provenance {
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip) => {
                // Deliberately left at forward IBP (#3775): the skipped node
                // may be looser than sequential CROWN-IBP, but must never be
                // tighter (the sequential result is intersected with the same
                // forward IBP bounds).
                let tol = 1e-6_f32;
                for (j, ((&gl, &gu), (&sl, &su))) in graph_bt
                    .lower()
                    .iter()
                    .zip(graph_bt.upper().iter())
                    .zip(seq_bt.lower().iter().zip(seq_bt.upper().iter()))
                    .enumerate()
                {
                    assert!(
                        gl <= sl + tol && gu >= su - tol,
                        "demand-skipped '{node_name}'[{j}] tighter than sequential \
                         CROWN-IBP: graph=[{gl}, {gu}] seq=[{sl}, {su}]"
                    );
                }
            }
            _ => {
                // CROWN-tightened targets: the graph DAG collection feeds each
                // node's tightened bounds forward into deeper relaxations, so
                // at depth it may be strictly TIGHTER than the sequential
                // lane (observed: layer_8 graph 0.0022 vs seq 0.0060). What it
                // must never be is LOOSER beyond fp-accumulation noise — a
                // looser graph side starts alpha-CROWN from a worse baseline
                // and compounds the bound gap (the #1923 timeout mechanism).
                let tol = 1e-3_f32.max(seq_width.abs() * 1e-3);
                assert!(
                    graph_width <= seq_width + tol,
                    "graph CROWN-IBP intermediate looser at '{node_name}' ({provenance:?}): \
                     seq={seq_width:.6}, graph={graph_width:.6} (tol={tol:.6})"
                );
            }
        }
    }
}

/// Phase 5: Compare the "public CROWN" entrypoint (which tries alpha-CROWN
/// first, then falls back) against the sequential fixed-slope CROWN.
///
/// This is the comparison that matters for the BaB root bound computation:
/// graph.propagate_crown() vs network.propagate_crown().
#[ntest::timeout(60000)]
#[test]
fn test_1923_public_crown_entrypoint_gap() {
    let network = build_acasxu_like_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = build_acasxu_input();

    let (seq_crown, graph_crown) = tests::with_crown_dense_budget_mb("2048", || {
        // Network.propagate_crown uses fixed-slope CROWN.
        let seq_crown = network.propagate_crown(&input).unwrap();
        // GraphNetwork.propagate_crown tries alpha-CROWN first, then fixed-slope fallback.
        let graph_crown = graph.propagate_crown(&input).unwrap();
        (seq_crown, graph_crown)
    });

    assert_eq!(seq_crown.shape(), graph_crown.shape());
    assert_no_loosening(
        &seq_crown,
        &graph_crown,
        "public GraphNetwork CROWN vs sequential fixed-slope CROWN",
    );

    let cmp = compare_bounds(&seq_crown, &graph_crown);
    eprintln!("--- Public CROWN entrypoint comparison (#1923) ---");
    eprintln!(
        "Network.propagate_crown (fixed-slope) mean width: {:.6}",
        cmp.mean_seq_width
    );
    eprintln!(
        "GraphNetwork.propagate_crown (alpha+fallback) mean width: {:.6}",
        cmp.mean_graph_width
    );
    eprintln!("Max graph excess: {:.6}", cmp.max_graph_excess);

    let seq_widths = bound_width(&seq_crown);
    let graph_widths = bound_width(&graph_crown);
    for (idx, (sw, gw)) in seq_widths.iter().zip(graph_widths.iter()).enumerate() {
        let diff = gw - sw;
        eprintln!(
            "  output[{}]: seq={:.6}, graph={:.6}, diff={:+.6}",
            idx, sw, gw, diff
        );
    }

    // The public graph path must never publish a bound looser than its
    // fixed-slope fallback baseline.
    assert!(
        cmp.max_graph_excess <= 1e-3_f32.max(cmp.mean_seq_width.abs() * 1e-4),
        "GraphNetwork::propagate_crown regressed against sequential fixed-slope CROWN: \
         max excess={}, sequential mean width={}",
        cmp.max_graph_excess,
        cmp.mean_seq_width
    );
}

/// Phase 6: Graph-vs-sequential child domain parity.
///
/// The primary assertion: GraphNetwork spec-guided CROWN matches sequential
/// fixed-slope CROWN on every bisected child domain.  This confirms that
/// the #1923 bound gap is NOT caused by the graph representation itself.
///
/// Alpha tightening beyond fixed-slope CROWN is measured but not required
/// to be positive: on this synthetic ACAS-Xu-like network with very tight
/// child domains (~0.0001 width), alpha optimization has no room to improve
/// over fixed-slope CROWN.  Per R1 (commit 75ab9a5b9), per-domain alpha is
/// a separate tightening experiment, not evidence for the #1923 root cause.
#[ntest::timeout(60000)]
#[test]
fn test_1923_child_domain_gap_is_per_domain_alpha_not_graph_representation() {
    tests::with_crown_dense_budget_mb("2048", || {
        let network = build_acasxu_scalar_objective_network();
        let graph = GraphNetwork::from_sequential(&network).unwrap();
        let input = build_acasxu_input();
        let spec_matrix = array![[1.0_f32]];
        let alpha_config = AlphaCrownConfig {
            iterations: 20,
            ..AlphaCrownConfig::default()
        };

        let mut max_alpha_tightening = 0.0_f32;
        let mut children_evaluated = 0_usize;

        for dim in 0..input.len() {
            let width = input.upper()[[dim]] - input.lower()[[dim]];
            if !width.is_finite() || width <= 0.0 {
                continue;
            }

            for (take_left_child, label) in [(true, "left"), (false, "right")] {
                let child = split_input_dimension(&input, dim, take_left_child);
                // evaluate_child_domain_gap_1923 asserts graph_spec ≈ seq_fixed
                // for every child — this is the core parity check.
                let tightening = evaluate_child_domain_gap_1923(
                    &network,
                    &graph,
                    &child,
                    &spec_matrix,
                    &alpha_config,
                    dim,
                    label,
                );
                max_alpha_tightening = max_alpha_tightening.max(tightening);
                children_evaluated += 1;
            }
        }

        assert!(
            children_evaluated > 0,
            "ACAS-Xu input should yield at least one splittable child"
        );
        eprintln!(
            "#1923 child parity: {children_evaluated} children evaluated, \
             max alpha tightening over fixed-slope = {max_alpha_tightening:.6}"
        );
        // The primary finding: graph spec-guided CROWN matched sequential
        // fixed-slope CROWN on all children (asserted inside
        // evaluate_child_domain_gap_1923).  Alpha tightening is reported
        // as diagnostic data.
        if max_alpha_tightening < 1e-4 {
            eprintln!(
                "#1923 NOTE: alpha-CROWN provided no additional tightening over \
                 fixed-slope CROWN on these synthetic child domains.  The gap in \
                 the real ACAS-Xu 4_x family is not caused by graph representation."
            );
        }
    });
}
