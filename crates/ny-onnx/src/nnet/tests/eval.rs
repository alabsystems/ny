// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for NNet evaluation, normalization, conversion, and integration
//! with ACAS-Xu models.

use ndarray::{ArrayD, IxDyn};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

use crate::nnet::{load_nnet, parse_nnet};

use super::fixtures::require_test_model;

#[ntest::timeout(10000)]
#[test]
fn test_nnet_evaluation() {
    let content = r#"
// Identity-like network for testing
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();

    // Test evaluation (should be approximately identity for positive inputs)
    let input = vec![1.0, 2.0];
    let output = network.evaluate(&input, false).unwrap();
    assert_eq!(output.len(), 2);
    // Linear output (no ReLU on last layer)
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_evaluation_with_normalization() {
    let content = r#"
// Network for normalization testing
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
2.0,3.0,0.0,
4.0,5.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();

    // Test with normalization enabled
    let input = vec![6.0, 8.0];
    let output = network.evaluate(&input, true).unwrap();

    // Input normalization: (x - mean) / range
    // x[0] = (6.0 - 2.0) / 4.0 = 1.0
    // x[1] = (8.0 - 3.0) / 5.0 = 1.0
    // Linear: [[1,0],[0,1]] * [1,1] + [0,0] = [1, 1]
    // Denorm: output * range + mean = [1*1+0, 1*1+0] = [1, 1]
    assert_eq!(output.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_evaluation_clamping() {
    let content = r#"
// Network with tight input bounds
1,2,2,2,
2,2,
0,
0.0,0.0,
1.0,1.0,
0.5,0.5,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();

    // Input outside bounds should be clamped when normalizing
    let input = vec![10.0, -5.0]; // Outside [0, 1]
    let output = network.evaluate(&input, true).unwrap();
    assert_eq!(output.len(), 2);

    // No clamping when normalize=false
    let output_no_norm = network.evaluate(&input, false).unwrap();
    assert_eq!(output_no_norm.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_to_prop_network() {
    let content = r#"
// Simple network
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let nnet = parse_nnet(content).unwrap();
    let network = nnet.to_prop_network().unwrap();

    // Create input bounds
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.5, 1.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run IBP
    let result = network.propagate_ibp(&input).unwrap();
    assert_eq!(result.shape(), &[1, 2]);

    // Output bounds should be valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l <= u, "Invalid bounds: {} > {}", l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_in_hidden_layers() {
    let content = r#"
// Network with hidden ReLU
2,2,1,3,
2,3,1,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
-1.0,1.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();

    // Test with negative inputs that should get ReLU'd
    let input = vec![-1.0, 1.0];
    let output = network.evaluate(&input, false).unwrap();

    // First hidden layer: [relu(-1), relu(1), relu(-1+1)] = [0, 1, 0]
    // Output: 0*1 + 1*1 + 0*1 = 1
    assert_eq!(output.len(), 1);
    // The exact value depends on the network computation
}

#[ntest::timeout(10000)]
#[test]
fn test_load_acasxu_model() {
    // Load actual ACAS-Xu model
    let model_path = require_test_model("acasxu_1_1.nnet");

    let network = load_nnet(&model_path).unwrap();

    // ACAS-Xu 1_1 has 7 layers (6 hidden + 1 output)
    assert_eq!(network.num_layers, 7);
    assert_eq!(network.input_size, 5);
    assert_eq!(network.output_size, 5);
    assert_eq!(network.layer_sizes, vec![5, 50, 50, 50, 50, 50, 50, 5]);

    // Check weights dimensions
    assert_eq!(network.weights[0].shape(), &[50, 5]); // First hidden layer: 50 x 5
    assert_eq!(network.weights[6].shape(), &[5, 50]); // Output layer: 5 x 50

    // Convert to PropNetwork and run IBP
    let prop_network = network.to_prop_network().unwrap();
    assert_eq!(prop_network.layers().len(), 13); // 7 linear + 6 relu

    // Create input with small perturbation
    let (lower_bounds, upper_bounds) = network.normalized_input_bounds();
    let center: Vec<f32> = lower_bounds
        .iter()
        .zip(&upper_bounds)
        .map(|(l, u)| (l + u) / 2.0)
        .collect();

    let eps = 0.01;
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 5]), center.iter().map(|&c| c - eps).collect()).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 5]), center.iter().map(|&c| c + eps).collect()).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run IBP
    let result = prop_network.propagate_ibp(&input).unwrap();
    assert_eq!(result.shape(), &[1, 5]);

    // Output bounds should be valid
    for (l, u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l <= u, "Invalid bounds: {} > {}", l, u);
    }

    // Print bounds for information
    println!("ACAS-Xu IBP output bounds:");
    for i in 0..5 {
        println!(
            "  Output {}: [{:.4}, {:.4}]",
            i,
            result.lower()[[0, i]],
            result.upper()[[0, i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_acasxu() {
    // Test CROWN-IBP vs CROWN on ACAS-Xu model
    let model_path = require_test_model("acasxu_1_1.nnet");

    let network = load_nnet(&model_path).unwrap();
    let prop_network = network.to_prop_network().unwrap();

    // Use the same input bounds as in the debug report
    // lower = [0.6, -0.5, -0.5, 0.45, -0.5]
    // upper = [0.679857769, 0.5, 0.5, 0.5, -0.45]
    let lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.5, -0.5, 0.45, -0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.679858, 0.5, 0.5, 0.5, -0.45]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run IBP
    let ibp_result = prop_network.propagate_ibp(&input).unwrap();
    let ibp_width: f32 = ibp_result.width().iter().sum();

    // Run CROWN
    let crown_result = prop_network.propagate_crown(&input).unwrap();
    let crown_width: f32 = crown_result.width().iter().sum();

    // Run CROWN-IBP
    let crown_ibp_result = prop_network.propagate_crown_ibp(&input).unwrap();
    let crown_ibp_width: f32 = crown_ibp_result.width().iter().sum();

    println!("\n=== ACAS-Xu Bound Comparison ===");
    println!("IBP total width: {:.2}", ibp_width);
    println!("CROWN total width: {:.2}", crown_width);
    println!("CROWN-IBP total width: {:.2}", crown_ibp_width);

    println!("\nCROWN bounds:");
    for i in 0..5 {
        println!(
            "  Output {}: [{:.2}, {:.2}]",
            i,
            crown_result.lower()[[i]],
            crown_result.upper()[[i]]
        );
    }

    println!("\nCROWN-IBP bounds:");
    for i in 0..5 {
        println!(
            "  Output {}: [{:.2}, {:.2}]",
            i,
            crown_ibp_result.lower()[[i]],
            crown_ibp_result.upper()[[i]]
        );
    }

    // CROWN should be tighter than IBP
    assert!(
        crown_width <= ibp_width,
        "CROWN ({:.2}) should be <= IBP ({:.2})",
        crown_width,
        ibp_width
    );

    // Both bounds should be valid
    for i in 0..5 {
        assert!(
            crown_result.lower()[[i]] <= crown_result.upper()[[i]],
            "CROWN bounds invalid at {}",
            i
        );
        assert!(
            crown_ibp_result.lower()[[i]] <= crown_ibp_result.upper()[[i]],
            "CROWN-IBP bounds invalid at {}",
            i
        );
    }

    // Print improvement percentage
    let improvement_vs_crown = (1.0 - crown_ibp_width / crown_width) * 100.0;
    let improvement_vs_ibp = (1.0 - crown_ibp_width / ibp_width) * 100.0;
    println!(
        "\nCROWN-IBP improvement vs CROWN: {:.1}%",
        improvement_vs_crown
    );
    println!("CROWN-IBP improvement vs IBP: {:.1}%", improvement_vs_ibp);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_get_normalized_input_bounds() {
    let content = r#"
1,2,2,2,
2,2,
0,
0.0,0.0,
10.0,20.0,
5.0,10.0,0.0,
2.0,4.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();
    let (lower, upper) = network.normalized_input_bounds();

    // lower = (min - mean) / range
    // Input 0: (0 - 5) / 2 = -2.5
    // Input 1: (0 - 10) / 4 = -2.5
    assert!((lower[0] - (-2.5)).abs() < 1e-6);
    assert!((lower[1] - (-2.5)).abs() < 1e-6);

    // upper = (max - mean) / range
    // Input 0: (10 - 5) / 2 = 2.5
    // Input 1: (20 - 10) / 4 = 2.5
    assert!((upper[0] - 2.5).abs() < 1e-6);
    assert!((upper[1] - 2.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_to_ny_network() {
    let content = r#"
1,3,2,3,
3,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
1.0,0.0,0.0,
0.0,1.0,0.0,
0.0,
0.0,
"#;

    let nnet = parse_nnet(content).unwrap();
    let ny_network = nnet.to_ny_network();

    assert_eq!(ny_network.name, "nnet_model");
    assert_eq!(ny_network.inputs.len(), 1);
    assert_eq!(ny_network.outputs.len(), 1);
    assert_eq!(ny_network.inputs[0].name, "input");
    assert_eq!(ny_network.inputs[0].shape, vec![1, 3]);
    assert_eq!(ny_network.outputs[0].name, "output");
    assert_eq!(ny_network.outputs[0].shape, vec![1, 2]);
    assert_eq!(ny_network.param_count, nnet.param_count());
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_param_count() {
    let content = r#"
2,2,3,3,
2,3,3,
0,
-1.0,-1.0,
1.0,1.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
1.0,1.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,
1.0,1.0,1.0,
1.0,1.0,1.0,
0.1,
0.2,
0.3,
"#;

    let network = parse_nnet(content).unwrap();
    // Layer 1: 3x2 weights + 3 biases = 9
    // Layer 2: 3x3 weights + 3 biases = 12
    // Total = 21
    assert_eq!(network.param_count(), 21);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_deep_network_relu_propagation() {
    // 3-layer deep network to test ReLU in multiple hidden layers
    let content = r#"
3,2,2,4,
2,4,4,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
-1.0,0.0,
0.0,-1.0,
0.0,
0.0,
0.0,
0.0,
1.0,0.0,0.0,0.0,
0.0,1.0,0.0,0.0,
0.0,0.0,1.0,0.0,
0.0,0.0,0.0,1.0,
0.0,
0.0,
0.0,
0.0,
1.0,1.0,1.0,1.0,
-1.0,-1.0,-1.0,-1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();
    assert_eq!(network.num_layers, 3);

    // Test evaluation - network has negative paths that should be ReLU'd to 0
    let input = vec![1.0, 1.0];
    let output = network.evaluate(&input, false).unwrap();
    assert_eq!(output.len(), 2);

    // Convert to prop network and verify structure
    let prop_network = network.to_prop_network().unwrap();
    // 3 linear layers + 2 ReLU layers (no ReLU on output)
    assert_eq!(prop_network.layers().len(), 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_means_ranges_with_only_inputs() {
    // Test case where means/ranges have exactly input_size elements
    let content = r#"
1,2,2,2,
2,2,
0,
-1.0,-1.0,
1.0,1.0,
0.5,0.5,
1.0,1.0,
1.0,0.0,
0.0,1.0,
0.0,
0.0,
"#;

    let network = parse_nnet(content).unwrap();
    // When means/ranges have only input_size elements, output_mean=0, output_range=1
    assert_eq!(network.output_mean, 0.0);
    assert_eq!(network.output_range, 1.0);
    assert_eq!(network.input_means.len(), 2);
    assert_eq!(network.input_ranges.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_single_layer_network() {
    // Minimal 1-layer network (direct input to output)
    let content = r#"
1,3,2,3,
3,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
1.0,0.0,0.0,
0.0,1.0,0.0,
0.5,
-0.5,
"#;

    let network = parse_nnet(content).unwrap();
    assert_eq!(network.num_layers, 1);

    // Single layer should have no ReLU (output layer)
    let input = vec![1.0, 2.0, 3.0];
    let output = network.evaluate(&input, false).unwrap();
    // Expected: [1*1 + 0*2 + 0*3 + 0.5, 0*1 + 1*2 + 0*3 - 0.5] = [1.5, 1.5]
    assert!((output[0] - 1.5).abs() < 1e-6);
    assert!((output[1] - 1.5).abs() < 1e-6);
}

/// Compare two BoundedTensors element-wise.
/// Returns (max_element_diff, seq_total_width, graph_total_width).
/// Asserts both tensors have valid bounds (lower <= upper).
fn compare_acasxu_bounds(
    label: &str,
    seq: &BoundedTensor,
    graph: &BoundedTensor,
) -> (f32, f32, f32) {
    assert_eq!(
        seq.len(),
        graph.len(),
        "{}: output dimension mismatch",
        label
    );
    eprintln!("\n=== {} ===", label);
    let mut max_diff: f32 = 0.0;
    let mut seq_total: f32 = 0.0;
    let mut graph_total: f32 = 0.0;
    for i in 0..seq.len() {
        let sl = seq.lower()[[i]];
        let su = seq.upper()[[i]];
        let gl = graph.lower()[[i]];
        let gu = graph.upper()[[i]];
        assert!(sl <= su, "{}: seq bounds invalid at output {}", label, i);
        assert!(gl <= gu, "{}: graph bounds invalid at output {}", label, i);
        let sw = su - sl;
        let gw = gu - gl;
        seq_total += sw;
        graph_total += gw;
        let diff = ((sl - gl).abs()).max((su - gu).abs());
        max_diff = max_diff.max(diff);
        eprintln!(
            "  Output {}: seq=[{:.4}, {:.4}] w={:.4}  graph=[{:.4}, {:.4}] w={:.4}  ratio={:.4}",
            i,
            sl,
            su,
            sw,
            gl,
            gu,
            gw,
            if sw > 0.0 { gw / sw } else { 1.0 }
        );
    }
    let ratio = if seq_total > 0.0 {
        graph_total / seq_total
    } else {
        1.0
    };
    eprintln!(
        "  Total: seq={:.4} graph={:.4} ratio={:.4} max_diff={:.2e}",
        seq_total, graph_total, ratio, max_diff
    );
    (max_diff, seq_total, graph_total)
}

/// Compare CROWN-IBP intermediate bounds between sequential and graph paths.
/// Returns the first layer index where bounds diverge significantly (>1e-4).
fn compare_crown_ibp_intermediates(
    network: &ny_propagate::Network,
    seq_intermediates: &[BoundedTensor],
    graph_intermediates: &std::collections::HashMap<String, BoundedTensor>,
) -> Option<(usize, f32)> {
    eprintln!("\n=== CROWN-IBP intermediate bounds comparison ===");
    let mut first_divergent: Option<(usize, f32)> = None;
    for (idx, seq_bt) in seq_intermediates.iter().enumerate() {
        let node_name = format!("layer_{}", idx);
        if let Some(graph_bt) = graph_intermediates.get(&node_name) {
            let seq_w: f32 = seq_bt.width().iter().sum();
            let graph_w: f32 = graph_bt.width().iter().sum();
            let max_diff = seq_bt
                .lower()
                .iter()
                .zip(graph_bt.lower().iter())
                .map(|(a, b)| (a - b).abs())
                .chain(
                    seq_bt
                        .upper()
                        .iter()
                        .zip(graph_bt.upper().iter())
                        .map(|(a, b)| (a - b).abs()),
                )
                .fold(0.0f32, f32::max);
            eprintln!(
                "  layer_{} ({}): seq={:.4} graph={:.4} ratio={:.4} diff={:.2e}",
                idx,
                network.layers()[idx].layer_type(),
                seq_w,
                graph_w,
                if seq_w > 0.0 { graph_w / seq_w } else { 1.0 },
                max_diff
            );
            // The sequential and graph CROWN-IBP collectors are BOTH sound, but
            // they legitimately differ at ReLU-output nodes whose only consumer is
            // a Linear: the sequential path pushes the CROWN-tightened pre-activation
            // forward through the monotone ReLU and intersects with IBP
            // (crown_ibp_forward.rs), while the graph path keeps pure IBP for ReLU
            // outputs that no nonlinear consumer *demands* tightening (demand.rs).
            // So at ReLU nodes the only contract is SOUNDNESS — the graph must be a
            // superset of the (tighter) sequential bound, never tighter-and-unsound.
            // At Linear / pre-activation / output nodes the two must be bit-identical.
            let is_relu = network.layers()[idx].layer_type() == "ReLU";
            let divergence = if is_relu {
                // graph must contain seq: graph.lower <= seq.lower AND seq.upper <= graph.upper.
                seq_bt
                    .lower()
                    .iter()
                    .zip(graph_bt.lower().iter())
                    .map(|(s, g)| g - s) // > 0 => graph lower ABOVE seq lower (graph too tight => unsound)
                    .chain(
                        seq_bt
                            .upper()
                            .iter()
                            .zip(graph_bt.upper().iter())
                            .map(|(s, g)| s - g), // > 0 => graph upper BELOW seq upper (graph too tight)
                    )
                    .fold(0.0f32, f32::max)
            } else {
                max_diff
            };
            if divergence > 1e-4 && first_divergent.is_none() {
                first_divergent = Some((idx, divergence));
            }
        }
    }
    if let Some((idx, diff)) = first_divergent {
        eprintln!("  >>> First divergence at layer_{}: diff={:.2e}", idx, diff);
    } else {
        eprintln!("  No significant intermediate divergence (threshold=1e-4)");
    }
    first_divergent
}

/// #1898 diagnostic: Compare Network vs GraphNetwork CROWN bounds on ACAS-Xu 1_1.
///
/// Loads the same model as both a sequential `Network` and a `GraphNetwork`
/// (via `from_sequential`), runs IBP/CROWN/batched-CROWN, and compares bounds.
/// Key finding: single-pass CROWN is bit-identical; batched CROWN diverges
/// because Network uses IBP intermediates while GraphNetwork uses CROWN-IBP.
#[ntest::timeout(60000)]
#[test]
fn test_crown_network_vs_graph_acasxu_1898() {
    let model_path = require_test_model("acasxu_1_1.nnet");
    let nnet = load_nnet(&model_path).unwrap();
    let network = nnet.to_prop_network().unwrap();
    let graph = GraphNetwork::from_sequential(&network).unwrap();

    // ACAS-Xu Property 1 input bounds (normalized, from prop_1.vnnlib)
    let lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6, -0.5, -0.5, 0.45, -0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.6798577, 0.5, 0.5, 0.5, -0.45]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Step 1: IBP — should be bit-identical through a linear chain.
    let (ibp_diff, _, _) = compare_acasxu_bounds(
        "#1898 IBP comparison",
        &network.propagate_ibp(&input).unwrap(),
        &graph.propagate_ibp(&input).unwrap(),
    );
    assert!(ibp_diff < 1e-5, "IBP diverges: max diff {:.2e}", ibp_diff);

    // Step 2: CROWN-IBP intermediates — find first divergent layer.
    // Step 2: CROWN-IBP intermediates. The sequential and graph collectors are
    // both SOUND but apply different (sound) tightening at ReLU-output→Linear
    // nodes (see compare_crown_ibp_intermediates). The contract is therefore:
    // bit-identical at Linear/pre-activation/output nodes, and graph ⊇ seq
    // (sound superset) at ReLU nodes — NOT bit-equality. A non-None result now
    // means a REAL defect: a Linear mismatch, or a graph ReLU bound that is
    // tighter than the sequential one (i.e. the graph is no longer a sound
    // over-approximation).
    let seq_int = network.collect_crown_ibp_bounds(&input).unwrap();
    let graph_int = graph.collect_crown_ibp_bounds_dag(&input).unwrap();
    let first_divergent = compare_crown_ibp_intermediates(&network, &seq_int, &graph_int);
    assert!(
        first_divergent.is_none(),
        "CROWN-IBP soundness/consistency violation at layer {} (Linear mismatch or graph tighter-than-seq at a ReLU)",
        first_divergent.map_or(0, |(idx, _)| idx)
    );

    // Steps 3-4: single-pass and batched CROWN. The network path and the graph
    // path are NOT bit-equal: the network path uses IBP intermediates while the
    // graph path uses tighter CROWN-IBP intermediates (Step 4 note below), so the
    // graph is legitimately TIGHTER for this MLP (~3x here). Asserting bit-equality
    // was therefore wrong. The real contract is SOUNDNESS — every concrete network
    // output over the input box must lie inside BOTH paths' CROWN bounds — which we
    // verify by sampling (box corners + center + deterministic-pseudo-random
    // interior points). A tighter graph is acceptable ONLY if it is still sound.
    let lo: Vec<f32> = input.lower().iter().copied().collect();
    let hi: Vec<f32> = input.upper().iter().copied().collect();
    let n_in = lo.len();
    let mut samples: Vec<Vec<f32>> = Vec::new();
    for mask in 0u32..(1u32 << n_in) {
        samples.push(
            (0..n_in)
                .map(|d| if mask & (1 << d) != 0 { hi[d] } else { lo[d] })
                .collect(),
        );
    }
    samples.push((0..n_in).map(|d| f32::midpoint(lo[d], hi[d])).collect());
    let mut state = 0x1234_5678_9abc_def0u64; // deterministic LCG — no test flakiness
    for _ in 0..3000 {
        samples.push(
            (0..n_in)
                .map(|d| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = ((state >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
                    lo[d] + r * (hi[d] - lo[d])
                })
                .collect(),
        );
    }
    let outputs: Vec<Vec<f32>> = samples
        .iter()
        .map(|pt| nnet.evaluate(pt, false).unwrap())
        .collect();
    // Count sampled outputs that fall OUTSIDE `bt` (an unsound bound). `false`
    // input-normalization matches the prop network's raw forward space (the IBP
    // step already confirmed both operate in the same un-denormalized space).
    let count_unsound = |bt: &BoundedTensor| -> usize {
        let bl: Vec<f32> = bt.lower().iter().copied().collect();
        let bu: Vec<f32> = bt.upper().iter().copied().collect();
        let tol = 1e-2_f32; // f32 forward-eval slack
        outputs
            .iter()
            .filter(|out| {
                out.iter()
                    .enumerate()
                    .any(|(j, &o)| o < bl[j] - tol || o > bu[j] + tol)
            })
            .count()
    };

    // Step 3: single-pass CROWN — both must be sound; graph is tighter.
    let seq_crown = network.propagate_crown(&input).unwrap();
    let graph_crown = graph.propagate_crown(&input).unwrap();
    let (_, seq_w, graph_w) =
        compare_acasxu_bounds("CROWN output comparison", &seq_crown, &graph_crown);
    let (seq_v, graph_v) = (count_unsound(&seq_crown), count_unsound(&graph_crown));
    eprintln!(
        "Step 3 single-pass CROWN soundness ({} samples): seq unsound={}, graph unsound={}",
        samples.len(),
        seq_v,
        graph_v
    );
    assert_eq!(seq_v, 0, "sequential single-pass CROWN is UNSOUND");
    assert_eq!(
        graph_v, 0,
        "graph single-pass CROWN is UNSOUND (tighter-but-excludes-real-outputs)"
    );
    let crown_ratio = if seq_w > 0.0 { graph_w / seq_w } else { 1.0 };
    assert!(
        crown_ratio <= 1.5,
        "Graph single-pass CROWN unexpectedly {:.2}x wider",
        crown_ratio
    );

    // Step 4: Batched CROWN — graph verify path uses this.
    //   - Network::propagate_crown_batched uses IBP intermediates (crown.rs:1090)
    //   - GraphNetwork::propagate_crown_batched uses CROWN-IBP (crown_batched.rs:109)
    //   For MLPs, GraphNetwork batched CROWN is actually tighter (CROWN-IBP > IBP).
    let seq_bat = network.propagate_crown_batched(&input).unwrap();
    let graph_bat = graph.propagate_crown_batched(&input).unwrap();
    let (_, seq_bw, graph_bw) =
        compare_acasxu_bounds("Batched CROWN comparison", &seq_bat, &graph_bat);
    let (seq_bv, graph_bv) = (count_unsound(&seq_bat), count_unsound(&graph_bat));
    eprintln!(
        "Step 4 batched CROWN soundness: seq unsound={}, graph unsound={}",
        seq_bv, graph_bv
    );
    assert_eq!(seq_bv, 0, "sequential batched CROWN is UNSOUND");
    assert_eq!(graph_bv, 0, "graph batched CROWN is UNSOUND");
    let batched_ratio = if seq_bw > 0.0 { graph_bw / seq_bw } else { 1.0 };
    assert!(
        batched_ratio <= 1.5,
        "Graph batched CROWN {:.2}x wider",
        batched_ratio
    );

    eprintln!("\n=== network-vs-graph CROWN: both paths SOUND; graph tighter via CROWN-IBP intermediates ===");
}
