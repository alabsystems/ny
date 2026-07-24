// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common::assert_finite_and_ordered;
use super::boundary::{
    assert_boundary_windows_valid, extract_waveform_boundary_bounds, KOKORO_BOUNDARY_SAMPLES,
};
use super::graph_support::{
    first_conv_transpose_node, first_instance_norm_node, instance_norm_node_count,
    vocoder_prefix_subgraph,
};
use super::model::{
    bounded_kokoro_features_input, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use super::*;

// ---------------------------------------------------------------------------
// Packet 2: IBP on prefix subgraph (#3500)
//
// Extracts a prefix subgraph up to the first ConvTranspose1d node and runs
// IBP on it. This verifies that the shallowest upsampling stage produces
// finite, ordered bounds within a reasonable CPU timeout.
// Reference: designs/2026-03-11-issue-3500-shallow-vocoder-subpath.md §Packet 2
// ---------------------------------------------------------------------------

// Budget: ~90-120s for graph conversion const-fold (har=[22,61] through
// HiFi-GAN upsampler) + ~20-45s for prefix IBP.  300s accommodates the
// corrected har contract at features_t=1.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_graph_ibp_kokoro_vocoder_prefix_subgraph_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");

    let cut_node = first_conv_transpose_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);
    eprintln!(
        "prefix subgraph: {} nodes (full: {}), output: {}",
        prefix.num_nodes(),
        graph.num_nodes(),
        cut_node
    );

    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);
    let output = prefix
        .propagate_ibp(&input)
        .expect("prefix IBP should complete");

    assert_finite_and_ordered(&output, "prefix IBP output");

    let flat_len = output.lower().len();
    let max_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .fold(0.0_f32, f32::max);
    let mean_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / flat_len as f32;
    eprintln!(
        "prefix IBP: {} output elements, max_width={:.4e}, mean_width={:.4e}",
        flat_len, max_width, mean_width
    );

    // Boundary extraction: first/last 240 samples (10ms at 24kHz).
    // On the prefix subgraph (15360 elements), this exercises the real
    // extraction machinery that will be used with GPU-accelerated full-graph
    // bounds with GPU-accelerated IBP (#3397).
    let (first_lower, first_upper, last_lower, last_upper) =
        extract_waveform_boundary_bounds(&output, KOKORO_BOUNDARY_SAMPLES);
    let effective_n = KOKORO_BOUNDARY_SAMPLES.min(flat_len);
    assert_eq!(first_lower.len(), effective_n);
    assert_eq!(last_upper.len(), effective_n);
    assert_boundary_windows_valid(
        &first_lower,
        &first_upper,
        &last_lower,
        &last_upper,
        flat_len,
        "prefix IBP",
    );
}

// ---------------------------------------------------------------------------
// Packet 2b: IBP through fused InstanceNorm prefix (#3591)
//
// Extends the prefix past the first ConvTranspose to include the first fused
// InstanceNorm1d node. Validates acceptance criterion 3 of #3591: IBP flows
// through the monolithic InstanceNorm layer that replaces the decomposed
// ReduceMean→Sub→Pow→...→Div pattern.
//
// Graph topology at the cut point:
//   [0] LeakyReLU → [1] ConvTranspose1d → [2] AddConstant
//   → [3] InstanceNorm1d → ...
//
// The ConvTranspose1d at [1] dominates cost (~45s for 15360 output elements).
// Extending to [3] adds trivial element-wise ops.  Graph conversion const-fold
// adds ~90-120s under the corrected har contract (see prefix_subgraph_3500).
// ---------------------------------------------------------------------------

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_graph_ibp_through_fused_instance_norm_prefix_3591() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");

    let instnorm_node = first_instance_norm_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &instnorm_node);
    let prefix_instnorm_count = instance_norm_node_count(&prefix);
    eprintln!(
        "InstanceNorm prefix: {} nodes (full: {}), output: {}, InstanceNorm1d count: {}",
        prefix.num_nodes(),
        graph.num_nodes(),
        instnorm_node,
        prefix_instnorm_count,
    );
    assert!(
        prefix_instnorm_count > 0,
        "prefix subgraph through first InstanceNorm1d should contain at least one fused node"
    );

    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);
    let output = prefix
        .propagate_ibp(&input)
        .expect("IBP through fused InstanceNorm prefix should complete");

    assert_finite_and_ordered(&output, "InstanceNorm prefix IBP");

    let flat_len = output.lower().len();
    let max_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .fold(0.0_f32, f32::max);
    let mean_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / flat_len as f32;
    eprintln!(
        "InstanceNorm prefix IBP: {} output elements, max_width={:.4e}, mean_width={:.4e}",
        flat_len, max_width, mean_width
    );
}

// ---------------------------------------------------------------------------
// Packet 5: IBP on deeper prefix (#3500)
//
// Progressive deepening to find the deepest CPU-viable cut point.
// At T=6, the unfused normalization path inflates the graph: "resblocks.0"
// covers 408 nodes (vs 60 in the T=8 fused topology) because the ONNX
// decomposed ReduceMean/Sub/Mul/Pow/Sqrt/Div nodes remain alongside
// the fused InstanceNorm1d nodes. Cutting by Conv1d layer type is robust
// across both fused and unfused topologies.
// ---------------------------------------------------------------------------

/// Find the first Conv1d node after the first ConvTranspose1d.
///
/// This gives a prefix including: initial activation, ConvTranspose1d
/// upsampling, normalization/AdaIN processing, snake activation, and
/// the first dilated convolution. Includes meaningful ResBlock processing
/// beyond just the upsampling while staying CPU-tractable.
pub(super) fn first_conv1d_after_conv_transpose(graph: &GraphNetwork) -> String {
    let topo = graph.topological_sort().expect("topo sort should succeed");
    let mut past_conv_transpose = false;
    for name in &topo {
        let layer_type = graph
            .node(name)
            .map(|n| n.layer().layer_type().to_string())
            .unwrap_or_default();
        if layer_type == "ConvTranspose1d" || layer_type == "ConvTranspose2d" {
            past_conv_transpose = true;
            continue;
        }
        if past_conv_transpose && layer_type == "Conv1d" {
            return name.clone();
        }
    }
    panic!("vocoder should have Conv1d after ConvTranspose")
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_graph_ibp_kokoro_vocoder_first_conv1d_prefix_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");

    let cut_node = first_conv1d_after_conv_transpose(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);
    eprintln!(
        "first-conv1d prefix: {} nodes (full: {}), output: {}",
        prefix.num_nodes(),
        graph.num_nodes(),
        cut_node
    );

    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);
    let output = prefix
        .propagate_ibp(&input)
        .expect("first-conv1d prefix IBP should complete");

    assert_finite_and_ordered(&output, "first-conv1d prefix IBP output");

    let flat_len = output.lower().len();
    let max_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .fold(0.0_f32, f32::max);
    let mean_width: f32 = output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / flat_len as f32;
    eprintln!(
        "first-conv1d prefix IBP: {} output elements, max_width={:.4e}, mean_width={:.4e}",
        flat_len, max_width, mean_width
    );

    let (first_lower, first_upper, last_lower, last_upper) =
        extract_waveform_boundary_bounds(&output, KOKORO_BOUNDARY_SAMPLES);
    let effective_n = KOKORO_BOUNDARY_SAMPLES.min(flat_len);
    assert_eq!(first_lower.len(), effective_n);
    assert_boundary_windows_valid(
        &first_lower,
        &first_upper,
        &last_lower,
        &last_upper,
        flat_len,
        "first-conv1d prefix IBP",
    );
}

// ---------------------------------------------------------------------------
// Runtime measurements (#3500, Packets 3+4)
//
// Measured 2026-03-11 iter 1392:
//
// Packet 2 (IBP, 1st prefix):
//   - 1st ConvTranspose prefix (2 nodes, /ups.0/ConvTranspose): IBP passes
//     in 45s, 15360 output elements, max_width=1.27e-2, mean_width=8.72e-3
//   - Boundary extraction (iter 1393): 240 samples from 15360 total
//     first: [-8.82e-3, -4.11e-3], last: [-1.39e-2, -9.04e-3]
//     Total time with boundary extraction: 56s
//
// Packet 3 (CROWN, 1st prefix):
//   - CROWN on the same 2-node prefix: timed out at 180s
//   - CROWN backward computes per-output-element Jacobians across 15360
//     elements — too expensive for CPU even at the shallowest prefix
//   - Boundary-only spec-guided CROWN (480 boundary samples instead of all
//     15360 outputs) also timed out at 180s on CPU
//   - Boundary-only spec-guided CROWN with GPU engine threaded through
//     `propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline`
//     also timed out at 180s on the same harness
//   - Historical note: those "GPU" retries were measured before #3598 landed,
//     when the 1D-convolution backward path still ignored `GemmEngine` and
//     remained CPU-bound at `Conv1d` / `ConvTranspose1d`
//
// Packet 4 (progressive deepening):
//   - 2nd ConvTranspose prefix (177 nodes, /ups.1/ConvTranspose): IBP
//     timed out at 120s
//
// Packet 5 (progressive deepening, 2026-03-12 iter 1426):
//   - 176 nodes (through /LeakyRelu_1, before 2nd ConvTranspose):
//     IBP timed out at 300s
//   - 408 nodes (through /resblocks.0/adain2.2/Sub_1, unfused normalization
//     at T=6 inflates node count vs T=8 fused inventory):
//     IBP timed out at 300s
//   - 12 nodes (through /resblocks.0/convs1.0/Conv, first Conv1d after
//     first ConvTranspose): IBP passes in 79s, 15360 output elements,
//     max_width=1.19e2, mean_width=8.20e1
//     Boundary: first: [-2.23e1, 2.27e1], last: [-2.02e1, 2.00e1]
//     Bounds are valid but very wide (10^4x wider than 2-node prefix)
//     because the ResBlock processing amplifies interval widths
//
// Packet 6 (deep prefix CROWN, 2026-03-12 iter 1429):
//   - 12-node prefix spec-guided CROWN with 4 boundary specs (8 rows):
//     IBP node bounds: 138.6s, CROWN backward: 2.3s, total: 140.9s
//     Result: 8/8 tighter, 0 equal — CROWN tightens ALL tested boundary
//     samples on the deep prefix. This is the first evidence that CROWN
//     backward propagation through the ResBlock (Conv1d + InstanceNorm +
//     SnakeActivation) produces strictly tighter bounds than IBP.
//   - Implication: GPU-accelerated full-graph CROWN (#3622) should deliver
//     meaningful tightening across the full vocoder, not just match IBP.
//
// Conclusion: CPU-viable IBP cut points are 2 nodes (45s) and 12 nodes (79-139s).
// The 2-node prefix gives tight bounds (CROWN == IBP, no tightening room).
// The 12-node prefix gives wide bounds but CROWN tightens ALL 8 tested specs.
// Prefix CROWN is green on HEAD after #3598 (Conv1d engine threading).
// Full-graph waveform output requires GPU acceleration via #3622 (remaining
// GemmEngine threading). #3619 and #3597 are now closed.
// ---------------------------------------------------------------------------
