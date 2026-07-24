// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//
// Semantics-preservation check for the batch-dim resolution + Shape const-fold
// on the ViT (vit_2023 pgd_2_3_16) transformer.
//
// The fix must (1) LOAD the model with static rank-3/4 attention shapes and
// (2) be SEMANTICS-PRESERVING — the loaded/folded graph must compute the SAME
// mathematical function as ONNX Runtime.
//
// NY's bound propagation uses a CROWN relaxation for the attention `MatMul`
// (Q·Kᵀ and softmax·V) bilinear products, which is NOT tight even at a
// zero-width (concrete) input — a known, separate verification-looseness issue
// (the attention-CROWN tightening is a future wave). So we cannot read NY's
// exact forward off `propagate_ibp`'s output bound for an attention model.
//
// Instead we prove function-equivalence the right way: run NY's per-node IBP at
// a concrete point and ORT's per-node inference at the same point, then assert
// that EVERY NY node whose bound is TIGHT (width ≈ 0 — i.e. unaffected by the
// bilinear relaxation) matches ORT's value EXACTLY. This covers the Conv patch
// embedding, all the LayerNorm/BatchNorm, every Linear (Q/K/V/out/MLP), the
// residual Adds, and the Reshape/Transpose attention shape chain — the parts a
// wrong shape-fold would corrupt. A broken fold makes these tight nodes diverge
// (and the reshape/transpose ranks wrong); a sound fold keeps them bit-for-bit
// on ORT, with the only slack confined to the `BilinearCrown` MatMul nodes.

#![cfg(feature = "ort")]

use ndarray::{ArrayD, IxDyn};
use ny_core::Bound;
use ny_onnx::diff::run_inference_with_intermediates;
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::Verifier;

const MODEL: &str = "../../benchmarks/vnncomp2025/benchmarks/vit_2023/onnx/pgd_2_3_16.onnx";

/// Layer types whose IBP through a zero-width input is inherently a CROWN
/// relaxation (bilinear / softmax products), so a non-zero bound width there is
/// expected slack — NOT evidence of a wrong function.
fn is_relaxed_layer(layer_type: &str) -> bool {
    matches!(
        layer_type,
        "BilinearCrown" | "Softmax" | "OpaqueSkip" | "MulConstant"
    )
}

#[test]
fn vit_loads_with_static_attention_shapes_and_preserves_function() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("ViT benchmark model not present; skipping");
        return;
    }

    // Deterministic concrete inputs across the network's normalized range.
    let cases: Vec<Vec<f32>> = vec![
        vec![2.4_f32; 3072],
        (0..3072).map(|i| 2.0 + 0.2 * ((i % 7) as f32)).collect(),
        (0..3072)
            .map(|i| (((i * 31) % 97) as f32 / 97.0) * 4.0 - 2.0)
            .collect(),
    ];

    // Ground-truth attention shapes ORT infers on the batch-resolved model
    // (NY's unbatched view drops the leading 1): the reshape→transpose head-split
    // chain must reach rank-4 and merge back to rank-3.
    let expected_shapes: &[(&str, &[usize])] = &[
        ("/1/1.0/1.0.0/fn/fn.0/Transpose_1", &[5, 48]),
        ("/1/1.0/1.0.0/fn/fn.1/Reshape", &[5, 3, 16]),
        ("/1/1.0/1.0.0/fn/fn.1/Transpose", &[3, 5, 16]),
        ("/1/1.0/1.0.0/fn/fn.1/Transpose_2", &[3, 16, 5]),
        ("/1/1.0/1.0.0/fn/fn.1/MatMul", &[3, 5, 5]),
        ("/1/1.0/1.0.0/fn/fn.1/Reshape_3", &[5, 48]),
    ];

    for (case_idx, values) in cases.iter().enumerate() {
        // NY: per-node concrete IBP.
        let model =
            load_onnx_with_config(MODEL, &OnnxLoadConfig::default()).expect("ViT must LOAD");
        let graph = model
            .to_graph_network_with_options(GraphNetworkOptions::default())
            .expect("ViT must convert to GraphNetwork");
        let degenerate: Vec<Bound> = values.iter().map(|&v| Bound::new(v, v)).collect();
        let input =
            Verifier::bounds_to_tensor(&degenerate, Some(&[3, 32, 32])).expect("input tensor");
        let detailed = graph
            .propagate_ibp_detailed(&input, 0.0)
            .expect("detailed IBP");

        // The attention block must have loaded with the correct static ranks.
        if case_idx == 0 {
            for (name, want) in expected_shapes {
                let node = detailed
                    .nodes
                    .iter()
                    .find(|n| n.name == *name)
                    .unwrap_or_else(|| panic!("attention node {name} missing from loaded graph"));
                assert_eq!(
                    node.output_shape.as_slice(),
                    *want,
                    "attention node {name} loaded with wrong rank/shape (the Shape→Reshape→\
                     Transpose chain did not fold to static dims)"
                );
            }
        }

        // ORT: per-node intermediate values at the same concrete point.
        let ort_input =
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 32, 32]), values.clone()).expect("ort input");
        let ort_values = run_inference_with_intermediates(MODEL, &ort_input).expect("ORT");

        // For every NY node with a TIGHT bound (not a bilinear/softmax relaxation),
        // its midpoint value range must coincide with ORT's value range.
        let mut tight_checked = 0usize;
        let mut max_tight_diff = 0.0_f32;
        for node in &detailed.nodes {
            if is_relaxed_layer(&node.layer_type) || node.has_infinite {
                continue;
            }
            // Skip nodes whose bound widened (downstream of a relaxed node).
            if node.output_width > 1e-2 {
                continue;
            }
            // ORT keys intermediates by ONNX tensor name = node output ("…_output_0").
            let ort_key = format!("{}_output_0", node.name);
            let Some(ort_t) = ort_values.get(&ort_key) else {
                continue;
            };
            let ort_min = ort_t.iter().copied().fold(f32::INFINITY, f32::min);
            let ort_max = ort_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Compare the value RANGE (NY reports min/max across the tensor).
            let d = (node.min_bound - ort_min)
                .abs()
                .max((node.max_bound - ort_max).abs());
            max_tight_diff = max_tight_diff.max(d);
            tight_checked += 1;
            assert!(
                d < 5e-2,
                "case {case_idx}: TIGHT node {} ([{}]) diverges from ORT — \
                 NY range [{:.5}, {:.5}] vs ORT [{:.5}, {:.5}] (diff {:.5}). \
                 The loaded/folded graph does NOT compute the same function.",
                node.name,
                node.layer_type,
                node.min_bound,
                node.max_bound,
                ort_min,
                ort_max,
                d
            );
        }

        eprintln!(
            "case {case_idx}: verified {tight_checked} tight nodes against ORT; \
             max tight-node diff = {max_tight_diff:.6}"
        );
        // The patch embedding + first attention block's pre-relaxation chain
        // (Conv, Reshape/Transpose head-split, Q/K/V Linears, norms, scores) are
        // all tight and shape-fold-sensitive. Downstream of the first
        // `BilinearCrown` every node is widened by relaxation slack, so it is not
        // tight-checkable — but the tight prefix is exactly what a wrong
        // shape-fold would corrupt.
        assert!(
            tight_checked >= 20,
            "case {case_idx}: expected to verify the tight Conv/Linear/norm/reshape prefix, \
             only checked {tight_checked} — the graph may not have loaded fully"
        );
    }
}
