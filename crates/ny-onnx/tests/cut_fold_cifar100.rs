// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN C2 (CPU-first gate) root-margin experiment
//! (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C2, gate (c) falsification lane).
//!
//! Loads the REAL cifar100 `CIFAR100_resnet_medium` + prop885 ε=0.0039 box,
//! generates k=3 L1 cuts from conv1 (C1.5, `generate_l1_cuts`), then measures
//! the CPU graph-CROWN root lower margin of the 99 classification objectives
//! (`y_label − y_j`) WITHOUT cuts and WITH the cuts folded into the first
//! ReLU's lower-side post-activation coefficients at a fixed λ grid
//! (dark `NY_CUT_FOLD` registry, fold site: `dispatch_relu_backward`).
//!
//! Gates:
//! (a) λ=0 / empty registry ⇒ bounds EXACTLY equal to baseline;
//! (b) the λ-margin table (printed) — SUCCESS iff some λ strictly improves
//!     the min margin; a no-improvement result is a valid measurement too.
//!
//! Run (slow, real benchmark assets — release strongly recommended):
//! `cargo test -p ny-onnx --release --test cut_fold_cifar100 -- --ignored --nocapture`

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_propagate::beta_crown::{
    clear_cut_fold, cut_fold_applied_count, generate_l1_cuts, reset_cut_fold_applied_count,
    set_cut_fold, CutFoldEntry,
};
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;
use std::collections::HashMap;

/// prop885 is a targeted-robustness property: label 44, unsafe iff any
/// `Y_j >= Y_44` (j != 44). See the vnnlib header comment.
const LABEL: usize = 44;
const NUM_CLASSES: usize = 100;

#[test]
#[ignore = "multi-minute experiment on real cifar100 benchmark assets; run with --ignored"]
fn c2_cut_fold_root_margin_experiment() {
    // Surface the fold-site magnitude diagnostic (cut_fold `debug!`).
    let _ = tracing_subscriber::fmt()
        .with_env_filter("ny_propagate::beta_crown::bab_cuts::cut_fold=debug")
        .with_writer(std::io::stderr)
        .try_init();
    let dir = workspace_root().join("benchmarks/vnncomp2025/benchmarks/cifar100_2024");
    let onnx_path = dir.join("onnx/CIFAR100_resnet_medium.onnx");
    let vnnlib_path =
        dir.join("vnnlib/CIFAR100_resnet_medium_prop_idx_885_sidx_7654_eps_0.0039.vnnlib");
    if !onnx_path.exists() || !vnnlib_path.exists() {
        eprintln!("SKIP: cifar100 benchmark assets not present under {dir:?}");
        return;
    }

    // Force the proven CPU graph backward loop: disable the root-candidate
    // fast paths (forward-linear C-margin / GPU resnet / alpha rebuild) that
    // would otherwise answer the root query without ever running the CPU
    // per-node backward where the C2 fold site lives.
    // (Serialized + restored via the blessed env choke point — clippy env
    // wall; guards restore the pre-test environment when the test exits.)
    let _env_lock = ny_test_utils::env::lock_env();
    let _g_gpu = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_GPU", "0");
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");
    let _g_fold_off = ny_test_utils::env::ScopedEnvVar::unset("NY_CUT_FOLD");
    clear_cut_fold();

    // --- Load model as a GraphNetwork (matrix mode: cuts require the Dense
    // ReLU backward — same policy as the reference's conv_mode='matrix' when
    // cuts are enabled, abcrown.py:228-231). ---
    let onnx_model = load_onnx(&onnx_path).expect("load CIFAR100_resnet_medium.onnx");
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model
        .to_graph_network_with_options(graph_options)
        .expect("to_graph_network");
    graph.set_use_patches_mode(false);

    // --- Input box from the vnnlib (3072 = 3x32x32, CHW). ---
    let vnnlib = load_vnnlib(&vnnlib_path).expect("load prop885 vnnlib");
    assert_eq!(vnnlib.num_inputs, 3 * 32 * 32);
    assert_eq!(vnnlib.num_outputs, NUM_CLASSES);
    let (xl, xu) = vnnlib.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xl.clone()).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xu.clone()).expect("upper shape"),
    )
    .expect("input BoundedTensor");

    // --- Identify the FIRST ReLU node and its producer conv. ---
    let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
    let relu_name = exec_order
        .iter()
        .find(|name| {
            graph
                .node(name)
                .is_some_and(|node| matches!(node.layer(), Layer::ReLU(_)))
        })
        .expect("network must contain a ReLU")
        .clone();
    let relu_node = graph.node(&relu_name).expect("relu node");
    let conv_name = relu_node.inputs()[0].clone();
    let conv_node = graph.node(&conv_name).unwrap_or_else(|| {
        panic!("first ReLU '{relu_name}' input '{conv_name}' not found in graph")
    });
    let Layer::Conv2d(conv) = conv_node.layer() else {
        panic!(
            "first ReLU '{}' producer '{}' is {}, not Conv2d — L1 cut pre-activations \
             would not be the conv affine form",
            relu_name,
            conv_name,
            conv_node.layer().layer_type()
        );
    };
    assert_eq!(
        conv_node.inputs(),
        [ny_propagate::NETWORK_INPUT.to_string()],
        "conv1 must read the network input directly (no pre-normalization) for \
         the L1 cut affine rows to be exact"
    );
    assert_eq!(
        conv.input_shape,
        Some((32, 32)),
        "conv1 input_shape must be set for cut generation"
    );
    println!("first ReLU node: '{relu_name}', producer conv: '{conv_name}'");

    // --- Generate the k=3 L1 cuts on the real box (C1.5). ---
    let cuts = generate_l1_cuts(conv, &xl, &xu, 3);
    assert!(!cuts.is_empty(), "expected L1 cuts at the prop885 box");
    let total_b: f64 = cuts.iter().map(|c| f64::from(c.cut.bound)).sum();
    println!(
        "generated {} k=3 L1 cuts (sum of bounds B = {:.3})",
        cuts.len(),
        total_b
    );

    // --- 99 margin objectives: row_j = e_label − e_j (j != label). ---
    let mut spec = Array2::<f32>::zeros((NUM_CLASSES - 1, NUM_CLASSES));
    let mut row = 0usize;
    for j in 0..NUM_CLASSES {
        if j == LABEL {
            continue;
        }
        spec[[row, LABEL]] = 1.0;
        spec[[row, j]] = -1.0;
        row += 1;
    }

    // --- Fixed intermediate bounds shared by every run (isolates the fold's
    // effect on the final objective backward, the C2 measurement). ---
    let node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear intermediate bounds");

    // Each run returns (floored, raw):
    // - `floored` = the API result, elementwise max(crown, IBP-spec floor) —
    //   the IBP floor is fold-INDEPENDENT, so raw crown is the C2 signal;
    // - `raw` = the captured NETWORK_INPUT LinearBounds concretized over the
    //   box (the pure CPU graph-CROWN backward product). `None` linear means
    //   the pass fell back to IBP entirely — a measurement blocker we report.
    let run = |graph: &ny_propagate::GraphNetwork| -> (Vec<f32>, Vec<f32>) {
        let (bounds, linear) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(
                &input,
                &spec,
                None,
                &node_bounds,
            )
            .expect("spec CROWN backward");
        let linear = linear.expect(
            "spec CROWN fell back to IBP (no linear map) — CPU backward loop \
             did not complete; raw crown margins unavailable",
        );
        let raw = linear.concretize_sound(&input);
        (
            bounds.lower().iter().copied().collect(),
            raw.lower().iter().copied().collect(),
        )
    };
    let min_of = |margins: &[f32]| margins.iter().copied().fold(f32::INFINITY, f32::min);

    // --- Baseline (fold gate off). ---
    let t0 = std::time::Instant::now();
    let (base_floored, base_raw) = run(&graph);
    println!(
        "baseline min margin: floored {:+.6} | raw crown {:+.6}  ({} objectives, {:.1}s)",
        min_of(&base_floored),
        min_of(&base_raw),
        base_floored.len(),
        t0.elapsed().as_secs_f32()
    );

    // --- Gate (a): NY_CUT_FOLD=1 with an EMPTY registry must be exactly equal. ---
    let _g_fold_on = ny_test_utils::env::ScopedEnvVar::set("NY_CUT_FOLD", "1");
    reset_cut_fold_applied_count();
    let (gated_floored, gated_raw) = run(&graph);
    assert_eq!(
        cut_fold_applied_count(),
        0,
        "empty registry must never fold"
    );
    assert_eq!(
        base_floored, gated_floored,
        "gate (a) FAILED: empty-registry NY_CUT_FOLD=1 run differs from baseline"
    );
    assert_eq!(
        base_raw, gated_raw,
        "gate (a) FAILED: empty-registry raw crown margins differ from baseline"
    );
    println!("gate (a) PASSED: empty-registry run exactly equals baseline (floored AND raw)");

    // --- λ grid: same λ for all cuts; entry = summed coeffs + −λ·ΣB. ---
    println!(
        "\n{:>8}  {:>14}  {:>14}  {:>10}  {:>12}  {:>12}",
        "lambda", "min floored", "min raw crown", "raw delta", "rows moved", "max row d"
    );
    println!(
        "{:>8}  {:>+14.6}  {:>+14.6}  {:>10}  {:>12}  {:>12}",
        "baseline",
        min_of(&base_floored),
        min_of(&base_raw),
        "-",
        "-",
        "-"
    );
    let mut any_improved = false;
    for &lambda in &[0.05f32, 0.1, 0.25, 0.5, 1.0] {
        let mut coeff_sum: HashMap<u32, f64> = HashMap::new();
        let mut bias_shift = 0.0f64;
        for cut in &cuts {
            for (&n, &cc) in cut.cut.neurons.iter().zip(&cut.cut.cc) {
                *coeff_sum.entry(n).or_insert(0.0) += f64::from(lambda) * f64::from(cc);
            }
            bias_shift -= f64::from(lambda) * f64::from(cut.cut.bound);
        }
        let mut coeffs: Vec<(u32, f32)> =
            coeff_sum.into_iter().map(|(n, c)| (n, c as f32)).collect();
        coeffs.sort_unstable_by_key(|&(n, _)| n);
        set_cut_fold(
            graph.cut_fold_scope(),
            &relu_name,
            CutFoldEntry {
                coeffs,
                bias_shift: bias_shift as f32,
            },
        );
        reset_cut_fold_applied_count();
        let (floored, raw) = run(&graph);
        assert!(
            cut_fold_applied_count() >= 1,
            "λ={lambda}: fold was never applied — wrong node name or non-Dense ReLU path"
        );
        let m_raw = min_of(&raw);
        let delta_raw = m_raw - min_of(&base_raw);
        let rows_moved = raw.iter().zip(&base_raw).filter(|(a, b)| a != b).count();
        let max_row_delta = raw
            .iter()
            .zip(&base_raw)
            .map(|(a, b)| a - b)
            .fold(f32::NEG_INFINITY, f32::max);
        if delta_raw > 0.0 {
            any_improved = true;
        }
        println!(
            "{lambda:>8}  {:>+14.6}  {m_raw:>+14.6}  {delta_raw:>+10.6}  {rows_moved:>12}  {max_row_delta:>+12.6}",
            min_of(&floored)
        );
    }
    println!(
        "\nC2 root-margin verdict (raw CPU crown): {}",
        if any_improved {
            "SOME λ STRICTLY IMPROVES the min root margin (SUCCESS)"
        } else {
            "no λ improves the min root margin at the root (measured FAILURE — valid result)"
        }
    );

    // Cleanup: leave the registry dark again; the env guards above restore
    // the pre-test environment on drop.
    clear_cut_fold();
}
