// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! f32-SOUNDNESS-SLACK measurement on the REAL cifar100 conv-DAG
//! (`docs/CERTIFIED_CUT_CROWN_DESIGN.md`, the f32-slack-vs-relaxation crux).
//!
//! The ~0.3 cifar100 margin gap vs α,β-CROWN was hypothesised to be f32
//! SOUNDNESS SLACK — the outward-rounded `lower_a_err` the sound backward folds
//! into the bias at every node — which an f64 backward would eliminate. This
//! test MEASURES that slack directly on `CIFAR100_resnet_medium`: it runs the
//! real CPU graph-CROWN margin backward with `NY_SLACK_PROBE=1` (which sums, per
//! objective, the exact margin-units each eager fold subtracts) and prints the
//! binding objective's total accumulated f32 slack next to its margin. If the
//! slack is ≪ 0.3, the gap is relaxation looseness (f64 cannot close it), not
//! precision — the decisive datum the GPU/Metal lane could not produce.
//!
//! Run (real benchmark assets; release):
//! `cargo run -p ny-onnx --release --example slack_probe_cifar100`

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;

/// prop_idx_116 (sidx_8002) — the vnnlib header states `label: 85`.
const LABEL: usize = 85;
const NUM_CLASSES: usize = 100;

fn main() {
    let dir = workspace_root().join("benchmarks/vnncomp2025/benchmarks/cifar100_2024");
    let onnx_path = dir.join("onnx/CIFAR100_resnet_medium.onnx");
    let vnnlib_path =
        dir.join("vnnlib/CIFAR100_resnet_medium_prop_idx_116_sidx_8002_eps_0.0039.vnnlib");
    assert!(
        onnx_path.exists() && vnnlib_path.exists(),
        "cifar100 benchmark assets are required under {}; install the VNN-COMP 2025 corpus",
        dir.display()
    );

    // Force the proven CPU graph backward loop (the only arm carrying the eager
    // coeff-err fold the probe sums), and arm the probe.
    // (Serialized + restored via the blessed env choke point — clippy env
    // wall; guards restore the pre-test environment when the test exits.)
    let _env_lock = ny_test_utils::env::lock_env();
    let _g_gpu = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_GPU", "0");
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");
    let _g_probe = ny_test_utils::env::ScopedEnvVar::set("NY_SLACK_PROBE", "1");

    let onnx_model = load_onnx(&onnx_path).expect("load CIFAR100_resnet_medium.onnx");
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model
        .to_graph_network_with_options(graph_options)
        .expect("to_graph_network");
    graph.set_use_patches_mode(false);

    let vnnlib = load_vnnlib(&vnnlib_path).expect("load vnnlib");
    assert_eq!(vnnlib.num_inputs, 3 * 32 * 32);
    assert_eq!(vnnlib.num_outputs, NUM_CLASSES);
    let (xl, xu) = vnnlib.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xl).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xu).expect("upper shape"),
    )
    .expect("input BoundedTensor");

    // 99 margin objectives row_j = e_label − e_j (j != label).
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

    let node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear intermediate bounds");

    // Drain any prior accumulation, run the sound CPU margin backward (the eager
    // folds accumulate into the probe), then read the per-objective slack.
    let _ = ny_propagate::bounds::slack_probe_take();
    let t0 = std::time::Instant::now();
    let (bounds, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec, None, &node_bounds)
        .expect("spec CROWN backward");
    let linear = linear.expect("CPU backward fell back to IBP (no linear map)");
    let raw = linear.concretize_sound(&input);
    let raw_lower: Vec<f32> = raw.lower().iter().copied().collect();
    let floored_lower: Vec<f32> = bounds.lower().iter().copied().collect();
    let slack = ny_propagate::bounds::slack_probe_take();

    // Binding (min-margin) objective and its accumulated f32 slack.
    let (mut worst_row, mut worst_margin) = (0usize, f32::INFINITY);
    for (r, &m) in raw_lower.iter().enumerate() {
        if m < worst_margin {
            worst_margin = m;
            worst_row = r;
        }
    }
    let binding_slack = slack.get(worst_row).copied().unwrap_or(0.0);
    let max_slack = slack.iter().copied().fold(0.0f64, f64::max);
    let sum_slack: f64 = slack.iter().sum();
    let min_floored = floored_lower.iter().copied().fold(f32::INFINITY, f32::min);

    println!(
        "\n===== cifar100 f32 SOUNDNESS SLACK measurement (real CIFAR100_resnet_medium) ====="
    );
    println!(
        "objectives={}  backward={:.1}s  fold_rows_touched={}",
        raw_lower.len(),
        t0.elapsed().as_secs_f32(),
        slack.len()
    );
    println!("binding objective row={worst_row}  raw_margin_lb={worst_margin:+.6}  min_floored={min_floored:+.6}");
    println!("binding_f32_slack   = {binding_slack:.8}  (margin-units the f32 rounding removed from the BINDING objective)");
    println!("max_row_f32_slack   = {max_slack:.8}  (worst over all 99 objectives)");
    println!("total_f32_slack_sum = {sum_slack:.8}  (all rows)");
    println!(
        "VERDICT: f32 soundness slack on the binding objective = {binding_slack:.6}.\n\
         The ny-vs-α,β-CROWN gap is ~0.3. If binding_f32_slack ≪ 0.3, an f64 backward CANNOT\n\
         close the gap — it is RELAXATION looseness, not precision slack.\n\
         => {}",
        if binding_slack < 0.03 {
            "CONFIRMED: f32 slack is negligible vs the 0.3 gap; f64 is a dead end for the gate."
        } else {
            "f32 slack is a material fraction of the gap; f64 warrants further study."
        }
    );

    // Sanity: the probe must never invent slack — every accumulated value ≥ 0
    // (each fold penalty p = Σ e·mag with e = |err| ≥ 0, mag ≥ 0).
    assert!(
        slack.iter().all(|&s| s >= 0.0),
        "slack accumulator must be non-negative"
    );

    // Env guards above restore the pre-test environment on drop.
}
