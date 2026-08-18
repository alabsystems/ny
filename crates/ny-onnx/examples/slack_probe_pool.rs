// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! POOL generalization of `slack_probe_cifar100.rs` — measure the f32
//! certified-error charge (`NY_SLACK_PROBE`) against the binding root margin on
//! ANY cifar100/tinyimagenet row, in BOTH dense and patches coefficient modes.
//!
//! This is the decisive datum for the "EFT-compensated certified error" (S2)
//! question: the probe sums, per objective row, the exact margin-units the
//! sound backward's certified coefficient-error channel subtracts from the
//! lower bound. An EFT/a-posteriori channel can recover AT MOST that much.
//!
//! Diagnostic only: print-only, never mutates a bound or a verdict.
//!
//! Run:
//! ```text
//! NY_PROBE_ONNX=... NY_PROBE_VNNLIB=... NY_PROBE_PATCHES=1 \
//!   cargo run -p ny-onnx --release --example slack_probe_pool
//! ```

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_tensor::BoundedTensor;
use std::path::PathBuf;

fn parse_label(vnnlib_path: &std::path::Path) -> usize {
    let text = std::fs::read_to_string(vnnlib_path).expect("read vnnlib");
    for line in text.lines().take(5) {
        if let Some(idx) = line.find("label:") {
            let rest = &line[idx + "label:".len()..];
            let digits: String = rest
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if !digits.is_empty() {
                return digits.parse().expect("label digits");
            }
        }
    }
    panic!("no `label:` header comment in {}", vnnlib_path.display());
}

fn main() {
    let onnx_path = PathBuf::from(std::env::var("NY_PROBE_ONNX").expect("NY_PROBE_ONNX"));
    let vnnlib_path = PathBuf::from(std::env::var("NY_PROBE_VNNLIB").expect("NY_PROBE_VNNLIB"));
    let patches = std::env::var("NY_PROBE_PATCHES").ok().as_deref() == Some("1");
    let label = parse_label(&vnnlib_path);

    let _env_lock = ny_test_utils::env::lock_env();
    let _g_gpu = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_GPU", "0");
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");
    let _g_probe = ny_test_utils::env::ScopedEnvVar::set("NY_SLACK_PROBE", "1");

    let onnx_model = load_onnx(&onnx_path).expect("load onnx");
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model
        .to_graph_network_with_options(graph_options)
        .expect("to_graph_network");
    graph.set_use_patches_mode(patches);

    let vnnlib = load_vnnlib(&vnnlib_path).expect("load vnnlib");
    let num_classes = vnnlib.num_outputs;
    let n_in = vnnlib.num_inputs;
    let side = ((n_in / 3) as f64).sqrt().round() as usize;
    assert_eq!(3 * side * side, n_in, "expected a 3xSxS image input");

    let (xl, xu) = vnnlib.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, side, side]), xl).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[3, side, side]), xu).expect("upper shape"),
    )
    .expect("input BoundedTensor");

    let mut spec = Array2::<f32>::zeros((num_classes - 1, num_classes));
    let mut row = 0usize;
    for j in 0..num_classes {
        if j == label {
            continue;
        }
        spec[[row, label]] = 1.0;
        spec[[row, j]] = -1.0;
        row += 1;
    }

    let t_fwd = std::time::Instant::now();
    let node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear intermediate bounds");
    let fwd_secs = t_fwd.elapsed().as_secs_f32();

    let _ = ny_propagate::bounds::slack_probe_take();
    let t0 = std::time::Instant::now();
    let (_bounds, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec, None, &node_bounds)
        .expect("spec CROWN backward");
    let linear = linear.expect("CPU backward fell back to IBP (no linear map)");
    let raw = linear.concretize_sound(&input);
    let raw_lower: Vec<f32> = raw.lower().iter().copied().collect();
    let slack = ny_propagate::bounds::slack_probe_take();

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
    let verified = raw_lower.iter().filter(|&&m| m > 0.0).count();

    println!(
        "RESULT vnnlib={} patches={} label={} objectives={} fwd_s={:.1} bwd_s={:.1} \
         binding_row={} binding_margin_lb={:+.6} binding_slack={:.8} max_slack={:.8} \
         sum_slack={:.8} rows_verified={}/{} slack_over_gap={:.3e}",
        vnnlib_path.file_name().unwrap().to_string_lossy(),
        patches as u8,
        label,
        raw_lower.len(),
        fwd_secs,
        t0.elapsed().as_secs_f32(),
        worst_row,
        worst_margin,
        binding_slack,
        max_slack,
        sum_slack,
        verified,
        raw_lower.len(),
        if worst_margin < 0.0 {
            binding_slack / (-worst_margin as f64)
        } else {
            0.0
        }
    );

    assert!(
        slack.iter().all(|&s| s >= 0.0),
        "slack accumulator must be non-negative"
    );
}
