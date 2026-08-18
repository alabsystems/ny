// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Oracle-head CEILING: how much of the cifar100 margin gap is recoverable by a
//! PERFECT last-ReLU (FC-head) pre-activation box, on the REAL conv-DAG.
//!
//! The ay-milp program's W3 thesis is that an EXACT leaf/head solve beats the
//! CROWN triangle relaxation and yields a verdict CROWN cannot. This test
//! measures the CEILING of that benefit for the FC-head: it replaces the last
//! ReLU's pre-activation box with the TRUE sampled range (the tightest any
//! head-solve, exact-MILP included, could ever achieve — sampled ⊆ true, so this
//! is optimistic) and remeasures the min root margin. If the recovered margin is
//! ≪ the ~0.3 gap, the looseness is DISTRIBUTED across depth (a single head/leaf
//! solve cannot close the gate — the full-depth exact solve is the 494-binary MIP
//! that times out); if it recovers ≈0.3, an exact head-solve IS the path.
//!
//! UNSOUND BY DESIGN (sampled bounds underestimate the true range) — a
//! measurement of the achievable ceiling only, never a verdict.
//!
//! Run: `cargo run -p ny-onnx --release --example oracle_head_cifar100`

use ndarray::{Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;

const LABEL: usize = 85; // prop_idx_116 header: label 85
const NUM_CLASSES: usize = 100;
const SAMPLES: usize = 4000;

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
    // (Serialized + restored via the blessed env choke point — clippy env
    // wall; guards restore the pre-test environment when the test exits.)
    let _env_lock = ny_test_utils::env::lock_env();
    let _g_gpu = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_GPU", "0");
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");

    let onnx_model = load_onnx(&onnx_path).expect("load onnx");
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model
        .to_graph_network_with_options(graph_options)
        .expect("to_graph_network");
    graph.set_use_patches_mode(false);

    let vnnlib = load_vnnlib(&vnnlib_path).expect("load vnnlib");
    let (xl, xu) = vnnlib.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xl).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xu).expect("upper"),
    )
    .expect("input");

    // 99 margin objectives e_label − e_j.
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

    // LAST ReLU node and its pre-activation producer (the FC-head).
    let exec_order: Vec<String> = graph.exec_order().expect("order").to_vec();
    let last_relu = exec_order
        .iter()
        .rev()
        .find(|n| {
            graph
                .node(n)
                .is_some_and(|nd| matches!(nd.layer(), Layer::ReLU(_)))
        })
        .expect("a ReLU")
        .clone();
    let pre_node = graph.node(&last_relu).expect("relu").inputs()[0].clone();
    println!("FC-head: last ReLU '{last_relu}', pre-activation node '{pre_node}'");

    let mut node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear bounds");

    let min_of = |m: &[f32]| m.iter().copied().fold(f32::INFINITY, f32::min);
    let run = |graph: &ny_propagate::GraphNetwork,
               nb: &std::collections::HashMap<String, BoundedTensor>|
     -> Vec<f32> {
        let (_bounds, linear) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec, None, nb)
            .expect("spec CROWN");
        let linear = linear.expect("no linear map (fell back to IBP)");
        linear
            .concretize_sound(&input)
            .lower()
            .iter()
            .copied()
            .collect()
    };

    // Baseline min margin (CROWN relaxation at the head).
    let base = run(&graph, &node_bounds);
    let base_min = min_of(&base);

    // ORACLE head box: sample SAMPLES inputs, forward the TRUE net, take the
    // elementwise min/max of the FC-head pre-activation.
    let orig = node_bounds.get(&pre_node).expect("pre_node bounds").clone();
    let n = orig.flatten().lower().len();
    let mut lo = vec![f32::INFINITY; n];
    let mut hi = vec![f32::NEG_INFINITY; n];
    let iflat = input.flatten();
    let il: Vec<f32> = iflat.lower().iter().copied().collect();
    let iu: Vec<f32> = iflat.upper().iter().copied().collect();
    let nin = il.len();
    let mut rng: u64 = 0x243f_6a88_85a3_08d3;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..SAMPLES {
        let mut pt = Vec::with_capacity(nin);
        for k in 0..nin {
            let t = next() as f32;
            pt.push(il[k] + (iu[k] - il[k]) * t);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), pt).expect("pt shape");
        let point = BoundedTensor::new(arr.clone(), arr).expect("pt");
        let acts = graph
            .collect_node_activations_pointwise(&point, None)
            .expect("pointwise acts");
        let v = acts.get(&pre_node).expect("pre_node act");
        let vf = v.flatten();
        for (k, &x) in vf.lower().iter().enumerate() {
            if x < lo[k] {
                lo[k] = x;
            }
            if x > hi[k] {
                hi[k] = x;
            }
        }
    }
    let shape = orig.lower().shape().to_vec();
    let oracle = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lo.clone()).expect("lo shape"),
        ArrayD::from_shape_vec(IxDyn(&shape), hi.clone()).expect("hi shape"),
    )
    .expect("oracle box");

    // Width shrink factor (how much tighter the oracle head box is).
    let ow: f64 = lo.iter().zip(&hi).map(|(l, h)| (*h - *l) as f64).sum();
    let bw: f64 = {
        let of = orig.flatten();
        of.lower()
            .iter()
            .zip(of.upper().iter())
            .map(|(l, h)| (*h - *l) as f64)
            .sum()
    };

    node_bounds.insert(pre_node, oracle);
    let oracle_run = run(&graph, &node_bounds);
    let oracle_min = min_of(&oracle_run);

    let recovered = oracle_min - base_min;
    println!("\n===== cifar100 ORACLE-HEAD CEILING (real CIFAR100_resnet_medium, {SAMPLES} samples) =====");
    println!("FC-head pre-activation total width: CROWN {bw:.2} -> oracle(true-sampled) {ow:.2}  ({:.1}% of CROWN)", 100.0 * ow / bw);
    println!("baseline min margin (CROWN head)   = {base_min:+.6}");
    println!("oracle   min margin (PERFECT head) = {oracle_min:+.6}");
    println!("recovered by a perfect FC-head     = {recovered:+.6}  (CEILING of any head/leaf solve, incl. exact MILP)");
    println!(
        "VERDICT: a perfect FC-head recovers {recovered:.4} of margin. The ny-vs-α,β gap is ~0.3.\n=> {}",
        if recovered >= 0.25 {
            "the FC-head is the dominant bottleneck; an EXACT head-solve (ay-milp W3) could close the gate."
        } else if recovered >= 0.05 {
            "the FC-head is a PARTIAL contributor; the looseness is distributed across depth — a single head/leaf solve is insufficient, the full-depth exact solve is the 494-binary MIP that times out."
        } else {
            "the FC-head is NOT the bottleneck; the looseness is upstream/distributed — head/leaf exact solve cannot close the gate."
        }
    );

    // Env guards above restore the pre-test environment on drop.
}
