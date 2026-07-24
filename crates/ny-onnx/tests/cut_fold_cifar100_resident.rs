// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN C2 root-margin experiment on the SOUND RESIDENT GPU
//! lane (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C2) — the lane that produces
//! the real prop885 root margins (the CPU graph-CROWN lane is numerically
//! exploded on this net; see `cut_fold_cifar100.rs` for that measurement).
//!
//! Loads the REAL cifar100 `CIFAR100_resnet_medium` + prop885 ε=0.0039 box,
//! generates k=3 L1 cuts from conv1 (C1.5, `generate_l1_cuts`), then measures
//! the 99-objective root lower margins through the C-matrix-seeded sound
//! GPU-resident resnet backward (#w4-root-gpu route) WITHOUT cuts and WITH
//! the cuts folded at the first ReLU's lower-side post-activation coefficient
//! (dark `NY_CUT_FOLD_RESIDENT` registry; fold site:
//! `backward_branch_cut_fold` in `crown_backward_sound_resident.rs`).
//!
//! Gates:
//! (a) NY_CUT_FOLD_RESIDENT=1 with an empty registry ⇒ margins EXACTLY equal
//!     to baseline; a registered λ=0 entry (which exercises the host-side
//!     branch SPLIT at the fold ReLU) must ALSO be exactly equal — that is
//!     the split-transparency proof;
//! (b) the λ-margin table (printed) — SUCCESS iff some λ strictly improves
//!     the min margin; a no-improvement result is a valid measurement too.
//!
//! C2b (`c2b_signed_cut_fold_resident_root_margin_experiment`): the corrected
//! transfer gate after C2's measured negative — OBJECTIVE-SIGNED groups. The
//! λ=0 capture pass reads the incoming lower-side coefficient row `A` at the
//! fold frontier (dark `NY_CUT_FOLD_CAPTURE`), groups are picked among
//! Relu-unstable neurons with `a_i < 0` on the WORST margin row (where the
//! `+λ·cc` fold cancels upper-chord intercept payments instead of adding
//! mass), and the λ grid re-runs for `cc = 1`, `cc ∝ |a|`, and the
//! first-order-positive filtered set.
//!
//! Run (requires the Metal/Vulkan GPU + real benchmark assets). The two
//! experiments mutate process env + the global fold registry — run ONE test
//! at a time (name filter or `--test-threads=1`):
//! `cargo test -p ny-onnx --release --test cut_fold_cifar100_resident -- --ignored --nocapture --test-threads=1`

use ndarray::{Array2, ArrayD, IxDyn};
use ny_gpu::wgpu_device::{
    clear_resident_cut_fold, reset_resident_cut_fold_applied_count,
    resident_cut_fold_applied_count, set_resident_cut_fold, take_resident_cut_fold_capture,
    ResidentCutFold,
};
use ny_gpu::{Backend, ComputeDevice};
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_propagate::beta_crown::{generate_l1_cuts, generate_l1_cuts_signed, L1Cut, SignedCcMode};
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;
use std::collections::HashMap;

/// prop885 is a targeted-robustness property: label 44, unsafe iff any
/// `Y_j >= Y_44` (j != 44).
const LABEL: usize = 44;
const NUM_CLASSES: usize = 100;

#[test]
#[ignore = "multi-minute experiment on real cifar100 benchmark assets + GPU; run with --ignored"]
fn c2_cut_fold_resident_root_margin_experiment() {
    let dir = workspace_root().join("benchmarks/vnncomp2025/benchmarks/cifar100_2024");
    let onnx_path = dir.join("onnx/CIFAR100_resnet_medium.onnx");
    let vnnlib_path =
        dir.join("vnnlib/CIFAR100_resnet_medium_prop_idx_885_sidx_7654_eps_0.0039.vnnlib");
    if !onnx_path.exists() || !vnnlib_path.exists() {
        eprintln!("SKIP: cifar100 benchmark assets not present under {dir:?}");
        return;
    }
    let Ok(device) = ComputeDevice::new(Backend::Wgpu) else {
        eprintln!("SKIP: no wgpu GPU adapter available");
        return;
    };
    let engine: &dyn ny_core::GemmEngine = &device;

    // Isolate the #w4-root-gpu route (b): disable the forward-linear C-margin
    // root candidates (a)/(c) so the measured margins are the resident resnet
    // backward's alone; keep NY_SPEC_ROOT_GPU at its default (ON). Lift the
    // cumulative resnet-GPU time budget so the 8-run λ grid never gets its
    // later runs silently refused (which would fall back to the CPU loop and
    // poison the comparison — the cache discriminator below would catch it).
    // (Serialized + restored via the blessed env choke point — clippy env
    // wall; guards restore the pre-test environment when the test exits.)
    let _env_lock = ny_test_utils::env::lock_env();
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");
    let _g_budget = ny_test_utils::env::ScopedEnvVar::set("NY_RESNET_GPU_TIME_BUDGET_MS", "600000");
    let _g_trace = ny_test_utils::env::ScopedEnvVar::set("NY_RESNET_GPU_TRACE", "1");
    let _g_fold_off = ny_test_utils::env::ScopedEnvVar::unset("NY_CUT_FOLD_RESIDENT");
    clear_resident_cut_fold();

    // --- Load model as a GraphNetwork (same options as the CPU-lane C2
    // experiment, for comparability). ---
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

    // --- Identify the FIRST ReLU node and its producer conv (the fold target:
    // the resident fold registry targets the LAST Activation in fold order,
    // which is exactly this innermost ReLU). ---
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
            "first ReLU '{}' producer '{}' is {}, not Conv2d",
            relu_name,
            conv_name,
            conv_node.layer().layer_type()
        );
    };
    assert_eq!(
        conv_node.inputs(),
        [ny_propagate::NETWORK_INPUT.to_string()],
        "conv1 must read the network input directly for the L1 cut rows to be exact"
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

    // --- Fixed intermediate bounds shared by every run (same map as the
    // CPU-lane experiment: isolates the fold's effect on the final objective
    // backward). ---
    let node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear intermediate bounds");

    // One root evaluation through the C-matrix-seeded sound GPU resnet
    // backward. The `None` cache is the discriminator that the GPU root pass
    // actually FIRED (the CPU loop would capture a linear cache).
    let run = |graph: &ny_propagate::GraphNetwork| -> Vec<f32> {
        let (bounds, cache) = graph
            .propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
                &input,
                &spec,
                Some(engine),
                &node_bounds,
                None,
            )
            .expect("spec CROWN root");
        assert!(
            cache.is_none(),
            "expected the GPU resnet root pass to fire (no linear cache); a \
             Some(cache) means the CPU backward loop answered instead — wrong lane"
        );
        bounds.lower().iter().copied().collect()
    };
    let min_of = |margins: &[f32]| margins.iter().copied().fold(f32::INFINITY, f32::min);

    // Build the fold entry for a given λ: summed per-neuron coefficients
    // (accumulated in f64) + the −λ·ΣB lower-bias shift.
    let entry_for = |lambda: f32| -> ResidentCutFold {
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
        ResidentCutFold {
            coeffs,
            bias_shift: bias_shift as f32,
            // Legacy C2 experiment lane: no pre-activation channel, plain-f32
            // add (byte-identical to the pre-stem behaviour).
            pre_coeffs: Vec::new(),
            sound_round: false,
        }
    };

    // --- Baseline (fold gate off). ---
    let t0 = std::time::Instant::now();
    let base = run(&graph);
    println!(
        "baseline min margin (resident GPU root): {:+.6}  ({} objectives, {:.1}s)",
        min_of(&base),
        base.len(),
        t0.elapsed().as_secs_f32()
    );

    // --- Gate (a1): NY_CUT_FOLD_RESIDENT=1 with an EMPTY registry ⇒ exactly
    // equal (no split, no fold — the zero-cost dark path). ---
    let _g_fold_on = ny_test_utils::env::ScopedEnvVar::set("NY_CUT_FOLD_RESIDENT", "1");
    reset_resident_cut_fold_applied_count();
    let gated = run(&graph);
    assert_eq!(
        resident_cut_fold_applied_count(),
        0,
        "empty registry must never fold"
    );
    assert_eq!(
        base, gated,
        "gate (a1) FAILED: empty-registry NY_CUT_FOLD_RESIDENT=1 run differs from baseline"
    );
    println!("gate (a1) PASSED: empty-registry run exactly equals baseline");

    // --- Gate (a2): λ=0 entry ⇒ the branch IS split at the fold ReLU (the
    // host round-trip runs) but every coefficient add is +0.0 and the bias
    // shift is 0 — margins must still be exactly equal. This is the
    // split-transparency proof for the fold site. ---
    set_resident_cut_fold(entry_for(0.0));
    reset_resident_cut_fold_applied_count();
    let lambda0 = run(&graph);
    assert!(
        resident_cut_fold_applied_count() >= 1,
        "λ=0: fold site was never exercised — target ReLU not reached by the \
         resident segment walk"
    );
    assert_eq!(
        base, lambda0,
        "gate (a2) FAILED: λ=0 split run differs from baseline (branch split \
         at the fold ReLU is not bit-transparent)"
    );
    println!(
        "gate (a2) PASSED: λ=0 (split exercised, applied={}) exactly equals baseline",
        resident_cut_fold_applied_count()
    );

    // --- λ grid. ---
    println!(
        "\n{:>8}  {:>14}  {:>10}  {:>12}  {:>12}",
        "lambda", "min margin", "delta", "rows moved", "max row d"
    );
    println!(
        "{:>8}  {:>+14.6}  {:>10}  {:>12}  {:>12}",
        "baseline",
        min_of(&base),
        "-",
        "-",
        "-"
    );
    let mut any_improved = false;
    for &lambda in &[0.05f32, 0.1, 0.25, 0.5, 1.0] {
        set_resident_cut_fold(entry_for(lambda));
        reset_resident_cut_fold_applied_count();
        let margins = run(&graph);
        assert!(
            resident_cut_fold_applied_count() >= 1,
            "λ={lambda}: fold was never applied"
        );
        let m = min_of(&margins);
        let delta = m - min_of(&base);
        let rows_moved = margins.iter().zip(&base).filter(|(a, b)| a != b).count();
        let max_row_delta = margins
            .iter()
            .zip(&base)
            .map(|(a, b)| a - b)
            .fold(f32::NEG_INFINITY, f32::max);
        if delta > 0.0 {
            any_improved = true;
        }
        println!(
            "{lambda:>8}  {m:>+14.6}  {delta:>+10.6}  {rows_moved:>12}  {max_row_delta:>+12.6}"
        );
    }
    println!(
        "\nC2 root-margin verdict (resident GPU lane): {}",
        if any_improved {
            "SOME λ STRICTLY IMPROVES the min root margin (SUCCESS)"
        } else {
            "no λ improves the min root margin at the root (measured FAILURE — valid result)"
        }
    );

    // Cleanup: leave the registry dark again; the env guards above restore
    // the pre-test environment on drop.
    clear_resident_cut_fold();
}

/// Build the resident fold entry for a cut set at a given λ: summed
/// per-neuron coefficients (f64-accumulated) + the `−λ·ΣB` lower-bias shift.
fn fold_entry(cuts: &[L1Cut], lambda: f32) -> ResidentCutFold {
    let mut coeff_sum: HashMap<u32, f64> = HashMap::new();
    let mut bias_shift = 0.0f64;
    for cut in cuts {
        for (&n, &cc) in cut.cut.neurons.iter().zip(&cut.cut.cc) {
            *coeff_sum.entry(n).or_insert(0.0) += f64::from(lambda) * f64::from(cc);
        }
        bias_shift -= f64::from(lambda) * f64::from(cut.cut.bound);
    }
    let mut coeffs: Vec<(u32, f32)> = coeff_sum.into_iter().map(|(n, c)| (n, c as f32)).collect();
    coeffs.sort_unstable_by_key(|&(n, _)| n);
    ResidentCutFold {
        coeffs,
        bias_shift: bias_shift as f32,
        ..Default::default()
    }
}

/// C2b — objective-SIGNED groups (`docs/CERTIFIED_CUT_CROWN_DESIGN.md`,
/// "objective-aware signed groups" next-lever note). Same lane, same box,
/// same fold site as the C2 experiment above; only the GROUP SELECTION
/// changes: neurons with NEGATIVE incoming lower-side objective coefficient
/// at the fold ReLU (captured via `NY_CUT_FOLD_CAPTURE` on a λ=0 pass),
/// where `+λ·cc_i` cancels the upper-chord intercept `|a_i|·u(−l)/(u−l)`
/// instead of adding fresh relu mass.
///
/// SUCCESS iff some λ strictly improves the min root margin vs baseline;
/// a second negative here means root-level L1 cuts are dead on this net.
#[test]
#[ignore = "multi-minute experiment on real cifar100 benchmark assets + GPU; run with --ignored"]
fn c2b_signed_cut_fold_resident_root_margin_experiment() {
    let dir = workspace_root().join("benchmarks/vnncomp2025/benchmarks/cifar100_2024");
    let onnx_path = dir.join("onnx/CIFAR100_resnet_medium.onnx");
    let vnnlib_path =
        dir.join("vnnlib/CIFAR100_resnet_medium_prop_idx_885_sidx_7654_eps_0.0039.vnnlib");
    if !onnx_path.exists() || !vnnlib_path.exists() {
        eprintln!("SKIP: cifar100 benchmark assets not present under {dir:?}");
        return;
    }
    let Ok(device) = ComputeDevice::new(Backend::Wgpu) else {
        eprintln!("SKIP: no wgpu GPU adapter available");
        return;
    };
    let engine: &dyn ny_core::GemmEngine = &device;

    // Same route isolation as the C2 experiment; the budget covers the longer
    // (up to 3-variant) grid.
    // (Serialized + restored via the blessed env choke point — clippy env
    // wall; guards restore the pre-test environment when the test exits.)
    let _env_lock = ny_test_utils::env::lock_env();
    let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
    let _g_alpha = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_ALPHA", "0");
    let _g_budget =
        ny_test_utils::env::ScopedEnvVar::set("NY_RESNET_GPU_TIME_BUDGET_MS", "3600000");
    let _g_trace = ny_test_utils::env::ScopedEnvVar::set("NY_RESNET_GPU_TRACE", "1");
    let _g_fold_off = ny_test_utils::env::ScopedEnvVar::unset("NY_CUT_FOLD_RESIDENT");
    let _g_cap_off = ny_test_utils::env::ScopedEnvVar::unset("NY_CUT_FOLD_CAPTURE");
    clear_resident_cut_fold();

    let onnx_model = load_onnx(&onnx_path).expect("load CIFAR100_resnet_medium.onnx");
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model
        .to_graph_network_with_options(graph_options)
        .expect("to_graph_network");
    graph.set_use_patches_mode(false);

    let vnnlib = load_vnnlib(&vnnlib_path).expect("load prop885 vnnlib");
    assert_eq!(vnnlib.num_inputs, 3 * 32 * 32);
    assert_eq!(vnnlib.num_outputs, NUM_CLASSES);
    let (xl, xu) = vnnlib.split_input_bounds_f32();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xl.clone()).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), xu.clone()).expect("upper shape"),
    )
    .expect("input BoundedTensor");

    // First ReLU + its producer conv (the fold target), as in C2.
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
            "first ReLU '{}' producer '{}' is {}, not Conv2d",
            relu_name,
            conv_name,
            conv_node.layer().layer_type()
        );
    };
    assert_eq!(
        conv_node.inputs(),
        [ny_propagate::NETWORK_INPUT.to_string()],
        "conv1 must read the network input directly for the L1 cut rows to be exact"
    );
    println!("first ReLU node: '{relu_name}', producer conv: '{conv_name}'");

    // Conv1 output layout (for the capture-dim check + flat indexing).
    let ksh = conv.kernel.shape();
    let (out_c, kh, kw) = (ksh[0], ksh[2], ksh[3]);
    let (ih, iw) = conv.input_shape.expect("conv1 input shape");
    let oh = (ih + 2 * conv.padding.0 - (conv.dilation.0 * (kh - 1) + 1)) / conv.stride.0 + 1;
    let ow = (iw + 2 * conv.padding.1 - (conv.dilation.1 * (kw - 1) + 1)) / conv.stride.1 + 1;

    // 99 margin objectives, fixed intermediate bounds, root runner — as in C2.
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
    let run = |graph: &ny_propagate::GraphNetwork| -> Vec<f32> {
        let (bounds, cache) = graph
            .propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
                &input,
                &spec,
                Some(engine),
                &node_bounds,
                None,
            )
            .expect("spec CROWN root");
        assert!(
            cache.is_none(),
            "expected the GPU resnet root pass to fire (no linear cache); a \
             Some(cache) means the CPU backward loop answered instead — wrong lane"
        );
        bounds.lower().iter().copied().collect()
    };
    let min_of = |margins: &[f32]| margins.iter().copied().fold(f32::INFINITY, f32::min);

    // --- Baseline + worst row. ---
    let t0 = std::time::Instant::now();
    let base = run(&graph);
    let base_min = min_of(&base);
    let worst_row = base
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("nonempty margins");
    let worst_class = if worst_row < LABEL {
        worst_row
    } else {
        worst_row + 1
    };
    println!(
        "baseline min margin (resident GPU root): {:+.6} at row {} (class {}), \
         {} objectives, {:.1}s",
        base_min,
        worst_row,
        worst_class,
        base.len(),
        t0.elapsed().as_secs_f32()
    );

    // --- Capture pass: λ=0-equivalent EMPTY entry exercises the branch split
    // at the fold ReLU and copies the lower-side frontier row(s); margins must
    // stay exactly equal (split + capture transparency). ---
    let _g_fold_on = ny_test_utils::env::ScopedEnvVar::set("NY_CUT_FOLD_RESIDENT", "1");
    let g_cap_on = ny_test_utils::env::ScopedEnvVar::set("NY_CUT_FOLD_CAPTURE", "1");
    set_resident_cut_fold(ResidentCutFold {
        coeffs: Vec::new(),
        bias_shift: 0.0,
        ..Default::default()
    });
    reset_resident_cut_fold_applied_count();
    let cap_run = run(&graph);
    assert!(
        resident_cut_fold_applied_count() >= 1,
        "capture pass: fold site was never exercised"
    );
    assert_eq!(
        base, cap_run,
        "gate FAILED: capture pass (split + empty fold) differs from baseline"
    );
    let cap = take_resident_cut_fold_capture().expect("capture must be stored");
    drop(g_cap_on); // capture flag off for the λ grid below (restores pre-guard state)
    assert_eq!(
        cap.num_specs,
        NUM_CLASSES - 1,
        "one captured row per objective"
    );
    assert_eq!(
        cap.dim,
        out_c * oh * ow,
        "captured frontier dim must be conv1's flat output ({out_c}x{oh}x{ow})"
    );
    println!(
        "gate PASSED: capture pass exactly equals baseline; captured A is {}x{}",
        cap.num_specs, cap.dim
    );
    let a_worst = &cap.lower_a[worst_row * cap.dim..(worst_row + 1) * cap.dim];

    // --- Signed group variants + diagnostics (host-only, cheap). ---
    let (cuts_u, diag_u) =
        generate_l1_cuts_signed(conv, &xl, &xu, 3, a_worst, SignedCcMode::Uniform, false);
    let (cuts_p, diag_p) =
        generate_l1_cuts_signed(conv, &xl, &xu, 3, a_worst, SignedCcMode::PropAbsA, false);
    let (cuts_f, diag_f) =
        generate_l1_cuts_signed(conv, &xl, &xu, 3, a_worst, SignedCcMode::Uniform, true);

    // Sign-blind C2 reference on this row: how much negative mass did the
    // top-3-by-upper groups cover, and how much ≥0 mass did they add?
    let blind = generate_l1_cuts(conv, &xl, &xu, 3);
    let (blind_neg, blind_pos) = blind
        .iter()
        .flat_map(|c| &c.cut.neurons)
        .map(|&i| f64::from(a_worst[i as usize]))
        .fold((0.0f64, 0.0f64), |(n, p), a| {
            (n + a.min(0.0), p + a.max(0.0))
        });

    println!("\nnegative-mass diagnostic (worst row {worst_row}, ReLU '{relu_name}'):");
    println!(
        "  Σ min(a,0) over ALL {} neurons   = {:+.4}",
        cap.dim, diag_u.total_neg_mass
    );
    println!(
        "  Σ min(a,0) over UNSTABLE neurons = {:+.4}",
        diag_u.unstable_neg_mass
    );
    println!(
        "  negative-a unstable neurons      = {}",
        diag_u.negative_unstable_neurons
    );
    println!(
        "  sign-blind C2 groups (reference): covered neg mass {:+.4}, added ≥0 mass {:+.4}",
        blind_neg, blind_pos
    );
    for (name, cuts, diag) in [
        ("signed cc=1", &cuts_u, &diag_u),
        ("signed cc∝|a|", &cuts_p, &diag_p),
        ("signed cc=1 +filter", &cuts_f, &diag_f),
    ] {
        println!(
            "  {name}: {} cuts ({} same-pos, {} cross-pos), {} neurons, covered neg mass \
             {:+.4}, ΣB {:.4}, Σcc·u(−l)/(u−l) {:.4} (first-order net {:+.4}/λ)",
            cuts.len(),
            diag.same_pos_groups,
            diag.cross_pos_groups,
            diag.grouped_neurons,
            diag.covered_neg_mass,
            diag.sum_b,
            diag.intercept_recovery_rate,
            diag.intercept_recovery_rate - diag.sum_b,
        );
    }
    assert!(
        !cuts_u.is_empty(),
        "expected negative-a unstable neurons at the prop885 box"
    );

    // --- Gate: λ=0 with the SIGNED entry (adds +0.0 at the signed indices)
    // must stay exactly equal. ---
    set_resident_cut_fold(fold_entry(&cuts_u, 0.0));
    reset_resident_cut_fold_applied_count();
    let lambda0 = run(&graph);
    assert!(
        resident_cut_fold_applied_count() >= 1,
        "λ=0: fold site not exercised"
    );
    assert_eq!(
        base, lambda0,
        "gate FAILED: λ=0 signed-entry run differs from baseline"
    );
    println!("gate PASSED: λ=0 signed entry exactly equals baseline\n");

    // --- λ grids. ---
    let grid = [0.02f32, 0.05, 0.1, 0.25, 0.5];
    let mut best: Option<(String, f32, f32)> = None; // (variant, λ, delta)
    let mut variants: Vec<(&str, &Vec<L1Cut>)> =
        vec![("signed cc=1", &cuts_u), ("signed cc∝|a|", &cuts_p)];
    if cuts_f.len() < cuts_u.len() && !cuts_f.is_empty() {
        variants.push(("signed cc=1 +filter", &cuts_f));
    } else if cuts_f.is_empty() {
        println!("(first-order filter removes every group — filtered grid skipped)");
    }
    for (name, cuts) in variants {
        println!(
            "\n[{name}] {:>8}  {:>14}  {:>10}  {:>14}  {:>10}  {:>6}  {:>6}",
            "lambda", "min margin", "delta", "worst-row m", "w-delta", "rows+", "rows-"
        );
        println!(
            "[{name}] {:>8}  {:>+14.6}  {:>10}  {:>+14.6}  {:>10}  {:>6}  {:>6}",
            "baseline", base_min, "-", base[worst_row], "-", "-", "-"
        );
        for &lambda in &grid {
            set_resident_cut_fold(fold_entry(cuts, lambda));
            reset_resident_cut_fold_applied_count();
            let margins = run(&graph);
            assert!(
                resident_cut_fold_applied_count() >= 1,
                "λ={lambda}: fold was never applied"
            );
            let m = min_of(&margins);
            let delta = m - base_min;
            let w_delta = margins[worst_row] - base[worst_row];
            let rows_up = margins.iter().zip(&base).filter(|(a, b)| a > b).count();
            let rows_down = margins.iter().zip(&base).filter(|(a, b)| a < b).count();
            if delta > 0.0 && best.as_ref().is_none_or(|(_, _, d)| delta > *d) {
                best = Some((name.to_string(), lambda, delta));
            }
            println!(
                "[{name}] {lambda:>8}  {m:>+14.6}  {delta:>+10.6}  {:>+14.6}  \
                 {w_delta:>+10.6}  {rows_up:>6}  {rows_down:>6}",
                margins[worst_row]
            );
        }
    }

    println!(
        "\nC2b root-margin verdict (objective-signed groups, resident GPU lane): {}",
        match &best {
            Some((name, lambda, delta)) => format!(
                "SOME λ STRICTLY IMPROVES the min root margin (SUCCESS): best {name} λ={lambda} \
                 delta {delta:+.6}"
            ),
            None => "no λ improves the min root margin — signed groups ALSO loosen; \
                     root-level L1 cuts are dead on this net (measured, decisive)"
                .to_string(),
        }
    );

    // Cleanup: leave the registry dark again; the env guards above restore
    // the pre-test environment on drop.
    clear_resident_cut_fold();
}
