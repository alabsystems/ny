// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::branching::resolve_branching;
use super::load_preset;
use ny_propagate::BranchingHeuristic;
use std::path::Path;

#[test]
fn lsnc_relu_gpu_bab_sidecar_stays_on_cpu_and_keeps_input_split_3870() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let main = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let sidecar = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu_gpu_bab.yaml")).unwrap();

    assert_eq!(main.general.device.as_deref(), None);
    // device: wgpu intentionally NOT set — WGPU kernel dispatch overhead dominates
    // sub-millisecond CPU CROWN backward for small models like lsnc_relu (#3870)
    assert_eq!(sidecar.general.device.as_deref(), None);
    assert_eq!(sidecar.attack.pgd_order.as_deref(), Some("before"));
    assert_eq!(sidecar.attack.pgd_restarts, Some(100));
    assert_eq!(sidecar.bab.batch_size, Some(1_000_000));
    assert_eq!(sidecar.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(sidecar.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(sidecar.bab.branching.input_split.adv_check, Some(-1));
    assert_eq!(sidecar.bab.clip.relaxed, Some(true));
    assert_eq!(sidecar.bab.clip.relaxed_iterations, Some(20));

    let resolved = resolve_branching(&sidecar)
        .unwrap()
        .expect("lsnc_relu GPU-BaB sidecar should resolve a branching mode");
    assert!(matches!(resolved.heuristic, BranchingHeuristic::InputSplit));
    assert!(
        !resolved.use_relu_split,
        "input-split GPU-BaB sidecar should stay on graph input split"
    );
}

#[test]
fn cersyve_gpu_bab_sidecar_stays_on_cpu_and_keeps_complete_clip_3870() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let main = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    let sidecar = load_preset(&repo_root.join("configs/vnncomp25/cersyve_gpu_bab.yaml")).unwrap();

    assert_eq!(main.general.device.as_deref(), None);
    // device: wgpu intentionally NOT set — WGPU dispatch overhead dominates for
    // small input-split models (#3870)
    assert_eq!(sidecar.general.device.as_deref(), None);
    assert_eq!(sidecar.attack.pgd_order.as_deref(), Some("before"));
    assert_eq!(sidecar.attack.pgd_restarts, Some(100));
    assert_eq!(sidecar.bab.batch_size, Some(500_000));
    assert_eq!(sidecar.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(sidecar.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(sidecar.bab.branching.input_split.adv_check, Some(-1));
    assert_eq!(sidecar.bab.clip.relaxed, Some(true));
    assert_eq!(sidecar.bab.clip.relaxed_iterations, Some(20));
    assert_eq!(sidecar.bab.clip.clip_type.as_deref(), Some("complete"));

    let resolved = resolve_branching(&sidecar)
        .unwrap()
        .expect("cersyve GPU-BaB sidecar should resolve a branching mode");
    assert!(matches!(resolved.heuristic, BranchingHeuristic::InputSplit));
    assert!(
        !resolved.use_relu_split,
        "input-split GPU-BaB sidecar should stay on graph input split"
    );
}
