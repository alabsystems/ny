// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::Duration;

use super::build_batch_size::{
    auto_build_batch_size_override_4354, maybe_apply_build_batch_size_autotune_4354,
};
use super::dispatch::{run_bab_with_fallback, DispatchContext};
use super::ibp_check::ibp_check_vnnlib_safe;
use super::routing::{
    auto_backend_default, resolve_beta_crown_backend, route_conv_model_to_graph,
    AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS,
};
use super::BetaCrownModel;
use super::{
    apply_instance_overrides, apply_vgg_abcrown_bound_mode, count_perturbed_inputs,
    effective_interm_transfer, effective_upfront_pgd, effective_upfront_pgd_with_vgg,
    resolve_pgd_attack, resolve_vgg_abcrown_decision, root_post_c_survivor_enabled_from_value,
    BetaCrownInstanceOverrides,
};
use crate::preset::{apply_preset, load_preset, PresetConfig};
use crate::{BackendArg, CompleteVerifierArg, MipSolverArg};
use ndarray::{arr1, arr2};
use ny_onnx::{
    vnnlib::{parse_vnnlib, OutputConstraint, VnnLibSpec},
    OnnxLoadConfig,
};
use ny_propagate::{
    layers::{LinearLayer, ReLULayer},
    BetaCrownConfig, BranchingHeuristic, GraphDomainBatchRecord, GraphNetwork, GraphNode,
    InputSplitBatchRecord, Layer,
};
use ny_tensor::BoundedTensor;
use tempfile::tempdir;

/// Helper to build a VnnLibSpec with a single conjunctive clause.
fn spec_with_clause(constraints: Vec<OutputConstraint>) -> VnnLibSpec {
    VnnLibSpec {
        num_inputs: 1,
        num_outputs: 2,
        input_bounds: vec![(0.0, 1.0)],
        output_constraints: constraints.clone(),
        output_constraint_clauses: vec![constraints],
        is_disjunction: false,
        version: None,
        per_clause_input_bounds: vec![Default::default()],
        declared_input_bounds: Vec::new(),
        dual_network: None,
    }
}

#[test]
fn typed_instance_override_arms_only_sparse_root_crown() {
    let mut config = BetaCrownConfig::default();
    let baseline = serde_json::to_value(&config).unwrap();
    apply_instance_overrides(&mut config, BetaCrownInstanceOverrides::default());
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        baseline,
        "default interactive route must be unchanged"
    );

    apply_instance_overrides(
        &mut config,
        BetaCrownInstanceOverrides {
            root_sparse_interm_crown: true,
        },
    );
    assert!(config.root_sparse_interm_crown);
    assert_eq!(config.root_sparse_interm_crown_max_secs, 2);
    assert_eq!(config.root_sparse_interm_crown_max_dim, 8_192);
    assert_eq!(config.root_sparse_interm_crown_max_rows, 512);
    assert_eq!(config.root_sparse_interm_crown_max_targets, 4);
}

#[test]
fn post_c_survivor_env_requires_the_exact_literal_one() {
    assert!(root_post_c_survivor_enabled_from_value(Some("1")));
    for value in [None, Some("0"), Some("true"), Some("01"), Some(" 1")] {
        assert!(
            !root_post_c_survivor_enabled_from_value(value),
            "only NY_ROOT_POST_C_SURVIVOR=1 may arm Stage B; got {value:?}"
        );
    }
}

#[test]
fn test_auto_build_batch_size_override_matches_nn4sys_thresholds_4354() {
    assert_eq!(
        auto_build_batch_size_override_4354(10_000_001, 1_001, true, true),
        Some(1_000),
        "large graph input-split runs should receive the alpha-beta-CROWN build_batch_size cap"
    );
    assert_eq!(
        auto_build_batch_size_override_4354(10_000_000, 1_001, true, true),
        None,
        "the parameter threshold is strict to mirror the reference tuning rule"
    );
    assert_eq!(
        auto_build_batch_size_override_4354(10_000_001, 1_000, true, true),
        None,
        "the spec threshold is strict to mirror the reference tuning rule"
    );
    assert_eq!(
        auto_build_batch_size_override_4354(10_000_001, 1_001, true, false),
        None,
        "non-input-split runs must not inherit the nn4sys-specific build_batch_size cap"
    );
}

#[test]
fn test_maybe_apply_build_batch_size_autotune_preserves_existing_override_4354() {
    let mut config = BetaCrownConfig {
        build_batch_size: Some(256),
        ..Default::default()
    };
    let spec = VnnLibSpec {
        num_inputs: 1,
        num_outputs: 1,
        input_bounds: vec![(0.0, 1.0)],
        output_constraints: vec![OutputConstraint::LessEqConst(0, 0.0); 1_001],
        output_constraint_clauses: vec![vec![OutputConstraint::LessEqConst(0, 0.0); 1_001]],
        is_disjunction: false,
        version: None,
        per_clause_input_bounds: vec![Default::default()],
        declared_input_bounds: Vec::new(),
        dual_network: None,
    };

    maybe_apply_build_batch_size_autotune_4354(&mut config, 20_000_000, Some(&spec), true, true);

    assert_eq!(
        config.build_batch_size,
        Some(256),
        "auto-tuning must not overwrite an explicit preset/CLI build_batch_size"
    );
}

fn build_graph_input_split_disjunction_fixture_4357(
) -> (BetaCrownModel, BoundedTensor, VnnLibSpec, BetaCrownConfig) {
    let linear1 = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear");
    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        Some(arr1(&[0.5_f32, 0.5_f32])),
    )
    .expect("anti-correlated output linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");

    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (or
    (<= Y_0 0.55)
    (<= Y_1 0.55)
))
"#,
    )
    .expect("valid disjunction spec");

    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 2,
        batch_size: 1,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    };

    (
        BetaCrownModel::Graph(Box::new(graph)),
        input,
        vnnlib,
        config,
    )
}

#[test]
fn test_resolve_beta_crown_backend_uses_preset_when_backend_omitted() {
    // Preset `device: wgpu` wins over the auto default (here a small input that
    // would otherwise auto-pick CPU). No auto-reason returned (preset decided).
    let (backend, reason) = resolve_beta_crown_backend(None, false, Some("wgpu"), Some(5), true);
    assert_eq!(backend, BackendArg::Wgpu);
    assert!(reason.is_none());
}

#[test]
fn test_resolve_beta_crown_backend_gpu_overrides_preset_when_backend_omitted() {
    let (backend, reason) = resolve_beta_crown_backend(None, true, Some("cpu"), Some(5), true);
    assert_eq!(backend, BackendArg::Wgpu);
    assert!(reason.is_none());
}

#[test]
fn test_resolve_beta_crown_backend_explicit_cpu_overrides_preset() {
    // Explicit --backend cpu wins even with a large input that would auto-pick GPU.
    let (backend, reason) = resolve_beta_crown_backend(
        Some(BackendArg::Cpu),
        false,
        Some("wgpu"),
        Some(150528),
        true,
    );
    assert_eq!(backend, BackendArg::Cpu);
    assert!(reason.is_none());
}

#[test]
fn test_resolve_beta_crown_backend_explicit_cpu_overrides_gpu() {
    let (backend, reason) = resolve_beta_crown_backend(
        Some(BackendArg::Cpu),
        true,
        Some("wgpu"),
        Some(150528),
        true,
    );
    assert_eq!(backend, BackendArg::Cpu);
    assert!(reason.is_none());
}

#[test]
fn test_resolve_beta_crown_backend_explicit_wgpu_overrides_small_input() {
    // Explicit --backend wgpu wins even for a tiny input that would auto-pick CPU.
    let (backend, reason) =
        resolve_beta_crown_backend(Some(BackendArg::Wgpu), false, None, Some(5), true);
    assert_eq!(backend, BackendArg::Wgpu);
    assert!(reason.is_none());
}

#[test]
fn test_auto_backend_default_large_input_picks_gpu_when_available() {
    // cifar100 (3072), traffic_signs (12288), yolo (8112), vggnet16 (150528) ...
    for count in [3072usize, 8112, 9408, 12288, 150528] {
        let (backend, _reason) = auto_backend_default(Some(count), true);
        assert_eq!(backend, BackendArg::Wgpu, "input {count} should pick GPU");
    }
}

#[test]
fn test_auto_backend_default_large_input_falls_back_cpu_without_gpu() {
    // Large input but no GPU compiled/available → CPU (still sound, just slower).
    let (backend, _reason) = auto_backend_default(Some(150528), false);
    assert_eq!(backend, BackendArg::Cpu);
}

#[test]
fn test_auto_backend_default_small_input_picks_cpu_even_with_gpu() {
    // acasxu (5), cersyve (4), sat_relu (30), collins_rul (400) → CPU.
    for count in [4usize, 5, 30, 400, 1000] {
        let (backend, _reason) = auto_backend_default(Some(count), true);
        assert_eq!(backend, BackendArg::Cpu, "input {count} should pick CPU");
    }
}

#[test]
fn test_auto_backend_default_threshold_is_strict_greater_than_1000() {
    // Exactly 1000 stays CPU; 1001 flips to GPU when available.
    assert_eq!(
        auto_backend_default(Some(AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS), true).0,
        BackendArg::Cpu
    );
    assert_eq!(
        auto_backend_default(Some(AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS + 1), true).0,
        BackendArg::Wgpu
    );
}

#[test]
fn test_auto_backend_default_unknown_input_picks_cpu() {
    // No VNN-LIB spec to size from (epsilon-ball mode) → conservative CPU default.
    let (backend, _reason) = auto_backend_default(None, true);
    assert_eq!(backend, BackendArg::Cpu);
}

#[test]
fn test_resolve_beta_crown_backend_auto_default_reports_reason() {
    // No explicit/legacy/preset signal → AUTO path returns Some(reason).
    let (backend, reason) = resolve_beta_crown_backend(None, false, None, Some(150528), true);
    assert_eq!(backend, BackendArg::Wgpu);
    assert!(reason.is_some());

    let (backend, reason) = resolve_beta_crown_backend(None, false, None, Some(5), true);
    assert_eq!(backend, BackendArg::Cpu);
    assert!(reason.is_some());
}

#[test]
fn test_gpu_available_hint_does_not_force_small_input_to_gpu_vnncomp_routing() {
    // Regression (#vnncomp-gpu-routing): the scored VNN-COMP entry derives a GPU
    // CAPABILITY HINT from `GPU_AVAILABLE` and feeds it as `gpu_available` (the size
    // gate), NOT as the legacy `gpu` FORCE. With the force OFF, a small ACAS input
    // (5 elements) on a GPU box must still route to the CPU input-split BaB — this
    // is the verification that was lost when `GPU_AVAILABLE` forced wgpu and turned
    // ~12-14s CPU `unsat`s into `unknown` timeouts (prop_4 net_1_1, prop_3 net_1_2).
    let (backend, reason) = resolve_beta_crown_backend(
        None,    // backend: auto
        false,   // gpu: NO legacy force (the fix)
        None,    // preset_device
        Some(5), // ACAS 5-input net
        true,    // gpu_available: capability hint = GPU present
    );
    assert_eq!(
        backend,
        BackendArg::Cpu,
        "ACAS must stay on CPU input-split BaB"
    );
    assert!(
        reason.is_some(),
        "AUTO size-gate decision should report a reason"
    );

    // Same hint, but a LARGE conv-dominated input still routes to GPU: the hint is
    // genuinely used where it helps — GPU is NOT disabled, only un-forced.
    let (backend, _) = resolve_beta_crown_backend(None, false, None, Some(150_528), true);
    assert_eq!(backend, BackendArg::Wgpu, "large conv still uses GPU");

    // Contrast: the EXPLICIT human `--gpu` force (a distinct call site) is preserved
    // and still overrides the size gate even for a tiny input.
    let (backend, reason) = resolve_beta_crown_backend(None, true, None, Some(5), true);
    assert_eq!(
        backend,
        BackendArg::Wgpu,
        "explicit --gpu still forces wgpu"
    );
    assert!(
        reason.is_none(),
        "an explicit force is not an AUTO decision"
    );
}

#[test]
fn test_resolve_beta_crown_backend_explicit_cpu_preset_pins_cpu() {
    // Preset `device: cpu` explicitly pins CPU (not AUTO) even for a large input.
    let (backend, reason) =
        resolve_beta_crown_backend(None, false, Some("cpu"), Some(150528), true);
    assert_eq!(backend, BackendArg::Cpu);
    assert!(reason.is_none());
}

#[test]
fn test_route_conv_model_to_graph_forces_matrix_mode_off_sequential_kfsb_3813() {
    let (use_graph, use_relu_split) = route_conv_model_to_graph(
        true,
        CompleteVerifierArg::Bab,
        false,
        false,
        true,
        false,
        false,
    );

    assert!(
        use_graph,
        "#3813: matrix conv_mode should not stay on the sequential Conv2d path"
    );
    assert!(
        use_relu_split,
        "#3813: matrix conv_mode should promote Conv2d models to graph ReLU splitting"
    );
}

#[test]
fn test_route_conv_model_to_graph_keeps_sequential_kfsb_in_patches_mode_3813() {
    let (use_graph, use_relu_split) = route_conv_model_to_graph(
        true,
        CompleteVerifierArg::Bab,
        false,
        false,
        true,
        false,
        true,
    );

    assert!(
        !use_graph,
        "#3813: sequential-only kfsb should stay off the graph path when patches mode remains active"
    );
    assert!(
        !use_relu_split,
        "#3813: patches-mode sequential kfsb should not be rewritten to relu split"
    );
}

#[test]
fn test_route_conv_model_to_graph_preserves_input_split_under_matrix_mode_3813() {
    let (use_graph, use_relu_split) = route_conv_model_to_graph(
        true,
        CompleteVerifierArg::Bab,
        false,
        true,
        true,
        false,
        false,
    );

    assert!(use_graph);
    assert!(
        !use_relu_split,
        "#3813: explicit graph-compatible input splitting should survive matrix-mode routing"
    );
}

#[test]
fn test_run_bab_with_fallback_writes_input_split_metrics_jsonl_4357() {
    let (mut model_net, input, vnnlib, config) = build_graph_input_split_disjunction_fixture_4357();
    let output_dir = tempdir().expect("tempdir");
    let metrics_path = output_dir.path().join("input_split_metrics.jsonl");
    let property = None;
    let onnx_load_config = OnnxLoadConfig::default();
    let proof_opts = super::ProofOpts::default();
    let mut ctx = DispatchContext {
        model_path: Path::new("unused.onnx"),
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        vnnlib_spec: Some(&vnnlib),
        property: &property,
        epsilon: 0.0,
        effective_threshold: 0.0,
        verify_upper: false,
        output_dim: 2,
        const_output_idx: None,
        has_relational: false,
        use_relu_split: false,
        gpu_bab: false,
        run_upfront_pgd: false,
        gemm_engine: None,
        compute_device: None,
        allow_heuristic_logsoftmax: false,
        allow_heuristic_softmax: false,
        input_split_metrics_jsonl: Some(metrics_path.as_path()),
        domain_batch_metrics_jsonl: None,
        proof_opts: &proof_opts,
        json: true,
        sigmoid_peeled: false,
    };

    run_bab_with_fallback(&mut ctx, CompleteVerifierArg::Bab, MipSolverArg::AY)
        .expect("direct CLI dispatch should complete");

    let contents = std::fs::read_to_string(&metrics_path).expect("sidecar should exist");
    let lines: Vec<_> = contents.lines().collect();
    assert!(
        !lines.is_empty(),
        "disjunctive input-split CLI run should emit at least one metrics record"
    );

    for line in lines {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("sidecar line should be valid JSON");
        assert_eq!(
            value["schema_version"],
            InputSplitBatchRecord::schema_version()
        );
        assert_eq!(value["record_kind"], InputSplitBatchRecord::record_kind());
    }
}

#[test]
fn test_run_bab_with_fallback_writes_domain_batch_metrics_jsonl_4398() {
    let (mut model_net, input, vnnlib, config) = build_graph_input_split_disjunction_fixture_4357();
    let output_dir = tempdir().expect("tempdir");
    let metrics_path = output_dir.path().join("graph_domain_batch_metrics.jsonl");
    let property = None;
    let onnx_load_config = OnnxLoadConfig::default();
    let proof_opts = super::ProofOpts::default();
    let mut ctx = DispatchContext {
        model_path: Path::new("unused.onnx"),
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        vnnlib_spec: Some(&vnnlib),
        property: &property,
        epsilon: 0.0,
        effective_threshold: 0.0,
        verify_upper: false,
        output_dim: 2,
        const_output_idx: None,
        has_relational: false,
        use_relu_split: false,
        gpu_bab: false,
        run_upfront_pgd: false,
        gemm_engine: None,
        compute_device: None,
        allow_heuristic_logsoftmax: false,
        allow_heuristic_softmax: false,
        input_split_metrics_jsonl: None,
        domain_batch_metrics_jsonl: Some(metrics_path.as_path()),
        proof_opts: &proof_opts,
        json: true,
        sigmoid_peeled: false,
    };

    run_bab_with_fallback(&mut ctx, CompleteVerifierArg::Bab, MipSolverArg::AY)
        .expect("direct CLI dispatch should complete");

    let contents = std::fs::read_to_string(&metrics_path).expect("sidecar should exist");
    let lines: Vec<_> = contents.lines().collect();
    assert!(
        !lines.is_empty(),
        "direct CLI run should emit at least one shared domain-batch record"
    );

    for line in lines {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("sidecar line should be valid JSON");
        assert_eq!(
            value["schema_version"],
            GraphDomainBatchRecord::schema_version()
        );
        assert_eq!(value["record_kind"], GraphDomainBatchRecord::record_kind());
        assert_eq!(value["caller_lane"], "input_split_dense_spec");
    }
}

#[test]
fn test_effective_interm_transfer_defaults_to_enabled_without_preset_4358() {
    assert!(
        effective_interm_transfer(false, None),
        "#4358: CLI status should report the enabled default when no preset override is present"
    );
    assert!(
        effective_interm_transfer(true, None),
        "#4358: explicit CLI enable should stay enabled without a preset"
    );
}

#[test]
fn test_effective_upfront_pgd_skips_input_bab_schedule_4354() {
    let mut preset = PresetConfig::default();
    preset.attack.pgd_order = Some("input_bab".to_string());

    assert!(
        resolve_pgd_attack(true, Some(&preset)),
        "#4354: input_bab should still count as PGD-enabled for the verifier"
    );
    assert!(
        !effective_upfront_pgd(true, Some(&preset)),
        "#4354: input_bab should skip the global upfront PGD stage even with PGD enabled"
    );
    assert!(
        !effective_upfront_pgd(false, Some(&preset)),
        "#4354: the upfront PGD stage never runs with PGD disabled"
    );
}

#[test]
fn vgg_abcrown_policy_matches_upstream_strict_thresholds() {
    let mut preset = PresetConfig::default();
    preset.model.vgg_abcrown_treatment = Some(true);
    preset.attack.pgd_order = Some("input_bab".to_string());

    let at_100 = resolve_vgg_abcrown_decision(Some(&preset), Some(100)).unwrap();
    assert_eq!(
        at_100.rewrite_mode,
        ny_propagate::VggMaxPoolRewriteMode::Sequential
    );
    assert!(at_100.use_forward_bounds);
    assert!(!at_100.prioritize_attack);

    let above_100 = resolve_vgg_abcrown_decision(Some(&preset), Some(101)).unwrap();
    assert_eq!(
        above_100.rewrite_mode,
        ny_propagate::VggMaxPoolRewriteMode::Residual
    );
    assert!(!above_100.use_forward_bounds);
    assert!(!above_100.prioritize_attack);

    let at_10k = resolve_vgg_abcrown_decision(Some(&preset), Some(10_000)).unwrap();
    assert!(!at_10k.prioritize_attack);
    assert!(!effective_upfront_pgd_with_vgg(
        true,
        Some(&preset),
        Some(at_10k)
    ));

    let above_10k = resolve_vgg_abcrown_decision(Some(&preset), Some(10_001)).unwrap();
    assert!(above_10k.prioritize_attack);
    assert!(effective_upfront_pgd_with_vgg(
        true,
        Some(&preset),
        Some(above_10k)
    ));
    assert!(
        !effective_upfront_pgd_with_vgg(false, Some(&preset), Some(above_10k)),
        "--no-pgd-attack remains the highest-precedence disable"
    );
}

#[test]
fn vgg_abcrown_gate_off_is_a_complete_policy_noop() {
    let preset = PresetConfig::default();
    assert_eq!(
        resolve_vgg_abcrown_decision(Some(&preset), Some(150_528)),
        None
    );
    assert_eq!(resolve_vgg_abcrown_decision(None, Some(150_528)), None);

    let mut config = BetaCrownConfig::default();
    let before = serde_json::to_value(&config).unwrap();
    apply_vgg_abcrown_bound_mode(&mut config, None);
    assert_eq!(serde_json::to_value(&config).unwrap(), before);
}

#[test]
fn vgg_abcrown_policy_counts_only_nonzero_width_inputs_and_sets_bound_mode() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = vec![(0.0, 0.0), (-1.0, 1.0), (2.0, 2.0), (3.0, 3.5)];
    assert_eq!(count_perturbed_inputs(&spec), 2);

    let mut preset = PresetConfig::default();
    preset.model.vgg_abcrown_treatment = Some(true);
    let forward = resolve_vgg_abcrown_decision(Some(&preset), Some(2));
    let mut config = BetaCrownConfig::default();
    apply_vgg_abcrown_bound_mode(&mut config, forward);
    assert!(!config.use_alpha_crown);
    assert!(config.use_forward_bounds);

    let crown = resolve_vgg_abcrown_decision(Some(&preset), Some(101));
    apply_vgg_abcrown_bound_mode(&mut config, crown);
    assert!(!config.use_alpha_crown);
    assert!(!config.use_forward_bounds);
}

#[test]
fn test_effective_upfront_pgd_treats_skip_as_disabled_4354() {
    let mut preset = PresetConfig::default();
    preset.attack.pgd_order = Some("skip".to_string());

    assert!(
        !resolve_pgd_attack(true, Some(&preset)),
        "#4354: pgd_order=skip must win over the default-on --pgd-attack value \
         (true cannot signal an explicit CLI enable)"
    );
    assert!(
        !resolve_pgd_attack(false, Some(&preset)),
        "#4354: --no-pgd-attack keeps PGD disabled under a skip preset"
    );
    assert!(
        !effective_upfront_pgd(resolve_pgd_attack(true, Some(&preset)), Some(&preset)),
        "#4354: skip should leave the upfront PGD stage disabled"
    );
}

#[test]
fn test_resolve_pgd_attack_defaults_on_without_preset_signal() {
    assert!(
        resolve_pgd_attack(true, None),
        "no preset: the default-on CLI value keeps PGD enabled"
    );
    assert!(
        !resolve_pgd_attack(false, None),
        "--no-pgd-attack disables PGD without a preset"
    );

    let preset = PresetConfig::default();
    assert!(
        resolve_pgd_attack(true, Some(&preset)),
        "a preset without pgd_order keeps the default-on behavior"
    );
}

#[test]
fn test_resolve_pgd_attack_no_pgd_attack_wins_over_preset_enable() {
    let mut preset = PresetConfig::default();
    preset.attack.pgd_order = Some("before".to_string());

    assert!(
        resolve_pgd_attack(true, Some(&preset)),
        "a PGD-enabling preset keeps PGD on"
    );
    assert!(
        !resolve_pgd_attack(false, Some(&preset)),
        "--no-pgd-attack is always explicit and wins over a PGD-enabling preset"
    );
}

#[test]
fn test_apply_preset_marks_nn4sys_skip_schedule_disabled_4354() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();

    assert_eq!(preset.attack.pgd_order.as_deref(), Some("skip"));
    assert_eq!(preset.attack.pgd_restarts, Some(10));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    assert!(
        !config.enable_pgd_attack,
        "apply_preset should decode nn4sys pgd_order=skip as preset-driven PGD disabled"
    );
}

#[test]
fn test_effective_interm_transfer_preserves_explicit_false_preset_4358() {
    let mut preset = PresetConfig::default();
    preset.bab.interm_transfer = Some(false);

    assert!(
        !effective_interm_transfer(false, Some(&preset)),
        "#4358: explicit preset false must override the enabled default in CLI status output"
    );
    assert!(
        effective_interm_transfer(true, Some(&preset)),
        "#4358: explicit CLI enable should still win over a preset false override"
    );
}

/// Legacy helper superseded by `PhaseBudgetLedger::remaining()` (#2206 Packet B).
fn remaining_timeout_after_pgd(total_timeout: Duration, pgd_elapsed: Duration) -> Option<u64> {
    total_timeout
        .checked_sub(pgd_elapsed)
        .map(|remaining| remaining.as_secs())
        .filter(|remaining_secs| *remaining_secs > 0)
}

#[test]
fn test_remaining_timeout_after_pgd_subtracts_elapsed_3779() {
    let total = Duration::from_secs(20);
    let pgd_elapsed = Duration::from_secs(7);
    assert_eq!(remaining_timeout_after_pgd(total, pgd_elapsed), Some(13));
}

#[test]
fn test_remaining_timeout_after_pgd_exhausted_returns_none_3779() {
    let total = Duration::from_secs(20);
    let pgd_elapsed = Duration::from_secs(20);
    assert_eq!(remaining_timeout_after_pgd(total, pgd_elapsed), None);
}

#[test]
fn test_ibp_check_less_than_boundary_refuted() {
    // lower[0] == upper[1] exactly → LessThan(0,1) (Y_0 < Y_1) is impossible
    // because Y_0 >= lower[0] = 5.0 = upper[1] >= Y_1, so Y_0 >= Y_1.
    let ibp_lower = &[5.0_f32, 0.0];
    let ibp_upper = &[10.0_f32, 5.0];
    let spec = spec_with_clause(vec![OutputConstraint::LessThan(0, 1)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_less_eq_boundary_not_refuted() {
    // lower[0] == upper[1] exactly → LessEq(0,1) (Y_0 <= Y_1) is NOT refuted
    // because Y_0 = 5.0 = Y_1 satisfies Y_0 <= Y_1.
    let ibp_lower = &[5.0_f32, 0.0];
    let ibp_upper = &[10.0_f32, 5.0];
    let spec = spec_with_clause(vec![OutputConstraint::LessEq(0, 1)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_than_boundary_refuted() {
    // upper[0] == lower[1] exactly → GreaterThan(0,1) (Y_0 > Y_1) is impossible
    // because Y_0 <= upper[0] = 5.0 = lower[1] <= Y_1, so Y_0 <= Y_1.
    let ibp_lower = &[0.0_f32, 5.0];
    let ibp_upper = &[5.0_f32, 10.0];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterThan(0, 1)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_eq_boundary_not_refuted() {
    // upper[0] == lower[1] exactly → GreaterEq(0,1) (Y_0 >= Y_1) is NOT refuted
    let ibp_lower = &[0.0_f32, 5.0];
    let ibp_upper = &[5.0_f32, 10.0];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterEq(0, 1)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_less_than_const_refuted_above_threshold() {
    // lower[0] well above const → LessThanConst(0, 5.0) (Y_0 < 5.0) is impossible
    let ibp_lower = &[6.0_f32];
    let ibp_upper = &[10.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::LessThanConst(0, 5.0)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_less_than_const_boundary_conservative() {
    // lower[0] = 5.0 exactly = const → with directed rounding (next_up), the
    // check requires lower[0] >= next_up(5.0), so this is conservatively NOT
    // refuted. Sound: we don't falsely claim safe at the f32 boundary.
    let ibp_lower = &[5.0_f32];
    let ibp_upper = &[10.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::LessThanConst(0, 5.0)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_less_eq_const_boundary_not_refuted() {
    // lower[0] = 5.0, const = 5.0 → LessEqConst(0, 5.0) (Y_0 <= 5.0) not refuted
    let ibp_lower = &[5.0_f32];
    let ibp_upper = &[10.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::LessEqConst(0, 5.0)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_than_const_refuted_below_threshold() {
    // upper[0] well below const → GreaterThanConst(0, 5.0) (Y_0 > 5.0) is impossible
    let ibp_lower = &[0.0_f32];
    let ibp_upper = &[4.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterThanConst(0, 5.0)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_than_const_boundary_conservative() {
    // upper[0] = 5.0 exactly = const → with directed rounding (next_down), the
    // check requires upper[0] <= next_down(5.0), so this is conservatively NOT
    // refuted. Sound: we don't falsely claim safe at the f32 boundary.
    let ibp_lower = &[0.0_f32];
    let ibp_upper = &[5.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterThanConst(0, 5.0)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_eq_const_boundary_not_refuted() {
    // upper[0] = 5.0, const = 5.0 → GreaterEqConst(0, 5.0) (Y_0 >= 5.0) not refuted
    let ibp_lower = &[0.0_f32];
    let ibp_upper = &[5.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterEqConst(0, 5.0)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

// --- gpu_bab auto-promotion tests (#4406) ---

#[test]
fn test_auto_promote_gpu_bab_graph_bound_impact_single_objective_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        should_auto_promote_gpu_bab(false, true, &BranchingHeuristic::BoundImpact, Some(&spec)),
        "graph + BoundImpact + single-objective should auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_graph_input_split_single_objective_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        should_auto_promote_gpu_bab(false, true, &BranchingHeuristic::InputSplit, Some(&spec)),
        "graph + InputSplit + single-objective should auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_sequential_model_does_not_promote_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        !should_auto_promote_gpu_bab(false, false, &BranchingHeuristic::BoundImpact, Some(&spec)),
        "sequential model should NOT auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_unsupported_heuristic_does_not_promote_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        !should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::FilteredSmartBranching,
            Some(&spec)
        ),
        "unsupported heuristic (FilteredSmartBranching) should NOT auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_disjunctive_multi_clause_excluded_4409() {
    use super::should_auto_promote_gpu_bab;
    let mut spec = VnnLibSpec::new();
    spec.is_disjunction = true;
    spec.output_constraint_clauses = vec![
        vec![OutputConstraint::LessThanConst(0, 1.0)],
        vec![OutputConstraint::LessThanConst(1, 2.0)],
    ];
    assert!(
        !should_auto_promote_gpu_bab(false, true, &BranchingHeuristic::BoundImpact, Some(&spec)),
        "multi-clause disjunctive specs should NOT auto-promote (#4409)"
    );
}

#[test]
fn test_auto_promote_gpu_bab_explicit_cli_flag_always_wins_4406() {
    use super::should_auto_promote_gpu_bab;
    let mut spec = VnnLibSpec::new();
    spec.is_disjunction = true;
    spec.output_constraint_clauses = vec![
        vec![OutputConstraint::LessThanConst(0, 1.0)],
        vec![OutputConstraint::LessThanConst(1, 2.0)],
    ];
    assert!(
        should_auto_promote_gpu_bab(true, true, &BranchingHeuristic::BoundImpact, Some(&spec)),
        "explicit --gpu-bab CLI flag should always enable, even on disjunctive specs"
    );
}

#[test]
fn test_auto_promote_gpu_bab_no_spec_promotes_4406() {
    use super::should_auto_promote_gpu_bab;
    assert!(
        should_auto_promote_gpu_bab(false, true, &BranchingHeuristic::BoundImpact, None),
        "graph + BoundImpact + no spec should auto-promote (epsilon-ball, no disjunction)"
    );
}

/// Preset-cuts clobber regression: a preset's `bab.cuts.enabled: true` must
/// survive config construction when the CLI did not pass `--enable-cuts` —
/// the `ny vnncomp` entry point always passes enable_cuts=false, and the old
/// unconditional `config.enable_cuts = enable_cuts_effective` silently
/// disabled preset cuts on every supported model class.
#[test]
fn resolve_enable_cuts_preserves_preset_when_cli_silent() {
    use super::resolve_enable_cuts;
    assert!(
        resolve_enable_cuts(true, false, false, true),
        "preset-enabled cuts must survive a silent CLI"
    );
    assert!(
        !resolve_enable_cuts(false, false, false, true),
        "no preset, no CLI: cuts stay off"
    );
}

#[test]
fn resolve_enable_cuts_cli_flags_win() {
    use super::resolve_enable_cuts;
    assert!(
        resolve_enable_cuts(false, true, false, true),
        "--enable-cuts enables without preset"
    );
    assert!(
        !resolve_enable_cuts(true, false, true, true),
        "--no-cuts disables preset-enabled cuts"
    );
    // main.rs collapses --enable-cuts + --no-cuts into cli_enable=false before
    // this point; the disable flag still wins.
    assert!(!resolve_enable_cuts(true, false, true, true));
}

#[test]
fn resolve_enable_cuts_unsupported_model_class_forces_off() {
    use super::resolve_enable_cuts;
    // Graph input splitting does not support cuts (#3813): preset and CLI both lose.
    assert!(!resolve_enable_cuts(true, false, false, false));
    assert!(!resolve_enable_cuts(false, true, false, false));
}

/// general.complete_verifier preset field (sat_relu/malbeware MIP routing):
/// parse the three legal values, reject junk, None when unset.
#[test]
fn resolve_preset_complete_verifier_parses_and_rejects() {
    use super::resolve_preset_complete_verifier;
    use crate::preset::PresetConfig;
    use crate::subcommands::CompleteVerifierArg;

    let mut p = PresetConfig::default();
    assert_eq!(resolve_preset_complete_verifier(Some(&p)).unwrap(), None);
    assert_eq!(resolve_preset_complete_verifier(None).unwrap(), None);

    p.general.complete_verifier = Some("mip".into());
    assert_eq!(
        resolve_preset_complete_verifier(Some(&p)).unwrap(),
        Some(CompleteVerifierArg::Mip)
    );
    p.general.complete_verifier = Some("bab".into());
    assert_eq!(
        resolve_preset_complete_verifier(Some(&p)).unwrap(),
        Some(CompleteVerifierArg::Bab)
    );
    p.general.complete_verifier = Some("auto".into());
    assert_eq!(
        resolve_preset_complete_verifier(Some(&p)).unwrap(),
        Some(CompleteVerifierArg::Auto)
    );
    p.general.complete_verifier = Some("gurobi".into());
    assert!(resolve_preset_complete_verifier(Some(&p)).is_err());
}

/// solver.alpha-crown.softmax preset field: "complex" opts into the graph
/// rewrite (also accepted under bab.alpha-crown), unknown values warn and keep
/// the default direct-LSE relaxation, None stays off.
#[test]
fn resolve_preset_softmax_complex_parses_and_falls_back() {
    use super::resolve_preset_softmax_complex;

    let mut p = PresetConfig::default();
    assert!(!resolve_preset_softmax_complex(None));
    assert!(!resolve_preset_softmax_complex(Some(&p)));

    p.solver.alpha_crown.softmax = Some("complex".into());
    assert!(resolve_preset_softmax_complex(Some(&p)));

    // Unknown values keep the sound default relaxation instead of erroring.
    p.solver.alpha_crown.softmax = Some("fancy".into());
    assert!(!resolve_preset_softmax_complex(Some(&p)));

    // bab.alpha-crown fallback location.
    p.solver.alpha_crown.softmax = None;
    p.bab.alpha_crown.softmax = Some("complex".into());
    assert!(resolve_preset_softmax_complex(Some(&p)));
}
