// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::{Duration, Instant};

use super::build_batch_size::{
    auto_build_batch_size_override_4354, maybe_apply_build_batch_size_autotune_4354,
};
use super::dispatch::{run_bab_with_fallback, DispatchContext};
use super::ibp_check::ibp_check_vnnlib_safe;
use super::output::EffectiveTreatmentProjection;
use super::routing::{
    auto_backend_default, explicitly_requests_wgpu, resolve_beta_crown_backend,
    resolve_beta_crown_backend_request, route_conv_model_to_graph,
    AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS,
};
use super::BetaCrownModel;
use super::{
    apply_instance_overrides, apply_vgg_abcrown_bound_mode, attack_steering_route,
    count_perturbed_inputs, effective_engine_phase_budget, effective_interm_transfer,
    effective_upfront_pgd, effective_upfront_pgd_with_vgg, require_deferred_pgd_consumer,
    resolve_deferred_pgd_owner, resolve_internal_authority_deadline, resolve_overall_deadline,
    resolve_pgd_attack, resolve_preset_config, resolve_vgg_abcrown_decision,
    root_post_c_survivor_enabled_from_value, terminal_peel_policy, AppliedTerminalPeel,
    AttackSteeringRoute, BetaCrownInstanceOverrides, BetaCrownPresetSnapshot, CganInputLeafRoute,
    DeferredPgdOwner, SharedAttackGemmOnly, TerminalPeelPolicy,
};
use crate::commands::backend::BackendRequestSource;
use crate::preset::{apply_preset, load_preset, PresetConfig};
use crate::{BackendArg, CompleteVerifierArg, MipSolverArg};
use ndarray::{arr1, arr2};
use ny_core::GemmEngine;
use ny_onnx::{
    vnnlib::{parse_vnnlib, OutputConstraint, VnnLibSpec},
    OnnxLoadConfig,
};
use ny_propagate::{
    layers::{LinearLayer, ReLULayer},
    BetaCrownConfig, BranchingHeuristic, GraphDomainBatchRecord, GraphNetwork, GraphNode,
    InputSplitBatchRecord, Layer, Network,
};
use ny_tensor::BoundedTensor;
use tempfile::tempdir;

/// Helper to build a VnnLibSpec with a single conjunctive clause.
fn spec_with_clause(constraints: Vec<OutputConstraint>) -> VnnLibSpec {
    let num_outputs = constraints
        .iter()
        .filter_map(|constraint| match constraint {
            OutputConstraint::LessEq(i, j)
            | OutputConstraint::LessThan(i, j)
            | OutputConstraint::GreaterEq(i, j)
            | OutputConstraint::GreaterThan(i, j) => Some((*i).max(*j) + 1),
            OutputConstraint::LessEqConst(i, _)
            | OutputConstraint::LessThanConst(i, _)
            | OutputConstraint::GreaterEqConst(i, _)
            | OutputConstraint::GreaterThanConst(i, _) => Some(*i + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    VnnLibSpec {
        num_inputs: 1,
        num_outputs,
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
    apply_instance_overrides(&mut config, &BetaCrownInstanceOverrides::default());
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        baseline,
        "default interactive route must be unchanged"
    );

    apply_instance_overrides(
        &mut config,
        &BetaCrownInstanceOverrides {
            root_sparse_interm_crown: true,
            ..BetaCrownInstanceOverrides::default()
        },
    );
    assert!(config.root_sparse_interm_crown);
    assert_eq!(config.root_sparse_interm_crown_max_secs, 2);
    assert_eq!(config.root_sparse_interm_crown_max_dim, 8_192);
    assert_eq!(config.root_sparse_interm_crown_max_rows, 512);
    assert_eq!(config.root_sparse_interm_crown_max_targets, 4);
}

#[test]
fn typed_cgan_route_arms_only_the_input_leaf_consult() {
    let mut config = BetaCrownConfig::default();
    let baseline = config.clone();
    apply_instance_overrides(
        &mut config,
        &BetaCrownInstanceOverrides {
            cgan_input_leaf_route: Some(CganInputLeafRoute::Cgan2023),
            ..BetaCrownInstanceOverrides::default()
        },
    );
    assert!(config.input_split_input_leaf_oracle);
    let mut expected = baseline;
    expected.input_split_input_leaf_oracle = true;
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&expected).unwrap(),
        "the typed category route must not change any other verifier policy"
    );
}

#[test]
fn typed_traffic_terminal_softmax_peel_request_reaches_loader_precedence() {
    let default_overrides = BetaCrownInstanceOverrides::default();
    assert_eq!(
        terminal_peel_policy(false, &default_overrides),
        TerminalPeelPolicy::Off
    );
    assert_eq!(
        terminal_peel_policy(true, &default_overrides),
        TerminalPeelPolicy::InteractiveLegacy
    );

    let traffic_overrides = BetaCrownInstanceOverrides {
        traffic_terminal_softmax_peel: true,
        ..BetaCrownInstanceOverrides::default()
    };
    assert_eq!(
        terminal_peel_policy(false, &traffic_overrides),
        TerminalPeelPolicy::TrafficSoftmaxSingleGroup
    );
    assert_eq!(
        terminal_peel_policy(true, &traffic_overrides),
        TerminalPeelPolicy::TrafficSoftmaxSingleGroup,
        "typed traffic policy must not broaden to the legacy activation surface"
    );
}

#[test]
fn applied_terminal_peel_receipt_preserves_exact_activation_kind() {
    for (layer_type, expected) in [
        (ny_core::LayerType::Softmax, AppliedTerminalPeel::Softmax),
        (
            ny_core::LayerType::LogSoftmax,
            AppliedTerminalPeel::LogSoftmax,
        ),
        (ny_core::LayerType::Sigmoid, AppliedTerminalPeel::Sigmoid),
    ] {
        let report = ny_onnx::PeelOffReport {
            peeled: true,
            layer_type: Some(layer_type),
            reason: None,
        };
        assert_eq!(AppliedTerminalPeel::from_report(&report).unwrap(), expected);
    }

    let declined = ny_onnx::PeelOffReport {
        peeled: false,
        layer_type: None,
        reason: Some("declined".to_string()),
    };
    assert_eq!(
        AppliedTerminalPeel::from_report(&declined).unwrap(),
        AppliedTerminalPeel::None
    );

    let malformed = ny_onnx::PeelOffReport {
        peeled: true,
        layer_type: None,
        reason: None,
    };
    assert!(AppliedTerminalPeel::from_report(&malformed).is_err());
}

#[test]
fn vnncomp_preset_snapshot_is_authoritative_over_later_path_changes() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("preset.yaml");
    std::fs::write(
        &path,
        "bab:\n  phase_budget:\n    post_bab_pgd_fraction: 0.0\n    \
         vnncomp_post_bab_attack: false\n",
    )
    .unwrap();
    let frozen = BetaCrownPresetSnapshot::load(&path);

    std::fs::write(
        &path,
        "bab:\n  phase_budget:\n    post_bab_pgd_fraction: 0.25\n    \
         vnncomp_post_bab_attack: true\n",
    )
    .unwrap();
    let resolved = resolve_preset_config(Some(&path), Some(&frozen))
        .unwrap()
        .expect("frozen loaded preset");
    assert_eq!(resolved.bab.phase_budget.post_bab_pgd_fraction, Some(0.0));
    assert_eq!(
        resolved.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false)
    );

    let interactive = resolve_preset_config(Some(&path), None)
        .unwrap()
        .expect("interactive path load");
    assert_eq!(
        interactive.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.25),
        "interactive calls retain path-based loading"
    );

    let invalid = BetaCrownPresetSnapshot::Invalid("frozen preset parse failure".into());
    std::fs::write(&path, "bab:\n  phase_budget: {}\n").unwrap();
    let error = resolve_preset_config(Some(&path), Some(&invalid))
        .expect_err("an invalid VNN-COMP snapshot must not be healed by rereading the path");
    assert!(error.to_string().contains("frozen preset parse failure"));
}

#[test]
fn authoritative_deadline_charges_setup_without_extending_inner_timeout() {
    let verification_start = Instant::now();
    let earlier_outer = verification_start
        .checked_sub(Duration::from_secs(7))
        .unwrap()
        + Duration::from_secs(20);
    let resolved = resolve_overall_deadline(
        verification_start,
        Duration::from_secs(19),
        Some(earlier_outer),
    )
    .unwrap();
    assert_eq!(resolved, Some(earlier_outer));

    let shorter_inner = resolve_overall_deadline(
        verification_start,
        Duration::from_secs(5),
        Some(verification_start + Duration::from_secs(20)),
    )
    .unwrap();
    assert_eq!(
        shorter_inner,
        Some(verification_start + Duration::from_secs(5))
    );
}

#[test]
fn authoritative_deadline_caps_unbounded_and_preserves_expiry() {
    let verification_start = Instant::now();
    let expired = verification_start
        .checked_sub(Duration::from_secs(1))
        .unwrap();
    assert_eq!(
        resolve_overall_deadline(verification_start, Duration::ZERO, Some(expired)).unwrap(),
        Some(expired)
    );
    assert_eq!(
        resolve_overall_deadline(verification_start, Duration::ZERO, None).unwrap(),
        None
    );
}

#[test]
fn outer_deferred_deadline_uses_the_frozen_absolute_authority_without_reanchoring() {
    let ingress = Instant::now();
    let configured = ingress + Duration::from_secs(100);
    let frozen = ingress + Duration::from_millis(71_250);
    assert_eq!(
        resolve_internal_authority_deadline(Some(configured), true, Some(frozen))
            .expect("frozen outer authority"),
        Some(frozen)
    );
    assert_eq!(
        frozen.saturating_duration_since(ingress + Duration::from_secs(20)),
        Duration::from_millis(51_250),
        "later setup completion must not slide the frozen cutoff"
    );

    let shorter = ingress + Duration::from_mins(1);
    assert_eq!(
        resolve_internal_authority_deadline(Some(shorter), true, Some(frozen))
            .expect("configured cap"),
        Some(shorter),
        "a deliberately shorter local timeout must still cap the outer authority"
    );
    assert_eq!(
        resolve_internal_authority_deadline(Some(configured), false, None)
            .expect("direct CLI/internal owner"),
        Some(configured),
        "non-outer routes retain their historical deadline"
    );
    assert!(resolve_internal_authority_deadline(Some(configured), true, None).is_err());

    let expired = ingress
        .checked_sub(Duration::from_secs(1))
        .expect("representable earlier instant");
    assert_eq!(
        resolve_internal_authority_deadline(Some(configured), true, Some(expired))
            .expect("expired frozen authority"),
        Some(expired),
        "an expired authority must never be extended to setup completion"
    );
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
fn beta_crown_backend_request_preserves_every_selection_source() {
    let explicit =
        resolve_beta_crown_backend_request(Some(BackendArg::Wgpu), false, None, Some(5), true);
    assert_eq!(explicit.backend, BackendArg::Wgpu);
    assert_eq!(explicit.source, BackendRequestSource::ExplicitBackend);
    assert!(explicit.selection_reason.is_none());

    let legacy = resolve_beta_crown_backend_request(None, true, Some("cpu"), Some(5), true);
    assert_eq!(legacy.backend, BackendArg::Wgpu);
    assert_eq!(legacy.source, BackendRequestSource::LegacyGpuFlag);
    assert!(legacy.selection_reason.is_none());

    let preset = resolve_beta_crown_backend_request(None, false, Some("wgpu"), Some(5), true);
    assert_eq!(preset.backend, BackendArg::Wgpu);
    assert_eq!(preset.source, BackendRequestSource::Preset);
    assert!(preset.selection_reason.is_none());

    let auto = resolve_beta_crown_backend_request(None, false, None, Some(150_528), true);
    assert_eq!(auto.backend, BackendArg::Wgpu);
    assert_eq!(auto.source, BackendRequestSource::Auto);
    assert!(auto.selection_reason.is_some());
}

#[test]
fn test_explicit_wgpu_probe_scope_excludes_auto_and_cpu_overrides() {
    assert!(explicitly_requests_wgpu(
        Some(BackendArg::Wgpu),
        false,
        Some("cpu")
    ));
    assert!(explicitly_requests_wgpu(None, true, Some("cpu")));
    assert!(explicitly_requests_wgpu(None, false, Some("wgpu")));

    assert!(!explicitly_requests_wgpu(
        Some(BackendArg::Cpu),
        true,
        Some("wgpu")
    ));
    assert!(!explicitly_requests_wgpu(None, false, None));
}

#[test]
fn test_auto_backend_default_large_input_picks_gpu_when_available() {
    // cifar100 (3072), traffic_signs (12288), yolo (8112), vggnet16 (150528) ...
    for count in [3072usize, 8112, 9408, 12288, 150528] {
        let (backend, _reason) = auto_backend_default(Some(count), true);
        assert_eq!(backend, BackendArg::Wgpu, "input {count} should pick GPU");
    }
}

/// #36 residual: a skipped WGPU probe (NVIDIA host) is a statement about the
/// PROOF backend, not about falsification steering. The shared CUDA engine
/// cannot carry `as_gpu_crown_backward`, so routing attacks there statically
/// declines both batched exact-VJP lanes (measured: soundnessbench model_0
/// timeout@145.4s vs sat@38.2s).
///
/// #attack-steering-segv RESOLVED: the fault was process EXIT (`_dl_fini`
/// tearing down the NVIDIA GL stack under the still-arming thread), not the
/// route. With the scored path ending via `_exit` the route is the default
/// again on a probe-skipped host, and `NY_ATTACK_STEERING_WGPU=0` is the
/// opt-OUT lever.
///
/// Env-free by construction: this test never sets the variable, so it pins the
/// answer for an unset environment (the scored configuration).
#[test]
fn test_attack_steering_takes_wgpu_route_when_probe_skipped() {
    // Probe skipped ⇒ still take the capability-bearing route: the shared CUDA
    // engine cannot carry `as_gpu_crown_backward`, and both batched exact-VJP
    // attack lanes statically decline without it.
    assert_eq!(
        attack_steering_route(BackendArg::Wgpu, false, true, true),
        AttackSteeringRoute::WgpuDevice
    );
    // Probe NOT skipped ⇒ unchanged.
    assert_eq!(
        attack_steering_route(BackendArg::Wgpu, false, false, true),
        AttackSteeringRoute::WgpuDevice
    );
    // An explicit CPU request disarms attack steering too.
    assert_eq!(
        attack_steering_route(BackendArg::Cpu, false, true, false),
        AttackSteeringRoute::Disabled
    );
}

/// A live proof-qualification refusal changes only the effective proof backend;
/// it must not erase the original WGPU request from verdict-neutral attack
/// routing.
#[test]
fn test_qualification_refusal_keeps_attack_steering_armed() {
    assert_eq!(
        attack_steering_route(BackendArg::Wgpu, false, true, true),
        AttackSteeringRoute::WgpuDevice,
        "a proof refusal must not erase the WGPU attack request"
    );
    assert_eq!(
        attack_steering_route(BackendArg::Wgpu, false, false, true),
        AttackSteeringRoute::WgpuDevice,
        "a proof refusal must retain the WGPU attack wrapper on probed hosts"
    );
}

/// A proof-only AUTO policy rewrite to CPU must consume the original WGPU
/// candidate when deciding verdict-neutral attack steering.
#[test]
fn test_auto_proof_cpu_rewrite_keeps_original_gpu_attack_intent() {
    let original_auto_candidate = BackendArg::Wgpu;
    let post_policy_proof_backend = BackendArg::Cpu;
    assert_eq!(post_policy_proof_backend, BackendArg::Cpu);
    assert_eq!(
        attack_steering_route(
            original_auto_candidate,
            false,
            false,
            original_auto_candidate == BackendArg::Wgpu,
        ),
        AttackSteeringRoute::WgpuDevice,
        "proof-only CPU policy must not disable verdict-neutral GPU steering"
    );
}

#[test]
fn qualified_proof_device_preempts_every_separate_wgpu_attack_route() {
    for (requested, probe_skipped, accelerator_requested) in [
        (BackendArg::Wgpu, false, true),
        (BackendArg::Wgpu, true, true),
        (BackendArg::Cpu, false, false),
    ] {
        assert_eq!(
            attack_steering_route(requested, true, probe_skipped, accelerator_requested),
            AttackSteeringRoute::ProofDevice,
            "one qualified proof context must be reused for attack steering"
        );
    }
}

struct PoisonGpuCapabilityEngine;

impl ny_core::GpuCrownBackward for PoisonGpuCapabilityEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[ny_core::GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        Err(ny_core::NyError::UnsupportedOp(
            "poison capability must stay unreachable".into(),
        ))
    }
}

impl GemmEngine for PoisonGpuCapabilityEngine {
    fn backend_provenance(&self) -> &'static str {
        "poison-gpu-capabilities"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Ok(vec![1.0])
    }

    fn gemm_f32_fast(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Ok(vec![2.0])
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
        Some(self)
    }
}

#[test]
fn test_shared_attack_gemm_only_masks_gpu_capabilities() {
    let inner: std::sync::Arc<dyn GemmEngine> = std::sync::Arc::new(PoisonGpuCapabilityEngine);
    assert!(inner.as_gpu_crown_backward().is_some());

    let steering = SharedAttackGemmOnly::new(inner);
    assert_eq!(steering.backend_provenance(), "poison-gpu-capabilities");
    assert_eq!(
        steering
            .gemm_f32(1, 1, 1, &[3.0], &[4.0])
            .expect("ordinary GEMM delegated"),
        vec![1.0]
    );
    assert_eq!(
        steering
            .gemm_f32_fast(1, 1, 1, &[3.0], &[4.0])
            .expect("fast attack GEMM delegated"),
        vec![2.0]
    );
    assert!(steering.as_gpu_crown_backward().is_none());
    assert!(steering.as_gpu_ibp_forward().is_none());
    assert!(steering.as_gpu_ibp_forward_ext().is_none());
    assert!(steering.as_gpu_dag_ibp_forward_ext().is_none());
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
fn late_sequential_conjunction_graph_route_excludes_reducible_same_lhs() {
    // CONTRACT, restored by `c93b4b59` ("restore the conjunctive gate — acasxu
    // prop_3/prop_4 solve again"): a same-LHS REDUCIBLE conjunction with no ReLU
    // splitting stays on the sequential lane and its per-domain CROWN-IBP
    // intermediates. Only ReLU splitting or a NON-reducible conjunction upgrades.
    //
    // This test previously asserted the opposite — it was written against the
    // window in which `faa66c38` had DELETED that exclusion, and `c93b4b59` never
    // updated it. It asserted `.expect("BaB should plan the late Graph route")`
    // on exactly the shape the gate now excludes on purpose.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.0_f32], [0.0], [0.0]]), None)
            .expect("three-output sequential fixture"),
    ));
    let model = BetaCrownModel::Sequential(Box::new(network));
    let incoming = BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: false,
        use_crown_ibp: true,
        input_split_ibp_enhancement: false,
        ..BetaCrownConfig::default()
    };

    // Reducible same-LHS (both rows share `Y_0`): the acasxu prop_2/3/4 shape.
    let reducible = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
"#,
    )
    .expect("same-LHS conjunction");
    assert!(
        super::planned_late_sequential_conjunction_graph_config(
            &model,
            &incoming,
            Some(&reducible),
            false,
            false,
        )
        .is_none(),
        "a reducible same-LHS conjunction must stay on the sequential lane"
    );

    // Non-reducible conjunction (chained LHS): this one DOES upgrade.
    let chained = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_1))
"#,
    )
    .expect("chained conjunction");
    let planned = super::planned_late_sequential_conjunction_graph_config(
        &model,
        &incoming,
        Some(&chained),
        false,
        false,
    )
    .expect("a non-reducible conjunction upgrades to the Graph route");
    assert!(
        super::dispatch::routes_to_relational_verifier(Some(&chained), false),
        "the shared dispatch predicate must classify this conjunction as relational"
    );

    // THE FORWARD-BOUNDS BOOTSTRAP IS NOW UNREACHABLE, and this pins that.
    //
    // `config_for_sequential_conjunction_graph` sets use_forward_bounds /
    // !use_crown_ibp / input_split_ibp_enhancement only when
    //   !use_relu_split && !use_alpha_crown && normalize_same_lhs_reduction(..).is_some()
    // while `should_upgrade_sequential_conjunction_to_graph` admits a route only when
    //   use_relu_split || normalize_same_lhs_reduction(..).is_none()
    // Those are mutually exclusive: with use_relu_split the bootstrap's first
    // conjunct fails, and without it the gate wants `is_none()` where the
    // bootstrap wants `is_some()`. So no input can reach a planned config with
    // the bootstrap applied.
    //
    // Asserting the CARRIED-THROUGH values (not the bootstrap ones) is what makes
    // this a tripwire: if the gate is ever loosened again, the bootstrap silently
    // becomes live and this fails, forcing a deliberate decision about a path
    // that currently has no coverage.
    assert!(
        !planned.use_forward_bounds,
        "the forward-bounds bootstrap is unreachable under the restored gate"
    );
    assert!(planned.use_crown_ibp, "incoming use_crown_ibp is carried");
    assert!(!planned.input_split_ibp_enhancement);

    let mut deferred = PresetConfig::default();
    deferred.attack.pgd_order = Some("after".to_string());
    let direct_error =
        require_deferred_pgd_consumer(Some(&deferred), true, false, true, false, false, false)
            .expect_err("the planned late Graph route must not drop deferred PGD");
    assert!(direct_error
        .to_string()
        .contains("post-BaB attack consumer"));
    assert!(
        require_deferred_pgd_consumer(Some(&deferred), true, false, true, false, false, true)
            .is_ok(),
        "the VNN-COMP outer consumer owns deferred PGD for the same planned route"
    );

    let projection = EffectiveTreatmentProjection::from_resolved(
        &planned,
        true,
        true,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    );
    let json = serde_json::to_value(projection).expect("projection serializes");
    assert_eq!(json["route"]["model_kind"], "graph");
    assert_eq!(
        json["route"]["late_sequential_conjunction_graph_upgrade"],
        true
    );

    assert!(
        super::planned_late_sequential_conjunction_graph_config(
            &model,
            &incoming,
            Some(&chained),
            false,
            true,
        )
        .is_none(),
        "MIP-only Sequential dispatch must not claim a Graph BaB route"
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
    let effective_treatment = EffectiveTreatmentProjection::from_resolved(
        &config,
        true,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(vnnlib.is_disjunction),
    );
    let mut ctx = DispatchContext {
        model_path: Path::new("unused.onnx"),
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        effective_treatment: &effective_treatment,
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
        engine_owns_deferred_pgd: false,
        outer_wrapper_owns_deferred_pgd: false,
        safenlp_direct_mip_first: false,
        cgan_input_leaf_route: None,
        gemm_engine: None,
        attack_engine_source: super::attack_arming::AttackEngineSource::disarmed(),
        compute_device: None,
        allow_heuristic_logsoftmax: false,
        allow_heuristic_softmax: false,
        input_split_metrics_jsonl: Some(metrics_path.as_path()),
        domain_batch_metrics_jsonl: None,
        verification_start: Instant::now(),
        overall_deadline: None,
        post_bab_wrapper_attack_enabled: None,
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
    let effective_treatment = EffectiveTreatmentProjection::from_resolved(
        &config,
        true,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(vnnlib.is_disjunction),
    );
    let mut ctx = DispatchContext {
        model_path: Path::new("unused.onnx"),
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        effective_treatment: &effective_treatment,
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
        engine_owns_deferred_pgd: false,
        outer_wrapper_owns_deferred_pgd: false,
        safenlp_direct_mip_first: false,
        cgan_input_leaf_route: None,
        gemm_engine: None,
        attack_engine_source: super::attack_arming::AttackEngineSource::disarmed(),
        compute_device: None,
        allow_heuristic_logsoftmax: false,
        allow_heuristic_softmax: false,
        input_split_metrics_jsonl: None,
        domain_batch_metrics_jsonl: Some(metrics_path.as_path()),
        verification_start: Instant::now(),
        overall_deadline: None,
        post_bab_wrapper_attack_enabled: None,
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
fn compat_free_after_is_deferred_and_never_builds_an_upfront_treatment() {
    let mut preset = PresetConfig::default();
    preset.attack.pgd_order = Some("after".to_string());

    assert!(resolve_pgd_attack(true, Some(&preset)));
    assert!(!effective_upfront_pgd(true, Some(&preset)));
    let forced_vgg_priority = super::VggAbcrownDecision {
        perturbed_count: 10_001,
        rewrite_mode: ny_propagate::VggMaxPoolRewriteMode::Residual,
        use_forward_bounds: false,
        prioritize_attack: true,
    };
    assert!(
        !effective_upfront_pgd_with_vgg(true, Some(&preset), Some(forced_vgg_priority)),
        "deferred placement must win over the large-VGG upfront optimization"
    );

    let config = BetaCrownConfig {
        enable_pgd_attack: true,
        ..BetaCrownConfig::default()
    };
    let projection = EffectiveTreatmentProjection::from_resolved(
        &config,
        true,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    )
    .with_deferred_pgd_schedule(true);
    let projected = serde_json::to_value(projection).unwrap();
    assert_eq!(projected["attack"]["schedule"], "deferred");
    assert_eq!(projected["route"]["run_upfront_pgd"], false);
    assert!(
        !super::dispatch::routes_to_relational_verifier(None, false),
        "a property-free Sequential invocation takes verify_standard"
    );
    assert!(
        require_deferred_pgd_consumer(Some(&preset), true, false, false, false, true, false)
            .is_ok(),
        "standard Sequential verify_standard owns the engine's internal deferred fallback"
    );
    assert_eq!(
        resolve_deferred_pgd_owner(true, true, false),
        DeferredPgdOwner::InternalEngine
    );
    assert_eq!(
        resolve_deferred_pgd_owner(true, true, true),
        DeferredPgdOwner::OuterWrapper,
        "the explicit outer route wins instead of running both consumers"
    );
    assert_eq!(
        resolve_deferred_pgd_owner(true, false, true),
        DeferredPgdOwner::OuterWrapper
    );
    assert_eq!(
        resolve_deferred_pgd_owner(false, true, true),
        DeferredPgdOwner::None,
        "a non-deferred schedule has no deferred consumer owner"
    );
    for (direct_route, model_is_graph, late_graph, mip_only) in [
        ("relational Sequential", false, false, false),
        ("Graph", true, false, false),
        ("late Graph", false, true, false),
        ("MIP-only", false, false, true),
    ] {
        let error = require_deferred_pgd_consumer(
            Some(&preset),
            true,
            model_is_graph,
            late_graph,
            mip_only,
            false,
            false,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("post-BaB attack consumer"),
            "direct {direct_route} must reject an unconsumed deferred slice: {error}"
        );
        assert!(
            require_deferred_pgd_consumer(
                Some(&preset),
                true,
                model_is_graph,
                late_graph,
                mip_only,
                false,
                true,
            )
            .is_ok(),
            "VNN-COMP must admit {direct_route} only with its outer consumer enabled"
        );
    }

    preset.attack.ny_pgd_order_compat = Some(crate::preset::NyPgdOrderCompat::Upfront);
    assert!(
        effective_upfront_pgd(true, Some(&preset)),
        "the explicit shipped compat:upfront contract must retain historical placement"
    );
    assert!(
        require_deferred_pgd_consumer(Some(&preset), true, true, true, true, false, false).is_ok(),
        "compat:upfront is consumed internally and does not need an outer deferred phase"
    );
    preset.attack.ny_pgd_order_compat = None;
    assert!(
        require_deferred_pgd_consumer(Some(&preset), false, true, true, true, false, false).is_ok(),
        "an explicit PGD disable removes the deferred attack and its consumer requirement"
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

    let mut projected_config = BetaCrownConfig {
        enable_pgd_attack: true,
        ..BetaCrownConfig::default()
    };
    apply_vgg_abcrown_bound_mode(&mut projected_config, Some(at_100));
    let projected = EffectiveTreatmentProjection::from_resolved(
        &projected_config,
        true,
        false,
        true,
        false,
        false,
        true,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    );
    let projected = serde_json::to_value(projected).expect("projection serializes");
    assert_eq!(projected["route"]["vgg_abcrown_treatment_active"], true);
    assert_eq!(projected["attack"]["schedule"], "input_bab");

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
fn disabled_engine_pgd_releases_its_post_bab_reservation() {
    let mut configured = ny_propagate::PhaseBudgetConfig {
        post_bab_pgd_fraction: 0.10,
        ..ny_propagate::PhaseBudgetConfig::default()
    };
    let baseline = serde_json::to_value(&configured).unwrap();

    let disabled = effective_engine_phase_budget(false, configured.clone());
    assert_eq!(disabled.post_bab_pgd_fraction, 0.0);
    configured.post_bab_pgd_fraction = 0.0;
    assert_eq!(
        serde_json::to_value(&disabled).unwrap(),
        serde_json::to_value(&configured).unwrap(),
        "disabling engine PGD must change only its scheduling-only post-BaB fraction"
    );

    let enabled = effective_engine_phase_budget(
        true,
        ny_propagate::PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..ny_propagate::PhaseBudgetConfig::default()
        },
    );
    assert_eq!(serde_json::to_value(enabled).unwrap(), baseline);
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
fn test_ibp_check_less_than_const_boundary_refuted_exactly() {
    // f32 5.0 embeds exactly in f64, so Y_0 >= 5.0 refutes Y_0 < 5.0.
    let ibp_lower = &[5.0_f32];
    let ibp_upper = &[10.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::LessThanConst(0, 5.0)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
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
fn test_ibp_check_greater_than_const_boundary_refuted_exactly() {
    // f32 5.0 embeds exactly in f64, so Y_0 <= 5.0 refutes Y_0 > 5.0.
    let ibp_lower = &[0.0_f32];
    let ibp_upper = &[5.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterThanConst(0, 5.0)]);
    assert!(ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_greater_eq_const_boundary_not_refuted() {
    // upper[0] = 5.0, const = 5.0 → GreaterEqConst(0, 5.0) (Y_0 >= 5.0) not refuted
    let ibp_lower = &[0.0_f32];
    let ibp_upper = &[5.0_f32];
    let spec = spec_with_clause(vec![OutputConstraint::GreaterEqConst(0, 5.0)]);
    assert!(!ibp_check_vnnlib_safe(ibp_lower, ibp_upper, &spec));
}

#[test]
fn test_ibp_check_malformed_output_intervals_fail_closed() {
    let spec = spec_with_clause(vec![OutputConstraint::LessEqConst(0, 0.0)]);
    let malformed = [
        (&[][..], &[][..]),
        (&[1.0][..], &[][..]),
        (&[f32::NAN][..], &[2.0][..]),
        (&[1.0][..], &[f32::INFINITY][..]),
        (&[2.0][..], &[1.0][..]),
    ];
    for (lower, upper) in malformed {
        assert!(
            !ibp_check_vnnlib_safe(lower, upper, &spec),
            "malformed box {lower:?}..{upper:?} must not prove safety"
        );
    }

    let mut wrong_shape = spec;
    wrong_shape.num_outputs = 2;
    assert!(!ibp_check_vnnlib_safe(&[1.0], &[2.0], &wrong_shape));
}

#[test]
fn test_ibp_check_nonfinite_constants_fail_closed() {
    for constant in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let spec = spec_with_clause(vec![OutputConstraint::LessEqConst(0, constant)]);
        assert!(!ibp_check_vnnlib_safe(&[1.0], &[2.0], &spec));
    }
}

#[test]
fn test_ibp_check_compares_thresholds_in_exact_f64() {
    let just_below_one = f64::from_bits(1.0_f64.to_bits() - 1);
    let just_above_one = f64::from_bits(1.0_f64.to_bits() + 1);
    let lower = &[1.0_f32];
    let upper = &[1.0_f32];

    let less = spec_with_clause(vec![OutputConstraint::LessEqConst(0, just_below_one)]);
    assert!(ibp_check_vnnlib_safe(lower, upper, &less));
    let greater = spec_with_clause(vec![OutputConstraint::GreaterEqConst(0, just_above_one)]);
    assert!(ibp_check_vnnlib_safe(lower, upper, &greater));
}

// --- gpu_bab auto-promotion tests (#4406) ---

fn canonical_two_singleton_disjunction() -> VnnLibSpec {
    let clauses = vec![
        vec![OutputConstraint::GreaterEq(0, 1)],
        vec![OutputConstraint::GreaterEq(1, 0)],
    ];
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = clauses.iter().flatten().cloned().collect();
    spec.output_constraint_clauses = clauses;
    spec.is_disjunction = true;
    spec
}

#[test]
fn test_auto_promote_gpu_bab_graph_bound_impact_single_objective_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::BoundImpact,
            false,
            Some(&spec)
        ),
        "graph + BoundImpact + single-objective should auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_graph_input_split_single_objective_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::InputSplit,
            false,
            Some(&spec)
        ),
        "graph + InputSplit + single-objective should auto-promote to gpu_bab"
    );
}

#[test]
fn test_auto_promote_gpu_bab_sequential_model_does_not_promote_4406() {
    use super::should_auto_promote_gpu_bab;
    let spec = VnnLibSpec::new();
    assert!(
        !should_auto_promote_gpu_bab(
            false,
            false,
            &BranchingHeuristic::BoundImpact,
            false,
            Some(&spec)
        ),
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
            false,
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
        !should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::BoundImpact,
            false,
            Some(&spec)
        ),
        "multi-clause disjunctive specs should NOT auto-promote (#4409)"
    );
}

#[test]
fn test_auto_promote_exact_two_singletons_for_graph_input_split() {
    use super::{should_auto_promote_gpu_bab, supports_independent_singleton_domain_list_spec};
    let spec = canonical_two_singleton_disjunction();

    assert!(supports_independent_singleton_domain_list_spec(&spec));
    assert!(should_auto_promote_gpu_bab(
        false,
        true,
        &BranchingHeuristic::InputSplit,
        true,
        Some(&spec)
    ));
    assert!(
        !should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::InputSplit,
            false,
            Some(&spec)
        ),
        "the structural match alone must not arm an automatic category treatment"
    );
    assert!(
        !should_auto_promote_gpu_bab(
            false,
            true,
            &BranchingHeuristic::BoundImpact,
            true,
            Some(&spec)
        ),
        "the decomposed exception is InputSplit-only"
    );
    assert!(
        !should_auto_promote_gpu_bab(
            false,
            false,
            &BranchingHeuristic::InputSplit,
            true,
            Some(&spec)
        ),
        "sequential models must remain on their existing route"
    );
}

#[test]
fn test_two_singleton_domain_list_gate_rejects_noncanonical_shapes() {
    use super::supports_independent_singleton_domain_list_spec;
    let canonical = canonical_two_singleton_disjunction();

    let mut three_clauses = canonical.clone();
    three_clauses
        .output_constraint_clauses
        .push(vec![OutputConstraint::GreaterEqConst(0, 0.0)]);
    three_clauses.output_constraints = three_clauses
        .output_constraint_clauses
        .iter()
        .flatten()
        .cloned()
        .collect();
    assert!(!supports_independent_singleton_domain_list_spec(
        &three_clauses
    ));

    let mut multirow = canonical.clone();
    multirow.output_constraint_clauses[0].push(OutputConstraint::GreaterEqConst(0, -1.0));
    multirow.output_constraints = multirow
        .output_constraint_clauses
        .iter()
        .flatten()
        .cloned()
        .collect();
    assert!(!supports_independent_singleton_domain_list_spec(&multirow));

    let mut clause_box = canonical.clone();
    clause_box.per_clause_input_bounds = vec![Default::default(), Default::default()];
    clause_box.per_clause_input_bounds[1].insert(0, (0.0, 1.0));
    assert!(!supports_independent_singleton_domain_list_spec(
        &clause_box
    ));

    let mut flat_mismatch = canonical.clone();
    flat_mismatch.output_constraints.swap(0, 1);
    assert!(!supports_independent_singleton_domain_list_spec(
        &flat_mismatch
    ));

    let mut not_disjunctive = canonical;
    not_disjunctive.is_disjunction = false;
    assert!(!supports_independent_singleton_domain_list_spec(
        &not_disjunctive
    ));
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
        should_auto_promote_gpu_bab(
            true,
            true,
            &BranchingHeuristic::BoundImpact,
            false,
            Some(&spec)
        ),
        "explicit --gpu-bab CLI flag should always enable, even on disjunctive specs"
    );
}

#[test]
fn test_auto_promote_gpu_bab_no_spec_promotes_4406() {
    use super::should_auto_promote_gpu_bab;
    assert!(
        should_auto_promote_gpu_bab(false, true, &BranchingHeuristic::BoundImpact, false, None),
        "graph + BoundImpact + no spec should auto-promote (epsilon-ball, no disjunction)"
    );
}

/// Resolution preserves an explicit preset request so final validation can
/// emit the cut-authority quarantine error instead of silently changing it.
#[test]
fn resolve_enable_cuts_preserves_preset_when_cli_silent() {
    use super::resolve_enable_cuts;
    assert!(
        resolve_enable_cuts(true, false, false, true),
        "a preset cut request must survive until quarantine validation"
    );
    assert!(
        !resolve_enable_cuts(false, false, false, true),
        "no preset, no CLI: cuts stay off"
    );
}

#[test]
fn resolve_enable_cuts_cli_flags_win() {
    use super::{resolve_enable_cuts, validate_cut_request};
    assert!(
        resolve_enable_cuts(false, true, false, true),
        "--enable-cuts must survive resolution until quarantine validation"
    );
    assert!(
        !resolve_enable_cuts(true, false, true, true),
        "--no-cuts disables preset-enabled cuts"
    );
    // main.rs collapses --enable-cuts + --no-cuts into cli_enable=false before
    // this point; the disable flag still wins.
    assert!(!resolve_enable_cuts(true, false, true, true));
    assert!(validate_cut_request(true, true).is_ok());
}

#[test]
fn resolve_enable_cuts_unsupported_model_class_forces_off() {
    use super::{resolve_enable_cuts, validate_cut_request};
    // Graph input splitting does not support cuts (#3813): effective preset
    // and CLI requests are rejected before this defensive resolver forces off
    // unsupported authority.
    assert!(!resolve_enable_cuts(true, false, false, false));
    let err = validate_cut_request(true, false).expect_err("must reject requested cuts");
    assert!(err
        .to_string()
        .contains("--enable-cuts / preset bab.cuts.enabled is unsupported"));
    assert!(validate_cut_request(false, false).is_ok());
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
