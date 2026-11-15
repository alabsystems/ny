// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::commands::beta_crown::{
    attack_arming::AttackEngineSource,
    dispatch::{run_bab_with_fallback, DispatchContext},
    output::{self, EffectiveTreatmentProjection},
    verify::phase_budget::PhaseBudgetLedger,
    BetaCrownModel, ProofOpts,
};
use crate::subcommands::{BackendArg, CompleteVerifierArg, MipSolverArg};
use ndarray::{arr1, arr2};
use ny_onnx::{
    vnnlib::{parse_vnnlib, VnnLibSpec},
    OnnxLoadConfig,
};
use ny_propagate::{
    layers::{LinearLayer, ReLULayer},
    BetaCrownConfig, Layer, Network, SingleHiddenHullMode,
};
use ny_tensor::BoundedTensor;

fn fixture(constant: f32, lp: bool) -> (Network, BoundedTensor, VnnLibSpec) {
    let mut network = Network::new();
    if lp {
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 1.0], [1.0, 0.0], [0.0, 1.0]]),
                Some(arr1(&[-1.0, 0.0, 0.0])),
            )
            .unwrap(),
        ));
    } else {
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32], [1.0]]), None).unwrap(),
        ));
    }
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            if lp {
                arr2(&[[-1.0_f32, 0.9, 0.1], [0.0, 0.0, 0.0]])
            } else {
                arr2(&[[1.0_f32, -1.0], [0.0, 0.0]])
            },
            Some(arr1(&[constant, 0.0])),
        )
        .unwrap(),
    ));
    let (input, property) = if lp {
        (BoundedTensor::new(arr1(&[0.0_f32, 0.0]).into_dyn(), arr1(&[1.0_f32, 1.0]).into_dyn()).unwrap(),
         "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(assert (>= X_0 0))\n(assert (<= X_0 1))\n(assert (>= X_1 0))\n(assert (<= X_1 1))\n(assert (<= Y_0 Y_1))")
    } else {
        (BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
         "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n(assert (<= Y_0 Y_1))")
    };
    (network, input, parse_vnnlib(property).unwrap())
}

fn dispatch_attempt(
    constant: f32,
    lp: bool,
    cancel: bool,
) -> (
    Result<()>,
    Option<output::CapturedVerification>,
    Arc<InstanceVerificationResources>,
) {
    dispatch_attempt_with_capture(constant, lp, cancel, output::take_captured_verification)
}

fn dispatch_attempt_with_capture(
    constant: f32,
    lp: bool,
    cancel: bool,
    capture: impl FnOnce() -> Option<output::CapturedVerification>,
) -> (
    Result<()>,
    Option<output::CapturedVerification>,
    Arc<InstanceVerificationResources>,
) {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(20);
    let resources = InstanceVerificationResources::start(Some(deadline)).unwrap();
    if cancel {
        resources.cancel();
    }
    let (network, input, spec) = fixture(constant, lp);
    let mut model = BetaCrownModel::Sequential(Box::new(network));
    let config = BetaCrownConfig {
        single_hidden_hull: SingleHiddenHullMode::Root,
        timeout: Duration::from_secs(20),
        use_alpha_crown: false,
        max_domains: 32,
        ..BetaCrownConfig::default()
    };
    let effective = EffectiveTreatmentProjection::from_resolved(
        &config,
        false,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    );
    let loader = OnnxLoadConfig::default();
    let proof_opts = ProofOpts::default();
    let property = None;
    let mut ctx = DispatchContext {
        input_head_continuation: None,
        model_path: std::path::Path::new("unused-hull-publication.onnx"),
        onnx_load_config: &loader,
        model_net: &mut model,
        input: &input,
        config: &config,
        effective_treatment: &effective,
        vnnlib_spec: Some(&spec),
        property: &property,
        epsilon: 0.0,
        effective_threshold: 0.0,
        verify_upper: false,
        output_dim: 2,
        const_output_idx: None,
        has_relational: true,
        use_relu_split: false,
        gpu_bab: false,
        run_upfront_pgd: false,
        engine_owns_deferred_pgd: false,
        outer_wrapper_owns_deferred_pgd: false,
        gemm_engine: None,
        attack_engine_source: AttackEngineSource::disarmed(),
        compute_device: None,
        allow_heuristic_logsoftmax: false,
        allow_heuristic_softmax: false,
        input_split_metrics_jsonl: None,
        domain_batch_metrics_jsonl: None,
        verification_start: started,
        overall_deadline: Some(deadline),
        verification_resources: Some(resources.clone()),
        post_bab_wrapper_attack_enabled: None,
        attack_concurrent_wrapper: None,
        json: true,
        sigmoid_peeled: false,
        proof_opts: &proof_opts,
    };
    output::begin_capture();
    let result = run_bab_with_fallback(&mut ctx, CompleteVerifierArg::Bab, MipSolverArg::AY);
    let captured = capture();
    output::end_capture();
    (result, captured, resources)
}

pub(crate) fn ordinary_dispatch(
    constant: f32,
    lp: bool,
    cancel: bool,
) -> (
    output::CapturedVerification,
    Arc<InstanceVerificationResources>,
) {
    let (result, captured, resources) = dispatch_attempt(constant, lp, cancel);
    result.expect("ordinary production dispatch");
    (captured.expect("ordinary dispatch must emit"), resources)
}

thread_local! { static CANCEL_AFTER_FORMAT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(in crate::commands::beta_crown) fn after_format(proof: &Arc<HullPublicationProof>) {
    if CANCEL_AFTER_FORMAT.with(|flag| flag.replace(false)) {
        proof.resources().cancel();
    }
}

pub(crate) fn plain_capture(json: &str) -> output::CapturedVerification {
    output::begin_capture();
    output::emit_competition_json(json);
    let captured = output::take_captured_verification().unwrap();
    output::end_capture();
    captured
}

#[test]
fn hull_publication_live_capture_refuses_mip_snapshot_and_retains_owned_publication() {
    let (result, captured, resources) = dispatch_attempt_with_capture(0.75, false, false, || {
        // Both byte-only consumers must leave the live authority untouched.
        // This capture comes from the actual dispatcher and original replay.
        assert!(output::snapshot_captured_json().is_none());
        assert!(output::take_captured_json().is_none());
        output::take_captured_verification()
    });
    result.expect("ordinary production dispatch");
    let captured = captured.expect("byte-only refusal must retain the owned capture");
    assert!(captured.has_hull());
    assert!(captured.hull_elapsed().is_some());
    let (json, authority) = captured.into_parts();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["status"],
        "verified"
    );
    let authority = authority.expect("publication authority moves with its JSON");
    authority.checkpoint().unwrap();
    assert!(authority.lower_bound() > 0.0 && authority.lower_bound() <= 0.75);
    assert_eq!(resources.budget().receipt().unwrap().proposals_reserved, 1);
    assert!(resources.budget().receipt().unwrap().live_persistent_bytes > OWNER_BYTES);
    drop(authority);
    assert_eq!(
        resources.budget().receipt().unwrap().live_persistent_bytes,
        OWNER_BYTES
    );
}

#[test]
fn hull_publication_ordinary_dispatch_retains_original_scope_and_completed_counts() {
    let (captured, resources) = ordinary_dispatch(0.75, false, false);
    assert!(
        captured.has_hull(),
        "IBP cannot close the duplicate-ReLU cancellation box"
    );
    assert!(output::take_captured_json().is_none());
    let value: serde_json::Value = serde_json::from_str(captured.json()).unwrap();
    assert_eq!(value["status"], "verified");
    assert_eq!(value["hull_resource"]["search_evaluations"], 2);
    assert_eq!(value["hull_resource"]["certificate_proposals"], 1);
    assert_eq!(value["hull_resource"]["certificate_evaluations"], 1);
    assert_eq!(value["hull_resource"]["original_replays"], 1);
    assert_eq!(value["hull_resource"]["selected_policy"], "proportional");
    let (_, authority) = captured.into_parts();
    let authority = authority.unwrap();
    authority.checkpoint().unwrap();
    // Independently, duplicate hidden ReLUs cancel exactly for every input.
    assert!(authority.lower_bound() > 0.0 && authority.lower_bound() <= 0.75);
    assert_eq!(resources.budget().receipt().unwrap().proposals_reserved, 1);
}

#[test]
fn hull_publication_phase_children_share_exact_counters_and_resident_owner() {
    let deadline = Instant::now() + Duration::from_secs(20);
    let owner = InstanceVerificationResources::start(Some(deadline)).unwrap();
    let ledger = PhaseBudgetLedger::from_deadline(Some(deadline), Default::default())
        .with_verification_resources(Some(owner.clone()));
    let child = ledger.child_with_timeout_secs(3);
    let inherited = child.verification_resources().unwrap();
    assert!(Arc::ptr_eq(inherited, &owner));
    assert!(inherited.budget().shares_ledger(owner.budget()));
    assert!(child.overall_deadline().unwrap() <= deadline);
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        OWNER_BYTES
    );
}

#[test]
fn hull_publication_receipt_clones_retain_paid_payload_until_last_owner() {
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    let (_, authority) = captured.into_parts();
    let authority = authority.unwrap();
    let weak = Arc::downgrade(&authority);
    let receipt =
        crate::commands::unsat_receipt::UnsatReceipt::from_original_hull(authority).unwrap();
    let retained = receipt.clone();
    drop(receipt);
    assert!(weak.upgrade().is_some());
    assert!(owner.budget().receipt().unwrap().live_persistent_bytes > OWNER_BYTES);
    assert_eq!(
        retained.published_summary().2,
        crate::commands::unsat_receipt::PublishedUnsatSource::OriginalHull
    );
    drop(retained);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        OWNER_BYTES
    );
}

#[test]
fn hull_publication_cancel_after_capture_refuses_receipt() {
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    let (_, authority) = captured.into_parts();
    owner.cancel();
    assert!(
        crate::commands::unsat_receipt::UnsatReceipt::from_original_hull(authority.unwrap())
            .is_err()
    );
}

#[test]
fn hull_publication_cancel_during_completed_formatting_never_captures_verified() {
    CANCEL_AFTER_FORMAT.with(|flag| flag.set(true));
    let (result, captured, owner) = dispatch_attempt(0.75, false, false);
    assert!(result.is_err());
    assert!(captured.is_none());
    assert!(owner.is_cancelled());
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        OWNER_BYTES
    );
}

#[test]
fn hull_publication_cancelled_dispatch_publishes_no_positive_authority() {
    let (captured, _) = ordinary_dispatch(0.75, false, true);
    assert!(!captured.has_hull());
    assert_ne!(
        serde_json::from_str::<serde_json::Value>(captured.json()).unwrap()["status"],
        "verified"
    );
}

#[test]
fn hull_publication_open_sat_and_existing_lp_fallback_are_preserved() {
    for constant in [0.0, -0.25] {
        let (captured, owner) = ordinary_dispatch(constant, false, false);
        assert!(!captured.has_hull());
        assert_ne!(
            serde_json::from_str::<serde_json::Value>(captured.json()).unwrap()["status"],
            "verified"
        );
        assert_eq!(owner.budget().receipt().unwrap().proposals_reserved, 1);
    }
    // Triangle is -.35; the existing AY ideal-hull LP proves the actual
    // positive .05 minimum. Keeping that route is a cumulative requirement.
    let (captured, owner) = ordinary_dispatch(0.05, true, false);
    assert!(
        !captured.has_hull(),
        "legacy LP has not gained the new resource receipt"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(captured.json()).unwrap()["status"],
        "verified"
    );
    assert_eq!(owner.budget().receipt().unwrap().proposals_reserved, 1);
}

#[test]
fn hull_publication_expired_original_deadline_refuses_owner() {
    assert!(InstanceVerificationResources::start(Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test clock supports an expired deadline")
    ))
    .is_err());
}

#[test]
fn hull_publication_foreign_equal_limit_context_cannot_bind_replay() {
    let deadline = Instant::now() + Duration::from_secs(20);
    let owner = InstanceVerificationResources::start(Some(deadline)).unwrap();
    let foreign = InstanceVerificationResources::start(Some(deadline)).unwrap();
    let (network, input, spec) = fixture(0.75, false);
    let context = foreign.context(deadline);
    let checked = super::super::single_hidden_hull::resource::try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &network,
        &input,
        &spec,
        context.query(),
        Instant::now(),
    )
    .unwrap();
    assert!(HullPublicationProof::from_checked(checked, owner.context(deadline)).is_err());
}

#[test]
fn hull_publication_old_capture_generation_cannot_emit_into_next_instance() {
    output::begin_capture();
    let deadline = Instant::now() + Duration::from_secs(20);
    let owner = InstanceVerificationResources::start(Some(deadline)).unwrap();
    let (network, input, spec) = fixture(0.75, false);
    let context = owner.context(deadline);
    let checked = super::super::single_hidden_hull::resource::try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &network,
        &input,
        &spec,
        context.query(),
        Instant::now(),
    )
    .unwrap();
    let (report, proof) = HullPublicationProof::from_checked(checked, context).unwrap();
    output::end_capture();
    output::begin_capture();
    let config = BetaCrownConfig::default();
    let effective = EffectiveTreatmentProjection::from_resolved(
        &config,
        false,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    );
    assert!(output::output_hull_result(report, proof, None, 0.0, true, &effective).is_err());
    assert!(output::take_captured_verification().is_none());
    output::end_capture();
}

#[test]
fn hull_publication_serializer_cap_is_bounded_without_growth() {
    use std::io::Write;
    let mut writer = JsonWriter::new().unwrap();
    let actual_capacity = writer.bytes.capacity();
    writer.write_all(&vec![b'x'; HULL_JSON_BYTES]).unwrap();
    assert!(writer.write_all(b"!").is_err());
    assert_eq!(writer.bytes.capacity(), actual_capacity);
}

#[test]
fn hull_publication_source_cap_refuses_before_serialization_and_preserves_replay() {
    output::begin_capture();
    let deadline = Instant::now() + Duration::from_secs(20);
    let owner = InstanceVerificationResources::start(Some(deadline)).unwrap();
    let (network, input, spec) = fixture(0.75, false);
    let context = owner.context(deadline);
    let checked = super::super::single_hidden_hull::resource::try_single_hidden_hull_root_in(
        SingleHiddenHullMode::Root,
        &network,
        &input,
        &spec,
        context.query(),
        Instant::now(),
    )
    .unwrap();
    let (report, proof) = HullPublicationProof::from_checked(checked, context).unwrap();
    let before = owner.budget().receipt().unwrap().live_persistent_bytes;
    let config = BetaCrownConfig::default();
    let oversized = EffectiveTreatmentProjection::from_resolved(
        &config,
        false,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "x".repeat(HULL_JSON_BYTES + 1),
        Some(false),
    );
    assert!(
        output::output_hull_result(report.clone(), proof.clone(), None, 0.0, true, &oversized)
            .is_err()
    );
    assert!(output::take_captured_verification().is_none());
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        before
    );
    proof.checkpoint().unwrap();
    let bounded = EffectiveTreatmentProjection::from_resolved(
        &config,
        false,
        false,
        false,
        false,
        false,
        false,
        CompleteVerifierArg::Bab,
        BackendArg::Cpu,
        "cpu",
        Some(false),
    );
    output::output_hull_result(report, proof, None, 0.0, true, &bounded).unwrap();
    assert!(output::take_captured_verification().unwrap().has_hull());
    assert_eq!(owner.budget().receipt().unwrap().proposals_reserved, 1);
    output::end_capture();
}

#[test]
fn hull_publication_underpaid_and_foreign_storage_cannot_authorize() {
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    let (_, authority) = captured.into_parts();
    let authority = authority.unwrap();
    let small = authority.proof.reserve_publication(1).unwrap();
    assert!(HullPublicationAuthority::new(authority.proof.clone(), small).is_err());
    let foreign =
        InstanceVerificationResources::start(Some(Instant::now() + Duration::from_secs(20)))
            .unwrap();
    let mut lease = foreign
        .budget()
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: HULL_JSON_BYTES * 2 + OWNER_BYTES,
                ..Default::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    assert!(HullPublicationAuthority::new(
        authority.proof.clone(),
        lease.commit_persistent().unwrap()
    )
    .is_err());
    assert!(authority._storage.shares_ledger(owner.budget()));
    authority.checkpoint().unwrap();
}
