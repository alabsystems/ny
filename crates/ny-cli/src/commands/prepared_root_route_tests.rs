// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual automatic-policy entry and original-file issuer, not imported proof
//! or numerical arrays. These controls are authored for successor qualification.

use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

const TEST_CAP: usize = 32 << 20;
fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn defaults() -> (LeverSet, MarginRowRuntimePolicy) {
    let levers = LeverSet::freeze_with(ny_levers::all(), |_| None);
    let policy = MarginRowRuntimePolicy::from_levers(&levers).unwrap();
    (levers, policy)
}
fn resources(at: Instant) -> Arc<PreparedRouteResources> {
    PreparedRouteResources::start_observed(
        TEST_CAP,
        ProcessMemoryEnvelope::Unbounded,
        Some(at),
        None,
        Instant::now(),
    )
    .unwrap()
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn info(name: &str, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: values.to_vec(),
        ..Default::default()
    }
}
fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: input.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(outputs: usize, target: usize, constant: bool, positive: bool) -> ModelProto {
    let mut head = node("Gemm", &["flat", "w1", "b1"], "headpre");
    let mut output = node("Gemm", &["head", "w2", "b2"], "output");
    for gemm in [&mut head, &mut output] {
        gemm.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
    }
    let mut weights = vec![0.0; 2 * outputs];
    if !constant {
        weights[2 * target] = 1.0;
        weights[2 * target + 1] = -1.0;
    }
    let mut bias = vec![0.0; outputs];
    bias[target] = if positive { 0.75 } else { -0.75 };
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![info("input", &[1, 1, 1, 1])],
            output: vec![info("output", &[1, outputs as i64])],
            initializer: vec![
                tensor("wc", &[1, 1, 1, 1], &[1.0]),
                tensor("bc", &[1], &[0.0]),
                tensor("w1", &[2, 1], &[1.0, 1.0]),
                tensor("b1", &[2], &[-1.0, -1.0]),
                tensor("w2", &[outputs as i64, 2], &weights),
                tensor("b2", &[outputs as i64], &bias),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                head,
                node("Relu", &["headpre"], "head"),
                output,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property(outputs: usize, target: usize) -> String {
    let mut text = String::from("(declare-const X_0 Real)\n");
    for c in 0..outputs {
        text.push_str(&format!("(declare-const Y_{c} Real)\n"));
    }
    text.push_str("(assert (>= X_0 0))\n(assert (<= X_0 3))\n(assert (or");
    for c in (0..outputs).rev().filter(|&c| c != target) {
        if c % 2 == 0 {
            text.push_str(&format!(" (>= Y_{c} Y_{target})"));
        } else {
            text.push_str(&format!(" (< Y_{target} Y_{c})"));
        }
    }
    text.push_str("))\n");
    text
}
fn files(
    outputs: usize,
    target: usize,
    constant: bool,
    positive: bool,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("original.onnx");
    let vnn = dir.path().join("original.vnnlib");
    std::fs::write(
        &onnx,
        model(outputs, target, constant, positive).encode_to_vec(),
    )
    .unwrap();
    std::fs::write(&vnn, property(outputs, target)).unwrap();
    (dir, onnx, vnn)
}
fn run_fixture(
    model: &Path,
    property: &Path,
    resource: Arc<PreparedRouteResources>,
) -> PreparedRouteOutcome {
    let (levers, policy) = defaults();
    let admitted = admission(false, &policy, &levers).0.unwrap().tree(1);
    run_admitted(
        model,
        property,
        &policy,
        admitted,
        resource,
        PreparedRouteReceipt::initial(&policy),
    )
}

#[test]
fn route_actual_default_policy_automatically_proves_original_root() {
    let (_dir, onnx, vnn) = files(3, 2, true, true);
    let (levers, policy) = defaults();
    assert_eq!(policy.root_precision.err_lane, ErrLane::PerCol);
    assert_eq!(policy.root_precision.err_lane.token(), "off");
    match try_prepared_root_route(
        &onnx,
        &vnn,
        &policy,
        admission(false, &policy, &levers),
        Some(clock()),
        None,
    ) {
        PreparedRouteOutcome::Proven { receipt, report } => {
            receipt.checkpoint_owner(None).unwrap();
            assert_eq!(report.status, PreparedRouteStatus::RootProven);
            assert_eq!(report.requested_error_lane, "off");
            assert_eq!(report.proof_arithmetic, PROOF_ARITHMETIC);
            assert_eq!(report.class_count, 2);
            assert!(report.root.unwrap().terminal_closed);
            let reference = report.root.unwrap().reference;
            assert_eq!(
                reference.arithmetic,
                ny_propagate::margin_row::PreparedReferenceArithmetic::Binary32Certified
            );
            assert_eq!(
                (reference.expected_conv_ops, reference.completed_conv_ops),
                (1, 1)
            );
            assert_eq!(reference.binary32_conv_dot_ops, 1);
            assert!(reference.multiply_add_pairs > 0);
            assert!(report.tree.is_none());
            assert!(report.byte_cap <= ROOT_SOURCE_BYTES_CAP && report.work_reserved > 0);
        }
        _ => panic!("actual default automatic original root did not produce its genuine proof"),
    }
}

#[test]
fn route_binary32_constant_only_original_conv_reports_coverage_without_dot_work() {
    let (_dir, onnx, vnn) = files(3, 2, true, true);
    let mut original = model(3, 2, true, true);
    original
        .graph
        .as_mut()
        .unwrap()
        .initializer
        .iter_mut()
        .find(|t| t.name == "wc")
        .unwrap()
        .float_data[0] = 0.0;
    std::fs::write(&onnx, original.encode_to_vec()).unwrap();
    let PreparedRouteOutcome::Proven { receipt, report } =
        run_fixture(&onnx, &vnn, resources(clock()))
    else {
        panic!("constant original proof");
    };
    let reference = report.root.unwrap().reference;
    assert_eq!(
        (reference.expected_conv_ops, reference.completed_conv_ops),
        (1, 1)
    );
    assert_eq!(reference.binary32_conv_dot_ops, 0);
    assert_eq!(reference.multiply_add_pairs, 0);
    assert!(reference.constant_outputs > 0 && reference.converted_cells > 0);
    receipt.checkpoint_owner(None).unwrap();
}

#[test]
fn route_real_open_root_closes_two_leaf_tree_with_65_original_classes() {
    let (_dir, onnx, vnn) = files(66, 17, false, true);
    match run_fixture(&onnx, &vnn, resources(clock())) {
        PreparedRouteOutcome::Proven { receipt, report } => {
            assert_eq!(report.status, PreparedRouteStatus::TreeProven);
            assert_eq!(report.class_count, 65);
            assert!(report.root.unwrap().ordinary_completed);
            assert!(!report.root.unwrap().terminal_closed);
            assert_eq!(report.root.unwrap().reference.binary32_conv_dot_ops, 1);
            assert!(report.root.unwrap().reference.multiply_add_pairs > 0);
            let tree = report.tree.unwrap();
            assert_eq!(tree.expansions, 1);
            assert_eq!(tree.bound_leaves, 2);
            let owner = receipt.original_prepared_authority().unwrap();
            assert_eq!(owner.target(), 17);
            assert_eq!(
                owner.classes(),
                (0_usize..66).rev().filter(|&c| c != 17).collect::<Vec<_>>()
            );
            assert!(owner.min_margin() > 0.0);
            receipt.checkpoint_owner(None).unwrap();
        }
        _ => panic!("actual original two-leaf tree did not close complete original objective"),
    }
}

#[test]
fn route_actual_negative_original_stays_open_and_retains_cancelled_owner() {
    let (_dir, onnx, vnn) = files(3, 1, false, false);
    let resources = resources(clock());
    let memory = resources.budget.memory_authority().unwrap();
    let outcome = run_fixture(&onnx, &vnn, resources.clone());
    match outcome {
        PreparedRouteOutcome::Open {
            owner: PreparedRouteOpen::Tree(open),
            report,
        } => {
            assert_eq!(report.status, PreparedRouteStatus::Open);
            assert_eq!(report.root.unwrap().reference.binary32_conv_dot_ops, 1);
            assert_eq!(open.classes().len(), 2);
            assert!(open.open_domains() > 0);
            open.checkpoint(None).unwrap();
            resources.cancel();
            assert!(open.checkpoint(None).is_err());
            let alias = open.clone();
            drop(open);
            drop(resources);
            assert!(live(&memory) >= OWNER_BYTES);
            drop(alias);
            assert_eq!(live(&memory), 0);
        }
        _ => panic!("negative original cannot acquire a proof"),
    }
}

#[test]
fn route_receipt_last_alias_holds_caller_resource_claim_and_ledger() {
    let (_dir, onnx, vnn) = files(3, 1, true, true);
    let resources = resources(clock());
    let budget = resources.budget.clone();
    let memory = budget.memory_authority().unwrap();
    let weak = Arc::downgrade(&resources);
    let receipt = match run_fixture(&onnx, &vnn, resources.clone()) {
        PreparedRouteOutcome::Proven { receipt, .. } => receipt,
        _ => panic!("actual original constant root must prove"),
    };
    assert!(receipt
        .original_prepared_authority()
        .unwrap()
        .budget()
        .shares_ledger(&budget));
    let alias = receipt.clone();
    drop(resources);
    drop(receipt);
    assert!(weak.upgrade().is_some());
    assert!(live(&memory) >= OWNER_BYTES);
    alias.checkpoint_owner(None).unwrap();
    // Weak aliases exist only for this observation; release the control block
    // observer before the last genuine strong owner's storage claim drops.
    drop(weak);
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn route_caller_minimum_phase_clock_never_renews_in_clones() {
    let original = clock();
    let phase = Instant::now() + Duration::from_secs(20);
    let owner = PreparedRouteResources::start_observed(
        TEST_CAP,
        ProcessMemoryEnvelope::Unbounded,
        Some(original),
        Some(phase),
        Instant::now(),
    )
    .unwrap();
    let alias = owner.clone();
    assert!(owner.deadline(None).unwrap() <= phase);
    assert!(alias.deadline(Some(original)).unwrap() <= phase);
    let budget = owner.budget.clone();
    let memory = budget.memory_authority().unwrap();
    drop(owner);
    assert!(live(&memory) >= OWNER_BYTES);
    alias.cancel();
    assert!(alias.checkpoint(None).is_err());
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Verify inherited cutoff and ownership through cloned proof and receipt aliases."
)]
fn route_real_proof_inherits_single_original_phase_minus_reserve_cutoff() {
    let (_dir, onnx, vnn) = files(3, 1, true, true);
    let original = clock();
    let phase = Instant::now() + Duration::from_secs(20);
    let resources = PreparedRouteResources::start_observed(
        TEST_CAP,
        ProcessMemoryEnvelope::Unbounded,
        Some(original),
        Some(phase),
        Instant::now(),
    )
    .unwrap();
    let cutoff = resources.certified_entry_deadline().unwrap();
    assert_eq!(
        cutoff,
        resources.deadline.checked_sub(CERTIFICATE_RESERVE).unwrap()
    );
    assert!(cutoff <= phase.checked_sub(CERTIFICATE_RESERVE).unwrap());
    let resource_alias = resources.clone();
    match run_fixture(&onnx, &vnn, resources) {
        PreparedRouteOutcome::Proven { receipt, report } => {
            assert_eq!(report.certified_entry_cutoff, Some(cutoff));
            assert_eq!(
                report.original_deadline,
                Some(resource_alias.original_deadline)
            );
            let proof = receipt.original_prepared_authority().unwrap();
            assert!(proof.deadline() <= cutoff);
            assert!(proof.session().unwrap().deadline() <= cutoff);
            assert!(proof.clone().restricted(Some(original)).unwrap().deadline() <= cutoff);
            receipt.clone().checkpoint_owner(Some(original)).unwrap();
            // The resource clock is not renewed or mistaken for proof time.
            assert_eq!(
                resource_alias.deadline(Some(original)).unwrap(),
                resource_alias.deadline
            );
            assert!(resource_alias.deadline >= cutoff + CERTIFICATE_RESERVE);
        }
        _ => panic!("actual original proof must preserve earlier cutoff through all aliases"),
    }
}

#[test]
fn route_unsupported_precision_and_armed_doses_decline_before_files() {
    let (levers, base) = defaults();
    let absent = Path::new("/no/prepared/profile/source");
    for variant in 0..8 {
        let mut policy = base;
        match variant {
            0 => policy.root_precision.root_f32 = true,
            1 => policy.root_precision.tableau_f32 = true,
            2 => policy.root_precision.err_lane = ErrLane::Envelope,
            3 => policy.root_precision.box_source = RootBoxSource::PatchBackward,
            4 => policy.root_precision.ibp_seed = true,
            5 => policy.backward_interm.enabled = true,
            6 => policy.root_alpha.alpha_enabled = true,
            _ => policy.domain_alpha.enabled = true,
        }
        match try_prepared_root_route(
            absent,
            absent,
            &policy,
            admission(false, &policy, &levers),
            Some(clock()),
            None,
        ) {
            PreparedRouteOutcome::Refused { report } => {
                assert_eq!(report.status, PreparedRouteStatus::Declined);
                assert_eq!(report.byte_cap, 0);
                assert_eq!(report.work_reserved, 0);
            }
            _ => panic!("unsupported actual policy must decline before source work"),
        }
    }
}

#[test]
fn route_frozen_clip_beta_gpu_and_frozen_dual_requests_decline_unchanged() {
    let absent = Path::new("/no/prepared/profile/source");
    for name in [
        "NY_MARGIN_ROW_CLIP",
        "NY_MARGIN_ROW_CLIP_ROWS",
        "NY_MARGIN_ROW_CLIP_INTERM",
        "NY_MARGIN_ROW_CLIP_STATEFUL",
        "NY_MARGIN_ROW_BETA",
        "NY_MARGIN_ROW_GPU_BACKWARD",
        "NY_MARGIN_ROW_GPU_CONV",
        "NY_MARGIN_ROW_FROZEN_DUAL_BAB",
        "NY_MARGIN_ROW_PARENT_LIFT",
    ] {
        let levers =
            LeverSet::freeze_with(ny_levers::all(), |key| (key == name).then(|| "1".into()));
        let policy = MarginRowRuntimePolicy::from_levers(&levers).unwrap();
        match try_prepared_root_route(
            absent,
            absent,
            &policy,
            admission(false, &policy, &levers),
            Some(clock()),
            None,
        ) {
            PreparedRouteOutcome::Refused { report } => {
                assert_eq!(report.status, PreparedRouteStatus::Declined)
            }
            _ => panic!("armed independent consumer cannot be silently discharged"),
        }
        assert!(levers
            .resolved(ny_levers::all().get(name).unwrap())
            .unwrap()
            .value
            .as_bool());
    }
}

#[test]
fn route_memory_policy_is_independent_finite_and_never_widens() {
    let cap = admitted_byte_cap(
        96 << 20,
        ProcessMemoryEnvelope::Bounded {
            headroom_bytes: 80 << 20,
        },
    )
    .unwrap();
    assert_eq!(cap, (80_usize << 20).min(ROOT_SOURCE_BYTES_CAP));
    assert!(cap > 32_usize << 20);
    assert_eq!(
        admitted_byte_cap(24 << 20, ProcessMemoryEnvelope::Unbounded).unwrap(),
        24_usize << 20
    );
    assert_eq!(
        admitted_byte_cap(usize::MAX, ProcessMemoryEnvelope::Unbounded).unwrap(),
        ROOT_SOURCE_BYTES_CAP
    );
    assert_eq!(
        admitted_byte_cap(
            TEST_CAP,
            ProcessMemoryEnvelope::Bounded { headroom_bytes: 0 }
        )
        .unwrap(),
        0
    );
    assert!(admitted_byte_cap(TEST_CAP, ProcessMemoryEnvelope::Unavailable).is_err());
}

#[test]
fn route_unknown_memory_missing_clock_and_reserve_refuse_before_owner() {
    assert!(PreparedRouteResources::start_observed(
        TEST_CAP,
        ProcessMemoryEnvelope::Unavailable,
        Some(clock()),
        None,
        Instant::now()
    )
    .is_err());
    assert!(PreparedRouteResources::start_observed(
        OWNER_BYTES - 1,
        ProcessMemoryEnvelope::Unbounded,
        Some(clock()),
        None,
        Instant::now()
    )
    .is_err());
    let now = Instant::now();
    assert!(PreparedRouteResources::start_observed(
        TEST_CAP,
        ProcessMemoryEnvelope::Unbounded,
        Some(now + CERTIFICATE_RESERVE),
        None,
        now
    )
    .is_err());
    if ny_core::instance_budget::deadline().is_none() {
        assert!(PreparedRouteResources::start_observed(
            TEST_CAP,
            ProcessMemoryEnvelope::Unbounded,
            None,
            None,
            now
        )
        .is_err());
    }
}

#[test]
fn route_work_quota_is_once_finite_millisecond_allowance_not_cpu_rate() {
    assert_eq!(
        scheduling_work(Duration::from_millis(123)).unwrap(),
        123 * WORK_PER_MILLISECOND
    );
    assert_eq!(scheduling_work(Duration::MAX).unwrap(), MAX_WORK);
    assert!(scheduling_work(Duration::ZERO).is_err());
    let owner = resources(clock());
    let alias = owner.clone();
    let reserved = owner.budget.receipt().unwrap().work_reserved;
    let lease = owner
        .budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: owner.work_allowance - reserved,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    drop(lease);
    assert_eq!(
        alias.budget.receipt().unwrap().work_reserved,
        owner.work_allowance
    );
    assert!(alias
        .budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 1,
                ..WorkRequest::default()
            }
        )
        .is_err());
}

#[test]
fn route_source_size_declines_before_snapshot_or_decode_admission() {
    let (dir, onnx, vnn) = files(3, 1, true, true);
    let owner = resources(clock());
    let before = owner.budget.receipt().unwrap();
    std::fs::File::create(&onnx)
        .unwrap()
        .set_len((owner.byte_cap / 8 + 1) as u64)
        .unwrap();
    let error = OriginalPreparedRootSession::prepare_original_files_in_caller(
        &onnx,
        &vnn,
        MarginRowRuntimePolicy::default().root_precision,
        PreparedRootProofOptions::default(),
        owner.clone(),
        clock(),
    )
    .err()
    .expect("oversized source must decline before snapshot");
    assert!(error.to_string().contains("preservation profile"));
    let after = owner.budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.work_reserved, before.work_reserved);
    drop(dir);
}

#[test]
fn route_no_fixed_vocabulary_ceiling_authenticates_4099_original_classes() {
    let (_dir, onnx, vnn) = files(4100, 17, true, true);
    let owner = resources(clock());
    owner.validate_objective(65_537, 4100).unwrap();
    assert!(owner.validate_objective(usize::MAX, 4100).is_err());
    assert!(owner
        .validate_objective(owner.byte_cap / 40 + 1, 4100)
        .is_err());
    // The existing staged parser's normalizer quote alone is 16 KiB/output,
    // exceeding this small unit ledger at 4,100 outputs. The positive attempt
    // uses the actual automatic CPU/process memory policy, with no raised or
    // supplied test cap; all production census admission remains mandatory.
    drop(owner);
    let (levers, policy) = defaults();
    match try_prepared_root_route(
        &onnx,
        &vnn,
        &policy,
        admission(false, &policy, &levers),
        Some(clock()),
        None,
    ) {
        PreparedRouteOutcome::Proven { receipt, report } => {
            assert_eq!(report.class_count, 4099);
            let authority = receipt.original_prepared_authority().unwrap();
            assert_eq!(authority.classes().len(), 4099);
            assert_eq!(authority.target(), 17);
            receipt.checkpoint_owner(None).unwrap();
        }
        _ => panic!("actual finite class storage must decide admission, not fixed vocabulary cap"),
    }
}

#[test]
fn route_parallel_and_cache_requests_preserve_frozen_grammar_and_limits() {
    for (raw, expected) in [
        ("1", 7),
        (" YES ", 7),
        ("0", 1),
        ("off", 1),
        ("16", 16),
        ("bad", 1),
    ] {
        assert_eq!(parallel_width(raw, 7), expected);
    }
    let levers = LeverSet::freeze_with(ny_levers::all(), |name| match name {
        "NY_MARGIN_ROW_PARALLEL" => Some("2".into()),
        "NY_MARGIN_ROW_LRU" => Some("0".into()),
        _ => None,
    });
    let policy = MarginRowRuntimePolicy::from_levers(&levers).unwrap();
    let tree = admission(false, &policy, &levers).0.unwrap().tree(7);
    assert_eq!(tree.lru_capacity, 0);
    assert_eq!(tree.schedule, PreparedTreeSchedule::Parallel { width: 2 });
    assert_eq!(tree.max_expansions, 32);
}

#[test]
fn route_cancelled_original_caller_cannot_start_source_work() {
    let (_dir, onnx, vnn) = files(3, 1, true, true);
    let owner = resources(clock());
    let before = owner.budget.receipt().unwrap();
    owner.cancel();
    match run_fixture(&onnx, &vnn, owner.clone()) {
        PreparedRouteOutcome::Refused { report } => {
            assert_eq!(report.status, PreparedRouteStatus::Refused)
        }
        _ => panic!("cancelled original caller must not start or publish"),
    }
    assert_eq!(owner.budget.receipt().unwrap().admitted, before.admitted);
}

// ---- #prepared-root-route: the extracted admission ---------------------------

/// Origin's inline admission (`policy_admission` at 7f4564cfe, with its
/// `parallel_width`), with the directed Source-default adaptation: the
/// default head portfolio now runs earlier, while explicit group doses still
/// decline. All other decision/reason/tree options retain bit identity.
fn origin_inline_admission(
    policy: &MarginRowRuntimePolicy,
    levers: &LeverSet,
    workers: usize,
) -> std::result::Result<PreparedTreeOptions, &'static str> {
    use ny_levers::decls::{margin_row as d, sound_channel as s};
    fn origin_parallel_width(raw: &str, workers: usize) -> usize {
        let raw = raw.trim();
        if raw == "1"
            || ["true", "on", "yes"]
                .iter()
                .any(|v| raw.eq_ignore_ascii_case(v))
        {
            workers.max(1)
        } else if raw.is_empty()
            || raw == "0"
            || ["false", "off", "no"]
                .iter()
                .any(|v| raw.eq_ignore_ascii_case(v))
        {
            1
        } else {
            raw.parse::<usize>().unwrap_or(1).max(1)
        }
    }
    if policy.root_precision.root_f32
        || policy.root_precision.tableau_f32
        || policy.root_precision.err_lane != ErrLane::PerCol
        || policy.root_precision.ibp_seed
        || policy.root_precision.box_source != RootBoxSource::Tableau
    {
        return Err("unsupported-root-precision-or-source-dose");
    }
    let default = MarginRowRuntimePolicy::default();
    let explicit_head_group = policy.head_groups.is_some()
        && (policy.head_groups != default.head_groups
            || levers
                .resolved(&d::MARGIN_ROW_HEAD_GROUPS)
                .ok_or("missing-frozen-head-groups")?
                .source
                != ny_levers::Source::Default
            || *frozen_value(levers, &d::MARGIN_ROW_HEAD_GROUPS)? != LeverValue::Bool(true));
    if policy.width != default.width
        || policy.backward_interm.enabled
        || policy.root_alpha.alpha_enabled
        || policy.root_alpha.rebake_enabled
        || policy.root_alpha.spec_alpha_enabled
        || policy.joint_refresh.rounds != 0
        || policy.root_certify_once.enabled
        || explicit_head_group
        || policy.domain_alpha.enabled
        || policy.epoch.depth != 0
        || policy.epoch.tier0_exact != 0
        || policy.epoch.parent_topk != 0
        || policy.epoch.debug_trace
        || policy.beta_per_column
        || policy.column_retire
        || policy.parallel_prebuild
        || policy.wave.mode != TreeMode::Legacy
    {
        return Err("unsupported-armed-runtime-dose");
    }
    for decl in [
        &d::MARGIN_ROW_BETA,
        &d::MARGIN_ROW_CLIP_STATEFUL,
        &d::MARGIN_ROW_BRANCH_REDUCE_MAX,
        &d::MARGIN_ROW_PARENT_LIFT,
        &d::MARGIN_ROW_GPU_BACKWARD,
        &d::MARGIN_ROW_GPU_SHADOW,
        &d::MARGIN_ROW_GPU_PROF,
        &d::MARGIN_ROW_GPU_CONV,
        &d::MARGIN_ROW_DOMAIN_STACK,
        &d::MARGIN_ROW_FROZEN_DUAL_BAB,
        &s::MARGIN_ROW_CLIP,
        &s::MARGIN_ROW_CLIP_HEAD,
        &s::MARGIN_ROW_CLIP_INTERM,
        &s::MARGIN_ROW_CLIP_ROWS,
        &s::MARGIN_ROW_CLOSURE_ADAPTIVE,
    ] {
        if frozen_value(levers, decl)?.as_bool() {
            return Err("unsupported-armed-consumer-dose");
        }
    }
    let lru_capacity = frozen_value(levers, &d::MARGIN_ROW_LRU)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("invalid-frozen-lru")?;
    let parallel = frozen_value(levers, &d::MARGIN_ROW_PARALLEL)?
        .as_str()
        .ok_or("invalid-frozen-parallel")?;
    let width = origin_parallel_width(parallel, workers);
    Ok(PreparedTreeOptions {
        max_expansions: MAX_EXPANSIONS,
        lru_capacity,
        schedule: if width == 1 {
            PreparedTreeSchedule::Serial
        } else {
            PreparedTreeSchedule::Parallel { width }
        },
    })
}

/// Every frozen lever set the sweep asks: the defaults, then every margin-row,
/// sound-channel and root-alpha declaration at each of a set of raw tokens
/// (an invalid token resolves to the declaration default, as on the scored
/// path), each with the policy those levers resolve to. (Declarations are
/// taken from their registries, never spelled: no lever-name literal.)
fn admission_sweep() -> Vec<(String, LeverSet, MarginRowRuntimePolicy)> {
    let mut out = Vec::new();
    let (levers, policy) = defaults();
    out.push(("defaults".to_owned(), levers, policy));
    let tokens = [
        "1", "0", "2", "16", "yes", "off", "bad", "wave", "legacy", "",
    ];
    let decls = ny_levers::decls::margin_row::MARGIN_ROW_LEVERS
        .decls()
        .iter()
        .chain(ny_levers::decls::sound_channel::SOUND_CHANNEL_LEVERS.decls())
        .chain(ny_levers::decls::root_alpha::ROOT_ALPHA_LEVERS.decls());
    for decl in decls {
        for token in tokens {
            let levers = LeverSet::freeze_with(ny_levers::all(), |name| {
                (name == decl.name).then(|| token.to_owned())
            });
            if let Ok(policy) = MarginRowRuntimePolicy::from_levers(&levers) {
                out.push((format!("{}={token:?}", decl.name), levers, policy));
            }
        }
    }
    out
}

/// A policy-field dose.
type Dose = (&'static str, fn(&mut MarginRowRuntimePolicy));

/// The policy-field doses the frozen levers alone may not reach.
const POLICY_DOSES: [Dose; 20] = [
    ("root_f32", |p| p.root_precision.root_f32 = true),
    ("tableau_f32", |p| p.root_precision.tableau_f32 = true),
    ("err_lane", |p| {
        p.root_precision.err_lane = ErrLane::Envelope
    }),
    ("box_source", |p| {
        p.root_precision.box_source = RootBoxSource::PatchBackward;
    }),
    ("ibp_seed", |p| p.root_precision.ibp_seed = true),
    ("backward_interm", |p| p.backward_interm.enabled = true),
    ("alpha", |p| p.root_alpha.alpha_enabled = true),
    ("rebake", |p| p.root_alpha.rebake_enabled = true),
    ("spec_alpha", |p| p.root_alpha.spec_alpha_enabled = true),
    ("joint_refresh", |p| p.joint_refresh.rounds = 1),
    ("certify_once", |p| p.root_certify_once.enabled = true),
    ("domain_alpha", |p| p.domain_alpha.enabled = true),
    ("epoch_depth", |p| p.epoch.depth = 1),
    ("epoch_tier0", |p| p.epoch.tier0_exact = 1),
    ("epoch_topk", |p| p.epoch.parent_topk = 1),
    ("epoch_trace", |p| p.epoch.debug_trace = true),
    ("beta_per_column", |p| p.beta_per_column = true),
    ("column_retire", |p| p.column_retire = true),
    ("parallel_prebuild", |p| p.parallel_prebuild = true),
    ("wave", |p| p.wave.mode = TreeMode::Wave),
];

fn policy_mutations(base: &MarginRowRuntimePolicy) -> Vec<(&'static str, MarginRowRuntimePolicy)> {
    POLICY_DOSES
        .iter()
        .map(|(name, mutate)| {
            let mut policy = *base;
            mutate(&mut policy);
            (*name, policy)
        })
        .collect()
}

/// BIT-IDENTITY: the extracted predicate decides EXACTLY what origin's inline
/// admission decided — the same admit/decline, the same reason token and the
/// same tree options on every worker count — over every margin-row lever at
/// a sweep of raw tokens and every policy-field dose; and the classwise
/// experiment steps the route aside whatever the policy (origin's
/// `vnncomp.rs` branch, now inside the predicate).
#[test]
fn prepared_root_admission_is_bit_identical_to_the_inline_admission() {
    let mut cases = 0usize;
    let mut admitted = 0usize;
    let mut declined = std::collections::BTreeSet::new();
    let sweep = admission_sweep();
    for (name, levers, policy) in &sweep {
        let mut policies = vec![("frozen", *policy)];
        policies.extend(policy_mutations(policy));
        for (dose, policy) in policies {
            let new = admission(false, &policy, levers);
            for workers in [1, 7] {
                let want = origin_inline_admission(&policy, levers, workers);
                let got = new
                    .0
                    .map(|a| a.tree(workers))
                    .map_err(PreparedRootDecline::reason);
                assert_eq!(got, want, "{name} {dose} workers={workers}");
                cases += 1;
            }
            match new.decline() {
                None => admitted += 1,
                Some(decline) => {
                    declined.insert(decline.reason());
                }
            }
            let classwise = admission(true, &policy, levers);
            assert!(classwise.classwise_armed(), "{name} {dose}");
            assert_eq!(
                classwise.decline().map(PreparedRootDecline::reason),
                Some("armed-original-classwise-policy")
            );
        }
    }
    // Non-vacuity: both outcomes and every policy refusal were exercised.
    assert!(
        sweep.len() > 100,
        "the sweep must reach the margin-row levers"
    );
    assert!(
        admitted > 0 && cases > 1000,
        "admitted={admitted} cases={cases}"
    );
    for reason in [
        "unsupported-root-precision-or-source-dose",
        "unsupported-armed-runtime-dose",
        "unsupported-armed-consumer-dose",
    ] {
        assert!(
            declined.contains(reason),
            "{reason} never declined: {declined:?}"
        );
    }
}

/// THE LANE RUNS ONLY ON THE ADMISSION'S ANSWER: a declined admission
/// returns the route's refusal before any resource or source work, with the
/// receipt origin's inline admission wrote (the `prepared_root` flight note
/// is `format!("{report:?}")` of it, so it is compared as text); an admitted
/// one proceeds past admission (here to the absent source).
#[test]
fn prepared_root_lane_runs_only_on_the_admissions_answer() {
    let absent = Path::new("/no/prepared/profile/source");
    let mut proceeded = 0usize;
    let mut stopped = 0usize;
    for (name, levers, policy) in admission_sweep() {
        let answer = admission(false, &policy, &levers);
        let outcome = try_prepared_root_route(absent, absent, &policy, answer, Some(clock()), None);
        let PreparedRouteOutcome::Refused { report } = outcome else {
            panic!("{name}: an absent source never proves or opens");
        };
        match answer.decline() {
            Some(decline) => {
                let mut want = PreparedRouteReceipt::initial(&policy);
                want.reason = decline.reason();
                assert_eq!(format!("{report:?}"), format!("{want:?}"), "{name}");
                assert_eq!(report.status, PreparedRouteStatus::Declined, "{name}");
                stopped += 1;
            }
            None => {
                // Past admission: the caller's resources, then the source.
                assert!(
                    matches!(
                        report.reason,
                        "caller-resource-admission" | "original-source-or-root-refused"
                    ),
                    "{name}: {}",
                    report.reason
                );
                proceeded += 1;
            }
        }
    }
    assert!(
        proceeded > 0 && stopped > 0,
        "proceeded={proceeded} stopped={stopped}"
    );
}

/// THE ROUTER ASKS THE SAME PREDICATE: `router_admission` on a frozen lever
/// set is `admission` on the policy those levers resolve to, in the
/// competition form (no classwise experiment): admitted exactly when the lane
/// would run, declined with the lane's own reason otherwise.
#[test]
fn prepared_root_router_admission_is_the_lane_admission() {
    use ny_core::router::decl::Admission;
    let mut declined = 0usize;
    for (name, levers, policy) in admission_sweep() {
        let lane = admission(false, &policy, &levers);
        let want = match lane.decline() {
            None => Admission::Admit,
            Some(decline) => {
                declined += 1;
                Admission::Decline(decline.reason())
            }
        };
        assert_eq!(router_admission(&levers), want, "{name}");
    }
    assert!(
        declined > 0,
        "positive control: the sweep declines somewhere"
    );
}

/// DEFAULT-ON IS UNCHANGED: a row with no category file and zero `NY_*`
/// still runs the route (the predicate admits the declaration defaults), and
/// so do the router and the census on that lever set.
#[test]
fn prepared_root_still_runs_by_default() {
    let (levers, policy) = defaults();
    let answer = admission(false, &policy, &levers);
    assert!(answer.decline().is_none(), "{answer:?}");
    let census = crate::router::census::no_category_levers();
    assert_eq!(
        router_admission(&census),
        ny_core::router::decl::Admission::Admit
    );
    assert_eq!(
        format!(
            "{:?}",
            MarginRowRuntimePolicy::from_levers(&census).unwrap()
        ),
        format!("{policy:?}"),
        "the census lever context is the defaults"
    );
}

#[test]
fn default_head_portfolio_keeps_prepared_fallback_but_explicit_doses_decline() {
    use ny_levers::decls::margin_row as d;
    let (levers, policy) = defaults();
    assert!(policy.head_groups.unwrap().directional_cuts());
    assert!(admission(false, &policy, &levers).decline().is_none());
    for (raw, expected) in [("1", Some("unsupported-armed-runtime-dose")), ("0", None)] {
        let frozen = LeverSet::freeze_with(ny_levers::all(), |name| {
            (name == d::MARGIN_ROW_HEAD_GROUPS.name).then(|| raw.to_owned())
        });
        let policy = MarginRowRuntimePolicy::from_levers(&frozen).unwrap();
        assert_eq!(
            admission(false, &policy, &frozen)
                .decline()
                .map(PreparedRootDecline::reason),
            expected
        );
    }
    // Even a numerically default true is an explicit typed request. The
    // prepared engine must leave that requested legacy group route its dose.
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    let configured = LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        (decl.name == d::MARGIN_ROW_HEAD_GROUPS.name).then_some(LeverValue::Bool(true))
    })
    .unwrap();
    let configured_policy = MarginRowRuntimePolicy::from_levers(&configured).unwrap();
    assert_eq!(configured_policy, policy);
    assert_eq!(
        admission(false, &configured_policy, &configured)
            .decline()
            .map(PreparedRootDecline::reason),
        Some("unsupported-armed-runtime-dose")
    );
    let changed = MarginRowRuntimePolicy {
        head_groups: policy
            .head_groups
            .map(|group| group.with_directional_cuts(false)),
        ..policy
    };
    assert_eq!(
        admission(false, &changed, &levers)
            .decline()
            .map(PreparedRootDecline::reason),
        Some("unsupported-armed-runtime-dose")
    );
}

// ---- review(prepared-root-route): the census's superset claim, pinned -----
//
// The admission census scopes this route's moat with the ROUTER's start-time
// property read (`spec_shape::read_spec`): it admits a row only when that read
// is argmax-margin robustness (or unknown). That is a superset of the rows the
// route can run on only while two facts about the route's OWN property gates
// hold, so they are pinned here rather than assumed.

/// The route's property acceptance, as a SUPERSET of the staged parser: the
/// raw-source objective authentication at the staged ceiling, the production
/// parse with its certified input box, and their cross-check
/// (`parse_staged_original_property` additionally runs a refusing-only token
/// census, which can only shrink this set).
fn route_accepts_property(text: &str) -> bool {
    use super::super::original_property_census::MAX_STAGED_PROPERTY_BYTES;
    use super::super::prepared_root_publication::{
        authenticate_raw_objective_with_cap, cross_check,
    };
    let owner = resources(clock());
    let at = owner.deadline(None).unwrap();
    let Ok(objective) =
        authenticate_raw_objective_with_cap(text, owner.budget(), at, MAX_STAGED_PROPERTY_BYTES)
    else {
        return false;
    };
    let Ok((spec, input)) = ny_onnx::vnnlib::parse_vnnlib_with_certified_input_box(text) else {
        return false;
    };
    cross_check(&objective, &spec, &input).is_ok() && objective.inputs > 0 && objective.outputs > 1
}

/// The router's start-time archetype of `text`, read through the router's own
/// file read (`read_spec`), exactly as the census reads a corpus property.
fn router_archetype(text: &str) -> Option<ny_core::router::decl::SpecArchetype> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("property.vnnlib");
    std::fs::write(&path, text).unwrap();
    crate::commands::beta_crown::spec_shape::read_spec(&path).archetype()
}

/// CONTAINMENT: every property the route accepts is one the router reads as
/// argmax-margin robustness, so a census row refused `spec-not-served` (a
/// known half-space, disjunctive, non-linear or relational read) is a row the
/// route refuses too, and the census's spec scoping never under-scopes the
/// moat. Adversarial shapes: strict and mixed-direction atoms, singleton
/// `and` wrappers, a multi-atom clause, two targets, a threshold atom, a
/// conjunction split over assertions, per-clause input boxes, a missing class,
/// a top-level conjunction, larger-side and mixed-side atoms.
#[test]
fn prepared_root_property_admission_is_inside_the_routers_robustness_archetype() {
    use ny_core::router::decl::SpecArchetype;
    const HEAD3: &str = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n\
                         (declare-const Y_1 Real)\n(declare-const Y_2 Real)\n\
                         (assert (>= X_0 0))\n(assert (<= X_0 3))\n";
    let two = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
               (assert (>= X_0 0))\n(assert (<= X_0 3))\n(assert (<= Y_0 Y_1))\n"
        .to_owned();
    let fun = "(declare-fun X_0 () Real)\n(declare-fun Y_0 () Real)\n(declare-fun Y_1 () Real)\n\
               (declare-fun Y_2 () Real)\n(assert (>= X_0 0))\n(assert (<= X_0 3))\n\
               (assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))\n"
        .to_owned();
    let with = |body: &str| format!("{HEAD3}{body}\n");
    // (name, property, the route's expected answer: Some(accept) or None =
    // containment only)
    let cases: Vec<(&str, String, Option<bool>)> = vec![
        (
            "or-le",
            with("(assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))"),
            Some(true),
        ),
        (
            "or-strict-mixed",
            with("(assert (or (> Y_1 Y_0) (< Y_0 Y_2)))"),
            Some(true),
        ),
        (
            "or-and-singletons",
            with("(assert (or (and (>= Y_1 Y_0)) (and (>= Y_2 Y_0))))"),
            Some(true),
        ),
        ("fixture", property(3, 1), Some(true)),
        ("two-class-direct", two, None),
        ("declare-fun", fun, None),
        (
            "multi-atom-clause",
            with("(assert (or (and (<= Y_0 Y_1) (<= Y_0 Y_2))))"),
            Some(false),
        ),
        (
            "two-targets",
            with("(assert (or (<= Y_0 Y_1) (<= Y_1 Y_2)))"),
            Some(false),
        ),
        ("threshold", with("(assert (<= Y_0 0.5))"), Some(false)),
        (
            "split-asserts",
            with("(assert (<= Y_0 Y_1))\n(assert (<= Y_0 Y_2))"),
            Some(false),
        ),
        (
            "boxed-clauses",
            with(
                "(assert (or (and (>= X_0 0) (<= X_0 1) (<= Y_0 Y_1)) \
                 (and (>= X_0 1) (<= X_0 3) (<= Y_0 Y_2))))",
            ),
            Some(false),
        ),
        (
            "missing-class",
            with("(assert (or (<= Y_0 Y_1)))"),
            Some(false),
        ),
        (
            "top-and",
            with("(assert (and (<= Y_0 Y_1) (<= Y_0 Y_2)))"),
            Some(false),
        ),
        (
            "larger-side",
            with("(assert (or (>= Y_0 Y_1) (>= Y_0 Y_2)))"),
            Some(false),
        ),
        (
            "mixed-sides",
            with("(assert (or (<= Y_0 Y_1) (>= Y_0 Y_2)))"),
            Some(false),
        ),
    ];
    let (mut accepted, mut refused_outside) = (0usize, 0usize);
    for (name, text, want) in &cases {
        let route = route_accepts_property(text);
        let read = router_archetype(text);
        if let Some(want) = want {
            assert_eq!(
                route, *want,
                "{name}: the route's property gate moved\n{text}"
            );
        }
        if route {
            accepted += 1;
            assert_eq!(
                read,
                Some(SpecArchetype::Robustness),
                "{name}: the route accepts a property the router does not read as robustness, so \
                 the census would refuse (`spec-not-served`) a row the route runs on\n{text}"
            );
        } else if matches!(
            read,
            Some(SpecArchetype::Halfspace | SpecArchetype::Disjunctive)
        ) {
            refused_outside += 1;
        }
    }
    // Non-vacuity: the route accepted robustness shapes, and refused the
    // shapes the census refuses as not served.
    assert!(accepted >= 4, "accepted={accepted}");
    assert!(refused_outside >= 5, "refused_outside={refused_outside}");
}

/// CEILING: the route refuses a property longer than its staged ceiling by
/// the FILE LENGTH, before any parse, and that ceiling is below the router's
/// spec-read ceiling, so a census row whose spec read is `spec>16MiB` is a row
/// the route refuses (the keyless moat plan's superset tail leaves them out
/// on exactly this test).
#[test]
fn prepared_root_property_ceiling_is_below_the_router_spec_read_ceiling() {
    use super::super::original_property_census::MAX_STAGED_PROPERTY_BYTES;
    use crate::commands::beta_crown::spec_shape::SPEC_READ_MAX_BYTES;
    assert!(
        (MAX_STAGED_PROPERTY_BYTES as u64) < SPEC_READ_MAX_BYTES,
        "staged ceiling {MAX_STAGED_PROPERTY_BYTES} vs router read ceiling {SPEC_READ_MAX_BYTES}"
    );
    let (_dir, onnx, vnn) = files(3, 2, true, true);
    // Positive control: the same fixture proves at its own length.
    match run_fixture(&onnx, &vnn, resources(clock())) {
        PreparedRouteOutcome::Proven { report, .. } => {
            assert_eq!(report.status, PreparedRouteStatus::RootProven);
        }
        _ => panic!("the ceiling fixture must prove before it is padded"),
    }
    let mut text = property(3, 2);
    text.push_str(&" ".repeat(MAX_STAGED_PROPERTY_BYTES + 1 - text.len()));
    std::fs::write(&vnn, &text).unwrap();
    assert_eq!(
        std::fs::metadata(&vnn).unwrap().len(),
        (MAX_STAGED_PROPERTY_BYTES + 1) as u64
    );
    match run_fixture(&onnx, &vnn, resources(clock())) {
        PreparedRouteOutcome::Refused { report } => {
            assert_eq!(report.reason, "original-source-or-root-refused");
            assert_eq!(report.class_count, 0, "refused before the property parse");
            assert!(report.root.is_none());
        }
        _ => panic!("a property one byte over the staged ceiling must be refused"),
    }
}

#[test]
fn route_storage_observation_requires_preparation_and_keeps_default_row_storage_off() {
    let (_dir, onnx, vnn) = files(3, 2, true, true);
    let (_, policy) = defaults();
    assert!(PreparedRouteReceipt::initial(&policy).row_storage.is_none());
    let resource = resources(clock());
    let PreparedRouteOutcome::Proven { receipt, report } =
        run_fixture(&onnx, &vnn, resource.clone())
    else {
        panic!("actual original file root");
    };
    receipt.checkpoint_owner(None).unwrap();
    assert_eq!(
        report.row_storage,
        Some(PreparedRowStorageReceipt::default())
    );
    assert!(report.root.unwrap().reference.binary32_conv_dot_ops > 0);
    let text = format!("{report:?}");
    assert!(text.contains("row_storage: Some(PreparedRowStorageReceipt"));
    assert!(text.contains("payload_bytes_after_total: 0"));
    drop(receipt);
    drop(resource);
    let absent = onnx.with_file_name("missing-original.onnx");
    let PreparedRouteOutcome::Refused { report } = run_fixture(&absent, &vnn, resources(clock()))
    else {
        panic!("missing original file must refuse");
    };
    assert!(report.row_storage.is_none());
    assert!(report.root.is_none());
}
