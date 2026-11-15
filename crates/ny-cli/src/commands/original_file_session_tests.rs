// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine original snapshots and reusable work through open/proven outcomes.
//! No test creates a source authority from numerical planes or a proof token.

use super::super::unsat_receipt::UnsatReceipt;
use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{WorkLimits, WorkReserve},
    },
    margin_row::PreparedRootRetention,
};
use prost::Message;
use std::time::Duration;

const CAP: usize = 32 << 20;

fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
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
fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(positive: bool) -> ModelProto {
    let mut first = node("Gemm", &["flat", "w1", "b1"], "headpre");
    let mut last = node("Gemm", &["head", "w2", "b2"], "output");
    for gemm in [&mut first, &mut last] {
        gemm.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
    }
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![info("input", &[1, 1, 1, 1])],
            output: vec![info("output", &[1, 3])],
            initializer: vec![
                tensor("wc", &[1, 1, 1, 1], &[1.25]),
                tensor("bc", &[1], &[-0.25]),
                tensor("w1", &[2, 1], &[0.5, -0.25]),
                tensor("b1", &[2], &[0.75, 0.5]),
                tensor("w2", &[3, 2], &[0.5, -0.25, 0.75, 0.25, -0.25, 0.5]),
                tensor("b2", &[3], &[0.0, if positive { 2.0 } else { -2.0 }, 0.125]),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                first,
                node("Relu", &["headpre"], "head"),
                last,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property(target: usize, lo: &str, hi: &str) -> String {
    let classes: Vec<_> = (0..3).filter(|&c| c != target).rev().collect();
    format!("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(declare-const Y_2 Real)\n(assert (>= X_0 {lo}))\n(assert (<= X_0 {hi}))\n(assert (or (and (>= Y_{} Y_{target})) (< Y_{target} Y_{})))",classes[0],classes[1])
}
fn files(
    positive: bool,
    text: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("original.onnx");
    let vnn = dir.path().join("original.vnnlib");
    std::fs::write(&onnx, model(positive).encode_to_vec()).unwrap();
    std::fs::write(&vnn, text).unwrap();
    (dir, onnx, vnn)
}
fn prepare(
    onnx: &Path,
    vnn: &Path,
    budget: &WorkBudget,
    rows: bool,
) -> OriginalPreparedRootSession {
    OriginalPreparedRootSession::prepare_original_files_staged(
        onnx,
        vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows: rows,
            prefer_patch: false,
            ..Default::default()
        },
        budget,
        clock(),
    )
    .unwrap()
}
fn open(session: &OriginalPreparedRootSession) -> OriginalPreparedRootOpen {
    match session.try_prove_root(None).unwrap() {
        OriginalPreparedRootOutcome::Open(open) => open,
        OriginalPreparedRootOutcome::Proven(_) => panic!("genuine negative margin must stay open"),
    }
}
fn proven(session: &OriginalPreparedRootSession) -> OriginalPreparedRootPublication {
    match session.try_prove_root(None).unwrap() {
        OriginalPreparedRootOutcome::Proven(owner) => owner,
        OriginalPreparedRootOutcome::Open(_) => panic!("genuine positive root"),
    }
}
fn consumers() -> PreparedRootConsumerOptions {
    PreparedRootConsumerOptions {
        clip_rows: true,
        retain: Some(PreparedRootRetention {
            per_layer: 1,
            budget_bytes: 1 << 20,
        }),
        ..Default::default()
    }
}
// Independent checked integer arithmetic on the bounded dyadic fixture.
// This oracle is available in the no-mip lane and imports no production
// rounding/concretization helper. Its only floats are exact captured bits.
#[derive(Clone, Copy)]
struct Dyadic {
    numerator: i128,
    scale: u32,
}
impl Dyadic {
    fn new(mut numerator: i128, mut scale: u32) -> Self {
        if numerator == 0 {
            return Self {
                numerator: 0,
                scale: 0,
            };
        }
        let trailing = numerator.unsigned_abs().trailing_zeros().min(scale);
        numerator >>= trailing;
        scale -= trailing;
        Self { numerator, scale }
    }
    fn from_float(value: f64) -> Self {
        assert!(value.is_finite());
        if value == 0.0 {
            return Self::new(0, 0);
        }
        let bits = value.to_bits();
        let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap() - 1023 - 52;
        let mantissa = i128::from((bits & ((1u64 << 52) - 1)) | (1u64 << 52));
        let signed = if bits >> 63 == 0 { mantissa } else { -mantissa };
        assert!(
            (-110..=60).contains(&exponent),
            "bounded dyadic oracle fixture"
        );
        if exponent >= 0 {
            Self::new(
                signed
                    .checked_mul(1i128.checked_shl(exponent as u32).unwrap())
                    .unwrap(),
                0,
            )
        } else {
            Self::new(signed, (-exponent) as u32)
        }
    }
    fn aligned(self, scale: u32) -> i128 {
        self.numerator
            .checked_mul(1i128.checked_shl(scale - self.scale).unwrap())
            .unwrap()
    }
    fn add(self, rhs: Self) -> Self {
        let scale = self.scale.max(rhs.scale);
        Self::new(
            self.aligned(scale).checked_add(rhs.aligned(scale)).unwrap(),
            scale,
        )
    }
    fn sub(self, rhs: Self) -> Self {
        self.add(Self::new(-rhs.numerator, rhs.scale))
    }
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.numerator.checked_mul(rhs.numerator).unwrap(),
            self.scale.checked_add(rhs.scale).unwrap(),
        )
    }
    fn max_zero(self) -> Self {
        if self.numerator < 0 {
            Self::new(0, 0)
        } else {
            self
        }
    }
    fn positive(self) -> bool {
        self.numerator > 0
    }
    fn negative(self) -> bool {
        self.numerator < 0
    }
    fn le(self, rhs: Self) -> bool {
        let scale = self.scale.max(rhs.scale);
        self.aligned(scale) <= rhs.aligned(scale)
    }
}
fn rational(x: f64) -> Dyadic {
    Dyadic::from_float(x)
}
fn exact_margin(x: f64, class: usize, positive: bool) -> Dyadic {
    let trunk = rational(1.25)
        .mul(rational(x))
        .sub(rational(0.25))
        .max_zero();
    let h0 = rational(0.5).mul(trunk).add(rational(0.75)).max_zero();
    let h1 = rational(-0.25).mul(trunk).add(rational(0.5)).max_zero();
    let target = rational(if positive { 2.0 } else { -2.0 })
        .add(rational(0.75).mul(h0))
        .add(rational(0.25).mul(h1));
    let other = match class {
        0 => rational(0.5).mul(h0).sub(rational(0.25).mul(h1)),
        2 => rational(-0.25)
            .mul(h0)
            .add(rational(0.5).mul(h1))
            .add(rational(0.125)),
        _ => panic!("original target is class one"),
    };
    target.sub(other)
}

#[test]
fn original_file_session_open_retains_authentic_objective_box_and_actual_ordinary_work() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let before = live(&memory);
    let open = open(&session);
    assert_eq!(open.target(), 1);
    assert_eq!(open.classes(), [2, 0]);
    assert!(open.budget().shares_ledger(&budget));
    assert!(Arc::ptr_eq(&session.0, &open.0.original.0));
    assert!(open.attempt_receipt().ordinary_attempted && open.attempt_receipt().ordinary_completed);
    assert!(!open.attempt_receipt().terminal_closed && !open.attempt_receipt().ordinary_refused);
    for class in open.classes() {
        assert!(exact_margin(0.0, *class, false).negative());
    }
    open.checkpoint(None).unwrap();
    assert!(live(&memory) > before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((open, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_last_open_clone_keeps_source_and_claims_until_actual_last_owner() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let opened = open(&session);
    let alias = opened.clone();
    let held = live(&memory);
    drop((opened, session));
    assert_eq!(live(&memory), held);
    alias.checkpoint(None).unwrap();
    let original = alias.session().unwrap();
    drop(alias);
    assert!(live(&memory) > 0 && live(&memory) < held);
    original.checkpoint(None).unwrap();
    drop(original);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_reuse_after_file_deletion_spends_same_ledger_without_reload_or_recompile()
{
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let first = open(&session);
    let alias = first.session().unwrap();
    std::fs::remove_file(&onnx).unwrap();
    std::fs::remove_file(&vnn).unwrap();
    let before = budget.receipt().unwrap();
    let second = open(&alias);
    assert!(Arc::ptr_eq(&session.0, &second.0.original.0));
    assert_eq!(
        first.0.open.session().compiled_digest(),
        second.0.open.session().compiled_digest()
    );
    assert!(budget.receipt().unwrap().work_reserved > before.work_reserved);
    assert_eq!(
        budget.receipt().unwrap().proposals_reserved,
        before.proposals_reserved
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((first, second, session, alias));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_genuine_signed_ordinary_positive_enters_existing_private_receipt() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(true, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, true);
    let OriginalPreparedRootOutcome::Proven(owner) = session
        .try_prove_root_with_consumers(consumers(), None)
        .unwrap()
    else {
        panic!("positive genuine original ordinary route");
    };
    assert!(
        owner.attempt_receipt().ordinary_attempted && owner.attempt_receipt().ordinary_completed
    );
    assert!(!owner.attempt_receipt().ordinary_refused);
    assert_eq!(owner.target(), 1);
    assert_eq!(owner.classes(), [2, 0]);
    for x in [-1.0, 0.0, 0.2, 0.5, 1.0] {
        for &class in owner.classes() {
            let exact = exact_margin(x, class, true);
            assert!(exact.positive() && rational(owner.min_margin()).le(exact));
        }
    }
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    receipt.checkpoint_owner(None).unwrap();
    let held = receipt.original_prepared_authority().unwrap();
    assert!(Arc::ptr_eq(&held.0.original, &session.0));
    assert_eq!(held.model_sha256(), session.model_sha256());
    drop((held, receipt, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_cheap_positive_and_legacy_staged_api_preserve_exact_margin_and_order() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(true, &property(1, "0.1", "0.3"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let owner = proven(&session);
    assert!(owner.attempt_receipt().terminal_closed && !owner.attempt_receipt().ordinary_attempted);
    let legacy = OriginalPreparedRootPublication::prove_original_files_staged(
        &onnx,
        &vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            ..Default::default()
        },
        &budget,
        clock(),
    )
    .unwrap();
    assert_eq!(owner.min_margin().to_bits(), legacy.min_margin().to_bits());
    assert_eq!(owner.classes(), legacy.classes());
    assert_eq!(owner.model_sha256(), legacy.model_sha256());
    assert_eq!(owner.property_sha256(), legacy.property_sha256());
    drop((owner, legacy, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_open_and_proven_aliases_cannot_renew_successful_earlier_clocks() {
    let (budget, _, _) = resources();
    let (_d, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let early = Instant::now() + Duration::from_secs(30);
    let opened = open(&session).restricted(Some(early)).unwrap();
    assert!(opened.clone().restricted(Some(clock())).unwrap().deadline() <= early);
    let alias = opened.session().unwrap();
    assert!(alias.clone().restricted(None).unwrap().deadline() <= early);
    assert!(alias.checkpoint(Some(Instant::now())).is_err());
    let (_d, onnx, vnn) = files(true, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let owner = proven(&session).restricted(Some(early)).unwrap();
    let receipt = UnsatReceipt::from_original_prepared(owner, Some(clock())).unwrap();
    let authority = receipt.original_prepared_authority().unwrap();
    assert!(authority.deadline() <= early && authority.session().unwrap().deadline() <= early);
}

#[test]
fn original_file_session_cancelled_open_source_cannot_gain_new_authority_or_fresh_work_allowance() {
    let (budget, memory, cancel) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let opened = open(&session);
    let before = budget.receipt().unwrap();
    let held = live(&memory);
    cancel.cancel();
    assert!(opened.checkpoint(None).is_err());
    assert!(session.try_prove_root(None).is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(live(&memory), held);
    drop((opened, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_outcome_header_cap_refuses_before_new_attempt_and_preserves_completed_source(
) {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let need = outcome_owner_bytes().unwrap();
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - live(&memory) - need + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    let held = live(&memory);
    assert!(session.try_prove_root(None).is_err());
    assert_eq!(live(&memory), held);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(blocker);
    let opened = open(&session);
    opened.checkpoint(None).unwrap();
    drop((opened, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_mutated_paths_and_foreign_hashes_cannot_relabel_held_source() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(false, &property(1, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let model_hash = *session.model_sha256();
    let property_hash = *session.property_sha256();
    std::fs::write(&onnx, model(true).encode_to_vec()).unwrap();
    std::fs::write(&vnn, property(0, "0", "0")).unwrap();
    session
        .validate_asset_hashes(model_hash, property_hash, None)
        .unwrap();
    assert!(session
        .validate_asset_hashes([0; 32], property_hash, None)
        .is_err());
    let opened = open(&session);
    assert_eq!(opened.target(), 1);
    assert_eq!(session.0.model_bytes, model(false).encode_to_vec());
    drop((opened, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_source_unsafe_target_remains_open_without_caller_label_override() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(true, &property(0, "-1", "1"));
    let session = prepare(&onnx, &vnn, &budget, false);
    let opened = open(&session);
    assert_eq!(opened.target(), 0);
    assert_eq!(opened.classes(), [2, 1]);
    assert!(opened.attempt_receipt().ordinary_completed);
    drop((opened, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_actual_row_f32_request_keeps_original_compute_policy_and_objective() {
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(true, &property(1, "-1", "1"));
    let session = OriginalPreparedRootSession::prepare_original_files_staged(
        &onnx,
        &vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows: true,
            row_f32: true,
            prefer_patch: false,
            ..Default::default()
        },
        &budget,
        clock(),
    )
    .unwrap();
    let OriginalPreparedRootOutcome::Proven(owner) = session
        .try_prove_root_with_consumers(consumers(), None)
        .unwrap()
    else {
        panic!("actual stored-f32 original row request");
    };
    assert!(owner.attempt_receipt().ordinary_completed);
    assert!(!owner.0.proof.policy().root_f32 && !owner.0.proof.policy().tableau_f32);
    assert_eq!(owner.classes(), [2, 0]);
    assert!(rational(owner.min_margin()).le(exact_margin(0.0, 2, true)));
    drop((owner, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_session_expired_original_entry_refuses_before_any_file_or_owner_admission() {
    let (budget, memory, _) = resources();
    let before = budget.receipt().unwrap();
    assert!(OriginalPreparedRootSession::prepare_original_files_staged(
        Path::new("missing-original-model.onnx"),
        Path::new("missing-original-property.vnnlib"),
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        &budget,
        Instant::now(),
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), 0);
}
