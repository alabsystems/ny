// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original property geometry, same-ledger staged proof and downgrade guard.
//! These controls require genuine source parsers/producers, never raw boxes.

use super::super::prepared_root_publication::{
    streamed_spec_agreement, OriginalPreparedRootPublication,
};
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
    margin_row::{PreparedRootProofOptions, RootPrecisionRuntimePolicy},
};
use prost::Message;
use std::{path::Path, time::Duration};

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(cap: usize, work: u64) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: work,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
    (budget, memory, cancellation)
}
fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}
fn source_claim(text: &str, budget: &WorkBudget) -> PersistentReservation {
    budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: text.len() + META,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap()
}
fn property(target: usize, lower: &str, upper: &str) -> String {
    let classes: Vec<_> = (0..3).filter(|&c| c != target).rev().collect();
    format!("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(declare-const Y_2 Real)\n(assert (>= X_0 {lower}))\n(assert (<= X_0 {upper}))\n(assert (or (and (>= Y_{} Y_{target})) (<= Y_{target} Y_{})))", classes[0], classes[1])
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
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(target: usize, positive: bool) -> ModelProto {
    let mut first = node("Gemm", &["flat", "w1", "b1"], "headpre");
    first.attribute.push(AttributeProto {
        name: "transB".into(),
        r#type: 2,
        i: Some(1),
        ..Default::default()
    });
    let mut final_node = node("Gemm", &["head", "w2", "b2"], "output");
    final_node.attribute = first.attribute.clone();
    let mut bias = [0f32; 3];
    bias[target] = if positive { 2.0 } else { -2.0 };
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
                tensor("wc", &[1, 1, 1, 1], &[1.0]),
                tensor("bc", &[1], &[0.0]),
                tensor("w1", &[2, 1], &[1.0, -1.0]),
                tensor("b1", &[2], &[0.0; 2]),
                tensor("w2", &[3, 2], &[0.0; 6]),
                tensor("b2", &[3], &bias),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                first,
                node("Relu", &["headpre"], "head"),
                final_node,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn files(
    source: &ModelProto,
    property: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("original.onnx");
    let vnn = dir.path().join("original.vnnlib");
    std::fs::write(&model, source.encode_to_vec()).unwrap();
    std::fs::write(&vnn, property).unwrap();
    (dir, model, vnn)
}
fn prove(
    model: &Path,
    property: &Path,
    budget: &WorkBudget,
) -> Result<OriginalPreparedRootPublication> {
    OriginalPreparedRootPublication::prove_original_files_staged(
        model,
        property,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        at(),
    )
}

#[test]
fn staged_original_property_actual_cifar100_339544_bytes_retains_complete_objective_and_exact_box()
{
    let path = Path::new("<home>/ny/benchmarks/vnncomp2025/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_54_sidx_9735_eps_0.0039.vnnlib");
    let text = std::fs::read_to_string(path).expect("qualification requires the unchanged actual CIFAR100 original property; no skip or replacement");
    assert_eq!(text.len(), 339544);
    assert_eq!(
        format!("{:x}", Sha256::digest(text.as_bytes())),
        "eaa92f0026004640ee28abf6d294a826f528269386c4a52366e65a75ed0fe635"
    );
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let claim = source_claim(&text, &budget);
    let staged = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
    drop(claim);
    let (property, raw_claim, parsed_claim, receipt, _) = staged.into_parts();
    assert_eq!(receipt.input_count, 3072);
    assert_eq!(receipt.output_count, 100);
    assert_eq!(property.objective.target, 54);
    assert_eq!(
        property.objective.classes,
        (0..100).filter(|&c| c != 54).collect::<Vec<_>>()
    );
    assert_eq!(property.spec.output_constraint_clauses.len(), 99);
    assert!(property
        .spec
        .output_constraint_clauses
        .iter()
        .all(|c| c.len() == 1));
    let oracle_census = census(&property.source, &budget, at()).unwrap();
    let mut oracle_lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: receipt.parser_scratch_capacity_bound
                    + receipt.retained_capacity_bound,
                work: oracle_census.work().unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let (reference, certified) = parse_vnnlib_with_certified_input_box(&property.source).unwrap();
    assert_eq!(property.spec.input_bounds, reference.input_bounds);
    assert_eq!(
        property.spec.output_constraints,
        reference.output_constraints
    );
    assert_eq!(
        property.spec.output_constraint_clauses,
        reference.output_constraint_clauses
    );
    assert_eq!(property.input.lower, certified.lower());
    assert_eq!(property.input.upper, certified.upper());
    drop(reference);
    drop(certified);
    oracle_lease.complete().unwrap();
    drop(oracle_lease);
    assert!(receipt.parser_scratch_capacity_bound < 32 << 20);
    assert!(receipt.retained_visible_bytes <= receipt.retained_capacity_bound);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().proposals_reserved, 0);
    assert!(raw_claim.shares_ledger(&budget) && parsed_claim.shares_ledger(&budget));
    drop(property);
    drop(raw_claim);
    drop(parsed_claim);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_all_200_actual_cifar100_grammars_are_censused_without_size_multiplier()
{
    let dir = Path::new("<home>/ny/benchmarks/vnncomp2025/benchmarks/cifar100_2024/vnnlib");
    let paths: Vec<_> = std::fs::read_dir(dir)
        .expect("actual CIFAR100 originals required")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vnnlib"))
        .collect();
    assert_eq!(
        paths.len(),
        200,
        "qualification requires every installed original property"
    );
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    for path in paths {
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.len() > 64 << 10 && text.len() <= MAX_STAGED_PROPERTY_BYTES);
        let census = census(&text, &budget, at()).unwrap();
        assert_eq!((census.inputs, census.outputs), (3072, 100));
        assert!(
            census.scratch().unwrap() + census.retained().unwrap() + text.len() + META < 32 << 20
        );
    }
    assert_eq!(live(&memory), 0);
}

fn historical_property_work(c: Census) -> u64 {
    let limbs = (16 * (c.max_digits + c.max_scale) + 2200).div_ceil(32);
    u64::try_from(
        128 * c.bytes
            + (c.decimals + c.inputs) * limbs * limbs * 256
            + c.outputs * c.outputs * 32
            + META,
    )
    .unwrap()
}

#[test]
fn staged_original_property_actual_cifar_work_quote_prices_real_literals_and_coordinates() {
    let path = Path::new("<home>/ny/benchmarks/vnncomp2025/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_54_sidx_9735_eps_0.0039.vnnlib");
    let text = std::fs::read_to_string(path).expect("unchanged actual CIFAR property required");
    assert_eq!(text.len(), 339544);
    assert_eq!(
        format!("{:x}", Sha256::digest(text.as_bytes())),
        "eaa92f0026004640ee28abf6d294a826f528269386c4a52366e65a75ed0fe635"
    );
    // Same limits as the existing actual-property control; this is parser
    // phase evidence, never a claim that a whole CIFAR root has completed.
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let c = census(&text, &budget, at()).unwrap();
    assert_eq!((c.inputs, c.outputs, c.decimals), (3072, 100, 6144));
    assert!(c.work().unwrap() < historical_property_work(c) / 2);
    assert!(c.literal_work > 0 && c.comparison_work > 0 && c.conversion_work > 0);
    let claim = source_claim(&text, &budget);
    let before = budget.receipt().unwrap().work_reserved;
    let parsed = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
    assert!(budget.receipt().unwrap().work_reserved - before < historical_property_work(c) / 2);
    let (property, raw, parsed_claim, receipt, _) = parsed.into_parts();
    assert_eq!(property.objective.target, 54);
    assert_eq!(receipt.input_count, 3072);
    assert_eq!(
        property.objective.classes,
        (0..100).filter(|&a| a != 54).collect::<Vec<_>>()
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(property);
    drop(raw);
    drop(parsed_claim);
    drop(claim);
    assert_eq!(live(&memory), 0);
}

fn many_coordinate_property(inputs: usize, first_lower: &str, extra_bounds: usize) -> String {
    use std::fmt::Write;
    let mut text = String::new();
    for i in 0..inputs {
        writeln!(text, "(declare-const X_{i} Real)").unwrap();
    }
    for i in 0..3 {
        writeln!(text, "(declare-const Y_{i} Real)").unwrap();
    }
    for i in 0..inputs {
        writeln!(
            text,
            "(assert (>= X_{i} {}))",
            if i == 0 { first_lower } else { "0" }
        )
        .unwrap();
        writeln!(text, "(assert (<= X_{i} 1))").unwrap();
    }
    for _ in 0..extra_bounds {
        writeln!(text, "(assert (>= X_0 0))").unwrap();
    }
    text.push_str("(assert (or (and (>= Y_2 Y_1)) (<= Y_1 Y_0)))");
    text
}

#[test]
fn staged_original_property_one_high_scale_coordinate_does_not_tax_unrelated_coordinates() {
    // The actual unchanged source grammar admits scale324 and refuses325.
    validate_literal("1e-324").unwrap();
    assert!(validate_literal("1e-325").is_err());
    let text = many_coordinate_property(128, "1e-324", 0);
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let c = census(&text, &budget, at()).unwrap();
    assert_eq!(c.max_scale, 324);
    assert!(c.work().unwrap() < historical_property_work(c) / 8);
    // This authored snapshot grows through writeln!, so its retained capacity
    // can exceed len()+META. Pay the actual moved String on the same ledger;
    // the parser's source-ownership guard must remain unchanged.
    let claim = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: text.capacity().checked_add(META).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    assert!(claim.shares_ledger(&budget) && claim.bytes() >= text.capacity());
    let parsed = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
    let (property, raw, parsed_claim, _, _) = parsed.into_parts();
    // At the maximum admitted scale, exact1e-324 lies below half the smallest
    // subnormal and genuinely encloses outward at zero. Other coordinates
    // keep their authored [0,1].
    assert_eq!(property.input.lower, vec![0.0; 128]);
    assert_eq!(property.input.upper, vec![1.0; 128]);
    assert_eq!(property.objective.classes, [2, 0]);
    drop(property);
    drop(raw);
    drop(parsed_claim);
    drop(claim);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_late_short_bounds_preserve_paid_wide_comparison_cost() {
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let original = census(&many_coordinate_property(4, "1e-324", 0), &budget, at()).unwrap();
    let repeated = census(&many_coordinate_property(4, "1e-324", 64), &budget, at()).unwrap();
    let comparison = rational_work(
        DecimalWidth {
            digits: 1,
            scale: 324,
        },
        64,
    )
    .unwrap();
    assert_eq!(repeated.conversion_work, original.conversion_work);
    assert_eq!(
        repeated.comparison_work - original.comparison_work,
        comparison * 64
    );
    assert!(repeated.work().unwrap() > original.work().unwrap());
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_cost_metadata_refuses_at_its_paid_capacity_frontier() {
    let text = property(1, "0.1", "0.3");
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let claim = source_claim(&text, &budget);
    let occupied = budget.receipt().unwrap().live_persistent_bytes;
    let metadata = META + size_of::<DecimalWidth>();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: (32 << 20) - occupied - metadata + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let before = budget.receipt().unwrap();
    assert!(census(&text, &budget, at()).is_err());
    let after = budget.receipt().unwrap();
    assert!(after.refused > before.refused);
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    drop(blocker);
    let parsed = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
    drop(parsed);
    drop(claim);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_one_mib_grammar_does_not_grant_memory_or_work_credit() {
    let text = property(1, "0.1", "0.3");
    let (small, memory, _) = resources(24 << 10, 100_000_000_000);
    let claim = source_claim(&text, &small);
    let before = small.receipt().unwrap();
    assert!(parse_staged_original_property(text.clone(), &claim, &small, at()).is_err());
    assert!(small.receipt().unwrap().refused > before.refused);
    assert_eq!(small.receipt().unwrap().live_scratch_bytes, 0);
    drop(claim);
    assert_eq!(live(&memory), 0);
    let (work_limited, memory, _) =
        resources(32 << 20, u64::try_from(text.len() * 32 + META).unwrap());
    let claim = source_claim(&text, &work_limited);
    assert!(parse_staged_original_property(text, &claim, &work_limited, at()).is_err());
    assert_eq!(work_limited.receipt().unwrap().proposals_reserved, 0);
    drop(claim);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_adversarial_atom_count_depth_sparse_ids_and_decimal_scale_refuse() {
    let good = property(1, "0.1", "0.3");
    let cases = [
        good.replace("(<= Y_1 Y_0)", "(<= Y_1 Y_0) (<= Y_1 Y_0)"),
        good.replace("(>= Y_2 Y_1)", "(and (and (and (and (>= Y_2 Y_1)))))"),
        good.replace("X_0", "X_1000000000"),
        good.replace("0.1", "1e-325"),
        good.replace("0.1", &"1".repeat(65)),
        good.replace("(<= Y_1 Y_0)", "(<= (+ Y_1 0) Y_0)"),
        good.replace("(<= Y_1 Y_0)", "(<= Y_1 Y_2)"),
    ];
    for text in cases {
        let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
        let claim = source_claim(&text, &budget);
        assert!(parse_staged_original_property(text, &claim, &budget, at()).is_err());
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        drop(claim);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn staged_original_property_size_limit_and_cancel_deadline_fail_before_parser_admission() {
    let (budget, memory, cancel) = resources(32 << 20, 100_000_000_000);
    let text = " ".repeat(MAX_STAGED_PROPERTY_BYTES + 1);
    let before = budget.receipt().unwrap();
    assert!(census(&text, &budget, at()).is_err());
    assert_eq!(budget.receipt().unwrap(), before);
    let text = property(1, "0.1", "0.3");
    assert!(census(&text, &budget, Instant::now()).is_err());
    cancel.cancel();
    assert!(census(&text, &budget, at()).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_foreign_snapshot_ledger_and_last_source_owner_remain_binding() {
    let text = property(1, "0.1", "0.3");
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let (foreign, _, _) = resources(32 << 20, 100_000_000_000);
    let foreign_claim = source_claim(&text, &foreign);
    assert!(parse_staged_original_property(text.clone(), &foreign_claim, &budget, at()).is_err());
    let claim = source_claim(&text, &budget);
    let staged = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
    drop(claim);
    assert!(
        live(&memory) > 0,
        "staged owner retains actual source and parsed claims"
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(staged);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_property_signed_zero_subnormal_and_non_dyadic_endpoints_match_genuine_certificate(
) {
    for (lo, hi) in [
        ("-0.0", "+0.0"),
        ("-5e-324", "5e-324"),
        ("-0.1", "0.3"),
        ("0.1", "0.1"),
    ] {
        let text = property(1, lo, hi);
        let (_spec, certificate) = parse_vnnlib_with_certified_input_box(&text).unwrap();
        let (budget, _, _) = resources(32 << 20, 100_000_000_000);
        let claim = source_claim(&text, &budget);
        let staged = parse_staged_original_property(text, &claim, &budget, at()).unwrap();
        assert_eq!(
            staged.property.input.lower[0].to_bits(),
            certificate.lower()[0].to_bits()
        );
        assert_eq!(
            staged.property.input.upper[0].to_bits(),
            certificate.upper()[0].to_bits()
        );
        if lo == "-0.1" {
            // Independent signed integer cross multiplication, no production
            // rounding helpers and no rounded decimal multiplication.
            let compare = |value: f64, numerator: i128| {
                let bits = value.to_bits();
                let mantissa = i128::from((1u64 << 52) | (bits & ((1u64 << 52) - 1)));
                let signed = if value.is_sign_negative() {
                    -mantissa
                } else {
                    mantissa
                };
                let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap() - 1023 - 52;
                assert!((-100..=0).contains(&exponent));
                (signed * 10).cmp(&(numerator * (1i128 << -exponent)))
            };
            assert_ne!(
                compare(staged.property.input.lower[0], -1),
                std::cmp::Ordering::Greater
            );
            assert_ne!(
                compare(staged.property.input.upper[0], 3),
                std::cmp::Ordering::Less
            );
        }
    }
}

#[test]
fn staged_original_files_actual_positive_retains_source_order_and_releases_all_parser_decoder_scratch(
) {
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "0.1", "0.3"));
    let mut observed = false;
    let owner = OriginalPreparedRootPublication::prove_staged_inner(
        &onnx,
        &vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        &budget,
        at(),
        Some(&mut |budget| {
            observed = true;
            assert_eq!(
                budget.receipt().unwrap().live_scratch_bytes,
                0,
                "ceased property/decoder scratch must be gone before construction"
            );
            assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
        }),
    )
    .unwrap();
    assert!(observed);
    assert_eq!(owner.target(), 1);
    assert_eq!(owner.classes(), [2, 0]);
    assert!(owner.min_margin() > 1.99);
    assert!(owner.property_allocation_receipt().is_some());
    owner.checkpoint(None).unwrap();
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    let clone = owner.clone();
    let held = live(&memory);
    drop(owner);
    assert_eq!(live(&memory), held);
    drop(clone);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_files_after_compilation_cancellation_and_unsafe_objective_publish_nothing() {
    let (budget, memory, cancel) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "-1", "1"));
    assert!(OriginalPreparedRootPublication::prove_staged_inner(
        &onnx,
        &vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        &budget,
        at(),
        Some(&mut |_| cancel.cancel())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(0, "-1", "1"));
    assert!(prove(&onnx, &vnn, &budget).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_files_property_parsing_completes_before_distinct_model_snapshot_admission() {
    let mut source = model(1, true);
    source.doc_string = "p".repeat(1 << 20);
    let (_dir, onnx, vnn) = files(&source, &property(1, "0.1", "0.3"));
    let (budget, memory, _) = resources(256 << 10, 100_000_000_000);
    let before = budget.receipt().unwrap();
    assert!(prove(&onnx, &vnn, &budget).is_err());
    let after = budget.receipt().unwrap();
    // Actual property snapshot, census and genuine parser stages complete
    // before the model's one MiB snapshot claim refuses. Paying that future
    // model reservation during property parsing would erase these arrivals.
    assert!(after.completed >= before.completed + 3);
    assert!(after.refused > before.refused);
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, 0);
    assert_eq!(after.proposals_reserved, 0);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_files_earlier_exported_output_cannot_authorize_unused_positive_tail() {
    let mut source = model(1, true);
    let graph = source.graph.as_mut().unwrap();
    graph.output[0] = info("trunk", &[1, 1, 1, 1]);
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&source, &property(1, "-1", "1"));
    assert!(prove(&onnx, &vnn, &budget).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_g3_stream_matches_existing_positive_negative_strict_and_malformed_objectives() {
    let (budget, _, _) = resources(32 << 20, 100_000_000_000);
    let ordinary = ny_onnx::vnnlib::parse_vnnlib(&property(1, "-1", "1")).unwrap();
    let strict = ny_onnx::vnnlib::parse_vnnlib(
        &property(1, "-1", "1")
            .replace(">= Y_2", "> Y_2")
            .replace("<= Y_1", "< Y_1"),
    )
    .unwrap();
    let mut conjunction = ordinary.clone();
    conjunction.is_disjunction = false;
    let mut clauses = ordinary.clone();
    clauses.per_clause_input_bounds.push(BTreeMap::new());
    for spec in [&ordinary, &strict, &conjunction, &clauses] {
        for (target, classes) in [
            (1, vec![2, 0]),
            (0, vec![1, 2]),
            (1, vec![2]),
            (1, vec![2, 2]),
            (1, vec![9]),
        ] {
            let old = super::super::margin_row_bab::margin_row_spec_agreement_owned(
                spec, target, &classes,
            );
            let new = streamed_spec_agreement(spec, target, &classes, &budget, at()).unwrap();
            assert_eq!(old.0, new.report);
            assert_eq!(old.1, new.fragment);
        }
    }
}

#[test]
fn staged_original_g3_workspace_is_linear_and_refuses_without_paid_cap_or_deadline() {
    let text = property(1, "-1", "1");
    let spec = ny_onnx::vnnlib::parse_vnnlib(&text).unwrap();
    let (budget, memory, _) = resources(9 << 10, 100_000_000_000);
    let held = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: 2 << 10,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    assert!(streamed_spec_agreement(&spec, 1, &[2, 0], &budget, at()).is_err());
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(held);
    assert_eq!(live(&memory), 0);
    let (budget, _, _) = resources(32 << 20, 100_000_000_000);
    assert!(streamed_spec_agreement(&spec, 1, &[2, 0], &budget, Instant::now()).is_err());
    let before = budget.receipt().unwrap();
    let agreement = streamed_spec_agreement(&spec, 1, &[2, 0], &budget, at()).unwrap();
    let after = budget.receipt().unwrap();
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    assert!(
        after.live_scratch_bytes > 0,
        "G3 report remains paid through its actual consumer"
    );
    drop(agreement);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn staged_original_g3_variable_invalid_class_diagnostic_is_paid_before_formatting_and_retained() {
    let spec = ny_onnx::vnnlib::parse_vnnlib(&property(1, "-1", "1")).unwrap();
    let invalid = vec![usize::MAX; 2000];
    let (small, memory, _) = resources(24 << 10, 100_000_000_000);
    let before = small.receipt().unwrap();
    assert!(streamed_spec_agreement(&spec, 1, &invalid, &small, at()).is_err());
    let after = small.receipt().unwrap();
    assert_eq!(
        after.completed,
        before.completed + 1,
        "only the paid class/capacity scan completes"
    );
    assert!(after.refused > before.refused);
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources(256 << 10, 100_000_000_000);
    let legacy = super::super::margin_row_bab::margin_row_spec_agreement_owned(&spec, 1, &invalid);
    let result = streamed_spec_agreement(&spec, 1, &invalid, &budget, at()).unwrap();
    assert_eq!(result.report, legacy.0);
    assert_eq!(result.fragment, legacy.1);
    let actual_strings = result.fragment.capacity()
        + result
            .report
            .trips()
            .iter()
            .map(|t| t.detail.capacity())
            .sum::<usize>();
    assert!(actual_strings > META);
    assert!(budget.receipt().unwrap().live_scratch_bytes >= actual_strings);
    drop(result);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_files_invalid_utf8_error_does_not_retain_unpaid_snapshot_bytes() {
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "-1", "1"));
    std::fs::write(&vnn, vec![0xffu8; 200_000]).unwrap();
    let (budget, memory, _) = resources(2 << 20, 100_000_000_000);
    let error = prove(&onnx, &vnn, &budget).expect_err("non-UTF8 original property must refuse");
    assert_eq!(
        error.to_string(),
        "staged original property snapshot must be UTF-8"
    );
    assert!(
        error.downcast_ref::<std::string::FromUtf8Error>().is_none(),
        "returned failure cannot retain the original Vec after its snapshot lease released"
    );
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(error);
}

#[test]
fn staged_original_files_restricted_owner_clone_preserves_successful_earlier_handoff_clock() {
    let (budget, _, _) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&model(2, true), &property(2, "-1", "1"));
    let owner = prove(&onnx, &vnn, &budget).unwrap();
    let original = owner.deadline();
    let earlier = Instant::now() + Duration::from_secs(10);
    let restricted = owner.restricted(Some(earlier)).unwrap();
    assert!(restricted.deadline() < original);
    assert_eq!(restricted.deadline(), earlier);
    let cloned = restricted.clone().restricted(None).unwrap();
    assert_eq!(cloned.deadline(), earlier);
    assert_eq!(cloned.model_sha256(), restricted.model_sha256());
    assert!(cloned.restricted(Some(Instant::now())).is_err());
}

#[test]
fn staged_original_files_repeated_checkpoint_class_and_endpoint_scan_spends_original_work() {
    let (budget, memory, _) = resources(32 << 20, 100_000_000_000);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "0.1", "0.3"));
    let owner = prove(&onnx, &vnn, &budget).unwrap();
    let original_deadline = owner.deadline();
    let before = budget.receipt().unwrap();
    owner.checkpoint(None).unwrap();
    owner.checkpoint(None).unwrap();
    let after = budget.receipt().unwrap();
    // Original input has one endpoint pair and two ordered class IDs. Each
    // own scan is admitted before checking and the downstream objective
    // validator remains independently paid on this exact original ledger.
    assert!(after.work_reserved >= before.work_reserved + 2 * ((1 + 2 * 2) * 32 + 8192));
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(owner.deadline(), original_deadline);
    drop(owner);
    assert_eq!(live(&memory), 0);
}
