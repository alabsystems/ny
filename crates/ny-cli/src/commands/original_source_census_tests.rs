// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_onnx::onnx_proto::{tensor_shape_proto::Dimension, OperatorSetIdProto};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{WorkLimits, WorkReserve},
    },
    margin_row::{PreparedRootProofOptions, RootPrecisionRuntimePolicy},
};
use std::time::Duration;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
    (budget, memory, cancel)
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
fn tensor(name: &str, dims: &[i64], values: &[f32], raw: bool) -> TensorProto {
    let mut tensor = TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        ..Default::default()
    };
    if raw {
        tensor.raw_data = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
    } else {
        tensor.float_data = values.to_vec();
    }
    tensor
}
fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(hidden: usize, raw: bool) -> ModelProto {
    let mut first = node("Gemm", &["flat", "w1", "b1"], "pre");
    first.attribute.push(AttributeProto {
        name: "transB".into(),
        r#type: 2,
        i: Some(1),
        ..Default::default()
    });
    let mut final_node = node("Gemm", &["head", "w2", "b2"], "output");
    final_node.attribute.push(AttributeProto {
        name: "transB".into(),
        r#type: 2,
        i: Some(1),
        ..Default::default()
    });
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
                tensor("wc", &[1, 1, 1, 1], &[1.25], raw),
                tensor("bc", &[1], &[-0.25], raw),
                tensor("w1", &[hidden as i64, 1], &vec![0.5; hidden], raw),
                tensor("b1", &[hidden as i64], &vec![0.25; hidden], raw),
                tensor("w2", &[3, hidden as i64], &vec![0.0; 3 * hidden], raw),
                tensor("b2", &[3], &[0.0, -0.25, 2.0], raw),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                first,
                node("Relu", &["pre"], "head"),
                final_node,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn scalar(out: &mut Vec<u8>, tag: u32, value: u64) {
    varint(out, u64::from(tag) << 3);
    varint(out, value);
}
fn message(out: &mut Vec<u8>, tag: u32, bytes: &[u8]) {
    varint(out, (u64::from(tag) << 3) | 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}
fn paid_visible(source: &[u8], budget: &WorkBudget, at: Instant) -> (Census, usize) {
    let census = census_source(source, budget, at).unwrap();
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: census.decoded_bytes + source.len() + META,
                work: charged(source.len(), 8192).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let decoded = ModelProto::decode(source).unwrap();
    let actual = decoded_visible_bytes(&decoded, budget, at).unwrap();
    assert!(actual <= census.decoded_bytes);
    drop(decoded);
    lease.complete().unwrap();
    (census, actual)
}

#[test]
fn staged_original_source_compact_actual_model_fits_quota_rejected_by_blanket_and_finishes_same_parameters(
) {
    let (budget, memory, _) = resources(4 << 20);
    let source = model(1024, true).encode_to_vec();
    assert!(source.len() * 512 > 4 << 20);
    let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
    let receipt = staged.receipt();
    assert!(receipt.parameter_cells > 5000);
    assert!(receipt.decoded_visible_bytes > 0 && receipt.extracted_visible_bytes > 0);
    assert!(receipt.decode_scratch_bytes < 4 << 20);
    assert_eq!(
        *staged.source_sha256(),
        <[u8; 32]>::from(Sha256::digest(&source))
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(live(&memory), staged.compiled().resident_bytes());
    let source_identity = *staged.compiled().source_digest();
    let actual_identity = *staged.compiled().compiled_digest();
    let compiled = staged.into_compilation(&budget, Some(at())).unwrap();
    let proof = compiled
        .prove_declared_domain(
            &[-1.0],
            &[1.0],
            2,
            &[1, 0],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions::default(),
            &budget,
            Some(at()),
        )
        .unwrap();
    assert_eq!(*proof.source_digest(), source_identity);
    assert_eq!(*proof.compiled_digest(), actual_identity);
    // Independent original-output control: every final weight is authored
    // exact zero, so y=[0,-1/4,2] for every declared original input.
    assert!(proof.min_margin() > 1.99 && proof.min_margin() <= 2.0);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_typed_and_raw_parameters_reconcile_actual_capacities_and_exact_declaration(
) {
    let (budget, memory, _) = resources(32 << 20);
    let raw = model(8, true).encode_to_vec();
    let typed = model(8, false).encode_to_vec();
    let a = compile_staged_original_source(&raw, &budget, at()).unwrap();
    let b = compile_staged_original_source(&typed, &budget, at()).unwrap();
    assert_eq!(a.compiled().source_digest(), b.compiled().source_digest());
    assert_eq!(
        a.compiled().compiled_digest(),
        b.compiled().compiled_digest()
    );
    assert_ne!(a.source_sha256(), b.source_sha256());
    for receipt in [a.receipt(), b.receipt()] {
        assert!(receipt.decoded_visible_bytes <= receipt.decoded_capacity_bound);
        assert!(receipt.extracted_visible_bytes <= receipt.extraction_capacity_bound);
        assert!(receipt.parameter_cells > 0);
    }
    drop((a, b));
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_header_heavy_geometry_is_priced_by_messages_not_only_numeric_bytes() {
    let (budget, _, _) = resources(32 << 20);
    let heavy = ModelProto {
        graph: Some(GraphProto {
            node: vec![NodeProto::default(); 1025],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();
    let compact = model(1024, true).encode_to_vec();
    let (header, actual_header) = paid_visible(&heavy, &budget, at());
    let (parameters, _) = paid_visible(&compact, &budget, at());
    assert_eq!(header.nodes, 1025);
    assert_eq!(header.parameter_cells, 0);
    assert!(actual_header > heavy.len() * 20);
    assert!(header.decoded_bytes > parameters.decoded_bytes);
    assert!(header.extraction_bytes().unwrap() > parameters.extraction_bytes().unwrap());
}

#[test]
fn staged_original_source_actual_cifar_work_quotes_follow_wire_and_parameter_references() {
    let base = std::path::Path::new(
        "<home>/ny/benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx",
    );
    let (budget, memory, _) = resources(32 << 20);
    for (name, expected_bytes, expected_cells, expected_nodes, hash) in [
        (
            "CIFAR100_resnet_medium.onnx",
            10156168,
            2536344,
            59,
            "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        ),
        (
            "CIFAR100_resnet_large.onnx",
            15243961,
            3808152,
            61,
            "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        ),
    ] {
        let source =
            std::fs::read(base.join(name)).expect("unchanged actual original CIFAR model required");
        assert_eq!(source.len(), expected_bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&source)), hash);
        let before = budget.receipt().unwrap().work_reserved;
        let c = census_source(&source, &budget, at()).unwrap();
        let scan_work = budget.receipt().unwrap().work_reserved - before;
        assert_eq!(
            (c.parameter_cells, c.nodes),
            (expected_cells, expected_nodes)
        );
        assert_eq!(c.materialization_cells, c.parameter_cells);
        assert!(scan_work < charged(source.len(), 256).unwrap() / 8);
        assert!(
            c.production_work(source.len()).unwrap() < charged(source.len(), 8192).unwrap() / 32
        );
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
    // This is admission/cost metadata, not a whole-model/root solve receipt.
    assert_eq!(live(&memory), 0);
}

fn reused_conv_source(repeats: usize) -> Vec<u8> {
    let mut original = model(2, true);
    let graph = original.graph.as_mut().unwrap();
    graph.input = vec![info("input", &[1, 128, 4, 4])];
    let head_weight = graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "w1")
        .unwrap();
    *head_weight = tensor("w1", &[2, 2048], &vec![0.5; 4096], true);
    graph
        .initializer
        .retain(|t| !matches!(t.name.as_str(), "wc" | "bc"));
    graph.initializer.push(tensor(
        "shared_conv",
        &[128, 128, 3, 3],
        &vec![0.0; 128 * 128 * 9],
        true,
    ));
    let mut nodes = Vec::new();
    let mut previous = "input".to_string();
    for i in 0..repeats {
        let output = format!("shared_output_{i}");
        let mut conv = NodeProto {
            op_type: "Conv".into(),
            input: vec![previous, "shared_conv".into()],
            output: vec![output.clone()],
            ..Default::default()
        };
        conv.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        nodes.push(conv);
        previous = output;
    }
    // Retain the actual final Gemm/Relu/Gemm and property-shaped head.
    let first_gemm = graph.node.iter().position(|n| n.op_type == "Gemm").unwrap();
    nodes.push(NodeProto {
        op_type: "Relu".into(),
        input: vec![previous],
        output: vec!["trunk".into()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: "Flatten".into(),
        input: vec!["trunk".into()],
        output: vec!["flat".into()],
        ..Default::default()
    });
    nodes.extend(graph.node[first_gemm..].iter().cloned());
    graph.node = nodes;
    original.encode_to_vec()
}

#[test]
fn staged_original_source_valid_reused_conv_parameters_refuse_before_extraction_at_original_cap() {
    let source = reused_conv_source(64);
    let (budget, memory, _) = resources(32 << 20);
    let c = census_source(&source, &budget, at()).unwrap();
    assert!(c.materialization_cells > c.parameter_cells * 50);
    assert!(c.extraction_bytes().unwrap() > 32 << 20);
    // Independently authenticate this real graph under its own paid decoded
    // envelope; refusal must arise from repeated materialization, not syntax.
    let mut guard = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: c.decoded_bytes + META,
                work: c.production_work(source.len()).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    super::super::prepared_root_source::authenticate_original_twin_surface(&source, &budget, at())
        .unwrap();
    guard.complete().unwrap();
    drop(guard);
    let before = live(&memory);
    let mut decoded = false;
    let mut observe = || decoded = true;
    assert!(compile_inner(&source, &budget, at(), Some(&mut observe)).is_err());
    assert!(
        !decoded,
        "complete reused-parameter capacity must refuse before decoding/extraction"
    );
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn staged_original_source_constant_shape_references_are_priced_per_actual_use() {
    let mut original = model(8, true);
    let graph = original.graph.as_mut().unwrap();
    let flatten = graph
        .node
        .iter()
        .position(|n| n.op_type == "Flatten")
        .unwrap();
    let previous = graph.node[flatten].input[0].clone();
    let constant = NodeProto {
        op_type: "Constant".into(),
        output: vec!["shared_shape".into()],
        attribute: vec![AttributeProto {
            name: "value".into(),
            r#type: 4,
            t: Some(TensorProto {
                data_type: 7,
                dims: vec![2],
                int64_data: vec![1, 1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    graph.node.splice(
        flatten..=flatten,
        [
            constant,
            NodeProto {
                op_type: "Reshape".into(),
                input: vec![previous, "shared_shape".into()],
                output: vec!["reshaped_once".into()],
                ..Default::default()
            },
            NodeProto {
                op_type: "Reshape".into(),
                input: vec!["reshaped_once".into(), "shared_shape".into()],
                output: vec!["flat".into()],
                ..Default::default()
            },
        ],
    );
    let bytes = original.encode_to_vec();
    let (budget, memory, _) = resources(32 << 20);
    let c = census_source(&bytes, &budget, at()).unwrap();
    assert_eq!(c.materialization_cells, c.parameter_cells + 4);
    let compiled = compile_staged_original_source(&bytes, &budget, at()).unwrap();
    assert!(
        compiled.receipt().extracted_visible_bytes <= compiled.receipt().extraction_capacity_bound
    );
    drop(compiled);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_reused_parameters_spend_work_for_every_conversion_visit() {
    let (budget, memory, _) = resources(32 << 20);
    let once = reused_conv_source(1);
    let twice = reused_conv_source(2);
    let a = census_source(&once, &budget, at()).unwrap();
    let b = census_source(&twice, &budget, at()).unwrap();
    assert_eq!(a.parameter_cells, b.parameter_cells);
    assert_eq!(
        b.materialization_cells - a.materialization_cells,
        128 * 128 * 9
    );
    assert!(
        b.production_work(twice.len()).unwrap() - a.production_work(once.len()).unwrap()
            >= (128 * 128 * 9 * 96) as u64
    );
    assert!(b.extraction_bytes().unwrap() - a.extraction_bytes().unwrap() >= 128 * 128 * 9 * 80);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_repeated_unpacked_float_and_packed_int64_vectors_include_growth_and_signed_bits(
) {
    let (budget, _, _) = resources(32 << 20);
    let values = [
        0.0f32,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        0.5,
        -0.25,
        1.0,
        2.0,
        -4.0,
    ];
    let mut tensor = Vec::new();
    message(&mut tensor, 1, &[9]);
    scalar(&mut tensor, 2, 1);
    for value in values {
        varint(&mut tensor, (4 << 3) | 5);
        tensor.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let mut graph = Vec::new();
    message(&mut graph, 5, &tensor);
    let mut source = Vec::new();
    message(&mut source, 7, &graph);
    let (census, actual) = paid_visible(&source, &budget, at());
    assert_eq!(census.parameter_cells, values.len());
    assert!(actual <= census.decoded_bytes);
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: census.decoded_bytes + source.len() + META,
                work: charged(source.len(), 8192).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let decoded = ModelProto::decode(&source[..]).unwrap();
    let actual_bits: Vec<_> = decoded.graph.unwrap().initializer[0]
        .float_data
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(actual_bits, values.map(f32::to_bits));
    drop(actual_bits);
    lease.complete().unwrap();
}

#[test]
fn staged_original_source_unknown_nonallocating_wire_does_not_create_false_parameter_credit() {
    let (budget, _, _) = resources(32 << 20);
    let mut source = model(2, true).encode_to_vec();
    let before = census_source(&source, &budget, at()).unwrap();
    scalar(&mut source, 100, u64::MAX);
    message(&mut source, 101, &[7; 1024]);
    let after = census_source(&source, &budget, at()).unwrap();
    assert_eq!(before.decoded_bytes, after.decoded_bytes);
    assert_eq!(before.parameter_cells, after.parameter_cells);
    assert_eq!(before.messages, after.messages);
    assert_eq!(
        paid_visible(&source, &budget, at()).0.decoded_bytes,
        after.decoded_bytes
    );
}

#[test]
fn staged_original_source_malformed_wire_and_unsupported_heap_surface_refuse_without_decode_claim()
{
    let (budget, memory, _) = resources(32 << 20);
    let invalid = [
        vec![0],
        vec![0xff; 11],
        vec![0x3a, 0x7f, 0],
        vec![0x3b],
        vec![0x0e],
        vec![0x3a, 1, 0],
    ];
    for source in invalid {
        assert!(compile_staged_original_source(&source, &budget, at()).is_err());
        assert_eq!(live(&memory), 0);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
    let mut original = model(2, true);
    original.graph.as_mut().unwrap().node[0]
        .attribute
        .push(AttributeProto {
            name: "body".into(),
            r#type: 5,
            g: Some(GraphProto::default()),
            ..Default::default()
        });
    let source = original.encode_to_vec();
    assert!(census_source(&source, &budget, at()).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_predecode_header_quota_and_full_decode_quota_refuse_without_refresh() {
    let source = model(16, true).encode_to_vec();
    let (small, memory, _) = resources(META - 1);
    let before = small.receipt().unwrap().admitted;
    assert!(compile_staged_original_source(&source, &small, at()).is_err());
    assert_eq!(small.receipt().unwrap().admitted, before);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, 0);
    let (budget, memory, _) = resources(META);
    assert!(compile_staged_original_source(&source, &budget, at()).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, 1); // paid wire census only
    assert_eq!(live(&memory), 0);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, META);
}

#[test]
fn staged_original_source_cancellation_after_actual_decode_releases_partial_workspace_and_publishes_nothing(
) {
    let (budget, memory, cancel) = resources(32 << 20);
    let source = model(32, true).encode_to_vec();
    let mut observed = false;
    let mut after = || {
        observed = true;
        cancel.cancel();
    };
    assert!(compile_inner(&source, &budget, at(), Some(&mut after)).is_err());
    assert!(observed);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn staged_original_source_expired_and_foreign_transfer_keep_original_clock_and_ledger() {
    let (budget, memory, _) = resources(32 << 20);
    let source = model(2, true).encode_to_vec();
    let before = budget.receipt().unwrap().admitted;
    assert!(compile_staged_original_source(&source, &budget, Instant::now()).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
    let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
    let (foreign, foreign_memory, _) = resources(32 << 20);
    assert!(staged.into_compilation(&foreign, Some(at())).is_err());
    assert_eq!(live(&memory), 0);
    assert_eq!(live(&foreign_memory), 0);
    assert_eq!(foreign.receipt().unwrap().admitted, 0);
    let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
    let original = staged.compiled().deadline(None).unwrap().unwrap();
    let early = original
        .checked_sub(Duration::from_secs(10))
        .expect("test clock supports an expired deadline");
    let compiled = staged.into_compilation(&budget, Some(early)).unwrap();
    assert_eq!(compiled.deadline(Some(original)).unwrap(), Some(early));
    drop(compiled);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_nonfinite_parameter_cannot_publish_compiled_original_owner() {
    let (budget, memory, _) = resources(32 << 20);
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let mut original = model(2, true);
        original.graph.as_mut().unwrap().initializer[0].raw_data =
            value.to_bits().to_le_bytes().to_vec();
        assert!(compile_staged_original_source(&original.encode_to_vec(), &budget, at()).is_err());
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn staged_original_source_aliases_keep_one_actual_compiled_claim_after_decoder_last_owner_releases()
{
    let (budget, memory, _) = resources(32 << 20);
    let source = model(2, true).encode_to_vec();
    let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let alias = staged.compiled().clone();
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    drop(staged);
    assert_eq!(live(&memory), alias.resident_bytes());
    alias.checkpoint(&budget, None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}

fn native_original_float(index: usize) -> f32 {
    match index {
        0 => 0.0,
        1 => -0.0,
        2 => f32::from_bits(1),
        3 => f32::from_bits(0x007f_ffff),
        i => (i as f32 - 1900.0) / 4096.0,
    }
}
fn native_affine_model(hidden: usize, raw: bool, symbolic_batch: bool) -> ModelProto {
    let mut input = info("input", &[1, 30]);
    let mut output = info("output", &[1, 2]);
    if symbolic_batch {
        input
            .r#type
            .as_mut()
            .unwrap()
            .tensor_type
            .as_mut()
            .unwrap()
            .shape
            .as_mut()
            .unwrap()
            .dim[0]
            .value = Some(Value::DimParam("original_input_batch".into()));
        output
            .r#type
            .as_mut()
            .unwrap()
            .tensor_type
            .as_mut()
            .unwrap()
            .shape
            .as_mut()
            .unwrap()
            .dim[0]
            .value = Some(Value::DimParam("original_output_batch".into()));
    }
    // Original ONNX MatMul is [input,output]; each original FLOAT bit is
    // deliberately distinct from its transposed declaration's neighboring cell.
    let first: Vec<_> = (0..30 * hidden).map(native_original_float).collect();
    let bias: Vec<_> = (0..hidden).map(|i| (i as f32 - 64.0) / 128.0).collect();
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![input],
            output: vec![output],
            initializer: vec![
                tensor("w1", &[30, hidden as i64], &first, raw),
                tensor("b1", &[hidden as i64], &bias, raw),
                tensor("w2", &[hidden as i64, 2], &vec![0.0; hidden * 2], raw),
                tensor("b2", &[1, 2], &[0.75, 0.0], raw),
            ],
            node: vec![
                node("MatMul", &["input", "w1"], "first_product"),
                node("Add", &["first_product", "b1"], "pre"),
                node("Relu", &["pre"], "head"),
                node("MatMul", &["head", "w2"], "second_product"),
                node("Add", &["second_product", "b2"], "output"),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn staged_original_source_native30x128_exact_row_major_transpose_raw_typed_and_three_decode_claims()
{
    let (budget, memory, _) = resources(32 << 20);
    let raw = native_affine_model(128, true, true).encode_to_vec();
    let typed = native_affine_model(128, false, true).encode_to_vec();
    // The old rank-four extractor contract remains unchanged.
    assert!(super::super::margin_row_bab::extract_twin_spec_named_bytes(&raw).is_none());
    let a = compile_staged_original_source(&raw, &budget, at()).unwrap();
    let b = compile_staged_original_source(&typed, &budget, at()).unwrap();
    let mut transposed = vec![0.0; 30 * 128];
    for input in 0..30 {
        for output in 0..128 {
            transposed[output * 30 + input] =
                f64::from(native_original_float(input * 128 + output));
        }
    }
    let exact = TwinSpec {
        n_in: 30,
        ops: vec![
            TwinOpSpec::Gemm {
                input: 0,
                weight: transposed,
                bias: (0..128)
                    .map(|j| f64::from((j as f32 - 64.0) / 128.0))
                    .collect(),
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (128, 30),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![0.0; 256],
                bias: vec![0.75, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 128),
            },
        ],
    };
    let independently_declared =
        PreparedRootCompilation::compile_declared_twin(exact, &budget, Some(at())).unwrap();
    assert_eq!(
        a.compiled().source_digest(),
        independently_declared.source_digest()
    );
    assert_eq!(
        a.compiled().compiled_digest(),
        independently_declared.compiled_digest()
    );
    assert_eq!(a.compiled().source_digest(), b.compiled().source_digest());
    assert_eq!(
        a.compiled().compiled_digest(),
        b.compiled().compiled_digest()
    );
    assert_ne!(a.source_sha256(), b.source_sha256());
    for receipt in [a.receipt(), b.receipt()] {
        assert_eq!(
            (receipt.node_occurrences, receipt.parameter_cells),
            (5, 30 * 128 + 128 + 128 * 2 + 2)
        );
        assert!(receipt.decoded_visible_bytes <= receipt.decoded_capacity_bound);
        assert!(receipt.extracted_visible_bytes <= receipt.extraction_capacity_bound);
    }
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    let retained = a.compiled().resident_bytes()
        + b.compiled().resident_bytes()
        + independently_declared.resident_bytes();
    assert_eq!(live(&memory), retained);
    drop((a, b, independently_declared));
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_native_original_owner_transfers_to_actual_terminal_without_redecode() {
    let (budget, memory, _) = resources(32 << 20);
    let source = native_affine_model(128, true, true).encode_to_vec();
    let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
    let sha = *staged.source_sha256();
    assert_eq!(sha, <[u8; 32]>::from(Sha256::digest(&source)));
    let proof = staged
        .into_compilation(&budget, Some(at()))
        .unwrap()
        .prove_declared_domain(
            &vec![-1.0; 30],
            &vec![1.0; 30],
            0,
            &[1],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions::default(),
            &budget,
            Some(at()),
        )
        .unwrap();
    // Independent original final MatMul/Add is identically [3/4,0].
    assert!(proof.min_margin() > 0.0 && proof.min_margin() <= 0.75);
    assert!(proof.attempt_receipt().terminal_closed);
    assert!(!proof.attempt_receipt().ordinary_attempted);
    assert_eq!((proof.target(), proof.classes()), (0, &[1][..]));
    assert!(proof.budget().shares_ledger(&budget));
    let alias = proof.clone();
    drop(proof);
    assert!(live(&memory) > 0);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_unused_opsets_pay_quadratic_guard_before_any_decode() {
    let (budget, memory, _) = resources(32 << 20);
    let mut original = native_affine_model(2, true, false);
    for i in 0..256 {
        original.opset_import.push(OperatorSetIdProto {
            domain: format!("unused.original.{i}"),
            version: 1,
        });
    }
    let source = original.encode_to_vec();
    let c = census_source(&source, &budget, at()).unwrap();
    assert_eq!(c.opsets, 257);
    assert!(
        c.metadata_work().unwrap() >= 128 * c.opsets * c.opsets + 64 * c.string_bytes * c.opsets
    );
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: c.decoded_bytes + source.len() + META,
                work: c.production_work(source.len()).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let decoded = ModelProto::decode(source.as_slice()).unwrap();
    assert!(decoded_visible_bytes(&decoded, &budget, at()).unwrap() <= c.decoded_bytes);
    drop(decoded);
    lease.complete().unwrap();
    drop(lease);
    let memory_small = MemoryAuthority::new(32 << 20);
    let cancel = CancellationToken::default();
    let limited = WorkBudget::new_in(
        &memory_small,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: 32 << 20,
            persistent_bytes: 32 << 20,
            max_work: c.production_work(source.len()).unwrap() - 1,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let mut decoded_observed = false;
    assert!(compile_inner(
        &source,
        &limited,
        at(),
        Some(&mut || decoded_observed = true)
    )
    .is_err());
    assert!(!decoded_observed);
    assert_eq!(live(&memory_small), 0);
    assert_eq!(live(&memory), 0);
}

#[test]
fn staged_original_source_native_unused_unique_imports_compile_but_duplicate_standard_alias_refuses(
) {
    let (budget, memory, _) = resources(32 << 20);
    let mut original = native_affine_model(2, true, false);
    original.opset_import.push(OperatorSetIdProto {
        domain: "unused.custom".into(),
        version: 1,
    });
    let compiled =
        compile_staged_original_source(&original.encode_to_vec(), &budget, at()).unwrap();
    assert_eq!(
        (compiled.compiled().n_in(), compiled.compiled().n_out()),
        (30, 2)
    );
    drop(compiled);
    assert_eq!(live(&memory), 0);
    original.opset_import.push(OperatorSetIdProto {
        domain: "ai.onnx".into(),
        version: 13,
    });
    assert!(compile_staged_original_source(&original.encode_to_vec(), &budget, at()).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
#[cfg(feature = "external-models")]
fn staged_original_source_actual_original_safenlp_both_model_identities_compile_under_unchanged_caps(
) {
    let base = std::path::Path::new("<home>/ny-model-cache/high-impact/vnncomp2025-safenlp/8b7b811b78ce6a329dc96f04ae6652da3c245948/onnx");
    let memory = MemoryAuthority::new(32 << 20);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: 32 << 20,
            persistent_bytes: 32 << 20,
            max_work: 1 << 30,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(32 << 20).unwrap();
    for (category, expected_bytes, digest) in [
        (
            "medical",
            18443,
            "5c6b9f0d3e78db73f260a1d3a76bd36543030a1f3dfe54b103fcbf4ec08b762b",
        ),
        (
            "ruarobot",
            18423,
            "a9857f1f8f443ec03f4ede912146bf0aa3ba20d2ecb84301a8e631524576d5ff",
        ),
    ] {
        let source = std::fs::read(base.join(category).join("perturbations_0.onnx"))
            .expect("unchanged actual original SafeNLP asset required");
        assert_eq!(source.len(), expected_bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&source)), digest);
        let c = census_source(&source, &budget, at()).unwrap();
        assert_eq!(
            (c.nodes, c.parameter_cells, c.materialization_cells),
            (5, 4226, 4226)
        );
        assert!(c.production_work(source.len()).unwrap() < 1 << 30);
        let staged = compile_staged_original_source(&source, &budget, at()).unwrap();
        assert_eq!(
            (staged.compiled().n_in(), staged.compiled().n_out()),
            (30, 2)
        );
        let receipt = staged.receipt();
        assert!(receipt.decoded_visible_bytes <= receipt.decoded_capacity_bound);
        assert!(receipt.extracted_visible_bytes <= receipt.extraction_capacity_bound);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        let alias = staged.compiled().clone();
        drop(staged);
        assert_eq!(live(&memory), alias.resident_bytes());
        alias.checkpoint(&budget, None).unwrap();
        drop(alias);
        assert_eq!(live(&memory), 0);
    }
    // This checks real original decoding/compilation/ownership admission. It
    // makes no competition solve, default-route, replay or runtime claim.
}
