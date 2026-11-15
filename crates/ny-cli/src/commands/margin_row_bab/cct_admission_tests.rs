// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cct-margin-row-admission pins: the robustness parser's `<=` spelling (both
//! VNNLIB corpora, true class on the LEFT, fail-closed on everything else),
//! the extractor's certified `Gemm -> BatchNormalization` fold, and the
//! admission probe's root-tableau ceiling.
//!
//! Every positive has a negative that perturbs exactly one fact and must
//! decline. The real-corpus test reads the gitignored `benchmarks/` checkout
//! and prints `skipping` when it is absent (a worktree has none) — a gate that
//! wants it must see its `[cct-admission] real corpus:` line, not just `ok`.

use std::path::{Path, PathBuf};

use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
use ny_onnx::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::margin_row::bounds::{
    compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
};
use ny_propagate::margin_row::engine::{BackwardEngine, LaneDir};
use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet};
use prost::Message;

use super::*;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, deprecated)]
mod fp_environment {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    use std::marker::PhantomData;
    use std::rc::Rc;

    // Only rounding, FTZ and DAZ are changed; exception masks are preserved.
    const CONTROLS: u32 = (3 << 13) | (1 << 15) | (1 << 6);

    struct Restore {
        saved: u32,
        // FP control belongs to this thread, including during unwinding.
        _thread: PhantomData<Rc<()>>,
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: the value was read from this same thread's MXCSR and
            // this guard cannot be moved to another thread.
            unsafe { _mm_setcsr(self.saved) };
        }
    }

    pub(super) fn with_controls<T>(controls: u32, body: impl FnOnce() -> T) -> T {
        assert_eq!(controls & !CONTROLS, 0);
        // SAFETY: x86_64 has SSE; reading the calling thread's FP control
        // register accesses no memory owned by another thread.
        let saved = unsafe { _mm_getcsr() };
        let restore = Restore {
            saved,
            _thread: PhantomData,
        };
        // SAFETY: retain all reserved bits and exception masks from a valid
        // register value, changing only documented FP controls for this test.
        unsafe { _mm_setcsr((saved & !CONTROLS) | controls) };
        let result = body();
        drop(restore);
        result
    }
}

// ---------------------------------------------------------------------------
// temp-file helpers
// ---------------------------------------------------------------------------

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("mr_cct_{tag}_{}_{nanos}.{ext}", std::process::id()))
}

/// Parse `text` as a vnnlib file through the production entry point.
#[allow(clippy::type_complexity)]
fn parse_text(tag: &str, text: &str) -> Option<(Vec<f64>, Vec<f64>, usize, Vec<usize>)> {
    let path = temp_path(tag, "vnnlib");
    std::fs::write(&path, text).expect("write temp vnnlib");
    let parsed = parse_vnnlib_robustness(&path);
    std::fs::remove_file(&path).ok();
    parsed
}

/// VNNLIB 1.0 (flat `X_i` / `Y_j`) robustness file over a 4-pixel box with
/// 4 outputs, carrying the given output disjunct lines verbatim.
fn flat_vnnlib(disjuncts: &[&str]) -> String {
    let mut s = String::new();
    for i in 0..4 {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..4 {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..4 {
        let lo = 0.125 * f64::from(i);
        s.push_str(&format!("(assert (>= X_{i} {lo}))\n"));
        s.push_str(&format!("(assert (<= X_{i} {}))\n", lo + 0.25));
    }
    s.push_str("(assert (or\n");
    for d in disjuncts {
        s.push_str("  ");
        s.push_str(d);
        s.push('\n');
    }
    s.push_str("))\n");
    s
}

/// VNNLIB 2.0 (bracket `X[0,c,h,w]` / `Y[0,j]`) file encoding the SAME box
/// as [`flat_vnnlib`] (NCHW `[1, 1, 2, 2]`: flat index = 2h + w).
fn bracket_vnnlib(disjuncts: &[&str]) -> String {
    let mut s = String::from(
        "(vnnlib-version <2.0>)\n\n(declare-network N\n    \
         (declare-input  X float32 [1, 1, 2, 2])\n    \
         (declare-output Y float32 [1, 4])\n)\n",
    );
    for h in 0..2 {
        for w in 0..2 {
            let lo = 0.125 * f64::from(2 * h + w);
            s.push_str(&format!("(assert (>= X[0,0,{h},{w}] {lo}))\n"));
            s.push_str(&format!("(assert (<= X[0,0,{h},{w}] {}))\n", lo + 0.25));
        }
    }
    s.push_str("(assert (or\n");
    for d in disjuncts {
        s.push_str("  ");
        s.push_str(d);
        s.push('\n');
    }
    s.push_str("))\n");
    s
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------

/// The pure disjunct parser: TRUE class on the LEFT of `<=`, both spellings,
/// exactly one comparison, distinct outputs.
#[test]
fn le_disjunct_parser_true_class_is_left_operand() {
    assert_eq!(
        parse_le_robustness_disjunct("(and (<= Y_2 Y_0))"),
        Some((2, 0))
    );
    assert_eq!(
        parse_le_robustness_disjunct("(and (<= Y_0 Y_2))"),
        Some((0, 2))
    );
    assert_eq!(
        parse_le_robustness_disjunct("(and (<= Y[0,2] Y[0,7]))"),
        Some((2, 7))
    );
    assert_eq!(
        parse_le_robustness_disjunct("(and (<= Y[0, 12] Y[0, 3]) )"),
        Some((12, 3))
    );
    for bad in [
        "(and (<= Y_2 Y_2))",              // self comparison
        "(and (<= Y_2 0.5))",              // constant operand
        "(and (<= 0.5 Y_2))",              // constant operand
        "(and (<= Y_2 Y_0) (<= Y_2 Y_1))", // a CONJUNCTION is a different property
        "(and (<= Y_2 Y_0)",               // unbalanced
        "(and (<= Y_2 Y_0)) junk",         // trailing content
        "(and (<= Y[1,2] Y[0,0]))",        // batch index must be 0
        "(and (<= Y[0,0,2] Y[0,0]))",      // wrong rank
        "(and (<= Y_ Y_0))",               // no digits
        "(and (<= Y_2a Y_0))",             // junk inside the operand
        "(and (< Y_2 Y_0))",               // strict: not this spelling
        "(and (>= Y_0 Y_2))",              // the other spelling has its own arm
    ] {
        assert_eq!(parse_le_robustness_disjunct(bad), None, "accepted {bad:?}");
    }
}

/// Both VNNLIB corpora spell cct's property `(<= Y_t Y_j)` / `(<= Y[0,t]
/// Y[0,j])`; both must parse to the IDENTICAL `(lo, hi, t, adv)`, with the
/// true class taken from the LEFT operand — and a file mixing the `>=` and
/// `<=` spellings of the same disjunction must agree with both.
#[test]
fn le_robustness_spec_both_spellings_parse_identically() {
    let flat = parse_text(
        "le_flat",
        &flat_vnnlib(&[
            "(and (<= Y_2 Y_0))",
            "(and (<= Y_2 Y_1))",
            "(and (<= Y_2 Y_3))",
        ]),
    )
    .expect("flat <= robustness spec parses");
    let bracket = parse_text(
        "le_bracket",
        &bracket_vnnlib(&[
            "(and (<= Y[0,2] Y[0,0]))",
            "(and (<= Y[0,2] Y[0,1]))",
            "(and (<= Y[0,2] Y[0,3]))",
        ]),
    )
    .expect("bracket <= robustness spec parses");
    let mixed = parse_text(
        "le_mixed",
        &flat_vnnlib(&[
            "(and (>= Y_0 Y_2))",
            "(and (<= Y_2 Y_1))",
            "(and (<= Y_2 Y_3))",
        ]),
    )
    .expect("mixed >= / <= spellings of one disjunction parse");
    let ge = parse_text(
        "ge_flat",
        &flat_vnnlib(&[
            "(and (>= Y_0 Y_2))",
            "(and (>= Y_1 Y_2))",
            "(and (>= Y_3 Y_2))",
        ]),
    )
    .expect("the historical >= spelling still parses");
    for (name, got) in [("flat", &flat), ("bracket", &bracket), ("mixed", &mixed)] {
        assert_eq!(
            got.2, 2,
            "{name}: true class must be the LEFT operand of <="
        );
        assert_eq!(got.3, vec![0, 1, 3], "{name}: adversarial set");
        assert_eq!(
            got, &ge,
            "{name}: must equal the >= spelling of the same property"
        );
    }
    assert_eq!(flat.0, vec![0.0, 0.125, 0.25, 0.375]);
    assert_eq!(flat.1, vec![0.25, 0.375, 0.5, 0.625]);
}

/// Everything that is not one clean robustness disjunction fails CLOSED
/// (`None`, so the lane never fires): a conjunction, a constant threshold,
/// two different "true" classes, a dropped strict disjunct, a malformed line.
#[test]
fn le_robustness_spec_fails_closed_on_non_robustness_forms() {
    let cases: [(&str, Vec<&str>); 6] = [
        // Two different left operands: no single true class.
        ("two_true", vec!["(and (<= Y_2 Y_0))", "(and (<= Y_1 Y_0))"]),
        // A conjunction inside one disjunct.
        ("conj", vec!["(and (<= Y_2 Y_0) (<= Y_2 Y_1))"]),
        // A constant threshold.
        ("const", vec!["(and (<= Y_2 0.5))"]),
        // The trusted parser records a strict disjunct the hand parser drops:
        // the output-side set-equality cross-check must refuse.
        (
            "strict_dropped",
            vec!["(and (<= Y_2 Y_0))", "(and (< Y_2 Y_1))"],
        ),
        // Operand mis-order against a >= disjunct: `>= Y_0 Y_2` says true 2,
        // `<= Y_0 Y_1` says true 0 — inconsistent, refuse.
        (
            "mis_order",
            vec!["(and (>= Y_0 Y_2))", "(and (<= Y_0 Y_1))"],
        ),
        // Malformed operand.
        ("malformed", vec!["(and (<= Y_2 Y_0x))"]),
    ];
    for (tag, lines) in cases {
        assert!(
            parse_text(tag, &flat_vnnlib(&lines)).is_none(),
            "{tag}: {lines:?} must fail closed"
        );
    }
}

// ---------------------------------------------------------------------------
// extractor: certified Gemm -> BatchNormalization fold
// ---------------------------------------------------------------------------

fn f32_tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name: name.to_owned(),
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

fn int_attr(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        i: Some(v),
        r#type: 2,
        ..Default::default()
    }
}

fn float_attr(name: &str, v: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        f: Some(v),
        r#type: 1,
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], attribute: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: outputs.iter().map(|s| (*s).to_owned()).collect(),
        op_type: op.to_owned(),
        attribute,
        ..Default::default()
    }
}

/// One knob per negative test; `Default` is the clean CNN7-shaped head.
#[derive(Default, Clone, Copy)]
struct Variant {
    bn_training_mode: bool,
    bn_three_outputs: bool,
    gemm1_second_consumer: bool,
    negative_var: bool,
    bn_on_final_gemm: bool,
}

const GAMMA: [f32; 3] = [1.5, -0.75, 0.0];
const BETA: [f32; 3] = [0.1, -0.2, 0.3];
const MEAN: [f32; 3] = [0.05, -0.4, 0.2];
const VAR: [f32; 3] = [0.9, 2.5, 1.0e-6];
const W1: [f32; 24] = [
    0.3, -0.2, 0.1, 0.4, -0.5, 0.25, 0.05, -0.15, //
    -0.35, 0.45, 0.2, -0.1, 0.3, -0.25, 0.15, 0.1, //
    0.2, 0.1, -0.3, 0.35, 0.05, 0.4, -0.2, -0.45,
];
const B1: [f32; 3] = [0.05, -0.1, 0.2];
const CW: [f32; 2] = [0.8, -0.6];
const CB: [f32; 2] = [0.1, 0.2];
// Conv BatchNormalization (folded by the pre-existing Conv arm).
const CGAMMA: [f32; 2] = [1.2, -0.9];
const CBETA: [f32; 2] = [0.05, -0.1];
const CMEAN: [f32; 2] = [0.1, -0.2];
const CVAR: [f32; 2] = [0.5, 1.5];
const W2: [f32; 6] = [0.7, -0.3, 0.5, -0.2, 0.6, 0.1];
const B2: [f32; 2] = [0.0, 0.05];

/// `x[1,1,2,2] -> Conv(1->2, 1x1) -> BatchNormalization(2) -> Relu -> Flatten
/// -> Gemm(8->3) -> BatchNormalization(3) -> Relu -> Gemm(3->2)`: the CNN7
/// shape in miniature (or, with `bn_on_final_gemm`, the FC BN after the final
/// Gemm instead).
fn cnn7_head_model(v: Variant) -> ModelProto {
    let dim = |d: i64| Dimension {
        value: Some(dimension::Value::DimValue(d)),
    };
    let input = ValueInfoProto {
        name: "x".to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![dim(1), dim(1), dim(2), dim(2)],
                }),
            }),
        }),
    };
    let var: Vec<f32> = if v.negative_var {
        vec![0.9, -2.5, 1.0e-6]
    } else {
        VAR.to_vec()
    };
    let bn_attrs = {
        let mut a = vec![float_attr("epsilon", 1e-5)];
        if v.bn_training_mode {
            a.push(int_attr("training_mode", 1));
        }
        a
    };
    let bn_outputs: Vec<&str> = if v.bn_three_outputs {
        vec!["bn", "running_mean_out", "running_var_out"]
    } else {
        vec!["bn"]
    };
    let mut initializer = vec![
        f32_tensor("cw", &[2, 1, 1, 1], &CW),
        f32_tensor("cb", &[2], &CB),
        f32_tensor("cgamma", &[2], &CGAMMA),
        f32_tensor("cbeta", &[2], &CBETA),
        f32_tensor("cmean", &[2], &CMEAN),
        f32_tensor("cvar", &[2], &CVAR),
        f32_tensor("w1", &[3, 8], &W1),
        f32_tensor("b1", &[3], &B1),
        f32_tensor("gamma", &[3], &GAMMA),
        f32_tensor("beta", &[3], &BETA),
        f32_tensor("mean", &[3], &MEAN),
        f32_tensor("var", &[3], &var),
        f32_tensor("w2", &[2, 3], &W2),
        f32_tensor("b2", &[2], &B2),
    ];
    let gemm = |inp: &str, w: &str, b: &str, out: &str| {
        node("Gemm", &[inp, w, b], &[out], vec![int_attr("transB", 1)])
    };
    let mut nodes = vec![
        node("Conv", &["x", "cw", "cb"], &["c"], Vec::new()),
        node(
            "BatchNormalization",
            &["c", "cgamma", "cbeta", "cmean", "cvar"],
            &["cbn"],
            vec![float_attr("epsilon", 1e-5)],
        ),
        node("Relu", &["cbn"], &["r1"], Vec::new()),
        node("Flatten", &["r1"], &["f"], Vec::new()),
    ];
    if v.bn_on_final_gemm {
        // Gemm1 -> Relu -> Gemm2 -> BN(2): the BN folds into the FINAL Gemm.
        initializer.push(f32_tensor("g2", &[2], &[1.1, 0.9]));
        initializer.push(f32_tensor("be2", &[2], &[0.0, 0.1]));
        initializer.push(f32_tensor("m2", &[2], &[0.2, -0.1]));
        initializer.push(f32_tensor("v2", &[2], &[1.0, 0.5]));
        nodes.push(gemm("f", "w1", "b1", "g1"));
        nodes.push(node("Relu", &["g1"], &["r2"], Vec::new()));
        nodes.push(gemm("r2", "w2", "b2", "g2out"));
        nodes.push(node(
            "BatchNormalization",
            &["g2out", "g2", "be2", "m2", "v2"],
            &["out"],
            vec![float_attr("epsilon", 1e-5)],
        ));
    } else {
        nodes.push(gemm("f", "w1", "b1", "g1"));
        nodes.push(node(
            "BatchNormalization",
            &["g1", "gamma", "beta", "mean", "var"],
            &bn_outputs,
            bn_attrs,
        ));
        if v.gemm1_second_consumer {
            // A second reader of the UNFOLDED Gemm output: folding would
            // change what it sees, so the fold must not happen.
            nodes.push(node("Relu", &["g1"], &["side"], Vec::new()));
        }
        nodes.push(node("Relu", &["bn"], &["r2"], Vec::new()));
        nodes.push(gemm("r2", "w2", "b2", "out"));
    }
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: nodes,
            initializer,
            input: vec![input],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn extract_model(tag: &str, model: &ModelProto) -> Option<TwinSpec> {
    let path = temp_path(tag, "onnx");
    std::fs::write(&path, model.encode_to_vec()).expect("write temp onnx");
    let spec = extract_twin_spec(&path);
    std::fs::remove_file(&path).ok();
    spec
}

fn subnormal_head_model() -> ModelProto {
    let mut model = cnn7_head_model(Variant::default());
    let graph = model.graph.as_mut().expect("fixture graph");
    let delta = f32::from_bits(1);
    let amplifier = f32::from_bits((127 + 100) << 23);
    let epsilon = f32::from_bits((127 - 50) << 23);
    for tensor in &mut graph.initializer {
        let values = match tensor.name.as_str() {
            "cw" => vec![1.0, 0.0],
            "cb" | "cbeta" | "cmean" => vec![0.0; 2],
            "cgamma" | "cvar" => vec![1.0; 2],
            "w1" => {
                let mut row = vec![0.0; 24];
                row[0] = delta;
                row
            }
            "b1" | "beta" | "mean" => vec![0.0; 3],
            "gamma" | "var" => vec![1.0; 3],
            "w2" => vec![0.0, 0.0, 0.0, amplifier, 0.0, 0.0],
            "b2" => vec![epsilon, 0.0],
            other => panic!("unexpected fixture initializer {other}"),
        };
        *tensor = f32_tensor(&tensor.name, &tensor.dims, &values);
    }
    for node in &mut graph.node {
        if node.op_type == "BatchNormalization" {
            node.attribute = vec![float_attr("epsilon", 0.0)];
        }
    }
    model
}

/// Erasing a binary32 subnormal would reverse this exact source margin.
/// Exercise the real ONNX decoder, both BN folds, compiler and y-row producer.
#[test]
fn extractor_retains_subnormal_weight_and_negative_amplified_source_margin() {
    ny_core::require_f64_interval_proof_environment().expect("qualified source producer");
    let spec = extract_model("subnormal_head", &subnormal_head_model())
        .expect("a preserved binary32 subnormal is normal in the f64 fold");
    let delta = ny_core::f32_to_f64_exact(f32::from_bits(1));
    let TwinOpSpec::Gemm { weight, .. } = &spec.ops[3] else {
        panic!("fixture head Gemm");
    };
    assert_eq!(weight[0].to_bits(), delta.to_bits());

    let net = TwinNet::compile(&spec).expect("compile the extracted source");
    let root = RootGates::build(&net, &[1.0; 4], &[1.0; 4], RoundMode::Outward, None)
        .expect("qualified point-box root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("native source rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    assert!(ybox.ly[0] <= delta && delta <= ybox.uy[0]);
    assert!(
        ybox.ly[0] > 0.0,
        "the charged fold must preserve the positive feature"
    );

    let (weights, biases, (_, width)) = net.gemm2();
    let amplifier = weights[width];
    let epsilon = biases[0];
    // Identity trunk/BN and x=1 give y=2^-149 exactly:
    // 2^-50 - 2^100*relu(y) = -2^-50. Pricing the full returned y-box
    // must retain the negative sign even after both carried fold budgets.
    assert_eq!(epsilon - amplifier * delta, -epsilon);
    assert!(epsilon - amplifier * ybox.ly[0] < 0.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn extractor_refuses_unqualified_modes_before_source_folding() {
    ny_core::require_f64_interval_proof_environment().expect("qualified test entry");
    let model = subnormal_head_model();
    let raw = f32_tensor("tiny", &[1], &[f32::from_bits(1)]);
    let inline = TensorProto {
        raw_data: Vec::new(),
        float_data: vec![f32::from_bits(1)],
        ..raw.clone()
    };
    let expected_bits = ny_core::f32_to_f64_exact(f32::from_bits(1)).to_bits();
    for controls in [1 << 15, 1 << 6, 1 << 13, 2 << 13, 3 << 13] {
        let (qualified, refused, raw_bits, inline_bits) =
            fp_environment::with_controls(controls, || {
                (
                    ny_core::has_f64_interval_proof_environment(),
                    extract_model("unqualified_source", &model).is_none(),
                    tensor_f32(&raw).expect("bit-exact raw decoder")[0].to_bits(),
                    tensor_f32(&inline).expect("bit-exact inline decoder")[0].to_bits(),
                )
            });
        assert!(!qualified, "invalid control bits {controls:#x}");
        assert!(refused, "the actual extractor accepted {controls:#x}");
        assert_eq!(raw_bits, expected_bits);
        assert_eq!(inline_bits, expected_bits);
        ny_core::require_f64_interval_proof_environment().expect("restore the caller's FP mode");
    }
    assert!(extract_model("restored_source", &model).is_some());
}

/// THE EXTRACTOR PIN. CNN7's FC-BN head folds into the first head Gemm WITH
/// its certified error, the net satisfies the compiled head contract, and —
/// the lane-level soundness check on `Conv -> BN -> ReLU -> Gemm -> BN -> ReLU
/// -> Gemm` — the lane's certified root y-box AND per-class margin lower
/// bounds enclose the EXACT (unfolded, f64-evaluated) network at sampled
/// inputs.
#[test]
fn extractor_folds_gemm_batchnorm_head_with_certified_error() {
    let spec = extract_model("fold_ok", &cnn7_head_model(Variant::default()))
        .expect("Conv -> BN -> Relu -> Flatten -> Gemm -> BN -> Relu -> Gemm extracts");
    // Both BNs absorbed: Conv(folded), Relu, Flatten, Gemm(folded), Relu, Gemm.
    assert_eq!(spec.ops.len(), 6);
    let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        shape,
        ..
    } = &spec.ops[3]
    else {
        panic!("op 3 must be the folded head Gemm");
    };
    assert_eq!(*shape, (3, 8));
    assert_eq!(
        *weight_rel_err,
        ny_propagate::margin_row::rounding::RHO_FOLD
    );
    assert_eq!(bias_err.len(), 3);
    assert!(bias_err.iter().all(|e| e.is_finite() && *e > 0.0));
    assert!(spec.ops[3].is_folded_gemm());
    assert!(!spec.ops[5].is_folded_gemm(), "the final Gemm stays exact");
    let eps = f64::from(1e-5_f32);
    for o in 0..3 {
        let a = f64::from(GAMMA[o]) / (f64::from(VAR[o]) + eps).sqrt();
        for i in 0..8 {
            assert_eq!(
                weight[o * 8 + i].to_bits(),
                (a * f64::from(W1[o * 8 + i])).to_bits(),
                "folded weight ({o},{i}) is not the Conv+BN arm's f64 product"
            );
        }
        let expect_b = (f64::from(BETA[o]) - a * f64::from(MEAN[o])) + a * f64::from(B1[o]);
        assert_eq!(bias[o].to_bits(), expect_b.to_bits(), "folded bias row {o}");
    }
    assert!(twin_spec_compiles_to_margin_row_head(&spec));
    let net = TwinNet::compile(&spec).expect("folded CNN7 head compiles");

    // Sampled containment of the EXACT unfolded map in the certified y-box.
    let lo = vec![-0.5, 0.0, -0.25, 0.1];
    let hi = vec![0.5, 0.6, 0.25, 0.9];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    // Root certified per-class margin lower bound, Y_0 - Y_1 (the lane's
    // three sound paths combined exactly as its root evaluation does).
    let (t, adv) = (0usize, vec![1usize]);
    let mode = RoundMode::Outward;
    let mb = MarginBatch::new(&net, t, &adv).expect("margin batch");
    let gates = head_gates(&ybox, mode);
    let ms = margin_seed(&mb, &gates, &ybox, mode);
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("direct pass");
    let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
    let ald = row_dots(&root, &al);
    let aud = row_dots(&root, &au);
    let m2v = compose_viay(&eng, &mb, &gates, &al, &au, &ald, &aud, mode);
    let margin_bound = direct[0].max(ms.m1[0]).max(m2v[0]);
    let mut min_margin = f64::INFINITY;
    let mut state = 0x5EED_u64;
    let mut unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    for _ in 0..400 {
        let x: Vec<f64> = (0..4).map(|i| lo[i] + unit() * (hi[i] - lo[i])).collect();
        // Conv 1x1 (2 channels) -> BN -> Relu -> Flatten (channel-major), all
        // from the UNFOLDED f32 payloads widened exactly to f64.
        let z: Vec<f64> = (0..2)
            .flat_map(|c| {
                let sc = f64::from(CGAMMA[c]) / (f64::from(CVAR[c]) + eps).sqrt();
                x.iter().map(move |&xi| {
                    let v = f64::from(CW[c]) * xi + f64::from(CB[c]);
                    (sc * (v - f64::from(CMEAN[c])) + f64::from(CBETA[c])).max(0.0)
                })
            })
            .collect();
        let mut y_all = [0.0f64; 3];
        for o in 0..3 {
            let u: f64 =
                f64::from(B1[o]) + (0..8).map(|i| f64::from(W1[o * 8 + i]) * z[i]).sum::<f64>();
            let s = f64::from(GAMMA[o]) / (f64::from(VAR[o]) + eps).sqrt();
            let y = s * (u - f64::from(MEAN[o])) + f64::from(BETA[o]);
            assert!(
                ybox.ly[o] - 1e-9 <= y && y <= ybox.uy[o] + 1e-9,
                "exact Gemm->BN y[{o}] = {y} escapes the certified box [{}, {}]",
                ybox.ly[o],
                ybox.uy[o]
            );
            y_all[o] = y;
        }
        let out = |k: usize| {
            f64::from(B2[k])
                + (0..3)
                    .map(|o| f64::from(W2[k * 3 + o]) * y_all[o].max(0.0))
                    .sum::<f64>()
        };
        min_margin = min_margin.min(out(t) - out(adv[0]));
    }
    assert!(
        margin_bound <= min_margin + 1e-9,
        "certified root margin bound {margin_bound} above the exact sampled minimum {min_margin}"
    );
}

/// Each perturbation of ONE structural fact must decline: training-mode BN,
/// a BN emitting running statistics, a second reader of the unfolded Gemm
/// output, an uncertifiable fold (`var + eps < 0`), and a BN on the FINAL Gemm
/// (extracts, but the head contract and the compiler both refuse it).
#[test]
fn extractor_declines_uncertifiable_or_misplaced_fc_bn() {
    for (tag, v) in [
        (
            "training",
            Variant {
                bn_training_mode: true,
                ..Variant::default()
            },
        ),
        (
            "three_out",
            Variant {
                bn_three_outputs: true,
                ..Variant::default()
            },
        ),
        (
            "second_reader",
            Variant {
                gemm1_second_consumer: true,
                ..Variant::default()
            },
        ),
        (
            "neg_var",
            Variant {
                negative_var: true,
                ..Variant::default()
            },
        ),
    ] {
        assert!(
            extract_model(tag, &cnn7_head_model(v)).is_none(),
            "{tag}: must decline the whole net"
        );
    }
    let final_bn = extract_model(
        "final_bn",
        &cnn7_head_model(Variant {
            bn_on_final_gemm: true,
            ..Variant::default()
        }),
    )
    .expect("a BN after the final Gemm still folds structurally");
    assert!(final_bn.ops.last().is_some_and(TwinOpSpec::is_folded_gemm));
    assert!(
        !twin_spec_compiles_to_margin_row_head(&final_bn),
        "the admission probe must refuse a folded FINAL Gemm"
    );
    assert!(
        TwinNet::compile(&final_bn).is_err(),
        "the compiler must refuse a folded FINAL Gemm"
    );
}

/// The admission probe and the compiler share ONE root-tableau predicate: a
/// structurally valid net above the ceiling is refused by both.
#[test]
fn head_contract_refuses_root_tableau_above_cap() {
    let head = |n_in: usize| TwinSpec {
        n_in,
        ops: vec![
            TwinOpSpec::Relu { input: 0 },
            TwinOpSpec::Gemm {
                input: 1,
                weight: vec![0.0; 2 * n_in],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, n_in),
            },
            TwinOpSpec::Relu { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.0; 4],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    assert!(twin_spec_compiles_to_margin_row_head(&head(8)));
    let wide = head(40_000); // 40_000 * 40_001 > 2^30
    assert!(!wide.root_tableau_within_cap());
    assert!(!twin_spec_compiles_to_margin_row_head(&wide));
    assert!(TwinNet::compile(&wide).is_err());
}

// ---------------------------------------------------------------------------
// real corpus (skips without the gitignored benchmark checkout)
// ---------------------------------------------------------------------------

fn bench(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks")
        .join(rel)
}

/// The real cct files: both corpora's idx448 spec parse identically (true
/// class 2, adversarial 0,1,3..9); cifar10 CNN7 extracts with a folded head
/// and passes the head contract and the compiler; TinyImageNet CNN7 extracts
/// but is REFUSED by the root-tableau ceiling (its f64 tableau would be 26 GB
/// per matrix). Also pins the one other category the parser widening admits:
/// relusplitter's oval21 `cifar_base_kw` rows (`(<= Y_t Y_j)`, two-Gemm head).
#[test]
fn cct_real_corpus_admission() {
    let cct = "vnncomp2026/benchmarks/challenging_certified_training_2026";
    let v1 = bench(&format!(
        "{cct}/1.0/vnnlib/cifar10_eps2_cnn7/cifar10_eps2_cnn7_idx448_sample2.vnnlib"
    ));
    let v2 = bench(&format!(
        "{cct}/2.0/vnnlib/cifar10_eps2_cnn7/cifar10_eps2_cnn7_idx448_sample2.vnnlib"
    ));
    let cifar = bench(&format!("{cct}/2.0/onnx/cifar10_eps2_cnn7.onnx"));
    let tiny = bench(&format!("{cct}/2.0/onnx/tinyimagenet_eps1_cnn7.onnx"));
    if ![&v1, &v2, &cifar, &tiny].iter().all(|p| p.is_file()) {
        eprintln!("skipping: cct benchmark checkout absent (a worktree has no benchmarks/)");
        return;
    }
    let p1 = parse_vnnlib_robustness(&v1).expect("1.0 corpus parses");
    let p2 = parse_vnnlib_robustness(&v2).expect("2.0 corpus parses");
    assert_eq!(p1, p2, "the two corpora encode the identical property");
    assert_eq!(p1.2, 2, "label 2 is the LEFT operand of every <=");
    assert_eq!(p1.3, vec![0, 1, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(p1.0.len(), 3072);

    let spec = extract_twin_spec(&cifar).expect("cifar10 CNN7 extracts");
    assert_eq!(spec.n_in, 3072);
    assert_eq!(
        spec.ops.len(),
        14,
        "5x(Conv,Relu), Flatten, Gemm+BN, Relu, Gemm"
    );
    assert!(spec.ops[11].is_folded_gemm(), "the FC-BN head is folded");
    assert_eq!(spec.root_tableau_entries(), Some(65_536 * 3_073));
    assert!(twin_spec_compiles_to_margin_row_head(&spec));
    TwinNet::compile(&spec).expect("cifar10 CNN7 compiles");

    let tspec = extract_twin_spec(&tiny).expect("tinyimagenet CNN7 extracts structurally");
    assert_eq!(tspec.root_tableau_entries(), Some(262_144 * 12_289));
    assert!(
        !twin_spec_compiles_to_margin_row_head(&tspec),
        "the tiny CNN7 tableau is above the ceiling; the probe must refuse it"
    );
    let err = TwinNet::compile(&tspec)
        .err()
        .expect("compiler refuses too");
    assert!(format!("{err}").contains("root tableau"), "{err}");

    let rs = "vnncomp2025/benchmarks/relusplitter";
    let oval_onnx = bench(&format!("{rs}/onnx/oval21-benchmark_cifar_base_kw.onnx"));
    let oval_vnnlib = bench(&format!(
        "{rs}/vnnlib/oval21-benchmark_cifar_base_kw-img3365-eps0.0407843137254902.vnnlib"
    ));
    let oval = if oval_onnx.is_file() && oval_vnnlib.is_file() {
        let (_, _, t, adv) = parse_vnnlib_robustness(&oval_vnnlib).expect("oval21 parses");
        assert_eq!((t, adv.len()), (9, 9));
        let ospec = extract_twin_spec(&oval_onnx).expect("oval21 base extracts");
        assert!(twin_spec_compiles_to_margin_row_head(&ospec));
        "admitted"
    } else {
        "absent"
    };
    eprintln!(
        "[cct-admission] real corpus: cct parse OK (t=2, 9 adv), cifar10 cnn7 ADMITTED \
         (folded head, {} tableau entries), tinyimagenet cnn7 REFUSED (tableau cap), \
         relusplitter oval21 {oval}",
        65_536 * 3_073
    );
}

/// Proof bundle v2 (ENGINE_PROGRAM §1.7 B1): the extractor records, for EVERY
/// Relu op of the spec, the ONNX Relu node's `input[0]` name and flat size —
/// the identity an offline checker resolves literals against — in the SAME
/// walk, and `extract_twin_spec` is unchanged by it. Folded BatchNorms leave
/// the Relu's input as the BN OUTPUT's name (the same mathematical tensor the
/// lane splits, folded or not).
#[test]
fn twin_site_names_resolve_every_relu_op() {
    let model = cnn7_head_model(Variant::default());
    let path = temp_path("site_names", "onnx");
    std::fs::write(&path, model.encode_to_vec()).expect("write temp onnx");
    let named = extract_twin_spec_named(&path);
    let plain = extract_twin_spec(&path);
    std::fs::remove_file(&path).ok();
    let (spec, names) = named.expect("the cnn7 head extracts");
    let plain = plain.expect("the cnn7 head extracts");
    assert_eq!(
        format!("{spec:?}"),
        format!("{plain:?}"),
        "the spec is unchanged"
    );
    let relu_ops: Vec<usize> = spec
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, TwinOpSpec::Relu { .. }))
        .map(|(k, _)| k)
        .collect();
    assert_eq!(
        names.iter().map(|(op, _, _)| *op).collect::<Vec<_>>(),
        relu_ops,
        "one name per Relu op, in op order"
    );
    let graph = model.graph.as_ref().expect("graph");
    for (op, name, size) in &names {
        assert!(
            graph
                .node
                .iter()
                .any(|n| n.op_type == "Relu" && n.input.first() == Some(name)),
            "`{name}` is not a Relu node's input[0]"
        );
        let TwinOpSpec::Relu { input } = spec.ops[*op] else {
            panic!("op {op} is not a Relu");
        };
        let producer_size = if input == 0 {
            spec.n_in
        } else {
            match &spec.ops[input - 1] {
                TwinOpSpec::Conv { oshape, .. } => oshape.0 * oshape.1 * oshape.2,
                TwinOpSpec::Gemm { shape, .. } => shape.0,
                other => panic!("unexpected producer {other:?}"),
            }
        };
        assert_eq!(*size, producer_size);
    }
    // The folded BNs: the trunk Relu reads `cbn`, the head Relu reads `bn`.
    assert_eq!(names[0].1, "cbn");
    assert_eq!(names[1].1, "bn");
}
