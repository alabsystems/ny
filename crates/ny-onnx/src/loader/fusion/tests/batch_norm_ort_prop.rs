// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adversarial enclosure tests for the extended BN folds
//! (#cgan-structural-fold): 500 random ConvTranspose+BN pairs, 500 random
//! Gemm+Reshape+BN triples, and 500 random BN+Reshape+Gemm forward tails, each
//! checked against ONNX Runtime on the ORIGINAL (unfolded) graph.
//!
//! Soundness gate (the moat): the fold rewrites network WEIGHTS, so a folded
//! network that disagrees with the unfolded one is a wrong-verdict factory.
//! The landed Conv/Gemm BN-fold convention (mirrored here and in
//! alpha-beta-CROWN's onnx_opt.py) is: the fold equations are EXACT in real
//! arithmetic and are evaluated once in f32. The acceptance gate therefore is:
//!
//!   for random weights & points, ORT(unfolded, x) must lie within the folded
//!   network's outward-rounded point-box IBP interval, widened by
//!   `FOLD_TOL_REL * max(1, |y|)` — slack for (a) the f32 rounding of the fold
//!   equations themselves and (b) ORT-vs-ny summation-order differences that
//!   exist even WITHOUT folding. FOLD_TOL_REL = 1e-4; a wrong channel axis or
//!   a wrong block map produces order-1 errors with random per-channel scales,
//!   so the gate is far tighter than any layout bug and far looser than
//!   legitimate f32 round-off (measured ~1e-6 relative on these magnitudes).
//!
//! Determinism: a fixed-seed xorshift64* generator, so every run checks the
//! same 1500 cases and a failure is exactly reproducible from its case index.

use crate::onnx_proto::{
    self, attribute_type, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ndarray::ArrayD;
use ny_core::LayerType;
use ny_tensor::BoundedTensor;
use prost::Message;

const CASES_PER_PATTERN: usize = 500;
const FOLD_TOL_REL: f32 = 1e-4;

/// xorshift64* — deterministic, dependency-free case generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in [lo, hi] inclusive.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// Uniform f32 in [lo, hi].
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + unit * (hi - lo)
    }

    fn f32_vec(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| self.f32_in(lo, hi)).collect()
    }

    fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn f32_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&value| onnx_proto::tensor_shape_proto::Dimension {
                            value: Some(
                                onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value),
                            ),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn f32_initializer(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name: name.to_string(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn i64_initializer(name: &str, dims: &[i64], data: &[i64]) -> TensorProto {
    let mut raw_data = Vec::with_capacity(data.len() * 8);
    for value in data {
        raw_data.extend_from_slice(&value.to_le_bytes());
    }
    TensorProto {
        dims: dims.to_vec(),
        data_type: 7,
        name: name.to_string(),
        raw_data,
        ..Default::default()
    }
}

fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: format!("{op_type}_node"),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn ints_attr(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INT,
        i: value,
        ..Default::default()
    }
}

fn float_attr(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::FLOAT,
        f: value,
        ..Default::default()
    }
}

fn encode_model(graph: GraphProto) -> Vec<u8> {
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "ny-bn-fold-prop".to_string(),
        graph: Some(graph),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode test model");
    bytes
}

/// Random BN parameters over `channels`, with adversarial gammas (negative /
/// near-zero included) and the cgan-style epsilon spread ({0.8, 1e-3, 1e-5}).
fn random_bn_initializers(rng: &mut Rng, channels: usize) -> (Vec<TensorProto>, f32) {
    let gamma = rng.f32_vec(channels, -1.5, 1.5);
    let beta = rng.f32_vec(channels, -1.0, 1.0);
    let mean = rng.f32_vec(channels, -1.0, 1.0);
    let var = rng.f32_vec(channels, 0.1, 2.0);
    let epsilon = [0.8_f32, 1e-3, 1e-5][rng.below(3)];
    (
        vec![
            f32_initializer("bn_gamma", &[channels as i64], &gamma),
            f32_initializer("bn_beta", &[channels as i64], &beta),
            f32_initializer("bn_mean", &[channels as i64], &mean),
            f32_initializer("bn_var", &[channels as i64], &var),
        ],
        epsilon,
    )
}

/// Random ConvTranspose+BN model. Returns (model bytes, input point).
fn random_conv_transpose_bn_case(rng: &mut Rng) -> (Vec<u8>, ArrayD<f32>) {
    let c_in = rng.range(1, 3);
    let c_out = rng.range(1, 4);
    let k = rng.range(1, 3);
    let stride = rng.range(1, 2);
    let pad = if k >= 3 { rng.range(0, 1) } else { 0 };
    let h = rng.range(2, 4);
    let w = rng.range(2, 4);
    let h_out = (h - 1) * stride + k - 2 * pad;
    let w_out = (w - 1) * stride + k - 2 * pad;
    let with_bias = rng.bool();

    let kernel = rng.f32_vec(c_in * c_out * k * k, -1.5, 1.5);
    let mut initializers = vec![f32_initializer(
        "ct_w",
        &[c_in as i64, c_out as i64, k as i64, k as i64],
        &kernel,
    )];
    let mut ct_inputs = vec!["x", "ct_w"];
    if with_bias {
        let bias = rng.f32_vec(c_out, -1.0, 1.0);
        initializers.push(f32_initializer("ct_b", &[c_out as i64], &bias));
        ct_inputs.push("ct_b");
    }
    let (bn_inits, epsilon) = random_bn_initializers(rng, c_out);
    initializers.extend(bn_inits);

    let mut ct = node("ConvTranspose", &ct_inputs, &["ct_y"]);
    ct.attribute
        .push(ints_attr("kernel_shape", &[k as i64, k as i64]));
    ct.attribute
        .push(ints_attr("strides", &[stride as i64, stride as i64]));
    ct.attribute.push(ints_attr(
        "pads",
        &[pad as i64, pad as i64, pad as i64, pad as i64],
    ));
    ct.attribute.push(ints_attr("dilations", &[1, 1]));
    ct.attribute.push(int_attr("group", 1));

    let mut bn = node(
        "BatchNormalization",
        &["ct_y", "bn_gamma", "bn_beta", "bn_mean", "bn_var"],
        &["y"],
    );
    bn.attribute.push(float_attr("epsilon", epsilon));

    let graph = GraphProto {
        node: vec![ct, bn],
        name: "convt_bn_prop".to_string(),
        initializer: initializers,
        input: vec![f32_value_info("x", &[1, c_in as i64, h as i64, w as i64])],
        output: vec![f32_value_info(
            "y",
            &[1, c_out as i64, h_out as i64, w_out as i64],
        )],
        ..Default::default()
    };

    let point = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, c_in, h, w]),
        rng.f32_vec(c_in * h * w, -1.0, 1.0),
    )
    .expect("input point");
    (encode_model(graph), point)
}

/// Random Gemm+Reshape+BN model ([1, in] -> Gemm [1, F=C*bh*bw] -> Reshape
/// [-1, C, bh, bw] -> BN). Returns (model bytes, input point).
fn random_gemm_reshape_bn_case(rng: &mut Rng) -> (Vec<u8>, ArrayD<f32>) {
    let in_dim = rng.range(1, 5);
    let channels = rng.range(1, 4);
    let bh = rng.range(1, 3);
    let bw = rng.range(1, 3);
    let features = channels * bh * bw;
    let trans_b = rng.bool();
    let with_bias = rng.bool();
    // Alternate batch-entry encodings for the reshape target, as seen in the
    // cgan fleet: Constant-node-backed vs initializer, -1 vs literal 1.
    let batch_entry = if rng.bool() { -1_i64 } else { 1 };

    let weight = rng.f32_vec(features * in_dim, -1.5, 1.5);
    let weight_dims: [i64; 2] = if trans_b {
        [features as i64, in_dim as i64]
    } else {
        [in_dim as i64, features as i64]
    };
    let mut initializers = vec![f32_initializer("gemm_w", &weight_dims, &weight)];
    let mut gemm_inputs = vec!["x", "gemm_w"];
    if with_bias {
        let bias = rng.f32_vec(features, -1.0, 1.0);
        initializers.push(f32_initializer("gemm_b", &[features as i64], &bias));
        gemm_inputs.push("gemm_b");
    }
    initializers.push(i64_initializer(
        "target_shape",
        &[4],
        &[batch_entry, channels as i64, bh as i64, bw as i64],
    ));
    let (bn_inits, epsilon) = random_bn_initializers(rng, channels);
    initializers.extend(bn_inits);

    let mut gemm = node("Gemm", &gemm_inputs, &["gemm_y"]);
    gemm.attribute.push(int_attr("transB", i64::from(trans_b)));
    gemm.attribute.push(float_attr("alpha", 1.0));
    gemm.attribute.push(float_attr("beta", 1.0));

    let reshape = node("Reshape", &["gemm_y", "target_shape"], &["reshape_y"]);

    let mut bn = node(
        "BatchNormalization",
        &["reshape_y", "bn_gamma", "bn_beta", "bn_mean", "bn_var"],
        &["y"],
    );
    bn.attribute.push(float_attr("epsilon", epsilon));

    let graph = GraphProto {
        node: vec![gemm, reshape, bn],
        name: "gemm_reshape_bn_prop".to_string(),
        initializer: initializers,
        input: vec![f32_value_info("x", &[1, in_dim as i64])],
        output: vec![f32_value_info(
            "y",
            &[1, channels as i64, bh as i64, bw as i64],
        )],
        ..Default::default()
    };

    let point =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, in_dim]), rng.f32_vec(in_dim, -1.0, 1.0))
            .expect("input point");
    (encode_model(graph), point)
}

#[derive(Clone, Copy)]
struct TailGemmEncoding {
    trans_b: bool,
    explicit_affine_defaults: bool,
}

/// Random BN+Reshape+Gemm forward-tail model
/// (`[1,C,H,W] -> BN -> Reshape [1,F] -> Gemm [1,out]`). The case index
/// deterministically crosses both Gemm weight layouts with implicit/explicit
/// exact affine defaults, both accepted exporter reshape targets, and every
/// row-invariant Gemm C representation accepted by the fold.
fn random_bn_reshape_gemm_case(
    rng: &mut Rng,
    case: usize,
) -> (Vec<u8>, ArrayD<f32>, TailGemmEncoding) {
    let channels = rng.range(1, 4);
    let height = rng.range(1, 3);
    let width = rng.range(1, 3);
    let features = channels * height * width;
    let outputs = rng.range(1, 5);

    // Low case bits form a deterministic coverage matrix rather than leaving
    // the important Gemm encodings to probabilistic sampling.
    let trans_b = case & 1 != 0;
    let explicit_affine_defaults = case & 2 != 0;
    let explicit_trans_b = trans_b || case & 4 != 0;
    let target = if case & 8 == 0 {
        vec![-1_i64, features as i64]
    } else {
        vec![1_i64, -1]
    };

    let weight = rng.f32_vec(features * outputs, -1.5, 1.5);
    let weight_dims: [i64; 2] = if trans_b {
        [outputs as i64, features as i64]
    } else {
        [features as i64, outputs as i64]
    };
    let mut initializers = vec![
        f32_initializer("gemm_w", &weight_dims, &weight),
        i64_initializer("target_shape", &[2], &target),
    ];
    let mut gemm_inputs = vec!["flat_y", "gemm_w"];
    match (case >> 4) & 3 {
        0 => {}
        1 => {
            initializers.push(f32_initializer("gemm_b", &[], &[rng.f32_in(-1.0, 1.0)]));
            gemm_inputs.push("gemm_b");
        }
        2 => {
            initializers.push(f32_initializer(
                "gemm_b",
                &[outputs as i64],
                &rng.f32_vec(outputs, -1.0, 1.0),
            ));
            gemm_inputs.push("gemm_b");
        }
        _ => {
            initializers.push(f32_initializer(
                "gemm_b",
                &[1, outputs as i64],
                &rng.f32_vec(outputs, -1.0, 1.0),
            ));
            gemm_inputs.push("gemm_b");
        }
    }
    let (bn_inits, epsilon) = random_bn_initializers(rng, channels);
    initializers.extend(bn_inits);

    let mut bn = node(
        "BatchNormalization",
        &["x", "bn_gamma", "bn_beta", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(float_attr("epsilon", epsilon));
    let reshape = node("Reshape", &["bn_y", "target_shape"], &["flat_y"]);
    let mut gemm = node("Gemm", &gemm_inputs, &["y"]);
    if explicit_trans_b {
        gemm.attribute.push(int_attr("transB", i64::from(trans_b)));
    }
    if explicit_affine_defaults {
        gemm.attribute.push(float_attr("alpha", 1.0));
        gemm.attribute.push(float_attr("beta", 1.0));
    }

    let graph = GraphProto {
        node: vec![bn, reshape, gemm],
        name: "bn_reshape_gemm_prop".to_string(),
        initializer: initializers,
        input: vec![f32_value_info(
            "x",
            &[1, channels as i64, height as i64, width as i64],
        )],
        output: vec![f32_value_info("y", &[1, outputs as i64])],
        ..Default::default()
    };

    let point = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, channels, height, width]),
        rng.f32_vec(features, -1.0, 1.0),
    )
    .expect("input point");
    (
        encode_model(graph),
        point,
        TailGemmEncoding {
            trans_b,
            explicit_affine_defaults,
        },
    )
}

/// Core check for one case: the fold fired, and ORT on the ORIGINAL graph is
/// enclosed by the folded network's point-box IBP within FOLD_TOL_REL.
fn assert_folded_encloses_ort(case: usize, model_bytes: &[u8], point: &ArrayD<f32>) {
    // ORT reference on the unfolded bytes.
    let ort_outputs =
        crate::diff::run_inference_bytes(model_bytes, point).expect("ORT inference on test model");
    let ort_y = ort_outputs.first().expect("ORT output");

    // NY load — the extended fold fires here (default ON).
    let model = crate::load_onnx_bytes("bn_fold_prop", model_bytes).expect("NY load of test model");
    assert!(
        !model
            .network
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::BatchNorm),
        "case {case}: BatchNormalization survived the fold (layers: {:?})",
        model
            .network
            .layers
            .iter()
            .map(|layer| format!("{:?}", layer.layer_type))
            .collect::<Vec<_>>()
    );

    // Point-box IBP through the folded graph (unbatched input).
    let graph = model.to_graph_network().expect("graph conversion");
    let unbatched: Vec<usize> = point.shape()[1..].to_vec();
    let point_unbatched = point
        .clone()
        .into_shape_with_order(ndarray::IxDyn(&unbatched))
        .expect("strip batch axis");
    let bounds =
        BoundedTensor::new(point_unbatched.clone(), point_unbatched).expect("degenerate point box");
    let output = graph.propagate_ibp(&bounds).expect("folded point IBP");

    let lower = output.lower();
    let upper = output.upper();
    assert_eq!(
        lower.len(),
        ort_y.len(),
        "case {case}: output arity mismatch (ny {} vs ort {})",
        lower.len(),
        ort_y.len()
    );
    for (idx, ((l, u), y)) in lower.iter().zip(upper.iter()).zip(ort_y.iter()).enumerate() {
        let tol = FOLD_TOL_REL * y.abs().max(1.0);
        assert!(
            (l - tol) <= *y && *y <= (u + tol),
            "case {case} elem {idx}: ORT (unfolded) {y} escapes folded point-IBP \
             [{l}, {u}] beyond tol {tol}"
        );
        // Midpoint closeness: catches a fold that silently widens instead of
        // mis-centering (enclosure alone would tolerate arbitrarily wide boxes).
        let mid = 0.5 * (l + u);
        assert!(
            (mid - y).abs() <= tol,
            "case {case} elem {idx}: folded midpoint {mid} vs ORT {y} beyond tol {tol}"
        );
    }
}

#[test]
fn prop_conv_transpose_bn_fold_encloses_ort_500() {
    let mut rng = Rng::new(0x000C_6A72_2023);
    for case in 0..CASES_PER_PATTERN {
        let (bytes, point) = random_conv_transpose_bn_case(&mut rng);
        assert_folded_encloses_ort(case, &bytes, &point);
    }
}

#[test]
fn prop_gemm_reshape_bn_fold_encloses_ort_500() {
    let mut rng = Rng::new(0x6E44_5245_5348);
    for case in 0..CASES_PER_PATTERN {
        let (bytes, point) = random_gemm_reshape_bn_case(&mut rng);
        assert_folded_encloses_ort(case, &bytes, &point);
    }
}

#[test]
fn prop_bn_reshape_gemm_fold_encloses_ort_500() {
    let mut rng = Rng::new(0xB1A5_7A11_6E44);
    let mut covered = [[false; 2]; 2];
    for case in 0..CASES_PER_PATTERN {
        let (bytes, point, encoding) = random_bn_reshape_gemm_case(&mut rng, case);
        covered[usize::from(encoding.trans_b)][usize::from(encoding.explicit_affine_defaults)] =
            true;
        assert_folded_encloses_ort(case, &bytes, &point);
    }
    assert_eq!(
        covered,
        [[true, true], [true, true]],
        "both transB layouts must be covered with implicit and explicit exact defaults"
    );
}

/// Real-net probe (run explicitly): on an actual cgan_2023 generator network,
/// verify (a) the extended folds remove every generator BN (head Gemm->
/// Reshape->BN + all ConvTranspose->BN pairs), (b) the folded network still
/// encloses ORT's unfolded forward at the input-box midpoint, and (c) the
/// root IBP bounds with the folds ON are equal-or-tighter than with the folds
/// OFF (a single fused affine interval map is algebraically contained in the
/// composition of the two interval maps it replaces).
///
/// Usage:
///   NY_CGAN_PROBE_ONNX=.../cGAN_imgSz32_nCh_1.onnx \
///   NY_CGAN_PROBE_VNNLIB=.../cGAN_imgSz32_nCh_1_prop_0_....vnnlib \
///   cargo test -p ny-onnx --lib cgan_probe_fold_root_bounds -- --ignored --nocapture
#[test]
#[ignore = "needs NY_CGAN_PROBE_ONNX / NY_CGAN_PROBE_VNNLIB pointing at benchmark files"]
fn cgan_probe_fold_root_bounds() {
    let onnx_path = std::env::var("NY_CGAN_PROBE_ONNX").expect("NY_CGAN_PROBE_ONNX not set");
    let vnnlib_path = std::env::var("NY_CGAN_PROBE_VNNLIB").expect("NY_CGAN_PROBE_VNNLIB not set");
    let bytes = std::fs::read(&onnx_path).expect("read probe onnx");
    let spec = crate::vnnlib::load_vnnlib(&vnnlib_path).expect("parse probe vnnlib");

    let bn_count = |model: &crate::OnnxModel| {
        model
            .network
            .layers
            .iter()
            .filter(|layer| layer.layer_type == LayerType::BatchNorm)
            .count()
    };

    // Serialized env scope (clippy env wall); pre-test state restored on exit.
    let (model_off, model_on) = ny_test_utils::env::with_env_edits(|env| {
        // Fold OFF (kill switch).
        env.set("NY_BN_FOLD_EXT", "0");
        let model_off = crate::load_onnx_bytes("cgan_probe_off", &bytes).expect("load (fold off)");
        // Fold ON (default).
        env.remove("NY_BN_FOLD_EXT");
        let model_on = crate::load_onnx_bytes("cgan_probe_on", &bytes).expect("load (fold on)");
        (model_off, model_on)
    });

    let off_bns = bn_count(&model_off);
    let on_bns = bn_count(&model_on);
    println!("BatchNorm layers: fold OFF = {off_bns}, fold ON = {on_bns}");
    assert!(
        on_bns < off_bns,
        "extended folds should remove generator BN layers ({on_bns} vs {off_bns})"
    );

    // Input box from the vnnlib property (unbatched [num_inputs]).
    let num_inputs = spec.num_inputs;
    let lower: Vec<f32> = spec.input_bounds.iter().map(|(l, _)| *l as f32).collect();
    let upper: Vec<f32> = spec.input_bounds.iter().map(|(_, u)| *u as f32).collect();
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[num_inputs]), lower).expect("lb");
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[num_inputs]), upper).expect("ub");
    let mid: ArrayD<f32> = (&lower + &upper) * 0.5;
    let box_input = BoundedTensor::new(lower, upper).expect("input box");

    // ORT reference at the midpoint on the ORIGINAL bytes (batched [1, n]).
    let mid_batched = mid
        .clone()
        .into_shape_with_order(ndarray::IxDyn(&[1, num_inputs]))
        .expect("batched midpoint");
    let ort_y = crate::diff::run_inference_bytes(&bytes, &mid_batched).expect("ORT forward");
    let ort_y = ort_y.first().expect("ORT output");

    let graph_on = model_on.to_graph_network().expect("graph (fold on)");
    let graph_off = model_off.to_graph_network().expect("graph (fold off)");

    // (b) point enclosure on the real net.
    let point_box = BoundedTensor::new(mid.clone(), mid).expect("point box");
    let point_out = graph_on
        .propagate_ibp(&point_box)
        .expect("point IBP (fold on)");
    for (idx, ((l, u), y)) in point_out
        .lower()
        .iter()
        .zip(point_out.upper().iter())
        .zip(ort_y.iter())
        .enumerate()
    {
        let tol = FOLD_TOL_REL * y.abs().max(1.0);
        assert!(
            (l - tol) <= *y && *y <= (u + tol),
            "elem {idx}: ORT {y} escapes folded point-IBP [{l}, {u}] (tol {tol})"
        );
    }
    println!("point enclosure vs ORT: OK ({} outputs)", ort_y.len());

    // (c) root IBP tightness, fold ON vs OFF.
    let root_on = graph_on
        .propagate_ibp(&box_input)
        .expect("root IBP (fold on)");
    let root_off = graph_off
        .propagate_ibp(&box_input)
        .expect("root IBP (fold off)");
    for (idx, (((l_on, u_on), l_off), u_off)) in root_on
        .lower()
        .iter()
        .zip(root_on.upper().iter())
        .zip(root_off.lower().iter())
        .zip(root_off.upper().iter())
        .enumerate()
    {
        println!(
            "root IBP Y_{idx}: fold ON [{l_on:.6e}, {u_on:.6e}]  fold OFF [{l_off:.6e}, {u_off:.6e}]"
        );
        let slack_l = FOLD_TOL_REL * l_off.abs().max(1.0);
        let slack_u = FOLD_TOL_REL * u_off.abs().max(1.0);
        assert!(
            *l_on >= l_off - slack_l && *u_on <= u_off + slack_u,
            "elem {idx}: folded root bounds looser than unfolded beyond fold tolerance"
        );
    }
}
