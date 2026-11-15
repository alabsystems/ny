// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Dense scalar-V1 fixture and an independent exact ideal support reference.
//!
//! The reference proves a nominal rational inequality. Source must separately
//! charge its first/output errors and replay the original objective at runtime.

use ny_onnx::onnx_proto::{
    tensor_shape_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};

pub const INPUTS: usize = 30;
pub const HEADS: usize = 78;
pub const NEGATIVE: usize = 48;
const BLOCK: usize = 15;
const PER_BLOCK: usize = 24;
const SUPPORT: usize = 8;
const U: i64 = 65_536;
const D: i64 = 4_194_304;
const SUPPORT_DENOMINATOR: i64 = 8 * U * 64;

fn input(i: usize, permuted: bool) -> usize {
    if permuted {
        (11 * i + 9) % INPUTS
    } else {
        i
    }
}

fn head(h: usize, permuted: bool) -> usize {
    if permuted {
        (17 * h + 13) % HEADS
    } else {
        h
    }
}

/// Direct stored first-layer numerators, in input-major ONNX MatMul order.
/// They are authored from the closed formula, not a rounded matrix product.
fn first_numerators() -> Vec<i64> {
    let mut first = vec![0; INPUTS * HEADS];
    for i in 0..INPUTS {
        for p in 0..INPUTS {
            first[i * HEADS + p] = U + if i == p { D } else { 0 };
        }
        for block in 0..2 {
            for k in 0..PER_BLOCK {
                let h = INPUTS + PER_BLOCK * block + k;
                let inside = i / BLOCK == block;
                let p = inside && i % BLOCK == k % BLOCK;
                let q = inside && i % BLOCK == (k + 1) % BLOCK;
                let beta = if inside { U } else { 0 }
                    + if p { 64 } else { 0 }
                    + if q { (k + 1) as i64 } else { 0 };
                first[i * HEADS + h] = 15 * U + 64 + (k + 1) as i64 + 64 * beta;
            }
        }
    }
    first
}

fn positive_numerators() -> Vec<i64> {
    (0..INPUTS)
        .map(|i| {
            let coordinate = i % BLOCK;
            let perturbations: i64 = (0..PER_BLOCK)
                .map(|k| {
                    (if k % BLOCK == coordinate { 64 } else { 0 })
                        + if (k + 1) % BLOCK == coordinate {
                            (k + 1) as i64
                        } else {
                            0
                        }
                })
                .sum();
            PER_BLOCK as i64 * U + perturbations
        })
        .collect()
}

fn matrices(permuted: bool) -> (Vec<f32>, Vec<f32>) {
    let numerators = first_numerators();
    let positive = positive_numerators();
    let mut first = vec![0.0; INPUTS * HEADS];
    let mut final_weight = vec![0.0; HEADS * 2];
    for i in 0..INPUTS {
        for h in 0..HEADS {
            first[input(i, permuted) * HEADS + head(h, permuted)] =
                numerators[i * HEADS + h] as f32 / D as f32;
        }
    }
    for p in 0..INPUTS {
        final_weight[2 * head(p, permuted)] = positive[p] as f32 / U as f32;
    }
    for h in INPUTS..HEADS {
        final_weight[2 * head(h, permuted)] = -1.0;
    }
    (first, final_weight)
}

fn info(name: &str, width: i64) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: [1, width]
                        .into_iter()
                        .map(|n| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(n)),
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
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

pub fn model(bias: f32, permuted: bool) -> ModelProto {
    assert!(bias.is_finite());
    let (first, final_weight) = matrices(permuted);
    let mut graph = GraphProto {
        name: "authored_dense_allocated_scalar30".into(),
        input: vec![info("x", INPUTS as i64)],
        output: vec![info("output", 2)],
        initializer: vec![
            tensor("w1", &[INPUTS as i64, HEADS as i64], &first),
            tensor("b1", &[HEADS as i64], &vec![0.0; HEADS]),
            tensor("w2", &[HEADS as i64, 2], &final_weight),
            tensor("b2", &[2], &[bias, 0.0]),
        ],
        ..Default::default()
    };
    for (op, inputs, output) in [
        ("MatMul", vec!["x", "w1"], "pre_bias"),
        ("Add", vec!["pre_bias", "b1"], "pre_relu"),
        ("Relu", vec!["pre_relu"], "relu"),
        ("MatMul", vec!["relu", "w2"], "final_product"),
        ("Add", vec!["final_product", "b2"], "output"),
    ] {
        graph.node.push(NodeProto {
            op_type: op.into(),
            input: inputs.into_iter().map(str::to_owned).collect(),
            output: vec![output.into()],
            ..Default::default()
        });
    }
    ModelProto {
        ir_version: 7,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        ..Default::default()
    }
}

pub fn property() -> String {
    let mut text = String::new();
    for j in 0..INPUTS {
        text.push_str(&format!("(declare-const X_{j} Real)\n"));
    }
    text.push_str("(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n");
    for j in 0..INPUTS {
        text.push_str(&format!("(assert (>= X_{j} -1))\n(assert (<= X_{j} 1))\n"));
    }
    text.push_str("(assert (<= Y_0 Y_1))\n");
    text
}

/// Independent literal beta coefficients in units U, before x -> z.
fn beta_numerators(block: usize, k: usize) -> [i64; INPUTS] {
    let mut beta = [0; INPUTS];
    for i in block * BLOCK..(block + 1) * BLOCK {
        beta[i] = U;
    }
    beta[block * BLOCK + k % BLOCK] += 64;
    beta[block * BLOCK + (k + 1) % BLOCK] += (k + 1) as i64;
    beta
}

fn assert_encoding_and_geometry() {
    assert_eq!(HEADS, INPUTS + NEGATIVE);
    assert_eq!(NEGATIVE, 2 * PER_BLOCK);
    assert_eq!(SUPPORT_DENOMINATOR, 33_554_432);
    let first = first_numerators();
    let positive = positive_numerators();
    assert!(first.iter().all(|&n| n > 0 && n < 1 << 24));
    assert!(positive.iter().all(|&n| n > 0 && n < 1 << 24));
    for permuted in [false, true] {
        let mut inputs: Vec<_> = (0..INPUTS).map(|i| input(i, permuted)).collect();
        let mut heads: Vec<_> = (0..HEADS).map(|h| head(h, permuted)).collect();
        inputs.sort_unstable();
        heads.sort_unstable();
        assert_eq!(inputs, (0..INPUTS).collect::<Vec<_>>());
        assert_eq!(heads, (0..HEADS).collect::<Vec<_>>());
        let (encoded, output) = matrices(permuted);
        for i in 0..INPUTS {
            for h in 0..HEADS {
                let actual = encoded[input(i, permuted) * HEADS + head(h, permuted)];
                assert!(actual > 0.0);
                assert_eq!(f64::from(actual) * D as f64, first[i * HEADS + h] as f64);
            }
        }
        for h in 0..HEADS {
            let expected = if h < INPUTS { positive[h] } else { -U };
            assert_eq!(
                f64::from(output[2 * head(h, permuted)]) * U as f64,
                expected as f64
            );
            assert_eq!(output[2 * head(h, permuted) + 1], 0.0);
        }
    }
    // Positive first coefficients make the row-zero cross-product criterion
    // complete. Products are below2^48 and are evaluated as exact integers.
    for a in 0..HEADS {
        for b in a + 1..HEADS {
            assert!(
                (1..INPUTS)
                    .any(|i| first[i * HEADS + a] * first[b] != first[i * HEADS + b] * first[a]),
                "proportional heads {a}/{b}"
            );
        }
    }
    for block in 0..2 {
        let mut all_beta = [0; INPUTS];
        for k in 0..PER_BLOCK {
            let beta = beta_numerators(block, k);
            let total: i64 = beta.iter().sum();
            for i in 0..INPUTS {
                let n = INPUTS + PER_BLOCK * block + k;
                assert_eq!(first[i * HEADS + n], 64 * beta[i] + total);
                all_beta[i] += beta[i];
            }
        }
        for i in 0..INPUTS {
            assert_eq!(
                all_beta[i],
                if i / BLOCK == block { positive[i] } else { 0 }
            );
        }
    }
}

fn group_worst_loss(block: usize, first_k: usize) -> (i64, usize) {
    let betas: Vec<_> = (first_k..first_k + SUPPORT)
        .map(|k| beta_numerators(block, k))
        .collect();
    let mut all_beta = [0; INPUTS];
    for beta in &betas {
        for i in 0..INPUTS {
            all_beta[i] += beta[i];
        }
    }
    let mut worst = 0;
    let mut visits = 0;
    for mask in 0usize..1 << SUPPORT {
        let s = i64::from(mask.count_ones());
        let mut selected_beta = [0; INPUTS];
        for (bit, beta) in betas.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                for i in 0..INPUTS {
                    selected_beta[i] += beta[i];
                }
            }
        }
        let mut vnum = [0; INPUTS];
        for i in 0..INPUTS {
            vnum[i] = s * all_beta[i] - 8 * selected_beta[i];
        }
        let common: i64 = vnum.iter().sum();
        // z_i=x_i+(sum_j x_j)/64: the common residual is charged on ALL30
        // coordinates, including coordinates outside the selected block.
        let loss: i64 = vnum.iter().map(|&v| (64 * v + common).abs()).sum();
        if mask == 0 || mask == (1 << SUPPORT) - 1 {
            assert_eq!(loss, 0, "empty/full masks must actually complete");
        }
        worst = worst.max(loss);
        visits += 1;
    }
    (worst, visits)
}

pub fn assert_exact_reference() {
    assert_encoding_and_geometry();
    let mut worst_loss = 0;
    let mut visits = 0;
    let mut groups = 0;
    for block in 0..2 {
        for first_k in (0..PER_BLOCK).step_by(SUPPORT) {
            let (loss, count) = group_worst_loss(block, first_k);
            worst_loss += loss;
            visits += count;
            groups += 1;
        }
    }
    assert_eq!(groups, 6);
    assert_eq!(visits, 1536);
    assert!(
        16 * worst_loss <= SUPPORT_DENOMINATOR,
        "six-group ideal support loss exceeds1/16: {worst_loss}/{SUPPORT_DENOMINATOR}"
    );
    let safe_numerator = SUPPORT_DENOMINATOR / 2 - worst_loss;
    assert!(16 * safe_numerator >= 7 * SUPPORT_DENOMINATOR);
    // For nonnegative beta, ReLU(sum beta*z)<=sum beta*ReLU(z).
    // Positive output masses sum each block's24 beta rows exactly. Thus the
    // ideal margin is at least its bias; x=0 attains the bias. Bias+/-0.5
    // yields ideal minima+/-0.5; the six independently allocated rows retain
    // a lower bound >=7/16 for the safe sibling, before Source FP charges.
    for bias in [0.5f32, -0.5] {
        let authored = model(bias, false);
        let graph = authored.graph.unwrap();
        let b1 = graph.initializer.iter().find(|t| t.name == "b1").unwrap();
        assert!(b1.raw_data.iter().all(|&byte| byte == 0));
        let b2 = graph.initializer.iter().find(|t| t.name == "b2").unwrap();
        assert_eq!(&b2.raw_data[..4], &bias.to_le_bytes());
        assert_eq!(&b2.raw_data[4..], &0.0f32.to_le_bytes());
    }
}
