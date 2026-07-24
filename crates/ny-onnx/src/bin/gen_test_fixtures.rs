// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
use ny_onnx::onnx_proto;
use prost::Message;
use std::path::{Path, PathBuf};

const TEST_MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/models");

fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: value,
        s: Vec::new(),
        t: None,
        r#type: onnx_proto::attribute_type::INT,
        floats: Vec::new(),
        ints: Vec::new(),
    }
}

fn attr_ints(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: onnx_proto::attribute_type::INTS,
        floats: Vec::new(),
        ints: values.to_vec(),
    }
}

fn deterministic_weights(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + 17) % 23) as f32 - 11.0) * scale)
        .collect()
}

fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: attrs,
    }
}

fn write_onnx_model(path: &Path, graph: onnx_proto::GraphProto) {
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("Failed to encode ONNX");
    std::fs::write(path, buf).expect("Failed to write ONNX");
}

fn write_matmul_transpose_b_const(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![node(
            "matmul",
            "MatMul",
            &["input", "weight"],
            &["output"],
            vec![attr_int("transpose_b", 1)],
        )],
        name: "matmul_transpose_b_const".to_string(),
        initializer: vec![tensor_f32(
            "weight",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
        input: vec![tensor_value_info("input", &[1, 3])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_minimal_attention_core(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("q_matmul", "MatMul", &["x", "wq"], &["q"], Vec::new()),
            node("k_matmul", "MatMul", &["x", "wk"], &["k"], Vec::new()),
            node("v_matmul", "MatMul", &["x", "wv"], &["v"], Vec::new()),
            node(
                "scores",
                "MatMul",
                &["q", "k"],
                &["scores"],
                vec![attr_int("transpose_b", 1)],
            ),
            node("softmax", "Softmax", &["scores"], &["probs"], Vec::new()),
            node(
                "context",
                "MatMul",
                &["probs", "v"],
                &["context"],
                Vec::new(),
            ),
        ],
        name: "minimal_attention_core".to_string(),
        initializer: vec![
            tensor_f32("wq", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            tensor_f32("wk", &[2, 2], &[1.0, 0.5, -0.5, 1.0]),
            tensor_f32("wv", &[2, 2], &[0.2, 0.1, 0.3, -0.4]),
        ],
        input: vec![tensor_value_info("x", &[1, 2])],
        output: vec![tensor_value_info("context", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mul_binary(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        name: "mul_binary".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("a", &[1, 2]),
            tensor_value_info("b", &[1, 2]),
        ],
        output: vec![tensor_value_info("out", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mul_const_broadcast(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        name: "mul_const_broadcast".to_string(),
        initializer: vec![
            tensor_f32("a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            tensor_f32("b", &[2], &[5.0, 6.0]),
        ],
        input: Vec::new(),
        output: vec![tensor_value_info("out", &[2, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mul_binary_activation_inputs(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("relu", "Relu", &["x"], &["a"], Vec::new()),
            node("sigmoid", "Sigmoid", &["x"], &["b"], Vec::new()),
            node("mul", "Mul", &["a", "b"], &["out"], Vec::new()),
        ],
        name: "mul_binary_activation_inputs".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("x", &[1, 2])],
        output: vec![tensor_value_info("out", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mul_binary_activation_broadcast(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("relu", "Relu", &["x"], &["a"], Vec::new()),
            node("sigmoid", "Sigmoid", &["y"], &["b"], Vec::new()),
            node("mul", "Mul", &["a", "b"], &["out"], Vec::new()),
        ],
        name: "mul_binary_activation_broadcast".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("x", &[1, 2, 3]),
            tensor_value_info("y", &[1, 1, 3]),
        ],
        output: vec![tensor_value_info("out", &[1, 2, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mul_const_incompatible_shapes(path: &Path) {
    // Both inputs are constants with incompatible shapes for element-wise broadcast
    // This should trigger try_convert_mul returning None (line 317 in arithmetic.rs)
    let graph = onnx_proto::GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        name: "mul_const_incompatible_shapes".to_string(),
        initializer: vec![
            tensor_f32("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            tensor_f32("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
        input: Vec::new(),
        output: vec![tensor_value_info("out", &[2, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_single_linear(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![node(
            "gemm",
            "Gemm",
            &["input", "weight", "bias"],
            &["output"],
            vec![attr_int("transB", 1)],
        )],
        name: "single_linear".to_string(),
        initializer: vec![
            tensor_f32("weight", &[3, 2], &[1.0, 2.0, 3.0, -1.0, -2.0, 1.0]),
            tensor_f32("bias", &[3], &[0.5, -0.5, 1.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_linear_relu(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "gemm",
                "Gemm",
                &["input", "weight", "bias"],
                &["linear_out"],
                vec![attr_int("transB", 1)],
            ),
            node("relu", "Relu", &["linear_out"], &["output"], Vec::new()),
        ],
        name: "linear_relu".to_string(),
        initializer: vec![
            tensor_f32("weight", &[3, 2], &[1.0, 2.0, 3.0, -1.0, -2.0, 1.0]),
            tensor_f32("bias", &[3], &[0.5, -0.5, 1.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_simple_mlp(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "gemm1",
                "Gemm",
                &["input", "w1", "b1"],
                &["fc1_out"],
                vec![attr_int("transB", 1)],
            ),
            node("relu", "Relu", &["fc1_out"], &["relu_out"], Vec::new()),
            node(
                "gemm2",
                "Gemm",
                &["relu_out", "w2", "b2"],
                &["output"],
                vec![attr_int("transB", 1)],
            ),
        ],
        name: "simple_mlp".to_string(),
        initializer: vec![
            tensor_f32("w1", &[4, 2], &[1.0, 0.5, -1.0, 0.5, 0.5, 1.0, 0.5, -1.0]),
            tensor_f32("b1", &[4], &[0.1, 0.1, 0.1, 0.1]),
            tensor_f32("w2", &[2, 4], &[1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
            tensor_f32("b2", &[2], &[0.0, 0.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_single_conv2d(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![node(
            "conv",
            "Conv",
            &["input", "kernel", "bias"],
            &["output"],
            vec![
                attr_ints("kernel_shape", &[3, 3]),
                attr_ints("strides", &[1, 1]),
                attr_ints("pads", &[0, 0, 0, 0]),
            ],
        )],
        name: "single_conv2d".to_string(),
        initializer: vec![
            tensor_f32(
                "kernel",
                &[1, 1, 3, 3],
                &[-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
            ),
            tensor_f32("bias", &[1], &[0.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 5, 5])],
        output: vec![tensor_value_info("output", &[1, 1, 3, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_conv_relu(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "conv",
                "Conv",
                &["input", "kernel", "bias"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[2, 2]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("pads", &[0, 0, 0, 0]),
                ],
            ),
            node("relu", "Relu", &["conv_out"], &["output"], Vec::new()),
        ],
        name: "conv_relu".to_string(),
        initializer: vec![
            tensor_f32(
                "kernel",
                &[2, 1, 2, 2],
                &[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            ),
            tensor_f32("bias", &[2], &[0.0, 0.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 4, 4])],
        output: vec![tensor_value_info("output", &[1, 2, 3, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_mnist_conv(path: &Path) {
    let conv_weight = deterministic_weights(2 * 3 * 3, 0.08);
    let fc_weight = deterministic_weights(5 * 72, 0.025);
    let fc_bias = vec![0.0; 5];
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "conv",
                "Conv",
                &["input", "conv_kernel", "conv_bias"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("pads", &[0, 0, 0, 0]),
                ],
            ),
            node("relu", "Relu", &["conv_out"], &["relu_out"], Vec::new()),
            node(
                "flatten",
                "Flatten",
                &["relu_out"],
                &["flat_out"],
                vec![attr_int("axis", 1)],
            ),
            node(
                "gemm",
                "Gemm",
                &["flat_out", "fc_weight", "fc_bias"],
                &["output"],
                vec![attr_int("transB", 1)],
            ),
        ],
        name: "mnist_conv".to_string(),
        initializer: vec![
            tensor_f32("conv_kernel", &[2, 1, 3, 3], &conv_weight),
            tensor_f32("conv_bias", &[2], &[0.0, 0.0]),
            tensor_f32("fc_weight", &[5, 72], &fc_weight),
            tensor_f32("fc_bias", &[5], &fc_bias),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 8, 8])],
        output: vec![tensor_value_info("output", &[1, 5])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_duration_predictor_surrogate(path: &Path) {
    let weight = deterministic_weights(8 * 50, 0.01);
    let bias = vec![0.0; 50];
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "matmul",
                "MatMul",
                &["encoded_features", "weight"],
                &["matmul_out"],
                Vec::new(),
            ),
            node(
                "add",
                "Add",
                &["matmul_out", "bias"],
                &["duration_logits"],
                Vec::new(),
            ),
        ],
        name: "kokoro_duration_predictor_surrogate".to_string(),
        initializer: vec![
            tensor_f32("weight", &[8, 50], &weight),
            tensor_f32("bias", &[50], &bias),
        ],
        input: vec![tensor_value_info("encoded_features", &[1, 4, 8])],
        output: vec![tensor_value_info("duration_logits", &[1, 4, 50])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_conv_relu_maxpool(path: &Path) {
    let conv_weight = deterministic_weights(2 * 3 * 3, 0.05);
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "conv",
                "Conv",
                &["input", "conv_weight", "conv_bias"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("pads", &[1, 1, 1, 1]),
                ],
            ),
            node("relu", "Relu", &["conv_out"], &["relu_out"], Vec::new()),
            node(
                "maxpool",
                "MaxPool",
                &["relu_out"],
                &["output"],
                vec![
                    attr_ints("kernel_shape", &[2, 2]),
                    attr_ints("strides", &[2, 2]),
                ],
            ),
        ],
        name: "conv_relu_maxpool".to_string(),
        initializer: vec![
            tensor_f32("conv_weight", &[2, 1, 3, 3], &conv_weight),
            tensor_f32("conv_bias", &[2], &[0.0, 0.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 8, 8])],
        output: vec![tensor_value_info("output", &[1, 2, 4, 4])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_cnn_with_flatten(path: &Path) {
    let conv_weight = deterministic_weights(4 * 3 * 3, 0.04);
    let fc_weight = deterministic_weights(2 * 64, 0.02);
    let fc_bias = vec![0.0; 2];
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "conv",
                "Conv",
                &["input", "conv_weight", "conv_bias"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("pads", &[1, 1, 1, 1]),
                ],
            ),
            node("relu", "Relu", &["conv_out"], &["relu_out"], Vec::new()),
            node(
                "maxpool",
                "MaxPool",
                &["relu_out"],
                &["pool_out"],
                vec![
                    attr_ints("kernel_shape", &[2, 2]),
                    attr_ints("strides", &[2, 2]),
                ],
            ),
            node(
                "flatten",
                "Flatten",
                &["pool_out"],
                &["flat_out"],
                vec![attr_int("axis", 1)],
            ),
            node(
                "gemm",
                "Gemm",
                &["flat_out", "fc_weight", "fc_bias"],
                &["output"],
                vec![attr_int("transB", 1)],
            ),
        ],
        name: "cnn_with_flatten".to_string(),
        initializer: vec![
            tensor_f32("conv_weight", &[4, 1, 3, 3], &conv_weight),
            tensor_f32("conv_bias", &[4], &[0.0, 0.0, 0.0, 0.0]),
            tensor_f32("fc_weight", &[2, 64], &fc_weight),
            tensor_f32("fc_bias", &[2], &fc_bias),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 8, 8])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_test_cnn_maxpool(path: &Path) {
    // Companion to tests/models/test_cnn_maxpool.vnnlib: the same
    // Conv -> ReLU -> MaxPool -> Flatten -> Gemm net as cnn_with_flatten,
    // but paired with a 10x wider input box ([0.4, 0.6] per pixel). Over
    // that box the unsafe region (Y_0 < Y_1) is unreachable (multi-start
    // sign-descent puts the true min of Y_0 - Y_1 at ~ +0.084), yet root
    // CROWN cannot prove it, so the BaB loop must actually split (>= 2
    // explored domains) — exactly what the CNN BaB-loop integration test
    // exercises.
    let conv_weight = deterministic_weights(4 * 3 * 3, 0.04);
    let fc_weight = deterministic_weights(2 * 64, 0.02);
    let fc_bias = vec![0.0, 0.0];
    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "conv",
                "Conv",
                &["input", "conv_weight", "conv_bias"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[3, 3]),
                    attr_ints("strides", &[1, 1]),
                    attr_ints("pads", &[1, 1, 1, 1]),
                ],
            ),
            node("relu", "Relu", &["conv_out"], &["relu_out"], Vec::new()),
            node(
                "maxpool",
                "MaxPool",
                &["relu_out"],
                &["pool_out"],
                vec![
                    attr_ints("kernel_shape", &[2, 2]),
                    attr_ints("strides", &[2, 2]),
                ],
            ),
            node(
                "flatten",
                "Flatten",
                &["pool_out"],
                &["flat_out"],
                vec![attr_int("axis", 1)],
            ),
            node(
                "gemm",
                "Gemm",
                &["flat_out", "fc_weight", "fc_bias"],
                &["output"],
                vec![attr_int("transB", 1)],
            ),
        ],
        name: "test_cnn_maxpool".to_string(),
        initializer: vec![
            tensor_f32("conv_weight", &[4, 1, 3, 3], &conv_weight),
            tensor_f32("conv_bias", &[4], &[0.0, 0.0, 0.0, 0.0]),
            tensor_f32("fc_weight", &[2, 64], &fc_weight),
            tensor_f32("fc_bias", &[2], &fc_bias),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 8, 8])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn write_const_activation_binary_op(path: &Path) {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("relu", "Relu", &["input"], &["activation"], Vec::new()),
            node("neg", "Neg", &["constant"], &["neg_constant"], Vec::new()),
            node(
                "add",
                "Add",
                &["activation", "neg_constant"],
                &["output"],
                Vec::new(),
            ),
        ],
        name: "const_activation_binary_op".to_string(),
        initializer: vec![tensor_f32("constant", &[1, 2], &[1.0, -0.5])],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    write_onnx_model(path, graph);
}

fn ensure_models_dir() -> PathBuf {
    let dir = PathBuf::from(TEST_MODELS_DIR);
    std::fs::create_dir_all(&dir).expect("Failed to create tests/models directory");
    dir
}

fn main() {
    let dir = ensure_models_dir();
    let fixtures: &[(&str, fn(&Path))] = &[
        ("single_linear.onnx", write_single_linear),
        ("linear_relu.onnx", write_linear_relu),
        ("simple_mlp.onnx", write_simple_mlp),
        ("single_conv2d.onnx", write_single_conv2d),
        ("conv_relu.onnx", write_conv_relu),
        ("mnist_conv.onnx", write_mnist_conv),
        (
            "matmul_transpose_b_const.onnx",
            write_matmul_transpose_b_const,
        ),
        ("minimal_attention_core.onnx", write_minimal_attention_core),
        (
            "kokoro_duration_predictor_surrogate.onnx",
            write_duration_predictor_surrogate,
        ),
        ("conv_relu_maxpool.onnx", write_conv_relu_maxpool),
        ("cnn_with_flatten.onnx", write_cnn_with_flatten),
        ("test_cnn_maxpool.onnx", write_test_cnn_maxpool),
        (
            "const_activation_binary_op.onnx",
            write_const_activation_binary_op,
        ),
        ("mul_binary.onnx", write_mul_binary),
        ("mul_const_broadcast.onnx", write_mul_const_broadcast),
        (
            "mul_binary_activation_inputs.onnx",
            write_mul_binary_activation_inputs,
        ),
        (
            "mul_binary_activation_broadcast.onnx",
            write_mul_binary_activation_broadcast,
        ),
        (
            "mul_const_incompatible_shapes.onnx",
            write_mul_const_incompatible_shapes,
        ),
    ];

    for (name, write_fixture) in fixtures {
        let path = dir.join(name);
        write_fixture(&path);
        println!("Wrote {}", path.display());
    }
}
