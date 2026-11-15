// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-attack-twin — the twin is type-segregated from every bound API, and
//! on a model whose only obstacle is a rounded constant `MatMul` it folds that
//! cone, admits the exact point-VJP, and differentiates the authored function.

use std::marker::PhantomData;

use ny_propagate::GraphNetwork;

use super::PointFoldedAttackTwin;

/// Compile-time trait probe (the `impls` crate idiom): an inherent associated
/// `const` takes precedence over the trait default ONLY when its impl's bound
/// holds, so `Probe::<T>::X` reads `true` exactly when `T` implements the bound.
#[allow(dead_code)]
struct Probe<T: ?Sized>(PhantomData<T>);

trait Fallback {
    const DEREFS_TO_GRAPH: bool = false;
    const AS_REF_GRAPH: bool = false;
    const BORROWS_GRAPH: bool = false;
    const INTO_GRAPH: bool = false;
    const INTO_ONNX_MODEL: bool = false;
}

impl<T: ?Sized> Fallback for Probe<T> {}

impl<T: ?Sized + std::ops::Deref<Target = GraphNetwork>> Probe<T> {
    const DEREFS_TO_GRAPH: bool = true;
}

impl<T: ?Sized + AsRef<GraphNetwork>> Probe<T> {
    const AS_REF_GRAPH: bool = true;
}

impl<T: ?Sized + std::borrow::Borrow<GraphNetwork>> Probe<T> {
    const BORROWS_GRAPH: bool = true;
}

impl<T: Into<GraphNetwork>> Probe<T> {
    const INTO_GRAPH: bool = true;
}

impl<T: Into<crate::OnnxModel>> Probe<T> {
    const INTO_ONNX_MODEL: bool = true;
}

/// SOUNDNESS: a point fold is not an enclosure, so the twin must never reach a
/// bound API. Every bound path takes a `GraphNetwork` (by reference or by
/// value, or an `OnnxModel` to build one); the twin reaches none of them by
/// deref, `AsRef`, `Borrow` or conversion. The positive controls prove the probe
/// is not vacuous.
#[test]
fn attack_twin_is_type_segregated_from_every_bound_api() {
    // Positive controls: types that DO reach a GraphNetwork are detected, so
    // the probe is not vacuous.
    let controls = [
        (
            "Box<GraphNetwork>: Deref",
            Probe::<Box<GraphNetwork>>::DEREFS_TO_GRAPH,
        ),
        (
            "Box<GraphNetwork>: AsRef",
            Probe::<Box<GraphNetwork>>::AS_REF_GRAPH,
        ),
        (
            "Box<GraphNetwork>: Borrow",
            Probe::<Box<GraphNetwork>>::BORROWS_GRAPH,
        ),
        (
            "GraphNetwork: Into<GraphNetwork>",
            Probe::<GraphNetwork>::INTO_GRAPH,
        ),
        (
            "OnnxModel: Into<OnnxModel>",
            Probe::<crate::OnnxModel>::INTO_ONNX_MODEL,
        ),
    ];
    for (what, detected) in controls {
        assert!(detected, "probe control failed to detect {what}");
    }

    // The twin, and a reference to it, reach none of them.
    let twin_routes = [
        (
            "Deref<Target = GraphNetwork>",
            Probe::<PointFoldedAttackTwin>::DEREFS_TO_GRAPH,
        ),
        (
            "AsRef<GraphNetwork>",
            Probe::<PointFoldedAttackTwin>::AS_REF_GRAPH,
        ),
        (
            "Borrow<GraphNetwork>",
            Probe::<PointFoldedAttackTwin>::BORROWS_GRAPH,
        ),
        (
            "Into<GraphNetwork>",
            Probe::<PointFoldedAttackTwin>::INTO_GRAPH,
        ),
        (
            "Into<OnnxModel>",
            Probe::<PointFoldedAttackTwin>::INTO_ONNX_MODEL,
        ),
        (
            "&twin: Deref<Target = GraphNetwork>",
            Probe::<&PointFoldedAttackTwin>::DEREFS_TO_GRAPH,
        ),
        (
            "&twin: AsRef<GraphNetwork>",
            Probe::<&PointFoldedAttackTwin>::AS_REF_GRAPH,
        ),
        (
            "&twin: Borrow<GraphNetwork>",
            Probe::<&PointFoldedAttackTwin>::BORROWS_GRAPH,
        ),
    ];
    for (route, reaches) in twin_routes {
        assert!(
            !reaches,
            "the point-folded ATTACK twin must not implement {route}: a point fold is not \
             an enclosure, and that route would hand it to a bound API"
        );
    }
}

#[cfg(feature = "ort")]
mod with_ort {
    use ndarray::{Array2, ArrayD, IxDyn};
    use prost::Message;

    use crate::onnx_proto::{
        tensor_shape_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
        TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use crate::{
        load_onnx_bytes_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
        ShapeInferBackend,
    };

    use super::super::PointFoldedAttackTwin;

    const FLOAT: i32 = 1;
    const INT64: i32 = 7;

    /// Constant factors whose binary32 products are NOT exactly representable,
    /// so the loader's exact-fold gate declines `A @ B` and fails it closed.
    const A: [f32; 4] = [0.1, 0.3, 0.7, 0.9];
    const B: [f32; 4] = [0.3, 0.6, 0.2, 0.7];

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn matrix(name: &str, values: &[f32; 4]) -> TensorProto {
        TensorProto {
            dims: vec![2, 2],
            data_type: FLOAT,
            name: name.to_string(),
            float_data: values.to_vec(),
            ..TensorProto::default()
        }
    }

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|s| (*s).to_string()).collect(),
            output: vec![output.to_string()],
            name: format!("{op}_{output}"),
            op_type: op.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    /// `V = sum(x ⊙ (x @ (A @ B)))` over `x: [1, 2]` — the Lyapunov shape, with
    /// its matrix produced by a rounded constant-only `MatMul`.
    fn lyapunov_model() -> Vec<u8> {
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(GraphProto {
                node: vec![
                    node("MatMul", &["a", "b"], "p"),
                    node("MatMul", &["x", "p"], "xp"),
                    node("Mul", &["x", "xp"], "xpx"),
                    node("ReduceSum", &["xpx", "axes"], "v"),
                ],
                name: "lyapunov".to_string(),
                initializer: vec![
                    matrix("a", &A),
                    matrix("b", &B),
                    TensorProto {
                        dims: vec![1],
                        data_type: INT64,
                        name: "axes".to_string(),
                        int64_data: vec![1],
                        ..TensorProto::default()
                    },
                ],
                input: vec![value_info("x", &[1, 2])],
                output: vec![value_info("v", &[1, 1])],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        }
        .encode_to_vec()
    }

    fn load_config() -> OnnxLoadConfig {
        OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess)
    }

    /// `P = A @ B` in f64 from the binary32 factors (row-major 2x2).
    fn exact_p() -> [[f64; 2]; 2] {
        let mut p = [[0.0_f64; 2]; 2];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = (0..2)
                    .map(|k| f64::from(A[2 * i + k]) * f64::from(B[2 * k + j]))
                    .sum();
            }
        }
        p
    }

    #[test]
    fn attack_twin_point_folds_a_rounded_constant_matmul_cone() {
        let bytes = lyapunov_model();

        // The REAL graph is not attackable by gradient: the rounded constant
        // product fails closed, even under the widened fragment.
        let real_attackable = load_onnx_bytes_with_config("real", &bytes, &load_config())
            .ok()
            .and_then(|model| {
                model
                    .to_graph_network_with_options(GraphNetworkOptions {
                        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
                        ..GraphNetworkOptions::default()
                    })
                    .ok()
            })
            .is_some_and(|graph| graph.supports_attack_point_gradient_with_exact_reductions());
        assert!(
            !real_attackable,
            "the fixture must reproduce the failure the twin exists for"
        );

        let twin = PointFoldedAttackTwin::build_from_bytes(&bytes, 2, &load_config())
            .unwrap_or_else(|error| panic!("the twin must build: {error}"));
        assert_eq!(twin.folded_tensors(), 1, "exactly `p` is point-folded");
        assert!(
            twin.supports_gradient(),
            "the twin must be inside the exact point-VJP fragment (first refusal: {:?})",
            twin.first_gradient_refusal()
        );
        let shape = twin.input_shape(&[1, 2]);
        assert_eq!(shape.iter().product::<usize>(), 2);

        let mut authored = crate::diff::OrtForward::from_model_bytes(&bytes, 2, false)
            .expect("the authored model runs under ONNX Runtime");
        let p = exact_p();
        for x in [[0.25_f32, -0.5], [-0.8, 0.6], [0.9, 0.1], [-0.3, -0.7]] {
            let point = ArrayD::from_shape_vec(IxDyn(&shape), x.to_vec()).expect("point");

            // Gradient: dV/dx = (P + Pᵀ) x.
            let row = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("spec row");
            let grad: Vec<f32> = twin
                .point_gradient(&point, &row, None)
                .expect("the twin's walk must not error")
                .expect("an admitted twin yields a gradient")
                .iter()
                .copied()
                .collect();
            for i in 0..2 {
                let reference: f64 = (0..2).map(|j| (p[i][j] + p[j][i]) * f64::from(x[j])).sum();
                let tol = 1e-4 * (1.0 + reference.abs());
                assert!(
                    (f64::from(grad[i]) - reference).abs() <= tol,
                    "coord {i} at {x:?}: twin gradient {} vs (P + Pᵀ)x {reference}",
                    grad[i]
                );
            }

            // Fidelity: the twin's forward is the authored network's ORT value.
            let twin_value = twin.point_forward(&point, None).expect("twin forward");
            let ort_value = authored.run(&x).expect("authored ORT forward");
            assert_eq!(twin_value.len(), 1);
            assert_eq!(ort_value.len(), 1);
            let tol = 1e-5 * (1.0 + f64::from(ort_value[0]).abs());
            assert!(
                (f64::from(twin_value[0]) - f64::from(ort_value[0])).abs() <= tol,
                "at {x:?}: twin forward {} vs authored ORT {}",
                twin_value[0],
                ort_value[0]
            );
        }
    }
}
