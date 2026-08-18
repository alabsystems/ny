// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{BilinearCrownLayer, LinearLayer};
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_matmul(&self, spec: &LayerSpec) -> Result<Layer> {
        // MatMul in ONNX: C = A @ B
        // If one input is a weight (constant), treat it as a Linear layer
        // Otherwise, treat it as a bounded MatMul

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "MatMul {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Check if either input is a constant weight
        let a_is_weight = self.weights.get(input_a).is_some();
        let b_is_weight = self.weights.get(input_b).is_some();

        let transpose_b = match spec.attributes.get("transpose_b") {
            Some(AttributeValue::Int(v)) => *v != 0,
            Some(AttributeValue::Float(v)) => *v != 0.0,
            Some(_) => {
                return Err(NyError::ModelLoad(format!(
                    "MatMul {} has invalid transpose_b attribute type",
                    spec.name
                )));
            }
            None => false,
        };

        let scale = match spec.attributes.get("scale") {
            Some(AttributeValue::Float(v)) => Some(*v),
            Some(AttributeValue::Int(v)) => Some(i64_to_f32_checked(
                *v,
                &format!("MatMul {} scale attribute", spec.name),
            )?),
            Some(_) => {
                return Err(NyError::ModelLoad(format!(
                    "MatMul {} has invalid scale attribute type",
                    spec.name
                )));
            }
            None => None,
        };
        if let Some(scale) = scale {
            if !scale.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "MatMul {} scale must be finite, got {}",
                    spec.name, scale
                )));
            }
        }

        if b_is_weight && !a_is_weight {
            // Standard case: A @ W where W is a constant weight
            // This is equivalent to a Linear layer (without bias)
            let weight = self.weights.get(input_b).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "MatMul {} missing constant weight input '{}'",
                    spec.name, input_b
                ))
            })?;

            // ONNX MatMul spec: if B is 1D (K,), promote to (K, 1), compute matmul,
            // then remove the appended dimension. The builder inserts a Squeeze(-1) node
            // after this Linear layer to handle the dimension removal.
            let weight = if weight.ndim() == 1 {
                let k = weight.len();
                debug!(
                    "MatMul {} has 1D weight B ({}), promoting to ({}, 1) per ONNX spec",
                    spec.name, k, k
                );
                weight
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[k, 1]))
                    .map_err(|e| {
                        NyError::ModelLoad(format!("Cannot reshape 1D MatMul weight to 2D: {}", e))
                    })?
            } else {
                weight.clone()
            };

            // MatMul semantics: input shape (*, K), weight shape (K, N), output (*, N)
            // For Linear layer, we need weight shape (N, K) so we transpose
            if weight.ndim() != 2 {
                return Err(NyError::ModelLoad(format!(
                    "MatMul weight must be 2D, got {}D",
                    weight.ndim()
                )));
            }

            let weight_2d = weight
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| NyError::ModelLoad(format!("Cannot reshape MatMul weight: {}", e)))?;

            // MatMul supports an optional `transpose_b` attribute in our fused graphs.
            //
            // - If `transpose_b=false`: MatMul is A @ W, with W expected to be (K, N).
            //   Linear expects (N, K), so we transpose here.
            // - If `transpose_b=true`: MatMul is A @ W^T, with W expected to be (N, K).
            //   Linear expects (N, K), so we use W as-is.
            let mut linear_weight = if transpose_b {
                weight_2d
            } else {
                weight_2d.t().to_owned()
            };
            if let Some(s) = scale {
                linear_weight = exact_scale_weights(linear_weight, s, &spec.name)?;
            }
            debug!(
                "MatMul {} with constant second input converted to Linear layer",
                spec.name
            );
            Ok(Layer::Linear(LinearLayer::new(linear_weight, None)?))
        } else if a_is_weight && !b_is_weight {
            // Less common: W @ B where W is constant
            // This is also equivalent to a Linear layer, but weight is already in correct format
            // MatMul semantics: W shape (M, K), B shape (K,), output (M,)
            // For Linear layer, weight should be (out_features, in_features) = (M, K) - no transpose needed
            let weight = self.weights.get(input_a).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "MatMul {} missing constant weight input '{}'",
                    spec.name, input_a
                ))
            })?;

            let activation_shape = self.tensor_shapes.get(input_b).ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "MatMul {} with constant first input requires a known rank-1 second input",
                    spec.name
                ))
            })?;
            // `Layer::Linear` contracts the LAST axis of its input, so the runtime
            // operand must be a VECTOR in ny's INTERNAL (unbatched) convention.
            // `activation_shape` is the recorded ONNX shape, which is not the
            // internal shape for a batched model, so test the internal rank:
            //
            //  * ONNX rank-1 `[K]`  -> internal `[K]` under BOTH conventions
            //    (`internal_shape_from_onnx_shape` keeps rank <= 1 verbatim).
            //  * ONNX rank-2 `[K, 1]` in a BATCHED model -> internal rank 1, so the
            //    tensor arriving at the Linear is `[K]`. The exporter's column
            //    vector `W(M,K) @ B(K,1)` then satisfies the exact identity
            //      W(M,K) @ B(K,1) == reshape(W(M,K) @ squeeze(B), (M,1))
            //    and the plain Linear computes it with no inserted nodes.
            //
            // Everything else must keep failing closed: a genuine `[K, N]` with
            // N > 1 is NOT a last-axis contraction (Linear would silently compute
            // `B @ W^T` whenever N == K), rank >= 3 likewise, and `[K, 1]` in an
            // UNBATCHED model maps verbatim to an internal rank-2 tensor.
            let onnx_vector = activation_shape.len() == 1;
            let internal_vector_column =
                !self.model_unbatched && activation_shape.len() == 2 && activation_shape[1] == 1;
            if !(onnx_vector || internal_vector_column) || activation_shape[0] <= 0 {
                return Err(NyError::UnsupportedOp(format!(
                    "MatMul {} cannot lower constant-left W @ B with runtime B shape {:?} to a last-axis Linear; only an internally rank-1 vector is equivalent",
                    spec.name, activation_shape
                )));
            }
            // `transpose_b` is NOT applied by this branch (the weight is used as-is
            // at the `linear_weight` binding below). That is a no-op only for a
            // genuine ONNX rank-1 operand, where B^T == B. Refuse the column-vector
            // spelling rather than silently drop the transpose.
            if transpose_b && !onnx_vector {
                return Err(NyError::UnsupportedOp(format!(
                    "MatMul {} cannot lower constant-left W @ B^T with runtime B shape {:?}; transpose_b is not applied on this path",
                    spec.name, activation_shape
                )));
            }
            let contract_len = activation_shape[0] as usize;

            // ONNX MatMul spec: if A is 1D (K,), promote to (1, K), compute matmul,
            // then remove the prepended dimension. The builder inserts a Squeeze node
            // after this Linear layer to handle the dimension removal.
            let weight = if weight.ndim() == 1 {
                let k = weight.len();
                debug!(
                    "MatMul {} has 1D weight A ({}), promoting to (1, {}) per ONNX spec",
                    spec.name, k, k
                );
                weight
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[1, k]))
                    .map_err(|e| {
                        NyError::ModelLoad(format!("Cannot reshape 1D MatMul weight to 2D: {}", e))
                    })?
            } else {
                weight.clone()
            };

            if weight.ndim() != 2 {
                return Err(NyError::ModelLoad(format!(
                    "MatMul weight must be 2D, got {}D",
                    weight.ndim()
                )));
            }

            let weight_2d = weight
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| NyError::ModelLoad(format!("Cannot reshape MatMul weight: {}", e)))?;

            // Fail CLOSED at build time on a K mismatch. Letting this reach
            // `Linear::propagate_ibp` would raise ShapeMismatch at an UNTAINTED
            // node, which aborts the whole IBP pass (graph_ibp.rs:590); an
            // UnsupportedOp here degrades to an OpaqueSkip instead.
            if weight_2d.ncols() != contract_len {
                return Err(NyError::UnsupportedOp(format!(
                    "MatMul {} constant-left weight has in_features {} but runtime B contributes {} (ONNX shape {:?})",
                    spec.name,
                    weight_2d.ncols(),
                    contract_len,
                    activation_shape
                )));
            }

            // No transpose needed: W @ x expects W in (out, in) format which matches Linear
            let mut linear_weight = weight_2d;
            if let Some(s) = scale {
                linear_weight = exact_scale_weights(linear_weight, s, &spec.name)?;
            }
            debug!(
                "MatMul {} with constant first input converted to Linear layer",
                spec.name
            );
            Ok(Layer::Linear(LinearLayer::new(linear_weight, None)?))
        } else if !a_is_weight && !b_is_weight {
            // Neither input is a weight - true bounded MatMul (e.g., Q @ K^T or probs @ V).
            // Use BilinearCrownLayer for all bilinear MatMuls regardless of transpose_b.
            // BilinearCrown uses CompactMcCormick for seq > 64 (#286), avoiding the
            // O(seq^4) dense identity that blocks the MatMul retry path in crown_batched.rs.
            // CompactMcCormick supports both transpose_b configurations (verified by proptests).
            debug!(
                "MatMul {} is a bounded binary operation (both inputs are activations), using BilinearCrownLayer",
                spec.name
            );
            Ok(Layer::BilinearCrown(
                BilinearCrownLayer::try_new(transpose_b, scale).map_err(|err| {
                    NyError::InvalidSpec(format!("MatMul {} scale invalid: {err}", spec.name))
                })?,
            ))
        } else {
            // Both inputs are constant. The const-folder deliberately declines to
            // fold this when the dot product rounds: `try_fold_matmul_exact`
            // (const_fold/ops/elementwise.rs:174, comment at :171-173) keeps it a
            // graph operation rather than exactifying a rounded center.
            //
            // But `Layer::MatMul` is the BOUNDED BINARY op — it requires two
            // ACTIVATION inputs and here it has none, so graph construction hard
            // errors ("expects 2 activation inputs for binary op, got 0") and the
            // whole model fails to load. On lsnc_relu that is one model backing
            // 80/80 rows: a guaranteed 0 for the entire benchmark.
            //
            // Fail CLOSED instead. `UnsupportedOp` degrades this node to a sound
            // OpaqueSkip (unbounded, taints its consumers) rather than aborting the
            // load, so every other node still lowers and the falsification lanes
            // can still run. Sound: the skip only ever widens, no verdict can be
            // derived from a tainted output, and a `sat` still has to clear the
            // ONNX-Runtime oracle gate.
            //
            // MEASURED on lsnc_relu (80 rows, scored path): 0 solved before, 9 sat
            // after. A deliberately UNSOUND upper bound — folding the whole cone to
            // rounded f32 point constants, i.e. zero interval width and therefore
            // strictly tighter than any sound carrier could be — also yields
            // exactly those same 9 rows and zero unsat. So the interval-valued
            // frozen-constant program is bounded above by this on this benchmark;
            // there is nothing further to win here by carrying the residual.
            warn!(
                "MatMul {} has both constant inputs and did not constant-fold (the exact-dot \
                 gate declined a rounded product); failing closed to a sound skip rather than \
                 emitting a binary MatMul layer with no activation inputs",
                spec.name
            );
            Err(NyError::UnsupportedOp(format!(
                "MatMul {} has two constant inputs that did not fold; ny has no exact carrier \
                 for a rounded constant dot product, so the node cannot be lowered",
                spec.name
            )))
        }
    }
}

fn exact_scale_weights(
    weight: ndarray::Array2<f32>,
    scale: f32,
    node_name: &str,
) -> Result<ndarray::Array2<f32>> {
    let shape = weight.raw_dim();
    let values = weight
        .iter()
        .map(|&value| {
            let exact = (value as f64) * (scale as f64);
            let rounded = value * scale;
            (rounded.is_finite() && rounded as f64 == exact).then_some(rounded)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            NyError::UnsupportedConfiguration(format!(
                "MatMul {node_name} scale would round a constant weight product"
            ))
        })?;
    ndarray::Array2::from_shape_vec(shape, values).map_err(|error| {
        NyError::InvalidSpec(format!(
            "MatMul {node_name} could not materialize exactly scaled weights: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    /// Helper: constant-left `W @ B` spec with weight shape (M, K).
    fn constant_left_ctx_spec(
        m: usize,
        k: usize,
        b_shape: Vec<i64>,
    ) -> (
        WeightStore,
        HashMap<String, Vec<i64>>,
        HashSet<String>,
        LayerSpec,
    ) {
        let mut weights = WeightStore::new();
        let values: Vec<f32> = (0..(m * k)).map(|i| i as f32 + 1.0).collect();
        weights.insert(
            "w".to_string(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[m, k]), values).unwrap(),
        );
        let shapes = HashMap::from([("b".to_string(), b_shape)]);
        let spec = LayerSpec {
            name: "constant_left".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["w".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        (weights, shapes, HashSet::new(), spec)
    }

    /// #ml4acopf: a BATCHED model's `[K, 1]` column vector is internally rank-1
    /// `[K]`, so `W(M,K) @ B(K,1)` lowers to a plain last-axis Linear. This is
    /// the whole ml4acopf_2024 benchmark (72 sites across 9 nets).
    #[test]
    fn constant_left_matmul_accepts_batched_column_vector() {
        let (weights, shapes, constants, spec) = constant_left_ctx_spec(14, 11, vec![11, 1]);
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let layer = context
            .convert_matmul(&spec)
            .expect("column vector must lower");
        match layer {
            Layer::Linear(linear) => {
                // Weight is used as-is: (out_features, in_features) == (M, K).
                assert_eq!(linear.out_features(), 14);
                assert_eq!(linear.in_features(), 11);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    /// The same ONNX shape in an UNBATCHED model maps verbatim to an internal
    /// rank-2 tensor, which is NOT a last-axis contraction. Must stay refused.
    #[test]
    fn constant_left_matmul_rejects_column_vector_when_model_unbatched() {
        let (weights, shapes, constants, spec) = constant_left_ctx_spec(14, 11, vec![11, 1]);
        let context = ConvertContext::new(&weights, &shapes, &constants).with_model_unbatched(true);
        let error = context.convert_matmul(&spec).unwrap_err();
        assert!(
            matches!(&error, NyError::UnsupportedOp(m) if m.contains("internally rank-1")),
            "got {error:?}"
        );
    }

    /// THE SOUNDNESS BOUNDARY: a genuine `[K, N]` with N > 1 is NOT covered by
    /// the squeeze identity. Critically, when N == K a last-axis Linear would
    /// silently compute `B @ W^T` (right shape, wrong values), so this must
    /// keep failing closed even though nothing downstream would flag it.
    #[test]
    fn constant_left_matmul_rejects_true_matrix_runtime_operand() {
        for b_shape in [vec![11, 2], vec![11, 11], vec![11, 7]] {
            let (weights, shapes, constants, spec) =
                constant_left_ctx_spec(14, 11, b_shape.clone());
            let context = ConvertContext::new(&weights, &shapes, &constants);
            let error = context.convert_matmul(&spec).unwrap_err();
            assert!(
                matches!(&error, NyError::UnsupportedOp(m) if m.contains("internally rank-1")),
                "shape {b_shape:?} must stay refused, got {error:?}"
            );
        }
    }

    /// Rank >= 3 stays refused regardless of a trailing singleton.
    #[test]
    fn constant_left_matmul_rejects_rank3_runtime_operand() {
        for b_shape in [vec![1, 11, 1], vec![2, 11, 1]] {
            let (weights, shapes, constants, spec) =
                constant_left_ctx_spec(14, 11, b_shape.clone());
            let context = ConvertContext::new(&weights, &shapes, &constants);
            let error = context.convert_matmul(&spec).unwrap_err();
            assert!(
                matches!(&error, NyError::UnsupportedOp(m) if m.contains("internally rank-1")),
                "shape {b_shape:?} must stay refused, got {error:?}"
            );
        }
    }

    /// A K mismatch must fail CLOSED at build time (-> OpaqueSkip) rather than
    /// reach `Linear::propagate_ibp`, where a ShapeMismatch at an untainted node
    /// aborts the entire IBP pass.
    #[test]
    fn constant_left_matmul_rejects_column_vector_with_wrong_k() {
        let (weights, shapes, constants, spec) = constant_left_ctx_spec(14, 11, vec![9, 1]);
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let error = context.convert_matmul(&spec).unwrap_err();
        assert!(
            matches!(&error, NyError::UnsupportedOp(m) if m.contains("in_features")),
            "got {error:?}"
        );
    }

    /// `transpose_b` is not applied on the constant-left path; refuse the
    /// column-vector spelling rather than silently drop it.
    #[test]
    fn constant_left_matmul_rejects_column_vector_with_transpose_b() {
        let (weights, shapes, constants, mut spec) = constant_left_ctx_spec(14, 11, vec![11, 1]);
        spec.attributes
            .insert("transpose_b".to_string(), AttributeValue::Int(1));
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let error = context.convert_matmul(&spec).unwrap_err();
        assert!(
            matches!(&error, NyError::UnsupportedOp(m) if m.contains("transpose_b")),
            "got {error:?}"
        );
    }

    /// Two constant inputs that the exact-dot gate declined must FAIL CLOSED, not
    /// emit a bounded-binary `Layer::MatMul` with zero activation inputs.
    ///
    /// The old behaviour hard-errored during graph construction ("expects 2
    /// activation inputs for binary op, got 0"), which failed the entire model
    /// load — 80/80 rows of lsnc_relu, a guaranteed 0 for the whole benchmark.
    /// An `UnsupportedOp` here degrades the single node to a sound OpaqueSkip and
    /// lets the rest of the model load.
    #[test]
    fn both_constant_matmul_fails_closed_instead_of_emitting_a_binary_layer() {
        let mut weights = WeightStore::new();
        for (name, rows, cols) in [("a", 2usize, 3usize), ("b", 3usize, 2usize)] {
            let values: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 + 0.5).collect();
            weights.insert(
                name.to_string(),
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[rows, cols]), values).unwrap(),
            );
        }
        let spec = LayerSpec {
            name: "both_const".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        // Bind the empty maps: `ConvertContext` borrows them, so inline
        // temporaries would be dropped at the end of the `let` statement.
        let producers = HashMap::new();
        let constants = HashSet::new();
        let ctx = ConvertContext::new(&weights, &producers, &constants);
        let error = ctx
            .convert_matmul(&spec)
            .expect_err("two constant inputs must fail closed, not lower to a binary MatMul");
        assert!(
            matches!(&error, NyError::UnsupportedOp(message) if message.contains("two constant inputs")),
            "{error}"
        );
    }

    #[test]
    fn constant_left_matmul_rejects_matrix_runtime_operand() {
        let mut weights = WeightStore::new();
        weights.insert(
            "w".to_string(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
                .unwrap(),
        );
        let shapes = HashMap::from([("b".to_string(), vec![1, 2, 2])]);
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "constant_left_matrix".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["w".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        let error = context.convert_matmul(&spec).unwrap_err();
        assert!(matches!(error, NyError::UnsupportedOp(message) if message.contains("rank-1")));
    }

    #[test]
    fn constant_matmul_rejects_inexact_weight_scaling() {
        let mut weights = WeightStore::new();
        weights.insert(
            "w".to_string(),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![0.1]).unwrap(),
        );
        let shapes = HashMap::from([("x".to_string(), vec![1, 1])]);
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "inexact_scale".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Float(0.1))]),
        };
        assert!(matches!(
            context.convert_matmul(&spec),
            Err(NyError::UnsupportedConfiguration(message)) if message.contains("round")
        ));
    }

    /// Regression test for #2666: MatMul with extra inputs must be rejected.
    #[test]
    fn convert_matmul_rejects_extra_inputs_2666() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_op".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("MatMul with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_matmul_rejects_precision_losing_scale_attr_4149() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_lossy_scale".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Int(16_777_217))]),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("precision-losing MatMul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg)
                    if msg.contains("precision loss")
                        && msg.contains("MatMul matmul_lossy_scale scale attribute")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_matmul_rejects_non_finite_scale_attr_4307() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_nan_scale".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Float(f32::NAN))]),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("non-finite MatMul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("MatMul matmul_nan_scale scale must be finite")
            ),
            "unexpected error: {error:?}"
        );
    }
}
