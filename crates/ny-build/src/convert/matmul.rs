// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{BilinearCrownLayer, IntervalConstLayer, LinearLayer};
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::interval_const::center_and_radius;
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

        if matches!(
            self.linear_storage,
            crate::LinearStoragePolicy::OriginalPoint(_)
        ) && (a_is_weight
            || b_is_weight
            || self.weights.get_interval(input_a).is_some()
            || self.weights.get_interval(input_b).is_some())
        {
            return Err(NyError::UnsupportedConfiguration(
                "original point storage: static/interval MatMul affine conversion needs a separately priced adapter; dynamic activation MatMul remains supported".into(),
            ));
        }

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

        // INTERVAL-VALUED CONSTANT OPERAND.
        //
        // The const-folder could not represent this constant exactly and stored
        // a rigorous enclosure instead. An enclosure is still a CONSTANT, so the
        // classification is the one it always had — `activation @ constant` is a
        // Linear, not a bilinear MatMul — and the rank-1 "MatMul requires at
        // least 2D inputs" raise that used to kill alpha-CROWN never fires.
        //
        // Only the `activation @ interval` shape is carried. Everything else
        // (interval on the left, both sides interval, or a `scale` attribute
        // whose exact-scaling helper has no interval form) REFUSES here and
        // degrades to the sound unbounded skip below.
        let a_is_interval = self.weights.get_interval(input_a).is_some();
        let b_is_interval = self.weights.get_interval(input_b).is_some();
        if b_is_interval && !a_is_interval && !a_is_weight {
            return self.convert_matmul_interval_weight(spec, input_b, transpose_b, scale);
        }
        if a_is_interval || b_is_interval {
            warn!(
                "MatMul {} has an interval-valued constant in a position the carrier does not \
                 support (a_interval={a_is_interval}, b_interval={b_is_interval}, \
                 a_weight={a_is_weight}, b_weight={b_is_weight}); failing closed to a sound skip",
                spec.name
            );
            return Err(NyError::UnsupportedOp(format!(
                "MatMul {} has an interval-valued constant operand in an unsupported position",
                spec.name
            )));
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
            // BilinearCrown uses CompactMcCormick for seq > 64, avoiding the
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
            // Both inputs are constant, and NEITHER the bit-exact fold nor the
            // interval carrier produced a value for this node.
            //
            // `Layer::MatMul` is the BOUNDED BINARY op — it requires two
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
            // WHAT THIS COMMENT USED TO SAY, AND WHY IT WAS WRONG. It claimed that
            // an UNSOUND point-fold of the whole cone "also yields exactly those
            // same 9 rows and zero unsat ... there is nothing further to win here
            // by carrying the residual". The MEASUREMENT is the opposite: an
            // offline bit-exact pre-fold (max |orig - folded| = 0.0 over 2000
            // points in the official box) makes beta-CROWN actually RUN — root gap
            // -29.86 -> -0.0008 in 7 s, depth 13, 5801 domains, all 9 sampled rows
            // moving `unknown` -> `TIMEOUT`. The mechanism the comment described
            // ("nothing to win") confused a bound that cannot be computed with a
            // bound that is already tight.
            //
            // WHAT IS ACTUALLY TRUE, AND IT MATTERS FOR SCOPE. The point fold is
            // an UPPER BOUND on what any sound carrier can achieve, because it has
            // zero width. That ceiling was measured at 9 sat / 0 unsat, with the
            // root gap flat at ~1e-4 from t=16 s to t=80 s while the domain queue
            // grew 10x to 40,526. So what the interval carrier buys on this family
            // is that the verifier RUNS AT ALL, and that a +-inf leaves a scored
            // path — not points. The residue behind it is a RELAXATION FLOOR, a
            // second wall, and it is not a carrier problem.
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

impl ConvertContext<'_> {
    /// Lower `activation @ interval_constant` to [`Layer::IntervalConst`].
    ///
    /// Mirrors the point `b_is_weight && !a_is_weight` branch above — the same
    /// 1-D promotion, the same 2-D requirement, the same transpose convention —
    /// applied to BOTH endpoints of the enclosure, then converted to the
    /// `center +- radius` form the propagation layer consumes.
    fn convert_matmul_interval_weight(
        &self,
        spec: &LayerSpec,
        input_b: &str,
        transpose_b: bool,
        scale: Option<f32>,
    ) -> Result<Layer> {
        if scale.is_some() {
            // `exact_scale_weights` publishes a scaled constant only when the
            // product is exactly representable; there is no interval form of
            // that certificate yet, and silently scaling the midpoint alone
            // would drop the enclosure.
            return Err(NyError::UnsupportedOp(format!(
                "MatMul {} scales an interval-valued constant; no exact interval scaling \
                 certificate exists",
                spec.name
            )));
        }
        let (lower, upper) = self.weights.get_interval(input_b).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "MatMul {} missing interval constant input '{}'",
                spec.name, input_b
            ))
        })?;
        let context = format!("MatMul {}", spec.name);
        let (center, radius) = center_and_radius(lower, upper, &context)?;

        let promote = |value: ndarray::ArrayD<f32>| -> Result<ndarray::Array2<f32>> {
            let value = if value.ndim() == 1 {
                let k = value.len();
                value
                    .into_shape_with_order(ndarray::IxDyn(&[k, 1]))
                    .map_err(|e| {
                        NyError::ModelLoad(format!(
                            "Cannot reshape 1D interval MatMul constant to 2D: {e}"
                        ))
                    })?
            } else {
                value
            };
            if value.ndim() != 2 {
                return Err(NyError::ModelLoad(format!(
                    "MatMul interval constant must be 2D, got {}D",
                    value.ndim()
                )));
            }
            let two_d = value.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                NyError::ModelLoad(format!("Cannot reshape MatMul interval constant: {e}"))
            })?;
            // Linear expects (out, in); MatMul without `transpose_b` supplies
            // (in, out). The transpose is a pure permutation — exact for the
            // center and for the radius alike.
            Ok(if transpose_b {
                two_d
            } else {
                two_d.t().to_owned()
            })
        };

        let center = promote(center)?;
        let radius = promote(radius)?;
        let layer = IntervalConstLayer::try_new_linear(LinearLayer::new(center, None)?, radius)?;
        debug!(
            "MatMul {} with an interval-valued constant second input converted to {} \
             (max radius {:e})",
            spec.name,
            layer.kind_name(),
            layer.max_radius()
        );
        Ok(Layer::IntervalConst(layer))
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

    /// Regression test: MatMul with extra inputs must be rejected.
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
