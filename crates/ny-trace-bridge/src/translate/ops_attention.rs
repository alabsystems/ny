// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention family: `Sdpa`, `SdpaCausal`, `RotaryEmbedding`.
//!
//! Ported from NN's `trace_to_graph_layerspec_attention.rs` (the trace-path
//! ground truth; the dispatch arms live in
//! `trace_to_graph_layerspec_dispatch_extended.rs`). (`MultiHeadAttention` is
//! NOT in scope: it stays in the dispatch's refused-forever arm.)
//!
//! ## SDPA decomposition
//!
//! `Sdpa { scale }` with inputs Q, K, V, [mask] decomposes to:
//! 1. `MatMul(Q, K^T)` with `transpose_b=1`, `scale`
//! 2. `Add(scores, mask)` (if mask present)
//! 3. `Softmax(axis=last_dim)` → attention weights
//! 4. `MatMul(attn_weights, V)` → output
//!
//! Both MatMul ops have two activation inputs (Q/K and attn/V are all
//! variable), so graph-build produces a bilinear CROWN layer with McCormick
//! relaxation.
//!
//! ## SdpaCausal decomposition
//!
//! Same as SDPA but uses `CausalSoftmax` instead of `Softmax+mask`, avoiding
//! the O(S²) mask allocation.
//!
//! ## RotaryEmbedding
//!
//! Maps to `LayerType::RoPE` with cos/sin frequency weights extracted from
//! the `TraceOp::RotaryEmbedding` variant's `cos_cache` / `sin_cache` payloads
//! (`head_dim` / `offset` are not consumed here, mirroring NN's dispatch —
//! the caches are already narrowed to the traced positions).

use std::collections::HashMap;

use ny_build::AttributeValue;
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp, WeightPayload};

use super::{
    checked_f64_to_f32, dim_as_i64, insert_payload, op_name, simple_spec, Ctx, NodeOutput,
};

/// Translate an attention-family op (`Sdpa`, `SdpaCausal`, `RotaryEmbedding`) node.
///
/// Any other op routed here refuses with the same fail-closed
/// [`NyError::UnsupportedOp`] error the pre-split catch-all arm produced.
pub(super) fn translate_attention(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    match &node.op {
        TraceOp::Sdpa { scale } => translate_sdpa(
            name,
            *scale,
            input_tensors,
            output_tensor,
            &node.output_shape,
            ctx,
        ),
        TraceOp::SdpaCausal { scale } => translate_sdpa_causal(
            name,
            *scale,
            input_tensors,
            output_tensor,
            &node.output_shape,
            ctx,
        ),
        TraceOp::RotaryEmbedding {
            cos_cache,
            sin_cache,
            ..
        } => translate_rope(
            name,
            cos_cache,
            sin_cache,
            input_tensors,
            output_tensor,
            ctx,
        ),
        other => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// SDPA
// ---------------------------------------------------------------------------

/// Decompose `TraceOp::Sdpa { scale }` into MatMul + [Add] + Softmax + MatMul.
///
/// Inputs: Q (idx 0), K (idx 1), V (idx 2), optional mask (idx 3).
/// Output shape: `[B, H, S_q, head_dim]`.
fn translate_sdpa(
    name: &str,
    scale: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "Sdpa requires at least 3 inputs (Q, K, V), got {}",
            input_tensors.len()
        )));
    }
    if output_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "Sdpa: expected 4D output shape, got {output_shape:?}"
        )));
    }
    let has_mask = input_tensors.len() >= 4;
    let scale_f32 = checked_f64_to_f32(scale, "Sdpa scale")?;

    let q_tensor = &input_tensors[0];
    let k_tensor = &input_tensors[1];
    let v_tensor = &input_tensors[2];

    // Derive intermediate shapes.
    // output = [B, H, S_q, D]. K = [B, H, S_kv, D]. scores = [B, H, S_q, S_kv].
    let b = output_shape[0];
    let h = output_shape[1];
    let s_q = output_shape[2];

    let k_shape = ctx.tensor_shapes.get(k_tensor).ok_or_else(|| {
        NyError::InternalError("Sdpa: K tensor shape not found in context".to_string())
    })?;
    if k_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "Sdpa: expected 4D K shape, got {k_shape:?}"
        )));
    }
    let s_kv_i64 = k_shape[2];

    let b_i64 = dim_as_i64(b, "Sdpa batch")?;
    let h_i64 = dim_as_i64(h, "Sdpa heads")?;
    let sq_i64 = dim_as_i64(s_q, "Sdpa S_q")?;
    let scores_shape = vec![b_i64, h_i64, sq_i64, s_kv_i64];

    sdpa_decompose(
        name,
        scale_f32,
        q_tensor,
        k_tensor,
        v_tensor,
        has_mask.then(|| &input_tensors[3]),
        &scores_shape,
        output_tensor,
        false, // not causal
        ctx,
    )
}

// ---------------------------------------------------------------------------
// SdpaCausal
// ---------------------------------------------------------------------------

/// Decompose `TraceOp::SdpaCausal { scale }` into MatMul + CausalSoftmax + MatMul.
///
/// Inputs: Q (idx 0), K (idx 1), V (idx 2). No mask input.
/// Output shape: `[B, H, S, head_dim]`.
fn translate_sdpa_causal(
    name: &str,
    scale: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "SdpaCausal requires 3 inputs (Q, K, V), got {}",
            input_tensors.len()
        )));
    }
    if output_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "SdpaCausal: expected 4D output shape, got {output_shape:?}"
        )));
    }
    let scale_f32 = checked_f64_to_f32(scale, "SdpaCausal scale")?;

    let q_tensor = &input_tensors[0];
    let k_tensor = &input_tensors[1];
    let v_tensor = &input_tensors[2];

    let b = output_shape[0];
    let h = output_shape[1];
    let s_q = output_shape[2];

    let k_shape = ctx.tensor_shapes.get(k_tensor).ok_or_else(|| {
        NyError::InternalError("SdpaCausal: K tensor shape not found in context".to_string())
    })?;
    if k_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "SdpaCausal: expected 4D K shape, got {k_shape:?}"
        )));
    }
    let s_kv_i64 = k_shape[2];

    let b_i64 = dim_as_i64(b, "SdpaCausal batch")?;
    let h_i64 = dim_as_i64(h, "SdpaCausal heads")?;
    let sq_i64 = dim_as_i64(s_q, "SdpaCausal S_q")?;
    let scores_shape = vec![b_i64, h_i64, sq_i64, s_kv_i64];

    sdpa_decompose(
        name,
        scale_f32,
        q_tensor,
        k_tensor,
        v_tensor,
        None, // no mask
        &scores_shape,
        output_tensor,
        true, // causal
        ctx,
    )
}

// ---------------------------------------------------------------------------
// Shared SDPA decomposition
// ---------------------------------------------------------------------------

/// Shared decomposition for both `Sdpa` and `SdpaCausal`.
///
/// Emits 3-4 LayerSpecs:
/// 1. MatMul(Q, K^T) with transpose_b + scale
/// 2. Optional Add(scores, mask)
/// 3. Softmax or CausalSoftmax
/// 4. MatMul(attn_weights, V)
#[allow(clippy::too_many_arguments)]
fn sdpa_decompose(
    name: &str,
    scale_f32: f32,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    mask_tensor: Option<&String>,
    scores_shape: &[i64],
    output_tensor: &str,
    causal: bool,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let mut specs = Vec::new();

    // Step 1: MatMul(Q, K^T) with transpose_b=1 and scale.
    // graph-build's matmul converter folds scale into the bilinear CROWN layer.
    let mm1_name = format!("{name}_sdpa_qk");
    let mm1_out = format!("{mm1_name}_out");
    let mut mm1_attrs = HashMap::new();
    mm1_attrs.insert("transpose_b".to_string(), AttributeValue::Int(1));
    mm1_attrs.insert("scale".to_string(), AttributeValue::Float(scale_f32));
    ctx.tensor_shapes
        .insert(mm1_out.clone(), scores_shape.to_vec());
    specs.push(simple_spec(
        &mm1_name,
        LayerType::MatMul,
        vec![q_tensor.to_string(), k_tensor.to_string()],
        &mm1_out,
        mm1_attrs,
    ));

    // Step 2: Optional mask addition (only for non-causal Sdpa with mask).
    let softmax_input = if let Some(mask) = mask_tensor {
        let add_name = format!("{name}_sdpa_mask");
        let add_out = format!("{add_name}_out");
        ctx.tensor_shapes
            .insert(add_out.clone(), scores_shape.to_vec());
        specs.push(simple_spec(
            &add_name,
            LayerType::Add,
            vec![mm1_out, mask.clone()],
            &add_out,
            HashMap::new(),
        ));
        add_out
    } else {
        mm1_out
    };

    // Step 3: Softmax or CausalSoftmax on the last dimension.
    // Softmax axis: trace dim 3 (last of 4D [B,H,S_q,S_kv]). The translator
    // passes the trace dim directly (no +1 ONNX adjustment) because
    // graph-build's convert_elementwise does NOT subtract 1 for softmax axes
    // (unlike convert_reductions). Mirrors NN's translate_softmax convention.
    let sm_name = format!("{name}_sdpa_sm");
    let sm_out = format!("{sm_name}_out");
    let mut sm_attrs = HashMap::new();
    // Dimension 3 = last dimension of 4D scores tensor.
    sm_attrs.insert("axis".to_string(), AttributeValue::Int(3));
    ctx.tensor_shapes
        .insert(sm_out.clone(), scores_shape.to_vec());
    let softmax_type = if causal {
        LayerType::CausalSoftmax
    } else {
        LayerType::Softmax
    };
    specs.push(simple_spec(
        &sm_name,
        softmax_type,
        vec![softmax_input],
        &sm_out,
        sm_attrs,
    ));

    // Step 4: MatMul(attn_weights, V) → output.
    // Final step uses the parent name so output = "{name}_out".
    specs.push(simple_spec(
        name,
        LayerType::MatMul,
        vec![sm_out, v_tensor.to_string()],
        output_tensor,
        HashMap::new(),
    ));

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// RotaryEmbedding
// ---------------------------------------------------------------------------

/// Translate `TraceOp::RotaryEmbedding` to `LayerType::RoPE`.
///
/// NY's RoPE layer expects 3 inputs:
/// - `inputs[0]`: activation tensor (shape `[..., head_dim]`)
/// - `inputs[1]`: cos_freqs weight (shape `[head_dim/2]`)
/// - `inputs[2]`: sin_freqs weight (shape `[head_dim/2]`)
///
/// The cos/sin data is extracted from the TraceOp's `cos_cache` and `sin_cache`
/// payload fields.
fn translate_rope(
    name: &str,
    cos_cache: &WeightPayload,
    sin_cache: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.is_empty() {
        return Err(NyError::UnsupportedOp(
            "RotaryEmbedding has no inputs".to_string(),
        ));
    }

    let cos_name = format!("{name}_cos");
    let sin_name = format!("{name}_sin");
    insert_payload(ctx, cos_cache, &cos_name, "RotaryEmbedding cos_cache")?;
    insert_payload(ctx, sin_cache, &sin_name, "RotaryEmbedding sin_cache")?;

    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::RoPE,
        vec![input_tensors[0].clone(), cos_name, sin_name],
        output_tensor,
        HashMap::new(),
    )))
}

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel};
    use ny_core::{LayerType, NyError};

    use super::super::translate;
    use crate::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn find<'a>(model: &'a GraphModel, name: &str) -> &'a ny_build::LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer '{name}' present"))
    }

    fn build_ok(model: &GraphModel, what: &str) {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what} GraphModel builds a graph network: {e:?}"));
    }

    /// Sdpa (no mask) decomposes into MatMul(QKᵀ) → Softmax → MatMul(·V) with
    /// NN's exact attribute emission (transpose_b=1, scale, axis=3).
    #[test]
    fn sdpa_decomposes_matmul_softmax_matmul() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 3, 4]),
            node(
                1,
                "attn",
                TraceOp::Sdpa { scale: 0.125 },
                &[0, 0, 0],
                &[1, 2, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("sdpa translates");

        assert_eq!(count(&model, &LayerType::MatMul), 2, "QKᵀ and attn·V");
        assert_eq!(count(&model, &LayerType::Softmax), 1, "one Softmax");
        assert_eq!(count(&model, &LayerType::CausalSoftmax), 0, "not causal");
        // Only the input-identity Add — no mask Add.
        assert_eq!(count(&model, &LayerType::Add), 1, "no mask Add");

        let qk = find(&model, "layer0_trace_1_sdpa_qk");
        assert_eq!(qk.layer_type, LayerType::MatMul);
        assert_eq!(
            qk.attributes.get("transpose_b"),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(
            qk.attributes.get("scale"),
            Some(&AttributeValue::Float(0.125))
        );
        assert_eq!(
            qk.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_0_out".to_string()
            ],
            "QKᵀ consumes Q and K"
        );

        let sm = find(&model, "layer0_trace_1_sdpa_sm");
        assert_eq!(sm.layer_type, LayerType::Softmax);
        assert_eq!(sm.attributes.get("axis"), Some(&AttributeValue::Int(3)));
        assert_eq!(sm.inputs, vec!["layer0_trace_1_sdpa_qk_out".to_string()]);

        // Final MatMul uses the parent name so output = "{name}_out".
        let out = find(&model, "layer0_trace_1");
        assert_eq!(out.layer_type, LayerType::MatMul);
        assert_eq!(
            out.inputs,
            vec![
                "layer0_trace_1_sdpa_sm_out".to_string(),
                "layer0_trace_0_out".to_string()
            ],
            "attn·V consumes softmax output and V"
        );
        assert_eq!(out.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert!(out.attributes.is_empty(), "no attrs on attn·V MatMul");

        build_ok(&model, "sdpa");
    }

    /// Sdpa with a 4th (mask) input inserts Add(scores, mask) before Softmax.
    #[test]
    fn sdpa_with_mask_adds_mask_before_softmax() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 3, 4]),
            node(
                1,
                "mask",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.0; 2 * 3 * 3], vec![1, 2, 3, 3]),
                },
                &[],
                &[1, 2, 3, 3],
            ),
            node(
                2,
                "attn",
                TraceOp::Sdpa { scale: 0.5 },
                &[0, 0, 0, 1],
                &[1, 2, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("masked sdpa translates");

        // Input-identity Add + mask Add.
        assert_eq!(count(&model, &LayerType::Add), 2, "mask Add present");

        let mask_add = find(&model, "layer0_trace_2_sdpa_mask");
        assert_eq!(mask_add.layer_type, LayerType::Add);
        assert_eq!(
            mask_add.inputs,
            vec![
                "layer0_trace_2_sdpa_qk_out".to_string(),
                "layer0_trace_1_out".to_string()
            ],
            "mask Add consumes scores and the mask tensor"
        );

        let sm = find(&model, "layer0_trace_2_sdpa_sm");
        assert_eq!(sm.layer_type, LayerType::Softmax);
        assert_eq!(
            sm.inputs,
            vec!["layer0_trace_2_sdpa_mask_out".to_string()],
            "softmax consumes the masked scores"
        );

        build_ok(&model, "masked sdpa");
    }

    /// SdpaCausal uses CausalSoftmax (no mask Add) with axis=3.
    #[test]
    fn sdpa_causal_uses_causal_softmax() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 3, 4]),
            node(
                1,
                "attn",
                TraceOp::SdpaCausal { scale: 0.125 },
                &[0, 0, 0],
                &[1, 2, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("causal sdpa translates");

        assert_eq!(count(&model, &LayerType::MatMul), 2, "QKᵀ and attn·V");
        assert_eq!(
            count(&model, &LayerType::CausalSoftmax),
            1,
            "causal softmax"
        );
        assert_eq!(count(&model, &LayerType::Softmax), 0, "no plain Softmax");
        assert_eq!(count(&model, &LayerType::Add), 1, "no mask Add");

        let qk = find(&model, "layer0_trace_1_sdpa_qk");
        assert_eq!(
            qk.attributes.get("transpose_b"),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(
            qk.attributes.get("scale"),
            Some(&AttributeValue::Float(0.125))
        );

        let sm = find(&model, "layer0_trace_1_sdpa_sm");
        assert_eq!(sm.layer_type, LayerType::CausalSoftmax);
        assert_eq!(sm.attributes.get("axis"), Some(&AttributeValue::Int(3)));

        build_ok(&model, "causal sdpa");
    }

    /// RotaryEmbedding maps to a RoPE layer with cos/sin weight inputs.
    #[test]
    fn rope_emits_rope_layer_with_cos_sin_weights() {
        // 2 pairs (head_dim=4): unit rotations for angles [0, π/4] so the
        // graph-build RoPE invariant (cos² + sin² ≈ 1) holds.
        let angle0 = 0.0_f32;
        let angle1 = std::f32::consts::FRAC_PI_4;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "rope",
                TraceOp::RotaryEmbedding {
                    head_dim: 4,
                    offset: 0,
                    cos_cache: WeightPayload::f32(vec![angle0.cos(), angle1.cos()], vec![2]),
                    sin_cache: WeightPayload::f32(vec![angle0.sin(), angle1.sin()], vec![2]),
                },
                &[0],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("rope translates");

        assert_eq!(count(&model, &LayerType::RoPE), 1, "one RoPE layer");
        let rope = find(&model, "layer0_trace_1");
        assert_eq!(rope.layer_type, LayerType::RoPE);
        assert_eq!(
            rope.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_cos".to_string(),
                "layer0_trace_1_sin".to_string()
            ],
            "RoPE consumes activation + cos + sin"
        );
        assert_eq!(rope.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert!(rope.attributes.is_empty(), "RoPE carries no attributes");
        assert!(model.weights.contains_key("layer0_trace_1_cos"));
        assert!(model.weights.contains_key("layer0_trace_1_sin"));

        build_ok(&model, "rope");
    }

    /// Sdpa with a non-4D output shape stays refused (fail closed).
    #[test]
    fn sdpa_non_4d_output_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3, 4]),
            node(1, "attn", TraceOp::Sdpa { scale: 0.5 }, &[0, 0, 0], &[3, 4]),
        ]);
        let err = translate(&graph).expect_err("non-4D sdpa refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("expected 4D output shape")),
            "expected 4D-output refusal, got {err:?}"
        );
    }

    /// Sdpa with fewer than 3 inputs stays refused (fail closed).
    #[test]
    fn sdpa_too_few_inputs_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 3, 4]),
            node(
                1,
                "attn",
                TraceOp::Sdpa { scale: 0.5 },
                &[0, 0],
                &[1, 2, 3, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("2-input sdpa refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("at least 3 inputs")),
            "expected 3-input refusal, got {err:?}"
        );
    }
}
