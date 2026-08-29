// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention family: `Sdpa`, `SdpaCausal`, `RotaryEmbedding` (plus the
//! opaque `MultiHeadAttention` refusal).
//!
//! Ported from NN's `trace_to_graph_layerspec_attention.rs` (the trace-path
//! ground truth; the dispatch arms live in
//! `trace_to_graph_layerspec_dispatch_extended.rs`), then extended: the
//! maskless Sdpa/SdpaCausal forms emit ny's NATIVE fused self-attention
//! layer instead of NN's generic decomposition.
//!
//! ## Native fused emission (maskless Sdpa / SdpaCausal)
//!
//! One `LayerType::MultiHeadAttention` spec with inputs `[Q, K, V]` and
//! attributes `scale` (Float; always emitted) plus `causal` (Int 1,
//! SdpaCausal only). ny-build's convert arm reads exactly those attributes
//! and produces `Layer::SelfAttention(SelfAttentionLayer::new(mask, scale))`
//! (ny-propagate `layers/attention/mod.rs`), whose wiring —
//! `inputs[0..3] = Q, K, V`, computing `softmax((Q @ K^T) * scale) @ V`
//! with K transposed on its trailing dims — matches the traced Sdpa
//! semantics exactly (Q `[B,H,S_q,D]`, K/V `[B,H,S_kv,D]`), i.e. the same
//! contraction the decomposition spelled out. The scale must be explicit:
//! `GraphNetwork` refuses a fused attention node without one (head_dim is
//! not inferrable at graph-construction time), and the trace always
//! records it.
//!
//! Why native: wherever the fused layer is propagated directly, the ternary
//! paths are strictly tighter than the generic decomposition — its IBP
//! applies the softmax row-stochastic simplex lever on `probs @ V`
//! (#softmax-V-lever) and its CROWN backward takes the default-on
//! `propagate_crown_ternary` surface (#attn-crown; the whisper-attention
//! tightness win). Where the graph instead decomposes the fused node on
//! insertion (`GraphNetwork::try_add_node` → `decompose_self_attention`),
//! it lowers to BilinearCrown + Softmax/CausalSoftmax + BilinearCrown —
//! the SAME primitives ny-build produced from the old MatMul/Softmax
//! LayerSpec decomposition — so the native emission is never looser than
//! the decomposition it replaces (pinned by the native-vs-decomposed IBP
//! test below).
//!
//! ## Generic SDPA decomposition (kept where the native layer cannot apply)
//!
//! - **Masked Sdpa** (4th input): `SelfAttentionLayer` has no mask input.
//! - **Constant Q/K/V**: ny-build pins ternary layers to exactly three
//!   dynamic activation inputs (`ny-build/src/graph/inputs.rs`), so a
//!   constant operand cannot wire into the fused node; the decomposition
//!   supports it (MatMul-with-constant lowers to Linear).
//!
//! `Sdpa { scale }` with inputs Q, K, V, [mask] decomposes to:
//! 1. `MatMul(Q, K^T)` with `transpose_b=1`, `scale`
//! 2. `Add(scores, mask)` (if mask present)
//! 3. `Softmax(axis=last_dim)` → attention weights
//! 4. `MatMul(attn_weights, V)` → output
//!
//! Both MatMul ops have two activation inputs (Q/K and attn/V are all
//! variable), so graph-build produces a bilinear CROWN layer with McCormick
//! relaxation. `SdpaCausal` swaps `CausalSoftmax` for `Softmax+mask`,
//! avoiding the O(S²) mask allocation.
//!
//! ## RotaryEmbedding
//!
//! Maps to `LayerType::RoPE` with cos/sin frequency weights extracted from
//! the `TraceOp::RotaryEmbedding` variant's `cos_cache` / `sin_cache` payloads
//! (`head_dim` / `offset` are not consumed here, mirroring NN's dispatch —
//! the caches are already narrowed to the traced positions).
//!
//! ## MultiHeadAttention (the fused composite)
//!
//! LOWERED, where ny has a layer for the parameterization. The refusal that
//! used to stand here was a WIRE-FORMAT refusal — the variant carried only
//! `num_heads`/`num_kv_heads`/`head_dim`, so a lowering would have had to
//! invent the projections — and nn-core now serializes `scale`, all four
//! projections, and the `rope`/`cross_attention`/`has_mask` discriminators.
//! Maskless, RoPE-free, non-GQA self- and cross-attention lowers to the four
//! Linears + head reshape/transpose + the SAME native fused layer the maskless
//! `Sdpa` arm emits. Masked, RoPE-fused and grouped-query nodes are refused
//! individually, each naming the layer ny is actually missing — see
//! [`translate_multi_head_attention`]. Traced models that surface attention as
//! separate Linear projections + Sdpa/SdpaCausal keep taking those arms.

use std::collections::HashMap;

use ny_build::{AttributeValue, LayerSpec};
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp, WeightPayload};

use super::{
    checked_f64_to_f32, dim_as_i64, insert_payload, op_name, ops_core, shape_to_i64, simple_spec,
    Ctx, NodeOutput,
};

/// Translate an attention-family op (`Sdpa`, `SdpaCausal`, `RotaryEmbedding`)
/// node; `MultiHeadAttention` refuses here with the wire-format-gap message.
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
        // The fused MultiHeadAttention node LOWERS now: the projections are on
        // the wire, so the enclosure can be built from the trace alone. The
        // parameterizations ny still has no layer for are refused
        // individually — see [`translate_multi_head_attention`].
        TraceOp::MultiHeadAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            rope,
            cross_attention,
            has_mask,
        } => translate_multi_head_attention(
            name,
            MhaGeometry {
                num_heads: *num_heads,
                num_kv_heads: *num_kv_heads,
                head_dim: *head_dim,
                scale: *scale,
                cross_attention: *cross_attention,
                has_mask: *has_mask,
                has_rope: rope.is_some(),
            },
            MhaProjections {
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                out_weight,
                out_bias,
            },
            input_tensors,
            output_tensor,
            &node.output_shape,
            ctx,
        ),
        other => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// MultiHeadAttention (the fused composite)
// ---------------------------------------------------------------------------

/// Head geometry and kernel-shape flags of one fused MHA node.
#[derive(Debug, Clone, Copy)]
struct MhaGeometry {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    cross_attention: bool,
    has_mask: bool,
    has_rope: bool,
}

/// The four frozen projections of one fused MHA node.
struct MhaProjections<'a> {
    q_weight: &'a WeightPayload,
    q_bias: &'a Option<WeightPayload>,
    k_weight: &'a WeightPayload,
    k_bias: &'a Option<WeightPayload>,
    v_weight: &'a WeightPayload,
    v_bias: &'a Option<WeightPayload>,
    out_weight: &'a WeightPayload,
    out_bias: &'a Option<WeightPayload>,
}

/// Translate the fused `TraceOp::MultiHeadAttention` into ny's native
/// attention layer surrounded by its four projections.
///
/// # Why this is no longer refused
///
/// The refusal that stood here was a WIRE-FORMAT refusal: the variant carried
/// only `num_heads`/`num_kv_heads`/`head_dim`, so any lowering would have had
/// to invent the projection weights, and a bound over invented weights is a
/// false proof. nn-core now serializes `scale` and all four projections
/// (weight + optional bias) plus the `rope`/`cross_attention`/`has_mask`
/// discriminators, recorded from the same values the compute closure used. The
/// enclosure can therefore be built from the trace alone, and the refusal's
/// premise is gone.
///
/// # The emitted graph
///
/// Replicates `MultiHeadAttention::forward`'s compute body op-for-op
/// (nn-core `layers/attention/multi_head.rs`), with `kv_src = x` for
/// self-attention and `kv_src = inputs[1]` for cross-attention:
///
/// ```text
/// q = Linear(x,      q_weight, q_bias)   [B, S_q,  H  * hd]
/// k = Linear(kv_src, k_weight, k_bias)   [B, S_kv, Hkv* hd]
/// v = Linear(kv_src, v_weight, v_bias)   [B, S_kv, Hkv* hd]
///     Reshape  [B, S, H, hd]  then  Transpose(0,2,1,3) -> [B, H, S, hd]
/// a = MultiHeadAttention(q, k, v, scale) [B, H, S_q, hd]
///     Transpose(0,2,1,3) then Reshape    [B, S_q, H * hd]
/// y = Linear(a, out_weight, out_bias)    [B, S_q, D_out]
/// ```
///
/// The fused node is `LayerType::MultiHeadAttention` with an explicit `scale`
/// attribute — ny-build's convert arm turns it into
/// `Layer::SelfAttention(SelfAttentionLayer::new(mask, scale))`, whose
/// `softmax((Q @ K^T) * scale) @ V` over `Q [B,H,S_q,hd]`, `K/V [B,H,S_kv,hd]`
/// is exactly what `sdpa` computed. `scale` is the RECORDED multiplier, not
/// `1/sqrt(head_dim)` re-derived, so a custom scale cannot silently diverge.
///
/// # Soundness
///
/// Every step is either an exact reindex (Reshape/Transpose) or an affine map
/// with frozen weights (the four Linears); the only approximation is the
/// attention core, and that is the SAME `SelfAttentionLayer` the maskless
/// `Sdpa` arm already emits — this arm neither widens nor tightens it. The
/// projections are lowered through the same `LayerType::Linear` +
/// `transB = 1` spec [`ops_core::translate_linear`] emits, so their float
/// contract is the graph-wide one (`propagate_ibp` round-to-nearest,
/// `propagate_ibp_sound` certified per-node widening) rather than anything
/// special to this arm.
///
/// # Refusals — each names what ny is actually missing
///
/// * **`has_mask`** — `SelfAttentionLayer` has no mask input. This is the same
///   gap that sends masked `Sdpa` down the generic decomposition, but here the
///   mask is an INPUT EDGE of a fused node whose scores tensor is never
///   materialized as a graph tensor, so there is nothing to add it to without
///   decomposing the whole composite.
/// * **`rope`** — RoPE is applied to Q and K *inside* the fused kernel, between
///   the transpose and the attention. The captured windows are lowerable in
///   principle (`RotaryEmbedding` has an arm), but that arm is written against
///   a standalone node, and splicing it between two internal tensors of this
///   composite is a separate piece of work; refusing is fail-closed, emitting
///   the un-rotated attention would silently model a different function.
/// * **`num_kv_heads != num_heads` (GQA)** — the kernel repeats each K/V head
///   `num_heads / num_kv_heads` times before attending. Without that repeat the
///   fused layer would contract `H` query heads against `Hkv` key heads, which
///   is a shape error at best and the wrong pairing at worst.
/// * **a constant `x` or `kv_src`** — ny-build pins ternary layers to exactly
///   three DYNAMIC activation inputs (`ny-build/src/graph/inputs.rs`); a
///   constant operand is filtered out of the wiring and the build fails the
///   arity check. Refused by name here rather than left to fail deep in build.
///
/// Traced models that surface attention as separate Linear projections +
/// `Sdpa`/`SdpaCausal` are unaffected — they keep taking those arms.
fn translate_multi_head_attention(
    name: &str,
    geom: MhaGeometry,
    proj: MhaProjections<'_>,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if geom.has_mask {
        return Err(NyError::UnsupportedOp(
            "MultiHeadAttention with an attention mask cannot be lowered: ny's fused \
             SelfAttentionLayer takes exactly three inputs (Q, K, V) and has no mask operand, \
             and this node's scores tensor is never a graph tensor, so there is nothing to add \
             the mask to without decomposing the composite. The projections ARE on the wire now \
             — this is a layer gap, not a wire gap. Maskless self- and cross-attention lower"
                .to_string(),
        ));
    }
    if geom.has_rope {
        return Err(NyError::UnsupportedOp(
            "MultiHeadAttention with fused RoPE cannot be lowered: the rotation is applied to Q \
             and K INSIDE the kernel, between the head transpose and the attention, and the \
             bridge's RotaryEmbedding arm is written against a standalone trace node rather \
             than an internal tensor of this composite. The captured cos/sin windows are on the \
             wire, so this is an unported lowering rather than a wire gap — but emitting the \
             un-rotated attention would silently model a different function, so it fails closed"
                .to_string(),
        ));
    }
    if geom.num_heads == 0 || geom.num_kv_heads == 0 || geom.head_dim == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "MultiHeadAttention: degenerate head geometry num_heads={} num_kv_heads={} \
             head_dim={}",
            geom.num_heads, geom.num_kv_heads, geom.head_dim
        )));
    }
    if geom.num_kv_heads != geom.num_heads {
        return Err(NyError::UnsupportedOp(format!(
            "MultiHeadAttention with grouped-query attention (num_heads={}, num_kv_heads={}) \
             cannot be lowered: the kernel repeats each K/V head {} times before attending \
             (nn's repeat_kv), and the bridge has no lowering for that repeat, so the fused \
             layer would contract {} query heads against {} key heads — the wrong pairing. \
             num_kv_heads == num_heads lowers",
            geom.num_heads,
            geom.num_kv_heads,
            geom.num_heads / geom.num_kv_heads,
            geom.num_heads,
            geom.num_kv_heads,
        )));
    }

    // -- Operands. Input order is the wire contract: x, [kv_input], [mask].
    //    `has_mask` is refused above, so a second input here is the K/V source
    //    exactly when `cross_attention` says so.
    let expected_inputs = 1 + usize::from(geom.cross_attention);
    if input_tensors.len() != expected_inputs {
        return Err(NyError::InternalError(format!(
            "MultiHeadAttention: expected {expected_inputs} input(s) for \
             cross_attention={} / has_mask={}, got {}",
            geom.cross_attention,
            geom.has_mask,
            input_tensors.len()
        )));
    }
    let x = input_tensors[0].clone();
    let kv_src = if geom.cross_attention {
        input_tensors[1].clone()
    } else {
        x.clone()
    };
    if ctx.constant_tensors.contains(&x) || ctx.constant_tensors.contains(&kv_src) {
        return Err(NyError::UnsupportedOp(
            "MultiHeadAttention with a constant Q or K/V source cannot be lowered: ny-build \
             pins ternary layers to exactly three DYNAMIC activation inputs, so a constant \
             operand is filtered out of the wiring and the fused node fails its arity check"
                .to_string(),
        ));
    }

    // -- Geometry. nn's forward reads (b, s_q, _d) from x and s_kv from the
    //    K/V source, so both must be the rank-3 [B, S, D] the kernel requires.
    let x_shape = lookup_shape(ctx, &x, "MultiHeadAttention x")?;
    let kv_shape = lookup_shape(ctx, &kv_src, "MultiHeadAttention kv_src")?;
    if x_shape.len() != 3 || kv_shape.len() != 3 {
        return Err(NyError::UnsupportedOp(format!(
            "MultiHeadAttention: expected rank-3 [B, S, D] operands, got x {x_shape:?} and \
             kv_src {kv_shape:?} — nn's forward destructures x with dims3(), so any other rank \
             means the trace disagrees with the op it records"
        )));
    }
    let (batch, s_q) = (x_shape[0], x_shape[1]);
    let s_kv = kv_shape[1];
    if kv_shape[0] != batch {
        return Err(NyError::InternalError(format!(
            "MultiHeadAttention: K/V source batch {} does not match x batch {batch}",
            kv_shape[0]
        )));
    }
    if output_shape.len() != 3 || output_shape[0] != batch || output_shape[1] != s_q {
        return Err(NyError::InternalError(format!(
            "MultiHeadAttention: recorded output shape {output_shape:?} contradicts the \
             [B={batch}, S_q={s_q}, D_out] the composite produces"
        )));
    }

    let q_dim = geom.num_heads.checked_mul(geom.head_dim).ok_or_else(|| {
        NyError::InternalError("MultiHeadAttention: H * head_dim overflows".into())
    })?;
    let kv_dim = geom
        .num_kv_heads
        .checked_mul(geom.head_dim)
        .ok_or_else(|| {
            NyError::InternalError("MultiHeadAttention: H_kv * head_dim overflows".into())
        })?;

    // Projection shapes are checked against the geometry rather than trusted:
    // a weight that disagrees would reshape into the wrong head layout and
    // produce a confidently wrong bound.
    check_projection(proj.q_weight, proj.q_bias, q_dim, x_shape[2], "q")?;
    check_projection(proj.k_weight, proj.k_bias, kv_dim, kv_shape[2], "k")?;
    check_projection(proj.v_weight, proj.v_bias, kv_dim, kv_shape[2], "v")?;
    check_projection(
        proj.out_weight,
        proj.out_bias,
        output_shape[2],
        q_dim,
        "out",
    )?;

    let scale = checked_f64_to_f32(geom.scale, "MultiHeadAttention scale")?;

    let mut specs = Vec::with_capacity(12);

    // -- Q/K/V: Linear -> Reshape [B, S, H, hd] -> Transpose to [B, H, S, hd].
    let q = mha_project_to_heads(
        name,
        "q",
        proj.q_weight,
        proj.q_bias,
        &x,
        batch,
        s_q,
        geom.num_heads,
        geom.head_dim,
        &mut specs,
        ctx,
    )?;
    let k = mha_project_to_heads(
        name,
        "k",
        proj.k_weight,
        proj.k_bias,
        &kv_src,
        batch,
        s_kv,
        geom.num_kv_heads,
        geom.head_dim,
        &mut specs,
        ctx,
    )?;
    let v = mha_project_to_heads(
        name,
        "v",
        proj.v_weight,
        proj.v_bias,
        &kv_src,
        batch,
        s_kv,
        geom.num_kv_heads,
        geom.head_dim,
        &mut specs,
        ctx,
    )?;

    // -- The fused attention core. `causal = false`: this op carries no causal
    //    flag, and a causal trace records the mask as an explicit operand,
    //    which `has_mask` refuses above.
    let attn_name = format!("{name}_attn");
    let attn_out = format!("{attn_name}_out");
    specs.extend(native_self_attention(&attn_name, scale, false, &q, &k, &v, &attn_out).specs);
    let attn_shape = vec![batch, geom.num_heads, s_q, geom.head_dim];
    register_shape(ctx, &attn_out, &attn_shape, "MultiHeadAttention attn")?;

    // -- Merge heads back: Transpose(0,2,1,3) then Reshape [B, S_q, H*hd].
    let merge_t_name = format!("{name}_merge_t");
    let merge_t_out = format!("{merge_t_name}_out");
    specs.push(head_transpose_spec(&merge_t_name, &attn_out, &merge_t_out));
    register_shape(
        ctx,
        &merge_t_out,
        &[batch, s_q, geom.num_heads, geom.head_dim],
        "MultiHeadAttention merge transpose",
    )?;

    let merge_r_name = format!("{name}_merge_r");
    let merge_r_out = format!("{merge_r_name}_out");
    let merged_shape = vec![batch, s_q, q_dim];
    specs.push(mha_reshape_spec(
        &merge_r_name,
        &merge_t_out,
        &merge_r_out,
        &merged_shape,
    )?);
    register_shape(ctx, &merge_r_out, &merged_shape, "MultiHeadAttention merge")?;

    // -- Output projection. This spec carries the NODE's name, so the fused
    //    trace node's output tensor is produced by a layer named for it.
    specs.extend(
        ops_core::translate_linear(
            name,
            proj.out_weight,
            proj.out_bias,
            &[merge_r_out],
            output_tensor,
            ctx,
        )?
        .specs,
    );

    Ok(NodeOutput { specs })
}

/// Check one projection's captured shapes against the geometry the head
/// reshape assumes. Fails closed: a disagreement means the reshape below would
/// slice the projection into the wrong head layout.
fn check_projection(
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    out_features: usize,
    in_features: usize,
    which: &str,
) -> Result<()> {
    if weight.shape != [out_features, in_features] {
        return Err(NyError::InternalError(format!(
            "MultiHeadAttention: {which} projection weight {:?} contradicts the geometry \
             [out_features={out_features}, in_features={in_features}] implied by the head \
             counts and the recorded operand shapes",
            weight.shape
        )));
    }
    if let Some(b) = bias {
        if b.shape != [out_features] {
            return Err(NyError::InternalError(format!(
                "MultiHeadAttention: {which} projection bias {:?} contradicts \
                 [out_features={out_features}]",
                b.shape
            )));
        }
    }
    Ok(())
}

/// Emit `Linear -> Reshape [B, S, heads, head_dim] -> Transpose(0,2,1,3)` for
/// one of Q/K/V, returning the `[B, heads, S, head_dim]` tensor name.
///
/// Mirrors nn's
/// `proj.forward(src).reshape([b, s, heads, head_dim])?.transpose(1, 2)`.
#[allow(clippy::too_many_arguments)]
fn mha_project_to_heads(
    node_name: &str,
    which: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    src: &str,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    specs: &mut Vec<LayerSpec>,
    ctx: &mut Ctx,
) -> Result<String> {
    let proj_name = format!("{node_name}_{which}_proj");
    let proj_out = format!("{proj_name}_out");
    specs.extend(
        ops_core::translate_linear(&proj_name, weight, bias, &[src.to_string()], &proj_out, ctx)?
            .specs,
    );
    let flat = heads.checked_mul(head_dim).ok_or_else(|| {
        NyError::InternalError(format!(
            "MultiHeadAttention {which}: heads * head_dim overflows"
        ))
    })?;
    register_shape(
        ctx,
        &proj_out,
        &[batch, seq, flat],
        "MultiHeadAttention projection",
    )?;

    let rs_name = format!("{node_name}_{which}_heads");
    let rs_out = format!("{rs_name}_out");
    let head_shape = vec![batch, seq, heads, head_dim];
    specs.push(mha_reshape_spec(&rs_name, &proj_out, &rs_out, &head_shape)?);
    register_shape(ctx, &rs_out, &head_shape, "MultiHeadAttention head split")?;

    let t_name = format!("{node_name}_{which}_t");
    let t_out = format!("{t_name}_out");
    specs.push(head_transpose_spec(&t_name, &rs_out, &t_out));
    register_shape(
        ctx,
        &t_out,
        &[batch, heads, seq, head_dim],
        "MultiHeadAttention head transpose",
    )?;
    Ok(t_out)
}

/// `Transpose` swapping axes 1 and 2 of a rank-4 tensor — nn's
/// `.transpose(1, 2)` on `[B, S, H, hd]` / `[B, H, S, hd]`, which is its own
/// inverse and so serves both the split and the merge direction.
fn head_transpose_spec(name: &str, input: &str, output: &str) -> LayerSpec {
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), AttributeValue::Ints(vec![0, 2, 1, 3]));
    simple_spec(
        name,
        LayerType::Transpose,
        vec![input.to_string()],
        output,
        attrs,
    )
}

/// `Reshape` to an explicit static shape.
fn mha_reshape_spec(name: &str, input: &str, output: &str, shape: &[usize]) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "shape".to_string(),
        AttributeValue::Ints(shape_to_i64(shape, name)?),
    );
    Ok(simple_spec(
        name,
        LayerType::Reshape,
        vec![input.to_string()],
        output,
        attrs,
    ))
}

/// Record an intermediate tensor's shape so later lookups resolve it.
fn register_shape(ctx: &mut Ctx, tensor: &str, shape: &[usize], context: &str) -> Result<()> {
    let dims = shape_to_i64(shape, context)?;
    ctx.tensor_shapes.entry(tensor.to_string()).or_insert(dims);
    Ok(())
}

/// Look up a previously translated tensor's recorded shape as `usize` dims.
fn lookup_shape(ctx: &Ctx, tensor: &str, context: &str) -> Result<Vec<usize>> {
    let shape = ctx.tensor_shapes.get(tensor).ok_or_else(|| {
        NyError::InternalError(format!("{context}: input shape not found for {tensor}"))
    })?;
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                NyError::InternalError(format!("{context}: negative dimension {dim} in {tensor}"))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Native fused self-attention emission
// ---------------------------------------------------------------------------

/// Whether the native fused self-attention emission applies to this Q/K/V
/// wiring.
///
/// ny-build pins ternary layers to EXACTLY three dynamic activation inputs
/// (`ny-build/src/graph/inputs.rs`): a Q/K/V operand that resolved to a
/// constant tensor is filtered out of the graph wiring and the build fails
/// closed on the arity check. The generic decomposition DOES support
/// constant operands (ny-build lowers MatMul-with-a-constant-side to
/// Linear), so those keep it.
fn native_self_attention_applies(ctx: &Ctx, q: &str, k: &str, v: &str) -> bool {
    !(ctx.constant_tensors.contains(q)
        || ctx.constant_tensors.contains(k)
        || ctx.constant_tensors.contains(v))
}

/// Emit the native fused self-attention LayerSpec.
///
/// `LayerType::MultiHeadAttention` with inputs `[Q, K, V]` is ny-build's
/// fused-attention spec: its convert arm (`ny-build/src/convert/mod.rs`)
/// reads the `scale` (Float) and `causal` (Int 1) attributes emitted here
/// and produces `Layer::SelfAttention(SelfAttentionLayer::new(mask, scale))`
/// in Standard or Causal mode. The scale is ALWAYS emitted: graph insertion
/// refuses a fused attention node without an explicit scale (head_dim is
/// not inferrable at graph-construction time), and the trace always records
/// one.
fn native_self_attention(
    name: &str,
    scale_f32: f32,
    causal: bool,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    output_tensor: &str,
) -> NodeOutput {
    let mut attrs = HashMap::new();
    attrs.insert("scale".to_string(), AttributeValue::Float(scale_f32));
    if causal {
        attrs.insert("causal".to_string(), AttributeValue::Int(1));
    }
    NodeOutput::one(simple_spec(
        name,
        LayerType::MultiHeadAttention,
        vec![
            q_tensor.to_string(),
            k_tensor.to_string(),
            v_tensor.to_string(),
        ],
        output_tensor,
        attrs,
    ))
}

// ---------------------------------------------------------------------------
// SDPA
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Sdpa { scale }`: native fused self-attention when
/// maskless with all-activation Q/K/V, otherwise the MatMul + [Add] +
/// Softmax + MatMul decomposition.
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

    // Native fused emission for the maskless all-activation form. Placed
    // AFTER the shape validation above so switching emission never widens
    // the accepted surface: every input the decomposition would have
    // refused is still refused with the same error.
    if !has_mask && native_self_attention_applies(ctx, q_tensor, k_tensor, v_tensor) {
        return Ok(native_self_attention(
            name,
            scale_f32,
            false, // standard (bidirectional) mask mode
            q_tensor,
            k_tensor,
            v_tensor,
            output_tensor,
        ));
    }

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

/// Translate `TraceOp::SdpaCausal { scale }`: native fused causal
/// self-attention when Q/K/V are all activations, otherwise the MatMul +
/// CausalSoftmax + MatMul decomposition.
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

    // Native fused emission (Causal mode). Same ordering rationale as the
    // Sdpa arm: validation first, so acceptance never widens.
    if native_self_attention_applies(ctx, q_tensor, k_tensor, v_tensor) {
        return Ok(native_self_attention(
            name,
            scale_f32,
            true, // causal mask mode
            q_tensor,
            k_tensor,
            v_tensor,
            output_tensor,
        ));
    }

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

    /// Maskless Sdpa emits ny's NATIVE fused self-attention: exactly one
    /// MultiHeadAttention spec wired `[Q, K, V]` with the explicit scale and
    /// no causal attribute — no MatMul/Softmax decomposition specs at all.
    #[test]
    fn sdpa_emits_native_self_attention() {
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

        assert_eq!(
            count(&model, &LayerType::MultiHeadAttention),
            1,
            "one fused attention spec"
        );
        assert_eq!(count(&model, &LayerType::MatMul), 0, "no decomposition");
        assert_eq!(count(&model, &LayerType::Softmax), 0, "no decomposition");
        assert_eq!(count(&model, &LayerType::CausalSoftmax), 0, "not causal");
        // Only the input-identity Add.
        assert_eq!(count(&model, &LayerType::Add), 1, "no mask Add");

        let attn = find(&model, "layer0_trace_1");
        assert_eq!(attn.layer_type, LayerType::MultiHeadAttention);
        assert_eq!(
            attn.attributes.get("scale"),
            Some(&AttributeValue::Float(0.125)),
            "explicit scale (graph insertion refuses a fused attention without one)"
        );
        assert_eq!(attn.attributes.get("causal"), None, "standard mode");
        assert_eq!(
            attn.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_0_out".to_string(),
                "layer0_trace_0_out".to_string(),
            ],
            "fused node consumes Q, K, V in trace order"
        );
        assert_eq!(attn.outputs, vec!["layer0_trace_1_out".to_string()]);

        build_ok(&model, "native sdpa");
    }

    /// Sdpa with a 4th (mask) input KEEPS the generic decomposition (the
    /// native SelfAttentionLayer has no mask input) and inserts
    /// Add(scores, mask) before Softmax.
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

        assert_eq!(
            count(&model, &LayerType::MultiHeadAttention),
            0,
            "masked Sdpa must NOT take the native fused emission"
        );
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

    /// SdpaCausal emits the native fused self-attention in CAUSAL mode:
    /// exactly one MultiHeadAttention spec with `causal = 1` and the
    /// explicit scale (ny-build's convert arm maps it to
    /// `AttentionMask::Causal`).
    #[test]
    fn sdpa_causal_emits_native_causal_self_attention() {
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

        assert_eq!(
            count(&model, &LayerType::MultiHeadAttention),
            1,
            "one fused attention spec"
        );
        assert_eq!(count(&model, &LayerType::MatMul), 0, "no decomposition");
        assert_eq!(
            count(&model, &LayerType::CausalSoftmax),
            0,
            "no decomposition"
        );
        assert_eq!(count(&model, &LayerType::Softmax), 0, "no decomposition");
        assert_eq!(count(&model, &LayerType::Add), 1, "no mask Add");

        let attn = find(&model, "layer0_trace_1");
        assert_eq!(attn.layer_type, LayerType::MultiHeadAttention);
        assert_eq!(
            attn.attributes.get("scale"),
            Some(&AttributeValue::Float(0.125))
        );
        assert_eq!(
            attn.attributes.get("causal"),
            Some(&AttributeValue::Int(1)),
            "causal mode flag"
        );

        build_ok(&model, "native causal sdpa");
    }

    /// Sdpa with a CONSTANT operand keeps the generic decomposition: ny-build
    /// pins ternary layers to exactly three dynamic activation inputs, so a
    /// constant V could never wire into the fused node (the decomposition
    /// lowers MatMul-with-constant to Linear instead). Translation-level pin
    /// only — the 4D-constant build behavior is ny-build's, unchanged here.
    #[test]
    fn sdpa_constant_operand_keeps_decomposition() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 3, 4]),
            node(
                1,
                "v_const",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.5; 24], vec![1, 2, 3, 4]),
                },
                &[],
                &[1, 2, 3, 4],
            ),
            node(
                2,
                "attn",
                TraceOp::Sdpa { scale: 0.5 },
                &[0, 0, 1],
                &[1, 2, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("constant-V sdpa translates");
        assert_eq!(
            count(&model, &LayerType::MultiHeadAttention),
            0,
            "constant operand must not take the native fused emission"
        );
        assert_eq!(count(&model, &LayerType::MatMul), 2, "decomposition kept");
        assert_eq!(count(&model, &LayerType::Softmax), 1, "decomposition kept");
    }

    /// f64 reference SDPA over flat row-major `[B, H, S, D]` with Q = K = V
    /// = `x` (per-head `softmax((x xᵀ) * scale) x`, causally masked when
    /// asked).
    fn reference_sdpa(
        x: &[f32],
        b: usize,
        h: usize,
        s: usize,
        d: usize,
        scale: f64,
        causal: bool,
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; b * h * s * d];
        for bi in 0..b {
            for hh in 0..h {
                let base = (bi * h + hh) * s * d;
                let at = |i: usize, dd: usize| x[base + i * d + dd] as f64;
                for i in 0..s {
                    let jmax = if causal { i + 1 } else { s };
                    let mut row = vec![0.0_f64; jmax];
                    for (j, r) in row.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for dd in 0..d {
                            acc += at(i, dd) * at(j, dd);
                        }
                        *r = acc * scale;
                    }
                    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut denom = 0.0;
                    let probs: Vec<f64> = row
                        .iter()
                        .map(|&v| {
                            let e = (v - m).exp();
                            denom += e;
                            e
                        })
                        .collect();
                    for dd in 0..d {
                        let mut acc = 0.0;
                        for (j, &p) in probs.iter().enumerate() {
                            acc += (p / denom) * at(j, dd);
                        }
                        out[base + i * d + dd] = acc;
                    }
                }
            }
        }
        out
    }

    /// Native emission is SOUND and NO LOOSER than the generic decomposition
    /// on the same box.
    ///
    /// The decomposed baseline is produced through the production path by an
    /// all-zero additive mask: `x + 0.0` is exact in f32 interval
    /// arithmetic, so the masked decomposition's IBP equals the maskless
    /// decomposition this emission replaced. Soundness: concrete Q=K=V
    /// points sampled inside the box must land inside the native bounds
    /// (f64 reference attention). Tightness: per element, the native
    /// interval is contained in the decomposed one (float-dust slack only).
    #[test]
    fn sdpa_native_ibp_sound_and_no_looser_than_decomposition() {
        let (b, h, s, d) = (1_usize, 2, 3, 4);
        let n = b * h * s * d;
        let scale = 0.5_f64;
        let center: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).sin() * 0.5).collect();
        let radius = 0.05_f32;

        let native_graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[b, h, s, d]),
            node(
                1,
                "attn",
                TraceOp::Sdpa { scale },
                &[0, 0, 0],
                &[b, h, s, d],
            ),
        ]);
        let decomposed_graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[b, h, s, d]),
            node(
                1,
                "mask",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.0; b * h * s * s], vec![b, h, s, s]),
                },
                &[],
                &[b, h, s, s],
            ),
            node(
                2,
                "attn",
                TraceOp::Sdpa { scale },
                &[0, 0, 0, 1],
                &[b, h, s, d],
            ),
        ]);

        let native = translate(&native_graph).expect("native sdpa translates");
        assert_eq!(count(&native, &LayerType::MultiHeadAttention), 1);
        let decomposed = translate(&decomposed_graph).expect("decomposed sdpa translates");
        assert_eq!(count(&decomposed, &LayerType::MultiHeadAttention), 0);

        let shape = ndarray::IxDyn(&[b, h, s, d]);
        let lower = ndarray::ArrayD::from_shape_vec(
            shape.clone(),
            center.iter().map(|c| c - radius).collect(),
        )
        .expect("lower");
        let upper =
            ndarray::ArrayD::from_shape_vec(shape, center.iter().map(|c| c + radius).collect())
                .expect("upper");
        let input = ny_tensor::BoundedTensor::new(lower, upper).expect("valid box");

        let native_out = native
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("native builds")
            .propagate_ibp(&input)
            .expect("native IBP");
        let dec_out = decomposed
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("decomposed builds")
            .propagate_ibp(&input)
            .expect("decomposed IBP");
        assert_eq!(native_out.shape(), &[b, h, s, d]);
        assert_eq!(dec_out.shape(), native_out.shape());

        // Soundness: sampled concrete points inside the box are enclosed.
        let clamp = |v: f32, i: usize| v.clamp(center[i] - radius, center[i] + radius);
        let samples: Vec<Vec<f32>> = vec![
            center.clone(),
            center.iter().map(|c| c - radius).collect(),
            center.iter().map(|c| c + radius).collect(),
            center
                .iter()
                .enumerate()
                .map(|(i, c)| clamp(if i % 2 == 0 { c + radius } else { c - radius }, i))
                .collect(),
            center
                .iter()
                .enumerate()
                .map(|(i, c)| clamp(if i % 3 == 0 { c - radius } else { c + radius }, i))
                .collect(),
        ];
        for (si, x) in samples.iter().enumerate() {
            let reference = reference_sdpa(x, b, h, s, d, scale, false);
            for bi in 0..b {
                for hh in 0..h {
                    for i in 0..s {
                        for dd in 0..d {
                            let r = reference[((bi * h + hh) * s + i) * d + dd];
                            let lo = native_out.lower()[[bi, hh, i, dd]] as f64;
                            let hi = native_out.upper()[[bi, hh, i, dd]] as f64;
                            assert!(
                                lo - 1e-5 <= r && r <= hi + 1e-5,
                                "sample {si} [{bi},{hh},{i},{dd}]: reference {r} \
                                 outside native bounds [{lo}, {hi}]"
                            );
                        }
                    }
                }
            }
        }

        // Tightness: native ⊆ decomposed, elementwise.
        for bi in 0..b {
            for hh in 0..h {
                for i in 0..s {
                    for dd in 0..d {
                        let nl = native_out.lower()[[bi, hh, i, dd]];
                        let nu = native_out.upper()[[bi, hh, i, dd]];
                        let dl = dec_out.lower()[[bi, hh, i, dd]];
                        let du = dec_out.upper()[[bi, hh, i, dd]];
                        assert!(
                            nl >= dl - 1e-6 && nu <= du + 1e-6,
                            "[{bi},{hh},{i},{dd}]: native [{nl}, {nu}] looser than \
                             decomposed [{dl}, {du}]"
                        );
                    }
                }
            }
        }
    }

    /// SdpaCausal native emission pins the exact causal attention on a point
    /// box — a Standard/Causal mode mix-up would move row 0 (self-only under
    /// the causal mask) and every masked position.
    #[test]
    fn sdpa_causal_native_ibp_pins_reference_on_point_box() {
        let (b, h, s, d) = (1_usize, 1, 3, 2);
        let scale = 0.7_f64;
        let x = vec![0.4_f32, -0.2, 0.1, 0.9, -0.5, 0.3];
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[b, h, s, d]),
            node(
                1,
                "attn",
                TraceOp::SdpaCausal { scale },
                &[0, 0, 0],
                &[b, h, s, d],
            ),
        ]);
        let model = translate(&graph).expect("causal sdpa translates");
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, h, s, d]), x.clone())
            .expect("shape");
        let input = ny_tensor::BoundedTensor::new(point.clone(), point).expect("valid box");
        let out = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("causal builds")
            .propagate_ibp(&input)
            .expect("causal IBP");

        let reference = reference_sdpa(&x, b, h, s, d, scale, true);
        for i in 0..s {
            for dd in 0..d {
                let r = reference[i * d + dd];
                let lo = out.lower()[[0, 0, i, dd]] as f64;
                let hi = out.upper()[[0, 0, i, dd]] as f64;
                assert!(
                    (lo - r).abs() < 1e-4 && (hi - r).abs() < 1e-4,
                    "[{i},{dd}]: point-box bounds [{lo}, {hi}] should pin causal \
                     reference {r}"
                );
            }
        }
    }

    /// Build a fused MHA witness with real projections for `[B=1, S, D]`.
    ///
    /// Weights are distinctive (no symmetry that could cancel a transposed
    /// axis) and the geometry is `H = num_heads`, `hd = head_dim`, `D = H*hd`,
    /// so the four projections are square `[D, D]`.
    fn mha_op(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cross_attention: bool,
        has_mask: bool,
        rope: Option<Box<crate::schema::TraceMhaRope>>,
    ) -> TraceOp {
        let d = num_heads * head_dim;
        let kv_d = num_kv_heads * head_dim;
        // Deterministic, asymmetric entries.
        let fill = |rows: usize, cols: usize, seed: f32| {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32) * 0.037 + seed).sin() * 0.5)
                .collect();
            WeightPayload::f32(data, vec![rows, cols])
        };
        let bias = |n: usize, seed: f32| {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.11 + seed).cos() * 0.25)
                .collect();
            WeightPayload::f32(data, vec![n])
        };
        TraceOp::MultiHeadAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            q_weight: fill(d, d, 0.1),
            q_bias: Some(bias(d, 0.2)),
            k_weight: fill(kv_d, d, 0.3),
            k_bias: None,
            v_weight: fill(kv_d, d, 0.5),
            v_bias: Some(bias(kv_d, 0.6)),
            out_weight: fill(d, d, 0.7),
            out_bias: Some(bias(d, 0.8)),
            rope,
            cross_attention,
            has_mask,
        }
    }

    fn mha_rope(head_dim: usize, seq: usize) -> Box<crate::schema::TraceMhaRope> {
        let half = head_dim / 2;
        let cache = || WeightPayload::f32(vec![0.5; seq * half], vec![seq, half]);
        Box::new(crate::schema::TraceMhaRope {
            head_dim,
            offset: 0,
            q_cos_cache: cache(),
            q_sin_cache: cache(),
            k_cos_cache: cache(),
            k_sin_cache: cache(),
        })
    }

    /// Self-attention MHA lowers to the four projections + head
    /// reshape/transpose + ny's NATIVE fused attention layer, and the emitted
    /// graph BUILDS.
    ///
    /// This is the case the old wire-format refusal blocked: the projections
    /// are on the wire now, so there is something to build the enclosure from.
    #[test]
    fn multi_head_attention_lowers_to_projections_plus_native_attention() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 8]),
            node(
                1,
                "mha",
                mha_op(2, 2, 4, false, false, None),
                &[0],
                &[1, 3, 8],
            ),
        ]);
        let model = translate(&graph).expect("fused MHA translates");

        // Four Linears: q, k, v projections plus the output projection (which
        // carries the node's own name).
        assert_eq!(
            count(&model, &LayerType::Linear),
            4,
            "q/k/v projections + output projection"
        );
        assert_eq!(
            count(&model, &LayerType::MultiHeadAttention),
            1,
            "one native fused attention core"
        );
        assert_eq!(count(&model, &LayerType::MatMul), 0, "no decomposition");
        assert_eq!(count(&model, &LayerType::Softmax), 0, "no decomposition");
        // Head split (3) + merge (1) reshapes, head transposes (3 + 1).
        assert_eq!(count(&model, &LayerType::Reshape), 4);
        assert_eq!(count(&model, &LayerType::Transpose), 4);

        let attn = find(&model, "layer0_trace_1_attn");
        assert_eq!(attn.layer_type, LayerType::MultiHeadAttention);
        assert_eq!(
            attn.attributes.get("scale"),
            Some(&AttributeValue::Float(0.5)),
            "the RECORDED scale (1/sqrt(4)), not a re-derivation"
        );
        assert_eq!(attn.attributes.get("causal"), None, "no causal flag on MHA");
        assert_eq!(
            attn.inputs,
            vec![
                "layer0_trace_1_q_t_out".to_string(),
                "layer0_trace_1_k_t_out".to_string(),
                "layer0_trace_1_v_t_out".to_string(),
            ],
            "fused core consumes the head-transposed Q, K, V"
        );

        // The node's output tensor is produced by the output projection.
        let out_proj = find(&model, "layer0_trace_1");
        assert_eq!(out_proj.layer_type, LayerType::Linear);
        assert_eq!(out_proj.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert_eq!(
            out_proj.attributes.get("transB"),
            Some(&AttributeValue::Int(1)),
            "PyTorch [out, in] weight convention, same as translate_linear"
        );

        build_ok(&model, "fused MHA");
    }

    /// The lowered MHA graph propagates IBP, and a point box gives finite
    /// bounds of the traced output shape.
    ///
    /// The enclosure claim is structural — every step is an exact reindex or
    /// an affine map with frozen weights except the attention core, which is
    /// the same `SelfAttentionLayer` the `Sdpa` arm emits — so what is checked
    /// here is that the wiring actually propagates end to end at the right
    /// shape rather than silently producing a vacuous or mis-shaped result.
    #[test]
    fn multi_head_attention_propagates_ibp() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 8]),
            node(
                1,
                "mha",
                mha_op(2, 2, 4, false, false, None),
                &[0],
                &[1, 3, 8],
            ),
        ]);
        let model = translate(&graph).expect("fused MHA translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("fused MHA builds");

        let values: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.21).sin()).collect();
        let point =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, 8]), values.clone()).unwrap();
        let bounds = ny_tensor::BoundedTensor::new(point.clone(), point).expect("point bounds");
        let out = network.propagate_ibp(&bounds).expect("fused MHA IBP");
        assert_eq!(out.shape(), &[1, 3, 8], "traced output shape survives");
        for (lo, hi) in out.lower().iter().zip(out.upper().iter()) {
            assert!(
                lo.is_finite() && hi.is_finite(),
                "finite bounds: [{lo}, {hi}]"
            );
            assert!(lo <= hi, "ordered bounds: [{lo}, {hi}]");
        }

        // A widened box must ENCLOSE the point-box result: monotone in the
        // input box by construction.
        let eps = 0.05_f32;
        let lower = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 8]),
            values.iter().map(|v| v - eps).collect(),
        )
        .unwrap();
        let upper = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 8]),
            values.iter().map(|v| v + eps).collect(),
        )
        .unwrap();
        let wide = ny_tensor::BoundedTensor::new(lower, upper).expect("wide bounds");
        let wide_out = network.propagate_ibp(&wide).expect("wide IBP");
        for i in 0..out.lower().len() {
            let (plo, phi) = (
                out.lower().as_slice().unwrap()[i],
                out.upper().as_slice().unwrap()[i],
            );
            let (wlo, whi) = (
                wide_out.lower().as_slice().unwrap()[i],
                wide_out.upper().as_slice().unwrap()[i],
            );
            assert!(
                wlo <= plo && phi <= whi,
                "cell {i}: widening the input box must not narrow the output \
                 ([{plo}, {phi}] not inside [{wlo}, {whi}])"
            );
        }
    }

    /// Cross-attention lowers too: K/V come from input 1, Q from input 0, and
    /// the K/V projections consume the SECOND operand.
    #[test]
    fn multi_head_cross_attention_wires_kv_from_the_second_input() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 8]),
            node(1, "enc", TraceOp::Input, &[], &[1, 5, 8]),
            node(
                2,
                "mha",
                mha_op(2, 2, 4, true, false, None),
                &[0, 1],
                &[1, 3, 8],
            ),
        ]);
        let model = super::super::translate_multi_input(&graph)
            .expect("cross-attention MHA translates (two variable inputs)")
            .model;

        let q_proj = find(&model, "layer0_trace_2_q_proj");
        assert_eq!(
            q_proj.inputs[0], "layer0_trace_0_out",
            "Q projects the query source"
        );
        for which in ["k", "v"] {
            let proj = find(&model, &format!("layer0_trace_2_{which}_proj"));
            assert_eq!(
                proj.inputs[0], "layer0_trace_1_out",
                "{which} projects the SECOND operand (the K/V source)"
            );
        }
        build_ok(&model, "cross-attention MHA");
    }

    /// The three parameterizations ny has no layer for stay refused — each
    /// naming the MISSING LAYER, not a missing wire field, because the wire
    /// now carries everything.
    #[test]
    fn multi_head_attention_refuses_what_ny_has_no_layer_for() {
        use std::collections::HashMap;

        let refuse = |op: TraceOp, inputs: &[u64], n_in: usize| -> NyError {
            let mut ctx = super::super::Ctx::new(false);
            // Register operand shapes the arm looks up.
            for i in 0..n_in {
                ctx.tensor_shapes
                    .insert(format!("layer0_trace_{i}_out"), vec![1, 3, 8]);
            }
            let names: Vec<String> = (0..n_in).map(|i| format!("layer0_trace_{i}_out")).collect();
            let mha = node(9, "mha", op, inputs, &[1, 3, 8]);
            match super::translate_attention(
                &mha,
                "layer0_trace_9",
                &names,
                "layer0_trace_9_out",
                &HashMap::new(),
                &mut ctx,
            ) {
                Ok(_) => panic!("must stay refused"),
                Err(err) => err,
            }
        };

        // Masked: SelfAttentionLayer has no mask input.
        let err = refuse(mha_op(2, 2, 4, false, true, None), &[0, 1], 2);
        assert!(
            matches!(&err, NyError::UnsupportedOp(m)
                if m.contains("attention mask")
                    && m.contains("no mask operand")
                    && m.contains("layer gap, not a wire gap")),
            "masked refusal names the layer gap: {err:?}"
        );

        // Fused RoPE: the rotation happens inside the kernel.
        let err = refuse(mha_op(2, 2, 4, false, false, Some(mha_rope(4, 3))), &[0], 1);
        assert!(
            matches!(&err, NyError::UnsupportedOp(m)
                if m.contains("fused RoPE")
                    && m.contains("INSIDE the kernel")
                    && m.contains("unported lowering rather than a wire gap")),
            "RoPE refusal names the unported lowering: {err:?}"
        );

        // GQA: the K/V head repeat has no lowering.
        let err = refuse(mha_op(4, 2, 4, false, false, None), &[0], 1);
        assert!(
            matches!(&err, NyError::UnsupportedOp(m)
                if m.contains("grouped-query attention")
                    && m.contains("repeat_kv")
                    && m.contains("wrong pairing")),
            "GQA refusal names the missing repeat: {err:?}"
        );
    }

    /// A projection whose captured shape contradicts the head geometry fails
    /// CLOSED rather than being reshaped into the wrong head layout.
    #[test]
    fn multi_head_attention_rejects_a_projection_that_contradicts_the_geometry() {
        use std::collections::HashMap;

        let mut op = mha_op(2, 2, 4, false, false, None);
        if let TraceOp::MultiHeadAttention { q_weight, .. } = &mut op {
            // Claims H*hd = 8 but carries a [6, 8] weight.
            *q_weight = WeightPayload::f32(vec![0.0; 48], vec![6, 8]);
        }
        let mut ctx = super::super::Ctx::new(false);
        ctx.tensor_shapes
            .insert("layer0_trace_0_out".to_string(), vec![1, 3, 8]);
        let mha = node(1, "mha", op, &[0], &[1, 3, 8]);
        let err = super::translate_attention(
            &mha,
            "layer0_trace_1",
            &["layer0_trace_0_out".to_string()],
            "layer0_trace_1_out",
            &HashMap::new(),
            &mut ctx,
        );
        let err = match err {
            Ok(_) => panic!("a contradictory projection must fail closed"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, NyError::InternalError(m)
                if m.contains("q projection weight") && m.contains("contradicts the geometry")),
            "got {err:?}"
        );
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
