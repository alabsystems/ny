// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Named faithful architecture rules; no geometry/naming inference.
use super::manifest::{digest_serialized, tensor_hash_start};
use super::{
    add, mul, BlockIdentity, DecoderArchitecture, DecoderContract,
    DecoderContractRefusal as Refusal, DecoderOptions, DecoderWeights, LoweredDecoder,
    LoweringManifest, OperationIdentity, Result, TensorIdentity,
};
use crate::{AttributeValue, DataType, LayerSpec, Network, TensorSpec, WeightRef};
use ndarray::{ArrayD, IxDyn};
use ny_build::GraphModel;
use ny_core::LayerType;
use ny_propagate::certify::work::WorkRequest;
use sha2::Digest as _;
use std::collections::{BTreeMap, HashMap, HashSet};

struct Writer<'a> {
    weights: &'a mut DecoderWeights,
    layers: Vec<LayerSpec>,
    shapes: HashMap<String, Vec<i64>>,
    operations: Vec<OperationIdentity>,
    blocks: Vec<BlockIdentity>,
    used: HashSet<String>,
    options: &'a DecoderOptions,
}
impl Writer<'_> {
    fn required(&mut self, name: &str, shape: &[usize]) -> Result<()> {
        super::checkpoint(self.options)?;
        let value = self
            .weights
            .weights
            .get(name)
            .ok_or_else(|| Refusal::Tensor(format!("missing {name}")))?;
        if value.shape() != shape {
            return Err(Refusal::Tensor(format!(
                "{name}: expected {shape:?}, got {:?}",
                value.shape()
            )));
        }
        self.used.insert(name.to_string());
        Ok(())
    }
    fn op(
        &mut self,
        name: String,
        kind: LayerType,
        inputs: Vec<String>,
        shape: Vec<i64>,
        attrs: HashMap<String, AttributeValue>,
    ) -> Result<String> {
        if self.layers.len() >= self.options.max_nodes {
            return Err(Refusal::Resource("lowering node cap"));
        }
        super::checkpoint(self.options)?;
        let output = format!("{name}:out");
        let parameters = attrs
            .iter()
            .map(|(k, v)| {
                let value = match v {
                    AttributeValue::Int(x) => serde_json::json!(x),
                    AttributeValue::Ints(x) => serde_json::json!(x),
                    AttributeValue::Float(x) => serde_json::json!(x),
                    AttributeValue::Floats(x) => serde_json::json!(x),
                    AttributeValue::String(x) => serde_json::json!(x),
                };
                (k.clone(), value)
            })
            .collect();
        let tensors = inputs
            .iter()
            .filter_map(|name| self.weights.identities.get(name).cloned())
            .collect();
        self.operations.push(OperationIdentity {
            name: name.clone(),
            operator: kind.to_string(),
            inputs: inputs.clone(),
            output: output.clone(),
            output_shape: shape.clone(),
            parameters,
            tensors,
        });
        self.shapes.insert(output.clone(), shape);
        self.layers.push(LayerSpec {
            name,
            layer_type: kind,
            inputs,
            outputs: vec![output.clone()],
            weights: None,
            attributes: attrs,
        });
        Ok(output)
    }
    fn block(&mut self, name: &str) -> Result<()> {
        let operations = std::mem::take(&mut self.operations);
        let digest = digest_serialized(b"block", &(name, &operations))?;
        self.blocks.push(BlockIdentity {
            name: name.to_string(),
            operations,
            digest,
        });
        Ok(())
    }
    fn linear(
        &mut self,
        node: &str,
        input: &str,
        weight: &str,
        bias: Option<&str>,
        out: usize,
        inner: usize,
        seq: usize,
    ) -> Result<String> {
        self.required(weight, &[out, inner])?;
        let mut inputs = vec![input.to_string(), weight.to_string()];
        if let Some(b) = bias {
            self.required(b, &[out])?;
            inputs.push(b.to_string());
        }
        let output = self.op(
            node.to_string(),
            LayerType::Linear,
            inputs,
            vec![1, seq as i64, out as i64],
            HashMap::from([("transB".into(), AttributeValue::Int(1))]),
        )?;
        self.layers.last_mut().expect("op appended").weights = Some(WeightRef {
            name: weight.to_string(),
            shape: vec![out, inner],
            original_dtype: DataType::Float32,
        });
        Ok(output)
    }
    fn norm(
        &mut self,
        node: &str,
        input: &str,
        prefix: &str,
        size: usize,
        seq: usize,
        epsilon: f32,
        rms: bool,
    ) -> Result<String> {
        let weight = format!("{prefix}.weight");
        self.required(&weight, &[size])?;
        let mut inputs = vec![input.to_string(), weight];
        if !rms {
            let bias = format!("{prefix}.bias");
            self.required(&bias, &[size])?;
            inputs.push(bias);
        }
        self.op(
            node.to_string(),
            if rms {
                LayerType::RMSNorm
            } else {
                LayerType::LayerNorm
            },
            inputs,
            vec![1, seq as i64, size as i64],
            HashMap::from([
                ("epsilon".into(), AttributeValue::Float(epsilon)),
                (
                    "normalized_shape".into(),
                    AttributeValue::Ints(vec![size as i64]),
                ),
            ]),
        )
    }
    fn reshape(&mut self, node: &str, input: &str, shape: &[i64]) -> Result<String> {
        self.op(
            node.into(),
            LayerType::Reshape,
            vec![input.into()],
            shape.to_vec(),
            HashMap::from([("shape".into(), AttributeValue::Ints(shape.to_vec()))]),
        )
    }
    fn transpose(
        &mut self,
        node: &str,
        input: &str,
        shape: &[i64],
        perm: &[i64],
    ) -> Result<String> {
        self.op(
            node.into(),
            LayerType::Transpose,
            vec![input.into()],
            shape.to_vec(),
            HashMap::from([("perm".into(), AttributeValue::Ints(perm.to_vec()))]),
        )
    }
    fn slice(
        &mut self,
        node: &str,
        input: &str,
        start: usize,
        end: usize,
        shape: &[i64],
    ) -> Result<String> {
        self.op(
            node.into(),
            LayerType::Slice,
            vec![input.into()],
            shape.to_vec(),
            HashMap::from([
                ("axis".into(), AttributeValue::Int(-1)),
                ("start".into(), AttributeValue::Int(start as i64)),
                ("end".into(), AttributeValue::Int(end as i64)),
            ]),
        )
    }
    fn binary(
        &mut self,
        node: &str,
        kind: LayerType,
        a: &str,
        b: &str,
        shape: &[i64],
    ) -> Result<String> {
        self.op(
            node.into(),
            kind,
            vec![a.into(), b.into()],
            shape.to_vec(),
            HashMap::new(),
        )
    }
    fn constant(&mut self, name: &str, shape: &[usize], values: Vec<f32>) -> Result<()> {
        if self.weights.weights.contains_key(name) {
            return Err(Refusal::Tensor(format!(
                "generated constant collision {name}"
            )));
        }
        let mut hash = tensor_hash_start(name, shape, "generated-pinned-table-F32");
        for &v in &values {
            if !v.is_finite() {
                return Err(Refusal::Tensor("nonfinite positional table".into()));
            }
            hash.update(v.to_bits().to_le_bytes());
        }
        let digest = hash.finalize().into();
        self.weights.identities.insert(
            name.into(),
            TensorIdentity {
                name: name.into(),
                shape: shape.to_vec(),
                source_dtype: "generated-pinned-table-F32".into(),
                source_digest: digest,
                decoded_f32_digest: digest,
            },
        );
        self.weights.weights.insert(
            name.into(),
            ArrayD::from_shape_vec(IxDyn(shape), values)
                .map_err(|e| Refusal::Tensor(e.to_string()))?,
        );
        // Constants already use the internal unbatched convention [1,S,R/2].
        self.shapes
            .insert(name.into(), shape.iter().map(|&d| d as i64).collect());
        Ok(())
    }
    fn rotary(
        &mut self,
        node: &str,
        input: &str,
        heads: usize,
        seq: usize,
        dim: usize,
        rotary: usize,
    ) -> Result<String> {
        let shape = [1, heads as i64, seq as i64, (rotary / 2) as i64];
        let a = self.slice(&format!("{node}.first_half"), input, 0, rotary / 2, &shape)?;
        let b = self.slice(
            &format!("{node}.second_half"),
            input,
            rotary / 2,
            rotary,
            &shape,
        )?;
        let ac = self.binary(
            &format!("{node}.a_cos"),
            LayerType::Mul,
            &a,
            "_ny_decoder.cos",
            &shape,
        )?;
        let bs = self.binary(
            &format!("{node}.b_sin"),
            LayerType::Mul,
            &b,
            "_ny_decoder.sin",
            &shape,
        )?;
        let bc = self.binary(
            &format!("{node}.b_cos"),
            LayerType::Mul,
            &b,
            "_ny_decoder.cos",
            &shape,
        )?;
        let as_ = self.binary(
            &format!("{node}.a_sin"),
            LayerType::Mul,
            &a,
            "_ny_decoder.sin",
            &shape,
        )?;
        let first = self.binary(
            &format!("{node}.rotated_first"),
            LayerType::Sub,
            &ac,
            &bs,
            &shape,
        )?;
        let second = self.binary(
            &format!("{node}.rotated_second"),
            LayerType::Add,
            &bc,
            &as_,
            &shape,
        )?;
        let mut pieces = vec![first, second];
        if rotary < dim {
            let pass = self.slice(
                &format!("{node}.pass"),
                input,
                rotary,
                dim,
                &[1, heads as i64, seq as i64, (dim - rotary) as i64],
            )?;
            pieces.push(pass);
        }
        self.op(
            format!("{node}.concat"),
            LayerType::Concat,
            pieces,
            vec![1, heads as i64, seq as i64, dim as i64],
            HashMap::from([("axis".into(), AttributeValue::Int(-1))]),
        )
    }
    fn repeat_kv(
        &mut self,
        node: &str,
        input: &str,
        kv: usize,
        query: usize,
        seq: usize,
        dim: usize,
    ) -> Result<String> {
        if kv == query {
            return Ok(input.into());
        }
        let shaped = self.reshape(
            &format!("{node}.group_view"),
            input,
            &[1, kv as i64, 1, seq as i64, dim as i64],
        )?;
        let tiled = self.op(
            format!("{node}.repeat_groups"),
            LayerType::Tile,
            vec![shaped],
            vec![1, kv as i64, (query / kv) as i64, seq as i64, dim as i64],
            HashMap::from([
                ("axis".into(), AttributeValue::Int(-3)),
                ("reps".into(), AttributeValue::Int((query / kv) as i64)),
            ]),
        )?;
        self.reshape(
            &format!("{node}.query_head_view"),
            &tiled,
            &[1, query as i64, seq as i64, dim as i64],
        )
    }
}

/// Authoritative graph lowering; exact named shapes and operators, no fallback
/// to architecture guesses. Positions are bound per query and attention is a
/// complete causal full-prefix computation without external KV pages.
pub fn lower_decoder(
    mut weights: DecoderWeights,
    contract: &DecoderContract,
    positions: &[u64],
    options: &DecoderOptions,
) -> Result<LoweredDecoder> {
    super::checkpoint(options)?;
    let p = contract.parameters().clone();
    let seq = positions.len();
    if seq == 0
        || seq > options.max_sequence
        || seq > p.max_positions
        || positions.iter().any(|&v| v >= p.max_positions as u64)
    {
        return Err(Refusal::Resource("sequence or position cap"));
    }
    for &v in positions {
        if v > 1 << 24 {
            return Err(Refusal::Unsupported(
                "positions must be exactly representable binary32".into(),
            ));
        }
    }
    let node_cap = add(mul(p.num_layers, 100)?, 8)?;
    if node_cap > options.max_nodes {
        return Err(Refusal::Resource("decoder node admission"));
    }
    let table_bytes = mul(mul(seq, p.rotary_dim)?, 4)?;
    let metadata_bytes = mul(node_cap, 4096)?;
    let extra_bytes = add(
        table_bytes,
        add(metadata_bytes, mul(weights.identities.len(), 1024)?)?,
    )?;
    if add(weights.resident_bytes, extra_bytes)? > options.max_resident_bytes
        || metadata_bytes > options.max_scratch_bytes
    {
        return Err(Refusal::Resource("lowered decoder residency"));
    }
    let work = add(mul(seq, p.rotary_dim)?, node_cap)? as u64;
    if work > options.max_terms || options.max_proposals == 0 {
        return Err(Refusal::Resource("lowering work admission"));
    }
    let mut lease = weights
        .budget
        .lease(WorkRequest {
            scratch_bytes: metadata_bytes,
            persistent_bytes: extra_bytes,
            work,
            proposals: 1,
        })
        .map_err(Refusal::Budget)?;
    let original_tensors = weights.identities.values().cloned().collect::<Vec<_>>();
    let source_bytes = weights.source_bytes;
    let coefficients = weights.coefficients;
    let mut writer = Writer {
        weights: &mut weights,
        layers: Vec::with_capacity(node_cap),
        shapes: HashMap::new(),
        operations: Vec::new(),
        blocks: Vec::new(),
        used: HashSet::new(),
        options,
    };
    writer.shapes.insert(
        "embedded_tokens".into(),
        vec![1, seq as i64, p.hidden_size as i64],
    );
    let half = p.rotary_dim / 2;
    let mut cos = Vec::with_capacity(mul(seq, half)?);
    let mut sin = Vec::with_capacity(mul(seq, half)?);
    for &position in positions {
        lease.checkpoint().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        for k in 0..half {
            // HF default RoPE: inverse frequency then a binary32 position
            // product. These generated values are explicit pinned constants.
            let exponent = (2 * k) as f32 / p.rotary_dim as f32;
            let inv = 1.0f32 / p.rope_base.powf(exponent);
            let angle = inv * position as f32;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    writer.constant("_ny_decoder.cos", &[1, seq, half], cos)?;
    writer.constant("_ny_decoder.sin", &[1, seq, half], sin)?;
    let (embedding, final_norm, head) = match p.architecture {
        DecoderArchitecture::GptNeoX => (
            "gpt_neox.embed_in.weight",
            "gpt_neox.final_layer_norm",
            "embed_out.weight",
        ),
        DecoderArchitecture::Qwen2 => ("model.embed_tokens.weight", "model.norm", "lm_head.weight"),
    };
    writer.required(embedding, &[p.vocab_size, p.hidden_size])?;
    // The original token lookup is included in the manifest, even though the
    // continuous verification graph intentionally starts at its embedding.
    writer.operations.push(OperationIdentity {
        name: "token_embedding_lookup".into(),
        operator: "CheckedTokenLookup".into(),
        inputs: vec!["token_ids".into(), embedding.into()],
        output: "embedded_tokens".into(),
        output_shape: vec![1, seq as i64, p.hidden_size as i64],
        parameters: BTreeMap::new(),
        tensors: vec![writer
            .weights
            .identities
            .get(embedding)
            .expect("required tensor")
            .clone()],
    });
    writer.block("embedding_and_positions")?;
    let mut previous = "embedded_tokens".to_string();
    let mut block_outputs = Vec::with_capacity(p.num_layers);
    let qdim = mul(p.query_heads, p.head_dim)?;
    let kvdim = mul(p.key_value_heads, p.head_dim)?;
    for index in 0..p.num_layers {
        lease.checkpoint().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        let prefix = match p.architecture {
            DecoderArchitecture::GptNeoX => format!("gpt_neox.layers.{index}"),
            DecoderArchitecture::Qwen2 => format!("model.layers.{index}"),
        };
        let node = format!("decoder.block.{index}");
        let shape = [1, seq as i64, p.hidden_size as i64];
        let rms = p.architecture == DecoderArchitecture::Qwen2;
        let norm1 = writer.norm(
            &format!("{node}.input_norm"),
            &previous,
            &format!("{prefix}.input_layernorm"),
            p.hidden_size,
            seq,
            p.norm_epsilon,
            rms,
        )?;
        let heads_shape = [1, p.query_heads as i64, seq as i64, p.head_dim as i64];
        let kv_shape = [1, p.key_value_heads as i64, seq as i64, p.head_dim as i64];
        let (q, k, v) = if !rms {
            let w = format!("{prefix}.attention.query_key_value.weight");
            let b = format!("{prefix}.attention.query_key_value.bias");
            let packed = writer.linear(
                &format!("{node}.packed_qkv"),
                &norm1,
                &w,
                p.attention_bias.then_some(b.as_str()),
                mul(qdim, 3)?,
                p.hidden_size,
                seq,
            )?;
            let viewed = writer.reshape(
                &format!("{node}.packed_heads"),
                &packed,
                &[1, seq as i64, p.query_heads as i64, (3 * p.head_dim) as i64],
            )?;
            let split = writer.transpose(
                &format!("{node}.head_major_qkv"),
                &viewed,
                &[1, p.query_heads as i64, seq as i64, (3 * p.head_dim) as i64],
                &[0, 2, 1, 3],
            )?;
            (
                writer.slice(
                    &format!("{node}.query"),
                    &split,
                    0,
                    p.head_dim,
                    &heads_shape,
                )?,
                writer.slice(
                    &format!("{node}.key"),
                    &split,
                    p.head_dim,
                    2 * p.head_dim,
                    &kv_shape,
                )?,
                writer.slice(
                    &format!("{node}.value"),
                    &split,
                    2 * p.head_dim,
                    3 * p.head_dim,
                    &kv_shape,
                )?,
            )
        } else {
            let mut output = Vec::with_capacity(3);
            for (name, heads, dim) in [
                ("q", p.query_heads, qdim),
                ("k", p.key_value_heads, kvdim),
                ("v", p.key_value_heads, kvdim),
            ] {
                let w = format!("{prefix}.self_attn.{name}_proj.weight");
                let b = format!("{prefix}.self_attn.{name}_proj.bias");
                let projected = writer.linear(
                    &format!("{node}.{name}_projection"),
                    &norm1,
                    &w,
                    Some(&b),
                    dim,
                    p.hidden_size,
                    seq,
                )?;
                let viewed = writer.reshape(
                    &format!("{node}.{name}_heads"),
                    &projected,
                    &[1, seq as i64, heads as i64, p.head_dim as i64],
                )?;
                output.push(writer.transpose(
                    &format!("{node}.{name}_head_major"),
                    &viewed,
                    &[1, heads as i64, seq as i64, p.head_dim as i64],
                    &[0, 2, 1, 3],
                )?);
            }
            (output[0].clone(), output[1].clone(), output[2].clone())
        };
        let q = writer.rotary(
            &format!("{node}.query_rope"),
            &q,
            p.query_heads,
            seq,
            p.head_dim,
            p.rotary_dim,
        )?;
        let k = writer.rotary(
            &format!("{node}.key_rope"),
            &k,
            p.key_value_heads,
            seq,
            p.head_dim,
            p.rotary_dim,
        )?;
        let k = writer.repeat_kv(
            &format!("{node}.key_gqa"),
            &k,
            p.key_value_heads,
            p.query_heads,
            seq,
            p.head_dim,
        )?;
        let v = writer.repeat_kv(
            &format!("{node}.value_gqa"),
            &v,
            p.key_value_heads,
            p.query_heads,
            seq,
            p.head_dim,
        )?;
        let scores = writer.op(
            format!("{node}.scores"),
            LayerType::MatMul,
            vec![q, k],
            vec![1, p.query_heads as i64, seq as i64, seq as i64],
            HashMap::from([
                ("transpose_b".into(), AttributeValue::Int(1)),
                (
                    "scale".into(),
                    AttributeValue::Float((p.head_dim as f64).powf(-0.5) as f32),
                ),
            ]),
        )?;
        let probs = writer.op(
            format!("{node}.causal_probabilities"),
            LayerType::CausalSoftmax,
            vec![scores],
            vec![1, p.query_heads as i64, seq as i64, seq as i64],
            HashMap::from([("axis".into(), AttributeValue::Int(-1))]),
        )?;
        let context = writer.binary(
            &format!("{node}.weighted_values"),
            LayerType::MatMul,
            &probs,
            &v,
            &heads_shape,
        )?;
        let token_major = writer.transpose(
            &format!("{node}.token_major_context"),
            &context,
            &[1, seq as i64, p.query_heads as i64, p.head_dim as i64],
            &[0, 2, 1, 3],
        )?;
        let flat = writer.reshape(
            &format!("{node}.flatten_context"),
            &token_major,
            &[1, seq as i64, qdim as i64],
        )?;
        let (out_w, out_b) = if !rms {
            (
                format!("{prefix}.attention.dense.weight"),
                p.attention_bias
                    .then(|| format!("{prefix}.attention.dense.bias")),
            )
        } else {
            (format!("{prefix}.self_attn.o_proj.weight"), None)
        };
        let attention = writer.linear(
            &format!("{node}.attention_output"),
            &flat,
            &out_w,
            out_b.as_deref(),
            p.hidden_size,
            qdim,
            seq,
        )?;
        let mlp_input = if p.parallel_residual {
            previous.clone()
        } else {
            writer.binary(
                &format!("{node}.attention_residual"),
                LayerType::Add,
                &attention,
                &previous,
                &shape,
            )?
        };
        let norm2 = writer.norm(
            &format!("{node}.post_attention_norm"),
            &mlp_input,
            &format!("{prefix}.post_attention_layernorm"),
            p.hidden_size,
            seq,
            p.norm_epsilon,
            rms,
        )?;
        let mlp = if !rms {
            let up_w = format!("{prefix}.mlp.dense_h_to_4h.weight");
            let up_b = format!("{prefix}.mlp.dense_h_to_4h.bias");
            let up = writer.linear(
                &format!("{node}.mlp_up"),
                &norm2,
                &up_w,
                Some(&up_b),
                p.intermediate_size,
                p.hidden_size,
                seq,
            )?;
            let activated = writer.op(
                format!("{node}.gelu_erf"),
                LayerType::GELU,
                vec![up],
                vec![1, seq as i64, p.intermediate_size as i64],
                HashMap::from([("approximate".into(), AttributeValue::String("none".into()))]),
            )?;
            let w = format!("{prefix}.mlp.dense_4h_to_h.weight");
            let b = format!("{prefix}.mlp.dense_4h_to_h.bias");
            writer.linear(
                &format!("{node}.mlp_down"),
                &activated,
                &w,
                Some(&b),
                p.hidden_size,
                p.intermediate_size,
                seq,
            )?
        } else {
            let gate_w = format!("{prefix}.mlp.gate_proj.weight");
            let up_w = format!("{prefix}.mlp.up_proj.weight");
            let gate = writer.linear(
                &format!("{node}.mlp_gate"),
                &norm2,
                &gate_w,
                None,
                p.intermediate_size,
                p.hidden_size,
                seq,
            )?;
            let up = writer.linear(
                &format!("{node}.mlp_up"),
                &norm2,
                &up_w,
                None,
                p.intermediate_size,
                p.hidden_size,
                seq,
            )?;
            let activated = writer.op(
                format!("{node}.silu"),
                LayerType::SiLU,
                vec![gate],
                vec![1, seq as i64, p.intermediate_size as i64],
                HashMap::new(),
            )?;
            let product = writer.binary(
                &format!("{node}.swiglu"),
                LayerType::Mul,
                &activated,
                &up,
                &[1, seq as i64, p.intermediate_size as i64],
            )?;
            writer.linear(
                &format!("{node}.mlp_down"),
                &product,
                &format!("{prefix}.mlp.down_proj.weight"),
                None,
                p.hidden_size,
                p.intermediate_size,
                seq,
            )?
        };
        previous = if p.parallel_residual {
            let branches = writer.binary(
                &format!("{node}.parallel_branches"),
                LayerType::Add,
                &mlp,
                &attention,
                &shape,
            )?;
            writer.binary(
                &format!("{node}.output"),
                LayerType::Add,
                &branches,
                &previous,
                &shape,
            )?
        } else {
            writer.binary(
                &format!("{node}.output"),
                LayerType::Add,
                &mlp,
                &mlp_input,
                &shape,
            )?
        };
        block_outputs.push(previous.clone());
        writer.block(&format!("block.{index}"))?;
    }
    let hidden = writer.norm(
        "decoder.final_norm",
        &previous,
        final_norm,
        p.hidden_size,
        seq,
        p.norm_epsilon,
        p.architecture == DecoderArchitecture::Qwen2,
    )?;
    let head_name = if p.tie_word_embeddings {
        if let Some(other) = writer.weights.weights.get(head) {
            let embed = writer
                .weights
                .weights
                .get(embedding)
                .expect("required embedding");
            if other.shape() != embed.shape()
                || !other
                    .iter()
                    .zip(embed)
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
            {
                return Err(Refusal::Tensor(
                    "tied head differs from embedding coefficients".into(),
                ));
            }
            writer.used.insert(head.into());
        }
        embedding
    } else {
        head
    };
    let logits = writer.linear(
        "decoder.logits",
        &hidden,
        head_name,
        None,
        p.vocab_size,
        p.hidden_size,
        seq,
    )?;
    writer.block("final_norm_and_head")?;
    for tensor in &original_tensors {
        if !writer.used.contains(&tensor.name) {
            return Err(Refusal::Unsupported(format!(
                "unconsumed original tensor {}",
                tensor.name
            )));
        }
    }
    let mut outputs = vec![
        TensorSpec {
            name: logits.clone(),
            shape: vec![1, seq as i64, p.vocab_size as i64],
            dtype: DataType::Float32,
        },
        TensorSpec {
            name: hidden.clone(),
            shape: vec![1, seq as i64, p.hidden_size as i64],
            dtype: DataType::Float32,
        },
    ];
    outputs.extend(block_outputs.iter().map(|name| TensorSpec {
        name: name.clone(),
        shape: vec![1, seq as i64, p.hidden_size as i64],
        dtype: DataType::Float32,
    }));
    let network = Network {
        name: "closed_decoder_reference".into(),
        inputs: vec![TensorSpec {
            name: "embedded_tokens".into(),
            shape: vec![1, seq as i64, p.hidden_size as i64],
            dtype: DataType::Float32,
        }],
        outputs,
        layers: std::mem::take(&mut writer.layers),
        param_count: usize::try_from(coefficients)
            .map_err(|_| Refusal::Resource("parameter count"))?,
    };
    let shapes = std::mem::take(&mut writer.shapes);
    let blocks = std::mem::take(&mut writer.blocks);
    drop(writer);
    let mut manifest=LoweringManifest{schema_version:1,parameters:p.clone(),positions:positions.to_vec(),original_tensors,
        input_boundary:"checked original embedding lookup; live embedded coordinates; full causal prefix; no external KV cache/tokenizer/logit processors".into(),
        blocks,lowering_digest:[0;32],source_bytes_hashed:source_bytes,decoded_coefficients_hashed:coefficients};
    manifest.lowering_digest = digest_serialized(b"lowering", &manifest)?;
    lease.complete().map_err(Refusal::Budget)?;
    let extra_reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
    let graph_model = GraphModel::new(network, weights.weights).with_tensor_shapes(shapes);
    super::checkpoint(options)?;
    Ok(LoweredDecoder {
        graph_model,
        manifest,
        embedding: embedding.into(),
        hidden_output: hidden,
        logits_output: logits,
        block_outputs,
        sequence: seq,
        budget: weights.budget,
        weight_reservation: weights.reservation,
        extra_reservation,
        resident_bytes: add(weights.resident_bytes, extra_bytes)?,
    })
}
