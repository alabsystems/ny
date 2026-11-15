// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Explicit supported semantics and explicit nonsemantic metadata allowlist.
use super::{DecoderContractRefusal as Refusal, DecoderOptions, Result};
use serde::{
    de::{MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DecoderArchitecture {
    GptNeoX,
    Qwen2,
}

/// Every value is consumed by a named operator or validated inference rule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecoderParameters {
    pub architecture: DecoderArchitecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub max_positions: usize,
    pub norm_epsilon: f32,
    pub parallel_residual: bool,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    /// This milestone supports complete causal attention; windows refuse.
    pub causal_attention: bool,
    /// Decoded binary32 coefficients and observations, with versioned mixed
    /// binary32/binary64 diagnostic graph arithmetic and hashed finite RoPE
    /// tables. Linear uses original-weight binary32 products without a deadline
    /// and pollable binary64 contraction with one narrowing under a deadline;
    /// Causal softmax uses max-shift binary64 libm and a single nearest-f32
    /// publication. Other operators include binary64 reductions/nonlinear
    /// interval centers. Each point observation separately binds its actual
    /// completed software arms; this profile alone permits multiple arms.
    /// This is not a Torch or deployed-kernel execution contract.
    pub reference_profile: String,
}

/// Private construction prevents unvalidated combinations of public parameters.
#[derive(Debug, Clone)]
pub struct DecoderContract {
    parameters: DecoderParameters,
}
impl DecoderContract {
    pub fn parameters(&self) -> &DecoderParameters {
        &self.parameters
    }

    /// Transfer the validated owned configuration without cloning its profile.
    pub(super) fn into_parameters(self) -> DecoderParameters {
        self.parameters
    }

    pub fn from_json(json: &[u8], options: &DecoderOptions) -> Result<Self> {
        super::checkpoint(options)?;
        if json.len() > options.max_config_bytes {
            return Err(Refusal::Resource("configuration byte cap"));
        }
        let scratch = super::add(super::mul(json.len(), 16)?, 4096)?;
        if scratch > options.max_scratch_bytes {
            return Err(Refusal::Resource("configuration scratch cap"));
        }
        let budget = super::budget(options)?;
        let mut lease = budget
            .lease(ny_propagate::certify::work::WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: 0,
                work: json.len() as u64,
                proposals: 1,
            })
            .map_err(Refusal::Budget)?;
        // Conflicting duplicate semantic fields cannot disappear into Value's
        // ordinary last-writer-wins object map.
        let ClosedObject(map): ClosedObject =
            serde_json::from_slice(json).map_err(|e| Refusal::Config(format!("JSON: {e}")))?;
        let map = &map;
        let architecture = match string(map, "model_type")? {
            "gpt_neox" => DecoderArchitecture::GptNeoX,
            "qwen2" => DecoderArchitecture::Qwen2,
            other => {
                return Err(Refusal::Unsupported(format!(
                    "decoder architecture {other}"
                )))
            }
        };
        for key in map.keys() {
            if !metadata(key) && !common_semantic(key) && !architecture_semantic(architecture, key)
            {
                return Err(Refusal::Unsupported(format!(
                    "unknown configuration field '{key}'"
                )));
            }
        }
        let expected_architecture = match architecture {
            DecoderArchitecture::GptNeoX => "GPTNeoXForCausalLM",
            DecoderArchitecture::Qwen2 => "Qwen2ForCausalLM",
        };
        if let Some(architectures) = map.get("architectures") {
            if architectures != &serde_json::json!([expected_architecture]) {
                return Err(Refusal::Unsupported(
                    "architectures must identify the supported causal LM".into(),
                ));
            }
        }
        for key in ["quantization_config", "rope_scaling"] {
            if map.get(key).is_some_and(|v| !v.is_null()) {
                return Err(Refusal::Unsupported(format!(
                    "{key} needs its own lowering contract"
                )));
            }
        }
        for key in [
            "is_encoder_decoder",
            "add_cross_attention",
            "use_mrope",
            "use_sliding_window",
        ] {
            if bool_or(map, key, false)? {
                return Err(Refusal::Unsupported(format!("{key}=true")));
            }
        }
        if map
            .get("_attn_implementation")
            .is_some_and(|v| v != "eager")
        {
            return Err(Refusal::Unsupported(
                "reference attention is CPU eager".into(),
            ));
        }
        if let Some(types) = map.get("layer_types") {
            if !types.is_null()
                && !types.as_array().is_some_and(|v| {
                    v.len() == usize_value(map, "num_hidden_layers").unwrap_or(0)
                        && v.iter().all(|t| t == "full_attention")
                })
            {
                return Err(Refusal::Unsupported(
                    "layer_types require full_attention in every layer".into(),
                ));
            }
        }
        // These fields describe behavior deliberately disabled in inference.
        for key in ["attention_dropout", "hidden_dropout"] {
            let p = number_or(map, key, 0.0)?;
            if !(0.0..=1.0).contains(&p) {
                return Err(Refusal::Config(format!("invalid {key}")));
            }
        }
        for key in [
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        ] {
            bool_or(map, key, key == "use_cache" || key == "return_dict")?;
        }
        let hidden_size = usize_value(map, "hidden_size")?;
        let query_heads = usize_value(map, "num_attention_heads")?;
        let key_value_heads = match architecture {
            DecoderArchitecture::GptNeoX => query_heads,
            DecoderArchitecture::Qwen2 => {
                // A missing source count is not evidence for MHA: Qwen's HF
                // constructor has its own default, independent of Q heads.
                usize_value(map, "num_key_value_heads")?
            }
        };
        if hidden_size % query_heads != 0 || query_heads % key_value_heads != 0 {
            return Err(Refusal::Config("hidden/head or Q/KV divisibility".into()));
        }
        let head_dim = optional_usize(map, "head_dim")?.unwrap_or(hidden_size / query_heads);
        let (rotary_dim, rope_base, norm_epsilon, parallel_residual, attention_bias) =
            match architecture {
                DecoderArchitecture::GptNeoX => {
                    if head_dim != hidden_size / query_heads {
                        return Err(Refusal::Unsupported("GPTNeoX head_dim override".into()));
                    }
                    if string(map, "hidden_act")? != "gelu" {
                        return Err(Refusal::Unsupported(
                            "GPTNeoX activation must be GELU-erf".into(),
                        ));
                    }
                    let pct = number_or(map, "rotary_pct", 0.25)?;
                    if !(0.0..=1.0).contains(&pct) {
                        return Err(Refusal::Config("rotary_pct".into()));
                    }
                    let r = (head_dim as f64 * pct) as usize;
                    (
                        r,
                        positive_f32(map, "rotary_emb_base", 10000.0)?,
                        positive_f32(map, "layer_norm_eps", 1e-5)?,
                        bool_or(map, "use_parallel_residual", true)?,
                        bool_or(map, "attention_bias", true)?,
                    )
                }
                DecoderArchitecture::Qwen2 => {
                    if string(map, "hidden_act")? != "silu" {
                        return Err(Refusal::Unsupported(
                            "Qwen2 activation must be SiLU/SwiGLU".into(),
                        ));
                    }
                    (
                        head_dim,
                        positive_f32(map, "rope_theta", 10000.0)?,
                        positive_f32(map, "rms_norm_eps", 1e-6)?,
                        false,
                        true,
                    )
                }
            };
        if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
            return Err(Refusal::Config(
                "rotary dimension must be nonzero, even and <= head_dim".into(),
            ));
        }
        let max_positions = usize_value(map, "max_position_embeddings")?;
        let num_layers = usize_value(map, "num_hidden_layers")?;
        // Disabled windows are consumed and checked, never guessed active.
        optional_usize(map, "sliding_window")?;
        if let Some(max) = map.get("max_window_layers") {
            max.as_u64()
                .ok_or_else(|| Refusal::Config("max_window_layers".into()))?;
        }
        let parameters = DecoderParameters {
            architecture,
            hidden_size,
            intermediate_size: usize_value(map, "intermediate_size")?,
            vocab_size: usize_value(map, "vocab_size")?,
            num_layers,
            query_heads,
            key_value_heads,
            head_dim,
            rotary_dim,
            rope_base,
            max_positions,
            norm_epsilon,
            parallel_residual,
            attention_bias,
            tie_word_embeddings: bool_or(map, "tie_word_embeddings", false)?,
            causal_attention: true,
            reference_profile: "decoded-f32-coefficients/ny-graph-point-mixed-f32-f64/linear-original-weight-f32-or-deadline-f64/causal-max-shift-libm-f64-one-f32-store/completed-arm-observation-v1/half-split-pinned-table/v3"
                .into(),
        };
        super::mul(query_heads, head_dim)?;
        super::mul(key_value_heads, head_dim)?;
        lease.complete().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        Ok(Self { parameters })
    }
}

struct ClosedObject(Map<String, Value>);
impl<'de> Deserialize<'de> for ClosedObject {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct ObjectVisitor;
        impl<'de> Visitor<'de> for ObjectVisitor {
            type Value = ClosedObject;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a configuration object without duplicate fields")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut access: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut map = Map::new();
                while let Some((key, value)) = access.next_entry::<String, Value>()? {
                    if map.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!("duplicate field {key}")));
                    }
                }
                Ok(ClosedObject(map))
            }
        }
        deserializer.deserialize_map(ObjectVisitor)
    }
}

fn metadata(key: &str) -> bool {
    matches!(
        key,
        "architectures"
            | "torch_dtype"
            | "dtype"
            | "transformers_version"
            | "_name_or_path"
            | "bos_token_id"
            | "eos_token_id"
            | "pad_token_id"
            | "initializer_range"
            | "classifier_dropout"
            | "problem_type"
            | "id2label"
            | "label2id"
            | "finetuning_task"
    )
}
fn common_semantic(key: &str) -> bool {
    matches!(
        key,
        "model_type"
            | "hidden_size"
            | "intermediate_size"
            | "vocab_size"
            | "num_hidden_layers"
            | "num_attention_heads"
            | "max_position_embeddings"
            | "hidden_act"
            | "tie_word_embeddings"
            | "rope_scaling"
            | "quantization_config"
            | "attention_dropout"
            | "use_cache"
            | "is_encoder_decoder"
            | "add_cross_attention"
            | "output_attentions"
            | "output_hidden_states"
            | "return_dict"
            | "_attn_implementation"
    )
}
fn architecture_semantic(a: DecoderArchitecture, key: &str) -> bool {
    match a {
        DecoderArchitecture::GptNeoX => matches!(
            key,
            "rotary_emb_base"
                | "rotary_pct"
                | "layer_norm_eps"
                | "use_parallel_residual"
                | "attention_bias"
                | "hidden_dropout"
        ),
        DecoderArchitecture::Qwen2 => matches!(
            key,
            "num_key_value_heads"
                | "head_dim"
                | "rope_theta"
                | "rms_norm_eps"
                | "use_sliding_window"
                | "sliding_window"
                | "max_window_layers"
                | "layer_types"
                | "use_mrope"
        ),
    }
}
fn string<'a>(m: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    m.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Refusal::Config(format!("required string {key}")))
}
fn usize_value(m: &Map<String, Value>, key: &str) -> Result<usize> {
    optional_usize(m, key)?
        .ok_or_else(|| Refusal::Config(format!("required positive integer {key}")))
}
fn optional_usize(m: &Map<String, Value>, key: &str) -> Result<Option<usize>> {
    match m.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .map(Some)
            .ok_or_else(|| Refusal::Config(format!("positive integer {key}"))),
    }
}
fn bool_or(m: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    m.get(key)
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| Refusal::Config(format!("boolean {key}")))
        })
        .unwrap_or(Ok(default))
}
fn number_or(m: &Map<String, Value>, key: &str, default: f64) -> Result<f64> {
    let n = m
        .get(key)
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| Refusal::Config(format!("number {key}")))
        })
        .unwrap_or(Ok(default))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(Refusal::Config(format!("finite {key}")))
    }
}
fn positive_f32(m: &Map<String, Value>, key: &str, default: f64) -> Result<f32> {
    let n = number_or(m, key, default)? as f32;
    if n.is_finite() && n > 0.0 {
        Ok(n)
    } else {
        Err(Refusal::Config(format!("positive finite binary32 {key}")))
    }
}
