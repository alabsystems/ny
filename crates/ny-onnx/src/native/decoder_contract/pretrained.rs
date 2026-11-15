// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Authentic whole-checkpoint owner for the pinned LLM-A profile. Fixed token
//! IDs are the input boundary: the pinned tokenizer is recorded, but this API
//! does not certify retokenization, text substitutions or a deployed kernel.

use super::{
    add, mul, DecoderArchitecture, DecoderContract, DecoderContractRefusal as Refusal,
    DecoderOptions, DecoderParameters, DecoderWeights, Result,
};
use ny_propagate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use sha2::{Digest as _, Sha256};
use std::{io::Read, path::Path, sync::Arc, time::Instant};

pub const PYTHIA14M_REVISION: &str = "cf967c0a9a04383db6f7b1108d86b2962634b4ac";
pub const PYTHIA_EXACT_REAL_PROFILE: &str = "pythia-gptneox/serialized-f16-bf16-f32-exact-real/decimal-epsilon/analytic-rope/gelu-erf/causal-complete/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PretrainedAssetIdentity {
    pub name: &'static str,
    pub sha256: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct PretrainedDecoderManifest {
    pub repository: &'static str,
    pub revision: &'static str,
    pub numeric_profile: &'static str,
    /// Architecture reference only; its finite/device execution is NOT claimed.
    pub operator_reference: &'static str,
    pub tokenizer_policy: &'static str,
    pub assets: [PretrainedAssetIdentity; 6],
    pub bos_token: usize,
    pub eos_token: usize,
    /// No implicit BOS, truncation, sampling, processors, or external KV.
    pub input_policy: &'static str,
}

const ASSETS: [(&str, &str); 6] = [
    (
        "config.json",
        "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17",
    ),
    (
        "model.safetensors",
        "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1",
    ),
    (
        "tokenizer.json",
        "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2",
    ),
    (
        "tokenizer_config.json",
        "eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809",
    ),
    (
        "special_tokens_map.json",
        "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c",
    ),
    (
        "generation_config.json",
        "494cbba887cc299273f0c54d7612596918e74c7881a27cbd95e4e7e2464ea090",
    ),
];

fn digest(text: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("fixed registry digest");
    }
    out
}

pub(super) struct PretrainedInner {
    pub(super) weights: DecoderWeights,
    pub(super) parameters: DecoderParameters,
    pub(super) manifest: PretrainedDecoderManifest,
    pub(super) deadline: Instant,
    _resident: PersistentReservation,
}

/// Clone shares the actual immutable parameters and original paid owner.
#[derive(Clone)]
pub struct PretrainedDecoder(pub(super) Arc<PretrainedInner>);
impl PretrainedDecoder {
    pub fn manifest(&self) -> &PretrainedDecoderManifest {
        &self.0.manifest
    }
    pub fn parameters(&self) -> &DecoderParameters {
        &self.0.parameters
    }
    pub fn original_tensors(&self) -> impl Iterator<Item = &super::TensorIdentity> {
        self.0.weights.identities()
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.weights.reservation.bytes() + self.0._resident.bytes()
    }
    pub fn deadline(&self) -> Instant {
        self.0.deadline
    }
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(super) fn budget(&self) -> &WorkBudget {
        &self.0.weights.budget
    }
    pub(super) fn checkpoint(&self, requested: Option<Instant>) -> Result<Instant> {
        if !self.0._resident.shares_ledger(self.budget())
            || !self.0.weights.reservation.shares_ledger(self.budget())
        {
            return Err(Refusal::Resource(
                "pretrained retained original ledger differs",
            ));
        }
        self.budget()
            .checkpoint_for(LeaseClass::Fallback)
            .map_err(Refusal::Budget)?;
        ny_core::require_f64_interval_proof_environment()
            .map_err(|e| Refusal::Build(e.to_string()))?;
        let at = [
            Some(self.0.deadline),
            requested,
            ny_core::instance_budget::deadline(),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap();
        if Instant::now() >= at {
            return Err(Refusal::Deadline);
        }
        Ok(at)
    }
    pub(super) fn tensor(&self, name: &str, shape: &[usize]) -> Result<&[f32]> {
        tensor(&self.0.weights, name, shape)
    }
}

fn tensor<'a>(weights: &'a DecoderWeights, name: &str, shape: &[usize]) -> Result<&'a [f32]> {
    let array = weights
        .weights
        .get(name)
        .ok_or_else(|| Refusal::Tensor(format!("missing original {name}")))?;
    if array.shape() != shape {
        return Err(Refusal::Tensor(format!("original shape differs {name}")));
    }
    array
        .as_slice()
        .ok_or_else(|| Refusal::Tensor(format!("noncontiguous original {name}")))
}

fn validate(weights: &DecoderWeights, p: &DecoderParameters, at: Instant) -> Result<()> {
    if p.architecture != DecoderArchitecture::GptNeoX
        || !p.parallel_residual
        || !p.attention_bias
        || p.tie_word_embeddings
        || p.query_heads != p.key_value_heads
        || !matches!(p.rotary_dim, 2 | 4 | 8)
        || p.rope_base != 10000.0
        || p.norm_epsilon != 0.00001f32
    {
        return Err(Refusal::Unsupported(
            "whole decoder requires checked NeoX parallel/GELU/analytic base10000 profile".into(),
        ));
    }
    let h = p.hidden_size;
    let i = p.intermediate_size;
    tensor(weights, "gpt_neox.embed_in.weight", &[p.vocab_size, h])?;
    tensor(weights, "embed_out.weight", &[p.vocab_size, h])?;
    for name in [
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    ] {
        tensor(weights, name, &[h])?;
    }
    for n in 0..p.num_layers {
        weights
            .budget
            .checkpoint_for(LeaseClass::Fallback)
            .map_err(Refusal::Budget)?;
        if Instant::now() >= at {
            return Err(Refusal::Deadline);
        }
        let prefix = format!("gpt_neox.layers.{n}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            tensor(weights, &format!("{prefix}.{norm}.weight"), &[h])?;
            tensor(weights, &format!("{prefix}.{norm}.bias"), &[h])?;
        }
        for (suffix, o, inner) in [
            ("attention.query_key_value", mul(3, h)?, h),
            ("attention.dense", h, h),
            ("mlp.dense_h_to_4h", i, h),
            ("mlp.dense_4h_to_h", h, i),
        ] {
            tensor(weights, &format!("{prefix}.{suffix}.weight"), &[o, inner])?;
            tensor(weights, &format!("{prefix}.{suffix}.bias"), &[o])?;
        }
    }
    if weights.weights.len() != add(mul(12, p.num_layers)?, 4)? {
        return Err(Refusal::Tensor(
            "extra original tensors outside closed full decoder".into(),
        ));
    }
    Ok(())
}

/// All six files are read from this directory and checked against the pinned
/// registry. The checkpoint digest is checked on the SAME held decode bytes.
pub fn load_pretrained_pythia14m(
    directory: impl AsRef<Path>,
    options: &DecoderOptions,
) -> Result<PretrainedDecoder> {
    if directory.as_ref().as_os_str().len() > 4096 {
        return Err(Refusal::Resource("pretrained asset directory length"));
    }
    let budget = options.budget.as_ref().ok_or(Refusal::Resource(
        "whole pretrained owner requires original ledger",
    ))?;
    let header_request = WorkRequest {
        scratch_bytes: 16384,
        persistent_bytes: 16384,
        work: 128 * 1024,
        proposals: 0,
    };
    super::admit(options, &header_request)?;
    let mut admission = budget
        .lease_for(LeaseClass::Fallback, header_request)
        .map_err(Refusal::Budget)?;
    // The unchanged closed JSON parser is an Optional phase. Preserve its
    // actual inherited cutoff in this new owner instead of renewing it at
    // the later Fallback decoder/operator phase.
    let parser_clock = budget
        .lease(WorkRequest::default())
        .map_err(Refusal::Budget)?;
    let at = [
        admission.deadline(),
        parser_clock.deadline(),
        options.deadline,
        ny_core::instance_budget::deadline(),
    ]
    .into_iter()
    .flatten()
    .min()
    .ok_or(Refusal::Resource(
        "whole pretrained owner requires finite original clock",
    ))?;
    drop(parser_clock);
    let local = DecoderOptions {
        deadline: Some(at),
        ..options.clone()
    };
    let directory = directory.as_ref();
    let mut config = None;
    for (name, expected) in ASSETS {
        if name == "model.safetensors" {
            continue;
        }
        admission.checkpoint().map_err(Refusal::Budget)?;
        super::checkpoint(&local)?;
        let mut file =
            std::fs::File::open(directory.join(name)).map_err(|e| Refusal::Io(e.to_string()))?;
        let n = usize::try_from(
            file.metadata()
                .map_err(|e| Refusal::Io(e.to_string()))?
                .len(),
        )
        .map_err(|_| Refusal::Resource("pretrained metadata length"))?;
        if n > options.max_file_bytes || n > 4 * 1024 * 1024 {
            return Err(Refusal::Resource("pretrained metadata cap"));
        }
        let request = WorkRequest {
            scratch_bytes: add(n, 4096)?,
            persistent_bytes: 0,
            work: (n as u64)
                .checked_mul(64)
                .ok_or(Refusal::Resource("asset hash work"))?,
            proposals: 0,
        };
        super::admit(&local, &request)?;
        if add(header_request.scratch_bytes, request.scratch_bytes)? > options.max_scratch_bytes {
            return Err(Refusal::Resource(
                "pretrained metadata complete scratch peak",
            ));
        }
        let mut lease = budget
            .lease_for(LeaseClass::Fallback, request)
            .map_err(Refusal::Budget)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(n)
            .map_err(|_| Refusal::Resource("asset bytes allocation"))?;
        if bytes.capacity() > n {
            return Err(Refusal::Resource("asset byte capacity"));
        }
        while bytes.len() < n {
            lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(&local)?;
            bytes.resize((bytes.len() + 4096).min(n), 0);
        }
        for block in bytes.chunks_mut(4096) {
            lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(&local)?;
            file.read_exact(block)
                .map_err(|e| Refusal::Io(e.to_string()))?;
        }
        let mut extra = [0];
        if file
            .read(&mut extra)
            .map_err(|e| Refusal::Io(e.to_string()))?
            != 0
        {
            return Err(Refusal::Io("pretrained metadata changed".into()));
        }
        let mut hash = Sha256::new();
        for block in bytes.chunks(4096) {
            lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(&local)?;
            hash.update(block);
        }
        let actual: [u8; 32] = hash.finalize().into();
        if actual != digest(expected) {
            return Err(Refusal::Tensor(format!(
                "pinned original asset differs {name}"
            )));
        }
        if name == "config.json" {
            let parse_options = DecoderOptions {
                max_scratch_bytes: options
                    .max_scratch_bytes
                    .checked_sub(add(header_request.scratch_bytes, request.scratch_bytes)?)
                    .ok_or(Refusal::Resource("config and asset scratch overlap"))?,
                ..local.clone()
            };
            config = Some(DecoderContract::from_json(&bytes, &parse_options)?);
        }
        drop(bytes);
        lease.complete().map_err(Refusal::Budget)?;
    }
    // Finish the bounded metadata walk before checkpoint decode, releasing
    // its scratch while retaining actual config/manifest/header ownership.
    admission.complete().map_err(Refusal::Budget)?;
    let resident = admission.commit_persistent().map_err(Refusal::Budget)?;
    let decode_options = DecoderOptions {
        max_resident_bytes: options
            .max_resident_bytes
            .checked_sub(resident.bytes())
            .ok_or(Refusal::Resource(
                "pretrained source and header resident cap",
            ))?,
        ..local
    };
    let weights = super::load::load_pinned_decoder_safetensors(
        &directory.join("model.safetensors"),
        digest(ASSETS[1].1),
        &decode_options,
    )?;
    if !weights.budget.shares_ledger(budget)
        || weights.pinned_file_digest != Some(digest(ASSETS[1].1))
    {
        return Err(Refusal::Resource("pretrained source ledger/digest differs"));
    }
    let parameters = config
        .ok_or_else(|| Refusal::Config("missing pinned config".into()))?
        .into_parameters();
    if (
        parameters.hidden_size,
        parameters.intermediate_size,
        parameters.vocab_size,
        parameters.num_layers,
        parameters.query_heads,
        parameters.head_dim,
        parameters.rotary_dim,
        parameters.max_positions,
    ) != (128, 512, 50304, 6, 4, 32, 8, 2048)
    {
        return Err(Refusal::Config("pinned LLM-A architecture differs".into()));
    }
    let mut seal = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: 16384,
                persistent_bytes: 0,
                work: 128 * 1024,
                proposals: 0,
            },
        )
        .map_err(Refusal::Budget)?;
    validate(&weights, &parameters, at)?;
    let manifest=PretrainedDecoderManifest { repository:"EleutherAI/pythia-14m",revision:PYTHIA14M_REVISION,
        operator_reference:"https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py",
        tokenizer_policy:"pinned GPTNeoXTokenizer/NFC/ByteLevel regex/no prefix space/identity single-template/endoftext BOS=EOS=UNK0/no automatic BOS/no padding or truncation; API accepts fixed IDs, not text",
        numeric_profile:PYTHIA_EXACT_REAL_PROFILE,assets:ASSETS.map(|(name,sha)|PretrainedAssetIdentity{name,sha256:digest(sha)}),
        bos_token:0,eos_token:0,input_policy:"fixed supplied vocabulary IDs/no implicit BOS/no truncation/positions0..S-1/full causal mask/strict greedy H1" };
    let result = PretrainedDecoder(Arc::new(PretrainedInner {
        weights,
        parameters,
        manifest,
        deadline: at,
        _resident: resident,
    }));
    result.checkpoint(None)?;
    seal.complete().map_err(Refusal::Budget)?;
    Ok(result)
}

#[cfg(test)]
pub(super) fn authored_fixture(
    weights: DecoderWeights,
    parameters: DecoderParameters,
    at: Instant,
) -> Result<PretrainedDecoder> {
    let mut lease = weights
        .budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: 16384,
                persistent_bytes: 16384,
                work: 128 * 1024,
                proposals: 0,
            },
        )
        .map_err(Refusal::Budget)?;
    let at = lease.deadline().map_or(at, |d| d.min(at));
    validate(&weights, &parameters, at)?;
    let manifest = PretrainedDecoderManifest {
        repository: "authored-test-fixture",
        revision: "authored",
        numeric_profile: PYTHIA_EXACT_REAL_PROFILE,
        operator_reference: "authored-test-original-NeoX-operators",
        tokenizer_policy: "authored fixed IDs; no text claim",
        assets: ASSETS.map(|(name, _)| PretrainedAssetIdentity {
            name,
            sha256: [0; 32],
        }),
        bos_token: 0,
        eos_token: 0,
        input_policy: "authored fixed-token original model",
    };
    lease.complete().map_err(Refusal::Budget)?;
    let resident = lease.commit_persistent().map_err(Refusal::Budget)?;
    Ok(PretrainedDecoder(Arc::new(PretrainedInner {
        weights,
        parameters,
        manifest,
        deadline: at,
        _resident: resident,
    })))
}
