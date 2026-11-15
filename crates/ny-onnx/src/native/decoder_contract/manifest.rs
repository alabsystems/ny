// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Canonical operator and original-tensor content identities, not proof tokens.
use super::{DecoderContractRefusal as Refusal, DecoderParameters, Result};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorIdentity {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
    /// Original serialized tensor bytes (or explicitly decoded F32 input).
    pub source_digest: [u8; 32],
    pub decoded_f32_digest: [u8; 32],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationIdentity {
    pub name: String,
    pub operator: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub output_shape: Vec<i64>,
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub tensors: Vec<TensorIdentity>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BlockIdentity {
    pub name: String,
    pub operations: Vec<OperationIdentity>,
    pub digest: [u8; 32],
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoweringManifest {
    pub schema_version: u32,
    pub parameters: DecoderParameters,
    pub positions: Vec<u64>,
    pub original_tensors: Vec<TensorIdentity>,
    /// Embedding lookup is an explicit frontend boundary. No tokenizer or
    /// generation processor is silently incorporated into the lowered graph.
    pub input_boundary: String,
    pub blocks: Vec<BlockIdentity>,
    pub lowering_digest: [u8; 32],
    pub source_bytes_hashed: u64,
    pub decoded_coefficients_hashed: u64,
}
impl LoweringManifest {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Refusal::Build(format!("manifest JSON: {e}")))
    }
}
pub(super) fn digest_serialized<T: Serialize>(domain: &[u8], value: &T) -> Result<[u8; 32]> {
    // Manifests contain hashes/parameters, never proportional tensor data.
    let bytes =
        serde_json::to_vec(value).map_err(|e| Refusal::Build(format!("manifest hash: {e}")))?;
    let mut hash = Sha256::new();
    hash.update(b"ny-decoder-contract/v1");
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    Ok(hash.finalize().into())
}
pub(super) fn tensor_hash_start(name: &str, shape: &[usize], dtype: &str) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-decoder-tensor/v1");
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((dtype.len() as u64).to_le_bytes());
    hash.update(dtype.as_bytes());
    hash.update((shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hash.update((dim as u64).to_le_bytes());
    }
    hash
}
