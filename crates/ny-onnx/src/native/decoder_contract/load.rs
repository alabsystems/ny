// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Admitted, finite coefficient loading with original hashes fused into decode.
use super::manifest::tensor_hash_start;
use super::{add, mul, DecoderContractRefusal as Refusal, DecoderOptions, Result, TensorIdentity};
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use safetensors::{Dtype, SafeTensors};
use sha2::Digest as _;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

const DECODED_HASH_BLOCK_COEFFICIENTS: usize = 4096;
const DECODED_HASH_BLOCK_BYTES: usize = DECODED_HASH_BLOCK_COEFFICIENTS * 4;

#[cfg(test)]
mod block_hash_tests;

/// Owned original decoded coefficients and their checked content identities.
/// Fields cannot be mutated independently of their manifest or memory lease.
#[derive(Debug)]
pub struct DecoderWeights {
    pub(super) weights: WeightStore,
    pub(super) identities: BTreeMap<String, TensorIdentity>,
    pub(super) source_bytes: u64,
    pub(super) coefficients: u64,
    pub(super) resident_bytes: usize,
    pub(super) budget: WorkBudget,
    pub(super) reservation: PersistentReservation,
    /// Present only after hashing and checking the same held serialized file.
    pub(super) pinned_file_digest: Option<[u8; 32]>,
}
impl DecoderWeights {
    pub fn identities(&self) -> impl Iterator<Item = &TensorIdentity> {
        self.identities.values()
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    /// Explicit binary32 coefficient input for authored fixtures or producers.
    /// No original F16/BF16 serialization identity is fabricated.
    pub fn from_store(weights: WeightStore, options: &DecoderOptions) -> Result<Self> {
        let budget = super::budget(options)?;
        let coefficients = weights
            .iter()
            .try_fold(0usize, |n, (_, a)| add(n, a.len()))?;
        let metadata = weights.iter().try_fold(0usize, |n, (key, array)| {
            add(
                n,
                add(add(1024, mul(key.len(), 4)?)?, mul(array.ndim(), 32)?)?,
            )
        })?;
        let resident_bytes = add(mul(coefficients, 4)?, metadata)?;
        if resident_bytes > options.max_resident_bytes
            || coefficients as u64 > options.max_terms
            || options.max_scratch_bytes < 4096
        {
            return Err(Refusal::Resource("decoded coefficient admission"));
        }
        let mut lease = budget
            .lease(WorkRequest {
                scratch_bytes: 4096,
                persistent_bytes: resident_bytes,
                work: coefficients as u64,
                proposals: 1,
            })
            .map_err(Refusal::Budget)?;
        let mut identities = BTreeMap::new();
        let mut names: Vec<_> = weights.keys().collect();
        names.sort_unstable();
        for name in names {
            lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            let a = weights
                .get(name)
                .ok_or_else(|| Refusal::Tensor("coefficient disappeared".into()))?;
            let mut hash = tensor_hash_start(name, a.shape(), "F32-decoded-input");
            for (index, &v) in a.iter().enumerate() {
                if index % 4096 == 0 {
                    lease.checkpoint().map_err(Refusal::Budget)?;
                    super::checkpoint(options)?;
                }
                if !v.is_finite() {
                    return Err(Refusal::Tensor(format!("nonfinite coefficient {name}")));
                }
                hash.update(v.to_bits().to_le_bytes());
            }
            let digest = hash.finalize().into();
            identities.insert(
                name.to_string(),
                TensorIdentity {
                    name: name.to_string(),
                    shape: a.shape().to_vec(),
                    source_dtype: "F32-decoded-input".into(),
                    source_digest: digest,
                    decoded_f32_digest: digest,
                },
            );
        }
        lease.complete().map_err(Refusal::Budget)?;
        let reservation = lease.commit_persistent().map_err(Refusal::Budget)?;
        Ok(Self {
            weights,
            identities,
            source_bytes: (coefficients as u64) * 4,
            coefficients: coefficients as u64,
            resident_bytes,
            budget,
            reservation,
            pinned_file_digest: None,
        })
    }
}

/// This narrow authoritative path accepts a single SafeTensors checkpoint.
/// F16 and BF16 decode exactly to F32; F32 remains bit-identical. Quantized,
/// F64, sharded and guessed GGUF formats need separate explicit contracts.
pub fn load_decoder_safetensors(
    path: impl AsRef<Path>,
    options: &DecoderOptions,
) -> Result<DecoderWeights> {
    load_safetensors_inner(path.as_ref(), None, options)
}

pub(super) fn load_pinned_decoder_safetensors(
    path: &Path,
    expected: [u8; 32],
    options: &DecoderOptions,
) -> Result<DecoderWeights> {
    load_safetensors_inner(path, Some(expected), options)
}

fn load_safetensors_inner(
    path: &Path,
    expected: Option<[u8; 32]>,
    options: &DecoderOptions,
) -> Result<DecoderWeights> {
    let budget = super::budget(options)?;
    let mut file = std::fs::File::open(path).map_err(|e| Refusal::Io(e.to_string()))?;
    let file_bytes = usize::try_from(
        file.metadata()
            .map_err(|e| Refusal::Io(e.to_string()))?
            .len(),
    )
    .map_err(|_| Refusal::Resource("checkpoint size"))?;
    if file_bytes > options.max_file_bytes || file_bytes < 8 {
        return Err(Refusal::Resource("checkpoint source byte cap"));
    }
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| Refusal::Io(e.to_string()))?;
    let header_bytes = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| Refusal::Resource("checkpoint header size"))?;
    if header_bytes > options.max_config_bytes || header_bytes > file_bytes - 8 {
        return Err(Refusal::Resource("checkpoint header byte cap"));
    }
    // SafeTensors also materializes header strings/shapes; the raw source
    // buffer alone is not the parser's complete live scratch allocation.
    // The decoded-byte block overlaps the original source/parser and every
    // decode phase. Its bounded payload is leased before either proportional
    // allocation; the existing 4096-byte bookkeeping includes its Vec header.
    let source_scratch = add(
        add(file_bytes, add(mul(header_bytes, 16)?, 4096)?)?,
        DECODED_HASH_BLOCK_BYTES,
    )?;
    if source_scratch > options.max_scratch_bytes {
        return Err(Refusal::Resource("checkpoint parser scratch cap"));
    }
    let source_request = WorkRequest {
        scratch_bytes: source_scratch,
        persistent_bytes: 0,
        // Price the one bounded buffer initialization separately from the
        // unchanged source-byte and decoded-coordinate visits.
        work: (file_bytes as u64)
            .checked_add(DECODED_HASH_BLOCK_BYTES as u64)
            .and_then(|n| {
                n.checked_add(if expected.is_some() {
                    (file_bytes as u64).checked_mul(64)?
                } else {
                    0
                })
            })
            .ok_or(Refusal::Resource("checkpoint source work count"))?,
        proposals: 1,
    };
    if expected.is_some() {
        super::admit(options, &source_request)?;
    }
    let class = if expected.is_some() {
        LeaseClass::Fallback
    } else {
        LeaseClass::Optional
    };
    let mut source_lease = budget
        .lease_for(class, source_request)
        .map_err(Refusal::Budget)?;
    let mut decoded_bytes = Vec::new();
    decoded_bytes
        .try_reserve_exact(DECODED_HASH_BLOCK_BYTES)
        .map_err(|_| Refusal::Resource("decoded hash block allocation"))?;
    if decoded_bytes.capacity() > DECODED_HASH_BLOCK_BYTES {
        return Err(Refusal::Resource("decoded hash block capacity"));
    }
    decoded_bytes.resize(DECODED_HASH_BLOCK_BYTES, 0u8);
    source_lease.checkpoint().map_err(Refusal::Budget)?;
    super::checkpoint(options)?;
    let mut bytes = if expected.is_some() {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_bytes)
            .map_err(|_| Refusal::Resource("pinned source allocation"))?;
        if bytes.capacity() > file_bytes {
            return Err(Refusal::Resource("pinned source actual capacity"));
        }
        while bytes.len() < file_bytes {
            source_lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            bytes.resize((bytes.len() + 4096).min(file_bytes), 0u8);
        }
        bytes
    } else {
        vec![0u8; file_bytes]
    };
    if expected.is_some() && bytes.capacity() > file_bytes {
        return Err(Refusal::Resource("pinned source actual capacity"));
    }
    bytes[..8].copy_from_slice(&prefix);
    if expected.is_some() {
        for block in bytes[8..].chunks_mut(4096) {
            source_lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            file.read_exact(block)
                .map_err(|e| Refusal::Io(e.to_string()))?;
        }
    } else {
        file.read_exact(&mut bytes[8..])
            .map_err(|e| Refusal::Io(e.to_string()))?;
    }
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|e| Refusal::Io(e.to_string()))?
        != 0
    {
        return Err(Refusal::Io("checkpoint changed during load".into()));
    }
    source_lease.checkpoint().map_err(Refusal::Budget)?;
    super::checkpoint(options)?;
    let pinned_file_digest = if let Some(expected) = expected {
        let mut hash = sha2::Sha256::new();
        for block in bytes.chunks(4096) {
            source_lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            hash.update(block);
        }
        let actual: [u8; 32] = hash.finalize().into();
        if actual != expected {
            return Err(Refusal::Tensor(
                "pinned checkpoint file digest differs".into(),
            ));
        }
        Some(actual)
    } else {
        None
    };
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|e| Refusal::Tensor(format!("SafeTensors: {e}")))?;
    let mut names = tensors.names();
    names.sort_unstable();
    let mut coefficients = 0usize;
    let mut metadata = 0usize;
    for name in &names {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| Refusal::Tensor(e.to_string()))?;
        let count = tensor.shape().iter().try_fold(1usize, |n, &d| mul(n, d))?;
        let width = match tensor.dtype() {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
            dtype => return Err(Refusal::Unsupported(format!("tensor {name}: {dtype:?}"))),
        };
        if tensor.data().len() != mul(count, width)? {
            return Err(Refusal::Tensor(format!("byte shape {name}")));
        }
        coefficients = add(coefficients, count)?;
        metadata = add(
            metadata,
            add(
                add(1024, mul(name.len(), 4)?)?,
                mul(tensor.shape().len(), 32)?,
            )?,
        )?;
    }
    let resident_bytes = add(mul(coefficients, 4)?, metadata)?;
    if resident_bytes > options.max_resident_bytes || coefficients as u64 > options.max_terms {
        return Err(Refusal::Resource("decoded checkpoint cap"));
    }
    let mut decoded_lease = budget
        .lease_for(
            class,
            WorkRequest {
                scratch_bytes: 0,
                persistent_bytes: resident_bytes,
                work: coefficients as u64,
                proposals: 1,
            },
        )
        .map_err(Refusal::Budget)?;
    let mut weights = WeightStore::new();
    let mut identities = BTreeMap::new();
    let mut source_bytes = 0u64;
    for name in names {
        decoded_lease.checkpoint().map_err(Refusal::Budget)?;
        super::checkpoint(options)?;
        let tensor = tensors
            .tensor(name)
            .map_err(|e| Refusal::Tensor(e.to_string()))?;
        let dtype = format!("{:?}", tensor.dtype());
        let mut original_hash = tensor_hash_start(name, tensor.shape(), &dtype);
        let mut decoded_hash = tensor_hash_start(name, tensor.shape(), "F32-decoded");
        let width = if tensor.dtype() == Dtype::F32 { 4 } else { 2 };
        let mut values = Vec::with_capacity(tensor.data().len() / width);
        if expected.is_some() && values.capacity() > tensor.data().len() / width {
            return Err(Refusal::Resource("pinned decoded actual capacity"));
        }
        // Hash original bytes and decoded bits in the SAME coefficient pass.
        for block in tensor
            .data()
            .chunks(width * DECODED_HASH_BLOCK_COEFFICIENTS)
        {
            decoded_lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
            original_hash.update(block);
            for (index, raw) in block.chunks_exact(width).enumerate() {
                let value = match tensor.dtype() {
                    Dtype::F16 => {
                        half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32()
                    }
                    Dtype::BF16 => {
                        half::bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32()
                    }
                    Dtype::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    _ => unreachable!("dtype admitted"),
                };
                if !value.is_finite() {
                    return Err(Refusal::Tensor(format!("nonfinite {name}")));
                }
                // Identical ascending-coordinate little-endian F32 bit stream
                // as the scalar updates. Hash it once per bounded block, with
                // no second tensor pass or full decoded-byte allocation.
                let start = index * 4;
                decoded_bytes[start..start + 4].copy_from_slice(&value.to_bits().to_le_bytes());
                values.push(value);
            }
            decoded_hash.update(&decoded_bytes[..block.len() / width * 4]);
            #[cfg(test)]
            block_hash_tests::after_decoded_block();
            decoded_lease.checkpoint().map_err(Refusal::Budget)?;
            super::checkpoint(options)?;
        }
        source_bytes = source_bytes
            .checked_add(tensor.data().len() as u64)
            .ok_or(Refusal::Resource("source hash count"))?;
        let array = ArrayD::from_shape_vec(IxDyn(tensor.shape()), values)
            .map_err(|e| Refusal::Tensor(e.to_string()))?;
        identities.insert(
            name.to_string(),
            TensorIdentity {
                name: name.to_string(),
                shape: tensor.shape().to_vec(),
                source_dtype: dtype,
                source_digest: original_hash.finalize().into(),
                decoded_f32_digest: decoded_hash.finalize().into(),
            },
        );
        weights.insert(name.to_string(), array);
    }
    drop(tensors);
    drop(bytes);
    drop(decoded_bytes);
    source_lease.complete().map_err(Refusal::Budget)?;
    drop(source_lease);
    decoded_lease.complete().map_err(Refusal::Budget)?;
    let reservation = decoded_lease.commit_persistent().map_err(Refusal::Budget)?;
    super::checkpoint(options)?;
    Ok(DecoderWeights {
        weights,
        identities,
        source_bytes,
        coefficients: coefficients as u64,
        resident_bytes,
        budget,
        reservation,
        pinned_file_digest,
    })
}
