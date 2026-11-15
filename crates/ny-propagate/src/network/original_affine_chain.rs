// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! A sealed local exact-real proof for an ordered original affine/activation chain.
//!
//! This explicit adapter does not change generic `PointLinear` eligibility.
//! It accepts immutable original affine owners, mathematical ReLU, and exact
//! mathematical GELU-erf or SiLU through a separately checked scalar adapter,
//! with one connected input/output edge per node. Supplied input boxes remain
//! caller assumptions. No graph, decoder, deployed kernel, UNSAT or vocabulary
//! coverage authority is minted by these local reports.

mod activation;
mod activation_stage;
mod prepare;
mod relu;
mod report;
#[cfg(test)]
mod tests;

pub use prepare::PreparedOriginalAffineChain;
pub use report::{OriginalChainForwardReport, OriginalChainLinearReport};

use std::sync::Arc;
use std::time::Instant;

use ndarray::{ArrayBase, Data, Dimension};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use sha2::{Digest as _, Sha256};

use crate::certify::scope::{Digest, ScopeIdentity};
use crate::certify::work::{PersistentReservation, RetainedOwner, WorkRequest};
use crate::layers::linear::original_affine::OriginalAffinePhase;
use crate::layers::{Layer, PointLinearLayer};
use crate::network::original_affine_proof::{
    OriginalAffineIds, OriginalAffineIntervalReport, OriginalAffineLinearReport,
    OriginalAffineOptions, OriginalLogitDifference, PreparedOriginalAffine,
};
use crate::LinearBounds;

pub const ORIGINAL_CHAIN_EXACT_REAL_PROFILE: &str =
    "exact-real-original-binary32-affine/relu-outward-error-carrier/v1";
pub use activation::decoder_scalar::{CheckedDecoderMath, DecoderInterval, DecoderMathCounts};
pub const ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE: &str =
    "exact-real-original-binary32-affine/relu/gelu-erf-outward-error-carrier/v1";
pub const ORIGINAL_CHAIN_SILU_EXACT_REAL_PROFILE: &str =
    "exact-real-original-binary32-affine/relu/silu-outward-error-carrier/v1";
pub const ORIGINAL_CHAIN_GELU_SILU_EXACT_REAL_PROFILE: &str =
    "exact-real-original-binary32-affine/relu/gelu-erf/silu-outward-error-carrier/v1";
const MAX_NODES: usize = 128;
const MAX_ELEMENTS: usize = 4 * 1024 * 1024;
const META: usize = 16 * 1024;
const POLL: usize = 1024;

/// A unary node in the declared order. Every adjacent output/input label must
/// match. A supplied DAG branch or unsupported layer is refused, not flattened.
#[derive(Clone, Copy)]
pub struct OriginalChainNode<'a> {
    pub ids: OriginalAffineIds,
    pub layer: &'a Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginalChainKernel {
    Interval,
    SignedLinear,
    SparseLogitDifferences,
}

/// Exact local input/demand/output binding, without widening/reuse authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginalChainBinding {
    pub kernel: OriginalChainKernel,
    pub query: Digest,
    pub output: Digest,
}

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("original-affine chain: {message}"))
}
fn numeric(message: &str) -> NyError {
    NyError::NumericalInstability(format!("original-affine chain: {message}"))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("size overflow"))
}
fn as_work(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid("work overflow"))
}
fn earlier(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
fn cap(n: usize) -> Result<usize> {
    if n > MAX_ELEMENTS {
        Err(NyError::UnsupportedConfiguration(
            "original-affine chain exceeds the 4 Mi element profile".into(),
        ))
    } else {
        Ok(n)
    }
}

fn hash_word(hash: &mut Sha256, value: usize, phase: &mut OriginalAffinePhase) -> Result<()> {
    phase.charge_work(1)?;
    hash.update(as_work(value)?.to_le_bytes());
    Ok(())
}
fn hash_array<S: Data<Elem = f32>, D: Dimension>(
    hash: &mut Sha256,
    array: &ArrayBase<S, D>,
    phase: &mut OriginalAffinePhase,
) -> Result<()> {
    hash_word(hash, array.ndim(), phase)?;
    for &n in array.shape() {
        hash_word(hash, n, phase)?;
    }
    for (i, value) in array.iter().enumerate() {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(1)?;
        hash.update(value.to_bits().to_le_bytes());
    }
    phase.checkpoint()
}
fn interval_terms(boxes: &BoundedTensor) -> Result<usize> {
    if boxes.lower().shape() != boxes.upper().shape() {
        return Err(invalid("endpoint shape mismatch"));
    }
    add(mul(boxes.len(), 2)?, mul(add(boxes.shape().len(), 1)?, 2)?)
}
fn hash_interval(boxes: &BoundedTensor, phase: &mut OriginalAffinePhase) -> Result<Digest> {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-chain-box/v1");
    hash_array(&mut hash, boxes.lower(), phase)?;
    hash_array(&mut hash, boxes.upper(), phase)?;
    Ok(Digest(hash.finalize().into()))
}
fn linear_shape(bounds: &LinearBounds) -> Result<(usize, usize)> {
    let shape = bounds.lower_a().dim();
    if bounds.upper_a().dim() != shape
        || bounds.lower_b().len() != shape.0
        || bounds.upper_b().len() != shape.0
        || bounds.lower_a_err().is_some_and(|e| e.dim() != shape)
        || bounds.upper_a_err().is_some_and(|e| e.dim() != shape)
    {
        return Err(invalid("row/error/bias shape mismatch"));
    }
    cap(mul(shape.0, shape.1)?)?;
    Ok(shape)
}
fn linear_terms(bounds: &LinearBounds) -> Result<usize> {
    linear_shape(bounds)?;
    let mut terms = add(
        mul(bounds.lower_a().len(), 2)?,
        mul(bounds.lower_b().len(), 2)?,
    )?;
    for error in [bounds.lower_a_err(), bounds.upper_a_err()] {
        terms = add(terms, error.map_or(0, |e| e.len()))?;
    }
    add(terms, 20)
}
fn hash_linear(bounds: &LinearBounds, phase: &mut OriginalAffinePhase) -> Result<Digest> {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-chain-signed-demand/v1");
    hash_array(&mut hash, bounds.lower_a(), phase)?;
    hash_array(&mut hash, bounds.upper_a(), phase)?;
    hash_array(&mut hash, bounds.lower_b(), phase)?;
    hash_array(&mut hash, bounds.upper_b(), phase)?;
    for error in [bounds.lower_a_err(), bounds.upper_a_err()] {
        hash_word(&mut hash, usize::from(error.is_some()), phase)?;
        if let Some(error) = error {
            hash_array(&mut hash, error, phase)?;
        }
    }
    Ok(Digest(hash.finalize().into()))
}
fn request(work: usize, persistent: usize) -> Result<WorkRequest> {
    Ok(WorkRequest {
        scratch_bytes: 32 * 1024,
        persistent_bytes: persistent,
        work: as_work(add(work, 1024)?)?,
        proposals: 1,
    })
}
fn query_digest(scope: &ScopeIdentity, kernel: OriginalChainKernel, sources: &[Digest]) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-chain-query/v1");
    for digest in [
        scope.model,
        scope.lowering,
        scope.coordinates,
        scope.assumptions,
    ] {
        hash.update(digest.0);
    }
    hash.update([match kernel {
        OriginalChainKernel::Interval => 0,
        OriginalChainKernel::SignedLinear => 1,
        OriginalChainKernel::SparseLogitDifferences => 2,
    }]);
    for source in sources {
        hash.update(source.0);
    }
    Digest(hash.finalize().into())
}

pub(super) struct ChainOwner {
    owners: Vec<PointLinearLayer>,
    scope: ScopeIdentity,
    deadline: Option<Instant>,
    preparation_metadata_bytes: usize,
    reservation: PersistentReservation,
}
pub(super) enum Stage<'a> {
    Affine(PreparedOriginalAffine<'a>),
    Relu,
    GeluErf,
    SiLu,
}
pub(super) struct OwnedBox(RetainedOwner<OwnedBoxData>);
pub(super) struct OwnedBoxData {
    bounds: BoundedTensor,
}
impl std::ops::Deref for OwnedBox {
    type Target = OwnedBoxData;

    fn deref(&self) -> &OwnedBoxData {
        &self.0
    }
}
pub(super) struct OwnedRows {
    bounds: LinearBounds,
    _reservation: PersistentReservation,
}
pub(super) enum Fact {
    Affine(OriginalAffineIntervalReport),
    Relu(OwnedBox),
    GeluErf(activation_stage::OwnedActivationFact),
    SiLu(activation_stage::OwnedActivationFact),
}
impl Fact {
    fn bounds(&self) -> &BoundedTensor {
        match self {
            Self::Affine(report) => report.bounds(),
            Self::Relu(report) => &report.bounds,
            Self::GeluErf(report) => report.bounds(),
            Self::SiLu(report) => report.bounds(),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Self::Affine(report) => report.reserved_bytes(),
            Self::Relu(report) => report.0.bytes(),
            Self::GeluErf(report) => report.reserved_bytes(),
            Self::SiLu(report) => report.reserved_bytes(),
        }
    }
}
enum Carrier {
    Affine(OriginalAffineLinearReport),
    Relu(OwnedRows),
    GeluErf(OwnedRows),
    SiLu(OwnedRows),
}
impl Carrier {
    fn bounds(&self) -> &LinearBounds {
        match self {
            Self::Affine(report) => report.bounds(),
            Self::Relu(report) => &report.bounds,
            Self::GeluErf(report) => &report.bounds,
            Self::SiLu(report) => &report.bounds,
        }
    }
}
