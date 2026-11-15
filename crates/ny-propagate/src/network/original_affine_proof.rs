// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checked local affine bounds over an immutable original coefficient owner.
//!
//! This API qualifies ONE mathematical-real affine map with original binary32
//! coefficient bits. It does not qualify diagnostic point execution, a graph,
//! a decoder trajectory, supplied upstream enclosures, or vocabulary coverage.
//! Every phase spends the original owner's ledger; outputs retain their lease
//! and coefficient owner across report clones. Scope labels identify a caller's
//! coordinates, but cannot create whole-network arithmetic authority.

mod prepare;
mod report;
#[cfg(test)]
mod tests;

pub use prepare::PreparedOriginalAffine;
pub use report::{OriginalAffineIntervalReport, OriginalAffineLinearReport};

use std::time::Instant;

use ndarray::{ArrayBase, Data, Dimension};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use sha2::{Digest as _, Sha256};

use crate::certify::scope::{Digest, ScopeIdentity};
use crate::certify::work::{LeaseClass, WorkBudget, WorkRequest};
use crate::layers::linear::original_affine::OriginalAffinePhase;
use crate::LinearBounds;

/// The closed arithmetic profile proved by this adapter. The coefficients are
/// the exact real values of original finite binary32 bits, including subnormals.
/// No deployed floating point execution or Torch kernel equivalence is claimed.
pub const ORIGINAL_AFFINE_EXACT_REAL_PROFILE: &str =
    "exact-real-original-binary32-affine/f64-outward-with-stored-error/v1";

/// Caller coordinate/lowering labels, bound to the actual immutable owner.
/// They are necessary scope identifiers, never independent arithmetic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginalAffineIds {
    pub lowering: Digest,
    pub node: Digest,
    pub input: Digest,
    pub output: Digest,
}

/// A caller's existing phase authority. All three classes spend the SAME
/// owner's cumulative counters; this API never constructs a nested budget.
#[derive(Debug, Clone, Copy)]
pub struct OriginalAffineOptions<'a> {
    pub budget: &'a WorkBudget,
    pub class: LeaseClass,
    pub deadline: Option<Instant>,
}

/// One explicitly demanded positive target-minus-competitor logit direction.
/// A list of these rows is not evidence of complete vocabulary coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginalLogitDifference {
    pub target: usize,
    pub competitor: usize,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginalAffineKernel {
    Interval,
    SignedLinear,
    SparseLogitDifferences,
}

const METADATA_BYTES: usize = 2048;
const HASH_POLL: usize = 4096;

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("original-affine context: {message}"))
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

fn extend_request(mut request: WorkRequest, hash_terms: usize) -> Result<WorkRequest> {
    request.persistent_bytes = add(request.persistent_bytes, METADATA_BYTES)?;
    request.work = request
        .work
        .checked_add(as_work(add(hash_terms, 256)?)?)
        .ok_or_else(|| invalid("phase work overflow"))?;
    request.proposals = request
        .proposals
        .checked_add(1)
        .ok_or_else(|| invalid("phase proposal overflow"))?;
    Ok(request)
}

fn hash_word(hash: &mut Sha256, value: usize, phase: &mut OriginalAffinePhase) -> Result<()> {
    phase.charge_work(1)?;
    hash.update(
        u64::try_from(value)
            .map_err(|_| invalid("scope dimension overflow"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_array<S: Data<Elem = f32>, D: Dimension>(
    hash: &mut Sha256,
    array: &ArrayBase<S, D>,
    phase: &mut OriginalAffinePhase,
) -> Result<()> {
    hash_word(hash, array.ndim(), phase)?;
    for &dim in array.shape() {
        hash_word(hash, dim, phase)?;
    }
    for (index, value) in array.iter().enumerate() {
        if index.is_multiple_of(HASH_POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(1)?;
        hash.update(value.to_bits().to_le_bytes());
    }
    phase.checkpoint()
}

fn interval_terms(bounds: &BoundedTensor) -> Result<usize> {
    if bounds.lower().shape() != bounds.upper().shape() {
        return Err(invalid("interval endpoint shape mismatch"));
    }
    add(
        mul(bounds.lower().len(), 2)?,
        mul(add(bounds.shape().len(), 1)?, 2)?,
    )
}
fn hash_interval(bounds: &BoundedTensor, phase: &mut OriginalAffinePhase) -> Result<Digest> {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-interval-input-output/v1");
    hash_array(&mut hash, bounds.lower(), phase)?;
    hash_array(&mut hash, bounds.upper(), phase)?;
    Ok(Digest(hash.finalize().into()))
}

fn linear_shape(bounds: &LinearBounds) -> Result<(usize, usize)> {
    let (rows, cols) = bounds.lower_a().dim();
    if bounds.upper_a().dim() != (rows, cols)
        || bounds.lower_b().len() != rows
        || bounds.upper_b().len() != rows
        || bounds
            .lower_a_err()
            .is_some_and(|e| e.dim() != (rows, cols))
        || bounds
            .upper_a_err()
            .is_some_and(|e| e.dim() != (rows, cols))
    {
        return Err(invalid("incoming carrier shape"));
    }
    Ok((rows, cols))
}

fn linear_terms(bounds: &LinearBounds) -> Result<usize> {
    linear_shape(bounds)?;
    let arrays = add(
        add(bounds.lower_a().len(), bounds.upper_a().len())?,
        add(bounds.lower_b().len(), bounds.upper_b().len())?,
    )?;
    let errors = add(
        bounds.lower_a_err().map_or(0, |e| e.len()),
        bounds.upper_a_err().map_or(0, |e| e.len()),
    )?;
    add(add(arrays, errors)?, 20)
}

fn hash_linear(bounds: &LinearBounds, phase: &mut OriginalAffinePhase) -> Result<Digest> {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-signed-carrier/v1");
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

fn output_linear_terms(rows: usize, cols: usize) -> Result<usize> {
    add(add(mul(mul(rows, cols)?, 4)?, mul(rows, 2)?)?, 20)
}

/// A report's exact local request/output binding. This contains no region
/// widening/reuse authority and is not an UNSAT or complete-network token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginalAffineBinding {
    pub kernel: OriginalAffineKernel,
    pub query: Digest,
    pub output: Digest,
}

fn query_digest(scope: &ScopeIdentity, kernel: OriginalAffineKernel, source: Digest) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-affine-local-query/v1");
    for digest in [
        scope.model,
        scope.lowering,
        scope.coordinates,
        scope.assumptions,
        source,
    ] {
        hash.update(digest.0);
    }
    hash.update([match kernel {
        OriginalAffineKernel::Interval => 0,
        OriginalAffineKernel::SignedLinear => 1,
        OriginalAffineKernel::SparseLogitDifferences => 2,
    }]);
    Digest(hash.finalize().into())
}
