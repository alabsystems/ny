// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::certify::work::PersistentReservation;

use super::prepare::PreparedOwner;
use super::*;

struct ReportOwner<T> {
    output: T,
    original: Arc<PreparedOwner>,
    binding: OriginalAffineBinding,
    reservation: PersistentReservation,
}

fn finish<T>(
    output: T,
    original: Arc<PreparedOwner>,
    binding: OriginalAffineBinding,
    mut phase: OriginalAffinePhase,
) -> Result<Arc<ReportOwner<T>>> {
    if add(size_of::<ReportOwner<T>>(), 4 * size_of::<usize>())? > METADATA_BYTES {
        return Err(invalid("report owner metadata capacity"));
    }
    phase.retain_metadata(METADATA_BYTES)?;
    let reservation = phase.commit()?;
    Ok(Arc::new(ReportOwner {
        output,
        original,
        binding,
        reservation,
    }))
}

/// Immutable checked local interval result. Clones retain one actual output,
/// original coefficient owner, prepared scope and reservation until last drop.
/// Borrowed array access does not admit caller-created copies or a graph proof.
#[derive(Clone)]
pub struct OriginalAffineIntervalReport(Arc<ReportOwner<BoundedTensor>>);

impl OriginalAffineIntervalReport {
    pub(super) fn finish(
        output: BoundedTensor,
        original: Arc<PreparedOwner>,
        binding: OriginalAffineBinding,
        phase: OriginalAffinePhase,
    ) -> Result<Self> {
        finish(output, original, binding, phase).map(Self)
    }
    pub fn bounds(&self) -> &BoundedTensor {
        &self.0.output
    }
    pub fn scope(&self) -> &ScopeIdentity {
        &self.0.original.scope
    }
    pub fn binding(&self) -> OriginalAffineBinding {
        self.0.binding
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.reservation.bytes()
    }
    pub fn original_owner_bytes(&self) -> usize {
        self.0.original.owner.reserved_bytes()
    }
}

/// Immutable checked local row carrier with attached uncertainty planes.
/// There is no owned-array extraction: the managed result's original owner
/// and reservation travel with the arrays across every report clone.
#[derive(Clone)]
pub struct OriginalAffineLinearReport(Arc<ReportOwner<LinearBounds>>);

impl OriginalAffineLinearReport {
    pub(super) fn finish(
        output: LinearBounds,
        original: Arc<PreparedOwner>,
        binding: OriginalAffineBinding,
        phase: OriginalAffinePhase,
    ) -> Result<Self> {
        finish(output, original, binding, phase).map(Self)
    }
    pub fn bounds(&self) -> &LinearBounds {
        &self.0.output
    }
    pub fn scope(&self) -> &ScopeIdentity {
        &self.0.original.scope
    }
    pub fn binding(&self) -> OriginalAffineBinding {
        self.0.binding
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.reservation.bytes()
    }
    pub fn original_owner_bytes(&self) -> usize {
        self.0.original.owner.reserved_bytes()
    }
}
