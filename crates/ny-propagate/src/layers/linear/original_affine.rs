// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact-real affine proof kernels over immutable borrowed original binary32
//! coefficients. This module grants no Layer or whole-network eligibility.
//!
//! Every operation receives one explicitly admitted existing-ledger phase.
//! Returned arrays must remain under that phase until its persistent reservation
//! is attached to their private report owner. These crate-private kernels do
//! not create budgets, cache derived weights, or grant diagnostic arms authority.

mod crown;
mod ibp;
mod sparse;
#[cfg(test)]
mod tests;

pub(crate) use crown::{crown, crown_request};
pub(crate) use ibp::{ibp, ibp_request};
pub(crate) use sparse::{seed_logit_differences, sparse_request, LogitDifference};

use ndarray::{Array2, ArrayView1, ArrayView2};
use ny_core::{require_f64_interval_proof_environment, NyError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::bias::{
    add_f64_up, f32_to_f64_exact, nonnegative_f32_error_or_infinity, publish_error_up_normal,
};
use crate::certify::work::{
    LeaseClass, PersistentReservation, RetainedOwner, WorkBudget, WorkLease, WorkRefusal,
    WorkRequest,
};

const POLL_WORK: u64 = 4096;
const MAX_ELEMENTS: usize = 4 * 1024 * 1024;
const MAX_RANK: usize = 1024;
const SCRATCH_BYTES: usize = 32 * 1024;
const TILE: usize = 256;
const VECTOR_EXTRA: usize = 128;
const OUTPUT_EXTRA: usize = 16 * 1024;

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("original-affine proof: {message}"))
}
fn numeric(message: &str) -> NyError {
    NyError::NumericalInstability(format!("original-affine proof: {message}"))
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
fn cap_elements(value: usize) -> Result<usize> {
    if value > MAX_ELEMENTS {
        return Err(NyError::UnsupportedConfiguration(
            "original-affine proof exceeds the 4 Mi element profile".into(),
        ));
    }
    Ok(value)
}
fn work_error(error: WorkRefusal) -> NyError {
    if error == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("original-affine shared phase deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("original-affine shared phase: {error}"))
    }
}
fn earlier(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

// Identity carries no arithmetic authority or resource allowance. The
// allocation-free counter never wraps; exhaustion refuses a new phase.
static NEXT_PHASE_IDENTITY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OriginalAffinePhaseIdentity(u64);

fn next_phase_identity(counter: &AtomicU64) -> Result<OriginalAffinePhaseIdentity> {
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current != 0).then(|| current.checked_add(1)).flatten()
        })
        .map(OriginalAffinePhaseIdentity)
        .map_err(|_| {
            NyError::UnsupportedConfiguration("original-affine phase identity exhausted".into())
        })
}

/// One aggregate phase reservation on an existing authority, with private
/// consumed-work and live-output-capacity credits. Construction never grants
/// a new ledger. Caller-owned original parameters are priced by their owner.
pub(crate) struct OriginalAffinePhase {
    identity: OriginalAffinePhaseIdentity,
    lease: WorkLease,
    work_left: u64,
    persistent_left: usize,
    scratch_bytes: usize,
    deadline: Option<Instant>,
    since_poll: u64,
    #[cfg(test)]
    fail_after: Option<u64>,
    #[cfg(test)]
    spent: u64,
}

impl OriginalAffinePhase {
    pub(crate) fn admit(
        budget: &WorkBudget,
        class: LeaseClass,
        request: WorkRequest,
        deadline: Option<Instant>,
    ) -> Result<Self> {
        require_f64_interval_proof_environment()?;
        let identity = next_phase_identity(&NEXT_PHASE_IDENTITY)?;
        let lease = budget.lease_for(class, request).map_err(work_error)?;
        let deadline = earlier(
            earlier(deadline, lease.deadline()),
            ny_core::instance_budget::deadline(),
        );
        let phase = Self {
            identity,
            lease,
            work_left: request.work,
            persistent_left: request.persistent_bytes,
            scratch_bytes: request.scratch_bytes,
            deadline,
            since_poll: 0,
            #[cfg(test)]
            fail_after: None,
            #[cfg(test)]
            spent: 0,
        };
        phase.checkpoint()?;
        Ok(phase)
    }

    /// Bind private phase-paid preparation to this exact admitted phase.
    /// A clone of its ledger or a later phase does not share this identity.
    pub(crate) fn identity(&self) -> OriginalAffinePhaseIdentity {
        self.identity
    }

    /// The actual inherited phase clock, including the original lease cutoff.
    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Restrict an inherited owner/query deadline; this cannot extend a lease.
    pub(crate) fn restrict_deadline(&mut self, deadline: Option<Instant>) -> Result<()> {
        self.deadline = earlier(self.deadline, deadline);
        self.checkpoint()
    }

    pub(crate) fn checkpoint(&self) -> Result<()> {
        self.lease.checkpoint().map_err(work_error)?;
        if earlier(self.deadline, ny_core::instance_budget::deadline())
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(NyError::DeadlineExceeded(
                "original-affine effective deadline".into(),
            ));
        }
        #[cfg(test)]
        if self.fail_after.is_some_and(|limit| self.spent >= limit) {
            return Err(NyError::DeadlineExceeded(
                "original-affine injected poll".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn entry(&self) -> Result<()> {
        self.checkpoint()?;
        require_f64_interval_proof_environment()?;
        if self.scratch_bytes < SCRATCH_BYTES {
            return Err(work_error(WorkRefusal::ScratchLimit));
        }
        Ok(())
    }

    fn spend(&mut self, amount: u64) -> Result<()> {
        self.work_left = self
            .work_left
            .checked_sub(amount)
            .ok_or_else(|| work_error(WorkRefusal::WorkLimit))?;
        #[cfg(test)]
        {
            self.spent = self.spent.saturating_add(amount);
        }
        self.since_poll = self
            .since_poll
            .checked_add(amount)
            .ok_or_else(|| invalid("poll work overflow"))?;
        if self.since_poll >= POLL_WORK {
            self.since_poll = 0;
            self.checkpoint()?;
        }
        Ok(())
    }

    /// Price caller scope/input/report work on this same admitted phase.
    /// The caller must also checkpoint within bounded processing quanta;
    /// reserving a bulk scan does not make that scan interruptible.
    pub(crate) fn charge_work(&mut self, amount: u64) -> Result<()> {
        self.spend(amount)
    }

    fn retain_bytes(&mut self, bytes: usize) -> Result<()> {
        self.checkpoint()?;
        self.persistent_left = self
            .persistent_left
            .checked_sub(bytes)
            .ok_or_else(|| work_error(WorkRefusal::PersistentLimit))?;
        Ok(())
    }

    /// Retain caller metadata/capacity beside the returned kernel arrays.
    /// All such objects must travel with the phase's persistent report owner.
    pub(crate) fn retain_metadata(&mut self, bytes: usize) -> Result<()> {
        self.retain_bytes(bytes)
    }

    fn output_vec(&mut self, len: usize) -> Result<Vec<f32>> {
        let allowed = add(mul(len, size_of::<f32>())?, VECTOR_EXTRA)?;
        self.retain_bytes(allowed)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|e| invalid(&format!("output allocation failed: {e}")))?;
        self.checkpoint()?;
        if mul(values.capacity(), size_of::<f32>())? > allowed {
            return Err(work_error(WorkRefusal::PersistentLimit));
        }
        while values.len() < len {
            let count = (len - values.len()).min(POLL_WORK as usize);
            self.spend(as_work(count)?)?;
            values.resize(values.len() + count, 0.0);
            self.checkpoint()?;
        }
        Ok(values)
    }

    /// Caller must attach this reservation to the private owner of ALL returned
    /// arrays before dropping phase authority. No public array escape exists.
    pub(crate) fn commit(self) -> Result<PersistentReservation> {
        self.checkpoint()?;
        self.lease.commit_persistent().map_err(work_error)
    }

    pub(crate) fn commit_retained_with<T>(
        self,
        build: impl FnOnce() -> T,
    ) -> Result<RetainedOwner<T>> {
        self.checkpoint()?;
        self.lease.commit_retained_with(build).map_err(work_error)
    }
}

/// Validated original parameters, borrowing their immutable logical contents.
/// Validate once inside an admitted phase, then reuse across streamed batches.
/// No copy or signed/transposed/absolute/spectral cache is retained here.
#[derive(Clone, Copy)]
pub(crate) struct OriginalAffineView<'a> {
    weight: ArrayView2<'a, f32>,
    bias: Option<ArrayView1<'a, f32>>,
}

impl<'a> OriginalAffineView<'a> {
    pub(crate) fn validation_request(
        outputs: usize,
        inputs: usize,
        bias: bool,
    ) -> Result<WorkRequest> {
        if outputs == 0 || inputs == 0 {
            return Err(invalid("nonempty original affine dimensions required"));
        }
        let terms = add(mul(outputs, inputs)?, if bias { outputs } else { 0 })?;
        Ok(WorkRequest {
            scratch_bytes: SCRATCH_BYTES,
            persistent_bytes: 0,
            work: as_work(add(mul(terms, 4)?, 1024)?)?,
            proposals: 0,
        })
    }

    pub(crate) fn validate(
        weight: ArrayView2<'a, f32>,
        bias: Option<ArrayView1<'a, f32>>,
        phase: &mut OriginalAffinePhase,
    ) -> Result<Self> {
        phase.entry()?;
        Self::validation_request(weight.nrows(), weight.ncols(), bias.is_some())?;
        if bias.is_some_and(|bias| bias.len() != weight.nrows()) {
            return Err(invalid("original bias shape mismatch"));
        }
        for &value in weight
            .iter()
            .chain(bias.iter().flat_map(|bias| bias.iter()))
        {
            phase.spend(4)?;
            if !value.is_finite() {
                return Err(numeric("nonfinite original coefficient"));
            }
        }
        phase.checkpoint()?;
        Ok(Self { weight, bias })
    }

    pub(crate) fn in_features(self) -> usize {
        self.weight.ncols()
    }
    pub(crate) fn out_features(self) -> usize {
        self.weight.nrows()
    }
    fn bias_at(self, row: usize) -> f64 {
        self.bias.map_or(0.0, |bias| f32_to_f64_exact(bias[row]))
    }
}

/// Once-validated immutable owned-array headers. Keeping references here avoids
/// retaining duplicate ndarray view headers in every prepared chain stage.
/// OriginalAffineView remains the generic strided-parameter kernel adapter.
#[derive(Clone, Copy)]
pub(crate) struct OriginalAffineArrays<'a> {
    weight: &'a Array2<f32>,
    bias: Option<&'a ndarray::Array1<f32>>,
}

impl<'a> OriginalAffineArrays<'a> {
    pub(crate) fn validate(
        weight: &'a Array2<f32>,
        bias: Option<&'a ndarray::Array1<f32>>,
        phase: &mut OriginalAffinePhase,
    ) -> Result<Self> {
        OriginalAffineView::validate(weight.view(), bias.map(|bias| bias.view()), phase)?;
        Ok(Self { weight, bias })
    }

    pub(crate) fn in_features(self) -> usize {
        self.weight.ncols()
    }

    pub(crate) fn out_features(self) -> usize {
        self.weight.nrows()
    }

    pub(crate) fn view(self) -> OriginalAffineView<'a> {
        // Only these same immutable arrays passed the original validation.
        // ndarray preserves their pointer, dimensions and strides in each view.
        OriginalAffineView {
            weight: self.weight.view(),
            bias: self.bias.map(|bias| bias.view()),
        }
    }
}

/// All prediction arithmetic is checked. The extra payload covers vector and
/// ndarray metadata plus conservative allocator-capacity headroom; it is a
/// requested-capacity allowance, not a bound on allocator RSS.
fn output_request(elements: usize, work: usize) -> Result<WorkRequest> {
    Ok(WorkRequest {
        scratch_bytes: SCRATCH_BYTES,
        persistent_bytes: add(mul(mul(elements, size_of::<f32>())?, 2)?, OUTPUT_EXTRA)?,
        work: as_work(add(work, 1024)?)?,
        proposals: 0,
    })
}

fn array2(rows: usize, columns: usize, values: Vec<f32>) -> Result<Array2<f32>> {
    Array2::from_shape_vec((rows, columns), values).map_err(|_| invalid("output reshape"))
}

fn stored_coefficient(lower: f64, upper: f64, propagated_error: f64) -> Result<(f32, f32)> {
    if !lower.is_finite()
        || !upper.is_finite()
        || lower > upper
        || !propagated_error.is_finite()
        || propagated_error < 0.0
    {
        return Err(numeric("nonfinite coefficient interval/error"));
    }
    let mut stored = (lower * 0.5 + upper * 0.5) as f32;
    if !stored.is_finite() {
        return Err(numeric("coefficient storage overflow"));
    }
    // A downstream f32 kernel may flush a subnormal coefficient. Publish zero
    // deliberately and pay its full exact difference in a normal error plane.
    if stored.to_bits() & 0x7f80_0000 == 0 {
        stored = 0.0;
    }
    let center = f32_to_f64_exact(stored);
    let rounding = add_f64_up(center, -lower)
        .max(add_f64_up(upper, -center))
        .max(0.0);
    let error = publish_error_up_normal(add_f64_up(rounding, propagated_error));
    if !error.is_finite() {
        return Err(numeric("coefficient-error storage overflow"));
    }
    Ok((stored, error))
}

fn incoming_error(value: f32) -> Result<f64> {
    let error = nonnegative_f32_error_or_infinity(value);
    if error.is_finite() {
        Ok(error)
    } else {
        Err(numeric("poisoned incoming coefficient error"))
    }
}

#[cfg(test)]
mod phase_identity_tests {
    use super::*;

    #[test]
    fn original_affine_activation_phase_identity_nonce_never_wraps_or_reuses_zero() {
        let local = AtomicU64::new(u64::MAX - 1);
        let last = next_phase_identity(&local).unwrap();
        assert_eq!(last, OriginalAffinePhaseIdentity(u64::MAX - 1));
        assert_eq!(local.load(Ordering::Relaxed), u64::MAX);
        assert!(next_phase_identity(&local).is_err());
        assert!(next_phase_identity(&local).is_err());
        assert_eq!(local.load(Ordering::Relaxed), u64::MAX);
        let zero = AtomicU64::new(0);
        assert!(next_phase_identity(&zero).is_err());
        assert_eq!(zero.load(Ordering::Relaxed), 0);
        // This test does not mutate the process-wide production counter.
        let ordinary = AtomicU64::new(1);
        assert_ne!(
            next_phase_identity(&ordinary).unwrap(),
            next_phase_identity(&ordinary).unwrap()
        );
    }
}
