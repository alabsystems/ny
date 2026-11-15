// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Private original-row group support for mathematical real normalization.
//!
//! This kernel grants no network, wire or cache authority. The complete group
//! and its original anchor are rescanned; no supplied radius/maxima are trusted.
//! Existing GeometryF64 storage and borrowed originals must already be retained
//! by their caller. This phase admits only its own bounded scratch and scalar
//! result on the same explicitly supplied ledger. The result retains that claim.
//!
//! With f(x)=Px/t(x), a_g=P diag(gamma)(w_g-w_t), and r_j=w_j-w_g,
//! m_j(x)=m_j(x0)+a_g dot (f(x)-f(x0))
//!                    +P diag(gamma)r_j dot (f(x)-f(x0)).
//! Orthogonal projection is a contraction and ||Df||<=1/t for eps>0. Hence
//! max_j m_j(x) <= max_j m_j(x0)+(||a_g||+||gamma||inf max_j||r_j||) L,
//! where L<=min(rho/t_min,2sqrt(H)). Beta/bias occur only in the center margin.
//! A separately admitted rescue keeps each member's center paired with its
//! own checked ||P diag(gamma)(w_j-w_t)|| before taking the complete maximum.

use super::f64_support::{GeometryF64, SupportError};
use super::geometry::{sqrt_interval, square};
use super::I;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use std::time::Instant;

const MAX_HIDDEN: usize = 65_536;
const MAX_VOCAB: usize = 1 << 20;
const SCRATCH_FLOOR: usize = 16 * 1024;
const RESULT_BYTES: usize = 512;
const WORK_PER_COORDINATE: u64 = 32;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GroupSupportError {
    Input(&'static str),
    Arithmetic(&'static str),
    Work(WorkRefusal),
    Deadline,
}
type Result<T> = std::result::Result<T, GroupSupportError>;

/// Complete immutable original head view, and one canonical contiguous group.
/// The group's authored first row is its anchor; target is excluded exactly
/// once from maxima/coverage but is still validated if present in the range.
#[derive(Clone, Copy)]
pub(crate) struct GroupSupportRequest<'a> {
    pub(crate) weights: &'a [f64],
    pub(crate) bias: Option<&'a [f64]>,
    pub(crate) hidden: usize,
    pub(crate) target: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) gamma: &'a [f64],
    pub(crate) beta: Option<&'a [f64]>,
}

fn shape(request: &GroupSupportRequest<'_>) -> Result<usize> {
    let h = request.hidden;
    if h == 0 || h > MAX_HIDDEN || !request.weights.len().is_multiple_of(h) {
        return Err(GroupSupportError::Input("group original head shape"));
    }
    let v = request.weights.len() / h;
    if v == 0
        || v > MAX_VOCAB
        || request.target >= v
        || request.start >= request.end
        || request.end > v
        || request.gamma.len() != h
        || request.beta.is_some_and(|beta| beta.len() != h)
        || request.bias.is_some_and(|bias| bias.len() != v)
    {
        return Err(GroupSupportError::Input(
            "group canonical range/parameter shape",
        ));
    }
    let competitors = request.end
        - request.start
        - usize::from((request.start..request.end).contains(&request.target));
    if competitors == 0 {
        return Err(GroupSupportError::Input("group has no original competitor"));
    }
    Ok(competitors)
}

/// Conservative unchanged-ledger admission, before allocating an anchor image.
/// Coordinate visits and reserved scalar work are deliberately distinct.
pub(crate) fn group_support_request(request: &GroupSupportRequest<'_>) -> Result<WorkRequest> {
    shape(request)?;
    let h = request.hidden;
    let members = request.end - request.start;
    let terms = members
        .checked_mul(h)
        .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
    let work = u64::try_from(terms)
        .ok()
        .and_then(|terms| terms.checked_mul(WORK_PER_COORDINATE))
        .and_then(|work| u64::try_from(h).ok()?.checked_mul(256)?.checked_add(work))
        .and_then(|work| {
            u64::try_from(members)
                .ok()?
                .checked_mul(128)?
                .checked_add(work)
        })
        .and_then(|work| work.checked_add(4096))
        .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
    let scratch_bytes = h
        .checked_mul(64)
        .and_then(|bytes| bytes.checked_add(SCRATCH_FLOOR))
        .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
    Ok(WorkRequest {
        scratch_bytes,
        persistent_bytes: RESULT_BYTES,
        work,
        proposals: 1,
    })
}

/// Separate prepaid rescue after the original anchor/radius candidate remains
/// open. Three bounded original scans per member cover its center and the two
/// projected-direction norm passes. No rows-by-hidden derived buffer is built.
pub(crate) fn member_group_support_request(
    request: &GroupSupportRequest<'_>,
) -> Result<WorkRequest> {
    let mut predicted = group_support_request(request)?;
    let terms = (request.end - request.start)
        .checked_mul(request.hidden)
        .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
    predicted.work = predicted
        .work
        .checked_add(
            u64::try_from(terms)
                .ok()
                .and_then(|n| n.checked_mul(2 * WORK_PER_COORDINATE))
                .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?,
        )
        .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
    Ok(predicted)
}

fn earlier(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Private consumed credits on one existing ledger; this never creates a
/// budget or refreshes a query's work, cancellation, time or byte allowance.
pub(crate) struct GroupSupportPhase {
    lease: WorkLease,
    budget: WorkBudget,
    class: LeaseClass,
    request: WorkRequest,
    work_left: u64,
    deadline: Option<Instant>,
    visits: u64,
}
impl GroupSupportPhase {
    pub(crate) fn admit(
        budget: &WorkBudget,
        class: LeaseClass,
        request: WorkRequest,
        deadline: Option<Instant>,
    ) -> Result<Self> {
        let lease = budget
            .lease_for(class, request)
            .map_err(GroupSupportError::Work)?;
        let deadline = earlier(
            earlier(deadline, lease.deadline()),
            ny_core::instance_budget::deadline(),
        );
        let phase = Self {
            lease,
            budget: budget.clone(),
            class,
            request,
            work_left: request.work,
            deadline,
            visits: 0,
        };
        phase.checkpoint()?;
        ny_core::require_f64_interval_proof_environment()
            .map_err(|_| GroupSupportError::Arithmetic("group binary64 proof environment"))?;
        Ok(phase)
    }

    fn checkpoint(&self) -> Result<()> {
        self.lease.checkpoint().map_err(GroupSupportError::Work)?;
        if earlier(self.deadline, ny_core::instance_budget::deadline())
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(GroupSupportError::Deadline);
        }
        Ok(())
    }

    fn scalar_work(&mut self, count: u64) -> Result<()> {
        self.work_left = self
            .work_left
            .checked_sub(count)
            .ok_or(GroupSupportError::Work(WorkRefusal::WorkLimit))?;
        self.checkpoint()
    }

    fn coordinates(&mut self, count: usize) -> Result<()> {
        let count =
            u64::try_from(count).map_err(|_| GroupSupportError::Work(WorkRefusal::Overflow))?;
        self.scalar_work(
            count
                .checked_mul(WORK_PER_COORDINATE)
                .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?,
        )?;
        self.visits = self
            .visits
            .checked_add(count)
            .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GroupSupportStats {
    pub(crate) coordinate_visits: u64,
    pub(crate) original_member_visits: u64,
    pub(crate) reserved_work: u64,
    pub(crate) consumed_work: u64,
    pub(crate) scratch_bytes: usize,
    pub(crate) persistent_bytes: usize,
}

/// Sealed only to this private construction, conditional on the original view
/// and checked region. This is not an exportable verification certificate.
#[derive(Debug)]
pub(crate) struct CheckedGroupUpper {
    upper: f64,
    center_upper: f64,
    raw_radius_upper: f64,
    projected_anchor_norm_upper: f64,
    displacement_upper: f64,
    target: usize,
    start: usize,
    end: usize,
    competitors: usize,
    stats: GroupSupportStats,
    _reservation: PersistentReservation,
}
impl CheckedGroupUpper {
    pub(crate) fn upper(&self) -> f64 {
        self.upper
    }
    pub(crate) fn center_upper(&self) -> f64 {
        self.center_upper
    }
    pub(crate) fn raw_radius_upper(&self) -> f64 {
        self.raw_radius_upper
    }
    pub(crate) fn projected_anchor_norm_upper(&self) -> f64 {
        self.projected_anchor_norm_upper
    }
    pub(crate) fn displacement_upper(&self) -> f64 {
        self.displacement_upper
    }
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.start, self.end)
    }
    pub(crate) fn target(&self) -> usize {
        self.target
    }
    pub(crate) fn competitors(&self) -> usize {
        self.competitors
    }
    pub(crate) fn stats(&self) -> GroupSupportStats {
        self.stats
    }
}

/// Distinct scalar support for max_j(center_j + ||P Γ(w_j-w_t)|| L).
/// Private construction retains the same phase claim; this has no wire/cache
/// authority and is not an affine envelope or a complete network certificate.
#[derive(Debug)]
pub(crate) struct CheckedMemberGroupUpper {
    upper: f64,
    unpaired_upper: f64,
    target: usize,
    start: usize,
    end: usize,
    competitors: usize,
    stats: GroupSupportStats,
    _reservation: PersistentReservation,
}
impl CheckedMemberGroupUpper {
    pub(crate) fn upper(&self) -> f64 {
        self.upper
    }
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.start, self.end)
    }
    pub(crate) fn target(&self) -> usize {
        self.target
    }
    pub(crate) fn competitors(&self) -> usize {
        self.competitors
    }
    pub(crate) fn stats(&self) -> GroupSupportStats {
        self.stats
    }
    #[cfg(test)]
    fn unpaired_upper(&self) -> f64 {
        self.unpaired_upper
    }
}

fn support_error(error: SupportError<GroupSupportError>) -> GroupSupportError {
    match error {
        SupportError::Arithmetic(why) => GroupSupportError::Arithmetic(why),
        SupportError::Poll(error) => error,
    }
}
fn finite(value: I, why: &'static str) -> Result<I> {
    if value.finite() {
        Ok(value)
    } else {
        Err(GroupSupportError::Arithmetic(why))
    }
}

/// Rebuild original center maxima and radii, then commit a private scalar result.
/// Success checks every group member, including an excluded in-range target.
/// No decision threshold can short-circuit that complete original scan.
pub(crate) fn group_upper(
    geometry: &GeometryF64,
    request: &GroupSupportRequest<'_>,
    mut phase: GroupSupportPhase,
) -> Result<CheckedGroupUpper> {
    phase.checkpoint()?;
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| GroupSupportError::Arithmetic("group binary64 proof environment"))?;
    let competitors = shape(request)?;
    let required = group_support_request(request)?;
    if geometry.hidden_size() != request.hidden {
        return Err(GroupSupportError::Input("group checked region shape"));
    }
    if phase.request.scratch_bytes < required.scratch_bytes {
        return Err(GroupSupportError::Work(WorkRefusal::ScratchLimit));
    }
    if phase.request.persistent_bytes < required.persistent_bytes {
        return Err(GroupSupportError::Work(WorkRefusal::PersistentLimit));
    }
    if phase.request.work < required.work {
        return Err(GroupSupportError::Work(WorkRefusal::WorkLimit));
    }
    if phase.request.proposals < required.proposals {
        return Err(GroupSupportError::Work(WorkRefusal::ProposalLimit));
    }
    phase
        .lease
        .mark_materialized()
        .map_err(GroupSupportError::Work)?;
    let h = request.hidden;
    let target = &request.weights[request.target * h..(request.target + 1) * h];
    let anchor = &request.weights[request.start * h..(request.start + 1) * h];
    let target_bias = request.bias.map_or(0.0, |bias| bias[request.target]);
    if !target_bias.is_finite() {
        return Err(GroupSupportError::Arithmetic("group finite target bias"));
    }
    let image = geometry
        .normalized_anchor_image(request.gamma, request.beta, &mut |count| {
            phase.coordinates(count)
        })
        .map_err(support_error)?;
    let direction_norm = geometry
        .projected_direction_norm(anchor, target, request.gamma, &mut |count| {
            phase.coordinates(count)
        })
        .map_err(support_error)?;
    phase.scalar_work(128)?;
    let displacement = geometry
        .normalized_displacement_upper(&mut |count| phase.coordinates(count))
        .map_err(support_error)?;
    let mut gamma_max = 0.0f64;
    for block in request.gamma.chunks(256) {
        phase.coordinates(block.len())?;
        for &gamma in block {
            gamma_max = gamma_max.max(gamma.abs());
        }
    }
    let mut center_upper = f64::NEG_INFINITY;
    let mut radius_upper = 0.0f64;
    let mut original_member_visits = 0u64;
    let mut seen_competitors = 0usize;
    for member in request.start..request.end {
        phase.scalar_work(64)?;
        let row = &request.weights[member * h..(member + 1) * h];
        let row_bias = request.bias.map_or(0.0, |bias| bias[member]);
        if !row_bias.is_finite() {
            return Err(GroupSupportError::Arithmetic(
                "group finite original member bias",
            ));
        }
        let mut center_margin = finite(
            I::point(row_bias).sub(I::point(target_bias)),
            "group original bias difference",
        )?;
        let mut radius2 = I::point(0.0);
        for start in (0..h).step_by(256) {
            let end = (start + 256).min(h);
            phase.coordinates(end - start)?;
            original_member_visits = original_member_visits
                .checked_add((end - start) as u64)
                .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
            for i in start..end {
                if !row[i].is_finite() || !target[i].is_finite() || !anchor[i].is_finite() {
                    return Err(GroupSupportError::Arithmetic(
                        "group finite original member",
                    ));
                }
                let difference = finite(
                    I::point(row[i]).sub(I::point(target[i])),
                    "group original margin difference",
                )?;
                center_margin = finite(
                    center_margin.add(difference.mul(image[i])),
                    "group original center margin",
                )?;
                let residual = finite(
                    I::point(row[i]).sub(I::point(anchor[i])),
                    "group original radius difference",
                )?;
                radius2 = finite(radius2.add(square(residual)), "group original radius2")?;
            }
        }
        phase.scalar_work(64)?;
        let radius = sqrt_interval(radius2)
            .ok_or(GroupSupportError::Arithmetic("group original radius norm"))?;
        radius_upper = radius_upper.max(radius.hi);
        if member != request.target {
            center_upper = center_upper.max(center_margin.hi);
            seen_competitors += 1;
        }
    }
    if seen_competitors != competitors || !center_upper.is_finite() {
        return Err(GroupSupportError::Arithmetic(
            "group complete finite competitor maximum",
        ));
    }
    phase.scalar_work(128)?;
    let lipschitz = finite(
        I::point(direction_norm.hi).add(I::point(gamma_max).mul(I::point(radius_upper))),
        "group learned residual norm",
    )?;
    let upper = finite(
        I::point(center_upper).add(lipschitz.mul(I::point(displacement))),
        "group complete support",
    )?
    .hi;
    // Commit releases scratch. Drop its last live vector before transferring
    // only the scalar result's capacity into the retained reservation.
    drop(image);
    phase.checkpoint()?;
    let stats = GroupSupportStats {
        coordinate_visits: phase.visits,
        original_member_visits,
        reserved_work: phase.request.work,
        consumed_work: phase.request.work - phase.work_left,
        scratch_bytes: phase.request.scratch_bytes,
        persistent_bytes: phase.request.persistent_bytes,
    };
    let reservation = phase
        .lease
        .commit_persistent()
        .map_err(GroupSupportError::Work)?;
    phase
        .budget
        .checkpoint_for(phase.class)
        .map_err(GroupSupportError::Work)?;
    if earlier(phase.deadline, ny_core::instance_budget::deadline())
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        return Err(GroupSupportError::Deadline);
    }
    Ok(CheckedGroupUpper {
        upper,
        center_upper,
        raw_radius_upper: radius_upper,
        projected_anchor_norm_upper: direction_norm.hi,
        displacement_upper: displacement,
        target: request.target,
        start: request.start,
        end: request.end,
        competitors,
        stats,
        _reservation: reservation,
    })
}

/// Recompute every original member, keep its center and projected norm paired,
/// and exclude the target from EVERY maximum. An in-range target is still
/// finite-validated before exclusion. Original biases/beta enter the center
/// only; signed/zero gamma is never divided out or treated as invertible.
pub(crate) fn member_group_upper(
    geometry: &GeometryF64,
    request: &GroupSupportRequest<'_>,
    mut phase: GroupSupportPhase,
) -> Result<CheckedMemberGroupUpper> {
    phase.checkpoint()?;
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| GroupSupportError::Arithmetic("member group binary64 proof environment"))?;
    let competitors = shape(request)?;
    let required = member_group_support_request(request)?;
    if geometry.hidden_size() != request.hidden {
        return Err(GroupSupportError::Input(
            "member group checked region shape",
        ));
    }
    if phase.request.scratch_bytes < required.scratch_bytes {
        return Err(GroupSupportError::Work(WorkRefusal::ScratchLimit));
    }
    if phase.request.persistent_bytes < required.persistent_bytes {
        return Err(GroupSupportError::Work(WorkRefusal::PersistentLimit));
    }
    if phase.request.work < required.work {
        return Err(GroupSupportError::Work(WorkRefusal::WorkLimit));
    }
    if phase.request.proposals < required.proposals {
        return Err(GroupSupportError::Work(WorkRefusal::ProposalLimit));
    }
    phase
        .lease
        .mark_materialized()
        .map_err(GroupSupportError::Work)?;
    let h = request.hidden;
    let target = &request.weights[request.target * h..(request.target + 1) * h];
    let target_bias = request.bias.map_or(0.0, |bias| bias[request.target]);
    if !target_bias.is_finite() {
        return Err(GroupSupportError::Arithmetic(
            "member group finite target bias",
        ));
    }
    let image = geometry
        .normalized_anchor_image(request.gamma, request.beta, &mut |count| {
            phase.coordinates(count)
        })
        .map_err(support_error)?;
    phase.scalar_work(128)?;
    let displacement = geometry
        .normalized_displacement_upper(&mut |count| phase.coordinates(count))
        .map_err(support_error)?;
    let mut maximum = f64::NEG_INFINITY;
    let mut center_max = f64::NEG_INFINITY;
    let mut norm_max = 0.0f64;
    let mut original_member_visits = 0u64;
    let mut seen = 0usize;
    for member in request.start..request.end {
        phase.scalar_work(64)?;
        let row = &request.weights[member * h..(member + 1) * h];
        let bias = request.bias.map_or(0.0, |bias| bias[member]);
        if !bias.is_finite() {
            return Err(GroupSupportError::Arithmetic(
                "member group finite original bias",
            ));
        }
        let mut center = finite(
            I::point(bias).sub(I::point(target_bias)),
            "member group bias difference",
        )?;
        for start in (0..h).step_by(256) {
            let end = (start + 256).min(h);
            phase.coordinates(end - start)?;
            original_member_visits = original_member_visits
                .checked_add((end - start) as u64)
                .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
            for i in start..end {
                if !row[i].is_finite() || !target[i].is_finite() {
                    return Err(GroupSupportError::Arithmetic(
                        "member group finite original member",
                    ));
                }
                let difference = finite(
                    I::point(row[i]).sub(I::point(target[i])),
                    "member group original difference",
                )?;
                center = finite(
                    center.add(difference.mul(image[i])),
                    "member group original center",
                )?;
            }
        }
        if member == request.target {
            continue;
        }
        let norm = geometry
            .projected_direction_norm(row, target, request.gamma, &mut |count| {
                phase.coordinates(count)?;
                original_member_visits = original_member_visits
                    .checked_add(count as u64)
                    .ok_or(GroupSupportError::Work(WorkRefusal::Overflow))?;
                Ok(())
            })
            .map_err(support_error)?;
        phase.scalar_work(64)?;
        let upper = finite(
            center.add(I::point(norm.hi).mul(I::point(displacement))),
            "member group paired support",
        )?
        .hi;
        maximum = maximum.max(upper);
        center_max = center_max.max(center.hi);
        norm_max = norm_max.max(norm.hi);
        seen += 1;
    }
    if seen != competitors || !maximum.is_finite() {
        return Err(GroupSupportError::Arithmetic(
            "member group complete finite support",
        ));
    }
    phase.scalar_work(128)?;
    let unpaired_upper = finite(
        I::point(center_max).add(I::point(norm_max).mul(I::point(displacement))),
        "member group comparison support",
    )?
    .hi;
    drop(image);
    phase.checkpoint()?;
    let stats = GroupSupportStats {
        coordinate_visits: phase.visits,
        original_member_visits,
        reserved_work: phase.request.work,
        consumed_work: phase.request.work - phase.work_left,
        scratch_bytes: phase.request.scratch_bytes,
        persistent_bytes: phase.request.persistent_bytes,
    };
    let reservation = phase
        .lease
        .commit_persistent()
        .map_err(GroupSupportError::Work)?;
    phase
        .budget
        .checkpoint_for(phase.class)
        .map_err(GroupSupportError::Work)?;
    if earlier(phase.deadline, ny_core::instance_budget::deadline())
        .is_some_and(|d| Instant::now() >= d)
    {
        return Err(GroupSupportError::Deadline);
    }
    Ok(CheckedMemberGroupUpper {
        upper: maximum,
        unpaired_upper,
        target: request.target,
        start: request.start,
        end: request.end,
        competitors,
        stats,
        _reservation: reservation,
    })
}

#[cfg(test)]
mod tests;
