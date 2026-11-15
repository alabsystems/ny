// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! A complete, bounded replacement of one ordered phase prefix.
//!
//! For a parent `P` and a chosen path `s[0..k]`, the destination `P+s` and
//! the `k` prefixes `P+s[..j]+opposite(s[j])` cover the parent. Binary phase
//! assignments belong to exactly one member. The actual closed phase domains
//! overlap at zero (`Off: z <= 0`, `On: z >= 0`); they are a cover, not a
//! geometrically disjoint partition. Frozen residual enclosures can overlap
//! further. Every sibling retains its actual, shorter depth.
//!
//! This is a combinatorial fact conditional on the borrowed immutable
//! [`FrozenDualProblem`] and its ordered original split rows. A caller must
//! separately qualify that chart's original network/domain provenance. There
//! is no numeric floor, node certificate, pruning, closure, or scheduling
//! authority here. In particular a positive destination bound says nothing
//! about its siblings. The original parent remains available on refusal.
//!
//! Paths are views: no temporary `Vec<Side>` is hidden in iteration. A later
//! evaluator/queue adapter must explicitly price any materialized paths and
//! its numeric scratch, iteration and row reads. Only this representation's owned capacities are
//! charged here; the borrowed chart remains its caller's responsibility.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use crate::margin_row::zono_state::{FrozenDualProblem, Side, SplitRow};

use super::FROZEN_DUAL_MAX_DEPTH;

/// Metadata includes the owned Arc headers, vectors, reservation and budget
/// handles. This is leased capacity, not a measured allocator/RSS assertion.
const METADATA_BYTES: usize = 1024;
const POLL_EVERY: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefixCoverRefusal {
    Shape,
    Depth,
    Order,
    DuplicateRow,
    ForeignProblem,
    ForeignLedger,
    Parent,
    Incomplete,
    Allocation,
    Budget(WorkRefusal),
}

impl From<WorkRefusal> for PrefixCoverRefusal {
    fn from(value: WorkRefusal) -> Self {
        Self::Budget(value)
    }
}

/// These are proposal inputs, not an asserted domain theorem. Row identities
/// must match the actual borrowed problem's prefix before any cover is returned.
pub(super) struct PrefixCoverRequest<'a> {
    pub ordered_rows: &'a [usize],
    pub parent: &'a [Side],
    pub chosen_path: &'a [Side],
    pub deadline: Option<Instant>,
}

#[derive(Debug, Clone, Copy)]
struct Sibling {
    depth: usize,
    flipped: Side,
}

#[derive(Debug)]
struct CoverStorage {
    rows: Vec<usize>,
    chosen: Vec<Side>,
    siblings: Vec<Sibling>,
    parent_depth: usize,
    deadline: Option<Instant>,
    budget: WorkBudget,
    _reservation: PersistentReservation,
}

/// Immutable metadata plus the actual original ordered chart. Clone shares
/// one capacity claim; the last clone releases it. The chart is never copied.
#[derive(Debug, Clone)]
pub(super) struct CompletePrefixCover<'problem> {
    problem: &'problem FrozenDualProblem,
    storage: Arc<CoverStorage>,
}

fn opposite(side: Side) -> Side {
    match side {
        Side::Off => Side::On,
        Side::On => Side::Off,
    }
}

fn shape(
    problem: &FrozenDualProblem,
    request: &PrefixCoverRequest<'_>,
) -> Result<(usize, usize), PrefixCoverRefusal> {
    let depth = request
        .parent
        .len()
        .checked_add(request.chosen_path.len())
        .ok_or(WorkRefusal::Overflow)?;
    if request.chosen_path.is_empty() || request.ordered_rows.len() != depth {
        return Err(PrefixCoverRefusal::Shape);
    }
    if depth > FROZEN_DUAL_MAX_DEPTH || depth > problem.max_depth() {
        return Err(PrefixCoverRefusal::Depth);
    }
    Ok((depth, request.chosen_path.len()))
}

fn prediction(depth: usize, steps: usize) -> Result<WorkRequest, PrefixCoverRefusal> {
    let persistent_bytes = depth
        .checked_mul(size_of::<usize>() + size_of::<Side>())
        .and_then(|bytes| {
            steps
                .checked_mul(size_of::<Sibling>())
                .and_then(|n| bytes.checked_add(n))
        })
        .and_then(|bytes| bytes.checked_add(METADATA_BYTES))
        .ok_or(WorkRefusal::Overflow)?;
    // All pairwise duplicate comparisons, bounded linear validation/copy,
    // descriptor construction, final validation and checkpoint bookkeeping.
    let work = depth
        .checked_mul(depth.saturating_sub(1))
        .map(|n| n / 2)
        .and_then(|n| depth.checked_mul(64).and_then(|v| n.checked_add(v)))
        .and_then(|n| steps.checked_mul(64).and_then(|v| n.checked_add(v)))
        .and_then(|n| n.checked_add(1024))
        .ok_or(WorkRefusal::Overflow)?;
    Ok(WorkRequest {
        scratch_bytes: 0,
        persistent_bytes,
        work: u64::try_from(work).map_err(|_| WorkRefusal::Overflow)?,
        proposals: 1,
    })
}

/// Capacity/work census for this cover only, with no new allowance. Shape
/// checks are constant work; ordered identity checks happen under admission.
pub(super) fn admission(
    problem: &FrozenDualProblem,
    request: &PrefixCoverRequest<'_>,
) -> Result<WorkRequest, PrefixCoverRefusal> {
    let (depth, steps) = shape(problem, request)?;
    prediction(depth, steps)
}

fn external_deadline(deadline: Option<Instant>) -> Result<(), PrefixCoverRefusal> {
    let now = Instant::now();
    if [deadline, ny_core::instance_budget::deadline()]
        .into_iter()
        .flatten()
        .any(|end| now >= end)
    {
        return Err(WorkRefusal::Deadline.into());
    }
    Ok(())
}

fn checkpoint(lease: &WorkLease, deadline: Option<Instant>) -> Result<(), PrefixCoverRefusal> {
    lease.checkpoint()?;
    external_deadline(deadline)
}

fn validate_rows(
    problem: &FrozenDualProblem,
    rows: &[usize],
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<(), PrefixCoverRefusal> {
    let mut visits = 0usize;
    for (i, &row) in rows.iter().enumerate() {
        if i % POLL_EVERY == 0 {
            checkpoint(lease, deadline)?;
        }
        if problem.order()[i] != row {
            return Err(PrefixCoverRefusal::Order);
        }
        for &previous in &rows[..i] {
            if visits.is_multiple_of(POLL_EVERY) {
                checkpoint(lease, deadline)?;
            }
            visits += 1;
            if previous == row {
                return Err(PrefixCoverRefusal::DuplicateRow);
            }
        }
    }
    checkpoint(lease, deadline)
}

pub(super) fn build<'problem>(
    problem: &'problem FrozenDualProblem,
    request: &PrefixCoverRequest<'_>,
    budget: &WorkBudget,
) -> Result<CompletePrefixCover<'problem>, PrefixCoverRefusal> {
    build_inner(problem, request, budget, || {})
}

// The private hook permits deterministic cancellation controls at actual
// checkpoint boundaries. Production supplies a no-op; it is not a proposer
// or an arithmetic/coverage callback and cannot change metadata.
fn build_inner<'problem>(
    problem: &'problem FrozenDualProblem,
    request: &PrefixCoverRequest<'_>,
    budget: &WorkBudget,
    mut checkpoint_hook: impl FnMut(),
) -> Result<CompletePrefixCover<'problem>, PrefixCoverRefusal> {
    let (depth, steps) = shape(problem, request)?;
    external_deadline(request.deadline)?;
    let predicted = prediction(depth, steps)?;
    let mut lease = budget.lease(predicted)?;
    checkpoint(&lease, request.deadline)?;
    validate_rows(problem, request.ordered_rows, &lease, request.deadline)?;
    lease.mark_materialized()?;

    let mut rows = Vec::new();
    let mut chosen = Vec::new();
    let mut siblings = Vec::new();
    rows.try_reserve_exact(depth)
        .map_err(|_| PrefixCoverRefusal::Allocation)?;
    chosen
        .try_reserve_exact(depth)
        .map_err(|_| PrefixCoverRefusal::Allocation)?;
    siblings
        .try_reserve_exact(steps)
        .map_err(|_| PrefixCoverRefusal::Allocation)?;
    let actual_capacity = rows
        .capacity()
        .checked_mul(size_of::<usize>())
        .and_then(|n| {
            chosen
                .capacity()
                .checked_mul(size_of::<Side>())
                .and_then(|v| n.checked_add(v))
        })
        .and_then(|n| {
            siblings
                .capacity()
                .checked_mul(size_of::<Sibling>())
                .and_then(|v| n.checked_add(v))
        })
        .and_then(|n| n.checked_add(size_of::<CoverStorage>() + 4 * size_of::<usize>()))
        .ok_or(WorkRefusal::Overflow)?;
    if actual_capacity > predicted.persistent_bytes {
        return Err(PrefixCoverRefusal::Allocation);
    }
    for i in 0..depth {
        if i % POLL_EVERY == 0 {
            checkpoint_hook();
            checkpoint(&lease, request.deadline)?;
        }
        rows.push(problem.order()[i]);
        chosen.push(if i < request.parent.len() {
            request.parent[i]
        } else {
            request.chosen_path[i - request.parent.len()]
        });
    }
    for (j, &side) in request.chosen_path.iter().enumerate() {
        if j % POLL_EVERY == 0 {
            checkpoint_hook();
            checkpoint(&lease, request.deadline)?;
        }
        siblings.push(Sibling {
            depth: request.parent.len() + j + 1,
            flipped: opposite(side),
        });
    }
    checkpoint(&lease, request.deadline)?;
    let reservation = lease.commit_persistent()?;
    let cover = CompletePrefixCover {
        problem,
        storage: Arc::new(CoverStorage {
            rows,
            chosen,
            siblings,
            parent_depth: request.parent.len(),
            deadline: request.deadline,
            budget: budget.clone(),
            _reservation: reservation,
        }),
    };
    budget.checkpoint_for(LeaseClass::Optional)?;
    external_deadline(request.deadline)?;
    Ok(cover)
}

#[derive(Debug, Clone, Copy)]
enum MemberKind {
    Parent,
    Destination,
    FirstDisagreement(usize),
}

/// A borrowed prefix view. It cannot be converted into a node bound or an
/// owned unpriced evaluator/queue path by this module.
#[derive(Debug, Clone, Copy)]
pub(super) struct PrefixMember<'cover, 'problem> {
    cover: &'cover CompletePrefixCover<'problem>,
    kind: MemberKind,
}

impl<'cover, 'problem> PrefixMember<'cover, 'problem> {
    pub(super) fn depth(self) -> usize {
        match self.kind {
            MemberKind::Parent => self.cover.storage.parent_depth,
            MemberKind::Destination => self.cover.storage.chosen.len(),
            MemberKind::FirstDisagreement(j) => self.cover.storage.siblings[j].depth,
        }
    }

    pub(super) fn sides(self) -> impl ExactSizeIterator<Item = &'cover Side> + 'cover {
        (0..self.depth()).map(move |i| match self.kind {
            MemberKind::FirstDisagreement(j) if i + 1 == self.depth() => {
                &self.cover.storage.siblings[j].flipped
            }
            _ => &self.cover.storage.chosen[i],
        })
    }

    pub(super) fn ordered_splits(
        self,
    ) -> impl ExactSizeIterator<Item = (SplitRow<'problem>, &'cover Side)> + 'cover {
        let problem: &'problem FrozenDualProblem = self.cover.problem;
        self.sides()
            .enumerate()
            .map(move |(i, side)| (problem.split_row(i), side))
    }
}

impl<'problem> CompletePrefixCover<'problem> {
    /// Recovery view, separate from the replacement cover's members.
    pub(super) fn parent(&self) -> PrefixMember<'_, 'problem> {
        PrefixMember {
            cover: self,
            kind: MemberKind::Parent,
        }
    }

    /// Destination first, then first-disagreement siblings in increasing
    /// depth. This iterator allocates no paths or queue entries.
    pub(super) fn members(&self) -> impl ExactSizeIterator<Item = PrefixMember<'_, 'problem>> {
        // Construction limits this length to MAX_DEPTH; the exclusive usize
        // range implements ExactSizeIterator, unlike RangeInclusive<usize>.
        let member_count = self.storage.siblings.len() + 1;
        (0..member_count).map(move |i| PrefixMember {
            cover: self,
            kind: if i == 0 {
                MemberKind::Destination
            } else {
                MemberKind::FirstDisagreement(i - 1)
            },
        })
    }

    /// Fresh complete structural replay under the same caller ledger's
    /// Replay deadline, restricted by any earlier construction/caller deadline.
    /// A bit-identical cloned problem is still foreign.
    /// Nothing here replays a numeric bound or authenticates network origins.
    pub(super) fn replay(
        &self,
        problem: &FrozenDualProblem,
        expected_parent: &[Side],
        budget: &WorkBudget,
        deadline: Option<Instant>,
    ) -> Result<(), PrefixCoverRefusal> {
        if !std::ptr::eq(self.problem, problem) {
            return Err(PrefixCoverRefusal::ForeignProblem);
        }
        if !self.storage.budget.shares_ledger(budget) {
            return Err(PrefixCoverRefusal::ForeignLedger);
        }
        let storage = &self.storage;
        let deadline = match (storage.deadline, deadline) {
            (Some(original), Some(requested)) => Some(original.min(requested)),
            (Some(original), None) => Some(original),
            (None, requested) => requested,
        };
        let depth = storage.chosen.len();
        let steps = depth
            .checked_sub(storage.parent_depth)
            .ok_or(PrefixCoverRefusal::Incomplete)?;
        if depth > FROZEN_DUAL_MAX_DEPTH || depth > problem.max_depth() {
            return Err(PrefixCoverRefusal::Depth);
        }
        if steps == 0 || storage.rows.len() != depth || storage.siblings.len() != steps {
            return Err(PrefixCoverRefusal::Incomplete);
        }
        if expected_parent.len() != storage.parent_depth {
            return Err(PrefixCoverRefusal::Parent);
        }
        external_deadline(deadline)?;
        let mut request = prediction(depth, steps)?;
        request.persistent_bytes = 0;
        request.proposals = 0;
        let mut lease = budget.lease_for(LeaseClass::Replay, request)?;
        validate_rows(problem, &storage.rows, &lease, deadline)?;
        for (i, &side) in expected_parent.iter().enumerate() {
            if i % POLL_EVERY == 0 {
                checkpoint(&lease, deadline)?;
            }
            if storage.chosen[i] != side {
                return Err(PrefixCoverRefusal::Parent);
            }
        }
        for (j, sibling) in storage.siblings.iter().enumerate() {
            if j % POLL_EVERY == 0 {
                checkpoint(&lease, deadline)?;
            }
            if sibling.depth != storage.parent_depth + j + 1
                || sibling.flipped != opposite(storage.chosen[storage.parent_depth + j])
            {
                return Err(PrefixCoverRefusal::Incomplete);
            }
        }
        checkpoint(&lease, deadline)?;
        lease.complete()?;
        checkpoint(&lease, deadline)
    }
}

#[cfg(test)]
mod tests;
