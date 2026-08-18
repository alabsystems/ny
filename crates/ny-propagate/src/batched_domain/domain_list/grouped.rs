// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

//! Packed OR-of-AND objective state for the grouped GPU DomainList lane.
//!
//! A VNN-LIB disjunction is represented as rows packed clause-by-clause.  A
//! clause is discharged when any row in that clause has a certified
//! `lower > threshold`; the whole domain is verified only when every clause is
//! discharged.  The scalar queue interval is therefore
//! `min_clause(max_row(row_bound - threshold))`.
//!
//! The production grouped GPU executor remains unqualified/default-off.  These
//! types provide the clause-preserving storage and fail-closed decision
//! contract it must use when that executor is wired.

use ndarray::{ArrayD, Axis};
use ny_core::{NyError, Result};
use ny_tensor::TensorStorage;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

use super::super::ConstraintTuple;
use super::{PickedDomains, ProcessedDomains};

/// Collision-resistant identity of the ordered objective rows and their spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GroupedSpecFingerprint([u8; 32]);

/// Immutable description of rows packed clause-by-clause.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupedDisjunctiveLayout {
    thresholds: Vec<f32>,
    clause_sizes: Vec<usize>,
    spec_fingerprint: GroupedSpecFingerprint,
}

impl GroupedDisjunctiveLayout {
    /// Build and validate a packed grouped layout.
    ///
    /// Empty layouts, zero-width clauses, overflowing totals, non-finite
    /// thresholds, and uncovered/trailing rows are rejected before any queue
    /// state is created.
    ///
    /// `canonical_objective` must encode the ordered objective rows and every
    /// model/spec detail that changes their meaning. The resulting SHA-256
    /// fingerprint distinguishes objectives that happen to have identical row
    /// counts, clause sizes, and thresholds.
    pub(crate) fn new(
        thresholds: Vec<f32>,
        clause_sizes: Vec<usize>,
        canonical_objective: &[u8],
    ) -> Result<Self> {
        if canonical_objective.is_empty() {
            return Err(NyError::InvalidSpec(
                "grouped DomainList: empty canonical objective identity".to_string(),
            ));
        }
        if thresholds.is_empty() {
            return Err(NyError::InvalidSpec(
                "grouped DomainList: empty threshold rows".to_string(),
            ));
        }
        if clause_sizes.is_empty() {
            return Err(NyError::InvalidSpec(
                "grouped DomainList: empty clause_sizes".to_string(),
            ));
        }
        let total_rows = clause_sizes
            .iter()
            .try_fold(0usize, |total, &size| {
                if size == 0 {
                    None
                } else {
                    total.checked_add(size)
                }
            })
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "grouped DomainList: zero-width clause or clause-size overflow".to_string(),
                )
            })?;
        if total_rows != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList: clause_sizes sum {total_rows} != threshold rows {}",
                thresholds.len()
            )));
        }
        if let Some((row, threshold)) = thresholds
            .iter()
            .enumerate()
            .find(|(_, threshold)| !threshold.is_finite())
        {
            return Err(NyError::NumericalInstability(format!(
                "grouped DomainList: threshold row {row} is non-finite ({threshold})"
            )));
        }
        let mut hasher = Sha256::new();
        hasher.update(b"ny.grouped-objective.v1\0");
        hasher.update(
            u64::try_from(canonical_objective.len())
                .map_err(|_| {
                    NyError::InvalidSpec(
                        "grouped DomainList: canonical objective identity is too large".to_string(),
                    )
                })?
                .to_le_bytes(),
        );
        hasher.update(canonical_objective);
        for threshold in &thresholds {
            hasher.update(threshold.to_bits().to_le_bytes());
        }
        for &clause_size in &clause_sizes {
            hasher.update(
                u64::try_from(clause_size)
                    .map_err(|_| {
                        NyError::InvalidSpec(
                            "grouped DomainList: clause size does not fit fingerprint".to_string(),
                        )
                    })?
                    .to_le_bytes(),
            );
        }
        let spec_fingerprint = GroupedSpecFingerprint(hasher.finalize().into());

        Ok(Self {
            thresholds,
            clause_sizes,
            spec_fingerprint,
        })
    }

    pub(crate) fn row_count(&self) -> usize {
        self.thresholds.len()
    }

    pub(crate) fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    pub(crate) fn clause_sizes(&self) -> &[usize] {
        &self.clause_sizes
    }

    pub(crate) fn spec_fingerprint(&self) -> GroupedSpecFingerprint {
        self.spec_fingerprint
    }

    /// Reduce one domain's per-row interval to the exact grouped scalar
    /// interval used for queue ordering and verification.
    pub(crate) fn summarize(
        &self,
        row_lowers: &[f32],
        row_uppers: &[f32],
    ) -> Result<GroupedBoundSummary> {
        if row_lowers.len() != self.row_count() || row_uppers.len() != self.row_count() {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList: row interval shape mismatch \
                 (expected {}, lower={}, upper={})",
                self.row_count(),
                row_lowers.len(),
                row_uppers.len()
            )));
        }

        let mut offset = 0usize;
        let mut lower_margin = f32::INFINITY;
        let mut upper_margin = f32::INFINITY;
        for &clause_size in &self.clause_sizes {
            let mut clause_lower = f32::NEG_INFINITY;
            let mut clause_upper = f32::NEG_INFINITY;
            for row in offset..offset + clause_size {
                let lower = row_lowers[row];
                let upper = row_uppers[row];
                if !lower.is_finite() || !upper.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "grouped DomainList: row {row} has non-finite interval \
                         [{lower}, {upper}]"
                    )));
                }
                if lower > upper {
                    return Err(NyError::InvalidSpec(format!(
                        "grouped DomainList: row {row} has inverted interval \
                         [{lower}, {upper}]"
                    )));
                }
                let lower_gap = lower - self.thresholds[row];
                let upper_gap = upper - self.thresholds[row];
                if !lower_gap.is_finite() || !upper_gap.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "grouped DomainList: row {row} threshold subtraction overflowed \
                         (bounds=[{lower}, {upper}], threshold={})",
                        self.thresholds[row]
                    )));
                }
                clause_lower = clause_lower.max(lower_gap);
                clause_upper = clause_upper.max(upper_gap);
            }
            lower_margin = lower_margin.min(clause_lower);
            upper_margin = upper_margin.min(clause_upper);
            offset += clause_size;
        }

        if !lower_margin.is_finite() || !upper_margin.is_finite() || lower_margin > upper_margin {
            return Err(NyError::NumericalInstability(format!(
                "grouped DomainList: invalid reduced interval \
                 [{lower_margin}, {upper_margin}]"
            )));
        }
        Ok(GroupedBoundSummary {
            lower_margin,
            upper_margin,
        })
    }
}

/// Clause-correct scalar summary for one packed domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GroupedBoundSummary {
    lower_margin: f32,
    upper_margin: f32,
}

impl GroupedBoundSummary {
    pub(crate) fn lower_margin(self) -> f32 {
        self.lower_margin
    }

    pub(crate) fn upper_margin(self) -> f32 {
        self.upper_margin
    }

    /// Strict comparison is intentional: equality does not discharge a row.
    pub(crate) fn is_verified(self) -> bool {
        self.lower_margin > 0.0
    }
}

/// Packed row intervals with shape `[domain, row]`.
#[derive(Debug, Clone)]
pub(super) struct PackedGroupedBounds {
    row_lowers: ArrayD<f32>,
    row_uppers: ArrayD<f32>,
}

impl PackedGroupedBounds {
    pub(super) fn new(row_lowers: ArrayD<f32>, row_uppers: ArrayD<f32>) -> Self {
        Self {
            row_lowers,
            row_uppers,
        }
    }

    pub(super) fn row_lowers(&self) -> &ArrayD<f32> {
        &self.row_lowers
    }

    pub(super) fn row_uppers(&self) -> &ArrayD<f32> {
        &self.row_uppers
    }

    /// Validate the physical packing and return one clause-correct summary per
    /// domain. Validation is deliberately all-row, not keep-mask filtered:
    /// malformed GPU output must trigger fallback rather than hide in a dropped
    /// row.
    pub(super) fn summaries(
        &self,
        layout: &GroupedDisjunctiveLayout,
        expected_batch: usize,
    ) -> Result<Vec<GroupedBoundSummary>> {
        let expected_shape = [expected_batch, layout.row_count()];
        if self.row_lowers.ndim() != 2
            || self.row_uppers.ndim() != 2
            || self.row_lowers.shape() != expected_shape
            || self.row_uppers.shape() != expected_shape
        {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList: packed bounds must both have shape {:?} \
                 (lower={:?}, upper={:?})",
                expected_shape,
                self.row_lowers.shape(),
                self.row_uppers.shape()
            )));
        }

        let mut summaries = Vec::with_capacity(expected_batch);
        for domain in 0..expected_batch {
            let lowers: Vec<f32> = self
                .row_lowers
                .index_axis(Axis(0), domain)
                .iter()
                .copied()
                .collect();
            let uppers: Vec<f32> = self
                .row_uppers
                .index_axis(Axis(0), domain)
                .iter()
                .copied()
                .collect();
            summaries.push(layout.summarize(&lowers, &uppers)?);
        }
        Ok(summaries)
    }
}

/// Queue-local identity assigned to a picked parent.
///
/// The numeric payload is private. Callers may copy an ID into a resolution,
/// but the consumed lease verifies the exact ordered ID vector, so swapping or
/// duplicating IDs cannot release another parent's proof obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GroupedDomainId(u64);

impl GroupedDomainId {
    pub(super) fn from_queue_counter(value: u64) -> Self {
        Self(value)
    }
}

/// Opaque, non-cloneable authority for one picked batch.
#[derive(Debug)]
struct GroupedLeaseAuthority {
    lease_id: u64,
    queue_token: Arc<()>,
}

/// A DomainList batch accompanied by read-only grouped state and a consumed
/// lease. All authority-bearing fields are private.
#[derive(Debug)]
pub(crate) struct PickedGroupedDomains {
    domains: PickedDomains,
    row_bounds: PackedGroupedBounds,
    layout: GroupedDisjunctiveLayout,
    domain_ids: Vec<GroupedDomainId>,
    authority: Option<GroupedLeaseAuthority>,
}

impl PickedGroupedDomains {
    pub(crate) fn domains(&self) -> &PickedDomains {
        &self.domains
    }

    pub(super) fn row_bounds(&self) -> &PackedGroupedBounds {
        &self.row_bounds
    }

    pub(crate) fn layout(&self) -> &GroupedDisjunctiveLayout {
        &self.layout
    }

    pub(crate) fn domain_ids(&self) -> &[GroupedDomainId] {
        &self.domain_ids
    }

    pub(super) fn summaries(&self) -> Result<Vec<GroupedBoundSummary>> {
        self.row_bounds
            .summaries(&self.layout, self.domains.batch_size)
    }

    /// Mint ordered, non-cloneable child tokens for one exact input
    /// bisection. This occurs before either child is evaluated.
    pub(crate) fn mint_input_bisection_tokens(
        &self,
        parent_index: usize,
        first: &ProcessedDomains,
        second: &ProcessedDomains,
    ) -> Result<[GroupedChildEvaluationToken; 2]> {
        self.mint_child_tokens(
            parent_index,
            first,
            second,
            GroupedSplitKind::InputBisection,
        )
    }

    /// Mint ordered, non-cloneable child tokens for one exact phase split.
    /// This occurs before either child is evaluated.
    pub(crate) fn mint_phase_split_tokens(
        &self,
        parent_index: usize,
        first: &ProcessedDomains,
        second: &ProcessedDomains,
    ) -> Result<[GroupedChildEvaluationToken; 2]> {
        self.mint_child_tokens(parent_index, first, second, GroupedSplitKind::PhaseSplit)
    }

    fn mint_child_tokens(
        &self,
        parent_index: usize,
        first: &ProcessedDomains,
        second: &ProcessedDomains,
        split_kind: GroupedSplitKind,
    ) -> Result<[GroupedChildEvaluationToken; 2]> {
        let lease_authority = self.authority.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "cannot mint grouped child tokens from an empty/unleased batch".to_string(),
            )
        })?;
        let parent_domain_id = *self.domain_ids.get(parent_index).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "grouped child token parent index {parent_index} is out of range"
            ))
        })?;
        let parent_metadata = self.domains.metadata.get(parent_index).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "grouped child token missing parent metadata at index {parent_index}"
            ))
        })?;
        let parent = GroupedLeasedDomain {
            id: parent_domain_id,
            summary: *self.summaries()?.get(parent_index).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "grouped child token missing parent summary at index {parent_index}"
                ))
            })?,
            input_lowers: input_row(
                &self.domains.input_lowers,
                self.domains.batch_size,
                parent_index,
                "token parent lower",
            )?,
            input_uppers: input_row(
                &self.domains.input_uppers,
                self.domains.batch_size,
                parent_index,
                "token parent upper",
            )?,
            depth: parent_metadata.depth(),
            constraints: parent_metadata.constraints().to_vec(),
        };
        let child_domains = [
            describe_processed_child(first, "token first child")?,
            describe_processed_child(second, "token second child")?,
        ];
        match split_kind {
            GroupedSplitKind::InputBisection => {
                validate_input_bisection_domains(&parent, &child_domains)?;
            }
            GroupedSplitKind::PhaseSplit => {
                validate_phase_split_domains(&parent, &child_domains)?;
            }
        }

        let [first_domain, second_domain] = child_domains;
        let make_token = |child_slot, domain| GroupedChildEvaluationToken {
            authority: GroupedChildAuthority {
                queue_token: Arc::clone(&lease_authority.queue_token),
                lease_id: lease_authority.lease_id,
                parent_domain_id,
                split_kind,
                child_slot,
                spec_fingerprint: self.layout.spec_fingerprint(),
            },
            domain,
            layout: self.layout.clone(),
        };
        Ok([make_token(0, first_domain), make_token(1, second_domain)])
    }

    pub(super) fn new(
        domains: PickedDomains,
        row_bounds: PackedGroupedBounds,
        layout: GroupedDisjunctiveLayout,
        domain_ids: Vec<GroupedDomainId>,
        authority: Option<(u64, Arc<()>)>,
    ) -> Self {
        Self {
            domains,
            row_bounds,
            layout,
            domain_ids,
            authority: authority.map(|(lease_id, queue_token)| GroupedLeaseAuthority {
                lease_id,
                queue_token,
            }),
        }
    }

    /// Test-only corruption hook proving the lease is bound to row summaries.
    #[cfg(test)]
    pub(super) fn swap_rows_for_test(&mut self, left: usize, right: usize) {
        let rows = self.layout.row_count();
        let lowers = self
            .row_bounds
            .row_lowers
            .as_slice_mut()
            .expect("test packed lowers contiguous");
        let uppers = self
            .row_bounds
            .row_uppers
            .as_slice_mut()
            .expect("test packed uppers contiguous");
        for row in 0..rows {
            lowers.swap(left * rows + row, right * rows + row);
            uppers.swap(left * rows + row, right * rows + row);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupedChildDisposition {
    Queued,
    Verified,
    CertifiedEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupedSplitKind {
    InputBisection,
    PhaseSplit,
}

/// Canonical child identity captured before evaluation.
#[derive(Debug, Clone)]
struct GroupedChildDomain {
    input_lowers: ArrayD<f32>,
    input_uppers: ArrayD<f32>,
    depth: usize,
    constraints: Vec<ConstraintTuple>,
}

/// Unforgeable association between one child and the leased search obligation
/// it came from.
#[derive(Debug)]
struct GroupedChildAuthority {
    queue_token: Arc<()>,
    lease_id: u64,
    parent_domain_id: GroupedDomainId,
    split_kind: GroupedSplitKind,
    child_slot: usize,
    spec_fingerprint: GroupedSpecFingerprint,
}

/// Non-cloneable token minted from a leased parent before child evaluation.
///
/// The future grouped evaluator must consume this token and return
/// [`EvaluatedGroupedChild`]. Raw row bounds cannot directly carry a queued or
/// verified verdict.
#[derive(Debug)]
pub(crate) struct GroupedChildEvaluationToken {
    authority: GroupedChildAuthority,
    domain: GroupedChildDomain,
    layout: GroupedDisjunctiveLayout,
}

/// Opaque evaluator result bound to exactly one pre-minted child token.
///
/// All fields and the production constructor are private. This prevents a
/// caller from combining the processed domain for one child with row bounds
/// evaluated for another child or objective.
#[derive(Debug)]
pub(crate) struct EvaluatedGroupedChild {
    authority: GroupedChildAuthority,
    layout: GroupedDisjunctiveLayout,
    domain: GroupedChildDomain,
    processed: Box<ProcessedDomains>,
    row_bounds: Option<PackedGroupedBounds>,
    disposition: GroupedChildDisposition,
}

/// Non-cloneable authority minted before an initial grouped queue batch is
/// evaluated.
#[derive(Debug)]
pub(crate) struct GroupedRootEvaluationToken {
    queue_token: Arc<()>,
    layout: GroupedDisjunctiveLayout,
    domains: Vec<GroupedChildDomain>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupedRootDisposition {
    Queued,
    Verified,
    CertifiedEmpty,
    UnresolvedDropped,
}

/// Opaque trusted-evaluator result for initial grouped queue entries.
#[derive(Debug)]
pub(crate) struct EvaluatedGroupedRoots {
    entries: Vec<SealedGroupedQueueEntry>,
}

/// The only payload accepted by the private grouped queue append path.
///
/// Construction is restricted to trusted root evaluation or a fully validated
/// child resolution. `ProcessedDomains::keep_mask` is always neutral
/// (`[true]`); only `disposition` may discharge or drop an obligation.
#[derive(Debug)]
pub(super) struct SealedGroupedQueueEntry {
    queue_token: Arc<()>,
    layout: GroupedDisjunctiveLayout,
    domain: GroupedChildDomain,
    processed: Box<ProcessedDomains>,
    row_bounds: PackedGroupedBounds,
    disposition: GroupedRootDisposition,
}

impl SealedGroupedQueueEntry {
    /// Open a validated queued entry for the storage module. No raw caller can
    /// construct this opaque type.
    pub(super) fn into_queued_payload(self) -> Result<(ProcessedDomains, PackedGroupedBounds)> {
        if self.disposition != GroupedRootDisposition::Queued {
            return Err(NyError::InternalError(
                "attempted to append a non-queued sealed grouped entry".to_string(),
            ));
        }
        let summary = self.row_bounds.summaries(&self.layout, 1)?[0];
        let mut processed = *self.processed;
        processed.global_lbs[0] = summary.lower_margin();
        processed.global_ubs[0] = summary.upper_margin();
        processed.metadata[0].update_bounds(summary.lower_margin(), summary.upper_margin())?;
        Ok((processed, self.row_bounds))
    }
}

impl GroupedChildEvaluationToken {
    /// Trusted evaluator boundary. Keep private: the production grouped
    /// evaluator will live below this module and consume the token while
    /// computing `processed` and `row_bounds` for its embedded domain/spec.
    fn seal_evaluation(
        self,
        evaluator_layout: &GroupedDisjunctiveLayout,
        processed: ProcessedDomains,
        row_bounds: Option<PackedGroupedBounds>,
        disposition: GroupedChildDisposition,
    ) -> Result<EvaluatedGroupedChild> {
        if evaluator_layout.spec_fingerprint() != self.authority.spec_fingerprint
            || evaluator_layout != &self.layout
        {
            return Err(NyError::InvalidSpec(
                "grouped child evaluator objective/spec fingerprint mismatch".to_string(),
            ));
        }

        let evaluated_domain =
            describe_processed_child(&processed, "sealed child evaluator output")?;
        if !self.domain.bitwise_eq(&evaluated_domain) {
            return Err(NyError::InvalidSpec(
                "grouped child evaluator returned a different canonical child domain".to_string(),
            ));
        }

        match (disposition, row_bounds.as_ref()) {
            (GroupedChildDisposition::Queued, Some(bounds)) => {
                bounds.summaries(&self.layout, 1)?;
            }
            (GroupedChildDisposition::Verified, Some(bounds)) => {
                if !bounds.summaries(&self.layout, 1)?[0].is_verified() {
                    return Err(NyError::InvalidSpec(
                        "grouped child evaluator cannot seal Verified: at least one clause \
                         remains unresolved"
                            .to_string(),
                    ));
                }
            }
            (GroupedChildDisposition::CertifiedEmpty, None) => {}
            (GroupedChildDisposition::Queued | GroupedChildDisposition::Verified, None) => {
                return Err(NyError::InvalidSpec(
                    "grouped child evaluator omitted required row bounds".to_string(),
                ));
            }
            (GroupedChildDisposition::CertifiedEmpty, Some(_)) => {
                return Err(NyError::InvalidSpec(
                    "grouped child evaluator attached row bounds to CertifiedEmpty".to_string(),
                ));
            }
        }

        Ok(EvaluatedGroupedChild {
            authority: self.authority,
            layout: self.layout,
            domain: self.domain,
            processed: Box::new(processed),
            row_bounds,
            disposition,
        })
    }
}

impl GroupedRootEvaluationToken {
    /// Trusted root-evaluator boundary. Keep private for the same reason as
    /// child sealing: raw processed payloads and row bounds must never acquire
    /// proof authority outside the evaluator that actually computed them.
    fn seal_evaluation(
        self,
        evaluator_queue_token: &Arc<()>,
        evaluator_layout: &GroupedDisjunctiveLayout,
        processed: ProcessedDomains,
        row_bounds: PackedGroupedBounds,
        dispositions: Vec<GroupedRootDisposition>,
    ) -> Result<EvaluatedGroupedRoots> {
        if !Arc::ptr_eq(&self.queue_token, evaluator_queue_token) {
            return Err(NyError::InvalidSpec(
                "grouped root evaluator belongs to a different queue".to_string(),
            ));
        }
        if evaluator_layout != &self.layout
            || evaluator_layout.spec_fingerprint() != self.layout.spec_fingerprint()
        {
            return Err(NyError::InvalidSpec(
                "grouped root evaluator objective/spec fingerprint mismatch".to_string(),
            ));
        }

        let evaluated_domains =
            describe_processed_roots(&processed, "sealed root evaluator output")?;
        if evaluated_domains.len() != self.domains.len() || dispositions.len() != self.domains.len()
        {
            return Err(NyError::InvalidSpec(format!(
                "grouped root evaluator length mismatch: token={}, evaluated={}, dispositions={}",
                self.domains.len(),
                evaluated_domains.len(),
                dispositions.len()
            )));
        }
        for (index, (canonical, evaluated)) in
            self.domains.iter().zip(&evaluated_domains).enumerate()
        {
            if !canonical.bitwise_eq(evaluated) {
                return Err(NyError::InvalidSpec(format!(
                    "grouped root evaluator changed canonical domain {index}"
                )));
            }
        }

        let summaries = row_bounds.summaries(&self.layout, self.domains.len())?;
        let mut entries = Vec::with_capacity(self.domains.len());
        for (index, ((domain, disposition), summary)) in self
            .domains
            .into_iter()
            .zip(dispositions)
            .zip(summaries)
            .enumerate()
        {
            if disposition == GroupedRootDisposition::Verified && !summary.is_verified() {
                return Err(NyError::InvalidSpec(format!(
                    "grouped root evaluator cannot seal domain {index} Verified: \
                     at least one clause remains unresolved"
                )));
            }
            entries.push(SealedGroupedQueueEntry {
                queue_token: Arc::clone(&self.queue_token),
                layout: self.layout.clone(),
                domain,
                processed: Box::new(single_processed_domain(&processed, index)?),
                row_bounds: single_grouped_bounds(&row_bounds, index)?,
                disposition,
            });
        }
        Ok(EvaluatedGroupedRoots { entries })
    }
}

// Test-only stand-ins for the trusted grouped evaluator. They deliberately
// remain visible only to the parent DomainList module in test builds; no
// crate-wide raw verdict-bearing constructor exists in production.
#[cfg(test)]
pub(super) fn evaluate_grouped_queued_for_test(
    token: GroupedChildEvaluationToken,
    evaluator_layout: &GroupedDisjunctiveLayout,
    processed: ProcessedDomains,
    row_bounds: PackedGroupedBounds,
) -> Result<EvaluatedGroupedChild> {
    token.seal_evaluation(
        evaluator_layout,
        processed,
        Some(row_bounds),
        GroupedChildDisposition::Queued,
    )
}

#[cfg(test)]
pub(super) fn evaluate_grouped_verified_for_test(
    token: GroupedChildEvaluationToken,
    evaluator_layout: &GroupedDisjunctiveLayout,
    processed: ProcessedDomains,
    row_bounds: PackedGroupedBounds,
) -> Result<EvaluatedGroupedChild> {
    token.seal_evaluation(
        evaluator_layout,
        processed,
        Some(row_bounds),
        GroupedChildDisposition::Verified,
    )
}

#[cfg(test)]
pub(super) fn evaluate_grouped_empty_for_test(
    token: GroupedChildEvaluationToken,
    evaluator_layout: &GroupedDisjunctiveLayout,
    processed: ProcessedDomains,
) -> Result<EvaluatedGroupedChild> {
    token.seal_evaluation(
        evaluator_layout,
        processed,
        None,
        GroupedChildDisposition::CertifiedEmpty,
    )
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) enum TestGroupedRootDisposition {
    Queued,
    Verified,
    CertifiedEmpty,
    UnresolvedDropped,
}

#[cfg(test)]
pub(super) fn evaluate_grouped_roots_for_test(
    evaluator_queue: &super::DomainList,
    token: GroupedRootEvaluationToken,
    evaluator_layout: &GroupedDisjunctiveLayout,
    processed: ProcessedDomains,
    row_bounds: PackedGroupedBounds,
    dispositions: Vec<TestGroupedRootDisposition>,
) -> Result<EvaluatedGroupedRoots> {
    let grouped = evaluator_queue.grouped.as_ref().ok_or_else(|| {
        NyError::InvalidSpec(
            "test grouped root evaluator called with a scalar DomainList".to_string(),
        )
    })?;
    token.seal_evaluation(
        &grouped.queue_token,
        evaluator_layout,
        processed,
        row_bounds,
        dispositions
            .into_iter()
            .map(|disposition| match disposition {
                TestGroupedRootDisposition::Queued => GroupedRootDisposition::Queued,
                TestGroupedRootDisposition::Verified => GroupedRootDisposition::Verified,
                TestGroupedRootDisposition::CertifiedEmpty => {
                    GroupedRootDisposition::CertifiedEmpty
                }
                TestGroupedRootDisposition::UnresolvedDropped => {
                    GroupedRootDisposition::UnresolvedDropped
                }
            })
            .collect(),
    )
}

/// Typed, structurally validated outcome for one leased parent.
#[derive(Debug)]
pub(crate) enum GroupedParentOutcome {
    /// The parent's immutable lease summary discharges every clause.
    Verified,
    /// Exactly two input boxes form an exact finite bisection of the parent.
    InputBisection {
        children: Box<[EvaluatedGroupedChild; 2]>,
    },
    /// Exactly two histories append opposite phases for one split predicate.
    PhaseSplit {
        children: Box<[EvaluatedGroupedChild; 2]>,
    },
    /// Search space was lost; exhaustion must remain Unknown.
    UnresolvedDropped,
}

impl GroupedParentOutcome {
    pub(crate) fn input_bisection(
        first: EvaluatedGroupedChild,
        second: EvaluatedGroupedChild,
    ) -> Self {
        Self::InputBisection {
            children: Box::new([first, second]),
        }
    }

    pub(crate) fn phase_split(first: EvaluatedGroupedChild, second: EvaluatedGroupedChild) -> Self {
        Self::PhaseSplit {
            children: Box::new([first, second]),
        }
    }
}

/// One ordered resolution entry, bound to a queue-issued domain ID.
#[derive(Debug)]
pub(crate) struct GroupedParentResolution {
    domain_id: GroupedDomainId,
    outcome: GroupedParentOutcome,
}

impl GroupedParentResolution {
    pub(crate) fn new(domain_id: GroupedDomainId, outcome: GroupedParentOutcome) -> Self {
        Self { domain_id, outcome }
    }
}

/// Validated accounting returned when a picked grouped batch leaves the
/// executor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GroupedBatchCompletion {
    pub(crate) verified: usize,
    pub(crate) replaced_by_children: usize,
    pub(crate) unresolved_dropped: usize,
}

/// Fail-closed queue-exhaustion classification for the future grouped executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupedQueueStatus {
    /// Queued or in-flight domains remain.
    Pending,
    /// The queue drained with every parent accounted for and no eviction.
    ExhaustedVerified,
    /// Search space was dropped or state accounting became inconclusive.
    ExhaustedUnknown,
}

/// Optional clause-aware sidecar storage owned by `DomainList`.
pub(super) struct GroupedDisjunctiveStorage {
    pub(super) layout: GroupedDisjunctiveLayout,
    pub(super) row_lowers: Box<dyn TensorStorage + Send>,
    pub(super) row_uppers: Box<dyn TensorStorage + Send>,
    pub(super) next_lease_id: u64,
    pub(super) next_domain_id: u64,
    pub(super) active_leases: HashMap<u64, GroupedLeaseState>,
    pub(super) unresolved_dropped: usize,
    /// Prevent an uninitialized empty queue from being mistaken for a proof.
    pub(super) search_started: bool,
    /// Unforgeable in-process identity preventing lease IDs from one queue
    /// being completed against another queue with the same layout.
    pub(super) queue_token: Arc<()>,
}

/// Canonical immutable payload for one picked parent.
pub(super) struct GroupedLeasedDomain {
    id: GroupedDomainId,
    summary: GroupedBoundSummary,
    input_lowers: ArrayD<f32>,
    input_uppers: ArrayD<f32>,
    depth: usize,
    constraints: Vec<ConstraintTuple>,
}

/// Search-space accounting attached to one picked batch. Decisions are made
/// from this canonical payload, never caller-owned mutable rows.
pub(super) struct GroupedLeaseState {
    domains: Vec<GroupedLeasedDomain>,
}

impl GroupedBoundSummary {
    fn bitwise_eq(self, other: Self) -> bool {
        self.lower_margin.to_bits() == other.lower_margin.to_bits()
            && self.upper_margin.to_bits() == other.upper_margin.to_bits()
    }
}

fn input_row(
    values: &ArrayD<f32>,
    expected_batch: usize,
    domain: usize,
    label: &str,
) -> Result<ArrayD<f32>> {
    if values.ndim() == 0 || values.shape()[0] != expected_batch {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {label} batch mismatch: expected {expected_batch}, shape={:?}",
            values.shape()
        )));
    }
    Ok(values.index_axis(Axis(0), domain).to_owned().into_dyn())
}

fn validate_finite_box(lowers: &ArrayD<f32>, uppers: &ArrayD<f32>, context: &str) -> Result<()> {
    if lowers.shape() != uppers.shape() {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} input shape mismatch: lower={:?}, upper={:?}",
            lowers.shape(),
            uppers.shape()
        )));
    }
    for (dimension, (&lower, &upper)) in lowers.iter().zip(uppers.iter()).enumerate() {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "grouped DomainList {context} input dimension {dimension} is non-finite \
                 [{lower}, {upper}]"
            )));
        }
        if lower > upper {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList {context} input dimension {dimension} is inverted \
                 [{lower}, {upper}]"
            )));
        }
    }
    Ok(())
}

fn validate_history(history: &[ConstraintTuple], context: &str) -> Result<()> {
    for (index, (node, _neuron, _phase, split_point)) in history.iter().enumerate() {
        if node.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList {context} constraint {index} has an empty node name"
            )));
        }
        if split_point.is_some_and(|point| !point.is_finite()) {
            return Err(NyError::NumericalInstability(format!(
                "grouped DomainList {context} constraint {index} has non-finite split point"
            )));
        }
    }
    Ok(())
}

impl GroupedLeaseState {
    pub(super) fn new(
        picked: &PickedDomains,
        domain_ids: &[GroupedDomainId],
        summaries: &[GroupedBoundSummary],
    ) -> Result<Self> {
        if picked.batch_size != domain_ids.len()
            || picked.batch_size != summaries.len()
            || picked.batch_size != picked.metadata.len()
        {
            return Err(NyError::InternalError(format!(
                "grouped DomainList lease payload length mismatch: batch={}, ids={}, \
                 summaries={}, metadata={}",
                picked.batch_size,
                domain_ids.len(),
                summaries.len(),
                picked.metadata.len()
            )));
        }

        let mut domains = Vec::with_capacity(picked.batch_size);
        for domain in 0..picked.batch_size {
            let input_lowers = input_row(
                &picked.input_lowers,
                picked.batch_size,
                domain,
                "leased lower",
            )?;
            let input_uppers = input_row(
                &picked.input_uppers,
                picked.batch_size,
                domain,
                "leased upper",
            )?;
            validate_finite_box(&input_lowers, &input_uppers, "leased parent")?;
            let metadata = &picked.metadata[domain];
            validate_history(metadata.constraints(), "leased parent")?;
            domains.push(GroupedLeasedDomain {
                id: domain_ids[domain],
                summary: summaries[domain],
                input_lowers,
                input_uppers,
                depth: metadata.depth(),
                constraints: metadata.constraints().to_vec(),
            });
        }
        Ok(Self { domains })
    }

    fn validate_picked(&self, picked: &PickedGroupedDomains) -> Result<()> {
        if self.domains.len() != picked.domains.batch_size
            || picked.domain_ids.len() != self.domains.len()
        {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList consumed lease length mismatch: lease={}, picked={}, ids={}",
                self.domains.len(),
                picked.domains.batch_size,
                picked.domain_ids.len()
            )));
        }
        let summaries = picked.summaries()?;
        for (domain, leased) in self.domains.iter().enumerate() {
            if picked.domain_ids[domain] != leased.id {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList domain ID mismatch at position {domain}"
                )));
            }
            if !summaries[domain].bitwise_eq(leased.summary) {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList row summary mismatch for leased domain {domain}"
                )));
            }
            let input_lowers = input_row(
                &picked.domains.input_lowers,
                picked.domains.batch_size,
                domain,
                "picked lower",
            )?;
            let input_uppers = input_row(
                &picked.domains.input_uppers,
                picked.domains.batch_size,
                domain,
                "picked upper",
            )?;
            let metadata = picked.domains.metadata.get(domain).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "grouped DomainList picked metadata missing domain {domain}"
                ))
            })?;
            if !arrays_bitwise_eq(&input_lowers, &leased.input_lowers)
                || !arrays_bitwise_eq(&input_uppers, &leased.input_uppers)
                || metadata.depth() != leased.depth
                || !constraints_bitwise_eq(metadata.constraints(), &leased.constraints)
            {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList picked payload mismatch for leased domain {domain}"
                )));
            }
        }
        Ok(())
    }
}

fn arrays_bitwise_eq(left: &ArrayD<f32>, right: &ArrayD<f32>) -> bool {
    left.shape() == right.shape()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn constraints_bitwise_eq(left: &[ConstraintTuple], right: &[ConstraintTuple]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.0 == right.0
                && left.1 == right.1
                && left.2 == right.2
                && match (left.3, right.3) {
                    (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
                    (None, None) => true,
                    _ => false,
                }
        })
}

impl GroupedChildDomain {
    fn bitwise_eq(&self, other: &Self) -> bool {
        arrays_bitwise_eq(&self.input_lowers, &other.input_lowers)
            && arrays_bitwise_eq(&self.input_uppers, &other.input_uppers)
            && self.depth == other.depth
            && constraints_bitwise_eq(&self.constraints, &other.constraints)
    }
}

fn describe_processed_roots(
    processed: &ProcessedDomains,
    context: &str,
) -> Result<Vec<GroupedChildDomain>> {
    let batch_size = processed.global_lbs.len();
    if batch_size == 0 {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} cannot be empty"
        )));
    }
    if processed.global_ubs.len() != batch_size
        || processed.metadata.len() != batch_size
        || processed.keep_mask.len() != batch_size
        || processed.keep_mask.iter().any(|keep| !keep)
    {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} must have one neutral keep=true payload per domain \
             (lower={}, upper={}, metadata={}, keep_mask={:?})",
            batch_size,
            processed.global_ubs.len(),
            processed.metadata.len(),
            processed.keep_mask
        )));
    }

    let mut domains = Vec::with_capacity(batch_size);
    for domain in 0..batch_size {
        let input_lowers = input_row(&processed.input_lowers, batch_size, domain, "root lower")?;
        let input_uppers = input_row(&processed.input_uppers, batch_size, domain, "root upper")?;
        validate_finite_box(&input_lowers, &input_uppers, context)?;
        let metadata = &processed.metadata[domain];
        validate_history(metadata.constraints(), context)?;
        domains.push(GroupedChildDomain {
            input_lowers,
            input_uppers,
            depth: metadata.depth(),
            constraints: metadata.constraints().to_vec(),
        });
    }
    Ok(domains)
}

fn batched_single_row(
    values: &ArrayD<f32>,
    expected_batch: usize,
    domain: usize,
    context: &str,
) -> Result<ArrayD<f32>> {
    if values.ndim() == 0 || values.shape()[0] != expected_batch || domain >= expected_batch {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} batch mismatch: expected {expected_batch}, \
             domain={domain}, shape={:?}",
            values.shape()
        )));
    }
    Ok(values
        .index_axis(Axis(0), domain)
        .insert_axis(Axis(0))
        .to_owned()
        .into_dyn())
}

fn single_processed_domain(
    processed: &ProcessedDomains,
    domain: usize,
) -> Result<ProcessedDomains> {
    let batch_size = processed.global_lbs.len();
    let mut layer_lowers = HashMap::with_capacity(processed.layer_lowers.len());
    for (name, values) in &processed.layer_lowers {
        layer_lowers.insert(
            name.clone(),
            batched_single_row(values, batch_size, domain, "root layer lower")?,
        );
    }
    let mut layer_uppers = HashMap::with_capacity(processed.layer_uppers.len());
    for (name, values) in &processed.layer_uppers {
        layer_uppers.insert(
            name.clone(),
            batched_single_row(values, batch_size, domain, "root layer upper")?,
        );
    }
    Ok(ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: batched_single_row(
            &processed.input_lowers,
            batch_size,
            domain,
            "root input lower",
        )?,
        input_uppers: batched_single_row(
            &processed.input_uppers,
            batch_size,
            domain,
            "root input upper",
        )?,
        global_lbs: vec![*processed.global_lbs.get(domain).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "grouped DomainList root lower bound missing domain {domain}"
            ))
        })?],
        global_ubs: vec![*processed.global_ubs.get(domain).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "grouped DomainList root upper bound missing domain {domain}"
            ))
        })?],
        metadata: vec![processed.metadata.get(domain).cloned().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "grouped DomainList root metadata missing domain {domain}"
            ))
        })?],
        keep_mask: vec![true],
    })
}

fn single_grouped_bounds(
    row_bounds: &PackedGroupedBounds,
    domain: usize,
) -> Result<PackedGroupedBounds> {
    let batch_size = row_bounds.row_lowers.shape().first().copied().unwrap_or(0);
    Ok(PackedGroupedBounds::new(
        batched_single_row(&row_bounds.row_lowers, batch_size, domain, "root row lower")?,
        batched_single_row(&row_bounds.row_uppers, batch_size, domain, "root row upper")?,
    ))
}

fn describe_processed_child(
    processed: &ProcessedDomains,
    context: &str,
) -> Result<GroupedChildDomain> {
    if processed.global_lbs.len() != 1
        || processed.global_ubs.len() != 1
        || processed.metadata.len() != 1
        || processed.keep_mask != [true]
    {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} child must contain exactly one neutral \
             keep=true payload (lower={}, upper={}, metadata={}, keep_mask={:?})",
            processed.global_lbs.len(),
            processed.global_ubs.len(),
            processed.metadata.len(),
            processed.keep_mask
        )));
    }
    let input_lowers = input_row(&processed.input_lowers, 1, 0, "child lower")?;
    let input_uppers = input_row(&processed.input_uppers, 1, 0, "child upper")?;
    validate_finite_box(&input_lowers, &input_uppers, context)?;
    let metadata = &processed.metadata[0];
    validate_history(metadata.constraints(), context)?;

    Ok(GroupedChildDomain {
        input_lowers,
        input_uppers,
        depth: metadata.depth(),
        constraints: metadata.constraints().to_vec(),
    })
}

struct GroupedChildDescriptor {
    domain: GroupedChildDomain,
    disposition: GroupedChildDisposition,
}

#[allow(clippy::too_many_arguments)]
fn describe_child(
    child: &EvaluatedGroupedChild,
    parent: &GroupedLeasedDomain,
    lease_id: u64,
    queue_token: &Arc<()>,
    split_kind: GroupedSplitKind,
    child_slot: usize,
    layout: &GroupedDisjunctiveLayout,
    context: &str,
) -> Result<GroupedChildDescriptor> {
    let authority = &child.authority;
    if !Arc::ptr_eq(&authority.queue_token, queue_token)
        || authority.lease_id != lease_id
        || authority.parent_domain_id != parent.id
        || authority.split_kind != split_kind
        || authority.child_slot != child_slot
        || authority.spec_fingerprint != layout.spec_fingerprint()
        || &child.layout != layout
    {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} evaluated-child authority mismatch"
        )));
    }

    let evaluated_domain = describe_processed_child(&child.processed, context)?;
    if !child.domain.bitwise_eq(&evaluated_domain) {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} evaluated-child domain/token mismatch"
        )));
    }

    match (child.disposition, child.row_bounds.as_ref()) {
        (GroupedChildDisposition::Queued, Some(row_bounds)) => {
            row_bounds.summaries(layout, 1)?;
        }
        (GroupedChildDisposition::Verified, Some(row_bounds)) => {
            if !row_bounds.summaries(layout, 1)?[0].is_verified() {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList {context} child cannot be marked Verified: \
                     at least one clause remains unresolved"
                )));
            }
        }
        (GroupedChildDisposition::CertifiedEmpty, None) => {}
        (GroupedChildDisposition::Queued | GroupedChildDisposition::Verified, None) => {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList {context} evaluated child lost row bounds"
            )));
        }
        (GroupedChildDisposition::CertifiedEmpty, Some(_)) => {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList {context} CertifiedEmpty child has row bounds"
            )));
        }
    }

    Ok(GroupedChildDescriptor {
        domain: child.domain.clone(),
        disposition: child.disposition,
    })
}

fn validate_child_depth(
    parent: &GroupedLeasedDomain,
    child: &GroupedChildDomain,
    context: &str,
) -> Result<()> {
    let expected = parent.depth.checked_add(1).ok_or_else(|| {
        NyError::InternalError("grouped DomainList parent depth overflow".to_string())
    })?;
    if child.depth != expected {
        return Err(NyError::InvalidSpec(format!(
            "grouped DomainList {context} child depth {} != parent depth {} + 1",
            child.depth, parent.depth
        )));
    }
    Ok(())
}

fn validate_input_bisection_domains(
    parent: &GroupedLeasedDomain,
    children: &[GroupedChildDomain; 2],
) -> Result<()> {
    let [first, second] = children;
    validate_child_depth(parent, first, "input-bisection first")?;
    validate_child_depth(parent, second, "input-bisection second")?;
    if !constraints_bitwise_eq(&first.constraints, &parent.constraints)
        || !constraints_bitwise_eq(&second.constraints, &parent.constraints)
    {
        return Err(NyError::InvalidSpec(
            "grouped DomainList input-bisection child history must equal parent history"
                .to_string(),
        ));
    }

    let parent_lowers: Vec<f32> = parent.input_lowers.iter().copied().collect();
    let parent_uppers: Vec<f32> = parent.input_uppers.iter().copied().collect();
    let first_lowers: Vec<f32> = first.input_lowers.iter().copied().collect();
    let first_uppers: Vec<f32> = first.input_uppers.iter().copied().collect();
    let second_lowers: Vec<f32> = second.input_lowers.iter().copied().collect();
    let second_uppers: Vec<f32> = second.input_uppers.iter().copied().collect();
    if first.input_lowers.shape() != parent.input_lowers.shape()
        || second.input_lowers.shape() != parent.input_lowers.shape()
    {
        return Err(NyError::InvalidSpec(
            "grouped DomainList input-bisection child shape differs from parent".to_string(),
        ));
    }

    let mut split_dimension = None;
    for dimension in 0..parent_lowers.len() {
        let parent_lower = parent_lowers[dimension];
        let parent_upper = parent_uppers[dimension];
        let unchanged = first_lowers[dimension] == parent_lower
            && first_uppers[dimension] == parent_upper
            && second_lowers[dimension] == parent_lower
            && second_uppers[dimension] == parent_upper;
        if unchanged {
            continue;
        }

        let first_left = first_lowers[dimension] == parent_lower
            && second_uppers[dimension] == parent_upper
            && first_uppers[dimension] == second_lowers[dimension]
            && parent_lower < first_uppers[dimension]
            && first_uppers[dimension] < parent_upper;
        let second_left = second_lowers[dimension] == parent_lower
            && first_uppers[dimension] == parent_upper
            && second_uppers[dimension] == first_lowers[dimension]
            && parent_lower < second_uppers[dimension]
            && second_uppers[dimension] < parent_upper;
        if (!first_left && !second_left) || split_dimension.replace(dimension).is_some() {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList input children are not one exact finite bisection \
                 (dimension {dimension})"
            )));
        }
    }
    if split_dimension.is_none() {
        return Err(NyError::InvalidSpec(
            "grouped DomainList input children duplicate the parent instead of bisecting it"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_input_bisection(
    parent: &GroupedLeasedDomain,
    children: &[EvaluatedGroupedChild; 2],
    layout: &GroupedDisjunctiveLayout,
    lease_id: u64,
    queue_token: &Arc<()>,
) -> Result<()> {
    let first = describe_child(
        &children[0],
        parent,
        lease_id,
        queue_token,
        GroupedSplitKind::InputBisection,
        0,
        layout,
        "input-bisection first",
    )?;
    let second = describe_child(
        &children[1],
        parent,
        lease_id,
        queue_token,
        GroupedSplitKind::InputBisection,
        1,
        layout,
        "input-bisection second",
    )?;
    if first.disposition == GroupedChildDisposition::CertifiedEmpty
        || second.disposition == GroupedChildDisposition::CertifiedEmpty
    {
        return Err(NyError::InvalidSpec(
            "grouped DomainList finite input bisection cannot use CertifiedEmpty".to_string(),
        ));
    }
    validate_input_bisection_domains(parent, &[first.domain, second.domain])
}

fn same_phase_predicate(left: &ConstraintTuple, right: &ConstraintTuple) -> bool {
    left.0 == right.0
        && left.1 == right.1
        && match (left.3, right.3) {
            (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
            (None, None) => true,
            _ => false,
        }
}

fn phase_is_structurally_empty(
    parent_history: &[ConstraintTuple],
    appended: &ConstraintTuple,
) -> bool {
    parent_history
        .iter()
        .any(|existing| same_phase_predicate(existing, appended) && existing.2 != appended.2)
}

fn validate_phase_split_domains(
    parent: &GroupedLeasedDomain,
    children: &[GroupedChildDomain; 2],
) -> Result<()> {
    let [first, second] = children;
    validate_child_depth(parent, first, "phase-split first")?;
    validate_child_depth(parent, second, "phase-split second")?;
    if !arrays_bitwise_eq(&first.input_lowers, &parent.input_lowers)
        || !arrays_bitwise_eq(&first.input_uppers, &parent.input_uppers)
        || !arrays_bitwise_eq(&second.input_lowers, &parent.input_lowers)
        || !arrays_bitwise_eq(&second.input_uppers, &parent.input_uppers)
    {
        return Err(NyError::InvalidSpec(
            "grouped DomainList phase-split children must retain the exact parent input box"
                .to_string(),
        ));
    }
    let parent_len = parent.constraints.len();
    if first.constraints.len() != parent_len + 1
        || second.constraints.len() != parent_len + 1
        || !constraints_bitwise_eq(&first.constraints[..parent_len], &parent.constraints)
        || !constraints_bitwise_eq(&second.constraints[..parent_len], &parent.constraints)
    {
        return Err(NyError::InvalidSpec(
            "grouped DomainList phase-split history must be exact parent history plus one"
                .to_string(),
        ));
    }
    let first_phase = &first.constraints[parent_len];
    let second_phase = &second.constraints[parent_len];
    if !same_phase_predicate(first_phase, second_phase) || first_phase.2 == second_phase.2 {
        return Err(NyError::InvalidSpec(
            "grouped DomainList phase children must append opposite phases for the \
             same node/index/split point"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_phase_split(
    parent: &GroupedLeasedDomain,
    children: &[EvaluatedGroupedChild; 2],
    layout: &GroupedDisjunctiveLayout,
    lease_id: u64,
    queue_token: &Arc<()>,
) -> Result<()> {
    let first = describe_child(
        &children[0],
        parent,
        lease_id,
        queue_token,
        GroupedSplitKind::PhaseSplit,
        0,
        layout,
        "phase-split first",
    )?;
    let second = describe_child(
        &children[1],
        parent,
        lease_id,
        queue_token,
        GroupedSplitKind::PhaseSplit,
        1,
        layout,
        "phase-split second",
    )?;
    let dispositions = [first.disposition, second.disposition];
    let domains = [first.domain, second.domain];
    validate_phase_split_domains(parent, &domains)?;
    let parent_len = parent.constraints.len();
    let first_phase = &domains[0].constraints[parent_len];
    let second_phase = &domains[1].constraints[parent_len];
    for (disposition, appended, label) in [
        (dispositions[0], first_phase, "first"),
        (dispositions[1], second_phase, "second"),
    ] {
        if disposition == GroupedChildDisposition::CertifiedEmpty
            && !phase_is_structurally_empty(&parent.constraints, appended)
        {
            return Err(NyError::InvalidSpec(format!(
                "grouped DomainList phase-split {label} child lacks a structural \
                 CertifiedEmpty witness"
            )));
        }
    }
    Ok(())
}

fn collect_queued_children(
    outcome: GroupedParentOutcome,
    queued: &mut Vec<SealedGroupedQueueEntry>,
) -> Result<()> {
    let children = match outcome {
        GroupedParentOutcome::InputBisection { children }
        | GroupedParentOutcome::PhaseSplit { children } => children,
        GroupedParentOutcome::Verified | GroupedParentOutcome::UnresolvedDropped => return Ok(()),
    };
    for child in *children {
        if child.disposition == GroupedChildDisposition::Queued {
            let row_bounds = child.row_bounds.ok_or_else(|| {
                NyError::InternalError(
                    "sealed queued grouped child lost its required row bounds".to_string(),
                )
            })?;
            queued.push(SealedGroupedQueueEntry {
                queue_token: child.authority.queue_token,
                layout: child.layout,
                domain: child.domain,
                processed: child.processed,
                row_bounds,
                disposition: GroupedRootDisposition::Queued,
            });
        }
    }
    Ok(())
}

/// Private proof state emitted only after every lease and exact-cover check
/// succeeds. Queue mutation consumes this state; callers cannot construct it.
struct SealedGroupedResolution {
    completion: GroupedBatchCompletion,
    next_unresolved_dropped: usize,
    queued_children: Vec<SealedGroupedQueueEntry>,
}

impl super::DomainList {
    /// Mint a queue/spec/domain-bound token before initial grouped evaluation.
    ///
    /// The raw payload defines only the search obligation. It must use a
    /// neutral keep mask and cannot discharge or drop any domain.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mint_grouped_root_evaluation_token(
        &self,
        roots: &ProcessedDomains,
    ) -> Result<GroupedRootEvaluationToken> {
        self.validate_grouped_alignment()?;
        let grouped = self.grouped.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "cannot mint grouped root token from a scalar DomainList".to_string(),
            )
        })?;
        Ok(GroupedRootEvaluationToken {
            queue_token: Arc::clone(&grouped.queue_token),
            layout: grouped.layout.clone(),
            domains: describe_processed_roots(roots, "root token payload")?,
        })
    }

    /// Accept an opaque trusted root-evaluator result.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn accept_grouped_root_evaluation(
        &mut self,
        evaluated: EvaluatedGroupedRoots,
    ) -> Result<()> {
        self.append_grouped_queue_entries(evaluated.entries)
    }

    /// Private grouped append boundary. Every accepted entry was sealed either
    /// by the trusted root evaluator or by a fully validated child resolution.
    fn append_grouped_queue_entries(
        &mut self,
        entries: Vec<SealedGroupedQueueEntry>,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.validate_grouped_alignment()?;
        let guarded_unresolved_dropped = {
            let grouped = self.grouped.as_ref().ok_or_else(|| {
                NyError::InvalidSpec(
                    "grouped queue entry cannot be appended to a scalar DomainList".to_string(),
                )
            })?;
            let mut unresolved_dropped = 0usize;
            let mut queued_guards = 0usize;
            for (index, entry) in entries.iter().enumerate() {
                if !Arc::ptr_eq(&entry.queue_token, &grouped.queue_token)
                    || entry.layout != grouped.layout
                    || entry.layout.spec_fingerprint() != grouped.layout.spec_fingerprint()
                {
                    return Err(NyError::InvalidSpec(format!(
                        "sealed grouped queue entry {index} authority mismatch"
                    )));
                }
                let evaluated_domain =
                    describe_processed_child(&entry.processed, "sealed queue entry")?;
                if !entry.domain.bitwise_eq(&evaluated_domain) {
                    return Err(NyError::InvalidSpec(format!(
                        "sealed grouped queue entry {index} domain mismatch"
                    )));
                }
                let summary = entry.row_bounds.summaries(&grouped.layout, 1)?[0];
                if entry.disposition == GroupedRootDisposition::Verified && !summary.is_verified() {
                    return Err(NyError::InvalidSpec(format!(
                        "sealed grouped queue entry {index} has an invalid Verified disposition"
                    )));
                }
                if entry.disposition == GroupedRootDisposition::UnresolvedDropped {
                    unresolved_dropped = unresolved_dropped.checked_add(1).ok_or_else(|| {
                        NyError::InternalError(
                            "grouped queue unresolved-drop delta overflow".to_string(),
                        )
                    })?;
                }
                if entry.disposition == GroupedRootDisposition::Queued {
                    queued_guards = queued_guards.checked_add(1).ok_or_else(|| {
                        NyError::InternalError(
                            "grouped queue append guard delta overflow".to_string(),
                        )
                    })?;
                }
            }
            let guarded_delta = unresolved_dropped
                .checked_add(queued_guards)
                .ok_or_else(|| {
                    NyError::InternalError("grouped queue guarded-drop delta overflow".to_string())
                })?;
            grouped
                .unresolved_dropped
                .checked_add(guarded_delta)
                .ok_or_else(|| {
                    NyError::InternalError(
                        "grouped DomainList unresolved-drop counter overflow".to_string(),
                    )
                })?
        };

        // Guard every queued obligation as dropped before storage mutation. A
        // successful append removes its guard; an error leaves Unknown (and
        // child resolution also keeps the parent lease live).
        let grouped = self.grouped.as_mut().ok_or_else(|| {
            NyError::InternalError("grouped queue append lost grouped DomainList state".to_string())
        })?;
        grouped.search_started = true;
        grouped.unresolved_dropped = guarded_unresolved_dropped;

        for entry in entries {
            if entry.disposition != GroupedRootDisposition::Queued {
                continue;
            }
            self.append_sealed_grouped_queued(entry)?;
            let grouped = self.grouped.as_mut().ok_or_else(|| {
                NyError::InternalError(
                    "grouped queue append lost state while removing guard".to_string(),
                )
            })?;
            grouped.unresolved_dropped =
                grouped.unresolved_dropped.checked_sub(1).ok_or_else(|| {
                    NyError::InternalError(
                        "grouped queue append guard counter underflow".to_string(),
                    )
                })?;
        }
        Ok(())
    }

    pub(super) fn validate_grouped_alignment(&self) -> Result<()> {
        let Some(grouped) = self.grouped.as_ref() else {
            return Ok(());
        };
        let domains = self.metadata.len();
        let input_lowers = self.input_lowers.len();
        let input_uppers = self.input_uppers.len();
        let global_lowers = self.global_lbs.len();
        let global_uppers = self.global_ubs.len();
        if grouped.row_lowers.len() != domains
            || grouped.row_uppers.len() != domains
            || input_lowers != domains
            || input_uppers != domains
            || global_lowers != domains
            || global_uppers != domains
        {
            return Err(NyError::InternalError(format!(
                "grouped DomainList storage misalignment: domains={domains}, \
                 row_lowers={}, row_uppers={}, input_lowers={input_lowers}, \
                 input_uppers={input_uppers}, global_lowers={global_lowers}, \
                 global_uppers={global_uppers}",
                grouped.row_lowers.len(),
                grouped.row_uppers.len()
            )));
        }
        for name in &self.config.layer_names {
            let lower_storage = self.layer_lowers.get(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "grouped DomainList missing lower storage for layer '{name}'"
                ))
            })?;
            let upper_storage = self.layer_uppers.get(name).ok_or_else(|| {
                NyError::InternalError(format!(
                    "grouped DomainList missing upper storage for layer '{name}'"
                ))
            })?;
            if lower_storage.len() != domains || upper_storage.len() != domains {
                return Err(NyError::InternalError(format!(
                    "grouped DomainList layer '{name}' misalignment: domains={domains}, \
                     lower={}, upper={}",
                    lower_storage.len(),
                    upper_storage.len()
                )));
            }
        }
        Ok(())
    }

    /// Atomically validate and resolve a picked grouped batch.
    ///
    /// The consumed, non-cloneable lease is checked against canonical domain
    /// IDs, row summaries, boxes, depths, and histories captured at pick time.
    /// Split parents are released only after a structural exact-cover proof has
    /// validated and every unresolved child has been enqueued.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn resolve_grouped_batch(
        &mut self,
        picked: PickedGroupedDomains,
        resolutions: Vec<GroupedParentResolution>,
    ) -> Result<GroupedBatchCompletion> {
        self.validate_grouped_alignment()?;
        let authority = picked.authority.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "resolve_grouped_batch called for an empty/unleased batch".to_string(),
            )
        })?;
        let lease_id = authority.lease_id;
        let sealed = {
            let grouped = self.grouped.as_ref().ok_or_else(|| {
                NyError::InvalidSpec(
                    "resolve_grouped_batch called on a scalar DomainList".to_string(),
                )
            })?;
            if !Arc::ptr_eq(&authority.queue_token, &grouped.queue_token) {
                return Err(NyError::InvalidSpec(
                    "grouped DomainList resolution belongs to a different queue".to_string(),
                ));
            }
            if picked.layout != grouped.layout {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList resolution layout mismatch for lease {lease_id}"
                )));
            }
            let lease = grouped.active_leases.get(&lease_id).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "grouped DomainList batch lease {lease_id} is absent or already consumed"
                ))
            })?;
            lease.validate_picked(&picked)?;
            if resolutions.len() != lease.domains.len() {
                return Err(NyError::InvalidSpec(format!(
                    "grouped DomainList resolution count mismatch: lease={}, resolutions={}",
                    lease.domains.len(),
                    resolutions.len()
                )));
            }

            let mut completion = GroupedBatchCompletion::default();
            for (domain, (leased, resolution)) in
                lease.domains.iter().zip(resolutions.iter()).enumerate()
            {
                if resolution.domain_id != leased.id {
                    return Err(NyError::InvalidSpec(format!(
                        "grouped DomainList resolution domain ID mismatch at position {domain}"
                    )));
                }
                match &resolution.outcome {
                    GroupedParentOutcome::Verified => {
                        if !leased.summary.is_verified() {
                            return Err(NyError::InvalidSpec(format!(
                                "grouped DomainList leased domain {domain} cannot be marked \
                                 Verified: at least one clause remains unresolved"
                            )));
                        }
                        completion.verified += 1;
                    }
                    GroupedParentOutcome::InputBisection { children } => {
                        validate_input_bisection(
                            leased,
                            children,
                            &grouped.layout,
                            lease_id,
                            &grouped.queue_token,
                        )?;
                        completion.replaced_by_children += 1;
                    }
                    GroupedParentOutcome::PhaseSplit { children } => {
                        validate_phase_split(
                            leased,
                            children,
                            &grouped.layout,
                            lease_id,
                            &grouped.queue_token,
                        )?;
                        completion.replaced_by_children += 1;
                    }
                    GroupedParentOutcome::UnresolvedDropped => {
                        completion.unresolved_dropped += 1;
                    }
                }
            }
            let next_unresolved_dropped = grouped
                .unresolved_dropped
                .checked_add(completion.unresolved_dropped)
                .ok_or_else(|| {
                    NyError::InternalError(
                        "grouped DomainList unresolved-drop counter overflow".to_string(),
                    )
                })?;
            let mut queued_children = Vec::new();
            for resolution in resolutions {
                collect_queued_children(resolution.outcome, &mut queued_children)?;
            }
            SealedGroupedResolution {
                completion,
                next_unresolved_dropped,
                queued_children,
            }
        };

        // Structural validation above is all-or-nothing. Enqueue children while
        // the parent lease remains live; any storage error therefore leaves a
        // pending obligation and cannot become a false Verified result.
        self.append_grouped_queue_entries(sealed.queued_children)?;

        let grouped = self.grouped.as_mut().ok_or_else(|| {
            NyError::InternalError(
                "resolve_grouped_batch lost grouped DomainList state".to_string(),
            )
        })?;
        if grouped.active_leases.remove(&lease_id).is_none() {
            return Err(NyError::InternalError(format!(
                "resolve_grouped_batch lost active lease {lease_id}"
            )));
        }
        grouped.unresolved_dropped = sealed.next_unresolved_dropped;
        Ok(sealed.completion)
    }

    /// Classify queue exhaustion without allowing in-flight, evicted, or
    /// silently dropped unresolved domains to produce Verified.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn grouped_queue_status(&self) -> Result<GroupedQueueStatus> {
        self.validate_grouped_alignment()?;
        let grouped = self.grouped.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("grouped_queue_status called on a scalar DomainList".to_string())
        })?;
        if !grouped.search_started {
            return Ok(GroupedQueueStatus::ExhaustedUnknown);
        }
        if !self.metadata.is_empty() || !grouped.active_leases.is_empty() {
            return Ok(GroupedQueueStatus::Pending);
        }
        if self.evicted > 0 || grouped.unresolved_dropped > 0 {
            Ok(GroupedQueueStatus::ExhaustedUnknown)
        } else {
            Ok(GroupedQueueStatus::ExhaustedVerified)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_two_clause_truth_table_never_verifies_an_unresolved_clause() {
        // Two clauses with two rows each:
        // (r0 OR r1) AND (r2 OR r3).
        let layout =
            GroupedDisjunctiveLayout::new(vec![0.0; 4], vec![2, 2], b"truth-table objective")
                .expect("valid layout");

        for mask in 0u8..16 {
            let lowers: Vec<f32> = (0..4)
                .map(|row| if mask & (1 << row) != 0 { 1.0 } else { -1.0 })
                .collect();
            let uppers: Vec<f32> = lowers.iter().map(|lower| lower + 0.25).collect();
            let summary = layout.summarize(&lowers, &uppers).unwrap();
            let expected = (mask & 0b0011 != 0) && (mask & 0b1100 != 0);
            assert_eq!(
                summary.is_verified(),
                expected,
                "wrong grouped decision for row truth mask {mask:04b}"
            );
        }
    }

    #[test]
    fn threshold_equality_is_unresolved() {
        let layout =
            GroupedDisjunctiveLayout::new(vec![0.0, 0.0], vec![1, 1], b"equality objective")
                .unwrap();
        let summary = layout.summarize(&[0.0, 1.0], &[0.5, 1.5]).unwrap();
        assert!(!summary.is_verified());
        assert_eq!(summary.lower_margin(), 0.0);
    }

    #[test]
    fn malformed_layout_and_bounds_fail_closed() {
        assert!(GroupedDisjunctiveLayout::new(Vec::new(), vec![1], b"malformed").is_err());
        assert!(GroupedDisjunctiveLayout::new(vec![0.0], Vec::new(), b"malformed").is_err());
        assert!(GroupedDisjunctiveLayout::new(vec![0.0], vec![0, 1], b"malformed").is_err());
        assert!(GroupedDisjunctiveLayout::new(vec![0.0], vec![2], b"malformed").is_err());
        assert!(GroupedDisjunctiveLayout::new(vec![f32::NAN], vec![1], b"malformed").is_err());
        assert!(GroupedDisjunctiveLayout::new(vec![0.0], vec![1], b"").is_err());

        let layout =
            GroupedDisjunctiveLayout::new(vec![0.0, 0.0], vec![1, 1], b"malformed rows").unwrap();
        assert!(layout.summarize(&[1.0], &[2.0]).is_err());
        assert!(layout.summarize(&[f32::NAN, 1.0], &[2.0, 2.0]).is_err());
        assert!(layout.summarize(&[2.0, 1.0], &[1.0, 2.0]).is_err());
    }
}
