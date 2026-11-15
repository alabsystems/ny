// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-owner flat grouped support and independent complete-cover replay.
//!
//! This private adapter has separate evidence from the qualified v1 row wire.
//! Every terminal group rebuilds checked support from its original rows: the
//! old anchor/radius bound, and a member-paired rescue only when needed.
//! No serialized/cache geometry has authority. All phases spend the supplied
//! original owner's ledger; the full direct fallback remains prepaid. There is
//! no automatic route, resident bridge, bounded group wire or runtime claim.

use super::*;
use crate::certify::work::WorkRefusal;
use crate::layers::normalization::radial::f64_group_support::{
    group_support_request, group_upper, member_group_support_request, member_group_upper,
    GroupSupportError, GroupSupportPhase, GroupSupportRequest,
};

const GROUP_SCHEMA: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NormalizedGroupPolicy {
    pub(crate) group_width: usize,
}

/// Distinct untrusted terminal forms. Deriving serde does not admit a decoded
/// allocation, export a bounded wire or convey arithmetic/coverage authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub(crate) enum NormalizedTerminalEvidence {
    OriginalGroup {
        start: usize,
        end: usize,
        upper: f64,
    },
    OriginalRow {
        competitor: usize,
        upper: f64,
        method: NormalizedSupportMethod,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NormalizedGroupedHeadEvidence {
    pub(crate) schema_version: u32,
    pub(crate) scope_digest: [u8; 32],
    pub(crate) target: usize,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) normalization: FinalNormalizationKind,
    /// Authenticates the deterministic partition, not an original assumption.
    pub(crate) group_width: usize,
    pub(crate) entries: Vec<NormalizedTerminalEvidence>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct NormalizedGroupedHeadStats {
    pub(crate) original: NormalizedHeadStats,
    pub(crate) group_work_prediction: u64,
    pub(crate) member_rescue_work_prediction: u64,
    pub(crate) member_rescue_profile_enabled: bool,
    pub(crate) member_rescues_attempted: usize,
    pub(crate) member_rescues_admitted: usize,
    pub(crate) member_rescues_completed: usize,
    pub(crate) member_rescues_admission_refused: usize,
    pub(crate) member_rescues_arithmetic_refused: usize,
    pub(crate) member_rescues_improved: usize,
    pub(crate) member_rescues_closed_groups: usize,
    pub(crate) member_rescue_coordinate_visits: u64,
    pub(crate) member_rescue_original_visits: u64,
    pub(crate) member_rescue_consumed_work: u64,
    /// Outer fallback plus each admitted old/member group phase exactly once.
    pub(crate) reserved_work: u64,
    pub(crate) group_profile_enabled: bool,
    pub(crate) groups_attempted: usize,
    pub(crate) groups_admitted: usize,
    pub(crate) groups_completed: usize,
    pub(crate) groups_admission_refused: usize,
    pub(crate) groups_arithmetic_refused: usize,
    pub(crate) terminal_groups: usize,
    pub(crate) grouped_competitors: usize,
    pub(crate) direct_rows: usize,
    pub(crate) completed_group_coordinate_visits: u64,
    pub(crate) completed_group_original_visits: u64,
    pub(crate) completed_group_consumed_work: u64,
    pub(crate) proposal_coverage_visits: u64,
    /// The kernel returns no partial failure receipt. Such visits are omitted
    /// from completed counters; their admitted work remains fully charged.
    pub(crate) unobserved_failed_group_attempts: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CertifiedGroupedNormalizedNextToken {
    inner: Arc<CertifiedGroupedNormalizedNextTokenInner>,
}
#[derive(Debug)]
struct CertifiedGroupedNormalizedNextTokenInner {
    target: usize,
    competitors: usize,
    maximum_upper: Option<f64>,
    scope: CertificateScope,
    provenance: NextTokenProvenance,
    checked_stats: NormalizedGroupedHeadStats,
    reservation: PersistentReservation,
    owner: SharedAffineHead,
}
impl CertifiedGroupedNormalizedNextToken {
    pub(crate) fn target(&self) -> usize {
        self.inner.target
    }
    pub(crate) fn covered_competitors(&self) -> usize {
        self.inner.competitors
    }
    pub(crate) fn maximum_competitor_upper(&self) -> Option<f64> {
        self.inner.maximum_upper
    }
    pub(crate) fn scope(&self) -> &CertificateScope {
        &self.inner.scope
    }
    pub(crate) fn provenance(&self) -> NextTokenProvenance {
        self.inner.provenance
    }
    pub(crate) fn checked_stats(&self) -> &NormalizedGroupedHeadStats {
        &self.inner.checked_stats
    }
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.inner.reservation.bytes()
    }
    pub(crate) fn shared_head_bytes(&self) -> usize {
        self.inner.owner.reserved_bytes()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedGroupedNextTokenOutcome {
    Certified(CertifiedGroupedNormalizedNextToken),
    Unknown,
}

#[derive(Debug)]
pub(crate) struct NormalizedGroupedHeadReport {
    pub(crate) outcome: NormalizedGroupedNextTokenOutcome,
    pub(crate) evidence: NormalizedGroupedHeadEvidence,
    pub(crate) unresolved: Vec<usize>,
    pub(crate) stats: NormalizedGroupedHeadStats,
    provenance: NextTokenProvenance,
    reservation: PersistentReservation,
    owner: SharedAffineHead,
    budget: WorkBudget,
}
impl NormalizedGroupedHeadReport {
    pub(crate) fn provenance(&self) -> NextTokenProvenance {
        self.provenance
    }
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub(crate) fn shared_head_bytes(&self) -> usize {
        self.owner.reserved_bytes()
    }
    pub(crate) fn replay(
        &self,
        norm: FinalNormalization<'_>,
        pre: HiddenBox<'_>,
        target: usize,
        limits: &VocabOptions,
    ) -> Result<Self> {
        replay_shared_grouped_normalized_next_token(
            &self.owner,
            norm,
            pre,
            target,
            &self.evidence,
            NormalizedHeadOptions {
                budget: &self.budget,
                limits,
            },
        )
    }
}

fn evidence_error(why: &'static str) -> NormalizedHeadRefusal {
    DecoderRefusal::Evidence(why).into()
}
fn group_error(error: GroupSupportError) -> NormalizedHeadRefusal {
    match error {
        GroupSupportError::Input(why) => evidence_error(why),
        GroupSupportError::Arithmetic(why) => DecoderRefusal::NonFinite(why).into(),
        GroupSupportError::Work(error) => DecoderRefusal::Budget(error).into(),
        GroupSupportError::Deadline => DecoderRefusal::Deadline.into(),
    }
}
fn environment() -> Result<()> {
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| evidence_error("grouped normalized binary64 proof environment"))
}
fn checked_work_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or(resource("grouped normalized work overflow"))
}

fn maximum_groups(vocab: usize, width: usize) -> Result<usize> {
    if vocab == 0 || width == 0 {
        return Err(DecoderRefusal::Shape("grouped normalized partition").into());
    }
    if vocab == 1 {
        return Ok(0);
    }
    Ok(vocab / width + usize::from(!vocab.is_multiple_of(width)))
}
fn extra_group_work(vocab: usize, hidden: usize, width: usize) -> Result<u64> {
    let groups = maximum_groups(vocab, width)?;
    if groups == 0 {
        return Ok(0);
    }
    let work = add(
        add(
            mul(mul(vocab, hidden)?, 32)?,
            mul(mul(groups, hidden)?, 256)?,
        )?,
        add(mul(vocab, 128)?, mul(groups, 4096)?)?,
    )?;
    u64::try_from(work).map_err(|_| resource("grouped normalized work overflow"))
}

fn extra_member_work(vocab: usize, hidden: usize, width: usize) -> Result<u64> {
    if maximum_groups(vocab, width)? == 0 {
        return Ok(0);
    }
    checked_work_add(
        extra_group_work(vocab, hidden, width)?,
        u64::try_from(mul(mul(vocab, hidden)?, 64)?)
            .map_err(|_| resource("member group work overflow"))?,
    )
}

/// Pre-fund this worst case and independent replay on the SAME caller ledger.
/// This is a bound, not a constructor, refund, admission or measured work.
pub(crate) fn normalized_group_work_bound(
    vocab: usize,
    hidden: usize,
    width: usize,
) -> Result<u64> {
    checked_work_add(
        normalized_work_bound(vocab, hidden)?,
        checked_work_add(
            extra_group_work(vocab, hidden, width)?,
            extra_member_work(vocab, hidden, width)?,
        )?,
    )
}

#[derive(Clone)]
struct Ranges {
    next: usize,
    vocab: usize,
    target: usize,
    width: usize,
}
impl Ranges {
    fn new(vocab: usize, target: usize, width: usize) -> Self {
        Self {
            next: usize::from(target == 0),
            vocab,
            target,
            width,
        }
    }
}
impl Iterator for Ranges {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.vocab {
            return None;
        }
        let start = self.next;
        let end = start + self.width.min(self.vocab - start);
        self.next = end + usize::from(end == self.target);
        Some((start, end))
    }
}

fn kernel_request<'a>(
    head: AffineHead<'a>,
    norm: FinalNormalization<'a>,
    target: usize,
    start: usize,
    end: usize,
) -> GroupSupportRequest<'a> {
    GroupSupportRequest {
        weights: head.weights,
        bias: Some(head.bias),
        hidden: head.hidden_size,
        target,
        start,
        end,
        gamma: norm.gamma,
        beta: norm.beta,
    }
}

fn shape_grouped(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    width: usize,
    options: NormalizedHeadOptions<'_>,
) -> Result<(WorkRequest, u64, bool, u64, bool)> {
    let head = owner.as_head();
    let groups = maximum_groups(head.vocab_size, width)?;
    let mut request = shape_with_owner(head, norm, pre, target, options, Some(owner))?;
    let old_evidence = add(
        mul(head.vocab_size - 1, size_of::<NormalizedRowEvidence>())?,
        size_of::<NormalizedHeadEvidence>(),
    )?;
    let new_evidence = add(
        mul(head.vocab_size - 1, size_of::<NormalizedTerminalEvidence>())?,
        size_of::<NormalizedGroupedHeadEvidence>(),
    )?;
    if new_evidence > options.limits.max_evidence_bytes {
        return Err(resource("grouped normalized evidence cap"));
    }
    request.persistent_bytes = add(
        request
            .persistent_bytes
            .checked_sub(old_evidence)
            .ok_or(resource("grouped normalized evidence pricing"))?,
        add(new_evidence, 4096)?,
    )?;
    let live = options.budget.receipt().map_err(DecoderRefusal::Budget)?;
    let resident = add(
        add(live.live_persistent_bytes, request.persistent_bytes)?,
        owner.resident_bytes_outside_ledger(options.budget),
    )?;
    let scratch = add(live.live_scratch_bytes, request.scratch_bytes)?;
    if resident as u64 > options.limits.max_persistent_bytes
        || scratch as u64 > options.limits.max_scratch_bytes
    {
        return Err(resource("grouped normalized baseline capacity"));
    }
    let extra = extra_group_work(head.vocab_size, head.hidden_size, width)?;
    let member_extra = extra_member_work(head.vocab_size, head.hidden_size, width)?;
    let enabled = if let Some((start, end)) = Ranges::new(head.vocab_size, target, width).next() {
        let group = group_support_request(&kernel_request(head, norm, target, start, end))
            .map_err(group_error)?;
        checked_work_add(request.work, extra)? <= options.limits.max_terms
            && checked_work_add(request.proposals, groups as u64)? <= options.limits.max_proposals
            && add(resident, group.persistent_bytes)? as u64 <= options.limits.max_persistent_bytes
            && add(scratch, group.scratch_bytes)? as u64 <= options.limits.max_scratch_bytes
    } else {
        false
    };
    // This separate worst-case check cannot disable an old-closing group.
    // The old scalar result and the new rescue coexist, but old phase scratch
    // has already been released. No optional allowance is refunded/refilled.
    let member_enabled = if enabled {
        let (start, end) = Ranges::new(head.vocab_size, target, width)
            .next()
            .ok_or(resource("member partition"))?;
        let old = group_support_request(&kernel_request(head, norm, target, start, end))
            .map_err(group_error)?;
        let new = member_group_support_request(&kernel_request(head, norm, target, start, end))
            .map_err(group_error)?;
        checked_work_add(checked_work_add(request.work, extra)?, member_extra)?
            <= options.limits.max_terms
            && checked_work_add(
                request.proposals,
                (groups as u64)
                    .checked_mul(2)
                    .ok_or(resource("member proposals"))?,
            )? <= options.limits.max_proposals
            && add(add(resident, old.persistent_bytes)?, new.persistent_bytes)? as u64
                <= options.limits.max_persistent_bytes
            && add(scratch, new.scratch_bytes)? as u64 <= options.limits.max_scratch_bytes
    } else {
        false
    };
    Ok((request, extra, enabled, member_extra, member_enabled))
}

fn header(
    head: AffineHead<'_>,
    norm: FinalNormalization<'_>,
    target: usize,
    proposal: &NormalizedGroupedHeadEvidence,
    limits: &VocabOptions,
) -> Result<()> {
    if proposal.schema_version != GROUP_SCHEMA
        || proposal.target != target
        || proposal.vocab_size != head.vocab_size
        || proposal.hidden_size != head.hidden_size
        || proposal.normalization != norm.kind
        || proposal.group_width == 0
        || proposal.entries.len() > head.vocab_size - 1
    {
        return Err(evidence_error("grouped normalized replay header"));
    }
    if add(
        mul(
            proposal.entries.capacity(),
            size_of::<NormalizedTerminalEvidence>(),
        )?,
        size_of::<NormalizedGroupedHeadEvidence>(),
    )? > limits.max_evidence_bytes
    {
        return Err(resource("grouped normalized proposal capacity"));
    }
    Ok(())
}

/// Reconstruct the exact deterministic partition without trusting entry counts.
/// Every original competitor is visited even for one complete negative group.
fn coverage(
    evidence: &NormalizedGroupedHeadEvidence,
    lease: &WorkLease,
    limits: &VocabOptions,
    mut visit: impl FnMut(usize, f64) -> Result<()>,
) -> Result<()> {
    let mut entry = 0usize;
    let mut covered = 0usize;
    for (start, end) in Ranges::new(evidence.vocab_size, evidence.target, evidence.group_width) {
        checkpoint(lease, limits)?;
        if let Some(NormalizedTerminalEvidence::OriginalGroup {
            start: s,
            end: e,
            upper,
        }) = evidence.entries.get(entry)
        {
            if *s != start || *e != end || !upper.is_finite() {
                return Err(evidence_error("grouped normalized terminal range"));
            }
            for competitor in start..end {
                if competitor == evidence.target {
                    continue;
                }
                if covered.is_multiple_of(CHUNK) {
                    checkpoint(lease, limits)?;
                }
                visit(competitor, *upper)?;
                covered += 1;
            }
            entry += 1;
        } else {
            for competitor in start..end {
                if competitor == evidence.target {
                    continue;
                }
                if covered.is_multiple_of(CHUNK) {
                    checkpoint(lease, limits)?;
                }
                let Some(NormalizedTerminalEvidence::OriginalRow {
                    competitor: id,
                    upper,
                    ..
                }) = evidence.entries.get(entry)
                else {
                    return Err(evidence_error("grouped normalized direct cover"));
                };
                if *id != competitor || !upper.is_finite() {
                    return Err(evidence_error("grouped normalized direct cover"));
                }
                visit(competitor, *upper)?;
                covered += 1;
                entry += 1;
            }
        }
    }
    if covered != evidence.vocab_size - 1 || entry != evidence.entries.len() {
        return Err(evidence_error("grouped normalized complete cover"));
    }
    checkpoint(lease, limits)?;
    Ok(())
}

pub(crate) fn verify_shared_grouped_normalized_next_token(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    policy: NormalizedGroupPolicy,
    options: NormalizedHeadOptions<'_>,
) -> Result<NormalizedGroupedHeadReport> {
    run_grouped(
        owner,
        norm,
        pre,
        target,
        policy.group_width,
        None,
        options,
        LeaseClass::Optional,
        NextTokenProvenance::CheckedQuery,
    )
}
pub(crate) fn replay_shared_grouped_normalized_next_token(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    evidence: &NormalizedGroupedHeadEvidence,
    options: NormalizedHeadOptions<'_>,
) -> Result<NormalizedGroupedHeadReport> {
    run_grouped(
        owner,
        norm,
        pre,
        target,
        evidence.group_width,
        Some(evidence),
        options,
        LeaseClass::Replay,
        NextTokenProvenance::OriginalReplay,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_grouped(
    owner: &SharedAffineHead,
    norm: FinalNormalization<'_>,
    pre: HiddenBox<'_>,
    target: usize,
    width: usize,
    proposal: Option<&NormalizedGroupedHeadEvidence>,
    options: NormalizedHeadOptions<'_>,
    class: LeaseClass,
    provenance: NextTokenProvenance,
) -> Result<NormalizedGroupedHeadReport> {
    environment()?;
    let head = owner.as_head();
    let (request, extra, enabled, member_extra, member_enabled) =
        shape_grouped(owner, norm, pre, target, width, options)?;
    if let Some(evidence) = proposal {
        header(head, norm, target, evidence, options.limits)?;
    }
    let mut lease = options
        .budget
        .lease_for(class, request)
        .map_err(DecoderRefusal::Budget)?;
    lease.mark_materialized().map_err(DecoderRefusal::Budget)?;
    let mut stats = NormalizedGroupedHeadStats {
        original: NormalizedHeadStats {
            reserved_work: request.work,
            persistent_bytes: request.persistent_bytes,
            scratch_bytes: request.scratch_bytes,
            ..NormalizedHeadStats::default()
        },
        group_work_prediction: extra,
        member_rescue_work_prediction: member_extra,
        member_rescue_profile_enabled: member_enabled,
        reserved_work: request.work,
        group_profile_enabled: enabled,
        ..NormalizedGroupedHeadStats::default()
    };
    let (scope, digest) = scope(
        head,
        norm,
        pre,
        target,
        &lease,
        options.limits,
        &mut stats.original,
        Some(owner),
    )?;
    if let Some(evidence) = proposal {
        if evidence.scope_digest != digest {
            return Err(evidence_error("grouped normalized original scope"));
        }
        coverage(evidence, &lease, options.limits, |_, _| {
            visits(&mut stats.proposal_coverage_visits, 1)?;
            Ok(())
        })?;
        if !enabled {
            for block in evidence.entries.chunks(CHUNK) {
                checkpoint(&lease, options.limits)?;
                if block
                    .iter()
                    .any(|entry| matches!(entry, NormalizedTerminalEvidence::OriginalGroup { .. }))
                {
                    return Err(resource("grouped normalized replay group prediction"));
                }
            }
        }
    }
    let geometry = GeometryF64::new(
        pre.lo,
        pre.hi,
        norm.epsilon,
        norm.kind == FinalNormalizationKind::LayerNorm,
        &mut |count| {
            checkpoint(&lease, options.limits)?;
            visits(&mut stats.original.kernel_coordinate_visits, count)
        },
    )
    .map_err(kernel_error)?;
    let mut entries = reserve(head.vocab_size - 1)?;
    let mut proposal_index = 0usize;
    for (start, end) in Ranges::new(head.vocab_size, target, width) {
        checkpoint(&lease, options.limits)?;
        let group_claim =
            proposal.and_then(|evidence| match evidence.entries.get(proposal_index) {
                Some(NormalizedTerminalEvidence::OriginalGroup { upper, .. }) => Some(*upper),
                _ => None,
            });
        let try_group = group_claim.is_some() || (proposal.is_none() && enabled);
        if try_group {
            let original = kernel_request(head, norm, target, start, end);
            let group_request = group_support_request(&original).map_err(group_error)?;
            stats.groups_attempted += 1;
            let phase = match GroupSupportPhase::admit(
                options.budget,
                class,
                group_request,
                options.limits.deadline,
            ) {
                Ok(phase) => {
                    stats.groups_admitted += 1;
                    stats.reserved_work =
                        checked_work_add(stats.reserved_work, group_request.work)?;
                    Some(phase)
                }
                Err(GroupSupportError::Work(
                    error @ (WorkRefusal::ScratchLimit
                    | WorkRefusal::PersistentLimit
                    | WorkRefusal::WorkLimit
                    | WorkRefusal::ProposalLimit
                    | WorkRefusal::MemoryLimit),
                )) if group_claim.is_none() => {
                    let _ = error;
                    stats.groups_admission_refused += 1;
                    None
                }
                Err(error) => return Err(group_error(error)),
            };
            if let Some(phase) = phase {
                let bound = match group_upper(&geometry, &original, phase) {
                    Ok(bound) => Some(bound),
                    Err(GroupSupportError::Arithmetic(_)) if group_claim.is_none() => {
                        environment()?;
                        checkpoint(&lease, options.limits)?;
                        stats.groups_arithmetic_refused += 1;
                        stats.unobserved_failed_group_attempts += 1;
                        None
                    }
                    Err(error) => return Err(group_error(error)),
                };
                if let Some(bound) = bound {
                    let observed = bound.stats();
                    stats.groups_completed += 1;
                    stats.completed_group_coordinate_visits = checked_work_add(
                        stats.completed_group_coordinate_visits,
                        observed.coordinate_visits,
                    )?;
                    stats.completed_group_original_visits = checked_work_add(
                        stats.completed_group_original_visits,
                        observed.original_member_visits,
                    )?;
                    stats.completed_group_consumed_work = checked_work_add(
                        stats.completed_group_consumed_work,
                        observed.consumed_work,
                    )?;
                    if bound.range() != (start, end) || bound.target() != target {
                        return Err(evidence_error("grouped normalized kernel binding"));
                    }
                    let old_upper = bound.upper();
                    let mut checked_upper = old_upper;
                    let needs_rescue =
                        group_claim.map_or(old_upper >= 0.0, |claimed| old_upper > claimed);
                    if group_claim.is_some() && needs_rescue && !member_enabled {
                        return Err(resource("grouped normalized replay member prediction"));
                    }
                    if needs_rescue && member_enabled {
                        stats.member_rescues_attempted += 1;
                        let rescue_request =
                            member_group_support_request(&original).map_err(group_error)?;
                        let phase = match GroupSupportPhase::admit(
                            options.budget,
                            class,
                            rescue_request,
                            options.limits.deadline,
                        ) {
                            Ok(phase) => {
                                stats.member_rescues_admitted += 1;
                                stats.reserved_work =
                                    checked_work_add(stats.reserved_work, rescue_request.work)?;
                                Some(phase)
                            }
                            Err(GroupSupportError::Work(
                                error @ (WorkRefusal::ScratchLimit
                                | WorkRefusal::PersistentLimit
                                | WorkRefusal::WorkLimit
                                | WorkRefusal::ProposalLimit
                                | WorkRefusal::MemoryLimit),
                            )) if group_claim.is_none() => {
                                let _ = error;
                                stats.member_rescues_admission_refused += 1;
                                None
                            }
                            Err(error) => return Err(group_error(error)),
                        };
                        if let Some(phase) = phase {
                            match member_group_upper(&geometry, &original, phase) {
                                Ok(rescue) => {
                                    if rescue.range() != bound.range()
                                        || rescue.target() != bound.target()
                                        || rescue.competitors() != bound.competitors()
                                    {
                                        return Err(evidence_error(
                                            "grouped normalized member binding",
                                        ));
                                    }
                                    let observed = rescue.stats();
                                    stats.member_rescues_completed += 1;
                                    stats.member_rescue_coordinate_visits = checked_work_add(
                                        stats.member_rescue_coordinate_visits,
                                        observed.coordinate_visits,
                                    )?;
                                    stats.member_rescue_original_visits = checked_work_add(
                                        stats.member_rescue_original_visits,
                                        observed.original_member_visits,
                                    )?;
                                    stats.member_rescue_consumed_work = checked_work_add(
                                        stats.member_rescue_consumed_work,
                                        observed.consumed_work,
                                    )?;
                                    checked_upper = checked_upper.min(rescue.upper());
                                    stats.member_rescues_improved +=
                                        usize::from(checked_upper < old_upper);
                                    stats.member_rescues_closed_groups +=
                                        usize::from(old_upper >= 0.0 && checked_upper < 0.0);
                                    drop(rescue); // Only the checked scalar remains under outer evidence.
                                }
                                Err(GroupSupportError::Arithmetic(_)) if group_claim.is_none() => {
                                    environment()?;
                                    checkpoint(&lease, options.limits)?;
                                    stats.member_rescues_arithmetic_refused += 1;
                                    stats.unobserved_failed_group_attempts += 1;
                                }
                                Err(error) => return Err(group_error(error)),
                            }
                        }
                    }
                    let upper = if let Some(claimed) = group_claim {
                        if checked_upper > claimed {
                            return Err(evidence_error("grouped normalized fresh group replay"));
                        }
                        claimed
                    } else {
                        checked_upper
                    };
                    if group_claim.is_some() || upper < 0.0 {
                        stats.terminal_groups += 1;
                        stats.grouped_competitors += bound.competitors();
                        entries.push(NormalizedTerminalEvidence::OriginalGroup {
                            start,
                            end,
                            upper,
                        });
                        proposal_index += usize::from(proposal.is_some());
                        // Its separate temporary result reservation is released;
                        // outer-owned evidence retains only the checked scalar.
                        drop(bound);
                        continue;
                    }
                }
            }
        }
        for competitor in start..end {
            if competitor == target {
                continue;
            }
            checkpoint(&lease, options.limits)?;
            let claimed = proposal.map(|evidence| &evidence.entries[proposal_index]);
            let (demand, hint) = if let Some(NormalizedTerminalEvidence::OriginalRow {
                upper,
                method,
                ..
            }) = claimed
            {
                (SupportDemand::AtMost(*upper), Some((*upper, *method)))
            } else if claimed.is_some() {
                return Err(evidence_error("grouped normalized expected direct row"));
            } else {
                (SupportDemand::StrictNegative, None)
            };
            let bound = geometry
                .upper_for(
                    head.row(competitor),
                    head.row(target),
                    norm.gamma,
                    norm.beta,
                    head.bias[competitor],
                    head.bias[target],
                    demand,
                    &mut |count| {
                        checkpoint(&lease, options.limits)?;
                        visits(&mut stats.original.kernel_coordinate_visits, count)
                    },
                )
                .map_err(kernel_error)?;
            let method = if bound
                .taylor_upper
                .is_some_and(|value| value < bound.dense_upper)
            {
                stats.original.radial_taylor_rows += 1;
                NormalizedSupportMethod::RadialTaylor
            } else {
                stats.original.projected_dense_rows += 1;
                NormalizedSupportMethod::ProjectedDense
            };
            stats.original.dense_screened_rows += usize::from(bound.dense_screened);
            stats.original.cauchy_screened_rows += usize::from(bound.cauchy_screened);
            stats.original.radial_taylor_attempted_rows += usize::from(bound.taylor_attempted);
            stats.original.projected_zero_rows += usize::from(bound.projection_zero);
            let (upper, method) = if let Some((claimed, hint)) = hint {
                if bound.upper > claimed {
                    return Err(evidence_error("grouped normalized fresh row replay"));
                }
                (claimed, hint)
            } else {
                (bound.upper, method)
            };
            stats.direct_rows += 1;
            entries.push(NormalizedTerminalEvidence::OriginalRow {
                competitor,
                upper,
                method,
            });
            proposal_index += usize::from(proposal.is_some());
        }
    }
    drop(geometry);
    let evidence = NormalizedGroupedHeadEvidence {
        schema_version: GROUP_SCHEMA,
        scope_digest: digest,
        target,
        vocab_size: head.vocab_size,
        hidden_size: head.hidden_size,
        normalization: norm.kind,
        group_width: width,
        entries,
    };
    finish_grouped(
        owner, scope, evidence, stats, lease, options, class, provenance,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_grouped(
    owner: &SharedAffineHead,
    scope: CertificateScope,
    evidence: NormalizedGroupedHeadEvidence,
    mut stats: NormalizedGroupedHeadStats,
    mut lease: WorkLease,
    options: NormalizedHeadOptions<'_>,
    class: LeaseClass,
    provenance: NextTokenProvenance,
) -> Result<NormalizedGroupedHeadReport> {
    let mut maximum: Option<f64> = None;
    let mut unresolved = reserve(evidence.vocab_size - 1)?;
    coverage(&evidence, &lease, options.limits, |competitor, upper| {
        maximum = Some(maximum.map_or(upper, |value| value.max(upper)));
        if upper >= 0.0 {
            unresolved.push(competitor);
        }
        visits(&mut stats.original.coverage_visits, 1)?;
        Ok(())
    })?;
    let actual_evidence = add(
        mul(
            evidence.entries.capacity(),
            size_of::<NormalizedTerminalEvidence>(),
        )?,
        size_of::<NormalizedGroupedHeadEvidence>(),
    )?;
    if actual_evidence > options.limits.max_evidence_bytes
        || unresolved.capacity() > evidence.vocab_size - 1
    {
        return Err(resource("grouped normalized actual result capacity"));
    }
    // Verify actual retained output/scope/metadata capacities fit the original
    // component reservation; original normalization slices remain caller-owned.
    let retained = add(
        add(actual_evidence, mul(unresolved.capacity(), 8)?)?,
        add(mul(evidence.hidden_size, 16)?, 4096)?,
    )?;
    if retained > stats.original.persistent_bytes {
        return Err(resource("grouped normalized retained capacity"));
    }
    environment()?;
    checkpoint(&lease, options.limits)?;
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let outcome = if maximum.is_none_or(|upper| upper < 0.0) {
        NormalizedGroupedNextTokenOutcome::Certified(CertifiedGroupedNormalizedNextToken {
            inner: Arc::new(CertifiedGroupedNormalizedNextTokenInner {
                target: evidence.target,
                competitors: evidence.vocab_size - 1,
                maximum_upper: maximum,
                scope,
                provenance,
                checked_stats: stats.clone(),
                reservation: reservation.clone(),
                owner: owner.clone(),
            }),
        })
    } else {
        NormalizedGroupedNextTokenOutcome::Unknown
    };
    environment()?;
    options
        .budget
        .checkpoint_for(class)
        .map_err(DecoderRefusal::Budget)?;
    if options
        .limits
        .deadline
        .is_some_and(|deadline| Instant::now() >= deadline)
        || ny_core::instance_budget::deadline().is_some_and(|deadline| Instant::now() >= deadline)
    {
        return Err(DecoderRefusal::Deadline.into());
    }
    Ok(NormalizedGroupedHeadReport {
        outcome,
        evidence,
        unresolved,
        stats,
        provenance,
        reservation,
        owner: owner.clone(),
        budget: options.budget.clone(),
    })
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "external-models"))]
mod original_probe;

// Distinct private schema2 ingress; qualified schema1/public routes are intact.
pub(crate) mod wire;

// One existing scope-owner allocation now retains the whole sealed header.
const _: () = assert!(
    size_of::<NormalizedGroupedHeadReport>()
        + size_of::<CertifiedGroupedNormalizedNextTokenInner>()
        + 2 * size_of::<usize>()
        <= 4096
);
