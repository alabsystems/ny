// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-head replay and the only private next-token certificate constructor.
use super::next_token::NextTokenReport;
use super::numeric::{anchor_margin, dense_upper, radius};
use super::shared_head::{incremental_head_bytes, resident_with_owner};
use super::vocab_tree::{
    checked_mul, default_budget, geometry, group_bound, identity, scope, scope_digest,
    validate_box, validate_head,
};
use super::{
    AffineHead, DecoderRefusal, GroupEvidence, HiddenBox, Meter, SharedAffineHead, VocabEvidence,
    VocabOptions, VocabStats,
};
use crate::certify::scope::CertificateScope;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};
use std::sync::Arc;

/// How the private checker established this sealed decision. Both paths have
/// certificate authority; independent replay reconstructs every terminal group
/// from original coefficients rather than reusing checked prepared geometry.
/// This says nothing about a report's subsequently mutated raw evidence bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum NextTokenProvenance {
    CheckedQuery,
    OriginalReplay,
}

/// No public/deserialization constructor. Its scope and strict complete-cover
/// bound are minted only after checked arithmetic and original vocabulary cover.
#[derive(Debug, Clone)]
pub struct CertifiedNextToken {
    inner: Arc<CertifiedNextTokenInner>,
}
#[derive(Debug)]
struct CertifiedNextTokenInner {
    target: usize,
    competitors: usize,
    maximum_upper: Option<f64>,
    scope: CertificateScope,
    reservation: PersistentReservation,
    head_owner: Option<SharedAffineHead>,
    provenance: NextTokenProvenance,
}
// The former single scope Arc is now the single complete certificate owner.
// Its payload/control block is included before the mint allocation.
pub(super) const fn certificate_owner_bytes() -> usize {
    size_of::<CertifiedNextTokenInner>() + 2 * size_of::<usize>()
}
impl CertifiedNextToken {
    pub fn target(&self) -> usize {
        self.inner.target
    }
    pub fn covered_competitors(&self) -> usize {
        self.inner.competitors
    }
    /// None means a vocabulary with only its target, a vacuous full cover.
    pub fn maximum_competitor_upper(&self) -> Option<f64> {
        self.inner.maximum_upper
    }
    pub fn scope(&self) -> &CertificateScope {
        &self.inner.scope
    }
    /// Provenance of the sealed certificate in its private scope. Raw exported
    /// evidence remains untrusted and must be replayed after any modification.
    pub fn provenance(&self) -> NextTokenProvenance {
        self.inner.provenance
    }
    pub fn reserved_bytes(&self) -> usize {
        self.inner.reservation.bytes()
    }
    /// Original head storage shared with other certificates/trees, counted
    /// once in the common ledger rather than in each incremental reservation.
    pub fn shared_head_bytes(&self) -> usize {
        self.inner
            .head_owner
            .as_ref()
            .map_or(0, SharedAffineHead::reserved_bytes)
    }
}
#[derive(Debug, Clone)]
pub enum NextTokenOutcome {
    Certified(CertifiedNextToken),
    /// A tie or positive/unfinished competitor bound supplies no argmax claim.
    Unknown,
}

fn coverage(
    evidence: &VocabEvidence,
    max_vocab: usize,
    max_nodes: usize,
) -> Result<(), DecoderRefusal> {
    if evidence.schema_version != 1
        || evidence.vocab_size == 0
        || evidence.target >= evidence.vocab_size
        || evidence.hidden_size == 0
    {
        return Err(DecoderRefusal::Evidence("schema or shape"));
    }
    if evidence.vocab_size > max_vocab || evidence.groups.len() > max_nodes {
        return Err(DecoderRefusal::Resource("evidence shape cap"));
    }
    let members = evidence
        .groups
        .iter()
        .try_fold(0usize, |n, g| n.checked_add(g.members.len()))
        .ok_or(DecoderRefusal::Resource("cover count overflow"))?;
    if members != evidence.vocab_size - 1 {
        return Err(DecoderRefusal::Evidence("incomplete vocabulary"));
    }
    let mut seen = vec![false; evidence.vocab_size];
    for group in &evidence.groups {
        if group.members.is_empty()
            || !group.members.contains(&group.anchor_token)
            || !group.upper.is_finite()
        {
            return Err(DecoderRefusal::Evidence("group shape, anchor or bound"));
        }
        let mut previous = None;
        for &id in &group.members {
            if id >= evidence.vocab_size
                || id == evidence.target
                || seen[id]
                || previous.is_some_and(|p| id <= p)
            {
                return Err(DecoderRefusal::Evidence(
                    "duplicate, target or invalid competitor",
                ));
            }
            previous = Some(id);
            seen[id] = true;
        }
    }
    if (0..evidence.vocab_size).any(|id| id != evidence.target && !seen[id]) {
        return Err(DecoderRefusal::Evidence("vocabulary hole"));
    }
    Ok(())
}

/// Trusted query/replay paths alone call this; evidence still passes full cover
/// and strictness before the private constructor can execute.
pub(super) fn finish(
    scope: CertificateScope,
    evidence: VocabEvidence,
    unresolved: Vec<usize>,
    stats: VocabStats,
    lease: WorkLease,
    budget: &WorkBudget,
    options: &VocabOptions,
    class: LeaseClass,
    provenance: NextTokenProvenance,
) -> Result<NextTokenReport, DecoderRefusal> {
    finish_with_owner(
        scope, evidence, unresolved, stats, lease, budget, options, class, None, provenance,
    )
}

pub(super) fn finish_with_owner(
    scope: CertificateScope,
    evidence: VocabEvidence,
    unresolved: Vec<usize>,
    stats: VocabStats,
    mut lease: WorkLease,
    budget: &WorkBudget,
    options: &VocabOptions,
    class: LeaseClass,
    head_owner: Option<SharedAffineHead>,
    provenance: NextTokenProvenance,
) -> Result<NextTokenReport, DecoderRefusal> {
    if let Some(owner) = &head_owner {
        owner.validate_identity(scope.identity())?;
    }
    coverage(&evidence, evidence.vocab_size, evidence.groups.len().max(1))?;
    let maximum_upper = evidence.groups.iter().map(|g| g.upper).reduce(f64::max);
    Meter::with_budget_class(options, budget.clone(), class)?.checkpoint()?;
    // Keep scratch charged through the complete-cover bitmap check. Callers
    // drop their temporary vectors before transferring the live lease here.
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let outcome = if maximum_upper.is_none_or(|v| v < 0.0) {
        if !unresolved.is_empty() {
            return Err(DecoderRefusal::Evidence("unresolved strict cover"));
        }
        NextTokenOutcome::Certified(CertifiedNextToken {
            inner: Arc::new(CertifiedNextTokenInner {
                target: evidence.target,
                competitors: evidence.vocab_size - 1,
                maximum_upper,
                scope,
                reservation: reservation.clone(),
                head_owner: head_owner.clone(),
                provenance,
            }),
        })
    } else {
        NextTokenOutcome::Unknown
    };
    let report = NextTokenReport {
        outcome,
        evidence,
        unresolved,
        stats,
        reservation,
        head_owner,
        provenance,
    };
    Meter::with_budget_class(options, budget.clone(), class)?.checkpoint()?;
    Ok(report)
}

/// Replay serialized proposals from ORIGINAL head rows. Every terminal group's
/// coefficients, anchor logits, residual radii, numeric allowance and complete
/// membership are reconstructed; no prepared geometry or claimed score is used.
pub fn replay_next_token(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    evidence: &VocabEvidence,
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    replay_with_owner(head, hidden, anchor, target, evidence, options, None)
}

pub(super) fn replay_with_owner(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    evidence: &VocabEvidence,
    options: &VocabOptions,
    head_owner: Option<SharedAffineHead>,
) -> Result<NextTokenReport, DecoderRefusal> {
    let scoped = head_owner
        .as_ref()
        .map(|owner| owner.scoped_options(options))
        .transpose()?;
    let options = scoped.as_ref().unwrap_or(options);
    let budget = default_budget(options)?;
    let mut meter = Meter::with_budget_class(options, budget.clone(), LeaseClass::Replay)?;
    validate_head(head, anchor, target, options)?;
    validate_box(hidden, head.hidden_size)?;
    super::execution::validate_input_domain(head.arithmetic, hidden)?;
    let domain_work = if matches!(head.arithmetic, super::HeadArithmetic::Binary32 { .. }) {
        head.hidden_size
    } else {
        0
    };
    meter.charge(domain_work)?;
    if evidence.vocab_size != head.vocab_size
        || evidence.hidden_size != head.hidden_size
        || evidence.target != target
    {
        return Err(DecoderRefusal::Evidence("request shape binding"));
    }
    // Bound vectors before a proportional coverage allocation or serde clone.
    if evidence.groups.len() > options.max_nodes || evidence.vocab_size > options.max_vocab {
        return Err(DecoderRefusal::Resource("replay node cap"));
    }
    let member_count = evidence
        .groups
        .iter()
        .try_fold(0usize, |n, g| n.checked_add(g.members.len()))
        .ok_or(DecoderRefusal::Resource("replay member count"))?;
    if member_count != head.vocab_size - 1 {
        return Err(DecoderRefusal::Evidence("incomplete vocabulary"));
    }
    // Input capacity can exceed initialized length (including poisoned spare
    // capacity). The output clone allocates lengths; count both explicitly.
    let mut input_bytes = size_of::<VocabEvidence>()
        .checked_add(checked_mul(
            evidence.groups.capacity(),
            size_of::<GroupEvidence>(),
        )?)
        .ok_or(DecoderRefusal::Resource("input evidence metadata"))?;
    for group in &evidence.groups {
        input_bytes = input_bytes
            .checked_add(checked_mul(group.members.capacity(), size_of::<usize>())?)
            .ok_or(DecoderRefusal::Resource("input evidence capacity"))?;
    }
    let evidence_bytes = checked_mul(member_count, 16)?
        .checked_add(checked_mul(
            evidence.groups.len(),
            size_of::<GroupEvidence>(),
        )?)
        .and_then(|n| n.checked_add(input_bytes))
        .and_then(|n| n.checked_add(head.hidden_size.checked_mul(16)?))
        .and_then(|n| n.checked_add(size_of::<NextTokenReport>()))
        .and_then(|n| n.checked_add(certificate_owner_bytes()))
        .ok_or(DecoderRefusal::Resource("replay memory overflow"))?;
    let head_bytes = incremental_head_bytes(head, head_owner.as_ref())?;
    let persistent = evidence_bytes
        .checked_add(head_bytes)
        .ok_or(DecoderRefusal::Resource("replay resident size"))?;
    let scratch = checked_mul(head.hidden_size, 32)?
        .checked_add(head.vocab_size)
        .ok_or(DecoderRefusal::Resource("replay scratch size"))?;
    if resident_with_owner(persistent, head_owner.as_ref())? as u64 > options.max_persistent_bytes
        || scratch as u64 > options.max_scratch_bytes
    {
        return Err(DecoderRefusal::Resource("replay memory cap"));
    }
    let work = (head.weights.len() as u64)
        .checked_add(head.bias.len() as u64)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| {
            n.checked_add(
                (member_count as u64)
                    .checked_mul(head.hidden_size as u64)?
                    .checked_mul(4)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                (evidence.groups.len() as u64)
                    .checked_mul(head.hidden_size as u64)?
                    .checked_mul(4)?,
            )
        })
        .and_then(|n| n.checked_add((member_count as u64).checked_mul(9)?.checked_add(2)?))
        .and_then(|n| n.checked_add(domain_work as u64))
        .ok_or(DecoderRefusal::Resource("replay work overflow"))?;
    let lease = budget
        .lease_for(
            LeaseClass::Replay,
            WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: persistent,
                work,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    coverage(evidence, options.max_vocab, options.max_nodes)?;
    meter.charge(
        member_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(1))
            .ok_or(DecoderRefusal::Resource("input cover work"))?,
    )?;
    let checked_identity = identity(head, &mut meter)?;
    if let Some(owner) = &head_owner {
        owner.validate_identity(&checked_identity)?;
    }
    let scope = scope(&checked_identity, hidden, anchor, target)?;
    if evidence.scope_digest != scope_digest(&scope) {
        return Err(DecoderRefusal::Evidence(
            "head, box, anchor or numeric scope",
        ));
    }
    let rad = radius(hidden, anchor)?;
    let mut unresolved = Vec::with_capacity(member_count);
    let mut stats = VocabStats {
        persistent_bytes: persistent as u64,
        scratch_bytes: scratch as u64,
        ..VocabStats::default()
    };
    for group in &evidence.groups {
        meter.checkpoint()?;
        let upper = if let Some(claim) = &group.group_bounds {
            let geom = geometry(head, &group.members, group.anchor_token, &mut meter)?;
            let mut at_anchor = f64::NEG_INFINITY;
            for &id in &group.members {
                meter.charge(head.hidden_size)?;
                at_anchor = at_anchor.max(anchor_margin(head, anchor, target, id)?);
            }
            meter.charge(checked_mul(head.hidden_size, 4)?)?;
            let checked = group_bound(
                head,
                hidden,
                anchor,
                target,
                group.anchor_token,
                at_anchor,
                &geom,
                &rad,
            )?;
            // Exact deterministic replay rejects poisoned radii and every field
            // used in a claimed minimum, including componentwise alternatives.
            if &checked != claim {
                return Err(DecoderRefusal::Evidence("group summary changed"));
            }
            checked.upper
        } else {
            let mut upper = f64::NEG_INFINITY;
            for &id in &group.members {
                meter.charge(checked_mul(head.hidden_size, 3)?)?;
                let bound = dense_upper(head, hidden, target, id)?;
                upper = upper.max(bound);
                stats.dense_rows += 1;
                if bound >= 0.0 {
                    unresolved.push(id);
                }
            }
            upper
        };
        if upper.to_bits() != group.upper.to_bits() {
            return Err(DecoderRefusal::Evidence("claimed group upper"));
        }
        if upper >= 0.0 && group.group_bounds.is_some() {
            unresolved.extend(&group.members);
        }
        stats.groups_visited += 1;
        if upper < 0.0 {
            stats.groups_excluded += 1;
        }
    }
    meter.checkpoint()?;
    meter.charge(
        member_count
            .checked_mul(5)
            .and_then(|n| n.checked_add(1))
            .ok_or(DecoderRefusal::Resource("output cover work"))?,
    )?;
    stats.query_terms = meter.terms();
    drop(rad);
    if head_owner.is_none() {
        finish(
            scope,
            evidence.clone(),
            unresolved,
            stats,
            lease,
            &budget,
            options,
            LeaseClass::Replay,
            NextTokenProvenance::OriginalReplay,
        )
    } else {
        finish_with_owner(
            scope,
            evidence.clone(),
            unresolved,
            stats,
            lease,
            &budget,
            options,
            LeaseClass::Replay,
            head_owner,
            NextTokenProvenance::OriginalReplay,
        )
    }
}

/// Replay v1 explicit-ID or v2 original-ID range evidence. Encoded parser
/// storage and full v2 expansion are independently admitted before allocation;
/// both routes use the same original-head arithmetic and strict full cover.
pub fn replay_next_token_json(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    json: &[u8],
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    super::wire::replay_json(head, hidden, anchor, target, json, options)
}

/// Inexpensive dense reference: exact-real box support of each original margin
/// row, plus only the pinned executed-head allowance when requested. It also
/// produces full-cover evidence and uses the same independent replay gate.
pub fn verify_dense_next_token(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    verify_dense_with_owner(head, hidden, anchor, target, options, None)
}

pub(super) fn verify_dense_with_owner(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
    head_owner: Option<SharedAffineHead>,
) -> Result<NextTokenReport, DecoderRefusal> {
    let scoped = head_owner
        .as_ref()
        .map(|owner| owner.scoped_options(options))
        .transpose()?;
    let options = scoped.as_ref().unwrap_or(options);
    let budget = default_budget(options)?;
    let mut meter = Meter::with_budget_class(options, budget.clone(), LeaseClass::Fallback)?;
    validate_head(head, anchor, target, options)?;
    validate_box(hidden, head.hidden_size)?;
    super::execution::validate_input_domain(head.arithmetic, hidden)?;
    let domain_work = if matches!(head.arithmetic, super::HeadArithmetic::Binary32 { .. }) {
        head.hidden_size
    } else {
        0
    };
    meter.charge(domain_work)?;
    let scratch = checked_mul(head.vocab_size, 16)?
        .checked_add(checked_mul(head.hidden_size, 16)?)
        .and_then(|n| n.checked_add(size_of::<GroupEvidence>()))
        .and_then(|n| n.checked_add(size_of::<VocabEvidence>()))
        .and_then(|n| n.checked_add(size_of::<CertificateScope>()))
        .ok_or(DecoderRefusal::Resource("dense allocation size"))?;
    let head_bytes = incremental_head_bytes(head, head_owner.as_ref())?;
    if scratch as u64 > options.max_scratch_bytes
        || resident_with_owner(head_bytes, head_owner.as_ref())? as u64
            > options.max_persistent_bytes
    {
        return Err(DecoderRefusal::Resource("dense memory cap"));
    }
    let work = (head.weights.len() as u64)
        .checked_add(head.bias.len() as u64)
        .and_then(|n| {
            n.checked_add(
                (head.vocab_size as u64)
                    .checked_mul(head.hidden_size as u64)?
                    .checked_mul(3)?,
            )
        })
        .and_then(|n| n.checked_add(head.vocab_size as u64))
        .and_then(|n| n.checked_add(domain_work as u64))
        .ok_or(DecoderRefusal::Resource("dense work"))?;
    // The original input head is also counted by replay. This lease retains
    // only temporary proposal allocation; replay independently admits its head.
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: 0,
                work,
                proposals: 1,
            },
        )
        .map_err(DecoderRefusal::Budget)?;
    let scope = scope(&identity(head, &mut meter)?, hidden, anchor, target)?;
    meter.charge(head.vocab_size - 1)?;
    let mut members = Vec::with_capacity(head.vocab_size - 1);
    members.extend((0..head.vocab_size).filter(|&id| id != target));
    let mut upper = f64::NEG_INFINITY;
    for &id in &members {
        meter.charge(checked_mul(head.hidden_size, 3)?)?;
        upper = upper.max(dense_upper(head, hidden, target, id)?);
    }
    let groups = if members.is_empty() {
        Vec::new()
    } else {
        vec![GroupEvidence {
            anchor_token: members[0],
            members,
            group_bounds: None,
            upper,
        }]
    };
    let evidence = VocabEvidence {
        schema_version: 1,
        scope_digest: scope_digest(&scope),
        target,
        vocab_size: head.vocab_size,
        hidden_size: head.hidden_size,
        groups,
    };
    let mut replay_options = options.clone();
    replay_options.budget = Some(budget);
    let mut report = replay_with_owner(
        head,
        hidden,
        anchor,
        target,
        &evidence,
        &replay_options,
        head_owner,
    )?;
    report.stats.preprocessing_terms = meter.terms();
    lease.complete().map_err(DecoderRefusal::Budget)?;
    Ok(report)
}
