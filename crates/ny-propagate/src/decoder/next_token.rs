// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete deterministic group traversal and strict next-token decisions.
use super::certify::{
    certificate_owner_bytes, finish_with_owner, NextTokenOutcome, NextTokenProvenance,
};
use super::numeric::{dense_upper, radius};
use super::vocab_tree::{
    checked_mul, group_bound, prepare_vocab_tree, scope, scope_digest, validate_box,
    PreparedVocabTree,
};
use super::{
    AffineHead, DecoderRefusal, GroupEvidence, HiddenBox, Meter, SharedAffineHead, VocabEvidence,
    VocabOptions, VocabStats,
};
use crate::certify::work::{LeaseClass, PersistentReservation, WorkRequest};

/// The outcome is conditional on this exact head and supplied certified box.
/// Mutating exported evidence cannot construct or alter a certified outcome.
#[derive(Debug)]
pub struct NextTokenReport {
    pub outcome: NextTokenOutcome,
    pub evidence: VocabEvidence,
    pub unresolved: Vec<usize>,
    pub stats: VocabStats,
    pub(super) reservation: PersistentReservation,
    pub(super) head_owner: Option<SharedAffineHead>,
    pub(super) provenance: NextTokenProvenance,
}
impl NextTokenReport {
    pub fn is_certified(&self) -> bool {
        matches!(self.outcome, NextTokenOutcome::Certified(_))
    }
    /// Origin of the sealed decision, not an endorsement of raw `evidence`
    /// after mutation. CheckedQuery already has valid private checker authority.
    pub fn provenance(&self) -> NextTokenProvenance {
        // `outcome` is a compatibility public field. Replacing it cannot lend
        // the previous report's origin to a different sealed certificate.
        match &self.outcome {
            NextTokenOutcome::Certified(proof) => proof.provenance(),
            NextTokenOutcome::Unknown => self.provenance,
        }
    }
    /// Bytes remain charged while the report or a cloned certificate lives.
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub fn shared_head_bytes(&self) -> usize {
        self.head_owner
            .as_ref()
            .map_or(0, SharedAffineHead::reserved_bytes)
    }
}

/// Cold convenience path. Warm workloads should explicitly prepare one tree.
pub fn verify_next_token(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    prepare_vocab_tree(head, anchor, target, options)?.verify(hidden, options)
}

pub(super) fn query(
    tree: &PreparedVocabTree<'_>,
    hidden: HiddenBox<'_>,
    options: &VocabOptions,
) -> Result<NextTokenReport, DecoderRefusal> {
    let mut meter = Meter::with_budget(options, tree.budget.clone())?;
    validate_box(hidden, tree.head.hidden_size)?;
    super::execution::validate_input_domain(tree.head.arithmetic, hidden)?;
    let domain_work = if matches!(tree.head.arithmetic, super::HeadArithmetic::Binary32 { .. }) {
        tree.head.hidden_size
    } else {
        0
    };
    meter.charge(domain_work)?;
    if tree.head.vocab_size > options.max_vocab || tree.head.hidden_size > options.max_hidden {
        return Err(DecoderRefusal::Resource("query dimension cap"));
    }
    let evidence_bytes = checked_mul(tree.competitors.len(), 16)?
        .checked_add(checked_mul(tree.nodes.len(), size_of::<GroupEvidence>())?)
        .and_then(|n| n.checked_add(tree.head.hidden_size.checked_mul(16)?))
        .and_then(|n| n.checked_add(size_of::<NextTokenReport>()))
        .and_then(|n| n.checked_add(certificate_owner_bytes()))
        .ok_or(DecoderRefusal::Resource("query evidence size"))?;
    let scratch = checked_mul(tree.head.hidden_size, 16)?
        .checked_add(checked_mul(tree.head.vocab_size, 2)?)
        .and_then(|n| n.checked_add(tree.nodes.len().checked_mul(size_of::<usize>())?))
        .ok_or(DecoderRefusal::Resource("query scratch size"))?;
    if tree
        .stats
        .persistent_bytes
        .saturating_add(evidence_bytes as u64)
        > options.max_persistent_bytes
        || scratch as u64 > options.max_scratch_bytes
    {
        return Err(DecoderRefusal::Resource("query memory cap"));
    }
    let work = (tree.nodes.len() as u64)
        .checked_mul(tree.head.hidden_size as u64)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| {
            n.checked_add(
                (tree.competitors.len() as u64)
                    .checked_mul(tree.head.hidden_size as u64)?
                    .checked_mul(3)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                (tree.competitors.len() as u64)
                    .checked_mul(5)?
                    .checked_add(1)?,
            )
        })
        .and_then(|n| n.checked_add(tree.head.hidden_size as u64))
        .and_then(|n| n.checked_add(domain_work as u64))
        .ok_or(DecoderRefusal::Resource("query work size"))?;
    let lease = tree
        .budget
        .lease(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: evidence_bytes,
            work,
            proposals: 1,
        })
        .map_err(DecoderRefusal::Budget)?;
    meter.charge(tree.head.hidden_size)?;
    let rad = radius(hidden, &tree.anchor)?;
    let mut stack = Vec::with_capacity(tree.nodes.len());
    if !tree.nodes.is_empty() {
        stack.push(0);
    }
    let mut groups = Vec::with_capacity(tree.nodes.len());
    let mut unresolved = Vec::with_capacity(tree.competitors.len());
    let mut stats = tree.stats.clone();
    stats.scratch_bytes = scratch as u64;
    while let Some(index) = stack.pop() {
        meter.checkpoint()?;
        if stats.groups_visited >= options.max_nodes {
            return Err(DecoderRefusal::Resource("query node cap"));
        }
        stats.groups_visited += 1;
        let node = &tree.nodes[index];
        let members = &tree.competitors[node.start..node.end];
        meter.charge(checked_mul(tree.head.hidden_size, 4)?)?;
        let bounds = group_bound(
            tree.head,
            hidden,
            &tree.anchor,
            tree.target,
            node.anchor_id,
            node.anchor_margin,
            &node.geometry,
            &rad,
        )?;
        if bounds.upper < 0.0 {
            stats.groups_excluded += 1;
            groups.push(GroupEvidence {
                members: members.to_vec(),
                anchor_token: node.anchor_id,
                upper: bounds.upper,
                group_bounds: Some(bounds),
            });
        } else if let Some((left, right)) = node.children {
            // Left-first vocabulary order is deterministic, independent of timing.
            stack.push(right);
            stack.push(left);
        } else {
            let mut upper = f64::NEG_INFINITY;
            for &id in members {
                meter.charge(checked_mul(tree.head.hidden_size, 3)?)?;
                let bound = dense_upper(tree.head, hidden, tree.target, id)?;
                upper = upper.max(bound);
                stats.dense_rows += 1;
                if bound >= 0.0 {
                    unresolved.push(id);
                }
            }
            groups.push(GroupEvidence {
                members: members.to_vec(),
                anchor_token: node.anchor_id,
                upper,
                group_bounds: None,
            });
        }
    }
    let scope = scope(&tree.identity, hidden, &tree.anchor, tree.target)?;
    let evidence = VocabEvidence {
        schema_version: 1,
        scope_digest: scope_digest(&scope),
        target: tree.target,
        vocab_size: tree.head.vocab_size,
        hidden_size: tree.head.hidden_size,
        groups,
    };
    // Export copies every ID; the independent cover ledger initializes,
    // validates and checks its full vocabulary bitmap before minting.
    meter.charge(
        tree.competitors
            .len()
            .checked_mul(5)
            .and_then(|n| n.checked_add(1))
            .ok_or(DecoderRefusal::Resource("cover work"))?,
    )?;
    stats.query_terms = meter.terms();
    stats.persistent_bytes = stats.persistent_bytes.saturating_add(evidence_bytes as u64);
    meter.checkpoint()?;
    drop(rad);
    drop(stack);
    finish_with_owner(
        scope,
        evidence,
        unresolved,
        stats,
        lease,
        &tree.budget,
        options,
        LeaseClass::Optional,
        tree.head_owner.clone(),
        NextTokenProvenance::CheckedQuery,
    )
}
