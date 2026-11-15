// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Target-independent checked vocabulary geometry and complete changing-target queries.
//!
//! Cold construction checks the original immutable head and every node's exact
//! membership. A query descends the target path, emits target-free sibling
//! covers, and checks the target leaf's remaining members separately. Cached
//! geometry is never relabeled with a different anchor or membership. Exported
//! evidence uses the existing original-head D2 replay, which rebuilds all
//! terminal geometry independently and charges that work separately.
use super::certify::{finish_with_owner, NextTokenProvenance};
use super::next_token::NextTokenReport;
use super::numeric::{anchor_margin, dense_upper, radius};
use super::shared_head::{incremental_head_bytes, resident_with_owner};
use super::vocab_tree::{
    checked_mul, default_budget, geometry, group_bound, identity, scope, scope_digest,
    validate_box, validate_head, Geometry,
};
use super::{
    AffineHead, DecoderRefusal, GroupEvidence, HeadArithmetic, HiddenBox, Meter, SharedAffineHead,
    VocabEvidence, VocabOptions, VocabStats,
};
use crate::certify::scope::ScopeIdentity;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};

#[derive(Debug)]
struct ModelNode {
    start: usize,
    end: usize,
    geometry: Geometry,
    children: Option<(usize, usize)>,
}

/// Actual cold checker terms and conservative capacity charges. Terms count
/// checked scalar coordinates/IDs, using the same convention as D2 statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelTreePreparationStats {
    pub identity_terms: u64,
    pub geometry_terms: u64,
    pub membership_terms: u64,
    pub nodes: usize,
    pub persistent_bytes: u64,
    pub scratch_bytes: u64,
}

/// Actual warm work, separate from cold geometry and optional later replay.
/// `anchor_terms` still scans every original competitor at the supplied anchor.
/// Cached geometry construction costs zero warm terms; newly formed target-leaf
/// subsets, dense fallback, and the complete coverage ledger remain charged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelTreeQueryStats {
    pub anchor_terms: u64,
    pub cached_geometry_reuses: usize,
    pub subset_geometry_terms: u64,
    pub bound_terms: u64,
    pub dense_terms: u64,
    pub membership_terms: u64,
    pub coverage_terms: u64,
    pub query_terms: u64,
}

/// Same private D2 certification outcome and strict replayable evidence, with
/// the model-tree cold/warm work decomposition retained alongside it.
#[derive(Debug)]
pub struct ModelTreeReport {
    pub report: NextTokenReport,
    pub stats: ModelTreeQueryStats,
}
impl ModelTreeReport {
    pub fn is_certified(&self) -> bool {
        self.report.is_certified()
    }
}

/// Immutable model-level geometry. Target, anchor, and hidden box are deliberately
/// absent from the cache identity. Head rows and biases remain borrowed immutable;
/// arithmetic and every original coefficient are bound by `head_identity`.
/// No Clone: another allocation needs another admission, while budget clones
/// retain one common allowance across changing-target queries.
#[derive(Debug)]
pub struct PreparedModelVocabTree<'a> {
    head: AffineHead<'a>,
    nodes: Vec<ModelNode>,
    head_identity: ScopeIdentity,
    stats: ModelTreePreparationStats,
    budget: WorkBudget,
    reservation: PersistentReservation,
    head_owner: Option<SharedAffineHead>,
}
impl PreparedModelVocabTree<'_> {
    pub fn head_identity(&self) -> &ScopeIdentity {
        &self.head_identity
    }
    pub fn preparation_stats(&self) -> &ModelTreePreparationStats {
        &self.stats
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub fn shared_head_bytes(&self) -> usize {
        self.head_owner
            .as_ref()
            .map_or(0, SharedAffineHead::reserved_bytes)
    }
    pub fn budget(&self) -> WorkBudget {
        self.budget.clone()
    }
    /// Complete vocabulary verification for any target/anchor/box. A different
    /// budget in `options` cannot replace this tree's existing resident ledger.
    pub fn verify(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        options: &VocabOptions,
    ) -> Result<ModelTreeReport, DecoderRefusal> {
        query(self, hidden, anchor, target, options)
    }
    /// Independent original-head replay. All summaries and the whole head
    /// identity are recomputed; returned D2 statistics report that separate work.
    pub fn replay(
        &self,
        hidden: HiddenBox<'_>,
        anchor: &[f64],
        target: usize,
        evidence: &VocabEvidence,
        options: &VocabOptions,
    ) -> Result<NextTokenReport, DecoderRefusal> {
        let mut replay_options = options.clone();
        replay_options.budget = Some(self.budget.clone());
        super::certify::replay_with_owner(
            self.head,
            hidden,
            anchor,
            target,
            evidence,
            &replay_options,
            self.head_owner.clone(),
        )
    }
}

fn add(a: usize, b: usize) -> Result<usize, DecoderRefusal> {
    a.checked_add(b)
        .ok_or(DecoderRefusal::Resource("model tree size overflow"))
}
fn work_add(a: u64, b: u64) -> Result<u64, DecoderRefusal> {
    a.checked_add(b)
        .ok_or(DecoderRefusal::Resource("model tree work overflow"))
}
fn work_mul(a: u64, b: u64) -> Result<u64, DecoderRefusal> {
    a.checked_mul(b)
        .ok_or(DecoderRefusal::Resource("model tree work overflow"))
}

/// Prepare complete target-independent geometry once. Deterministic contiguous
/// vocabulary ranges retain every row, including duplicates and future targets.
/// All original head bytes (although borrowed), node vector capacities, geometry
/// vectors, and temporary ID storage are admitted before proportional allocation.
pub fn prepare_model_vocab_tree<'a>(
    head: AffineHead<'a>,
    options: &VocabOptions,
) -> Result<PreparedModelVocabTree<'a>, DecoderRefusal> {
    prepare_with_owner(head, options, None)
}

pub(super) fn prepare_with_owner<'a>(
    head: AffineHead<'a>,
    options: &VocabOptions,
    head_owner: Option<SharedAffineHead>,
) -> Result<PreparedModelVocabTree<'a>, DecoderRefusal> {
    let scoped = head_owner
        .as_ref()
        .map(|owner| owner.scoped_options(options))
        .transpose()?;
    let options = scoped.as_ref().unwrap_or(options);
    let budget = default_budget(options)?;
    let mut meter = Meter::with_budget(options, budget.clone())?;
    if head.vocab_size == 0
        || head.hidden_size == 0
        || head.bias.len() != head.vocab_size
        || head.weights.len() != checked_mul(head.vocab_size, head.hidden_size)?
    {
        return Err(DecoderRefusal::Shape("model head"));
    }
    if head.vocab_size > options.max_vocab || head.hidden_size > options.max_hidden {
        return Err(DecoderRefusal::Resource("model head dimension cap"));
    }
    let leaf_count = head.vocab_size.div_ceil(options.leaf_size);
    let power = leaf_count
        .checked_next_power_of_two()
        .ok_or(DecoderRefusal::Resource("model tree size"))?;
    let node_cap = power
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(DecoderRefusal::Resource("model tree size"))?;
    if node_cap > options.max_nodes {
        return Err(DecoderRefusal::Resource("model tree node cap"));
    }
    let levels = power.trailing_zeros() as usize + 1;
    let vectors = if head.arithmetic != HeadArithmetic::ExactReal {
        2
    } else {
        1
    };
    let head_bytes = incremental_head_bytes(head, head_owner.as_ref())?;
    let geometry_bytes = checked_mul(checked_mul(node_cap, head.hidden_size)?, 8 * vectors)?;
    let metadata = checked_mul(node_cap, size_of::<ModelNode>())?;
    let persistent = add(
        add(add(head_bytes, geometry_bytes)?, metadata)?,
        size_of::<PreparedModelVocabTree<'a>>(),
    )?;
    let scratch = add(
        add(
            checked_mul(head.vocab_size, size_of::<usize>())?,
            checked_mul(head.hidden_size, 32)?,
        )?,
        checked_mul(levels, 256)?,
    )?;
    if resident_with_owner(persistent, head_owner.as_ref())? as u64 > options.max_persistent_bytes
        || scratch as u64 > options.max_scratch_bytes
    {
        return Err(DecoderRefusal::Resource(
            "model tree preparation memory cap",
        ));
    }
    let work = work_add(
        add(head.weights.len(), head.bias.len())? as u64,
        work_add(
            work_mul(
                work_mul(head.vocab_size as u64, head.hidden_size as u64)?,
                levels as u64,
            )?,
            work_add(head.vocab_size as u64, node_cap as u64)?,
        )?,
    )?;
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: persistent,
            work,
            proposals: 1,
        })
        .map_err(DecoderRefusal::Budget)?;
    let head_identity = identity(head, &mut meter)?;
    if let Some(owner) = &head_owner {
        owner.validate_identity(&head_identity)?;
    }
    let identity_terms = meter.terms();
    let members: Vec<usize> = (0..head.vocab_size).collect();
    meter.charge(head.vocab_size)?;
    let mut nodes = Vec::with_capacity(node_cap);
    fn build(
        head: AffineHead<'_>,
        members: &[usize],
        start: usize,
        end: usize,
        leaf: usize,
        nodes: &mut Vec<ModelNode>,
        meter: &mut Meter<'_>,
    ) -> Result<(), DecoderRefusal> {
        meter.charge(1)?;
        let index = nodes.len();
        let checked = geometry(head, &members[start..end], start, meter)?;
        nodes.push(ModelNode {
            start,
            end,
            geometry: checked,
            children: None,
        });
        if end - start > leaf {
            let middle = start + (end - start) / 2;
            let left = nodes.len();
            build(head, members, start, middle, leaf, nodes, meter)?;
            let right = nodes.len();
            build(head, members, middle, end, leaf, nodes, meter)?;
            nodes[index].children = Some((left, right));
        }
        Ok(())
    }
    build(
        head,
        &members,
        0,
        head.vocab_size,
        options.leaf_size,
        &mut nodes,
        &mut meter,
    )?;
    let membership_terms = work_add(head.vocab_size as u64, nodes.len() as u64)?;
    let stats = ModelTreePreparationStats {
        identity_terms,
        geometry_terms: meter.terms() - identity_terms - membership_terms,
        membership_terms,
        nodes: nodes.len(),
        persistent_bytes: resident_with_owner(persistent, head_owner.as_ref())? as u64,
        scratch_bytes: scratch as u64,
    };
    drop(members);
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    meter.checkpoint()?;
    Ok(PreparedModelVocabTree {
        head,
        nodes,
        head_identity,
        stats,
        budget,
        reservation,
        head_owner,
    })
}

fn query(
    tree: &PreparedModelVocabTree<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
) -> Result<ModelTreeReport, DecoderRefusal> {
    let mut meter = Meter::with_budget(options, tree.budget.clone())?;
    validate_head(tree.head, anchor, target, options)?;
    validate_box(hidden, tree.head.hidden_size)?;
    super::execution::validate_input_domain(tree.head.arithmetic, hidden)?;
    if tree.nodes.len() > options.max_nodes {
        return Err(DecoderRefusal::Resource("model query node cap"));
    }
    let n = tree.head.vocab_size;
    let d = tree.head.hidden_size;
    let count = n - 1;
    let groups_bytes = checked_mul(tree.nodes.len(), size_of::<GroupEvidence>())?;
    let evidence_bytes = add(
        add(
            add(checked_mul(count, 16)?, groups_bytes)?,
            checked_mul(d, 16)?,
        )?,
        size_of::<ModelTreeReport>()
            + size_of::<crate::certify::scope::CertificateScope>()
            + 2 * size_of::<usize>(),
    )?;
    let scratch = add(
        add(checked_mul(n, 8)?, checked_mul(tree.nodes.len(), 16)?)?,
        add(checked_mul(d, 32)?, n)?,
    )?;
    if add(tree.stats.persistent_bytes as usize, evidence_bytes)? as u64
        > options.max_persistent_bytes
        || scratch as u64 > options.max_scratch_bytes
        || evidence_bytes > options.max_evidence_bytes
    {
        return Err(DecoderRefusal::Resource("model query memory cap"));
    }
    // Includes worst-case original anchor scans, new subset geometry, cached
    // bound checks, dense fallback and all coverage/maximum/stack ID visits.
    let work = work_add(
        work_mul(work_mul(n as u64, d as u64)?, 5)?,
        work_add(
            if matches!(tree.head.arithmetic, HeadArithmetic::Binary32 { .. }) {
                d as u64
            } else {
                0
            },
            work_add(
                work_mul(work_mul(tree.nodes.len() as u64, d as u64)?, 4)?,
                work_add(
                    work_mul(n as u64, 12)?,
                    work_mul(tree.nodes.len() as u64, 3)?,
                )?,
            )?,
        )?,
    )?;
    let lease = tree
        .budget
        .lease(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: evidence_bytes,
            work,
            proposals: 1,
        })
        .map_err(DecoderRefusal::Budget)?;
    let mut stats = ModelTreeQueryStats::default();
    if matches!(tree.head.arithmetic, HeadArithmetic::Binary32 { .. }) {
        meter.charge(d)?;
    }
    meter.charge(d)?;
    let rad = radius(hidden, anchor)?;
    let mut margins = Vec::with_capacity(n);
    for id in 0..n {
        if id == target {
            margins.push(f64::NEG_INFINITY);
        } else {
            meter.charge(d)?;
            stats.anchor_terms = work_add(stats.anchor_terms, d as u64)?;
            margins.push(anchor_margin(tree.head, anchor, target, id)?);
        }
    }
    let mut maximums = vec![f64::NEG_INFINITY; tree.nodes.len()];
    for (index, node) in tree.nodes.iter().enumerate().rev() {
        meter.charge(1)?;
        stats.membership_terms = work_add(stats.membership_terms, 1)?;
        maximums[index] = if let Some((left, right)) = node.children {
            maximums[left].max(maximums[right])
        } else {
            meter.charge(node.end - node.start)?;
            stats.membership_terms =
                work_add(stats.membership_terms, (node.end - node.start) as u64)?;
            margins[node.start..node.end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        };
    }
    let mut stack = Vec::with_capacity(tree.nodes.len());
    stack.push(0);
    let mut groups = Vec::with_capacity(tree.nodes.len());
    let mut unresolved = Vec::with_capacity(count);
    let mut d2_stats = VocabStats {
        preprocessing_terms: work_add(
            work_add(tree.stats.identity_terms, tree.stats.geometry_terms)?,
            tree.stats.membership_terms,
        )?,
        persistent_bytes: add(tree.stats.persistent_bytes as usize, evidence_bytes)? as u64,
        scratch_bytes: scratch as u64,
        ..VocabStats::default()
    };
    while let Some(index) = stack.pop() {
        meter.charge(1)?;
        stats.membership_terms = work_add(stats.membership_terms, 1)?;
        d2_stats.groups_visited += 1;
        let node = &tree.nodes[index];
        let contains_target = node.start <= target && target < node.end;
        if contains_target {
            if let Some((left, right)) = node.children {
                stack.push(right);
                stack.push(left);
                continue;
            }
            // Filtered collect may grow to a minimum capacity above its length.
            // The admitted target-leaf remainder has an exact known size.
            let mut members = Vec::with_capacity(node.end - node.start - 1);
            members.extend((node.start..node.end).filter(|&id| id != target));
            if members.is_empty() {
                continue;
            }
            // This subset has a different membership (and possibly anchor).
            // Its full-node superset radius never enters certificate evidence.
            let anchor_id = members[0];
            let before = meter.terms();
            let subset = geometry(tree.head, &members, anchor_id, &mut meter)?;
            stats.subset_geometry_terms =
                work_add(stats.subset_geometry_terms, meter.terms() - before)?;
            meter.charge(checked_mul(d, 4)?)?;
            stats.bound_terms = work_add(stats.bound_terms, checked_mul(d, 4)? as u64)?;
            let bound = group_bound(
                tree.head,
                hidden,
                anchor,
                target,
                anchor_id,
                maximums[index],
                &subset,
                &rad,
            )?;
            if bound.upper < 0.0 {
                d2_stats.groups_excluded += 1;
                groups.push(GroupEvidence {
                    members,
                    anchor_token: anchor_id,
                    upper: bound.upper,
                    group_bounds: Some(bound),
                });
            } else {
                dense_group(
                    tree.head,
                    hidden,
                    target,
                    members,
                    &mut meter,
                    &mut stats,
                    &mut d2_stats,
                    &mut unresolved,
                    &mut groups,
                )?;
            }
            continue;
        }
        // Exactly this range and its start anchor were checked at cold build.
        // Both are target-free here; the original borrowed head is immutable.
        stats.cached_geometry_reuses += 1;
        meter.charge(checked_mul(d, 4)?)?;
        stats.bound_terms = work_add(stats.bound_terms, checked_mul(d, 4)? as u64)?;
        let bound = group_bound(
            tree.head,
            hidden,
            anchor,
            target,
            node.start,
            maximums[index],
            &node.geometry,
            &rad,
        )?;
        if bound.upper < 0.0 {
            d2_stats.groups_excluded += 1;
            groups.push(GroupEvidence {
                members: (node.start..node.end).collect(),
                anchor_token: node.start,
                upper: bound.upper,
                group_bounds: Some(bound),
            });
        } else if let Some((left, right)) = node.children {
            stack.push(right);
            stack.push(left);
        } else {
            dense_group(
                tree.head,
                hidden,
                target,
                (node.start..node.end).collect(),
                &mut meter,
                &mut stats,
                &mut d2_stats,
                &mut unresolved,
                &mut groups,
            )?;
        }
    }
    let scope = scope(&tree.head_identity, hidden, anchor, target)?;
    let evidence = VocabEvidence {
        schema_version: 1,
        scope_digest: scope_digest(&scope),
        target,
        vocab_size: n,
        hidden_size: d,
        groups,
    };
    let coverage_terms = add(checked_mul(count, 5)?, 1)?;
    meter.charge(coverage_terms)?;
    stats.coverage_terms = coverage_terms as u64;
    stats.query_terms = meter.terms();
    d2_stats.query_terms = stats.query_terms;
    drop(rad);
    drop(margins);
    drop(maximums);
    drop(stack);
    let report = finish_with_owner(
        scope,
        evidence,
        unresolved,
        d2_stats,
        lease,
        &tree.budget,
        options,
        LeaseClass::Optional,
        tree.head_owner.clone(),
        NextTokenProvenance::CheckedQuery,
    )?;
    Ok(ModelTreeReport { report, stats })
}

#[allow(clippy::too_many_arguments)]
fn dense_group(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    target: usize,
    members: Vec<usize>,
    meter: &mut Meter<'_>,
    stats: &mut ModelTreeQueryStats,
    d2_stats: &mut VocabStats,
    unresolved: &mut Vec<usize>,
    groups: &mut Vec<GroupEvidence>,
) -> Result<(), DecoderRefusal> {
    let mut upper = f64::NEG_INFINITY;
    for &id in &members {
        let terms = checked_mul(head.hidden_size, 3)?;
        meter.charge(terms)?;
        stats.dense_terms = work_add(stats.dense_terms, terms as u64)?;
        let bound = dense_upper(head, hidden, target, id)?;
        upper = upper.max(bound);
        d2_stats.dense_rows += 1;
        if bound >= 0.0 {
            unresolved.push(id);
        }
    }
    if upper < 0.0 {
        d2_stats.groups_excluded += 1;
    }
    groups.push(GroupEvidence {
        anchor_token: members[0],
        members,
        group_bounds: None,
        upper,
    });
    Ok(())
}
