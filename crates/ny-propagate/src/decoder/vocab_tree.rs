// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Deterministic complete-vocabulary geometry, checked once from original rows.
use super::numeric::*;
use super::shared_head::{incremental_head_bytes, resident_with_owner};
use super::{
    AffineHead, DecoderRefusal, GroupBounds, HeadArithmetic, HiddenBox, Meter, SharedAffineHead,
    VocabOptions, VocabStats,
};
use crate::certify::scope::{
    BoxRegion, CertificateScope, Digest, NumericalSemantics, ScopeIdentity, SCOPE_SCHEMA_VERSION,
};
use crate::certify::work::{PersistentReservation, WorkBudget, WorkRequest};
use sha2::{Digest as _, Sha256};
use std::time::Duration;

#[derive(Debug)]
pub(super) struct Geometry {
    pub component: Vec<f64>,
    pub l1: f64,
    pub l2: f64,
    pub linf: f64,
    pub absolute_weights: Vec<f64>,
    pub absolute_bias: f64,
}
#[derive(Debug)]
pub(super) struct Node {
    pub start: usize,
    pub end: usize,
    pub anchor_id: usize,
    pub anchor_margin: f64,
    pub geometry: Geometry,
    pub children: Option<(usize, usize)>,
}

/// Checked immutable cache; original coefficients are borrowed and cannot be
/// mutated while it exists. Preparation and query evidence retain budget leases.
#[derive(Debug)]
pub struct PreparedVocabTree<'a> {
    pub(super) head: AffineHead<'a>,
    pub(super) target: usize,
    pub(super) anchor: Vec<f64>,
    pub(super) competitors: Vec<usize>,
    pub(super) nodes: Vec<Node>,
    pub(super) identity: ScopeIdentity,
    pub(super) stats: VocabStats,
    pub(super) budget: WorkBudget,
    pub(super) reservation: PersistentReservation,
    pub(super) head_owner: Option<SharedAffineHead>,
}
impl PreparedVocabTree<'_> {
    pub fn target(&self) -> usize {
        self.target
    }
    pub fn anchor(&self) -> &[f64] {
        &self.anchor
    }
    pub fn preparation_stats(&self) -> &VocabStats {
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
    /// Queries share this tree's original ledger; they cannot obtain fresh
    /// resident allowances by substituting a different budget in `options`.
    pub fn verify(
        &self,
        hidden: HiddenBox<'_>,
        options: &VocabOptions,
    ) -> Result<super::next_token::NextTokenReport, DecoderRefusal> {
        super::next_token::query(self, hidden, options)
    }
}

pub(super) fn checked_mul(a: usize, b: usize) -> Result<usize, DecoderRefusal> {
    a.checked_mul(b)
        .ok_or(DecoderRefusal::Resource("dimension overflow"))
}
pub(super) fn validate_head(
    head: AffineHead<'_>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
) -> Result<(), DecoderRefusal> {
    if head.vocab_size == 0
        || head.hidden_size == 0
        || target >= head.vocab_size
        || anchor.len() != head.hidden_size
        || head.bias.len() != head.vocab_size
    {
        return Err(DecoderRefusal::Shape("head, target or anchor"));
    }
    if head.vocab_size > options.max_vocab || head.hidden_size > options.max_hidden {
        return Err(DecoderRefusal::Resource("head dimension cap"));
    }
    if head.weights.len() != checked_mul(head.vocab_size, head.hidden_size)? {
        return Err(DecoderRefusal::Shape("row-major head weights"));
    }
    if !anchor.iter().all(|x| x.is_finite()) {
        return Err(DecoderRefusal::NonFinite("head or anchor"));
    }
    Ok(())
}
pub(super) fn validate_box(hidden: HiddenBox<'_>, dim: usize) -> Result<(), DecoderRefusal> {
    if hidden.lo.len() != dim || hidden.hi.len() != dim {
        return Err(DecoderRefusal::Shape("hidden box"));
    }
    if !hidden
        .lo
        .iter()
        .zip(hidden.hi)
        .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
    {
        return Err(DecoderRefusal::NonFinite("hidden endpoints"));
    }
    Ok(())
}

pub(super) fn identity(
    head: AffineHead<'_>,
    meter: &mut Meter<'_>,
) -> Result<ScopeIdentity, DecoderRefusal> {
    let mut hash = Sha256::new();
    hash.update(b"ny-affine-vocabulary-head/v1");
    hash.update((head.vocab_size as u64).to_le_bytes());
    hash.update((head.hidden_size as u64).to_le_bytes());
    for values in [head.weights, head.bias] {
        for block in values.chunks(4096) {
            meter.charge(block.len())?;
            for &value in block {
                if !value.is_finite() {
                    return Err(DecoderRefusal::NonFinite("original head"));
                }
                if matches!(head.arithmetic, HeadArithmetic::Binary32 { .. })
                    && super::execution::binary32_bits(value).is_none()
                {
                    return Err(DecoderRefusal::Shape(
                        "head coefficient is not exact binary32",
                    ));
                }
                hash.update(value.to_bits().to_le_bytes());
            }
        }
    }
    let numerical = match head.arithmetic {
        HeadArithmetic::ExactReal => NumericalSemantics::ExactReal,
        HeadArithmetic::SequentialBinary64 => NumericalSemantics::EnclosedExecution {
            contract: Digest::of_bytes(
                b"head-kernel",
                b"bias-first;ascending;separate-f64-multiply-add;rn-even;gradual-underflow/v1",
            ),
        },
        HeadArithmetic::Binary32 { reduction } => NumericalSemantics::EnclosedExecution {
            contract: Digest::of_bytes(b"head-kernel", reduction.contract_bytes()),
        },
    };
    Ok(ScopeIdentity {
        schema_version: SCOPE_SCHEMA_VERSION,
        model: Digest(hash.finalize().into()),
        lowering: Digest::of_bytes(b"lowering", b"original-dense-affine-head/v1"),
        coordinates: Digest::of_bytes(b"coordinates", &(head.hidden_size as u64).to_le_bytes()),
        assumptions: Digest::of_bytes(
            b"assumption",
            if matches!(head.arithmetic, HeadArithmetic::Binary32 { .. }) {
                super::execution::BINARY32_INPUT_ASSUMPTIONS
            } else {
                b"caller-independently-enclosed-hidden-state;no-logit-processors/v1"
            },
        ),
        numerical,
    })
}
pub(super) fn scope(
    identity: &ScopeIdentity,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
) -> Result<CertificateScope, DecoderRefusal> {
    let mut hash = Sha256::new();
    hash.update(b"strict-complete-vocabulary-argmax/v1");
    hash.update((target as u64).to_le_bytes());
    for &a in anchor {
        hash.update(a.to_bits().to_le_bytes());
    }
    CertificateScope::new(
        identity.clone(),
        BoxRegion::new(hidden.lo.to_vec(), hidden.hi.to_vec())
            .map_err(|_| DecoderRefusal::Shape("scope box"))?,
        Digest(hash.finalize().into()),
    )
    .map_err(|_| DecoderRefusal::Evidence("scope"))
}
pub(super) fn scope_digest(scope: &CertificateScope) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny-vocabulary-scope/v1");
    let id = scope.identity();
    hash.update(id.schema_version.to_le_bytes());
    for value in [
        id.model,
        id.lowering,
        id.coordinates,
        id.assumptions,
        scope.property(),
    ] {
        hash.update(value.0);
    }
    match id.numerical {
        NumericalSemantics::ExactReal => hash.update([0]),
        NumericalSemantics::EnclosedExecution { contract } => {
            hash.update([1]);
            hash.update(contract.0);
        }
    }
    for &v in scope.region().lo().iter().chain(scope.region().hi()) {
        hash.update(v.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

pub(super) fn geometry(
    head: AffineHead<'_>,
    members: &[usize],
    anchor_id: usize,
    meter: &mut Meter<'_>,
) -> Result<Geometry, DecoderRefusal> {
    if members.is_empty() || !members.contains(&anchor_id) {
        return Err(DecoderRefusal::Evidence("group anchor"));
    }
    let mut component = vec![0.0f64; head.hidden_size];
    let mut absolute_weights = if head.arithmetic != HeadArithmetic::ExactReal {
        vec![0.0f64; head.hidden_size]
    } else {
        Vec::new()
    };
    let (mut l1, mut l2, mut linf, mut absolute_bias) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &id in members {
        meter.charge(head.hidden_size)?;
        let (mut row_l1, mut row_squared) = (0.0, 0.0);
        for (k, (&w, &center)) in head.row(id).iter().zip(head.row(anchor_id)).enumerate() {
            // The target cancels exactly: (w_j-w_t)-(w_anchor-w_t)=w_j-w_anchor.
            // Only competitor IDs in this checked group enter any residual radius.
            let residual = abs_upper(difference(w, center)?);
            component[k] = component[k].max(residual);
            row_l1 = up_add(row_l1, residual)?;
            row_squared = up_add(row_squared, up_mul(residual, residual)?)?;
            linf = linf.max(residual);
            if !absolute_weights.is_empty() {
                absolute_weights[k] = absolute_weights[k].max(w.abs());
            }
        }
        l1 = l1.max(row_l1);
        l2 = l2.max(sqrt_up(row_squared)?);
        absolute_bias = absolute_bias.max(head.bias[id].abs());
    }
    Ok(Geometry {
        component,
        l1,
        l2,
        linf,
        absolute_weights,
        absolute_bias,
    })
}

pub(super) fn group_bound(
    head: AffineHead<'_>,
    hidden: HiddenBox<'_>,
    anchor: &[f64],
    target: usize,
    anchor_id: usize,
    at_anchor: f64,
    geometry: &Geometry,
    rad: &[f64],
) -> Result<GroupBounds, DecoderRefusal> {
    let (mut rho1, mut rho2sq, mut rhoinf, mut component) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&r, &m) in rad.iter().zip(&geometry.component) {
        rho1 = up_add(rho1, r)?;
        rho2sq = up_add(rho2sq, up_mul(r, r)?)?;
        rhoinf = rhoinf.max(r);
        component = up_add(component, up_mul(r, m)?)?;
    }
    let residual_l1_linf = up_mul(geometry.l1, rhoinf)?;
    let residual_l2_l2 = up_mul(geometry.l2, sqrt_up(rho2sq)?)?;
    let residual_linf_l1 = up_mul(geometry.linf, rho1)?;
    let residual = component
        .min(residual_l1_linf)
        .min(residual_l2_l2)
        .min(residual_linf_l1);
    let anchor_upper = anchor_support(head, hidden, anchor, target, anchor_id)?;
    let execution_error = if head.arithmetic == HeadArithmetic::ExactReal {
        0.0
    } else {
        let mut absolute = geometry.absolute_bias;
        for ((&w, &lo), &hi) in geometry
            .absolute_weights
            .iter()
            .zip(hidden.lo)
            .zip(hidden.hi)
        {
            absolute = up_add(absolute, up_mul(w, lo.abs().max(hi.abs()))?)?;
        }
        up_add(
            arithmetic_error(head.arithmetic, absolute, head.hidden_size)?,
            execution_error(head, target, hidden)?,
        )?
    };
    let upper = up_add(
        up_add(up_add(at_anchor, anchor_upper)?, residual)?,
        execution_error,
    )?;
    Ok(GroupBounds {
        at_anchor_upper: at_anchor,
        anchor_upper,
        radius_l1: geometry.l1,
        radius_l2: geometry.l2,
        radius_linf: geometry.linf,
        residual_componentwise: component,
        residual_l1_linf,
        residual_l2_l2,
        residual_linf_l1,
        execution_error,
        upper,
    })
}

pub(super) fn default_budget(options: &VocabOptions) -> Result<WorkBudget, DecoderRefusal> {
    if let Some(budget) = &options.budget {
        return Ok(budget.clone());
    }
    WorkBudget::new(
        options.deadline,
        usize::try_from(options.max_scratch_bytes)
            .map_err(|_| DecoderRefusal::Resource("scratch size"))?,
        usize::try_from(options.max_persistent_bytes)
            .map_err(|_| DecoderRefusal::Resource("resident size"))?,
        options.max_terms,
        options.max_proposals,
        Duration::ZERO,
    )
    .map_err(DecoderRefusal::Budget)
}

/// Full preprocessing is charged, including the borrowed original head, all
/// competitor rows, anchor logits and every prepared group radius. No shortlist.
pub fn prepare_vocab_tree<'a>(
    head: AffineHead<'a>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
) -> Result<PreparedVocabTree<'a>, DecoderRefusal> {
    prepare_with_owner(head, anchor, target, options, None)
}

pub(super) fn prepare_with_owner<'a>(
    head: AffineHead<'a>,
    anchor: &[f64],
    target: usize,
    options: &VocabOptions,
    head_owner: Option<SharedAffineHead>,
) -> Result<PreparedVocabTree<'a>, DecoderRefusal> {
    let scoped = head_owner
        .as_ref()
        .map(|owner| owner.scoped_options(options))
        .transpose()?;
    let options = scoped.as_ref().unwrap_or(options);
    let budget = default_budget(options)?;
    let mut meter = Meter::with_budget(options, budget.clone())?;
    validate_head(head, anchor, target, options)?;
    let count = head.vocab_size - 1;
    let leaf_count = count.max(1).div_ceil(options.leaf_size);
    let power = leaf_count
        .checked_next_power_of_two()
        .ok_or(DecoderRefusal::Resource("tree size"))?;
    let node_cap = power
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(DecoderRefusal::Resource("tree size"))?;
    if node_cap > options.max_nodes {
        return Err(DecoderRefusal::Resource("tree node cap"));
    }
    let levels = power.trailing_zeros() as usize + 1;
    let vectors = if head.arithmetic != HeadArithmetic::ExactReal {
        2
    } else {
        1
    };
    let head_bytes = incremental_head_bytes(head, head_owner.as_ref())?;
    let geometry_bytes = checked_mul(checked_mul(node_cap, head.hidden_size)?, 8 * vectors)?;
    let node_metadata = checked_mul(node_cap, size_of::<Node>())?;
    let row_storage = checked_mul(count, 16)?;
    let anchor_storage = checked_mul(head.hidden_size, 8)?;
    let persistent = head_bytes
        .checked_add(geometry_bytes)
        .and_then(|n| n.checked_add(node_metadata))
        .and_then(|n| n.checked_add(row_storage))
        .and_then(|n| n.checked_add(anchor_storage))
        .and_then(|n| n.checked_add(size_of::<PreparedVocabTree<'a>>()))
        .ok_or(DecoderRefusal::Resource("persistent size"))?;
    let scratch = head
        .hidden_size
        .checked_mul(32)
        .and_then(|n| n.checked_add(count.checked_mul(8)?))
        .ok_or(DecoderRefusal::Resource("scratch size"))?;
    if resident_with_owner(persistent, head_owner.as_ref())? as u64 > options.max_persistent_bytes
        || scratch as u64 > options.max_scratch_bytes
    {
        return Err(DecoderRefusal::Resource("preparation memory cap"));
    }
    let work = (head.weights.len() as u64)
        .checked_add(head.bias.len() as u64)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| {
            n.checked_add(
                (count as u64)
                    .checked_mul(head.hidden_size as u64)?
                    .checked_mul((levels + 1) as u64)?,
            )
        })
        .ok_or(DecoderRefusal::Resource("preparation work"))?;
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: persistent,
            work,
            proposals: 1,
        })
        .map_err(DecoderRefusal::Budget)?;
    let identity = identity(head, &mut meter)?;
    if let Some(owner) = &head_owner {
        owner.validate_identity(&identity)?;
    }
    let mut competitors = Vec::with_capacity(count);
    competitors.extend((0..head.vocab_size).filter(|&id| id != target));
    let mut margins = Vec::with_capacity(count);
    for &id in &competitors {
        meter.charge(head.hidden_size)?;
        margins.push(anchor_margin(head, anchor, target, id)?);
    }
    let mut nodes = Vec::with_capacity(node_cap);
    fn build(
        head: AffineHead<'_>,
        competitors: &[usize],
        margins: &[f64],
        start: usize,
        end: usize,
        leaf: usize,
        nodes: &mut Vec<Node>,
        meter: &mut Meter<'_>,
    ) -> Result<usize, DecoderRefusal> {
        let index = nodes.len();
        let members = &competitors[start..end];
        let anchor_id = members[0];
        let geometry = geometry(head, members, anchor_id, meter)?;
        let anchor_margin = margins[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        nodes.push(Node {
            start,
            end,
            anchor_id,
            anchor_margin,
            geometry,
            children: None,
        });
        if end - start > leaf {
            let middle = start + (end - start) / 2;
            let left = build(
                head,
                competitors,
                margins,
                start,
                middle,
                leaf,
                nodes,
                meter,
            )?;
            let right = build(head, competitors, margins, middle, end, leaf, nodes, meter)?;
            nodes[index].children = Some((left, right));
        }
        Ok(index)
    }
    if count > 0 {
        build(
            head,
            &competitors,
            &margins,
            0,
            count,
            options.leaf_size,
            &mut nodes,
            &mut meter,
        )?;
    }
    drop(margins);
    lease.complete().map_err(DecoderRefusal::Budget)?;
    let stats = VocabStats {
        preprocessing_terms: meter.terms(),
        persistent_bytes: resident_with_owner(persistent, head_owner.as_ref())? as u64,
        scratch_bytes: scratch as u64,
        ..VocabStats::default()
    };
    let reservation = lease.commit_persistent().map_err(DecoderRefusal::Budget)?;
    let result = PreparedVocabTree {
        head,
        target,
        anchor: anchor.to_vec(),
        competitors,
        nodes,
        identity,
        stats,
        budget,
        reservation,
        head_owner,
    };
    meter.checkpoint()?;
    Ok(result)
}
