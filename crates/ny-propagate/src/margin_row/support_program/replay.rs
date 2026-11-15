// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Borrowed original topology and actually closed phase leaves for independent
//! replay. These observations never grant numeric or publication authority.
//! The independent checker must reconstruct the original model, objective,
//! region coverage and every leaf bound from the authenticated source files.

use std::time::Instant;

use ny_core::Result;

use super::{
    cli_proof::PreparedRootProof,
    domain::{PreparedDomain, PreparedPhase, PreparedPhaseSide},
    ir::{checkpoint, invalid, refusal},
};
use crate::{
    certify::work::{LeaseClass, WorkBudget, WorkLease, WorkRequest},
    margin_row::net::{TwinNet, TwinOp},
};

const POLL: usize = 256;

/// Allocation census advice derived from an actual proof. Counts alone carry
/// no source, complete-cover or numerical authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedReplayStats {
    pub site_count: usize,
    pub leaf_count: usize,
    pub literal_count: usize,
    pub max_depth: usize,
}

/// One actually retained original phase literal. No public constructor or
/// mutable field can turn an arbitrary tuple into producer ancestry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedReplayLiteral {
    op: usize,
    index: usize,
    positive: bool,
}
impl PreparedReplayLiteral {
    pub fn op(self) -> usize {
        self.op
    }
    pub fn index(self) -> usize {
        self.index
    }
    pub fn positive(self) -> bool {
        self.positive
    }
}

/// Borrowed for the duration of a synchronous visitor. This leaf cannot
/// outlive the proof's original source, conditional evaluations and claims.
pub struct PreparedReplayLeaf<'a> {
    domain: Option<&'a PreparedDomain>,
    classes: &'a [usize],
    head_op: usize,
    empty: bool,
    margin: f64,
    budget: &'a WorkBudget,
    deadline: Instant,
}
impl PreparedReplayLeaf<'_> {
    pub fn classes(&self) -> &[usize] {
        self.classes
    }
    pub fn is_empty(&self) -> bool {
        self.empty
    }
    pub fn literal_count(&self) -> usize {
        self.domain.map_or(0, PreparedDomain::depth)
    }
    /// Diagnostic observation only. Numeric replay must recompute this bound.
    pub fn min_margin(&self) -> f64 {
        self.margin
    }
    /// Conjunctive original literals, newest first. Each bounded scan checks
    /// the original ledger, cancellation token and inherited minimum clock.
    pub fn literals(&self) -> impl Iterator<Item = Result<PreparedReplayLiteral>> + '_ {
        self.domain
            .into_iter()
            .flat_map(|domain| domain.phases())
            .enumerate()
            .map(move |(i, phase)| {
                if i % POLL == 0 {
                    checkpoint(self.budget, Some(self.deadline))?;
                }
                let (op, index, side) = match phase {
                    PreparedPhase::Trunk {
                        op, neuron, side, ..
                    } => (op, neuron, side),
                    PreparedPhase::Head { neuron, side } => (self.head_op, neuron, side),
                };
                Ok(PreparedReplayLiteral {
                    op,
                    index,
                    positive: side == PreparedPhaseSide::Active,
                })
            })
    }
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("replay census overflow"))
}
fn work(n: usize, scale: u64) -> Result<u64> {
    u64::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| invalid("replay work quote overflow"))
}
fn admit(proof: &PreparedRootProof, units: u64, at: Instant) -> Result<(WorkLease, Instant)> {
    let lease = proof
        .budget()
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: units,
                ..WorkRequest::default()
            },
        )
        .map_err(refusal)?;
    let at = lease.deadline().map_or(at, |d| d.min(at));
    checkpoint(proof.budget(), Some(at))?;
    Ok((lease, at))
}

impl PreparedRootProof {
    fn replay_net(&self) -> &TwinNet {
        self.publication().source().prefix().root().model().net()
    }

    /// Census without allocating path or objective copies. The actual phase
    /// chain's owned depth gives each leaf's literal count in constant time.
    pub fn replay_stats(&self, requested: Option<Instant>) -> Result<PreparedReplayStats> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let net = self.replay_net();
        let leaves = self
            .tree_receipt()
            .map_or(Ok(1), |r| add(r.bound_leaves, r.empty_leaves))?;
        let (mut lease, at) = admit(self, work(add(net.ops.len(), leaves)?, 256)?, at)?;
        let mut stats = PreparedReplayStats {
            site_count: 0,
            leaf_count: 0,
            literal_count: 0,
            max_depth: 0,
        };
        for (op, node) in net.ops.iter().enumerate() {
            if op % POLL == 0 {
                checkpoint(self.budget(), Some(at))?;
            }
            if matches!(node, TwinOp::Relu { .. }) {
                stats.site_count = add(stats.site_count, 1)?;
            }
        }
        if let Some(tree) = self.publication().tree() {
            for (i, (evaluation, _)) in tree.leaves().enumerate() {
                if i % POLL == 0 {
                    checkpoint(self.budget(), Some(at))?;
                }
                let depth = evaluation.domain().depth();
                stats.leaf_count = add(stats.leaf_count, 1)?;
                stats.literal_count = add(stats.literal_count, depth)?;
                stats.max_depth = stats.max_depth.max(depth);
            }
        } else {
            stats.leaf_count = 1;
        }
        if stats.leaf_count != leaves || stats.site_count == 0 {
            return Err(invalid("replay census lost genuine closed topology"));
        }
        checkpoint(self.budget(), Some(at))?;
        lease.complete().map_err(refusal)?;
        Ok(stats)
    }

    /// Original compiled ReLU op IDs and input widths in topological order.
    /// The source-file owner binds these to the authored ONNX input names.
    pub fn visit_replay_sites(
        &self,
        requested: Option<Instant>,
        visitor: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let net = self.replay_net();
        let (mut lease, at) = admit(self, work(net.ops.len(), 256)?, at)?;
        for (op, node) in net.ops.iter().enumerate() {
            if op % POLL == 0 {
                checkpoint(self.budget(), Some(at))?;
            }
            if let TwinOp::Relu { input, .. } = node {
                let size = *net
                    .tsize
                    .get(*input)
                    .ok_or_else(|| invalid("replay ReLU input missing"))?;
                visitor(op, size)?;
                checkpoint(self.budget(), Some(at))?;
            }
        }
        checkpoint(self.budget(), Some(at))?;
        lease.complete().map_err(refusal)?;
        Ok(())
    }

    /// Visit the actual retained bound/empty leaves. Export has no numeric
    /// input or constructor and cannot turn Open work into a proof. Callback
    /// allocations and serialization are separately prepaid by the exporter.
    pub fn visit_replay_leaves(
        &self,
        requested: Option<Instant>,
        visitor: &mut dyn FnMut(PreparedReplayLeaf<'_>) -> Result<()>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let stats = self.replay_stats(Some(at))?;
        let net = self.replay_net();
        let head_op = add(net.i_gemm1, 1)?;
        if !matches!(net.ops.get(head_op), Some(TwinOp::Relu { .. })) {
            return Err(invalid("replay missing actual original head ReLU"));
        }
        let scans = add(
            add(stats.literal_count, stats.leaf_count)?,
            stats
                .leaf_count
                .checked_mul(self.classes().len())
                .ok_or_else(|| invalid("replay objective quote overflow"))?,
        )?;
        let (mut lease, at) = admit(self, work(scans, 512)?, at)?;
        if let Some(tree) = self.publication().tree() {
            for (i, (evaluation, path)) in tree.leaves().enumerate() {
                if i % POLL == 0 {
                    checkpoint(self.budget(), Some(at))?;
                }
                let domain = evaluation.domain();
                let _original_path = path.path(Some(at))?;
                if !domain.same_domain(path.domain()) || !domain.same_root(tree.root()) {
                    return Err(invalid("replay leaf lost original domain ancestry"));
                }
                let empty = evaluation.crossing().is_some();
                let mut margin = f64::INFINITY;
                if !empty {
                    for (j, &bound) in evaluation.bounds(Some(at))?.iter().enumerate() {
                        if j % POLL == 0 {
                            checkpoint(self.budget(), Some(at))?;
                        }
                        if !bound.is_finite() || bound <= 0.0 {
                            return Err(invalid("replay leaf is not actually closed"));
                        }
                        margin = margin.min(bound);
                    }
                }
                visitor(PreparedReplayLeaf {
                    domain: Some(domain),
                    classes: self.classes(),
                    head_op,
                    empty,
                    margin,
                    budget: self.budget(),
                    deadline: at,
                })?;
                checkpoint(self.budget(), Some(at))?;
            }
        } else {
            visitor(PreparedReplayLeaf {
                domain: None,
                classes: self.classes(),
                head_op,
                empty: false,
                margin: self.min_margin(),
                budget: self.budget(),
                deadline: at,
            })?;
        }
        checkpoint(self.budget(), Some(at))?;
        lease.complete().map_err(refusal)?;
        Ok(())
    }
}
