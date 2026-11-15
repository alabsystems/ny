// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original phase regions. A phase node owns only one actual original literal;
//! numerical gate overrides are admitted once for a particular evaluation.
//! Root rows are never relabelled as conditional row certificates.

use super::cli_proof::PreparedRootOpen;
use super::ir::{checkpoint, invalid, refusal};
use super::root_adapter::PreparedRootAdapter;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use crate::certify::{CoverLiteral, CoverSite, LeafPath};
use crate::margin_row::bab::{PreparedRootEvaluation, PreparedRootYBox};
use crate::margin_row::engine::{DomainGates, FrozenGateFrame, GateVecs};
use crate::margin_row::root::LayerGates;
use crate::margin_row::rounding::RoundMode;
use ny_core::{NyError, Result};
use std::{mem::size_of, sync::Arc, time::Instant};

const META: usize = 4096;
const POLL: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) enum PreparedPhaseSide {
    Inactive,
    Active,
}
impl PreparedPhaseSide {
    pub(in crate::margin_row) fn sign(self) -> i8 {
        match self {
            Self::Inactive => -1,
            Self::Active => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) enum PreparedPhase {
    Trunk {
        op: usize,
        layer: usize,
        neuron: usize,
        unstable_position: usize,
        side: PreparedPhaseSide,
    },
    Head {
        neuron: usize,
        side: PreparedPhaseSide,
    },
}

struct DomainRoot {
    open: PreparedRootOpen,
    head_box: PreparedRootYBox,
    resident: PersistentReservation,
}
struct PhaseNode {
    parent: Option<Arc<PhaseNode>>,
    phase: PreparedPhase,
    depth: usize,
    resident: PersistentReservation,
}
impl Drop for PhaseNode {
    fn drop(&mut self) {
        // Releasing a long uniquely owned phase chain must not recurse on the
        // thread stack. Shared ancestors stay intact on their real owners.
        let mut parent = self.parent.take();
        while let Some(node) = parent {
            match Arc::try_unwrap(node) {
                Ok(mut owned) => {
                    parent = owned.parent.take();
                    drop(owned);
                }
                Err(_shared) => break,
            }
        }
    }
}

/// The exact root owner and immutable actual-literal chain define the region.
/// A caller's raw split tuples, hash, box or supplied rows cannot mint it.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedDomain {
    root: Arc<DomainRoot>,
    tip: Option<Arc<PhaseNode>>,
    deadline: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) enum PreparedSplitSite {
    Trunk { op: usize, neuron: usize },
    Head { neuron: usize },
}

pub(in crate::margin_row) struct PreparedDomainPath {
    path: LeafPath,
    domain: PreparedDomain,
    deadline: Instant,
    resident: PersistentReservation,
}
impl PreparedDomainPath {
    pub(in crate::margin_row) fn path(&self, requested: Option<Instant>) -> Result<&LeafPath> {
        self.domain.checkpoint(Some(self.domain.deadline(Some(
            requested.map_or(self.deadline, |d| d.min(self.deadline)),
        ))?))?;
        if !self.resident.shares_ledger(self.domain.budget()) {
            return Err(invalid("conditional cover path foreign ledger"));
        }
        Ok(&self.path)
    }
    pub(in crate::margin_row) fn domain(&self) -> &PreparedDomain {
        &self.domain
    }
}

impl PreparedDomain {
    pub(in crate::margin_row) fn from_open(
        open: PreparedRootOpen,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let mut at = open.deadline(requested)?;
        open.checkpoint(Some(at))?;
        if open.ordinary_evaluation().is_none() {
            return Err(NyError::UnsupportedConfiguration(
                "prepared tree needs its actual complete ordinary root".into(),
            ));
        }
        let budget = open.session().budget();
        let bytes = root_bytes();
        let mut lease = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: bytes,
                    work: 4096,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        open.checkpoint(Some(at))?;
        let head_box = open
            .ordinary_evaluation()
            .expect("checked")
            .copy_ybox(Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        let result = Self {
            root: Arc::new(DomainRoot {
                open,
                head_box,
                resident: lease.commit_persistent().map_err(refusal)?,
            }),
            tip: None,
            deadline: at,
        };
        result.checkpoint(Some(at))?;
        Ok(result)
    }
    pub(in crate::margin_row) fn adapter(&self) -> &PreparedRootAdapter {
        self.root_evaluation().adapter()
    }
    pub(in crate::margin_row) fn root_evaluation(&self) -> &PreparedRootEvaluation {
        self.root
            .open
            .ordinary_evaluation()
            .expect("private original domain constructor")
    }
    /// Borrow the actual immutable open owner retained at domain creation.
    /// Callers must checkpoint this domain before sharing it for publication;
    /// a supplied session or digest cannot substitute for this original root.
    pub(in crate::margin_row) fn root_open(&self) -> &PreparedRootOpen {
        &self.root.open
    }
    pub(in crate::margin_row) fn budget(&self) -> &WorkBudget {
        self.root.open.session().budget()
    }
    pub(in crate::margin_row) fn target(&self) -> usize {
        self.root.open.target()
    }
    pub(in crate::margin_row) fn classes(&self) -> &[usize] {
        self.root.open.classes()
    }
    pub(in crate::margin_row) fn phase_count(&self) -> usize {
        self.tip.as_ref().map_or(0, |p| p.depth)
    }
    pub(in crate::margin_row) fn depth(&self) -> usize {
        self.phase_count()
    }
    pub(in crate::margin_row) fn same_root(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.root, &other.root)
    }
    pub(in crate::margin_row) fn same_domain(&self, other: &Self) -> bool {
        self.same_root(other)
            && match (&self.tip, &other.tip) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.root.open.deadline(Some(
            requested.map_or(self.deadline, |d| d.min(self.deadline)),
        ))
    }
    /// Full immutable source validation occurs at phase/publication boundaries.
    /// Arithmetic callbacks use cheap_checkpoint instead of repeating it.
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.root.open.checkpoint(Some(at))?;
        if !self.root.resident.shares_ledger(self.budget())
            || self.root.resident.bytes() != root_bytes()
        {
            return Err(invalid("conditional domain missing original root owner"));
        }
        if self.tip.as_ref().is_some_and(|tip| {
            !tip.resident.shares_ledger(self.budget())
                || tip.resident.bytes() != phase_bytes()
                || tip.depth == 0
        }) {
            return Err(invalid(
                "conditional phase chain lost original header claim",
            ));
        }
        self.cheap_checkpoint(at)
    }
    pub(in crate::margin_row) fn cheap_checkpoint(&self, at: Instant) -> Result<()> {
        checkpoint(self.budget(), Some(at.min(self.deadline)))
    }
    pub(in crate::margin_row) fn phases(&self) -> impl Iterator<Item = PreparedPhase> + '_ {
        std::iter::successors(self.tip.as_deref(), |node| node.parent.as_deref())
            .map(|node| node.phase)
    }
    fn phase_fixed_polled(&self, site: PreparedSplitSite, at: Instant) -> Result<bool> {
        for (i, p) in self.phases().enumerate() {
            poll(self, at, i)?;
            let fixed = match (site, p) {
                (PreparedSplitSite::Head { neuron: a }, PreparedPhase::Head { neuron: b, .. }) => {
                    a == b
                }
                (
                    PreparedSplitSite::Trunk { op: a, neuron: x },
                    PreparedPhase::Trunk {
                        op: b, neuron: y, ..
                    },
                ) => a == b && x == y,
                _ => false,
            };
            if fixed {
                return Ok(true);
            }
        }
        Ok(false)
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(in crate::margin_row) fn phase_fixed(
        &self,
        site: PreparedSplitSite,
        requested: Option<Instant>,
    ) -> Result<bool> {
        let mut at = self.deadline(requested)?;
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: work(add(self.depth(), 1)?, 64)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        let fixed = self.phase_fixed_polled(site, at)?;
        self.cheap_checkpoint(at)?;
        lease.complete().map_err(refusal)?;
        Ok(fixed)
    }
    pub(in crate::margin_row) fn next_split(
        &self,
        requested: Option<Instant>,
    ) -> Result<Option<PreparedSplitSite>> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let net = self.adapter().net();
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: work(
                        add(
                            mul(net.n_y, add(self.classes().len(), self.depth())?)?,
                            mul(
                                add(
                                    self.adapter().receipt().final_unstable_rows,
                                    net.trunk_relus.len(),
                                )?,
                                add(self.depth(), 1)?,
                            )?,
                        )?,
                        128,
                    )?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        let ybox = self.root.head_box.view(Some(at))?;
        let (weights, _, (_, ny)) = net.gemm2();
        let available = |i| -> Result<bool> {
            Ok(ybox.ly[i] < 0.0
                && ybox.uy[i] > 0.0
                && !self.phase_fixed_polled(PreparedSplitSite::Head { neuron: i }, at)?)
        };
        let mut selected = None;
        for i in 0..ny {
            poll(self, at, i)?;
            if !available(i)? {
                continue;
            }
            for (r, &class) in self.classes().iter().enumerate() {
                poll(self, at, r)?;
                if weights[self.target() * ny + i] - weights[class * ny + i] < 0.0 {
                    selected = Some(PreparedSplitSite::Head { neuron: i });
                    break;
                }
            }
            if selected.is_some() {
                break;
            }
        }
        if selected.is_none() {
            for i in 0..ny {
                poll(self, at, i)?;
                if available(i)? {
                    selected = Some(PreparedSplitSite::Head { neuron: i });
                    break;
                }
            }
        }
        if selected.is_none() {
            for (li, gate) in self.adapter().gates().layers.iter().enumerate() {
                poll(self, at, li)?;
                for (i, &neuron) in gate.unst.iter().enumerate() {
                    poll(self, at, i)?;
                    let site = PreparedSplitSite::Trunk {
                        op: gate.op,
                        neuron,
                    };
                    if !self.phase_fixed_polled(site, at)? {
                        selected = Some(site);
                        break;
                    }
                }
                if selected.is_some() {
                    break;
                }
            }
        }
        self.cheap_checkpoint(at)?;
        lease.complete().map_err(refusal)?;
        Ok(selected)
    }
    pub(in crate::margin_row) fn split(
        &self,
        site: PreparedSplitSite,
        side: PreparedPhaseSide,
        requested: Option<Instant>,
    ) -> Result<Self> {
        match site {
            PreparedSplitSite::Trunk { op, neuron } => {
                self.split_trunk(op, neuron, side, requested)
            }
            PreparedSplitSite::Head { neuron } => self.split_head(neuron, side, requested),
        }
    }
    pub(in crate::margin_row) fn cover_path(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedDomainPath> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let n = self.depth();
        let persistent = add(
            size_of::<PreparedDomainPath>() + META,
            mul(n, size_of::<CoverLiteral>())?,
        )?;
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: persistent,
                    scratch_bytes: mul(n, size_of::<CoverLiteral>())?,
                    work: work(mul(add(n, 1)?, add(n, 1)?)?, 64)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        let mut literals = exact_vec(n)?;
        for (i, phase) in self.phases().enumerate() {
            poll(self, at, i)?;
            let (site, side) = match phase {
                PreparedPhase::Head { neuron, side } => (
                    CoverSite::Head {
                        index: u32::try_from(neuron).map_err(|_| {
                            invalid("original head cover ID exceeds gate representation")
                        })?,
                    },
                    side,
                ),
                PreparedPhase::Trunk {
                    layer,
                    unstable_position,
                    side,
                    ..
                } => (
                    CoverSite::Trunk {
                        layer: u32::try_from(layer).map_err(|_| {
                            invalid("original trunk cover layer exceeds gate representation")
                        })?,
                        pos: u32::try_from(unstable_position).map_err(|_| {
                            invalid("original trunk cover position exceeds gate representation")
                        })?,
                    },
                    side,
                ),
            };
            literals.push(CoverLiteral {
                site,
                positive: side.sign() > 0,
            });
        }
        // Insertion sort has bounded cooperative scalar moves, and feeds the
        // existing canonical constructor an already sorted, unique list.
        for i in 1..n {
            let item = literals[i];
            let mut j = i;
            while j > 0 && literals[j - 1] > item {
                poll(self, at, j)?;
                literals[j] = literals[j - 1];
                j -= 1;
            }
            literals[j] = item;
            poll(self, at, i)?;
        }
        let path = LeafPath::from_literals(literals)
            .ok_or_else(|| invalid("actual original cover path contradiction"))?;
        self.cheap_checkpoint(at)?;
        lease.mark_materialized().map_err(refusal)?;
        let result = PreparedDomainPath {
            path,
            domain: self.clone(),
            deadline: at,
            resident: lease.commit_persistent().map_err(refusal)?,
        };
        result.path(Some(at))?;
        Ok(result)
    }
    pub(in crate::margin_row) fn split_trunk(
        &self,
        op: usize,
        neuron: usize,
        side: PreparedPhaseSide,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let net = self.adapter().net();
        let layer = net
            .trunk_relus
            .binary_search(&op)
            .map_err(|_| invalid("phase does not name original trunk ReLU"))?;
        let gate = &self.adapter().gates().layers[layer];
        let pos = gate
            .unst
            .binary_search(&neuron)
            .map_err(|_| invalid("phase does not name an original unstable neuron"))?;
        if neuron >= gate.n || !(gate.l[neuron] < 0.0 && gate.u[neuron] > 0.0) {
            return Err(invalid("original trunk phase geometry changed"));
        }
        self.push(
            PreparedPhase::Trunk {
                op,
                layer,
                neuron,
                unstable_position: pos,
                side,
            },
            requested,
        )
    }
    pub(in crate::margin_row) fn split_head(
        &self,
        neuron: usize,
        side: PreparedPhaseSide,
        requested: Option<Instant>,
    ) -> Result<Self> {
        if neuron >= self.adapter().net().n_y {
            return Err(invalid("head phase outside original Gemm1"));
        }
        self.push(PreparedPhase::Head { neuron, side }, requested)
    }
    fn push(&self, phase: PreparedPhase, requested: Option<Instant>) -> Result<Self> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let depth = add(self.phase_count(), 1)?;
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: phase_bytes(),
                    work: work(depth, 128)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        for (i, existing) in self.phases().enumerate() {
            poll(self, at, i)?;
            let duplicate = match (phase, existing) {
                (
                    PreparedPhase::Trunk {
                        op: a, neuron: x, ..
                    },
                    PreparedPhase::Trunk {
                        op: b, neuron: y, ..
                    },
                ) => a == b && x == y,
                (PreparedPhase::Head { neuron: a, .. }, PreparedPhase::Head { neuron: b, .. }) => {
                    a == b
                }
                _ => false,
            };
            if duplicate {
                return Err(invalid("repeated original phase literal"));
            }
        }
        self.cheap_checkpoint(at)?;
        lease.mark_materialized().map_err(refusal)?;
        let result = Self {
            root: self.root.clone(),
            tip: Some(Arc::new(PhaseNode {
                parent: self.tip.clone(),
                phase,
                depth,
                resident: lease.commit_persistent().map_err(refusal)?,
            })),
            deadline: at,
        };
        result.checkpoint(Some(at))?;
        Ok(result)
    }
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prepared collector integration")
    )]
    pub(in crate::margin_row) fn matches_legacy_splits(
        &self,
        trunk: &[(usize, usize, i8)],
        heads: &[(usize, i8)],
    ) -> Result<bool> {
        let at = self.deadline(None)?;
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: work(add(add(self.phase_count(), trunk.len())?, heads.len())?, 64)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        let mut ti = trunk.len();
        let mut hi = heads.len();
        for (i, phase) in self.phases().enumerate() {
            poll(self, at, i)?;
            match phase {
                PreparedPhase::Trunk {
                    layer,
                    unstable_position,
                    side,
                    ..
                } => {
                    if ti == 0 || trunk[ti - 1] != (layer, unstable_position, side.sign()) {
                        return Ok(false);
                    }
                    ti -= 1;
                }
                PreparedPhase::Head { neuron, side } => {
                    if hi == 0 || heads[hi - 1] != (neuron, side.sign()) {
                        return Ok(false);
                    }
                    hi -= 1;
                }
            }
        }
        self.cheap_checkpoint(at)?;
        lease.complete().map_err(refusal)?;
        Ok(ti == 0 && hi == 0)
    }
    pub(in crate::margin_row) fn frame(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedDomainFrame> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        let count = self.adapter().gates().layers.len();
        let mut header = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: work(mul(add(count, 1)?, add(self.phase_count(), 1)?)?, 128)?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = header.deadline().map_or(at, |d| d.min(at));
        let mut bytes = add(size_of::<PreparedDomainFrame>(), META)?;
        for (li, gate) in self.adapter().gates().layers.iter().enumerate() {
            poll(self, at, li)?;
            let mut changed = false;
            for (i, p) in self.phases().enumerate() {
                poll(self, at, i)?;
                if matches!(p,PreparedPhase::Trunk{layer,..} if layer==li) {
                    changed = true;
                    break;
                }
            }
            if changed {
                // BTreeMap has no public capacity accessor. One whole node
                // per key (4096 bytes) conservatively pays internal/leaf node
                // storage, separately from all four exact Vec capacities.
                bytes = add(bytes, add(mul(gate.n, 32)?, 4096)?)?;
            }
        }
        let mut lease = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: bytes,
                    work: work(
                        add(bytes / 8, mul(add(count, 1)?, add(self.phase_count(), 1)?)?)?,
                        64,
                    )?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        at = lease.deadline().map_or(at, |d| d.min(at));
        let mut overrides = DomainGates::default();
        for (i, phase) in self.phases().enumerate() {
            poll(self, at, i)?;
            let PreparedPhase::Trunk {
                layer,
                neuron,
                side,
                ..
            } = phase
            else {
                continue;
            };
            if let std::collections::btree_map::Entry::Vacant(entry) = overrides.layers.entry(layer)
            {
                let gate = &self.adapter().gates().layers[layer];
                let copy = |v: &[f64]| -> Result<Vec<f64>> {
                    let mut out = exact_vec(v.len())?;
                    for (j, &x) in v.iter().enumerate() {
                        poll(self, at, j)?;
                        out.push(x);
                    }
                    Ok(out)
                };
                let value = GateVecs {
                    alpha: copy(&gate.alpha)?,
                    s: copy(&gate.s)?,
                    c: copy(&gate.c)?,
                    ms: copy(&gate.ms)?,
                };
                entry.insert(value);
            }
            let gate = overrides
                .layers
                .get_mut(&layer)
                .expect("inserted actual layer");
            let slope = if side == PreparedPhaseSide::Active {
                1.0
            } else {
                0.0
            };
            gate.alpha[neuron] = slope;
            gate.s[neuron] = slope;
            gate.c[neuron] = 0.0;
            gate.ms[neuron] = slope;
        }
        self.checkpoint(Some(at))?;
        header.complete().map_err(refusal)?;
        lease.mark_materialized().map_err(refusal)?;
        let mut retained_domain = self.clone();
        retained_domain.deadline = retained_domain.deadline.min(at);
        let result = PreparedDomainFrame {
            domain: retained_domain,
            overrides,
            deadline: at,
            resident: lease.commit_persistent().map_err(refusal)?,
        };
        result.checkpoint(Some(at))?;
        Ok(result)
    }
}

/// Actual original gate reader plus one paid set of conditional overrides.
/// layer() is the unchanged original baseline: overrides are applied ONCE by
/// the common backward kernel. This frame cannot grant root-row authority.
pub(in crate::margin_row) struct PreparedDomainFrame {
    domain: PreparedDomain,
    overrides: DomainGates,
    deadline: Instant,
    resident: PersistentReservation,
}
impl PreparedDomainFrame {
    pub(in crate::margin_row) fn domain(&self) -> &PreparedDomain {
        &self.domain
    }
    pub(in crate::margin_row) fn overrides(&self) -> &DomainGates {
        &self.overrides
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.domain.deadline(Some(
            requested.map_or(self.deadline, |d| d.min(self.deadline)),
        ))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.domain.checkpoint(Some(at))?;
        if !self.resident.shares_ledger(self.domain.budget()) {
            return Err(invalid("conditional gate frame foreign ledger"));
        }
        Ok(())
    }
    #[expect(dead_code, reason = "prepared collector integration")]
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.resident.bytes()
    }
}
impl FrozenGateFrame for PreparedDomainFrame {
    fn mode(&self) -> RoundMode {
        RoundMode::Outward
    }
    fn layer_count(&self) -> usize {
        self.domain.adapter().gates().layers.len()
    }
    fn layer(&self, i: usize) -> Result<&LayerGates> {
        self.domain
            .adapter()
            .gates()
            .layers
            .get(i)
            .ok_or_else(|| invalid("conditional reader missing original layer"))
    }
    fn lo(&self) -> &[f64] {
        &self.domain.adapter().gates().lo
    }
    fn hi(&self) -> &[f64] {
        &self.domain.adapter().gates().hi
    }
    fn mid(&self) -> &[f64] {
        &self.domain.adapter().gates().mid
    }
    fn rad(&self) -> &[f64] {
        &self.domain.adapter().gates().rad
    }
    fn xabs(&self) -> &[f64] {
        &self.domain.adapter().gates().xabs
    }
    fn checkpoint(&self) -> Result<()> {
        self.domain.cheap_checkpoint(self.deadline)
    }
    fn cooperative(&self) -> bool {
        true
    }
}

fn root_bytes() -> usize {
    size_of::<DomainRoot>() + 2 * size_of::<usize>() + META
}
fn phase_bytes() -> usize {
    size_of::<PhaseNode>() + 2 * size_of::<usize>() + META
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| invalid("conditional frame allocation refused"))?;
    if v.capacity() != n {
        return Err(invalid("conditional frame unexpected allocator capacity"));
    }
    Ok(v)
}
fn poll(domain: &PreparedDomain, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        domain.cheap_checkpoint(at)?;
    }
    Ok(())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("conditional owner byte overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("conditional owner byte overflow"))
}
fn work(n: usize, k: usize) -> Result<u64> {
    u64::try_from(mul(n, k)?).map_err(|_| invalid("conditional work overflow"))
}

#[cfg(test)]
#[path = "domain_tests.rs"]
mod tests;
