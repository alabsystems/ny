// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original prepared-root evidence through the existing class authorizer.
//!
//! The completed construction and terminal are retained together. Only their
//! sealed, strictly positive original margins become root class covers; the
//! original `authorize_unsat` remains the sole token producer. G2 evaluates
//! the stored original operators independently with paid, cooperative scalar
//! reference arithmetic and the existing `ContainmentAudit` comparisons.
//! Sampling is downgrade-only, never a numerical proof or proof-bundle replay.
//! CLI property agreement and retention through the final file publication
//! remain mandatory. This owner deliberately has no consuming token escape.

use super::construction::PreparedRootConstruction;
use super::ir::{checkpoint, invalid, refusal};
use super::terminal::{PreparedTerminal, PreparedTerminalReceipt};
use crate::certify::margin_row as cert_mr;
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};
use crate::certify::{
    aligned_single_class_covers, authorize_unsat, ratify, ClassCover, GuardReport, TokenScope,
    UnsatToken,
};
use crate::guards::containment::{
    CertifiedFrame, ContainmentAudit, ContainmentOutcome, MAX_GUARD_BYTES,
};
use crate::margin_row::bab::{
    PreparedRootEvaluation, PreparedRootEvaluationReceipt, PreparedTreeCertificate,
    PreparedTreeReceipt,
};
use crate::margin_row::net::TwinOp;
use crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
use crate::margin_row::RootPrecisionRuntimePolicy;
use ndarray::Array2;
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const POLL: usize = 256;
const META: usize = 4096;
const MAX_LAYER_SAMPLES: usize = 4096;
const POINT_MACS_PER_SEC: f64 = 5e8;
const GUARD_FRACTION: f64 = 0.02;

struct SharedSourceInner {
    construction: PreparedRootConstruction,
    resident: PersistentReservation,
    deadline: Instant,
}

/// The independently paid sharing header lives as long as ANY proof,
/// clipping or retained-row consumer. Sharing the construction cannot drop
/// its original arrays or turn a publisher's released claim into free memory.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRootSource(Arc<SharedSourceInner>);

impl PreparedRootSource {
    pub(in crate::margin_row) fn construction(&self) -> &PreparedRootConstruction {
        &self.0.construction
    }
    pub(in crate::margin_row) fn budget(&self) -> &WorkBudget {
        self.construction().prefix().root().budget()
    }
    /// This header claim only; original model/root/prefix/product claims stay
    /// on their existing owners and the same shared ledger.
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.construction().deadline(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        qualify(self.budget(), at)?;
        self.construction().reference_checkpoint(Some(at))?;
        if !self.0.resident.shares_ledger(self.budget())
            || self.resident_bytes() != shared_source_bytes()
        {
            return Err(invalid(
                "prepared shared source missing original header reservation",
            ));
        }
        Ok(())
    }
}

pub(in crate::margin_row) fn share_prepared_root(
    construction: PreparedRootConstruction,
    requested: Option<Instant>,
) -> Result<PreparedRootSource> {
    let budget = construction.prefix().root().budget().clone();
    let mut at = construction.deadline(requested)?;
    let bytes = shared_source_bytes();
    let mut lease = admit(
        &budget,
        WorkRequest {
            persistent_bytes: bytes,
            work: work(1, 128)?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &lease);
    qualify(&budget, at)?;
    lease.mark_materialized().map_err(refusal)?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    let owner = PreparedRootSource(Arc::new(SharedSourceInner {
        construction,
        resident,
        deadline: at,
    }));
    owner.checkpoint(Some(at))?;
    Ok(owner)
}
fn shared_source_bytes() -> usize {
    size_of::<SharedSourceInner>() + 2 * size_of::<usize>() + META
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) struct PreparedPublicationReceipt {
    pub classes: usize,
    pub target: usize,
    pub containment_points: usize,
    pub containment_checks: u64,
    pub containment_blind: u64,
    pub containment_skipped: bool,
    pub policy: RootPrecisionRuntimePolicy,
    pub reference: super::PreparedReferenceReceipt,
    pub terminal: PreparedTerminalReceipt,
    pub ordinary: Option<PreparedRootEvaluationReceipt>,
    pub tree: Option<PreparedTreeReceipt>,
}

struct PublicationInner {
    frame: PreparedCertifiedFrame,
    token: UnsatToken,
    deadline: Instant,
    receipt: PreparedPublicationReceipt,
    resident: PersistentReservation,
}

/// One immutable original proof owner and one last-owner reservation. A token
/// reference is private to the lane; public ingress must retain this owner
/// through property ratification and every final writer checkpoint.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedPublication(Arc<PublicationInner>, Instant);

impl PreparedPublication {
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootConstruction {
        self.0.frame.source.construction()
    }
    pub(in crate::margin_row) fn source_owner(&self) -> &PreparedRootSource {
        &self.0.frame.source
    }
    pub(in crate::margin_row) fn terminal(&self) -> &PreparedTerminal {
        &self.0.frame.terminal
    }
    pub(in crate::margin_row) fn ordinary(&self) -> Option<&PreparedRootEvaluation> {
        self.0.frame.ordinary.as_ref()
    }
    pub(super) fn tree(&self) -> Option<&PreparedTreeCertificate> {
        self.0.frame.tree.as_ref()
    }
    pub(in crate::margin_row) fn token(&self) -> &UnsatToken {
        &self.0.token
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedPublicationReceipt {
        self.0.receipt
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        let at = self.1.min(self.0.deadline);
        self.0
            .frame
            .deadline(Some(requested.map_or(at, |d| d.min(at))))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.0.frame.validate(at)?;
        if !self
            .0
            .resident
            .shares_ledger(self.source().prefix().root().budget())
            || self.receipt().reference != self.source().reference_receipt()
            || !self.token().publishable()
            || self.token().classes().len() != self.terminal().adversaries().len()
        {
            return Err(invalid(
                "prepared publication lost original owner/class authority",
            ));
        }
        replay_classes(&self.0.frame, self.token(), at)?;
        qualify(self.source().prefix().root().budget(), at)
    }
    pub(in crate::margin_row) fn validate_policy(
        &self,
        expected: RootPrecisionRuntimePolicy,
        requested: Option<Instant>,
    ) -> Result<()> {
        self.checkpoint(requested)?;
        if expected != self.receipt().policy {
            return Err(invalid(
                "prepared publication cannot fulfill a different frozen precision policy",
            ));
        }
        Ok(())
    }
    pub(in crate::margin_row) fn validate_objective(
        &self,
        target: usize,
        ordered_classes: &[usize],
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        self.terminal()
            .validate_source(self.source(), target, ordered_classes, Some(at))
    }
    /// G3 and any later guards can only refuse the already authorized proof.
    /// No token is returned, and the exact owner remains live on success.
    pub(in crate::margin_row) fn ratified(
        mut self,
        guards: &GuardReport,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let at = self.deadline(requested)?;
        self.1 = self.1.min(at);
        self.checkpoint(Some(at))?;
        let budget = self.source().prefix().root().budget();
        let n = self.token().classes().len();
        let mut lease = admit(
            budget,
            WorkRequest {
                scratch_bytes: add(mul(n, size_of::<usize>())?, META)?,
                work: work(n, 32)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        // Ratification examines only downgrade guards. Its temporary token
        // copy is fully paid and is never handed to a publisher.
        ratify(self.token().clone(), guards)
            .map_err(|_| invalid("prepared publication downgrade guard refused"))?;
        self.checkpoint(Some(at))?;
        lease.complete().map_err(refusal)?;
        Ok(self)
    }
}

/// A frame is built only from the checked completed producer, never supplied
/// boxes, seed rows, a RootGates facsimile or a numeric bound chosen by a caller.
struct PreparedCertifiedFrame {
    source: PreparedRootSource,
    terminal: PreparedTerminal,
    ordinary: Option<PreparedRootEvaluation>,
    tree: Option<PreparedTreeCertificate>,
    tree_lower: Option<f64>,
    plan: PointPlan,
    deadline: Instant,
    policy: RootPrecisionRuntimePolicy,
    expected_target: usize,
}

impl PreparedCertifiedFrame {
    fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        let at = requested.map_or(self.deadline, |d| d.min(self.deadline));
        let at = self
            .source
            .deadline(Some(at))?
            .min(self.terminal.deadline(Some(at))?);
        let at = match &self.ordinary {
            Some(evaluation) => at.min(evaluation.deadline(Some(at))?),
            None => at,
        };
        Ok(self
            .tree
            .as_ref()
            .map_or(at, |tree| at.min(tree.deadline())))
    }
    fn validate(&self, at: Instant) -> Result<()> {
        self.source.checkpoint(Some(at))?;
        qualify(self.source.budget(), at)?;
        self.terminal.validate_source(
            self.source.construction(),
            self.terminal.target(),
            self.terminal.adversaries(),
            Some(at),
        )?;
        if let Some(tree) = &self.tree {
            tree.checkpoint(Some(at))?;
            tree.root()
                .adapter()
                .validate_source(&self.source, Some(at))?;
            let cover = tree.cover(Some(at))?;
            if !tree.source().same_source(&self.source)
                || !tree.budget().shares_ledger(self.source.budget())
                || tree.target() != self.terminal.target()
                || tree.classes().len() != self.terminal.adversaries().len()
                || tree.receipt().bound_leaves == 0
                || !cover.min_margin().is_finite()
                || cover.min_margin() <= 0.0
                || self.tree_lower.map(f64::to_bits) != Some(cover.min_margin().to_bits())
            {
                return Err(invalid("prepared publication foreign or margin-free tree"));
            }
            let budget = self.source.budget();
            let mut lease = admit(
                budget,
                WorkRequest {
                    work: work(tree.classes().len(), 32)?,
                    ..WorkRequest::default()
                },
                at,
            )?;
            let at = min_lease(at, &lease);
            for (i, (&actual, &expected)) in tree
                .classes()
                .iter()
                .zip(self.terminal.adversaries())
                .enumerate()
            {
                poll(budget, at, i)?;
                if actual != expected {
                    return Err(invalid("prepared publication tree class order changed"));
                }
            }
            qualify(budget, at)?;
            lease.complete().map_err(refusal)?;
        } else if self.tree_lower.is_some() {
            return Err(invalid("prepared publication detached tree margin"));
        }
        if let Some(evaluation) = &self.ordinary {
            evaluation
                .adapter()
                .validate_source(&self.source, Some(at))?;
            evaluation.checkpoint(Some(at))?;
            if evaluation.target() != self.terminal.target()
                || evaluation.adversaries().len() != self.terminal.adversaries().len()
                || evaluation.bounds(Some(at))?.len() != self.terminal.lower().len()
            {
                return Err(invalid("prepared publication foreign ordinary objective"));
            }
            let budget = self.source.budget();
            let mut lease = admit(
                budget,
                WorkRequest {
                    work: work(evaluation.adversaries().len(), 32)?,
                    ..WorkRequest::default()
                },
                at,
            )?;
            let at = min_lease(at, &lease);
            for (i, (&actual, &expected)) in evaluation
                .adversaries()
                .iter()
                .zip(self.terminal.adversaries())
                .enumerate()
            {
                poll(budget, at, i)?;
                if actual != expected {
                    return Err(invalid("prepared publication ordinary class order changed"));
                }
            }
            qualify(budget, at)?;
            lease.complete().map_err(refusal)?;
        }
        let root = self.source.construction().prefix().root();
        if root.policy() != self.policy || self.policy.root_f32 || self.policy.tableau_f32 {
            // The current authentic root does not yet deliver these distinct
            // products. Row f32 backing/reference computation cannot fulfill
            // a root/tableau precision request here.
            return Err(invalid(
                "prepared publication missing requested root/tableau precision delivery",
            ));
        }
        self.source.construction().validate_reference(Some(at))?;
        let r = self.terminal.receipt();
        let net = root.model().net();
        if r.head_lower_columns != net.n_y
            || r.head_upper_columns != net.n_y
            || r.margin_columns != self.terminal.adversaries().len()
            || self.source.construction().prefix().layer_count() != net.trunk_relus.len()
            || self.terminal.target() != self.expected_target
            || self.terminal.adversaries().len() != net.n_out.saturating_sub(1)
        {
            return Err(invalid(
                "prepared publication incomplete original producer coverage",
            ));
        }
        qualify(root.budget(), at)
    }

    // Borrow the actual sealed evaluation once for a paid scan. No supplied
    // scalar, mutable RootEval, or copied vector becomes proof evidence.
    fn ordinary_bounds(&self, at: Instant) -> Result<Option<&[f64]>> {
        self.ordinary
            .as_ref()
            .map(|evaluation| evaluation.bounds(Some(at)))
            .transpose()
    }

    fn lower(&self, column: usize, ordinary: Option<&[f64]>) -> Result<f64> {
        // This immutable value was derived only from the genuine complete
        // partition. Validation retains and checks that exact producer; the
        // scalar cannot be supplied by a caller or detach its leaf owners.
        if let Some(lower) = self.tree_lower {
            return Ok(lower);
        }
        let terminal = self.terminal.lower()[column];
        if !terminal.is_finite() {
            return Err(invalid("prepared publication nonfinite terminal margin"));
        }
        match ordinary {
            Some(bounds) => {
                let value = bounds[column];
                if !value.is_finite() {
                    return Err(invalid("prepared publication nonfinite ordinary margin"));
                }
                Ok(terminal.max(value))
            }
            None => Ok(terminal),
        }
    }

    fn audit_checked(
        &self,
        points: &Array2<f64>,
        audit: &mut ContainmentAudit,
        requested: Option<Instant>,
    ) -> Result<()> {
        let at = self.deadline(requested)?;
        self.validate(at)?;
        let root = self.source.construction().prefix().root();
        let budget = root.budget();
        let net = root.model().net();
        if points.nrows() != net.n_in || points.ncols() == 0 {
            return Err(invalid("prepared containment point shape mismatch"));
        }
        let count = points.ncols();
        let mut lease = admit(
            budget,
            WorkRequest {
                scratch_bytes: add(self.plan.state_bytes, self.plan.frame_bytes)?,
                work: self
                    .plan
                    .point_work
                    .checked_mul(count as u64)
                    .ok_or_else(overflow)?,
                ..WorkRequest::default()
            },
            at,
        )?;
        let at = min_lease(at, &lease);
        let frame = self.source.construction().prefix().frame(Some(at))?;
        let ordinary = self.ordinary_bounds(at)?;
        lease.mark_materialized().map_err(refusal)?;
        for sample in 0..count {
            qualify(budget, at)?;
            let mut values: Vec<Option<Vec<f64>>> = filled(net.ops.len() + 1, None, budget, at)?;
            let mut input = exact_vec(net.n_in, budget, at)?;
            for i in 0..net.n_in {
                poll(budget, at, i)?;
                let x = points[[i, sample]];
                if !x.is_finite() || x < root.lo()[i] || x > root.hi()[i] {
                    return Err(invalid(
                        "prepared containment point outside exact original domain",
                    ));
                }
                input.push(x);
            }
            values[0] = Some(input);
            for (op, original) in net.ops.iter().enumerate().take(net.i_gemm1 + 1) {
                qualify(budget, at)?;
                if let TwinOp::Relu { input, layer } = original {
                    let gate = frame
                        .gate(*layer)
                        .ok_or_else(|| invalid("missing original containment gate"))?;
                    let pre = values[*input]
                        .as_ref()
                        .ok_or_else(|| invalid("missing original containment preactivation"))?;
                    if gate.op != op || pre.len() != gate.n {
                        return Err(invalid("original containment gate mapping changed"));
                    }
                    let stride = gate.n.div_ceil(MAX_LAYER_SAMPLES).max(1);
                    for i in (0..gate.n).step_by(stride) {
                        poll(budget, at, i / stride)?;
                        audit.check_box("trunk_pre", i, sample, gate.l[i], gate.u[i], pre[i]);
                    }
                }
                let mut out = exact_vec(net.tsize[op + 1], budget, at)?;
                match original {
                    TwinOp::Conv(c) => {
                        let src = value(&values, c.input)?;
                        let spatial = mul(c.oshape.1, c.oshape.2)?;
                        for j in 0..net.tsize[op + 1] {
                            poll(budget, at, j)?;
                            let mut sum = 0.0;
                            for tap in 0..c.k_fwd {
                                poll(budget, at, tap)?;
                                let index = c.gather[tap * spatial + j % spatial];
                                let w = c.wmat[(j / spatial) * c.k_fwd + tap];
                                if index != usize::MAX && w != 0.0 {
                                    sum += std::hint::black_box(w * src[index]);
                                }
                            }
                            out.push(sum + c.bias[j / spatial]);
                        }
                    }
                    TwinOp::ChannelAffine {
                        input,
                        scale,
                        shift,
                        ..
                    } => {
                        let src = value(&values, *input)?;
                        for i in 0..src.len() {
                            poll(budget, at, i)?;
                            out.push(std::hint::black_box(scale[i] * src[i]) + shift[i]);
                        }
                    }
                    TwinOp::Add { lhs, rhs } => {
                        let a = value(&values, *lhs)?;
                        let b = value(&values, *rhs)?;
                        for i in 0..a.len() {
                            poll(budget, at, i)?;
                            out.push(a[i] + b[i]);
                        }
                    }
                    TwinOp::Flatten { input } | TwinOp::Relu { input, .. } => {
                        let src = value(&values, *input)?;
                        let relu = matches!(original, TwinOp::Relu { .. });
                        for (i, &x) in src.iter().enumerate() {
                            poll(budget, at, i)?;
                            out.push(if relu { x.max(0.0) } else { x });
                        }
                    }
                    TwinOp::Gemm {
                        input,
                        weight,
                        bias,
                        shape,
                        ..
                    } => {
                        let src = value(&values, *input)?;
                        for o in 0..shape.0 {
                            poll(budget, at, o)?;
                            let mut sum = bias[o];
                            for i in 0..shape.1 {
                                poll(budget, at, i)?;
                                let w = weight[o * shape.1 + i];
                                if w != 0.0 {
                                    sum += std::hint::black_box(w * src[i]);
                                }
                            }
                            out.push(sum);
                        }
                    }
                }
                if out.len() != net.tsize[op + 1] {
                    return Err(invalid("original containment interpreter width changed"));
                }
                if op == net.i_gemm1 {
                    for (i, &y) in out.iter().enumerate() {
                        poll(budget, at, i)?;
                        audit.check_box(
                            "head_y",
                            i,
                            sample,
                            self.terminal.head_lo()[i],
                            self.terminal.head_hi()[i],
                            y,
                        );
                    }
                    let (weights, bias, (outputs, heads)) = net.gemm2();
                    let mut scores = exact_vec(outputs, budget, at)?;
                    let mut scale = 0.0f64;
                    for o in 0..outputs {
                        poll(budget, at, o)?;
                        let mut score = bias[o];
                        for j in 0..heads {
                            poll(budget, at, j)?;
                            let w = weights[o * heads + j];
                            if w != 0.0 {
                                score += std::hint::black_box(w * out[j].max(0.0));
                            }
                        }
                        scale = scale.max(score.abs());
                        scores.push(score);
                    }
                    for (column, &class) in self.terminal.adversaries().iter().enumerate() {
                        poll(budget, at, column)?;
                        let margin = scores[self.terminal.target()] - scores[class];
                        audit.check_lower(
                            "root_dj",
                            class,
                            sample,
                            self.lower(column, ordinary)?,
                            margin,
                        );
                        audit.check_positive("margin", class, sample, margin, scale);
                    }
                }
                values[op + 1] = Some(out);
            }
        }
        qualify(budget, at)?;
        drop(frame);
        lease.complete().map_err(refusal)
    }
}

impl CertifiedFrame for PreparedCertifiedFrame {
    fn input_box(&self) -> (&[f64], &[f64]) {
        let r = self.source.construction().prefix().root();
        (r.lo(), r.hi())
    }
    fn point_cost_secs(&self) -> f64 {
        self.plan.macs as f64 / POINT_MACS_PER_SEC
    }
    fn point_bytes(&self) -> u64 {
        u64::try_from(self.plan.state_bytes.saturating_add(self.plan.input_bytes))
            .unwrap_or(u64::MAX)
    }
    fn audit(
        &self,
        points: &Array2<f64>,
        audit: &mut ContainmentAudit,
    ) -> std::result::Result<(), String> {
        self.audit_checked(points, audit, None)
            .map_err(|e| e.to_string())
    }
}

pub(in crate::margin_row) fn authorize_prepared_terminal(
    source: PreparedRootSource,
    terminal: PreparedTerminal,
    expected_target: usize,
    expected_policy: RootPrecisionRuntimePolicy,
    requested: Option<Instant>,
) -> Result<PreparedPublication> {
    authorize_inner(
        source,
        terminal,
        expected_target,
        expected_policy,
        requested,
        None,
    )
}

/// Adopt only the actual ordinary producer's sealed evaluation. Its YPack,
/// source, numerical planes and original claims remain retained in the frame.
pub(in crate::margin_row) fn authorize_prepared_evaluation(
    source: PreparedRootSource,
    terminal: PreparedTerminal,
    evaluation: PreparedRootEvaluation,
    expected_target: usize,
    expected_policy: RootPrecisionRuntimePolicy,
    requested: Option<Instant>,
) -> Result<PreparedPublication> {
    authorize_with_evaluation_inner(
        source,
        terminal,
        Some(evaluation),
        None,
        expected_target,
        expected_policy,
        requested,
        None,
    )
}

/// Publish only a complete partition produced by the real prepared driver.
/// The original root, conditional evaluations and paid cover remain held by
/// the frame throughout G2, G3, and every subsequent writer checkpoint.
pub(in crate::margin_row) fn authorize_prepared_tree(
    certificate: PreparedTreeCertificate,
    requested: Option<Instant>,
) -> Result<PreparedPublication> {
    let at = requested.map_or(certificate.deadline(), |d| d.min(certificate.deadline()));
    certificate.checkpoint(Some(at))?;
    let open = certificate.root().root_open();
    open.checkpoint(Some(at))?;
    let source = open.session().source().clone();
    let terminal = open.terminal().clone();
    let ordinary = open.ordinary_evaluation().cloned();
    let target = open.target();
    let policy = open.session().policy();
    authorize_with_evaluation_inner(
        source,
        terminal,
        ordinary,
        Some(certificate),
        target,
        policy,
        Some(at),
        None,
    )
}

// Controls interrupt the real completed authorizer/guard; they cannot inject
// covers, supplied positive bounds or a publication token.
fn authorize_inner(
    source: PreparedRootSource,
    terminal: PreparedTerminal,
    expected_target: usize,
    expected_policy: RootPrecisionRuntimePolicy,
    requested: Option<Instant>,
    observer: Option<&mut dyn FnMut()>,
) -> Result<PreparedPublication> {
    authorize_with_evaluation_inner(
        source,
        terminal,
        None,
        None,
        expected_target,
        expected_policy,
        requested,
        observer,
    )
}

fn authorize_with_evaluation_inner(
    source: PreparedRootSource,
    terminal: PreparedTerminal,
    ordinary: Option<PreparedRootEvaluation>,
    tree: Option<PreparedTreeCertificate>,
    expected_target: usize,
    expected_policy: RootPrecisionRuntimePolicy,
    requested: Option<Instant>,
    mut observer: Option<&mut dyn FnMut()>,
) -> Result<PreparedPublication> {
    let mut at = source
        .deadline(requested)?
        .min(terminal.deadline(requested)?);
    if let Some(evaluation) = &ordinary {
        at = at.min(evaluation.deadline(Some(at))?);
        evaluation.adapter().validate_source(&source, Some(at))?;
    }
    let tree_lower = if let Some(certificate) = &tree {
        at = at.min(certificate.deadline());
        certificate.checkpoint(Some(at))?;
        if !certificate.source().same_source(&source)
            || !certificate.budget().shares_ledger(source.budget())
            || certificate.receipt().bound_leaves == 0
        {
            return Err(invalid(
                "prepared publication tree must retain its original source",
            ));
        }
        let lower = certificate.cover(Some(at))?.min_margin();
        if !lower.is_finite() || lower <= 0.0 {
            return Err(invalid(
                "prepared publication tree has no finite positive original margin",
            ));
        }
        Some(lower)
    } else {
        None
    };
    let budget = source.budget().clone();
    qualify(&budget, at)?;
    source.checkpoint(Some(at))?;
    terminal.validate_source(
        source.construction(),
        terminal.target(),
        terminal.adversaries(),
        Some(at),
    )?;
    if source.construction().prefix().root().policy() != expected_policy
        || expected_policy.root_f32
        || expected_policy.tableau_f32
        || terminal.target() != expected_target
        || terminal.adversaries().len()
            != source
                .construction()
                .prefix()
                .root()
                .model()
                .net()
                .n_out
                .saturating_sub(1)
    {
        return Err(invalid(
            "prepared publication requested policy is not actual original delivery",
        ));
    }
    source.construction().validate_reference(Some(at))?;
    let n = terminal.adversaries().len();
    let mut header = admit(
        &budget,
        WorkRequest {
            scratch_bytes: META,
            work: work(
                add(
                    mul(
                        source
                            .construction()
                            .prefix()
                            .root()
                            .model()
                            .net()
                            .ops
                            .len(),
                        64,
                    )?,
                    n,
                )?,
                128,
            )?,
            ..WorkRequest::default()
        },
        at,
    )?;
    at = min_lease(at, &header);
    let plan = point_plan(source.construction(), at)?;
    header.complete().map_err(refusal)?;
    drop(header);
    let persistent = add(owner_bytes(), mul(n, size_of::<usize>())?)?;
    let scratch = add(
        add(
            mul(n, size_of::<ClassCover>())?,
            mul(n, 5 * size_of::<usize>())?,
        )?,
        META,
    )?;
    let charged_work = work(mul(n, add(class_steps(n), 128)?)?, 64)?;
    let mut lease = admit(
        &budget,
        WorkRequest {
            persistent_bytes: persistent,
            scratch_bytes: scratch,
            work: charged_work,
            proposals: 0,
        },
        at,
    )?;
    at = min_lease(at, &lease);
    let frame = PreparedCertifiedFrame {
        source,
        terminal,
        ordinary,
        tree,
        tree_lower,
        plan,
        deadline: at,
        policy: expected_policy,
        expected_target,
    };
    frame.validate(at)?;
    let ordinary_bounds = frame.ordinary_bounds(at)?;
    // Canonicalization is proof bookkeeping only: the terminal continues to
    // retain the exact caller order. Each sorted class carries its ORIGINAL
    // numerical column, so order changes cannot silently relabel a margin.
    let mut order = filled(n, (usize::MAX, usize::MAX), &budget, at)?;
    for (column, &class) in frame.terminal.adversaries().iter().enumerate() {
        poll(&budget, at, column)?;
        if class
            >= frame
                .source
                .construction()
                .prefix()
                .root()
                .model()
                .net()
                .n_out
            || class == expected_target
        {
            return Err(invalid(
                "prepared Instance publication invalid original class",
            ));
        }
        let position = if class > expected_target {
            class - 1
        } else {
            class
        };
        if order[position].1 != usize::MAX {
            return Err(invalid(
                "prepared Instance publication repeated original class",
            ));
        }
        order[position] = (class, column);
    }
    qualify(&budget, at)?;
    let mut required = exact_vec(n, &budget, at)?;
    let mut covers = exact_vec(if frame.tree.is_some() { 1 } else { n }, &budget, at)?;
    let mut expected_class = 0usize;
    for (i, &(class, column)) in order.iter().enumerate() {
        poll(&budget, at, i)?;
        if expected_class == expected_target {
            expected_class = add(expected_class, 1)?;
        }
        if class != expected_class {
            return Err(invalid(
                "prepared Instance publication missing an original class",
            ));
        }
        expected_class = add(expected_class, 1)?;
        if i > 0 && order[i - 1].0 >= class {
            return Err(invalid(
                "prepared publication original canonical classes are not unique",
            ));
        }
        if frame.tree.is_none() {
            let lower = frame.lower(column, ordinary_bounds)?;
            let cover = cert_mr::root_class(
                frame.source.construction().prefix().root().mode(),
                class,
                lower,
            )
            .ok_or_else(|| {
                invalid("prepared terminal original class margin is not strictly positive")
            })?;
            covers.push(cover);
        }
        required.push(class);
    }
    if expected_class == expected_target {
        expected_class = add(expected_class, 1)?;
    }
    if expected_class
        != frame
            .source
            .construction()
            .prefix()
            .root()
            .model()
            .net()
            .n_out
    {
        return Err(invalid(
            "prepared Instance publication incomplete original class domain",
        ));
    }
    lease.mark_materialized().map_err(refusal)?;
    if let Some(tree) = &frame.tree {
        let cover = tree.cover(Some(at))?;
        for (i, (&actual, &expected)) in cover.classes().iter().zip(&required).enumerate() {
            poll(&budget, at, i)?;
            if actual != expected {
                return Err(invalid(
                    "prepared publication tree canonical class mapping changed",
                ));
            }
        }
        if cover.classes().len() != required.len() {
            return Err(invalid("prepared publication tree misses original classes"));
        }
        // The temporary class-only cover copy and authorizer token overlap
        // were included in the existing five-class-plane scratch allowance.
        covers.push(ClassCover::Tree(cover.clone()));
    } else if !aligned_single_class_covers(&required, &covers) {
        return Err(invalid(
            "prepared publication did not produce canonical original class covers",
        ));
    }
    let outcome = paid_containment(
        &frame,
        ny_core::proof_profile::audit_rates().containment_points,
    )?;
    frame.validate(at)?;
    let guards = outcome.report();
    let token = authorize_unsat(TokenScope::Instance, &required, &covers, &guards)
        .map_err(|e| invalid(&format!("prepared original class authorizer refused: {e}")))?;
    replay_classes(&frame, &token, at)?;
    if let Some(hook) = observer.as_mut() {
        hook();
    }
    frame.validate(at)?;
    qualify(&budget, at)?;
    let receipt = PreparedPublicationReceipt {
        classes: n,
        target: expected_target,
        containment_points: outcome.points,
        containment_checks: outcome.audit.checks(),
        containment_blind: outcome.audit.blind(),
        containment_skipped: outcome.skipped.is_some(),
        policy: expected_policy,
        reference: frame.source.construction().reference_receipt(),
        terminal: frame.terminal.receipt(),
        ordinary: frame.ordinary.as_ref().map(PreparedRootEvaluation::receipt),
        tree: frame.tree.as_ref().map(PreparedTreeCertificate::receipt),
    };
    let actual = add(
        owner_bytes(),
        mul(token.classes().len(), size_of::<usize>())?,
    )?;
    if actual > persistent {
        return Err(cap_error(
            actual,
            persistent,
            "prepared publication actual owner capacity",
        ));
    }
    drop(covers);
    drop(required);
    drop(order);
    drop(outcome);
    let resident = lease.commit_persistent().map_err(refusal)?;
    let owner = PreparedPublication(
        Arc::new(PublicationInner {
            frame,
            token,
            deadline: at,
            receipt,
            resident,
        }),
        at,
    );
    owner.checkpoint(Some(at))?;
    Ok(owner)
}

#[derive(Clone, Copy)]
struct PointPlan {
    state_bytes: usize,
    input_bytes: usize,
    frame_bytes: usize,
    point_work: u64,
    macs: usize,
}
fn point_plan(source: &PreparedRootConstruction, at: Instant) -> Result<PointPlan> {
    let root = source.prefix().root();
    let net = root.model().net();
    let budget = root.budget();
    let mut rows = 0usize;
    let mut macs = 0usize;
    let mut operations = 0usize;
    for (i, &n) in net.tsize.iter().enumerate() {
        poll(budget, at, i)?;
        if i <= net.i_gemm1 + 1 {
            rows = add(rows, n)?;
        }
    }
    for (i, op) in net.ops.iter().enumerate() {
        poll(budget, at, i)?;
        let n = net.tsize[i + 1];
        let m = match op {
            TwinOp::Conv(c) => mul(n, c.k_fwd)?,
            TwinOp::Gemm { shape, .. } => mul(shape.0, shape.1)?,
            _ => n,
        };
        macs = add(macs, m)?;
        operations = add(operations, add(mul(m, 64)?, mul(n, 128)?)?)?;
    }
    let state_bytes = add(
        add(
            mul(add(rows, net.n_out)?, 8)?,
            mul(net.ops.len() + 1, size_of::<Option<Vec<f64>>>())?,
        )?,
        META,
    )?;
    let input_bytes = mul(net.n_in, 8)?;
    let frame_bytes = add(
        mul(
            source.prefix().layer_count(),
            size_of::<&crate::margin_row::root::LayerGates>(),
        )?,
        size_of::<super::prefix::PreparedPrefixFrame<'_>>(),
    )?;
    Ok(PointPlan {
        state_bytes,
        input_bytes,
        frame_bytes,
        point_work: work(operations, 4)?,
        macs,
    })
}

// Prelease sampling BEFORE any sample allocation. A resource refusal skips
// this downgrade-only guard, but cancellation/deadline/live-environment
// refusal still aborts publication at the original final checkpoint.
fn paid_containment(frame: &PreparedCertifiedFrame, max_points: u32) -> Result<ContainmentOutcome> {
    let started = Instant::now();
    let at = frame.deadline(None)?;
    let root = frame.source.construction().prefix().root();
    let budget = root.budget();
    let per = frame.point_bytes().max(1);
    let mut count =
        (max_points as usize).min(usize::try_from(MAX_GUARD_BYTES / per).unwrap_or(usize::MAX));
    let duration = at
        .saturating_duration_since(started)
        .mul_f64(GUARD_FRACTION);
    let cost = frame.point_cost_secs();
    if cost.is_finite() && cost > 0.0 {
        let fit = (duration.as_secs_f64() / cost).floor();
        count = count.min(if fit.is_finite() && fit > 0.0 {
            fit.min(1e9) as usize
        } else {
            0
        });
    }
    if count == 0 {
        return Ok(skipped("budget"));
    }
    let lease = loop {
        qualify(budget, at)?;
        let bytes = add(mul(frame.plan.input_bytes, count)?, META)?;
        // Both the generated sample slab and the actual scalar evaluator are
        // simultaneously live. Admit the complete overlap before the slab.
        let peak = preflight(
            budget,
            add(
                bytes,
                add(frame.plan.state_bytes, mul(frame.plan.frame_bytes, 2)?)?,
            )?,
            "prepared containment complete original overlap",
        );
        let trial = peak.and_then(|()| {
            admit(
                budget,
                WorkRequest {
                    scratch_bytes: bytes,
                    work: work(mul(root.lo().len(), count)?, 64)?,
                    ..WorkRequest::default()
                },
                at,
            )
        });
        match trial {
            Ok(lease) => break lease,
            Err(_) => {
                qualify(budget, at)?;
                if count <= 1 {
                    return Ok(skipped("resources"));
                }
                count = count.div_ceil(2);
            }
        }
    };
    let mut lease = lease;
    let at = min_lease(at, &lease);
    let points = sample_points_paid(root.lo(), root.hi(), count, budget, at)?;
    lease.mark_materialized().map_err(refusal)?;
    let mut audit = ContainmentAudit::default();
    let guard_deadline = started.checked_add(duration).unwrap_or(at).min(at);
    let evaluated = frame
        .audit_checked(&points, &mut audit, Some(guard_deadline))
        .map_err(|e| e.to_string());
    qualify(budget, at)?;
    drop(points);
    lease.complete().map_err(refusal)?;
    Ok(containment_completion(
        evaluated,
        audit,
        count,
        started.elapsed().as_secs_f64(),
    ))
}
fn containment_completion(
    evaluated: std::result::Result<(), String>,
    audit: ContainmentAudit,
    count: usize,
    secs: f64,
) -> ContainmentOutcome {
    match evaluated {
        Ok(()) => ContainmentOutcome {
            points: count,
            audit,
            skipped: None,
            secs,
        },
        Err(reason) => ContainmentOutcome {
            points: 0,
            // Incomplete sampling cannot erase an already observed actual
            // enclosure violation. report() remains downgrade-only and must
            // still trip even if a later point exhausted the phase budget.
            audit,
            skipped: Some(format!("evaluator: {reason}")),
            secs,
        },
    }
}
fn skipped(reason: &str) -> ContainmentOutcome {
    ContainmentOutcome {
        points: 0,
        audit: ContainmentAudit::default(),
        skipped: Some(reason.to_owned()),
        secs: 0.0,
    }
}
fn sample_points_paid(
    lo: &[f64],
    hi: &[f64],
    count: usize,
    budget: &WorkBudget,
    at: Instant,
) -> Result<Array2<f64>> {
    let mut slab = filled(mul(lo.len(), count)?, 0.0, budget, at)?;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let alternating = if lo.is_empty() {
        0
    } else {
        usize::BITS as usize - lo.len().leading_zeros() as usize
    };
    for sample in 0..count {
        qualify(budget, at)?;
        for i in 0..lo.len() {
            poll(budget, at, i)?;
            let x = match sample {
                0 => 0.5 * lo[i] + 0.5 * hi[i],
                1 => lo[i],
                2 => hi[i],
                s if s - 3 < alternating => {
                    let stride = 1usize << (s - 3);
                    if (i / stride) % 2 == 1 {
                        hi[i]
                    } else {
                        lo[i]
                    }
                }
                _ => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (1.0 - u) * lo[i] + u * hi[i]
                }
            };
            slab[i * count + sample] = x.max(lo[i]).min(hi[i]);
        }
    }
    Array2::from_shape_vec((lo.len(), count), slab)
        .map_err(|_| invalid("prepared containment sample slab shape changed"))
}
fn replay_classes(frame: &PreparedCertifiedFrame, token: &UnsatToken, at: Instant) -> Result<()> {
    let terminal = &frame.terminal;
    let budget = terminal.prefix().root().budget();
    let n = terminal.adversaries().len();
    let mut lease = admit(
        budget,
        WorkRequest {
            work: work(n, 64)?,
            ..WorkRequest::default()
        },
        at,
    )?;
    if token.scope() != TokenScope::Instance
        || token.classes().len() != n
        || n != terminal
            .prefix()
            .root()
            .model()
            .net()
            .n_out
            .saturating_sub(1)
        || !token.min_margin().is_finite()
        || token.min_margin() <= 0.0
    {
        return Err(invalid("prepared publication class token is incomplete"));
    }
    let mut expected = 0usize;
    for (i, &class) in token.classes().iter().enumerate() {
        poll(budget, at, i)?;
        if expected == terminal.target() {
            expected = add(expected, 1)?;
        }
        if class != expected {
            return Err(invalid(
                "prepared publication token omits an original class",
            ));
        }
        expected = add(expected, 1)?;
    }
    if expected == terminal.target() {
        expected = add(expected, 1)?;
    }
    if expected != terminal.prefix().root().model().net().n_out {
        return Err(invalid("prepared publication token class domain changed"));
    }
    let mut minimum = f64::INFINITY;
    let ordinary = frame.ordinary_bounds(at)?;
    for (i, &class) in terminal.adversaries().iter().enumerate() {
        poll(budget, at, i)?;
        let lower = frame.lower(i, ordinary)?;
        let position = if class > terminal.target() {
            class - 1
        } else {
            class
        };
        if !lower.is_finite()
            || lower <= 0.0
            || class == terminal.target()
            || token.classes().get(position) != Some(&class)
        {
            return Err(invalid(
                "prepared publication original ordered class replay refused",
            ));
        }
        minimum = minimum.min(lower);
    }
    if minimum.to_bits() != token.min_margin().to_bits() {
        return Err(invalid("prepared publication margin token changed"));
    }
    qualify(budget, at)?;
    lease.complete().map_err(refusal)
}
fn value(values: &[Option<Vec<f64>>], id: usize) -> Result<&[f64]> {
    values
        .get(id)
        .and_then(Option::as_deref)
        .ok_or_else(|| invalid("missing original containment edge"))
}
fn qualify(budget: &WorkBudget, at: Instant) -> Result<()> {
    checkpoint(budget, Some(at))?;
    ny_core::require_f64_interval_proof_environment()?;
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return Err(invalid("prepared publication TwoSum environment refused"));
    }
    Ok(())
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, Some(at))?;
    }
    Ok(())
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| d.min(at))
}
fn overflow() -> NyError {
    invalid("prepared publication capacity/work overflow")
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(n: usize, k: usize) -> Result<u64> {
    u64::try_from(add(mul(n, k)?, META)?).map_err(|_| overflow())
}
fn class_steps(n: usize) -> usize {
    usize::BITS as usize - n.leading_zeros() as usize
}
fn owner_bytes() -> usize {
    size_of::<PublicationInner>() + 2 * size_of::<usize>() + META
}
fn cap_error(required_bytes: usize, budget_bytes: usize, site: &'static str) -> NyError {
    NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site,
    }
}
fn preflight(budget: &WorkBudget, new: usize, site: &'static str) -> Result<()> {
    let r = budget.receipt().map_err(refusal)?;
    let required = add(add(r.live_scratch_bytes, r.live_persistent_bytes)?, new)?;
    if required > ROOT_SOURCE_BYTES_CAP {
        Err(cap_error(required, ROOT_SOURCE_BYTES_CAP, site))
    } else {
        Ok(())
    }
}
fn admit(budget: &WorkBudget, request: WorkRequest, at: Instant) -> Result<WorkLease> {
    qualify(budget, at)?;
    preflight(
        budget,
        add(request.scratch_bytes, request.persistent_bytes)?,
        "prepared publication total original live",
    )?;
    budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)
}
fn exact_vec<T>(n: usize, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    qualify(budget, at)?;
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| invalid("prepared publication allocation refused"))?;
    qualify(budget, at)?;
    if v.capacity() != n {
        return Err(cap_error(
            mul(v.capacity(), size_of::<T>())?,
            mul(n, size_of::<T>())?,
            "prepared publication exact actual capacity",
        ));
    }
    Ok(v)
}
fn filled<T: Clone>(n: usize, x: T, budget: &WorkBudget, at: Instant) -> Result<Vec<T>> {
    let mut v = exact_vec(n, budget, at)?;
    for i in 0..n {
        poll(budget, at, i)?;
        v.push(x.clone());
    }
    Ok(v)
}

#[cfg(test)]
#[path = "publication_tests.rs"]
mod tests;
