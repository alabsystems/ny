// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Retained prepared proof facade for an original-source CLI owner.
//!
//! A `TwinSpec` is a conditional mathematical declaration. This facade proves
//! that declaration; it does not authenticate an ONNX pathname or a VNN-LIB
//! objective. The CLI must extract the moved declaration from its retained
//! source bytes and authenticate the exact input domain and complete unsafe
//! objective before publication. No token can be detached from this owner.

use std::{fmt, time::Instant};

use ny_core::Result;

use super::{
    compiler::OriginalModel,
    construction::{build_prepared_root, PreparedConstructionOptions},
    ordinary_pass::{build_prepared_ordinary_rows, PreparedOrdinaryRows},
    prefix::OriginalRoot,
    publication::{
        authorize_prepared_evaluation, authorize_prepared_terminal, authorize_prepared_tree,
        share_prepared_root, PreparedPublication, PreparedRootSource,
    },
    root_adapter::{project_prepared_root, PreparedRootAdapter},
    terminal::build_prepared_terminal,
};
use crate::{
    certify::{
        work::{LeaseClass, WorkBudget, WorkRequest},
        GuardReport,
    },
    margin_row::{
        bab::{prepared_root_eval_reuse, PreparedRootEvaluation},
        root::RetainCfg,
        BackwardIntermRuntimePolicy, RootPrecisionRuntimePolicy, TwinSpec,
    },
};

/// Scheduling and persistent row requests, preserved by the actual builder.
/// These do not claim that clipping or downstream ranker delivery is complete.
#[derive(Clone, Copy, Debug)]
pub struct PreparedRootProofOptions {
    pub reference_arithmetic: super::PreparedReferenceArithmetic,
    pub keep_input_rows: bool,
    pub row_f32: bool,
    pub prefer_patch: bool,
    pub dense_tile: usize,
    pub patch_tile: usize,
    pub patch_ratio: f64,
}

impl Default for PreparedRootProofOptions {
    fn default() -> Self {
        let options = PreparedConstructionOptions::default();
        Self {
            reference_arithmetic: options.reference_arithmetic,
            keep_input_rows: options.keep_input_rows,
            row_f32: options.row_f32,
            prefer_patch: options.prefer_patch,
            dense_tile: options.dense_tile,
            patch_tile: options.patch_tile,
            patch_ratio: options.patch_ratio,
        }
    }
}

/// Immutable original construction, original terminal and authorized instance
/// proof. Clone shares their exact owners and last-owner reservations.
#[derive(Clone)]
pub struct PreparedRootProof {
    publication: PreparedPublication,
    session: PreparedRootSession,
    attempt: PreparedRootAttemptReceipt,
}

/// Actual root scheduling observations. A refused or open attempt grants no
/// verdict, and these counters are not time or memory measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparedRootAttemptReceipt {
    pub reference: super::PreparedReferenceReceipt,
    pub terminal_closed: bool,
    pub ordinary_attempted: bool,
    pub ordinary_completed: bool,
    pub ordinary_refused: bool,
}

/// Historical observations of completed f32 nominal-row storage transfers.
/// Totals count the actual old/new owned payload and newly allocated nominal
/// capacities across transfers; the admitted peak is a maximum, not a sum.
/// Original conservative reservations and f64 error planes remain independent.
/// This value owns no source/ledger claim, authorizes no arithmetic or writer,
/// and does not mean root/tableau f32 kernels, replay or default routing ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparedRowStorageReceipt {
    pub tiles: usize,
    pub coefficient_cells_stored: usize,
    pub bias_cells_stored: usize,
    pub nominal_cells_changed: usize,
    pub payload_bytes_before_total: usize,
    pub payload_bytes_after_total: usize,
    pub nominal_allocation_bytes_total: usize,
    pub max_admitted_peak_bytes: usize,
}

/// Explicit downstream products; these cannot be silently discharged by an
/// early terminal proof. Construction must have retained the required rows.
#[derive(Clone, Copy, Debug)]
pub struct PreparedRootRetention {
    pub per_layer: usize,
    pub budget_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PreparedRootConsumerOptions {
    pub backward_interm: BackwardIntermRuntimePolicy,
    pub clip_rows: bool,
    pub retain: Option<PreparedRootRetention>,
}

impl PreparedRootConsumerOptions {
    fn requested(self) -> bool {
        self.backward_interm.enabled || self.clip_rows || self.retain.is_some()
    }
}

/// A genuine completed source can remain useful after a root proof is open.
/// No model parameters, supplied bounds, mutable gates or tokens are imported.
#[derive(Clone)]
pub struct PreparedRootSession {
    source: PreparedRootSource,
    policy: RootPrecisionRuntimePolicy,
    deadline: Instant,
}

/// The actual ordinary evaluation is retained for later checked search. A
/// child domain must acquire distinct original-domain authority; this owner
/// does not relabel unconditional root rows as a descendant certificate.
#[derive(Clone)]
pub struct PreparedRootOpen {
    session: PreparedRootSession,
    terminal: super::terminal::PreparedTerminal,
    ordinary: Option<PreparedRootEvaluation>,
    deadline: Instant,
    receipt: PreparedRootAttemptReceipt,
}

pub enum PreparedRootOutcome {
    Proven(PreparedRootProof),
    Open(PreparedRootOpen),
}

// Check the independently retained requested projection before consuming its
// rows. Source equality alone cannot distinguish projections of the same source
// with different product demands. Cloning the adapter shares its existing claim;
// it does not clone the row owner or force the shared-payload copy route.
fn evaluate_requested_projection(
    expected: &PreparedRootAdapter,
    rows: PreparedOrdinaryRows,
    target: usize,
    ordered_classes: &[usize],
    requested: Option<Instant>,
) -> Result<PreparedRootEvaluation> {
    let mut at = expected.deadline(Some(rows.deadline(requested)?))?;
    let budget = expected.source().budget();
    // A fixed quote covers the identity/ledger/header checks, inherited clocks
    // and original source/environment checkpoints. No numerical row is scanned.
    let mut binding = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 512,
                ..WorkRequest::default()
            },
        )
        .map_err(super::ir::refusal)?;
    at = binding.deadline().map_or(at, |deadline| at.min(deadline));
    rows.validate_adapter(expected, Some(at))?;
    super::ir::checkpoint(budget, Some(at))?;
    binding.complete().map_err(super::ir::refusal)?;
    drop(binding);
    prepared_root_eval_reuse(rows, target, ordered_classes, Some(at))
}

impl PreparedRootSession {
    pub fn compile_declared_twin(
        spec: TwinSpec,
        lower: &[f64],
        upper: &[f64],
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let model = OriginalModel::compile(spec, budget, requested)?;
        Self::from_compiled_model(
            model,
            super::ir::OriginalInputDomain { lower, upper },
            policy,
            options,
            budget,
            requested,
        )
    }

    pub(super) fn from_compiled_model(
        model: OriginalModel,
        domain: super::ir::OriginalInputDomain<'_>,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        if !model.budget().shares_ledger(budget) {
            return Err(super::ir::invalid(
                "prepared session foreign original ledger",
            ));
        }
        let root = OriginalRoot::new(model, domain.lower, domain.upper, policy, requested)?;
        let construction = build_prepared_root(
            root,
            PreparedConstructionOptions {
                reference_arithmetic: options.reference_arithmetic,
                keep_input_rows: options.keep_input_rows,
                row_f32: options.row_f32,
                prefer_patch: options.prefer_patch,
                dense_tile: options.dense_tile,
                patch_tile: options.patch_tile,
                patch_ratio: options.patch_ratio,
            },
            requested,
        )?;
        let source = share_prepared_root(construction, requested)?;
        let session = Self {
            deadline: source.deadline(requested)?,
            source,
            policy,
        };
        session.checkpoint(requested)?;
        Ok(session)
    }

    pub fn budget(&self) -> &WorkBudget {
        self.source.budget()
    }
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.source.deadline(Some(
            requested.map_or(self.deadline, |d| d.min(self.deadline)),
        ))
    }
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        self.source.checkpoint(Some(self.deadline(requested)?))
    }
    pub fn restricted(&self, requested: Option<Instant>) -> Result<Self> {
        let at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        Ok(Self {
            source: self.source.clone(),
            policy: self.policy,
            deadline: at,
        })
    }
    pub fn declared_lower(&self) -> &[f64] {
        self.source.construction().prefix().root().lo()
    }
    pub fn declared_upper(&self) -> &[f64] {
        self.source.construction().prefix().root().hi()
    }
    pub fn source_digest(&self) -> &[u8; 32] {
        self.source.construction().identity().source_digest()
    }
    pub fn compiled_digest(&self) -> &[u8; 32] {
        self.source.construction().identity().compiled_digest()
    }
    /// Observations only; the retained source is the proof authority.
    pub fn reference_receipt(&self) -> super::PreparedReferenceReceipt {
        self.source.construction().reference_receipt()
    }
    /// Copy historical delivery observations without rescanning rows,
    /// admitting work, retaining owners or renewing the source clock. The
    /// existing checkpoint/proof/publication APIs still decide all validity.
    pub fn row_storage_receipt(&self) -> PreparedRowStorageReceipt {
        let stats = self.source.construction().stats();
        PreparedRowStorageReceipt {
            tiles: stats.f32_tiles,
            coefficient_cells_stored: stats.f32_coefficient_cells,
            bias_cells_stored: stats.f32_bias_cells,
            nominal_cells_changed: stats.f32_changed_cells,
            payload_bytes_before_total: stats.f32_payload_bytes_before_total,
            payload_bytes_after_total: stats.f32_payload_bytes_after_total,
            nominal_allocation_bytes_total: stats.f32_nominal_allocation_bytes_total,
            max_admitted_peak_bytes: stats.f32_max_admitted_peak_bytes,
        }
    }
    pub fn policy(&self) -> RootPrecisionRuntimePolicy {
        self.policy
    }
    pub fn n_out(&self) -> usize {
        self.source
            .construction()
            .prefix()
            .root()
            .model()
            .net()
            .n_out
    }

    /// Compare genuine source ownership, independently of matching hashes or
    /// equal resource limits. This observation cannot mint proof authority.
    pub fn same_source(&self, other: &Self) -> bool {
        self.source.same_source(&other.source)
    }
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootSource {
        &self.source
    }

    pub fn try_prove_root(
        &self,
        target: usize,
        ordered_classes: &[usize],
        requested: Option<Instant>,
    ) -> Result<PreparedRootOutcome> {
        self.try_prove_root_with_consumers(
            target,
            ordered_classes,
            PreparedRootConsumerOptions::default(),
            requested,
        )
    }

    pub fn try_prove_root_with_consumers(
        &self,
        target: usize,
        ordered_classes: &[usize],
        consumers: PreparedRootConsumerOptions,
        requested: Option<Instant>,
    ) -> Result<PreparedRootOutcome> {
        self.try_prove_inner(target, ordered_classes, consumers, requested, None)
    }

    // A control observes actual completed terminal work and may cancel or
    // occupy the original ledger. It cannot supply any numerical evidence.
    fn try_prove_inner(
        &self,
        target: usize,
        ordered_classes: &[usize],
        consumers: PreparedRootConsumerOptions,
        requested: Option<Instant>,
        mut after_terminal: Option<&mut dyn FnMut()>,
    ) -> Result<PreparedRootOutcome> {
        let mut at = self.deadline(requested)?;
        self.checkpoint(Some(at))?;
        if target >= self.n_out() || ordered_classes.len() != self.n_out().saturating_sub(1) {
            return Err(super::ir::invalid(
                "prepared root requires complete original objective",
            ));
        }
        let terminal = build_prepared_terminal(
            self.source.construction(),
            target,
            ordered_classes,
            Some(at),
        )?;
        at = at.min(terminal.deadline(Some(at))?);
        let mut scan = self
            .budget()
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: u64::try_from(ordered_classes.len())
                        .ok()
                        .and_then(|n| n.checked_mul(32))
                        .and_then(|n| n.checked_add(4096))
                        .ok_or_else(|| super::ir::invalid("prepared root closure scan overflow"))?,
                    ..WorkRequest::default()
                },
            )
            .map_err(super::ir::refusal)?;
        at = scan.deadline().map_or(at, |d| d.min(at));
        let mut receipt = PreparedRootAttemptReceipt {
            reference: self.reference_receipt(),
            terminal_closed: true,
            ..PreparedRootAttemptReceipt::default()
        };
        for (i, &bound) in terminal.lower().iter().enumerate() {
            if i % 256 == 0 {
                super::ir::checkpoint(self.budget(), Some(at))?;
            }
            if !bound.is_finite() {
                return Err(super::ir::invalid(
                    "prepared terminal nonfinite root margin",
                ));
            }
            receipt.terminal_closed &= bound > 0.0;
        }
        scan.complete().map_err(super::ir::refusal)?;
        if let Some(observer) = after_terminal.as_mut() {
            observer();
        }
        self.checkpoint(Some(at))?;
        if receipt.terminal_closed && !consumers.requested() {
            let publication = authorize_prepared_terminal(
                self.source.clone(),
                terminal,
                target,
                self.policy,
                Some(at),
            )?;
            return Ok(PreparedRootOutcome::Proven(PreparedRootProof {
                publication,
                session: self.restricted(Some(at))?,
                attempt: receipt,
            }));
        }
        receipt.ordinary_attempted = true;
        let evaluated = (|| {
            let adapter = project_prepared_root(
                self.source.clone(),
                self.policy,
                consumers.backward_interm,
                consumers.clip_rows,
                consumers.retain.map(|cfg| RetainCfg {
                    per_layer: cfg.per_layer,
                    budget_bytes: cfg.budget_bytes,
                }),
                Some(at),
            )?;
            let rows = build_prepared_ordinary_rows(adapter.clone(), Some(at))?;
            evaluate_requested_projection(&adapter, rows, target, ordered_classes, Some(at))
        })();
        let ordinary = match evaluated {
            Ok(evaluation) => {
                receipt.ordinary_completed = true;
                at = at.min(evaluation.deadline(Some(at))?);
                Some(evaluation)
            }
            Err(
                ny_core::NyError::CpuMemoryExceeded { .. }
                | ny_core::NyError::UnsupportedConfiguration(_),
            ) => {
                self.checkpoint(Some(at))?;
                receipt.ordinary_refused = true;
                None
            }
            Err(error) => return Err(error),
        };
        let mut closed = receipt.terminal_closed;
        if let Some(evaluation) = &ordinary {
            evaluation
                .adapter()
                .validate_source(&self.source, Some(at))?;
            let bounds = evaluation.bounds(Some(at))?;
            if evaluation.target() != target || bounds.len() != terminal.lower().len() {
                return Err(super::ir::invalid(
                    "prepared ordinary original objective changed",
                ));
            }
            let mut lease = self
                .budget()
                .lease_for(
                    LeaseClass::Fallback,
                    WorkRequest {
                        work: u64::try_from(bounds.len())
                            .ok()
                            .and_then(|n| n.checked_mul(32))
                            .and_then(|n| n.checked_add(4096))
                            .ok_or_else(|| {
                                super::ir::invalid("prepared ordinary closure scan overflow")
                            })?,
                        ..WorkRequest::default()
                    },
                )
                .map_err(super::ir::refusal)?;
            at = lease.deadline().map_or(at, |d| d.min(at));
            closed = true;
            for (i, (&lower, &other)) in terminal.lower().iter().zip(bounds).enumerate() {
                if i % 256 == 0 {
                    super::ir::checkpoint(self.budget(), Some(at))?;
                }
                if !other.is_finite() {
                    return Err(super::ir::invalid(
                        "prepared ordinary nonfinite root margin",
                    ));
                }
                closed &= lower.max(other) > 0.0;
            }
            lease.complete().map_err(super::ir::refusal)?;
        }
        let ordinary = match (closed, ordinary) {
            (true, Some(evaluation)) => {
                let publication = authorize_prepared_evaluation(
                    self.source.clone(),
                    terminal,
                    evaluation,
                    target,
                    self.policy,
                    Some(at),
                )?;
                return Ok(PreparedRootOutcome::Proven(PreparedRootProof {
                    publication,
                    session: self.restricted(Some(at))?,
                    attempt: receipt,
                }));
            }
            (_, ordinary) => ordinary,
        };
        let open = PreparedRootOpen {
            session: self.restricted(Some(at))?,
            terminal,
            ordinary,
            deadline: at,
            receipt,
        };
        open.checkpoint(Some(at))?;
        Ok(PreparedRootOutcome::Open(open))
    }
}

impl PreparedRootOpen {
    pub fn session(&self) -> &PreparedRootSession {
        &self.session
    }
    pub fn target(&self) -> usize {
        self.terminal.target()
    }
    pub fn classes(&self) -> &[usize] {
        self.terminal.adversaries()
    }
    pub fn receipt(&self) -> PreparedRootAttemptReceipt {
        self.receipt
    }
    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        let at = self
            .session
            .deadline(Some(
                requested.map_or(self.deadline, |d| d.min(self.deadline)),
            ))?
            .min(self.terminal.deadline(Some(self.deadline))?);
        Ok(match &self.ordinary {
            Some(evaluation) => at.min(evaluation.deadline(Some(at))?),
            None => at,
        })
    }
    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.session.checkpoint(Some(at))?;
        self.terminal.validate_source(
            self.session.source.construction(),
            self.target(),
            self.classes(),
            Some(at),
        )?;
        if let Some(evaluation) = &self.ordinary {
            evaluation
                .adapter()
                .validate_source(self.session.source(), Some(at))?;
            evaluation.checkpoint(Some(at))?;
            if evaluation.target() != self.target()
                || evaluation.adversaries().len() != self.classes().len()
            {
                return Err(super::ir::invalid("prepared open owner foreign objective"));
            }
            let mut lease = self
                .session
                .budget()
                .lease_for(
                    LeaseClass::Fallback,
                    WorkRequest {
                        work: u64::try_from(self.classes().len())
                            .ok()
                            .and_then(|n| n.checked_mul(32))
                            .and_then(|n| n.checked_add(4096))
                            .ok_or_else(|| {
                                super::ir::invalid("prepared open owner validation overflow")
                            })?,
                        ..WorkRequest::default()
                    },
                )
                .map_err(super::ir::refusal)?;
            let at = lease.deadline().map_or(at, |d| d.min(at));
            for (i, (&actual, &expected)) in evaluation
                .adversaries()
                .iter()
                .zip(self.classes())
                .enumerate()
            {
                if i % 256 == 0 {
                    super::ir::checkpoint(self.session.budget(), Some(at))?;
                }
                if actual != expected {
                    return Err(super::ir::invalid(
                        "prepared open owner class order changed",
                    ));
                }
            }
            self.session.checkpoint(Some(at))?;
            lease.complete().map_err(super::ir::refusal)?;
        }
        Ok(())
    }
    pub(in crate::margin_row) fn terminal(&self) -> &super::terminal::PreparedTerminal {
        &self.terminal
    }
    pub(in crate::margin_row) fn ordinary_evaluation(&self) -> Option<&PreparedRootEvaluation> {
        self.ordinary.as_ref()
    }
}

impl fmt::Debug for PreparedRootProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Historical observations only: formatting does not checkpoint,
        // ratify or authorize a writer, and these distinct claims are not
        // summed or described as physical heap/RSS.
        let root = self.session.source.construction().prefix().root();
        let projection = self
            .publication
            .ordinary()
            .map(PreparedRootEvaluation::adapter);
        f.debug_struct("PreparedRootProof")
            .field("target", &self.target())
            .field("classes", &self.classes())
            .field("min_margin", &self.min_margin())
            .field("source_region", &root.region())
            .field("parameter_contract", &root.model().identity().contract())
            .field("source_policy", &self.session.policy())
            .field("prepared_reference", &self.session.reference_receipt())
            .field(
                "compiled_capacity_with_owner_allowance_bytes",
                &root.model().actual_bytes(),
            )
            .field("model_reserved_bytes", &root.model().resident_bytes())
            .field(
                "terminal_reserved_bytes",
                &self.publication.terminal().resident_bytes(),
            )
            .field(
                "publication_reserved_bytes",
                &self.publication.resident_bytes(),
            )
            .field(
                "projection_reserved_bytes",
                &projection.map(PreparedRootAdapter::resident_bytes),
            )
            .field(
                "projection_reference_conv_ops",
                &projection.map(|a| a.receipt().reference_conv_ops),
            )
            .field(
                "projection_backward_interm_request",
                &projection.map(|a| a.receipt().backward_interm),
            )
            .field(
                "projection_construction",
                &projection.map(|a| a.receipt().source),
            )
            .finish_non_exhaustive()
    }
}

impl PreparedRootProof {
    /// Complete an actual conditional search while retaining its original
    /// root session and every checked leaf. No caller objective or scalar is
    /// accepted; the complete tree supplies its own original declaration.
    pub fn from_prepared_tree(
        certificate: crate::margin_row::bab::PreparedTreeCertificate,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let at = requested.map_or(certificate.deadline(), |d| d.min(certificate.deadline()));
        certificate.checkpoint(Some(at))?;
        let open = certificate.root().root_open();
        open.checkpoint(Some(at))?;
        let session = open.session().restricted(Some(at))?;
        let attempt = open.receipt();
        let publication = authorize_prepared_tree(certificate, Some(at))?;
        let result = Self {
            publication,
            session,
            attempt,
        };
        result.checkpoint(Some(at))?;
        Ok(result)
    }

    pub fn tree_receipt(&self) -> Option<crate::margin_row::bab::PreparedTreeReceipt> {
        self.publication.receipt().tree
    }
    pub(super) fn publication(&self) -> &PreparedPublication {
        &self.publication
    }
    /// Prove the actual moved declaration using the caller's existing ledger.
    /// There is no fresh budget, supplied gate/root, or numeric proof import.
    pub fn prove_declared_twin(
        spec: TwinSpec,
        lower: &[f64],
        upper: &[f64],
        target: usize,
        ordered_classes: &[usize],
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let model = OriginalModel::compile(spec, budget, requested)?;
        Self::from_compiled_model(
            model,
            super::ir::OriginalInputDomain { lower, upper },
            target,
            ordered_classes,
            policy,
            options,
            budget,
            requested,
        )
    }

    /// Sibling producer seam for a staged, genuine compiler owner. It cannot
    /// import a mutable net, a caller digest, a supplied proof or free bytes.
    /// The old moved-declaration entry follows this identical numerical path.
    pub(super) fn from_compiled_model(
        model: OriginalModel,
        domain: super::ir::OriginalInputDomain<'_>,
        target: usize,
        ordered_classes: &[usize],
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let session = PreparedRootSession::from_compiled_model(
            model, domain, policy, options, budget, requested,
        )?;
        match session.try_prove_root(target, ordered_classes, requested)? {
            PreparedRootOutcome::Proven(proof) => {
                proof.checkpoint(requested)?;
                Ok(proof)
            }
            PreparedRootOutcome::Open(_) => Err(super::ir::invalid(
                "prepared terminal and ordinary root leave original classes open",
            )),
        }
    }

    pub fn budget(&self) -> &WorkBudget {
        self.publication.source_owner().budget()
    }

    pub fn session(&self) -> &PreparedRootSession {
        &self.session
    }

    pub fn attempt_receipt(&self) -> PreparedRootAttemptReceipt {
        self.attempt
    }

    pub fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.publication
            .deadline(Some(self.session.deadline(requested)?))
    }

    pub fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        if !self
            .session
            .source()
            .same_source(self.publication.source_owner())
        {
            return Err(super::ir::invalid(
                "prepared proof retained a foreign original session source",
            ));
        }
        self.session.checkpoint(Some(at))?;
        self.publication
            .validate_policy(self.session.policy(), Some(at))
    }

    pub fn declared_lower(&self) -> &[f64] {
        self.publication.source().prefix().root().lo()
    }

    pub fn declared_upper(&self) -> &[f64] {
        self.publication.source().prefix().root().hi()
    }

    pub fn source_digest(&self) -> &[u8; 32] {
        self.publication.source().identity().source_digest()
    }

    pub fn compiled_digest(&self) -> &[u8; 32] {
        self.publication.source().identity().compiled_digest()
    }

    pub fn target(&self) -> usize {
        self.publication.terminal().target()
    }

    /// Exact producer order, including the original class IDs.
    pub fn classes(&self) -> &[usize] {
        self.publication.terminal().adversaries()
    }

    pub fn n_out(&self) -> usize {
        self.publication
            .source()
            .prefix()
            .root()
            .model()
            .net()
            .n_out
    }

    pub fn min_margin(&self) -> f64 {
        self.publication.token().min_margin()
    }

    pub fn policy(&self) -> RootPrecisionRuntimePolicy {
        self.publication.receipt().policy
    }

    pub fn validate_objective(
        &self,
        target: usize,
        ordered_classes: &[usize],
        requested: Option<Instant>,
    ) -> Result<()> {
        self.publication
            .validate_objective(target, ordered_classes, requested)
    }

    /// Guards may only downgrade the existing proof. The same complete
    /// publication remains owned on success, and no `UnsatToken` is returned.
    pub fn ratified(self, guards: &GuardReport, requested: Option<Instant>) -> Result<Self> {
        let at = self.deadline(requested)?;
        Ok(Self {
            publication: self.publication.ratified(guards, Some(at))?,
            session: self.session.restricted(Some(at))?,
            attempt: self.attempt,
        })
    }
}

#[cfg(test)]
#[path = "cli_proof_ordinary_tests.rs"]
mod ordinary_tests;

#[cfg(test)]
#[path = "cli_proof_tree_tests.rs"]
mod tree_tests;
