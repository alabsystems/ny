// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original-file authority for the retained prepared-root proof.
//!
//! The private constructor reads each file once, authenticates the complete
//! property directly from that snapshot, and moves the declaration extracted
//! from those same model bytes into the real prepared producer. Neither a
//! caller target, a supplied proof, dimensions, nor a JSON record can attach
//! unrelated source files to an already authorized proof. The publisher must
//! retain this owner and call `checkpoint` immediately before committing its
//! result. Existing loaders and engines outside this entry are not included
//! in its resource census.

use std::{
    fmt, fs::File, io::Read, mem::size_of, ops::Deref, path::Path, sync::Arc, time::Instant,
};

use anyhow::{anyhow, bail, ensure, Result};
use ny_onnx::vnnlib::{
    parse_vnnlib_with_certified_input_box, CertifiedInputBox, OutputConstraint, VnnLibSpec,
};
use ny_propagate::{
    certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest},
    margin_row::{
        bab::{
            MarginRowBab, PreparedTreeOpen, PreparedTreeOpenReason, PreparedTreeOptions,
            PreparedTreeOutcome, PreparedTreeReceipt,
        },
        PreparedRootAttemptReceipt, PreparedRootConsumerOptions, PreparedRootOpen,
        PreparedRootOutcome, PreparedRootProof, PreparedRootProofOptions, PreparedRootSession,
        RootPrecisionRuntimePolicy,
    },
};
use sha2::{Digest, Sha256};

const META: usize = 8192;
const POLL: usize = 4096;
// This source parser deliberately bounds numeric work before entering the
// ordinary/exact parsers. Their broader VNN-LIB grammar remains available to
// the existing fallback; this owner cannot silently normalize that grammar.
const MAX_LITERAL_BYTES: usize = 64;
const MAX_LITERAL_SCALE: i32 = 324;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain the bounded direct-file issuer grammar limit until its route is admitted."
    )
)]
const MAX_PROPERTY_BYTES: usize = 64 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    Le,
    Ge,
    Lt,
    Gt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Atom {
    relation: Relation,
    left: usize,
    right: usize,
}

/// Raw-source authenticated complete target robustness objective. All arrays
/// are private and source ordered; there is no constructor from a `VnnLibSpec`.
pub(super) struct OriginalObjective {
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    pub(super) target: usize,
    pub(super) classes: Vec<usize>,
    pub(super) atoms: Vec<Atom>,
}

pub(super) struct OriginalProperty {
    pub(super) source: String,
    pub(super) spec: VnnLibSpec,
    pub(super) input: OriginalInputEnclosure,
    pub(super) objective: OriginalObjective,
    pub(super) sha256: [u8; 32],
}

/// A private paid copy of the source certificate's exact outward endpoints.
/// Keeping only these planes makes every retained capacity visible here.
pub(super) struct OriginalInputEnclosure {
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
}
impl OriginalInputEnclosure {
    fn len(&self) -> usize {
        self.lower.len()
    }
    fn lower(&self) -> &[f64] {
        &self.lower
    }
    fn upper(&self) -> &[f64] {
        &self.upper
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain paid input-certificate copying for the bounded direct-file issuer."
        )
    )]
    fn from_certificate(
        input: &CertifiedInputBox,
        budget: &WorkBudget,
        at: Instant,
    ) -> Result<Self> {
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        lower.try_reserve_exact(input.len())?;
        upper.try_reserve_exact(input.len())?;
        ensure!(
            lower.capacity() <= input.len() && upper.capacity() <= input.len(),
            "original endpoint capacity exceeds source admission"
        );
        for (i, (&lo, &hi)) in input.lower().iter().zip(input.upper()).enumerate() {
            poll(budget, at, i)?;
            lower.push(lo);
            upper.push(hi);
        }
        Ok(Self { lower, upper })
    }
}

// Storage precedes its reservations, so every actual source/model/session
// payload drops before its last-owner claims. No weak aliases are exported.
struct OriginalPreparedFileInner {
    model_bytes: Vec<u8>,
    model_sha256: [u8; 32],
    property: OriginalProperty,
    session: PreparedRootSession,
    source_digest: [u8; 32],
    budget: WorkBudget,
    deadline: Instant,
    resident: PersistentReservation,
    // The staged issuer retains its independently paid property payload;
    // parser and decoder scratch have ceased before proof construction.
    property_resident: Option<PersistentReservation>,
    model_resident: Option<PersistentReservation>,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the stored staged-allocation receipt and original source-owner layout."
        )
    )]
    allocation: Option<super::original_property_census::OriginalPropertyAllocationReceipt>,
    // The started caller's phase/resources owner outlives every retained
    // source claim. It owns no source, so this shared handoff has no cycle.
    resources: Option<Arc<super::prepared_root_route::PreparedRouteResources>>,
}

impl OriginalPreparedFileInner {
    fn checkpoint_resources(&self, at: Instant) -> Result<()> {
        if let Some(resources) = &self.resources {
            ensure!(
                resources.budget().shares_ledger(&self.budget),
                "prepared original caller resource ledger changed"
            );
            resources.checkpoint(Some(at))?;
        }
        Ok(())
    }
}

struct PreparedPublicationInner {
    proof: PreparedRootProof,
    original: Arc<OriginalPreparedFileInner>,
    _resident: PersistentReservation,
}

// Private control compatibility only. This implementation is not a public
// source/session getter and cannot attach a supplied proof to file authority.
impl Deref for PreparedPublicationInner {
    type Target = OriginalPreparedFileInner;
    fn deref(&self) -> &Self::Target {
        &self.original
    }
}

/// One authenticated original-file session can keep useful completed work
/// after an unsuccessful root close. Targets and classes always come from its
/// held raw property, never from caller labels or imported numerical arrays.
#[derive(Clone)]
pub(crate) struct OriginalPreparedRootSession(Arc<OriginalPreparedFileInner>, Instant);

struct OriginalPreparedOpenInner {
    open: PreparedRootOpen,
    original: OriginalPreparedRootSession,
    _resident: PersistentReservation,
}

#[derive(Clone)]
pub(crate) struct OriginalPreparedRootOpen(Arc<OriginalPreparedOpenInner>, Instant);

pub(crate) enum OriginalPreparedRootOutcome {
    Proven(OriginalPreparedRootPublication),
    Open(OriginalPreparedRootOpen),
}

struct OriginalPreparedTreeOpenInner {
    tree: PreparedTreeOpen,
    original: OriginalPreparedRootOpen,
    // Actual frontier/evaluation storage drops before this paid Arc header.
    _resident: PersistentReservation,
}

/// An immutable alias retains the actual unfinished tree, not a reconstruction
/// recipe. Resumption consumes its sole owner and preserves that frontier's
/// expansion limit, source, existing paid work and producer deadline.
#[derive(Clone)]
pub(crate) struct OriginalPreparedTreeOpen(Arc<OriginalPreparedTreeOpenInner>, Instant);

pub(crate) enum OriginalPreparedTreeOutcome {
    Proven(OriginalPreparedRootPublication),
    Open(OriginalPreparedTreeOpen),
}

/// Actual original model/property bytes, the actual extracted declaration,
/// and the same authorized proof remain owned through every staged writer.
#[derive(Clone)]
pub(crate) struct OriginalPreparedRootPublication(Arc<PreparedPublicationInner>, Instant);

impl fmt::Debug for OriginalPreparedRootPublication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedRootPublication")
            .field("target", &self.target())
            .field("classes", &self.classes())
            .field("min_margin", &self.min_margin())
            .finish_non_exhaustive()
    }
}

impl OriginalPreparedRootSession {
    /// Read and authenticate each original once. Ceased parser and decoder
    /// scratch releases before the one genuine compiled model is constructed
    /// into this reusable, original-objective-bound session.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the budget-bound staged issuer for reusable original-session adapters."
        )
    )]
    pub(crate) fn prepare_original_files_staged(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
    ) -> Result<Self> {
        Self::prepare_staged_inner(
            model_path,
            property_path,
            policy,
            options,
            budget,
            requested,
            None,
        )
    }

    /// Use the already started caller's resources for the same staged source
    /// preparation. Every Open/Proven alias retains this guard; this entry
    /// creates neither a fresh budget nor a new source/objective declaration.
    pub(crate) fn prepare_original_files_in_caller(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        resources: Arc<super::prepared_root_route::PreparedRouteResources>,
        requested: Instant,
    ) -> Result<Self> {
        let at = resources.deadline(Some(requested))?;
        resources.checkpoint(Some(at))?;
        let owner = Self::prepare_staged_guarded_inner(
            model_path,
            property_path,
            policy,
            options,
            resources.budget(),
            at,
            Some(resources.clone()),
            None,
        )?;
        resources.checkpoint(Some(owner.deadline()))?;
        Ok(owner)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the staged construction hook shared by original-session issuers."
        )
    )]
    fn prepare_staged_inner(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
        before_construction: Option<&mut dyn FnMut(&WorkBudget)>,
    ) -> Result<Self> {
        Self::prepare_staged_guarded_inner(
            model_path,
            property_path,
            policy,
            options,
            budget,
            requested,
            None,
            before_construction,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_staged_guarded_inner(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
        resources: Option<Arc<super::prepared_root_route::PreparedRouteResources>>,
        mut before_construction: Option<&mut dyn FnMut(&WorkBudget)>,
    ) -> Result<Self> {
        use super::original_property_census::{
            parse_staged_original_property, MAX_STAGED_PROPERTY_BYTES,
        };
        checkpoint(budget, requested)?;
        ensure!(
            budget.memory_authority().is_some(),
            "staged prepared CLI source requires original caller memory authority"
        );
        let mut model_file = File::open(model_path)?;
        let mut property_file = File::open(property_path)?;
        let model_len = source_len(&model_file)?;
        let property_len = source_len(&property_file)?;
        if let Some(resources) = &resources {
            ensure!(
                resources.budget().shares_ledger(budget),
                "prepared original caller resource ledger changed"
            );
            resources.checkpoint(Some(requested))?;
            resources.validate_source_lengths(model_len, property_len)?;
        }
        ensure!(
            property_len <= MAX_STAGED_PROPERTY_BYTES,
            "staged prepared property exceeds one MiB grammar surface"
        );
        let snapshot_bytes = checked_add(property_len, source_owner_bytes()?)?;
        let mut snapshot = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: META,
                persistent_bytes: snapshot_bytes,
                work: charged(property_len, 64)?,
                ..WorkRequest::default()
            },
        )?;
        let mut at = snapshot.deadline().map_or(requested, |d| d.min(requested));
        let source =
            String::from_utf8(read_snapshot(&mut property_file, property_len, budget, at)?)
                .map_err(|_| anyhow!("staged original property snapshot must be UTF-8"))?;
        ensure!(
            source.capacity() <= property_len,
            "actual staged source capacities exceed snapshot admission"
        );
        snapshot.mark_materialized()?;
        let resident = snapshot.commit_persistent()?;
        let property = parse_staged_original_property(source, &resident, budget, at)?;
        let (property, resident, property_resident, allocation, property_at) =
            property.into_parts();
        at = at.min(property_at);
        if let Some(resources) = &resources {
            resources.checkpoint(Some(at))?;
            resources.validate_objective(property.objective.inputs, property.objective.outputs)?;
        }
        // Reading the model starts only after the complete property parser
        // has released its workspace. The file descriptor was opened once;
        // the later snapshot still has the original clock and exact-byte
        // binding. Its separate claim is acquired BEFORE its Vec allocation.
        let mut model_snapshot = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: META,
                persistent_bytes: checked_add(model_len, META)?,
                work: charged(model_len, 64)?,
                ..WorkRequest::default()
            },
        )?;
        at = model_snapshot.deadline().map_or(at, |d| d.min(at));
        let model_bytes = read_snapshot(&mut model_file, model_len, budget, at)?;
        ensure!(
            model_bytes.capacity() <= model_len,
            "actual staged model snapshot capacity exceeds admission"
        );
        model_snapshot.mark_materialized()?;
        let model_resident = model_snapshot.commit_persistent()?;
        // The property parser claim has already shed all ceased scratch. The
        // source compiler uses exactly these retained bytes and transfers its
        // real model before releasing its separately paid decoder workspace.
        let staged = super::original_source_census::compile_staged_original_source(
            &model_bytes,
            budget,
            at,
        )?;
        ensure!(
            staged.compiled().n_in() == property.objective.inputs
                && staged.compiled().n_out() == property.objective.outputs,
            "staged original model dimensions disagree with complete property"
        );
        let model_sha256 = *staged.source_sha256();
        at = staged.compiled().deadline(Some(at))?.unwrap_or(at).min(at);
        let compiled = staged.into_compilation(budget, Some(at))?;
        if let Some(observe) = before_construction.as_mut() {
            observe(budget);
        }
        checkpoint(budget, at)?;
        let session = compiled.prepare_declared_domain(
            property.input.lower(),
            property.input.upper(),
            policy,
            options,
            budget,
            Some(at),
        )?;
        ensure!(
            session.budget().shares_ledger(budget) && session.n_out() == property.objective.outputs,
            "staged original prepared source changed objective or ledger"
        );
        at = at.min(session.deadline(Some(at))?);
        let source_digest = *session.source_digest();
        let owner = Self(
            Arc::new(OriginalPreparedFileInner {
                model_bytes,
                model_sha256,
                property,
                session,
                source_digest,
                budget: budget.clone(),
                deadline: at,
                resident,
                property_resident: Some(property_resident),
                model_resident: Some(model_resident),
                allocation: Some(allocation),
                resources,
            }),
            at,
        );
        owner.checkpoint(None)?;
        Ok(owner)
    }

    pub(crate) fn budget(&self) -> &WorkBudget {
        &self.0.budget
    }
    pub(crate) fn deadline(&self) -> Instant {
        self.1.min(self.0.deadline)
    }
    pub(crate) fn target(&self) -> usize {
        self.0.property.objective.target
    }
    pub(crate) fn classes(&self) -> &[usize] {
        &self.0.property.objective.classes
    }
    /// Historical observations only; this grants no source/writer authority.
    pub(crate) fn row_storage_receipt(
        &self,
    ) -> ny_propagate::margin_row::PreparedRowStorageReceipt {
        self.0.session.row_storage_receipt()
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain exact model-byte binding for retained session consumers."
        )
    )]
    pub(crate) fn model_sha256(&self) -> &[u8; 32] {
        &self.0.model_sha256
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain exact property-byte binding for retained session consumers."
        )
    )]
    pub(crate) fn property_sha256(&self) -> &[u8; 32] {
        &self.0.property.sha256
    }
    pub(crate) fn restricted(mut self, requested: Option<Instant>) -> Result<Self> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        self.1 = at;
        Ok(self)
    }
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.0.checkpoint_resources(at)?;
        checkpoint(self.budget(), at)?;
        let mut lease = self.budget().lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: charged(
                    checked_add(self.0.property.input.len(), self.classes().len())?,
                    32,
                )?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        self.0.session.checkpoint(Some(at))?;
        ensure!(
            self.0.resident.shares_ledger(self.budget())
                && self
                    .0
                    .property_resident
                    .as_ref()
                    .is_none_or(|r| r.shares_ledger(self.budget()))
                && self
                    .0
                    .model_resident
                    .as_ref()
                    .is_none_or(|r| r.shares_ledger(self.budget()))
                && self.0.session.budget().shares_ledger(self.budget())
                && self.0.session.source_digest() == &self.0.source_digest
                && self.0.session.n_out() == self.0.property.objective.outputs,
            "prepared original reusable source authority changed"
        );
        let lower = self.0.session.declared_lower();
        let upper = self.0.session.declared_upper();
        ensure!(
            lower.len() == self.0.property.input.len() && upper.len() == lower.len(),
            "prepared original session input dimensions changed"
        );
        for (i, ((&lo, &hi), (&source_lo, &source_hi))) in lower
            .iter()
            .zip(upper)
            .zip(
                self.0
                    .property
                    .input
                    .lower()
                    .iter()
                    .zip(self.0.property.input.upper()),
            )
            .enumerate()
        {
            poll(self.budget(), at, i)?;
            ensure!(
                lo.to_bits() == source_lo.to_bits() && hi.to_bits() == source_hi.to_bits(),
                "prepared original session lost exact source-certified input enclosure"
            );
        }
        lease.complete()?;
        self.0.checkpoint_resources(at)?;
        checkpoint(self.budget(), at)
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain original-file authentication before session reuse or export."
        )
    )]
    pub(crate) fn validate_asset_hashes(
        &self,
        model: [u8; 32],
        property: [u8; 32],
        requested: Option<Instant>,
    ) -> Result<()> {
        ensure!(
            model == *self.model_sha256() && property == *self.property_sha256(),
            "prepared session belongs to different original model/property bytes"
        );
        self.checkpoint(requested)
    }
    pub(crate) fn try_prove_root(
        &self,
        requested: Option<Instant>,
    ) -> Result<OriginalPreparedRootOutcome> {
        self.try_prove_root_with_consumers(PreparedRootConsumerOptions::default(), requested)
    }
    pub(crate) fn try_prove_root_with_consumers(
        &self,
        consumers: PreparedRootConsumerOptions,
        requested: Option<Instant>,
    ) -> Result<OriginalPreparedRootOutcome> {
        let mut at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        // The actual outcome header is prepaid before any new attempt. Its
        // lease overlaps the retained original source and all numerical work;
        // it is not a credit for those separately admitted producers.
        let mut outcome_lease = outcome_owner_lease(self.budget(), at)?;
        at = outcome_lease.deadline().map_or(at, |d| d.min(at));
        let outcome = self.0.session.try_prove_root_with_consumers(
            self.target(),
            self.classes(),
            consumers,
            Some(at),
        )?;
        match outcome {
            PreparedRootOutcome::Proven(proof) => {
                at = at.min(proof.deadline(Some(at))?);
                let agreement = streamed_spec_agreement(
                    &self.0.property.spec,
                    self.target(),
                    self.classes(),
                    self.budget(),
                    at,
                )?;
                at = at.min(agreement.deadline);
                let proof = proof.ratified(&agreement.report, Some(at))?;
                drop(agreement);
                let original = self.clone().restricted(Some(at))?;
                original.checkpoint(Some(at))?;
                outcome_lease.mark_materialized()?;
                let resident = outcome_lease.commit_persistent()?;
                let owner = OriginalPreparedRootPublication(
                    Arc::new(PreparedPublicationInner {
                        proof,
                        original: original.0,
                        _resident: resident,
                    }),
                    at,
                );
                owner.checkpoint(Some(at))?;
                Ok(OriginalPreparedRootOutcome::Proven(owner))
            }
            PreparedRootOutcome::Open(open) => {
                at = at.min(open.deadline(Some(at))?);
                let original = self.clone().restricted(Some(at))?;
                open.checkpoint(Some(at))?;
                outcome_lease.mark_materialized()?;
                let resident = outcome_lease.commit_persistent()?;
                let owner = OriginalPreparedRootOpen(
                    Arc::new(OriginalPreparedOpenInner {
                        open,
                        original,
                        _resident: resident,
                    }),
                    at,
                );
                owner.checkpoint(Some(at))?;
                Ok(OriginalPreparedRootOutcome::Open(owner))
            }
        }
    }
}

impl OriginalPreparedRootOpen {
    pub(crate) fn deadline(&self) -> Instant {
        self.1.min(self.0.original.deadline())
    }
    pub(crate) fn budget(&self) -> &WorkBudget {
        self.0.original.budget()
    }
    pub(crate) fn target(&self) -> usize {
        self.0.original.target()
    }
    pub(crate) fn classes(&self) -> &[usize] {
        self.0.original.classes()
    }
    pub(crate) fn attempt_receipt(&self) -> PreparedRootAttemptReceipt {
        self.0.open.receipt()
    }
    /// A restricted original-file alias, not a bare conditional session.
    pub(crate) fn session(&self) -> Result<OriginalPreparedRootSession> {
        self.0.original.clone().restricted(Some(self.deadline()))
    }
    pub(crate) fn restricted(mut self, requested: Option<Instant>) -> Result<Self> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        self.1 = at;
        Ok(self)
    }
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.0.original.checkpoint(Some(at))?;
        self.0.open.checkpoint(Some(at))?;
        ensure!(
            self.0._resident.shares_ledger(self.budget())
                && self.0.open.target() == self.target()
                && self.0.open.classes().len() == self.classes().len()
                && self.0.open.session().budget().shares_ledger(self.budget())
                && self
                    .0
                    .open
                    .session()
                    .same_source(&self.0.original.0.session)
                && self.0.open.session().source_digest()
                    == self.0.original.0.session.source_digest()
                && self.0.open.session().compiled_digest()
                    == self.0.original.0.session.compiled_digest(),
            "prepared open result lost its authentic original scope"
        );
        let mut lease = self.budget().lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: charged(self.classes().len(), 32)?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        for (i, (&actual, &expected)) in
            self.0.open.classes().iter().zip(self.classes()).enumerate()
        {
            poll(self.budget(), at, i)?;
            ensure!(
                actual == expected,
                "prepared open original class order changed"
            );
        }
        lease.complete()?;
        checkpoint(self.budget(), at)
    }

    /// Continue the actual open root on its original file-bound source. The
    /// caller selects scheduling limits only; it cannot supply an objective,
    /// a numerical certificate, or a separately compiled model.
    pub(crate) fn try_tree(
        &self,
        options: PreparedTreeOptions,
        requested: Option<Instant>,
    ) -> Result<OriginalPreparedTreeOutcome> {
        let mut at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        let outcome_lease = tree_outcome_owner_lease(self.budget(), at)?;
        at = outcome_lease.deadline().map_or(at, |d| d.min(at));
        let outcome = MarginRowBab::run_prepared(self.0.open.clone(), options, Some(at))?;
        finish_original_tree(
            self.clone().restricted(Some(at))?,
            outcome,
            outcome_lease,
            at,
        )
    }
}

impl OriginalPreparedTreeOpen {
    pub(crate) fn deadline(&self) -> Instant {
        self.1
            .min(self.0.original.deadline())
            .min(self.0.tree.deadline())
    }
    pub(crate) fn budget(&self) -> &WorkBudget {
        self.0.original.budget()
    }
    pub(crate) fn target(&self) -> usize {
        self.0.original.target()
    }
    pub(crate) fn classes(&self) -> &[usize] {
        self.0.original.classes()
    }
    pub(crate) fn receipt(&self) -> PreparedTreeReceipt {
        self.0.tree.receipt()
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain stop-reason reporting for the original unfinished-tree adapter."
        )
    )]
    pub(crate) fn reason(&self) -> PreparedTreeOpenReason {
        self.0.tree.reason()
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain frontier reporting for the original unfinished-tree adapter."
        )
    )]
    pub(crate) fn open_domains(&self) -> usize {
        self.0.tree.open_domains()
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain source-bound session reuse from the unfinished-tree owner."
        )
    )]
    pub(crate) fn session(&self) -> Result<OriginalPreparedRootSession> {
        self.checkpoint(None)?;
        self.0.original.session()?.restricted(Some(self.deadline()))
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain deadline narrowing on immutable unfinished-tree aliases."
        )
    )]
    pub(crate) fn restricted(mut self, requested: Option<Instant>) -> Result<Self> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        self.1 = at;
        Ok(self)
    }
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.0.original.checkpoint(Some(at))?;
        self.0.tree.checkpoint(Some(at))?;
        let root = self.0.tree.root_open();
        ensure!(
            self.0._resident.shares_ledger(self.budget())
                && self.0._resident.bytes() >= tree_outcome_owner_bytes()?
                && root
                    .session()
                    .same_source(&self.0.original.0.original.0.session)
                && root.session().budget().shares_ledger(self.budget())
                && self.0.tree.target() == self.target()
                && self.0.tree.classes().len() == self.classes().len(),
            "prepared tree open lost its exact original file session"
        );
        let mut scan = self.budget().lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work: charged(self.classes().len(), 32)?,
                ..WorkRequest::default()
            },
        )?;
        let at = scan.deadline().map_or(at, |d| d.min(at));
        for (i, (&actual, &expected)) in
            self.0.tree.classes().iter().zip(self.classes()).enumerate()
        {
            poll(self.budget(), at, i)?;
            ensure!(
                actual == expected,
                "prepared tree original class order changed"
            );
        }
        scan.complete()?;
        checkpoint(self.budget(), at)
    }

    /// A shared alias visibly refuses resumption; it never copies a frontier,
    /// recompiles the source or grants fresh expansion/work allowance. The
    /// surviving aliases retain the actual work. This consuming entry uses
    /// the same failure semantics as the library's consuming resume method.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain consuming resumption of the actual paid frontier without recompilation."
        )
    )]
    pub(crate) fn resume(self, requested: Option<Instant>) -> Result<OriginalPreparedTreeOutcome> {
        let mut at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        ensure!(
            Arc::strong_count(&self.0) == 1,
            "prepared tree resumption requires its sole immutable owner"
        );
        // Old and replacement headers overlap until this consuming transfer
        // finishes. Existing tree payload claims move; they are not duplicated.
        let outcome_lease = tree_outcome_owner_lease(self.budget(), at)?;
        at = outcome_lease.deadline().map_or(at, |d| d.min(at));
        let inner = Arc::try_unwrap(self.0)
            .map_err(|_| anyhow!("prepared tree resumption still has a live alias"))?;
        let OriginalPreparedTreeOpenInner {
            tree,
            original,
            _resident: _old_header,
        } = inner;
        let outcome = tree.restricted(Some(at))?.resume(Some(at))?;
        finish_original_tree(original.restricted(Some(at))?, outcome, outcome_lease, at)
    }
}

fn finish_original_tree(
    original: OriginalPreparedRootOpen,
    outcome: PreparedTreeOutcome,
    mut outcome_lease: WorkLease,
    mut at: Instant,
) -> Result<OriginalPreparedTreeOutcome> {
    at = at.min(original.deadline());
    original.checkpoint(Some(at))?;
    match outcome {
        PreparedTreeOutcome::Covered(certificate) => {
            at = at.min(certificate.deadline());
            let proof = PreparedRootProof::from_prepared_tree(certificate, Some(at))?;
            let owner = finish_original_tree_proof(original, proof, outcome_lease, at)?;
            Ok(OriginalPreparedTreeOutcome::Proven(owner))
        }
        PreparedTreeOutcome::Open(tree) => {
            at = at.min(tree.deadline());
            tree.checkpoint(Some(at))?;
            ensure!(
                tree.root_open()
                    .session()
                    .same_source(&original.0.original.0.session),
                "tree open belongs to a different original file session"
            );
            let tree = tree.restricted(Some(at))?;
            let original = original.restricted(Some(at))?;
            outcome_lease.mark_materialized()?;
            let resident = outcome_lease.commit_persistent()?;
            let owner = OriginalPreparedTreeOpen(
                Arc::new(OriginalPreparedTreeOpenInner {
                    tree,
                    original,
                    _resident: resident,
                }),
                at,
            );
            owner.checkpoint(Some(at))?;
            Ok(OriginalPreparedTreeOutcome::Open(owner))
        }
    }
}

/// Private completion of the real tree factory, never a public attachment
/// API. Equal file hashes, dimensions, endpoints and ledger caps do not replace
/// exact producer identity. The unchanged original G3 ratification still runs.
fn finish_original_tree_proof(
    original: OriginalPreparedRootOpen,
    proof: PreparedRootProof,
    mut outcome_lease: WorkLease,
    mut at: Instant,
) -> Result<OriginalPreparedRootPublication> {
    original.checkpoint(Some(at))?;
    ensure!(
        proof.tree_receipt().is_some()
            && proof.session().same_source(&original.0.original.0.session)
            && proof.budget().shares_ledger(original.budget()),
        "tree proof belongs to a different original file session"
    );
    at = at.min(original.deadline()).min(proof.deadline(Some(at))?);
    proof.validate_objective(original.target(), original.classes(), Some(at))?;
    let agreement = streamed_spec_agreement(
        &original.0.original.0.property.spec,
        original.target(),
        original.classes(),
        original.budget(),
        at,
    )?;
    at = at.min(agreement.deadline);
    let proof = proof.ratified(&agreement.report, Some(at))?;
    drop(agreement);
    let source = original.session()?.restricted(Some(at))?;
    source.checkpoint(Some(at))?;
    outcome_lease.mark_materialized()?;
    let resident = outcome_lease.commit_persistent()?;
    let owner = OriginalPreparedRootPublication(
        Arc::new(PreparedPublicationInner {
            proof,
            original: source.0,
            _resident: resident,
        }),
        at,
    );
    owner.checkpoint(Some(at))?;
    Ok(owner)
}

fn tree_outcome_owner_lease(budget: &WorkBudget, at: Instant) -> Result<WorkLease> {
    checkpoint(budget, at)?;
    Ok(budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            persistent_bytes: tree_outcome_owner_bytes()?,
            ..WorkRequest::default()
        },
    )?)
}

fn tree_outcome_owner_bytes() -> Result<usize> {
    let header =
        size_of::<PreparedPublicationInner>().max(size_of::<OriginalPreparedTreeOpenInner>());
    let alignment =
        align_of::<PreparedPublicationInner>().max(align_of::<OriginalPreparedTreeOpenInner>());
    checked_add(
        checked_add(header, checked_mul(2, size_of::<usize>())?)?,
        checked_add(alignment.saturating_sub(1), META)?,
    )
}

fn outcome_owner_lease(budget: &WorkBudget, at: Instant) -> Result<WorkLease> {
    checkpoint(budget, at)?;
    Ok(budget.lease_for(
        LeaseClass::Fallback,
        WorkRequest {
            persistent_bytes: outcome_owner_bytes()?,
            ..WorkRequest::default()
        },
    )?)
}

fn outcome_owner_bytes() -> Result<usize> {
    let header = size_of::<PreparedPublicationInner>().max(size_of::<OriginalPreparedOpenInner>());
    let alignment =
        align_of::<PreparedPublicationInner>().max(align_of::<OriginalPreparedOpenInner>());
    checked_add(
        checked_add(header, checked_mul(2, size_of::<usize>())?)?,
        checked_add(alignment.saturating_sub(1), META)?,
    )
}

fn source_owner_bytes() -> Result<usize> {
    checked_add(
        checked_add(
            size_of::<OriginalPreparedFileInner>(),
            checked_mul(2, size_of::<usize>())?,
        )?,
        checked_add(
            align_of::<OriginalPreparedFileInner>().saturating_sub(1),
            META,
        )?,
    )
}

impl OriginalPreparedRootPublication {
    /// Staged original-file issuer. Source snapshots, parsed property and
    /// genuine compiled model retain separate same-ledger claims; completed
    /// parser/decoder workspaces release before root construction. No large
    /// source is admitted merely because it fits the grammar byte limit.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the staged original-file issuer until its standalone route is admitted."
        )
    )]
    pub(crate) fn prove_original_files_staged(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
    ) -> Result<Self> {
        Self::prove_staged_inner(
            model_path,
            property_path,
            policy,
            options,
            budget,
            requested,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain staged issuance under the original construction hook and ledger."
        )
    )]
    pub(super) fn prove_staged_inner(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
        before_construction: Option<&mut dyn FnMut(&WorkBudget)>,
    ) -> Result<Self> {
        let session = OriginalPreparedRootSession::prepare_staged_inner(
            model_path,
            property_path,
            policy,
            options,
            budget,
            requested,
            before_construction,
        )?;
        match session.try_prove_root(Some(requested))? {
            OriginalPreparedRootOutcome::Proven(owner) => Ok(owner),
            OriginalPreparedRootOutcome::Open(_) => {
                bail!("prepared terminal and ordinary root leave original classes open")
            }
        }
    }

    /// Only this complete source-to-producer transaction can construct the
    /// CLI authority. `budget` is the caller's already started instance ledger.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the bounded direct-file source-to-proof transaction."
        )
    )]
    pub(crate) fn prove_original_files(
        model_path: &Path,
        property_path: &Path,
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Instant,
    ) -> Result<Self> {
        checkpoint(budget, requested)?;
        ensure!(
            budget.memory_authority().is_some(),
            "prepared CLI source requires original caller memory authority"
        );
        let mut model_file = File::open(model_path)?;
        let mut property_file = File::open(property_path)?;
        let model_len = source_len(&model_file)?;
        let property_len = source_len(&property_file)?;
        ensure!(
            property_len <= MAX_PROPERTY_BYTES,
            "prepared CLI direct property exceeds bounded parser surface"
        );
        // Price source decoding/AST/exact-decimal work BEFORE reading or
        // decoding. The restricted syntax below has flat atoms, <=64-byte
        // decimals and scale <=324. The deliberately conservative 512 bytes
        // per source byte covers decoded protobuf repeated-field headers,
        // temporary parameter vectors, ordinary/exact token/AST storage and
        // bounded BigInt endpoints; the owned payload has a separate claim.
        // Large original sources may refuse this entry and retain fallback.
        let scratch = checked_add(
            checked_mul(checked_add(model_len, property_len)?, 512)?,
            META,
        )?;
        let persistent = checked_add(
            checked_add(model_len, checked_mul(property_len, 256)?)?,
            source_owner_bytes()?,
        )?;
        let mut lease = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: persistent,
                work: charged(checked_add(model_len, property_len)?, 8192)?,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(requested, |d| d.min(requested));
        checkpoint(budget, at)?;
        let model_bytes = read_snapshot(&mut model_file, model_len, budget, at)?;
        let property_bytes = read_snapshot(&mut property_file, property_len, budget, at)?;
        let source = String::from_utf8(property_bytes)?;
        let objective = authenticate_raw_objective(&source, budget, at)?;
        checkpoint(budget, at)?;
        let (spec, input) = parse_vnnlib_with_certified_input_box(&source)?;
        checkpoint(budget, at)?;
        cross_check(&objective, &spec, &input)?;
        let retained_input = OriginalInputEnclosure::from_certificate(&input, budget, at)?;
        drop(input);
        let model_sha256 = hash(&model_bytes, budget, at)?;
        let property_sha256 = hash(source.as_bytes(), budget, at)?;
        super::prepared_root_source::authenticate_original_twin_surface(&model_bytes, budget, at)?;
        // No pathname is re-opened: this is the unchanged actual extractor
        // over the very bytes retained and hashed above.
        let (twin, names) =
            super::margin_row_bab::extract_original_twin_spec_named_bytes(&model_bytes, budget, at)
                .ok_or_else(|| {
                    anyhow!("original model is outside the strict twin extraction family")
                })?;
        checkpoint(budget, at)?;
        ensure!(
            twin.n_in == objective.inputs,
            "original model input and source property disagree"
        );
        // Named metadata is not consumed by this producer. Drop it while the
        // complete decoding lease remains live; do not attach it to a token.
        drop(names);
        let proof = PreparedRootProof::prove_declared_twin(
            twin,
            retained_input.lower(),
            retained_input.upper(),
            objective.target,
            &objective.classes,
            policy,
            options,
            budget,
            Some(at),
        )?;
        ensure!(
            proof.budget().shares_ledger(budget) && proof.n_out() == objective.outputs,
            "original prepared source changed model/objective or ledger"
        );
        // G3 uses the immutable production parse. Source authentication above
        // already established semantics; this existing guard can only refuse.
        let g3_cells = checked_add(
            checked_mul(
                checked_add(checked_mul(objective.outputs, 2)?, 1)?,
                objective.outputs,
            )?,
            objective.inputs,
        )?;
        ensure!(
            checked_mul(g3_cells, 8)? <= scratch,
            "original G3 scratch exceeds admitted source envelope"
        );
        let (guards, _fragment) = super::margin_row_bab::margin_row_spec_agreement_owned(
            &spec,
            objective.target,
            &objective.classes,
        );
        checkpoint(budget, at)?;
        let proof = proof.ratified(&guards, Some(at))?;
        drop(guards);
        drop(_fragment);
        let at = at.min(proof.deadline(Some(at))?);
        let source_digest = *proof.source_digest();
        let property = OriginalProperty {
            source,
            spec,
            input: retained_input,
            objective,
            sha256: property_sha256,
        };
        let actual = visible_bytes(&model_bytes, &property, budget, at)?;
        ensure!(
            actual <= persistent,
            "prepared original source allocator capacity exceeds admitted ownership"
        );
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        let session = proof.session().restricted(Some(at))?;
        let original = Arc::new(OriginalPreparedFileInner {
            model_bytes,
            model_sha256,
            property,
            session,
            source_digest,
            budget: budget.clone(),
            deadline: at,
            resident,
            property_resident: None,
            model_resident: None,
            allocation: None,
            resources: None,
        });
        let mut outcome_lease = outcome_owner_lease(budget, at)?;
        let at = outcome_lease.deadline().map_or(at, |d| d.min(at));
        outcome_lease.mark_materialized()?;
        let outcome_resident = outcome_lease.commit_persistent()?;
        let owner = Self(
            Arc::new(PreparedPublicationInner {
                proof,
                original,
                _resident: outcome_resident,
            }),
            at,
        );
        owner.checkpoint(None)?;
        Ok(owner)
    }

    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let inherited = self.1.min(self.0.deadline);
        let at = requested.map_or(inherited, |d| d.min(inherited));
        self.0.checkpoint_resources(at)?;
        checkpoint(&self.0.budget, at)?;
        let class_work = checked_mul(self.classes().len().max(self.0.proof.classes().len()), 2)?;
        let work = charged(checked_add(self.0.property.input.len(), class_work)?, 32)?;
        let mut lease = self.0.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                work,
                ..WorkRequest::default()
            },
        )?;
        let at = lease.deadline().map_or(at, |d| d.min(at));
        self.0.proof.checkpoint(Some(at))?;
        ensure!(
            self.0._resident.shares_ledger(&self.0.budget)
                && self.0.resident.shares_ledger(&self.0.budget)
                && self
                    .0
                    .property_resident
                    .as_ref()
                    .is_none_or(|r| r.shares_ledger(&self.0.budget))
                && self
                    .0
                    .model_resident
                    .as_ref()
                    .is_none_or(|r| r.shares_ledger(&self.0.budget))
                && self.0.proof.budget().shares_ledger(&self.0.budget)
                && self.0.proof.session().same_source(&self.0.session)
                && self.0.proof.source_digest() == &self.0.source_digest
                && self.0.proof.compiled_digest() == self.0.session.compiled_digest()
                && self.0.proof.target() == self.target()
                && self.0.proof.classes().len() == self.classes().len()
                && self.0.proof.n_out() == self.0.property.objective.outputs
                && self.0.proof.min_margin().is_finite()
                && self.0.proof.min_margin() > 0.0,
            "prepared original source/proof publication authority changed"
        );
        for (i, (&proof_class, &original_class)) in self
            .0
            .proof
            .classes()
            .iter()
            .zip(self.classes())
            .enumerate()
        {
            poll(&self.0.budget, at, i)?;
            ensure!(
                proof_class == original_class,
                "prepared original class order changed"
            );
        }
        let lower = self.0.proof.declared_lower();
        let upper = self.0.proof.declared_upper();
        ensure!(
            lower.len() == self.0.property.input.len() && upper.len() == lower.len(),
            "prepared original declared input dimensions changed"
        );
        for (i, ((&lo, &hi), (&source_lo, &source_hi))) in lower
            .iter()
            .zip(upper)
            .zip(
                self.0
                    .property
                    .input
                    .lower()
                    .iter()
                    .zip(self.0.property.input.upper()),
            )
            .enumerate()
        {
            poll(&self.0.budget, at, i)?;
            ensure!(
                lo.to_bits() == source_lo.to_bits() && hi.to_bits() == source_hi.to_bits(),
                "prepared proof lost exact source-certified input enclosure"
            );
        }
        self.0
            .proof
            .validate_objective(self.target(), self.classes(), Some(at))?;
        lease.complete()?;
        self.0.checkpoint_resources(at)?;
        checkpoint(&self.0.budget, at)
    }

    pub(crate) fn budget(&self) -> &WorkBudget {
        &self.0.budget
    }
    pub(crate) fn deadline(&self) -> Instant {
        self.1.min(self.0.deadline)
    }
    /// Consume a successful handoff and retain its earlier publication clock.
    /// This value restriction neither allocates nor changes the shared proof,
    /// source owners, cancellation token, or ledger. Clones keep this clock.
    pub(crate) fn restricted(mut self, requested: Option<Instant>) -> Result<Self> {
        let inherited = self.1.min(self.0.deadline);
        let at = requested.map_or(inherited, |d| d.min(inherited));
        self.checkpoint(Some(at))?;
        self.1 = at;
        Ok(self)
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain exact model-byte identity for publication consumers."
        )
    )]
    pub(crate) fn model_sha256(&self) -> &[u8; 32] {
        &self.0.model_sha256
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain exact property-byte identity for publication consumers."
        )
    )]
    pub(crate) fn property_sha256(&self) -> &[u8; 32] {
        &self.0.property.sha256
    }
    pub(crate) fn target(&self) -> usize {
        self.0.property.objective.target
    }
    pub(crate) fn classes(&self) -> &[usize] {
        &self.0.property.objective.classes
    }
    pub(crate) fn min_margin(&self) -> f64 {
        self.0.proof.min_margin()
    }
    pub(crate) fn attempt_receipt(&self) -> PreparedRootAttemptReceipt {
        self.0.proof.attempt_receipt()
    }
    pub(crate) fn tree_receipt(&self) -> Option<PreparedTreeReceipt> {
        self.0.proof.tree_receipt()
    }
    /// Borrow the held original snapshots and production parse for an
    /// independently replayable export. This neither reloads a pathname nor
    /// releases the sealed proof/source owner or any of its paid claims.
    pub(crate) fn replay_original_source(
        &self,
        requested: Option<Instant>,
    ) -> Result<(&[u8], &str, &VnnLibSpec)> {
        self.checkpoint(requested)?;
        Ok((
            &self.0.model_bytes,
            &self.0.property.source,
            &self.0.property.spec,
        ))
    }
    pub(crate) fn replay_stats(
        &self,
        requested: Option<Instant>,
    ) -> Result<ny_propagate::margin_row::PreparedReplayStats> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        let stats = self.0.proof.replay_stats(Some(at))?;
        self.checkpoint(Some(at))?;
        Ok(stats)
    }
    pub(crate) fn visit_replay_sites(
        &self,
        requested: Option<Instant>,
        visitor: &mut dyn FnMut(usize, usize) -> ny_core::Result<()>,
    ) -> Result<()> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        self.0.proof.visit_replay_sites(Some(at), visitor)?;
        self.checkpoint(Some(at))
    }
    pub(crate) fn visit_replay_leaves(
        &self,
        requested: Option<Instant>,
        visitor: &mut dyn FnMut(
            ny_propagate::margin_row::PreparedReplayLeaf<'_>,
        ) -> ny_core::Result<()>,
    ) -> Result<()> {
        let at = requested.map_or(self.deadline(), |d| d.min(self.deadline()));
        self.checkpoint(Some(at))?;
        self.0.proof.visit_replay_leaves(Some(at), visitor)?;
        self.checkpoint(Some(at))
    }
    /// Keep the authenticated source binding and the successful earlier
    /// proof/publication clock when reusing its completed original session.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain authenticated session reuse with the earlier publication deadline."
        )
    )]
    pub(crate) fn session(&self) -> Result<OriginalPreparedRootSession> {
        self.checkpoint(None)?;
        OriginalPreparedRootSession(self.0.original.clone(), self.deadline()).restricted(None)
    }
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the staged source-allocation receipt projection for publication consumers."
        )
    )]
    pub(super) fn property_allocation_receipt(
        &self,
    ) -> Option<super::original_property_census::OriginalPropertyAllocationReceipt> {
        self.0.allocation
    }
    /// Bind an already authenticated writer/capture instance to these exact
    /// original bytes; equal dimensions or an expected target do not suffice.
    pub(crate) fn validate_asset_hashes(
        &self,
        model: [u8; 32],
        property: [u8; 32],
        requested: Option<Instant>,
    ) -> Result<()> {
        ensure!(
            model == self.0.model_sha256 && property == self.0.property.sha256,
            "prepared publisher belongs to different original model/property bytes"
        );
        self.checkpoint(requested)
    }
}

/// Same G3 probe order, coordinates, margins and production predicate as the
/// existing adapter. One paid vector is reused after each complete evaluation;
/// no preview or guard can add numerical proof authority. Quadratic evaluation
/// work remains charged even though workspace is linear in input/output width.
pub(super) struct StagedSpecAgreement {
    pub(super) report: ny_propagate::certify::GuardReport,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the paid source-agreement diagnostic beside its work lease."
        )
    )]
    pub(super) fragment: String,
    deadline: Instant,
    // Guard strings/Vecs remain paid until their actual ratification consumer
    // releases this result. Completing work does not release its byte claim.
    _workspace: WorkLease,
}

pub(super) fn streamed_spec_agreement(
    spec: &VnnLibSpec,
    target: usize,
    classes: &[usize],
    budget: &WorkBudget,
    requested: Instant,
) -> Result<StagedSpecAgreement> {
    use ny_propagate::{
        certify::{GuardId, GuardReport},
        guards::spec_agreement::SpecAgreementAudit,
    };
    checkpoint(budget, requested)?;
    let mut scan = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            scratch_bytes: META,
            work: charged(
                checked_add(spec.output_constraint_clauses.len(), classes.len())?,
                32,
            )?,
            ..WorkRequest::default()
        },
    )?;
    let requested = scan.deadline().map_or(requested, |d| d.min(requested));
    let probes = checked_add(
        checked_add(checked_mul(classes.len(), 2)?, spec.num_outputs)?,
        1,
    )?;
    let mut atoms = spec.output_constraints.len();
    for (i, clause) in spec.output_constraint_clauses.iter().enumerate() {
        poll(budget, requested, i)?;
        atoms = checked_add(atoms, clause.len())?;
    }
    let mut invalid_classes = target >= spec.num_outputs;
    for (i, &class) in classes.iter().enumerate() {
        poll(budget, requested, i)?;
        invalid_classes |= class >= spec.num_outputs || class == target;
    }
    let per_probe = checked_add(
        checked_add(spec.input_bounds.len(), spec.declared_input_bounds.len())?,
        checked_add(checked_add(atoms, classes.len())?, spec.num_outputs)?,
    )?;
    let mut scratch = checked_add(
        checked_mul(checked_add(spec.input_bounds.len(), spec.num_outputs)?, 8)?,
        META,
    )?;
    if invalid_classes {
        // Preserve the legacy diagnostic verbatim. Debug formatting of the
        // whole class slice is variable storage, even on a refusal path.
        // ceil(bits/3) bounds usize decimal digits since log2(10)>3; pay
        // separators, fixed text and fourfold String growth BEFORE formatting.
        let digits = usize::try_from(usize::BITS.div_ceil(3))?;
        let error_payload = checked_add(checked_mul(classes.len(), checked_add(digits, 2)?)?, 256)?;
        scratch = checked_add(scratch, checked_mul(error_payload, 4)?)?;
    }
    scan.complete()?;
    drop(scan);
    let mut lease = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            scratch_bytes: scratch,
            work: charged(
                checked_add(
                    checked_mul(probes, per_probe)?,
                    checked_mul(spec.num_outputs, classes.len())?,
                )?,
                64,
            )?,
            ..WorkRequest::default()
        },
    )?;
    let at = lease.deadline().map_or(requested, |d| d.min(requested));
    checkpoint(budget, at)?;
    if !spec.per_clause_input_bounds.is_empty() {
        lease.complete()?;
        return Ok(StagedSpecAgreement {
            report: GuardReport::new(),
            fragment: "g3=skipped(per_clause_boxes)".to_string(),
            deadline: at,
            _workspace: lease,
        });
    }
    if invalid_classes {
        let mut report = GuardReport::new();
        report.trip(
            GuardId::SpecAgreement,
            format!(
                "lane classes (t={target}, adv={classes:?}) are not outputs of the spec (n={})",
                spec.num_outputs
            ),
        );
        lease.complete()?;
        return Ok(StagedSpecAgreement {
            report,
            fragment: "g3=ran g3_class_range=refused".to_string(),
            deadline: at,
            _workspace: lease,
        });
    }
    let mut x = Vec::new();
    x.try_reserve_exact(spec.input_bounds.len())?;
    let mut y = Vec::new();
    y.try_reserve_exact(spec.num_outputs)?;
    ensure!(
        x.capacity() <= spec.input_bounds.len() && y.capacity() <= spec.num_outputs,
        "staged G3 actual vector capacities exceed workspace admission"
    );
    for (i, &(lo, hi)) in spec.input_bounds.iter().enumerate() {
        poll(budget, at, i)?;
        x.push(f64::midpoint(lo, hi));
    }
    for i in 0..spec.num_outputs {
        poll(budget, at, i)?;
        y.push(0.0f64);
    }
    y[target] = 1.0;
    let mut audit = SpecAgreementAudit::default();
    let mut sample = 0usize;
    let mut note = |y: &[f64]| -> Result<()> {
        checkpoint(budget, at)?;
        let mut engine_min = f64::INFINITY;
        for (i, &class) in classes.iter().enumerate() {
            poll(budget, at, i)?;
            engine_min = engine_min.min(y[target] - y[class]);
        }
        let violated = super::vnncomp::property_violated_f64(spec, &x, y);
        audit.note(sample, engine_min, violated, 1.5);
        sample = checked_add(sample, 1)?;
        checkpoint(budget, at)
    };
    note(&y)?;
    for &class in classes {
        for d in [0.5f64, -0.5] {
            y[class] = 1.0 + d;
            note(&y)?;
        }
        y[class] = 0.0;
    }
    for class in 0..spec.num_outputs {
        poll(budget, at, class)?;
        if class != target && !classes.contains(&class) {
            y[class] = 1.5;
            note(&y)?;
            y[class] = 0.0;
        }
    }
    let report = audit.report();
    let fragment = format!("g3=ran {}", audit.fragment());
    drop(x);
    drop(y);
    lease.complete()?;
    checkpoint(budget, at)?;
    Ok(StagedSpecAgreement {
        report,
        fragment,
        deadline: at,
        _workspace: lease,
    })
}

fn source_len(file: &File) -> Result<usize> {
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file(),
        "prepared original source must be a regular file"
    );
    usize::try_from(metadata.len()).map_err(Into::into)
}

fn read_snapshot(file: &mut File, len: usize, budget: &WorkBudget, at: Instant) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len)?;
    ensure!(
        bytes.capacity() <= len,
        "original source allocator capacity grew beyond admission"
    );
    let mut block = [0u8; POLL];
    while bytes.len() < len {
        checkpoint(budget, at)?;
        let want = block.len().min(len - bytes.len());
        file.read_exact(&mut block[..want])?;
        bytes.extend_from_slice(&block[..want]);
    }
    let mut tail = [0u8; 1];
    ensure!(
        file.read(&mut tail)? == 0,
        "original source changed length while captured"
    );
    checkpoint(budget, at)?;
    Ok(bytes)
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain bounded source hashing for the direct-file issuer and replay binding."
    )
)]
fn hash(bytes: &[u8], budget: &WorkBudget, at: Instant) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    for block in bytes.chunks(POLL) {
        checkpoint(budget, at)?;
        digest.update(block);
    }
    Ok(digest.finalize().into())
}

fn checkpoint(budget: &WorkBudget, at: Instant) -> Result<()> {
    budget.checkpoint_for(LeaseClass::Replay)?;
    ensure!(
        Instant::now() < at,
        "prepared CLI original deadline expired"
    );
    ny_core::require_f64_interval_proof_environment()?;
    Ok(())
}
fn poll(budget: &WorkBudget, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, at)?;
    }
    Ok(())
}
fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow!("prepared source byte overflow"))
}
fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("prepared source byte overflow"))
}
fn charged(n: usize, scale: usize) -> Result<u64> {
    u64::try_from(checked_add(checked_mul(n, scale)?, 8192)?).map_err(Into::into)
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain actual-capacity reconciliation for the bounded direct-file issuer."
    )
)]
fn visible_bytes(
    model: &Vec<u8>,
    property: &OriginalProperty,
    budget: &WorkBudget,
    at: Instant,
) -> Result<usize> {
    // The parser output is retained inside its already admitted conservative
    // 256*source-byte claim. Source-restricted declarations and direct atoms
    // bound all vector/hash-map contents by source bytes. This floor still
    // reconciles every directly available dynamic capacity explicitly.
    let spec = &property.spec;
    let mut bytes = checked_add(model.capacity(), property.source.capacity())?;
    for amount in [
        checked_mul(spec.input_bounds.capacity(), size_of::<(f64, f64)>())?,
        checked_mul(
            spec.declared_input_bounds.capacity(),
            size_of::<(f64, f64)>(),
        )?,
        checked_mul(
            spec.output_constraints.capacity(),
            size_of::<OutputConstraint>(),
        )?,
        checked_mul(
            spec.output_constraint_clauses.capacity(),
            size_of::<Vec<OutputConstraint>>(),
        )?,
        checked_mul(property.objective.classes.capacity(), size_of::<usize>())?,
        checked_mul(property.objective.atoms.capacity(), size_of::<Atom>())?,
        checked_mul(
            spec.per_clause_input_bounds.capacity(),
            size_of::<std::collections::BTreeMap<usize, (f64, f64)>>(),
        )?,
        checked_mul(property.input.lower.capacity(), size_of::<f64>())?,
        checked_mul(property.input.upper.capacity(), size_of::<f64>())?,
        source_owner_bytes()?,
    ] {
        bytes = checked_add(bytes, amount)?;
    }
    for (i, clause) in spec.output_constraint_clauses.iter().enumerate() {
        poll(budget, at, i)?;
        bytes = checked_add(
            bytes,
            checked_mul(clause.capacity(), size_of::<OutputConstraint>())?,
        )?;
    }
    if let Some(version) = &spec.version {
        bytes = checked_add(bytes, version.capacity())?;
    }
    Ok(bytes)
}

/// A constant-memory token cursor. It never allocates a complete syntax tree
/// or normalizes arithmetic/Boolean formulas into proof authority.
struct Cursor<'a> {
    text: &'a str,
    position: usize,
    budget: &'a WorkBudget,
    at: Instant,
}
impl<'a> Cursor<'a> {
    fn new(text: &'a str, budget: &'a WorkBudget, at: Instant) -> Self {
        Self {
            text,
            position: 0,
            budget,
            at,
        }
    }
    fn token(&mut self) -> Result<Option<&'a str>> {
        let bytes = self.text.as_bytes();
        while self.position < bytes.len() {
            poll(self.budget, self.at, self.position)?;
            if bytes[self.position].is_ascii_whitespace() {
                self.position += 1;
                continue;
            }
            if bytes[self.position] == b';' {
                while self.position < bytes.len() && bytes[self.position] != b'\n' {
                    poll(self.budget, self.at, self.position)?;
                    self.position += 1;
                }
                continue;
            }
            break;
        }
        if self.position == bytes.len() {
            return Ok(None);
        }
        let start = self.position;
        let first = bytes[self.position];
        self.position += 1;
        if first == b'(' || first == b')' {
            return Ok(Some(&self.text[start..self.position]));
        }
        ensure!(
            first != b'"' && first != b'|',
            "prepared source rejects quoted or aliased tokens"
        );
        while self.position < bytes.len()
            && !bytes[self.position].is_ascii_whitespace()
            && !matches!(bytes[self.position], b'(' | b')' | b';')
        {
            poll(self.budget, self.at, self.position)?;
            self.position += 1;
        }
        Ok(Some(&self.text[start..self.position]))
    }
    fn required(&mut self) -> Result<&'a str> {
        self.token()?
            .ok_or_else(|| anyhow!("truncated prepared original source"))
    }
    fn expect(&mut self, want: &str) -> Result<()> {
        ensure!(self.required()? == want, "prepared source expected {want}");
        Ok(())
    }
}

fn canonical(token: &str, prefix: &str) -> Option<usize> {
    let digits = token.strip_prefix(prefix)?;
    if digits.is_empty()
        || !digits.bytes().all(|b| b.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}
fn relation(token: &str) -> Result<Relation> {
    match token {
        "<=" => Ok(Relation::Le),
        ">=" => Ok(Relation::Ge),
        "<" => Ok(Relation::Lt),
        ">" => Ok(Relation::Gt),
        _ => bail!("prepared objective requires direct signed comparisons"),
    }
}

pub(super) fn validate_literal(token: &str) -> Result<()> {
    ensure!(
        token.len() <= MAX_LITERAL_BYTES && !token.is_empty(),
        "prepared numeric literal exceeds bounded surface"
    );
    let unsigned = token
        .strip_prefix('-')
        .or_else(|| token.strip_prefix('+'))
        .unwrap_or(token);
    let (mantissa, exponent) = match unsigned
        .split_once('e')
        .or_else(|| unsigned.split_once('E'))
    {
        Some((m, e)) => (m, e.parse::<i32>()?),
        None => (unsigned, 0),
    };
    ensure!(
        exponent.unsigned_abs() <= MAX_LITERAL_SCALE as u32,
        "prepared literal exponent exceeds bounded surface"
    );
    let mut pieces = mantissa.split('.');
    let integer = pieces.next().unwrap_or_default();
    let fraction = pieces.next().unwrap_or_default();
    ensure!(
        pieces.next().is_none()
            && (!integer.is_empty() || !fraction.is_empty())
            && integer.bytes().all(|b| b.is_ascii_digit())
            && fraction.bytes().all(|b| b.is_ascii_digit()),
        "prepared source literal is not a direct decimal"
    );
    let scale = i32::try_from(fraction.len())?
        .checked_sub(exponent)
        .ok_or_else(|| anyhow!("prepared literal scale overflow"))?;
    ensure!(
        scale.unsigned_abs() <= MAX_LITERAL_SCALE as u32,
        "prepared literal scale exceeds bounded surface"
    );
    Ok(())
}

fn parse_output_atom(cursor: &mut Cursor<'_>, first: &str) -> Result<Atom> {
    let relation = relation(first)?;
    let left = canonical(cursor.required()?, "Y_")
        .ok_or_else(|| anyhow!("prepared output is not a direct canonical Y"))?;
    let right = canonical(cursor.required()?, "Y_")
        .ok_or_else(|| anyhow!("prepared output is not a direct canonical Y"))?;
    cursor.expect(")")?;
    ensure!(left != right, "prepared objective rejects self comparison");
    Ok(Atom {
        relation,
        left,
        right,
    })
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retain direct-file raw objective authentication under its original grammar cap."
    )
)]
fn authenticate_raw_objective(
    text: &str,
    budget: &WorkBudget,
    at: Instant,
) -> Result<OriginalObjective> {
    authenticate_raw_objective_impl(text, budget, at, MAX_PROPERTY_BYTES, false)
}

pub(super) fn authenticate_raw_objective_with_cap(
    text: &str,
    budget: &WorkBudget,
    at: Instant,
    source_cap: usize,
) -> Result<OriginalObjective> {
    authenticate_raw_objective_impl(text, budget, at, source_cap, true)
}

fn authenticate_raw_objective_impl(
    text: &str,
    budget: &WorkBudget,
    at: Instant,
    source_cap: usize,
    guard_capacity: bool,
) -> Result<OriginalObjective> {
    ensure!(
        text.len() <= source_cap,
        "prepared direct property source cap"
    );
    let mut cursor = Cursor::new(text, budget, at);
    let mut inputs = 0usize;
    let mut outputs = 0usize;
    let mut atoms = Vec::new();
    let mut output_seen = false;
    // No sparse declaration can manufacture a huge ordinary parser dimension.
    // Exact ascending IDs are required before any assertion.
    let mut assertions = false;
    while let Some(token) = cursor.token()? {
        ensure!(
            token == "(",
            "prepared source needs complete top-level forms"
        );
        match cursor.required()? {
            kind @ ("declare-const" | "declare-fun") => {
                ensure!(!assertions, "prepared declaration appears after assertions");
                let name = cursor.required()?;
                if let Some(i) = canonical(name, "X_") {
                    ensure!(
                        i == inputs,
                        "prepared source requires complete ordered X declarations"
                    );
                    inputs = inputs
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("input count overflow"))?;
                } else if let Some(i) = canonical(name, "Y_") {
                    ensure!(
                        i == outputs,
                        "prepared source requires complete ordered Y declarations"
                    );
                    outputs = outputs
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("output count overflow"))?;
                } else {
                    bail!("prepared declaration is outside canonical X/Y scalar grammar");
                }
                // A const has `Real`, while a zero-argument fun has `() Real`.
                if kind == "declare-fun" {
                    cursor.expect("(")?;
                    cursor.expect(")")?;
                    cursor.expect("Real")?;
                } else {
                    cursor.expect("Real")?;
                }
                cursor.expect(")")?;
            }
            "assert" => {
                assertions = true;
                ensure!(
                    inputs > 0 && outputs >= 2,
                    "prepared robustness requires bounded inputs and >=2 outputs"
                );
                cursor.expect("(")?;
                let op = cursor.required()?;
                if op == "or" {
                    ensure!(!output_seen, "prepared source has multiple output formulas");
                    output_seen = true;
                    loop {
                        let start = cursor.required()?;
                        if start == ")" {
                            break;
                        }
                        ensure!(start == "(", "prepared OR requires direct singleton atoms");
                        let first = cursor.required()?;
                        let atom = if first == "and" {
                            cursor.expect("(")?;
                            let first = cursor.required()?;
                            let atom = parse_output_atom(&mut cursor, first)?;
                            cursor.expect(")")?;
                            atom
                        } else {
                            parse_output_atom(&mut cursor, first)?
                        };
                        ensure!(
                            !guard_capacity || atoms.len() < outputs - 1,
                            "staged source output atom capacity exceeds complete objective"
                        );
                        atoms.try_reserve(1)?;
                        atoms.push(atom);
                    }
                } else {
                    // One direct output is sufficient only for n_out=2. The
                    // final complete class census enforces that condition.
                    let rel = relation(op)?;
                    let left = cursor.required()?;
                    let right = cursor.required()?;
                    cursor.expect(")")?;
                    if canonical(left, "Y_").is_some() || canonical(right, "Y_").is_some() {
                        ensure!(!output_seen, "prepared source has multiple output formulas");
                        output_seen = true;
                        let left =
                            canonical(left, "Y_").ok_or_else(|| anyhow!("mixed output atom"))?;
                        let right =
                            canonical(right, "Y_").ok_or_else(|| anyhow!("mixed output atom"))?;
                        ensure!(left != right, "self output comparison");
                        ensure!(
                            !guard_capacity || atoms.len() < outputs - 1,
                            "staged source output atom capacity exceeds complete objective"
                        );
                        atoms.try_reserve(1)?;
                        atoms.push(Atom {
                            relation: rel,
                            left,
                            right,
                        });
                    } else {
                        let index = match (canonical(left, "X_"), canonical(right, "X_")) {
                            (Some(i), None) => {
                                validate_literal(right)?;
                                i
                            }
                            (None, Some(i)) => {
                                validate_literal(left)?;
                                i
                            }
                            _ => bail!("prepared input requires one direct X and decimal"),
                        };
                        ensure!(index < inputs, "prepared input references undeclared X");
                    }
                }
                cursor.expect(")")?;
            }
            "set-logic" => {
                ensure!(
                    cursor.required()? == "QF_LRA",
                    "prepared source requires QF_LRA logic"
                );
                cursor.expect(")")?;
            }
            "check-sat" | "exit" => cursor.expect(")")?,
            _ => bail!("prepared source contains unsupported top-level semantics"),
        }
    }
    ensure!(
        output_seen && atoms.len() == outputs - 1,
        "prepared objective must cover ALL non-target original classes"
    );
    let mut target = None;
    let mut classes = Vec::new();
    classes.try_reserve_exact(atoms.len())?;
    for (i, atom) in atoms.iter().enumerate() {
        poll(budget, at, i)?;
        ensure!(
            atom.left < outputs && atom.right < outputs,
            "prepared output references undeclared Y"
        );
        let (t, a) = match atom.relation {
            Relation::Le | Relation::Lt => (atom.left, atom.right),
            Relation::Ge | Relation::Gt => (atom.right, atom.left),
        };
        ensure!(
            target.is_none_or(|old| old == t),
            "prepared unsafe atoms disagree on target"
        );
        target = Some(t);
        ensure!(
            !classes.contains(&a),
            "prepared unsafe objective duplicates original class"
        );
        classes.push(a);
    }
    checkpoint(budget, at)?;
    Ok(OriginalObjective {
        inputs,
        outputs,
        target: target.ok_or_else(|| anyhow!("missing original target"))?,
        classes,
        atoms,
    })
}

pub(super) fn cross_check(
    objective: &OriginalObjective,
    spec: &VnnLibSpec,
    input: &CertifiedInputBox,
) -> Result<()> {
    ensure!(
        spec.num_inputs == objective.inputs
            && input.len() == objective.inputs
            && spec.num_outputs == objective.outputs
            && spec.dual_network.is_none()
            && spec.per_clause_input_bounds.iter().all(|b| b.is_empty()),
        "prepared certified original parser dimension/region disagreement"
    );
    ensure!(
        spec.output_constraints.len() == objective.atoms.len()
            && spec.output_constraint_clauses.len() == objective.atoms.len()
            && spec.output_constraint_clauses.iter().all(|c| c.len() == 1),
        "prepared original complete OR shape disagreement"
    );
    ensure!(
        objective.atoms.len() == 1 || spec.is_disjunction,
        "prepared original OR lost by ordinary parser"
    );
    for ((atom, ordinary), clause) in objective
        .atoms
        .iter()
        .zip(&spec.output_constraints)
        .zip(&spec.output_constraint_clauses)
    {
        let matches = match (atom.relation, ordinary) {
            (Relation::Le, OutputConstraint::LessEq(i, j))
            | (Relation::Lt, OutputConstraint::LessThan(i, j))
            | (Relation::Ge, OutputConstraint::GreaterEq(i, j))
            | (Relation::Gt, OutputConstraint::GreaterThan(i, j)) => {
                atom.left == *i && atom.right == *j
            }
            _ => false,
        };
        ensure!(
            matches && clause[0] == *ordinary,
            "prepared raw output source disagrees with original production parse"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "prepared_root_publication_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "original_file_session_tests.rs"]
mod session_tests;

#[cfg(test)]
#[path = "original_file_tree_tests.rs"]
mod tree_tests;

#[cfg(test)]
#[path = "native_affine_source_tests.rs"]
mod native_affine_tests;
