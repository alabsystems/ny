// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automatic original-file prepared proof attempt, before the legacy fallback.
//!
//! This is a separate certified engine, not a replacement for the frozen
//! legacy policy. In particular legacy `err_envelope=off` means mandatory
//! `ErrLane::PerCol`, not absence of error bounds. Both its textual request and
//! the actual outward four-plane proof arithmetic are reported separately.
//! Explicit armed policies this engine cannot execute decline before source
//! reads. The default Source head portfolio is consulted before this route;
//! its default setting does not disable this fallback on unrelated models.
//!
//! Memory derives from NY's existing CPU Dense budget, narrowed by the strict
//! process envelope and the unchanged source ceiling. It never borrows the
//! hull lane's 32 MiB budget. One finite work allowance is declared at this
//! caller: remaining original milliseconds times 2^20, capped at 2^50. This
//! is a scheduling allowance, not a measured processing rate or speed promise.
//! The original clock remains authoritative. No child or resume replenishes
//! the ledger. One certified entry cutoff, original/phase minimum minus the
//! existing 100 ms reserve, is inherited by source, root, tree and proof.
//! An expired prepared proof cannot publish during that reserved interval.
//!
//! Initial source admission leaves model bytes at most one eighth of the
//! declared byte cap for compilation overlap. The existing staged property's
//! grammar ceiling remains unchanged. Necessary original domain/class storage
//! is checked without a fixed input or vocabulary ceiling; actual source,
//! operator, and product quotes still determine complete admission. Search
//! attempts at most 32 expansions. Qualification must measure opportunity
//! cost on original instances before claiming a faster default. This call is
//! synchronous.
//! Earlier loaded models and other legacy phases have not joined this census.

use std::{
    mem::size_of,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, ensure, Result};
use ny_levers::{LeverDecl, LeverSet, LeverValue};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{
            LeaseClass, PersistentReservation, WorkBudget, WorkLimits, WorkRequest, WorkReserve,
        },
    },
    margin_row::{
        bab::{PreparedTreeOptions, PreparedTreeReceipt, PreparedTreeSchedule},
        root_box::{RootBoxSource, ROOT_SOURCE_BYTES_CAP},
        ErrLane, MarginRowRuntimePolicy, PreparedRootAttemptReceipt, PreparedRootProofOptions,
        PreparedRowStorageReceipt, TreeMode,
    },
    network::crown_memory::{
        cpu_crown_dense_budget_bytes, process_memory_envelope, ProcessMemoryEnvelope,
    },
};

use super::{
    prepared_root_publication::{
        OriginalPreparedRootOpen, OriginalPreparedRootOutcome, OriginalPreparedRootSession,
        OriginalPreparedTreeOpen, OriginalPreparedTreeOutcome,
    },
    unsat_receipt::UnsatReceipt,
};

const OWNER_BYTES: usize = 8192;
const CERTIFICATE_RESERVE: Duration = Duration::from_millis(100);
const WORK_PER_MILLISECOND: u64 = 1 << 20;
const MAX_WORK: u64 = 1 << 50;
const MAX_EXPANSIONS: usize = 32;
const PROOF_ARITHMETIC: &str = "outward-per-column-four-plane";

/// A fixed-size caller header; the ledger's own bootstrap Arc/Mutex metadata
/// must exist before it can admit its first lease. Its conservative fixed
/// envelope and this Arc are paid before constructing this shared wrapper.
/// The original-file issuer retains this owner through Open, proof, receipt
/// and the last actual writer. It never owns a source, so there is no cycle.
#[derive(Debug)]
pub(crate) struct PreparedRouteResources {
    budget: WorkBudget,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retain the shared cancellation owner and charged caller-header layout"
        )
    )]
    cancellation: CancellationToken,
    original_deadline: Instant,
    deadline: Instant,
    byte_cap: usize,
    work_allowance: u64,
    _resident: PersistentReservation,
}

impl PreparedRouteResources {
    fn start(original: Option<Instant>, phase: Option<Instant>) -> Result<Arc<Self>> {
        let observed = process_memory_envelope();
        Self::start_observed(
            cpu_crown_dense_budget_bytes(),
            observed,
            original,
            phase,
            Instant::now(),
        )
    }

    fn start_observed(
        cpu_bytes: usize,
        envelope: ProcessMemoryEnvelope,
        original: Option<Instant>,
        phase: Option<Instant>,
        now: Instant,
    ) -> Result<Arc<Self>> {
        let original = earliest(original, ny_core::instance_budget::deadline())
            .ok_or_else(|| anyhow!("prepared route requires original finite deadline"))?;
        let deadline = phase.map_or(original, |at| at.min(original));
        ensure!(
            deadline.saturating_duration_since(now) > CERTIFICATE_RESERVE,
            "prepared entry lacks certificate reserve"
        );
        let cap = admitted_byte_cap(cpu_bytes, envelope)?;
        ensure!(
            cap >= OWNER_BYTES,
            "prepared caller header exceeds memory policy"
        );
        let work_allowance = scheduling_work(original.saturating_duration_since(now))?;
        let memory = MemoryAuthority::new(cap);
        let cancellation = CancellationToken::default();
        let budget = WorkBudget::new_in(
            &memory,
            &cancellation,
            WorkLimits {
                deadline: Some(deadline),
                scratch_bytes: cap,
                persistent_bytes: cap,
                max_work: work_allowance,
                max_proposals: 1,
                certificate_reserve: CERTIFICATE_RESERVE,
                reserve: WorkReserve::default(),
            },
        )?;
        budget.tighten_total_byte_limit(cap)?;
        budget.checkpoint_for(LeaseClass::Optional)?;
        let mut lease = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: OWNER_BYTES,
                work: 512,
                ..WorkRequest::default()
            },
        )?;
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        let owner = Arc::new(Self {
            budget,
            cancellation,
            original_deadline: original,
            deadline,
            byte_cap: cap,
            work_allowance,
            _resident: resident,
        });
        owner.checkpoint(None)?;
        Ok(owner)
    }

    pub(crate) fn budget(&self) -> &WorkBudget {
        &self.budget
    }
    pub(crate) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        let at = requested.map_or(self.deadline, |at| at.min(self.deadline));
        self.checkpoint(Some(at))?;
        Ok(at)
    }
    fn certified_entry_deadline(&self) -> Result<Instant> {
        let at = self
            .deadline
            .checked_sub(CERTIFICATE_RESERVE)
            .ok_or_else(|| anyhow!("prepared certificate cutoff overflow"))?;
        self.checkpoint(Some(at))?;
        Ok(at)
    }
    pub(crate) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = requested.map_or(self.deadline, |at| at.min(self.deadline));
        self.budget.checkpoint_for(LeaseClass::Fallback)?;
        ensure!(
            Instant::now() < at && at <= self.original_deadline,
            "prepared caller deadline expired or renewed"
        );
        ensure!(
            self._resident.shares_ledger(&self.budget) && self._resident.bytes() >= OWNER_BYTES,
            "prepared caller lost original storage authority"
        );
        Ok(())
    }
    pub(crate) fn validate_source_lengths(&self, model: usize, property: usize) -> Result<()> {
        self.checkpoint(None)?;
        ensure!(
            model <= self.byte_cap / 8
                && property <= super::original_property_census::MAX_STAGED_PROPERTY_BYTES,
            "original sources exceed automatic prepared preservation profile"
        );
        Ok(())
    }
    pub(crate) fn validate_objective(&self, inputs: usize, outputs: usize) -> Result<()> {
        self.checkpoint(None)?;
        ensure!(
            inputs > 0 && outputs > 1,
            "empty original prepared objective"
        );
        // Necessary storage, not compilation admission: the actual original
        // root owns lo/hi/mid/rad/xabs, and the objective owns all usize IDs.
        // Full parser/model/product predictions remain separately preleased.
        let input = inputs.checked_mul(5 * size_of::<f64>());
        let classes = (outputs - 1).checked_mul(size_of::<usize>());
        let minimum = input
            .zip(classes)
            .and_then(|(a, b)| a.checked_add(b))
            .ok_or_else(|| anyhow!("original objective necessary storage overflow"))?;
        ensure!(
            minimum <= self.byte_cap,
            "original objective necessary storage exceeds caller memory policy"
        );
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }
}

const _: () =
    assert!(size_of::<PreparedRouteResources>() + 2 * size_of::<usize>() + 4096 <= OWNER_BYTES);

fn earliest(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
fn admitted_byte_cap(cpu_bytes: usize, envelope: ProcessMemoryEnvelope) -> Result<usize> {
    let process = match envelope {
        ProcessMemoryEnvelope::Bounded { headroom_bytes } => {
            usize::try_from(headroom_bytes).unwrap_or(usize::MAX)
        }
        ProcessMemoryEnvelope::Unbounded => usize::MAX,
        ProcessMemoryEnvelope::Unavailable => {
            return Err(anyhow!("strict memory envelope unavailable"))
        }
    };
    Ok(cpu_bytes.min(process).min(ROOT_SOURCE_BYTES_CAP))
}
fn scheduling_work(remaining: Duration) -> Result<u64> {
    let millis = remaining.as_millis();
    let work = millis
        .checked_mul(u128::from(WORK_PER_MILLISECOND))
        .ok_or_else(|| anyhow!("prepared scheduling allowance overflow"))?;
    let work = work.min(u128::from(MAX_WORK)) as u64;
    ensure!(work >= 512, "prepared scheduling allowance exhausted");
    Ok(work)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedRouteStatus {
    Declined,
    Refused,
    RootProven,
    TreeProven,
    Open,
}

/// Capacities and admitted work are resource receipts, not measured RSS,
/// runtimes, solve gains, or fulfillment of unsupported legacy tuning doses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedRouteReceipt {
    pub(crate) status: PreparedRouteStatus,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retain the requested error lane in the route audit receipt"
        )
    )]
    pub(crate) requested_error_lane: &'static str,
    pub(crate) reference_arithmetic: ny_propagate::margin_row::PreparedReferenceArithmetic,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retain the executed proof arithmetic in the route audit receipt"
        )
    )]
    pub(crate) proof_arithmetic: &'static str,
    pub(crate) reason: &'static str,
    pub(crate) byte_cap: usize,
    pub(crate) work_allowance: u64,
    pub(crate) work_reserved: u64,
    pub(crate) peak_leased_bytes: usize,
    pub(crate) class_count: usize,
    #[expect(dead_code, reason = "preserve original route resource receipt")]
    pub(crate) max_expansions: usize,
    pub(crate) original_deadline: Option<Instant>,
    pub(crate) certified_entry_cutoff: Option<Instant>,
    pub(crate) root: Option<PreparedRootAttemptReceipt>,
    pub(crate) row_storage: Option<PreparedRowStorageReceipt>,
    pub(crate) tree: Option<PreparedTreeReceipt>,
}
impl PreparedRouteReceipt {
    fn initial(policy: &MarginRowRuntimePolicy) -> Self {
        Self {
            status: PreparedRouteStatus::Declined,
            requested_error_lane: policy.root_precision.err_lane.token(),
            reference_arithmetic:
                ny_propagate::margin_row::PreparedReferenceArithmetic::Binary32Certified,
            proof_arithmetic: PROOF_ARITHMETIC,
            reason: "not-admitted",
            byte_cap: 0,
            work_allowance: 0,
            work_reserved: 0,
            peak_leased_bytes: 0,
            class_count: 0,
            max_expansions: MAX_EXPANSIONS,
            original_deadline: None,
            certified_entry_cutoff: None,
            root: None,
            row_storage: None,
            tree: None,
        }
    }
    fn observe(&mut self, resources: &PreparedRouteResources) {
        self.byte_cap = resources.byte_cap;
        self.work_allowance = resources.work_allowance;
        self.original_deadline = Some(resources.original_deadline);
        self.certified_entry_cutoff = resources.deadline.checked_sub(CERTIFICATE_RESERVE);
        if let Ok(r) = resources.budget.receipt() {
            self.work_reserved = r.work_reserved;
            self.peak_leased_bytes = r.peak_leased_bytes;
        }
    }
}

/// The actual unfinished source stays owned until the caller selects fallback.
/// No numeric partial is exposed and no fresh ledger is offered for resuming.
pub(crate) enum PreparedRouteOpen {
    #[expect(
        dead_code,
        reason = "unfinished root retains source and resource owners"
    )]
    Root(OriginalPreparedRootOpen),
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "unfinished tree retains source and resource owners until fallback"
        )
    )]
    Tree(OriginalPreparedTreeOpen),
}
pub(crate) enum PreparedRouteOutcome {
    Proven {
        receipt: UnsatReceipt,
        report: PreparedRouteReceipt,
    },
    Open {
        owner: PreparedRouteOpen,
        report: PreparedRouteReceipt,
    },
    Refused {
        report: PreparedRouteReceipt,
    },
}

// ---- #prepared-root-route: THE admission ------------------------------------
//
// Whether the route runs on a row is decided by ONE pure function,
// [`admission`]: the raw classwise experiment steps it aside, and the frozen
// margin-row policy must be the default one this engine executes (no armed
// dose, the consumer switches off, a parseable LRU and parallel token). The
// scored entry point (`vnncomp.rs`) calls it and hands its answer to
// [`try_prepared_root_route`], which cannot run without one (the token has no
// public constructor); the router registry asks the same function through
// [`router_admission`], and so does the admission census. Resource admission
// (a finite original deadline, the certificate reserve, a strict process
// memory envelope) and the original-file compile are NOT admission: they run
// after it, read the clock and the process, and refuse inside the route.

/// The admission's refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedRootDecline {
    /// The raw `NY_MARGIN_ROW_CLASSWISE` experiment is armed (the entry
    /// point's own read): the route steps aside before any report.
    ClasswiseArmed,
    /// The frozen policy arms something this engine cannot execute.
    Policy(&'static str),
}

impl PreparedRootDecline {
    /// The stable reason token (the route's `reason` field).
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::ClasswiseArmed => "armed-original-classwise-policy",
            Self::Policy(reason) => reason,
        }
    }
}

/// The answer of [`admission`]: the only way to obtain one, so the lane cannot
/// start without asking the predicate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedRootAdmission(std::result::Result<AdmittedPolicy, PreparedRootDecline>);

impl PreparedRootAdmission {
    /// The raw classwise experiment stepped the route aside.
    pub(crate) const fn classwise_armed(&self) -> bool {
        matches!(self.0, Err(PreparedRootDecline::ClasswiseArmed))
    }
    /// The refusal, if the route does not run.
    pub(crate) const fn decline(&self) -> Option<PreparedRootDecline> {
        match self.0 {
            Ok(_) => None,
            Err(decline) => Some(decline),
        }
    }
}

/// What the admission admitted: the frozen tree controls the route runs with.
#[derive(Clone, Copy, Debug)]
struct AdmittedPolicy {
    lru_capacity: usize,
    parallel: ParallelRequest,
}

impl AdmittedPolicy {
    /// The tree options on `workers` threads (the width is a schedule, never
    /// an admission).
    const fn tree(self, workers: usize) -> PreparedTreeOptions {
        let width = self.parallel.width(workers);
        PreparedTreeOptions {
            max_expansions: MAX_EXPANSIONS,
            lru_capacity: self.lru_capacity,
            schedule: if width == 1 {
                PreparedTreeSchedule::Serial
            } else {
                PreparedTreeSchedule::Parallel { width }
            },
        }
    }
}

/// THE admission of the prepared-root route (one implementation: the scored
/// entry point, the router registry and the admission census call it). Pure:
/// no clock, no environment, no file. `classwise_requested` is the entry
/// point's own read of the raw classwise experiment.
pub(crate) fn admission(
    classwise_requested: bool,
    policy: &MarginRowRuntimePolicy,
    levers: &LeverSet,
) -> PreparedRootAdmission {
    if classwise_requested {
        return PreparedRootAdmission(Err(PreparedRootDecline::ClasswiseArmed));
    }
    PreparedRootAdmission(policy_gate(policy, levers).map_err(PreparedRootDecline::Policy))
}

/// The router's question to [`admission`], from the frozen lever set the
/// router started with: the margin-row policy those levers resolve to (a set
/// that resolves to none never reaches the route: the scored entry point
/// returns `unknown` first). The router reads no environment, so it asks in
/// the competition form, where the raw classwise experiment is never set
/// (`run_instance.sh` exports zero `NY_*`): a row run with it set sees the
/// lane decline where the router admitted, so the router can only over-admit.
pub(crate) fn router_admission(levers: &LeverSet) -> ny_core::router::decl::Admission {
    use ny_core::router::decl::Admission;
    let Ok(policy) = MarginRowRuntimePolicy::from_levers(levers) else {
        return Admission::Decline("invalid-frozen-margin-row-policy");
    };
    match admission(false, &policy, levers).decline() {
        None => Admission::Admit,
        Some(decline) => Admission::Decline(decline.reason()),
    }
}

fn policy_gate(
    policy: &MarginRowRuntimePolicy,
    levers: &LeverSet,
) -> std::result::Result<AdmittedPolicy, &'static str> {
    use ny_levers::decls::{margin_row as d, sound_channel as s};
    if policy.root_precision.root_f32
        || policy.root_precision.tableau_f32
        || policy.root_precision.err_lane != ErrLane::PerCol
        || policy.root_precision.ibp_seed
        || policy.root_precision.box_source != RootBoxSource::Tableau
    {
        return Err("unsupported-root-precision-or-source-dose");
    }
    let default = MarginRowRuntimePolicy::default();
    // The Source head portfolio is now a default earlier stage. Preserve this
    // automatic fallback after that consultation; an explicit group dose or
    // a nondefault nested policy still cannot be silently consumed here.
    let head_group_dose = if policy.head_groups.is_some() {
        let setting = levers
            .resolved(&d::MARGIN_ROW_HEAD_GROUPS)
            .ok_or("missing-frozen-head-groups")?;
        policy.head_groups != default.head_groups
            || setting.source != ny_levers::Source::Default
            || setting.value != LeverValue::Bool(true)
    } else {
        false
    };
    if policy.width != default.width
        || policy.backward_interm.enabled
        || policy.root_alpha.alpha_enabled
        || policy.root_alpha.rebake_enabled
        || policy.root_alpha.spec_alpha_enabled
        || policy.joint_refresh.rounds != 0
        || policy.root_certify_once.enabled
        || head_group_dose
        || policy.domain_alpha.enabled
        || policy.epoch.depth != 0
        || policy.epoch.tier0_exact != 0
        || policy.epoch.parent_topk != 0
        || policy.epoch.debug_trace
        || policy.beta_per_column
        || policy.column_retire
        || policy.parallel_prebuild
        || policy.wave.mode != TreeMode::Legacy
    {
        return Err("unsupported-armed-runtime-dose");
    }
    // Dormant tuning does not arm a dose. These independent switches do, and
    // cannot be discharged by constructing a different numerical engine.
    for decl in [
        &d::MARGIN_ROW_BETA,
        &d::MARGIN_ROW_CLIP_STATEFUL,
        &d::MARGIN_ROW_BRANCH_REDUCE_MAX,
        &d::MARGIN_ROW_PARENT_LIFT,
        &d::MARGIN_ROW_GPU_BACKWARD,
        &d::MARGIN_ROW_GPU_SHADOW,
        &d::MARGIN_ROW_GPU_PROF,
        &d::MARGIN_ROW_GPU_CONV,
        &d::MARGIN_ROW_DOMAIN_STACK,
        &d::MARGIN_ROW_FROZEN_DUAL_BAB,
        &s::MARGIN_ROW_CLIP,
        &s::MARGIN_ROW_CLIP_HEAD,
        &s::MARGIN_ROW_CLIP_INTERM,
        &s::MARGIN_ROW_CLIP_ROWS,
        &s::MARGIN_ROW_CLOSURE_ADAPTIVE,
    ] {
        if frozen_value(levers, decl)?.as_bool() {
            return Err("unsupported-armed-consumer-dose");
        }
    }
    let lru_capacity = frozen_value(levers, &d::MARGIN_ROW_LRU)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("invalid-frozen-lru")?;
    let parallel = frozen_value(levers, &d::MARGIN_ROW_PARALLEL)?
        .as_str()
        .ok_or("invalid-frozen-parallel")?;
    // Same documented legacy token grammar; no process/global preset read.
    Ok(AdmittedPolicy {
        lru_capacity,
        parallel: ParallelRequest::parse(parallel),
    })
}
fn frozen_value<'a>(
    set: &'a LeverSet,
    decl: &'static LeverDecl,
) -> std::result::Result<&'a LeverValue, &'static str> {
    set.resolved(decl)
        .map(|r| &r.value)
        .ok_or("missing-frozen-policy-entry")
}
/// The frozen `margin_row.parallel` request, in the documented legacy token
/// grammar: every worker, or a fixed width (at least 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParallelRequest {
    Workers,
    Width(usize),
}

impl ParallelRequest {
    fn parse(raw: &str) -> Self {
        let raw = raw.trim();
        if raw == "1"
            || ["true", "on", "yes"]
                .iter()
                .any(|v| raw.eq_ignore_ascii_case(v))
        {
            Self::Workers
        } else if raw.is_empty()
            || raw == "0"
            || ["false", "off", "no"]
                .iter()
                .any(|v| raw.eq_ignore_ascii_case(v))
        {
            Self::Width(1)
        } else {
            Self::Width(raw.parse::<usize>().unwrap_or(1).max(1))
        }
    }
    const fn width(self, workers: usize) -> usize {
        match self {
            Self::Workers if workers > 1 => workers,
            Self::Workers => 1,
            Self::Width(width) => width,
        }
    }
}

#[cfg(test)]
fn parallel_width(raw: &str, workers: usize) -> usize {
    ParallelRequest::parse(raw).width(workers)
}

/// No category/path allowlist or arming environment switch. The actual frozen
/// operator policy and finite independently observed resources decide entry:
/// the caller hands in [`admission`]'s answer (the one predicate the router
/// and the census ask too), so a refused admission declines before any
/// resource or source work, exactly as the inline admission did.
pub(crate) fn try_prepared_root_route(
    model: &Path,
    property: &Path,
    policy: &MarginRowRuntimePolicy,
    admission: PreparedRootAdmission,
    original_deadline: Option<Instant>,
    phase_deadline: Option<Instant>,
) -> PreparedRouteOutcome {
    let mut report = PreparedRouteReceipt::initial(policy);
    let admitted = match admission.0 {
        Ok(v) => v.tree(rayon::current_num_threads()),
        Err(decline) => {
            report.reason = decline.reason();
            return PreparedRouteOutcome::Refused { report };
        }
    };
    let resources = match PreparedRouteResources::start(original_deadline, phase_deadline) {
        Ok(v) => v,
        Err(e) => {
            report.reason = "caller-resource-admission";
            eprintln!(
                "[prepared-route] declined reason={} error={e}",
                report.reason
            );
            return PreparedRouteOutcome::Refused { report };
        }
    };
    run_admitted(model, property, policy, admitted, resources, report)
}

fn run_admitted(
    model: &Path,
    property: &Path,
    policy: &MarginRowRuntimePolicy,
    tree: PreparedTreeOptions,
    resources: Arc<PreparedRouteResources>,
    mut report: PreparedRouteReceipt,
) -> PreparedRouteOutcome {
    report.status = PreparedRouteStatus::Refused;
    report.observe(&resources);
    let outcome = (|| -> Result<PreparedRouteOutcome> {
        let at = resources.certified_entry_deadline()?;
        // Same authentic file issuer, one compile, no copied objective labels.
        let session = OriginalPreparedRootSession::prepare_original_files_in_caller(
            model,
            property,
            policy.root_precision,
            PreparedRootProofOptions {
                reference_arithmetic: report.reference_arithmetic,
                ..PreparedRootProofOptions::default()
            },
            resources.clone(),
            at,
        )?;
        report.row_storage = Some(session.row_storage_receipt());
        report.class_count = session.classes().len();
        match session.try_prove_root(Some(at))? {
            OriginalPreparedRootOutcome::Proven(owner) => {
                report.root = Some(owner.attempt_receipt());
                let receipt = UnsatReceipt::from_original_prepared(owner, Some(at))?;
                report.status = PreparedRouteStatus::RootProven;
                report.reason = "original-root-proof";
                report.observe(&resources);
                Ok(PreparedRouteOutcome::Proven { receipt, report })
            }
            OriginalPreparedRootOutcome::Open(open) => {
                report.root = Some(open.attempt_receipt());
                match open.try_tree(tree, Some(at)) {
                    Ok(OriginalPreparedTreeOutcome::Proven(owner)) => {
                        report.tree = owner.tree_receipt();
                        let receipt = UnsatReceipt::from_original_prepared(owner, Some(at))?;
                        report.status = PreparedRouteStatus::TreeProven;
                        report.reason = "original-tree-cover";
                        report.observe(&resources);
                        Ok(PreparedRouteOutcome::Proven { receipt, report })
                    }
                    Ok(OriginalPreparedTreeOutcome::Open(tree)) => {
                        report.tree = Some(tree.receipt());
                        report.status = PreparedRouteStatus::Open;
                        report.reason = "original-tree-open";
                        report.observe(&resources);
                        Ok(PreparedRouteOutcome::Open {
                            owner: PreparedRouteOpen::Tree(tree),
                            report,
                        })
                    }
                    Err(e) => {
                        eprintln!("[prepared-route] tree refused error={e}");
                        report.status = PreparedRouteStatus::Open;
                        report.reason = "tree-refused-root-retained";
                        report.observe(&resources);
                        Ok(PreparedRouteOutcome::Open {
                            owner: PreparedRouteOpen::Root(open),
                            report,
                        })
                    }
                }
            }
        }
    })();
    match outcome {
        Ok(v) => v,
        Err(e) => {
            report.reason = "original-source-or-root-refused";
            report.observe(&resources);
            eprintln!(
                "[prepared-route] refused reason={} error={e}",
                report.reason
            );
            PreparedRouteOutcome::Refused { report }
        }
    }
}

#[cfg(test)]
#[path = "prepared_root_route_tests.rs"]
mod tests;
