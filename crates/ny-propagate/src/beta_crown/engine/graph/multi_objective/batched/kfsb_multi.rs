// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Wave-batched kFSB branch selection for the multi-objective graph BaB lane
//! (#kfsb-multi, barrier 2 — dark, `NY_MO_KFSB=1`, default OFF).
//!
//! MEASURED MOTIVATION (prop1498 d5 worst child, LP-exact min-of-2-children on
//! 48 candidate splits): exact child evaluation SEPARATES candidates — best
//! split lift +0.012 vs the intercept-argmax's ~0.0 (the depth-invariance
//! plateau). The lane honors the configured αβ-CROWN reduction policy; under
//! `Min`, one-sided splits are ranked by their surviving child because an empty
//! half is `+inf`, while the VNN-COMP CIFAR/TinyImageNet presets select `Max`.
//! The multi-objective lane's
//! advisory selector (`select_graph_branch_multi`) never child-evaluates; this
//! module brings the single-objective GPU-kFSB discipline (score → filter →
//! SIMULATE both children → pick by reduce-op) to the wave level:
//!
//! 1. PRE-SCORE each wave domain's unstable neurons with
//!    `compute_graph_babsr_scores_from_bounds`, seeded with that domain's worst
//!    unverified straggler's margin row (objective-directed BaBSR).
//! 2. FILTER to top-k by main score ∪ top-k by backup intercept
//!    (`kfsb_shared::select_graph_kfsb_eval_candidates`, k = `fsb_candidates`),
//!    plus an optional stratified top-1-per-unstable-ReLU-layer quota
//!    (`NY_KFSB_LAYER_QUOTA=1` — the probe showed stem-layer candidates like
//!    Relu_13/Relu_5 matter but never crack the global top-k).
//! 3. SIMULATE: both children per candidate via `with_constraint`, bounded for
//!    the WHOLE WAVE through the existing dense-spec domain-batched backward
//!    (`propagate_crown_with_batched_domains_full_specs`) — single-shot, no
//!    β-opt, ONE spec row per call (children are bucketed by their domain's
//!    straggler row, so a chunk of C children costs one C-domain × 1-spec
//!    backward). The `clip_child_node_bounds` research hook is retained, but its
//!    shared production authority gate is quarantined; simulations currently use
//!    inherited bounds unchanged.
//! 4. PICK per domain: argmax of the configured
//!    `kfsb_reduce_op(active_lb, inactive_lb)` on that domain's straggler row,
//!    with a main-score tiebreak. An INFEASIBLE child counts as `+inf` (that
//!    side is empty ⇒ the split is one-sided and free); an
//!    EVAL-FAILED child counts as `-inf`; a candidate with both sides `-inf`
//!    is skipped.
//! 5. COMMIT: at adaptive depth one, use the winner's already-built children.
//!    At depth d>1, put that winner first, append the next distinct simulated
//!    candidates, and build the complete feasible 2^d truth table from the
//!    untouched parent before the normal child pipeline.
//!
//! The retired M27/M28 dense experiment is now strictly advice-only. When its
//! legacy gates are enabled, each authoritative first-child fixpoint is reduced
//! while borrowed to one scalar ambiguity proxy in a fixed inline store. The
//! observer may emit diagnostics, but it cannot return a root identity, replay
//! receipt, bound, child, or verdict; incomplete capture preserves the exact
//! historical one-step winner.
//!
//! The typed July-2026 experiment is a separate, default-off lane. Its
//! published defaults price exactly 15 main-BaBSR roots during the first five
//! canonical outer BaB waves for one deterministic frontier-worst parent,
//! using independently refreshed second BaBSR scores and the f64 paper
//! recurrence with λ=0.5. It reuses M27's identity-only authority boundary but
//! never constructs or commits private second-level leaves and never performs
//! phase fixing. Phase 1 is lower-bound-only: upper-bound verification declines
//! typed advice because it would require a separate direction-normalized prep
//! to preserve the historical selector byte-for-byte.
//!
//! By default this remains ADVISORY-ONLY ⇒ SOUNDNESS-FREE: everything here
//! only chooses WHICH neuron to split; the committed children flow through the
//! same bounding/verdict pipeline as the advisory path. The typed, default-off
//! certificate-reuse policy (exact env override
//! `NY_MO_KFSB_CERT_REUSE=1`) may additionally retain each historical
//! simulation's certified lower endpoint as one scalar and intersect it into a
//! committed lower-bound-mode leaf. Its subset/partition proof is documented at
//! [`apply_kfsb_reusable_lower_certificate`]; private/paper simulations, stale
//! deadlines, malformed bounds, and upper-bound mode cannot publish.
//!
//! COST per wave: `Σ_d 2·|candidates_d|` simulated children (≤ `2·(2k + L)`
//! per domain, L = unstable-layer count under the quota), each bounded by a
//! 1-spec-row share of a chunked batched backward — vs the main pipeline's
//! `2·|domains|` children × |union unverified specs| rows. With k=7 the
//! selection pass costs roughly `k×(1/S)` of the main child pass (S = live
//! spec rows), i.e. comparable to it on cifar100 where S ≈ 8.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomains};
use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::config::{
    DepthTwoBranchLookaheadConfig, DepthTwoBranchLookaheadMode, DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES,
};
use crate::beta_crown::domain::{
    GraphBabDomain, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain, ObjectiveAggregation,
};
use crate::{GraphNetwork, NETWORK_INPUT};

use super::super::super::super::branching::kfsb_shared::{
    kfsb_reduce, select_graph_kfsb_eval_candidates, select_graph_kfsb_eval_candidates_exact_total,
    GraphKfsbCandidate,
};
use super::super::super::super::BetaCrownVerifier;
use super::super::shared::{build_spec_matrix, spec_bounds_to_vec};
use super::batched_dense_specs::{clip_interm_resnet_enabled, graph_bab_domain_shim};
use super::children::{
    KfsbCertEffect, KfsbCertReceipt, KfsbCertScope, KFSB_CERT_PARENT_ID_MAX_BYTES,
};
use super::multi_depth::{
    cap_multi_objective_parent_depth, cap_multi_objective_wave_depth,
    expand_multi_objective_truth_table, multi_depth_plan_is_complete,
};

const COMPLETE_CLIP_DECISION_AUTHORITY_RESERVE: std::time::Duration =
    std::time::Duration::from_secs(5);

/// One wave's single captured certificate-policy decision.
///
/// `Some` means both the typed/env policy and a still-live authoritative
/// deadline admitted certificate work. The selector never re-reads either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KfsbCertAuthority {
    deadline: std::time::Instant,
}

fn capture_kfsb_cert_authority(
    armed: bool,
    authority_deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> Option<KfsbCertAuthority> {
    let deadline = armed.then_some(authority_deadline).flatten()?;
    (now < deadline).then_some(KfsbCertAuthority { deadline })
}

fn probe_kfsb_cert_authority(
    probe: bool,
    armed: bool,
    authority: Option<KfsbCertAuthority>,
    domains: usize,
    committed: usize,
    receipts: usize,
) {
    if probe && armed {
        eprintln!(
            "[kfsb-cert-authority] armed=1 live={} domains={} committed={} receipts={}",
            usize::from(authority.is_some()),
            domains,
            committed,
            receipts,
        );
    }
}

/// Admit one simulated proof-side endpoint.
///
/// `completed_at < authority_deadline` is deliberately strict. The simulator
/// checks the deadline before a chunk, but a backend call may finish after it;
/// such a result remains usable by the historical advisory ranker and carries
/// no certificate authority. Only lower-bound mode is supported because a
/// split carried by beta certifies the lower endpoint, not the returned upper.
fn admit_kfsb_sim_lower_certificate(
    authority: Option<KfsbCertAuthority>,
    verify_upper: bool,
    paper_only: bool,
    completed_at: std::time::Instant,
    lower: f32,
    upper: f32,
) -> Option<f32> {
    let authority = authority?;
    (!verify_upper
        && !paper_only
        && completed_at < authority.deadline
        && lower.is_finite()
        && upper.is_finite()
        && lower <= upper)
        .then_some(lower)
}

/// Return the latest deadline private Complete Clipping decision work may use
/// while preserving the full authority reserve. Equality is already exhausted:
/// callers need strictly positive time in which to begin private work.
pub(super) fn complete_clip_decision_scoring_deadline(
    now: std::time::Instant,
    authority_deadline: std::time::Instant,
) -> Option<std::time::Instant> {
    let scoring_deadline =
        authority_deadline.checked_sub(COMPLETE_CLIP_DECISION_AUTHORITY_RESERVE)?;
    (now < scoring_deadline).then_some(scoring_deadline)
}

fn complete_clip_decision_capture_probe_enabled() -> bool {
    kfsb_probe_enabled()
        || std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1")
}

fn complete_clip_has_full_spec_las(
    caches: &[Option<Arc<crate::batched_domain::CachedLinearBounds>>],
    expected_specs: usize,
) -> bool {
    expected_specs > 0 && caches.len() == expected_specs && caches.iter().all(Option::is_some)
}

#[cfg(test)]
#[test]
fn complete_clip_full_spec_cache_rejects_length_mismatch_and_missing_rows() {
    use crate::batched_domain::CachedLinearBounds;

    let full = vec![
        Some(Arc::new(CachedLinearBounds::default())),
        Some(Arc::new(CachedLinearBounds::default())),
    ];
    assert!(complete_clip_has_full_spec_las(&full, 2));
    assert!(!complete_clip_has_full_spec_las(&full, 1));
    assert!(!complete_clip_has_full_spec_las(&full, 3));
    assert!(!complete_clip_has_full_spec_las(
        &[Some(Arc::new(CachedLinearBounds::default())), None],
        2
    ));
    assert!(!complete_clip_has_full_spec_las(&[], 0));
}

#[allow(clippy::too_many_arguments)]
fn record_complete_clip_decision_capture(
    status: &'static str,
    reason: &'static str,
    source: &'static str,
    domains: usize,
    specs: usize,
    min_depth: usize,
    max_depth: usize,
    published: usize,
    elapsed: std::time::Duration,
) {
    tracing::debug!(
        status,
        reason,
        source,
        domains,
        specs,
        min_depth,
        max_depth,
        published,
        wall_ms = elapsed.as_millis(),
        "complete-clip decision precompute"
    );
    if complete_clip_decision_capture_probe_enabled() {
        eprintln!(
            "[complete-clip-decision-precompute] status={status} reason={reason} source={source} \
             domains={domains} specs={specs} min_depth={min_depth} max_depth={max_depth} \
             published={published} wall_ms={}",
            elapsed.as_millis()
        );
    }
}

/// Master gate ENV override (kill switch): `NY_MO_KFSB=1` force-ARMS the
/// wave-batched kFSB selector on the multi-objective lane, `NY_MO_KFSB=0`
/// force-DISARMS it. The self-contained alias `NY_BRANCH_KFSB_CHILDSIM=1` also
/// force-ARMS (so the child-sim scoring at the selector actually runs). Any
/// other value (or unset, no childsim) ⇒ `None`, so the gate falls back to
/// `config.use_kfsb_multi_branching` (the preset opt-in). The env thus
/// overrides the preset in EITHER direction, preserving the A/B kill switch.
fn kfsb_multi_env_override() -> Option<bool> {
    match std::env::var("NY_MO_KFSB").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ if kfsb_childsim_gate_enabled() => Some(true),
        _ => None,
    }
}

/// #branch-kfsb-childsim: a self-contained single switch for the wave-batched
/// kFSB CHILD-SIMULATION branch selector. `NY_BRANCH_KFSB_CHILDSIM=1` arms the
/// exact same lane as `NY_MO_KFSB=1` — it scores each candidate split by the
/// ACTUAL bound on BOTH children (simulated in one wave-batched backward) and
/// picks the argmax worst-child bound, instead of the objective-blind intercept
/// `(-l·u)/(u-l)` proxy the default `select_graph_branch_multi` falls back to
/// under the auto-selected `Kfsb` heuristic.
///
/// MEASURED (cifar100 CIFAR100_resnet_medium prop_4429 @100 s, cuda): the
/// intercept default splits ONE layer (Relu_57) for all 1023 domains, wastes
/// 40 % of splits (lift < 1e-3), and verifies 0 sub-domains; this lane wastes
/// 0 % and verifies 10 — the frontier-worst climbs from −0.965 to −0.79. When
/// the candidate budget is not explicitly overridden (`NY_MO_KFSB_K` unset), it
/// pins `k = 2` — the measured throughput/quality sweet spot (k=2 explores 4×
/// the domains of k=7 at an identical verified count and frontier-worst). It
/// only arms where a kFSB heuristic is active (auto-branching's choice for every
/// high-dim conv net: cifar100 / tinyimagenet / traffic_signs / …). Default OFF
/// ⇒ byte-identical. Split selection is advisory; the independent typed,
/// default-off certificate-reuse policy may additionally publish a
/// strictly-authorized scalar lower proof as documented at
/// [`apply_kfsb_reusable_lower_certificate`].
fn kfsb_childsim_gate_enabled() -> bool {
    std::env::var("NY_BRANCH_KFSB_CHILDSIM").ok().as_deref() == Some("1")
}

/// Resolve the candidate budget once using the selector's historical
/// precedence. Typed gate checks call this only after the policy has already
/// passed its default-off round gate, so an omitted typed policy does not add
/// any environment reads to the legacy path.
fn kfsb_multi_candidate_count(configured: usize) -> usize {
    std::env::var("NY_MO_KFSB_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            if kfsb_childsim_gate_enabled() {
                2
            } else {
                configured
            }
        })
}

/// Stratified layer quota (`NY_KFSB_LAYER_QUOTA=1`, dark): additionally admit
/// each unstable ReLU layer's top-1 main-score candidate to the eval set.
fn kfsb_layer_quota_enabled() -> bool {
    std::env::var("NY_KFSB_LAYER_QUOTA").ok().as_deref() == Some("1")
}

/// One-line per-wave probe (`NY_MO_KFSB_PROBE=1`).
pub(super) fn kfsb_probe_enabled() -> bool {
    std::env::var("NY_MO_KFSB_PROBE").ok().as_deref() == Some("1")
}

/// Winner-parity prescore (`NY_MO_KFSB_CACHED_LA=1`, dark): use the exact
/// per-objective lower-A coefficients captured by the domain's preceding CROWN
/// pass.  Unset/`0` preserves the historical fixed-slope proxy byte-for-byte.
fn kfsb_cached_la_enabled() -> bool {
    resolve_kfsb_cached_la_enabled(std::env::var("NY_MO_KFSB_CACHED_LA").ok().as_deref())
}

pub(super) fn resolve_kfsb_cached_la_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Default-off precision-consistent f64 kFSB observer. Only the exact `"1"`
/// spelling arms it; malformed values fail closed to the historical path.
///
/// This does not arm kFSB itself. The hook is reachable only from the existing
/// wave-batched kFSB selector, so a sealed run must separately enable that lane
/// through its preset or `NY_MO_KFSB=1`.
fn kfsb_f64_shadow_enabled() -> bool {
    resolve_kfsb_f64_shadow_enabled(std::env::var("NY_MO_KFSB_F64_SHADOW").ok().as_deref())
}

pub(super) fn resolve_kfsb_f64_shadow_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Match the scalar quantity scored by `BetaCrownConfig::child_bound_value`:
/// lower(c) in the normal mode, and -upper(c) = lower(-c) in verify-upper
/// mode.
pub(super) fn kfsb_f64_shadow_objective(
    objective: &[f32],
    verify_upper_bound: bool,
) -> std::borrow::Cow<'_, [f32]> {
    if verify_upper_bound {
        std::borrow::Cow::Owned(
            objective
                .iter()
                .copied()
                .map(|coefficient| -coefficient)
                .collect(),
        )
    } else {
        std::borrow::Cow::Borrowed(objective)
    }
}

/// The one-shot observer gives its certified fold phase a five-second private
/// deadline and starts only when ten seconds remain reserved for the
/// authoritative verifier. The fold polls that deadline inside its expensive
/// scalar Linear/Conv loops. Segment extraction is a fixed at-most-six-call
/// prelude rather than a preemptible kernel, so every extraction is checked
/// both before entry and before its fold result may be retained.
const KFSB_F64_SHADOW_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);
const KFSB_F64_AUTHORITY_RESERVE: std::time::Duration = std::time::Duration::from_secs(10);

pub(super) fn kfsb_f64_shadow_deadline(
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    let shadow_deadline = now.checked_add(KFSB_F64_SHADOW_BUDGET)?;
    if let Some(authority_deadline) = authority_deadline {
        let latest_finish = authority_deadline.checked_sub(KFSB_F64_AUTHORITY_RESERVE)?;
        if shadow_deadline > latest_finish {
            return None;
        }
    }
    Some(shadow_deadline)
}

/// Whether the private f64 observer may start or retain work. The first
/// condition enforces the fold phase's own five-second envelope; the second
/// requires ten seconds to remain for the authoritative BaB loop at each
/// admission/retention check. The legacy segment-extraction prelude is not
/// preemptible, so this reserve is deliberately best-effort across extraction.
pub(super) fn kfsb_f64_shadow_budget_available(
    now: std::time::Instant,
    shadow_deadline: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> bool {
    now < shadow_deadline
        && authority_deadline.is_none_or(|deadline| {
            now.checked_add(KFSB_F64_AUTHORITY_RESERVE)
                .is_some_and(|reserved_until| reserved_until < deadline)
        })
}

/// Observation-only winner oracle.  It reports every evaluated candidate's
/// direct-C child values plus the `Min` and `Max` winners without changing the
/// configured pick.
fn kfsb_winner_probe_enabled() -> bool {
    std::env::var("NY_MO_KFSB_WINNER_PROBE").ok().as_deref() == Some("1")
}

fn kfsb_winner_probe_domains() -> usize {
    std::env::var("NY_MO_KFSB_WINNER_PROBE_DOMAINS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
}

/// Effective reduce op for the wave-batched multi-objective kFSB lane.
///
/// Match αβ-CROWN by honoring the configured `bab.branching.reduceop`; the
/// VNN-COMP CIFAR-100 and TinyImageNet presets both select `Max`. The optional
/// `NY_MO_KFSB_REDUCE=min|max` override remains available for controlled A/B
/// measurements. Unknown override values leave the configured policy intact.
pub(super) fn resolve_kfsb_multi_reduce_op(
    configured: crate::beta_crown::KfsbReduceOp,
    env_override: Option<&str>,
) -> crate::beta_crown::KfsbReduceOp {
    match env_override {
        Some("max") => crate::beta_crown::KfsbReduceOp::Max,
        Some("min") => crate::beta_crown::KfsbReduceOp::Min,
        _ => configured,
    }
}

/// The lane's effective reduce op, reading the `NY_MO_KFSB_REDUCE` A/B env.
fn kfsb_multi_reduce_op(
    configured: crate::beta_crown::KfsbReduceOp,
) -> crate::beta_crown::KfsbReduceOp {
    resolve_kfsb_multi_reduce_op(
        configured,
        std::env::var("NY_MO_KFSB_REDUCE").ok().as_deref(),
    )
}

/// Chunk width for the simulation's batched backward calls
/// (`NY_MO_KFSB_CHUNK=<n>`, default 64 — the same width as the main lane's
/// GPU single-pass chunk, bounding both memory and deadline overrun).
fn kfsb_sim_chunk() -> usize {
    std::env::var("NY_MO_KFSB_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// Hard resource envelope for the optional attribution consumer. These caps
/// are typed constants, not environment policy: malformed or enormous kFSB
/// overrides can disable the attribution experiment, but can never enlarge its
/// retained portfolios, private child set, or one-call batch.
const ATTRIBUTION_MAX_RANKED_CANDIDATES: usize = 4_096;
const ATTRIBUTION_MAX_K: usize = 10;
const ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES: usize = 30;
const ATTRIBUTION_MAX_UNION_CANDIDATES: usize = 40;
const ATTRIBUTION_MAX_DISTINGUISHING_CANDIDATES: usize = 20;
const ATTRIBUTION_MAX_APPENDED_CANDIDATES: usize = 10;
const ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS: usize = 20;
const ATTRIBUTION_MAX_PRIVATE_MEMBERS: usize = 80;
const ATTRIBUTION_MAX_PRIVATE_CHUNK: usize = 16;
const ATTRIBUTION_DIAG_MAX_ELIGIBLE_ROUNDS: usize = 1;

#[inline]
fn attribution_private_chunk(requested: usize) -> usize {
    requested.clamp(1, ATTRIBUTION_MAX_PRIVATE_CHUNK)
}

/// Default-off, observation-only adaptive-depth probe. The exact `"1"`
/// spelling is intentional: malformed values fail closed to the historical
/// path.
fn adaptive_depth_shadow_enabled() -> bool {
    resolve_adaptive_depth_shadow_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SHADOW").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_shadow_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Legacy default-off SELECT spelling. It now arms captured-fixpoint
/// observation only; malformed values retain the historical path.
fn adaptive_depth_select_enabled() -> bool {
    resolve_adaptive_depth_select_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SELECT").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_select_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Preserve the retired COMMIT gate's deterministic depth-two scheduling.
/// The captured-fixpoint proxy itself does not evaluate this depth.
const ADAPTIVE_DEPTH_LEGACY_COMMIT_HORIZON: usize = 2;

/// Legacy default-off COMMIT spelling. It now arms observation only and cannot
/// publish a root, replay receipt, child, bound, or verdict.
fn adaptive_depth_commit_enabled() -> bool {
    resolve_adaptive_depth_commit_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_COMMIT").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_commit_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// The scalar observation gets at most one second and may start only while five
/// seconds remain reserved for the authoritative BaB loop.
const ADAPTIVE_DEPTH_SHADOW_BUDGET: std::time::Duration = std::time::Duration::from_secs(1);
const ADAPTIVE_DEPTH_AUTHORITY_RESERVE: std::time::Duration = std::time::Duration::from_secs(5);

pub(super) fn adaptive_depth_shadow_deadline(
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    let shadow_deadline = now.checked_add(ADAPTIVE_DEPTH_SHADOW_BUDGET)?;
    if let Some(authority_deadline) = authority_deadline {
        let latest_finish = authority_deadline.checked_sub(ADAPTIVE_DEPTH_AUTHORITY_RESERVE)?;
        if shadow_deadline > latest_finish {
            return None;
        }
    }
    Some(shadow_deadline)
}

/// Whether optional shadow work may continue right now.  The private
/// one-second deadline bounds the whole observer (including BaBSR rescoring),
/// while the second condition prevents any new side from consuming the five
/// seconds reserved for authoritative search.
pub(super) fn adaptive_depth_shadow_budget_available(
    now: std::time::Instant,
    shadow_deadline: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> bool {
    now < shadow_deadline
        && authority_deadline.is_none_or(|deadline| {
            now.checked_add(ADAPTIVE_DEPTH_AUTHORITY_RESERVE)
                .is_some_and(|reserved_until| reserved_until < deadline)
        })
}

/// Claim the verifier-local M27/M28 observation attempt exactly once. Admission
/// declines consume the same one-shot as completed or deadline-refused work, so
/// a later wave cannot silently move the observed/acted decision point.
pub(super) fn claim_adaptive_depth_attempt(fired: &std::sync::atomic::AtomicBool) -> bool {
    fired
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
}

/// For the typed experiment, one admission created at wave-selector entry
/// bounds its entire optional lifetime: exact-portfolio construction,
/// paper-only first-child simulation, and branch-specific second scoring. The
/// later observation-only attribution diagnostic receives a separate admission
/// only after typed/adaptive work has completed; it cannot reset this clock.
///
/// Keeping this value immutable is deliberate. No later phase may reset the
/// one-second clock after earlier optional work has already consumed time.
#[derive(Clone, Copy, Debug)]
pub(super) struct DepthTwoLookaheadBudget {
    started_at: std::time::Instant,
    private_deadline: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
}

impl DepthTwoLookaheadBudget {
    pub(super) fn admit(
        now: std::time::Instant,
        authority_deadline: Option<std::time::Instant>,
    ) -> Option<Self> {
        let private_deadline = adaptive_depth_shadow_deadline(now, authority_deadline)?;
        Some(Self {
            started_at: now,
            private_deadline,
            authority_deadline,
        })
    }

    #[inline]
    pub(super) fn available_at(self, now: std::time::Instant) -> bool {
        adaptive_depth_shadow_budget_available(now, self.private_deadline, self.authority_deadline)
    }

    #[inline]
    fn available_now(self) -> bool {
        self.available_at(std::time::Instant::now())
    }

    #[cfg(test)]
    pub(super) fn expired_at(now: std::time::Instant) -> Self {
        Self {
            started_at: now,
            private_deadline: now,
            authority_deadline: None,
        }
    }
}

/// A complete second-level score for one first-level side. `Infeasible`
/// represents a half-space proved empty by `with_constraint`; it is not a
/// propagated infinity and is the only infinity admitted by the recurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum DepthTwoLookaheadSideScore {
    Infeasible,
    Finite(f64),
}

#[inline]
fn depth_two_lookahead_policy_supported(policy: DepthTwoBranchLookaheadConfig) -> bool {
    (1..=DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES).contains(&policy.candidates)
        && (1..=crate::beta_crown::config::DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS)
            .contains(&policy.top_rounds)
        && policy.discount.is_finite()
        && (0.0..=1.0).contains(&policy.discount)
}

/// Paper recurrence
///
/// `one_step + λ * active*inactive/(active+inactive+1)`.
///
/// The algebraically equivalent `lo / (1 + (lo + 1) / hi)` form avoids
/// overflowing the product for large finite inputs. One infeasible first-level
/// side uses the mathematical limit (the finite side score); two infeasible
/// sides yield positive infinity. Malformed, negative, or non-finite advice
/// fails closed.
pub(super) fn depth_two_lookahead_score(
    one_step: f64,
    active: DepthTwoLookaheadSideScore,
    inactive: DepthTwoLookaheadSideScore,
    discount: f64,
) -> Option<f64> {
    if !discount.is_finite() || !(0.0..=1.0).contains(&discount) {
        return None;
    }
    let finite = |value: f64| value.is_finite() && value >= 0.0;
    let balance = match (active, inactive) {
        (DepthTwoLookaheadSideScore::Infeasible, DepthTwoLookaheadSideScore::Infeasible) => {
            return (one_step == f64::INFINITY).then_some(f64::INFINITY);
        }
        (DepthTwoLookaheadSideScore::Infeasible, DepthTwoLookaheadSideScore::Finite(value))
        | (DepthTwoLookaheadSideScore::Finite(value), DepthTwoLookaheadSideScore::Infeasible) => {
            if !finite(value) {
                return None;
            }
            value
        }
        (
            DepthTwoLookaheadSideScore::Finite(active),
            DepthTwoLookaheadSideScore::Finite(inactive),
        ) => {
            if !finite(active) || !finite(inactive) {
                return None;
            }
            let lo = active.min(inactive);
            let hi = active.max(inactive);
            if hi == 0.0 {
                0.0
            } else {
                lo / (1.0 + (lo + 1.0) / hi)
            }
        }
    };
    if !one_step.is_finite() || !balance.is_finite() {
        return None;
    }
    let score = one_step + discount * balance;
    score.is_finite().then_some(score)
}

/// Select a complete depth-2 portfolio winner while preserving the historical
/// one-step winner on exact score ties. A missing/NaN score, duplicate
/// candidate, incomplete portfolio, or absent historical root declines advice;
/// positive infinity remains reserved for an exact all-infeasible root.
pub(super) fn select_complete_depth_two_lookahead(
    scores: &[(usize, Option<f64>)],
    expected: usize,
    historical_candidate: usize,
) -> Option<(usize, f64)> {
    if expected == 0 || scores.len() != expected {
        return None;
    }
    let mut seen = std::collections::HashSet::with_capacity(expected);
    let mut complete = Vec::with_capacity(expected);
    for &(candidate, score) in scores {
        let score = score?;
        if score == f64::NEG_INFINITY || score.is_nan() || !seen.insert(candidate) {
            return None;
        }
        complete.push((candidate, score));
    }
    let historical_score = complete
        .iter()
        .find_map(|&(candidate, score)| (candidate == historical_candidate).then_some(score))?;
    let mut best = (historical_candidate, historical_score);
    for &(candidate, score) in &complete {
        if score > best.1 {
            best = (candidate, score);
        }
    }
    Some(best)
}

/// Deterministically choose the lowest-bound parent, breaking exact ties by
/// its stable wave slot. NaN entries are ineligible.
pub(super) fn select_depth_two_frontier_worst_slot(
    slot_values: impl IntoIterator<Item = (usize, f32)>,
) -> Option<usize> {
    slot_values
        .into_iter()
        .filter(|(_, value)| !value.is_nan())
        .min_by(|(a_slot, a_value), (b_slot, b_value)| {
            a_value.total_cmp(b_value).then_with(|| a_slot.cmp(b_slot))
        })
        .map(|(slot, _)| slot)
}

/// Select the decision-relevant worst unverified objective row.
///
/// Lower-bound verification minimizes `lower`; upper-bound verification
/// minimizes the direction-normalized value `-upper` (equivalently, it picks
/// the highest raw upper). NaNs are pessimistically normalized to `-inf`, as
/// they are everywhere else in this advisory selector.
#[cfg(test)]
pub(super) fn select_kfsb_straggler(
    objective_bounds: &[(f32, f32)],
    verified: &[bool],
    verify_upper_bound: bool,
) -> Option<(usize, f32)> {
    let mut straggler = None;
    for (objective_index, &(lower, upper)) in objective_bounds.iter().enumerate() {
        if verified.get(objective_index).copied().unwrap_or(false) {
            continue;
        }
        let value = if verify_upper_bound { -upper } else { lower };
        let value = if value.is_nan() {
            f32::NEG_INFINITY
        } else {
            value
        };
        if straggler.is_none_or(|(_, worst)| value < worst) {
            straggler = Some((objective_index, value));
        }
    }
    straggler
}

/// Materialize the historical score vector while separately recording whether
/// every unstable identity had an actual BaBSR entry. The zero default remains
/// the legacy lane's exact behavior; typed exact-total advice may use the
/// vector only when `complete` is true, so a missing entry can never become a
/// fabricated zero-scored paper candidate.
pub(super) fn materialize_kfsb_candidates_with_completeness(
    unstable: &[(String, usize)],
    mut score_for: impl FnMut(&(String, usize)) -> Option<(f32, f32)>,
) -> (Vec<GraphKfsbCandidate>, bool) {
    let mut complete = true;
    let candidates = unstable
        .iter()
        .map(|(node_name, neuron_idx)| {
            let (main_score, backup_score) = score_for(&(node_name.clone(), *neuron_idx))
                .unwrap_or_else(|| {
                    complete = false;
                    (0.0, 0.0)
                });
            GraphKfsbCandidate {
                node_name: node_name.clone(),
                neuron_idx: *neuron_idx,
                main_score,
                backup_score,
            }
        })
        .collect();
    (candidates, complete)
}

/// Build a lightweight attribution-primary index order without mutating or
/// deep-cloning the historical main-score candidates consumed by layer quotas
/// and exact-total/depth-two portfolios. Missing/non-finite evidence declines
/// the whole optional view atomically.
pub(super) fn rank_kfsb_candidate_indices_by_attribution(
    main_scored: &[GraphKfsbCandidate],
    mut attribution_score: impl FnMut(&GraphKfsbCandidate) -> Option<f64>,
) -> Option<Vec<usize>> {
    if main_scored.len() > ATTRIBUTION_MAX_RANKED_CANDIDATES {
        return None;
    }
    let mut scores = Vec::with_capacity(main_scored.len());
    for candidate in main_scored {
        let score = attribution_score(candidate)?;
        if !score.is_finite() || score < 0.0 {
            return None;
        }
        scores.push(score);
    }
    let mut ranked: Vec<usize> = (0..main_scored.len()).collect();
    ranked.sort_by(|&a, &b| {
        scores[b]
            .total_cmp(&scores[a])
            .then_with(|| {
                crate::cmp_utils::nan_last_descending_cmp(
                    &main_scored[a].main_score,
                    &main_scored[b].main_score,
                )
            })
            // `main_scored` is already in stable historical main-score order.
            // Preserve it exactly when both optional and main scores tie.
            .then_with(|| a.cmp(&b))
    });
    Some(ranked)
}

/// Attribution changes only the primary top-k channel. Backup top-k remains
/// the historical backup-score channel, and callers retain the main-sorted
/// slice for every independent consumer.
fn select_attribution_primary_kfsb_candidates(
    main_scored: &[GraphKfsbCandidate],
    attribution_order: &[usize],
    k: usize,
) -> Vec<GraphKfsbCandidate> {
    let mut selected = Vec::new();
    let mut seen: std::collections::HashSet<(&str, usize)> = std::collections::HashSet::new();
    for &index in attribution_order.iter().take(k) {
        let candidate = &main_scored[index];
        if seen.insert((candidate.node_name.as_str(), candidate.neuron_idx)) {
            selected.push(candidate.clone());
        }
    }

    let mut backup_order: Vec<usize> = (0..main_scored.len()).collect();
    backup_order.sort_by(|&a, &b| {
        crate::cmp_utils::nan_propagating_cmp(
            &main_scored[a].backup_score,
            &main_scored[b].backup_score,
        )
        // Historical stable sort preserved the already-main-sorted input for
        // backup ties; make that ordering explicit in the index view.
        .then_with(|| a.cmp(&b))
    });
    for index in backup_order.into_iter().take(k) {
        let candidate = &main_scored[index];
        if seen.insert((candidate.node_name.as_str(), candidate.neuron_idx)) {
            selected.push(candidate.clone());
        }
    }
    selected
}

/// Build an attribution-directed portfolio inside the fixed private/advisory
/// envelope. The primary and backup channels may each contribute `k`, and the
/// typed strategy admits at most ten from either channel. Layer
/// quota additions are staged and the whole optional portfolio is discarded
/// if the fixed cap would be crossed.
fn select_bounded_attribution_kfsb_candidates(
    main_scored: &[GraphKfsbCandidate],
    attribution_order: &[usize],
    k: usize,
    layer_quota: bool,
) -> Option<Vec<GraphKfsbCandidate>> {
    if k == 0 || k > ATTRIBUTION_MAX_K {
        return None;
    }
    let mut selected =
        select_attribution_primary_kfsb_candidates(main_scored, attribution_order, k);
    if selected.len() > ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES {
        return None;
    }
    if !layer_quota {
        return Some(selected);
    }

    let mut seen: std::collections::HashSet<(String, usize)> = selected
        .iter()
        .map(|candidate| (candidate.node_name.clone(), candidate.neuron_idx))
        .collect();
    let mut layers_done: std::collections::HashSet<String> = selected
        .iter()
        .map(|candidate| candidate.node_name.clone())
        .collect();
    for candidate in main_scored {
        if !layers_done.insert(candidate.node_name.clone()) {
            continue;
        }
        if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
            if selected.len() == ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES {
                return None;
            }
            selected.push(candidate.clone());
        }
    }
    Some(selected)
}

/// Append only identities not already present in `union`, preserving both the
/// incumbent prefix and the diagnostic portfolio's native order.
fn append_unique_kfsb_candidates(
    union: &mut Vec<GraphKfsbCandidate>,
    additions: &[GraphKfsbCandidate],
) {
    let mut seen: std::collections::HashSet<(String, usize)> = union
        .iter()
        .map(|candidate| (candidate.node_name.clone(), candidate.neuron_idx))
        .collect();
    for candidate in additions {
        if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
            union.push(candidate.clone());
        }
    }
}

/// Recover one portfolio's order in the constructed/simulation union.
/// Candidates whose two child constructions both failed are absent exactly as
/// they would be from a real kFSB preparation in that arm.
fn kfsb_portfolio_indices(
    union: &[GraphKfsbCandidate],
    portfolio: &[GraphKfsbCandidate],
) -> Vec<usize> {
    portfolio
        .iter()
        .filter_map(|candidate| {
            union.iter().position(|member| {
                member.node_name == candidate.node_name && member.neuron_idx == candidate.neuron_idx
            })
        })
        .collect()
}

fn install_incomplete_attribution_diag(prep: &mut DomainPrep, plan: &AttributionDiagOverlayPlan) {
    let within_caps = attribution_diag_plan_within_caps(plan);
    prep.attribution_diag = Some(AttributionDiagPrep {
        coverage: if plan.coverage == AttributionDiagCoverage::Complete && !within_caps {
            AttributionDiagCoverage::ResourceCapped
        } else {
            plan.coverage
        },
        overlay_complete: false,
        historical_candidates: if within_caps {
            kfsb_portfolio_indices(&prep.candidates, &plan.historical_candidates)
        } else {
            Vec::new()
        },
        attribution_candidates: if within_caps {
            kfsb_portfolio_indices(&prep.candidates, &plan.attribution_candidates)
        } else {
            Vec::new()
        },
        distinguishing_candidates: Vec::new(),
    });
}

fn install_resource_capped_attribution_diag(
    prep: &mut DomainPrep,
    plan: &AttributionDiagOverlayPlan,
) {
    install_incomplete_attribution_diag(prep, plan);
    if let Some(diag) = prep.attribution_diag.as_mut() {
        diag.coverage = AttributionDiagCoverage::ResourceCapped;
    }
}

#[cfg(test)]
#[test]
fn attribution_ranking_is_complete_atomic_and_does_not_reorder_main_consumers() {
    let candidate = |node: &str, idx: usize, main: f32, backup: f32| GraphKfsbCandidate {
        node_name: node.to_string(),
        neuron_idx: idx,
        main_score: main,
        backup_score: backup,
    };
    let main = vec![
        candidate("a", 0, 3.0, 0.0),
        // Historical backup scores are better when lower/more negative.
        candidate("b", 0, 2.0, -9.0),
        candidate("c", 0, 1.0, 1.0),
    ];

    let ranked = rank_kfsb_candidate_indices_by_attribution(&main, |candidate| {
        Some(match candidate.node_name.as_str() {
            "c" => 5.0,
            "a" => 1.0,
            _ => 0.0,
        })
    })
    .expect("complete prior");
    assert_eq!(
        ranked
            .iter()
            .map(|&index| main[index].node_name.as_str())
            .collect::<Vec<_>>(),
        ["c", "a", "b"]
    );
    assert!(
        rank_kfsb_candidate_indices_by_attribution(&main, |candidate| {
            (candidate.node_name != "b").then_some(1.0)
        })
        .is_none()
    );

    let all_zero = rank_kfsb_candidate_indices_by_attribution(&main, |_| Some(0.0))
        .expect("covered all-zero prior");
    assert_eq!(
        all_zero
            .iter()
            .map(|&index| main[index].node_name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "c"],
        "zero attribution ties must preserve historical main order"
    );

    let eval = select_attribution_primary_kfsb_candidates(&main, &ranked, 1);
    let historical = select_graph_kfsb_eval_candidates(&main, 1, true);
    assert_eq!(
        historical
            .iter()
            .map(|candidate| candidate.node_name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"],
        "fixture must exercise the historical lower-is-better backup channel"
    );
    assert_eq!(
        eval.iter()
            .map(|candidate| candidate.node_name.as_str())
            .collect::<Vec<_>>(),
        ["c", "b"],
        "attribution owns primary top-k while historical backup top-k remains composed"
    );
    assert_eq!(
        select_depth_two_root_portfolio(&main, true, 2)
            .expect("complete main portfolio")
            .iter()
            .map(|candidate| candidate.node_name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"],
        "depth-two/exact-total must retain the untouched main-score view"
    );
    assert_eq!(main[0].node_name, "a", "main view must remain untouched");
}

/// Build the exact-total paper root portfolio only from a complete score map.
/// Merely having `total` finite values after legacy zero-filling is not enough:
/// an absent BaBSR entry could otherwise outrank a real negative score.
pub(super) fn select_depth_two_root_portfolio(
    scored: &[GraphKfsbCandidate],
    score_map_complete: bool,
    total: usize,
) -> Option<Vec<GraphKfsbCandidate>> {
    if !score_map_complete {
        return None;
    }
    let selected = select_graph_kfsb_eval_candidates_exact_total(scored, total);
    (selected.len() == total).then_some(selected)
}

/// Deterministically rank unique candidates by their already-computed one-step
/// child score. No new bounds are produced here: this is a pure view over the
/// scored kFSB simulations. Exact score, main prescore, node name, neuron index,
/// then original position form a total deterministic ordering.
pub(super) fn rank_adaptive_depth_candidates(
    candidates: &[GraphKfsbCandidate],
    side_values: &[(f32, f32)],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = candidates
        .iter()
        .zip(side_values)
        .enumerate()
        .filter_map(|(idx, (_candidate, &(active, inactive)))| {
            if active == f32::NEG_INFINITY && inactive == f32::NEG_INFINITY {
                return None;
            }
            let score = kfsb_reduce(reduce_op, active, inactive);
            (!score.is_nan()).then_some((idx, score))
        })
        .collect();
    ranked.sort_by(|(a_idx, a_score), (b_idx, b_score)| {
        let a = &candidates[*a_idx];
        let b = &candidates[*b_idx];
        let a_main = if a.main_score.is_nan() {
            f32::NEG_INFINITY
        } else {
            a.main_score
        };
        let b_main = if b.main_score.is_nan() {
            f32::NEG_INFINITY
        } else {
            b.main_score
        };
        b_score
            .total_cmp(a_score)
            .then_with(|| b_main.total_cmp(&a_main))
            .then_with(|| a.node_name.cmp(&b.node_name))
            .then_with(|| a.neuron_idx.cmp(&b.neuron_idx))
            .then_with(|| a_idx.cmp(b_idx))
    });

    let mut seen: std::collections::HashSet<(String, usize)> = std::collections::HashSet::new();
    ranked.retain(|(idx, _)| {
        let candidate = &candidates[*idx];
        seen.insert((candidate.node_name.clone(), candidate.neuron_idx))
    });
    ranked
}

pub(super) type AdaptiveDepthShadowNodeBounds = HashMap<String, Arc<BoundedTensor>>;

pub(super) const ADAPTIVE_DEPTH_SHADOW_ROOTS: usize = 3;
const ADAPTIVE_DEPTH_SHADOW_MAX_CAPTURE_CANDIDATES: usize = 64;
const ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS: usize = 2 * ADAPTIVE_DEPTH_SHADOW_MAX_CAPTURE_CANDIDATES;
const ADAPTIVE_DEPTH_PROXY_MAX_HEAP_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdaptiveDepthPrivatePeakDecline {
    ArithmeticOverflow,
    PeakCapExceeded,
}

/// Checked receipt for the advice-only proxy's bounded metadata allocations.
/// A caller lists all representations that will coexist before constructing
/// them; overflow and peaks above the immutable cap are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AdaptiveDepthPrivatePeakLedger {
    cap_bytes: usize,
    admitted_peak_bytes: usize,
}

impl AdaptiveDepthPrivatePeakLedger {
    pub(super) const fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            admitted_peak_bytes: 0,
        }
    }

    pub(super) fn admit<I>(
        &mut self,
        simultaneously_live: I,
    ) -> Result<usize, AdaptiveDepthPrivatePeakDecline>
    where
        I: IntoIterator<Item = usize>,
    {
        let mut stage_bytes = 0usize;
        for component in simultaneously_live {
            stage_bytes = stage_bytes
                .checked_add(component)
                .ok_or(AdaptiveDepthPrivatePeakDecline::ArithmeticOverflow)?;
        }
        if stage_bytes > self.cap_bytes {
            return Err(AdaptiveDepthPrivatePeakDecline::PeakCapExceeded);
        }
        self.admitted_peak_bytes = self.admitted_peak_bytes.max(stage_bytes);
        Ok(stage_bytes)
    }

    pub(super) const fn admitted_peak_bytes(self) -> usize {
        self.admitted_peak_bytes
    }
}

struct AdaptiveDepthShadowCaptureSlot {
    sim_index: usize,
    proxy_score: Option<f32>,
}

/// Reduce one authoritative constrained-forward result while it is borrowed.
///
/// The proxy is the largest normalized distance of an unconstrained unstable
/// activation from its nearer phase boundary. It is deliberately advisory:
/// it authenticates that the whole activation census is finite and ordered,
/// but it is not a propagated second-child bound.
fn adaptive_depth_fixpoint_proxy_from_bounds<B>(
    graph: &GraphNetwork,
    relu_nodes: &[String],
    domain: &MultiObjectiveGraphBabDomain,
    node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    mut budget_available: B,
) -> Option<f32>
where
    B: FnMut() -> bool,
{
    if !budget_available() {
        return None;
    }
    let mut proxy = 0.0_f32;
    for node_name in relu_nodes {
        if !budget_available() {
            return None;
        }
        let node = graph.nodes.get(node_name)?;
        if !matches!(&node.layer, crate::Layer::ReLU(_) | crate::Layer::Sign(_)) {
            return None;
        }
        let pre_name = node.inputs.first()?;
        let bounds = if pre_name == NETWORK_INPUT {
            domain.input_bounds()
        } else {
            node_bounds.get(pre_name)?.as_ref()
        };
        if bounds.lower().len() != bounds.upper().len() {
            return None;
        }
        for (neuron_idx, (&lower, &upper)) in
            bounds.lower().iter().zip(bounds.upper().iter()).enumerate()
        {
            if !budget_available() {
                return None;
            }
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return None;
            }
            if domain
                .history()
                .is_constrained(node_name, neuron_idx)
                .is_some()
                || !(lower < 0.0 && upper > 0.0)
            {
                continue;
            }
            let width = upper - lower;
            if !width.is_finite() || width <= 0.0 {
                return None;
            }
            let ambiguity = (-lower).min(upper) / width;
            if !ambiguity.is_finite() {
                return None;
            }
            proxy = proxy.max(ambiguity);
        }
    }
    budget_available().then_some(proxy)
}

/// Fixed-size capture store. Authoritative result maps are reduced while
/// borrowed to one scalar; no tensor, map, String, or Arc lifetime is extended.
pub(super) struct AdaptiveDepthShadowCapture {
    prep_index: usize,
    slots: [Option<AdaptiveDepthShadowCaptureSlot>; ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS],
}

impl AdaptiveDepthShadowCapture {
    #[cfg(test)]
    pub(super) fn from_candidate_indices(
        prep_index: usize,
        sides: &[[SideSlot; 2]],
        candidate_indices: &[usize],
        sims_len: usize,
    ) -> Option<Self> {
        let candidate_indices: [usize; ADAPTIVE_DEPTH_SHADOW_ROOTS] =
            candidate_indices.try_into().ok()?;
        if candidate_indices[0] == candidate_indices[1]
            || candidate_indices[0] == candidate_indices[2]
            || candidate_indices[1] == candidate_indices[2]
        {
            return None;
        }
        let mut slots: [Option<AdaptiveDepthShadowCaptureSlot>;
            ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS] = std::array::from_fn(|_| None);
        let mut slot_index = 0;
        for candidate_index in candidate_indices {
            for side in sides.get(candidate_index)? {
                let SideSlot::Sim(sim_index) = side else {
                    continue;
                };
                if *sim_index >= sims_len
                    || slots[..slot_index].iter().any(|slot| {
                        slot.as_ref()
                            .is_some_and(|slot| slot.sim_index == *sim_index)
                    })
                {
                    return None;
                }
                *slots.get_mut(slot_index)? = Some(AdaptiveDepthShadowCaptureSlot {
                    sim_index: *sim_index,
                    proxy_score: None,
                });
                slot_index += 1;
            }
        }
        Some(Self { prep_index, slots })
    }

    /// Capture every authoritative legacy side under fixed cardinality. Exact
    /// postscore top-three authentication may not evict candidates online
    /// because the historical epsilon preference is not transitive.
    fn from_all_candidate_sides<B>(
        prep_index: usize,
        sides: &[[SideSlot; 2]],
        sims_len: usize,
        mut budget_available: B,
    ) -> Option<Self>
    where
        B: FnMut() -> bool,
    {
        if !budget_available() {
            return None;
        }
        if !(ADAPTIVE_DEPTH_SHADOW_ROOTS..=ADAPTIVE_DEPTH_SHADOW_MAX_CAPTURE_CANDIDATES)
            .contains(&sides.len())
        {
            return None;
        }
        let mut slots: [Option<AdaptiveDepthShadowCaptureSlot>;
            ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS] = std::array::from_fn(|_| None);
        let mut slot_index = 0usize;
        for candidate_sides in sides {
            if !budget_available() {
                return None;
            }
            for side in candidate_sides {
                if !budget_available() {
                    return None;
                }
                let SideSlot::Sim(sim_index) = side else {
                    continue;
                };
                if *sim_index >= sims_len
                    || slots[..slot_index].iter().any(|slot| {
                        slot.as_ref()
                            .is_some_and(|slot| slot.sim_index == *sim_index)
                    })
                {
                    return None;
                }
                *slots.get_mut(slot_index)? = Some(AdaptiveDepthShadowCaptureSlot {
                    sim_index: *sim_index,
                    proxy_score: None,
                });
                slot_index += 1;
            }
        }
        budget_available().then_some(Self { prep_index, slots })
    }

    #[cfg(test)]
    pub(super) fn planned_slot_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    #[cfg(test)]
    pub(super) const fn slot_capacity() -> usize {
        ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS
    }

    pub(super) fn captured_score_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.proxy_score.is_some())
            .count()
    }

    pub(super) fn contains_sim(&self, sim_index: usize) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|slot| slot.sim_index == sim_index)
    }

    pub(super) fn insert_proxy_score(&mut self, sim_index: usize, proxy_score: f32) -> bool {
        if !proxy_score.is_finite() {
            return false;
        }
        let Some(slot) = self
            .slots
            .iter_mut()
            .flatten()
            .find(|slot| slot.sim_index == sim_index)
        else {
            return false;
        };
        if slot.proxy_score.is_some() {
            return false;
        }
        slot.proxy_score = Some(proxy_score);
        true
    }

    fn proxy_score(&self, sim_index: usize) -> Option<f32> {
        self.slots
            .iter()
            .flatten()
            .find(|slot| slot.sim_index == sim_index)
            .and_then(|slot| slot.proxy_score)
    }
}

fn adaptive_depth_captured_fixpoint_side_proxy(
    side: &SideSlot,
    capture: &AdaptiveDepthShadowCapture,
) -> Option<f32> {
    match side {
        SideSlot::Infeasible => Some(f32::INFINITY),
        SideSlot::Failed => None,
        SideSlot::Sim(sim_index) => capture.proxy_score(*sim_index),
    }
}

/// Deterministic advice-only winner over the authenticated three-root proxy.
/// Ties preserve historical rank zero; NaN fails closed.
pub(super) fn adaptive_depth_proxy_recommended_rank(
    proxy_scores: &[f32; ADAPTIVE_DEPTH_SHADOW_ROOTS],
) -> Option<usize> {
    if proxy_scores.iter().any(|score| score.is_nan()) {
        return None;
    }
    let mut recommended_rank = 0usize;
    for rank in 1..ADAPTIVE_DEPTH_SHADOW_ROOTS {
        if proxy_scores[rank] > proxy_scores[recommended_rank] {
            recommended_rank = rank;
        }
    }
    Some(recommended_rank)
}

pub(super) fn clear_shadow_cached_las(child: &mut MultiObjectiveGraphBabDomain) -> bool {
    let count = child.objective_bounds().len();
    child.set_cached_las(vec![None; count]).is_ok()
}

/// The committed selection for one wave domain: either the winner split's
/// already-built children or an adaptive-depth truth table (0..=2^d entries;
/// infeasible leaves are simply absent).
pub(in crate::beta_crown::engine::graph) type KfsbMultiChildren =
    Vec<(MultiObjectiveGraphBabDomain, bool, KfsbCertEffect)>;

/// Where a candidate side's value comes from during simulation.
pub(super) enum SideSlot {
    /// `with_constraint` proved the half-space empty — the side is resolved.
    Infeasible,
    /// Simulated child at `sims[i]`; its bound lands in `sim_values[i]`.
    Sim(usize),
    /// Child construction failed — the side counts as `-inf` at pick time, but
    /// any candidate containing it is forbidden from committing because its
    /// two half-spaces no longer form a complete cover.
    Failed,
}

/// Coverage status for the verifier-lifetime bounded attribution target.
///
/// Keep row absence separate from an incomplete node/index view: the former
/// measures the three-row root cap, while the latter measures producer graph
/// coverage. Conflating them made it impossible to tell which expansion could
/// improve engagement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttributionDiagCoverage {
    RowMissing,
    RootPriorStale,
    CandidateIncomplete,
    ResourceCapped,
    Complete,
}

/// Decision-only metadata for one attribution diagnostic union.
///
/// Candidate indices are in each portfolio's own rank/admission order, so the
/// historical picker's first-seen tie contract remains exact. Every identity
/// present in only one arm is re-simulated through the same private diagnostic
/// route; shared identities alone may reuse incumbent evidence.
pub(super) struct AttributionDiagPrep {
    coverage: AttributionDiagCoverage,
    overlay_complete: bool,
    historical_candidates: Vec<usize>,
    attribution_candidates: Vec<usize>,
    distinguishing_candidates: Vec<usize>,
}

/// Pre-construction form of [`AttributionDiagPrep`]. Child construction may
/// refuse individual identities, so rank-order identities are mapped to final
/// candidate indices only after the shared union has been built.
struct AttributionDiagOverlayPlan {
    coverage: AttributionDiagCoverage,
    historical_candidates: Vec<GraphKfsbCandidate>,
    attribution_candidates: Vec<GraphKfsbCandidate>,
}

fn attribution_diag_plan_within_caps(plan: &AttributionDiagOverlayPlan) -> bool {
    if plan.historical_candidates.len() > ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES
        || plan.attribution_candidates.len() > ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES
    {
        return false;
    }
    let historical: std::collections::HashSet<(&str, usize)> = plan
        .historical_candidates
        .iter()
        .map(|candidate| (candidate.node_name.as_str(), candidate.neuron_idx))
        .collect();
    let attribution: std::collections::HashSet<(&str, usize)> = plan
        .attribution_candidates
        .iter()
        .map(|candidate| (candidate.node_name.as_str(), candidate.neuron_idx))
        .collect();
    if historical.len() != plan.historical_candidates.len()
        || attribution.len() != plan.attribution_candidates.len()
    {
        return false;
    }
    historical.union(&attribution).count() <= ATTRIBUTION_MAX_UNION_CANDIDATES
        && (plan.coverage != AttributionDiagCoverage::Complete
            || historical.symmetric_difference(&attribution).count()
                <= ATTRIBUTION_MAX_DISTINGUISHING_CANDIDATES)
}

/// A root attribution is exact only for the unsplit root state that produced
/// it. Descendants have different bounds/alphas; treating the publication as a
/// current-domain score there is stale advice, so they retain historical kFSB.
#[inline]
fn root_attribution_prior_is_fresh(domain_depth: usize) -> bool {
    domain_depth == 0
}

/// Claim at most one eligible attribution-diagnostic wave for this verifier.
/// A wave without a viable target does not consume the lifetime attempt.
fn claim_attribution_diag_target(
    requested: bool,
    target: Option<usize>,
    fired: &std::sync::atomic::AtomicBool,
) -> Option<usize> {
    use std::sync::atomic::Ordering;

    if !requested {
        return None;
    }
    let target = target?;
    fired
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .ok()
        .map(|_| target)
}

fn attribution_diag_round_is_eligible(
    requested: bool,
    bab_round: usize,
    fired: &std::sync::atomic::AtomicBool,
) -> bool {
    requested
        && bab_round < ATTRIBUTION_DIAG_MAX_ELIGIBLE_ROUNDS
        && !fired.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stage ordering for authoritative/typed candidate simulations. Attribution
/// diagnostics use a separate private pass and value vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KfsbSimulationClass {
    Incumbent,
    DepthTwoOverlay,
}

/// Resolve typed simulation membership without weakening incumbent authority.
fn kfsb_simulation_class(
    incumbent: bool,
    depth_two_member: bool,
    depth_two_admitted: bool,
) -> Option<KfsbSimulationClass> {
    if incumbent {
        Some(KfsbSimulationClass::Incumbent)
    } else if depth_two_member && depth_two_admitted {
        Some(KfsbSimulationClass::DepthTwoOverlay)
    } else {
        None
    }
}

/// Whether a simulated candidate still covers the complete parent region.
///
/// `Infeasible` is authoritative proof that a half-space is empty and may be
/// omitted from the emitted children. `Failed` is merely missing work: even if
/// `Max` ranks the other finite side first, publishing it alone would silently
/// drop a reachable half-space.
#[inline]
fn side_pair_has_distinct_simulations(sides: &[SideSlot; 2]) -> bool {
    !matches!(
        (&sides[0], &sides[1]),
        (SideSlot::Sim(active), SideSlot::Sim(inactive)) if active == inactive
    )
}

#[inline]
fn candidate_cover_complete<T>(
    candidate: &GraphKfsbCandidate,
    sides: &[SideSlot; 2],
    simulations: &[Option<T>],
) -> bool {
    candidate.main_score.is_finite()
        && side_pair_has_distinct_simulations(sides)
        && sides.iter().all(|side| match side {
            SideSlot::Infeasible => true,
            SideSlot::Sim(index) => simulations.get(*index).is_some_and(Option::is_some),
            SideSlot::Failed => false,
        })
}

/// Atomically consume the feasible children of one complete binary cover.
///
/// The full preflight happens before the first `take`, so malformed, missing,
/// duplicated, or failed slots leave every simulation untouched. A proved
/// infeasible side contributes no child but still completes the cover.
fn take_complete_candidate_cover<T>(
    candidate: &GraphKfsbCandidate,
    sides: &[SideSlot; 2],
    simulations: &mut [Option<T>],
) -> Option<Vec<(T, bool)>> {
    if !candidate_cover_complete(candidate, sides, simulations) {
        return None;
    }
    let mut children = Vec::with_capacity(2);
    for (side, is_active) in sides.iter().zip([true, false]) {
        if let SideSlot::Sim(index) = side {
            // Safe after the complete-cover preflight above; no mutation
            // occurs between the check and this take.
            children.push((simulations[*index].take()?, is_active));
        }
    }
    Some(children)
}

/// Per-domain preparation: straggler row + filtered candidates + built children.
pub(super) struct DomainPrep {
    /// Position in `domains_with_unstable` (NOT the parent result index).
    pub(super) slot: usize,
    /// Straggler objective index (drives both the score seed and the pick row).
    pub(super) straggler: usize,
    /// Number of unstable candidates whose pre-score came from cached lA.
    pub(super) cached_score_candidates: usize,
    /// Prefix length of the historical top-k ∪ backup ∪ layer-quota
    /// portfolio. Every non-lookahead picker is restricted to this prefix.
    pub(super) legacy_candidates_len: usize,
    /// Indices into `candidates` for the complete, exact-total main-BaBSR
    /// portfolio. The order is the paper portfolio order, independent of
    /// which candidates were already present in the legacy prefix.
    pub(super) depth_two_lookahead_candidates: Option<Vec<usize>>,
    /// Present for at most one frontier-worst root domain over the verifier's
    /// lifetime and only under `NY_ATTR_BRANCH_DIAG=1`. Its appended identities
    /// are observation-only; the typed depth-two overlay has its own separately
    /// checked authority.
    pub(super) attribution_diag: Option<AttributionDiagPrep>,
    pub(super) candidates: Vec<GraphKfsbCandidate>,
    /// Per candidate: [active, inactive] side slots.
    pub(super) sides: Vec<[SideSlot; 2]>,
}

/// Exact-total identities selected during the target's shared BaBSR scoring.
/// Child construction is deliberately deferred until *after* the Rayon
/// legacy-prep barrier, so Shadow work cannot occupy a worker while another
/// domain is still preparing its historical prefix.
pub(super) struct DepthTwoLookaheadOverlayPlan {
    pub(super) selected: Vec<GraphKfsbCandidate>,
}

impl DomainPrep {
    fn legacy_prefix(&self) -> Option<(&[GraphKfsbCandidate], &[[SideSlot; 2]])> {
        if self.legacy_candidates_len == 0
            || self.candidates.len() != self.sides.len()
            || self.legacy_candidates_len > self.candidates.len()
        {
            return None;
        }
        Some((
            &self.candidates[..self.legacy_candidates_len],
            &self.sides[..self.legacy_candidates_len],
        ))
    }
}

#[derive(Clone, Copy)]
enum SimulatedSideProof {
    Empty,
    Lower(f32),
    Missing,
}

fn simulated_side_proof(side: &SideSlot, simulated_lowers: &[Option<f32>]) -> SimulatedSideProof {
    match side {
        SideSlot::Infeasible => SimulatedSideProof::Empty,
        SideSlot::Sim(index) => simulated_lowers
            .get(*index)
            .copied()
            .flatten()
            .filter(|lower| lower.is_finite())
            .map_or(SimulatedSideProof::Missing, SimulatedSideProof::Lower),
        SideSlot::Failed => SimulatedSideProof::Missing,
    }
}

/// A complete active/inactive pair partitions its parent. Consequently the
/// minimum of the two certified side lowers is a lower bound on the whole
/// parent. A construction-proved empty side contributes no points, so the
/// surviving side alone certifies the parent. Missing simulation authority
/// refuses the pair; two empty sides are represented elsewhere as domain
/// infeasibility and deliberately do not mint a floating-point certificate.
fn simulated_parent_pair_lower(
    sides: &[SideSlot; 2],
    simulated_lowers: &[Option<f32>],
) -> Option<f32> {
    // One simulation cannot stand for both complementary half-spaces. Reject
    // the whole provenance pair before either parent-wide or literal-side
    // reuse can observe it.
    if !side_pair_has_distinct_simulations(sides) {
        return None;
    }
    match (
        simulated_side_proof(&sides[0], simulated_lowers),
        simulated_side_proof(&sides[1], simulated_lowers),
    ) {
        (SimulatedSideProof::Lower(active), SimulatedSideProof::Lower(inactive)) => {
            Some(active.min(inactive))
        }
        (SimulatedSideProof::Empty, SimulatedSideProof::Lower(lower))
        | (SimulatedSideProof::Lower(lower), SimulatedSideProof::Empty) => Some(lower),
        (SimulatedSideProof::Empty, SimulatedSideProof::Empty)
        | (SimulatedSideProof::Missing, _)
        | (_, SimulatedSideProof::Missing) => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ReusableKfsbLower {
    lower: f32,
    scope: KfsbCertScope,
}

fn strongest_reusable_lower(
    best: Option<ReusableKfsbLower>,
    lower: f32,
    scope: KfsbCertScope,
) -> Option<ReusableKfsbLower> {
    if !lower.is_finite() {
        return best;
    }
    match best {
        Some(current) if current.lower >= lower => Some(current),
        _ => Some(ReusableKfsbLower { lower, scope }),
    }
}

/// Return the strongest simulated lower certificate applicable to `child`.
///
/// There are two independent sound sources:
///
/// * every complete candidate pair certifies the parent by exhaustive
///   active/inactive partition, hence certifies every child of that parent;
/// * if the committed history contains one simulated side, the child region is
///   a subset of that side, so that side's lower also certifies the child.
///
/// This first experiment is deliberately restricted to pure ReLU-at-zero
/// histories. The suffix is accepted only when the inherited prefix is exact
/// and the split-count/depth deltas contain only those appended constraints.
/// GenBaB, norm, future split kinds, and stale/mispaired children fail closed.
fn reusable_kfsb_lower_for_child(
    parent: &MultiObjectiveGraphBabDomain,
    child: &MultiObjectiveGraphBabDomain,
    prep: &DomainPrep,
    simulated_lowers: &[Option<f32>],
) -> Option<ReusableKfsbLower> {
    if child.verify_upper()
        || child.aggregation() != ObjectiveAggregation::Disjunctive
        || prep.candidates.len() != prep.sides.len()
    {
        return None;
    }

    let parent_history = parent.history();
    let child_history = child.history();
    if !parent_history.is_pure_relu_at_zero() || !child_history.is_pure_relu_at_zero() {
        return None;
    }
    let parent_len = parent_history.constraints.len();
    let child_prefix = child_history.constraints.get(..parent_len)?;
    if child_prefix != parent_history.constraints.as_slice() {
        return None;
    }
    let suffix = child_history.constraints.get(parent_len..)?;
    if child.depth().checked_sub(parent.depth()) != Some(suffix.len())
        || child_history
            .split_count
            .checked_sub(parent_history.split_count)
            != Some(suffix.len())
    {
        return None;
    }

    let mut best = prep
        .sides
        .iter()
        .filter_map(|sides| simulated_parent_pair_lower(sides, simulated_lowers))
        .fold(None, |best, lower| {
            strongest_reusable_lower(best, lower, KfsbCertScope::ParentCover)
        });

    for constraint in suffix {
        for (candidate, sides) in prep.candidates.iter().zip(&prep.sides) {
            if !side_pair_has_distinct_simulations(sides)
                || candidate.node_name != constraint.node_name
                || candidate.neuron_idx != constraint.neuron_idx
            {
                continue;
            }
            let side = if constraint.is_active {
                &sides[0]
            } else {
                &sides[1]
            };
            if let SimulatedSideProof::Lower(lower) = simulated_side_proof(side, simulated_lowers) {
                best = strongest_reusable_lower(
                    best,
                    lower,
                    KfsbCertScope::LiteralSide {
                        node_name: constraint.node_name.clone(),
                        neuron_idx: constraint.neuron_idx,
                        is_active: constraint.is_active,
                    },
                );
            }
        }
    }
    best
}

/// Intersect one reusable KFSB lower into a committed child.
///
/// Soundness invariant: if the helper above returns `c`, then every reachable
/// value of the selected objective on `child` is at least `c`. The inherited
/// lower is independently sound by child-subset containment, so `max(old,c)` is
/// sound. The upper endpoint and every non-target row remain bit-identical.
/// Ordered/finite/layout checks happen before mutation, and a crossing refuses
/// the entire proposal rather than manufacturing an interval.
#[must_use]
fn apply_kfsb_reusable_lower_certificate(
    authority: Option<KfsbCertAuthority>,
    publication_at: std::time::Instant,
    parent: &MultiObjectiveGraphBabDomain,
    child: &mut MultiObjectiveGraphBabDomain,
    prep: &DomainPrep,
    simulated_lowers: &[Option<f32>],
    thresholds: &[f32],
) -> KfsbCertEffect {
    let Some(authority) = authority.filter(|authority| publication_at < authority.deadline) else {
        return KfsbCertEffect::None;
    };
    if child.verify_upper()
        || child.objective_bounds.len() != thresholds.len()
        || child.verified().len() != thresholds.len()
        || child.cached_las().len() != thresholds.len()
        || prep.straggler >= thresholds.len()
        || thresholds.iter().any(|threshold| !threshold.is_finite())
    {
        return KfsbCertEffect::None;
    }
    let Some(certificate) = reusable_kfsb_lower_for_child(parent, child, prep, simulated_lowers)
    else {
        return KfsbCertEffect::None;
    };
    let Some(&(old_lower, old_upper)) = child.objective_bounds.get(prep.straggler) else {
        return KfsbCertEffect::None;
    };
    if !old_lower.is_finite()
        || !old_upper.is_finite()
        || old_lower > old_upper
        || !certificate.lower.is_finite()
        || child.verified()[prep.straggler]
        || crate::beta_crown::BetaCrownConfig::domain_is_verified_for_mode(
            false,
            old_lower,
            old_upper,
            thresholds[prep.straggler],
        )
    {
        return KfsbCertEffect::None;
    }
    let tightened_lower = old_lower.max(certificate.lower);
    if tightened_lower > old_upper || tightened_lower.to_bits() == old_lower.to_bits() {
        return KfsbCertEffect::None;
    }

    // Do not publish a mere improvement. The reusable lower must strictly
    // discharge the target row; other rows may remain unresolved and will be
    // sent through the ordinary pruned child evaluator.
    if !crate::beta_crown::BetaCrownConfig::domain_is_verified_for_mode(
        false,
        tightened_lower,
        old_upper,
        thresholds[prep.straggler],
    ) {
        return KfsbCertEffect::None;
    }

    // Bind the receipt to the exact semantic parent path. A capped, owned
    // identity avoids both hash collisions and unbounded receipt retention;
    // refusal leaves the ordinary child evaluator fully authoritative.
    let Some(parent_history_identity) = parent
        .history()
        .exact_provenance_identity()
        .filter(|identity| identity.len() <= KFSB_CERT_PARENT_ID_MAX_BYTES)
        .map(Arc::<[u8]>::from)
    else {
        return KfsbCertEffect::None;
    };

    let mut tightened = child.objective_bounds.clone();
    tightened[prep.straggler].0 = tightened_lower;
    // `update_bounds` computes the complete mask and priority before assigning
    // any field, so refusal is transactional without cloning deep domain state.
    if child.update_bounds(tightened, thresholds, false).is_err() {
        return KfsbCertEffect::None;
    }
    debug_assert!(child.verified()[prep.straggler]);

    // The verified row is excluded from later bound targets, but Complete-Clip
    // still averages the full spec set when this child becomes a next-wave
    // parent. The immutable Arc-backed cache therefore remains shared.
    let receipt = KfsbCertReceipt {
        row: prep.straggler,
        scope: certificate.scope,
        parent_history_identity,
        lower_bits: tightened_lower.to_bits(),
        authority_deadline: authority.deadline,
    };
    if child.all_verified() {
        KfsbCertEffect::ChildComplete(receipt)
    } else {
        KfsbCertEffect::RowVerified(receipt)
    }
}

#[cfg(test)]
mod kfsb_cert_reuse_tests {
    use ndarray::arr1;

    use super::*;
    use crate::{GraphNode, Layer, ReLULayer};

    fn fixture(
        objective_bounds: (f32, f32),
        threshold: f32,
        upper: bool,
    ) -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        fixture_rows(vec![objective_bounds], &[threshold], upper)
    }

    fn fixture_rows(
        objective_bounds: Vec<(f32, f32)>,
        thresholds: &[f32],
        upper: bool,
    ) -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32; 4]).into_dyn(),
            arr1(&[1.0_f32; 4]).into_dyn(),
        )
        .expect("valid input");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let parent = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            objective_bounds,
            &input,
            thresholds,
            upper,
        )
        .expect("root domain");
        (graph, parent)
    }

    fn candidate(neuron_idx: usize) -> GraphKfsbCandidate {
        GraphKfsbCandidate {
            node_name: "relu".to_string(),
            neuron_idx,
            main_score: 1.0,
            backup_score: 0.0,
        }
    }

    fn prep(candidates: Vec<GraphKfsbCandidate>, sides: Vec<[SideSlot; 2]>) -> DomainPrep {
        DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: candidates.len(),
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates,
            sides,
        }
    }

    fn split(
        graph: &GraphNetwork,
        parent: &MultiObjectiveGraphBabDomain,
        neuron_idx: usize,
        active: bool,
        threshold: f32,
        upper: bool,
    ) -> MultiObjectiveGraphBabDomain {
        split_rows(graph, parent, neuron_idx, active, &[threshold], upper)
    }

    fn split_rows(
        graph: &GraphNetwork,
        parent: &MultiObjectiveGraphBabDomain,
        neuron_idx: usize,
        active: bool,
        thresholds: &[f32],
        upper: bool,
    ) -> MultiObjectiveGraphBabDomain {
        parent
            .with_constraint(
                graph,
                GraphNeuronConstraint::new("relu".into(), neuron_idx, active, 1.0).unwrap(),
                upper,
                thresholds,
            )
            .unwrap()
            .unwrap()
    }

    #[test]
    fn simulation_admission_fails_closed() {
        let now = std::time::Instant::now();
        let future = now + std::time::Duration::from_secs(1);
        let authority = Some(KfsbCertAuthority { deadline: future });
        assert_eq!(
            capture_kfsb_cert_authority(true, Some(future), now),
            authority
        );
        assert_eq!(capture_kfsb_cert_authority(false, Some(future), now), None);
        assert_eq!(capture_kfsb_cert_authority(true, None, now), None);
        assert_eq!(capture_kfsb_cert_authority(true, Some(now), now), None);
        assert_eq!(
            admit_kfsb_sim_lower_certificate(authority, false, false, now, 0.25, 1.0),
            Some(0.25)
        );
        for refused in [
            admit_kfsb_sim_lower_certificate(None, false, false, now, 0.25, 1.0),
            admit_kfsb_sim_lower_certificate(authority, true, false, now, 0.25, 1.0),
            admit_kfsb_sim_lower_certificate(authority, false, true, now, 0.25, 1.0),
            admit_kfsb_sim_lower_certificate(authority, false, false, future, 0.25, 1.0),
            admit_kfsb_sim_lower_certificate(authority, false, false, now, f32::NAN, 1.0),
            admit_kfsb_sim_lower_certificate(authority, false, false, now, 0.25, f32::INFINITY),
            admit_kfsb_sim_lower_certificate(authority, false, false, now, 1.0, 0.25),
        ] {
            assert_eq!(refused, None);
        }
    }

    #[test]
    fn complete_pair_uses_min_and_empty_side_uses_survivor() {
        assert_eq!(
            simulated_parent_pair_lower(
                &[SideSlot::Sim(0), SideSlot::Sim(1)],
                &[Some(0.4), Some(0.25)]
            ),
            Some(0.25)
        );
        assert_eq!(
            simulated_parent_pair_lower(&[SideSlot::Infeasible, SideSlot::Sim(0)], &[Some(0.6)]),
            Some(0.6)
        );
        for sides in [
            [SideSlot::Failed, SideSlot::Sim(0)],
            [SideSlot::Infeasible, SideSlot::Infeasible],
            [SideSlot::Sim(0), SideSlot::Sim(0)],
        ] {
            assert_eq!(simulated_parent_pair_lower(&sides, &[Some(0.6)]), None);
        }
    }

    #[test]
    fn max_ranked_incomplete_cover_refuses_atomically_but_infeasible_is_complete() {
        let candidates = [candidate(0)];
        let (winner, _, _) = pick_kfsb_candidate(
            &candidates,
            [(0.75, f32::NEG_INFINITY)],
            crate::beta_crown::KfsbReduceOp::Max,
        )
        .expect("Max can rank the surviving finite side");
        assert_eq!(winner, 0);

        let mut simulations = vec![Some(7_u8)];
        assert!(take_complete_candidate_cover(
            &candidates[winner],
            &[SideSlot::Sim(0), SideSlot::Failed],
            &mut simulations,
        )
        .is_none());
        assert_eq!(
            simulations,
            vec![Some(7)],
            "refusal must not consume Sim(0)"
        );

        for malformed in [
            [SideSlot::Sim(1), SideSlot::Infeasible],
            [SideSlot::Sim(0), SideSlot::Sim(0)],
        ] {
            let mut simulations = vec![Some(7_u8)];
            assert!(
                take_complete_candidate_cover(&candidates[0], &malformed, &mut simulations,)
                    .is_none()
            );
            assert_eq!(simulations, vec![Some(7)]);
        }
        let mut missing = vec![None::<u8>];
        assert!(take_complete_candidate_cover(
            &candidates[0],
            &[SideSlot::Sim(0), SideSlot::Infeasible],
            &mut missing,
        )
        .is_none());
        let mut nonfinite_candidate = candidate(0);
        nonfinite_candidate.main_score = f32::NAN;
        let mut simulations = vec![Some(8_u8)];
        assert!(take_complete_candidate_cover(
            &nonfinite_candidate,
            &[SideSlot::Sim(0), SideSlot::Infeasible],
            &mut simulations,
        )
        .is_none());
        assert_eq!(simulations, vec![Some(8)]);

        let mut simulations = vec![Some(9_u8)];
        assert_eq!(
            take_complete_candidate_cover(
                &candidates[0],
                &[SideSlot::Sim(0), SideSlot::Infeasible],
                &mut simulations,
            ),
            Some(vec![(9, true)])
        );
        assert_eq!(simulations, vec![None]);
    }

    #[test]
    fn apply_refusals_are_transactional() {
        let side_prep = prep(
            vec![candidate(0)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(1)]],
        );
        let now = std::time::Instant::now();
        let deadline = now + std::time::Duration::from_secs(1);
        let authority = Some(KfsbCertAuthority { deadline });
        let check_unchanged = |parent: &MultiObjectiveGraphBabDomain,
                               mut child: MultiObjectiveGraphBabDomain,
                               authority: Option<KfsbCertAuthority>,
                               publication_at: std::time::Instant,
                               certificates: &[Option<f32>],
                               thresholds: &[f32],
                               prep: &DomainPrep| {
            let before = (
                child.objective_bounds.clone(),
                child.verified.clone(),
                child.priority().to_bits(),
                child
                    .cached_las()
                    .iter()
                    .map(Option::is_some)
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                apply_kfsb_reusable_lower_certificate(
                    authority,
                    publication_at,
                    parent,
                    &mut child,
                    prep,
                    certificates,
                    thresholds,
                ),
                KfsbCertEffect::None
            );
            assert_eq!(
                (
                    child.objective_bounds.clone(),
                    child.verified.clone(),
                    child.priority().to_bits(),
                    child
                        .cached_las()
                        .iter()
                        .map(Option::is_some)
                        .collect::<Vec<_>>(),
                ),
                before
            );
        };

        let (graph, partial_parent) = fixture((-1.0, 2.0), 0.75, false);
        check_unchanged(
            &partial_parent,
            split(&graph, &partial_parent, 0, true, 0.75, false),
            authority,
            now,
            &[Some(0.5), None],
            &[0.75],
            &side_prep,
        );
        for (authority, publication_at) in [(None, now), (authority, deadline)] {
            check_unchanged(
                &partial_parent,
                split(&graph, &partial_parent, 0, true, 0.75, false),
                authority,
                publication_at,
                &[Some(1.0), None],
                &[0.75],
                &side_prep,
            );
        }
        let aliased_prep = prep(
            vec![candidate(0)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(0)]],
        );
        check_unchanged(
            &partial_parent,
            split(&graph, &partial_parent, 0, true, 0.75, false),
            authority,
            now,
            &[Some(1.0)],
            &[0.75],
            &aliased_prep,
        );
        let (upper_graph, upper_parent) = fixture((-1.0, 2.0), 0.75, true);
        check_unchanged(
            &upper_parent,
            split(&upper_graph, &upper_parent, 0, true, 0.75, true),
            authority,
            now,
            &[Some(1.0), None],
            &[0.75],
            &side_prep,
        );
        let (graph, crossing_parent) = fixture((-1.0, 0.1), 0.0, false);
        check_unchanged(
            &crossing_parent,
            split(&graph, &crossing_parent, 0, true, 0.0, false),
            authority,
            now,
            &[Some(0.5), None],
            &[0.0],
            &side_prep,
        );
        let mut malformed = split(&graph, &crossing_parent, 0, true, 0.0, false);
        malformed.verified.push(false);
        check_unchanged(
            &crossing_parent,
            malformed,
            authority,
            now,
            &[Some(0.05), None],
            &[0.0],
            &side_prep,
        );
        let constrained_parent = split(&graph, &crossing_parent, 0, true, 0.0, false);
        let wrong_prefix = split(&graph, &crossing_parent, 1, true, 0.0, false);
        let neuron_one_prep = prep(
            vec![candidate(1)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(1)]],
        );
        check_unchanged(
            &constrained_parent,
            wrong_prefix,
            authority,
            now,
            &[Some(0.05), None],
            &[0.0],
            &neuron_one_prep,
        );
    }

    #[test]
    fn nonzero_straggler_certificate_updates_only_target_row() {
        let thresholds = [0.0, 0.5];
        let (graph, parent) = fixture_rows(vec![(0.1, 2.0), (-1.0, 2.0)], &thresholds, false);
        assert_eq!(parent.verified(), &[true, false]);
        let mut child = split_rows(&graph, &parent, 0, true, &thresholds, false);
        let row_zero_before = child.objective_bounds[0];
        let mut row_one_prep = prep(
            vec![candidate(0)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(1)]],
        );
        row_one_prep.straggler = 1;

        let now = std::time::Instant::now();
        let deadline = now + std::time::Duration::from_secs(1);
        let effect = apply_kfsb_reusable_lower_certificate(
            Some(KfsbCertAuthority { deadline }),
            now,
            &parent,
            &mut child,
            &row_one_prep,
            &[Some(0.8), None],
            &thresholds,
        );
        assert!(matches!(
            effect,
            KfsbCertEffect::ChildComplete(KfsbCertReceipt {
                row: 1,
                scope: KfsbCertScope::LiteralSide { .. },
                lower_bits,
                authority_deadline,
                ..
            }) if lower_bits == 0.8_f32.to_bits() && authority_deadline == deadline
        ));
        assert_eq!(child.objective_bounds[0], row_zero_before);
        assert_eq!(child.objective_bounds[1], (0.8, 2.0));
        assert_eq!(child.verified(), &[true, true]);
        assert!(child.all_verified());
    }

    #[test]
    fn partial_row_certificate_preserves_full_spec_cache_for_next_wave_clip() {
        let thresholds = [0.0, 0.5, 0.0];
        let (graph, parent) = fixture_rows(
            vec![(0.1, 2.0), (-1.0, 2.0), (-0.4, 2.0)],
            &thresholds,
            false,
        );
        let mut child = split_rows(&graph, &parent, 0, true, &thresholds, false);
        child
            .set_cached_las(vec![
                Some(crate::batched_domain::CachedLinearBounds::default()),
                Some(crate::batched_domain::CachedLinearBounds::default()),
                Some(crate::batched_domain::CachedLinearBounds::default()),
            ])
            .expect("cache shape");
        let mut row_one_prep = prep(
            vec![candidate(0)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(1)]],
        );
        row_one_prep.straggler = 1;
        let now = std::time::Instant::now();
        let deadline = now + std::time::Duration::from_secs(1);

        let effect = apply_kfsb_reusable_lower_certificate(
            Some(KfsbCertAuthority { deadline }),
            now,
            &parent,
            &mut child,
            &row_one_prep,
            &[Some(0.8), None],
            &thresholds,
        );
        assert!(matches!(
            effect,
            KfsbCertEffect::RowVerified(KfsbCertReceipt {
                row: 1,
                scope: KfsbCertScope::LiteralSide { .. },
                lower_bits,
                authority_deadline,
                ..
            }) if lower_bits == 0.8_f32.to_bits() && authority_deadline == deadline
        ));
        assert_eq!(child.verified(), &[true, true, false]);
        assert!(!child.all_verified());
        assert_eq!(child.objective_bounds()[1], (0.8, 2.0));
        assert!(child.cached_las()[0].is_some());
        assert!(child.cached_las()[1].is_some());
        assert!(child.cached_las()[2].is_some());
        assert!(complete_clip_has_full_spec_las(
            child.cached_las(),
            thresholds.len()
        ));
        let pruned = super::super::super::shared::prune_verified_multi_objective_targets(
            &[vec![1.0], vec![2.0], vec![3.0]],
            &thresholds,
            child.verified(),
        );
        assert_eq!(pruned.active_indices, vec![2]);
    }

    #[test]
    fn depth_four_uses_literal_phases_and_strongest_matching_side() {
        let (graph, parent) = fixture((-1.0, 2.0), 0.5, false);
        let phases = [true, false, true, false];
        let mut leaf = parent.clone();
        for (neuron, phase) in phases.into_iter().enumerate() {
            leaf = split(&graph, &leaf, neuron, phase, 0.5, false);
        }
        let depth_prep = prep(
            (0..4).map(candidate).collect(),
            (0..4)
                .map(|neuron| [SideSlot::Sim(2 * neuron), SideSlot::Sim(2 * neuron + 1)])
                .collect(),
        );
        // Every opposite side is missing, so no parent-pair proof exists. The
        // four literal history phases contribute 0.1, 0.3, 0.6 and 0.4; max
        // must select 0.6 regardless of the outer tuple's advisory first-side bool.
        let certificates = [
            Some(0.1),
            None,
            None,
            Some(0.3),
            Some(0.6),
            None,
            None,
            Some(0.4),
        ];
        let advisory_first_side = false;
        assert_ne!(advisory_first_side, phases[0]);
        let now = std::time::Instant::now();
        let deadline = now + std::time::Duration::from_secs(1);
        let effect = apply_kfsb_reusable_lower_certificate(
            Some(KfsbCertAuthority { deadline }),
            now,
            &parent,
            &mut leaf,
            &depth_prep,
            &certificates,
            &[0.5],
        );
        assert!(matches!(
            effect,
            KfsbCertEffect::ChildComplete(KfsbCertReceipt {
                scope: KfsbCertScope::LiteralSide { .. },
                lower_bits,
                ..
            }) if lower_bits == 0.6_f32.to_bits()
        ));
        assert_eq!(leaf.depth(), 4);
        assert_eq!(leaf.objective_bounds[0], (0.6, 2.0));
        assert!(leaf.all_verified());

        let mut unrelated = split(&graph, &parent, 3, true, 0.5, false);
        let pair_prep = prep(
            vec![candidate(0)],
            vec![[SideSlot::Sim(0), SideSlot::Sim(1)]],
        );
        let effect = apply_kfsb_reusable_lower_certificate(
            Some(KfsbCertAuthority { deadline }),
            now,
            &parent,
            &mut unrelated,
            &pair_prep,
            &[Some(0.8), Some(0.55)],
            &[0.5],
        );
        assert!(matches!(
            effect,
            KfsbCertEffect::ChildComplete(KfsbCertReceipt {
                scope: KfsbCertScope::ParentCover,
                lower_bits,
                ..
            }) if lower_bits == 0.55_f32.to_bits()
        ));
        assert_eq!(unrelated.objective_bounds[0].0, 0.55);
    }
}

struct DepthTwoLookaheadCaptureSlot {
    sim_index: usize,
    node_bounds: Option<AdaptiveDepthShadowNodeBounds>,
}

/// First-level constrained-forward maps retained for one typed lookahead
/// parent. The hard 2×15 cap is independent of wave/frontier size.
pub(super) struct DepthTwoLookaheadCapture {
    prep_index: usize,
    slots: Vec<DepthTwoLookaheadCaptureSlot>,
}

impl DepthTwoLookaheadCapture {
    pub(super) fn new(
        prep_index: usize,
        prep: &DomainPrep,
        sims_len: usize,
        expected_candidates: usize,
    ) -> Option<Self> {
        if expected_candidates == 0
            || expected_candidates > DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES
            || prep.candidates.len() != prep.sides.len()
        {
            return None;
        }
        let candidate_indices = prep.depth_two_lookahead_candidates.as_ref()?;
        if candidate_indices.len() != expected_candidates {
            return None;
        }
        let mut seen_candidates = std::collections::HashSet::new();
        let mut seen = std::collections::HashSet::new();
        let mut slots = Vec::with_capacity(2 * expected_candidates);
        for &candidate_index in candidate_indices {
            if !seen_candidates.insert(candidate_index) {
                return None;
            }
            let sides = prep.sides.get(candidate_index)?;
            for side in sides {
                match side {
                    SideSlot::Infeasible => {}
                    SideSlot::Failed => return None,
                    SideSlot::Sim(sim_index) => {
                        if *sim_index >= sims_len || !seen.insert(*sim_index) {
                            return None;
                        }
                        slots.push(DepthTwoLookaheadCaptureSlot {
                            sim_index: *sim_index,
                            node_bounds: None,
                        });
                    }
                }
            }
        }
        (slots.len() <= 2 * DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES)
            .then_some(Self { prep_index, slots })
    }

    fn contains_sim(&self, sim_index: usize) -> bool {
        self.slots.iter().any(|slot| slot.sim_index == sim_index)
    }

    fn insert_node_bounds(&mut self, sim_index: usize, node_bounds: AdaptiveDepthShadowNodeBounds) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| slot.sim_index == sim_index)
        {
            slot.node_bounds = Some(node_bounds);
        }
    }

    fn node_bounds(&self, sim_index: usize) -> Option<&AdaptiveDepthShadowNodeBounds> {
        self.slots
            .iter()
            .find(|slot| slot.sim_index == sim_index)
            .and_then(|slot| slot.node_bounds.as_ref())
    }

    #[cfg(test)]
    pub(super) fn planned_slot_count(&self) -> usize {
        self.slots.len()
    }
}

const KFSB_F64_SHADOW_TOP_K: usize = 3;
const KFSB_F64_SHADOW_MAX_CANDIDATES: usize = 64;
const KFSB_F64_SHADOW_PENDING_MAP_CAP: usize = 2 * KFSB_F64_SHADOW_TOP_K;
type KfsbF64ShadowNodeBounds = HashMap<String, Arc<BoundedTensor>>;

/// One candidate retained by the precision observer. The node maps are moved
/// out of the already-computed f32 simulation result; their tensors are
/// Arc-backed, so the observer does not deep-clone the frontier.
pub(super) struct KfsbF64ShadowCandidate {
    pub(super) candidate_index: usize,
    pub(super) f32_sides: [f32; 2],
    pub(super) f32_score: f32,
    side_node_bounds: [Option<KfsbF64ShadowNodeBounds>; 2],
}

/// Streaming post-f32 top-three capture for one deterministic wave parent.
///
/// Simulation results arrive in sim-index order today, so a candidate normally
/// leaves `pending` as soon as its second side arrives. The explicit six-map
/// cap is a fail-closed belt against a future result-order change. Together
/// with the retained top three, the observer owns at most twelve Arc-backed
/// maps. Candidate metadata is capped at 64 and never scales with the
/// wave/frontier size.
pub(super) struct KfsbF64ShadowCapture {
    pub(super) prep_index: usize,
    sim_to_candidate_side: HashMap<usize, (usize, usize)>,
    pending: HashMap<usize, KfsbF64ShadowNodeBounds>,
    completed: std::collections::HashSet<usize>,
    pub(super) top: Vec<KfsbF64ShadowCandidate>,
    declined: bool,
}

impl KfsbF64ShadowCapture {
    pub(super) fn new(prep_index: usize, prep: &DomainPrep, sims_len: usize) -> Option<Self> {
        let (legacy_candidates, legacy_sides) = prep.legacy_prefix()?;
        if legacy_candidates.len() < KFSB_F64_SHADOW_TOP_K
            || legacy_candidates.len() > KFSB_F64_SHADOW_MAX_CANDIDATES
        {
            return None;
        }
        let mut sim_to_candidate_side = HashMap::new();
        for (candidate_index, sides) in legacy_sides.iter().enumerate() {
            for (side_index, side) in sides.iter().enumerate() {
                if let SideSlot::Sim(sim_index) = side {
                    if *sim_index >= sims_len
                        || sim_to_candidate_side
                            .insert(*sim_index, (candidate_index, side_index))
                            .is_some()
                    {
                        return None;
                    }
                }
            }
        }
        Some(Self {
            prep_index,
            sim_to_candidate_side,
            pending: HashMap::new(),
            completed: std::collections::HashSet::new(),
            top: Vec::with_capacity(KFSB_F64_SHADOW_TOP_K),
            declined: false,
        })
    }

    fn contains_sim(&self, sim_index: usize) -> bool {
        !self.declined && self.sim_to_candidate_side.contains_key(&sim_index)
    }

    pub(super) fn record(
        &mut self,
        sim_index: usize,
        node_bounds: KfsbF64ShadowNodeBounds,
        prep: &DomainPrep,
        sim_values: &[Option<f32>],
        reduce_op: crate::beta_crown::KfsbReduceOp,
    ) {
        if self.declined {
            return;
        }
        let Some(&(candidate_index, _)) = self.sim_to_candidate_side.get(&sim_index) else {
            return;
        };
        if !self.pending.contains_key(&sim_index)
            && self.pending.len() >= KFSB_F64_SHADOW_PENDING_MAP_CAP
        {
            self.declined = true;
            self.pending.clear();
            self.top.clear();
            return;
        }
        self.pending.insert(sim_index, node_bounds);
        if self.completed.contains(&candidate_index) {
            return;
        }
        let Some(sides) = prep.sides.get(candidate_index) else {
            self.declined = true;
            self.pending.clear();
            self.top.clear();
            return;
        };
        let ready = sides.iter().all(|side| match side {
            SideSlot::Infeasible | SideSlot::Failed => true,
            SideSlot::Sim(index) => {
                sim_values.get(*index).copied().flatten().is_some()
                    && self.pending.contains_key(index)
            }
        });
        if !ready {
            return;
        }
        self.completed.insert(candidate_index);

        let side_value = |side: &SideSlot| -> f32 {
            match side {
                SideSlot::Infeasible => f32::INFINITY,
                SideSlot::Failed => f32::NEG_INFINITY,
                SideSlot::Sim(index) => sim_values
                    .get(*index)
                    .copied()
                    .flatten()
                    .unwrap_or(f32::NEG_INFINITY),
            }
        };
        let f32_sides = [side_value(&sides[0]), side_value(&sides[1])];
        if f32_sides == [f32::NEG_INFINITY; 2] {
            for side in sides {
                if let SideSlot::Sim(index) = side {
                    self.pending.remove(index);
                }
            }
            return;
        }
        let f32_score = kfsb_reduce(reduce_op, f32_sides[0], f32_sides[1]);
        if f32_score.is_nan() {
            for side in sides {
                if let SideSlot::Sim(index) = side {
                    self.pending.remove(index);
                }
            }
            return;
        }
        let mut side_node_bounds: [Option<KfsbF64ShadowNodeBounds>; 2] = [None, None];
        for (side_index, side) in sides.iter().enumerate() {
            if let SideSlot::Sim(index) = side {
                side_node_bounds[side_index] = self.pending.remove(index);
            }
        }
        self.top.push(KfsbF64ShadowCandidate {
            candidate_index,
            f32_sides,
            f32_score,
            side_node_bounds,
        });
        self.top.sort_by(|a, b| {
            let a_candidate = &prep.candidates[a.candidate_index];
            let b_candidate = &prep.candidates[b.candidate_index];
            let a_main = if a_candidate.main_score.is_nan() {
                f32::NEG_INFINITY
            } else {
                a_candidate.main_score
            };
            let b_main = if b_candidate.main_score.is_nan() {
                f32::NEG_INFINITY
            } else {
                b_candidate.main_score
            };
            b.f32_score
                .total_cmp(&a.f32_score)
                .then_with(|| b_main.total_cmp(&a_main))
                .then_with(|| a_candidate.node_name.cmp(&b_candidate.node_name))
                .then_with(|| a_candidate.neuron_idx.cmp(&b_candidate.neuron_idx))
                .then_with(|| a.candidate_index.cmp(&b.candidate_index))
        });
        self.top.truncate(KFSB_F64_SHADOW_TOP_K);
    }

    pub(super) fn complete(&self) -> bool {
        !self.declined && self.top.len() == KFSB_F64_SHADOW_TOP_K
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveDepthAuthoritySide {
    Infeasible,
    Sim(usize),
}

/// The only value allowed to cross from private depth-2 evaluation back to
/// authoritative branch selection. It identifies one already-built root and
/// its exact preparation; it contains no private leaf or private bound.
#[derive(Debug)]
pub(super) struct AdaptiveDepthAuthoritySelection {
    pub(super) prep_index: usize,
    pub(super) parent_index: usize,
    pub(super) slot: usize,
    pub(super) straggler: usize,
    pub(super) candidates_len: usize,
    pub(super) sides_len: usize,
    pub(super) candidate_index: usize,
    pub(super) node_name: String,
    pub(super) neuron_idx: usize,
    pub(super) main_score_bits: u32,
    pub(super) backup_score_bits: u32,
    sides: [AdaptiveDepthAuthoritySide; 2],
}

/// Final root choice after optional typed July advice.
#[derive(Clone, Copy, Debug, PartialEq)]
struct KfsbWinnerDecision {
    selected_candidate: usize,
    selected_score: f32,
}

impl KfsbWinnerDecision {
    fn selected(self) -> (usize, f32) {
        (self.selected_candidate, self.selected_score)
    }
}

fn adaptive_depth_authority_side(side: &SideSlot) -> Option<AdaptiveDepthAuthoritySide> {
    match side {
        SideSlot::Infeasible => Some(AdaptiveDepthAuthoritySide::Infeasible),
        SideSlot::Sim(index) => Some(AdaptiveDepthAuthoritySide::Sim(*index)),
        SideSlot::Failed => None,
    }
}

pub(super) fn adaptive_depth_authority_identity(
    prep_index: usize,
    parent_index: usize,
    prep: &DomainPrep,
    candidate_index: usize,
) -> Option<AdaptiveDepthAuthoritySelection> {
    let candidate = prep.candidates.get(candidate_index)?;
    let sides = prep.sides.get(candidate_index)?;
    Some(AdaptiveDepthAuthoritySelection {
        prep_index,
        parent_index,
        slot: prep.slot,
        straggler: prep.straggler,
        candidates_len: prep.candidates.len(),
        sides_len: prep.sides.len(),
        candidate_index,
        node_name: candidate.node_name.clone(),
        neuron_idx: candidate.neuron_idx,
        main_score_bits: candidate.main_score.to_bits(),
        backup_score_bits: candidate.backup_score.to_bits(),
        sides: [
            adaptive_depth_authority_side(&sides[0])?,
            adaptive_depth_authority_side(&sides[1])?,
        ],
    })
}

/// Extend an already-issued authority receipt across the one known-safe
/// attribution suffix. The selected candidate and its side identities must
/// remain byte-identical, and only equal candidate/side tail growth is
/// accepted. This preserves the resolver's exact-length mutation guard while
/// keeping observation-only appends from invalidating prior advice.
fn extend_adaptive_depth_authority_for_attribution_suffix(
    selection: &mut AdaptiveDepthAuthoritySelection,
    prep_index: usize,
    prep: &DomainPrep,
) -> bool {
    let Some(diag) = prep.attribution_diag.as_ref() else {
        return false;
    };
    if diag.coverage != AttributionDiagCoverage::Complete
        || !diag.overlay_complete
        || selection.prep_index != prep_index
        || selection.slot != prep.slot
        || selection.straggler != prep.straggler
        || selection.candidates_len != selection.sides_len
        || prep.candidates.len() != prep.sides.len()
        || prep.candidates.len() < selection.candidates_len
        || selection.candidate_index >= selection.candidates_len
    {
        return false;
    }
    let Some(candidate) = prep.candidates.get(selection.candidate_index) else {
        return false;
    };
    let Some(sides) = prep.sides.get(selection.candidate_index) else {
        return false;
    };
    let Some(active_side) = adaptive_depth_authority_side(&sides[0]) else {
        return false;
    };
    let Some(inactive_side) = adaptive_depth_authority_side(&sides[1]) else {
        return false;
    };
    if selection.node_name != candidate.node_name
        || selection.neuron_idx != candidate.neuron_idx
        || selection.main_score_bits != candidate.main_score.to_bits()
        || selection.backup_score_bits != candidate.backup_score.to_bits()
        || selection.sides[0] != active_side
        || selection.sides[1] != inactive_side
    {
        return false;
    }
    selection.candidates_len = prep.candidates.len();
    selection.sides_len = prep.sides.len();
    true
}

#[cfg(test)]
mod attribution_authority_rebind_tests {
    use super::*;

    fn candidate(name: &str, neuron_idx: usize, main_score: f32) -> GraphKfsbCandidate {
        GraphKfsbCandidate {
            node_name: name.to_string(),
            neuron_idx,
            main_score,
            backup_score: -main_score,
        }
    }

    fn authority_with_complete_suffix() -> (DomainPrep, AdaptiveDepthAuthoritySelection) {
        let mut prep = DomainPrep {
            slot: 3,
            straggler: 5,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate("selected", 7, 4.0)],
            sides: vec![[SideSlot::Infeasible, SideSlot::Infeasible]],
        };
        let selection = adaptive_depth_authority_identity(11, 13, &prep, 0)
            .expect("valid selected-root receipt");
        prep.candidates.push(candidate("diagnostic", 9, 2.0));
        prep.sides
            .push([SideSlot::Infeasible, SideSlot::Infeasible]);
        prep.attribution_diag = Some(AttributionDiagPrep {
            coverage: AttributionDiagCoverage::Complete,
            overlay_complete: true,
            historical_candidates: vec![0],
            attribution_candidates: vec![1],
            distinguishing_candidates: vec![0, 1],
        });
        (prep, selection)
    }

    fn assert_rebind_refused_without_mutation(
        prep: &DomainPrep,
        selection: &mut AdaptiveDepthAuthoritySelection,
    ) {
        let before = format!("{selection:?}");
        assert!(!extend_adaptive_depth_authority_for_attribution_suffix(
            selection, 11, prep,
        ));
        assert_eq!(format!("{selection:?}"), before);
    }

    #[test]
    fn append_only_attribution_suffix_rebinds_and_resolves_the_same_root() {
        let (prep, mut selection) = authority_with_complete_suffix();

        assert!(extend_adaptive_depth_authority_for_attribution_suffix(
            &mut selection,
            11,
            &prep,
        ));
        assert_eq!(selection.candidates_len, 2);
        assert_eq!(selection.sides_len, 2);
        assert_eq!(selection.node_name, "selected");
        assert_eq!(selection.neuron_idx, 7);
        let (candidate_index, score) = resolve_adaptive_depth_authority_candidate(
            &selection,
            11,
            13,
            &prep,
            &[],
            &[],
            crate::beta_crown::KfsbReduceOp::Min,
        )
        .expect("rebound receipt resolves");
        assert_eq!(candidate_index, 0);
        assert_eq!(score, f32::INFINITY);
    }

    #[test]
    fn selected_prefix_identity_mutation_refuses_without_mutating_receipt() {
        let (mut prep, mut selection) = authority_with_complete_suffix();
        prep.candidates[0].neuron_idx += 1;

        assert_rebind_refused_without_mutation(&prep, &mut selection);
    }

    #[test]
    fn incomplete_attribution_metadata_refuses_without_mutating_receipt() {
        let (mut prep, mut selection) = authority_with_complete_suffix();
        prep.attribution_diag
            .as_mut()
            .expect("diagnostic metadata")
            .overlay_complete = false;

        assert_rebind_refused_without_mutation(&prep, &mut selection);

        let (mut prep, mut selection) = authority_with_complete_suffix();
        prep.attribution_diag
            .as_mut()
            .expect("diagnostic metadata")
            .coverage = AttributionDiagCoverage::CandidateIncomplete;

        assert_rebind_refused_without_mutation(&prep, &mut selection);
    }

    #[test]
    fn shared_failed_attribution_suffix_does_not_revoke_selected_receipt() {
        let (mut prep, mut selection) = authority_with_complete_suffix();
        prep.sides[1][0] = SideSlot::Failed;
        let diag = prep.attribution_diag.as_mut().expect("diagnostic metadata");
        diag.historical_candidates = vec![0, 1];
        diag.attribution_candidates = vec![1];
        diag.distinguishing_candidates = vec![0];

        assert!(extend_adaptive_depth_authority_for_attribution_suffix(
            &mut selection,
            11,
            &prep,
        ));
        assert_eq!(
            resolve_adaptive_depth_authority_candidate(
                &selection,
                11,
                13,
                &prep,
                &[],
                &[],
                crate::beta_crown::KfsbReduceOp::Min,
            )
            .map(|(candidate_index, _)| candidate_index),
            Some(0),
        );
    }

    #[test]
    fn incomplete_diagnostic_without_suffix_leaves_existing_receipt_resolvable() {
        let mut prep = DomainPrep {
            slot: 3,
            straggler: 5,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate("selected", 7, 4.0)],
            sides: vec![[SideSlot::Infeasible, SideSlot::Infeasible]],
        };
        let selection = adaptive_depth_authority_identity(11, 13, &prep, 0)
            .expect("valid selected-root receipt");
        let plan = AttributionDiagOverlayPlan {
            coverage: AttributionDiagCoverage::Complete,
            historical_candidates: vec![candidate("selected", 7, 4.0)],
            attribution_candidates: vec![candidate("diagnostic", 9, 2.0)],
        };

        install_incomplete_attribution_diag(&mut prep, &plan);

        assert_eq!(prep.candidates.len(), 1);
        assert_eq!(prep.sides.len(), 1);
        assert_eq!(
            resolve_adaptive_depth_authority_candidate(
                &selection,
                11,
                13,
                &prep,
                &[],
                &[],
                crate::beta_crown::KfsbReduceOp::Min,
            )
            .map(|(candidate_index, _)| candidate_index),
            Some(0),
        );
    }
}

/// Validate the private evaluator's identity against the untouched current
/// preparation and require every authoritative first-level child to remain
/// available. This function does not take or mutate any simulation domain.
pub(super) fn resolve_adaptive_depth_authority_candidate(
    selection: &AdaptiveDepthAuthoritySelection,
    prep_index: usize,
    parent_index: usize,
    prep: &DomainPrep,
    sim_values: &[Option<f32>],
    sims: &[Option<MultiObjectiveGraphBabDomain>],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32)> {
    if selection.prep_index != prep_index
        || selection.parent_index != parent_index
        || selection.slot != prep.slot
        || selection.straggler != prep.straggler
        || selection.candidates_len != prep.candidates.len()
        || selection.sides_len != prep.sides.len()
    {
        return None;
    }
    let candidate = prep.candidates.get(selection.candidate_index)?;
    let sides = prep.sides.get(selection.candidate_index)?;
    if selection.node_name != candidate.node_name
        || selection.neuron_idx != candidate.neuron_idx
        || selection.main_score_bits != candidate.main_score.to_bits()
        || selection.backup_score_bits != candidate.backup_score.to_bits()
        || selection.sides[0] != adaptive_depth_authority_side(&sides[0])?
        || selection.sides[1] != adaptive_depth_authority_side(&sides[1])?
    {
        return None;
    }

    let side_value = |side: AdaptiveDepthAuthoritySide| -> Option<f32> {
        match side {
            AdaptiveDepthAuthoritySide::Infeasible => Some(f32::INFINITY),
            AdaptiveDepthAuthoritySide::Sim(index) => {
                sims.get(index)?.as_ref()?;
                sim_values.get(index).copied().flatten()
            }
        }
    };
    let active = side_value(selection.sides[0])?;
    let inactive = side_value(selection.sides[1])?;
    if active.is_nan()
        || inactive.is_nan()
        || (active == f32::NEG_INFINITY && inactive == f32::NEG_INFINITY)
    {
        return None;
    }
    let score = kfsb_reduce(reduce_op, active, inactive);
    (!score.is_nan()).then_some((selection.candidate_index, score))
}

/// Pick one child-evaluated kFSB candidate with the historical deterministic
/// main-score tiebreak.  Kept pure so the winner oracle can price `Min` and
/// winner-compatible `Max` on the exact same child values.
pub(super) fn pick_kfsb_candidate<I>(
    candidates: &[GraphKfsbCandidate],
    side_values: I,
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32, f32)>
where
    I: IntoIterator<Item = (f32, f32)>,
{
    let mut best: Option<(usize, f32, f32)> = None; // (candidate, score, main)
    for (ci, ((active_val, inactive_val), candidate)) in
        side_values.into_iter().zip(candidates).enumerate()
    {
        if active_val == f32::NEG_INFINITY && inactive_val == f32::NEG_INFINITY {
            continue;
        }
        let score = kfsb_reduce(reduce_op, active_val, inactive_val);
        if score.is_nan() {
            continue;
        }
        let main = candidate.main_score;
        let is_better = best
            .as_ref()
            .map(|(_, best_score, best_main)| {
                score > *best_score + 1e-6
                    || ((score - *best_score).abs() <= 1e-6
                        && !main.is_nan()
                        && (best_main.is_nan() || main > *best_main))
            })
            .unwrap_or(true);
        if is_better {
            best = Some((ci, score, main));
        }
    }
    best
}

/// Apply the exact historical child-simulation picker to one ordered portfolio
/// embedded in a larger deduplicated union. The returned index is in the union,
/// while first-seen ties follow `portfolio_indices`, not union insertion order.
fn pick_attribution_diag_portfolio(
    prep: &DomainPrep,
    diag: &AttributionDiagPrep,
    portfolio_indices: &[usize],
    incumbent_sim_values: &[Option<f32>],
    diagnostic_sim_values: &HashMap<usize, f32>,
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32)> {
    let mut candidates = Vec::with_capacity(portfolio_indices.len());
    let mut values = Vec::with_capacity(portfolio_indices.len());
    let side_value = |candidate_index: usize, side: &SideSlot| match side {
        SideSlot::Infeasible => f32::INFINITY,
        SideSlot::Sim(index) => {
            let value = if diag.distinguishing_candidates.contains(&candidate_index) {
                diagnostic_sim_values.get(index).copied()
            } else {
                incumbent_sim_values.get(*index).copied().flatten()
            };
            value.unwrap_or(f32::NEG_INFINITY)
        }
        SideSlot::Failed => f32::NEG_INFINITY,
    };
    for &candidate_index in portfolio_indices {
        let candidate = prep.candidates.get(candidate_index)?;
        let sides = prep.sides.get(candidate_index)?;
        candidates.push(candidate.clone());
        values.push((
            side_value(candidate_index, &sides[0]),
            side_value(candidate_index, &sides[1]),
        ));
    }
    let (portfolio_index, score, _) = pick_kfsb_candidate(&candidates, values, reduce_op)?;
    Some((*portfolio_indices.get(portfolio_index)?, score))
}

/// A paired comparison is published only when every constructed simulation in
/// both portfolios returned a non-NaN value. Private-budget expiry therefore
/// produces `simulation_incomplete`, never a fabricated loss for candidates
/// that simply did not run. A `Failed` side is admissible only on an identity
/// shared by both arms; on a distinguishing identity it would be asymmetric
/// construction evidence and makes the comparison incomplete.
fn attribution_diag_simulation_complete(
    prep: &DomainPrep,
    diag: &AttributionDiagPrep,
    incumbent_sim_values: &[Option<f32>],
    diagnostic_sim_values: &HashMap<usize, f32>,
) -> bool {
    if !diag.overlay_complete {
        return false;
    }
    let mut seen = std::collections::HashSet::new();
    diag.historical_candidates
        .iter()
        .chain(&diag.attribution_candidates)
        .filter(|candidate_index| seen.insert(**candidate_index))
        .all(|&candidate_index| {
            prep.sides.get(candidate_index).is_some_and(|sides| {
                sides.iter().all(|side| match side {
                    SideSlot::Infeasible => true,
                    SideSlot::Failed => !diag.distinguishing_candidates.contains(&candidate_index),
                    SideSlot::Sim(index) => {
                        let value = if diag.distinguishing_candidates.contains(&candidate_index) {
                            diagnostic_sim_values.get(index).copied()
                        } else {
                            incumbent_sim_values.get(*index).copied().flatten()
                        };
                        value.is_some_and(|value| !value.is_nan())
                    }
                })
            })
        })
}

/// Compare paired simulated outcomes with the selector's own 1e-6 score-tie
/// convention. Equal infinities are ties; NaNs are not admissible evidence.
fn attribution_diag_score_cmp(prior: f32, historical: f32) -> Option<std::cmp::Ordering> {
    if prior.is_nan() || historical.is_nan() {
        return None;
    }
    if prior == historical
        || (prior.is_finite() && historical.is_finite() && (prior - historical).abs() <= 1e-6)
    {
        return Some(std::cmp::Ordering::Equal);
    }
    Some(prior.total_cmp(&historical))
}

#[derive(Default)]
struct AttributionDiagWaveCounters {
    target_prepared: usize,
    row_hit: usize,
    row_miss: usize,
    root_prior_stale: usize,
    candidate_incomplete: usize,
    resource_capped: usize,
    simulation_incomplete: usize,
    compared: usize,
    top_change: usize,
    prior_win: usize,
    prior_tie: usize,
    prior_loss: usize,
}

fn record_incomplete_attribution_diag_coverage(
    counters: &mut AttributionDiagWaveCounters,
    coverage: AttributionDiagCoverage,
) {
    match coverage {
        AttributionDiagCoverage::RowMissing => counters.row_miss += 1,
        AttributionDiagCoverage::RootPriorStale => counters.root_prior_stale += 1,
        AttributionDiagCoverage::CandidateIncomplete => {
            counters.row_hit += 1;
            counters.candidate_incomplete += 1;
        }
        AttributionDiagCoverage::ResourceCapped => {
            counters.row_hit += 1;
            counters.resource_capped += 1;
        }
        AttributionDiagCoverage::Complete => {}
    }
}

#[cfg(test)]
mod attribution_portfolio_diag_tests {
    use super::*;

    fn candidate(name: &str, main_score: f32) -> GraphKfsbCandidate {
        GraphKfsbCandidate {
            node_name: name.to_string(),
            neuron_idx: 0,
            main_score,
            backup_score: 0.0,
        }
    }

    #[test]
    fn attribution_diagnostic_is_one_shot_and_only_eligible_targets_claim_it() {
        let fired = std::sync::atomic::AtomicBool::new(false);
        assert!(!attribution_diag_round_is_eligible(false, 0, &fired));
        assert!(attribution_diag_round_is_eligible(true, 0, &fired));
        assert!(!attribution_diag_round_is_eligible(true, 1, &fired));
        assert_eq!(claim_attribution_diag_target(false, Some(7), &fired), None);
        assert!(!fired.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(claim_attribution_diag_target(true, None, &fired), None);
        assert!(!fired.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(
            claim_attribution_diag_target(true, Some(7), &fired),
            Some(7)
        );
        assert!(fired.load(std::sync::atomic::Ordering::Relaxed));
        assert!(!attribution_diag_round_is_eligible(true, 0, &fired));
        assert_eq!(claim_attribution_diag_target(true, Some(9), &fired), None);
    }

    #[test]
    fn oversized_row_missing_details_remain_a_row_miss_not_a_hit() {
        let mut prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate("incumbent", 1.0)],
            sides: vec![[SideSlot::Infeasible, SideSlot::Infeasible]],
        };
        let plan = AttributionDiagOverlayPlan {
            coverage: AttributionDiagCoverage::RowMissing,
            historical_candidates: (0..=ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES)
                .map(|index| candidate(&format!("historical_{index}"), index as f32))
                .collect(),
            attribution_candidates: Vec::new(),
        };

        install_incomplete_attribution_diag(&mut prep, &plan);
        let diag = prep
            .attribution_diag
            .as_ref()
            .expect("incomplete telemetry");
        assert_eq!(diag.coverage, AttributionDiagCoverage::RowMissing);
        assert!(diag.historical_candidates.is_empty());
        assert!(diag.attribution_candidates.is_empty());

        let mut counters = AttributionDiagWaveCounters::default();
        record_incomplete_attribution_diag_coverage(&mut counters, diag.coverage);
        assert_eq!(counters.row_miss, 1);
        assert_eq!(counters.row_hit, 0);
        assert_eq!(counters.resource_capped, 0);
    }

    #[test]
    fn root_attribution_never_advises_descendant_domains() {
        assert!(root_attribution_prior_is_fresh(0));
        assert!(!root_attribution_prior_is_fresh(1));
        assert!(!root_attribution_prior_is_fresh(usize::MAX));
    }

    #[test]
    fn attribution_resource_envelope_is_hard_and_environment_independent() {
        assert_eq!(
            (
                ATTRIBUTION_MAX_K,
                ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES,
                ATTRIBUTION_MAX_UNION_CANDIDATES,
                ATTRIBUTION_MAX_DISTINGUISHING_CANDIDATES,
                ATTRIBUTION_MAX_APPENDED_CANDIDATES,
                ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS,
                ATTRIBUTION_MAX_PRIVATE_MEMBERS,
                ATTRIBUTION_MAX_PRIVATE_CHUNK,
                ATTRIBUTION_DIAG_MAX_ELIGIBLE_ROUNDS,
            ),
            (10, 30, 40, 20, 10, 20, 80, 16, 1),
        );
        assert_eq!(attribution_private_chunk(0), 1);
        assert_eq!(attribution_private_chunk(7), 7);
        assert_eq!(
            attribution_private_chunk(usize::MAX),
            ATTRIBUTION_MAX_PRIVATE_CHUNK
        );
        assert_eq!(
            ATTRIBUTION_MAX_PRIVATE_MEMBERS,
            2 * ATTRIBUTION_MAX_UNION_CANDIDATES
        );
        assert_eq!(
            ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS,
            2 * ATTRIBUTION_MAX_APPENDED_CANDIDATES
        );

        let scored = (0..100)
            .map(|index| candidate(&format!("relu_{index}"), 100.0 - index as f32))
            .collect::<Vec<_>>();
        let order = (0..scored.len()).collect::<Vec<_>>();
        let admitted =
            select_bounded_attribution_kfsb_candidates(&scored, &order, ATTRIBUTION_MAX_K, false)
                .expect("largest admitted primary+backup request");
        assert!(admitted.len() <= ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES);
        assert!(select_bounded_attribution_kfsb_candidates(
            &scored,
            &order,
            ATTRIBUTION_MAX_K + 1,
            false,
        )
        .is_none());
        assert!(select_bounded_attribution_kfsb_candidates(&scored, &order, 1, true).is_none());

        let oversized_plan = AttributionDiagOverlayPlan {
            coverage: AttributionDiagCoverage::Complete,
            historical_candidates: scored[..=ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES].to_vec(),
            attribution_candidates: Vec::new(),
        };
        assert!(!attribution_diag_plan_within_caps(&oversized_plan));

        let disjoint_plan = AttributionDiagOverlayPlan {
            coverage: AttributionDiagCoverage::Complete,
            historical_candidates: (0..20)
                .map(|index| candidate(&format!("historical_{index}"), index as f32))
                .collect(),
            attribution_candidates: (0..20)
                .map(|index| candidate(&format!("attribution_{index}"), index as f32))
                .collect(),
        };
        assert_eq!(
            disjoint_plan.historical_candidates.len() + disjoint_plan.attribution_candidates.len(),
            ATTRIBUTION_MAX_UNION_CANDIDATES
        );
        assert!(
            !attribution_diag_plan_within_caps(&disjoint_plan),
            "a within-union plan still refuses an oversized symmetric difference"
        );

        let exact_plan = AttributionDiagOverlayPlan {
            coverage: AttributionDiagCoverage::Complete,
            historical_candidates: (0..30)
                .map(|index| candidate(&format!("shared_{index}"), index as f32))
                .collect(),
            attribution_candidates: (10..40)
                .map(|index| candidate(&format!("shared_{index}"), index as f32))
                .collect(),
        };
        assert!(attribution_diag_plan_within_caps(&exact_plan));
    }

    #[test]
    fn incumbent_precedes_typed_overlay_membership() {
        assert_eq!(
            kfsb_simulation_class(false, true, true),
            Some(KfsbSimulationClass::DepthTwoOverlay)
        );
        assert_eq!(kfsb_simulation_class(false, true, false), None);
        assert_eq!(
            kfsb_simulation_class(true, true, true),
            Some(KfsbSimulationClass::Incumbent)
        );
    }

    #[test]
    fn distinguishing_arms_use_private_evidence_without_replacing_incumbent() {
        let mut prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate("historical", 1.0), candidate("attribution", 2.0)],
            sides: vec![
                [SideSlot::Sim(0), SideSlot::Sim(1)],
                [SideSlot::Sim(2), SideSlot::Sim(3)],
            ],
        };
        let incumbent_sim_values = vec![Some(100.0), Some(100.0), None, None];
        let diagnostic_sim_values = HashMap::from([(0, 1.0), (1, 1.0), (2, 10.0), (3, 10.0)]);
        let (incumbent_candidates, incumbent_sides) =
            prep.legacy_prefix().expect("valid incumbent prefix");
        let incumbent = pick_kfsb_candidate(
            incumbent_candidates,
            incumbent_sides.iter().map(|sides| {
                let value = |side: &SideSlot| match side {
                    SideSlot::Sim(index) => incumbent_sim_values[*index].unwrap(),
                    SideSlot::Infeasible => f32::INFINITY,
                    SideSlot::Failed => f32::NEG_INFINITY,
                };
                (value(&sides[0]), value(&sides[1]))
            }),
            crate::beta_crown::KfsbReduceOp::Min,
        )
        .expect("incumbent pick");
        assert_eq!(incumbent.0, 0);
        assert_eq!(incumbent.1, 100.0);

        let diag = AttributionDiagPrep {
            coverage: AttributionDiagCoverage::Complete,
            overlay_complete: true,
            historical_candidates: vec![0],
            attribution_candidates: vec![1],
            distinguishing_candidates: vec![0, 1],
        };
        assert!(attribution_diag_simulation_complete(
            &prep,
            &diag,
            &incumbent_sim_values,
            &diagnostic_sim_values,
        ));
        let mut missing_historical_private = diagnostic_sim_values.clone();
        missing_historical_private.remove(&0);
        assert!(
            !attribution_diag_simulation_complete(
                &prep,
                &diag,
                &incumbent_sim_values,
                &missing_historical_private,
            ),
            "an incumbent historical-only identity still requires private evidence",
        );

        let historical_diagnostic = pick_attribution_diag_portfolio(
            &prep,
            &diag,
            &[0],
            &incumbent_sim_values,
            &diagnostic_sim_values,
            crate::beta_crown::KfsbReduceOp::Min,
        )
        .expect("historical private observation");
        let attribution_diagnostic = pick_attribution_diag_portfolio(
            &prep,
            &diag,
            &[1],
            &incumbent_sim_values,
            &diagnostic_sim_values,
            crate::beta_crown::KfsbReduceOp::Min,
        )
        .expect("attribution private observation");
        assert_eq!(historical_diagnostic, (0, 1.0));
        assert_eq!(attribution_diagnostic, (1, 10.0));
        prep.sides[0][0] = SideSlot::Failed;
        assert!(
            !attribution_diag_simulation_complete(
                &prep,
                &diag,
                &incumbent_sim_values,
                &diagnostic_sim_values,
            ),
            "a distinguishing-side construction failure is not fair evidence",
        );
    }

    #[test]
    fn attribution_union_deduplicates_shared_identity_and_preserves_arm_order() {
        let historical = vec![candidate("shared", 3.0), candidate("historical", 2.0)];
        let attribution = vec![candidate("attribution", 4.0), candidate("shared", 3.0)];
        let mut union = historical.clone();
        append_unique_kfsb_candidates(&mut union, &attribution);

        assert_eq!(
            union
                .iter()
                .map(|candidate| candidate.node_name.as_str())
                .collect::<Vec<_>>(),
            vec!["shared", "historical", "attribution"],
        );
        assert_eq!(kfsb_portfolio_indices(&union, &historical), vec![0, 1]);
        assert_eq!(kfsb_portfolio_indices(&union, &attribution), vec![2, 0]);
    }

    #[test]
    fn shared_identity_reuses_incumbent_evidence_for_both_arms() {
        let prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 2,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![
                candidate("shared", 3.0),
                candidate("historical", 2.0),
                candidate("attribution", 4.0),
            ],
            sides: vec![
                [SideSlot::Sim(0), SideSlot::Sim(1)],
                [SideSlot::Sim(2), SideSlot::Sim(3)],
                [SideSlot::Sim(4), SideSlot::Sim(5)],
            ],
        };
        let diag = AttributionDiagPrep {
            coverage: AttributionDiagCoverage::Complete,
            overlay_complete: true,
            historical_candidates: vec![0, 1],
            attribution_candidates: vec![2, 0],
            distinguishing_candidates: vec![1, 2],
        };
        let incumbent_sim_values = vec![Some(3.0), Some(3.0), Some(99.0), Some(99.0), None, None];
        let diagnostic_sim_values = HashMap::from([(2, 1.0), (3, 1.0), (4, 10.0), (5, 10.0)]);

        assert!(attribution_diag_simulation_complete(
            &prep,
            &diag,
            &incumbent_sim_values,
            &diagnostic_sim_values,
        ));
        assert_eq!(
            pick_attribution_diag_portfolio(
                &prep,
                &diag,
                &diag.historical_candidates,
                &incumbent_sim_values,
                &diagnostic_sim_values,
                crate::beta_crown::KfsbReduceOp::Min,
            ),
            Some((0, 3.0)),
        );
        assert_eq!(
            pick_attribution_diag_portfolio(
                &prep,
                &diag,
                &diag.attribution_candidates,
                &incumbent_sim_values,
                &diagnostic_sim_values,
                crate::beta_crown::KfsbReduceOp::Min,
            ),
            Some((2, 10.0)),
        );
    }
}

/// Rank a bounded portfolio by repeatedly applying the exact historical kFSB
/// picker. This preserves its 1e-6 score tie, main-score tie-break, and
/// first-seen contract instead of inventing a nearby total order.
pub(super) fn rank_kfsb_candidate_portfolio(
    candidates: &[GraphKfsbCandidate],
    side_values: &[(f32, f32)],
    reduce_op: crate::beta_crown::KfsbReduceOp,
    limit: usize,
) -> Vec<(usize, f32)> {
    rank_kfsb_candidate_portfolio_with_budget(candidates, side_values, reduce_op, limit, || true)
        .unwrap_or_default()
}

fn rank_kfsb_candidate_portfolio_with_budget<B>(
    candidates: &[GraphKfsbCandidate],
    side_values: &[(f32, f32)],
    reduce_op: crate::beta_crown::KfsbReduceOp,
    limit: usize,
    mut budget_available: B,
) -> Option<Vec<(usize, f32)>>
where
    B: FnMut() -> bool,
{
    if !budget_available() {
        return None;
    }
    if candidates.len() != side_values.len() || limit == 0 {
        return Some(Vec::new());
    }
    let mut ranked = Vec::new();
    ranked.try_reserve_exact(limit.min(candidates.len())).ok()?;
    while ranked.len() < limit.min(candidates.len()) {
        if !budget_available() {
            return None;
        }
        let mut best: Option<(usize, f32, f32)> = None;
        for candidate_index in 0..candidates.len() {
            if !budget_available() {
                return None;
            }
            if ranked
                .iter()
                .any(|(selected_index, _)| *selected_index == candidate_index)
            {
                continue;
            }
            let (active, inactive) = side_values[candidate_index];
            if active == f32::NEG_INFINITY && inactive == f32::NEG_INFINITY {
                continue;
            }
            let score = kfsb_reduce(reduce_op, active, inactive);
            if score.is_nan() {
                continue;
            }
            let main = candidates[candidate_index].main_score;
            let is_better = best
                .as_ref()
                .map(|(_, best_score, best_main)| {
                    score > *best_score + 1e-6
                        || ((score - *best_score).abs() <= 1e-6
                            && !main.is_nan()
                            && (best_main.is_nan() || main > *best_main))
                })
                .unwrap_or(true);
            if is_better {
                best = Some((candidate_index, score, main));
            }
        }
        let Some((candidate_index, score, _)) = best else {
            break;
        };
        ranked.push((candidate_index, score));
    }
    budget_available().then_some(ranked)
}

/// Apply the historical picker to a subset without changing its first-seen
/// semantics. The supplied subset may be in observation/rank order; absent
/// candidates are represented by the picker's existing both-failed sentinel
/// so iteration still follows the original `candidates` order.
pub(super) fn pick_kfsb_candidate_subset_original_order(
    candidates: &[GraphKfsbCandidate],
    indexed_side_values: &[(usize, (f32, f32))],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32, f32)> {
    let mut side_values = vec![(f32::NEG_INFINITY, f32::NEG_INFINITY); candidates.len()];
    let mut seen = vec![false; candidates.len()];
    for &(candidate_index, values) in indexed_side_values {
        if candidate_index >= candidates.len() || seen[candidate_index] {
            return None;
        }
        seen[candidate_index] = true;
        side_values[candidate_index] = values;
    }
    pick_kfsb_candidate(candidates, side_values, reduce_op)
}

impl BetaCrownVerifier {
    #[inline]
    fn depth_two_lookahead_supported_at_round(
        &self,
        bab_round: usize,
        candidate_count: usize,
    ) -> bool {
        self.config
            .depth_two_branch_lookahead
            .enabled_at_round(bab_round)
            && depth_two_lookahead_policy_supported(self.config.depth_two_branch_lookahead)
            && matches!(self.config.branching_heuristic, BranchingHeuristic::Kfsb)
            // The shared historical prep chooses its objective by raw lower
            // bound. Reinterpreting it in upper mode would violate Off/Shadow
            // identity, so phase 1 fails closed until a separate typed prep
            // exists for direction-normalized `-upper`.
            && !self.config.verify_upper_bound
            && kfsb_multi_reduce_op(self.config.kfsb_reduce_op)
                == crate::beta_crown::KfsbReduceOp::Min
            && candidate_count > 0
    }

    /// Whether the wave-batched kFSB selector should run for this config:
    /// ARMED (env `NY_MO_KFSB` override, else `config.use_kfsb_multi_branching`)
    /// AND a kFSB heuristic AND a nonzero candidate budget.
    ///
    /// Tri-state arming: `NY_MO_KFSB=1` forces on, `NY_MO_KFSB=0` forces off
    /// (kill switch), and when the env is unset the preset field
    /// `use_kfsb_multi_branching` decides (default false ⇒ byte-identical to the
    /// pre-#kfsb-multi advisory path everywhere it is not a cifar100 preset).
    #[cfg(test)]
    pub(in crate::beta_crown::engine::graph) fn kfsb_multi_wave_enabled(&self) -> bool {
        let armed = kfsb_multi_env_override().unwrap_or(self.config.use_kfsb_multi_branching);
        armed
            && matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
            )
            && self.config.fsb_candidates > 0
    }

    /// Precompute DomainClipper objectives for the final leaves that will
    /// actually be committed. αβ-CROWN repeats the just-bounded parent's lAs
    /// across prospective depth children and only clamps each leaf's bounds;
    /// it does not run a child CROWN pass here. NY mirrors that by consuming
    /// every inherited per-objective cache, scoring against history-clamped
    /// leaf bounds, and persisting only compact neuron identities.
    pub(in crate::beta_crown::engine::graph) fn precompute_complete_clip_committed_decisions(
        &self,
        graph: &GraphNetwork,
        groups: &[(
            &MultiObjectiveGraphBabDomain,
            Vec<&MultiObjectiveGraphBabDomain>,
        )],
        objectives: &[Vec<f32>],
        engine: &dyn GemmEngine,
    ) {
        let child_count = groups
            .iter()
            .map(|(_, children)| children.len())
            .sum::<usize>();
        if !self.config.enable_clip_interm_domain
            || self.config.clip_interm_topk == 0
            || groups.is_empty()
            || child_count == 0
        {
            return;
        }
        let started = std::time::Instant::now();
        let min_depth = groups
            .iter()
            .flat_map(|(_, children)| children.iter().map(|child| child.depth()))
            .min()
            .unwrap_or(0);
        let max_depth = groups
            .iter()
            .flat_map(|(_, children)| children.iter().map(|child| child.depth()))
            .max()
            .unwrap_or(0);
        let refuse = |reason, source, specs| {
            record_complete_clip_decision_capture(
                "refused",
                reason,
                source,
                child_count,
                specs,
                min_depth,
                max_depth,
                0,
                started.elapsed(),
            );
        };
        let scoring_deadline = match self.effective_graph_bab_deadline() {
            Some(authority_deadline) => {
                let Some(deadline) = complete_clip_decision_scoring_deadline(
                    std::time::Instant::now(),
                    authority_deadline,
                ) else {
                    refuse("authority_reserve", "none", objectives.len());
                    return;
                };
                Some(deadline)
            }
            None => None,
        };
        let Some(root_bounds) = self.complete_clip_root_bounds_for_decision_precompute(
            graph,
            groups[0].0.input_bounds(),
            scoring_deadline,
        ) else {
            refuse("root_bank", "none", objectives.len());
            return;
        };

        let specs = objectives.len();
        // The concrete image-DAG root path intentionally returns no host
        // coefficient cache, and later dense-spec waves likewise return
        // `None` for active rows. Reconstruct only those missing parents in
        // one wave-wide GPU gather; never run one backward per prospective
        // child.
        let missing_group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(index, (parent, _))| {
                let full = complete_clip_has_full_spec_las(parent.cached_las(), objectives.len());
                (!full).then_some(index)
            })
            .collect();
        let mut gpu_mean_las = HashMap::new();
        // Snapshots in `gpu_mean_las` may come from the host reconstruct on
        // machines with no sound GPU backend; the source label must say so.
        let mut mean_las_from_host = false;
        // WHICH step refused, not merely THAT one did. Previously every failure
        // below collapsed into one `gpu_full_spec_la_refused` bucket, which on
        // relusplitter oval21 absorbed 801 of 877 waves with no way to tell the
        // spec-matrix build from an admission predicate from the gather itself.
        let mut gpu_refusal: Option<&'static str> = None;
        if !missing_group_indices.is_empty() {
            let missing_parents: Vec<&MultiObjectiveGraphBabDomain> = missing_group_indices
                .iter()
                .map(|&index| groups[index].0)
                .collect();
            match build_spec_matrix(objectives) {
                None => gpu_refusal = Some("spec_matrix_build"),
                Some(spec_matrix) => {
                    // Admission is classified BEFORE the call so the reported reason
                    // names the exact predicate. The call re-checks it internally; a
                    // start-headroom flip between the two evaluations now reports
                    // `gather_recheck_start_headroom` rather than hiding — it can
                    // never admit a wave the guard refused.
                    if let Some(reason) = crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_gpu_mean_refusal_reason(
                        &missing_parents,
                        &spec_matrix,
                        scoring_deadline,
                    ) {
                        gpu_refusal = Some(reason);
                    } else {
                        // The probed variant reports WHICH internal exit refused,
                        // measured AT that exit, instead of the flat `gather_failed`
                        // (710 of 878 waves on relusplitter oval21, all at
                        // `wall_ms=0`). Identical control flow and identical
                        // snapshots; only the label is new.
                        match crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_mean_las_from_gpu_probed(
                            graph,
                            root_bounds.as_ref(),
                            &missing_parents,
                            &spec_matrix,
                            engine,
                            scoring_deadline,
                            // #clip-chain-gather: DARK, default OFF. It currently
                            // clears the selector-gather refusal but still stops
                            // at `no_sound_gpu_backend`, so it has no scored
                            // delivery path. OFF => the extraction's
                            // `allow_pure_chain` is exactly the historical
                            // `bab_chain_wide_enabled()`.
                            crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_chain_gather_enabled(),
                        ) {
                            Err(reason) => {
                                // A host with no admissible sound GPU CROWN backend
                                // cannot run the wide gather at all. The WGPU source
                                // gate is open, but an absent explicit request or a
                                // failed live qualification still leaves no admitted
                                // device. Fall back to the host reconstruct: one
                                // spec-seeded CPU CROWN pass per parent, selection-only
                                // output. Keyed on the standalone capability probe
                                // rather than the exit label so a renamed reason cannot
                                // silently disable the fallback.
                                if crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_sound_gpu_backend_available(
                                    engine,
                                    scoring_deadline,
                                ) {
                                    gpu_refusal = Some(reason);
                                } else {
                                    match crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_mean_las_from_host(
                                        self,
                                        graph,
                                        &missing_parents,
                                        &spec_matrix,
                                        engine,
                                        scoring_deadline,
                                    ) {
                                        Some(snapshots)
                                            if snapshots.len() == missing_group_indices.len() =>
                                        {
                                            for (group_index, snapshot) in missing_group_indices
                                                .iter()
                                                .copied()
                                                .zip(snapshots)
                                            {
                                                if let Some(snapshot) = snapshot {
                                                    gpu_mean_las.insert(group_index, snapshot);
                                                    mean_las_from_host = true;
                                                } else {
                                                    gpu_refusal = Some("host_snapshot_none");
                                                }
                                            }
                                        }
                                        Some(_) => gpu_refusal = Some("host_len_mismatch"),
                                        None => gpu_refusal = Some("host_gather_failed"),
                                    }
                                }
                            }
                            Ok(snapshots)
                                if snapshots.len() != missing_group_indices.len() =>
                            {
                                gpu_refusal = Some("gather_len_mismatch");
                            }
                            Ok(snapshots) => {
                                for (group_index, snapshot) in
                                    missing_group_indices.iter().copied().zip(snapshots)
                                {
                                    if let Some(snapshot) = snapshot {
                                        gpu_mean_las.insert(group_index, snapshot);
                                    } else {
                                        gpu_refusal = Some("gather_snapshot_none");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut published = 0usize;
        let mut used_cached = false;
        let mut used_gpu_mean = false;
        for (group_index, (parent, children)) in groups.iter().enumerate() {
            // DomainClipScorer averages the full spec dimension. Individual
            // NY verified flags do not prune this list; every objective cache
            // stays present either in the host cache or the compact GPU mean.
            let all_cached_las: Option<Vec<&crate::batched_domain::CachedLinearBounds>> =
                parent.cached_las().iter().map(Option::as_deref).collect();
            let all_cached_las =
                all_cached_las.filter(|items| !items.is_empty() && items.len() == objectives.len());
            let mean_las = gpu_mean_las.get(&group_index);
            for child in children {
                if scoring_deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                    break;
                }
                let decisions = if let Some(all_cached_las) = all_cached_las.as_ref() {
                    used_cached = true;
                    crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_decisions_from_cached_las(
                        graph,
                        root_bounds.as_ref(),
                        child.node_bounds(),
                        child.history(),
                        all_cached_las,
                        self.config.clip_interm_topk,
                        scoring_deadline,
                    )
                } else if let Some(mean_las) = mean_las {
                    used_gpu_mean = true;
                    crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_decisions_from_mean_las(
                        graph,
                        root_bounds.as_ref(),
                        child.node_bounds(),
                        child.history(),
                        mean_las,
                        self.config.clip_interm_topk,
                        scoring_deadline,
                    )
                } else {
                    None
                };
                let Some(decisions) = decisions else {
                    continue;
                };
                if self.publish_complete_clip_decisions(
                    graph,
                    child.input_bounds(),
                    child.history(),
                    decisions,
                ) {
                    published = published.saturating_add(1);
                }
            }
        }
        let status = if published == child_count {
            "completed"
        } else if published > 0 {
            "partial"
        } else {
            "refused"
        };
        let reason = if published == child_count {
            "ok"
        } else if scoring_deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            "scoring_deadline"
        } else {
            // Was one flat `gpu_full_spec_la_refused`; now names the predicate.
            gpu_refusal.unwrap_or("scoring_or_publish")
        };
        // Host and GPU cannot both fill the map in one wave: the host lane only
        // runs when no sound GPU backend is admissible.
        let mean_label = if mean_las_from_host {
            "host_mean_la"
        } else {
            "gpu_mean_la"
        };
        let source = match (used_cached, used_gpu_mean) {
            (true, true) if mean_las_from_host => "cached+host_mean_la",
            (true, true) => "cached+gpu_mean_la",
            (true, false) => "cached_full_spec_la",
            (false, true) => mean_label,
            (false, false) => "none",
        };
        record_complete_clip_decision_capture(
            status,
            reason,
            source,
            child_count,
            specs,
            min_depth,
            max_depth,
            published,
            started.elapsed(),
        );
    }

    /// Round-aware arming for the typed experiment. The legacy gate above is
    /// unchanged; an explicit `NY_MO_KFSB=0` remains a kill switch for both.
    pub(in crate::beta_crown::engine::graph) fn kfsb_multi_wave_enabled_at_round(
        &self,
        bab_round: usize,
    ) -> bool {
        let typed_policy = self
            .config
            .depth_two_branch_lookahead
            .enabled_at_round(bab_round)
            && self.depth_two_lookahead_supported_at_round(
                bab_round,
                kfsb_multi_candidate_count(self.config.fsb_candidates),
            );
        // Shadow is strictly piggyback-only. Letting an observation-only mode
        // OR-arm this selector would replace the historical advisory brancher
        // with one-step kFSB even though typed advice has no authority.
        let typed_select = typed_policy
            && self.config.depth_two_branch_lookahead.mode == DepthTwoBranchLookaheadMode::Select;
        let armed = kfsb_multi_env_override()
            .unwrap_or(self.config.use_kfsb_multi_branching || typed_select);
        armed
            && matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
            )
            && self.config.fsb_candidates > 0
    }

    /// Wave-batched kFSB branch selection + child commit (#kfsb-multi).
    ///
    /// Returns `parent_idx → committed children` for every wave domain whose
    /// selection completed; misses fall back to the advisory path in the
    /// caller. INFALLIBLE by design (never fails the run): every internal
    /// error just drops the affected domain from the map.
    // Justification: the selector threads the same verification context as the
    // caller (graph, wave, relu nodes, objectives, thresholds, engine).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn select_graph_branch_kfsb_multi_batched(
        &self,
        bab_round: usize,
        graph: &GraphNetwork,
        domains_with_unstable: &[MultiObjDomainWithUnstable<'_>],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
    ) -> HashMap<usize, KfsbMultiChildren> {
        // A/B isolation overrides (measurement-only): NY_MO_KFSB_K trims the
        // candidate count (simulation cost scales with it) and NY_MO_KFSB_REDUCE
        // (`min`|`max`) overrides the configured reduce op (see
        // `resolve_kfsb_multi_reduce_op`) to isolate selection quality from
        // simulation cost without a preset edit.
        // Candidate budget: explicit `NY_MO_KFSB_K` always wins. Otherwise the
        // self-contained `NY_BRANCH_KFSB_CHILDSIM` switch pins the measured
        // throughput/quality sweet spot (k=2 — 4× the domains of k=7 at an
        // identical verified count on cifar100), while the legacy `NY_MO_KFSB`
        // path keeps `fsb_candidates` (byte-identical to before this gate).
        let k = kfsb_multi_candidate_count(self.config.fsb_candidates);
        let depth_two_policy = self.config.depth_two_branch_lookahead;
        let depth_two_requested = self.depth_two_lookahead_supported_at_round(bab_round, k);
        // Typed support itself requires k>0, so the historical zero-budget
        // early return remains exact and cannot suppress viable typed work.
        if k == 0 || domains_with_unstable.is_empty() {
            return HashMap::new();
        }
        let layer_quota = kfsb_layer_quota_enabled();
        // Read the dark gate once per wave.  In particular, the disabled hot
        // path below enters the historical proxy computation directly; it
        // neither consults a domain cache nor rebuilds the proxy score map.
        let cached_la_enabled = kfsb_cached_la_enabled();
        // Snapshot the composed α/β authority deadline exactly once at the
        // historical wave-entry point. The optional budget, historical
        // preparation, and later simulation checks all reuse this authority.
        let authority_deadline = self.effective_graph_bab_deadline();
        let wave_entry_now = std::time::Instant::now();
        if authority_deadline.is_some_and(|deadline| wave_entry_now >= deadline) {
            return HashMap::new();
        }
        // Capture the composed typed/env policy once for this whole wave. No
        // consumer is allowed to re-read the environment and retroactively
        // grant or revoke authority from an already-produced receipt.
        let kfsb_cert_reuse_armed = self.config.kfsb_cert_reuse_armed();
        let kfsb_cert_authority =
            capture_kfsb_cert_authority(kfsb_cert_reuse_armed, authority_deadline, wave_entry_now);
        let kfsb_probe = kfsb_probe_enabled();
        // Admit the whole optional experiment before any candidate expansion
        // or simulation. This exact deadline is threaded through every later
        // phase; no post-simulation phase may mint a fresh one-second budget.
        let depth_two_budget = depth_two_requested
            .then(|| DepthTwoLookaheadBudget::admit(std::time::Instant::now(), authority_deadline))
            .flatten();
        let depth_two_target_slot = depth_two_budget.and_then(|_| {
            select_depth_two_frontier_worst_slot(
                domains_with_unstable
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, _, unstable))| !unstable.is_empty())
                    .filter_map(|(slot, (_, domain, _))| {
                        domain
                            .critical_objective_index(thresholds)
                            .ok()
                            .flatten()
                            .map(|_| (slot, -domain.priority()))
                    }),
            )
        });
        // #attr-branch-diag: price one deterministic frontier-worst domain in
        // at most ONE eligible wave over this verifier's lifetime. Only
        // identities missing from the other selected portfolio are appended
        // (two children each), rather than widening every domain in a large
        // frontier and risking an OOM. A wave with no eligible target does not
        // consume the one-shot.
        let attr_branch_diag_requested =
            crate::network::gap_attribution::attribution_branch_diag_enabled();
        let attr_diag_round_eligible = attribution_diag_round_is_eligible(
            attr_branch_diag_requested,
            bab_round,
            &self.attribution_diag_fired,
        );
        let attr_diag_candidate_slot = attr_diag_round_eligible
            .then(|| {
                select_depth_two_frontier_worst_slot(
                    domains_with_unstable
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, _, unstable))| !unstable.is_empty())
                        .filter(|(_, (_, domain, _))| {
                            root_attribution_prior_is_fresh(domain.depth())
                        })
                        .filter_map(|(slot, (_, domain, _))| {
                            domain
                                .critical_objective_index(thresholds)
                                .ok()
                                .flatten()
                                .map(|_| (slot, -domain.priority()))
                        }),
                )
            })
            .flatten();
        let attr_diag_target_slot = claim_attribution_diag_target(
            attr_branch_diag_requested,
            attr_diag_candidate_slot,
            &self.attribution_diag_fired,
        );

        // ── 1+2+3a: per-domain pre-score, filter, and child construction ──
        // (parallel; the score backward dominates this stage's cost).
        // #bab-throughput: this stage is unmeasured by `mo-wave-stage`, which
        // only sees the main child pass. On cifar100 two waves account for
        // ~11 s of a ~42 s BaB window; the rest is here and in stages 3b/5.
        let __prepare_t =
            crate::phase_telemetry::phase_telemetry_enabled().then(std::time::Instant::now);
        let mut sims: Vec<Option<MultiObjectiveGraphBabDomain>> = Vec::new();
        let mut sim_owner: Vec<(usize, usize)> = Vec::new(); // (prep index, straggler row)
        let preps_raw: Vec<
            Option<(
                DomainPrep,
                Vec<(usize, MultiObjectiveGraphBabDomain)>,
                Option<DepthTwoLookaheadOverlayPlan>,
                Option<AttributionDiagOverlayPlan>,
            )>,
        > = domains_with_unstable
            .par_iter()
            .enumerate()
            .map(|(slot, (_idx, domain, unstable))| {
                self.kfsb_multi_prepare_domain(
                    graph,
                    slot,
                    domain,
                    unstable,
                    objectives,
                    thresholds,
                    k,
                    layer_quota,
                    cached_la_enabled,
                    authority_deadline,
                    attr_diag_target_slot == Some(slot),
                    if depth_two_target_slot == Some(slot) {
                        depth_two_budget.map(|budget| (depth_two_policy.candidates, budget))
                    } else {
                        None
                    },
                )
            })
            .collect();
        let mut preps: Vec<DomainPrep> = Vec::new();
        let mut depth_two_overlay_plan = None;
        let mut attribution_diag_overlay_plan = None;
        for prep in preps_raw.into_iter().flatten() {
            let (mut prep, children, overlay_plan, attr_plan) = prep;
            // Renumber this prep's local sim indices into the wave-global list.
            let base = sims.len();
            for sides in &mut prep.sides {
                for side in sides.iter_mut() {
                    if let SideSlot::Sim(local) = side {
                        *local += base;
                    }
                }
            }
            let prep_index = preps.len();
            for (_local, child) in children {
                sim_owner.push((prep_index, prep.straggler));
                sims.push(Some(child));
            }
            if let Some(plan) = overlay_plan {
                depth_two_overlay_plan = Some((prep_index, plan));
            }
            if let Some(plan) = attr_plan {
                attribution_diag_overlay_plan = Some((prep_index, plan));
            }
            preps.push(prep);
        }
        // `k` is the load-bearing number: candidates (and therefore both the
        // score sort and the `with_constraint` child construction) scale with
        // it, and `sims` is exactly the 2·|candidates| the next stage bounds.
        if let Some(t) = __prepare_t {
            eprintln!(
                "[phase] mo-kfsb-prepare domains={} k={} preps={} sims={} secs={:.2}",
                domains_with_unstable.len(),
                k,
                preps.len(),
                sims.len(),
                t.elapsed().as_secs_f64(),
            );
        }
        if preps.is_empty() {
            if attr_diag_round_eligible {
                let target_slot = attr_diag_target_slot
                    .map(|slot| slot.to_string())
                    .unwrap_or_else(|| "none".to_string());
                eprintln!(
                    "[attr-diag] wave selected={} target_slot={} target_prepared=0 row_hit=0 row_miss=0 \
                     root_prior_stale=0 candidate_incomplete=0 resource_capped=0 \
                     simulation_incomplete=0 compared=0 top_change=0 prior_win=0 \
                     prior_tie=0 prior_loss=0 prior_published={}",
                    usize::from(attr_diag_target_slot.is_some()),
                    target_slot,
                    usize::from(crate::network::gap_attribution::attribution_prior_published()),
                );
            }
            probe_kfsb_cert_authority(
                kfsb_probe,
                kfsb_cert_reuse_armed,
                kfsb_cert_authority,
                domains_with_unstable.len(),
                0,
                0,
            );
            return HashMap::new();
        }
        // Phase 2: every historical prefix is now fully prepared and globally
        // indexed. Append the one target's missing paper roots transactionally
        // under the entry-created budget. An expiry/fault leaves all legacy
        // preps, children, and indices exactly as they were at the barrier.
        if let (Some(budget), Some((prep_index, plan))) = (depth_two_budget, depth_two_overlay_plan)
        {
            let target_slot = preps.get(prep_index).map(|prep| prep.slot);
            if let (Some(prep), Some((_, domain, _))) = (
                preps.get_mut(prep_index),
                target_slot.and_then(|slot| domains_with_unstable.get(slot)),
            ) {
                self.append_depth_two_lookahead_overlay(
                    graph,
                    domain,
                    thresholds,
                    prep_index,
                    prep,
                    plan,
                    budget,
                    &mut sims,
                    &mut sim_owner,
                );
            }
        }

        // The precision observer captures one deterministic WORST parent only.
        // It is deliberately selected before simulation from immutable parent
        // bounds, but its candidate portfolio is ranked only after the f32
        // child simulation values arrive. Gate-off performs no capture, clone,
        // atomic write, or precision work.
        let mut kfsb_f64_shadow_capture = None;
        // Typed July work has priority over every legacy observer. A legacy
        // flag may neither disable the authority-capable typed lane nor
        // contend for its capture/deadline resources; its one-shot remains
        // available for a later wave where no typed budget was admitted.
        if depth_two_budget.is_none()
            && kfsb_f64_shadow_enabled()
            && !self
                .kfsb_f64_shadow_fired
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            let target = preps
                .iter()
                .enumerate()
                .filter(|(_, prep)| {
                    (KFSB_F64_SHADOW_TOP_K..=KFSB_F64_SHADOW_MAX_CANDIDATES)
                        .contains(&prep.legacy_candidates_len)
                        && prep.legacy_prefix().is_some()
                })
                .filter_map(|(prep_index, prep)| {
                    let (_, domain, _) = domains_with_unstable.get(prep.slot)?;
                    let bound = domain
                        .objective_bounds()
                        .get(prep.straggler)
                        .copied()
                        .map(|bounds| self.config.child_bound_value(Some(bounds)))?;
                    (!bound.is_nan()).then_some((prep_index, bound))
                })
                .min_by(|(a_index, a_bound), (b_index, b_bound)| {
                    a_bound
                        .total_cmp(b_bound)
                        .then_with(|| a_index.cmp(b_index))
                })
                .map(|(prep_index, _)| prep_index);
            if let Some(prep_index) = target {
                // Claim the verifier-lifetime attempt BEFORE retaining any
                // result maps. Even an incomplete/expired observation therefore
                // cannot turn into a per-wave capture tax.
                if self
                    .kfsb_f64_shadow_fired
                    .compare_exchange(
                        false,
                        true,
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    let now = std::time::Instant::now();
                    if kfsb_f64_shadow_deadline(now, authority_deadline).is_none() {
                        eprintln!(
                            "[kfsb-f64-shadow] authority=false one_shot=true \
                             skipped=authority-reserve slot={} straggler={} top_k=0 folds=0",
                            preps[prep_index].slot, preps[prep_index].straggler,
                        );
                    } else {
                        kfsb_f64_shadow_capture =
                            KfsbF64ShadowCapture::new(prep_index, &preps[prep_index], sims.len());
                        if kfsb_f64_shadow_capture.is_none() {
                            eprintln!(
                                "[kfsb-f64-shadow] authority=false one_shot=true \
                                 skipped=capture-preflight slot={} straggler={} top_k=0 folds=0",
                                preps[prep_index].slot, preps[prep_index].straggler,
                            );
                        }
                    }
                }
            }
        }

        // Compute the historical depth policy before optional captures.
        // COMMIT-only legacy observation preserves its old depth-two
        // scheduling and leaves the one-shot untouched at other horizons.
        let requested_split_depth = self
            .config
            .effective_relu_split_depth(domains_with_unstable.len());
        let wave_candidate_count = preps
            .iter()
            // Typed lookahead may append a private exact-total portfolio.
            // It grants authority only to one first-level root identity; the
            // historical batch-fill depth policy must not observe those extra
            // candidates (especially in Shadow mode).
            .map(|prep| prep.legacy_candidates_len)
            .max()
            .unwrap_or(0);
        let wave_split_depth = cap_multi_objective_wave_depth(
            requested_split_depth,
            domains_with_unstable.len(),
            wave_candidate_count,
            self.config.batch_size,
        );
        let parent_split_depth_for_prep = |prep: &DomainPrep| {
            domains_with_unstable.get(prep.slot).map(|(_, parent, _)| {
                cap_multi_objective_parent_depth(
                    wave_split_depth,
                    parent.depth(),
                    self.config.max_depth,
                )
            })
        };

        // ── 3b: wave-batched simulation, bucketed by straggler row so every
        // batched call carries exactly ONE spec row. ──
        let mut sim_values: Vec<Option<f32>> = vec![None; sims.len()];
        // #kfsb-cert-reuse: retain ONLY the proof-side scalar. In particular,
        // do not extend the lifetime of `DomainSpecCrownResult.node_bounds` for
        // the frontier: k=7 and a 256-domain wave can have thousands of
        // simulations, so retaining their maps is an avoidable OOM hazard.
        // `None` keeps the disabled path allocation-free and exact.
        let mut simulated_lower_certificates = kfsb_cert_authority.map(|_| vec![None; sims.len()]);
        // The legacy observer reduces each targeted first-child fixpoint to one
        // scalar in a fixed inline store; no result map lifetime is extended.
        let legacy_scalar_observer_allowed = depth_two_budget.is_none();
        let adaptive_depth_shadow_requested =
            legacy_scalar_observer_allowed && adaptive_depth_shadow_enabled();
        let adaptive_depth_select_requested =
            legacy_scalar_observer_allowed && adaptive_depth_select_enabled();
        let adaptive_depth_shadow_or_select_requested =
            adaptive_depth_shadow_requested || adaptive_depth_select_requested;
        let adaptive_depth_commit_requested =
            legacy_scalar_observer_allowed && adaptive_depth_commit_enabled();
        let adaptive_depth_target_prep = (!self
            .adaptive_depth_shadow_fired
            .load(std::sync::atomic::Ordering::Relaxed)
            && (adaptive_depth_shadow_or_select_requested || adaptive_depth_commit_requested))
            .then(|| {
                preps.iter().position(|prep| {
                    prep.legacy_candidates_len >= ADAPTIVE_DEPTH_SHADOW_ROOTS
                        && prep.legacy_prefix().is_some()
                        && (adaptive_depth_shadow_or_select_requested
                            || (adaptive_depth_commit_requested
                                && parent_split_depth_for_prep(prep)
                                    == Some(ADAPTIVE_DEPTH_LEGACY_COMMIT_HORIZON)))
                })
            })
            .flatten();
        let adaptive_depth_target_production_depth = adaptive_depth_target_prep
            .and_then(|prep_index| parent_split_depth_for_prep(&preps[prep_index]));
        let adaptive_depth_commit_target_eligible =
            adaptive_depth_target_production_depth == Some(ADAPTIVE_DEPTH_LEGACY_COMMIT_HORIZON);
        let adaptive_depth_observation_requested = adaptive_depth_target_prep.is_some()
            && (adaptive_depth_shadow_or_select_requested
                || (adaptive_depth_commit_requested && adaptive_depth_commit_target_eligible));

        // Claim before any private capture/census work. The immutable budget is
        // admitted lazily when the authoritative backend first returns a map,
        // so unavoidable main-path propagation time is not charged as optional
        // work. Capture planning, every census, and final telemetry then share
        // that one receipt; no observer phase may rebase the one-second clock.
        let adaptive_depth_observation_claimed = adaptive_depth_observation_requested
            && claim_adaptive_depth_attempt(&self.adaptive_depth_shadow_fired);
        let mut adaptive_depth_budget_started = false;
        let mut adaptive_depth_budget: Option<DepthTwoLookaheadBudget> = None;
        let mut adaptive_depth_capture: Option<AdaptiveDepthShadowCapture> = None;
        let mut depth_two_lookahead_capture = depth_two_budget
            .filter(|budget| budget.available_now())
            .and(depth_two_target_slot)
            .and_then(|target_slot| preps.iter().position(|prep| prep.slot == target_slot))
            .and_then(|prep_index| {
                DepthTwoLookaheadCapture::new(
                    prep_index,
                    &preps[prep_index],
                    sims.len(),
                    depth_two_policy.candidates,
                )
            });
        // Typed optional candidates are partitioned from the incumbent batch.
        // Attribution diagnostics are appended and re-simulated only after
        // this whole authoritative/typed stage completes.
        let buckets: Vec<(usize, Vec<usize>, KfsbSimulationClass)> =
            if depth_two_target_slot.is_some() {
                let mut incumbent_sim_indices = vec![false; sims.len()];
                for sim_index in preps
                    .iter()
                    .filter_map(DomainPrep::legacy_prefix)
                    .flat_map(|(_, sides)| sides.iter().flatten())
                    .filter_map(|side| match side {
                        SideSlot::Sim(index) => Some(*index),
                        SideSlot::Infeasible | SideSlot::Failed => None,
                    })
                {
                    if let Some(is_incumbent) = incumbent_sim_indices.get_mut(sim_index) {
                        *is_incumbent = true;
                    }
                }
                let mut depth_two_sim_indices = vec![false; sims.len()];
                for sim_index in preps
                    .iter()
                    .filter_map(|prep| {
                        prep.depth_two_lookahead_candidates
                            .as_ref()
                            .map(|candidates| (prep, candidates))
                    })
                    .flat_map(|(prep, candidates)| {
                        candidates
                            .iter()
                            .filter_map(|&candidate_index| prep.sides.get(candidate_index))
                            .flatten()
                    })
                    .filter_map(|side| match side {
                        SideSlot::Sim(index) => Some(*index),
                        SideSlot::Infeasible | SideSlot::Failed => None,
                    })
                {
                    if let Some(is_depth_two) = depth_two_sim_indices.get_mut(sim_index) {
                        *is_depth_two = true;
                    }
                }
                let mut incumbent_buckets: HashMap<usize, Vec<usize>> = HashMap::new();
                let mut depth_two_overlay_buckets: HashMap<usize, Vec<usize>> = HashMap::new();
                for (i, &(_, row)) in sim_owner.iter().enumerate() {
                    match kfsb_simulation_class(
                        incumbent_sim_indices.get(i).copied().unwrap_or(false),
                        depth_two_sim_indices.get(i).copied().unwrap_or(false),
                        depth_two_lookahead_capture.is_some(),
                    ) {
                        Some(KfsbSimulationClass::Incumbent) => {
                            incumbent_buckets.entry(row).or_default().push(i);
                        }
                        Some(KfsbSimulationClass::DepthTwoOverlay) => {
                            depth_two_overlay_buckets.entry(row).or_default().push(i);
                        }
                        None => {}
                    }
                }
                incumbent_buckets
                    .into_iter()
                    .map(|(row, members)| (row, members, KfsbSimulationClass::Incumbent))
                    .chain(
                        depth_two_overlay_buckets.into_iter().map(|(row, members)| {
                            (row, members, KfsbSimulationClass::DepthTwoOverlay)
                        }),
                    )
                    .collect()
            } else {
                // Exact legacy path: construct the same single bucket map directly,
                // without even allocating the typed partition set.
                let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
                for (i, &(_, row)) in sim_owner.iter().enumerate() {
                    buckets.entry(row).or_default().push(i);
                }
                buckets
                    .into_iter()
                    .map(|(row, members)| (row, members, KfsbSimulationClass::Incumbent))
                    .collect()
            };
        let chunk_size = kfsb_sim_chunk();
        let clip = clip_interm_resnet_enabled();
        let mut depth_two_overlay_verifier: Option<BetaCrownVerifier> = None;
        // #bab-throughput: stage 3b accumulated across ALL buckets/chunks and
        // printed ONCE below — a per-chunk line would be unreadable at k=7 on a
        // wide wave. `secs` is the whole stage (shim build + clip + batched
        // build + backward + result folding); `fwd/bwd/mat` is the primitive's
        // own `DenseSpecStageTiming`, so `secs - (fwd+bwd+mat)` IS the
        // per-chunk assembly overhead this lane pays on top of the backward.
        let __sim_t =
            crate::phase_telemetry::phase_telemetry_enabled().then(std::time::Instant::now);
        let mut __sim_children = 0usize;
        let mut __sim_chunks = 0usize;
        let mut __sim_fwd = 0.0f64;
        let mut __sim_bwd = 0.0f64;
        let mut __sim_mat = 0.0f64;
        // #sim-starves-evaluation: the kFSB simulation is ADVISORY -- it ranks
        // candidate splits so the wave commits good ones. It had no budget of its
        // own: the chunk loop below broke only on
        // `past_effective_graph_bab_deadline()`, i.e. the GLOBAL BaB deadline, so
        // ranking consumed whatever time it was given, up to every constructed
        // candidate.
        //
        // MEASURED on cifar100 idx_8600 (official 100s budget): BaB round 1 spent
        // 9.35s entirely in this stage and emitted NO `mo-wave-stage` at all --
        // zero children evaluated -- while 16 children sat queued and the whole
        // run explored exactly ONE domain. Making the simulation 6.2x cheaper per
        // candidate did not help: it simply submitted 384 candidates instead of
        // 128 in the same wall clock, because the only thing stopping it was the
        // budget it was supposed to be helping spend.
        //
        // Give ranking a bounded slice of the time remaining and leave the rest
        // for evaluating children. `NY_KFSB_SIM_SHARE` overrides the fraction for
        // A/B; 0 disables the cap and restores the previous behaviour.
        //
        // SOUND BY CONSTRUCTION: stopping early means ranking over fewer
        // candidates, so the wave may commit a WORSE split. It cannot produce a
        // wrong bound -- every committed child is still evaluated by the ordinary
        // sound backward, and an unranked candidate is simply not chosen.
        let sim_deadline = {
            let share = ny_levers::read(&ny_levers::decls::wide_lane::KFSB_SIM_SHARE)
                .value
                .as_f64()
                .map_or(0.35_f32, |share| share as f32);
            (share > 0.0)
                .then(|| {
                    self.effective_graph_bab_deadline().map(|global| {
                        let now = std::time::Instant::now();
                        now + global.saturating_duration_since(now).mul_f32(share)
                    })
                })
                .flatten()
        };
        'buckets: for (row, members, simulation_class) in buckets {
            if sim_deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
                break 'buckets;
            }
            let paper_only = simulation_class == KfsbSimulationClass::DepthTwoOverlay;
            if paper_only && !depth_two_budget.is_some_and(DepthTwoLookaheadBudget::available_now) {
                depth_two_lookahead_capture = None;
                break;
            }
            let Some(spec_matrix) = build_spec_matrix(&[objectives[row].clone()]) else {
                continue;
            };
            for chunk in members.chunks(chunk_size) {
                // Deadline between chunks: unprocessed children stay failed
                // (-inf) and their domains fall back to the advisory path.
                if paper_only {
                    if !depth_two_budget.is_some_and(DepthTwoLookaheadBudget::available_now) {
                        depth_two_lookahead_capture = None;
                        break 'buckets;
                    }
                } else if self.past_effective_graph_bab_deadline()
                    || sim_deadline.is_some_and(|limit| std::time::Instant::now() >= limit)
                {
                    break 'buckets;
                }
                let __clip_t = std::time::Instant::now();
                let build_shim = |i: usize| -> GraphBabDomain {
                    let child = sims[i].as_ref().expect("sim child pending");
                    let mut shim = graph_bab_domain_shim(child);
                    if clip {
                        if let Some(clipped) = self.clip_child_node_bounds(graph, child, engine) {
                            shim.node_bounds = clipped;
                            // #cone-delta: the clip replaced the map the delta
                            // was tracked against — fail closed to full-history
                            // seeding for this shim.
                            shim.delta_pre_nodes =
                                crate::beta_crown::domain::delta_pre_nodes_unknown();
                        }
                    }
                    shim
                };
                // #clip-interm-par (M1): parallelize the per-child clip when armed.
                // `collect` preserves chunk order => the per-bucket ONE-spec-row
                // invariant (buckets above) is untouched.
                let shims: Vec<GraphBabDomain> =
                    if clip && super::batched_dense_specs::clip_interm_par_enabled() {
                        chunk
                            .par_iter()
                            .map(|&i| {
                                let _g = crate::faer_parallelism::RayonTaskGuard::new();
                                build_shim(i)
                            })
                            .collect()
                    } else {
                        chunk.iter().map(|&i| build_shim(i)).collect()
                    };
                if paper_only
                    && !depth_two_budget.is_some_and(DepthTwoLookaheadBudget::available_now)
                {
                    depth_two_lookahead_capture = None;
                    break 'buckets;
                }
                let shim_refs: Vec<&GraphBabDomain> = shims.iter().collect();
                if std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[clip-resnet] stage=kfsb n={} par={} secs={:.3}",
                        chunk.len(),
                        (clip && super::batched_dense_specs::clip_interm_par_enabled()) as u8,
                        __clip_t.elapsed().as_secs_f64()
                    );
                }
                let batched = match BatchedDomains::from_graph_domains_with_options(
                    &shim_refs,
                    relu_nodes,
                    BatchedDomainOptions {
                        enable_interm_transfer: self.config.enable_interm_transfer,
                    },
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::debug!(
                            "kfsb-multi: BatchedDomains build failed ({e}); chunk falls back"
                        );
                        continue;
                    }
                };
                // These candidate children are simulations. Their scalar
                // bounds rank the split; the result cache is at most retained
                // by the typed lookahead or precision observer, while the
                // committed child below is the original `sims[i]`. Match
                // αβ-CROWN's schedule by clipping only that child later.
                if paper_only
                    && !depth_two_budget.is_some_and(DepthTwoLookaheadBudget::available_now)
                {
                    depth_two_lookahead_capture = None;
                    break 'buckets;
                }
                let propagated = if paper_only {
                    let Some(budget) = depth_two_budget else {
                        depth_two_lookahead_capture = None;
                        break 'buckets;
                    };
                    let overlay_verifier = depth_two_overlay_verifier.get_or_insert_with(|| {
                        let mut shadow_config = self.config.clone();
                        shadow_config.timeout = ADAPTIVE_DEPTH_SHADOW_BUDGET;
                        let mut verifier = self.with_config_from(shadow_config);
                        // Pin to the entry-created deadline. Constructor
                        // latency and earlier optional work cannot extend it.
                        verifier.config.alpha_config.deadline = Some(budget.private_deadline);
                        verifier
                    });
                    if !budget.available_now() {
                        depth_two_lookahead_capture = None;
                        break 'buckets;
                    }
                    let _complete_clip_suppression = overlay_verifier
                        .complete_clip_deadline_overrides
                        .suppress_complete_clip_scoped();
                    // #layer-deadline-suppression: this is the ADVISORY kFSB
                    // simulation -- it ranks candidate splits. Its constrained
                    // forward previously handed the BaB deadline to the layer
                    // kernels, which routes every Conv2d to the certified-f64
                    // scalar loop rather than im2col + faer GEMM. Suppress only
                    // the LAYER authority here; the per-node loop still polls, so
                    // the pass still refuses cooperatively. The resulting bound is
                    // looser-or-equal, which is the safe direction for a ranking
                    // score and remains valid as a certified lower bound.
                    let _layer_deadline_suppression = overlay_verifier
                        .complete_clip_deadline_overrides
                        .suppress_layer_deadline_scoped();
                    overlay_verifier.propagate_crown_with_batched_domains_full_specs_timed(
                        graph,
                        &shim_refs,
                        &batched,
                        &spec_matrix,
                        engine,
                    )
                } else {
                    let _complete_clip_suppression = self
                        .complete_clip_deadline_overrides
                        .suppress_complete_clip_scoped();
                    // #layer-deadline-suppression: see the sibling arm above.
                    let _layer_deadline_suppression = self
                        .complete_clip_deadline_overrides
                        .suppress_layer_deadline_scoped();
                    self.propagate_crown_with_batched_domains_full_specs_timed(
                        graph,
                        &shim_refs,
                        &batched,
                        &spec_matrix,
                        engine,
                    )
                };
                // Read the primitive's own stage split (kept by the `_timed`
                // entry above instead of dropped with the rest of the result)
                // and hand the plain `Vec<DomainSpecCrownResult>` on: everything
                // below is byte-for-byte the untimed lane.
                //
                // THIS MUST STAY ABOVE `simulation_completed_at`. The untimed
                // primitive moved `.results` out and dropped the remainder of
                // `BatchedSpecBackwardResult` INSIDE the callee, i.e. before the
                // completion instant was sampled. Doing the same here keeps the
                // deallocation on the same side of that clock. Sampling the
                // instant first would make it strictly earlier, and it is the
                // sole clock feeding `admit_kfsb_sim_lower_certificate`'s
                // `completed_at < authority.deadline` test -- so a simulated
                // lower bound that previously missed the authority deadline
                // could gain certificate authority, with telemetry DISARMED.
                // That is a proof-admission change, not a measurement.
                let propagated = propagated.map(|dense_out| {
                    if __sim_t.is_some() {
                        __sim_children += chunk.len();
                        __sim_chunks += 1;
                        if let Some(t) = dense_out.stage_timing {
                            __sim_fwd += t.forward_elapsed_s;
                            __sim_bwd += t.backward_elapsed_s;
                            __sim_mat += t.materialize_elapsed_s;
                        }
                    }
                    dense_out.results
                });
                // Snapshot completion immediately after the backend returns.
                // A historical value may still rank advisory splits after a
                // late return, but it cannot gain certificate authority.
                let simulation_completed_at =
                    kfsb_cert_authority.map(|_| std::time::Instant::now());
                // A paper-only call that completes after either deadline has
                // no authority. Drop its result and the whole typed portfolio;
                // all historical simulation values are already retained.
                if paper_only
                    && !depth_two_budget.is_some_and(DepthTwoLookaheadBudget::available_now)
                {
                    depth_two_lookahead_capture = None;
                    break 'buckets;
                }
                match propagated {
                    Ok(results) if results.len() == chunk.len() => {
                        for (&i, result) in chunk.iter().zip(results) {
                            if paper_only
                                && !depth_two_budget
                                    .is_some_and(DepthTwoLookaheadBudget::available_now)
                            {
                                depth_two_lookahead_capture = None;
                                break 'buckets;
                            }
                            if adaptive_depth_observation_claimed && !adaptive_depth_budget_started
                            {
                                adaptive_depth_budget_started = true;
                                adaptive_depth_budget = DepthTwoLookaheadBudget::admit(
                                    std::time::Instant::now(),
                                    authority_deadline,
                                );
                                if let Some(budget) = adaptive_depth_budget {
                                    adaptive_depth_capture =
                                        adaptive_depth_target_prep.and_then(|prep_index| {
                                            let (_, legacy_sides) =
                                                preps[prep_index].legacy_prefix()?;
                                            AdaptiveDepthShadowCapture::from_all_candidate_sides(
                                                prep_index,
                                                legacy_sides,
                                                sims.len(),
                                                || budget.available_now(),
                                            )
                                        });
                                } else {
                                    eprintln!(
                                        "[mo-adaptive-fixpoint-proxy] outcome=declined \
                                         reason=authority-reserve authority=0 commit=0"
                                    );
                                }
                            }
                            let bounds = spec_bounds_to_vec(&result.output_bounds);
                            if let Some(&(l, u)) = bounds.first() {
                                sim_values[i] = Some(self.config.child_bound_value(Some((l, u))));
                                if bounds.len() == 1 {
                                    if let (Some(completed_at), Some(certificates)) = (
                                        simulation_completed_at,
                                        simulated_lower_certificates.as_mut(),
                                    ) {
                                        let verify_upper = sims
                                            .get(i)
                                            .and_then(Option::as_ref)
                                            .is_none_or(MultiObjectiveGraphBabDomain::verify_upper);
                                        certificates[i] = admit_kfsb_sim_lower_certificate(
                                            kfsb_cert_authority,
                                            verify_upper,
                                            paper_only,
                                            completed_at,
                                            l,
                                            u,
                                        );
                                    }
                                }
                            }
                            if adaptive_depth_capture.is_some()
                                && !adaptive_depth_budget
                                    .is_some_and(DepthTwoLookaheadBudget::available_now)
                            {
                                adaptive_depth_capture = None;
                            }
                            let should_capture = adaptive_depth_capture
                                .as_ref()
                                .is_some_and(|capture| capture.contains_sim(i));
                            let should_capture_f64 = kfsb_f64_shadow_capture
                                .as_ref()
                                .is_some_and(|capture| capture.contains_sim(i));
                            if should_capture {
                                let proxy_score = adaptive_depth_budget
                                    .filter(|budget| budget.available_now())
                                    .and_then(|budget| {
                                        sims.get(i).and_then(Option::as_ref).and_then(|domain| {
                                            adaptive_depth_fixpoint_proxy_from_bounds(
                                                graph,
                                                relu_nodes,
                                                domain,
                                                &result.node_bounds,
                                                || budget.available_now(),
                                            )
                                        })
                                    });
                                if !adaptive_depth_budget
                                    .is_some_and(DepthTwoLookaheadBudget::available_now)
                                {
                                    adaptive_depth_capture = None;
                                } else if let (Some(capture), Some(proxy_score)) =
                                    (adaptive_depth_capture.as_mut(), proxy_score)
                                {
                                    let _ = capture.insert_proxy_score(i, proxy_score);
                                }
                            }
                            if depth_two_lookahead_capture
                                .as_ref()
                                .is_some_and(|capture| capture.contains_sim(i))
                            {
                                if !depth_two_budget
                                    .is_some_and(DepthTwoLookaheadBudget::available_now)
                                {
                                    // Legacy propagation itself is
                                    // authoritative, but retaining its map for
                                    // typed scoring is optional and may not
                                    // cross the entry-created deadline.
                                    depth_two_lookahead_capture = None;
                                } else if let Some(capture) = depth_two_lookahead_capture.as_mut() {
                                    // Clone only when the typed experiment is
                                    // active. Tensors are Arc-backed; the
                                    // authoritative result remains untouched.
                                    capture.insert_node_bounds(i, result.node_bounds.clone());
                                }
                            }
                            if should_capture_f64 {
                                if let Some(capture) = kfsb_f64_shadow_capture.as_mut() {
                                    capture.record(
                                        i,
                                        result.node_bounds,
                                        &preps[capture.prep_index],
                                        &sim_values,
                                        kfsb_multi_reduce_op(self.config.kfsb_reduce_op),
                                    );
                                }
                            }
                        }
                    }
                    Ok(results) => {
                        tracing::debug!(
                            "kfsb-multi: result count {} != chunk {} — chunk dropped",
                            results.len(),
                            chunk.len()
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            "kfsb-multi: dense-spec backward failed ({e}); chunk dropped"
                        );
                    }
                }
            }
        }
        // Only report when a chunk actually ran. Four paths reach here with zero
        // chunks executed (empty buckets, every `build_spec_matrix` returning
        // None, the deadline breaking out, or the paper-only budget already
        // spent), and `sims=0 chunks=0 ... secs=0.00` is indistinguishable from
        // "the wide backward ran and was instantaneous" -- the precise confusion
        // these markers exist to remove.
        if let Some(t) = __sim_t.filter(|_| __sim_chunks > 0) {
            eprintln!(
                "[phase] mo-kfsb-sim sims={} chunks={} chunk_size={} \
                 fwd={:.2}s bwd={:.2}s mat={:.2}s secs={:.2}",
                __sim_children,
                __sim_chunks,
                chunk_size,
                __sim_fwd,
                __sim_bwd,
                __sim_mat,
                t.elapsed().as_secs_f64(),
            );
        }

        // Typed July-2026 path: price exactly the configured total root
        // portfolio from independently refreshed first-child BaBSR scores.
        // Only an identity for one original root may cross back.
        let mut depth_two_lookahead_selection = depth_two_budget.and_then(|budget| {
            self.evaluate_depth_two_lookahead_advice(
                graph,
                domains_with_unstable,
                relu_nodes,
                objectives,
                &preps,
                &sim_values,
                &sims,
                depth_two_policy,
                budget,
                depth_two_lookahead_capture.as_ref(),
            )
        });

        // Legacy M27/M28 authority is retired until a bounded implementation
        // can propagate true second children. This observer only summarizes
        // already-produced first-child fixpoints and cannot return authority.
        self.observe_adaptive_depth_captured_fixpoint_proxy(
            &preps,
            &sim_values,
            adaptive_depth_capture,
            adaptive_depth_target_prep,
            adaptive_depth_budget,
        );
        // Observation only: the captured maps and every authoritative input are
        // borrowed immutably. The helper can emit telemetry, but it cannot
        // replace `sim_values`, choose a split, raise a bound, or move a child.
        if let Some(capture) = kfsb_f64_shadow_capture {
            self.observe_kfsb_f64_shadow(
                graph,
                objectives,
                &preps,
                &sim_values,
                &sims,
                capture,
                authority_deadline,
            );
        }

        // Authority-capable typed/adaptive advice and every pre-existing
        // observer run first. The attribution diagnostic therefore cannot
        // consume their private deadline or suppress an otherwise-valid
        // receipt. Its own construction and propagation share one newly
        // admitted private budget, and all failure paths retain the incumbent
        // prefix. Values are allocated only when a complete overlay is armed.
        let mut attribution_diag_sim_values: Option<HashMap<usize, f32>> = None;
        if let Some((prep_index, plan)) = attribution_diag_overlay_plan.take() {
            let target_slot = preps.get(prep_index).map(|prep| prep.slot);
            if plan.coverage != AttributionDiagCoverage::Complete {
                if let Some(prep) = preps.get_mut(prep_index) {
                    install_incomplete_attribution_diag(prep, &plan);
                }
            } else if let (Some(budget), Some((_, domain, _))) = (
                DepthTwoLookaheadBudget::admit(std::time::Instant::now(), authority_deadline),
                target_slot.and_then(|slot| domains_with_unstable.get(slot)),
            ) {
                let appended = preps.get_mut(prep_index).is_some_and(|prep| {
                    self.append_attribution_diag_overlay(
                        graph,
                        domain,
                        thresholds,
                        prep_index,
                        prep,
                        &plan,
                        budget,
                        &mut sims,
                        &mut sim_owner,
                    )
                });
                // Optional suffix indices receive no incumbent value or
                // reusable certificate. Existing authority receipts are
                // rebound only across this validated append-only suffix.
                sim_values.resize(sims.len(), None);
                if appended {
                    let prep = &preps[prep_index];
                    if depth_two_lookahead_selection
                        .as_ref()
                        .is_some_and(|selection| selection.prep_index == prep_index)
                        && !depth_two_lookahead_selection
                            .as_mut()
                            .is_some_and(|selection| {
                                extend_adaptive_depth_authority_for_attribution_suffix(
                                    selection, prep_index, prep,
                                )
                            })
                    {
                        depth_two_lookahead_selection = None;
                    }
                    attribution_diag_sim_values = Some(self.simulate_attribution_diag_overlay(
                        graph, relu_nodes, objectives, engine, prep_index, &preps, &sims, budget,
                    ));
                }
            } else if let Some(prep) = preps.get_mut(prep_index) {
                install_incomplete_attribution_diag(prep, &plan);
            }
        }

        // ── 4: pick every wave winner without consuming its simulated
        // children. They remain available for exact depth-one fallback if an
        // adaptive truth-table expansion refuses atomically. ──
        let mut probe_lines: Vec<String> = Vec::new();
        let mut attr_diag_counters =
            attr_diag_round_eligible.then(AttributionDiagWaveCounters::default);
        let mut attr_diag_detail: Option<String> = None;
        let winner_probe = kfsb_winner_probe_enabled();
        let winner_probe_domains = winner_probe.then(kfsb_winner_probe_domains).unwrap_or(0);
        let mut winner_probe_lines: Vec<String> = Vec::new();
        let mut winners: Vec<Option<KfsbWinnerDecision>> = Vec::with_capacity(preps.len());
        let empty_attribution_diag_sim_values = HashMap::new();
        let attribution_diag_sim_values = attribution_diag_sim_values
            .as_ref()
            .unwrap_or(&empty_attribution_diag_sim_values);
        for (prep_index, prep) in preps.iter().enumerate() {
            if let (Some(diag), Some(counters)) =
                (prep.attribution_diag.as_ref(), attr_diag_counters.as_mut())
            {
                counters.target_prepared += 1;
                match diag.coverage {
                    coverage @ (AttributionDiagCoverage::RowMissing
                    | AttributionDiagCoverage::RootPriorStale
                    | AttributionDiagCoverage::CandidateIncomplete
                    | AttributionDiagCoverage::ResourceCapped) => {
                        record_incomplete_attribution_diag_coverage(counters, coverage);
                    }
                    AttributionDiagCoverage::Complete => {
                        counters.row_hit += 1;
                        if !attribution_diag_simulation_complete(
                            prep,
                            diag,
                            &sim_values,
                            attribution_diag_sim_values,
                        ) {
                            counters.simulation_incomplete += 1;
                            attr_diag_detail = Some(format!(
                                "[attr-diag] decision slot={} row={} \
                                 coverage=simulation-incomplete historical_n={} attribution_n={}",
                                prep.slot,
                                prep.straggler,
                                diag.historical_candidates.len(),
                                diag.attribution_candidates.len(),
                            ));
                            // Do not turn private-budget misses into prior
                            // losses. The incumbent winner below still uses its
                            // ordinary available simulations.
                        } else {
                            let reduce_op = kfsb_multi_reduce_op(self.config.kfsb_reduce_op);
                            let historical = pick_attribution_diag_portfolio(
                                prep,
                                diag,
                                &diag.historical_candidates,
                                &sim_values,
                                attribution_diag_sim_values,
                                reduce_op,
                            );
                            let attribution = pick_attribution_diag_portfolio(
                                prep,
                                diag,
                                &diag.attribution_candidates,
                                &sim_values,
                                attribution_diag_sim_values,
                                reduce_op,
                            );
                            let format_pick = |pick: Option<(usize, f32)>| {
                                pick.and_then(|(candidate_index, score)| {
                                    prep.candidates.get(candidate_index).map(|candidate| {
                                        format!(
                                            "{}:{}@{score:.6e}",
                                            candidate.node_name, candidate.neuron_idx
                                        )
                                    })
                                })
                                .unwrap_or_else(|| "none".to_string())
                            };
                            attr_diag_detail = Some(format!(
                                "[attr-diag] decision slot={} row={} coverage=complete \
                                 historical_n={} attribution_n={} historical={} attribution={}",
                                prep.slot,
                                prep.straggler,
                                diag.historical_candidates.len(),
                                diag.attribution_candidates.len(),
                                format_pick(historical),
                                format_pick(attribution),
                            ));
                            if let (
                                Some((historical_candidate, historical_score)),
                                Some((attribution_candidate, attribution_score)),
                            ) = (historical, attribution)
                            {
                                if let Some(ordering) =
                                    attribution_diag_score_cmp(attribution_score, historical_score)
                                {
                                    counters.compared += 1;
                                    counters.top_change +=
                                        usize::from(attribution_candidate != historical_candidate);
                                    match ordering {
                                        std::cmp::Ordering::Greater => counters.prior_win += 1,
                                        std::cmp::Ordering::Equal => counters.prior_tie += 1,
                                        std::cmp::Ordering::Less => counters.prior_loss += 1,
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let Some((legacy_candidates, legacy_sides)) = prep.legacy_prefix() else {
                winners.push(None);
                continue;
            };
            let side_value = |side: &SideSlot| -> f32 {
                match side {
                    SideSlot::Infeasible => f32::INFINITY,
                    SideSlot::Sim(i) => sim_values
                        .get(*i)
                        .copied()
                        .flatten()
                        .unwrap_or(f32::NEG_INFINITY),
                    SideSlot::Failed => f32::NEG_INFINITY,
                }
            };
            let best = pick_kfsb_candidate(
                legacy_candidates,
                legacy_sides
                    .iter()
                    .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                kfsb_multi_reduce_op(self.config.kfsb_reduce_op),
            );
            if winner_probe && prep.slot < winner_probe_domains {
                let format_pick = |pick: Option<(usize, f32, f32)>| {
                    pick.map(|(idx, score, _)| {
                        format!(
                            "{}:{}@{score:.5}",
                            legacy_candidates[idx].node_name, legacy_candidates[idx].neuron_idx
                        )
                    })
                    .unwrap_or_else(|| "none".to_string())
                };
                let min_pick = pick_kfsb_candidate(
                    legacy_candidates,
                    legacy_sides
                        .iter()
                        .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                    crate::beta_crown::KfsbReduceOp::Min,
                );
                let max_pick = pick_kfsb_candidate(
                    legacy_candidates,
                    legacy_sides
                        .iter()
                        .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                    crate::beta_crown::KfsbReduceOp::Max,
                );
                let candidate_values = legacy_candidates
                    .iter()
                    .zip(legacy_sides)
                    .map(|(candidate, sides)| {
                        let active = side_value(&sides[0]);
                        let inactive = side_value(&sides[1]);
                        format!(
                            "{}:{}(a={active:.5},i={inactive:.5},min={:.5},max={:.5})",
                            candidate.node_name,
                            candidate.neuron_idx,
                            active.min(inactive),
                            active.max(inactive),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                winner_probe_lines.push(format!(
                    "slot={} straggler={} cached_gate={} cached_candidates={}/{} min={} max={} diverged={} candidates=[{}]",
                    prep.slot,
                    prep.straggler,
                    cached_la_enabled as u8,
                    prep.cached_score_candidates,
                    legacy_candidates.len(),
                    format_pick(min_pick),
                    format_pick(max_pick),
                    min_pick.map(|pick| pick.0) != max_pick.map(|pick| pick.0),
                    candidate_values,
                ));
            }
            let Some((one_step_winner, one_step_score, _)) = best else {
                winners.push(None);
                continue; // no evaluable candidate — advisory fallback
            };
            let (parent_idx, domain, _) = &domains_with_unstable[prep.slot];
            let typed_override = depth_two_lookahead_selection
                .as_ref()
                .and_then(|selection| {
                    resolve_adaptive_depth_authority_candidate(
                        selection,
                        prep_index,
                        *parent_idx,
                        prep,
                        &sim_values,
                        &sims,
                        kfsb_multi_reduce_op(self.config.kfsb_reduce_op),
                    )
                });
            let decision = if let Some((winner, score)) = typed_override {
                KfsbWinnerDecision {
                    selected_candidate: winner,
                    selected_score: score,
                }
            } else {
                KfsbWinnerDecision {
                    selected_candidate: one_step_winner,
                    selected_score: one_step_score,
                }
            };
            let (winner, score) = decision.selected();
            if kfsb_probe {
                let parent_lb = domain
                    .objective_bounds
                    .get(prep.straggler)
                    .map(|(l, _)| *l)
                    .unwrap_or(f32::NAN);
                probe_lines.push(format!(
                    "slot={} cand={}:{} score={:.5} lift={:.5} n_cands={}",
                    prep.slot,
                    prep.candidates[winner].node_name,
                    prep.candidates[winner].neuron_idx,
                    score,
                    score - parent_lb,
                    legacy_candidates.len(),
                ));
            }
            winners.push(Some(decision));
        }
        if let Some(detail) = attr_diag_detail {
            eprintln!("{detail}");
        }
        if let Some(counters) = attr_diag_counters {
            let target_slot = attr_diag_target_slot
                .map(|slot| slot.to_string())
                .unwrap_or_else(|| "none".to_string());
            eprintln!(
                "[attr-diag] wave selected={} target_slot={} target_prepared={} row_hit={} \
                 row_miss={} root_prior_stale={} candidate_incomplete={} resource_capped={} \
                 simulation_incomplete={} compared={} top_change={} prior_win={} prior_tie={} \
                 prior_loss={} prior_published={}",
                usize::from(attr_diag_target_slot.is_some()),
                target_slot,
                counters.target_prepared,
                counters.row_hit,
                counters.row_miss,
                counters.root_prior_stale,
                counters.candidate_incomplete,
                counters.resource_capped,
                counters.simulation_incomplete,
                counters.compared,
                counters.top_change,
                counters.prior_win,
                counters.prior_tie,
                counters.prior_loss,
                usize::from(crate::network::gap_attribution::attribution_prior_published()),
            );
        }

        // ── 5: commit the winner. When the reference batch-fill policy asks
        // for d>1, rank distinct simulation-safe candidates after the winner
        // and build their complete feasible truth table from the untouched
        // parent. Any shortage, construction fault, cap, or deadline refusal
        // leaves the existing depth-one simulations untouched for fallback.
        // #bab-throughput: the third unmeasured stage. At d>1 every parent pays
        // a `2^d` truth-table expansion, each leaf built by `with_constraint`
        // from the untouched parent — a cost that scales with the wave depth
        // and is invisible in both `mo-wave-stage` and the sim line above.
        let __commit_t =
            crate::phase_telemetry::phase_telemetry_enabled().then(std::time::Instant::now);
        let mut depth_fallbacks = kfsb_probe.then(Vec::<String>::new);
        let mut committed: HashMap<usize, KfsbMultiChildren> = HashMap::new();
        for (prep, winner) in preps.into_iter().zip(&winners) {
            let (parent_idx, parent, _) = &domains_with_unstable[prep.slot];

            // A complete simulated candidate pair covers the untouched parent,
            // not merely either prospective child. When the selected row is
            // the parent's sole straggler, publish that parent-wide proof
            // directly and avoid constructing/evaluating split leaves that
            // would both be terminal. Candidate simulation, ranking, and
            // optional observers have already run at this point. Every
            // admitted pair is folded by `reusable_kfsb_lower_for_child`, and
            // only the typed `ChildComplete` result may cross this boundary.
            // A partial/refused proposal is discarded without consuming `sims`
            // and the historical split path below remains byte-for-byte.
            let only_straggler_unresolved = prep.straggler < thresholds.len()
                && parent.verified().len() == thresholds.len()
                && parent
                    .verified()
                    .iter()
                    .enumerate()
                    .all(|(row, &verified)| verified == (row != prep.straggler));
            if only_straggler_unresolved {
                if let Some(certificates) = simulated_lower_certificates.as_ref() {
                    let mut parent_cover = parent.clone_for_verified_close();
                    let effect = apply_kfsb_reusable_lower_certificate(
                        kfsb_cert_authority,
                        std::time::Instant::now(),
                        parent,
                        &mut parent_cover,
                        &prep,
                        certificates,
                        thresholds,
                    );
                    if let KfsbCertEffect::ChildComplete(receipt) = effect {
                        let effect = KfsbCertEffect::ParentComplete(receipt);
                        // No later phase can consume this parent's simulated
                        // children. Release their potentially large domain
                        // caches before retaining the lightweight close shell.
                        for side in prep.sides.iter().flatten() {
                            if let SideSlot::Sim(sim_index) = side {
                                if let Some(simulation) = sims.get_mut(*sim_index) {
                                    simulation.take();
                                }
                            }
                        }
                        committed.insert(*parent_idx, vec![(parent_cover, false, effect)]);
                        continue;
                    }
                }
            }

            let Some(decision) = *winner else {
                continue;
            };
            let (winner, _) = decision.selected();
            let parent_split_depth = cap_multi_objective_parent_depth(
                wave_split_depth,
                parent.depth(),
                self.config.max_depth,
            );
            // `prefilter_batch` admits only parents below max_depth. Keep the
            // depth-one fallback sound if that invariant is ever changed.
            if parent_split_depth == 0 {
                if let Some(fallbacks) = depth_fallbacks.as_mut() {
                    fallbacks.push(format!("{parent_idx}:max_depth"));
                }
                continue;
            }

            // Ranking may keep a candidate with one failed side under `Max`:
            // max(finite, -inf) is finite. It must never commit, because the
            // surviving child alone does not cover the parent. Also reject a
            // missing/taken or duplicated simulation before any adaptive
            // replay can consume state. With no committed entry the caller's
            // ordinary selector reconstructs a complete split instead.
            let winner_cover_complete = prep
                .candidates
                .get(winner)
                .zip(prep.sides.get(winner))
                .is_some_and(|(candidate, sides)| {
                    candidate_cover_complete(candidate, sides, &sims)
                });
            if !winner_cover_complete {
                if let Some(fallbacks) = depth_fallbacks.as_mut() {
                    fallbacks.push(format!("{parent_idx}:winner_incomplete_cover"));
                }
                continue;
            }

            let mut expanded = None;
            if parent_split_depth > 1 && expanded.is_none() {
                let Some((legacy_candidates, legacy_sides)) = prep.legacy_prefix() else {
                    if let Some(fallbacks) = depth_fallbacks.as_mut() {
                        fallbacks.push(format!("{parent_idx}:invalid_legacy_prefix"));
                    }
                    continue;
                };
                let side_value = |side: &SideSlot| -> f32 {
                    match side {
                        SideSlot::Infeasible => f32::INFINITY,
                        SideSlot::Sim(i) => sim_values
                            .get(*i)
                            .copied()
                            .flatten()
                            .unwrap_or(f32::NEG_INFINITY),
                        SideSlot::Failed => f32::NEG_INFINITY,
                    }
                };
                let side_values: Vec<(f32, f32)> = legacy_sides
                    .iter()
                    .map(|sides| (side_value(&sides[0]), side_value(&sides[1])))
                    .collect();
                let commit_safe = |candidate_idx: usize| {
                    prep.candidates
                        .get(candidate_idx)
                        .zip(prep.sides.get(candidate_idx))
                        .is_some_and(|(candidate, sides)| {
                            candidate_cover_complete(candidate, sides, &sims)
                        })
                };

                let mut plan = Vec::with_capacity(parent_split_depth);
                if commit_safe(winner) {
                    let candidate = &prep.candidates[winner];
                    plan.push((
                        candidate.node_name.clone(),
                        candidate.neuron_idx,
                        candidate.main_score,
                    ));
                    for (candidate_idx, _) in rank_adaptive_depth_candidates(
                        legacy_candidates,
                        &side_values,
                        kfsb_multi_reduce_op(self.config.kfsb_reduce_op),
                    ) {
                        if plan.len() == parent_split_depth {
                            break;
                        }
                        if !commit_safe(candidate_idx) {
                            continue;
                        }
                        let candidate = &prep.candidates[candidate_idx];
                        if plan.iter().any(|(node, neuron, _)| {
                            node == &candidate.node_name && *neuron == candidate.neuron_idx
                        }) {
                            continue;
                        }
                        plan.push((
                            candidate.node_name.clone(),
                            candidate.neuron_idx,
                            candidate.main_score,
                        ));
                    }
                }

                if multi_depth_plan_is_complete(plan.len(), parent_split_depth) {
                    let max_leaves_per_parent =
                        1usize.checked_shl(parent_split_depth as u32).unwrap_or(2);
                    match expand_multi_objective_truth_table(
                        graph,
                        parent,
                        thresholds,
                        &plan,
                        max_leaves_per_parent,
                        authority_deadline,
                    ) {
                        Ok(children) => expanded = Some(children),
                        Err(reason) => {
                            if let Some(fallbacks) = depth_fallbacks.as_mut() {
                                fallbacks.push(format!("{parent_idx}:{reason:?}"));
                            }
                        }
                    }
                } else if let Some(fallbacks) = depth_fallbacks.as_mut() {
                    fallbacks.push(format!(
                        "{parent_idx}:plan_shortage(have={},need={parent_split_depth})",
                        plan.len(),
                    ));
                }
            }

            let children = match expanded {
                Some(children) => children,
                None => {
                    let (fallback_winner, _) = decision.selected();
                    let Some(children) = prep
                        .candidates
                        .get(fallback_winner)
                        .zip(prep.sides.get(fallback_winner))
                        .and_then(|(candidate, sides)| {
                            take_complete_candidate_cover(candidate, sides, &mut sims)
                        })
                    else {
                        // Recheck at the mutation boundary. This is defensive
                        // for the selected winner and required for a restored
                        // historical winner: its original incomplete cover
                        // must preserve the ordinary-selector fallback rather
                        // than publishing either reachable half alone.
                        continue;
                    };
                    children
                }
            };
            let children: KfsbMultiChildren = children
                .into_iter()
                .map(|(mut child, is_active)| {
                    // Strict pre-publication check per leaf. A large depth-four
                    // truth table can cross the deadline while it is being
                    // assembled; every accepted effect carries the single
                    // wave-entry authority through the final boundary.
                    let cert_effect = simulated_lower_certificates.as_ref().map_or(
                        KfsbCertEffect::None,
                        |certificates| {
                            apply_kfsb_reusable_lower_certificate(
                                kfsb_cert_authority,
                                std::time::Instant::now(),
                                parent,
                                &mut child,
                                &prep,
                                certificates,
                                thresholds,
                            )
                        },
                    );
                    (child, is_active, cert_effect)
                })
                .collect();
            committed.insert(*parent_idx, children);
        }
        if let Some(t) = __commit_t {
            let leaves: usize = committed.values().map(|children| children.len()).sum();
            eprintln!(
                "[phase] mo-kfsb-commit parents={} leaves={} wave_depth={} secs={:.2}",
                committed.len(),
                leaves,
                wave_split_depth,
                t.elapsed().as_secs_f64(),
            );
        }
        // The common post-child-creation boundary precomputes Complete Clip
        // decisions for both these committed leaves and every per-domain
        // fallback. Keep this flattened view only for deterministic telemetry.
        let committed_children: Vec<&MultiObjectiveGraphBabDomain> = domains_with_unstable
            .iter()
            .flat_map(|(parent_idx, _, _)| {
                committed
                    .get(parent_idx)
                    .into_iter()
                    .flat_map(|children| children.iter().map(|(child, _, _)| child))
            })
            .collect();
        let kfsb_cert_receipts = committed
            .values()
            .flat_map(|children| children.iter())
            .filter(|(_, _, effect)| effect.receipt().is_some())
            .count();
        let kfsb_parent_closes = committed
            .values()
            .flat_map(|children| children.iter())
            .filter(|(_, _, effect)| matches!(effect, KfsbCertEffect::ParentComplete(_)))
            .count();
        probe_kfsb_cert_authority(
            kfsb_probe,
            kfsb_cert_reuse_armed,
            kfsb_cert_authority,
            domains_with_unstable.len(),
            committed.len(),
            kfsb_cert_receipts,
        );
        if kfsb_probe {
            let mut depth_counts = std::collections::BTreeMap::<usize, usize>::new();
            for child in &committed_children {
                *depth_counts.entry(child.depth()).or_default() += 1;
            }
            let depth_hist = if depth_counts.is_empty() {
                "none".to_string()
            } else {
                depth_counts
                    .into_iter()
                    .map(|(depth, count)| format!("{depth}:{count}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let fallbacks = depth_fallbacks
                .as_ref()
                .filter(|items| !items.is_empty())
                .map(|items| items.join(","))
                .unwrap_or_else(|| "none".to_string());
            eprintln!(
                "[kfsb-multi] wave: domains={} sims={} committed={} parent_closes={} requested_depth={} wave_depth={} leaves={} depth_hist={} fallbacks={} | {}",
                domains_with_unstable.len(),
                sim_owner.len(),
                committed.len(),
                kfsb_parent_closes,
                requested_split_depth,
                wave_split_depth,
                committed_children.len(),
                depth_hist,
                fallbacks,
                probe_lines.join(" ; ")
            );
        }
        if winner_probe {
            eprintln!(
                "[kfsb-winner-oracle] domains={} logged={} | {}",
                domains_with_unstable.len(),
                winner_probe_lines.len(),
                winner_probe_lines.join(" ; ")
            );
        }
        committed
    }

    /// Re-score one worst parent's post-f32 top-three candidate portfolio with
    /// the certified CPU-f64 lineage fold. This function is telemetry-only:
    /// every authoritative argument is immutable and the only mutable state is
    /// the verifier-lifetime one-shot atomic.
    fn observe_kfsb_f64_shadow(
        &self,
        graph: &GraphNetwork,
        objectives: &[Vec<f32>],
        preps: &[DomainPrep],
        sim_values: &[Option<f32>],
        sims: &[Option<MultiObjectiveGraphBabDomain>],
        capture: KfsbF64ShadowCapture,
        authority_deadline: Option<std::time::Instant>,
    ) {
        let Some(prep) = preps.get(capture.prep_index) else {
            return;
        };
        let Some((legacy_candidates, legacy_sides)) = prep.legacy_prefix() else {
            return;
        };
        if !capture.complete() {
            eprintln!(
                "[kfsb-f64-shadow] authority=false one_shot=true \
                 skipped=incomplete-f32-capture slot={} straggler={} top_k={} folds=0",
                prep.slot,
                prep.straggler,
                capture.top.len(),
            );
            return;
        }
        let Some(authoritative_objective) = objectives.get(prep.straggler) else {
            return;
        };
        // `child_bound_value` scores lower(c) in the normal mode and
        // -upper(c) in verify-upper mode. The latter is exactly lower(-c), so
        // negating the output row makes the certified f64 observer compare the
        // same quantity as the authoritative f32 selector.
        let shadow_objective = kfsb_f64_shadow_objective(
            authoritative_objective.as_slice(),
            self.config.verify_upper_bound,
        );
        let reduce_op = kfsb_multi_reduce_op(self.config.kfsb_reduce_op);
        let side_value = |side: &SideSlot| -> f32 {
            match side {
                SideSlot::Infeasible => f32::INFINITY,
                SideSlot::Sim(index) => sim_values
                    .get(*index)
                    .copied()
                    .flatten()
                    .unwrap_or(f32::NEG_INFINITY),
                SideSlot::Failed => f32::NEG_INFINITY,
            }
        };
        let all_f32_sides: Vec<(f32, f32)> = legacy_sides
            .iter()
            .map(|sides| (side_value(&sides[0]), side_value(&sides[1])))
            .collect();
        let authoritative_top = rank_kfsb_candidate_portfolio(
            legacy_candidates,
            &all_f32_sides,
            reduce_op,
            KFSB_F64_SHADOW_TOP_K,
        );
        let captured_top: Vec<usize> = capture
            .top
            .iter()
            .map(|candidate| candidate.candidate_index)
            .collect();
        if authoritative_top.len() != KFSB_F64_SHADOW_TOP_K
            || authoritative_top
                .iter()
                .map(|(candidate_index, _)| *candidate_index)
                .ne(captured_top.iter().copied())
        {
            eprintln!(
                "[kfsb-f64-shadow] authority=false one_shot=true \
                 skipped=portfolio-mismatch slot={} straggler={} captured={captured_top:?} \
                 authoritative={:?} folds=0",
                prep.slot,
                prep.straggler,
                authoritative_top
                    .iter()
                    .map(|(candidate_index, _)| *candidate_index)
                    .collect::<Vec<_>>(),
            );
            return;
        }
        let started = std::time::Instant::now();
        // Reuse the effective deadline captured at wave entry. In particular,
        // a call-scoped ledger/MIP handoff can be earlier than the verifier's
        // configured alpha deadline; re-reading that static value here would
        // let this telemetry-only observer spend reserved post-BaB time.
        let Some(shadow_deadline) = kfsb_f64_shadow_deadline(started, authority_deadline) else {
            eprintln!(
                "[kfsb-f64-shadow] authority=false one_shot=true skipped=authority-reserve \
                 slot={} straggler={} top_k={} folds=0",
                prep.slot,
                prep.straggler,
                capture.top.len(),
            );
            return;
        };

        let full_f32_pick =
            pick_kfsb_candidate(legacy_candidates, all_f32_sides.iter().copied(), reduce_op);
        let format_candidate = |candidate_index: usize, score: f32| -> String {
            legacy_candidates
                .get(candidate_index)
                .map(|candidate| {
                    format!(
                        "{}:{}@{score:.6}",
                        candidate.node_name, candidate.neuron_idx
                    )
                })
                .unwrap_or_else(|| "invalid".to_string())
        };
        let f32_winner = full_f32_pick
            .map(|(candidate_index, score, _)| format_candidate(candidate_index, score))
            .unwrap_or_else(|| "none".to_string());

        let mut attempted_folds = 0usize;
        let mut budget_expired = false;
        let mut observed_side_values: Vec<Option<(f32, f32)>> =
            Vec::with_capacity(KFSB_F64_SHADOW_TOP_K);
        let mut detail = Vec::with_capacity(KFSB_F64_SHADOW_TOP_K);
        for (rank, captured) in capture.top.iter().enumerate() {
            let Some(sides) = prep.sides.get(captured.candidate_index) else {
                observed_side_values.push(None);
                detail.push(format!(
                    "r{rank}=invalid(f32={:.6},f64=unsupported)",
                    captured.f32_score
                ));
                continue;
            };
            let mut f64_sides: [Option<f32>; 2] = [None, None];
            for (side_index, side) in sides.iter().enumerate() {
                match side {
                    SideSlot::Infeasible => {
                        f64_sides[side_index] = Some(f32::INFINITY);
                    }
                    SideSlot::Failed => {}
                    SideSlot::Sim(sim_index) => {
                        if !kfsb_f64_shadow_budget_available(
                            std::time::Instant::now(),
                            shadow_deadline,
                            authority_deadline,
                        ) {
                            budget_expired = true;
                            continue;
                        }
                        let Some(node_bounds) = captured.side_node_bounds[side_index].as_ref()
                        else {
                            continue;
                        };
                        let Some(child) = sims.get(*sim_index).and_then(Option::as_ref) else {
                            continue;
                        };
                        let Some(fold) =
                            crate::beta_crown::engine::graph::propagation::batched::prep_resnet_domain(
                                graph,
                                graph.output_name(),
                                node_bounds,
                                child.input_bounds.as_ref(),
                                Some(child.beta_state()),
                                Some(child.alpha_state()),
                                crate::network::bab_chain_wide_enabled(),
                            )
                        else {
                            continue;
                        };
                        if !kfsb_f64_shadow_budget_available(
                            std::time::Instant::now(),
                            shadow_deadline,
                            authority_deadline,
                        ) {
                            budget_expired = true;
                            continue;
                        }
                        attempted_folds += 1;
                        let observed = crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::sound_f64_lower_bound_with_deadline(
                            &fold.segments,
                            shadow_objective.as_ref(),
                            &fold.beta_signed,
                            &fold.in_lo,
                            &fold.in_hi,
                            None,
                            Some(shadow_deadline),
                        );
                        if kfsb_f64_shadow_budget_available(
                            std::time::Instant::now(),
                            shadow_deadline,
                            authority_deadline,
                        ) {
                            f64_sides[side_index] = observed;
                        } else {
                            budget_expired = true;
                        }
                    }
                }
            }
            let observed = f64_sides[0].zip(f64_sides[1]);
            let observed_score =
                observed.map(|(active, inactive)| kfsb_reduce(reduce_op, active, inactive));
            observed_side_values.push(observed);
            let candidate = &legacy_candidates[captured.candidate_index];
            let f64_text = observed_score
                .filter(|score| !score.is_nan())
                .map(|score| format!("{score:.6}"))
                .unwrap_or_else(|| "unsupported".to_string());
            detail.push(format!(
                "r{rank}={}:{}(f32_a={:.6},f32_i={:.6},f32={:.6},f64={f64_text})",
                candidate.node_name,
                candidate.neuron_idx,
                captured.f32_sides[0],
                captured.f32_sides[1],
                captured.f32_score,
            ));
        }

        // Price the f64 winner only when every retained candidate produced both
        // sides. A partial portfolio is logged as incomplete, never promoted or
        // silently compared against fewer alternatives.
        let complete = observed_side_values.iter().all(Option::is_some);
        let f64_winner = if complete {
            let indexed_side_values: Vec<(usize, (f32, f32))> = capture
                .top
                .iter()
                .zip(&observed_side_values)
                .filter_map(|(candidate, values)| {
                    values.map(|values| (candidate.candidate_index, values))
                })
                .collect();
            pick_kfsb_candidate_subset_original_order(
                legacy_candidates,
                &indexed_side_values,
                reduce_op,
            )
            .map(|(candidate_index, score, _)| format_candidate(candidate_index, score))
            .unwrap_or_else(|| "none".to_string())
        } else {
            "none".to_string()
        };
        eprintln!(
            "[kfsb-f64-shadow] authority=false one_shot=true slot={} straggler={} \
             direction={} top_k={} complete={} folds={} budget_expired={} f32_winner={} \
             f64_top3_winner={} elapsed_s={:.6} | {}",
            prep.slot,
            prep.straggler,
            if self.config.verify_upper_bound {
                "negated-upper"
            } else {
                "lower"
            },
            capture.top.len(),
            u8::from(complete),
            attempted_folds,
            u8::from(budget_expired),
            f32_winner,
            f64_winner,
            started.elapsed().as_secs_f64(),
            detail.join(" ; "),
        );
    }

    /// Select the second-level BaBSR branch independently for one private
    /// lookahead child.
    ///
    /// The child's `node_bounds` must already be the constrained-forward
    /// fixpoint captured by its one-step kFSB simulation. Recomputing the
    /// objective-directed score against THAT cache is the critical difference
    /// from the historical prefix probe, which reused one global second
    /// candidate for both root children.
    #[cfg(test)]
    pub(super) fn select_adaptive_depth_base_candidate(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        self.select_adaptive_depth_base_candidate_impl(
            graph,
            relu_nodes,
            domain,
            objective,
            kfsb_multi_reduce_op(self.config.kfsb_reduce_op),
            None,
        )
    }

    /// Paper-shaped second-level BaBSR score. Unlike the legacy M27 observer,
    /// this typed lane pins the conservative `Min` score and is not affected
    /// by measurement-only `NY_MO_KFSB_REDUCE` overrides.
    fn select_depth_two_base_candidate_with_budget(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
        shadow_deadline: std::time::Instant,
        authority_deadline: Option<std::time::Instant>,
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        self.select_adaptive_depth_base_candidate_impl(
            graph,
            relu_nodes,
            domain,
            objective,
            crate::beta_crown::KfsbReduceOp::Min,
            Some((shadow_deadline, authority_deadline)),
        )
    }

    fn select_adaptive_depth_base_candidate_impl(
        &self,
        graph: &GraphNetwork,
        relu_nodes: &[String],
        domain: &MultiObjectiveGraphBabDomain,
        objective: &[f32],
        reduce_op: crate::beta_crown::KfsbReduceOp,
        budget: Option<(std::time::Instant, Option<std::time::Instant>)>,
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        let check_budget = || {
            if budget.is_some_and(|(shadow_deadline, authority_deadline)| {
                !adaptive_depth_shadow_budget_available(
                    std::time::Instant::now(),
                    shadow_deadline,
                    authority_deadline,
                )
            }) {
                Err(ny_core::NyError::DeadlineExceeded(
                    "branch-specific BaBSR shadow side exhausted its private budget".to_string(),
                ))
            } else {
                Ok(())
            }
        };
        check_budget()?;
        let unstable = self.find_unstable_graph_neurons_multi(graph, domain, relu_nodes);
        check_budget()?;
        if unstable.is_empty() {
            return Ok(None);
        }
        // #string-key-churn: `unstable` holds one entry per unstable NEURON
        // (measured: 15,004 across 16 domains on cifar100 idx_8600 round 1) but
        // spans only ~20 distinct node names. Cloning inside the collect
        // allocated a String per neuron to build a set of twenty. Dedupe on
        // `&str` first and clone only what survives.
        let unstable_nodes: std::collections::HashSet<String> = {
            let mut distinct: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for (name, _) in &unstable {
                distinct.insert(name.as_str());
            }
            distinct.into_iter().map(str::to_string).collect()
        };
        let score_parts = if let Some((shadow_deadline, _)) = budget {
            self.compute_graph_babsr_scores_from_bounds_until(
                graph,
                domain.node_bounds(),
                domain.input_bounds(),
                reduce_op,
                Some(objective),
                Some(&unstable_nodes),
                shadow_deadline,
            )?
        } else {
            self.compute_graph_babsr_scores_from_bounds(
                graph,
                domain.node_bounds(),
                domain.input_bounds(),
                reduce_op,
                Some(objective),
                Some(&unstable_nodes),
            )?
        };
        check_budget()?;
        let mut candidates: Vec<GraphKfsbCandidate> = unstable
            .into_iter()
            .filter_map(|(node_name, neuron_idx)| {
                // #string-key-churn: the map is keyed `(String, usize)`, so a
                // borrowed lookup is not possible without a second index. Build
                // the key by MOVING the name we already own, copy the (2 x f32,
                // `Copy`) parts out, then move the name back -- instead of
                // allocating a throwaway String per unstable neuron.
                let key = (node_name, neuron_idx);
                let parts = *score_parts.get(&key)?;
                let (node_name, neuron_idx) = key;
                parts.main_score.is_finite().then_some(GraphKfsbCandidate {
                    node_name,
                    neuron_idx,
                    main_score: parts.main_score,
                    backup_score: parts.backup_score,
                })
            })
            .collect();
        candidates.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
                .then_with(|| {
                    crate::cmp_utils::nan_last_descending_cmp(&a.backup_score, &b.backup_score)
                })
                .then_with(|| a.node_name.cmp(&b.node_name))
                .then_with(|| a.neuron_idx.cmp(&b.neuron_idx))
        });
        check_budget()?;
        candidates.into_iter().next().map(Some).ok_or_else(|| {
            ny_core::NyError::InternalError(
                "branch-specific lookahead found unstable neurons but no finite BaBSR score"
                    .to_string(),
            )
        })
    }

    /// Price the typed July-2026 depth-2 portfolio from branch-specific second
    /// BaBSR scores. This is advice only: the result contains an identity for
    /// one original first-level candidate and no private bounds.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_depth_two_lookahead_advice(
        &self,
        graph: &GraphNetwork,
        domains_with_unstable: &[MultiObjDomainWithUnstable<'_>],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        preps: &[DomainPrep],
        sim_values: &[Option<f32>],
        sims: &[Option<MultiObjectiveGraphBabDomain>],
        policy: DepthTwoBranchLookaheadConfig,
        budget: DepthTwoLookaheadBudget,
        capture: Option<&DepthTwoLookaheadCapture>,
    ) -> Option<AdaptiveDepthAuthoritySelection> {
        if policy.mode == DepthTwoBranchLookaheadMode::Off
            || !depth_two_lookahead_policy_supported(policy)
            || !budget.available_now()
        {
            return None;
        }
        let capture = capture?;
        let prep = preps.get(capture.prep_index)?;
        let (legacy_candidates, legacy_sides) = prep.legacy_prefix()?;
        let candidate_indices = prep.depth_two_lookahead_candidates.as_ref()?;
        if candidate_indices.len() != policy.candidates
            || prep.candidates.len() != prep.sides.len()
            || candidate_indices
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != policy.candidates
        {
            return None;
        }
        let (parent_index, domain, _) = domains_with_unstable.get(prep.slot)?;
        let objective = objectives.get(prep.straggler)?;
        let budget_available = || budget.available_now();

        // Pin the fallback root to the exact historical one-step portfolio.
        // Missing simulations retain the legacy -inf behavior here; only the
        // separate paper portfolio below requires complete finite evidence.
        let historical_side_value = |side: &SideSlot| -> f32 {
            match side {
                SideSlot::Infeasible => f32::INFINITY,
                SideSlot::Failed => f32::NEG_INFINITY,
                SideSlot::Sim(index) => sim_values
                    .get(*index)
                    .copied()
                    .flatten()
                    .unwrap_or(f32::NEG_INFINITY),
            }
        };
        let (historical_candidate, historical_score, _) = pick_kfsb_candidate(
            legacy_candidates,
            legacy_sides.iter().map(|sides| {
                (
                    historical_side_value(&sides[0]),
                    historical_side_value(&sides[1]),
                )
            }),
            crate::beta_crown::KfsbReduceOp::Min,
        )?;
        if !candidate_indices.contains(&historical_candidate) {
            // Advice cannot honestly preserve the historical root on a tie if
            // that root is outside the exact-total paper portfolio.
            return None;
        }

        let complete_side_value = |side: &SideSlot| -> Option<f32> {
            match side {
                SideSlot::Infeasible => Some(f32::INFINITY),
                SideSlot::Failed => None,
                SideSlot::Sim(index) => {
                    sims.get(*index)?.as_ref()?;
                    let value = sim_values.get(*index).copied().flatten()?;
                    value.is_finite().then_some(value)
                }
            }
        };
        let paper_one_step_values = candidate_indices
            .iter()
            .map(|&candidate_index| {
                let sides = prep.sides.get(candidate_index)?;
                complete_side_value(&sides[0]).zip(complete_side_value(&sides[1]))
            })
            .collect::<Option<Vec<_>>>()?;

        let mut scored: Vec<(usize, Option<f64>)> = Vec::with_capacity(policy.candidates);
        let mut detail = Vec::with_capacity(policy.candidates);
        for (&candidate_index, &(active_bound, inactive_bound)) in
            candidate_indices.iter().zip(&paper_one_step_values)
        {
            if !budget_available() {
                return None;
            }
            let sides = prep.sides.get(candidate_index)?;
            let mut second = [DepthTwoLookaheadSideScore::Finite(0.0); 2];
            let mut choices = [String::new(), String::new()];
            for (side_index, side) in sides.iter().enumerate() {
                match side {
                    SideSlot::Infeasible => {
                        second[side_index] = DepthTwoLookaheadSideScore::Infeasible;
                        choices[side_index] = "empty".to_string();
                    }
                    SideSlot::Failed => return None,
                    SideSlot::Sim(sim_index) => {
                        let mut child = sims.get(*sim_index)?.as_ref()?.clone();
                        let node_bounds = capture.node_bounds(*sim_index)?;
                        child.node_bounds =
                            crate::beta_crown::domain::NodeBoundsMap::from_shared_hash_map(
                                node_bounds.clone(),
                            );
                        child.delta_pre_nodes.clear();
                        if !clear_shadow_cached_las(&mut child) || !budget_available() {
                            return None;
                        }
                        match self.select_depth_two_base_candidate_with_budget(
                            graph,
                            &child,
                            relu_nodes,
                            objective,
                            budget.private_deadline,
                            budget.authority_deadline,
                        ) {
                            Ok(Some(candidate))
                                if candidate.main_score.is_finite()
                                    && candidate.main_score >= 0.0 =>
                            {
                                second[side_index] = DepthTwoLookaheadSideScore::Finite(f64::from(
                                    candidate.main_score,
                                ));
                                choices[side_index] =
                                    format!("{}:{}", candidate.node_name, candidate.neuron_idx);
                            }
                            Ok(None) if budget_available() => {
                                // No remaining unstable split: no fabricated
                                // second-level gain for this terminal side.
                                second[side_index] = DepthTwoLookaheadSideScore::Finite(0.0);
                                choices[side_index] = "terminal".to_string();
                            }
                            Ok(Some(_)) | Ok(None) | Err(_) => return None,
                        }
                    }
                }
            }
            if !budget_available() {
                return None;
            }
            let one_step = if active_bound == f32::INFINITY && inactive_bound == f32::INFINITY {
                f64::INFINITY
            } else {
                f64::from(active_bound.min(inactive_bound))
            };
            let score = depth_two_lookahead_score(one_step, second[0], second[1], policy.discount);
            scored.push((candidate_index, score));
            detail.push(format!(
                "{}:{}@one={one_step:.6}/second={}|{}/score={}",
                prep.candidates[candidate_index].node_name,
                prep.candidates[candidate_index].neuron_idx,
                choices[0],
                choices[1],
                score
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "invalid".to_string()),
            ));
        }
        if !budget_available() {
            return None;
        }
        let (winner, lookahead_score) =
            select_complete_depth_two_lookahead(&scored, policy.candidates, historical_candidate)?;
        let parent_value = self
            .config
            .child_bound_value(domain.objective_bounds().get(prep.straggler).copied());
        let authority = policy.mode == DepthTwoBranchLookaheadMode::Select;
        eprintln!(
            "[mo-depth2-lookahead] mode={:?} slot={} straggler={} parent={:.6} roots={} \
             one_step={}:{}@{:.6} selected={}:{}@{:.6} changed={} authority={} \
             budget_ms={} elapsed_ms={} candidates=[{}]",
            policy.mode,
            prep.slot,
            prep.straggler,
            parent_value,
            policy.candidates,
            prep.candidates[historical_candidate].node_name,
            prep.candidates[historical_candidate].neuron_idx,
            historical_score,
            prep.candidates[winner].node_name,
            prep.candidates[winner].neuron_idx,
            lookahead_score,
            u8::from(winner != historical_candidate),
            u8::from(authority),
            ADAPTIVE_DEPTH_SHADOW_BUDGET.as_millis(),
            budget.started_at.elapsed().as_millis(),
            detail.join(";"),
        );
        // Formatting or a blocked diagnostic sink may cross the immutable
        // private deadline. No Select identity can leave the observer without
        // one last post-telemetry admission check.
        if !budget_available() {
            return None;
        }
        let selection = authority.then(|| {
            adaptive_depth_authority_identity(capture.prep_index, *parent_index, prep, winner)
        })??;
        budget_available().then_some(selection)
    }

    /// Bounded replacement for the retired private dense M27/M28 treatment.
    ///
    /// It authenticates the exact historical winner plus two alternatives,
    /// then scans already-captured one-step child fixpoints for a branch-local
    /// second-decision ambiguity proxy. It constructs no child, score map,
    /// spec matrix, BatchedDomains, or backend workspace and cannot publish a
    /// root identity, replay receipt, bound, child, or verdict.
    #[allow(clippy::too_many_arguments)]
    fn observe_adaptive_depth_captured_fixpoint_proxy(
        &self,
        preps: &[DomainPrep],
        sim_values: &[Option<f32>],
        capture: Option<AdaptiveDepthShadowCapture>,
        target_prep: Option<usize>,
        budget: Option<DepthTwoLookaheadBudget>,
    ) {
        let Some(budget) = budget else {
            return;
        };
        let Some(prep_index) = target_prep else {
            return;
        };
        let Some(prep) = preps.get(prep_index) else {
            return;
        };
        if !budget.available_now() {
            eprintln!(
                "[mo-adaptive-fixpoint-proxy] outcome=declined reason=budget authority=0 commit=0"
            );
            return;
        }
        let Some(capture) = capture.filter(|capture| capture.prep_index == prep_index) else {
            eprintln!(
                "[mo-adaptive-fixpoint-proxy] outcome=declined reason=incomplete-capture authority=0 commit=0"
            );
            return;
        };
        let Some((legacy_candidates, legacy_sides)) = prep.legacy_prefix() else {
            return;
        };
        if legacy_candidates.len() != legacy_sides.len()
            || !(ADAPTIVE_DEPTH_SHADOW_ROOTS..=ADAPTIVE_DEPTH_SHADOW_MAX_CAPTURE_CANDIDATES)
                .contains(&legacy_candidates.len())
        {
            return;
        }

        // The ledger exists before the first size-dependent allocation. The
        // complete side-value vector and exact three-entry ranked portfolio
        // are the only private heap owners; capture storage is inline.
        let mut ledger = AdaptiveDepthPrivatePeakLedger::new(ADAPTIVE_DEPTH_PROXY_MAX_HEAP_BYTES);
        let Some(values_bytes) = legacy_sides.len().checked_mul(size_of::<(f32, f32)>()) else {
            return;
        };
        let Some(portfolio_bytes) =
            ADAPTIVE_DEPTH_SHADOW_ROOTS.checked_mul(size_of::<(usize, f32)>())
        else {
            return;
        };
        if ledger.admit([values_bytes, portfolio_bytes]).is_err() {
            eprintln!(
                "[mo-adaptive-fixpoint-proxy] outcome=declined reason=portfolio-peak authority=0 commit=0"
            );
            return;
        }

        let side_value = |side: &SideSlot| -> Option<f32> {
            match side {
                SideSlot::Infeasible => Some(f32::INFINITY),
                SideSlot::Failed => None,
                SideSlot::Sim(index) => {
                    capture.proxy_score(*index)?;
                    sim_values
                        .get(*index)
                        .copied()
                        .flatten()
                        .filter(|value| value.is_finite())
                }
            }
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(legacy_sides.len()).is_err() {
            return;
        }
        for sides in legacy_sides {
            if !budget.available_now() {
                return;
            }
            let Some(active) = side_value(&sides[0]) else {
                return;
            };
            let Some(inactive) = side_value(&sides[1]) else {
                return;
            };
            values.push((active, inactive));
        }
        let reduce_op = kfsb_multi_reduce_op(self.config.kfsb_reduce_op);
        let Some(portfolio) = rank_kfsb_candidate_portfolio_with_budget(
            legacy_candidates,
            &values,
            reduce_op,
            ADAPTIVE_DEPTH_SHADOW_ROOTS,
            || budget.available_now(),
        ) else {
            return;
        };
        if portfolio.len() != ADAPTIVE_DEPTH_SHADOW_ROOTS {
            return;
        }
        for left in 0..ADAPTIVE_DEPTH_SHADOW_ROOTS {
            for right in (left + 1)..ADAPTIVE_DEPTH_SHADOW_ROOTS {
                let a = &legacy_candidates[portfolio[left].0];
                let b = &legacy_candidates[portfolio[right].0];
                if a.node_name == b.node_name && a.neuron_idx == b.neuron_idx {
                    return;
                }
            }
        }

        let mut proxy_scores = [f32::NAN; ADAPTIVE_DEPTH_SHADOW_ROOTS];
        for (rank, (candidate_index, _)) in portfolio.iter().enumerate() {
            if !budget.available_now() {
                return;
            }
            let Some(sides) = legacy_sides.get(*candidate_index) else {
                return;
            };
            let Some(active) = adaptive_depth_captured_fixpoint_side_proxy(&sides[0], &capture)
            else {
                return;
            };
            let Some(inactive) = adaptive_depth_captured_fixpoint_side_proxy(&sides[1], &capture)
            else {
                return;
            };
            let score = kfsb_reduce(reduce_op, active, inactive);
            if score.is_nan() {
                return;
            }
            proxy_scores[rank] = score;
        }

        let Some(recommended_rank) = adaptive_depth_proxy_recommended_rank(&proxy_scores) else {
            return;
        };
        let historical = &legacy_candidates[portfolio[0].0];
        let recommended = &legacy_candidates[portfolio[recommended_rank].0];
        eprintln!(
            "[mo-adaptive-fixpoint-proxy] outcome=observed authority=0 commit=0 evaluated_dense_depth=0 roots=3 changed={} historical={}:{} recommended={}:{} proxy={:.6} capture_scores={} heap_payload_peak_bytes={} elapsed_ms={}",
            u8::from(recommended_rank != 0),
            historical.node_name,
            historical.neuron_idx,
            recommended.node_name,
            recommended.neuron_idx,
            proxy_scores[recommended_rank],
            capture.captured_score_count(),
            ledger.admitted_peak_bytes(),
            budget.started_at.elapsed().as_millis(),
        );
    }

    /// Append the target's missing exact-total roots only after every domain's
    /// historical prefix has crossed the Rayon preparation barrier.
    ///
    /// The append is transactional: all constraints and global indices are
    /// staged privately, and any deadline expiry, malformed identity, or
    /// construction failure drops the whole typed plan without touching
    /// `prep`, `sims`, or `sim_owner`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_depth_two_lookahead_overlay(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        thresholds: &[f32],
        prep_index: usize,
        prep: &mut DomainPrep,
        plan: DepthTwoLookaheadOverlayPlan,
        budget: DepthTwoLookaheadBudget,
        sims: &mut Vec<Option<MultiObjectiveGraphBabDomain>>,
        sim_owner: &mut Vec<(usize, usize)>,
    ) -> bool {
        let expected = plan.selected.len();
        if expected == 0
            || expected > DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES
            || prep.depth_two_lookahead_candidates.is_some()
            || prep.legacy_prefix().is_none()
            || !budget.available_now()
        {
            return false;
        }

        let mut seen = std::collections::HashSet::with_capacity(expected);
        let mut candidate_indices = Vec::with_capacity(expected);
        let mut overlay_candidates = Vec::new();
        let mut overlay_sides = Vec::new();
        let mut overlay_children = Vec::new();
        let sims_base = sims.len();

        for candidate in plan.selected {
            let identity = (candidate.node_name.clone(), candidate.neuron_idx);
            if !seen.insert(identity) || !budget.available_now() {
                return false;
            }
            if let Some(candidate_index) = prep.candidates.iter().position(|existing| {
                existing.node_name == candidate.node_name
                    && existing.neuron_idx == candidate.neuron_idx
            }) {
                let existing = &prep.candidates[candidate_index];
                if existing.main_score.to_bits() != candidate.main_score.to_bits()
                    || existing.backup_score.to_bits() != candidate.backup_score.to_bits()
                    || prep.sides[candidate_index]
                        .iter()
                        .any(|side| matches!(side, SideSlot::Failed))
                {
                    return false;
                }
                candidate_indices.push(candidate_index);
                continue;
            }

            let mut pair = Vec::with_capacity(2);
            for is_active in [true, false] {
                if !budget.available_now() {
                    return false;
                }
                let constraint = GraphNeuronConstraint {
                    node_name: candidate.node_name.clone(),
                    neuron_idx: candidate.neuron_idx,
                    is_active,
                    score: candidate.main_score,
                };
                let result = domain.with_constraint(graph, constraint, false, thresholds);
                if !budget.available_now() {
                    // `with_constraint` is not preemptible. A result completed
                    // outside the private envelope is discarded transactionally.
                    return false;
                }
                match result {
                    Ok(Some(child)) => {
                        let Some(sim_index) = sims_base.checked_add(overlay_children.len()) else {
                            return false;
                        };
                        overlay_children.push(child);
                        pair.push(SideSlot::Sim(sim_index));
                    }
                    Ok(None) => pair.push(SideSlot::Infeasible),
                    Err(ref error) if error.is_infeasible_domain() => {
                        pair.push(SideSlot::Infeasible);
                    }
                    Err(_) => return false,
                }
            }
            let Ok(pair) = <Vec<SideSlot> as TryInto<[SideSlot; 2]>>::try_into(pair) else {
                return false;
            };
            let Some(candidate_index) = prep.candidates.len().checked_add(overlay_candidates.len())
            else {
                return false;
            };
            candidate_indices.push(candidate_index);
            overlay_candidates.push(candidate);
            overlay_sides.push(pair);
        }
        if candidate_indices.len() != expected || !budget.available_now() {
            return false;
        }

        prep.candidates.extend(overlay_candidates);
        prep.sides.extend(overlay_sides);
        for child in overlay_children {
            sim_owner.push((prep_index, prep.straggler));
            sims.push(Some(child));
        }
        prep.depth_two_lookahead_candidates = Some(candidate_indices);
        true
    }

    /// Transactionally install the one-domain attribution diagnostic union.
    ///
    /// Existing incumbent or depth-two identities and their child domains are
    /// reused. Only missing identities are constructed, under the same private
    /// deadline that later bounds their diagnostic simulation. Any expiry or
    /// malformed identity drops the staged suffix and publishes incomplete
    /// telemetry; the already-prepared incumbent prefix is never discarded.
    #[allow(clippy::too_many_arguments)]
    fn append_attribution_diag_overlay(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        thresholds: &[f32],
        prep_index: usize,
        prep: &mut DomainPrep,
        plan: &AttributionDiagOverlayPlan,
        budget: DepthTwoLookaheadBudget,
        sims: &mut Vec<Option<MultiObjectiveGraphBabDomain>>,
        sim_owner: &mut Vec<(usize, usize)>,
    ) -> bool {
        if prep.attribution_diag.is_some()
            || prep.candidates.len() != prep.sides.len()
            || plan.coverage != AttributionDiagCoverage::Complete
            || !attribution_diag_plan_within_caps(plan)
            || !budget.available_now()
        {
            install_incomplete_attribution_diag(prep, plan);
            return false;
        }

        let mut planned_identities = std::collections::HashSet::new();
        let missing_identities = plan
            .historical_candidates
            .iter()
            .chain(&plan.attribution_candidates)
            .filter(|candidate| {
                planned_identities.insert((candidate.node_name.as_str(), candidate.neuron_idx))
            })
            .filter(|candidate| {
                !prep.candidates.iter().any(|existing| {
                    existing.node_name == candidate.node_name
                        && existing.neuron_idx == candidate.neuron_idx
                })
            })
            .count();
        if missing_identities > ATTRIBUTION_MAX_APPENDED_CANDIDATES
            || missing_identities.saturating_mul(2) > ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS
        {
            install_resource_capped_attribution_diag(prep, plan);
            return false;
        }

        let staged = (|| {
            let historical_identities: std::collections::HashSet<(String, usize)> = plan
                .historical_candidates
                .iter()
                .map(|candidate| (candidate.node_name.clone(), candidate.neuron_idx))
                .collect();
            let attribution_identities: std::collections::HashSet<(String, usize)> = plan
                .attribution_candidates
                .iter()
                .map(|candidate| (candidate.node_name.clone(), candidate.neuron_idx))
                .collect();
            let distinguishing_identities: std::collections::HashSet<(String, usize)> =
                historical_identities
                    .symmetric_difference(&attribution_identities)
                    .cloned()
                    .collect();
            let mut union = plan.historical_candidates.clone();
            append_unique_kfsb_candidates(&mut union, &plan.attribution_candidates);
            if union.len() > ATTRIBUTION_MAX_UNION_CANDIDATES {
                return None;
            }
            let mut identity_to_index: HashMap<(String, usize), usize> = HashMap::new();
            let mut overlay_candidates = Vec::new();
            let mut overlay_sides = Vec::new();
            let mut overlay_children = Vec::new();
            let sims_base = sims.len();

            for candidate in union {
                if !budget.available_now() {
                    return None;
                }
                let identity = (candidate.node_name.clone(), candidate.neuron_idx);
                let is_distinguishing = distinguishing_identities.contains(&identity);
                let mut existing = prep.candidates.iter().enumerate().filter(|(_, member)| {
                    member.node_name == candidate.node_name
                        && member.neuron_idx == candidate.neuron_idx
                });
                if let Some((candidate_index, member)) = existing.next() {
                    let sides = prep.sides.get(candidate_index)?;
                    if existing.next().is_some()
                        || member.main_score.to_bits() != candidate.main_score.to_bits()
                        || member.backup_score.to_bits() != candidate.backup_score.to_bits()
                        || (is_distinguishing
                            && sides.iter().any(|side| matches!(side, SideSlot::Failed)))
                        || identity_to_index
                            .insert(identity, candidate_index)
                            .is_some()
                    {
                        return None;
                    }
                    continue;
                }

                let mut pair = Vec::with_capacity(2);
                let mut usable = false;
                for is_active in [true, false] {
                    if !budget.available_now() {
                        return None;
                    }
                    let constraint = GraphNeuronConstraint {
                        node_name: candidate.node_name.clone(),
                        neuron_idx: candidate.neuron_idx,
                        is_active,
                        score: candidate.main_score,
                    };
                    let result = domain.with_constraint(graph, constraint, false, thresholds);
                    if !budget.available_now() {
                        return None;
                    }
                    match result {
                        Ok(Some(child)) => {
                            if overlay_children.len() == ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS {
                                return None;
                            }
                            let sim_index = sims_base.checked_add(overlay_children.len())?;
                            overlay_children.push(child);
                            pair.push(SideSlot::Sim(sim_index));
                            usable = true;
                        }
                        Ok(None) => {
                            pair.push(SideSlot::Infeasible);
                            usable = true;
                        }
                        Err(ref error) if error.is_infeasible_domain() => {
                            pair.push(SideSlot::Infeasible);
                            usable = true;
                        }
                        Err(_) if is_distinguishing => return None,
                        Err(_) => pair.push(SideSlot::Failed),
                    }
                }
                if !usable {
                    return None;
                }
                let pair: [SideSlot; 2] = pair.try_into().ok()?;
                let candidate_index = prep
                    .candidates
                    .len()
                    .checked_add(overlay_candidates.len())?;
                if overlay_candidates.len() == ATTRIBUTION_MAX_APPENDED_CANDIDATES {
                    return None;
                }
                if identity_to_index
                    .insert(identity, candidate_index)
                    .is_some()
                {
                    return None;
                }
                overlay_candidates.push(candidate);
                overlay_sides.push(pair);
            }

            let map_portfolio = |portfolio: &[GraphKfsbCandidate]| {
                portfolio
                    .iter()
                    .map(|candidate| {
                        identity_to_index
                            .get(&(candidate.node_name.clone(), candidate.neuron_idx))
                            .copied()
                    })
                    .collect::<Option<Vec<_>>>()
            };
            let historical_candidates = map_portfolio(&plan.historical_candidates)?;
            let attribution_candidates = map_portfolio(&plan.attribution_candidates)?;
            let mut distinguishing_candidates = Vec::new();
            let mut distinguishing_seen = std::collections::HashSet::new();
            for (candidate, &candidate_index) in plan
                .historical_candidates
                .iter()
                .zip(&historical_candidates)
            {
                let shared = plan.attribution_candidates.iter().any(|other| {
                    other.node_name == candidate.node_name
                        && other.neuron_idx == candidate.neuron_idx
                });
                if !shared && distinguishing_seen.insert(candidate_index) {
                    distinguishing_candidates.push(candidate_index);
                }
            }
            for (candidate, &candidate_index) in plan
                .attribution_candidates
                .iter()
                .zip(&attribution_candidates)
            {
                let shared = plan.historical_candidates.iter().any(|other| {
                    other.node_name == candidate.node_name
                        && other.neuron_idx == candidate.neuron_idx
                });
                if !shared && distinguishing_seen.insert(candidate_index) {
                    distinguishing_candidates.push(candidate_index);
                }
            }
            for &candidate_index in &distinguishing_candidates {
                let sides = if candidate_index < prep.sides.len() {
                    prep.sides.get(candidate_index)?
                } else {
                    overlay_sides.get(candidate_index - prep.sides.len())?
                };
                if sides.iter().any(|side| matches!(side, SideSlot::Failed)) {
                    return None;
                }
            }
            if !budget.available_now() {
                return None;
            }
            if overlay_candidates.len() > ATTRIBUTION_MAX_APPENDED_CANDIDATES
                || overlay_children.len() > ATTRIBUTION_MAX_PRIVATE_CHILD_SHELLS
                || distinguishing_candidates.len() > ATTRIBUTION_MAX_DISTINGUISHING_CANDIDATES
            {
                return None;
            }
            Some((
                overlay_candidates,
                overlay_sides,
                overlay_children,
                historical_candidates,
                attribution_candidates,
                distinguishing_candidates,
            ))
        })();

        let Some((
            overlay_candidates,
            overlay_sides,
            overlay_children,
            historical_candidates,
            attribution_candidates,
            distinguishing_candidates,
        )) = staged
        else {
            install_incomplete_attribution_diag(prep, plan);
            return false;
        };
        if !budget.available_now() {
            install_incomplete_attribution_diag(prep, plan);
            return false;
        }

        prep.candidates.extend(overlay_candidates);
        prep.sides.extend(overlay_sides);
        for child in overlay_children {
            sim_owner.push((prep_index, prep.straggler));
            sims.push(Some(child));
        }
        prep.attribution_diag = Some(AttributionDiagPrep {
            coverage: plan.coverage,
            overlay_complete: true,
            historical_candidates,
            attribution_candidates,
            distinguishing_candidates,
        });
        true
    }

    /// Re-simulate every arm-distinguishing identity through one uniform
    /// private verifier. Shared identities are intentionally absent and may
    /// reuse the one incumbent result seen by both portfolios.
    #[allow(clippy::too_many_arguments)]
    fn simulate_attribution_diag_overlay(
        &self,
        graph: &GraphNetwork,
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        engine: &dyn GemmEngine,
        prep_index: usize,
        preps: &[DomainPrep],
        sims: &[Option<MultiObjectiveGraphBabDomain>],
        budget: DepthTwoLookaheadBudget,
    ) -> HashMap<usize, f32> {
        let Some(prep) = preps.get(prep_index) else {
            return HashMap::new();
        };
        let Some(diag) = prep.attribution_diag.as_ref() else {
            return HashMap::new();
        };
        if !diag.overlay_complete
            || diag.historical_candidates.len() > ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES
            || diag.attribution_candidates.len() > ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES
            || diag.distinguishing_candidates.len() > ATTRIBUTION_MAX_DISTINGUISHING_CANDIDATES
            || !budget.available_now()
        {
            return HashMap::new();
        }
        let mut members = Vec::with_capacity(ATTRIBUTION_MAX_PRIVATE_MEMBERS);
        let mut seen = std::collections::HashSet::with_capacity(ATTRIBUTION_MAX_PRIVATE_MEMBERS);
        for &candidate_index in &diag.distinguishing_candidates {
            let Some(sides) = prep.sides.get(candidate_index) else {
                return HashMap::new();
            };
            for side in sides {
                match side {
                    SideSlot::Sim(sim_index) => {
                        if *sim_index >= sims.len()
                            || sims[*sim_index].is_none()
                            || !seen.insert(*sim_index)
                            || members.len() == ATTRIBUTION_MAX_PRIVATE_MEMBERS
                        {
                            return HashMap::new();
                        }
                        members.push(*sim_index);
                    }
                    SideSlot::Infeasible => {}
                    SideSlot::Failed => return HashMap::new(),
                }
            }
        }
        if members.is_empty() {
            return HashMap::new();
        }
        let Some(objective) = objectives.get(prep.straggler) else {
            return HashMap::new();
        };
        let Some(spec_matrix) = build_spec_matrix(&[objective.clone()]) else {
            return HashMap::new();
        };
        // Sparse by private member, never by the (potentially huge) wave-wide
        // simulation vector. This fixes the diagnostic's retained memory at
        // `ATTRIBUTION_MAX_PRIVATE_MEMBERS` scalar entries.
        let mut values = HashMap::with_capacity(members.len());
        let mut shadow_config = self.config.clone();
        shadow_config.timeout = ADAPTIVE_DEPTH_SHADOW_BUDGET;
        let mut shadow_verifier = self.with_config_from(shadow_config);
        shadow_verifier.config.alpha_config.deadline = Some(budget.private_deadline);
        let clip = clip_interm_resnet_enabled();
        for chunk in members.chunks(attribution_private_chunk(kfsb_sim_chunk())) {
            if !budget.available_now() {
                return values;
            }
            let build_shim = |i: usize| -> GraphBabDomain {
                let child = sims[i].as_ref().expect("diagnostic sim preflighted");
                let mut shim = graph_bab_domain_shim(child);
                if clip {
                    if let Some(clipped) = self.clip_child_node_bounds(graph, child, engine) {
                        shim.node_bounds = clipped;
                        shim.delta_pre_nodes = crate::beta_crown::domain::delta_pre_nodes_unknown();
                    }
                }
                shim
            };
            let shims: Vec<GraphBabDomain> =
                if clip && super::batched_dense_specs::clip_interm_par_enabled() {
                    chunk
                        .par_iter()
                        .map(|&i| {
                            let _guard = crate::faer_parallelism::RayonTaskGuard::new();
                            build_shim(i)
                        })
                        .collect()
                } else {
                    chunk.iter().map(|&i| build_shim(i)).collect()
                };
            if !budget.available_now() {
                return values;
            }
            let shim_refs: Vec<&GraphBabDomain> = shims.iter().collect();
            let Ok(batched) = BatchedDomains::from_graph_domains_with_options(
                &shim_refs,
                relu_nodes,
                BatchedDomainOptions {
                    enable_interm_transfer: self.config.enable_interm_transfer,
                },
            ) else {
                return values;
            };
            if !budget.available_now() {
                return values;
            }
            let _complete_clip_suppression = shadow_verifier
                .complete_clip_deadline_overrides
                .suppress_complete_clip_scoped();
            let propagated = shadow_verifier.propagate_crown_with_batched_domains_full_specs(
                graph,
                &shim_refs,
                &batched,
                &spec_matrix,
                engine,
            );
            if !budget.available_now() {
                values.clear();
                return values;
            }
            let Ok(results) = propagated else {
                return values;
            };
            if results.len() != chunk.len() {
                return values;
            }
            for (&sim_index, result) in chunk.iter().zip(results) {
                let bounds = spec_bounds_to_vec(&result.output_bounds);
                let Some(&(lower, upper)) = bounds.first() else {
                    return values;
                };
                values.insert(
                    sim_index,
                    self.config.child_bound_value(Some((lower, upper))),
                );
            }
        }
        values
    }

    /// Steps 1+2+3a for one wave domain: straggler row, objective-directed
    /// pre-scores, top-k ∪ backup ∪ layer-quota filter, and both children per
    /// candidate. Returns `None` (advisory fallback) when the domain has no
    /// unverified objective, the score backward fails, or no candidate yields
    /// any feasible-or-infeasible side.
    // Justification: mirrors the caller's context; splitting further would
    // just add a one-use struct.
    #[allow(clippy::too_many_arguments)]
    fn kfsb_multi_prepare_domain(
        &self,
        graph: &GraphNetwork,
        slot: usize,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        k: usize,
        layer_quota: bool,
        cached_la_enabled: bool,
        deadline: Option<std::time::Instant>,
        attribution_diag_target: bool,
        depth_two_request: Option<(usize, DepthTwoLookaheadBudget)>,
    ) -> Option<(
        DomainPrep,
        Vec<(usize, MultiObjectiveGraphBabDomain)>,
        Option<DepthTwoLookaheadOverlayPlan>,
        Option<AttributionDiagOverlayPlan>,
    )> {
        if unstable.is_empty() || deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            return None;
        }
        // The wave lane is already gated/paid for; route its objective seed
        // through the same aggregation-, threshold-, and direction-aware row
        // policy as standard graph branching. Invalid advisory metadata fails
        // open by declining this optional scorer.
        let straggler = domain.critical_objective_index(thresholds).ok().flatten()?;
        let seed_row = objectives.get(straggler)?.as_slice();

        // Objective-directed BaBSR pre-scores, stopped early at the unstable set.
        // #string-key-churn: same shape as the sibling site -- one entry per
        // unstable NEURON, ~20 distinct names. Dedupe on `&str`, then clone only
        // the survivors.
        let unstable_nodes: std::collections::HashSet<String> = {
            let mut distinct: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for (n, _) in unstable.iter() {
                distinct.insert(n.as_str());
            }
            distinct.into_iter().map(str::to_string).collect()
        };
        let reduce_op = kfsb_multi_reduce_op(self.config.kfsb_reduce_op);
        let (score_parts, cached_score_keys) = if cached_la_enabled {
            let mut cached_parts = domain
                .cached_la_for_objective(straggler)
                .map(|cached_la| {
                    self.compute_graph_babsr_scores_from_cached_la(
                        graph,
                        &domain.node_bounds,
                        &domain.input_bounds,
                        cached_la,
                        reduce_op,
                        &unstable_nodes,
                    )
                })
                .unwrap_or_default();
            if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                return None;
            }
            let cached_score_keys: std::collections::HashSet<(&str, usize)> = unstable
                .iter()
                .filter(|candidate| cached_parts.contains_key(*candidate))
                .map(|(node_name, neuron_idx)| (node_name.as_str(), *neuron_idx))
                .collect();
            let needs_proxy = unstable
                .iter()
                .any(|candidate| !cached_parts.contains_key(candidate));
            if needs_proxy {
                if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                    return None;
                }
                let proxy_parts = match deadline {
                    Some(deadline) => self.compute_graph_babsr_scores_from_bounds_until(
                        graph,
                        &domain.node_bounds,
                        &domain.input_bounds,
                        reduce_op,
                        Some(seed_row),
                        Some(&unstable_nodes),
                        deadline,
                    ),
                    None => self.compute_graph_babsr_scores_from_bounds(
                        graph,
                        &domain.node_bounds,
                        &domain.input_bounds,
                        reduce_op,
                        Some(seed_row),
                        Some(&unstable_nodes),
                    ),
                }
                .ok()?;
                if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                    return None;
                }
                for candidate in unstable {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        cached_parts.entry(candidate.clone())
                    {
                        if let Some(parts) = proxy_parts.get(candidate) {
                            entry.insert(*parts);
                        }
                    }
                }
            }
            (cached_parts, cached_score_keys)
        } else {
            // Exact off-path anchor: this is the pre-gate expression and its
            // returned map is consumed directly, without an O(unstable) copy.
            let score_parts = match deadline {
                Some(deadline) => self.compute_graph_babsr_scores_from_bounds_until(
                    graph,
                    &domain.node_bounds,
                    &domain.input_bounds,
                    reduce_op,
                    Some(seed_row),
                    Some(&unstable_nodes),
                    deadline,
                ),
                None => self.compute_graph_babsr_scores_from_bounds(
                    graph,
                    &domain.node_bounds,
                    &domain.input_bounds,
                    reduce_op,
                    Some(seed_row),
                    Some(&unstable_nodes),
                ),
            }
            .ok()?;
            if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                return None;
            }
            (score_parts, std::collections::HashSet::new())
        };
        let (mut scored, depth_two_score_map_complete) = if depth_two_request.is_some() {
            materialize_kfsb_candidates_with_completeness(unstable, |candidate| {
                score_parts
                    .get(candidate)
                    .map(|parts| (parts.main_score, parts.backup_score))
            })
        } else {
            // Exact legacy path: preserve the historical zero-fill expression
            // and avoid even the typed completeness branch per candidate.
            (
                unstable
                    .iter()
                    .map(|(node_name, neuron_idx)| {
                        let parts = score_parts
                            .get(&(node_name.clone(), *neuron_idx))
                            .copied()
                            .unwrap_or_default();
                        GraphKfsbCandidate {
                            node_name: node_name.clone(),
                            neuron_idx: *neuron_idx,
                            main_score: parts.main_score,
                            backup_score: parts.backup_score,
                        }
                    })
                    .collect(),
                false,
            )
        };
        scored.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });

        // #attr-branch (DARK, NY_ATTR_BRANCH=1): re-rank by the root gap
        // attribution before the top-k cut.
        //
        // Measured motivation: at the cifar100 root 8 of 1366 unstable neurons
        // carry HALF the binding margin row's gap and ~50 carry ninety percent
        // (theory doc Sec 6c), while `main_score` above is the triangle
        // intercept `-l*u/(u-l)` times `coeff.min(0.0)` -- a worst-case-over-
        // the-box proxy that discards the `coeff >= 0` branch entirely.
        //
        // This only REORDERS; it never removes a candidate. The pool therefore
        // cannot be emptied, so the `NoUnstable` -> run-level Unknown hazard is
        // structurally unreachable from here. The row snapshot must cover
        // EVERY unstable candidate before the optional sort runs. One missing
        // node/index makes the whole domain retain its historical ordering --
        // an uncovered neuron is "no opinion", not "known inert".
        //
        // The prior is looked up for THIS domain's straggler row. kFSB already
        // collapses the 99 objectives to one worst-unverified row per domain,
        // and the attribution is per row -- blending rows was measured to wash
        // out the concentration (6 rows put 1181 of 1366 neurons above zero,
        // against d50 = 8 for a single row).
        let attribution_branching_enabled =
            crate::network::gap_attribution::attribution_branching_enabled();
        let attribution_requested = attribution_branching_enabled || attribution_diag_target;
        let root_prior_fresh = root_attribution_prior_is_fresh(domain.depth());
        let (attribution_order, attribution_diag_coverage) =
            if attribution_requested && root_prior_fresh {
                match crate::network::gap_attribution::attribution_prior_for_row(straggler) {
                    None => (
                        None,
                        attribution_diag_target.then_some(AttributionDiagCoverage::RowMissing),
                    ),
                    Some(prior) => {
                        if scored.len() > ATTRIBUTION_MAX_RANKED_CANDIDATES {
                            (
                                None,
                                attribution_diag_target
                                    .then_some(AttributionDiagCoverage::ResourceCapped),
                            )
                        } else {
                            let order =
                                rank_kfsb_candidate_indices_by_attribution(&scored, |candidate| {
                                    prior.score(&candidate.node_name, candidate.neuron_idx)
                                });
                            let coverage = attribution_diag_target.then_some(if order.is_some() {
                                AttributionDiagCoverage::Complete
                            } else {
                                AttributionDiagCoverage::CandidateIncomplete
                            });
                            (order, coverage)
                        }
                    }
                }
            } else if attribution_requested {
                (
                    None,
                    attribution_diag_target.then_some(AttributionDiagCoverage::RootPriorStale),
                )
            } else {
                (None, None)
            };

        // This is the only attribution-directed portfolio allowed to become
        // the advisory incumbent. Oversized k/quota policy declines the root
        // advice atomically and retains historical kFSB.
        let bounded_attribution_candidates = attribution_order.as_ref().and_then(|order| {
            select_bounded_attribution_kfsb_candidates(&scored, order, k, layer_quota)
        });
        let attribution_portfolio_admitted = bounded_attribution_candidates.is_some();

        // Construct only the historical portfolio in this Rayon phase. The
        // target's exact-total identities are retained as a plan and any
        // missing paper roots are appended after the global legacy-prep
        // barrier, never concurrently with another domain's legacy work.
        let mut attribution_diag_overlay_plan = None;
        let candidates = if attribution_diag_target {
            let mut historical_candidates = select_graph_kfsb_eval_candidates(&scored, k, true);
            if layer_quota {
                append_layer_quota_candidates(&scored, &mut historical_candidates);
            }

            // Preserve the currently armed selector as the commit-authorized
            // prefix. In the intended DIAG-only arm this is the historical
            // main+backup portfolio; if the scored ATTR arm is also set, its
            // bounded attribution portfolio remains incumbent instead.
            let incumbent_is_attribution =
                attribution_branching_enabled && bounded_attribution_candidates.is_some();
            let incumbent = if incumbent_is_attribution {
                bounded_attribution_candidates
                    .as_ref()
                    .expect("checked attribution incumbent")
                    .clone()
            } else {
                historical_candidates.clone()
            };
            let mut coverage =
                attribution_diag_coverage.unwrap_or(AttributionDiagCoverage::RowMissing);
            let (diagnostic_historical, diagnostic_attribution) =
                if coverage == AttributionDiagCoverage::Complete {
                    if historical_candidates.len() <= ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES {
                        if let Some(attribution) = bounded_attribution_candidates.as_ref() {
                            (historical_candidates, attribution.clone())
                        } else {
                            coverage = AttributionDiagCoverage::ResourceCapped;
                            (Vec::new(), Vec::new())
                        }
                    } else {
                        coverage = AttributionDiagCoverage::ResourceCapped;
                        (Vec::new(), Vec::new())
                    }
                } else if historical_candidates.len() <= ATTRIBUTION_MAX_PORTFOLIO_CANDIDATES {
                    (historical_candidates, Vec::new())
                } else {
                    // Preserve the primary refusal (for example RowMissing).
                    // Oversized detail vectors are optional telemetry and are
                    // simply omitted when no complete comparison can run.
                    (Vec::new(), Vec::new())
                };
            let mut diagnostic_plan = AttributionDiagOverlayPlan {
                coverage,
                historical_candidates: diagnostic_historical,
                attribution_candidates: diagnostic_attribution,
            };
            if diagnostic_plan.coverage == AttributionDiagCoverage::Complete
                && !attribution_diag_plan_within_caps(&diagnostic_plan)
            {
                diagnostic_plan.coverage = AttributionDiagCoverage::ResourceCapped;
                diagnostic_plan.historical_candidates.clear();
                diagnostic_plan.attribution_candidates.clear();
            }
            attribution_diag_overlay_plan = Some(diagnostic_plan);
            incumbent
        } else {
            // Exact pre-diagnostic path: one selected portfolio and no diagnostic
            // identity, membership map, or candidate clone.
            let mut selected = bounded_attribution_candidates
                .unwrap_or_else(|| select_graph_kfsb_eval_candidates(&scored, k, true));
            if layer_quota {
                // The bounded attribution builder already included its quota.
                // Only the historical fallback needs the legacy unbounded
                // layer-quota helper to preserve gate-off behavior exactly.
                if !attribution_portfolio_admitted {
                    append_layer_quota_candidates(&scored, &mut selected);
                }
            }
            selected
        };
        let depth_two_overlay_plan = depth_two_request.and_then(|(total, budget)| {
            budget
                .available_now()
                .then(|| {
                    select_depth_two_root_portfolio(&scored, depth_two_score_map_complete, total)
                })
                .flatten()
                .map(|selected| DepthTwoLookaheadOverlayPlan { selected })
        });
        if candidates.is_empty() {
            return None;
        }

        // Both children per candidate. `Ok(None)` / infeasible-domain errors
        // mean the half-space is EMPTY (side resolved); other errors mark the
        // side failed (-inf) — a candidate with any usable side stays ranked.
        let mut sides: Vec<[SideSlot; 2]> = Vec::with_capacity(candidates.len());
        let mut children: Vec<(usize, MultiObjectiveGraphBabDomain)> = Vec::new();
        let mut kept: Vec<GraphKfsbCandidate> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                return None;
            }
            let mut pair: Vec<SideSlot> = Vec::with_capacity(2);
            let mut usable = false;
            for is_active in [true, false] {
                if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                    return None;
                }
                let constraint = GraphNeuronConstraint {
                    node_name: candidate.node_name.clone(),
                    neuron_idx: candidate.neuron_idx,
                    is_active,
                    score: candidate.main_score,
                };
                match domain.with_constraint(graph, constraint, false, thresholds) {
                    Ok(Some(child)) => {
                        let local = children.len();
                        children.push((local, child));
                        pair.push(SideSlot::Sim(local));
                        usable = true;
                    }
                    Ok(None) => {
                        pair.push(SideSlot::Infeasible);
                        usable = true;
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        pair.push(SideSlot::Infeasible);
                        usable = true;
                    }
                    Err(_) => {
                        // Reusing Infeasible here would over-reward a broken
                        // side; Failed counts as -inf at pick time so the
                        // candidate survives only on its other side's merit.
                        pair.push(SideSlot::Failed);
                    }
                }
            }
            if usable {
                let pair: [SideSlot; 2] = match pair.try_into() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                sides.push(pair);
                kept.push(candidate);
            }
        }
        if kept.is_empty() {
            return None;
        }
        let legacy_candidates_len = kept.len();
        let cached_score_candidates = kept
            .iter()
            .take(legacy_candidates_len)
            .filter(|candidate| {
                cached_score_keys.contains(&(candidate.node_name.as_str(), candidate.neuron_idx))
            })
            .count();
        Some((
            DomainPrep {
                slot,
                straggler,
                cached_score_candidates,
                legacy_candidates_len,
                depth_two_lookahead_candidates: None,
                attribution_diag: None,
                candidates: kept,
                sides,
            },
            children,
            depth_two_overlay_plan,
            attribution_diag_overlay_plan,
        ))
    }
}

/// Stratified layer quota: admit each unstable ReLU layer's top-1 main-score
/// candidate (dedup against the already-selected set). `scored` is sorted by
/// main score descending, so the first sighting of a layer is its best.
pub(super) fn append_layer_quota_candidates(
    scored: &[GraphKfsbCandidate],
    candidates: &mut Vec<GraphKfsbCandidate>,
) {
    let mut seen: std::collections::HashSet<(String, usize)> = candidates
        .iter()
        .map(|c| (c.node_name.clone(), c.neuron_idx))
        .collect();
    let mut layers_done: std::collections::HashSet<String> =
        candidates.iter().map(|c| c.node_name.clone()).collect();
    for candidate in scored {
        if !layers_done.insert(candidate.node_name.clone()) {
            continue;
        }
        if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
            candidates.push(candidate.clone());
        }
    }
}
