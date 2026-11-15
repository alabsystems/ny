// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-BaB lifecycle state and result assembly.
//!
//! `GraphBabLifecycle` collects the running counters and unresolved flags
//! shared across all three graph-BaB verifiers (ReLU split, multi-objective,
//! GPU BaB). Builder methods replace the 20+ inline `BetaCrownResult` struct
//! literals scattered across those verifiers.

use std::time::{Duration, Instant};

use ny_tensor::BoundedTensor;

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

/// Shared lifecycle state for graph-BaB verification loops.
///
/// Accumulates counters, unresolved flags, and the start timestamp needed
/// to construct `BetaCrownResult` at any termination point. All three
/// graph-BaB modes (single-objective ReLU split, multi-objective, GPU BaB)
/// share this type instead of maintaining separate state structs.
///
/// # Unresolved reason flags
///
/// Multiple distinct conditions can leave domains unresolved. Each gets a
/// separate flag so the terminal `Unknown` reason string preserves the actual
/// cause instead of collapsing into a generic message.
#[derive(Clone)]
pub(crate) struct GraphBabLifecycle {
    pub(crate) start_time: Instant,
    pub(crate) domains_explored: usize,
    pub(crate) domains_verified: usize,
    /// Parents whose evaluated children ALL raised `InfeasibleDomain`, so the
    /// batched multi-objective fold recorded `Children([])`: nothing queued.
    /// Previously invisible — a frontier draining through infeasible pairs
    /// read identically to one draining through proofs. Counted by the queue
    /// fold; printed beside `domains_verified` in the lane summaries. Never
    /// read by a verdict itself; since #spine-legacy-lanes the same fold ALSO
    /// raises `unresolved_due_to_propagation_failure` for a pair (the M2c
    /// certified-infeasibility rule: an emptiness close with no certificate
    /// may not publish `Verified`).
    ///
    /// #vit-bank-gate: the batched lane's two other infeasibility dispositions
    /// are counted here too — a NoUnstable domain whose own propagation raised
    /// `InfeasibleDomain` (`InfeasibleEmpty`; it ALSO counts in
    /// `domains_verified`, exactly as the `AlreadyVerified` it replaced) and a
    /// split parent with zero evaluable children (`InfeasibleCoverRefused`; it
    /// ALSO raises `unresolved_due_to_propagation_failure`, exactly as the
    /// `PropagationFailure` it replaced). The `[mo-fold] infeasible-{pair,
    /// empty,cover}` lines say which one fired. A vit bank requires
    /// `[bab-stop] ... infeasible=0`.
    pub(crate) domains_infeasible: usize,
    pub(crate) max_depth_reached: usize,
    pub(crate) cuts_generated: usize,
    pub(crate) unresolved_due_to_depth: bool,
    pub(crate) unresolved_due_to_unsplittable: bool,
    pub(crate) unresolved_due_to_no_branch: bool,
    pub(crate) unresolved_due_to_no_unstable_neurons: bool,
    pub(crate) unresolved_due_to_genbab_no_split: bool,
    pub(crate) unresolved_due_to_propagation_failure: bool,
    pub(crate) unresolved_due_to_violated_drop: bool,
    pub(crate) unresolved_due_to_eviction: bool,
    /// #cap-extend (DEFER-dont-DROP): the effective domain cap once the
    /// disjunctive multi-clause loop has granted its ONE near-closure
    /// extension. `None` — the state every mode but that one ever sees, and
    /// the shipped default there — means the configured `max_domains` governs
    /// byte-for-byte as before.
    ///
    /// It lives on the lifecycle rather than in the loop because the cap is
    /// enforced at FOUR sites: the loop head plus the three per-domain checks
    /// inside `disjunctive_multi_clause::{process_batch, child_batch}`. A
    /// grant that updates only the loop head is silently inert — the next
    /// batch stops at the un-extended cap and reports it (measured: the
    /// `[cap-extend] granted: cap 24 -> 192` marker fired and the run still
    /// returned `Domain limit 24`).
    pub(crate) cap_extend_effective_max_domains: Option<usize>,
    /// #cap-extend: the closest-to-closure queue-head gap observed so far,
    /// `lower - threshold` of the most promising unresolved domain. Updated
    /// by the disjunctive loop head after each batch and read by the cap
    /// trip, which can fire from the loop head OR from inside a batch. It
    /// lives here for the same reason the cap does: with `batch_size > 1` the
    /// trip is USUALLY mid-batch, where the queue-head history is not
    /// otherwise in scope. Starts at `NEG_INFINITY`, which denies.
    pub(crate) cap_extend_closest_gap_seen: f32,
    /// #vit-margin-alpha-complex (P8 bank gate): the certified objective
    /// LOWER bounds a ROOT-terminal `Verified` rested on, recorded by the root
    /// site that published it (`maybe_finish_at_root`, the dd-zonotope root).
    /// The `Finished` result carries only the output enclosure, so without
    /// this the verdict's actual authority never reached the containment
    /// fence (`objective_lowers=UNFENCED`). Read only by the default-off fence
    /// (`bab.mo_containment_fence`); never read by a verdict. `None` until a
    /// root site records it.
    pub(crate) root_verified_objective_lowers: Option<Vec<f32>>,
    /// #vit-v3-bankable: domains the Graph-MIP LEAF oracle closed on every
    /// row (`GraphMipLeafVerdict::VerifiedAllRows` in the multi-objective
    /// queue fold). Such a close carries no bound the per-row cover frame can
    /// record, so an armed `bab.mo_cover_certify` frame gives up on it
    /// (UNBUILT). Counted, never read by a verdict.
    pub(crate) domains_verified_by_leaf_oracle: usize,
}

impl GraphBabLifecycle {
    /// Derive a deadline without allowing platform clock overflow.
    ///
    /// An unrepresentable budget is treated as already expired. This is
    /// fail-closed and prevents a malformed low-level timeout from panicking
    /// at graph-BaB deadline seams.
    pub(crate) fn fail_closed_deadline(start_time: Instant, budget: Duration) -> Instant {
        start_time.checked_add(budget).unwrap_or(start_time)
    }

    /// Derive a deadline from this lifecycle's start time.
    pub(crate) fn deadline(&self, budget: Duration) -> Instant {
        Self::fail_closed_deadline(self.start_time, budget)
    }

    pub(crate) fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            domains_explored: 0,
            domains_verified: 0,
            domains_infeasible: 0,
            max_depth_reached: 0,
            cuts_generated: 0,
            unresolved_due_to_depth: false,
            unresolved_due_to_unsplittable: false,
            unresolved_due_to_no_branch: false,
            unresolved_due_to_no_unstable_neurons: false,
            unresolved_due_to_genbab_no_split: false,
            unresolved_due_to_propagation_failure: false,
            unresolved_due_to_violated_drop: false,
            unresolved_due_to_eviction: false,
            cap_extend_effective_max_domains: None,
            cap_extend_closest_gap_seen: f32::NEG_INFINITY,
            root_verified_objective_lowers: None,
            domains_verified_by_leaf_oracle: 0,
        }
    }

    /// The domain cap in force: a granted #cap-extend extension when one
    /// exists, else the configured `max_domains`. Every enforcement site must
    /// call this rather than reading `config.max_domains` directly.
    pub(crate) fn domain_cap(&self, config_max_domains: usize) -> usize {
        self.cap_extend_effective_max_domains
            .unwrap_or(config_max_domains)
    }

    /// Build a `BetaCrownResult` with the given status and current counters.
    /// #bab-stop: a compact tag for the outcome. Deliberately NOT `Debug` --
    /// `Unknown`'s reason is a free-form sentence that can carry commas and
    /// newlines, which would make this line unparseable.
    fn status_tag(status: &BabVerificationStatus) -> &'static str {
        match status {
            BabVerificationStatus::Verified => "verified",
            BabVerificationStatus::Violated { .. } => "violated",
            BabVerificationStatus::PotentialViolation { .. } => "potential",
            BabVerificationStatus::Unknown { .. } => "unknown",
            BabVerificationStatus::Timeout => "timeout",
        }
    }

    /// The unresolved flags that actually fired, `+`-joined, or `none`.
    ///
    /// These eight booleans are the ONLY record of why a BaB loop stopped
    /// short, and until #bab-stop none of them left the process.
    pub(crate) fn unresolved_tags(&self) -> String {
        let mut tags: Vec<&'static str> = Vec::new();
        if self.unresolved_due_to_depth {
            tags.push("depth");
        }
        if self.unresolved_due_to_unsplittable {
            tags.push("unsplittable");
        }
        if self.unresolved_due_to_no_branch {
            tags.push("no_branch");
        }
        if self.unresolved_due_to_no_unstable_neurons {
            tags.push("no_unstable");
        }
        if self.unresolved_due_to_genbab_no_split {
            tags.push("genbab_no_split");
        }
        if self.unresolved_due_to_propagation_failure {
            tags.push("propagation_failure");
        }
        if self.unresolved_due_to_violated_drop {
            tags.push("violated_drop");
        }
        if self.unresolved_due_to_eviction {
            tags.push("eviction");
        }
        if tags.is_empty() {
            "none".to_string()
        } else {
            tags.join("+")
        }
    }

    /// #bab-stop: ONE line per BaB exit, on stderr, always on.
    ///
    /// Why this exists. A graph BaB that returns having explored ZERO domains
    /// and a graph BaB that exhausted a 562 s slice are, in today's logs, the
    /// same three numbers: `[depth-drop] dropped=0 deepest=0 cap=10000`. That
    /// ambiguity cost a whole diagnosis on ml4acopf `14_ieee_prop1`, where the
    /// shared-bootstrap arm's clause 39 returned in MICROSECONDS while holding
    /// a 562 s allowance and the row published `unknown` with 98.0 % of its
    /// 600 s budget unspent -- and nothing in the log could say whether the
    /// loop timed out, hit the domain cap, found no unstable neuron to branch
    /// on, or never ran. The lifecycle already knew: it tracks four counters
    /// and eight distinct unresolved reasons, and printed none of them.
    ///
    /// `eprintln!`, not `tracing`, on purpose: `ny benchmarks run` forces
    /// `RUST_LOG=error` and discards the child's stdout, so a `tracing` line
    /// here would be invisible in exactly the sweeps that need it. This
    /// follows `[depth-drop]` and `[warmup-fallback]`, which are always-on for
    /// the same reason. No env read and no typed key, so it adds no ratchet
    /// debt and cannot be left dark by a config.
    ///
    /// It is pure telemetry: no verdict, bound or counter is read back from
    /// it, so it cannot move a result.
    fn emit_stop_line(&self, status: &BabVerificationStatus) {
        // #phase-cost: this lifecycle's work denominators for the run's cost
        // census (write-only; nothing reads them back).
        {
            use ny_core::phase_cost::{count, CostCounter};
            count(CostCounter::DomainsExplored, self.domains_explored as u64);
            count(CostCounter::DomainsVerified, self.domains_verified as u64);
            count(
                CostCounter::DomainsInfeasible,
                self.domains_infeasible as u64,
            );
            count(CostCounter::BabRuns, 1);
        }
        // #router-r1: the graph BaB's final progress sample (write-only; one
        // relaxed atomic load while the router is off).
        if ny_core::router::bus::enabled() {
            ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::frontier(
                ny_core::router::decl::ids::RELU_SPLIT_GRAPH,
                self.domains_explored as u64,
                self.domains_explored as u64,
                self.domains_verified as u64,
                0,
            ));
        }
        eprintln!(
            "[bab-stop] status={} explored={} verified={} infeasible={} deepest={} \
cuts={} elapsed_s={:.3} unresolved={}",
            Self::status_tag(status),
            self.domains_explored,
            self.domains_verified,
            self.domains_infeasible,
            self.max_depth_reached,
            self.cuts_generated,
            self.start_time.elapsed().as_secs_f64(),
            self.unresolved_tags(),
        );
        // #genbab-selector-convention: process-cumulative GenBaB selector
        // counters, printed only once a GenBaB selection has run in this
        // process (every other category prints nothing extra). Telemetry only.
        if let Some(line) = crate::beta_crown::nonlinear_branching::genbab_select_summary_line() {
            eprintln!("{line}");
        }
    }

    pub(crate) fn build_result(&self, status: BabVerificationStatus) -> BetaCrownResult {
        self.emit_stop_line(&status);
        BetaCrownResult {
            result: status,
            domains_explored: self.domains_explored,
            time_elapsed: self.start_time.elapsed(),
            max_depth_reached: self.max_depth_reached,
            output_bounds: None,
            cuts_generated: self.cuts_generated,
            domains_verified: self.domains_verified,
        }
    }

    /// Build a `BetaCrownResult` with output bounds (used for root early exits).
    pub(crate) fn build_result_with_bounds(
        &self,
        status: BabVerificationStatus,
        output_bounds: BoundedTensor,
    ) -> BetaCrownResult {
        self.emit_stop_line(&status);
        BetaCrownResult {
            result: status,
            domains_explored: self.domains_explored,
            time_elapsed: self.start_time.elapsed(),
            max_depth_reached: self.max_depth_reached,
            output_bounds: Some(output_bounds),
            cuts_generated: self.cuts_generated,
            domains_verified: self.domains_verified,
        }
    }

    /// Build a `Timeout` result.
    pub(crate) fn timeout_result(&self) -> BetaCrownResult {
        self.build_result(BabVerificationStatus::Timeout)
    }

    /// Build an `Unknown` result for hitting the domain exploration limit.
    pub(crate) fn domain_limit_result(&self, max_domains: usize) -> BetaCrownResult {
        self.build_result(BabVerificationStatus::Unknown {
            reason: format!("Domain limit {} reached", max_domains),
        })
    }

    /// Check timeout and domain-limit termination conditions.
    /// Returns `Some(result)` if the loop should terminate.
    pub(crate) fn check_termination(
        &self,
        timeout: Duration,
        max_domains: usize,
    ) -> Option<BetaCrownResult> {
        if self.start_time.elapsed() > timeout {
            return Some(self.timeout_result());
        }
        if self.domains_explored >= max_domains {
            return Some(self.domain_limit_result(max_domains));
        }
        None
    }

    /// Whether any unresolved flags are set.
    pub(crate) fn has_unresolved(&self) -> bool {
        self.unresolved_due_to_depth
            || self.unresolved_due_to_unsplittable
            || self.unresolved_due_to_no_branch
            || self.unresolved_due_to_no_unstable_neurons
            || self.unresolved_due_to_genbab_no_split
            || self.unresolved_due_to_propagation_failure
            || self.unresolved_due_to_violated_drop
            || self.unresolved_due_to_eviction
    }

    /// Build a human-readable reason string from the active unresolved flags.
    pub(crate) fn unresolved_reason(&self) -> String {
        let mut parts = Vec::new();
        if self.unresolved_due_to_depth {
            parts.push(format!("Max depth {} reached", self.max_depth_reached));
        }
        if self.unresolved_due_to_unsplittable {
            parts.push(
                "Unsplittable domain (invalid split dimension, non-finite, or zero-width bounds)"
                    .to_string(),
            );
        }
        if self.unresolved_due_to_no_branch {
            parts.push("No unstable ReLU/Sign neurons left in some domains".to_string());
        }
        if self.unresolved_due_to_no_unstable_neurons {
            parts.push("No unstable ReLU/Sign neurons left in some domains (GPU BaB)".to_string());
        }
        if self.unresolved_due_to_genbab_no_split {
            parts.push("GenBaB found no splittable nodes in some domains".to_string());
        }
        if self.unresolved_due_to_propagation_failure {
            parts.push("Child propagation failed for some domains".to_string());
        }
        if self.unresolved_due_to_violated_drop {
            parts.push("Some domains conclusively violated the property".to_string());
        }
        if self.unresolved_due_to_eviction {
            parts.push(
                "Queue cap (max_queue_size / max_queue_bytes) evicted unverified domains before they were explored"
                    .to_string(),
            );
        }
        parts.join("; ")
    }

    /// Build the final result after the BaB loop exits normally.
    ///
    /// If any domains were unresolved, returns Unknown with cause-specific
    /// reasons. Otherwise returns Verified (queue exhaustion = all verified).
    pub(crate) fn build_final_result(&self) -> BetaCrownResult {
        if self.has_unresolved() {
            self.build_result(BabVerificationStatus::Unknown {
                reason: self.unresolved_reason(),
            })
        } else {
            self.build_result(BabVerificationStatus::Verified)
        }
    }
}

#[cfg(test)]
mod tests {

    /// #bab-stop: the eight unresolved flags are the only record of why a BaB
    /// stopped short, and this renders exactly the ones that fired.
    #[test]
    fn unresolved_tags_render_only_the_flags_that_fired() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        assert_eq!(lc.unresolved_tags(), "none");

        lc.unresolved_due_to_no_unstable_neurons = true;
        assert_eq!(lc.unresolved_tags(), "no_unstable");

        lc.unresolved_due_to_depth = true;
        assert_eq!(lc.unresolved_tags(), "depth+no_unstable");

        lc.unresolved_due_to_eviction = true;
        assert_eq!(lc.unresolved_tags(), "depth+no_unstable+eviction");
    }

    /// THE DISCRIMINATION THIS EXISTS FOR. On ml4acopf `14_ieee_prop1` the
    /// shared-bootstrap arm's clause 39 returned in microseconds holding a
    /// 562 s slice, and the control's SAME clause split to the 10000 depth
    /// cap. Both printed the identical `[depth-drop] dropped=0 deepest=0`
    /// three-number line at the point the two diverged, so the logs could not
    /// tell "never ran" from "ran and exhausted". These two lifecycles are
    /// those two cases; the stop line must not render them the same.
    #[test]
    fn a_bab_that_never_ran_is_distinguishable_from_one_that_exhausted() {
        let never_ran = GraphBabLifecycle::new(Instant::now());

        let mut exhausted = GraphBabLifecycle::new(Instant::now());
        exhausted.domains_explored = 39_739;
        exhausted.max_depth_reached = 10_000;
        exhausted.unresolved_due_to_depth = true;

        assert_eq!(never_ran.domains_explored, 0);
        assert_eq!(never_ran.max_depth_reached, 0);
        assert_eq!(never_ran.unresolved_tags(), "none");
        assert_eq!(exhausted.unresolved_tags(), "depth");
        assert_ne!(
            (
                never_ran.domains_explored,
                never_ran.max_depth_reached,
                never_ran.unresolved_tags()
            ),
            (
                exhausted.domains_explored,
                exhausted.max_depth_reached,
                exhausted.unresolved_tags()
            ),
            "the stop line must separate a BaB that never ran from one that exhausted its budget"
        );
    }

    /// Every status maps to a stable, parseable one-word tag -- never `Debug`,
    /// whose `Unknown` reason is a free-form sentence carrying commas.
    #[test]
    fn status_tags_are_single_words_and_total() {
        let cases = [
            (BabVerificationStatus::Verified, "verified"),
            (BabVerificationStatus::Timeout, "timeout"),
            (
                BabVerificationStatus::PotentialViolation { witness: None },
                "potential",
            ),
            (
                BabVerificationStatus::Unknown {
                    reason: "Clause 39: No constraint was provably violated, 1 checked".to_string(),
                },
                "unknown",
            ),
        ];
        for (status, want) in cases {
            let got = GraphBabLifecycle::status_tag(&status);
            assert_eq!(got, want);
            assert!(
                !got.contains(' ') && !got.contains(',') && !got.contains('\n'),
                "tag {got:?} would break the line format"
            );
        }
    }
    use super::*;

    #[test]
    fn test_lifecycle_new_defaults() {
        let lc = GraphBabLifecycle::new(Instant::now());
        assert_eq!(lc.domains_explored, 0);
        assert_eq!(lc.domains_verified, 0);
        assert_eq!(lc.max_depth_reached, 0);
        assert_eq!(lc.cuts_generated, 0);
        assert!(!lc.has_unresolved());
    }

    #[test]
    fn unrepresentable_deadline_fails_closed_at_start() {
        let start = Instant::now();
        assert_eq!(
            GraphBabLifecycle::fail_closed_deadline(start, Duration::from_secs(u64::MAX)),
            start
        );
    }

    #[test]
    fn test_lifecycle_build_result_fields() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.build_result(BabVerificationStatus::Verified);
        assert_eq!(result.result, BabVerificationStatus::Verified);
        assert_eq!(result.domains_explored, 0);
        assert!(result.output_bounds.is_none());
        assert_eq!(result.cuts_generated, 0);
    }

    #[test]
    fn test_lifecycle_timeout_result() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.timeout_result();
        assert_eq!(result.result, BabVerificationStatus::Timeout);
    }

    #[test]
    fn test_lifecycle_domain_limit_result() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.domains_explored = 50;
        let result = lc.domain_limit_result(100);
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(reason.contains("Domain limit 100 reached"));
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
        assert_eq!(result.domains_explored, 50);
    }

    #[test]
    fn test_lifecycle_has_unresolved_flags() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        assert!(!lc.has_unresolved());

        lc.unresolved_due_to_depth = true;
        assert!(lc.has_unresolved());

        lc.unresolved_due_to_depth = false;
        lc.unresolved_due_to_violated_drop = true;
        assert!(lc.has_unresolved());
    }

    #[test]
    fn test_lifecycle_unsplittable_reason() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.unresolved_due_to_unsplittable = true;

        let reason = lc.unresolved_reason();
        assert!(reason.contains("Unsplittable domain"));
        assert!(reason.contains("invalid split dimension"));
    }

    #[test]
    fn test_lifecycle_unresolved_reason_combines_all() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.max_depth_reached = 7;
        lc.unresolved_due_to_depth = true;
        lc.unresolved_due_to_propagation_failure = true;
        lc.unresolved_due_to_violated_drop = true;

        let reason = lc.unresolved_reason();
        assert!(reason.contains("Max depth 7 reached"));
        assert!(reason.contains("Child propagation failed"));
        assert!(reason.contains("conclusively violated"));
    }

    #[test]
    fn test_lifecycle_build_final_result_verified() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.build_final_result();
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    /// Queue-cap eviction discards unverified domains, so a drained queue
    /// must produce Unknown, never Verified.
    #[test]
    fn test_lifecycle_eviction_flag_forces_unknown() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        assert!(!lc.has_unresolved());

        lc.unresolved_due_to_eviction = true;
        assert!(lc.has_unresolved());

        let result = lc.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("evicted unverified domains"),
                    "reason must attribute the Unknown to queue-cap eviction, got: {reason}"
                );
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn test_lifecycle_build_final_result_unknown() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.unresolved_due_to_no_branch = true;
        let result = lc.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(reason.contains("No unstable ReLU/Sign neurons"));
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
    }

    /// The sequential-BaB `no_branch` and GPU-BaB `no_unstable_neurons` flags
    /// must produce distinct reason strings so diagnostics identify the source.
    #[test]
    fn test_lifecycle_no_branch_vs_no_unstable_neurons_reasons_distinct() {
        let mut lc_seq = GraphBabLifecycle::new(Instant::now());
        lc_seq.unresolved_due_to_no_branch = true;
        let reason_seq = lc_seq.unresolved_reason();

        let mut lc_gpu = GraphBabLifecycle::new(Instant::now());
        lc_gpu.unresolved_due_to_no_unstable_neurons = true;
        let reason_gpu = lc_gpu.unresolved_reason();

        assert_eq!(
            reason_seq,
            "No unstable ReLU/Sign neurons left in some domains"
        );
        assert_eq!(
            reason_gpu,
            "No unstable ReLU/Sign neurons left in some domains (GPU BaB)"
        );
    }
}
