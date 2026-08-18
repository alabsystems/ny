// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #batched-bab wide-lane DECLINE TALLY — observability ONLY.
//!
//! The production counter `ny_gpu::wide_resnet_batched_taken_count()` reports how
//! many times the domain-stacked ("wide") GPU CROWN pass PUBLISHED a result. It
//! cannot say why the other candidate batches did not go wide, and a lane that
//! covers 6 of N batches is indistinguishable from one that had only 6 candidates.
//! This module is the missing denominator: a process-global, monotonic tally of
//!
//! * `candidate` — a multi-domain batch reached the propagate-side wide entry;
//! * `attempt`   — that batch reached a batched GPU trait entry;
//! * one counter per [`WideLaneDecline`] reason for every refusal in between.
//!
//! This receipt belongs only to the graph/BaB internal wide lane that writes
//! `candidate`. The independently gated margin-row coefficient batch has its
//! own `gpu_batch_attempts` / `gpu_batch_ok` receipt and MUST NOT write here;
//! mixing a source with no matching candidate would corrupt this denominator.
//!
//! CONTRACT (enforced by review + the pins in this module's test block):
//! - Writing is ONE relaxed [`AtomicU64::fetch_add`]. No allocation, no locking,
//!   no formatting, no environment read on the write path — it is safe to call
//!   from inside the per-batch hot path and from rayon workers.
//! - NOTHING reads these counters to make a decision. No verdict, bound, gate,
//!   or control-flow branch may depend on them; the only readers are the CLI's
//!   dark `[wide-lane]` readout and tests. A tally that is never written and a
//!   tally that is never read are both behaviour-identical to today.
//! - Monotonic and process-global (never reset outside tests) so the CLI can
//!   print it from either completion site, including the timing-out one that a
//!   scored deep row takes.
//!
//! Read [`WideLaneDecline`]'s variant docs as the map of every refusal on the
//! wide path; each variant is written at exactly one site.

use std::sync::atomic::{AtomicU64, Ordering};

/// Why one candidate domain batch did not go through the wide (domain-stacked)
/// GPU CROWN pass.
///
/// Variants are grouped by WHERE the refusal happens:
/// - `Wave*` — the BaB wave partition (ny-propagate `batched_multi`): the child
///   never became part of a batch that reaches the wide entry at all.
/// - `Entry*` — the propagate-side wide entry (`try_gpu_beta_batched_resnet_opt`):
///   a batch existed but never reached a GPU trait call, or the GPU result was
///   rejected on the host afterwards.
/// - `Gpu*` — the GPU batched trait entries / wide assembly (ny-gpu
///   `crown_backward.rs`): the batch reached the device seam and was refused
///   there.
///
/// Ordering is stable and load-bearing only for the printed report; new variants
/// must be APPENDED (the discriminant indexes the counter array).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum WideLaneDecline {
    // ---- wave partition (ny-propagate/batched_multi) -----------------------
    /// A BaB child was excluded from the domain-batchable set (cuts active, not
    /// the dense-spec branch, per-disjunct alphas, a missing ReLU node bound, or
    /// a non-finite first objective) and went to the per-child serial path, so
    /// it can never appear in a wide batch.
    WaveChildNotBatchable = 0,
    /// The requested GPU single-pass chunk width was capped back to the scored
    /// default because an authoritative deadline was in force — the scored
    /// configuration cannot widen the batch even when asked to.
    WaveChunkCappedByDeadline = 1,

    // ---- propagate-side wide entry (batched.rs) ----------------------------
    /// `NY_RESNET_BETA_GPU=0`: the resnet GPU beta lane is switched off.
    EntryResnetBetaGpuDisabled = 2,
    /// The graph BaB deadline had already expired when the entry was called.
    EntryDeadlineExpired = 3,
    /// No backend advertising a sound GPU CROWN (and honouring the deadline) was
    /// available for the wide call.
    EntryNoSoundBackend = 4,
    /// The graph has no `Conv2d`, so the resnet suffix lane does not apply.
    EntryGraphNotConv = 5,
    /// Seed/objective shape predicate refused: zero output dim or spec count,
    /// `num_specs > 512`, `num_specs*output_dim > 2^24`, or a seed length that
    /// does not match `num_specs*output_dim`.
    EntrySeedShapeRejected = 6,
    /// The batch held a single domain: there is nothing to stack, so the wide
    /// pass is skipped by construction (not a defect — the denominator's floor).
    EntrySingleDomainBatch = 7,
    /// `NY_BAB_RESNET_BATCHED=0`: the batched lane is switched off, so the entry
    /// runs the per-domain serial/rayon loop.
    EntryBatchedGateOff = 8,
    /// Bound (non-β-opt) lane: per-domain prep (`prep_resnet_domain_with`)
    /// refused for at least one domain, so no batch could be assembled.
    EntryStackerPrepRefused = 9,
    /// Bound lane: the GPU batched entry returned a result count that did not
    /// match the domain count.
    EntryStackerResultCount = 10,
    /// Bound lane: the runtime wide↔serial re-fold guard rejected the batch.
    EntryStackerRefoldGuard = 11,
    /// Bound lane: a returned per-domain result failed the publishability check
    /// (shape / non-finite / ordering).
    EntryStackerUnpublishable = 12,
    /// β-opt lane: `NY_BAB_RESNET_WIDE_BETA=0`, so an eligible batch runs the
    /// serial per-domain β ascent instead of the wide grad backward.
    EntryWideBetaGateOff = 13,
    /// β-opt lane: per-domain prep refused for at least one domain.
    EntryWideBetaPrepRefused = 14,
    /// β-opt lane: `gpu_beta_optimize_wide` declined (iterate-0 GPU refusal,
    /// unpublishable iterate-0 result, or its own re-fold guard).
    EntryWideBetaDeclined = 15,

    // ---- GPU batched trait entries / wide assembly (ny-gpu) ----------------
    /// A batched trait entry was called with zero domains.
    GpuEmptyBatch = 16,
    /// HOLE 7 homogeneity gate: the domains do not share one network skeleton
    /// (variant sequence, per-layer dims, or shared-weight equality).
    GpuHomogeneityMismatch = 17,
    /// HOLE 8: the skeleton contains `ActivationReluDualAlpha` or `MaxPool2d`,
    /// whose backward shaders are not domain-block-indexed.
    GpuUnbatchableLayer = 18,
    /// The GPU bound entry saw a single-domain batch, so it went straight to the
    /// per-domain kernel without attempting a wide pass.
    GpuSingleDomainBatch = 19,
    /// `NY_BAB_RESNET_WIDE=0`: the wide single-pass kernel is switched off for
    /// A/B, so the entry uses the byte-identical reference stacker.
    GpuWideEnvDisabled = 20,
    /// Even ONE domain's stacked rows would exceed
    /// `max_compute_workgroups_per_dimension` for this skeleton's widest 1-D
    /// dispatch, so no device-safe wide group exists.
    GpuDispatchLimitTooWide = 21,
    /// The optional per-domain pre-activation-lower table did not cover the
    /// whole batch.
    GpuPreLowerShapeMismatch = 22,
    /// A coefficient-frontier capture was requested for a batch that does not
    /// fit one device-safe group; concatenating captures across sub-chunks is
    /// not implemented, so the capture is declined rather than issued oversized.
    GpuCoeffOverDispatchLimit = 23,
    /// `stack_wide_segments` refused (structural mismatch while stacking the
    /// per-domain relaxation blocks onto the shared skeleton).
    GpuSegmentStackRefused = 24,
    /// The shared spec seed's lengths did not match `num_specs * current_dim`.
    GpuSeedShapeRefused = 25,
    /// A domain's input box length differed from domain 0's.
    GpuInputBoxMismatch = 26,
    /// The per-ReLU signed-β table could not be domain-stacked.
    GpuBetaTableStackRefused = 27,
    /// The per-segment frontier abs-max table could not be domain-stacked.
    GpuFrontierTableStackRefused = 28,
    /// The per-ReLU node abs-max table could not be domain-stacked.
    GpuNodeTableStackRefused = 29,
    /// The per-ReLU pre-activation-lower table could not be domain-stacked.
    GpuPreLowerTableStackRefused = 30,
    /// The wide resident fold returned a deadline error.
    GpuWideInnerDeadline = 31,
    /// The wide resident fold returned a host memory-cap refusal.
    GpuWideInnerMemoryCap = 32,
    /// The wide resident fold returned any other error.
    GpuWideInnerError = 33,
    /// The wide resident fold returned a row count other than
    /// `n_domains * num_specs`.
    GpuWideOutputLenMismatch = 34,
    /// A device-safe sub-chunk group failed after an earlier group succeeded, so
    /// the whole batch declines to the serial reference.
    GpuSubchunkGroupFailed = 35,
    /// The coefficient-capturing wide entry produced an empty frontier.
    GpuCoeffFrontierEmpty = 36,
    /// #batched-bab HOLE-7 SUB-GROUPING (dark): the batch was heterogeneous and
    /// the sub-grouping lane split it into homogeneous runs instead of refusing
    /// the whole wave. Counted as a decline of the WHOLE-batch wide pass; the
    /// per-group publications are visible in the taken counter.
    GpuHomogeneitySubgrouped = 37,
}

impl WideLaneDecline {
    /// Every variant, in discriminant order — the report's row order.
    pub const ALL: [WideLaneDecline; Self::COUNT] = [
        WideLaneDecline::WaveChildNotBatchable,
        WideLaneDecline::WaveChunkCappedByDeadline,
        WideLaneDecline::EntryResnetBetaGpuDisabled,
        WideLaneDecline::EntryDeadlineExpired,
        WideLaneDecline::EntryNoSoundBackend,
        WideLaneDecline::EntryGraphNotConv,
        WideLaneDecline::EntrySeedShapeRejected,
        WideLaneDecline::EntrySingleDomainBatch,
        WideLaneDecline::EntryBatchedGateOff,
        WideLaneDecline::EntryStackerPrepRefused,
        WideLaneDecline::EntryStackerResultCount,
        WideLaneDecline::EntryStackerRefoldGuard,
        WideLaneDecline::EntryStackerUnpublishable,
        WideLaneDecline::EntryWideBetaGateOff,
        WideLaneDecline::EntryWideBetaPrepRefused,
        WideLaneDecline::EntryWideBetaDeclined,
        WideLaneDecline::GpuEmptyBatch,
        WideLaneDecline::GpuHomogeneityMismatch,
        WideLaneDecline::GpuUnbatchableLayer,
        WideLaneDecline::GpuSingleDomainBatch,
        WideLaneDecline::GpuWideEnvDisabled,
        WideLaneDecline::GpuDispatchLimitTooWide,
        WideLaneDecline::GpuPreLowerShapeMismatch,
        WideLaneDecline::GpuCoeffOverDispatchLimit,
        WideLaneDecline::GpuSegmentStackRefused,
        WideLaneDecline::GpuSeedShapeRefused,
        WideLaneDecline::GpuInputBoxMismatch,
        WideLaneDecline::GpuBetaTableStackRefused,
        WideLaneDecline::GpuFrontierTableStackRefused,
        WideLaneDecline::GpuNodeTableStackRefused,
        WideLaneDecline::GpuPreLowerTableStackRefused,
        WideLaneDecline::GpuWideInnerDeadline,
        WideLaneDecline::GpuWideInnerMemoryCap,
        WideLaneDecline::GpuWideInnerError,
        WideLaneDecline::GpuWideOutputLenMismatch,
        WideLaneDecline::GpuSubchunkGroupFailed,
        WideLaneDecline::GpuCoeffFrontierEmpty,
        WideLaneDecline::GpuHomogeneitySubgrouped,
    ];

    /// Number of decline reasons (the counter array's length).
    pub const COUNT: usize = 38;

    /// Stable, greppable snake_case label used in the `[wide-lane]` readout.
    /// These strings are a diagnostic contract: tooling greps them, so rename
    /// only together with the consumers.
    pub const fn as_str(self) -> &'static str {
        match self {
            WideLaneDecline::WaveChildNotBatchable => "wave_child_not_batchable",
            WideLaneDecline::WaveChunkCappedByDeadline => "wave_chunk_capped_by_deadline",
            WideLaneDecline::EntryResnetBetaGpuDisabled => "entry_resnet_beta_gpu_disabled",
            WideLaneDecline::EntryDeadlineExpired => "entry_deadline_expired",
            WideLaneDecline::EntryNoSoundBackend => "entry_no_sound_backend",
            WideLaneDecline::EntryGraphNotConv => "entry_graph_not_conv",
            WideLaneDecline::EntrySeedShapeRejected => "entry_seed_shape_rejected",
            WideLaneDecline::EntrySingleDomainBatch => "entry_single_domain_batch",
            WideLaneDecline::EntryBatchedGateOff => "entry_batched_gate_off",
            WideLaneDecline::EntryStackerPrepRefused => "entry_stacker_prep_refused",
            WideLaneDecline::EntryStackerResultCount => "entry_stacker_result_count",
            WideLaneDecline::EntryStackerRefoldGuard => "entry_stacker_refold_guard",
            WideLaneDecline::EntryStackerUnpublishable => "entry_stacker_unpublishable",
            WideLaneDecline::EntryWideBetaGateOff => "entry_wide_beta_gate_off",
            WideLaneDecline::EntryWideBetaPrepRefused => "entry_wide_beta_prep_refused",
            WideLaneDecline::EntryWideBetaDeclined => "entry_wide_beta_declined",
            WideLaneDecline::GpuEmptyBatch => "gpu_empty_batch",
            WideLaneDecline::GpuHomogeneityMismatch => "gpu_homogeneity_mismatch_hole7",
            WideLaneDecline::GpuUnbatchableLayer => "gpu_unbatchable_layer_hole8",
            WideLaneDecline::GpuSingleDomainBatch => "gpu_single_domain_batch",
            WideLaneDecline::GpuWideEnvDisabled => "gpu_wide_env_disabled",
            WideLaneDecline::GpuDispatchLimitTooWide => "gpu_dispatch_limit_too_wide",
            WideLaneDecline::GpuPreLowerShapeMismatch => "gpu_pre_lower_shape_mismatch",
            WideLaneDecline::GpuCoeffOverDispatchLimit => "gpu_coeff_over_dispatch_limit",
            WideLaneDecline::GpuSegmentStackRefused => "gpu_segment_stack_refused",
            WideLaneDecline::GpuSeedShapeRefused => "gpu_seed_shape_refused",
            WideLaneDecline::GpuInputBoxMismatch => "gpu_input_box_mismatch",
            WideLaneDecline::GpuBetaTableStackRefused => "gpu_beta_table_stack_refused",
            WideLaneDecline::GpuFrontierTableStackRefused => "gpu_frontier_table_stack_refused",
            WideLaneDecline::GpuNodeTableStackRefused => "gpu_node_table_stack_refused",
            WideLaneDecline::GpuPreLowerTableStackRefused => "gpu_pre_lower_table_stack_refused",
            WideLaneDecline::GpuWideInnerDeadline => "gpu_wide_inner_deadline",
            WideLaneDecline::GpuWideInnerMemoryCap => "gpu_wide_inner_memory_cap",
            WideLaneDecline::GpuWideInnerError => "gpu_wide_inner_error",
            WideLaneDecline::GpuWideOutputLenMismatch => "gpu_wide_output_len_mismatch",
            WideLaneDecline::GpuSubchunkGroupFailed => "gpu_subchunk_group_failed",
            WideLaneDecline::GpuCoeffFrontierEmpty => "gpu_coeff_frontier_empty",
            WideLaneDecline::GpuHomogeneitySubgrouped => "gpu_homogeneity_subgrouped",
        }
    }
}

/// One zero counter, copied into every slot of [`COUNTS`]. `const` items are
/// copied at each use, so this is 38 distinct atomics, not 38 aliases.
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);

static COUNTS: [AtomicU64; WideLaneDecline::COUNT] = [ZERO; WideLaneDecline::COUNT];

/// Multi-domain batches that reached the propagate-side wide entry.
static CANDIDATES: AtomicU64 = AtomicU64::new(0);

/// Candidate batches that reached a batched GPU trait entry (the denominator the
/// published-count is directly comparable against).
static ATTEMPTS: AtomicU64 = AtomicU64::new(0);

/// Record that a candidate domain batch reached the propagate-side wide entry.
/// One relaxed atomic increment; never read by any decision.
#[inline]
pub fn note_wide_lane_candidate() {
    CANDIDATES.fetch_add(1, Ordering::Relaxed);
}

/// Record that a candidate batch reached a batched GPU trait entry.
/// One relaxed atomic increment; never read by any decision.
#[inline]
pub fn note_wide_lane_attempt() {
    ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

/// Runs published by the HOMOGENEOUS SUB-GROUPING path (review defect 2).
///
/// A sub-grouped wave is ONE attempt that publishes SEVERAL runs, so counting
/// it as a decline both (a) files a success under `declines:` and (b) makes
/// the documented `attempts - published` coverage gap go negative. It gets its
/// own counter, reported beside the tally and never inside it.
static SUBGROUPED_RUNS: AtomicU64 = AtomicU64::new(0);

/// Record one published sub-group run.
pub fn note_wide_lane_subgrouped_run() {
    SUBGROUPED_RUNS.fetch_add(1, Ordering::Relaxed);
}

/// Published sub-group runs so far in this process.
#[must_use]
pub fn wide_lane_subgrouped_runs() -> u64 {
    SUBGROUPED_RUNS.load(Ordering::Relaxed)
}

/// Record WHY a candidate batch did not take the wide lane.
/// One relaxed atomic increment; never read by any decision.
#[inline]
pub fn note_wide_lane_decline(reason: WideLaneDecline) {
    COUNTS[reason as usize].fetch_add(1, Ordering::Relaxed);
}

/// Convenience for `Option`-returning refusal paths:
/// `return note_wide_lane_decline_none(reason);`
#[inline]
pub fn note_wide_lane_decline_none<T>(reason: WideLaneDecline) -> Option<T> {
    note_wide_lane_decline(reason);
    None
}

/// Candidate batches seen at the propagate-side wide entry.
pub fn wide_lane_candidate_count() -> u64 {
    CANDIDATES.load(Ordering::Relaxed)
}

/// Candidate batches that reached a batched GPU trait entry.
pub fn wide_lane_attempt_count() -> u64 {
    ATTEMPTS.load(Ordering::Relaxed)
}

/// Count for one decline reason.
pub fn wide_lane_decline_count(reason: WideLaneDecline) -> u64 {
    COUNTS[reason as usize].load(Ordering::Relaxed)
}

/// Every NON-ZERO decline reason, in discriminant order. Allocates — a reader-side
/// convenience for the CLI readout and tests only.
pub fn wide_lane_decline_tally() -> Vec<(&'static str, u64)> {
    WideLaneDecline::ALL
        .iter()
        .filter_map(|&reason| {
            let count = wide_lane_decline_count(reason);
            (count > 0).then_some((reason.as_str(), count))
        })
        .collect()
}

/// One-line, greppable rendering of the tally for the dark `[wide-lane]` readout.
/// `published` is the caller's `ny_gpu::wide_resnet_batched_taken_count()` — this
/// crate deliberately does not depend on the GPU crate.
///
/// Shape: `candidates=<n> gpu_attempts=<n> published=<n> declines: a=1 b=2`
/// (or `declines: none`). Reader-side only; allocates.
pub fn format_wide_lane_tally(published: u64) -> String {
    let tally = wide_lane_decline_tally();
    let mut out = format!(
        "candidates={} gpu_attempts={} published={} subgrouped_runs={} declines:",
        wide_lane_candidate_count(),
        wide_lane_attempt_count(),
        published,
        wide_lane_subgrouped_runs(),
    );
    if tally.is_empty() {
        out.push_str(" none");
    } else {
        for (name, count) in tally {
            out.push(' ');
            out.push_str(name);
            out.push('=');
            out.push_str(&count.to_string());
        }
    }
    out
}

/// Zero every counter. TEST-ONLY: production is monotonic for the life of the
/// process so either CLI completion site prints the same run's totals.
#[doc(hidden)]
pub fn reset_wide_lane_telemetry_for_tests() {
    CANDIDATES.store(0, Ordering::Relaxed);
    ATTEMPTS.store(0, Ordering::Relaxed);
    SUBGROUPED_RUNS.store(0, Ordering::Relaxed);
    for slot in &COUNTS {
        slot.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-global, so the mutating tests below must not run
    /// concurrently with each other (cargo runs a crate's tests in parallel).
    /// A poisoned lock is irrelevant here — the state is reset on entry.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The discriminant indexes [`COUNTS`]; a variant whose position in `ALL`
    /// disagrees with its discriminant would silently increment its neighbour's
    /// counter and mislabel every future measurement.
    #[test]
    fn all_variants_are_in_discriminant_order() {
        for (index, reason) in WideLaneDecline::ALL.iter().enumerate() {
            assert_eq!(
                *reason as usize,
                index,
                "{} is at position {index} but has discriminant {}",
                reason.as_str(),
                *reason as usize
            );
        }
        assert_eq!(WideLaneDecline::ALL.len(), WideLaneDecline::COUNT);
    }

    /// Labels are the diagnostic contract the readout greps: they must be unique
    /// and non-empty, or two reasons collapse into one row.
    #[test]
    fn labels_are_unique_and_nonempty() {
        let mut seen = std::collections::HashSet::new();
        for reason in WideLaneDecline::ALL {
            let label = reason.as_str();
            assert!(!label.is_empty(), "{reason:?} has an empty label");
            assert!(seen.insert(label), "duplicate wide-lane label {label}");
        }
    }

    /// `const ZERO` must produce INDEPENDENT atomics (the copy-per-use rule); if
    /// it aliased, every reason would share one counter.
    #[test]
    fn counters_are_independent() {
        let _guard = exclusive();
        reset_wide_lane_telemetry_for_tests();
        note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
        note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
        note_wide_lane_decline(WideLaneDecline::EntryWideBetaGateOff);
        assert_eq!(
            wide_lane_decline_count(WideLaneDecline::GpuHomogeneityMismatch),
            2
        );
        assert_eq!(
            wide_lane_decline_count(WideLaneDecline::EntryWideBetaGateOff),
            1
        );
        assert_eq!(wide_lane_decline_count(WideLaneDecline::GpuEmptyBatch), 0);
        reset_wide_lane_telemetry_for_tests();
    }

    /// CPU PIN, one per tally variant: every reason can be recorded, is reported
    /// under its own label, and disappears from the tally when zero. This is the
    /// pin the task asks for — it fails the moment a variant is added to the enum
    /// without a label or a slot.
    #[test]
    fn every_decline_variant_records_and_reports() {
        let _guard = exclusive();
        for (index, reason) in WideLaneDecline::ALL.iter().enumerate() {
            reset_wide_lane_telemetry_for_tests();
            assert!(
                wide_lane_decline_tally().is_empty(),
                "tally must be empty after reset"
            );
            // A distinct count per variant catches an off-by-one slot mapping.
            let times = (index as u64) + 1;
            for _ in 0..times {
                note_wide_lane_decline(*reason);
            }
            assert_eq!(wide_lane_decline_count(*reason), times);
            assert_eq!(
                wide_lane_decline_tally(),
                vec![(reason.as_str(), times)],
                "exactly one row expected for {}",
                reason.as_str()
            );
            let line = format_wide_lane_tally(7);
            assert!(
                line.contains(&format!("{}={times}", reason.as_str())),
                "readout {line} must name {}",
                reason.as_str()
            );
            assert!(
                line.contains("published=7"),
                "readout {line} lost published"
            );
        }
        reset_wide_lane_telemetry_for_tests();
    }

    /// The readout must be legible (and honest) when nothing declined.
    #[test]
    fn empty_tally_reads_as_none() {
        let _guard = exclusive();
        reset_wide_lane_telemetry_for_tests();
        note_wide_lane_candidate();
        note_wide_lane_attempt();
        note_wide_lane_attempt();
        note_wide_lane_subgrouped_run();
        assert_eq!(wide_lane_subgrouped_runs(), 1);
        reset_wide_lane_telemetry_for_tests();
        assert_eq!(wide_lane_subgrouped_runs(), 0);
        note_wide_lane_candidate();
        note_wide_lane_attempt();
        note_wide_lane_attempt();
        assert_eq!(
            format_wide_lane_tally(2),
            "candidates=1 gpu_attempts=2 published=2 subgrouped_runs=0 declines: none"
        );
        reset_wide_lane_telemetry_for_tests();
    }

    /// `note_wide_lane_decline_none` is the refusal-path helper: it must both
    /// record and yield `None` (a helper that forgot the increment would leave
    /// the very declines we are hunting invisible).
    #[test]
    fn decline_none_helper_records_and_returns_none() {
        let _guard = exclusive();
        reset_wide_lane_telemetry_for_tests();
        let value: Option<u32> =
            note_wide_lane_decline_none(WideLaneDecline::GpuSegmentStackRefused);
        assert!(value.is_none());
        assert_eq!(
            wide_lane_decline_count(WideLaneDecline::GpuSegmentStackRefused),
            1
        );
        reset_wide_lane_telemetry_for_tests();
    }
}
