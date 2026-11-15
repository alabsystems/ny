// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-seed`: the plan, memory admission and observables for the
//! seeded, optionally parallel child-refine prepass
//! (`shared::install_child_interm_refine_overrides_scheduled`).
//!
//! Three typed, default-dark keys ride on the existing
//! `#cgan-child-interm-refine` prepass and are inert without its per-child
//! budget (`bab.alpha_crown.cgan_child_interm_refine_max_secs`):
//!
//! * `cgan_child_interm_refine_seed_from_parent` — start each child cascade
//!   from its forward-linear baseline INTERSECTED with the inherited root map
//!   and the domain's own override;
//! * `cgan_child_interm_refine_parallel` — run up to K child cascades of one
//!   batch concurrently, each tranche admitted against memory (below);
//! * `cgan_child_interm_refine_fresh_clip_map` — hand the domain's own map to
//!   the fresh exact-domain clip (`disjunctive_multi_clause/process_batch.rs`).
//!
//! # Memory admission (the uncapped competition configuration)
//!
//! `vnncomp_scripts/run_instance.sh` installs no cgroup and no RLIMIT, so a
//! cgroup-only check (`batching::override_rebound_has_process_headroom`)
//! refuses by construction exactly where this must run. Admission here reads
//! `crown_memory::process_memory_headroom_bytes`, the narrowest of the cgroup,
//! RLIMIT_AS and the self-governing `MemAvailable − reserve` / self-RSS
//! backstop terms — the last of which exists precisely for the uncapped case.
//! A tranche of `w` workers is admitted only if `w × worker_estimate` fits in
//! `min(headroom / 2, 48 GiB)`; an unobservable headroom runs serially.
//!
//! The per-worker estimate is deliberately a worst case: two live dense
//! coefficient pairs of the widest activation (each capped by the live Dense
//! budget, which is itself ≤ headroom/8), plus six copies of the inherited map
//! and a 256 MiB floor. It is a POLICY bound, not a soundness input: admission
//! can only change how many children run at once, never what any of them
//! publishes. `peak_rss_mb` on the per-batch line is what calibrates it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

use ny_tensor::BoundedTensor;

use super::shared::{ChildIntermRefineBatch, ChildRefineSchedule};
use crate::beta_crown::config::BetaCrownConfig;

/// Per-worker floor for allocator, traversal and small-matrix scratch.
const CHILD_SEED_WORKER_FLOOR_BYTES: u64 = 256 << 20;
/// Bytes charged per inherited-map cell: lower+upper `f32` (8 B) times six
/// live copies (root seed read, forward-linear baseline, its clone, the seeded
/// baseline, the cascade result, and the merged override).
const CHILD_SEED_BYTES_PER_MAP_CELL: u64 = 48;
/// Dense coefficient pairs a backward walk can hold at once (the current
/// layer's and the next one's).
const CHILD_SEED_LIVE_DENSE_PAIRS: u64 = 2;
/// Bytes per dense pair element: lower and upper `f32` coefficient.
const CHILD_SEED_DENSE_PAIR_ELEMENT_BYTES: u64 = 8;
/// A parallel tranche may spend at most this fraction (1/N) of live headroom.
const CHILD_SEED_HEADROOM_DIVISOR: u64 = 2;
/// Absolute ceiling on the estimated bytes of one parallel tranche, whatever
/// the host reports. Unified-memory hosts share this with the GPU.
const CHILD_SEED_ABSOLUTE_CAP_BYTES: u64 = 48 << 30;
/// Hard ceiling on the width of one tranche.
const CHILD_SEED_MAX_WIDTH: usize = 16;

/// One parallel prepass at a time per process (see the installer).
pub(super) static CHILD_REFINE_EXECUTION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The admitted width for one tranche and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChildRefineAdmission {
    /// Workers to run in this tranche; always at least 1.
    pub(super) width: usize,
    /// Fixed-vocabulary reason, printed on the per-batch line.
    pub(super) reason: &'static str,
    /// True when parallelism was requested and memory forced it to serial.
    pub(super) memory_declined: bool,
}

/// Pure admission policy; see the module doc.
///
/// `requested` is the configured ceiling, `worker_bytes` the per-worker
/// estimate, `headroom` the live process headroom (`None` = unobservable).
pub(super) fn admitted_width(
    requested: usize,
    worker_bytes: u64,
    headroom: Option<u64>,
) -> ChildRefineAdmission {
    let requested = requested.min(CHILD_SEED_MAX_WIDTH);
    if requested <= 1 {
        return ChildRefineAdmission {
            width: 1,
            reason: "serial-requested",
            memory_declined: false,
        };
    }
    let Some(headroom) = headroom else {
        return ChildRefineAdmission {
            width: 1,
            reason: "headroom-unobservable",
            memory_declined: true,
        };
    };
    let budget = (headroom / CHILD_SEED_HEADROOM_DIVISOR).min(CHILD_SEED_ABSOLUTE_CAP_BYTES);
    let fit = usize::try_from(budget / worker_bytes.max(1)).unwrap_or(usize::MAX);
    if fit < 2 {
        return ChildRefineAdmission {
            width: 1,
            reason: "memory-declined",
            memory_declined: true,
        };
    }
    let width = requested.min(fit);
    ChildRefineAdmission {
        width,
        reason: if width < requested {
            "memory-capped"
        } else {
            "admitted"
        },
        memory_declined: false,
    }
}

/// Conservative per-worker resident estimate for one child cascade.
///
/// `inherited_root` sizes the map copies and the widest activation; with no
/// map the dense term falls back to the full live Dense budget.
pub(super) fn child_refine_worker_estimate_bytes(
    inherited_root: Option<&HashMap<String, BoundedTensor>>,
    dense_budget_bytes: u64,
) -> u64 {
    let (map_cells, widest) = inherited_root.map_or((0u64, 0u64), |map| {
        map.values().fold((0u64, 0u64), |(total, widest), bounds| {
            let cells = u64::try_from(bounds.lower().len()).unwrap_or(u64::MAX);
            (total.saturating_add(cells), widest.max(cells))
        })
    });
    let dense_pair = if widest == 0 {
        dense_budget_bytes
    } else {
        dense_budget_bytes.min(
            widest
                .saturating_mul(widest)
                .saturating_mul(CHILD_SEED_DENSE_PAIR_ELEMENT_BYTES),
        )
    };
    CHILD_SEED_WORKER_FLOOR_BYTES
        .saturating_add(map_cells.saturating_mul(CHILD_SEED_BYTES_PER_MAP_CELL))
        .saturating_add(dense_pair.saturating_mul(CHILD_SEED_LIVE_DENSE_PAIRS))
}

/// Whether any `#cgan-child-seed` key is set on this config.
pub(super) fn any_child_seed_key_armed(config: &BetaCrownConfig) -> bool {
    let alpha = &config.alpha_config;
    alpha.cgan_child_interm_refine_seed_from_parent
        || alpha.cgan_child_interm_refine_parallel > 1
        || alpha.cgan_child_interm_refine_fresh_clip_map
}

/// Resolve one deferred batch's child-refine plan: the historical per-child
/// alpha config (unchanged gate — `None` exactly when it was `None` before)
/// plus the `#cgan-child-seed` schedule.
///
/// Prints the one-time `[cgan-child-seed] armed` line whenever any of the
/// three keys is set, INCLUDING when the prepass itself is inert, so a key
/// that reached the config without the per-child budget reads `prepass=inert`
/// rather than as a silent negative.
pub(super) fn child_refine_plan(
    config: &BetaCrownConfig,
    deadline: Option<Instant>,
    inherited_root: Option<&HashMap<String, BoundedTensor>>,
) -> Option<(crate::AlphaCrownConfig, ChildRefineSchedule)> {
    let refine = super::shared::child_interm_refine_alpha_config(config, deadline);
    if any_child_seed_key_armed(config) {
        announce_armed(config, refine.is_some());
    }
    let refine = refine?;
    let alpha = &config.alpha_config;
    let requested_width = alpha.cgan_child_interm_refine_parallel.max(1);
    let worker_estimate_bytes = if requested_width > 1 {
        let dense_budget =
            u64::try_from(crate::network::crown_memory::cpu_crown_dense_budget_bytes())
                .unwrap_or(u64::MAX);
        child_refine_worker_estimate_bytes(inherited_root, dense_budget)
    } else {
        0
    };
    Some((
        refine,
        ChildRefineSchedule {
            seed_from_parent: alpha.cgan_child_interm_refine_seed_from_parent,
            requested_width,
            worker_estimate_bytes,
            headroom_probe: crate::network::crown_memory::process_memory_headroom_bytes,
        },
    ))
}

/// Whether a deferred rebound should keep the child-local map it consumed on
/// the domain for the fresh clip. True only when the prepass ran AND the
/// fresh-clip key is set; otherwise the rebound clears it exactly as before.
pub(super) fn retain_map_for_fresh_clip(config: &BetaCrownConfig, prepass_ran: bool) -> bool {
    prepass_ran && config.alpha_config.cgan_child_interm_refine_fresh_clip_map
}

static ARMED: Once = Once::new();

fn announce_armed(config: &BetaCrownConfig, prepass_armed: bool) {
    ARMED.call_once(|| {
        let alpha = &config.alpha_config;
        let prepass = if prepass_armed {
            "armed"
        } else if alpha
            .cgan_child_interm_refine_max_secs
            .is_none_or(|secs| !secs.is_finite() || secs <= 0.0)
        {
            "inert:max_secs_unset"
        } else {
            "inert:root_unarmed"
        };
        eprintln!(
            "[cgan-child-seed] armed seed_from_parent={} parallel={} fresh_clip_map={} \
             prepass={prepass} source=bab.alpha_crown.cgan_child_interm_refine_*",
            u8::from(alpha.cgan_child_interm_refine_seed_from_parent),
            alpha.cgan_child_interm_refine_parallel,
            u8::from(alpha.cgan_child_interm_refine_fresh_clip_map),
        );
    });
}

static BATCHES: AtomicU64 = AtomicU64::new(0);

/// The per-batch engagement line. Printed for EVERY prepass batch while any
/// `#cgan-child-seed` key is set (never when all three are dark, so the
/// historical prepass keeps its stderr byte-identical). Fixed schema; a
/// zero-valued field is a measured zero, never an omitted one.
pub(super) fn report_batch(
    config: &BetaCrownConfig,
    tally: &ChildIntermRefineBatch,
    schedule: &ChildRefineSchedule,
    wall: Duration,
) {
    if !any_child_seed_key_armed(config) {
        return;
    }
    let batch = BATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    let children = tally.installed + tally.declined;
    let mean_child_s = if children == 0 {
        0.0
    } else {
        tally.child_nanos as f64 / 1e9 / children as f64
    };
    let headroom = (schedule.headroom_probe)();
    let admission = admitted_width(
        schedule.requested_width,
        schedule.worker_estimate_bytes,
        headroom,
    );
    let (rss_mb, peak_rss_mb) = self_rss_mb();
    let headroom_mb = headroom.map(|bytes| bytes >> 20);
    eprintln!(
        "[cgan-child-seed] batch={batch} children={children} installed={} declined={} \
         seeded={} seed_unavailable={} seed_nodes={} seed_declined_nodes={} \
         seed_tightened_elems={} unstable_rows_unseeded={} unstable_rows_seeded={} \
         width_max={} requested={} tranches={} memory_declined_tranches={} admission_now={} \
         worker_est_mb={} headroom_mb={} wall_s={:.3} mean_child_s={:.3} rss_mb={} \
         peak_rss_mb={}",
        tally.installed,
        tally.declined,
        tally.seeded,
        tally.seed_unavailable,
        tally.seed_nodes,
        tally.seed_declined_nodes,
        tally.seed_tightened_elems,
        tally.unstable_rows_unseeded,
        tally.unstable_rows_seeded,
        tally.max_width,
        schedule.requested_width,
        tally.tranches,
        tally.memory_declined_tranches,
        admission.reason,
        schedule.worker_estimate_bytes >> 20,
        fmt_opt(headroom_mb),
        wall.as_secs_f64(),
        mean_child_s,
        fmt_opt(rss_mb),
        fmt_opt(peak_rss_mb),
    );
}

static FRESH_CLIP_CALLS: AtomicU64 = AtomicU64::new(0);
static FRESH_CLIP_WITH_MAP: AtomicU64 = AtomicU64::new(0);
static FRESH_CLIP_REFUTED_WITH_MAP: AtomicU64 = AtomicU64::new(0);
static FRESH_CLIP_APPLIED_WITH_MAP: AtomicU64 = AtomicU64::new(0);

/// `cgan_child_interm_refine_fresh_clip_map` engagement: called once per
/// fresh clip while the key is set. Prints cumulative counts for the first
/// four calls and every 32nd thereafter.
pub(super) fn observe_fresh_clip_map(had_map: bool, refuted: bool, applied: bool) {
    let calls = FRESH_CLIP_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let with_map =
        FRESH_CLIP_WITH_MAP.fetch_add(u64::from(had_map), Ordering::Relaxed) + u64::from(had_map);
    let refuted_with_map = FRESH_CLIP_REFUTED_WITH_MAP
        .fetch_add(u64::from(had_map && refuted), Ordering::Relaxed)
        + u64::from(had_map && refuted);
    let applied_with_map = FRESH_CLIP_APPLIED_WITH_MAP
        .fetch_add(u64::from(had_map && applied), Ordering::Relaxed)
        + u64::from(had_map && applied);
    if calls <= 4 || calls.is_multiple_of(32) {
        eprintln!(
            "[cgan-child-seed] fresh_clip calls={calls} with_map={with_map} \
             refuted_with_map={refuted_with_map} applied_with_map={applied_with_map}"
        );
    }
}

fn fmt_opt(value: Option<u64>) -> String {
    value.map_or_else(|| "na".to_string(), |v| v.to_string())
}

/// `(VmRSS, VmHWM)` of this process in MiB, where the platform exposes them.
fn self_rss_mb() -> (Option<u64>, Option<u64>) {
    #[cfg(target_os = "linux")]
    {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return (None, None);
        };
        let field = |name: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kib| kib.parse::<u64>().ok())
                .map(|kib| kib >> 10)
        };
        (field("VmRSS:"), field("VmHWM:"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        (None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::{admitted_width, child_refine_worker_estimate_bytes, CHILD_SEED_MAX_WIDTH};
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    const GIB: u64 = 1 << 30;

    /// The uncapped competition shape: no cgroup, a MemAvailable-derived
    /// headroom. Admission must engage there (the cgroup-only check refuses
    /// it by construction) and must scale down, never up, with the estimate.
    #[test]
    fn child_seed_admission_engages_uncapped_and_narrows_with_memory() {
        let admitted = admitted_width(4, GIB, Some(100 * GIB));
        assert_eq!(admitted.width, 4);
        assert_eq!(admitted.reason, "admitted");

        // 13.5 GiB workers against 100 GiB headroom: budget min(50, 48) = 48
        // GiB fits 3, so a request for 4 is capped, not granted.
        let capped = admitted_width(4, 13 * GIB + GIB / 2, Some(100 * GIB));
        assert_eq!(capped.width, 3);
        assert_eq!(capped.reason, "memory-capped");

        // The absolute cap binds even on an enormous host.
        let huge_host = admitted_width(16, 13 * GIB, Some(10_000 * GIB));
        assert_eq!(huge_host.width, 3, "48 GiB absolute cap / 13 GiB = 3");

        // Less than two workers' worth: serial, flagged as a memory decline.
        let declined = admitted_width(4, 30 * GIB, Some(100 * GIB));
        assert_eq!(declined.width, 1);
        assert!(declined.memory_declined);
    }

    /// Fail closed: an unobservable headroom, a zero headroom (the self-RSS
    /// backstop's refuse-everything value) or a serial request all run 1.
    #[test]
    fn child_seed_admission_fails_closed_to_serial() {
        assert_eq!(admitted_width(8, GIB, None).width, 1);
        assert!(admitted_width(8, GIB, None).memory_declined);
        assert_eq!(admitted_width(8, GIB, Some(0)).width, 1);
        assert_eq!(admitted_width(0, GIB, Some(100 * GIB)).width, 1);
        assert_eq!(admitted_width(1, GIB, Some(100 * GIB)).width, 1);
        assert!(!admitted_width(1, GIB, Some(100 * GIB)).memory_declined);
        // A zero estimate cannot divide by zero or grant unbounded width.
        assert_eq!(
            admitted_width(usize::MAX, 0, Some(u64::MAX)).width,
            CHILD_SEED_MAX_WIDTH
        );
    }

    /// The estimate is monotone in the inherited map and never below the
    /// floor; with no map it charges the whole live Dense budget.
    #[test]
    fn child_seed_worker_estimate_is_conservative() {
        let dense_budget = 12 * GIB;
        let no_map = child_refine_worker_estimate_bytes(None, dense_budget);
        assert!(
            no_map >= 2 * dense_budget,
            "no map must charge two full dense pairs"
        );

        let mut map = HashMap::new();
        let cells = 1000usize;
        map.insert(
            "wide".to_string(),
            BoundedTensor::new(
                ArrayD::zeros(IxDyn(&[cells])),
                ArrayD::ones(IxDyn(&[cells])),
            )
            .expect("bounds"),
        );
        let small = child_refine_worker_estimate_bytes(Some(&map), dense_budget);
        // 1000^2 * 8 B per pair, two pairs, plus 48 B/cell and the floor.
        assert_eq!(small, (256u64 << 20) + 1000 * 48 + 2 * 1000 * 1000 * 8,);
        map.insert(
            "wider".to_string(),
            BoundedTensor::new(
                ArrayD::zeros(IxDyn(&[4 * cells])),
                ArrayD::ones(IxDyn(&[4 * cells])),
            )
            .expect("bounds"),
        );
        let larger = child_refine_worker_estimate_bytes(Some(&map), dense_budget);
        assert!(larger > small, "the estimate must grow with the map");
        // And the dense term never exceeds the live Dense budget per pair.
        let capped = child_refine_worker_estimate_bytes(Some(&map), 1024);
        assert_eq!(capped, (256u64 << 20) + 5000 * 48 + 2 * 1024);
    }
}
