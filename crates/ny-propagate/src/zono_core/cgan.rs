// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-zono`: the typed-key plan, the per-batch tally and its
//! fixed-schema observable, and the process receipt the CLI turns into a
//! flight note.
//!
//! The lane itself (job snapshot, memory admission, parallel tranches, map
//! installation) lives beside the other deferred-rebound prepasses in
//! `beta_crown/engine/graph/input_split/child_zono.rs`; this module holds the
//! parts the CLI and the tests must reach from outside the engine.

use std::sync::Mutex;

use ny_levers::Source;

use super::ZonoRefusal;
use crate::beta_crown::config::BetaCrownConfig;

/// Resolved `bab.alpha_crown.cgan_child_zono_refresh` for one run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CganChildZonoPlan {
    /// Generator budget per child (never zero; the key's validator refuses 0).
    pub max_generators: usize,
    /// Per-child wall budget in seconds (`None`: only the BaB deadline).
    pub max_secs: Option<f64>,
    /// Where the arming came from. The key has NO environment reader, so this
    /// is always [`Source::Config`] when a plan exists — pinned by
    /// `child_zono_preset_source_is_typed` so an env shortcut can never be
    /// added silently.
    pub source: Source,
}

/// The plan for `config`, or `None` when the key is dark (the default). A
/// `None` plan means the prepass does no work and prints nothing.
#[must_use]
pub fn cgan_child_zono_plan(config: &BetaCrownConfig) -> Option<CganChildZonoPlan> {
    let refresh = &config.alpha_config.cgan_child_zono_refresh;
    if !refresh.enabled {
        return None;
    }
    Some(CganChildZonoPlan {
        max_generators: usize::try_from(refresh.max_generators)
            .unwrap_or(usize::MAX)
            .max(1),
        max_secs: refresh
            .max_secs
            .filter(|secs| secs.is_finite() && *secs > 0.0)
            .map(f64::from),
        source: Source::Config,
    })
}

/// Which evaluator ran a deferred batch's children (#m8-cgan-leaf-batch).
///
/// Both publish bitwise-identical maps; they differ only in how the work is
/// scheduled. KEYLESS: the prepass uses [`ChildZonoEvaluator::Batch`] in
/// production whenever the memory headroom can be observed, and falls back
/// to [`ChildZonoEvaluator::Serial`] (today's schedule) when it cannot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChildZonoEvaluator {
    /// Today's schedule: one `ZonoProgram::forward` per child (children of a
    /// tranche on parallel workers), every row pass sequential.
    #[default]
    Serial,
    /// `ZonoProgram::forward_batch`: the tranche's children concurrently AND
    /// every row pass row-parallel (the reference's own blocks).
    Batch,
}

impl ChildZonoEvaluator {
    /// Stable lowercase token for observables.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Batch => "batch",
        }
    }
}

/// What one deferred batch's zonotope prepass did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CganChildZonoBatchTally {
    /// Children the prepass was asked to refresh.
    pub children: usize,
    /// Children whose certified map was installed.
    pub installed: usize,
    /// Installed children that already carried an override (merged into it).
    pub merged_into_override: usize,
    /// Refusals: unsupported graph or layer.
    pub refused_unsupported: usize,
    /// Refusals: per-child or BaB deadline.
    pub refused_deadline: usize,
    /// Refusals: NaN/Inf or a failed internal check.
    pub refused_numeric: usize,
    /// Refusals: memory admission.
    pub refused_memory: usize,
    /// Refusals: anything else (shape, merge).
    pub refused_other: usize,
    /// Generator rows at the output, summed over installed children.
    pub gens_spent: usize,
    /// Most generator rows any child held.
    pub gens_peak: usize,
    /// Rows folded into remainders, summed over children.
    pub folded: usize,
    /// Unstable ReLU rows under the zonotope alone (summed).
    pub unstable_zono: usize,
    /// Unstable rows under the inherited maps, over ReLUs that had an
    /// inherited entry (summed over installed children).
    pub unstable_before: usize,
    /// Unstable rows after intersecting with the zonotope, same ReLUs.
    pub unstable_after: usize,
    /// Summed per-child zonotope wall time, nanoseconds.
    pub child_nanos: u64,
    /// Largest certified remainder over every installed child.
    pub cert_charge_max: f64,
    /// Widest tranche run.
    pub width_max: usize,
    /// Tranches run.
    pub tranches: usize,
    /// #m8-cgan-leaf-batch: [`ChildZonoEvaluator::Batch`] when at least one
    /// tranche ran through `forward_batch`, else [`ChildZonoEvaluator::Serial`].
    pub engine: ChildZonoEvaluator,
    /// Tranches that ran through `forward_batch`.
    pub batch_tranches: usize,
    /// Generator-kernel row blocks run as rayon tasks, summed over installed
    /// children (telemetry only).
    pub row_tasks: usize,
    /// Wall time of the whole prepass for this batch, nanoseconds (compile,
    /// estimates, tranches and installation).
    pub prepass_nanos: u64,
}

impl CganChildZonoBatchTally {
    /// Total refusals.
    #[must_use]
    pub fn refused(&self) -> usize {
        self.refused_unsupported
            + self.refused_deadline
            + self.refused_numeric
            + self.refused_memory
            + self.refused_other
    }

    /// Count one refusal by kind.
    pub fn record_refusal(&mut self, refusal: &ZonoRefusal) {
        match refusal {
            ZonoRefusal::UnsupportedGraph(_) | ZonoRefusal::UnsupportedLayer { .. } => {
                self.refused_unsupported += 1;
            }
            ZonoRefusal::Deadline => self.refused_deadline += 1,
            ZonoRefusal::NonFinite(_) | ZonoRefusal::Internal(_) => self.refused_numeric += 1,
            ZonoRefusal::Memory { .. } => self.refused_memory += 1,
            ZonoRefusal::Shape(_) => self.refused_other += 1,
        }
    }

    /// Mean per-child zonotope wall time in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_ms(&self) -> f64 {
        let done = self.installed + self.refused();
        if done == 0 {
            0.0
        } else {
            self.child_nanos as f64 / 1e6 / done as f64
        }
    }

    /// Wall time of the whole prepass for this batch, milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn prepass_ms(&self) -> f64 {
        self.prepass_nanos as f64 / 1e6
    }

    /// The fixed-schema per-batch observable. The first eleven fields are the
    /// brief's schema, in its order; diagnostic fields follow `refused=`.
    /// #m8-cgan-leaf-batch appends ` engine=<serial|batch> row_tasks=..
    /// prepass_ms=.. m8-cgan-leaf-batch/v1` at the END (append-only).
    #[must_use]
    pub fn render(&self, batch: u64, plan: &CganChildZonoPlan, peak_rss_mb: Option<u64>) -> String {
        format!(
            "[cgan-child-zono] armed=1 children={} installed={} gens_spent={} folded={} \
             unstable_before={} unstable_after={} mean_ms={:.1} peak_rss_mb={} \
             cert_charge_max={:.3e} refused={} batch={batch} width_max={} tranches={} \
             unstable_zono={} merged_into_override={} refused_unsupported={} \
             refused_deadline={} refused_numeric={} refused_memory={} refused_other={} \
             max_generators={} engine={} row_tasks={} prepass_ms={:.1} m8-cgan-leaf-batch/v1",
            self.children,
            self.installed,
            self.gens_spent,
            self.folded,
            self.unstable_before,
            self.unstable_after,
            self.mean_ms(),
            peak_rss_mb.map_or_else(|| "na".to_string(), |mb| mb.to_string()),
            self.cert_charge_max,
            self.refused(),
            self.width_max,
            self.tranches,
            self.unstable_zono,
            self.merged_into_override,
            self.refused_unsupported,
            self.refused_deadline,
            self.refused_numeric,
            self.refused_memory,
            self.refused_other,
            plan.max_generators,
            self.engine.token(),
            self.row_tasks,
            self.prepass_ms(),
        )
    }
}

/// Run-level totals for the flight note.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CganChildZonoReceipt {
    /// Prepass batches run.
    pub batches: u64,
    /// Summed [`CganChildZonoBatchTally`] over the run.
    pub totals: CganChildZonoBatchTally,
    /// Largest VmHWM observed at a batch end, MiB.
    pub peak_rss_mb: Option<u64>,
    /// Generator budget of the plan.
    pub max_generators: usize,
    /// Fresh-clip calls seen while armed.
    pub fresh_clip_calls: u64,
    /// ... of which carried the domain's refreshed map.
    pub fresh_clip_with_map: u64,
    /// ... of which refuted every clause.
    pub fresh_clip_refuted_with_map: u64,
    /// ... of which applied a clipped box.
    pub fresh_clip_applied_with_map: u64,
}

impl CganChildZonoReceipt {
    /// Fixed-schema flight-note payload.
    #[must_use]
    pub fn render(&self) -> String {
        let t = &self.totals;
        format!(
            "armed=1 batches={} children={} installed={} refused={} gens_spent={} gens_peak={} \
             folded={} unstable_before={} unstable_after={} mean_ms={:.1} peak_rss_mb={} \
             cert_charge_max={:.3e} max_generators={} fresh_clip_calls={} \
             fresh_clip_with_map={} fresh_clip_refuted_with_map={} fresh_clip_applied_with_map={}",
            self.batches,
            t.children,
            t.installed,
            t.refused(),
            t.gens_spent,
            t.gens_peak,
            t.folded,
            t.unstable_before,
            t.unstable_after,
            t.mean_ms(),
            self.peak_rss_mb
                .map_or_else(|| "na".to_string(), |mb| mb.to_string()),
            t.cert_charge_max,
            self.max_generators,
            self.fresh_clip_calls,
            self.fresh_clip_with_map,
            self.fresh_clip_refuted_with_map,
            self.fresh_clip_applied_with_map,
        )
    }
}

static RECEIPT: Mutex<Option<CganChildZonoReceipt>> = Mutex::new(None);

/// Clear the receipt before one top-level verification.
pub fn reset_cgan_child_zono_receipt() {
    *RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Take the receipt after one top-level verification. `None` unless the
/// prepass ran (armed) at least once since the last reset, so a dark run adds
/// no flight note.
pub fn take_cgan_child_zono_receipt() -> Option<CganChildZonoReceipt> {
    RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

/// Fold one batch into the receipt.
pub(crate) fn record_batch(
    tally: &CganChildZonoBatchTally,
    plan: &CganChildZonoPlan,
    peak_rss_mb: Option<u64>,
) {
    let mut guard = RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let receipt = guard.get_or_insert_with(CganChildZonoReceipt::default);
    receipt.batches += 1;
    receipt.max_generators = plan.max_generators;
    receipt.peak_rss_mb = match (receipt.peak_rss_mb, peak_rss_mb) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let t = &mut receipt.totals;
    t.children += tally.children;
    t.installed += tally.installed;
    t.merged_into_override += tally.merged_into_override;
    t.refused_unsupported += tally.refused_unsupported;
    t.refused_deadline += tally.refused_deadline;
    t.refused_numeric += tally.refused_numeric;
    t.refused_memory += tally.refused_memory;
    t.refused_other += tally.refused_other;
    t.gens_spent += tally.gens_spent;
    t.gens_peak = t.gens_peak.max(tally.gens_peak);
    t.folded += tally.folded;
    t.unstable_zono += tally.unstable_zono;
    t.unstable_before += tally.unstable_before;
    t.unstable_after += tally.unstable_after;
    t.child_nanos = t.child_nanos.saturating_add(tally.child_nanos);
    t.cert_charge_max = t.cert_charge_max.max(tally.cert_charge_max);
    t.width_max = t.width_max.max(tally.width_max);
    t.tranches += tally.tranches;
    if tally.engine == ChildZonoEvaluator::Batch {
        t.engine = ChildZonoEvaluator::Batch;
    }
    t.batch_tranches += tally.batch_tranches;
    t.row_tasks += tally.row_tasks;
    t.prepass_nanos = t.prepass_nanos.saturating_add(tally.prepass_nanos);
}

/// Fold one armed fresh-clip call into the receipt.
pub(crate) fn record_fresh_clip(
    had_map: bool,
    refuted: bool,
    applied: bool,
) -> (u64, u64, u64, u64) {
    let mut guard = RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let receipt = guard.get_or_insert_with(CganChildZonoReceipt::default);
    receipt.fresh_clip_calls += 1;
    receipt.fresh_clip_with_map += u64::from(had_map);
    receipt.fresh_clip_refuted_with_map += u64::from(had_map && refuted);
    receipt.fresh_clip_applied_with_map += u64::from(had_map && applied);
    (
        receipt.fresh_clip_calls,
        receipt.fresh_clip_with_map,
        receipt.fresh_clip_refuted_with_map,
        receipt.fresh_clip_applied_with_map,
    )
}

/// `(VmRSS, VmHWM)` of this process in MiB, where the platform exposes them.
#[must_use]
pub fn self_rss_mb() -> (Option<u64>, Option<u64>) {
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
