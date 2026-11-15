// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-zono` (V2, `bab.alpha_crown.cgan_child_zono_refresh`,
//! default-dark): certified per-child DeepZ zonotope intermediates for the
//! input-split deferred rebound.
//!
//! # What it changes
//!
//! Today every deferred cGAN child is rebounded against the FROZEN root
//! intermediate map intersected with fresh child IBP, and IBP is vacuous in
//! the discriminator (BN_24's upper bound reaches 2030). Vibecheck re-derives
//! every intermediate on every leaf with a batched DeepZ zonotope; the Z1 gate
//! replicated that in certified f64 and closed row 7 in 15 leaves where ny's
//! frozen families close none of the same 35 frontier boxes.
//!
//! Armed, this prepass runs BEFORE the rebound partition, for every deferred
//! child:
//!
//! 1. compile the graph once per batch into a [`ZonoProgram`] (sequential
//!    Gemm/Reshape/ConvTranspose/BN/Conv/ReLU chain; anything else refuses and
//!    the whole batch stays on today's path);
//! 2. run the certified zonotope over the child's own box, intersecting every
//!    ReLU's pre-activation box with the INHERITED map (root CROWN map, then
//!    the domain's own override) before DeepZ picks its slope;
//! 3. publish the resulting pre-activation boxes. A domain that already
//!    carries an override (complete clip, `#cgan-child-interm-refine`) gets
//!    them merged into it shrink-only; every other domain gets them as its
//!    OWN base map in the batched dense-spec rebound (root map merged with
//!    the zonotope entries, shrink-only), which the batched kernel's per-child
//!    IBP refresh then intersects with fresh IBP exactly as before;
//! 4. the refreshed entries stay on the domain for the fresh exact-domain
//!    clip of the same iteration, which intersects them with the root map and
//!    fresh IBP.
//!
//! # Soundness
//!
//! The child box is a sub-box of the root box, so the root map, any override
//! (by its own contract) and the child's zonotope all enclose the child's
//! reachable set, and so does any intersection of them. DeepZ's identity holds
//! for any valid enclosure. Every merge is `merge_reference_bound_maps`:
//! shrink-only, a shape mismatch keeps the current entry, a disjoint element
//! keeps the current element. Any refusal, NaN, memory decline or deadline
//! leaves that child exactly on its historical path.
//!
//! # Memory (no governor without a cgroup — CLAUDE.md)
//!
//! Each child's peak is estimated BEFORE it runs from the program's shapes,
//! the inherited maps' unstable counts (a sound bound on its fresh symbols)
//! and the generator budget. A tranche is admitted only while the summed
//! estimates fit `min(headroom / 2, 48 GiB)`, where the headroom is
//! `crown_memory::process_memory_headroom_bytes` (MemAvailable-derived when
//! there is no cgroup, i.e. the competition configuration). A child whose own
//! estimate does not fit is refused (`refused_memory`), never run. An
//! unobservable headroom runs serially under a 4 GiB per-child cap.
//!
//! # Determinism
//!
//! Every child is a pure function of its snapshotted box and inherited maps;
//! all GEMMs run `Par::Seq`; results are applied strictly in deferred-index
//! order. Parallel tranches therefore publish maps bit-identical to the
//! serial schedule (`child_zono_parallel_bit_identical_to_serial`).
//!
//! # #m8-cgan-leaf-batch: the batch evaluator (keyless, production default)
//!
//! Memory admission holds a tranche at 9-10 children on 20 threads, so the
//! serial evaluator leaves half the pool idle. [`ChildZonoEvaluator::Batch`]
//! runs each tranche through `ZonoProgram::forward_batch`: the children
//! concurrently AND every `K`-row pass of each child as the reference's own
//! row blocks on the rayon pool. It is bitwise the serial evaluator
//! (`child_zono_batch_evaluator_is_bit_identical_to_serial`), so it needs no
//! key and changes no verdict-bearing value. A batched tranche reserves the
//! pool-wide row-task scratch (`ZonoProgram::row_task_scratch_bytes`) ONCE on
//! top of its children's estimates; a tranche that cannot afford it runs on
//! the serial evaluator, and an unobservable headroom keeps the serial
//! evaluator with sequential rows (today's schedule exactly).
//!
//! ROUTER (R1 pending: `CapabilityDecl` rows are declared as data here and in
//! the registry `gaps`, wired when R1 lands):
//!
//! | id | kind | admission | progress / unit | safe points | resources | stop reasons | knobs |
//! |---|---|---|---|---|---|---|---|
//! | `cgan-leaf-batch` | default evaluator of `cgan-child-zono` (keyless; no slot) | `run_child_zono_prepass_scheduled`: headroom observable and the tranche's first child fits with `ZonoProgram::row_task_scratch_bytes` | Children / Tranche | BetweenUnits (tranche boundary; children applied in deferred-index order) | all threads; memory = sum of child `estimate_peak_bytes` + the reserve, within `min(headroom / 2, 48 GiB)` | deadline, memory, non_finite, unsupported, shape | none: width and reserve are derived |

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use ny_tensor::BoundedTensor;

use super::shared::ChildIntermRefineDomain;
use crate::beta_crown::config::BetaCrownConfig;
use crate::zono_core::cgan::{self, CganChildZonoBatchTally, CganChildZonoPlan};
pub(super) use crate::zono_core::ChildZonoEvaluator;
use crate::zono_core::{
    BatchNormCoefficients, RowPar, ZonoBatchChild, ZonoForwardOptions, ZonoForwardOutcome,
    ZonoProgram, ZonoRefusal,
};
use crate::GraphNetwork;

/// A parallel tranche may spend at most `headroom / CHILD_ZONO_HEADROOM_DIVISOR`.
const CHILD_ZONO_HEADROOM_DIVISOR: u64 = 2;
/// Absolute ceiling on one tranche's estimated bytes, whatever the host says.
/// Unified-memory hosts share this with the GPU.
const CHILD_ZONO_ABSOLUTE_CAP_BYTES: u64 = 48 << 30;
/// Per-child ceiling when the headroom cannot be observed (serial only).
const CHILD_ZONO_UNOBSERVED_CHILD_CAP_BYTES: u64 = 4 << 30;
/// Hard ceiling on the width of one tranche.
const CHILD_ZONO_MAX_WIDTH: usize = 16;

/// The zonotope entries one armed prepass produced, by deferred-domain index.
#[derive(Debug, Default)]
pub(super) struct ChildZonoBatch {
    entries: HashMap<usize, Arc<HashMap<String, BoundedTensor>>>,
    tally: CganChildZonoBatchTally,
}

impl ChildZonoBatch {
    /// What the prepass did in this batch.
    pub(super) fn tally(&self) -> &CganChildZonoBatchTally {
        &self.tally
    }

    /// The published pre-activation entries for domain `idx`, if installed.
    pub(super) fn entries_for(&self, idx: usize) -> Option<&Arc<HashMap<String, BoundedTensor>>> {
        self.entries.get(&idx)
    }

    /// Per-domain replacements for the batched rebound's shared root map:
    /// for each ReLU-input node the zonotope published, the ROOT entry merged
    /// shrink-only with it (`merge_reference_bound_maps`: a shape mismatch
    /// keeps the root entry, a disjoint element keeps the root element).
    /// `None` for a domain without entries or whose merge failed — it keeps
    /// the shared root map. `None` overall when there is no root map or no
    /// domain has entries: the caller then takes the historical shared-map
    /// call unchanged.
    pub(super) fn domain_overrides_for(
        &self,
        indices: &[usize],
        root: Option<&HashMap<String, BoundedTensor>>,
    ) -> Option<Vec<Option<HashMap<String, BoundedTensor>>>> {
        let root = root?;
        if !indices.iter().any(|idx| self.entries.contains_key(idx)) {
            return None;
        }
        Some(
            indices
                .iter()
                .map(|idx| {
                    let entries = self.entries.get(idx)?;
                    let root_subset: HashMap<String, BoundedTensor> = entries
                        .keys()
                        .filter_map(|name| root.get(name).map(|b| (name.clone(), b.clone())))
                        .collect();
                    match crate::network::merge_reference_bound_maps(
                        Some(&root_subset),
                        Some(entries.as_ref()),
                    ) {
                        Ok(Some(map)) => Some(map),
                        Ok(None) | Err(_) => None,
                    }
                })
                .collect(),
        )
    }
}

/// One deferred child's inputs, snapshotted before any zonotope runs.
struct ChildZonoJob {
    idx: usize,
    child_box: Arc<BoundedTensor>,
    existing: Option<Arc<HashMap<String, BoundedTensor>>>,
    estimate_bytes: u64,
}

fn inherited_maps<'a>(
    root: Option<&'a HashMap<String, BoundedTensor>>,
    existing: Option<&'a HashMap<String, BoundedTensor>>,
) -> Vec<&'a HashMap<String, BoundedTensor>> {
    root.into_iter().chain(existing).collect()
}

/// `min(caller deadline, started + max_secs)`: can only bring it nearer.
fn child_deadline(
    deadline: Option<Instant>,
    max_secs: Option<f64>,
    started: Instant,
) -> Option<Instant> {
    let cap = max_secs
        .and_then(|secs| Duration::try_from_secs_f64(secs).ok())
        .and_then(|budget| started.checked_add(budget));
    match (deadline, cap) {
        (Some(d), Some(c)) => Some(d.min(c)),
        (d, c) => d.or(c),
    }
}

/// The armed prepass with the production schedule. `None` when the key is
/// dark: no work, no output.
pub(super) fn run_child_zono_prepass<D: ChildIntermRefineDomain>(
    config: &BetaCrownConfig,
    graph: &GraphNetwork,
    root: Option<&HashMap<String, BoundedTensor>>,
    deadline: Option<Instant>,
    deferred_indices: &[usize],
    domains: &mut [D],
) -> Option<ChildZonoBatch> {
    let plan = cgan::cgan_child_zono_plan(config)?;
    let width = rayon::current_num_threads().clamp(1, CHILD_ZONO_MAX_WIDTH);
    run_child_zono_prepass_scheduled(
        &plan,
        graph,
        root,
        deadline,
        deferred_indices,
        domains,
        width,
        crate::network::crown_memory::process_memory_headroom_bytes,
        PRODUCTION_CHILD_ZONO_EVALUATOR,
    )
}

/// The evaluator the production prepass runs (#m8-cgan-leaf-batch): the
/// batch evaluator, bitwise identical to the serial one.
pub(super) const PRODUCTION_CHILD_ZONO_EVALUATOR: ChildZonoEvaluator = ChildZonoEvaluator::Batch;

/// The armed prepass under an explicit schedule (width ceiling, headroom
/// probe and evaluator), so the serial/parallel and serial/batch bit-identity
/// tests can pin every arm.
///
/// Returns `None` when the program itself is refused (every child then keeps
/// today's path exactly); otherwise the entries of every installed child.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_child_zono_prepass_scheduled<D: ChildIntermRefineDomain>(
    plan: &CganChildZonoPlan,
    graph: &GraphNetwork,
    root: Option<&HashMap<String, BoundedTensor>>,
    deadline: Option<Instant>,
    deferred_indices: &[usize],
    domains: &mut [D],
    width_ceiling: usize,
    headroom_probe: fn() -> Option<u64>,
    evaluator: ChildZonoEvaluator,
) -> Option<ChildZonoBatch> {
    let prepass_started = Instant::now();
    announce_armed(plan, width_ceiling);
    let mut tally = CganChildZonoBatchTally::default();
    let mut jobs: Vec<ChildZonoJob> = deferred_indices
        .iter()
        .filter_map(|&idx| {
            let domain = domains.get_mut(idx)?;
            Some(ChildZonoJob {
                idx,
                child_box: Arc::clone(domain.refine_input_bounds()),
                existing: domain.refine_override_slot().clone(),
                estimate_bytes: 0,
            })
        })
        .collect();
    tally.children = jobs.len();
    let Some(first) = jobs.first() else {
        tally.prepass_nanos = nanos_since(prepass_started);
        report_batch(plan, &tally);
        return Some(ChildZonoBatch {
            entries: HashMap::new(),
            tally,
        });
    };
    let program = match ZonoProgram::compile_with_ibp_shapes_and_deadline(
        graph,
        first.child_box.as_ref(),
        deadline,
    ) {
        Ok(program) => {
            announce_program_batch_norm(&program);
            program
        }
        Err(refusal) => {
            for _ in &jobs {
                tally.record_refusal(&refusal);
            }
            announce_program_refusal(&refusal);
            tally.prepass_nanos = nanos_since(prepass_started);
            report_batch(plan, &tally);
            return None;
        }
    };
    // The serial evaluator's per-child options; `forward_batch` runs the
    // same options with `RowPar::Rayon`.
    let opts = ZonoForwardOptions {
        max_generators: plan.max_generators,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    // #m8-cgan-leaf-batch: the pool-wide row-task scratch a batched tranche
    // reserves once (policy bound, never a soundness input).
    let row_reserve = program.row_task_scratch_bytes(rayon::current_num_threads());
    for job in &mut jobs {
        let inherited = inherited_maps(root, job.existing.as_deref());
        let unstable = program.unstable_upper_bounds(&inherited);
        let input_symbols = job
            .child_box
            .lower()
            .iter()
            .zip(job.child_box.upper().iter())
            .filter(|(lo, hi)| hi > lo)
            .count();
        job.estimate_bytes =
            program.estimate_peak_bytes(input_symbols, plan.max_generators, &unstable);
    }

    let run = |job: &ChildZonoJob| -> (Result<ZonoForwardOutcome, ZonoRefusal>, u64) {
        let started = Instant::now();
        let inherited = inherited_maps(root, job.existing.as_deref());
        let result = program.forward(
            job.child_box.as_ref(),
            &opts,
            &inherited,
            child_deadline(deadline, plan.max_secs, started),
            None,
        );
        (
            result,
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        )
    };

    let mut outcomes: Vec<Option<(Result<ZonoForwardOutcome, ZonoRefusal>, u64)>> =
        (0..jobs.len()).map(|_| None).collect();
    let mut cursor = 0usize;
    while cursor < jobs.len() {
        let headroom = headroom_probe();
        let (budget, width_cap) = match headroom {
            Some(bytes) => (
                (bytes / CHILD_ZONO_HEADROOM_DIVISOR).min(CHILD_ZONO_ABSOLUTE_CAP_BYTES),
                width_ceiling.clamp(1, CHILD_ZONO_MAX_WIDTH),
            ),
            None => (CHILD_ZONO_UNOBSERVED_CHILD_CAP_BYTES, 1),
        };
        if jobs[cursor].estimate_bytes > budget {
            outcomes[cursor] = Some((
                Err(ZonoRefusal::Memory {
                    estimate_bytes: jobs[cursor].estimate_bytes,
                    budget_bytes: budget,
                }),
                0,
            ));
            cursor += 1;
            continue;
        }
        // Batch only when the headroom is observable AND the tranche's first
        // child fits with the row-task reserve; otherwise today's schedule.
        let batched = evaluator == ChildZonoEvaluator::Batch
            && headroom.is_some()
            && jobs[cursor]
                .estimate_bytes
                .checked_add(row_reserve)
                .is_some_and(|total| total <= budget);
        let mut end = cursor + 1;
        let mut spent = jobs[cursor].estimate_bytes + if batched { row_reserve } else { 0 };
        while end < jobs.len() && end - cursor < width_cap {
            match spent.checked_add(jobs[end].estimate_bytes) {
                Some(total) if total <= budget => {
                    spent = total;
                    end += 1;
                }
                _ => break,
            }
        }
        let tranche = &jobs[cursor..end];
        let results: Vec<(Result<ZonoForwardOutcome, ZonoRefusal>, u64)> = if batched {
            let started = Instant::now();
            let children: Vec<ZonoBatchChild<'_>> = tranche
                .iter()
                .map(|job| ZonoBatchChild {
                    input: job.child_box.as_ref(),
                    inherited: inherited_maps(root, job.existing.as_deref()),
                    deadline: child_deadline(deadline, plan.max_secs, started),
                })
                .collect();
            tally.batch_tranches += 1;
            tally.engine = ChildZonoEvaluator::Batch;
            program.forward_batch(&children, &opts)
        } else if tranche.len() > 1 {
            use rayon::prelude::*;
            tranche
                .par_iter()
                .map(|job| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    run(job)
                })
                .collect()
        } else {
            tranche.iter().map(&run).collect()
        };
        for (offset, result) in results.into_iter().enumerate() {
            outcomes[cursor + offset] = Some(result);
        }
        tally.tranches += 1;
        tally.width_max = tally.width_max.max(tranche.len());
        cursor = end;
    }

    let mut batch = ChildZonoBatch::default();
    for (job, outcome) in jobs.iter().zip(outcomes) {
        let Some((result, nanos)) = outcome else {
            continue;
        };
        tally.child_nanos = tally.child_nanos.saturating_add(nanos);
        match result {
            Ok(outcome) => apply_outcome(domains, job, outcome, &mut tally, &mut batch),
            Err(refusal) => tally.record_refusal(&refusal),
        }
    }
    tally.prepass_nanos = nanos_since(prepass_started);
    report_batch(plan, &tally);
    batch.tally = tally;
    Some(batch)
}

fn nanos_since(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Install one child's result: shrink-only into an existing override, and
/// record the entries for the batched rebound and the fresh clip.
fn apply_outcome<D: ChildIntermRefineDomain>(
    domains: &mut [D],
    job: &ChildZonoJob,
    outcome: ZonoForwardOutcome,
    tally: &mut CganChildZonoBatchTally,
    batch: &mut ChildZonoBatch,
) {
    if outcome.published.is_empty() {
        tally.refused_other += 1;
        return;
    }
    if let Some(existing) = job.existing.as_deref() {
        let merged = match crate::network::merge_reference_bound_maps(
            Some(existing),
            Some(&outcome.published),
        ) {
            Ok(Some(merged)) => merged,
            Ok(None) | Err(_) => {
                tally.refused_other += 1;
                return;
            }
        };
        let Some(domain) = domains.get_mut(job.idx) else {
            tally.refused_other += 1;
            return;
        };
        *domain.refine_override_slot() = Some(Arc::new(merged));
        tally.merged_into_override += 1;
    }
    tally.installed += 1;
    tally.row_tasks += outcome.stats.row_tasks;
    tally.gens_spent += outcome.stats.gens_final;
    tally.gens_peak = tally.gens_peak.max(outcome.stats.gens_peak);
    tally.folded += outcome.stats.folded;
    tally.cert_charge_max = tally.cert_charge_max.max(outcome.stats.max_remainder);
    for relu in &outcome.relus {
        tally.unstable_zono += relu.zono_unstable;
        if let Some(before) = relu.inherited_unstable {
            tally.unstable_before += before;
            tally.unstable_after += relu.unstable;
        }
    }
    batch.entries.insert(job.idx, Arc::new(outcome.published));
}

static ARMED: Once = Once::new();
static PROGRAM_REFUSED: Once = Once::new();
static PROGRAM_BATCH_NORM: Once = Once::new();
static BATCHES: AtomicU64 = AtomicU64::new(0);
static FRESH_CLIP_CALLS_PRINTED: AtomicU64 = AtomicU64::new(0);

fn announce_armed(plan: &CganChildZonoPlan, width_ceiling: usize) {
    ARMED.call_once(|| {
        eprintln!(
            "[cgan-child-zono] armed=1 max_generators={} max_secs={} width_ceiling={} \
             source={} key=bab.alpha_crown.cgan_child_zono_refresh \
             route=input-split-deferred-rebound-prepass",
            plan.max_generators,
            plan.max_secs
                .map_or_else(|| "none".to_string(), |secs| format!("{secs}")),
            width_ceiling,
            plan.source.as_str(),
        );
    });
}

/// #zono-exact-bn observable, once per process: how many compiled BatchNorm
/// steps charge the binary64 enclosure of their exact-real coefficients
/// (`bn_exact_f64`) vs the binary32 affine (`bn_binary32`). Every
/// ONNX-converted BatchNorm is exact; a nonzero `bn_binary32` on a scored
/// cGAN means the tight charge did not arrive.
fn announce_program_batch_norm(program: &ZonoProgram) {
    PROGRAM_BATCH_NORM.call_once(|| {
        let steps = program.batch_norm_coefficients();
        let exact = steps
            .iter()
            .filter(|(_, source)| *source == BatchNormCoefficients::ExactF64)
            .count();
        eprintln!(
            "[cgan-child-zono] program=compiled steps={} bn_exact_f64={exact} bn_binary32={}",
            program.steps().len(),
            steps.len() - exact
        );
    });
}

fn announce_program_refusal(refusal: &ZonoRefusal) {
    PROGRAM_REFUSED.call_once(|| {
        eprintln!(
            "[cgan-child-zono] program=refused reason={} detail={refusal}",
            refusal.token()
        );
    });
}

/// Fold the batch into the run receipt and print the fixed-schema line: every
/// batch up to 16, every 16th after, and any batch with a refusal or nothing
/// installed up to batch 256.
fn report_batch(plan: &CganChildZonoPlan, tally: &CganChildZonoBatchTally) {
    let batch = BATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    let (_, peak_rss_mb) = cgan::self_rss_mb();
    cgan::record_batch(tally, plan, peak_rss_mb);
    let anomalous = tally.installed == 0 || tally.refused() > 0;
    if batch <= 16 || batch.is_multiple_of(16) || (anomalous && batch <= 256) {
        eprintln!("{}", tally.render(batch, plan, peak_rss_mb));
    }
}

/// Armed fresh-clip engagement: folds into the receipt and prints cumulative
/// counts for the first four calls and every 32nd after.
pub(super) fn observe_fresh_clip(had_map: bool, refuted: bool, applied: bool) {
    let (calls, with_map, refuted_with_map, applied_with_map) =
        cgan::record_fresh_clip(had_map, refuted, applied);
    let printed = FRESH_CLIP_CALLS_PRINTED.fetch_add(1, Ordering::Relaxed) + 1;
    if printed <= 4 || printed.is_multiple_of(32) {
        eprintln!(
            "[cgan-child-zono] fresh_clip calls={calls} with_map={with_map} \
             refuted_with_map={refuted_with_map} applied_with_map={applied_with_map}"
        );
    }
}

#[cfg(test)]
#[path = "child_zono_tests.rs"]
mod tests;
