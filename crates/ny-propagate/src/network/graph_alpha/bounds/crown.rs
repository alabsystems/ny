// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::crown_tighten::{crown_ibp_downstream_resweep_enabled, CrownIbpCollectionMode};
use super::demand::sparse_relu_rows_enabled;
use super::*;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};
use ny_core::GemmEngine;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(test)]
use super::demand::{sparse_relu_rows_from_raw, SPARSE_RELU_ROWS_ENV};
#[cfg(test)]
use crate::bounds::LinearBounds;

#[cfg(test)]
thread_local! {
    static CGAN_ATOMIC_COLLECTION_ENTRIES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static CGAN_COMPLETE_COLLECTION_ENTRIES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only observation scope for complete typed cGAN collection entries.
#[cfg(test)]
pub(crate) struct CganCompleteCollectionEntryCounter {
    previous: Option<usize>,
}

#[cfg(test)]
impl CganCompleteCollectionEntryCounter {
    pub(crate) fn start() -> Self {
        let previous = CGAN_COMPLETE_COLLECTION_ENTRIES.with(|slot| slot.replace(Some(0)));
        Self { previous }
    }

    pub(crate) fn entries(&self) -> usize {
        CGAN_COMPLETE_COLLECTION_ENTRIES.with(|slot| {
            slot.get()
                .expect("cGAN complete collection counter scope must still be active")
        })
    }
}

#[cfg(test)]
impl Drop for CganCompleteCollectionEntryCounter {
    fn drop(&mut self) {
        CGAN_COMPLETE_COLLECTION_ENTRIES.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
fn record_cgan_complete_collection_entry() {
    CGAN_COMPLETE_COLLECTION_ENTRIES.with(|slot| {
        if let Some(entries) = slot.get() {
            slot.set(Some(entries.saturating_add(1)));
        }
    });
}

/// Test-only, thread-local observation scope for full typed-collector entries.
///
/// This deliberately counts the outer options entry rather than target
/// backward calls, so a regression can distinguish one transaction from a
/// duplicate AnalyticChain recollection without process-global synchronization.
#[cfg(test)]
pub(crate) struct CganAtomicCollectionEntryCounter {
    previous: Option<usize>,
}

#[cfg(test)]
impl CganAtomicCollectionEntryCounter {
    pub(crate) fn start() -> Self {
        let previous = CGAN_ATOMIC_COLLECTION_ENTRIES.with(|slot| slot.replace(Some(0)));
        Self { previous }
    }

    pub(crate) fn entries(&self) -> usize {
        CGAN_ATOMIC_COLLECTION_ENTRIES.with(|slot| {
            slot.get()
                .expect("cGAN atomic collection counter scope must still be active")
        })
    }
}

#[cfg(test)]
impl Drop for CganAtomicCollectionEntryCounter {
    fn drop(&mut self) {
        CGAN_ATOMIC_COLLECTION_ENTRIES.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
fn record_cgan_atomic_collection_entry() {
    CGAN_ATOMIC_COLLECTION_ENTRIES.with(|slot| {
        if let Some(entries) = slot.get() {
            slot.set(Some(entries.saturating_add(1)));
        }
    });
}

/// Default-dark cGAN lane: permit a deadline-truncated collection to satisfy a
/// later, exactly matching collection request.
///
/// Unset, `0`, and every value except the exact string `1` preserve the
/// historical complete-only lookup. `NY_DISABLE_CROWN_COLLECTION_CACHE=1`
/// remains the stronger kill switch and bypasses both lookup and store.
const SERVE_TRUNCATED_COLLECTION_CACHE_ENV: &str = "NY_CROWN_SERVE_TRUNCATED_CACHE";

fn serve_truncated_collection_cache_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn serve_truncated_collection_cache_enabled() -> bool {
    serve_truncated_collection_cache_from_raw(
        std::env::var(SERVE_TRUNCATED_COLLECTION_CACHE_ENV)
            .ok()
            .as_deref(),
    )
}

/// #cgan-truncated-serve-telemetry: tagged stderr breadcrumbs for the
/// default-dark truncated-reuse lane.
///
/// WHY stderr and not `debug!`: every wrong inference in the collection-cache
/// investigation came from log ABSENCE under a misconfigured filter
/// (`docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md`, "the logging mechanism
/// was the whole problem"). Engagement telemetry for a dark lever must survive
/// the vnncomp log filter, or a null measurement is vacuous (measurement
/// parity R7/R9). Emits nothing unless `NY_CROWN_SERVE_TRUNCATED_CACHE=1`, so
/// a default run's stderr is byte-identical; production callers of the scope
/// builder and the truncated lookup branch are additionally unreachable with
/// the gate off. Rate-limited (first 64 events, then powers of two) because
/// per-child input-split collections can reach the store path thousands of
/// times per run.
fn truncated_lane_event(stage: &str, detail: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    if !serve_truncated_collection_cache_enabled() {
        return;
    }
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    let n = EVENTS.fetch_add(1, Ordering::Relaxed).saturating_add(1);
    if n <= 64 || n.is_power_of_two() {
        eprintln!("[NY_CROWN_TRUNCATED_SERVE_V1] stage={stage} n={n} {detail}");
    }
}

fn crown_ibp_tightening_deadline(
    now: Instant,
    outer_deadline: Option<Instant>,
    tightening_cap: Option<Duration>,
) -> Option<Instant> {
    let Some(cap) = tightening_cap else {
        return outer_deadline;
    };
    let local_deadline = now.checked_add(cap).unwrap_or(now);
    Some(outer_deadline.map_or(local_deadline, |outer| outer.min(local_deadline)))
}

/// Ignore only sub-millisecond `Instant` capture jitter when comparing the
/// producer's tightening authority with a prospective consumer's. A genuinely
/// longer-budget request must recompute rather than inherit short-run quality.
const TRUNCATED_REUSE_BUDGET_JITTER: Duration = Duration::from_millis(1);

fn truncated_scope_can_serve(
    producer: &crate::network::core::graph::CrownIbpTruncatedReuseScope,
    consumer: &crate::network::core::graph::CrownIbpTruncatedReuseScope,
) -> bool {
    producer.policy == consumer.policy
        && consumer.tightening_budget
            <= producer
                .tightening_budget
                .saturating_add(TRUNCATED_REUSE_BUDGET_JITTER)
}

/// Options for CROWN-IBP bounds collection on a DAG graph.
///
/// Consolidates the optional parameters (engine, deadline, precomputed IBP,
/// width threshold) into a single struct. This replaces the combinatorial
/// explosion of 10+ convenience wrapper methods with a single options-based
/// entry point ([`GraphNetwork::collect_crown_ibp_bounds_dag_with_options`]).
///
/// Part of #3812 (API explosion cleanup).
#[derive(Default)]
pub(crate) struct CrownIbpCollectOptions<'a> {
    /// GPU/accelerated GEMM engine for per-node CROWN backward passes (#3549).
    pub engine: Option<&'a dyn GemmEngine>,
    /// Deadline for the CROWN-IBP collection. Remaining nodes fall back to IBP
    /// bounds when exceeded (#3109).
    pub deadline: Option<Instant>,
    /// Apply `deadline` to the mandatory foundational IBP pass as well as the
    /// CROWN tightening sweep. The historical public collectors leave this
    /// false so an expired tightening budget can still return a complete,
    /// plain-IBP-backed map. Full-phase verifier callers set it when returning
    /// a timely Timeout verdict is more important than completing that map.
    pub deadline_includes_ibp: bool,
    /// Pre-computed IBP bounds (#3596). When provided, skips the internal IBP
    /// forward pass, allowing the full deadline budget for CROWN tightening.
    /// On deep graphs (e.g., ECAPA-TDNN ~188 nodes), the IBP forward pass can
    /// consume the entire deadline budget.
    pub precomputed_ibp: Option<HashMap<String, BoundedTensor>>,
    /// Minimum IBP width to trigger CROWN tightening at a node (#3499). Nodes
    /// with IBP `max_width()` below this threshold are skipped, saving budget
    /// for deeper layers where CROWN tightening is most impactful.
    pub min_width_to_tighten: Option<f32>,
    /// Caller-local cap on only the CROWN tightening sweep, started after the
    /// mandatory IBP map is available. The foundational IBP pass keeps its
    /// historical behavior, and the caller's outer deadline value is never
    /// mutated.
    pub tightening_cap: Option<Duration>,
    /// Allow a safe, instant complete-map cache hit, but never serve a
    /// deadline-truncated entry and never store/merge a fresh capped result.
    ///
    /// This mode preserves deduplication when an earlier phase already
    /// completed the same map without letting a partial map produced under a
    /// different time allocation contaminate the experiment.
    pub complete_cache_lookup_only: bool,
    /// Typed root collection strategy. The default preserves the ordinary
    /// all-demanded collector.
    pub collection_mode: CrownIbpCollectionMode,
    /// Row ceiling for demanded CROWN targets (`#cgan-narrow-target-budget`).
    ///
    /// `None` (default) keeps the historical all-demanded candidate set. When
    /// set, a target whose objective-row count exceeds the ceiling is declined
    /// as a tightening candidate and keeps the caller-supplied baseline
    /// enclosure, so the collection budget is spent on the narrow targets
    /// instead of being split in proportion to row count.
    ///
    /// Like `min_width_to_tighten` this changes WHICH nodes are tightened, so
    /// it suppresses complete-map caching for the same reason.
    pub cgan_narrow_target_max_rows: Option<usize>,
}

impl GraphNetwork {
    /// Collect CROWN-IBP bounds with full options control.
    ///
    /// This is the single entry point that all convenience wrappers delegate to.
    /// Returns per-node bounds plus fallback provenance metadata.
    ///
    /// See [`CrownIbpCollectOptions`] for parameter documentation.
    ///
    /// # Input-keyed collection cache (#cgan-collection-cache)
    ///
    /// SOUNDNESS INVARIANT: a CROWN/IBP node-bounds map computed for input box
    /// `B` on network `N` is a set of valid enclosures for `(N, B)` FOREVER —
    /// reuse is sound iff the network is the same object/weights (enforced by
    /// the per-object cache + the pure-clone adoption contract) and the input
    /// box is BIT-EXACT identical (enforced by hashing the f32 bit patterns of
    /// the flattened lower/upper arrays). BaB children split the box, so they
    /// can never hit the root's entry.
    ///
    /// A complete (untruncated) map is served to every later same-key call
    /// regardless of that call's remaining budget — this is the whole point
    /// (on cgan_2023 the alpha warmup re-ran the collection up to 5x and the
    /// last, most-truncated map reached BaB instead of the complete precheck
    /// map). The deadline therefore is not part of the ordinary fingerprint.
    /// The opt-in truncated lane is stricter: its separate scope records the
    /// producer's tightening budget and serves only a same-policy consumer
    /// whose available budget is no larger; a more-authoritative consumer
    /// misses and recomputes. Requests whose OPTIONS change what would be
    /// computed (caller-supplied IBP maps, width-threshold skips) bypass the
    /// cache entirely — every duplicated producer in the verify flow
    /// (disjunctive precheck, alpha-warmup reference bounds, iteration-0 output
    /// backward, sequential-alpha re-reference) uses default options.
    pub(crate) fn collect_crown_ibp_bounds_dag_with_options(
        &self,
        input: &BoundedTensor,
        options: CrownIbpCollectOptions<'_>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        #[cfg(test)]
        if options.collection_mode == CrownIbpCollectionMode::CganSparseTargetComplete {
            record_cgan_atomic_collection_entry();
        }
        #[cfg(test)]
        if options.collection_mode == CrownIbpCollectionMode::CganComplete {
            record_cgan_complete_collection_entry();
        }

        // Snapshot the collector gates that participate in complete-cache or
        // provenance identity. The same values identify any cache entry and
        // drive the fresh collection, so an environment mutation cannot make
        // lookup/store policy disagree with the target walks performed by this
        // call.
        let downstream_resweep = crown_ibp_downstream_resweep_enabled();
        let deadline_salvage_policy =
            target_backward::PartialCrownDeadlineSalvagePolicy::from_environment();
        let prefix_cost_admission_enabled = budget_policy::prefix_cost_admission_enabled();
        let runtime_policy = self.crown_ibp_runtime_policy();
        let cut_segment = target_backward::crown_cut_segment_from_env();
        let published_margin_indices = crate::output_margin_seed::published_margin_indices();
        let tightening_cap = options.tightening_cap;
        let complete_cache_lookup_only =
            options.complete_cache_lookup_only || tightening_cap.is_some();
        if let Some(cap) = tightening_cap {
            let now = Instant::now();
            let outer_remaining_secs = options
                .deadline
                .map(|outer| outer.saturating_duration_since(now).as_secs_f64())
                .unwrap_or(f64::INFINITY);
            eprintln!(
                "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=request requested_secs={} \
                 outer_remaining_secs={outer_remaining_secs:.3}",
                cap.as_secs(),
            );
        }
        // #lsnc-determinism-diagnostic (task #36): default-OFF kill-switch to
        // physically bypass the input-keyed CROWN-IBP collection cache (both
        // lookup and store/merge). Set `NY_DISABLE_CROWN_COLLECTION_CACHE=1` to
        // reduce the pipeline to the pre-#cgan-collection-cache path (every
        // duplicated producer recomputes; nothing is served or merged). Sparse
        // ReLU-row collection also bypasses this cache because its objective
        // subset is not represented in the cache fingerprint. Both gates are
        // default-off, preserving the historical cached path byte-for-byte.
        // Read once per call; the collection is not a hot loop.
        //
        // #cgan-restart-root-collection-cache (2026-08-11): non-Standard
        // (typed cGAN) modes used to be a THIRD bypass term here. That made
        // the typed root collection uncacheable end-to-end: on cgan row 7 the
        // initial phase's ~270 s CganComplete collection was discarded and the
        // disjunctive restart lane — whose call-local restart cache is created
        // fresh AFTER that phase and therefore always cold-misses — re-collected
        // the bit-identical root box under its clipped budget. The mode is now
        // part of the cache FINGERPRINT instead (see `cache_tag`), which keeps
        // the promised cache separation (entries never serve across modes)
        // while letting same-mode same-box consumers reuse the map. Serving is
        // sound: exact key+fingerprint match, and the serve path intersects
        // the caller's own precomputed IBP map (#cgan-precomputed-ibp-cache).
        let cache_disabled = std::env::var_os("NY_DISABLE_CROWN_COLLECTION_CACHE")
            .is_some_and(|v| v == "1")
            || sparse_relu_rows_enabled();
        // #cgan-precomputed-ibp-cache (2026-08-03): `precomputed_ibp` used to
        // force a cache BYPASS (lookup and store). MEASURED on cgan_2023: the
        // root box is collected 7 times, 1 necessary — 403s of an 851s budget
        // is duplicate work and `reusing cached collection` never fires once.
        // Collections 4-6 (276.8s) are attributed to exactly this exclusion.
        //
        // The exclusion was over-conservative. `precomputed_ibp` does not change
        // which CROWN backward runs; it only supplies the IBP side of the
        // CROWN∩IBP intersection. A cached result and the caller's IBP map are
        // both valid enclosures of the same (network, box), so intersecting them
        // at SERVE time is sound — the identical argument the merge path already
        // relies on (see `intersection_per_element` below) — and yields a result
        // at least as tight as the uncached path would have produced.
        //
        // `min_width_to_tighten` stays excluded: it changes WHICH nodes are
        // tightened, so a served map could omit work the caller asked for.
        let cacheable = !cache_disabled
            && options.min_width_to_tighten.is_none()
            && options.cgan_narrow_target_max_rows.is_none();
        // #cgan-truncated-cache is intentionally narrower than the general
        // complete-map cache:
        //   * explicit opt-in;
        //   * ConvTranspose graphs only (the measured cgan surface);
        //   * deadline-bearing calls only (a no-deadline caller is allowed to
        //     finish missing targets and must therefore recompute).
        //
        // The lookup performs the remaining exact scope checks (resource
        // policy, chunk schedule, objective subset, cuts, full map coverage).
        let lookup_now = Instant::now();
        // A capped consumer's true tightening authority is min(outer deadline,
        // cap) — the same clip the producer side applies via
        // `crown_ibp_tightening_deadline`. Using the unclipped remaining time
        // here could only over-state consumer authority and REFUSE more serves,
        // but the symmetric clip keeps producer/consumer budgets comparable.
        let consumer_tightening_budget = options
            .deadline
            .map(|deadline| deadline.saturating_duration_since(lookup_now))
            .map(|remaining| match tightening_cap {
                Some(cap) => remaining.min(cap),
                None => remaining,
            });
        // #cgan-restart-root-collection-cache: the typed cGAN modes always
        // carry a tightening cap, which forces `complete_cache_lookup_only`
        // above — that must not permanently lock them out of the (still
        // explicitly opt-in, exactly scope-matched) truncated-reuse lane, or
        // a truncated initial-phase root map can never reach the restart
        // lane's identical-box consumer. Standard capped callers keep the
        // historical complete-only contract byte-for-byte.
        let truncated_reuse_requested = cacheable
            && (!complete_cache_lookup_only
                || options.collection_mode.requires_shrink_only_publication())
            && serve_truncated_collection_cache_enabled()
            && self.has_conv_transpose2d_layers()
            && consumer_tightening_budget.is_some_and(|budget| !budget.is_zero());
        let truncated_reuse_allowed_now = truncated_reuse_requested
            && options
                .deadline
                .is_some_and(|deadline| Instant::now() < deadline);
        let cache_key = cacheable.then(|| {
            let fingerprint = self.crown_ibp_collection_cache_fingerprint_with_runtime_policy(
                input,
                options.engine.is_some(),
                downstream_resweep,
                deadline_salvage_policy,
                options.collection_mode,
                runtime_policy,
                cut_segment,
                published_margin_indices.as_deref(),
            );
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            hasher.write(&fingerprint);
            (hasher.finish(), fingerprint)
        });
        // #cgan-cache-values: retained at debug level. Every inference drawn
        // from log ABSENCE in this investigation was wrong; this prints the
        // values that settle it. MEASURED on cgan_2023: the SAME key with the
        // SAME fingerprint is looked up 6 times with cacheable=true and still
        // serves nothing, which localises the block to the truncated branch
        // below (scope_match / covers_graph), not to keying.
        {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&cache_key.as_ref().map(|(_, fp)| fp.clone()), &mut h);
            debug!(
                "#cgan-cache-values LOOKUP cacheable={} key={:?} fp_hash={:016x} engine={} \
                 patches={} nodes={}",
                cacheable,
                cache_key.as_ref().map(|(k, _)| format!("{k:016x}")),
                std::hash::Hasher::finish(&h),
                options.engine.is_some(),
                self.use_patches_mode,
                self.nodes.len(),
            );
        }
        if let Some((key, ref fp)) = cache_key {
            // #cgan-engine-agnostic-serve (2026-08-03; removed unannounced by
            // 12ef71f6c and RESTORED 2026-08-27). The fingerprint embeds
            // `engine_present` and the wrappers split on it —
            // `..._with_precomputed_ibp` threads no engine while `..._and_engine`
            // does — so two callers collecting the SAME box on the SAME network
            // never see each other's entry. MEASURED on cgan_2023: stable key,
            // 12 merge events on it, ZERO serves, 403 s of an 851 s budget spent
            // recollecting a bit-identical box.
            //
            // Serving across the engine bit is SOUND: an engine-computed and a
            // CPU-computed collection are both valid enclosures of the same
            // (network, box) — the argument the merge path already relies on —
            // and the served result is intersected with the caller's own IBP map
            // below, so what publishes is at least as tight as either.
            //
            // The bridge spans the backend bit and NOTHING ELSE. Every identity
            // field added by #frozen-crown-authority — the frozen LeverSet
            // digest, the objective publication, the cut segment and the whole
            // runtime policy — is carried unchanged into `alt_fp`, so exactness
            // is relaxed on exactly one axis, deliberately, with a measurement.
            let alt_fp = self.crown_ibp_collection_cache_fingerprint_with_runtime_policy(
                input,
                options.engine.is_none(),
                downstream_resweep,
                deadline_salvage_policy,
                options.collection_mode,
                runtime_policy,
                cut_segment,
                published_margin_indices.as_deref(),
            );
            use std::hash::Hasher;
            let mut alt_hasher = std::collections::hash_map::DefaultHasher::new();
            alt_hasher.write(&alt_fp);
            let alt_key = alt_hasher.finish();
            if let Some(hit) = self
                .crown_ibp_collection_lookup_with_prefix_cost_policy(
                    key,
                    fp,
                    truncated_reuse_allowed_now,
                    consumer_tightening_budget,
                    prefix_cost_admission_enabled,
                    runtime_policy,
                    cut_segment,
                    published_margin_indices.as_ref(),
                )
                .or_else(|| {
                    // Same box, other engine state. Sound per the note above.
                    self.crown_ibp_collection_lookup_with_prefix_cost_policy(
                        alt_key,
                        &alt_fp,
                        truncated_reuse_allowed_now,
                        consumer_tightening_budget,
                        prefix_cost_admission_enabled,
                        runtime_policy,
                        cut_segment,
                        published_margin_indices.as_ref(),
                    )
                    .inspect(|_| {
                        debug!(
                            "CROWN-IBP DAG: served cached collection across the engine bit \
                             for alternate key {alt_key:016x} (#cgan-engine-agnostic-serve)"
                        );
                    })
                })
            {
                if tightening_cap.is_some() {
                    eprintln!(
                        "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=complete-cache-hit \
                         tightening_skipped=true"
                    );
                }
                // Serve-time intersection of the caller's IBP map (see the
                // #cgan-precomputed-ibp-cache note above). Sound: both operands
                // enclose the same (network, box); keeping the tighter side per
                // element is a valid enclosure. Shape drift or a disjoint
                // intersection leaves the cached bound untouched rather than
                // publishing a narrower-than-valid box.
                let hit = match options.precomputed_ibp.as_ref() {
                    Some(ibp) => Self::intersect_served_collection_with_ibp(hit, ibp),
                    None => hit,
                };
                return Ok(hit);
            }
        }

        // #cgan-reclaim CROSS-MODE serve (NY_CGAN_RECLAIM=1, default dark ⇒
        // byte-identical). MEASURED (2026-08-19 st/cg arms): the CLI per-output
        // precheck's ~385 s Standard-mode collection STORES for this exact box,
        // yet the in-verifier CganComplete consumer re-collects for ≤223 s more
        // and `stage=complete-cache-hit` never appears — the fingerprints differ
        // ONLY in the collection-mode tag byte (`cache_tag`: Standard=0 vs
        // CganComplete=1; every other field — box bits, shape, node count,
        // output node, patches mode, resweep/salvage policy — is identical, and
        // the engine bit is already bridged above).
        //
        // The mode tag exists for two reasons, and neither forbids THIS serve:
        // (a) target selection — but `CganComplete` runs "the ordinary
        // all-demanded cascade" (its own doc), the same demanded target set as
        // Standard, and only a COMPLETE entry (no time truncation) is served
        // here, so coverage is the full map either way; (b) shrink-only
        // publication vs the typed consumer's certified forward-linear
        // baseline — enforced AT SERVE TIME below with the bug-19
        // `census_commit::shrink_only_commit` machinery: both operands are
        // independently certified enclosures of the same (graph, box), the
        // elementwise `[max(l), min(u)]` is sound and never wider than the
        // baseline, and disjoint-after-rounding f32 corners fail closed to the
        // served certified value (counted, never manufactured). Truncated
        // entries never serve across modes — their reuse scopes are
        // mode-specific and stay separated.
        if crate::cgan_reclaim::cgan_reclaim_armed()
            && cacheable
            && options.collection_mode != CrownIbpCollectionMode::Standard
            && options.collection_mode.requires_shrink_only_publication()
        {
            if let Some(baseline) = options.precomputed_ibp.as_ref() {
                use std::hash::Hasher;
                let lookup_std = |engine_present: bool| {
                    let std_fp = self.crown_ibp_collection_cache_fingerprint_with_runtime_policy(
                        input,
                        engine_present,
                        downstream_resweep,
                        deadline_salvage_policy,
                        CrownIbpCollectionMode::Standard,
                        runtime_policy,
                        cut_segment,
                        published_margin_indices.as_deref(),
                    );
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    hasher.write(&std_fp);
                    let std_key = hasher.finish();
                    // allow_truncated=false: COMPLETE Standard entries only.
                    self.crown_ibp_collection_lookup_with_prefix_cost_policy(
                        std_key,
                        &std_fp,
                        false,
                        consumer_tightening_budget,
                        prefix_cost_admission_enabled,
                        runtime_policy,
                        cut_segment,
                        published_margin_indices.as_ref(),
                    )
                    .map(|hit| (hit, std_key))
                };
                let served = lookup_std(options.engine.is_some())
                    .or_else(|| lookup_std(options.engine.is_none()));
                match served {
                    Some((mut hit, std_key)) => {
                        if tightening_cap.is_some() {
                            eprintln!(
                                "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=complete-cache-hit \
                                 tightening_skipped=true"
                            );
                        }
                        let stats =
                            census_commit::shrink_only_commit(&mut hit.bounds, baseline, true);
                        census_commit::emit_census_commit(
                            "cgan-reclaim-cross-mode-serve",
                            true,
                            &stats,
                        );
                        eprintln!(
                            "[cgan-reclaim] stage=cross-mode-serve consumer_mode={:?} \
                             std_key={std_key:016x} nodes={} baseline_nodes={} \
                             tightened_elems={} kept_elems={} disjoint_kept={}",
                            options.collection_mode,
                            hit.bounds.len(),
                            baseline.len(),
                            stats.elems_would_loosen,
                            stats.elems_candidate_tighter,
                            stats.elems_disjoint_kept,
                        );
                        return Ok(hit);
                    }
                    None => {
                        // Name WHY the cross-mode serve refused, so the A/B run
                        // can distinguish "no Standard entry" (precheck did not
                        // store / different box) from "entry present but
                        // truncated" (precheck collection was deadline-cut).
                        let std_fp_hash = |engine_present: bool| {
                            let fp = self
                                .crown_ibp_collection_cache_fingerprint_with_runtime_policy(
                                input,
                                engine_present,
                                downstream_resweep,
                                deadline_salvage_policy,
                                CrownIbpCollectionMode::Standard,
                                runtime_policy,
                                cut_segment,
                                published_margin_indices.as_deref(),
                            );
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            hasher.write(&fp);
                            (hasher.finish(), fp)
                        };
                        let candidates = [
                            std_fp_hash(options.engine.is_some()),
                            std_fp_hash(options.engine.is_none()),
                        ];
                        let (present, complete) = self
                            .cached_crown_ibp_collection
                            .slots
                            .read()
                            .ok()
                            .map(|slots| {
                                let entry = slots.iter().find(|e| {
                                    candidates.iter().any(|(key, fp)| {
                                        e.key == *key && e.fingerprint.as_ref() == fp.as_slice()
                                    })
                                });
                                (entry.is_some(), entry.is_some_and(|e| e.complete))
                            })
                            .unwrap_or((false, false));
                        eprintln!(
                            "[cgan-reclaim] stage=cross-mode-miss consumer_mode={:?} \
                             standard_entry_present={present} standard_entry_complete={complete}",
                            options.collection_mode,
                        );
                    }
                }
            }
        }

        let ibp_bounds = match options.precomputed_ibp {
            Some(precomputed) => precomputed,
            None if options.deadline_includes_ibp => self
                .collect_node_bounds_with_engine_and_deadline(
                    input,
                    options.engine,
                    options.deadline,
                )?,
            None => self.collect_node_bounds_with_engine(input, options.engine)?,
        };
        // A hard full-phase authority also covers the boundary between the
        // foundational forward pass and CROWN tightening.  The historical
        // wrappers deliberately exclude that pass from their deadline and may
        // still publish its complete map as an IBP fallback below.
        if options.deadline_includes_ibp
            && options
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(NyError::DeadlineExceeded(
                "CROWN-IBP DAG: deadline exceeded after foundational IBP collection".to_string(),
            ));
        }
        let tightening_start = Instant::now();
        let tightening_deadline =
            crown_ibp_tightening_deadline(tightening_start, options.deadline, tightening_cap);
        // Preserve whether this is caller/cap authority before the collector
        // adds its own aggregate Patches scheduling deadline. Cooperative
        // kernels may poll either kind, but only hard authority may disable an
        // otherwise exact native Patches route up front.
        let tightening_deadline_is_hard = options.deadline.is_some() || tightening_cap.is_some();
        let producer_tightening_budget = tightening_deadline
            .map(|deadline| deadline.saturating_duration_since(tightening_start));
        if let Some(cap) = tightening_cap {
            let effective_remaining_secs = tightening_deadline
                .map(|effective| {
                    effective
                        .saturating_duration_since(tightening_start)
                        .as_secs_f64()
                })
                .unwrap_or(f64::INFINITY);
            let clipped_by_outer = options.deadline.is_some_and(|outer| {
                let local = tightening_start
                    .checked_add(cap)
                    .unwrap_or(tightening_start);
                outer <= local
            });
            eprintln!(
                "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=tightening-start \
                 requested_secs={} effective_remaining_secs={effective_remaining_secs:.3} \
                 clipped_by_outer={clipped_by_outer}",
                cap.as_secs(),
            );
        }
        let fresh = self.collect_crown_ibp_bounds_core_inner_with_mode(
            input,
            ibp_bounds,
            tightening_deadline,
            tightening_deadline_is_hard,
            options.engine,
            options.min_width_to_tighten,
            cut_segment,
            published_margin_indices.clone(),
            downstream_resweep,
            deadline_salvage_policy,
            options.collection_mode,
            options.cgan_narrow_target_max_rows,
            prefix_cost_admission_enabled,
            runtime_policy,
        )?;
        // #cgan-restart-root-collection-cache: typed cGAN producers always run
        // capped (`complete_cache_lookup_only`), so the historical guard meant
        // they could never STORE — the initial phase's root collection was
        // discarded and every later same-box consumer re-collected. Letting
        // them store is monotone-safe: same-key entries MERGE by per-element
        // intersection and the most-complete side wins, so a capped/truncated
        // store can never degrade an existing entry, and a stored truncated
        // entry is only ever served under the exact truncated-reuse scope (or
        // not at all when unscoped). Standard capped callers keep the
        // historical no-store contract unchanged.
        let store_allowed = !complete_cache_lookup_only
            || options.collection_mode.requires_shrink_only_publication();
        match cache_key {
            Some((key, fp)) if store_allowed => {
                // Capture the exact producer policy even when the collection
                // itself exhausted its deadline. A later live deadline can
                // serve this entry only under the same descriptor and no more
                // tightening authority than this producer had.
                let truncated_scope = truncated_reuse_requested
                    .then(|| {
                        self.crown_ibp_truncated_reuse_scope_with_prefix_cost_policy_and_snapshots(
                            &fresh,
                            producer_tightening_budget,
                            prefix_cost_admission_enabled,
                            runtime_policy,
                            cut_segment,
                            published_margin_indices.as_ref(),
                        )
                    })
                    .flatten();
                // #cgan-scope-diff: print the STORED scope verbatim. Five
                // hypotheses about which field differs were formed by reading
                // code and all five were wrong; this prints the struct so the
                // sixth answer comes from a diff, not a guess.
                debug!("#cgan-scope-diff STORE key={key:016x} scope={truncated_scope:?}");
                Ok(self.crown_ibp_collection_store(key, fp, fresh, truncated_scope))
            }
            _ => Ok(fresh),
        }
    }

    /// Bit-exact cache key for the CROWN-IBP collection cache
    /// (#cgan-collection-cache): the input box's f32 bit patterns (lower then
    /// upper, flattened) and shape, plus a coverage descriptor of the request
    /// — node count and output node (graph-shape sanity guard), conv-mode
    /// policy (`use_patches_mode` changes which backward the collector runs),
    /// and engine presence (GEMM engine routing can change f32 summation
    /// order). Any single-ULP change to any bound — including `-0.0` vs `0.0`
    /// — produces a different key.
    #[cfg(test)]
    fn crown_ibp_collection_cache_key(&self, input: &BoundedTensor, engine_present: bool) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(&self.crown_ibp_collection_cache_fingerprint(input, engine_present));
        hasher.finish()
    }

    /// The EXACT byte string the cache key hashes. Stored in the cache entry
    /// and compared on every hit/merge so a u64 hash collision across
    /// different boxes can never serve or merge a wrong map (adversarial-
    /// review hardening: the key alone is a hash, not an identity).
    pub(crate) fn crown_ibp_collection_cache_fingerprint(
        &self,
        input: &BoundedTensor,
        engine_present: bool,
    ) -> Vec<u8> {
        let published_margin_indices = crate::output_margin_seed::published_margin_indices();
        self.crown_ibp_collection_cache_fingerprint_with_runtime_policy(
            input,
            engine_present,
            crown_ibp_downstream_resweep_enabled(),
            target_backward::PartialCrownDeadlineSalvagePolicy::from_environment(),
            CrownIbpCollectionMode::Standard,
            self.crown_ibp_runtime_policy(),
            target_backward::crown_cut_segment_from_env(),
            published_margin_indices.as_deref(),
        )
    }

    fn crown_ibp_collection_cache_fingerprint_with_runtime_policy(
        &self,
        input: &BoundedTensor,
        engine_present: bool,
        downstream_resweep: bool,
        deadline_salvage_policy: target_backward::PartialCrownDeadlineSalvagePolicy,
        collection_mode: CrownIbpCollectionMode,
        runtime_policy: crate::network::core::graph::CrownIbpRuntimePolicy,
        cut_segment: usize,
        published_margin_indices: Option<&[usize]>,
    ) -> Vec<u8> {
        let mut fp = Vec::with_capacity(8 * input.lower().len() + 8 * input.shape().len() + 64);
        fp.extend_from_slice(b"ny.crown-ibp-coverage.v9\0");
        for v in input.lower().iter().chain(input.upper().iter()) {
            fp.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        fp.extend_from_slice(&(input.lower().len() as u64).to_le_bytes());
        fp.extend_from_slice(&(input.shape().len() as u64).to_le_bytes());
        for d in input.shape() {
            fp.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        fp.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        fp.extend_from_slice(&(self.output_node.len() as u64).to_le_bytes());
        fp.extend_from_slice(self.output_node.as_bytes());
        fp.push(u8::from(self.use_patches_mode));
        fp.push(u8::from(engine_present));
        // This dark gate changes the computed enclosure map (never its
        // soundness), so a process that toggles it must not receive a
        // pre-toggle cache entry.
        fp.push(u8::from(downstream_resweep));
        // Deadline salvage changes only truncated target quality, not
        // soundness, but that is still a distinct cache/provenance policy.
        fp.push(u8::from(deadline_salvage_policy.is_enabled()));
        // Snapshot every graph runtime control explicitly, including for
        // compatibility graphs that carry no frozen LeverSet. The same policy
        // drives the fresh collection, so environment drift after entry cannot
        // relabel or reuse work under a different chunk/Patches/provenance arm.
        fp.extend_from_slice(&(runtime_policy.objective_chunk_rows as u64).to_le_bytes());
        fp.push(u8::from(
            runtime_policy.chunk_projection_abort_enabled,
        ));
        fp.push(u8::from(runtime_policy.chunk_adaptive_growth_enabled));
        fp.push(u8::from(runtime_policy.chunk_wave_parallel_enabled));
        fp.push(u8::from(
            runtime_policy.patches_deadline_flat_bias_enabled,
        ));
        fp.push(u8::from(runtime_policy.honest_provenance_enabled));
        // The collection mode decides target selection and publication policy
        // (#cgan-restart-root-collection-cache); entries never serve across
        // modes.
        fp.push(collection_mode.cache_tag());
        // Cut segments trade correlation quality for a shorter backward walk.
        // A no-cut caller must never inherit a cut-produced map (or vice versa).
        fp.extend_from_slice(&(cut_segment as u64).to_le_bytes());
        // The raw, sorted objective publication controls both OUTPUT row
        // selection and intermediate influence cones. Length-prefix the exact
        // indices so two objective scopes can never share quality state.
        match published_margin_indices {
            Some(indices) => {
                fp.push(1);
                fp.extend_from_slice(&(indices.len() as u64).to_le_bytes());
                for &index in indices {
                    fp.extend_from_slice(&(index as u64).to_le_bytes());
                }
            }
            None => fp.push(0),
        }
        // Scored execution carries the exact receipt-owning set on the graph.
        // This makes every declared lever part of cache identity by
        // construction instead of relying only on a hand-maintained field list.
        match self.frozen_levers.as_ref() {
            Some(levers) => {
                fp.push(1);
                fp.extend_from_slice(&levers.fingerprint());
            }
            None => fp.push(0),
        }
        fp
    }

    /// Focused cache-policy tests override provenance without mutating the
    /// process environment. Production always supplies one complete snapshot.
    #[cfg(test)]
    fn crown_ibp_collection_cache_fingerprint_for_policies(
        &self,
        input: &BoundedTensor,
        engine_present: bool,
        downstream_resweep: bool,
        deadline_salvage_policy: target_backward::PartialCrownDeadlineSalvagePolicy,
        collection_mode: CrownIbpCollectionMode,
        honest_provenance_enabled: bool,
        cut_segment: usize,
        published_margin_indices: Option<&[usize]>,
    ) -> Vec<u8> {
        let mut runtime_policy = self.crown_ibp_runtime_policy();
        runtime_policy.honest_provenance_enabled = honest_provenance_enabled;
        self.crown_ibp_collection_cache_fingerprint_with_runtime_policy(
            input,
            engine_present,
            downstream_resweep,
            deadline_salvage_policy,
            collection_mode,
            runtime_policy,
            cut_segment,
            published_margin_indices,
        )
    }

    /// Whether every requested CROWN target completed.
    ///
    /// A sound fallback map is not necessarily quality-complete. In particular,
    /// memory admission and backend capability can improve between consumers;
    /// promoting such a map to the policy-agnostic complete lane would suppress
    /// a fresh tightening pass that may now solve the property. Only genuine
    /// CROWN results and documented structural omissions are complete.
    fn crown_ibp_result_is_complete(result: &GraphCrownIbpBoundsResult) -> bool {
        result.all_requested_crown_targets_completed()
    }

    /// Completeness order for two collections of the same key: number of
    /// nodes that got real CROWN tightening.
    fn crown_ibp_result_crown_count(result: &GraphCrownIbpBoundsResult) -> usize {
        result
            .provenance
            .values()
            .filter(|p| matches!(p, BoundsProvenance::Crown))
            .count()
    }

    /// Exact producer/consumer policy descriptor for the default-dark
    /// truncated-map lane.
    ///
    /// Objective-dependent OUTPUT subset seeding and cut-segment sweeps are
    /// deliberately refused rather than keyed: both are experimental
    /// collection contexts, and the cGAN beneficiary uses neither. Ordinary
    /// CROWN-IBP collection never reads BaB α, β, or the cut-fold registry
    /// (every target backward receives `alpha_state=None` and no BaB cut
    /// policy), so those states cannot become stale inputs to this map.
    #[cfg(test)]
    fn crown_ibp_truncated_reuse_scope(
        &self,
        result: &GraphCrownIbpBoundsResult,
        tightening_budget: Option<Duration>,
    ) -> Option<crate::network::core::graph::CrownIbpTruncatedReuseScope> {
        self.crown_ibp_truncated_reuse_scope_with_prefix_cost_policy(
            result,
            tightening_budget,
            budget_policy::prefix_cost_admission_enabled(),
            self.crown_ibp_runtime_policy().honest_provenance_enabled,
        )
    }

    #[cfg(test)]
    fn crown_ibp_truncated_reuse_scope_with_prefix_cost_policy(
        &self,
        result: &GraphCrownIbpBoundsResult,
        tightening_budget: Option<Duration>,
        prefix_cost_admission_enabled: bool,
        honest_provenance_enabled: bool,
    ) -> Option<crate::network::core::graph::CrownIbpTruncatedReuseScope> {
        let published_margin_indices = crate::output_margin_seed::published_margin_indices();
        let mut runtime_policy = self.crown_ibp_runtime_policy();
        runtime_policy.honest_provenance_enabled = honest_provenance_enabled;
        self.crown_ibp_truncated_reuse_scope_with_prefix_cost_policy_and_snapshots(
            result,
            tightening_budget,
            prefix_cost_admission_enabled,
            runtime_policy,
            target_backward::crown_cut_segment_from_env(),
            published_margin_indices.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn crown_ibp_truncated_reuse_scope_with_prefix_cost_policy_and_snapshots(
        &self,
        result: &GraphCrownIbpBoundsResult,
        tightening_budget: Option<Duration>,
        prefix_cost_admission_enabled: bool,
        runtime_policy: crate::network::core::graph::CrownIbpRuntimePolicy,
        cut_segment: usize,
        published_margin_indices: Option<&std::sync::Arc<[usize]>>,
    ) -> Option<crate::network::core::graph::CrownIbpTruncatedReuseScope> {
        let honest_provenance_enabled = runtime_policy.honest_provenance_enabled;
        // Every refusal below names itself on the armed lane
        // (#cgan-truncated-serve-telemetry): the Aug-03 investigation measured
        // 7 of 48 lookups failing with `current=None` and could only NARROW the
        // cause to two candidate early-returns; these markers make the next
        // armed run decisive instead of inferential.
        let Some(tightening_budget) = tightening_budget.filter(|budget| !budget.is_zero()) else {
            truncated_lane_event("scope-refused", "reason=zero-tightening-budget");
            return None;
        };
        if Self::crown_ibp_result_crown_count(result) == 0 {
            truncated_lane_event("scope-refused", "reason=no-crown-nodes");
            return None;
        }
        // This lane is for time-truncated work only. Resource/backend/shape
        // fallbacks can become executable for a later consumer even under the
        // same requested policy (for example after host memory is released),
        // so they must never acquire reusable truncated authority.
        let reusable_reason = |reason: CrownIbpFallbackReason| {
            matches!(
                reason,
                CrownIbpFallbackReason::DeadlineExceeded
                    | CrownIbpFallbackReason::PerNodeDeadlineExceeded
                    | CrownIbpFallbackReason::PerNodeBudgetBelowFloor
                    | CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                    | CrownIbpFallbackReason::PatchesBudgetExceeded
                    | CrownIbpFallbackReason::WalkCostRefused
                    | CrownIbpFallbackReason::DemandDrivenSkip
                    | CrownIbpFallbackReason::StableReluRowsSkipped
                    | CrownIbpFallbackReason::ObjectiveConeRowsSkipped
                    | CrownIbpFallbackReason::CrownVacuousResult
            )
        };
        let has_unscoped_fallback = result.provenance.values().any(|provenance| {
            matches!(provenance, BoundsProvenance::ForwardFallback(reason) if !reusable_reason(*reason))
        }) || result
            .fallback_events
            .iter()
            .any(|event| !reusable_reason(event.reason));
        if has_unscoped_fallback {
            truncated_lane_event("scope-refused", "reason=resource-or-backend-fallback");
            return None;
        }
        if cut_segment != 0 {
            truncated_lane_event("scope-refused", "reason=cut-segment-armed");
            return None;
        }

        let output_name = if self.output_name().is_empty() {
            let Some(last) = self.exec_order().ok().and_then(|order| order.last()) else {
                truncated_lane_event("scope-refused", "reason=no-output-node-name");
                return None;
            };
            last.as_str()
        } else {
            self.output_name()
        };
        // A published margin subset is objective-specific. Refuse even though
        // its scattered map is sound; the dark lane must not let one
        // objective's row selection become another objective's quality state.
        //
        // #cgan-truncated-scope-output-dim (2026-08-18, defect (E) of
        // `docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md`): this guard used
        // to read the output dimension from `result.bounds`, so a
        // deadline-truncated map that happened to lack the OUTPUT node's entry
        // silently refused to build a scope at all (`?` on the map lookup) —
        // measured as the `current=None` residue that kept `scope_match=false`
        // after every other cache defect was fixed. The dimension exists only
        // to consult this guard, and when it is unavailable the guard is
        // consulted on the STRICTLY more conservative predicate "any margin
        // publication exists on this thread" (see
        // `output_margin_seed::margin_subset_published`): every map the
        // dim-consulted guard refuses is still refused, so the objective-leak
        // cannot slip through, and the documented-unsound shortcut of skipping
        // the guard outright is not taken. When the map DOES contain the
        // output node, behavior is bit-identical to the historical path.
        match result.bounds.get(output_name) {
            Some(output_bounds) => {
                if crate::output_margin_seed::margin_subset_indices_from_published(
                    published_margin_indices,
                    output_bounds.len(),
                )
                .is_some()
                {
                    truncated_lane_event("scope-refused", "reason=margin-subset-engaged");
                    return None;
                }
            }
            None => {
                if published_margin_indices.is_some() {
                    truncated_lane_event(
                        "scope-refused",
                        "reason=margin-published-output-dim-unknown",
                    );
                    return None;
                }
                truncated_lane_event(
                    "scope-built-without-output-node",
                    &format!("output_node={output_name}"),
                );
            }
        }

        let budget =
            budget_policy::resolve_per_node_time_budget(&self.crown_ibp_per_node_time_budget);
        let patches_budget = budget_policy::resolve_patches_tightening_budget();
        Some(crate::network::core::graph::CrownIbpTruncatedReuseScope {
            policy: crate::network::core::graph::CrownIbpTruncatedReusePolicy {
                deadline_bearing: true,
                frozen_lever_fingerprint: self
                    .frozen_levers
                    .as_ref()
                    .map(|levers| levers.fingerprint()),
                effective_per_node_floor_bits: budget.floor_secs.to_bits(),
                effective_per_node_cap_bits: budget.cap_secs.to_bits(),
                per_node_cap_is_explicit: budget.cap_is_explicit,
                patches_budget_bits: patches_budget.floor_secs.to_bits(),
                patches_budget_is_explicit: patches_budget.is_explicit,
                dim_cap_scale_enabled: budget_policy::dim_cap_scale_enabled(),
                conv_patches_collect_enabled: patches_budget.conv_patches_collect_enabled,
                crown_mem_cap_env: std::env::var_os("NY_CROWN_MEM_CAP_MB"),
                patches_gpu_env: std::env::var_os("NY_PATCHES_GPU"),
                conv_skip_dead_f32_env: std::env::var_os("NY_CONV_SKIP_DEAD_F32"),
                convtranspose_sound_f64_gpu_env: std::env::var_os("NY_CONVTRANSPOSE_SOUND_F64_GPU"),
                patches_reentry_min_rows_env: std::env::var_os("NY_PATCHES_REENTRY_MIN_ROWS"),
                fast_f32_gemm_installed: crate::fast_f32_gemm::is_installed(),
                sound_f64_gemm_installed: crate::sound_f64_gemm::is_installed(),
                objective_chunk_rows: runtime_policy.objective_chunk_rows,
                chunk_aware_budget_enabled: budget_policy::crown_chunk_aware_budget_enabled(),
                chunk_wave_parallel_enabled: runtime_policy.chunk_wave_parallel_enabled,
                chunk_wave_workers: if runtime_policy.chunk_wave_parallel_enabled {
                    rayon::current_num_threads()
                } else {
                    1
                },
                chunk_abort_enabled: runtime_policy.chunk_projection_abort_enabled,
                chunk_grow_enabled: runtime_policy.chunk_adaptive_growth_enabled,
                patches_deadline_flat_bias_enabled: runtime_policy
                    .patches_deadline_flat_bias_enabled,
                crown_honest_provenance_enabled: honest_provenance_enabled,
                hopeless_class_skip_enabled: crown_tighten::hopeless_class_skip_enabled(),
                prefix_cost_admission_enabled,
            },
            tightening_budget,
        })
    }

    /// A truncated cache entry may be served only when it still contains one
    /// explicit bound and provenance tag for EVERY graph node. A malformed or
    /// genuinely partial target set misses and follows the historical fresh
    /// collection/fallback path.
    fn crown_ibp_truncated_result_covers_graph(&self, result: &GraphCrownIbpBoundsResult) -> bool {
        let Ok(exec_order) = self.exec_order() else {
            return false;
        };
        exec_order
            .iter()
            .all(|name| result.bounds.contains_key(name) && result.provenance.contains_key(name))
    }

    /// Intersect a SERVED cached collection with the caller's precomputed IBP
    /// map (#cgan-precomputed-ibp-cache).
    ///
    /// Soundness: the cached entry and `ibp` are both valid enclosures of the
    /// same (network, input box) — the cached one because every collection
    /// publishes only valid enclosures, the IBP one by construction. Keeping
    /// the tighter side per element is therefore a valid enclosure, the same
    /// argument the cache MERGE path already relies on. The result is at least
    /// as tight as the uncached path would have produced, since that path
    /// intersects the very same IBP map into a freshly computed CROWN.
    ///
    /// Fail-safe in both degenerate directions: a shape mismatch or a disjoint
    /// intersection leaves the cached bound untouched rather than publishing a
    /// box narrower than either operand justifies. Nodes present only in `ibp`
    /// are NOT added — the served map's node set is what the caller's contract
    /// expects, and widening it here would change provenance without evidence.
    fn intersect_served_collection_with_ibp(
        mut hit: GraphCrownIbpBoundsResult,
        ibp: &HashMap<String, BoundedTensor>,
    ) -> GraphCrownIbpBoundsResult {
        let mut tightened = 0usize;
        for (name, cached) in hit.bounds.iter_mut() {
            let Some(ibp_bound) = ibp.get(name) else {
                continue;
            };
            if ibp_bound.shape() != cached.shape() {
                continue;
            }
            if let Some((merged, _disjoint)) = cached.intersection_per_element(ibp_bound) {
                *cached = merged;
                tightened += 1;
            }
        }
        if tightened > 0 {
            debug!(
                "CROWN-IBP DAG: served cached collection intersected with caller IBP \
                 ({tightened} nodes tightened) (#cgan-precomputed-ibp-cache)"
            );
        }
        hit
    }

    /// Serve a cached collection for `key` when the cached entry is complete,
    /// or when the default-dark truncated lane passes its directional policy,
    /// authority, and graph-coverage checks.
    #[cfg(test)]
    fn crown_ibp_collection_lookup(
        &self,
        key: u64,
        fingerprint: &[u8],
        allow_truncated: bool,
        consumer_tightening_budget: Option<Duration>,
    ) -> Option<GraphCrownIbpBoundsResult> {
        let published_margin_indices = crate::output_margin_seed::published_margin_indices();
        self.crown_ibp_collection_lookup_with_prefix_cost_policy(
            key,
            fingerprint,
            allow_truncated,
            consumer_tightening_budget,
            budget_policy::prefix_cost_admission_enabled(),
            self.crown_ibp_runtime_policy(),
            target_backward::crown_cut_segment_from_env(),
            published_margin_indices.as_ref(),
        )
    }

    fn crown_ibp_collection_lookup_with_prefix_cost_policy(
        &self,
        key: u64,
        fingerprint: &[u8],
        allow_truncated: bool,
        consumer_tightening_budget: Option<Duration>,
        prefix_cost_admission_enabled: bool,
        runtime_policy: crate::network::core::graph::CrownIbpRuntimePolicy,
        cut_segment: usize,
        published_margin_indices: Option<&std::sync::Arc<[usize]>>,
    ) -> Option<GraphCrownIbpBoundsResult> {
        let guard = self.cached_crown_ibp_collection.slots.read().ok()?;
        // Exact (key, fingerprint) match across the retained entries. The
        // fingerprint disambiguates a u64 hash collision between different boxes,
        // so multiple same-hash entries coexist and only the TRUE box is ever
        // served (#cgan-collection-multislot).
        let entry = guard
            .iter()
            .find(|e| e.key == key && e.fingerprint.as_ref() == fingerprint)?;
        // A truncated (time-budget-cut) map is never CLAIMED complete. Gate-off
        // retains the historical miss/re-run policy. Gate-on can serve the
        // result only as the exact partial result it is: CROWN∩IBP entries plus
        // explicit IBP fallbacks and their original provenance/events.
        if !entry.complete {
            let current_scope = allow_truncated
                .then(|| {
                    self.crown_ibp_truncated_reuse_scope_with_prefix_cost_policy_and_snapshots(
                        &entry.result,
                        consumer_tightening_budget,
                        prefix_cost_admission_enabled,
                        runtime_policy,
                        cut_segment,
                        published_margin_indices,
                    )
                })
                .flatten();
            debug!(
                "#cgan-scope-diff LOOKUP key={key:016x} current={current_scope:?} stored={:?}",
                entry.truncated_reuse_scope
            );
            let scope_matches = entry
                .truncated_reuse_scope
                .as_ref()
                .zip(current_scope.as_ref())
                .is_some_and(|(producer, consumer)| truncated_scope_can_serve(producer, consumer));
            let covers_graph =
                scope_matches && self.crown_ibp_truncated_result_covers_graph(&entry.result);
            if !covers_graph {
                // #cgan-truncated-serve-telemetry: on the ARMED lane, name the
                // failing predicate and how many exec-order nodes the entry is
                // missing, so one run distinguishes a scope defect from a
                // genuinely partial map (the two candidate residuals the
                // Aug-03/Aug-11 investigations could not separate from logs).
                if allow_truncated {
                    let missing_nodes = self.exec_order().map_or(usize::MAX, |order| {
                        order
                            .iter()
                            .filter(|name| {
                                !(entry.result.bounds.contains_key(*name)
                                    && entry.result.provenance.contains_key(*name))
                            })
                            .count()
                    });
                    truncated_lane_event(
                        "miss",
                        &format!(
                            "key={key:016x} stored_scope={} current_scope={} \
                             scope_match={scope_matches} missing_nodes={missing_nodes} \
                             crown={}",
                            entry.truncated_reuse_scope.is_some(),
                            current_scope.is_some(),
                            entry.crown_count,
                        ),
                    );
                }
                debug!(
                    "CROWN-IBP DAG: cached collection for key {key:016x} is truncated \
                     ({} crown nodes) — re-running (allow_truncated={}, scope_match={}, \
                     full_coverage={}, graph={:p})",
                    entry.crown_count,
                    allow_truncated,
                    scope_matches,
                    scope_matches && self.crown_ibp_truncated_result_covers_graph(&entry.result),
                    std::ptr::from_ref(self),
                );
                return None;
            }
            self.cached_crown_ibp_collection
                .hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.cached_crown_ibp_collection
                .truncated_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info!(
                "CROWN-IBP DAG: #cgan-truncated-cache engaged — reusing partial \
                 collection unchanged ({} nodes, {} crown, {} explicit fallbacks, \
                 key {:016x}, complete=false)",
                entry.result.bounds.len(),
                entry.crown_count,
                entry.result.fallback_events.len(),
                key,
            );
            truncated_lane_event(
                "serve",
                &format!(
                    "key={key:016x} nodes={} crown={}",
                    entry.result.bounds.len(),
                    entry.crown_count,
                ),
            );
            return Some((*entry.result).clone());
        }
        self.cached_crown_ibp_collection
            .hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        info!(
            "CROWN-IBP DAG: reusing cached collection ({} nodes, key {:016x})",
            entry.result.bounds.len(),
            key
        );
        Some((*entry.result).clone())
    }

    /// Store policy (#cgan-collection-cache v2): same-key maps are MERGED by
    /// per-node per-element INTERSECTION. Both maps are valid enclosures for
    /// the identical (graph, input box), so their intersection is a valid
    /// enclosure at least as tight as either — this dissolves the old
    /// ordering problem where a later budget-starved re-run with an EQUAL
    /// crown-node count (but worse per-chunk quality, e.g. a deadline-cut
    /// BN_11 chunked backward) overwrote a tighter map ("the LAST truncated
    /// collection wins"), and makes repeated truncated re-runs monotonically
    /// tightening instead of wasted.
    fn crown_ibp_collection_store(
        &self,
        key: u64,
        fingerprint: Vec<u8>,
        fresh: GraphCrownIbpBoundsResult,
        mut truncated_reuse_scope: Option<crate::network::core::graph::CrownIbpTruncatedReuseScope>,
    ) -> GraphCrownIbpBoundsResult {
        let fresh_complete = Self::crown_ibp_result_is_complete(&fresh);
        let Ok(mut guard) = self.cached_crown_ibp_collection.slots.write() else {
            return fresh; // poisoned lock: skip caching, never block the result
        };
        // Same-box entry = exact (key, fingerprint). A u64 hash collision across
        // DIFFERENT boxes has a different fingerprint, so it is a SEPARATE entry
        // (never merged) — the retained set removes the old single-slot's
        // collision-replace hazard entirely (#cgan-collection-multislot).
        // NOTE: merging is not a cache HIT (the caller still computed a fresh
        // collection); `hits` counts lookup serves only.
        let existing_idx = guard
            .iter()
            .position(|e| e.key == key && e.fingerprint.as_ref() == fingerprint.as_slice());
        // A gated truncated producer may merge only with an entry captured
        // under the IDENTICAL truncated-reuse POLICY. Producer budget is
        // ordered rather than compared for equality: the intersection of two
        // same-policy maps dominates both, so it inherits the larger producer
        // authority. If a process enables the
        // gate after an unscoped entry already exists (or changes a resource
        // knob mid-process), replace it with the fresh result instead of
        // laundering stale quality state into a newly scoped entry.
        let truncated_scope_mismatch = !fresh_complete
            && truncated_reuse_scope.is_some()
            && existing_idx.is_some_and(|idx| {
                !guard[idx].complete
                    && match (
                        guard[idx].truncated_reuse_scope.as_ref(),
                        truncated_reuse_scope.as_ref(),
                    ) {
                        (Some(cached), Some(fresh)) => cached.policy != fresh.policy,
                        _ => true,
                    }
            });
        if truncated_scope_mismatch {
            info!(
                "CROWN-IBP DAG: #cgan-truncated-cache producer policy changed for key \
                 {key:016x}; replacing the old partial entry instead of merging it"
            );
        }
        if !truncated_scope_mismatch {
            if let Some(idx) = existing_idx {
                if !guard[idx].complete {
                    if let (Some(cached), Some(fresh)) = (
                        guard[idx].truncated_reuse_scope.as_ref(),
                        truncated_reuse_scope.as_mut(),
                    ) {
                        if cached.policy == fresh.policy {
                            fresh.tightening_budget =
                                fresh.tightening_budget.max(cached.tightening_budget);
                        }
                    }
                }
            }
        }
        let merged = match existing_idx.filter(|_| !truncated_scope_mismatch) {
            Some(idx) => {
                let entry = &guard[idx];
                let mut merged = fresh;
                let mut tightened_nodes = 0usize;
                for (name, cached_bound) in &entry.result.bounds {
                    match merged.bounds.get(name) {
                        Some(fresh_bound) if fresh_bound.shape() == cached_bound.shape() => {
                            // Intersection of two valid enclosures for the same
                            // box is a valid enclosure (see the soundness
                            // invariant on `collect_crown_ibp_bounds_dag_with_options`).
                            if let Some((tightened, _disjoint)) =
                                fresh_bound.intersection_per_element(cached_bound)
                            {
                                tightened_nodes += 1;
                                merged.bounds.insert(name.clone(), tightened);
                            }
                        }
                        // Shape drift (defensive; same graph+box should never):
                        // keep the fresh side untouched.
                        Some(_) => {}
                        None => {
                            merged.bounds.insert(name.clone(), cached_bound.clone());
                        }
                    }
                }
                // Provenance: a node is CROWN-grade if either side got there.
                for (name, prov) in &entry.result.provenance {
                    if matches!(prov, BoundsProvenance::Crown) {
                        merged
                            .provenance
                            .insert(name.clone(), BoundsProvenance::Crown);
                    }
                }
                // Completeness: a cached complete CROWN target promotes the
                // matching fresh partial target above, so the merged map may
                // inherit the complete side's time-event-free list. A cached
                // deterministic fallback does NOT erase fresh partial-row
                // provenance: more time could still finish those rows, so keep
                // the partial event and cache authority explicitly truncated.
                if entry.complete
                    && !fresh_complete
                    && !merged.provenance.values().any(|provenance| {
                        matches!(
                            provenance,
                            BoundsProvenance::ForwardFallback(
                                CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
                            )
                        )
                    })
                {
                    merged.fallback_events = entry.result.fallback_events.clone();
                }
                info!(
                    "CROWN-IBP DAG: merged re-run into cached collection ({} nodes, \
                     {} intersected, key {:016x}, complete={})",
                    merged.bounds.len(),
                    tightened_nodes,
                    key,
                    Self::crown_ibp_result_is_complete(&merged),
                );
                merged
            }
            None => fresh,
        };
        let merged_complete = Self::crown_ibp_result_is_complete(&merged);
        let merged_crown_count = Self::crown_ibp_result_crown_count(&merged);
        let new_entry = crate::network::core::graph::CrownIbpCollectionCacheEntry {
            key,
            fingerprint: std::sync::Arc::from(fingerprint),
            result: std::sync::Arc::new(merged.clone()),
            complete: merged_complete,
            crown_count: merged_crown_count,
            truncated_reuse_scope,
        };
        // Insert most-recently-stored first; a merged same-box entry is moved to
        // the front (kept hot). Bounded LRU eviction only drops the LEAST-recent
        // DIFFERENT-box entry — never a wrong or looser serve, since every entry
        // is served only on an exact key+fingerprint match.
        match existing_idx {
            Some(idx) => {
                guard.remove(idx);
                guard.insert(0, new_entry);
            }
            None => {
                guard.insert(0, new_entry);
                let cap = crate::network::core::graph::CROWN_IBP_COLLECTION_CACHE_CAP;
                if guard.len() > cap {
                    guard.truncate(cap);
                }
            }
        }
        // debug: per-child input-split collections (cersyve-class lanes) store
        // thousands of never-reused unique-key entries — info would spam -v.
        debug!(
            "CROWN-IBP DAG: collection cached ({} nodes, {} crown, complete={}, key {:016x}, entries={}, graph={:p})",
            merged.bounds.len(),
            merged_crown_count,
            merged_complete,
            key,
            guard.len(),
            std::ptr::from_ref(self),
        );
        merged
    }

    /// Deterministically install a sound mixed-quality fixture through the
    /// production store path. Tests build it from an independently completed
    /// collection, retain at least one CROWN target, and attach an explicit
    /// deadline fallback to another target; no timing race is needed.
    #[cfg(test)]
    pub(crate) fn seed_truncated_collection_cache_for_test(
        &self,
        input: &BoundedTensor,
        result: GraphCrownIbpBoundsResult,
    ) -> GraphCrownIbpBoundsResult {
        assert!(!Self::crown_ibp_result_is_complete(&result));
        assert!(Self::crown_ibp_result_crown_count(&result) > 0);
        assert!(self.crown_ibp_truncated_result_covers_graph(&result));
        let scope = self
            .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
            .expect("mixed fixture must satisfy the truncated-reuse scope");
        let key = self.crown_ibp_collection_cache_key(input, false);
        let fingerprint = self.crown_ibp_collection_cache_fingerprint(input, false);
        self.crown_ibp_collection_store(key, fingerprint, result, Some(scope))
    }

    /// Collect CROWN-IBP bounds at each node in the graph.
    ///
    /// Convenience wrapper — equivalent to
    /// `collect_crown_ibp_bounds_dag_with_options(input, Default::default())`.
    pub fn collect_crown_ibp_bounds_dag(
        &self,
        input: &BoundedTensor,
    ) -> Result<HashMap<String, BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_dag_with_options(input, CrownIbpCollectOptions::default())?
            .bounds)
    }

    /// Collect CROWN-IBP bounds with optional GPU/accelerated GEMM engine (#3549).
    pub fn collect_crown_ibp_bounds_dag_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_dag_with_options(
                input,
                CrownIbpCollectOptions {
                    engine,
                    ..Default::default()
                },
            )?
            .bounds)
    }

    /// Collect CROWN-IBP bounds with a deadline (#3109).
    pub fn collect_crown_ibp_bounds_dag_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_dag_with_options(
                input,
                CrownIbpCollectOptions {
                    deadline,
                    ..Default::default()
                },
            )?
            .bounds)
    }

    /// Collect CROWN-IBP bounds with deadline and optional GPU GEMM engine (#3549).
    pub fn collect_crown_ibp_bounds_dag_with_deadline_and_engine(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_dag_with_options(
                input,
                CrownIbpCollectOptions {
                    engine,
                    deadline,
                    ..Default::default()
                },
            )?
            .bounds)
    }

    /// Collect CROWN-IBP bounds while applying one authoritative deadline to
    /// both the foundational IBP pass and the subsequent tightening sweep.
    ///
    /// Unlike the historical deadline wrapper, this returns
    /// [`ny_core::NyError::DeadlineExceeded`] when the complete IBP map cannot
    /// be built in time. It is intentionally crate-private for verifier root
    /// phases that must emit a timely Timeout verdict.
    pub(crate) fn collect_crown_ibp_bounds_dag_with_hard_deadline_and_engine(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_dag_with_options(
                input,
                CrownIbpCollectOptions {
                    engine,
                    deadline,
                    deadline_includes_ibp: true,
                    ..Default::default()
                },
            )?
            .bounds)
    }

    /// Collect CROWN-IBP bounds at each node plus per-node fallback provenance.
    pub fn collect_crown_ibp_bounds_dag_with_status(
        &self,
        input: &BoundedTensor,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(input, CrownIbpCollectOptions::default())
    }

    /// Collect CROWN-IBP bounds with per-node provenance and optional GPU engine (#3549).
    pub fn collect_crown_ibp_bounds_dag_with_status_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                engine,
                ..Default::default()
            },
        )
    }

    /// Collect CROWN-IBP bounds using pre-computed IBP bounds (#3596).
    pub fn collect_crown_ibp_bounds_dag_with_precomputed_ibp(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                deadline,
                precomputed_ibp: Some(ibp_bounds),
                ..Default::default()
            },
        )
    }

    /// Collect CROWN-IBP bounds with pre-computed IBP and width skip (#3499).
    pub fn collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        min_width_to_tighten: f32,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                deadline,
                precomputed_ibp: Some(ibp_bounds),
                min_width_to_tighten: Some(min_width_to_tighten),
                ..Default::default()
            },
        )
    }

    /// Collect CROWN-IBP bounds with pre-computed IBP and GPU engine (#3549).
    pub fn collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                engine,
                deadline,
                precomputed_ibp: Some(ibp_bounds),
                ..Default::default()
            },
        )
    }

    /// Collect CROWN-IBP bounds with pre-computed IBP, GPU engine, and width skip.
    pub fn collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine_and_width_threshold(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
        min_width_to_tighten: f32,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                engine,
                deadline,
                precomputed_ibp: Some(ibp_bounds),
                min_width_to_tighten: Some(min_width_to_tighten),
                ..Default::default()
            },
        )
    }

    /// Core CROWN-IBP collection with optional deadline (#3109) and engine (#3549).
    ///
    /// When deadline is exceeded mid-collection, remaining nodes use IBP fallback
    /// with `DeadlineExceeded` provenance. This is sound (IBP bounds are valid,
    /// just looser than CROWN-IBP) and ensures the BaB loop doesn't overrun its
    /// time budget on CROWN-IBP bound collection alone.
    ///
    /// Retained for internal callers that already have the 3-parameter signature.
    pub(crate) fn collect_crown_ibp_bounds_dag_with_status_and_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                engine,
                deadline,
                ..Default::default()
            },
        )
    }

    /// CROWN-IBP collection with a caller-local cap on only the tightening
    /// sweep. Used by DAG alpha's pre-loop fixed-slope fallback when the
    /// preferred forward-linear reference is unavailable.
    ///
    /// The cap begins after the mandatory IBP map and never mutates the outer
    /// deadline. Capped calls may consume an already-complete exact-key cache
    /// entry, but never serve a truncated entry or store/merge their fresh map.
    pub(crate) fn collect_crown_ibp_bounds_dag_with_status_deadline_and_tightening_cap(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
        tightening_cap: Duration,
    ) -> Result<GraphCrownIbpBoundsResult> {
        eprintln!(
            "[NY_CROWN_IBP_COLLECTOR_CAP_V1] stage=graph-crown-step1-dispatch \
             requested_secs={}",
            tightening_cap.as_secs(),
        );
        self.collect_crown_ibp_bounds_dag_with_options(
            input,
            CrownIbpCollectOptions {
                engine,
                deadline,
                tightening_cap: Some(tightening_cap),
                complete_cache_lookup_only: true,
                ..Default::default()
            },
        )
    }

    // Core CROWN-IBP tightening loop is in crown_tighten.rs.

    /// Run backward CROWN from a target node to the network input.
    ///
    /// Returns CROWN bounds at the target node by:
    /// 1. Finding all nodes on paths from input to target
    /// 2. Running backward CROWN through this subgraph
    /// 3. Concretizing at the input
    ///
    /// `per_node_deadline` is an optional per-node time budget (#3499). When
    /// set, the backward layer loop bails out early if the budget is exhausted,
    /// returning `UnsupportedConfiguration` so the tightening loop can fall
    /// back to IBP and move on to other nodes.
    ///
    /// `chunk_override` (#cgan-bn11-chunk): explicit objective row-chunk size
    /// for the bound-equivalent chunked backward; `None` keeps the env-driven
    /// (`NY_CROWN_OBJ_CHUNK`) behavior byte-for-byte.
    ///
    /// `cut_ctx` (#crown-cut-segment): optional backward-to-nearest-bounded-cut
    /// context threaded from the CROWN-IBP sweep (`NY_CROWN_CUT_SEGMENT`);
    /// `None` (the default-OFF gate and every non-sweep caller) keeps the
    /// backward walk byte-identical.
    #[allow(clippy::too_many_arguments)] // Mirrors propagate_crown_to_node_core's threading (#3549).
    pub(crate) fn propagate_crown_to_node(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_ibp_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        per_node_deadline: Option<Instant>,
        chunk_override: Option<usize>,
        cut_ctx: Option<&target_backward::CrownCutContext>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_core(
            input,
            target_node,
            crown_ibp_bounds,
            ibp_bounds,
            None,
            engine,
            "CROWN-IBP",
            per_node_deadline,
            false,
            chunk_override,
            cut_ctx,
            None,
        )
    }

    /// Collector-specific backward CROWN that uses patches mode for spatial
    /// Conv2d targets even when the global conv_mode is matrix (#3813).
    ///
    /// The CROWN-IBP intermediate bounds collector doesn't use cutting planes,
    /// so the matrix-mode constraint (required for BaB cuts) doesn't apply.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)] // Mirrors propagate_crown_to_node_core's threading (#3549).
    pub(crate) fn propagate_crown_to_node_for_collector(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_ibp_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        per_node_deadline: Option<Instant>,
        chunk_override: Option<usize>,
        cut_ctx: Option<&target_backward::CrownCutContext>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_core(
            input,
            target_node,
            crown_ibp_bounds,
            ibp_bounds,
            None,
            engine,
            "CROWN-IBP-collector",
            per_node_deadline,
            true,
            chunk_override,
            cut_ctx,
            None,
        )
    }

    /// Execute an M1-admitted full-objective route with its retained fixed-wave
    /// plan. The core compares this expectation with the live driver decision
    /// immediately before dispatch and fails back to IBP on any drift.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // M1 retained-plan seam remains default-unwired.
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_with_fixed_wave_plan(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_ibp_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        per_node_deadline: Option<Instant>,
        chunk_override: Option<usize>,
        cut_ctx: Option<&target_backward::CrownCutContext>,
        collector_patches_override: bool,
        expected_fixed_waves: target_backward::ObjectiveChunkFixedWavePlan,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_to_node_core(
            input,
            target_node,
            crown_ibp_bounds,
            ibp_bounds,
            None,
            engine,
            if collector_patches_override {
                "CROWN-IBP-collector"
            } else {
                "CROWN-IBP"
            },
            per_node_deadline,
            collector_patches_override,
            chunk_override,
            cut_ctx,
            Some(expected_fixed_waves),
        )
    }

    /// Collector authority boundary for deadline-truncated objective chunks.
    ///
    /// When `deadline_salvage_policy` is explicitly armed, this can return an
    /// explicitly tagged partial result when a chunked walk times out;
    /// default-dark returns the typed deadline. The caller must retain fallback
    /// provenance and must not promote a partial target to a CROWN-complete
    /// cut/cache entry.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha) fn propagate_crown_to_node_with_partial_for_collector(
        &self,
        input: &BoundedTensor,
        target_node: &str,
        crown_ibp_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        per_node_deadline: Option<Instant>,
        deadline_is_hard: bool,
        chunk_override: Option<usize>,
        cut_ctx: Option<&target_backward::CrownCutContext>,
        collector_patches_override: bool,
        deadline_salvage_policy: target_backward::PartialCrownDeadlineSalvagePolicy,
        expected_fixed_waves: Option<target_backward::ObjectiveChunkFixedWavePlan>,
        runtime_policy: crate::network::core::graph::CrownIbpRuntimePolicy,
    ) -> Result<target_backward::TargetCrownCollectionResult> {
        self.propagate_crown_to_node_core_for_collector(
            input,
            target_node,
            crown_ibp_bounds,
            ibp_bounds,
            engine,
            if collector_patches_override {
                "CROWN-IBP-collector"
            } else {
                "CROWN-IBP"
            },
            per_node_deadline,
            deadline_is_hard,
            collector_patches_override,
            chunk_override,
            cut_ctx,
            deadline_salvage_policy,
            expected_fixed_waves,
            runtime_policy,
        )
    }

    /// All ancestor nodes of a target (nodes that can reach target).
    /// Returns nodes in topological order (dependencies before dependents).
    ///
    /// Uses the cached all-ancestors map (#2220 Packet A) so repeated calls
    /// during CROWN-IBP collection are O(1) lookups instead of O(N+E) BFS each.
    pub(super) fn ancestors(&self, target: &str) -> Result<Vec<String>> {
        let all = self.all_ancestors()?;
        Ok(all.get(target).cloned().unwrap_or_default())
    }

    /// NaN-safe element-wise addition of all 4 coefficient fields (lower_a,
    /// lower_b, upper_a, upper_b) from `new_bounds` into `existing`.
    #[cfg(test)]
    fn safe_add_all(existing: &mut LinearBounds, new_bounds: &LinearBounds) {
        *existing.lower_a_mut() = Self::safe_add(existing.lower_a(), new_bounds.lower_a(), true);
        *existing.lower_b_mut() = Self::safe_add(existing.lower_b(), new_bounds.lower_b(), true);
        *existing.upper_a_mut() = Self::safe_add(existing.upper_a(), new_bounds.upper_a(), false);
        *existing.upper_b_mut() = Self::safe_add(existing.upper_b(), new_bounds.upper_b(), false);
    }

    /// Accumulate linear bounds to a node during backward CROWN-IBP pass.
    ///
    /// Uses NaN-safe addition (matching `accumulate_bounds_to_input` in graph_crown)
    /// to prevent NaN corruption from propagating through the CROWN-IBP chain.
    /// When INF + (-INF) produces NaN, lower bounds fall back to NEG_INFINITY
    /// and upper bounds fall back to INFINITY (sound but conservative).
    #[cfg(test)]
    pub(super) fn accumulate_crown_ibp_bounds(
        input_name: &str,
        new_bounds: LinearBounds,
        node_linear_bounds: &mut HashMap<String, LinearBounds>,
        input_accumulated: &mut bool,
    ) {
        if input_name == NETWORK_INPUT {
            if *input_accumulated {
                if let Some(existing) = node_linear_bounds.get_mut(NETWORK_INPUT) {
                    Self::safe_add_all(existing, &new_bounds);
                }
            } else {
                node_linear_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            }
        } else if let Some(existing) = node_linear_bounds.get_mut(input_name) {
            Self::safe_add_all(existing, &new_bounds);
        } else {
            node_linear_bounds.insert(input_name.to_string(), new_bounds);
        }
    }
}

#[cfg(test)]
mod collection_cache_key_tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::core::GraphNode;
    use crate::types::CrownIbpPerNodeTimeBudget;
    use ndarray::{arr1, arr2, Array1};

    fn small_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        let l1 = LinearLayer::new(
            arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
            Some(arr1(&[0.1_f32, -0.1])),
        )
        .unwrap();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["l1".into()],
        ));
        graph.set_output("relu");
        graph
    }

    fn unit_box() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-0.5_f32, -0.25]).into_dyn(),
            arr1(&[0.5_f32, 0.75]).into_dyn(),
        )
        .unwrap()
    }

    #[test]
    fn downstream_resweep_gate_is_part_of_collection_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = target_backward::PartialCrownDeadlineSalvagePolicy::Disabled;
        let historical = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        let reswept = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            true,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        assert_ne!(
            historical, reswept,
            "a bound-changing collector gate must invalidate pre-gate cache entries"
        );
    }

    #[test]
    fn deadline_salvage_gate_is_part_of_collection_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            target_backward::PartialCrownDeadlineSalvagePolicy::Disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        let enabled = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            target_backward::PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        assert_ne!(
            disabled, enabled,
            "deadline-row salvage changes truncated quality and must invalidate cache identity"
        );
    }

    #[test]
    fn honest_provenance_guard_is_part_of_complete_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = target_backward::PartialCrownDeadlineSalvagePolicy::Disabled;
        let guarded = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        let unguarded = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            false,
            0,
            None,
        );
        assert_ne!(
            guarded, unguarded,
            "a guard-off complete entry must never serve guard-on provenance"
        );
    }

    /// #cgan-restart-root-collection-cache: the collection mode is cache
    /// identity — a typed cGAN entry and a Standard entry for the SAME box
    /// must never serve each other.
    #[test]
    fn collection_mode_is_part_of_collection_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = target_backward::PartialCrownDeadlineSalvagePolicy::Disabled;
        let standard = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        let cgan_complete = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::CganComplete,
            true,
            0,
            None,
        );
        let cgan_sparse = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::CganSparseTargetComplete,
            true,
            0,
            None,
        );
        assert_ne!(standard, cgan_complete);
        assert_ne!(standard, cgan_sparse);
        assert_ne!(cgan_complete, cgan_sparse);
    }

    #[test]
    fn cut_segment_is_part_of_complete_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = target_backward::PartialCrownDeadlineSalvagePolicy::Disabled;
        let full_prefix = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            0,
            None,
        );
        let cut = graph.crown_ibp_collection_cache_fingerprint_for_policies(
            &input,
            false,
            false,
            disabled,
            CrownIbpCollectionMode::Standard,
            true,
            4,
            None,
        );
        assert_ne!(
            full_prefix, cut,
            "a cut-produced enclosure must not suppress a requested full-prefix walk"
        );
    }

    #[test]
    fn exact_objective_rows_are_part_of_complete_cache_identity() {
        let graph = small_graph();
        let input = unit_box();
        let disabled = target_backward::PartialCrownDeadlineSalvagePolicy::Disabled;
        let fingerprint = |rows: Option<&[usize]>| {
            graph.crown_ibp_collection_cache_fingerprint_for_policies(
                &input,
                false,
                false,
                disabled,
                CrownIbpCollectionMode::Standard,
                true,
                0,
                rows,
            )
        };
        assert_ne!(fingerprint(None), fingerprint(Some(&[3, 7])));
        assert_ne!(fingerprint(Some(&[3, 7])), fingerprint(Some(&[3, 8])));
        assert_ne!(fingerprint(Some(&[3, 7])), fingerprint(Some(&[3, 7, 8])));
    }

    #[test]
    fn receipt_owning_lever_set_is_graph_local_cache_authority() {
        let input = unit_box();
        let default_set =
            std::sync::Arc::new(ny_levers::LeverSet::freeze_with(ny_levers::all(), |_| None));
        let changed_set =
            std::sync::Arc::new(ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
                use ny_levers::decls::crown_ibp_budget::{
                    CROWN_HONEST_PROVENANCE, CROWN_OBJ_CHUNK, NO_CHUNK_ABORT, NO_CHUNK_GROW,
                    NO_CHUNK_WAVE_PAR, PATCHES_DEADLINE_FLAT_BIAS,
                };
                match name {
                    value if value == CROWN_OBJ_CHUNK.name => Some("32".to_owned()),
                    value if value == NO_CHUNK_ABORT.name => Some("1".to_owned()),
                    value if value == NO_CHUNK_GROW.name => Some("1".to_owned()),
                    value if value == NO_CHUNK_WAVE_PAR.name => Some("0".to_owned()),
                    value if value == PATCHES_DEADLINE_FLAT_BIAS.name => Some("1".to_owned()),
                    value if value == CROWN_HONEST_PROVENANCE.name => Some("0".to_owned()),
                    _ => None,
                }
            }));

        let mut graph = small_graph();
        graph.set_frozen_levers(Some(std::sync::Arc::clone(&default_set)));
        let default_fp = graph.crown_ibp_collection_cache_fingerprint(&input, false);
        let cloned = graph.clone();
        assert!(std::sync::Arc::ptr_eq(
            cloned.frozen_levers().expect("clone retains authority"),
            &default_set,
        ));
        assert_eq!(
            graph.crown_ibp_runtime_policy(),
            crate::network::core::graph::CrownIbpRuntimePolicy {
                objective_chunk_rows: 0,
                chunk_projection_abort_enabled: true,
                chunk_adaptive_growth_enabled: true,
                chunk_wave_parallel_enabled: false,
                patches_deadline_flat_bias_enabled: false,
                honest_provenance_enabled: true,
            }
        );
        assert!(
            !graph
                .crown_ibp_runtime_policy()
                .requires_graph_native_chunk_execution(),
            "default chunk policy may retain the sequential shortcut"
        );

        graph.set_frozen_levers(Some(std::sync::Arc::clone(&changed_set)));
        assert_eq!(
            graph.crown_ibp_runtime_policy(),
            crate::network::core::graph::CrownIbpRuntimePolicy {
                objective_chunk_rows: 32,
                chunk_projection_abort_enabled: false,
                chunk_adaptive_growth_enabled: false,
                chunk_wave_parallel_enabled: true,
                patches_deadline_flat_bias_enabled: true,
                honest_provenance_enabled: false,
            }
        );
        assert!(
            graph
                .crown_ibp_runtime_policy()
                .requires_graph_native_chunk_execution(),
            "an explicit chunk treatment must use the graph-native executor"
        );
        let mut derived = GraphNetwork::new();
        derived.inherit_frozen_levers_from(&graph);
        assert!(std::sync::Arc::ptr_eq(
            derived
                .frozen_levers()
                .expect("derived graph retains authority"),
            &changed_set,
        ));
        let changed_fp = graph.crown_ibp_collection_cache_fingerprint(&input, false);
        assert_ne!(
            default_fp, changed_fp,
            "every declared frozen lever must participate in scored cache identity"
        );
    }

    /// #cgan-restart-root-collection-cache two-phase pin (cgan row 7 shape):
    /// the initial phase's typed cGAN root collection (capped +
    /// precomputed-IBP, i.e. `complete_cache_lookup_only`) must be STORED in
    /// the multislot cache, and the disjunctive restart lane's identical-box
    /// same-mode consumer — running on a clone that adopted the bound caches,
    /// exactly as `disjunctive_unified.rs` builds its BaB graph — must be
    /// SERVED from it instead of re-collecting the bit-identical root box.
    #[test]
    fn typed_cgan_capped_collection_stores_and_serves_across_restart_clone() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_DISABLE_CROWN_COLLECTION_CACHE",
                SERVE_TRUNCATED_COLLECTION_CACHE_ENV,
                SPARSE_RELU_ROWS_ENV,
                "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
                "NY_CROWN_DEADLINE_CHUNK_SALVAGE",
            ] {
                env.remove(key);
            }

            let graph = small_graph();
            let input = unit_box();
            let ibp = graph.collect_node_bounds(&input).expect("IBP baseline");

            // Phase 1: initial-phase shape (alpha.rs CganComplete branch) —
            // typed mode, tightening cap, precomputed IBP baseline, generous
            // deadline so the tiny collection completes.
            let phase1 = graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now() + Duration::from_mins(1)),
                        precomputed_ibp: Some(ibp.clone()),
                        tightening_cap: Some(Duration::from_mins(5)),
                        collection_mode: CrownIbpCollectionMode::CganComplete,
                        ..Default::default()
                    },
                )
                .expect("phase-1 typed collection");
            {
                let slots = graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock");
                let entry = slots.first().expect(
                    "capped typed cGAN producer must STORE its collection (the \
                     historical complete_cache_lookup_only guard discarded it)",
                );
                assert!(entry.complete, "fixture collection must be complete");
            }

            // Restart lane: fresh clone + bound-cache adoption (the exact
            // disjunctive_unified.rs pattern), same box, same mode, tighter
            // remaining budget.
            let mut restart_graph = graph.clone();
            restart_graph.adopt_bound_caches_from(&graph);
            assert_eq!(
                restart_graph
                    .cached_crown_ibp_collection
                    .hits
                    .load(std::sync::atomic::Ordering::Relaxed),
                0
            );
            let served = restart_graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now() + Duration::from_secs(30)),
                        precomputed_ibp: Some(ibp),
                        tightening_cap: Some(Duration::from_mins(5)),
                        collection_mode: CrownIbpCollectionMode::CganComplete,
                        ..Default::default()
                    },
                )
                .expect("restart-lane typed collection");
            assert_eq!(
                restart_graph
                    .cached_crown_ibp_collection
                    .hits
                    .load(std::sync::atomic::Ordering::Relaxed),
                1,
                "the restart-lane consumer must be SERVED from the adopted \
                 multislot cache instead of re-collecting the identical root box"
            );
            for name in phase1.bounds.keys() {
                assert!(
                    served.bounds.contains_key(name),
                    "served map must cover every phase-1 node ({name})"
                );
            }

            // Cache separation stays real: a Standard consumer of the SAME box
            // must NOT be served the typed entry (distinct fingerprint), and
            // its own collection stores a second, independent entry.
            let standard = restart_graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now() + Duration::from_secs(30)),
                        ..Default::default()
                    },
                )
                .expect("standard collection");
            assert!(!standard.bounds.is_empty());
            assert_eq!(
                restart_graph
                    .cached_crown_ibp_collection
                    .hits
                    .load(std::sync::atomic::Ordering::Relaxed),
                1,
                "a Standard-mode consumer must not hit the typed cGAN entry"
            );
            let slots = restart_graph
                .cached_crown_ibp_collection
                .slots
                .read()
                .expect("cache lock");
            assert_eq!(
                slots.len(),
                2,
                "typed and Standard collections must occupy separate entries"
            );
        });
    }

    /// #cgan-reclaim: a COMPLETE Standard-mode entry (the CLI per-output
    /// precheck's exact producer shape — Standard mode, no cap, deadline
    /// present) must serve a typed `CganComplete` consumer over the
    /// bit-identical adopted box when `NY_CGAN_RECLAIM=1`, shrink-only
    /// intersected against the consumer's own baseline; DARK, the modes stay
    /// separated byte-for-byte (zero hits).
    #[test]
    fn cgan_reclaim_serves_complete_standard_entry_across_modes() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                "NY_DISABLE_CROWN_COLLECTION_CACHE",
                SERVE_TRUNCATED_COLLECTION_CACHE_ENV,
                SPARSE_RELU_ROWS_ENV,
                "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
                "NY_CROWN_DEADLINE_CHUNK_SALVAGE",
                "NY_CGAN_RECLAIM",
            ] {
                env.remove(key);
            }

            let graph = small_graph();
            let input = unit_box();

            // Producer: the precheck shape — Standard mode, uncapped, with a
            // live deadline. Stores a COMPLETE Standard-tag entry.
            let standard = graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now() + Duration::from_mins(1)),
                        ..Default::default()
                    },
                )
                .expect("standard precheck-shape collection");
            assert!(!standard.bounds.is_empty());
            {
                let slots = graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock");
                assert!(
                    slots.first().is_some_and(|entry| entry.complete),
                    "fixture Standard collection must store a COMPLETE entry"
                );
            }

            let ibp = graph.collect_node_bounds(&input).expect("IBP baseline");

            // DARK control: the typed consumer must MISS across modes.
            let mut dark = graph.clone();
            dark.adopt_bound_caches_from(&graph);
            dark.collect_crown_ibp_bounds_dag_with_options(
                &input,
                CrownIbpCollectOptions {
                    deadline: Some(Instant::now() + Duration::from_mins(1)),
                    precomputed_ibp: Some(ibp.clone()),
                    tightening_cap: Some(Duration::from_mins(5)),
                    collection_mode: CrownIbpCollectionMode::CganComplete,
                    ..Default::default()
                },
            )
            .expect("typed collection (dark)");
            assert_eq!(
                dark.cached_crown_ibp_collection
                    .hits
                    .load(std::sync::atomic::Ordering::Relaxed),
                0,
                "dark: typed cGAN entries must never serve across modes"
            );

            // ARMED: the cross-mode serve fires and stays shrink-only vs the
            // consumer's baseline.
            env.set("NY_CGAN_RECLAIM", "1");
            let mut armed = graph.clone();
            armed.adopt_bound_caches_from(&graph);
            let served = armed
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now() + Duration::from_mins(1)),
                        precomputed_ibp: Some(ibp.clone()),
                        tightening_cap: Some(Duration::from_mins(5)),
                        collection_mode: CrownIbpCollectionMode::CganComplete,
                        ..Default::default()
                    },
                )
                .expect("typed collection (armed)");
            assert_eq!(
                armed
                    .cached_crown_ibp_collection
                    .hits
                    .load(std::sync::atomic::Ordering::Relaxed),
                1,
                "armed: the COMPLETE Standard entry must serve the typed consumer"
            );
            for (name, bound) in &served.bounds {
                let Some(base) = ibp.get(name) else { continue };
                if base.shape() != bound.shape() {
                    continue;
                }
                for ((&sl, &su), (&bl, &bu)) in bound
                    .lower()
                    .iter()
                    .zip(bound.upper())
                    .zip(base.lower().iter().zip(base.upper()))
                {
                    assert!(
                        sl >= bl && su <= bu,
                        "served map must be shrink-only vs the consumer baseline \
                         (node {name}: served [{sl}, {su}] vs baseline [{bl}, {bu}])"
                    );
                }
            }
        });
    }

    fn deadline_mixed_result(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        mut result: GraphCrownIbpBoundsResult,
    ) -> GraphCrownIbpBoundsResult {
        let ibp = graph.collect_node_bounds(input).expect("fixture IBP map");
        result.bounds.insert(
            "l1".to_string(),
            ibp.get("l1").expect("l1 IBP bound").clone(),
        );
        result.provenance.insert(
            "l1".to_string(),
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
        );
        result
            .fallback_events
            .push(crate::types::CrownIbpFallbackEvent {
                layer_index: 0,
                layer_type: "Linear".to_string(),
                reason: CrownIbpFallbackReason::DeadlineExceeded,
                details: "deterministic mixed-quality cache fixture".to_string(),
            });
        assert!(
            GraphNetwork::crown_ibp_result_crown_count(&result) > 0,
            "fixture must retain at least one genuinely CROWN-tagged target"
        );
        result
    }

    /// #cgan-collection-cache: the key must match ONLY for bit-exact input
    /// boxes — a single-ULP nudge to any endpoint is a different box and must
    /// produce a different key (BaB children may split arbitrarily close to
    /// the root box and must never hit its entry).
    #[test]
    fn collection_cache_key_requires_bit_exact_box() {
        let graph = small_graph();
        let input = unit_box();
        let rebuilt = BoundedTensor::new(input.lower().clone(), input.upper().clone()).unwrap();

        let key = graph.crown_ibp_collection_cache_key(&input, false);
        assert_eq!(
            key,
            graph.crown_ibp_collection_cache_key(&rebuilt, false),
            "bit-identical box must produce the identical key"
        );

        // 1-ULP nudge on one upper endpoint → different key.
        let mut upper = input.upper().clone();
        {
            let slice = upper.as_slice_mut().unwrap();
            slice[0] = ny_tensor::next_up_f32(slice[0]);
        }
        let nudged = BoundedTensor::new(input.lower().clone(), upper).unwrap();
        assert_ne!(
            key,
            graph.crown_ibp_collection_cache_key(&nudged, false),
            "a 1-ULP different box must MISS (different key)"
        );

        // 1-ULP nudge on one lower endpoint → different key.
        let mut lower = input.lower().clone();
        {
            let slice = lower.as_slice_mut().unwrap();
            slice[1] = ny_tensor::next_down_f32(slice[1]);
        }
        let nudged_lower = BoundedTensor::new(lower, input.upper().clone()).unwrap();
        assert_ne!(
            key,
            graph.crown_ibp_collection_cache_key(&nudged_lower, false),
            "a 1-ULP different lower endpoint must MISS"
        );
    }

    /// #cgan-collection-cache: the coverage descriptor (engine presence,
    /// conv-mode policy) is part of the key, so a collection computed under a
    /// different backward policy is never served.
    #[test]
    fn collection_cache_key_includes_coverage_descriptor() {
        let mut graph = small_graph();
        let input = unit_box();

        let base = graph.crown_ibp_collection_cache_key(&input, false);
        assert_ne!(
            base,
            graph.crown_ibp_collection_cache_key(&input, true),
            "engine presence must change the key"
        );

        graph.set_use_patches_mode(false);
        assert_ne!(
            base,
            graph.crown_ibp_collection_cache_key(&input, false),
            "conv-mode policy must change the key"
        );
    }

    #[test]
    fn truncated_cache_gate_requires_exact_one() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!serve_truncated_collection_cache_from_raw(raw));
        }
        assert!(serve_truncated_collection_cache_from_raw(Some("1")));
    }

    #[test]
    fn sparse_relu_rows_gate_requires_exact_one() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!sparse_relu_rows_from_raw(raw));
        }
        assert!(sparse_relu_rows_from_raw(Some("1")));
    }

    #[test]
    fn caller_local_tightening_cap_preserves_and_never_extends_outer_deadline() {
        let now = Instant::now();
        let outer = now + Duration::from_secs(90);
        assert_eq!(
            crown_ibp_tightening_deadline(now, Some(outer), None),
            Some(outer),
            "absent cap must pass the caller's deadline through exactly"
        );
        assert_eq!(
            crown_ibp_tightening_deadline(now, Some(outer), Some(Duration::from_secs(15))),
            Some(now + Duration::from_secs(15))
        );

        let tighter_outer = now + Duration::from_secs(5);
        assert_eq!(
            crown_ibp_tightening_deadline(now, Some(tighter_outer), Some(Duration::from_secs(15))),
            Some(tighter_outer),
            "a local cap must never extend the authoritative outer deadline"
        );
        assert_eq!(
            outer,
            now + Duration::from_secs(90),
            "resolving a tightening deadline must not mutate the caller's value"
        );
    }

    #[test]
    fn hard_deadline_covers_foundational_ibp_without_changing_historical_wrapper() {
        let input = unit_box();
        let expired = Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("test instant supports a one-millisecond subtraction"),
        );

        let hard_error = small_graph()
            .collect_crown_ibp_bounds_dag_with_hard_deadline_and_engine(&input, expired, None)
            .expect_err("hard full-phase deadline must abort before foundational IBP work");
        assert!(
            matches!(hard_error, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {hard_error:?}"
        );

        let historical = small_graph()
            .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, expired, None)
            .expect("historical wrapper must retain its complete IBP fallback contract");
        assert!(
            historical.bounds.contains_key("l1") && historical.bounds.contains_key("relu"),
            "historical wrapper must still return a complete IBP-backed map"
        );
    }

    #[test]
    fn sparse_relu_rows_gate_bypasses_cache_lookup_and_store() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_DISABLE_CROWN_COLLECTION_CACHE");
            env.remove(SPARSE_RELU_ROWS_ENV);

            let graph = small_graph();
            let input = unit_box();
            graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("gate-off collection should populate the cache");
            assert_eq!(
                graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .len(),
                1
            );
            assert_eq!(graph.crown_ibp_collection_cache_hits(), 0);

            env.set(SPARSE_RELU_ROWS_ENV, "1");
            graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("sparse collection should recompute without cache access");
            assert_eq!(
                graph.crown_ibp_collection_cache_hits(),
                0,
                "sparse collection must not look up the existing full-row entry"
            );
            assert_eq!(
                graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .len(),
                1,
                "sparse collection must not store or merge an entry"
            );

            let sparse_first_graph = small_graph();
            sparse_first_graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("sparse first collection should still complete");
            assert!(
                sparse_first_graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .is_empty(),
                "sparse first collection must leave an empty cache"
            );

            env.set(SPARSE_RELU_ROWS_ENV, "0");
            graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("gate-off collection should reuse the original entry");
            assert_eq!(
                graph.crown_ibp_collection_cache_hits(),
                1,
                "exact-zero gate must preserve historical cache lookup"
            );
        });
    }

    #[test]
    fn caller_local_complete_lookup_only_serves_complete_and_never_stores_fresh() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_DISABLE_CROWN_COLLECTION_CACHE");
            env.remove(SPARSE_RELU_ROWS_ENV);

            let graph = small_graph();
            let input = unit_box();
            graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("ordinary collection should populate the cache");
            assert_eq!(
                graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .len(),
                1
            );

            graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        complete_cache_lookup_only: true,
                        ..Default::default()
                    },
                )
                .expect("complete-only policy should serve the complete map");
            assert_eq!(
                graph.crown_ibp_collection_cache_hits(),
                1,
                "complete-only policy should preserve the safe deduplication hit"
            );
            assert_eq!(
                graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .len(),
                1,
                "complete-only lookup must leave the existing entry unchanged"
            );

            let fresh_graph = small_graph();
            fresh_graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        complete_cache_lookup_only: true,
                        ..Default::default()
                    },
                )
                .expect("cache miss must still compute fresh sound bounds");
            assert!(
                fresh_graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .is_empty(),
                "a fresh complete-only call must not store or merge its result"
            );

            let implicitly_isolated_graph = small_graph();
            implicitly_isolated_graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        tightening_cap: Some(Duration::from_secs(15)),
                        ..Default::default()
                    },
                )
                .expect("a capped caller must be isolated even if it omits the cache flag");
            assert!(
                implicitly_isolated_graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")
                    .is_empty(),
                "tightening_cap must force complete-lookup-only/no-store policy internally"
            );

            let expired_graph = small_graph();
            let expired = expired_graph
                .collect_crown_ibp_bounds_dag_with_options(
                    &input,
                    CrownIbpCollectOptions {
                        deadline: Some(Instant::now()),
                        tightening_cap: Some(Duration::from_secs(15)),
                        complete_cache_lookup_only: true,
                        ..Default::default()
                    },
                )
                .expect("an expired tightening deadline must return the complete IBP-backed map");
            assert!(
                expired.bounds.contains_key("l1") && expired.bounds.contains_key("relu"),
                "every graph node must retain a sound bound after capped tightening expires"
            );
            assert!(
                matches!(
                    expired.provenance.get("l1"),
                    Some(BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::DeadlineExceeded
                    ))
                ),
                "the demanded target must carry explicit deadline-fallback provenance"
            );
        });
    }

    #[test]
    fn stable_relu_rows_skip_is_not_time_truncation() {
        let result = GraphCrownIbpBoundsResult {
            bounds: HashMap::new(),
            provenance: HashMap::new(),
            fallback_events: vec![crate::types::CrownIbpFallbackEvent {
                layer_index: 0,
                layer_type: "Linear".to_string(),
                reason: CrownIbpFallbackReason::StableReluRowsSkipped,
                details: "all demanded ReLU rows are IBP-stable".to_string(),
            }],
        };
        assert!(
            GraphNetwork::crown_ibp_result_is_complete(&result),
            "a deterministic all-stable skip must remain cache-complete"
        );
    }

    #[test]
    fn memory_fallback_is_not_complete_cache_authority() {
        let reason = CrownIbpFallbackReason::MemoryBudgetExceeded;
        let mut provenance = HashMap::new();
        provenance.insert(
            "wide".to_string(),
            BoundsProvenance::ForwardFallback(reason),
        );
        let result = GraphCrownIbpBoundsResult {
            bounds: HashMap::new(),
            provenance,
            fallback_events: vec![crate::types::CrownIbpFallbackEvent {
                layer_index: 0,
                layer_type: "Linear".to_string(),
                reason,
                details: "host headroom was temporarily below admission".to_string(),
            }],
        };
        assert!(
            !GraphNetwork::crown_ibp_result_is_complete(&result),
            "released memory may admit a fresh CROWN walk, so the fallback cannot be complete"
        );
    }

    #[test]
    fn partial_crown_row_provenance_stays_truncated_through_cache_serve() {
        let input = unit_box();
        let source = small_graph();
        let completed = source
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("independently completed source map");

        // Seed a different graph's empty cache so the partial entry cannot
        // merge with (and legitimately inherit completeness from) the source's
        // complete entry.
        let graph = small_graph();
        let mut partial = deadline_mixed_result(&graph, &input, completed);
        partial.provenance.insert(
            "l1".to_string(),
            BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded,
            ),
        );
        let event = partial
            .fallback_events
            .last_mut()
            .expect("deadline fixture event");
        event.reason = CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded;
        event.details = "retained 1/2 completed CROWN rows over certified IBP".to_string();

        assert!(
            !GraphNetwork::crown_ibp_result_is_complete(&partial),
            "partial-row salvage is still a time-truncated collection"
        );
        let mut eventless = partial.clone();
        eventless.fallback_events.clear();
        assert!(
            !GraphNetwork::crown_ibp_result_is_complete(&eventless),
            "partial provenance alone must fail closed even if an event is accidentally dropped"
        );
        let stored = graph.seed_truncated_collection_cache_for_test(&input, partial);
        assert!(
            !GraphNetwork::crown_ibp_result_is_complete(&stored),
            "store must not promote partial rows to complete"
        );
        assert!(matches!(
            stored.provenance.get("l1"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
            ))
        ));

        let key = graph.crown_ibp_collection_cache_key(&input, false);
        let fingerprint = graph.crown_ibp_collection_cache_fingerprint(&input, false);
        let served = graph
            .crown_ibp_collection_lookup(key, &fingerprint, true, Some(Duration::from_mins(1)))
            .expect("same-scope truncated cache serve");
        assert!(
            !GraphNetwork::crown_ibp_result_is_complete(&served),
            "cache serve must preserve incomplete authority"
        );
        assert!(matches!(
            served.provenance.get("l1"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
            ))
        ));
        let cache = graph
            .cached_crown_ibp_collection
            .slots
            .read()
            .expect("cache lock");
        assert!(!cache.first().expect("stored partial entry").complete);
    }

    /// #cgan-truncated-scope-output-dim (defect (E) of
    /// `docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md`): a truncated map
    /// that lacks the OUTPUT node's bounds must still build a reuse scope when
    /// no margin publication exists — the historical `?` on the map lookup
    /// silently refused, which kept `scope_match=false` (measured
    /// `current=None` on 7/48 cgan lookups) after every other cache defect was
    /// fixed. With a publication held, the scope must refuse (the documented
    /// UNSOUND shortcut is skipping the guard; the sound fallback is the
    /// strictly more conservative "any publication refuses").
    #[test]
    fn truncated_scope_builds_when_map_lacks_output_node() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(SERVE_TRUNCATED_COLLECTION_CACHE_ENV);
            env.remove("NY_CROWN_CUT_SEGMENT");

            let graph = small_graph();
            let input = unit_box();
            let completed = graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("complete fixture collection");
            let mut result = deadline_mixed_result(&graph, &input, completed);
            // Drop only the OUTPUT node's BOUNDS entry: provenance keeps its
            // Crown tag so `crown_ibp_result_crown_count` stays positive and
            // the scope decision isolates the output-dim lookup.
            assert!(
                result.bounds.remove("relu").is_some(),
                "fixture must have had the output node's bounds to remove"
            );

            let scope =
                graph.crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)));
            assert!(
                scope.is_some(),
                "a map lacking the output node's bounds must still build a scope \
                 when no margin publication exists (defect (E))"
            );

            {
                let _publication =
                    crate::output_margin_seed::MarginOutputSeedGuard::publish(vec![0]);
                assert!(
                    graph
                        .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
                        .is_none(),
                    "with the output dim unknown, ANY live margin publication must \
                     refuse the scope (conservative form of the objective-leak guard)"
                );
            }
            assert!(
                graph
                    .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
                    .is_some(),
                "dropping the publication guard must restore scope construction"
            );
        });
    }

    /// When the map DOES contain the output node, the guard stays
    /// dim-consulted and bit-identical to the historical path: a publication
    /// that would not engage for the node's true width (here dim 2, far below
    /// `MARGIN_SUBSET_MIN_OUTPUT_DIM`) must not refuse the scope.
    #[test]
    fn truncated_scope_with_output_node_keeps_dim_consulted_guard() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(SERVE_TRUNCATED_COLLECTION_CACHE_ENV);
            env.remove("NY_CROWN_CUT_SEGMENT");

            let graph = small_graph();
            let input = unit_box();
            let completed = graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("complete fixture collection");
            let result = deadline_mixed_result(&graph, &input, completed);
            assert!(result.bounds.contains_key("relu"));

            let _publication = crate::output_margin_seed::MarginOutputSeedGuard::publish(vec![0]);
            assert!(
                graph
                    .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
                    .is_some(),
                "a publication below the subset's minimum output width must not \
                 refuse a map whose output-node width is known"
            );
        });
    }

    /// The dark lookup must not promote a malformed partial target set, must
    /// retain exact input identity, and must reject producer/consumer resource
    /// drift even though the cached bounds remain individually sound.
    #[test]
    fn truncated_lookup_refuses_partial_key_and_stale_policy() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                SERVE_TRUNCATED_COLLECTION_CACHE_ENV,
                "NY_DISABLE_CROWN_COLLECTION_CACHE",
                "NY_CROWN_CUT_SEGMENT",
                "NY_CROWN_OBJ_CHUNK",
                "NY_NO_CHUNK_ABORT",
                "NY_NO_CHUNK_GROW",
                "NY_NO_CHUNK_WAVE_PAR",
                "NY_DENSE_BUDGET_MB",
                "NY_PATCHES_BUDGET_SECS",
                "NY_DIM_CAP_SCALE",
                budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV,
                "NY_CONV_PATCHES_COLLECT",
                "NY_CROWN_MEM_CAP_MB",
                "NY_PATCHES_GPU",
                "NY_CONV_SKIP_DEAD_F32",
                "NY_CONVTRANSPOSE_SOUND_F64_GPU",
                "NY_PATCHES_REENTRY_MIN_ROWS",
            ] {
                env.remove(key);
            }

            let mut graph = small_graph();
            let source = small_graph();
            let input = unit_box();
            let truncated = deadline_mixed_result(
                &source,
                &input,
                source
                    .collect_crown_ibp_bounds_dag_with_status(&input)
                    .expect("completed source map"),
            );
            assert!(!GraphNetwork::crown_ibp_result_is_complete(&truncated));
            assert!(graph.crown_ibp_truncated_result_covers_graph(&truncated));

            let key = graph.crown_ibp_collection_cache_key(&input, false);
            let fp = graph.crown_ibp_collection_cache_fingerprint(&input, false);
            let scope = graph
                .crown_ibp_truncated_reuse_scope(&truncated, Some(Duration::from_mins(1)))
                .expect("default cut-free, objective-independent scope");
            assert_eq!(
                scope.policy.chunk_wave_workers, 1,
                "the guarded sequential route must not key cache quality on ambient Rayon width"
            );
            let opposite_prefix_policy = graph
                .crown_ibp_truncated_reuse_scope_with_prefix_cost_policy(
                    &truncated,
                    Some(Duration::from_mins(1)),
                    !scope.policy.prefix_cost_admission_enabled,
                    scope.policy.crown_honest_provenance_enabled,
                )
                .expect("an injected admission policy must still form a valid scope");
            assert_ne!(
                scope.policy.prefix_cost_admission_enabled,
                opposite_prefix_policy.policy.prefix_cost_admission_enabled
            );
            assert!(
                !truncated_scope_can_serve(&scope, &opposite_prefix_policy),
                "the snapshotted admission route must be part of cache identity"
            );

            graph.crown_ibp_collection_store(
                key,
                fp.clone(),
                truncated.clone(),
                Some(scope.clone()),
            );
            let mut lower_authority_scope = scope.clone();
            lower_authority_scope.tightening_budget = Duration::from_secs(30);
            graph.crown_ibp_collection_store(
                key,
                fp.clone(),
                truncated.clone(),
                Some(lower_authority_scope),
            );
            assert_eq!(
                graph
                    .cached_crown_ibp_collection
                    .slots
                    .read()
                    .expect("cache lock")[0]
                    .truncated_reuse_scope
                    .as_ref()
                    .expect("stored scope")
                    .tightening_budget,
                Duration::from_mins(1),
                "same-policy merge must retain the larger producer authority"
            );

            // Budget reuse is directional. A later/depleted caller may reuse a
            // map produced with more authority; a longer-budget caller must get
            // the chance to improve it instead of inheriting short-run quality.
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_secs(59)),)
                    .is_some(),
                "a smaller remaining tightening budget should reuse the producer map"
            );
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_secs(61)),)
                    .is_none(),
                "a larger remaining tightening budget must recompute"
            );
            let mut within_jitter = scope.clone();
            within_jitter.tightening_budget += TRUNCATED_REUSE_BUDGET_JITTER;
            assert!(truncated_scope_can_serve(&scope, &within_jitter));
            within_jitter.tightening_budget += Duration::from_nanos(1);
            assert!(
                !truncated_scope_can_serve(&scope, &within_jitter),
                "only the documented sub-millisecond capture tolerance may exceed producer authority"
            );

            // Every resolved scheduling knob lives in the exact policy, while
            // producer time is compared directionally above.
            macro_rules! assert_policy_miss {
                ($field:ident, $value:expr) => {{
                    let mut changed = scope.clone();
                    changed.policy.$field = $value;
                    assert!(
                        !truncated_scope_can_serve(&scope, &changed),
                        concat!(stringify!($field), " must be cache-identifying")
                    );
                }};
            }
            assert_policy_miss!(
                effective_per_node_floor_bits,
                (f64::from_bits(scope.policy.effective_per_node_floor_bits) + 1.0).to_bits()
            );
            assert_policy_miss!(
                effective_per_node_cap_bits,
                (f64::from_bits(scope.policy.effective_per_node_cap_bits) + 1.0).to_bits()
            );
            assert_policy_miss!(
                per_node_cap_is_explicit,
                !scope.policy.per_node_cap_is_explicit
            );
            assert_policy_miss!(
                patches_budget_bits,
                (f64::from_bits(scope.policy.patches_budget_bits) + 1.0).to_bits()
            );
            assert_policy_miss!(
                patches_budget_is_explicit,
                !scope.policy.patches_budget_is_explicit
            );
            assert_policy_miss!(dim_cap_scale_enabled, !scope.policy.dim_cap_scale_enabled);
            assert_policy_miss!(
                conv_patches_collect_enabled,
                !scope.policy.conv_patches_collect_enabled
            );
            assert_policy_miss!(
                patches_deadline_flat_bias_enabled,
                !scope.policy.patches_deadline_flat_bias_enabled
            );
            assert_policy_miss!(
                crown_honest_provenance_enabled,
                !scope.policy.crown_honest_provenance_enabled
            );
            assert_policy_miss!(
                hopeless_class_skip_enabled,
                !scope.policy.hopeless_class_skip_enabled
            );
            assert_policy_miss!(
                prefix_cost_admission_enabled,
                !scope.policy.prefix_cost_admission_enabled
            );

            // A hash/fingerprint mismatch is a hard miss for truncated maps.
            let mut upper = input.upper().clone();
            let old_upper = upper.as_slice().expect("contiguous")[0];
            upper.as_slice_mut().expect("contiguous")[0] = ny_tensor::next_up_f32(old_upper);
            let other = BoundedTensor::new(input.lower().clone(), upper).unwrap();
            let other_key = graph.crown_ibp_collection_cache_key(&other, false);
            let other_fp = graph.crown_ibp_collection_cache_fingerprint(&other, false);
            assert!(
                graph
                    .crown_ibp_collection_lookup(
                        other_key,
                        &other_fp,
                        true,
                        Some(Duration::from_mins(1))
                    )
                    .is_none(),
                "a different input box must never receive the cached partial map"
            );

            // Removing one graph target makes the entry unservable; production
            // falls through to the ordinary recompute/IBP fallback path.
            {
                let mut slots = graph
                    .cached_crown_ibp_collection
                    .slots
                    .write()
                    .expect("cache lock");
                let entry = slots
                    .iter_mut()
                    .find(|entry| entry.key == key && entry.fingerprint.as_ref() == fp.as_slice())
                    .expect("stored truncated entry");
                let mut partial = (*entry.result).clone();
                partial.bounds.remove("l1");
                entry.result = std::sync::Arc::new(partial);
            }
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "a missing target must recompute, never masquerade as covered"
            );

            // Restore the full result but change a producer policy that does
            // not invalidate the general sound cache. Exact truncated scope
            // matching must still reject the stale quality state.
            {
                let mut slots = graph
                    .cached_crown_ibp_collection
                    .slots
                    .write()
                    .expect("cache lock");
                let entry = slots
                    .iter_mut()
                    .find(|entry| entry.key == key && entry.fingerprint.as_ref() == fp.as_slice())
                    .expect("stored truncated entry");
                entry.result = std::sync::Arc::new(truncated);
                entry.truncated_reuse_scope = Some(scope);
            }
            graph.set_crown_ibp_per_node_time_budget(CrownIbpPerNodeTimeBudget {
                floor_secs: Some(0.25),
                cap_secs: Some(99.0),
            prefix_cost_weighted: false,
                max_target_share_frac: None,
            });
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "changed deadline/resource policy must force recomputation"
            );

            // The aggregate patches budget and dim-aware per-target deadline
            // cap are separate producer resource policies and must also match.
            graph.set_crown_ibp_per_node_time_budget(CrownIbpPerNodeTimeBudget::default());
            env.set("NY_PATCHES_BUDGET_SECS", "17");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "changed aggregate patches budget must force recomputation"
            );
            env.remove("NY_PATCHES_BUDGET_SECS");
            // Keep the effective patches budget equal to the stored default
            // while toggling padded-conv composition. The explicit boolean in
            // the scope must still make this a miss.
            env.set("NY_PATCHES_BUDGET_SECS", "5");
            env.set("NY_CONV_PATCHES_COLLECT", "1");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "changed padded-conv patches policy must force recomputation"
            );
            env.remove("NY_CONV_PATCHES_COLLECT");
            env.remove("NY_PATCHES_BUDGET_SECS");
            env.set("NY_DIM_CAP_SCALE", "0");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "changed dim-aware cap policy must force recomputation"
            );
            env.remove("NY_DIM_CAP_SCALE");
            env.set(budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)),)
                    .is_none(),
                "changed chunk-aware scheduling policy must force recomputation"
            );
            env.remove(budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV);
            graph.set_chunk_wave_parallel_enabled(true);
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)),)
                    .is_none(),
                "changed chunk wave route must force recomputation"
            );
            graph.set_chunk_wave_parallel_enabled(false);
            // Keep the following legacy policy check independent from both
            // the graph-local decision and any ambient interactive override.
            env.remove("NY_NO_CHUNK_WAVE_PAR");
            env.set("NY_CONV_SKIP_DEAD_F32", "0");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "changed lower-level convolution schedule must force recomputation"
            );
            env.remove("NY_CONV_SKIP_DEAD_F32");

            // A cut-segment request is quarantined rather than sharing a
            // cut-free partial collection.
            env.set("NY_CROWN_CUT_SEGMENT", "4");
            assert!(
                graph
                    .crown_ibp_collection_lookup(key, &fp, true, Some(Duration::from_mins(1)))
                    .is_none(),
                "cut state must never reuse a cut-free truncated map"
            );
        });
    }

    #[test]
    fn truncated_scope_refuses_objective_subset_state() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_CROWN_CUT_SEGMENT");
            env.remove("NY_CROWN_OBJ_CHUNK");
            env.remove("NY_NO_CHUNK_ABORT");
            env.remove("NY_NO_CHUNK_GROW");
            env.remove("NY_NO_CHUNK_WAVE_PAR");
            env.remove("NY_DENSE_BUDGET_MB");
            env.remove("NY_PATCHES_BUDGET_SECS");
            env.remove("NY_DIM_CAP_SCALE");
            env.remove(budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV);
            env.remove("NY_CONV_PATCHES_COLLECT");
            env.remove("NY_CROWN_MEM_CAP_MB");
            env.remove("NY_PATCHES_GPU");
            env.remove("NY_CONV_SKIP_DEAD_F32");
            env.remove("NY_CONVTRANSPOSE_SOUND_F64_GPU");
            env.remove("NY_PATCHES_REENTRY_MIN_ROWS");

            let graph = small_graph();
            let input = unit_box();
            let mut result = deadline_mixed_result(
                &graph,
                &input,
                graph
                    .collect_crown_ibp_bounds_dag_with_status(&input)
                    .expect("completed source map"),
            );
            let mut no_crown = result.clone();
            for provenance in no_crown.provenance.values_mut() {
                *provenance =
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded);
            }
            assert!(
                graph
                    .crown_ibp_truncated_reuse_scope(&no_crown, Some(Duration::from_mins(1)))
                    .is_none(),
                "an all-IBP deadline result must never poison a later useful phase"
            );
            let wide = BoundedTensor::new(
                Array1::from_elem(600, -1.0_f32).into_dyn(),
                Array1::from_elem(600, 1.0_f32).into_dyn(),
            )
            .unwrap();
            result.bounds.insert("relu".to_string(), wide);
            assert!(
                graph
                    .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
                    .is_some(),
                "wide output without a published objective remains objective-independent"
            );
            let _objective = crate::output_margin_seed::MarginOutputSeedGuard::publish(vec![1, 7]);
            assert!(
                graph
                    .crown_ibp_truncated_reuse_scope(&result, Some(Duration::from_mins(1)))
                    .is_none(),
                "objective-subset state is deliberately uncacheable"
            );
        });
    }

    #[test]
    fn complete_entry_dominates_mismatched_truncated_scope() {
        ny_test_utils::env::with_env_edits(|env| {
            for key in [
                SERVE_TRUNCATED_COLLECTION_CACHE_ENV,
                "NY_CROWN_CUT_SEGMENT",
                "NY_CROWN_OBJ_CHUNK",
                "NY_NO_CHUNK_ABORT",
                "NY_NO_CHUNK_GROW",
                "NY_NO_CHUNK_WAVE_PAR",
                "NY_DENSE_BUDGET_MB",
                "NY_PATCHES_BUDGET_SECS",
                "NY_DIM_CAP_SCALE",
                budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV,
                "NY_CONV_PATCHES_COLLECT",
                "NY_CROWN_MEM_CAP_MB",
                "NY_PATCHES_GPU",
                "NY_CONV_SKIP_DEAD_F32",
                "NY_CONVTRANSPOSE_SOUND_F64_GPU",
                "NY_PATCHES_REENTRY_MIN_ROWS",
            ] {
                env.remove(key);
            }

            let graph = small_graph();
            let input = unit_box();
            let complete = graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .expect("complete collection");
            assert!(GraphNetwork::crown_ibp_result_is_complete(&complete));

            let mixed = deadline_mixed_result(&graph, &input, complete);
            let scope = graph
                .crown_ibp_truncated_reuse_scope(&mixed, Some(Duration::from_mins(1)))
                .expect("mixed scope");
            let key = graph.crown_ibp_collection_cache_key(&input, false);
            let fp = graph.crown_ibp_collection_cache_fingerprint(&input, false);
            let merged = graph.crown_ibp_collection_store(key, fp.clone(), mixed, Some(scope));

            assert!(
                GraphNetwork::crown_ibp_result_is_complete(&merged),
                "a fresh partial collection must not replace a cached complete map"
            );
            let served = graph
                .crown_ibp_collection_lookup(key, &fp, false, None)
                .expect("complete entry remains servable");
            assert!(GraphNetwork::crown_ibp_result_is_complete(&served));
        });
    }
}

/// `#cgan-narrow-target-budget` regression suite.
///
/// The ceiling is a SCHEDULING policy: it decides which demanded targets spend
/// collection budget, and a declined target keeps the caller-supplied baseline
/// enclosure. These tests pin both halves of that contract — that the ceiling
/// really does change which targets are tightened (otherwise the lever is dead
/// capability), and that every published bound still CONTAINS the reachable
/// set on both sides of the switch (otherwise the lever is a false proof).
#[cfg(test)]
mod narrow_target_budget_tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::{arr1, Array1, Array2};

    /// 2 -> 8 -> ReLU -> 3 -> ReLU -> 1.
    ///
    /// The 8-row hidden target stands in for a wide cGAN generator node and the
    /// 3-row target for a narrow discriminator node, so a ceiling between them
    /// separates the two exactly as it does on the official rows.
    fn wide_then_narrow_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        // The wide target must sit DOWNSTREAM OF A RELU: a linear layer applied
        // straight to the input box is bounded EXACTLY by interval arithmetic,
        // so CROWN has nothing to tighten and the fixture is inert — which is
        // what the `fixture is inert` guard below caught when this was first
        // written. SIX rows, deliberately ABOVE the ceiling=4 the test uses: at
        // or under it this pre-layer would itself be demanded, and refining it
        // would tighten what flows into `wide`, so `wide` would come back
        // tighter than baseline while genuinely being declined.
        let w0 = Array2::from_shape_fn((6, 2), |(r, c)| 0.9 - 0.35 * (r as f32) + 0.55 * (c as f32));
        let l0 = LinearLayer::new(w0, Some(Array1::from_shape_fn(6, |r| 0.1 - 0.08 * r as f32)))
            .unwrap();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(l0)));
        graph.add_node(GraphNode::new(
            "pre_relu",
            Layer::ReLU(ReLULayer),
            vec!["pre".into()],
        ));
        let w1 = Array2::from_shape_fn((8, 6), |(r, c)| {
            0.6 - 0.17 * (r as f32) + 0.43 * (c as f32) - 0.09 * ((r * c) as f32)
        });
        let l1 = LinearLayer::new(w1, Some(Array1::from_shape_fn(8, |r| 0.05 * r as f32 - 0.2)))
            .unwrap();
        graph.add_node(GraphNode::new(
            "wide",
            Layer::Linear(l1),
            vec!["pre_relu".into()],
        ));
        graph.add_node(GraphNode::new(
            "wide_relu",
            Layer::ReLU(ReLULayer),
            vec!["wide".into()],
        ));
        let w2 = Array2::from_shape_fn((3, 8), |(r, c)| {
            0.31 - 0.11 * (r as f32) + 0.07 * (c as f32) - 0.05 * ((r + c) as f32)
        });
        let l2 =
            LinearLayer::new(w2, Some(Array1::from_shape_fn(3, |r| 0.1 - 0.04 * r as f32))).unwrap();
        graph.add_node(GraphNode::new(
            "narrow",
            Layer::Linear(l2),
            vec!["wide_relu".into()],
        ));
        graph.add_node(GraphNode::new(
            "narrow_relu",
            Layer::ReLU(ReLULayer),
            vec!["narrow".into()],
        ));
        let l3 = LinearLayer::new(
            Array2::from_shape_fn((1, 3), |(_, c)| 0.9 - 0.3 * c as f32),
            Some(arr1(&[0.05_f32])),
        )
        .unwrap();
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(l3),
            vec!["narrow_relu".into()],
        ));
        graph.set_output("out");
        graph
    }

    fn box_input() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-0.75_f32, -0.5]).into_dyn(),
            arr1(&[0.6_f32, 0.85]).into_dyn(),
        )
        .unwrap()
    }

    fn collect(graph: &GraphNetwork, max_rows: Option<usize>) -> HashMap<String, BoundedTensor> {
        let input = box_input();
        let ibp = graph.collect_node_bounds(&input).expect("IBP baseline");
        graph
            .collect_crown_ibp_bounds_dag_with_options(
                &input,
                CrownIbpCollectOptions {
                    deadline: Some(Instant::now() + Duration::from_mins(1)),
                    precomputed_ibp: Some(ibp),
                    tightening_cap: Some(Duration::from_mins(5)),
                    collection_mode: CrownIbpCollectionMode::CganComplete,
                    cgan_narrow_target_max_rows: max_rows,
                    ..Default::default()
                },
            )
            .expect("typed cGAN collection")
            .bounds
    }

    /// An independently computed reachable envelope: the true per-node
    /// activation at each of a grid of concrete input points, taken through the
    /// pointwise activation collector (`#cgan-eval`), which is exact to ~ULP.
    ///
    /// This is the analogue of the ORT-sampled envelope used to qualify the
    /// lever offline. It is an INNER approximation of the reachable set, so a
    /// bound that fails to contain it is unsound beyond any doubt.
    fn reachable_envelope(graph: &GraphNetwork) -> HashMap<String, (Array1<f32>, Array1<f32>)> {
        let input = box_input();
        let lo = input.lower().to_owned();
        let hi = input.upper().to_owned();
        let steps = 21;
        let mut env: HashMap<String, (Array1<f32>, Array1<f32>)> = HashMap::new();
        for i in 0..steps {
            for j in 0..steps {
                let t0 = i as f32 / (steps - 1) as f32;
                let t1 = j as f32 / (steps - 1) as f32;
                let pt = arr1(&[
                    lo[0] + t0 * (hi[0] - lo[0]),
                    lo[1] + t1 * (hi[1] - lo[1]),
                ])
                .into_dyn();
                let point = BoundedTensor::new(pt.clone(), pt).unwrap();
                let acts = graph
                    .collect_node_activations_pointwise(&point, None)
                    .expect("pointwise activations");
                for (name, value) in acts {
                    let v: Array1<f32> =
                        value.center().iter().copied().collect::<Vec<_>>().into();
                    match env.get_mut(&name) {
                        Some((mn, mx)) => {
                            for (k, x) in v.iter().enumerate() {
                                if *x < mn[k] {
                                    mn[k] = *x;
                                }
                                if *x > mx[k] {
                                    mx[k] = *x;
                                }
                            }
                        }
                        None => {
                            env.insert(name, (v.clone(), v));
                        }
                    }
                }
            }
        }
        env
    }

    /// Assert every published node bound CONTAINS the reachable envelope.
    ///
    /// Deliberately a containment assertion and never a finiteness one: a
    /// tighter bound that excludes a reachable point is exactly the false-proof
    /// shape this suite exists to catch, and `is_finite()` stays green through
    /// it.
    fn assert_contains(
        bounds: &HashMap<String, BoundedTensor>,
        env: &HashMap<String, (Array1<f32>, Array1<f32>)>,
        label: &str,
    ) {
        let tol = 1e-4_f32;
        let mut checked = 0usize;
        for (name, bound) in bounds {
            let Some((mn, mx)) = env.get(name) else {
                continue;
            };
            let lower: Vec<f32> = bound.lower().iter().copied().collect();
            let upper: Vec<f32> = bound.upper().iter().copied().collect();
            if lower.len() != mn.len() {
                continue;
            }
            checked += 1;
            for k in 0..lower.len() {
                assert!(
                    lower[k] <= mn[k] + tol,
                    "{label}: node '{name}' row {k} lower bound {} EXCLUDES reachable min {} \
                     (a tighter bound that excludes a reachable point is a false proof)",
                    lower[k],
                    mn[k],
                );
                assert!(
                    upper[k] >= mx[k] - tol,
                    "{label}: node '{name}' row {k} upper bound {} EXCLUDES reachable max {}",
                    upper[k],
                    mx[k],
                );
            }
        }
        assert!(
            checked >= 4,
            "{label}: containment check covered only {checked} nodes — the envelope join \
             matched nothing, so this assertion is a no-op rather than a pass"
        );
    }

    /// The ceiling must actually change which targets are tightened. Without
    /// this the lever could ship as dead capability with a green suite.
    #[test]
    fn narrow_ceiling_declines_the_wide_target_and_retains_the_narrow_one() {
        let graph = wide_then_narrow_graph();
        let baseline = graph
            .collect_node_bounds(&box_input())
            .expect("IBP baseline");
        let all_demanded = collect(&graph, None);
        let ceiling = collect(&graph, Some(4));

        let wide_base = baseline["wide"].max_width();
        let wide_all = all_demanded["wide"].max_width();
        let wide_ceiling = ceiling["wide"].max_width();
        let narrow_base = baseline["narrow"].max_width();
        let narrow_ceiling = ceiling["narrow"].max_width();

        assert!(
            wide_all < wide_base - 1e-6,
            "fixture is inert: the 8-row target must be tightenable when demanded \
             (baseline {wide_base}, all-demanded {wide_all})"
        );
        assert!(
            (wide_ceiling - wide_base).abs() <= 1e-6,
            "ceiling=4 must DECLINE the 8-row target and leave it at the supplied \
             baseline (baseline {wide_base}, got {wide_ceiling})"
        );
        assert!(
            narrow_ceiling < narrow_base - 1e-6,
            "ceiling=4 must still tighten the 3-row target (baseline {narrow_base}, \
             got {narrow_ceiling})"
        );
    }

    /// Soundness gate. Both sides of the switch must contain the reachable set.
    #[test]
    fn every_published_bound_contains_the_reachable_envelope() {
        let graph = wide_then_narrow_graph();
        let env = reachable_envelope(&graph);
        assert_contains(&collect(&graph, None), &env, "all-demanded");
        assert_contains(&collect(&graph, Some(4)), &env, "ceiling=4");
        assert_contains(&collect(&graph, Some(1)), &env, "ceiling=1");
    }

    /// The declined target is published at an enclosure the collector already
    /// held, so the ceiling can never widen a bound relative to the baseline it
    /// was seeded with.
    #[test]
    fn ceiling_never_widens_a_published_bound_below_its_baseline() {
        let graph = wide_then_narrow_graph();
        let baseline = graph
            .collect_node_bounds(&box_input())
            .expect("IBP baseline");
        for ceiling in [Some(1), Some(4), Some(8), None] {
            let published = collect(&graph, ceiling);
            for (name, bound) in &published {
                let Some(base) = baseline.get(name) else {
                    continue;
                };
                if base.lower().len() != bound.lower().len() {
                    continue;
                }
                for (k, (b, p)) in base.lower().iter().zip(bound.lower().iter()).enumerate() {
                    assert!(
                        *p >= *b - 1e-4,
                        "ceiling={ceiling:?}: node '{name}' row {k} lower {p} is BELOW the \
                         seeded baseline {b}; the published map must stay shrink-only"
                    );
                }
                for (k, (b, p)) in base.upper().iter().zip(bound.upper().iter()).enumerate() {
                    assert!(
                        *p <= *b + 1e-4,
                        "ceiling={ceiling:?}: node '{name}' row {k} upper {p} is ABOVE the \
                         seeded baseline {b}; the published map must stay shrink-only"
                    );
                }
            }
        }
    }
}
