// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};
use ny_core::GemmEngine;
use std::collections::HashMap;
use std::time::Instant;

#[cfg(test)]
use crate::bounds::LinearBounds;

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
    /// Pre-computed IBP bounds (#3596). When provided, skips the internal IBP
    /// forward pass, allowing the full deadline budget for CROWN tightening.
    /// On deep graphs (e.g., ECAPA-TDNN ~188 nodes), the IBP forward pass can
    /// consume the entire deadline budget.
    pub precomputed_ibp: Option<HashMap<String, BoundedTensor>>,
    /// Minimum IBP width to trigger CROWN tightening at a node (#3499). Nodes
    /// with IBP `max_width()` below this threshold are skipped, saving budget
    /// for deeper layers where CROWN tightening is most impactful.
    pub min_width_to_tighten: Option<f32>,
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
    /// The collection is deterministic given (graph, input, options) except
    /// for its TIME-budget truncation, so the deadline is deliberately NOT
    /// part of the key: a complete (untruncated) map is served to every later
    /// same-key call regardless of that call's remaining budget — this is the
    /// whole point (on cgan_2023 the alpha warmup re-ran the collection up to
    /// 5x and the last, most-truncated map reached BaB instead of the complete
    /// precheck map). Requests whose OPTIONS change what would be computed
    /// (caller-supplied IBP maps, width-threshold skips) bypass the cache
    /// entirely — every duplicated producer in the verify flow (disjunctive
    /// precheck, alpha-warmup reference bounds, iteration-0 output backward,
    /// sequential-alpha re-reference) uses default options.
    pub(crate) fn collect_crown_ibp_bounds_dag_with_options(
        &self,
        input: &BoundedTensor,
        options: CrownIbpCollectOptions<'_>,
    ) -> Result<GraphCrownIbpBoundsResult> {
        // #lsnc-determinism-diagnostic (task #36): default-OFF kill-switch to
        // physically bypass the input-keyed CROWN-IBP collection cache (both
        // lookup and store/merge). Set `NY_DISABLE_CROWN_COLLECTION_CACHE=1` to
        // reduce the pipeline to the pre-#cgan-collection-cache path (every
        // duplicated producer recomputes; nothing is served or merged). This
        // exists so a reproducibility investigation can A/B the cache without a
        // rebuild. Default-off => byte-identical to the cached path, so the cgan
        // beneficiary (the cache's raison d'être) is untouched unless the env is
        // explicitly set. Read once per call; the collection is not a hot loop.
        let cache_disabled =
            std::env::var_os("NY_DISABLE_CROWN_COLLECTION_CACHE").is_some_and(|v| v == "1");
        let cacheable = !cache_disabled
            && options.precomputed_ibp.is_none()
            && options.min_width_to_tighten.is_none();
        let cache_key = cacheable.then(|| {
            (
                self.crown_ibp_collection_cache_key(input, options.engine.is_some()),
                self.crown_ibp_collection_cache_fingerprint(input, options.engine.is_some()),
            )
        });
        if let Some((key, ref fp)) = cache_key {
            if let Some(hit) = self.crown_ibp_collection_lookup(key, fp) {
                return Ok(hit);
            }
        }

        let ibp_bounds = match options.precomputed_ibp {
            Some(precomputed) => precomputed,
            None => self.collect_node_bounds_with_engine(input, options.engine)?,
        };
        let fresh = self.collect_crown_ibp_bounds_core_inner(
            input,
            ibp_bounds,
            options.deadline,
            options.engine,
            options.min_width_to_tighten,
        )?;
        match cache_key {
            Some((key, fp)) => Ok(self.crown_ibp_collection_store(key, fp, fresh)),
            None => Ok(fresh),
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
        let mut fp = Vec::with_capacity(8 * input.lower().len() + 8 * input.shape().len() + 64);
        fp.extend_from_slice(b"ny.crown-ibp-coverage.v2\0");
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
        fp
    }

    /// Whether a collection result was truncated by a TIME budget (overall
    /// deadline, per-node share, or the aggregate patches budget). Only these
    /// reasons depend on wall-clock luck; every other fallback reason
    /// (memory budget, unsupported op, shape/NaN, demand skip) is a
    /// deterministic function of (graph, input, options) and recurs
    /// identically on a re-run.
    fn crown_ibp_result_is_complete(result: &GraphCrownIbpBoundsResult) -> bool {
        !result.fallback_events.iter().any(|event| {
            matches!(
                event.reason,
                CrownIbpFallbackReason::DeadlineExceeded
                    | CrownIbpFallbackReason::PerNodeDeadlineExceeded
                    | CrownIbpFallbackReason::PatchesBudgetExceeded
            )
        })
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

    /// Serve a cached collection for `key` when the cached entry is COMPLETE
    /// (a re-run could never compute more). A truncated cached entry is not
    /// served here — the caller runs the collection, and
    /// [`Self::crown_ibp_collection_store`] then keeps/returns whichever map
    /// is more complete.
    fn crown_ibp_collection_lookup(
        &self,
        key: u64,
        fingerprint: &[u8],
    ) -> Option<GraphCrownIbpBoundsResult> {
        let guard = self.cached_crown_ibp_collection.slots.read().ok()?;
        // Exact (key, fingerprint) match across the retained entries. The
        // fingerprint disambiguates a u64 hash collision between different boxes,
        // so multiple same-hash entries coexist and only the TRUE box is ever
        // served (#cgan-collection-multislot).
        let entry = guard
            .iter()
            .find(|e| e.key == key && e.fingerprint.as_ref() == fingerprint)?;
        // A truncated (time-budget-cut) map is never served as a complete one:
        // the caller re-runs and `crown_ibp_collection_store` keeps whichever map
        // is more complete (soundness: respect the `complete` flag).
        if !entry.complete {
            debug!(
                "CROWN-IBP DAG: cached collection for key {key:016x} is truncated \
                 ({} crown nodes) — re-running (graph={:p})",
                entry.crown_count,
                std::ptr::from_ref(self),
            );
            return None;
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
        let merged = match existing_idx {
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
                // Completeness: the merged map dominates a complete side, so
                // it is complete if EITHER side is. `crown_ibp_result_is_complete`
                // reads fallback events, so adopt the complete side's (time-
                // event-free) event list when only the cached side was complete.
                if entry.complete && !fresh_complete {
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
        )
    }

    /// Collector-specific backward CROWN that uses patches mode for spatial
    /// Conv2d targets even when the global conv_mode is matrix (#3813).
    ///
    /// The CROWN-IBP intermediate bounds collector doesn't use cutting planes,
    /// so the matrix-mode constraint (required for BaB cuts) doesn't apply.
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
    use ndarray::{arr1, arr2};

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
}
