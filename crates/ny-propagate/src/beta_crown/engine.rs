// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! β-CROWN verifier implementation.
//!
//! The core branch-and-bound verification engine, split into focused
//! submodules (core execution, branching, bounds, cuts, and optimization).

mod backward;
mod bounds;
mod branching;
mod complete_clip_engine;
pub(in crate::beta_crown::engine) mod complete_clip_intermediate;
mod complete_clip_precomputed;
mod core;
mod cut_gate;
mod cuts;
// `pub(crate)` only so the crate root can re-export the sequential
// clip-interm-domain CAPABILITY predicate (`domain::clip`) — the CLI's
// preset/engine contract check must read the engine's own answer rather than
// duplicate a constant that could drift. Nothing else here is public.
pub(crate) mod domain;
mod domain_results;
pub(crate) mod graph;
mod input_split;
mod joint_margin;
mod optimization;
mod pgd;
mod relaxed_clip;
#[cfg(test)]
mod relaxed_clip_multi_objective;
mod tensor_ext;
mod verify_phases;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::beta_crown::config::BetaCrownConfig;
use crate::network::ResnetSegmentSkeleton;
use crate::GraphNetwork;
pub use graph::domain_batch::{
    GraphDomainBatchCallerLane, GraphDomainBatchMetricsSink, GraphDomainBatchRecord,
};
pub use graph::input_split::metrics::{
    DenseSpecReboundMode, InputSplitBatchRecord, InputSplitMetricsSink,
};
pub use graph::propagation::batched::{BatchedSpecBackwardResult, DenseSpecStageTiming};
pub use graph::DomainSpecCrownResult;
pub use joint_margin::JointMarginCloser;

#[derive(Default)]
pub struct BetaCrownVerifier {
    /// Configuration.
    pub config: BetaCrownConfig,
    engine: Option<Arc<dyn GemmEngine>>,
    /// `#split-hint`: per-input influence magnitudes derived ONCE from the ROOT
    /// plane, used by the input-split dimension selector when a domain has no
    /// plane of its own.
    ///
    /// # Why this exists
    /// `input_split_scores` scores a dimension by `width` alone whenever
    /// `domain.linear_bounds` is `None` — an OUTPUT-BLIND choice. Measured on
    /// cgan `upsample_prop_0`: 16 consecutive batches with the queue pinned at
    /// exactly 2640, `head_depth` climbing 78 -> 93, and `best_gap` FROZEN at
    /// -0.0001. Fifteen levels of splitting bought zero bound improvement. The
    /// root HAS a plane; children are built with `linear_bounds: None` and throw
    /// that information away.
    ///
    /// # ⛔ Why it is a SEPARATE field and not a synthesized `linear_bounds`
    /// `domain.linear_bounds` is NOT selector-only. `screen_child` feeds it to
    /// `clip_with_precomputed_linear` / `complete_clip_with_precomputed_specs`
    /// and, on `clip_outcome.verified`, does `domains_verified += 1; return` —
    /// DECLARING THE DOMAIN VERIFIED — then tightens the child input box from
    /// it. Putting an uncertified vector there is a false-`unsat` generator.
    /// **This field is read at exactly one site, the scorer, and must never be
    /// consumed by a bound, a clip, or a verification.**
    ///
    /// # Soundness
    /// WHICH dimension is split is schedule-only: splitting any dimension is
    /// sound, so this needs no certification. `OnceLock` so it is written once
    /// at the root and read freely thereafter without `&mut`.
    root_split_hint: std::sync::OnceLock<Vec<f32>>,
    input_split_metrics_sink: Option<Arc<dyn InputSplitMetricsSink>>,
    graph_domain_batch_metrics_sink: Option<Arc<dyn GraphDomainBatchMetricsSink>>,
    /// Optional per-domain joint-margin closer for same-LHS conjunctive
    /// (max-diff) input-split BaB. Runtime-only (not part of serializable
    /// config): holds the maxpool-stripped signed-diff net used to certify a
    /// tighter joint lower bound than the single-conjunct MaxPool relaxation.
    /// See `joint_margin::JointMarginCloser`. `None` on every other path, so
    /// this is byte-for-byte inert unless the CLI attaches it for a same-LHS
    /// max-diff reduction (acasxu prop_2/3/4).
    joint_margin_closer: Option<Arc<JointMarginCloser>>,
    /// Concrete original-singleton source continuation. It never imports
    /// generic leaves as proof and shares the original invocation authority.
    input_head_continuation: Option<Arc<crate::margin_row::InputHeadContinuation>>,
    /// Optional Graph-MIP LEAF oracle (increment 6,
    /// `docs/GRAPH_MIP_LEAF_SOLVER.md`): consulted by the graph ReLU-split
    /// BaB right before an UNDECIDED child is requeued, to decide the
    /// subdomain exactly (split premises pinned, certified-UNSAT-only
    /// admission). Runtime-only and attached by the MIP-enabled CLI unless
    /// `NY_GRAPH_MIP_LEAF=0`; `None` is byte-for-byte inert.
    graph_mip_leaf_oracle: Option<Arc<dyn crate::beta_crown::graph_mip_leaf::GraphMipLeafOracle>>,
    /// #vit-margin-alpha-complex (P8): optional INDEPENDENT forward (ONNX
    /// Runtime, attached by ny-cli only when `bab.mo_root_ort_adjudication`
    /// is armed) that the default-off containment fence uses to adjudicate a
    /// ROOT-terminal multi-objective `Verified`. Runtime-only; `None` (never
    /// attached) is byte-for-byte inert. Preserved across `with_config_from`.
    root_adjudication_oracle:
        Option<Arc<dyn crate::beta_crown::root_adjudication::RootAdjudicationOracle>>,
    /// #relusplitter-lp-refine (wave-6 C7): optional ROOT intermediate-bound
    /// refiner (ny-cli's rigorous LP refiner, attached only when
    /// `bab.mip_refine_intermediates` is armed). The multi-objective graph
    /// root intersects its certified candidates shrink-only into the stored
    /// node boxes before the root objective pass and the BaB setup. Runtime-
    /// only; `None` (never attached) is byte-for-byte inert. Preserved across
    /// `with_config_from`.
    root_interm_refiner: Option<Arc<dyn crate::beta_crown::root_interm_refine::RootIntermRefiner>>,
    /// Exact root intermediate-bound result shared by deterministic restarts of
    /// one top-level grouped-disjunctive verification call. The CLI attaches a
    /// fresh cache for explicitly typed cGAN roots and under the ordinary
    /// `NY_DISJUNCTIVE_RESTART_ROOT_CACHE=1` opt-in; other verifiers keep
    /// `None`. `with_config_from` shares the same `Arc` so a completed first
    /// restart can serve the second restart's configured graph clone without
    /// escaping the top-level call.
    disjunctive_restart_root_cache:
        Option<Arc<graph::input_split::root_bounds::InputSplitRootBoundsCache>>,
    /// #extract-skeleton increment 3: verifier-lifetime cross-batch cache of
    /// static extraction skeletons (see [`ResnetSkeletonCache`]). REUSE only —
    /// a hit is the same skeleton the increment-2 per-call build would have
    /// produced, every use re-validates `matches_graph` + `cache_key` before
    /// folding, and every refusal still falls back to the legacy per-domain
    /// extraction (the fail-closed spine); the cache never changes what a
    /// fold produces.
    pub(crate) skeleton_cache: ResnetSkeletonCache,
    /// Exact unconstrained root node bounds used to produce reusable
    /// Complete-Clipping affine templates. One entry per verifier, keyed by
    /// live graph scope and every input-box bit.
    pub(crate) complete_clip_root_bounds_cache:
        graph::propagation::batched::interm_refine::CompleteClipRootBoundsCache,
    /// Dynamic outer-BaB deadline scopes.
    ///
    /// This was introduced for optional Complete Clipping work, but is also
    /// the authoritative boundary for nested graph-BaB α/β optimization,
    /// branch scoring, and dense child propagation. The root/config deadline
    /// may be later than a caller's ledger-reserved BaB slice.
    pub(crate) complete_clip_deadline_overrides:
        graph::propagation::batched::interm_refine::CompleteClipDeadlineOverrides,
    /// #gather-score (boxlift charter Inc 4, DARK `NY_MO_GATHER_SCORE=1`):
    /// advisory per-domain branch scores harvested from the wide-β lane's
    /// already-paid A_lower gather, keyed by split-set fingerprint. Advisory
    /// only — a hit reorders branch candidates, a miss falls back to the
    /// shipped scorer; entries never affect bounds or verdicts.
    pub(crate) gather_score_cache: graph::propagation::batched::gather_score::GatherScoreCache,
    /// Bounded adaptive-depth kFSB evaluation budget. The dark SHADOW, SELECT,
    /// and COMMIT hooks share one verifier-lifetime attempt: they may price one
    /// top-three tree portfolio, but can never turn every BaB wave into an
    /// unbounded diagnostic or commit workload.
    pub(crate) adaptive_depth_shadow_fired: std::sync::atomic::AtomicBool,
    /// Observation-only precision-consistent kFSB shadow budget. The exact
    /// `NY_MO_KFSB_F64_SHADOW=1` gate may re-score one wave's worst parent's
    /// post-f32 top-three portfolio with the certified CPU-f64 lineage fold.
    /// It is deliberately one-shot per verifier so diagnostic work cannot
    /// become a per-wave tax.
    pub(crate) kfsb_f64_shadow_fired: std::sync::atomic::AtomicBool,
    /// Observation-only root-attribution portfolio comparison budget. The
    /// diagnostic may claim at most one eligible wave over this verifier's
    /// lifetime; descendant waves and later eligible waves remain on the
    /// ordinary selector without minting another private deadline.
    pub(crate) attribution_diag_fired: std::sync::atomic::AtomicBool,
    /// #graph-bab-cover-ledger: the completeness-ledger frame a ledgered graph
    /// BaB driver arms for one invocation (`certify::graph`). Idle — one
    /// relaxed atomic load per recording site — unless a driver armed it.
    /// Fresh per verifier (`with_config_from` builds a new one), so a
    /// sub-verifier never records into its parent's frame.
    pub(crate) graph_cover: crate::certify::graph::GraphCoverSlot,
    /// #la-warm-frontier: engagement counters of the frontier-complete lA warm
    /// start (captures, resumes, refusals). Shared with every sub-verifier
    /// built by `with_config_from`, so one lane's `[la-warm-frontier]` line
    /// counts its sub-verifiers' seeds too. Telemetry only.
    pub(crate) la_warm_frontier_counters:
        Arc<graph::propagation::warm_frontier::WarmFrontierCounters>,
    /// #kfsb-production: the armed single-shot kFSB scorer's counters, its
    /// per-decision line cap and the lane's R4 fan-out lease. Shared with
    /// every sub-verifier built by `with_config_from` (one lane, one
    /// `[kfsb-single-shot] summary` line, one line budget). Telemetry and
    /// admission only.
    pub(crate) kfsb_single_shot_telemetry:
        Arc<graph::gpu_bab::kfsb_fanout::KfsbSingleShotTelemetry>,
    /// #vit-v3-bankable: the multi-objective cover-certification run state
    /// (replay base, β hints, leaf-oracle closes) paired with a rows frame on
    /// `graph_cover`. Idle unless a disjunctive multi-objective call armed it
    /// (`bab.mo_cover_certify`). Fresh per verifier, like `graph_cover`.
    pub(crate) mo_cover: graph::mo_cover::MoCoverState,
}

/// #extract-skeleton increment 3: verifier-lifetime cache of
/// [`ResnetSegmentSkeleton`]s, keyed by `(start_node, allow_pure_chain)` —
/// the design's cross-batch tier (docs/EXTRACT_SKELETON_DESIGN.md §1,
/// "Increment 3"). Key space is tiny: one start node by default, +k with
/// interm-refine seeds.
///
/// Concurrency contract: preps run inside rayon fan-outs, so the `Mutex` is
/// held for MAP OPERATIONS ONLY — the build always runs OUTSIDE the lock
/// (double-checked insert) and never blocks concurrent callers. Two callers
/// racing a cold (or stale) key may therefore BOTH build; the last insert
/// wins. That is sound and cost-only: the skeleton's static content is
/// (topology, weights, start_node)-determined (the increment-1 oracles prove
/// fold-correctness regardless of exemplar), so racing builds are equivalent,
/// and every consumer re-validates the skeleton it got before folding.
#[derive(Default)]
pub(crate) struct ResnetSkeletonCache {
    inner: Mutex<HashMap<(String, bool), Arc<ResnetSegmentSkeleton>>>,
}

impl ResnetSkeletonCache {
    /// Serve the skeleton for `(start_node, allow_pure_chain)`, building via
    /// `build` on a miss. Every hit re-validates
    /// [`ResnetSegmentSkeleton::matches_graph`] against the CURRENT `graph`:
    /// a stale entry (conv geometry / broadcasts were baked from another
    /// graph's bounds shapes) is never served — it is rebuilt and replaced.
    /// `build` returning `None` is NOT cached (no negative caching): the
    /// refusal stays per-call and the caller falls back to the legacy
    /// per-domain extraction (fail closed). The `NY_EXTRACT_SKELETON=0`
    /// kill-switch is honored HERE as well as inside the build, so a WARM
    /// cache stops serving the moment the gate is off (wholesale revert).
    pub(crate) fn get_or_build(
        &self,
        graph: &GraphNetwork,
        start_node: &str,
        allow_pure_chain: bool,
        build: impl FnOnce() -> Option<ResnetSegmentSkeleton>,
    ) -> Option<Arc<ResnetSegmentSkeleton>> {
        if !crate::network::extract_skeleton_enabled() {
            return None;
        }
        let key = (start_node.to_owned(), allow_pure_chain);
        {
            let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = map.get(&key) {
                if hit.matches_graph(graph) {
                    return Some(Arc::clone(hit));
                }
                // Stale for THIS graph: fall through to rebuild; the insert
                // below replaces the entry.
            }
        }
        // Build OUTSIDE the lock (rayon fan-outs stay unblocked); a `None`
        // build caches nothing.
        let built = Arc::new(build()?);
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        // Double-checked insert resolution: a racer may have inserted while we
        // built — LAST write wins (see the type doc: racing builds are
        // equivalent, so this is cost-only, never correctness).
        map.insert(key, Arc::clone(&built));
        Some(built)
    }
}

#[derive(Debug)]
struct RelaxedClipOutcome {
    bounds: BoundedTensor,
    verified: bool,
}

/// Grouped-safe outcome from joint multi-spec relaxed clipping.
///
/// Unlike `RelaxedClipOutcome`, this does NOT collapse per-row lower bounds into
/// a single `verified` boolean. Grouped/disjunctive callers need the per-row
/// lower bounds for their own OR-within-clause and AND-across-clauses reduction.
///
/// Production code now uses `batched_relaxed_clip_from_flat` with
/// direct concretization, so this struct is only needed by test helpers.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MultiSpecRelaxedClipOutcome {
    /// The clipped input bounds.
    pub(crate) bounds: BoundedTensor,
    /// True when joint clipping made the child box empty (x_l > x_u on some dim).
    pub(crate) infeasible_after_clip: bool,
    /// Per-row concretized lower bounds after clipping, shape `[n_rows]`.
    /// Callers use these for grouped reduction instead of a raw scalar proof bit.
    pub(crate) postclip_lower_bounds: Vec<f32>,
}

#[cfg(test)]
mod tests;
