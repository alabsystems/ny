// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #la-warm-frontier — frontier-complete lA warm start on DAG graphs.
//!
//! The lA warm start seeds a BaB child's backward pass at its last split node
//! `N` from the block its parent's pass held for `N`. On a graph with a skip
//! connection or a multi-consumer node that single block is NOT a bound on the
//! objective: the parent still held the skip operand's block and the bias every
//! binary / n-ary node above `N` had diverted to the network-input channel
//! (`warm_start_guard.rs` has the counterexample). The #warm-dag guard therefore
//! refuses every such seed, and residual networks lost the warm start except in
//! their final unary chain.
//!
//! This module restores it CORRECTLY. When a capturing pass pops a node that a
//! split can name, it also records the rest of its backward state at that
//! instant — every other pending block and the input channel, each with its
//! certified coefficient error folded into its bias — as one
//! [`WarmFrontier`] (`batched_domain/domain_list/warm_frontier.rs`, carried
//! next to `N`'s block in the child's [`CachedLinearBounds`]). A child whose
//! guard refuses the single block, and whose cache holds a frontier for its
//! branch node that validates against the child's graph and widths, RESUMES
//! the backward pass from `N`'s block plus the whole frontier: it pops `N` and
//! everything below with its OWN boxes, split constraints, β and α, and the
//! other pending nodes as the parent had them. A child with no usable frontier
//! takes the cold pass from the output, exactly as under the guard alone. Chains
//! (and splits in a residual net's final unary chain) keep the historical
//! single-block seed untouched.
//!
//! Typed key: `bab.la_warm_frontier` → `BetaCrownConfig::la_warm_frontier`,
//! default `false`. Dark is BIT-IDENTICAL to the guard-only tree: no capture
//! site records a frontier and no seed site consults one (every new branch sits
//! behind the key).
//!
//! # Why a resumed bound bounds the objective on the child's region
//!
//! Lower side (the upper side is the mirror image with `<=` and outward-up
//! folds; upper mode is computed directly, not by negation). Fix a topological
//! order `π` of the graph (`GraphNetwork::dispatch_plan`, shared by every
//! pass); write `y_v(x)` for node `v`'s output and `y_in(x) = x`.
//!
//! **Pass invariant.** A reverse-topological CROWN pass over a domain `D` with
//! region `R_D` (its input box intersected with its split constraints) keeps a
//! pending set `S` of blocks `(A_v, b_v)` (possibly with certified coefficient
//! error `E_v`) such that for every `x ∈ R_D`
//!
//! ```text
//!   c · y_out(x)  >=  Σ_{v ∈ S}  min_{|Ã - A_v| <= E_v} ( Ã · y_v(x) ) + b_v        (INV)
//! ```
//!
//! It starts as `S = {out}` with `A = c, b = 0` (equality). Popping `v` (only
//! after every successor of `v` has been popped, so no block reaches `v`
//! later) replaces `v`'s term by a linear lower bound in the terms of `v`'s
//! inputs that is valid whenever the pre-activations lie in `D`'s certified
//! boxes — true for every `x ∈ R_D` — plus, for a split at `v`, a Lagrangian
//! term `β·s·z_v(x) >= 0` with `β >= 0` that holds because the split constraint
//! holds on `R_D`; binary / n-ary nodes route their bias to the `in` channel;
//! merges are certified (f64 sidecar or charged f32 roundoff). So INV holds
//! after every step, and the final `S = {in}` concretized over the input box is
//! a lower bound of `c · y_out` on `R_D`.
//!
//! **Capture.** When the parent's pass on `D_p` pops `N`, INV holds with
//! `S = {N} ∪ F`. The capture stores `N`'s block (the ordinary cache entry) and
//! `F`, each with `E_v` folded outward over `D_p`'s certified box `B_v ∋ y_v(x)`
//! (`x ∈ R_p`; the input box for `in`): `min_{|Ã-A|<=E} Ã·y >= A·y - E·max|B_v|`
//! term by term, so INV holds on `R_p` with the stored, error-free blocks.
//!
//! **Descent.** The child's region satisfies `R_c ⊆ R_p`: its split history
//! extends every ancestor's (and an input-split child's box is a sub-box). So
//! the stored INV holds on `R_c` — the parent's relaxations and β for every
//! node after `N` in `π` were valid on the superset. The cache may come from
//! any ancestor, not only the parent; the argument is the same.
//!
//! **Resume.** The child seeds exactly `S = {N} ∪ F` (INV holds on `R_c`) and
//! continues the reverse-topological traversal with its own certified boxes,
//! split constraints (including the new one at `N`), β >= 0 and α — every step
//! preserves INV on `R_c` by the pass argument above. Seed validation requires
//! every member of `F` to precede `N` in `π` (at `N`'s pop only such nodes can
//! be pending); since blocks only ever flow to a node's inputs, which precede it,
//! no node after `N` in `π` — i.e. none the parent already relaxed — ever holds
//! a block in the child, and every node that does is popped once, after all its
//! successors. The final concretization is over the child's input box, which
//! contains `R_c`. Hence the resumed bound is a bound on the objective over the
//! child's region. ∎
//!
//! **Resumed passes capture too.** A resumed pass satisfies INV on its own
//! region from its first step, so every block and frontier it captures for its
//! own descendants is exactly as valid as a cold pass's (its boxes, used for
//! the folds, are certified on its region). Nodes it never processed (after
//! `N` in `π`) simply have no cache entry, so a grandchild splitting there
//! takes the cold pass, as with the single-block seed.
//!
//! The frontier is therefore an exact linear identity of the objective GIVEN the
//! parent's relaxations above `N`: dropping any member (the #warm-dag defect) or
//! any diverted bias breaks INV. It is complete by construction (a snapshot of
//! the whole pending set at the pop, taken atomically; any member that cannot be
//! stored refuses the entire frontier) and validated structurally at every seed
//! (row count = the child's seed rows, member widths = the child's node widths,
//! members precede `N`, and `N`'s block is the EXACT block captured with the
//! frontier — a 128-bit digest, because the multi-objective lane merges caches
//! per node across passes and a fresh block next to an ancestor's frontier
//! would be a sum of terms from two different passes, not a bound).
//!
//! **Trust boundary (unchanged).** Which objective rows a cache belongs to is
//! the caller's association, exactly as for the single-block seed; the seed
//! sites check the row count. Relaxations above `N` are the parent's, so the
//! resumed bound can be looser than a cold pass on the child's tighter boxes —
//! never invalid.
//!
//! # Wiring
//!
//! * Capture (`is_frontier_capture_site` nodes only — ReLU/Sign and every
//!   GenBaB-splittable node — and only when something else is pending): the
//!   constrained pass (`constraints/backward/dispatch.rs`, attached in
//!   `finalize.rs`), the three batched cores (`batched.rs`: scalar, dense-spec,
//!   per-domain-objective, attached by the DomainList and per-disjunct callers),
//!   and the multi-objective ROOT pass (`network/graph_crown/spec_propagation`).
//! * Seed (only when the #warm-dag guard refuses the single block): the
//!   constrained setup (`constraints/backward/setup.rs` — multi-objective
//!   sequential / batched-per-child / per-disjunct / SPSA / optimizer
//!   baseline), the scalar batched core (DomainList) and the dense-spec batched
//!   core.
//! * Observable: `[la-warm-frontier] lane=… armed=1 captured=… resumed=…` once
//!   per armed multi-objective / DomainList run ([`WarmFrontierLaneScope`]), and
//!   the per-verifier counters behind it ([`WarmFrontierCounts`]).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::{CachedLinearBounds, FrontierBlock, FrontierCapture, WarmFrontier};
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::network::CrownMergeAccumulator;
use crate::{GraphNetwork, Layer, LinearBounds, NETWORK_INPUT};

/// Whether a split can name `layer`'s node, i.e. whether a child's
/// `last_branch_node()` can be this node: ReLU / Sign splits and every GenBaB
/// splittable operator (`nonlinear_branching::selector::is_splittable`, minus
/// RmsNorm, whose inv_rms constraints are never a `last_branch_node`). Only
/// these nodes record a frontier, which bounds the capture's memory.
pub(crate) fn is_frontier_capture_site(layer: &Layer) -> bool {
    layer.is_elementwise_activation()
        || matches!(
            layer,
            Layer::Sign(_) | Layer::BilinearCrown(_) | Layer::MulBinary(_)
        )
}

/// Per-verifier engagement counters (shared with sub-verifiers built by
/// `with_config_from`). Telemetry only: nothing reads them to decide a bound.
#[derive(Debug, Default)]
pub(crate) struct WarmFrontierCounters {
    captured: AtomicU64,
    capture_empty: AtomicU64,
    capture_refused: AtomicU64,
    resumed_constrained: AtomicU64,
    resumed_batched: AtomicU64,
    resumed_dense_spec: AtomicU64,
    single_seed: AtomicU64,
    refused_missing: AtomicU64,
    refused_invalid: AtomicU64,
}

/// A snapshot of [`WarmFrontierCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WarmFrontierCounts {
    /// Frontiers recorded at a capture site.
    pub(crate) captured: u64,
    /// Capture sites where nothing else was pending (no frontier needed).
    pub(crate) capture_empty: u64,
    /// Captures refused (unfoldable / non-finite / memory / shape).
    pub(crate) capture_refused: u64,
    /// Children resumed from a frontier in the constrained pass.
    pub(crate) resumed_constrained: u64,
    /// Children resumed from a frontier in the scalar batched core.
    pub(crate) resumed_batched: u64,
    /// Children resumed from a frontier in the dense-spec batched core.
    pub(crate) resumed_dense_spec: u64,
    /// Guard-accepted single-block seeds counted while armed.
    pub(crate) single_seed: u64,
    /// Guard-refused seeds with no frontier for the branch node (cold pass).
    pub(crate) refused_missing: u64,
    /// Guard-refused seeds whose frontier failed validation (cold pass).
    pub(crate) refused_invalid: u64,
}

impl WarmFrontierCounts {
    /// Children resumed from a frontier at any seed site.
    pub(crate) fn resumed(&self) -> u64 {
        self.resumed_constrained + self.resumed_batched + self.resumed_dense_spec
    }

    fn since(self, before: Self) -> Self {
        Self {
            captured: self.captured.saturating_sub(before.captured),
            capture_empty: self.capture_empty.saturating_sub(before.capture_empty),
            capture_refused: self.capture_refused.saturating_sub(before.capture_refused),
            resumed_constrained: self
                .resumed_constrained
                .saturating_sub(before.resumed_constrained),
            resumed_batched: self.resumed_batched.saturating_sub(before.resumed_batched),
            resumed_dense_spec: self
                .resumed_dense_spec
                .saturating_sub(before.resumed_dense_spec),
            single_seed: self.single_seed.saturating_sub(before.single_seed),
            refused_missing: self.refused_missing.saturating_sub(before.refused_missing),
            refused_invalid: self.refused_invalid.saturating_sub(before.refused_invalid),
        }
    }
}

/// The seed site that resumed a child (for the counters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WarmFrontierSite {
    Constrained,
    Batched,
    DenseSpec,
}

impl WarmFrontierCounters {
    pub(crate) fn snapshot(&self) -> WarmFrontierCounts {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        WarmFrontierCounts {
            captured: load(&self.captured),
            capture_empty: load(&self.capture_empty),
            capture_refused: load(&self.capture_refused),
            resumed_constrained: load(&self.resumed_constrained),
            resumed_batched: load(&self.resumed_batched),
            resumed_dense_spec: load(&self.resumed_dense_spec),
            single_seed: load(&self.single_seed),
            refused_missing: load(&self.refused_missing),
            refused_invalid: load(&self.refused_invalid),
        }
    }

    /// Record one capture outcome; returns the frontier when one was recorded.
    pub(crate) fn record_capture(&self, capture: FrontierCapture) -> Option<WarmFrontier> {
        match capture {
            FrontierCapture::Recorded(frontier) => {
                self.captured.fetch_add(1, Ordering::Relaxed);
                Some(frontier)
            }
            FrontierCapture::Empty => {
                self.capture_empty.fetch_add(1, Ordering::Relaxed);
                None
            }
            FrontierCapture::Refused(reason) => {
                self.capture_refused.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(reason, "#la-warm-frontier capture refused");
                None
            }
        }
    }

    pub(crate) fn record_resumed(&self, site: WarmFrontierSite) {
        let counter = match site {
            WarmFrontierSite::Constrained => &self.resumed_constrained,
            WarmFrontierSite::Batched => &self.resumed_batched,
            WarmFrontierSite::DenseSpec => &self.resumed_dense_spec,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_single_seed(&self) {
        self.single_seed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_refusal(&self, refusal: FrontierSeedRefusal) {
        let counter = match refusal {
            FrontierSeedRefusal::Missing => &self.refused_missing,
            FrontierSeedRefusal::Invalid(reason) => {
                tracing::debug!(reason, "#la-warm-frontier seed refused");
                &self.refused_invalid
            }
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

impl BetaCrownVerifier {
    /// The counters when `bab.la_warm_frontier` is armed, else `None` — the
    /// capture sites record a frontier exactly when this is `Some`.
    pub(crate) fn la_warm_frontier_armed(&self) -> Option<&WarmFrontierCounters> {
        self.config
            .la_warm_frontier
            .then_some(&*self.la_warm_frontier_counters)
    }

    /// Snapshot of this verifier's (and its `with_config_from` children's)
    /// frontier counters.
    #[cfg(test)]
    pub(crate) fn la_warm_frontier_counts(&self) -> WarmFrontierCounts {
        self.la_warm_frontier_counters.snapshot()
    }
}

/// Snapshot the pending set of a `CrownMergeAccumulator` (the constrained and
/// root passes), AFTER the popped node's own entry was taken, and store it as
/// a frontier paired with `branch_block` — the popped node's block exactly as
/// the pass stored it in its capture map. `box_of` returns the capturing
/// domain's certified box for a key (the input box for `NETWORK_INPUT`). A
/// resource refusal is a `Refused` capture; only a deadline is an `Err`.
pub(crate) fn capture_accumulator_frontier<'b, F>(
    pending: &CrownMergeAccumulator,
    branch_block: &LinearBounds,
    box_of: F,
    deadline: Option<Instant>,
) -> Result<FrontierCapture>
where
    F: FnMut(&str) -> Option<&'b BoundedTensor>,
{
    if pending.is_empty() {
        return Ok(FrontierCapture::Empty);
    }
    let snapshot = match pending.snapshot_dense_with_deadline(deadline) {
        Ok(snapshot) => snapshot,
        Err(error) if error.is_deadline_exceeded() => return Err(error),
        Err(NyError::CpuMemoryExceeded { .. }) => return Ok(FrontierCapture::Refused("memory")),
        Err(_) => return Ok(FrontierCapture::Refused("snapshot")),
    };
    WarmFrontier::capture(branch_block, snapshot, box_of, deadline)
}

/// Why a guard-refused child could not resume from a frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrontierSeedRefusal {
    /// The cache holds no frontier for the branch node.
    Missing,
    /// The frontier does not fit the child's pass (stable reason token).
    Invalid(&'static str),
}

/// A validated resume point: `N`'s block and the rebuilt members, all
/// materialized BEFORE anything is seeded, so a seed site publishes all of it
/// or none of it.
pub(crate) struct FrontierSeed {
    /// The branch node's own block.
    pub(crate) branch_block: LinearBounds,
    /// `(key, block)` for every other pending node and the input channel.
    pub(crate) members: Vec<(String, LinearBounds)>,
}

/// Validate the frontier `cache` holds for `branch_node` against a child pass
/// whose seed has `rows` rows, whose network input has `input_dim` elements
/// and whose node output widths are `width_of`, and rebuild it.
///
/// Checks (any failure refuses; the child then takes the cold pass): a frontier
/// and `N`'s block exist and the block is the EXACT one the frontier was
/// captured with (`BlockDigest`); `N`'s block has `rows` rows and `N`'s width; every
/// member has `rows` rows, is `NETWORK_INPUT` with `input_dim` columns or a
/// graph node that PRECEDES `N` in the dispatch plan's topological order with
/// that node's width, and is not `N` itself; every rebuilt block is finite.
pub(crate) fn resolve_frontier_seed(
    graph: &GraphNetwork,
    branch_node: &str,
    cache: &CachedLinearBounds,
    rows: usize,
    input_dim: usize,
    width_of: impl Fn(&str) -> Option<usize>,
) -> std::result::Result<FrontierSeed, FrontierSeedRefusal> {
    use FrontierSeedRefusal::{Invalid, Missing};
    let frontier = cache.frontier(branch_node).ok_or(Missing)?;
    if frontier.rows() != rows || frontier.is_empty() {
        return Err(Invalid("rows"));
    }
    // Pairing: the frontier is an identity only next to the EXACT block it
    // was captured with (a per-node cache merge can replace that block).
    if cache.block_digest(branch_node) != Some(frontier.branch_digest()) {
        return Err(Invalid("branch_mismatch"));
    }
    let branch_block = cache
        .linear_bounds(branch_node)
        .ok_or(Invalid("branch_block"))?;
    if branch_block.num_outputs() != rows
        || branch_block.upper_a().nrows() != rows
        || Some(branch_block.num_inputs()) != width_of(branch_node)
    {
        return Err(Invalid("branch_shape"));
    }
    let plan = graph.dispatch_plan().map_err(|_| Invalid("plan"))?;
    let branch_idx = plan
        .index_of(branch_node)
        .filter(|&idx| idx != plan.network_input_idx)
        .ok_or(Invalid("branch_unknown"))?;
    let mut members = Vec::with_capacity(frontier.len());
    for (key, block) in frontier.members() {
        let block: &FrontierBlock = block;
        if block.rows() != rows {
            return Err(Invalid("member_rows"));
        }
        if key == branch_node {
            return Err(Invalid("member_is_branch"));
        }
        if key == NETWORK_INPUT {
            if block.cols() != input_dim {
                return Err(Invalid("input_width"));
            }
        } else {
            let idx = plan
                .index_of(key)
                .filter(|&idx| idx != plan.network_input_idx)
                .ok_or(Invalid("member_unknown"))?;
            if idx >= branch_idx {
                return Err(Invalid("member_order"));
            }
            if width_of(key) != Some(block.cols()) {
                return Err(Invalid("member_width"));
            }
        }
        let rebuilt = block.to_linear_bounds().map_err(|_| Invalid("rebuild"))?;
        members.push((key.to_string(), rebuilt));
    }
    Ok(FrontierSeed {
        branch_block,
        members,
    })
}

/// Prints one `[la-warm-frontier]` line with this verifier's counter deltas
/// when dropped. Built only when the key is armed, so a dark run prints
/// nothing.
pub(crate) struct WarmFrontierLaneScope {
    counters: Arc<WarmFrontierCounters>,
    before: WarmFrontierCounts,
    lane: &'static str,
}

impl WarmFrontierLaneScope {
    pub(crate) fn enter(verifier: &BetaCrownVerifier, lane: &'static str) -> Option<Self> {
        verifier.config.la_warm_frontier.then(|| Self {
            counters: Arc::clone(&verifier.la_warm_frontier_counters),
            before: verifier.la_warm_frontier_counters.snapshot(),
            lane,
        })
    }
}

impl Drop for WarmFrontierLaneScope {
    fn drop(&mut self) {
        let delta = self.counters.snapshot().since(self.before);
        eprintln!(
            "[la-warm-frontier] lane={} armed=1 captured={} capture_empty={} capture_refused={} \
             resumed={} resumed_constrained={} resumed_batched={} resumed_dense_spec={} \
             single_seed={} refused_missing={} refused_invalid={}",
            self.lane,
            delta.captured,
            delta.capture_empty,
            delta.capture_refused,
            delta.resumed(),
            delta.resumed_constrained,
            delta.resumed_batched,
            delta.resumed_dense_spec,
            delta.single_seed,
            delta.refused_missing,
            delta.refused_invalid,
        );
    }
}
