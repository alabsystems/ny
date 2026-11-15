// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M7 (ENGINE_PROGRAM §2 M7, `#graph-lane-search`): the BATCHED graph-domain
//! α/β search request of the graph β-CROWN lane.
//!
//! The graph lane (cora, malbeware, relusplitter and the other `GraphNetwork`
//! categories) bounds a ROOT and then waves of β-CROWN sub-domains. Each
//! domain is the input box intersected with its ReLU split literals
//! (`z_j ≥ 0` or `z_j ≤ 0` on a pre-activation), and its certified bound
//! depends on two families the lane optimises today with a few certified
//! iterations each:
//!
//! * the ReLU lower slopes `α ∈ [0, 1]` of every neuron unstable in the
//!   domain's certified boxes;
//! * one Lagrange multiplier `β_k ≥ 0` per split literal `s_k z_k ≥ 0`
//!   (`s_k = +1` active, `−1` inactive): weak duality gives
//!   `min f ≥ min (f − Σ β_k s_k z_k)` over any relaxation.
//!
//! A [`GraphBabRequest`] hands a batch of such domains to an UNTRUSTED
//! backend, over the lane's plain-data [`super::GraphSearchNet`] copy (the V2
//! lowering, self-checked against the lane's own point forward, G1) with the
//! lane's certified boxes as FROZEN reference boxes. The root is the domain
//! with no split. Every output is [`Untrusted`]; the only product is one
//! [`GraphDomainHint`] per domain, which the lane consumes through its ingest
//! gate: non-finite values refused, slopes clamped into `[0, 1]` and written
//! only on neurons unstable in the domain's CERTIFIED boxes, multipliers
//! clamped to `≥ 0` and written only on the domain's own split literals, then
//! ONE certified evaluation by the lane's existing evaluator, kept per row only
//! when strictly better. The device never certifies.

use std::time::Instant;

use super::{GraphAlphaHint, TensorBox, Untrusted};

/// One ReLU split literal of a domain, in search-copy terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphDomainSplit {
    /// The ReLU's op index in the [`super::GraphSearchNet`].
    pub op: usize,
    /// Flat neuron of the ReLU's pre-activation (= of its output).
    pub neuron: u32,
    /// `true`: the domain has `z ≥ 0` (active); `false`: `z ≤ 0` (inactive).
    pub active: bool,
}

/// One domain of a [`GraphBabRequest`].
#[derive(Debug, Clone, Default)]
pub struct GraphDomainRequest {
    /// The domain's ReLU split literals (one multiplier each, in this order).
    pub splits: Vec<GraphDomainSplit>,
    /// The domain's own reference boxes: each REPLACES the batch box of its
    /// tensor for this domain (typically every ReLU pre-activation of a child,
    /// which its splits and its own bounding pass have tightened).
    pub boxes: Vec<TensorBox>,
    /// The domain's input box lower sides (empty = the batch's box).
    pub lo: Vec<f64>,
    /// The domain's input box upper sides (empty = the batch's box).
    pub hi: Vec<f64>,
    /// Objective rows this domain searches (indices into
    /// [`GraphBabRequest::rows`]; empty = every row).
    pub rows: Vec<usize>,
    /// Starting slopes `(op, per-neuron)` per ReLU site (a missing or
    /// wrong-width site starts at the adaptive heuristic).
    pub init_relu: Vec<(usize, Vec<f32>)>,
    /// Starting multipliers, aligned with `splits` (missing, non-finite or
    /// negative → 0).
    pub init_beta: Vec<f32>,
}

/// One M7 batch: the shared frozen boxes and objective, and the domains.
#[derive(Debug, Clone)]
pub struct GraphBabRequest {
    /// Input box lower sides (every domain's, unless it carries its own).
    pub lo: Vec<f64>,
    /// Input box upper sides.
    pub hi: Vec<f64>,
    /// Reference boxes: at least every operand of every relaxed op.
    pub boxes: Vec<TensorBox>,
    /// Objective rows over the last tensor (LOWER objectives).
    pub rows: Vec<Vec<f64>>,
    /// Per-row thresholds (a row counts as verified above its threshold).
    pub thresholds: Vec<f64>,
    /// The domains.
    pub domains: Vec<GraphDomainRequest>,
    /// Iterations per domain after its initial evaluation.
    pub iters: usize,
    /// Adam step of the slopes.
    pub lr_alpha: f64,
    /// Adam step of the multipliers.
    pub lr_beta: f64,
    /// Wall deadline: a domain not started by then gets no hint; a started
    /// one returns its best iterate so far.
    pub deadline: Option<Instant>,
    /// Worker threads (0 = the backend's choice).
    pub threads: usize,
    /// Admitted working-set bytes (program + every worker's domain).
    pub mem_cap_bytes: u64,
}

/// The M7 hint of one domain.
#[derive(Debug, Clone, Default)]
pub struct GraphDomainHint {
    /// The domain's relaxation parameters in V2's layout: `(op, slopes)` per
    /// ReLU site, full width (neurons the search held stable — including the
    /// domain's split neurons — carry their exact limit, 1 active / 0
    /// inactive), plus the product planes when the net has any.
    pub alpha: GraphAlphaHint,
    /// One multiplier per split literal, aligned with the request's `splits`.
    pub beta: Vec<f32>,
}

/// Untrusted telemetry of one searched domain.
#[derive(Debug)]
pub struct GraphDomainStats {
    /// Iterations completed after the initial evaluation.
    pub iters: usize,
    /// Full (all searched rows) evaluations.
    pub evals: usize,
    /// Worst searched-row slack at the starting parameters.
    pub init_slack: Untrusted<f64>,
    /// Worst searched-row slack at the returned parameters.
    pub best_slack: Untrusted<f64>,
    /// Rows the search saw verified at the start / at the returned point.
    pub verified_init: usize,
    /// Rows the search saw verified at the returned parameters.
    pub verified_best: usize,
    /// Why the domain stopped: `verified`, `patience`, `iters` or `deadline`.
    pub stop: &'static str,
}

/// Untrusted telemetry of one M7 batch.
#[derive(Debug)]
pub struct GraphBabStats {
    /// Domains in the request.
    pub domains: usize,
    /// Domains that produced a hint.
    pub searched: usize,
    /// Domains refused (malformed literal or box) — no hint.
    pub refused: usize,
    /// Domains not started before the deadline — no hint.
    pub late: usize,
    /// Objective rows of the batch.
    pub rows: usize,
    /// ReLU sites of the lowered program.
    pub relu_sites: usize,
    /// Split literals over every searched domain.
    pub splits: usize,
    /// Worker threads used.
    pub threads: usize,
    /// Wall seconds of lowering + search.
    pub search_secs: f64,
    /// Per-domain telemetry, aligned with the request (`None` = no hint).
    pub per_domain: Vec<Option<GraphDomainStats>>,
}

/// What `SearchBackend::graph_bab_batch` returns: one hint slot per domain,
/// aligned with the request (`None` = that domain was refused or late).
#[derive(Debug)]
pub struct GraphBabOutcome {
    /// The hints, for the ingest gate only.
    pub hints: Untrusted<Vec<Option<GraphDomainHint>>>,
    /// Telemetry.
    pub stats: GraphBabStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_bab_request_types_default_to_an_empty_unsplit_domain() {
        let domain = GraphDomainRequest::default();
        assert_eq!(domain.splits.len(), 0);
        assert!(domain.boxes.is_empty() && domain.lo.is_empty() && domain.hi.is_empty());
        assert!(domain.rows.is_empty() && domain.init_relu.is_empty());
        assert_eq!(domain.init_beta.len(), 0);
        let hint = GraphDomainHint::default();
        assert!(hint.alpha.relu.is_empty() && hint.beta.is_empty());
        let split = GraphDomainSplit {
            op: 3,
            neuron: 7,
            active: false,
        };
        let copy = split;
        assert_eq!(copy, split);
        assert!(!copy.active && copy.op == 3 && copy.neuron == 7);
    }
}
