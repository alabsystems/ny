// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-crown-interm: unconditional engagement lines for the alpha reference
//! map.
//!
//! # Why these lines exist
//!
//! The typed `solver.alpha-crown.fix_interm_bounds: false` route used to be
//! visible only through `info!` lines that the vnncomp log filter drops. A
//! measurement of it therefore could not tell "CROWN intermediates ran and did
//! not help" from "the route never engaged" or "every target fell back to IBP
//! under its deadline" — and on vit_2023 the last one is the documented failure
//! mode (`18/19 DEMANDED targets did not complete CROWN`, 2026-08-08). A null
//! read without these lines is VOID, not negative.
//!
//! * `[ref-map] stage=collect source=crown-ibp route=… dag=… nodes=N
//!   crown_done=K ibp_fallback=F deadline_hits=D vacuous=V skipped=S
//!   cap_secs=… secs=S n=…` — one per CROWN-IBP reference collection
//!   (`bounds/alpha.rs`, the `!fix_interm_bounds || deep_seq || dense_dag`
//!   branch). `route=fix_interm` is the typed route.
//! * `[ref-map] stage=publish route=fix_interm source=artifact|recollect …` —
//!   which map the DAG-alpha dispatch published after its loop.
//! * `[ref-map] stage=alpha-post source=crown-alpha …` — the all-node alpha
//!   pass (`collect_crown_bounds_with_alpha`). On vit that is the typed route's
//!   post-loop publication; the same pass also serves the sequential
//!   collection and warm starts, which print it too.
//! * `[softmax-complex] stage=… decomposed=a/b shift=fixed-index guard_skips=g
//!   demand_sub=d crown_done=c ibp_fallback=f` — alongside the first and third,
//!   whenever the graph carries the fixed-index softmax rewrite: how many of the
//!   `x - x_{j*}` Exp inputs were demanded and how many finished with CROWN.
//!   Under plain IBP those inputs widen to `u_i - l_{j*}`, so the rewrite is
//!   worth something only where `crown_done` is non-zero.
//!
//! # Cost
//!
//! Print-only. Each site owns a counter: the first 16 events always print, then
//! only powers of two, so an input-split lane that re-collects per domain cannot
//! flood stderr. Nothing here is computed unless the line is printed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::network::core::GraphNetwork;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};

/// Event ordinal when this event should print, else `None`.
pub(super) fn next_event(counter: &AtomicUsize) -> Option<usize> {
    let n = counter.fetch_add(1, Ordering::Relaxed).saturating_add(1);
    (n <= 16 || n.is_power_of_two()).then_some(n)
}

/// How one node's provenance counts in the engagement census.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProvenanceClass {
    /// CROWN completed and its (CROWN ∩ IBP) box was published.
    CrownDone,
    /// CROWN completed but lost finiteness on ~every element
    /// (`CrownVacuousResult`): completed, yet worth nothing.
    Vacuous,
    /// A demanded target kept its IBP box because a clock ran out
    /// (collection deadline, per-node share, cost refusal, below-floor share).
    DeadlineFallback,
    /// A demanded target kept its IBP box for a non-time reason
    /// (propagation error, shape mismatch, memory budget, empty intersection).
    OtherFallback,
    /// Not requested work: demand-pruned, stable ReLU rows, objective cone,
    /// width threshold.
    Structural,
}

/// Exhaustive on purpose: a new fallback reason must be classified here.
pub(super) fn classify(provenance: BoundsProvenance) -> ProvenanceClass {
    match provenance {
        BoundsProvenance::Crown => ProvenanceClass::CrownDone,
        BoundsProvenance::ForwardFallback(reason) => match reason {
            CrownIbpFallbackReason::CrownVacuousResult => ProvenanceClass::Vacuous,
            CrownIbpFallbackReason::DeadlineExceeded
            | CrownIbpFallbackReason::PerNodeDeadlineExceeded
            | CrownIbpFallbackReason::PatchesBudgetExceeded
            | CrownIbpFallbackReason::WalkCostRefused
            | CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
            | CrownIbpFallbackReason::PerNodeBudgetBelowFloor => ProvenanceClass::DeadlineFallback,
            CrownIbpFallbackReason::CrownPropagationError
            | CrownIbpFallbackReason::ShapeMismatch
            | CrownIbpFallbackReason::EmptyIntersection
            | CrownIbpFallbackReason::MemoryBudgetExceeded
            // Partial row publication is neither a whole-target completion
            // nor evidence that a deadline was exceeded. Its event owns stage.
            | CrownIbpFallbackReason::SelectedHeadRows => ProvenanceClass::OtherFallback,
            CrownIbpFallbackReason::WidthBelowThreshold
            | CrownIbpFallbackReason::DemandDrivenSkip
            | CrownIbpFallbackReason::StableReluRowsSkipped
            | CrownIbpFallbackReason::ObjectiveConeRowsSkipped => ProvenanceClass::Structural,
        },
    }
}

/// Counts over a set of node provenances.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ProvenanceCensus {
    pub(super) crown_done: usize,
    pub(super) vacuous: usize,
    pub(super) deadline_hits: usize,
    pub(super) other_fallback: usize,
    pub(super) structural: usize,
}

impl ProvenanceCensus {
    pub(super) fn add(&mut self, class: ProvenanceClass) {
        match class {
            ProvenanceClass::CrownDone => self.crown_done += 1,
            ProvenanceClass::Vacuous => self.vacuous += 1,
            ProvenanceClass::DeadlineFallback => self.deadline_hits += 1,
            ProvenanceClass::OtherFallback => self.other_fallback += 1,
            ProvenanceClass::Structural => self.structural += 1,
        }
    }

    /// Demanded targets that published their IBP box (clock or error).
    pub(super) fn ibp_fallback(&self) -> usize {
        self.deadline_hits + self.other_fallback
    }

    /// Demanded (requested) targets: everything except structural skips.
    pub(super) fn demanded(&self) -> usize {
        self.crown_done + self.vacuous + self.ibp_fallback()
    }
}

fn cap_label(cap: Option<Duration>) -> String {
    cap.map_or_else(|| "none".to_string(), |cap| cap.as_secs().to_string())
}

/// `[ref-map] stage=collect …` for one CROWN-IBP reference collection, plus
/// the fixed-index softmax census when the graph carries that rewrite.
pub(super) fn emit_crown_ibp_reference_collect(
    graph: &GraphNetwork,
    result: &GraphCrownIbpBoundsResult,
    route: &str,
    is_dag: bool,
    tightening_cap: Option<Duration>,
    elapsed: Duration,
) {
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    let Some(n) = next_event(&EVENTS) else {
        return;
    };
    let mut census = ProvenanceCensus::default();
    for provenance in result.provenance.values() {
        census.add(classify(*provenance));
    }
    eprintln!(
        "[ref-map] stage=collect source=crown-ibp route={route} dag={} nodes={} crown_done={} \
         ibp_fallback={} deadline_hits={} vacuous={} skipped={} cap_secs={} secs={:.3} n={n}",
        u8::from(is_dag),
        result.bounds.len(),
        census.crown_done,
        census.ibp_fallback(),
        census.deadline_hits,
        census.vacuous,
        census.structural,
        cap_label(tightening_cap),
        elapsed.as_secs_f64(),
    );
    let (shift_nodes, remaining_softmax) = graph.softmax_fixed_index_census();
    if shift_nodes.is_empty() {
        return;
    }
    let mut shift = ProvenanceCensus::default();
    for name in &shift_nodes {
        if let Some(provenance) = result.provenance.get(name.as_str()) {
            shift.add(classify(*provenance));
        }
    }
    emit_softmax_fixed_index_line(
        "collect",
        shift_nodes.len(),
        remaining_softmax,
        shift.demanded(),
        shift.crown_done,
        shift.demanded() - shift.crown_done,
    );
}

/// `[ref-map] stage=publish …`: which map the typed route published after the
/// DAG-alpha loop. `source=artifact` is the optimizer's own shrink-only map
/// (reused under `crown_ibp_reference_cap_secs`); `source=recollect` is the
/// historical post-loop re-collection.
pub(super) fn emit_typed_route_publication(
    source: &str,
    nodes: usize,
    cap: Option<Duration>,
    elapsed: Duration,
) {
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    let Some(n) = next_event(&EVENTS) else {
        return;
    };
    eprintln!(
        "[ref-map] stage=publish route=fix_interm source={source} nodes={nodes} cap_secs={} \
         secs={:.3} n={n}",
        cap_label(cap),
        elapsed.as_secs_f64(),
    );
}

/// Outcome counts of one post-loop all-node alpha pass
/// (`collect_crown_bounds_with_alpha`).
#[derive(Debug, Clone, Default)]
pub(super) struct AlphaPostPassCensus {
    pub(super) nodes: usize,
    pub(super) crown_done: usize,
    pub(super) ibp_fallback: usize,
    /// Nodes filled from the reference map because the pass deadline expired
    /// before they were reached.
    pub(super) deadline_hits: usize,
    /// Nodes whose CROWN backward completed, by name (for the softmax census).
    pub(super) completed: std::collections::HashSet<String>,
}

/// `[ref-map] stage=alpha-post …` plus the fixed-index softmax census.
pub(super) fn emit_alpha_post_pass(
    graph: &GraphNetwork,
    census: &AlphaPostPassCensus,
    deadline_bounded: bool,
    elapsed: Duration,
) {
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    let Some(n) = next_event(&EVENTS) else {
        return;
    };
    // The post-pass also serves the sequential collection and warm starts, so
    // `dag=` is read from the graph rather than assumed.
    let dag = match graph.exec_order() {
        Ok(order) => !graph.is_sequential_graph(order),
        Err(_) => true,
    };
    eprintln!(
        "[ref-map] stage=alpha-post source=crown-alpha dag={} nodes={} crown_done={} \
         ibp_fallback={} deadline_hits={} deadline={} secs={:.3} n={n}",
        u8::from(dag),
        census.nodes,
        census.crown_done,
        census.ibp_fallback,
        census.deadline_hits,
        u8::from(deadline_bounded),
        elapsed.as_secs_f64(),
    );
    // #vit-collector-throughput: the collector-kernel engagement census rides
    // the same retained event (prints only while a collector-kernel key is
    // armed). The alpha-post pass is the other heavy consumer of the
    // bilinear kernel (vit.md §2.4: `deadline_hits=78 … secs=10.000`).
    crate::layers::collector_kernels::emit_census("alpha-post");
    let (shift_nodes, remaining_softmax) = graph.softmax_fixed_index_census();
    if shift_nodes.is_empty() {
        return;
    }
    let done = shift_nodes
        .iter()
        .filter(|name| census.completed.contains(name.as_str()))
        .count();
    emit_softmax_fixed_index_line(
        "alpha-post",
        shift_nodes.len(),
        remaining_softmax,
        shift_nodes.len(),
        done,
        shift_nodes.len() - done,
    );
}

fn emit_softmax_fixed_index_line(
    stage: &str,
    decomposed: usize,
    remaining_softmax: usize,
    demand_sub: usize,
    crown_done: usize,
    ibp_fallback: usize,
) {
    eprintln!(
        "[softmax-complex] stage={stage} decomposed={decomposed}/{} shift=fixed-index \
         guard_skips={remaining_softmax} demand_sub={demand_sub} crown_done={crown_done} \
         ibp_fallback={ibp_fallback}",
        decomposed + remaining_softmax,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fallback_reason_has_exactly_one_class() {
        use crate::types::CrownIbpFallbackReason as R;
        let cases = [
            (R::CrownVacuousResult, ProvenanceClass::Vacuous),
            (R::DeadlineExceeded, ProvenanceClass::DeadlineFallback),
            (
                R::PerNodeDeadlineExceeded,
                ProvenanceClass::DeadlineFallback,
            ),
            (R::PatchesBudgetExceeded, ProvenanceClass::DeadlineFallback),
            (R::WalkCostRefused, ProvenanceClass::DeadlineFallback),
            (
                R::PartialCrownRowsDeadlineExceeded,
                ProvenanceClass::DeadlineFallback,
            ),
            (
                R::PerNodeBudgetBelowFloor,
                ProvenanceClass::DeadlineFallback,
            ),
            (R::CrownPropagationError, ProvenanceClass::OtherFallback),
            (R::ShapeMismatch, ProvenanceClass::OtherFallback),
            (R::EmptyIntersection, ProvenanceClass::OtherFallback),
            (R::MemoryBudgetExceeded, ProvenanceClass::OtherFallback),
            (R::SelectedHeadRows, ProvenanceClass::OtherFallback),
            (R::WidthBelowThreshold, ProvenanceClass::Structural),
            (R::DemandDrivenSkip, ProvenanceClass::Structural),
            (R::StableReluRowsSkipped, ProvenanceClass::Structural),
            (R::ObjectiveConeRowsSkipped, ProvenanceClass::Structural),
        ];
        for (reason, class) in cases {
            assert_eq!(
                classify(BoundsProvenance::ForwardFallback(reason)),
                class,
                "{reason:?}"
            );
        }
        assert_eq!(
            classify(BoundsProvenance::Crown),
            ProvenanceClass::CrownDone
        );
    }

    #[test]
    fn census_counts_demanded_targets_and_fallbacks() {
        let mut census = ProvenanceCensus::default();
        for class in [
            ProvenanceClass::CrownDone,
            ProvenanceClass::CrownDone,
            ProvenanceClass::Vacuous,
            ProvenanceClass::DeadlineFallback,
            ProvenanceClass::OtherFallback,
            ProvenanceClass::Structural,
        ] {
            census.add(class);
        }
        assert_eq!(census.crown_done, 2);
        assert_eq!(census.ibp_fallback(), 2);
        assert_eq!(census.demanded(), 5);
        assert_eq!(census.structural, 1);
    }

    #[test]
    fn event_gate_prints_the_first_sixteen_then_powers_of_two() {
        let counter = AtomicUsize::new(0);
        let printed: Vec<usize> = (0..100).filter_map(|_| next_event(&counter)).collect();
        let expected: Vec<usize> = (1..=16).chain([32, 64]).collect();
        assert_eq!(printed, expected);
    }
}
