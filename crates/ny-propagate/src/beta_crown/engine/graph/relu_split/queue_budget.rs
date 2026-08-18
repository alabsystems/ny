// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Memory budget for the graph ReLU-split branch-and-bound domain queue
//! (#ml4acopf-bab-queue-mem).
//!
//! # The defect
//!
//! `bab_loop.rs` keeps its frontier in a `BinaryHeap<GraphBabDomain>` with no
//! size and no memory cap. Every queued `GraphBabDomain` owns a full copy of
//! the per-node bounds map plus its per-neuron α state, so one domain costs
//! `8 * Σ node elements + 32 * unstable α neurons` bytes — measured at 1.37 MB
//! on `ml4acopf_2024` `118_ieee_ml4acopf-linear-residual`. The only limits that
//! exist are COUNT based: `max_domains = 100_000` (checked once per wave) and
//! `max_queue_size = 500_000` (which only `DomainList` consults, not this
//! heap). 100,000 × 1.37 MB = 137 GB, so on that model the host OOM-killer
//! arrives first: measured 116.3 GiB RSS at 81,920 resident domains, `rc=137`,
//! with the queue itself accounting for 96% of the footprint.
//!
//! # The bound
//!
//! `GraphBabQueueBudget` caps the summed `estimate_graph_domain_bytes` of the
//! resident queue at `BetaCrownConfig::max_queue_bytes`, evicting the
//! lowest-priority (least promising) domains down to a low-water mark when the
//! cap is exceeded, and additionally caps how many bytes one popped wave may
//! hold outside the queue so the transient batch + children peak stays within a
//! small multiple of the budget.
//!
//! # Soundness
//!
//! Eviction never changes a bound. It only changes WHICH unexplored sub-boxes
//! get expanded, and every evicted domain is by construction unverified. The
//! caller therefore sets `GraphBabLifecycle::unresolved_due_to_eviction`, which
//! makes `has_unresolved()` true and forces `build_final_result()` to report
//! `Unknown` instead of `Verified` for the rest of the run. The bounds carried
//! by the surviving domains are the same bounds the unbounded loop would have
//! carried, so nothing is ever looser than the intermediate bounds the domain
//! already held; the cost is completeness only.
//!
//! # Gate
//!
//! `max_queue_bytes == 0` means unlimited and is the default: `enforce` returns
//! `None` without touching the queue and `wave_bytes()` returns `None`, so the
//! BaB loop runs its original code path with one `usize` comparison added per
//! wave and per pop. Categories that do not set the key are byte-identical.
//!
//! # Scope
//!
//! Both graph ReLU-split `BinaryHeap<GraphBabDomain>` routes use this budget:
//! the ordinary `relu_split::bab_loop` and the precomputed-bounds entry point in
//! `relu_split_bounds.rs`. Keeping the estimator, eviction ordering, and
//! lifecycle latch shared prevents the two structurally identical frontiers
//! from drifting apart.

use std::collections::BinaryHeap;

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::adaptive_microbatch::estimate_graph_domain_bytes;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

/// A popped wave may hold at most `1 / WAVE_BYTE_SHARE_DEN` of the queue budget
/// outside the queue. The wave's children (2 per parent) are materialized while
/// the parents are still alive, so the transient peak is bounded by
/// `budget + 3 * budget / WAVE_BYTE_SHARE_DEN` = 1.75x the budget at 4.
const WAVE_BYTE_SHARE_DEN: usize = 4;

/// After an eviction the queue is left at `EVICT_LOW_WATER_NUM /
/// EVICT_LOW_WATER_DEN` of the budget so the O(n log n) rebuild is amortized
/// over several waves instead of firing on every single push.
const EVICT_LOW_WATER_NUM: usize = 7;
const EVICT_LOW_WATER_DEN: usize = 8;

/// Outcome of one budget enforcement pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueueEviction {
    /// Domains discarded by this pass (0 when the queue was already inside the
    /// budget). Any non-zero value must set
    /// `GraphBabLifecycle::unresolved_due_to_eviction`.
    pub(crate) evicted: usize,
    /// Summed `estimate_graph_domain_bytes` of the queue before the pass.
    pub(crate) bytes_before: usize,
    /// Summed `estimate_graph_domain_bytes` of the queue after the pass.
    pub(crate) bytes_after: usize,
}

/// Byte budget for the resident graph ReLU-split BaB queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphBabQueueBudget {
    /// Cap in bytes on the resident queue. `0` disables the budget entirely.
    queue_bytes: usize,
}

impl GraphBabQueueBudget {
    /// Read the budget from the engine config (`bab.max_queue_bytes`).
    pub(crate) fn from_config(config: &BetaCrownConfig) -> Self {
        Self {
            queue_bytes: config.max_queue_bytes,
        }
    }

    /// Explicit budget, for tests.
    #[cfg(test)]
    pub(crate) fn with_queue_bytes(queue_bytes: usize) -> Self {
        Self { queue_bytes }
    }

    /// Whether the budget is disabled (the shipped default).
    pub(crate) fn is_unlimited(self) -> bool {
        self.queue_bytes == 0
    }

    /// Byte ceiling for one popped wave, or `None` when unlimited.
    ///
    /// The popped batch lives OUTSIDE the heap, so the queue cap alone does not
    /// bound it: a 8,192-domain wave of 1.37 MB domains is 11 GB on its own.
    /// Always at least one domain is admitted regardless (the caller guarantees
    /// forward progress by never refusing the first pop).
    pub(crate) fn wave_bytes(self) -> Option<usize> {
        if self.is_unlimited() {
            None
        } else {
            Some((self.queue_bytes / WAVE_BYTE_SHARE_DEN).max(1))
        }
    }

    /// Enforce the byte cap on `queue`, evicting the lowest-priority domains.
    ///
    /// Returns `None` when the budget is disabled — the queue is not even
    /// traversed in that case. Otherwise returns the measured before/after
    /// sizes and the number of domains discarded.
    ///
    /// Domains are kept in the queue's own pop order (highest priority first),
    /// so the retained frontier is exactly the prefix branch-and-bound would
    /// have expanded next. Ordering is deterministic: the heap's backing array
    /// order is a pure function of the push/pop sequence and
    /// `estimate_graph_domain_bytes` sums lengths only (no map-iteration-order
    /// dependence).
    pub(crate) fn enforce(self, queue: &mut BinaryHeap<GraphBabDomain>) -> Option<QueueEviction> {
        if self.is_unlimited() {
            return None;
        }
        let bytes_before: usize = queue
            .iter()
            .map(estimate_graph_domain_bytes)
            .fold(0usize, usize::saturating_add);
        if bytes_before <= self.queue_bytes {
            return Some(QueueEviction {
                evicted: 0,
                bytes_before,
                bytes_after: bytes_before,
            });
        }

        let low_water = (self.queue_bytes / EVICT_LOW_WATER_DEN)
            .saturating_mul(EVICT_LOW_WATER_NUM)
            .max(1);
        let mut domains: Vec<GraphBabDomain> = std::mem::take(queue).into_vec();
        // Descending priority: the domains the heap would pop first sort first.
        domains.sort_unstable_by(|a, b| b.cmp(a));

        let mut bytes_after = 0usize;
        let mut keep = 0usize;
        for domain in &domains {
            let bytes = estimate_graph_domain_bytes(domain);
            if keep > 0 && bytes_after.saturating_add(bytes) > low_water {
                break;
            }
            bytes_after = bytes_after.saturating_add(bytes);
            keep += 1;
        }
        let evicted = domains.len() - keep;
        domains.truncate(keep);
        *queue = BinaryHeap::from(domains);

        Some(QueueEviction {
            evicted,
            bytes_before,
            bytes_after,
        })
    }
}

/// Enforce a graph heap's resident byte budget and latch completeness loss.
///
/// Every evicted domain is unexplored search space, so a nonzero eviction must
/// permanently prevent queue exhaustion from producing `Verified`. The route
/// label is telemetry only; both heap entry points use identical mechanics.
pub(crate) fn enforce_graph_queue_budget(
    budget: GraphBabQueueBudget,
    queue: &mut BinaryHeap<GraphBabDomain>,
    lifecycle: &mut GraphBabLifecycle,
    route: &'static str,
) {
    let Some(outcome) = budget.enforce(queue) else {
        return;
    };
    if outcome.evicted > 0 {
        lifecycle.unresolved_due_to_eviction = true;
        tracing::info!(
            route,
            evicted = outcome.evicted,
            queue_len = queue.len(),
            bytes_before = outcome.bytes_before,
            bytes_after = outcome.bytes_after,
            "Graph BaB: queue byte budget evicted unexplored domains (result forced to Unknown)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::result::BabVerificationStatus;
    use ny_tensor::BoundedTensor;

    /// Domain carrying `elements` f32 pairs of node bounds, so
    /// `estimate_graph_domain_bytes` is dominated by a predictable term.
    fn sized_domain(priority: f32, elements: usize) -> GraphBabDomain {
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .expect("invariant: symmetric bounds are valid");
        let mut node_bounds = std::collections::HashMap::new();
        node_bounds.insert(
            "n0".to_string(),
            BoundedTensor::new(
                ndarray::Array1::from_elem(elements, -1.0f32).into_dyn(),
                ndarray::Array1::from_elem(elements, 1.0f32).into_dyn(),
            )
            .expect("invariant: symmetric bounds are valid"),
        );
        let mut domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false)
            .expect("invariant: finite test bounds are valid");
        domain.priority = priority;
        domain
    }

    fn queue_of(priorities: &[f32], elements: usize) -> BinaryHeap<GraphBabDomain> {
        priorities
            .iter()
            .map(|&p| sized_domain(p, elements))
            .collect()
    }

    /// Default config leaves the budget disabled: `enforce` is a no-op and the
    /// wave cap is absent, so the BaB loop keeps its original behaviour.
    #[ntest::timeout(5000)]
    #[test]
    fn unlimited_budget_is_inert() {
        let budget = GraphBabQueueBudget::from_config(&BetaCrownConfig::default());
        assert!(budget.is_unlimited(), "shipped default must be unlimited");
        assert_eq!(budget.wave_bytes(), None);

        let mut queue = queue_of(&[1.0, 2.0, 3.0], 1024);
        assert_eq!(budget.enforce(&mut queue), None);
        assert_eq!(queue.len(), 3, "no-op enforcement must not touch the queue");
    }

    /// A queue inside its budget is reported but not modified.
    #[ntest::timeout(5000)]
    #[test]
    fn under_budget_keeps_every_domain() {
        let mut queue = queue_of(&[1.0, 2.0, 3.0], 1024);
        let resident: usize = queue.iter().map(estimate_graph_domain_bytes).sum();
        let budget = GraphBabQueueBudget::with_queue_bytes(resident * 2);

        let outcome = budget.enforce(&mut queue).expect("budget is armed");
        assert_eq!(outcome.evicted, 0);
        assert_eq!(outcome.bytes_before, resident);
        assert_eq!(outcome.bytes_after, resident);
        assert_eq!(queue.len(), 3);
    }

    /// THE MEMORY BOUND: an over-budget queue is cut back below the cap and the
    /// highest-priority domains — the ones BaB would expand next — survive.
    #[ntest::timeout(5000)]
    #[test]
    fn over_budget_evicts_lowest_priority_and_respects_the_cap() {
        let per_domain = estimate_graph_domain_bytes(&sized_domain(0.0, 1024));
        // Room for 4 domains; 16 are queued.
        let budget = GraphBabQueueBudget::with_queue_bytes(per_domain * 4);
        let priorities: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut queue = queue_of(&priorities, 1024);

        let outcome = budget.enforce(&mut queue).expect("budget is armed");
        assert!(outcome.evicted > 0, "over-budget queue must evict");
        assert_eq!(outcome.evicted, 16 - queue.len());
        assert!(
            outcome.bytes_after <= per_domain * 4,
            "resident bytes {} must respect the {} byte cap",
            outcome.bytes_after,
            per_domain * 4
        );
        let resident: usize = queue.iter().map(estimate_graph_domain_bytes).sum();
        assert_eq!(resident, outcome.bytes_after, "reported bytes must be real");

        // Survivors are the top of the priority order, popped highest first.
        let mut survivors: Vec<f32> = Vec::new();
        while let Some(domain) = queue.pop() {
            survivors.push(domain.priority);
        }
        let mut expected = priorities;
        expected.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
        expected.truncate(survivors.len());
        assert_eq!(survivors, expected);
    }

    /// Repeated enforcement is idempotent once the queue is inside the cap.
    #[ntest::timeout(5000)]
    #[test]
    fn enforcement_is_idempotent() {
        let per_domain = estimate_graph_domain_bytes(&sized_domain(0.0, 512));
        let budget = GraphBabQueueBudget::with_queue_bytes(per_domain * 3);
        let priorities: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut queue = queue_of(&priorities, 512);

        let first = budget.enforce(&mut queue).expect("armed");
        assert!(first.evicted > 0);
        let second = budget.enforce(&mut queue).expect("armed");
        assert_eq!(second.evicted, 0);
        assert_eq!(second.bytes_before, first.bytes_after);
    }

    /// A budget smaller than one domain still leaves the queue non-empty:
    /// forward progress is never traded for the byte bound.
    #[ntest::timeout(5000)]
    #[test]
    fn budget_below_one_domain_keeps_one_domain() {
        let budget = GraphBabQueueBudget::with_queue_bytes(1);
        let mut queue = queue_of(&[1.0, 2.0, 3.0], 4096);
        let outcome = budget.enforce(&mut queue).expect("armed");
        assert_eq!(queue.len(), 1);
        assert_eq!(outcome.evicted, 2);
        assert_eq!(
            queue.peek().expect("one domain left").priority,
            3.0,
            "the single survivor must be the highest-priority domain"
        );
    }

    /// Eviction is a completeness loss, not a proof: both heap routes use this
    /// helper so a later drained queue can never be reported as Verified.
    #[ntest::timeout(5000)]
    #[test]
    fn eviction_latches_lifecycle_and_forces_unknown() {
        let per_domain = estimate_graph_domain_bytes(&sized_domain(0.0, 1024));
        let budget = GraphBabQueueBudget::with_queue_bytes(per_domain);
        let mut queue = queue_of(&[1.0, 2.0, 3.0], 1024);
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());

        enforce_graph_queue_budget(budget, &mut queue, &mut lifecycle, "test-heap");

        assert!(queue.len() < 3);
        assert!(lifecycle.unresolved_due_to_eviction);
        assert!(matches!(
            lifecycle.build_final_result().result,
            BabVerificationStatus::Unknown { .. }
        ));
    }

    /// The wave cap is a fixed share of the queue budget and is always at
    /// least one byte so the caller's "always admit the first domain" rule is
    /// the only thing that can be binding.
    #[ntest::timeout(5000)]
    #[test]
    fn wave_bytes_is_a_share_of_the_budget() {
        let budget = GraphBabQueueBudget::with_queue_bytes(4096);
        assert_eq!(budget.wave_bytes(), Some(4096 / WAVE_BYTE_SHARE_DEN));
        assert_eq!(
            GraphBabQueueBudget::with_queue_bytes(1).wave_bytes(),
            Some(1)
        );
    }
}
