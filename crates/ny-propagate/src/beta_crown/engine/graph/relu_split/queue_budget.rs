// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Estimated payload budget for graph ReLU-split branch-and-bound queues
//! (#ml4acopf-bab-queue-mem), shared by ordinary and precomputed-bounds routes.
//!
//! When `max_queue_bytes` is exceeded, retain the highest-priority prefix down
//! to a low-water mark. Always keep one domain, even if it alone exceeds the
//! budget. The ordinary loop targets a fraction of that budget for each popped
//! wave, again admitting at least one domain. These estimates do not cap total
//! process memory.
//!
//! Eviction discards unexplored domains without changing surviving bounds.
//! The caller latches `unresolved_due_to_eviction`, preventing `Verified` when
//! the queue drains; a later concrete counterexample can still refute the spec.
//! Zero (the default) disables budgeting without traversing the queue.

use std::collections::BinaryHeap;

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::adaptive_microbatch::estimate_graph_domain_bytes;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

/// Popped-wave target: `1 / WAVE_BYTE_SHARE_DEN` of the queue's payload budget.
/// The caller always admits its first domain, even if that exceeds the target.
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
    /// Estimated resident payload cap; one domain may exceed it. `0` disables it.
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
    /// Popped domains live outside the heap. The caller always admits the first
    /// domain for forward progress, even if it exceeds this estimate.
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
    /// traversed in that case. Otherwise returns estimated before/after payload
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
