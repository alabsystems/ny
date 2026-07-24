// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective child collection helpers for GPU-batched BaB.

use crate::beta_crown::domain::{MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain};

use super::super::super::super::domain_results::MultiObjectiveGraphDomainResult;

pub(super) type MultiObjectiveChildCreationResult =
    (usize, Vec<(usize, MultiObjectiveGraphBabDomain, bool)>);
pub(super) type MultiObjectiveCollectedChildren = Vec<(usize, MultiObjectiveGraphBabDomain, bool)>;
pub(super) type MultiObjectiveParentLookup<'a> =
    std::collections::HashMap<usize, &'a MultiObjectiveGraphBabDomain>;

/// Collect multi-objective children from creation results, handling lookup failures.
///
/// Returns `(all_children, parent_domain_lookup)`. For parents that are not found
/// in `domains_with_unstable`, inserts `PropagationFailure` into `quick_results`.
pub(super) fn collect_multi_objective_children<'a>(
    domains_with_unstable: &'a [MultiObjDomainWithUnstable<'a>],
    child_creation_results: &[MultiObjectiveChildCreationResult],
    quick_results: &mut std::collections::HashMap<usize, MultiObjectiveGraphDomainResult>,
) -> (
    MultiObjectiveCollectedChildren,
    MultiObjectiveParentLookup<'a>,
) {
    let mut all_children: MultiObjectiveCollectedChildren = Vec::new();
    let mut parent_domain_lookup: MultiObjectiveParentLookup<'a> = std::collections::HashMap::new();

    // O(1) index from parent idx → parent domain ref, replacing a per-iteration
    // linear `.find()` over `domains_with_unstable` (was O(D²) for batch size D,
    // up to thousands of domains). Each `idx` is the unique `.enumerate()` index
    // carried from `domains_to_process`, so a key maps to exactly one domain —
    // identical to the first-match `.find()` semantics.
    let domain_by_idx: std::collections::HashMap<usize, &'a MultiObjectiveGraphBabDomain> =
        domains_with_unstable
            .iter()
            .map(|(i, d, _)| (*i, *d))
            .collect();

    for (parent_idx, children_info) in child_creation_results {
        let Some(parent_domain) = domain_by_idx.get(parent_idx).copied() else {
            tracing::warn!(
                "process_graph_domains_batched_gpu_multi_objective: missing parent domain for idx {} (#1993)",
                parent_idx
            );
            quick_results.insert(
                *parent_idx,
                MultiObjectiveGraphDomainResult::PropagationFailure,
            );
            continue;
        };
        parent_domain_lookup.insert(*parent_idx, parent_domain);

        for (_, child, is_active) in children_info {
            all_children.push((*parent_idx, child.clone(), *is_active));
        }
    }

    (all_children, parent_domain_lookup)
}
