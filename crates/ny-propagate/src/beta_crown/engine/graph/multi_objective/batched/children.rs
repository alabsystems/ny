// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective child collection helpers for GPU-batched BaB.

use crate::beta_crown::domain::{MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain};

use super::super::super::super::domain_results::MultiObjectiveGraphDomainResult;

/// Maximum collision-free parent-path identity retained in a KFSB receipt.
/// Exceeding it only disables this optional reuse path.
pub(in crate::beta_crown::engine::graph) const KFSB_CERT_PARENT_ID_MAX_BYTES: usize = 64 * 1024;

/// Region on which a reused KFSB lower-bound proof was established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum KfsbCertScope {
    /// A complete active/inactive simulation pair lower-bounds the parent.
    ParentCover,
    /// An exact committed ReLU literal is a subset of one simulated side.
    LiteralSide {
        node_name: String,
        neuron_idx: usize,
        is_active: bool,
    },
}

/// Exact authority carried by one KFSB certificate mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) struct KfsbCertReceipt {
    pub(in crate::beta_crown::engine::graph) row: usize,
    pub(in crate::beta_crown::engine::graph) scope: KfsbCertScope,
    /// Collision-free identity of the parent split path that authorized the
    /// proof. The consumer independently recomputes and checks this identity.
    pub(in crate::beta_crown::engine::graph) parent_history_identity: std::sync::Arc<[u8]>,
    pub(in crate::beta_crown::engine::graph) lower_bits: u32,
    pub(in crate::beta_crown::engine::graph) authority_deadline: std::time::Instant,
}

/// Authoritative effect of KFSB certificate reuse on one committed child.
///
/// `None` is deliberately explicit: downstream code must never infer KFSB
/// provenance from the child's verification mask or terminal state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum KfsbCertEffect {
    #[default]
    None,
    /// Exactly one target row became verified; other rows remain pending.
    RowVerified(KfsbCertReceipt),
    /// The target row became verified and completed the whole child.
    ChildComplete(KfsbCertReceipt),
    /// A parent-wide pair proof completed the untouched parent directly.
    /// This is not a split child and consumers must require an exact zero-
    /// suffix parent clone before bypassing evaluation.
    ParentComplete(KfsbCertReceipt),
}

impl KfsbCertEffect {
    pub(in crate::beta_crown::engine::graph) fn receipt(&self) -> Option<&KfsbCertReceipt> {
        match self {
            Self::None => None,
            Self::RowVerified(receipt)
            | Self::ChildComplete(receipt)
            | Self::ParentComplete(receipt) => Some(receipt),
        }
    }
}

pub(super) type MultiObjectiveChildCreationResult = (
    usize,
    Vec<(usize, MultiObjectiveGraphBabDomain, bool, KfsbCertEffect)>,
);
pub(super) type MultiObjectiveCollectedChildren =
    Vec<(usize, MultiObjectiveGraphBabDomain, bool, KfsbCertEffect)>;
pub(super) type MultiObjectiveParentLookup<'a> =
    std::collections::HashMap<usize, &'a MultiObjectiveGraphBabDomain>;

/// Collect multi-objective children from creation results, handling lookup failures.
///
/// Returns `(all_children, parent_domain_lookup)`. For parents that are not found
/// in `domains_with_unstable`, inserts `PropagationFailure` into `quick_results`.
pub(super) fn collect_multi_objective_children<'a>(
    domains_with_unstable: &'a [MultiObjDomainWithUnstable<'a>],
    child_creation_results: Vec<MultiObjectiveChildCreationResult>,
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
        let Some(parent_domain) = domain_by_idx.get(&parent_idx).copied() else {
            tracing::warn!(
                "process_graph_domains_batched_gpu_multi_objective: missing parent domain for idx {} (#1993)",
                parent_idx,
            );
            quick_results.insert(
                parent_idx,
                MultiObjectiveGraphDomainResult::PropagationFailure,
            );
            continue;
        };
        parent_domain_lookup.insert(parent_idx, parent_domain);

        for (_, child, is_active, cert_effect) in children_info {
            all_children.push((parent_idx, child, is_active, cert_effect));
        }
    }

    (all_children, parent_domain_lookup)
}
