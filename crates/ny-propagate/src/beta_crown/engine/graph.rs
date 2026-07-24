// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork-specific β-CROWN verifier logic.

use std::collections::HashMap;
use std::sync::Arc;

use ny_tensor::BoundedTensor;

use crate::bounds::GraphAlphaCrownIntermediate;
use crate::LinearBounds;

/// Per-domain CROWN result: output bounds + intermediate node bounds cache.
///
/// The first element is the propagated output bounds for this domain.
/// The second is a map from node name to the bounded tensor at that node,
/// used as a cache for child-domain forward passes in BaB.
///
/// #cone-delta increment 2: entries are `Arc`-shared. Nodes the split's cone
/// did not touch alias the parent domain's tensors (`Arc::clone`); recomputed
/// nodes carry fresh allocations. Values are identical to the historical
/// deep-cloned map by construction — only allocation/ownership changed.
pub(crate) type DomainCrownResult = (BoundedTensor, HashMap<String, Arc<BoundedTensor>>);

/// Per-domain dense-spec CROWN result: output bounds + node cache + input linear bounds.
///
/// Extends `DomainCrownResult` with the final input `LinearBounds` from the CROWN
/// backward pass. The input linear data is needed by input-split scoring for:
/// - split-dimension scoring (SB heuristic),
/// - clip reuse,
/// - multi-objective pruning parity.
///
/// Part of #4116 Packet A: dense-spec result surface.
#[derive(Debug)]
pub struct DomainSpecCrownResult {
    pub output_bounds: BoundedTensor,
    /// Per-node bounds map, `Arc`-shared with the source forward cache
    /// (#cone-delta increment 2): installing it on a domain is a move, not a
    /// re-Arc deep clone.
    pub node_bounds: HashMap<String, Arc<BoundedTensor>>,
    pub input_linear: Option<LinearBounds>,
}

/// Per-domain CROWN result with intermediate A-matrices for gradient computation.
///
/// Extends `DomainCrownResult` with `GraphAlphaCrownIntermediate` which stores
/// the A matrices at constrained ReLU nodes, enabling analytical β gradient
/// computation without additional forward passes.
pub(crate) type DomainCrownResultWithIntermediates = (
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
    GraphAlphaCrownIntermediate,
);

/// Multi-objective verification result: per-objective (lower, upper) bounds + node bounds.
///
/// Used by multi-objective β-CROWN and analytical gradient optimization.
/// The first element contains scalar bounds for each objective (e.g., `Y_i - Y_j`).
/// The second is the shared node bounds cache from the first objective's propagation.
#[cfg(test)]
pub(crate) type MultiObjectiveResult = (Vec<(f32, f32)>, HashMap<String, Arc<BoundedTensor>>);

pub(crate) mod adaptive_microbatch;
pub(in crate::beta_crown::engine::graph) mod clip_alpha;
pub(in crate::beta_crown::engine::graph) mod clip_complete;
pub(crate) mod domain_batch;
pub(crate) mod domain_conversion;
#[cfg(test)]
pub(crate) mod forward_mode_test_support;
pub(crate) mod gpu_bab;
#[doc(hidden)]
pub mod gpu_beta_debug;
pub(crate) mod input_split;
mod multi_objective;
mod objectives;
pub(crate) mod propagation;
mod relu_split;
mod relu_split_bounds;
pub(crate) mod shared;

#[cfg(test)]
pub(crate) use relu_split_bounds::test_non_finite_domain_result_in_relu_split_bounds;

#[cfg(test)]
mod tests;
