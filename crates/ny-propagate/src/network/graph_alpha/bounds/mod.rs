// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN/IBP bound collection helpers for alpha-CROWN on `GraphNetwork`.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::{AlphaCrownConfig, GraphAlphaState, Optimizer};
use crate::layers::Layer;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, dispatch_backward_layer_finite_boundary, DispatchContext,
};
use crate::network::core::{
    crown_backward_step_patches_with_deadline_authority, CrownStepResult, GraphNetwork,
    GraphTargetShapeContract, NETWORK_INPUT,
};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::MulBinaryRelaxationMode;
use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, warn};

type GraphAlphaCollectionResult = (
    std::collections::HashMap<String, BoundedTensor>,
    GraphAlphaState,
);

/// Internal publication disposition for the root DAG-alpha collector.
///
/// A phase-cap checkpoint contains only a certified intermediate-bound map and
/// relaxation parameters.  It never carries verdict authority; a downstream
/// certified objective evaluation (CROWN or sound box projection) must
/// establish every claimed verdict independently of the optimizer fold.
pub(crate) enum GraphAlphaCollectionOutcome {
    Complete(GraphAlphaCollectionResult),
    PhaseCapCheckpoint {
        result: GraphAlphaCollectionResult,
        completed_iterations: usize,
        optimizer_updates_completed: usize,
    },
}

impl GraphAlphaCollectionOutcome {
    fn into_result(self) -> GraphAlphaCollectionResult {
        match self {
            Self::Complete(result) | Self::PhaseCapCheckpoint { result, .. } => result,
        }
    }
}

mod alpha;
mod alpha_dag_dispatch;
// Bug #19 budget-monotonicity: shrink-only publication of the DAG-alpha root
// intermediate map (NY_CENSUS_MONOTONE=1 / NY_CENSUS_COMMIT_TELEMETRY=1,
// default OFF => byte-identical). See census_commit.rs module docs.
mod census_commit;

pub(in crate::network::graph_alpha) use alpha::crown_ibp_collector_cap;
pub(crate) use alpha::AlphaReferenceBoundsSource;

/// A same-graph, same-input certified reference map supplied by an exact
/// caller-owned cache. The source travels with the map so DAG alpha preserves
/// typed publication/reuse semantics without inferring engagement from flags.
#[derive(Clone)]
pub(crate) struct PrecomputedAlphaReferenceBounds {
    pub(crate) bounds: std::collections::HashMap<String, BoundedTensor>,
    pub(crate) source: AlphaReferenceBoundsSource,
}
mod alpha_explicit;
#[cfg(test)]
pub(in crate::network::graph_alpha) use alpha_explicit::{
    run_with_m1_alpha_trace, M1AlphaBudgetOutcome, M1AlphaTraceEvent,
};
pub(crate) mod budget_policy;
// #cgan-stacked-backward (NY_CGAN_STACKED_BACKWARD=1, default OFF =>
// byte-identical): shared-prefix stacked backward planner for cgan trunk
// graphs. See docs/CGAN_STACKED_BACKWARD_2026-08-19.md.
mod cgan_stacked;
mod crown;
mod crown_repropagate;
#[cfg(test)]
pub(crate) use crown::CganCompleteCollectionEntryCounter;
mod crown_tighten;
mod demand;
// #root-joint-demand-rank: the collector's demand selector, re-exported so the
// armed root-joint interm-α lane ranks its targets by the SAME demanded list
// whose CROWN degradation it repairs (one selector, two consumers).
pub(crate) use demand::nodes_requiring_crown_tightening;
mod div;
mod gpu_suffix;
mod ibp;
mod ibp_batched;
mod patches_target;
mod reciprocal_support;
mod resident_patches_root;
mod sequential;
mod sqrt_support;
mod target_backward;
mod target_backward_patches;
mod warm_start;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "crown_repropagate_tests.rs"]
mod crown_repropagate_tests;

#[cfg(test)]
#[path = "gpu_suffix_tests.rs"]
mod gpu_suffix_tests;

#[cfg(test)]
#[path = "div_fallback_tests.rs"]
mod div_fallback_tests;

#[cfg(test)]
#[path = "channel_only_alpha_tests.rs"]
mod channel_only_alpha_tests;

#[cfg(test)]
#[path = "kokoro_tests.rs"]
mod kokoro_tests;

#[cfg(test)]
#[path = "residual_patches_tests.rs"]
mod residual_patches_tests;
