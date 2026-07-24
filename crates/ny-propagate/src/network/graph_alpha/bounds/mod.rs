// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN/IBP bound collection helpers for alpha-CROWN on `GraphNetwork`.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::{AlphaCrownConfig, GraphAlphaState, Optimizer};
use crate::layers::Layer;
use crate::network::backward_dispatch::{dispatch_backward_layer, DispatchContext};
use crate::network::core::{
    crown_backward_step_patches, CrownStepResult, GraphNetwork, GraphTargetShapeContract,
    NETWORK_INPUT,
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

mod alpha;
mod alpha_dag_dispatch;

pub(crate) use alpha::AlphaReferenceBoundsSource;
mod alpha_explicit;
pub(crate) mod budget_policy;
mod crown;
mod crown_tighten;
mod demand;
mod div;
mod gpu_suffix;
mod ibp;
mod ibp_batched;
mod patches_target;
mod reciprocal_support;
mod sequential;
mod sqrt_support;
mod target_backward;
mod target_backward_patches;
mod warm_start;

#[cfg(test)]
mod tests;

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
