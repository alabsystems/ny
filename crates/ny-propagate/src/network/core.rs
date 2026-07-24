// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network types and graph representations for bound propagation.

pub(crate) mod crown_utils;
pub(crate) mod graph;
pub(crate) mod mode_mutators;
mod sequential;

pub(crate) use crown_utils::tighten_crown_with_forward_bounds;
pub(crate) use graph::{
    apply_dense_backward_dispatch_result, try_dense_spatial_patches_reentry,
    GraphTargetShapeContract,
};
pub use graph::{
    GraphNetwork, GraphNode, SoftmaxComplexReport, VggMaxPoolRewriteMode, VggMaxPoolRewriteReport,
    ZonotopePropagationOptions, ZonotopeSoftmaxMode, NETWORK_INPUT, SOFTMAX_COMPLEX_SHIFT_GUARD,
};
pub(crate) use sequential::crown_backward_step_patches;
pub(crate) use sequential::extract_relu_gpu_layer_with_alpha;
pub(crate) use sequential::tighten_crown_output;
pub(crate) use sequential::tighten_crown_output_with_provenance;
pub(crate) use sequential::try_extract_single_gpu_layer;
pub(crate) use sequential::CrownStepFallback;
pub(crate) use sequential::CrownStepResult;
pub use sequential::Network;
pub(crate) use sequential::{apply_bn_werr_to_host_relu, try_extract_batch_norm_conv1x1};

#[cfg(test)]
mod adain_crown_ibp_gate_tests;
#[cfg(test)]
mod mode_mutator_tests;
#[cfg(test)]
mod tests;
