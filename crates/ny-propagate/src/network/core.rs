// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network types and graph representations for bound propagation.

pub(crate) mod crown_utils;
pub(crate) mod graph;
pub(crate) mod mode_mutators;
pub(crate) mod sequential;

pub(crate) use crown_utils::tighten_crown_with_forward_bounds;
pub(crate) use graph::{
    apply_dense_backward_dispatch_result_with_deadline,
    try_dense_spatial_patches_reentry_with_deadline, GraphTargetShapeContract,
};
pub use graph::{
    compose_one_axis_dnf_observations, GraphNetwork, GraphNode, OneAxisAffineCertificate,
    OneAxisAlgebraClass, OneAxisAlgebraReport, OneAxisConstraintRelation, OneAxisCoreGuard,
    OneAxisDecline, OneAxisDeclineReason, OneAxisExactProblem, OneAxisGroupedContextCertificate,
    OneAxisGroupedMemberCertificate, OneAxisGroupedPhaseAttempt, OneAxisGroupedPhaseCertificate,
    OneAxisGroupedPhaseLimits, OneAxisGroupedReplayResult, OneAxisOutputConstraint,
    OneAxisPeeledConstraint, OneAxisPhaseAttempt, OneAxisPhaseCellCertificate,
    OneAxisPhaseCertificate, OneAxisPhaseDecline, OneAxisPhaseDeclineReason, OneAxisPhaseLimits,
    OneAxisPhaseObservation, OneAxisRational, OneAxisReplayResult, OneAxisWrapperEnclosure,
    SoftmaxComplexReport, VggMaxPoolRewriteMode, VggMaxPoolRewriteReport,
    ZonotopePropagationOptions, ZonotopeSoftmaxMode, NETWORK_INPUT,
    ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION, ONE_AXIS_MAX_EDGES, ONE_AXIS_MAX_NODES,
    ONE_AXIS_MAX_RANK, ONE_AXIS_MAX_TENSOR_ELEMENTS, ONE_AXIS_MAX_TOTAL_ELEMENTS,
    ONE_AXIS_PHASE_CERTIFICATE_VERSION, SOFTMAX_COMPLEX_SHIFT_GUARD,
};
pub(crate) use sequential::extract_relu_gpu_layer_with_alpha;
pub(crate) use sequential::materialize_terminal_crown_bounds_with_deadline;
pub(crate) use sequential::tighten_crown_output;
pub(crate) use sequential::tighten_crown_output_with_deadline;
pub(crate) use sequential::tighten_crown_output_with_provenance_and_deadline;
pub(crate) use sequential::try_extract_single_gpu_layer;
pub(crate) use sequential::CrownStepFallback;
pub(crate) use sequential::CrownStepResult;
pub use sequential::Network;
pub(crate) use sequential::{apply_bn_werr_to_host_relu, try_extract_batch_norm_conv1x1};
pub(crate) use sequential::{
    crown_backward_step_patches, crown_backward_step_patches_spec_crown,
    crown_backward_step_patches_with_deadline_authority, SpecPatchesStepError,
};
pub(crate) use sequential::{gpu_relu_affine_cell, GpuReluAffineVariant};

#[cfg(test)]
mod adain_crown_ibp_gate_tests;
#[cfg(test)]
mod mode_mutator_tests;
#[cfg(test)]
mod tests;
