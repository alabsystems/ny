// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

//! MIP/LP solving for neural network verification, on the ay backend.
//!
//! SOLVER POLICY (docs/SOLVER_POLICY.md): all solving in ny happens on ay
//! (or ny's own native engines). HiGHS was deleted at LG3: verified ay
//! certificates are the independent cross-check, exercised by the
//! `mip-diff` gate's `--certify` mode.
//!
//! Provides complete MIP verification and LP bound tightening for FC+ReLU
//! networks.
//!
//! # Use Cases
//!
//! 1. **Complete MIP verification**: Encode network + property as MILP,
//!    check feasibility. SAT = counterexample, UNSAT = verified.
//!
//! 2. **LP bound tightening**: Solve LP relaxations to tighten intermediate
//!    neuron bounds, reducing the search space for BaB.
//!
//! # Architecture
//!
//! - [`ir`] — Solver-neutral MILP problem IR + per-backend lowerings
//! - [`encoder`] — Network → [`ir::MilpProblem`] encoding (Big-M ReLU)
//! - [`solver`] — Backend dispatch, solve + result extraction
//! - [`config`] — Solver configuration
//! - [`error`] — Typed errors via `thiserror`
//!
//! Part of #1763.

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

mod ay;
mod ay_lib;
pub mod certified_auxiliary_bounds64;
pub mod certified_box64;
mod certified_linear_lower;
pub mod config;
pub mod constrained_zonotope64;
pub mod constrained_zonotope_affine;
pub mod constrained_zonotope_axis_stabilizer;
pub mod constrained_zonotope_batched_adam;
pub mod constrained_zonotope_conv2d;
pub mod constrained_zonotope_coordinate_dual;
pub mod constrained_zonotope_dual;
pub mod constrained_zonotope_relu;
pub mod constrained_zonotope_relu_tail_dual;
pub mod constrained_zonotope_tail_lp;
pub mod dump;
pub mod encoder;
pub mod error;
pub mod ir;
pub mod solver;
pub mod tighten;
// Unwired measurement primitive (verified SDP lower bound; VSDP method). Not on
// any verdict path; soundness proven by the module's own oracle tests.
pub mod verified_sdp_bound;

pub use ay_lib::{prove_infeasible_with_row_farkas, RowSide};
pub use certified_auxiliary_bounds64::{
    CertifiedAuxiliaryBounds64, CertifiedAuxiliaryBounds64Error,
};
pub use certified_box64::{
    certified_box_affine_unwired, certified_box_conv2d_unwired, unconstrained_zonotope_box_unwired,
    CertifiedBox64, CertifiedBox64Error, CertifiedBox64Limits, CertifiedBoxAffinePlan,
    CertifiedBoxConv2dPlan, CertifiedBoxHullPlan,
};
pub use certified_linear_lower::{
    certify_linear_lower_bound_at_with_ay, certify_linear_lower_bound_at_with_ay_admission,
    certify_linear_lower_bound_with_ay, certify_linear_lower_bound_with_ay_admission,
    CertifiedLinearLowerBound, CertifiedLinearLowerBoundConfig, CertifiedLinearLowerDecisionConfig,
    CertifiedLinearLowerProofRoute, CertifiedLinearLowerWorkerAdmission,
    CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
};
pub use config::{MipBackend, MipConfig};
pub use constrained_zonotope64::{
    ConstrainedZonotope64, ConstrainedZonotope64Error, SparseGenerator64,
};
pub use constrained_zonotope_affine::{
    constrained_zonotope_affine_unwired, ConstrainedZonotopeAffineError,
    ConstrainedZonotopeAffineLimits, ConstrainedZonotopeAffinePlan,
};
pub use constrained_zonotope_axis_stabilizer::{
    propose_axis_stabilization_unwired, AxisStabilizerError, AxisStabilizerLimits,
    AxisStabilizerPlan, AxisStabilizerProposal, CertifiedAxisPhase, CertifiedAxisProposal,
    AXIS_STABILIZER_HARD_MAX_AXES, AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS,
    AXIS_STABILIZER_HARD_MAX_VALUE_DIM,
};
pub use constrained_zonotope_batched_adam::{
    propose_batched_adam_unwired, BatchedAdamConfig, BatchedAdamLimits, BatchedAdamPlan,
    BatchedAdamProposal, BatchedAdamProposerError, BatchedAdamStatus,
    BATCHED_ADAM_HARD_MAX_ALPHA_DIM, BATCHED_ADAM_HARD_MAX_BASELINE_DUAL_TERMS,
    BATCHED_ADAM_HARD_MAX_CONSTRAINTS, BATCHED_ADAM_HARD_MAX_CONSTRAINT_ELEMENTS,
    BATCHED_ADAM_HARD_MAX_DIRECTIONS, BATCHED_ADAM_HARD_MAX_DIRECTION_ELEMENTS,
    BATCHED_ADAM_HARD_MAX_GEMM_PRODUCTS, BATCHED_ADAM_HARD_MAX_GENERATOR_NONZEROS,
    BATCHED_ADAM_HARD_MAX_ITERATIONS, BATCHED_ADAM_HARD_MAX_MULTIPLIER_ELEMENTS,
    BATCHED_ADAM_HARD_MAX_PROJECTION_PRODUCTS, BATCHED_ADAM_HARD_MAX_VALUE_DIM,
    BATCHED_ADAM_HARD_MAX_WALL_TIME, BATCHED_ADAM_HARD_MAX_WORKING_F32_ELEMENTS,
};
pub use constrained_zonotope_conv2d::{
    constrained_zonotope_conv2d_unwired, ConstrainedZonotopeConv2dError,
    ConstrainedZonotopeConv2dLimits, ConstrainedZonotopeConv2dPlan, ConstrainedZonotopeConv2dSpec,
};
pub use constrained_zonotope_coordinate_dual::{
    propose_coordinate_dual_unwired, CoordinateDualConfig, CoordinateDualLimits,
    CoordinateDualProposal, CoordinateDualProposerError,
};
pub use constrained_zonotope_dual::{
    evaluate_constrained_zonotope64_dual, evaluate_constrained_zonotope_dual,
    evaluate_constrained_zonotope_dual_with_box_remainder, ConstrainedZonotopeDualBounds,
    ConstrainedZonotopeDualError,
};
pub use constrained_zonotope_relu::{
    transform_relu_projected_constraints_unwired,
    transform_relu_projected_constraints_with_auxiliary_bounds_unwired, transform_relu_unwired,
    transform_relu_with_auxiliary_bounds_unwired, ReluTransformError, ReluTransformLimits,
    RELU_HARD_MAX_CONSTRAINTS, RELU_HARD_MAX_CONSTRAINT_ELEMENTS, RELU_HARD_MAX_EXACT_TERMS,
    RELU_HARD_MAX_GENERATOR_NNZ, RELU_HARD_MAX_OUTPUT_ALPHA_DIM, RELU_HARD_MAX_UNSTABLE,
    RELU_HARD_MAX_VALUE_DIM,
};
pub use constrained_zonotope_relu_tail_dual::{
    bound_relu_tail_triangle_dual_unwired,
    bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired,
    bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired,
    exact_relu_tail_margin_from_f64_rows, prepare_relu_tail_triangle_dual_unwired,
    ExactReluTailMargin, PreparedReluTailGeometry64, ReluTailBoxCutAdamSchedule,
    ReluTailBoxCutCertificate, ReluTailBoxCutDualResult, ReluTailBoxCutOptimizedResult,
    ReluTailBoxCutOptimizerConfig, ReluTailBoxCutOptimizerLimits, ReluTailBoxCutOptimizerPlan,
    ReluTailBoxCutOptimizerStatus, ReluTailBoxCutSelection, ReluTailBoxCutStatus,
    ReluTailDualConfig, ReluTailDualError, ReluTailDualLimits, ReluTailDualPlan,
    ReluTailDualResult, ReluTailDualStatus, RELU_TAIL_BOX_CUT_HARD_MAX_EXACT_REPLAYS,
    RELU_TAIL_BOX_CUT_HARD_MAX_ITERATIONS, RELU_TAIL_BOX_CUT_HARD_MAX_MULTIPLIER,
    RELU_TAIL_BOX_CUT_HARD_MAX_RESTARTS, RELU_TAIL_BOX_CUT_HARD_MAX_SEARCH_WORK,
    RELU_TAIL_BOX_CUT_HARD_MAX_VARIABLES, RELU_TAIL_BOX_CUT_HARD_MAX_WALL_TIME,
    RELU_TAIL_DUAL_HARD_MAX_ALPHA_DIM, RELU_TAIL_DUAL_HARD_MAX_BASELINE_TERMS,
    RELU_TAIL_DUAL_HARD_MAX_CONSTRAINTS, RELU_TAIL_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS,
    RELU_TAIL_DUAL_HARD_MAX_GENERATOR_NONZEROS, RELU_TAIL_DUAL_HARD_MAX_INPUT_RATIONAL_BITS,
    RELU_TAIL_DUAL_HARD_MAX_INTERMEDIATE_RATIONAL_BITS, RELU_TAIL_DUAL_HARD_MAX_ITERATIONS,
    RELU_TAIL_DUAL_HARD_MAX_OPTIMIZABLE_SLOPES, RELU_TAIL_DUAL_HARD_MAX_SEARCH_WORK,
    RELU_TAIL_DUAL_HARD_MAX_TOTAL_RATIONAL_BITS, RELU_TAIL_DUAL_HARD_MAX_VALUE_DIM,
    RELU_TAIL_DUAL_HARD_MAX_WALL_TIME,
};
pub use constrained_zonotope_tail_lp::{
    diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired,
    diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired,
    ConstrainedZonotopeTailLpConfig, ConstrainedZonotopeTailLpDiagnostic,
    ConstrainedZonotopeTailLpError, ConstrainedZonotopeTailLpLimits, ConstrainedZonotopeTailLpPlan,
    TailLpExactMilpAssessment, TailLpInconclusiveReason, TailLpMarginDiagnostic,
    TailLpMarginOutcome,
};
pub use encoder::{encode_feedforward, MipEncoder, MipParts};
pub use error::MipError;
pub use ir::MilpProblem;
pub use solver::{MipResult, MipSolver, Sense, SplitUnsatCache};
pub use tighten::{obbt_relaxation_bounds, LpTightener, RelaxationObbt};

/// Convenience type alias for Results using [`MipError`].
pub type Result<T> = std::result::Result<T, MipError>;

/// Serialize a [`MilpProblem`] to exact-rational QF_LRA SMT-LIB in the
/// ay-native DECISION form (`_dec` corpus spelling): `check-sat` over the
/// encoded constraints, with ReLU indicator binaries lowered to `{0,1}`
/// disjunctions and every f64 bound/coefficient emitted as its precise dyadic
/// rational. This is the exact byte stream the ay backend streams to the
/// solver on a feasibility check — routed here for standalone corpus capture.
pub fn to_smtlib_decision(problem: &MilpProblem) -> Result<String> {
    ay::to_smtlib(problem, None)
}

/// Serialize a [`MilpProblem`] to exact-rational QF_LRA SMT-LIB in the
/// OPTIMIZATION form (`_min` corpus spelling): the same declarations and
/// assertions as [`to_smtlib_decision`] plus a `(minimize c<col>)` directive
/// over `col`. Identical exact-rational lowering; matches the byte stream the
/// ay backend streams on `optimize_col` (minimize lane).
pub fn to_smtlib_minimize(problem: &MilpProblem, col: ir::Col) -> Result<String> {
    ay::to_smtlib(
        problem,
        Some(ay::ObjectiveSpec {
            col,
            sense: ay::ObjSense::Minimize,
        }),
    )
}

#[cfg(test)]
mod tests;
