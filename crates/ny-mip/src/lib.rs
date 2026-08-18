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
// Unwired falsification primitive (LP-guided sign-space search over binarized
// `Sign` conv suffixes; docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md). Not
// on any verdict path: its outcome enum has no verified/unsat variant by
// construction, so it can only ever propose a witness for the caller's
// existing, unweakened validation gate to accept or reject.
pub mod bnn_sign_space;
pub mod certified_auxiliary_bounds64;
pub mod certified_box64;
mod certified_linear_lower;
pub mod config;
pub mod constrained_zonotope64;
pub mod constrained_zonotope_affine;
pub mod constrained_zonotope_alpha_bisection;
pub mod constrained_zonotope_axis_stabilizer;
pub mod constrained_zonotope_ay_lp_dual;
pub mod constrained_zonotope_batch_norm;
pub mod constrained_zonotope_batched_adam;
pub mod constrained_zonotope_call_budget;
pub mod constrained_zonotope_conv2d;
pub mod constrained_zonotope_conv_transpose2d;
pub mod constrained_zonotope_coordinate_dual;
pub mod constrained_zonotope_dual;
pub mod constrained_zonotope_order_reduction;
pub mod constrained_zonotope_relu;
pub mod constrained_zonotope_relu_tail_dual;
pub mod constrained_zonotope_tail_lp;
pub mod dump;
pub mod encoder;
pub mod error;
pub mod ir;
pub mod shared_tree_profile;
pub mod solver;
pub mod star_dual;
pub mod star_lp;
pub mod star_verify;

#[cfg(test)]
#[path = "star_acasxu_tests.rs"]
mod star_acasxu_tests;
pub mod tighten;
// Unwired measurement primitive (verified SDP lower bound; VSDP method). Not on
// any verdict path; soundness proven by the module's own oracle tests.
pub mod verified_sdp_bound;

pub use ay_lib::{prove_infeasible_with_row_farkas, RowSide};
pub use bnn_sign_space::TrustRegion;
pub use bnn_sign_space::{
    classify_first_layer_unwired, f32_replay_slack_floor, falsify_bnn_sign_suffix_unwired,
    logits_at_unwired, realizability_probe_unwired, BinaryStage, ConvSpec, InputGeometry, PoolSpec,
    ReferenceForward, SegmentMove, SignSpaceActivation, SignSpaceAffine, SignSpaceCandidate,
    SignSpaceError, SignSpaceLimits, SignSpaceOutcome, SignSpaceRefusal, SignSpaceRequest,
    UnitClassification, UnitPhase, SIGN_SPACE_EXACT_ACCUMULATION_LIMIT, SIGN_SPACE_HARD_MAX_FLAT,
    SIGN_SPACE_HARD_MAX_FREE_UNITS, SIGN_SPACE_HARD_MAX_LP_COLUMNS,
    SIGN_SPACE_HARD_MAX_LP_NONZEROS, SIGN_SPACE_HARD_MAX_LP_ROWS, SIGN_SPACE_HARD_MAX_LP_SOLVES,
    SIGN_SPACE_HARD_MAX_POOL_AREA, SIGN_SPACE_HARD_MAX_STAGES, SIGN_SPACE_HARD_MAX_UNITS,
    SIGN_SPACE_HARD_MAX_WALL_TIME,
};
pub use certified_auxiliary_bounds64::{
    CertifiedAuxiliaryBounds64, CertifiedAuxiliaryBounds64BudgetError,
    CertifiedAuxiliaryBounds64Error,
};
pub use certified_box64::{
    certified_box_affine_unwired, certified_box_conv2d_unwired,
    certified_box_from_remainder_only_zonotope_unwired_with_budget,
    certified_box_relu_recenter_unwired_with_budget, unconstrained_zonotope_box_unwired,
    CertifiedBox64, CertifiedBox64BridgeError, CertifiedBox64Error, CertifiedBox64Limits,
    CertifiedBoxAffinePlan, CertifiedBoxConv2dPlan, CertifiedBoxHullPlan,
};
pub use certified_linear_lower::{
    certify_continuous_root_infeasibility_with_ay_until,
    certify_continuous_root_infeasibility_with_ay_until_admission,
    certify_linear_lower_bound_at_with_ay,
    certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_admission,
    certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_adaptive_four_leaf_comb_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_adaptive_three_leaf_target_fsb_unwired,
    certify_linear_lower_bound_at_with_ay_admission,
    certify_linear_lower_bound_at_with_ay_branch_advice,
    certify_linear_lower_bound_at_with_ay_branch_advice_admission,
    certify_linear_lower_bound_at_with_ay_branch_advice_with_target_fsb_probe_limits_unwired,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_range_logical_admission,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_selector_solve_profile_admission,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_until_unwired,
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_unwired,
    certify_linear_lower_bound_at_with_ay_parallel_selector_tree_admission,
    certify_linear_lower_bound_at_with_ay_parallel_selector_tree_unwired,
    certify_linear_lower_bound_with_ay, certify_linear_lower_bound_with_ay_admission,
    CertifiedContinuousRootInfeasibility, CertifiedLinearLowerBound,
    CertifiedLinearLowerBoundConfig, CertifiedLinearLowerDecisionConfig,
    CertifiedLinearLowerProofRoute, CertifiedLinearLowerTargetFsbProbeLimits,
    CertifiedLinearLowerWorkerAdmission, CERTIFIED_LINEAR_LOWER_COMPACT_TREE_PREFIX_PROBE,
    CERTIFIED_LINEAR_LOWER_COMPACT_TREE_ROOT_PROBE, CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
    CERTIFIED_LINEAR_LOWER_SELECTOR_CHAIN_DISTRESS_PROBE_ITERS,
};
pub use config::{MipBackend, MipConfig, MipFeasibilityIngress};
pub use constrained_zonotope64::{
    ConstrainedZonotope64, ConstrainedZonotope64Error, SparseGenerator64,
};
pub use constrained_zonotope_affine::{
    constrained_zonotope_affine_unwired, constrained_zonotope_affine_unwired_with_budget,
    ConstrainedZonotopeAffineBudgetError, ConstrainedZonotopeAffineError,
    ConstrainedZonotopeAffineLimits, ConstrainedZonotopeAffinePlan,
};
pub use constrained_zonotope_alpha_bisection::{
    bisect_constrained_zonotope_protected_alpha_unwired,
    bisect_constrained_zonotope_protected_alpha_unwired_with_budget,
    ConstrainedZonotopeAlphaBisection, ConstrainedZonotopeAlphaBisectionBudgetError,
    ConstrainedZonotopeAlphaBisectionError, ConstrainedZonotopeAlphaBisectionLimits,
    ConstrainedZonotopeAlphaBisectionPlan,
};
pub use constrained_zonotope_axis_stabilizer::{
    propose_axis_stabilization_unwired, AxisStabilizerError, AxisStabilizerLimits,
    AxisStabilizerPlan, AxisStabilizerProposal, CertifiedAxisPhase, CertifiedAxisProposal,
    AXIS_STABILIZER_HARD_MAX_AXES, AXIS_STABILIZER_HARD_MAX_DIRECTION_ELEMENTS,
    AXIS_STABILIZER_HARD_MAX_VALUE_DIM,
};
pub use constrained_zonotope_ay_lp_dual::{
    propose_ay_lp_dual_unwired, AyLpDualConfig, AyLpDualLimits, AyLpDualProposal,
    AyLpDualProposerError, AY_LP_DUAL_HARD_MAX_ALPHA_DIM,
    AY_LP_DUAL_HARD_MAX_CERTIFICATE_MULTIPLIERS, AY_LP_DUAL_HARD_MAX_CERTIFICATE_RATIONAL_BITS,
    AY_LP_DUAL_HARD_MAX_CERTIFICATE_TOTAL_BITS, AY_LP_DUAL_HARD_MAX_CONSTRAINTS,
    AY_LP_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS, AY_LP_DUAL_HARD_MAX_CONSTRAINT_NONZEROS,
    AY_LP_DUAL_HARD_MAX_GENERATOR_NONZEROS, AY_LP_DUAL_HARD_MAX_MEMORY_BYTES,
};
pub use constrained_zonotope_batch_norm::{
    certify_batch_norm_affine_surrogate_unwired,
    certify_batch_norm_affine_surrogate_unwired_with_budget,
    constrained_zonotope_batch_norm_unwired, constrained_zonotope_batch_norm_unwired_with_budget,
    ConstrainedZonotopeBatchNormAffineCertificateLimits, ConstrainedZonotopeBatchNormBudgetError,
    ConstrainedZonotopeBatchNormError, ConstrainedZonotopeBatchNormLimits,
    ConstrainedZonotopeBatchNormMode, ConstrainedZonotopeBatchNormPlan,
    ConstrainedZonotopeBatchNormSpec, ExactBatchNormAffineSurrogateCertificate,
    ExactBatchNormChannelAffineCertificate,
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
pub use constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallAttempt, ConstrainedZonotopeCallBudget,
    ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallOutcome,
    ConstrainedZonotopeCallReport, CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL,
};
pub use constrained_zonotope_conv2d::{
    constrained_zonotope_conv2d_unwired, constrained_zonotope_conv2d_unwired_with_budget,
    ConstrainedZonotopeConv2dBudgetError, ConstrainedZonotopeConv2dError,
    ConstrainedZonotopeConv2dLimits, ConstrainedZonotopeConv2dPlan, ConstrainedZonotopeConv2dSpec,
};
pub use constrained_zonotope_conv_transpose2d::{
    constrained_zonotope_conv_transpose2d_unwired,
    constrained_zonotope_conv_transpose2d_unwired_with_budget,
    ConstrainedZonotopeConvTranspose2dBudgetError, ConstrainedZonotopeConvTranspose2dError,
    ConstrainedZonotopeConvTranspose2dLimits, ConstrainedZonotopeConvTranspose2dPlan,
    ConstrainedZonotopeConvTranspose2dSpec,
};
pub use constrained_zonotope_coordinate_dual::{
    propose_coordinate_dual_unwired, propose_coordinate_dual_unwired_with_budget,
    CoordinateDualBudgetError, CoordinateDualConfig, CoordinateDualLimits, CoordinateDualProposal,
    CoordinateDualProposerError,
};
pub use constrained_zonotope_dual::{
    evaluate_constrained_zonotope64_dual, evaluate_constrained_zonotope64_dual_with_budget,
    evaluate_constrained_zonotope_dual, evaluate_constrained_zonotope_dual_with_box_remainder,
    evaluate_constrained_zonotope_dual_with_box_remainder_and_budget,
    evaluate_constrained_zonotope_dual_with_budget, ConstrainedZonotopeDualBounds,
    ConstrainedZonotopeDualBudgetError, ConstrainedZonotopeDualError,
};
pub use constrained_zonotope_order_reduction::{
    constrained_zonotope_order_reduce_unwired_with_budget,
    ConstrainedZonotopeOrderReductionBudgetError, ConstrainedZonotopeOrderReductionError,
    ConstrainedZonotopeOrderReductionLimits, ConstrainedZonotopeOrderReductionPlan,
    ORDER_REDUCTION_HARD_MAX_CONSTRAINTS, ORDER_REDUCTION_HARD_MAX_CONSTRAINT_ELEMENTS,
    ORDER_REDUCTION_HARD_MAX_GENERATOR_NNZ, ORDER_REDUCTION_HARD_MAX_INPUT_ALPHA_DIM,
    ORDER_REDUCTION_HARD_MAX_OUTPUT_ALPHA_DIM, ORDER_REDUCTION_HARD_MAX_VALUE_DIM,
};
pub use constrained_zonotope_relu::{
    transform_relu_projected_constraints_unwired,
    transform_relu_projected_constraints_unwired_with_budget,
    transform_relu_projected_constraints_with_auxiliary_bounds_unwired,
    transform_relu_projected_constraints_with_auxiliary_bounds_unwired_with_budget,
    transform_relu_unwired, transform_relu_unwired_with_budget,
    transform_relu_with_auxiliary_bounds_unwired,
    transform_relu_with_auxiliary_bounds_unwired_with_budget, ReluTransformBudgetError,
    ReluTransformError, ReluTransformLimits, RELU_HARD_MAX_CONSTRAINTS,
    RELU_HARD_MAX_CONSTRAINT_ELEMENTS, RELU_HARD_MAX_EXACT_TERMS, RELU_HARD_MAX_GENERATOR_NNZ,
    RELU_HARD_MAX_OUTPUT_ALPHA_DIM, RELU_HARD_MAX_UNSTABLE, RELU_HARD_MAX_VALUE_DIM,
};
pub use constrained_zonotope_relu_tail_dual::{
    bound_relu_tail_triangle_dual_unwired, bound_relu_tail_triangle_dual_unwired_with_budget,
    bound_relu_tail_triangle_dual_with_auxiliary_bounds_unwired,
    bound_relu_tail_triangle_dual_with_auxiliary_box_cut_unwired,
    exact_relu_tail_margin_from_f64_rows, prepare_relu_tail_triangle_dual_unwired,
    ExactReluTailMargin, PreparedReluTailGeometry64, ReluTailBoxCutAdamSchedule,
    ReluTailBoxCutBudgetedResult, ReluTailBoxCutCertificate, ReluTailBoxCutDualResult,
    ReluTailBoxCutOptimizedResult, ReluTailBoxCutOptimizerConfig, ReluTailBoxCutOptimizerLimits,
    ReluTailBoxCutOptimizerPlan, ReluTailBoxCutOptimizerStatus, ReluTailBoxCutSelection,
    ReluTailBoxCutStatus, ReluTailConvBatchNormPullbackBudgetError,
    ReluTailConvBatchNormPullbackError, ReluTailConvBatchNormPullbackLimits,
    ReluTailConvBatchNormPullbackM17M20Result, ReluTailConvBatchNormPullbackPlan,
    ReluTailConvBatchNormPullbackResult, ReluTailDualBudgetError, ReluTailDualConfig,
    ReluTailDualError, ReluTailDualLimits, ReluTailDualPlan, ReluTailDualResult,
    ReluTailDualStatus, RELU_TAIL_BOX_CUT_HARD_MAX_EXACT_REPLAYS,
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
pub use constrained_zonotope_relu_tail_dual::{
    prepare_relu_tail_triangle_dual_unwired_attempt_with_budget,
    prepare_relu_tail_triangle_dual_unwired_with_budget,
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
pub use solver::{
    MipResult, MipSolver, OneSidedSatDecline, OneSidedSatProbe, OneSidedSatWitness, Sense,
    SplitUnsatCache,
};
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
