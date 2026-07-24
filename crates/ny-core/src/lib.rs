// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]
// Under tRustc contract verification (`--cfg trust_verify`) the first-class
// `#[ensures]`/`#[requires]` attributes live in the unstable `core::contracts`
// module; this feature gate enables them. Stable rustc never sees this (cfg-gated)
// and falls back to the NY-owned `trust` compatibility macros. Mirrors ny-cert.
#![cfg_attr(trust_verify, feature(contracts))]

//! Core types and traits for ny neural network verification.
//!
//! This crate provides the foundational abstractions for bound propagation
//! in neural networks, enabling formal verification of properties like
//! robustness and equivalence.
//!
//! Implementation details live in module files; lib.rs is intentionally kept
//! small and focused on re-exports.

mod axis;
mod bound;
mod bound_tightener;
mod counterexample;
mod custom_op;
/// Double-double (two-f64) compensated arithmetic + certified error envelope
/// (`#dd-zonotope`).
pub mod dd;
/// One-time known-answer bit-exactness probe authorizing the double-double path.
pub mod dd_selfcheck;
mod display;
pub mod eft;
mod error;
mod gemm;
/// #mn-head-facet HEAD coupling-facet f64-recovery fold registry (shared seam).
pub mod head_f64_fold;
pub mod joint_alpha_grad;
mod layer_output;
mod layer_type;
mod nan_math;
mod output_constraint;
/// Floating-point precision tags for mixed-precision verification (P8).
pub mod precision;
mod proof;
mod reshape_copy_axis;
/// Certified Cut-CROWN resident cut-fold registry (shared writer/reader seam).
pub mod resident_cut_fold;
mod soundness;
mod verification_result;
mod verification_spec;
mod violation;

/// ONNX axis resolution with negative-index handling and bounds validation.
pub use axis::{resolve_axis, resolve_axis_i32};
/// Scalar bound type representing an interval [lower, upper].
pub use bound::Bound;
/// Trait for external bound tightening backends (LP, MIP, etc.).
pub use bound_tightener::BoundTightener;
/// Rich counterexample with per-layer intermediate values for debugging.
pub use counterexample::InformativeCounterexample;
/// Concrete counterexample (input/output) and per-element interval bounds (P6).
pub use counterexample::{Counterexample, PerElementBounds};
/// Custom operator schema metadata used by ONNX loader-time registration.
pub use custom_op::{
    CustomOpAttribute, CustomOpAttributeType, CustomOpAttributeValue, CustomOpSchema,
    CustomOpSchemaRegistry, CustomOpSpec,
};
/// Truncate a name to fit within a given width, prepending "..." if too long.
pub use display::truncate_name;
/// Error and result types for ny operations.
pub use error::{NyError, Result};
/// Sound activation-backward coefficient + error propagation (CROWN GPU-resident
/// backward, task #15 increment 3).
pub use gemm::crown_activation_error_step;
/// FTZ-safe additive underflow floor for sound f32 GPU error terms (Metal-safe).
pub use gemm::ftz_safe_underflow_floor;
/// Check whether a CROWN A-matrix coefficient is within safe magnitude bounds (#1932).
pub use gemm::is_crown_coeff_safe;
/// f64 variant of [`is_crown_coeff_safe`] for normalization CROWN backward paths (#3228).
pub use gemm::is_crown_coeff_safe_f64;
/// Parameters for fused GPU conv_transpose_2d (GEMM + col2im) (#3813).
pub use gemm::ConvTranspose2dParams;
/// GEMM engine abstraction for pluggable matrix multiplication backends.
pub use gemm::GemmEngine;
/// GPU-accelerated CROWN backward pass trait — keeps A-matrices on device (#3397).
pub use gemm::GpuCrownBackward;
/// Result from a beta-gradient GPU CROWN resnet backward: β-folded bounds + gathered A-values.
pub use gemm::GpuCrownBetaGradResult;
/// Result from a gradient-capturing GPU CROWN resnet backward: bounds + per-ReLU grads.
pub use gemm::GpuCrownGradResult;
/// Per-layer descriptor for GPU CROWN backward (Linear or Activation).
pub use gemm::GpuCrownLayer;
/// Result from GPU CROWN backward: concretized lower and upper bounds.
pub use gemm::GpuCrownResult;
/// Seed state for GPU CROWN backward that starts from an existing suffix frontier.
pub use gemm::GpuCrownSeed;
/// One-pass wide CROWN capture: sound bounds, dual gradients, and affine frontier.
pub use gemm::GpuCrownTrajectoryResult;
/// Cached-plan extension for graph-DAG GPU-resident IBP backends (#4276, #4318).
pub use gemm::GpuDagIbpForwardExt;
/// Cached execution plan for graph-DAG GPU-resident IBP forwards (#4276, #4318).
pub use gemm::GpuDagIbpModelPlan;
/// Per-op descriptor for graph-DAG GPU-resident IBP forward (#4276, #4318).
pub use gemm::GpuDagIbpOp;
/// Complete graph-DAG resident IBP plan descriptor (#4276, #4318).
pub use gemm::GpuDagIbpPlanDesc;
/// GPU-resident IBP forward pass trait — keeps bounds on device across all layers (#4081).
pub use gemm::GpuIbpForward;
/// Cached-plan extension for GPU-resident IBP forward backends (#4268).
pub use gemm::GpuIbpForwardExt;
/// Per-layer descriptor for GPU IBP forward (Linear, ReLU, View).
pub use gemm::GpuIbpLayer;
/// Cached execution plan for repeated GPU-resident IBP forwards (#4268).
pub use gemm::GpuIbpModelPlan;
/// Result from GPU IBP forward: flattened output bounds and shape.
pub use gemm::GpuIbpResult;
/// Downloaded input-relative coeff frontier of a batched resnet CROWN backward (#clip-interm-resnet-batched).
pub use gemm::GpuResidentCoeffBatched;
/// One BaB subdomain's per-domain operands for a batched sound resnet CROWN backward (#batched-bab).
pub use gemm::GpuResnetBatchedDomainRef;
/// Backward-order resnet decomposition for the sound GPU-resident CROWN backward.
pub use gemm::GpuResnetSegment;
/// Naive triple-loop CPU GEMM for testing and fallback (no SIMD/tiling).
pub use gemm::NaiveCpuGemmEngine;
/// Maximum magnitude for CROWN A-matrix coefficients before proactive row degradation (#1932).
pub use gemm::CROWN_COEFF_MAX;
/// Conservative fallback bound for NaN/Inf sanitization across all bound propagation paths.
pub use gemm::FALLBACK_BOUND;
/// Sentinel index for network input in [`GpuDagIbpOp`] input fields.
pub use gemm::NETWORK_INPUT_IDX;
/// Output type wrapping a layer's computed tensors.
pub use layer_output::LayerOutput;
/// Enum of all supported neural network layer types.
pub use layer_type::LayerType;
/// NaN-propagating max for fold-based upper bound computation (#2577).
pub use nan_math::nan_propagating_max;
/// NaN-propagating max for f64 fold-based computation (#2616).
pub use nan_math::nan_propagating_max_f64;
/// NaN-propagating max(a, 0.0) for CROWN coefficient splitting (#2415, #2654).
pub use nan_math::nan_propagating_max_zero;
/// NaN-propagating min for fold-based lower bound computation (#2577).
pub use nan_math::nan_propagating_min;
/// NaN-propagating min for f64 fold-based computation (#2616).
pub use nan_math::nan_propagating_min_f64;
/// NaN-propagating min(a, 0.0) for CROWN coefficient splitting (#2415, #2654).
pub use nan_math::nan_propagating_min_zero;
/// Output-constraint kind: `Le` (a·y <= b) or `Ge` (a·y >= b) for halfspace properties (P7).
pub use output_constraint::ConstraintKind;
/// Output-constraint property: interval bounds, linear halfspace, or argmax-margin robustness (P7).
pub use output_constraint::OutputConstraint;
/// Floating-point precision tag (F32 default / F16 / Bf16) for mixed-precision widening (P8).
pub use precision::FloatPrecision;
/// SOUND outward-rounding precision primitives: widen f32 bounds to a deployed
/// precision grid (P8). F32 is a strict no-op; F16/Bf16 over-approximate.
/// `summation_error_bound` is the SOUND accumulation-error primitive (Higham
/// backward error) for reductions realized in a deployed precision.
pub use precision::{
    precision_round_error_bound, round_to_precision_outward, summation_error_bound, widen_bound,
    widen_bounds_for_precision, widen_bounds_for_precision_owned,
};
/// Verification proof artifacts: format, stats, and proof objects.
pub use proof::{ProofFormat, ProofStats, VerificationProof};
/// Internal reshape sentinel helpers for dimensions copied from a specific input axis.
pub use reshape_copy_axis::{reshape_copy_axis_from_sentinel, reshape_copy_axis_sentinel};
/// Soundness provenance tracking: which heuristics were used during verification.
pub use soundness::{HeuristicUsed, SoundnessProvenance, VerificationSoundnessMode};
/// Typed tag identifying which verification method produced a result.
pub use verification_result::MethodUsed;
/// Verification result with outcome and optional counterexample.
pub use verification_result::{UnknownReason, VerificationResult};
/// Checked dimension product with `InvalidSpec` error routing for propagation guards.
pub use verification_spec::checked_dim_product;
/// Checked shape product: returns `None` on overflow instead of wrapping (#2638).
pub use verification_spec::checked_shape_product;
/// Specification defining what property to verify about a network.
pub use verification_spec::VerificationSpec;
/// Violation types for reporting which output constraints were broken.
pub use violation::{ViolatedConstraint, ViolationType};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_bound_nan_guards;
#[cfg(test)]
mod tests_verification_spec_contract;
