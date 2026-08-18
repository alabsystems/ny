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
mod floating_point;
mod gemm;
/// #mn-head-facet HEAD coupling-facet f64-recovery fold registry (shared seam).
pub mod head_f64_fold;
/// #phase-window: invariant I1 — no fixed seconds. A phase window derives from
/// predicted cost on this host and is admitted only inside a FRACTION of what
/// remains; one that does not fit declines rather than half-running.
/// #instance-budget: the one deadline that means "the run is over", published
/// once so deep gates can apply invariant I1 against the REAL remaining budget.
pub mod instance_budget;
pub mod joint_alpha_grad;
mod layer_output;
mod layer_type;
mod nan_math;
mod output_constraint;
/// #phase-scheduler: the marginal-value loop that composes I1/I2/I3 — spend the
/// next block wherever `Δ(min_r slack_r)/second` is highest.
pub mod phase_scheduler;
pub mod phase_window;
/// #phase-yield: invariant I2 — a phase's expiry is a phase event, never an
/// instance event. See docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md.
pub mod phase_yield;
/// Floating-point precision tags for mixed-precision verification (P8).
pub mod precision;
mod proof;
mod reshape_copy_axis;
/// Certified Cut-CROWN resident cut-fold registry (shared writer/reader seam).
pub mod resident_cut_fold;
/// Arithmetic-only, call-local resident Cut-CROWN shadow transport.
pub mod resident_cut_shadow;
/// #span-profile: hierarchical self/total wall-time accounting across rayon
/// workers (`NY_SPAN_PROFILE=1`). Diagnostics only; inert when unset.
pub mod span_profile;

mod soundness;
mod verification_result;
mod verification_spec;
mod violation;
/// Historical #s1 differential-qualification ledger for GPU-produced bounds.
/// This ledger remains inert by construction: `corpus_is_green()` returns
/// `false` unconditionally and production qualification does not consult it.
/// Current WGPU authority instead requires an explicit typed device request and
/// a passing five-rung live report; only that exact device's public CROWN seam
/// opens. Its reviewed resident Conv route is admitted, while ordinary devices,
/// host Conv, segment-resident streams, and other non-CROWN operations stay
/// quarantined. The six blockers in the rejected corpus design remain
/// documented on [`wgpu_verdict::corpus_is_green`], and a tripwire pins that
/// historical gate shut.
pub mod wgpu_verdict;
/// #batched-bab wide-lane decline tally: WHY a candidate domain batch did not
/// take the domain-stacked GPU CROWN pass. Observability only — written with a
/// single relaxed atomic increment and read by nothing that makes a decision.
pub mod wide_lane_telemetry;

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
/// Floating-point environment probes and bit-exact representation conversion.
pub use floating_point::{
    f32_affine_eval_error, f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up,
    has_f64_interval_proof_environment, require_f64_interval_proof_environment,
};
/// Certify a backend-issued schedule for exact pre-descriptor static payload bytes.
pub use gemm::certify_gpu_bab_bound_static_schedule;
/// Sound activation-backward coefficient + error propagation (CROWN GPU-resident
/// backward, task #15 increment 3).
pub use gemm::crown_activation_error_step;
/// FTZ-safe additive underflow floor for sound f32 GPU error terms (Metal-safe).
pub use gemm::ftz_safe_underflow_floor;
/// Canonical retained-BaB v1 static payload identity shared by producer and core.
pub use gemm::gpu_bab_bound_static_payload_identity_v1;
/// Check whether a CROWN A-matrix coefficient is within safe magnitude bounds (#1932).
pub use gemm::is_crown_coeff_safe;
/// f64 variant of [`is_crown_coeff_safe`] for normalization CROWN backward paths (#3228).
pub use gemm::is_crown_coeff_safe_f64;
/// Fail-closed guard for routes that cannot charge a layer's `CertifiedWeightError`.
pub use gemm::refuse_uncharged_certified_weight_error;
/// Resnet-decomposition form of [`refuse_uncharged_certified_weight_error`].
pub use gemm::refuse_uncharged_certified_weight_error_segments;
/// Certified affine coefficients + their error, published by a sound GPU CROWN walk.
pub use gemm::CertifiedCoeffs;
/// Certified per-layer weight/bias error carried INTO a GPU CROWN walk (BN-fold terms).
pub use gemm::CertifiedWeightError;
/// Parameters for fused GPU conv_transpose_2d (GEMM + col2im) (#3813).
pub use gemm::ConvTranspose2dParams;
/// GEMM engine abstraction for pluggable matrix multiplication backends.
pub use gemm::GemmEngine;
/// Core-created context for one accepted raw backend phase open.
pub use gemm::GpuBabBoundAcceptedOpen;
/// Core-created context for one accepted raw backend wave.
pub use gemm::GpuBabBoundAcceptedWave;
/// Checked half-open range into a typed arena.
pub use gemm::GpuBabBoundArenaRange;
/// Raw backend close outcome.
pub use gemm::GpuBabBoundBackendCloseDisposition;
/// Exact raw close/release receipt.
pub use gemm::GpuBabBoundBackendCloseReceipt;
/// Untrusted exact per-domain completed outcome association.
pub use gemm::GpuBabBoundBackendDomainOutcome;
/// Raw bounded-domain outcome class (certified pruning remains closed).
pub use gemm::GpuBabBoundBackendDomainOutcomeKind;
/// Raw backend postaccept failure category.
pub use gemm::GpuBabBoundBackendFailureKind;
/// Explicit backend/registration-epoch/generation/nonce audit identity.
pub use gemm::GpuBabBoundBackendIssuerIdentity;
/// Raw backend phase-open outcome.
pub use gemm::GpuBabBoundBackendOpen;
/// Raw backend open failure category.
pub use gemm::GpuBabBoundBackendOpenFailureKind;
/// Allocation-free raw phase preparation outcome.
pub use gemm::GpuBabBoundBackendOpenPreparation;
/// Raw accountable retained-allocation open receipt.
pub use gemm::GpuBabBoundBackendOpenReceipt;
/// Raw allocation-free preaccept decision.
pub use gemm::GpuBabBoundBackendPrepareDisposition;
/// Stable backend registration with a core-owned O(1) burn ledger.
pub use gemm::GpuBabBoundBackendRegistration;
/// Untrusted backend result row.
pub use gemm::GpuBabBoundBackendRow;
/// Allocation-free raw backend static-schedule answer.
pub use gemm::GpuBabBoundBackendScheduleDisposition;
/// Untrusted exact backend static-schedule evidence.
pub use gemm::GpuBabBoundBackendScheduleEvidence;
/// Immutable schema/kernel bundle bound into a schedule-qualified registration.
pub use gemm::GpuBabBoundBackendScheduleIdentity;
/// Backend-facing raw phase session adapter.
pub use gemm::GpuBabBoundBackendSession;
/// Raw postaccept backend disposition.
pub use gemm::GpuBabBoundBackendWaveDisposition;
/// Raw all-terminal wave receipt.
pub use gemm::GpuBabBoundBackendWaveReceipt;
/// Actual owned dynamic operand arenas for one wave.
pub use gemm::GpuBabBoundDomainArena;
/// Exact parent/child/domain/view transcript for one wave domain.
pub use gemm::GpuBabBoundDomainTranscript;
/// Closed-role immutable f32 plan tensor.
pub use gemm::GpuBabBoundF32Tensor;
/// Semantic role for one immutable f32 plan tensor.
pub use gemm::GpuBabBoundF32TensorRole;
/// Immutable layer-neutral resident graph and objective data.
pub use gemm::GpuBabBoundGraphPlan;
/// Checked accountable device-memory categories.
pub use gemm::GpuBabBoundMemoryReceipt;
/// Explicit source-reviewed numerical TCB phase factory.
pub use gemm::GpuBabBoundNumericalTcb;
/// Exact typed per-domain views into the dynamic operand arenas.
pub use gemm::GpuBabBoundOperandView;
/// Immutable clone-cheap ownership for a fallibly populated GPU-BaB payload.
pub use gemm::GpuBabBoundOwnedSlice;
/// Canonical parent group and child coverage descriptor.
pub use gemm::GpuBabBoundParentGroup;
/// Core-validated consuming close outcome.
pub use gemm::GpuBabBoundPhaseCloseDisposition;
/// Clean reason a backend did not open a retained phase.
pub use gemm::GpuBabBoundPhaseDecline;
/// Immutable root-phase transcript and memory authority.
pub use gemm::GpuBabBoundPhaseDescriptor;
/// Core-owned, non-cloneable live phase authority.
pub use gemm::GpuBabBoundPhaseLease;
/// Typed core phase-open state.
pub use gemm::GpuBabBoundPhaseOpen;
/// Typed terminal open failure with its raw receipt.
pub use gemm::GpuBabBoundPhaseOpenFailure;
/// Backend-recommended retained-phase scheduling policy.
pub use gemm::GpuBabBoundPhasePolicy;
/// Exact backend/session/phase receipt transcript.
pub use gemm::GpuBabBoundPhaseTranscript;
/// Core-created raw preflight context.
pub use gemm::GpuBabBoundPreparedWave;
/// Typed nonfallback failure before core claims a backend issuer.
pub use gemm::GpuBabBoundProviderFailure;
/// Classification of a preclaim provider terminal.
pub use gemm::GpuBabBoundProviderFailureKind;
/// Non-cloneable core-sealed schedule evidence without a payload borrow.
pub use gemm::GpuBabBoundScheduleCertificate;
/// Typed pre-descriptor schedule-certification outcome.
pub use gemm::GpuBabBoundScheduleCertification;
/// Private-field core invocation for schedule certification.
pub use gemm::GpuBabBoundScheduleTcbInvocation;
/// Reason an existing phase can issue no new wave capability.
pub use gemm::GpuBabBoundSessionTerminal;
/// Exact source of one static graph/phase resident payload.
pub use gemm::GpuBabBoundStaticPayloadSource;
/// Borrowed, validated schedule-free static payload request.
pub use gemm::GpuBabBoundStaticScheduleRequest;
/// Core-validated static residency/padding/H2D equation for phase open.
pub use gemm::GpuBabBoundStaticTransferReceipt;
/// Same-parent contiguous subchunk carrying all objective rows.
pub use gemm::GpuBabBoundSubchunk;
/// Private-field core invocation presented only to the numerical TCB seam.
pub use gemm::GpuBabBoundTcbInvocation;
/// Core classification of an accepted terminal failure.
pub use gemm::GpuBabBoundTerminalFailureKind;
/// Exact all-terminal accepted-wave transcript.
pub use gemm::GpuBabBoundTerminalTranscript;
/// Exact typed H2D/D2H transfer equation.
pub use gemm::GpuBabBoundTransferReceipt;
/// Closed-role immutable u32 plan tensor.
pub use gemm::GpuBabBoundU32Tensor;
/// Semantic role for one immutable u32 plan tensor.
pub use gemm::GpuBabBoundU32TensorRole;
/// Core-validated association and terminal class for one domain.
pub use gemm::GpuBabBoundValidatedDomainOutcome;
/// Core-validated bounded-domain class (certified pruning remains closed).
pub use gemm::GpuBabBoundValidatedDomainOutcomeKind;
/// Immutable row issued only after consuming capability validation.
pub use gemm::GpuBabBoundValidatedRow;
/// Receipt issued only after consuming capability validation.
pub use gemm::GpuBabBoundValidatedWaveReceipt;
/// Non-cloneable exact-once accepted-wave capability.
pub use gemm::GpuBabBoundWaveCapability;
/// Clean preaccept wave decline reason.
pub use gemm::GpuBabBoundWaveDecline;
/// Mandatory completed/failure/deadline postaccept disposition.
pub use gemm::GpuBabBoundWaveDisposition;
/// Core-owned terminal failure with mandatory raw receipt.
pub use gemm::GpuBabBoundWaveFailure;
/// Split invalid/declined/accepted/session-terminal preflight result.
pub use gemm::GpuBabBoundWavePreparation;
/// Owned canonical request data for one candidate wave.
pub use gemm::GpuBabBoundWaveRequest;
/// One unary or fan-out operation in a GPU backward DAG sweep.
pub use gemm::GpuBackwardOp;
/// Dense reverse-topological slot identifier for a GPU backward DAG sweep.
pub use gemm::GpuBackwardSlot;
/// Hyperparameters for one fused GPU-resident β-only projected-Adam ascent.
pub use gemm::GpuBetaAdamConfig;
/// One domain's borrowed inputs for a fused GPU-resident β-only Adam ascent.
pub use gemm::GpuBetaAdamDomainRef;
/// Per-domain output from a fused GPU-resident β-only Adam ascent.
pub use gemm::GpuBetaAdamDomainResult;
/// ReLU/neuron/union-column mapping for one sparse resident β parameter.
pub use gemm::GpuBetaAdamMapping;
/// One caller-aligned sparse β parameter and its incoming Adam state.
pub use gemm::GpuBetaAdamParam;
/// Output from one fused GPU-resident β-only Adam call.
pub use gemm::GpuBetaAdamResult;
/// Returned optimizer snapshot for one sparse resident β parameter.
pub use gemm::GpuBetaAdamState;
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
/// One canonically associated identity-row injection in an intermediate sweep.
pub use gemm::GpuIntermediateInjection;
/// Backend-neutral reverse-DAG plan for a multi-depth intermediate sweep.
pub use gemm::GpuIntermediateSweepPlan;
/// Auditable device resource receipt for a completed intermediate sweep.
pub use gemm::GpuIntermediateSweepReceipt;
/// Borrowed operands and call-local authority for an intermediate sweep.
pub use gemm::GpuIntermediateSweepRequest;
/// Backend-recommended capacity policy for comprehensive intermediate sweeps.
pub use gemm::GpuIntermediateSweepResourcePolicy;
/// Atomic, all-target output from an intermediate sweep.
pub use gemm::GpuIntermediateSweepResult;
/// Bounds and exact request association for one intermediate target.
pub use gemm::GpuIntermediateTargetResult;
/// Downloaded input-relative coeff frontier of a batched resnet CROWN backward (#clip-interm-resnet-batched).
pub use gemm::GpuResidentCoeffBatched;
/// Zero-authority acknowledgement of an observation-only resident-Patches root plan.
pub use gemm::GpuResidentPatchesRootObservation;
/// Observation-only bounded multi-target implicit-Patches root plan.
pub use gemm::GpuResidentPatchesRootPlan;
/// One target's immutable metadata in a resident-Patches root plan.
pub use gemm::GpuResidentPatchesRootTargetPlan;
/// One BaB subdomain's per-domain operands for a batched sound resnet CROWN backward (#batched-bab).
pub use gemm::GpuResnetBatchedDomainRef;
/// Backward-order resnet decomposition for the sound GPU-resident CROWN backward.
pub use gemm::GpuResnetSegment;
/// Naive triple-loop CPU GEMM for testing and fallback (no SIMD/tiling).
pub use gemm::NaiveCpuGemmEngine;
/// An engine's declared crossover for the deadline-bounded sound f64 `A·W` seam.
pub use gemm::SoundF64GemmAdmission;
/// Completely validated atomic resident-wave result whose rows may be intersected.
pub use gemm::ValidatedGpuBabBoundWaveResult;
/// Completely validated atomic sweep result whose intervals may be published.
pub use gemm::ValidatedGpuIntermediateSweepResult;
/// Maximum magnitude for CROWN A-matrix coefficients before proactive row degradation (#1932).
pub use gemm::CROWN_COEFF_MAX;
/// Hard row cap for call-local deadline-bounded ResNet sound CROWN.
pub use gemm::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;
/// Conservative fallback bound for NaN/Inf sanitization across all bound propagation paths.
pub use gemm::FALLBACK_BOUND;
/// Host-side validation ceiling for values in each resident typed arena.
pub use gemm::GPU_BAB_BOUND_MAX_ARENA_VALUES;
/// Finite core ceiling for dispatches in one accepted resident wave.
pub use gemm::GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE;
/// Host-side validation ceiling for domains in one resident BaB bound wave.
pub use gemm::GPU_BAB_BOUND_MAX_DOMAINS;
/// Host-side validation ceiling for a resident phase's objective union.
pub use gemm::GPU_BAB_BOUND_MAX_OBJECTIVES;
/// Finite core ceiling for queue submissions in one accepted resident wave.
pub use gemm::GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE;
/// Conservative fixed host charge for one GPU-BaB owned-slice wrapper.
pub use gemm::GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES;
/// Host-side validation ceiling for intermediate-sweep operations.
pub use gemm::GPU_INTERMEDIATE_SWEEP_MAX_OPS;
/// Host-side validation ceiling for aggregate selected rows.
pub use gemm::GPU_INTERMEDIATE_SWEEP_MAX_ROWS;
/// Host-side validation ceiling for intermediate-sweep graph slots.
pub use gemm::GPU_INTERMEDIATE_SWEEP_MAX_SLOTS;
/// Host-side validation ceiling for injected intermediate targets.
pub use gemm::GPU_INTERMEDIATE_SWEEP_MAX_TARGETS;
/// Host-side validation ceiling for one target tensor's rank.
pub use gemm::GPU_INTERMEDIATE_SWEEP_MAX_TARGET_RANK;
/// Hard device-workspace ceiling carried by resident-Patches root plans.
pub use gemm::GPU_RESIDENT_PATCHES_ROOT_MAX_DEVICE_BYTES;
/// Hard aggregate row cap for resident-Patches root plans.
pub use gemm::GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS;
/// Hard target-count cap for resident-Patches root plans.
pub use gemm::GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS;
/// Sentinel index for network input in [`GpuDagIbpOp`] input fields.
pub use gemm::NETWORK_INPUT_IDX;
/// Historical engine-independent MAC floor for the sound f64 `A·W` seam.
pub use gemm::SOUND_F64_GEMM_DEFAULT_MIN_MACS;
/// Promotion-grade retained-domain v2 authority, history, transition, and
/// receipt types. V1 full-upload APIs remain available independently.
pub use gemm::{
    GpuBabBoundAcceptedResidentDomain, GpuBabBoundAcceptedResidentMaintenance,
    GpuBabBoundAcceptedResidentWave, GpuBabBoundBackendResidentMaintenanceDisposition,
    GpuBabBoundBackendResidentMaintenancePrepareDisposition,
    GpuBabBoundBackendResidentMaintenanceReceipt, GpuBabBoundBackendResidentPrepareDisposition,
    GpuBabBoundBackendResidentWaveDisposition, GpuBabBoundBackendResidentWaveReceipt,
    GpuBabBoundPreparedResidentGroup, GpuBabBoundPreparedResidentMaintenance,
    GpuBabBoundPreparedResidentWave, GpuBabBoundProposedResidentDomain,
    GpuBabBoundResidentConstruction, GpuBabBoundResidentDomainPolicy, GpuBabBoundResidentF32Family,
    GpuBabBoundResidentFamilyTransfer, GpuBabBoundResidentHostAudit,
    GpuBabBoundResidentMaintenanceCapability, GpuBabBoundResidentMaintenanceDisposition,
    GpuBabBoundResidentMaintenanceFailure, GpuBabBoundResidentMaintenanceMemoryReceipt,
    GpuBabBoundResidentMaintenancePreparation, GpuBabBoundResidentMaintenanceRequest,
    GpuBabBoundResidentMemoryReceipt, GpuBabBoundResidentParentGroup,
    GpuBabBoundResidentParentSource, GpuBabBoundResidentSlotRef, GpuBabBoundResidentSlotTranscript,
    GpuBabBoundResidentSourceAudit, GpuBabBoundResidentSourceClass,
    GpuBabBoundResidentSourcePresence, GpuBabBoundResidentTransferReceipt,
    GpuBabBoundResidentWaveCapability, GpuBabBoundResidentWaveDecline,
    GpuBabBoundResidentWaveDisposition, GpuBabBoundResidentWaveFailure,
    GpuBabBoundResidentWavePreparation, GpuBabBoundResidentWaveRequest,
    GpuBabBoundSplitHistoryArena, GpuBabBoundSplitHistoryLiteral, GpuBabBoundSplitHistoryPhase,
    GpuBabBoundSplitHistoryView, GpuBabBoundValidatedPhaseClose,
    GpuBabBoundValidatedResidentDomainOutcomeRef, GpuBabBoundValidatedResidentDomainOutcomes,
    GpuBabBoundValidatedResidentRowRef, GpuBabBoundValidatedResidentRows,
    GpuBabBoundValidatedResidentWaveReceipt, ValidatedGpuBabBoundResidentMaintenanceResult,
    ValidatedGpuBabBoundResidentWaveResult, GPU_BAB_BOUND_MAX_APPEND_SPLITS,
    GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES, GPU_BAB_BOUND_MAX_RESIDENT_DOMAIN_SLOTS,
    GPU_BAB_BOUND_MAX_RETAINED_V2_CORE_HOST_CHARGED_BYTES, GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS,
    GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS,
};
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
/// Arithmetic-only lower-cut transport for one call-local resident shadow.
pub use resident_cut_shadow::{
    ResidentCutShadowDisposition, ResidentCutShadowObservation, ResidentCutShadowOutcome,
    ResidentCutShadowPolicy, ResidentLowerCutCarrier, ResidentLowerCutChannel, ResidentLowerCutRow,
};
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
