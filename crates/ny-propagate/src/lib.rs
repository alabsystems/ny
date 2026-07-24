// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]
#![allow(clippy::too_many_arguments)]

//! Bound propagation algorithms for neural network verification.
//!
//! Implements multiple propagation strategies with increasing precision:
//! - IBP (Interval Bound Propagation): Fastest, loosest bounds
//! - CROWN: Linear relaxation, tighter bounds
//! - α-CROWN: Optimized CROWN with learnable parameters
//! - β-CROWN: Branch and bound for complete verification
//!
//! # Quick Start
//!
//! Use the [`prelude`] for convenient imports of common types:
//!
//! ```rust,no_run
//! # use ny_propagate::prelude::*;
//! # fn main() -> ny_core::Result<()> {
//! # let network: Network = unimplemented!();
//! # let input: BoundedTensor = unimplemented!();
//! # let threshold: f32 = 0.0;
//! let config = BetaCrownConfig::default();
//! let verifier = BetaCrownVerifier::new(config);
//! let result = verifier.verify(&network, &input, threshold)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Parallel Verification
//!
//! For sequence models, use [`parallel::ParallelVerifier`] to parallelize
//! verification across positions for near-linear speedup with cores.

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

// --- Internal modules (pub(crate) — use root re-exports instead) ---
pub(crate) mod batched_constraint_store;
/// Linear and interval bound types, α-CROWN configuration, and soundness arithmetic.
///
/// User-facing types (`LinearBounds`, `AlphaCrownConfig`, etc.) are also re-exported
/// at the crate root for convenience. Internal arithmetic helpers (`safe_add_for_bounds`,
/// `interval_mul_for_bounds`, etc.) are accessible here for verification proofs.
pub mod bounds;
pub(crate) mod clip_interm_domain;
/// NaN-safe comparison utilities for f32 sorting (#2981 Slice 3).
pub(crate) mod cmp_utils;
pub(crate) mod complete_clip;
/// Multi-network bound composition for system-level verification (#3517).
pub mod composition;
/// Domain clipping strategies and configuration for tightening input bounds.
/// Certified sparse-input double-double zonotope forward pass (`#dd-zonotope`).
pub mod dd_zonotope;
pub mod domain_clip;
pub(crate) mod faer_parallelism;
pub(crate) mod invprop;
pub(crate) mod l2_lever_gate;
pub(crate) mod relaxed_clip;
pub(crate) mod sdp_crown;

/// Shape compatibility utilities (broadcasting, etc.) — shared by `layers/`, `bounds/`, `network/`.
pub mod shape;
pub(crate) mod util;

// --- Public modules (stable API surface) ---
pub mod analysis;
pub(crate) mod batched_domain;
/// Public timing shim for the batched dense-spec CROWN backward (M2 throughput gate).
pub mod bench_batched;
/// β-CROWN verifier: branch-and-bound with CROWN bounds for complete verification.
pub mod beta_crown;
pub mod elimination;
pub mod equivalence;
/// Process-global optional IEEE RN-f32 GEMM accelerator (e.g. cuBLAS Sgemm) that
/// the backend `ComputeDevice` consults to offload its engine-routed `gemm_f32`
/// traffic (IBP forward, PGD/attack, BaB) — the f32 sibling of `sound_f64_gemm`.
pub mod fast_f32_gemm;
/// Input-Manifold Bound (IMB) root floor — seam-cut affine tail + prefix-BaB
/// certified floor (`NY_IMB=1`, default-OFF; STAGE 1 is log-only).
pub mod imb;
/// Layer implementations for IBP and CROWN bound propagation.
pub mod layers;
/// Margin-row twin-wall BaB lane (#twinwall): certified-outward sparse CROWN
/// margin rows over frozen gates for the cifar100/tinyimagenet resnet family.
pub mod margin_row;
/// Sound multi-neuron (k-ReLU) relaxation — 2-ReLU convex-hull group facets.
pub mod multineuron;
/// Network representations (sequential and graph) with bound propagation engines.
pub mod network;
/// Scoped publication of spec-referenced OUTPUT indices so full-width
/// OUTPUT-node CROWN backwards can seed only the k referenced identity rows
/// and scatter them over sound IBP bounds (#margin-subset-seed).
pub(crate) mod output_margin_seed;
/// Parallel verification across sequence positions for near-linear speedup.
pub mod parallel;
pub(crate) mod pgd_attack;
/// Dark print-only phase telemetry (`NY_PHASE_TELEMETRY=1`, default unset ⇒
/// byte-identical, no output): stderr markers at root-pipeline phase
/// boundaries so lever pricing can use per-phase durations instead of the
/// unpriceable single-row wall deltas (#phase-telemetry).
pub(crate) mod phase_telemetry;
/// Convenient re-exports of commonly used types.
pub mod prelude;
/// Probabilistic bounds via Monte Carlo sampling with CROWN certificates (#3921).
pub mod probabilistic;
mod random;
/// Thread-local scratch-buffer pool that recycles the per-domain / per-layer
/// CROWN backward faer f64 operands and products so the relational / iso
/// input-split rebound reduces temporary allocation churn. Retention is capped
/// per worker; bit-identical and gated by `NY_REBOUND_SCRATCH` (#rebound-scratch).
pub(crate) mod rebound_scratch;
/// Deterministic multi-seed restart knob for the bound-optimization RNG (task
/// #36). `set_rng_restart_offset(i)` makes the next `crate::random::rng()` seed
/// `NY_RNG_SEED_base + i`; the returned guard restores offset 0 on drop.
pub use random::{
    set_restart_offset as set_rng_restart_offset, RestartOffsetGuard as RngRestartGuard,
};
/// Process-global optional sound f64 GEMM accelerator (e.g. CUDA cuBLAS) for the
/// CPU CROWN backward's `A·W` / `|A|·|W|` products. Sound (order-independent
/// `γ_n·S` bound) and works even under `sound_gpu_gate`.
pub mod sound_f64_gemm;
/// Process-global gate that forces verdict-deciding CROWN onto the proven-sound
/// CPU path (suppresses the unsound GPU f32 CROWN fast-path) for VNN-COMP.
pub mod sound_gpu_gate;
/// Soundness provenance and preflight utilities.
pub mod soundness;
/// Streaming verification for memory-efficient processing of large networks.
pub mod streaming;
/// Shared types: propagation config, verification checkpoints, layer progress tracking.
pub mod types;
/// Verifier trait: common interface for all verification strategies.
pub mod verifier;

/// Batched constraint buffer for β-CROWN domain splitting.
pub use batched_constraint_store::BatchedConstraintBuffer;
/// β-CROWN verifier API: configuration, results, branching heuristics, and solver settings.
pub use beta_crown::{
    reset_bab_frontier_export, take_bab_frontier_seeds, BabFrontierSeed, BabVerificationStatus,
    BatchedSpecBackwardResult, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier,
    BranchingHeuristic, ConvMode, DenseSpecReboundMode, DenseSpecStageTiming,
    DomainSpecCrownResult, GraphDomainBatchCallerLane, GraphDomainBatchMetricsSink,
    GraphDomainBatchRecord, GraphPrecomputedBounds, InputClipType, InputSplitBatchRecord,
    InputSplitMetricsSink, JointMarginCloser, KfsbReduceOp, PhaseBudgetConfig,
    BAB_FRONTIER_CORNER_BOXES,
};

/// Propagation configuration, verification checkpoints, and layer progress tracking.
pub use types::{
    compute_model_hash, BlockProgress, CrownIbpPerNodeTimeBudget, LayerProgress,
    MulBinaryRelaxationMode, PropagationConfig, PropagationMethod, VerificationCheckpoint,
};

/// Layer entry points.
pub use layers::{BoundPropagation, Layer};

/// Linear and interval bound types, α-CROWN configuration.
///
/// Internal arithmetic helpers (`safe_add_for_bounds`, `batched_interval_matvec`, etc.)
/// are accessible via [`bounds`] module for verification proofs.
pub use bounds::{
    AlphaCrownConfig, AlphaSpecEarlyExit, AlphaState, BatchedLinearBounds, GradientMethod,
    LinearBounds, MultiSpecKeep, Optimizer,
};

/// Network representations (sequential and graph) with bound propagation engines.
///
/// `relu_crown_relaxation` is pub for Kani proof access but not called from production runtime.
/// `relu_ibp` has been moved to `layers::activations::relu::ibp`.
pub use network::{
    point_vjp_forward_masks, point_vjp_resnet_forward_masks, BlockSpec, BlockSpecEntry,
    BlockWiseCrownResult, GraphNetwork, GraphNode, Network, PointVjpBatchPlan, PointVjpResnetPlan,
    PointVjpWavePlan, SoftmaxComplexReport, VggMaxPoolRewriteMode, VggMaxPoolRewriteReport,
    ZonotopePropagationOptions, ZonotopeSoftmaxMode, NETWORK_INPUT, SOFTMAX_COMPLEX_SHIFT_GUARD,
};

/// ATTACK-only soft-sign surrogate sharpness (β) control for the point-gradient
/// `Layer::Sign` arm (traffic_signs BNNs). Thread-local; the falsify lanes ramp
/// β (≈2 smooth/exploratory → ≈20 sharp/decisive) to crack tight boxes a fixed
/// β gets stuck on. Soundness-neutral — β only scales the non-certified attack
/// direction and never feeds a verdict.
pub use network::{
    attack_sign_beta, set_attack_sign_beta, AttackSignBetaGuard, DEFAULT_ATTACK_SIGN_BETA,
};

/// Double-precision (f64) propagation types for soundnessbench/sat_relu.
/// Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).
pub use network::f64_propagate::{
    convert_network_to_f64, evaluate_network_f64, propagate_network_f64, F64PropagationMode,
    SequentialLayerF64,
};

/// Sound f64 interval graph forward for (near-)point cell inputs (#cctsdb Phase C).
pub use network::graph_ibp_f64_cell::Interval64;

/// Seed-axis count for the f64 first-order centered form (#f64-mvf): callers
/// gating the centered pass on box shape MUST use this so the gate matches
/// the walk's seeding rule.
pub use network::graph_ibp_f64_mvf::centered_seed_axes_f32;
pub use network::graph_ibp_f64_mvf::CenteredMono;

/// Kill-switch gate for the batched multi-box f64 forward
/// (#f64-batch-boxes, `NY_F64_BATCH_BOXES=0` disables).
pub use network::graph_ibp_f64_batch::batch_boxes_enabled;

/// Prepared f64 Linear weights for the batched multi-box f64 walks
/// (#f64-batch-boxes): build once per run via
/// `GraphNetwork::build_f64_weight_cache`, pass to the `_cached` entries.
pub use network::graph_ibp_f64_batch::F64WeightCache;

/// Dead-neuron analysis: classify each neuron as live / dead / constant (#4505).
pub use analysis::{
    analyze_neurons, analyze_neurons_with_epsilon, AnalysisResult, NeuronAnalysis, NeuronStatus,
};

/// Dead-neuron elimination: construct optimized network from analysis (#4505).
pub use elimination::{
    eliminate_and_verify, eliminate_dead_neurons, EliminationAction, EliminationCertificate,
    EliminationEntry, EliminationVerification,
};

/// Equivalence verification: prove ||f(x) - g(x)|| < eps via difference network (#4484).
pub use equivalence::{build_difference_network, verify_equivalence, EquivalenceResult};

/// Common verifier trait for all verification strategies.
pub use verifier::Verifier;

/// Parallel verification across sequence positions for near-linear speedup.
pub use parallel::{
    verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
    verify_parallel_with_method_and_engine, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier,
};

/// Streaming verification for memory-efficient processing of large networks.
pub use streaming::{
    estimate_memory_savings, CheckpointedBounds, StreamingConfig, StreamingVerifier,
};

// NOTE: ny-core types (Bound, NyError, Result, VerificationResult, VerificationSpec)
// and ny-tensor types (BoundedTensor) are NOT publicly re-exported here.
// Import them from their defining crates: ny_core and ny_tensor.
// See #2229 for rationale: re-exports create ambiguous import paths.
//
// Internal crate code that previously used `crate::BoundedTensor` etc. is supported
// via pub(crate) re-exports below.
#[allow(unused_imports)]
pub(crate) use ny_core::{NyError, Result};
#[allow(unused_imports)]
pub(crate) use ny_tensor::BoundedTensor;
pub(crate) use util::{contiguous_flat_slice, contiguous_flat_slice_mut};

// --- Internal re-exports for crate-internal use (tests, beta_crown, etc.) ---
// These items were removed from the public API (#2217) but are still needed
// by internal modules. `pub(crate)` keeps them accessible within the crate
// without leaking to external consumers.
// #2217: allow(unused_imports) needed because staged-index clippy in multi-worker
// mode doesn't see the crate-internal consumers of these pub(crate) re-exports.
// Waiver in .code_quality_waivers.toml tracks this.
#[allow(unused_imports)]
pub(crate) use batched_domain::{
    BatchedDomainOptions, BatchedDomains, BatchedDomainsBuilder, CachedLinearBounds, DomainList,
    DomainListConfig, DomainMetadata, DomainUpdate, EvaluatedGroupedChild, EvaluatedGroupedRoots,
    GroupedBatchCompletion, GroupedBoundSummary, GroupedChildEvaluationToken,
    GroupedDisjunctiveLayout, GroupedDomainId, GroupedParentOutcome, GroupedParentResolution,
    GroupedQueueStatus, GroupedRootEvaluationToken, GroupedSpecFingerprint, PickedDomains,
    PickedGroupedDomains, ProcessedDomains,
};
#[allow(unused_imports)]
pub(crate) use beta_crown::{
    AdaptiveOptConfig, ArenaConstraintStore, BabDomain, ConstraintHeader, ConstraintOrigin,
    ConstraintSense, CutPool, CutTerm, CuttingPlane, DomainConstraintStore,
    IntermediateLinearBounds, LRScheduler, LinearConstraintRef, NeuronConstraint, SplitHistory,
};
#[allow(unused_imports)]
pub(crate) use bounds::{AlphaCrownIntermediate, GraphAlphaState};
/// Output constraints for INVPROP backward propagation.
pub use invprop::OutputConstraints;
#[allow(unused_imports)]
pub(crate) use invprop::{InvpropConfig, InvpropState, LayerGammas};
#[allow(unused_imports)]
pub(crate) use layers::*;
#[cfg(test)]
pub(crate) use network::AttentionGraphBuilder;
pub use pgd_attack::gama;
pub use pgd_attack::{
    auto_alpha, project_to_bounds, project_to_bounds_in_place, AdamClippingParams, PgdAlphaMode,
    PgdAttacker, PgdConfig, PgdInitialization, PgdOptimizer, PgdResult, PgdStepState,
    GAMA_LAMBDA_DEFAULT,
};
pub use sound_gpu_gate::{is_sound_gpu_crown_required, set_sound_gpu_crown_required};
#[allow(unused_imports)]
pub use soundness::{
    count_sqrt_negative_domain_from_bounds, count_sqrt_negative_domain_graph,
    count_sqrt_negative_domain_network, soundness_provenance_for_graph,
    soundness_provenance_for_network,
};
#[allow(unused_imports)]
pub(crate) use types::{
    truncate_name, BoundsProvenance, CrownIbpBoundsResult, CrownIbpFallbackEvent,
    CrownIbpFallbackReason, GraphCrownIbpBoundsResult, LayerByLayerResult, NodeBoundsInfo,
};
/// Block-wise verification result types.
pub use types::{BlockBoundsInfo, BlockWiseResult};

/// Multi-model pipeline composition for system-level verification (#3920).
pub use composition::certificate::{BoundCertificate, BoundCertificationResult, BoundProvenance};
pub use composition::mixer::{compose_linear_mix, MixerSpec};
pub use composition::pipeline::{PipelineCertificate, PipelineStage, PipelineVerifier};
pub use composition::properties::{
    check_ducking_snr, check_priority_routing, check_spatial_ild, PropertyResult,
};

#[cfg(test)]
mod tests;
