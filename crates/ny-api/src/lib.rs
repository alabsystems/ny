// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Stable public API facade for ny integration.
//!
//! The always-on surface provides bounds, spec, soundness, and host-boundary
//! materialization helpers from `ny-core` and `ny-tensor`.
//!
//! With the `propagate` feature enabled, additional curated modules expose the
//! graph, model-build, layer, composition, verifier, and parallel-verification
//! surface
//! intended for external verifier consumers. Direct
//! `ny_propagate::*` imports outside the curated surface remain internal and
//! unstable.
//!
//! Prefer explicit imports from curated submodules such as `ny_api::graph`,
//! `ny_api::layers`, `ny_api::model`, `ny_api::verify`, and
//! `ny_api::parallel`. `ny_api::prelude::*` is the one curated wildcard
//! option for consumers that want a convenience import.
//!
//! The optional `gpu` feature (OFF by default) adds `ny_api::gpu`: a real
//! `ny-gpu` edge so a consumer can name and construct ny's typed GPU verdict
//! request. It is deliberately absent from the prelude — reaching a GPU
//! authority transition should be visible at the call site as a
//! `ny_api::gpu::` path. Reachability is not trust: ny qualifies every device
//! by live measurement and refuses, typed, when a rung does not pass.

/// Verification annotation types for attaching metadata to specs.
pub use crate::annotation::{AnnotatedSpec, AnnotationConstraint, SpecBuilder};
/// Stable device-to-host materialization boundary for verification inputs.
pub use crate::materialize::VerificationBoundsSource;
/// NaN-propagating min/max for sound bound computation (#3119).
pub use ny_core::{nan_propagating_max, nan_propagating_min};
/// Core error and result types from ny-core.
pub use ny_core::{
    Bound, MethodUsed, NyError, Result, UnknownReason, VerificationResult, VerificationSpec,
};
/// Soundness provenance types for tracking heuristic usage (#3119).
pub use ny_core::{HeuristicUsed, SoundnessProvenance, VerificationSoundnessMode};
/// Bounded tensor types from ny-tensor for specifying input regions.
/// `BoundedTensor64` is the double-precision carrier consumed by the f64
/// propagation helpers in [`verify`](crate::verify).
pub use ny_tensor::{BoundedScalar, BoundedTensor, BoundedTensor64, GenericBounds};

/// Verification annotation helpers: attach metadata (name/tags) to specs for external integrations.
pub mod annotation;
/// Curated inter-model composition surface (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod composition;
/// Device/lazy bounds materialization helpers for external integrations.
pub mod materialize;

/// Graph and network carrier types (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod graph;
/// Curated layer types for external consumer APIs (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod layers;
/// The `NY_*` lever surface: discover what exists, deliver the `SafeToSearch`
/// subset as typed data, and audit what the process environment is already
/// contributing (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod levers;
/// Owned graph-build contract and model specification types (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod model;
/// Parallel verification across sequence positions (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod parallel;
/// Verifier entry points and configuration (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod verify;

// --- Facade expansion: promote ny-propagate capabilities downstream consumers
// previously reached past the facade for, into the curated stable surface. ---

/// Dead-neuron analysis and elimination (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod analysis;
/// Branch-and-bound (β-CROWN) complete-verification surface (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod bab;
/// Proof-carrying certificate surface (exact-rational, Clean-checkable; requires `cert`).
#[cfg(feature = "cert")]
pub mod cert;
/// Complete MIP verification surface (ay; requires `complete`).
///
/// Carries both the curated `ny-mip` surface — encode/solve, LP and OBBT
/// tightening, SMT-LIB export, star search, and the Farkas infeasibility-evidence
/// producer — and the graph-level terminal
/// ([`complete::verify_graph_complete_mip`]) that the
/// [`ladder`] escalates into.
#[cfg(feature = "complete")]
pub mod complete;
pub mod conformance;
/// DSP / audio graph helpers (e.g. Kokoro STFT) used by TTS verification (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod dsp;
/// Operation soundness-coverage conformance: classify each `LayerType` by how
/// soundly bound-propagation handles it. Always available (operates on ny-core types).
/// Process-wide compute-engine bootstrap (the CPU GEMM floor embedded
/// consumers otherwise never install; requires `propagate`, which is where the
/// installers live — a bootstrap that silently installed nothing would be a
/// vacuous capability, so the module is absent instead).
#[cfg(feature = "propagate")]
pub mod engines;
/// Network equivalence verification: prove ||f(x) - g(x)|| < eps via a
/// difference network (requires `propagate` feature).
#[cfg(feature = "propagate")]
pub mod equivalence;
/// GPU participation — reachable, measurement-qualified, fail-closed (requires
/// the optional `gpu` feature, which is OFF by default).
///
/// Without `gpu`, `ny-gpu` is not in this crate's dependency graph and no
/// facade consumer can name a ny GPU type. With it, the typed verdict request
/// and the device that consumes it become reachable through `ny_api::gpu::*` —
/// and ny still decides participation by live measurement, refusing with a
/// typed report whenever a qualification rung does not pass. Read the module
/// docs before using it: what a caller gets differs sharply by platform AND by
/// which admission mode it names.
#[cfg(feature = "gpu")]
pub mod gpu;
/// SOUND deterministic global Lipschitz certification in exact rational
/// arithmetic — distinct from the optimistic estimate in [`probabilistic`]
/// (requires `cert`; NY ext 2).
#[cfg(feature = "cert")]
pub mod lipschitz;
/// Probabilistic bounds via Monte-Carlo sampling with CROWN certificates (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod probabilistic;
/// Streaming / memory-efficient verification of large networks (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod streaming;
/// Invariance / equivariance properties for point-cloud-style networks:
/// permutation invariance and finite-rotation invariance via difference
/// networks (requires `propagate`; NY ext 3).
#[cfg(feature = "propagate")]
pub mod symmetry;

// --- Wave 2 capability surface ---

/// Rich output-property constraints (halfspace / argmax-margin) and the encoder
/// that turns them into a verifiable augmented network (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod constraints;
/// Laddered model-level driver: IBP → α-CROWN → CROWN → β-CROWN → MIP escalation
/// in one call, recording method/soundness per run (requires `propagate`).
#[cfg(feature = "propagate")]
pub mod ladder;
/// Opt-in, policy-aware precision widening (P8): widen f32-proven output bounds to
/// remain SOUND at a deployed f16/bf16 precision before the verdict (requires
/// `propagate`). Default (all-F32) policy is a strict no-op.
#[cfg(feature = "propagate")]
pub mod precision;
/// Incremental / batch verification session: load a network once and verify many
/// specs against it, reusing the network and caching identical queries (P10;
/// requires `propagate`).
#[cfg(feature = "propagate")]
pub mod session;

/// Convenient re-exports of commonly used API types.
///
/// The always-on prelude covers annotation, core, tensor, and bounds-source
/// helpers. With `feature = "propagate"`, it also exposes the curated graph,
/// layer, model, verification, and parallel-verification facade as a single
/// wildcard-import surface.
pub mod prelude {
    pub use crate::annotation::{AnnotatedSpec, AnnotationConstraint, SpecBuilder};
    #[cfg(feature = "propagate")]
    pub use crate::graph::{GraphNetwork, GraphNode, SequentialNetwork, NETWORK_INPUT};
    #[cfg(feature = "propagate")]
    pub use crate::layers::*;
    pub use crate::materialize::VerificationBoundsSource;
    #[cfg(feature = "propagate")]
    pub use crate::model::{
        AttributeValue, CompoundNodePolicy, DataType, GraphModel, GraphModelBuilder,
        GraphNetworkOptions, LayerSpec, LayerType, MissingOutputPolicy, NetworkSpec, TensorSpec,
        WeightRef, WeightStore,
    };
    #[cfg(feature = "propagate")]
    pub use crate::parallel::{
        verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
        verify_parallel_with_method_and_engine, ParallelConfig, ParallelVerificationResult,
        ParallelVerifier,
    };
    #[cfg(feature = "propagate")]
    pub use crate::precision::{
        verify_with_precision_policy, verify_with_sound_precision, widen_bounds_for_policy,
        widen_bounds_for_policy_owned, widen_result_for_policy, MixedPrecisionPolicy,
    };
    #[cfg(feature = "propagate")]
    pub use crate::verify::{
        GemmEngine, MulBinaryRelaxationMode, NaiveCpuGemmEngine, PropagationConfig,
        PropagationMethod, Verifier,
    };
    #[cfg(feature = "propagate")]
    pub use ny_core::FloatPrecision;
    pub use ny_core::{
        nan_propagating_max, nan_propagating_min, Bound, HeuristicUsed, MethodUsed, NyError,
        Result, SoundnessProvenance, UnknownReason, VerificationResult, VerificationSoundnessMode,
        VerificationSpec,
    };
    pub use ny_tensor::{BoundedScalar, BoundedTensor, GenericBounds};
}
