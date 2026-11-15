// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network types and graph representations for bound propagation.
//!
//! This module contains the core network abstractions:
//! - [`Network`]: Sequential layer-based network
//! - [`GraphNetwork`]: DAG-based computation graph for attention patterns
//! - [`GraphNode`]: A single node in a computation graph
//!
//! Re-exports [`broadcast_shapes`] from [`crate::shape`] for backward compatibility.

pub(crate) mod alpha_crown;
pub(crate) mod alpha_crown_loop;
pub(crate) mod backward_dispatch;
mod core;
/// Shared CPU Dense-materialization policy used by sequential CROWN fallbacks.
pub mod crown_memory;
mod crown_quality;
pub(crate) use crown_quality::honest_provenance_enabled;
mod dispatch;
/// Compile-time dispatch-coverage matrix: per-`Layer`, per-dispatcher record of
/// SUPPORTED / DELIBERATELY-FALLS-BACK / REFUSED, enforced by exhaustive
/// no-catch-all matches plus runtime cross-checks against the real dispatchers.
pub mod dispatch_matrix;
pub mod dsp;
/// Double-precision (f64) network propagation for soundnessbench/sat_relu.
pub mod f64_propagate;
mod graph_alpha;
pub(crate) mod raw_root_cut;
pub(crate) use raw_root_cut::{evaluate_raw_root_cut, quote_raw_root_cut, RawRootCutLimits};
#[cfg(test)]
mod graph_builder;
// CROWN extension trait (extracted from graph.rs) - do not remove
mod graph_crown;
pub use graph_crown::attention_region::AttentionRegionReceipt;
/// Certified scalar f64 CROWN backward for tiny graph nets — the lsnc "f64
/// tail pass" (docs/LSNC_F64_TAIL_DESIGN.md; gate `NY_F64_TAIL=1`, default
/// OFF; band `NY_F64_TAIL_BAND`). Fail-closed: only certified `Verified`
/// outcomes may change caller state.
pub(crate) mod graph_crown_f64_tail;
mod graph_ibp;
/// Batched multi-box sound f64 interval forward (#f64-batch-boxes): stacks a
/// wave of clause boxes into fat interval GEMMs so the Rump kernel fires
/// (kill-switch `NY_F64_BATCH_BOXES=0`).
pub mod graph_ibp_f64_batch;
/// Sound f64 interval forward for (near-)point cell inputs (#cctsdb Phase C).
pub mod graph_ibp_f64_cell;
/// Fast SOUND interval matrix product: Rump midpoint-radius form on plain
/// f64 GEMMs (#f64-blas-gemm) — the large-shape kernel behind the f64 cell's
/// Linear/MatMul (kill-switch `NY_F64_BLAS=0`).
mod graph_ibp_f64_gemm;
/// Sound f64 first-order (mean-value/centered form) bounds via interval
/// forward-mode derivatives (#f64-mvf) — quadratic-convergence enclosures for
/// the nn4sys mscn multi-axis band-plateau clauses.
pub mod graph_ibp_f64_mvf;
mod graph_point_trace;
/// #vit-root-alpha (ENGINE_PROGRAM V2): the injected untrusted search backend
/// of the graph lane's root α search (`margin_alpha_search: joint_f32`),
/// registered by ny-cli; this crate sees it only as `&dyn SearchBackend`.
pub mod graph_root_search;
/// #graph-lane-search (ENGINE_PROGRAM M7): V2's plain-data `GraphSearchNet`
/// lowering of a `GraphNetwork` and its G1 self-check, shared by the graph
/// β-CROWN lane's batched domain search. Nothing certified reads its output.
pub(crate) use graph_alpha::search_lowering as graph_search_lowering;
pub(crate) mod ibp;
/// Sealed local original affine/ReLU chains; generic PointLinear remains quarantined.
pub mod original_affine_chain;
/// Priced local exact-real proof views of immutable original affine owners.
/// This does not relax diagnostic PointLinear whole-network eligibility.
pub mod original_affine_proof;
pub(crate) mod proof_eligibility;

#[cfg(test)]
mod dispatch_coverage_data;
#[cfg(test)]
mod dispatch_coverage_parser;
#[cfg(test)]
mod dispatch_coverage_tests;

// Re-export core types
pub(crate) use alpha_crown::NetworkAlphaCrownExt;
#[cfg(test)]
pub(crate) use core::extract_relu_gpu_layer_with_alpha;
pub use core::graph::crown_block_wise::LayerNormValidationStats;
pub use core::graph::crown_block_wise::{BlockSpec, BlockSpecEntry, BlockWiseCrownResult};
pub(crate) use core::graph::dispatch_plan::{CrownDispatchPlan, SharedNameIndex};
pub(crate) use core::graph::merge_accumulator::{CrownMergeAccumulator, SoleCarrierWitness};
pub use core::graph::{
    forward_linear_admission_record, forward_linear_measured_rate, set_forward_linear_f32_seam,
    ForwardLinearAdmissionRecord, ForwardLinearRateObservation,
};
pub(crate) use core::graph::{
    TrackedStringMap, TrackedStringMapAllocationFactV1, TRACKED_STRING_MAP_ALLOCATION_MODEL_V1,
};
pub use core::{
    compose_one_axis_dnf_observations, GraphNetwork, GraphNode, Network, OneAxisAffineCertificate,
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
pub(crate) use core::{crown_backward_step_patches, try_extract_single_gpu_layer, CrownStepResult};
pub(crate) use core::{gpu_relu_affine_cell, GpuReluAffineVariant};
pub(crate) use graph_alpha::atomic_cuda_margin_step::{
    root_alpha_cuda_margin_step_enabled, AtomicCudaMarginStepCommit, AtomicCudaMarginStepOutcome,
    AtomicCudaMarginStepRequest,
};
pub(crate) use graph_alpha::atomic_cuda_rows::{
    root_alpha_cuda_rows_enabled, AtomicCudaRowsCommit, AtomicCudaRowsOutcome,
    AtomicCudaRowsRefusal, AtomicCudaRowsRequest,
};
pub use graph_alpha::budget_policy::{
    collector_walk_admission_record, CollectorWalkAdmissionRecord,
};
pub(crate) use graph_alpha::merge_reference_bound_maps;
/// #root-joint-demand-rank: the CROWN-IBP collector's demand selector (which
/// nodes need tightened bounds), consumed by the armed root-joint interm-α
/// target ranking so both lanes agree on what "DEMANDED" means.
pub(crate) use graph_alpha::nodes_requiring_crown_tightening;
/// #fwd-root-screen (V4): C1's exact lookup-table helpers, reused (not forked)
/// by the certified zonotope core's piecewise-linear band.
pub(crate) use graph_alpha::{check_lookup_shapes, pwl_point_enclosure, pwl_range_f64};
/// #inert-ascent-exit telemetry: which rule retired the DAG alpha ascent
/// (`converged` / `patience` / `inert`) and at which iteration. Read by the
/// dark key's regression test; the module itself stays private, matching
/// every other item this file lifts out of `graph_alpha`.
// #inert-ascent-exit: both targets are `#[cfg(test)]` in `graph_alpha.rs`
// ("Test-only route evidence"), so the re-export must carry the same gate. It
// did not, and the plain `--lib` build had nothing to import (E0432). The
// branch that added this was validated under `cargo test`, where the cfg is on
// and it compiles — the mirror of CLAUDE.md's "cargo check does not build test
// targets": here a test-only item was re-exported unconditionally, so the TEST
// build was the green one and the LIB build was broken.
#[cfg(test)]
pub(crate) use graph_alpha::{clear_last_alpha_ascent_exit, last_alpha_ascent_exit};
// #attr-branch: the branching selector reads the root gap-attribution prior.
pub(crate) use graph_alpha::gap_attribution;
/// #metaroom-chain-wide: chain-permitting extraction + its opt-in gate for the BaB
/// batched β lane (pure conv-chain suffixes → `[Chain(layers)]`, dark by default).
pub(crate) use graph_alpha::resnet_decompose::bab_chain_wide_enabled;
/// Resnet suffix decomposition reused by the beta_crown BaB engine's per-domain GPU
/// beta backward (#unsat-keystone step 4). Returns segments + fold-order ReLU names.
pub(crate) use graph_alpha::resnet_decompose::extract_gpu_resnet_segments_with_relu_names;
pub(crate) use graph_alpha::resnet_decompose::extract_gpu_segments_with_relu_names_ext;
/// #clip-gather-probe L3: telemetry-only label naming WHICH `None` exit the last
/// segment extraction on this thread took. Never read by any decision.
pub(crate) use graph_alpha::resnet_decompose::extract_segments_last_refusal;
pub(crate) use graph_alpha::resnet_decompose::resnet_beta_gpu_batched_enabled;
/// Opt-out gate (default ON) for the beta-capable GPU resnet per-domain backward,
/// shared by the three beta_crown injection sites (#unsat-keystone step 4).
pub(crate) use graph_alpha::resnet_decompose::resnet_beta_gpu_enabled;
pub(crate) use graph_alpha::resnet_decompose::resnet_beta_gpu_wide_alpha_enabled;
pub(crate) use graph_alpha::resnet_decompose::resnet_beta_gpu_wide_beta_enabled;
pub(crate) use graph_alpha::resnet_decompose::resnet_refold_guard_enabled;
/// #extract-skeleton: static/dynamic split of the resnet segment extraction —
/// build the skeleton once per batched-BaB call, fold it per domain
/// (bit-identical to the legacy extraction; any refusal falls back to it).
/// Default-on gate with the `NY_EXTRACT_SKELETON=0` kill-switch.
pub(crate) use graph_alpha::resnet_skeleton::build_resnet_segment_skeleton;
pub(crate) use graph_alpha::resnet_skeleton::extract_skeleton_enabled;
pub(crate) use graph_alpha::resnet_skeleton::ResnetSegmentSkeleton;
/// #root-alpha-margin effective gate (typed preset default plus env override),
/// read by the multi-objective root to decide whether to hand the warmup a spec
/// objective for ranking its alpha iterates.
pub(crate) use graph_alpha::root_alpha_margin_enabled_with;
#[cfg(test)]
pub(crate) use graph_alpha::CganCompleteCollectionEntryCounter;
pub(crate) use graph_alpha::GraphAlphaCollectionOutcome;
pub(crate) use graph_alpha::PrecomputedAlphaReferenceBounds;
#[cfg(test)]
pub(crate) use graph_builder::AttentionGraphBuilder;
pub(crate) use graph_crown::spec_propagation::collect_intermediate_bounds;
pub(crate) use graph_crown::spec_propagation::SpecCrownRequest;
pub(crate) use graph_crown::GraphNetworkCrownExt;
/// ATTACK-only soft-sign β (sharpness) control for the point-VJP `Layer::Sign`
/// surrogate. Thread-local; ramped by the falsify lanes to crack tight BNN
/// boxes. Soundness-neutral — β never feeds a verdict.
pub use graph_crown::{
    attack_sign_beta, set_attack_sign_beta, smooth_sign_forward_enabled, AttackSignBetaGuard,
    AttackSteWindowsGuard, DEFAULT_ATTACK_SIGN_BETA,
};
pub(crate) use graph_crown::{backward_div_to_numerator, DivBackwardResult};
/// Batched point-VJP plan + CPU mask capture for the one-wide-GPU-pass exact
/// gradient attack (#batched-vjp). Attack-only; never verdict-feeding.
pub use graph_crown::{
    point_vjp_forward_masks, point_vjp_resnet_forward_masks, PointVjpBatchPlan, PointVjpResnetPlan,
    PointVjpWavePlan,
};
/// Exact-f64 twin of the chain / side-branch point-VJP plans for the float64
/// target-margin repair (#sbench-f64-attack). Attack-only; never verdict-feeding.
pub use graph_crown::{PointVjpF64Plan, PointVjpF64Tape};
pub use graph_ibp_f64_mvf::{
    centered_seed_axis_indices_f32, PointPhaseEventDiagnostics, ReluPhaseEventCandidate,
};
pub use graph_point_trace::{
    GraphPointArm, GraphPointStep, GraphPointTrace, GraphPointTraceAdmission,
};

// Completed diagnostic Linear arms are shared by graph and point-only fronts.
pub use crate::layers::linear::{LinearPointArm, LinearPointReceipt};

// Re-export forward-bound tightening — lives in core/ as a shared utility
pub(crate) use core::tighten_crown_with_forward_bounds;
// Re-export post-concretization tightening for fast.rs and streaming.
// Subsumes the separate has_degraded_bounds re-export.
pub(crate) use core::tighten_crown_output;
pub(crate) use core::tighten_crown_output_with_deadline;
// Re-export provenance-tracking variant for graph_crown.
pub(crate) use core::tighten_crown_output_with_provenance_and_deadline;
// Compatibility re-export: ny-build imports ny_propagate::network::broadcast_shapes.
// Canonical location is now crate::shape::broadcast_shapes.
pub use crate::shape::broadcast_shapes;
pub use graph_crown_f64_tail::f64_tail_unsupported_census;
// Test-only: decomposed normalization CROWN for proptest.
#[cfg(test)]
pub(crate) use crate::layers::normalization::decomposed::decomposed_norm_crown_backward;
#[cfg(test)]
pub(crate) use crate::layers::normalization::decomposed::decomposed_rms_norm_crown_backward;
#[cfg(test)]
pub(crate) use core::graph::crown_block_wise::BlockAlphaState;

/// Public seam for the disjunctive box screen (#nn4sys-dual): ONE certified
/// f64 CROWN tail attempt over `input_bounds` against `spec_matrix` rows
/// grouped by `clause_sizes` (same contract as `f64_tail_verify`: a clause
/// group is refuted when SOME row's certified f64 lower bound strictly
/// exceeds its threshold; the attempt succeeds only when EVERY group is
/// refuted). Anchors are collected internally with the graph's own sound IBP
/// intermediates. FAIL-CLOSED: any internal miss (anchor collection error,
/// unsupported op, shape surprise, expired deadline, non-verified outcome)
/// returns `false` — `true` is the only state-changing answer and it carries
/// the tail's full certified-outward envelope (docs/LSNC_F64_TAIL_DESIGN.md).
pub fn f64_tail_box_attempt(
    graph: &GraphNetwork,
    input_bounds: &ny_tensor::BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn ny_core::GemmEngine>,
    deadline: Option<std::time::Instant>,
) -> bool {
    // First-N debug telemetry (NY_DUAL_F64_TAIL_DEBUG=1): where attempts die.
    static DEBUG_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let debug = std::env::var("NY_DUAL_F64_TAIL_DEBUG").ok().as_deref() == Some("1")
        && DEBUG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12;
    let node_bounds = match collect_intermediate_bounds(graph, input_bounds, deadline, engine) {
        Ok(nb) => nb,
        Err(e) => {
            if debug {
                eprintln!("[dual-f64-tail] anchors FAILED: {e}");
            }
            return false;
        }
    };
    let outcome = graph_crown_f64_tail::f64_tail_verify(
        graph,
        input_bounds,
        spec_matrix,
        thresholds,
        clause_sizes,
        None,
        Some(&node_bounds),
        engine,
        deadline,
    );
    if debug {
        match &outcome {
            graph_crown_f64_tail::F64TailOutcome::Verified { .. } => {
                eprintln!("[dual-f64-tail] VERIFIED ({} rows)", thresholds.len());
            }
            graph_crown_f64_tail::F64TailOutcome::NotVerified { min_gap_f64 } => {
                eprintln!("[dual-f64-tail] not-verified min_gap={min_gap_f64:.3e}");
            }
            graph_crown_f64_tail::F64TailOutcome::Unsupported => {
                let mut bad: Vec<String> = Vec::new();
                if let Ok(needed) = graph.output_ancestors() {
                    for name in &needed {
                        if let Some(node) = graph.node(name) {
                            if !graph_crown_f64_tail::f64_tail_supports_layer_probe(node.layer()) {
                                bad.push(format!("{}:{}", name, node.layer().layer_type()));
                            }
                        }
                    }
                }
                bad.sort();
                bad.dedup();
                eprintln!(
                    "[dual-f64-tail] UNSUPPORTED (op class / shape / anchors); unsupported layers: {:?}",
                    &bad[..bad.len().min(8)]
                );
            }
        }
    }
    matches!(
        outcome,
        graph_crown_f64_tail::F64TailOutcome::Verified { .. }
    )
}
