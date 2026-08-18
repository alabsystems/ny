// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared backward CROWN core for constrained graph propagation.
//!
//! Extracted from the two near-duplicate backward passes in `constraints/mod.rs`:
//! - `propagate_crown_with_graph_constraints` (standard path)
//! - `propagate_crown_with_graph_constraints_storing_intermediates` (gradient path)
//!
//! Both share ~90% of their match arms. This module unifies them behind a single
//! traversal function parameterized by `BackwardMode`, eliminating drift risk.
//!
//! Part of #1813 (wave 2 dedup).
//! Split into directory module by #4293.

mod dispatch;
mod finalize;
mod linear;
mod relu;
mod setup;

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::{
    GpuCrownBackward, GpuCrownResult, GpuCrownSeed, NyError, Result,
    DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::domain::GraphCrownContext;
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::network::CrownMergeAccumulator;
use crate::{GraphNetwork, LinearBounds, NETWORK_INPUT};

use super::super::super::super::BetaCrownVerifier;
use super::lookups::ConstraintLookups;
use super::patches::ConstrainedPatchesPolicy;

const MAX_GPU_BETA_SEED_ROWS: usize = 512;
const MAX_GPU_BETA_SEED_ELEMENTS: usize = 16 * 1024 * 1024;

/// Controls whether the backward pass stores intermediate A-matrices for gradient computation.
pub(in crate::beta_crown::engine::graph::propagation) enum BackwardMode {
    /// Standard backward CROWN: no intermediate storage, includes debug diagnostics.
    Standard,
    /// Stores A matrices at constrained ReLU nodes for analytical gradient computation.
    /// Requires constraint lookups to identify which ReLU nodes to capture.
    StoringIntermediates { lookups: Box<ConstraintLookups> },
}

/// Groups the immutable graph parameters for constrained backward CROWN.
pub(in crate::beta_crown::engine::graph::propagation) struct BackwardParams<'a> {
    pub(in crate::beta_crown::engine::graph::propagation) graph: &'a GraphNetwork,
    pub(in crate::beta_crown::engine::graph::propagation) constrained_input: &'a BoundedTensor,
    pub(in crate::beta_crown::engine::graph::propagation) exec_order: &'a [String],
    pub(in crate::beta_crown::engine::graph::propagation) context: &'a GraphCrownContext<'a>,
    pub(in crate::beta_crown::engine::graph::propagation) beta_state:
        Option<&'a crate::beta_crown::state::GraphBetaState>,
    pub(in crate::beta_crown::engine::graph::propagation) objective: Option<&'a [f32]>,
    /// Multi-row spec matrix for batched spec-guided CROWN (#4306).
    /// When set, seeds the backward pass with an (N, D) matrix instead of
    /// identity or a single-row objective. Takes precedence over `objective`.
    pub(in crate::beta_crown::engine::graph::propagation) spec_matrix:
        Option<&'a ndarray::Array2<f32>>,
    pub(in crate::beta_crown::engine::graph::propagation) seed_cache:
        Option<&'a CachedLinearBounds>,
    pub(in crate::beta_crown::engine::graph::propagation) capture_linear_bounds: bool,
    /// Per-node deadline for intra-kernel timeout enforcement (#3795).
    /// When set, the DispatchContext carries this deadline so expensive
    /// backward kernels (e.g., Conv2d) can bail early.
    pub(in crate::beta_crown::engine::graph::propagation) deadline: Option<std::time::Instant>,
    pub(in crate::beta_crown::engine::graph::propagation) patches_policy: ConstrainedPatchesPolicy,
}

/// Full result of the constrained backward CROWN pass including concretization.
pub(in crate::beta_crown::engine::graph::propagation) struct BackwardCrownResult {
    /// Concretized output bounds.
    pub(in crate::beta_crown::engine::graph::propagation) output_bounds: BoundedTensor,
    /// Intermediate storage (populated only in `StoringIntermediates` mode).
    pub(in crate::beta_crown::engine::graph::propagation) intermediate:
        Option<GraphAlphaCrownIntermediate>,
    /// Full cached lA coefficients captured during the backward pass.
    pub(in crate::beta_crown::engine::graph::propagation) captured_la: Option<CachedLinearBounds>,
}

struct ConstrainedBackwardState {
    node_crown_bounds: CrownMergeAccumulator,
    intermediate: Option<GraphAlphaCrownIntermediate>,
    captured_linear_bounds: Option<HashMap<String, LinearBounds>>,
    input_accumulated: bool,
}

struct ConstrainedBackwardSetup<'graph, 'mode> {
    output_node: &'graph str,
    output_shape: Vec<usize>,
    output_dim: usize,
    input_dim: usize,
    mode_lookups: Option<&'mode ConstraintLookups>,
    /// A GPU error or malformed suffix receipt makes the remainder of this
    /// constrained backward pass CPU-only. This prevents a partial fallback
    /// from probing the same untrusted backend again at a shorter suffix.
    gpu_suffix_runtime_refused: bool,
    state: ConstrainedBackwardState,
}

fn bounded_beta_chunk_ranges(rows: usize, capacity: usize) -> Option<Vec<std::ops::Range<usize>>> {
    if rows < 2 || !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&capacity) {
        return None;
    }
    let chunk_count = rows.div_ceil(capacity);
    if rows < chunk_count.checked_mul(2)? {
        return None;
    }
    let base = rows / chunk_count;
    let larger = rows % chunk_count;
    let mut start = 0usize;
    let mut ranges = Vec::with_capacity(chunk_count);
    for ordinal in 0..chunk_count {
        let width = base + usize::from(ordinal < larger);
        if !(2..=capacity).contains(&width) {
            return None;
        }
        let end = start.checked_add(width)?;
        ranges.push(start..end);
        start = end;
    }
    (start == rows).then_some(ranges)
}

/// Plan physical CUDA rows for a logical objective stream.
///
/// The backend requires every transaction to contain `2..=capacity` rows. A
/// one-row stream therefore duplicates its row, and an odd stream at capacity
/// two duplicates its final row. Publication truncates these staging-only rows.
fn bounded_beta_physical_plan(
    logical_rows: usize,
    capacity: usize,
) -> Option<(usize, Vec<std::ops::Range<usize>>)> {
    if logical_rows == 0 {
        return None;
    }
    let physical_rows = logical_rows.max(2);
    if let Some(ranges) = bounded_beta_chunk_ranges(physical_rows, capacity) {
        return Some((physical_rows, ranges));
    }
    let padded_rows = physical_rows.checked_add(1)?;
    bounded_beta_chunk_ranges(padded_rows, capacity).map(|ranges| (padded_rows, ranges))
}

fn bounded_beta_stream_requested_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

#[inline]
fn bounded_beta_stream_requested_for_call(raw_wide: Option<&str>, bounded_facade: bool) -> bool {
    bounded_facade || bounded_beta_stream_requested_from_value(raw_wide)
}

/// Select a wide backend without allowing finite-deadline work to initialize
/// the process-global CUDA factory.
///
/// The unbounded branch retains the historical lazy initialization path. A
/// bounded caller may observe only a backend that some earlier work has already
/// materialized; `None` falls through to the existing sound CPU propagation.
fn resolve_sound_wide_gpu_for_authority<'a>(
    deadline: Option<std::time::Instant>,
    preinitialized: impl FnOnce() -> Option<&'a dyn GpuCrownBackward>,
    initialize_legacy: impl FnOnce() -> Option<&'a dyn GpuCrownBackward>,
) -> Option<&'a dyn GpuCrownBackward> {
    if deadline.is_some() {
        preinitialized()
    } else {
        initialize_legacy()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_deadline_bounded_beta_stream(
    gpu: &dyn GpuCrownBackward,
    segments: &[ny_core::GpuResnetSegment],
    seed_rows: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
    frontier_abs: &[Vec<f32>],
    node_abs: &[Vec<f32>],
    deadline: std::time::Instant,
) -> Result<GpuCrownResult> {
    run_deadline_bounded_beta_stream_with_clock(
        gpu,
        segments,
        seed_rows,
        num_specs,
        output_dim,
        input_lower,
        input_upper,
        beta_signed,
        frontier_abs,
        node_abs,
        deadline,
        std::time::Instant::now,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_deadline_bounded_beta_stream_with_clock(
    gpu: &dyn GpuCrownBackward,
    segments: &[ny_core::GpuResnetSegment],
    seed_rows: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
    frontier_abs: &[Vec<f32>],
    node_abs: &[Vec<f32>],
    deadline: std::time::Instant,
    mut now: impl FnMut() -> std::time::Instant,
) -> Result<GpuCrownResult> {
    if now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "deadline-bounded beta stream expired before validation".into(),
        ));
    }
    if num_specs == 0 || output_dim == 0 {
        return Err(NyError::InvalidSpec(
            "deadline-bounded beta stream requires a nonzero row count and output width".into(),
        ));
    }
    let expected_coefficients = num_specs.checked_mul(output_dim).ok_or_else(|| {
        NyError::InvalidSpec("deadline-bounded beta stream coefficient shape overflow".into())
    })?;
    if seed_rows.len() != expected_coefficients {
        return Err(NyError::InvalidSpec(format!(
            "deadline-bounded beta stream coefficient shape mismatch: expected \
             {expected_coefficients}, got {}",
            seed_rows.len()
        )));
    }
    if seed_rows.iter().any(|value| !value.is_finite()) {
        return Err(NyError::NumericalInstability(
            "deadline-bounded beta stream received a non-finite seed coefficient".into(),
        ));
    }
    let capacity = gpu.deadline_bounded_resnet_sound_beta_max_rows();
    let (physical_rows, chunks) =
        bounded_beta_physical_plan(num_specs, capacity).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "deadline-bounded beta stream cannot plan {num_specs} logical rows \
                 at capacity {capacity}"
            ))
        })?;
    let physical_coefficients = physical_rows.checked_mul(output_dim).ok_or_else(|| {
        NyError::InvalidSpec(
            "deadline-bounded beta stream physical coefficient shape overflow".into(),
        )
    })?;
    if physical_coefficients > MAX_GPU_BETA_SEED_ELEMENTS {
        return Err(NyError::InvalidSpec(format!(
            "deadline-bounded beta stream requires {physical_coefficients} physical \
             coefficients, exceeding cap {MAX_GPU_BETA_SEED_ELEMENTS}"
        )));
    }
    let mut lower_bounds = Vec::with_capacity(physical_rows);
    let mut upper_bounds = Vec::with_capacity(physical_rows);
    for range in chunks {
        if now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "deadline-bounded beta stream expired between chunks".into(),
            ));
        }
        let coefficient_start = range.start.checked_mul(output_dim).ok_or_else(|| {
            NyError::InvalidSpec("deadline-bounded beta stream coefficient overflow".into())
        })?;
        let coefficient_end = range.end.checked_mul(output_dim).ok_or_else(|| {
            NyError::InvalidSpec("deadline-bounded beta stream coefficient overflow".into())
        })?;
        let chunk_rows = range.len();
        let coefficients = if physical_rows != num_specs {
            let mut duplicated = Vec::new();
            duplicated
                .try_reserve_exact(coefficient_end - coefficient_start)
                .map_err(|_| NyError::CpuMemoryExceeded {
                    required_bytes: (coefficient_end - coefficient_start)
                        .saturating_mul(size_of::<f32>()),
                    budget_bytes: MAX_GPU_BETA_SEED_ELEMENTS.saturating_mul(size_of::<f32>()),
                    site: "deadline-bounded beta duplicate staging",
                })?;
            for physical_row in range.clone() {
                let logical_row = physical_row.min(num_specs - 1);
                let start = logical_row * output_dim;
                duplicated.extend_from_slice(&seed_rows[start..start + output_dim]);
            }
            duplicated
        } else {
            seed_rows[coefficient_start..coefficient_end].to_vec()
        };
        if now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "deadline-bounded beta stream expired while staging a chunk".into(),
            ));
        }
        let seed = GpuCrownSeed {
            lower_a: coefficients.clone().into(),
            upper_a: coefficients.into(),
            lower_b: vec![0.0; chunk_rows].into(),
            upper_b: vec![0.0; chunk_rows].into(),
            num_specs: chunk_rows,
            current_dim: output_dim,
        };
        let result = gpu.crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
            segments,
            &seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
            deadline,
        )?;
        if now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "deadline-bounded beta stream received a late chunk".into(),
            ));
        }
        if result.lower_bounds.len() != chunk_rows
            || result.upper_bounds.len() != chunk_rows
            || result
                .lower_bounds
                .iter()
                .zip(&result.upper_bounds)
                .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(NyError::NumericalInstability(
                "deadline-bounded beta stream received malformed chunk result".into(),
            ));
        }
        lower_bounds.extend(result.lower_bounds);
        upper_bounds.extend(result.upper_bounds);
    }
    if now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "deadline-bounded beta stream completed outside its authority".into(),
        ));
    }
    if lower_bounds.len() != physical_rows || upper_bounds.len() != physical_rows {
        return Err(NyError::InternalError(
            "deadline-bounded beta stream lost rows before atomic publication".into(),
        ));
    }
    lower_bounds.truncate(num_specs);
    upper_bounds.truncate(num_specs);
    Ok(GpuCrownResult {
        lower_bounds,
        upper_bounds,
    })
}

enum BoundedBetaStreamDisposition {
    Accepted(GpuCrownResult),
    CpuFallback(NyError),
}

fn bounded_beta_stream_disposition(result: Result<GpuCrownResult>) -> BoundedBetaStreamDisposition {
    match result {
        Ok(result) => BoundedBetaStreamDisposition::Accepted(result),
        Err(error) => BoundedBetaStreamDisposition::CpuFallback(error),
    }
}

fn finite_beta_gpu_funnel_is_declined(deadline: Option<std::time::Instant>) -> Result<bool> {
    super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained finite beta GPU funnel refusal",
    )?;
    Ok(deadline.is_some())
}

fn resolve_pre_activation<'a>(
    first_input: &str,
    constrained_input: &'a BoundedTensor,
    bounds_cache: &'a HashMap<String, Arc<BoundedTensor>>,
) -> Result<&'a BoundedTensor> {
    if first_input == NETWORK_INPUT {
        Ok(constrained_input)
    } else {
        bounds_cache
            .get(first_input)
            .map(|a| a.as_ref())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for {} not found",
                    first_input
                ))
            })
    }
}

impl BetaCrownVerifier {
    /// Shared backward CROWN pass for constrained graph propagation.
    ///
    /// Traverses the graph in reverse topological order, propagating linear bounds
    /// backward through each layer, then concretizes and applies cut planes.
    /// The `mode` parameter controls whether intermediate A-matrices are stored
    /// at constrained ReLU nodes for gradient computation.
    ///
    /// Both `propagate_crown_with_graph_constraints` and
    /// `propagate_crown_with_graph_constraints_storing_intermediates` delegate to this.
    pub(in crate::beta_crown::engine::graph::propagation) fn backward_crown_constrained(
        &self,
        params: &BackwardParams<'_>,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        mode: BackwardMode,
    ) -> Result<BackwardCrownResult> {
        if params
            .deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            return Err(NyError::DeadlineExceeded(
                "constrained backward CROWN: deadline exceeded before dispatch".to_string(),
            ));
        }
        // #unsat-keystone step 4: GPU beta-capable resnet fast-path. This is the SHARED
        // per-domain backward funnel — the BaB root bound, child bounds, and multi-
        // objective spec-guided passes all reach here — so one injection covers them all,
        // batched over the spec_matrix rows. Default ON (opt out NY_RESNET_BETA_GPU=0),
        // sound (β≥0 is a valid Lagrangian dual + the GPU resnet backward is a sound
        // enclosure), CPU fallback on any miss → the 0-wrong moat holds. Standard mode
        // only (no intermediate/lA capture on this path).
        if matches!(mode, BackwardMode::Standard) {
            // Seed collection, resnet-segment extraction, beta/window staging,
            // and input copies have no cooperative host seam. Finite work
            // declines the optional funnel before any of that O(N) preparation
            // and continues on the audited CPU route.
            let gpu_output = if finite_beta_gpu_funnel_is_declined(params.deadline)? {
                None
            } else {
                self.try_gpu_beta_constrained_backward(params, bounds_cache_mut)
            };
            if let Some(output_bounds) = gpu_output {
                return Ok(BackwardCrownResult {
                    output_bounds,
                    intermediate: None,
                    captured_la: None,
                });
            }
        }
        if params
            .deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            return Err(NyError::DeadlineExceeded(
                "constrained backward CROWN: deadline exceeded before CPU fallback".to_string(),
            ));
        }
        let is_standard = matches!(mode, BackwardMode::Standard);
        let mut setup = self.initialize_constrained_backward(params, &mode, &*bounds_cache_mut)?;

        for node_name in params.exec_order.iter().rev() {
            // Per-node deadline check: a single constrained backward pass through a
            // deep residual conv net (e.g. TinyImageNet ResNet) can be expensive, and
            // the per-domain check in the BaB loop only fires between domains. Without
            // this, one slow domain's backward overruns the wall-clock budget badly
            // (observed 23 min on a 90s budget once binary/residual child propagation
            // was enabled). Bailing here yields a sound Timeout/unresolved domain.
            if params
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return Err(NyError::DeadlineExceeded(
                    "constrained backward CROWN: deadline exceeded".to_string(),
                ));
            }
            if let Some(result) = self.process_constrained_backward_node(
                params,
                is_standard,
                node_name,
                bounds_cache_mut,
                &mut setup,
            )? {
                return Ok(result);
            }
        }

        self.finalize_constrained_backward(params, is_standard, bounds_cache_mut, setup)
    }

    /// GPU beta-capable resnet backward for the constrained (per-domain) bound
    /// (#unsat-keystone step 4). Seeds from the `spec_matrix` (batched over its N rows)
    /// or a single `objective`, decomposes the output suffix (alpha=None — default ReLU
    /// slopes from the CONSTRAINED `bounds_cache` already reflect the splits), folds the
    /// per-ReLU β-CROWN dual from `beta_state`, and runs the sound GPU resnet backward.
    /// Returns `Some(output_bounds)` (shape `[N]`) when applicable, else `None` → caller
    /// runs the proven CPU constrained backward. Gated + sound (β≥0 valid dual + sound
    /// GPU enclosure); default ON, opt out `NY_RESNET_BETA_GPU=0`.
    fn try_gpu_beta_constrained_backward(
        &self,
        params: &BackwardParams<'_>,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    ) -> Option<BoundedTensor> {
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        let deadline = params.deadline;
        if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            return None;
        }
        let local_gpu = params
            .context
            .engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())
            .filter(|g| crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(*g, deadline));
        // ReLU-only splits (the additive ±β term is the ReLU split_point=0 dual).
        if !params.context.history.genbab_constraints.is_empty() {
            return None;
        }
        let graph = params.graph;
        if !graph
            .nodes
            .values()
            .any(|n| matches!(n.layer, crate::layers::Layer::Conv2d(_)))
        {
            return None;
        }
        // Seed: spec_matrix (N×D, batched) preferred; else a single objective (1×D).
        // Skip the identity-seed (intermediate-node) case — left to CPU.
        let (n_specs, od) = if let Some(matrix) = params.spec_matrix {
            (matrix.nrows(), matrix.ncols())
        } else {
            let o = params.objective?;
            (1, o.len())
        };
        if od == 0
            || n_specs == 0
            || n_specs > MAX_GPU_BETA_SEED_ROWS
            || n_specs
                .checked_mul(od)
                .is_none_or(|elements| elements > MAX_GPU_BETA_SEED_ELEMENTS)
        {
            return None;
        }
        let seed_vec = if let Some(matrix) = params.spec_matrix {
            matrix.iter().copied().collect()
        } else {
            params.objective?.to_vec()
        };
        // The CLI's typed WGPU proof route threads its one qualified CROWN
        // device through the propagation context; a refused qualification is
        // already a reported CPU fallback and leaves `local_gpu` empty. Under
        // the exact CUDA-wide experiment gate, also resolve the separately
        // registered native CUDA engine for its narrower call-local beta
        // capability. It need not (and must not) claim that its ordinary wide
        // calls honor deadlines.
        let bounded_facade = params
            .context
            .engine
            .is_some_and(|engine| engine.forbids_unbounded_cpu_fallback());
        let bounded_requested = deadline.is_some()
            && bounded_beta_stream_requested_for_call(
                std::env::var("NY_CUDA_WIDE").ok().as_deref(),
                bounded_facade,
            );
        let bounded_gpu = if bounded_requested {
            let Some(gpu) = resolve_sound_wide_gpu_for_authority(
                deadline,
                crate::sound_gpu_gate::preinitialized_sound_gpu_crown_for_wide,
                crate::sound_gpu_gate::global_sound_gpu_crown_for_wide,
            ) else {
                eprintln!(
                    "[cuda-bounded-beta-stream] status=refused rows={n_specs} \
                     capacity=0 chunks=0 fallback=cpu reason=backend-unavailable"
                );
                return None;
            };
            let capacity = gpu.deadline_bounded_resnet_sound_beta_max_rows();
            if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&capacity) {
                eprintln!(
                    "[cuda-bounded-beta-stream] status=refused rows={n_specs} \
                     capacity={capacity} chunks=0 fallback=cpu reason=invalid-capability"
                );
                return None;
            }
            Some(gpu)
        } else {
            None
        };
        if local_gpu.is_none() && bounded_gpu.is_none() {
            return None;
        }
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        let (segments, relu_names, frontier_abs, node_abs) =
            crate::network::extract_gpu_resnet_segments_with_relu_names(
                graph,
                params.constrained_input,
                &graph.output_node,
                bounds_cache,
                bounds_cache,
                None,
            )?;
        let mut beta_signed: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for name in &relu_names {
            let nn = bounds_cache.get(name)?.lower().len();
            let mut bs = vec![0.0f32; nn];
            if let Some(beta) = params.beta_state {
                for entry in beta.entries_for_node(name) {
                    if entry.split_point().abs() < 1e-6 {
                        let idx = entry.neuron_idx();
                        if idx < nn {
                            bs[idx] = entry.signed_value();
                        }
                    }
                }
            }
            beta_signed.push(bs);
        }
        let in_lo: Vec<f32> = params.constrained_input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = params.constrained_input.upper().iter().copied().collect();
        // Complete Clip and the host-side segment/dual preparation above are
        // optional work. Recheck the deadline immediately before the resident
        // launch so they cannot consume the remaining budget and then admit a
        // fresh GPU proof. Returning `None` preserves the established CPU
        // fallback, whose first per-node check returns DeadlineExceeded.
        if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            return None;
        }
        let result = match (bounded_gpu, deadline) {
            (Some(gpu), Some(authority)) => {
                let capacity = gpu.deadline_bounded_resnet_sound_beta_max_rows();
                let (physical_rows, chunks) = bounded_beta_physical_plan(n_specs, capacity)
                    .map_or((0, 0), |(rows, ranges)| (rows, ranges.len()));
                eprintln!(
                    "[cuda-bounded-beta-stream] status=attempt rows={n_specs} \
                     physical_rows={physical_rows} \
                     capacity={capacity} chunks={chunks}"
                );
                match bounded_beta_stream_disposition(run_deadline_bounded_beta_stream(
                    gpu,
                    &segments,
                    &seed_vec,
                    n_specs,
                    od,
                    &in_lo,
                    &in_hi,
                    &beta_signed,
                    &frontier_abs,
                    &node_abs,
                    authority,
                )) {
                    BoundedBetaStreamDisposition::Accepted(result) => {
                        eprintln!(
                            "[cuda-bounded-beta-stream] status=accepted rows={n_specs} \
                             physical_rows={physical_rows} \
                             capacity={capacity} chunks={chunks}"
                        );
                        result
                    }
                    BoundedBetaStreamDisposition::CpuFallback(error) => {
                        eprintln!(
                            "[cuda-bounded-beta-stream] status=refused rows={n_specs} \
                             physical_rows={physical_rows} \
                             capacity={capacity} chunks={chunks} fallback=cpu reason={error}"
                        );
                        // The bounded stream is one atomic proof transaction.
                        // A refused/late/malformed chunk cannot fall through to
                        // another GPU surface; `None` restores the established
                        // constrained CPU backward from its untouched inputs.
                        return None;
                    }
                }
            }
            _ => {
                let gpu = local_gpu?;
                let seed = GpuCrownSeed {
                    lower_a: seed_vec.clone().into(),
                    upper_a: seed_vec.into(),
                    lower_b: vec![0.0f32; n_specs].into(),
                    upper_b: vec![0.0f32; n_specs].into(),
                    num_specs: n_specs,
                    current_dim: od,
                };
                let _gpu_deadline_scope =
                    crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
                gpu.crown_backward_gpu_resnet_sound_beta(
                    &segments,
                    &seed,
                    &in_lo,
                    &in_hi,
                    &beta_signed,
                    &frontier_abs,
                    &node_abs,
                )
                .ok()?
            }
        };
        if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            return None;
        }
        if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, n_specs) {
            return None;
        }
        if probe {
            eprintln!(
                "[beta-gpu-funnel] SUCCESS n_specs={n_specs} relus={} od={od}",
                relu_names.len()
            );
        }
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_specs]), result.lower_bounds)
                .ok()?;
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_specs]), result.upper_bounds)
                .ok()?;
        BoundedTensor::new(lower, upper).ok()
    }
}

#[cfg(test)]
mod bounded_beta_stream_tests {
    use std::cell::Cell;
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn finite_beta_gpu_funnel_declines_before_host_preparation() {
        let live = std::time::Instant::now() + std::time::Duration::from_secs(30);
        assert!(finite_beta_gpu_funnel_is_declined(Some(live))
            .expect("live finite work should decline the optional GPU funnel"));
        assert!(!finite_beta_gpu_funnel_is_declined(None)
            .expect("no-deadline work preserves the legacy GPU funnel"));

        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("one-second deadline subtraction");
        let error = finite_beta_gpu_funnel_is_declined(Some(expired))
            .expect_err("expired finite work must remain terminal");
        assert!(error.is_deadline_exceeded());
    }

    struct RecordingBoundedBetaGpu {
        calls: Mutex<Vec<usize>>,
        fail_call: Option<usize>,
        malformed_call: Option<usize>,
        capacity: usize,
    }

    impl RecordingBoundedBetaGpu {
        fn new(fail_call: Option<usize>) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail_call,
                malformed_call: None,
                capacity: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            }
        }

        fn with_capacity(capacity: usize) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail_call: None,
                malformed_call: None,
                capacity,
            }
        }

        fn malformed_on(call: usize) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail_call: None,
                malformed_call: Some(call),
                capacity: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            }
        }
    }

    impl GpuCrownBackward for RecordingBoundedBetaGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test-only ordinary route".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn deadline_bounded_resnet_sound_beta_max_rows(&self) -> usize {
            self.capacity
        }

        fn crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
            &self,
            _segments: &[ny_core::GpuResnetSegment],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            _beta_signed: &[Vec<f32>],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
            _deadline: std::time::Instant,
        ) -> Result<GpuCrownResult> {
            let mut calls = self.calls.lock().expect("recording lock");
            calls.push(seed.num_specs);
            if self.fail_call == Some(calls.len()) {
                return Err(NyError::InternalError(
                    "injected bounded beta chunk failure".into(),
                ));
            }
            if self.malformed_call == Some(calls.len()) {
                return Ok(GpuCrownResult {
                    lower_bounds: vec![f32::NAN; seed.num_specs],
                    upper_bounds: vec![0.0; seed.num_specs],
                });
            }
            let lower_bounds = seed
                .lower_a
                .chunks_exact(seed.current_dim)
                .map(|row| row[0])
                .collect::<Vec<_>>();
            let upper_bounds = lower_bounds.iter().map(|value| value + 1.0).collect();
            Ok(GpuCrownResult {
                lower_bounds,
                upper_bounds,
            })
        }
    }

    #[test]
    fn bounded_beta_stream_is_default_dark_and_requires_the_exact_one_spelling() {
        assert!(
            !bounded_beta_stream_requested_from_value(None),
            "an absent NY_CUDA_WIDE must leave the stream dark"
        );
        assert!(bounded_beta_stream_requested_from_value(Some("1")));
        for value in [Some(""), Some("0"), Some("01"), Some("true"), Some(" 1")] {
            assert!(!bounded_beta_stream_requested_from_value(value));
        }
        for value in [None, Some(""), Some("0"), Some("malformed")] {
            assert!(
                bounded_beta_stream_requested_for_call(value, true),
                "an admitted bounded facade must arm its matching CUDA route"
            );
            assert!(
                !bounded_beta_stream_requested_for_call(value, false),
                "legacy callers retain exact NY_CUDA_WIDE parsing"
            );
        }
        assert!(bounded_beta_stream_requested_for_call(Some("1"), false));
    }

    #[test]
    fn finite_bounded_beta_resolution_never_invokes_a_cold_blocking_factory() {
        let preinitialized_calls = Cell::new(0usize);
        let factory_calls = Cell::new(0usize);
        let selected = resolve_sound_wide_gpu_for_authority(
            Some(std::time::Instant::now()),
            || {
                preinitialized_calls.set(preinitialized_calls.get() + 1);
                None
            },
            || {
                factory_calls.set(factory_calls.get() + 1);
                // Models legacy CUDA initialization with no polling seam.
                // Finite authority must never enter this closure.
                std::thread::park();
                None
            },
        );
        assert!(selected.is_none());
        assert_eq!(preinitialized_calls.get(), 1);
        assert_eq!(
            factory_calls.get(),
            0,
            "finite bounded-beta resolution must never invoke a cold factory"
        );
    }

    #[test]
    fn chunk_plan_streams_tinyimagenet_rows_without_a_singleton() {
        let ranges = bounded_beta_chunk_ranges(199, 8).expect("199 rows are streamable");
        assert_eq!(ranges.len(), 25);
        assert!(ranges.iter().all(|range| (2..=8).contains(&range.len())));
        assert_eq!(ranges.first(), Some(&(0..8)));
        assert_eq!(ranges.last(), Some(&(192..199)));
        assert_eq!(ranges.iter().map(std::ops::Range::len).sum::<usize>(), 199);
        assert!(bounded_beta_chunk_ranges(3, 2).is_none());
        assert!(bounded_beta_chunk_ranges(1, 8).is_none());
        assert!(bounded_beta_chunk_ranges(8, 9).is_none());
    }

    #[test]
    fn bounded_beta_stream_honors_one_two_eight_and_nine_row_boundaries() {
        for (rows, expected_calls) in [(1, vec![2]), (2, vec![2]), (8, vec![8]), (9, vec![5, 4])] {
            let gpu = RecordingBoundedBetaGpu::new(None);
            let seed_rows = (0..rows).map(|row| row as f32).collect::<Vec<_>>();
            let result = run_deadline_bounded_beta_stream(
                &gpu,
                &[],
                &seed_rows,
                rows,
                1,
                &[],
                &[],
                &[],
                &[],
                &[],
                std::time::Instant::now() + std::time::Duration::from_secs(5),
            )
            .expect("each boundary row count must be streamable");
            assert_eq!(result.lower_bounds, seed_rows);
            assert_eq!(
                result.upper_bounds,
                seed_rows
                    .iter()
                    .map(|value| value + 1.0)
                    .collect::<Vec<_>>(),
                "a duplicated physical row must never be published"
            );
            assert_eq!(
                gpu.calls.lock().expect("recording lock").as_slice(),
                expected_calls
            );
        }
    }

    #[test]
    fn capacity_two_pads_odd_tail_without_publishing_duplicate_rows() {
        for rows in [1usize, 3, 99] {
            let gpu = RecordingBoundedBetaGpu::with_capacity(2);
            let seed_rows = (0..rows).map(|row| row as f32).collect::<Vec<_>>();
            let result = run_deadline_bounded_beta_stream(
                &gpu,
                &[],
                &seed_rows,
                rows,
                1,
                &[],
                &[],
                &[],
                &[],
                &[],
                std::time::Instant::now() + std::time::Duration::from_secs(5),
            )
            .expect("odd logical rows must be serviceable at capacity two");
            assert_eq!(result.lower_bounds, seed_rows);
            assert_eq!(result.lower_bounds.len(), rows);
            let calls = gpu.calls.lock().expect("recording lock");
            assert!(calls.iter().all(|&width| width == 2));
            assert_eq!(calls.iter().sum::<usize>(), rows.max(2).next_multiple_of(2));
        }
    }

    #[test]
    fn bounded_beta_stream_preserves_row_order_and_waits_for_every_chunk() {
        let gpu = RecordingBoundedBetaGpu::new(None);
        let seed_rows = (0..199).map(|row| row as f32).collect::<Vec<_>>();
        let result = run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &seed_rows,
            199,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        )
        .expect("complete stream");
        assert_eq!(result.lower_bounds, seed_rows);
        assert_eq!(
            result.upper_bounds,
            seed_rows
                .iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>()
        );
        let calls = gpu.calls.lock().expect("recording lock");
        assert_eq!(calls.len(), 25);
        assert!(calls.iter().all(|rows| (2..=8).contains(rows)));
        assert_eq!(calls.iter().sum::<usize>(), 199);
    }

    #[test]
    fn bounded_beta_stream_discards_a_partial_prefix_on_any_chunk_error() {
        let gpu = RecordingBoundedBetaGpu::new(Some(3));
        let seed_rows = (0..199).map(|row| row as f32).collect::<Vec<_>>();
        let disposition = bounded_beta_stream_disposition(run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &seed_rows,
            199,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        ));
        assert!(
            matches!(
                disposition,
                BoundedBetaStreamDisposition::CpuFallback(NyError::InternalError(_))
            ),
            "one refused chunk must select the exact CPU fallback disposition"
        );
        assert_eq!(
            gpu.calls.lock().expect("recording lock").as_slice(),
            &[8, 8, 8]
        );
    }

    #[test]
    fn every_bounded_cuda_refusal_selects_cpu_fallback() {
        for error in [
            NyError::UnsupportedOp("injected unsupported capability".into()),
            NyError::InvalidSpec("injected malformed request".into()),
            NyError::NumericalInstability("injected non-finite result".into()),
            NyError::DeadlineExceeded("injected late result".into()),
            NyError::InternalError("injected backend failure".into()),
        ] {
            assert!(matches!(
                bounded_beta_stream_disposition(Err(error)),
                BoundedBetaStreamDisposition::CpuFallback(_)
            ));
        }
    }

    #[test]
    fn bounded_beta_stream_refuses_bad_inputs_before_backend_dispatch() {
        let gpu = RecordingBoundedBetaGpu::new(None);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

        let non_finite = run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &[0.0, f32::NAN],
            2,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            deadline,
        )
        .expect_err("non-finite seed rows must refuse before dispatch");
        assert!(matches!(non_finite, NyError::NumericalInstability(_)));

        let overflow = run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &[],
            usize::MAX,
            2,
            &[],
            &[],
            &[],
            &[],
            &[],
            deadline,
        )
        .expect_err("an overflowing seed shape must refuse before dispatch");
        assert!(matches!(overflow, NyError::InvalidSpec(_)));

        let expired = run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &[0.0, 1.0],
            2,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_nanos(1))
                .expect("one nanosecond must be representable"),
        )
        .expect_err("an expired stream must refuse before dispatch");
        assert!(expired.is_deadline_exceeded());
        assert!(
            gpu.calls.lock().expect("recording lock").is_empty(),
            "invalid or expired requests must not invoke the backend"
        );
    }

    #[test]
    fn bounded_beta_stream_rejects_a_malformed_chunk_without_publication() {
        let gpu = RecordingBoundedBetaGpu::malformed_on(2);
        let seed_rows = (0..16).map(|row| row as f32).collect::<Vec<_>>();
        let error = run_deadline_bounded_beta_stream(
            &gpu,
            &[],
            &seed_rows,
            16,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        )
        .expect_err("a malformed second chunk must discard the completed prefix");
        assert!(matches!(error, NyError::NumericalInstability(_)));
        assert_eq!(
            gpu.calls.lock().expect("recording lock").as_slice(),
            &[8, 8]
        );
    }

    #[test]
    fn bounded_beta_stream_discards_completed_rows_when_a_chunk_expires() {
        let gpu = RecordingBoundedBetaGpu::new(None);
        let seed_rows = (0..16).map(|row| row as f32).collect::<Vec<_>>();
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(1);
        let before_deadline = deadline
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("one second must be representable");
        let polls = Cell::new(0usize);
        let attempt = run_deadline_bounded_beta_stream_with_clock(
            &gpu,
            &[],
            &seed_rows,
            16,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            deadline,
            || {
                let poll = polls.get() + 1;
                polls.set(poll);
                if poll >= 7 {
                    deadline
                } else {
                    before_deadline
                }
            },
        );
        let sentinel = GpuCrownResult {
            lower_bounds: vec![-123.0, -456.0],
            upper_bounds: vec![123.0, 456.0],
        };
        let mut caller_visible = sentinel.clone();
        if let Ok(result) = &attempt {
            caller_visible = result.clone();
        }
        let error =
            attempt.expect_err("a second late chunk must discard the completed first chunk");
        assert!(error.is_deadline_exceeded());
        assert_eq!(
            caller_visible, sentinel,
            "a late chunk must leave the caller-visible output wholly untouched"
        );
        assert_eq!(
            gpu.calls.lock().expect("recording lock").as_slice(),
            &[8, 8],
            "the fake clock expires immediately after the second backend return"
        );
        assert_eq!(polls.get(), 7);
    }
}
