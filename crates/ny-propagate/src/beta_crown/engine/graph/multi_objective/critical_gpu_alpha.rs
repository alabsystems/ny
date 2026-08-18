// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! One-step, state-paired alpha optimization for the critical root row.
//!
//! This module deliberately has no fallback CROWN surface.  The host replay is
//! used only as a gradient oracle; both scalar evaluations come from the sound,
//! direct-C GPU ResNet fold.  Its private deadline is cooperative: an in-flight
//! CUDA call cannot be preempted, but every setup, replay, optimizer, evaluation,
//! and publication boundary rejects work completed at or after the deadline.
//! The caller receives the initial and final evaluations only as typed
//! bound/state pairs and may select the final pair only when its certified lower
//! bound strictly improves the initial pair.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::AdaptiveOptConfig;
use crate::beta_crown::engine::graph::propagation::batched::{
    build_alpha_bridge,
    wide_alpha_true::{true_alpha_grads_for_row_gpu_until, TrueGradGpuReplayOps},
};
use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
use crate::bounds::GraphAlphaState;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

const CRITICAL_GPU_ALPHA_LR_MULTIPLIERS: [f32; 3] = [0.3, 1.0, 2.0];
const CRITICAL_GPU_ALPHA_MAX_LR: f32 = 0.25;
pub(super) const CRITICAL_GPU_ALPHA_PUBLICATION_RESERVE: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CriticalGpuAlphaStepRefusal {
    DeadlineExpired,
    InvalidLearningRateSchedule,
    NoSoundGpuRoute,
    InvalidInitialState,
    AlphaBridgeUnavailable,
    AlphaBridgeMismatch,
    OutputContractUnavailable,
    TopologyUnavailable,
    InitialDirectUnavailable,
    InitialDirectError,
    InvalidInitialDirectBound,
    HostReplayUnavailable,
    InvalidGradient,
    NoTrackedGradient,
    InvalidOptimizedState,
    FinalDirectUnavailable,
    FinalDirectError,
    InvalidFinalDirectBound,
}

impl CriticalGpuAlphaStepRefusal {
    pub(super) fn telemetry_reason(self) -> &'static str {
        match self {
            Self::DeadlineExpired => "deadline_expired",
            Self::InvalidLearningRateSchedule => "invalid_learning_rate_schedule",
            Self::NoSoundGpuRoute => "no_sound_gpu_route",
            Self::InvalidInitialState => "invalid_initial_state",
            Self::AlphaBridgeUnavailable => "alpha_bridge_unavailable",
            Self::AlphaBridgeMismatch => "alpha_bridge_mismatch",
            Self::OutputContractUnavailable => "output_contract_unavailable",
            Self::TopologyUnavailable => "topology_unavailable",
            Self::InitialDirectUnavailable => "initial_direct_unavailable",
            Self::InitialDirectError => "initial_direct_error",
            Self::InvalidInitialDirectBound => "invalid_initial_direct_bound",
            Self::HostReplayUnavailable => "host_replay_unavailable",
            Self::InvalidGradient => "invalid_gradient",
            Self::NoTrackedGradient => "no_tracked_gradient",
            Self::InvalidOptimizedState => "invalid_optimized_state",
            Self::FinalDirectUnavailable => "final_direct_unavailable",
            Self::FinalDirectError => "final_direct_error",
            Self::InvalidFinalDirectBound => "invalid_final_direct_bound",
        }
    }
}

fn classify_direct_c_error(
    error: &NyError,
    ordinary: CriticalGpuAlphaStepRefusal,
) -> CriticalGpuAlphaStepRefusal {
    if error.is_deadline_exceeded() {
        CriticalGpuAlphaStepRefusal::DeadlineExpired
    } else {
        ordinary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CriticalGpuAlphaStateIdentity {
    pub(super) parameter_count: usize,
    pub(super) fingerprint: u64,
}

#[derive(Debug)]
pub(super) struct CriticalGpuAlphaCertifiedPair {
    /// Direct-C certified enclosure evaluated with this pair's exact bridge.
    pub(super) bounds: BoundedTensor,
    /// Bit-exact round-trip of the bridge used to evaluate `bounds`.
    pub(super) state: GraphDomainAlphaState,
    pub(super) state_identity: CriticalGpuAlphaStateIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CriticalGpuAlphaCandidateTrace {
    pub(super) ordinal: usize,
    pub(super) adam_t: usize,
    pub(super) alpha_lr: f32,
    pub(super) lower: f32,
    pub(super) lift_from_initial: f32,
    pub(super) state_identity: CriticalGpuAlphaStateIdentity,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CriticalGpuAlphaSearchProvenance {
    pub(super) base_lr: f32,
    pub(super) candidates: Vec<CriticalGpuAlphaCandidateTrace>,
    pub(super) selected_ordinal: usize,
    pub(super) selected_lr: f32,
    pub(super) gradient_replays: usize,
}

impl CriticalGpuAlphaSearchProvenance {
    /// Revalidate that the exported pair is exactly the strict-best completed
    /// candidate described by the trace. Root publication calls this before
    /// either the bound or state becomes authoritative.
    pub(super) fn matches_best_candidate(
        &self,
        initial: &CriticalGpuAlphaCertifiedPair,
        pair: &CriticalGpuAlphaCertifiedPair,
    ) -> bool {
        let Some((initial_lower, _)) = scalar_finite_ordered(&initial.bounds) else {
            return false;
        };
        let candidate_lrs = CRITICAL_GPU_ALPHA_LR_MULTIPLIERS.map(|scale| self.base_lr * scale);
        if !valid_alpha_lr(self.base_lr)
            || candidate_lrs.iter().any(|&lr| !valid_alpha_lr(lr))
            || self.candidates.is_empty()
            || self.candidates.len() > CRITICAL_GPU_ALPHA_LR_MULTIPLIERS.len()
            || self.gradient_replays != 1
        {
            return false;
        }
        let mut best = None;
        for (expected_ordinal, candidate) in self.candidates.iter().enumerate() {
            if candidate.ordinal != expected_ordinal
                || candidate.adam_t != 1
                || !valid_alpha_lr(candidate.alpha_lr)
                || candidate.alpha_lr.to_bits() != candidate_lrs[expected_ordinal].to_bits()
                || !candidate.lower.is_finite()
                || !candidate.lift_from_initial.is_finite()
                || candidate.lift_from_initial.to_bits()
                    != (candidate.lower - initial_lower).to_bits()
                || candidate.state_identity.parameter_count == 0
            {
                return false;
            }
            if best
                .map(|current: &CriticalGpuAlphaCandidateTrace| candidate.lower > current.lower)
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
        let Some(best) = best else {
            return false;
        };
        let Some((pair_lower, _)) = scalar_finite_ordered(&pair.bounds) else {
            return false;
        };
        best.ordinal == self.selected_ordinal
            && best.alpha_lr.to_bits() == self.selected_lr.to_bits()
            && best.lower.to_bits() == pair_lower.to_bits()
            && best.state_identity == pair.state_identity
    }
}

#[derive(Debug)]
pub(super) struct CriticalGpuAlphaStepOutput {
    /// Certified direct-C baseline and the exact state that produced it.
    pub(super) initial: CriticalGpuAlphaCertifiedPair,
    /// Certified post-Adam direct-C candidate and its exact producing state.
    pub(super) final_candidate: CriticalGpuAlphaCertifiedPair,
    /// Present only for the independently gated LR bracket. `None` preserves
    /// the sealed one-step route and its existing telemetry contract.
    pub(super) search_provenance: Option<CriticalGpuAlphaSearchProvenance>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CriticalGpuAlphaSearchPolicy {
    pub(super) base_lr: f32,
    pub(super) candidate_lrs: [f32; 3],
    pub(super) work_deadline: Instant,
    pub(super) hard_deadline: Instant,
}

impl CriticalGpuAlphaSearchPolicy {
    pub(super) fn new(
        base_lr: f32,
        hard_deadline: Instant,
    ) -> Result<Self, CriticalGpuAlphaStepRefusal> {
        Self::new_at(base_lr, hard_deadline, Instant::now())
    }

    fn new_at(
        base_lr: f32,
        hard_deadline: Instant,
        now: Instant,
    ) -> Result<Self, CriticalGpuAlphaStepRefusal> {
        if !valid_alpha_lr(base_lr) {
            return Err(CriticalGpuAlphaStepRefusal::InvalidLearningRateSchedule);
        }
        let candidate_lrs = CRITICAL_GPU_ALPHA_LR_MULTIPLIERS.map(|scale| base_lr * scale);
        if candidate_lrs.iter().any(|&lr| !valid_alpha_lr(lr)) {
            return Err(CriticalGpuAlphaStepRefusal::InvalidLearningRateSchedule);
        }
        let work_deadline = hard_deadline
            .checked_sub(CRITICAL_GPU_ALPHA_PUBLICATION_RESERVE)
            .ok_or(CriticalGpuAlphaStepRefusal::DeadlineExpired)?;
        if now >= work_deadline {
            return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
        }
        Ok(Self {
            base_lr,
            candidate_lrs,
            work_deadline,
            hard_deadline,
        })
    }
}

#[inline]
fn valid_alpha_lr(lr: f32) -> bool {
    lr.is_finite() && lr > 0.0 && lr <= CRITICAL_GPU_ALPHA_MAX_LR
}

pub(super) fn critical_gpu_alpha_lr_bracket_enabled_from_value(enable: Option<&str>) -> bool {
    matches!(enable, Some("1"))
}

pub(super) fn critical_gpu_alpha_lr_bracket_enabled() -> bool {
    critical_gpu_alpha_lr_bracket_enabled_from_value(
        std::env::var("NY_ROOT_CRITICAL_GPU_ALPHA_LR_BRACKET")
            .ok()
            .as_deref(),
    )
}

#[inline]
pub(super) fn deadline_open(deadline: Instant) -> bool {
    Instant::now() < deadline
}

fn alpha_neuron_is_finite(neuron: &AlphaNeuronState) -> bool {
    let alpha = neuron.alpha();
    alpha.is_finite()
        && (0.0..=1.0).contains(&alpha)
        && neuron.grad().is_finite()
        && neuron.velocity().is_finite()
        && neuron.adam_m().is_finite()
        && neuron.adam_v().is_finite()
        && neuron.adam_v_max().is_finite()
}

#[inline]
pub(super) fn fingerprint_bytes(hash: &mut u64, bytes: &[u8]) {
    // FNV-1a is intentionally simple and stable across Rust/toolchain versions.
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

#[inline]
pub(super) fn fingerprint_u64(hash: &mut u64, value: u64) {
    fingerprint_bytes(hash, &value.to_le_bytes());
}

/// Stable identity of every lower/upper alpha value.
///
/// Optimizer moments are validated as finite but omitted from the fingerprint:
/// the bridge is a bound-evaluation state and intentionally does not carry
/// moments.  Requiring a bit-exact alpha round trip catches node-order,
/// channel/spatial expansion, and sparse-index mismatches.
pub(super) fn alpha_state_identity(
    state: &GraphDomainAlphaState,
) -> Option<CriticalGpuAlphaStateIdentity> {
    let parameter_count = state.len();
    if parameter_count == 0 {
        return None;
    }
    let upper_count: usize = state.upper_neurons().values().map(HashMap::len).sum();
    if upper_count != parameter_count {
        return None;
    }

    let mut node_names: Vec<_> = state.neurons().keys().map(String::as_str).collect();
    node_names.sort_unstable();
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    fingerprint_u64(&mut hash, parameter_count as u64);
    for node_name in node_names {
        let lower = state.neurons().get(node_name)?;
        let upper = state.upper_neurons().get(node_name)?;
        if lower.len() != upper.len() {
            return None;
        }
        fingerprint_u64(&mut hash, node_name.len() as u64);
        fingerprint_bytes(&mut hash, node_name.as_bytes());
        let mut neuron_indices: Vec<_> = lower.keys().copied().collect();
        neuron_indices.sort_unstable();
        for neuron_idx in neuron_indices {
            let lower_neuron = lower.get(&neuron_idx)?;
            let upper_neuron = upper.get(&neuron_idx)?;
            if !alpha_neuron_is_finite(lower_neuron) || !alpha_neuron_is_finite(upper_neuron) {
                return None;
            }
            fingerprint_u64(&mut hash, neuron_idx as u64);
            fingerprint_u64(&mut hash, u64::from(lower_neuron.alpha().to_bits()));
            fingerprint_u64(&mut hash, u64::from(upper_neuron.alpha().to_bits()));
        }
    }
    Some(CriticalGpuAlphaStateIdentity {
        parameter_count,
        fingerprint: hash,
    })
}

pub(super) fn build_checked_alpha_bridge(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    state: &GraphDomainAlphaState,
    invalid_state: CriticalGpuAlphaStepRefusal,
) -> Result<
    (
        GraphAlphaState,
        GraphDomainAlphaState,
        CriticalGpuAlphaStateIdentity,
    ),
    CriticalGpuAlphaStepRefusal,
> {
    let identity = alpha_state_identity(state).ok_or(invalid_state)?;
    let bridge = build_alpha_bridge(graph, node_bounds, Some(state))
        .ok_or(CriticalGpuAlphaStepRefusal::AlphaBridgeUnavailable)?;
    let recovered = GraphDomainAlphaState::from_root_alpha_state_borrowed(
        &bridge,
        graph,
        node_bounds,
        &GraphSplitHistory::new(),
        input,
    );
    let recovered_identity =
        alpha_state_identity(&recovered).ok_or(CriticalGpuAlphaStepRefusal::AlphaBridgeMismatch)?;
    if recovered_identity != identity {
        return Err(CriticalGpuAlphaStepRefusal::AlphaBridgeMismatch);
    }
    Ok((bridge, recovered, identity))
}

fn scalar_finite_ordered(bounds: &BoundedTensor) -> Option<(f32, f32)> {
    if bounds.shape() != [1] {
        return None;
    }
    let lower = bounds.lower()[[0]];
    let upper = bounds.upper()[[0]];
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_one_critical_gpu_alpha_step(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    engine: &dyn GemmEngine,
    spec_matrix: &ndarray::Array2<f32>,
    deadline: Instant,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &AdaptiveOptConfig,
) -> Result<CriticalGpuAlphaStepOutput, CriticalGpuAlphaStepRefusal> {
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let sound_gpu = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .filter(|gpu| gpu.provides_deadline_bounded_single_row_resnet_sound());
    if sound_gpu.is_none() || spec_matrix.nrows() != 1 || spec_matrix.ncols() == 0 {
        return Err(CriticalGpuAlphaStepRefusal::NoSoundGpuRoute);
    }
    if spec_matrix.iter().any(|value| !value.is_finite()) {
        return Err(CriticalGpuAlphaStepRefusal::NoSoundGpuRoute);
    }

    let input_flat = input.flatten();
    let in_lo: Vec<f32> = input_flat.lower().iter().copied().collect();
    let in_hi: Vec<f32> = input_flat.upper().iter().copied().collect();
    if in_lo.len() != in_hi.len()
        || in_lo.is_empty()
        || in_lo
            .iter()
            .zip(&in_hi)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Err(CriticalGpuAlphaStepRefusal::InvalidInitialState);
    }

    let (initial_bridge, initial_round_trip, initial_identity) = build_checked_alpha_bridge(
        graph,
        input,
        node_bounds,
        initial_state,
        CriticalGpuAlphaStepRefusal::InvalidInitialState,
    )?;
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }

    // Direct-C is the only evaluator admitted on this lane.  In particular,
    // never call the legacy independent-output surrogate.
    let initial_bounds = match SpecCrownRequest::new(graph, input, spec_matrix, Some(engine))
        .node_bounds(node_bounds)
        .alpha_state_opt(Some(&initial_bridge))
        .deadline_opt(Some(deadline))
        .run_alpha_sound_gpu_bounds_only()
    {
        Ok(Some(bounds)) => bounds,
        Ok(None) => return Err(CriticalGpuAlphaStepRefusal::InitialDirectUnavailable),
        Err(error) => {
            return Err(classify_direct_c_error(
                &error,
                CriticalGpuAlphaStepRefusal::InitialDirectError,
            ));
        }
    };
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let (initial_lower, _initial_upper) = scalar_finite_ordered(&initial_bounds)
        .ok_or(CriticalGpuAlphaStepRefusal::InvalidInitialDirectBound)?;

    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let exec_order = graph
        .exec_order()
        .map_err(|_| CriticalGpuAlphaStepRefusal::OutputContractUnavailable)?;
    let output_name = if graph.output_name().is_empty() {
        exec_order
            .last()
            .map(String::as_str)
            .ok_or(CriticalGpuAlphaStepRefusal::OutputContractUnavailable)?
    } else {
        graph.output_name()
    };
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let (segments, relu_names, frontier_abs, node_abs) =
        crate::network::extract_gpu_resnet_segments_with_relu_names(
            graph,
            input,
            output_name,
            node_bounds,
            node_bounds,
            Some(&initial_bridge),
        )
        .ok_or(CriticalGpuAlphaStepRefusal::TopologyUnavailable)?;
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let spec_row_view = spec_matrix.row(0);
    let spec_row = spec_row_view
        .as_slice()
        .ok_or(CriticalGpuAlphaStepRefusal::TopologyUnavailable)?;
    // #true-grad-gpu-replay: evaluate the replay's backward walk on the armed
    // sound GPU lane when available (direction × width); the tolerance
    // cross-check inside stays the live oracle, and every refusal falls closed
    // to the same CPU implementation under the remaining absolute deadline.
    let gpu_replay_ops =
        sound_gpu.and_then(|gpu| TrueGradGpuReplayOps::new(gpu, &frontier_abs, &node_abs));
    let gradients = true_alpha_grads_for_row_gpu_until(
        gpu_replay_ops.as_ref(),
        &segments,
        spec_row,
        &[],
        &in_lo,
        &in_hi,
        relu_names.len(),
        initial_lower,
        false,
        Some(deadline),
    )
    .ok_or(CriticalGpuAlphaStepRefusal::HostReplayUnavailable)?;
    if gradients.len() != relu_names.len() {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }

    let mut optimized_state = initial_state.clone();
    optimized_state.zero_grad();
    let mut seen_nodes = BTreeSet::new();
    let mut visited_parameters = 0usize;
    let mut nonzero_parameters = 0usize;
    for (relu_name, gradient_row) in relu_names.iter().zip(&gradients) {
        if !deadline_open(deadline) {
            return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
        }
        if !seen_nodes.insert(relu_name.as_str()) {
            return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
        }
        for (neuron_idx, &gradient) in gradient_row.iter().enumerate() {
            // Cooperative polling bounds a large host-side gradient projection;
            // it cannot interrupt a CUDA kernel already executing.
            if neuron_idx.is_multiple_of(4096) && !deadline_open(deadline) {
                return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
            }
            if !gradient.is_finite() {
                return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
            }
            if optimized_state.neuron(relu_name, neuron_idx).is_some() {
                visited_parameters += 1;
                if gradient != 0.0 {
                    optimized_state.accumulate_grad(relu_name, neuron_idx, gradient);
                    nonzero_parameters += 1;
                }
            }
        }
    }
    if visited_parameters != optimized_state.len() {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    if nonzero_parameters == 0 {
        return Err(CriticalGpuAlphaStepRefusal::NoTrackedGradient);
    }
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }

    let max_gradient = optimized_state.gradient_step_adam(adaptive_config, 1);
    if !max_gradient.is_finite() || max_gradient <= 0.0 {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    // A single shared lower/upper alpha preserves the fast batched child lane.
    optimized_state.sync_upper_from_lower();
    optimized_state.zero_grad();
    let (final_bridge, final_state, state_identity) = build_checked_alpha_bridge(
        graph,
        input,
        node_bounds,
        &optimized_state,
        CriticalGpuAlphaStepRefusal::InvalidOptimizedState,
    )?;
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }

    // The final direct-C evaluation and this exact round-tripped state form one
    // candidate pair. The caller must retain the initial pair when this step
    // does not strictly improve its certified direct-C lower bound.
    let final_bounds = match SpecCrownRequest::new(graph, input, spec_matrix, Some(engine))
        .node_bounds(node_bounds)
        .alpha_state_opt(Some(&final_bridge))
        .deadline_opt(Some(deadline))
        .run_alpha_sound_gpu_bounds_only()
    {
        Ok(Some(bounds)) => bounds,
        Ok(None) => return Err(CriticalGpuAlphaStepRefusal::FinalDirectUnavailable),
        Err(error) => {
            return Err(classify_direct_c_error(
                &error,
                CriticalGpuAlphaStepRefusal::FinalDirectError,
            ));
        }
    };
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    scalar_finite_ordered(&final_bounds)
        .ok_or(CriticalGpuAlphaStepRefusal::InvalidFinalDirectBound)?;

    Ok(CriticalGpuAlphaStepOutput {
        initial: CriticalGpuAlphaCertifiedPair {
            bounds: initial_bounds,
            state: initial_round_trip,
            state_identity: initial_identity,
        },
        final_candidate: CriticalGpuAlphaCertifiedPair {
            bounds: final_bounds,
            state: final_state,
            state_identity,
        },
        search_provenance: None,
    })
}

pub(super) fn project_critical_margin_gradient(
    initial_state: &GraphDomainAlphaState,
    relu_names: &[String],
    gradients: &[Vec<f32>],
    deadline: Instant,
) -> Result<GraphDomainAlphaState, CriticalGpuAlphaStepRefusal> {
    if gradients.len() != relu_names.len() {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    let mut projected_state = initial_state.clone();
    projected_state.zero_grad();
    let mut seen_nodes = BTreeSet::new();
    let mut visited_parameters = 0usize;
    let mut nonzero_parameters = 0usize;
    for (relu_name, gradient_row) in relu_names.iter().zip(gradients) {
        if !deadline_open(deadline) {
            return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
        }
        if !seen_nodes.insert(relu_name.as_str()) {
            return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
        }
        for (neuron_idx, &gradient) in gradient_row.iter().enumerate() {
            if neuron_idx.is_multiple_of(4096) && !deadline_open(deadline) {
                return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
            }
            if !gradient.is_finite() {
                return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
            }
            if projected_state.neuron(relu_name, neuron_idx).is_some() {
                visited_parameters += 1;
                if gradient != 0.0 {
                    projected_state.accumulate_grad(relu_name, neuron_idx, gradient);
                    nonzero_parameters += 1;
                }
            }
        }
    }
    if visited_parameters != projected_state.len() {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    if nonzero_parameters == 0 {
        return Err(CriticalGpuAlphaStepRefusal::NoTrackedGradient);
    }
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    Ok(projected_state)
}

pub(super) fn step_critical_alpha_candidate(
    projected_state: &GraphDomainAlphaState,
    adaptive_config: &AdaptiveOptConfig,
    alpha_lr: f32,
    deadline: Instant,
) -> Result<GraphDomainAlphaState, CriticalGpuAlphaStepRefusal> {
    if !valid_alpha_lr(alpha_lr) || !deadline_open(deadline) {
        return Err(if valid_alpha_lr(alpha_lr) {
            CriticalGpuAlphaStepRefusal::DeadlineExpired
        } else {
            CriticalGpuAlphaStepRefusal::InvalidLearningRateSchedule
        });
    }
    let mut candidate_state = projected_state.clone();
    let mut candidate_config = adaptive_config.clone();
    candidate_config.alpha_lr = alpha_lr;
    let max_gradient = candidate_state.gradient_step_adam(&candidate_config, 1);
    if !max_gradient.is_finite() || max_gradient <= 0.0 {
        return Err(CriticalGpuAlphaStepRefusal::InvalidGradient);
    }
    candidate_state.sync_upper_from_lower();
    candidate_state.zero_grad();
    if !deadline_open(deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    Ok(candidate_state)
}

fn retain_strict_best_candidate(
    best: &mut Option<(
        CriticalGpuAlphaCertifiedPair,
        CriticalGpuAlphaCandidateTrace,
    )>,
    pair: CriticalGpuAlphaCertifiedPair,
    trace: CriticalGpuAlphaCandidateTrace,
) -> Result<(), CriticalGpuAlphaStepRefusal> {
    let (candidate_lower, _) = scalar_finite_ordered(&pair.bounds)
        .ok_or(CriticalGpuAlphaStepRefusal::InvalidFinalDirectBound)?;
    if candidate_lower.to_bits() != trace.lower.to_bits()
        || pair.state_identity != trace.state_identity
    {
        return Err(CriticalGpuAlphaStepRefusal::InvalidOptimizedState);
    }
    let replace = if let Some((current_pair, _)) = best.as_ref() {
        let (current_lower, _) = scalar_finite_ordered(&current_pair.bounds)
            .ok_or(CriticalGpuAlphaStepRefusal::InvalidFinalDirectBound)?;
        candidate_lower > current_lower
    } else {
        true
    };
    if replace {
        *best = Some((pair, trace));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_critical_gpu_alpha_lr_bracket(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    engine: &dyn GemmEngine,
    spec_matrix: &ndarray::Array2<f32>,
    hard_deadline: Instant,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &AdaptiveOptConfig,
    base_lr: f32,
) -> Result<CriticalGpuAlphaStepOutput, CriticalGpuAlphaStepRefusal> {
    // Policy construction is deliberately first: an invalid top-level LR or
    // insufficient publication reserve must fail before extraction/replay.
    let policy = CriticalGpuAlphaSearchPolicy::new(base_lr, hard_deadline)?;
    let work_deadline = policy.work_deadline;
    let sound_gpu = engine
        .as_gpu_crown_backward()
        .filter(|gpu| gpu.provides_sound_gpu_crown())
        .filter(|gpu| gpu.provides_deadline_bounded_single_row_resnet_sound());
    if sound_gpu.is_none() || spec_matrix.nrows() != 1 || spec_matrix.ncols() == 0 {
        return Err(CriticalGpuAlphaStepRefusal::NoSoundGpuRoute);
    }
    if spec_matrix.iter().any(|value| !value.is_finite()) {
        return Err(CriticalGpuAlphaStepRefusal::NoSoundGpuRoute);
    }

    let input_flat = input.flatten();
    let in_lo: Vec<f32> = input_flat.lower().iter().copied().collect();
    let in_hi: Vec<f32> = input_flat.upper().iter().copied().collect();
    if in_lo.len() != in_hi.len()
        || in_lo.is_empty()
        || in_lo
            .iter()
            .zip(&in_hi)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Err(CriticalGpuAlphaStepRefusal::InvalidInitialState);
    }

    let (initial_bridge, initial_round_trip, initial_identity) = build_checked_alpha_bridge(
        graph,
        input,
        node_bounds,
        initial_state,
        CriticalGpuAlphaStepRefusal::InvalidInitialState,
    )?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let initial_bounds = match SpecCrownRequest::new(graph, input, spec_matrix, Some(engine))
        .node_bounds(node_bounds)
        .alpha_state_opt(Some(&initial_bridge))
        .deadline_opt(Some(work_deadline))
        .run_alpha_sound_gpu_bounds_only()
    {
        Ok(Some(bounds)) => bounds,
        Ok(None) => return Err(CriticalGpuAlphaStepRefusal::InitialDirectUnavailable),
        Err(error) => {
            return Err(classify_direct_c_error(
                &error,
                CriticalGpuAlphaStepRefusal::InitialDirectError,
            ));
        }
    };
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let (initial_lower, _initial_upper) = scalar_finite_ordered(&initial_bounds)
        .ok_or(CriticalGpuAlphaStepRefusal::InvalidInitialDirectBound)?;

    let exec_order = graph
        .exec_order()
        .map_err(|_| CriticalGpuAlphaStepRefusal::OutputContractUnavailable)?;
    let output_name = if graph.output_name().is_empty() {
        exec_order
            .last()
            .map(String::as_str)
            .ok_or(CriticalGpuAlphaStepRefusal::OutputContractUnavailable)?
    } else {
        graph.output_name()
    };
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let (segments, relu_names, frontier_abs, node_abs) =
        crate::network::extract_gpu_resnet_segments_with_relu_names(
            graph,
            input,
            output_name,
            node_bounds,
            node_bounds,
            Some(&initial_bridge),
        )
        .ok_or(CriticalGpuAlphaStepRefusal::TopologyUnavailable)?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let spec_row_view = spec_matrix.row(0);
    let spec_row = spec_row_view
        .as_slice()
        .ok_or(CriticalGpuAlphaStepRefusal::TopologyUnavailable)?;
    // #true-grad-gpu-replay: same seam as the sealed one-step route — the
    // replay's backward walk on the armed sound GPU lane, fail-closed to the
    // same CPU implementation under the remaining deadline on any refusal.
    let gpu_replay_ops =
        sound_gpu.and_then(|gpu| TrueGradGpuReplayOps::new(gpu, &frontier_abs, &node_abs));
    let gradients = true_alpha_grads_for_row_gpu_until(
        gpu_replay_ops.as_ref(),
        &segments,
        spec_row,
        &[],
        &in_lo,
        &in_hi,
        relu_names.len(),
        initial_lower,
        false,
        Some(work_deadline),
    )
    .ok_or(CriticalGpuAlphaStepRefusal::HostReplayUnavailable)?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let projected_state =
        project_critical_margin_gradient(initial_state, &relu_names, &gradients, work_deadline)?;

    let mut traces = Vec::with_capacity(policy.candidate_lrs.len());
    let mut best = None;
    for (ordinal, &alpha_lr) in policy.candidate_lrs.iter().enumerate() {
        if !deadline_open(work_deadline) {
            break;
        }
        let candidate_state = match step_critical_alpha_candidate(
            &projected_state,
            adaptive_config,
            alpha_lr,
            work_deadline,
        ) {
            Ok(state) => state,
            Err(CriticalGpuAlphaStepRefusal::DeadlineExpired) => break,
            Err(reason) => return Err(reason),
        };
        let (candidate_bridge, candidate_round_trip, state_identity) = build_checked_alpha_bridge(
            graph,
            input,
            node_bounds,
            &candidate_state,
            CriticalGpuAlphaStepRefusal::InvalidOptimizedState,
        )?;
        if !deadline_open(work_deadline) {
            break;
        }
        let candidate_bounds = match SpecCrownRequest::new(graph, input, spec_matrix, Some(engine))
            .node_bounds(node_bounds)
            .alpha_state_opt(Some(&candidate_bridge))
            .deadline_opt(Some(work_deadline))
            .run_alpha_sound_gpu_bounds_only()
        {
            Ok(Some(bounds)) => bounds,
            Ok(None) if !deadline_open(work_deadline) => break,
            Ok(None) => return Err(CriticalGpuAlphaStepRefusal::FinalDirectUnavailable),
            Err(error) if error.is_deadline_exceeded() => break,
            Err(_) => return Err(CriticalGpuAlphaStepRefusal::FinalDirectError),
        };
        // A CUDA call is cooperative rather than preemptible. Never admit a
        // candidate that returned after the work cutoff, and never return any
        // search result after the hard authority deadline.
        if !deadline_open(policy.hard_deadline) {
            return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
        }
        if !deadline_open(work_deadline) {
            break;
        }
        let (lower, _upper) = scalar_finite_ordered(&candidate_bounds)
            .ok_or(CriticalGpuAlphaStepRefusal::InvalidFinalDirectBound)?;
        let lift_from_initial = lower - initial_lower;
        if !lift_from_initial.is_finite() {
            return Err(CriticalGpuAlphaStepRefusal::InvalidFinalDirectBound);
        }
        let trace = CriticalGpuAlphaCandidateTrace {
            ordinal,
            adam_t: 1,
            alpha_lr,
            lower,
            lift_from_initial,
            state_identity,
        };
        retain_strict_best_candidate(
            &mut best,
            CriticalGpuAlphaCertifiedPair {
                bounds: candidate_bounds,
                state: candidate_round_trip,
                state_identity,
            },
            trace,
        )?;
        traces.push(trace);
    }
    if !deadline_open(policy.hard_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired);
    }
    let (final_candidate, selected_trace) =
        best.ok_or(CriticalGpuAlphaStepRefusal::DeadlineExpired)?;
    let search_provenance = CriticalGpuAlphaSearchProvenance {
        base_lr: policy.base_lr,
        candidates: traces,
        selected_ordinal: selected_trace.ordinal,
        selected_lr: selected_trace.alpha_lr,
        gradient_replays: 1,
    };
    let initial = CriticalGpuAlphaCertifiedPair {
        bounds: initial_bounds,
        state: initial_round_trip,
        state_identity: initial_identity,
    };
    if !search_provenance.matches_best_candidate(&initial, &final_candidate) {
        return Err(CriticalGpuAlphaStepRefusal::InvalidOptimizedState);
    }

    Ok(CriticalGpuAlphaStepOutput {
        initial,
        final_candidate,
        search_provenance: Some(search_provenance),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::state::AlphaNeuronState;
    use crate::layers::{Layer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, ArrayD, IxDyn};

    fn test_certified_pair(lower: f32, alpha: f32) -> CriticalGpuAlphaCertifiedPair {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu0".into(), 0, AlphaNeuronState::new(alpha));
        let state_identity = alpha_state_identity(&state).expect("test state identity");
        CriticalGpuAlphaCertifiedPair {
            bounds: BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[lower + 1.0]).into_dyn())
                .expect("ordered scalar bound"),
            state,
            state_identity,
        }
    }

    fn test_trace(
        ordinal: usize,
        alpha_lr: f32,
        pair: &CriticalGpuAlphaCertifiedPair,
    ) -> CriticalGpuAlphaCandidateTrace {
        CriticalGpuAlphaCandidateTrace {
            ordinal,
            adam_t: 1,
            alpha_lr,
            lower: pair.bounds.lower()[[0]],
            lift_from_initial: pair.bounds.lower()[[0]] + 2.0,
            state_identity: pair.state_identity,
        }
    }

    #[test]
    fn lr_bracket_gate_and_schedule_are_exact_and_fail_closed() {
        assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(None));
        assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(Some("0")));
        assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(Some(
            "true"
        )));
        assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(Some(
            " 1"
        )));
        assert!(!critical_gpu_alpha_lr_bracket_enabled_from_value(Some(
            "1 "
        )));
        assert!(critical_gpu_alpha_lr_bracket_enabled_from_value(Some("1")));

        let now = Instant::now();
        let hard_deadline = now + Duration::from_secs(2);
        let policy = CriticalGpuAlphaSearchPolicy::new_at(0.1, hard_deadline, now)
            .expect("CIFAR base LR is admissible");
        assert_eq!(policy.base_lr.to_bits(), 0.1_f32.to_bits());
        assert_eq!(
            policy.candidate_lrs.map(f32::to_bits),
            [0.1_f32 * 0.3, 0.1, 0.2].map(f32::to_bits)
        );
        assert_eq!(
            policy.work_deadline,
            hard_deadline
                .checked_sub(CRITICAL_GPU_ALPHA_PUBLICATION_RESERVE)
                .expect("future hard deadline has a publication reserve")
        );
        assert_eq!(policy.hard_deadline, hard_deadline);

        for invalid in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.1,
            0.13,
            0.25,
        ] {
            assert_eq!(
                CriticalGpuAlphaSearchPolicy::new_at(invalid, hard_deadline, now).unwrap_err(),
                CriticalGpuAlphaStepRefusal::InvalidLearningRateSchedule
            );
        }
        assert_eq!(
            CriticalGpuAlphaSearchPolicy::new_at(
                0.1,
                now + CRITICAL_GPU_ALPHA_PUBLICATION_RESERVE,
                now,
            )
            .unwrap_err(),
            CriticalGpuAlphaStepRefusal::DeadlineExpired
        );
    }

    #[test]
    fn direct_c_error_classifier_preserves_deadline_authority() {
        assert_eq!(
            classify_direct_c_error(
                &NyError::DeadlineExceeded("test deadline".into()),
                CriticalGpuAlphaStepRefusal::InitialDirectError,
            ),
            CriticalGpuAlphaStepRefusal::DeadlineExpired
        );
        assert_eq!(
            classify_direct_c_error(
                &NyError::InvalidSpec("test failure".into()),
                CriticalGpuAlphaStepRefusal::FinalDirectError,
            ),
            CriticalGpuAlphaStepRefusal::FinalDirectError
        );
    }

    #[test]
    fn one_replay_projection_feeds_independent_t1_candidates() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut initial = GraphDomainAlphaState::empty();
        initial.insert("relu0".into(), 0, AlphaNeuronState::new(0.5));
        let projected = project_critical_margin_gradient(
            &initial,
            &["relu0".to_string()],
            &[vec![1.0]],
            deadline,
        )
        .expect("one finite tracked gradient");
        let config = AdaptiveOptConfig::default();
        let lr0 = step_critical_alpha_candidate(&projected, &config, 0.03, deadline)
            .expect("lr0 candidate");
        let lr1 = step_critical_alpha_candidate(&projected, &config, 0.1, deadline)
            .expect("lr1 candidate");
        let lr2 = step_critical_alpha_candidate(&projected, &config, 0.2, deadline)
            .expect("lr2 candidate");

        let source = projected.neuron("relu0", 0).expect("source neuron");
        assert_eq!(source.alpha().to_bits(), 0.5_f32.to_bits());
        assert_eq!(source.adam_m().to_bits(), 0.0_f32.to_bits());
        assert_eq!(source.adam_v().to_bits(), 0.0_f32.to_bits());
        let candidates = [&lr0, &lr1, &lr2];
        for candidate in candidates {
            let lower = candidate.neuron("relu0", 0).expect("candidate neuron");
            assert!(lower.adam_m() > 0.0);
            assert!(lower.adam_v() > 0.0);
            assert_eq!(
                candidate.alpha("relu0", 0).to_bits(),
                candidate.alpha_upper("relu0", 0).to_bits(),
                "each candidate must retain shared lower/upper alpha"
            );
        }
        assert_eq!(
            lr0.neuron("relu0", 0).expect("lr0").adam_m().to_bits(),
            lr1.neuron("relu0", 0).expect("lr1").adam_m().to_bits()
        );
        assert_eq!(
            lr1.neuron("relu0", 0).expect("lr1").adam_m().to_bits(),
            lr2.neuron("relu0", 0).expect("lr2").adam_m().to_bits()
        );
        assert!(
            lr0.alpha("relu0", 0) < lr1.alpha("relu0", 0)
                && lr1.alpha("relu0", 0) < lr2.alpha("relu0", 0)
        );
    }

    #[test]
    fn gradient_projection_rejects_invariant_faults_atomically() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut initial = GraphDomainAlphaState::empty();
        initial.insert("relu0".into(), 0, AlphaNeuronState::new(0.5));
        let original = alpha_state_identity(&initial).expect("initial identity");

        let cases = [
            (
                vec!["relu0".to_string(), "relu0".to_string()],
                vec![vec![1.0], vec![1.0]],
                CriticalGpuAlphaStepRefusal::InvalidGradient,
            ),
            (
                vec!["other".to_string()],
                vec![vec![1.0]],
                CriticalGpuAlphaStepRefusal::InvalidGradient,
            ),
            (
                vec!["relu0".to_string()],
                vec![vec![f32::NAN]],
                CriticalGpuAlphaStepRefusal::InvalidGradient,
            ),
            (
                vec!["relu0".to_string()],
                vec![vec![0.0]],
                CriticalGpuAlphaStepRefusal::NoTrackedGradient,
            ),
        ];
        for (names, gradients, expected) in cases {
            assert_eq!(
                project_critical_margin_gradient(&initial, &names, &gradients, deadline)
                    .unwrap_err(),
                expected
            );
            assert_eq!(
                alpha_state_identity(&initial),
                Some(original),
                "projection refusal must not mutate the source state"
            );
        }
    }

    #[test]
    fn strict_best_candidate_retains_exact_pair_and_provenance() {
        for winner in 0..3 {
            let lowers = if winner == 0 {
                [-0.1, -0.2, -0.3]
            } else if winner == 1 {
                [-0.3, -0.1, -0.2]
            } else {
                [-0.3, -0.2, -0.1]
            };
            let lrs = [0.1_f32 * 0.3, 0.1, 0.2];
            let mut best = None;
            let mut traces = Vec::new();
            for ordinal in 0..3 {
                let pair = test_certified_pair(lowers[ordinal], 0.2 + ordinal as f32 * 0.2);
                let trace = test_trace(ordinal, lrs[ordinal], &pair);
                retain_strict_best_candidate(&mut best, pair, trace)
                    .expect("finite certified test candidate");
                traces.push(trace);
            }
            let (pair, selected) = best.expect("three candidates");
            let initial = test_certified_pair(-2.0, 0.1);
            let provenance = CriticalGpuAlphaSearchProvenance {
                base_lr: 0.1,
                candidates: traces,
                selected_ordinal: selected.ordinal,
                selected_lr: selected.alpha_lr,
                gradient_replays: 1,
            };
            assert_eq!(selected.ordinal, winner);
            assert!(provenance.matches_best_candidate(&initial, &pair));
            assert_eq!(
                alpha_state_identity(&pair.state),
                Some(selected.state_identity)
            );
        }

        let first = test_certified_pair(-0.1, 0.25);
        let first_trace = test_trace(0, 0.03, &first);
        let first_identity = first.state_identity;
        let tied = test_certified_pair(-0.1, 0.75);
        let tied_trace = test_trace(1, 0.1, &tied);
        let mut best = None;
        retain_strict_best_candidate(&mut best, first, first_trace)
            .expect("finite first candidate");
        retain_strict_best_candidate(&mut best, tied, tied_trace).expect("finite tied candidate");
        assert_eq!(
            best.expect("tie has a winner").0.state_identity,
            first_identity,
            "strict comparison retains the earlier certified pair on ties"
        );
    }

    #[test]
    fn alpha_state_identity_is_stable_and_covers_upper_state() {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu_b".into(), 3, AlphaNeuronState::new(0.25));
        state.insert("relu_a".into(), 7, AlphaNeuronState::new(0.75));
        let first = alpha_state_identity(&state).expect("finite nonempty alpha state");
        let second = alpha_state_identity(&state).expect("identity must be deterministic");
        assert_eq!(first, second);
        assert_eq!(first.parameter_count, 2);

        state
            .upper_neurons_mut()
            .get_mut("relu_a")
            .and_then(|node| node.get_mut(&7))
            .expect("upper alpha")
            .set_alpha(0.5);
        let changed = alpha_state_identity(&state).expect("changed state remains valid");
        assert_ne!(first.fingerprint, changed.fingerprint);
    }

    #[test]
    fn full_spatial_alpha_bridge_round_trip_is_bit_exact() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["pre".into()],
        ));
        let pre = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![-1.0_f32; 8]).expect("lower shape"),
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0_f32; 8]).expect("upper shape"),
        )
        .expect("spatial pre-activation bounds");
        let mut node_bounds = HashMap::new();
        node_bounds.insert("pre".to_string(), pre);
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        let mut state = GraphDomainAlphaState::from_graph_borrowed_bounds(
            &graph,
            &node_bounds,
            &GraphSplitHistory::new(),
            &input,
        );
        for neuron_idx in 0..8 {
            state
                .neuron_mut("relu0", neuron_idx)
                .expect("full-spatial neuron")
                .set_alpha((neuron_idx as f32 + 1.0) / 10.0);
            state
                .upper_neurons_mut()
                .get_mut("relu0")
                .and_then(|node| node.get_mut(&neuron_idx))
                .expect("full-spatial upper neuron")
                .set_alpha((8 - neuron_idx) as f32 / 10.0);
        }
        let expected = alpha_state_identity(&state).expect("state identity");
        let (_bridge, recovered, recovered_identity) = build_checked_alpha_bridge(
            &graph,
            &input,
            &node_bounds,
            &state,
            CriticalGpuAlphaStepRefusal::InvalidInitialState,
        )
        .expect("full-spatial bridge must round trip");
        assert_eq!(recovered_identity, expected);
        assert_eq!(alpha_state_identity(&recovered), Some(expected));
        for neuron_idx in 0..8 {
            assert_eq!(
                recovered.alpha("relu0", neuron_idx).to_bits(),
                state.alpha("relu0", neuron_idx).to_bits()
            );
            assert_eq!(
                recovered.alpha_upper("relu0", neuron_idx).to_bits(),
                state.alpha_upper("relu0", neuron_idx).to_bits()
            );
        }
    }
}
