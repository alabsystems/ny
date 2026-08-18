// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root-only output-conditioned dense-head refinement.
//!
//! The treatment is deliberately narrow: one unresolved disjunct, one crossing
//! coordinate in a final `Linear -> ReLU -> Linear` head, one non-negative gamma
//! step, and one ordinary same-row CROWN replay. Conditioned hidden bounds stay
//! call-local and can authorize only a complete root refutation; they are never
//! published as unconditional graph, root-cache, or BaB bounds.

use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::beta_crown::config::VerificationArtifactAuthority;
use crate::bounds::GraphAlphaState;
use crate::layers::Layer;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;
use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const OUTPUT_CONDITIONED_HEAD_ENV: &str = "NY_ROOT_OUTPUT_CONDITIONED_HEAD";
const OUTPUT_CONDITIONED_ROOT_MAX_RUNTIME: Duration = Duration::from_secs(2);

/// Initial proof surface: one unresolved disjunct and at most one sound-CUDA
/// small-batch worth of target coordinates.
pub(super) const OUTPUT_CONDITIONED_MAX_ROWS: usize = 1;
pub(super) const OUTPUT_CONDITIONED_MAX_COORDINATES: usize =
    ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;

fn parse_output_conditioned_head_gate(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn output_conditioned_head_enabled() -> bool {
    parse_output_conditioned_head_gate(std::env::var(OUTPUT_CONDITIONED_HEAD_ENV).ok().as_deref())
}

/// SHA-256 identity supplied by the same canonical encoders that own the
/// corresponding graph/property/domain object.
///
/// A conditioned result is never looked up by a short/non-cryptographic hash.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct OutputConditionedDigest([u8; 32]);

impl OutputConditionedDigest {
    pub(super) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Complete semantic scope of an output-conditioned hidden bound or root
/// refutation.
///
/// Every field participates in equality. In particular, a bound conditioned on
/// row `r`'s violation premise cannot be reused for another row, another input
/// box/BaB history, another reference-bound map, or another target node.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct OutputConditionedProofScope {
    pub(super) row_index: usize,
    pub(super) target_preactivation: String,
    /// Collision-free identity of this in-process graph and its semantic clones.
    ///
    /// The SHA below is useful in transcripts, but the graph-local token is the
    /// exact authority boundary: independently loaded same-shaped graphs never
    /// compare equal, while an unchanged configured clone keeps the token.
    pub(super) graph_scope: CutFoldScope,
    pub(super) graph_sha256: OutputConditionedDigest,
    pub(super) input_box_sha256: OutputConditionedDigest,
    pub(super) property_sha256: OutputConditionedDigest,
    pub(super) objective_row_sha256: OutputConditionedDigest,
    pub(super) reference_bounds_sha256: OutputConditionedDigest,
    pub(super) split_history_sha256: OutputConditionedDigest,
}

impl OutputConditionedProofScope {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        row_index: usize,
        target_preactivation: String,
        graph_scope: CutFoldScope,
        graph_sha256: OutputConditionedDigest,
        input_box_sha256: OutputConditionedDigest,
        property_sha256: OutputConditionedDigest,
        objective_row_sha256: OutputConditionedDigest,
        reference_bounds_sha256: OutputConditionedDigest,
        split_history_sha256: OutputConditionedDigest,
    ) -> Self {
        Self {
            row_index,
            target_preactivation,
            graph_scope,
            graph_sha256,
            input_box_sha256,
            property_sha256,
            objective_row_sha256,
            reference_bounds_sha256,
            split_history_sha256,
        }
    }
}

/// Publication receipt for the sound two-node evaluator.
///
/// Construction checks only the final strict root predicate and binding fields;
/// it is not itself a proof checker. The call-local evaluator constructs it only
/// after the combined seed has been propagated with outward rounding and an
/// ordinary same-row replay independently establishes the strict predicate.
/// Its root caller consumes it as a terminal boolean proof, never as a numeric
/// unconditional bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OutputConditionedRootRefutation {
    pub(super) scope: OutputConditionedProofScope,
    pub(super) conditional_lower_bits: u32,
    pub(super) threshold_bits: u32,
    pub(super) transcript_sha256: OutputConditionedDigest,
}

impl OutputConditionedRootRefutation {
    pub(super) fn new(
        scope: OutputConditionedProofScope,
        conditional_lower: f32,
        threshold: f32,
        transcript_sha256: OutputConditionedDigest,
    ) -> Option<Self> {
        (conditional_lower.is_finite() && threshold.is_finite() && conditional_lower > threshold)
            .then_some(Self {
                scope,
                conditional_lower_bits: conditional_lower.to_bits(),
                threshold_bits: threshold.to_bits(),
                transcript_sha256,
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OutputConditionedHeadRefusal {
    CertificateExportAuthority,
    GateDisabled,
    ConjunctiveProperty,
    TruncatedBackward,
    MalformedObjectives,
    NoUnresolvedRow,
    MultipleUnresolvedRows,
    NoEligibleDenseHead,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OutputConditionedHeadPlan {
    pub(super) row_index: usize,
    pub(super) target_preactivation: String,
    pub(super) max_coordinates: usize,
}

/// Select the deepest crossing ReLU preactivation on the output ancestry whose
/// producer is a `Linear` layer.
///
/// This intentionally uses graph topology and layer kinds, never exporter node
/// names such as `Gemm_56`.
fn select_dense_head_preactivation(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Option<String> {
    let order = graph.exec_order().ok()?;
    let output = if graph.output_name().is_empty() {
        order.last()?.as_str()
    } else {
        graph.output_name()
    };
    let output_ancestors = graph.all_ancestors().ok()?.get(output)?;

    for relu_name in order.iter().rev() {
        if !output_ancestors.iter().any(|name| name == relu_name) {
            continue;
        }
        let relu = graph.node(relu_name)?;
        if !matches!(relu.layer(), Layer::ReLU(_)) {
            continue;
        }
        let preactivation = relu.inputs().first()?;
        let producer = graph.node(preactivation)?;
        if !matches!(producer.layer(), Layer::Linear(_)) {
            continue;
        }
        let bounds = node_bounds.get(preactivation)?;
        let finite_ordered = bounds
            .lower()
            .iter()
            .zip(bounds.upper())
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper);
        let crossing = bounds
            .lower()
            .iter()
            .zip(bounds.upper())
            .any(|(&lower, &upper)| lower < 0.0 && upper > 0.0);
        if finite_ordered && crossing {
            return Some(preactivation.clone());
        }
    }
    None
}

/// Pure, fail-closed root admission for the two-node sound evaluator and its
/// premise-scoped, terminal-only publication contract.
pub(super) fn build_output_conditioned_head_plan(
    gate_enabled: bool,
    artifact_authority: VerificationArtifactAuthority,
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
    objectives: &[Vec<f32>],
    objective_bounds: &[(f32, f32)],
    thresholds: &[f32],
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Result<OutputConditionedHeadPlan, OutputConditionedHeadRefusal> {
    if artifact_authority != VerificationArtifactAuthority::VerdictOnly {
        return Err(OutputConditionedHeadRefusal::CertificateExportAuthority);
    }
    if !gate_enabled {
        return Err(OutputConditionedHeadRefusal::GateDisabled);
    }
    if conjunctive {
        return Err(OutputConditionedHeadRefusal::ConjunctiveProperty);
    }
    if crown_backward_layers.is_some() {
        return Err(OutputConditionedHeadRefusal::TruncatedBackward);
    }
    let output_dim = objectives.first().map(Vec::len).unwrap_or(0);
    if output_dim == 0
        || objectives.len() != objective_bounds.len()
        || objectives.len() != thresholds.len()
        || objectives
            .iter()
            .any(|row| row.len() != output_dim || row.iter().any(|value| !value.is_finite()))
        || objective_bounds
            .iter()
            .any(|&(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        || thresholds.iter().any(|value| !value.is_finite())
    {
        return Err(OutputConditionedHeadRefusal::MalformedObjectives);
    }

    let unresolved: Vec<usize> = objective_bounds
        .iter()
        .zip(thresholds)
        .enumerate()
        .filter_map(|(row, ((lower, _upper), threshold))| (*lower <= *threshold).then_some(row))
        .collect();
    if unresolved.len() > OUTPUT_CONDITIONED_MAX_ROWS {
        return Err(OutputConditionedHeadRefusal::MultipleUnresolvedRows);
    }
    let row_index = match unresolved.as_slice() {
        [] => return Err(OutputConditionedHeadRefusal::NoUnresolvedRow),
        [row] => *row,
        _ => return Err(OutputConditionedHeadRefusal::MultipleUnresolvedRows),
    };
    let target_preactivation = select_dense_head_preactivation(graph, node_bounds)
        .ok_or(OutputConditionedHeadRefusal::NoEligibleDenseHead)?;

    Ok(OutputConditionedHeadPlan {
        row_index,
        target_preactivation,
        max_coordinates: OUTPUT_CONDITIONED_MAX_COORDINATES,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OutputConditionedCoordinateTreatment {
    coordinate: usize,
    tail_coefficient: f64,
    gamma_lower: f32,
    gamma_upper: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct OutputConditionedCoordinateBounds {
    scope: OutputConditionedProofScope,
    treatment: OutputConditionedCoordinateTreatment,
    lower: f32,
    upper: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OutputConditionedExecutionRefusal {
    Plan(OutputConditionedHeadRefusal),
    Deadline,
    Scope,
    NoEligibleCoordinate,
    ConditionedBackward,
    NoConditionedGain,
    Replay,
    ReplayDidNotRefute,
    MalformedPublication,
}

fn classify_output_conditioned_error(
    error: &NyError,
    ordinary: OutputConditionedExecutionRefusal,
) -> OutputConditionedExecutionRefusal {
    if error.is_deadline_exceeded() {
        OutputConditionedExecutionRefusal::Deadline
    } else {
        ordinary
    }
}

/// Complete call-local result. It carries no numeric publication surface: the
/// caller can use the receipt only to finish the admitted disjunctive property
/// at the root. Ordinary objective bounds remain unchanged and therefore can
/// never misrepresent a premise-local lower bound as an unconditional one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OutputConditionedRootAcceptance {
    pub(super) row_index: usize,
    pub(super) target_coordinate: usize,
    pub(super) gamma_lower_bits: u32,
    pub(super) gamma_upper_bits: u32,
    pub(super) receipt: OutputConditionedRootRefutation,
}

impl OutputConditionedRootAcceptance {
    pub(super) fn gamma_lower(&self) -> f32 {
        f32::from_bits(self.gamma_lower_bits)
    }

    pub(super) fn gamma_upper(&self) -> f32 {
        f32::from_bits(self.gamma_upper_bits)
    }

    /// Converts the premise-local receipt into a terminal disjunctive count
    /// only when it accounts for the sole row without an ordinary proof.
    ///
    /// Keeping this invariant beside the receipt prevents a future caller
    /// from treating the receipt as a generic numeric-bound improvement.
    pub(super) fn terminal_verified_count(
        &self,
        ordinary_verified_count: usize,
        objective_count: usize,
        conjunctive: bool,
    ) -> Option<usize> {
        if conjunctive
            || self.row_index >= objective_count
            || ordinary_verified_count.checked_add(1) != Some(objective_count)
        {
            return None;
        }
        Some(objective_count)
    }
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, value: &[u8]) {
    hash_usize(hasher, value.len());
    hasher.update(value);
}

fn hash_f32_array(hasher: &mut Sha256, array: &ndarray::ArrayD<f32>, deadline: Instant) -> bool {
    hash_usize(hasher, array.ndim());
    for &dimension in array.shape() {
        hash_usize(hasher, dimension);
    }
    hash_usize(hasher, array.len());
    for (index, &value) in array.iter().enumerate() {
        if index.is_multiple_of(4096) && Instant::now() >= deadline {
            return false;
        }
        hasher.update(value.to_bits().to_le_bytes());
    }
    true
}

fn hash_bounded_tensor(hasher: &mut Sha256, bounds: &BoundedTensor, deadline: Instant) -> bool {
    if !hash_f32_array(hasher, bounds.lower(), deadline)
        || !hash_f32_array(hasher, bounds.upper(), deadline)
    {
        return false;
    }
    match bounds.l2_constraint() {
        Some(l2) => {
            hasher.update([1]);
            hash_usize(hasher, l2.axis());
            hash_f32_array(hasher, l2.center(), deadline)
                && hash_f32_array(hasher, l2.radius(), deadline)
        }
        None => {
            hasher.update([0]);
            true
        }
    }
}

fn finish_digest(hasher: Sha256) -> OutputConditionedDigest {
    OutputConditionedDigest::new(hasher.finalize().into())
}

fn graph_identity(graph: &GraphNetwork, deadline: Instant) -> Option<OutputConditionedDigest> {
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_GRAPH_V1\0");
    // `CutFoldScope` is collision-free for independently built graph instances
    // and is retained only by semantic clones. The topology below makes the
    // transcript human/audit stable within that exact authority boundary.
    hash_bytes(
        &mut hasher,
        format!("{:?}", graph.cut_fold_scope()).as_bytes(),
    );
    hash_bytes(&mut hasher, graph.output_name().as_bytes());
    let order = graph.exec_order().ok()?;
    hash_usize(&mut hasher, order.len());
    for (index, name) in order.iter().enumerate() {
        if index.is_multiple_of(8) && Instant::now() >= deadline {
            return None;
        }
        let node = graph.node(name)?;
        hash_bytes(&mut hasher, name.as_bytes());
        hash_bytes(&mut hasher, node.layer().layer_type().as_bytes());
        hash_usize(&mut hasher, node.inputs().len());
        for input in node.inputs() {
            hash_bytes(&mut hasher, input.as_bytes());
        }
    }
    Some(finish_digest(hasher))
}

fn input_identity(input: &BoundedTensor, deadline: Instant) -> Option<OutputConditionedDigest> {
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_INPUT_V1\0");
    hash_bounded_tensor(&mut hasher, input, deadline).then(|| finish_digest(hasher))
}

fn property_identity(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    deadline: Instant,
) -> Option<OutputConditionedDigest> {
    if objectives.len() != thresholds.len() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_PROPERTY_V1\0");
    hash_usize(&mut hasher, objectives.len());
    for (row_index, (row, threshold)) in objectives.iter().zip(thresholds).enumerate() {
        if row_index.is_multiple_of(8) && Instant::now() >= deadline {
            return None;
        }
        hash_usize(&mut hasher, row.len());
        for &coefficient in row {
            hasher.update(coefficient.to_bits().to_le_bytes());
        }
        hasher.update(threshold.to_bits().to_le_bytes());
    }
    Some(finish_digest(hasher))
}

fn objective_row_identity(
    row_index: usize,
    row: &[f32],
    threshold: f32,
) -> OutputConditionedDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_OBJECTIVE_ROW_V1\0");
    hash_usize(&mut hasher, row_index);
    hash_usize(&mut hasher, row.len());
    for &coefficient in row {
        hasher.update(coefficient.to_bits().to_le_bytes());
    }
    hasher.update(threshold.to_bits().to_le_bytes());
    finish_digest(hasher)
}

fn reference_bounds_identity(
    reference_bounds: &HashMap<String, BoundedTensor>,
    deadline: Instant,
) -> Option<OutputConditionedDigest> {
    let mut names: Vec<&String> = reference_bounds.keys().collect();
    names.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_REFERENCE_BOUNDS_V1\0");
    hash_usize(&mut hasher, names.len());
    for (index, name) in names.into_iter().enumerate() {
        if index.is_multiple_of(8) && Instant::now() >= deadline {
            return None;
        }
        hash_bytes(&mut hasher, name.as_bytes());
        if !hash_bounded_tensor(&mut hasher, reference_bounds.get(name)?, deadline) {
            return None;
        }
    }
    Some(finish_digest(hasher))
}

fn root_split_history_identity() -> OutputConditionedDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_SPLIT_HISTORY_V1\0ROOT_EMPTY");
    finish_digest(hasher)
}

fn build_proof_scope(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    reference_bounds: &HashMap<String, BoundedTensor>,
    plan: &OutputConditionedHeadPlan,
    deadline: Instant,
) -> Option<OutputConditionedProofScope> {
    let objective_row = objectives.get(plan.row_index)?;
    let threshold = *thresholds.get(plan.row_index)?;
    Some(OutputConditionedProofScope::new(
        plan.row_index,
        plan.target_preactivation.clone(),
        graph.cut_fold_scope(),
        graph_identity(graph, deadline)?,
        input_identity(input, deadline)?,
        property_identity(objectives, thresholds, deadline)?,
        objective_row_identity(plan.row_index, objective_row, threshold),
        reference_bounds_identity(reference_bounds, deadline)?,
        root_split_history_identity(),
    ))
}

/// Select one coordinate in an exact final `Linear -> ReLU -> Linear` head.
///
/// Ranking is heuristic only and therefore uses f64 round-to-nearest: every
/// selected non-negative gamma is independently certified by the two-seed
/// backward. Stable ascending traversal gives a deterministic coordinate tie
/// break. The gamma is the one-step cancellation scale `1 / |cW_j|`.
fn select_single_coordinate_treatment(
    graph: &GraphNetwork,
    plan: &OutputConditionedHeadPlan,
    objective_row: &[f32],
    reference_bounds: &HashMap<String, BoundedTensor>,
) -> Option<OutputConditionedCoordinateTreatment> {
    if plan.max_coordinates == 0 {
        return None;
    }
    let order = graph.exec_order().ok()?;
    let output_name = if graph.output_name().is_empty() {
        order.last()?.as_str()
    } else {
        graph.output_name()
    };
    let output_node = graph.node(output_name)?;
    let Layer::Linear(output_linear) = output_node.layer() else {
        return None;
    };
    let [relu_name] = output_node.inputs() else {
        return None;
    };
    let relu_node = graph.node(relu_name)?;
    if !matches!(relu_node.layer(), Layer::ReLU(_)) {
        return None;
    }
    let [preactivation] = relu_node.inputs() else {
        return None;
    };
    if preactivation != &plan.target_preactivation {
        return None;
    }

    let target_bounds = reference_bounds.get(&plan.target_preactivation)?.flatten();
    let target_width = target_bounds.len();
    if objective_row.len() != output_linear.weight.nrows()
        || target_width != output_linear.weight.ncols()
    {
        return None;
    }

    let mut selected: Option<(f64, OutputConditionedCoordinateTreatment)> = None;
    for coordinate in 0..target_width {
        let lower = target_bounds.lower()[[coordinate]];
        let upper = target_bounds.upper()[[coordinate]];
        if !(lower.is_finite() && upper.is_finite() && lower < 0.0 && upper > 0.0) {
            continue;
        }
        let tail_coefficient = objective_row
            .iter()
            .enumerate()
            .map(|(output, &coefficient)| {
                coefficient as f64 * output_linear.weight[[output, coordinate]] as f64
            })
            .sum::<f64>();
        let score = tail_coefficient.abs() * (upper as f64 - lower as f64);
        if !tail_coefficient.is_finite()
            || tail_coefficient == 0.0
            || !score.is_finite()
            || score <= 0.0
        {
            continue;
        }
        let gamma = (1.0_f64 / tail_coefficient.abs()) as f32;
        if !gamma.is_finite() || gamma <= 0.0 {
            continue;
        }
        let treatment = if tail_coefficient < 0.0 {
            OutputConditionedCoordinateTreatment {
                coordinate,
                tail_coefficient,
                gamma_lower: gamma,
                gamma_upper: 0.0,
            }
        } else {
            OutputConditionedCoordinateTreatment {
                coordinate,
                tail_coefficient,
                gamma_lower: 0.0,
                gamma_upper: gamma,
            }
        };
        if selected
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            selected = Some((score, treatment));
        }
    }
    selected.map(|(_, treatment)| treatment)
}

/// Build the call-local alpha state used by the ordinary conditioned replay.
///
/// The inherited root state remains the starting point when present. For the
/// exact final dense-head ReLU, positive active-row tail coefficients use the
/// valid lower slope `alpha=1`. This deterministic endpoint is important after
/// a negative-tail coordinate has been conditioned: on the dependency fixture
/// it preserves the positive `ReLU(x)` branch and exposes the `3/7` proof,
/// whereas the symmetric-box tie heuristic (`alpha=0`) discards it. Every
/// installed value is still an ordinary alpha-CROWN relaxation parameter in
/// `[0,1]`; the state is consumed once and never published.
fn conditioned_replay_alpha_state(
    graph: &GraphNetwork,
    plan: &OutputConditionedHeadPlan,
    objective_row: &[f32],
    reference_bounds: &HashMap<String, BoundedTensor>,
    inherited: Option<&GraphAlphaState>,
) -> Option<GraphAlphaState> {
    let order = graph.exec_order().ok()?;
    let output_name = if graph.output_name().is_empty() {
        order.last()?.as_str()
    } else {
        graph.output_name()
    };
    let output_node = graph.node(output_name)?;
    let Layer::Linear(output_linear) = output_node.layer() else {
        return None;
    };
    let [relu_name] = output_node.inputs() else {
        return None;
    };
    let relu_node = graph.node(relu_name)?;
    if !matches!(relu_node.layer(), Layer::ReLU(_)) {
        return None;
    }
    let [preactivation] = relu_node.inputs() else {
        return None;
    };
    if preactivation != &plan.target_preactivation
        || objective_row.len() != output_linear.weight.nrows()
    {
        return None;
    }
    let target_bounds = reference_bounds.get(&plan.target_preactivation)?;
    if target_bounds.len() != output_linear.weight.ncols() {
        return None;
    }

    let mut state = inherited.cloned().unwrap_or_else(GraphAlphaState::new);
    if state.alpha(relu_name).is_none() {
        state.add_relu_node(relu_name, target_bounds, false).ok()?;
    }
    let (lower_alpha, _upper_alpha) = state.relu_alpha_pair_mut(relu_name)?;
    if lower_alpha.len() != target_bounds.len() {
        return None;
    }
    for coordinate in 0..target_bounds.len() {
        let tail_coefficient = objective_row
            .iter()
            .enumerate()
            .map(|(output, &coefficient)| {
                coefficient as f64 * output_linear.weight[[output, coordinate]] as f64
            })
            .sum::<f64>();
        if !tail_coefficient.is_finite() {
            return None;
        }
        if tail_coefficient > 0.0 {
            lower_alpha[coordinate] = 1.0;
        }
    }
    Some(state)
}

/// Construct a private reference map containing exactly one conditioned target
/// coordinate. The source map is immutable and cloned only after every
/// endpoint/scope check, so a refusal has zero external mutation.
fn private_conditioned_reference_bounds(
    expected_scope: &OutputConditionedProofScope,
    reference_bounds: &HashMap<String, BoundedTensor>,
    candidate: &OutputConditionedCoordinateBounds,
) -> Option<HashMap<String, BoundedTensor>> {
    if &candidate.scope != expected_scope {
        return None;
    }
    let current = reference_bounds.get(&expected_scope.target_preactivation)?;
    if current.l2_constraint().is_some() || candidate.treatment.coordinate >= current.len() {
        return None;
    }
    let coordinate = candidate.treatment.coordinate;
    let old_lower = *current.lower().iter().nth(coordinate)?;
    let old_upper = *current.upper().iter().nth(coordinate)?;
    if !old_lower.is_finite()
        || !old_upper.is_finite()
        || old_lower > old_upper
        || !candidate.lower.is_finite()
        || !candidate.upper.is_finite()
        || candidate.lower > candidate.upper
    {
        return None;
    }
    let new_lower = old_lower.max(candidate.lower);
    let new_upper = old_upper.min(candidate.upper);
    if new_lower > new_upper
        || (new_lower.to_bits() == old_lower.to_bits()
            && new_upper.to_bits() == old_upper.to_bits())
    {
        return None;
    }

    let mut lower = current.lower().clone();
    let mut upper = current.upper().clone();
    *lower.iter_mut().nth(coordinate)? = new_lower;
    *upper.iter_mut().nth(coordinate)? = new_upper;
    let target = BoundedTensor::new(lower, upper).ok()?;
    let mut private = reference_bounds.clone();
    private.insert(expected_scope.target_preactivation.clone(), target);
    Some(private)
}

fn transcript_identity(
    scope: &OutputConditionedProofScope,
    candidate: &OutputConditionedCoordinateBounds,
    conditional_lower: f32,
    conditional_upper: f32,
) -> OutputConditionedDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"NY_OUTPUT_CONDITIONED_TRANSCRIPT_V1\0");
    hash_usize(&mut hasher, scope.row_index);
    hash_bytes(&mut hasher, scope.target_preactivation.as_bytes());
    hasher.update(scope.graph_sha256.0);
    hasher.update(scope.input_box_sha256.0);
    hasher.update(scope.property_sha256.0);
    hasher.update(scope.objective_row_sha256.0);
    hasher.update(scope.reference_bounds_sha256.0);
    hasher.update(scope.split_history_sha256.0);
    hash_usize(&mut hasher, candidate.treatment.coordinate);
    hasher.update(candidate.treatment.tail_coefficient.to_bits().to_le_bytes());
    hasher.update(candidate.treatment.gamma_lower.to_bits().to_le_bytes());
    hasher.update(candidate.treatment.gamma_upper.to_bits().to_le_bytes());
    hasher.update(candidate.lower.to_bits().to_le_bytes());
    hasher.update(candidate.upper.to_bits().to_le_bytes());
    hasher.update(conditional_lower.to_bits().to_le_bytes());
    hasher.update(conditional_upper.to_bits().to_le_bytes());
    finish_digest(hasher)
}

#[allow(clippy::too_many_arguments)]
fn run_output_conditioned_root_refutation(
    gate_enabled: bool,
    artifact_authority: VerificationArtifactAuthority,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    objective_bounds: &[(f32, f32)],
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
    reference_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Result<OutputConditionedRootAcceptance, OutputConditionedExecutionRefusal> {
    // Defense in depth for direct/internal callers: refuse certificate-export
    // authority before even consulting the clock. The public wrapper performs
    // the same check before its gate, hashes, evaluator, receipt, or telemetry.
    if artifact_authority != VerificationArtifactAuthority::VerdictOnly {
        return Err(OutputConditionedExecutionRefusal::Plan(
            OutputConditionedHeadRefusal::CertificateExportAuthority,
        ));
    }
    if Instant::now() >= deadline {
        return Err(OutputConditionedExecutionRefusal::Deadline);
    }
    let plan = build_output_conditioned_head_plan(
        gate_enabled,
        artifact_authority,
        conjunctive,
        crown_backward_layers,
        objectives,
        objective_bounds,
        thresholds,
        graph,
        reference_bounds,
    )
    .map_err(OutputConditionedExecutionRefusal::Plan)?;
    let scope = build_proof_scope(
        graph,
        input,
        objectives,
        thresholds,
        reference_bounds,
        &plan,
        deadline,
    )
    .ok_or(OutputConditionedExecutionRefusal::Scope)?;
    let objective_row = objectives
        .get(plan.row_index)
        .ok_or(OutputConditionedExecutionRefusal::Scope)?;
    let threshold = *thresholds
        .get(plan.row_index)
        .ok_or(OutputConditionedExecutionRefusal::Scope)?;
    let treatment =
        select_single_coordinate_treatment(graph, &plan, objective_row, reference_bounds)
            .ok_or(OutputConditionedExecutionRefusal::NoEligibleCoordinate)?;
    debug_assert!(treatment.gamma_lower >= 0.0 && treatment.gamma_upper >= 0.0);

    // The additional frontier deliberately uses fixed-slope ReLUs. The audited
    // two-seed core refuses alpha-ReLU for this frontier until its coefficient
    // error channel is independently closed. The ordinary proof replay below
    // may still use the authoritative root alpha state.
    let (lower, upper) = graph
        .propagate_output_conditioned_crown_to_node_subset(
            input,
            &plan.target_preactivation,
            &[treatment.coordinate],
            objective_row,
            threshold,
            &[treatment.gamma_lower],
            &[treatment.gamma_upper],
            reference_bounds,
            reference_bounds,
            None,
            engine,
            Some(deadline),
        )
        .map_err(|error| {
            classify_output_conditioned_error(
                &error,
                OutputConditionedExecutionRefusal::ConditionedBackward,
            )
        })?;
    if Instant::now() >= deadline || lower.len() != 1 || upper.len() != 1 {
        return Err(OutputConditionedExecutionRefusal::Deadline);
    }
    let candidate = OutputConditionedCoordinateBounds {
        scope: scope.clone(),
        treatment,
        lower: lower[0],
        upper: upper[0],
    };
    let private_bounds = private_conditioned_reference_bounds(&scope, reference_bounds, &candidate)
        .ok_or(OutputConditionedExecutionRefusal::NoConditionedGain)?;
    if Instant::now() >= deadline {
        return Err(OutputConditionedExecutionRefusal::Deadline);
    }

    let same_row_spec =
        ndarray::Array2::from_shape_vec((1, objective_row.len()), objective_row.clone())
            .map_err(|_| OutputConditionedExecutionRefusal::Replay)?;
    let replay_alpha_state =
        conditioned_replay_alpha_state(graph, &plan, objective_row, reference_bounds, alpha_state)
            .ok_or(OutputConditionedExecutionRefusal::Replay)?;
    // This is the sole authority boundary: the ordinary spec backward consumes
    // the private premise-scoped box and must itself prove `LB > threshold`.
    // `run()` cannot return or publish a linear/root cache.
    let replay = SpecCrownRequest::new(graph, input, &same_row_spec, engine)
        .node_bounds(&private_bounds)
        .alpha_state_opt(Some(&replay_alpha_state))
        .deadline_opt(Some(deadline))
        .truncate_after_opt(crown_backward_layers)
        .run()
        .map_err(|error| {
            classify_output_conditioned_error(&error, OutputConditionedExecutionRefusal::Replay)
        })?
        .flatten();
    if Instant::now() >= deadline
        || replay.len() != 1
        || !replay.lower()[[0]].is_finite()
        || !replay.upper()[[0]].is_finite()
        || replay.lower()[[0]] > replay.upper()[[0]]
    {
        return Err(OutputConditionedExecutionRefusal::Deadline);
    }
    let conditional_lower = replay.lower()[[0]];
    let conditional_upper = replay.upper()[[0]];
    let historical_lower = objective_bounds
        .get(plan.row_index)
        .map(|bounds| bounds.0)
        .ok_or(OutputConditionedExecutionRefusal::MalformedPublication)?;
    if conditional_lower <= threshold || conditional_lower <= historical_lower {
        return Err(OutputConditionedExecutionRefusal::ReplayDidNotRefute);
    }
    // Recheck the terminal-only publication invariant after all work. If any
    // other row is ordinary-unresolved, a conditional numeric marker could
    // escape into BaB and is therefore forbidden.
    if objective_bounds.len() != thresholds.len()
        || objective_bounds
            .iter()
            .zip(thresholds)
            .enumerate()
            .any(|(row, ((lower, _), threshold))| row != plan.row_index && *lower <= *threshold)
    {
        return Err(OutputConditionedExecutionRefusal::MalformedPublication);
    }
    let transcript = transcript_identity(&scope, &candidate, conditional_lower, conditional_upper);
    let receipt =
        OutputConditionedRootRefutation::new(scope, conditional_lower, threshold, transcript)
            .ok_or(OutputConditionedExecutionRefusal::ReplayDidNotRefute)?;
    if Instant::now() >= deadline {
        return Err(OutputConditionedExecutionRefusal::Deadline);
    }
    Ok(OutputConditionedRootAcceptance {
        row_index: plan.row_index,
        target_coordinate: treatment.coordinate,
        gamma_lower_bits: treatment.gamma_lower.to_bits(),
        gamma_upper_bits: treatment.gamma_upper.to_bits(),
        receipt,
    })
}

/// Exact-gated, deadline-atomic root entry point.
///
/// No caller-owned object is mutably borrowed. Every failure, including a late
/// result, therefore falls back to the ordinary root with zero mutation.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_output_conditioned_root_refutation(
    artifact_authority: VerificationArtifactAuthority,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    objective_bounds: &[(f32, f32)],
    conjunctive: bool,
    crown_backward_layers: Option<usize>,
    reference_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    authority_deadline: Option<Instant>,
) -> Option<OutputConditionedRootAcceptance> {
    // Certificate-export authority is fail-closed before the gate, clock,
    // hashes, evaluator, receipt construction, or telemetry. The current
    // external certificate format has no two-node conditional transcript.
    if artifact_authority != VerificationArtifactAuthority::VerdictOnly {
        return None;
    }
    if !output_conditioned_head_enabled() {
        return None;
    }
    let started_at = Instant::now();
    let local_deadline = started_at.checked_add(OUTPUT_CONDITIONED_ROOT_MAX_RUNTIME)?;
    let deadline = authority_deadline.map_or(local_deadline, |outer| outer.min(local_deadline));
    if deadline <= started_at {
        return None;
    }
    match run_output_conditioned_root_refutation(
        true,
        artifact_authority,
        graph,
        input,
        objectives,
        thresholds,
        objective_bounds,
        conjunctive,
        crown_backward_layers,
        reference_bounds,
        alpha_state,
        engine,
        deadline,
    ) {
        Ok(accepted) => Some(accepted),
        Err(reason) => {
            tracing::debug!(
                ?reason,
                elapsed_ms = started_at.elapsed().as_millis() as u64,
                "Output-conditioned hidden root treatment refused"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::{BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier};
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::{GraphNode, NETWORK_INPUT};
    use ndarray::{arr1, arr2};

    fn bounded(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).unwrap()
    }

    #[test]
    fn conditioned_error_classifier_preserves_deadline_authority() {
        assert_eq!(
            classify_output_conditioned_error(
                &NyError::DeadlineExceeded("test deadline".into()),
                OutputConditionedExecutionRefusal::Replay,
            ),
            OutputConditionedExecutionRefusal::Deadline
        );
        assert_eq!(
            classify_output_conditioned_error(
                &NyError::InvalidSpec("test failure".into()),
                OutputConditionedExecutionRefusal::ConditionedBackward,
            ),
            OutputConditionedExecutionRefusal::ConditionedBackward
        );
    }

    fn linear() -> LinearLayer {
        LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None).unwrap()
    }

    fn named_dense_head(prefix: &str) -> (GraphNetwork, HashMap<String, BoundedTensor>) {
        let early_pre = format!("{prefix}_early_pre");
        let early_relu = format!("{prefix}_early_relu");
        let head_pre = format!("{prefix}_head_pre");
        let head_relu = format!("{prefix}_head_relu");
        let output = format!("{prefix}_output");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            &early_pre,
            Layer::Linear(linear()),
            vec![NETWORK_INPUT.to_string()],
        ));
        graph.add_node(GraphNode::new(
            &early_relu,
            Layer::ReLU(ReLULayer),
            vec![early_pre.clone()],
        ));
        graph.add_node(GraphNode::new(
            &head_pre,
            Layer::Linear(linear()),
            vec![early_relu],
        ));
        graph.add_node(GraphNode::new(
            &head_relu,
            Layer::ReLU(ReLULayer),
            vec![head_pre.clone()],
        ));
        graph.add_node(GraphNode::new(
            &output,
            Layer::Linear(linear()),
            vec![head_relu],
        ));
        graph.set_output(&output);

        let bounds = HashMap::from([
            (early_pre, bounded(&[-1.0, -0.5], &[1.0, 0.5])),
            (head_pre, bounded(&[-0.25, -0.1], &[0.75, 0.3])),
        ]);
        (graph, bounds)
    }

    fn digest(byte: u8) -> OutputConditionedDigest {
        OutputConditionedDigest::new([byte; 32])
    }

    fn scope() -> OutputConditionedProofScope {
        OutputConditionedProofScope::new(
            3,
            "head".to_string(),
            CutFoldScope::fresh(),
            digest(1),
            digest(2),
            digest(3),
            digest(4),
            digest(5),
            digest(6),
        )
    }

    #[test]
    fn gate_accepts_exactly_one() {
        assert!(parse_output_conditioned_head_gate(Some("1")));
        for raw in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!parse_output_conditioned_head_gate(raw));
        }
    }

    #[test]
    fn selector_is_topological_and_exporter_name_independent() {
        let (graph_a, bounds_a) = named_dense_head("alpha");
        let (graph_b, bounds_b) = named_dense_head("renamed");
        assert_eq!(
            select_dense_head_preactivation(&graph_a, &bounds_a).as_deref(),
            Some("alpha_head_pre")
        );
        assert_eq!(
            select_dense_head_preactivation(&graph_b, &bounds_b).as_deref(),
            Some("renamed_head_pre")
        );
    }

    #[test]
    fn selector_ignores_dead_dense_branch() {
        let (mut graph, mut bounds) = named_dense_head("live");
        graph.add_node(GraphNode::new(
            "dead_pre",
            Layer::Linear(linear()),
            vec![NETWORK_INPUT.to_string()],
        ));
        graph.add_node(GraphNode::new(
            "dead_relu",
            Layer::ReLU(ReLULayer),
            vec!["dead_pre".to_string()],
        ));
        bounds.insert("dead_pre".to_string(), bounded(&[-5.0, -5.0], &[5.0, 5.0]));
        assert_eq!(
            select_dense_head_preactivation(&graph, &bounds).as_deref(),
            Some("live_head_pre")
        );
    }

    #[test]
    fn planner_admits_only_one_unresolved_row() {
        let (graph, bounds) = named_dense_head("model");
        let objectives = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let plan = build_output_conditioned_head_plan(
            true,
            VerificationArtifactAuthority::VerdictOnly,
            false,
            None,
            &objectives,
            &[(0.5, 1.0), (-0.5, 1.0)],
            &[0.0, 0.0],
            &graph,
            &bounds,
        )
        .unwrap();
        assert_eq!(plan.row_index, 1);
        assert_eq!(plan.target_preactivation, "model_head_pre");
        assert_eq!(plan.max_coordinates, 8);

        assert_eq!(
            build_output_conditioned_head_plan(
                true,
                VerificationArtifactAuthority::VerdictOnly,
                false,
                None,
                &objectives,
                &[(-0.5, 1.0), (-0.25, 1.0)],
                &[0.0, 0.0],
                &graph,
                &bounds,
            ),
            Err(OutputConditionedHeadRefusal::MultipleUnresolvedRows)
        );
    }

    #[test]
    fn planner_is_default_dark_and_refuses_truncation() {
        let (graph, bounds) = named_dense_head("model");
        let objectives = vec![vec![1.0, -1.0]];
        let objective_bounds = [(-0.5, 1.0)];
        let thresholds = [0.0];
        assert_eq!(
            build_output_conditioned_head_plan(
                false,
                VerificationArtifactAuthority::VerdictOnly,
                false,
                None,
                &objectives,
                &objective_bounds,
                &thresholds,
                &graph,
                &bounds,
            ),
            Err(OutputConditionedHeadRefusal::GateDisabled)
        );
        assert_eq!(
            build_output_conditioned_head_plan(
                true,
                VerificationArtifactAuthority::VerdictOnly,
                false,
                Some(3),
                &objectives,
                &objective_bounds,
                &thresholds,
                &graph,
                &bounds,
            ),
            Err(OutputConditionedHeadRefusal::TruncatedBackward)
        );
        assert_eq!(
            build_output_conditioned_head_plan(
                true,
                VerificationArtifactAuthority::CertificateExport,
                false,
                None,
                &objectives,
                &objective_bounds,
                &thresholds,
                &graph,
                &bounds,
            ),
            Err(OutputConditionedHeadRefusal::CertificateExportAuthority)
        );
    }

    #[test]
    fn proof_scope_isolates_every_semantic_dimension() {
        let baseline = scope();
        let mut changed = baseline.clone();
        changed.row_index += 1;
        assert_ne!(baseline, changed);

        let mut changed = baseline.clone();
        changed.target_preactivation = "other".to_string();
        assert_ne!(baseline, changed);

        let mut changed = baseline.clone();
        changed.graph_scope = CutFoldScope::fresh();
        assert_ne!(baseline, changed);

        for field in 0..6 {
            let mut changed = baseline.clone();
            match field {
                0 => changed.graph_sha256 = digest(9),
                1 => changed.input_box_sha256 = digest(9),
                2 => changed.property_sha256 = digest(9),
                3 => changed.objective_row_sha256 = digest(9),
                4 => changed.reference_bounds_sha256 = digest(9),
                5 => changed.split_history_sha256 = digest(9),
                _ => unreachable!(),
            }
            assert_ne!(baseline, changed);
        }
    }

    #[test]
    fn refutation_receipt_requires_strict_finite_root_proof() {
        assert!(OutputConditionedRootRefutation::new(scope(), 0.1, 0.0, digest(7)).is_some());
        assert!(OutputConditionedRootRefutation::new(scope(), 0.0, 0.0, digest(7)).is_none());
        assert!(
            OutputConditionedRootRefutation::new(scope(), f32::INFINITY, 0.0, digest(7)).is_none()
        );
    }

    #[test]
    fn terminal_receipt_accounts_for_exactly_one_disjunctive_row() {
        let receipt = OutputConditionedRootRefutation::new(scope(), 0.1, 0.0, digest(7)).unwrap();
        let acceptance = OutputConditionedRootAcceptance {
            row_index: 3,
            target_coordinate: 0,
            gamma_lower_bits: 0.0_f32.to_bits(),
            gamma_upper_bits: 1.0_f32.to_bits(),
            receipt,
        };

        assert_eq!(acceptance.terminal_verified_count(3, 4, false), Some(4));
        assert_eq!(acceptance.terminal_verified_count(3, 4, true), None);
        assert_eq!(acceptance.terminal_verified_count(2, 4, false), None);
        assert_eq!(acceptance.terminal_verified_count(4, 4, false), None);

        let mut out_of_range = acceptance;
        out_of_range.row_index = 4;
        assert_eq!(out_of_range.terminal_verified_count(3, 4, false), None);
    }

    #[test]
    fn one_coordinate_selector_projects_gamma_nonnegative_on_the_relevant_side() {
        let (graph, bounds) = named_dense_head("ranked");
        let plan = OutputConditionedHeadPlan {
            row_index: 0,
            target_preactivation: "ranked_head_pre".to_string(),
            max_coordinates: 1,
        };
        let selected =
            select_single_coordinate_treatment(&graph, &plan, &[-1.0, 0.0], &bounds).unwrap();
        assert_eq!(selected.coordinate, 0);
        assert!(selected.tail_coefficient < 0.0);
        assert!(selected.gamma_lower.is_finite() && selected.gamma_lower > 0.0);
        assert_eq!(selected.gamma_upper.to_bits(), 0.0_f32.to_bits());

        let selected =
            select_single_coordinate_treatment(&graph, &plan, &[1.0, 0.0], &bounds).unwrap();
        assert_eq!(selected.coordinate, 0);
        assert!(selected.tail_coefficient > 0.0);
        assert_eq!(selected.gamma_lower.to_bits(), 0.0_f32.to_bits());
        assert!(selected.gamma_upper.is_finite() && selected.gamma_upper > 0.0);
    }

    #[test]
    fn private_conditioned_map_is_scope_exact_and_never_mutates_source() {
        let expected = scope();
        let reference =
            HashMap::from([("head".to_string(), bounded(&[-1.0_f32, -0.5], &[1.0, 0.5]))]);
        let before = reference["head"].clone();
        let treatment = OutputConditionedCoordinateTreatment {
            coordinate: 0,
            tail_coefficient: 1.0,
            gamma_lower: 0.0,
            gamma_upper: 1.0,
        };
        let mut foreign_scope = expected.clone();
        foreign_scope.objective_row_sha256 = digest(99);
        let foreign = OutputConditionedCoordinateBounds {
            scope: foreign_scope,
            treatment,
            lower: -1.0,
            upper: -0.25,
        };
        assert!(private_conditioned_reference_bounds(&expected, &reference, &foreign).is_none());
        assert_eq!(reference["head"].lower(), before.lower());
        assert_eq!(reference["head"].upper(), before.upper());

        let local = private_conditioned_reference_bounds(
            &expected,
            &reference,
            &OutputConditionedCoordinateBounds {
                scope: expected.clone(),
                treatment,
                lower: -1.0,
                upper: -0.25,
            },
        )
        .expect("matching scope and strict intersection");
        assert_eq!(local["head"].upper().iter().next().copied(), Some(-0.25));
        assert_eq!(reference["head"].upper(), before.upper());
    }

    #[test]
    fn reference_identity_changes_on_one_endpoint_ulp() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let baseline = HashMap::from([("head".to_string(), bounded(&[-1.0], &[1.0]))]);
        let changed = HashMap::from([(
            "head".to_string(),
            bounded(&[-1.0], &[f32::from_bits(1.0_f32.to_bits() + 1)]),
        )]);
        assert_ne!(
            reference_bounds_identity(&baseline, deadline),
            reference_bounds_identity(&changed, deadline)
        );
    }

    /// Dependency fixture requested by the root-publication audit.
    ///
    /// For every x, `ReLU(0.5x) = 0.5 ReLU(x)`, so the exact output is
    /// `y = -2 ReLU(0.5x) + ReLU(x) + 1 = 1`. Ordinary fixed-slope CROWN loses
    /// that shared dependence and returns a lower bound at most 0.4.
    fn dependent_relu_head() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "target",
            Layer::Linear(
                LinearLayer::new(arr2(&[[0.5_f32], [1.0]]), None)
                    .expect("two dependent preactivations"),
            ),
        ));
        graph.add_node(GraphNode::new(
            "activation",
            Layer::ReLU(ReLULayer),
            vec!["target".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(arr2(&[[-2.0_f32, 1.0]]), Some(arr1(&[1.0])))
                    .expect("dependency-cancelling output"),
            ),
            vec!["activation".to_string()],
        ));
        graph.set_output("output");
        (graph, bounded(&[-1.0], &[1.0]))
    }

    fn verify_dependent_relu_root(
        threshold: f32,
        artifact_authority: VerificationArtifactAuthority,
    ) -> crate::beta_crown::BetaCrownResult {
        let (graph, input) = dependent_relu_head();
        let config = BetaCrownConfig {
            verification_artifact_authority: artifact_authority,
            use_alpha_crown: false,
            root_beta_iterations: 0,
            beta_iterations: 0,
            max_depth: 0,
            max_domains: 1,
            batch_size: 1,
            timeout: Duration::from_secs(5),
            ..Default::default()
        };
        BetaCrownVerifier::new(config)
            .verify_graph_relu_split_multi_objective(&graph, &input, &[vec![1.0_f32]], &[threshold])
            .expect("dependency root verification")
    }

    #[test]
    fn no_deadline_conditioned_oracle_remains_exact_but_finite_transaction_declines() {
        // Under the alleged violation y <= 0.4, coordinate 0 and
        // gamma_lower=0.5 prove z0 >= -0.2. Replaying the ordinary objective
        // against only that private bound yields 3/7 ~= 0.42857 > 0.4.
        let (graph, input) = dependent_relu_head();
        let references = graph.collect_node_bounds(&input).expect("reference IBP");
        let source_target = references["target"].clone();
        let ordinary = SpecCrownRequest::new(&graph, &input, &arr2(&[[1.0_f32]]), None)
            .node_bounds(&references)
            .run()
            .expect("ordinary fixed-slope replay")
            .flatten();
        let ordinary_bounds = [(ordinary.lower()[[0]], ordinary.upper()[[0]])];
        assert!(
            ordinary_bounds[0].0 <= 0.4,
            "fixture must not be proved by ordinary root CROWN: {:?}",
            ordinary_bounds[0]
        );

        let plan = build_output_conditioned_head_plan(
            true,
            VerificationArtifactAuthority::VerdictOnly,
            false,
            None,
            &[vec![1.0]],
            &ordinary_bounds,
            &[0.4],
            &graph,
            &references,
        )
        .expect("dependency fixture must be admitted");
        let replay_alpha = conditioned_replay_alpha_state(&graph, &plan, &[1.0], &references, None)
            .expect("call-local final-head alpha state");
        assert_eq!(
            replay_alpha.alpha("activation").unwrap()[1].to_bits(),
            1.0_f32.to_bits(),
            "positive tail coordinate must retain its valid identity lower slope"
        );
        let ordinary_with_replay_alpha =
            SpecCrownRequest::new(&graph, &input, &arr2(&[[1.0_f32]]), None)
                .node_bounds(&references)
                .alpha_state_opt(Some(&replay_alpha))
                .run()
                .expect("ordinary replay with identical local alpha policy")
                .flatten();
        assert!(
            ordinary_with_replay_alpha.lower()[[0]] <= 0.4,
            "the alpha policy alone must not prove the fixture before conditioning"
        );
        let treatment =
            select_single_coordinate_treatment(&graph, &plan, &[1.0], &references).unwrap();
        assert_eq!(
            treatment.coordinate, 0,
            "score tie must select coordinate 0"
        );
        assert_eq!(treatment.tail_coefficient, -2.0);
        assert_eq!(treatment.gamma_lower.to_bits(), 0.5_f32.to_bits());
        assert_eq!(treatment.gamma_upper.to_bits(), 0.0_f32.to_bits());

        let (conditioned_lower, _) = graph
            .propagate_output_conditioned_crown_to_node_subset(
                &input,
                "target",
                &[0],
                &[1.0],
                0.4,
                &[0.5],
                &[0.0],
                &references,
                &references,
                None,
                None,
                None,
            )
            .expect("two-seed dependency treatment");
        assert!(
            (-0.200_002..=-0.199_999).contains(&conditioned_lower[0]),
            "conditioned z0 lower must enclose exact -0.2, got {}",
            conditioned_lower[0]
        );

        let refusal = run_output_conditioned_root_refutation(
            true,
            VerificationArtifactAuthority::VerdictOnly,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.4],
            &ordinary_bounds,
            false,
            None,
            &references,
            None,
            None,
            Instant::now() + Duration::from_secs(5),
        )
        .expect_err("finite output-conditioned transaction must decline before evaluator setup");
        assert_eq!(
            refusal,
            OutputConditionedExecutionRefusal::ConditionedBackward
        );
        assert_eq!(references["target"].lower(), source_target.lower());
        assert_eq!(references["target"].upper(), source_target.upper());
    }

    #[ntest::timeout(20000)]
    #[test]
    fn production_root_gate_and_artifact_authority_are_fail_closed() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(OUTPUT_CONDITIONED_HEAD_ENV);
            let gate_off =
                verify_dependent_relu_root(0.4, VerificationArtifactAuthority::VerdictOnly);
            assert_ne!(
                gate_off.result,
                BabVerificationStatus::Verified,
                "ordinary root CROWN must not prove the dependency fixture with the gate dark"
            );

            env.set(OUTPUT_CONDITIONED_HEAD_ENV, "1");
            let verdict_only =
                verify_dependent_relu_root(0.4, VerificationArtifactAuthority::VerdictOnly);
            assert_ne!(
                verdict_only.result,
                BabVerificationStatus::Verified,
                "finite output-conditioned setup must fail closed to the ordinary root route"
            );

            let certificate =
                verify_dependent_relu_root(0.4, VerificationArtifactAuthority::CertificateExport);
            assert_ne!(
                certificate.result,
                BabVerificationStatus::Verified,
                "certificate-export authority must refuse the unexportable conditional receipt"
            );

            for satisfiable_threshold in [1.0_f32, 1.1] {
                let satisfiable = verify_dependent_relu_root(
                    satisfiable_threshold,
                    VerificationArtifactAuthority::VerdictOnly,
                );
                assert_ne!(
                    satisfiable.result,
                    BabVerificationStatus::Verified,
                    "satisfiable premise y <= {satisfiable_threshold} must produce no receipt"
                );
            }
        });
    }

    #[test]
    fn certificate_authority_refuses_before_any_root_receipt() {
        let (graph, input) = dependent_relu_head();
        let references = graph.collect_node_bounds(&input).expect("reference IBP");
        let before = references["target"].clone();
        assert_eq!(
            run_output_conditioned_root_refutation(
                true,
                VerificationArtifactAuthority::CertificateExport,
                &graph,
                &input,
                &[vec![1.0]],
                &[0.4],
                &[(0.0, 2.0)],
                false,
                None,
                &references,
                None,
                None,
                Instant::now() + Duration::from_secs(5),
            ),
            Err(OutputConditionedExecutionRefusal::Plan(
                OutputConditionedHeadRefusal::CertificateExportAuthority
            ))
        );

        ny_test_utils::env::with_env_edits(|env| {
            env.set(OUTPUT_CONDITIONED_HEAD_ENV, "1");
            assert!(
                try_output_conditioned_root_refutation(
                    VerificationArtifactAuthority::CertificateExport,
                    &graph,
                    &input,
                    &[vec![1.0]],
                    &[0.4],
                    &[(0.0, 2.0)],
                    false,
                    None,
                    &references,
                    None,
                    None,
                    Some(Instant::now() + Duration::from_secs(5)),
                )
                .is_none(),
                "the exact ambient gate must not grant verdict-only authority"
            );
        });
        assert_eq!(references["target"].lower(), before.lower());
        assert_eq!(references["target"].upper(), before.upper());
    }

    #[test]
    fn expired_transaction_refuses_before_any_candidate_state_exists() {
        let (graph, input) = dependent_relu_head();
        let references = graph.collect_node_bounds(&input).expect("reference IBP");
        let before = references["target"].clone();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("monotonic subtraction");
        assert_eq!(
            run_output_conditioned_root_refutation(
                true,
                VerificationArtifactAuthority::VerdictOnly,
                &graph,
                &input,
                &[vec![1.0]],
                &[0.4],
                &[(0.0, 2.0)],
                false,
                None,
                &references,
                None,
                None,
                expired,
            ),
            Err(OutputConditionedExecutionRefusal::Deadline)
        );
        assert_eq!(references["target"].lower(), before.lower());
        assert_eq!(references["target"].upper(), before.upper());
    }
}
