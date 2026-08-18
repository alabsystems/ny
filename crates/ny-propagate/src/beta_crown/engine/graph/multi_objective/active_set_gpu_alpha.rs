// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Disconnected planning and selection core for bounded active-set root alpha.
//!
//! This module deliberately performs no graph extraction, gradient replay,
//! optimizer update, GPU call, publication, or root mutation. It establishes
//! the typed invariants that the later execution increment must cross:
//!
//! - K=1 is classified back to the sealed critical-row route without active-set
//!   validation or arithmetic;
//! - K=2..=8 retains every unresolved row and sorts them deterministically by
//!   certified historical slack;
//! - K>8 refuses before inspecting any row payload;
//! - candidate comparison moves one whole certified vector/state pair;
//! - a stable fingerprint binds that state to every row and interval bit; and
//! - exactly one current binding row may be claimed for host replay.

#![allow(dead_code)] // Phase-2 core: root execution is intentionally deferred.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS};
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::AdaptiveOptConfig;
use crate::beta_crown::engine::graph::propagation::batched::wide_alpha_true::{
    true_alpha_grads_for_row_gpu_until, TrueGradGpuReplayOps,
};
use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
use crate::network::{build_resnet_segment_skeleton, SpecCrownRequest};
use crate::GraphNetwork;

use super::critical_gpu_alpha::{
    build_checked_alpha_bridge, deadline_open, fingerprint_bytes, fingerprint_u64,
    project_critical_margin_gradient, step_critical_alpha_candidate, CriticalGpuAlphaSearchPolicy,
    CriticalGpuAlphaStepRefusal,
};

const ACTIVE_SET_PAIR_FINGERPRINT_DOMAIN: &[u8] = b"ny-active-set-certified-pair-v1";
const ACTIVE_SET_FULL_STATE_FINGERPRINT_DOMAIN: &[u8] = b"ny-active-set-full-alpha-state-v1";
const FNV1A64_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActiveSetGpuAlphaRefusal {
    NoUnresolvedRows,
    TooManyUnresolvedRows { count: usize, maximum: usize },
    InvalidHistoricalInterval { source_row_index: usize },
    InvalidThreshold { source_row_index: usize },
    InvalidHistoricalSlack { source_row_index: usize },
    DuplicateSourceRow { source_row_index: usize },
    InvalidCertifiedVector,
    InvalidAlphaState,
    CertifiedPairStateMismatch,
    CertifiedPairScoreMismatch,
    CertifiedPairFingerprintMismatch,
    BindingReplayAlreadyClaimed,
}

impl ActiveSetGpuAlphaRefusal {
    pub(super) fn telemetry_reason(self) -> &'static str {
        match self {
            Self::NoUnresolvedRows => "no_unresolved_rows",
            Self::TooManyUnresolvedRows { .. } => "too_many_unresolved_rows",
            Self::InvalidHistoricalInterval { .. } => "invalid_historical_interval",
            Self::InvalidThreshold { .. } => "invalid_threshold",
            Self::InvalidHistoricalSlack { .. } => "invalid_historical_slack",
            Self::DuplicateSourceRow { .. } => "duplicate_source_row",
            Self::InvalidCertifiedVector => "invalid_certified_vector",
            Self::InvalidAlphaState => "invalid_alpha_state",
            Self::CertifiedPairStateMismatch => "certified_pair_state_mismatch",
            Self::CertifiedPairScoreMismatch => "certified_pair_score_mismatch",
            Self::CertifiedPairFingerprintMismatch => "certified_pair_fingerprint_mismatch",
            Self::BindingReplayAlreadyClaimed => "binding_replay_already_claimed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ActiveSetUnresolvedRow {
    source_row_index: usize,
    historical_lower: f32,
    historical_upper: f32,
    threshold: f32,
}

impl ActiveSetUnresolvedRow {
    pub(super) fn new(
        source_row_index: usize,
        historical_lower: f32,
        historical_upper: f32,
        threshold: f32,
    ) -> Self {
        Self {
            source_row_index,
            historical_lower,
            historical_upper,
            threshold,
        }
    }

    pub(super) fn source_row_index(self) -> usize {
        self.source_row_index
    }

    pub(super) fn historical_lower(self) -> f32 {
        self.historical_lower
    }

    pub(super) fn historical_upper(self) -> f32 {
        self.historical_upper
    }

    pub(super) fn threshold(self) -> f32 {
        self.threshold
    }

    fn historical_slack(self) -> f32 {
        canonical_zero(self.historical_lower - self.threshold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveSetCriticalRowDelegation {
    source_row_index: usize,
}

impl ActiveSetCriticalRowDelegation {
    pub(super) fn source_row_index(self) -> usize {
        self.source_row_index
    }
}

#[derive(Debug)]
pub(super) enum ActiveSetGpuAlphaClassification {
    DelegateSealedCriticalRow(ActiveSetCriticalRowDelegation),
    Optimize(ActiveSetGpuAlphaPlan),
}

#[derive(Debug)]
pub(super) struct ActiveSetGpuAlphaPlan {
    rows: Box<[ActiveSetUnresolvedRow]>,
    binding_replay_claimed: bool,
}

impl ActiveSetGpuAlphaPlan {
    pub(super) fn rows(&self) -> &[ActiveSetUnresolvedRow] {
        &self.rows
    }

    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Claim the sole host replay against one exact certified pair.
    ///
    /// The returned token identifies one row only and carries the whole-pair
    /// fingerprint/state identity that selected it. A second claim refuses.
    pub(super) fn claim_binding_row_replay(
        &mut self,
        pair: &ActiveSetGpuAlphaCertifiedPair,
    ) -> Result<ActiveSetBindingRowReplay, ActiveSetGpuAlphaRefusal> {
        if self.binding_replay_claimed {
            return Err(ActiveSetGpuAlphaRefusal::BindingReplayAlreadyClaimed);
        }
        pair.validate(self)?;
        let lower = pair.lower_vector();
        let mut binding_ordinal = 0usize;
        let mut binding_slack = canonical_zero(lower[0] - self.rows[0].threshold);
        for (ordinal, (&value, row)) in lower.iter().zip(self.rows.iter()).enumerate().skip(1) {
            let slack = canonical_zero(value - row.threshold);
            if slack.total_cmp(&binding_slack) == Ordering::Less {
                binding_ordinal = ordinal;
                binding_slack = slack;
            }
        }
        if binding_slack.to_bits() != pair.score.min_slack.to_bits() {
            return Err(ActiveSetGpuAlphaRefusal::CertifiedPairScoreMismatch);
        }
        self.binding_replay_claimed = true;
        Ok(ActiveSetBindingRowReplay {
            active_ordinal: binding_ordinal,
            source_row_index: self.rows[binding_ordinal].source_row_index,
            certified_lower: lower[binding_ordinal],
            threshold: self.rows[binding_ordinal].threshold,
            state_identity: pair.state_identity,
            certified_pair_fingerprint: pair.fingerprint,
        })
    }
}

/// Classify the complete unresolved root set before any active-set work.
///
/// Length is inspected first. This is load-bearing: K=1 must delegate to the
/// sealed scalar implementation even if its numerical payload is malformed,
/// and K>8 must refuse before validation, sorting, extraction, or replay.
pub(super) fn classify_active_set_gpu_alpha(
    unresolved: &[ActiveSetUnresolvedRow],
) -> Result<ActiveSetGpuAlphaClassification, ActiveSetGpuAlphaRefusal> {
    match unresolved.len() {
        0 => return Err(ActiveSetGpuAlphaRefusal::NoUnresolvedRows),
        1 => {
            return Ok(ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(
                ActiveSetCriticalRowDelegation {
                    source_row_index: unresolved[0].source_row_index,
                },
            ));
        }
        count if count > DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS => {
            return Err(ActiveSetGpuAlphaRefusal::TooManyUnresolvedRows {
                count,
                maximum: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            });
        }
        _ => {}
    }

    let mut source_rows = BTreeSet::new();
    let mut rows = Vec::with_capacity(unresolved.len());
    for &row in unresolved {
        if !row.historical_lower.is_finite()
            || !row.historical_upper.is_finite()
            || row.historical_lower > row.historical_upper
        {
            return Err(ActiveSetGpuAlphaRefusal::InvalidHistoricalInterval {
                source_row_index: row.source_row_index,
            });
        }
        if !row.threshold.is_finite() {
            return Err(ActiveSetGpuAlphaRefusal::InvalidThreshold {
                source_row_index: row.source_row_index,
            });
        }
        if !row.historical_slack().is_finite() {
            return Err(ActiveSetGpuAlphaRefusal::InvalidHistoricalSlack {
                source_row_index: row.source_row_index,
            });
        }
        if !source_rows.insert(row.source_row_index) {
            return Err(ActiveSetGpuAlphaRefusal::DuplicateSourceRow {
                source_row_index: row.source_row_index,
            });
        }
        rows.push(row);
    }
    rows.sort_by(|left, right| {
        left.historical_slack()
            .total_cmp(&right.historical_slack())
            .then_with(|| left.source_row_index.cmp(&right.source_row_index))
    });

    Ok(ActiveSetGpuAlphaClassification::Optimize(
        ActiveSetGpuAlphaPlan {
            rows: rows.into_boxed_slice(),
            binding_replay_claimed: false,
        },
    ))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ActiveSetGpuAlphaScore {
    rows_certified: usize,
    min_slack: f32,
    negative_slack_sum: f32,
}

impl ActiveSetGpuAlphaScore {
    pub(super) fn rows_certified(self) -> usize {
        self.rows_certified
    }

    pub(super) fn min_slack(self) -> f32 {
        self.min_slack
    }

    pub(super) fn negative_slack_sum(self) -> f32 {
        self.negative_slack_sum
    }

    /// Higher is better for all three components.
    pub(super) fn cmp_lexicographic(self, other: Self) -> Ordering {
        self.rows_certified
            .cmp(&other.rows_certified)
            .then_with(|| self.min_slack.total_cmp(&other.min_slack))
            .then_with(|| self.negative_slack_sum.total_cmp(&other.negative_slack_sum))
    }

    fn from_slacks(slacks: &[f32]) -> Result<Self, ActiveSetGpuAlphaRefusal> {
        let Some(&first) = slacks.first() else {
            return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
        };
        if !first.is_finite() {
            return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
        }
        let mut rows_certified = usize::from(first > 0.0);
        let mut min_slack = first;
        let mut negative_slack_sum = first.min(0.0);
        for &slack in &slacks[1..] {
            if !slack.is_finite() {
                return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
            }
            rows_certified += usize::from(slack > 0.0);
            min_slack = min_slack.min(slack);
            negative_slack_sum += slack.min(0.0);
            if !negative_slack_sum.is_finite() {
                return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
            }
        }
        Ok(Self {
            rows_certified,
            min_slack: canonical_zero(min_slack),
            negative_slack_sum: canonical_zero(negative_slack_sum),
        })
    }

    fn bit_eq(self, other: Self) -> bool {
        self.rows_certified == other.rows_certified
            && self.min_slack.to_bits() == other.min_slack.to_bits()
            && self.negative_slack_sum.to_bits() == other.negative_slack_sum.to_bits()
    }
}

/// Stable identity of the complete stored active-set optimizer state.
///
/// This is deliberately separate from the sealed critical route's alpha-only
/// identity. Active-set continuation retains optimizer state, so every raw
/// lower/upper neuron field and the exact two-sided map layout are authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveSetGpuAlphaFullStateIdentity {
    parameter_count: usize,
    fingerprint: u64,
}

impl ActiveSetGpuAlphaFullStateIdentity {
    pub(super) fn parameter_count(self) -> usize {
        self.parameter_count
    }

    pub(super) fn fingerprint(self) -> u64 {
        self.fingerprint
    }
}

pub(super) fn active_set_full_state_identity(
    state: &GraphDomainAlphaState,
) -> Option<ActiveSetGpuAlphaFullStateIdentity> {
    let lower = state.neurons();
    let upper = state.upper_neurons();
    let parameter_count = state.len();
    let upper_parameter_count: usize = upper.values().map(std::collections::HashMap::len).sum();
    if parameter_count == 0
        || upper_parameter_count != parameter_count
        || lower.len() != upper.len()
    {
        return None;
    }

    let mut lower_node_names: Vec<_> = lower.keys().map(String::as_str).collect();
    let mut upper_node_names: Vec<_> = upper.keys().map(String::as_str).collect();
    lower_node_names.sort_unstable();
    upper_node_names.sort_unstable();
    if lower_node_names != upper_node_names {
        return None;
    }

    let mut hash = FNV1A64_OFFSET_BASIS;
    fingerprint_bytes(&mut hash, ACTIVE_SET_FULL_STATE_FINGERPRINT_DOMAIN);
    fingerprint_u64(&mut hash, lower_node_names.len() as u64);
    fingerprint_u64(&mut hash, parameter_count as u64);
    for node_name in lower_node_names {
        let lower_neurons = lower.get(node_name)?;
        let upper_neurons = upper.get(node_name)?;
        if lower_neurons.len() != upper_neurons.len() {
            return None;
        }
        let mut lower_indices: Vec<_> = lower_neurons.keys().copied().collect();
        let mut upper_indices: Vec<_> = upper_neurons.keys().copied().collect();
        lower_indices.sort_unstable();
        upper_indices.sort_unstable();
        if lower_indices != upper_indices {
            return None;
        }

        fingerprint_u64(&mut hash, node_name.len() as u64);
        fingerprint_bytes(&mut hash, node_name.as_bytes());
        fingerprint_u64(&mut hash, lower_indices.len() as u64);
        for neuron_idx in lower_indices {
            fingerprint_u64(&mut hash, neuron_idx as u64);
            // Side tags make lower/upper role swaps observable even when every
            // other layout bit matches.
            fingerprint_u64(&mut hash, 0);
            fingerprint_full_neuron(&mut hash, lower_neurons.get(&neuron_idx)?)?;
            fingerprint_u64(&mut hash, 1);
            fingerprint_full_neuron(&mut hash, upper_neurons.get(&neuron_idx)?)?;
        }
    }
    Some(ActiveSetGpuAlphaFullStateIdentity {
        parameter_count,
        fingerprint: hash,
    })
}

fn fingerprint_full_neuron(hash: &mut u64, neuron: &AlphaNeuronState) -> Option<()> {
    // Read raw stored alpha rather than `alpha()`: the accessor sanitizes, which
    // would collapse distinct corrupt stored states onto the same identity.
    let alpha = neuron.alpha;
    let fields = [
        alpha,
        neuron.grad,
        neuron.velocity,
        neuron.adam_m,
        neuron.adam_v,
        neuron.adam_v_max,
    ];
    if !alpha.is_finite()
        || !(0.0..=1.0).contains(&alpha)
        || fields.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    for value in fields {
        fingerprint_u64(hash, u64::from(value.to_bits()));
    }
    Some(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveSetCertifiedPairFingerprint(u64);

impl ActiveSetCertifiedPairFingerprint {
    pub(super) fn value(self) -> u64 {
        self.0
    }
}

/// One indivisible direct-C active-vector/state snapshot.
///
/// Fields remain private so selection cannot transplant an interval row onto a
/// different alpha state. Construction and validation bind all interval bits,
/// sorted source rows, thresholds, and the complete stored optimizer identity.
#[derive(Debug)]
pub(super) struct ActiveSetGpuAlphaCertifiedPair {
    bounds: BoundedTensor,
    state: GraphDomainAlphaState,
    state_identity: ActiveSetGpuAlphaFullStateIdentity,
    score: ActiveSetGpuAlphaScore,
    fingerprint: ActiveSetCertifiedPairFingerprint,
}

impl ActiveSetGpuAlphaCertifiedPair {
    pub(super) fn new(
        plan: &ActiveSetGpuAlphaPlan,
        bounds: BoundedTensor,
        state: GraphDomainAlphaState,
    ) -> Result<Self, ActiveSetGpuAlphaRefusal> {
        let state_identity = active_set_full_state_identity(&state)
            .ok_or(ActiveSetGpuAlphaRefusal::InvalidAlphaState)?;
        let (lower, upper) = finite_ordered_vector(&bounds, plan.len())?;
        let score = score_for_plan(plan, &lower)?;
        let fingerprint = certified_pair_fingerprint(plan, &lower, &upper, state_identity);
        Ok(Self {
            bounds,
            state,
            state_identity,
            score,
            fingerprint,
        })
    }

    pub(super) fn bounds(&self) -> &BoundedTensor {
        &self.bounds
    }

    pub(super) fn state(&self) -> &GraphDomainAlphaState {
        &self.state
    }

    pub(super) fn state_identity(&self) -> ActiveSetGpuAlphaFullStateIdentity {
        self.state_identity
    }

    pub(super) fn score(&self) -> ActiveSetGpuAlphaScore {
        self.score
    }

    pub(super) fn fingerprint(&self) -> ActiveSetCertifiedPairFingerprint {
        self.fingerprint
    }

    pub(super) fn into_bound_state_pair(self) -> (BoundedTensor, GraphDomainAlphaState) {
        (self.bounds, self.state)
    }

    pub(super) fn validate(
        &self,
        plan: &ActiveSetGpuAlphaPlan,
    ) -> Result<(), ActiveSetGpuAlphaRefusal> {
        let current_identity = active_set_full_state_identity(&self.state)
            .ok_or(ActiveSetGpuAlphaRefusal::CertifiedPairStateMismatch)?;
        if current_identity != self.state_identity {
            return Err(ActiveSetGpuAlphaRefusal::CertifiedPairStateMismatch);
        }
        let (lower, upper) = finite_ordered_vector(&self.bounds, plan.len())?;
        let current_score = score_for_plan(plan, &lower)?;
        if !current_score.bit_eq(self.score) {
            return Err(ActiveSetGpuAlphaRefusal::CertifiedPairScoreMismatch);
        }
        let current_fingerprint =
            certified_pair_fingerprint(plan, &lower, &upper, current_identity);
        if current_fingerprint != self.fingerprint {
            return Err(ActiveSetGpuAlphaRefusal::CertifiedPairFingerprintMismatch);
        }
        Ok(())
    }

    fn lower_vector(&self) -> Vec<f32> {
        self.bounds.lower().iter().copied().collect()
    }
}

/// Retain only a complete pair under the charter's lexicographic score.
///
/// Strict comparison preserves the earlier pair on a complete score tie.
pub(super) fn retain_lexicographic_best_pair(
    plan: &ActiveSetGpuAlphaPlan,
    best: &mut Option<ActiveSetGpuAlphaCertifiedPair>,
    candidate: ActiveSetGpuAlphaCertifiedPair,
) -> Result<bool, ActiveSetGpuAlphaRefusal> {
    candidate.validate(plan)?;
    let replace = if let Some(current) = best.as_ref() {
        current.validate(plan)?;
        candidate.score.cmp_lexicographic(current.score) == Ordering::Greater
    } else {
        true
    };
    if replace {
        *best = Some(candidate);
    }
    Ok(replace)
}

/// Linear-use token for the sole binding-row host replay.
///
/// It is intentionally neither `Copy` nor `Clone`; the execution increment
/// must consume this contract when it performs the one replay.
#[derive(Debug, PartialEq)]
pub(super) struct ActiveSetBindingRowReplay {
    active_ordinal: usize,
    source_row_index: usize,
    certified_lower: f32,
    threshold: f32,
    state_identity: ActiveSetGpuAlphaFullStateIdentity,
    certified_pair_fingerprint: ActiveSetCertifiedPairFingerprint,
}

impl ActiveSetBindingRowReplay {
    pub(super) fn active_ordinal(&self) -> usize {
        self.active_ordinal
    }

    pub(super) fn source_row_index(&self) -> usize {
        self.source_row_index
    }

    pub(super) fn certified_lower(&self) -> f32 {
        self.certified_lower
    }

    pub(super) fn threshold(&self) -> f32 {
        self.threshold
    }

    pub(super) fn state_identity(&self) -> ActiveSetGpuAlphaFullStateIdentity {
        self.state_identity
    }

    pub(super) fn certified_pair_fingerprint(&self) -> ActiveSetCertifiedPairFingerprint {
        self.certified_pair_fingerprint
    }
}

fn finite_ordered_vector(
    bounds: &BoundedTensor,
    expected_rows: usize,
) -> Result<(Vec<f32>, Vec<f32>), ActiveSetGpuAlphaRefusal> {
    if bounds.shape() != [expected_rows] {
        return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
    }
    let lower: Vec<f32> = bounds.lower().iter().copied().collect();
    let upper: Vec<f32> = bounds.upper().iter().copied().collect();
    if lower.len() != expected_rows
        || upper.len() != expected_rows
        || lower
            .iter()
            .zip(&upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    {
        return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
    }
    Ok((lower, upper))
}

fn score_for_plan(
    plan: &ActiveSetGpuAlphaPlan,
    lower: &[f32],
) -> Result<ActiveSetGpuAlphaScore, ActiveSetGpuAlphaRefusal> {
    if lower.len() != plan.len() {
        return Err(ActiveSetGpuAlphaRefusal::InvalidCertifiedVector);
    }
    let slacks: Vec<f32> = lower
        .iter()
        .zip(plan.rows.iter())
        .map(|(&value, row)| canonical_zero(value - row.threshold))
        .collect();
    ActiveSetGpuAlphaScore::from_slacks(&slacks)
}

fn certified_pair_fingerprint(
    plan: &ActiveSetGpuAlphaPlan,
    lower: &[f32],
    upper: &[f32],
    state_identity: ActiveSetGpuAlphaFullStateIdentity,
) -> ActiveSetCertifiedPairFingerprint {
    let mut hash = FNV1A64_OFFSET_BASIS;
    fingerprint_bytes(&mut hash, ACTIVE_SET_PAIR_FINGERPRINT_DOMAIN);
    fingerprint_u64(&mut hash, plan.len() as u64);
    fingerprint_u64(&mut hash, state_identity.parameter_count as u64);
    fingerprint_u64(&mut hash, state_identity.fingerprint);
    for ((row, &lo), &hi) in plan.rows.iter().zip(lower).zip(upper) {
        fingerprint_u64(&mut hash, row.source_row_index as u64);
        fingerprint_u64(&mut hash, u64::from(row.historical_lower.to_bits()));
        fingerprint_u64(&mut hash, u64::from(row.historical_upper.to_bits()));
        fingerprint_u64(&mut hash, u64::from(row.threshold.to_bits()));
        fingerprint_u64(&mut hash, u64::from(lo.to_bits()));
        fingerprint_u64(&mut hash, u64::from(hi.to_bits()));
    }
    ActiveSetCertifiedPairFingerprint(hash)
}

#[inline]
fn canonical_zero(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActiveSetGpuAlphaExecutionRefusal {
    Core(ActiveSetGpuAlphaRefusal),
    Step(CriticalGpuAlphaStepRefusal),
    InvalidSpecMatrix,
    NoSoundGpuRoute,
    OutputContractUnavailable,
    SkeletonUnavailable,
    InitialDirectUnavailable,
    InitialDirectError,
    CandidateDirectUnavailable,
    CandidateDirectError,
    ReplayPairMismatch,
    IncompleteCandidateBracket,
}

impl ActiveSetGpuAlphaExecutionRefusal {
    pub(super) fn telemetry_reason(self) -> &'static str {
        match self {
            Self::Core(reason) => reason.telemetry_reason(),
            Self::Step(reason) => reason.telemetry_reason(),
            Self::InvalidSpecMatrix => "invalid_spec_matrix",
            Self::NoSoundGpuRoute => "no_sound_gpu_route",
            Self::OutputContractUnavailable => "output_contract_unavailable",
            Self::SkeletonUnavailable => "skeleton_unavailable",
            Self::InitialDirectUnavailable => "initial_direct_unavailable",
            Self::InitialDirectError => "initial_direct_error",
            Self::CandidateDirectUnavailable => "candidate_direct_unavailable",
            Self::CandidateDirectError => "candidate_direct_error",
            Self::ReplayPairMismatch => "replay_pair_mismatch",
            Self::IncompleteCandidateBracket => "incomplete_candidate_bracket",
        }
    }
}

impl From<ActiveSetGpuAlphaRefusal> for ActiveSetGpuAlphaExecutionRefusal {
    fn from(value: ActiveSetGpuAlphaRefusal) -> Self {
        Self::Core(value)
    }
}

impl From<CriticalGpuAlphaStepRefusal> for ActiveSetGpuAlphaExecutionRefusal {
    fn from(value: CriticalGpuAlphaStepRefusal) -> Self {
        Self::Step(value)
    }
}

fn classify_active_set_direct_error(
    error: &NyError,
    ordinary: ActiveSetGpuAlphaExecutionRefusal,
) -> ActiveSetGpuAlphaExecutionRefusal {
    if error.is_deadline_exceeded() {
        CriticalGpuAlphaStepRefusal::DeadlineExpired.into()
    } else {
        ordinary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActiveSetGpuAlphaSelectedCandidate {
    Initial,
    Candidate { ordinal: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ActiveSetGpuAlphaCandidateTrace {
    ordinal: usize,
    alpha_lr: f32,
    score: ActiveSetGpuAlphaScore,
    state_identity: ActiveSetGpuAlphaFullStateIdentity,
    pair_fingerprint: ActiveSetCertifiedPairFingerprint,
}

impl ActiveSetGpuAlphaCandidateTrace {
    pub(super) fn ordinal(self) -> usize {
        self.ordinal
    }

    pub(super) fn alpha_lr(self) -> f32 {
        self.alpha_lr
    }

    pub(super) fn score(self) -> ActiveSetGpuAlphaScore {
        self.score
    }

    pub(super) fn state_identity(self) -> ActiveSetGpuAlphaFullStateIdentity {
        self.state_identity
    }

    pub(super) fn pair_fingerprint(self) -> ActiveSetCertifiedPairFingerprint {
        self.pair_fingerprint
    }
}

#[derive(Debug)]
pub(super) struct ActiveSetGpuAlphaExecutionOutput {
    selected_pair: ActiveSetGpuAlphaCertifiedPair,
    initial_score: ActiveSetGpuAlphaScore,
    initial_state_identity: ActiveSetGpuAlphaFullStateIdentity,
    initial_pair_fingerprint: ActiveSetCertifiedPairFingerprint,
    candidate_traces: Vec<ActiveSetGpuAlphaCandidateTrace>,
    selected: ActiveSetGpuAlphaSelectedCandidate,
    base_lr: f32,
    gradient_replays: usize,
}

impl ActiveSetGpuAlphaExecutionOutput {
    pub(super) fn selected_pair(&self) -> &ActiveSetGpuAlphaCertifiedPair {
        &self.selected_pair
    }

    pub(super) fn candidate_traces(&self) -> &[ActiveSetGpuAlphaCandidateTrace] {
        &self.candidate_traces
    }

    pub(super) fn selected(&self) -> ActiveSetGpuAlphaSelectedCandidate {
        self.selected
    }

    pub(super) fn base_lr(&self) -> f32 {
        self.base_lr
    }

    pub(super) fn gradient_replays(&self) -> usize {
        self.gradient_replays
    }

    pub(super) fn into_selected_pair(self) -> ActiveSetGpuAlphaCertifiedPair {
        self.selected_pair
    }

    pub(super) fn validate(
        &self,
        plan: &ActiveSetGpuAlphaPlan,
    ) -> Result<(), ActiveSetGpuAlphaExecutionRefusal> {
        self.selected_pair.validate(plan)?;
        if self.gradient_replays != 1
            || self.candidate_traces.len() != 3
            || !self.base_lr.is_finite()
            || self.base_lr <= 0.0
        {
            return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
        }
        let expected_lrs = [0.3_f32, 1.0, 2.0].map(|scale| self.base_lr * scale);
        for (ordinal, (trace, expected_lr)) in
            self.candidate_traces.iter().zip(expected_lrs).enumerate()
        {
            if trace.ordinal != ordinal
                || trace.alpha_lr.to_bits() != expected_lr.to_bits()
                || !trace.alpha_lr.is_finite()
                || trace.alpha_lr <= 0.0
            {
                return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
            }
        }
        let selected_matches = match self.selected {
            ActiveSetGpuAlphaSelectedCandidate::Initial => {
                self.selected_pair.score.bit_eq(self.initial_score)
                    && self.selected_pair.state_identity == self.initial_state_identity
                    && self.selected_pair.fingerprint == self.initial_pair_fingerprint
            }
            ActiveSetGpuAlphaSelectedCandidate::Candidate { ordinal } => {
                self.candidate_traces.get(ordinal).is_some_and(|trace| {
                    self.selected_pair.score.bit_eq(trace.score)
                        && self.selected_pair.state_identity == trace.state_identity
                        && self.selected_pair.fingerprint == trace.pair_fingerprint
                })
            }
        };
        selected_matches
            .then_some(())
            .ok_or(ActiveSetGpuAlphaExecutionRefusal::ReplayPairMismatch)
    }
}

#[cfg(test)]
pub(super) fn complete_initial_execution_output_for_test(
    selected_pair: ActiveSetGpuAlphaCertifiedPair,
    base_lr: f32,
) -> ActiveSetGpuAlphaExecutionOutput {
    let initial_score = selected_pair.score();
    let initial_state_identity = selected_pair.state_identity();
    let initial_pair_fingerprint = selected_pair.fingerprint();
    let candidate_traces = [0.3_f32, 1.0, 2.0]
        .into_iter()
        .enumerate()
        .map(|(ordinal, scale)| ActiveSetGpuAlphaCandidateTrace {
            ordinal,
            alpha_lr: base_lr * scale,
            score: initial_score,
            state_identity: initial_state_identity,
            pair_fingerprint: initial_pair_fingerprint,
        })
        .collect();
    ActiveSetGpuAlphaExecutionOutput {
        selected_pair,
        initial_score,
        initial_state_identity,
        initial_pair_fingerprint,
        candidate_traces,
        selected: ActiveSetGpuAlphaSelectedCandidate::Initial,
        base_lr,
        gradient_replays: 1,
    }
}

fn active_spec_matrix(
    plan: &ActiveSetGpuAlphaPlan,
    full_spec_matrix: &ndarray::Array2<f32>,
) -> Result<ndarray::Array2<f32>, ActiveSetGpuAlphaExecutionRefusal> {
    if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&plan.len())
        || full_spec_matrix.ncols() == 0
        || full_spec_matrix.iter().any(|value| !value.is_finite())
    {
        return Err(ActiveSetGpuAlphaExecutionRefusal::InvalidSpecMatrix);
    }
    let mut active = ndarray::Array2::zeros((plan.len(), full_spec_matrix.ncols()));
    for (active_ordinal, row) in plan.rows().iter().enumerate() {
        let source = row.source_row_index();
        if source >= full_spec_matrix.nrows() {
            return Err(ActiveSetGpuAlphaExecutionRefusal::InvalidSpecMatrix);
        }
        active
            .row_mut(active_ordinal)
            .assign(&full_spec_matrix.row(source));
    }
    Ok(active)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_active_set_gpu_alpha_lr_bracket(
    plan: &mut ActiveSetGpuAlphaPlan,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    engine: &dyn GemmEngine,
    full_spec_matrix: &ndarray::Array2<f32>,
    hard_deadline: Instant,
    initial_state: &GraphDomainAlphaState,
    adaptive_config: &AdaptiveOptConfig,
    base_lr: f32,
) -> Result<ActiveSetGpuAlphaExecutionOutput, ActiveSetGpuAlphaExecutionRefusal> {
    // Policy and K/capacity checks precede all extraction and GPU work.
    let policy = CriticalGpuAlphaSearchPolicy::new(base_lr, hard_deadline)?;
    let work_deadline = policy.work_deadline;
    let spec_matrix = active_spec_matrix(plan, full_spec_matrix)?;
    let Some(gpu) = engine
        .as_gpu_crown_backward()
        .filter(|candidate| candidate.provides_sound_gpu_crown())
    else {
        return Err(ActiveSetGpuAlphaExecutionRefusal::NoSoundGpuRoute);
    };
    let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
    if capacity > DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS || capacity < plan.len() {
        return Err(ActiveSetGpuAlphaExecutionRefusal::NoSoundGpuRoute);
    }
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired.into());
    }

    let (initial_bridge, _initial_round_trip, _initial_alpha_identity) =
        build_checked_alpha_bridge(
            graph,
            input,
            node_bounds,
            initial_state,
            CriticalGpuAlphaStepRefusal::InvalidInitialState,
        )?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired.into());
    }
    let exec_order = graph
        .exec_order()
        .map_err(|_| ActiveSetGpuAlphaExecutionRefusal::OutputContractUnavailable)?;
    let output_name = if graph.output_name().is_empty() {
        exec_order
            .last()
            .map(String::as_str)
            .ok_or(ActiveSetGpuAlphaExecutionRefusal::OutputContractUnavailable)?
    } else {
        graph.output_name()
    };
    let skeleton = build_resnet_segment_skeleton(
        graph,
        input,
        output_name,
        node_bounds,
        node_bounds,
        Some(&initial_bridge),
        false,
    )
    .ok_or(ActiveSetGpuAlphaExecutionRefusal::SkeletonUnavailable)?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired.into());
    }

    let initial_result = match SpecCrownRequest::new(graph, input, &spec_matrix, Some(engine))
        .node_bounds(node_bounds)
        .alpha_state_opt(Some(&initial_bridge))
        .deadline_opt(Some(work_deadline))
        .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
    {
        Ok(Some(result)) => result,
        Ok(None) => return Err(ActiveSetGpuAlphaExecutionRefusal::InitialDirectUnavailable),
        Err(error) => {
            return Err(classify_active_set_direct_error(
                &error,
                ActiveSetGpuAlphaExecutionRefusal::InitialDirectError,
            ));
        }
    };
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired.into());
    }
    let initial_pair =
        ActiveSetGpuAlphaCertifiedPair::new(plan, initial_result.bounds, initial_state.clone())?;
    let replay = plan.claim_binding_row_replay(&initial_pair)?;
    if replay.state_identity() != initial_pair.state_identity()
        || replay.certified_pair_fingerprint() != initial_pair.fingerprint()
    {
        return Err(ActiveSetGpuAlphaExecutionRefusal::ReplayPairMismatch);
    }
    let replay_spec_view = spec_matrix.row(replay.active_ordinal());
    let replay_spec = replay_spec_view
        .as_slice()
        .ok_or(ActiveSetGpuAlphaExecutionRefusal::InvalidSpecMatrix)?;
    // This is the sole true replay. The linear-use token above binds it to
    // the exact initial K-vector/state and to the folded CUDA operands returned
    // by that same evaluation.
    //
    // #true-grad-gpu-replay: the replay's backward walk runs on the armed sound
    // GPU lane when routable. The bounded-rows result carries no certified-error
    // concretization tables, so the operands use the trait-sanctioned EMPTY
    // tables (pre-concretization path); a non-finite/exploded device payload is
    // rejected inside and the seam falls closed to the same CPU implementation
    // under the remaining absolute deadline — as does every other refusal.
    let gpu_replay_ops = TrueGradGpuReplayOps::new(gpu, &[], &[]);
    let gradients = true_alpha_grads_for_row_gpu_until(
        gpu_replay_ops.as_ref(),
        &initial_result.segments,
        replay_spec,
        &[],
        &initial_result.input_lower,
        &initial_result.input_upper,
        initial_result.relu_names.len(),
        replay.certified_lower(),
        false,
        Some(work_deadline),
    )
    .ok_or(CriticalGpuAlphaStepRefusal::HostReplayUnavailable)?;
    if !deadline_open(work_deadline) {
        return Err(CriticalGpuAlphaStepRefusal::DeadlineExpired.into());
    }
    let projected_state = project_critical_margin_gradient(
        initial_pair.state(),
        &initial_result.relu_names,
        &gradients,
        work_deadline,
    )?;

    let initial_score = initial_pair.score();
    let initial_state_identity = initial_pair.state_identity();
    let initial_pair_fingerprint = initial_pair.fingerprint();
    let mut selected = ActiveSetGpuAlphaSelectedCandidate::Initial;
    let mut best = None;
    retain_lexicographic_best_pair(plan, &mut best, initial_pair)?;
    let mut candidate_traces = Vec::with_capacity(policy.candidate_lrs.len());
    for (ordinal, &alpha_lr) in policy.candidate_lrs.iter().enumerate() {
        if !deadline_open(work_deadline) {
            return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
        }
        let candidate_state = step_critical_alpha_candidate(
            &projected_state,
            adaptive_config,
            alpha_lr,
            work_deadline,
        )?;
        let (candidate_bridge, _round_trip, _alpha_identity) = build_checked_alpha_bridge(
            graph,
            input,
            node_bounds,
            &candidate_state,
            CriticalGpuAlphaStepRefusal::InvalidOptimizedState,
        )?;
        if !deadline_open(work_deadline) {
            return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
        }
        let candidate_result = match SpecCrownRequest::new(graph, input, &spec_matrix, Some(engine))
            .node_bounds(node_bounds)
            .alpha_state_opt(Some(&candidate_bridge))
            .deadline_opt(Some(work_deadline))
            .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
        {
            Ok(Some(result)) => result,
            Ok(None) => {
                return Err(ActiveSetGpuAlphaExecutionRefusal::CandidateDirectUnavailable);
            }
            Err(error) => {
                return Err(classify_active_set_direct_error(
                    &error,
                    ActiveSetGpuAlphaExecutionRefusal::CandidateDirectError,
                ));
            }
        };
        if !deadline_open(work_deadline) {
            return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
        }
        let candidate =
            ActiveSetGpuAlphaCertifiedPair::new(plan, candidate_result.bounds, candidate_state)?;
        let trace = ActiveSetGpuAlphaCandidateTrace {
            ordinal,
            alpha_lr,
            score: candidate.score(),
            state_identity: candidate.state_identity(),
            pair_fingerprint: candidate.fingerprint(),
        };
        if retain_lexicographic_best_pair(plan, &mut best, candidate)? {
            selected = ActiveSetGpuAlphaSelectedCandidate::Candidate { ordinal };
        }
        candidate_traces.push(trace);
    }
    if candidate_traces.len() != policy.candidate_lrs.len() || !deadline_open(policy.hard_deadline)
    {
        return Err(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket);
    }
    let output = ActiveSetGpuAlphaExecutionOutput {
        selected_pair: best.ok_or(ActiveSetGpuAlphaExecutionRefusal::IncompleteCandidateBracket)?,
        initial_score,
        initial_state_identity,
        initial_pair_fingerprint,
        candidate_traces,
        selected,
        base_lr: policy.base_lr,
        gradient_replays: 1,
    };
    output.validate(plan)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::state::AlphaNeuronState;
    use ndarray::{ArrayD, IxDyn};

    #[derive(Debug, Clone, Copy)]
    enum TestStateSide {
        Lower,
        Upper,
    }

    #[derive(Debug, Clone, Copy)]
    enum TestStateField {
        Alpha,
        Grad,
        Velocity,
        AdamM,
        AdamV,
        AdamVMax,
    }

    #[test]
    fn direct_error_classifier_preserves_deadline_authority() {
        assert_eq!(
            classify_active_set_direct_error(
                &NyError::DeadlineExceeded("test deadline".into()),
                ActiveSetGpuAlphaExecutionRefusal::InitialDirectError,
            ),
            ActiveSetGpuAlphaExecutionRefusal::Step(CriticalGpuAlphaStepRefusal::DeadlineExpired)
        );
        assert_eq!(
            classify_active_set_direct_error(
                &NyError::InvalidSpec("test failure".into()),
                ActiveSetGpuAlphaExecutionRefusal::CandidateDirectError,
            ),
            ActiveSetGpuAlphaExecutionRefusal::CandidateDirectError
        );
    }

    const TEST_STATE_SIDES: [TestStateSide; 2] = [TestStateSide::Lower, TestStateSide::Upper];
    const TEST_STATE_FIELDS: [TestStateField; 6] = [
        TestStateField::Alpha,
        TestStateField::Grad,
        TestStateField::Velocity,
        TestStateField::AdamM,
        TestStateField::AdamV,
        TestStateField::AdamVMax,
    ];

    fn unresolved(source_row_index: usize, lower: f32) -> ActiveSetUnresolvedRow {
        ActiveSetUnresolvedRow::new(source_row_index, lower, lower + 1.0, 0.0)
    }

    fn active_plan(rows: &[ActiveSetUnresolvedRow]) -> ActiveSetGpuAlphaPlan {
        match classify_active_set_gpu_alpha(rows).expect("active-set classification") {
            ActiveSetGpuAlphaClassification::Optimize(plan) => plan,
            ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(_) => {
                panic!("test expected K>=2 active plan")
            }
        }
    }

    fn alpha_state(alpha: f32) -> GraphDomainAlphaState {
        let mut state = GraphDomainAlphaState::empty();
        state.insert("relu0".into(), 0, AlphaNeuronState::new(alpha));
        state
    }

    fn test_neuron_mut(
        state: &mut GraphDomainAlphaState,
        side: TestStateSide,
    ) -> &mut AlphaNeuronState {
        let maps = match side {
            TestStateSide::Lower => state.neurons_mut(),
            TestStateSide::Upper => state.upper_neurons_mut(),
        };
        maps.get_mut("relu0")
            .and_then(|neurons| neurons.get_mut(&0))
            .expect("test neuron")
    }

    fn set_raw_test_field(
        state: &mut GraphDomainAlphaState,
        side: TestStateSide,
        field: TestStateField,
        value: f32,
    ) {
        let neuron = test_neuron_mut(state, side);
        match field {
            TestStateField::Alpha => neuron.alpha = value,
            TestStateField::Grad => neuron.grad = value,
            TestStateField::Velocity => neuron.velocity = value,
            TestStateField::AdamM => neuron.adam_m = value,
            TestStateField::AdamV => neuron.adam_v = value,
            TestStateField::AdamVMax => neuron.adam_v_max = value,
        }
    }

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower shape"),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper shape"),
        )
        .expect("ordered finite test bounds")
    }

    fn pair(
        plan: &ActiveSetGpuAlphaPlan,
        lower: &[f32],
        upper: &[f32],
        alpha: f32,
    ) -> ActiveSetGpuAlphaCertifiedPair {
        ActiveSetGpuAlphaCertifiedPair::new(plan, bounds(lower, upper), alpha_state(alpha))
            .expect("valid test pair")
    }

    #[test]
    fn k1_delegates_without_active_set_validation_and_k_over_cap_refuses_first() {
        let invalid_scalar = ActiveSetUnresolvedRow::new(17, f32::NAN, f32::NEG_INFINITY, f32::NAN);
        match classify_active_set_gpu_alpha(&[invalid_scalar]).expect("K=1 delegates untouched") {
            ActiveSetGpuAlphaClassification::DelegateSealedCriticalRow(delegation) => {
                assert_eq!(delegation.source_row_index(), 17);
            }
            ActiveSetGpuAlphaClassification::Optimize(_) => panic!("K=1 entered active-set work"),
        }

        let invalid_wide = vec![invalid_scalar; DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1];
        assert_eq!(
            classify_active_set_gpu_alpha(&invalid_wide).unwrap_err(),
            ActiveSetGpuAlphaRefusal::TooManyUnresolvedRows {
                count: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
                maximum: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            },
            "capacity refusal must win before malformed-row or duplicate-row inspection"
        );
        assert_eq!(
            classify_active_set_gpu_alpha(&[]).unwrap_err(),
            ActiveSetGpuAlphaRefusal::NoUnresolvedRows
        );
    }

    #[test]
    fn k2_through_k8_keep_every_unresolved_row_sorted_by_slack_then_source() {
        for count in 2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS {
            let input: Vec<_> = (0..count)
                .rev()
                .map(|row| unresolved(100 + row, row as f32 - 4.0))
                .collect();
            let plan = active_plan(&input);
            assert_eq!(plan.len(), count);
            assert!(!plan.is_empty());
            let observed: Vec<_> = plan
                .rows()
                .iter()
                .map(|row| row.source_row_index())
                .collect();
            let expected: Vec<_> = (0..count).map(|row| 100 + row).collect();
            assert_eq!(observed, expected, "K={count} lost or reordered a row");
        }

        let tied = active_plan(&[unresolved(9, -1.0), unresolved(3, -1.0)]);
        assert_eq!(
            tied.rows()
                .iter()
                .map(|row| row.source_row_index())
                .collect::<Vec<_>>(),
            vec![3, 9]
        );
    }

    #[test]
    fn active_plan_rejects_invalid_or_duplicate_rows() {
        let cases = [
            (
                vec![
                    ActiveSetUnresolvedRow::new(0, f32::NAN, 1.0, 0.0),
                    unresolved(1, -1.0),
                ],
                ActiveSetGpuAlphaRefusal::InvalidHistoricalInterval {
                    source_row_index: 0,
                },
            ),
            (
                vec![
                    ActiveSetUnresolvedRow::new(0, 2.0, 1.0, 0.0),
                    unresolved(1, -1.0),
                ],
                ActiveSetGpuAlphaRefusal::InvalidHistoricalInterval {
                    source_row_index: 0,
                },
            ),
            (
                vec![
                    ActiveSetUnresolvedRow::new(0, -1.0, 1.0, f32::INFINITY),
                    unresolved(1, -1.0),
                ],
                ActiveSetGpuAlphaRefusal::InvalidThreshold {
                    source_row_index: 0,
                },
            ),
            (
                vec![
                    ActiveSetUnresolvedRow::new(0, f32::MAX, f32::MAX, -f32::MAX),
                    unresolved(1, -1.0),
                ],
                ActiveSetGpuAlphaRefusal::InvalidHistoricalSlack {
                    source_row_index: 0,
                },
            ),
            (
                vec![unresolved(4, -2.0), unresolved(4, -1.0)],
                ActiveSetGpuAlphaRefusal::DuplicateSourceRow {
                    source_row_index: 4,
                },
            ),
        ];
        for (rows, expected) in cases {
            assert_eq!(classify_active_set_gpu_alpha(&rows).unwrap_err(), expected);
        }
    }

    #[test]
    fn score_order_is_exactly_rows_then_min_then_negative_sum() {
        let more_certified =
            ActiveSetGpuAlphaScore::from_slacks(&[-100.0, 0.1, 0.2]).expect("finite score");
        let fewer_certified =
            ActiveSetGpuAlphaScore::from_slacks(&[-0.1, -0.1, 1.0]).expect("finite score");
        assert_eq!(
            more_certified.cmp_lexicographic(fewer_certified),
            Ordering::Greater
        );

        let tighter_min =
            ActiveSetGpuAlphaScore::from_slacks(&[-1.5, -1.4, 1.0]).expect("finite score");
        let looser_min =
            ActiveSetGpuAlphaScore::from_slacks(&[-2.0, -0.1, 1.0]).expect("finite score");
        assert_eq!(tighter_min.cmp_lexicographic(looser_min), Ordering::Greater);

        let better_sum =
            ActiveSetGpuAlphaScore::from_slacks(&[-2.0, -0.5, 1.0]).expect("finite score");
        let worse_sum =
            ActiveSetGpuAlphaScore::from_slacks(&[-2.0, -1.0, 1.0]).expect("finite score");
        assert_eq!(better_sum.cmp_lexicographic(worse_sum), Ordering::Greater);
        assert_eq!(better_sum.rows_certified(), 1);
        assert_eq!(better_sum.min_slack().to_bits(), (-2.0_f32).to_bits());
        assert_eq!(
            better_sum.negative_slack_sum().to_bits(),
            (-2.5_f32).to_bits()
        );
    }

    #[test]
    fn rows_certified_requires_strict_positive_slack_at_ieee_boundaries() {
        let equality = ActiveSetGpuAlphaScore::from_slacks(&[0.0]).expect("zero slack");
        let negative_zero =
            ActiveSetGpuAlphaScore::from_slacks(&[-0.0]).expect("negative zero slack");
        let smallest_positive = f32::from_bits(1);
        let positive =
            ActiveSetGpuAlphaScore::from_slacks(&[smallest_positive]).expect("positive subnormal");
        let smallest_negative = f32::from_bits(0x8000_0001);
        let negative =
            ActiveSetGpuAlphaScore::from_slacks(&[smallest_negative]).expect("negative subnormal");

        assert_eq!(equality.rows_certified(), 0);
        assert_eq!(negative_zero.rows_certified(), 0);
        assert_eq!(positive.rows_certified(), 1);
        assert_eq!(positive.min_slack().to_bits(), smallest_positive.to_bits());
        assert_eq!(negative.rows_certified(), 0);
        assert_eq!(negative.min_slack().to_bits(), smallest_negative.to_bits());
    }

    #[test]
    fn full_state_identity_covers_every_raw_field_on_both_sides_and_rejects_nonfinite() {
        let baseline = alpha_state(0.2);
        let baseline_identity =
            active_set_full_state_identity(&baseline).expect("baseline full identity");
        assert_eq!(baseline_identity.parameter_count(), 1);
        assert_ne!(baseline_identity.fingerprint(), 0);

        for side in TEST_STATE_SIDES {
            for field in TEST_STATE_FIELDS {
                let finite_value = match field {
                    TestStateField::Alpha => 0.3,
                    TestStateField::Grad => 1.0,
                    TestStateField::Velocity => 2.0,
                    TestStateField::AdamM => 3.0,
                    TestStateField::AdamV => 4.0,
                    TestStateField::AdamVMax => 5.0,
                };
                let mut changed = baseline.clone();
                set_raw_test_field(&mut changed, side, field, finite_value);
                assert_ne!(
                    active_set_full_state_identity(&changed),
                    Some(baseline_identity),
                    "finite {side:?} {field:?} drift was not fingerprinted"
                );

                let mut nonfinite = baseline.clone();
                set_raw_test_field(&mut nonfinite, side, field, f32::NAN);
                assert_eq!(
                    active_set_full_state_identity(&nonfinite),
                    None,
                    "non-finite {side:?} {field:?} was admitted"
                );
            }
        }

        for side in TEST_STATE_SIDES {
            let mut out_of_range = baseline.clone();
            set_raw_test_field(&mut out_of_range, side, TestStateField::Alpha, 1.000_000_1);
            assert_eq!(
                active_set_full_state_identity(&out_of_range),
                None,
                "raw {side:?} alpha outside [0,1] was sanitized instead of refused"
            );
        }
    }

    #[test]
    fn full_state_identity_is_order_stable_and_rejects_two_sided_layout_drift() {
        let mut first = GraphDomainAlphaState::empty();
        first.insert("relu_b".into(), 3, AlphaNeuronState::new(0.3));
        first.insert("relu_a".into(), 7, AlphaNeuronState::new(0.7));
        let mut reverse = GraphDomainAlphaState::empty();
        reverse.insert("relu_a".into(), 7, AlphaNeuronState::new(0.7));
        reverse.insert("relu_b".into(), 3, AlphaNeuronState::new(0.3));
        assert_eq!(
            active_set_full_state_identity(&first),
            active_set_full_state_identity(&reverse),
            "hash-map insertion order must not affect the sorted identity"
        );

        let mut extra_upper_node = first.clone();
        extra_upper_node
            .upper_neurons_mut()
            .insert("extra".into(), Default::default());
        assert_eq!(active_set_full_state_identity(&extra_upper_node), None);

        let mut wrong_upper_index = first.clone();
        let upper_node = wrong_upper_index
            .upper_neurons_mut()
            .get_mut("relu_a")
            .expect("upper node");
        let neuron = upper_node.remove(&7).expect("upper neuron");
        upper_node.insert(8, neuron);
        assert_eq!(active_set_full_state_identity(&wrong_upper_index), None);
    }

    #[test]
    fn best_pair_selection_never_constructs_an_elementwise_mixture() {
        let plan = active_plan(&[unresolved(0, -2.0), unresolved(1, -1.0)]);
        let initial = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        let initial_fingerprint = initial.fingerprint();
        let crossing_candidate = pair(&plan, &[0.2, -2.0], &[1.0, -1.0], 0.8);
        let candidate_fingerprint = crossing_candidate.fingerprint();
        let mut best = None;

        assert!(retain_lexicographic_best_pair(&plan, &mut best, initial)
            .expect("initial pair is valid"));
        assert!(
            !retain_lexicographic_best_pair(&plan, &mut best, crossing_candidate)
                .expect("crossing candidate is valid")
        );

        let retained = best.expect("one complete pair retained");
        assert_eq!(retained.fingerprint(), initial_fingerprint);
        assert_ne!(retained.fingerprint(), candidate_fingerprint);
        assert_eq!(
            retained
                .bounds()
                .lower()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![-1.0, 0.5],
            "selection must not synthesize the tempting [0.2, 0.5] vector"
        );
        assert_eq!(
            retained.state().alpha("relu0", 0).to_bits(),
            0.2_f32.to_bits()
        );
    }

    #[test]
    fn complete_score_ties_retain_the_earlier_whole_pair() {
        let plan = active_plan(&[unresolved(0, -2.0), unresolved(1, -1.0)]);
        let first = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        let first_fingerprint = first.fingerprint();
        let tied = pair(&plan, &[-1.0, 0.5], &[0.25, 1.25], 0.8);
        let mut best = Some(first);

        assert!(
            !retain_lexicographic_best_pair(&plan, &mut best, tied).expect("tied pair is valid")
        );
        assert_eq!(
            best.expect("earlier pair retained").fingerprint(),
            first_fingerprint
        );
    }

    #[test]
    fn pair_fingerprint_binds_state_vector_upper_bits_and_row_layout() {
        let plan = active_plan(&[unresolved(10, -2.0), unresolved(11, -1.0)]);
        let baseline = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        let different_state = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.8);
        let mut optimizer_state = alpha_state(0.2);
        optimizer_state
            .neuron_mut("relu0", 0)
            .expect("lower neuron")
            .adam_m = 0.125;
        let different_optimizer_state = ActiveSetGpuAlphaCertifiedPair::new(
            &plan,
            bounds(&[-1.0, 0.5], &[0.0, 1.0]),
            optimizer_state,
        )
        .expect("optimizer-bearing pair");
        let different_upper = pair(&plan, &[-1.0, 0.5], &[0.25, 1.0], 0.2);
        let other_plan = active_plan(&[unresolved(10, -2.0), unresolved(12, -1.0)]);
        let different_layout = pair(&other_plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        let endpoint_changed_plan = active_plan(&[
            ActiveSetUnresolvedRow::new(10, -2.0, -0.75, 0.0),
            unresolved(11, -1.0),
        ]);
        let different_historical_endpoint =
            pair(&endpoint_changed_plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);

        assert_ne!(baseline.fingerprint(), different_state.fingerprint());
        assert_ne!(
            baseline.fingerprint(),
            different_optimizer_state.fingerprint()
        );
        assert_ne!(baseline.fingerprint(), different_upper.fingerprint());
        assert_ne!(baseline.fingerprint(), different_layout.fingerprint());
        assert_ne!(
            baseline.fingerprint(),
            different_historical_endpoint.fingerprint(),
            "changing only one historical endpoint bit must change the pair fingerprint"
        );
        assert!(baseline.fingerprint().value() != 0);
    }

    #[test]
    fn pair_validation_refuses_state_score_and_fingerprint_drift() {
        let plan = active_plan(&[unresolved(0, -2.0), unresolved(1, -1.0)]);
        assert_eq!(
            ActiveSetGpuAlphaCertifiedPair::new(
                &plan,
                bounds(&[-1.0, 0.5], &[0.0, 1.0]),
                GraphDomainAlphaState::empty(),
            )
            .unwrap_err(),
            ActiveSetGpuAlphaRefusal::InvalidAlphaState
        );
        let wrong_shape = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![-1.0, 0.5]).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.0, 1.0]).expect("upper"),
        )
        .expect("ordered wrong-shape bounds");
        assert_eq!(
            ActiveSetGpuAlphaCertifiedPair::new(&plan, wrong_shape, alpha_state(0.2)).unwrap_err(),
            ActiveSetGpuAlphaRefusal::InvalidCertifiedVector
        );

        let mut state_drift = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        state_drift
            .state
            .upper_neurons_mut()
            .get_mut("relu0")
            .and_then(|neurons| neurons.get_mut(&0))
            .expect("upper neuron")
            .set_alpha(0.3);
        assert_eq!(
            state_drift.validate(&plan).unwrap_err(),
            ActiveSetGpuAlphaRefusal::CertifiedPairStateMismatch
        );

        let mut score_drift = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        score_drift.score.min_slack = -0.75;
        assert_eq!(
            score_drift.validate(&plan).unwrap_err(),
            ActiveSetGpuAlphaRefusal::CertifiedPairScoreMismatch
        );

        let mut fingerprint_drift = pair(&plan, &[-1.0, 0.5], &[0.0, 1.0], 0.2);
        fingerprint_drift.fingerprint.0 ^= 1;
        assert_eq!(
            fingerprint_drift.validate(&plan).unwrap_err(),
            ActiveSetGpuAlphaRefusal::CertifiedPairFingerprintMismatch
        );
    }

    #[test]
    fn binding_replay_selects_current_worst_row_and_can_be_claimed_once() {
        let mut plan = active_plan(&[unresolved(7, -3.0), unresolved(2, -2.0)]);
        assert_eq!(
            plan.rows()
                .iter()
                .map(|row| row.source_row_index())
                .collect::<Vec<_>>(),
            vec![7, 2],
            "historical ordering starts with source row 7"
        );
        let certified = pair(&plan, &[0.25, -0.75], &[1.0, 0.0], 0.4);
        let expected_fingerprint = certified.fingerprint();
        let expected_identity = certified.state_identity();

        let replay = plan
            .claim_binding_row_replay(&certified)
            .expect("first replay claim");
        assert_eq!(replay.active_ordinal(), 1);
        assert_eq!(replay.source_row_index(), 2);
        assert_eq!(replay.certified_lower().to_bits(), (-0.75_f32).to_bits());
        assert_eq!(replay.threshold().to_bits(), 0.0_f32.to_bits());
        assert_eq!(replay.state_identity(), expected_identity);
        assert_eq!(replay.certified_pair_fingerprint(), expected_fingerprint);
        assert_eq!(
            plan.claim_binding_row_replay(&certified).unwrap_err(),
            ActiveSetGpuAlphaRefusal::BindingReplayAlreadyClaimed
        );
    }

    #[test]
    fn mismatched_pair_refusal_does_not_consume_the_binding_replay() {
        let mut plan = active_plan(&[unresolved(0, -2.0), unresolved(1, -1.0)]);
        let other_plan = active_plan(&[unresolved(0, -2.0), unresolved(2, -1.0)]);
        let mismatched = pair(&other_plan, &[-1.0, -0.5], &[0.0, 0.0], 0.2);
        assert_eq!(
            plan.claim_binding_row_replay(&mismatched).unwrap_err(),
            ActiveSetGpuAlphaRefusal::CertifiedPairFingerprintMismatch
        );

        let matching = pair(&plan, &[-1.0, -0.5], &[0.0, 0.0], 0.2);
        assert!(plan.claim_binding_row_replay(&matching).is_ok());
    }

    #[test]
    fn binding_ties_select_the_first_deterministically_sorted_active_row() {
        let mut plan = active_plan(&[unresolved(9, -1.0), unresolved(3, -1.0)]);
        let tied = pair(&plan, &[-0.5, -0.5], &[0.0, 0.0], 0.2);

        let replay = plan
            .claim_binding_row_replay(&tied)
            .expect("one tied binding replay");
        assert_eq!(replay.active_ordinal(), 0);
        assert_eq!(
            replay.source_row_index(),
            3,
            "historical-slack ties sort by source row and replay ties retain that first row"
        );
    }
}
