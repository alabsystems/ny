// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Call-local authority boundary for exact-certified 2-ReLU lower cuts.
//!
//! This module is intentionally private. Its historical entry point remains
//! production-hard-disabled. The M1 call-context entry is consumed only by an
//! observation-only resident backend and has no verdict conversion. It seals
//! the narrow semantic gap between [`ExactRelu2FacetCertificate`] (evidence
//! about one octahedron) and a future verdict-bearing resident CROWN call:
//!
//! 1. receive a sealed resident-call context that binds the exact graph, input
//!    box, intermediate bounds, alpha state, lower-objective seed, engine,
//!    frontier generation, and deadline;
//! 2. resolve one exact ReLU and its ordered neuron pair;
//! 3. freshly reproduce the pair's [`Octahedron2`] from those borrowed objects;
//! 4. require bit-exact equality with every certificate's retained support; and
//! 5. build immutable, row-local lower contributions with `lambda >= 0`.
//!
//! There is no global registry, no raw [`super::Facet`] input/output, no upper
//! contribution, no violation decision, and no cache token. The test replay
//! retains borrows of every supplied component and is neither `Clone` nor
//! serializable, so it cannot outlive those components. The dormant call
//! context additionally binds the replay to a non-zero-sized request seal and
//! exact snapshot generations.
//!
//! A shared Rust lifetime only proves simultaneous liveness; it does not prove
//! that independently supplied graph, bounds, alpha, seed, and frontier values
//! came from the same resident call or domain. Consequently the context
//! constructor and evidence-bearing carrier stay within this private
//! `multineuron` implementation. A future activation must construct the context
//! from one actual resident call, advance every generation at its mutation
//! boundary, and consume the replay synchronously.
//!
//! The future backend must atomically consume each [`DirectedLowerChannel`] as
//! `(stored_value, source_abs_error)`: add `stored_value` only to the lower
//! coefficient frontier, add `source_abs_error` to its certified coefficient
//! error lane, and apply the bias as `stored_value - source_abs_error`. It must
//! additionally charge the exact error of the resident f32 mutation itself.

use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use ny_core::dd::{next_up_f64, two_sum};
use ny_core::dd_selfcheck::dd_selfcheck_ok;
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownSeed, GpuResnetSegment, NyError,
    ResidentCutShadowOutcome, ResidentCutShadowPolicy, ResidentLowerCutCarrier,
    ResidentLowerCutChannel, ResidentLowerCutRow, Result,
};
use ny_tensor::{next_up_f32, BoundedTensor};

use super::producer::combined_row_octahedron_with_deadline;
use super::{ExactRelu2FacetCertificate, Octahedron2};
use crate::bounds::GraphAlphaState;
use crate::layers::Layer;
use crate::GraphNetwork;

/// Bounded proof surface for the first resident replay.
const MAX_AUTHORITY_FACETS: usize = 8;
const MAX_AUTHORITY_ROWS: usize = 64;

/// Typed production gate.
///
/// The only production state is `Disabled`; there is intentionally no public,
/// crate-visible, CLI, config, or environment constructor for an enabled value.
/// The test-only state lets deterministic tests exercise the otherwise sealed
/// construction path without creating a shippable authority toggle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CertifiedCutAuthorityGate {
    state: CertifiedCutAuthorityGateState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CertifiedCutAuthorityGateState {
    #[default]
    Disabled,
    #[cfg(test)]
    TestOnlyEnabled,
}

impl CertifiedCutAuthorityGate {
    #[cfg(test)]
    const fn test_only_enabled() -> Self {
        Self {
            state: CertifiedCutAuthorityGateState::TestOnlyEnabled,
        }
    }

    const fn is_enabled(self) -> bool {
        match self.state {
            CertifiedCutAuthorityGateState::Disabled => false,
            #[cfg(test)]
            CertifiedCutAuthorityGateState::TestOnlyEnabled => true,
        }
    }
}

/// Borrowed lower half of one exact CROWN seed.
///
/// Deliberately omits `upper_a` and `upper_b`: this authority slice has no type
/// through which it could mutate or authorize an upper bound.
struct LowerSeedRef<'a> {
    a: &'a [f32],
    b: &'a [f32],
    num_specs: usize,
    current_dim: usize,
}

impl<'a> LowerSeedRef<'a> {
    fn from_seed(seed: &'a GpuCrownSeed) -> Self {
        Self {
            a: &seed.lower_a,
            b: &seed.lower_b,
            num_specs: seed.num_specs,
            current_dim: seed.current_dim,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.num_specs == 0 || self.num_specs > MAX_AUTHORITY_ROWS || self.current_dim == 0 {
            return Err(NyError::InvalidSpec(format!(
                "certified lower-cut authority: seed shape is outside the first-slice limit \
                 (num_specs={}, current_dim={})",
                self.num_specs, self.current_dim
            )));
        }
        let expected_a = self
            .num_specs
            .checked_mul(self.current_dim)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "certified lower-cut authority: lower seed shape overflow".into(),
                )
            })?;
        if self.a.len() != expected_a || self.b.len() != self.num_specs {
            return Err(NyError::InvalidSpec(format!(
                "certified lower-cut authority: malformed lower seed \
                 (a={}, expected_a={expected_a}, b={}, expected_b={})",
                self.a.len(),
                self.b.len(),
                self.num_specs
            )));
        }
        if !self.a.iter().chain(self.b).all(|value| value.is_finite()) {
            return Err(NyError::NumericalInstability(
                "certified lower-cut authority: non-finite lower seed".into(),
            ));
        }
        Ok(())
    }
}

/// Borrowed component bundle for one prospective lower-cut replay.
///
/// `target_relu` names the actual graph ReLU, not merely a same-width activation
/// descriptor. Its unary predecessor is resolved from `graph` inside the
/// builder and becomes the fresh octahedron target.
#[allow(clippy::too_many_arguments)]
struct CurrentLowerCutRequest<'a> {
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    alpha_state: &'a GraphAlphaState,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    engine: Option<&'a dyn GemmEngine>,
    lower_seed: LowerSeedRef<'a>,
    target_relu: &'a str,
    ordered_neurons: [usize; 2],
    deadline: Instant,
}

// Production intentionally has no constructor. A future constructor must take
// one opaque resident-call/domain context; co-borrowing independently supplied
// components under a common lifetime is not provenance.
#[cfg(test)]
impl<'a> CurrentLowerCutRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        alpha_state: &'a GraphAlphaState,
        node_bounds: &'a HashMap<String, BoundedTensor>,
        engine: Option<&'a dyn GemmEngine>,
        seed: &'a GpuCrownSeed,
        target_relu: &'a str,
        ordered_neurons: [usize; 2],
        deadline: Instant,
    ) -> Self {
        Self {
            graph,
            input,
            alpha_state,
            node_bounds,
            engine,
            lower_seed: LowerSeedRef::from_seed(seed),
            target_relu,
            ordered_neurons,
            deadline,
        }
    }
}

/// One validated nonnegative, finite row/facet multiplier.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NonNegativeMultiplier(f32);

impl NonNegativeMultiplier {
    fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "certified lower-cut authority: lambda must be finite and nonnegative, got {value}"
            )));
        }
        // Canonicalize -0.0 so bit identity never depends on the caller's zero
        // sign and the all-zero fast exit remains exact.
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }

    const fn get(self) -> f32 {
        self.0
    }
}

/// Stored f32 channel plus an outward absolute source-error charge.
///
/// If `q` is the exact real sum of all `lambda * coefficient` products used to
/// build this channel, then
///
/// `q in [value - reduction_error, value + reduction_error]`.
///
/// Products are exact in f64 because each operand is stored f32 (at most 48
/// significand bits). [`two_sum`] captures every f64 addition residual; the
/// residual accumulation and final f64-to-f32 gap are rounded upward.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DirectedLowerChannel {
    value: f32,
    reduction_error: f32,
}

impl DirectedLowerChannel {
    const fn value(self) -> f32 {
        self.value
    }

    const fn reduction_error(self) -> f32 {
        self.reduction_error
    }
}

/// Explicit lower-only channels for one objective row.
#[derive(Debug)]
struct LowerCutRowReplay {
    multipliers: Vec<NonNegativeMultiplier>,
    pre: [DirectedLowerChannel; 2],
    post: [DirectedLowerChannel; 2],
    bias: DirectedLowerChannel,
}

impl LowerCutRowReplay {
    fn pre(&self) -> &[DirectedLowerChannel; 2] {
        &self.pre
    }

    fn post(&self) -> &[DirectedLowerChannel; 2] {
        &self.post
    }

    const fn bias(&self) -> DirectedLowerChannel {
        self.bias
    }

    fn multipliers(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.multipliers.iter().map(|value| value.get())
    }
}

/// Immutable evidence-bearing replay for one component bundle.
///
/// The private certificate vector preserves proof evidence, and no raw facet is
/// exposed. Retaining the component borrows prevents persistence beyond their
/// lifetimes; it does not prove same-call/domain provenance.
struct CertifiedLowerCutReplay<'a> {
    request: CurrentLowerCutRequest<'a>,
    pre_node: &'a str,
    target_width: usize,
    fresh_support: Octahedron2,
    certificates: Vec<ExactRelu2FacetCertificate>,
    rows: Vec<LowerCutRowReplay>,
}

/// Exact call-local generations that must move in lockstep with a cut carrier.
///
/// These are not hashes and are never exposed outside the private
/// `multineuron` implementation.  The opaque call seal below supplies request
/// identity; generations reject a carrier after any bound/optimizer/frontier
/// component advances inside that same request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResidentCutSnapshotGenerations {
    pub(super) domain: u64,
    pub(super) bounds: u64,
    pub(super) alpha: u64,
    pub(super) beta: u64,
    pub(super) objective: u64,
    pub(super) decomposition: u64,
    pub(super) frontier: u64,
}

impl ResidentCutSnapshotGenerations {
    /// Initial generations for one newly constructed production call context.
    ///
    /// Request identity comes from the non-zero-sized seal, not these values.
    /// Starting every independent context at one is therefore intentional.
    pub(super) const fn initial() -> Self {
        Self {
            domain: 1,
            bounds: 1,
            alpha: 1,
            beta: 1,
            objective: 1,
            decomposition: 1,
            frontier: 1,
        }
    }

    #[cfg(test)]
    pub(super) const fn fixture() -> Self {
        Self::initial()
    }
}

/// Non-zero-sized, non-cloneable identity owned by one resident-call context.
struct ResidentCutCallSeal([u8; 1]);

/// Opaque semantic authority for one exact resident call and one domain.
///
/// Every proof-relevant object is borrowed together and the carrier retains a
/// reference to this context's unique seal.  There is no public constructor,
/// registry, environment switch, stable hash, or serializable identity.
#[allow(clippy::too_many_arguments)]
pub(super) struct ResidentCutCallContext<'a> {
    seal: ResidentCutCallSeal,
    generations: Cell<ResidentCutSnapshotGenerations>,
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    alpha_state: &'a GraphAlphaState,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    engine: Option<&'a dyn GemmEngine>,
    seed: &'a GpuCrownSeed,
    target_relu: &'a str,
    ordered_neurons: [usize; 2],
    segments: &'a [GpuResnetSegment],
    relu_names: &'a [String],
    beta_signed: &'a [Vec<f32>],
    frontier_abs: &'a [Vec<f32>],
    node_abs: &'a [Vec<f32>],
    resident_input_lower: &'a [f32],
    resident_input_upper: &'a [f32],
    deadline: Instant,
}

/// Semantic proof plus arithmetic transport, inseparable within one call.
///
/// `proof` retains the freshly replayed exact certificates. `transport` alone
/// remains arithmetic-only and cannot cross into the reference backend without
/// `validate_bound_carrier` proving exact context and generation identity.
pub(super) struct BoundResidentLowerCutCarrier<'context, 'data> {
    seal: &'context ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    proof: CertifiedLowerCutReplay<'data>,
    transport: ResidentLowerCutCarrier,
}

impl<'a> ResidentCutCallContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        generations: ResidentCutSnapshotGenerations,
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        alpha_state: &'a GraphAlphaState,
        node_bounds: &'a HashMap<String, BoundedTensor>,
        engine: Option<&'a dyn GemmEngine>,
        seed: &'a GpuCrownSeed,
        target_relu: &'a str,
        ordered_neurons: [usize; 2],
        segments: &'a [GpuResnetSegment],
        relu_names: &'a [String],
        beta_signed: &'a [Vec<f32>],
        frontier_abs: &'a [Vec<f32>],
        node_abs: &'a [Vec<f32>],
        resident_input_lower: &'a [f32],
        resident_input_upper: &'a [f32],
        deadline: Instant,
    ) -> Self {
        Self {
            seal: ResidentCutCallSeal([0xC7]),
            generations: Cell::new(generations),
            graph,
            input,
            alpha_state,
            node_bounds,
            engine,
            seed,
            target_relu,
            ordered_neurons,
            segments,
            relu_names,
            beta_signed,
            frontier_abs,
            node_abs,
            resident_input_lower,
            resident_input_upper,
            deadline,
        }
    }

    /// Build a complete carrier after validating the exact decomposition and
    /// freshly replaying every exact-support certificate.
    pub(super) fn build_bound_carrier<'context>(
        &'context self,
        certificates: &[ExactRelu2FacetCertificate],
        row_lambdas: &[Vec<f32>],
    ) -> Result<Option<BoundResidentLowerCutCarrier<'context, 'a>>> {
        check_deadline(self.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let (target_activation, activation_widths) = self.validate_resident_snapshot()?;

        let request = CurrentLowerCutRequest {
            graph: self.graph,
            input: self.input,
            alpha_state: self.alpha_state,
            node_bounds: self.node_bounds,
            engine: self.engine,
            lower_seed: LowerSeedRef::from_seed(self.seed),
            target_relu: self.target_relu,
            ordered_neurons: self.ordered_neurons,
            deadline: self.deadline,
        };
        let mut deadline_check = |stage| check_deadline(self.deadline, stage);
        let Some(proof) = build_certified_lower_cut_replay_with(
            request,
            certificates,
            row_lambdas,
            dd_selfcheck_ok(),
            &mut deadline_check,
            |request, pre_node| {
                combined_row_octahedron_with_deadline(
                    request.graph,
                    request.input,
                    request.alpha_state,
                    Some(request.node_bounds),
                    pre_node,
                    request.ordered_neurons[0],
                    request.ordered_neurons[1],
                    request.engine,
                    Some(request.deadline),
                )
            },
        )?
        else {
            return Ok(None);
        };

        let rows = proof
            .rows()
            .iter()
            .map(|row| {
                let pre = [
                    transport_channel(row.pre()[0])?,
                    transport_channel(row.pre()[1])?,
                ];
                let post = [
                    transport_channel(row.post()[0])?,
                    transport_channel(row.post()[1])?,
                ];
                ResidentLowerCutRow::try_new(
                    row.multipliers().collect(),
                    pre,
                    post,
                    transport_channel(row.bias())?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let transport = ResidentLowerCutCarrier::try_new(
            target_activation,
            proof.target_width(),
            proof.ordered_neurons(),
            rows,
            self.deadline,
        )?;
        transport.validate_for_call(
            activation_widths.len(),
            activation_widths[target_activation],
            self.seed.num_specs,
            self.deadline,
        )?;
        check_deadline(self.deadline, AuthorityDeadlineStage::BeforePublish)?;

        Ok(Some(BoundResidentLowerCutCarrier {
            seal: &self.seal,
            generations: self.generations.get(),
            proof,
            transport,
        }))
    }

    /// Atomically validate the full semantic carrier immediately before use.
    pub(super) fn validate_bound_carrier(
        &self,
        carrier: &BoundResidentLowerCutCarrier<'_, 'a>,
    ) -> Result<()> {
        if !std::ptr::eq(
            std::ptr::from_ref(&self.seal),
            std::ptr::from_ref(carrier.seal),
        ) {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut carrier belongs to another call/domain context".into(),
            ));
        }
        if self.generations.get() != carrier.generations {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut carrier snapshot generation is stale".into(),
            ));
        }
        if carrier.proof.target_relu() != self.target_relu
            || carrier.proof.ordered_neurons() != self.ordered_neurons
            || carrier.proof.rows().len() != self.seed.num_specs
        {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut semantic proof no longer matches the call".into(),
            ));
        }
        let (target_activation, activation_widths) = self.validate_resident_snapshot()?;
        if carrier.transport.target_activation() != target_activation {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut target moved in the current decomposition".into(),
            ));
        }
        carrier.transport.validate_for_call(
            activation_widths.len(),
            activation_widths[target_activation],
            self.seed.num_specs,
            self.deadline,
        )
    }

    pub(super) const fn deadline(&self) -> Instant {
        self.deadline
    }

    pub(super) const fn seed(&self) -> &GpuCrownSeed {
        self.seed
    }

    pub(super) fn transport<'context>(
        &self,
        carrier: &'context BoundResidentLowerCutCarrier<'_, 'a>,
    ) -> Result<&'context ResidentLowerCutCarrier> {
        self.validate_bound_carrier(carrier)?;
        Ok(&carrier.transport)
    }

    /// Consume one semantic carrier synchronously through the actual resident
    /// backend call.
    ///
    /// The first frontier-generation advance occurs after the final semantic
    /// validation and before the backend receives the arithmetic transport.
    /// The second occurs after the synchronous backend returns, including its
    /// refusal paths. Consequently the evidence-bearing carrier is stale after
    /// one attempt and cannot be replayed against a later frontier. Device work
    /// remains scratch-only and observation-only in the backend.
    pub(super) fn run_backend_shadow(
        &self,
        carrier: &BoundResidentLowerCutCarrier<'_, 'a>,
        gpu: &dyn GpuCrownBackward,
        binding_row: usize,
    ) -> Result<ResidentCutShadowOutcome> {
        self.validate_bound_carrier(carrier)?;
        let transport = &carrier.transport;
        self.advance_frontier_generation()?;
        let outcome = gpu.crown_backward_gpu_resnet_sound_beta_cut_shadow(
            ResidentCutShadowPolicy::Shadow,
            self.segments,
            self.seed,
            self.resident_input_lower,
            self.resident_input_upper,
            self.beta_signed,
            self.frontier_abs,
            self.node_abs,
            Some(transport),
            binding_row,
            self.deadline,
        );
        self.advance_frontier_generation()?;
        outcome
    }

    #[cfg(test)]
    pub(super) fn set_generations_for_test(&self, generations: ResidentCutSnapshotGenerations) {
        self.generations.set(generations);
    }

    fn validate_resident_snapshot(&self) -> Result<(usize, Vec<usize>)> {
        if Instant::now() >= self.deadline {
            return Err(NyError::DeadlineExceeded(
                "resident lower-cut snapshot deadline expired".into(),
            ));
        }
        let input = self.input.flatten();
        if !slice_bits_equal(input.lower().as_slice(), self.resident_input_lower)
            || !slice_bits_equal(input.upper().as_slice(), self.resident_input_upper)
        {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut input box does not bit-match the semantic domain".into(),
            ));
        }
        let activation_widths = resident_activation_widths(self.segments)?;
        if activation_widths.len() != self.relu_names.len()
            || self.beta_signed.len() != self.relu_names.len()
            || self.node_abs.len() != self.relu_names.len()
            || self.frontier_abs.len() != self.segments.len()
        {
            return Err(NyError::InvalidSpec(
                "resident lower-cut decomposition/frontier shapes are inconsistent".into(),
            ));
        }
        for (index, &width) in activation_widths.iter().enumerate() {
            if self.beta_signed[index].len() != width || self.node_abs[index].len() != width {
                return Err(NyError::InvalidSpec(format!(
                    "resident lower-cut activation {index} width disagrees with beta/node frontier"
                )));
            }
        }
        if self
            .beta_signed
            .iter()
            .flatten()
            .chain(self.frontier_abs.iter().flatten())
            .chain(self.node_abs.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "resident lower-cut decomposition/frontier contains non-finite data".into(),
            ));
        }
        let mut matches = self
            .relu_names
            .iter()
            .enumerate()
            .filter(|(_, name)| name.as_str() == self.target_relu);
        let (target_activation, _) = matches.next().ok_or_else(|| {
            NyError::InvalidSpec(
                "resident lower-cut target ReLU is absent from the exact decomposition".into(),
            )
        })?;
        if matches.next().is_some() {
            return Err(NyError::InvalidSpec(
                "resident lower-cut target ReLU occurs more than once in the decomposition".into(),
            ));
        }
        Ok((target_activation, activation_widths))
    }

    fn advance_frontier_generation(&self) -> Result<()> {
        let mut generations = self.generations.get();
        generations.frontier = generations.frontier.checked_add(1).ok_or_else(|| {
            NyError::SoundnessRefusal(
                "resident lower-cut frontier generation exhausted within one call".into(),
            )
        })?;
        self.generations.set(generations);
        Ok(())
    }
}

fn transport_channel(channel: DirectedLowerChannel) -> Result<ResidentLowerCutChannel> {
    ResidentLowerCutChannel::try_new(channel.value(), channel.reduction_error())
}

fn slice_bits_equal(left: Option<&[f32]>, right: &[f32]) -> bool {
    left.is_some_and(|left| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
    })
}

fn resident_activation_widths(segments: &[GpuResnetSegment]) -> Result<Vec<usize>> {
    let mut widths = Vec::new();
    let mut visit = |layers: &[GpuCrownLayer]| -> Result<()> {
        for layer in layers {
            match layer {
                GpuCrownLayer::Activation { num_neurons, .. }
                | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                    if *num_neurons == 0 {
                        return Err(NyError::InvalidSpec(
                            "resident lower-cut decomposition has a zero-width activation".into(),
                        ));
                    }
                    widths.push(*num_neurons);
                }
                _ => {}
            }
        }
        Ok(())
    };
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                visit(layers)?;
            }
            GpuResnetSegment::ResidualProj(function, projection) => {
                visit(function)?;
                visit(projection)?;
            }
        }
    }
    if widths.is_empty() {
        return Err(NyError::InvalidSpec(
            "resident lower-cut decomposition contains no activation".into(),
        ));
    }
    Ok(widths)
}

impl std::fmt::Debug for CertifiedLowerCutReplay<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CertifiedLowerCutReplay")
            .field("target_relu", &self.request.target_relu)
            .field("pre_node", &self.pre_node)
            .field("ordered_neurons", &self.request.ordered_neurons)
            .field("target_width", &self.target_width)
            .field("certificate_count", &self.certificates.len())
            .field("row_count", &self.rows.len())
            .finish_non_exhaustive()
    }
}

impl CertifiedLowerCutReplay<'_> {
    fn target_relu(&self) -> &str {
        self.request.target_relu
    }

    const fn pre_node(&self) -> &str {
        self.pre_node
    }

    const fn ordered_neurons(&self) -> [usize; 2] {
        self.request.ordered_neurons
    }

    const fn target_width(&self) -> usize {
        self.target_width
    }

    fn rows(&self) -> &[LowerCutRowReplay] {
        &self.rows
    }

    const fn fresh_support(&self) -> &Octahedron2 {
        &self.fresh_support
    }

    fn certificate_count(&self) -> usize {
        self.certificates.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthorityDeadlineStage {
    BeforeValidation,
    AfterTargetValidation,
    AfterMultiplierValidation,
    BeforeFreshSupport,
    AfterFreshSupport,
    AfterCertificateValidation(usize),
    BeforeRowReduction(usize),
    AfterRowReduction(usize),
    BeforePublish,
}

fn check_deadline(deadline: Instant, stage: AuthorityDeadlineStage) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "certified lower-cut authority: deadline exceeded at {stage:?}"
        )));
    }
    Ok(())
}

/// Construct the historical module-local, production-hard-disabled replay.
///
/// `row_lambdas[row][facet]` is the immutable multiplier snapshot for one
/// lower objective. `Ok(None)` means the production-hard-disabled gate or an
/// exact all-zero multiplier state; neither path produces or calls a cut
/// backend. Every other refusal is fail-closed.
fn build_certified_lower_cut_replay<'a>(
    gate: CertifiedCutAuthorityGate,
    request: CurrentLowerCutRequest<'a>,
    certificates: &[ExactRelu2FacetCertificate],
    row_lambdas: &[Vec<f32>],
) -> Result<Option<CertifiedLowerCutReplay<'a>>> {
    // Default/off returns before graph, seed, certificate, multiplier, or
    // deadline inspection. This is the hard off-parity boundary.
    if !gate.is_enabled() {
        return Ok(None);
    }

    let deadline = request.deadline;
    let mut deadline_check = |stage| check_deadline(deadline, stage);
    build_certified_lower_cut_replay_with(
        request,
        certificates,
        row_lambdas,
        dd_selfcheck_ok(),
        &mut deadline_check,
        |request, pre_node| {
            combined_row_octahedron_with_deadline(
                request.graph,
                request.input,
                request.alpha_state,
                Some(request.node_bounds),
                pre_node,
                request.ordered_neurons[0],
                request.ordered_neurons[1],
                request.engine,
                Some(request.deadline),
            )
        },
    )
}

#[allow(clippy::type_complexity)]
fn build_certified_lower_cut_replay_with<'a, C, P>(
    request: CurrentLowerCutRequest<'a>,
    certificates: &[ExactRelu2FacetCertificate],
    row_lambdas: &[Vec<f32>],
    eft_authorized: bool,
    check: &mut C,
    produce_fresh_support: P,
) -> Result<Option<CertifiedLowerCutReplay<'a>>>
where
    C: FnMut(AuthorityDeadlineStage) -> Result<()>,
    P: FnOnce(&CurrentLowerCutRequest<'a>, &'a str) -> Result<Octahedron2>,
{
    check(AuthorityDeadlineStage::BeforeValidation)?;

    request.lower_seed.validate()?;
    let (pre_node, target_width) = validate_target(&request)?;
    check(AuthorityDeadlineStage::AfterTargetValidation)?;

    let multipliers = validate_multipliers(
        request.lower_seed.num_specs,
        certificates.len(),
        row_lambdas,
    )?;
    check(AuthorityDeadlineStage::AfterMultiplierValidation)?;
    if multipliers
        .iter()
        .flatten()
        .all(|multiplier| multiplier.get() == 0.0)
    {
        // Exact off0: no fresh CROWN production and no future backend call.
        return Ok(None);
    }
    if !eft_authorized {
        return Err(NyError::SoundnessRefusal(
            "certified lower-cut authority: IEEE error-free-transform self-check failed".into(),
        ));
    }

    check(AuthorityDeadlineStage::BeforeFreshSupport)?;
    let fresh_support = produce_fresh_support(&request, pre_node)?;
    check(AuthorityDeadlineStage::AfterFreshSupport)?;
    if !fresh_support.both_unstable() {
        return Err(NyError::InvalidSpec(
            "certified lower-cut authority: target pair is not both unstable".into(),
        ));
    }

    for (facet_index, certificate) in certificates.iter().enumerate() {
        if !octahedron_bits_equal(certificate.support_domain(), &fresh_support) {
            return Err(NyError::SoundnessRefusal(format!(
                "certified lower-cut authority: certificate {facet_index} support does not \
                 bit-match the freshly reproduced request support"
            )));
        }
        let facet = certificate.facet();
        if !facet
            .a
            .iter()
            .chain([&facet.b])
            .all(|value| value.is_finite())
        {
            return Err(NyError::NumericalInstability(format!(
                "certified lower-cut authority: certificate {facet_index} contains non-finite data"
            )));
        }
        check(AuthorityDeadlineStage::AfterCertificateValidation(
            facet_index,
        ))?;
    }

    // Assemble into a local vector and publish only after every row succeeds
    // and the final deadline checkpoint passes.
    let mut rows = Vec::with_capacity(multipliers.len());
    for (row_index, row_multipliers) in multipliers.into_iter().enumerate() {
        check(AuthorityDeadlineStage::BeforeRowReduction(row_index))?;
        rows.push(reduce_row(certificates, row_multipliers)?);
        check(AuthorityDeadlineStage::AfterRowReduction(row_index))?;
    }
    check(AuthorityDeadlineStage::BeforePublish)?;

    Ok(Some(CertifiedLowerCutReplay {
        request,
        pre_node,
        target_width,
        fresh_support,
        certificates: certificates.to_vec(),
        rows,
    }))
}

fn validate_target<'a>(request: &CurrentLowerCutRequest<'a>) -> Result<(&'a str, usize)> {
    let relu = request.graph.node(request.target_relu).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "certified lower-cut authority: target ReLU '{}' is absent",
            request.target_relu
        ))
    })?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: target '{}' is not an exact ReLU",
            request.target_relu
        )));
    }
    let pre_node = relu.require_unary_input()?;
    let output_ancestors = request
        .graph
        .all_ancestors()?
        .get(request.graph.output_name())
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "certified lower-cut authority: graph output has no ancestor set".into(),
            )
        })?;
    if !output_ancestors
        .iter()
        .any(|name| name == request.target_relu)
    {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: target '{}' is not in the current output request",
            request.target_relu
        )));
    }

    let pre_width = request
        .node_bounds
        .get(pre_node)
        .map(|bounds| bounds.flatten().len())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "certified lower-cut authority: current bounds omit pre-activation '{pre_node}'"
            ))
        })?;
    let post_width = request
        .node_bounds
        .get(request.target_relu)
        .map(|bounds| bounds.flatten().len())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "certified lower-cut authority: current bounds omit target ReLU '{}'",
                request.target_relu
            ))
        })?;
    if pre_width == 0 || pre_width != post_width {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: target/pre width mismatch \
             (target={post_width}, pre={pre_width})"
        )));
    }
    let [first, second] = request.ordered_neurons;
    if first == second || first >= pre_width || second >= pre_width {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: invalid ordered pair \
             ({first}, {second}) for width {pre_width}"
        )));
    }
    Ok((pre_node, pre_width))
}

fn validate_multipliers(
    expected_rows: usize,
    facet_count: usize,
    row_lambdas: &[Vec<f32>],
) -> Result<Vec<Vec<NonNegativeMultiplier>>> {
    if facet_count == 0 || facet_count > MAX_AUTHORITY_FACETS {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: facet count {facet_count} is outside 1..={MAX_AUTHORITY_FACETS}"
        )));
    }
    if row_lambdas.len() != expected_rows {
        return Err(NyError::InvalidSpec(format!(
            "certified lower-cut authority: multiplier rows {} do not match lower objectives {expected_rows}",
            row_lambdas.len()
        )));
    }
    row_lambdas
        .iter()
        .enumerate()
        .map(|(row, values)| {
            if values.len() != facet_count {
                return Err(NyError::InvalidSpec(format!(
                    "certified lower-cut authority: row {row} has {} multipliers for {facet_count} facets",
                    values.len()
                )));
            }
            values
                .iter()
                .copied()
                .map(NonNegativeMultiplier::new)
                .collect()
        })
        .collect()
}

fn reduce_row(
    certificates: &[ExactRelu2FacetCertificate],
    multipliers: Vec<NonNegativeMultiplier>,
) -> Result<LowerCutRowReplay> {
    let reduce_coordinate =
        |coordinate: usize| {
            directed_reduce(certificates.iter().zip(&multipliers).map(
                |(certificate, multiplier)| {
                    (
                        f64::from(multiplier.get()),
                        f64::from(certificate.facet().a[coordinate]),
                    )
                },
            ))
        };

    let pre = [reduce_coordinate(0)?, reduce_coordinate(1)?];
    let post = [reduce_coordinate(2)?, reduce_coordinate(3)?];
    let bias = directed_reduce(certificates.iter().zip(&multipliers).map(
        |(certificate, multiplier)| {
            (
                f64::from(multiplier.get()),
                -f64::from(certificate.facet().b),
            )
        },
    ))?;
    Ok(LowerCutRowReplay {
        multipliers,
        pre,
        post,
        bias,
    })
}

fn directed_reduce(products: impl IntoIterator<Item = (f64, f64)>) -> Result<DirectedLowerChannel> {
    if !dd_selfcheck_ok() {
        return Err(NyError::SoundnessRefusal(
            "certified lower-cut authority: IEEE error-free-transform self-check failed".into(),
        ));
    }
    let mut sum = 0.0_f64;
    let mut reduction_error = 0.0_f64;
    for (left, right) in products {
        let product = left * right;
        if !left.is_finite() || !right.is_finite() || !product.is_finite() {
            return Err(NyError::NumericalInstability(
                "certified lower-cut authority: non-finite channel product".into(),
            ));
        }
        // A binary32-by-binary32 product is exact in binary64. TwoSum then
        // captures the exact residual of adding it to the running binary64 sum.
        let (next, residual) = two_sum(sum, product);
        if !next.is_finite() || !residual.is_finite() {
            return Err(NyError::NumericalInstability(
                "certified lower-cut authority: non-finite channel reduction".into(),
            ));
        }
        sum = next;
        reduction_error = add_nonnegative_up(reduction_error, residual.abs())?;
    }

    let value = sum as f32;
    if !value.is_finite() {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: reduced channel is not representable as finite f32"
                .into(),
        ));
    }
    let conversion_gap = (f64::from(value) - sum).abs();
    // The subtraction above is itself an f64 operation. Direct it upward
    // before combining it with the already-directed reduction residual.
    let conversion_gap = if conversion_gap == 0.0 {
        0.0
    } else {
        next_up_f64(conversion_gap)
    };
    reduction_error = add_nonnegative_up(reduction_error, conversion_gap)?;
    Ok(DirectedLowerChannel {
        value,
        reduction_error: f64_to_f32_up(reduction_error)?,
    })
}

fn add_nonnegative_up(left: f64, right: f64) -> Result<f64> {
    if !left.is_finite() || !right.is_finite() || left < 0.0 || right < 0.0 {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: invalid nonnegative error term".into(),
        ));
    }
    let sum = left + right;
    if !sum.is_finite() {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: error accumulator overflow".into(),
        ));
    }
    Ok(if sum == 0.0 { 0.0 } else { next_up_f64(sum) })
}

fn f64_to_f32_up(value: f64) -> Result<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: invalid source error".into(),
        ));
    }
    if value == 0.0 {
        return Ok(0.0);
    }
    let mut encoded = value as f32;
    if !encoded.is_finite() {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: source error is not representable as finite f32".into(),
        ));
    }
    if f64::from(encoded) < value {
        encoded = next_up_f32(encoded);
    }
    if !encoded.is_finite() || f64::from(encoded) < value {
        return Err(NyError::NumericalInstability(
            "certified lower-cut authority: directed source-error conversion failed".into(),
        ));
    }
    Ok(encoded)
}

fn octahedron_bits_equal(left: &Octahedron2, right: &Octahedron2) -> bool {
    [
        left.l1.to_bits() == right.l1.to_bits(),
        left.u1.to_bits() == right.u1.to_bits(),
        left.l2.to_bits() == right.l2.to_bits(),
        left.u2.to_bits() == right.u2.to_bits(),
        left.s_lo.to_bits() == right.s_lo.to_bits(),
        left.s_hi.to_bits() == right.s_hi.to_bits(),
        left.d_lo.to_bits() == right.d_lo.to_bits(),
        left.d_hi.to_bits() == right.d_hi.to_bits(),
    ]
    .into_iter()
    .all(|equal| equal)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;
    use std::time::Duration;

    use ndarray::{arr1, arr2};
    use num_rational::BigRational;

    use super::*;
    use crate::layers::activations::ReLULayer;
    use crate::{GraphNode, LinearLayer};

    struct Fixture {
        graph: GraphNetwork,
        input: BoundedTensor,
        node_bounds: HashMap<String, BoundedTensor>,
        alpha: GraphAlphaState,
        seed: GpuCrownSeed,
    }

    impl Fixture {
        fn new(num_specs: usize) -> Self {
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input(
                "pre",
                Layer::Linear(
                    LinearLayer::new(
                        arr2(&[[1.0_f32, 0.25], [-0.5, 1.0]]),
                        Some(arr1(&[0.125_f32, -0.25])),
                    )
                    .expect("finite linear fixture"),
                ),
            ));
            graph.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer::new()),
                vec!["pre".into()],
            ));
            graph.set_output("relu");
            let input = BoundedTensor::new(
                arr1(&[-1.0_f32, -1.0]).into_dyn(),
                arr1(&[1.0_f32, 1.0]).into_dyn(),
            )
            .expect("finite input fixture");
            let node_bounds = graph
                .collect_node_bounds_with_engine(&input, None)
                .expect("fixture IBP");
            let lower_a = vec![0.25_f32; num_specs * 2];
            let lower_b = vec![0.0_f32; num_specs];
            let seed = GpuCrownSeed {
                lower_a: Arc::from(lower_a.clone()),
                upper_a: Arc::from(lower_a),
                lower_b: Arc::from(lower_b.clone()),
                upper_b: Arc::from(lower_b),
                num_specs,
                current_dim: 2,
            };
            Self {
                graph,
                input,
                node_bounds,
                alpha: GraphAlphaState::new(),
                seed,
            }
        }

        fn request<'a>(
            &'a self,
            target_relu: &'a str,
            ordered_neurons: [usize; 2],
        ) -> CurrentLowerCutRequest<'a> {
            CurrentLowerCutRequest::new(
                &self.graph,
                &self.input,
                &self.alpha,
                &self.node_bounds,
                None,
                &self.seed,
                target_relu,
                ordered_neurons,
                Instant::now() + Duration::from_secs(30),
            )
        }

        fn fresh_support(&self, ordered_neurons: [usize; 2]) -> Octahedron2 {
            combined_row_octahedron_with_deadline(
                &self.graph,
                &self.input,
                &self.alpha,
                Some(&self.node_bounds),
                "pre",
                ordered_neurons[0],
                ordered_neurons[1],
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("fresh fixture support")
        }

        fn certificates(&self, ordered_neurons: [usize; 2]) -> Vec<ExactRelu2FacetCertificate> {
            let support = self.fresh_support(ordered_neurons);
            let checker =
                super::super::ExactRelu2Support::new(&support).expect("finite fixture support");
            [
                [-0.5_f32, -0.25, 1.0, 0.75],
                [0.375_f32, -0.625, -0.5, 1.25],
                [-0.125_f32, 0.875, 0.625, -0.375],
            ]
            .into_iter()
            .map(|normal| {
                checker
                    .certify_normal_certificate(normal)
                    .expect("finite exact fixture certificate")
            })
            .collect()
        }
    }

    #[test]
    fn production_gate_is_hard_disabled_and_returns_before_inspection() {
        let fixture = Fixture::new(1);
        let request = fixture.request("missing-target", [0, 0]);
        let result = build_certified_lower_cut_replay(
            CertifiedCutAuthorityGate::default(),
            request,
            &[],
            &[],
        )
        .expect("hard-disabled authority must be an exact no-op");
        assert!(result.is_none());
        assert!(!CertifiedCutAuthorityGate::default().is_enabled());
    }

    #[test]
    fn test_builder_binds_fixture_target_pair_support_and_lower_rows() {
        let fixture = Fixture::new(2);
        let certificates = fixture.certificates([0, 1]);
        let lambdas = vec![vec![0.25_f32, 0.5, 0.75], vec![1.0_f32, 0.0, 0.125]];
        let replay = build_certified_lower_cut_replay(
            CertifiedCutAuthorityGate::test_only_enabled(),
            fixture.request("relu", [0, 1]),
            &certificates,
            &lambdas,
        )
        .expect("valid live authority build")
        .expect("nonzero carrier");

        assert_eq!(replay.target_relu(), "relu");
        assert_eq!(replay.pre_node(), "pre");
        assert_eq!(replay.ordered_neurons(), [0, 1]);
        assert_eq!(replay.target_width(), 2);
        assert_eq!(replay.rows().len(), 2);
        assert_eq!(replay.certificate_count(), certificates.len());
        assert!(octahedron_bits_equal(
            replay.fresh_support(),
            certificates[0].support_domain()
        ));
        assert_eq!(
            replay.rows()[0].multipliers().collect::<Vec<_>>(),
            lambdas[0]
        );

        for (row, lambda_row) in replay.rows().iter().zip(&lambdas) {
            for coordinate in 0..2 {
                assert_channel_encloses_exact(
                    row.pre()[coordinate],
                    &certificates,
                    lambda_row,
                    |facet| facet.a[coordinate],
                );
                assert_channel_encloses_exact(
                    row.post()[coordinate],
                    &certificates,
                    lambda_row,
                    |facet| facet.a[coordinate + 2],
                );
            }
            assert_channel_encloses_exact(row.bias(), &certificates, lambda_row, |facet| -facet.b);
        }
    }

    #[test]
    fn support_or_order_mismatch_fails_closed() {
        let fixture = Fixture::new(1);
        let certificates = fixture.certificates([0, 1]);
        let err = build_certified_lower_cut_replay(
            CertifiedCutAuthorityGate::test_only_enabled(),
            fixture.request("relu", [1, 0]),
            &certificates[..1],
            &[vec![0.5]],
        )
        .expect_err("reversing an asymmetric ordered pair must invalidate support");
        assert!(matches!(err, NyError::SoundnessRefusal(_)));
    }

    #[test]
    fn certificate_from_another_input_domain_fails_closed() {
        let source = Fixture::new(1);
        let certificates = source.certificates([0, 1]);

        let mut current = Fixture::new(1);
        current.input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.25]).into_dyn(),
            arr1(&[0.75_f32, 0.5]).into_dyn(),
        )
        .expect("finite changed input");
        current.node_bounds = current
            .graph
            .collect_node_bounds_with_engine(&current.input, None)
            .expect("changed-domain bounds");

        let err = build_certified_lower_cut_replay(
            CertifiedCutAuthorityGate::test_only_enabled(),
            current.request("relu", [0, 1]),
            &certificates[..1],
            &[vec![0.5]],
        )
        .expect_err("a certificate from another input domain must not be reassociated");
        assert!(matches!(err, NyError::SoundnessRefusal(_)));
    }

    #[test]
    fn exact_relu_identity_and_lambda_shape_fail_closed() {
        let fixture = Fixture::new(1);
        let certificates = fixture.certificates([0, 1]);
        let not_relu = build_certified_lower_cut_replay(
            CertifiedCutAuthorityGate::test_only_enabled(),
            fixture.request("pre", [0, 1]),
            &certificates[..1],
            &[vec![0.5]],
        )
        .expect_err("a same-width affine node is not an exact ReLU target");
        assert!(matches!(not_relu, NyError::InvalidSpec(_)));

        for invalid in [f32::NAN, f32::INFINITY, -0.125] {
            let err = build_certified_lower_cut_replay(
                CertifiedCutAuthorityGate::test_only_enabled(),
                fixture.request("relu", [0, 1]),
                &certificates[..1],
                &[vec![invalid]],
            )
            .expect_err("invalid lambda must fail before authority publication");
            assert!(matches!(err, NyError::InvalidSpec(_)));
        }
    }

    #[test]
    fn all_zero_skips_fresh_production_and_mid_row_deadline_publishes_nothing() {
        let fixture = Fixture::new(2);
        let certificates = fixture.certificates([0, 1]);
        let producer_calls = Cell::new(0usize);
        let mut no_deadline = |_| Ok(());
        let zero = build_certified_lower_cut_replay_with(
            fixture.request("relu", [0, 1]),
            &certificates[..1],
            &[vec![0.0], vec![-0.0]],
            true,
            &mut no_deadline,
            |_, _| {
                producer_calls.set(producer_calls.get() + 1);
                Ok(fixture.fresh_support([0, 1]))
            },
        )
        .expect("all-zero carrier");
        assert!(zero.is_none());
        assert_eq!(producer_calls.get(), 0, "off0 must skip fresh CROWN");

        let support = fixture.fresh_support([0, 1]);
        let mut checker = |stage| {
            if stage == AuthorityDeadlineStage::AfterRowReduction(0) {
                Err(NyError::DeadlineExceeded(
                    "deterministic authority test deadline".into(),
                ))
            } else {
                Ok(())
            }
        };
        let err = build_certified_lower_cut_replay_with(
            fixture.request("relu", [0, 1]),
            &certificates[..1],
            &[vec![0.25], vec![0.5]],
            true,
            &mut checker,
            |_, _| Ok(support),
        )
        .expect_err("mid-row deadline must discard the local partial vector");
        assert!(matches!(err, NyError::DeadlineExceeded(_)));
    }

    #[test]
    fn failed_eft_selfcheck_refuses_before_support_or_publication() {
        let fixture = Fixture::new(1);
        let certificates = fixture.certificates([0, 1]);
        let producer_calls = Cell::new(0usize);
        let mut no_deadline = |_| Ok(());
        let err = build_certified_lower_cut_replay_with(
            fixture.request("relu", [0, 1]),
            &certificates[..1],
            &[vec![0.5]],
            false,
            &mut no_deadline,
            |_, _| {
                producer_calls.set(producer_calls.get() + 1);
                Ok(fixture.fresh_support([0, 1]))
            },
        )
        .expect_err("a failed EFT self-check must refuse the entire local carrier");
        assert!(matches!(err, NyError::SoundnessRefusal(_)));
        assert_eq!(
            producer_calls.get(),
            0,
            "failed EFT authorization must refuse before fresh support work"
        );
    }

    #[test]
    fn directed_reducer_encloses_adversarial_exact_rational_sum() {
        // Cancellation plus subnormal products exercises both TwoSum residuals
        // and the final directed f64->f32 source-error conversion.
        let operands = [
            (f32::MAX, 0.5_f32),
            (f32::MAX, -0.5_f32),
            (f32::from_bits(1), f32::from_bits(1)),
            (1.000_000_1_f32, 1.000_000_1_f32),
            (1.0_f32, -1.0_f32),
        ];
        let channel = directed_reduce(
            operands
                .iter()
                .map(|&(left, right)| (f64::from(left), f64::from(right))),
        )
        .expect("finite adversarial reduction");
        let exact = operands
            .iter()
            .fold(BigRational::from_integer(0.into()), |acc, &(l, r)| {
                acc + rational_f32(l) * rational_f32(r)
            });
        assert_channel_contains_rational(channel, &exact);
    }

    #[test]
    fn directed_reducer_matches_many_exact_rational_oracles() {
        let lambdas = [
            0.0_f32,
            f32::from_bits(1),
            2.0_f32.powi(-80),
            0.125,
            0.999_999_94,
            1.0,
            1024.0,
            1.0e10,
        ];
        let coefficients = [
            -1.0e10_f32,
            -1024.0,
            -1.000_000_1,
            -f32::from_bits(1),
            0.0,
            f32::from_bits(1),
            0.999_999_94,
            1.0e10,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _case in 0..256 {
            let count = 1 + (next_xorshift(&mut state) as usize % MAX_AUTHORITY_FACETS);
            let operands: Vec<(f32, f32)> = (0..count)
                .map(|_| {
                    let lambda = lambdas[next_xorshift(&mut state) as usize % lambdas.len()];
                    let coefficient =
                        coefficients[next_xorshift(&mut state) as usize % coefficients.len()];
                    (lambda, coefficient)
                })
                .collect();
            let channel = directed_reduce(
                operands
                    .iter()
                    .map(|&(left, right)| (f64::from(left), f64::from(right))),
            )
            .expect("bounded finite exact-oracle reduction");
            let exact = operands.iter().fold(
                BigRational::from_integer(0.into()),
                |acc, &(left, right)| acc + rational_f32(left) * rational_f32(right),
            );
            assert_channel_contains_rational(channel, &exact);
        }
    }

    fn next_xorshift(state: &mut u64) -> u64 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        value
    }

    fn rational_f32(value: f32) -> BigRational {
        BigRational::from_float(value).expect("finite f32 is an exact dyadic")
    }

    fn assert_channel_encloses_exact(
        channel: DirectedLowerChannel,
        certificates: &[ExactRelu2FacetCertificate],
        lambdas: &[f32],
        coefficient: impl Fn(super::super::Facet) -> f32,
    ) {
        let exact = certificates.iter().zip(lambdas).fold(
            BigRational::from_integer(0.into()),
            |acc, (certificate, &lambda)| {
                acc + rational_f32(lambda) * rational_f32(coefficient(certificate.facet()))
            },
        );
        assert_channel_contains_rational(channel, &exact);
    }

    fn assert_channel_contains_rational(channel: DirectedLowerChannel, exact: &BigRational) {
        let stored = rational_f32(channel.value());
        let error = rational_f32(channel.reduction_error());
        assert!(
            stored.clone() - error.clone() <= exact.clone() && exact.clone() <= stored + error,
            "directed channel does not enclose exact sum: channel={channel:?}, exact={exact}"
        );
    }
}
