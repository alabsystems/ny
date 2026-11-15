// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Two same-source local certificates and a predeclared box/prefix portfolio. Arithmetic
//! kernels have no authority without this immutable joint call and owner lease.
use super::super::local_box_l1::{self, LocalBoxL1Arithmetic, LocalBoxL1Quote};
use super::*;
use ny_core::{ResidentLocalL1LowerCutCarrier, ResidentTwoLocalL1LowerCutCarrier};

#[path = "root_call_two_local_l1/full_head_advice.rs"]
mod full_head_advice;

#[path = "root_call_two_local_l1/head_box_floor.rs"]
mod head_box_floor;
#[path = "root_call_two_local_l1/head_source.rs"]
mod head_source;

const LOCAL_EXTRA_BYTES: usize = 4 << 20;
// Both complete kernel quotes and the full source phase are separate. This
// covers33 channels, fixed joint custody/generation and bounded observations.
const LOCAL_FIXED_WORK: u64 = 2 << 20;
const LOCAL_DIAGNOSTIC_BYTES: usize = 8192;
// A separate, conservative comparison envelope: both complete one-row phases
// plus unchanged kernels. No shared materialization discount or batch API.
const COMPARISON_FIXED_WORK: u64 = 4 << 20;

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeadBoxPlan {
    quote: head_box_floor::Quote,
    threshold: f32,
    keep_single3_prefix: bool,
}

pub(crate) struct RawTwoLocalL1RootCutQuote<'a> {
    base: RawRootCutQuote<'a>,
    seal: ResidentCutCallSeal,
    active: [usize; 2],
    neurons: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    multipliers: [f32; 2],
    kernel: [LocalBoxL1Quote; 2],
    compare_single3: bool,
    source_shape: [usize; 3],
    full_head_advice: Option<full_head_advice::Quote>,
    head_box: Option<HeadBoxPlan>,
}
impl RawTwoLocalL1RootCutQuote<'_> {
    pub(crate) fn work(&self) -> u64 {
        self.base.work
    }
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.base.scratch
    }
    pub(crate) fn candidate_count(&self) -> usize {
        1 + usize::from(self.keep_single3_prefix()) + usize::from(self.head_box.is_some())
    }
    fn keep_single3_prefix(&self) -> bool {
        self.compare_single3 && self.head_box.is_none_or(|p| p.keep_single3_prefix)
    }
    pub(crate) fn uses_head_box_floor(&self) -> bool {
        self.head_box.is_some()
    }
    /// Declare both possible branches before numerical entry. The same source
    /// box proof may finish on a strict query proof; otherwise a complete bundle
    /// prefix is required. Keep the single3 prefix too only when the aggregate fits.
    pub(crate) fn try_enable_head_box_floor(
        &mut self,
        threshold: f32,
        limits: RawRootCutLimits,
    ) -> bool {
        if self.head_box.is_some()
            || !ny_core::is_crown_coeff_safe(threshold)
            || self.base.old.0 > threshold
            || threshold >= self.base.old.1
            || limits.max_nodes < self.base.extraction.max_nodes
            || Instant::now() >= self.base.deadline
        {
            return false;
        }
        let [e, d, o] = self.source_shape;
        let Some(floor) = head_box_floor::quote(e, d, o) else {
            return false;
        };
        let Some(scratch) = self.base.scratch.checked_add(floor.scratch_bytes) else {
            return false;
        };
        if scratch > limits.max_scratch_bytes {
            return false;
        }
        let Some(phase) = phase_work(self.base.full_quote, 1, 1) else {
            return false;
        };
        let fixed = if self.compare_single3 {
            COMPARISON_FIXED_WORK
        } else {
            LOCAL_FIXED_WORK
        };
        let work = |keep: bool| {
            let calls = 1 + u64::from(keep);
            phase
                .checked_mul(calls)?
                .checked_add(self.kernel[0].work)?
                .checked_add(self.kernel[1].work)?
                .checked_add(fixed)?
                .checked_add(EXECUTION_OBSERVATION_WORK.checked_mul(calls)?)?
                .checked_add(self.full_head_advice.map_or(0, |q| q.work))?
                .checked_add(floor.work)
        };
        let selected = [self.compare_single3, false].into_iter().find_map(|keep| {
            let price = work(keep)?;
            (price <= limits.max_work).then_some((keep, price))
        });
        let Some((keep_single3_prefix, work)) = selected else {
            return false;
        };
        self.base.work = work;
        self.base.scratch = scratch;
        self.head_box = Some(HeadBoxPlan {
            quote: floor,
            threshold,
            keep_single3_prefix,
        });
        self.base.candidate_count = self.candidate_count();
        true
    }
    /// Multiplier advice is independent of prefix comparison and box proof.
    /// Add its complete charge to the existing predeclared plan atomically;
    /// a failed shape-only upgrade leaves every prior branch and charge intact.
    /// The owner includes geometry/requote work before choosing these limits.
    pub(crate) fn try_enable_full_head_advice(&mut self, limits: RawRootCutLimits) -> bool {
        if self.full_head_advice.is_some()
            || !valid_selection(
                self.active,
                self.neurons,
                self.weights,
                self.source_shape[1],
            )
            || Instant::now() >= self.base.deadline
        {
            return false;
        }
        let [e, d, o] = self.source_shape;
        let Some(advice) = full_head_advice::quote(e, d, o) else {
            return false;
        };
        let Some(work) = self.base.work.checked_add(advice.work) else {
            return false;
        };
        let Some(scratch) = self.base.scratch.checked_add(advice.scratch_bytes) else {
            return false;
        };
        if work > limits.max_work
            || scratch > limits.max_scratch_bytes
            || limits.max_nodes < self.base.extraction.max_nodes
        {
            return false;
        }
        self.base.work = work;
        self.base.scratch = scratch;
        self.full_head_advice = Some(advice);
        true
    }
    pub(crate) fn uses_full_head_advice(&self) -> bool {
        self.full_head_advice.is_some()
    }
    /// Pure upgrade of an already authenticated shape quote. The owner selects
    /// this only for exactly nine or ten ranked cells, before geometry/numerics.
    /// Group order remains canonical: six/seven complement rows, then three.
    pub(crate) fn with_single3_comparison(mut self, limits: RawRootCutLimits) -> Option<Self> {
        if !matches!(self.active, [6 | 7, 3])
            || self.compare_single3
            || self.head_box.is_some()
            || Instant::now() >= self.base.deadline
        {
            return None;
        }
        let work = phase_work(self.base.full_quote, 1, 1)?
            .checked_mul(2)?
            .checked_add(self.kernel[0].work)?
            .checked_add(self.kernel[1].work)?
            .checked_add(COMPARISON_FIXED_WORK)?
            .checked_add(EXECUTION_OBSERVATION_WORK.checked_mul(2)?)?
            .checked_add(self.full_head_advice.map_or(0, |q| q.work))?;
        // Both calls are serial and use one immutable extraction. All fixed
        // transports/results coexist inside the existing 4 MiB metadata reserve.
        let fixed = self.kernel[0]
            .scratch_bytes
            .checked_add(self.kernel[1].scratch_bytes)?
            .checked_add(2usize.checked_mul(size_of::<LocalBoxL1Arithmetic<'_>>())?)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<FreshTwoLocalSupport<'_, '_>>())?
            .checked_add(size_of::<TwoLocalContext<'_, '_>>())?
            .checked_add(size_of::<TwoLocalCarrier<'_>>())?
            .checked_add(size_of::<Single3Carrier<'_>>())?
            .checked_add(size_of::<CandidateScalePair<'_>>())?
            .checked_add(size_of::<ResidentTwoLocalL1LowerCutCarrier>())?
            .checked_add(size_of::<ResidentLocalL1LowerCutCarrier>())?
            .checked_add(2usize.checked_mul(size_of::<BoundResidentLowerResult<'_>>())?)?
            .checked_add(size_of::<CompletedRawTwoLocalL1RootCut>())?
            .checked_add(size_of::<FullHeadMultiplierAdvice>())?
            .checked_add(LOCAL_DIAGNOSTIC_BYTES.checked_mul(2)?)?;
        if fixed > LOCAL_EXTRA_BYTES
            || work > limits.max_work
            || self.base.scratch > limits.max_scratch_bytes
            || limits.max_nodes < self.base.extraction.max_nodes
        {
            return None;
        }
        self.base.work = work;
        self.base.candidate_count = 2;
        self.compare_single3 = true;
        Some(self)
    }
}
pub(crate) struct CompletedRawTwoLocalL1RootCut {
    bounds: (f32, f32),
    raw_candidate_lower: f32,
    bundle_candidate_lower: Option<f32>,
    head_box_candidate_lower: Option<f32>,
    single3_candidate_lower: Option<f32>,
    full_head_advice: Option<full_head_advice::Decision>,
    original_multipliers: [f32; 2],
    active: [usize; 2],
    neurons: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    multipliers: [f32; 2],
    support_bounds: [f32; 2],
    radii: [f64; 2],
    sign_counts: [usize; 2],
    original_local_bounds: [f64; 2],
    pre_lower: [[f32; 8]; 2],
    pre_upper: [[f32; 8]; 2],
}
/// Observation only: neither scores nor floating support planes certify a bound.
pub(crate) struct FullHeadMultiplierAdvice {
    pub(crate) original: [f32; 2],
    pub(crate) chosen: [f32; 2],
    pub(crate) branch_upper: [f32; 2],
    pub(crate) original_score: f64,
    pub(crate) chosen_score: f64,
    pub(crate) score_evaluations: usize,
    pub(crate) full_scans: usize,
    pub(crate) support_planes: usize,
    pub(crate) lp_candidates: usize,
}
impl CompletedRawTwoLocalL1RootCut {
    pub(crate) fn bounds(&self) -> (f32, f32) {
        self.bounds
    }
    /// Best completed certified candidate before merging the original bound.
    /// This scalar does not bypass merge or the final root acceptance barriers.
    pub(crate) fn raw_candidate_lower(&self) -> f32 {
        self.raw_candidate_lower
    }
    /// Actual bundle fold before any candidate choice or original-bound merge.
    pub(crate) fn bundle_candidate_lower(&self) -> Option<f32> {
        self.bundle_candidate_lower
    }
    pub(crate) fn head_box_candidate_lower(&self) -> Option<f32> {
        self.head_box_candidate_lower
    }
    pub(crate) fn candidate_count(&self) -> usize {
        self.cut_calls() + usize::from(self.head_box_candidate_lower.is_some())
    }
    /// Present only when the declared transaction actually completes this prefix.
    pub(crate) fn single3_candidate_lower(&self) -> Option<f32> {
        self.single3_candidate_lower
    }
    /// Advisory scores never carry bound or acceptance authority.
    pub(crate) fn multiplier_advice(&self) -> Option<FullHeadMultiplierAdvice> {
        self.full_head_advice
            .as_ref()
            .map(|v| FullHeadMultiplierAdvice {
                original: self.original_multipliers,
                chosen: v.multipliers,
                branch_upper: v.branch_upper,
                original_score: v.original_score,
                chosen_score: v.chosen_score,
                score_evaluations: v.score_evaluations,
                full_scans: v.full_scans,
                support_planes: v.support_planes,
                lp_candidates: v.lp_candidates,
            })
    }
    pub(crate) fn support_calls(&self) -> usize {
        0
    }
    pub(crate) fn cut_calls(&self) -> usize {
        usize::from(self.single3_candidate_lower.is_some())
            + usize::from(self.bundle_candidate_lower.is_some())
    }
    pub(crate) fn certificates(&self) -> usize {
        2
    }
    pub(crate) fn active_lens(&self) -> [usize; 2] {
        self.active
    }
    pub(crate) fn ordered_neurons(&self) -> [[usize; 8]; 2] {
        self.neurons
    }
    pub(crate) fn weights(&self) -> [[f32; 8]; 2] {
        self.weights
    }
    pub(crate) fn multipliers(&self) -> [f32; 2] {
        self.multipliers
    }
    pub(crate) fn support_bounds(&self) -> [f32; 2] {
        self.support_bounds
    }
    pub(crate) fn bn_radii(&self) -> [f64; 2] {
        self.radii
    }
    pub(crate) fn sign_counts(&self) -> [usize; 2] {
        self.sign_counts
    }
    pub(crate) fn original_local_bounds(&self) -> [f64; 2] {
        self.original_local_bounds
    }
    pub(crate) fn pre_lower(&self) -> &[[f32; 8]; 2] {
        &self.pre_lower
    }
    pub(crate) fn pre_upper(&self) -> &[[f32; 8]; 2] {
        &self.pre_upper
    }
}
fn valid_selection(m: [usize; 2], n: [[usize; 8]; 2], w: [[f32; 8]; 2], width: usize) -> bool {
    if m.iter().any(|m| !(1..=8).contains(m)) {
        return false;
    }
    for g in 0..2 {
        if !n[g][..m[g]].windows(2).all(|v| v[0] < v[1])
            || n[g][m[g] - 1] >= width
            || n[g][m[g]..].iter().any(|v| *v != 0)
            || w[g][m[g]..].iter().any(|v| v.to_bits() != 0)
            || w[g][..m[g]].iter().any(|v| {
                v.to_bits() < (1f32 / 256.).to_bits()
                    || v.to_bits() > 1f32.to_bits()
                    || (ny_core::f32_to_f64_exact(*v) * 256.).fract() != 0.
            })
        {
            return false;
        }
    }
    // Padding zeros are not active neurons; neuron0 in the other group is legal.
    !n[0][..m[0]].iter().any(|i| n[1][..m[1]].contains(i))
}
fn same_weights(a: [[f32; 8]; 2], b: [[f32; 8]; 2]) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(a, b)| a.to_bits() == b.to_bits())
}
fn same_multipliers(a: [f32; 2], b: [f32; 2]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_raw_two_local_l1_root_cut<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    active: [usize; 2],
    neurons: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    multipliers: [f32; 2],
    limits: RawRootCutLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<RawTwoLocalL1RootCutQuote<'a>> {
    let mut guard = Poll {
        end: deadline,
        callback: poll,
        alive: true,
    };
    if !guard.check()
        || limits.max_nodes == 0
        || limits.max_nodes > 128
        || !ny_core::is_crown_coeff_safe(old.0)
        || !ny_core::is_crown_coeff_safe(old.1)
        || old.0 > old.1
        || multipliers
            .iter()
            .any(|v| !ny_core::is_crown_coeff_safe(*v) || *v <= 0.)
        || target.len() > 1024
        || objective.is_empty()
        || objective.len() > 8192
        || objective.iter().any(|x| !ny_core::is_crown_coeff_safe(*x))
    {
        return None;
    }
    let extraction = ExtractionLimits {
        max_nodes: limits.max_nodes,
        max_caller_bytes: limits.max_scratch_bytes.min(256 << 20),
    };
    let relu = graph.node(target)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return None;
    }
    let pre_node = relu.require_unary_input().ok()?;
    if pre_node.len() > 1024 {
        return None;
    }
    let pre = bounds.get(pre_node)?;
    let post = bounds.get(target)?;
    if pre.len() != post.len() || !valid_selection(active, neurons, weights, pre.len()) {
        return None;
    }
    // The genuine source owner admits only structurally authenticated raw
    // authored-f32 graphs (unit Gemm alpha/beta, no affine merging/custom lowering).
    // Select this original Linear, never its charged BN descriptor or a cache.
    let node = graph.node(pre_node)?;
    let linear = match node.layer() {
        Layer::Linear(l) => l,
        Layer::BatchNorm(_) => {
            let name = node.require_unary_input().ok()?;
            if name.len() > 1024 {
                return None;
            }
            let Layer::Linear(l) = graph.node(name)?.layer() else {
                return None;
            };
            l
        }
        _ => return None,
    };
    if linear.out_features() != pre.len() {
        return None;
    }
    let kernel = [local_box_l1::quote(linear.in_features(), linear.out_features())?; 2];
    let mut saw = false;
    let mut current = graph.output_name();
    for _ in 0..limits.max_nodes {
        if !guard.check() || current.len() > 1024 {
            return None;
        }
        if current == crate::network::NETWORK_INPUT {
            break;
        }
        saw |= current == target;
        current = graph.node(current)?.require_unary_input().ok()?;
    }
    if !saw || current != crate::network::NETWORK_INPUT {
        return None;
    }
    // Shape quote only. Prefix-shaped bytes retain the existing conservative
    // cache/live envelope; there is no prefix numeric work or support API call.
    let prefix_quote = quote_charged_raw_bn_chain(
        graph,
        input,
        pre_node,
        bounds,
        bounds,
        alpha,
        extraction,
        deadline,
        &mut || guard.check(),
    )?;
    let full_quote = quote_charged_raw_bn_chain(
        graph,
        input,
        graph.output_name(),
        bounds,
        bounds,
        alpha,
        extraction,
        deadline,
        &mut || guard.check(),
    )?;
    if prefix_quote.output_dim != pre.len() || full_quote.output_dim != objective.len() {
        return None;
    }
    let scratch = phase_bytes(prefix_quote, 4)?
        .max(phase_bytes(full_quote, 1)?)
        .checked_add(retained_prefix_cache_bytes(prefix_quote)?)?
        .checked_add(ADVICE_BYTES)?
        .checked_add(EXECUTION_OBSERVATION_BYTES)?
        .checked_add(LOCAL_EXTRA_BYTES)?;
    let work = phase_work(full_quote, 1, 1)?
        .checked_add(kernel[0].work)?
        .checked_add(kernel[1].work)?
        .checked_add(LOCAL_FIXED_WORK)?
        .checked_add(EXECUTION_OBSERVATION_WORK)?;
    if size_of::<ResidentTwoLocalL1LowerCutCarrier>() > 640
        || kernel[0]
            .scratch_bytes
            .checked_add(kernel[1].scratch_bytes)?
            .checked_add(2usize.checked_mul(size_of::<LocalBoxL1Arithmetic<'_>>())?)?
            .checked_add(size_of::<RawTwoLocalL1RootCutQuote<'_>>())?
            .checked_add(size_of::<FreshTwoLocalSupport<'_, '_>>())?
            .checked_add(size_of::<TwoLocalContext<'_, '_>>())?
            .checked_add(size_of::<TwoLocalCarrier<'_>>())?
            .checked_add(size_of::<ResidentTwoLocalL1LowerCutCarrier>())?
            .checked_add(size_of::<CandidateScalePair<'_>>())?
            .checked_add(size_of::<CompletedRawTwoLocalL1RootCut>())?
            .checked_add(size_of::<FullHeadMultiplierAdvice>())?
            .checked_add(LOCAL_DIAGNOSTIC_BYTES)?
            > LOCAL_EXTRA_BYTES
        || work > limits.max_work
        || scratch > limits.max_scratch_bytes
        || !guard.check()
    {
        return None;
    }
    let base = RawRootCutQuote {
        graph,
        input,
        bounds,
        alpha,
        objective,
        old,
        target,
        pre_node,
        pair: [neurons[0][0], neurons[0][1]],
        lambda_scale: multipliers[0],
        candidate_count: 1,
        extraction,
        prefix_quote,
        full_quote,
        work,
        scratch,
        deadline,
    };
    Some(RawTwoLocalL1RootCutQuote {
        base,
        seal: ResidentCutCallSeal([0xB8]),
        active,
        neurons,
        weights,
        multipliers,
        kernel,
        compare_single3: false,
        source_shape: [linear.in_features(), linear.out_features(), objective.len()],
        full_head_advice: None,
        head_box: None,
    })
}

struct FreshTwoLocalSupport<'q, 'a> {
    seal: &'q ResidentCutCallSeal,
    arithmetic: [LocalBoxL1Arithmetic<'a>; 2],
}
struct TwoLocalContext<'q, 'a> {
    quote: &'q RawTwoLocalL1RootCutQuote<'a>,
    base: ResidentCutCallContext<'a>,
    target_activation: usize,
    activation_count: usize,
    width: usize,
}
struct TwoLocalCarrier<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    active: [usize; 2],
    neurons: [[usize; 8]; 2],
    weights: [[f32; 8]; 2],
    bounds: [f32; 2],
    multipliers: [f32; 2],
    selected_scale: Option<&'c CandidateScalePair<'c>>,
    transport: ResidentTwoLocalL1LowerCutCarrier,
}
// Private selection authority for one declared advice policy. Scores remain
// advisory; the fresh certificate and original-product carrier prove the cut.
struct CandidateScalePair<'c> {
    seal: &'c ResidentCutCallSeal,
    quote: full_head_advice::Quote,
    original_multipliers: [f32; 2],
    decision: full_head_advice::Decision,
}
struct Single3Carrier<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    neurons: [usize; 8],
    weights: [f32; 8],
    bound: f32,
    transport: ResidentLocalL1LowerCutCarrier,
}
impl<'q, 'a> TwoLocalContext<'q, 'a> {
    fn new(q: &'q RawTwoLocalL1RootCutQuote<'a>, base: ResidentCutCallContext<'a>) -> Result<Self> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("two-local L1 IEEE".into()));
        }
        let (target_activation, widths) = base.validate_resident_snapshot()?;
        LowerSeedRef::from_seed(base.seed).validate()?;
        let width = widths[target_activation];
        let b = &q.base;
        if !std::ptr::eq(base.graph, b.graph)
            || !std::ptr::eq(base.input, b.input)
            || !std::ptr::eq(base.node_bounds, b.bounds)
            || b.alpha.is_some_and(|a| !std::ptr::eq(base.alpha_state, a))
            || base.target_relu != b.target
            || base.ordered_neurons != [q.neurons[0][0], q.neurons[0][1]]
            || base.deadline != b.deadline
            || base.seed.num_specs != 1
            || base.seed.current_dim != b.objective.len()
            || !slice_bits_equal(Some(b.objective), &base.seed.lower_a)
            || !slice_bits_equal(Some(b.objective), &base.seed.upper_a)
            || base.seed.lower_b.len() != 1
            || base.seed.upper_b.len() != 1
            || base.seed.lower_b[0].to_bits() != 0
            || base.seed.upper_b[0].to_bits() != 0
            || base.beta_signed.iter().flatten().any(|v| *v != 0.)
            || !valid_selection(q.active, q.neurons, q.weights, width)
        {
            return Err(NyError::SoundnessRefusal(
                "two-local L1 original snapshot differs".into(),
            ));
        }
        Ok(Self {
            quote: q,
            base,
            target_activation,
            activation_count: widths.len(),
            width,
        })
    }
    fn validate_scale(&self, choice: &CandidateScalePair<'_>) -> Result<()> {
        let q = self.quote;
        let d = &choice.decision;
        let original_reference =
            (0..2).all(|g| d.multipliers[g].to_bits() == q.multipliers[g].to_bits());
        if !std::ptr::eq(choice.seal, &raw const q.seal)
            || q.full_head_advice != Some(choice.quote)
            || !valid_selection(q.active, q.neurons, q.weights, self.width)
            || (0..2).any(|g| {
                let value = d.multipliers[g];
                let cap = d.branch_upper[g];
                choice.original_multipliers[g].to_bits() != q.multipliers[g].to_bits()
                    || !ny_core::is_crown_coeff_safe(value)
                    || value <= 0.
                    || f64::from(value) < f64::from(q.multipliers[g]) / 16.
                    || value > q.multipliers[g]
                    || !cap.is_finite() || cap < 0. || cap > q.multipliers[g]
                    // Empty rectangles may have a zero diagnostic cap. The
                    // separately scored original pair never claims membership.
                    || (!original_reference && (cap <= 0. || value > cap))
            })
            || !d.original_score.is_finite()
            || !d.chosen_score.is_finite()
            || d.chosen_score < d.original_score
            || d.full_scans != d.score_evaluations
            || !(1..=17).contains(&d.score_evaluations)
            || d.support_planes > 16
            || d.support_planes + 1 != d.score_evaluations
            || d.lp_candidates > 4120
            || (d.support_planes == 0 && (!original_reference || d.lp_candidates != 0))
        {
            return Err(NyError::SoundnessRefusal(
                "foreign full-head multiplier pair".into(),
            ));
        }
        Ok(())
    }
    fn choose_scale<'c>(
        &'c self,
        fresh: &FreshTwoLocalSupport<'_, '_>,
        guard: &mut Poll<'_>,
    ) -> Option<CandidateScalePair<'c>> {
        let q = self.quote;
        let paid = q.full_head_advice?;
        if !std::ptr::eq(fresh.seal, &raw const q.seal)
            || fresh.arithmetic.iter().enumerate().any(|(g, f)| {
                !f.matches_source(q.base.graph, q.base.input, q.base.bounds, q.base.target)
                    || f.neurons() != &q.neurons[g][..q.active[g]]
                    || f.weights().len() != q.active[g]
                    || f.weights()
                        .iter()
                        .zip(q.weights[g])
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                    || f.quote() != q.kernel[g]
                    || f.sign_count() != 1usize << (q.active[g] - 1)
                    || !ny_core::is_crown_coeff_safe(f.bound())
                    || f.bound() < 0.
            })
        {
            return None;
        }
        let decision = full_head_advice::choose(self, fresh, paid, guard)?;
        let choice = CandidateScalePair {
            seal: &self.quote.seal,
            quote: paid,
            original_multipliers: q.multipliers,
            decision,
        };
        self.validate_scale(&choice).ok()?;
        Some(choice)
    }
    fn build<'c>(&'c self, fresh: &FreshTwoLocalSupport<'_, '_>) -> Result<TwoLocalCarrier<'c>> {
        self.build_with_scale(fresh, None)
    }
    fn build_with_scale<'c>(
        &'c self,
        fresh: &FreshTwoLocalSupport<'_, '_>,
        selected_scale: Option<&'c CandidateScalePair<'c>>,
    ) -> Result<TwoLocalCarrier<'c>> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal(
                "two-local L1 reduction IEEE".into(),
            ));
        }
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let q = self.quote;
        let b = &q.base;
        let multipliers = if let Some(choice) = selected_scale {
            self.validate_scale(choice)?;
            choice.decision.multipliers
        } else {
            if q.full_head_advice.is_some() {
                return Err(NyError::SoundnessRefusal(
                    "missing full-head multiplier choice".into(),
                ));
            }
            q.multipliers
        };
        if !std::ptr::eq(fresh.seal, &raw const q.seal) {
            return Err(NyError::SoundnessRefusal(
                "foreign two-local L1 certificate seal".into(),
            ));
        }
        for g in 0..2 {
            let f = &fresh.arithmetic[g];
            if !f.matches_source(b.graph, b.input, b.bounds, b.target)
                || f.neurons() != &q.neurons[g][..q.active[g]]
                || f.weights().len() != q.active[g]
                || f.weights()
                    .iter()
                    .zip(q.weights[g])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                || f.sign_count() != 1usize << (q.active[g] - 1)
                || f.quote() != q.kernel[g]
                || !ny_core::is_crown_coeff_safe(f.bound())
                || f.bound() < 0.
            {
                return Err(NyError::SoundnessRefusal(
                    "foreign two-local L1 group certificate".into(),
                ));
            }
        }
        let zero = ResidentLowerCutChannel::try_new(0., 0.)?;
        let mut pre = [[zero; 8]; 2];
        let mut post = pre;
        for g in 0..2 {
            let channel = |v: f32| {
                transport_channel(directed_reduce(std::iter::once((
                    ny_core::f32_to_f64_exact(multipliers[g]),
                    ny_core::f32_to_f64_exact(v),
                )))?)
            };
            for i in 0..q.active[g] {
                pre[g][i] = channel(-q.weights[g][i])?;
                post[g][i] = channel(2. * q.weights[g][i])?;
            }
        }
        let bounds = [fresh.arithmetic[0].bound(), fresh.arithmetic[1].bound()];
        // Both factors are ORIGINAL f32 values. Never sum rounded group biases.
        let bias = transport_channel(directed_reduce((0..2).map(|g| {
            (
                ny_core::f32_to_f64_exact(multipliers[g]),
                ny_core::f32_to_f64_exact(-bounds[g]),
            )
        }))?)?;
        let transport = ResidentTwoLocalL1LowerCutCarrier::try_new(
            self.target_activation,
            self.width,
            q.active,
            q.neurons,
            multipliers,
            pre,
            post,
            bias,
            self.base.deadline,
        )?;
        transport.validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        Ok(TwoLocalCarrier {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            active: q.active,
            neurons: q.neurons,
            weights: q.weights,
            bounds,
            multipliers,
            selected_scale,
            transport,
        })
    }
    fn build_single3<'c>(
        &'c self,
        fresh: &FreshTwoLocalSupport<'_, '_>,
    ) -> Result<Single3Carrier<'c>> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal(
                "single3 comparison reduction IEEE".into(),
            ));
        }
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let q = self.quote;
        let b = &q.base;
        let f = &fresh.arithmetic[1];
        if !q.compare_single3
            || !matches!(q.active, [6 | 7, 3])
            || !std::ptr::eq(fresh.seal, &raw const q.seal)
            || !f.matches_source(b.graph, b.input, b.bounds, b.target)
            || f.neurons() != &q.neurons[1][..3]
            || f.weights().len() != 3
            || f.weights()
                .iter()
                .zip(q.weights[1])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || f.sign_count() != 4
            || f.quote() != q.kernel[1]
            || !ny_core::is_crown_coeff_safe(f.bound())
            || f.bound() < 0.
        {
            return Err(NyError::SoundnessRefusal(
                "foreign single3 local certificate".into(),
            ));
        }
        let channel = |v: f32| {
            transport_channel(directed_reduce(std::iter::once((
                ny_core::f32_to_f64_exact(q.multipliers[1]),
                ny_core::f32_to_f64_exact(v),
            )))?)
        };
        let zero = ResidentLowerCutChannel::try_new(0., 0.)?;
        let mut pre = [zero; 8];
        let mut post = pre;
        for i in 0..3 {
            pre[i] = channel(-q.weights[1][i])?;
            post[i] = channel(2. * q.weights[1][i])?;
        }
        // Reduce the original single product independently. Never subtract a
        // rounded complement bias/radius from the aggregate bundle channel.
        let transport = ResidentLocalL1LowerCutCarrier::try_new(
            self.target_activation,
            self.width,
            3,
            q.neurons[1],
            q.multipliers[1],
            pre,
            post,
            channel(-f.bound())?,
            self.base.deadline,
        )?;
        transport.validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        Ok(Single3Carrier {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            neurons: q.neurons[1],
            weights: q.weights[1],
            bound: f.bound(),
            transport,
        })
    }
    fn run_single3<'c>(
        &'c self,
        c: &Single3Carrier<'_>,
        gpu: &dyn GpuCrownBackward,
    ) -> Result<BoundResidentLowerResult<'c>> {
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let q = self.quote;
        if !q.compare_single3
            || !matches!(q.active, [6 | 7, 3])
            || !std::ptr::eq(c.seal, &raw const self.base.seal)
            || c.generations != self.base.generations.get()
            || c.neurons != q.neurons[1]
            || c.weights
                .iter()
                .zip(q.weights[1])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || !ny_core::is_crown_coeff_safe(c.bound)
            || c.bound < 0.
            || !gpu.provides_bounded_local_l1_lower_cut()
            || c.transport.active_len() != 3
            || *c.transport.ordered_neurons() != c.neurons
            || c.transport.multiplier().to_bits() != q.multipliers[1].to_bits()
            || c.transport.target_activation() != self.target_activation
        {
            return Err(NyError::SoundnessRefusal(
                "foreign/stale single3 comparison carrier".into(),
            ));
        }
        c.transport
            .validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        self.base.advance_frontier_generation()?;
        let result = gpu.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            self.base.segments,
            self.base.seed,
            self.base.resident_input_lower,
            self.base.resident_input_upper,
            &c.transport,
            self.base.deadline,
        );
        self.base.advance_frontier_generation()?;
        let lower = result?;
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforePublish)?;
        if !ny_core::is_crown_coeff_safe(lower) {
            return Err(NyError::SoundnessRefusal(
                "unsafe single3 comparison lower".into(),
            ));
        }
        Ok(BoundResidentLowerResult {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            lower,
        })
    }
    fn run<'c>(
        &'c self,
        c: &TwoLocalCarrier<'_>,
        gpu: &dyn GpuCrownBackward,
    ) -> Result<BoundResidentLowerResult<'c>> {
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let expected_multipliers = if let Some(choice) = c.selected_scale {
            self.validate_scale(choice)?;
            choice.decision.multipliers
        } else {
            if self.quote.full_head_advice.is_some() {
                return Err(NyError::SoundnessRefusal(
                    "missing advised carrier choice".into(),
                ));
            }
            self.quote.multipliers
        };
        if !std::ptr::eq(c.seal, &raw const self.base.seal)
            || c.generations != self.base.generations.get()
            || c.active != self.quote.active
            || c.neurons != self.quote.neurons
            || !same_weights(c.weights, self.quote.weights)
            || c.bounds
                .iter()
                .any(|v| !ny_core::is_crown_coeff_safe(*v) || *v < 0.)
            || !gpu.provides_bounded_two_local_l1_lower_cut()
            || *c.transport.active_lens() != c.active
            || *c.transport.ordered_neurons() != c.neurons
            || !same_multipliers(c.multipliers, expected_multipliers)
            || !same_multipliers(*c.transport.multipliers(), expected_multipliers)
            || c.transport.target_activation() != self.target_activation
        {
            return Err(NyError::SoundnessRefusal(
                "foreign/stale two-local L1 carrier".into(),
            ));
        }
        c.transport
            .validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        self.base.advance_frontier_generation()?;
        let result = gpu.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            self.base.segments,
            self.base.seed,
            self.base.resident_input_lower,
            self.base.resident_input_upper,
            &c.transport,
            self.base.deadline,
        );
        self.base.advance_frontier_generation()?;
        let lower = result?;
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforePublish)?;
        if !ny_core::is_crown_coeff_safe(lower) {
            return Err(NyError::SoundnessRefusal("unsafe two-local lower".into()));
        }
        Ok(BoundResidentLowerResult {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            lower,
        })
    }
}

pub(crate) fn evaluate_raw_two_local_l1_root_cut(
    q: RawTwoLocalL1RootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CompletedRawTwoLocalL1RootCut> {
    let mut guard = Poll {
        end: q.base.deadline,
        callback: poll,
        alive: true,
    };
    let mut obs = ExecutionObservation::default();
    let mut certificates = 0usize;
    let result = evaluate_steps(&q, engine, &mut guard, &mut obs, &mut certificates);
    tracing::info!(
        phase = obs.phase,
        error_class = obs.error_class,
        certificates_ok = certificates,
        support_api_entered = 0u8,
        support_api_ok = 0u8,
        lower_api_entered = obs.lower_api_entered,
        lower_api_ok = obs.lower_api_ok,
        merges_ok = obs.merges_ok,
        poll_alive_at_return = guard.alive,
        deadline_expired_at_return = Instant::now() >= q.base.deadline,
        complete = result.is_some(),
        "Certified two-local L1 root cut execution detail"
    );
    result
}
fn evaluate_group<'a>(
    q: &RawTwoLocalL1RootCutQuote<'a>,
    g: usize,
    guard: &mut Poll<'_>,
    obs: &mut ExecutionObservation,
) -> Option<LocalBoxL1Arithmetic<'a>> {
    if !guard.check() {
        return None;
    }
    obs.phase = if g == 0 {
        "local-certificate-0"
    } else {
        "local-certificate-1"
    };
    let b = &q.base;
    match local_box_l1::evaluate(
        b.graph,
        b.input,
        b.bounds,
        b.target,
        &q.neurons[g][..q.active[g]],
        &q.weights[g][..q.active[g]],
        ny_core::CertifiedWeightError::EXACT,
        q.kernel[g],
        b.deadline,
        &mut || guard.check(),
    ) {
        Ok(v) => Some(v),
        Err(e) => {
            obs.error_class = match e {
                local_box_l1::Refusal::Deadline => "deadline",
                local_box_l1::Refusal::Cancelled => "cancelled",
                local_box_l1::Refusal::Environment => "environment",
                local_box_l1::Refusal::Budget => "budget",
                local_box_l1::Refusal::Shape => "shape",
                local_box_l1::Refusal::Source => "source",
                local_box_l1::Refusal::Values => "values",
                local_box_l1::Refusal::Arithmetic => "arithmetic",
            };
            None
        }
    }
}
fn evaluate_steps(
    q: &RawTwoLocalL1RootCutQuote<'_>,
    engine: &dyn GemmEngine,
    guard: &mut Poll<'_>,
    obs: &mut ExecutionObservation,
    certificates: &mut usize,
) -> Option<CompletedRawTwoLocalL1RootCut> {
    if !guard.check() {
        return None;
    }
    obs.phase = "fp-environment";
    obs.result(ny_core::require_f64_interval_proof_environment())?;
    if !dd_selfcheck_ok() {
        return None;
    }
    obs.phase = "backend-capabilities";
    let gpu = engine.as_gpu_crown_backward()?;
    if !gpu.provides_bounded_two_local_l1_lower_cut()
        || (q.keep_single3_prefix() && !gpu.provides_bounded_local_l1_lower_cut())
    {
        return None;
    }
    let b = &q.base;
    // The first kernel consumes this attempt. The complete predeclared plan is
    // all-or-nothing: any required certificate or fold refusal leaves the old bound.
    // The separately declared box-proof branch may finish before optional prefixes.
    let first = evaluate_group(q, 0, guard, obs)?;
    *certificates = 1;
    let second = evaluate_group(q, 1, guard, obs)?;
    *certificates = 2;
    let fresh = FreshTwoLocalSupport {
        seal: &q.seal,
        arithmetic: [first, second],
    };
    if !guard.check() {
        return None;
    }
    obs.phase = "full-extraction";
    let full = extract_charged_raw_bn_chain(
        b.graph,
        b.input,
        b.graph.output_name(),
        b.bounds,
        b.bounds,
        b.alpha,
        b.extraction,
        b.deadline,
        &mut || guard.check(),
    )?;
    if full.quote != b.full_quote || !guard.check() {
        return None;
    }
    let lower = b.input.lower().as_slice()?;
    let upper = b.input.upper().as_slice()?;
    let seed = exact_seed(1, b.objective.len(), b.objective.to_vec());
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let fallback_alpha = GraphAlphaState::new();
    let base = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        b.graph,
        b.input,
        b.alpha.unwrap_or(&fallback_alpha),
        b.bounds,
        Some(engine),
        &seed,
        b.target,
        [q.neurons[0][0], q.neurons[0][1]],
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        lower,
        upper,
        b.deadline,
    );
    obs.phase = "two-local-snapshot";
    let context = obs.result(TwoLocalContext::new(q, base))?;
    if !guard.check() {
        return None;
    }
    // This additional full-D pass was paid before entry; no proof, fold or
    // result is salvaged if advice refuses. Arrays die before either GPU call.
    let selected_scale = if q.full_head_advice.is_some() {
        obs.phase = "full-head-multiplier-advice";
        let choice = context.choose_scale(&fresh, guard)?;
        if !guard.check() {
            return None;
        }
        Some(choice)
    } else {
        None
    };
    let baseline = ny_core::GpuCrownResult {
        lower_bounds: vec![b.old.0],
        upper_bounds: vec![b.old.1],
    };
    let mut best = b.old.0;
    let mut head_box_candidate_lower = None;
    if let Some(plan) = q.head_box {
        obs.phase = "head-box-carrier";
        let carrier = obs.result(context.build_with_scale(&fresh, selected_scale.as_ref()))?;
        obs.phase = "head-box-certified-contraction";
        let completed = head_box_floor::evaluate(&context, &carrier, &fresh, plan.quote, guard)?;
        let value = completed.consume(&context)?;
        if value > b.old.1 || !guard.check() {
            return None;
        }
        head_box_candidate_lower = Some(value);
        if value > best {
            best = value;
        }
    }
    // The predeclared box branch is a complete proof of the SAME original
    // query. No prefix API is entered when it already proves that query.
    let box_proves_query = q
        .head_box
        .is_some_and(|plan| head_box_candidate_lower.is_some_and(|v| v > plan.threshold));
    let mut single3_candidate_lower = None;
    let mut bundle_candidate_lower = None;
    if !box_proves_query {
        if q.keep_single3_prefix() {
            obs.phase = "single3-comparison-carrier";
            let carrier = obs.result(context.build_single3(&fresh))?;
            obs.phase = "single3-comparison-lower-api";
            obs.lower_api_entered += 1;
            let result = obs.result(context.run_single3(&carrier, gpu))?;
            obs.lower_api_ok += 1;
            if !guard.check() {
                return None;
            }
            let raw = result.lower;
            obs.phase = "single3-comparison-merge";
            let merged = obs.result(context.base.merge_backend_lower(result, &baseline))?;
            obs.merges_ok += 1;
            if merged.upper_bounds[0].to_bits() != b.old.1.to_bits() || !guard.check() {
                return None;
            }
            if merged.lower_bounds[0] > best {
                best = merged.lower_bounds[0];
            }
            single3_candidate_lower = Some(raw);
        }
        obs.phase = "two-local-carrier";
        let carrier = obs.result(context.build_with_scale(&fresh, selected_scale.as_ref()))?;
        obs.phase = "two-local-lower-api";
        obs.lower_api_entered += 1;
        let result = obs.result(context.run(&carrier, gpu))?;
        obs.lower_api_ok += 1;
        if !guard.check() {
            return None;
        }
        obs.phase = "two-local-merge";
        bundle_candidate_lower = Some(result.lower);
        let merged = obs.result(context.base.merge_backend_lower(result, &baseline))?;
        obs.merges_ok += 1;
        if merged.upper_bounds[0].to_bits() != b.old.1.to_bits() || !guard.check() {
            return None;
        }
        if merged.lower_bounds[0] > best {
            best = merged.lower_bounds[0];
        }
    }
    let raw_candidate_lower = [
        single3_candidate_lower,
        bundle_candidate_lower,
        head_box_candidate_lower,
    ]
    .into_iter()
    .flatten()
    .reduce(|current, value| if value > current { value } else { current })?;
    if !guard.check() {
        return None;
    }
    obs.phase = "complete";
    Some(CompletedRawTwoLocalL1RootCut {
        bounds: (best, b.old.1),
        raw_candidate_lower,
        bundle_candidate_lower,
        head_box_candidate_lower,
        single3_candidate_lower,
        full_head_advice: selected_scale.as_ref().map(|v| v.decision),
        original_multipliers: q.multipliers,
        active: q.active,
        neurons: q.neurons,
        weights: q.weights,
        multipliers: selected_scale
            .as_ref()
            .map_or(q.multipliers, |v| v.decision.multipliers),
        support_bounds: [fresh.arithmetic[0].bound(), fresh.arithmetic[1].bound()],
        radii: [
            fresh.arithmetic[0].bn_radius(),
            fresh.arithmetic[1].bn_radius(),
        ],
        sign_counts: [
            fresh.arithmetic[0].sign_count(),
            fresh.arithmetic[1].sign_count(),
        ],
        original_local_bounds: [
            fresh.arithmetic[0].original_local_bound(),
            fresh.arithmetic[1].original_local_bound(),
        ],
        pre_lower: [
            *fresh.arithmetic[0].pre_lower(),
            *fresh.arithmetic[1].pre_lower(),
        ],
        pre_upper: [
            *fresh.arithmetic[0].pre_upper(),
            *fresh.arithmetic[1].pre_upper(),
        ],
    })
}

#[cfg(test)]
#[path = "root_call_two_local_l1_tests.rs"]
pub(crate) mod tests;
