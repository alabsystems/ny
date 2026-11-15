// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One same-source local affine certificate and one full lower fold. No prefix
//! support is executed. Original-root/load/lease custody remains the owner gate.
use super::super::local_box_l1::{self, LocalBoxL1Arithmetic, LocalBoxL1Quote};
use super::*;
use ny_core::ResidentLocalL1LowerCutCarrier;

const LOCAL_EXTRA_BYTES: usize = 4 << 20;
// Additional to the kernel's own fixed charge and full source/fold validation:
// all17 carrier channel constructions/checks, source identity, bounded metadata,
// one immutable snapshot validation, generation/merge and observation setup.
const LOCAL_FIXED_WORK: u64 = 1 << 20;
// Fixed scalar result copies and additional numeric fields in the existing
// completed event; reserved WITHIN the unchanged 4 MiB metadata allowance.
const LOCAL_UNARY_DIAGNOSTIC_BYTES: usize = 4096;

pub(crate) struct RawLocalL1RootCutQuote<'a> {
    base: RawRootCutQuote<'a>,
    seal: ResidentCutCallSeal,
    active: usize,
    neurons: [usize; 8],
    weights: [f32; 8],
    multiplier: f32,
    kernel: LocalBoxL1Quote,
}
impl RawLocalL1RootCutQuote<'_> {
    pub(crate) fn work(&self) -> u64 {
        self.base.work
    }
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.base.scratch
    }
    pub(crate) fn candidate_count(&self) -> usize {
        1
    }
}
pub(crate) struct CompletedRawLocalL1RootCut {
    bounds: (f32, f32),
    active: usize,
    neurons: [usize; 8],
    weights: [f32; 8],
    multiplier: f32,
    bound: f32,
    radius: f64,
    sign_count: usize,
    original_local_bound: f64,
    pre_lower: [f32; 8],
    pre_upper: [f32; 8],
}
impl CompletedRawLocalL1RootCut {
    pub(crate) fn bounds(&self) -> (f32, f32) {
        self.bounds
    }
    pub(crate) fn support_calls(&self) -> usize {
        0
    }
    pub(crate) fn cut_calls(&self) -> usize {
        1
    }
    pub(crate) fn active_len(&self) -> usize {
        self.active
    }
    pub(crate) fn ordered_neurons(&self) -> [usize; 8] {
        self.neurons
    }
    pub(crate) fn weights(&self) -> [f32; 8] {
        self.weights
    }
    pub(crate) fn multiplier(&self) -> f32 {
        self.multiplier
    }
    pub(crate) fn support_bound(&self) -> f32 {
        self.bound
    }
    pub(crate) fn bn_radius(&self) -> f64 {
        self.radius
    }
    pub(crate) fn sign_count(&self) -> usize {
        self.sign_count
    }
    pub(crate) fn original_local_bound(&self) -> f64 {
        self.original_local_bound
    }
    pub(crate) fn pre_lower(&self) -> &[f32; 8] {
        &self.pre_lower
    }
    pub(crate) fn pre_upper(&self) -> &[f32; 8] {
        &self.pre_upper
    }
}
fn valid_selection(m: usize, n: [usize; 8], w: [f32; 8], width: usize) -> bool {
    (1..=8).contains(&m)
        && n[..m].windows(2).all(|v| v[0] < v[1])
        && n[m - 1] < width
        && n[m..].iter().all(|n| *n == 0)
        && w[m..].iter().all(|w| w.to_bits() == 0)
        && w[..m].iter().all(|w| {
            w.to_bits() >= (1f32 / 256.).to_bits()
                && w.to_bits() <= 1f32.to_bits()
                && (ny_core::f32_to_f64_exact(*w) * 256.).fract() == 0.
        })
}
fn same_weights(a: [f32; 8], b: [f32; 8]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_raw_local_l1_root_cut<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    active: usize,
    neurons: [usize; 8],
    weights: [f32; 8],
    multiplier: f32,
    limits: RawRootCutLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<RawLocalL1RootCutQuote<'a>> {
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
        || !ny_core::is_crown_coeff_safe(multiplier)
        || multiplier <= 0.
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
    let kernel = local_box_l1::quote(linear.in_features(), linear.out_features())?;
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
        .checked_add(kernel.work)?
        .checked_add(LOCAL_FIXED_WORK)?
        .checked_add(EXECUTION_OBSERVATION_WORK)?;
    if size_of::<ResidentLocalL1LowerCutCarrier>() > 512
        || kernel
            .scratch_bytes
            .checked_add(size_of::<RawLocalL1RootCutQuote<'_>>())?
            .checked_add(size_of::<CompletedRawLocalL1RootCut>())?
            .checked_add(LOCAL_UNARY_DIAGNOSTIC_BYTES)?
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
        pair: [neurons[0], neurons[1]],
        lambda_scale: multiplier,
        candidate_count: 1,
        extraction,
        prefix_quote,
        full_quote,
        work,
        scratch,
        deadline,
    };
    Some(RawLocalL1RootCutQuote {
        base,
        seal: ResidentCutCallSeal([0xE8]),
        active,
        neurons,
        weights,
        multiplier,
        kernel,
    })
}

struct FreshLocalSupport<'q, 'a> {
    seal: &'q ResidentCutCallSeal,
    arithmetic: LocalBoxL1Arithmetic<'a>,
}
struct LocalContext<'q, 'a> {
    quote: &'q RawLocalL1RootCutQuote<'a>,
    base: ResidentCutCallContext<'a>,
    target_activation: usize,
    activation_count: usize,
    width: usize,
}
struct LocalCarrier<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    active: usize,
    neurons: [usize; 8],
    weights: [f32; 8],
    bound: f32,
    transport: ResidentLocalL1LowerCutCarrier,
}
impl<'q, 'a> LocalContext<'q, 'a> {
    fn new(q: &'q RawLocalL1RootCutQuote<'a>, base: ResidentCutCallContext<'a>) -> Result<Self> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("local L1 IEEE".into()));
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
            || base.ordered_neurons != [q.neurons[0], q.neurons[1]]
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
                "local L1 original snapshot differs".into(),
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
    fn build<'c>(&'c self, fresh: &FreshLocalSupport<'_, '_>) -> Result<LocalCarrier<'c>> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("local L1 reduction IEEE".into()));
        }
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let q = self.quote;
        let f = &fresh.arithmetic;
        let b = &q.base;
        if !std::ptr::eq(fresh.seal, &raw const q.seal)
            || !f.matches_source(b.graph, b.input, b.bounds, b.target)
            || f.neurons() != &q.neurons[..q.active]
            || f.weights().len() != q.active
            || f.weights()
                .iter()
                .zip(q.weights)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || f.sign_count() != 1usize << (q.active - 1)
            || f.quote() != q.kernel
            || !ny_core::is_crown_coeff_safe(f.bound())
            || f.bound() < 0.
        {
            return Err(NyError::SoundnessRefusal(
                "foreign local L1 certificate".into(),
            ));
        }
        let channel = |v: f32| {
            transport_channel(directed_reduce(std::iter::once((
                ny_core::f32_to_f64_exact(q.multiplier),
                ny_core::f32_to_f64_exact(v),
            )))?)
        };
        let zero = ResidentLowerCutChannel::try_new(0., 0.)?;
        let mut pre = [zero; 8];
        let mut post = [zero; 8];
        for i in 0..q.active {
            pre[i] = channel(-q.weights[i])?;
            post[i] = channel(2. * q.weights[i])?;
        }
        let transport = ResidentLocalL1LowerCutCarrier::try_new(
            self.target_activation,
            self.width,
            q.active,
            q.neurons,
            q.multiplier,
            pre,
            post,
            channel(-f.bound())?,
            self.base.deadline,
        )?;
        transport.validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        Ok(LocalCarrier {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            active: q.active,
            neurons: q.neurons,
            weights: q.weights,
            bound: f.bound(),
            transport,
        })
    }
    fn run<'c>(
        &'c self,
        c: &LocalCarrier<'_>,
        gpu: &dyn GpuCrownBackward,
    ) -> Result<BoundResidentLowerResult<'c>> {
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        if !std::ptr::eq(c.seal, &raw const self.base.seal)
            || c.generations != self.base.generations.get()
            || c.active != self.quote.active
            || c.neurons != self.quote.neurons
            || !same_weights(c.weights, self.quote.weights)
            || !ny_core::is_crown_coeff_safe(c.bound)
            || c.bound < 0.
            || !gpu.provides_bounded_local_l1_lower_cut()
            || c.transport.active_len() != c.active
            || *c.transport.ordered_neurons() != c.neurons
            || c.transport.multiplier().to_bits() != self.quote.multiplier.to_bits()
            || c.transport.target_activation() != self.target_activation
        {
            return Err(NyError::SoundnessRefusal(
                "foreign/stale local L1 carrier".into(),
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
            return Err(NyError::SoundnessRefusal("unsafe local lower".into()));
        }
        Ok(BoundResidentLowerResult {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            lower,
        })
    }
}

pub(crate) fn evaluate_raw_local_l1_root_cut(
    q: RawLocalL1RootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CompletedRawLocalL1RootCut> {
    let mut guard = Poll {
        end: q.base.deadline,
        callback: poll,
        alive: true,
    };
    let mut obs = ExecutionObservation::default();
    let result = evaluate_steps(&q, engine, &mut guard, &mut obs);
    tracing::info!(
        phase = obs.phase,
        error_class = obs.error_class,
        support_api_entered = 0u8,
        support_api_ok = 0u8,
        lower_api_entered = obs.lower_api_entered,
        lower_api_ok = obs.lower_api_ok,
        merges_ok = obs.merges_ok,
        poll_alive_at_return = guard.alive,
        deadline_expired_at_return = Instant::now() >= q.base.deadline,
        complete = result.is_some(),
        "Certified local L1 root cut execution detail"
    );
    result
}
fn evaluate_steps(
    q: &RawLocalL1RootCutQuote<'_>,
    engine: &dyn GemmEngine,
    guard: &mut Poll<'_>,
    obs: &mut ExecutionObservation,
) -> Option<CompletedRawLocalL1RootCut> {
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
    if !gpu.provides_bounded_local_l1_lower_cut() {
        return None;
    }
    let b = &q.base;
    obs.phase = "local-certificate";
    // First numerical entry consumes this attempt. The outer owner never
    // returns to another family, even if only part of this CPU sweep completes.
    let arithmetic = local_box_l1::evaluate(
        b.graph,
        b.input,
        b.bounds,
        b.target,
        &q.neurons[..q.active],
        &q.weights[..q.active],
        ny_core::CertifiedWeightError::EXACT,
        q.kernel,
        b.deadline,
        &mut || guard.check(),
    );
    let arithmetic = match arithmetic {
        Ok(v) => v,
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
            return None;
        }
    };
    let fresh = FreshLocalSupport {
        seal: &q.seal,
        arithmetic,
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
        [q.neurons[0], q.neurons[1]],
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        lower,
        upper,
        b.deadline,
    );
    obs.phase = "local-snapshot";
    let context = obs.result(LocalContext::new(q, base))?;
    if !guard.check() {
        return None;
    }
    obs.phase = "local-carrier";
    let carrier = obs.result(context.build(&fresh))?;
    obs.phase = "local-lower-api";
    obs.lower_api_entered += 1;
    let result = obs.result(context.run(&carrier, gpu))?;
    obs.lower_api_ok += 1;
    if !guard.check() {
        return None;
    }
    let baseline = ny_core::GpuCrownResult {
        lower_bounds: vec![b.old.0],
        upper_bounds: vec![b.old.1],
    };
    obs.phase = "local-merge";
    let merged = obs.result(context.base.merge_backend_lower(result, &baseline))?;
    obs.merges_ok += 1;
    if merged.upper_bounds[0].to_bits() != b.old.1.to_bits() || !guard.check() {
        return None;
    }
    obs.phase = "complete";
    Some(CompletedRawLocalL1RootCut {
        bounds: (merged.lower_bounds[0], b.old.1),
        active: q.active,
        neurons: q.neurons,
        weights: q.weights,
        multiplier: q.multiplier,
        bound: fresh.arithmetic.bound(),
        radius: fresh.arithmetic.bn_radius(),
        sign_count: fresh.arithmetic.sign_count(),
        original_local_bound: fresh.arithmetic.original_local_bound(),
        pre_lower: *fresh.arithmetic.pre_lower(),
        pre_upper: *fresh.arithmetic.pre_upper(),
    })
}

#[cfg(test)]
#[path = "root_call_local_l1_tests.rs"]
pub(crate) mod tests;
