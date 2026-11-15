// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Same-call raw-source construction, one checked support call and two cuts.
//! No historical shadow result or externally supplied facet becomes authority.

use super::*;
use crate::multineuron::ExactRelu2Support;
use crate::network::raw_root_cut::{
    extract_charged_raw_bn_chain, quote_charged_raw_bn_chain, ExtractionLimits, ExtractionQuote,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawRootCutLimits {
    pub max_nodes: usize,
    pub max_scratch_bytes: usize,
    pub max_work: u64,
}

/// Move-only, unforgeable outside this module. These immutable borrows bind
/// quote and execution to exactly the original source/box/bounds/objective.
pub(crate) struct RawRootCutQuote<'a> {
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    pre_node: &'a str,
    pair: [usize; 2],
    lambda_scale: f32,
    candidate_count: usize,
    extraction: ExtractionLimits,
    prefix_quote: ExtractionQuote,
    full_quote: ExtractionQuote,
    work: u64,
    scratch: usize,
    deadline: Instant,
}
impl RawRootCutQuote<'_> {
    pub(crate) fn work(&self) -> u64 {
        self.work
    }
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.scratch
    }
    pub(crate) fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

/// Only the complete same-call evaluation below constructs this object.
/// Original-root/query ownership and actual lease custody belong to the caller.
pub(crate) struct CompletedRawRootCut {
    bounds: (f32, f32),
    support_calls: usize,
    cut_calls: usize,
    normal: [f32; 4],
    lambda_scale: f32,
}
impl CompletedRawRootCut {
    pub(crate) fn bounds(&self) -> (f32, f32) {
        self.bounds
    }
    pub(crate) fn support_calls(&self) -> usize {
        self.support_calls
    }
    pub(crate) fn cut_calls(&self) -> usize {
        self.cut_calls
    }
    pub(crate) fn cut_normal(&self) -> [f32; 4] {
        self.normal
    }
    pub(crate) fn lambda_scale(&self) -> f32 {
        self.lambda_scale
    }
}

struct Poll<'a> {
    end: Instant,
    callback: &'a mut dyn FnMut() -> bool,
    alive: bool,
}
impl Poll<'_> {
    fn check(&mut self) -> bool {
        self.alive = self.alive && Instant::now() < self.end && (self.callback)();
        self.alive
    }
}

// Observation only: these fields cannot construct a CompletedRawRootCut. API
// entries/Ok returns do not count GPU kernels or confer completed-cut authority.
// One fixed-size record is emitted after the existing synchronous call returns.
#[derive(Debug)]
struct ExecutionObservation {
    phase: &'static str,
    error_class: &'static str,
    support_api_entered: u8,
    support_api_ok: u8,
    lower_api_entered: u8,
    lower_api_ok: u8,
    merges_ok: u8,
    poll_alive_at_return: bool,
    deadline_expired_at_return: bool,
}
impl Default for ExecutionObservation {
    fn default() -> Self {
        Self {
            phase: "entry",
            error_class: "none",
            support_api_entered: 0,
            support_api_ok: 0,
            lower_api_entered: 0,
            lower_api_ok: 0,
            merges_ok: 0,
            poll_alive_at_return: true,
            deadline_expired_at_return: false,
        }
    }
}
impl ExecutionObservation {
    fn result<T>(&mut self, result: Result<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                // Preserve the outer typed class, without unwrapping a layer
                // error, formatting arbitrary text, or changing any refusal.
                self.error_class = match error {
                    NyError::DeadlineExceeded(_) => "deadline",
                    NyError::SoundnessRefusal(_) => "soundness",
                    NyError::NumericalInstability(_) => "numerical",
                    NyError::ShapeMismatch { .. } => "shape",
                    NyError::GpuMemoryExceeded { .. } => "gpu-memory",
                    NyError::CpuMemoryExceeded { .. } => "cpu-memory",
                    NyError::GpuBatchCapacityExceeded { .. } => "gpu-capacity",
                    NyError::UnsupportedOp(_) => "unsupported-op",
                    NyError::UnsupportedLayer(_) => "unsupported-layer",
                    NyError::UnsupportedConfiguration(_) => "unsupported-config",
                    NyError::LayerError { .. } => "layer-wrapper",
                    NyError::InternalError(_) => "internal",
                    _ => "other",
                };
                None
            }
        }
    }

    fn emit(&self, complete: bool) {
        tracing::info!(
            phase = self.phase,
            error_class = self.error_class,
            support_api_entered = self.support_api_entered,
            support_api_ok = self.support_api_ok,
            lower_api_entered = self.lower_api_entered,
            lower_api_ok = self.lower_api_ok,
            merges_ok = self.merges_ok,
            poll_alive_at_return = self.poll_alive_at_return,
            deadline_expired_at_return = self.deadline_expired_at_return,
            complete,
            "Certified root cut execution detail"
        );
    }
}

// Additive accounting for the bounded observer and its fixed stage/counter
// updates. Existing owner caps and all proof/extraction quotes remain intact.
const EXECUTION_OBSERVATION_BYTES: usize = 512;
const EXECUTION_OBSERVATION_WORK: u64 = 65_536;

// The two-column advice scan occurs only AFTER the whole execution lease.
// No extra source pass, descriptor clone, certificate, or backend call is used.
// Up to128 shape nodes, eight name hash/comparison actions per node,
// <=1024 bytes/name and16 units/byte, plus8192 scalar/layout/normal work.
const ADVICE_FIXED_WORK: u64 = (1 << 24) + 8192;
const ADVICE_WORK_PER_OUTPUT: u64 = 64;
const ADVICE_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
struct CutAdvice {
    normal: [f32; 4],
    lambda_scale: f32,
}
impl CutAdvice {
    fn uniform(lambda_scale: f32) -> Self {
        Self {
            normal: [-1., -1., 2., 2.],
            lambda_scale,
        }
    }
}

// Q8 is deliberate: arbitrary f32 normals are outside the quoted rational
// scratch proof. This is advice only; the fresh exact facet/full fold below
// authenticates the actual stored normal and lambda independently.
fn q8_advice(magnitudes: [f64; 2]) -> Option<CutAdvice> {
    if magnitudes.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return None;
    }
    let largest = magnitudes[0].max(magnitudes[1]);
    let lambda_scale = (0.5 * largest) as f32;
    if !ny_core::is_crown_coeff_safe(lambda_scale) || lambda_scale <= 0. {
        return None;
    }
    let mut weights = [0f32; 2];
    for i in 0..2 {
        let ratio = magnitudes[i] / largest;
        if !ratio.is_finite() || !((1.0 / 256.0)..=1.0).contains(&ratio) {
            return None;
        }
        let numerator = (256. * ratio).round();
        if !(1.0..=256.0).contains(&numerator) {
            return None;
        }
        weights[i] = (numerator / 256.) as f32;
        // These products must be exact, not merely finite.
        if f64::from(weights[i]) != numerator / 256.
            || f64::from(2. * weights[i]) != 2. * f64::from(weights[i])
        {
            return None;
        }
    }
    Some(CutAdvice {
        normal: [-weights[0], -weights[1], 2. * weights[0], 2. * weights[1]],
        lambda_scale,
    })
}

fn choose_advice(q: &RawRootCutQuote<'_>, guard: &mut Poll<'_>) -> Option<CutAdvice> {
    if !guard.check() {
        return None;
    }
    let candidate = (|| {
        if q.graph.output_name().len() > 1024 {
            return None;
        }
        let output = q.graph.node(q.graph.output_name())?;
        let Layer::Linear(linear) = output.layer() else {
            return None;
        };
        let width = linear.in_features();
        let mut current = output.require_unary_input().ok()?;
        // Authored Conv->ReLU->Flatten->Gemm retains a shape-only node. These
        // two operators preserve row-major flat coordinate order. Every
        // intermediate bound must retain that same indexed width; transpose,
        // permutation and arbitrary operations are deliberately not followed.
        for _ in 0..128 {
            if !guard.check() {
                return None;
            }
            if current.len() > 1024 {
                return None;
            }
            if current == q.target {
                break;
            }
            let shape = q.graph.node(current)?;
            if !matches!(shape.layer(), Layer::Flatten(_) | Layer::Reshape(_)) {
                return None;
            }
            let parent = shape.require_unary_input().ok()?;
            for name in [current, parent] {
                if name.len() > 1024 {
                    return None;
                }
                let b = q.bounds.get(name)?;
                if b.shape().len() > 4
                    || b.len() != width
                    || b.lower().as_slice().is_none()
                    || b.upper().as_slice().is_none()
                {
                    return None;
                }
            }
            current = parent;
        }
        if current != q.target {
            return None;
        }
        if linear.out_features() != q.objective.len()
            || width != q.prefix_quote.output_dim
            || q.pair.iter().any(|&i| i >= width)
        {
            return None;
        }
        let weight = linear.weight.as_slice()?;
        if weight.len() != width.checked_mul(q.objective.len())? {
            return None;
        }
        let mut coefficients = [0f64; 2];
        for (row, &objective) in q.objective.iter().enumerate() {
            if row % 128 == 0 && !guard.check() {
                return None;
            }
            for i in 0..2 {
                let w = weight[row * width + q.pair[i]];
                if !ny_core::is_crown_coeff_safe(objective) || !ny_core::is_crown_coeff_safe(w) {
                    return None;
                }
                coefficients[i] += f64::from(objective) * f64::from(w);
            }
        }
        let advice = q8_advice([-coefficients[0], -coefficients[1]])?;
        if q.candidate_count == 2 && advice.lambda_scale * 0.5 == 0. {
            return None;
        }
        Some(advice)
    })();
    // Numeric/advice refusal keeps the uniform candidate; cancellation never
    // falls back, including a transient false inside the two-column scan.
    guard
        .check()
        .then(|| candidate.unwrap_or_else(|| CutAdvice::uniform(q.lambda_scale)))
}

enum AdviceMode {
    Weighted,
    #[cfg(test)]
    Uniform,
}

fn phase_bytes(q: ExtractionQuote, rows: usize) -> Option<usize> {
    crate::network::raw_root_cut::bounded_fold_bytes(q, rows)
}

fn phase_work(q: ExtractionQuote, rows: u64, repetitions: u64) -> Option<u64> {
    crate::network::raw_root_cut::bounded_fold_work(q, rows, repetitions)
}

fn retained_prefix_cache_bytes(q: ExtractionQuote) -> Option<usize> {
    // Explicit-copy CUDA keeps A/B/C scratch after the four-row support call.
    // Charge its possible old allocation plus transactional growth overlap,
    // independently of the later full fold's live per-phase workspace.
    // GEMMs batch both lower and upper rows (8 prefix rows).
    16usize.checked_mul(
        8usize
            .checked_mul(q.max_frontier_dim)?
            .checked_add(q.total_weight_words)?
            .checked_add(8usize.checked_mul(q.max_gemm_output_per_row)?)?,
    )
}

/// Shape quote under the caller's bounded setup lease. It allocates no layer
/// descriptors or bound-map copies and scans no full weight tensors. The BN
/// layout helper's bounded channel validation is included in setup; all full
/// parameter/source checks follow the complete work/byte lease. No engine or
/// global accessor is used here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_raw_root_cut<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    pair: [usize; 2],
    lambda_scale: f32,
    limits: RawRootCutLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<RawRootCutQuote<'a>> {
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
        || !ny_core::is_crown_coeff_safe(lambda_scale)
        || lambda_scale <= 0.0
        || target.len() > 1024
        || objective.is_empty()
        || objective.len() > 8192
        || objective.iter().any(|&x| !ny_core::is_crown_coeff_safe(x))
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
    let pre = bounds.get(pre_node)?;
    let post = bounds.get(target)?;
    if pre.is_empty()
        || pre.len() != post.len()
        || pair[0] == pair[1]
        || pair.iter().any(|&i| i >= pre.len())
    {
        return None;
    }
    let mut saw = false;
    let mut current = graph.output_name();
    for _ in 0..limits.max_nodes {
        if !guard.check() {
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
    let mut alive = || guard.check();
    let prefix_quote = quote_charged_raw_bn_chain(
        graph, input, pre_node, bounds, bounds, alpha, extraction, deadline, &mut alive,
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
        &mut alive,
    )?;
    if prefix_quote.output_dim != pre.len() || full_quote.output_dim != objective.len() {
        return None;
    }
    // Owned prefix descriptors/seed/host temporaries drop before full extraction.
    // CUDA cached device storage is explicitly additional, not assumed freed.
    // The fixed support/certificate fits the per-phase 1MiB term.
    if 4 * size_of::<CutAdvice>()
        + 2 * size_of::<CompletedRawRootCut>()
        + 2 * size_of::<Option<[f32; 4]>>()
        + 2 * size_of::<Option<f32>>()
        > ADVICE_BYTES
        || size_of::<ExecutionObservation>() > EXECUTION_OBSERVATION_BYTES
    {
        return None;
    }
    let scratch = phase_bytes(prefix_quote, 4)?
        .max(phase_bytes(full_quote, 1)?)
        .checked_add(retained_prefix_cache_bytes(prefix_quote)?)?
        .checked_add(ADVICE_BYTES)?
        .checked_add(EXECUTION_OBSERVATION_BYTES)?;
    let advice_work = ADVICE_FIXED_WORK
        .checked_add(ADVICE_WORK_PER_OUTPUT.checked_mul(u64::try_from(objective.len()).ok()?)?)?;
    let common_work = phase_work(prefix_quote, 4, 1)?
        .checked_add(16_000_000)?
        .checked_add(advice_work)?
        .checked_add(EXECUTION_OBSERVATION_WORK)?;
    let two_work = common_work.checked_add(phase_work(full_quote, 1, 2)?)?;
    // Finite predeclared advice only: if both fully priced folds do not fit,
    // retain one full-scale proposal. Never discount its proof/value checks.
    let (candidate_count, work) = if two_work <= limits.max_work && lambda_scale * 0.5 != 0. {
        (2, two_work)
    } else {
        (1, common_work.checked_add(phase_work(full_quote, 1, 1)?)?)
    };
    if scratch > limits.max_scratch_bytes || work > limits.max_work || !guard.check() {
        return None;
    }
    Some(RawRootCutQuote {
        graph,
        input,
        bounds,
        alpha,
        objective,
        old,
        target,
        pre_node,
        pair,
        lambda_scale,
        candidate_count,
        extraction,
        prefix_quote,
        full_quote,
        work,
        scratch,
        deadline,
    })
}

/// Call-local immutable support. It cannot be created from a serialized/raw
/// Octahedron or moved across a different input/source owner by any public API.
struct FreshSupport {
    octahedron: Octahedron2,
    certificate: ExactRelu2FacetCertificate,
}

fn exact_seed(rows: usize, dim: usize, a: Vec<f32>) -> GpuCrownSeed {
    let a: std::sync::Arc<[f32]> = a.into();
    let b: std::sync::Arc<[f32]> = vec![0.; rows].into();
    GpuCrownSeed {
        lower_a: a.clone(),
        upper_a: a,
        lower_b: b.clone(),
        upper_b: b,
        num_specs: rows,
        current_dim: dim,
    }
}

// The only caller supplies a support produced synchronously from the same
// opaque quote, then creates the resident context from that quote. There is no
// public closure or raw-support parameter at the root entry.
fn carrier_from_fresh<'c, 'a>(
    context: &'c ResidentCutCallContext<'a>,
    fresh: &FreshSupport,
    lambda: f32,
) -> Result<BoundResidentLowerCutCarrier<'c, 'a>> {
    let (target_activation, widths) = context.validate_resident_snapshot()?;
    let request = CurrentLowerCutRequest {
        graph: context.graph,
        input: context.input,
        alpha_state: context.alpha_state,
        node_bounds: context.node_bounds,
        engine: context.engine,
        lower_seed: LowerSeedRef::from_seed(context.seed),
        target_relu: context.target_relu,
        ordered_neurons: context.ordered_neurons,
        deadline: context.deadline,
    };
    request.lower_seed.validate()?;
    let pre_node = context
        .graph
        .node(context.target_relu)
        .ok_or_else(|| NyError::SoundnessRefusal("raw target lost".into()))?
        .require_unary_input()?;
    let certificates = vec![fresh.certificate.clone()];
    let row = reduce_row(&certificates, vec![NonNegativeMultiplier::new(lambda)?])?;
    let transport_row = ResidentLowerCutRow::try_new(
        vec![lambda],
        [
            transport_channel(row.pre()[0])?,
            transport_channel(row.pre()[1])?,
        ],
        [
            transport_channel(row.post()[0])?,
            transport_channel(row.post()[1])?,
        ],
        transport_channel(row.bias())?,
    )?;
    let width = widths[target_activation];
    let transport = ResidentLowerCutCarrier::try_new(
        target_activation,
        width,
        context.ordered_neurons,
        vec![transport_row],
        context.deadline,
    )?;
    transport.validate_for_call(widths.len(), width, 1, context.deadline)?;
    Ok(BoundResidentLowerCutCarrier {
        seal: &context.seal,
        generations: context.generations.get(),
        proof: CertifiedLowerCutReplay {
            request,
            pre_node,
            target_width: width,
            fresh_support: fresh.octahedron.clone(),
            certificates,
            rows: vec![row],
        },
        transport,
    })
}

/// Root must hold the complete quoted lease for this entire synchronous call.
/// Every error/cancellation returns None; old bounds are borrowed values and
/// never mutated. The upper result remains exactly the original upper bits.
pub(crate) fn evaluate_raw_root_cut(
    q: RawRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CompletedRawRootCut> {
    evaluate_raw_root_cut_inner(q, engine, poll, AdviceMode::Weighted)
}

fn evaluate_raw_root_cut_inner(
    q: RawRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
    mode: AdviceMode,
) -> Option<CompletedRawRootCut> {
    let mut observation = ExecutionObservation::default();
    let result = evaluate_raw_root_cut_observed(q, engine, poll, mode, &mut observation);
    observation.emit(result.is_some());
    result
}

// This wrapper records final poll/deadline state without another callback poll.
// Deadline-at-return is descriptive; it does not identify the cause of None.
fn evaluate_raw_root_cut_observed(
    q: RawRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
    mode: AdviceMode,
    observation: &mut ExecutionObservation,
) -> Option<CompletedRawRootCut> {
    let deadline = q.deadline;
    let mut guard = Poll {
        end: deadline,
        callback: poll,
        alive: true,
    };
    let result = evaluate_raw_root_cut_steps(q, engine, &mut guard, mode, observation);
    observation.poll_alive_at_return = guard.alive;
    observation.deadline_expired_at_return = Instant::now() >= deadline;
    result
}

fn evaluate_raw_root_cut_steps(
    q: RawRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    guard: &mut Poll<'_>,
    mode: AdviceMode,
    observation: &mut ExecutionObservation,
) -> Option<CompletedRawRootCut> {
    if !guard.check() {
        return None;
    }
    observation.phase = "fp-environment";
    observation.result(ny_core::require_f64_interval_proof_environment())?;
    observation.phase = "dd-selfcheck";
    if !dd_selfcheck_ok() {
        return None;
    }
    observation.phase = "backend-capabilities";
    let gpu = engine.as_gpu_crown_backward()?;
    if !gpu.provides_bounded_support4() || !gpu.provides_bounded_lower_cut() {
        return None;
    }
    observation.phase = "advice";
    let advice = match mode {
        AdviceMode::Weighted => choose_advice(&q, guard)?,
        #[cfg(test)]
        AdviceMode::Uniform => CutAdvice::uniform(q.lambda_scale),
    };
    if !guard.check() {
        return None;
    }
    observation.phase = "input-slices";
    let lower = q.input.lower().as_slice()?;
    let upper = q.input.upper().as_slice()?;
    let fresh = {
        observation.phase = "prefix-extraction";
        let prefix = extract_charged_raw_bn_chain(
            q.graph,
            q.input,
            q.pre_node,
            q.bounds,
            q.bounds,
            q.alpha,
            q.extraction,
            q.deadline,
            &mut || guard.check(),
        )?;
        observation.phase = "prefix-identity";
        if prefix.quote != q.prefix_quote || !guard.check() {
            return None;
        }
        observation.phase = "support-seed";
        let d = prefix.output_dim;
        let [i, j] = q.pair;
        let mut a = vec![0.; 4usize.checked_mul(d)?];
        a[i] = 1.;
        a[d + j] = 1.;
        a[2 * d + i] = 1.;
        a[2 * d + j] = 1.;
        a[3 * d + i] = 1.;
        a[3 * d + j] = -1.;
        let seed = exact_seed(4, d, a);
        observation.phase = "support-api";
        observation.support_api_entered += 1;
        let support = observation.result(gpu.crown_backward_gpu_resnet_support4_with_deadline(
            &prefix.segments,
            &seed,
            lower,
            upper,
            q.deadline,
        ))?;
        observation.support_api_ok += 1;
        observation.phase = "support-shape";
        if !guard.check() || support.lower_bounds.len() != 4 || support.upper_bounds.len() != 4 {
            return None;
        }
        observation.phase = "support-values";
        for (&l, &u) in support.lower_bounds.iter().zip(&support.upper_bounds) {
            if !ny_core::is_crown_coeff_safe(l) || !ny_core::is_crown_coeff_safe(u) || l > u {
                return None;
            }
        }
        let l = &support.lower_bounds;
        let u = &support.upper_bounds;
        let octahedron = Octahedron2 {
            l1: f64::from(l[0]),
            u1: f64::from(u[0]),
            l2: f64::from(l[1]),
            u2: f64::from(u[1]),
            s_lo: f64::from(l[2]),
            s_hi: f64::from(u[2]),
            d_lo: f64::from(l[3]),
            d_hi: f64::from(u[3]),
        };
        // The exact checker is restricted to dyadics widened from safe f32.
        // Q8 normals have denominator<=256 and magnitude<=2. With the
        // safe-f32 support bounds, reduced coordinates have denominator<=2^150
        // and numerator<2^185. Four-term products/sums and unreduced rational
        // comparisons stay below354 bits, inside the original1024-bit reserve.
        // There are <=180 vertices; quote retains1MiB/16M exact-check work.
        let coordinates = [
            octahedron.l1,
            octahedron.u1,
            octahedron.l2,
            octahedron.u2,
            octahedron.s_lo,
            octahedron.s_hi,
            octahedron.d_lo,
            octahedron.d_hi,
        ];
        observation.phase = "support-eligibility";
        if coordinates
            .iter()
            .any(|&v| f64::from(v as f32).to_bits() != v.to_bits())
            || !guard.check()
        {
            return None;
        }
        // Fresh support can stabilize a pair selected from coarser crossing
        // bounds. The exact checker covers all closed orthants, including
        // stable segments/points; its certificate remains useful while the
        // full fold retains the original map's looser ReLU relaxation. Keep
        // exact nonempty-support certification and every completion guard.
        observation.phase = "facet-construction";
        let checker = ExactRelu2Support::new(&octahedron)?;
        observation.phase = "facet-certificate";
        let certificate = checker.certify_normal_certificate(advice.normal)?;
        if !guard.check() {
            return None;
        }
        FreshSupport {
            octahedron,
            certificate,
        }
    }; // prefix descriptors, four-row seed and checker scratch are gone here
    observation.phase = "full-extraction";
    let full = extract_charged_raw_bn_chain(
        q.graph,
        q.input,
        q.graph.output_name(),
        q.bounds,
        q.bounds,
        q.alpha,
        q.extraction,
        q.deadline,
        &mut || guard.check(),
    )?;
    observation.phase = "full-identity";
    if full.quote != q.full_quote || !guard.check() {
        return None;
    }
    observation.phase = "target-identity";
    let target_count = full
        .relu_names
        .iter()
        .filter(|name| name.as_str() == q.target)
        .count();
    if target_count != 1 {
        return None;
    }
    observation.phase = "context";
    let seed = exact_seed(1, q.objective.len(), q.objective.to_vec());
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let fallback_alpha = GraphAlphaState::new();
    let context = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        q.graph,
        q.input,
        q.alpha.unwrap_or(&fallback_alpha),
        q.bounds,
        Some(engine),
        &seed,
        q.target,
        q.pair,
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        lower,
        upper,
        q.deadline,
    );
    let baseline = ny_core::GpuCrownResult {
        lower_bounds: vec![q.old.0],
        upper_bounds: vec![q.old.1],
    };
    let mut best = q.old.0;
    let lambdas = [advice.lambda_scale * 0.5, advice.lambda_scale];
    for &lambda in &lambdas[2 - q.candidate_count..] {
        if !guard.check() {
            return None;
        }
        observation.phase = "carrier";
        let carrier = observation.result(carrier_from_fresh(&context, &fresh, lambda))?;
        observation.phase = "lower-api";
        observation.lower_api_entered += 1;
        let result = observation.result(context.run_backend_lower(&carrier, gpu))?;
        observation.lower_api_ok += 1;
        if !guard.check() {
            return None;
        }
        observation.phase = "merge";
        let merged = observation.result(context.merge_backend_lower(result, &baseline))?;
        observation.merges_ok += 1;
        observation.phase = "upper-bits";
        if merged.upper_bounds[0].to_bits() != q.old.1.to_bits() {
            return None;
        }
        if merged.lower_bounds[0] > best {
            best = merged.lower_bounds[0];
        }
    }
    observation.phase = "final-check";
    if !guard.check() {
        return None;
    }
    observation.phase = "complete";
    Some(CompletedRawRootCut {
        bounds: (best, q.old.1),
        support_calls: 1,
        cut_calls: q.candidate_count,
        normal: advice.normal,
        lambda_scale: advice.lambda_scale,
    })
}

#[cfg(test)]
#[path = "root_call_tests.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "root_call_weighted_tests.rs"]
mod weighted_tests;

#[path = "root_call_two_pair.rs"]
mod two_pair;
pub(crate) use two_pair::{
    evaluate_raw_two_pair_root_cut, quote_raw_two_pair_root_cut, CompletedRawTwoPairRootCut,
    RawTwoPairRootCutQuote,
};

#[cfg(test)]
pub(crate) use two_pair::tests as two_pair_test_support;

#[path = "root_call_triplet.rs"]
mod root_call_triplet;
#[cfg(test)]
pub(crate) use root_call_triplet::tests as triplet_test_support;
pub(crate) use root_call_triplet::{
    evaluate_raw_triplet_root_cut, quote_raw_triplet_root_cut, CompletedRawTripletRootCut,
    RawTripletRootCutQuote,
};

#[path = "root_call_local_l1.rs"]
mod root_call_local_l1;
#[cfg(test)]
pub(crate) use root_call_local_l1::tests as local_l1_test_support;
pub(crate) use root_call_local_l1::{
    evaluate_raw_local_l1_root_cut, quote_raw_local_l1_root_cut, CompletedRawLocalL1RootCut,
    RawLocalL1RootCutQuote,
};

#[path = "root_call_two_local_l1.rs"]
mod root_call_two_local_l1;
#[cfg(test)]
pub(crate) use root_call_two_local_l1::tests as two_local_l1_test_support;
pub(crate) use root_call_two_local_l1::{
    evaluate_raw_two_local_l1_root_cut, quote_raw_two_local_l1_root_cut,
    CompletedRawTwoLocalL1RootCut, RawTwoLocalL1RootCutQuote,
};
