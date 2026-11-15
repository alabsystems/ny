// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Same-call weighted L1 authority from four signed support rows. Transport
//! carries arithmetic only; this private module owns the original-row proof.
use super::*;
use ny_core::ResidentTripletLowerCutCarrier;

const TRIPLET_EXTRA_BYTES: usize = 4 << 20;
const TRIPLET_FIXED_WORK: u64 = 1 << 20;
// Retain the two-pair envelope conservatively. No second facet enumeration or
// selected-column walk is executed; the fixed sign certificate is smaller.
const TRIPLET_CERTIFICATE_WORK: u64 = 32_000_000;
const SIGNS: [[f32; 3]; 4] = [[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]];

pub(crate) struct RawTripletRootCutQuote<'a> {
    base: RawRootCutQuote<'a>,
    seal: ResidentCutCallSeal,
    neurons: [usize; 3],
    weights: [f32; 3],
    multiplier: f32,
    work: u64,
    scratch: usize,
}
impl RawTripletRootCutQuote<'_> {
    pub(crate) fn work(&self) -> u64 {
        self.work
    }
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.scratch
    }
    pub(crate) fn candidate_count(&self) -> usize {
        1
    }
}
pub(crate) struct CompletedRawTripletRootCut {
    bounds: (f32, f32),
    neurons: [usize; 3],
    weights: [f32; 3],
    multiplier: f32,
    support_bound: f32,
}
impl CompletedRawTripletRootCut {
    pub(crate) fn bounds(&self) -> (f32, f32) {
        self.bounds
    }
    pub(crate) fn support_calls(&self) -> usize {
        1
    }
    pub(crate) fn cut_calls(&self) -> usize {
        1
    }
    pub(crate) fn ordered_neurons(&self) -> [usize; 3] {
        self.neurons
    }
    pub(crate) fn weights(&self) -> [f32; 3] {
        self.weights
    }
    pub(crate) fn multiplier(&self) -> f32 {
        self.multiplier
    }
    pub(crate) fn support_bound(&self) -> f32 {
        self.support_bound
    }
}
fn valid_weights(w: [f32; 3]) -> bool {
    // Inspect bits before arithmetic; DAZ cannot turn a bad subnormal into zero.
    w.iter().all(|v| {
        v.to_bits() >= (1f32 / 256.).to_bits()
            && v.to_bits() <= 1f32.to_bits()
            && (f64::from(*v) * 256.).fract() == 0.
    })
}
fn indices_ok(n: [usize; 3], width: usize) -> bool {
    n[0] < n[1] && n[1] < n[2] && n[2] < width
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_raw_triplet_root_cut<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    neurons: [usize; 3],
    weights: [f32; 3],
    multiplier: f32,
    limits: RawRootCutLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<RawTripletRootCutQuote<'a>> {
    if !valid_weights(weights) || multiplier.to_bits() == 0 || multiplier.to_bits() >= 0x7f80_0000 {
        return None;
    }
    let mut base = quote_raw_root_cut(
        graph,
        input,
        bounds,
        alpha,
        objective,
        old,
        target,
        [neurons[0], neurons[1]],
        multiplier,
        limits,
        deadline,
        poll,
    )?;
    if !indices_ok(neurons, base.prefix_quote.output_dim) {
        return None;
    }
    let pre = bounds.get(base.pre_node)?;
    let (l, u) = (pre.lower().as_slice()?, pre.upper().as_slice()?);
    for i in neurons {
        if !ny_core::is_crown_coeff_safe(l[i]) || !ny_core::is_crown_coeff_safe(u[i]) || l[i] > u[i]
        {
            return None;
        }
    }
    let work = phase_work(base.prefix_quote, 4, 1)?
        .checked_add(phase_work(base.full_quote, 1, 1)?)?
        .checked_add(TRIPLET_CERTIFICATE_WORK)?
        .checked_add(ADVICE_FIXED_WORK.checked_mul(2)?)?
        .checked_add(
            ADVICE_WORK_PER_OUTPUT
                .checked_mul(2)?
                .checked_mul(u64::try_from(objective.len()).ok()?)?,
        )?
        .checked_add(EXECUTION_OBSERVATION_WORK)?
        .checked_add(TRIPLET_FIXED_WORK)?;
    let scratch = base.scratch.checked_add(TRIPLET_EXTRA_BYTES)?;
    if work > limits.max_work
        || scratch > limits.max_scratch_bytes
        || Instant::now() >= deadline
        || !poll()
    {
        return None;
    }
    base.candidate_count = 1;
    Some(RawTripletRootCutQuote {
        base,
        seal: ResidentCutCallSeal([0xD3]),
        neurons,
        weights,
        multiplier,
        work,
        scratch,
    })
}
fn signed_rows(w: [f32; 3]) -> [[f32; 3]; 4] {
    // Q8 sign changes are exact and no product rounding enters the seed.
    std::array::from_fn(|r| std::array::from_fn(|i| SIGNS[r][i] * w[i]))
}
struct FreshTripletSupport<'q> {
    quote_seal: &'q ResidentCutCallSeal,
    neurons: [usize; 3],
    weights: [f32; 3],
    rows: [[f32; 3]; 4],
    lower: [f32; 4],
    upper: [f32; 4],
    bound: f32,
}
fn sign_bound(l: [f32; 4], u: [f32; 4]) -> Option<f32> {
    let mut b = 0f32;
    for r in 0..4 {
        if !ny_core::is_crown_coeff_safe(l[r]) || !ny_core::is_crown_coeff_safe(u[r]) || l[r] > u[r]
        {
            return None;
        }
        b = b.max(u[r]).max(-l[r]);
    }
    // Zero is a conservative widening if a bad trusted backend supplied eight
    // negative maxima. Source authority still rests on the checked support API.
    ny_core::is_crown_coeff_safe(b).then_some(b)
}
fn same_weights(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
struct TripletContext<'q, 'a> {
    quote: &'q RawTripletRootCutQuote<'a>,
    base: ResidentCutCallContext<'a>,
    target_activation: usize,
    activation_count: usize,
    width: usize,
}
struct TripletCarrier<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    neurons: [usize; 3],
    weights: [f32; 3],
    bound: f32,
    transport: ResidentTripletLowerCutCarrier,
}
impl<'q, 'a> TripletContext<'q, 'a> {
    fn new(q: &'q RawTripletRootCutQuote<'a>, base: ResidentCutCallContext<'a>) -> Result<Self> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("triplet IEEE".into()));
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
            || base.beta_signed.iter().flatten().any(|&v| v != 0.)
            || !indices_ok(q.neurons, width)
            || !valid_weights(q.weights)
        {
            return Err(NyError::SoundnessRefusal(
                "triplet original snapshot/seed differs".into(),
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
    fn build<'c>(&'c self, fresh: &FreshTripletSupport<'_>) -> Result<TripletCarrier<'c>> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("triplet reduction IEEE".into()));
        }
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        let q = self.quote;
        if !std::ptr::eq(fresh.quote_seal, &raw const q.seal)
            || fresh.neurons != q.neurons
            || !same_weights(fresh.weights, q.weights)
            || fresh
                .rows
                .iter()
                .zip(signed_rows(q.weights))
                .any(|(&a, b)| !same_weights(a, b))
            || sign_bound(fresh.lower, fresh.upper).map(f32::to_bits) != Some(fresh.bound.to_bits())
            || !ny_core::is_crown_coeff_safe(q.multiplier)
            || q.multiplier <= 0.
        {
            return Err(NyError::SoundnessRefusal(
                "foreign triplet sign certificate".into(),
            ));
        }
        // sum_i w_i|z_i| = max over all eight sign forms. Four rows and their
        // negatives cover exactly these forms, on this immutable source/domain.
        let channel = |v: f32| {
            transport_channel(directed_reduce(std::iter::once((
                f64::from(q.multiplier),
                f64::from(v),
            )))?)
        };
        let pre = [
            channel(-q.weights[0])?,
            channel(-q.weights[1])?,
            channel(-q.weights[2])?,
        ];
        let post = [
            channel(2. * q.weights[0])?,
            channel(2. * q.weights[1])?,
            channel(2. * q.weights[2])?,
        ];
        let bias = channel(-fresh.bound)?;
        let transport = ResidentTripletLowerCutCarrier::try_new(
            self.target_activation,
            self.width,
            q.neurons,
            q.multiplier,
            pre,
            post,
            bias,
            self.base.deadline,
        )?;
        transport.validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        Ok(TripletCarrier {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            neurons: q.neurons,
            weights: q.weights,
            bound: fresh.bound,
            transport,
        })
    }
    fn run<'c>(
        &'c self,
        c: &TripletCarrier<'_>,
        gpu: &dyn GpuCrownBackward,
    ) -> Result<BoundResidentLowerResult<'c>> {
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        if !std::ptr::eq(c.seal, &raw const self.base.seal)
            || c.generations != self.base.generations.get()
            || c.neurons != self.quote.neurons
            || !same_weights(c.weights, self.quote.weights)
            || !ny_core::is_crown_coeff_safe(c.bound)
            || c.bound < 0.
            || !gpu.provides_bounded_triplet_lower_cut()
            || c.transport.ordered_neurons() != self.quote.neurons
            || c.transport.target_activation() != self.target_activation
            || !c.transport.has_nonzero_multiplier()
        {
            return Err(NyError::SoundnessRefusal(
                "foreign/stale/incompatible triplet carrier".into(),
            ));
        }
        c.transport
            .validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        self.base.advance_frontier_generation()?;
        let result = gpu.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
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
            return Err(NyError::SoundnessRefusal("unsafe triplet lower".into()));
        }
        Ok(BoundResidentLowerResult {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            lower,
        })
    }
}
pub(crate) fn evaluate_raw_triplet_root_cut(
    q: RawTripletRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CompletedRawTripletRootCut> {
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
        support_api_entered = obs.support_api_entered,
        support_api_ok = obs.support_api_ok,
        lower_api_entered = obs.lower_api_entered,
        lower_api_ok = obs.lower_api_ok,
        merges_ok = obs.merges_ok,
        poll_alive_at_return = guard.alive,
        deadline_expired_at_return = Instant::now() >= q.base.deadline,
        complete = result.is_some(),
        "Certified triplet root cut execution detail"
    );
    result
}
fn evaluate_steps(
    q: &RawTripletRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    guard: &mut Poll<'_>,
    obs: &mut ExecutionObservation,
) -> Option<CompletedRawTripletRootCut> {
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
    if !gpu.provides_bounded_support4() || !gpu.provides_bounded_triplet_lower_cut() {
        return None;
    }
    let b = &q.base;
    let lower = b.input.lower().as_slice()?;
    let upper = b.input.upper().as_slice()?;
    let fresh = {
        obs.phase = "prefix-extraction";
        let prefix = extract_charged_raw_bn_chain(
            b.graph,
            b.input,
            b.pre_node,
            b.bounds,
            b.bounds,
            b.alpha,
            b.extraction,
            b.deadline,
            &mut || guard.check(),
        )?;
        if prefix.quote != b.prefix_quote || !guard.check() {
            return None;
        }
        let d = prefix.output_dim;
        let rows = signed_rows(q.weights);
        let mut a = vec![0.; 4usize.checked_mul(d)?];
        for r in 0..4 {
            for k in 0..3 {
                a[r * d + q.neurons[k]] = rows[r][k];
            }
        }
        let seed = exact_seed(4, d, a);
        obs.phase = "support-api";
        obs.support_api_entered += 1;
        let support = obs.result(gpu.crown_backward_gpu_resnet_support4_with_deadline(
            &prefix.segments,
            &seed,
            lower,
            upper,
            b.deadline,
        ))?;
        obs.support_api_ok += 1;
        if !guard.check() {
            return None;
        }
        obs.phase = "sign-certificate";
        let l: [f32; 4] = support.lower_bounds.try_into().ok()?;
        let u: [f32; 4] = support.upper_bounds.try_into().ok()?;
        let bound = sign_bound(l, u)?;
        FreshTripletSupport {
            quote_seal: &q.seal,
            neurons: q.neurons,
            weights: q.weights,
            rows,
            lower: l,
            upper: u,
            bound,
        }
    }; // Drop complete prefix/seed host storage before full extraction; backend cache remains priced.
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
    obs.phase = "triplet-snapshot";
    let context = obs.result(TripletContext::new(q, base))?;
    if !guard.check() {
        return None;
    }
    obs.phase = "triplet-carrier";
    let carrier = obs.result(context.build(&fresh))?;
    obs.phase = "triplet-lower-api";
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
    obs.phase = "triplet-merge";
    let merged = obs.result(context.base.merge_backend_lower(result, &baseline))?;
    obs.merges_ok += 1;
    if merged.upper_bounds[0].to_bits() != b.old.1.to_bits() || !guard.check() {
        return None;
    }
    obs.phase = "complete";
    Some(CompletedRawTripletRootCut {
        bounds: (merged.lower_bounds[0], b.old.1),
        neurons: q.neurons,
        weights: q.weights,
        multiplier: q.multiplier,
        support_bound: fresh.bound,
    })
}
#[cfg(test)]
#[path = "root_call_triplet_tests.rs"]
pub(crate) mod tests;
