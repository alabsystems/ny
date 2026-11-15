// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One original objective, two disjoint pairs, one shared support call and one
//! complete joint fold. This module adds no query owner or backend permission.

use super::*;
use ny_core::ResidentTwoPairLowerCutCarrier;

// Conservative additions to the existing one-pair phase accounting. Both
// selected-column advice passes are fully paid below; root top-four selection
// belongs to the caller's setup lease and is not performed by this module.
const JOINT_EXTRA_BYTES: usize = 4 << 20;
const JOINT_FIXED_WORK: u64 = 1 << 20;
const JOINT_EXACT_WORK: u64 = 32_000_000;

/// Opaque borrowed source/shape authority. No full weight-tensor scans or
/// descriptor/map clones occur before the caller reserves the whole quote.
/// Bounded shape/objective/marginal checks require the caller's setup lease.
pub(crate) struct RawTwoPairRootCutQuote<'a> {
    base: RawRootCutQuote<'a>,
    seal: ResidentCutCallSeal,
    pairs: [[usize; 2]; 2],
    fallback_scales: [f32; 2],
    work: u64,
    scratch: usize,
}
impl RawTwoPairRootCutQuote<'_> {
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

/// Constructible only after both facets and the complete joint fold succeed.
pub(crate) struct CompletedRawTwoPairRootCut {
    bounds: (f32, f32),
    pairs: [[usize; 2]; 2],
    normals: [[f32; 4]; 2],
    scales: [f32; 2],
}
impl CompletedRawTwoPairRootCut {
    pub(crate) fn bounds(&self) -> (f32, f32) {
        self.bounds
    }
    pub(crate) fn support_calls(&self) -> usize {
        1
    }
    pub(crate) fn cut_calls(&self) -> usize {
        1
    }
    pub(crate) fn ordered_pairs(&self) -> [[usize; 2]; 2] {
        self.pairs
    }
    pub(crate) fn cut_normals(&self) -> [[f32; 4]; 2] {
        self.normals
    }
    pub(crate) fn lambda_scales(&self) -> [f32; 2] {
        self.scales
    }
}

fn distinct_pairs(pairs: [[usize; 2]; 2], width: usize) -> bool {
    let indices = [pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]];
    indices
        .iter()
        .enumerate()
        .all(|(n, &i)| i < width && !indices[..n].contains(&i))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_raw_two_pair_root_cut<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    alpha: Option<&'a GraphAlphaState>,
    objective: &'a [f32],
    old: (f32, f32),
    target: &'a str,
    pairs: [[usize; 2]; 2],
    fallback_scales: [f32; 2],
    limits: RawRootCutLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<RawTwoPairRootCutQuote<'a>> {
    // Bit validation must not mistake a subnormal for zero under DAZ. This
    // shape-only quote constructs no reduced channel or numerical proof.
    if fallback_scales.iter().any(|x| {
        let b = x.to_bits();
        b >> 31 != 0 || b == 0 || b >= 0x7f80_0000
    }) {
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
        pairs[0],
        fallback_scales[0],
        limits,
        deadline,
        poll,
    )?;
    if !distinct_pairs(pairs, base.prefix_quote.output_dim) {
        return None;
    }
    let pre = bounds.get(base.pre_node)?;
    let lower = pre.lower().as_slice()?;
    let upper = pre.upper().as_slice()?;
    for &i in pairs.iter().flatten() {
        if !ny_core::is_crown_coeff_safe(lower[i])
            || !ny_core::is_crown_coeff_safe(upper[i])
            || lower[i] > upper[i]
        {
            return None;
        }
    }
    // One four-row prefix and ONE full fold. Two exact checkers are used
    // sequentially, with both small certificates retained through publication.
    let work = phase_work(base.prefix_quote, 4, 1)?
        .checked_add(phase_work(base.full_quote, 1, 1)?)?
        .checked_add(JOINT_EXACT_WORK)?
        .checked_add(ADVICE_FIXED_WORK.checked_mul(2)?)?
        .checked_add(
            ADVICE_WORK_PER_OUTPUT
                .checked_mul(2)?
                .checked_mul(u64::try_from(objective.len()).ok()?)?,
        )?
        .checked_add(EXECUTION_OBSERVATION_WORK)?
        .checked_add(JOINT_FIXED_WORK)?;
    let scratch = base.scratch.checked_add(JOINT_EXTRA_BYTES)?;
    if work > limits.max_work
        || scratch > limits.max_scratch_bytes
        || Instant::now() >= deadline
        || !poll()
    {
        return None;
    }
    base.candidate_count = 1;
    Some(RawTwoPairRootCutQuote {
        base,
        seal: ResidentCutCallSeal([0xD2]),
        pairs,
        fallback_scales,
        work,
        scratch,
    })
}

// This temporary contains only immutable borrows and fixed Copy shape quotes.
// It cannot allocate a source clone or acquire new provenance. Both calls to
// the original bounded two-column advice walk are paid by the joint quote.
fn second_advice_view<'a>(q: &RawTwoPairRootCutQuote<'a>) -> RawRootCutQuote<'a> {
    let b = &q.base;
    RawRootCutQuote {
        graph: b.graph,
        input: b.input,
        bounds: b.bounds,
        alpha: b.alpha,
        objective: b.objective,
        old: b.old,
        target: b.target,
        pre_node: b.pre_node,
        pair: q.pairs[1],
        lambda_scale: q.fallback_scales[1],
        candidate_count: 1,
        extraction: b.extraction,
        prefix_quote: b.prefix_quote,
        full_quote: b.full_quote,
        work: b.work,
        scratch: b.scratch,
        deadline: b.deadline,
    }
}

struct FreshJointSupport<'q> {
    quote_seal: &'q ResidentCutCallSeal,
    pairs: [[usize; 2]; 2],
    facets: [FreshSupport; 2],
}

// One shared snapshot, one nonzero seal, one generation state. No second
// ResidentCutCallContext and no repeated source/decomposition validation.
struct JointContext<'q, 'a> {
    quote: &'q RawTwoPairRootCutQuote<'a>,
    base: ResidentCutCallContext<'a>,
    target_activation: usize,
    activation_count: usize,
    width: usize,
}
struct JointCarrier<'c> {
    seal: &'c ResidentCutCallSeal,
    generations: ResidentCutSnapshotGenerations,
    pairs: [[usize; 2]; 2],
    octahedra: [Octahedron2; 2],
    certificates: [ExactRelu2FacetCertificate; 2],
    transport: ResidentTwoPairLowerCutCarrier,
}

fn source_marginals(q: &RawTwoPairRootCutQuote<'_>, pair: [usize; 2]) -> Result<[f32; 4]> {
    let b = q
        .base
        .bounds
        .get(q.base.pre_node)
        .ok_or_else(|| NyError::SoundnessRefusal("joint pre-node disappeared".into()))?;
    let l = b
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::SoundnessRefusal("joint lower layout".into()))?;
    let u = b
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::SoundnessRefusal("joint upper layout".into()))?;
    if pair.iter().any(|&i| i >= l.len() || i >= u.len()) {
        return Err(NyError::SoundnessRefusal("joint pre-node width".into()));
    }
    let values = [l[pair[0]], u[pair[0]], l[pair[1]], u[pair[1]]];
    if values.iter().any(|&v| !ny_core::is_crown_coeff_safe(v))
        || values[0] > values[1]
        || values[2] > values[3]
    {
        return Err(NyError::SoundnessRefusal(
            "joint original marginals unsafe".into(),
        ));
    }
    Ok(values)
}

impl<'q, 'a> JointContext<'q, 'a> {
    fn new(
        quote: &'q RawTwoPairRootCutQuote<'a>,
        base: ResidentCutCallContext<'a>,
    ) -> Result<Self> {
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal("joint IEEE selfcheck".into()));
        }
        let (target_activation, widths) = base.validate_resident_snapshot()?;
        LowerSeedRef::from_seed(base.seed).validate()?;
        let width = widths[target_activation];
        let q = &quote.base;
        if !std::ptr::eq(base.graph, q.graph)
            || !std::ptr::eq(base.input, q.input)
            || !std::ptr::eq(base.node_bounds, q.bounds)
            || base.target_relu != q.target
            || base.ordered_neurons != quote.pairs[0]
            || base.deadline != q.deadline
            || base.seed.num_specs != 1
            || base.seed.current_dim != q.objective.len()
            || !slice_bits_equal(Some(q.objective), &base.seed.lower_a)
            || !slice_bits_equal(Some(q.objective), &base.seed.upper_a)
            || base.seed.lower_b.len() != 1
            || base.seed.upper_b.len() != 1
            || base.seed.lower_b[0].to_bits() != 0
            || base.seed.upper_b[0].to_bits() != 0
            || base.beta_signed.iter().flatten().any(|&v| v != 0.)
            || !distinct_pairs(quote.pairs, width)
        {
            return Err(NyError::SoundnessRefusal(
                "joint original snapshot/seed differs".into(),
            ));
        }
        for pair in quote.pairs {
            source_marginals(quote, pair)?;
        }
        Ok(Self {
            quote,
            base,
            target_activation,
            activation_count: widths.len(),
            width,
        })
    }

    fn build<'c>(
        &'c self,
        fresh: &FreshJointSupport<'_>,
        advice: [CutAdvice; 2],
    ) -> Result<JointCarrier<'c>> {
        // Required BEFORE reducing f32 products or constructing old channels.
        ny_core::require_f64_interval_proof_environment()?;
        if !dd_selfcheck_ok() {
            return Err(NyError::SoundnessRefusal(
                "joint reduction IEEE selfcheck".into(),
            ));
        }
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        if !std::ptr::eq(fresh.quote_seal, &raw const self.quote.seal)
            || fresh.pairs != self.quote.pairs
        {
            return Err(NyError::SoundnessRefusal(
                "foreign joint support source/pair layout".into(),
            ));
        }
        for n in 0..2 {
            let f = &fresh.facets[n];
            let expected = source_marginals(self.quote, self.quote.pairs[n])?;
            let actual = [
                f.octahedron.l1,
                f.octahedron.u1,
                f.octahedron.l2,
                f.octahedron.u2,
            ];
            if !octahedron_bits_equal(f.certificate.support_domain(), &f.octahedron)
                || actual
                    .iter()
                    .zip(expected)
                    .any(|(&a, b)| a.to_bits() != f64::from(b).to_bits())
                || f.certificate
                    .facet()
                    .a
                    .iter()
                    .zip(advice[n].normal)
                    .any(|(&a, b)| a.to_bits() != b.to_bits())
                || !ny_core::is_crown_coeff_safe(advice[n].lambda_scale)
                || advice[n].lambda_scale <= 0.
            {
                return Err(NyError::SoundnessRefusal(
                    "joint certificate/marginal/advice identity".into(),
                ));
            }
        }
        let (pre, post, bias) = reduce_two_facets(&fresh.facets, advice)?;
        let transport = ResidentTwoPairLowerCutCarrier::try_new(
            self.target_activation,
            self.width,
            self.quote.pairs,
            [advice[0].lambda_scale, advice[1].lambda_scale],
            pre,
            post,
            bias,
            self.base.deadline,
        )?;
        transport.validate_for_call(self.activation_count, self.width, 1, self.base.deadline)?;
        Ok(JointCarrier {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            pairs: self.quote.pairs,
            octahedra: [
                fresh.facets[0].octahedron.clone(),
                fresh.facets[1].octahedron.clone(),
            ],
            certificates: [
                fresh.facets[0].certificate.clone(),
                fresh.facets[1].certificate.clone(),
            ],
            transport,
        })
    }

    fn run<'c>(
        &'c self,
        carrier: &JointCarrier<'_>,
        gpu: &dyn GpuCrownBackward,
    ) -> Result<BoundResidentLowerResult<'c>> {
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforeValidation)?;
        if !std::ptr::eq(carrier.seal, &raw const self.base.seal)
            || carrier.generations != self.base.generations.get()
            || carrier.pairs != self.quote.pairs
            || !gpu.provides_bounded_two_pair_lower_cut()
            || carrier.transport.ordered_pairs() != self.quote.pairs
            || carrier.transport.target_activation() != self.target_activation
            || !carrier.transport.has_nonzero_multiplier()
            || (0..2).any(|n| {
                !octahedron_bits_equal(
                    carrier.certificates[n].support_domain(),
                    &carrier.octahedra[n],
                )
            })
        {
            return Err(NyError::SoundnessRefusal(
                "foreign/stale/incompatible joint carrier".into(),
            ));
        }
        carrier.transport.validate_for_call(
            self.activation_count,
            self.width,
            1,
            self.base.deadline,
        )?;
        self.base.advance_frontier_generation()?;
        let result = gpu.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            self.base.segments,
            self.base.seed,
            self.base.resident_input_lower,
            self.base.resident_input_upper,
            &carrier.transport,
            self.base.deadline,
        );
        self.base.advance_frontier_generation()?; // Refusal also consumes the carrier.
        let lower = result?;
        check_deadline(self.base.deadline, AuthorityDeadlineStage::BeforePublish)?;
        if !ny_core::is_crown_coeff_safe(lower) {
            return Err(NyError::SoundnessRefusal(
                "unsafe complete joint lower".into(),
            ));
        }
        Ok(BoundResidentLowerResult {
            seal: &self.base.seal,
            generations: self.base.generations.get(),
            lower,
        })
    }
}

fn reduce_two_facets(
    facets: &[FreshSupport; 2],
    advice: [CutAdvice; 2],
) -> Result<(
    [ResidentLowerCutChannel; 4],
    [ResidentLowerCutChannel; 4],
    ResidentLowerCutChannel,
)> {
    ny_core::require_f64_interval_proof_environment()?;
    if !dd_selfcheck_ok() {
        return Err(NyError::SoundnessRefusal(
            "joint channel IEEE selfcheck".into(),
        ));
    }
    let channel = |pair: usize, coordinate: usize| {
        transport_channel(directed_reduce(std::iter::once((
            f64::from(advice[pair].lambda_scale),
            f64::from(facets[pair].certificate.facet().a[coordinate]),
        )))?)
    };
    let pre = [
        channel(0, 0)?,
        channel(0, 1)?,
        channel(1, 0)?,
        channel(1, 1)?,
    ];
    let post = [
        channel(0, 2)?,
        channel(0, 3)?,
        channel(1, 2)?,
        channel(1, 3)?,
    ];
    // Reduce BOTH original exact f32 products, never add two rounded centers.
    let bias = transport_channel(directed_reduce((0..2).map(|n| {
        (
            f64::from(advice[n].lambda_scale),
            -f64::from(facets[n].certificate.facet().b),
        )
    }))?)?;
    Ok((pre, post, bias))
}

pub(crate) fn evaluate_raw_two_pair_root_cut(
    q: RawTwoPairRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    poll: &mut dyn FnMut() -> bool,
) -> Option<CompletedRawTwoPairRootCut> {
    let mut guard = Poll {
        end: q.base.deadline,
        callback: poll,
        alive: true,
    };
    let mut observation = ExecutionObservation::default();
    let result = evaluate_steps(&q, engine, &mut guard, &mut observation);
    observation.poll_alive_at_return = guard.alive;
    observation.deadline_expired_at_return = Instant::now() >= q.base.deadline;
    // Separate message: a partial pair never appears as an old completed cut.
    tracing::info!(
        phase = observation.phase,
        error_class = observation.error_class,
        support_api_entered = observation.support_api_entered,
        support_api_ok = observation.support_api_ok,
        lower_api_entered = observation.lower_api_entered,
        lower_api_ok = observation.lower_api_ok,
        merges_ok = observation.merges_ok,
        poll_alive_at_return = observation.poll_alive_at_return,
        deadline_expired_at_return = observation.deadline_expired_at_return,
        complete = result.is_some(),
        "Certified two-pair root cut execution detail"
    );
    result
}

fn evaluate_steps(
    q: &RawTwoPairRootCutQuote<'_>,
    engine: &dyn GemmEngine,
    guard: &mut Poll<'_>,
    obs: &mut ExecutionObservation,
) -> Option<CompletedRawTwoPairRootCut> {
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
    if !gpu.provides_bounded_support4() || !gpu.provides_bounded_two_pair_lower_cut() {
        return None;
    }
    obs.phase = "advice";
    let advice = [
        choose_advice(&q.base, guard)?,
        choose_advice(&second_advice_view(q), guard)?,
    ];
    if !guard.check() {
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
        let mut a = vec![0.; 4usize.checked_mul(d)?];
        for n in 0..2 {
            let [i, j] = q.pairs[n];
            a[2 * n * d + i] = 1.;
            a[2 * n * d + j] = 1.;
            a[(2 * n + 1) * d + i] = 1.;
            a[(2 * n + 1) * d + j] = -1.;
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
        if !guard.check() || support.lower_bounds.len() != 4 || support.upper_bounds.len() != 4 {
            return None;
        }
        for (&l, &u) in support.lower_bounds.iter().zip(&support.upper_bounds) {
            if !ny_core::is_crown_coeff_safe(l) || !ny_core::is_crown_coeff_safe(u) || l > u {
                return None;
            }
        }
        let mut facets = Vec::with_capacity(2);
        for n in 0..2 {
            obs.phase = if n == 0 {
                "first-facet"
            } else {
                "second-facet"
            };
            if !guard.check() {
                return None;
            }
            let [l1, u1, l2, u2] = obs.result(source_marginals(q, q.pairs[n]))?;
            let octahedron = Octahedron2 {
                l1: f64::from(l1),
                u1: f64::from(u1),
                l2: f64::from(l2),
                u2: f64::from(u2),
                s_lo: f64::from(support.lower_bounds[2 * n]),
                s_hi: f64::from(support.upper_bounds[2 * n]),
                d_lo: f64::from(support.lower_bounds[2 * n + 1]),
                d_hi: f64::from(support.upper_bounds[2 * n + 1]),
            };
            // All eight coordinates are exact safe-f32 widenings. Each Q8
            // checker has the same <=180 candidates/1024-bit arithmetic quote
            // as the old path. Stable, boundary and degenerate support is legal.
            let checker = ExactRelu2Support::new(&octahedron)?;
            let certificate = checker.certify_normal_certificate(advice[n].normal)?;
            facets.push(FreshSupport {
                octahedron,
                certificate,
            });
        }
        if !guard.check() {
            return None;
        }
        let facets: [FreshSupport; 2] = facets.try_into().ok()?;
        FreshJointSupport {
            quote_seal: &q.seal,
            pairs: q.pairs,
            facets,
        }
    }; // Prefix/checker/seed host allocations drop; cached backend buffers paid.
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
        q.pairs[0],
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        lower,
        upper,
        b.deadline,
    );
    obs.phase = "joint-snapshot";
    let context = obs.result(JointContext::new(q, base))?;
    if !guard.check() {
        return None;
    }
    obs.phase = "joint-carrier";
    let carrier = obs.result(context.build(&fresh, advice))?;
    obs.phase = "joint-lower-api";
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
    obs.phase = "joint-merge";
    let merged = obs.result(context.base.merge_backend_lower(result, &baseline))?;
    obs.merges_ok += 1;
    if merged.upper_bounds[0].to_bits() != b.old.1.to_bits() || !guard.check() {
        return None;
    }
    obs.phase = "complete";
    Some(CompletedRawTwoPairRootCut {
        bounds: (merged.lower_bounds[0], b.old.1),
        pairs: q.pairs,
        normals: [advice[0].normal, advice[1].normal],
        scales: [advice[0].lambda_scale, advice[1].lambda_scale],
    })
}

#[cfg(test)]
#[path = "root_call_two_pair_tests.rs"]
pub(crate) mod tests;
