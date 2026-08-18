// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Non-authoritative Cut-CROWN shadow orchestration and host reference seam.
//!
//! The historical host-reference wrapper below has only the default
//! [`CertifiedCutShadowPolicy::Disabled`] production state; its `Shadow` permit
//! remains test-only. The separate M1 entry constructs a live semantic context
//! and calls the optimized CUDA resident trait override, still observation-only.
//! This prevents the already-measured weak f64 side-backward from being
//! mislabeled as a resident Cut-CROWN result.

use std::collections::HashMap;
use std::time::Instant;

#[cfg(test)]
use ny_core::dd::next_up_f64;
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownResult, GpuCrownSeed, GpuResnetSegment, NyError,
    ResidentCutShadowObservation, ResidentCutShadowOutcome, Result,
};
#[cfg(test)]
use ny_tensor::next_up_f32;
use ny_tensor::BoundedTensor;

#[cfg(test)]
use super::certified_cut_authority::BoundResidentLowerCutCarrier;
use super::certified_cut_authority::{ResidentCutCallContext, ResidentCutSnapshotGenerations};
use super::{
    certified_coupling_facet_certificates_exact_with_deadline,
    combined_row_octahedron_with_deadline, ExactRelu2FacetCertificate,
};
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

/// Typed policy for the semantic authority wrapper.
#[derive(Clone, Copy, Debug, Default)]
pub(super) enum CertifiedCutShadowPolicy<'permit> {
    /// Execute only the unchanged baseline closure.
    #[default]
    Disabled,
    /// Execute a reference observation after the baseline completes.
    Shadow(&'permit ReferenceShadowPermit),
}

/// Non-production authority for the host reference oracle.
///
/// It is private, non-cloneable, and constructible only in this module's unit
/// tests.  No CLI/config/environment path can manufacture it.
#[derive(Debug)]
pub(super) struct ReferenceShadowPermit {
    _private: (),
}

/// Exact default-dark gate for the production observation lane.
///
/// This gate cannot affect a verdict: the backend outcome exposes the ordinary
/// beta fold as its only consumable bound. It merely authorizes the extra exact
/// support construction and scratch resident pass.
pub(crate) fn production_resident_cut_shadow_enabled() -> bool {
    std::env::var("NY_CUT_CROWN_RESIDENT_SHADOW")
        .ok()
        .as_deref()
        == Some("1")
}

/// Same-call semantic inputs for one real resident Cut-CROWN observation.
///
/// Every field is borrowed from the serial domain call immediately surrounding
/// the backend dispatch. No value is serializable, cloneable as a request, or
/// stored in a process-global registry.
#[allow(clippy::too_many_arguments)]
pub(crate) struct ProductionResidentCutShadowRequest<'a> {
    pub(crate) graph: &'a GraphNetwork,
    pub(crate) input: &'a BoundedTensor,
    pub(crate) alpha_state: &'a GraphAlphaState,
    pub(crate) node_bounds: &'a HashMap<String, BoundedTensor>,
    pub(crate) engine: &'a dyn GemmEngine,
    pub(crate) gpu: &'a dyn GpuCrownBackward,
    pub(crate) seed: &'a GpuCrownSeed,
    pub(crate) segments: &'a [GpuResnetSegment],
    pub(crate) relu_names: &'a [String],
    pub(crate) beta_signed: &'a [Vec<f32>],
    pub(crate) frontier_abs: &'a [Vec<f32>],
    pub(crate) node_abs: &'a [Vec<f32>],
    pub(crate) resident_input_lower: &'a [f32],
    pub(crate) resident_input_upper: &'a [f32],
    pub(crate) binding_row: usize,
    pub(crate) deadline: Instant,
}

/// Produce one deterministic, exact-certified k=2 candidate from the live
/// domain and execute it synchronously on the CUDA resident shadow backend.
///
/// Candidate policy is intentionally bounded for M1: scan the exact resident
/// ReLU order, take the first target with two unstable pre-activation neurons,
/// select its two widest neurons, exact-certify the strongest coupling facet,
/// and use `lambda=1` for every lower objective row. A policy miss is a clean
/// `UnsupportedOp`; it never substitutes a weaker side-backward or raw facet.
pub(crate) fn run_production_resident_cut_shadow(
    request: ProductionResidentCutShadowRequest<'_>,
) -> Result<ResidentCutShadowOutcome> {
    if Instant::now() >= request.deadline {
        return Err(NyError::DeadlineExceeded(
            "resident Cut-CROWN shadow expired before candidate selection".into(),
        ));
    }
    if request.binding_row >= request.seed.num_specs {
        return Err(NyError::InvalidSpec(
            "resident Cut-CROWN shadow binding row is outside the objective seed".into(),
        ));
    }
    if let Some(projected) =
        super::certified_cut_m2_shadow::maybe_run_production_resident_cut_m2_projected(&request)
    {
        return projected;
    }
    let (target_relu, ordered_neurons, certificates) = select_live_candidate(&request)?;
    let row_lambdas = vec![vec![1.0_f32; certificates.len()]; request.seed.num_specs];

    let context = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        request.graph,
        request.input,
        request.alpha_state,
        request.node_bounds,
        Some(request.engine),
        request.seed,
        target_relu,
        ordered_neurons,
        request.segments,
        request.relu_names,
        request.beta_signed,
        request.frontier_abs,
        request.node_abs,
        request.resident_input_lower,
        request.resident_input_upper,
        request.deadline,
    );
    let carrier = context
        .build_bound_carrier(&certificates, &row_lambdas)?
        .ok_or_else(|| {
            NyError::SoundnessRefusal(
                "resident Cut-CROWN shadow candidate reduced to an all-zero carrier".into(),
            )
        })?;
    context.run_backend_shadow(&carrier, request.gpu, request.binding_row)
}

fn select_live_candidate<'a>(
    request: &ProductionResidentCutShadowRequest<'a>,
) -> Result<(&'a str, [usize; 2], Vec<ExactRelu2FacetCertificate>)> {
    if request.relu_names.is_empty() || request.relu_names.len() != request.beta_signed.len() {
        return Err(NyError::InvalidSpec(
            "resident Cut-CROWN shadow has inconsistent ReLU decomposition".into(),
        ));
    }
    for target_relu in request.relu_names {
        if Instant::now() >= request.deadline {
            return Err(NyError::DeadlineExceeded(
                "resident Cut-CROWN shadow expired during target selection".into(),
            ));
        }
        let Some(node) = request.graph.node(target_relu) else {
            continue;
        };
        if !matches!(node.layer(), crate::layers::Layer::ReLU(_)) {
            continue;
        }
        let Ok(pre_node) = node.require_unary_input() else {
            continue;
        };
        let Some(pre_bounds) = request.node_bounds.get(pre_node) else {
            continue;
        };
        let flat = pre_bounds.flatten();
        let mut unstable = flat
            .lower()
            .iter()
            .zip(flat.upper().iter())
            .enumerate()
            .filter_map(|(index, (&lower, &upper))| {
                let width = upper - lower;
                (lower < 0.0 && upper > 0.0 && width.is_finite()).then_some((index, width))
            })
            .collect::<Vec<_>>();
        unstable.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let Some(&(first, _)) = unstable.first() else {
            continue;
        };
        let Some(&(second, _)) = unstable.get(1) else {
            continue;
        };
        let ordered_neurons = [first, second];
        let support = combined_row_octahedron_with_deadline(
            request.graph,
            request.input,
            request.alpha_state,
            Some(request.node_bounds),
            pre_node,
            first,
            second,
            Some(request.engine),
            Some(request.deadline),
        )?;
        let certificates = certified_coupling_facet_certificates_exact_with_deadline(
            &support,
            Some(request.deadline),
        )?;
        let strongest = certificates.into_iter().max_by(|left, right| {
            let post_strength = |certificate: &ExactRelu2FacetCertificate| {
                let facet = certificate.facet();
                facet.a[2].abs() + facet.a[3].abs()
            };
            post_strength(left)
                .partial_cmp(&post_strength(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(strongest) = strongest {
            return Ok((target_relu.as_str(), ordered_neurons, vec![strongest]));
        }
    }
    Err(NyError::UnsupportedOp(
        "resident Cut-CROWN shadow found no exact-certified live k=2 candidate".into(),
    ))
}

#[cfg(test)]
impl ReferenceShadowPermit {
    const fn for_test() -> Self {
        Self { _private: () }
    }
}

/// Run the typed shadow seam.
///
/// The disabled branch returns before `shadow` is invoked, so certificate
/// construction, cut-specific decomposition, and reference arithmetic remain
/// unexecuted.  Any shadow refusal is a telemetry miss and returns the exact
/// baseline with no observation.
#[allow(unused_variables)]
pub(super) fn run_certified_cut_shadow<B, S>(
    policy: CertifiedCutShadowPolicy<'_>,
    baseline: B,
    shadow: S,
) -> Result<ResidentCutShadowOutcome>
where
    B: FnOnce() -> Result<GpuCrownResult>,
    S: FnOnce(&ReferenceShadowPermit, &GpuCrownResult) -> Result<ResidentCutShadowObservation>,
{
    let baseline = baseline()?;
    match policy {
        CertifiedCutShadowPolicy::Disabled => Ok(ResidentCutShadowOutcome::disabled(baseline)),
        CertifiedCutShadowPolicy::Shadow(permit) => match shadow(permit, &baseline) {
            Ok(observation)
                if baseline
                    .lower_bounds
                    .get(observation.binding_row())
                    .is_some_and(|value| {
                        value.to_bits() == observation.baseline_lower().to_bits()
                    }) =>
            {
                // The guard above establishes every fallible baseline-binding
                // condition while retaining ownership of `baseline`.
                ResidentCutShadowOutcome::try_observed(baseline, observation)
            }
            Ok(_) | Err(_) => Ok(ResidentCutShadowOutcome::rejected(baseline)),
        },
    }
}

/// Center/error pair used only by the host reference mutation.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct ReferenceDirectedLower {
    center: f32,
    abs_error: f32,
}

#[cfg(test)]
impl ReferenceDirectedLower {
    fn try_new(center: f32, abs_error: f32) -> Result<Self> {
        if !center.is_finite() || !abs_error.is_finite() || abs_error < 0.0 {
            return Err(NyError::NumericalInstability(
                "cut shadow reference frontier contains invalid center/error".into(),
            ));
        }
        Ok(Self {
            center: if center == 0.0 { 0.0 } else { center },
            abs_error: if abs_error == 0.0 { 0.0 } else { abs_error },
        })
    }
}

/// Lower-only target frontier for the test/reference backend.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
struct ReferenceLowerFrontier {
    rows: usize,
    width: usize,
    post: Vec<ReferenceDirectedLower>,
    pre: Vec<ReferenceDirectedLower>,
    bias: Vec<ReferenceDirectedLower>,
}

#[cfg(test)]
impl ReferenceLowerFrontier {
    fn try_new(
        rows: usize,
        width: usize,
        post: Vec<ReferenceDirectedLower>,
        pre: Vec<ReferenceDirectedLower>,
        bias: Vec<ReferenceDirectedLower>,
    ) -> Result<Self> {
        let expected = rows.checked_mul(width).ok_or_else(|| {
            NyError::InvalidSpec("cut shadow reference frontier shape overflow".into())
        })?;
        if rows == 0
            || width == 0
            || post.len() != expected
            || pre.len() != expected
            || bias.len() != rows
        {
            return Err(NyError::InvalidSpec(
                "cut shadow reference frontier has inconsistent lower-only shapes".into(),
            ));
        }
        Ok(Self {
            rows,
            width,
            post,
            pre,
            bias,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceMutationStage {
    BeforeValidation,
    AfterValidation,
    AfterRow(usize),
    BeforePublish,
}

/// Apply all pre/post/bias channels to scratch storage, then publish atomically.
#[cfg(test)]
fn apply_reference_lower_cut(
    context: &ResidentCutCallContext<'_>,
    carrier: &BoundResidentLowerCutCarrier<'_, '_>,
    frontier: &mut ReferenceLowerFrontier,
) -> Result<()> {
    let deadline = context.deadline();
    let mut check = |_| {
        if Instant::now() >= deadline {
            Err(NyError::DeadlineExceeded(
                "cut shadow reference mutation exceeded its deadline".into(),
            ))
        } else {
            Ok(())
        }
    };
    apply_reference_lower_cut_with(context, carrier, frontier, &mut check)
}

#[cfg(test)]
fn apply_reference_lower_cut_with<C>(
    context: &ResidentCutCallContext<'_>,
    carrier: &BoundResidentLowerCutCarrier<'_, '_>,
    frontier: &mut ReferenceLowerFrontier,
    check: &mut C,
) -> Result<()>
where
    C: FnMut(ReferenceMutationStage) -> Result<()>,
{
    check(ReferenceMutationStage::BeforeValidation)?;
    context.validate_bound_carrier(carrier)?;
    let transport = context.transport(carrier)?;
    if frontier.rows != transport.rows().len() || frontier.width != transport.target_width() {
        return Err(NyError::InvalidSpec(
            "cut shadow reference frontier does not match the complete carrier".into(),
        ));
    }
    let expected = frontier.rows.checked_mul(frontier.width).ok_or_else(|| {
        NyError::InvalidSpec("cut shadow reference frontier shape overflow".into())
    })?;
    if frontier.post.len() != expected
        || frontier.pre.len() != expected
        || frontier.bias.len() != frontier.rows
        || frontier
            .post
            .iter()
            .chain(&frontier.pre)
            .chain(&frontier.bias)
            .any(|value| {
                !value.center.is_finite() || !value.abs_error.is_finite() || value.abs_error < 0.0
            })
    {
        return Err(NyError::InvalidSpec(
            "cut shadow reference frontier failed complete pre-mutation validation".into(),
        ));
    }
    check(ReferenceMutationStage::AfterValidation)?;

    // All arithmetic lands in scratch.  A late deadline, malformed channel, or
    // overflow below leaves the caller's frontier byte-identical.
    let mut scratch = frontier.clone();
    let pair = transport.ordered_neurons();
    for (row_index, row) in transport.rows().iter().enumerate() {
        for pair_position in 0..2 {
            let column = pair[pair_position];
            let index = row_index * frontier.width + column;
            scratch.post[index] =
                reference_add_lower(scratch.post[index], row.post()[pair_position])?;
            scratch.pre[index] = reference_add_lower(scratch.pre[index], row.pre()[pair_position])?;
        }
        // Bias is applied exactly once per row, after both coefficient halves.
        scratch.bias[row_index] = reference_add_lower(scratch.bias[row_index], row.bias())?;
        check(ReferenceMutationStage::AfterRow(row_index))?;
    }
    check(ReferenceMutationStage::BeforePublish)?;
    *frontier = scratch;
    Ok(())
}

/// Add one stored source channel and charge the resident f32 mutation outward.
#[cfg(test)]
fn reference_add_lower(
    base: ReferenceDirectedLower,
    source: ny_core::ResidentLowerCutChannel,
) -> Result<ReferenceDirectedLower> {
    let intended_center = f64::from(base.center) + f64::from(source.value());
    if !intended_center.is_finite() {
        return Err(NyError::NumericalInstability(
            "cut shadow reference resident add overflowed".into(),
        ));
    }
    let stored_center = intended_center as f32;
    if !stored_center.is_finite() {
        return Err(NyError::NumericalInstability(
            "cut shadow reference resident center is not finite f32".into(),
        ));
    }
    let mutation_gap = (f64::from(stored_center) - intended_center).abs();
    let mut total_error = add_nonnegative_up(
        f64::from(base.abs_error),
        f64::from(source.source_abs_error()),
    )?;
    total_error = add_nonnegative_up(total_error, mutation_gap)?;
    ReferenceDirectedLower::try_new(stored_center, f64_to_f32_up(total_error)?)
}

#[cfg(test)]
fn add_nonnegative_up(left: f64, right: f64) -> Result<f64> {
    if !left.is_finite() || !right.is_finite() || left < 0.0 || right < 0.0 {
        return Err(NyError::NumericalInstability(
            "cut shadow reference error term is invalid".into(),
        ));
    }
    let sum = left + right;
    if !sum.is_finite() {
        return Err(NyError::NumericalInstability(
            "cut shadow reference error accumulation overflowed".into(),
        ));
    }
    Ok(if sum == 0.0 { 0.0 } else { next_up_f64(sum) })
}

#[cfg(test)]
fn f64_to_f32_up(value: f64) -> Result<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(NyError::NumericalInstability(
            "cut shadow reference error conversion input is invalid".into(),
        ));
    }
    if value == 0.0 {
        return Ok(0.0);
    }
    let mut encoded = value as f32;
    if !encoded.is_finite() {
        return Err(NyError::NumericalInstability(
            "cut shadow reference error does not fit finite f32".into(),
        ));
    }
    if f64::from(encoded) < value {
        encoded = next_up_f32(encoded);
    }
    if !encoded.is_finite() || f64::from(encoded) < value {
        return Err(NyError::NumericalInstability(
            "cut shadow reference directed conversion failed".into(),
        ));
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    use ndarray::{arr1, arr2};

    use super::*;
    use crate::bounds::GraphAlphaState;
    use crate::layers::activations::ReLULayer;
    use crate::multineuron::certified_cut_authority::ResidentCutSnapshotGenerations;
    use crate::multineuron::{
        combined_row_octahedron_with_deadline, ExactRelu2FacetCertificate, ExactRelu2Support,
    };
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};
    use ny_core::{GpuCrownLayer, GpuCrownSeed, GpuResnetSegment, ResidentCutShadowDisposition};
    use ny_tensor::BoundedTensor;

    struct DiamondFixture {
        graph: GraphNetwork,
        input: BoundedTensor,
        bounds: HashMap<String, BoundedTensor>,
        alpha: GraphAlphaState,
        seed: GpuCrownSeed,
        segments: Vec<GpuResnetSegment>,
        relu_names: Vec<String>,
        beta_signed: Vec<Vec<f32>>,
        frontier_abs: Vec<Vec<f32>>,
        node_abs: Vec<Vec<f32>>,
        input_lower: Vec<f32>,
        input_upper: Vec<f32>,
    }

    impl DiamondFixture {
        fn new() -> Self {
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input(
                "pre",
                Layer::Linear(
                    LinearLayer::new(
                        arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]),
                        Some(arr1(&[0.0_f32, 0.0])),
                    )
                    .expect("finite diamond linear"),
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
            .expect("finite diamond input");
            let bounds = graph
                .collect_node_bounds_with_engine(&input, None)
                .expect("diamond IBP");
            let activation = GpuCrownLayer::Activation {
                lower_slope: vec![0.0, 0.0],
                upper_slope: vec![0.5, 0.5],
                lower_intercept: vec![0.0, 0.0],
                upper_intercept: vec![1.0, 1.0],
                num_neurons: 2,
            };
            let linear = GpuCrownLayer::Linear {
                weight: Arc::from([1.0_f32, 1.0, 1.0, -1.0]),
                bias: Some(Arc::from([0.0_f32, 0.0])),
                out_features: 2,
                in_features: 2,
                cert_err: Default::default(),
            };
            Self {
                graph,
                input,
                bounds,
                alpha: GraphAlphaState::new(),
                seed: GpuCrownSeed {
                    lower_a: Arc::from([-1.0_f32, -1.0]),
                    upper_a: Arc::from([-1.0_f32, -1.0]),
                    lower_b: Arc::from([0.0_f32]),
                    upper_b: Arc::from([0.0_f32]),
                    num_specs: 1,
                    current_dim: 2,
                },
                segments: vec![GpuResnetSegment::Chain(vec![activation, linear])],
                relu_names: vec!["relu".into()],
                beta_signed: vec![vec![0.0, 0.0]],
                frontier_abs: vec![vec![1.0, 1.0]],
                node_abs: vec![vec![2.0, 2.0]],
                input_lower: vec![-1.0, -1.0],
                input_upper: vec![1.0, 1.0],
            }
        }

        fn context(&self, deadline: Instant) -> ResidentCutCallContext<'_> {
            ResidentCutCallContext::new(
                ResidentCutSnapshotGenerations::fixture(),
                &self.graph,
                &self.input,
                &self.alpha,
                &self.bounds,
                None,
                &self.seed,
                "relu",
                [0, 1],
                &self.segments,
                &self.relu_names,
                &self.beta_signed,
                &self.frontier_abs,
                &self.node_abs,
                &self.input_lower,
                &self.input_upper,
                deadline,
            )
        }

        fn certificate(&self) -> ExactRelu2FacetCertificate {
            let support = combined_row_octahedron_with_deadline(
                &self.graph,
                &self.input,
                &self.alpha,
                Some(&self.bounds),
                "pre",
                0,
                1,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("fresh diamond support");
            ExactRelu2Support::new(&support)
                .expect("valid exact diamond support")
                .certify_normal_certificate([-0.5, -0.5, 1.0, 1.0])
                .expect("closed-form diamond facet")
        }
    }

    fn exact(value: f32) -> ReferenceDirectedLower {
        ReferenceDirectedLower::try_new(value, 0.0).expect("finite exact reference value")
    }

    fn diamond_frontier() -> ReferenceLowerFrontier {
        ReferenceLowerFrontier::try_new(
            1,
            2,
            vec![exact(-1.0), exact(-1.0)],
            vec![exact(0.0), exact(0.0)],
            vec![exact(0.0)],
        )
        .expect("valid diamond lower frontier")
    }

    #[test]
    fn disabled_returns_bit_exact_baseline_without_shadow_construction() {
        let shadow_calls = AtomicUsize::new(0);
        let expected = GpuCrownResult {
            lower_bounds: vec![f32::from_bits(0xbead_beef)],
            upper_bounds: vec![f32::from_bits(0x3f12_3456)],
        };
        let run = run_certified_cut_shadow(
            CertifiedCutShadowPolicy::default(),
            || Ok(expected.clone()),
            |_, _| {
                shadow_calls.fetch_add(1, Ordering::SeqCst);
                Err(NyError::SoundnessRefusal(
                    "disabled shadow closure must not execute".into(),
                ))
            },
        )
        .expect("disabled shadow wrapper");
        assert_eq!(shadow_calls.load(Ordering::SeqCst), 0);
        assert_eq!(run.disposition(), ResidentCutShadowDisposition::Disabled);
        assert!(run.observation().is_none());
        assert_eq!(run.into_baseline(), expected);
    }

    #[test]
    fn exact_diamond_reference_shadow_tightens_but_returns_baseline() {
        let fixture = DiamondFixture::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        let context = fixture.context(deadline);
        let certificate = fixture.certificate();
        let carrier = context
            .build_bound_carrier(&[certificate], &[vec![1.0]])
            .expect("valid bound carrier")
            .expect("nonzero carrier");
        let permit = ReferenceShadowPermit::for_test();
        let expected = GpuCrownResult {
            lower_bounds: vec![-3.0],
            upper_bounds: vec![0.0],
        };
        let run = run_certified_cut_shadow(
            CertifiedCutShadowPolicy::Shadow(&permit),
            || Ok(expected.clone()),
            |_, baseline| {
                let mut frontier = diamond_frontier();
                apply_reference_lower_cut(&context, &carrier, &mut frontier)?;

                // λ=1 cancels both post coefficients. The complete cut is then
                // -0.5*x1 -0.5*x2 -1 = -u0 -1 over u∈[-1,1]^2, whose exact
                // minimum is -2. This is the small-network reference oracle.
                assert_eq!(frontier.post, vec![exact(0.0), exact(0.0)]);
                assert_eq!(frontier.pre, vec![exact(-0.5), exact(-0.5)]);
                assert!(
                    (-1.0 - 4.0 * f32::EPSILON..=-1.0).contains(&frontier.bias[0].center),
                    "exact support's outward f32 RHS must move the lower bias only downward"
                );
                let input_coeff = [
                    frontier.pre[0].center + frontier.pre[1].center,
                    frontier.pre[0].center - frontier.pre[1].center,
                ];
                let shadow_lower =
                    frontier.bias[0].center - input_coeff[0].abs() - input_coeff[1].abs();
                assert!(
                    (-2.0 - 4.0 * f32::EPSILON..=-2.0).contains(&shadow_lower),
                    "reference lower {shadow_lower} must enclose the exact network minimum -2"
                );
                ResidentCutShadowObservation::try_new(0, baseline.lower_bounds[0], shadow_lower)
            },
        )
        .expect("reference shadow wrapper");

        assert_eq!(run.baseline(), &expected);
        assert_eq!(run.disposition(), ResidentCutShadowDisposition::Observed);
        let observation = run.observation().expect("completed shadow telemetry");
        assert_eq!(observation.binding_row(), 0);
        assert_eq!(observation.baseline_lower(), -3.0);
        assert!((-2.0 - 4.0 * f32::EPSILON..=-2.0).contains(&observation.shadow_lower()));
        assert!(observation.delta() > 0.999_999 && observation.delta() <= 1.0);
        assert_eq!(run.into_baseline(), expected);
    }

    #[test]
    fn stale_and_cross_request_carriers_are_refused() {
        let first = DiamondFixture::new();
        let second = DiamondFixture::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        let first_context = first.context(deadline);
        let second_context = second.context(deadline);
        let carrier = first_context
            .build_bound_carrier(&[first.certificate()], &[vec![1.0]])
            .expect("first carrier")
            .expect("nonzero first carrier");

        assert!(
            second_context.validate_bound_carrier(&carrier).is_err(),
            "same-shaped independent requests must not exchange authority"
        );
        let original = ResidentCutSnapshotGenerations::fixture();
        let stale_generations = [
            ResidentCutSnapshotGenerations {
                domain: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                bounds: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                alpha: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                beta: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                objective: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                decomposition: 2,
                ..original
            },
            ResidentCutSnapshotGenerations {
                frontier: 2,
                ..original
            },
        ];
        for stale in stale_generations {
            first_context.set_generations_for_test(stale);
            assert!(
                first_context.validate_bound_carrier(&carrier).is_err(),
                "every advanced semantic generation must invalidate the old carrier"
            );
        }
        first_context.set_generations_for_test(original);
        first_context
            .validate_bound_carrier(&carrier)
            .expect("restoring the exact snapshot restores local validity");
    }

    #[test]
    fn real_backend_attempt_advances_generation_and_prevents_replay() {
        struct GenerationBackend {
            calls: AtomicUsize,
        }

        impl GpuCrownBackward for GenerationBackend {
            fn crown_backward_gpu(
                &self,
                _layers: &[GpuCrownLayer],
                _spec: &[f32],
                _num_specs: usize,
                _input_lower: &[f32],
                _input_upper: &[f32],
            ) -> Result<GpuCrownResult> {
                Err(NyError::UnsupportedOp(
                    "generation test uses only the beta-resnet entry".into(),
                ))
            }

            fn crown_backward_gpu_resnet_sound_beta(
                &self,
                _segments: &[GpuResnetSegment],
                _seed: &GpuCrownSeed,
                _input_lower: &[f32],
                _input_upper: &[f32],
                _beta_signed: &[Vec<f32>],
                _frontier_abs: &[Vec<f32>],
                _node_abs: &[Vec<f32>],
            ) -> Result<GpuCrownResult> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(GpuCrownResult {
                    lower_bounds: vec![-3.0],
                    upper_bounds: vec![0.0],
                })
            }
        }

        let fixture = DiamondFixture::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        let context = fixture.context(deadline);
        let carrier = context
            .build_bound_carrier(&[fixture.certificate()], &[vec![1.0]])
            .expect("carrier build")
            .expect("nonzero carrier");
        let backend = GenerationBackend {
            calls: AtomicUsize::new(0),
        };
        let outcome = context
            .run_backend_shadow(&carrier, &backend, 0)
            .expect("default backend returns its baseline on capability miss");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::BackendUnavailable
        );
        assert_eq!(outcome.into_baseline().lower_bounds, vec![-3.0]);
        assert!(
            context.validate_bound_carrier(&carrier).is_err(),
            "one synchronous backend attempt must stale the carrier generation"
        );
    }

    #[test]
    fn production_candidate_binds_live_diamond_and_reaches_backend_once() {
        struct ObservingBackend {
            calls: AtomicUsize,
        }

        impl GpuCrownBackward for ObservingBackend {
            fn crown_backward_gpu(
                &self,
                _layers: &[GpuCrownLayer],
                _spec: &[f32],
                _num_specs: usize,
                _input_lower: &[f32],
                _input_upper: &[f32],
            ) -> Result<GpuCrownResult> {
                Err(NyError::UnsupportedOp(
                    "production candidate test uses only the cut entry".into(),
                ))
            }

            fn crown_backward_gpu_resnet_sound_beta_cut_shadow(
                &self,
                policy: ny_core::ResidentCutShadowPolicy,
                _segments: &[GpuResnetSegment],
                seed: &GpuCrownSeed,
                _input_lower: &[f32],
                _input_upper: &[f32],
                _beta_signed: &[Vec<f32>],
                _frontier_abs: &[Vec<f32>],
                _node_abs: &[Vec<f32>],
                carrier: Option<&ny_core::ResidentLowerCutCarrier>,
                binding_row: usize,
                deadline: Instant,
            ) -> Result<ResidentCutShadowOutcome> {
                assert_eq!(policy, ny_core::ResidentCutShadowPolicy::Shadow);
                assert_eq!(seed.num_specs, 1);
                assert_eq!(binding_row, 0);
                let carrier = carrier.expect("complete arithmetic transport");
                assert_eq!(carrier.target_activation(), 0);
                assert_eq!(carrier.target_width(), 2);
                assert_eq!(carrier.ordered_neurons(), [0, 1]);
                assert_eq!(carrier.rows().len(), 1);
                assert!(carrier.has_nonzero_multiplier());
                assert_eq!(carrier.deadline(), deadline);
                self.calls.fetch_add(1, Ordering::SeqCst);

                let baseline = GpuCrownResult {
                    lower_bounds: vec![-3.0],
                    upper_bounds: vec![0.0],
                };
                let observation = ResidentCutShadowObservation::try_new(0, -3.0, -2.0)?;
                ResidentCutShadowOutcome::try_observed(baseline, observation)
            }
        }

        let fixture = DiamondFixture::new();
        let engine = ny_core::NaiveCpuGemmEngine;
        let backend = ObservingBackend {
            calls: AtomicUsize::new(0),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        let outcome = run_production_resident_cut_shadow(ProductionResidentCutShadowRequest {
            graph: &fixture.graph,
            input: &fixture.input,
            alpha_state: &fixture.alpha,
            node_bounds: &fixture.bounds,
            engine: &engine,
            gpu: &backend,
            seed: &fixture.seed,
            segments: &fixture.segments,
            relu_names: &fixture.relu_names,
            beta_signed: &fixture.beta_signed,
            frontier_abs: &fixture.frontier_abs,
            node_abs: &fixture.node_abs,
            resident_input_lower: &fixture.input_lower,
            resident_input_upper: &fixture.input_upper,
            binding_row: 0,
            deadline,
        })
        .expect("live exact diamond candidate");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Observed
        );
        assert_eq!(outcome.observation().expect("telemetry").delta(), 1.0);
        assert_eq!(outcome.into_baseline().lower_bounds, vec![-3.0]);
    }

    #[test]
    fn projected_m2_dispatches_only_binding_row_snapshots_through_the_cut_backend() {
        struct ProjectedBackend {
            calls: AtomicUsize,
        }

        impl GpuCrownBackward for ProjectedBackend {
            fn crown_backward_gpu(
                &self,
                _layers: &[GpuCrownLayer],
                _spec: &[f32],
                _num_specs: usize,
                _input_lower: &[f32],
                _input_upper: &[f32],
            ) -> Result<GpuCrownResult> {
                Err(NyError::UnsupportedOp(
                    "projected M2 test uses only the cut entry".into(),
                ))
            }

            fn crown_backward_gpu_resnet_sound_beta_cut_shadow(
                &self,
                policy: ny_core::ResidentCutShadowPolicy,
                _segments: &[GpuResnetSegment],
                seed: &GpuCrownSeed,
                _input_lower: &[f32],
                _input_upper: &[f32],
                _beta_signed: &[Vec<f32>],
                _frontier_abs: &[Vec<f32>],
                _node_abs: &[Vec<f32>],
                carrier: Option<&ny_core::ResidentLowerCutCarrier>,
                binding_row: usize,
                deadline: Instant,
            ) -> Result<ResidentCutShadowOutcome> {
                assert_eq!(policy, ny_core::ResidentCutShadowPolicy::Shadow);
                assert_eq!(seed.num_specs, 2);
                assert_eq!(binding_row, 1);
                let carrier = carrier.expect("M2 snapshot has a complete transport");
                assert_eq!(carrier.deadline(), deadline);
                assert_eq!(carrier.rows().len(), 2);
                assert!(
                    carrier.rows()[0]
                        .multipliers()
                        .iter()
                        .all(|value| value.to_bits() == 0.0_f32.to_bits()),
                    "the non-binding objective row must remain exact lambda zero"
                );
                assert!(
                    carrier.rows()[1]
                        .multipliers()
                        .iter()
                        .any(|value| *value > 0.0),
                    "every dispatched M2 snapshot must be nonzero on the binding row"
                );
                assert!(carrier.rows()[1]
                    .multipliers()
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=4.0).contains(value)));

                let lambda_sum = carrier.rows()[1].multipliers().iter().copied().sum::<f32>();
                let delta = 4.0 - (lambda_sum - 2.0) * (lambda_sum - 2.0);
                self.calls.fetch_add(1, Ordering::SeqCst);
                let baseline = GpuCrownResult {
                    lower_bounds: vec![-8.0, -3.0],
                    upper_bounds: vec![0.0, 0.0],
                };
                let observation = ResidentCutShadowObservation::try_new(1, -3.0, -3.0 + delta)?;
                ResidentCutShadowOutcome::try_observed(baseline, observation)
            }
        }

        let fixture = DiamondFixture::new();
        let seed = GpuCrownSeed {
            lower_a: Arc::from([-1.0_f32, -1.0, -1.0, -1.0]),
            upper_a: Arc::from([-1.0_f32, -1.0, -1.0, -1.0]),
            lower_b: Arc::from([0.0_f32, 0.0]),
            upper_b: Arc::from([0.0_f32, 0.0]),
            num_specs: 2,
            current_dim: 2,
        };
        let engine = ny_core::NaiveCpuGemmEngine;
        let backend = ProjectedBackend {
            calls: AtomicUsize::new(0),
        };
        let request = ProductionResidentCutShadowRequest {
            graph: &fixture.graph,
            input: &fixture.input,
            alpha_state: &fixture.alpha,
            node_bounds: &fixture.bounds,
            engine: &engine,
            gpu: &backend,
            seed: &seed,
            segments: &fixture.segments,
            relu_names: &fixture.relu_names,
            beta_signed: &fixture.beta_signed,
            frontier_abs: &fixture.frontier_abs,
            node_abs: &fixture.node_abs,
            resident_input_lower: &fixture.input_lower,
            resident_input_upper: &fixture.input_upper,
            binding_row: 1,
            deadline: Instant::now() + Duration::from_secs(30),
        };
        let outcome =
            super::super::certified_cut_m2_shadow::run_production_resident_cut_m2_projected(
                &request,
            )
            .expect("M2 live diamond search");

        assert!(backend.calls.load(Ordering::SeqCst) >= 2);
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Observed
        );
        assert_eq!(
            outcome.observation().expect("M2 telemetry").binding_row(),
            1
        );
        assert_eq!(outcome.into_baseline().lower_bounds, vec![-8.0, -3.0]);
    }

    #[test]
    fn reference_mutation_is_atomic_on_mid_row_deadline() {
        let fixture = DiamondFixture::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        let context = fixture.context(deadline);
        let carrier = context
            .build_bound_carrier(&[fixture.certificate()], &[vec![1.0]])
            .expect("carrier build")
            .expect("nonzero carrier");
        let mut frontier = diamond_frontier();
        let before = frontier.clone();
        let mut check = |stage| {
            if stage == ReferenceMutationStage::AfterRow(0) {
                Err(NyError::DeadlineExceeded(
                    "deterministic post-row expiry".into(),
                ))
            } else {
                Ok(())
            }
        };
        let error = apply_reference_lower_cut_with(&context, &carrier, &mut frontier, &mut check)
            .expect_err("late scratch result must not publish");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(frontier, before);
    }

    #[test]
    fn reference_add_charges_source_and_resident_rounding_error() {
        let base = exact(1.0);
        let source =
            ny_core::ResidentLowerCutChannel::try_new(2.0_f32.powi(-25), 2.0_f32.powi(-30))
                .expect("finite source channel");
        let sum = reference_add_lower(base, source).expect("directed reference add");
        assert_eq!(sum.center, 1.0, "the tiny add rounds out of the center");
        assert!(
            sum.abs_error > source.source_abs_error(),
            "resident mutation gap must be charged in addition to source error"
        );
        let exact_sum = 1.0_f64 + f64::from(source.value());
        assert!(
            f64::from(sum.center) - f64::from(sum.abs_error) <= exact_sum
                && exact_sum <= f64::from(sum.center) + f64::from(sum.abs_error)
        );
    }

    #[test]
    fn expired_or_malformed_shadow_publishes_no_observation() {
        let permit = ReferenceShadowPermit::for_test();
        let expected = GpuCrownResult {
            lower_bounds: vec![-3.0],
            upper_bounds: vec![0.0],
        };
        let run = run_certified_cut_shadow(
            CertifiedCutShadowPolicy::Shadow(&permit),
            || Ok(expected.clone()),
            |_, _| {
                Err(NyError::DeadlineExceeded(
                    "deterministic shadow expiry".into(),
                ))
            },
        )
        .expect("shadow refusal retains baseline");
        assert_eq!(run.disposition(), ResidentCutShadowDisposition::Rejected);
        assert!(run.observation().is_none());
        assert_eq!(run.into_baseline(), expected);

        let mismatched = run_certified_cut_shadow(
            CertifiedCutShadowPolicy::Shadow(&permit),
            || {
                Ok(GpuCrownResult {
                    lower_bounds: vec![-3.0],
                    upper_bounds: vec![0.0],
                })
            },
            |_, _| ResidentCutShadowObservation::try_new(0, -4.0, -2.0),
        )
        .expect("misbound observation retains baseline");
        assert_eq!(
            mismatched.disposition(),
            ResidentCutShadowDisposition::Rejected
        );
        assert!(mismatched.observation().is_none());
        assert_eq!(mismatched.into_baseline().lower_bounds, vec![-3.0]);
    }

    #[test]
    fn concurrent_contexts_keep_independent_call_local_identity() {
        let barrier = Arc::new(Barrier::new(2));
        let successes = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let barrier = Arc::clone(&barrier);
                let successes = Arc::clone(&successes);
                scope.spawn(move || {
                    let fixture = DiamondFixture::new();
                    let context = fixture.context(Instant::now() + Duration::from_secs(30));
                    let carrier = match context
                        .build_bound_carrier(&[fixture.certificate()], &[vec![0.0, 1.0]])
                    {
                        Ok(_) => panic!("lambda shape mismatch must refuse before publication"),
                        Err(error) => error,
                    };
                    assert!(matches!(carrier, NyError::InvalidSpec(_)));

                    let carrier = context
                        .build_bound_carrier(&[fixture.certificate()], &[vec![1.0]])
                        .expect("independent carrier build")
                        .expect("nonzero independent carrier");
                    barrier.wait();
                    context
                        .validate_bound_carrier(&carrier)
                        .expect("own concurrent context identity");
                    successes.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(successes.load(Ordering::SeqCst), 2);
    }
}
