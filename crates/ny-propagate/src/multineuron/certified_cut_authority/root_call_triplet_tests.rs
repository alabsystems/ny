// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine source/descriptor oracle for joint authority, never a fixed scalar.
use super::super::two_pair_test_support::{baseline, JointBackend};
use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2};
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive};
use ny_core::{ResidentLowerCutCarrier, ResidentTwoPairLowerCutCarrier};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn rat(x: f32) -> Q {
    Q::from_float(f64::from(x)).unwrap()
}

fn fail() -> NyError {
    NyError::UnsupportedOp("exact toy supports only tiny exact Linear/ReLU chains".into())
}

fn rounded(x: &Q, lower: bool) -> Result<f32> {
    let mut v = x.to_f32().ok_or_else(fail)?;
    for _ in 0..3 {
        if !v.is_finite() {
            return Err(fail());
        }
        if (lower && rat(v) <= *x) || (!lower && rat(v) >= *x) {
            return Ok(v);
        }
        v = if lower {
            ny_tensor::next_down_f32(v)
        } else {
            next_up_f32(v)
        };
    }
    Err(fail())
}

fn guard(end: Instant) -> Result<()> {
    if Instant::now() >= end {
        Err(NyError::DeadlineExceeded("exact toy deadline".into()))
    } else {
        Ok(())
    }
}

fn chain(s: &[GpuResnetSegment]) -> Result<&[GpuCrownLayer]> {
    match s {
        [GpuResnetSegment::Chain(v)] if v.len() <= 8 => Ok(v),
        _ => Err(fail()),
    }
}

fn exact_objective(seed: &GpuCrownSeed, rows: usize) -> Result<()> {
    if seed.num_specs != rows
        || seed.current_dim == 0
        || seed.current_dim > 16
        || seed.lower_a.len() != rows * seed.current_dim
        || seed.upper_a.len() != seed.lower_a.len()
        || seed.lower_b.len() != rows
        || seed.upper_b.len() != rows
        || seed
            .lower_a
            .iter()
            .chain(seed.lower_b.iter())
            .any(|v| !v.is_finite())
        || seed
            .lower_a
            .iter()
            .zip(seed.upper_a.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        || seed
            .lower_b
            .iter()
            .zip(seed.upper_b.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(fail());
    }
    Ok(())
}

fn fold(
    layers: &[GpuCrownLayer],
    mut a: Vec<Q>,
    mut b: Q,
    lower: bool,
    carrier: Option<&ResidentTripletLowerCutCarrier>,
    end: Instant,
) -> Result<(Vec<Q>, Q)> {
    let mut activation = 0;
    let mut applied = false;
    for layer in layers {
        guard(end)?;
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                cert_err,
            } => {
                if !cert_err.is_exact()
                    || a.len() != *out_features
                    || *in_features > 1024
                    || *out_features > 32
                    || weight.len() > 32 * 1024
                    || weight.len() != out_features * in_features
                    || bias.as_ref().is_some_and(|v| v.len() != *out_features)
                    || weight
                        .iter()
                        .chain(bias.iter().flat_map(|v| v.iter()))
                        .any(|x| !x.is_finite())
                {
                    return Err(fail());
                }
                if let Some(bias) = bias {
                    for i in 0..a.len() {
                        b += &a[i] * rat(bias[i]);
                    }
                }
                let mut next = vec![rat(0.); *in_features];
                for i in 0..*out_features {
                    for j in 0..*in_features {
                        if j % 32 == 0 {
                            guard(end)?;
                        }
                        next[j] += &a[i] * rat(weight[i * in_features + j]);
                    }
                }
                a = next;
            }
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                let (lower_slope, upper_slope, third, upper_intercept, num_neurons, dual) =
                    match layer {
                        GpuCrownLayer::Activation {
                            lower_slope,
                            upper_slope,
                            lower_intercept,
                            upper_intercept,
                            num_neurons,
                        } => (
                            lower_slope,
                            upper_slope,
                            lower_intercept,
                            upper_intercept,
                            num_neurons,
                            false,
                        ),
                        GpuCrownLayer::ActivationReluDualAlpha {
                            lower_pos_slope,
                            cross_slope,
                            upper_neg_slope,
                            cross_intercept,
                            num_neurons,
                        } => (
                            lower_pos_slope,
                            cross_slope,
                            upper_neg_slope,
                            cross_intercept,
                            num_neurons,
                            true,
                        ),
                        _ => unreachable!(),
                    };
                if a.len() != *num_neurons
                    || [lower_slope, upper_slope, third, upper_intercept]
                        .iter()
                        .any(|v| v.len() != a.len() || v.iter().any(|x| !x.is_finite()))
                {
                    return Err(fail());
                }
                let hit = carrier.filter(|c| c.target_activation() == activation);
                if let Some(c) = hit {
                    if !lower {
                        return Err(fail());
                    }
                    let r = c;
                    // Refuse unsupported enclosures, rather than ignoring them.
                    if r.pre()
                        .iter()
                        .chain(r.post())
                        .chain(std::iter::once(&r.bias()))
                        .any(|v| v.source_abs_error() != 0.)
                    {
                        return Err(fail());
                    }
                    for (p, &j) in c.ordered_neurons().iter().enumerate() {
                        a[j] += rat(r.post()[p].value());
                    }
                }
                for i in 0..a.len() {
                    let use_lower = (a[i] >= rat(0.)) == lower;
                    let (s, t) = if use_lower {
                        // Dual's third buffer is the upper-path MINORANT,
                        // never an intercept. Interpret the supplied source
                        // descriptor independently of production extraction.
                        (
                            if dual && !lower {
                                third[i]
                            } else {
                                lower_slope[i]
                            },
                            if dual { 0. } else { third[i] },
                        )
                    } else {
                        (upper_slope[i], upper_intercept[i])
                    };
                    b += &a[i] * rat(t);
                    a[i] *= rat(s);
                }
                if let Some(c) = hit {
                    let r = c;
                    for (p, &j) in c.ordered_neurons().iter().enumerate() {
                        a[j] += rat(r.pre()[p].value());
                    }
                    b += rat(r.bias().value());
                    applied = true;
                }
                activation += 1;
            }
            _ => return Err(fail()),
        }
    }
    if carrier.is_some() && !applied {
        return Err(fail());
    }
    guard(end)?;
    Ok((a, b))
}

fn concrete(a: &[Q], mut b: Q, l: &[f32], u: &[f32], lower: bool, end: Instant) -> Result<f32> {
    guard(end)?;
    if a.len() != l.len()
        || l.len() != u.len()
        || l.iter()
            .zip(u)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return Err(fail());
    }
    for i in 0..a.len() {
        if i % 32 == 0 {
            guard(end)?;
        }
        b += &a[i]
            * rat(if (a[i] >= rat(0.)) == lower {
                l[i]
            } else {
                u[i]
            });
    }
    guard(end)?;
    rounded(&b, lower)
}

#[derive(Default)]
pub(crate) struct TripletBackend {
    pub(crate) joint: JointBackend,
    pub(crate) triplet_calls: AtomicUsize,
    pub(crate) refuse_triplet: AtomicUsize,
    corrupt_support: AtomicUsize,
}
impl GemmEngine for TripletBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no ordinary permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for TripletBackend {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        panic!("no ordinary permission")
    }
    fn provides_bounded_support4(&self) -> bool {
        true
    }
    fn provides_bounded_lower_cut(&self) -> bool {
        true
    }
    fn provides_bounded_two_pair_lower_cut(&self) -> bool {
        true
    }
    fn provides_bounded_triplet_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_support4_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        end: Instant,
    ) -> Result<ny_core::GpuCrownResult> {
        let mut result = self
            .joint
            .crown_backward_gpu_resnet_support4_with_deadline(s, q, l, u, end)?;
        match self.corrupt_support.load(Ordering::SeqCst) {
            1 => result.lower_bounds[0] = f32::NAN,
            2 => {
                result.lower_bounds[0] = 2.;
                result.upper_bounds[0] = 1.;
            }
            3 => {
                result.upper_bounds.pop();
            }
            _ => {}
        }
        Ok(result)
    }
    fn crown_backward_gpu_resnet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        self.joint
            .ordinary
            .crown_backward_gpu_resnet_lower_cut_with_deadline(s, q, l, u, c, end)
    }
    fn crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoPairLowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        self.joint
            .crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(s, q, l, u, c, end)
    }
    fn crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTripletLowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        guard(end)?;
        exact_objective(q, 1)?;
        let layers = chain(s)?;
        let widths: Vec<_> = layers
            .iter()
            .filter_map(|x| match x {
                GpuCrownLayer::Activation { num_neurons, .. }
                | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => Some(*num_neurons),
                _ => None,
            })
            .collect();
        c.validate_for_call(
            widths.len(),
            *widths.get(c.target_activation()).ok_or_else(fail)?,
            1,
            end,
        )?;
        self.triplet_calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_triplet.load(Ordering::SeqCst) != 0 {
            return Err(NyError::DeadlineExceeded("late triplet refusal".into()));
        }
        let (a, b) = fold(
            layers,
            q.lower_a.iter().map(|&x| rat(x)).collect(),
            rat(q.lower_b[0]),
            true,
            Some(c),
            end,
        )?;
        concrete(&a, b, l, u, true, end)
    }
}
pub(crate) fn cycle_fixture(
    shift: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.5, -0.5, 0.], [0., 0.5, -0.5], [-0.5, 0., 0.5]]),
                None,
            )
            .unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[-2., -2., -2.]]), Some(arr1(&[2.25 + shift]))).unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 3]).into_dyn(), arr1(&[1.; 3]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
fn limits() -> RawRootCutLimits {
    RawRootCutLimits {
        max_nodes: 16,
        max_scratch_bytes: 32 << 20,
        max_work: 200_000_000,
    }
}
fn quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> RawTripletRootCutQuote<'a> {
    quote_raw_triplet_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [0, 1, 2],
        [1.; 3],
        1.,
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}
#[test]
fn triplet_original_cycle_strictly_separates_all_current_single_pairs_and_unsafe_shift() {
    for shift in [0., -0.5] {
        let (g, x, b) = cycle_fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -0.75 + shift);
        // Exact authored algebra: sum z=0 and sum|z|<=2; enumerate original
        // cube vertices as an independent rational witness/minimum oracle.
        let mut min = rat(100.);
        for a in [-1., 1.] {
            for b in [-1., 1.] {
                for c in [-1., 1.] {
                    let z = [
                        (rat(a) - rat(b)) / rat(2.),
                        (rat(b) - rat(c)) / rat(2.),
                        (rat(c) - rat(a)) / rat(2.),
                    ];
                    let f = rat(2.25 + shift) - z.iter().map(|z| z.abs()).sum::<Q>();
                    min = min.min(f);
                }
            }
        }
        assert_eq!(min, rat(0.25 + shift));
        for pair in [[0, 1], [0, 2], [1, 2]] {
            let q = quote_raw_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                pair,
                1.,
                limits(),
                end,
                &mut || true,
            )
            .unwrap();
            assert_eq!(q.candidate_count(), 2);
            let result =
                evaluate_raw_root_cut(q, &TripletBackend::default().joint.ordinary, &mut || true)
                    .unwrap();
            assert!(result.bounds().0 <= -0.75 + shift);
        }
        let engine = TripletBackend::default();
        let result =
            evaluate_raw_triplet_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
                .unwrap();
        assert_eq!(result.bounds(), (0.25 + shift, old.1));
        assert_eq!(result.support_bound(), 2.);
        assert_eq!(
            engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
            1
        );
        assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn triplet_sign_certificate_handles_asymmetry_points_zero_and_bad_support() {
    let weights = [0.25, 0.5, 1.];
    assert!(valid_weights(weights));
    let rows = signed_rows(weights);
    let z = [-2., 0., 3.];
    let values = rows.map(|r| r.iter().zip(z).map(|(w, z)| w * z).sum::<f32>());
    assert_eq!(sign_bound(values, values), Some(3.5));
    assert_eq!(sign_bound([0.; 4], [0.; 4]), Some(0.));
    for w in [
        [0., 1., 1.],
        [-1., 1., 1.],
        [f32::NAN, 1., 1.],
        [f32::from_bits(1), 1., 1.],
        [0.1, 1., 1.],
    ] {
        assert!(!valid_weights(w));
    }
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    for code in 1..=3 {
        let engine = TripletBackend::default();
        engine.corrupt_support.store(code, Ordering::SeqCst);
        assert!(
            evaluate_raw_triplet_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
                .is_none()
        );
        assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn triplet_each_poll_and_late_fold_refuse_without_publication() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let mut polls = 0;
    assert!(evaluate_raw_triplet_root_cut(
        quote(&g, &x, &b, old, end),
        &TripletBackend::default(),
        &mut || {
            polls += 1;
            true
        }
    )
    .is_some());
    for stop in 1..=polls {
        let mut n = 0;
        assert!(evaluate_raw_triplet_root_cut(
            quote(&g, &x, &b, old, end),
            &TripletBackend::default(),
            &mut || {
                n += 1;
                n != stop
            }
        )
        .is_none());
    }
    let engine = TripletBackend::default();
    engine.refuse_triplet.store(1, Ordering::SeqCst);
    assert!(
        evaluate_raw_triplet_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true).is_none()
    );
    assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.joint.ordinary.cut_calls.load(Ordering::SeqCst), 0);
}
#[test]
fn triplet_exact_quote_caps_indices_and_expired_entry() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let (work, bytes) = (q.work(), q.scratch_bytes());
    for (w, m, ok) in [
        (work, bytes, true),
        (work - 1, bytes, false),
        (work, bytes - 1, false),
    ] {
        assert_eq!(
            quote_raw_triplet_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                [0, 1, 2],
                [1.; 3],
                1.,
                RawRootCutLimits {
                    max_work: w,
                    max_scratch_bytes: m,
                    ..limits()
                },
                end,
                &mut || true
            )
            .is_some(),
            ok
        );
    }
    for n in [[0, 0, 2], [1, 0, 2], [0, 1, 3]] {
        assert!(quote_raw_triplet_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            n,
            [1.; 3],
            1.,
            limits(),
            end,
            &mut || true
        )
        .is_none());
    }
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(quote_raw_triplet_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        [0, 1, 2],
        [1.; 3],
        1.,
        limits(),
        expired,
        &mut || true
    )
    .is_none());
}
#[test]
fn triplet_same_source_certificate_rows_weights_seed_and_consumed_generation() {
    let (g, x, b) = cycle_fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let other = quote(&g, &x, &b, old, end);
    let engine = TripletBackend::default();
    let full = extract_charged_raw_bn_chain(
        &g,
        &x,
        g.output_name(),
        &b,
        &b,
        None,
        q.base.extraction,
        end,
        &mut || true,
    )
    .unwrap();
    let seed = exact_seed(1, 1, vec![1.]);
    let bad_seed = exact_seed(1, 1, vec![2.]);
    let alpha = GraphAlphaState::new();
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    assert!(TripletContext::new(
        &q,
        ResidentCutCallContext::new(
            ResidentCutSnapshotGenerations::initial(),
            &g,
            &x,
            &alpha,
            &b,
            Some(&engine),
            &bad_seed,
            "relu",
            [0, 1],
            &full.segments,
            &full.relu_names,
            &beta,
            &full.frontier_abs,
            &full.node_abs,
            x.lower().as_slice().unwrap(),
            x.upper().as_slice().unwrap(),
            end
        )
    )
    .is_err());
    let context = TripletContext::new(
        &q,
        ResidentCutCallContext::new(
            ResidentCutSnapshotGenerations::initial(),
            &g,
            &x,
            &alpha,
            &b,
            Some(&engine),
            &seed,
            "relu",
            [0, 1],
            &full.segments,
            &full.relu_names,
            &beta,
            &full.frontier_abs,
            &full.node_abs,
            x.lower().as_slice().unwrap(),
            x.upper().as_slice().unwrap(),
            end,
        ),
    )
    .unwrap();
    // Obtain these bounds from genuine descriptor support, then corrupt only
    // private certificate metadata to verify all three positions/signs bind.
    let prefix = extract_charged_raw_bn_chain(
        &g,
        &x,
        q.base.pre_node,
        &b,
        &b,
        None,
        q.base.extraction,
        end,
        &mut || true,
    )
    .unwrap();
    let rows = signed_rows(q.weights);
    let a: Vec<f32> = rows.into_iter().flatten().collect();
    let support = engine
        .crown_backward_gpu_resnet_support4_with_deadline(
            &prefix.segments,
            &exact_seed(4, 3, a),
            x.lower().as_slice().unwrap(),
            x.upper().as_slice().unwrap(),
            end,
        )
        .unwrap();
    let lower: [f32; 4] = support.lower_bounds.try_into().unwrap();
    let upper: [f32; 4] = support.upper_bounds.try_into().unwrap();
    let mut fresh = FreshTripletSupport {
        quote_seal: &q.seal,
        neurons: q.neurons,
        weights: q.weights,
        rows,
        lower,
        upper,
        bound: sign_bound(lower, upper).unwrap(),
    };
    fresh.quote_seal = &other.seal;
    assert!(context.build(&fresh).is_err());
    fresh.quote_seal = &q.seal;
    fresh.neurons = [0, 1, 1];
    assert!(context.build(&fresh).is_err());
    fresh.neurons = q.neurons;
    fresh.weights[2] = 0.5;
    assert!(context.build(&fresh).is_err());
    fresh.weights = q.weights;
    fresh.rows[3][2] = -1.;
    assert!(context.build(&fresh).is_err());
    fresh.rows = rows;
    fresh.bound = 1.;
    assert!(context.build(&fresh).is_err());
    fresh.bound = 2.;
    let carrier = context.build(&fresh).unwrap();
    let out = context.run(&carrier, &engine).unwrap();
    assert!(context.run(&carrier, &engine).is_err());
    let old_result = ny_core::GpuCrownResult {
        lower_bounds: vec![old.0],
        upper_bounds: vec![old.1],
    };
    assert_eq!(
        context
            .base
            .merge_backend_lower(out, &old_result)
            .unwrap()
            .lower_bounds,
        [0.25]
    );
    let carrier = context.build(&fresh).unwrap();
    engine.refuse_triplet.store(1, Ordering::SeqCst);
    assert!(context.run(&carrier, &engine).is_err());
    assert!(context.run(&carrier, &engine).is_err());
}
#[test]
#[allow(unsafe_code)]
fn triplet_reduction_uses_original_products_and_early_ieee_guard() {
    // A non-power-of-two scalar times a Q8 coefficient needs a nonzero
    // transported radius. Check exact rational containment, not CPU closeness.
    ny_core::require_f64_interval_proof_environment().unwrap();
    let lambda = f32::from_bits(0x3f000001);
    let w = 255f32 / 256.;
    for v in [-w, 2. * w, -f32::from_bits(0x3f800001)] {
        let ch = transport_channel(
            directed_reduce(std::iter::once((f64::from(lambda), f64::from(v)))).unwrap(),
        )
        .unwrap();
        let exact = rat(lambda) * rat(v);
        let center = rat(ch.value());
        let radius = rat(ch.source_abs_error());
        assert!(exact >= center.clone() - radius.clone() && exact <= center + radius);
    }
    #[cfg(target_arch = "x86_64")]
    std::thread::spawn(|| {
        #[allow(deprecated)]
        unsafe {
            struct Restore(u32);
            impl Drop for Restore {
                fn drop(&mut self) {
                    #[allow(deprecated)]
                    unsafe {
                        std::arch::x86_64::_mm_setcsr(self.0)
                    }
                }
            }
            let saved = std::arch::x86_64::_mm_getcsr();
            let _restore = Restore(saved);
            let (g, x, b) = cycle_fixture(0.);
            let end = Instant::now() + Duration::from_secs(30);
            let old = baseline(&g, &x, &b, end);
            let q = quote(&g, &x, &b, old, end);
            let engine = TripletBackend::default();
            std::arch::x86_64::_mm_setcsr(saved | (1 << 6) | (1 << 15));
            assert!(evaluate_raw_triplet_root_cut(q, &engine, &mut || true).is_none());
            assert_eq!(
                engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
                0
            );
        }
    })
    .join()
    .unwrap();
}

#[test]
fn triplet_exact_oracle_wide_fold_values_caps_errors_and_deadline() {
    use ny_core::ResidentLowerCutChannel;
    let end = Instant::now() + Duration::from_secs(30);
    let channel = |value, error| ResidentLowerCutChannel::try_new(value, error).unwrap();
    let carrier = |error_slot: Option<usize>| {
        let mut pre = [channel(-1., 0.); 3];
        let mut post = [channel(2., 0.); 3];
        let mut bias = channel(-2.125, 0.);
        match error_slot {
            Some(0) => pre[0] = channel(-1., 0.125),
            Some(1) => post[0] = channel(2., 0.125),
            Some(2) => bias = channel(-2.125, 0.125),
            _ => {}
        }
        ResidentTripletLowerCutCarrier::try_new(0, 3, [0, 1, 2], 1., pre, post, bias, end).unwrap()
    };
    let lo = vec![-1.; 1024];
    let hi = vec![1.; 1024];
    let mut weight = vec![0.; 3 * 1024];
    for j in 0..1024 {
        weight[j] = 1. / 1024.;
        weight[1024 + j] = if j % 2 == 0 { 1. } else { -1. } / 1024.;
        weight[2048 + j] = -1. / 2048.;
    }
    let source = GpuCrownLayer::Linear {
        weight: weight.into(),
        bias: Some(vec![0.25, -0.5, 0.125].into()),
        out_features: 3,
        in_features: 1024,
        cert_err: ny_core::CertifiedWeightError::EXACT,
    };
    let activation = GpuCrownLayer::Activation {
        lower_slope: vec![0.; 3],
        upper_slope: vec![0.625, 0.25, 0.625],
        lower_intercept: vec![0.; 3],
        upper_intercept: vec![0.46875, 0.375, 0.234375],
        num_neurons: 3,
    };
    // A,C range over |A|+|C|<=1. Checking its four vertices proves
    // |A+1/4|+|C-1/2|+|-A/2+1/8| <=17/8, with equality at A=1,C=0.
    let mut max_l1 = rat(0.);
    for (a, c) in [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)] {
        let z = [
            rat(a) + rat(0.25),
            rat(c) - rat(0.5),
            -rat(a) / rat(2.) + rat(0.125),
        ];
        let total = z.iter().fold(rat(0.), |sum, v| sum + v.abs());
        max_l1 = max_l1.max(total);
    }
    assert_eq!(max_l1, rat(2.125));
    let engine = TripletBackend::default();
    for shift in [0., -1.] {
        let output = GpuCrownLayer::Linear {
            weight: vec![-2.; 3].into(),
            bias: Some(vec![3.25 + shift].into()),
            out_features: 1,
            in_features: 3,
            cert_err: ny_core::CertifiedWeightError::EXACT,
        };
        let layers = vec![output, activation.clone(), source.clone()];
        let c = carrier(None);
        let (a, b) = fold(&layers, vec![rat(1.)], rat(0.), true, Some(&c), end).unwrap();
        // Post channels cancel -2, then pre=-1 gives -sum(z). Bias is
        // 3.25+shift-2.125-(-.125)=1.25+shift; parity coefficients below.
        assert_eq!(b, rat(1.25 + shift));
        assert!(a
            .iter()
            .enumerate()
            .all(|(j, v)| *v == rat(if j % 2 == 0 { -3. / 2048. } else { 1. / 2048. })));
        assert_eq!(concrete(&a, b, &lo, &hi, true, end).unwrap(), 0.25 + shift);
        let value = engine
            .crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
                &[GpuResnetSegment::Chain(layers.clone())],
                &exact_seed(1, 1, vec![1.]),
                &lo,
                &hi,
                &c,
                end,
            )
            .unwrap();
        assert_eq!(value, 0.25 + shift);
        if shift < 0. {
            // A=1,C=0 is realized by all even inputs1 and all odd inputs1.
            // This original network has value -.25 there, so no false proof.
            assert_eq!(rat(3.25 + shift) - rat(2.) * rat(1.25), rat(-0.25));
            assert!(value < 0.);
        }
        for slot in 0..3 {
            assert!(matches!(
                fold(
                    &layers,
                    vec![rat(1.)],
                    rat(0.),
                    true,
                    Some(&carrier(Some(slot))),
                    end
                ),
                Err(NyError::UnsupportedOp(_))
            ));
        }
    }
    assert_eq!(engine.triplet_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        engine.joint.ordinary.support_calls.load(Ordering::SeqCst),
        0
    );
    assert!(matches!(
        exact_objective(&exact_seed(1, 17, vec![0.; 17]), 1),
        Err(NyError::UnsupportedOp(_))
    ));

    // Exercise the inclusive matrix limit with nonzero source rows and a
    // hand-computed exact fold: 32 identical mean rows plus 1/8 biases.
    let affine = |inputs: usize, outputs: usize, error| GpuCrownLayer::Linear {
        weight: vec![1. / 1024.; inputs * outputs].into(),
        bias: Some(vec![0.125; outputs].into()),
        out_features: outputs,
        in_features: inputs,
        cert_err: error,
    };
    let (a, b) = fold(
        &[affine(1024, 32, ny_core::CertifiedWeightError::EXACT)],
        vec![rat(1.); 32],
        rat(0.),
        true,
        None,
        end,
    )
    .unwrap();
    assert!(a.iter().all(|v| *v == rat(1. / 32.)));
    assert_eq!(b, rat(4.));
    assert_eq!(concrete(&a, b.clone(), &lo, &hi, true, end).unwrap(), -28.);
    assert_eq!(concrete(&a, b, &lo, &hi, false, end).unwrap(), 36.);
    for (inputs, outputs) in [(1025, 1), (1, 33)] {
        assert!(matches!(
            fold(
                &[affine(
                    inputs,
                    outputs,
                    ny_core::CertifiedWeightError::EXACT
                )],
                vec![rat(1.); outputs],
                rat(0.),
                true,
                None,
                end,
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    for error in [
        ny_core::CertifiedWeightError {
            weight_rel_err: 0.125,
            bias_abs_err: 0.,
        },
        ny_core::CertifiedWeightError {
            weight_rel_err: 0.,
            bias_abs_err: 0.125,
        },
    ] {
        assert!(matches!(
            fold(
                &[affine(1024, 1, error)],
                vec![rat(1.)],
                rat(0.),
                true,
                None,
                end,
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        fold(
            &[affine(1024, 1, ny_core::CertifiedWeightError::EXACT)],
            vec![rat(1.)],
            rat(0.),
            true,
            None,
            expired,
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        concrete(&a, rat(0.), &lo, &hi, true, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        concrete(&a, rat(0.), &lo[..1023], &hi, true, end),
        Err(NyError::UnsupportedOp(_))
    ));
}
