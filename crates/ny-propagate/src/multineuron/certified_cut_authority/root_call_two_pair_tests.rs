// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine source/descriptor oracle for joint authority, never a fixed scalar.
use super::super::test_support::ExactToyBackend;
use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2};
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive};
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
    carrier: Option<&ResidentTwoPairLowerCutCarrier>,
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
                    || *in_features > 16
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
                    for (p, &j) in c.ordered_pairs().iter().flatten().enumerate() {
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
                    for (p, &j) in c.ordered_pairs().iter().flatten().enumerate() {
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

fn concrete(a: &[Q], mut b: Q, l: &[f32], u: &[f32], lower: bool) -> Result<f32> {
    if a.len() != l.len()
        || l.len() != u.len()
        || l.iter()
            .zip(u)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return Err(fail());
    }
    for i in 0..a.len() {
        b += &a[i]
            * rat(if (a[i] >= rat(0.)) == lower {
                l[i]
            } else {
                u[i]
            });
    }
    rounded(&b, lower)
}

#[derive(Default)]
pub(crate) struct JointBackend {
    pub(crate) ordinary: ExactToyBackend,
    pub(crate) joint_calls: AtomicUsize,
    pub(crate) refuse_joint: AtomicUsize,
    last_joint_bits: std::sync::atomic::AtomicU32,
    corrupt_support: AtomicUsize,
}
impl GemmEngine for JointBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("joint path must not acquire ordinary GEMM")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for JointBackend {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        panic!("joint path must not acquire ordinary CROWN")
    }
    fn provides_bounded_support4(&self) -> bool {
        true
    }
    fn provides_bounded_two_pair_lower_cut(&self) -> bool {
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
        let mut out = self
            .ordinary
            .crown_backward_gpu_resnet_support4_with_deadline(s, q, l, u, end)?;
        match self.corrupt_support.load(Ordering::SeqCst) {
            1 => {
                out.lower_bounds[2] = 3.;
                out.upper_bounds[2] = 4.;
            } // Empty against original [-1,1]^2.
            2 => out.lower_bounds[0] = f32::NAN,
            3 => out.upper_bounds.pop().map(|_| ()).unwrap_or(()),
            _ => {}
        }
        Ok(out)
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
        self.joint_calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_joint.load(Ordering::SeqCst) != 0 {
            return Err(NyError::DeadlineExceeded("late exact joint refusal".into()));
        }
        let (a, b) = fold(
            layers,
            q.lower_a.iter().map(|&x| rat(x)).collect(),
            rat(q.lower_b[0]),
            true,
            Some(c),
            end,
        )?;
        let out = concrete(&a, b, l, u, true)?;
        guard(end)?;
        self.last_joint_bits.store(out.to_bits(), Ordering::SeqCst);
        Ok(out)
    }
}

pub(crate) fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[
                    [0.5, 0.5, 0., 0.],
                    [0.5, -0.5, 0., 0.],
                    [0., 0., 0.5, 0.5],
                    [0., 0., 0.5, -0.5],
                    [1., 0., 1., 0.],
                ]),
                Some(arr1(&[0., 0., 0., 0., 4.])),
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
            LinearLayer::new(
                arr2(&[[-2., -2., -2., -2., 1.]]),
                Some(arr1(&[-1.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 4]).into_dyn(), arr1(&[1.; 4]).into_dyn()).unwrap();
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
pub(crate) fn baseline(
    g: &GraphNetwork,
    x: &BoundedTensor,
    b: &HashMap<String, BoundedTensor>,
    end: Instant,
) -> (f32, f32) {
    let full = extract_charged_raw_bn_chain(
        g,
        x,
        g.output_name(),
        b,
        b,
        None,
        ExtractionLimits {
            max_nodes: 16,
            max_caller_bytes: 16 << 20,
        },
        end,
        &mut || true,
    )
    .unwrap();
    let l = x.lower().as_slice().unwrap();
    let u = x.upper().as_slice().unwrap();
    let layers = chain(&full.segments).unwrap();
    let side = |lower| {
        let (a, z) = fold(layers, vec![rat(1.)], rat(0.), lower, None, end).unwrap();
        concrete(&a, z, l, u, lower).unwrap()
    };
    (side(true), side(false))
}
fn quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> RawTwoPairRootCutQuote<'a> {
    quote_raw_two_pair_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [[0, 1], [2, 3]],
        [1.; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn joint_original_source_closes_two_diamonds_where_every_single_pair_fails() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    assert!(old.0 <= -1.75 && old.1 >= 2.25);
    // F=9/4-(|z0|+|z1|+|z2|+|z3|), with each independent diamond
    // satisfying |z0|+|z1|<=1. Exact original-network minimum is1/4.
    for i in 0..4 {
        for j in i + 1..4 {
            let q = quote_raw_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                [i, j],
                1.,
                limits(),
                end,
                &mut || true,
            )
            .unwrap();
            let single =
                evaluate_raw_root_cut(q, &ExactToyBackend::default(), &mut || true).unwrap();
            assert!(single.bounds().0 < 0., "pair {i},{j}");
        }
    }
    let backend = JointBackend::default();
    let result =
        evaluate_raw_two_pair_root_cut(quote(&g, &x, &b, old, end), &backend, &mut || true)
            .unwrap();
    assert_eq!(result.bounds().0, 0.25);
    assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
    assert_eq!(result.ordered_pairs(), [[0, 1], [2, 3]]);
    assert_eq!(result.lambda_scales(), [1., 1.]);
    assert_eq!(result.support_calls(), 1);
    assert_eq!(result.cut_calls(), 1);
    assert_eq!(backend.ordinary.support_calls.load(Ordering::SeqCst), 1);
    assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn joint_negative_property_empty_support_and_backend_refusal_publish_no_false_positive() {
    let (g, x, b) = fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let result = evaluate_raw_two_pair_root_cut(
        quote(&g, &x, &b, old, end),
        &JointBackend::default(),
        &mut || true,
    )
    .unwrap();
    assert_eq!(result.bounds().0, -0.25);
    assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
    for corrupt in 1..=3 {
        let backend = JointBackend::default();
        backend.corrupt_support.store(corrupt, Ordering::SeqCst);
        assert!(
            evaluate_raw_two_pair_root_cut(quote(&g, &x, &b, old, end), &backend, &mut || true)
                .is_none()
        );
        assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 0);
    }
    let backend = JointBackend::default();
    backend.refuse_joint.store(1, Ordering::SeqCst);
    assert!(
        evaluate_raw_two_pair_root_cut(quote(&g, &x, &b, old, end), &backend, &mut || true)
            .is_none()
    );
    assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 1);
    assert!(evaluate_raw_two_pair_root_cut(
        quote(&g, &x, &b, old, end),
        &ExactToyBackend::default(),
        &mut || true
    )
    .is_none());
}

#[test]
fn joint_every_poll_and_exact_resource_edges_preserve_atomicity() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let exact = RawRootCutLimits {
        max_work: q.work(),
        max_scratch_bytes: q.scratch_bytes(),
        ..limits()
    };
    assert_eq!(q.candidate_count(), 1);
    let make = |cap| {
        quote_raw_two_pair_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            [[0, 1], [2, 3]],
            [1.; 2],
            cap,
            end,
            &mut || true,
        )
    };
    assert!(make(exact).is_some());
    assert!(make(RawRootCutLimits {
        max_work: exact.max_work - 1,
        ..exact
    })
    .is_none());
    assert!(make(RawRootCutLimits {
        max_scratch_bytes: exact.max_scratch_bytes - 1,
        ..exact
    })
    .is_none());
    let mut polls = 0;
    evaluate_raw_two_pair_root_cut(q, &JointBackend::default(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    for fail_at in 1..=polls {
        let mut n = 0;
        assert!(evaluate_raw_two_pair_root_cut(
            make(exact).unwrap(),
            &JointBackend::default(),
            &mut || {
                n += 1;
                n != fail_at
            }
        )
        .is_none());
        assert_eq!(n, fail_at);
    }
    for pairs in [[[0, 1], [1, 2]], [[0, 1], [2, 99]], [[0, 0], [2, 3]]] {
        assert!(quote_raw_two_pair_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            pairs,
            [1.; 2],
            limits(),
            end,
            &mut || true
        )
        .is_none());
    }
}

fn fresh<'q>(q: &'q RawTwoPairRootCutQuote<'_>) -> FreshJointSupport<'q> {
    // Independently genuine fixture relation bounds, reconstructed by exact
    // descriptor support arithmetic, not a preselected returned lower scalar.
    let b = &q.base;
    let prefix = extract_charged_raw_bn_chain(
        b.graph,
        b.input,
        b.pre_node,
        b.bounds,
        b.bounds,
        b.alpha,
        b.extraction,
        b.deadline,
        &mut || true,
    )
    .unwrap();
    let d = prefix.output_dim;
    let mut a = vec![0.; 4 * d];
    for n in 0..2 {
        let [i, j] = q.pairs[n];
        a[2 * n * d + i] = 1.;
        a[2 * n * d + j] = 1.;
        a[(2 * n + 1) * d + i] = 1.;
        a[(2 * n + 1) * d + j] = -1.;
    }
    let seed = exact_seed(4, d, a);
    let r = ExactToyBackend::default()
        .crown_backward_gpu_resnet_support4_with_deadline(
            &prefix.segments,
            &seed,
            b.input.lower().as_slice().unwrap(),
            b.input.upper().as_slice().unwrap(),
            b.deadline,
        )
        .unwrap();
    let make = |n: usize| {
        let [l1, u1, l2, u2] = source_marginals(q, q.pairs[n]).unwrap();
        let octahedron = Octahedron2 {
            l1: f64::from(l1),
            u1: f64::from(u1),
            l2: f64::from(l2),
            u2: f64::from(u2),
            s_lo: f64::from(r.lower_bounds[2 * n]),
            s_hi: f64::from(r.upper_bounds[2 * n]),
            d_lo: f64::from(r.lower_bounds[2 * n + 1]),
            d_hi: f64::from(r.upper_bounds[2 * n + 1]),
        };
        let certificate = ExactRelu2Support::new(&octahedron)
            .unwrap()
            .certify_normal_certificate([-1., -1., 2., 2.])
            .unwrap();
        FreshSupport {
            octahedron,
            certificate,
        }
    };
    FreshJointSupport {
        quote_seal: &q.seal,
        pairs: q.pairs,
        facets: [make(0), make(1)],
    }
}

#[test]
fn joint_context_rejects_foreign_support_seed_pairs_and_consumed_generation() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let foreign_q = quote(&g, &x, &b, old, end);
    let backend = JointBackend::default();
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
    let wrong_seed = exact_seed(1, 1, vec![-1.]);
    let alpha = GraphAlphaState::new();
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let make = |seed| {
        ResidentCutCallContext::new(
            ResidentCutSnapshotGenerations::initial(),
            &g,
            &x,
            &alpha,
            &b,
            Some(&backend),
            seed,
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
        )
    };
    assert!(JointContext::new(&q, make(&wrong_seed)).is_err());
    let context = JointContext::new(&q, make(&seed)).unwrap();
    let advice = [CutAdvice::uniform(1.); 2];
    assert!(context.build(&fresh(&foreign_q), advice).is_err());
    let mut bad = fresh(&q);
    bad.pairs.swap(0, 1);
    assert!(context.build(&bad, advice).is_err());
    let mut bad = fresh(&q);
    bad.facets[1].octahedron.s_hi = 0.5;
    assert!(context.build(&bad, advice).is_err());
    let f = fresh(&q);
    let carrier = context.build(&f, advice).unwrap();
    let value = context.run(&carrier, &backend).unwrap();
    assert!(context.run(&carrier, &backend).is_err());
    let foreign = JointContext::new(&foreign_q, make(&seed)).unwrap();
    let baseline = ny_core::GpuCrownResult {
        lower_bounds: vec![old.0],
        upper_bounds: vec![old.1],
    };
    assert!(foreign.base.merge_backend_lower(value, &baseline).is_err());
}

#[test]
fn joint_bias_reduction_encloses_original_products_not_rounded_pair_sums() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let q = quote(&g, &x, &b, (-1.75, 4.25), end);
    let f = fresh(&q);
    let advice = [
        CutAdvice::uniform(f32::from_bits(0x3f800001)),
        CutAdvice::uniform(f32::from_bits(0x39800001)),
    ];
    let (_, _, bias) = reduce_two_facets(&f.facets, advice).unwrap();
    let exact = -(rat(advice[0].lambda_scale) * rat(f.facets[0].certificate.facet().b)
        + rat(advice[1].lambda_scale) * rat(f.facets[1].certificate.facet().b));
    let value = rat(bias.value());
    let error = rat(bias.source_abs_error());
    assert!(&value - &error <= exact && exact <= &value + &error);
    assert!(bias.source_abs_error() > 0.);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code)]
fn joint_non_ieee_environment_refuses_before_source_or_channel_work() {
    std::thread::spawn(|| {
        // Thread-local RAII restoration even if an assertion panics; no env var
        // or ambient process-wide floating state is changed.
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: restore the valid MXCSR value saved on this same
                // thread; the aligned u32 stays live for the instruction.
                unsafe {
                    std::arch::asm!("ldmxcsr [{}]",in(reg)&self.0,options(nostack,preserves_flags));
                }
            }
        }
        let (g, x, b) = fixture(0.);
        let end = Instant::now() + Duration::from_secs(30);
        let q = quote(&g, &x, &b, (-1.75, 4.25), end);
        let mut previous = 0u32;
        // SAFETY: STMXCSR writes exactly four bytes into this live u32.
        unsafe {
            std::arch::asm!("stmxcsr [{}]",in(reg)&mut previous,options(nostack,preserves_flags));
        }
        let restore = Restore(previous);
        let bad = previous | (1 << 6) | (1 << 15);
        // SAFETY: only the supported DAZ and FTZ bits of the saved value
        // change; this thread's RAII guard restores it even during unwinding.
        unsafe {
            std::arch::asm!("ldmxcsr [{}]",in(reg)&bad,options(nostack,preserves_flags));
        }
        let backend = JointBackend::default();
        assert!(evaluate_raw_two_pair_root_cut(q, &backend, &mut || true).is_none());
        assert_eq!(backend.ordinary.support_calls.load(Ordering::SeqCst), 0);
        assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 0);
        drop(restore);
    })
    .join()
    .unwrap();
}

// Additional source controls preserve dual alpha throughout an actual private
// support->certificate->joint-fold call. The exact interpreter is not CUDA.
fn alpha_sensitive_fixture(
    shift: f32,
) -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, BoundedTensor>,
    GraphAlphaState,
) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[
                    [0.5, 0.5, 0., 0., 0.],
                    [0.5, -0.5, 0., 0., 0.],
                    [0., 0., 0.5, 0.5, 0.],
                    [0., 0., 0.5, -0.5, 0.],
                    [0., 0., 0., 0., 1.],
                    [1., 0., 1., 0., -3.],
                ]),
                Some(arr1(&[0., 0., 0., 0., 0., 6.])),
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
            LinearLayer::new(
                arr2(&[[-2., -2., -2., -2., 3., 1.]]),
                Some(arr1(&[-3.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 5]).into_dyn(), arr1(&[1.; 5]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha.add_relu_node("relu", &b["pre"], false).unwrap();
    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(0.5);
    upper.fill(0.5);
    lower[4] = 1.;
    upper[4] = 0.;
    (g, x, b, alpha)
}
fn source_linear_value(g: &GraphNetwork, node: &str, x: &[Q]) -> Vec<Q> {
    let Layer::Linear(layer) = &g.node(node).unwrap().layer else {
        panic!()
    };
    layer
        .weight()
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter().zip(x).fold(
                layer.bias().map_or(rat(0.), |b| rat(b[i])),
                |sum, (&w, x)| sum + rat(w) * x,
            )
        })
        .collect()
}
fn alpha_sensitive_original(g: &GraphNetwork, x: &[Q]) -> Q {
    let hidden: Vec<_> = source_linear_value(g, "pre", x)
        .into_iter()
        .map(|z| z.max(rat(0.)))
        .collect();
    source_linear_value(g, "out", &hidden)[0].clone()
}
fn alpha_baseline(
    g: &GraphNetwork,
    x: &BoundedTensor,
    b: &HashMap<String, BoundedTensor>,
    alpha: &GraphAlphaState,
    end: Instant,
) -> (f32, f32) {
    let full = extract_charged_raw_bn_chain(
        g,
        x,
        g.output_name(),
        b,
        b,
        Some(alpha),
        ExtractionLimits {
            max_nodes: 16,
            max_caller_bytes: 16 << 20,
        },
        end,
        &mut || true,
    )
    .unwrap();
    assert!(chain(&full.segments)
        .unwrap()
        .iter()
        .any(|l| matches!(l, GpuCrownLayer::ActivationReluDualAlpha { .. })));
    let side = |lower| {
        let (a, z) = fold(
            chain(&full.segments).unwrap(),
            vec![rat(1.)],
            rat(0.),
            lower,
            None,
            end,
        )
        .unwrap();
        concrete(
            &a,
            z,
            x.lower().as_slice().unwrap(),
            x.upper().as_slice().unwrap(),
            lower,
        )
        .unwrap()
    };
    (side(true), side(false))
}

#[test]
fn joint_dual_alpha_preserves_source_gain_and_unsafe_shift() {
    for shift in [0., -0.5] {
        let (g, x, b, alpha) = alpha_sensitive_fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        // Independent original-parameter oracle: sum of two diamond L1 norms
        // <=2, and 3*(R(w)-w)>=0, hence F>=1/4+shift. The following attained
        // witness plus exact identity over every box vertex/zero cell checks
        // the actual authored graph, not the backend's returned scalar.
        for code in 0usize..243 {
            let mut n = code;
            let values: Vec<_> = (0..5)
                .map(|_| {
                    let v = (n % 3) as f32 - 1.;
                    n /= 3;
                    rat(v)
                })
                .collect();
            let [u, v, s, t, w] = <&[Q; 5]>::try_from(values.as_slice()).unwrap();
            let diamond1 = u.clone().abs().max(v.clone().abs());
            let diamond2 = s.clone().abs().max(t.clone().abs());
            let expected =
                rat(2.25 + shift) - diamond1 - diamond2 + rat(3.) * (w.clone().max(rat(0.)) - w);
            let actual = alpha_sensitive_original(&g, &values);
            assert_eq!(actual, expected);
            assert!(actual >= rat(0.25 + shift));
        }
        assert_eq!(
            alpha_sensitive_original(&g, &[rat(1.), rat(0.), rat(1.), rat(0.), rat(1.)]),
            rat(0.25 + shift)
        );
        let old = alpha_baseline(&g, &x, &b, &alpha, end);
        assert!(old.0 <= -1.75 + shift && old.0 > -1.751 + shift);
        let make = |with_alpha| {
            quote_raw_two_pair_root_cut(
                &g,
                &x,
                &b,
                if with_alpha { Some(&alpha) } else { None },
                &[1.],
                old,
                "relu",
                [[0, 1], [2, 3]],
                [1.; 2],
                limits(),
                end,
                &mut || true,
            )
            .unwrap()
        };
        // None reproduces the historical box-only rewrite on this source.
        // The affine prefix is identical, so only the full-fold alpha differs.
        let rebuilt_backend = JointBackend::default();
        let rebuilt =
            evaluate_raw_two_pair_root_cut(make(false), &rebuilt_backend, &mut || true).unwrap();
        assert_eq!(
            f32::from_bits(rebuilt_backend.last_joint_bits.load(Ordering::SeqCst)),
            -2.75 + shift
        );
        assert_eq!(rebuilt.bounds(), old);
        let backend = JointBackend::default();
        let result = evaluate_raw_two_pair_root_cut(make(true), &backend, &mut || true).unwrap();
        assert_eq!(result.bounds().0, 0.25 + shift);
        assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
        assert_eq!((result.support_calls(), result.cut_calls()), (1, 1));
        assert_eq!(backend.ordinary.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 1);
        assert_eq!(alpha.relu_alpha_pair("relu").unwrap().0[4], 1.);
        assert_eq!(alpha.relu_alpha_pair("relu").unwrap().1[4], 0.);
    }
}

#[test]
fn joint_dual_alpha_keeps_exact_budget_and_atomic_cancellation_guards() {
    let (g, x, b, alpha) = alpha_sensitive_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = alpha_baseline(&g, &x, &b, &alpha, end);
    let make = |caps| {
        quote_raw_two_pair_root_cut(
            &g,
            &x,
            &b,
            Some(&alpha),
            &[1.],
            old,
            "relu",
            [[0, 1], [2, 3]],
            [1.; 2],
            caps,
            end,
            &mut || true,
        )
    };
    let q = make(limits()).unwrap();
    let caps = RawRootCutLimits {
        max_work: q.work(),
        max_scratch_bytes: q.scratch_bytes(),
        ..limits()
    };
    assert!(make(caps).is_some());
    assert!(make(RawRootCutLimits {
        max_work: caps.max_work - 1,
        ..caps
    })
    .is_none());
    assert!(make(RawRootCutLimits {
        max_scratch_bytes: caps.max_scratch_bytes - 1,
        ..caps
    })
    .is_none());
    let mut polls = 0;
    evaluate_raw_two_pair_root_cut(make(caps).unwrap(), &JointBackend::default(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    for failed in 1..=polls {
        let mut count = 0;
        assert!(evaluate_raw_two_pair_root_cut(
            make(caps).unwrap(),
            &JointBackend::default(),
            &mut || {
                count += 1;
                count != failed
            }
        )
        .is_none());
        assert_eq!(count, failed);
    }
    let backend = JointBackend::default();
    backend.refuse_joint.store(1, Ordering::SeqCst);
    assert!(evaluate_raw_two_pair_root_cut(make(caps).unwrap(), &backend, &mut || true).is_none());
    assert_eq!(backend.joint_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn dual_alpha_original_prefix_support_retains_distinct_upper_minorant() {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(arr2(&[[1.], [1.]]), Some(arr1(&[0., 2.]))).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[-1., 1.], [1., 0.], [0., 1.], [2., -1.]]),
                Some(arr1(&[-2., 0., -2., 2.])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.]).into_dyn(), arr1(&[1.]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha.add_relu_node("relu", &b["pre"], false).unwrap();
    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(0.);
    upper.fill(0.);
    upper[0] = 1.;
    let end = Instant::now() + Duration::from_secs(30);
    let support = |alpha: &GraphAlphaState| {
        let full = extract_charged_raw_bn_chain(
            &g,
            &x,
            "out",
            &b,
            &b,
            Some(alpha),
            ExtractionLimits {
                max_nodes: 16,
                max_caller_bytes: 16 << 20,
            },
            end,
            &mut || true,
        )
        .unwrap();
        let seed = exact_seed(
            4,
            4,
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
        );
        ExactToyBackend::default()
            .crown_backward_gpu_resnet_support4_with_deadline(
                &full.segments,
                &seed,
                &[-1.],
                &[1.],
                end,
            )
            .unwrap()
    };
    let dual = support(&alpha);
    assert_eq!(
        dual.upper_bounds[0], 0.,
        "w-R(w) must use alpha_U=1 on its negative upper coefficient"
    );
    alpha.relu_alpha_pair_mut("relu").unwrap().1[0] = 0.;
    let ordinary_lower_only = support(&alpha);
    assert_eq!(ordinary_lower_only.upper_bounds[0], 1.);
    // Direct exact original-network evaluation checks all four complete rows,
    // including the negative-upper path, independently of descriptor routing.
    for w in [-1., 0., 1.] {
        let hidden: Vec<_> = source_linear_value(&g, "pre", &[rat(w)])
            .into_iter()
            .map(|z| z.max(rat(0.)))
            .collect();
        let truth = source_linear_value(&g, "out", &hidden);
        for (i, value) in truth.iter().enumerate() {
            for result in [&dual, &ordinary_lower_only] {
                assert!(
                    rat(result.lower_bounds[i]) <= *value && *value <= rat(result.upper_bounds[i])
                );
            }
        }
    }
}
