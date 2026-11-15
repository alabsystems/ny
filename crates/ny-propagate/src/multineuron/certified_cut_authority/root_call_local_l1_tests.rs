// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine source/descriptor oracle for local authority, never a fixed scalar.
use super::super::two_pair_test_support::JointBackend;
use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2, Array2};
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
    carrier: Option<&ResidentLocalL1LowerCutCarrier>,
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
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
                cert_err,
            } => {
                // Exact1x1 channel affine descriptor emitted by raw BN. This
                // oracle refuses nonzero parameter uncertainty; it never treats
                // an uncertain stored matrix as an exact original network.
                if (
                    *kernel_h, *kernel_w, *stride_h, *stride_w, *pad_h, *pad_w, *out_h, *out_w,
                    *in_h, *in_w,
                ) != (1, 1, 1, 1, 0, 0, 1, 1, 1, 1)
                    || !cert_err.is_exact()
                {
                    return Err(fail());
                }
                let affine = GpuCrownLayer::Linear {
                    weight: weight_col.clone(),
                    bias: bias_expanded.clone(),
                    out_features: *out_channels,
                    in_features: *in_channels,
                    cert_err: *cert_err,
                };
                (a, b) = fold(&[affine], a, b, lower, None, end)?;
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
                    for (p, &j) in c.ordered_neurons()[..c.active_len()].iter().enumerate() {
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
                    for (p, &j) in c.ordered_neurons()[..c.active_len()].iter().enumerate() {
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
pub(crate) struct LocalBackend {
    pub(crate) local_calls: AtomicUsize,
    pub(crate) refuse_local: AtomicUsize,
}
impl GemmEngine for LocalBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no ordinary permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for LocalBackend {
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
    fn provides_bounded_local_l1_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentLocalL1LowerCutCarrier,
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
        self.local_calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_local.load(Ordering::SeqCst) != 0 {
            return Err(NyError::DeadlineExceeded("late local refusal".into()));
        }
        let (a, b) = fold(
            layers,
            q.lower_a.iter().map(|&x| rat(x)).collect(),
            rat(q.lower_b[0]),
            true,
            Some(c),
            end,
        )?;
        concrete(&a, b, l, u, true)
    }
}
/// Original affine/ReLU source. Each pair is a genuine diamond, all eight
/// original inputs participate, and the ninth stable coordinate carries sum u.
pub(crate) fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut w = Array2::<f32>::zeros((9, 8));
    for p in 0..4 {
        w[[2 * p, 2 * p]] = 0.5;
        w[[2 * p, 2 * p + 1]] = 0.5;
        w[[2 * p + 1, 2 * p]] = 0.5;
        w[[2 * p + 1, 2 * p + 1]] = -0.5;
        w[[8, 2 * p]] = 1.;
    }
    let mut bias = vec![0.; 9];
    bias[8] = 5.;
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(w, Some(ndarray::Array1::from_vec(bias))).unwrap()),
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
                arr2(&[[-2., -2., -2., -2., -2., -2., -2., -2., 1.]]),
                Some(arr1(&[-0.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 8]).into_dyn(), arr1(&[1.; 8]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
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
    let layers = chain(&full.segments).unwrap();
    let side = |lower| {
        let (a, z) = fold(layers, vec![rat(1.)], rat(0.), lower, None, end).unwrap();
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
) -> RawLocalL1RootCutQuote<'a> {
    quote_raw_local_l1_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1.; 8],
        1.,
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}
#[test]
fn local_original_four_diamonds_strict_gain_unsafe_and_old_family_separation() {
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(30);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -3.75 + shift);
        // A concave original output attains its box minimum at a vertex.
        // Independent exact rational evaluation reads the authored formula,
        // never the returned certificate or backend scalar.
        let mut minimum = rat(100.);
        for bits in 0..256 {
            let z: Vec<Q> = (0..4)
                .flat_map(|p| {
                    let u = rat(if bits & (1 << (2 * p)) == 0 { -1. } else { 1. });
                    let v = rat(if bits & (1 << (2 * p + 1)) == 0 {
                        -1.
                    } else {
                        1.
                    });
                    [(&u + &v) / rat(2.), (&u - &v) / rat(2.)]
                })
                .collect();
            minimum = minimum.min(rat(4.25 + shift) - z.iter().map(|v| v.abs()).sum::<Q>());
        }
        assert_eq!(minimum, rat(0.25 + shift));
        let oldq = quote_raw_two_pair_root_cut(
            &g,
            &x,
            &b,
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
        .unwrap();
        let ordinary =
            evaluate_raw_two_pair_root_cut(oldq, &JointBackend::default(), &mut || true).unwrap();
        assert!(ordinary.bounds().0 <= -1.75 + shift);
        let engine = LocalBackend::default();
        assert!(!engine.provides_bounded_support4());
        let result =
            evaluate_raw_local_l1_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
                .unwrap();
        assert!(result.bounds().0 > 0.24 + shift && result.bounds().0 <= 0.25 + shift);
        assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
        assert_eq!(
            (
                result.support_calls(),
                result.cut_calls(),
                result.sign_count()
            ),
            (0, 1, 128)
        );
        assert!(result.support_bound() >= 4. && result.support_bound() < 4.00001);
        assert!(result.original_local_bound() >= 4. && result.original_local_bound() < 4.00001);
        for i in 0..8 {
            let n = result.ordered_neurons()[i];
            assert_eq!(
                result.pre_lower()[i].to_bits(),
                b["pre"].lower()[ndarray::IxDyn(&[n])].to_bits()
            );
            assert_eq!(
                result.pre_upper()[i].to_bits(),
                b["pre"].upper()[ndarray::IxDyn(&[n])].to_bits()
            );
        }
        assert_eq!(engine.local_calls.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn local_complete_quote_prices_full_only_kernel_channels_and_exact_caps() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let (work, bytes) = (q.work(), q.scratch_bytes());
    assert!(
        q.kernel.scratch_bytes
            + size_of::<RawLocalL1RootCutQuote<'_>>()
            + size_of::<CompletedRawLocalL1RootCut>()
            + LOCAL_UNARY_DIAGNOSTIC_BYTES
            <= LOCAL_EXTRA_BYTES
    );
    assert_eq!(
        work,
        phase_work(q.base.full_quote, 1, 1).unwrap()
            + q.kernel.work
            + LOCAL_FIXED_WORK
            + EXECUTION_OBSERVATION_WORK
    );
    assert!(
        work < phase_work(q.base.prefix_quote, 4, 1).unwrap()
            + q.kernel.work
            + LOCAL_FIXED_WORK
            + EXECUTION_OBSERVATION_WORK
            + phase_work(q.base.full_quote, 1, 1).unwrap()
    );
    for (w, m, ok) in [
        (work, bytes, true),
        (work - 1, bytes, false),
        (work, bytes - 1, false),
    ] {
        assert_eq!(
            quote_raw_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                8,
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1.; 8],
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
    for n in [
        [0, 1, 2, 3, 4, 5, 6, 6],
        [0, 1, 2, 3, 4, 5, 7, 6],
        [0, 1, 2, 3, 4, 5, 6, 9],
    ] {
        assert!(quote_raw_local_l1_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            8,
            n,
            [1.; 8],
            1.,
            limits(),
            end,
            &mut || true
        )
        .is_none());
    }
    let mut bad = [1.; 8];
    bad[7] = 0.1;
    assert!(quote_raw_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        bad,
        1.,
        limits(),
        end,
        &mut || true
    )
    .is_none());
}
#[test]
fn local_each_poll_and_entered_late_refusal_publish_no_result() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut polls = 0;
    assert!(evaluate_raw_local_l1_root_cut(
        quote(&g, &x, &b, old, end),
        &LocalBackend::default(),
        &mut || {
            polls += 1;
            true
        }
    )
    .is_some());
    for stop in 1..=polls {
        let mut p = 0;
        assert!(evaluate_raw_local_l1_root_cut(
            quote(&g, &x, &b, old, end),
            &LocalBackend::default(),
            &mut || {
                p += 1;
                p != stop
            }
        )
        .is_none());
    }
    let engine = LocalBackend::default();
    engine.refuse_local.store(1, Ordering::SeqCst);
    assert!(
        evaluate_raw_local_l1_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
            .is_none()
    );
    assert_eq!(engine.local_calls.load(Ordering::SeqCst), 1);
    let mut expired = quote(&g, &x, &b, old, end);
    expired.base.deadline = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let engine = LocalBackend::default();
    assert!(evaluate_raw_local_l1_root_cut(expired, &engine, &mut || true).is_none());
    assert_eq!(engine.local_calls.load(Ordering::SeqCst), 0);
}
#[test]
fn local_all_eight_source_map_seed_seal_and_generation_are_bound() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let other = quote(&g, &x, &b, old, end);
    let engine = LocalBackend::default();
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
    macro_rules! make {
        ($seed:expr) => {
            ResidentCutCallContext::new(
                ResidentCutSnapshotGenerations::initial(),
                &g,
                &x,
                &alpha,
                &b,
                Some(&engine),
                $seed,
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
    }
    assert!(LocalContext::new(&q, make!(&bad_seed)).is_err());
    let context = LocalContext::new(&q, make!(&seed)).unwrap();
    macro_rules! kernel {
        ($map:expr,$n:expr,$w:expr) => {
            local_box_l1::evaluate(
                &g,
                &x,
                $map,
                "relu",
                $n,
                $w,
                ny_core::CertifiedWeightError::EXACT,
                q.kernel,
                end,
                &mut || true,
            )
            .unwrap()
        };
    }
    let mut fresh = FreshLocalSupport {
        seal: &other.seal,
        arithmetic: kernel!(&b, &q.neurons, &q.weights),
    };
    assert!(context.build(&fresh).is_err());
    fresh.seal = &q.seal;
    let copied = b.clone();
    fresh.arithmetic = kernel!(&copied, &q.neurons, &q.weights);
    assert!(context.build(&fresh).is_err());
    // Only the eighth index/weight differs; the reused pair-shaped snapshot
    // must not authenticate either foreign selection.
    fresh.arithmetic = kernel!(&b, &[0, 1, 2, 3, 4, 5, 6, 8], &q.weights);
    assert!(context.build(&fresh).is_err());
    let mut weights = q.weights;
    weights[7] = 0.5;
    fresh.arithmetic = kernel!(&b, &q.neurons, &weights);
    assert!(context.build(&fresh).is_err());
    fresh.arithmetic = kernel!(&b, &q.neurons, &q.weights);
    let carrier = context.build(&fresh).unwrap();
    let result = context.run(&carrier, &engine).unwrap();
    assert!(context.run(&carrier, &engine).is_err());
    let prior = ny_core::GpuCrownResult {
        lower_bounds: vec![old.0],
        upper_bounds: vec![old.1],
    };
    let merged = context.base.merge_backend_lower(result, &prior).unwrap();
    assert!(merged.lower_bounds[0] > 0.24);
    let carrier = context.build(&fresh).unwrap();
    engine.refuse_local.store(1, Ordering::SeqCst);
    assert!(context.run(&carrier, &engine).is_err());
    assert!(context.run(&carrier, &engine).is_err());
}
#[test]
fn local_source_resolution_refuses_wrong_parent_and_missing_original_box() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(30);
    let old = baseline(&g, &x, &b, end);
    let mut missing = b.clone();
    missing.remove("pre");
    assert!(quote_raw_local_l1_root_cut(
        &g,
        &x,
        &missing,
        None,
        &[1.],
        old,
        "relu",
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1.; 8],
        1.,
        limits(),
        end,
        &mut || true
    )
    .is_none());
    let q = quote(&g, &x, &b, old, end);
    assert!(local_box_l1::evaluate(
        &g,
        &x,
        &b,
        "pre",
        &q.neurons,
        &q.weights,
        ny_core::CertifiedWeightError::EXACT,
        q.kernel,
        end,
        &mut || true
    )
    .is_err());
    let uncertain = ny_core::CertifiedWeightError {
        weight_rel_err: 1. / 1024.,
        bias_abs_err: 0.,
    };
    assert!(local_box_l1::evaluate(
        &g,
        &x,
        &b,
        "relu",
        &q.neurons,
        &q.weights,
        uncertain,
        q.kernel,
        end,
        &mut || true
    )
    .is_err());
    let old_engine = JointBackend::default();
    assert!(evaluate_raw_local_l1_root_cut(q, &old_engine, &mut || true).is_none());
    assert_eq!(old_engine.ordinary.support_calls.load(Ordering::SeqCst), 0);
}
#[test]
#[allow(unsafe_code)]
fn local_original_product_channels_and_non_ieee_entry_fail_closed() {
    let lambda = f32::from_bits(0x3f000001);
    let w = 255f32 / 256.;
    for v in [-w, 2. * w, -f32::from_bits(0x40800001)] {
        let c = transport_channel(
            directed_reduce(std::iter::once((
                ny_core::f32_to_f64_exact(lambda),
                ny_core::f32_to_f64_exact(v),
            )))
            .unwrap(),
        )
        .unwrap();
        let exact = rat(lambda) * rat(v);
        assert!(exact >= rat(c.value()) - rat(c.source_abs_error()));
        assert!(exact <= rat(c.value()) + rat(c.source_abs_error()));
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
            let (g, x, b) = fixture(0.);
            let end = Instant::now() + Duration::from_secs(30);
            let old = baseline(&g, &x, &b, end);
            let q = quote(&g, &x, &b, old, end);
            let engine = LocalBackend::default();
            std::arch::x86_64::_mm_setcsr(saved | (1 << 6) | (1 << 15));
            assert!(evaluate_raw_local_l1_root_cut(q, &engine, &mut || true).is_none());
            assert_eq!(engine.local_calls.load(Ordering::SeqCst), 0);
        }
    })
    .join()
    .unwrap();
}
