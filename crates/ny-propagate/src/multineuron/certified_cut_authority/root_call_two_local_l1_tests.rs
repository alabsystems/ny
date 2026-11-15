// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine source/descriptor oracle for joint local authority; no canned scalar.
use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, Array2};
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive};
use ny_core::{ResidentLocalL1LowerCutCarrier, ResidentTwoLocalL1LowerCutCarrier};

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

#[path = "root_call_two_local_l1/small_head_advice_tests.rs"]
mod small_head_advice_tests;
pub(crate) use small_head_advice_tests::fixture as small_advice_fixture;

#[path = "root_call_two_local_l1/low_arity_tests.rs"]
mod low_arity_tests;
pub(crate) use low_arity_tests::fixture as low_arity_fixture;

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

#[derive(Clone, Copy)]
enum Cut<'a> {
    Single(&'a ResidentLocalL1LowerCutCarrier),
    Bundle(&'a ResidentTwoLocalL1LowerCutCarrier),
}
impl Cut<'_> {
    fn target_activation(self) -> usize {
        match self {
            Self::Single(c) => c.target_activation(),
            Self::Bundle(c) => c.target_activation(),
        }
    }
    fn bias(self) -> ResidentLowerCutChannel {
        match self {
            Self::Single(c) => c.bias(),
            Self::Bundle(c) => c.bias(),
        }
    }
    fn channels(self, post: bool) -> Vec<(usize, ResidentLowerCutChannel)> {
        match self {
            Self::Single(c) => (0..c.active_len())
                .map(|i| {
                    (
                        c.ordered_neurons()[i],
                        if post { c.post()[i] } else { c.pre()[i] },
                    )
                })
                .collect(),
            Self::Bundle(c) => (0..2)
                .flat_map(|g| {
                    (0..c.active_lens()[g]).map(move |i| {
                        (
                            c.ordered_neurons()[g][i],
                            if post { c.post()[g][i] } else { c.pre()[g][i] },
                        )
                    })
                })
                .collect(),
        }
    }
    fn validate(self, count: usize, width: usize, end: Instant) -> Result<()> {
        match self {
            Self::Single(c) => c.validate_for_call(count, width, 1, end),
            Self::Bundle(c) => c.validate_for_call(count, width, 1, end),
        }
    }
}

fn fold(
    layers: &[GpuCrownLayer],
    mut a: Vec<Q>,
    mut b: Q,
    lower: bool,
    carrier: Option<Cut<'_>>,
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
                // Test-only bounded descriptor interpreter: at most 32,768
                // matrix cells and 1,024 exact rational frontier entries.
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
                    // Coefficient uncertainty remains unsupported. Constant bias
                    // uncertainty is subtracted exactly once below for this lower fold.
                    if c.channels(false)
                        .into_iter()
                        .chain(c.channels(true))
                        .map(|(_, v)| v)
                        .any(|v| v.source_abs_error() != 0.)
                    {
                        return Err(fail());
                    }
                    for (j, v) in c.channels(true) {
                        a[j] += rat(v.value());
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
                    for (j, v) in c.channels(false) {
                        a[j] += rat(v.value());
                    }
                    b += rat(c.bias().value()) - rat(c.bias().source_abs_error());
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

/// Bounded independent rational interpreter shared with spatial-prefix tests.
/// It consumes complete copied source descriptors and exact objective rows;
/// no caller-supplied expected scalar is accepted.
pub(crate) fn exact_plain_rows(
    s: &[GpuResnetSegment],
    q: &GpuCrownSeed,
    l: &[f32],
    u: &[f32],
    end: Instant,
) -> Result<ny_core::GpuCrownResult> {
    guard(end)?;
    if !(1..=8).contains(&q.num_specs) {
        return Err(fail());
    }
    exact_objective(q, q.num_specs)?;
    let layers = chain(s)?;
    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();
    for i in 0..q.num_specs {
        for lower in [true, false] {
            let a = q.lower_a[i * q.current_dim..(i + 1) * q.current_dim]
                .iter()
                .map(|v| rat(*v))
                .collect();
            let (a, b) = fold(layers, a, rat(q.lower_b[i]), lower, None, end)?;
            let value = concrete(&a, b, l, u, lower)?;
            if lower {
                lower_bounds.push(value);
            } else {
                upper_bounds.push(value);
            }
        }
    }
    Ok(ny_core::GpuCrownResult {
        lower_bounds,
        upper_bounds,
    })
}

fn run(
    s: &[GpuResnetSegment],
    q: &GpuCrownSeed,
    l: &[f32],
    u: &[f32],
    c: Cut<'_>,
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
    c.validate(
        widths.len(),
        *widths.get(c.target_activation()).ok_or_else(fail)?,
        end,
    )?;
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
#[derive(Default)]
pub(crate) struct BundleBackend {
    pub(crate) bundle_calls: AtomicUsize,
    pub(crate) refuse_bundle: AtomicUsize,
    returned_lower_bits: AtomicU32,
}
impl GemmEngine for BundleBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no ordinary permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for BundleBackend {
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
    fn provides_bounded_two_local_l1_lower_cut(&self) -> bool {
        true
    }
    fn crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoLocalL1LowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        guard(end)?;
        self.bundle_calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_bundle.load(Ordering::SeqCst) != 0 {
            return Err(NyError::DeadlineExceeded("late bundle refusal".into()));
        }
        let lower = run(s, q, l, u, Cut::Bundle(c), end)?;
        self.returned_lower_bits
            .store(lower.to_bits(), Ordering::SeqCst);
        Ok(lower)
    }
}
#[derive(Default)]
pub(crate) struct SingleBackend {
    pub(crate) local_calls: AtomicUsize,
}
impl GemmEngine for SingleBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no ordinary permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for SingleBackend {
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
        self.local_calls.fetch_add(1, Ordering::SeqCst);
        run(s, q, l, u, Cut::Single(c), end)
    }
}
/// Eight original independent diamonds and one stable sum carrier; all16 inputs participate.
pub(crate) fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut w = Array2::<f32>::zeros((17, 16));
    for k in 0..8 {
        w[[2 * k, 2 * k]] = 0.5;
        w[[2 * k, 2 * k + 1]] = 0.5;
        w[[2 * k + 1, 2 * k]] = 0.5;
        w[[2 * k + 1, 2 * k + 1]] = -0.5;
        w[[16, 2 * k]] = 1.;
    }
    let mut bias = vec![0.; 17];
    bias[16] = 17.;
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
    let mut final_w = vec![-2.; 17];
    final_w[16] = 1.;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 17), final_w).unwrap(),
                Some(arr1(&[-8.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 16]).into_dyn(), arr1(&[1.; 16]).into_dyn()).unwrap();
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
const IDS: [[usize; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]];
fn quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> RawTwoLocalL1RootCutQuote<'a> {
    quote_raw_two_local_l1_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [8, 8],
        IDS,
        [[1.; 8]; 2],
        [1.; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn two_local_exact_oracle_charges_aggregate_bias_radius_once() {
    let end = Instant::now() + Duration::from_secs(30);
    // Unequal outward group bounds make the exact two-product sum fall between
    // f32 values. This is the same constant reduction used by the real builder.
    let bounds = [4.0_f32, next_up_f32(2.0)];
    let bias = transport_channel(
        directed_reduce(bounds.map(|b| (1.0_f64, ny_core::f32_to_f64_exact(-b)))).unwrap(),
    )
    .unwrap();
    let exact_bias = -rat(bounds[0]) - rat(bounds[1]);
    let radius = rat(bias.source_abs_error());
    assert!(radius > rat(0.));
    assert!(rat(bias.value()) > exact_bias);
    assert!(rat(bias.value()) - &radius <= exact_bias);
    assert!(exact_bias <= rat(bias.value()) + &radius);

    let zero = ResidentLowerCutChannel::try_new(0., 0.).unwrap();
    let make = |pre, post| {
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            6,
            [3, 3],
            [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]],
            [1., 1.],
            pre,
            post,
            bias,
            end,
        )
        .unwrap()
    };
    let layers = [GpuCrownLayer::Activation {
        lower_slope: vec![0.; 6],
        upper_slope: vec![0.5; 6],
        lower_intercept: vec![0.; 6],
        upper_intercept: vec![0.5; 6],
        num_neurons: 6,
    }];
    // The exact represented constant is negative, while using only its rounded
    // center would be positive. This tests interval arithmetic, not proof custody.
    let half_gap = (rat(bounds[1]) - rat(2.)) / rat(2.);
    let incoming = -&exact_bias - &half_gap;
    let exact_constant = &incoming + &exact_bias;
    let center_only = &incoming + rat(bias.value());
    assert!(exact_constant < rat(0.) && center_only > rat(0.));
    let c = make([[zero; 8]; 2], [[zero; 8]; 2]);
    c.validate_for_call(1, 6, 1, end).unwrap();
    let (a, lower) = fold(
        &layers,
        vec![rat(0.); 6],
        incoming.clone(),
        true,
        Some(Cut::Bundle(&c)),
        end,
    )
    .unwrap();
    assert!(a.iter().all(|v| *v == rat(0.)));
    assert_eq!(lower, &center_only - &radius);
    assert_eq!(&lower + &radius, center_only); // exactly one radius charge
    assert!(lower <= exact_constant);
    assert!(concrete(&a, lower, &[-1.; 6], &[1.; 6], true).unwrap() < 0.);

    for post in [false, true] {
        let mut pre_channels = [[zero; 8]; 2];
        let mut post_channels = pre_channels;
        let uncertain = ResidentLowerCutChannel::try_new(0., 0.125).unwrap();
        if post {
            post_channels[0][0] = uncertain;
        } else {
            pre_channels[0][0] = uncertain;
        }
        let c = make(pre_channels, post_channels);
        assert!(matches!(
            fold(
                &layers,
                vec![rat(0.); 6],
                incoming.clone(),
                true,
                Some(Cut::Bundle(&c)),
                end
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    // Keep the expanded test oracle bounded independently of production limits.
    let affine = |inputs, outputs| GpuCrownLayer::Linear {
        weight: vec![0.; inputs * outputs].into(),
        bias: None,
        out_features: outputs,
        in_features: inputs,
        cert_err: ny_core::CertifiedWeightError::EXACT,
    };
    for (inputs, outputs, admitted) in [(1024, 32, true), (1025, 1, false), (1, 33, false)] {
        let result = fold(
            &[affine(inputs, outputs)],
            vec![rat(0.); outputs],
            rat(0.),
            true,
            None,
            end,
        );
        if admitted {
            let (a, b) = result.unwrap();
            assert_eq!(a.len(), inputs);
            assert!(a.iter().all(|v| *v == rat(0.)) && b == rat(0.));
        } else {
            assert!(matches!(result, Err(NyError::UnsupportedOp(_))));
        }
    }
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        fold(
            &[affine(1024, 1)],
            vec![rat(0.)],
            rat(0.),
            true,
            None,
            expired
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn two_local_original_sixteen_strict_single_group_separation_and_unsafe() {
    // Independent exact source proof: each of eight disjoint pairs has max1,
    // attained by u=1,v=0; convexity bounds the box from its four corners.
    for (u, v) in [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)] {
        let z0 = (rat(u) + rat(v)) / rat(2.);
        let z1 = (rat(u) - rat(v)) / rat(2.);
        assert_eq!(z0.abs() + z1.abs(), rat(1.));
    }
    let mut max_pairs = 0;
    for mask in 0u32..(1 << 16) {
        let m = mask.count_ones();
        if !(3..=8).contains(&m) {
            continue;
        }
        let p = (0..8).filter(|k| (mask >> (2 * k)) & 3 == 3).count();
        assert!(p <= 4);
        max_pairs = max_pairs.max(p);
        // Unit-weight selected support m-p, unselected triangle cost16-m.
        assert!(rat(8.25) - rat((m as usize - p + 16 - m as usize) as f32) <= rat(-3.75));
    }
    assert_eq!(max_pairs, 4);
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(120);
        let old = baseline(&g, &x, &b, end);
        // Collected intervals and extracted triangles are outward rounded;
        // their real fold need not equal the ideal exact-box triangle by bits.
        let ideal_triangle = -7.75 + shift;
        assert!(old.0 <= ideal_triangle && old.0 > ideal_triangle - 0.0001);
        let single = SingleBackend::default();
        for lambda in [0.5, 1.] {
            let q = quote_raw_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                8,
                IDS[0],
                [1.; 8],
                lambda,
                limits(),
                end,
                &mut || true,
            )
            .unwrap();
            let result = evaluate_raw_local_l1_root_cut(q, &single, &mut || true).unwrap();
            assert!(result.bounds().0 <= -3.75 + shift);
        }
        assert_eq!(single.local_calls.load(Ordering::SeqCst), 2);
        let engine = BundleBackend::default();
        assert!(
            !engine.provides_bounded_support4() && !engine.provides_bounded_local_l1_lower_cut()
        );
        let result =
            evaluate_raw_two_local_l1_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
                .unwrap();
        assert!(result.bounds().0 > 0.24 + shift && result.bounds().0 <= 0.25 + shift);
        assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
        assert_eq!(
            (
                result.certificates(),
                result.support_calls(),
                result.cut_calls()
            ),
            (2, 0, 1)
        );
        assert_eq!(result.active_lens(), [8, 8]);
        assert_eq!(result.ordered_neurons(), IDS);
        assert_eq!(result.multipliers(), [1., 1.]);
        assert_eq!(result.sign_counts(), [128, 128]);
        for g in 0..2 {
            assert!(result.support_bounds()[g] >= 4. && result.support_bounds()[g] < 4.00001);
            assert!(
                result.original_local_bounds()[g] >= 4.
                    && result.original_local_bounds()[g] < 4.00001
            );
            assert_eq!(result.bn_radii()[g], 0.);
            for i in 0..8 {
                let n = result.ordered_neurons()[g][i];
                assert_eq!(
                    result.pre_lower()[g][i].to_bits(),
                    b["pre"].lower()[ndarray::IxDyn(&[n])].to_bits()
                );
                assert_eq!(
                    result.pre_upper()[g][i].to_bits(),
                    b["pre"].upper()[ndarray::IxDyn(&[n])].to_bits()
                );
            }
        }
        assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn two_local_quote_full_only_prices_both_kernels_and_all_group_boundaries() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let (work, bytes) = (q.work(), q.scratch_bytes());
    assert_eq!(
        work,
        phase_work(q.base.full_quote, 1, 1).unwrap()
            + q.kernel[0].work
            + q.kernel[1].work
            + LOCAL_FIXED_WORK
            + EXECUTION_OBSERVATION_WORK
    );
    assert!(work < work + phase_work(q.base.prefix_quote, 4, 1).unwrap());
    assert!(
        q.kernel[0].scratch_bytes
            + q.kernel[1].scratch_bytes
            + 2 * size_of::<LocalBoxL1Arithmetic<'_>>()
            + size_of::<RawTwoLocalL1RootCutQuote<'_>>()
            + size_of::<FreshTwoLocalSupport<'_, '_>>()
            + size_of::<TwoLocalContext<'_, '_>>()
            + size_of::<TwoLocalCarrier<'_>>()
            + size_of::<ResidentTwoLocalL1LowerCutCarrier>()
            + size_of::<CompletedRawTwoLocalL1RootCut>()
            + LOCAL_DIAGNOSTIC_BYTES
            <= LOCAL_EXTRA_BYTES
    );
    for (w, m, ok) in [
        (work, bytes, true),
        (work - 1, bytes, false),
        (work, bytes - 1, false),
    ] {
        assert_eq!(
            quote_raw_two_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                [8, 8],
                IDS,
                [[1.; 8]; 2],
                [1.; 2],
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
    for a in 3..=8 {
        for c in 3..=8 {
            let mut ids = [[0; 8]; 2];
            let mut w = [[0.; 8]; 2];
            for i in 0..a {
                ids[0][i] = i;
                w[0][i] = 1.;
            }
            for i in 0..c {
                ids[1][i] = 8 + i;
                w[1][i] = 1.;
            }
            assert!(quote_raw_two_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                [a, c],
                ids,
                w,
                [1.; 2],
                limits(),
                end,
                &mut || true
            )
            .is_some());
            let mut alias = ids;
            alias[1][0] = 0;
            assert!(!valid_selection([a, c], alias, w, 17));
        }
    }
    for active in [[2, 8], [8, 2], [9, 8], [8, 9]] {
        assert!(!valid_selection(active, IDS, [[1.; 8]; 2], 17));
    }
    let mut bad = IDS;
    bad[1][7] = 17;
    assert!(!valid_selection([8, 8], bad, [[1.; 8]; 2], 17));
    let mut bad = [[1.; 8]; 2];
    bad[1][7] = 0.1;
    assert!(!valid_selection([8, 8], IDS, bad, 17));
    let mut padded = IDS;
    padded[1] = [8, 9, 10, 0, 0, 0, 0, 0];
    let mut w = [[1.; 8]; 2];
    w[1] = [1., 1., 1., 0., 0., 0., 0., 0.];
    assert!(valid_selection([8, 3], padded, w, 17)); // neuron0 does not alias inactive0
    w[1][7] = -0.;
    assert!(!valid_selection([8, 3], padded, w, 17));
}

#[test]
fn two_local_every_poll_second_kernel_refusal_and_late_fold_publish_nothing() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(120);
    let old = baseline(&g, &x, &b, end);
    let mut polls = 0;
    assert!(evaluate_raw_two_local_l1_root_cut(
        quote(&g, &x, &b, old, end),
        &BundleBackend::default(),
        &mut || {
            polls += 1;
            true
        }
    )
    .is_some());
    let mut second_refusal = false;
    for stop in 1..=polls {
        let q = quote(&g, &x, &b, old, end);
        let engine = BundleBackend::default();
        let mut p = 0;
        let mut callback = || {
            p += 1;
            p != stop
        };
        let mut guard = Poll {
            end,
            callback: &mut callback,
            alive: true,
        };
        let mut obs = ExecutionObservation::default();
        let mut certificates = 0;
        assert!(evaluate_steps(&q, &engine, &mut guard, &mut obs, &mut certificates).is_none());
        if certificates < 2 {
            assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
        }
        second_refusal |= certificates == 1 && obs.phase == "local-certificate-1";
    }
    assert!(second_refusal);
    let engine = BundleBackend::default();
    engine.refuse_bundle.store(1, Ordering::SeqCst);
    assert!(
        evaluate_raw_two_local_l1_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
            .is_none()
    );
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
    let mut q = quote(&g, &x, &b, old, end);
    q.kernel[1].work -= 1;
    let engine = BundleBackend::default();
    let mut callback = || true;
    let mut guard = Poll {
        end,
        callback: &mut callback,
        alive: true,
    };
    let mut obs = ExecutionObservation::default();
    let mut certificates = 0;
    assert!(evaluate_steps(&q, &engine, &mut guard, &mut obs, &mut certificates).is_none());
    assert_eq!(
        (certificates, obs.phase, obs.error_class),
        (1, "local-certificate-1", "budget")
    );
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
    let mut q = quote(&g, &x, &b, old, end);
    q.base.deadline = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).is_none());
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn two_local_both_group_slots_all_indices_scales_map_seed_and_generation_are_bound() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let q = quote(&g, &x, &b, old, end);
    let other = quote(&g, &x, &b, old, end);
    let engine = BundleBackend::default();
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
        ($seed:expr,$map:expr) => {
            ResidentCutCallContext::new(
                ResidentCutSnapshotGenerations::initial(),
                &g,
                &x,
                &alpha,
                $map,
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
    let copied = b.clone();
    assert!(TwoLocalContext::new(&q, make!(&bad_seed, &b)).is_err());
    assert!(TwoLocalContext::new(&q, make!(&seed, &copied)).is_err());
    let context = TwoLocalContext::new(&q, make!(&seed, &b)).unwrap();
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
                q.kernel[0],
                end,
                &mut || true,
            )
            .unwrap()
        };
    }
    let mut fresh = FreshTwoLocalSupport {
        seal: &other.seal,
        arithmetic: [
            kernel!(&b, &q.neurons[0], &q.weights[0]),
            kernel!(&b, &q.neurons[1], &q.weights[1]),
        ],
    };
    assert!(context.build(&fresh).is_err());
    fresh.seal = &q.seal;
    fresh.arithmetic.swap(0, 1);
    assert!(context.build(&fresh).is_err());
    fresh.arithmetic.swap(0, 1);
    fresh.arithmetic[1] = kernel!(&copied, &q.neurons[1], &q.weights[1]);
    assert!(context.build(&fresh).is_err());
    let mut ids = q.neurons[1];
    ids[7] = 16;
    fresh.arithmetic[1] = kernel!(&b, &ids, &q.weights[1]);
    assert!(context.build(&fresh).is_err());
    let mut weights = q.weights[1];
    weights[7] = 0.5;
    fresh.arithmetic[1] = kernel!(&b, &q.neurons[1], &weights);
    assert!(context.build(&fresh).is_err());
    fresh.arithmetic[1] = kernel!(&b, &q.neurons[1][..7], &q.weights[1][..7]);
    assert!(context.build(&fresh).is_err());
    fresh.arithmetic[1] = kernel!(&b, &q.neurons[1], &q.weights[1]);
    let mut foreign = context.build(&fresh).unwrap();
    let t = &foreign.transport;
    foreign.transport = ResidentTwoLocalL1LowerCutCarrier::try_new(
        t.target_activation(),
        t.target_width(),
        *t.active_lens(),
        *t.ordered_neurons(),
        [1., 0.5],
        *t.pre(),
        *t.post(),
        t.bias(),
        t.deadline(),
    )
    .unwrap();
    assert!(context.run(&foreign, &engine).is_err());
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
    let carrier = context.build(&fresh).unwrap();
    let result = context.run(&carrier, &engine).unwrap();
    assert!(context.run(&carrier, &engine).is_err());
    let merged = context
        .base
        .merge_backend_lower(
            result,
            &ny_core::GpuCrownResult {
                lower_bounds: vec![old.0],
                upper_bounds: vec![old.1],
            },
        )
        .unwrap();
    assert!(merged.lower_bounds[0] > 0.24);
    assert_eq!(merged.upper_bounds[0].to_bits(), old.1.to_bits());
    let carrier = context.build(&fresh).unwrap();
    engine.refuse_bundle.store(1, Ordering::SeqCst);
    assert!(context.run(&carrier, &engine).is_err());
    assert!(context.run(&carrier, &engine).is_err());
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 2);
}

#[test]
#[allow(unsafe_code)]
fn two_local_distinct_original_products_bias_radius_and_early_ieee_guard() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let multipliers = [f32::from_bits(0x3f800001), f32::from_bits(0x3eaaaaab)];
    let mut weights = [[1.; 8]; 2];
    weights[1][7] = 0.5;
    let q = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        [8, 8],
        IDS,
        weights,
        multipliers,
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
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
    let alpha = GraphAlphaState::new();
    let engine = BundleBackend::default();
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let base = ResidentCutCallContext::new(
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
    );
    let context = TwoLocalContext::new(&q, base).unwrap();
    let mut callback = || true;
    let mut guard = Poll {
        end,
        callback: &mut callback,
        alive: true,
    };
    let mut obs = ExecutionObservation::default();
    let fresh = FreshTwoLocalSupport {
        seal: &q.seal,
        arithmetic: [
            evaluate_group(&q, 0, &mut guard, &mut obs).unwrap(),
            evaluate_group(&q, 1, &mut guard, &mut obs).unwrap(),
        ],
    };
    let c = context.build(&fresh).unwrap();
    assert!(same_multipliers(*c.transport.multipliers(), multipliers));
    let exact = -rat(multipliers[0]) * rat(fresh.arithmetic[0].bound())
        - rat(multipliers[1]) * rat(fresh.arithmetic[1].bound());
    let bias = c.transport.bias();
    assert!(
        rat(bias.value()) - rat(bias.source_abs_error()) <= exact
            && exact <= rat(bias.value()) + rat(bias.source_abs_error())
    );
    assert!(bias.source_abs_error() > 0.);
    for g in 0..2 {
        for i in 0..8 {
            for (post, factor) in [(false, -1.), (true, 2.)] {
                let channel = if post {
                    c.transport.post()[g][i]
                } else {
                    c.transport.pre()[g][i]
                };
                let exact = rat(multipliers[g]) * rat(factor * weights[g][i]);
                assert!(
                    rat(channel.value()) - rat(channel.source_abs_error()) <= exact
                        && exact <= rat(channel.value()) + rat(channel.source_abs_error())
                );
            }
        }
    }
    // This oracle intentionally does not execute nonzero coefficient-radius
    // channels; broader source errors are tested by the independent checked backend.
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
    #[cfg(target_arch = "x86_64")]
    std::thread::spawn(|| {
        let (g, x, b) = fixture(0.);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    std::arch::asm!("ldmxcsr [{}]",in(reg)&self.0,options(nostack,preserves_flags));
                }
            }
        }
        let mut saved = 0u32;
        unsafe {
            std::arch::asm!("stmxcsr [{}]",in(reg)&mut saved,options(nostack,preserves_flags));
        }
        let restore = Restore(saved);
        for bad in [
            saved | (1 << 6),
            saved | (1 << 15),
            (saved & !(3 << 13)) | (1 << 13),
        ] {
            let q = quote(&g, &x, &b, old, end);
            let engine = BundleBackend::default();
            unsafe {
                std::arch::asm!("ldmxcsr [{}]",in(reg)&bad,options(nostack,preserves_flags));
            }
            assert!(evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).is_none());
            assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 0);
            unsafe {
                std::arch::asm!("ldmxcsr [{}]",in(reg)&restore.0,options(nostack,preserves_flags));
            }
        }
    })
    .join()
    .unwrap();
}

pub(crate) fn identity_bn_fixture(
    shift: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    use crate::layers::BatchNormLayer;
    let (mut g, x, _) = fixture(shift);
    let mut bn = BatchNormLayer::new(
        &arr1(&[1.; 17]).into_dyn(),
        &arr1(&[0.; 17]).into_dyn(),
        &arr1(&[0.; 17]).into_dyn(),
        &arr1(&[1.; 17]).into_dyn(),
        0.,
    )
    .unwrap();
    // Authored exact identity, so zero radii have an independent exact proof.
    bn.scale_err.fill(0.);
    bn.bias_err.fill(0.);
    g.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["pre".into()],
    ));
    g.nodes.test_insert(
        "relu".into(),
        GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["bn".into()]),
    );
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
#[test]
fn two_local_original_raw_bn_wiring_source_refusals_and_uncertainty_are_honest() {
    let (g, x, b) = identity_bn_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let engine = BundleBackend::default();
    let result =
        evaluate_raw_two_local_l1_root_cut(quote(&g, &x, &b, old, end), &engine, &mut || true)
            .unwrap();
    assert!(result.bounds().0 > 0.24 && result.bounds().0 <= 0.25);
    assert_eq!(result.bn_radii(), [0., 0.]);
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
    let mut missing = b.clone();
    missing.remove("bn");
    assert!(quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &missing,
        None,
        &[1.],
        old,
        "relu",
        [8, 8],
        IDS,
        [[1.; 8]; 2],
        [1.; 2],
        limits(),
        end,
        &mut || true
    )
    .is_none());
    for malformed in [false, true] {
        let mut altered = g.clone();
        let Layer::BatchNorm(mut bn) = g.node("bn").unwrap().layer().clone() else {
            unreachable!()
        };
        bn.scale_err
            .fill(if malformed { f32::NAN } else { 1. / 1024. });
        bn.bias_err.fill(1. / 64.);
        altered.nodes.test_insert(
            "bn".into(),
            GraphNode::new("bn", Layer::BatchNorm(bn), vec!["pre".into()]),
        );
        let refs = if malformed {
            b.clone()
        } else {
            altered.collect_node_bounds(&x).unwrap()
        };
        let q = quote_raw_two_local_l1_root_cut(
            &altered,
            &x,
            &refs,
            None,
            &[1.],
            (-100., 100.),
            "relu",
            [8, 8],
            IDS,
            [[1.; 8]; 2],
            [1.; 2],
            limits(),
            end,
            &mut || true,
        );
        // The quote validates BN parameter layout and finite radii already.
        // Malformed radii must refuse here; finite uncertainty reaches the
        // deliberately exact-only backend oracle and is refused there.
        assert_eq!(q.is_none(), malformed);
        let engine = BundleBackend::default();
        assert!(q
            .and_then(|q| evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true))
            .is_none());
        assert_eq!(
            engine.bundle_calls.load(Ordering::SeqCst),
            if malformed { 0 } else { 1 }
        );
    }
}

#[test]
fn two_local_raw_candidate_observation_preserves_gain_no_gain_and_unsafe_merge() {
    // The existing independent diamond proof gives the exact minimum .25+shift.
    // Use it as a sound stronger baseline to force a no-gain merge while the
    // same genuine descriptor interpreter returns the same raw candidate.
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(120);
        let triangle = baseline(&g, &x, &b, end);
        let exact_minimum = 0.25 + shift;
        for lambda in [0.5, 1.] {
            let mut candidate_bits = None;
            for old_lower in [triangle.0, exact_minimum] {
                let old = (old_lower, triangle.1);
                let q = quote_raw_two_local_l1_root_cut(
                    &g,
                    &x,
                    &b,
                    None,
                    &[1.],
                    old,
                    "relu",
                    [8, 8],
                    IDS,
                    [[1.; 8]; 2],
                    [lambda; 2],
                    limits(),
                    end,
                    &mut || true,
                )
                .unwrap();
                let engine = BundleBackend::default();
                let result = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
                let raw = result.raw_candidate_lower();
                assert!(ny_core::is_crown_coeff_safe(raw));
                assert_eq!(
                    raw.to_bits(),
                    engine.returned_lower_bits.load(Ordering::SeqCst)
                );
                if let Some(bits) = candidate_bits {
                    assert_eq!(raw.to_bits(), bits);
                } else {
                    candidate_bits = Some(raw.to_bits());
                }
                assert_eq!(result.bounds().0.to_bits(), old_lower.max(raw).to_bits());
                assert_eq!(result.bounds().1.to_bits(), old.1.to_bits());
                if old_lower == triangle.0 {
                    assert!(raw > old_lower);
                    assert_eq!(result.bounds().0.to_bits(), raw.to_bits());
                } else {
                    assert!(raw <= exact_minimum);
                    assert_eq!(result.bounds().0.to_bits(), old_lower.to_bits());
                }
                if lambda == 0.5 {
                    // A deliberately weaker, still sound cut makes the raw
                    // diagnostic visibly different from the strong baseline.
                    assert!(raw > -3.751 + shift && raw <= -3.75 + shift);
                    assert!(raw < exact_minimum);
                } else {
                    assert!(raw > 0.24 + shift && raw <= exact_minimum);
                }
                if shift < 0. {
                    assert!(raw < 0. && result.bounds().0 < 0.);
                }
                assert_eq!(
                    (
                        result.certificates(),
                        result.support_calls(),
                        result.cut_calls()
                    ),
                    (2, 0, 1)
                );
                assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
            }
        }
    }
}

// Comparison controls execute the real exact-descriptor interpreter, never a
// supplied lower scalar. Both one-row transports retain their actual source.
#[derive(Default)]
pub(crate) struct ComparisonBackend {
    local_calls: AtomicUsize,
    bundle_calls: AtomicUsize,
    pub(crate) entered: AtomicUsize,
    allow_first_bundle: bool,
    fail_call: AtomicUsize,
    raw: [AtomicU32; 2],
}
impl GemmEngine for ComparisonBackend {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("comparison has no ordinary GEMM permission")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for ComparisonBackend {
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        panic!("comparison has no ordinary CROWN permission")
    }
    fn provides_bounded_local_l1_lower_cut(&self) -> bool {
        true
    }
    fn provides_bounded_two_local_l1_lower_cut(&self) -> bool {
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
        assert_eq!(self.entered.fetch_add(1, Ordering::SeqCst), 0);
        assert_eq!(c.active_len(), 3);
        self.local_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_call.load(Ordering::SeqCst) == 1 {
            return Err(fail());
        }
        let raw = run(s, q, l, u, Cut::Single(c), end)?;
        self.raw[0].store(raw.to_bits(), Ordering::SeqCst);
        Ok(raw)
    }
    fn crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        c: &ResidentTwoLocalL1LowerCutCarrier,
        end: Instant,
    ) -> Result<f32> {
        assert_eq!(
            self.entered.fetch_add(1, Ordering::SeqCst),
            usize::from(!self.allow_first_bundle)
        );
        self.bundle_calls.fetch_add(1, Ordering::SeqCst);
        match self.fail_call.load(Ordering::SeqCst) {
            2 => return Err(fail()),
            3 => {
                return Err(NyError::DeadlineExceeded(
                    "second checked fold deadline refusal".into(),
                ))
            }
            _ => {}
        }
        let raw = run(s, q, l, u, Cut::Bundle(c), end)?;
        self.raw[1].store(raw.to_bits(), Ordering::SeqCst);
        Ok(raw)
    }
}

fn comparison_fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    // Three independent signed cycles. Each has sum|z|<=2, with equality at
    // original input corners. F=25/4-sum|z| has exact minimum1/4+shift.
    let mut w = Array2::<f32>::zeros((10, 9));
    for k in 0..3 {
        let j = 3 * k;
        w[[j, j]] = 0.5;
        w[[j, j + 1]] = 0.5;
        w[[j + 1, j + 1]] = 0.5;
        w[[j + 1, j + 2]] = 0.5;
        w[[j + 2, j + 2]] = 0.5;
        w[[j + 2, j]] = -0.5;
        w[[9, j + 1]] = 1.;
        w[[9, j + 2]] = 1.;
    }
    let mut bias = vec![0.; 10];
    bias[9] = 7.;
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
    let mut a = vec![-2.; 10];
    a[9] = 1.;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 10), a).unwrap(),
                Some(arr1(&[-0.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 9]).into_dyn(), arr1(&[1.; 9]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
const COMPARISON_IDS: [[usize; 8]; 2] = [[0, 1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]];
const COMPARISON_WEIGHTS: [[f32; 8]; 2] = [
    [1., 1., 1., 1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0., 0., 0., 0.],
];
fn comparison_quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
    lambda: f32,
) -> RawTwoLocalL1RootCutQuote<'a> {
    joint_comparison_quote(g, x, b, old, end, [lambda, 1.])
}
fn joint_comparison_quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
    multipliers: [f32; 2],
) -> RawTwoLocalL1RootCutQuote<'a> {
    quote_raw_two_local_l1_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [6, 3],
        COMPARISON_IDS,
        COMPARISON_WEIGHTS,
        multipliers,
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
    .with_single3_comparison(limits())
    .unwrap()
}

#[test]
fn two_local_single3_comparison_prices_two_full_phases_caps_and_canonical_shapes() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    for first in [6usize, 7, 8] {
        let mut ids = [[0; 8]; 2];
        let mut weights = [[0.; 8]; 2];
        for i in 0..first {
            ids[0][i] = i;
            weights[0][i] = 1.;
        }
        for i in 0..3 {
            ids[1][i] = first + i;
            weights[1][i] = 1.;
        }
        let make = || {
            quote_raw_two_local_l1_root_cut(
                &g,
                &x,
                &b,
                None,
                &[1.],
                old,
                "relu",
                [first, 3],
                ids,
                weights,
                [1.; 2],
                limits(),
                end,
                &mut || true,
            )
            .unwrap()
        };
        let q = make();
        let old_work = q.work();
        let bytes = q.scratch_bytes();
        let expected = 2 * phase_work(q.base.full_quote, 1, 1).unwrap()
            + q.kernel[0].work
            + q.kernel[1].work
            + COMPARISON_FIXED_WORK
            + 2 * EXECUTION_OBSERVATION_WORK;
        assert_eq!(
            old_work,
            phase_work(q.base.full_quote, 1, 1).unwrap()
                + q.kernel[0].work
                + q.kernel[1].work
                + LOCAL_FIXED_WORK
                + EXECUTION_OBSERVATION_WORK
        );
        assert_eq!(q.candidate_count(), 1);
        let upgraded = q.with_single3_comparison(limits());
        if first == 8 {
            assert!(upgraded.is_none());
            continue;
        }
        let q = upgraded.unwrap();
        assert_eq!(
            (q.work(), q.scratch_bytes(), q.candidate_count()),
            (expected, bytes, 2)
        );
        assert!(q.work() > old_work);
        assert!(q.with_single3_comparison(limits()).is_none());
        for (work, scratch, ok) in [
            (expected, bytes, true),
            (expected - 1, bytes, false),
            (expected, bytes - 1, false),
        ] {
            let mut cap = limits();
            cap.max_work = work;
            cap.max_scratch_bytes = scratch;
            assert_eq!(make().with_single3_comparison(cap).is_some(), ok);
        }
        let mut expired = make();
        expired.base.deadline = Instant::now();
        assert!(expired.with_single3_comparison(limits()).is_none());
        let mut overflow = make();
        overflow.kernel[0].work = u64::MAX;
        assert!(overflow.with_single3_comparison(limits()).is_none());
    }
    let reversed = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        [3, 6],
        [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8, 0, 0]],
        [
            [1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0.],
        ],
        [1.; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    assert!(reversed.with_single3_comparison(limits()).is_none());
    // The existing full sixteen-row proposal remains exactly one fold.
    let q = quote(&g, &x, &b, old, end);
    assert_eq!(q.candidate_count(), 1);
    assert!(q.with_single3_comparison(limits()).is_none());
}

#[test]
fn two_local_single3_comparison_retains_actual_best_for_both_winners_and_unsafe_shift() {
    for shift in [0., -0.5] {
        let (g, x, b) = comparison_fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        // Independent exact cube support census for the source's three cycles.
        let mut maximum = 0;
        for bits in 0usize..512 {
            let a: Vec<i32> = (0..9)
                .map(|i| if bits & (1 << i) == 0 { -1 } else { 1 })
                .collect();
            let twice_abs: i32 = (0..3)
                .map(|k| {
                    let j = 3 * k;
                    (a[j] + a[j + 1]).abs() + (a[j + 1] + a[j + 2]).abs() + (a[j + 2] - a[j]).abs()
                })
                .sum();
            maximum = maximum.max(twice_abs);
        }
        assert_eq!(maximum, 12);
        for lambda in [1., 2.] {
            let engine = ComparisonBackend::default();
            let out = evaluate_raw_two_local_l1_root_cut(
                comparison_quote(&g, &x, &b, old, end, lambda),
                &engine,
                &mut || true,
            )
            .unwrap();
            let single = f32::from_bits(engine.raw[0].load(Ordering::SeqCst));
            let bundle = f32::from_bits(engine.raw[1].load(Ordering::SeqCst));
            assert_eq!(
                out.single3_candidate_lower().unwrap().to_bits(),
                single.to_bits()
            );
            assert_eq!(
                out.bundle_candidate_lower().unwrap().to_bits(),
                bundle.to_bits()
            );
            let selected = if single >= bundle { single } else { bundle };
            assert_eq!(out.raw_candidate_lower().to_bits(), selected.to_bits());
            assert_eq!(out.bounds().0.to_bits(), old.0.max(selected).to_bits());
            assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
            assert_eq!(
                (out.certificates(), out.support_calls(), out.cut_calls()),
                (2, 0, 2)
            );
            assert!(!engine.provides_bounded_support4());
            assert_eq!(
                (
                    engine.local_calls.load(Ordering::SeqCst),
                    engine.bundle_calls.load(Ordering::SeqCst)
                ),
                (1, 1)
            );
            assert!(out.bounds().0 <= 0.25 + shift);
            if lambda == 1. {
                assert!(bundle > single && out.bounds().0 > 0.24 + shift);
            } else {
                assert!(single > bundle);
            }
        }
        // A known sound incumbent may already be stronger. Neither raw scalar
        // replaces it; exact upper bits and the incoming lower are retained.
        let strong = (0.25 + shift, old.1);
        let out = evaluate_raw_two_local_l1_root_cut(
            comparison_quote(&g, &x, &b, strong, end, 2.),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert_eq!(out.bounds().0.to_bits(), strong.0.to_bits());
        assert_eq!(out.bounds().1.to_bits(), strong.1.to_bits());
    }
}

#[test]
fn two_local_single3_comparison_failure_and_late_barriers_publish_no_partial_candidate() {
    let (g, x, b) = comparison_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    for fail_at in [1usize, 2, 3] {
        let engine = ComparisonBackend::default();
        engine.fail_call.store(fail_at, Ordering::SeqCst);
        assert!(evaluate_raw_two_local_l1_root_cut(
            comparison_quote(&g, &x, &b, old, end, 1.),
            &engine,
            &mut || true
        )
        .is_none());
        assert_eq!(
            engine.entered.load(Ordering::SeqCst),
            if fail_at == 1 { 1 } else { 2 }
        );
    }
    // Cancellation immediately after either actual returned fold must refuse,
    // including the final barrier after both source-valid scalars exist.
    for stop_after in [1usize, 2] {
        let engine = ComparisonBackend::default();
        assert!(evaluate_raw_two_local_l1_root_cut(
            comparison_quote(&g, &x, &b, old, end, 1.),
            &engine,
            &mut || engine.entered.load(Ordering::SeqCst) < stop_after
        )
        .is_none());
        assert_eq!(engine.entered.load(Ordering::SeqCst), stop_after);
    }
    let no_local = BundleBackend::default();
    assert!(evaluate_raw_two_local_l1_root_cut(
        comparison_quote(&g, &x, &b, old, end, 1.),
        &no_local,
        &mut || true
    )
    .is_none());
    assert_eq!(no_local.bundle_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn two_local_single3_comparison_fresh_custody_bias_rounding_and_generations() {
    let (g, x, b) = comparison_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut q = comparison_quote(&g, &x, &b, old, end, 1.);
    q.multipliers[1] = f32::from_bits(0x3eaaaaab);
    let other = comparison_quote(&g, &x, &b, old, end, 1.);
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
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let alpha = GraphAlphaState::new();
    let engine = ComparisonBackend::default();
    let base = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        &g,
        &x,
        &alpha,
        &b,
        Some(&engine),
        &seed,
        "relu",
        [q.neurons[0][0], q.neurons[0][1]],
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        x.lower().as_slice().unwrap(),
        x.upper().as_slice().unwrap(),
        end,
    );
    let context = TwoLocalContext::new(&q, base).unwrap();
    macro_rules! make_kernel {
        ($map:expr,$group:expr,$weights:expr) => {
            local_box_l1::evaluate(
                &g,
                &x,
                $map,
                "relu",
                &q.neurons[$group][..q.active[$group]],
                $weights,
                ny_core::CertifiedWeightError::EXACT,
                q.kernel[$group],
                end,
                &mut || true,
            )
            .unwrap()
        };
    }
    let mut fresh = FreshTwoLocalSupport {
        seal: &q.seal,
        arithmetic: [
            make_kernel!(&b, 0, &q.weights[0][..6]),
            make_kernel!(&b, 1, &q.weights[1][..3]),
        ],
    };
    let c = context.build_single3(&fresh).unwrap();
    let exact = -rat(q.multipliers[1]) * rat(fresh.arithmetic[1].bound());
    let bias = c.transport.bias();
    assert!(rat(bias.value()) - rat(bias.source_abs_error()) <= exact);
    assert!(rat(bias.value()) + rat(bias.source_abs_error()) >= exact);
    assert_eq!(
        c.transport.multiplier().to_bits(),
        q.multipliers[1].to_bits()
    );
    fresh.seal = &other.seal;
    assert!(context.build_single3(&fresh).is_err());
    fresh.seal = &q.seal;
    let copied = b.clone();
    fresh.arithmetic[1] = make_kernel!(&copied, 1, &q.weights[1][..3]);
    assert!(context.build_single3(&fresh).is_err());
    fresh.arithmetic[1] = make_kernel!(&b, 1, &[1., 1., 0.5]);
    assert!(context.build_single3(&fresh).is_err());
    fresh.arithmetic[1] = make_kernel!(&b, 1, &q.weights[1][..3]);
    // An actual first fold changes the generation. Its carrier cannot run
    // again, and a prebuilt bundle carrier must not survive that transition.
    let bundle_before = context.build(&fresh).unwrap();
    let c = context.build_single3(&fresh).unwrap();
    let result = context.run_single3(&c, &engine).unwrap();
    assert!(context.run_single3(&c, &engine).is_err());
    assert!(context.run(&bundle_before, &engine).is_err());
    let baseline = ny_core::GpuCrownResult {
        lower_bounds: vec![old.0],
        upper_bounds: vec![old.1],
    };
    let first = context.base.merge_backend_lower(result, &baseline).unwrap();
    assert_eq!(first.upper_bounds[0].to_bits(), old.1.to_bits());
    let bundle = context.build(&fresh).unwrap();
    let result = context.run(&bundle, &engine).unwrap();
    let both = context.base.merge_backend_lower(result, &baseline).unwrap();
    assert_eq!(both.upper_bounds[0].to_bits(), old.1.to_bits());
    assert_eq!(engine.entered.load(Ordering::SeqCst), 2);
}

// The stable tenth row is essential: selected-only advice discards its source
// cancellation and prefers the old endpoint. This is an original network, not
// a supplied proxy vector or a backend returning a canned scalar.
pub(crate) fn full_head_fixture(
    shift: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    full_head_fixture_impl(shift, false)
}
fn full_head_fixture_impl(
    shift: f32,
    correlated: bool,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    if correlated {
        // u0=u1=t,u2=-t; the parent box is still [-1,1]^9. Actual backward
        // contraction retains this correlation, while the advisory box cannot.
        let mut prefix = Array2::<f32>::zeros((9, 7));
        prefix[[0, 0]] = 1.;
        prefix[[2, 0]] = 1.;
        prefix[[4, 0]] = -1.;
        for (row, col) in [(1, 1), (3, 2), (5, 3), (6, 4), (7, 5), (8, 6)] {
            prefix[[row, col]] = 1.;
        }
        g.add_node(GraphNode::from_input(
            "parent",
            Layer::Linear(LinearLayer::new(prefix, None).unwrap()),
        ));
    }
    let mut w = Array2::<f32>::zeros((10, 9));
    for k in 0..3 {
        let j = 2 * k;
        w[[j, j]] = 0.5;
        w[[j, j + 1]] = 0.5;
        w[[j + 1, j]] = 0.5;
        w[[j + 1, j + 1]] = -0.5;
        w[[9, j]] = if correlated {
            [5. / 4., 9. / 8., 11. / 8.][k]
        } else {
            21. / 16.
        };
    }
    w[[6, 6]] = 0.5;
    w[[6, 7]] = 0.5;
    w[[7, 7]] = 0.5;
    w[[7, 8]] = 0.5;
    w[[8, 8]] = 0.5;
    w[[8, 6]] = -0.5;
    w[[9, 7]] = 1.;
    w[[9, 8]] = 1.;
    let mut bias = vec![0.; 10];
    bias[..6].fill(0.5);
    bias[9] = 6.;
    let layer = Layer::Linear(LinearLayer::new(w, Some(ndarray::Array1::from_vec(bias))).unwrap());
    g.add_node(if correlated {
        GraphNode::new("pre", layer, vec!["parent".into()])
    } else {
        GraphNode::from_input("pre", layer)
    });
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    let mut a = vec![-2.; 10];
    a[9] = 1.;
    let constant = if correlated { 41. / 8. } else { 21. / 4. };
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 10), a).unwrap(),
                Some(arr1(&[constant + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let width = if correlated { 7 } else { 9 };
    let x = BoundedTensor::new(
        ndarray::Array1::from_elem(width, -1.).into_dyn(),
        ndarray::Array1::from_elem(width, 1.).into_dyn(),
    )
    .unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}

#[test]
fn two_local_full_head_quote_is_optional_atomic_and_prices_exact_extra_work_and_storage() {
    let native = full_head_advice::quote(32768, 512, 10).unwrap();
    assert_eq!(native.work, 3_045_976_064);
    assert_eq!(native.scratch_bytes, 1_155_072);
    assert_eq!(11_892_051_456u64 + native.work, 14_938_027_520);
    let (g, x, b) = full_head_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let make = || comparison_quote(&g, &x, &b, old, end, 1.);
    let before = make();
    let extra = full_head_advice::quote(9, 10, 1).unwrap();
    let work = before.work() + extra.work;
    let scratch = before.scratch_bytes() + extra.scratch_bytes;
    for (max_work, max_scratch, ok) in [
        (work, scratch, true),
        (work - 1, scratch, false),
        (work, scratch - 1, false),
    ] {
        let mut q = make();
        let mut cap = limits();
        cap.max_work = max_work;
        cap.max_scratch_bytes = max_scratch;
        assert_eq!(q.try_enable_full_head_advice(cap), ok);
        assert_eq!(q.uses_full_head_advice(), ok);
        assert_eq!(
            (q.work(), q.scratch_bytes(), q.candidate_count()),
            if ok {
                (work, scratch, 2)
            } else {
                (before.work(), before.scratch_bytes(), 2)
            }
        );
        if ok {
            assert!(!q.try_enable_full_head_advice(limits()));
            assert_eq!((q.work(), q.scratch_bytes()), (work, scratch));
        }
    }
    for mode in 0..5 {
        let mut q = make();
        match mode {
            0 => q.base.work = u64::MAX,
            1 => q.base.scratch = usize::MAX,
            2 => q.base.deadline = Instant::now(),
            3 => q.source_shape[0] = usize::MAX,
            _ => q.active = [8, 3],
        }
        let previous = (q.work(), q.scratch_bytes());
        assert!(!q.try_enable_full_head_advice(limits()));
        assert_eq!((q.work(), q.scratch_bytes()), previous);
        assert!(!q.uses_full_head_advice());
    }
    let mut one = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        [6, 3],
        COMPARISON_IDS,
        COMPARISON_WEIGHTS,
        [1.; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    let before_one = (one.work(), one.scratch_bytes());
    assert!(one.try_enable_full_head_advice(limits()));
    assert_eq!(one.work(), before_one.0 + extra.work);
    assert_eq!(one.scratch_bytes(), before_one.1 + extra.scratch_bytes);
    assert_eq!(one.candidate_count(), 1);
    let reverse = one.with_single3_comparison(limits()).unwrap();
    let mut forward = make();
    assert!(forward.try_enable_full_head_advice(limits()));
    // This configuration order became reachable when advice was decoupled
    // from comparison. The later comparison must preserve its entire price.
    assert_eq!(
        (
            reverse.work(),
            reverse.scratch_bytes(),
            reverse.candidate_count()
        ),
        (
            forward.work(),
            forward.scratch_bytes(),
            forward.candidate_count()
        ),
    );
    let q = make();
    let mut cap = limits();
    cap.max_nodes = q.base.extraction.max_nodes - 1;
    let mut q = q;
    assert!(!q.try_enable_full_head_advice(cap));
    assert_eq!(q.work(), before.work());
}

#[test]
fn two_local_full_head_background_finds_genuine_interior_gain_and_preserves_unsafe_incumbent() {
    for shift in [0., -23. / 16.] {
        let (g, x, b) = full_head_fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        // Exact source proof: each shifted diamond minimizes at u=1,v=0;
        // the cycle maximizes its absolute sum at a=b=c=1. The total minimum
        // is19/16+shift. The unsafe shift therefore has a real -1/4 witness.
        let Layer::Linear(pre) = g.node("pre").unwrap().layer() else {
            unreachable!()
        };
        let Layer::Linear(last) = g.node("out").unwrap().layer() else {
            unreachable!()
        };
        let zero = Q::from_integer(0.into());
        let original_value = |input: &[f32]| {
            let mut out = rat(last.bias.as_ref().unwrap()[0]);
            let mut stable = zero.clone();
            for j in 0..10 {
                let mut z = rat(pre.bias.as_ref().unwrap()[j]);
                for k in 0..9 {
                    z += rat(pre.weight[[j, k]]) * rat(input[k]);
                }
                if j == 9 {
                    stable = z.clone();
                }
                out += rat(last.weight[[0, j]]) * z.max(zero.clone());
            }
            (out, stable)
        };
        let mut minimum: Option<Q> = None;
        for bits in 0usize..512 {
            let input: Vec<f32> = (0..9)
                .map(|k| if bits & (1 << k) == 0 { -1. } else { 1. })
                .collect();
            let (value, stable) = original_value(&input);
            // The tenth ReLU is affine everywhere: its affine minimum is
            // attained at a cube vertex. The remaining objective is concave,
            // so its minimum also has a cube-vertex representative.
            assert!(stable >= rat(1. / 16.));
            if minimum.as_ref().is_none_or(|v| value < *v) {
                minimum = Some(value);
            }
        }
        let minimum = minimum.unwrap();
        assert_eq!(minimum, rat(19. / 16. + shift));
        let (witness, _) = original_value(&[1., 0., 1., 0., 1., 0., 1., 1., 1.]);
        assert_eq!(witness, minimum);
        if shift < 0. {
            assert_eq!(witness, rat(-0.25));
        }
        let truth = rounded(&minimum, true).unwrap();
        let endpoint = evaluate_raw_two_local_l1_root_cut(
            comparison_quote(&g, &x, &b, old, end, 1.),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert!(endpoint.single3_candidate_lower().unwrap() > -5. / 16. + shift - 0.001);
        assert!(endpoint.single3_candidate_lower().unwrap() <= -5. / 16. + shift);
        assert!(endpoint.bundle_candidate_lower().unwrap() > -11. / 16. + shift - 0.001);
        assert!(endpoint.bundle_candidate_lower().unwrap() <= -11. / 16. + shift);
        // Independent actual checked descriptor evaluation at the analytic
        // kink. The search is not required to return universal exact f32 bits.
        let kink = evaluate_raw_two_local_l1_root_cut(
            comparison_quote(&g, &x, &b, old, end, 3. / 8.),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert!(
            kink.bundle_candidate_lower().unwrap() > 0.249 + shift
                && kink.bundle_candidate_lower().unwrap() <= 0.25 + shift
        );
        for incoming in [old, (truth, old.1)] {
            let mut q = comparison_quote(&g, &x, &b, incoming, end, 1.);
            assert!(q.try_enable_full_head_advice(limits()));
            let engine = ComparisonBackend::default();
            let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
            let advice = out.multiplier_advice().unwrap();
            assert_eq!(advice.original, [1.; 2]);
            assert!(advice.chosen[0] > 0.37 && advice.chosen[0] < 0.38);
            assert!(advice.chosen_score > advice.original_score + 0.9);
            assert!(advice.score_evaluations <= 17 && advice.full_scans <= 17);
            assert_eq!(out.multipliers()[0].to_bits(), advice.chosen[0].to_bits());
            assert_eq!(out.multipliers()[1].to_bits(), 1f32.to_bits());
            let single = f32::from_bits(engine.raw[0].load(Ordering::SeqCst));
            let bundle = f32::from_bits(engine.raw[1].load(Ordering::SeqCst));
            assert_eq!(
                single.to_bits(),
                endpoint.single3_candidate_lower().unwrap().to_bits()
            );
            assert_eq!(
                out.bundle_candidate_lower().unwrap().to_bits(),
                bundle.to_bits()
            );
            assert!(bundle > endpoint.raw_candidate_lower() + 0.55);
            assert!(bundle > 0.249 + shift && bundle <= 0.25 + shift && bundle <= truth);
            assert!((bundle - kink.bundle_candidate_lower().unwrap()).abs() < 0.001);
            assert_eq!(
                out.raw_candidate_lower().to_bits(),
                single.max(bundle).to_bits()
            );
            assert_eq!(
                out.bounds().0.to_bits(),
                incoming.0.max(single.max(bundle)).to_bits()
            );
            assert_eq!(out.bounds().1.to_bits(), incoming.1.to_bits());
            assert_eq!(
                (out.certificates(), out.support_calls(), out.cut_calls()),
                (2, 0, 2)
            );
            assert_eq!(engine.entered.load(Ordering::SeqCst), 2);
            if shift < 0. {
                assert!(out.bounds().0 < 0.);
            }
        }
    }
}

#[test]
fn two_local_full_head_parent_proxy_does_not_dominate_unexecuted_correlated_prefix_bundle() {
    let (g, x, b) = full_head_fixture_impl(0., true);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let original = evaluate_raw_two_local_l1_root_cut(
        comparison_quote(&g, &x, &b, old, end, 1.),
        &ComparisonBackend::default(),
        &mut || true,
    )
    .unwrap();
    let mut q = comparison_quote(&g, &x, &b, old, end, 1.);
    assert!(q.try_enable_full_head_advice(limits()));
    let selected =
        evaluate_raw_two_local_l1_root_cut(q, &ComparisonBackend::default(), &mut || true).unwrap();
    let advice = selected.multiplier_advice().unwrap();
    assert!((advice.chosen[0] - 0.5).abs() < 0.001 && advice.chosen_score > advice.original_score);
    // Parent-box score peaks at .5, but the actual earlier Linear cancels
    // u0=u1=-u2. Its complete fold peaks at1. This deliberately documents the
    // policy tradeoff, not a theorem that advice improves every actual fold.
    assert!(
        original.bundle_candidate_lower().unwrap() > 0.124
            && original.bundle_candidate_lower().unwrap() <= 0.125
    );
    assert!(
        selected.bundle_candidate_lower().unwrap() > -0.126
            && selected.bundle_candidate_lower().unwrap() <= -0.124
    );
    assert!(
        selected.single3_candidate_lower().unwrap() > -0.376
            && selected.single3_candidate_lower().unwrap() <= -0.375
    );
    assert!(selected.raw_candidate_lower() < original.raw_candidate_lower() - 0.24);
    assert_eq!(
        selected.bounds().0.to_bits(),
        old.0.max(selected.raw_candidate_lower()).to_bits()
    );
    assert_eq!(selected.bounds().1.to_bits(), old.1.to_bits());
}

#[test]
fn two_local_full_head_advice_and_second_fold_refusals_publish_no_partial_candidate() {
    let (g, x, b) = full_head_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let make = || {
        let mut q = comparison_quote(&g, &x, &b, old, end, 1.);
        assert!(q.try_enable_full_head_advice(limits()));
        q
    };
    for fail_at in [1usize, 2, 3] {
        let engine = ComparisonBackend::default();
        engine.fail_call.store(fail_at, Ordering::SeqCst);
        assert!(evaluate_raw_two_local_l1_root_cut(make(), &engine, &mut || true).is_none());
        assert_eq!(
            engine.entered.load(Ordering::SeqCst),
            if fail_at == 1 { 1 } else { 2 }
        );
    }
    for stop_after in [0usize, 1, 2] {
        let engine = ComparisonBackend::default();
        assert!(
            evaluate_raw_two_local_l1_root_cut(make(), &engine, &mut || engine
                .entered
                .load(Ordering::SeqCst)
                < stop_after)
            .is_none()
        );
        assert_eq!(engine.entered.load(Ordering::SeqCst), stop_after);
    }
}

#[test]
fn two_local_full_head_selected_scale_retains_fresh_custody_directed_products_and_generation() {
    let (g, x, b) = joint_head_fixture(0., false);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut q = joint_comparison_quote(&g, &x, &b, old, end, [1., 2.]);
    assert!(q.try_enable_full_head_advice(limits()));
    let other = joint_comparison_quote(&g, &x, &b, old, end, [1., 2.]);
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
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let alpha = GraphAlphaState::new();
    let engine = ComparisonBackend::default();
    let base = ResidentCutCallContext::new(
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
    );
    let context = TwoLocalContext::new(&q, base).unwrap();
    let make = |group: usize| {
        local_box_l1::evaluate(
            &g,
            &x,
            &b,
            "relu",
            &q.neurons[group][..q.active[group]],
            &q.weights[group][..q.active[group]],
            ny_core::CertifiedWeightError::EXACT,
            q.kernel[group],
            end,
            &mut || true,
        )
        .unwrap()
    };
    let mut fresh = FreshTwoLocalSupport {
        seal: &q.seal,
        arithmetic: [make(0), make(1)],
    };
    let mut yes = || true;
    let mut guard = Poll {
        end,
        callback: &mut yes,
        alive: true,
    };
    fresh.seal = &other.seal;
    assert!(context.choose_scale(&fresh, &mut guard).is_none());
    fresh.seal = &q.seal;
    let mut choice = context.choose_scale(&fresh, &mut guard).unwrap();
    assert!(context.build(&fresh).is_err()); // An enabled quote cannot omit its selected-scale seal.
    choice.seal = &other.seal;
    assert!(context.build_with_scale(&fresh, Some(&choice)).is_err());
    choice.seal = &q.seal;
    let original = choice.decision;
    for group in 0..2 {
        for bad in [
            0.,
            f32::NAN,
            q.multipliers[group] * 1.01,
            q.multipliers[group] / 32.,
        ] {
            choice.decision = original;
            choice.decision.multipliers[group] = bad;
            assert!(context.validate_scale(&choice).is_err());
        }
        choice.decision = original;
        choice.original_multipliers[group] = q.multipliers[group] / 2.;
        assert!(context.validate_scale(&choice).is_err());
        choice.original_multipliers = q.multipliers;
        choice.decision.branch_upper[group] = 0.;
        assert!(context.validate_scale(&choice).is_err());
    }
    // A genuine empty guarded rectangle returns only its separately scored
    // original pair. Zero caps are diagnostics, never zero-positive carriers.
    choice.decision = original;
    choice.decision.multipliers = q.multipliers;
    choice.decision.branch_upper = [0.; 2];
    choice.decision.chosen_score = original.original_score;
    choice.decision.full_scans = 1;
    choice.decision.score_evaluations = 1;
    choice.decision.support_planes = 0;
    choice.decision.lp_candidates = 0;
    assert!(context.validate_scale(&choice).is_ok());
    let unchanged = context.build_with_scale(&fresh, Some(&choice)).unwrap();
    assert_eq!(*unchanged.transport.multipliers(), q.multipliers);
    choice.decision.multipliers[0] = q.multipliers[0] / 2.;
    assert!(context.validate_scale(&choice).is_err());
    for group in 0..2 {
        choice.decision = original;
        choice.decision.branch_upper[group] = f32::NAN;
        assert!(context.validate_scale(&choice).is_err());
    }
    choice.decision = original;
    choice.decision.chosen_score = f64::NAN;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    choice.decision.chosen_score = original.original_score - 1.;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    choice.decision.full_scans = 18;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    choice.decision.score_evaluations = 18;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    choice.decision.support_planes = 17;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    choice.decision.lp_candidates = 4121;
    assert!(context.validate_scale(&choice).is_err());
    choice.decision = original;
    let carrier = context.build_with_scale(&fresh, Some(&choice)).unwrap();
    let mut exact_bias = Q::from_integer(0.into());
    for group in 0..2 {
        let lambda = choice.decision.multipliers[group];
        exact_bias -= rat(lambda) * rat(fresh.arithmetic[group].bound());
        assert_eq!(
            carrier.transport.multipliers()[group].to_bits(),
            lambda.to_bits()
        );
        for i in 0..q.active[group] {
            let exact = rat(lambda) * rat(q.weights[group][i]);
            for (channel, expected) in [
                (carrier.transport.pre()[group][i], -exact.clone()),
                (
                    carrier.transport.post()[group][i],
                    exact * Q::from_integer(2.into()),
                ),
            ] {
                assert!(rat(channel.value()) - rat(channel.source_abs_error()) <= expected);
                assert!(rat(channel.value()) + rat(channel.source_abs_error()) >= expected);
            }
        }
    }
    let bias = carrier.transport.bias();
    assert!(rat(bias.value()) - rat(bias.source_abs_error()) <= exact_bias);
    assert!(rat(bias.value()) + rat(bias.source_abs_error()) >= exact_bias);
    let single = context.build_single3(&fresh).unwrap();
    assert_eq!(
        single.transport.multiplier().to_bits(),
        q.multipliers[1].to_bits()
    );
    assert_ne!(
        single.transport.multiplier().to_bits(),
        choice.decision.multipliers[1].to_bits()
    );
    let first = context.run_single3(&single, &engine).unwrap();
    let incoming = ny_core::GpuCrownResult {
        lower_bounds: vec![old.0],
        upper_bounds: vec![old.1],
    };
    context.base.merge_backend_lower(first, &incoming).unwrap();
    assert!(context.run(&carrier, &engine).is_err()); // Stale generation never reaches backend.
    let current = context.build_with_scale(&fresh, Some(&choice)).unwrap();
    let second = context.run(&current, &engine).unwrap();
    let merged = context.base.merge_backend_lower(second, &incoming).unwrap();
    assert_eq!(merged.upper_bounds[0].to_bits(), old.1.to_bits());
    assert_eq!(engine.entered.load(Ordering::SeqCst), 2);
}

// Shift the cycle as well as the diamonds. Its optimal multiplier now moves,
// so a complement-only search cannot recover the positive checked bound.
fn joint_head_fixture(
    shift: f32,
    correlated: bool,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    if correlated {
        let mut prefix = Array2::<f32>::zeros((9, 6));
        // Same parent box, but u0=u1=-u2 and b=c. These correlations are
        // visible to the real full fold and absent from the advisory box.
        for (row, col, value) in [
            (0, 0, 1.),
            (2, 0, 1.),
            (4, 0, -1.),
            (1, 1, 1.),
            (3, 2, 1.),
            (5, 3, 1.),
            (6, 4, 1.),
            (7, 5, 1.),
            (8, 5, 1.),
        ] {
            prefix[[row, col]] = value;
        }
        g.add_node(GraphNode::from_input(
            "parent",
            Layer::Linear(LinearLayer::new(prefix, None).unwrap()),
        ));
    }
    let mut w = Array2::<f32>::zeros((10, 9));
    for k in 0..3 {
        let j = 2 * k;
        w[[j, j]] = 0.5;
        w[[j, j + 1]] = 0.5;
        w[[j + 1, j]] = 0.5;
        w[[j + 1, j + 1]] = -0.5;
        w[[9, j]] = if correlated {
            [5. / 4., 9. / 8., 11. / 8.][k]
        } else {
            21. / 16.
        };
    }
    for (row, col, value) in [
        (6, 6, 0.5),
        (6, 7, 0.5),
        (7, 7, 0.5),
        (7, 8, 0.5),
        (8, 8, 0.5),
        (8, 6, -0.5),
    ] {
        w[[row, col]] = value;
    }
    w[[9, 7]] = 5. / 2.;
    w[[9, 8]] = if correlated { 3. / 2. } else { 5. / 2. };
    let mut bias = vec![0.5; 10];
    bias[9] = 9.;
    let layer = Layer::Linear(LinearLayer::new(w, Some(ndarray::Array1::from_vec(bias))).unwrap());
    g.add_node(if correlated {
        GraphNode::new("pre", layer, vec!["parent".into()])
    } else {
        GraphNode::from_input("pre", layer)
    });
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    let mut a = vec![-2.; 10];
    a[6..9].fill(-4.);
    a[9] = 1.;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 10), a).unwrap(),
                Some(arr1(&[39. / 4. + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let width = if correlated { 6 } else { 9 };
    let x = BoundedTensor::new(
        ndarray::Array1::from_elem(width, -1.).into_dyn(),
        ndarray::Array1::from_elem(width, 1.).into_dyn(),
    )
    .unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}

#[test]
fn two_local_joint_multipliers_improve_actual_fold_with_original_single_anchor_and_unsafe_control()
{
    for shift in [0., -31. / 16.] {
        let (g, x, b) = joint_head_fixture(shift, false);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        let Layer::Linear(pre) = g.node("pre").unwrap().layer() else {
            unreachable!()
        };
        let Layer::Linear(last) = g.node("out").unwrap().layer() else {
            unreachable!()
        };
        let zero = Q::from_integer(0.into());
        let value = |input: &[f32]| {
            let mut out = rat(last.bias.as_ref().unwrap()[0]);
            for j in 0..10 {
                let mut z = rat(pre.bias.as_ref().unwrap()[j]);
                for k in 0..9 {
                    z += rat(pre.weight[[j, k]]) * rat(input[k]);
                }
                if j == 9 {
                    assert!(z >= rat(1. / 16.));
                }
                out += rat(last.weight[[0, j]]) * z.max(zero.clone());
            }
            out
        };
        // Evaluate the actual fixture bytes at every cube vertex. The stable
        // tenth row stays positive, so the remaining objective is concave.
        let exact_min = (0usize..512)
            .map(|bits| {
                let input: Vec<f32> = (0..9)
                    .map(|k| if bits & (1 << k) == 0 { -1. } else { 1. })
                    .collect();
                value(&input)
            })
            .min()
            .unwrap();
        assert_eq!(exact_min, rat(27. / 16. + shift));
        assert_eq!(value(&[1.; 9]), exact_min);
        if shift < 0. {
            assert_eq!(exact_min, rat(-0.25));
        }
        let truth = rounded(&exact_min, true).unwrap();
        let endpoint = evaluate_raw_two_local_l1_root_cut(
            joint_comparison_quote(&g, &x, &b, old, end, [1., 2.]),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert!((endpoint.single3_candidate_lower().unwrap() - (-29. / 16. + shift)).abs() < 0.001);
        assert!((endpoint.bundle_candidate_lower().unwrap() - (-35. / 16. + shift)).abs() < 0.001);
        // Even the best complement for the original cycle multiplier is bad.
        let one_axis = evaluate_raw_two_local_l1_root_cut(
            joint_comparison_quote(&g, &x, &b, old, end, [3. / 8., 2.]),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert!((one_axis.bundle_candidate_lower().unwrap() - (-5. / 4. + shift)).abs() < 0.001);
        // Independent checked evaluation of the analytic pair, not a forced
        // bitwise assertion on the floating search's returned point.
        let kink = evaluate_raw_two_local_l1_root_cut(
            joint_comparison_quote(&g, &x, &b, old, end, [3. / 8., 1.]),
            &ComparisonBackend::default(),
            &mut || true,
        )
        .unwrap();
        assert!(
            kink.bundle_candidate_lower().unwrap() > 0.249 + shift
                && kink.bundle_candidate_lower().unwrap() <= 0.25 + shift
        );
        for incoming in [old, (truth, old.1)] {
            let mut q = joint_comparison_quote(&g, &x, &b, incoming, end, [1., 2.]);
            assert!(q.try_enable_full_head_advice(limits()));
            let engine = ComparisonBackend::default();
            let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
            let advice = out.multiplier_advice().unwrap();
            assert_eq!(advice.original, [1., 2.]);
            assert!((advice.chosen[0] - 3. / 8.).abs() < 0.001);
            assert!((advice.chosen[1] - 1.).abs() < 0.001);
            assert!(advice.chosen_score > advice.original_score + 2.4);
            assert!(
                advice.score_evaluations <= 17 && advice.full_scans == advice.score_evaluations
            );
            assert!(advice.support_planes <= 16 && advice.lp_candidates <= 4120);
            assert_eq!(out.multipliers(), advice.chosen);
            let single = f32::from_bits(engine.raw[0].load(Ordering::SeqCst));
            let bundle = f32::from_bits(engine.raw[1].load(Ordering::SeqCst));
            assert_eq!(
                single.to_bits(),
                endpoint.single3_candidate_lower().unwrap().to_bits()
            );
            assert!(bundle > 0.249 + shift && bundle <= 0.25 + shift && bundle <= truth);
            assert!((bundle - kink.bundle_candidate_lower().unwrap()).abs() < 0.001);
            assert_eq!(
                out.raw_candidate_lower().to_bits(),
                single.max(bundle).to_bits()
            );
            assert_eq!(
                out.bounds().0.to_bits(),
                incoming.0.max(single.max(bundle)).to_bits()
            );
            assert_eq!(out.bounds().1.to_bits(), incoming.1.to_bits());
            assert_eq!(
                (out.certificates(), out.support_calls(), out.cut_calls()),
                (2, 0, 2)
            );
            assert_eq!(engine.entered.load(Ordering::SeqCst), 2);
            if shift < 0. {
                assert!(out.bounds().0 < 0.);
            }
        }
    }
}

#[test]
fn two_local_joint_advice_correlated_prefix_retains_only_actual_single_anchor_dominance() {
    let (g, x, b) = joint_head_fixture(0., true);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let original = evaluate_raw_two_local_l1_root_cut(
        joint_comparison_quote(&g, &x, &b, old, end, [1., 2.]),
        &ComparisonBackend::default(),
        &mut || true,
    )
    .unwrap();
    let mut q = joint_comparison_quote(&g, &x, &b, old, end, [1., 2.]);
    assert!(q.try_enable_full_head_advice(limits()));
    let selected =
        evaluate_raw_two_local_l1_root_cut(q, &ComparisonBackend::default(), &mut || true).unwrap();
    let advice = selected.multiplier_advice().unwrap();
    assert!((advice.chosen[0] - 0.5).abs() < 0.001 && (advice.chosen[1] - 1.).abs() < 0.001);
    assert!(advice.chosen_score > advice.original_score + 0.99);
    assert!((original.bundle_candidate_lower().unwrap() + 0.25).abs() < 0.001);
    assert!((selected.bundle_candidate_lower().unwrap() + 1.).abs() < 0.001);
    let single = selected.single3_candidate_lower().unwrap();
    assert!((single + 0.75).abs() < 0.001);
    assert_eq!(
        single.to_bits(),
        original.single3_candidate_lower().unwrap().to_bits()
    );
    assert_eq!(selected.raw_candidate_lower().to_bits(), single.to_bits());
    assert!(selected.raw_candidate_lower() < original.raw_candidate_lower() - 0.49);
    assert_eq!(selected.bounds().0.to_bits(), old.0.max(single).to_bits());
    assert_eq!(selected.bounds().1.to_bits(), old.1.to_bits());
}

#[test]
fn head_box_floor_same_source_frontier_gain_fresh_custody_and_unsafe_boundary() {
    for offset in [0.5, 0., -0.5] {
        let (g, x, b) = head_box_frontier_fixture(offset);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= offset - 0.99);
        let ids = [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]];
        let weights = [[1., 1., 1., 0., 0., 0., 0., 0.]; 2];
        let q = quote_raw_two_local_l1_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            old,
            "relu",
            [3, 3],
            ids,
            weights,
            [1.; 2],
            limits(),
            end,
            &mut || true,
        )
        .unwrap();
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
        let alpha = GraphAlphaState::new();
        let engine = BundleBackend::default();
        let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
        let base = ResidentCutCallContext::new(
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
        );
        let context = TwoLocalContext::new(&q, base).unwrap();
        let mut yes = || true;
        let mut guard = Poll {
            end,
            callback: &mut yes,
            alive: true,
        };
        let mut obs = ExecutionObservation::default();
        let fresh = FreshTwoLocalSupport {
            seal: &q.seal,
            arithmetic: [
                evaluate_group(&q, 0, &mut guard, &mut obs).unwrap(),
                evaluate_group(&q, 1, &mut guard, &mut obs).unwrap(),
            ],
        };
        let carrier = context.build(&fresh).unwrap();
        let paid = head_box_floor::quote(2, 9, 1).unwrap();
        let result =
            head_box_floor::evaluate(&context, &carrier, &fresh, paid, &mut guard).unwrap();
        let lo = result.consume(&context).unwrap();
        assert!(lo > offset - 0.0001 && lo <= offset);
        assert!(lo > old.0 + 0.99);
        assert_eq!(lo > 0., offset > 0.);
        let stale = head_box_floor::evaluate(&context, &carrier, &fresh, paid, &mut guard).unwrap();
        let actual = context.run(&carrier, &engine).unwrap();
        assert!(stale.consume(&context).is_none());
        let merged = context
            .base
            .merge_backend_lower(
                actual,
                &ny_core::GpuCrownResult {
                    lower_bounds: vec![old.0],
                    upper_bounds: vec![old.1],
                },
            )
            .unwrap();
        assert!(merged.lower_bounds[0] <= offset - 0.99);
        assert!(head_box_floor::evaluate(&context, &carrier, &fresh, paid, &mut guard).is_none());
        let current = context.build(&fresh).unwrap();
        let mut bad = current;
        bad.bounds[0] = 1.;
        assert!(head_box_floor::evaluate(&context, &bad, &fresh, paid, &mut guard).is_none());
    }
}

fn head_box_frontier_fixture(
    offset: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("parent", Layer::ReLU(ReLULayer)));
    let mut w = Array2::<f32>::zeros((9, 2));
    w[[6, 0]] = 1.;
    w[[6, 1]] = 1.;
    w[[7, 1]] = -1.;
    let mut bias = ndarray::Array1::<f32>::zeros(9);
    bias[6] = 1.;
    bias[7] = 3.;
    g.add_node(GraphNode::new(
        "pre",
        Layer::Linear(LinearLayer::new(w, Some(bias)).unwrap()),
        vec!["parent".into()],
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    let mut a = Array2::<f32>::zeros((1, 9));
    a[[0, 6]] = 1.;
    a[[0, 7]] = 1.;
    for j in 0..6 {
        a[[0, j]] = -2.;
    }
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(a, Some(arr1(&[offset - 4.]))).unwrap()),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1., -1.]).into_dyn(), arr1(&[2., 2.]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
fn head_box_frontier_quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> RawTwoLocalL1RootCutQuote<'a> {
    quote_raw_two_local_l1_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [3, 3],
        [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]],
        [[1., 1., 1., 0., 0., 0., 0., 0.]; 2],
        [1.; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn head_box_floor_integrated_original_query_closes_without_prefix_and_preserves_unsafe() {
    for offset in [0.5, 0., -0.5] {
        let (g, x, b) = head_box_frontier_fixture(offset);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        let mut q = head_box_frontier_quote(&g, &x, &b, old, end);
        assert!(q.try_enable_head_box_floor(0., limits()));
        assert_eq!(q.candidate_count(), 2);
        let engine = BundleBackend::default();
        let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
        let floor = out.head_box_candidate_lower().unwrap();
        assert!(floor <= offset && floor > offset - 0.0001);
        assert!(out.bounds().0 > old.0 + 0.99 && out.bounds().0 <= offset);
        assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
        assert_eq!(out.certificates(), 2);
        assert_eq!(out.support_calls(), 0);
        assert_eq!(out.single3_candidate_lower(), None);
        assert_eq!(out.cut_calls(), usize::from(offset <= 0.));
        assert_eq!(out.candidate_count(), 1 + usize::from(offset <= 0.));
        assert_eq!(out.bundle_candidate_lower().is_some(), offset <= 0.);
        assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), out.cut_calls());
        assert_eq!(out.bounds().0 > 0., offset > 0.);
        eprintln!(
            "head-box-integrated offset={offset} old={} floor={floor} result={} prefix_calls={}",
            old.0,
            out.bounds().0,
            out.cut_calls()
        );
    }
}

#[test]
fn head_box_floor_plan_prices_both_prefix_options_and_refuses_exact_resource_edges() {
    let (g, x, b) = full_head_fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let make = || {
        let mut q = comparison_quote(&g, &x, &b, old, end, 1.);
        assert!(q.try_enable_full_head_advice(limits()));
        q
    };
    let mut all = make();
    let original_work = all.work();
    let original_scratch = all.scratch_bytes();
    assert!(all.try_enable_head_box_floor(1., limits()));
    assert!(all.keep_single3_prefix());
    assert_eq!(all.candidate_count(), 3);
    let full_work = all.work();
    let scratch = all.scratch_bytes();
    let one_work =
        full_work - phase_work(all.base.full_quote, 1, 1).unwrap() - EXECUTION_OBSERVATION_WORK;
    assert_eq!(
        full_work,
        original_work + head_box_floor::quote(9, 10, 1).unwrap().work
    );
    assert_eq!(
        scratch,
        original_scratch + head_box_floor::quote(9, 10, 1).unwrap().scratch_bytes
    );
    for (cap, keep) in [(full_work, true), (full_work - 1, false), (one_work, false)] {
        let mut q = make();
        let mut allowed = limits();
        allowed.max_work = cap;
        allowed.max_scratch_bytes = scratch;
        assert!(q.try_enable_head_box_floor(1., allowed));
        assert_eq!(q.keep_single3_prefix(), keep);
        assert_eq!(q.work(), if keep { full_work } else { one_work });
        assert_eq!(q.candidate_count(), if keep { 3 } else { 2 });
        let engine = ComparisonBackend {
            allow_first_bundle: !keep,
            ..Default::default()
        };
        let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
        assert!(out.head_box_candidate_lower().unwrap() < 1.);
        assert_eq!(out.cut_calls(), if keep { 2 } else { 1 });
        assert_eq!(out.single3_candidate_lower().is_some(), keep);
        assert_eq!(out.candidate_count(), if keep { 3 } else { 2 });
        assert_eq!(engine.entered.load(Ordering::SeqCst), out.cut_calls());
        let lower = out
            .head_box_candidate_lower()
            .unwrap()
            .max(out.bundle_candidate_lower().unwrap());
        let lower = out
            .single3_candidate_lower()
            .map_or(lower, |v| v.max(lower));
        assert_eq!(out.bounds().0.to_bits(), old.0.max(lower).to_bits());
    }
    for (work, bytes) in [(one_work - 1, scratch), (full_work, scratch - 1)] {
        let mut q = make();
        let before = (q.work(), q.scratch_bytes());
        let mut allowed = limits();
        allowed.max_work = work;
        allowed.max_scratch_bytes = bytes;
        assert!(!q.try_enable_head_box_floor(1., allowed));
        assert_eq!((q.work(), q.scratch_bytes()), before);
        assert!(!q.uses_head_box_floor());
    }
    for threshold in [f32::NAN, f32::INFINITY, old.0 - 1., old.1] {
        let mut q = make();
        assert!(!q.try_enable_head_box_floor(threshold, limits()));
    }
    assert!(!all.try_enable_head_box_floor(1., limits()));
    assert!(!all.try_enable_full_head_advice(limits()));
}

#[test]
fn head_box_floor_declared_early_proof_skips_refusal_but_required_prefix_and_polls_are_atomic() {
    for offset in [0.5, 0., -0.5] {
        let (g, x, b) = head_box_frontier_fixture(offset);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        let make = || {
            let mut q = head_box_frontier_quote(&g, &x, &b, old, end);
            assert!(q.try_enable_head_box_floor(0., limits()));
            q
        };
        let engine = BundleBackend::default();
        engine.refuse_bundle.store(1, Ordering::SeqCst);
        let out = evaluate_raw_two_local_l1_root_cut(make(), &engine, &mut || true);
        assert_eq!(out.is_some(), offset > 0.);
        assert_eq!(
            engine.bundle_calls.load(Ordering::SeqCst),
            usize::from(offset <= 0.)
        );
        let mut polls = 0;
        assert!(
            evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
                polls += 1;
                true
            })
            .is_some()
        );
        for stop in 0..polls {
            let mut seen = 0;
            assert!(
                evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
                    let live = seen < stop;
                    seen += 1;
                    live
                })
                .is_none(),
                "offset={offset} poll={stop}"
            );
        }
        let mut late = make();
        late.base.deadline = Instant::now();
        assert!(
            evaluate_raw_two_local_l1_root_cut(late, &BundleBackend::default(), &mut || true)
                .is_none()
        );
    }
}
