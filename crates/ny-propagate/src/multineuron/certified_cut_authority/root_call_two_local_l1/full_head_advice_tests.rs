// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent exact affine source and finite-search controls. These scores
//! confer no certificate or collector-engagement credit; the parent semantic
//! tests obtain fresh proofs and execute the genuine descriptor interpreter.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Signed;
use std::time::Duration;

const IDS: [[usize; 8]; 2] = [[0, 1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]];
const WEIGHTS: [[f32; 8]; 2] = [
    [1., 1., 1., 1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0., 0., 0., 0.],
];
fn rat(v: f32) -> Q {
    Q::from_float(f64::from(v)).unwrap()
}
fn r64(v: f64) -> Q {
    Q::from_float(v).unwrap()
}
fn zero() -> Q {
    Q::from_integer(0.into())
}
struct Data {
    matrix: Vec<f32>,
    bias: Vec<f32>,
    lower: Vec<f32>,
    upper: Vec<f32>,
    output: Vec<f32>,
    output_bias: Vec<f32>,
    objective: Vec<f32>,
    bn: Option<[Vec<f32>; 6]>,
}
impl Data {
    fn source(&self) -> Source<'_> {
        Source {
            matrix: &self.matrix,
            bias: Some(&self.bias),
            lower: &self.lower,
            upper: &self.upper,
            final_matrix: &self.output,
            final_bias: Some(&self.output_bias),
            objective: &self.objective,
            bn: self.bn.as_ref().map(|v| Bn {
                scale: &v[0],
                shift: &v[1],
                scale_error: &v[2],
                bias_error: &v[3],
                lower: &v[4],
                upper: &v[5],
            }),
        }
    }
    fn value(&self, x: &[f32; 9]) -> Q {
        let mut result = rat(self.output_bias[0]);
        for j in 0..10 {
            let mut z = rat(self.bias[j]);
            for k in 0..9 {
                z += rat(self.matrix[j * 9 + k]) * rat(x[k]);
            }
            result += rat(self.output[j]) * z.max(zero());
        }
        result
    }
}
fn fixture(shift: f32) -> (Data, GpuCrownLayer) {
    let mut matrix = vec![0.; 90];
    let mut bias = vec![0.; 10];
    for k in 0..3 {
        matrix[(2 * k) * 9 + 2 * k] = 0.5;
        matrix[(2 * k) * 9 + 2 * k + 1] = 0.5;
        matrix[(2 * k + 1) * 9 + 2 * k] = 0.5;
        matrix[(2 * k + 1) * 9 + 2 * k + 1] = -0.5;
        bias[2 * k] = 0.5;
        bias[2 * k + 1] = 0.5;
        matrix[9 * 9 + 2 * k] = 21. / 16.;
    }
    matrix[6 * 9 + 6] = 0.5;
    matrix[6 * 9 + 7] = 0.5;
    matrix[7 * 9 + 7] = 0.5;
    matrix[7 * 9 + 8] = 0.5;
    matrix[8 * 9 + 8] = 0.5;
    matrix[8 * 9 + 6] = -0.5;
    matrix[9 * 9 + 7] = 1.;
    matrix[9 * 9 + 8] = 1.;
    bias[9] = 6.;
    let mut output = vec![-2.; 10];
    output[9] = 1.;
    let mut upper = vec![0.75; 10];
    upper[6..9].fill(0.5);
    upper[9] = 1.;
    let mut intercept = vec![0.375; 10];
    intercept[6..9].fill(0.5);
    intercept[9] = 0.;
    (
        Data {
            matrix,
            bias,
            lower: vec![-1.; 9],
            upper: vec![1.; 9],
            output,
            output_bias: vec![21. / 4. + shift],
            objective: vec![1.],
            bn: None,
        },
        GpuCrownLayer::Activation {
            lower_slope: vec![1.; 10],
            upper_slope: upper,
            lower_intercept: vec![0.; 10],
            upper_intercept: intercept,
            num_neurons: 10,
        },
    )
}
fn with_proxy<T>(
    data: &Data,
    layer: &GpuCrownLayer,
    f: impl FnOnce(&Proxy<'_>, &mut Poll<'_>) -> T,
) -> T {
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let s = data.source();
    let p = contract(
        &s,
        Slopes::from_layer(layer, 10).unwrap(),
        quote(9, 10, 1).unwrap(),
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 1.],
        [6., 2.],
        &mut guard,
    )
    .unwrap();
    f(&p, &mut guard)
}
fn exact_group_support(d: &Data, ids: &[usize]) -> Q {
    let mut best = zero();
    // Fix first sign positive; retain both support endpoints. All arithmetic
    // is rational over the original source and its original nine-input box.
    for sign in 0..(1usize << (ids.len() - 1)) {
        let mut b = zero();
        let mut row = vec![zero(); 9];
        for (i, &j) in ids.iter().enumerate() {
            let s = if i == 0 || (sign >> (i - 1)) & 1 == 0 {
                rat(1.)
            } else {
                rat(-1.)
            };
            b += &s * rat(d.bias[j]);
            for k in 0..9 {
                row[k] += &s * rat(d.matrix[j * 9 + k]);
            }
        }
        let radius = row.into_iter().map(|v| v.abs()).fold(zero(), |a, b| a + b);
        best = best.max(b.abs() + radius);
    }
    best
}

#[test]
fn full_head_advice_asymmetric_source_has_interior_gain_and_unsafe_witness() {
    for shift in [0., -23. / 16.] {
        let (data, layer) = fixture(shift);
        assert_eq!(exact_group_support(&data, &IDS[0][..6]), rat(6.));
        assert_eq!(exact_group_support(&data, &IDS[1][..3]), rat(2.));
        let witness = [1., 0., 1., 0., 1., 0., 1., 1., 1.];
        assert_eq!(data.value(&witness), rat(19. / 16. + shift));
        with_proxy(&data, &layer, |p, guard| {
            for theta in [0., 1. / 16., 3. / 8., 1.] {
                let exact = rat(0.25 + shift) - rat(1.5) * (rat(theta) - rat(3. / 8.)).abs();
                assert_eq!(r64(p.score([theta, 1.], false, guard).unwrap()), exact);
            }
            assert_eq!(
                r64(p.score([1., 1.], true, guard).unwrap()),
                rat(-11. / 16. + shift)
            );
            let chosen = search(p, guard).unwrap();
            assert!(chosen.multipliers[0] >= 1. / 16. && chosen.multipliers[0] <= 1.);
            assert!(p.in_branch(chosen.multipliers));
            assert!(chosen.chosen_score > f64::from(0.24 + shift));
            assert!(chosen.chosen_score <= f64::from(0.25 + shift));
            assert!(chosen.chosen_score > chosen.original_score);
            assert!(chosen.score_evaluations <= 17 && chosen.full_scans <= 17);
            assert_eq!(chosen.full_scans, chosen.score_evaluations);
            assert!((chosen.multipliers[1] - 1.).abs() < 1e-6);
            if shift < 0. {
                assert!(chosen.chosen_score < 0.);
                assert!(data.value(&witness) < zero());
            }
        });
    }
    // The already qualified K=2 source has a stronger unshifted cycle and
    // two distinct objective backgrounds. Joint search must retain BOTH old
    // one-variable gains while keeping the cycle at its original multiplier.
    for row in 0..2 {
        for shift in [0., -23. / 16.] {
            let (mut data, layer) = fixture(shift);
            data.output[6..9].fill(-4.);
            data.bias[9] = 8.;
            data.matrix[88] = 2.;
            data.matrix[89] = 2.;
            let (optimum, peak) = if row == 0 {
                (3. / 8., 0.25)
            } else {
                data.output[9] = 65. / 64.;
                data.output_bias[0] -= 1. / 1024.;
                (171. / 512., 319. / 1024.)
            };
            let mut callback = || true;
            let mut g = Poll {
                end: Instant::now() + Duration::from_secs(30),
                callback: &mut callback,
                alive: true,
            };
            let source = data.source();
            let p = contract(
                &source,
                Slopes::from_layer(&layer, 10).unwrap(),
                quote(9, 10, 1).unwrap(),
                [6, 3],
                IDS,
                WEIGHTS,
                [1., 2.],
                [6., 2.],
                &mut g,
            )
            .unwrap();
            assert_eq!(
                r64(p.score([1., 2.], true, &mut g).unwrap()),
                rat(-11. / 16. + shift)
            );
            for theta in [1. / 16., optimum, 1.] {
                let exact = rat(peak + shift) - rat(1.5) * (rat(theta) - rat(optimum)).abs();
                assert_eq!(r64(p.score([theta, 2.], false, &mut g).unwrap()), exact);
            }
            let out = search(&p, &mut g).unwrap();
            assert!((out.multipliers[0] - optimum).abs() < 1e-5);
            assert!((out.multipliers[1] - 2.).abs() < 1e-5);
            assert!(out.chosen_score > f64::from(peak + shift - 0.01));
            assert!(out.chosen_score <= f64::from(peak + shift));
            if shift < 0. {
                assert!(out.chosen_score < 0.);
            }
        }
    }
}

#[test]
fn full_head_advice_background_cancellation_and_actual_branch_endpoint_are_distinct() {
    let (mut data, mut layer) = fixture(0.);
    let full = with_proxy(&data, &layer, |p, g| search(p, g).unwrap());
    // Omitting the stable row's input dependence changes the source proxy and
    // chooses the old endpoint. This is precisely the selected-only failure.
    data.matrix[81..90].fill(0.);
    let omitted = with_proxy(&data, &layer, |p, g| search(p, g).unwrap());
    assert_eq!(omitted.multipliers[0].to_bits(), 1f32.to_bits());
    assert!(full.multipliers[0] < 0.5);

    let (mut data, _) = fixture(0.);
    data.output[..6].fill(-1.);
    let GpuCrownLayer::Activation {
        lower_slope,
        lower_intercept,
        ..
    } = &mut layer
    else {
        panic!()
    };
    lower_slope[..6].fill(0.25);
    lower_intercept[..6].fill(-0.125);
    let ordinary = with_proxy(&data, &layer, |p, g| {
        assert_eq!(p.branch_hi[0], 0.5);
        assert!(!p.in_branch([1., 1.]));
        assert_ne!(
            p.score([1., 1.], true, g).unwrap(),
            p.score([1., 1.], false, g).unwrap()
        );
        let out = search(p, g).unwrap();
        assert!(out.multipliers[0] == 1. || p.in_branch(out.multipliers));
        out.original_score
    });
    let GpuCrownLayer::Activation {
        lower_slope,
        upper_slope,
        upper_intercept,
        ..
    } = &layer
    else {
        panic!()
    };
    let dual = GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope: lower_slope.clone(),
        cross_slope: upper_slope.clone(),
        upper_neg_slope: vec![0.875; 10],
        cross_intercept: upper_intercept.clone(),
        num_neurons: 10,
    };
    let dual_score = with_proxy(&data, &dual, |p, g| p.score([1., 1.], true, g).unwrap());
    // Six positive original post-coefficients of1: ordinary intercept−1/8
    // contributes−3/4; dual lower-positive is through the origin. Its third
    // slope belongs to the upper path and must never become this intercept.
    assert_eq!(r64(dual_score - ordinary), rat(0.75));
}

#[test]
fn full_head_advice_negative_raw_bn_and_absolute_radius_are_scored_from_source() {
    let (mut data, mut layer) = fixture(0.);
    let mut scale = vec![-1.; 10];
    scale[9] = 1.;
    let mut lower = vec![-0.5; 10];
    lower[6..9].fill(-1.);
    lower[9] = 1. / 16.;
    let mut upper = vec![1.5; 10];
    upper[6..9].fill(1.);
    upper[9] = 191. / 16.;
    data.bn = Some([
        scale,
        vec![0.; 10],
        vec![0.; 10],
        vec![0.; 10],
        lower,
        upper,
    ]);
    let GpuCrownLayer::Activation { upper_slope, .. } = &mut layer else {
        panic!()
    };
    upper_slope[..6].fill(0.25);
    let exact = with_proxy(&data, &layer, |p, g| {
        p.score([3. / 8., 1.], false, g).unwrap()
    });
    data.bn.as_mut().unwrap()[2][..6].fill(1. / 64.);
    data.bn.as_mut().unwrap()[3][..6].fill(1. / 128.);
    with_proxy(&data, &layer, |p, g| {
        assert_eq!(p.rows[0].scale, -1.);
        assert_eq!(r64(p.rows[0].radius), rat(1. / 32.));
        let charged = p.score([3. / 8., 1.], false, g).unwrap();
        // q=−1/2−theta/2=−11/16 on each of six selected BN rows.
        assert_eq!(r64(exact - charged), rat(33. / 256.));
        let result = search(p, g).unwrap();
        assert!(result.chosen_score >= result.original_score);
    });
}

#[test]
fn full_head_advice_quote_storage_finite_and_deadline_guards_are_bounded() {
    let q = quote(32768, 512, 10).unwrap();
    assert_eq!(q.work, 3_045_976_064);
    assert_eq!(q.scratch_bytes, 1_155_072);
    assert!(scratch_actual(32768, 512).unwrap() <= q.scratch_bytes);
    assert!(size_of::<Row>() <= 80);
    for width in 2..=8 {
        let small = quote(9, width, 1).unwrap();
        assert!(scratch_actual(9, width).unwrap() <= small.scratch_bytes);
    }
    for dims in [
        (0, 10, 1),
        (65537, 10, 1),
        (9, 1, 1),
        (9, 8193, 1),
        (9, 10, 0),
        (9, 8192, 129),
        (usize::MAX, 10, 1),
    ] {
        assert!(quote(dims.0, dims.1, dims.2).is_none());
    }
    let (mut data, layer) = fixture(0.);
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let mut paid = quote(9, 10, 1).unwrap();
    paid.work -= 1;
    assert!(contract(
        &data.source(),
        Slopes::from_layer(&layer, 10).unwrap(),
        paid,
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 1.],
        [6., 2.],
        &mut guard
    )
    .is_none());
    paid = quote(9, 10, 1).unwrap();
    data.matrix[89] = f32::NAN;
    assert!(contract(
        &data.source(),
        Slopes::from_layer(&layer, 10).unwrap(),
        paid,
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 1.],
        [6., 2.],
        &mut guard
    )
    .is_none());
    data.matrix[89] = 1.;
    data.lower[0] = 2.;
    assert!(contract(
        &data.source(),
        Slopes::from_layer(&layer, 10).unwrap(),
        paid,
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 1.],
        [6., 2.],
        &mut guard
    )
    .is_none());
    let (data, layer) = fixture(0.);
    with_proxy(&data, &layer, |p, g| {
        g.end = Instant::now();
        assert!(search(p, g).is_none());
    });
    with_proxy(&data, &layer, |p, _| {
        let mut calls = 0;
        let mut callback = || {
            calls += 1;
            calls < 5
        };
        let mut guard = Poll {
            end: Instant::now() + Duration::from_secs(30),
            callback: &mut callback,
            alive: true,
        };
        // The original score succeeds, then an entered joint value/gradient scan refuses.
        // The already scored original must not salvage the failed transaction.
        assert!(search(p, &mut guard).is_none());
        assert!(!guard.alive);
    });
    let (mut tiny, layer) = fixture(0.);
    tiny.output[..6].fill(-1. / 64.);
    with_proxy(&tiny, &layer, |p, g| {
        let out = search(p, g).unwrap();
        assert_eq!(out.multipliers[0].to_bits(), 1f32.to_bits());
        assert_eq!(out.score_evaluations, 1);
        assert_eq!(out.full_scans, 1);
        assert_eq!(out.original_score, out.chosen_score);
    });

    // A non-dyadic breakpoint from actual stored coefficient and Q8 weight:
    // accept the last representable point on the branch, refuse its successor.
    let (mut data, layer) = fixture(0.);
    data.output[..6].fill(-1. / 3.);
    let mut weights = WEIGHTS;
    weights[0][..6].fill(255. / 256.);
    let s = data.source();
    let p = contract(
        &s,
        Slopes::from_layer(&layer, 10).unwrap(),
        paid,
        [6, 3],
        IDS,
        weights,
        [1., 1.],
        [6. * 255. / 256., 2.],
        &mut guard,
    )
    .unwrap();
    let threshold = -f64::from(data.output[0]) / (2. * f64::from(weights[0][0]));
    let mut last = threshold as f32;
    if f64::from(last) > threshold {
        last = ny_tensor::next_down_f32(last);
    }
    assert!(p.in_branch([last, 1.]));
    assert!(!p.in_branch([next_up_f32(last), 1.]));
    let out = search(&p, &mut guard).unwrap();
    assert!(out.multipliers[0] == 1. || p.in_branch(out.multipliers));

    // Finite flat scores deterministically keep the original endpoint. This
    // exact constant-source edge exercises only the arithmetic helper; it does
    // not claim that the crossing-row owner would select a constant neuron.
    let (mut data, mut layer) = fixture(0.);
    data.matrix.fill(0.);
    data.bias[..9].fill(0.);
    let GpuCrownLayer::Activation {
        lower_slope,
        upper_slope,
        upper_intercept,
        ..
    } = &mut layer
    else {
        panic!()
    };
    lower_slope[..9].fill(0.);
    upper_slope[..9].fill(0.);
    upper_intercept[..9].fill(0.);
    let s = data.source();
    let p = contract(
        &s,
        Slopes::from_layer(&layer, 10).unwrap(),
        paid,
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 1.],
        [0., 0.],
        &mut guard,
    )
    .unwrap();
    let out = search(&p, &mut guard).unwrap();
    assert_eq!(out.multipliers[0].to_bits(), 1f32.to_bits());
    assert_eq!(out.original_score, out.chosen_score);
    assert_eq!(out.chosen_score, 45. / 4.);
}

fn joint_fixture(shift: f32) -> (Data, GpuCrownLayer) {
    let (mut data, mut layer) = fixture(shift);
    data.bias[6..9].fill(0.5);
    data.bias[9] = 9.;
    data.matrix[9 * 9 + 7] = 2.5;
    data.matrix[9 * 9 + 8] = 2.5;
    data.output[6..9].fill(-4.);
    data.output_bias[0] = 39. / 4. + shift;
    let GpuCrownLayer::Activation {
        upper_slope,
        upper_intercept,
        ..
    } = &mut layer
    else {
        panic!()
    };
    upper_slope[6..9].fill(0.75);
    upper_intercept[6..9].fill(0.375);
    (data, layer)
}
fn with_joint_proxy<T>(
    data: &Data,
    layer: &GpuCrownLayer,
    f: impl FnOnce(&Proxy<'_>, &mut Poll<'_>) -> T,
) -> T {
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let source = data.source();
    let proxy = contract(
        &source,
        Slopes::from_layer(layer, 10).unwrap(),
        quote(9, 10, 1).unwrap(),
        [6, 3],
        IDS,
        WEIGHTS,
        [1., 2.],
        [6., 3.5],
        &mut guard,
    )
    .unwrap();
    f(&proxy, &mut guard)
}
#[test]
fn full_head_joint_kelley_shifted_cycle_requires_both_multipliers() {
    for shift in [0., -31. / 16.] {
        let (data, layer) = joint_fixture(shift);
        assert_eq!(exact_group_support(&data, &IDS[0][..6]), rat(6.));
        assert_eq!(exact_group_support(&data, &IDS[1][..3]), rat(3.5));
        let witness = [1.; 9];
        assert_eq!(data.value(&witness), rat(27. / 16. + shift));
        // Exact rational original-source corner census. Negative ReLU terms
        // and a positive stable row make the original objective concave.
        let minimum = (0..512)
            .map(|mask| {
                let x = std::array::from_fn(|k| if mask & (1 << k) == 0 { -1. } else { 1. });
                data.value(&x)
            })
            .min()
            .unwrap();
        assert_eq!(minimum, rat(27. / 16. + shift));
        let stable_radius = data.matrix[81..90]
            .iter()
            .map(|v| rat(*v).abs())
            .fold(zero(), |a, b| a + b);
        assert_eq!(rat(data.bias[9]) - &stable_radius, rat(1. / 16.));
        assert_eq!(rat(data.bias[9]) + &stable_radius, rat(287. / 16.));
        with_joint_proxy(&data, &layer, |p, g| {
            for x in [0., 1. / 16., 3. / 8., 1.] {
                for y in [0., 1. / 16., 0.5, 1.] {
                    let exact = rat(0.75 + shift)
                        - rat(y)
                        - rat(1.5) * (rat(x) - rat(3. / 8.)).abs()
                        - rat(2.) * (rat(y) - rat(0.5)).abs();
                    assert_eq!(r64(p.score([x, 2. * y], false, g).unwrap()), exact);
                }
            }
            assert_eq!(
                r64(p.score([1., 2.], true, g).unwrap()),
                rat(-35. / 16. + shift)
            );
            assert_eq!(
                r64(p.score([0., 2.], false, g).unwrap()),
                rat(-29. / 16. + shift)
            );
            // The ENTIRE fixed-cycle line is bounded above by−5/4+shift.
            assert_eq!(
                r64(p.score([3. / 8., 2.], false, g).unwrap()),
                rat(-1.25 + shift)
            );
            let out = search(p, g).unwrap();
            assert!(p.in_branch(out.multipliers));
            assert!((out.multipliers[0] - 3. / 8.).abs() < 1e-5);
            assert!((out.multipliers[1] - 1.).abs() < 1e-5);
            assert!(out.chosen_score > f64::from(0.24 + shift));
            assert!(out.chosen_score <= f64::from(0.25 + shift));
            assert!(out.chosen_score > out.original_score);
            assert!(out.score_evaluations <= 17 && out.full_scans == out.score_evaluations);
            assert!(out.support_planes <= 16 && out.lp_candidates <= MAX_LP_CANDIDATES);
            if shift < 0. {
                assert!(out.chosen_score < 0.);
                assert!(data.value(&witness) < zero());
            }
        });
    }
}

fn analytic_proxy<'a>(
    lo: &'a [f32],
    hi: &'a [f32],
    u: Vec<f64>,
    v: [Vec<f64>; 2],
    beta0: f64,
    beta: [f64; 2],
) -> Proxy<'a> {
    let original = (0..u.len()).map(|k| u[k] + v[0][k] + v[1][k]).collect();
    Proxy {
        lower: lo,
        upper: hi,
        u,
        v,
        original,
        rows: vec![],
        beta0,
        beta,
        original_beta: beta0 + beta[0] + beta[1],
        lambda: [1.; 2],
        branch_hi: [1.; 2],
        selected: [[0; 8]; 2],
        active: [0; 2],
    }
}
#[test]
fn full_head_joint_kelley_uses_joint_kink_supergradients_and_coupled_steps() {
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let mut p = analytic_proxy(&[-1.], &[1.], vec![0.], [vec![1.], vec![-1.]], 0., [0.; 2]);
    let point = [0.5, 0.5];
    let evaluation = p.evaluate(point, false, &mut guard).unwrap();
    assert_eq!(evaluation.gradient, [-1., 1.]);
    let plane = Plane::sample([0.5, 0.5], evaluation).unwrap();
    for x in [1. / 16., 0.5, 1.] {
        for y in [1. / 16., 0.5, 1.] {
            assert!(
                p.score([x, y], false, &mut guard).unwrap()
                    <= plane.value([f64::from(x), f64::from(y)]).unwrap()
            );
        }
    }
    // Independently taking both left partials would produce(1,1), which is
    // NOT a joint supergradient at the diagonal: its plane fails at(.25,.25).
    let wrong_plane = Plane::sample(
        [f64::from(point[0]), f64::from(point[1])],
        Evaluation {
            gradient: [1.; 2],
            ..evaluation
        },
    )
    .unwrap();
    assert!(p.score([0.25; 2], false, &mut guard).unwrap() > wrong_plane.value([0.25; 2]).unwrap());
    p.u[0] = 0.;
    p.v[0][0] = 0.;
    p.v[1][0] = 0.;
    p.original[0] = 0.;
    p.rows.push(Row {
        q: [1., -1.],
        radius: 1.,
        ..Row::default()
    });
    assert_eq!(
        p.evaluate(point, false, &mut guard).unwrap().gradient,
        [-1., 1.]
    );

    // A single coordinate sweep started at(1,1) is stuck for this coupled
    // concave function: either coordinate alone decreases the score. Joint
    // Kelley moves along the diagonal to the lower corner.
    let p = analytic_proxy(
        &[-1.],
        &[1.],
        vec![0.],
        [vec![1.], vec![-1.]],
        0.,
        [0., -0.25],
    );
    let out = search(&p, &mut guard).unwrap();
    assert_eq!(out.multipliers, [1. / 16.; 2]);
    assert_eq!(out.original_score, -0.25);
    assert_eq!(out.chosen_score, -1. / 64.);

    // The optimum is the intersection of two non-axis-aligned kink lines.
    let p = analytic_proxy(
        &[-1., -1.],
        &[1., 1.],
        vec![-1., -0.25],
        [vec![1., 1.], vec![1., -1.]],
        0.25,
        [0.; 2],
    );
    let out = search(&p, &mut guard).unwrap();
    assert!((out.multipliers[0] - 0.625).abs() < 1e-6);
    assert!((out.multipliers[1] - 0.375).abs() < 1e-6);
    assert!(out.chosen_score > 0.24999 && out.chosen_score <= 0.25);
}

#[test]
fn full_head_joint_kelley_tiny_lp_corners_degeneracy_and_work_are_bounded() {
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let rect = Rectangle {
        low: [1. / 16.; 2],
        high: [1.; 2],
    };
    let mut e = Envelope::new();
    // All planes parallel: no valid determinant is needed to reach a corner.
    for i in 0..15 {
        e.push(Plane {
            slope: [1., -2.],
            intercept: i as f64,
        })
        .unwrap();
    }
    let mut visits = 0;
    assert_eq!(
        e.propose(rect, &mut visits, &mut guard).unwrap(),
        [1., 1. / 16.]
    );
    assert_eq!(visits, 879);
    let mut polls = 0;
    let mut cancel = || {
        polls += 1;
        polls < 2
    };
    let mut cancelled = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut cancel,
        alive: true,
    };
    let mut charged_visits = 0;
    assert!(e
        .propose(rect, &mut charged_visits, &mut cancelled)
        .is_none());
    assert_eq!(charged_visits, 32);
    assert!(!cancelled.alive);
    assert!(plane_difference(Plane::default(), Plane::default()).is_none());
    assert!(intersection([1., 1., 1.], [1., 1. + f64::EPSILON, 2.]).is_none());
    assert!(intersection([1., 0., 1.], [0., 1., 0.5]).is_some());
    assert!(plane_difference(
        Plane {
            slope: [f64::MAX, 0.],
            intercept: 0.
        },
        Plane {
            slope: [-f64::MAX, 0.],
            intercept: 0.
        }
    )
    .is_none());
    assert!(intersection([1., 0., f64::INFINITY], [0., 1., 0.5]).is_none());
    let mut invalid = Envelope::new();
    invalid
        .push(Plane {
            slope: [f64::INFINITY, 0.],
            intercept: 0.,
        })
        .unwrap();
    assert!(invalid.propose(rect, &mut 0, &mut guard).is_none());
    let mut flat = Envelope::new();
    flat.push(Plane::default()).unwrap();
    assert_eq!(flat.propose(rect, &mut 0, &mut guard).unwrap(), rect.low);
    // Enumerate the FULL declared15-solve schedule, even when repeated points
    // would normally stop search early. All geometric slots are counted.
    let mut cumulative = 0;
    let mut checks = 0;
    let mut all = Envelope::new();
    for m in 1..16 {
        all.push(Plane {
            slope: [m as f64, -1.],
            intercept: 0.,
        })
        .unwrap();
        let before = cumulative;
        all.propose(rect, &mut cumulative, &mut guard).unwrap();
        checks += (cumulative - before) * m;
    }
    assert_eq!(cumulative, 4120);
    assert_eq!(checks, 49_732);
    assert_eq!(MAX_LP_CANDIDATES, 4120);
    assert!(128 * MAX_LP_CANDIDATES + 64 * checks < (KELLEY_WORK as usize));
    assert_eq!(quote(32768, 512, 10).unwrap().work, 3_045_976_064);
    // Flat finite scores keep the original pair, and repeated stored-f32
    // corners terminate before exhausting the predeclared point budget.
    let p = analytic_proxy(&[-1.], &[1.], vec![0.], [vec![0.], vec![0.]], 2., [0.; 2]);
    let out = search(&p, &mut guard).unwrap();
    assert_eq!(out.multipliers, [1.; 2]);
    assert_eq!(out.chosen_score, 2.);
    assert!(out.support_planes <= 2 && out.score_evaluations <= 3);
}

#[test]
fn full_head_joint_kelley_both_branch_caps_preserve_original_descriptor_reference() {
    let (mut data, mut layer) = joint_fixture(0.);
    data.output[..6].fill(-1. / 3.);
    data.output[6..9].fill(-2. / 3.);
    let GpuCrownLayer::Activation {
        lower_slope,
        lower_intercept,
        ..
    } = &mut layer
    else {
        panic!()
    };
    lower_slope[..9].fill(0.25);
    lower_intercept[..9].fill(-0.125);
    let mut weights = WEIGHTS;
    weights[0][..6].fill(255. / 256.);
    weights[1][..3].fill(255. / 256.);
    let mut callback = || true;
    let mut guard = Poll {
        end: Instant::now() + Duration::from_secs(30),
        callback: &mut callback,
        alive: true,
    };
    let source = data.source();
    let p = contract(
        &source,
        Slopes::from_layer(&layer, 10).unwrap(),
        quote(9, 10, 1).unwrap(),
        [6, 3],
        IDS,
        weights,
        [1., 2.],
        [6., 3.5],
        &mut guard,
    )
    .unwrap();
    assert!(!p.in_branch([1., 2.]));
    let mut cap = [0.; 2];
    for g in 0..2 {
        let j = IDS[g][0];
        let exact = -f64::from(data.output[j]) / (2. * f64::from(weights[g][0]));
        cap[g] = exact as f32;
        if f64::from(cap[g]) > exact {
            cap[g] = ny_tensor::next_down_f32(cap[g]);
        }
    }
    assert!(p.in_branch(cap));
    for g in 0..2 {
        let mut over = cap;
        over[g] = next_up_f32(over[g]);
        assert!(!p.in_branch(over));
    }
    assert_ne!(
        p.score([1., 2.], true, &mut guard).unwrap(),
        p.score([1., 2.], false, &mut guard).unwrap()
    );
    let out = search(&p, &mut guard).unwrap();
    assert!(same_pair(out.multipliers, [1., 2.]) || p.in_branch(out.multipliers));
    assert!(out.chosen_score >= out.original_score);
    // Empty second branch: original pair remains a declared candidate; no
    // coordinate pass may hold this out-of-branch cycle scale inside a proxy.
    data.output[6..9].fill(-1. / 64.);
    let source = data.source();
    let p = contract(
        &source,
        Slopes::from_layer(&layer, 10).unwrap(),
        quote(9, 10, 1).unwrap(),
        [6, 3],
        IDS,
        weights,
        [1., 2.],
        [6., 3.5],
        &mut guard,
    )
    .unwrap();
    let out = search(&p, &mut guard).unwrap();
    assert_eq!(out.multipliers, [1., 2.]);
    assert_eq!(out.score_evaluations, 1);
    assert_eq!(out.full_scans, 1);
    assert_eq!(out.support_planes, 0);
    assert_eq!(out.lp_candidates, 0);
}
