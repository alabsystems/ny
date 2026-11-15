// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact-dyadic controls for the production checked fold. Only its unchanged
//! GPU size gate is bypassed; these are not real-device or root-authority tests.
use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

#[derive(Default)]
struct Cpu {
    polls: AtomicUsize,
    fail: AtomicUsize,
    calls: AtomicUsize,
}
impl GemmEngine for Cpu {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("two-local L1 must use full-K f64")
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut out = vec![0.; m * n];
        for i in 0..m {
            for j in 0..n {
                for h in 0..k {
                    out[i * n + j] += a[i * k + h] * b[h * n + j];
                }
            }
        }
        Ok(out)
    }
    fn poll_crown_backward_deadline(&self) -> Result<()> {
        let count = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if count == self.fail.load(Ordering::SeqCst) {
            Err(NyError::DeadlineExceeded(
                "two-local L1 injected transient poll".into(),
            ))
        } else {
            Ok(())
        }
    }
}
fn ch(value: f32, error: f32) -> ResidentLowerCutChannel {
    ResidentLowerCutChannel::try_new(value, error).unwrap()
}
fn linear(weight: &[f32], bias: &[f32], output: usize, input: usize) -> GpuCrownLayer {
    GpuCrownLayer::Linear {
        weight: Arc::from(weight),
        bias: Some(Arc::from(bias)),
        out_features: output,
        in_features: input,
        cert_err: Default::default(),
    }
}
fn seed(rows: usize, width: usize, a: &[f32]) -> GpuCrownSeed {
    GpuCrownSeed {
        lower_a: Arc::from(a),
        upper_a: Arc::from(a),
        lower_b: Arc::from(vec![0.; rows]),
        upper_b: Arc::from(vec![0.; rows]),
        num_specs: rows,
        current_dim: width,
    }
}
fn activation(lower: Vec<f32>, upper: Vec<f32>, intercept: Vec<f32>) -> GpuCrownLayer {
    let num_neurons = lower.len();
    GpuCrownLayer::Activation {
        lower_slope: lower,
        upper_slope: upper,
        lower_intercept: vec![0.; num_neurons],
        upper_intercept: intercept,
        num_neurons,
    }
}
fn diamonds(shift: f32) -> (Vec<GpuResnetSegment>, GpuCrownSeed, Instant) {
    // Eight independent diamonds; t=sum(u)+17 is stable. F=33/4-sum16|z|.
    let mut w = vec![0.; 17 * 16];
    for pair in 0..8 {
        for side in 0..2 {
            let row = 2 * pair + side;
            w[row * 16 + 2 * pair] = 0.5;
            w[row * 16 + 2 * pair + 1] = if side == 0 { 0.5 } else { -0.5 };
        }
        w[16 * 16 + 2 * pair] = 1.;
    }
    let mut b = vec![0.; 17];
    b[16] = 17.;
    let mut a = vec![-2.; 17];
    a[16] = 1.;
    let mut slopes = vec![0.5; 17];
    slopes[16] = 1.;
    let mut intercept = vec![0.5; 17];
    intercept[16] = 0.;
    (
        vec![GpuResnetSegment::Chain(vec![
            linear(&a, &[-8.75 + shift], 1, 17),
            activation(slopes.clone(), slopes, intercept),
            linear(&w, &b, 17, 16),
        ])],
        seed(1, 1, &[1.]),
        Instant::now() + Duration::from_secs(120),
    )
}
fn carrier(width: usize, t: Instant, bias_error: f32) -> ResidentTwoLocalL1LowerCutCarrier {
    ResidentTwoLocalL1LowerCutCarrier::try_new(
        0,
        width,
        [8, 8],
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        [1., 1.],
        [[ch(-1., 0.); 8]; 2],
        [[ch(2., 0.); 8]; 2],
        ch(-8., bias_error),
        t,
    )
    .unwrap()
}
#[test]
fn two_local_l1_strict_sixteen_closes_after_every_best_single_group_half_full_fails() {
    let (s, q, t) = diamonds(0.);
    let e = Cpu::default();
    let old = resnet_backward_f64_core(&e, &s, &q, &[-1.; 16], &[1.; 16], &mut None).unwrap();
    assert!(old.lower_bounds[0] <= -7.75 && old.lower_bounds[0] > -7.751);
    // For any m<=8 chosen cells with p complete diamonds, exact support=m-p.
    // The unselected triangles cost16-m, hence bound=-31/4+p<=-15/4.
    // Exhaust the integer membership census, then run every maximizing group.
    let mut max_pairs = 0;
    for mask in 0u32..(1 << 16) {
        let m = mask.count_ones();
        if !(3..=8).contains(&m) {
            continue;
        }
        let p = (0..8).filter(|&k| (mask >> (2 * k)) & 3 == 3).count();
        assert!(p <= 4);
        max_pairs = max_pairs.max(p);
    }
    assert_eq!(max_pairs, 4);
    let mut calls = 0;
    let mut best = f32::NEG_INFINITY;
    for pairs in 0u32..256 {
        if pairs.count_ones() != 4 {
            continue;
        }
        let mut ids = [0; 8];
        let mut position = 0;
        for k in 0..8 {
            if pairs & (1 << k) != 0 {
                ids[position] = 2 * k;
                ids[position + 1] = 2 * k + 1;
                position += 2;
            }
        }
        for lambda in [0.5, 1.] {
            let c = ResidentLocalL1LowerCutCarrier::try_new(
                0,
                17,
                8,
                ids,
                lambda,
                [ch(-lambda, 0.); 8],
                [ch(2. * lambda, 0.); 8],
                ch(-4. * lambda, 0.),
                t,
            )
            .unwrap();
            let bound = checked_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t).unwrap();
            assert!(bound <= -3.75);
            best = best.max(bound);
            calls += 1;
        }
    }
    assert_eq!(calls, 140);
    assert!(best > -3.751);
    // These dyadic supports are established by original-source algebra above;
    // no support4 call or private source-custody claim is made by this CPU test.
    let c = carrier(17, t, 0.);
    let got = checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t).unwrap();
    assert!(got <= 0.25 && got > 0.249);
    let (unsafe_s, _, _) = diamonds(-0.5);
    let got = checked_two_local_l1(&e, &unsafe_s, &q, &[-1.; 16], &[1.; 16], &c, t).unwrap();
    assert!(got <= -0.25 && got > -0.251);
    assert!(e.calls.load(Ordering::SeqCst) > 0);
}
#[test]
fn two_local_l1_all_thirty_two_radii_and_one_aggregate_bias_are_charged() {
    let (s, q, t) = diamonds(0.);
    let eps = 1. / 128.;
    for position in 0..16 {
        for after in [false, true] {
            let mut point = [0.; 16];
            for k in 0..8 {
                point[2 * k] = 1.;
            }
            point[2 * (position / 2) + 1] = if position % 2 == 0 { 1. } else { -1. };
            let mut z = [0.; 16];
            for k in 0..8 {
                z[2 * k] = f32::midpoint(point[2 * k], point[2 * k + 1]);
                z[2 * k + 1] = (point[2 * k] - point[2 * k + 1]) / 2.;
            }
            assert_eq!(z[position], 1.);
            let mut slopes: Vec<f32> = z.iter().map(|&x| if x > 0. { 1. } else { 0. }).collect();
            slopes.push(1.);
            let mut point_s = s.clone();
            if let GpuResnetSegment::Chain(l) = &mut point_s[0] {
                l[1] = activation(slopes.clone(), slopes, vec![0.; 17]);
            }
            let mut pre = [[ch(-1., 0.); 8]; 2];
            let mut post = [[ch(2., 0.); 8]; 2];
            if after {
                post[position / 8][position % 8] = ch(2., eps);
            } else {
                pre[position / 8][position % 8] = ch(-1., eps);
            }
            let base = carrier(17, t, 0.);
            let c = ResidentTwoLocalL1LowerCutCarrier::try_new(
                0,
                17,
                [8, 8],
                *base.ordered_neurons(),
                [1., 1.],
                pre,
                post,
                ch(-8., 0.),
                t,
            )
            .unwrap();
            let got =
                checked_two_local_l1(&Cpu::default(), &point_s, &q, &point, &point, &c, t).unwrap();
            let conservative = 0.25 - if after { 2. * eps } else { eps };
            assert!(got <= 0.25 - eps && got > conservative - 0.001);
        }
    }
    let c = carrier(17, t, 0.125);
    let got = checked_two_local_l1(&Cpu::default(), &s, &q, &[-1.; 16], &[1.; 16], &c, t).unwrap();
    assert!(got <= 0.125 && got > 0.124);
}
#[test]
fn two_local_l1_all_active_counts_sparse_slots_preserve_upper_and_one_cursor() {
    assert!(size_of::<Option<LowerCutRef<'_>>>() <= 3 * size_of::<usize>());
    assert!(size_of::<ResidentTwoLocalL1LowerCutCarrier>() <= 640);
    let t = Instant::now() + Duration::from_secs(60);
    for a in 1..=8 {
        for b in 1..=8 {
            let lens = [a, b];
            let mut ids = [[0; 8]; 2];
            let mut pre = [[ch(0., 0.); 8]; 2];
            let mut post = pre;
            for g in 0..2 {
                for i in 0..lens[g] {
                    ids[g][i] = 2 * i + g;
                    pre[g][i] = ch(-((g * 8 + i + 1) as f32), 0.);
                    post[g][i] = ch(2. * (g * 8 + i + 1) as f32, 0.);
                }
            }
            let c = ResidentTwoLocalL1LowerCutCarrier::try_new(
                0,
                17,
                lens,
                ids,
                [0.5, 2.],
                pre,
                post,
                ch(-3., 0.),
                t,
            )
            .unwrap();
            let arm = LowerCutRef::TwoLocalL1(&c);
            let mut coeff = [0.; 17];
            let mut err = [0.; 17];
            arm.apply_coefficients(&mut coeff, &mut err, 1, 17, true)
                .unwrap();
            for col in 0..17 {
                let g = col % 2;
                let i = col / 2;
                let expected = if i < lens[g] {
                    2. * (g * 8 + i + 1) as f64
                } else {
                    0.
                };
                assert_eq!(coeff[col], expected);
            }
            arm.apply_coefficients(&mut coeff, &mut err, 1, 17, false)
                .unwrap();
            for col in 0..17 {
                let g = col % 2;
                let i = col / 2;
                let expected = if i < lens[g] {
                    (g * 8 + i + 1) as f64
                } else {
                    0.
                };
                assert_eq!(coeff[col], expected);
            }
            let mut bias = [0.];
            let mut be = [0.];
            arm.apply_bias(&mut bias, &mut be).unwrap();
            assert_eq!(bias, [-3.]);
            assert!(arm
                .apply_coefficients(&mut coeff, &mut err, 2, 17, false)
                .is_err());
            assert!(arm.apply_bias(&mut [0.; 2], &mut [0.; 2]).is_err());
        }
    }
    let (s, q, t) = diamonds(0.);
    let e = Cpu::default();
    let original = resnet_fold_f64_core(&e, &s, &q, &mut None).unwrap();
    let c = carrier(17, t, 0.);
    let mut aux = Some(ActAux::checked_cut(LowerCutRef::TwoLocalL1(&c)));
    let cut = resnet_fold_f64_core(&e, &s, &q, &mut aux).unwrap();
    for (a, b) in [
        (&original.upper_a, &cut.upper_a),
        (&original.upper_err, &cut.upper_err),
        (&original.ub, &cut.ub),
        (&original.ub_err, &cut.ub_err),
    ] {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let done = aux.unwrap();
    assert!(done.lower_cut_applied);
    assert_eq!(done.cursor, 1);
}
#[test]
fn two_local_l1_distinct_scales_weights_and_outward_aggregate_bias_match_point_truth() {
    // Fixed original affine z: mixed signs isolate post/pre placement. Each
    // certificate is exact on this source; group scales and weights differ.
    let z = [-2., -1., 0.5, 1., 2., 3.];
    let w: [[f32; 3]; 2] = [[0.25, 0.5, 1.], [1., 0.5, 0.25]];
    let lambda = [f32::from_bits(0x3f800001), f32::from_bits(0x3eaaaaab)];
    let bounds = [1.5f32, 2.75];
    let exact_bias =
        -f64::from(lambda[0]) * f64::from(bounds[0]) - f64::from(lambda[1]) * f64::from(bounds[1]);
    // These two dyadic products and their sum fit in f64 for this test. The
    // backend receives the outward channel, not rounded product centers.
    let center = exact_bias as f32;
    let gap = (f64::from(center) - exact_bias).abs();
    let radius = ny_core::f64_to_f32_up(gap);
    assert!(gap > 0.);
    assert!(
        f64::from(center) - f64::from(radius) <= exact_bias
            && exact_bias <= f64::from(center) + f64::from(radius)
    );
    let mut pre = [[ch(0., 0.); 8]; 2];
    let mut post = pre;
    for g in 0..2 {
        for i in 0..3 {
            let exact = f64::from(lambda[g]) * f64::from(w[g][i]);
            let c = exact as f32;
            let e = ny_core::f64_to_f32_up((f64::from(c) - exact).abs());
            pre[g][i] = ch(-c, e);
            let twice = 2. * exact;
            let c = twice as f32;
            let e = ny_core::f64_to_f32_up((f64::from(c) - twice).abs());
            post[g][i] = ch(c, e);
        }
    }
    let t = Instant::now() + Duration::from_secs(60);
    let c = ResidentTwoLocalL1LowerCutCarrier::try_new(
        0,
        6,
        [3, 3],
        [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]],
        lambda,
        pre,
        post,
        ch(center, radius),
        t,
    )
    .unwrap();
    let slopes: Vec<f32> = z.iter().map(|x| if *x > 0. { 1. } else { 0. }).collect();
    let s = vec![GpuResnetSegment::Chain(vec![
        linear(&[0.; 6], &[0.25], 1, 6),
        activation(slopes.clone(), slopes, vec![0.; 6]),
        linear(&[0.; 6], &z, 6, 1),
    ])];
    let got =
        checked_two_local_l1(&Cpu::default(), &s, &seed(1, 1, &[1.]), &[0.], &[0.], &c, t).unwrap();
    assert!(got <= 0.25 && got > 0.249);
}
#[test]
fn two_local_l1_dual_alpha_and_output_parameter_uncertainty_remain_charged() {
    // Eight diamonds plus w and stable sum(u)-3w+12. F=33/4-T+3H.
    let mut w = vec![0.; 18 * 17];
    for k in 0..8 {
        for side in 0..2 {
            let row = 2 * k + side;
            w[row * 17 + 2 * k] = 0.5;
            w[row * 17 + 2 * k + 1] = if side == 0 { 0.5 } else { -0.5 };
        }
        w[17 * 17 + 2 * k] = 1.;
    }
    w[16 * 17 + 16] = 1.;
    w[17 * 17 + 16] = -3.;
    let mut bias = vec![0.; 18];
    bias[17] = 12.;
    let mut a = vec![-2.; 18];
    a[16] = 3.;
    a[17] = 1.;
    let mut lower = vec![0.5; 18];
    lower[16] = 1.;
    lower[17] = 1.;
    let mut upper = vec![0.5; 18];
    upper[16] = 0.;
    upper[17] = 1.;
    let mut cross = vec![0.5; 18];
    cross[17] = 1.;
    let mut intercept = vec![0.5; 18];
    intercept[17] = 0.;
    let mut s = vec![GpuResnetSegment::Chain(vec![
        linear(&a, &[-3.75], 1, 18),
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: lower,
            upper_neg_slope: upper,
            cross_slope: cross.clone(),
            cross_intercept: intercept.clone(),
            num_neurons: 18,
        },
        linear(&w, &bias, 18, 17),
    ])];
    let q = seed(1, 1, &[1.]);
    let t = Instant::now() + Duration::from_secs(60);
    let c = carrier(18, t, 0.);
    let got = checked_two_local_l1(&Cpu::default(), &s, &q, &[-1.; 17], &[1.; 17], &c, t).unwrap();
    assert!(got <= 0.25 && got > 0.249);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.weight_rel_err = 1. / 4096.;
            cert_err.bias_abs_err = 1. / 128.;
        }
    }
    let got = checked_two_local_l1(&Cpu::default(), &s, &q, &[-1.; 17], &[1.; 17], &c, t).unwrap();
    // Independent model family: T<=8,U=sum(u)<=8,H>=0 and rho<1.
    // F-rho*sum|W|h-delta =8.25-(1+rho)T-2rho*U+3(1-rho)H-12rho-delta.
    // Exact minimum .25-36rho-delta at u=1,v=0,w=0.
    let exact_min = 0.25 - 36. / 4096. - 1. / 128.;
    assert!(got <= exact_min && got > 0.1);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        l[0] = linear(&a, &[-3.75], 1, 18);
        let mut ordinary = vec![0.; 18];
        ordinary[17] = 1.;
        l[1] = activation(ordinary, cross, intercept);
    }
    let lost = checked_two_local_l1(&Cpu::default(), &s, &q, &[-1.; 17], &[1.; 17], &c, t).unwrap();
    assert!(lost <= -2.75 && lost > -2.751);
}
#[test]
fn two_local_l1_all_thirty_three_channels_and_source_guards_refuse_before_gemm() {
    let (s, q, t) = diamonds(0.);
    let base = carrier(17, t, 0.);
    let e = Cpu::default();
    assert!(matches!(
        run_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &base, t),
        Err(NyError::UnsupportedOp(_))
    ));
    for position in 0..33 {
        for error in [false, true] {
            let mut pre = *base.pre();
            let mut post = *base.post();
            let mut bias = base.bias();
            let bad = if error {
                ch(0., f32::MAX)
            } else {
                ch(f32::MAX, 0.)
            };
            if position < 16 {
                pre[position / 8][position % 8] = bad;
            } else if position < 32 {
                post[(position - 16) / 8][position % 8] = bad;
            } else {
                bias = bad;
            }
            let c = ResidentTwoLocalL1LowerCutCarrier::try_new(
                0,
                17,
                [8, 8],
                *base.ordered_neurons(),
                [1., 1.],
                pre,
                post,
                bias,
                t,
            )
            .unwrap();
            assert!(matches!(
                checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t),
                Err(NyError::SoundnessRefusal(_))
            ));
        }
    }
    let mut bad_q = q.clone();
    bad_q.upper_a = Arc::from([2.]);
    assert!(matches!(
        checked_two_local_l1(&e, &s, &bad_q, &[-1.; 16], &[1.; 16], &base, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut bad_s = s.clone();
        if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
            l[0] = linear(&[bad; 17], &[0.], 1, 17);
        }
        assert!(checked_two_local_l1(&e, &bad_s, &q, &[-1.; 16], &[1.; 16], &base, t).is_err());
    }
    let mut bad_s = s.clone();
    if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.bias_abs_err = -0.125;
        }
    }
    assert!(checked_two_local_l1(&e, &bad_s, &q, &[-1.; 16], &[1.; 16], &base, t).is_err());
    assert!(checked_two_local_l1(
        &e,
        &s,
        &seed(2, 1, &[1., 1.]),
        &[-1.; 16],
        &[1.; 16],
        &base,
        t
    )
    .is_err());
    assert!(checked_two_local_l1(&e, &s, &q, &[1.; 16], &[-1.; 16], &base, t).is_err());
    let wrong = ResidentTwoLocalL1LowerCutCarrier::try_new(
        1,
        17,
        [8, 8],
        *base.ordered_neurons(),
        [1., 1.],
        *base.pre(),
        *base.post(),
        base.bias(),
        t,
    )
    .unwrap();
    assert!(checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &wrong, t).is_err());
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}
#[test]
fn two_local_l1_every_transient_poll_final_barrier_and_expired_entry_refuse() {
    let (s, q, t) = diamonds(0.);
    let c = carrier(17, t, 0.);
    let success = Cpu::default();
    assert!(checked_two_local_l1(&success, &s, &q, &[-1.; 16], &[1.; 16], &c, t).is_ok());
    let count = success.polls.load(Ordering::SeqCst);
    assert!(count > 10);
    for fail in 1..=count {
        let e = Cpu::default();
        e.fail.store(fail, Ordering::SeqCst);
        assert!(
            matches!(
                checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t),
                Err(NyError::DeadlineExceeded(_))
            ),
            "poll {fail}/{count}"
        );
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let c = carrier(17, expired, 0.);
    let e = Cpu::default();
    assert!(matches!(
        checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code)]
fn two_local_l1_non_ieee_environment_refuses_before_any_contraction() {
    std::thread::spawn(|| {
        // Construct source/channels before changing the thread-local FP state.
        let (s, q, t) = diamonds(0.);
        let c = carrier(17, t, 0.);
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: restore the valid saved MXCSR on this same thread.
                unsafe {
                    std::arch::asm!("ldmxcsr [{}]",in(reg)&self.0,options(nostack,preserves_flags));
                }
            }
        }
        let mut previous = 0u32;
        // SAFETY: STMXCSR writes four bytes into the live aligned u32.
        unsafe {
            std::arch::asm!("stmxcsr [{}]",in(reg)&mut previous,options(nostack,preserves_flags));
        }
        let restore = Restore(previous);
        for bad in [
            previous | (1 << 6),
            previous | (1 << 15),
            (previous & !(3 << 13)) | (1 << 13),
        ] {
            // SAFETY: only supported DAZ/FTZ/rounding bits change; RAII restores
            // the original environment even if any assertion unwinds.
            unsafe {
                std::arch::asm!("ldmxcsr [{}]",in(reg)&bad,options(nostack,preserves_flags));
            }
            let e = Cpu::default();
            assert!(checked_two_local_l1(&e, &s, &q, &[-1.; 16], &[1.; 16], &c, t).is_err());
            assert_eq!(e.calls.load(Ordering::SeqCst), 0);
        }
        drop(restore);
    })
    .join()
    .unwrap();
}
