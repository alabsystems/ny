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
        panic!("local L1 must use full-K f64")
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
                "local L1 injected transient poll".into(),
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
    // Four independent diamonds z=(u+v,u-v)/2, plus stable sum(u)+5.
    // F=17/4-sum|z|, so exact minimum is 1/4+shift. Every input matters.
    let mut weight = vec![0.; 9 * 8];
    for pair in 0..4 {
        for side in 0..2 {
            let row = 2 * pair + side;
            weight[row * 8 + 2 * pair] = 0.5;
            weight[row * 8 + 2 * pair + 1] = if side == 0 { 0.5 } else { -0.5 };
        }
        weight[8 * 8 + 2 * pair] = 1.;
    }
    let mut bias = vec![0.; 9];
    bias[8] = 5.;
    let mut final_weight = vec![-2.; 9];
    final_weight[8] = 1.;
    let mut slopes = vec![0.5; 9];
    slopes[8] = 1.;
    let mut intercept = vec![0.5; 9];
    intercept[8] = 0.;
    (
        vec![GpuResnetSegment::Chain(vec![
            linear(&final_weight, &[-0.75 + shift], 1, 9),
            activation(slopes.clone(), slopes, intercept),
            linear(&weight, &bias, 9, 8),
        ])],
        seed(1, 1, &[1.]),
        Instant::now() + Duration::from_secs(120),
    )
}
fn carrier(width: usize, t: Instant, bias_error: f32) -> ResidentLocalL1LowerCutCarrier {
    ResidentLocalL1LowerCutCarrier::try_new(
        0,
        width,
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        1.,
        [ch(-1., 0.); 8],
        [ch(2., 0.); 8],
        ch(-4., bias_error),
        t,
    )
    .unwrap()
}

#[test]
fn local_l1_eight_coordinates_close_after_every_two_pair_half_full_candidate_fails() {
    let (s, q, t) = diamonds(0.);
    let e = Cpu::default();
    let old = resnet_backward_f64_core(&e, &s, &q, &[-1.; 8], &[1.; 8], &mut None).unwrap();
    assert!(old.lower_bounds[0] <= -3.75 && old.lower_bounds[0] > -3.751);
    let mut count = 0;
    let mut best = f32::NEG_INFINITY;
    for a in 0..8 {
        for b in a + 1..8 {
            for c in a + 1..8 {
                for d in c + 1..8 {
                    if b == c || b == d {
                        continue;
                    }
                    // Every four-index matching occurs once, ordered by its first pair.
                    // A genuine diamond pair has absolute support 1; independent cells 2.
                    let b0 = if a / 2 == b / 2 { 1. } else { 2. };
                    let b1 = if c / 2 == d / 2 { 1. } else { 2. };
                    for lambda in [0.5, 1.] {
                        let c = ResidentTwoPairLowerCutCarrier::try_new(
                            0,
                            9,
                            [[a, b], [c, d]],
                            [lambda; 2],
                            [ch(-lambda, 0.); 4],
                            [ch(2. * lambda, 0.); 4],
                            ch(-lambda * (b0 + b1), 0.),
                            t,
                        )
                        .unwrap();
                        let v = checked_two_pair(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t).unwrap();
                        assert!(v <= -1.75);
                        best = best.max(v);
                        count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(count, 420);
    assert!(best > -1.751);
    // The exact local relation sum |z|<=4 is proved above, not obtained from
    // any support4 entry. This test exercises arithmetic, not root custody.
    let c = carrier(9, t, 0.);
    let bound = checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t).unwrap();
    assert!(bound <= 0.25 && bound > 0.249);
    let (unsafe_s, _, _) = diamonds(-0.5);
    let unsafe_bound = checked_local_l1(&e, &unsafe_s, &q, &[-1.; 8], &[1.; 8], &c, t).unwrap();
    assert!(unsafe_bound <= -0.25 && unsafe_bound > -0.251);
    assert!(e.calls.load(Ordering::SeqCst) > 0);
}

#[test]
fn local_l1_all_sixteen_channel_radii_and_one_bias_are_charged() {
    let (s, q, t) = diamonds(0.);
    let eps = 1. / 128.;
    for position in 0..8 {
        for after in [false, true] {
            let mut point = [0.; 8];
            for pair in 0..4 {
                point[2 * pair] = 1.;
            }
            point[(position / 2) * 2 + 1] = if position % 2 == 0 { 1. } else { -1. };
            let mut z = [0.; 8];
            for pair in 0..4 {
                z[2 * pair] = f32::midpoint(point[2 * pair], point[2 * pair + 1]);
                z[2 * pair + 1] = (point[2 * pair] - point[2 * pair + 1]) / 2.;
            }
            assert_eq!(z[position], 1.);
            let mut slopes: Vec<f32> = z.iter().map(|&v| if v > 0. { 1. } else { 0. }).collect();
            slopes.push(1.);
            let mut point_s = s.clone();
            if let GpuResnetSegment::Chain(l) = &mut point_s[0] {
                l[1] = activation(slopes.clone(), slopes, vec![0.; 9]);
            }
            let mut pre = [ch(-1., 0.); 8];
            let mut post = [ch(2., 0.); 8];
            if after {
                post[position] = ch(2., eps);
            } else {
                pre[position] = ch(-1., eps);
            }
            let c = ResidentLocalL1LowerCutCarrier::try_new(
                0,
                9,
                8,
                [0, 1, 2, 3, 4, 5, 6, 7],
                1.,
                pre,
                post,
                ch(-4., 0.),
                t,
            )
            .unwrap();
            let v = checked_local_l1(&Cpu::default(), &point_s, &q, &point, &point, &c, t).unwrap();
            let conservative = 0.25 - if after { 2. * eps } else { eps };
            assert!(v <= 0.25 - eps && v > conservative - 0.001);
        }
    }
    let c = carrier(9, t, 0.125);
    let v = checked_local_l1(&Cpu::default(), &s, &q, &[-1.; 8], &[1.; 8], &c, t).unwrap();
    assert!(v <= 0.125 && v > 0.124);
}

#[test]
fn local_l1_active_lengths_sparse_order_upper_bits_and_cursor_are_preserved() {
    assert!(size_of::<Option<LowerCutRef<'_>>>() <= 3 * size_of::<usize>());
    assert!(size_of::<ResidentLocalL1LowerCutCarrier>() <= 320);
    let t = Instant::now() + Duration::from_secs(60);
    for active in 1..=8 {
        let mut indices = [0; 8];
        let mut pre = [ch(0., 0.); 8];
        let mut post = pre;
        for i in 0..active {
            indices[i] = 2 * i + 1;
            pre[i] = ch(-(i as f32 + 1.), 0.);
            post[i] = ch(2. * (i as f32 + 1.), 0.);
        }
        let c = ResidentLocalL1LowerCutCarrier::try_new(
            0,
            17,
            active,
            indices,
            1.,
            pre,
            post,
            ch(-3., 0.),
            t,
        )
        .unwrap();
        let arm = LowerCutRef::LocalL1(&c);
        let mut a = [0.; 17];
        let mut e = [0.; 17];
        arm.apply_coefficients(&mut a, &mut e, 1, 17, true).unwrap();
        for i in 0..17 {
            let expected = if i % 2 == 1 && i / 2 < active {
                2. * (i / 2 + 1) as f64
            } else {
                0.
            };
            assert_eq!(a[i], expected);
        }
        arm.apply_coefficients(&mut a, &mut e, 1, 17, false)
            .unwrap();
        for i in 0..17 {
            let expected = if i % 2 == 1 && i / 2 < active {
                (i / 2 + 1) as f64
            } else {
                0.
            };
            assert_eq!(a[i], expected);
        }
        let mut b = [0.];
        let mut be = [0.];
        arm.apply_bias(&mut b, &mut be).unwrap();
        assert_eq!(b, [-3.]);
        assert!(arm
            .apply_coefficients(&mut a, &mut e, 2, 17, false)
            .is_err());
        assert!(arm
            .apply_coefficients(&mut a, &mut e, 1, 16, false)
            .is_err());
        assert!(arm.apply_bias(&mut [0.; 2], &mut [0.; 2]).is_err());
    }
    let (s, q, t) = diamonds(0.);
    let e = Cpu::default();
    let before = resnet_fold_f64_core(&e, &s, &q, &mut None).unwrap();
    let c = carrier(9, t, 0.);
    let mut aux = Some(ActAux::checked_cut(LowerCutRef::LocalL1(&c)));
    let after = resnet_fold_f64_core(&e, &s, &q, &mut aux).unwrap();
    for (a, b) in [
        (&before.upper_a, &after.upper_a),
        (&before.upper_err, &after.upper_err),
        (&before.ub, &after.ub),
        (&before.ub_err, &after.ub_err),
    ] {
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let done = aux.unwrap();
    assert!(done.lower_cut_applied);
    assert_eq!(done.cursor, 1);
}

#[test]
fn local_l1_retains_asymmetric_dual_alpha_and_charges_model_uncertainty() {
    // Four diamonds plus independent w and stable sum(u)-3w+8.
    // F=17/4-sum|z|+3(ReLU(w)-w). Keeping alpha_L(w)=1 is essential.
    let mut weight = vec![0.; 10 * 9];
    for pair in 0..4 {
        for side in 0..2 {
            let row = 2 * pair + side;
            weight[row * 9 + 2 * pair] = 0.5;
            weight[row * 9 + 2 * pair + 1] = if side == 0 { 0.5 } else { -0.5 };
        }
        weight[9 * 9 + 2 * pair] = 1.;
    }
    weight[8 * 9 + 8] = 1.;
    weight[9 * 9 + 8] = -3.;
    let mut bias = vec![0.; 10];
    bias[9] = 8.;
    let mut final_weight = vec![-2.; 10];
    final_weight[8] = 3.;
    final_weight[9] = 1.;
    let mut lower = vec![0.5; 10];
    lower[8] = 1.;
    lower[9] = 1.;
    let mut upper = vec![0.5; 10];
    upper[8] = 0.;
    upper[9] = 1.;
    let mut cross = vec![0.5; 10];
    cross[9] = 1.;
    let mut intercept = vec![0.5; 10];
    intercept[9] = 0.;
    let mut s = vec![GpuResnetSegment::Chain(vec![
        linear(&final_weight, &[-3.75], 1, 10),
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: lower,
            upper_neg_slope: upper,
            cross_slope: cross.clone(),
            cross_intercept: intercept.clone(),
            num_neurons: 10,
        },
        linear(&weight, &bias, 10, 9),
    ])];
    let q = seed(1, 1, &[1.]);
    let t = Instant::now() + Duration::from_secs(60);
    let c = carrier(10, t, 0.);
    let v = checked_local_l1(&Cpu::default(), &s, &q, &[-1.; 9], &[1.; 9], &c, t).unwrap();
    assert!(v <= 0.25 && v > 0.249);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.weight_rel_err = 1. / 1024.;
            cert_err.bias_abs_err = 1. / 64.;
        }
    }
    let charged = checked_local_l1(&Cpu::default(), &s, &q, &[-1.; 9], &[1.; 9], &c, t).unwrap();
    // Let T=sum|z|<=4, U=sum(u)<=4, H=ReLU(w)-w>=0. The
    // independent output parameter-ball minimum is .25-20rho-delta:
    // F-rho*sum|W_j|h_j-delta = 4.25-(1+rho)T-2rho*U
    //   +3(1-rho)H-8rho-delta, attained at every u=1,v=0,w=0.
    let exact_min = 0.25 - 20. / 1024. - 1. / 64.;
    assert!(charged <= exact_min && charged > 0.15);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        l[0] = linear(&final_weight, &[-3.75], 1, 10);
        let mut ordinary = vec![0.; 10];
        ordinary[9] = 1.;
        l[1] = activation(ordinary, cross, intercept);
    }
    let lost = checked_local_l1(&Cpu::default(), &s, &q, &[-1.; 9], &[1.; 9], &c, t).unwrap();
    assert!(lost <= -2.75 && lost > -2.751);
}

#[test]
fn local_l1_refuses_unsafe_channels_sources_seeds_and_keeps_size_gate() {
    let (s, q, t) = diamonds(0.);
    let c = carrier(9, t, 0.);
    let e = Cpu::default();
    assert!(matches!(
        run_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t),
        Err(NyError::UnsupportedOp(_))
    ));
    for position in 0..17 {
        for error in [false, true] {
            let mut pre = [ch(-1., 0.); 8];
            let mut post = [ch(2., 0.); 8];
            let mut bias = ch(-4., 0.);
            let bad = if error {
                ch(0., f32::MAX)
            } else {
                ch(f32::MAX, 0.)
            };
            if position < 8 {
                pre[position] = bad;
            } else if position < 16 {
                post[position - 8] = bad;
            } else {
                bias = bad;
            }
            let bad_c = ResidentLocalL1LowerCutCarrier::try_new(
                0,
                9,
                8,
                [0, 1, 2, 3, 4, 5, 6, 7],
                1.,
                pre,
                post,
                bias,
                t,
            )
            .unwrap();
            assert!(matches!(
                checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &bad_c, t),
                Err(NyError::SoundnessRefusal(_))
            ));
        }
    }
    let mut bad_q = q.clone();
    bad_q.upper_a = Arc::from([2.]);
    assert!(matches!(
        checked_local_l1(&e, &s, &bad_q, &[-1.; 8], &[1.; 8], &c, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut bad_s = s.clone();
        if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
            l[0] = linear(&[bad; 9], &[0.], 1, 9);
        }
        assert!(checked_local_l1(&e, &bad_s, &q, &[-1.; 8], &[1.; 8], &c, t).is_err());
    }
    let mut bad_s = s.clone();
    if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.bias_abs_err = -0.125;
        }
    }
    assert!(checked_local_l1(&e, &bad_s, &q, &[-1.; 8], &[1.; 8], &c, t).is_err());
    assert!(checked_local_l1(&e, &s, &seed(2, 1, &[1., 1.]), &[-1.; 8], &[1.; 8], &c, t).is_err());
    assert!(checked_local_l1(&e, &s, &q, &[1.; 8], &[-1.; 8], &c, t).is_err());
    let wrong = ResidentLocalL1LowerCutCarrier::try_new(
        1,
        9,
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        1.,
        [ch(-1., 0.); 8],
        [ch(2., 0.); 8],
        ch(-4., 0.),
        t,
    )
    .unwrap();
    assert!(checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &wrong, t).is_err());
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn local_l1_every_transient_poll_final_barrier_and_expired_entry_refuse() {
    let (s, q, t) = diamonds(0.);
    let c = carrier(9, t, 0.);
    let success = Cpu::default();
    assert!(checked_local_l1(&success, &s, &q, &[-1.; 8], &[1.; 8], &c, t).is_ok());
    let count = success.polls.load(Ordering::SeqCst);
    assert!(count > 10);
    for fail in 1..=count {
        let e = Cpu::default();
        e.fail.store(fail, Ordering::SeqCst);
        assert!(
            matches!(
                checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t),
                Err(NyError::DeadlineExceeded(_))
            ),
            "poll {fail}/{count}"
        );
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let c = carrier(9, expired, 0.);
    let e = Cpu::default();
    assert!(matches!(
        checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code)]
fn local_l1_non_ieee_environment_refuses_before_any_contraction() {
    std::thread::spawn(|| {
        // Construct source/channels before changing the thread-local FP state.
        let (s, q, t) = diamonds(0.);
        let c = carrier(9, t, 0.);
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
            assert!(checked_local_l1(&e, &s, &q, &[-1.; 8], &[1.; 8], &c, t).is_err());
            assert_eq!(e.calls.load(Ordering::SeqCst), 0);
        }
        drop(restore);
    })
    .join()
    .unwrap();
}
