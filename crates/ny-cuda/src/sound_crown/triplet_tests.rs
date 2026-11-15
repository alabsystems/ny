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
        panic!("triplet must use full-K f64")
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
                "triplet injected transient poll".into(),
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
fn cycle() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Instant) {
    // z=((a-b)/2,(b-c)/2,(c-a)/2), a,b,c in [-1,1]. Thus sum z=0
    // and sum |z|=max(a,b,c)-min(a,b,c)<=2. F=9/4-sum|z|.
    (
        vec![GpuResnetSegment::Chain(vec![
            linear(&[-2.; 3], &[2.25], 1, 3),
            activation(vec![0.5; 3], vec![0.5; 3], vec![0.5; 3]),
            linear(
                &[0.5, -0.5, 0., 0., 0.5, -0.5, -0.5, 0., 0.5],
                &[0.; 3],
                3,
                3,
            ),
        ])],
        seed(1, 1, &[1.]),
        Instant::now() + Duration::from_secs(60),
    )
}
fn carrier(width: usize, deadline: Instant, bias_error: f32) -> ResidentTripletLowerCutCarrier {
    ResidentTripletLowerCutCarrier::try_new(
        0,
        width,
        [0, 1, 2],
        1.,
        [ch(-1., 0.); 3],
        [ch(2., 0.); 3],
        ch(-2., bias_error),
        deadline,
    )
    .unwrap()
}

#[test]
fn triplet_cycle_closes_after_all_single_pair_half_full_candidates_fail() {
    let (s, q, t) = cycle();
    let e = Cpu::default();
    let ordinary = resnet_backward_f64_core(&e, &s, &q, &[-1.; 3], &[1.; 3], &mut None).unwrap();
    assert!(ordinary.lower_bounds[0] <= -0.75 && ordinary.lower_bounds[0] > -0.751);
    for first in 0..3 {
        for second in first + 1..3 {
            for lambda in [0.5, 1.] {
                // Each pair has the reachable vertex (+1,-1), so its exact absolute
                // support is 2. The corresponding complete scalar lower remains -3/4.
                let c = ResidentLowerCutCarrier::try_new(
                    0,
                    3,
                    [first, second],
                    vec![ny_core::ResidentLowerCutRow::try_new(
                        vec![lambda],
                        [ch(-lambda, 0.); 2],
                        [ch(2. * lambda, 0.); 2],
                        ch(-2. * lambda, 0.),
                    )
                    .unwrap()],
                    t,
                )
                .unwrap();
                let v = checked_fold(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, t).unwrap();
                assert!(v <= -0.75 && v > -0.751);
            }
        }
    }
    let GpuResnetSegment::Chain(layers) = &s[0] else {
        unreachable!()
    };
    let prefix = vec![GpuResnetSegment::Chain(vec![layers[2].clone()])];
    let support_seed = seed(4, 3, &[1., 1., 1., 1., 1., -1., 1., -1., 1., -1., 1., 1.]);
    let support = support4_impl(&e, &prefix, &support_seed, &[-1.; 3], &[1.; 3], t, false).unwrap();
    for row in 0..4 {
        let exact = if row == 0 { 0. } else { 2. };
        assert!(support.lower_bounds[row] <= -exact && support.lower_bounds[row] > -exact - 0.001);
        assert!(support.upper_bounds[row] >= exact && support.upper_bounds[row] < exact + 0.001);
    }
    // Fixed exact algebra above supplies this numeric test's relation. The
    // production semantic owner separately constructs its certificate from
    // the actual outward support; this backend test grants no such authority.
    let c = carrier(3, t, 0.);
    let v = checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, t).unwrap();
    assert!(v <= 0.25 && v > 0.249);
    let (mut unsafe_s, _, _) = cycle();
    if let GpuResnetSegment::Chain(l) = &mut unsafe_s[0] {
        l[0] = linear(&[-2.; 3], &[1.75], 1, 3);
    }
    let unsafe_v = checked_triplet(&e, &unsafe_s, &q, &[-1.; 3], &[1.; 3], &c, t).unwrap();
    assert!(unsafe_v <= -0.25 && unsafe_v > -0.251);
    assert!(e.calls.load(Ordering::SeqCst) > 0);
}

#[test]
fn triplet_each_channel_radius_and_single_bias_are_charged_at_exact_points() {
    let epsilon = 1. / 128.;
    let (s, q, t) = cycle();
    // Rotate (+1,-1,-1): selected neuron is +1; other cells are 0 and -1.
    // Stable exact point relaxations remove unrelated box slack that could
    // conceal an omitted channel error. The nominal Lagrangian is 1/4.
    for position in 0..3 {
        for post in [false, true] {
            let mut point = [-1.; 3];
            point[position] = 1.;
            let z = [
                (point[0] - point[1]) / 2.,
                (point[1] - point[2]) / 2.,
                (point[2] - point[0]) / 2.,
            ];
            assert_eq!(z[position], 1.);
            let mut point_s = s.clone();
            let slopes: Vec<f32> = z.iter().map(|&v| if v > 0. { 1. } else { 0. }).collect();
            if let GpuResnetSegment::Chain(l) = &mut point_s[0] {
                l[1] = activation(slopes.clone(), slopes, vec![0.; 3]);
            }
            let mut pre = [ch(-1., 0.); 3];
            let mut after = [ch(2., 0.); 3];
            if post {
                after[position] = ch(2., epsilon);
            } else {
                pre[position] = ch(-1., epsilon);
            }
            let c = ResidentTripletLowerCutCarrier::try_new(
                0,
                3,
                [0, 1, 2],
                1.,
                pre,
                after,
                ch(-2., 0.),
                t,
            )
            .unwrap();
            let v = checked_triplet(&Cpu::default(), &point_s, &q, &point, &point, &c, t).unwrap();
            let exact = 0.25 - epsilon;
            let conservative = 0.25 - if post { 2. * epsilon } else { epsilon };
            assert!(v <= exact && v > conservative - 0.001);
        }
    }
    let c = carrier(3, t, 0.125);
    let v = checked_triplet(&Cpu::default(), &s, &q, &[-1.; 3], &[1.; 3], &c, t).unwrap();
    assert!(v <= 0.125 && v > 0.124);
}

#[test]
fn triplet_sparse_order_preserves_upper_frontier_and_fixed_storage() {
    assert!(size_of::<Option<LowerCutRef<'_>>>() <= 3 * size_of::<usize>());
    assert!(
        size_of::<ResidentTripletLowerCutCarrier>() <= size_of::<ResidentTwoPairLowerCutCarrier>()
    );
    let (s, q, t) = cycle();
    let ordered = ResidentTripletLowerCutCarrier::try_new(
        0,
        7,
        [5, 1, 3],
        1.,
        [ch(-1., 0.), ch(-2., 0.), ch(-3., 0.)],
        [ch(4., 0.), ch(5., 0.), ch(6., 0.)],
        ch(-7., 0.),
        t,
    )
    .unwrap();
    let arm = LowerCutRef::Triplet(&ordered);
    let mut a = [0.; 7];
    let mut errors = [0.; 7];
    arm.apply_coefficients(&mut a, &mut errors, 1, 7, true)
        .unwrap();
    assert_eq!(a, [0., 5., 0., 6., 0., 4., 0.]);
    arm.apply_coefficients(&mut a, &mut errors, 1, 7, false)
        .unwrap();
    assert_eq!(a, [0., 3., 0., 3., 0., 3., 0.]);
    let mut bias = [0.];
    let mut bias_error = [0.];
    arm.apply_bias(&mut bias, &mut bias_error).unwrap();
    assert_eq!(bias, [-7.]);
    assert!(arm
        .apply_coefficients(&mut a, &mut errors, 2, 7, false)
        .is_err());
    assert!(arm.apply_bias(&mut [0.; 2], &mut [0.; 2]).is_err());
    let e = Cpu::default();
    let before = resnet_fold_f64_core(&e, &s, &q, &mut None).unwrap();
    let c = carrier(3, t, 0.);
    let mut aux = Some(ActAux::checked_cut(LowerCutRef::Triplet(&c)));
    let after = resnet_fold_f64_core(&e, &s, &q, &mut aux).unwrap();
    for (a, b) in [
        (&before.upper_a, &after.upper_a),
        (&before.upper_err, &after.upper_err),
        (&before.ub, &after.ub),
        (&before.ub_err, &after.ub_err),
    ] {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let completed = aux.unwrap();
    assert!(completed.lower_cut_applied);
    assert_eq!(completed.cursor, 1);
}

#[test]
fn triplet_retains_dual_alpha_and_declared_source_bias_uncertainty() {
    // Add independent w and stable -3w+4. True F is the cycle objective
    // plus 3(ReLU(w)-w), so its minimum remains 1/4.
    let mut s = vec![GpuResnetSegment::Chain(vec![
        linear(&[-2., -2., -2., 3., 1.], &[-1.75], 1, 5),
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: vec![0.5, 0.5, 0.5, 1., 1.],
            cross_slope: vec![0.5, 0.5, 0.5, 0.5, 1.],
            upper_neg_slope: vec![0.5, 0.5, 0.5, 0., 1.],
            cross_intercept: vec![0.5, 0.5, 0.5, 0.5, 0.],
            num_neurons: 5,
        },
        linear(
            &[
                0.5, -0.5, 0., 0., 0., 0.5, -0.5, 0., -0.5, 0., 0.5, 0., 0., 0., 0., 1., 0., 0.,
                0., -3.,
            ],
            &[0., 0., 0., 0., 4.],
            5,
            4,
        ),
    ])];
    let q = seed(1, 1, &[1.]);
    let t = Instant::now() + Duration::from_secs(60);
    let c = carrier(5, t, 0.);
    let preserved = checked_triplet(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
    assert!(preserved <= 0.25 && preserved > 0.249);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.bias_abs_err = 0.125;
        }
    }
    let charged = checked_triplet(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
    // This is model parameter uncertainty, independent of the carrier radius.
    assert!(charged <= 0.125 && charged > 0.124);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.weight_rel_err = 1. / 1024.;
            cert_err.bias_abs_err = 1. / 64.;
        }
    }
    let parameter_ball =
        checked_triplet(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
    // Let T=sum|z|<=2 and H=ReLU(w)-w>=0. The nominal output is
    // 9/4-T+3H and sum_j |W_j|h_j=T+3H+4. For rho<1 its exact
    // independent parameter-ball minimum is 1/4-6rho-delta, attained
    // at T=2,w=0. A missing relative-weight charge violates this ceiling.
    let exact_parameter_min = 0.25 - 6. / 1024. - 1. / 64.;
    assert!(parameter_ball <= exact_parameter_min && parameter_ball > 0.15);
    if let GpuResnetSegment::Chain(l) = &mut s[0] {
        l[0] = linear(&[-2., -2., -2., 3., 1.], &[-1.75], 1, 5);
        l[1] = activation(
            vec![0., 0., 0., 0., 1.],
            vec![0.5, 0.5, 0.5, 0.5, 1.],
            vec![0.5, 0.5, 0.5, 0.5, 0.],
        );
    }
    let rebuilt = checked_triplet(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
    assert!(rebuilt <= -2.75 && rebuilt > -2.751);
}

#[test]
fn triplet_rejects_bad_source_seed_transport_and_keeps_production_size_gate() {
    let (s, q, t) = cycle();
    let c = carrier(3, t, 0.);
    let e = Cpu::default();
    assert!(matches!(
        run_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, t),
        Err(NyError::UnsupportedOp(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    let zero = ResidentTripletLowerCutCarrier::try_new(
        0,
        3,
        [0, 1, 2],
        0.,
        [ch(-1., 0.); 3],
        [ch(2., 0.); 3],
        ch(-2., 0.),
        t,
    )
    .unwrap();
    assert!(matches!(
        checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &zero, t),
        Err(NyError::UnsupportedOp(_))
    ));
    for value_slot in 0..7 {
        let mut pre = [ch(-1., 0.); 3];
        let mut post = [ch(2., 0.); 3];
        let mut bias = ch(-2., 0.);
        let unsafe_channel = ch(f32::MAX, 0.);
        if value_slot < 3 {
            pre[value_slot] = unsafe_channel;
        } else if value_slot < 6 {
            post[value_slot - 3] = unsafe_channel;
        } else {
            bias = unsafe_channel;
        }
        let bad = ResidentTripletLowerCutCarrier::try_new(0, 3, [0, 1, 2], 1., pre, post, bias, t)
            .unwrap();
        assert!(matches!(
            checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &bad, t),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    let mut bad_q = q.clone();
    bad_q.upper_a = Arc::from([2.]);
    assert!(matches!(
        checked_triplet(&e, &s, &bad_q, &[-1.; 3], &[1.; 3], &c, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut bad_s = s.clone();
        if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
            l[2] = linear(
                &[bad, -0.5, 0., 0., 0.5, -0.5, -0.5, 0., 0.5],
                &[0.; 3],
                3,
                3,
            );
        }
        assert!(checked_triplet(&e, &bad_s, &q, &[-1.; 3], &[1.; 3], &c, t).is_err());
    }
    let mut bad_s = s.clone();
    if let GpuResnetSegment::Chain(l) = &mut bad_s[0] {
        if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
            cert_err.bias_abs_err = -0.125;
        }
    }
    assert!(checked_triplet(&e, &bad_s, &q, &[-1.; 3], &[1.; 3], &c, t).is_err());
    assert!(checked_triplet(&e, &s, &seed(2, 1, &[1., 1.]), &[-1.; 3], &[1.; 3], &c, t).is_err());
    assert!(checked_triplet(&e, &s, &q, &[1.; 3], &[-1.; 3], &c, t).is_err());
    let wrong = ResidentTripletLowerCutCarrier::try_new(
        1,
        3,
        [0, 1, 2],
        1.,
        [ch(-1., 0.); 3],
        [ch(2., 0.); 3],
        ch(-2., 0.),
        t,
    )
    .unwrap();
    assert!(checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &wrong, t).is_err());
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn triplet_every_transient_poll_final_barrier_and_expired_entry_refuse() {
    let (s, q, t) = cycle();
    let c = carrier(3, t, 0.);
    let successful = Cpu::default();
    assert!(checked_triplet(&successful, &s, &q, &[-1.; 3], &[1.; 3], &c, t).is_ok());
    let count = successful.polls.load(Ordering::SeqCst);
    assert!(count > 10);
    for fail in 1..=count {
        let e = Cpu::default();
        e.fail.store(fail, Ordering::SeqCst);
        assert!(
            matches!(
                checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, t),
                Err(NyError::DeadlineExceeded(_))
            ),
            "poll {fail}/{count}"
        );
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let c = carrier(3, expired, 0.);
    let e = Cpu::default();
    assert!(matches!(
        checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        checked_triplet(&e, &s, &q, &[-1.; 3], &[1.; 3], &c, t),
        Err(NyError::SoundnessRefusal(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}
