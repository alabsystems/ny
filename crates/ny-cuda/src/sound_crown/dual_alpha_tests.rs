// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU exact-dyadic arithmetic controls for the production checked CUDA fold.
//! These bypass only the unchanged GPU size gate, never numerical guards.
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
        panic!("checked dual alpha must retain the f64 path")
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
            Err(NyError::DeadlineExceeded("dual alpha injected poll".into()))
        } else {
            Ok(())
        }
    }
}
fn dual(lower: Vec<f32>, chord: Vec<f32>, upper: Vec<f32>, intercept: Vec<f32>) -> GpuCrownLayer {
    let num_neurons = lower.len();
    GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope: lower,
        cross_slope: chord,
        upper_neg_slope: upper,
        cross_intercept: intercept,
        num_neurons,
    }
}
fn linear(w: &[f32], b: &[f32], o: usize, i: usize) -> GpuCrownLayer {
    GpuCrownLayer::Linear {
        weight: Arc::from(w),
        bias: Some(Arc::from(b)),
        out_features: o,
        in_features: i,
        cert_err: Default::default(),
    }
}
fn seed(rows: usize, width: usize, coefficients: &[f32]) -> GpuCrownSeed {
    GpuCrownSeed {
        lower_a: Arc::from(coefficients),
        upper_a: Arc::from(coefficients),
        lower_b: Arc::from(vec![0.; rows]),
        upper_b: Arc::from(vec![0.; rows]),
        current_dim: width,
        num_specs: rows,
    }
}
fn frontier(coefficient: f64, radius: f64) -> Frontier {
    Frontier {
        lower_a: vec![coefficient],
        upper_a: vec![coefficient],
        lower_err: vec![radius],
        upper_err: vec![radius],
        lb: vec![0.],
        ub: vec![0.],
        lb_err: vec![0.],
        ub_err: vec![0.],
        dim: 1,
    }
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn channel(value: f32, radius: f32) -> ResidentLowerCutChannel {
    ResidentLowerCutChannel::try_new(value, radius).unwrap()
}
fn joint(end: Instant, radius: f32) -> ResidentTwoPairLowerCutCarrier {
    ResidentTwoPairLowerCutCarrier::try_new(
        0,
        6,
        [[0, 1], [2, 3]],
        [1.; 2],
        [channel(-1., radius); 4],
        [channel(2., radius); 4],
        channel(-2., radius),
        end,
    )
    .unwrap()
}
fn isolator() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Instant) {
    // All five independent inputs participate. z1..4 are two diamonds,
    // neuron4 is w, and neuron5 is always-active u+s-3w+6 in [1,11].
    // F=9/4-sum|z|+3(ReLU(w)-w), whose exact minimum is 1/4.
    let s = vec![GpuResnetSegment::Chain(vec![
        linear(&[-2., -2., -2., -2., 3., 1.], &[-3.75], 1, 6),
        dual(
            vec![0., 0., 0., 0., 1., 1.],
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 1.],
            vec![0., 0., 0., 0., 0., 1.],
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.],
        ),
        linear(
            &[
                0.5, 0.5, 0., 0., 0., 0.5, -0.5, 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.5,
                -0.5, 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., -3.,
            ],
            &[0., 0., 0., 0., 0., 6.],
            6,
            5,
        ),
    ])];
    (
        s,
        seed(1, 1, &[1.]),
        Instant::now() + Duration::from_secs(30),
    )
}

#[test]
fn dual_alpha_preserves_asymmetric_branches_and_borrowed_buffers() {
    let layer = dual(
        vec![0.25, 0.25],
        vec![0.5, 0.5],
        vec![0.75, 0.75],
        vec![0.5, 0.5],
    );
    let view = ActivationView::from_layer(&layer).unwrap();
    let GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
        ..
    } = &layer
    else {
        unreachable!()
    };
    for (borrowed, original) in view.tables.iter().zip([
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
    ]) {
        assert_eq!(borrowed.as_ptr(), original.as_ptr());
    }
    let f = Frontier {
        lower_a: vec![2., -2.],
        upper_a: vec![2., -2.],
        lower_err: vec![0.; 2],
        upper_err: vec![0.; 2],
        lb: vec![0.],
        ub: vec![0.],
        lb_err: vec![0.],
        ub_err: vec![0.],
        dim: 2,
    };
    let out = backward_layers_f64(&Cpu::default(), &[layer], f, 1, &mut None).unwrap();
    assert_eq!(out.lower_a, [0.5, -1.]);
    assert_eq!(out.upper_a, [1., -1.5]);
    assert_eq!(out.lb, [-1.]);
    assert_eq!(out.ub, [1.]);
}

#[test]
fn dual_alpha_sign_crossing_errors_cover_both_minorants_and_chord_intercepts() {
    // Every product and endpoint is dyadic. Coefficient uncertainty crosses
    // zero, so choosing only the nominal slope or using alpha_L on the upper
    // path would miss a real branch. Alpha_U=1 is deliberately larger.
    for center in [-0.25, 0., 0.25] {
        let layer = dual(vec![0.125], vec![0.5], vec![1.], vec![0.5]);
        let f = backward_layers_f64(
            &Cpu::default(),
            &[layer],
            frontier(center, 0.5),
            1,
            &mut None,
        )
        .unwrap();
        for actual in [center - 0.5, 0., center + 0.5] {
            let lower = actual * if actual >= 0. { 0.125 } else { 0.5 };
            let upper = actual * if actual >= 0. { 0.5 } else { 1. };
            let li = if actual >= 0. { 0. } else { actual * 0.5 };
            let ui = if actual >= 0. { actual * 0.5 } else { 0. };
            assert!((lower - f.lower_a[0]).abs() <= f.lower_err[0]);
            assert!((upper - f.upper_a[0]).abs() <= f.upper_err[0]);
            assert!((li - f.lb[0]).abs() <= f.lb_err[0]);
            assert!((ui - f.ub[0]).abs() <= f.ub_err[0]);
        }
    }
}

#[test]
fn dual_alpha_symmetric_case_matches_ordinary_bits_and_wide_stays_unsupported() {
    let a = [0.25, -0.75, -0., 1.];
    let e = [0., 0.125, 0.25, 0.5];
    let lower = [0., 0.25, 0.5, 1.];
    let chord = [0.5, 0.75, 1., 1.];
    let ordinary = activation_step_f64(1, 4, &a, &a, &e, &e, &lower, &chord).unwrap();
    let asymmetric =
        activation_step_f64_branches(1, 4, &a, &a, &e, &e, &lower, &chord, Some(&lower)).unwrap();
    for (a, b) in [
        (&ordinary.0, &asymmetric.0),
        (&ordinary.1, &asymmetric.1),
        (&ordinary.2, &asymmetric.2),
        (&ordinary.3, &asymmetric.3),
    ] {
        assert_eq!(bits(a), bits(b));
    }
    let layer = dual(lower.to_vec(), chord.to_vec(), lower.to_vec(), vec![0.5; 4]);
    assert!(!layer_skeleton_matches(&layer, &layer));
    assert!(!validate_wide_layer_finite(&layer));
}

#[test]
fn dual_alpha_checked_support4_encloses_exact_relu_and_counts_beta_once() {
    let layers = vec![dual(
        vec![0.25, 1.],
        vec![0.5; 2],
        vec![0.75, 0.],
        vec![0.5; 2],
    )];
    let segments = vec![GpuResnetSegment::Chain(layers)];
    let coefficients = [1., 0., 0., 1., 1., 1., 1., -1.];
    let q = seed(4, 2, &coefficients);
    let end = Instant::now() + Duration::from_secs(30);
    let e = Cpu::default();
    let out = support4_impl(&e, &segments, &q, &[-1.; 2], &[1.; 2], end, false).unwrap();
    for x in [-1.0f32, 0., 1.] {
        for y in [-1.0f32, 0., 1.] {
            let values = [x.max(0.), y.max(0.)];
            for row in 0..4 {
                let exact =
                    coefficients[row * 2] * values[0] + coefficients[row * 2 + 1] * values[1];
                assert!(out.lower_bounds[row] <= exact && exact <= out.upper_bounds[row]);
            }
        }
    }
    assert_eq!(resident_activation_widths(&segments).unwrap(), [2]);
    let beta = vec![vec![0.; 2]];
    validate_deadline_bounded_beta_table(&e, &segments, &beta).unwrap();
    let mut aux = Some(ActAux::beta(&beta));
    resnet_fold_f64_core(&e, &segments, &q, &mut aux).unwrap();
    assert_eq!(aux.unwrap().cursor, 1);
    assert!(validate_deadline_bounded_beta_table(&e, &segments, &[]).is_err());
    assert!(validate_deadline_bounded_beta_table(&e, &segments, &[vec![0.; 1]]).is_err());
    assert!(matches!(
        support4(&e, &segments, &q, &[-1.; 2], &[1.; 2], end),
        Err(NyError::UnsupportedOp(_))
    ));
}

#[test]
fn dual_alpha_joint_isolator_retains_alpha_gain_and_upper_bits() {
    let (s, q, end) = isolator();
    let e = Cpu::default();
    let baseline = resnet_fold_f64_core(&e, &s, &q, &mut None).unwrap();
    let original = concretize_frontier(&baseline, 1, &[-1.; 5], &[1.; 5]);
    assert!(original.lower_bounds[0] <= -1.75 && original.lower_bounds[0] > -1.751);
    let carrier = joint(end, 0.);
    let mut aux = Some(ActAux::checked_cut(LowerCutRef::TwoPair(&carrier)));
    let folded = resnet_fold_f64_core(&e, &s, &q, &mut aux).unwrap();
    assert_eq!(aux.as_ref().unwrap().cursor, 1);
    assert!(aux.as_ref().unwrap().lower_cut_applied);
    for (before, after) in [
        (&baseline.upper_a, &folded.upper_a),
        (&baseline.upper_err, &folded.upper_err),
        (&baseline.ub, &folded.ub),
        (&baseline.ub_err, &folded.ub_err),
    ] {
        assert_eq!(bits(before), bits(after));
    }
    let lower = checked_two_pair(&e, &s, &q, &[-1.; 5], &[1.; 5], &carrier, end).unwrap();
    assert!(lower > 0.249 && lower <= 0.25);
    // A literal output shift makes a real unsafe point: u=s=w=1,v=t=0
    // attains -1/4. Retaining alpha must not manufacture a positive certificate.
    let mut unsafe_shift = s.clone();
    if let GpuResnetSegment::Chain(l) = &mut unsafe_shift[0] {
        l[0] = linear(&[-2., -2., -2., -2., 3., 1.], &[-4.25], 1, 6);
    }
    let shifted =
        checked_two_pair(&e, &unsafe_shift, &q, &[-1.; 5], &[1.; 5], &carrier, end).unwrap();
    assert!(shifted <= -0.25 && shifted > -0.251);
    // Explicitly recreate the old whole-layer heuristic conversion: only the
    // useful w alpha changes from1 to0; all ordinary chords stay the same.
    let mut heuristic = s.clone();
    if let GpuResnetSegment::Chain(l) = &mut heuristic[0] {
        l[1] = GpuCrownLayer::Activation {
            lower_slope: vec![0., 0., 0., 0., 0., 1.],
            upper_slope: vec![0.5, 0.5, 0.5, 0.5, 0.5, 1.],
            lower_intercept: vec![0.; 6],
            upper_intercept: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.],
            num_neurons: 6,
        };
    }
    let lost = checked_two_pair(&e, &heuristic, &q, &[-1.; 5], &[1.; 5], &carrier, end).unwrap();
    assert!(lost <= -2.75 && lost > -2.751);
    let single = ResidentLowerCutCarrier::try_new(
        0,
        6,
        [0, 1],
        vec![ny_core::ResidentLowerCutRow::try_new(
            vec![1.],
            [channel(-1., 0.); 2],
            [channel(2., 0.); 2],
            channel(-1., 0.),
        )
        .unwrap()],
        end,
    )
    .unwrap();
    let one = checked_fold(&e, &s, &q, &[-1.; 5], &[1.; 5], &single, end).unwrap();
    assert!(one <= -0.75 && one > -0.751);
    let charged =
        checked_two_pair(&e, &s, &q, &[-1.; 5], &[1.; 5], &joint(end, 1. / 128.), end).unwrap();
    assert!(charged < lower && charged <= 0.25 - 1. / 128.);
}

#[test]
fn dual_alpha_checked_validation_refuses_each_table_and_keeps_size_gate() {
    let (s, q, end) = isolator();
    for slot in 0..4 {
        for kind in 0..3 {
            let mut bad = s.clone();
            let GpuResnetSegment::Chain(l) = &mut bad[0] else {
                unreachable!()
            };
            let GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
                ..
            } = &mut l[1]
            else {
                unreachable!()
            };
            let tables = [
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
            ];
            let table = tables.into_iter().nth(slot).unwrap();
            match kind {
                0 => {
                    table.pop();
                }
                1 => table[0] = f32::NAN,
                _ => table[0] = ny_core::CROWN_COEFF_MAX,
            }
            let e = Cpu::default();
            assert!(
                checked_two_pair(&e, &bad, &q, &[-1.; 5], &[1.; 5], &joint(end, 0.), end).is_err()
            );
            assert_eq!(e.calls.load(Ordering::SeqCst), 0);
        }
    }
    let e = Cpu::default();
    assert!(matches!(
        run_two_pair(&e, &s, &q, &[-1.; 5], &[1.; 5], &joint(end, 0.), end),
        Err(NyError::UnsupportedOp(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn dual_alpha_checked_polls_refuse_every_boundary_including_completed_fold() {
    let (s, q, end) = isolator();
    let c = joint(end, 0.);
    let e = Cpu::default();
    checked_two_pair(&e, &s, &q, &[-1.; 5], &[1.; 5], &c, end).unwrap();
    let polls = e.polls.load(Ordering::SeqCst);
    let calls = e.calls.load(Ordering::SeqCst);
    assert!(calls > 0 && polls > 0);
    for index in 1..=polls {
        let e = Cpu::default();
        e.fail.store(index, Ordering::SeqCst);
        assert!(matches!(
            checked_two_pair(&e, &s, &q, &[-1.; 5], &[1.; 5], &c, end),
            Err(NyError::DeadlineExceeded(_))
        ));
        if index == polls {
            assert_eq!(e.calls.load(Ordering::SeqCst), calls);
        }
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("test clock supports an expired deadline");
    let e = Cpu::default();
    assert!(matches!(
        checked_two_pair(
            &e,
            &s,
            &q,
            &[-1.; 5],
            &[1.; 5],
            &joint(expired, 0.),
            expired
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    // The support-only surface has the same typed refusal guarantee, with a
    // dual activation genuinely inside the walked prefix.
    let s = vec![GpuResnetSegment::Chain(vec![dual(
        vec![0.25; 2],
        vec![0.5; 2],
        vec![0.75; 2],
        vec![0.5; 2],
    )])];
    let q = seed(4, 2, &[1., 0., 0., 1., 1., 1., 1., -1.]);
    let e = Cpu::default();
    support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], end, false).unwrap();
    for index in 1..=e.polls.load(Ordering::SeqCst) {
        let e = Cpu::default();
        e.fail.store(index, Ordering::SeqCst);
        assert!(matches!(
            support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], end, false),
            Err(NyError::DeadlineExceeded(_))
        ));
    }
}
