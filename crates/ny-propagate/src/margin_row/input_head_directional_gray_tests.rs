// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Gray-state containment against independent exact dyadic literal-mask forms.
use super::allocated_tests::{budget, candidate, end, source, spec, HEADS, INPUTS};
use super::*;

fn small() -> Source {
    let authored = InputHeadSpec::new(
        3,
        5,
        vec![
            0.25, 1.0, -0.5, 0.125, -0.25, -0.5, 0.25, 1.5, -0.25, 0.5, 0.125, -0.75, 0.375, 2.0,
            -1.25,
        ],
        vec![
            3.0 / 16.0,
            -5.0 / 32.0,
            7.0 / 64.0,
            -9.0 / 128.0,
            11.0 / 256.0,
        ],
        vec![1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        vec![0.5, 0.0],
        [134; 32],
    )
    .unwrap();
    let mut s = Source::build(&authored, &[-1.0; 3], &[1.0; 3], 0, 1, end()).unwrap();
    for h in 0..5 {
        let enlarged = (h + 1) as f64 / 128.0;
        assert!(enlarged > s.first_error[h]);
        s.first_error[h] = enlarged;
    }
    s
}

fn direction() -> Direction {
    Direction {
        heads: vec![(1, 0.25), (2, 0.5), (3, 1.0), (4, 1.25)],
    }
}

fn duplicate() -> Source {
    let authored = InputHeadSpec::new(
        1,
        2,
        vec![1.0, 1.0],
        vec![0.125; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![0.5, 0.0],
        [135; 32],
    )
    .unwrap();
    Source::build(&authored, &[-1.0], &[1.0], 0, 1, end()).unwrap()
}

fn duplicate_cut(s: &Source) -> CertifiedDirection<'_> {
    signed::certify_gray(
        s,
        Direction {
            heads: vec![(1, 1.0)],
        },
        signed::PositiveTerms(vec![(0, 1.0)]),
        vec![],
        &mut budget(),
    )
    .unwrap()
}

#[test]
fn gray_every_step_encloses_exact_literal_mask_with_nonzero_bias_errors_and_eta() {
    let s = small();
    let negative = direction();
    let eta = [(0, -0.375), (2, 0.625)];
    let mut seen = std::collections::BTreeSet::new();
    for step in 0..16usize {
        let mask = step ^ (step >> 1);
        let form = signed::gray_form(&s, &negative, &eta, step, &mut budget()).unwrap();
        let mut exact_b = 0i64;
        let mut exact_a = [-12_288i64, 0, 20_480]; // eta *32768
        let mut literal_b = I::p(0.0);
        let mut literal_a = [I::p(-0.375), I::p(0.0), I::p(0.625)];
        for (bit, &(h, tau)) in negative.heads.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                continue;
            }
            let tau_num = (4.0 * tau) as i64;
            let intercept = f64::from(s.spec.first_bias[h]) + s.first_error[h];
            let b_num = (8192.0 * intercept) as i64;
            assert_eq!(b_num as f64 / 8192.0, intercept);
            exact_b -= tau_num * b_num;
            literal_b = literal_b
                .add(
                    I::p(-tau)
                        .mul(
                            I::p(f64::from(s.spec.first_bias[h]))
                                .add(I::p(s.first_error[h]))
                                .unwrap(),
                        )
                        .unwrap(),
                )
                .unwrap();
            for j in 0..3 {
                let w = f64::from(s.spec.first[j * 5 + h]);
                let w_num = (8192.0 * w) as i64;
                assert_eq!(w_num as f64 / 8192.0, w);
                exact_a[j] -= tau_num * w_num;
                literal_a[j] = literal_a[j].add(I::p(-tau).mul(I::p(w)).unwrap()).unwrap();
            }
        }
        let exact_b = exact_b as f64 / 32768.0;
        assert!(form.b.lo <= exact_b && exact_b <= form.b.hi);
        assert!(literal_b.lo <= exact_b && exact_b <= literal_b.hi);
        for j in 0..3 {
            let exact = exact_a[j] as f64 / 32768.0;
            assert!(
                form.a[j].lo <= exact && exact <= form.a[j].hi,
                "Gray step{step} input{j} misses exact literal coefficient"
            );
            assert!(literal_a[j].lo <= exact && exact <= literal_a[j].hi);
            // Two independent outward evaluation orders need not have nested
            // interval widths. Both must contain the exact represented form.
            assert!(form.a[j].lo <= literal_a[j].hi && literal_a[j].lo <= form.a[j].hi);
        }
        assert!(seen.insert(mask));
    }
    assert_eq!(seen, (0..16).collect::<std::collections::BTreeSet<_>>());
}

#[test]
fn gray_final_visit_and_empty_visit_really_cover_all_256_algebraic_masks() {
    let authored = InputHeadSpec::new(
        1,
        9,
        std::iter::once(1.0)
            .chain((0..8).map(|bit| (1u32 << bit) as f32))
            .collect(),
        vec![0.0; 9],
        std::iter::once([1.0, 0.0])
            .chain((0..8).map(|_| [-1.0, 0.0]))
            .flatten()
            .collect(),
        vec![0.5, 0.0],
        [136; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[-1.0], &[1.0], 0, 1, end()).unwrap();
    let negative = Direction {
        heads: (1..9).map(|h| (h, 1.0)).collect(),
    };
    let mut actual_masks = std::collections::BTreeSet::new();
    for step in 0..256usize {
        let mask = step ^ (step >> 1);
        let form = signed::gray_form(&s, &negative, &[], step, &mut budget()).unwrap();
        // The actual affine state identifies its mask because weights are powers2.
        let exact = -(mask as f64);
        assert!(form.a[0].lo <= exact && exact <= form.a[0].hi);
        assert!(form.a[0].hi - form.a[0].lo < 1e-6);
        assert!(actual_masks.insert(mask));
    }
    assert_eq!(
        actual_masks,
        (0..256).collect::<std::collections::BTreeSet<_>>()
    );
    assert!(signed::gray_form(&s, &negative, &[], 256, &mut budget()).is_none());
}

#[test]
fn gray_replay_refuses_nonfinite_out_of_range_theta_without_a_certificate() {
    let s = small();
    let negative = direction();
    let positive = signed::PositiveTerms(vec![(0, 1.0)]);
    for theta in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -f64::EPSILON,
        1.0f64.next_up(),
    ] {
        assert!(
            signed::gray_theta(&s, &negative, &positive, &[], 5, theta, &mut budget()).is_none()
        );
    }
    for theta in [0.0, 0.25, 1.0] {
        assert!(
            signed::gray_theta(&s, &negative, &positive, &[], 5, theta, &mut budget()).is_some()
        );
    }
}

#[test]
fn gray_positive_and_negative_error_signs_both_reduce_geometry_floor() {
    let initial = duplicate_cut(&duplicate()).lower;
    for changed_head in [0, 1] {
        let mut s = duplicate();
        s.first_error[changed_head] += 0.25;
        let after = duplicate_cut(&s).lower;
        assert!(
            after < initial - 0.24,
            "head{changed_head}: {initial} -> {after}"
        );
    }
    let mut output_changed = duplicate();
    output_changed.bias = I::p(100.0);
    output_changed.output_error += 10.0;
    assert_eq!(
        duplicate_cut(&output_changed).lower.to_bits(),
        initial.to_bits()
    );
}

#[test]
fn gray_empty_mask_bounds_positive_support_and_cannot_be_zero_clamped() {
    let authored = InputHeadSpec::new(
        1,
        2,
        vec![0.0, 1.0],
        vec![-10.0, -2.0],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![0.0; 2],
        [137; 32],
    )
    .unwrap();
    let s = Source::build(&authored, &[1.0], &[3.0], 0, 1, end()).unwrap();
    let mut work = budget();
    let cut = signed::certify_gray(
        &s,
        Direction {
            heads: vec![(1, 1.0)],
        },
        signed::PositiveTerms(vec![(0, 1.0)]),
        vec![(0, 1.0)],
        &mut work,
    )
    .unwrap();
    assert_eq!(work.masks, 2);
    // Empty-mask eta*x has minimum1; the active branch alone has minimum~2.
    assert!(cut.lower > 0.99 && cut.lower <= 1.0, "{}", cut.lower);
}

#[test]
fn gray_last_original_input_residual_survives_dense_allocation() {
    let mut authored = spec(0.5, false);
    authored.first[79 * HEADS + 80] += 1.0;
    let s = Source::build(&authored, &[-1.0; INPUTS], &[1.0; INPUTS], 0, 1, end()).unwrap();
    let c = candidate(&s, 0, false);
    let cut = signed::certify_gray(&s, c.negative, c.positive, vec![], &mut budget()).unwrap();
    // x79=1, all other x=0 gives this block's exact signed geometry value-1.
    assert!(cut.lower <= -1.0, "last coordinate omitted: {}", cut.lower);
}

#[test]
fn gray_invalid_roles_and_eta_refuse_before_any_complete_mask() {
    let s = source(0.5, false);
    for bad in [
        vec![],
        vec![(80, 1.0)],
        vec![(0, f64::NAN)],
        vec![(0, -1.0)],
        vec![(0, 1.0), (0, 1.0)],
        vec![(HEADS, 1.0)],
    ] {
        let c = candidate(&s, 0, false);
        let mut work = budget();
        assert!(signed::certify_gray(
            &s,
            c.negative,
            signed::PositiveTerms(bad),
            vec![],
            &mut work
        )
        .is_none());
        assert_eq!(work.masks, 0);
    }
    for bad_eta in [
        vec![(INPUTS, 1.0)],
        vec![(79, f64::NAN)],
        vec![(79, 1.0), (79, 1.0)],
    ] {
        let c = candidate(&s, 0, false);
        assert!(signed::certify_gray(&s, c.negative, c.positive, bad_eta, &mut budget()).is_none());
    }
}
