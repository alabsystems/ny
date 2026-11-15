// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact fixed-family controls; nominal optimizer results grant no authority.
use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::time::Duration;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn r(n: i64, d: i64) -> Q {
    Q::new(n.into(), d.into())
}
struct Case {
    a: [Vec<f64>; 2],
    bias: [f64; 2],
    b: Vec<f64>,
    b_bias: f64,
    v: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    point: Vec<f64>,
}
impl Case {
    fn view(&self) -> Affines<'_> {
        Affines {
            a: [&self.a[0], &self.a[1]],
            bias: self.bias,
            b: &self.b,
            b_bias: self.b_bias,
            v: &self.v,
            lower: &self.lower,
            upper: &self.upper,
            point: &self.point,
        }
    }
}
fn constant(v: f64) -> Case {
    Case {
        a: [vec![1., 1.], vec![1., -1.]],
        bias: [0., 0.],
        b: vec![0.; 2],
        b_bias: 1.,
        v: vec![v, 0.],
        lower: vec![-1.; 2],
        upper: vec![1.; 2],
        point: vec![0.; 2],
    }
}
/// Independent exact original affine support, enumerating the small box corners.
fn support_q(a: &Affines<'_>, mask: usize, tau: &Q) -> Q {
    (0..1usize << a.b.len())
        .map(|corner| {
            let mut value = -tau * q(a.b_bias);
            for t in 0..2 {
                if mask & (1 << t) != 0 {
                    value += q(a.bias[t]);
                }
            }
            for i in 0..a.b.len() {
                let mut c = -tau * q(a.b[i]) - q(a.v[i]);
                for t in 0..2 {
                    if mask & (1 << t) != 0 {
                        c += q(a.a[t][i]);
                    }
                }
                value += c * q(if corner & (1 << i) == 0 {
                    a.lower[i]
                } else {
                    a.upper[i]
                });
            }
            value
        })
        .max()
        .unwrap()
}
/// A convex piecewise-linear branch attains a minimum at an endpoint or one
/// exact ratio. This oracle uses rational ratios and exhaustive corner prices.
fn minimum_q(a: &Affines<'_>, mask: usize) -> Q {
    let mut candidates = vec![q(0.), q(1.)];
    for i in 0..a.b.len() {
        if a.b[i] == 0. {
            continue;
        }
        let mut c = -q(a.v[i]);
        for t in 0..2 {
            if mask & (1 << t) != 0 {
                c += q(a.a[t][i]);
            }
        }
        let at = c / q(a.b[i]);
        if at >= q(0.) && at <= q(1.) {
            candidates.push(at);
        }
    }
    candidates
        .iter()
        .map(|tau| support_q(a, mask, tau))
        .min()
        .unwrap()
}
fn exact_case(c: &Case) -> Minimum {
    let a = c.view();
    let actual = minimize(&a, end(), &mut || true).unwrap();
    let old = [0., 0.5, 0.5, 1.];
    let mut expected = Vec::new();
    for mask in 0..4 {
        assert!(actual.tau[mask].is_finite() && (0. ..=1.).contains(&actual.tau[mask]));
        let best = minimum_q(&a, mask);
        assert_eq!(q(actual.mask_bounds[mask]), best, "mask{mask}");
        assert_eq!(support_q(&a, mask, &q(actual.tau[mask])), best);
        assert!(best <= support_q(&a, mask, &q(old[mask])));
        expected.push(best);
    }
    let h = expected.into_iter().max().unwrap();
    assert_eq!(q(actual.upper), h);
    let at =
        a.v.iter()
            .zip(a.point)
            .map(|(&v, &x)| q(v) * q(x))
            .sum::<Q>();
    assert_eq!(q(actual.price), h + at);
    actual
}

#[test]
fn branch_tau_constant_positive_separates_all_single_marginals_and_all_residual_splits() {
    // P=R(1)=1. For EACH chosen negative, correlated/anticorrelated corner
    // atoms of mass2/5 each plus origin mass1/5 realize mean0,P1,N4/5.
    // Hence every valid single-target affine row admits this projection.
    for sign in [1., -1.] {
        let atoms = [
            (r(2, 5), [q(1.), q(sign)]),
            (r(2, 5), [q(-1.), q(-sign)]),
            (r(1, 5), [q(0.), q(0.)]),
        ];
        for i in 0..2 {
            assert_eq!(atoms.iter().map(|(w, x)| w * &x[i]).sum::<Q>(), Q::zero());
        }
        assert_eq!(atoms.iter().map(|(w, _)| w.clone()).sum::<Q>(), q(1.));
        let head = atoms
            .iter()
            .map(|(w, x)| w * (&x[0] + q(sign) * &x[1]).max(Q::zero()))
            .sum::<Q>();
        assert_eq!(head, r(4, 5));
    }
    // For ANY theta and ANY v, combine empty support at(-1,0) with weight1/2,
    // mask1 at(1,1) with1/4, mask2 at(1,-1) with1/4. Their input means and
    // theta coefficients cancel. Thus every residual split has H>=3/4.
    let weights = [r(1, 2), r(1, 4), r(1, 4)];
    let points = [[-1., 0.], [1., 1.], [1., -1.]];
    for i in 0..2 {
        assert_eq!(
            (0..3).map(|j| &weights[j] * q(points[j][i])).sum::<Q>(),
            Q::zero()
        );
    }
    assert_eq!(-&weights[1] + &weights[2], Q::zero());
    let all_split_floor = &weights[1] * q(2.) + &weights[2] * q(1.);
    assert_eq!(all_split_floor, r(3, 4));
    let c = constant(0.75);
    let a = c.view();
    let old = [0., 0.5, 0.5, 1.];
    let old_h = (0..4)
        .map(|mask| support_q(&a, mask, &q(old[mask])))
        .max()
        .unwrap();
    assert_eq!(old_h, r(3, 4));
    let actual = exact_case(&c);
    assert_eq!(actual.tau, [1.; 4]);
    assert_eq!(actual.mask_bounds, [-0.25, 0.25, 0.25, 0.25]);
    let lhs = r(4, 5) + r(4, 5) - q(1.);
    assert_eq!(lhs, r(3, 5));
    assert!(lhs <= all_split_floor && lhs > q(actual.upper));
    let optimal = exact_case(&constant(1.));
    assert_eq!(optimal.upper, 0.);
}

#[test]
fn branch_tau_unstable_perturbed_family_beats_every_fixed_lambda_residual_split() {
    let eps = 1. / 16.;
    let mut c = Case {
        a: [vec![2., eps], vec![1., -eps]],
        bias: [0.; 2],
        b: vec![4., 0.],
        b_bias: 0.,
        v: vec![-0.5, 0.],
        lower: vec![-1.; 2],
        upper: vec![1.; 2],
        point: vec![0.; 2],
    };
    let a = c.view();
    let old = [0., 0.5, 0.5, 1.];
    assert_eq!(
        (0..4)
            .map(|mask| support_q(&a, mask, &q(old[mask])))
            .max()
            .unwrap(),
        q(0.5 + eps)
    );
    let actual = exact_case(&c);
    assert_eq!(actual.tau, [0.125, 0.625, 0.375, 0.875]);
    assert_eq!(actual.upper, eps);
    // At x=+/-1,y=0, any split residual slopes sum to-1. Therefore
    // average R(r0)+R(r1)=(|r0_x|+|r1_x|)/2>=1/2, regardless of v.
    let combined = q(2.) - q(4.) + q(1.);
    assert_eq!(combined, q(-1.));
    let every_split_floor = combined.abs() / q(2.);
    assert_eq!(every_split_floor, q(0.5));
    let lhs = q(1. / 8.) + q(1. / 8.);
    assert!(lhs <= every_split_floor && lhs > q(actual.upper));
    c.v[0] = 0.;
    let zero = exact_case(&c);
    assert_eq!(zero.tau, [0., 0.5, 0.25, 0.75]);
    assert_eq!(zero.upper, eps);
}

#[test]
fn branch_tau_matches_exact_small_box_breakpoints_and_prices_the_original_hint() {
    let cases = [
        Case {
            a: [vec![0.5], vec![-1.]],
            bias: [0.25, -0.5],
            b: vec![2.],
            b_bias: 0.25,
            v: vec![0.25],
            lower: vec![-2.],
            upper: vec![1.],
            point: vec![0.5],
        },
        Case {
            a: [vec![1., -0.5], vec![-0.5, 1.]],
            bias: [-0.25, 0.75],
            b: vec![-2., 0.5],
            b_bias: -0.5,
            v: vec![0.25, -0.25],
            lower: vec![-1., 1.],
            upper: vec![2., 3.],
            point: vec![0.5, 2.],
        },
        Case {
            a: [vec![0., 0.5, -1.], vec![1., -0.5, 1.]],
            bias: [0.5, -0.25],
            b: vec![0., -1., 2.],
            b_bias: 0.25,
            v: vec![0.5, 0., -0.25],
            lower: vec![-2., 0., 1.],
            upper: vec![1., 2., 1.],
            point: vec![0., 1., 1.],
        },
    ];
    for case in &cases {
        exact_case(case);
    }
}

#[test]
fn branch_tau_fixed_coordinates_use_full_linear_slope_and_zero_b_coordinates_keep_constants() {
    let c = Case {
        a: [vec![0., 1.], vec![0., -1.]],
        bias: [0.; 2],
        b: vec![2., 0.],
        b_bias: -3.,
        v: vec![0.; 2],
        lower: vec![2., -1.],
        upper: vec![2., 1.],
        point: vec![2., 0.],
    };
    // B=1 on this box although its stored intercept is negative. All weighted
    // breakpoints disappear; the full fixed-coordinate slope still selects1.
    let result = exact_case(&c);
    assert_eq!(result.tau, [1.; 4]);
    assert_eq!(result.mask_bounds, [-1., 0., 0., -1.]);
    let opposite = Case {
        a: [vec![0.], vec![0.]],
        bias: [0.; 2],
        b: vec![2.],
        b_bias: 3.,
        v: vec![0.],
        lower: vec![-2.],
        upper: vec![-2.],
        point: vec![-2.],
    };
    let result = exact_case(&opposite);
    assert_eq!(result.tau, [0.; 4]);
    assert_eq!(result.upper, 0.);
}

#[test]
fn branch_tau_outside_breakpoints_change_the_initial_derivative_at_both_endpoints() {
    for (b, head, expected) in [
        ([4., 2., 1.], [-4., 1., 2.], 0.),
        ([1., 2., 4.], [-1., 1., 8.], 1.),
    ] {
        let c = Case {
            a: [head.to_vec(), vec![0.; 3]],
            bias: [0.; 2],
            b: b.to_vec(),
            b_bias: 0.,
            v: vec![0.; 3],
            lower: vec![-1.; 3],
            upper: vec![1.; 3],
            point: vec![0.; 3],
        };
        // Ratios are[-1,1/2,2]. Ignoring either outside slope can pick1/2.
        let result = exact_case(&c);
        assert_eq!(result.tau[1], expected);
    }
}

#[test]
fn branch_tau_equal_breakpoints_are_stable_and_old_half_wins_flat_minimum_ties() {
    let c = Case {
        a: [vec![0., 1., 2.], vec![0.; 3]],
        bias: [0.; 2],
        b: vec![1., 2., 4.],
        b_bias: 0.,
        v: vec![0.; 3],
        lower: vec![-1.; 3],
        upper: vec![1.; 3],
        point: vec![0.; 3],
    };
    let a = c.view();
    let mut workspace = Workspace::new(&a, end(), &mut || true).unwrap();
    let (slope, _) = prepare_mask(&a, &mut workspace, 1, 0.5, end(), &mut || true).unwrap();
    workspace.sorted(end(), &mut || true).unwrap();
    assert_eq!(
        workspace
            .order
            .iter()
            .map(|&i| workspace.entries[i].coordinate)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(workspace.crossing(slope, end(), &mut || true), Some(0.5));
    exact_case(&c);
    let flat = Case {
        a: [vec![0.25, 0.75], vec![0.; 2]],
        bias: [0.; 2],
        b: vec![1.; 2],
        b_bias: 0.,
        v: vec![0.; 2],
        lower: vec![-1.; 2],
        upper: vec![1.; 2],
        point: vec![0.; 2],
    };
    let a = flat.view();
    let mut workspace = Workspace::new(&a, end(), &mut || true).unwrap();
    let (slope, _) = prepare_mask(&a, &mut workspace, 1, 0.5, end(), &mut || true).unwrap();
    workspace.sorted(end(), &mut || true).unwrap();
    assert_eq!(workspace.crossing(slope, end(), &mut || true), Some(0.25));
    assert_eq!(exact_case(&flat).tau[1], 0.5);
}

#[test]
fn branch_tau_preserves_negative_upper_and_avoids_unnecessary_full_width_overflow() {
    let c = Case {
        a: [vec![0.], vec![0.]],
        bias: [0.; 2],
        b: vec![0.],
        b_bias: 1.,
        v: vec![0.],
        lower: vec![-f64::MAX],
        upper: vec![f64::MAX],
        point: vec![0.],
    };
    assert!((c.upper[0] - c.lower[0]).is_infinite());
    let result = exact_case(&c);
    assert_eq!(
        (result.tau, result.upper, result.price),
        ([1.; 4], -1., -1.)
    );
}

#[test]
fn branch_tau_malformed_nonfinite_and_overflow_inputs_decline_without_partial_minimum() {
    for field in 0..9 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut c = constant(0.75);
            match field {
                0 => c.a[0][0] = bad,
                1 => c.a[1][0] = bad,
                2 => c.bias[0] = bad,
                3 => c.b[0] = bad,
                4 => c.b_bias = bad,
                5 => c.v[0] = bad,
                6 => c.lower[0] = bad,
                7 => c.upper[0] = bad,
                _ => c.point[0] = bad,
            }
            assert!(minimize(&c.view(), end(), &mut || true).is_none());
        }
    }
    let mut bad = constant(0.75);
    bad.upper[0] = -2.;
    assert!(minimize(&bad.view(), end(), &mut || true).is_none());
    let mut bad = constant(0.75);
    bad.a[1].pop();
    assert!(minimize(&bad.view(), end(), &mut || true).is_none());
    let mut bad = constant(0.75);
    bad.b[0] = f64::from_bits(1);
    assert!(minimize(&bad.view(), end(), &mut || true).is_none());
    let mut bad = constant(0.75);
    bad.a[0][0] = f64::MAX;
    bad.a[1][0] = f64::MAX;
    assert!(minimize(&bad.view(), end(), &mut || true).is_none());
    let empty = Affines {
        a: [&[], &[]],
        bias: [0.; 2],
        b: &[],
        b_bias: 0.,
        v: &[],
        lower: &[],
        upper: &[],
        point: &[],
    };
    assert!(minimize(&empty, end(), &mut || true).is_none());
    let wide = Case {
        a: [vec![0.; 129], vec![0.; 129]],
        bias: [0.; 2],
        b: vec![0.; 129],
        b_bias: 1.,
        v: vec![0.; 129],
        lower: vec![-1.; 129],
        upper: vec![1.; 129],
        point: vec![0.; 129],
    };
    assert!(minimize(&wide.view(), end(), &mut || true).is_none());
}

#[test]
fn branch_tau_expiry_and_transient_final_poll_refusal_do_not_return_a_partial_result() {
    let c = Case {
        a: [vec![0., 1., 2.], vec![1., -1., 0.]],
        bias: [0.; 2],
        b: vec![1., 2., 4.],
        b_bias: 0.,
        v: vec![0.; 3],
        lower: vec![-1.; 3],
        upper: vec![1.; 3],
        point: vec![0.; 3],
    };
    let mut polls = 0;
    minimize(&c.view(), end(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert!(polls > 3);
    for stop in [1, polls] {
        let mut seen = 0;
        assert!(minimize(&c.view(), end(), &mut || {
            seen += 1;
            seen != stop
        })
        .is_none());
        assert_eq!(seen, stop);
    }
    let mut polls = 0;
    assert!(minimize(
        &c.view(),
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        &mut || {
            polls += 1;
            true
        }
    )
    .is_none());
    assert_eq!(polls, 0);
}
