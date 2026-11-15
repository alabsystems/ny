// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored finite Source certificates; no benchmark or timing claim.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
pub(super) fn rat(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
pub(super) fn cycle(n: usize, bias: f64) -> HullMarginProblem {
    let mut w = vec![0.; n * n];
    for j in 0..n {
        w[j * n + j] = 1.;
        w[j * n + (j + 1) % n] = -1.;
    }
    HullMarginProblem::new(
        w,
        vec![0.; n],
        n,
        &vec![-1.; n],
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; n],
        vec![1.; n],
    )
    .unwrap()
}
pub(super) fn cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    for j in 0..p.hidden_dim {
        if p.c[j] > 0. {
            m.alpha[j] = 1.;
        } else {
            for i in 0..p.input_dim {
                m.g[j * p.input_dim + i] = 0.5 * p.w_row(j)[i];
            }
        }
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, lower, end(), &mut || true).unwrap()
}
fn graph(p: &HullMarginProblem, x: &[f64]) -> Q {
    rat(p.d)
        + (0..p.hidden_dim)
            .map(|j| {
                let z = rat(p.b1[j])
                    + (0..p.input_dim)
                        .map(|i| rat(p.w_row(j)[i]) * rat(x[i]))
                        .sum::<Q>();
                rat(p.c[j]) * z.max(Q::zero())
            })
            .sum::<Q>()
}
fn witness(s: &System, lambda: f64, b: &Budget) -> (Vec<f64>, Vec<f64>) {
    // This control supplies a rational uniform dual and a FLOAT spin-block
    // factor. The Source kernel, not the LLT result, establishes its bound.
    let n = s.n;
    let k = n - 1;
    let a: Vec<_> = (0..k * k)
        .map(|v| {
            let i = v / k + 1;
            let j = v % k + 1;
            (if i == j { lambda } else { 0. }) - s.q[i * n + j].lo.midpoint(s.q[i * n + j].hi)
        })
        .collect();
    let small = search::factor(&a, k, b).unwrap();
    let mut ll = vec![0.; n * n];
    let mut dual = vec![lambda; n];
    dual[0] = 0.;
    for i in 0..k {
        for j in 0..=i {
            ll[(i + 1) * n + j + 1] = small[i * k + j];
        }
    }
    (dual, ll)
}
#[test]
fn global_sdp_interval_polynomial_covers_all_original_asymmetric_box_corners() {
    let p = HullMarginProblem::new(
        vec![1., -0.5, -0.25, 1.5],
        vec![0.25, -0.5],
        2,
        &[-0.5, -1.25],
        &[2.],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.5, 0.25],
        vec![0.75, 1.25],
    )
    .unwrap();
    let c = cuts(&p);
    let g = vec![0.375, -0.125];
    let s = certificate::system(&c, &g, end(), &mut || true).unwrap();
    let lambda: Vec<_> = (0..s.n)
        .map(|i| {
            s.q[i * s.n..(i + 1) * s.n]
                .iter()
                .map(|v| v.lo.abs().max(v.hi.abs()))
                .sum::<f64>()
                + 1.
        })
        .collect();
    let (upper, _) =
        certificate::certify(&s, &lambda, &vec![0.; s.n * s.n], end(), &mut || true).unwrap();
    for bits in 0..4 {
        let x: Vec<_> = (0..2)
            .map(|i| {
                if bits & (1 << i) == 0 {
                    p.lower[i]
                } else {
                    p.upper[i]
                }
            })
            .collect();
        let negative = rat(p.d) - graph(&p, &x) - (0..2).map(|i| rat(g[i]) * rat(x[i])).sum::<Q>();
        assert!(rat(upper) >= negative);
        let mut t = vec![Q::from_integer(1.into())];
        for i in 0..2 {
            let m = p.lower[i].midpoint(p.upper[i]);
            let r = sub_up(p.upper[i], m).max(sub_up(m, p.lower[i])).max(0.);
            t.push((rat(x[i]) - rat(m)) / rat(r));
        }
        for signs in 0..4 {
            let mut z = t.clone();
            for j in 0..2 {
                z.push(rat(if signs & (1 << j) == 0 { -1. } else { 1. }));
            }
            let mut lo = rat(s.constant.lo);
            let mut hi = rat(s.constant.hi);
            for i in 0..s.n {
                for j in 0..s.n {
                    let v = &z[i] * &z[j];
                    let a = rat(s.q[i * s.n + j].lo) * &v;
                    let b = rat(s.q[i * s.n + j].hi) * v;
                    lo += a.clone().min(b.clone());
                    hi += a.max(b);
                }
            }
            let mut expected = Q::zero();
            for j in 0..2 {
                let pre = rat(p.b1[j]) + (0..2).map(|i| rat(p.w_row(j)[i]) * rat(x[i])).sum::<Q>();
                expected += rat(-p.c[j]) * pre * (Q::from_integer(1.into()) + &z[3 + j]) / rat(2.);
            }
            expected -= (0..2).map(|i| rat(g[i]) * rat(x[i])).sum::<Q>();
            assert!(
                lo <= expected && expected <= hi,
                "source polynomial coefficient enclosure"
            );
        }
    }
    assert!(upper.is_finite());
}
#[test]
fn global_sdp_arbitrary_indefinite_dual_and_gram_repair_retains_negative_diagonal_error() {
    let s = System {
        n: 2,
        q: vec![
            certificate::I::zero(),
            certificate::I { lo: 0.5, hi: 0.5 },
            certificate::I { lo: 0.5, hi: 0.5 },
            certificate::I::zero(),
        ],
        constant: certificate::I::zero(),
    };
    for lambda in [[0., 0.], [-3., -2.], [0.25, 0.125]] {
        let (h, correction) =
            certificate::certify(&s, &lambda, &[2., 0., 0., 2.], end(), &mut || true).unwrap();
        assert!(
            h >= 9. && correction >= 9. - lambda.iter().sum::<f64>(),
            "{h}, {correction}"
        );
    }
}
#[test]
fn global_sdp_odd9_rational_dual_has_an_actual_finite_checked_factor() {
    let p = cycle(9, 285. / 32.);
    let c = cuts(&p);
    let g = vec![0.; 9];
    let s = certificate::system(&c, &g, end(), &mut || true).unwrap();
    let b = Budget::new(&c, end(), Duration::ZERO, s.n).unwrap();
    let (lambda, l) = witness(&s, 1011. / 2048., &b);
    let (h, repair) = certificate::certify(&s, &lambda, &l, end(), &mut || true).unwrap();
    assert!(h <= 9099. / 1024. + 1e-8 && repair < 1e-8);
    let lower = root_lower(&c, &g, h, end(), &mut || true).unwrap();
    assert!(lower > 0.0204 && lower <= 21. / 1024.);
    assert_eq!(s.n, 19);
    assert_eq!(lambda[0], 0.);
}
#[test]
fn global_sdp_odd29_rational_dual_has_an_actual_finite_checked_factor() {
    let p = cycle(29, 29. - 1. / 64.);
    let c = cuts(&p);
    let g = vec![0.; 29];
    let s = certificate::system(&c, &g, end(), &mut || true).unwrap();
    let b = Budget::new(&c, end(), Duration::ZERO, s.n).unwrap();
    let (lambda, l) = witness(&s, 1023. / 2048., &b);
    let (h, repair) = certificate::certify(&s, &lambda, &l, end(), &mut || true).unwrap();
    assert!(h <= 29. * 1023. / 1024. + 1e-8 && repair < 1e-8);
    assert!(root_lower(&c, &g, h, end(), &mut || true).unwrap() > 0.0126);
}
#[test]
fn global_sdp_native_optimizer_closes_odd9_and_odd29_without_an_lp_or_spectral_recognizer() {
    for n in [9, 29] {
        let p = cycle(n, n as f64 - 1. / 64.);
        let c = cuts(&p);
        let scalar = c.checked_lower.to_bits();
        let m = c.multipliers.clone();
        let row = propose(&c, end(), Duration::ZERO)
            .expect("bounded native optimizer must generate an effective witness");
        let stats = row.stats();
        assert!(stats.lower > 0.005 && stats.optimizer_iterations > 0 && stats.factorizations > 0);
        assert!(
            stats.checked_candidates > 0
                && stats.work_units <= MAX_WORK
                && stats.admitted_bytes <= MAX_BYTES
        );
        assert!(stats.active_dimension >= 2 * n && stats.active_dimension <= stats.dimension);
        assert_eq!(c.proposal_lp_calls, 0);
        assert_eq!(c.checked_lower.to_bits(), scalar);
        assert_eq!(c.multipliers, m);
        let checked = row.replay(&c, end(), &mut || true).unwrap();
        assert_eq!(checked.lower.to_bits(), row.lower.to_bits());
    }
}
#[test]
fn global_sdp_unsafe_and_tie_original_cycle_points_never_get_a_positive_bound() {
    for bias in [7.75, 8.] {
        let p = cycle(9, bias);
        let c = cuts(&p);
        let row = propose(&c, end(), Duration::ZERO).unwrap();
        let x: Vec<_> = (0..9).map(|i| if i % 2 == 0 { 1. } else { -1. }).collect();
        let actual = graph(&p, &x);
        assert!(actual <= Q::zero());
        assert!(rat(row.lower) <= actual && row.lower <= 0.);
    }
}
#[test]
fn global_sdp_full_q_prices_positive_roles_and_last_coordinate_residual() {
    let p = HullMarginProblem::new(
        vec![1., -1., -1., 1., 0., 1.],
        vec![0., 0., 2.],
        2,
        &[-1., -1., 0.75],
        &[1.],
        1,
        &[(0, 1.)],
        0.,
        vec![-1., -1.],
        vec![1., 1.],
    )
    .unwrap();
    let c = cuts(&p);
    let g = vec![0.125, -0.375];
    let s = certificate::system(&c, &g, end(), &mut || true).unwrap();
    let lambda: Vec<_> = (0..s.n)
        .map(|i| {
            s.q[i * s.n..(i + 1) * s.n]
                .iter()
                .map(|v| v.lo.abs().max(v.hi.abs()))
                .sum::<f64>()
                + 0.125
        })
        .collect();
    let (h, _) =
        certificate::certify(&s, &lambda, &vec![0.; s.n * s.n], end(), &mut || true).unwrap();
    let lower = root_lower(&c, &g, h, end(), &mut || true).unwrap();
    let exact = (0..4)
        .map(|m| {
            graph(
                &p,
                &[
                    if m & 1 == 0 { -1. } else { 1. },
                    if m & 2 == 0 { -1. } else { 1. },
                ],
            )
        })
        .min()
        .unwrap();
    assert!(rat(lower) <= exact);
    assert_ne!(g[1], positive_point(&c).unwrap()[1]);
    // A tight valid authored upper makes omission of the final residual
    // return about zero, above the actual minimum; the loose DD bound alone
    // would not catch that omission.
    let tight = 2.500000000001;
    for mask in 0..4 {
        let x = [
            if mask & 1 == 0 { -1. } else { 1. },
            if mask & 2 == 0 { -1. } else { 1. },
        ];
        let neg = (0..2)
            .map(|j| {
                let z = (0..2).map(|i| rat(p.w_row(j)[i]) * rat(x[i])).sum::<Q>();
                rat(-p.c[j]) * z.max(Q::zero())
            })
            .sum::<Q>()
            - (0..2).map(|i| rat(g[i]) * rat(x[i])).sum::<Q>();
        assert!(neg <= rat(tight));
    }
    let strict = root_lower(&c, &g, tight, end(), &mut || true).unwrap();
    assert!(strict > -1.251 && strict < -1.249 && rat(strict) <= exact);
}
#[test]
fn global_sdp_fixed_input_and_zero_columns_remain_in_source_replay() {
    let p = HullMarginProblem::new(
        vec![0., 1., 0., -1.],
        vec![0.25, -0.125],
        2,
        &[-1., -1.],
        &[4.],
        1,
        &[(0, 1.)],
        0.,
        vec![0.5, -1.],
        vec![0.5, 1.],
    )
    .unwrap();
    let c = cuts(&p);
    let s = certificate::system(&c, &[0., 0.], end(), &mut || true).unwrap();
    assert_eq!(s.n, 5);
    assert!((0..s.n).any(|j| s.q[2 * s.n + j].lo != 0. || s.q[2 * s.n + j].hi != 0.));
    // Directed multiplication/cancellation may widen a mathematical zero.
    // Retain that row in Source and check its enclosure, not cosmetic zeros.
    assert!(s.q[s.n..2 * s.n].iter().all(|v| {
        v.lo.is_finite()
            && v.hi.is_finite()
            && v.lo <= 0.
            && v.hi >= 0.
            && v.lo.abs().max(v.hi.abs()) < 1e-300
    }));
    let lambda: Vec<_> = (0..s.n)
        .map(|i| {
            s.q[i * s.n..(i + 1) * s.n]
                .iter()
                .map(|v| v.lo.abs().max(v.hi.abs()))
                .sum::<f64>()
                + 1e-12
        })
        .collect();
    let (upper, _) =
        certificate::certify(&s, &lambda, &vec![0.; s.n * s.n], end(), &mut || true).unwrap();
    let lower = root_lower(&c, &[0., 0.], upper, end(), &mut || true).unwrap();
    assert!(upper.is_finite() && upper < 1.25000001 && lower > 2.74999999);
    for y in [-1., 1.] {
        let actual = graph(&p, &[0.5, y]);
        assert!(rat(upper) >= rat(p.d) - &actual && rat(lower) <= actual);
    }
}
