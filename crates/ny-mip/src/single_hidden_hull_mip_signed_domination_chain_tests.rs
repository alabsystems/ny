// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Independent rational vertex/mixture controls; authored externally, unrun.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn dot(a: &[Q], x: &[Q]) -> Q {
    a.iter().zip(x).map(|(a, x)| a * x).sum()
}
fn function(c: &[f64], d: f64, x: &[Q]) -> Q {
    (q(d) + c.iter().zip(x).map(|(&a, x)| q(a) * x).sum::<Q>()).max(Q::zero())
}
fn vertex(lower: &[f64], upper: &[f64], mask: usize) -> Vec<Q> {
    (0..lower.len())
        .map(|i| {
            q(if mask & (1 << i) == 0 {
                lower[i]
            } else {
                upper[i]
            })
        })
        .collect()
}
// Independent intercept oracle: exhaust all ORIGINAL box vertices, without
// reading the helper's orientation/order/prefix state or trusting its slope.
fn intercept(c: &[f64], d: f64, lo: &[f64], hi: &[f64], v: &[f64]) -> Q {
    let v: Vec<_> = v.iter().map(|&a| q(a)).collect();
    (0..1usize << c.len())
        .map(|mask| {
            let x = vertex(lo, hi, mask);
            function(c, d, &x) - dot(&v, &x)
        })
        .max()
        .unwrap()
}
const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
// Exhaustive exact permutation oracle from a table of original vertex values.
// It never sorts the advice. It enumerates every candidate plane independently.
fn permutation_planes(c: &[f64; 3], d: f64, lo: &[f64; 3], hi: &[f64; 3]) -> Vec<(Q, Vec<Q>)> {
    let base_mask = (0..3)
        .filter(|&i| c[i] < 0.)
        .fold(0usize, |mask, i| mask | (1 << i));
    let values: Vec<_> = (0..8)
        .map(|mask| function(c, d, &vertex(lo, hi, mask)))
        .collect();
    let base = vertex(lo, hi, base_mask);
    PERMUTATIONS
        .iter()
        .map(|order| {
            let mut mask = base_mask;
            let mut v = vec![Q::zero(); 3];
            for &i in order {
                let next = mask ^ (1 << i);
                let delta = &values[next] - &values[mask];
                v[i] = delta / (q(hi[i]) - q(lo[i])) * q(if c[i] > 0. { 1. } else { -1. });
                mask = next;
            }
            (&values[base_mask] - dot(&v, &base), v)
        })
        .collect()
}
fn fixture() -> ([f64; 3], f64, [f64; 3], [f64; 3]) {
    ([1., -2., 0.5], 0.25, [-1., -0.5, 1.], [1., 1.5, 3.])
}
#[test]
fn nominal_chain_exact_dyadic_vertex_oracle_matches_all_permutations_over_a_3d_grid() {
    let (c, d, lo, hi) = fixture();
    let planes = permutation_planes(&c, d, &lo, &hi);
    for a in [0., 0.25, 0.75, 1.] {
        for b in [0., 0.25, 0.75, 1.] {
            for z in [0., 0.25, 0.75, 1.] {
                let advice = [lo[0] + 2. * a, hi[1] - 2. * b, lo[2] + 2. * z];
                let x: Vec<_> = advice.iter().map(|&a| q(a)).collect();
                let v = nominal_chain_direction(&c, d, &lo, &hi, &advice).unwrap();
                let h = intercept(&c, d, &lo, &hi, &v);
                let v: Vec<_> = v.iter().map(|&a| q(a)).collect();
                let expected = planes.iter().map(|(h, v)| h + dot(v, &x)).min().unwrap();
                assert_eq!(&h + dot(&v, &x), expected);
                for mask in 0..8 {
                    let y = vertex(&lo, &hi, mask);
                    assert!(function(&c, d, &y) <= &h + dot(&v, &y));
                }
                // An independent interior point checks the ORIGINAL graph, not only vertices.
                let y = vec![q(-0.5), q(0.75), q(2.5)];
                assert!(function(&c, d, &y) <= &h + dot(&v, &y));
            }
        }
    }
}
#[test]
fn nominal_chain_exact_chain_mixture_proves_minimal_affine_value_without_solver_authority() {
    let (c, d, lo, hi) = fixture();
    let advice = [0.5, 1., 2.];
    // Oriented z=(3/4,1/4,1/2): the four equal atoms are explicit original
    // vertices. Their average is x*, so EVERY affine majorant is >= this mean f.
    let masks = [2, 3, 7, 5]; // base, +input0, +input2, +input1 (signed orientation).
    let atoms: Vec<_> = masks.iter().map(|&m| vertex(&lo, &hi, m)).collect();
    for i in 0..3 {
        assert_eq!(
            atoms.iter().map(|x| x[i].clone()).sum::<Q>() / q(4.),
            q(advice[i])
        );
    }
    let mixture = atoms.iter().map(|x| function(&c, d, x)).sum::<Q>() / q(4.);
    assert_eq!(mixture, q(15. / 16.));
    let v = nominal_chain_direction(&c, d, &lo, &hi, &advice).unwrap();
    let h = intercept(&c, d, &lo, &hi, &v);
    let x: Vec<_> = advice.iter().map(|&a| q(a)).collect();
    assert_eq!(
        h + dot(&v.iter().map(|&a| q(a)).collect::<Vec<_>>(), &x),
        mixture
    );
    for (h, v) in permutation_planes(&c, d, &lo, &hi) {
        assert!(h + dot(&v, &x) >= mixture);
    }
}
#[test]
fn nominal_chain_corner_plane_strictly_improves_centered_relu_scalar_secant() {
    let c = [1., 1.];
    let lo = [-1., -1.];
    let hi = [1., 1.];
    let x = [1., -1.];
    let v = nominal_chain_direction(&c, 0., &lo, &hi, &x).unwrap();
    assert_eq!(v, vec![0., 1.]);
    let h = intercept(&c, 0., &lo, &hi, &v);
    assert_eq!(h, q(1.));
    assert_eq!(h + q(v[0]) * q(x[0]) + q(v[1]) * q(x[1]), Q::zero());
    assert_eq!(q(1.) + q(0.5) * (q(x[0]) + q(x[1])), q(1.));
}
#[test]
fn nominal_chain_signed_translated_fixed_and_zero_coordinates_keep_original_graphs() {
    let c = [2., -1., 3., 0.];
    let lo = [1., -2., 0.5, -f64::MAX];
    let hi = [3., 2., 0.5, f64::MAX];
    for d in [-2., 10., -100.] {
        let v = nominal_chain_direction(&c, d, &lo, &hi, &[2., -1., 900., 0.]).unwrap();
        assert_eq!((v[2], v[3]), (0., 0.));
        let h = intercept(&c, d, &lo, &hi, &v);
        let vq: Vec<_> = v.iter().map(|&a| q(a)).collect();
        for mask in 0..16 {
            let x = vertex(&lo, &hi, mask);
            assert!(function(&c, d, &x) <= &h + dot(&vq, &x));
        }
        if d == 10. {
            assert_eq!(v, vec![2., -1., 0., 0.]);
        }
        if d == -100. {
            assert_eq!(v, vec![0.; 4]);
        }
    }
}
#[test]
fn nominal_chain_clamped_advice_and_deterministic_ties_choose_only_a_direction() {
    let (c, d, lo, hi) = fixture();
    let clamped = nominal_chain_direction(&c, d, &lo, &hi, &[1., -0.5, 2.]).unwrap();
    assert_eq!(
        clamped,
        nominal_chain_direction(&c, d, &lo, &hi, &[100., -100., 2.]).unwrap()
    );
    let tied = [0., 0.5, 2.];
    let first = nominal_chain_direction(&c, d, &lo, &hi, &tied).unwrap();
    for _ in 0..8 {
        assert_eq!(
            first,
            nominal_chain_direction(&c, d, &lo, &hi, &tied).unwrap()
        );
    }
    let x: Vec<_> = tied.iter().map(|&a| q(a)).collect();
    let h = intercept(&c, d, &lo, &hi, &first);
    assert_eq!(
        h + dot(&first.iter().map(|&a| q(a)).collect::<Vec<_>>(), &x),
        permutation_planes(&c, d, &lo, &hi)
            .iter()
            .map(|(h, v)| h + dot(v, &x))
            .min()
            .unwrap()
    );
}
#[test]
fn nominal_chain_malformed_nonfinite_and_finite_overflow_refuse_without_output() {
    assert!(nominal_chain_direction(&[1.], 0., &[], &[1.], &[0.]).is_none());
    assert!(nominal_chain_direction(&[1.], 0., &[1.], &[0.], &[0.]).is_none());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(nominal_chain_direction(&[bad], 0., &[0.], &[1.], &[0.]).is_none());
        assert!(nominal_chain_direction(&[1.], bad, &[0.], &[1.], &[0.]).is_none());
        assert!(nominal_chain_direction(&[1.], 0., &[bad], &[1.], &[0.]).is_none());
        assert!(nominal_chain_direction(&[1.], 0., &[0.], &[bad], &[0.]).is_none());
        assert!(nominal_chain_direction(&[1.], 0., &[0.], &[1.], &[bad]).is_none());
    }
    assert!(nominal_chain_direction(&[1.], 0., &[-f64::MAX], &[f64::MAX], &[0.]).is_none());
    assert!(nominal_chain_direction(&[f64::MAX], 0., &[0.], &[2.], &[0.]).is_none());
    assert!(nominal_chain_direction(&[f64::MAX], f64::MAX, &[0.], &[1.], &[0.]).is_none());
}
#[test]
fn nominal_chain_subnormal_zero_dimension_and_existing_128_input_boundary_are_bounded() {
    let tiny = f64::from_bits(1);
    assert_eq!(
        nominal_chain_direction(&[tiny], 0., &[0.], &[1.], &[0.5]),
        Some(vec![tiny])
    );
    assert!(nominal_chain_direction(&[tiny], 0., &[0.], &[0.5], &[0.]).is_none());
    assert_eq!(
        nominal_chain_direction(&[], 2., &[], &[], &[]),
        Some(vec![])
    );
    assert_eq!(
        nominal_chain_direction(
            &vec![1.; 128],
            256.,
            &vec![0.; 128],
            &vec![1.; 128],
            &vec![0.5; 128]
        ),
        Some(vec![1.; 128])
    );
    assert!(nominal_chain_direction(
        &vec![1.; 129],
        0.,
        &vec![0.; 129],
        &vec![1.; 129],
        &vec![0.; 129]
    )
    .is_none());
}
#[test]
fn nominal_chain_identical_slopes_with_different_intercepts_cannot_certify_a_row() {
    let a = nominal_chain_direction(&[1.], 2., &[-1.], &[1.], &[0.]).unwrap();
    let b = nominal_chain_direction(&[1.], 4., &[-1.], &[1.], &[0.]).unwrap();
    assert_eq!(a, b);
    let ha = intercept(&[1.], 2., &[-1.], &[1.], &a);
    let hb = intercept(&[1.], 4., &[-1.], &[1.], &b);
    assert_eq!((ha.clone(), hb), (q(2.), q(4.)));
    assert!(function(&[1.], 4., &[q(1.)]) > ha + q(a[0]));
    // Identical returned Vecs cannot own these distinct H values. Only the
    // existing fresh original-Source checker may produce an exported row.
}
