// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Separate unexecuted addendum; install as another child of corners::core.
//! Does not alter the frozen v1 core API or grant Source/replay authority.
use super::*;

fn power2(exponent: i32) -> f64 {
    assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}
fn raw_floor(coeff: &[I], lower: &[f64], upper: &[f64]) -> f64 {
    coeff
        .iter()
        .zip(lower.iter().zip(upper))
        .fold(0., |value, (&q, (&l, &u))| {
            finite(add_down(value, q.minimum(l, u).unwrap())).unwrap()
        })
}
fn repeated_ranker(lower: &[f64], upper: &[f64], q: &[I], bits: &[u8], k: usize) -> (Ranker, f64) {
    let raw = raw_floor(q, lower, upper);
    let mut rank = Ranker::new(lower, upper, bits, k).unwrap();
    let mut constant = 0.;
    for (c, &n) in bits.iter().enumerate() {
        for mask in 0..1usize << n {
            rank.push(c, mask, q, raw, &mut || true).unwrap();
        }
        constant = finite(add_down(constant, raw)).unwrap();
    }
    (rank, constant)
}
fn repeated_data(lower: &[f64], upper: &[f64], q: &[I], k: usize) -> Data {
    let (rank, constant) = repeated_ranker(lower, upper, q, &[1], k);
    let axes = rank
        .finish(constant, 0., &vec![0.; q.len()], &mut || true)
        .unwrap();
    let mut collect = Collector::new(axes).unwrap();
    let raw = raw_floor(q, lower, upper);
    for mask in 0..2 {
        collect.push(0, mask, q, raw, &mut || true).unwrap();
    }
    collect.finish(constant, 0., &mut || true).unwrap()
}

#[test]
fn corner_edge_finite_endpoints_do_not_require_finite_rounded_width() {
    let l = -power2(1023);
    let u = power2(1023);
    let q = power2(-1022);
    assert!((u - l).is_infinite());
    assert_eq!(q * l, -2.);
    assert_eq!(q * u, 2.);
    let data = repeated_data(&[l], &[u], &[I::point(q).unwrap()], 1);
    let bound = data
        .lower(&[I::zero()], I::zero(), &[0], &[0], &mut || true)
        .unwrap();
    // Independent exact affine oracle: q*x has minimum -2. A kernel that
    // multiplies q by the overflowing rounded width would spuriously refuse.
    assert!(bound.is_finite() && bound <= -2. && -2. - bound < 1e-10);
}

#[test]
fn corner_edge_exact_half_mass_subnormals_cannot_collapse_to_zero() {
    let input = power2(1023);
    for (source_coefficient, exact_minimum) in
        [(f64::MIN_POSITIVE, -1.), (f64::from_bits(1), -power2(-52))]
    {
        // This is the exact allocated-half-mass interval operation, NOT a
        // constructor override or a rounded point half product.
        let half = I::point(source_coefficient).unwrap().scale(0.5).unwrap();
        let data = repeated_data(&[-input], &[input], &[half], 1);
        let bound = data
            .lower(&[I::zero()], I::zero(), &[0], &[0], &mut || true)
            .unwrap();
        // Binary exponents independently give these exact real minima:
        // (2^-1022)/2 * (-2^1023)=-1;
        // (2^-1074)/2 * (-2^1023)=-2^-52.
        assert!(bound.is_finite() && bound <= exact_minimum && bound < 0.);
    }
    assert_eq!(f64::from_bits(1) * 0.5, 0.);
    assert!(0. > -power2(-52), "zero point pricing would be UNSOUND");
}

fn before_last_mask() -> (Collector, [I; 7], f64, f64) {
    let q = [I::point(0.25).unwrap(); 7];
    let lo = [-1.; 7];
    let hi = [1.; 7];
    let (rank, constant) = repeated_ranker(&lo, &hi, &q, &[1, 1], 7);
    let axes = rank.finish(constant, 0., &[0.; 7], &mut || true).unwrap();
    let mut collect = Collector::new(axes).unwrap();
    let raw = raw_floor(&q, &lo, &hi);
    // A full earlier component AND all but the last mask of the final one
    // have already been accepted; a late failure must not publish that prefix.
    for (c, mask) in [(0, 0), (0, 1), (1, 0)] {
        collect.push(c, mask, &q, raw, &mut || true).unwrap();
    }
    (collect, q, raw, constant)
}

#[test]
fn corner_edge_every_last_mask_checkpoint_refuses_partial_corner_component_data() {
    let (mut complete, q, raw, constant) = before_last_mask();
    let mut checks = 0;
    complete
        .push(1, 1, &q, raw, &mut || {
            checks += 1;
            true
        })
        .unwrap();
    assert!(checks > 2);
    assert!(complete.finish(constant, 0., &mut || true).is_some());
    // Learn the public cancellation checkpoints, not an implementation-specific
    // numeric count. Exhausting them includes the final corner block and final
    // component completion poll, after substantial valid data already exists.
    for stop in 1..=checks {
        let (mut collect, q, raw, constant) = before_last_mask();
        let mut called = 0;
        assert!(collect
            .push(1, 1, &q, raw, &mut || {
                called += 1;
                called != stop
            })
            .is_none());
        assert_eq!(called, stop);
        assert!(collect.push(1, 1, &q, raw, &mut || true).is_none());
        assert!(collect.finish(constant, 0., &mut || true).is_none());
    }
}

#[test]
fn corner_edge_final_rank_and_collection_publication_polls_are_mandatory() {
    let q = [I::point(0.25).unwrap(); 7];
    let (rank, constant) = repeated_ranker(&[-1.; 7], &[1.; 7], &q, &[1, 1], 7);
    let mut checks = 0;
    assert!(rank
        .finish(constant, 0., &[0.; 7], &mut || {
            checks += 1;
            true
        })
        .is_some());
    let (rank, constant) = repeated_ranker(&[-1.; 7], &[1.; 7], &q, &[1, 1], 7);
    let mut called = 0;
    assert!(rank
        .finish(constant, 0., &[0.; 7], &mut || {
            called += 1;
            called != checks
        })
        .is_none());
    assert_eq!(called, checks);
    let (mut collect, q, raw, constant) = before_last_mask();
    collect.push(1, 1, &q, raw, &mut || true).unwrap();
    let mut checks = 0;
    assert!(collect
        .finish(constant, 0., &mut || {
            checks += 1;
            true
        })
        .is_some());
    let (mut collect, q, raw, constant) = before_last_mask();
    collect.push(1, 1, &q, raw, &mut || true).unwrap();
    let mut called = 0;
    assert!(collect
        .finish(constant, 0., &mut || {
            called += 1;
            called != checks
        })
        .is_none());
    assert_eq!(called, checks);
}

#[test]
fn corner_edge_final_price_poll_refuses_a_completed_numeric_bound() {
    let data = repeated_data(&[-1.; 7], &[1.; 7], &[I::point(0.25).unwrap(); 7], 7);
    let mut checks = 0;
    let good = data
        .lower(&[I::zero(); 7], I::zero(), &[0], &[0], &mut || {
            checks += 1;
            true
        })
        .unwrap();
    assert!(good <= -1.75 && -1.75 - good < 1e-10);
    let mut called = 0;
    assert!(data
        .lower(&[I::zero(); 7], I::zero(), &[0], &[0], &mut || {
            called += 1;
            called != checks
        })
        .is_none());
    assert_eq!(called, checks);
}
