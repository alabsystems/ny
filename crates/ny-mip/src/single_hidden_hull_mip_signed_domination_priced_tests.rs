// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact examples, independent chain moments and original-encoder row checks.
use super::super::{dense, kernel, test_fixture as fixture};
use super::*;
use crate::single_hidden_hull::HullMarginProblem;
use fixture::{q, rows_hold, Fixture};
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_core::Bound;
use std::time::Duration;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} versus {b}");
}
fn head(weights: &[f64], value: f64) -> Head<'_> {
    Head {
        weights,
        bias: 0.,
        value,
    }
}
fn example(bias: f64) -> Fixture {
    Fixture {
        w: vec![vec![1., 0., 1., 1., 2., 1.], vec![1., 1., -1., 0., 0., 0.]],
        b: vec![vec![0.; 3], vec![bias, 0.]],
        dims: vec![2, 3, 2],
        input: vec![Bound::new(-1., 1.); 2],
        bounds: vec![vec![
            Bound::new(-1., 1.),
            Bound::new(-2., 2.),
            Bound::new(-3., 3.),
        ]],
    }
}
fn corrected() -> Candidate {
    improve(
        head(&[2., 1.], 0.75),
        &[head(&[1., 0.], 0.), head(&[1., 1.], 0.5)],
        &[2., 0.5],
        &[-1., -1.],
        &[1., 1.],
        &[-0.5, 1.],
        end(),
        &mut || true,
    )
    .unwrap()
}
fn row_candidate(candidate: Candidate) -> kernel::Proposal {
    kernel::Proposal {
        head: 2,
        positive: vec![0, 1],
        lambda: candidate.lambda,
        v: candidate.slope,
    }
}

#[test]
fn priced_chain_exact_separating_step_recovers_a_useful_coefficient_after_one_pass_misses() {
    let cache = Cache::new(&[-1., -1.], &[1., 1.], &[-0.5, 1.], end(), &mut || true).unwrap();
    let chain = cache.chain(&[-0.5, 0.5], 0., end(), &mut || true).unwrap();
    let partners = [head(&[1., 0.], 0.), head(&[1., 1.], 0.5)];
    let g = gradient(&chain, &partners, end(), &mut || true).unwrap();
    assert_eq!(g, vec![5. / 8., 1. / 4.]);
    assert_eq!(g.iter().map(|x| x * x).sum::<f64>(), 29. / 64.);
    assert_eq!(chain.envelope + 0.25, 1.);
    let answer = corrected();
    assert!(answer.attempted && answer.completed && answer.improved);
    close(answer.lambda[0], 43. / 29.);
    close(answer.lambda[1], 17. / 58.);
    close(answer.value, 43. / 58.);
    assert!(answer.value < 0.75 - 1. / 117.);
    // Independent exact fixed-mean formula at the two forced y=1 vertices.
    let [a, b] = [answer.lambda[0], answer.lambda[1]];
    let exact_price = b / 2. + 0.75 * (a - 1.).max(0.) + 0.25 * (3. - a - 2. * b).max(0.);
    close(answer.value, exact_price);
    assert_eq!(0.5 + 0.75 * 0f64.max(0.) + 0.25 * 0f64.max(0.), 0.5);
}

#[test]
fn priced_chain_zero_residual_coordinate_retains_its_mean_in_every_subgradient() {
    let cache = Cache::new(&[-1.], &[1.], &[0.5], end(), &mut || true).unwrap();
    let chain = cache.chain(&[0.], 1., end(), &mut || true).unwrap();
    assert_eq!(chain.mass, 1.);
    assert_eq!(chain.moments, vec![0.5]);
    assert_eq!(chain.slope, vec![0.]);
    assert_eq!(
        gradient(&chain, &[head(&[1.], 0.5)], end(), &mut || true).unwrap(),
        vec![0.]
    );
    // Omitting this mean would invent the false subgradient +1/2.
    assert_ne!(chain.moments[0], 0.);
    let tie = cache.chain(&[-0.], 0., end(), &mut || true).unwrap();
    assert_eq!((tie.mass, tie.moments[0]), (0.5, 0.25));
}

#[test]
fn priced_chain_fixed_coordinates_and_zero_layers_match_explicit_vertex_moments() {
    let lower = [-2., 1., 0.5, -1.];
    let upper = [2., 3., 0.5, 1.];
    let point = [-1., 2., 0.5, 0.5];
    let c = [-0.5, 1., 2., 0.];
    let cache = Cache::new(&lower, &upper, &point, end(), &mut || true).unwrap();
    let chain = cache.chain(&c, -2., end(), &mut || true).unwrap();
    let order = cache.order(&c, end(), &mut || true).unwrap();
    let mut vertex: Vec<_> = c
        .iter()
        .enumerate()
        .map(|(i, &a)| if a < 0. { upper[i] } else { lower[i] })
        .collect();
    let (mut mass, mut envelope, mut moments) = (0., 0., vec![0.; 4]);
    for rank in 0..=order.len() {
        let previous = if rank == 0 {
            1.
        } else {
            cache.theta(&c, order[rank - 1])
        };
        let next = order.get(rank).map_or(0., |&i| cache.theta(&c, i));
        let weight = previous - next;
        let residual = -2. + c.iter().zip(&vertex).map(|(a, x)| a * x).sum::<f64>();
        let sigma = if residual > 0. {
            1.
        } else if residual < 0. {
            0.
        } else {
            0.5
        };
        mass += weight * sigma;
        envelope += weight * residual.max(0.);
        for i in 0..4 {
            moments[i] += weight * sigma * vertex[i];
        }
        if let Some(&i) = order.get(rank) {
            vertex[i] = if c[i] < 0. { lower[i] } else { upper[i] };
        }
    }
    close(chain.mass, mass);
    close(chain.envelope, envelope);
    for (actual, expected) in chain.moments.iter().zip(moments) {
        close(*actual, expected);
    }
    close(chain.moments[2], 0.5 * chain.mass);
}

#[test]
fn priced_chain_cached_sign_merge_matches_declared_nominal_t_order_for_all_signs_and_ties() {
    let cache = Cache::new(
        &[-1.; 4],
        &[1.; 4],
        &[-1., -0.5, -0.5, 1.],
        end(),
        &mut || true,
    )
    .unwrap();
    assert_eq!(cache.descending, vec![3, 1, 2, 0]);
    assert_eq!(cache.ascending, vec![0, 1, 2, 3]);
    for mask in 0usize..16 {
        let c: Vec<_> = (0..4)
            .map(|i| if mask & (1 << i) == 0 { 1. } else { -1. })
            .collect();
        let actual = cache.order(&c, end(), &mut || true).unwrap();
        let mut expected: Vec<_> = (0..4).collect();
        expected.sort_by(|&i, &j| {
            cache
                .theta(&c, j)
                .total_cmp(&cache.theta(&c, i))
                .then_with(|| i.cmp(&j))
        });
        assert_eq!(actual, expected);
    }
    let zero = cache.order(&[0.; 4], end(), &mut || true).unwrap();
    assert_eq!(zero, cache.descending);
    let tiny_point = [2f64.powi(-54), 2f64.powi(-55)];
    let tiny = Cache::new(&[0.; 2], &[1.; 2], &tiny_point, end(), &mut || true).unwrap();
    assert_eq!(tiny.theta(&[-1.; 2], 0), tiny.theta(&[-1.; 2], 1));
    // Collapsed oriented ties retain ascending ORIGINAL t, deterministically.
    assert_eq!(
        tiny.order(&[-1.; 2], end(), &mut || true).unwrap(),
        vec![1, 0]
    );
}

#[test]
fn priced_chain_projected_step_can_activate_a_funded_zero_seed_partner() {
    let answer = improve(
        head(&[1., 1.], 0.75),
        &[head(&[1., 1.], 0.)],
        &[0.],
        &[-1.; 2],
        &[1.; 2],
        &[0.; 2],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(answer.attempted && answer.completed && answer.improved);
    assert!(answer.lambda[0] > 0. && answer.lambda[0] <= 256.);
    assert!(answer.value < 0.75);
}

#[test]
fn priced_chain_empty_target_gap_zero_norm_and_trial_overflow_keep_the_valid_seed() {
    let no_gap = improve(
        head(&[1.], 1.),
        &[head(&[1.], 0.)],
        &[1.],
        &[-1.],
        &[1.],
        &[0.],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(!no_gap.attempted && !no_gap.completed && !no_gap.improved);
    let zero_norm = improve(
        Head {
            weights: &[0.],
            bias: 1.,
            value: 0.5,
        },
        &[head(&[0.], 0.)],
        &[1.],
        &[-1.],
        &[1.],
        &[0.],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(zero_norm.attempted && !zero_norm.completed && !zero_norm.improved);
    assert_eq!((zero_norm.value, zero_norm.lambda), (1., vec![1.]));
    let overflow = improve(
        Head {
            weights: &[0.],
            bias: 1.,
            value: 0.5,
        },
        &[head(&[0.], f64::MAX)],
        &[0.],
        &[-1.],
        &[1.],
        &[0.],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(overflow.attempted && !overflow.completed && !overflow.improved);
    assert_eq!((overflow.value, overflow.lambda), (1., vec![0.]));
    let worse = improve(
        Head {
            weights: &[0.],
            bias: 1.,
            value: 0.5,
        },
        &[Head {
            weights: &[0.],
            bias: 1.,
            value: 0.9,
        }],
        &[0.99],
        &[-1.],
        &[1.],
        &[0.],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(worse.attempted && worse.completed && !worse.improved);
    assert_eq!(worse.lambda, vec![0.99]);
    close(worse.value, 0.901); // A target below the minimum can overshoot.
    assert_eq!(trial(&[0.], &[1.], 2., 0.).unwrap()[0], 0.); // Projection cannot go negative.
    assert_eq!(trial(&[256.], &[-1.], 2., 0.).unwrap(), vec![256.]);
}

#[test]
fn priced_chain_nonfinite_shape_outside_box_and_lost_increment_refuse() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(improve(
            head(&[1.], 0.),
            &[head(&[1.], 0.)],
            &[bad],
            &[-1.],
            &[1.],
            &[0.],
            end(),
            &mut || true
        )
        .is_none());
        assert!(Cache::new(&[-1.], &[1.], &[bad], end(), &mut || true).is_none());
    }
    assert!(Cache::new(&[1.], &[-1.], &[0.], end(), &mut || true).is_none());
    assert!(Cache::new(&[-1.], &[1.], &[2.], end(), &mut || true).is_none());
    assert!(Cache::new(&[-1.; 129], &[1.; 129], &[0.; 129], end(), &mut || true).is_none());
    assert!(improve(
        head(&[1.], 0.),
        &[],
        &[],
        &[-1.],
        &[1.],
        &[0.],
        end(),
        &mut || true
    )
    .is_none());
    let subnormal = [f64::from_bits(1)];
    let cache = Cache::new(&[0.], &subnormal, &[0.], end(), &mut || true).unwrap();
    assert!(cache
        .chain(&[f64::from_bits(1)], 0., end(), &mut || true)
        .is_none());
    let constant = Cache::new(&[], &[], &[], end(), &mut || true).unwrap();
    let result = constant.chain(&[], -1., end(), &mut || true).unwrap();
    assert_eq!((result.mass, result.envelope), (0., 0.));
}

#[test]
fn priced_chain_cancellation_at_final_reprice_never_returns_a_partial_or_renews_time() {
    let call = |poll: &mut dyn FnMut() -> bool| {
        improve(
            head(&[2., 1.], 0.75),
            &[head(&[1., 0.], 0.), head(&[1., 1.], 0.5)],
            &[2., 0.5],
            &[-1.; 2],
            &[1.; 2],
            &[-0.5, 1.],
            end(),
            poll,
        )
    };
    let mut count = 0;
    assert!(
        call(&mut || {
            count += 1;
            true
        })
        .unwrap()
        .completed
    );
    let mut seen = 0;
    assert!(call(&mut || {
        seen += 1;
        seen < count
    })
    .is_none());
    assert_eq!(seen, count);
    let mut transient = 0;
    assert!(call(&mut || {
        transient += 1;
        transient != count - 1
    })
    .is_none());
    assert_eq!(transient, count - 1); // No repoll revives a false callback.
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(improve(
        head(&[1.], 0.),
        &[head(&[1.], 0.)],
        &[1.],
        &[-1.],
        &[1.],
        &[0.],
        expired,
        &mut || true
    )
    .is_none());
    // The actual enclosing dense nominal routine must not revive the callback
    // through its scalar fallback. These are advice-only calls, no token/check.
    let p = example(0.125).source();
    let cuts = fixture::cuts(&p, &[2]);
    let row = cuts.signed.as_ref().unwrap();
    let paid = dense::quote_with_aliases(&p, 1, 0, 2, 0, 2).unwrap();
    let point = crate::single_hidden_hull::primal_advice::Point {
        input: vec![-0.5, 1.].into_boxed_slice(),
        values: vec![0., 0.5, 0.75].into_boxed_slice(),
        negative: vec![2].into_boxed_slice(),
        positive: vec![0, 1].into_boxed_slice(),
    };
    let mut total = 0;
    assert!(dense::advice(row, &cuts, &point, &paid, end(), &mut || {
        total += 1;
        true
    })
    .is_some_and(|advice| advice.useful));
    let mut seen = 0;
    assert!(dense::advice(row, &cuts, &point, &paid, end(), &mut || {
        seen += 1;
        seen != total - 2
    })
    .is_none_or(|advice| !advice.useful));
    assert_eq!(seen, total - 2);
}

#[test]
fn priced_chain_fresh_source_row_excludes_the_original_encoder_point_and_preserves_closed_ties() {
    let f = example(0.125);
    let p: HullMarginProblem = f.source();
    let original = f.parts();
    let columns =
        crate::single_hidden_hull::mip_cuts::binding::head_columns(&f.encoding(), &f.original())
            .unwrap();
    let mut fake = vec![Q::zero(); original.num_cols];
    fake[original.input_vars[0].0] = q(-0.5);
    fake[original.input_vars[1].0] = q(1.);
    for (j, z) in [-0.5, 0.5, 0.].into_iter().enumerate() {
        fake[2 + j] = q(z);
    }
    for (j, z) in [0., 0.5, 0.75].into_iter().enumerate() {
        fake[columns[j].0] = q(z);
    }
    // Positive selector columns remain in the relaxed IR; locate via original.
    for (j, col) in f.original().binary_vars.iter().enumerate() {
        fake[col.0] = q([0., 1., 0.5][j]);
    }
    fake[original.output_vars[0].0] = q(-0.125);
    rows_hold(&original.problem, &fake, None);
    let old = kernel::Proposal {
        head: 2,
        positive: vec![0, 1],
        lambda: vec![2., 0.5],
        v: vec![-0.5, 0.],
    };
    let old_upper = kernel::upper(&p, &old, end(), &mut || true).unwrap();
    let row = row_candidate(corrected());
    let upper = kernel::upper(&p, &row, end(), &mut || true).unwrap();
    let lhs = |a: &kernel::Proposal| {
        q(0.75) - q(a.lambda[1]) * q(0.5) - q(a.v[0]) * q(-0.5) - q(a.v[1]) * q(1.)
    };
    assert!(lhs(&old) <= q(old_upper));
    assert!(lhs(&row) > q(upper));
    for x in [-1., -0.5, 0., 0.5, 1.] {
        for y in [-1., -0.5, 0., 0.5, 1.] {
            let exact = (q(2.) * q(x) + q(y)).max(Q::zero())
                - q(row.lambda[0]) * q(x).max(Q::zero())
                - q(row.lambda[1]) * (q(x) + q(y)).max(Q::zero())
                - q(row.v[0]) * q(x)
                - q(row.v[1]) * q(y);
            assert!(exact <= q(upper));
        }
    }
    for bias in [0., -0.125] {
        let f = example(bias);
        let mut unsafe_ir = f.parts();
        let mut witness = vec![Q::zero(); unsafe_ir.num_cols];
        witness[unsafe_ir.output_vars[0].0] = q(bias);
        unsafe_ir.problem.add_row(
            f64::NEG_INFINITY,
            upper,
            [
                (columns[2], 1.),
                (columns[0], -row.lambda[0]),
                (columns[1], -row.lambda[1]),
                (unsafe_ir.input_vars[0], -row.v[0]),
                (unsafe_ir.input_vars[1], -row.v[1]),
            ],
        );
        rows_hold(&unsafe_ir.problem, &witness, None);
    }
}

#[test]
fn priced_chain_extra_work_and_payload_are_prepaid_even_for_a_no_step_seed() {
    let p = example(0.125).source();
    let quote = dense::quote_with_aliases(&p, 1, 0, 2, 0, 2).unwrap();
    assert_eq!((quote.step_work, quote.step_bytes), (7_936, 2_368));
    assert!(quote.total_work <= super::super::resources::MAX_DOMINATION_WORK);
    let no_step = improve(
        head(&[1., 0.], 1.),
        &[head(&[1., 0.], 0.)],
        &[1.],
        &[-1.; 2],
        &[1.; 2],
        &[0.; 2],
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(!no_step.attempted);
    assert!(quote.step_work > 0 && quote.step_bytes > 0);
    assert!(dense::quote_with_aliases(&p, 1, 664_696, 2, 0, 2).is_none());
}
