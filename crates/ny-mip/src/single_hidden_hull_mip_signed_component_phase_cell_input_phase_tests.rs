// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! External controls for cell::input_phase; no production admission is tested.
use super::*;
use certificate::I;
use num_traits::ToPrimitive;

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
    source: HullMarginProblem,
    component: Component,
}
impl Fixture {
    fn new(k: usize) -> Self {
        assert!((1..=2).contains(&k));
        let h = k + 1;
        let mut head_bias = vec![0.25];
        head_bias.extend(vec![-0.25; k]);
        let mut coefficients = vec![1.];
        coefficients.extend(vec![-2. / k as f64; k]);
        let mut second = coefficients.clone();
        second.resize(2 * h, 0.);
        let mut bounds = vec![Bound::new(-0.75, 1.25)];
        bounds.extend(vec![Bound::new(-1.25, 0.75); k]);
        // Literal dyadic Source: this is a valid exact lower margin of the
        // original network, but deliberately does not call the constructor's
        // downward output reduction. That reduction is covered by the separate
        // captured-original-source replay, not by this exact tie fixture.
        let source = HullMarginProblem {
            input_dim: 1,
            hidden_dim: h,
            w1: vec![1.; h],
            b1: head_bias.clone(),
            c: coefficients,
            d: 0.,
            lower: vec![-1.],
            upper: vec![1.],
        };
        let component = Component {
            negative: (1..h).collect(),
            positive: vec![0],
            positive_units: vec![2],
            g: vec![0.],
            // Half-alpha mixed masks are essential: [0,0,0,1] selects a
            // different base pair whose Ray residuals both vanish.
            alpha: if k == 1 {
                vec![0., 1.]
            } else {
                vec![0., 0.5, 0.5, 1.]
            },
        };
        Self {
            weights: vec![vec![1.; h], second],
            biases: vec![head_bias, vec![0., 0.]],
            dims: vec![1, h, 2],
            bounds: vec![bounds],
            source,
            component,
        }
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &[Bound::new(-1., 1.)],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}

/// Reconstruct a finite affine witness with exact rational arithmetic,
/// independently of the producer's interval accumulation and kernel advice.
fn exact_affine(p: &HullMarginProblem, c: &Component, mask: usize) -> (Vec<I>, I) {
    let mut q: Vec<Q> = c.g.iter().map(|&g| -rat(g)).collect();
    let mut constant = Q::zero();
    let mut add = |head: usize, coefficient: Q| {
        constant += &coefficient * rat(p.b1[head]);
        for (i, value) in q.iter_mut().enumerate() {
            *value += &coefficient * rat(p.w_row(head)[i]);
        }
    };
    for (slot, &head) in c.positive.iter().enumerate() {
        let mass = rat(p.c[head]) * rat(c.positive_units[slot] as f64) / rat(2.);
        add(head, mass * rat(c.alpha[mask * c.positive.len() + slot]));
    }
    for (bit, &head) in c.negative.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            add(head, rat(p.c[head]));
        }
    }
    let point = |value: Q| {
        let float = value.to_f64().unwrap();
        assert_eq!(
            rat(float),
            value,
            "fixture affine must be exactly representable"
        );
        I::point(float).unwrap()
    };
    (q.into_iter().map(point).collect(), point(constant))
}

fn check_row(row: &RowSpec, point: &[Q], label: &str) {
    let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
    assert!(!row.lb.is_finite() || lhs >= rat(row.lb), "{label}: lower");
    assert!(!row.ub.is_finite() || lhs <= rat(row.ub), "{label}: upper");
}

fn separation(k: usize) {
    let fixture = Fixture::new(k);
    let (p, c) = (&fixture.source, &fixture.component);
    let masks = 1usize << k;
    let mut scratch = Vec::with_capacity(masks);
    assert_eq!(scratch.capacity(), masks);
    let proposals =
        input_phase::propose(p, c, &[3. / 8.], &mut scratch, end(), &mut || true).unwrap();
    assert_eq!(scratch, [] as [f64; 0]);
    assert_eq!(scratch.capacity(), masks);
    assert_eq!(proposals, [vec![-1.], vec![-0.5], vec![-1.], vec![-0.5]]);

    let expected_base = if k == 1 {
        vec![0., -0.25]
    } else {
        vec![0., -0.125, -0.125, -0.25]
    };
    let expected_ray = if k == 1 {
        vec![0., -0.25]
    } else {
        vec![0., 0.25, 0.25, -0.25]
    };
    let expected_full = if k == 1 {
        vec![-1., 0.75]
    } else {
        vec![-1., -0.125, -0.125, 0.75]
    };
    let pairs = numeric::pairs(p, c, end(), &mut || true).unwrap();
    let mut tables = vec![Vec::new(); 4];
    for mask in 0..masks {
        let (q, constant) = exact_affine(p, c, mask);
        let base = numeric::price(p, &q, constant).unwrap();
        assert!(base <= expected_base[mask] && expected_base[mask] - base < 1e-12);
        let (cell, _) =
            numeric::strengthen(p, c, mask, &q, constant, base, &pairs, end(), &mut || true)
                .unwrap();
        let ray = ray::strengthen(p, c, mask, &q, constant, cell, end(), &mut || true).unwrap();
        let joint =
            joint::strengthen(p, c, mask, &q, constant, ray, &pairs, end(), &mut || true).unwrap();
        assert!(ray <= expected_ray[mask] && expected_ray[mask] - ray < 1e-12);
        assert_eq!(joint.to_bits(), ray.to_bits()); // Only one positive slot.
        let saved = ray::proposal(p, c, mask, &q, &mut || true);
        for (index, new_g) in proposals.iter().enumerate() {
            let value =
                input_phase::price(p, c, mask, &q, constant, new_g, saved, end(), &mut || true)
                    .unwrap();
            if index == 0 {
                assert!(value <= expected_full[mask] && expected_full[mask] - value < 1e-12);
            }
            tables[index].push(value);
        }
    }

    let encoding = fixture.encoding();
    let mut parts = fixture.original();
    let heads = binding::head_columns(&encoding, &parts).unwrap();
    let prefix = parts.problem.rows().to_vec();
    let mut fake = vec![Q::zero(); parts.num_cols];
    fake[parts.input_vars[0].0] = rat(3. / 8.);
    for j in 0..p.hidden_dim {
        fake[1 + j] = rat(3. / 8.) + rat(p.b1[j]);
        fake[heads[j].0] = rat(if j == 0 { 5. / 8. } else { 13. / 32. });
        fake[heads[j].0 + 1] = rat(if j == 0 { 0.5 } else { 0.75 });
    }
    fake[parts.output_vars[0].0] = rat(-3. / 16.);
    fake[parts.output_vars[1].0] = Q::zero();
    assert!(feasible(&parts, &fake)); // Includes the actual unsafe output row.

    // Universal fixed-g support control: a feasible table mixture, mass1/4
    // on mask0 and3/4 on all-active, has every marginal3/4 and cost-3/16.
    // Those two floors are attained at original graph x=-1 andx=1. Any valid
    // phase-only support is bounded by this mixture, exactly met by fake F.
    for bit in 0..k {
        let marginal = rat(0.25) * rat(0.) + rat(0.75) * rat(((masks - 1) >> bit & 1) as f64);
        assert_eq!(marginal, fake[heads[1 + bit].0 + 1]);
    }
    let old_ceiling = rat(0.25) * rat(0.) + rat(0.75) * rat(-0.25);
    assert_eq!(old_ceiling, fake[parts.output_vars[0].0]);

    for (index, new_g) in proposals.iter().enumerate() {
        // Known exact supports for these dyadic full/half fixtures. Selection
        // is independent of the kernel, which returns no selector support.
        let a = vec![(4. * new_g[0] + 0.5) / k as f64; k];
        let lower = repair(&tables[index], &a, end(), &mut || true).unwrap();
        assert!(lower <= 2. * new_g[0] && lower > 2. * new_g[0] - 1e-12);
        let mut terms = vec![(parts.input_vars[0], -2. * new_g[0])];
        for (j, head) in heads.iter().enumerate() {
            terms.push((*head, 2. * p.c[j]));
        }
        for (bit, coefficient) in a.iter().enumerate() {
            terms.push((crate::ir::Col(heads[1 + bit].0 + 1), *coefficient));
        }
        parts.problem.add_row(lower, f64::INFINITY, terms);
        let row = parts.problem.rows().last().unwrap();
        let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &fake[*j]).sum();
        let expected_violation = if index % 2 == 0 { 0.25 } else { 0.125 };
        assert!(rat(lower) - lhs > rat(expected_violation - 1e-12));
    }
    assert_eq!(&parts.problem.rows()[..prefix.len()], prefix.as_slice());
    assert!(!feasible(&parts, &fake));

    // Original encoder, all linear-piece endpoints, both zero hyperplanes,
    // and every closed-selector combination when duplicate heads equal zero.
    for x in [-1., -0.25, 0., 0.25, 0.75, 1.] {
        for ties in 0..1usize << p.hidden_dim {
            let mut point = vec![Q::zero(); parts.num_cols];
            point[parts.input_vars[0].0] = rat(x);
            let mut mask = 0usize;
            for j in 0..p.hidden_dim {
                let z = rat(x) + rat(p.b1[j]);
                let selector = if z.is_zero() {
                    (ties >> j) & 1
                } else {
                    usize::from(z > Q::zero())
                };
                point[1 + j] = z.clone();
                point[heads[j].0] = z.max(Q::zero());
                point[heads[j].0 + 1] = rat(selector as f64);
                if j != 0 {
                    mask |= selector << (j - 1);
                }
            }
            let f: Q = (0..p.hidden_dim)
                .map(|j| rat(p.c[j]) * &point[heads[j].0])
                .sum();
            point[parts.output_vars[0].0] = f.clone();
            for (j, column) in parts.problem.cols().iter().enumerate() {
                assert!(!column.lb.is_finite() || point[j] >= rat(column.lb));
                assert!(!column.ub.is_finite() || point[j] <= rat(column.ub));
            }
            for (row_index, row) in parts.problem.rows().iter().enumerate() {
                if row_index != encoding.unsafe_row.0 {
                    check_row(
                        row,
                        &point,
                        &format!("k={k} x={x} ties={ties} row={row_index}"),
                    );
                }
            }
            for (index, new_g) in proposals.iter().enumerate() {
                assert!(&f - rat(new_g[0]) * rat(x) >= rat(tables[index][mask]));
            }
        }
    }
}

#[test]
fn input_phase_k1_zero_face_fallback_separates_original_encoder() {
    separation(1);
}

#[test]
fn input_phase_k2_half_alpha_tie_separates_original_encoder_and_all_zero_choices() {
    separation(2);
}

#[test]
fn input_phase_actual_stored_g_controls_rounding_not_nominal_delta() {
    // Exact Source H(x)=-ReLU(-x)=x on [-2,-1], active mask1.
    let p = HullMarginProblem {
        input_dim: 1,
        hidden_dim: 1,
        w1: vec![-1.],
        b1: vec![0.],
        c: vec![-1.],
        d: 0.,
        lower: vec![-2.],
        upper: vec![-1.],
    };
    let c = Component {
        negative: vec![0],
        positive: vec![],
        positive_units: vec![],
        g: vec![1.],
        alpha: vec![],
    };
    let (q, constant) = exact_affine(&p, &c, 1);
    assert_eq!(q[0].lo, 0.);
    let half_ulp = f64::from_bits((1023u64 - 53) << 52);
    let actual_new_g = 1. + half_ulp;
    assert_eq!(actual_new_g, c.g[0]);
    let checked = input_phase::price(
        &p,
        &c,
        1,
        &q,
        constant,
        &[actual_new_g],
        None,
        end(),
        &mut || true,
    )
    .unwrap();
    assert!(checked <= 0. && checked.abs() < 1e-300);
    // Pricing the unrepresented delta would falsely demand0>=2^-53.
    assert!(rat(half_ulp) > Q::zero());
    for x in [-2., -1.5, -1.] {
        let h = -(-rat(x)).max(Q::zero());
        assert_eq!(&h - rat(actual_new_g) * rat(x), Q::zero());
        assert!(&h - rat(actual_new_g) * rat(x) >= rat(checked));
    }
    // A real representable one-ulp change must be priced as that exact value.
    let new_g = f64::from_bits(1f64.to_bits() + 1);
    let checked =
        input_phase::price(&p, &c, 1, &q, constant, &[new_g], None, end(), &mut || true).unwrap();
    let exact_minimum = rat(new_g) - rat(1.);
    assert!(rat(checked) <= exact_minimum && checked > 0.);
}

#[test]
fn input_phase_rejects_illegal_ray_head_sign_and_nonfinite_advice() {
    let f = Fixture::new(2);
    let (p, c) = (&f.source, &f.component);
    for mask in 0..4 {
        let (q, constant) = exact_affine(p, c, mask);
        for bad in [
            (0, 0.),
            (99, 0.),
            (1, f64::NAN),
            (1, f64::INFINITY),
            (1, f64::NEG_INFINITY),
            (1, if mask & 1 == 0 { -0.25 } else { 0.25 }),
        ] {
            assert!(input_phase::price(
                p,
                c,
                mask,
                &q,
                constant,
                &[-1.],
                Some(bad),
                end(),
                &mut || true
            )
            .is_none());
        }
        for zero in [0., -0.] {
            assert!(input_phase::price(
                p,
                c,
                mask,
                &q,
                constant,
                &[-1.],
                Some((1, zero)),
                end(),
                &mut || true
            )
            .is_some());
        }
    }
    let (q, constant) = exact_affine(p, c, 0);
    for coefficient in [0., 1., f64::NAN, f64::NEG_INFINITY] {
        let mut malformed = p.clone();
        malformed.c[1] = coefficient;
        assert!(input_phase::price(
            &malformed,
            c,
            0,
            &q,
            constant,
            &[-1.],
            Some((1, 0.)),
            end(),
            &mut || true
        )
        .is_none());
    }
}

#[test]
fn input_phase_cancellation_is_sticky_at_every_poll_and_scratch_clears() {
    let f = Fixture::new(2);
    let (p, c) = (&f.source, &f.component);
    let mut scratch = Vec::with_capacity(4);
    let mut count = 0usize;
    assert!(
        input_phase::propose(p, c, &[0.375], &mut scratch, end(), &mut || {
            count += 1;
            true
        })
        .is_some()
    );
    assert!(count > 4 && scratch.is_empty());
    for stop in 1..=count {
        let mut calls = 0usize;
        assert!(
            input_phase::propose(p, c, &[0.375], &mut scratch, end(), &mut || {
                calls += 1;
                calls != stop
            })
            .is_none()
        );
        assert_eq!(calls, stop);
        assert_eq!(scratch, [] as [f64; 0]);
        assert_eq!(scratch.capacity(), 4);
    }
    let (q, constant) = exact_affine(p, c, 1);
    let mut count = 0usize;
    assert!(input_phase::price(
        p,
        c,
        1,
        &q,
        constant,
        &[-1.],
        Some((2, 0.5)),
        end(),
        &mut || {
            count += 1;
            true
        }
    )
    .is_some());
    for stop in 1..=count {
        let mut calls = 0usize;
        assert!(input_phase::price(
            p,
            c,
            1,
            &q,
            constant,
            &[-1.],
            Some((2, 0.5)),
            end(),
            &mut || {
                calls += 1;
                calls != stop
            }
        )
        .is_none());
        assert_eq!(calls, stop);
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("existing one-second expired test clock");
    assert!(input_phase::propose(p, c, &[0.375], &mut scratch, expired, &mut || true).is_none());
    assert_eq!(scratch, [] as [f64; 0]);
    assert!(
        input_phase::price(p, c, 1, &q, constant, &[-1.], None, expired, &mut || true).is_none()
    );
}

#[test]
fn input_phase_malformed_shapes_advice_and_scratch_fail_without_panicking() {
    let f = Fixture::new(2);
    let (p, c) = (&f.source, &f.component);
    let rejected = |p: &HullMarginProblem, c: &Component, target: &[f64]| {
        let mut scratch = Vec::with_capacity(4);
        assert!(input_phase::propose(p, c, target, &mut scratch, end(), &mut || true).is_none());
        assert_eq!(scratch, [] as [f64; 0]);
    };
    for target in [
        vec![],
        vec![0., 0.],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![2.],
    ] {
        rejected(p, c, &target);
    }
    for which in 0..12 {
        let mut malformed = p.clone();
        match which {
            0 => malformed.input_dim = 0,
            1 => malformed.input_dim = 33,
            2 => malformed.hidden_dim += 1,
            3 => {
                malformed.w1.pop();
            }
            4 => {
                malformed.b1.pop();
            }
            5 => {
                malformed.c.pop();
            }
            6 => {
                malformed.lower.pop();
            }
            7 => {
                malformed.upper.pop();
            }
            8 => malformed.lower[0] = 2.,
            9 => malformed.upper[0] = f64::INFINITY,
            10 => malformed.w1[0] = f64::NAN,
            11 => malformed.b1[0] = f64::INFINITY,
            _ => unreachable!(),
        }
        rejected(&malformed, c, &[0.375]);
    }
    for which in 0..11 {
        let mut malformed = c.clone();
        match which {
            0 => malformed.negative.clear(),
            1 => malformed.negative.resize(9, 1),
            2 => {
                malformed.positive_units.pop();
            }
            3 => {
                malformed.g.pop();
            }
            4 => {
                malformed.alpha.pop();
            }
            5 => malformed.alpha[3] = f64::NAN,
            6 => malformed.alpha[3] = 1.5,
            7 => malformed.positive[0] = 99,
            8 => malformed.negative[0] = 99,
            9 => malformed.g[0] = f64::NAN,
            10 => malformed.positive_units[0] = 3,
            _ => unreachable!(),
        }
        rejected(p, &malformed, &[0.375]);
    }
    let mut occupied = Vec::with_capacity(4);
    occupied.push(17.);
    assert!(input_phase::propose(p, c, &[0.375], &mut occupied, end(), &mut || true).is_none());
    assert_eq!(occupied, [17.]); // Entry refusal must preserve caller data.
    let mut wrong_capacity = Vec::with_capacity(5);
    assert!(
        input_phase::propose(p, c, &[0.375], &mut wrong_capacity, end(), &mut || true).is_none()
    );
    assert_eq!(wrong_capacity, [] as [f64; 0]);
    let (q, constant) = exact_affine(p, c, 0);
    for (mask, original, constant, new_g) in [
        (4, q.clone(), constant, vec![-1.]),
        (0, vec![], constant, vec![-1.]),
        (0, q.clone(), constant, vec![]),
        (0, q.clone(), constant, vec![f64::NAN]),
        (0, vec![I { lo: 1., hi: 0. }], constant, vec![-1.]),
        (
            0,
            vec![I {
                lo: f64::NEG_INFINITY,
                hi: 0.,
            }],
            constant,
            vec![-1.],
        ),
        (0, q.clone(), I { lo: 1., hi: 0. }, vec![-1.]),
        (
            0,
            q,
            I {
                lo: 0.,
                hi: f64::INFINITY,
            },
            vec![-1.],
        ),
    ] {
        assert!(input_phase::price(
            p,
            c,
            mask,
            &original,
            constant,
            &new_g,
            None,
            end(),
            &mut || true
        )
        .is_none());
    }
}

#[test]
fn input_phase_no_direction_is_an_honest_refusal_and_retracted_alpha_stays_a_control() {
    let mut f = Fixture::new(1);
    f.component.alpha = vec![1., 1.]; // Residuals+1/-1, no common or zero face.
    let mut scratch = Vec::with_capacity(2);
    assert!(input_phase::propose(
        &f.source,
        &f.component,
        &[0.375],
        &mut scratch,
        end(),
        &mut || true
    )
    .is_none());
    assert_eq!(scratch, [] as [f64; 0]);
    let mut f = Fixture::new(2);
    f.component.alpha = vec![0., 0., 0., 1.];
    let mut scratch = Vec::with_capacity(4);
    assert!(input_phase::propose(
        &f.source,
        &f.component,
        &[0.375],
        &mut scratch,
        end(),
        &mut || true
    )
    .is_none());
    assert_eq!(scratch, [] as [f64; 0]);
}
