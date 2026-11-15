// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! UNRUN supplied numeric controls; child of native_publication::numeric.
//! Analytic Plane construction here is test-only. No Admission, Published,
//! BoundPublished, paid flag or production row license is manufactured.
use super::super::Term;
use super::*;
use num_rational::BigRational as Q;
use num_traits::{FromPrimitive, Zero};
use std::time::Duration;
fn q(v: f64) -> Q {
    Q::from_f64(v).unwrap()
}
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn source() -> HullMarginProblem {
    let e = 1. / 1024.;
    HullMarginProblem {
        input_dim: 3,
        hidden_dim: 4,
        w1: vec![1., 0., 0., 0., 1., 0., 1., 0., e, 0., 1., 2. * e],
        b1: vec![0.; 4],
        c: vec![1., 1., -1., -1.],
        d: 11. / 16.,
        lower: vec![-1.; 3],
        upper: vec![1.; 3],
    }
}
// Original-column labels are deliberately not authenticated here: this leaf
// tests arithmetic on supplied analytic rows, never producer authority.
fn analytic_plane(lower: f64, terms: Vec<(Atom, f64)>) -> Plane {
    Plane {
        lower_bits: lower.to_bits(),
        terms: terms
            .into_iter()
            .enumerate()
            .map(|(i, (atom, a))| Term {
                atom,
                coefficient_bits: a.to_bits(),
                original_column: i,
            })
            .collect(),
    }
}
fn frame<'a>(
    p: &'a HullMarginProblem,
    planes: &'a [Plane; 2],
    data: &'a Enclosure,
    until: Instant,
) -> SourceFrame<'a> {
    SourceFrame::new(p, [&planes[0], &planes[1]], data, until, &mut || true).unwrap()
}
fn planes() -> [Plane; 2] {
    let e = 1. / 1024.;
    let common = || {
        vec![
            (Atom::Head(0), 2.),
            (Atom::Head(1), 2.),
            (Atom::Head(2), -2.),
            (Atom::Head(3), -2.),
        ]
    };
    let mut unary = common();
    unary.push((Atom::Selector(2), 2. - 2. * e));
    let mut cube = common();
    cube.push((Atom::Selector(2), 0.5 + 2. * e));
    cube.push((Atom::Selector(3), -1.5 + 4. * e));
    [analytic_plane(-4. * e, unary), analytic_plane(-1.5, cube)]
}
fn head_value(n: &Node<'_, '_>, j: usize, x: &[Q]) -> Q {
    let h = n.heads[j];
    let p = n.source.p;
    match h.kind & 7 {
        ZERO => Q::zero(),
        LIVE => {
            q(p.b1[j])
                + (0..p.input_dim)
                    .map(|i| q(p.w_row(j)[i]) * &x[i])
                    .sum::<Q>()
        }
        POS => x[h.column as usize].clone(),
        NEG => {
            q(n.source.bounds[j].upper) * (q(1.) - &x[h.column as usize])
                - (0..n.source.heads[j].len as usize)
                    .map(|k| x[n.term(j, k).unwrap().1].clone())
                    .sum::<Q>()
        }
        _ => panic!("fixture maps all heads"),
    }
}
fn selector(n: &Node<'_, '_>, j: usize, x: &[Q]) -> Q {
    match n.heads[j].kind & 7 {
        ZERO => q(0.),
        LIVE => q(1.),
        NEG => q(1.) - &x[n.heads[j].column as usize],
        _ => panic!("negative selector"),
    }
}
fn plane_slack(n: &Node<'_, '_>, r: usize, x: &[Q]) -> Q {
    let plane = &n.source.planes[r];
    plane
        .terms()
        .iter()
        .map(|term| {
            let (atom, a) = term.semantic();
            q(a) * match atom {
                Atom::Input(i) => x[i].clone(),
                Atom::Head(j) => head_value(n, j, x),
                Atom::Selector(j) => selector(n, j, x),
            }
        })
        .sum::<Q>()
        - q(plane.lower())
}
fn check_box_and_base(n: &Node<'_, '_>, x: &[Q]) {
    let p = n.source.p;
    assert_eq!(x.len(), n.columns);
    for i in 0..p.input_dim {
        assert!(x[i] >= q(p.lower[i]) && x[i] <= q(p.upper[i]));
    }
    for (j, h) in n.heads.iter().enumerate() {
        match h.kind & 7 {
            POS => {
                let s = &x[h.column as usize];
                assert!(*s >= q(0.) && *s <= q(n.source.bounds[j].upper));
                assert!(
                    s - q(p.b1[j])
                        - (0..p.input_dim)
                            .map(|i| q(p.w_row(j)[i]) * &x[i])
                            .sum::<Q>()
                        >= q(0.)
                );
            }
            NEG => {
                let mu = &x[h.column as usize];
                assert!(*mu >= q(0.) && *mu <= q(1.));
                let range = q(I::point(n.source.bounds[j].upper)
                    .unwrap()
                    .sub(I::point(n.source.bounds[j].lower).unwrap())
                    .unwrap()
                    .hi);
                for k in 0..n.source.heads[j].len as usize {
                    let (i, c, _) = n.term(j, k).unwrap();
                    let t = &x[c];
                    assert!(*t >= q(0.) && *t <= range);
                    let w = p.w_row(j)[i];
                    let corner = if w > 0. { p.upper[i] } else { p.lower[i] };
                    assert!(
                        t + q(w.abs()) * (q(p.upper[i]) - q(p.lower[i])) * mu + q(w) * &x[i]
                            - q(w) * q(corner)
                            >= q(0.)
                    );
                }
            }
            _ => {}
        }
    }
}
#[test]
fn supplied_compact_node_cube_prices_a_strict_gain_beyond_all_unary() {
    let p = source();
    let supplied = planes();
    let e = q(1. / 1024.);
    let until = end();
    let data = Enclosure::new(&p, until, &mut || true).unwrap();
    // Analytic Source four-mask affine minima from alpha00/00/01/(3/4,1).
    // These rows are supplied ONLY to this arithmetic control, not minted by
    // a fabricated production publication/license or an LP outcome.
    let table = [q(0.), -q(1.) - &e, -q(2.) * &e, -q(0.25) - q(3.) * &e];
    for (mask, floor) in table.iter().enumerate() {
        let s0 = q((mask & 1) as f64);
        let s1 = q(((mask >> 1) & 1) as f64);
        assert!(q(2.) * floor + (q(2.) - q(2.) * &e) * &s0 >= q(supplied[0].lower()));
        assert!(
            q(2.) * floor + (q(0.5) + q(2.) * &e) * s0 + (-q(1.5) + q(4.) * &e) * s1
                >= q(supplied[1].lower())
        );
    }
    // The borrowed API always carries two rows. An exact 0>=0 second row is
    // equivalent to the old one-unary comparator; it adds no restriction.
    let mut old_planes = planes();
    old_planes[1] = analytic_plane(0., vec![]);
    let shared = frame(&p, &old_planes, &data, until);
    let old = Node::new(&shared, &[], &mut || true).unwrap();
    assert_eq!(old.columns, 11);
    assert_eq!(old.row_count(), Some(8));
    assert!(old.source.planes[1].terms().is_empty());
    assert_eq!(old.source.planes[1].lower(), 0.);
    let mut fake = vec![q(0.); old.columns];
    for (j, activity, output) in [(2, 179. / 256., 77. / 256.), (3, 51. / 128., 51. / 128.)] {
        let mu = old.heads[j].column as usize;
        fake[mu] = q(1.) - q(activity);
        let z = if j == 2 {
            q(1. / 1024.) * (q(1.) - q(2.) * &fake[mu])
        } else {
            Q::zero()
        };
        assert_eq!(old.source.heads[j].len, 2);
        fake[old.term(j, 1).unwrap().1] = z.clone();
        fake[old.term(j, 0).unwrap().1] = q(shared.bounds[j].upper) * q(activity) - q(output) - z;
    }
    check_box_and_base(&old, &fake);
    assert!(plane_slack(&old, 0, &fake) >= q(0.));
    assert_eq!(plane_slack(&old, 1, &fake), q(0.));
    let sum = (0..4)
        .map(|j| q(p.c[j]) * head_value(&old, j, &fake))
        .sum::<Q>();
    assert!(sum >= -q(1.) - &e + (q(0.75) - q(2.) * &e) * selector(&old, 3, &fake)); // omitted Y unary
    assert!(sum >= table.iter().min().unwrap().clone());
    assert_eq!(q(p.d) + sum, -q(3. / 256.));
    let shared = frame(&p, &supplied, &data, until);
    let node = Node::new(&shared, &[], &mut || true).unwrap();
    assert_eq!(node.row_count(), Some(8));
    // Check real existing root compact-builder column/row maps, no LP solve.
    let bounds = crate::single_hidden_hull::preactivation_bounds(&p);
    let roles = crate::single_hidden_hull::classify(&p, &bounds);
    let mut lp = crate::single_hidden_hull::build_hull_lp(&p, &roles, &bounds).unwrap();
    assert!(node.matches_lp(&lp, &[], &mut || true));
    // Exact dyadic evaluation of the ACTUAL old builder's rows/objective,
    // not an LP solve or a manufactured objective from semantic head values.
    for r in 0..lp.model.num_rows() {
        let (coeffs, lower, upper) = lp.model.row(lp.model.row_at(r).unwrap());
        let lhs = coeffs
            .iter()
            .map(|&(c, a)| q(a) * &fake[c as usize])
            .sum::<Q>();
        if lower.is_finite() {
            assert!(lhs >= q(lower));
        }
        if upper.is_finite() {
            assert!(lhs <= q(upper));
        }
    }
    let old_native = q(lp.constant)
        + lp.objective
            .iter()
            .map(|&(c, a)| q(a) * &fake[c.index()])
            .sum::<Q>();
    assert!(old_native < -q(1. / 128.));
    let mut run = PriceRun::new(node).unwrap();
    let cap = 2 * lp.model.num_cols();
    let batch = run.nominal_batch(&lp, &[], cap, &mut || true).unwrap();
    assert_eq!(batch.rows.len(), 2);
    for row in &batch.rows {
        assert!(row
            .terms
            .windows(2)
            .all(|w| w[0].0.index() < w[1].0.index()));
        assert!(row
            .terms
            .iter()
            .all(|&(c, a)| c.index() < run.node.columns && a.is_finite() && a != 0.));
        assert_eq!(row.terms.capacity(), row.terms.len());
    }
    assert!(run.nominal_batch(&lp, &[], cap, &mut || true).is_none());
    batch.append(&mut lp, &mut || true).unwrap();
    assert_eq!(lp.model.num_rows(), 8);
    let y = [1. / 3., 0., 1. / 3., 1. / 3., 0., 0., 1. / 9., 2. / 9.];
    let lower = run.lower(&y, &mut || true).unwrap();
    assert!(lower > 1. / 64., "interval={lower}");
    assert!(lower <= 7. / 384. + 1e-12);
}
#[test]
fn supplied_node_price_keeps_original_end_two_sign_cap_and_sticky_cancel() {
    let p = source();
    let planes = planes();
    let until = end();
    let data = Enclosure::new(&p, until, &mut || true).unwrap();
    let shared = frame(&p, &planes, &data, until);
    let new = || PriceRun::new(Node::new(&shared, &[], &mut || true).unwrap()).unwrap();
    let mut run = new();
    let y = vec![0.; run.node.row_count().unwrap()];
    assert!(run.lower(&y, &mut || true).is_some());
    assert!(run.lower(&y, &mut || true).is_some());
    assert!(run.lower(&y, &mut || true).is_none());
    assert_eq!(run.attempts, 2);
    let mut run = new();
    assert!(run.lower(&y, &mut || false).is_none());
    assert!(run.cancelled);
    assert!(run.lower(&y, &mut || true).is_none());
    let mut run = new();
    let mut polls = 0;
    assert!(run
        .lower(&y, &mut || {
            polls += 1;
            true
        })
        .is_some());
    let count = polls;
    let mut run = new();
    let mut polls = 0;
    assert!(run
        .lower(&y, &mut || {
            polls += 1;
            polls < count
        })
        .is_none());
    assert!(run.cancelled);
    let mut run = new();
    let mut bad = y.clone();
    bad[0] = f64::INFINITY;
    assert!(run.lower(&bad, &mut || true).is_none());
    assert_eq!(run.attempts, 1);
    let mut expired = frame(&p, &planes, &data, until);
    expired.end = Instant::now();
    assert!(Node::new(&expired, &[], &mut || true).is_none()); // no sleep, no renewal
}
#[test]
fn streamed_node_layout_and_scratch_have_literal_capacity_envelopes() {
    use std::mem::size_of;
    assert_eq!(size_of::<Bound>(), 16);
    assert_eq!(size_of::<RootHead>(), 12);
    assert_eq!(size_of::<NodeHead>(), 12);
    assert_eq!(size_of::<SplitSlot>(), 8);
    assert_eq!(size_of::<I>(), 16);
    assert_eq!(size_of::<Atom>(), 16);
    assert_eq!(size_of::<Term>(), 32);
    assert_eq!(size_of::<Plane>(), 32);
    assert_eq!(size_of::<(Col, f64)>(), 16);
    assert_eq!(size_of::<AdviceRow>(), 32);
    assert!(size_of::<AdviceBatch>() <= 128);
    assert!(size_of::<Enclosure>() <= 128);
    assert!(size_of::<SourceFrame<'_>>() <= 512);
    assert!(size_of::<PriceRun<'_, '_>>() <= 1024);
    let p = source();
    let supplied_planes = planes();
    let until = end();
    let data = Enclosure::new(&p, until, &mut || true).unwrap();
    let shared = frame(&p, &supplied_planes, &data, until);
    assert!(std::ptr::eq(
        shared.planes[0],
        std::ptr::from_ref(&supplied_planes[0])
    ));
    assert!(std::ptr::eq(
        shared.planes[1],
        std::ptr::from_ref(&supplied_planes[1])
    ));
    let splits = [HullSplit {
        neuron: 2,
        active: true,
    }];
    let node = Node::new(&shared, &splits, &mut || true).unwrap();
    let run = PriceRun::new(node).unwrap();
    let shared_bytes =
        16 * data.bounds.capacity() + 12 * data.heads.capacity() + 4 * data.inputs.capacity();
    assert!(shared_bytes <= 28 * p.hidden_dim + 4 * p.hidden_dim * p.input_dim);
    let bytes = 12 * run.node.heads.capacity()
        + 8 * run.node.slots.capacity()
        + 16 * run.residual.capacity()
        + 16 * run.heads.capacity()
        + 16 * run.selectors.capacity();
    assert_eq!(
        bytes,
        16 * run.node.columns + 44 * p.hidden_dim + 8 * splits.len()
    );
    assert_eq!(run.node.heads.capacity(), p.hidden_dim);
    assert_eq!(run.residual.capacity(), run.node.columns);
    let mut duplicate = planes();
    duplicate[0].terms.push(Term {
        atom: Atom::Head(0),
        coefficient_bits: 1f64.to_bits(),
        original_column: 0,
    });
    assert!(SourceFrame::new(
        &p,
        [&duplicate[0], &duplicate[1]],
        &data,
        until,
        &mut || true
    )
    .is_none());
}

#[test]
fn nominal_batch_merges_before_model_mutation_and_refuses_overflow_atomically() {
    let p = HullMarginProblem {
        input_dim: 1,
        hidden_dim: 1,
        w1: vec![1.],
        b1: vec![0.],
        c: vec![1.],
        d: 0.,
        lower: vec![1.],
        upper: vec![1.],
    };
    let big = f64::MAX * 0.75;
    let planes = [
        analytic_plane(0., vec![(Atom::Input(0), 1.)]),
        analytic_plane(0., vec![(Atom::Input(0), big), (Atom::Head(0), big)]),
    ];
    let until = end();
    let data = Enclosure::new(&p, until, &mut || true).unwrap();
    let shared = frame(&p, &planes, &data, until);
    let node = Node::new(&shared, &[], &mut || true).unwrap();
    let bounds = crate::single_hidden_hull::preactivation_bounds(&p);
    let roles = crate::single_hidden_hull::classify(&p, &bounds);
    let mut lp = crate::single_hidden_hull::build_hull_lp(&p, &roles, &bounds).unwrap();
    let before = lp.model.num_rows();
    let mut run = PriceRun::new(node).unwrap();
    let cap = 2 * lp.model.num_cols();
    assert!(run.nominal_batch(&lp, &[], cap, &mut || true).is_none());
    assert_eq!(lp.model.num_rows(), before);
    assert!(run.advice_started);
    assert!(run.nominal_batch(&lp, &[], cap, &mut || true).is_none());
    // A valid fully prepared batch canceled at its precommit poll adds no row.
    let planes = [
        analytic_plane(0., vec![(Atom::Input(0), 1.), (Atom::Head(0), -1.)]),
        analytic_plane(0., vec![]),
    ]; // exact inert second row, no added restriction
    let shared = frame(&p, &planes, &data, until);
    let node = Node::new(&shared, &[], &mut || true).unwrap();
    let mut run = PriceRun::new(node).unwrap();
    let batch = run.nominal_batch(&lp, &[], cap, &mut || true).unwrap();
    assert_eq!(batch.rows.len(), 2);
    assert!(batch.rows[0].terms.is_empty() && batch.rows[1].terms.is_empty());
    assert!(batch.append(&mut lp, &mut || false).is_none());
    assert_eq!(lp.model.num_rows(), before);
}
