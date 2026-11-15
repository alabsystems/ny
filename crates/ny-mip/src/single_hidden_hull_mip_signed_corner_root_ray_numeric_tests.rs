// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Unrun numerical refusal controls. These create no replay/publication authority.
use super::*;

#[test]
fn root_ray_inline_metadata_keeps_existing_cell_size_gates() {
    // These old eligibility gates must remain true even if root-Ray refuses.
    // The Cube Admission layout is unchanged; its private guard still runs in
    // genuine component admission exercised by the separate paid controls.
    assert!(size_of::<phase::cube::cell::Collector>() <= 1280);
    assert!(size_of::<phase::cube::cell::Table>() <= 192);
    assert!(
        FIXED_BYTES
            >= 2 * size_of::<Frame>()
                + 2 * size_of::<Finished>()
                + 2 * size_of::<Quote>()
                + 2 * size_of::<core::RootReplacement<'static>>()
                + 128
    );
}

fn fixture() -> (HullMarginProblem, Component) {
    let p = HullMarginProblem::new(
        vec![1., 0., 2., 0.],
        vec![0., 0.],
        2,
        &[-1., 1., 0., 0.],
        &[0., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; 2],
        vec![1.; 2],
    )
    .unwrap();
    let component = Component {
        negative: vec![0],
        positive: vec![1],
        positive_units: vec![2],
        g: vec![0.; 2],
        alpha: vec![0., 1.],
    };
    (p, component)
}

fn affine(p: &HullMarginProblem, c: &Component, mask: usize) -> (Vec<I>, I, f64) {
    let mut coeff = vec![I::zero(); 2];
    let mut constant = I::zero();
    let positive = c.coefficient(p, 0).unwrap().scale(c.alpha[mask]).unwrap();
    constant = constant.add(positive.scale(p.b1[1]).unwrap()).unwrap();
    for (i, q) in coeff.iter_mut().enumerate() {
        *q = q.add(positive.scale(p.w_row(1)[i]).unwrap()).unwrap();
    }
    if mask == 1 {
        let negative = I::point(p.c[0]).unwrap();
        constant = constant.add(negative.scale(p.b1[0]).unwrap()).unwrap();
        for (i, q) in coeff.iter_mut().enumerate() {
            *q = q.add(negative.scale(p.w_row(0)[i]).unwrap()).unwrap();
        }
    }
    let mut lower = constant.lo;
    for (i, q) in coeff.iter().enumerate() {
        lower = finite(add_down(lower, q.minimum(p.lower[i], p.upper[i]).unwrap())).unwrap();
    }
    (coeff, constant, lower)
}

fn axes(p: &HullMarginProblem, c: &Component) -> core::Axes {
    let mut ranker = core::Ranker::new(&p.lower, &p.upper, &[1], 2).unwrap();
    let mut minimum = f64::INFINITY;
    for mask in 0..2 {
        let (q, _, lower) = affine(p, c, mask);
        minimum = minimum.min(lower);
        ranker.push(0, mask, &q, lower, &mut || true).unwrap();
    }
    ranker.finish(minimum, p.d, &[0.; 2], &mut || true).unwrap()
}

fn builder(p: &HullMarginProblem, c: &Component, entries: [Option<(usize, f64)>; 2]) -> Builder {
    let quote = quote(2, 1, 2, 2, 1, 2).unwrap();
    let mut capture = Capture::new(quote).unwrap();
    for (mask, entry) in entries.into_iter().enumerate() {
        capture.push(0, mask, entry);
    }
    Builder::new(capture.finish().unwrap(), &axes(p, c), quote).unwrap()
}

#[allow(
    clippy::unnecessary_box_returns,
    reason = "Retain the production builder's boxed table owner in these controls."
)]
fn completed(
    p: &HullMarginProblem,
    c: &Component,
    entries: [Option<(usize, f64)>; 2],
) -> Box<Table> {
    let mut b = builder(p, c, entries);
    for mask in 0..2 {
        let (q, k, lower) = affine(p, c, mask);
        b.push(p, c, 0, mask, &q, k, lower, &mut || true).unwrap();
    }
    b.finish().unwrap()
}

#[test]
fn root_ray_capture_requires_one_complete_canonical_target() {
    let q = quote(2, 1, 2, 2, 1, 2).unwrap();
    let mut missing = Capture::new(q).unwrap();
    missing.push(0, 0, None);
    assert!(missing.finish().is_none());
    for malformed in [(0, 0), (1, 1), (0, 2)] {
        let mut c = Capture::new(q).unwrap();
        c.push(0, 0, None);
        c.push(malformed.0, malformed.1, None);
        c.push(0, 1, None); // A later correct item cannot repair poisoned coverage.
        assert!(c.finish().is_none());
    }
}

#[test]
fn root_ray_illegal_or_nonfinite_advice_falls_back_to_the_entire_raw_mask() {
    let (p, c) = fixture();
    let reference = completed(&p, &c, [None, None]);
    let bits: Vec<_> = reference.values.iter().map(|v| v.to_bits()).collect();
    for entries in [
        [Some((0, -1.)), Some((0, 1.))], // Both closed-phase signs are illegal.
        [Some((1, 2.)), Some((1, -2.))], // A positive Source head is not a phase.
        [Some((usize::MAX, 2.)), Some((2, -2.))],
        [Some((0, f64::NAN)), Some((0, f64::INFINITY))],
    ] {
        let table = completed(&p, &c, entries);
        assert_eq!(
            table.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bits
        );
    }
    // Independent graph witness: the original rounded Source evaluates to p.d
    // at x=0. Every proposed global lower floor must be no larger than it.
    let legal = completed(&p, &c, [None, Some((0, -1.))]);
    let floor = legal.values.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(floor <= 0. && floor > -1e-12);
    assert!(floor + p.d <= p.d);
    assert!(
        reference
            .values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            < -0.99
    );
}

#[test]
fn root_ray_second_replay_coverage_and_shape_refusal_are_sticky() {
    let (p, c) = fixture();
    let (q, k, lower) = affine(&p, &c, 0);
    let mut missing = builder(&p, &c, [None, None]);
    missing
        .push(&p, &c, 0, 0, &q, k, lower, &mut || true)
        .unwrap();
    assert!(missing.finish().is_none());
    for malformed in [0, 2] {
        let mut b = builder(&p, &c, [None, None]);
        b.push(&p, &c, 0, 0, &q, k, lower, &mut || true).unwrap();
        assert!(b
            .push(&p, &c, 0, malformed, &q, k, lower, &mut || true)
            .is_none());
        let (q1, k1, lower1) = affine(&p, &c, 1);
        assert!(b
            .push(&p, &c, 0, 1, &q1, k1, lower1, &mut || true)
            .is_none());
        assert!(b.finish().is_none());
    }
    let mut short = builder(&p, &c, [None, None]);
    assert!(short
        .push(&p, &c, 0, 0, &q[..1], k, lower, &mut || true)
        .is_none());
    assert!(short.finish().is_none());
}

#[test]
fn root_ray_late_final_poll_cannot_publish_a_complete_but_cancelled_table() {
    let (p, c) = fixture();
    // Calibrate only the number of cooperative checkpoints. The correctness
    // assertion is rejection after the last checkpoint of the last mask.
    let mut probe = builder(&p, &c, [None, None]);
    let (q0, k0, lower0) = affine(&p, &c, 0);
    probe
        .push(&p, &c, 0, 0, &q0, k0, lower0, &mut || true)
        .unwrap();
    let (q1, k1, lower1) = affine(&p, &c, 1);
    let mut checkpoints = 0;
    probe
        .push(&p, &c, 0, 1, &q1, k1, lower1, &mut || {
            checkpoints += 1;
            true
        })
        .unwrap();
    assert!(checkpoints > 2 && probe.finish().is_some());
    let mut b = builder(&p, &c, [None, None]);
    b.push(&p, &c, 0, 0, &q0, k0, lower0, &mut || true).unwrap();
    let mut count = 0;
    assert!(b
        .push(&p, &c, 0, 1, &q1, k1, lower1, &mut || {
            count += 1;
            count != checkpoints
        })
        .is_none());
    // Even a cooperative caller that would return true again cannot resurrect.
    assert!(b
        .push(&p, &c, 1, 0, &q0, k0, lower0, &mut || true)
        .is_none());
    assert!(b.finish().is_none());
}
