// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Analytic controls for genuine input-affine geometry, not output-margin cuts.
use super::feedback_tests::{budget, end, reference_hint};
use super::*;

fn spec(bias: f32) -> InputHeadSpec {
    let mut first = vec![0.0; 2 * 16];
    for h in 0..14 {
        first[h] = 1.0;
        first[16 + h] = 1.0;
    }
    first[14] = 1.0;
    first[16 + 15] = -1.0;
    let mut hidden_bias = vec![0.0; 16];
    hidden_bias[14] = 2.0;
    hidden_bias[15] = 2.0;
    let last = (0..16)
        .flat_map(|h| [if h < 14 { -1.0 } else { -14.0 }, 0.0])
        .collect();
    InputHeadSpec::new(2, 16, first, hidden_bias, last, vec![bias, 0.0], [103; 32]).unwrap()
}

fn source(bias: f32) -> Source {
    Source::build(&spec(bias), &[-1.0; 2], &[1.0; 2], 0, 1, end()).unwrap()
}

fn direction(start: usize) -> Direction {
    Direction {
        heads: (start..start + 7).map(|h| (h, 1.0)).collect(),
    }
}

fn cuts(s: &Source) -> Vec<CertifiedDirection<'_>> {
    let mut work = budget();
    [0, 7]
        .into_iter()
        .map(|start| certify_affine(s, direction(start), vec![(1, 7.0)], &mut work).unwrap())
        .collect()
}

fn floor(s: &Source) -> f64 {
    let rows = cuts(s);
    let model = triangle::Model::from_root_portfolio(s, &mut None, &rows, end()).unwrap();
    model
        .replay(s, &reference_hint(model.request(), 32), 1.0, end())
        .unwrap()
}

#[test]
fn affine_two_input_support_rows_close_the_exact_safe_margin_and_keep_unsafe_open() {
    // r=ReLU(u+v), p=u+2, n=2-v. Each selected seven-head group obeys
    // 7v-sum(r)>=-7. Twice this plus the stable p/n equations gives +1/2.
    // The true minimum is also +1/2: concavity reduces it to four vertices.
    let exact = [-1i32, 1]
        .into_iter()
        .flat_map(|u| {
            [-1i32, 1]
                .into_iter()
                .map(move |v| 169 - 28 * (u + 2) - 28 * (2 - v) - 28 * (u + v).max(0))
        })
        .min()
        .unwrap();
    assert_eq!(exact, 1);
    let s = source(84.5);
    for row in cuts(&s) {
        assert!(row.lower <= -7.0 && row.lower > -7.001);
        let owned = row.owned_row(&s, end()).unwrap();
        assert_eq!(owned.coefficients[0].0, 1);
        assert_eq!(owned.coefficients[0].1.lo, 7.0);
        assert_eq!(owned.coefficients.len(), 8);
    }
    assert!(floor(&s) > 0.49 && floor(&s) <= 0.5);
    let unsafe_floor = floor(&source(83.5));
    assert!(unsafe_floor <= -0.5 && unsafe_floor > -0.501);
}

#[test]
fn affine_positive_support_lower_is_not_clamped_to_zero() {
    // eta*u-sum_j ReLU(u-2)=7*min(u,2), minimum 7 on [1,3].
    // The independent input/head box floor is zero, so this is useful geometry.
    let spec = InputHeadSpec::new(
        1,
        7,
        vec![1.0; 7],
        vec![-2.0; 7],
        (0..7).flat_map(|_| [-1.0, 0.0]).collect(),
        vec![100.0, 0.0],
        [104; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[1.0], &[3.0], 0, 1, end()).unwrap();
    let mut work = budget();
    let row = certify_affine(&s, direction(0), vec![(0, 7.0)], &mut work).unwrap();
    assert_eq!(work.masks, 128);
    assert!(row.lower > 6.99 && row.lower <= 7.0, "{}", row.lower);
}

#[test]
fn affine_explicit_old_unstable_constant_rows_and_fixed_seven_cover_leave_a_gap() {
    let s = source(84.5);
    // At (u,v)=(1,-1), the scalar relaxation permits all fourteen r=1,
    // p=n=3. Constant cuts on unstable duplicate heads permit it too: each
    // such support has maximum 2*sum(tau). Stable p,n are excluded from that
    // old selector family. This does not claim dominance over arbitrary
    // all-head constant supports, which could encode the entire objective.
    let mut point = vec![1.0, -1.0];
    point.extend([1.0; 14]);
    point.extend([3.0, 3.0]);
    let old = triangle::Model::build(&s, end()).unwrap();
    let (_, rows) = old.into_phase_parts();
    for row in &rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(point[j])).unwrap()).unwrap()
        });
        assert!(lhs.lo >= row.lower.lo - 1e-8, "old row excludes comparator");
    }
    let old_value = 84.5 - 14.0 * 3.0 - 14.0 * 3.0 - 14.0;
    assert_eq!(old_value, -13.5);
    let cover = phase_cover::Model::build(&s, PhaseCoverPolicy::new(7).unwrap(), end()).unwrap();
    let request = cover.request();
    let selected: Vec<_> = request.leaves[0]
        .rows
        .iter()
        .filter_map(|row| {
            row.coefficients
                .iter()
                .find(|&&(j, a)| j >= 2 && a == -1.0)
                .map(|&(j, _)| j)
        })
        .collect();
    assert_eq!(selected.len(), 7);
    for &j in &selected {
        assert!((2..16).contains(&j));
        point[j] = 0.0;
    }
    for leaf in &request.leaves {
        for row in request.rows.iter().chain(&leaf.rows) {
            let lhs: f64 = row.coefficients.iter().map(|&(j, a)| a * point[j]).sum();
            assert!(
                lhs >= row.lower - 1e-8,
                "cover leaf {} excludes comparator",
                leaf.mask
            );
        }
    }
    assert_eq!(
        84.5 + request
            .objective
            .iter()
            .zip(&point)
            .map(|(a, x)| a * x)
            .sum::<f64>(),
        -6.5
    );
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn affine_replay_prices_the_input_residual_and_refuses_foreign_or_child_authority() {
    let s = source(84.5);
    let rows = cuts(&s);
    let model = triangle::Model::from_root_portfolio(&s, &mut None, &rows, end()).unwrap();
    let hint = reference_hint(model.request(), 32);
    let mut residual = model.request().objective.clone();
    for (row, &mass) in model.request().rows.iter().zip(&hint.row_duals) {
        for &(j, a) in &row.coefficients {
            residual[j] -= mass * a;
        }
    }
    assert!((residual[0] + 14.0).abs() < 1e-9, "{residual:?}");
    assert!(residual[1].abs() < 1e-9);
    assert!(residual[2..].iter().all(|v| v.abs() < 1e-9));
    assert!(model.replay_recorded(&s, &hint, 1.0, end(), true).is_none());
    let foreign = s.clone();
    assert!(rows[0].owned_row(&foreign, end()).is_none());
    assert!(model.replay(&foreign, &hint, 1.0, end()).is_none());
    let mut missing_stable = hint.clone();
    missing_stable.row_duals[model.request().slope_rows[15].unwrap() + 1] = 0.0;
    assert!(model.replay(&s, &missing_stable, 1.0, end()).unwrap() < -1.0);
}

#[test]
fn affine_geometry_uses_first_error_but_excludes_bias_and_output_error() {
    let s = source(84.5);
    let baseline = certify_affine(&s, direction(0), vec![(1, 7.0)], &mut budget()).unwrap();
    let mut changed = s.clone();
    changed.bias = I::p(1000.0);
    changed.output_error += 17.0;
    let independent =
        certify_affine(&changed, direction(0), vec![(1, 7.0)], &mut budget()).unwrap();
    assert_eq!(baseline.lower.to_bits(), independent.lower.to_bits());
    let mut error = s.clone();
    for e in &mut error.first_error[..7] {
        *e += 0.25;
    }
    for y in &mut error.y[..7] {
        y.lo -= 0.25;
        y.hi += 0.25;
    }
    let widened = certify_affine(&error, direction(0), vec![(1, 7.0)], &mut budget()).unwrap();
    assert!(
        widened.lower <= -8.75 && widened.lower > -8.751,
        "{}",
        widened.lower
    );
    let mut output = s.clone();
    output.output_error += 1.0;
    let once = floor(&s) - floor(&output);
    assert!((once - 1.0).abs() < 1e-10, "Eout difference {once}");
    let mut bias = s.clone();
    bias.bias = bias.bias.sub(I::p(0.25)).unwrap();
    assert!((floor(&s) - floor(&bias) - 0.25).abs() < 1e-10);
}

#[test]
fn affine_invalid_eta_and_incomplete_mask_cover_cannot_create_a_row() {
    let s = source(84.5);
    for eta in [
        vec![(0, 1.0), (0, 2.0)],
        vec![(1, 1.0), (0, 1.0)],
        vec![(2, 1.0)],
        vec![(0, f64::NAN)],
        vec![(0, f64::INFINITY)],
        vec![(0, 0.0)],
    ] {
        assert!(certify_affine(&s, direction(0), eta, &mut budget()).is_none());
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            EXPIRE_AFTER_MASK.with(|f| f.set(None));
        }
    }
    let _reset = Reset;
    EXPIRE_AFTER_MASK.with(|f| f.set(Some(128)));
    let mut work = budget();
    assert!(certify_affine(&s, direction(0), vec![(1, 7.0)], &mut work).is_none());
    assert_eq!(work.masks, 128);
}

#[test]
fn affine_native_feedback_discovers_input_geometry_with_shared_caps() {
    for bias in [84.5, 83.5] {
        let s = source(bias);
        let (r, prefix) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
        assert!(r.affine_cuts > 0);
        assert!(r.cuts <= MAX_ROUNDS * MAX_CUTS && r.attempted <= MAX_ATTEMPTS);
        assert!(r.masks <= MAX_MASKS && r.work_cells <= MAX_CELLS);
        assert!(
            r.rounds.len() <= MAX_ROUNDS && r.rounds.iter().all(|round| round.cuts <= MAX_CUTS)
        );
        assert_eq!(r.calls, 0);
        assert!(prefix.is_some());
        if bias == 84.5 {
            assert!(r.lower > 0.49 && r.lower <= 0.5, "{}", r.lower);
        } else {
            assert!(r.lower <= -0.5, "{}", r.lower);
        }
    }
}
