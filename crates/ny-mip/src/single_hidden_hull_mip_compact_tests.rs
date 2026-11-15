// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
struct Fixture {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    // Two independent pair relations plus a negative stable singleton.
    // Ideal individual floor -1.75; joint exact floor +.25 at bias -1.25.
    fn new(bias: f64) -> Self {
        Self {
            w: vec![
                vec![
                    1., 1., 0., 0., 1., -1., 0., 0., 0., 0., 1., 1., 0., 0., 1., -1., 1., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0.,
                ],
                vec![-1., -1., -1., -1., 1., 1., -0.5, 0., 0., 0., 0., 0., 0., 0.],
            ],
            b: vec![vec![0., 0., 0., 0., 2., 2., 1.], vec![bias, 0.]],
            dims: vec![4, 7, 2],
            bounds: vec![vec![Bound::new(-2., 2.); 4]
                .into_iter()
                .chain([Bound::new(1., 3.), Bound::new(1., 3.), Bound::new(1., 1.)])
                .collect()],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            4,
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 4],
            vec![1.; 4],
        )
        .unwrap()
    }
    fn parts(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.w,
            &self.b,
            &self.dims,
            &[Bound::new(-1., 1.); 4],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn binding(&self, parts: &MipParts) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.w,
            biases: &self.b,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(parts.problem.num_rows() - 1),
        }
    }
}
fn manual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    m.alpha[4] = 1.;
    m.alpha[5] = 1.;
    for j in 0..4 {
        for i in 0..4 {
            m.g[j * 4 + i] = 0.5 * p.w_row(j)[i];
        }
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    prepare(p, m, lower, end(), &mut || true).unwrap()
}
fn compact(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut cuts = manual(p);
    joint::discover_compact(&mut cuts, end(), &mut || true);
    assert_eq!(cuts.compact.as_ref().unwrap().pairs.len(), 2);
    cuts
}
fn refusal(cuts: &HullRootMipCuts<'_>, parts: &mut MipParts, f: &Fixture) {
    let original = parts.problem.clone();
    let vars = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let binding = f.binding(parts);
    assert!(cuts.append_to(parts, &binding, end()).is_none());
    assert_eq!(parts.problem, original);
    assert_eq!(parts.binary_vars, vars);
    assert_eq!(parts.binary_widths, widths);
}

#[test]
fn compact_pair_two_groups_and_negative_singleton_replay_full_q_without_changing_lp_scalar() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let cuts = compact(&p);
    let original = cuts.checked_lower;
    let vector = cuts.multipliers.clone();
    assert!(original < -1.749 && original > -1.751);
    let proof = cuts.compact_root_proof(end()).unwrap();
    assert!(proof.verified());
    assert!(proof.lower() > 0.249 && proof.lower() <= 0.25);
    assert_eq!(cuts.checked_lower.to_bits(), original.to_bits());
    assert_eq!(cuts.multipliers, vector);
    let mut parts = f.parts();
    let before = parts.problem.clone();
    let binding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &binding, end()).unwrap();
    assert_eq!(
        (
            report.individual_cuts,
            report.joint_cuts,
            report.compact_cuts,
            report.compact_pairs
        ),
        (5, 1, 1, 2)
    );
    assert_eq!(report.cuts, 6);
    assert_eq!(report.checked_hull_lower.to_bits(), original.to_bits());
    assert!(report.compact_root_lower.unwrap() > 0.249);
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.problem.cols(), before.cols());
    let expected = binding::head_columns(&binding, &f.parts()).unwrap();
    for &j in &[0, 1, 2, 3, 6] {
        let row = parts.problem.rows().last().unwrap();
        assert!(
            row.coeffs.contains(&(expected[j].0, -p.c[j])),
            "missing original negative head {j}"
        );
    }
}

#[test]
fn compact_pair_aggregate_row_contains_exact_rational_original_graph_grid() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let cuts = compact(&p);
    let row = cuts.compact.as_ref().unwrap();
    for grid in 0..81 {
        let mut k = grid;
        let mut x = Vec::new();
        for _ in 0..4 {
            x.push(q((k % 3) as f64 - 1.));
            k /= 3;
        }
        let mut lhs = Q::zero();
        for j in 0..p.hidden_dim {
            if p.c[j] < 0. {
                let z = q(p.b1[j]) + (0..4).map(|i| q(p.w_row(j)[i]) * &x[i]).sum::<Q>();
                lhs += q(-p.c[j]) * z.max(Q::zero());
            }
        }
        lhs -= (0..4).map(|i| q(row.g[i]) * &x[i]).sum::<Q>();
        assert!(lhs <= q(row.upper), "{grid}: {lhs}");
    }
}

#[test]
fn compact_pair_summed_point_prices_lost_last_coordinate_and_asymmetric_box() {
    let f = Fixture::new(-1.25);
    for (sign, l, u) in [(1., 1., 2.), (-1., -2., -1.), (1., -2., 3.)] {
        let mut p = f.source();
        p.lower[3] = l;
        p.upper[3] = u;
        let mut sum = Sum::new(4);
        for v in [sign * 1e16, sign, -sign * 1e16] {
            sum.add(&[0., 0., 0., v], 0., 1., end(), &mut || true)
                .unwrap();
        }
        let (g, h) = sum.finish(&p, end(), &mut || true).unwrap();
        assert_eq!(g[3], 0.);
        // An exact last-coordinate +/-1 is lost by point summation. Both
        // positive and negative boxes force strictly positive compensation.
        let exact = q(sign * 1e16) + q(sign) - q(sign * 1e16);
        assert_eq!(exact, q(sign));
        let a = (exact.clone() - q(g[3])) * q(l);
        let b = (exact - q(g[3])) * q(u);
        assert!(q(h) >= a.max(b));
    }
}

#[test]
fn compact_pair_selected_tuple_or_aggregate_mutation_refuses_atomic_append() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    for change in 0..7 {
        let mut cuts = compact(&p);
        let row = cuts.compact.as_mut().unwrap();
        match change {
            0 => row.g[3] = row.g[3].next_up(),
            1 => row.upper = row.upper.next_down(),
            2 => row.improvement = row.improvement.next_up(),
            3 => row.pairs[0].weights[0] *= 2.,
            4 => row.pairs[0].g[3] = f64::NAN,
            5 => row.pairs.push(row.pairs[0].clone()),
            _ => cuts.joint_completed = 0,
        }
        refusal(&cuts, &mut f.parts(), &f);
        assert!(cuts.compact_root_proof(end()).is_none(), "{change}");
    }
}

#[test]
fn compact_pair_foreign_weights_box_layout_or_output_bias_cannot_reuse_proof_bundle() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let cuts = compact(&p);
    let mut foreign = Fixture::new(-1.25);
    foreign.w[0][3] = 0.5;
    refusal(&cuts, &mut foreign.parts(), &foreign);
    let foreign = Fixture::new(-1.75);
    refusal(&cuts, &mut foreign.parts(), &foreign);
    let mut parts = f.parts();
    parts.problem.fix_col(parts.input_vars[3], 0.);
    refusal(&cuts, &mut parts, &f);
    let mut parts = f.parts();
    parts.output_vars.swap(0, 1);
    refusal(&cuts, &mut parts, &f);
}

#[test]
fn compact_pair_optional_search_refusal_preserves_every_individual_and_no_extra_row() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let mut cuts = manual(&p);
    joint::discover_compact(&mut cuts, end(), &mut || false);
    assert!(cuts.compact_mode && cuts.compact.is_none());
    let mut parts = f.parts();
    let binding = f.binding(&parts);
    let report = cuts.append_to(&mut parts, &binding, end()).unwrap();
    assert_eq!(
        (report.cuts, report.individual_cuts, report.compact_cuts),
        (5, 5, 0)
    );
    assert!(cuts.compact_root_proof(end()).is_none());
}

#[test]
fn compact_pair_expired_or_partial_replay_never_appends_or_renews_root_authority() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let cuts = compact(&p);
    for allowance in [0, 1, 20, 80] {
        let mut parts = f.parts();
        let before = parts.problem.clone();
        let mut n = 0;
        let binding = f.binding(&parts);
        assert!(cuts
            .append_with_poll(&mut parts, &binding, end(), &mut || {
                n += 1;
                n <= allowance
            })
            .is_none());
        assert_eq!(parts.problem, before);
    }
    let mut proof = cuts.compact_root_proof(end()).unwrap();
    proof.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(!proof.verified());
    let mut stale = compact(&p);
    stale.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(stale.compact_root_proof(end()).is_none());
    refusal(&stale, &mut f.parts(), &f);
}

#[test]
fn compact_pair_all_pair_attempts_and_matching_storage_share_single_bounded_search() {
    let h = 66;
    let w: Vec<_> = (0..h)
        .flat_map(|j| [1., if j % 2 == 0 { 1. } else { -1. }])
        .collect();
    let p = HullMarginProblem::new(
        w.clone(),
        vec![0.; h],
        2,
        &vec![-1.; h],
        &[1.],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; 2],
        vec![1.; 2],
    )
    .unwrap();
    let mut m = HullMultipliers::zeros(&p);
    m.g = w.iter().map(|v| 0.5 * v).collect();
    let lower = certify_hull_lower_bound(&p, &m).unwrap();
    let mut cuts = prepare(&p, m, lower, end(), &mut || true).unwrap();
    joint::discover_compact(&mut cuts, end(), &mut || true);
    assert_eq!(cuts.joint_attempted, joint::MAX_PAIRS);
    assert!(cuts.joint_completed <= joint::MAX_PAIRS);
    assert_eq!(cuts.joint.len(), 64);
    assert_eq!(cuts.joint_row_count(), 1);
    let row = cuts.compact.as_ref().unwrap();
    assert!(row.pairs.len() > 1 && row.pairs.len() <= h / 2);
    let mut seen = vec![false; h];
    for r in &row.pairs {
        for &j in &r.heads {
            assert!(!seen[j]);
            seen[j] = true;
        }
    }
    let before = (
        cuts.joint_attempted,
        cuts.joint_completed,
        row.upper.to_bits(),
    );
    joint::discover_compact(&mut cuts, end(), &mut || true);
    assert_eq!(
        (
            cuts.joint_attempted,
            cuts.joint_completed,
            cuts.compact.as_ref().unwrap().upper.to_bits()
        ),
        before
    );
}

#[test]
fn compact_pair_sum_overflow_and_nonfinite_selected_data_fail_closed() {
    let mut sum = Sum::new(1);
    sum.add(&[1e308], 0., 1., end(), &mut || true).unwrap();
    assert!(sum.add(&[1e308], 0., 1., end(), &mut || true).is_none());
    let f = Fixture::new(-1.25);
    let p = f.source();
    let mut cuts = compact(&p);
    cuts.compact.as_mut().unwrap().upper = f64::INFINITY;
    refusal(&cuts, &mut f.parts(), &f);
    assert!(cuts.compact_root_proof(end()).is_none());
}

#[test]
fn compact_pair_full_q_replay_sanitizes_alpha_and_never_omits_final_negative_head() {
    let f = Fixture::new(-1.25);
    let p = f.source();
    let mut cuts = compact(&p);
    let old = cuts.compact_root_proof(end()).unwrap().lower();
    cuts.multipliers.alpha[4] = f64::NAN;
    let row = replay(&cuts, end(), &mut || true).unwrap();
    let after = root_lower(&cuts, &row, end(), &mut || true).unwrap();
    assert!(after < old);
    // Last stable negative singleton contributes -.5; replacing it by zero
    // would incorrectly add .5. The authored safe full-q floor remains<=.25.
    let cuts = compact(&p);
    assert!(cuts.compact_root_proof(end()).unwrap().lower() <= 0.25);
}

#[path = "single_hidden_hull_mip_compact_actual_tests.rs"]
mod actual_tests;
