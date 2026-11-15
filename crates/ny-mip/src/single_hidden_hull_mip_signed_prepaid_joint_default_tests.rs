// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual CLI-selected prepaid policy on the original D15/H17 encoder.
//! No partition, captured point, pair, multiplier or mode bit is supplied.
use super::super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

const D: usize = 15;
const H: usize = 17;
const EPS: f64 = 1. / 128.;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn coordinate(j: usize) -> usize {
    match j {
        0 | 1 | 9 => 0,
        2..=8 => j - 1,
        10..=16 => j - 2,
        _ => panic!("head"),
    }
}
struct Fixture {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn new(bias: f64) -> Self {
        let mut first = vec![0.; D * H];
        for j in 0..H {
            first[j * D + coordinate(j)] = 1.;
        }
        let mut output = vec![0.; 2 * H];
        for j in 0..H {
            output[j] = match j {
                0 => 4.,
                1 => -1.,
                9 => -3.,
                _ => -EPS,
            };
        }
        // The raw original margin is exactly bias - EPS*sum_{i=1..14} R(x_i).
        // In particular, the three main heads cancel on the actual graph.
        Self {
            w: vec![first, output],
            b: vec![vec![0.; H], vec![bias, 0.]],
            dims: vec![D, H, 2],
            bounds: vec![vec![Bound::new(-1., 1.); H]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        let p = HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            D,
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; D],
            vec![1.; D],
        )
        .unwrap();
        assert_eq!(p.c.iter().filter(|&&c| c > 0.).count(), 1);
        assert_eq!(p.c.iter().filter(|&&c| c < 0.).count(), 16);
        p
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.w,
            &self.b,
            &self.dims,
            &[Bound::new(-1., 1.); D],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn parts(&self) -> MipParts {
        let mut parts = self.original();
        let positive = parts.binary_vars[0];
        assert!(parts.problem.relax_binary_integrality(&[positive]));
        parts.binary_vars.remove(0);
        parts.binary_widths.remove(0);
        parts
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.w,
            biases: &self.b,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}
fn point(f: &Fixture, parts: &MipParts, input: &[Q], zero_mask: usize) -> Vec<Q> {
    let original = f.original();
    let heads = binding::head_columns(&f.encoding(), &original).unwrap();
    let mut values = vec![Q::zero(); parts.num_cols];
    for i in 0..D {
        values[parts.input_vars[i].0] = input[i].clone();
    }
    for j in 0..H {
        let z = input[coordinate(j)].clone();
        values[D + j] = z.clone();
        values[heads[j].0] = z.clone().max(Q::zero());
        let active = if z.is_zero() {
            (zero_mask >> j) & 1
        } else {
            usize::from(z > Q::zero())
        };
        values[original.binary_vars[j].0] = q(active as f64);
    }
    for o in 0..2 {
        values[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..H)
                .map(|j| q(f.w[1][o * H + j]) * &values[heads[j].0])
                .sum::<Q>();
    }
    values
}
fn rows_hold(ir: &crate::ir::MilpProblem, values: &[Q], skip: Option<usize>) {
    for (j, c) in ir.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || values[j] >= q(c.lb), "column{j} lower");
        assert!(!c.ub.is_finite() || values[j] <= q(c.ub), "column{j} upper");
    }
    for (i, row) in ir.rows().iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let lhs: Q = row.coeffs.iter().map(|&(j, a)| q(a) * &values[j]).sum();
        assert!(!row.lb.is_finite() || lhs >= q(row.lb), "row{i} lower");
        assert!(!row.ub.is_finite() || lhs <= q(row.ub), "row{i} upper");
    }
}
fn lp(ir: &crate::ir::MilpProblem, parts: &MipParts, property: usize) -> f64 {
    let mut relaxed = crate::ir::MilpProblem::new();
    for col in ir.cols() {
        relaxed.add_col(col.obj, col.lb, col.ub);
    }
    for (i, row) in ir.rows().iter().enumerate() {
        if i != property {
            relaxed.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
            );
        }
    }
    let model = crate::ay_lib::to_ay_model(&relaxed).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &opts).unwrap();
    let bound = session
        .rigorous_bound_expr(
            &[
                (model.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (model.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&bound).unwrap()
}
fn solve(parts: MipParts) -> crate::MipResult {
    crate::MipSolver::new(
        parts,
        crate::MipConfig {
            backend: crate::MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    )
    .check_feasibility()
    .unwrap()
}
fn appended(f: &Fixture, p: &HullMarginProblem) -> (MipParts, HullMipCutReport) {
    // Exactly the new public entry selected by CLI strengthen_with_signed_refinement.
    let attempt = propose_root_mip_cuts_with_prepaid_joint_attempt(p, end(), Duration::ZERO);
    assert!(
        attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked,
        "capture={:?}",
        attempt.primal
    );
    let cuts = attempt
        .cuts
        .expect("actual default retains its checked owner");
    assert_eq!(cuts.proposal_lp_calls, 1);
    let row = cuts.signed.as_ref().expect("actual default signed owner");
    assert!((2..=16).contains(&row.plan.components.len()));
    let ticket = row
        .budget
        .prepaid
        .as_ref()
        .expect("genuine early admission");
    assert!(ticket.owned(row));
    let mut parts = f.parts();
    let original = parts.problem.clone();
    let binary_vars = parts.binary_vars.clone();
    let binary_widths = parts.binary_widths.clone();
    let cols = parts.num_cols;
    let before = original.num_rows();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert_eq!(parts.num_cols, cols);
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(&parts.problem.rows()[..before], original.rows());
    assert_eq!(parts.binary_vars, binary_vars);
    assert_eq!(parts.binary_widths, binary_widths);
    assert_eq!(parts.problem.num_rows() - before, report.cuts);
    let signed = report.signed.as_ref().unwrap();
    assert_eq!((report.proposal_lp_calls, signed.source_replays), (1, 1));
    assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    let joint = &signed.component_rows.joint;
    assert!(
        joint.prepaid_requested && joint.prepaid_admitted && joint.attempted,
        "joint={joint:?}"
    );
    assert_eq!(joint.prepaid_work_units, super::WORK);
    assert!(joint.prepaid_bytes > 0);
    assert!((4096..=super::WORK).contains(&joint.prepaid_used_work_units));
    assert_eq!((joint.prepaid_checker_slots, joint.checker_slots), (1, 1));
    assert_eq!(
        (
            joint.producer_attempts,
            joint.paid_pairs,
            joint.completed_pairs
        ),
        (1, 1, 1)
    );
    assert_eq!((joint.prepared_rows, joint.appended_rows), (1, 1));
    assert!(
        joint.checked_masks > 0 && joint.appended_terms > 0,
        "joint={joint:?}"
    );
    assert!(joint.work_units > 0 && joint.work_units <= joint.prepaid_used_work_units);
    let dom = &signed.component_rows.domination;
    assert_eq!(
        (
            dom.producer_attempts,
            dom.advice_attempts,
            dom.support_attempts,
            dom.appended_rows
        ),
        (0, 0, 0, 0)
    );
    for main in [-1.0_f64, -0.5, 0., 0.5, 1.] {
        for padding in [-1.0_f64, 0., 1.] {
            let mut input = vec![q(padding); D];
            input[0] = q(main);
            for zero_mask in [0, (1usize << H) - 1] {
                // Arbitrary graph points need not satisfy the unsafe property.
                rows_hold(
                    &parts.problem,
                    &point(f, &parts, &input, zero_mask),
                    Some(f.encoding().unsafe_row.0),
                );
            }
        }
    }
    (parts, report)
}

#[test]
fn prepaid_default_public_entry_checks_one_existing_union_and_actual_safe_original_model() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let (parts, _) = appended(&f, &p);
    let original = f.parts();
    let property = f.encoding().unsafe_row.0;
    let before = lp(&original.problem, &original, property);
    assert!(
        (before - (0.25 - 2. - 14. * EPS)).abs() < 1e-6,
        "original Big-M={before}"
    );
    let after = lp(&parts.problem, &parts, property);
    // This is an actual-default selection/closure assertion, without fixing the
    // partition or steering. A different selected pair must fail visibly here.
    assert!(
        after > 0.14 && after <= 9. / 64. + 1e-6,
        "augmented LP={after}, original={before}"
    );
    let mut minimizer = vec![q(1.); D];
    minimizer[0] = Q::zero();
    let graph = point(&f, &parts, &minimizer, 0);
    rows_hold(&parts.problem, &graph, Some(property));
    assert_eq!(graph[parts.output_vars[0].0], q(9. / 64.));
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}
fn actual_sat(bias: f64, expected: f64) {
    let f = Fixture::new(bias);
    let p = f.source();
    let (parts, _) = appended(&f, &p);
    let mut witness = vec![q(1.); D];
    witness[0] = Q::zero();
    for cells in 0..8usize {
        // Both closed phase cells at each of the three zero preactivations.
        let zero_mask = (cells & 1) | (cells & 2) | ((cells & 4) << 7);
        let graph = point(&f, &parts, &witness, zero_mask);
        rows_hold(&parts.problem, &graph, None);
        assert_eq!(graph[parts.output_vars[0].0], q(expected));
        assert!(graph[parts.output_vars[0].0] <= Q::zero());
    }
    match solve(parts) {
        crate::MipResult::Sat { input_values, .. } => {
            let input: Vec<_> = input_values.iter().map(|&x| q(x)).collect();
            let original = f.original();
            let graph = point(&f, &original, &input, 0);
            rows_hold(&original.problem, &graph, None);
            assert!(graph[original.output_vars[0].0] <= Q::zero());
        }
        other => panic!("{other:?}"),
    }
}
#[test]
fn prepaid_default_public_entry_preserves_original_unsafe_sat_witness() {
    actual_sat(14. * EPS - 0.25, -0.25);
}
#[test]
fn prepaid_default_public_entry_preserves_original_closed_zero_tie() {
    actual_sat(14. * EPS, 0.);
}
