// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Genuine original-encoder fixture; no discovery plan or captured point is supplied.
use super::super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

pub(super) const D: usize = 7;
pub(super) const H: usize = 13;
pub(super) const EPS: f64 = 1. / 128.;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn coordinate(j: usize) -> usize {
    match j {
        0..=2 | 9..=12 => 0,
        3..=8 => j - 2,
        _ => panic!("head"),
    }
}
pub(super) struct Fixture {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub dims: Vec<usize>,
    pub bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    pub(super) fn new(bias: f64) -> Self {
        let mut first = vec![0.; D * H];
        for j in 0..H {
            first[j * D + coordinate(j)] = 1.;
        }
        let mut output = vec![0.; 2 * H];
        for j in 0..H {
            output[j] = match j {
                0 => 6.,
                1..=2 | 9..=12 => -1.,
                _ => -EPS,
            };
        }
        // Raw original margin = bias - EPS*sum_{i=1..6} R(y_i).
        // Six coincident negative heads cancel +6R(x) on the true graph.
        // The old negative-only grouping may leave twelve singleton parents.
        // Paid root discovery must join all six principal parents itself;
        // no source-order compact grouping is assumed.
        Self {
            w: vec![first, output],
            b: vec![vec![0.; H], vec![bias, 0.]],
            dims: vec![D, H, 2],
            bounds: vec![vec![Bound::new(-1., 1.); H]],
        }
    }
    pub(super) fn source(&self) -> HullMarginProblem {
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
        assert_eq!(p.c.iter().filter(|&&c| c < 0.).count(), 12);
        p
    }
    pub(super) fn original(&self) -> MipParts {
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
    pub(super) fn parts(&self) -> MipParts {
        let mut parts = self.original();
        let positive = parts.binary_vars[0];
        assert!(parts.problem.relax_binary_integrality(&[positive]));
        parts.binary_vars.remove(0);
        parts.binary_widths.remove(0);
        parts
    }
    pub(super) fn encoding(&self) -> HullMipEncoding<'_> {
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
pub(super) fn point(f: &Fixture, parts: &MipParts, input: &[Q], zero_mask: usize) -> Vec<Q> {
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
pub(super) fn rows_hold(ir: &crate::ir::MilpProblem, values: &[Q], skip: Option<usize>) {
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
pub(super) fn lp(ir: &crate::ir::MilpProblem, parts: &MipParts, property: usize) -> f64 {
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
pub(super) fn solve(parts: MipParts) -> crate::MipResult {
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
