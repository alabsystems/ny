// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Shared original-encoder fixtures for phase-row soundness tests.
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
pub(super) struct Fixture {
    pub(super) w: Vec<Vec<f64>>,
    pub(super) b: Vec<Vec<f64>>,
    pub(super) dims: Vec<usize>,
    pub(super) bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    pub(super) fn new(bias: f64) -> Self {
        Self {
            w: vec![
                vec![1., -1., 1., 1., -1., -1.],
                vec![1., 1., -2., 1., 1., -2., 0., 0., 0., 0., 0., 0.],
            ],
            b: vec![vec![0.; 6], vec![bias, 0.]],
            dims: vec![1, 6, 2],
            bounds: vec![vec![Bound::new(-1., 1.); 6]],
        }
    }
    pub(super) fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            1,
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.],
            vec![1.],
        )
        .unwrap()
    }
    pub(super) fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.w,
            &self.b,
            &self.dims,
            &[Bound::new(-1., 1.)],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    pub(super) fn parts(&self) -> MipParts {
        let mut p = self.original();
        let vars = p.binary_vars.clone();
        let widths = p.binary_widths.clone();
        let positive: Vec<_> = vars
            .iter()
            .enumerate()
            .filter_map(|(j, v)| (j != 2 && j != 5).then_some(*v))
            .collect();
        assert!(p.problem.relax_binary_integrality(&positive));
        p.binary_vars = vec![vars[2], vars[5]];
        p.binary_widths = vec![widths[2], widths[5]];
        p
    }
    pub(super) fn encoding(&self, p: &MipParts) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.w,
            biases: &self.b,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(p.problem.num_rows() - 1),
        }
    }
}
pub(super) fn plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.],
                alpha: vec![0., 0., 1., 0.],
            },
            Component {
                negative: vec![5],
                positive: vec![3, 4],
                positive_units: vec![2, 2],
                g: vec![0.],
                alpha: vec![0., 0., 0., 1.],
            },
        ],
        leftover: vec![],
    }
}
pub(super) fn base(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    for j in [2, 5] {
        m.g[j] = 0.5 * p.w_row(j)[0];
    }
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    super::super::super::prepare(p, m, lower, end(), &mut || true).unwrap()
}
pub(super) fn make_row(c: &HullRootMipCuts<'_>, p: &Plan, b: &Budget) -> SignedMajorant {
    let cost = p.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(c, p, b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(c).collect());
    row(c, p, checked, cost, b, &seal)
}
pub(super) fn cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut c = base(p);
    let p = plan();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let row = make_row(&c, &p, &b);
    component_rows::admit(&c, &p, &b).unwrap();
    assert!(b.state.lock().unwrap().component_rows.phase.admitted);
    c.signed = Some(row);
    c
}
pub(super) fn append(f: &Fixture) -> (MipParts, HullMipCutReport, Row) {
    let p = f.source();
    let c = cuts(&p);
    let mut parts = f.parts();
    let encoding = f.encoding(&parts);
    let unsafe_row = encoding.unsafe_row;
    let before = parts.problem.clone();
    let report = c.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    let s = report.signed.as_ref().unwrap();
    let phase = &s.component_rows.phase;
    assert_eq!(s.source_replays, 1);
    assert_eq!(s.checked_masks, 8);
    assert_eq!(s.component_rows.appended_rows, 2);
    assert_eq!(phase.appended_rows, 2);
    assert_eq!(phase.collection_attempts, 1);
    assert_eq!(phase.collected_components, 2);
    assert_eq!(phase.reserved_work_units, 0);
    assert_eq!(phase.stop_reason, "appended");
    assert!(s.work_units <= MAX_WORK && s.admitted_bytes <= MAX_BYTES);
    assert_eq!(report.proposal_lp_calls, 0);
    (parts, report, unsafe_row)
}
/// Exact old-family prefix for finite-plane LP comparisons. The actual default
/// appender still runs all later optional families and solve() retains them.
pub(super) fn phase_prefix(f: &Fixture, parts: &MipParts, report: &HullMipCutReport) -> MipParts {
    let mut prefix = f.parts();
    let extra = &report.signed.as_ref().unwrap().component_rows;
    let end = parts.problem.num_rows() - extra.domination.appended_rows - extra.joint.appended_rows;
    for r in &parts.problem.rows()[prefix.problem.num_rows()..end] {
        prefix.problem.add_row(
            r.lb,
            r.ub,
            r.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
        );
    }
    prefix
}
pub(super) fn lp_floor(parts: &MipParts, unsafe_row: Row, drop_lifts: bool) -> f64 {
    let mut ir = crate::ir::MilpProblem::new();
    for c in parts.problem.cols() {
        ir.add_col(c.obj, c.lb, c.ub);
    }
    let end_rows = parts.problem.num_rows() - usize::from(drop_lifts) * 2;
    for (j, r) in parts.problem.rows().iter().enumerate().take(end_rows) {
        if j != unsafe_row.0 {
            ir.add_row(
                r.lb,
                r.ub,
                r.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
            );
        }
    }
    let m = crate::ay_lib::to_ay_model(&ir).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut s = ay_milp::LpSession::new(&m, &opts).unwrap();
    let outcome = s
        .rigorous_bound_expr(
            &[
                (m.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (m.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&outcome).unwrap()
}
pub(super) fn solve(parts: MipParts) -> MipResult {
    MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
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
