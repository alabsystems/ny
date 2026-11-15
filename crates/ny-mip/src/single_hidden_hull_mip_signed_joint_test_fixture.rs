// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original encoder and exact rational controls; authored, unexecuted.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn q(v: f64) -> Q {
    Q::from_float(v).unwrap()
}
pub(super) struct Fixture {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub dims: Vec<usize>,
    pub bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    pub fn new(bias: f64) -> Self {
        Self {
            w: vec![
                vec![1.; 3],
                vec![
                    4_f64.next_up().next_up(),
                    (-1_f64).next_up().next_up(),
                    (-3_f64).next_up().next_up(),
                    0.,
                    0.,
                    0.,
                ],
            ],
            b: vec![vec![0.; 3], vec![bias, 0.]],
            dims: vec![1, 3, 2],
            bounds: vec![vec![Bound::new(-1., 1.); 3]],
        }
    }
    pub fn source(&self) -> HullMarginProblem {
        // form_margin_down has two outward additions; preserve EXACT Source
        // masses4,-1,-3 without pretending the original output weights match.
        let p = HullMarginProblem::new(
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
        .unwrap();
        assert_eq!(p.c, vec![4., -1., -3.]);
        p
    }
    pub fn original(&self) -> MipParts {
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
    pub fn parts(&self) -> MipParts {
        let mut p = self.original();
        let vars = p.binary_vars.clone();
        assert!(p.problem.relax_binary_integrality(&[vars[0]]));
        p.binary_vars.remove(0);
        p.binary_widths.remove(0);
        p
    }
    pub fn encoding(&self, p: &MipParts) -> HullMipEncoding<'_> {
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
pub(super) fn plan(p: &HullMarginProblem, units: u8) -> Plan {
    assert!(units <= 2);
    let mut components = Vec::new();
    for (negative, units) in [(1, units), (2, 2 - units)] {
        let positive = if units == 0 { vec![] } else { vec![0] };
        let alpha = if units == 0 {
            vec![]
        } else {
            vec![
                0.,
                (-p.c[negative] / (p.c[0] * f64::from(units) * 0.5)).clamp(0., 1.),
            ]
        };
        components.push(Component {
            negative: vec![negative],
            positive,
            positive_units: if units == 0 { vec![] } else { vec![units] },
            g: vec![0.],
            alpha,
        });
    }
    Plan {
        components,
        leftover: vec![],
    }
}
pub(super) fn cuts(p: &HullMarginProblem, units: u8) -> HullRootMipCuts<'_> {
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let mut c = super::super::super::prepare(p, m, lower, end(), &mut || true).unwrap();
    let plan = plan(p, units);
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&c).collect());
    let row = row(&c, &plan, checked, cost, &b, &seal);
    component_rows::admit(&c, &plan, &b).unwrap();
    c.signed = Some(row);
    c
}
pub(super) fn append(f: &Fixture, units: u8) -> (MipParts, HullMipCutReport, usize) {
    let p = f.source();
    let c = cuts(&p, units);
    let mut parts = f.parts();
    let before = parts.problem.num_rows();
    let enc = f.encoding(&parts);
    let report = c.append_to(&mut parts, &enc, end()).unwrap();
    let s = report.signed.as_ref().unwrap();
    assert_eq!(s.source_replays, 1);
    assert_eq!(s.component_rows.joint.appended_rows, 1);
    assert_eq!(s.component_rows.joint.checked_masks, 4);
    assert_eq!(s.component_rows.joint.checker_slots, 1);
    assert_eq!(report.proposal_lp_calls, 0);
    assert!(s.work_units <= MAX_WORK && s.admitted_bytes <= MAX_BYTES);
    (parts, report, before)
}
pub(super) fn point(
    f: &Fixture,
    parts: &MipParts,
    x: Q,
    heads: [Q; 3],
    zero_mask: usize,
) -> Vec<Q> {
    let original = f.original();
    let enc = f.encoding(&original);
    let columns = binding::head_columns(&enc, &original).unwrap();
    let mut v = vec![Q::zero(); parts.num_cols];
    v[parts.input_vars[0].0] = x.clone();
    for j in 0..3 {
        v[1 + j] = x.clone();
        v[columns[j].0] = heads[j].clone();
        v[original.binary_vars[j].0] = q(if x == Q::zero() {
            ((zero_mask >> j) & 1) as f64
        } else if x > Q::zero() {
            1.
        } else {
            0.
        });
    }
    v[parts.output_vars[0].0] = q(f.b[1][0]) + (0..3).map(|j| q(f.w[1][j]) * &heads[j]).sum::<Q>();
    v
}
pub(super) fn fractional(f: &Fixture, parts: &MipParts, dyadic: bool) -> Vec<Q> {
    let heads = if dyadic {
        [q(3. / 16.), q(3. / 8.), q(1. / 4.)]
    } else {
        [q(1. / 4.), q(1. / 2.), Q::new(1.into(), 3.into())]
    };
    let mut v = point(f, parts, Q::zero(), heads, 0);
    let original = f.original();
    for col in original.binary_vars {
        v[col.0] = q(0.5);
    }
    v
}
pub(super) fn check_rows(ir: &crate::ir::MilpProblem, v: &[Q]) {
    check_rows_except(ir, v, None);
}
pub(super) fn check_rows_except(ir: &crate::ir::MilpProblem, v: &[Q], skip: Option<usize>) {
    for (j, c) in ir.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || v[j] >= q(c.lb));
        assert!(!c.ub.is_finite() || v[j] <= q(c.ub));
    }
    for (j, r) in ir.rows().iter().enumerate() {
        if skip == Some(j) {
            continue;
        }
        let lhs: Q = r.coeffs.iter().map(|(i, a)| q(*a) * &v[*i]).sum();
        assert!(!r.lb.is_finite() || lhs >= q(r.lb), "row {j} lower");
        assert!(!r.ub.is_finite() || lhs <= q(r.ub), "row {j} upper");
    }
}
pub(super) fn lp(ir: &crate::ir::MilpProblem, parts: &MipParts, unsafe_row: usize) -> f64 {
    let mut relaxed = crate::ir::MilpProblem::new();
    for c in ir.cols() {
        relaxed.add_col(c.obj, c.lb, c.ub);
    }
    for (i, r) in ir.rows().iter().enumerate() {
        if i != unsafe_row {
            relaxed.add_row(
                r.lb,
                r.ub,
                r.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
            );
        }
    }
    let m = crate::ay_lib::to_ay_model(&relaxed).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut s = ay_milp::LpSession::new(&m, &opts).unwrap();
    let result = s
        .rigorous_bound_expr(
            &[
                (m.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (m.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&result).unwrap()
}
