// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-encoder controls; external and unexecuted at authoring time.
use super::super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
pub(super) const EPS: f64 = 1. / 256.;
pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
pub(super) fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
pub(super) struct Fixture {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub dims: Vec<usize>,
    pub input: Vec<Bound>,
    pub bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    pub fn large(bias: f64) -> Self {
        let (d, h) = (25, 25);
        let mut w = vec![0.; d * h];
        for j in 0..h {
            w[j * d] = 1.;
            if j != 0 {
                w[j * d + j] = EPS;
            }
        }
        let mut out = vec![0.; 2 * h];
        out[0] = 24.;
        out[1..h].fill(-1.);
        let mut bounds = vec![Bound::new(-(1. + EPS) as f32, (1. + EPS) as f32); h];
        bounds[0] = Bound::new(-1., 1.);
        Self {
            w: vec![w, out],
            b: vec![vec![0.; h], vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![bounds],
        }
    }
    pub fn sparse(bias: f64) -> Self {
        // P1=R(x), P2=R(y), N=R(x+y). Raw original coefficients preserve ties.
        Self {
            w: vec![vec![1., 0., 0., 1., 1., 1.], vec![1., 1., -1., 0., 0., 0.]],
            b: vec![vec![0.; 3], vec![bias, 0.]],
            dims: vec![2, 3, 2],
            input: vec![Bound::new(-1., 1.); 2],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-2., 2.),
            ]],
        }
    }
    pub fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            self.dims[0],
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            self.input.iter().map(|b| f64::from(b.lower())).collect(),
            self.input.iter().map(|b| f64::from(b.upper())).collect(),
        )
        .unwrap()
    }
    pub fn original(&self) -> MipParts {
        let mut e =
            crate::encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds)
                .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    pub fn parts(&self) -> MipParts {
        let mut p = self.original();
        let source = self.source();
        let mut cursor = 0;
        let mut relaxed = Vec::new();
        for (j, b) in self.bounds[0].iter().enumerate() {
            if b.lower() < 0. && b.upper() > 0. {
                if source.c[j] > 0. {
                    relaxed.push(p.binary_vars[cursor]);
                }
                cursor += 1;
            }
        }
        for col in relaxed {
            assert!(p.problem.relax_binary_integrality(&[col]));
            let at = p.binary_vars.iter().position(|v| *v == col).unwrap();
            p.binary_vars.remove(at);
            p.binary_widths.remove(at);
        }
        p
    }
    pub fn encoding(&self) -> HullMipEncoding<'_> {
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
pub(super) fn plan(p: &HullMarginProblem, allocation: &[u8]) -> Plan {
    let negative: Vec<_> = (0..p.hidden_dim).filter(|&j| p.c[j] < 0.).collect();
    let positive: Vec<_> = (0..p.hidden_dim).filter(|&j| p.c[j] > 0.).collect();
    assert_eq!(allocation.len(), negative.chunks(8).len());
    assert_eq!(allocation.iter().map(|&u| usize::from(u)).sum::<usize>(), 2);
    let components = negative
        .chunks(8)
        .zip(allocation)
        .map(|(n, &units)| {
            let pos = if units == 0 { vec![] } else { positive.clone() };
            Component {
                negative: n.to_vec(),
                positive_units: vec![units; pos.len()],
                alpha: vec![0.5; (1usize << n.len()) * pos.len()],
                positive: pos,
                g: vec![0.; p.input_dim],
            }
        })
        .collect();
    Plan {
        components,
        leftover: vec![],
    }
}
pub(super) fn cuts<'a>(p: &'a HullMarginProblem, allocation: &[u8]) -> HullRootMipCuts<'a> {
    cuts_with_plan(p, plan(p, allocation))
}
pub(super) fn cuts_with_plan(p: &HullMarginProblem, plan: Plan) -> HullRootMipCuts<'_> {
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let mut c = prepare(p, m, lower, end(), &mut || true).unwrap();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&c).collect());
    let row = row(&c, &plan, checked, cost, &b, &seal);
    component_rows::admit(&c, &plan, &b).unwrap();
    assert!(row.lower < 0., "fixture requires an open retained root");
    c.signed = Some(row);
    c
}
pub(super) fn append(f: &Fixture, allocation: &[u8]) -> (MipParts, HullMipCutReport, usize) {
    let p = f.source();
    let c = cuts(&p, allocation);
    let mut parts = f.parts();
    let before = parts.problem.num_rows();
    let report = c.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert_eq!(report.proposal_lp_calls, 0);
    let s = report.signed.as_ref().unwrap();
    assert_eq!(s.source_replays, 1);
    assert!(s.work_units <= MAX_WORK && s.admitted_bytes <= MAX_BYTES);
    (parts, report, before)
}
pub(super) fn point(
    f: &Fixture,
    parts: &MipParts,
    inputs: &[Q],
    head_override: Option<&[Q]>,
    zero_mask: usize,
) -> Vec<Q> {
    let original = f.original();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let (d, h) = (f.dims[0], f.dims[1]);
    assert_eq!(inputs.len(), d);
    let mut v = vec![Q::zero(); parts.num_cols];
    for i in 0..d {
        v[parts.input_vars[i].0] = inputs[i].clone();
    }
    let mut binary = 0;
    for j in 0..h {
        let z = q(f.b[0][j]) + (0..d).map(|i| q(f.w[0][j * d + i]) * &inputs[i]).sum::<Q>();
        v[d + j] = z.clone();
        v[heads[j].0] = head_override.map_or_else(|| z.clone().max(Q::zero()), |r| r[j].clone());
        let b = f.bounds[0][j];
        if b.lower() < 0. && b.upper() > 0. {
            let phase = if z.is_zero() {
                (zero_mask >> j) & 1
            } else {
                usize::from(z > Q::zero())
            };
            v[original.binary_vars[binary].0] = q(phase as f64);
            binary += 1;
        }
    }
    for o in 0..2 {
        v[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..h)
                .map(|j| q(f.w[1][o * h + j]) * &v[heads[j].0])
                .sum::<Q>();
    }
    v
}
pub(super) fn fractional(f: &Fixture, parts: &MipParts) -> Vec<Q> {
    assert_eq!(f.dims, vec![25, 25, 2]);
    let mut heads = vec![q(0.25); 25];
    heads[0] = q(3. / 16.);
    let mut v = point(f, parts, &vec![Q::zero(); 25], Some(&heads), 0);
    for col in f.original().binary_vars {
        v[col.0] = q(0.5);
    }
    v
}
pub(super) fn rows_hold(ir: &crate::ir::MilpProblem, v: &[Q], skip: Option<usize>) {
    // Continuous-relaxation membership: integer flags are intentionally not asserted.
    for (i, c) in ir.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || v[i] >= q(c.lb));
        assert!(!c.ub.is_finite() || v[i] <= q(c.ub));
    }
    for (i, r) in ir.rows().iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &v[*j]).sum();
        assert!(!r.lb.is_finite() || lhs >= q(r.lb), "row{i} lower");
        assert!(!r.ub.is_finite() || lhs <= q(r.ub), "row{i} upper");
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
                r.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
            );
        }
    }
    let m = crate::ay_lib::to_ay_model(&relaxed).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&m, &opts).unwrap();
    let bound = session
        .rigorous_bound_expr(
            &[
                (m.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (m.col_at(parts.output_vars[1].0).unwrap(), -1.),
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
