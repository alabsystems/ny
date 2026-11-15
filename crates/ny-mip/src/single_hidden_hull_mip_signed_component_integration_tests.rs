// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied finite planes, real encoder/LP/AY; no default-discovery claim.
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::sync::Arc;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(v: f64) -> Q {
    Q::from_float(v).unwrap()
}
struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dimensions: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    // A=8*relu(x)-sum_8 relu(x)=0; B=2*relu(y)-relu(y)=relu(y).
    // Every negative is unique. The partition below is supplied, not discovered.
    fn new(bias: f64) -> Self {
        let mut first = vec![0.; 22];
        for j in 0..9 {
            first[2 * j] = 1.;
        }
        first[19] = 1.;
        first[21] = 1.;
        let mut output = vec![-1.; 11];
        output[0] = 8.;
        output[9] = 2.;
        output.extend([0.; 11]);
        Self {
            weights: vec![first, output],
            biases: vec![vec![0.; 11], vec![bias, 0.]],
            dimensions: vec![2, 11, 2],
            bounds: vec![vec![Bound::new(-1., 1.); 11]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            2,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 2],
            vec![1.; 2],
        )
        .unwrap()
    }
    fn original_parts(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dimensions,
            &[Bound::new(-1., 1.); 2],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    }
    fn sign_reduced_parts(&self) -> MipParts {
        let mut parts = self.original_parts();
        let vars = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        assert_eq!(vars.len(), 11);
        assert!(parts.problem.relax_binary_integrality(&[vars[0], vars[9]]));
        let keep: Vec<_> = (0..11).filter(|j| *j != 0 && *j != 9).collect();
        parts.binary_vars = keep.iter().map(|&j| vars[j]).collect();
        parts.binary_widths = keep.iter().map(|&j| widths[j]).collect();
        assert_eq!(parts.binary_vars.len(), 9);
        parts
    }
    fn binding(&self, parts: &MipParts) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dimensions,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(parts.problem.num_rows() - 1),
        }
    }
    fn original_margin(&self, x: &[f64]) -> Q {
        q(self.biases[1][0]) + q(x[1]).max(Q::zero())
    }
}
fn supplied_plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: (1..9).collect(),
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0., 0.],
                alpha: (0usize..256).map(|m| m.count_ones() as f64 / 8.).collect(),
            },
            Component {
                negative: vec![10],
                positive: vec![9],
                positive_units: vec![2],
                g: vec![0., 1.],
                alpha: vec![0., 1.],
            },
        ],
        leftover: vec![],
    }
}
fn supplied_cuts(p: &HullMarginProblem, export_components: bool) -> HullRootMipCuts<'_> {
    let mut m = HullMultipliers::zeros(p);
    m.alpha[0] = 0.5;
    m.alpha[9] = 0.25;
    for j in (1..9).chain([10]) {
        for i in 0..2 {
            m.g[2 * j + i] = 0.5 * p.w_row(j)[i];
        }
    }
    let scalar = certify_hull_lower_bound(p, &m).unwrap();
    let mut cuts = super::super::super::prepare(p, m, scalar, end(), &mut || true).unwrap();
    let plan = supplied_plan();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    let signed = row(&cuts, &plan, checked, cost, &b, &seal);
    if export_components {
        // Same production admission and first-publication replay, no new pass.
        admit(&cuts, &plan, &b).unwrap();
    }
    cuts.signed = Some(signed);
    cuts
}
fn appended(f: &Fixture, components: bool) -> (MipParts, HullMipCutReport, Row) {
    let p = f.source();
    let cuts = supplied_cuts(&p, components);
    let mut parts = f.sign_reduced_parts();
    let before = parts.problem.clone();
    let vars = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let encoding = f.binding(&parts);
    let unsafe_row = encoding.unsafe_row;
    let report = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.binary_vars, vars);
    assert_eq!(parts.binary_widths, widths);
    let signed = report.signed.as_ref().unwrap();
    assert_eq!(signed.source_replays, 1);
    assert_eq!(signed.checked_masks, 2 * (256 + 2));
    assert_eq!(report.proposal_lp_calls, 0); // Authored tuple, not an invented LP call.
    assert_eq!(
        signed.component_rows.appended_rows,
        if components { 2 } else { 0 }
    );
    if components {
        assert_eq!(signed.component_rows.collection_attempts, 1);
        assert_eq!(signed.component_rows.collected_components, 2);
        assert_eq!(signed.component_rows.appended_terms, 12);
        assert_eq!(signed.component_rows.reserved_work_units, 0);
        assert!(signed.component_rows.work_units > 0);
        assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    }
    (parts, report, unsafe_row)
}
// Isolate the qualified component/phase family while the REAL default appender
// still constructs later domination/union rows and the full model is certified.
fn component_prefix(f: &Fixture, parts: &MipParts, report: &HullMipCutReport) -> MipParts {
    let mut prefix = f.sign_reduced_parts();
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
fn actual_lp_floor(parts: &MipParts, unsafe_row: Row) -> f64 {
    // Diagnostic LP: retain ALL original/cut rows except the one unsafe row.
    let mut ir = crate::ir::MilpProblem::new();
    for col in parts.problem.cols() {
        ir.add_col(col.obj, col.lb, col.ub);
    }
    for (i, row) in parts.problem.rows().iter().enumerate() {
        if i != unsafe_row.0 {
            ir.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|&(j, c)| (crate::ir::Col(j), c)),
            );
        }
    }
    let model = crate::ay_lib::to_ay_model(&ir).unwrap();
    let opts = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &opts).unwrap();
    let result = session
        .rigorous_bound_expr(
            &[
                (model.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (model.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&result).unwrap()
}
fn actual_ay(parts: MipParts) -> MipResult {
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    );
    let started = Instant::now();
    let result = solver.check_feasibility().unwrap();
    eprintln!(
        "component-row original AY result={result:?} elapsed={:?}",
        started.elapsed()
    );
    result
}
fn values(f: &Fixture, parts: &MipParts, x: [f64; 2], relaxed_heads: Option<&[f64]>) -> Vec<Q> {
    let original = f.original_parts();
    let encoding = f.binding(&original);
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let mut v = vec![Q::zero(); parts.num_cols];
    v[parts.input_vars[0].0] = q(x[0]);
    v[parts.input_vars[1].0] = q(x[1]);
    for j in 0..11 {
        let z = if j < 9 { x[0] } else { x[1] };
        let h = relaxed_heads.map_or(z.max(0.), |r| r[j]);
        v[2 + j] = q(z);
        v[heads[j].0] = q(h);
        v[original.binary_vars[j].0] = q(if relaxed_heads.is_some() {
            h
        } else if z > 0. {
            1.
        } else {
            0.
        });
    }
    v[parts.output_vars[0].0] = q(f.biases[1][0])
        + (0..11)
            .map(|j| q(f.weights[1][j]) * &v[heads[j].0])
            .sum::<Q>();
    v
}
fn assert_lp_rows(parts: &MipParts, unsafe_row: Row, v: &[Q]) {
    for (i, col) in parts.problem.cols().iter().enumerate() {
        if col.lb.is_finite() {
            assert!(v[i] >= q(col.lb));
        }
        if col.ub.is_finite() {
            assert!(v[i] <= q(col.ub));
        }
    }
    for (i, row) in parts.problem.rows().iter().enumerate() {
        if i == unsafe_row.0 {
            continue;
        }
        let lhs = row.coeffs.iter().map(|&(j, c)| q(c) * &v[j]).sum::<Q>();
        if row.lb.is_finite() {
            assert!(lhs >= q(row.lb), "row {i}: {lhs} < {}", row.lb);
        }
        if row.ub.is_finite() {
            assert!(lhs <= q(row.ub), "row {i}: {lhs} > {}", row.ub);
        }
    }
}
#[test]
fn signed_component_rows_actual_encoder_lp_three_strict_floors_and_ay_certified_unsat() {
    let f = Fixture::new(0.75);
    let original = f.sign_reduced_parts();
    let unsafe_row = f.binding(&original).unsafe_row;
    let before = actual_lp_floor(&original, unsafe_row);
    let (aggregate, old_report, aggregate_unsafe) = appended(&f, false);
    let middle = actual_lp_floor(&aggregate, aggregate_unsafe);
    let (components, report, component_unsafe) = appended(&f, true);
    let prefix = component_prefix(&f, &components, &report);
    let after = actual_lp_floor(&prefix, component_unsafe);
    let full_after = actual_lp_floor(&components, component_unsafe);
    assert!(full_after >= after - 1e-6);
    assert!(
        report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .domination
            .appended_rows
            > 0
    );
    eprintln!("component-row actual LP floors before={before:?} aggregate={middle:?} components={after:?}");
    assert!(before > -3.751 && before <= -3.75);
    assert!(middle > -1.251 && middle <= -1.25 && middle > before + 2.49);
    assert!(after > 0.249 && after <= 0.25 && after > middle + 1.49);
    let phase_rows = report
        .signed
        .as_ref()
        .unwrap()
        .component_rows
        .phase
        .appended_rows;
    assert_eq!(phase_rows, 1);
    let extra = &report.signed.as_ref().unwrap().component_rows;
    assert_eq!(
        report.cuts,
        old_report.cuts
            + 2
            + phase_rows
            + extra.domination.appended_rows
            + extra.joint.appended_rows
    );
    // Independent feasible LP points fix the stated upper endpoints too.
    let mut h = vec![0.; 11];
    h[1..9].fill(0.25);
    let witness = values(&f, &aggregate, [0., -1.], Some(&h));
    assert_lp_rows(&aggregate, aggregate_unsafe, &witness);
    assert_eq!(witness[aggregate.output_vars[0].0], q(-1.25));
    let mut h = vec![0.; 11];
    h[10] = 0.5;
    let witness = values(&f, &components, [0., 0.], Some(&h));
    assert_lp_rows(&prefix, component_unsafe, &witness);
    assert_eq!(witness[components.output_vars[0].0], q(0.25));
    // The checked WHOLE-Q scalar stays negative: row count is not native proof.
    assert!(report.signed.as_ref().unwrap().lower.unwrap() < -1.249);
    let result = actual_ay(components);
    assert!(
        matches!(result, MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}
#[test]
fn signed_component_rows_actual_unsafe_and_zero_tie_keep_original_sat_witnesses() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias);
        let (parts, report, unsafe_row) = appended(&f, true);
        assert!(report.signed.as_ref().unwrap().lower.unwrap() <= bias);
        let original = values(&f, &parts, [0., 0.], None);
        assert_lp_rows(&parts, unsafe_row, &original);
        assert_eq!(f.original_margin(&[0., 0.]), q(bias));
        let result = actual_ay(parts);
        let MipResult::Sat { input_values, .. } = result else {
            panic!("original unsafe/tie must remain SAT: {result:?}")
        };
        assert_eq!(input_values.len(), 2);
        assert!(input_values
            .iter()
            .all(|x| x.is_finite() && *x >= -1. && *x <= 1.));
        assert!(f.original_margin(&input_values) <= Q::zero());
    }
}
#[test]
fn signed_component_rows_actual_original_graph_rational_samples_include_all_exported_rows() {
    let f = Fixture::new(0.75);
    let (parts, report, unsafe_row) = appended(&f, true);
    assert_eq!(
        report.signed.as_ref().unwrap().component_rows.appended_rows,
        2
    );
    for x in [-1., -0.5, 0., 0.5, 1.] {
        for y in [-1., -0.5, 0., 0.5, 1.] {
            let v = values(&f, &parts, [x, y], None);
            assert_lp_rows(&parts, unsafe_row, &v);
            assert_eq!(v[parts.output_vars[0].0], f.original_margin(&[x, y]));
        }
    }
}
