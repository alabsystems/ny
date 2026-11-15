// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied-table separation and actual policy routing are distinct controls.
use super::*;
use crate::single_hidden_hull::mip_cuts as root;
use num_rational::BigRational as Q;
use num_traits::Zero;

const E: f64 = 1. / 1024.;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn coexistence() -> Self {
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    1.,
                    0.,
                    0.,
                    E,
                    0.,
                    1.,
                    0.,
                    2. * E,
                    0.,
                    0.,
                    1.,
                    3. * E,
                    0.,
                    0.,
                    0.,
                    0.,
                ],
                vec![1., 1., 1., -1., -1., -1., -E, 0., 0., 0., 0., 0., 0., 0.],
            ],
            biases: vec![vec![0., 0., 0., 0., 0., 0., -1.], vec![0.75, 0.]],
            dims: vec![4, 7, 2],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-(1. + E) as f32, (1. + E) as f32),
                Bound::new(-(1. + 2. * E) as f32, (1. + 2. * E) as f32),
                Bound::new(-(1. + 3. * E) as f32, (1. + 3. * E) as f32),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn zero_unary() -> Self {
        Self {
            weights: vec![
                vec![0., 1., -1., 0.],
                vec![1., -1., -1., -E, 0., 0., 0., 0.],
            ],
            biases: vec![vec![1., 0., 0., -1.], vec![0.25, 0.]],
            dims: vec![1, 4, 2],
            bounds: vec![vec![
                Bound::new(1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn supplied(bias: f64) -> Self {
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0., // P0=x
                    0.,
                    1.,
                    0., // P1=y
                    1.,
                    0.,
                    E, // N0=x+e*z
                    0.,
                    1.,
                    2. * E, // N1=y+2e*z
                    0.,
                    0.,
                    0., // stable-dead spectator
                ],
                vec![1., 1., -1., -1., -E, 0., 0., 0., 0., 0.],
            ],
            biases: vec![vec![0., 0., 0., 0., -1.], vec![bias, 0.]],
            dims: vec![3, 5, 2],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-(1. + E) as f32, (1. + E) as f32),
                Bound::new(-(1. + 2. * E) as f32, (1. + 2. * E) as f32),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn with_unused_outputs(mut self, n: usize) -> Self {
        assert!(n >= 2);
        self.weights[1].resize(n * self.dims[1], 0.);
        self.biases[1].resize(n, 0.);
        self.dims[2] = n;
        self
    }
    fn public() -> Self {
        let (d, h) = (30, 17);
        let mut first = vec![0.; d * h];
        for j in 0..h {
            first[j * d + d - 1] = 1.;
        }
        let mut output = vec![0.; 2 * h];
        output[0] = 16.;
        output[1..h].fill(-1.);
        Self {
            weights: vec![first, output],
            biases: vec![vec![0.; h], vec![0.25, 0.]],
            dims: vec![d, h, 2],
            bounds: vec![vec![Bound::new(-1., 1.); h]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            self.dims[0],
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; self.dims[0]],
            vec![1.; self.dims[0]],
        )
        .unwrap()
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &vec![Bound::new(-1., 1.); self.dims[0]],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
    fn outputs(&self, parts: &MipParts, point: &mut [Q]) {
        let original = self.original();
        let heads = binding::head_columns(&self.encoding(), &original).unwrap();
        let h = self.dims[1];
        for o in 0..2 {
            point[parts.output_vars[o].0] = q(self.biases[1][o])
                + (0..h)
                    .map(|j| q(self.weights[1][o * h + j]) * &point[heads[j].0])
                    .sum::<Q>();
        }
    }
    fn point(&self, parts: &MipParts, inputs: &[Q], zero_mask: usize) -> Vec<Q> {
        let original = self.original();
        let heads = binding::head_columns(&self.encoding(), &original).unwrap();
        let (d, h) = (self.dims[0], self.dims[1]);
        assert_eq!(inputs.len(), d);
        assert_eq!(
            original.binary_vars.len(),
            self.bounds[0]
                .iter()
                .filter(|b| b.lower() < 0. && b.upper() > 0.)
                .count()
        );
        let mut point = vec![Q::zero(); parts.num_cols];
        for i in 0..d {
            point[parts.input_vars[i].0] = inputs[i].clone();
        }
        for j in 0..h {
            let z = q(self.biases[0][j])
                + (0..d)
                    .map(|i| q(self.weights[0][j * d + i]) * &inputs[i])
                    .sum::<Q>();
            point[d + j] = z.clone();
            point[heads[j].0] = z.clone().max(Q::zero());
            let bound = self.bounds[0][j];
            if bound.lower() < 0. && bound.upper() > 0. {
                let selector = crate::ir::Col(heads[j].0 + 1);
                assert!(original.binary_vars.contains(&selector));
                point[selector.0] = q(if z.is_zero() {
                    ((zero_mask >> j) & 1) as f64
                } else {
                    usize::from(z > Q::zero()) as f64
                });
            }
        }
        self.outputs(parts, &mut point);
        point
    }
}
fn plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2, 3],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.; 3],
                alpha: vec![0., 0., 0., 0., 0., 1., 0.75, 1.],
            },
            Component {
                negative: vec![4],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; 3],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    }
}
fn base(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let m = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    root::prepare(p, m, lower, end(), &mut || true).unwrap()
}
fn supplied_advice(
    p: &HullMarginProblem,
    until: Instant,
) -> (
    std::sync::Mutex<primal_advice::Captured<'_>>,
    primal_capture::Report,
) {
    // x,y,z,P0,P1,mu0,t0x,t0z,mu1,t1y,t1z. Activities are 1-mu.
    captured_advice(
        p,
        until,
        11,
        vec![0., 0., 0., 0., 0., 77. / 256., 0., 0., 77. / 128., 0., 0.],
    )
}
fn captured_advice(
    p: &HullMarginProblem,
    until: Instant,
    columns: usize,
    raw: Vec<f64>,
) -> (
    std::sync::Mutex<primal_advice::Captured<'_>>,
    primal_capture::Report,
) {
    // Real descriptor construction and prepaid capture, with deliberately
    // synthetic finite raw advice. No LP/primal-feasibility discovery claim.
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let mut attempt = primal_capture::Attempt::new_with_activity(p, until);
    attempt.admit(&roles);
    let lp = build_hull_lp_with_capture(p, &roles, &bounds, attempt.capture.as_mut()).unwrap();
    assert_eq!(lp.model.num_cols(), columns);
    assert!(attempt.enable_same_call(lp.model.num_cols()));
    assert_eq!(raw.len(), columns);
    attempt.finish(Some(raw.into_boxed_slice()), &mut || true);
    assert!(attempt.snapshot().decoded);
    let report = Arc::clone(&attempt.report);
    (attempt.take_account().unwrap(), report)
}
fn supplied(p: &HullMarginProblem, cube: bool) -> HullRootMipCuts<'_> {
    let mut cuts = base(p);
    // Bank the SAME supplied activity debt in both comparators before search.
    let (account, report) = supplied_advice(p, cuts.deadline);
    cuts.primal = Some(account);
    cuts.primal_report = Some(report);
    let plan = plan();
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    {
        // Before any search, the actual banked receipt is readable only under
        // its original search clock; later publication authority is refused.
        let advice = cuts.primal.as_ref().unwrap().lock().unwrap();
        assert!(b.authority > b.search_end);
        assert!(advice
            .relaxed_activity_debt(p, b.authority, &b.state)
            .is_none());
        assert!(advice.relaxed_activity(p, b.authority, &b.state).is_none());
        assert_eq!(
            advice.relaxed_activity_debt(p, b.search_end, &b.state),
            Some((96, 288))
        );
        let (ids, activity) = advice.relaxed_activity(p, b.search_end, &b.state).unwrap();
        assert_eq!(ids, &[2, 3]);
        assert_eq!(activity, &[179. / 256., 51. / 128.]);
    }
    b.phase_all_enabled = true;
    b.phase_cube_enabled = cube;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    assert!((checked.root_lower - (p.d - 1. - E)).abs() < 1e-11);
    // Both models retain the same real funded root policy and complete C2
    // owner, including its stable-dead spectator. No old family is hidden.
    root_coupling::admit(&cuts, &plan, cost, &mut b).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    component_rows::admit(&cuts, &plan, &b).unwrap();
    assert!(all_mode(&b));
    assert_eq!(
        b.state.lock().unwrap().component_rows.phase.cube.admitted,
        cube
    );
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}
fn zero_unary_owner(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let mut cuts = base(p);
    // Synthetic same-format activities(.75,.25) orient10→00→01. This
    // produces a nonconstant support; it is not original-LP discovery.
    let (account, report) = captured_advice(p, cuts.deadline, 5, vec![0., 0.25, 0., 0.75, 0.]);
    cuts.primal = Some(account);
    cuts.primal_report = Some(report);
    let plan = Plan {
        components: vec![
            Component {
                negative: vec![1, 2],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.],
                alpha: vec![1.; 4],
            },
            Component {
                negative: vec![3],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    };
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    b.portfolio_enabled = true;
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    component_rows::admit(&cuts, &plan, &b).unwrap();
    assert!(b.root_coupling.is_none());
    assert!(b.state.lock().unwrap().component_rows.phase.cube.admitted);
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}
fn rows_hold(parts: &MipParts, point: &[Q], skip: Option<usize>) {
    for (j, c) in parts.problem.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || point[j] >= q(c.lb), "column{j} lower");
        assert!(!c.ub.is_finite() || point[j] <= q(c.ub), "column{j} upper");
    }
    for (i, row) in parts.problem.rows().iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let lhs: Q = row.coeffs.iter().map(|(j, a)| q(*a) * &point[*j]).sum();
        assert!(!row.lb.is_finite() || lhs >= q(row.lb), "row{i} lower");
        assert!(!row.ub.is_finite() || lhs <= q(row.ub), "row{i} upper");
    }
}
fn appended(f: &Fixture, cube: bool) -> (MipParts, HullMipCutReport) {
    let p = f.source();
    let mut cuts = supplied(&p, cube);
    let captured_work = cuts.lp_primal_report().work_units;
    if cube {
        // Admission already transferred advice under the bank's original
        // search clock. Publication must not read the transient account again.
        let before = cuts.signed.as_ref().unwrap().stats();
        assert_eq!(
            before.component_rows.phase.cube.activity_transfer_attempts,
            1
        );
        assert_eq!(
            before.component_rows.phase.cube.transferred_activity_values,
            2
        );
        drop(cuts.primal.take().unwrap());
        assert!(cuts.primal.is_none());
        assert_eq!(cuts.lp_primal_report().work_units, captured_work);
        assert_eq!(
            cuts.signed.as_ref().unwrap().stats().work_units,
            before.work_units
        );
    }
    let mut parts = f.original();
    let old = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = report.signed.as_ref().unwrap();
    assert!(stats.root_coupling.admitted);
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.component_rows.phase.collection_attempts, 1);
    assert_eq!(
        stats.component_rows.phase.appended_rows,
        2 + usize::from(cube)
    );
    assert_eq!(
        stats.component_rows.phase.cube.appended_rows,
        usize::from(cube)
    );
    if cube {
        assert_eq!(
            stats.component_rows.phase.cube.activity_transfer_attempts,
            1
        );
        assert_eq!(
            stats.component_rows.phase.cube.transferred_activity_values,
            2
        );
        assert_eq!(stats.component_rows.phase.cube.activity_values, 2);
        assert_eq!(stats.component_rows.phase.cube.activity_components, 1);
        assert_eq!(
            stats.component_rows.phase.cube.capture_total_work,
            captured_work
        );
    }
    assert_eq!(report.proposal_lp_calls, 0); // supplied owner and advice
    assert_eq!(parts.problem.cols(), old.cols());
    assert_eq!(&parts.problem.rows()[..old.num_rows()], old.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.num_rows() - old.num_rows(), report.cuts);
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    (parts, report)
}
fn weighted(row: &RowSpec, weight: Q, coefficients: &mut [Q], rhs: &mut Q) {
    let bound = if weight >= Q::zero() { row.lb } else { row.ub };
    assert!(bound.is_finite());
    *rhs += &weight * q(bound);
    for (j, a) in &row.coeffs {
        coefficients[*j] += &weight * q(*a);
    }
}
fn lp_floor(parts: &MipParts, property: usize) -> f64 {
    let mut ir = crate::ir::MilpProblem::new();
    for col in parts.problem.cols() {
        ir.add_col(col.obj, col.lb, col.ub);
    }
    for (i, row) in parts.problem.rows().iter().enumerate() {
        if i != property {
            ir.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|(j, a)| (crate::ir::Col(*j), *a)),
            );
        }
    }
    let model = crate::ay_lib::to_ay_model(&ir).unwrap();
    let options = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &options).unwrap();
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

#[test]
fn cube_supplied_full_table_strictly_closes_all_unary_original_lp_and_ay() {
    let f = Fixture::supplied(11. / 16.);
    let original = f.original();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let (old, _) = appended(&f, false);
    let (lifted, _) = appended(&f, true);
    // Fixed dyadic fractional witness; every old row and the property remain.
    let mut fake = f.point(&old, &[q(0.), q(0.), q(0.)], 0);
    for (j, value) in [0.5, 0.5, 179. / 256., 51. / 128.].into_iter().enumerate() {
        fake[original.binary_vars[j].0] = q(value);
    }
    fake[heads[2].0] = q(77. / 256.);
    fake[heads[3].0] = q(51. / 128.);
    f.outputs(&old, &mut fake);
    rows_hold(&old, &fake, None);
    assert_eq!(
        &fake[old.output_vars[0].0] - &fake[old.output_vars[1].0],
        q(-3. / 256.)
    );

    let selectors = [original.binary_vars[2].0, original.binary_vars[3].0];
    let selected = |both: bool| {
        let rows: Vec<_> = lifted.problem.rows()[original.problem.num_rows()..]
            .iter()
            .filter(|r| {
                r.lb.is_finite()
                    && r.coeffs.iter().any(|(j, a)| *j == selectors[0] && *a != 0.)
                    && (r.coeffs.iter().any(|(j, a)| *j == selectors[1] && *a != 0.) == both)
            })
            .collect();
        assert_eq!(rows.len(), 1);
        rows[0]
    };
    let unary = selected(false);
    let cube = selected(true);
    let lhs: Q = cube.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
    assert!(lhs < q(cube.lb));

    // Fixed reviewed weights: original encoder1/3, unary2/9, cube4/9.
    // Every remaining coefficient, including ideal selector residuals, is
    // priced below; stored outward-rounded rows need not equal ideal rows.
    let a = Q::new(1.into(), 3.into());
    let b = Q::new(2.into(), 9.into());
    let c = Q::new(4.into(), 9.into());
    assert!(a > Q::zero() && b > Q::zero() && c > Q::zero());
    let mut coefficients = vec![Q::zero(); lifted.num_cols];
    let mut rhs = Q::zero();
    // Both actual Source-derived support rows are doubled.
    weighted(unary, &b / q(2.), &mut coefficients, &mut rhs);
    weighted(cube, &c / q(2.), &mut coefficients, &mut rhs);
    let n0_upper = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.ub == 1. + E
                && r.lb == f64::NEG_INFINITY
                && r.coeffs.contains(&(heads[2].0, 1.))
                && r.coeffs.contains(&(5, -1.))
                && r.coeffs.contains(&(selectors[0], 1. + E))
        })
        .unwrap();
    let n1_upper = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.ub == 0.
                && r.lb == f64::NEG_INFINITY
                && r.coeffs.contains(&(heads[3].0, 1.))
                && r.coeffs.contains(&(selectors[1], -(1. + 2. * E)))
        })
        .unwrap();
    let p0_lower = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.lb == 0.
                && r.ub == f64::INFINITY
                && r.coeffs.len() == 2
                && r.coeffs.contains(&(heads[0].0, 1.))
                && r.coeffs.contains(&(3, -1.))
        })
        .unwrap();
    weighted(n0_upper, -a.clone(), &mut coefficients, &mut rhs);
    weighted(n1_upper, -a.clone(), &mut coefficients, &mut rhs);
    weighted(p0_lower, a, &mut coefficients, &mut rhs);
    for j in [0usize, 2] {
        let affine = &original.problem.rows()[j];
        assert_eq!(affine.lb, affine.ub);
        let column = 3 + j;
        let coefficient = affine.coeffs.iter().find(|(i, _)| *i == column).unwrap().1;
        let weight = -coefficients[column].clone() / q(coefficient);
        weighted(affine, weight, &mut coefficients, &mut rhs);
        assert!(coefficients[column].is_zero());
    }
    let mut objective = vec![Q::zero(); lifted.num_cols];
    for j in 0..f.dims[1] {
        objective[heads[j].0] = q(f.weights[1][j]) - q(f.weights[1][f.dims[1] + j]);
    }
    let mut lower = q(f.biases[1][0]) - q(f.biases[1][1]) + rhs;
    for (j, (want, got)) in objective.iter().zip(&coefficients).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let column = &lifted.problem.cols()[j];
        assert!(
            column.lb.is_finite() && column.ub.is_finite(),
            "unpriced column{j}"
        );
        lower += (&residual * q(column.lb)).min(&residual * q(column.ub));
    }
    let ideal = Q::new(7.into(), 384.into());
    assert!(lower >= ideal - q(1e-11));
    assert!(lower > Q::new(1.into(), 64.into()));
    let before = lp_floor(&old, encoding.unsafe_row.0);
    let after = lp_floor(&lifted, encoding.unsafe_row.0);
    assert!(before <= -3. / 256. + 1e-10, "old all-unary LP={before}");
    assert!(after >= 7. / 384. - 1e-9, "cube LP={after}");
    let result = solve(lifted);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn cube_original_unsafe_exact_zero_and_all_closed_selector_cells_are_retained() {
    for bias in [2. * E, 3. * E] {
        let f = Fixture::supplied(bias);
        let (parts, _) = appended(&f, true);
        // Raw outputs use integer principal weights and an exact dyadic bias.
        // At (0,0,1), P0/P1 are zero and the true raw margin is bias-3e.
        for cells in 0..4 {
            let point = f.point(&parts, &[q(0.), q(0.), q(1.)], cells);
            rows_hold(&parts, &point, None);
            assert_eq!(
                &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0],
                q(bias - 3. * E)
            );
        }
        // All four principal heads are zero here. Every independent closed
        // selector choice must satisfy the exported cuts; skip only property.
        for cells in 0..16 {
            let point = f.point(&parts, &[q(0.), q(0.), q(0.)], cells);
            rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
        }
        let result = solve(parts);
        assert!(matches!(result, crate::MipResult::Sat { .. }), "{result:?}");
    }
}

#[test]
fn cube_staircase_advice_needs_full_serialized_repair_including_last_mask() {
    let table = [0., -1. - E, -2. * E, -0.25 - 3. * E];
    let heads = [2, 3];
    let weights = staircase(&table, &heads, Some(&[179. / 256., 51. / 128.])).unwrap();
    assert_eq!(&weights[..2], &[-0.25 - E, 0.75 - 2. * E]);
    let encoded = [-2. * weights[0], -2. * weights[1]];
    let lower = repair(&table, &encoded, end(), &mut || true).unwrap();
    assert!((-1.5 - 1e-12..=-1.5).contains(&lower));
    // Permuting bit and head order permutes the same real support.
    let swapped = [table[0], table[2], table[1], table[3]];
    let reordered = staircase(&swapped, &[3, 2], Some(&[51. / 128., 179. / 256.])).unwrap();
    assert_eq!(&reordered[..2], &[weights[1], weights[0]]);
    assert!(staircase(&table, &heads, Some(&[f64::NAN, 0.5])).is_none());
    assert!(staircase(&table, &heads, Some(&[0.5])).is_none());

    // A chain interpolation is not a certificate for a non-submodular table.
    let non_submodular = [0., 1., 1., 3.];
    let proposal = staircase(&non_submodular, &heads, None).unwrap();
    assert_eq!(&proposal[..2], &[1., 2.]);
    assert!(proposal[1] > non_submodular[2]); // invalid uncorrected mask01
    let coefficients = [-2. * proposal[0], -2. * proposal[1]];
    let repaired = repair(&non_submodular, &coefficients, end(), &mut || true).unwrap();
    assert!((-2. - 1e-12..=-2.).contains(&repaired));
    for mask in 0..4 {
        let right = q(2. * non_submodular[mask])
            + (0..2)
                .filter(|bit| mask & (1usize << bit) != 0)
                .map(|bit| q(coefficients[bit]))
                .sum::<Q>();
        assert!(q(repaired) <= right);
    }

    // Every one of the256 masks participates in both directed scans.
    let mut many: Vec<f64> = (0usize..256)
        .map(|m| {
            let n = m.count_ones() as f64;
            n * n
        })
        .collect();
    let head_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let w = staircase(&many, &head_ids, None).unwrap();
    let coefficients: Vec<_> = w.iter().map(|x| -2. * x).collect();
    let mut polls = 0;
    let old = repair(&many, &coefficients, end(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert_eq!(polls, 513);
    for mask in 0..256 {
        let right = q(2. * many[mask])
            + (0..8)
                .filter(|bit| mask & (1usize << bit) != 0)
                .map(|bit| q(coefficients[bit]))
                .sum::<Q>();
        assert!(q(old) <= right);
    }
    many[255] = -10_000.;
    let changed = repair(&many, &coefficients, end(), &mut || true).unwrap();
    assert!(changed < old - 1_000.); // no missing final-mask coverage
    for cancel_at in [1, 256, 257, 512, 513] {
        let mut calls = 0;
        assert!(repair(&many, &coefficients, end(), &mut || {
            calls += 1;
            calls != cancel_at
        })
        .is_none());
        assert_eq!(calls, cancel_at);
    }
    assert!(repair(&[f64::MAX; 4], &[1., 1.], end(), &mut || true).is_none());
    assert!(repair(&table, &[f64::INFINITY, 0.], end(), &mut || true).is_none());
    assert!(repair(&table, &encoded, Instant::now(), &mut || true).is_none());
}

#[test]
fn cube_public_successor_captures_same_lp_activity_and_old_all_stays_unchanged() {
    let f = Fixture::public();
    let p = f.source();
    for cube in [false, true] {
        // Actual source discovery: no supplied Plan, alpha, primal, or table.
        let attempt = if cube {
            propose_root_mip_cuts_with_mass_portfolio_phase_cube_attempt(&p, end(), Duration::ZERO)
        } else {
            propose_root_mip_cuts_with_mass_portfolio_all_phases_attempt(&p, end(), Duration::ZERO)
        };
        assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
        let cuts = attempt.cuts.expect("actual public policy owner");
        let owner = cuts.signed.as_ref().unwrap();
        assert!(owner.budget.portfolio_enabled && owner.budget.phase_all_enabled);
        assert_eq!(owner.budget.phase_cube_enabled, cube);
        assert!(all_mode(&owner.budget));
        assert_eq!(cuts.proposal_lp_calls, 1);
        let before = owner.stats();
        assert!(before.root_coupling.admitted && before.component_rows.phase.admitted);
        assert_eq!(before.component_rows.phase.cube.admitted, cube);
        {
            let advice = cuts.primal.as_ref().unwrap().lock().unwrap();
            assert!(owner.budget.authority > owner.budget.search_end);
            assert!(advice
                .relaxed_activity_debt(&p, owner.budget.authority, &owner.budget.state)
                .is_none());
            assert!(advice
                .relaxed_activity(&p, owner.budget.authority, &owner.budget.state)
                .is_none());
            let debt =
                advice.relaxed_activity_debt(&p, owner.budget.search_end, &owner.budget.state);
            if let Some(debt) = debt {
                assert_eq!(
                    debt,
                    if cube {
                        (32 * 17, 16 * 16 + 256)
                    } else {
                        (0, 0)
                    }
                );
            } else {
                assert!(!live(owner.budget.search_end));
            }
            let values = advice.relaxed_activity(&p, owner.budget.search_end, &owner.budget.state);
            if cube {
                if let Some((heads, activity)) = values {
                    assert_eq!(heads, &(1..17).collect::<Vec<_>>()[..]);
                    assert_eq!(activity.len(), 16);
                    assert!(activity
                        .iter()
                        .all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
                } else {
                    assert!(!live(owner.budget.search_end));
                }
            } else {
                assert!(values.is_none());
            }
        }
        let mut parts = f.original();
        let prefix = parts.problem.clone();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let stats = report.signed.as_ref().unwrap();
        assert_eq!(stats.source_replays, 1);
        assert_eq!(stats.component_rows.phase.collection_attempts, 1);
        let c = &stats.component_rows.phase.cube;
        assert_eq!(c.collection_attempts, usize::from(cube));
        assert_eq!(c.completed_tables, if cube { stats.components } else { 0 });
        assert_eq!(c.appended_rows, c.prepared_rows);
        assert!(c.appended_rows <= c.completed_tables);
        assert_eq!(
            stats.component_rows.phase.appended_rows,
            stats.component_rows.phase.prepared_rows
        );
        assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
        assert_eq!(parts.problem.cols(), prefix.cols());
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
        // Strict table gain belongs to the supplied control, not this route.
        for x in [-1., 0., 1.] {
            for cells in [0, (1usize << 17) - 1] {
                let mut input = vec![q(0.); 30];
                input[29] = q(x);
                let point = f.point(&parts, &input, cells);
                rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
                assert_eq!(
                    &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0],
                    q(0.25)
                );
            }
        }
    }
}

#[test]
fn cube_activity_capture_falls_back_without_refund_and_retains_owned_missing_point_debt() {
    use crate::single_hidden_hull::primal_advice::{self, RoleCounts};
    let (d, h, n) = (16usize, 512usize, 467usize);
    let mut first = vec![0.; d * h];
    for j in 0..h {
        first[j * d + d - 1] = 1.;
    }
    let mut output = vec![0.; 2 * h];
    output[..h - n].fill(1.);
    output[h - n..h].fill(-1.);
    let p = HullMarginProblem::new(
        first,
        vec![0.; h],
        d,
        &output,
        &[0.25, 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    let counts = RoleCounts {
        positive: h - n,
        negative: n,
    };
    let old = primal_advice::quote(&p, counts).unwrap().debt();
    assert_eq!(old.0, 664_608);
    assert!(old.0 <= primal_advice::CAP);
    assert!(old.0 + 32 * (n + 1) > primal_advice::CAP);
    assert!(primal_advice::quote_with_activity(&p, counts).is_none());
    let roles = classify(&p, &preactivation_bounds(&p));
    let mut attempt = primal_capture::Attempt::new_with_activity(&p, end());
    attempt.admit(&roles);
    let report = attempt.snapshot();
    assert!(report.admitted);
    assert_eq!((report.work_units, report.peak_bytes), old);
    attempt.finish(None, &mut || true);
    let mut account = attempt.take_account().unwrap().into_inner().unwrap();
    let ledger = Arc::new(());
    let until = Instant::now() + Duration::from_secs(1);
    account
        .bank_once(&p, until, &ledger, &mut |work, bytes| {
            assert_eq!((work, bytes), old);
            true
        })
        .unwrap();
    assert_eq!(
        account.relaxed_activity_debt(&p, until, &ledger),
        Some((0, 0))
    );
    assert!(account.relaxed_activity(&p, until, &ledger).is_none());

    let f = Fixture::supplied(11. / 16.);
    let p = f.source();
    let foreign = f.source();
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_activity(&p, until);
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    attempt.admit(&roles);
    let before = attempt.snapshot();
    attempt.finish(None, &mut || true);
    let after = attempt.snapshot();
    assert!(!after.decoded);
    assert_eq!(
        (after.work_units, after.peak_bytes),
        (before.work_units, before.peak_bytes)
    );
    let mut account = attempt.take_account().unwrap().into_inner().unwrap();
    let ledger = Arc::new(17u8);
    let bank_end = until
        .checked_sub(Duration::from_millis(1))
        .expect("overflow when subtracting duration from instant");
    assert!(account
        .relaxed_activity_debt(&p, bank_end, &ledger)
        .is_none());
    let mut debits = 0;
    assert!(account
        .bank_once(&foreign, bank_end, &ledger, &mut |_, _| {
            debits += 1;
            true
        })
        .is_none());
    assert_eq!(debits, 0);
    account
        .bank_once(&p, bank_end, &ledger, &mut |work, bytes| {
            debits += 1;
            assert_eq!((work, bytes), (before.work_units, before.peak_bytes));
            true
        })
        .unwrap();
    assert_eq!(debits, 1);
    assert_eq!(
        account.relaxed_activity_debt(&p, bank_end, &ledger),
        Some((96, 288))
    );
    assert!(account.relaxed_activity(&p, bank_end, &ledger).is_none());
    assert!(account
        .relaxed_activity_debt(&foreign, bank_end, &ledger)
        .is_none());
    assert!(account
        .relaxed_activity_debt(&p, bank_end, &Arc::new(17u8))
        .is_none());
    assert!(account
        .relaxed_activity_debt(&p, Instant::now(), &ledger)
        .is_none());
    assert!(account
        .bank_once(&p, bank_end, &ledger, &mut |_, _| {
            debits += 1;
            true
        })
        .is_none());
    assert_eq!(debits, 1);
}

#[test]
fn cube_aggregate_term_bound_covers_adversarial_partitions_and_plane_seals() {
    // Metadata-only enumeration: these are capacity shapes, not Source proofs.
    // Every K partition of three components is considered, including owners
    // whose positives have grown from an empty seed to the full sixteen.
    for d in [1usize, 30] {
        for a in 1usize..=8 {
            for b in 1usize..=8 {
                for c in 1usize..=8 {
                    let negatives = a + b + c;
                    let bound = term_bound(d, 3, negatives).unwrap();
                    for positives in [0usize, 1, 8, MAX_POSITIVE] {
                        let plan = Plan {
                            components: [a, b, c]
                                .into_iter()
                                .map(|k| Component {
                                    negative: (0..k).collect(),
                                    positive: (0..positives).collect(),
                                    positive_units: vec![2; positives],
                                    g: vec![1.; d],
                                    alpha: vec![],
                                })
                                .collect(),
                            leftover: vec![],
                        };
                        let row_terms: usize = plan
                            .components
                            .iter()
                            .map(|item| {
                                item.negative.len()
                                    + item.positive.len()
                                    + item.g.len()
                                    + item.negative.len()
                            })
                            .sum();
                        assert!(row_terms <= bound);
                        assert!(plane_bits(&plan).count() <= 2 * bound);
                        if positives == MAX_POSITIVE {
                            assert_eq!(row_terms, bound);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(term_bound(1, 1, 1), Some(19));
    assert_eq!(term_bound(30, 2, 100), Some(124)); // loose Source census
    assert!(term_bound(0, 1, 1).is_none());
    assert!(term_bound(1, 0, 1).is_none());
    assert!(term_bound(1, 2, 1).is_none());
    assert!(term_bound(usize::MAX, 1, 1).is_none());
    assert!(term_bound(1, usize::MAX, usize::MAX).is_none());
    assert!(term_bound(1, 1, usize::MAX).is_none());
}

fn positive_growth_fixture() -> (Fixture, Plan, Plan) {
    let h = MAX_POSITIVE + 5;
    let mut output = vec![1.; h];
    output[MAX_POSITIVE..].fill(-1.);
    output.resize(2 * h, 0.);
    let fixture = Fixture {
        weights: vec![vec![1.; h], output],
        biases: vec![vec![0.; h], vec![0., 0.]],
        dims: vec![1, h, 2],
        bounds: vec![vec![Bound::new(-1., 1.); h]],
    };
    let component = |negative: Vec<usize>, positive: Vec<usize>| Component {
        positive_units: vec![2; positive.len()],
        alpha: vec![0.; positive.len() * (1usize << negative.len())],
        negative,
        positive,
        g: vec![1.],
    };
    let seed = Plan {
        components: vec![
            component(vec![16, 17, 18], (0..8).collect()),
            component(vec![19], (8..16).collect()),
            component(vec![20], vec![]),
        ],
        leftover: vec![],
    };
    let grown = Plan {
        components: vec![
            component(vec![16, 17, 18], (0..16).collect()),
            component(vec![19], vec![]),
            component(vec![20], vec![]),
        ],
        leftover: vec![],
    };
    (fixture, seed, grown)
}

#[test]
fn cube_tighter_quote_keeps_a_checked_owner_after_positive_growth_to_sixteen() {
    let (fixture, seed, grown) = positive_growth_fixture();
    let p = fixture.source();
    let cuts = base(&p);
    seed.valid(&cuts).unwrap();
    grown.valid(&cuts).unwrap();
    let mut budget = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    budget.phase_all_enabled = true;
    budget.phase_cube_enabled = true;
    let cost = grown.cost(&p).unwrap();
    assert!(cost >= seed.cost(&p).unwrap());
    budget.reserve(cost).unwrap();
    budget.proof(cost, false).unwrap();
    let checked =
        certificate::check(&cuts, &grown, &budget, budget.search_end, &mut || true).unwrap();
    // Admission observes only eight positives per component; the later
    // independently checked owner assigns all sixteen to the larger cell.
    component_rows::admit(&cuts, &seed, &budget).unwrap();
    let quote = budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_ref()
        .unwrap()
        .quote;
    assert_eq!((quote.components, quote.negatives, quote.terms), (3, 5, 61));
    assert!(quote.cell.is_none());
    let seal = Arc::new(signature(&cuts).collect());
    let owner = row(&cuts, &grown, checked, cost, &budget, &seal);
    let replay = owner.replay(&cuts, end(), &mut || true).unwrap();
    let table = replay.cube.unwrap();
    assert!(table.owned(&owner, quote));
    assert!(table.bits().eq(table.integrity.iter().copied()));
    assert!(plane_bits(&grown).eq(table.planes.iter().copied()));
    assert!(table.planes.capacity() <= 2 * quote.terms);
    assert_eq!(table.values.iter().map(Vec::len).sum::<usize>(), 12);
    assert_eq!(owner.stats().source_replays, 1);
}

#[test]
fn cube_tighter_quote_retains_component_positive_and_negative_census_guards() {
    let (fixture, seed, grown) = positive_growth_fixture();
    let p = fixture.source();
    let cuts = base(&p);
    let mut budget = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    budget.phase_all_enabled = true;
    budget.phase_cube_enabled = true;
    let cost = grown.cost(&p).unwrap();
    budget.reserve(cost).unwrap();
    budget.proof(cost, false).unwrap();
    certificate::check(&cuts, &grown, &budget, budget.search_end, &mut || true).unwrap();
    component_rows::admit(&cuts, &seed, &budget).unwrap();
    // Exercise only the metadata gate for malformed layouts. No malformed
    // Plan is given a Source certificate or published as a row.
    budget.state.lock().unwrap().replay_attempts = 1;
    assert!(component_rows::begin(&budget, end()));
    assert!(collecting(&budget, &grown));
    let mut bad = grown.clone();
    bad.components[0].positive.push(0);
    assert!(!collecting(&budget, &bad));
    let mut bad = grown.clone();
    // [3,1,1] -> [2,2,2] keeps twelve masks and reduces visits28->24,
    // but exceeds the admitted five negative appearances. This isolates N
    // from the separate mask/visit caps supporting the aggregate bound.
    bad.components[0].negative.pop();
    bad.components[1].negative.push(18);
    bad.components[2].negative.push(16);
    assert!(!collecting(&budget, &bad));
    let mut bad = grown.clone();
    bad.components.push(grown.components[1].clone());
    assert!(!collecting(&budget, &bad));
    let mut bad = grown.clone();
    bad.components[1].negative.clear();
    assert!(!collecting(&budget, &bad));
    assert!(collecting(&budget, &grown));
}

#[test]
fn cube_adjacent_combined_work_and_payload_thresholds_retain_current_all_mode() {
    for payload in [false, true] {
        for deficit in [0usize, 1] {
            let f = Fixture::supplied(11. / 16.);
            let p = f.source();
            let cuts = base(&p);
            let plan = plan();
            // Obtain the real stateful offered quote in its own checked owner.
            // Its setup debt is paid; it cannot mint another offer afterward.
            let mut measurement = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
            measurement.phase_all_enabled = true;
            measurement.phase_cube_enabled = true;
            let cost = plan.cost(&p).unwrap();
            measurement.reserve(cost).unwrap();
            measurement.proof(cost, false).unwrap();
            certificate::check(
                &cuts,
                &plan,
                &measurement,
                measurement.search_end,
                &mut || true,
            )
            .unwrap();
            let extra = offer(&cuts, &plan, &measurement).unwrap();
            let measured_debt = measurement.state.lock().unwrap().work;
            assert!(offer(&cuts, &plan, &measurement).is_none());
            assert_eq!(measurement.state.lock().unwrap().work, measured_debt);
            assert_eq!((extra.components, extra.negatives, extra.masks), (2, 3, 6));

            let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
            b.phase_all_enabled = true;
            b.phase_cube_enabled = true;
            b.reserve(cost).unwrap();
            b.proof(cost, false).unwrap();
            certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
            let old = component_rows::quote(&cuts, &plan).unwrap();
            let all = all::quote(&cuts, &plan).unwrap();
            let component_setup = (p.hidden_dim + p.input_dim + plan.components.len() + 1) * 32;
            if payload {
                b.admitted_bytes = MAX_BYTES - old.bytes - all.bytes - extra.bytes + deficit;
            } else {
                let mut state = b.state.lock().unwrap();
                state.work =
                    MAX_WORK - state.reserved - component_setup - old.work - all.work - extra.work
                        + deficit;
            }
            let clocks = (b.authority, b.search_end);
            component_rows::admit(&cuts, &plan, &b).unwrap();
            assert!(all_mode(&b));
            let state = b.state.lock().unwrap();
            let report = &state.component_rows.phase;
            assert!(report.admitted && report.all_selector_mode);
            assert!(report.cube.attempted);
            assert_eq!(report.cube.admitted, deficit == 0);
            assert_eq!(report.cube.setup_work, extra.setup);
            assert_eq!(report.cube.work_units, extra.setup);
            assert_eq!(
                report.cube.reserved_work_units,
                if deficit == 0 {
                    extra.work - extra.setup
                } else {
                    0
                }
            );
            assert_eq!(
                report.admitted_bytes,
                all.bytes + if deficit == 0 { extra.bytes } else { 0 }
            );
            assert_eq!(
                report.quoted_rows,
                all.negatives + if deficit == 0 { extra.components } else { 0 }
            );
            assert!(state.work + state.reserved + state.component_reserved <= MAX_WORK);
            assert!(b.admitted_bytes + state.component_rows.admitted_bytes <= MAX_BYTES);
            assert_eq!((b.authority, b.search_end), clocks);
            assert_eq!(report.cube.collection_attempts, 0);
        }
    }
}

#[test]
fn cube_completed_table_tamper_foreign_binding_and_repeat_preserve_old_rows() {
    for fault in 0..4 {
        let f = Fixture::supplied(11. / 16.);
        let p = f.source();
        let cuts = supplied(&p, true);
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let replay = owner
            .replay_inner(&cuts, end(), &mut || true, false)
            .unwrap();
        let minima = replay.component_minima.unwrap();
        let conditional = replay.conditional.unwrap();
        let mut table = replay.cube.unwrap();
        assert_eq!(table.values.iter().map(Vec::len).sum::<usize>(), 6);
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("paid original component rows");
        };
        let prefix = parts.problem.clone();
        let foreign = f.original();
        let debt = owner.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            let lifts = super::super::prepare(
                owner,
                &cuts,
                &parts,
                binding,
                &conditional,
                &minima,
                &old,
                end(),
                &mut || true,
            )
            .unwrap();
            assert_eq!(lifts.len(), 2);
            assert!(prepare(
                owner,
                &cuts,
                &foreign,
                binding,
                &table,
                &minima,
                &old,
                end(),
                &mut || true
            )
            .is_none());
            assert!(
                !owner
                    .budget
                    .state
                    .lock()
                    .unwrap()
                    .phase_cube
                    .prepare_attempted
            );
            // Mutate only a payload produced by the actual full Source replay.
            // No test fabricates a Collected or its authority/integrity seal.
            match fault {
                0 => {
                    table.values[0].pop();
                }
                1 => {
                    table.values[0][3] += 0.5;
                }
                2 => {
                    *table.planes.last_mut().unwrap() ^= 1;
                }
                _ => {
                    table.integrity.pop();
                }
            }
            assert_eq!(
                prepare(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &table,
                    &minima,
                    &old,
                    end(),
                    &mut || true
                )
                .unwrap(),
                [] as [RowSpec; 0]
            );
            assert!(
                owner
                    .budget
                    .state
                    .lock()
                    .unwrap()
                    .phase_cube
                    .prepare_attempted
            );
            let mut polls = 0;
            assert_eq!(
                prepare(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &table,
                    &minima,
                    &old,
                    end(),
                    &mut || {
                        polls += 1;
                        true
                    }
                )
                .unwrap(),
                [] as [RowSpec; 0]
            );
            assert_eq!(polls, 1); // immutable once gate precedes any table walk
        })
        .unwrap();
        assert_eq!(parts.problem, prefix);
        let after = owner.stats();
        assert_eq!(after.work_units, debt);
        assert_eq!(after.source_replays, 1);
        assert_eq!(after.component_rows.phase.prepared_rows, 2);
        assert_eq!(after.component_rows.phase.cube.prepared_rows, 0);
        assert_eq!(after.component_rows.phase.cube.stop_reason, "table-context");
        assert_eq!(after.component_rows.phase.cube.prepare_attempts, 1);
    }
    let f = Fixture::supplied(11. / 16.);
    let p = f.source();
    let first = supplied(&p, true);
    let second = supplied(&p, true);
    let a = first.signed.as_ref().unwrap();
    let b = second.signed.as_ref().unwrap();
    let table = a
        .replay_inner(&first, end(), &mut || true, false)
        .unwrap()
        .cube
        .unwrap();
    let qa = a
        .budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_ref()
        .unwrap()
        .quote;
    let qb = b
        .budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_ref()
        .unwrap()
        .quote;
    assert!(table.owned(a, qa));
    assert!(!table.owned(b, qb)); // same Source still has another ledger/witness
}

#[test]
fn cube_first_last_mask_and_final_publication_cancellation_keep_debt_and_prefix() {
    for cut in [1usize, 6, usize::MAX] {
        let f = Fixture::supplied(11. / 16.);
        let p = f.source();
        let cuts = supplied(&p, true);
        let owner = cuts.signed.as_ref().unwrap();
        let initial = owner.stats();
        let mut parts = f.original();
        let prefix = parts.problem.clone();
        let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
            let state = owner.stats();
            if cut == usize::MAX {
                state.component_rows.phase.cube.prepared_rows == 0
            } else {
                state.checked_masks < initial.checked_masks + cut
            }
        });
        assert!(result.is_none());
        assert_eq!(parts.problem, prefix);
        let after = owner.stats();
        assert_eq!(after.component_rows.phase.cube.collection_attempts, 1);
        assert_eq!(after.component_rows.phase.cube.reserved_work_units, 0);
        assert_eq!(after.component_rows.phase.cube.appended_rows, 0);
        assert_eq!(after.component_rows.phase.cube.prepared_rows, 0);
        assert_eq!(
            after.component_rows.phase.cube.completed_tables,
            if cut == usize::MAX { 2 } else { 0 }
        );
        assert_eq!(after.source_replays, usize::from(cut == usize::MAX));
        assert!(after.component_rows.phase.cube.work_units > 0);
        // A second original hold may publish older rows, never a fresh cube.
        let second = cuts.append_to(&mut parts, &f.encoding(), end());
        if let Some(report) = second {
            assert_eq!(
                report
                    .signed
                    .as_ref()
                    .unwrap()
                    .component_rows
                    .phase
                    .cube
                    .appended_rows,
                0
            );
            let point = f.point(&parts, &[q(0.), q(0.), q(1.)], 0);
            rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
        } else {
            assert_eq!(parts.problem, prefix);
        }
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
        let last = owner.stats();
        assert!(last.work_units >= after.work_units && last.work_units <= MAX_WORK);
        assert_eq!(
            last.component_rows.phase.cube.work_units,
            after.component_rows.phase.cube.work_units
        );
        assert_eq!(last.component_rows.phase.cube.collection_attempts, 1);
        assert_eq!(last.component_rows.phase.cube.appended_rows, 0);
        assert_eq!(last.component_rows.phase.reserved_ir_rows, 0);
        assert!(owner.budget.state.lock().unwrap().replay_attempts <= 2);
        assert!(cuts.append_to(&mut parts, &f.encoding(), end()).is_none());
    }
}

#[test]
fn cube_zero_unary_family_owns_independent_once_header_and_preserves_original_prefix() {
    let f = Fixture::zero_unary();
    let p = f.source();
    let cuts = zero_unary_owner(&p);
    let owner = cuts.signed.as_ref().unwrap();
    assert!(!cuts.mass_portfolio_report().retained);
    let mut parts = f.original();
    let prefix = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let marker = parts.problem.margin_row();
    let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
    let replay = owner
        .replay_inner(&cuts, end(), &mut || true, false)
        .unwrap();
    let minima = replay.component_minima.unwrap();
    let conditional = replay.conditional.unwrap();
    let table = replay.cube.unwrap();
    // Actual outward kernel values, not a manufactured finite table.
    assert!((table.values[0][0] - 1.).abs() < 1e-12);
    assert_eq!(table.values[0][1], table.values[0][2]);
    assert!(table.values[0][1].abs() < 1e-12);
    assert!((table.values[0][3] - 1.).abs() < 1e-12);
    for pair in &conditional.values[0] {
        assert_eq!(pair[0], pair[1]);
    }
    let component_rows::Prepared::Rows(old) =
        component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
            .unwrap()
    else {
        panic!("paid original component rows");
    };
    let cubes = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        let unary = super::super::prepare(
            owner,
            &cuts,
            &parts,
            binding,
            &conditional,
            &minima,
            &old,
            end(),
            &mut || true,
        )
        .unwrap();
        assert_eq!(unary, [] as [RowSpec; 0]);
        assert!(!owner.budget.state.lock().unwrap().phase_all.pending);
        let rows = prepare(
            owner,
            &cuts,
            &parts,
            binding,
            &table,
            &minima,
            &old,
            end(),
            &mut || true,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(owner.budget.state.lock().unwrap().phase_all.pending);
        let selector0 = binding.column(1).unwrap().0;
        let selector1 = binding.column(2).unwrap().0;
        assert!(rows[0]
            .coeffs
            .iter()
            .any(|(j, a)| *j == selector0 && *a > 0.));
        assert!(rows[0]
            .coeffs
            .iter()
            .any(|(j, a)| *j == selector1 && *a > 0.));
        assert!(rows[0].lb > 1.999999999);
        // The real table support exceeds its scalar minimum at selectors0.
        // No additional original-LP gain is asserted for this header control.
        assert!(rows[0].lb > 2. * minima[0] + 1.);
        rows
    })
    .unwrap();
    let additional = cuts.len() + old.len() + cubes.len();
    let bound = owner
        .budget
        .state
        .lock()
        .unwrap()
        .phase_all
        .admission
        .as_ref()
        .unwrap()
        .quote
        .rows;
    let debt = owner.stats().work_units;
    let mut foreign = f.original();
    assert!(owner
        .reserve_portfolio_headers(&mut foreign, additional)
        .is_none());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, bound + 1)
        .is_none());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, additional)
        .is_some());
    assert!(owner
        .reserve_portfolio_headers(&mut parts, additional)
        .is_none());
    assert_eq!(
        owner.stats().component_rows.phase.reserved_ir_rows,
        additional
    );
    assert_eq!(owner.stats().work_units, debt);
    assert_eq!(parts.problem, prefix);
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.margin_row(), marker);

    // Fresh real publication covers the complete no-root/no-loser route.
    let cuts = zero_unary_owner(&p);
    let mut parts = f.original();
    let prefix = parts.problem.clone();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = report.signed.as_ref().unwrap();
    assert_eq!(stats.component_rows.phase.appended_rows, 1);
    assert_eq!(stats.component_rows.phase.cube.appended_rows, 1);
    assert_eq!(stats.component_rows.phase.cube.appended_selector_terms, 2);
    assert_eq!(stats.component_rows.phase.reserved_ir_rows, report.cuts);
    assert_eq!(stats.component_rows.joint.producer_attempts, 0);
    assert_eq!(stats.component_rows.domination.producer_attempts, 0);
    assert!(!cuts.mass_portfolio_report().retained);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    for x in [-1., 0., 1.] {
        for cells in [0, 15] {
            let point = f.point(&parts, &[q(x)], cells);
            rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
            assert_eq!(
                &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0],
                q(1.25 - x.abs())
            );
        }
    }

    // The accepted All coexistence geometry, now with a real retained losing
    // mass recipe AND a winner cube row in the same authenticated header batch.
    // This is a supplied-owner lifecycle control, not default LP discovery.
    let f = Fixture::coexistence();
    let p = f.source();
    let mut cuts = base(&p);
    // x,y,t,z,P0,P1,P2,mu0,t0x,t0z,mu1,t1y,t1z,mu2,t2t,t2z.
    let (account, report) = captured_advice(
        &p,
        cuts.deadline,
        16,
        vec![
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            77. / 256.,
            0.,
            0.,
            77. / 128.,
            0.,
            0.,
            1.,
            0.,
            0.,
        ],
    );
    cuts.primal = Some(account);
    cuts.primal_report = Some(report);
    let mut alpha = Vec::new();
    for mask in 0..8 {
        let pair = [[0., 0.], [0., 0.], [0., 1.], [0.75, 1.]][mask & 3];
        alpha.extend([pair[0], pair[1], ((mask >> 2) & 1) as f64]);
    }
    let winner = Plan {
        components: vec![
            Component {
                negative: vec![3, 4, 5],
                positive: vec![0, 1, 2],
                positive_units: vec![2; 3],
                g: vec![0.; 4],
                alpha,
            },
            Component {
                negative: vec![6],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; 4],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    };
    let old_portfolio_rows = cuts.len() + 1 + 3 * winner.components.len();
    assert_eq!(old_portfolio_rows, 11);
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    b.portfolio_enabled = true;
    let cost = winner.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &winner, &b, b.search_end, &mut || true).unwrap();
    assert!((checked.root_lower - (p.d - 1. - 4. * E)).abs() < 1e-11);
    root_coupling::admit(&cuts, &winner, cost, &mut b).unwrap();
    b.admit_sharing().unwrap();
    component_rows::admit(&cuts, &winner, &b).unwrap();
    assert!(all_mode(&b));
    assert!(b.state.lock().unwrap().component_rows.phase.cube.admitted);
    let combined_rows = b
        .state
        .lock()
        .unwrap()
        .phase_all
        .admission
        .as_ref()
        .unwrap()
        .quote
        .rows;
    assert_eq!(combined_rows, 15);
    let seal = Arc::new(signature(&cuts).collect());
    let mut loser = winner.clone();
    loser.components[0].positive_units[1] = 1;
    loser.components[1].positive = vec![1];
    loser.components[1].positive_units = vec![1];
    loser.components[1].alpha = vec![0., 0.];
    loser.components[1].g[3] = 1.;
    let losing_cost = loser.cost(&p).unwrap();
    b.reserve_at_least(losing_cost).unwrap();
    let mut permit = Some(portfolio::admit(&cuts, &loser, &b, &seal, 1).unwrap());
    portfolio::pay_final(&cuts, &b, losing_cost, permit.as_mut()).unwrap();
    let losing =
        portfolio::check_final(&cuts, &loser, &b, permit, checked.root_lower, &mut || true)
            .unwrap();
    assert!(losing.root_lower < checked.root_lower - 1.);
    assert!(portfolio::report(&b).retained);
    cuts.signed = Some(row(&cuts, &winner, checked, cost, &b, &seal));
    let mut parts = f.original();
    let prefix = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = report.signed.as_ref().unwrap();
    let phases = &stats.component_rows.phase;
    let portfolio = cuts.mass_portfolio_report();
    assert_eq!(phases.appended_rows, 4); // three unary and one cube
    assert_eq!(phases.cube.appended_rows, 1);
    assert_eq!(phases.cube.appended_selector_terms, 3);
    assert_eq!(phases.cube.activity_components, 1);
    assert_eq!(portfolio.appended_rows, 2);
    assert_eq!(portfolio.completed_collections, 1);
    assert_eq!(portfolio.export_attempts, 1);
    assert_eq!(report.cuts, 13);
    assert!(report.cuts > old_portfolio_rows && report.cuts <= combined_rows);
    assert_eq!(phases.reserved_ir_rows, report.cuts);
    assert_eq!(portfolio.reserved_ir_rows, report.cuts);
    assert_eq!(stats.source_replays, 1);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    let point = f.point(&parts, &[q(0.), q(0.), q(0.), q(1.)], 0);
    rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
}

#[test]
fn cube_missing_selector_and_oversized_original_ir_decline_only_optional_rows() {
    let f = Fixture::supplied(11. / 16.);
    let p = f.source();
    for missing in [1usize, 2] {
        let cuts = supplied(&p, true);
        let mut parts = f.original();
        let removed: Vec<_> = parts.binary_vars[2..2 + missing].to_vec();
        assert!(parts.problem.relax_binary_integrality(&removed));
        parts.binary_vars.drain(2..2 + missing);
        parts.binary_widths.drain(2..2 + missing);
        let prefix = parts.problem.clone();
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            for head in 2..2 + missing {
                assert!(binding.column(head).is_none());
            }
        })
        .unwrap();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let stats = report.signed.as_ref().unwrap();
        assert_eq!(stats.component_rows.phase.cube.completed_tables, 2);
        assert_eq!(stats.component_rows.phase.cube.appended_rows, 0);
        assert_eq!(stats.component_rows.phase.appended_rows, 2 - missing);
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
        for selector in &removed {
            assert!(!parts.problem.rows()[prefix.num_rows()..]
                .iter()
                .any(|r| r.coeffs.iter().any(|(j, a)| *j == selector.0 && *a != 0.)));
        }
        let point = f.point(&parts, &[q(0.), q(0.), q(1.)], 0);
        rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
    }
    let cuts = supplied(&p, true);
    let owner = cuts.signed.as_ref().unwrap();
    let original_limit = owner
        .budget
        .state
        .lock()
        .unwrap()
        .phase_all
        .admission
        .as_ref()
        .unwrap()
        .quote
        .original_rows;
    let expanded = Fixture::supplied(11. / 16.).with_unused_outputs(96);
    let mut parts = expanded.original();
    assert!(parts.problem.num_rows() > original_limit);
    let prefix = parts.problem.clone();
    assert!(
        binding::with_phase_binding_test(&cuts, &parts, &expanded.encoding(), |b| b
            .matches(&cuts, &parts))
        .unwrap()
    );
    let report = cuts
        .append_to(&mut parts, &expanded.encoding(), end())
        .unwrap();
    let stats = report.signed.as_ref().unwrap();
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.component_rows.phase.cube.completed_tables, 2);
    assert_eq!(stats.component_rows.phase.appended_rows, 0);
    assert_eq!(stats.component_rows.phase.cube.appended_rows, 0);
    assert_eq!(stats.component_rows.phase.reserved_ir_rows, 0);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
}

#[test]
fn cube_two_bit_lower_facets_beat_wrong_triangle_and_keep_certified_fallback() {
    // Supplied finite-table proposal control, not original-LP discovery.
    // All ten earlier bodies remain unchanged; actual Source/LP/AY coverage
    // is still provided by those production-owner controls.
    fn score(table: &[f64; 4], weights: [f64; 8], z: [f64; 2]) -> Q {
        let encoded = [-2. * weights[0], -2. * weights[1]];
        let mut polls = 0;
        let lower = repair(table, &encoded, end(), &mut || {
            polls += 1;
            true
        })
        .unwrap();
        assert_eq!(polls, 9); // unchanged two complete four-mask scans
        for mask in 0..4 {
            let rhs = q(2. * table[mask])
                + (0..2)
                    .filter(|bit| mask & (1usize << bit) != 0)
                    .map(|bit| q(encoded[bit]))
                    .sum::<Q>();
            assert!(q(lower) <= rhs, "serialized mask{mask}");
        }
        (q(lower) - q(encoded[0]) * q(z[0]) - q(encoded[1]) * q(z[1])) / q(2.)
    }
    let heads = [2, 3];
    let positive = [0., 1., 1., 3.];
    let z = [0.25, 0.25];
    let chain = staircase(&positive, &heads, Some(&z)).unwrap();
    assert_eq!(&chain[..2], &[1., 2.]);
    let old = score(&positive, chain, z);
    assert!(old <= q(-0.25) && old >= q(-0.25) - q(1e-12));
    let exact = support_weights(&positive, &heads, Some(&z)).unwrap();
    assert_eq!(&exact[..2], &[1., 1.]);
    let improved = score(&positive, exact, z);
    // Scalar0 and ALL unary conditional rows max(x,y)=1/4 are weaker.
    assert!(improved > q(0.25));
    assert!(improved >= q(0.5) - q(1e-12) && improved <= q(0.5));
    for (table, z, expected, ideal) in [
        (positive, [0.75, 0.75], [2., 2.], 2.),
        (positive, [0.25, 0.75], [1., 1.], 1.), // exact diagonal chooses00
        ([0., 2., 2., 1.], [0.75, 0.25], [2., -1.], 1.25),
        ([0., 2., 2., 1.], [0.25, 0.75], [-1., 2.], 1.25),
        ([0., 2., 2., 1.], [0.5, 0.5], [2., -1.], 0.5),
    ] {
        let weights = support_weights(&table, &heads, Some(&z)).unwrap();
        assert_eq!(&weights[..2], &expected);
        let value = score(&table, weights, z);
        assert!(value >= q(ideal) - q(1e-12) && value <= q(ideal));
    }
    // Structural missing advice uses the documented00 point; the raw old
    // staircase helper remains available to demonstrate why repair is needed.
    assert_eq!(&two_bit_envelope(&positive, None).unwrap()[..2], &[1., 1.]);
    assert_eq!(
        &support_weights(&positive, &heads, None).unwrap()[..2],
        &[1., 1.]
    );
    assert_eq!(&staircase(&positive, &heads, None).unwrap()[..2], &[1., 2.]);

    // An ordinary rounded sum cannot distinguish these from an exact tie.
    let tiny = f64::from_bits(1);
    assert_eq!(1. + tiny, 1.);
    assert_eq!(0.75 + (0.25f64).next_up(), 1.);
    for z in [[1., tiny], [tiny, 1.], [0.75, (0.25f64).next_up()]] {
        assert_eq!(
            &two_bit_envelope(&positive, Some(&z)).unwrap()[..2],
            &[2., 2.]
        );
    }
    for z in [
        [1., 0.],
        [0., 1.],
        [0.75, 0.25],
        [0.75, (0.25f64).next_down()],
    ] {
        assert_eq!(
            &two_bit_envelope(&positive, Some(&z)).unwrap()[..2],
            &[1., 1.]
        );
    }
    let negative = [0., 2., 2., 1.];
    let first = support_weights(&negative, &heads, Some(&[0.75, 0.25])).unwrap();
    let reversed = support_weights(&negative, &[3, 2], Some(&[0.25, 0.75])).unwrap();
    assert_eq!(&reversed[..2], &[first[1], first[0]]);

    // Uncertain mixed-difference sign is an explicit old-chain fallback.
    let affine = [1., 2., 3., 4.];
    assert!(two_bit_envelope(&affine, Some(&z)).is_none());
    assert_eq!(
        support_weights(&affine, &heads, Some(&z)),
        staircase(&affine, &heads, Some(&z))
    );
    assert!(two_bit_envelope(&positive, Some(&[f64::NAN, 0.5])).is_none());
    assert!(support_weights(&positive, &heads, Some(&[f64::NAN, 0.5])).is_none());
    assert!(two_bit_envelope(&positive, Some(&[0.5])).is_none());
    assert!(two_bit_envelope(&positive[..3], Some(&z)).is_none());
    let overflow = [f64::MAX; 4];
    assert!(two_bit_envelope(&overflow, Some(&z)).is_none());
    assert_eq!(
        support_weights(&overflow, &heads, Some(&z)),
        staircase(&overflow, &heads, Some(&z))
    );
    assert!(repair(&overflow, &[0., 0.], end(), &mut || true).is_none());
    let three = [0., 1., 1., 3., 1., 3., 3., 6.];
    assert_eq!(
        support_weights(&three, &[2, 3, 4], Some(&[0.25; 3])),
        staircase(&three, &[2, 3, 4], Some(&[0.25; 3]))
    );
}
