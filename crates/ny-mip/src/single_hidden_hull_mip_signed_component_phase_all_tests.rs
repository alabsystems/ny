// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! All-selector publication controls. Supplied-owner separation is not default discovery.
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
    fn with_unused_outputs(mut self, n: usize) -> Self {
        assert!(n >= 2);
        self.weights[1].resize(n * self.dims[1], 0.);
        self.biases[1].resize(n, 0.);
        self.dims[2] = n;
        self
    }
    fn coexistence() -> Self {
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0.,
                    0., // P0=x
                    0.,
                    1.,
                    0.,
                    0., // P1=y
                    0.,
                    0.,
                    1.,
                    0., // P2=t
                    1.,
                    0.,
                    0.,
                    E, // N0=x+e*z
                    0.,
                    1.,
                    0.,
                    2. * E, // N1=y+2e*z
                    0.,
                    0.,
                    1.,
                    3. * E, // N2=t+3e*z
                    0.,
                    0.,
                    0.,
                    0., // stable-dead spectator
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
    fn supplied(bias: f64) -> Self {
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    1.,
                    0.,
                    E,
                    0.,
                    1.,
                    2. * E,
                    0.,
                    0.,
                    0.,
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
        let unstable = self.bounds[0]
            .iter()
            .filter(|b| b.lower() < 0. && b.upper() > 0.)
            .count();
        assert_eq!(original.binary_vars.len(), unstable);
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
fn supplied(p: &HullMarginProblem, all: bool) -> HullRootMipCuts<'_> {
    supplied_in_context(p, all, true)
}
fn supplied_in_context(p: &HullMarginProblem, all: bool, funded_root: bool) -> HullRootMipCuts<'_> {
    let mut cuts = base(p);
    let plan = plan();
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    b.phase_all_enabled = all;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    assert!((checked.root_lower - (p.d - 1. - E)).abs() < 1e-11);
    // The same checked C2 Plan pays a real exclusive root Ticket in BOTH
    // comparison modes. The stable-dead spectator adds no graph contribution.
    // Older optional joint/domination families therefore have the same actual
    // root-policy scope, rather than being silently omitted from the old LP.
    if funded_root {
        root_coupling::admit(&cuts, &plan, cost, &mut b).unwrap();
        assert!(b.state.lock().unwrap().root_coupling.admitted);
    }
    let seal = Arc::new(signature(&cuts).collect());
    component_rows::admit(&cuts, &plan, &b).unwrap();
    assert!(b.state.lock().unwrap().component_rows.phase.admitted);
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
fn appended(f: &Fixture, all: bool) -> (MipParts, HullMipCutReport) {
    let p = f.source();
    let cuts = supplied(&p, all);
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
        if all { 2 } else { 1 }
    );
    assert_eq!(report.proposal_lp_calls, 0); // supplied-owner engineering control
    assert_eq!(parts.problem.cols(), old.cols());
    assert_eq!(&parts.problem.rows()[..old.num_rows()], old.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.num_rows() - old.num_rows(), report.cuts);
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    assert!(!cuts.mass_portfolio_report().retained); // independent all-row header authority
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
fn all_phase_supplied_smaller_gap_strictly_closes_original_objective_lp_and_ay() {
    let f = Fixture::supplied(23. / 32.);
    let original = f.original();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let (old, _) = appended(&f, false);
    let (all, _) = appended(&f, true);
    // Both compared models retain their complete original and old-family rows.
    let mut fake = f.point(&old, &[q(0.), q(0.), q(0.)], 0);
    for j in 0..4 {
        fake[original.binary_vars[j].0] = q(if j == 2 { 0.75 } else { 0.5 });
    }
    fake[heads[2].0] = q(0.25);
    fake[heads[3].0] = q(0.5);
    f.outputs(&old, &mut fake);
    rows_hold(&old, &fake, None);
    assert_eq!(
        &fake[old.output_vars[0].0] - &fake[old.output_vars[1].0],
        q(-1. / 32.)
    );

    let phase_row = |head: usize| {
        let selector = original.binary_vars[head].0;
        let rows: Vec<_> = all.problem.rows()[original.problem.num_rows()..]
            .iter()
            .filter(|r| {
                r.lb.is_finite() && r.coeffs.iter().any(|(j, a)| *j == selector && *a != 0.)
            })
            .collect();
        assert_eq!(rows.len(), 1);
        rows[0]
    };
    let smaller = phase_row(3);
    let lhs: Q = smaller.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
    assert!(lhs < q(smaller.lb));
    let mut combined = vec![Q::zero(); all.num_cols];
    let mut rhs = Q::zero();
    // The actual published phase rows are doubled. Weights3/20 and1/5
    // implement3/10 A+4/10 B; the original-row group below has weight3/10.
    weighted(
        phase_row(2),
        Q::new(3.into(), 20.into()),
        &mut combined,
        &mut rhs,
    );
    weighted(smaller, Q::new(1.into(), 5.into()), &mut combined, &mut rhs);
    let n0_upper = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.ub == 1. + E
                && r.lb == f64::NEG_INFINITY
                && r.coeffs.contains(&(heads[2].0, 1.))
                && r.coeffs.contains(&(5, -1.))
                && r.coeffs.contains(&(original.binary_vars[2].0, 1. + E))
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
                && r.coeffs
                    .contains(&(original.binary_vars[3].0, -(1. + 2. * E)))
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
    for row in [n0_upper, n1_upper] {
        weighted(row, Q::new((-3).into(), 10.into()), &mut combined, &mut rhs);
    }
    weighted(
        p0_lower,
        Q::new(3.into(), 10.into()),
        &mut combined,
        &mut rhs,
    );
    for j in [0usize, 2] {
        let affine = &original.problem.rows()[j];
        assert_eq!(affine.lb, affine.ub);
        let column = 3 + j;
        let a = affine.coeffs.iter().find(|(i, _)| *i == column).unwrap().1;
        let weight = -combined[column].clone() / q(a);
        weighted(affine, weight, &mut combined, &mut rhs);
        assert!(combined[column].is_zero());
    }
    let mut objective = vec![Q::zero(); all.num_cols];
    for j in 0..f.dims[1] {
        objective[heads[j].0] = q(f.weights[1][j]) - q(f.weights[1][f.dims[1] + j]);
    }
    let mut lower = q(f.biases[1][0]) - q(f.biases[1][1]) + rhs;
    for (j, (want, got)) in objective.iter().zip(&combined).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let c = &all.problem.cols()[j];
        assert!(c.lb.is_finite() && c.ub.is_finite(), "unpriced column{j}");
        lower += (&residual * q(c.lb)).min(&residual * q(c.ub));
    }
    assert!(lower >= Q::new(81.into(), 5120.into()) - q(1e-11));
    assert!(lower > Q::new(33.into(), 2560.into()));
    let before = lp_floor(&old, encoding.unsafe_row.0);
    let after = lp_floor(&all, encoding.unsafe_row.0);
    assert!(before <= -1. / 32. + 1e-10, "old LP={before}");
    assert!(after >= 81. / 5120. - 1e-9, "all-row LP={after}");
    let result = solve(all);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn all_phase_original_unsafe_and_exact_closed_zero_cells_are_retained() {
    for bias in [2. * E, 3. * E] {
        let f = Fixture::supplied(bias);
        let (parts, _) = appended(&f, true);
        // Original raw coefficients are exact integers, without output nudges.
        // P0 and P1 are zero; enumerate all four independent closed choices.
        for cells in 0..4 {
            let point = f.point(&parts, &[q(0.), q(0.), q(1.)], cells);
            rows_hold(&parts, &point, None);
            assert_eq!(
                &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0],
                q(bias - 3. * E)
            );
        }
        // Here all four heads are zero: exercise both values of both lifted
        // negative selectors too. This graph cell is outside the property.
        for cells in 0..16 {
            let point = f.point(&parts, &[q(0.), q(0.), q(0.)], cells);
            rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0));
        }
        let result = solve(parts);
        assert!(matches!(result, crate::MipResult::Sat { .. }), "{result:?}");
    }
}

#[test]
fn all_phase_public_successor_uses_actual_lp_and_legacy_portfolio_stays_one() {
    let f = Fixture::public();
    let p = f.source();
    for all in [false, true] {
        // This branch constructs its own LP, component masks, alpha and owners.
        // It does not supply the separation Plan or alter a returned mode.
        let attempt = if all {
            propose_root_mip_cuts_with_mass_portfolio_all_phases_attempt(&p, end(), Duration::ZERO)
        } else {
            propose_root_mip_cuts_with_mass_portfolio_attempt(&p, end(), Duration::ZERO)
        };
        assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
        let cuts = attempt.cuts.expect("actual public policy owner");
        let owner = cuts.signed.as_ref().unwrap();
        assert!(owner.budget.portfolio_enabled);
        assert_eq!(owner.budget.phase_all_enabled, all);
        assert_eq!(all_mode(&owner.budget), all);
        assert_eq!(cuts.proposal_lp_calls, 1);
        let stats = owner.stats();
        assert!(stats.root_coupling.requested && stats.root_coupling.admitted);
        assert!(stats.component_rows.phase.admitted);
        let mut parts = f.original();
        let prefix = parts.problem.clone();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let stats = report.signed.as_ref().unwrap();
        assert_eq!(stats.source_replays, 1);
        assert_eq!(stats.component_rows.phase.collection_attempts, 1);
        // Discovery must complete the real collector. Positive gaps/eligible
        // selectors belong to the discovered owner; the supplied control above
        // proves the strict value of the extra rows independently.
        assert_eq!(
            stats.component_rows.phase.collected_components,
            stats.components
        );
        assert_eq!(
            stats.component_rows.phase.appended_rows,
            stats.component_rows.phase.prepared_rows
        );
        assert!(
            stats.component_rows.phase.appended_rows <= if all { 16 } else { stats.components }
        );
        assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
        assert_eq!(parts.problem.cols(), prefix.cols());
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
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
fn all_phase_missing_largest_selector_still_exports_the_smaller_eligible_pair() {
    let f = Fixture::supplied(23. / 32.);
    let p = f.source();
    for all in [false, true] {
        let cuts = supplied(&p, all);
        let mut parts = f.original();
        let biggest = parts.binary_vars[2];
        let smaller = parts.binary_vars[3];
        assert!(parts.problem.relax_binary_integrality(&[biggest]));
        parts.binary_vars.remove(2);
        parts.binary_widths.remove(2);
        let prefix = parts.problem.clone();
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            assert!(binding.column(2).is_none());
            assert_eq!(binding.column(3), Some(smaller));
        })
        .unwrap();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        assert_eq!(
            report
                .signed
                .as_ref()
                .unwrap()
                .component_rows
                .phase
                .appended_rows,
            1
        );
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
        let phase_rows: Vec<_> = parts.problem.rows()[prefix.num_rows()..]
            .iter()
            .filter(|r| r.coeffs.iter().any(|(j, a)| *j == smaller.0 && *a != 0.))
            .collect();
        assert_eq!(phase_rows.len(), 1);
        assert!(!parts.problem.rows()[prefix.num_rows()..]
            .iter()
            .any(|r| r.coeffs.iter().any(|(j, a)| *j == biggest.0 && *a != 0.)));
    }
}

#[test]
fn all_phase_corrupt_incomplete_and_foreign_conditionals_preserve_old_rows() {
    for missing in [false, true] {
        let f = Fixture::supplied(23. / 32.);
        let p = f.source();
        let cuts = supplied(&p, true);
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let replay = owner
            .replay_inner(&cuts, end(), &mut || true, false)
            .unwrap();
        let minima = replay.component_minima.unwrap();
        let mut conditional = replay.conditional.unwrap();
        if missing {
            conditional.values[0].pop();
        } else {
            conditional.values[0][1][1] += 0.5;
        }
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("paid original component row");
        };
        assert_eq!(old.len(), 2);
        let prefix = parts.problem.clone();
        let foreign = f.original();
        let debt = owner.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            assert!(prepare(
                owner,
                &cuts,
                &foreign,
                binding,
                &conditional,
                &minima,
                &old,
                end(),
                &mut || true
            )
            .is_none());
            let rows = prepare(
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
            assert_eq!(rows, [] as [RowSpec; 0]);
            assert!(
                owner
                    .budget
                    .state
                    .lock()
                    .unwrap()
                    .phase_all
                    .prepare_attempted
            );
            assert_eq!(
                prepare(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &conditional,
                    &minima,
                    &old,
                    end(),
                    &mut || true
                )
                .unwrap(),
                [] as [RowSpec; 0]
            );
        })
        .unwrap();
        assert_eq!(parts.problem, prefix);
        assert_eq!(owner.stats().work_units, debt);
        assert_eq!(
            owner.stats().component_rows.phase.stop_reason,
            "all-coverage"
        );
        assert_eq!(owner.stats().component_rows.phase.appended_rows, 0);
        assert_eq!(owner.stats().source_replays, 1);
    }
    let f = Fixture::supplied(23. / 32.);
    let p = f.source();
    let c1 = supplied(&p, true);
    let c2 = supplied(&p, true);
    let a = c1.signed.as_ref().unwrap();
    let b = c2.signed.as_ref().unwrap();
    let replay = a.replay_inner(&c1, end(), &mut || true, false).unwrap();
    let minima = replay.component_minima.unwrap();
    let conditional = replay.conditional.unwrap();
    assert!(conditional.valid(a, &minima));
    assert!(!conditional.valid(b, &minima));
}

#[test]
fn all_phase_partial_mask_and_final_publication_cancellation_keep_debt_and_prefix() {
    for final_prepare in [false, true] {
        let f = Fixture::supplied(23. / 32.);
        let p = f.source();
        let cuts = supplied(&p, true);
        let owner = cuts.signed.as_ref().unwrap();
        let masks = owner.stats().checked_masks;
        let held = owner.budget.state.lock().unwrap().component_reserved;
        assert!(held > 0);
        let mut parts = f.original();
        let prefix = parts.problem.clone();
        let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
            let stats = owner.stats();
            if final_prepare {
                stats.component_rows.phase.prepared_rows < 2
            } else {
                stats.checked_masks < masks + 3
            }
        });
        assert!(result.is_none());
        assert_eq!(parts.problem, prefix);
        let after = owner.stats();
        assert_eq!(after.component_rows.phase.appended_rows, 0);
        assert_eq!(after.component_rows.phase.collection_attempts, 1);
        assert_eq!(after.component_rows.phase.reserved_work_units, 0);
        assert_eq!(after.source_replays, usize::from(final_prepare));
        assert!(after.component_rows.phase.work_units > 0);
        // The original second Source hold may still publish older global rows.
        // It cannot restart the one paid All collection or its header lease.
        let second = cuts.append_to(&mut parts, &f.encoding(), end());
        if let Some(report) = second {
            assert_eq!(
                report
                    .signed
                    .as_ref()
                    .unwrap()
                    .component_rows
                    .phase
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
            last.component_rows.phase.work_units,
            after.component_rows.phase.work_units
        );
        assert_eq!(last.component_rows.phase.collection_attempts, 1);
        assert_eq!(last.component_rows.phase.appended_rows, 0);
        assert_eq!(last.component_rows.phase.reserved_ir_rows, 0);
        assert!(owner.budget.state.lock().unwrap().replay_attempts <= 2);
        assert!(cuts.append_to(&mut parts, &f.encoding(), end()).is_none());
        assert_eq!(
            owner.stats().component_rows.phase.work_units,
            after.component_rows.phase.work_units
        );
    }
}

#[test]
fn all_phase_adjacent_work_and_payload_thresholds_fall_back_to_exact_one_mode() {
    for payload in [false, true] {
        for deficit in [0usize, 1] {
            let f = Fixture::supplied(23. / 32.);
            let p = f.source();
            let cuts = base(&p);
            let plan = plan();
            let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
            b.phase_all_enabled = true;
            let cost = plan.cost(&p).unwrap();
            b.reserve(cost).unwrap();
            b.proof(cost, false).unwrap();
            assert!(certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).is_some());
            let old = component_rows::quote(&cuts, &plan).unwrap();
            let one = quote(&cuts, &plan).unwrap();
            let all = all::quote(&cuts, &plan).unwrap();
            assert!(all.work > one.work && all.bytes > one.bytes);
            assert_eq!((all.components, all.negatives), (2, 3));
            let component_setup = (p.hidden_dim + p.input_dim + plan.components.len() + 1) * 32;
            if payload {
                b.admitted_bytes = MAX_BYTES - old.bytes - all.bytes + deficit;
            } else {
                let mut state = b.state.lock().unwrap();
                state.work =
                    MAX_WORK - state.reserved - component_setup - old.work - all.work + deficit;
            }
            let authority = b.authority;
            let search_end = b.search_end;
            component_rows::admit(&cuts, &plan, &b).unwrap();
            assert_eq!(all_mode(&b), deficit == 0);
            let state = b.state.lock().unwrap();
            let report = &state.component_rows.phase;
            assert!(state.component_rows.admitted && report.admitted);
            assert_eq!(report.all_selector_mode, deficit == 0);
            assert_eq!(report.quoted_rows, if deficit == 0 { 3 } else { 2 });
            assert_eq!(
                report.quoted_terms,
                if deficit == 0 { all.terms } else { one.terms }
            );
            assert_eq!(
                report.admitted_bytes,
                if deficit == 0 { all.bytes } else { one.bytes }
            );
            assert_eq!(
                report.work_units + report.reserved_work_units,
                if deficit == 0 { all.work } else { one.work }
            );
            assert!(state.work + state.reserved + state.component_reserved <= MAX_WORK);
            assert!(b.admitted_bytes + state.component_rows.admitted_bytes <= MAX_BYTES);
            assert_eq!(b.authority, authority);
            assert_eq!(b.search_end, search_end);
            assert_eq!(report.collection_attempts, 0);
        }
    }
}

#[test]
fn all_phase_independent_header_capability_is_bound_once_and_checks_original_row_limit() {
    let f = Fixture::supplied(23. / 32.);
    let p = f.source();
    // Actual publication without a root Ticket or losing recipe still owns an
    // exclusive All row envelope: optional joint/domination must not run.
    {
        let cuts = supplied_in_context(&p, true, false);
        assert!(cuts.signed.as_ref().unwrap().budget.root_coupling.is_none());
        assert!(!cuts.mass_portfolio_report().retained);
        let mut parts = f.original();
        let prefix = parts.problem.clone();
        let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
        let stats = report.signed.as_ref().unwrap();
        assert_eq!(stats.component_rows.phase.appended_rows, 2);
        assert!(!stats.component_rows.joint.attempted);
        assert_eq!(stats.component_rows.joint.producer_attempts, 0);
        assert!(!stats.component_rows.domination.attempted);
        assert_eq!(stats.component_rows.domination.producer_attempts, 0);
        assert_eq!(stats.component_rows.phase.reserved_ir_rows, report.cuts);
        assert_eq!(parts.problem.cols(), prefix.cols());
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    }
    let cuts = supplied_in_context(&p, true, false);
    let owner = cuts.signed.as_ref().unwrap();
    assert!(owner.budget.root_coupling.is_none());
    let mut parts = f.original();
    let original = parts.problem.clone();
    let binary = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let marker = parts.problem.margin_row();
    let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
    let replay = owner
        .replay_inner(&cuts, end(), &mut || true, false)
        .unwrap();
    let minima = replay.component_minima.unwrap();
    let conditional = replay.conditional.unwrap();
    let component_rows::Prepared::Rows(old) =
        component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
            .unwrap()
    else {
        panic!("paid component row");
    };
    let lifts = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
        prepare(
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
    })
    .unwrap()
    .unwrap();
    assert_eq!(lifts.len(), 2);
    assert!(!cuts.mass_portfolio_report().retained);
    let additional = cuts.len() + old.len() + lifts.len();
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
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(parts.problem.rows(), original.rows());
    assert_eq!(parts.problem.margin_row(), marker);
    assert_eq!(parts.binary_vars, binary);
    assert_eq!(parts.binary_widths, widths);

    // Bind real original encoders with exactly I and I+1 rows. Extra zero
    // output equations do not alter the selected two-output Source margin.
    let limit = 8 * (p.hidden_dim + p.input_dim + 1);
    let old_rows = f.original().problem.num_rows();
    assert!(limit > old_rows);
    for excess in [0usize, 1] {
        let f = Fixture::supplied(23. / 32.).with_unused_outputs(2 + limit - old_rows + excess);
        let cuts = supplied_in_context(&p, true, false);
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        assert_eq!(parts.problem.num_rows(), limit + excess);
        let before = parts.problem.clone();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let replay = owner
            .replay_inner(&cuts, end(), &mut || true, false)
            .unwrap();
        let minima = replay.component_minima.unwrap();
        let conditional = replay.conditional.unwrap();
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("paid original component row");
        };
        let lifts = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            prepare(
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
        })
        .unwrap()
        .unwrap();
        assert_eq!(old.len(), 2);
        assert_eq!(lifts.len(), if excess == 0 { 2 } else { 0 });
        if excess == 1 {
            assert_eq!(
                owner.stats().component_rows.phase.stop_reason,
                "all-original-ir-limit"
            );
            assert!(!owner.budget.state.lock().unwrap().phase_all.pending);
        }
        assert_eq!(parts.problem, before);
    }
}

#[test]
fn all_phase_retained_loser_coexists_above_old_portfolio_aggregate_header_bound() {
    let f = Fixture::coexistence();
    let p = f.source();
    let mut cuts = base(&p);
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
    let all_rows = all::quote(&cuts, &winner).unwrap().rows;
    assert_eq!(all_rows, 13);
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
    let original = parts.problem.clone();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let stats = report.signed.as_ref().unwrap();
    let phases = &stats.component_rows.phase;
    let portfolio = cuts.mass_portfolio_report();
    assert_eq!(phases.appended_rows, 3);
    assert_eq!(portfolio.appended_rows, 2);
    assert_eq!(portfolio.completed_collections, 1);
    assert_eq!(portfolio.export_attempts, 1);
    assert_eq!(report.cuts, 12);
    assert!(report.cuts > old_portfolio_rows && report.cuts <= all_rows);
    assert_eq!(phases.reserved_ir_rows, report.cuts);
    assert_eq!(portfolio.reserved_ir_rows, report.cuts);
    assert_eq!(stats.source_replays, 1);
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
}
