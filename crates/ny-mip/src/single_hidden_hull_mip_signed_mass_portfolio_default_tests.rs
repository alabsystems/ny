// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Public routing and original-encoder controls. The last two tests supply
//! checked owners explicitly; they are not default strict-loser discovery.
use super::*;
use crate::single_hidden_hull::mip_cuts as root;
use num_rational::BigRational as Q;
use num_traits::Zero;

#[path = "single_hidden_hull_mip_signed_mass_conditional_tests.rs"]
pub(super) mod conditional_tests;
#[path = "single_hidden_hull_mip_signed_mass_cube_tests.rs"]
pub(super) mod cube_tests;

const EPS: f64 = 1. / 256.;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn q(x: f64) -> Q {
    Q::from_float(x).unwrap()
}

pub(super) struct Fixture {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    pub(super) fn with_unused_outputs(mut self, n: usize) -> Self {
        assert!(n >= 2);
        self.w[1].resize(n * self.dims[1], 0.);
        self.b[1].resize(n, 0.);
        self.dims[2] = n;
        self
    }
    fn objective() -> Self {
        let e = 1. / 1024.;
        let mut f = Self::supplied(3. / 16.);
        f.w[0] = vec![1., 0., 0., -1., e, 0., 1., 0., e, -1., 2. * e, 0.];
        f.bounds = vec![vec![
            Bound::new(-1., 1.),
            Bound::new(-(1. + e) as f32, (1. + e) as f32),
            Bound::new(-(1. + e) as f32, (1. + e) as f32),
            Bound::new(-(1. + 2. * e) as f32, (1. + 2. * e) as f32),
        ]];
        f
    }
    fn public(bias: f64) -> Self {
        let (d, h) = (30, 17);
        let mut first = vec![0.; d * h];
        for j in 0..h {
            first[j * d + d - 1] = 1.;
        }
        let mut output = vec![0.; 2 * h];
        output[0] = 16.;
        output[1..h].fill(-1.);
        Self {
            w: vec![first, output],
            b: vec![vec![0.; h], vec![bias, 0.]],
            dims: vec![d, h, 2],
            bounds: vec![vec![Bound::new(-1., 1.); h]],
        }
    }
    pub(super) fn supplied(bias: f64) -> Self {
        Self {
            w: vec![
                vec![1., 0., 0., -1., EPS, 0., 1., 0., EPS, -1., 2. * EPS, 0.],
                vec![
                    1_f64.next_up().next_up(),
                    1_f64.next_up().next_up(),
                    (-1_f64).next_up().next_up(),
                    (-1_f64).next_up().next_up(),
                    0.,
                    0.,
                    0.,
                    0.,
                ],
            ],
            b: vec![vec![0.; 4], vec![bias.next_up().next_up(), 0.]],
            dims: vec![3, 4, 2],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-(1. + EPS) as f32, (1. + EPS) as f32),
                Bound::new(-(1. + EPS) as f32, (1. + EPS) as f32),
                Bound::new(-(1. + 2. * EPS) as f32, (1. + 2. * EPS) as f32),
            ]],
        }
    }
    pub(super) fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w[0].clone(),
            self.b[0].clone(),
            self.dims[0],
            &self.w[1],
            &self.b[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; self.dims[0]],
            vec![1.; self.dims[0]],
        )
        .unwrap()
    }
    pub(super) fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.w,
            &self.b,
            &self.dims,
            &vec![Bound::new(-1., 1.); self.dims[0]],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
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
    fn point(&self, parts: &MipParts, input: &[Q], zero_mask: usize) -> Vec<Q> {
        let original = self.original();
        let heads = binding::head_columns(&self.encoding(), &original).unwrap();
        let (d, h) = (self.dims[0], self.dims[1]);
        assert_eq!(input.len(), d);
        assert_eq!(
            original.binary_vars.len(),
            self.bounds[0]
                .iter()
                .filter(|b| b.lower() < 0. && b.upper() > 0.)
                .count()
        );
        let mut values = vec![Q::zero(); parts.num_cols];
        for i in 0..d {
            values[parts.input_vars[i].0] = input[i].clone();
        }
        for j in 0..h {
            let z = q(self.b[0][j])
                + (0..d)
                    .map(|i| q(self.w[0][j * d + i]) * &input[i])
                    .sum::<Q>();
            values[d + j] = z.clone();
            values[heads[j].0] = z.clone().max(Q::zero());
            if self.bounds[0][j].lower() < 0. && self.bounds[0][j].upper() > 0. {
                let selector = crate::ir::Col(heads[j].0 + 1);
                assert!(original.binary_vars.contains(&selector));
                values[selector.0] = q(if z.is_zero() {
                    ((zero_mask >> j) & 1) as f64
                } else {
                    usize::from(z > Q::zero()) as f64
                });
            }
        }
        for o in 0..2 {
            values[parts.output_vars[o].0] = q(self.b[1][o])
                + (0..h)
                    .map(|j| q(self.w[1][o * h + j]) * &values[heads[j].0])
                    .sum::<Q>();
        }
        values
    }
}
fn rows_hold(parts: &MipParts, values: &[Q], skip: Option<usize>, limit: usize) {
    for (j, c) in parts.problem.cols().iter().enumerate() {
        assert!(!c.lb.is_finite() || values[j] >= q(c.lb), "column{j} lower");
        assert!(!c.ub.is_finite() || values[j] <= q(c.ub), "column{j} upper");
    }
    for (i, row) in parts.problem.rows().iter().enumerate().take(limit) {
        if Some(i) == skip {
            continue;
        }
        let lhs: Q = row.coeffs.iter().map(|(j, a)| q(*a) * &values[*j]).sum();
        assert!(!row.lb.is_finite() || lhs >= q(row.lb), "row{i} lower");
        assert!(!row.ub.is_finite() || lhs <= q(row.ub), "row{i} upper");
    }
}

/// Real automatic strict-loser discovery for the conditional-portfolio successor.
/// No Plan, alpha, captured activity or producer capability is supplied.
#[test]
fn public_mass_strict_loser_preserves_an_exact_unresolved_original_lp_witness() {
    let (d, h) = (30, 18);
    let mut first = vec![0.; d * h];
    let mut biases = vec![0.; h];
    let mut bounds = vec![Bound::new(-1., 1.); h];
    for j in 0..h {
        first[j * d + d - 1] = 1.;
    }
    first[d + d - 1] = -1.;
    biases[1] = -1. / 64.;
    bounds[1] = Bound::new(-65. / 64., 63. / 64.);
    for i in 0..5 {
        let j = 3 + i;
        let m = 2. + i as f64 / 64.;
        first[j * d + d - 1] = m;
        biases[j] = -m - 0.25;
        bounds[j] = Bound::new((-2. * m - 0.25) as f32, -0.25);
    }
    let mut output = vec![0.; 2 * h];
    output[0] = 28.;
    output[1] = 1.;
    output[2] = -97. / 8.;
    output[3..h].fill(-1.);
    let f = Fixture {
        w: vec![first, output],
        b: vec![biases, vec![5. - 1. / 512., 0.]],
        dims: vec![d, h, 2],
        bounds: vec![bounds],
    };
    let p = f.source();
    let attempt = propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(
        &p,
        end(),
        Duration::ZERO,
    );
    assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
    let cuts = attempt.cuts.unwrap();
    let owner = cuts.signed.as_ref().unwrap();
    let before = cuts.mass_portfolio_report();
    eprintln!(
        "automatic conditional fixture: portfolio={before:?} stats={:?}",
        owner.stats()
    );
    assert!(
        before.retained,
        "automatic strict loser was not retained: {before:?}"
    );
    assert!(before.candidate_lower.unwrap() < before.scalar_threshold.unwrap());
    assert!((before.scalar_threshold.unwrap() + 1. / 512.).abs() < 1e-9);
    assert!((before.candidate_lower.unwrap() + 17. / 512.).abs() < 1e-9);
    let mut parts = f.original();
    let original = parts.problem.clone();
    let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
    cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    let mut values = vec![Q::zero(); parts.num_cols];
    for j in 0..h {
        values[d + j] = q(f.b[0][j]);
    }
    values[heads[0].0] = q(1.) / q(28.);
    values[heads[2].0] = q(8.) / q(97.);
    for j in 8..h {
        values[heads[j].0] = q(0.5);
    }
    for var in &parts.binary_vars {
        values[var.0] = q(0.5);
    }
    values[parts.output_vars[0].0] = q(-1. / 512.);
    rows_hold(&parts, &values, None, parts.problem.num_rows());
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(cuts.mass_portfolio_report().appended_rows, 11);
}

fn public_cuts(p: &HullMarginProblem, enabled: bool) -> HullRootMipCuts<'_> {
    // The shipping successor entry constructs its own LP, partition, alpha,
    // directions and owners. There is no supplied Plan or mode mutation here.
    let attempt = if enabled {
        propose_root_mip_cuts_with_mass_portfolio_attempt(p, end(), Duration::ZERO)
    } else {
        propose_root_mip_cuts_with_root_coupling_attempt(p, end(), Duration::ZERO)
    };
    assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
    let cuts = attempt.cuts.expect("actual public root-policy result");
    assert_eq!(cuts.proposal_lp_calls, 1);
    let owner = cuts.signed.as_ref().expect("actual banked signed owner");
    assert_eq!(owner.budget.portfolio_enabled, enabled);
    let stats = owner.stats();
    let root = &stats.root_coupling;
    assert!(
        root.requested && root.admitted && root.selected,
        "root={root:?}"
    );
    assert_eq!((root.parent_count, root.selected_negatives), (6, 6));
    assert_eq!((root.old_masks, root.new_masks), (32, 84));
    assert_eq!(stats.components, 11);
    let r = cuts.mass_portfolio_report();
    assert!(r.collection_attempts <= 1 && r.completed_collections <= r.collection_attempts);
    assert_eq!(
        (r.export_attempts, r.prepared_rows, r.appended_rows),
        (0, 0, 0)
    );
    if !enabled {
        assert_eq!(r, SignedMassPortfolioReport::default());
    } else {
        assert!(r.attempted && r.admitted, "portfolio={r:?}");
        assert_eq!((r.collection_attempts, r.completed_collections), (1, 1));
        assert_eq!(stats.mass.final_checks, 1);
    }
    if r.admitted {
        assert!(r.attempted && r.work_units > 0 && r.admitted_bytes > 0);
    }
    if r.completed_collections == 1 {
        assert!(r.admitted);
        assert_eq!(owner.stats().mass.final_checks, 1);
        assert!(r.candidate_lower.unwrap().is_finite());
        assert!(r.scalar_threshold.unwrap().is_finite());
    }
    if r.retained {
        assert_eq!(r.completed_collections, 1);
        assert!(r.candidate_lower.unwrap() < r.scalar_threshold.unwrap());
    }
    // The already qualified CLI fixture has K6 union topology, so its incoming
    // mass owner is whole. This control requires the new collector to run;
    // a better mass scalar still uses normal owner rows. It does not require
    // or claim automatic discovery of a strictly losing scalar portfolio.
    cuts
}
fn append_public(f: &Fixture, p: &HullMarginProblem) -> MipParts {
    let cuts = public_cuts(p, true);
    let mut parts = f.original();
    let before = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert_eq!(parts.problem.cols(), before.cols());
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.num_rows() - before.num_rows(), report.cuts);
    assert_eq!(report.proposal_lp_calls, 1);
    let signed = report.signed.as_ref().unwrap();
    assert_eq!(signed.source_replays, 1);
    assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    let r = cuts.mass_portfolio_report();
    assert!(r.export_attempts <= 1);
    assert_eq!(r.appended_rows, r.prepared_rows);
    if r.appended_rows > 0 {
        assert!(r.retained && r.export_attempts == 1);
    }
    parts
}

#[test]
fn portfolio_public_successor_routes_actual_lp_and_legacy_entry_stays_disabled() {
    let f = Fixture::public(0.25);
    let p = f.source();
    let legacy = public_cuts(&p, false);
    assert_eq!(
        legacy.mass_portfolio_report(),
        SignedMassPortfolioReport::default()
    );
    let parts = append_public(&f, &p);
    for x in [-1., 0., 1.] {
        for cell in [0, (1usize << 17) - 1] {
            let mut input = vec![Q::zero(); 30];
            input[29] = q(x);
            let point = f.point(&parts, &input, cell);
            rows_hold(&parts, &point, Some(f.encoding().unsafe_row.0), usize::MAX);
            // Exact original graph is +16R(x)-16R(x)+1/4. This proves graph
            // safety, not that this public run closes the root relaxation.
            assert_eq!(
                &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0],
                q(0.25)
            );
        }
    }
}
fn public_witness(bias: f64) {
    let f = Fixture::public(bias);
    let p = f.source();
    let parts = append_public(&f, &p);
    for x in [-1., 0., 1.] {
        for cell in [0, (1usize << 17) - 1] {
            let mut input = vec![Q::zero(); 30];
            input[29] = q(x);
            let point = f.point(&parts, &input, cell);
            rows_hold(&parts, &point, None, usize::MAX);
            let margin = &point[parts.output_vars[0].0] - &point[parts.output_vars[1].0];
            assert_eq!(margin, q(bias));
            assert!(margin <= Q::zero());
        }
    }
}
#[test]
fn portfolio_public_successor_preserves_original_unsafe_witness() {
    public_witness(-0.25);
}
#[test]
fn portfolio_public_successor_preserves_original_closed_zero_cells() {
    public_witness(0.);
}

pub(super) fn supplied_cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let (mut cuts, old, b, checked, signature) = tests::prepared(p);
    let old_lower = checked.root_lower;
    let shared = tests::loser();
    let cost = shared.cost(p).unwrap();
    b.reserve_at_least(cost).unwrap();
    let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
    pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
    let result = check_final(&cuts, &shared, &b, permit, old_lower, &mut || true).unwrap();
    assert!(result.root_lower < old_lower && report(&b).retained);
    component_rows::admit(&cuts, &old, &b).unwrap();
    let owner = row(&cuts, &old, checked, old.cost(p).unwrap(), &b, &signature);
    cuts.signed = Some(owner);
    cuts
}

#[test]
fn portfolio_supplied_loser_maps_strict_component_planes_through_original_encoder_once() {
    let f = Fixture::supplied(EPS);
    let p = f.source();
    assert_eq!(p.d, EPS);
    assert_eq!(p.c, [1., 1., -1., -1.]);
    let cuts = supplied_cuts(&p);
    let old_bits = witness(&cuts.signed.as_ref().unwrap().plan);
    let old_lower = cuts.signed.as_ref().unwrap().lower.to_bits();
    let mut parts = f.original();
    let original = parts.problem.clone();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &parts).unwrap();
    let expected = {
        let c = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
        c.portfolio
            .recipe
            .as_ref()
            .unwrap()
            .planes
            .iter()
            .map(|plane| {
                let mut coeffs: Vec<_> =
                    plane.heads.iter().map(|(j, a)| (heads[*j].0, *a)).collect();
                coeffs.extend(
                    plane
                        .input
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| **a != 0.)
                        .map(|(i, a)| (parts.input_vars[i].0, *a)),
                );
                (plane.lower, coeffs)
            })
            .collect::<Vec<_>>()
    };
    let appended = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    let r = cuts.mass_portfolio_report();
    assert_eq!(
        (
            r.collection_attempts,
            r.completed_collections,
            r.export_attempts
        ),
        (1, 1, 1)
    );
    assert_eq!((r.prepared_rows, r.appended_rows), (2, 2));
    assert_eq!(appended.proposal_lp_calls, 0); // supplied-owner engineering control
    assert_eq!(appended.signed.as_ref().unwrap().source_replays, 1);
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(
        &parts.problem.rows()[..original.num_rows()],
        original.rows()
    );
    assert_eq!(
        parts.problem.num_rows() - original.num_rows(),
        appended.cuts
    );
    let tail = parts.problem.num_rows() - 2;
    for (actual, (lower, coeffs)) in parts.problem.rows()[tail..].iter().zip(expected) {
        assert_eq!(actual.lb.to_bits(), lower.to_bits());
        assert_eq!(actual.ub, f64::INFINITY);
        assert_eq!(actual.coeffs, coeffs);
    }
    assert_eq!(witness(&cuts.signed.as_ref().unwrap().plan), old_bits);
    assert_eq!(cuts.signed.as_ref().unwrap().lower.to_bits(), old_lower);
    for x in [-1., 0., 1.] {
        for y in [-1., 0., 1.] {
            for z in [-1., 0., 1.] {
                for cells in [0, 15] {
                    let point = f.point(&parts, &[q(x), q(y), q(z)], cells);
                    rows_hold(&parts, &point, Some(encoding.unsafe_row.0), usize::MAX);
                }
            }
        }
    }
    // A fractional independent-triangle point satisfies the original graph
    // relaxation and older rows but violates an actually exported new plane.
    // The unsafe property row is excluded: no full-property separation claim.
    let mut fake = f.point(&parts, &[Q::zero(), Q::zero(), Q::zero()], 0);
    for j in [0usize, 2] {
        fake[heads[j].0] = q(0.5);
        fake[f.original().binary_vars[j].0] = q(0.5);
    }
    for o in 0..2 {
        fake[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..4)
                .map(|j| q(f.w[1][o * 4 + j]) * &fake[heads[j].0])
                .sum::<Q>();
    }
    rows_hold(&parts, &fake, Some(encoding.unsafe_row.0), tail);
    assert!(parts.problem.rows()[tail..].iter().any(|row| {
        let lhs: Q = row.coeffs.iter().map(|(j, a)| q(*a) * &fake[*j]).sum();
        lhs < q(row.lb)
    }));
    let fresh = f.original();
    binding::with_phase_binding_test(&cuts, &fresh, &encoding, |binding| {
        assert_eq!(
            (prepare(
                cuts.signed.as_ref().unwrap(),
                &cuts,
                &fresh,
                binding,
                end(),
                &mut || true
            ))
            .len(),
            0
        );
    })
    .unwrap();
    assert_eq!(cuts.mass_portfolio_report().export_attempts, 1);
}

#[test]
fn portfolio_supplied_export_rejects_foreign_binding_and_cancelled_commit_without_renewal() {
    for cancel in [false, true] {
        let f = Fixture::supplied(EPS);
        let p = f.source();
        let cuts = supplied_cuts(&p);
        let mut parts = f.original();
        let foreign = f.original();
        let encoding = f.encoding();
        let before = parts.problem.clone();
        binding::with_phase_binding_test(&cuts, &parts, &encoding, |binding| {
            assert_eq!(
                (prepare(
                    cuts.signed.as_ref().unwrap(),
                    &cuts,
                    &foreign,
                    binding,
                    end(),
                    &mut || true
                ))
                .len(),
                0
            );
        })
        .unwrap();
        assert_eq!(cuts.mass_portfolio_report().export_attempts, 0);
        assert!(cuts
            .signed
            .as_ref()
            .unwrap()
            .budget
            .state
            .lock()
            .unwrap()
            .portfolio
            .recipe
            .is_some());
        assert_eq!(parts.problem, before);
        let result = cuts.append_with_poll(&mut parts, &encoding, end(), &mut || {
            !cancel || cuts.mass_portfolio_report().prepared_rows == 0
        });
        let r = cuts.mass_portfolio_report();
        assert_eq!(r.export_attempts, 1);
        assert_eq!(r.prepared_rows, 2);
        assert_eq!(r.appended_rows, if cancel { 0 } else { 2 });
        assert_eq!(result.is_some(), !cancel);
        if cancel {
            assert_eq!(parts.problem, before);
        }
        let fresh = f.original();
        binding::with_phase_binding_test(&cuts, &fresh, &encoding, |binding| {
            assert_eq!(
                (prepare(
                    cuts.signed.as_ref().unwrap(),
                    &cuts,
                    &fresh,
                    binding,
                    end(),
                    &mut || true
                ))
                .len(),
                0
            );
        })
        .unwrap();
        assert_eq!(cuts.mass_portfolio_report().export_attempts, 1);
        assert!(cuts
            .signed
            .as_ref()
            .unwrap()
            .budget
            .state
            .lock()
            .unwrap()
            .portfolio
            .recipe
            .is_none());
    }
}

fn objective_cuts(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let multipliers = HullMultipliers::zeros(p);
    let lower = certify_hull_lower_bound(p, &multipliers).unwrap();
    let mut cuts = root::prepare(p, multipliers, lower, end(), &mut || true).unwrap();
    let mut old = tests::winner();
    old.components[0].g[0] = 3. / 32.;
    old.components[1].g[0] = -3. / 32.;
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let old_cost = old.cost(p).unwrap();
    b.reserve(old_cost).unwrap();
    b.proof(old_cost, false).unwrap();
    let checked = certificate::check(&cuts, &old, &b, b.search_end, &mut || true).unwrap();
    assert!((checked.root_lower + 2. / 1024.).abs() < 1e-12);
    root_coupling::admit(&cuts, &old, old_cost, &mut b).unwrap();
    b.admit_sharing().unwrap();
    b.portfolio_enabled = true;
    component_rows::admit(&cuts, &old, &b).unwrap();
    let signature = Arc::new(signature(&cuts).collect());
    let mut shared = tests::loser();
    shared.components[0].g[0] = -0.5 + 7. / 64.;
    let cost = shared.cost(p).unwrap();
    b.reserve_at_least(cost).unwrap();
    let mut permit = Some(admit(&cuts, &shared, &b, &signature, 1).unwrap());
    pay_final(&cuts, &b, cost, permit.as_mut()).unwrap();
    let losing = check_final(&cuts, &shared, &b, permit, checked.root_lower, &mut || true).unwrap();
    assert!(losing.root_lower < checked.root_lower && report(&b).retained);
    assert!((losing.root_lower - (3. / 16. - 2. * 7. / 64. - 2.5 / 1024.)).abs() < 1e-12);
    cuts.signed = Some(row(&cuts, &old, checked, old_cost, &b, &signature));
    cuts
}

fn lp_floor(parts: &MipParts, property: usize, limit: usize) -> f64 {
    let mut relaxed = crate::ir::MilpProblem::new();
    for col in parts.problem.cols() {
        relaxed.add_col(col.obj, col.lb, col.ub);
    }
    for (i, row) in parts.problem.rows().iter().enumerate().take(limit) {
        if i != property {
            relaxed.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|(j, a)| (crate::ir::Col(*j), *a)),
            );
        }
    }
    let model = crate::ay_lib::to_ay_model(&relaxed).unwrap();
    let options = SolveOpts::new()
        .with_deadline(end())
        .with_time_limit(Duration::from_secs(30));
    let mut session = ay_milp::LpSession::new(&model, &options).unwrap();
    let lower = session
        .rigorous_bound_expr(
            &[
                (model.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (model.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ],
            ay_milp::Sense::Minimize,
        )
        .unwrap();
    crate::lp_propose_certify::rigorous_lower_f64(&lower).unwrap()
}
fn weighted_row(row: &RowSpec, weight: Q, coefficients: &mut [Q], rhs: &mut Q) {
    let endpoint = if weight >= Q::zero() { row.lb } else { row.ub };
    assert!(endpoint.is_finite());
    *rhs += &weight * q(endpoint);
    for (j, a) in &row.coeffs {
        coefficients[*j] += &weight * q(*a);
    }
}

#[test]
fn portfolio_supplied_losing_scalar_strictly_closes_original_objective_lp_and_ay() {
    // Supplied legal owners: this is an objective-separation control, not
    // automatic discovery. Every consumed minimum comes from the Source check.
    let f = Fixture::objective();
    let p = f.source();
    assert_eq!(p.c, [1., 1., -1., -1.]);
    assert_eq!(p.d, 3. / 16.);
    let cuts = objective_cuts(&p);
    let original = f.original();
    let encoding = f.encoding();
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let mut parts = f.original();
    let appended = cuts.append_to(&mut parts, &encoding, end()).unwrap();
    let stats = cuts.mass_portfolio_report();
    assert_eq!(
        (
            stats.completed_collections,
            stats.export_attempts,
            stats.appended_rows
        ),
        (1, 1, 2)
    );
    assert!(stats.candidate_lower.unwrap() < stats.scalar_threshold.unwrap());
    assert_eq!(appended.proposal_lp_calls, 0);
    assert_eq!(appended.signed.as_ref().unwrap().source_replays, 1);
    assert_eq!(
        appended
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .phase
            .appended_rows,
        2
    );
    let tail = parts.problem.num_rows() - 2;
    // Predeclared robust closed-unsafe witness for the ACTUAL nudged original
    // coefficients: N0=N1=E+e/8, P0=P1=0, input0, all selectors1/2.
    // Its ideal margin is -e/4; the few original coefficient ulps leave slack.
    let mut fake = f.point(&parts, &[Q::zero(), Q::zero(), Q::zero()], 0);
    for j in 0..4 {
        fake[original.binary_vars[j].0] = q(0.5);
    }
    for j in [2usize, 3] {
        fake[heads[j].0] = q(3. / 32. + 1. / 8192.);
    }
    for o in 0..2 {
        fake[parts.output_vars[o].0] = q(f.b[1][o])
            + (0..4)
                .map(|j| q(f.w[1][o * 4 + j]) * &fake[heads[j].0])
                .sum::<Q>();
    }
    rows_hold(&parts, &fake, None, tail);
    assert!(&fake[parts.output_vars[0].0] - &fake[parts.output_vars[1].0] < Q::zero());

    // Rational certificate using ACTUAL exported coefficients and minima.
    // (13/20)*I+(7/20)*II cancels the nominal input direction; all actual
    // coefficient residuals, including original-output ulps, are box-priced.
    let rows = parts.problem.rows();
    let winner_a = rows[original.problem.num_rows()..tail]
        .iter()
        .find(|r| {
            r.coeffs.len() == 3
                && r.coeffs.contains(&(heads[0].0, 2.))
                && r.coeffs.contains(&(heads[2].0, -2.))
                && r.coeffs.contains(&(parts.input_vars[0].0, -3. / 16.))
        })
        .expect("actual doubled winning A row");
    let mut combined = vec![Q::zero(); parts.num_cols];
    let mut rhs = Q::zero();
    weighted_row(
        &rows[tail],
        Q::new(13.into(), 40.into()),
        &mut combined,
        &mut rhs,
    );
    weighted_row(
        &rows[tail + 1],
        Q::new(1.into(), 2.into()),
        &mut combined,
        &mut rhs,
    );
    weighted_row(
        winner_a,
        Q::new(7.into(), 80.into()),
        &mut combined,
        &mut rhs,
    );
    let n0_upper: Vec<_> = original
        .problem
        .rows()
        .iter()
        .filter(|r| {
            r.lb == f64::NEG_INFINITY && r.ub.is_finite() && r.coeffs.contains(&(heads[2].0, 1.))
        })
        .collect();
    assert_eq!(n0_upper.len(), 2);
    for row in n0_upper {
        weighted_row(row, Q::new((-7).into(), 80.into()), &mut combined, &mut rhs);
    }
    let p1_lower = original
        .problem
        .rows()
        .iter()
        .find(|r| {
            r.lb == 0.
                && r.ub == f64::INFINITY
                && r.coeffs.len() == 2
                && r.coeffs.contains(&(heads[1].0, 1.))
                && r.coeffs.contains(&(4usize, -1.))
        })
        .unwrap();
    weighted_row(
        p1_lower,
        Q::new(7.into(), 40.into()),
        &mut combined,
        &mut rhs,
    );
    for j in [1usize, 2] {
        let affine = &original.problem.rows()[j];
        assert_eq!(affine.lb, affine.ub);
        let column = 3 + j;
        let a = affine.coeffs.iter().find(|(i, _)| *i == column).unwrap().1;
        let weight = -combined[column].clone() / q(a);
        weighted_row(affine, weight, &mut combined, &mut rhs);
        assert!(combined[column].is_zero());
    }
    let mut objective = vec![Q::zero(); parts.num_cols];
    for j in 0..4 {
        objective[heads[j].0] = q(f.w[1][j]) - q(f.w[1][4 + j]);
    }
    let mut rigorous = q(f.b[1][0]) - q(f.b[1][1]) + rhs;
    for (j, (want, got)) in objective.iter().zip(&combined).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let col = &parts.problem.cols()[j];
        assert!(
            col.lb.is_finite() && col.ub.is_finite(),
            "unpriced column{j}"
        );
        rigorous += (&residual * q(col.lb)).min(&residual * q(col.ub));
    }
    assert!(rigorous >= Q::new(81.into(), 8192.into()) - q(1e-12));
    assert!(rigorous > Q::zero());
    let before = lp_floor(&parts, encoding.unsafe_row.0, tail);
    let after = lp_floor(&parts, encoding.unsafe_row.0, usize::MAX);
    assert!(before <= 0., "old original LP={before}");
    assert!(after >= 81. / 8192. - 1e-9, "augmented original LP={after}");
    // The raw original graph is safe by two ReLU Lipschitz gaps. This input
    // attains the ideal Source minimum; check the actual original rows too.
    let graph = f.point(&parts, &[q(0.), q(1.), q(1.)], 0);
    rows_hold(&parts, &graph, Some(encoding.unsafe_row.0), usize::MAX);
    assert!(&graph[parts.output_vars[0].0] - &graph[parts.output_vars[1].0] > Q::zero());
    let result = crate::MipSolver::new(
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
    .unwrap();
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}
