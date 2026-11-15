// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Production admission/publication controls. Source and encoder are authentic.
//! The separator overrides paid advisory target/activity only; it does not
//! claim that the original LP returned that particular fractional point.
use super::*;

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn new() -> Self {
        Self {
            weights: vec![vec![1.; 3], vec![1., -1., -1., 0., 0., 0.]],
            biases: vec![vec![0.25, -0.25, -0.25], vec![0.; 2]],
            dims: vec![1, 3, 2],
            bounds: vec![vec![
                Bound::new(-0.75, 1.25),
                Bound::new(-1.25, 0.75),
                Bound::new(-1.25, 0.75),
            ]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            1,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.],
            vec![1.],
        )
        .unwrap()
    }
    fn plan(&self) -> Plan {
        Plan {
            components: vec![Component {
                negative: vec![1, 2],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.],
                alpha: vec![0., 0.625, 0.625, 1.],
            }],
            leftover: vec![],
        }
    }
    fn original(&self) -> MipParts {
        let mut e = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &[Bound::new(-1., 1.)],
            &self.bounds,
        )
        .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
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
    fn fake(&self, parts: &MipParts, heads: &[crate::ir::Col]) -> Vec<Q> {
        let mut point = vec![Q::zero(); parts.num_cols];
        point[parts.input_vars[0].0] = rat(3. / 8.);
        for j in 0..3 {
            point[1 + j] = rat(3. / 8.) + rat(self.biases[0][j]);
            point[heads[j].0] = rat(if j == 0 { 5. / 8. } else { 13. / 32. });
            point[heads[j].0 + 1] = rat(if j == 0 { 0.5 } else { 0.75 });
        }
        point[parts.output_vars[0].0] = rat(-3. / 16.);
        point
    }
    fn check_closed_graph(&self, parts: &MipParts) {
        let original = self.original();
        let heads = binding::head_columns(&self.encoding(), &original).unwrap();
        for x in [-1., -0.25, 0., 0.25, 0.75, 1.] {
            for ties in 0..8 {
                let mut point = vec![Q::zero(); parts.num_cols];
                point[parts.input_vars[0].0] = rat(x);
                for j in 0..3 {
                    let z = rat(x) + rat(self.biases[0][j]);
                    let side = if z.is_zero() {
                        (ties >> j) & 1
                    } else {
                        usize::from(z > Q::zero())
                    };
                    point[1 + j] = z.clone();
                    point[heads[j].0] = z.max(Q::zero());
                    point[heads[j].0 + 1] = rat(side as f64);
                }
                // Original encoder output uses original output weights,
                // not the constructor's conservatively reduced Source.c.
                point[parts.output_vars[0].0] = (0..3)
                    .map(|j| rat(self.weights[1][j]) * &point[heads[j].0])
                    .sum();
                for (j, col) in parts.problem.cols().iter().enumerate() {
                    assert!(!col.lb.is_finite() || point[j] >= rat(col.lb));
                    assert!(!col.ub.is_finite() || point[j] <= rat(col.ub));
                }
                for (index, row) in parts.problem.rows().iter().enumerate() {
                    if index == self.encoding().unsafe_row.0 {
                        continue;
                    }
                    let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
                    assert!(
                        !row.lb.is_finite() || lhs >= rat(row.lb),
                        "x={x} ties={ties} row={index}"
                    );
                    assert!(
                        !row.ub.is_finite() || lhs <= rat(row.ub),
                        "x={x} ties={ties} row={index}"
                    );
                }
            }
        }
    }
}

/// One real root LP receipt, unmodified Source, original native reservation,
/// original child holds. No native calls or captured debt are fabricated.
fn funded<'a>(
    p: &'a HullMarginProblem,
    plan: &Plan,
) -> (Box<HullRootMipCuts<'a>>, Budget, certificate::Checked) {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let cuts = Box::new(propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap());
    let mut b = Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, plan, &b, b.search_end, &mut || true).unwrap();
    component_rows::admit(&cuts, plan, &b).unwrap();
    native_publication::fund(&cuts, &b).unwrap();
    child::prepay(&cuts, plan, &b).unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(native_publication::funding(&b).funded_calls, 1);
    (cuts, b, checked)
}
fn install<'a>(
    mut cuts: Box<HullRootMipCuts<'a>>,
    b: &Budget,
    plan: &Plan,
    checked: certificate::Checked,
) -> Box<HullRootMipCuts<'a>> {
    let cost = plan.cost(cuts.problem).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, plan, checked, cost, b, &seal));
    cuts
}
fn separator_owner<'a>(
    p: &'a HullMarginProblem,
    plan: &Plan,
    keep_input: bool,
) -> Box<HullRootMipCuts<'a>> {
    let (cuts, b, checked) = funded(p, plan);
    input_phase_resources::prepay(&cuts, &b).unwrap();
    {
        let mut counts = b.state.lock().unwrap();
        let a = counts.phase_cube.admission.as_mut().unwrap();
        // Deterministic advisory control AFTER genuine admission/payment.
        // The new family still reconstructs/repairs every Source cell.
        a.input_target = Some(vec![3. / 8.]);
        let activity = a.activity.as_mut().unwrap();
        activity[1] = Some(0.75);
        activity[2] = Some(0.75);
        if !keep_input {
            assert!(a.quote.cell.as_mut().unwrap().input_phase.take().is_some());
            a.input_target = None;
        }
    }
    install(cuts, &b, plan, checked)
}
fn total(b: &Budget) -> (usize, usize) {
    let c = b.state.lock().unwrap();
    (
        c.work + c.reserved + c.component_reserved,
        b.admitted_bytes
            + c.shared_bytes
            + c.direction.cache_bytes
            + c.child.cache_bytes
            + c.component_rows.admitted_bytes,
    )
}
fn protected(b: &Budget) -> (usize, usize, usize, usize, usize, usize, usize) {
    let native = native_publication::funding(b);
    let c = b.state.lock().unwrap();
    (
        native.funded_calls,
        native.prepaid_work_units,
        native.admitted_bytes,
        c.child.cache_bytes,
        c.child.work_units,
        c.prepaid_collection_work,
        c.prepaid_bind_work,
    )
}

#[test]
fn input_phase_original_owner_appends_distinct_changed_input_row_and_retains_closed_graph() {
    let f = Fixture::new();
    let p = f.source();
    let plan = f.plan();
    assert_eq!(plan.components[0].g, [0.]);
    assert_ne!(p.c[0].to_bits(), 1f64.to_bits()); // Ordinary constructor reduction.
    let old = separator_owner(&p, &plan, false);
    let new = separator_owner(&p, &plan, true);
    let before_native = protected(&new.signed.as_ref().unwrap().budget);
    let mut old_parts = f.original();
    let mut new_parts = f.original();
    let heads = binding::head_columns(&f.encoding(), &new_parts).unwrap();
    let fake = f.fake(&new_parts, &heads);
    assert!(feasible(&new_parts, &fake)); // Includes unsafe output comparison.
    old.append_to(&mut old_parts, &f.encoding(), end()).unwrap();
    new.append_to(&mut new_parts, &f.encoding(), end()).unwrap();
    assert!(feasible(&old_parts, &fake));
    assert!(!feasible(&new_parts, &fake));
    assert_eq!(
        new_parts.problem.num_rows(),
        old_parts.problem.num_rows() + 1
    );
    assert_eq!(
        &new_parts.problem.rows()[..old_parts.problem.num_rows()],
        old_parts.problem.rows()
    );
    let last = new_parts.problem.rows().last().unwrap();
    assert!(last
        .coeffs
        .iter()
        .any(|(col, value)| *col == new_parts.input_vars[0].0 && *value != 0.));
    let lhs: Q = last.coeffs.iter().map(|(j, a)| rat(*a) * &fake[*j]).sum();
    assert!(rat(last.lb) - lhs > rat(0.1));
    f.check_closed_graph(&new_parts);
    let stats = new.signed.as_ref().unwrap().stats();
    assert_eq!(stats.component_rows.phase.cube.cell.input_phase.masks, 4);
    assert!(stats
        .component_rows
        .phase
        .cube
        .cell
        .input_phase
        .selected_candidate
        .is_some());
    assert_eq!(stats.source_replays, 1);
    assert_eq!(new.proposal_lp_calls, 1);
    assert_eq!(
        protected(&new.signed.as_ref().unwrap().budget),
        before_native
    );
}

#[test]
fn input_phase_exact_and_one_short_optional_limits_preserve_native_and_child_credit() {
    let f = Fixture::new();
    let p = f.source();
    let plan = f.plan();
    for memory in [false, true] {
        for short in [0, 1] {
            let (cuts, b, checked) = funded(&p, &plan);
            let q = input_phase_resources::quote(1, 3, 2, 33, 1).unwrap();
            let before = protected(&b);
            let old_quote = b
                .state
                .lock()
                .unwrap()
                .phase_cube
                .admission
                .as_ref()
                .unwrap()
                .quote;
            let (work, bytes) = total(&b);
            if memory {
                b.state.lock().unwrap().shared_bytes += MAX_BYTES - bytes - q.bytes + short;
            } else {
                b.charge(MAX_WORK - work - q.work + short).unwrap();
            }
            let result = input_phase_resources::prepay(&cuts, &b);
            assert_eq!(
                result.is_some(),
                short == 0,
                "memory={memory}, short={short}"
            );
            assert_eq!(protected(&b), before);
            let after = total(&b);
            assert!(after.0 <= MAX_WORK && after.1 <= MAX_BYTES);
            if short == 0 {
                assert_eq!(
                    if memory { after.1 } else { after.0 },
                    if memory { MAX_BYTES } else { MAX_WORK }
                );
            }
            {
                let c = b.state.lock().unwrap();
                let new_quote = c.phase_cube.admission.as_ref().unwrap().quote;
                let old_cell = old_quote.cell.unwrap();
                let new_cell = new_quote.cell.unwrap();
                assert_eq!(new_cell.ray.is_some(), old_cell.ray.is_some());
                assert_eq!(new_cell.lp.is_some(), old_cell.lp.is_some());
                assert_eq!(new_cell.joint.is_some(), old_cell.joint.is_some());
                assert_eq!(new_cell.work, old_cell.work);
                assert_eq!(new_cell.bytes, old_cell.bytes);
                assert_eq!(new_cell.input_phase.is_some(), short == 0);
            }
            let debt = total(&b);
            assert!(input_phase_resources::prepay(&cuts, &b).is_none());
            assert_eq!(total(&b), debt);
            assert_eq!(protected(&b), before);
            let cuts = install(cuts, &b, &plan, checked);
            let row = cuts.signed.as_ref().unwrap();
            let first = row.replay(&cuts, end(), &mut || true).unwrap();
            assert!(first.cube.unwrap().cell.is_some());
            let second = row.replay_with_cache(&cuts, end(), &mut || true).unwrap();
            let cache = second
                .cache
                .expect("protected child collection must survive optional refusal");
            assert!(cache.bind(&cuts, end(), &mut || true).is_some());
            assert_eq!(native_publication::funding(&b).funded_calls, 1);
            assert_eq!(cuts.proposal_lp_calls, 1);
        }
    }
}

#[test]
fn input_phase_foreign_expired_and_missing_point_refusals_cannot_reuse_old_authority() {
    let f = Fixture::new();
    let p = f.source();
    let plan = f.plan();
    let (mut cuts, b, checked) = funded(&p, &plan);
    let before = protected(&b);
    let debt = total(&b);
    for fault in 0..4 {
        let mut foreign = b.clone();
        match fault {
            0 => foreign.source ^= 1,
            1 => foreign.authority = cuts.deadline + Duration::from_secs(1),
            2 => {
                foreign.search_end = Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("existing one-second expired test clock")
            }
            _ => foreign.portfolio_phase_enabled = false,
        }
        assert!(input_phase_resources::prepay(&cuts, &foreign).is_none());
        assert_eq!(total(&b), debt);
        assert_eq!(protected(&b), before);
        assert!(!b.state.lock().unwrap().phase_cube.input_phase_attempted);
    }
    // Remove only the captured account after all old legitimate receipts and
    // credits were banked. New optional debt persists on capture refusal.
    let account = cuts.primal.take();
    assert!(account.is_some());
    assert!(input_phase_resources::prepay(&cuts, &b).is_none());
    assert_eq!(protected(&b), before);
    let paid = total(&b);
    assert!(paid.0 > debt.0 && paid.1 > debt.1);
    assert_eq!(
        b.state
            .lock()
            .unwrap()
            .component_rows
            .phase
            .cube
            .cell
            .input_phase
            .stop_reason,
        "capture-refused"
    );
    cuts.primal = account;
    assert!(input_phase_resources::prepay(&cuts, &b).is_none());
    assert_eq!(total(&b), paid);
    let cuts = install(cuts, &b, &plan, checked);
    let replay = cuts
        .signed
        .as_ref()
        .unwrap()
        .replay(&cuts, end(), &mut || true)
        .unwrap();
    let table = replay.cube.unwrap().cell.unwrap();
    assert!(table.input_g.is_none());
    assert!(table.ray.is_some() && table.joint.is_some());
    assert_eq!(table.joint, table.ray);
    assert_eq!(protected(&b), before);
}

#[test]
fn input_phase_complete_future_positive_cap_quote_and_shape_limits() {
    // Synthetic accounting boundary, not the measured benchmark Plan. The
    // quote funds future P<=16 even when its seed component has fewer heads.
    let q = input_phase_resources::quote(30, 128, 8, 62, 16).unwrap();
    assert_eq!((q.d, q.work, q.bytes), (30, 12_230_128, 7_584));
    for (d, h, k, terms, components) in [
        (0, 128, 8, 32, 16),
        (33, 128, 8, 65, 16),
        (30, 128, 1, 62, 16),
        (30, 128, 9, 62, 16),
        (30, 128, 8, 61, 16),
        (30, 128, 8, 62, 0),
        (30, 128, 8, 62, 129),
        (30, usize::MAX, 8, 62, 16),
    ] {
        assert!(input_phase_resources::quote(d, h, k, terms, components).is_none());
    }
}

#[test]
fn input_phase_whole_table_rollback_retains_first_middle_and_last_legacy_gain() {
    use certificate::I;
    // Changing each positive bias selects which closed side admits its pair
    // update. Only the chosen mask has two distinct eligible positive slots.
    // This tests the real Collector::apply rollback, including an already
    // overwritten prefix, against the complete legacy path on the same Source.
    for trigger in [0usize, 1, 3] {
        let bias: Vec<_> = (0..2)
            .map(|bit| if trigger & (1 << bit) == 0 { -0.5 } else { 0.5 })
            .collect();
        let p = source(
            vec![1., 0., 0., 1., 1., 0., 0., 1.],
            vec![bias[0], bias[1], 0., 0.],
            &[2., 2., -1., -1.],
            2,
        );
        let mut alpha = vec![0.; 8];
        alpha[2 * trigger] = 0.5;
        alpha[2 * trigger + 1] = 0.5;
        let plan = Plan {
            components: vec![Component {
                negative: vec![2, 3],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.; 2],
                alpha,
            }],
            leftover: vec![],
        };
        let (cuts, b, _) = funded(&p, &plan);
        input_phase_resources::prepay(&cuts, &b).unwrap();
        let quote = b
            .state
            .lock()
            .unwrap()
            .phase_cube
            .admission
            .as_ref()
            .unwrap()
            .quote
            .cell
            .unwrap();
        let c = &plan.components[0];
        let make = |input| Collector {
            quote,
            table: Table {
                component: 0,
                values: Vec::with_capacity(4),
                activity: Some([0.5; 8]),
                ray: Some(Vec::with_capacity(4)),
                joint: Some(Vec::with_capacity(4)),
                input_g: None,
            },
            pairs: numeric::pairs(&p, c, end(), &mut || true).unwrap(),
            improved: 0,
            joint_improved: 0,
            ray_improved: 0,
            block_improved: 0,
            input_phase: input,
            input_phase_failed: false,
        };
        // Deterministic valid shifted slopes isolate rollback independently of
        // the optional proposal heuristic. All four prices still use Source.
        let state = input_phase_collect::State {
            alternatives: [
                (Vec::with_capacity(4), vec![0.25; 2]),
                (Vec::with_capacity(4), vec![-0.125; 2]),
                (Vec::with_capacity(4), vec![-0.25; 2]),
            ],
            primary_g: vec![0.125; 2],
            target: vec![0.; 2],
        };
        let mut old = make(None);
        let mut new = make(Some(state));
        let debt = total(&b);
        for mask in 0..4 {
            let mut coefficients = vec![I::zero(); 2];
            let mut constant = I::zero();
            for (slot, &head) in c.positive.iter().enumerate() {
                let a = I::point(c.alpha[mask * 2 + slot])
                    .unwrap()
                    .scale(p.c[head])
                    .unwrap()
                    .scale(c.weight(slot).unwrap())
                    .unwrap();
                numeric::add_head(&p, head, a, &mut coefficients, &mut constant).unwrap();
            }
            for (bit, &head) in c.negative.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    numeric::add_head(
                        &p,
                        head,
                        I::point(p.c[head]).unwrap(),
                        &mut coefficients,
                        &mut constant,
                    )
                    .unwrap();
                }
            }
            let base = numeric::price(&p, &coefficients, constant).unwrap();
            old.apply(
                &cuts,
                c,
                0,
                mask,
                &coefficients,
                constant,
                base,
                end(),
                &mut || true,
            )
            .unwrap();
            new.apply(
                &cuts,
                c,
                0,
                mask,
                &coefficients,
                constant,
                base,
                end(),
                &mut || true,
            )
            .unwrap();
            if mask < trigger {
                assert!(new.input_phase.is_some());
            } else {
                assert!(new.input_phase.is_none() && new.input_phase_failed);
            }
            if mask == trigger {
                assert!(
                    old.table.joint.as_ref().unwrap()[mask]
                        > old.table.ray.as_ref().unwrap()[mask] + 0.1
                );
                assert_eq!(new.table.joint, old.table.joint);
            }
        }
        assert_eq!(new.table.values, old.table.values);
        assert_eq!(new.table.ray, old.table.ray);
        assert_eq!(new.table.joint, old.table.joint);
        assert!(new.table.input_g.is_none());
        assert_eq!(new.block_improved, old.block_improved);
        assert_eq!(total(&b), debt); // Optional work was paid, never refunded.
        let table = new.finish(&plan, &b, end(), &mut || true).unwrap();
        assert!(table.input_g.is_none());
        assert_eq!(table.joint, old.table.joint);
        assert_eq!(
            b.state
                .lock()
                .unwrap()
                .component_rows
                .phase
                .cube
                .cell
                .input_phase
                .stop_reason,
            "legacy-or-numeric-rollback"
        );
    }
}

#[test]
fn input_phase_selected_g_and_table_tampering_refuse_authenticated_publication_once() {
    for fault in 0..7 {
        let f = Fixture::new();
        let p = f.source();
        let plan = f.plan();
        let cuts = separator_owner(&p, &plan, true);
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let replay = owner.replay(&cuts, end(), &mut || true).unwrap();
        let minima = replay.component_minima.unwrap();
        let conditional = replay.conditional.unwrap();
        let mut table = replay.cube.unwrap();
        assert!(table.cell.as_ref().unwrap().input_g.is_some());
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("original rows");
        };
        let before = parts.problem.clone();
        let debt = owner.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            phase::prepare(
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
            let cell = table.cell.as_mut().unwrap();
            match fault {
                0 => {
                    let g = &mut cell.input_g.as_mut().unwrap()[0];
                    *g = g.next_up();
                }
                1 => {
                    cell.input_g.as_mut().unwrap().pop();
                }
                2 => cell.input_g.as_mut().unwrap().push(0.),
                3 => cell.input_g.as_mut().unwrap()[0] = f64::NAN,
                4 => cell.input_g = None,
                5 => {
                    let v = &mut cell.joint.as_mut().unwrap()[3];
                    *v = v.next_up();
                }
                _ => {
                    cell.joint.as_mut().unwrap().pop();
                }
            }
            assert_eq!(
                (cube::prepare(
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
                .unwrap())
                .len(),
                0
            );
            let mut polls = 0;
            assert_eq!(
                (cube::prepare(
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
                .unwrap())
                .len(),
                0
            );
            assert_eq!(polls, 1);
        })
        .unwrap();
        assert_eq!(parts.problem, before);
        let after = owner.stats();
        assert_eq!(after.work_units, debt);
        assert_eq!(after.component_rows.phase.cube.prepare_attempts, 1);
        assert_eq!(after.component_rows.phase.cube.cell.joint.prepared_rows, 0);
        assert_eq!(after.component_rows.phase.cube.cell.lp.calls, 0);
    }
}

#[test]
fn input_phase_projected_selectors_are_repaired_and_old_zero_input_column_is_added() {
    for missing in [1usize, 2] {
        let f = Fixture::new();
        let p = f.source();
        let plan = f.plan();
        let cuts = separator_owner(&p, &plan, true);
        let owner = cuts.signed.as_ref().unwrap();
        let mut parts = f.original();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let removed = parts.binary_vars[1..=missing].to_vec();
        assert!(parts.problem.relax_binary_integrality(&removed));
        parts.binary_vars.drain(1..=missing);
        parts.binary_widths.drain(1..=missing);
        let prefix = parts.problem.clone();
        let replay = owner.replay(&cuts, end(), &mut || true).unwrap();
        let minima = replay.component_minima.unwrap();
        let table = replay.cube.unwrap();
        let cell = table.cell.as_ref().unwrap();
        let new_g = cell.input_g.as_ref().unwrap();
        assert!(new_g[0] != 0. && plan.components[0].g[0] == 0.);
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("original rows");
        };
        let q = owner
            .budget
            .state
            .lock()
            .unwrap()
            .phase_cube
            .admission
            .as_ref()
            .unwrap()
            .quote
            .cell
            .unwrap();
        let prepared = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            for head in 1..=missing {
                assert!(binding.column(head).is_none());
            }
            prepare(
                cell,
                q,
                owner,
                binding,
                &old[0],
                Family::Joint,
                &mut None,
                end(),
                &mut || true,
            )
            .unwrap()
            .into_option()
            .expect("changed-input row survives missing selectors")
        })
        .unwrap()
        .0;
        assert!(!old[0]
            .coeffs
            .iter()
            .any(|(j, a)| *j == parts.input_vars[0].0 && *a != 0.));
        let actual_input: Vec<_> = prepared
            .coeffs
            .iter()
            .filter(|(j, _)| *j == parts.input_vars[0].0)
            .collect();
        assert_eq!(actual_input.len(), 1);
        assert_eq!(actual_input[0].1.to_bits(), (-2. * new_g[0]).to_bits());
        for removed in &removed {
            assert!(!prepared
                .coeffs
                .iter()
                .any(|(j, a)| *j == removed.0 && *a != 0.));
        }
        // Check every original mask against the ACTUALLY serialized selector
        // coefficients, including bits projected away by the real binding.
        let values = cell.joint.as_ref().unwrap();
        for (mask, value) in values.iter().enumerate() {
            let mut rhs = rat(2.) * rat(*value);
            for bit in 0..2 {
                if mask & (1 << bit) != 0 {
                    let selector = heads[bit + 1].0 + 1;
                    rhs += prepared
                        .coeffs
                        .iter()
                        .filter(|(j, _)| *j == selector)
                        .map(|(_, a)| rat(*a))
                        .sum::<Q>();
                }
            }
            assert!(rat(prepared.lb) <= rhs, "missing={missing}, mask={mask}");
        }
        if missing == 2 {
            assert_eq!(
                prepared
                    .coeffs
                    .iter()
                    .filter(|(j, _)| parts.binary_vars.iter().any(|c| c.0 == *j))
                    .count(),
                0
            );
            assert!(prepared.lb < old[0].lb); // Lower alone is not a valid different-g usefulness test.
        }
        parts.problem.add_row(
            prepared.lb,
            prepared.ub,
            prepared
                .coeffs
                .into_iter()
                .map(|(j, a)| (crate::ir::Col(j), a)),
        );
        assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
        f.check_closed_graph(&parts);
    }
}
