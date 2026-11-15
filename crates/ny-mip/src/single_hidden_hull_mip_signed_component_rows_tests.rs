// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Completed replay, exact doubled rows, and optional resource refusals.
use super::super::tests::{cuts, end, rat};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn problem(
    c: Vec<f64>,
    w: Vec<f64>,
    bias: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
) -> HullMarginProblem {
    let mut p =
        HullMarginProblem::new(w, bias, lo.len(), &c, &[0.25], 1, &[(0, 1.)], 0., lo, hi).unwrap();
    // Kernel controls author exact dyadic coefficients, independently of the
    // public constructor's conservative output-margin coefficient enclosure.
    p.c = c;
    p.d = 0.25;
    p
}
fn small() -> HullMarginProblem {
    problem(
        vec![2., -1., -1.],
        vec![1.; 3],
        vec![0.; 3],
        vec![-1.],
        vec![1.],
    )
}
fn partition(p: &HullMarginProblem) -> Plan {
    Plan {
        components: (1..3)
            .map(|j| Component {
                negative: vec![j],
                positive: vec![0],
                positive_units: vec![1],
                g: vec![0.; p.input_dim],
                alpha: vec![0., 1.],
            })
            .collect(),
        leftover: vec![],
    }
}
fn sealed(c: &HullRootMipCuts<'_>, plan: &Plan, admit_rows: bool) -> SignedMajorant {
    let b = Budget::new(c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(c.problem).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(c, plan, &b, b.search_end, &mut || true).unwrap();
    let seal = Arc::new(signature(c).collect());
    let row = row(c, plan, checked, cost, &b, &seal);
    if admit_rows {
        admit(c, plan, &b).unwrap();
    }
    row
}
fn columns(p: &HullMarginProblem) -> (MipParts, Vec<crate::ir::Col>) {
    let e = crate::MipEncoder::new(&vec![Bound::new(-1., 1.); p.input_dim]).unwrap();
    // The scalar export helper consumes an already authenticated mapping.
    // Actual encoder/model authentication is covered separately below and in
    // the integration/CLI controls, rather than inferred from this helper.
    let mut parts = e.into_parts();
    let heads = (0..p.hidden_dim)
        .map(|_| parts.problem.add_col(0., f64::NEG_INFINITY, f64::INFINITY))
        .collect();
    parts.num_cols = parts.problem.num_cols();
    (parts, heads)
}
fn coefficient(row: &RowSpec, col: crate::ir::Col) -> f64 {
    row.coeffs
        .iter()
        .filter(|(j, _)| *j == col.0)
        .map(|(_, a)| *a)
        .sum()
}
fn stats(row: &SignedMajorant) -> SignedComponentRowsReport {
    row.stats().component_rows
}

#[test]
fn signed_component_rows_exact_doubling_keeps_half_mass_and_last_input_graph_inclusion() {
    let p = problem(
        vec![2., -1., -1.],
        vec![1., 0., 1., 1., 1., -1.],
        vec![0.25, -0.125, 0.5],
        vec![-1.; 2],
        vec![1.; 2],
    );
    let c = cuts(&p);
    let mut plan = partition(&p);
    plan.components[0].g = vec![0.25, -0.5];
    plan.components[1].g = vec![-0.125, 0.25];
    plan.components[0].alpha = vec![0.25, 0.75];
    plan.components[1].alpha = vec![0.5, 1.];
    let row = sealed(&c, &plan, true);
    let (parts, heads) = columns(&p);
    let before = parts.problem.clone();
    let (before_work, component_hold) = {
        let state = row.budget.state.lock().unwrap();
        (state.work, state.component_reserved)
    };
    assert!(component_hold > 0);
    // Actual scalar-first publication must collect complete local evidence,
    // rather than merely consume its protected work credit.
    let replay = row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let minima = replay.component_minima.unwrap();
    assert_eq!(minima.len(), plan.components.len());
    let collection = stats(&row);
    assert_eq!(collection.collection_attempts, 1);
    assert_eq!(collection.collected_components, minima.len());
    assert_eq!(collection.prepared_rows, 0);
    assert_eq!(collection.appended_rows, 0);
    assert_eq!(collection.reserved_work_units, 0);
    assert!(collection.work_units >= component_hold);
    let state = row.budget.state.lock().unwrap();
    assert_eq!(state.component_reserved, 0);
    assert_eq!(state.work, before_work + row.cost + component_hold);
    drop(state);
    let Prepared::Rows(out) =
        prepare(&row, &c, &parts, &heads, &minima, end(), &mut || true).unwrap()
    else {
        panic!("finite complete component set was refused")
    };
    assert_eq!(out.len(), 2);
    for (k, component) in plan.components.iter().enumerate() {
        let r = &out[k];
        assert_eq!(r.lb.to_bits(), (2. * minima[k]).to_bits());
        assert_eq!(coefficient(r, heads[0]), p.c[0]);
        assert_eq!(coefficient(r, heads[k + 1]), 2. * p.c[k + 1]);
        for i in 0..2 {
            assert_eq!(coefficient(r, parts.input_vars[i]), -2. * component.g[i]);
        }
        for a in -2..=2 {
            for b in -2..=2 {
                let x = [rat(a as f64 / 2.), rat(b as f64 / 2.)];
                let h: Vec<_> = (0..3)
                    .map(|j| {
                        (rat(p.b1[j]) + (0..2).map(|i| rat(p.w_row(j)[i]) * &x[i]).sum::<Q>())
                            .max(Q::zero())
                    })
                    .collect();
                let lhs = rat(p.c[0]) * &h[0] + rat(2. * p.c[k + 1]) * &h[k + 1]
                    - (0..2).map(|i| rat(2. * component.g[i]) * &x[i]).sum::<Q>();
                assert!(lhs >= rat(r.lb));
            }
        }
    }
    assert_eq!(parts.problem, before);
    assert_eq!(stats(&row).prepared_rows, 2);
    assert_eq!(stats(&row).appended_rows, 0);
    assert_eq!(row.stats().source_replays, 1);
    assert_eq!(c.proposal_lp_calls, 0);
}

#[test]
fn signed_component_rows_minimum_subnormal_half_exports_original_nonzero_coefficient() {
    let tiny = f64::from_bits(1);
    let p = problem(
        vec![tiny, -tiny, -tiny],
        vec![0.; 3],
        vec![0.; 3],
        vec![-1.],
        vec![1.],
    );
    let c = cuts(&p);
    let mut plan = partition(&p);
    for v in &mut plan.components {
        v.alpha.fill(0.);
    }
    let interval = plan.components[0].coefficient(&p, 0).unwrap();
    let half = rat(tiny) / rat(2.);
    assert_eq!(tiny * 0.5, 0.);
    assert!(rat(interval.lo) <= half && half <= rat(interval.hi));
    let row = sealed(&c, &plan, true);
    let (parts, heads) = columns(&p);
    let out = row
        .prepare_rows(&c, &parts, &heads, end(), &mut || true)
        .unwrap();
    assert_eq!(out.len(), 3);
    for r in &out[1..] {
        assert_eq!(coefficient(r, heads[0]).to_bits(), tiny.to_bits());
        assert!(r.coeffs.iter().any(|(_, v)| *v == -2. * tiny));
        assert!(r.lb.is_finite() && r.coeffs.iter().all(|(_, v)| v.is_finite()));
    }
}

#[test]
fn signed_component_rows_stable_active_free_original_column_accepts_exact_model_binding() {
    let weights = vec![vec![1.; 3], vec![2., -1., -1., 0., 0., 0.]];
    let biases = vec![vec![2., 0., 0.], vec![0.25, 0.]];
    let dims = vec![1, 3, 2];
    let bounds = vec![vec![
        Bound::new(1., 3.),
        Bound::new(-1., 1.),
        Bound::new(-1., 1.),
    ]];
    let p = HullMarginProblem::new(
        weights[0].clone(),
        biases[0].clone(),
        1,
        &weights[1],
        &biases[1],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    let mut c = cuts(&p);
    let row = sealed(&c, &partition(&p), true);
    c.signed = Some(row);
    let mut e =
        crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.)], &bounds)
            .unwrap();
    let unsafe_row = e.constrain_output_leq_row(0, 1).unwrap();
    let mut parts = e.into_parts();
    let encoding = HullMipEncoding {
        weights: &weights,
        biases: &biases,
        dimensions: &dims,
        hidden_bounds: &bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row,
    };
    let heads = binding::head_columns(&encoding, &parts).unwrap();
    let active = &parts.problem.cols()[heads[0].0];
    assert!(
        active.lb < 0. && active.ub > 0.,
        "stable head aliases an unrestricted affine column"
    );
    let old = parts.problem.clone();
    let r = c.append_to(&mut parts, &encoding, end()).unwrap();
    assert_eq!(r.signed.as_ref().unwrap().component_rows.appended_rows, 2);
    assert_eq!(&parts.problem.rows()[..old.num_rows()], old.rows());
    assert_eq!(parts.problem.cols(), old.cols());
    assert_eq!(r.signed.as_ref().unwrap().source_replays, 1);
    let row = c.signed.as_ref().unwrap();
    assert_eq!(
        row.replay_inner(&c, end(), &mut || true, false)
            .unwrap()
            .root_lower
            .to_bits(),
        row.lower.to_bits()
    );
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(stats(row).appended_rows, 2);
    assert_eq!(stats(row).stop_reason, "appended");
}

#[test]
fn signed_component_rows_allocation_or_sealed_witness_mutation_never_collects_planes() {
    for kind in 0..4 {
        let p = small();
        let c = cuts(&p);
        let mut row = sealed(&c, &partition(&p), true);
        match kind {
            0 => row.plan.components[0].positive_units[0] = 2,
            1 => row.plan.components[0].negative[0] = 2,
            2 => row.plan.components[0].g[0] = 0.125,
            _ => row.plan.components[0].alpha[1] = 0.25,
        }
        let (parts, heads) = columns(&p);
        assert!(row
            .prepare_rows(&c, &parts, &heads, end(), &mut || true)
            .is_none());
        assert_eq!(stats(&row).collected_components, 0);
        assert_eq!(stats(&row).prepared_rows, 0);
        assert_eq!(row.stats().source_replays, 0);
    }
}

#[test]
fn signed_component_rows_foreign_source_or_tuple_refuses_before_optional_replay() {
    let p = small();
    let c = cuts(&p);
    let row = sealed(&c, &partition(&p), true);
    let foreign = p.clone();
    let other = cuts(&foreign);
    let (parts, heads) = columns(&p);
    assert!(row
        .prepare_rows(&other, &parts, &heads, end(), &mut || true)
        .is_none());
    assert_eq!(row.budget.state.lock().unwrap().replay_attempts, 0);
    let mut changed = cuts(&p);
    changed.multipliers.alpha[0] = 0.25;
    assert!(row
        .prepare_rows(&changed, &parts, &heads, end(), &mut || true)
        .is_none());
    assert_eq!(row.stats().source_replays, 0);
    assert_eq!(stats(&row).prepared_rows, 0);
}

#[test]
fn signed_component_rows_last_mask_refusal_retains_second_complete_root_replay() {
    let p = small();
    let c = cuts(&p);
    let row = sealed(&c, &partition(&p), true);
    let (parts, heads) = columns(&p);
    let stop = row.budget.state.lock().unwrap().checked_masks + 4;
    let mut poll = || row.budget.state.lock().unwrap().checked_masks < stop;
    assert!(row
        .prepare_rows(&c, &parts, &heads, end(), &mut poll)
        .is_none());
    assert_eq!(row.budget.state.lock().unwrap().checked_masks, stop);
    assert_eq!(stats(&row).collection_attempts, 1);
    assert_eq!(stats(&row).collected_components, 0);
    assert_eq!(stats(&row).prepared_rows, 0);
    assert_eq!(row.stats().source_replays, 0);
    let replay = row.replay_inner(&c, end(), &mut || true, false).unwrap();
    assert_eq!(replay.root_lower.to_bits(), row.lower.to_bits());
    assert_eq!(row.budget.state.lock().unwrap().replay_attempts, 2);
}

#[test]
fn signed_component_rows_finite_doubling_overflow_discards_entire_optional_set_only() {
    let p = problem(
        vec![1., -1., -0.75 * f64::MAX],
        vec![0.; 3],
        vec![0.; 3],
        vec![-1.],
        vec![1.],
    );
    let c = cuts(&p);
    let mut plan = partition(&p);
    for v in &mut plan.components {
        v.alpha.fill(0.);
    }
    let row = sealed(&c, &plan, true);
    let (parts, heads) = columns(&p);
    let original = parts.problem.clone();
    let out = row
        .prepare_rows(&c, &parts, &heads, end(), &mut || true)
        .unwrap();
    assert_eq!(
        out.len(),
        1,
        "a finite earlier component cannot leak from a refused set"
    );
    assert_eq!(stats(&row).stop_reason, "doubling-overflow");
    assert_eq!(stats(&row).prepared_rows, 0);
    assert_eq!(stats(&row).appended_rows, 0);
    assert_eq!(parts.problem, original);
    assert_eq!(row.stats().source_replays, 1);
    assert_eq!(
        row.replay_inner(&c, end(), &mut || true, false)
            .unwrap()
            .root_lower
            .to_bits(),
        row.lower.to_bits()
    );
}

#[test]
fn signed_component_rows_optional_payload_or_work_refusal_preserves_both_publication_holds() {
    for payload in [false, true] {
        let p = small();
        let c = cuts(&p);
        let mut row = sealed(&c, &partition(&p), false);
        if payload {
            row.budget.admitted_bytes = MAX_BYTES;
        } else {
            let n = row.budget.remaining().unwrap();
            row.budget.charge(n).unwrap();
        }
        let held = row.budget.state.lock().unwrap().reserved;
        assert!(admit(&c, &row.plan, &row.budget).is_none());
        assert_eq!(row.budget.state.lock().unwrap().reserved, held);
        assert_eq!(held, 2 * row.cost);
        let (parts, heads) = columns(&p);
        assert_eq!(
            row.prepare_rows(&c, &parts, &heads, end(), &mut || true)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            row.lower.to_bits()
        );
        assert_eq!(row.stats().source_replays, 2);
        assert_eq!(row.budget.state.lock().unwrap().reserved, 0);
        assert_eq!(stats(&row).appended_rows, 0);
    }
}

#[test]
fn signed_component_rows_construction_hold_cannot_be_spent_by_advice_or_second_pass() {
    let p = small();
    let c = cuts(&p);
    let row = sealed(&c, &partition(&p), true);
    let (hold, publication) = {
        let b = row.budget.state.lock().unwrap();
        (b.component_reserved, b.reserved)
    };
    assert!(hold > 0 && publication == 2 * row.cost);
    let remaining = row.budget.remaining().unwrap();
    assert!(row.budget.charge(remaining + 1).is_none());
    row.budget.charge(remaining).unwrap();
    assert!(row.budget.proof(row.cost, false).is_none());
    let (parts, heads) = columns(&p);
    assert_eq!(
        row.prepare_rows(&c, &parts, &heads, end(), &mut || true)
            .unwrap()
            .len(),
        3
    );
    assert_eq!(row.budget.state.lock().unwrap().component_reserved, 0);
    assert_eq!(stats(&row).reserved_work_units, 0);
    assert_eq!(
        stats(&row).work_units,
        hold + (p.hidden_dim + p.input_dim + 2 + 1) * 32
            + (p.hidden_dim + p.input_dim + partition(&p).components.len() + c.len() + 2) * 32
    );
    assert_eq!(
        row.replay_inner(&c, end(), &mut || true, false)
            .unwrap()
            .root_lower
            .to_bits(),
        row.lower.to_bits()
    );
    assert_eq!(row.stats().work_units, MAX_WORK);
    assert_eq!(row.stats().source_replays, 2);
}

#[test]
fn signed_component_rows_original_clock_and_single_collection_are_never_renewed() {
    let p = small();
    let c = cuts(&p);
    let mut row = sealed(&c, &partition(&p), true);
    let (parts, heads) = columns(&p);
    let past = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    row.budget.authority = past;
    assert!(row
        .prepare_rows(
            &c,
            &parts,
            &heads,
            end() + Duration::from_secs(60),
            &mut || true
        )
        .is_none());
    assert_eq!(stats(&row).collection_attempts, 0);
    assert_eq!(row.stats().source_replays, 0);
    let row = sealed(&c, &partition(&p), true);
    assert_eq!(
        row.prepare_rows(&c, &parts, &heads, end(), &mut || true)
            .unwrap()
            .len(),
        3
    );
    assert!(admit(&c, &row.plan, &row.budget).is_none());
    assert_eq!(
        row.prepare_rows(&c, &parts, &heads, end(), &mut || true)
            .unwrap()
            .len(),
        1
    );
    assert!(row
        .prepare_rows(&c, &parts, &heads, end(), &mut || true)
        .is_none());
    assert_eq!(stats(&row).collection_attempts, 1);
    assert_eq!(row.stats().source_replays, 2);
}

#[test]
fn signed_component_rows_final_publisher_refusal_appends_neither_aggregate_nor_components() {
    let weights = vec![vec![1.; 3], vec![2., -1., -1., 0., 0., 0.]];
    let biases = vec![vec![0.; 3], vec![0.25, 0.]];
    let dims = vec![1, 3, 2];
    let bounds = vec![vec![Bound::new(-1., 1.); 3]];
    let p = HullMarginProblem::new(
        weights[0].clone(),
        biases[0].clone(),
        1,
        &weights[1],
        &biases[1],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    let mut c = cuts(&p);
    c.signed = Some(sealed(&c, &partition(&p), true));
    let mut encoder =
        crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.)], &bounds)
            .unwrap();
    let unsafe_row = encoder.constrain_output_leq_row(0, 1).unwrap();
    let mut parts = encoder.into_parts();
    let encoding = HullMipEncoding {
        weights: &weights,
        biases: &biases,
        dimensions: &dims,
        hidden_bounds: &bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row,
    };
    let before = parts.problem.clone();
    let binary = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let row = c.signed.as_ref().unwrap();
    let mut poll = || stats(row).prepared_rows == 0;
    assert!(c
        .append_with_poll(&mut parts, &encoding, end(), &mut poll)
        .is_none());
    assert_eq!(
        stats(row).prepared_rows,
        2,
        "refuse after the entire optional set is prepared"
    );
    assert_eq!(stats(row).appended_rows, 0);
    assert_eq!(parts.problem, before);
    assert_eq!(parts.binary_vars, binary);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(row.stats().source_replays, 1);
    assert_eq!(
        row.replay_inner(&c, end(), &mut || true, false)
            .unwrap()
            .root_lower
            .to_bits(),
        row.lower.to_bits()
    );
}

#[test]
fn signed_component_rows_admission_foreign_budget_or_expired_search_cannot_borrow_publication_hold()
{
    for kind in 0..3 {
        let p = small();
        let c = cuts(&p);
        let row = sealed(&c, &partition(&p), false);
        let mut b = row.budget.clone();
        match kind {
            0 => b.source ^= 1,
            1 => b.dimensions.0 += 1,
            _ => {
                b.search_end = Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("overflow when subtracting duration from instant")
            }
        }
        let held = b.state.lock().unwrap().reserved;
        assert!(admit(&c, &row.plan, &b).is_none());
        assert_eq!(b.state.lock().unwrap().component_reserved, 0);
        assert_eq!(b.state.lock().unwrap().reserved, held);
        let (parts, heads) = columns(&p);
        assert_eq!(
            row.prepare_rows(&c, &parts, &heads, end(), &mut || true)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            row.replay_inner(&c, end(), &mut || true, false)
                .unwrap()
                .root_lower
                .to_bits(),
            row.lower.to_bits()
        );
        assert_eq!(row.stats().source_replays, 2);
    }
}
