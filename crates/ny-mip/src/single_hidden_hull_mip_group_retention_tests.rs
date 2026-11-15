// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::tests::{budget, cycle, end, manual, mixed, pair_baseline};
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::Zero;

fn floor(cuts: &HullRootMipCuts<'_>) -> f64 {
    let row = compact::replay(cuts, end(), &mut || true).unwrap();
    compact::root_lower(cuts, &row, end(), &mut || true).unwrap()
}
fn row_bits(cuts: &HullRootMipCuts<'_>) -> (Vec<u64>, u64) {
    let row = compact::replay(cuts, end(), &mut || true).unwrap();
    (
        row.g.iter().map(|x| x.to_bits()).collect(),
        row.upper.to_bits(),
    )
}

#[test]
fn group_refine_neutral_cycle_discovery_closes_both_coordinate_head_permutations() {
    for permuted in [false, true] {
        let p = cycle(7, 6.5, permuted);
        let mut cuts = manual(&p);
        let old = cuts.checked_lower;
        let multipliers = cuts.multipliers.clone();
        joint::discover_compact(&mut cuts, end(), &mut || true);
        assert!(cuts.compact_root_proof(end()).is_none());
        if cuts.compact.is_some() {
            assert!(floor(&cuts) <= -0.499);
        }
        refine(&mut cuts, end(), Duration::ZERO);
        let proof = cuts.compact_root_proof(end()).unwrap();
        assert!(proof.verified() && proof.lower() > 0.499 && proof.lower() <= 0.5);
        assert_eq!(cuts.checked_lower.to_bits(), old.to_bits());
        assert_eq!(cuts.multipliers, multipliers);
        assert_eq!(cuts.proposal_lp_calls, 0);
    }
}

#[test]
fn group_refine_full_q_keeps_positive_and_final_negative_roles_in_safe_unsafe_and_tie() {
    for bias in [2., 1.5, 1.] {
        let p = mixed(bias);
        let mut cuts = pair_baseline(&p);
        let before = floor(&cuts);
        assert!(before <= bias - 2.5 && before > bias - 2.501);
        refine(&mut cuts, end(), Duration::ZERO);
        let after = floor(&cuts);
        let optimum = bias - 1.5;
        assert!(
            after <= optimum && after > optimum - 0.001,
            "bias={bias}, {after}"
        );
        assert!(after > before + 0.999);
        let proof = cuts.compact_root_proof(end());
        assert_eq!(proof.is_some(), bias == 2.);
        if let Some(proof) = proof {
            assert!(proof.lower() > 0.499 && proof.lower() <= 0.5);
        }
        assert_eq!(
            cuts.proposal_lp_calls, 0,
            "refinement must not hide a root LP"
        );
    }
}

#[test]
fn group_refine_expired_optional_authority_retains_live_completed_pair_baseline() {
    let p = mixed(2.);
    let mut cuts = pair_baseline(&p);
    let before = row_bits(&cuts);
    let lower = floor(&cuts);
    let original = cuts.multipliers.clone();
    refine(
        &mut cuts,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        Duration::ZERO,
    );
    assert_eq!(row_bits(&cuts), before);
    assert_eq!(floor(&cuts).to_bits(), lower.to_bits());
    assert_eq!(cuts.multipliers, original);
    assert!(
        cuts.deadline > Instant::now(),
        "optional stop must not expire original authority"
    );
}

#[test]
fn group_support_partial_last_mask_never_becomes_a_complete_certificate() {
    let p = cycle(3, 2.25, false);
    let cuts = manual(&p);
    let b = budget();
    b.state.lock().unwrap().masks = MAX_SEARCH_MASKS - 7;
    assert!(support(&cuts, &[0, 1, 2], &b, &mut || true).is_none());
    let counts = b.state.lock().unwrap();
    assert_eq!(counts.masks, MAX_SEARCH_MASKS);
    assert!(counts.attempts <= MAX_ATTEMPTS);
    assert!(counts.work <= MAX_WORK);
    drop(counts);
    let b = budget();
    let mut poll = || b.state.lock().unwrap().masks < 8;
    assert!(support(&cuts, &[0, 1, 2], &b, &mut poll).is_none());
    assert_eq!(
        b.state.lock().unwrap().masks,
        8,
        "all masks charged is insufficient when final-mask replay was refused"
    );
}

#[test]
fn group_refine_partial_last_mask_preserves_the_live_completed_pair_baseline() {
    let p = mixed(2.);
    let mut cuts = pair_baseline(&p);
    let before = row_bits(&cuts);
    let old_floor = floor(&cuts);
    let b = budget();
    // The first geometry component is the neutral odd3. Its final eighth
    // mask cannot be charged; no incomplete certificate may replace the
    // already completed independent pair covering the other coordinates.
    b.state.lock().unwrap().masks = MAX_SEARCH_MASKS - 7;
    search::refine_with_budget_and_poll(&mut cuts, &b, &mut || true);
    assert_eq!(b.state.lock().unwrap().masks, MAX_SEARCH_MASKS);
    assert_eq!(b.state.lock().unwrap().attempts, 1);
    assert_eq!(row_bits(&cuts), before);
    assert_eq!(floor(&cuts).to_bits(), old_floor.to_bits());
    assert!(cuts.deadline > Instant::now());
}

#[test]
fn group_shared_attempt_mask_and_work_caps_refuse_without_hiding_extra_work() {
    let p = cycle(3, 2.25, false);
    let cuts = manual(&p);
    let b = budget();
    b.state.lock().unwrap().attempts = MAX_ATTEMPTS;
    assert!(support(&cuts, &[0, 1, 2], &b, &mut || true).is_none());
    assert_eq!(b.state.lock().unwrap().attempts, MAX_ATTEMPTS);
    let b = budget();
    b.state.lock().unwrap().masks = MAX_SEARCH_MASKS;
    assert!(support(&cuts, &[0, 1, 2], &b, &mut || true).is_none());
    assert_eq!(b.state.lock().unwrap().masks, MAX_SEARCH_MASKS);
    let b = budget();
    b.state.lock().unwrap().work = MAX_WORK - 1;
    assert!(support(&cuts, &[0, 1, 2], &b, &mut || true).is_none());
    assert!(b.state.lock().unwrap().work <= MAX_WORK);
    assert!(Budget::new(
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        Duration::ZERO
    )
    .is_none());
}

#[test]
fn group_memory_permits_release_bytes_and_publication_reservation_rejects_overflow() {
    let b = budget();
    let permit = b.memory(MAX_BYTES - 128).unwrap();
    assert!(b.memory(129).is_none());
    assert!(b.state.lock().unwrap().peak_bytes <= MAX_BYTES);
    drop(permit);
    assert_eq!(b.state.lock().unwrap().bytes, 0);
    assert!(b.reserve(usize::MAX, 1).is_none());
    assert!(b.reserve(1, usize::MAX).is_none());
    assert!(b.reserve(1, MAX_SELECTED_MASKS + 1).is_none());
    assert_eq!(b.state.lock().unwrap().reserved_work, 0);
    assert_eq!(b.state.lock().unwrap().reserved_masks, 0);
}

#[test]
fn group_reserved_publication_survives_exhaustion_of_all_optional_search_credits() {
    let p = cycle(3, 2.25, false);
    let mut cuts = manual(&p);
    let b = budget();
    let row = support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap();
    let aggregate = compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).unwrap();
    let (work, masks) = replay_cost(&cuts, &aggregate.groups).unwrap();
    b.reserve(work, masks).unwrap();
    cuts.compact = Some(aggregate);
    cuts.compact_mode = true;
    let optional = {
        let s = b.state.lock().unwrap();
        MAX_WORK - s.work - s.reserved_work
    };
    b.charge(optional, 0, false).unwrap();
    assert!(b.charge(1, 0, false).is_none());
    // Two genuine support replays correspond to the existing append and
    // root-proof passes. Optional search must not consume their reserve.
    for _ in 0..2 {
        let proof = cuts.compact_root_proof(end()).unwrap();
        assert!(proof.verified() && proof.lower() > 0.249 && proof.lower() <= 0.25);
    }
    let s = b.state.lock().unwrap();
    assert!(s.work <= MAX_WORK && s.replay_masks == 2 * masks && s.reserved_masks == 0);
}

#[test]
fn group_aggregate_rejects_overlap_and_foreign_geometry_before_replacing_baseline() {
    let p = mixed(2.);
    let cuts = pair_baseline(&p);
    let original = row_bits(&cuts);
    let b = budget();
    let a = support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap();
    let overlap = support(&cuts, &[1, 2, 3], &b, &mut || true).unwrap();
    assert!(compact::build_group_aggregate(&cuts, vec![a, overlap], &b, &mut || true).is_none());
    assert_eq!(row_bits(&cuts), original);
    let mut foreign = mixed(2.);
    foreign.w1[0] = 1.5;
    let other = manual(&foreign);
    let row = support(&other, &[0, 1, 2], &budget(), &mut || true).unwrap();
    assert!(compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).is_none());
    assert_eq!(row_bits(&cuts), original);
}

#[test]
fn group_certificate_cannot_rebind_to_changed_source_at_the_same_problem_address() {
    let mut p = cycle(3, 2.25, false);
    let b = budget();
    let row = {
        let cuts = manual(&p);
        support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap()
    };
    // The old cuts borrow has ended, but p itself and its address persist.
    // We change only a first bias: heads/weights/point-g are unchanged.
    p.b1[0] = 10.;
    let cuts = manual(&p);
    assert!(compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).is_none());
}

#[test]
fn group_certificate_cannot_reset_its_shared_work_ledger_with_a_fresh_budget() {
    let p = cycle(3, 2.25, false);
    let cuts = manual(&p);
    let original = budget();
    let row = support(&cuts, &[0, 1, 2], &original, &mut || true).unwrap();
    assert!(original.state.lock().unwrap().masks >= 8);
    let fresh = budget();
    assert!(replay(&cuts, &row, &fresh, &mut || true).is_none());
    assert!(compact::build_group_aggregate(&cuts, vec![row], &fresh, &mut || true).is_none());
}

#[test]
fn group_aggregate_and_root_refuse_expired_stored_clock_without_renewal() {
    let p = cycle(3, 2.25, false);
    let cuts = manual(&p);
    let mut b = budget();
    let row = support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap();
    b.authority = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    b.search_end = b.authority;
    assert!(compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).is_none());
    let p = mixed(2.);
    let mut cuts = pair_baseline(&p);
    refine(&mut cuts, end(), Duration::ZERO);
    assert!(cuts.compact_root_proof(end()).is_some());
    cuts.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(cuts.compact_root_proof(end()).is_none());
}

#[test]
fn group_actual_ay_safe_lp_and_unsafe_witness_bind_the_original_odd3_graph() {
    for bias in [2.25, 1.75] {
        let base = cycle(3, bias, false);
        let weights = vec![base.w1.clone(), vec![-1., -1., -1., 0., 0., 0.]];
        let biases = vec![vec![0.; 3], vec![bias, 0.]];
        let dims = vec![3, 3, 2];
        let bounds = vec![vec![Bound::new(-2., 2.); 3]];
        let p = HullMarginProblem::new(
            weights[0].clone(),
            biases[0].clone(),
            3,
            &weights[1],
            &biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 3],
            vec![1.; 3],
        )
        .unwrap();
        assert_eq!(p.w1, base.w1);
        let mut cuts = manual(&p);
        let b = budget();
        let row = support(&cuts, &[0, 1, 2], &b, &mut || true).unwrap();
        cuts.compact =
            Some(compact::build_group_aggregate(&cuts, vec![row], &b, &mut || true).unwrap());
        cuts.compact_mode = true;
        let mut encoder =
            crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.); 3], &bounds)
                .unwrap();
        let unsafe_row = encoder.constrain_output_leq_row(0, 1).unwrap();
        let mut parts = encoder.into_parts();
        let binding = HullMipEncoding {
            weights: &weights,
            biases: &biases,
            dimensions: &dims,
            hidden_bounds: &bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row,
        };
        cuts.append_to(&mut parts, &binding, end()).unwrap();
        if bias == 2.25 {
            let mut ir = crate::ir::MilpProblem::new();
            for c in parts.problem.cols() {
                ir.add_col(c.obj, c.lb, c.ub);
            }
            for (i, r) in parts.problem.rows().iter().enumerate() {
                if i != unsafe_row.0 {
                    ir.add_row(
                        r.lb,
                        r.ub,
                        r.coeffs.iter().map(|(j, c)| (crate::ir::Col(*j), *c)),
                    );
                }
            }
            let model = crate::ay_lib::to_ay_model(&ir).unwrap();
            let opts = SolveOpts::new()
                .with_deadline(end())
                .with_time_limit(Duration::from_secs(30));
            let mut session = ay_milp::LpSession::new(&model, &opts).unwrap();
            let objective = [
                (model.col_at(parts.output_vars[0].0).unwrap(), 1.),
                (model.col_at(parts.output_vars[1].0).unwrap(), -1.),
            ];
            let out = session
                .rigorous_bound_expr(&objective, ay_milp::Sense::Minimize)
                .unwrap();
            let lower = crate::lp_propose_certify::rigorous_lower_f64(&out).unwrap();
            assert!(lower > 0.249 && lower <= 0.25, "{lower}");
        }
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
        let result = solver.check_feasibility().unwrap();
        if bias == 2.25 {
            assert!(matches!(result, MipResult::Unsat { certified: true }));
        } else {
            let MipResult::Sat { input_values, .. } = result else {
                panic!("original unsafe graph excluded")
            };
            let x: Vec<_> = input_values
                .iter()
                .take(3)
                .map(|v| Q::from_float(*v).unwrap())
                .collect();
            assert_eq!(x.len(), 3);
            assert!(x
                .iter()
                .all(|v| *v >= Q::from_integer((-1).into()) && *v <= Q::from_integer(1.into())));
            let mut margin = Q::from_float(bias).unwrap();
            for j in 0..3 {
                margin -= (&x[j] - &x[(j + 1) % 3]).max(Q::zero());
            }
            assert!(
                margin <= Q::zero(),
                "solver input failed original exact rational replay"
            );
            assert!(cuts.compact_root_proof(end()).is_none());
        }
    }
}
