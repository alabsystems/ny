// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact finite-table optima and original publication/resource controls.
use super::*;
use lp::{SignedTableSupportReport, Support};

pub(super) fn advised<'a>(p: &'a HullMarginProblem, plan: &Plan) -> HullRootMipCuts<'a> {
    let m = HullMultipliers::zeros(p);
    let floor = certify_hull_lower_bound(p, &m).unwrap();
    let mut cuts = root::prepare(p, m, floor, end(), &mut || true).unwrap();
    // Real descriptor/capture/debt path with synthetic finite advice. This
    // fixture claims no original-LP discovery or primal feasibility.
    let bounds = preactivation_bounds(p);
    let roles = classify(p, &bounds);
    let mut capture = primal_capture::Attempt::new_with_activity(p, cuts.deadline);
    capture.admit(&roles);
    let model = build_hull_lp_with_capture(p, &roles, &bounds, capture.capture.as_mut()).unwrap();
    assert!(capture.enable_same_call(model.model.num_cols()));
    capture.finish(
        Some(vec![0.5; model.model.num_cols()].into_boxed_slice()),
        &mut || true,
    );
    assert!(capture.snapshot().decoded);
    cuts.primal_report = Some(Arc::clone(&capture.report));
    cuts.primal = capture.take_account();
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    b.portfolio_phase_enabled = true;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, plan, &b, b.search_end, &mut || true).unwrap();
    component_rows::admit(&cuts, plan, &b).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, plan, checked, cost, &b, &seal));
    cuts
}

fn incumbent(values: &[f64], mapped: &[bool], activity: &[f64]) -> Support {
    let k = mapped.len();
    let mut a = [0.; 8];
    let lower = repair(values, &a[..k], end(), &mut || true).unwrap();
    let (next, lower) =
        support::choose(values, a, lower, mapped, Some(activity), end(), &mut || {
            true
        })
        .unwrap();
    a = next;
    Support { a, lower }
}
fn exact_check(values: &[f64], a: &Support, k: usize) {
    for (mask, &v) in values.iter().enumerate() {
        let mut rhs = rat(2.) * rat(v);
        for j in 0..k {
            if mask & (1 << j) != 0 {
                rhs += rat(a.a[j]);
            }
        }
        assert!(rat(a.lower) <= rhs, "mask={mask}");
    }
}
#[test]
fn table_lp_escapes_coordinate_stall_and_certifies_full_256_mask_optimum() {
    for k in [3, 8] {
        let values: Vec<_> = (0usize..1 << k)
            .map(|m| if m.is_power_of_two() { 0. } else { 1. })
            .collect();
        let mapped = vec![true; k];
        let activity = vec![0.5; k];
        let old = incumbent(&values, &mapped, &activity);
        let old_score = support::score(old.lower, &old.a[..k], Some(&activity)).unwrap();
        if k == 3 {
            assert!(old_score.hi.abs() < 1e-12);
        }
        let mut r = SignedTableSupportReport::default();
        let next = lp::choose(&values, &mapped, &activity, old, &mut r, end(), &mut || {
            true
        })
        .unwrap()
        .into_option()
        .unwrap();
        exact_check(&values, &next, k);
        // Independent primal: mass (k-2)/(2(k-1)) on all-one and
        // 1/(2(k-1)) on every singleton. Dual a_j=L=-2/(k-1).
        let optimum = Q::new((k - 2).into(), (k - 1).into());
        assert!(rat(r.score_lower.unwrap()) <= optimum);
        assert!(rat(r.envelope_upper.unwrap()) >= optimum);
        assert!(r.gap_upper.unwrap() < 1e-9, "k={k}, {r:?}");
        assert!(r.score_lower.unwrap() > old_score.hi);
        assert_eq!(r.calls, 1);
        assert!(r.pivots > 0 && r.pivots <= 32);
        assert_eq!(r.checked_masks, 1 << k);
    }
}
#[test]
fn table_lp_projects_missing_bits_and_does_not_invent_missing_marginals() {
    let values = [3., 7., -2., 9., 5., 11., 1., 13.];
    let mapped = [true, false, true];
    let activity = [0.5, 0.9, 0.5];
    let old = Support {
        a: [0.; 8],
        lower: -100.,
    };
    let mut r = SignedTableSupportReport::default();
    let next = lp::choose(&values, &mapped, &activity, old, &mut r, end(), &mut || {
        true
    })
    .unwrap()
    .into_option()
    .unwrap();
    assert_eq!(next.a[1], 0.);
    exact_check(&values, &next, 3);
    assert!(r.score_lower.unwrap() <= 8. && r.envelope_upper.unwrap() >= 8.);
    assert!(r.gap_upper.unwrap() < 1e-10);
    let singleton: Vec<_> = (0usize..8)
        .map(|m| if m.is_power_of_two() { 0. } else { 1. })
        .collect();
    let old = incumbent(&singleton, &mapped, &activity);
    let mut r = SignedTableSupportReport::default();
    let next = lp::choose(
        &singleton,
        &mapped,
        &activity,
        old,
        &mut r,
        end(),
        &mut || true,
    )
    .unwrap()
    .into_option();
    assert!(next.is_none());
    assert!(r.score_lower.unwrap() <= 0. && r.envelope_upper.unwrap() >= 0.);
    assert!(r.gap_upper.unwrap() < 1e-10);
}
#[test]
fn table_lp_corners_constant_tables_and_overflow_refuse_without_false_gap() {
    for s in [[0.; 3], [1.; 3], [0., 0.5, 1.]] {
        let mut r = SignedTableSupportReport::default();
        let old = Support {
            a: [0.; 8],
            lower: -6.,
        };
        let next = lp::choose(&[-3.; 8], &[true; 3], &s, old, &mut r, end(), &mut || true)
            .unwrap()
            .into_option();
        assert!(next.is_none());
        assert!(r.score_lower.unwrap() <= -6. && r.envelope_upper.unwrap() >= -6.);
        assert!(r.gap_upper.unwrap() < 1e-10);
    }
    let mut r = SignedTableSupportReport::default();
    assert!(lp::choose(
        &[f64::MAX; 8],
        &[true; 3],
        &[0.5; 3],
        Support {
            a: [0.; 8],
            lower: 0.
        },
        &mut r,
        end(),
        &mut || true
    )
    .unwrap()
    .into_option()
    .is_none());
    assert_eq!(r.stop_reason, "numerical-refusal");
    assert!(r.envelope_upper.is_none());
}
#[test]
fn table_lp_every_numeric_poll_is_sticky_and_keeps_actual_pivot_count() {
    let values = [1., 0., 0., 1., 0., 1., 1., 1.];
    let old = incumbent(&values, &[true; 3], &[0.5; 3]);
    let mut r = SignedTableSupportReport::default();
    let mut count = 0;
    lp::choose(
        &values,
        &[true; 3],
        &[0.5; 3],
        old,
        &mut r,
        end(),
        &mut || {
            count += 1;
            true
        },
    )
    .unwrap()
    .into_option();
    assert!(count > 30);
    for cutoff in 1..=count {
        let mut r = SignedTableSupportReport::default();
        let mut n = 0;
        let result = lp::choose(
            &values,
            &[true; 3],
            &[0.5; 3],
            old,
            &mut r,
            end(),
            &mut || {
                n += 1;
                n != cutoff
            },
        );
        assert!(result.is_none(), "cutoff={cutoff}/{count}");
        assert_eq!(n, cutoff);
        assert_eq!(r.calls, 1);
        assert!(r.pivots <= 32);
        assert_eq!(r.stop_reason, "cancelled");
    }
}
#[test]
fn table_lp_exact_and_adjacent_admission_keeps_both_earlier_families() {
    let (p, plan) = mixed();
    let full = owner(&p, &plan, true);
    let (mut work, mut bytes) = controls::total(&full);
    let joint = joint::quote(p.input_dim, p.hidden_dim, 2, p.input_dim + 32).unwrap();
    work -= joint.work;
    bytes -= joint.bytes;
    for memory in [false, true] {
        for excess in [0, 1] {
            let cuts = owner_with(&p, &plan, true, |b| {
                let mut c = b.state.lock().unwrap();
                if memory {
                    c.shared_bytes += MAX_BYTES - bytes + excess;
                } else {
                    c.work += MAX_WORK - work + excess;
                }
            });
            let r = cuts
                .signed
                .as_ref()
                .unwrap()
                .stats()
                .component_rows
                .phase
                .cube
                .cell;
            assert!(r.admitted && r.ray.admitted);
            assert_eq!(r.lp.admitted, excess == 0);
            let (w, m) = controls::total(&cuts);
            assert!(w <= MAX_WORK && m <= MAX_BYTES);
        }
    }
    assert!(lp::quote(32, 128, 8, 64).is_some());
    assert!(lp::quote(33, 128, 8, 64).is_none());
    assert!(lp::quote(32, usize::MAX, 8, 64).is_none());
    assert!(lp::quote(32, 128, 9, 64).is_none());
}

#[test]
fn table_lp_zero_pivot_limit_and_bad_mass_still_have_checked_upper_bounds() {
    let values = [1., 0., 0., 1., 0., 1., 1., 1.];
    let mut s = lp::simplex::Simplex::new(&values, &[true; 3], &[0.5; 3], &mut || true).unwrap();
    let mut r = SignedTableSupportReport::default();
    let a = s.run(0, &mut r, &mut || true).unwrap();
    assert_eq!(r.pivots, 0);
    assert_eq!(r.stop_reason, "pivot-limit");
    let lower = repair(&values, &a[..3], end(), &mut || true).unwrap();
    exact_check(&values, &Support { a, lower }, 3);
    for bad in [-1., 1e9, 0.1] {
        s.mass = [bad; 9];
        assert!(lp::upper(&s, &mut || true).unwrap() >= 0.5);
    }
    r.pivots = 33;
    assert!(s.run(32, &mut r, &mut || true).is_none());
    let mut r = SignedTableSupportReport::default();
    let old = Support {
        a: [0.; 8],
        lower: -1.,
    };
    let best = lp::choose(
        &values,
        &[false; 3],
        &[0.5; 3],
        old,
        &mut r,
        end(),
        &mut || true,
    )
    .unwrap()
    .into_option()
    .unwrap();
    assert_eq!(best.a, [0.; 8]);
    assert!(r.score_lower.unwrap() <= 0. && r.envelope_upper.unwrap() >= 0.);
    assert!(r.gap_upper.unwrap() < 1e-10);
}

#[test]
fn table_lp_original_distinct_row_retains_both_families_and_exact_closed_graph() {
    let p = source(
        vec![
            1., 2., 3., 4., 1., -1., -3., -2., 3., 1., 3., -2., -4., 4., 1.,
        ],
        vec![0.375, 0.125, -0.25, 0.375, -0.25],
        &[2., 3., -1., -2., -3.],
        3,
    );
    let plan = Plan {
        components: vec![Component {
            negative: vec![2, 3, 4],
            positive: vec![0, 1],
            positive_units: vec![2, 2],
            g: vec![0.; 3],
            alpha: vec![0.5; 16],
        }],
        leftover: vec![],
    };
    let weights = vec![
        p.w1.clone(),
        vec![2., 3., -1., -2., -3., 0., 0., 0., 0., 0.],
    ];
    let biases = vec![p.b1.clone(), vec![0.; 2]];
    let dims = vec![3, 5, 2];
    let bounds = vec![(0..5)
        .map(|j| {
            let radius: f64 = p.w_row(j).iter().map(|v| v.abs()).sum();
            // These small dyadic endpoints are exactly representable in f32.
            Bound::new((p.b1[j] - radius) as f32, (p.b1[j] + radius) as f32)
        })
        .collect::<Vec<_>>()];
    let original = || {
        let mut e =
            crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.); 3], &bounds)
                .unwrap();
        e.constrain_output_leq_row(0, 1).unwrap();
        e.into_parts()
    };
    let initial = original();
    let encoding = HullMipEncoding {
        weights: &weights,
        biases: &biases,
        dimensions: &dims,
        hidden_bounds: &bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row: Row(initial.problem.num_rows() - 1),
    };
    let heads = binding::head_columns(&encoding, &initial).unwrap();
    let old = advised(&p, &plan);
    old.signed
        .as_ref()
        .unwrap()
        .budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap()
        .lp = None;
    let new = advised(&p, &plan);
    for cuts in [&old, &new] {
        cuts.signed
            .as_ref()
            .unwrap()
            .budget
            .state
            .lock()
            .unwrap()
            .phase_cube
            .admission
            .as_mut()
            .unwrap()
            .quote
            .cell
            .as_mut()
            .unwrap()
            .joint = None;
    }
    let (mut before, mut after) = (original(), original());
    old.append_to(&mut before, &encoding, end()).unwrap();
    let report = new.append_to(&mut after, &encoding, end()).unwrap();
    assert_eq!(before.problem.cols(), after.problem.cols());
    assert_eq!(after.problem.num_rows(), before.problem.num_rows() + 1);
    assert_eq!(
        before.problem.rows(),
        &after.problem.rows()[..before.problem.num_rows()]
    );
    let stats = report.signed.unwrap();
    let cell = &stats.component_rows.phase.cube.cell;
    assert_eq!(cell.prepared_rows, 1);
    assert_eq!(cell.ray.prepared_rows, 1);
    assert_eq!(cell.appended_rows, 1);
    assert_eq!(cell.ray.appended_rows, 1);
    assert_eq!(cell.lp.appended_rows, 1);
    assert_eq!(cell.lp.calls, 1);
    assert!(cell.lp.activity_from_root);
    assert_eq!(cell.lp.checked_masks, 8);
    assert!(cell.lp.gap_upper.unwrap() < 1e-10);
    assert_eq!(stats.source_replays, 1);
    let last = after.problem.rows().last().unwrap();
    let mut inputs = Vec::new();
    for x in [-1., 0., 1.] {
        for y in [-1., 0., 1.] {
            for z in [-1., 0., 1.] {
                inputs.push(vec![rat(x), rat(y), rat(z)]);
            }
        }
    }
    // Each original head's exact zero hyperplane, including nondyadic inputs.
    for j in 0..5 {
        inputs.push(vec![
            -rat(p.b1[j]) / rat(p.w_row(j)[0]),
            Q::zero(),
            Q::zero(),
        ]);
    }
    for input in inputs {
        for ties in 0..32 {
            let mut point = vec![Q::zero(); initial.num_cols];
            for j in 0..3 {
                point[initial.input_vars[j].0] = input[j].clone();
            }
            for j in 0..5 {
                let z = rat(p.b1[j]) + (0..3).map(|i| rat(p.w_row(j)[i]) * &input[i]).sum::<Q>();
                point[3 + j] = z.clone();
                point[heads[j].0] = z.clone().max(Q::zero());
                point[heads[j].0 + 1] = rat(if z.is_zero() {
                    ((ties >> j) & 1) as f64
                } else {
                    usize::from(z > Q::zero()) as f64
                });
            }
            // The original output uses the network coefficients, not the
            // conservatively rounded margin coefficients in p.c.
            for output in 0..2 {
                point[initial.output_vars[output].0] = rat(biases[1][output])
                    + (0..5)
                        .map(|j| rat(weights[1][output * 5 + j]) * &point[heads[j].0])
                        .sum::<Q>();
            }
            for (j, column) in initial.problem.cols().iter().enumerate() {
                assert!(!column.lb.is_finite() || point[j] >= rat(column.lb));
                assert!(!column.ub.is_finite() || point[j] <= rat(column.ub));
            }
            // Bind the constructed point to the actual encoder too. A graph
            // point need not satisfy the unsafe predicate being investigated.
            for (index, row) in initial.problem.rows().iter().enumerate() {
                if index == encoding.unsafe_row.0 {
                    continue;
                }
                let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
                assert!(
                    !row.lb.is_finite() || lhs >= rat(row.lb),
                    "original lower row={index} input={input:?} ties={ties}"
                );
                assert!(
                    !row.ub.is_finite() || lhs <= rat(row.ub),
                    "original upper row={index} input={input:?} ties={ties}"
                );
            }
            let lhs: Q = last.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
            assert!(lhs >= rat(last.lb), "input={input:?} ties={ties}");
        }
    }
}
