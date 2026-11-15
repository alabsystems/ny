// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-encoder separation, complete-table support and refusal controls.
use super::*;

fn legacy<'a>(p: &'a HullMarginProblem, plan: &Plan) -> HullRootMipCuts<'a> {
    let cuts = owner(p, plan, true);
    let mut counts = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
    assert_eq!(counts.replay_attempts, 0);
    assert!(!counts.phase_cube.cell_attempted);
    counts
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap()
        .joint = None;
    counts
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap()
        .lp = None;
    assert!(counts
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap()
        .ray
        .take()
        .is_some());
    drop(counts);
    cuts
}
fn asymmetric() -> (HullMarginProblem, Plan) {
    let (_, plan) = mixed();
    (
        source(
            vec![1., 1. / 16., 1., -1. / 16.],
            vec![0.; 2],
            &[-1., -4.],
            2,
        ),
        plan,
    )
}

#[test]
fn ray_full_table_preserves_current_cell_and_checks_unclamped_beta() {
    let (p, plan) = asymmetric();
    let old = legacy(&p, &plan);
    let new = owner(&p, &plan, true);
    let a = old
        .signed
        .as_ref()
        .unwrap()
        .replay(&old, end(), &mut || true)
        .unwrap();
    let b = new
        .signed
        .as_ref()
        .unwrap()
        .replay(&new, end(), &mut || true)
        .unwrap();
    assert_eq!(a.constant.to_bits(), b.constant.to_bits());
    assert_eq!(a.root_lower.to_bits(), b.root_lower.to_bits());
    assert_eq!(a.g, b.g);
    let (a, b) = (a.cube.unwrap(), b.cube.unwrap());
    assert_eq!(a.values, b.values);
    let (a, b) = (a.cell.unwrap(), b.cell.unwrap());
    assert_eq!(a.values, b.values);
    for (actual, exact) in b
        .values
        .iter()
        .zip([0., -17. / 16., -53. / 16., -83. / 16.])
    {
        assert!(*actual <= exact && *actual > exact - 1e-11);
    }
    for (actual, exact) in b
        .ray
        .unwrap()
        .iter()
        .zip([0., -1. / 8., -1. / 2., -83. / 16.])
    {
        assert!(*actual <= exact && *actual > exact - 1e-11);
    }
    let r = new.signed.as_ref().unwrap().stats();
    assert_eq!(r.source_replays, 1);
    assert_eq!(r.checked_masks, 8);
    assert_eq!(r.component_rows.phase.cube.cell.ray.improved_masks, 2);
}

#[test]
fn ray_original_encoder_separates_entire_current_family_and_keeps_zero_ties() {
    let (p, plan) = asymmetric();
    let weights = vec![p.w1.clone(), vec![-1., -4., 0., 0.]];
    let biases = vec![p.b1.clone(), vec![0., 0.]];
    let dims = vec![2, 2, 2];
    let bounds = vec![vec![Bound::new(-17. / 16., 17. / 16.); 2]];
    let original = || {
        let mut e =
            crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.); 2], &bounds)
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
    let mut old_parts = original();
    legacy(&p, &plan)
        .append_to(&mut old_parts, &encoding, end())
        .unwrap();
    let mut new_parts = original();
    let new = owner(&p, &plan, true);
    let report = new.append_to(&mut new_parts, &encoding, end()).unwrap();
    assert_eq!(new_parts.problem.cols(), old_parts.problem.cols());
    assert_eq!(
        new_parts.problem.num_rows(),
        old_parts.problem.num_rows() + 1
    );
    for (old, new) in old_parts
        .problem
        .rows()
        .iter()
        .zip(new_parts.problem.rows())
    {
        assert_eq!(old, new);
    }
    let mut fake = vec![Q::zero(); initial.num_cols];
    fake[initial.input_vars[0].0] = rat(0.25);
    fake[2] = rat(0.25);
    fake[3] = rat(0.25);
    fake[heads[0].0] = rat(0.5);
    fake[heads[1].0] = rat(0.25);
    fake[heads[0].0 + 1] = rat(0.75);
    fake[heads[1].0 + 1] = rat(0.25);
    fake[initial.output_vars[0].0] = rat(-1.5);
    assert!(feasible(&old_parts, &fake));
    assert!(!feasible(&new_parts, &fake));
    let last = new_parts.problem.rows().last().unwrap();
    let lhs: Q = last.coeffs.iter().map(|(j, a)| rat(*a) * &fake[*j]).sum();
    assert!(rat(last.lb) - lhs > rat(0.25));
    let r = report.signed.unwrap();
    assert_eq!(r.source_replays, 1);
    assert_eq!(r.component_rows.phase.cube.cell.ray.appended_rows, 1);
    for x in [-1., -0.25, 0., 0.25, 1.] {
        for t in [-1., 0., 1.] {
            for ties in 0..4 {
                let mut point = vec![Q::zero(); initial.num_cols];
                point[initial.input_vars[0].0] = rat(x);
                point[initial.input_vars[1].0] = rat(t);
                for j in 0..2 {
                    let z = rat(x) + rat(if j == 0 { 1. / 16. } else { -1. / 16. }) * rat(t);
                    point[2 + j] = z.clone();
                    point[heads[j].0] = z.clone().max(Q::zero());
                    point[heads[j].0 + 1] = rat(if z.is_zero() {
                        ((ties >> j) & 1) as f64
                    } else {
                        usize::from(z > Q::zero()) as f64
                    });
                }
                point[initial.output_vars[0].0] =
                    -&point[heads[0].0] - rat(4.) * &point[heads[1].0];
                assert!(feasible(&new_parts, &point));
            }
        }
    }
}

#[test]
fn ray_support_uses_off_chain_improvement_and_checks_every_mask() {
    let old = [0., 0., -1., 0., -1., -1., -1., 0.];
    let new = [0., 0., -1., 0., -1., 0., -1., 0.];
    let activity = [0.5; 3];
    let weights = support_weights(&new, &[0, 1, 2], Some(&activity)).unwrap();
    assert_eq!(weights, [0.; 8]);
    let lower = repair(&new, &[0.; 3], end(), &mut || true).unwrap();
    assert_eq!(lower, repair(&old, &[0.; 3], end(), &mut || true).unwrap());
    let (a, bound) = support::choose(
        &new,
        [0.; 8],
        lower,
        &[true; 3],
        Some(&activity),
        end(),
        &mut || true,
    )
    .unwrap();
    let value = bound - a[..3].iter().sum::<f64>() * 0.5;
    assert!(value > -1.01 && value <= -1.);
    for mask in 0..8 {
        let lhs = rat(2.) * rat(new[mask])
            + (0..3)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| rat(a[i]))
                .sum::<Q>();
        assert!(lhs >= rat(bound));
    }
    let (a, _) = support::choose(
        &new,
        [0.; 8],
        lower,
        &[false, true, true],
        Some(&activity),
        end(),
        &mut || true,
    )
    .unwrap();
    assert_eq!(a[0], 0.);
}

#[test]
fn ray_numeric_poll_failure_is_sticky_and_overflow_retains_old_bound() {
    let (p, plan) = asymmetric();
    let q = [
        certificate::I::point(-4.).unwrap(),
        certificate::I::point(0.25).unwrap(),
    ];
    let zero = certificate::I::point(0.).unwrap();
    let mut count = 0;
    let result = ray::strengthen(
        &p,
        &plan.components[0],
        2,
        &q,
        zero,
        -53. / 16.,
        end(),
        &mut || {
            count += 1;
            true
        },
    )
    .unwrap();
    assert!(result > -0.50000001 && result <= -0.5);
    for cutoff in 1..=count {
        let mut n = 0;
        assert!(ray::strengthen(
            &p,
            &plan.components[0],
            2,
            &q,
            zero,
            -53. / 16.,
            end(),
            &mut || {
                n += 1;
                n != cutoff
            }
        )
        .is_none());
    }
    let mut huge = p;
    huge.b1[0] = f64::MAX;
    huge.c[0] = -f64::MAX;
    assert_eq!(
        ray::strengthen(
            &huge,
            &plan.components[0],
            2,
            &q,
            zero,
            -4.,
            end(),
            &mut || true
        ),
        Some(-4.)
    );
}

#[test]
fn ray_exact_and_adjacent_admission_preserves_old_cell() {
    let (p, plan) = asymmetric();
    let full = owner(&p, &plan, true);
    let (mut work, mut bytes) = controls::total(&full);
    let lp = full
        .signed
        .as_ref()
        .unwrap()
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
        .unwrap()
        .lp
        .unwrap();
    let joint = joint::quote(p.input_dim, p.hidden_dim, 2, p.input_dim + 32).unwrap();
    work -= joint.work;
    bytes -= joint.bytes;
    work -= lp.work;
    bytes -= lp.bytes;
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
                .cube;
            assert!(r.admitted && r.cell.admitted);
            assert_eq!(r.cell.ray.admitted, excess == 0);
            let (w, m) = controls::total(&cuts);
            assert!(w <= MAX_WORK && m <= MAX_BYTES);
        }
    }
    assert!(ray::quote(32, 128, 8, 64).is_some());
    assert!(ray::quote(33, 128, 8, 65).is_none());
}
