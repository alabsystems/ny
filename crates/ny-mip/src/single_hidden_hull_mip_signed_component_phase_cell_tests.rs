// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete original publication and closed-cell soundness controls.
use super::*;
use crate::single_hidden_hull::mip_cuts as root;
use num_rational::BigRational as Q;
use num_traits::Zero;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_controls.rs"]
mod controls;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_input_phase_production_tests.rs"]
mod input_phase_production_tests;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_input_phase_tests.rs"]
mod input_phase_tests;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_joint_tests.rs"]
mod joint_tests;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_lp_search_tests.rs"]
mod lp_search_tests;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_lp_tests.rs"]
mod lp_tests;
#[path = "single_hidden_hull_mip_signed_component_phase_cell_ray_tests.rs"]
mod ray_tests;
fn end() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn rat(v: f64) -> Q {
    Q::from_float(v).unwrap()
}
fn source(weights: Vec<f64>, bias: Vec<f64>, output: &[f64], d: usize) -> HullMarginProblem {
    let mut two_outputs = output.to_vec();
    two_outputs.resize(2 * output.len(), 0.);
    HullMarginProblem::new(
        weights,
        bias,
        d,
        &two_outputs,
        &[0., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap()
}
fn owner<'a>(p: &'a HullMarginProblem, plan: &Plan, cell: bool) -> HullRootMipCuts<'a> {
    owner_with(p, plan, cell, |_| {})
}
fn owner_with<'a>(
    p: &'a HullMarginProblem,
    plan: &Plan,
    cell: bool,
    before: impl FnOnce(&mut Budget),
) -> HullRootMipCuts<'a> {
    let m = HullMultipliers::zeros(p);
    let floor = certify_hull_lower_bound(p, &m).unwrap();
    let mut cuts = root::prepare(p, m, floor, end(), &mut || true).unwrap();
    let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    b.portfolio_phase_enabled = cell;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, plan, &b, b.search_end, &mut || true).unwrap();
    before(&mut b);
    component_rows::admit(&cuts, plan, &b).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    cuts.signed = Some(row(&cuts, plan, checked, cost, &b, &seal));
    cuts
}
fn mixed() -> (HullMarginProblem, Plan) {
    let p = source(
        vec![1., 1. / 16., 1., -1. / 16.],
        vec![0.; 2],
        &[-1., -1.],
        2,
    );
    let plan = Plan {
        components: vec![Component {
            negative: vec![0, 1],
            positive: vec![],
            positive_units: vec![],
            g: vec![0.; 2],
            alpha: vec![],
        }],
        leftover: vec![],
    };
    (p, plan)
}
#[test]
fn cell_full_table_keeps_scalar_bits_and_strictly_strengthens_both_mixed_cells() {
    let (p, plan) = mixed();
    let old = owner(&p, &plan, false);
    let new = owner(&p, &plan, true);
    let before = old
        .signed
        .as_ref()
        .unwrap()
        .replay(&old, end(), &mut || true)
        .unwrap();
    let after = new
        .signed
        .as_ref()
        .unwrap()
        .replay(&new, end(), &mut || true)
        .unwrap();
    assert_eq!(before.constant.to_bits(), after.constant.to_bits());
    assert_eq!(before.root_lower.to_bits(), after.root_lower.to_bits());
    assert_eq!(before.g, after.g);
    let old_table = before.cube.unwrap();
    let table = after.cube.unwrap();
    assert_eq!(old_table.values, table.values);
    let strong = table.cell.unwrap();
    assert_eq!(strong.values.len(), 4);
    for (actual, expected) in strong.values.iter().zip([0., -1. / 8., -1. / 8., -2.]) {
        assert!(*actual <= expected && *actual >= expected - 1e-12);
    }
    let stats = new.signed.as_ref().unwrap().stats();
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.checked_masks, 8);
    assert_eq!(stats.component_rows.phase.cube.cell.improved_masks, 2);
}
#[test]
fn cell_joint_block_escapes_fixed_alpha_stall_and_rejects_biased_decoy() {
    for decoy_bias in [-2., -0.9] {
        let p = source(
            vec![1., 1., 1.],
            vec![0., decoy_bias, -0.5],
            &[2., 3., -1.],
            1,
        );
        let plan = Plan {
            components: vec![Component {
                negative: vec![2],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.],
                alpha: vec![0., 0., 0.5, 0.],
            }],
            leftover: vec![],
        };
        let cuts = owner(&p, &plan, true);
        let replay = cuts
            .signed
            .as_ref()
            .unwrap()
            .replay(&cuts, end(), &mut || true)
            .unwrap();
        let cube = replay.cube.unwrap();
        assert!((cube.values[0][1] - 0.5).abs() < 1e-12);
        let strong = cube.cell.unwrap();
        assert!(strong.values[1] > 0.999999999 && strong.values[1] <= 1.);
        assert_eq!(
            cuts.signed
                .as_ref()
                .unwrap()
                .stats()
                .component_rows
                .phase
                .cube
                .cell
                .joint_improved_masks,
            1
        );
        // Original negative or closed-zero witnesses stay above every cell value.
        for x in [-1., 0., 0.5, 0.75, 1.] {
            for side in 0..2 {
                if (side == 0 && x > 0.5) || (side == 1 && x < 0.5) {
                    continue;
                }
                let exact = rat(2.) * rat(x).max(Q::zero())
                    + rat(3.) * (rat(x) + rat(decoy_bias)).max(Q::zero())
                    - (rat(x) - rat(0.5)).max(Q::zero());
                assert!(exact >= rat(strong.values[side]));
            }
        }
    }
}
fn feasible(parts: &MipParts, point: &[Q]) -> bool {
    parts.problem.cols().iter().zip(point).all(|(col, value)| {
        (!col.lb.is_finite() || *value >= rat(col.lb))
            && (!col.ub.is_finite() || *value <= rat(col.ub))
    }) && parts.problem.rows().iter().all(|row| {
        let value: Q = row
            .coeffs
            .iter()
            .map(|(col, a)| rat(*a) * &point[*col])
            .sum();
        (!row.lb.is_finite() || value >= rat(row.lb))
            && (!row.ub.is_finite() || value <= rat(row.ub))
    })
}
#[test]
fn cell_original_model_append_excludes_old_fractional_point_and_keeps_closed_graph() {
    let (p, plan) = mixed();
    let weights = vec![p.w1.clone(), vec![-1., -1., 0., 0.]];
    let biases = vec![p.b1.clone(), vec![0., 0.]];
    let dims = vec![2, 2, 2];
    let bounds = vec![vec![Bound::new(-17. / 16., 17. / 16.); 2]];
    let original = || {
        let mut encoder =
            crate::encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1., 1.); 2], &bounds)
                .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
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
    let old = owner(&p, &plan, false);
    old.append_to(&mut old_parts, &encoding, end()).unwrap();
    let mut new_parts = original();
    let new = owner(&p, &plan, true);
    let report = new.append_to(&mut new_parts, &encoding, end()).unwrap();
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
    for j in 0..2 {
        fake[2 + j] = rat(0.25);
    }
    fake[heads[0].0] = rat(0.5);
    fake[heads[1].0] = rat(0.25);
    fake[heads[0].0 + 1] = rat(0.75);
    fake[heads[1].0 + 1] = rat(0.25);
    fake[initial.output_vars[0].0] = rat(-0.75);
    assert!(feasible(&old_parts, &fake));
    assert!(!feasible(&new_parts, &fake));
    assert_eq!(
        report
            .signed
            .unwrap()
            .component_rows
            .phase
            .cube
            .cell
            .appended_rows,
        1
    );
    for x in [-1., 0., 1.] {
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
                point[initial.output_vars[0].0] = -&point[heads[0].0] - &point[heads[1].0];
                assert!(feasible(&new_parts, &point));
            }
        }
    }
}
