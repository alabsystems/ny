// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Small-random soundness checks for Clip-and-Verify algorithms.
//!
//! These proptests use exact 2D feasible-polytope enumeration to verify that:
//! - `relaxed_clip` never clips away feasible points.
//! - `complete_clip` returns sound dual bounds and never worsens the box baseline.

use crate::complete_clip::complete_clip;
use crate::relaxed_clip::relaxed_clip;
use ndarray::array;
use proptest::prelude::*;

use super::FP_TOLERANCE;

const CLIP_TOLERANCE: f32 = 5.0 * FP_TOLERANCE;
const DUAL_TOLERANCE: f32 = 1e-2;

#[derive(Debug, Clone)]
struct SmallClipCase {
    x_l: [f32; 2],
    x_u: [f32; 2],
    witness: [f32; 2],
    constraints: [[f32; 3]; 3],
    objective: [f32; 2],
}

fn small_clip_case() -> impl Strategy<Value = SmallClipCase> {
    (
        -2.0f32..2.0,
        -2.0f32..2.0,
        0.1f32..2.0,
        0.1f32..2.0,
        0.1f32..2.0,
        0.1f32..2.0,
        prop::collection::vec(-2.0f32..2.0, 6),
        prop::collection::vec(0.0f32..2.0, 3),
        prop::collection::vec(-2.0f32..2.0, 2),
    )
        .prop_map(
            |(
                witness_x1,
                witness_x2,
                margin_l1,
                margin_u1,
                margin_l2,
                margin_u2,
                coeffs,
                slacks,
                objective,
            )| {
                let witness = [witness_x1, witness_x2];
                let x_l = [witness_x1 - margin_l1, witness_x2 - margin_l2];
                let x_u = [witness_x1 + margin_u1, witness_x2 + margin_u2];
                let constraints = std::array::from_fn(|idx| {
                    let a1 = coeffs[2 * idx];
                    let a2 = coeffs[2 * idx + 1];
                    // By construction, the witness satisfies a1*x1 + a2*x2 + b <= 0.
                    let b = -(a1 * witness_x1 + a2 * witness_x2) - slacks[idx];
                    [a1, a2, b]
                });

                SmallClipCase {
                    x_l,
                    x_u,
                    witness,
                    constraints,
                    objective: [objective[0], objective[1]],
                }
            },
        )
}

fn enumerate_feasible_vertices_2d(
    x_l: [f32; 2],
    x_u: [f32; 2],
    constraints: &[[f32; 3]],
    witness: [f32; 2],
) -> Vec<[f32; 2]> {
    let mut all_constraints = constraints.to_vec();
    all_constraints.push([1.0, 0.0, -x_u[0]]);
    all_constraints.push([-1.0, 0.0, x_l[0]]);
    all_constraints.push([0.0, 1.0, -x_u[1]]);
    all_constraints.push([0.0, -1.0, x_l[1]]);

    let mut candidates = vec![
        witness,
        [x_l[0], x_l[1]],
        [x_l[0], x_u[1]],
        [x_u[0], x_l[1]],
        [x_u[0], x_u[1]],
    ];

    let eps = 1e-6_f32;
    for i in 0..all_constraints.len() {
        for j in (i + 1)..all_constraints.len() {
            let [a1, a2, b1] = all_constraints[i];
            let [c1, c2, b2] = all_constraints[j];
            let det = a1 * c2 - a2 * c1;
            if det.abs() <= eps {
                continue;
            }
            let x1 = (-b1 * c2 + b2 * a2) / det;
            let x2 = (-a1 * b2 + c1 * b1) / det;
            candidates.push([x1, x2]);
        }
    }

    let mut feasible = Vec::new();
    'candidate: for [x1, x2] in candidates {
        for [a1, a2, b] in &all_constraints {
            if a1 * x1 + a2 * x2 + b > eps {
                continue 'candidate;
            }
        }
        feasible.push([x1, x2]);
    }

    feasible
}

fn objective_value(objective: [f32; 2], point: [f32; 2]) -> f32 {
    objective[0] * point[0] + objective[1] * point[1]
}

fn box_corners(x_l: [f32; 2], x_u: [f32; 2]) -> [[f32; 2]; 4] {
    [
        [x_l[0], x_l[1]],
        [x_l[0], x_u[1]],
        [x_u[0], x_l[1]],
        [x_u[0], x_u[1]],
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(150) })]

    /// Relaxed clipping must overapproximate the feasible polytope defined by the
    /// original box and linear constraints. Every feasible vertex should remain
    /// inside the clipped box.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_relaxed_clip_preserves_feasible_vertices_348(
        case in small_clip_case(),
        iterations in 1usize..4,
    ) {
        let x_l = array![[case.x_l[0], case.x_l[1]]].into_dyn();
        let x_u = array![[case.x_u[0], case.x_u[1]]].into_dyn();
        let l_a = array![[
            [case.constraints[0][0], case.constraints[0][1]],
            [case.constraints[1][0], case.constraints[1][1]],
            [case.constraints[2][0], case.constraints[2][1]],
        ]]
        .into_dyn();
        let lbias = array![[case.constraints[0][2], case.constraints[1][2], case.constraints[2][2]]]
            .into_dyn();
        let thresholds = array![[0.0_f32, 0.0, 0.0]].into_dyn();

        let feasible_vertices = enumerate_feasible_vertices_2d(
            case.x_l,
            case.x_u,
            &case.constraints,
            case.witness,
        );
        prop_assert!(
            !feasible_vertices.is_empty(),
            "expected at least one feasible point: {case:?}"
        );

        let (new_l, new_u) =
            relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, iterations, true)
                .expect("relaxed_clip should accept feasible finite inputs");

        prop_assert!(
            new_l[[0, 0]] >= case.x_l[0] - CLIP_TOLERANCE
                && new_l[[0, 1]] >= case.x_l[1] - CLIP_TOLERANCE,
            "relaxed_clip widened lower bounds: new_l={new_l:?}, case={case:?}"
        );
        prop_assert!(
            new_u[[0, 0]] <= case.x_u[0] + CLIP_TOLERANCE
                && new_u[[0, 1]] <= case.x_u[1] + CLIP_TOLERANCE,
            "relaxed_clip widened upper bounds: new_u={new_u:?}, case={case:?}"
        );
        prop_assert!(
            new_l[[0, 0]] <= new_u[[0, 0]] + CLIP_TOLERANCE
                && new_l[[0, 1]] <= new_u[[0, 1]] + CLIP_TOLERANCE,
            "relaxed_clip returned inverted bounds: new_l={new_l:?}, new_u={new_u:?}, case={case:?}"
        );

        for vertex in feasible_vertices {
            prop_assert!(
                new_l[[0, 0]] - CLIP_TOLERANCE <= vertex[0]
                    && vertex[0] <= new_u[[0, 0]] + CLIP_TOLERANCE
                    && new_l[[0, 1]] - CLIP_TOLERANCE <= vertex[1]
                    && vertex[1] <= new_u[[0, 1]] + CLIP_TOLERANCE,
                "relaxed_clip removed feasible vertex {vertex:?}: new_l={new_l:?}, new_u={new_u:?}, case={case:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(150) })]

    /// Complete clipping should return a sound dual lower/upper bound relative to
    /// the exact 2D feasible polytope, while never degrading beyond the box-only
    /// baseline.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_complete_clip_sound_against_exact_vertices_348(
        case in small_clip_case(),
        rearrange in any::<bool>(),
        iterations in 1usize..4,
    ) {
        let x_l = array![[case.x_l[0], case.x_l[1]]].into_dyn();
        let x_u = array![[case.x_u[0], case.x_u[1]]].into_dyn();
        let objective = array![[[case.objective[0], case.objective[1]]]].into_dyn();
        let a_matrix = array![[
            [case.constraints[0][0], case.constraints[0][1]],
            [case.constraints[1][0], case.constraints[1][1]],
            [case.constraints[2][0], case.constraints[2][1]],
        ]]
        .into_dyn();
        let b_vector = array![[case.constraints[0][2], case.constraints[1][2], case.constraints[2][2]]]
            .into_dyn();

        let feasible_vertices = enumerate_feasible_vertices_2d(
            case.x_l,
            case.x_u,
            &case.constraints,
            case.witness,
        );
        prop_assert!(
            !feasible_vertices.is_empty(),
            "expected at least one feasible point: {case:?}"
        );

        let exact_min = feasible_vertices
            .iter()
            .map(|&point| objective_value(case.objective, point))
            .fold(f32::INFINITY, f32::min);
        let exact_max = feasible_vertices
            .iter()
            .map(|&point| objective_value(case.objective, point))
            .fold(f32::NEG_INFINITY, f32::max);

        let box_corners = box_corners(case.x_l, case.x_u);
        let baseline_min = box_corners
            .iter()
            .map(|&point| objective_value(case.objective, point))
            .fold(f32::INFINITY, f32::min);
        let baseline_max = box_corners
            .iter()
            .map(|&point| objective_value(case.objective, point))
            .fold(f32::NEG_INFINITY, f32::max);

        let min_bound = complete_clip(
            &x_l,
            &x_u,
            &objective,
            &a_matrix,
            &b_vector,
            -1.0,
            rearrange,
            iterations,
        )
        .expect("complete_clip min bound should accept feasible finite inputs");
        let max_bound = complete_clip(
            &x_l,
            &x_u,
            &objective,
            &a_matrix,
            &b_vector,
            1.0,
            rearrange,
            iterations,
        )
        .expect("complete_clip max bound should accept feasible finite inputs");

        prop_assert!(
            min_bound[[0, 0]] >= baseline_min - DUAL_TOLERANCE,
            "complete_clip min worsened box baseline: bound={}, baseline_min={}, case={case:?}",
            min_bound[[0, 0]],
            baseline_min,
        );
        prop_assert!(
            min_bound[[0, 0]] <= exact_min + DUAL_TOLERANCE,
            "complete_clip min is unsound: bound={}, exact_min={}, case={case:?}",
            min_bound[[0, 0]],
            exact_min,
        );
        prop_assert!(
            max_bound[[0, 0]] <= baseline_max + DUAL_TOLERANCE,
            "complete_clip max worsened box baseline: bound={}, baseline_max={}, case={case:?}",
            max_bound[[0, 0]],
            baseline_max,
        );
        prop_assert!(
            max_bound[[0, 0]] >= exact_max - DUAL_TOLERANCE,
            "complete_clip max is unsound: bound={}, exact_max={}, case={case:?}",
            max_bound[[0, 0]],
            exact_max,
        );
    }
}
