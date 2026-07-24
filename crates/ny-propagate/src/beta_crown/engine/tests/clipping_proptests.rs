// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;
use crate::complete_clip::complete_clip;
use crate::relaxed_clip::relaxed_clip;
use proptest::prelude::*;

const VERTEX_TOL: f32 = 1e-4;

fn nonzero_coeff() -> impl Strategy<Value = f32> {
    prop_oneof![-3.0f32..-0.25, 0.25f32..3.0]
}

fn linear_range_2d(lower: [f32; 2], upper: [f32; 2], coeffs: [f32; 2], bias: f32) -> (f32, f32) {
    let mut min = bias;
    let mut max = bias;
    for dim in 0..2 {
        if coeffs[dim] >= 0.0 {
            min += coeffs[dim] * lower[dim];
            max += coeffs[dim] * upper[dim];
        } else {
            min += coeffs[dim] * upper[dim];
            max += coeffs[dim] * lower[dim];
        }
    }
    (min, max)
}

fn dedup_points(points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    let mut unique: Vec<[f32; 2]> = Vec::new();
    'candidate: for point in points {
        for existing in &unique {
            if (point[0] - existing[0]).abs() <= VERTEX_TOL
                && (point[1] - existing[1]).abs() <= VERTEX_TOL
            {
                continue 'candidate;
            }
        }
        unique.push(point);
    }
    unique
}

fn enumerate_polytope_vertices_2d(
    lower: [f32; 2],
    upper: [f32; 2],
    constraints: &[[f32; 3]],
) -> Vec<[f32; 2]> {
    let mut all_constraints = constraints.to_vec();
    all_constraints.push([1.0, 0.0, -upper[0]]);
    all_constraints.push([-1.0, 0.0, lower[0]]);
    all_constraints.push([0.0, 1.0, -upper[1]]);
    all_constraints.push([0.0, -1.0, lower[1]]);

    let mut candidates = vec![
        [lower[0], lower[1]],
        [lower[0], upper[1]],
        [upper[0], lower[1]],
        [upper[0], upper[1]],
    ];

    for first in 0..all_constraints.len() {
        for second in (first + 1)..all_constraints.len() {
            let [a1, a2, b1] = all_constraints[first];
            let [c1, c2, b2] = all_constraints[second];
            let det = a1 * c2 - a2 * c1;
            if det.abs() <= VERTEX_TOL {
                continue;
            }
            let x0 = (-b1 * c2 + b2 * a2) / det;
            let x1 = (-a1 * b2 + c1 * b1) / det;
            candidates.push([x0, x1]);
        }
    }

    let mut feasible = Vec::new();
    'candidate: for [x0, x1] in candidates {
        for [a0, a1, b] in &all_constraints {
            if a0 * x0 + a1 * x1 + b > VERTEX_TOL {
                continue 'candidate;
            }
        }
        feasible.push([x0, x1]);
    }

    dedup_points(feasible)
}

fn objective_value(point: [f32; 2], coeffs: [f32; 2]) -> f32 {
    coeffs[0] * point[0] + coeffs[1] * point[1]
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[test]
    fn proptest_relaxed_clipping_child_preserves_feasible_polytope(
        x0_lower in -2.0f32..1.0,
        x0_width in 0.5f32..2.5,
        x1_lower in -2.0f32..1.0,
        x1_width in 0.5f32..2.5,
        w0 in nonzero_coeff(),
        w1 in nonzero_coeff(),
        bias in -1.5f32..1.5,
        split_dim in 0usize..2,
        take_left in any::<bool>(),
        threshold_mix in 0.15f32..0.85,
    ) {
        let parent_lower = [x0_lower, x1_lower];
        let parent_upper = [x0_lower + x0_width, x1_lower + x1_width];

        let mut child_lower = parent_lower;
        let mut child_upper = parent_upper;
        let midpoint = f32::midpoint(parent_lower[split_dim], parent_upper[split_dim]);
        if take_left {
            child_upper[split_dim] = midpoint;
        } else {
            child_lower[split_dim] = midpoint;
        }

        let coeffs = [w0, w1];
        let (child_min, child_max) = linear_range_2d(child_lower, child_upper, coeffs, bias);
        prop_assume!(child_max - child_min > 0.2);

        let threshold = child_min + threshold_mix * (child_max - child_min);
        let feasible_vertices =
            enumerate_polytope_vertices_2d(child_lower, child_upper, &[[w0, w1, bias - threshold]]);
        prop_assert!(
            !feasible_vertices.is_empty(),
            "constructed child must keep a non-empty feasible set"
        );

        let linear = LinearLayer::new(arr2(&[[w0, w1]]), Some(arr1(&[bias]))).unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));

        let original_input = BoundedTensor::new(
            arr1(&parent_lower).into_dyn(),
            arr1(&parent_upper).into_dyn(),
        ).unwrap();

        let (parent_min, parent_max) = linear_range_2d(parent_lower, parent_upper, coeffs, bias);
        let parent = BabDomain {
            history: SplitHistory::new(),
            lower_bound: parent_min,
            upper_bound: parent_max,
            priority: parent_min,
            layer_bounds: vec![Arc::new(original_input.clone())],
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: Some(Arc::new(original_input.clone())),
            input_split_count: 0,
            intermediate_bounds: IntermediateLinearBounds::empty(),
        };

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            relaxed_clip_iterations: 1,
            branching_heuristic: BranchingHeuristic::InputSplit,
            max_domains: 8,
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        let child = verifier
            .create_input_split_child(
                &network,
                &original_input,
                &parent,
                split_dim,
                child_lower[split_dim],
                child_upper[split_dim],
                threshold,
                None,
                None,
            )
            .unwrap();

        prop_assert!(
            child.is_some(),
            "relaxed clipping must not verify away a child with exact feasible vertices remaining"
        );

        let child = child.unwrap();
        let clipped = child.input_bounds.as_ref().unwrap().flatten();

        for dim in 0..2 {
            prop_assert!(
                clipped.lower()[[dim]] >= child_lower[dim] - VERTEX_TOL,
                "clipping widened lower dim {dim}: {} < {}",
                clipped.lower()[[dim]],
                child_lower[dim]
            );
            prop_assert!(
                clipped.upper()[[dim]] <= child_upper[dim] + VERTEX_TOL,
                "clipping widened upper dim {dim}: {} > {}",
                clipped.upper()[[dim]],
                child_upper[dim]
            );
        }

        // Per-dimension retention tolerance (soundness check, task #19).
        //
        // `enumerate_polytope_vertices_2d` admits a vertex when it violates the
        // constraint `w·x + (bias - threshold) <= 0` by at most VERTEX_TOL in
        // CONSTRAINT-value space. The relaxed clip operates in constraint space
        // with slope `w[dim]`, so it can SOUNDLY exclude such a near-boundary
        // (genuinely infeasible) vertex by up to `constraint_slack / |w[dim]|` in
        // X space: a constraint-space slack of s maps to an x-space distance of
        // s/|w[dim]| (proof: constraint_val(p) = w[dim]·(p[dim] - x*[dim]) + (>=0),
        // so |p[dim] - x*[dim]| <= constraint_val(p)/|w[dim]| <= VERTEX_TOL/|w[dim]|).
        // When |w[dim]| < 1 this amplifies the 1e-4 constraint slack past a flat
        // 1e-4 x-space tolerance, which is exactly the false-failure the persisted
        // seed (w0=-0.389) exhibited. The extra VERTEX_TOL absorbs f32 vertex-
        // computation noise. This is NOT slack for an unsound clip: a TRULY
        // feasible (violating) vertex has constraint_val <= 0 and is retained.
        let retain_tol = [
            VERTEX_TOL + VERTEX_TOL / w0.abs(),
            VERTEX_TOL + VERTEX_TOL / w1.abs(),
        ];
        for [x0, x1] in feasible_vertices {
            prop_assert!(
                x0 >= clipped.lower()[[0]] - retain_tol[0]
                    && x0 <= clipped.upper()[[0]] + retain_tol[0],
                "feasible vertex x0={} escaped clipped bounds [{}, {}] (tol {})",
                x0,
                clipped.lower()[[0]],
                clipped.upper()[[0]],
                retain_tol[0]
            );
            prop_assert!(
                x1 >= clipped.lower()[[1]] - retain_tol[1]
                    && x1 <= clipped.upper()[[1]] + retain_tol[1],
                "feasible vertex x1={} escaped clipped bounds [{}, {}] (tol {})",
                x1,
                clipped.lower()[[1]],
                clipped.upper()[[1]],
                retain_tol[1]
            );
        }
    }

    #[test]
    fn proptest_complete_clip_bounds_remain_sound_against_exact_vertices(
        x0_lower in -2.0f32..1.0,
        x0_width in 0.5f32..2.5,
        x1_lower in -2.0f32..1.0,
        x1_width in 0.5f32..2.5,
        obj0 in nonzero_coeff(),
        obj1 in nonzero_coeff(),
        c10 in nonzero_coeff(),
        c11 in nonzero_coeff(),
        c20 in nonzero_coeff(),
        c21 in nonzero_coeff(),
        anchor_mix0 in 0.1f32..0.9,
        anchor_mix1 in 0.1f32..0.9,
        margin1 in 0.0f32..0.5,
        margin2 in 0.0f32..0.5,
        rearrange in any::<bool>(),
    ) {
        let lower = [x0_lower, x1_lower];
        let upper = [x0_lower + x0_width, x1_lower + x1_width];
        let anchor = [
            lower[0] + anchor_mix0 * (upper[0] - lower[0]),
            lower[1] + anchor_mix1 * (upper[1] - lower[1]),
        ];

        let constraints = [
            [c10, c11, -(c10 * anchor[0] + c11 * anchor[1]) - margin1],
            [c20, c21, -(c20 * anchor[0] + c21 * anchor[1]) - margin2],
        ];
        let feasible_vertices = enumerate_polytope_vertices_2d(lower, upper, &constraints);
        prop_assert!(
            !feasible_vertices.is_empty(),
            "constructed complete-clip LP must stay feasible"
        );

        let x_l = arr2(&[[lower[0], lower[1]]]).into_dyn();
        let x_u = arr2(&[[upper[0], upper[1]]]).into_dyn();
        let objective = arr3(&[[[obj0, obj1]]]).into_dyn();
        let a_matrix = arr3(&[[[c10, c11], [c20, c21]]]).into_dyn();
        let b_vector = arr2(&[[constraints[0][2], constraints[1][2]]]).into_dyn();

        let lower_result =
            complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, rearrange, 1)
                .unwrap();
        let upper_result =
            complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, 1.0, rearrange, 1)
                .unwrap();

        let exact_min = feasible_vertices
            .iter()
            .map(|&point| objective_value(point, [obj0, obj1]))
            .fold(f32::INFINITY, f32::min);
        let exact_max = feasible_vertices
            .iter()
            .map(|&point| objective_value(point, [obj0, obj1]))
            .fold(f32::NEG_INFINITY, f32::max);

        let lower_bound = lower_result[[0, 0]];
        let upper_bound = upper_result[[0, 0]];

        prop_assert!(lower_bound.is_finite(), "lower bound must be finite");
        prop_assert!(upper_bound.is_finite(), "upper bound must be finite");
        prop_assert!(
            lower_bound <= exact_min + 1e-3,
            "complete_clip lower bound {} exceeded exact min {}",
            lower_bound,
            exact_min
        );
        prop_assert!(
            upper_bound >= exact_max - 1e-3,
            "complete_clip upper bound {} fell below exact max {}",
            upper_bound,
            exact_max
        );
        prop_assert!(
            lower_bound <= upper_bound + 1e-3,
            "complete_clip returned inverted bounds: {} > {}",
            lower_bound,
            upper_bound
        );
    }

    /// AC 4 (#3876): Inverted input bounds must produce InvalidSpec, not silent
    /// corruption. Generate arbitrary inverted boxes (x_l > x_u) and verify both
    /// complete_clip and relaxed_clip reject them.
    #[test]
    fn proptest_inverted_input_bounds_return_error_3876(
        center in -3.0f32..3.0,
        inversion in 0.01f32..2.0,
        obj in nonzero_coeff(),
        c0 in nonzero_coeff(),
        bias in -1.0f32..1.0,
    ) {
        // x_l > x_u by `inversion` amount
        let x_l = arr2(&[[center + inversion]]).into_dyn();
        let x_u = arr2(&[[center]]).into_dyn();
        let objective = arr3(&[[[obj]]]).into_dyn();
        let a_matrix = arr3(&[[[c0]]]).into_dyn();
        let b_vector = arr2(&[[bias]]).into_dyn();

        let complete_result =
            complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1);
        prop_assert!(
            complete_result.is_err(),
            "complete_clip must reject inverted bounds (x_l={}, x_u={}), got Ok",
            center + inversion,
            center,
        );

        let l_a = arr3(&[[[obj]]]).into_dyn();
        let lbias = arr2(&[[bias]]).into_dyn();
        let thresholds = arr2(&[[0.0_f32]]).into_dyn();
        let relaxed_result = relaxed_clip(
            &x_l,
            &x_u,
            &l_a,
            &lbias,
            &thresholds,
            1,
            true,
        );
        prop_assert!(
            relaxed_result.is_err(),
            "relaxed_clip must reject inverted bounds (x_l={}, x_u={}), got Ok",
            center + inversion,
            center,
        );
    }

    /// Regression audit for #3876: the new ordering guard iterates over every
    /// active batch/dimension slot, so the property test should cover nontrivial
    /// shapes instead of only the 1x1 fast path.
    #[test]
    fn proptest_inverted_input_bounds_return_error_nontrivial_shape_3876(
        c00 in -3.0f32..3.0,
        c01 in -3.0f32..3.0,
        c02 in -3.0f32..3.0,
        c10 in -3.0f32..3.0,
        c11 in -3.0f32..3.0,
        c12 in -3.0f32..3.0,
        inversion in 0.01f32..2.0,
        invert_slot in 0usize..6,
        obj0 in nonzero_coeff(),
        obj1 in nonzero_coeff(),
        obj2 in nonzero_coeff(),
        bias0 in -1.0f32..1.0,
        bias1 in -1.0f32..1.0,
    ) {
        let upper = [
            [c00 + 0.5, c01 + 0.5, c02 + 0.5],
            [c10 + 0.5, c11 + 0.5, c12 + 0.5],
        ];
        let ordered_lower = [
            [c00 - 0.5, c01 - 0.5, c02 - 0.5],
            [c10 - 0.5, c11 - 0.5, c12 - 0.5],
        ];
        let mut lower = ordered_lower;
        let batch = invert_slot / 3;
        let dim = invert_slot % 3;
        lower[batch][dim] = upper[batch][dim] + inversion;

        let centers = [
            [
                f32::midpoint(ordered_lower[0][0], upper[0][0]),
                f32::midpoint(ordered_lower[0][1], upper[0][1]),
                f32::midpoint(ordered_lower[0][2], upper[0][2]),
            ],
            [
                f32::midpoint(ordered_lower[1][0], upper[1][0]),
                f32::midpoint(ordered_lower[1][1], upper[1][1]),
                f32::midpoint(ordered_lower[1][2], upper[1][2]),
            ],
        ];

        let x_l = arr2(&lower).into_dyn();
        let x_u = arr2(&upper).into_dyn();
        let objective = arr2(&[[obj0, obj1, obj2]]).into_dyn();
        let a_matrix = arr3(&[
            [[obj0, obj1, obj2]],
            [[obj0, obj1, obj2]],
        ])
        .into_dyn();
        let b_vector = arr2(&[[
            -(obj0 * centers[0][0] + obj1 * centers[0][1] + obj2 * centers[0][2]) - bias0.abs(),
        ], [
            -(obj0 * centers[1][0] + obj1 * centers[1][1] + obj2 * centers[1][2]) - bias1.abs(),
        ]])
        .into_dyn();

        let complete_err =
            complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1)
                .expect_err("complete_clip must reject inverted bounds in any batch/dim");
        prop_assert!(
            matches!(complete_err, ny_core::NyError::InvalidSpec(_)),
            "complete_clip must surface InvalidSpec, got {complete_err:?}",
        );

        let l_a = arr3(&[
            [[obj0, obj1, obj2]],
            [[obj0, obj1, obj2]],
        ])
        .into_dyn();
        let lbias = arr2(&[[bias0], [bias1]]).into_dyn();
        let thresholds = arr2(&[[0.0_f32], [0.0_f32]]).into_dyn();
        let relaxed_err = relaxed_clip(
            &x_l,
            &x_u,
            &l_a,
            &lbias,
            &thresholds,
            1,
            true,
        )
        .expect_err("relaxed_clip must reject inverted bounds in any batch/dim");
        prop_assert!(
            matches!(relaxed_err, ny_core::NyError::InvalidSpec(_)),
            "relaxed_clip must surface InvalidSpec, got {relaxed_err:?}",
        );
    }
}
