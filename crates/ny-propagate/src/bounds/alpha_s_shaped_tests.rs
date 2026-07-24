// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, Array1};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::alpha_s_shaped::MonotoneSShapedAlpha;

fn checked_bounds(lower: Array1<f32>, upper: Array1<f32>) -> BoundedTensor {
    BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap()
}

fn fake_cross_tangents(l: f32, u: f32) -> (f32, f32) {
    (l - 0.5, u + 0.5)
}

fn arbitrary_projection_value() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(f32::NEG_INFINITY),
        Just(f32::INFINITY),
        Just(f32::NAN),
        -20.0f32..20.0,
    ]
}

prop_compose! {
    fn crossing_interval()(lower in -5.0f32..-0.01, upper in 0.01f32..5.0) -> (f32, f32) {
        (lower, upper)
    }
}

prop_compose! {
    fn negative_interval()(upper in -5.0f32..-0.01, width in 0.0f32..4.0) -> (f32, f32) {
        (upper - width, upper)
    }
}

prop_compose! {
    fn positive_interval()(lower in 0.01f32..5.0, width in 0.0f32..4.0) -> (f32, f32) {
        (lower, lower + width)
    }
}

fn assign_parent_paths(alpha: &mut MonotoneSShapedAlpha, values: &[f32]) {
    let (chunks3, _) = values.as_chunks::<3>();
    let mut chunks = chunks3.iter();

    alpha.tp_pos.lower_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_pos.upper_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_neg.lower_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_neg.upper_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_both_lower.lower_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_both_lower.upper_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_both_upper.lower_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    alpha.tp_both_upper.upper_path = Array1::from_vec(chunks.next().unwrap().to_vec());
    assert!(
        chunks.next().is_none(),
        "expected exactly 24 parent tangent values"
    );
}

#[test]
fn test_monotone_s_shaped_alpha_initializes_midpoint_and_cross_defaults() {
    let bounds = checked_bounds(arr1(&[-2.0, 1.0]), arr1(&[3.0, 4.0]));
    let alpha = MonotoneSShapedAlpha::from_bounds(&bounds, fake_cross_tangents).unwrap();

    assert_eq!(alpha.tp_pos.lower_path[0], 0.5);
    assert_eq!(alpha.tp_neg.lower_path[1], 2.5);
    assert_eq!(alpha.tp_both_lower.lower_path[0], -2.5);
    assert_eq!(alpha.tp_both_upper.upper_path[0], 3.5);
}

#[test]
fn test_monotone_s_shaped_alpha_clamps_crossing_domains() {
    let bounds = checked_bounds(arr1(&[-1.0]), arr1(&[2.0]));
    let mut alpha = MonotoneSShapedAlpha::from_bounds(&bounds, fake_cross_tangents).unwrap();
    let mut perturb = alpha.zeros_gradients();
    perturb.tp_both_lower.lower_path[0] = 1.0;
    perturb.tp_both_upper.upper_path[0] = -1.0;

    alpha.tp_both_lower.lower_path[0] = 10.0;
    alpha.tp_both_upper.upper_path[0] = -10.0;
    alpha.apply_perturbation(&perturb, 0.0);

    assert_eq!(alpha.tp_both_lower.lower_path[0], -1.5);
    assert_eq!(alpha.tp_both_upper.upper_path[0], 2.5);
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_monotone_warm_start_projection_preserves_child_invariants(
        (cross_lower, cross_upper) in crossing_interval(),
        (neg_lower, neg_upper) in negative_interval(),
        (pos_lower, pos_upper) in positive_interval(),
        parent_values in proptest::collection::vec(arbitrary_projection_value(), 24),
    ) {
        let child_bounds = checked_bounds(
            arr1(&[cross_lower, neg_lower, pos_lower]),
            arr1(&[cross_upper, neg_upper, pos_upper]),
        );
        let mut parent_alpha =
            MonotoneSShapedAlpha::from_bounds(&child_bounds, fake_cross_tangents).unwrap();
        let mut child_alpha =
            MonotoneSShapedAlpha::from_bounds(&child_bounds, fake_cross_tangents).unwrap();
        assign_parent_paths(&mut parent_alpha, &parent_values);

        child_alpha.warm_start_from(&parent_alpha);

        let tol = 1e-6_f32;
        for idx in 0..child_alpha.len() {
            let lower_path = child_alpha.lower_path_alpha(idx);
            let upper_path = child_alpha.upper_path_alpha(idx);

            for value in [
                lower_path.tp_pos,
                upper_path.tp_pos,
                lower_path.tp_neg,
                upper_path.tp_neg,
                lower_path.tp_both_lower,
                upper_path.tp_both_lower,
                lower_path.tp_both_upper,
                upper_path.tp_both_upper,
            ] {
                prop_assert!(
                    value.is_finite(),
                    "projected tangent must stay finite at index {idx}, got {value}"
                );
            }

            if child_alpha.mask_pos[idx] {
                for value in [lower_path.tp_pos, upper_path.tp_pos] {
                    prop_assert!(
                        value >= child_alpha.lower_bounds[idx] - tol
                            && value <= child_alpha.upper_bounds[idx] + tol,
                        "positive-interval tangent {value} escaped child bounds [{}, {}] at index {idx}",
                        child_alpha.lower_bounds[idx],
                        child_alpha.upper_bounds[idx],
                    );
                }
            }

            if child_alpha.mask_neg[idx] {
                for value in [lower_path.tp_neg, upper_path.tp_neg] {
                    prop_assert!(
                        value >= child_alpha.lower_bounds[idx] - tol
                            && value <= child_alpha.upper_bounds[idx] + tol,
                        "negative-interval tangent {value} escaped child bounds [{}, {}] at index {idx}",
                        child_alpha.lower_bounds[idx],
                        child_alpha.upper_bounds[idx],
                    );
                }
            }

            if child_alpha.mask_cross[idx] {
                for value in [lower_path.tp_both_lower, upper_path.tp_both_lower] {
                    prop_assert!(
                        value <= child_alpha.d_lower[idx] + tol,
                        "crossing lower tangent {value} exceeded d_lower {} at index {idx}",
                        child_alpha.d_lower[idx],
                    );
                }
                for value in [lower_path.tp_both_upper, upper_path.tp_both_upper] {
                    prop_assert!(
                        value >= child_alpha.d_upper[idx] - tol,
                        "crossing upper tangent {value} fell below d_upper {} at index {idx}",
                        child_alpha.d_upper[idx],
                    );
                }
            }
        }

        for optimizer_state in [
            &child_alpha.tp_pos.velocity_lower,
            &child_alpha.tp_pos.velocity_upper,
            &child_alpha.tp_pos.adam_m_lower,
            &child_alpha.tp_pos.adam_v_lower,
            &child_alpha.tp_pos.adam_m_upper,
            &child_alpha.tp_pos.adam_v_upper,
            &child_alpha.tp_neg.velocity_lower,
            &child_alpha.tp_neg.velocity_upper,
            &child_alpha.tp_neg.adam_m_lower,
            &child_alpha.tp_neg.adam_v_lower,
            &child_alpha.tp_neg.adam_m_upper,
            &child_alpha.tp_neg.adam_v_upper,
            &child_alpha.tp_both_lower.velocity_lower,
            &child_alpha.tp_both_lower.velocity_upper,
            &child_alpha.tp_both_lower.adam_m_lower,
            &child_alpha.tp_both_lower.adam_v_lower,
            &child_alpha.tp_both_lower.adam_m_upper,
            &child_alpha.tp_both_lower.adam_v_upper,
            &child_alpha.tp_both_upper.velocity_lower,
            &child_alpha.tp_both_upper.velocity_upper,
            &child_alpha.tp_both_upper.adam_m_lower,
            &child_alpha.tp_both_upper.adam_v_lower,
            &child_alpha.tp_both_upper.adam_m_upper,
            &child_alpha.tp_both_upper.adam_v_upper,
        ] {
            prop_assert!(
                optimizer_state.iter().all(|value| *value == 0.0),
                "warm-start must not copy optimizer state into the child domain"
            );
        }
    }
}
