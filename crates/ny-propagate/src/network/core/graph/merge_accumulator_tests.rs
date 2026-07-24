// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::CrownMergeAccumulator;
use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::network::core::GraphNetwork;
use crate::network::NETWORK_INPUT;
use ndarray::array;
use ny_tensor::{next_down_f32, next_up_f32};
fn test_graph() -> GraphNetwork {
    GraphNetwork {
        output_node: "output".to_string(),
        ..GraphNetwork::new()
    }
}

fn scalar_linear_bounds(value: f32) -> LinearBounds {
    LinearBounds::from_parts_unchecked(
        array![[value]],
        array![value],
        array![[value]],
        array![value],
    )
}

fn asymmetric_scalar_linear_bounds(lower: f32, upper: f32) -> LinearBounds {
    LinearBounds::from_parts_unchecked(
        array![[lower]],
        array![lower],
        array![[upper]],
        array![upper],
    )
}

fn merge_asymmetric_scalar_bounds(contributions: &[(f32, f32)]) -> LinearBounds {
    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Dense(asymmetric_scalar_linear_bounds(1.0, 1.0)),
    );
    for &(lower, upper) in contributions {
        accumulator
            .merge_dense("residual", asymmetric_scalar_linear_bounds(lower, upper))
            .expect("merge should succeed");
    }
    accumulator
        .take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist")
        .into_dense()
        .expect("merged residual entry should be dense")
}

fn assert_plain_cast_is_directionally_unsound(lower: f32, lower_extra: f32, upper: f32) {
    let exact_lower = 1.0_f64 + f64::from(lower) + f64::from(lower_extra);
    let exact_upper = 1.0_f64 + f64::from(upper);
    assert!(
        f64::from(exact_lower as f32) > exact_lower,
        "plain lower cast should round upward and become unsound"
    );
    assert!(
        f64::from(exact_upper as f32) < exact_upper,
        "plain upper cast should round downward and become unsound"
    );
}

#[test]
fn test_crown_merge_accumulator_preserves_three_term_cancellation_2657() {
    let graph = test_graph();
    let mut accumulator = CrownMergeAccumulator::new();
    let mut input_accumulated = false;

    // f32 loses the unit term when accumulating around 2^40, but f64 still
    // preserves it: ulp_f32(2^40)=2^17 while ulp_f64(2^40)=2^-12.
    let contributions = [1_099_511_627_776.0_f32, 1.0_f32, -1_099_511_627_776.0_f32];
    for contribution in contributions {
        graph
            .accumulate_dense_bounds_to_input(
                "residual",
                scalar_linear_bounds(contribution),
                &mut accumulator,
                1,
                1,
                &mut input_accumulated,
            )
            .expect("accumulation should succeed");
    }

    let merged = accumulator
        .take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist")
        .into_dense()
        .expect("merged residual entry should be dense");

    let mut naive_a = array![[contributions[0]]];
    let mut naive_b = array![contributions[0]];
    for contribution in contributions.iter().skip(1) {
        let add_a = array![[*contribution]];
        let add_b = array![*contribution];
        naive_a = GraphNetwork::safe_add(&naive_a, &add_a, true);
        naive_b = GraphNetwork::safe_add(&naive_b, &add_b, true);
    }

    assert_eq!(
        naive_a[[0, 0]],
        0.0,
        "serial f32 merge should lose the low-order term"
    );
    assert_eq!(
        naive_b[0], 0.0,
        "serial f32 bias merge should lose the low-order term"
    );
    assert_eq!(
        merged.lower_a()[[0, 0]],
        next_down_f32(1.0),
        "f64 sidecar merge should preserve the residual and round the lower side soundly"
    );
    assert_eq!(
        merged.lower_b()[0],
        next_down_f32(1.0),
        "f64 sidecar bias merge should preserve the residual and round the lower side soundly"
    );
    assert_eq!(merged.upper_a()[[0, 0]], next_up_f32(1.0));
    assert_eq!(merged.upper_b()[0], next_up_f32(1.0));
}

#[test]
fn test_crown_merge_accumulator_downcasts_with_directed_rounding_2657() {
    let lower_half_ulp = 2.0_f32.powi(-24);
    let lower_quarter_ulp = 2.0_f32.powi(-25);
    let upper_half_ulp = 2.0_f32.powi(-24);

    assert_plain_cast_is_directionally_unsound(lower_half_ulp, lower_quarter_ulp, upper_half_ulp);
    let merged = merge_asymmetric_scalar_bounds(&[
        (lower_half_ulp, upper_half_ulp),
        (lower_quarter_ulp, 0.0),
    ]);

    assert_eq!(
        merged.lower_a()[[0, 0]],
        1.0,
        "lower coefficient must round toward -inf on f64->f32 downcast"
    );
    assert_eq!(
        merged.lower_b()[0],
        1.0,
        "lower bias must round toward -inf on f64->f32 downcast"
    );
    assert_eq!(
        merged.upper_a()[[0, 0]],
        next_up_f32(1.0),
        "upper coefficient must round toward +inf on f64->f32 downcast"
    );
    assert_eq!(
        merged.upper_b()[0],
        next_up_f32(1.0),
        "upper bias must round toward +inf on f64->f32 downcast"
    );
}

#[test]
fn test_crown_merge_accumulator_missing_network_input_merge_errors_2657() {
    let graph = test_graph();
    let mut accumulator = CrownMergeAccumulator::new();
    let mut input_accumulated = true;

    let err = graph
        .accumulate_dense_bounds_to_input(
            NETWORK_INPUT,
            scalar_linear_bounds(1.0),
            &mut accumulator,
            1,
            1,
            &mut input_accumulated,
        )
        .expect_err("missing _input merge must surface the invariant break");
    let msg = format!("{err}");
    assert!(
        msg.contains("merge expected existing entry"),
        "missing-entry error should explain the merge invariant, got: {msg}"
    );
    assert!(
        accumulator
            .take(NETWORK_INPUT)
            .expect("take should succeed")
            .is_none(),
        "failed merge must not synthesize replacement _input bounds"
    );
}

#[test]
fn test_crown_merge_accumulator_has_only_key_after_dense_promotion_4023() {
    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        NETWORK_INPUT.to_string(),
        CrownBounds::Dense(scalar_linear_bounds(1.0)),
    );
    assert!(
        accumulator.has_only_key(NETWORK_INPUT),
        "#4023 regression: pending NETWORK_INPUT entry should count as the only key"
    );

    accumulator
        .merge_dense(NETWORK_INPUT, scalar_linear_bounds(2.0))
        .expect("NETWORK_INPUT merge should promote the entry into the dense sidecar");

    assert!(
        accumulator.has_only_key(NETWORK_INPUT),
        "#4023 regression: has_only_key must stay true after NETWORK_INPUT moves to merged_dense"
    );
    assert!(
        !accumulator.has_only_key("residual"),
        "nonexistent keys must not report as the sole entry"
    );
}

#[test]
fn test_crown_merge_accumulator_indexed_take_by_idx_round_trips_4296() {
    let exec_order = vec!["hidden".to_string(), "output".to_string()];
    let mut accumulator = CrownMergeAccumulator::new_indexed(&exec_order);
    accumulator.insert(
        "output".to_string(),
        CrownBounds::Dense(scalar_linear_bounds(1.0)),
    );

    assert!(
        accumulator.has_only_key("output"),
        "indexed mode should report the inserted output as the sole pending key"
    );

    let taken = accumulator
        .take_by_idx(1)
        .expect("take_by_idx should succeed for indexed accumulator")
        .expect("output entry should exist at exec_order index 1")
        .into_dense()
        .expect("indexed pending entry should remain dense");

    assert_eq!(
        taken.lower_a()[[0, 0]],
        1.0,
        "take_by_idx should return the original pending bounds without remapping"
    );
    assert_eq!(taken.lower_b()[0], 1.0);
    assert_eq!(taken.upper_a()[[0, 0]], 1.0);
    assert_eq!(taken.upper_b()[0], 1.0);
    assert!(
        accumulator
            .take("output")
            .expect("take by key should still succeed after indexed take")
            .is_none(),
        "take_by_idx must clear the indexed slot"
    );
}

#[test]
/// #4382 regression: merge_crown with compatible Patches keeps them in Patches
/// form instead of promoting to Dense.
fn test_crown_merge_accumulator_merge_crown_patches_stays_patches_4382() {
    use crate::bounds::patches::PatchesData;
    use crate::bounds::patches::PatchesLinearBounds;
    use ndarray::{Array1, ArrayD, IxDyn};

    let make_pb = |fill_lower: f32, fill_upper: f32| -> PatchesLinearBounds {
        let (oc, oh, ow) = (1, 2, 2);
        let (ic, kh, kw) = (1, 3, 3);
        let row_count = oc * oh * ow;
        PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(
                    IxDyn(&[oc, oh, ow, ic, kh, kw]),
                    fill_lower,
                )),
                stride: (1, 1),
                padding: (1, 1, 1, 1),
                identity: false,
                output_shape: (oc, oh, ow),
                input_shape: (ic, 4, 4),
                unstable_idx: None,
            },
            lower_b: Array1::from_elem(row_count, fill_lower),
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(
                    IxDyn(&[oc, oh, ow, ic, kh, kw]),
                    fill_upper,
                )),
                stride: (1, 1),
                padding: (1, 1, 1, 1),
                identity: false,
                output_shape: (oc, oh, ow),
                input_shape: (ic, 4, 4),
                unstable_idx: None,
            },
            upper_b: Array1::from_elem(row_count, fill_upper),
        }
    };

    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Patches(Box::new(make_pb(0.25, 0.75))),
    );

    accumulator
        .merge_crown(
            "residual",
            CrownBounds::Patches(Box::new(make_pb(0.1, 0.5))),
        )
        .expect("compatible patches merge should succeed");

    let taken = accumulator
        .take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist");
    assert!(
        taken.is_patches(),
        "#4382 regression: compatible Patches+Patches merge must stay in Patches form"
    );
}

#[test]
/// #4382 regression: merge_crown with incompatible patches (different stride)
/// promotes to Dense without panicking.
fn test_crown_merge_accumulator_merge_crown_incompatible_patches_promotes_dense_4382() {
    use crate::bounds::patches::PatchesData;
    use crate::bounds::patches::PatchesLinearBounds;
    use ndarray::{Array1, ArrayD, IxDyn};

    let make_pb = |stride: (usize, usize), fill: f32| -> PatchesLinearBounds {
        let (oc, oh, ow) = (1, 2, 2);
        let (ic, kh, kw) = (1, 3, 3);
        let row_count = oc * oh * ow;
        PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), fill)),
                stride,
                padding: (1, 1, 1, 1),
                identity: false,
                output_shape: (oc, oh, ow),
                input_shape: (ic, 4, 4),
                unstable_idx: None,
            },
            lower_b: Array1::from_elem(row_count, fill),
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), fill)),
                stride,
                padding: (1, 1, 1, 1),
                identity: false,
                output_shape: (oc, oh, ow),
                input_shape: (ic, 4, 4),
                unstable_idx: None,
            },
            upper_b: Array1::from_elem(row_count, fill),
        }
    };

    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Patches(Box::new(make_pb((1, 1), 0.5))),
    );

    // Different stride → incompatible → should promote to dense
    accumulator
        .merge_crown(
            "residual",
            CrownBounds::Patches(Box::new(make_pb((2, 2), 0.3))),
        )
        .expect("incompatible patches merge should not panic, should promote to dense");

    let taken = accumulator
        .take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist");
    assert!(
        !taken.is_patches(),
        "#4382 regression: incompatible patches must promote to Dense"
    );
}

#[test]
/// #4382 regression: merge_crown with mixed Dense+Patches promotes to Dense.
fn test_crown_merge_accumulator_merge_crown_mixed_dense_patches_promotes_4382() {
    use crate::bounds::patches::PatchesData;
    use crate::bounds::patches::PatchesLinearBounds;
    use ndarray::{Array1, ArrayD, IxDyn};

    let (oc, oh, ow) = (1, 2, 2);
    let (ic, kh, kw) = (1, 3, 3);
    let row_count = oc * oh * ow;
    let pb = PatchesLinearBounds {
        row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), 0.5)),
            stride: (1, 1),
            padding: (1, 1, 1, 1),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, 4, 4),
            unstable_idx: None,
        },
        lower_b: Array1::from_elem(row_count, 0.5),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), 0.5)),
            stride: (1, 1),
            padding: (1, 1, 1, 1),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, 4, 4),
            unstable_idx: None,
        },
        upper_b: Array1::from_elem(row_count, 0.5),
    };

    let mut accumulator = CrownMergeAccumulator::new();
    // Insert Dense first, then merge Patches → should promote Patches to Dense
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Dense(scalar_linear_bounds(1.0)),
    );

    // merge_crown with Patches on top of existing Dense: patches goes through
    // into_dense fallback since try_patches_merge finds Dense in pending
    accumulator
        .merge_crown("residual", CrownBounds::Patches(Box::new(pb)))
        .expect("mixed Dense+Patches merge should not panic");

    let taken = accumulator
        .take("residual")
        .expect("take should succeed")
        .expect("residual entry should exist");
    assert!(
        !taken.is_patches(),
        "#4382 regression: mixed Dense+Patches must promote to Dense"
    );
}

#[test]
fn test_crown_merge_accumulator_indexed_network_input_dense_promotion_4296() {
    let exec_order = vec!["output".to_string()];
    let mut accumulator = CrownMergeAccumulator::new_indexed(&exec_order);
    accumulator.insert(
        NETWORK_INPUT.to_string(),
        CrownBounds::Dense(scalar_linear_bounds(1.0)),
    );

    accumulator
        .merge_dense(NETWORK_INPUT, scalar_linear_bounds(2.0))
        .expect("indexed NETWORK_INPUT merge should promote the entry into the dense sidecar");

    assert!(
        accumulator.has_only_key(NETWORK_INPUT),
        "indexed dense promotion must preserve the sole-key invariant for NETWORK_INPUT"
    );

    let network_input_idx = exec_order.len();
    let merged = accumulator
        .take_by_idx(network_input_idx)
        .expect("take_by_idx should succeed for NETWORK_INPUT")
        .expect("NETWORK_INPUT entry should still exist after dense promotion")
        .into_dense()
        .expect("merged indexed entry should downcast back to dense bounds");

    assert!(
        merged.lower_a()[[0, 0]] <= 3.0 && merged.upper_a()[[0, 0]] >= 3.0,
        "indexed dense promotion must conservatively enclose the exact coefficient sum"
    );
    assert!(
        merged.lower_b()[0] <= 3.0 && merged.upper_b()[0] >= 3.0,
        "indexed dense promotion must conservatively enclose the exact bias sum"
    );
}
