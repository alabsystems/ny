// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::CrownMergeAccumulator;
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::network::core::GraphNetwork;
use crate::network::NETWORK_INPUT;
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::{Duration, Instant};

fn test_graph() -> GraphNetwork {
    GraphNetwork {
        output_node: "output".to_string(),
        ..GraphNetwork::new()
    }
}

fn anchored_merge_fixture(seed: f32) -> PatchesLinearBounds {
    let (oc, oh, ow) = (1, 2, 2);
    let (ic, kh, kw) = (1, 1, 1);
    let row_count = oc * oh * ow;
    let geometry =
        PatchGeometry::anchored(vec![0, 1], vec![0, 1]).expect("fixture axes are non-empty");
    PatchesLinearBounds {
        row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), seed)),
            geometry: geometry.clone(),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, oh, ow),
            unstable_idx: None,
        },
        lower_b: Array1::from_vec(vec![seed, seed + 1.0, seed + 2.0, seed + 3.0]),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(
                IxDyn(&[oc, oh, ow, ic, kh, kw]),
                seed + 4.0,
            )),
            geometry,
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, oh, ow),
            unstable_idx: None,
        },
        upper_b: Array1::from_vec(vec![seed + 5.0, seed + 6.0, seed + 7.0, seed + 8.0]),
    }
}

fn affine_merge_fixture(fill_lower: f32, fill_upper: f32) -> PatchesLinearBounds {
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
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
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
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, 4, 4),
            unstable_idx: None,
        },
        upper_b: Array1::from_elem(row_count, fill_upper),
    }
}

fn assert_patches_exact(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
    fn assert_data_exact(actual: &PatchesData, expected: &PatchesData) {
        assert_eq!(actual.coeff_err, expected.coeff_err);
        assert_eq!(actual.patches, expected.patches);
        assert_eq!(actual.geometry, expected.geometry);
        assert_eq!(actual.identity, expected.identity);
        assert_eq!(actual.output_shape, expected.output_shape);
        assert_eq!(actual.input_shape, expected.input_shape);
        assert_eq!(actual.unstable_idx, expected.unstable_idx);
    }

    assert_eq!(actual.row_count, expected.row_count);
    assert_data_exact(&actual.lower_a, &expected.lower_a);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_data_exact(&actual.upper_a, &expected.upper_a);
    assert_eq!(actual.upper_b, expected.upper_b);
}

fn assert_dense_exact(actual: &LinearBounds, expected: &LinearBounds) {
    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.upper_b(), expected.upper_b());
    assert_eq!(actual.lower_a_err(), expected.lower_a_err());
    assert_eq!(actual.upper_a_err(), expected.upper_a_err());
}

#[test]
fn network_input_bias_memory_refusal_preserves_pending_anchored_carrier() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");

        for indexed in [false, true] {
            let expected = anchored_merge_fixture(0.25);
            let mut accumulator = if indexed {
                CrownMergeAccumulator::new_indexed(&["output".to_string()])
            } else {
                CrownMergeAccumulator::new()
            };
            accumulator.insert(
                NETWORK_INPUT.to_string(),
                CrownBounds::Patches(Box::new(expected.clone())),
            );
            let mut input_accumulated = true;
            let error = GraphNetwork::accumulate_bias_to_network_input_crown(
                &array![1.0, 2.0, 3.0, 4.0],
                &array![5.0, 6.0, 7.0, 8.0],
                &mut accumulator,
                4,
                4,
                &mut input_accumulated,
            )
            .expect_err("zero-budget bias merge must refuse before materializing Patches");
            assert!(
                matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            assert!(
                input_accumulated,
                "a failed merge must not rewrite the caller's accumulation state"
            );

            let retained = accumulator
                .take(NETWORK_INPUT)
                .expect("pending carrier retrieval must succeed")
                .expect("failed bias merge must retain NETWORK_INPUT");
            let CrownBounds::Patches(retained) = retained else {
                panic!("failed bias merge changed the pending carrier type");
            };
            assert_patches_exact(&retained, &expected);
        }
    });
}

#[test]
fn first_network_input_bias_refuses_before_dense_zero_pair_allocation() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");

        let mut accumulator = CrownMergeAccumulator::new();
        let mut input_accumulated = false;
        let error = GraphNetwork::accumulate_bias_to_network_input_crown(
            &array![1.0, 2.0],
            &array![3.0, 4.0],
            &mut accumulator,
            2,
            4,
            &mut input_accumulated,
        )
        .expect_err("zero budget must refuse before constructing the first Dense bias carrier");
        assert!(matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }));
        assert!(accumulator.is_empty());
        assert!(!input_accumulated);
    });
}

fn merge_test_accumulator(indexed: bool) -> CrownMergeAccumulator {
    if indexed {
        CrownMergeAccumulator::new_indexed(&["residual".to_string()])
    } else {
        CrownMergeAccumulator::new()
    }
}

const ONE_MIB_PAIR_ROWS: usize = 256;
const ONE_MIB_PAIR_COLS: usize = 512;

fn one_mib_dense_pair_fixture(fill: f32) -> LinearBounds {
    LinearBounds::from_parts_unchecked(
        Array2::from_elem((ONE_MIB_PAIR_ROWS, ONE_MIB_PAIR_COLS), fill),
        Array1::from_elem(ONE_MIB_PAIR_ROWS, fill + 1.0),
        Array2::from_elem((ONE_MIB_PAIR_ROWS, ONE_MIB_PAIR_COLS), fill + 2.0),
        Array1::from_elem(ONE_MIB_PAIR_ROWS, fill + 3.0),
    )
}

fn one_mib_anchored_pair_fixture() -> PatchesLinearBounds {
    let (oc, oh, ow) = (1, 16, 16);
    let (ic, in_h, in_w) = (1, 16, 32);
    let row_count = oc * oh * ow;
    debug_assert_eq!(row_count, ONE_MIB_PAIR_ROWS);
    debug_assert_eq!(ic * in_h * in_w, ONE_MIB_PAIR_COLS);
    let origins: Vec<i128> = (0_i128..16).collect();
    let geometry =
        PatchGeometry::anchored(origins.clone(), origins).expect("fixture axes are non-empty");
    PatchesLinearBounds {
        row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, 1, 1]), 0.25)),
            geometry: geometry.clone(),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, in_h, in_w),
            unstable_idx: None,
        },
        lower_b: Array1::from_elem(row_count, -0.5),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, 1, 1]), 0.75)),
            geometry,
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, in_h, in_w),
            unstable_idx: None,
        },
        upper_b: Array1::from_elem(row_count, 0.5),
    }
}

fn accumulator_with_one_mib_merged_sidecar(indexed: bool) -> CrownMergeAccumulator {
    let mut accumulator = merge_test_accumulator(indexed);
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Dense(one_mib_dense_pair_fixture(0.25)),
    );
    accumulator
        .merge_crown(
            "residual",
            CrownBounds::Dense(one_mib_dense_pair_fixture(0.5)),
        )
        .expect("two Dense parents establish the f64 merge sidecar without promotion");
    accumulator
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
fn test_merge_binary64_cancellation_bias_is_directed_and_coeff_error_is_carried() {
    use ny_tensor::BoundedTensor;

    let large = 2.0_f32.powi(60);
    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Dense(scalar_linear_bounds(large)),
    );
    for contribution in [1.0, -large] {
        accumulator
            .merge_dense("residual", scalar_linear_bounds(contribution))
            .unwrap();
    }
    let merged = accumulator
        .take("residual")
        .unwrap()
        .unwrap()
        .into_dense()
        .unwrap();

    // Biases have no separate error carrier, so each merge addition is directed
    // in the bound's polarity. This remains sound even when cancellation makes
    // the directed binary64 interval wider than the exact residual.
    assert!(merged.lower_b()[0] <= 1.0 && merged.upper_b()[0] >= 1.0);

    // Coefficients cannot be rounded in one fixed direction, so their exact
    // TwoSum residual is carried through the certified coefficient-error
    // channel and discharged when concretized.
    let point = array![1.0].into_dyn();
    let concrete = merged.concretize_sound(&BoundedTensor::new(point.clone(), point).unwrap());
    assert!(concrete.lower()[[0]] <= 2.0 && concrete.upper()[[0]] >= 2.0);
}

#[test]
fn test_merge_error_addition_rounds_up_per_addend() {
    let large = 2.0_f64.powi(60);
    let mut existing = Some(array![[large]]);
    let incoming = array![[1.0_f32]];
    let zero_roundoff = array![[0.0_f64]];
    CrownMergeAccumulator::accumulate_err(&mut existing, Some(&incoming), &zero_roundoff, 1, 1);
    let accumulated = existing.unwrap()[[0, 0]];
    assert!(
        accumulated > large,
        "upward error sum {accumulated:e} did not cover exact {large:e} + 1"
    );

    let mut coefficient = array![[large]];
    let residual =
        CrownMergeAccumulator::accumulate_coeff_array(&mut coefficient, &array![[1.0_f32]]);
    assert_eq!(coefficient[[0, 0]], large);
    assert_eq!(residual[[0, 0]], 1.0);
}

#[test]
fn test_crown_merge_accumulator_downcasts_with_directed_rounding_2657() {
    let _env = crate::tests::lock_env_shared();
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
    let mut accumulator = CrownMergeAccumulator::new();
    accumulator.insert(
        "residual".to_string(),
        CrownBounds::Patches(Box::new(affine_merge_fixture(0.25, 0.75))),
    );

    accumulator
        .merge_crown(
            "residual",
            CrownBounds::Patches(Box::new(affine_merge_fixture(0.1, 0.5))),
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
fn soft_deadline_keeps_compatible_patches_merge_native_but_hard_deadline_does_not() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");
        let live = Instant::now() + Duration::from_secs(30);

        for indexed in [false, true] {
            let mut soft = merge_test_accumulator(indexed);
            soft.insert(
                "residual".to_string(),
                CrownBounds::Patches(Box::new(affine_merge_fixture(0.25, 0.75))),
            );
            soft.merge_crown_with_deadline_authority(
                "residual",
                CrownBounds::Patches(Box::new(affine_merge_fixture(0.1, 0.5))),
                Some(live),
                false,
            )
            .expect("a soft collector timestamp must preserve the native Patches merge");
            assert!(
                soft.take("residual")
                    .expect("soft merge take succeeds")
                    .expect("soft merge retains the residual")
                    .is_patches(),
                "soft-authority compatible merge must remain Patches under a zero dense budget"
            );

            let mut hard = merge_test_accumulator(indexed);
            hard.insert(
                "residual".to_string(),
                CrownBounds::Patches(Box::new(affine_merge_fixture(0.25, 0.75))),
            );
            let error = hard
                .merge_crown_with_deadline(
                    "residual",
                    CrownBounds::Patches(Box::new(affine_merge_fixture(0.1, 0.5))),
                    Some(live),
                )
                .expect_err("hard authority must retain the cooperative promotion policy");
            assert!(matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }));
            assert!(
                hard.take("residual")
                    .expect("hard refusal take succeeds")
                    .expect("hard refusal retains the pending residual")
                    .is_patches(),
                "hard-authority refusal must be atomic"
            );
        }
    });
}

#[test]
/// #4382 regression: merge_crown with incompatible patches (different stride)
/// promotes to Dense without panicking.
fn test_crown_merge_accumulator_merge_crown_incompatible_patches_promotes_dense_4382() {
    let _env = crate::tests::lock_env_shared();
    use crate::bounds::patches::PatchesData;
    use crate::bounds::patches::PatchesLinearBounds;
    use ndarray::{Array1, ArrayD, IxDyn};

    let make_pb = |stride: (usize, usize), fill: f32| -> PatchesLinearBounds {
        let (oc, oh, ow) = (1, 2, 2);
        let (ic, kh, kw) = (1, 3, 3);
        let row_count = oc * oh * ow;
        let padding = if stride == (1, 1) {
            (0, 0, 0, 0)
        } else {
            (1, 1, 1, 1)
        };
        PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), fill)),
                geometry: PatchGeometry::affine(stride, padding),
                identity: false,
                output_shape: (oc, oh, ow),
                input_shape: (ic, 4, 4),
                unstable_idx: None,
            },
            lower_b: Array1::from_elem(row_count, fill),
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), fill)),
                geometry: PatchGeometry::affine(stride, padding),
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
    // Excluded from overlapping an env WRITER. `NY_DENSE_BUDGET_MB` is read
    // process-globally by `crown_memory::explicit_cpu_crown_dense_budget_bytes`;
    // a concurrent test setting it to 0 starves this one's dense path, which
    // surfaced here as `budget_bytes: 0` / bounds no tighter than IBP rather
    // than as the race it is. Observed at --test-threads=8.
    let _env = crate::tests::lock_env_shared();
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
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, 4, 4),
            unstable_idx: None,
        },
        lower_b: Array1::from_elem(row_count, 0.5),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[oc, oh, ow, ic, kh, kw]), 0.5)),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (oc, oh, ow),
            input_shape: (ic, 4, 4),
            unstable_idx: None,
        },
        upper_b: Array1::from_elem(row_count, 0.5),
    };
    let dense = pb
        .to_dense()
        .expect("valid patches fixture must materialize to a dense seed");

    let mut accumulator = CrownMergeAccumulator::new();
    // Insert Dense first, then merge Patches → should promote Patches to Dense
    accumulator.insert("residual".to_string(), CrownBounds::Dense(dense));

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
fn mixed_merge_budget_refusal_preserves_pending_anchored_patches_in_both_stores() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");

        for indexed in [false, true] {
            let expected = anchored_merge_fixture(0.25);
            let mut accumulator = merge_test_accumulator(indexed);
            accumulator.insert(
                "residual".to_string(),
                CrownBounds::Patches(Box::new(expected.clone())),
            );

            let error = accumulator
                .merge_crown("residual", CrownBounds::Dense(LinearBounds::identity(4)))
                .expect_err("zero budget must refuse pending Patches + incoming Dense");
            assert!(
                matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );

            let pending = accumulator
                .take("residual")
                .expect("refused merge must retain an accessible pending entry")
                .expect("refused merge must not remove the pending entry");
            match pending {
                CrownBounds::Patches(actual) => assert_patches_exact(&actual, &expected),
                CrownBounds::Dense(_) => panic!(
                    "refused mixed merge changed the exact pending Anchored carrier to Dense"
                ),
            }
        }
    });
}

#[test]
fn expired_merge_preserves_pending_anchored_patches_in_both_stores() {
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");

    for indexed in [false, true] {
        let expected = anchored_merge_fixture(0.25);
        let mut accumulator = merge_test_accumulator(indexed);
        accumulator.insert(
            "residual".to_string(),
            CrownBounds::Patches(Box::new(expected.clone())),
        );

        let error = accumulator
            .merge_crown_with_deadline(
                "residual",
                CrownBounds::Dense(LinearBounds::identity(4)),
                Some(expired),
            )
            .expect_err("expired merge must be terminal before publication");
        assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));

        let pending = accumulator
            .take("residual")
            .expect("deadline-refused merge must retain the pending entry")
            .expect("pending entry must still exist");
        match pending {
            CrownBounds::Patches(actual) => assert_patches_exact(&actual, &expected),
            CrownBounds::Dense(_) => panic!("deadline-refused merge changed carrier type"),
        }
    }
}

#[test]
fn expired_frontier_snapshot_is_atomic_for_anchored_patches() {
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");

    for indexed in [false, true] {
        let expected = anchored_merge_fixture(0.5);
        let mut accumulator = merge_test_accumulator(indexed);
        accumulator.insert(
            "residual".to_string(),
            CrownBounds::Patches(Box::new(expected.clone())),
        );

        let error = accumulator
            .snapshot_dense_with_deadline(Some(expired))
            .expect_err("expired frontier snapshot must refuse atomically");
        assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));

        let pending = accumulator
            .take("residual")
            .expect("snapshot refusal must leave the frontier readable")
            .expect("snapshot refusal must retain the frontier entry");
        match pending {
            CrownBounds::Patches(actual) => assert_patches_exact(&actual, &expected),
            CrownBounds::Dense(_) => panic!("snapshot refusal changed carrier type"),
        }
    }
}

#[test]
fn mixed_merge_budget_refusal_preserves_dense_pending_and_incoming_patches_source() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");

        for indexed in [false, true] {
            let expected_dense = LinearBounds::identity(4);
            let incoming = anchored_merge_fixture(0.5);
            let incoming_snapshot = incoming.clone();
            let mut accumulator = merge_test_accumulator(indexed);
            accumulator.insert(
                "residual".to_string(),
                CrownBounds::Dense(expected_dense.clone()),
            );

            let error = accumulator
                .merge_crown("residual", CrownBounds::Patches(Box::new(incoming.clone())))
                .expect_err("zero budget must refuse pending Dense + incoming Patches");
            assert!(
                matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            assert_patches_exact(&incoming, &incoming_snapshot);

            let pending = accumulator
                .take("residual")
                .expect("refused merge must retain an accessible pending entry")
                .expect("refused merge must not remove the pending entry");
            match pending {
                CrownBounds::Dense(actual) => assert_dense_exact(&actual, &expected_dense),
                CrownBounds::Patches(_) => {
                    panic!("refused mixed merge changed the exact pending Dense carrier")
                }
            }
        }
    });
}

#[test]
fn third_patches_parent_accounts_for_live_merged_sidecar_and_refuses_atomically() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "4");

        for indexed in [false, true] {
            // Each relation's 256x512 lower/upper f32 coefficient pair is
            // exactly 1 MiB.  The historical one-sided receipt therefore
            // computed 1 MiB * 4 == the 4 MiB budget and admitted the third
            // parent.  Including the already-live f64 sidecar's logical pair
            // correctly includes `(1 MiB + 1 MiB) * 4` PLUS both retained
            // sources (the Anchored carrier and live f64 sidecar) and refuses.
            let mut expected_accumulator = accumulator_with_one_mib_merged_sidecar(indexed);
            let expected = expected_accumulator
                .take("residual")
                .expect("baseline sidecar must downcast")
                .expect("baseline sidecar must exist");

            let incoming = one_mib_anchored_pair_fixture();
            let incoming_snapshot = incoming.clone();
            let mut accumulator = accumulator_with_one_mib_merged_sidecar(indexed);
            let error = accumulator
                .merge_crown("residual", CrownBounds::Patches(Box::new(incoming.clone())))
                .expect_err("third Patches parent must include the live sidecar receipt");
            match error {
                ny_core::NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    ..
                } => {
                    assert!(
                        required_bytes > 8 * 1024 * 1024,
                        "full source+dense+sidecar peak must exceed the old pair-only receipt"
                    );
                    assert_eq!(budget_bytes, 4 * 1024 * 1024);
                }
                error => panic!("expected typed memory refusal, got {error:?}"),
            }
            assert_patches_exact(&incoming, &incoming_snapshot);

            let actual = accumulator
                .take("residual")
                .expect("refused third merge must retain the existing sidecar")
                .expect("refused third merge must not remove the sidecar");
            match (actual, expected) {
                (CrownBounds::Dense(actual), CrownBounds::Dense(expected)) => {
                    assert_dense_exact(&actual, &expected);
                }
                (actual, expected) => panic!(
                    "third-parent refusal changed carrier types: actual={actual:?}, expected={expected:?}"
                ),
            }
        }
    });
}

#[test]
fn test_crown_merge_accumulator_indexed_network_input_dense_promotion_4296() {
    let _env = crate::tests::lock_env_shared();
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

#[test]
fn expired_take_preserves_f64_sidecar_for_later_publication() {
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");

    for indexed in [false, true] {
        let mut accumulator = merge_test_accumulator(indexed);
        accumulator.insert(
            "residual".to_string(),
            CrownBounds::Dense(scalar_linear_bounds(1.0)),
        );
        accumulator
            .merge_crown("residual", CrownBounds::Dense(scalar_linear_bounds(2.0)))
            .expect("fixture merge establishes an f64 sidecar");

        let error = accumulator
            .take_with_deadline("residual", Some(expired))
            .expect_err("expired take must not erase the f64 sidecar");
        assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));

        let retained = accumulator
            .take("residual")
            .expect("ordinary take after refusal must succeed")
            .expect("deadline refusal must retain the sidecar")
            .into_dense()
            .expect("sidecar publishes Dense");
        assert!(retained.lower_a()[[0, 0]] <= 3.0);
        assert!(retained.upper_a()[[0, 0]] >= 3.0);
    }
}

#[test]
fn resident_aware_take_refuses_before_removing_f64_sidecar() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "4");

        for indexed in [false, true] {
            let mut accumulator = accumulator_with_one_mib_merged_sidecar(indexed);
            let error = accumulator
                .take_with_deadline_and_resident(
                    "residual",
                    Some(
                        Instant::now()
                            .checked_add(Duration::from_secs(30))
                            .expect("deadline fits in Instant"),
                    ),
                    2 * 1024 * 1024,
                )
                .expect_err("retained capture payload must be included in take admission");
            assert!(matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }));

            let retained = accumulator
                .take("residual")
                .expect("ordinary take after refusal must succeed")
                .expect("memory refusal must retain the f64 sidecar")
                .into_dense()
                .expect("retained sidecar publishes Dense");
            assert!(retained.lower_a()[[0, 0]] <= 0.75);
            assert!(retained.upper_a()[[0, 0]] >= 4.75);
        }
    });
}

#[test]
fn resident_aware_take_none_preserves_legacy_admission() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "4");

        for indexed in [false, true] {
            let mut accumulator = accumulator_with_one_mib_merged_sidecar(indexed);
            let retained = accumulator
                .take_with_deadline_and_resident("residual", None, usize::MAX)
                .expect("None must ignore finite-authority resident receipts")
                .expect("legacy sidecar must remain publishable")
                .into_dense()
                .expect("legacy sidecar publishes Dense");
            assert!(retained.lower_a()[[0, 0]] <= 0.75);
            assert!(retained.upper_a()[[0, 0]] >= 4.75);
        }
    });
}
