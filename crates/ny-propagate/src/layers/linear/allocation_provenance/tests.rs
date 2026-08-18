// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::mem::size_of;

use ndarray::{array, s, Array1, Array2, Axis, Ix1, Slice};
use ny_core::NyError;

use super::*;

#[derive(Clone, Copy, Debug)]
struct RawExpectation {
    base: usize,
    logical: usize,
    raw_len: usize,
    offset: Option<usize>,
    capacity: usize,
}

fn spare_weight(requested_capacity: usize) -> (Array2<f32>, RawExpectation) {
    let mut raw = Vec::with_capacity(requested_capacity);
    raw.extend_from_slice(&[91.0, 1.0, 2.0, 3.0, 4.0, 92.0]);
    let base = raw.as_ptr() as usize;
    let capacity = raw.capacity();
    let raw_len = raw.len();
    let logical = Array1::from_vec(raw).slice_axis_move(Axis(0), Slice::from(1..5));
    let logical_identity = logical.as_ptr() as usize;
    let weight = logical
        .into_shape_with_order((2, 2))
        .expect("four values reshape to 2x2");
    (
        weight,
        RawExpectation {
            base,
            logical: logical_identity,
            raw_len,
            offset: Some(1),
            capacity,
        },
    )
}

fn spare_bias(requested_capacity: usize) -> (Array1<f32>, RawExpectation) {
    let mut raw = Vec::with_capacity(requested_capacity);
    raw.extend_from_slice(&[93.0, 0.5, -0.25, 94.0]);
    let base = raw.as_ptr() as usize;
    let capacity = raw.capacity();
    let raw_len = raw.len();
    let bias = Array1::from_vec(raw).slice_axis_move(Axis(0), Slice::from(1..3));
    let logical = bias.as_ptr() as usize;
    (
        bias,
        RawExpectation {
            base,
            logical,
            raw_len,
            offset: Some(1),
            capacity,
        },
    )
}

fn fixture() -> LinearLayer {
    LinearLayer::new(
        array![[1.0_f32, -2.0], [3.0, 4.0]],
        Some(array![0.5, -0.25]),
    )
    .expect("valid Linear fixture")
}

fn warm_all(layer: &LinearLayer) {
    let _ = layer.weight_t_row_major();
    let _ = layer.w_pos_t_row_major();
    let _ = layer.w_neg_t_row_major();
}

fn observe(layer: &LinearLayer) -> LinearLayerAllocationObservationV1<'_> {
    layer
        .observe_host_allocations_v1(&mut |_| Ok(()))
        .expect("nonfailing observation poll")
}

fn accounted(layer: &LinearLayer) -> LinearLayerAllocationFragmentV1<'_> {
    match observe(layer) {
        LinearLayerAllocationObservationV1::Accounted(fragment) => fragment,
        other => panic!("expected Accounted, got {other:?}"),
    }
}

fn ndarray_fact(
    fragment: &LinearLayerAllocationFragmentV1<'_>,
    role: LinearNdArrayRoleV1,
) -> LinearNdArrayAllocationFactV1 {
    fragment
        .ndarray_facts()
        .iter()
        .flatten()
        .copied()
        .find(|fact| fact.role == role)
        .unwrap_or_else(|| panic!("missing ndarray fact for {role:?}"))
}

fn cache_fact(
    fragment: &LinearLayerAllocationFragmentV1<'_>,
    role: LinearTransposeCacheRoleV1,
) -> LinearTransposeCacheAllocationFactV1 {
    fragment
        .transpose_cache_facts()
        .iter()
        .copied()
        .find(|fact| fact.role == role)
        .unwrap_or_else(|| panic!("missing cache fact for {role:?}"))
}

#[test]
fn tracked_weight_and_bias_preserve_raw_owner_pointer_offset_and_spare_capacity() {
    let (weight, weight_expected) = spare_weight(31);
    let (bias, bias_expected) = spare_bias(29);
    let layer = LinearLayer::new(weight, Some(bias)).expect("valid spare-capacity Linear");

    assert_eq!(layer.weight().as_ptr() as usize, weight_expected.logical);
    assert_eq!(
        layer.bias().unwrap().as_ptr() as usize,
        bias_expected.logical
    );
    warm_all(&layer);
    let fragment = accounted(&layer);
    let weight = ndarray_fact(&fragment, LinearNdArrayRoleV1::Weight);
    let bias = ndarray_fact(&fragment, LinearNdArrayRoleV1::Bias);

    assert_eq!(weight.raw_allocation_identity, weight_expected.base);
    assert_eq!(weight.logical_data_identity, weight_expected.logical);
    assert_eq!(weight.raw_len, weight_expected.raw_len);
    assert_eq!(weight.logical_offset, weight_expected.offset);
    assert_eq!(weight.element_capacity, weight_expected.capacity);
    assert_eq!(weight.shape, [2, 2]);
    assert_eq!(weight.rank, 2);
    assert_eq!(
        weight.exact_payload_capacity_bytes,
        weight_expected.capacity * size_of::<f32>()
    );

    assert_eq!(bias.raw_allocation_identity, bias_expected.base);
    assert_eq!(bias.logical_data_identity, bias_expected.logical);
    assert_eq!(bias.raw_len, bias_expected.raw_len);
    assert_eq!(bias.logical_offset, bias_expected.offset);
    assert_eq!(bias.element_capacity, bias_expected.capacity);
    assert_eq!(bias.shape, [2, 0]);
    assert_eq!(bias.rank, 1);
}

#[test]
fn tracked_empty_owner_preserves_capacity_and_uses_no_logical_offset() {
    let raw = Vec::<f32>::with_capacity(17);
    let expected_base = raw.as_ptr() as usize;
    let expected_capacity = raw.capacity();
    let tracked = TrackedLinearArray::<Ix1>::new(Array1::from_vec(raw));
    let fact = tracked
        .allocation_fact(LinearNdArrayRoleV1::Bias)
        .expect("empty default-C owner is accountable");

    assert_eq!(fact.raw_allocation_identity, expected_base);
    assert_eq!(fact.raw_len, 0);
    assert_eq!(fact.logical_offset, None);
    assert_eq!(fact.element_capacity, expected_capacity);
    assert_eq!(fact.exact_payload_capacity_bytes, expected_capacity * 4);
}

#[test]
fn valid_noncanonical_bias_keeps_public_values_but_is_clean_unsupported() {
    let bias = Array1::from_vec(vec![0.5_f32, 99.0, -0.25, 98.0]).slice_move(s![..;2]);
    assert_eq!(bias.strides(), &[2]);
    let layer = LinearLayer::new(array![[1.0_f32, 2.0], [3.0, 4.0]], Some(bias))
        .expect("historically valid strided bias remains valid");
    assert_eq!(layer.bias().unwrap(), &array![0.5_f32, -0.25]);
    warm_all(&layer);

    let observation = observe(&layer);
    assert!(observation.permits_legacy_fallback());
    assert!(std::ptr::eq(observation.source(), &raw const layer));
    assert!(matches!(
        observation,
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::NdArrayNonCanonicalCLayout {
                role: LinearNdArrayRoleV1::Bias
            },
            ..
        }
    ));
}

#[test]
fn constructor_valid_zero_input_and_zero_output_layers_are_clean_unsupported() {
    let layers = [
        LinearLayer::new(Array2::zeros((2, 0)), Some(Array1::zeros(2)))
            .expect("zero-input Linear remains constructor-valid"),
        LinearLayer::new(Array2::zeros((0, 2)), Some(Array1::zeros(0)))
            .expect("zero-output Linear remains constructor-valid"),
    ];

    for layer in &layers {
        warm_all(layer);
        let observation = observe(layer);
        assert!(observation.permits_legacy_fallback());
        assert!(std::ptr::eq(observation.source(), layer));
        assert!(matches!(
            observation,
            LinearLayerAllocationObservationV1::Unsupported {
                reason: LinearLayerAllocationUnsupportedV1::EmptyWeightDimension,
                ..
            }
        ));
    }
}

#[test]
fn cold_partial_and_warm_cache_matrix_has_no_partial_fragment() {
    let layer = fixture();
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::ColdTransposeCache {
                role: LinearTransposeCacheRoleV1::Weight
            },
            ..
        }
    ));

    let _ = layer.weight_t_row_major();
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::ColdTransposeCache {
                role: LinearTransposeCacheRoleV1::PositiveWeight
            },
            ..
        }
    ));

    let _ = layer.w_pos_t_row_major();
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::ColdTransposeCache {
                role: LinearTransposeCacheRoleV1::NegativeWeight
            },
            ..
        }
    ));

    let _ = layer.w_neg_t_row_major();
    let fragment = accounted(&layer);
    assert!(fragment.matches_source(&layer));
    assert_eq!(fragment.accounting_model(), 1);
}

#[test]
fn warm_cache_facts_bind_exact_transpose_bits_capacity_and_distinct_controls() {
    let layer = fixture();
    warm_all(&layer);
    assert_eq!(layer.weight_t_row_major(), &[1.0, 3.0, -2.0, 4.0]);
    assert_eq!(layer.w_pos_t_row_major(), &[1.0, 3.0, 0.0, 4.0]);
    assert_eq!(layer.w_neg_t_row_major(), &[0.0, 0.0, -2.0, 0.0]);

    let fragment = accounted(&layer);
    let weight = cache_fact(&fragment, LinearTransposeCacheRoleV1::Weight);
    let positive = cache_fact(&fragment, LinearTransposeCacheRoleV1::PositiveWeight);
    let negative = cache_fact(&fragment, LinearTransposeCacheRoleV1::NegativeWeight);
    for fact in [weight, positive, negative] {
        assert_eq!(fact.element_len, 4);
        assert!(fact.element_capacity >= fact.element_len);
        assert_eq!(fact.exact_payload_capacity_bytes, fact.element_capacity * 4);
        assert_ne!(fact.vec_data_identity, 0);
        assert_eq!(
            fact.arc_control_allocation_request_bytes,
            arc_control_allocation_request_bytes_v1().unwrap()
        );
    }
    assert_ne!(weight.arc_control_identity, positive.arc_control_identity);
    assert_ne!(weight.arc_control_identity, negative.arc_control_identity);
    assert_ne!(positive.arc_control_identity, negative.arc_control_identity);
}

#[test]
fn manual_clone_recaptures_deep_owner_allocations_and_preserves_cache_aliases() {
    let (weight, expected) = spare_weight(37);
    let (bias, bias_expected) = spare_bias(35);
    let layer = LinearLayer::new(weight, Some(bias)).expect("valid spare-capacity Linear owners");
    warm_all(&layer);
    let cloned = layer.clone();

    let source_fragment = accounted(&layer);
    let cloned_fragment = accounted(&cloned);
    let source_weight = ndarray_fact(&source_fragment, LinearNdArrayRoleV1::Weight);
    let cloned_weight = ndarray_fact(&cloned_fragment, LinearNdArrayRoleV1::Weight);
    assert_eq!(source_weight.element_capacity, expected.capacity);
    assert_ne!(
        source_weight.raw_allocation_identity,
        cloned_weight.raw_allocation_identity
    );
    assert_ne!(
        source_weight.logical_data_identity,
        cloned_weight.logical_data_identity
    );
    assert_ne!(
        source_weight.element_capacity,
        cloned_weight.element_capacity
    );
    assert_eq!(cloned_weight.element_capacity, source_weight.raw_len);
    let source_bias = ndarray_fact(&source_fragment, LinearNdArrayRoleV1::Bias);
    let cloned_bias = ndarray_fact(&cloned_fragment, LinearNdArrayRoleV1::Bias);
    assert_eq!(source_bias.element_capacity, bias_expected.capacity);
    assert_ne!(source_bias.element_capacity, cloned_bias.element_capacity);
    assert_eq!(cloned_bias.element_capacity, source_bias.raw_len);

    for role in [
        LinearNdArrayRoleV1::Weight,
        LinearNdArrayRoleV1::Bias,
        LinearNdArrayRoleV1::PositiveWeight,
        LinearNdArrayRoleV1::NegativeWeight,
        LinearNdArrayRoleV1::RowL2Norms,
    ] {
        let source = ndarray_fact(&source_fragment, role);
        let cloned = ndarray_fact(&cloned_fragment, role);
        assert_eq!(source.role, cloned.role);
        assert_eq!(source.rank, cloned.rank);
        assert_eq!(source.shape, cloned.shape);
        assert_eq!(source.raw_len, cloned.raw_len);
        assert_eq!(source.logical_offset, cloned.logical_offset);
        assert_eq!(
            source.exact_payload_capacity_bytes,
            source.element_capacity * size_of::<f32>()
        );
        assert_eq!(
            cloned.exact_payload_capacity_bytes,
            cloned.element_capacity * size_of::<f32>()
        );
        assert_ne!(
            source.raw_allocation_identity,
            cloned.raw_allocation_identity
        );
        assert_ne!(source.logical_data_identity, cloned.logical_data_identity);
    }
    assert_eq!(layer.weight(), cloned.weight());
    assert_eq!(layer.bias(), cloned.bias());
    assert_eq!(layer.w_pos.as_array(), cloned.w_pos.as_array());
    assert_eq!(layer.w_neg.as_array(), cloned.w_neg.as_array());
    assert_eq!(
        layer.row_l2_norms.as_array(),
        cloned.row_l2_norms.as_array()
    );

    for (source, cloned) in source_fragment
        .faer_facts()
        .iter()
        .zip(cloned_fragment.faer_facts())
    {
        assert_eq!(source.role, cloned.role);
        assert_eq!(source.nrows, cloned.nrows);
        assert_eq!(source.ncols, cloned.ncols);
        assert_eq!(source.column_stride, cloned.column_stride);
        assert_ne!(source.allocation_identity, cloned.allocation_identity);
    }
    for role in [
        LinearTransposeCacheRoleV1::Weight,
        LinearTransposeCacheRoleV1::PositiveWeight,
        LinearTransposeCacheRoleV1::NegativeWeight,
    ] {
        let source = cache_fact(&source_fragment, role);
        let cloned = cache_fact(&cloned_fragment, role);
        assert_eq!(source.arc_control_identity, cloned.arc_control_identity);
        assert_eq!(source.vec_data_identity, cloned.vec_data_identity);
        assert_eq!(source.element_capacity, cloned.element_capacity);
    }
}

#[test]
fn failed_and_successful_bias_setters_preserve_custody_and_recapture() {
    let mut layer = fixture();
    warm_all(&layer);
    let before = accounted(&layer);
    let before_weight = ndarray_fact(&before, LinearNdArrayRoleV1::Weight);
    let before_bias = ndarray_fact(&before, LinearNdArrayRoleV1::Bias);
    let before_cache = cache_fact(&before, LinearTransposeCacheRoleV1::Weight);
    drop(before);

    assert!(layer.set_bias(Some(array![1.0_f32, 2.0, 3.0])).is_err());
    let after_failure = accounted(&layer);
    assert_eq!(
        ndarray_fact(&after_failure, LinearNdArrayRoleV1::Weight),
        before_weight
    );
    assert_eq!(
        ndarray_fact(&after_failure, LinearNdArrayRoleV1::Bias),
        before_bias
    );
    assert_eq!(
        cache_fact(&after_failure, LinearTransposeCacheRoleV1::Weight),
        before_cache
    );
    drop(after_failure);

    let (replacement, expected) = spare_bias(43);
    layer
        .set_bias(Some(replacement))
        .expect("shape-compatible bias replacement");
    assert_eq!(layer.bias().unwrap(), &array![0.5_f32, -0.25]);
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::ColdTransposeCache { .. },
            ..
        }
    ));
    warm_all(&layer);
    let after_success = accounted(&layer);
    let new_bias = ndarray_fact(&after_success, LinearNdArrayRoleV1::Bias);
    assert_eq!(new_bias.raw_allocation_identity, expected.base);
    assert_eq!(new_bias.logical_data_identity, expected.logical);
    assert_eq!(new_bias.raw_len, expected.raw_len);
    assert_eq!(new_bias.logical_offset, expected.offset);
    assert_eq!(new_bias.element_capacity, expected.capacity);
    assert_ne!(
        new_bias.raw_allocation_identity,
        before_bias.raw_allocation_identity
    );
    assert_ne!(
        ndarray_fact(&after_success, LinearNdArrayRoleV1::Weight).raw_allocation_identity,
        before_weight.raw_allocation_identity
    );
    assert_ne!(
        cache_fact(&after_success, LinearTransposeCacheRoleV1::Weight).arc_control_identity,
        before_cache.arc_control_identity
    );
}

#[test]
fn failed_and_successful_setters_preserve_transactionality_and_recapture() {
    let mut layer = fixture();
    warm_all(&layer);
    let before = accounted(&layer);
    let before_weight = ndarray_fact(&before, LinearNdArrayRoleV1::Weight);
    let before_cache = cache_fact(&before, LinearTransposeCacheRoleV1::Weight);
    drop(before);

    assert!(layer.set_weight(Array2::zeros((3, 2))).is_err());
    let after_failure = accounted(&layer);
    assert_eq!(
        ndarray_fact(&after_failure, LinearNdArrayRoleV1::Weight),
        before_weight
    );
    assert_eq!(
        cache_fact(&after_failure, LinearTransposeCacheRoleV1::Weight),
        before_cache
    );
    drop(after_failure);

    let (replacement, expected) = spare_weight(41);
    layer
        .set_weight(replacement)
        .expect("shape-compatible replacement");
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Unsupported {
            reason: LinearLayerAllocationUnsupportedV1::ColdTransposeCache { .. },
            ..
        }
    ));
    warm_all(&layer);
    let after_success = accounted(&layer);
    let new_weight = ndarray_fact(&after_success, LinearNdArrayRoleV1::Weight);
    assert_eq!(new_weight.raw_allocation_identity, expected.base);
    assert_eq!(new_weight.element_capacity, expected.capacity);
    assert_ne!(
        new_weight.raw_allocation_identity,
        before_weight.raw_allocation_identity
    );
    assert_ne!(
        cache_fact(&after_success, LinearTransposeCacheRoleV1::Weight).arc_control_identity,
        before_cache.arc_control_identity
    );
}

#[test]
fn faer_024_padding_and_allocation_request_boundaries_are_literal() {
    for (rows, columns, expected_stride, expected_bytes) in [
        (0, 2, 0, 0),
        (3, 2, 16, 128),
        (16, 2, 16, 128),
        (17, 2, 32, 256),
    ] {
        let matrix = Mat::<f32>::from_fn(rows, columns, |row, column| (row + column) as f32);
        let fact = faer_allocation_fact_v1(&matrix, LinearFaerRoleV1::Weight, rows, columns)
            .expect("pinned faer matrix matches the audited model");
        assert_eq!(fact.nrows, rows);
        assert_eq!(fact.ncols, columns);
        assert_eq!(fact.column_stride, expected_stride);
        assert_eq!(fact.exact_allocation_request_bytes, expected_bytes);
    }

    assert_eq!(expected_faer_f32_column_stride_v1(3), Some(16));
    assert_eq!(expected_faer_f32_column_stride_v1(16), Some(16));
    assert_eq!(expected_faer_f32_column_stride_v1(17), Some(32));
    assert_eq!(expected_faer_f32_column_stride_v1(usize::MAX), None);
}

#[test]
fn rust_195_arc_control_model_is_exact_for_the_pinned_64_bit_target() {
    let request = arc_control_allocation_request_bytes_v1().expect("sized Arc layout");
    #[cfg(target_pointer_width = "64")]
    assert_eq!(request, 48);
    assert!(request >= 2 * size_of::<AtomicUsize>() + size_of::<OnceLock<Vec<f32>>>());
}

#[test]
fn corrupted_warm_cache_bits_and_lengths_are_invalid_not_unsupported() {
    let layer = fixture();
    layer
        .weight_t_rm
        .set(vec![1.0, 3.0, -2.0, 99.0])
        .expect("fresh cache cell");
    let _ = layer.w_pos_t_row_major();
    let _ = layer.w_neg_t_row_major();
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::TransposeCacheValueMismatch {
                role: LinearTransposeCacheRoleV1::Weight,
                index: 3
            },
            ..
        }
    ));

    let layer = fixture();
    layer
        .weight_t_rm
        .set(vec![1.0, 3.0, -2.0])
        .expect("fresh cache cell");
    let _ = layer.w_pos_t_row_major();
    let _ = layer.w_neg_t_row_major();
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::TransposeCacheLengthMismatch {
                role: LinearTransposeCacheRoleV1::Weight,
                expected: 4,
                actual: 3
            },
            ..
        }
    ));
}

#[test]
fn cache_control_alias_is_invalid_even_when_values_would_match() {
    let mut layer =
        LinearLayer::new(array![[1.0_f32, 2.0], [3.0, 4.0]], None).expect("positive-only fixture");
    layer.w_pos_t_rm = layer.weight_t_rm.clone();
    warm_all(&layer);
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::TransposeCacheControlAlias {
                first: LinearTransposeCacheRoleV1::Weight,
                second: LinearTransposeCacheRoleV1::PositiveWeight
            },
            ..
        }
    ));
}

#[test]
fn injected_pointer_offset_and_capacity_invariants_fail_closed() {
    let mut layer = fixture();
    warm_all(&layer);
    let TrackedArrayStateV1::Accounted(mut state) = layer.weight.allocation else {
        panic!("fixture weight must be tracked")
    };
    state.logical_data_identity = state.logical_data_identity.wrapping_add(size_of::<f32>());
    layer.weight.allocation = TrackedArrayStateV1::Accounted(state);
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::NdArrayLogicalPointerInvariant {
                role: LinearNdArrayRoleV1::Weight
            },
            ..
        }
    ));

    let mut layer = fixture();
    warm_all(&layer);
    let TrackedArrayStateV1::Accounted(mut state) = layer.weight.allocation else {
        panic!("fixture weight must be tracked")
    };
    state.logical_offset = Some(usize::MAX);
    layer.weight.allocation = TrackedArrayStateV1::Accounted(state);
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::NdArrayRawOffsetInvariant {
                role: LinearNdArrayRoleV1::Weight
            },
            ..
        }
    ));

    let mut layer = fixture();
    warm_all(&layer);
    let TrackedArrayStateV1::Accounted(mut state) = layer.weight.allocation else {
        panic!("fixture weight must be tracked")
    };
    state.element_capacity = usize::MAX;
    layer.weight.allocation = TrackedArrayStateV1::Accounted(state);
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::NdArrayPayloadBytesOverflow {
                role: LinearNdArrayRoleV1::Weight
            },
            ..
        }
    ));
}

#[test]
fn coherent_nonfinite_source_bits_are_accounted_but_derived_corruption_is_invalid() {
    let nonfinite_layer = LinearLayer::new(
        array![[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]],
        Some(array![f32::NAN]),
    )
    .expect("constructor preserves nonfinite parameters");
    warm_all(&nonfinite_layer);
    let fragment = accounted(&nonfinite_layer);
    assert!(fragment.matches_source(&nonfinite_layer));

    let mut layer = fixture();
    warm_all(&layer);
    layer.weight_faer[(0, 0)] = 7.0;
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::FaerValueMismatch {
                role: LinearFaerRoleV1::Weight,
                row: 0,
                column: 0
            },
            ..
        }
    ));

    let mut layer = fixture();
    warm_all(&layer);
    layer.row_l2_norms.value[0] = 99.0;
    assert!(matches!(
        observe(&layer),
        LinearLayerAllocationObservationV1::Invalid {
            reason: LinearLayerAllocationInvalidV1::DerivedNdArrayValueMismatch {
                role: LinearNdArrayRoleV1::RowL2Norms,
                index: 0
            },
            ..
        }
    ));
}

#[test]
fn deadline_poll_error_is_propagated_without_typed_laundering() {
    let layer = fixture();
    warm_all(&layer);
    for target in [
        "linear allocation row norm values",
        "linear allocation bias values",
        "linear allocation fragment publication",
    ] {
        let error = layer
            .observe_host_allocations_v1(&mut |context| {
                if context == target {
                    Err(NyError::DeadlineExceeded(
                        "injected Linear allocation poll".into(),
                    ))
                } else {
                    Ok(())
                }
            })
            .expect_err("targeted poll must propagate expiry");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
    }
}

#[test]
fn deadline_polls_are_proportional_for_bias_and_row_norm_validation() {
    let rows = OBSERVATION_POLL_STRIDE + 1;
    let weight = Array2::from_shape_fn((rows, 1), |(row, _)| row as f32 + 1.0);
    let bias = Array1::from_shape_fn(rows, |row| -(row as f32));
    let layer = LinearLayer::new(weight, Some(bias)).expect("valid tall Linear fixture");
    warm_all(&layer);

    let mut contexts = Vec::new();
    let observation = layer
        .observe_host_allocations_v1(&mut |context| {
            contexts.push(context);
            Ok(())
        })
        .expect("polls do not expire");
    assert!(matches!(
        observation,
        LinearLayerAllocationObservationV1::Accounted(_)
    ));
    assert_eq!(
        contexts
            .iter()
            .filter(|&&context| context == "linear allocation row norm values")
            .count(),
        2
    );
    assert_eq!(
        contexts
            .iter()
            .filter(|&&context| context == "linear allocation bias values")
            .count(),
        2
    );
    assert_eq!(
        contexts
            .iter()
            .filter(|&&context| context == "linear allocation fragment publication")
            .count(),
        1
    );
}

#[test]
fn tracked_linear_owner_remains_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LinearLayer>();
}
