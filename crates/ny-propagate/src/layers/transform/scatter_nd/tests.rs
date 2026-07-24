// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, ArrayD, IxDyn};

fn make_bounded(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).unwrap()
}

#[ntest::timeout(5000)]
#[test]
fn test_activation_input_count_tracks_embedded_constants() {
    let static_indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 3_i64]).unwrap();
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), Some(static_indices), None);
    assert_eq!(layer.activation_input_count(), 1);

    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), None, None);
    assert_eq!(layer.activation_input_count(), 2);

    let layer = ScatterNdLayer::new(None, None, None);
    assert_eq!(layer.activation_input_count(), 3);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_static_indices_overwrite_constant_data() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 3_i64]).unwrap();
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), Some(indices), None);
    let updates = make_bounded(&[1.0, -2.0], &[2.0, 3.0]);

    let result = layer.propagate_ibp(&updates).unwrap();
    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.lower().as_slice().unwrap(), &[0.0, 1.0, 0.0, -2.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[0.0, 2.0, 0.0, 3.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_static_duplicate_indices_union_overwrites() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 1_i64]).unwrap();
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), Some(indices), None);
    let updates = make_bounded(&[1.0, -2.0], &[2.0, 3.0]);

    let result = layer.propagate_ibp(&updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[0.0, -2.0, 0.0, 0.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[0.0, 3.0, 0.0, 0.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_static_indices_overwrite_dynamic_data() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 3_i64]).unwrap();
    let layer = ScatterNdLayer::new(None, Some(indices), None);
    let data = make_bounded(&[0.0, 1.0, 2.0, 3.0], &[0.5, 1.5, 2.5, 3.5]);
    let updates = make_bounded(&[-1.0, 4.0], &[2.0, 5.0]);

    let result = layer.propagate_ibp_binary(&data, &updates).unwrap();
    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.lower().as_slice().unwrap(), &[0.0, -1.0, 2.0, 4.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[0.5, 2.0, 2.5, 5.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_dynamic_indices_unions_data_and_updates() {
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), None, None);
    let indices = make_bounded(&[0.0, 1.0], &[3.0, 3.0]);
    let updates = make_bounded(&[-1.0, 2.0], &[0.5, 4.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.shape(), &[4]);
    assert_eq!(
        result.lower().as_slice().unwrap(),
        &[-1.0, -1.0, -1.0, -1.0]
    );
    assert_eq!(result.upper().as_slice().unwrap(), &[4.0, 4.0, 4.0, 4.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_dynamic_indices_unions_dynamic_data_and_updates() {
    let layer = ScatterNdLayer::new(None, None, None);
    let data = make_bounded(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0]);
    let indices = make_bounded(&[0.0, 1.0], &[3.0, 3.0]);
    let updates = make_bounded(&[-1.0, 4.0], &[5.0, 6.0]);

    let result = layer
        .propagate_ibp_ternary(&data, &indices, &updates)
        .unwrap();
    assert_eq!(result.shape(), &[4]);
    assert_eq!(
        result.lower().as_slice().unwrap(),
        &[-1.0, -1.0, -1.0, -1.0]
    );
    assert_eq!(result.upper().as_slice().unwrap(), &[6.0, 6.0, 6.0, 6.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_propagate_linear_dynamic_indices_unsupported() {
    // Dynamic indices (indices = None) cannot be exactly linearized.
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), None, None);
    let bounds = LinearBounds::identity(4);
    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "expected UnsupportedOp, got {err}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_propagate_linear_duplicate_targets_unsupported() {
    // Duplicate write targets => union/last-write, not exactly linear => fallback.
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 1_i64]).unwrap();
    let layer = ScatterNdLayer::new(Some(ArrayD::zeros(IxDyn(&[4]))), Some(indices), None);
    let err = layer
        .propagate_linear(&LinearBounds::identity(4))
        .unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "expected UnsupportedOp for duplicate targets, got {err}"
    );
}

// CROWN backward exactness: with node_lb = identity, concretizing over the
// variable input's box must reproduce IBP exactly (overwrite is piecewise-linear
// with a fixed structure when indices are constant).
fn concretize_eq_ibp(crown: &LinearBounds, var: &BoundedTensor, ibp: &BoundedTensor) {
    let c = crown.concretize(var);
    let cl = c.lower().as_slice().unwrap();
    let cu = c.upper().as_slice().unwrap();
    let il = ibp.lower().as_slice().unwrap();
    let iu = ibp.upper().as_slice().unwrap();
    for i in 0..cl.len() {
        assert!(
            (cl[i] - il[i]).abs() < 1e-5,
            "lower[{i}]: {} vs {}",
            cl[i],
            il[i]
        );
        assert!(
            (cu[i] - iu[i]).abs() < 1e-5,
            "upper[{i}]: {} vs {}",
            cu[i],
            iu[i]
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_updates_variable_matches_ibp_and_dense() {
    // data_const (4) overwritten at positions [1, 3] by variable updates.
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1_i64, 3_i64]).unwrap();
    let layer = ScatterNdLayer::new(
        Some(arr1(&[10.0_f32, 20.0, 30.0, 40.0]).into_dyn()),
        Some(indices),
        None,
    );
    let updates = make_bounded(&[1.0, -2.0], &[2.0, 3.0]);
    let ibp = layer.propagate_ibp(&updates).unwrap();
    let crown = layer.crown_backward(&LinearBounds::identity(4)).unwrap();
    // Dense S (4 outputs x 2 updates): out1<-upd0, out3<-upd1; out0,out2 from data.
    let mut dense = Array2::<f32>::zeros((4, 2));
    dense[[1, 0]] = 1.0;
    dense[[3, 1]] = 1.0;
    assert_eq!(crown.lower_a(), &dense);
    assert_eq!(crown.upper_a(), &dense);
    // Bias holds the constant data at the unwritten positions (0 and 2), folded via f64 +
    // directed cast so each lower-bias entry ENCLOSES (rounds <=) the true value (#vnncomp-aw).
    for (got, want) in crown
        .lower_b()
        .iter()
        .zip([10.0_f32, 0.0, 30.0, 0.0].iter())
    {
        assert!(
            *got <= *want + 1e-6 && (*got - *want).abs() < 1e-4,
            "lower bias {got} must enclose true {want}"
        );
    }
    concretize_eq_ibp(&crown, &updates, &ibp);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_data_variable_matches_ibp_and_dense() {
    // variable data (4), constant updates overwrite positions [0, 2].
    let indices = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0_i64, 2_i64]).unwrap();
    let layer = ScatterNdLayer::new(None, Some(indices), Some(arr1(&[7.0_f32, -3.0]).into_dyn()));
    let data = make_bounded(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]);
    let ibp = layer.propagate_ibp(&data).unwrap();
    let crown = layer.crown_backward(&LinearBounds::identity(4)).unwrap();
    // Jacobian: identity except written positions 0 and 2 are zeroed (constant).
    let mut dense = Array2::<f32>::eye(4);
    dense[[0, 0]] = 0.0;
    dense[[2, 2]] = 0.0;
    assert_eq!(crown.lower_a(), &dense);
    assert_eq!(crown.upper_a(), &dense);
    // Bias: constant updates at written positions 0 and 2, folded via f64 + directed cast so
    // each lower-bias entry ENCLOSES (rounds <=) the true value (#vnncomp-aw-soundness).
    for (got, want) in crown.lower_b().iter().zip([7.0_f32, 0.0, -3.0, 0.0].iter()) {
        assert!(
            *got <= *want + 1e-6 && (*got - *want).abs() < 1e-4,
            "lower bias {got} must enclose true {want}"
        );
    }
    concretize_eq_ibp(&crown, &data, &ibp);
}

/// Regression (#vnncomp-aw-soundness self-audit): ScatterND CROWN backward folds the constant
/// `coeff * c` over the written positions into the bias; the f32 multiply + accumulation can
/// EXCLUDE the true value under cancellation. With incoming upper coeffs [2^24, 1, -2^24] and
/// unit constants, f32 left-to-right gives 2^24 + 1 -> 2^24 (drops the 1), then -2^24 -> 0,
/// while the true fold is 1. Pre-fix upper_b = 0 < 1 (a false-proof: upper bound below the true
/// value); after the fix the f64 + directed fold makes upper_b enclose 1.
#[test]
fn scatter_nd_constant_fold_encloses_under_cancellation() {
    let two24 = 16_777_216.0_f32; // 2^24
    let indices = ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![0_i64, 1, 2]).unwrap();
    let layer = ScatterNdLayer::new(
        None,
        Some(indices),
        Some(arr1(&[1.0_f32, 1.0, 1.0]).into_dyn()),
    );
    let a = Array2::from_shape_vec((1, 4), vec![two24, 1.0, -two24, 0.0]).unwrap();
    let node_lb =
        LinearBounds::new_or_conservative(a.clone(), arr1(&[0.0_f32]), a, arr1(&[0.0_f32]))
            .unwrap();
    let crown = layer.crown_backward(&node_lb).unwrap();
    let true_fold = 1.0_f64; // 2^24 + 1 - 2^24, exact
    assert!(
        crown.upper_b()[0] as f64 >= true_fold,
        "upper bias {} must ENCLOSE the true fold {true_fold} (pre-fix f32 gave 0)",
        crown.upper_b()[0]
    );
}

// ---------------------------------------------------------------------------
// Bounded-index dynamic ScatterND (#cctsdb B4)
// ---------------------------------------------------------------------------

/// Helper: bounded tensor with explicit shape.
fn make_bounded_shaped(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Singleton index intervals: the write target is certain, so the written
/// element takes EXACTLY the update value (data dropped) and untouched
/// elements keep data exactly — no global hull smearing.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_singleton_is_exact() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    // indices shape [1,1]: one row, depth 1, singleton value 1.
    let indices = make_bounded_shaped(&[1, 1], &[1.0], &[1.0]);
    let updates = make_bounded_shaped(&[1], &[5.0], &[5.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 5.0, 30.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 5.0, 30.0]);
}

/// Ranged index interval [1,2]: elements 1 and 2 are possibly written (hull of
/// data and update); element 0 is untouched (exact data).
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_range_hulls_only_reachable() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    let indices = make_bounded_shaped(&[1, 1], &[1.0], &[2.0]);
    let updates = make_bounded_shaped(&[1], &[5.0], &[5.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 5.0, 5.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 20.0, 30.0]);
}

/// Two singleton rows hitting the same element (duplicate targets): the value
/// is the hull over both candidate writers (ONNX duplicate order is
/// unspecified), data dropped.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_duplicate_rows_union() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    let indices = make_bounded_shaped(&[2, 1], &[1.0, 1.0], &[1.0, 1.0]);
    let updates = make_bounded_shaped(&[2], &[5.0, 7.0], &[5.0, 7.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 5.0, 30.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 7.0, 30.0]);
}

/// A fully out-of-range row is skipped (never widens data) — the patch-3
/// edge-cell invariant: the static-max-shape window {2,3,4} over axis length 4
/// writes exactly {2,3}, with the sentinel 4 rejected.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_out_of_range_row_rejected() {
    let data = ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    let indices = make_bounded_shaped(&[3, 1], &[2.0, 3.0, 4.0], &[2.0, 3.0, 4.0]);
    let updates = make_bounded_shaped(&[3], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 20.0, 0.0, 0.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 20.0, 0.0, 0.0]);
}

/// Non-finite index bounds degrade gracefully to the position-blind global
/// hull (previous behavior).
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_non_finite_falls_back_to_global_hull() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    let indices = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::INFINITY),
    )
    .unwrap();
    let updates = make_bounded_shaped(&[1], &[5.0], &[5.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    // Global hull: every element unions with the updates range.
    assert_eq!(result.lower().as_slice().unwrap(), &[5.0, 5.0, 5.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 20.0, 30.0]);
}

/// The cctsdb mask pattern in miniature: ones data, zero updates written at a
/// bounded (x, y) window over a [2, 4, 4] image. Every pixel reachable by the
/// window gets hull [0, 1]; unreachable pixels stay exactly 1.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_mask_window_hull() {
    let data = ArrayD::from_elem(IxDyn(&[2, 4, 4]), 1.0_f32);
    let layer = ScatterNdLayer::new(Some(data), None, None);
    // 2 rows (one per channel), depth 3: [c, y, x] with c singleton, y in
    // [0,2], x in [1,2].
    let indices = make_bounded_shaped(
        &[2, 3],
        &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        &[0.0, 2.0, 2.0, 1.0, 2.0, 2.0],
    );
    let updates = make_bounded_shaped(&[2], &[0.0, 0.0], &[0.0, 0.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    for c in 0..2 {
        for y in 0..4 {
            for x in 0..4 {
                let lo = result.lower()[[c, y, x]];
                let up = result.upper()[[c, y, x]];
                let reachable = y <= 2 && (1..=2).contains(&x);
                if reachable {
                    assert_eq!((lo, up), (0.0, 1.0), "pixel ({c},{y},{x}) must hull [0,1]");
                } else {
                    assert_eq!((lo, up), (1.0, 1.0), "pixel ({c},{y},{x}) must stay 1");
                }
            }
        }
    }
}

/// Negative bounded indices normalize (ONNX semantics): index interval [-1,-1]
/// over length 3 writes element 2 exactly.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_negative_normalized() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    let indices = make_bounded_shaped(&[1, 1], &[-1.0], &[-1.0]);
    let updates = make_bounded_shaped(&[1], &[5.0], &[5.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 20.0, 5.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 20.0, 5.0]);
}

/// Fractional index intervals contain no integer for [0.3, 0.7]-style ranges
/// only when the integer hull is empty; [0.5, 1.5] must cover exactly {1}.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_fractional_integer_hull() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    // True index is an integer in [0.5, 1.5] => exactly 1 (definite write).
    let indices = make_bounded_shaped(&[1, 1], &[0.5], &[1.5]);
    let updates = make_bounded_shaped(&[1], &[5.0], &[5.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[10.0, 5.0, 30.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[10.0, 5.0, 30.0]);
}

/// Partial indexing (index_depth < data rank, slice_len > 1): a singleton row
/// writes its whole slice exactly; other slices stay data-exact.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_bounded_indices_row_slice_write() {
    let data = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let layer = ScatterNdLayer::new(Some(data), None, None);
    // One row, depth 1 over axis 0: writes data[1, :].
    let indices = make_bounded_shaped(&[1, 1], &[1.0], &[1.0]);
    let updates = make_bounded_shaped(&[1, 2], &[5.0, 6.0], &[5.0, 6.0]);

    let result = layer.propagate_ibp_binary(&indices, &updates).unwrap();
    assert_eq!(
        result.lower().as_slice().unwrap(),
        &[1.0, 1.0, 5.0, 6.0, 1.0, 1.0]
    );
    assert_eq!(
        result.upper().as_slice().unwrap(),
        &[1.0, 1.0, 5.0, 6.0, 1.0, 1.0]
    );
}
