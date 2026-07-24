// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{IndexAddLayer, ScatterAddLayer};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn make_bounded(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).expect("valid upper"),
    )
    .expect("bounded tensor should construct")
}

#[test]
fn scatter_add_activation_input_count_tracks_embedded_constants() {
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::zeros(IxDyn(&[4]))),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 1]).unwrap()),
        None,
    );
    assert_eq!(layer.activation_input_count(), 1);
}

#[test]
fn scatter_add_static_indices_adds_exactly() {
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1_i64, 1]).unwrap()),
        None,
    );
    let src = make_bounded(&[2.0, -3.0], &[4.0, -1.0], &[2]);
    let output = layer
        .propagate_ibp(&src)
        .expect("ScatterAdd IBP should succeed");
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[10.0, 19.0, 30.0, 40.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[10.0, 23.0, 30.0, 40.0]
    );
}

#[test]
fn scatter_add_dynamic_indices_adds_global_contribution_range() {
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let index = make_bounded(&[0.0, 0.0], &[2.0, 2.0], &[2]);
    let src = make_bounded(&[-2.0, 1.5], &[3.0, 4.0], &[2]);
    let output = layer
        .propagate_ibp_binary(&index, &src)
        .expect("ScatterAdd IBP should succeed");
    assert_eq!(output.lower().as_slice().unwrap(), &[-2.0, -2.0, -2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[7.0, 7.0, 7.0]);
}

#[test]
fn index_add_static_indices_adds_selected_rows() {
    let layer = IndexAddLayer::new(
        1,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        Some(arr1(&[2_i64, 0]).into_dyn()),
        None,
    );
    let src = make_bounded(&[10.0, -1.0, 20.0, -2.0], &[10.0, 1.0, 20.0, 2.0], &[2, 2]);
    let output = layer
        .propagate_ibp(&src)
        .expect("IndexAdd IBP should succeed");
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[0.0, 2.0, 13.0, 2.0, 5.0, 26.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[2.0, 2.0, 13.0, 6.0, 5.0, 26.0]
    );
}

#[test]
fn index_add_dynamic_indices_adds_global_contribution_range() {
    let layer = IndexAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let index = make_bounded(&[0.0, 0.0], &[2.0, 2.0], &[2]);
    let src = make_bounded(&[-1.0, 2.0], &[3.0, 4.0], &[2]);
    let output = layer
        .propagate_ibp_binary(&index, &src)
        .expect("IndexAdd IBP should succeed");
    assert_eq!(output.lower().as_slice().unwrap(), &[-1.0, -1.0, -1.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[7.0, 7.0, 7.0]);
}

#[test]
fn scatter_add_propagate_linear_returns_unsupported() {
    let layer = ScatterAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[1]))), None, None);
    let err = layer
        .propagate_linear(&LinearBounds::identity(1))
        .expect_err("ScatterAdd CROWN should be unsupported");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// === CROWN backward exactness/soundness ===
//
// Strategy: With `node_lb = identity(output_size)`, the CROWN backward produces
// linear bounds whose concretization over the variable input's box must equal
// the IBP forward output (since identity spec means "output == output"). Because
// these ops are exactly linear with constant indices, CROWN must reproduce IBP
// exactly. We also build the dense Jacobian explicitly and check the CROWN
// A-matrix equals it.

/// Assert the CROWN LOWER bias is SOUND (`<=` the exact real value) and within a
/// tiny absolute margin of it. The const-shift bias fold accumulates in f64 and
/// directed-rounds OUTWARD (`next_down_f32`) for soundness (#vnncomp-aw-soundness),
/// so an exact integer expectation like `5.0` may legitimately land one ULP below
/// (e.g. `4.9999995`) and `0.0` at the smallest negative subnormal. We therefore
/// check sound-and-close rather than bit-exact.
fn assert_lower_bias_sound_close(crown: &LinearBounds, expected: &[f32]) {
    let lb = crown.lower_b().as_slice().unwrap();
    assert_eq!(lb.len(), expected.len(), "bias length mismatch");
    for (i, (&got, &exp)) in lb.iter().zip(expected.iter()).enumerate() {
        // Sound: lower bias must not exceed the exact value (allow a sub-ULP
        // tolerance to absorb the exact value's own f32 representation).
        assert!(
            got <= exp + 1e-5,
            "lower_b[{i}] = {got} exceeds expected {exp} (unsound lower bias)"
        );
        assert!(
            (got - exp).abs() <= 1e-5,
            "lower_b[{i}] = {got} too far from expected {exp}"
        );
    }
}

fn concretize_eq_ibp(crown: &LinearBounds, var: &BoundedTensor, ibp: &BoundedTensor) {
    let c = crown.concretize(var);
    let cl = c.lower().as_slice().unwrap();
    let cu = c.upper().as_slice().unwrap();
    let il = ibp.lower().as_slice().unwrap();
    let iu = ibp.upper().as_slice().unwrap();
    assert_eq!(cl.len(), il.len(), "lower length mismatch");
    for i in 0..cl.len() {
        assert!(
            (cl[i] - il[i]).abs() < 1e-5,
            "lower[{i}]: crown {} vs ibp {}",
            cl[i],
            il[i]
        );
        assert!(
            (cu[i] - iu[i]).abs() < 1e-5,
            "upper[{i}]: crown {} vs ibp {}",
            cu[i],
            iu[i]
        );
    }
}

#[test]
fn index_add_crown_data_variable_matches_ibp_and_dense() {
    // y = data_var (3) + index_add(axis=0, src_const) at indices [2, 0].
    // src_const = [10, -5]: adds 10 to position 2, -5 to position 0.
    let layer = IndexAddLayer::new(
        0,
        None,
        Some(arr1(&[2_i64, 0]).into_dyn()),
        Some(arr1(&[10.0_f32, -5.0]).into_dyn()),
    );
    let data = make_bounded(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[3]);
    let ibp = layer.propagate_ibp(&data).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(3))
        .expect("crown backward");
    // Data-variable: Jacobian is identity, bias = [-5, 0, 10].
    assert_eq!(crown.lower_a(), &ndarray::Array2::<f32>::eye(3));
    assert_eq!(crown.upper_a(), &ndarray::Array2::<f32>::eye(3));
    assert_lower_bias_sound_close(&crown, &[-5.0, 0.0, 10.0]);
    concretize_eq_ibp(&crown, &data, &ibp);
}

#[test]
fn index_add_crown_src_variable_matches_ibp_and_dense() {
    // y = data_const (2x3) + index_add(axis=1, src_var) with indices [2, 0].
    // Mirrors the fixed IBP test but with src as the variable operand.
    let layer = IndexAddLayer::new(
        1,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        Some(arr1(&[2_i64, 0]).into_dyn()),
        None,
    );
    let src = make_bounded(&[10.0, -1.0, 20.0, -2.0], &[10.0, 1.0, 20.0, 2.0], &[2, 2]);
    let ibp = layer.propagate_ibp(&src).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(6))
        .expect("crown backward");
    // Build the dense Jacobian S (6 outputs x 4 src) explicitly and compare.
    // src layout [2,2] row-major: (r,c) -> r*2 + c.
    // output layout [2,3] row-major: (r,k) -> r*3 + k.
    // index_add axis=1: src col 0 -> out col 2, src col 1 -> out col 0.
    let mut dense = ndarray::Array2::<f32>::zeros((6, 4));
    for r in 0..2 {
        // src col0 (k=0) adds to out col2
        dense[[r * 3 + 2, r * 2]] = 1.0;
        // src col1 (k=1) adds to out col0
        dense[[r * 3, r * 2 + 1]] = 1.0;
    }
    assert_eq!(crown.lower_a(), &dense, "src Jacobian mismatch");
    assert_eq!(crown.upper_a(), &dense);
    // bias == flattened data_const (sound-and-close after outward directed round).
    assert_lower_bias_sound_close(&crown, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    concretize_eq_ibp(&crown, &src, &ibp);
}

#[test]
fn scatter_add_crown_src_variable_matches_ibp_and_dense() {
    // y = data_const (4) + scatter_add(axis=-1, src_var) at indices [1, 1].
    // Both src elements scatter into output position 1 (accumulate).
    let layer = ScatterAddLayer::new(
        -1,
        Some(ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1_i64, 1]).unwrap()),
        None,
    );
    let src = make_bounded(&[2.0, -3.0], &[4.0, -1.0], &[2]);
    let ibp = layer.propagate_ibp(&src).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(4))
        .expect("crown backward");
    // Dense S (4 outputs x 2 src): both src cols add into output row 1.
    let mut dense = ndarray::Array2::<f32>::zeros((4, 2));
    dense[[1, 0]] = 1.0;
    dense[[1, 1]] = 1.0;
    assert_eq!(crown.lower_a(), &dense);
    assert_eq!(crown.upper_a(), &dense);
    assert_lower_bias_sound_close(&crown, &[10.0, 20.0, 30.0, 40.0]);
    concretize_eq_ibp(&crown, &src, &ibp);
}

#[test]
fn scatter_add_crown_data_variable_matches_ibp() {
    // y = data_var (4) + scatter_add(axis=-1, src_const [5, -7]) at indices [0, 3].
    let layer = ScatterAddLayer::new(
        -1,
        None,
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 3]).unwrap()),
        Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0_f32, -7.0]).unwrap()),
    );
    let data = make_bounded(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0], &[4]);
    let ibp = layer.propagate_ibp(&data).expect("ibp");
    let crown = layer
        .crown_backward(&LinearBounds::identity(4))
        .expect("crown backward");
    assert_eq!(crown.lower_a(), &ndarray::Array2::<f32>::eye(4));
    assert_lower_bias_sound_close(&crown, &[5.0, 0.0, 0.0, -7.0]);
    concretize_eq_ibp(&crown, &data, &ibp);
}

#[test]
fn index_add_crown_dynamic_indices_unsupported() {
    // Dynamic indices (indices = None) must fall back: UnsupportedOp.
    let layer = IndexAddLayer::new(-1, Some(ArrayD::zeros(IxDyn(&[3]))), None, None);
    let err = layer
        .propagate_linear(&LinearBounds::identity(3))
        .expect_err("dynamic indices should be unsupported");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}
