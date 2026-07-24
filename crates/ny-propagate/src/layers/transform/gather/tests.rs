// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{ArrayD, IxDyn};

fn make_bounded(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new(lower, upper).unwrap()
}

// =========================================================================
// Construction
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_new_infers_indices_shape() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 1]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    assert_eq!(layer.indices_shape, vec![2]);
}

#[ntest::timeout(5000)]
#[test]
fn test_new_explicit_shape_overrides() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 1]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![3]);
    assert_eq!(layer.indices_shape, vec![3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_new_no_indices_empty_shape() {
    let layer = GatherLayer::new(0, None, vec![]);
    assert!(layer.indices.is_none());
    assert!(layer.indices_shape.is_empty());
}

// =========================================================================
// IBP with static indices — axis 0
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_axis0_static_indices() {
    // Input: [3, 2] tensor, gather rows 0 and 2 -> output [2, 2]
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let input = make_bounded(lower, upper);

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    // Row 0: [1, 2] lower, [10, 20] upper
    // Row 2: [5, 6] lower, [50, 60] upper
    assert_eq!(result.lower()[[0, 0]], 1.0);
    assert_eq!(result.lower()[[1, 0]], 5.0);
    assert_eq!(result.upper()[[0, 1]], 20.0);
    assert_eq!(result.upper()[[1, 1]], 60.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_axis1_static_indices() {
    // Input: [2, 4] tensor, gather columns 1 and 3 along axis=1 -> [2, 2]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    )
    .unwrap();
    let input = make_bounded(lower, upper);

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
    let layer = GatherLayer::new(1, Some(indices), vec![]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    // Row 0, gathered cols [1,3]: lower [2, 4], upper [20, 40]
    assert_eq!(result.lower()[[0, 0]], 2.0);
    assert_eq!(result.lower()[[0, 1]], 4.0);
    assert_eq!(result.upper()[[0, 0]], 20.0);
    assert_eq!(result.upper()[[0, 1]], 40.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_negative_index() {
    // Negative index: -1 should mean last element
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let input = make_bounded(lower, upper);

    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.lower()[[0]], 3.0);
    assert_eq!(result.upper()[[0]], 30.0);
}

// =========================================================================
// IBP with dynamic indices — conservative fallback
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_dynamic_indices() {
    // No static indices: should conservatively take min/max across axis.
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 5.0, 2.0, 3.0, 4.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![10.0, 50.0, 20.0, 30.0, 40.0, 60.0]).unwrap();
    let input = make_bounded(lower, upper);

    let layer = GatherLayer::new(0, None, vec![2]);
    let result = layer.propagate_ibp(&input).unwrap();

    // Dynamic: reduce axis 0 with min(lower) and max(upper)
    // Output shape: [2, 2] (axis 0 replaced by indices_shape [2])
    assert_eq!(result.shape(), &[2, 2]);
    // Min of lower along axis 0: col0: min(1,2,4)=1, col1: min(5,3,6)=3
    // These are broadcast to [2, 2]
    assert_eq!(result.lower()[[0, 0]], 1.0);
    assert_eq!(result.lower()[[0, 1]], 3.0);
    // Max of upper along axis 0: col0: max(10,20,40)=40, col1: max(50,30,60)=60
    assert_eq!(result.upper()[[0, 0]], 40.0);
    assert_eq!(result.upper()[[0, 1]], 60.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_dynamic_indices_non_finite_input_falls_back_to_infinite_bounds() {
    // Unchecked inputs can contain NaN; dynamic-index min/max must not
    // silently skip NaN and narrow the result.
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, f32::NAN, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let layer = GatherLayer::new(0, None, vec![2]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for (&l, &u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l.is_infinite() && l.is_sign_negative());
        assert!(u.is_infinite() && u.is_sign_positive());
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_build_output_to_input_map_rejects_output_shape_overflow_3012() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0i64]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![2, (usize::MAX / 2) + 1]);
    let err = layer
        .build_output_to_input_map(
            &[1],
            &[2, (usize::MAX / 2) + 1],
            0,
            layer.indices.as_ref().expect("static indices"),
        )
        .expect_err("overflowing output shape should fail before map construction");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("Gather output shape product overflows")),
        "expected gather output-shape overflow error, got: {err:?}"
    );
}

// =========================================================================
// Soundness: IBP gather output bounds contain actual gather output
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_soundness() {
    // For any concrete x in [lower, upper], gather(x) should be in [out_lower, out_upper].
    // Check all box vertices (2^n corners), not just midpoint.
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let input = make_bounded(lower.clone(), upper.clone());

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    let result = layer.propagate_ibp(&input).unwrap();

    for mask in 0..(1usize << 4) {
        let mut x = Vec::with_capacity(4);
        for dim in 0..4 {
            let choose_upper = (mask & (1 << dim)) != 0;
            x.push(if choose_upper {
                upper[[dim]]
            } else {
                lower[[dim]]
            });
        }

        let x_arr = ArrayD::from_shape_vec(IxDyn(&[4]), x).unwrap();
        let gathered_indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
        let gathered = layer
            .gather_with_indices(&x_arr, 0, &gathered_indices)
            .unwrap();

        for i in 0..gathered.len() {
            let y = gathered[[i]];
            assert!(
                y >= result.lower()[[i]],
                "corner mask={mask} index={i}: gathered {y} < lower {}",
                result.lower()[[i]]
            );
            assert!(
                y <= result.upper()[[i]],
                "corner mask={mask} index={i}: gathered {y} > upper {}",
                result.upper()[[i]]
            );
        }
    }
}

// =========================================================================
// Error cases
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_ibp_gather_out_of_bounds_index() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let input = make_bounded(lower, upper);

    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![5i64]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    assert!(layer.propagate_ibp(&input).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_propagate_linear_dynamic_indices_returns_error() {
    // Dynamic indices: CROWN backward not supported (no static index info).
    let mut layer = GatherLayer::new(0, None, vec![2]);
    layer.set_input_shape(vec![3]);
    let bounds = LinearBounds::new(
        Array2::eye(2),
        ndarray::Array1::zeros(2),
        Array2::eye(2),
        ndarray::Array1::zeros(2),
    )
    .unwrap();
    assert!(layer.propagate_linear(&bounds).is_err());
}

// =========================================================================
// CROWN backward propagation (#3400)
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_1d_gather() {
    // Input [4], gather indices [1, 3] -> output [2].
    // Backward should expand [2] coefficients to [4], scattering to positions 1 and 3.
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
    let mut layer = GatherLayer::new(0, Some(indices), vec![]);
    layer.set_input_shape(vec![4]);

    let bounds = LinearBounds::new(
        Array2::eye(2),
        ndarray::Array1::zeros(2),
        Array2::eye(2),
        ndarray::Array1::zeros(2),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_a.shape(), &[2, 4]);

    // Row 0 maps output[0] → input[1] (gathered index 1)
    assert_eq!(result.lower_a[[0, 0]], 0.0);
    assert_eq!(result.lower_a[[0, 1]], 1.0);
    assert_eq!(result.lower_a[[0, 2]], 0.0);
    assert_eq!(result.lower_a[[0, 3]], 0.0);
    // Row 1 maps output[1] → input[3] (gathered index 3)
    assert_eq!(result.lower_a[[1, 0]], 0.0);
    assert_eq!(result.lower_a[[1, 1]], 0.0);
    assert_eq!(result.lower_a[[1, 2]], 0.0);
    assert_eq!(result.lower_a[[1, 3]], 1.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_2d_gather_axis0() {
    // Input [3, 2], gather axis=0, indices=[0, 2] -> output [2, 2].
    // Backward: 4 output elements -> 6 input elements.
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    let mut layer = GatherLayer::new(0, Some(indices), vec![]);
    layer.set_input_shape(vec![3, 2]);

    let bounds = LinearBounds::new(
        Array2::eye(4),
        ndarray::Array1::zeros(4),
        Array2::eye(4),
        ndarray::Array1::zeros(4),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_a.shape(), &[4, 6]);

    // Output flat 0 -> multi [0,0] -> input [0, 0] -> flat 0
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    // Output flat 1 -> multi [0,1] -> input [0, 1] -> flat 1
    assert_eq!(result.lower_a[[1, 1]], 1.0);
    // Output flat 2 -> multi [1,0] -> input [2, 0] -> flat 4
    assert_eq!(result.lower_a[[2, 4]], 1.0);
    // Output flat 3 -> multi [1,1] -> input [2, 1] -> flat 5
    assert_eq!(result.lower_a[[3, 5]], 1.0);

    // Exactly 4 nonzero entries
    let nonzero: usize = result.lower_a.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(nonzero, 4);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_2d_gather_axis1() {
    // Input [2, 4], gather axis=1, indices=[1, 3] -> output [2, 2].
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
    let mut layer = GatherLayer::new(1, Some(indices), vec![]);
    layer.set_input_shape(vec![2, 4]);

    let bounds = LinearBounds::new(
        Array2::eye(4),
        ndarray::Array1::zeros(4),
        Array2::eye(4),
        ndarray::Array1::zeros(4),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_a.shape(), &[4, 8]);

    // Output flat 0 -> multi [0,0] -> input [0, 1] -> flat 1
    assert_eq!(result.lower_a[[0, 1]], 1.0);
    // Output flat 1 -> multi [0,1] -> input [0, 3] -> flat 3
    assert_eq!(result.lower_a[[1, 3]], 1.0);
    // Output flat 2 -> multi [1,0] -> input [1, 1] -> flat 5
    assert_eq!(result.lower_a[[2, 5]], 1.0);
    // Output flat 3 -> multi [1,1] -> input [1, 3] -> flat 7
    assert_eq!(result.lower_a[[3, 7]], 1.0);

    let nonzero: usize = result.lower_a.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(nonzero, 4);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_duplicate_indices_accumulates() {
    // Input [3], gather indices [1, 1] -> output [2].
    // Both outputs gather from the same input position — backward must
    // accumulate (+=) not overwrite.
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 1]).unwrap();
    let mut layer = GatherLayer::new(0, Some(indices), vec![]);
    layer.set_input_shape(vec![3]);

    // Non-identity bounds: row 0 has [2, 3], row 1 has [4, 5]
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
        ndarray::Array1::zeros(2),
        Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
        ndarray::Array1::zeros(2),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_a.shape(), &[2, 3]);

    // Row 0: out[0] -> in[1] with coeff 2.0, out[1] -> in[1] with coeff 3.0
    // Accumulated at in[1]: 2.0 + 3.0 = 5.0
    assert_eq!(result.lower_a[[0, 0]], 0.0);
    assert_eq!(result.lower_a[[0, 1]], 5.0);
    assert_eq!(result.lower_a[[0, 2]], 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_preserves_bias() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    let mut layer = GatherLayer::new(0, Some(indices), vec![]);
    layer.set_input_shape(vec![4]);

    let bounds = LinearBounds::new(
        Array2::eye(2),
        ndarray::Array1::from_vec(vec![1.0, 2.0]),
        Array2::eye(2),
        ndarray::Array1::from_vec(vec![3.0, 4.0]),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_b.as_slice().unwrap(), &[1.0, 2.0]);
    assert_eq!(result.upper_b.as_slice().unwrap(), &[3.0, 4.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_requires_input_shape() {
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 1]).unwrap();
    let layer = GatherLayer::new(0, Some(indices), vec![]);
    let bounds = LinearBounds::new(
        Array2::eye(2),
        ndarray::Array1::zeros(2),
        Array2::eye(2),
        ndarray::Array1::zeros(2),
    )
    .unwrap();
    // Should error: input_shape not set
    assert!(layer.propagate_linear(&bounds).is_err());
}

/// Fail-before / pass-after repro for #vnncomp-aw-soundness (Gather scatter-add).
/// k=21 duplicate indices all map to input column 0, so the backward f32-accumulates
/// 21 coeffs into one cell. With a 1.0 plus twenty 1e-8 terms, f32 round-to-nearest
/// loses the tiny terms (stored 1.0) while the true real sum is 1.0000002 — a stored
/// coefficient TIGHTER than the true value (false-proof) unless a certified error is
/// attached. After the fix Gather emits err = next_up(gamma_21^f32 * S), so
/// [stored-err, stored+err] encloses the true real coefficient.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_duplicate_accumulation_certified_error_encloses_true_sum() {
    let k = 21usize;
    let indices = ArrayD::from_shape_vec(IxDyn(&[k]), vec![0i64; k]).unwrap();
    let mut layer = GatherLayer::new(0, Some(indices), vec![]);
    layer.set_input_shape(vec![1]);
    let mut coeffs = vec![1.0f32];
    coeffs.extend(std::iter::repeat_n(1e-8f32, k - 1));
    let a = Array2::from_shape_vec((1, k), coeffs.clone()).unwrap();
    let bounds = LinearBounds::new(
        a.clone(),
        ndarray::Array1::zeros(1),
        a,
        ndarray::Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(result.lower_a().shape(), &[1, 1]);
    let true_sum: f64 = coeffs.iter().map(|&v| v as f64).sum();
    let stored = result.lower_a()[[0, 0]] as f64;
    let err_arr = result
        .lower_a_err()
        .expect("duplicate-index gather must carry a certified coefficient error");
    let err = err_arr[[0, 0]] as f64;
    assert!(
        stored < true_sum,
        "expected f32 accumulation to lose the tiny terms: stored {stored} vs true {true_sum}"
    );
    assert!(
        stored + err >= true_sum,
        "certified interval must enclose true real sum: stored {stored} + err {err} < true {true_sum}"
    );
}
