// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for linear (fully-connected) layers.

use super::*;
use crate::tests::assert_linear_bounds_close;
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, FALLBACK_BOUND};
use ny_test_utils::CountingGemmEngine;
use std::sync::atomic::{AtomicUsize, Ordering};

mod spectral_regressions;

/// Helper: create a small LinearLayer with known weights and bias.
/// W = [[1, 2], [3, 4]], b = [0.5, -0.5]
/// Maps R^2 → R^2: y = Wx + b
fn make_2x2_layer() -> LinearLayer {
    let weight = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let bias = array![0.5_f32, -0.5];
    LinearLayer::new(weight, Some(bias)).expect("valid layer")
}

/// Helper: create a LinearLayer without bias.
fn make_2x2_layer_no_bias() -> LinearLayer {
    let weight = array![[1.0_f32, 2.0], [3.0, 4.0]];
    LinearLayer::new(weight, None).expect("valid layer")
}

struct AlwaysFailGemmEngine;

impl GemmEngine for AlwaysFailGemmEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(NyError::NumericalInstability(
            "injected GEMM failure for linear CROWN fallback test".to_string(),
        ))
    }
}

struct FlushAllF32GemmEngine;

impl GemmEngine for FlushAllF32GemmEngine {
    fn gemm_f32(&self, m: usize, _k: usize, n: usize, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        Ok(vec![0.0; m * n])
    }
}

struct PanicGemmEngine;

impl GemmEngine for PanicGemmEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        panic!("finite-deadline Linear IBP must not enter the opaque engine")
    }
}

struct DeadlineOnGemmCallEngine {
    fail_on_call: usize,
    deadline_expired: bool,
    calls: AtomicUsize,
}

impl DeadlineOnGemmCallEngine {
    fn expired(fail_on_call: usize) -> Self {
        Self {
            fail_on_call,
            deadline_expired: true,
            calls: AtomicUsize::new(0),
        }
    }

    fn unscoped(fail_on_call: usize) -> Self {
        Self {
            fail_on_call,
            deadline_expired: false,
            calls: AtomicUsize::new(0),
        }
    }
}

impl GemmEngine for DeadlineOnGemmCallEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_on_call {
            return Err(NyError::DeadlineExceeded(format!(
                "injected batched Linear CROWN deadline on GEMM call {call}"
            )));
        }
        ny_core::NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn poll_crown_backward_deadline(&self) -> Result<()> {
        if self.deadline_expired {
            Err(NyError::DeadlineExceeded(
                "injected expired batched Linear CROWN proxy".into(),
            ))
        } else {
            Ok(())
        }
    }
}

struct BoundedMemoryRefusalEngine;

impl GemmEngine for BoundedMemoryRefusalEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(NyError::CpuMemoryExceeded {
            required_bytes: 2,
            budget_bytes: 1,
            site: "test bounded Linear CROWN engine",
        })
    }

    fn forbids_unbounded_cpu_fallback(&self) -> bool {
        true
    }
}

fn batched_crown_2x2_bounds() -> BatchedLinearBounds {
    BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2, 2])),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2, 2])),
        vec![2],
        vec![2],
    )
}

fn assert_multi_position_matches_split_positions(
    multi_result: &LinearBounds,
    pos0_result: &LinearBounds,
    pos1_result: &LinearBounds,
    in_features: usize,
    tolerance: f32,
    label: &str,
) {
    for row in 0..multi_result.num_outputs() {
        for col in 0..in_features {
            let first_block_col = col;
            let second_block_col = in_features + col;

            let lower_first = multi_result.lower_a()[[row, first_block_col]];
            let lower_second = multi_result.lower_a()[[row, second_block_col]];
            let upper_first = multi_result.upper_a()[[row, first_block_col]];
            let upper_second = multi_result.upper_a()[[row, second_block_col]];

            assert!(
                (lower_first - pos0_result.lower_a()[[row, col]]).abs() <= tolerance,
                "{label}: first lower block mismatch at row={row}, col={col}"
            );
            assert!(
                (lower_second - pos1_result.lower_a()[[row, col]]).abs() <= tolerance,
                "{label}: second lower block mismatch at row={row}, col={col}"
            );
            assert!(
                (upper_first - pos0_result.upper_a()[[row, col]]).abs() <= tolerance,
                "{label}: first upper block mismatch at row={row}, col={col}"
            );
            assert!(
                (upper_second - pos1_result.upper_a()[[row, col]]).abs() <= tolerance,
                "{label}: second upper block mismatch at row={row}, col={col}"
            );
        }

        let expected_lower_b = pos0_result.lower_b()[row] + pos1_result.lower_b()[row];
        let expected_upper_b = pos0_result.upper_b()[row] + pos1_result.upper_b()[row];
        assert!(
            (multi_result.lower_b()[row] - expected_lower_b).abs() <= tolerance,
            "{label}: lower bias mismatch at row {row}: actual={}, expected={expected_lower_b}",
            multi_result.lower_b()[row]
        );
        assert!(
            (multi_result.upper_b()[row] - expected_upper_b).abs() <= tolerance,
            "{label}: upper bias mismatch at row {row}: actual={}, expected={expected_upper_b}",
            multi_result.upper_b()[row]
        );
    }
}

fn assert_close_1em5(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "{label} expected {expected}, got {actual}"
    );
}

// ===== Constructor tests =====

#[ntest::timeout(10000)]
#[test]
fn test_linear_new_dimensions() {
    let layer = make_2x2_layer();
    assert_eq!(layer.in_features(), 2);
    assert_eq!(layer.out_features(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_new_bias_shape_mismatch() {
    let weight = array![[1.0_f32, 2.0], [3.0, 4.0]]; // 2x2
    let bias = array![1.0_f32, 2.0, 3.0]; // length 3, expected 2
    let err = LinearLayer::new(weight, Some(bias)).expect_err("bias shape mismatch");
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_from_dynamic() {
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap();
    let layer = LinearLayer::from_dynamic(&weight, Some(&bias)).expect("valid dynamic");
    assert_eq!(layer.in_features(), 3);
    assert_eq!(layer.out_features(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_from_dynamic_rejects_3d_weight() {
    let weight = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
    let err = LinearLayer::from_dynamic(&weight, None).expect_err("3D weight");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch for 3D weight, got: {err:?}"
    );
}

// ===== Spectral norm tests =====

#[ntest::timeout(10000)]
#[test]
fn test_spectral_norm_identity() {
    // Identity matrix has spectral norm 1.0.
    let weight = Array2::<f32>::eye(3);
    let layer = LinearLayer::new(weight, None).expect("valid");
    assert!(
        (layer.spectral_norm() - 1.0).abs() < 1e-3,
        "identity spectral norm should be ~1.0, got {}",
        layer.spectral_norm()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_spectral_norm_scaled_identity() {
    // 5*I has spectral norm 5.0.
    let weight = Array2::<f32>::eye(3) * 5.0;
    let layer = LinearLayer::new(weight, None).expect("valid");
    assert!(
        (layer.spectral_norm() - 5.0).abs() < 1e-3,
        "5*I spectral norm should be ~5.0, got {}",
        layer.spectral_norm()
    );
}

// ===== IBP 1D tests =====

#[ntest::timeout(10000)]
#[test]
fn test_ibp_1d_point_interval() -> Result<()> {
    // For a point interval [x, x], IBP should give [Wx+b, Wx+b]
    let layer = make_2x2_layer();
    let x = array![1.0_f32, 2.0];
    let input = BoundedTensor::new(x.clone().into_dyn(), x.into_dyn())?;
    let output = layer.propagate_ibp(&input)?;

    // y = Wx + b = [[1,2],[3,4]] @ [1,2] + [0.5, -0.5] = [5, 11] + [0.5, -0.5] = [5.5, 10.5]
    let expected = array![5.5_f32, 10.5];
    for i in 0..2 {
        assert!(
            (output.lower()[i] - expected[i]).abs() < 1e-5,
            "lower[{i}] = {}, expected {}",
            output.lower()[i],
            expected[i]
        );
        assert!(
            (output.upper()[i] - expected[i]).abs() < 1e-5,
            "upper[{i}] = {}, expected {}",
            output.upper()[i],
            expected[i]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_1d_soundness() -> Result<()> {
    // Verify that IBP bounds contain the true output for sampled points
    let layer = make_2x2_layer();
    let lower = array![-1.0_f32, -2.0];
    let upper = array![1.0_f32, 2.0];
    let input = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn())?;
    let output = layer.propagate_ibp(&input)?;

    // Sample 100 points in [l, u] and verify bounds contain Wx+b
    for k in 0..100 {
        let t = k as f32 / 99.0;
        let x0 = lower[0] + t * (upper[0] - lower[0]);
        let x1 = lower[1] + (1.0 - t) * (upper[1] - lower[1]);
        // y = Wx + b
        let y0 = 1.0 * x0 + 2.0 * x1 + 0.5;
        let y1 = 3.0 * x0 + 4.0 * x1 - 0.5;

        assert!(
            output.lower()[0] <= y0 + 1e-5,
            "lower[0]={} > y0={} at t={}",
            output.lower()[0],
            y0,
            t
        );
        assert!(
            output.upper()[0] >= y0 - 1e-5,
            "upper[0]={} < y0={} at t={}",
            output.upper()[0],
            y0,
            t
        );
        assert!(
            output.lower()[1] <= y1 + 1e-5,
            "lower[1]={} > y1={} at t={}",
            output.lower()[1],
            y1,
            t
        );
        assert!(
            output.upper()[1] >= y1 - 1e-5,
            "upper[1]={} < y1={} at t={}",
            output.upper()[1],
            y1,
            t
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_1d_preserves_large_finite_bounds_above_fallback_2549() -> Result<()> {
    // #2549: finite values above FALLBACK_BOUND must be preserved (no narrowing clamp).
    let weight = array![[2.0e10_f32]];
    let layer = LinearLayer::new(weight, None)?;
    let input = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn())?;
    let output = layer.propagate_ibp(&input)?;

    let lo = output.lower()[[0]];
    let hi = output.upper()[[0]];
    assert!(lo.is_finite(), "lower must stay finite, got {lo}");
    assert!(hi.is_finite(), "upper must stay finite, got {hi}");
    assert!(lo <= -2.0e10_f32, "lower should preserve -2e10, got {lo}");
    assert!(hi >= 2.0e10_f32, "upper should preserve 2e10, got {hi}");
    assert!(lo <= -FALLBACK_BOUND, "lower must not be clamped to -1e10");
    assert!(hi >= FALLBACK_BOUND, "upper must not be clamped to 1e10");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_1d_no_bias() -> Result<()> {
    let layer = make_2x2_layer_no_bias();
    let x = array![1.0_f32, 2.0];
    let input = BoundedTensor::new(x.clone().into_dyn(), x.into_dyn())?;
    let output = layer.propagate_ibp(&input)?;

    // y = Wx = [[1,2],[3,4]] @ [1,2] = [5, 11]
    assert!(
        (output.lower()[0] - 5.0).abs() < 1e-5,
        "lower[0] expected 5.0, got {}",
        output.lower()[0]
    );
    assert!(
        (output.upper()[0] - 5.0).abs() < 1e-5,
        "upper[0] expected 5.0, got {}",
        output.upper()[0]
    );
    assert!(
        (output.lower()[1] - 11.0).abs() < 1e-5,
        "lower[1] expected 11.0, got {}",
        output.lower()[1]
    );
    assert!(
        (output.upper()[1] - 11.0).abs() < 1e-5,
        "upper[1] expected 11.0, got {}",
        output.upper()[1]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_1d_shape_mismatch() {
    let layer = make_2x2_layer(); // expects in_features=2
    let lower = array![1.0_f32, 2.0, 3.0].into_dyn(); // 3 features
    let upper = array![4.0_f32, 5.0, 6.0].into_dyn();
    let input = BoundedTensor::new(lower, upper).expect("bounds construction");
    let err = layer.propagate_ibp(&input).expect_err("shape mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch for 3-feature input on 2-feature layer, got: {err:?}"
    );
}

// ===== IBP batched (N-D) tests =====

#[ntest::timeout(10000)]
#[test]
fn test_ibp_batched_2d() -> Result<()> {
    // Input shape [batch=2, in_features=2], output shape [batch=2, out_features=2]
    // W = [[1,2],[3,4]], bias = [0.5, -0.5]. All weights positive.
    let layer = make_2x2_layer();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -2.0, 0.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[2, 2]);
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "lower {} > upper {}", l, u);
    }

    // Batch 0: input [-1,-2]..[1,2]
    // lower_out = W@[-1,-2]+bias = [-5,-11]+[0.5,-0.5] = [-4.5,-11.5]
    // upper_out = W@[1,2]+bias = [5,11]+[0.5,-0.5] = [5.5,10.5]
    let v = output.lower()[[0, 0]];
    assert!(
        (v - (-4.5)).abs() < 1e-5,
        "batch0 lower[0] expected -4.5, got {v}"
    );
    let v = output.upper()[[0, 0]];
    assert!(
        (v - 5.5).abs() < 1e-5,
        "batch0 upper[0] expected 5.5, got {v}"
    );
    let v = output.lower()[[0, 1]];
    assert!(
        (v - (-11.5)).abs() < 1e-5,
        "batch0 lower[1] expected -11.5, got {v}"
    );
    let v = output.upper()[[0, 1]];
    assert!(
        (v - 10.5).abs() < 1e-5,
        "batch0 upper[1] expected 10.5, got {v}"
    );

    // Batch 1: input [0,0]..[1,1]
    // lower_out = W@[0,0]+bias = [0.5,-0.5]
    // upper_out = W@[1,1]+bias = [3,7]+[0.5,-0.5] = [3.5,6.5]
    let v = output.lower()[[1, 0]];
    assert!(
        (v - 0.5).abs() < 1e-5,
        "batch1 lower[0] expected 0.5, got {v}"
    );
    let v = output.upper()[[1, 0]];
    assert!(
        (v - 3.5).abs() < 1e-5,
        "batch1 upper[0] expected 3.5, got {v}"
    );
    let v = output.lower()[[1, 1]];
    assert!(
        (v - (-0.5)).abs() < 1e-5,
        "batch1 lower[1] expected -0.5, got {v}"
    );
    let v = output.upper()[[1, 1]];
    assert!(
        (v - 6.5).abs() < 1e-5,
        "batch1 upper[1] expected 6.5, got {v}"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_batched_3d() -> Result<()> {
    // Input shape [2, 3, 2] → output shape [2, 3, 2]
    // W = [[1,2],[3,4]], bias = [0.5, -0.5]. Uniform input [-1, 1].
    let layer = make_2x2_layer();
    let lower = ArrayD::from_elem(IxDyn(&[2, 3, 2]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, 2]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[2, 3, 2]);
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "lower {} > upper {}", l, u);
    }

    // Every slice has the same input range [-1,1]×[-1,1], so all outputs match.
    // lower_out = W@[-1,-1]+bias = [-3,-7]+[0.5,-0.5] = [-2.5,-7.5]
    // upper_out = W@[1,1]+bias = [3,7]+[0.5,-0.5] = [3.5,6.5]
    let v = output.lower()[[0, 0, 0]];
    assert!(
        (v - (-2.5)).abs() < 1e-5,
        "lower[0,0,0] expected -2.5, got {v}"
    );
    let v = output.upper()[[0, 0, 0]];
    assert!((v - 3.5).abs() < 1e-5, "upper[0,0,0] expected 3.5, got {v}");
    let v = output.lower()[[0, 0, 1]];
    assert!(
        (v - (-7.5)).abs() < 1e-5,
        "lower[0,0,1] expected -7.5, got {v}"
    );
    let v = output.upper()[[0, 0, 1]];
    assert!((v - 6.5).abs() < 1e-5, "upper[0,0,1] expected 6.5, got {v}");
    // Spot-check another batch element — should be identical
    let v = output.lower()[[1, 2, 0]];
    assert!(
        (v - (-2.5)).abs() < 1e-5,
        "lower[1,2,0] expected -2.5, got {v}"
    );
    let v = output.upper()[[1, 2, 1]];
    assert!((v - 6.5).abs() < 1e-5, "upper[1,2,1] expected 6.5, got {v}");
    Ok(())
}

// ===== Sound IBP tests =====

#[ntest::timeout(10000)]
#[test]
fn test_ibp_sound_wider_than_ibp() -> Result<()> {
    let layer = make_2x2_layer();
    let lower = array![-1.0_f32, -2.0].into_dyn();
    let upper = array![1.0_f32, 2.0].into_dyn();
    let input = BoundedTensor::new(lower, upper)?;

    let ibp = layer.propagate_ibp(&input)?;
    let sound = layer.propagate_ibp_sound(&input)?;

    // Sound bounds should be at least as wide as IBP (they add ULP rounding)
    for i in 0..2 {
        assert!(
            sound.lower()[i] <= ibp.lower()[i] + 1e-10,
            "sound lower should be <= ibp lower"
        );
        assert!(
            sound.upper()[i] >= ibp.upper()[i] - 1e-10,
            "sound upper should be >= ibp upper"
        );
    }
    Ok(())
}

// ===== CROWN backward tests =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_identity_bounds() -> Result<()> {
    // With identity linear bounds (A=I, b=0), CROWN backward through linear
    // should give new_A = W, new_b = bias.
    let layer = make_2x2_layer();
    let identity_bounds = LinearBounds::identity(2);

    let result = layer.propagate_linear(&identity_bounds)?;
    let result = result.into_owned();

    // new_A = I @ W = W = [[1,2],[3,4]]
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "A[0,0] should be 1.0"
    );
    assert!(
        (result.lower_a[[0, 1]] - 2.0).abs() < 1e-5,
        "A[0,1] should be 2.0"
    );
    assert!(
        (result.lower_a[[1, 0]] - 3.0).abs() < 1e-5,
        "A[1,0] should be 3.0"
    );
    assert!(
        (result.lower_a[[1, 1]] - 4.0).abs() < 1e-5,
        "A[1,1] should be 4.0"
    );

    // new_b = I @ bias + 0 = [0.5, -0.5]
    assert!((result.lower_b[0] - 0.5).abs() < 1e-5, "b[0] should be 0.5");
    assert!(
        (result.lower_b[1] - (-0.5)).abs() < 1e-5,
        "b[1] should be -0.5"
    );

    // Upper bounds should match lower bounds for identity
    assert!(
        (result.upper_a[[0, 0]] - 1.0).abs() < 1e-5,
        "upper_a[0,0] expected 1.0, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0] - 0.5).abs() < 1e-5,
        "upper_b[0] expected 0.5, got {}",
        result.upper_b[0]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_no_bias() -> Result<()> {
    let layer = make_2x2_layer_no_bias();
    let identity_bounds = LinearBounds::identity(2);

    let result = layer.propagate_linear(&identity_bounds)?;
    let result = result.into_owned();

    // new_A = I @ W = W
    let v = result.lower_a[[0, 0]];
    assert!((v - 1.0).abs() < 1e-5, "lower_a[0,0] expected 1.0, got {v}");
    let v = result.lower_a[[0, 1]];
    assert!((v - 2.0).abs() < 1e-5, "lower_a[0,1] expected 2.0, got {v}");

    // new_b = 0 (no bias)
    assert!((result.lower_b[0]).abs() < 1e-5, "no bias → b=0");
    assert!((result.lower_b[1]).abs() < 1e-5, "no bias → b=0");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_shape_mismatch() {
    let layer = make_2x2_layer(); // out_features=2
                                  // LinearBounds with wrong input dimension (3 instead of 2)
    let bad_bounds = LinearBounds::new(
        Array2::zeros((2, 3)),
        Array1::zeros(2),
        Array2::zeros((2, 3)),
        Array1::zeros(2),
    )
    .unwrap();
    let err = layer
        .propagate_linear(&bad_bounds)
        .expect_err("shape mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch for CROWN with wrong A-matrix columns, got: {err:?}"
    );
}

// ===== #2709: GEMM-engine and sequence-dimension linear CROWN coverage =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_single_engine_matches_cpu_2709() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds = LinearBounds::new(
        array![[1.0_f32, -1.0], [0.5, 0.5]],
        array![0.25_f32, -0.75],
        array![[2.0_f32, 0.0], [-0.5, 1.5]],
        array![0.5_f32, 0.25],
    )?;

    let expected = layer.propagate_linear(&bounds)?.into_owned();
    let engine = CountingGemmEngine::new();
    let actual = layer
        .propagate_linear_with_engine(&bounds, Some(&engine))?
        .into_owned();

    assert!(
        engine.gemm_calls() >= 2,
        "#2709 single-domain GEMM path should invoke the engine for lower/upper coefficients"
    );
    assert_linear_bounds_close(&actual, &expected, 1e-6, "#2709 single-domain engine");
    Ok(())
}

#[test]
fn test_crown_single_engine_certificate_covers_daz_flushed_input() -> Result<()> {
    let exact = 2.0_f64.powi(-29);
    let layer = LinearLayer::new(array![[2.0_f32.powi(120)]], None)?;
    let tiny = f32::from_bits(1);
    let bounds = LinearBounds::new(array![[tiny]], array![0.0], array![[tiny]], array![0.0])?;

    let result = layer
        .propagate_linear_with_engine(&bounds, Some(&FlushAllF32GemmEngine))?
        .into_owned();
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.upper_a()[[0, 0]], 0.0);
    assert!(f64::from(result.lower_a_err().expect("lower certificate")[[0, 0]]) >= exact);
    assert!(f64::from(result.upper_a_err().expect("upper certificate")[[0, 0]]) >= exact);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_single_engine_nonfinite_row_fallback_matches_cpu_2709() -> Result<()> {
    let weight = array![[1e20_f32, 1.0], [1.0, 1.0]];
    let layer = LinearLayer::new(weight, None)?;
    let bounds = LinearBounds::new(
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )?;

    let expected = layer.propagate_linear(&bounds)?.into_owned();
    let engine = CountingGemmEngine::new();
    let actual = layer
        .propagate_linear_with_engine(&bounds, Some(&engine))?
        .into_owned();

    assert!(
        engine.gemm_calls() >= 2,
        "#2709 non-finite fallback test must exercise the GEMM path"
    );
    assert_linear_bounds_close(&actual, &expected, 1e-6, "#2709 non-finite GEMM fallback");
    assert_eq!(
        actual.lower_b()[0],
        f32::NEG_INFINITY,
        "#2709 overflow row should keep the conservative lower bias"
    );
    assert_eq!(
        actual.upper_b()[0],
        f32::INFINITY,
        "#2709 overflow row should keep the conservative upper bias"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_single_engine_error_falls_back_to_cpu_2709() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds = LinearBounds::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        array![0.25_f32, -0.75],
        array![[1.5_f32, -0.5], [0.25, 1.75]],
        array![0.5_f32, 0.125],
    )?;

    let expected = layer.propagate_linear(&bounds)?.into_owned();
    let actual = layer
        .propagate_linear_with_engine(&bounds, Some(&AlwaysFailGemmEngine))?
        .into_owned();

    assert_linear_bounds_close(&actual, &expected, 1e-6, "#2709 GEMM failure fallback");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_multi_position_backward_exact_2709() -> Result<()> {
    let layer = LinearLayer::new(
        array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
        Some(array![0.5_f32, 1.5]),
    )?;
    let bounds = LinearBounds::new(
        array![[1.0_f32, 0.0, 0.0, 1.0], [0.5, 0.5, -1.0, 2.0]],
        array![0.25_f32, -0.75],
        array![[1.0_f32, 0.0, 0.0, 1.0], [0.5, 0.5, -1.0, 2.0]],
        array![0.25_f32, -0.75],
    )?;

    let result = layer.propagate_linear(&bounds)?.into_owned();
    let expected = LinearBounds::new(
        array![
            [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.5, 3.5, 4.5, 7.0, 8.0, 9.0]
        ],
        array![2.25_f32, 2.75],
        array![
            [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.5, 3.5, 4.5, 7.0, 8.0, 9.0]
        ],
        array![2.25_f32, 2.75],
    )?;

    assert_linear_bounds_close(&result, &expected, 1e-6, "#2709 multi-position exact");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_multi_position_engine_matches_split_single_position_2709() -> Result<()> {
    let layer = make_2x2_layer();
    let multi_bounds = LinearBounds::new(
        array![[1.0_f32, -1.0, 0.5, 0.25], [0.0, 2.0, -1.0, 1.5]],
        Array1::zeros(2),
        array![[1.5_f32, 0.5, 0.25, -0.75], [1.0, -1.0, 0.5, 0.5]],
        Array1::zeros(2),
    )?;
    let pos0_bounds = LinearBounds::new(
        array![[1.0_f32, -1.0], [0.0, 2.0]],
        Array1::zeros(2),
        array![[1.5_f32, 0.5], [1.0, -1.0]],
        Array1::zeros(2),
    )?;
    let pos1_bounds = LinearBounds::new(
        array![[0.5_f32, 0.25], [-1.0, 1.5]],
        Array1::zeros(2),
        array![[0.25_f32, -0.75], [0.5, 0.5]],
        Array1::zeros(2),
    )?;

    let engine = CountingGemmEngine::new();
    let multi_result = layer
        .propagate_linear_with_engine(&multi_bounds, Some(&engine))?
        .into_owned();
    let pos0_result = layer.propagate_linear(&pos0_bounds)?.into_owned();
    let pos1_result = layer.propagate_linear(&pos1_bounds)?.into_owned();

    assert!(
        engine.gemm_calls() >= 4,
        "#2709 multi-position GEMM path should invoke the engine once per direction and position"
    );
    assert_multi_position_matches_split_positions(
        &multi_result,
        &pos0_result,
        &pos1_result,
        layer.in_features(),
        1e-6,
        "#2709 multi-position split equivalence",
    );

    Ok(())
}

// ===== CROWN backward consistency: IBP bounds must contain CROWN concretized bounds =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_concretized_within_ibp() -> Result<()> {
    // CROWN bounds concretized over the input domain should be at least as tight as IBP,
    // and both should be sound (contain true outputs).
    let layer = make_2x2_layer();
    let lower = array![-1.0_f32, -0.5];
    let upper = array![0.5_f32, 1.0];
    let input = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn())?;

    // IBP bounds
    let ibp = layer.propagate_ibp(&input)?;

    // CROWN backward: start with identity bounds at output
    let identity_bounds = LinearBounds::identity(2);
    let crown = layer.propagate_linear(&identity_bounds)?;
    let crown = crown.into_owned();

    // Concretize CROWN: for each output i,
    // crown_lower_i = sum(max(A_i,0)*l + min(A_i,0)*u) + b_i
    // crown_upper_i = sum(max(A_i,0)*u + min(A_i,0)*l) + b_i
    for i in 0..2 {
        let mut crown_lower = crown.lower_b[i];
        let mut crown_upper = crown.upper_b[i];
        for j in 0..2 {
            let la = crown.lower_a[[i, j]];
            let ua = crown.upper_a[[i, j]];
            crown_lower += la.max(0.0) * lower[j] + la.min(0.0) * upper[j];
            crown_upper += ua.max(0.0) * upper[j] + ua.min(0.0) * lower[j];
        }

        // CROWN concretized should be within IBP bounds (CROWN is at least as tight)
        assert!(
            crown_lower >= ibp.lower()[i] - 1e-4,
            "CROWN lower {} < IBP lower {} for output {}",
            crown_lower,
            ibp.lower()[i],
            i
        );
        assert!(
            crown_upper <= ibp.upper()[i] + 1e-4,
            "CROWN upper {} > IBP upper {} for output {}",
            crown_upper,
            ibp.upper()[i],
            i
        );
    }
    Ok(())
}

// ===== Batched CROWN backward tests =====

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_basic() -> Result<()> {
    let layer = make_2x2_layer();
    // BatchedLinearBounds with shape [out_dim=2, mid_dim=2] (no batch dims)
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![1.0, 0.0, 0.0, 1.0], // identity
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![1.0, 0.0, 0.0, 1.0], // identity
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let result = layer.propagate_linear_batched(&bounds)?;

    // new_A = I @ W = W, shape should be [2, 2]
    assert_eq!(result.lower_a.shape(), &[2, 2]);
    let v = result.lower_a[[0, 0]];
    assert!((v - 1.0).abs() < 1e-5, "lower_a[0,0] expected 1.0, got {v}");
    let v = result.lower_a[[0, 1]];
    assert!((v - 2.0).abs() < 1e-5, "lower_a[0,1] expected 2.0, got {v}");
    let v = result.lower_a[[1, 0]];
    assert!((v - 3.0).abs() < 1e-5, "lower_a[1,0] expected 3.0, got {v}");
    let v = result.lower_a[[1, 1]];
    assert!((v - 4.0).abs() < 1e-5, "lower_a[1,1] expected 4.0, got {v}");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_with_batch_dim() -> Result<()> {
    let layer = make_2x2_layer();
    // BatchedLinearBounds with shape [batch=2, out_dim=2, mid_dim=2]
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            // batch 0: identity, batch 1: scaled identity
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2, 2])),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap(),
        ArrayD::zeros(IxDyn(&[2, 2])),
        vec![2],
        vec![2],
    );
    let result = layer.propagate_linear_batched(&bounds)?;

    assert_eq!(result.lower_a.shape(), &[2, 2, 2]);

    // batch 0: I @ W = W
    let v = result.lower_a[[0, 0, 0]];
    assert!(
        (v - 1.0).abs() < 1e-5,
        "batch0 lower_a[0,0] expected 1.0, got {v}"
    );
    let v = result.lower_a[[0, 0, 1]];
    assert!(
        (v - 2.0).abs() < 1e-5,
        "batch0 lower_a[0,1] expected 2.0, got {v}"
    );

    // batch 1: 2I @ W = 2W
    let v = result.lower_a[[1, 0, 0]];
    assert!(
        (v - 2.0).abs() < 1e-5,
        "batch1 lower_a[0,0] expected 2.0, got {v}"
    );
    let v = result.lower_a[[1, 0, 1]];
    assert!(
        (v - 4.0).abs() < 1e-5,
        "batch1 lower_a[0,1] expected 4.0, got {v}"
    );
    let v = result.lower_a[[1, 1, 0]];
    assert!(
        (v - 6.0).abs() < 1e-5,
        "batch1 lower_a[1,0] expected 6.0, got {v}"
    );
    let v = result.lower_a[[1, 1, 1]];
    assert!(
        (v - 8.0).abs() < 1e-5,
        "batch1 lower_a[1,1] expected 8.0, got {v}"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_engine_matches_cpu_3597() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds = batched_crown_2x2_bounds();

    let expected = layer.propagate_linear_batched(&bounds)?;
    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched_maybe_engine(&bounds, Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "batched linear CROWN engine path should invoke GemmEngine, got {calls} calls"
    );

    for (idx, (&actual_value, &expected_value)) in actual
        .lower_a()
        .iter()
        .zip(expected.lower_a().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "lower_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_a()
        .iter()
        .zip(expected.upper_a().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "upper_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .lower_b()
        .iter()
        .zip(expected.lower_b().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "lower_b mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_b()
        .iter()
        .zip(expected.upper_b().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "upper_b mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }

    Ok(())
}

#[test]
fn batched_linear_crown_preserves_expired_proxy_deadline_from_either_gemm() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds = batched_crown_2x2_bounds();

    for fail_on_call in [1, 2] {
        let engine = DeadlineOnGemmCallEngine::expired(fail_on_call);
        let error = layer
            .propagate_linear_batched_maybe_engine(&bounds, Some(&engine))
            .expect_err("expired proxy deadline must be terminal");
        assert!(
            error.is_deadline_exceeded(),
            "GEMM call {fail_on_call} returned the wrong error: {error}"
        );
        assert_eq!(
            engine.calls.load(Ordering::SeqCst),
            fail_on_call,
            "batched Linear CROWN must stop immediately at the typed deadline"
        );
    }

    Ok(())
}

#[test]
fn batched_linear_crown_none_authority_keeps_engine_error_cpu_fallback() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds = batched_crown_2x2_bounds();
    let expected = layer.propagate_linear_batched(&bounds)?;

    let assert_exact_cpu = |actual: &BatchedLinearBounds| {
        assert_eq!(actual.lower_a, expected.lower_a);
        assert_eq!(actual.upper_a, expected.upper_a);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_b, expected.upper_b);
        assert_eq!(actual.lower_a_err, expected.lower_a_err);
        assert_eq!(actual.upper_a_err, expected.upper_a_err);
    };

    let ordinary =
        layer.propagate_linear_batched_maybe_engine(&bounds, Some(&AlwaysFailGemmEngine))?;
    assert_exact_cpu(&ordinary);

    for fail_on_call in [1, 2] {
        let engine = DeadlineOnGemmCallEngine::unscoped(fail_on_call);
        let actual = layer.propagate_linear_batched_maybe_engine(&bounds, Some(&engine))?;
        assert_exact_cpu(&actual);
        assert_eq!(
            engine.calls.load(Ordering::SeqCst),
            fail_on_call,
            "unscoped typed error must enter CPU fallback without retrying the engine"
        );
    }

    Ok(())
}

#[test]
fn bounded_linear_engine_entries_fail_closed_without_pollable_implementation() {
    let layer = make_2x2_layer();
    let batched = batched_crown_2x2_bounds();
    let scalar = LinearBounds::identity(2);
    let input = BoundedTensor::concrete(array![0.0_f32, 0.0].into_dyn()).unwrap();

    let error = layer
        .propagate_linear_batched_maybe_engine(&batched, Some(&BoundedMemoryRefusalEngine))
        .expect_err("bounded batched CROWN must fail before any opaque work");
    assert!(matches!(error, NyError::UnsupportedOp(_)));

    let error = layer
        .propagate_linear_batched_with_engine(&[&scalar], &BoundedMemoryRefusalEngine)
        .expect_err("bounded multi-domain CROWN must fail before any opaque work");
    assert!(matches!(error, NyError::UnsupportedOp(_)));

    let error = layer
        .propagate_linear_with_engine(&scalar, Some(&BoundedMemoryRefusalEngine))
        .expect_err("bounded unscoped scalar CROWN must not fall back to CPU");
    assert!(matches!(error, NyError::UnsupportedOp(_)));

    let error = layer
        .propagate_linear_with_engine_and_deadline(
            &scalar,
            Some(&BoundedMemoryRefusalEngine),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(1)),
        )
        .expect_err("a partial bounded scalar capability must fail closed");
    assert!(matches!(error, NyError::UnsupportedOp(_)));

    let error = layer
        .propagate_ibp_with_engine(&input, Some(&BoundedMemoryRefusalEngine))
        .expect_err("bounded unscoped Linear IBP must not fall back to CPU");
    assert!(matches!(error, NyError::UnsupportedOp(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_multi_domain_engine_matches_cpu_2709() -> Result<()> {
    let layer = make_2x2_layer();
    let bounds_a = LinearBounds::identity(2);
    let bounds_b = LinearBounds::new(
        array![[1.0_f32, -1.0], [0.5, 0.5]],
        array![0.25_f32, -0.75],
        array![[2.0_f32, 0.0], [-0.5, 1.5]],
        array![0.5_f32, 0.25],
    )?;

    let expected_a = layer.propagate_linear(&bounds_a)?.into_owned();
    let expected_b = layer.propagate_linear(&bounds_b)?.into_owned();

    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched_with_engine(&[&bounds_a, &bounds_b], &engine)?;

    assert_eq!(actual.len(), 2, "#2709 expected two multi-domain results");
    assert!(
        engine.gemm_calls() >= 2,
        "#2709 multi-domain GEMM path should invoke the engine"
    );
    assert_linear_bounds_close(&actual[0], &expected_a, 1e-6, "#2709 multi-domain domain 0");
    assert_linear_bounds_close(&actual[1], &expected_b, 1e-6, "#2709 multi-domain domain 1");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_multi_domain_engine_nonfinite_fallback_matches_cpu_2709() -> Result<()> {
    let weight = array![[1e20_f32, 1.0], [1.0, 1.0]];
    let layer = LinearLayer::new(weight, None)?;
    let overflow_bounds = LinearBounds::new(
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )?;
    let finite_bounds = LinearBounds::identity(2);

    let expected_overflow = layer.propagate_linear(&overflow_bounds)?.into_owned();
    let expected_finite = layer.propagate_linear(&finite_bounds)?.into_owned();

    let engine = CountingGemmEngine::new();
    let actual =
        layer.propagate_linear_batched_with_engine(&[&overflow_bounds, &finite_bounds], &engine)?;

    assert_eq!(actual.len(), 2, "#2709 expected two multi-domain results");
    assert!(
        engine.gemm_calls() >= 2,
        "#2709 multi-domain non-finite test must exercise the GEMM path"
    );
    assert_linear_bounds_close(
        &actual[0],
        &expected_overflow,
        1e-6,
        "#2709 multi-domain overflow domain",
    );
    assert_linear_bounds_close(
        &actual[1],
        &expected_finite,
        1e-6,
        "#2709 multi-domain finite domain",
    );
    assert_eq!(
        actual[0].lower_b()[0],
        f32::NEG_INFINITY,
        "#2709 overflow domain should keep the conservative lower bias"
    );
    assert_eq!(
        actual[0].upper_b()[0],
        f32::INFINITY,
        "#2709 overflow domain should keep the conservative upper bias"
    );
    Ok(())
}

/// Post-#2977: from_parts_unchecked now has debug_assert rejecting NaN in
/// A-matrices. The sanitization behavior tested here (zeroing non-finite rows,
/// ±inf bias fallback) is now only reachable in release mode where debug_assert
/// is elided. In debug mode, constructing BatchedLinearBounds with NaN panics.
#[ntest::timeout(10000)]
// The guard is a debug_assert — it cannot fire in release builds.
#[cfg(debug_assertions)]
#[test]
fn test_batched_crown_sanitizes_non_finite_faer_output() -> Result<()> {
    // #2977 from_parts_unchecked debug_assert blocks NaN injection in debug builds.
    // This test verifies the guard fires.
    let result = std::panic::catch_unwind(|| {
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, f32::NAN, 1.0]).unwrap(),
            ArrayD::zeros(IxDyn(&[2])),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, f32::NEG_INFINITY, 1.0]).unwrap(),
            ArrayD::zeros(IxDyn(&[2])),
            vec![2],
            vec![2],
        )
    });
    assert!(
        result.is_err(),
        "from_parts_unchecked should panic on NaN in debug builds (#2977)"
    );
    Ok(())
}

// ===== CROWN backward: non-identity coefficient matrices =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_non_identity_coefficients() -> Result<()> {
    // CROWN backward with A = [[1, -1], [0.5, 0.5]] (non-identity, non-square)
    // For y = Wx + b, propagation gives new_A = A @ W, new_b = A @ b + old_b
    let layer = make_2x2_layer(); // W = [[1,2],[3,4]], b = [0.5, -0.5]
    let a_matrix = array![[1.0_f32, -1.0], [0.5, 0.5]];
    let bounds = LinearBounds::new(
        a_matrix.clone(),
        Array1::zeros(2),
        a_matrix,
        Array1::zeros(2),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds)?;
    let result = result.into_owned();

    // new_A = A @ W = [[1,-1],[0.5,0.5]] @ [[1,2],[3,4]]
    //       = [[1*1+(-1)*3, 1*2+(-1)*4], [0.5*1+0.5*3, 0.5*2+0.5*4]]
    //       = [[-2, -2], [2, 3]]
    assert!((result.lower_a[[0, 0]] - (-2.0)).abs() < 1e-5, "A[0,0]");
    assert!((result.lower_a[[0, 1]] - (-2.0)).abs() < 1e-5, "A[0,1]");
    assert!((result.lower_a[[1, 0]] - 2.0).abs() < 1e-5, "A[1,0]");
    assert!((result.lower_a[[1, 1]] - 3.0).abs() < 1e-5, "A[1,1]");

    // new_b = A @ b = [[1,-1],[0.5,0.5]] @ [0.5,-0.5] = [1.0, 0.0]
    assert!((result.lower_b[0] - 1.0).abs() < 1e-5, "b[0]");
    assert!(result.lower_b[1].abs() < 1e-5, "b[1]");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_asymmetric_lower_upper() -> Result<()> {
    // Different lower_a and upper_a — CROWN backward should propagate them independently.
    // For linear layers, lower and upper bounds compose identically with W,
    // but the bias contribution differs if lower_b != upper_b.
    let layer = make_2x2_layer(); // W = [[1,2],[3,4]], b = [0.5, -0.5]
    let bounds = LinearBounds::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        array![0.0_f32, 0.0],
        array![[2.0_f32, 0.0], [0.0, 2.0]],
        array![1.0_f32, 1.0],
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds)?;
    let result = result.into_owned();

    // lower: new_A_L = I @ W = W, new_b_L = I @ b + 0 = b = [0.5, -0.5]
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "lower_a[0,0] expected 1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-5,
        "lower_b[0] expected 0.5, got {}",
        result.lower_b[0]
    );

    // upper: new_A_U = 2I @ W = 2W, new_b_U = 2I @ b + 1 = 2b + 1
    // 2*[0.5, -0.5] + [1, 1] = [2.0, 0.0]
    assert!((result.upper_a[[0, 0]] - 2.0).abs() < 1e-5, "2*W[0,0]");
    assert!((result.upper_a[[0, 1]] - 4.0).abs() < 1e-5, "2*W[0,1]");
    assert!((result.upper_b[0] - 2.0).abs() < 1e-5, "2*b[0]+1");
    assert!(result.upper_b[1].abs() < 1e-5, "2*b[1]+1");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_negative_weights() -> Result<()> {
    // Linear layer with mixed-sign weights: W = [[-1, 2], [3, -4]]
    let weight = array![[-1.0_f32, 2.0], [3.0, -4.0]];
    let bias = array![1.0_f32, -1.0];
    let layer = LinearLayer::new(weight, Some(bias))?;
    let identity_bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&identity_bounds)?;
    let result = result.into_owned();

    // new_A = I @ W = W = [[-1,2],[3,-4]]
    assert!(
        (result.lower_a[[0, 0]] - (-1.0)).abs() < 1e-5,
        "lower_a[0,0] expected -1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_a[[0, 1]] - 2.0).abs() < 1e-5,
        "lower_a[0,1] expected 2.0, got {}",
        result.lower_a[[0, 1]]
    );
    assert!(
        (result.lower_a[[1, 0]] - 3.0).abs() < 1e-5,
        "lower_a[1,0] expected 3.0, got {}",
        result.lower_a[[1, 0]]
    );
    assert!(
        (result.lower_a[[1, 1]] - (-4.0)).abs() < 1e-5,
        "lower_a[1,1] expected -4.0, got {}",
        result.lower_a[[1, 1]]
    );

    // new_b = I @ b = [1, -1]
    assert!(
        (result.lower_b[0] - 1.0).abs() < 1e-5,
        "lower_b[0] expected 1.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.lower_b[1] - (-1.0)).abs() < 1e-5,
        "lower_b[1] expected -1.0, got {}",
        result.lower_b[1]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_rectangular_weights() -> Result<()> {
    // Non-square weight: R^3 → R^2, W = [[1,0,-1],[2,1,0]], b = [0.5, -0.5]
    let weight = array![[1.0_f32, 0.0, -1.0], [2.0, 1.0, 0.0]];
    let bias = array![0.5_f32, -0.5];
    let layer = LinearLayer::new(weight, Some(bias))?;

    // Identity bounds at output: 2x2
    let identity_bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&identity_bounds)?;
    let result = result.into_owned();

    // new_A = I @ W = W (2x3)
    assert_eq!(result.lower_a.shape(), &[2, 3]);
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "lower_a[0,0] expected 1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_a[[0, 2]] - (-1.0)).abs() < 1e-5,
        "lower_a[0,2] expected -1.0, got {}",
        result.lower_a[[0, 2]]
    );
    assert!(
        (result.lower_a[[1, 1]] - 1.0).abs() < 1e-5,
        "lower_a[1,1] expected 1.0, got {}",
        result.lower_a[[1, 1]]
    );

    // new_b = I @ b = [0.5, -0.5]
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-5,
        "lower_b[0] expected 0.5, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.lower_b[1] - (-0.5)).abs() < 1e-5,
        "lower_b[1] expected -0.5, got {}",
        result.lower_b[1]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_zero_weights() -> Result<()> {
    // Degenerate: all-zero weights, bias = [1, 2]
    // Output is constant: y = 0*x + b = b for any x
    let weight = Array2::zeros((2, 2));
    let bias = array![1.0_f32, 2.0];
    let layer = LinearLayer::new(weight, Some(bias))?;
    let identity_bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&identity_bounds)?;
    let result = result.into_owned();

    // new_A = I @ 0 = 0
    for &v in result.lower_a.iter() {
        assert!(v.abs() < 1e-6, "zero weights → zero A");
    }
    // new_b = I @ b = b
    assert!(
        (result.lower_b[0] - 1.0).abs() < 1e-5,
        "lower_b[0] expected 1.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.lower_b[1] - 2.0).abs() < 1e-5,
        "lower_b[1] expected 2.0, got {}",
        result.lower_b[1]
    );
    Ok(())
}

// ===== CROWN backward soundness: sampling verification =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_soundness_sampling() -> Result<()> {
    // Verify CROWN bounds contain all true outputs by sampling random points.
    // For linear y = Wx + b, CROWN backward with identity gives exact bounds,
    // so concretized CROWN should match the true output exactly.
    let layer = make_2x2_layer(); // W = [[1,2],[3,4]], b = [0.5, -0.5]
    let lower = array![-1.0_f32, -2.0];
    let upper = array![1.0_f32, 2.0];

    let identity_bounds = LinearBounds::identity(2);
    let crown = layer.propagate_linear(&identity_bounds)?;
    let crown = crown.into_owned();

    // Concretize CROWN bounds over the input domain
    let mut crown_lo = [0.0_f32; 2];
    let mut crown_hi = [0.0_f32; 2];
    for i in 0..2 {
        crown_lo[i] = crown.lower_b[i];
        crown_hi[i] = crown.upper_b[i];
        for j in 0..2 {
            let la = crown.lower_a[[i, j]];
            let ua = crown.upper_a[[i, j]];
            crown_lo[i] += la.max(0.0) * lower[j] + la.min(0.0) * upper[j];
            crown_hi[i] += ua.max(0.0) * upper[j] + ua.min(0.0) * lower[j];
        }
    }

    // Sample 25 grid points and verify containment
    for xi in 0..5 {
        for xj in 0..5 {
            let x0 = lower[0] + (upper[0] - lower[0]) * (xi as f32 / 4.0);
            let x1 = lower[1] + (upper[1] - lower[1]) * (xj as f32 / 4.0);
            // y = Wx + b
            let y0 = 1.0 * x0 + 2.0 * x1 + 0.5;
            let y1 = 3.0 * x0 + 4.0 * x1 - 0.5;
            for (k, &y) in [y0, y1].iter().enumerate() {
                assert!(
                    crown_lo[k] <= y + 1e-5,
                    "CROWN lower {} > true output {} for output {} at ({}, {})",
                    crown_lo[k],
                    y,
                    k,
                    x0,
                    x1
                );
                assert!(
                    crown_hi[k] >= y - 1e-5,
                    "CROWN upper {} < true output {} for output {} at ({}, {})",
                    crown_hi[k],
                    y,
                    k,
                    x0,
                    x1
                );
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_negative_weights_soundness() -> Result<()> {
    // Soundness with mixed-sign weights: verify CROWN bounds contain all outputs.
    let weight = array![[-2.0_f32, 1.0], [0.5, -3.0]];
    let bias = array![0.0_f32, 1.0];
    let layer = LinearLayer::new(weight, Some(bias))?;
    let lower = array![-1.0_f32, -1.0];
    let upper = array![1.0_f32, 1.0];
    let input = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn())?;

    // IBP bounds
    let ibp = layer.propagate_ibp(&input)?;

    // CROWN bounds
    let identity_bounds = LinearBounds::identity(2);
    let crown = layer.propagate_linear(&identity_bounds)?;
    let crown = crown.into_owned();

    // Concretize CROWN
    let mut crown_lo = [0.0_f32; 2];
    let mut crown_hi = [0.0_f32; 2];
    for i in 0..2 {
        crown_lo[i] = crown.lower_b[i];
        crown_hi[i] = crown.upper_b[i];
        for j in 0..2 {
            let la = crown.lower_a[[i, j]];
            let ua = crown.upper_a[[i, j]];
            crown_lo[i] += la.max(0.0) * lower[j] + la.min(0.0) * upper[j];
            crown_hi[i] += ua.max(0.0) * upper[j] + ua.min(0.0) * lower[j];
        }
    }

    // For linear layers, CROWN with identity gives exact bounds = IBP bounds
    for i in 0..2 {
        assert!(
            (crown_lo[i] - ibp.lower().iter().nth(i).unwrap()).abs() < 1e-4,
            "CROWN lower {} != IBP lower {} for output {}",
            crown_lo[i],
            ibp.lower().iter().nth(i).unwrap(),
            i
        );
        assert!(
            (crown_hi[i] - ibp.upper().iter().nth(i).unwrap()).abs() < 1e-4,
            "CROWN upper {} != IBP upper {} for output {}",
            crown_hi[i],
            ibp.upper().iter().nth(i).unwrap(),
            i
        );
    }

    // Sample 25 grid points: all must be contained
    for xi in 0..5 {
        for xj in 0..5 {
            let x0 = lower[0] + (upper[0] - lower[0]) * (xi as f32 / 4.0);
            let x1 = lower[1] + (upper[1] - lower[1]) * (xj as f32 / 4.0);
            let y0 = -2.0 * x0 + 1.0 * x1 + 0.0;
            let y1 = 0.5 * x0 - 3.0 * x1 + 1.0;
            for (k, &y) in [y0, y1].iter().enumerate() {
                assert!(
                    crown_lo[k] <= y + 1e-5 && crown_hi[k] >= y - 1e-5,
                    "output {} = {} not in [{}, {}] at ({}, {})",
                    k,
                    y,
                    crown_lo[k],
                    crown_hi[k],
                    x0,
                    x1
                );
            }
        }
    }
    Ok(())
}

// ===== CROWN backward with incoming bias =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_with_incoming_bias() -> Result<()> {
    // Incoming bounds have non-zero bias: b_in = [10, -10].
    // new_b should include the pass-through of b_in + A @ layer_bias.
    let layer = make_2x2_layer(); // W = [[1,2],[3,4]], b = [0.5, -0.5]
    let bounds = LinearBounds::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        array![10.0_f32, -10.0],
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        array![10.0_f32, -10.0],
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds)?;
    let result = result.into_owned();

    // new_b = A @ bias + b_in = I @ [0.5, -0.5] + [10, -10] = [10.5, -10.5]
    assert!(
        (result.lower_b[0] - 10.5).abs() < 1e-5,
        "lower_b[0] expected 10.5, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.lower_b[1] - (-10.5)).abs() < 1e-5,
        "lower_b[1] expected -10.5, got {}",
        result.lower_b[1]
    );
    Ok(())
}

// ===== Batched CROWN with rectangular weight =====

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_rectangular() -> Result<()> {
    // R^3 → R^2 layer with batched bounds
    let weight = array![[1.0_f32, 0.0, -1.0], [0.0, 2.0, 1.0]];
    let bias = array![0.5_f32, -0.5];
    let layer = LinearLayer::new(weight, Some(bias))?;

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let result = layer.propagate_linear_batched(&bounds)?;

    // new_A = I @ W = W (shape [2, 3])
    assert_eq!(result.lower_a.shape(), &[2, 3]);
    let v = result.lower_a[[0, 0]];
    assert!((v - 1.0).abs() < 1e-5, "lower_a[0,0] expected 1.0, got {v}");
    let v = result.lower_a[[0, 2]];
    assert!(
        (v - (-1.0)).abs() < 1e-5,
        "lower_a[0,2] expected -1.0, got {v}"
    );
    let v = result.lower_a[[1, 1]];
    assert!((v - 2.0).abs() < 1e-5, "lower_a[1,1] expected 2.0, got {v}");

    // new_b = I @ b = [0.5, -0.5]
    assert!(
        (result.lower_b[[0]] - 0.5).abs() < 1e-5,
        "lower_b[0] expected 0.5, got {}",
        result.lower_b[[0]]
    );
    assert!(
        (result.lower_b[[1]] - (-0.5)).abs() < 1e-5,
        "lower_b[1] expected -0.5, got {}",
        result.lower_b[[1]]
    );
    Ok(())
}

// ===== NaN weight injection test (#2432) =====

/// Verify that a NaN weight in a linear layer produces non-finite (conservative) IBP bounds
/// instead of silently absorbing the NaN via f32::max(NaN, 0.0) = 0.0.
///
/// Before #2432, the weight split `w.max(0.0)` / `w.min(0.0)` would convert NaN → 0.0,
/// causing that weight's contribution to vanish and producing unsound bounds.
/// After #2432, `nan_propagating_max_zero` / `nan_propagating_min_zero` preserve NaN,
/// which propagates through matmul and produces non-finite bounds.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_nan_weight_produces_non_finite_bounds() {
    let mut weight = array![[1.0_f32, 2.0], [3.0, 4.0]];
    weight[[0, 1]] = f32::NAN; // inject NaN into one weight
    let layer = LinearLayer::new(weight, None).expect("layer construction should succeed");

    // Verify that the cached w_pos contains NaN (not zero)
    assert!(
        layer.w_pos()[[0, 1]].is_nan(),
        "NaN weight must propagate through positive split, got: {}",
        layer.w_pos()[[0, 1]]
    );

    // Run IBP with well-formed input bounds
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).expect("valid shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("valid shape");
    let input = BoundedTensor::new(lower, upper).expect("valid input");

    let result = layer.propagate_ibp(&input).expect("IBP should not error");

    // The output bound for row 0 (which uses the NaN weight) must be conservative.
    // NaN propagates through matmul, then new_repaired widens it to ±inf: a NaN
    // output proves nothing in either direction, and any finite substitute would
    // be an unsound tightening. See: #2467 and #3423.
    let lo = result.lower()[[0]];
    let hi = result.upper()[[0]];
    assert_eq!(
        lo,
        f32::NEG_INFINITY,
        "NaN weight must produce lower = -inf, got {lo}",
    );
    assert_eq!(
        hi,
        f32::INFINITY,
        "NaN weight must produce upper = +inf, got {hi}",
    );

    // Row 1 uses only finite weights [3.0, 4.0], but the NaN in column 1 of the
    // weight matrix also affects row 1 through the faer matmul. Depending on the
    // matmul implementation, row 1 may or may not be contaminated by the NaN weight
    // in column 1. We don't assert on row 1 behavior — the key property is that
    // NaN does NOT silently vanish into zero in the weight split.
}

// ===== #2681: Non-finite A-matrix row fallback to ±inf bias =====

#[ntest::timeout(10000)]
#[test]
fn test_crown_nonfinite_row_fallback_2681() -> Result<()> {
    // #2681: When A @ W overflows f32 and produces non-finite coefficients in a row,
    // the entire row's A-coefficients are zeroed and bias is set to ±inf.
    // This replaces the old per-coefficient zero-substitution (#2409) which was
    // unsound because dropping individual coefficients can inflate or deflate bounds
    // depending on the sign of the coefficient and input bounds.
    //
    // Setup: A = [[1e20, 0], [0, 1]] @ W = [[1e20, 1], [1, 1]]
    // Row 0: [1e20 * 1e20, 1e20 * 1] = [Inf, 1e20] → non-finite detected in row 0
    // Row 1: [0 * 1e20 + 1, 0 * 1 + 1] = [1, 1] → all finite, preserved
    let weight = array![[1e20_f32, 1.0], [1.0, 1.0]];
    let layer = LinearLayer::new(weight, None)?;

    let bounds = LinearBounds::new(
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds)?;
    let result = result.into_owned();

    // Row 0: entire row zeroed because it had a non-finite coefficient.
    // Both columns zeroed (not just the overflowing one).
    assert_eq!(result.lower_a[[0, 0]], 0.0, "overflow row zeroed");
    assert_eq!(
        result.lower_a[[0, 1]],
        0.0,
        "overflow row zeroed (entire row)"
    );
    assert_eq!(result.upper_a[[0, 0]], 0.0, "overflow row zeroed");
    assert_eq!(
        result.upper_a[[0, 1]],
        0.0,
        "overflow row zeroed (entire row)"
    );

    // Row 0 bias: ±inf (sound, maximally loose).
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "overflow row → -inf bias"
    );
    assert_eq!(result.upper_b[0], f32::INFINITY, "overflow row → +inf bias");

    // Row 1: unchanged (all finite, no overflow).
    assert_close_1em5(result.lower_a[[1, 0]], 1.0, "unaffected row lower_a[1,0]");
    assert_close_1em5(result.lower_a[[1, 1]], 1.0, "unaffected row lower_a[1,1]");
    assert!(result.lower_b[1].is_finite(), "unaffected row bias finite");

    // Concretize with negative input bounds x ∈ [-2, -1].
    let input = BoundedTensor::new(
        array![-2.0_f32, -2.0].into_dyn(),
        array![-1.0_f32, -1.0].into_dyn(),
    )?;
    let concrete = result.concretize_sound(&input);

    // Row 0: 0 @ x + (-inf) = -inf (lower), 0 @ x + inf = inf (upper).
    // Sound: [-inf, +inf] contains all possible outputs.
    assert_eq!(
        concrete.lower()[[0]],
        f32::NEG_INFINITY,
        "overflow row concretizes to -inf"
    );
    assert_eq!(
        concrete.upper()[[0]],
        f32::INFINITY,
        "overflow row concretizes to +inf"
    );

    // Row 1: A[1,:] @ x = x[0] + x[1], range [-4, -2].
    let lo1 = concrete.lower()[[1]];
    let up1 = concrete.upper()[[1]];
    assert!(lo1 <= -4.0 + 1e-5, "row 1 lower contains -4, got {lo1}");
    assert!(up1 >= -2.0 - 1e-5, "row 1 upper contains -2, got {up1}");

    Ok(())
}

/// Verify that the #2681 fix is sound: for any input in the domain, the concretized
/// CROWN bounds (after non-finite row fallback) must contain the true output.
///
/// The old per-coefficient zero-substitution was unsound because it could inflate
/// lower bounds for negative inputs. The new per-row fallback to ±inf is always sound.
#[ntest::timeout(10000)]
#[test]
fn test_crown_nonfinite_row_fallback_soundness_2681() -> Result<()> {
    // Construct a scenario where:
    // 1. A @ W overflows in row 0 (produces Inf)
    // 2. Input bounds include negative values (where old approach was unsound)
    // 3. Verify that the new approach produces sound bounds for all sampled inputs
    let weight = array![[1e20_f32, 1.0, -1.0], [1.0, 1.0, 1.0], [0.5, -0.5, 2.0]];
    let bias = array![0.5_f32, -0.5, 1.0];
    let layer = LinearLayer::new(weight, Some(bias))?;

    // A-matrix with large coefficient that will overflow when multiplied by W
    let bounds = LinearBounds::new(
        array![[1e20_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Array1::zeros(3),
        array![[1e20_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Array1::zeros(3),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds)?;
    let result = result.into_owned();

    // Row 0 should have ±inf bias (overflow detected)
    assert_eq!(result.lower_b[0], f32::NEG_INFINITY);
    assert_eq!(result.upper_b[0], f32::INFINITY);

    // Rows 1, 2 should be finite and correct
    let lb1 = result.lower_b[1];
    let lb2 = result.lower_b[2];
    assert!(lb1.is_finite(), "lower_b[1] not finite: {lb1}");
    assert!(lb2.is_finite(), "lower_b[2] not finite: {lb2}");

    // Concretize with input bounds spanning negative and positive values
    let input = BoundedTensor::new(
        array![-3.0_f32, -2.0, -1.0].into_dyn(),
        array![1.0_f32, 2.0, 3.0].into_dyn(),
    )?;
    let concrete = result.concretize_sound(&input);

    // Row 0: must contain all possible true outputs (sound).
    // The true linear transform for row 0 would be 1e20 * (W[0,:] @ x) + 0.5,
    // but since this overflows, we accept [-inf, +inf].
    assert!(
        concrete.lower()[[0]] <= concrete.upper()[[0]],
        "lower must be <= upper"
    );

    // Rows 1, 2: verify soundness by sampling
    let lower = array![-3.0_f32, -2.0, -1.0];
    let upper = array![1.0_f32, 2.0, 3.0];
    for xi in 0..5 {
        for xj in 0..5 {
            for xk in 0..3 {
                let x0 = lower[0] + (upper[0] - lower[0]) * (xi as f32 / 4.0);
                let x1 = lower[1] + (upper[1] - lower[1]) * (xj as f32 / 4.0);
                let x2 = lower[2] + (upper[2] - lower[2]) * (xk as f32 / 2.0);

                // True output for rows 1, 2 (row 0 overflows, skip)
                let y1 = 1.0 * x0 + 1.0 * x1 + 1.0 * x2 - 0.5;
                let y2 = 0.5 * x0 - 0.5 * x1 + 2.0 * x2 + 1.0;

                let lo1 = concrete.lower()[[1]];
                let hi1 = concrete.upper()[[1]];
                assert!(
                    lo1 <= y1 + 1e-4 && hi1 >= y1 - 1e-4,
                    "row 1: y={y1} not in [{lo1}, {hi1}] at ({x0}, {x1}, {x2})"
                );

                let lo2 = concrete.lower()[[2]];
                let hi2 = concrete.upper()[[2]];
                assert!(
                    lo2 <= y2 + 1e-4 && hi2 >= y2 - 1e-4,
                    "row 2: y={y2} not in [{lo2}, {hi2}] at ({x0}, {x1}, {x2})"
                );
            }
        }
    }

    Ok(())
}

/// Regression test for #2789: rounding_ulps must not truncate usize to u32.
///
/// The pattern `u32::try_from(in_features).unwrap_or(u32::MAX).saturating_add(2)`
/// must saturate at u32::MAX for oversized in_features, never wrapping via `as u32`.
#[ntest::timeout(10000)]
#[test]
fn test_rounding_ulps_no_truncation_on_large_in_features() {
    // Simulate the conversion used in propagate_ibp_sound and graph_ibp
    let convert = |in_features: usize| -> u32 {
        u32::try_from(in_features)
            .unwrap_or(u32::MAX)
            .saturating_add(2)
    };

    // Normal case: small in_features
    assert_eq!(convert(10), 12);
    assert_eq!(convert(0), 2);
    assert_eq!(convert(1), 3);

    // Boundary: at u32::MAX - 2, result is exactly u32::MAX
    assert_eq!(convert((u32::MAX - 2) as usize), u32::MAX);

    // Boundary: at u32::MAX - 1, saturating_add(2) saturates
    assert_eq!(convert((u32::MAX - 1) as usize), u32::MAX);

    // Boundary: at u32::MAX, saturating_add(2) saturates
    assert_eq!(convert(u32::MAX as usize), u32::MAX);

    // On 64-bit: above u32::MAX, try_from fails → unwrap_or(u32::MAX) → saturates
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(convert(u32::MAX as usize + 1), u32::MAX);
        assert_eq!(convert(usize::MAX), u32::MAX);
        // The OLD buggy pattern: `(in_features as u32).saturating_add(2)` would
        // truncate u32::MAX + 1 to 0, then add 2, giving 2 instead of u32::MAX.
        let buggy = |in_features: usize| -> u32 { (in_features as u32).saturating_add(2) };
        assert_eq!(buggy(u32::MAX as usize + 1), 2, "old pattern wraps to 2");
    }
}

// --- Rank-0 regression tests (#2868) ---

/// Rank-0 BoundedTensor must return Err, not panic, from Linear IBP.
#[test]
fn test_ibp_rank0_returns_error_not_panic() {
    let w = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
    let layer = LinearLayer::new(w, None).unwrap();

    let lower = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[]), vec![2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "rank-0 input should return Err, not panic");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("rank-0"),
        "Error should mention rank-0: {err_msg}"
    );
}

/// Regression test for #4221: ndarray 0.17 clone() preserves F-layout (column-major)
/// arrays, so weights from const-fold or from_dynamic can arrive non-contiguous.
/// `LinearLayer::new()` must normalize to standard layout so `as_slice()` succeeds
/// on all downstream GEMM paths.
#[ntest::timeout(10000)]
#[test]
fn test_linear_new_normalizes_non_contiguous_weight() {
    // Create a genuinely non-contiguous weight via reversed_axes on an ArrayD.
    // This simulates what const-fold or from_dynamic can produce: an F-layout
    // Array2 where as_slice() returns None.
    let weight_nd: ArrayD<f32> =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let transposed = weight_nd.view().reversed_axes().to_owned();
    let weight_2d = transposed.into_dimensionality::<ndarray::Ix2>().unwrap();

    // Precondition: this weight is NOT standard layout (as_slice returns None)
    assert!(
        weight_2d.as_slice().is_none(),
        "precondition: weight should be non-contiguous"
    );

    let layer = LinearLayer::new(weight_2d, None).expect("should accept non-contiguous weight");

    // Post: weight is now standard layout — as_slice() succeeds
    assert!(
        layer.weight.as_slice().is_some(),
        "weight should be contiguous after new()"
    );
    assert!(
        layer.weight.is_standard_layout(),
        "weight should be standard layout after new()"
    );

    // Verify values are preserved (shape is [3, 2] after transpose)
    assert_eq!(layer.weight.dim(), (3, 2));
    assert_eq!(layer.weight[[0, 0]], 1.0);
    assert_eq!(layer.weight[[0, 1]], 4.0);
    assert_eq!(layer.weight[[1, 0]], 2.0);
    assert_eq!(layer.weight[[1, 1]], 5.0);
    assert_eq!(layer.weight[[2, 0]], 3.0);
    assert_eq!(layer.weight[[2, 1]], 6.0);
}

#[test]
fn parameter_replacement_rebuilds_all_cached_weight_views_atomically() {
    use crate::layers::common::BoundPropagation;

    let mut layer = LinearLayer::new(
        array![[1.0_f32, -2.0], [3.0, 4.0]],
        Some(array![0.5_f32, -0.5]),
    )
    .unwrap();

    // Prime the lazy row-major transpose as well as exercising the eager
    // positive/negative and faer caches before replacement.
    assert_eq!(layer.weight_t_row_major(), &[1.0_f32, 3.0, -2.0, 4.0]);
    let input =
        BoundedTensor::new(array![1.0_f32, 2.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();
    let before = layer.propagate_ibp(&input).unwrap();
    for (actual, expected) in before.lower().iter().copied().zip([-2.5_f32, 10.5]) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }

    layer
        .replace_parameters(
            array![[-1.0_f32, 2.0], [0.5, -3.0]],
            Some(array![1.0_f32, 2.0]),
        )
        .unwrap();

    assert_eq!(layer.weight_t_row_major(), &[-1.0_f32, 0.5, 2.0, -3.0]);
    let after = layer.propagate_ibp(&input).unwrap();
    for (actual, expected) in after.lower().iter().copied().zip([4.0_f32, -3.5]) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }

    let snapshot_weight = layer.weight().clone();
    let snapshot_bias = layer.bias().cloned();
    assert!(layer.set_weight(array![[1.0_f32, 2.0, 3.0]]).is_err());
    assert_eq!(layer.weight(), &snapshot_weight);
    assert_eq!(layer.bias(), snapshot_bias.as_ref());
}

// ===== #4321: deadline-authoritative Linear CROWN backward =====

/// Build a wide LinearLayer (`in_features` -> `out_features`) and a many-row
/// CROWN frontier representative of a root spec-matrix backward.
fn make_wide_layer_and_bounds(
    num_specs: usize,
    out_features: usize,
    in_features: usize,
) -> (LinearLayer, LinearBounds) {
    let weight = Array2::from_shape_fn((out_features, in_features), |(o, i)| {
        0.01 * (((o * 7 + i * 3) % 11) as f32 - 5.0)
    });
    let bias = Array1::from_shape_fn(out_features, |o| 0.001 * (o as f32));
    let layer = LinearLayer::new(weight, Some(bias)).unwrap();

    let lower_a = Array2::from_shape_fn((num_specs, out_features), |(s, o)| {
        0.05 * (((s * 5 + o) % 9) as f32 - 4.0)
    });
    let upper_a = lower_a.mapv(|v| v + 0.1);
    let lower_b = Array1::from_shape_fn(num_specs, |s| 0.01 * (s as f32) - 0.2);
    let upper_b = lower_b.mapv(|v| v + 0.05);
    let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();
    (layer, bounds)
}

/// The deadline-aware pollable CPU path must remain numerically equivalent to
/// the ordinary path when the deadline is comfortably in the future.
#[ntest::timeout(10000)]
#[test]
fn test_linear_deadline_pollable_matches_unbounded_4321() -> Result<()> {
    let (layer, bounds) = make_wide_layer_and_bounds(200, 48, 32);
    let far_deadline = Some(std::time::Instant::now() + std::time::Duration::from_hours(1));

    let expected = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let deadline_bounded = layer
        .propagate_linear_with_engine_and_deadline(&bounds, None, far_deadline)?
        .into_owned();
    assert_linear_bounds_close(
        &deadline_bounded,
        &expected,
        1e-5,
        "#4321 pollable CPU ~= unbounded CPU",
    );

    // A finite authority must not enter the generic engine at all.
    let engine = CountingGemmEngine::new();
    let deadline_with_engine = layer
        .propagate_linear_with_engine_and_deadline(&bounds, Some(&engine), far_deadline)?
        .into_owned();
    assert_linear_bounds_close(
        &deadline_with_engine,
        &deadline_bounded,
        0.0,
        "#4321 finite deadline ignores opaque engine",
    );
    assert_eq!(engine.gemm_calls(), 0);
    Ok(())
}

/// An already-expired deadline must abort with `DeadlineExceeded` rather than
/// run any GEMM, on both large and small workloads.
#[ntest::timeout(10000)]
#[test]
fn test_linear_deadline_expired_aborts_4321() {
    let expired = Some(
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("Instant supports subtracting 1ms"),
    );

    let (layer_big, bounds_big) = make_wide_layer_and_bounds(200, 48, 32);
    assert!(
        matches!(
            layer_big.propagate_linear_with_engine_and_deadline(&bounds_big, None, expired),
            Err(NyError::DeadlineExceeded(_))
        ),
        "#4321 expired deadline must abort the large Linear backward"
    );

    let (layer_small, bounds_small) = make_wide_layer_and_bounds(8, 8, 8);
    assert!(
        matches!(
            layer_small.propagate_linear_with_engine_and_deadline(&bounds_small, None, expired),
            Err(NyError::DeadlineExceeded(_))
        ),
        "#4321 expired deadline must abort the small-workload Linear backward"
    );
}

/// With `deadline = None` the deadline-aware entry must behave exactly like the
/// plain entry (no chunking, no abort), even for large workloads.
#[ntest::timeout(10000)]
#[test]
fn test_linear_deadline_none_is_plain_path_4321() -> Result<()> {
    let (layer, bounds) = make_wide_layer_and_bounds(200, 48, 32);
    let expected = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let actual = layer
        .propagate_linear_with_engine_and_deadline(&bounds, None, None)?
        .into_owned();
    assert_linear_bounds_close(&actual, &expected, 0.0, "#4321 None deadline == plain path");
    Ok(())
}

// ===== Spec-matrix root output-bound: finite authority excludes ordinary GEMM =====
//
// The deep-ResNet root OUTPUT bound is a spec-matrix CROWN backward with ~199
// objective rows propagated through Linear/Conv layers; its wide `A @ W` GEMMs
// are the dominant cost on tinyimagenet/cifar100/vit/traffic_signs. A generic
// caller-supplied GemmEngine is not proof that its ordinary methods satisfy the
// bounded-dispatch contract. Large f64 `A @ W` may instead use the separately
// installed process-global sound engine's explicit `gemm_f64_with_deadline`
// method. These tests pin the remaining authority invariant on realistic wide
// and narrow workloads:
//
//   1. A future deadline returns the same pollable CPU result whether or not an
//      ordinary caller engine was supplied.
//   2. No ordinary caller-engine dispatch occurs under finite authority.
//
// `CountingGemmEngine` delegates to `NaiveCpuGemmEngine` (an in-crate CPU
// `GemmEngine`) while counting `gemm_f32` calls — the engine-routed math is CPU
// here, but the *dispatch* is exercised exactly as a GPU engine would be.

/// A wide spec-matrix Linear backward must not dispatch opaque engine work
/// while carrying a finite deadline.
#[ntest::timeout(10000)]
#[test]
fn test_linear_spec_root_finite_deadline_refuses_opaque_engine() -> Result<()> {
    let (layer, bounds) = make_wide_layer_and_bounds(199, 48, 32);
    let far_deadline = Some(std::time::Instant::now() + std::time::Duration::from_hours(1));

    let bounded_baseline = layer
        .propagate_linear_with_engine_and_deadline(&bounds, None, far_deadline)?
        .into_owned();

    let engine = CountingGemmEngine::new();
    let supplied_engine = layer
        .propagate_linear_with_engine_and_deadline(&bounds, Some(&engine), far_deadline)?
        .into_owned();

    assert_linear_bounds_close(
        &supplied_engine,
        &bounded_baseline,
        0.0,
        "spec-root finite-deadline result is engine-independent",
    );
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite-deadline root backward must not enter generic GEMM"
    );

    Ok(())
}

/// The same authority rule applies to a small root objective set.
#[ntest::timeout(10000)]
#[test]
fn test_linear_spec_root_small_finite_deadline_refuses_opaque_engine() -> Result<()> {
    let (layer, bounds) = make_wide_layer_and_bounds(32, 48, 32);
    let far_deadline = Some(std::time::Instant::now() + std::time::Duration::from_hours(1));

    let faer_baseline = layer
        .propagate_linear_with_engine_and_deadline(&bounds, None, far_deadline)?
        .into_owned();

    let engine = CountingGemmEngine::new();
    let engine_routed = layer
        .propagate_linear_with_engine_and_deadline(&bounds, Some(&engine), far_deadline)?
        .into_owned();

    assert_linear_bounds_close(
        &engine_routed,
        &faer_baseline,
        0.0,
        "small spec-root finite-deadline result is engine-independent",
    );
    assert_eq!(
        engine.gemm_calls(),
        0,
        "small finite-deadline root backward must not enter generic GEMM"
    );

    Ok(())
}

// ===== Deadline-authoritative Linear IBP forward =====

#[ntest::timeout(10000)]
#[test]
fn linear_ibp_finite_deadline_encloses_unbatched_and_batched_and_refuses_engine() -> Result<()> {
    let layer = LinearLayer::new(
        array![[2.0_f32, -3.0, 0.5], [-1.0, 4.0, 2.0]],
        Some(array![0.25_f32, -0.5]),
    )?;
    let future = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));

    let unbatched = BoundedTensor::new(
        array![-1.0_f32, 2.0, -4.0].into_dyn(),
        array![3.0_f32, 5.0, 1.0].into_dyn(),
    )?;
    let output =
        layer.propagate_ibp_with_engine_and_deadline(&unbatched, Some(&PanicGemmEngine), future)?;
    for (index, (exact_lower, exact_upper)) in [(-18.75_f32, 0.75_f32), (-3.5, 22.5)]
        .into_iter()
        .enumerate()
    {
        assert!(output.lower()[index] <= exact_lower);
        assert!(output.upper()[index] >= exact_upper);
    }

    let batched = BoundedTensor::new(
        array![[-1.0_f32, 2.0, -4.0], [0.0, -2.0, 1.0]].into_dyn(),
        array![[3.0_f32, 5.0, 1.0], [1.0, -1.0, 3.0]].into_dyn(),
    )?;
    let batched_output =
        layer.propagate_ibp_with_engine_and_deadline(&batched, Some(&PanicGemmEngine), future)?;
    let exact = [
        [(-18.75_f32, 0.75_f32), (-3.5, 22.5)],
        [(3.75, 9.75), (-7.5, 1.5)],
    ];
    for batch in 0..2 {
        for output_index in 0..2 {
            let (exact_lower, exact_upper) = exact[batch][output_index];
            assert!(batched_output.lower()[[batch, output_index]] <= exact_lower);
            assert!(batched_output.upper()[[batch, output_index]] >= exact_upper);
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn linear_ibp_deadline_none_is_exact_legacy_engine_path() -> Result<()> {
    let layer = make_2x2_layer();
    let input = BoundedTensor::new(
        array![-1.0_f32, -2.0].into_dyn(),
        array![1.0_f32, 2.0].into_dyn(),
    )?;
    let engine = CountingGemmEngine::new();
    let expected = layer.propagate_ibp_with_engine(&input, Some(&engine))?;
    let calls_after_expected = engine.gemm_calls();
    let actual = layer.propagate_ibp_with_engine_and_deadline(&input, Some(&engine), None)?;
    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
    assert!(calls_after_expected > 0);
    assert!(engine.gemm_calls() > calls_after_expected);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn linear_ibp_expired_and_oversized_finite_deadlines_fail_typed_before_engine() -> Result<()> {
    let layer = LinearLayer::new(array![[1.0_f32]], None)?;
    let input = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn())?;
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("Instant supports a 1ms subtraction");
    let error = layer
        .propagate_ibp_with_engine_and_deadline(&input, Some(&PanicGemmEngine), Some(expired))
        .expect_err("expired Linear IBP must fail before engine dispatch");
    assert!(error.is_deadline_exceeded());

    let oversized_shape = [4 * 1024 * 1024 + 1, 1];
    let oversized = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&oversized_shape)),
        ArrayD::ones(IxDyn(&oversized_shape)),
    )?;
    let error = layer
        .propagate_ibp_with_engine_and_deadline(
            &oversized,
            Some(&PanicGemmEngine),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        )
        .expect_err("oversized finite-deadline Linear IBP must trip the cap");
    assert!(
        matches!(error, NyError::CpuMemoryExceeded { .. }),
        "live cap refusal must remain distinct from deadline expiry: {error}"
    );

    let mut oversized_rank = vec![1usize; 1_025];
    oversized_rank[0] = 0;
    let metadata_only = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&oversized_rank)),
        ArrayD::zeros(IxDyn(&oversized_rank)),
    )?;
    let error = layer
        .propagate_ibp_with_engine_and_deadline(
            &metadata_only,
            Some(&PanicGemmEngine),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        )
        .expect_err("oversized rank must trip the finite metadata cap");
    assert!(
        matches!(error, NyError::CpuMemoryExceeded { .. }),
        "metadata cap must remain distinct from deadline expiry: {error}"
    );

    let expected_none = layer.propagate_ibp_with_engine(&metadata_only, None)?;
    let actual_none = layer.propagate_ibp_with_engine_and_deadline(&metadata_only, None, None)?;
    assert_eq!(actual_none.lower(), expected_none.lower());
    assert_eq!(actual_none.upper(), expected_none.upper());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn linear_ibp_zero_contraction_finite_deadline_traverses_output_cells() -> Result<()> {
    let output_features = 5_000;
    let layer = LinearLayer::new(
        Array2::zeros((output_features, 0)),
        Some(Array1::from_shape_fn(output_features, |index| index as f32)),
    )?;
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[0])))?;
    let output = layer.propagate_ibp_with_engine_and_deadline(
        &input,
        Some(&PanicGemmEngine),
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    )?;
    assert_eq!(output.shape(), &[output_features]);
    for index in [0, 4_095, 4_999] {
        assert!(output.lower()[index] <= index as f32);
        assert!(output.upper()[index] >= index as f32);
    }
    Ok(())
}
