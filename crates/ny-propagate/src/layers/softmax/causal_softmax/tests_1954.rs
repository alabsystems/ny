// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Dense reference that mirrors the production row-decomposed path.
///
/// Assembles a full `(total_size × total_size)` block-diagonal Jacobian
/// (verifying the structural claim from #1954) but estimates error margins
/// per-row using the same sampling hash as the production row helper.
/// This tests algebraic equivalence of the row-decomposition, not sampling
/// divergence from a hypothetical global-sampling alternative.
fn dense_heuristic_reference_1954(
    layer: &CausalSoftmaxLayer,
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> LinearBounds {
    let shape = pre_activation.shape();
    let ndim = shape.len();
    let seq_q = shape[ndim - 2];
    let seq_k = shape[ndim - 1];
    let total_size = seq_q * seq_k;
    let num_outputs = bounds.num_outputs();

    let pre_lower = pre_activation
        .lower()
        .view()
        .into_shape_with_order((seq_q, seq_k))
        .expect("test pre-activation lower shape should reshape to (seq_q, seq_k)");
    let pre_upper = pre_activation
        .upper()
        .view()
        .into_shape_with_order((seq_q, seq_k))
        .expect("test pre-activation upper shape should reshape to (seq_q, seq_k)");

    // Build per-row Jacobians and assemble into a full block-diagonal matrix.
    let mut full_jacobian = ndarray::Array2::<f32>::zeros((total_size, total_size));
    let mut b_approx = Array1::<f32>::zeros(total_size);
    let mut max_error_above = Array1::<f32>::zeros(total_size);
    let mut max_error_below = Array1::<f32>::zeros(total_size);

    for row_idx in 0..seq_q {
        let row_start = row_idx * seq_k;
        let row_lower = pre_lower.row(row_idx);
        let row_upper = pre_upper.row(row_idx);
        // Replica of the layer's center computation — must stay bit-identical to
        // causal_softmax/batched.rs (which keeps (l+u)/2 over f32::midpoint).
        #[allow(clippy::manual_midpoint)]
        let x_center: Array1<f32> = row_lower
            .iter()
            .zip(row_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();
        let y_center = layer.eval_row(&x_center, row_idx);
        let jac_row = layer.jacobian_row(&x_center, row_idx);
        let jx = jac_row.dot(&x_center);
        let b_row: Array1<f32> = &y_center - &jx;

        // Assemble into full block-diagonal Jacobian.
        for local_j in 0..seq_k {
            b_approx[row_start + local_j] = b_row[local_j];
            for local_k in 0..seq_k {
                full_jacobian[[row_start + local_j, row_start + local_k]] =
                    jac_row[[local_j, local_k]];
            }
        }

        // Per-row sampling matches production `propagate_linear_row_with_bounds_heuristic`.
        let num_samples = 50;
        let mut x_sample = x_center.clone();
        for sample_idx in 0..num_samples {
            x_sample.assign(&x_center);
            for i in 0..seq_k {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = row_lower[i] + (row_upper[i] - row_lower[i]) * t;
            }

            if sample_idx < seq_k * 2 {
                let dim = sample_idx / 2;
                if dim < seq_k {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        row_lower[dim]
                    } else {
                        row_upper[dim]
                    };
                }
            }

            let y_actual = layer.eval_row(&x_sample, row_idx);
            let y_approx: Array1<f32> = jac_row.dot(&x_sample) + &b_row;
            for i in 0..seq_k {
                let error = y_actual[i] - y_approx[i];
                let gi = row_start + i;
                if error > max_error_above[gi] {
                    max_error_above[gi] = error;
                }
                if -error > max_error_below[gi] {
                    max_error_below[gi] = -error;
                }
            }
        }
    }

    for i in 0..total_size {
        max_error_above[i] *= 1.1;
        max_error_below[i] *= 1.1;
        let min_margin = 1e-6_f32;
        if max_error_above[i] < min_margin {
            max_error_above[i] = min_margin;
        }
        if max_error_below[i] < min_margin {
            max_error_below[i] = min_margin;
        }
    }

    let mut new_lower_a_f64 = ndarray::Array2::<f64>::zeros((num_outputs, total_size));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a_f64 = ndarray::Array2::<f64>::zeros((num_outputs, total_size));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    for out_idx in 0..num_outputs {
        for i in 0..total_size {
            let la = bounds.lower_a()[[out_idx, i]];
            let ua = bounds.upper_a()[[out_idx, i]];

            if la > 0.0 {
                let la_f64 = la as f64;
                for k in 0..total_size {
                    new_lower_a_f64[[out_idx, k]] += la_f64 * full_jacobian[[i, k]] as f64;
                }
                new_lower_b_f64[out_idx] += la_f64 * (b_approx[i] - max_error_below[i]) as f64;
            } else if la < 0.0 {
                let la_f64 = la as f64;
                for k in 0..total_size {
                    new_lower_a_f64[[out_idx, k]] += la_f64 * full_jacobian[[i, k]] as f64;
                }
                new_lower_b_f64[out_idx] += la_f64 * (b_approx[i] + max_error_above[i]) as f64;
            }

            if ua > 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..total_size {
                    new_upper_a_f64[[out_idx, k]] += ua_f64 * full_jacobian[[i, k]] as f64;
                }
                new_upper_b_f64[out_idx] += ua_f64 * (b_approx[i] + max_error_above[i]) as f64;
            } else if ua < 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..total_size {
                    new_upper_a_f64[[out_idx, k]] += ua_f64 * full_jacobian[[i, k]] as f64;
                }
                new_upper_b_f64[out_idx] += ua_f64 * (b_approx[i] - max_error_below[i]) as f64;
            }
        }
    }

    LinearBounds::new_or_conservative(
        new_lower_a_f64.mapv(|x| x as f32),
        new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
        new_upper_a_f64.mapv(|x| x as f32),
        new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
    )
    .expect("dense reference should produce valid bounds")
}

#[test]
fn crown_heuristic_block_diagonal_matches_dense_reference_1954() {
    let layer = CausalSoftmaxLayer::new(-1)
        .with_heuristic_sampling(true)
        .with_window_size(1);
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![-1.0, -0.2, 0.3, -0.7, -0.1, 0.4, -0.5, 0.2, 0.6],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![0.6, 0.8, 1.2, 0.5, 0.9, 1.4, 0.7, 1.1, 1.8],
    )
    .unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(9);

    let actual = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();
    let expected = dense_heuristic_reference_1954(&layer, &bounds, &pre);
    let tol = 1e-5;

    for (idx, (&actual_value, &expected_value)) in actual
        .lower_a
        .iter()
        .zip(expected.lower_a.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "#1954 lower_a mismatch at {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_a
        .iter()
        .zip(expected.upper_a.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "#1954 upper_a mismatch at {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .lower_b
        .iter()
        .zip(expected.lower_b.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "#1954 lower_b mismatch at {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_b
        .iter()
        .zip(expected.upper_b.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "#1954 upper_b mismatch at {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
}

#[test]
fn crown_heuristic_scalar_matches_rowwise_block_assembly_1954() {
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![-1.0, -0.4, -0.2, -0.8, -0.1, 0.0, -0.6, 0.2, 0.4],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![0.4, 0.8, 1.0, 0.5, 0.9, 1.3, 0.7, 1.1, 1.6],
    )
    .unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    let lower_a = Array2::from_shape_vec(
        (2, 9),
        vec![
            1.0, -0.5, 0.25, -0.75, 0.5, 1.25, 0.1, -0.2, 0.3, -0.4, 0.8, -0.6, 0.2, -0.1, 0.7,
            -0.9, 0.4, -0.3,
        ],
    )
    .unwrap();
    let upper_a = Array2::from_shape_vec(
        (2, 9),
        vec![
            0.6, -0.3, 0.9, -0.2, 0.4, -0.8, 0.5, -0.7, 0.1, -0.2, 0.3, -0.4, 1.1, -0.5, 0.2, -0.6,
            0.7, -0.8,
        ],
    )
    .unwrap();
    let bounds =
        LinearBounds::new(lower_a, array![0.15, -0.35], upper_a, array![0.25, 0.45]).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();
    let expected = assemble_rowwise_scalar_heuristic_result(&layer, &bounds, &pre, 3, 3);

    assert_eq!(result.lower_a.shape(), expected.lower_a.shape());
    assert_eq!(result.upper_a.shape(), expected.upper_a.shape());
    for out_idx in 0..2 {
        for in_idx in 0..9 {
            assert!(
                (result.lower_a[[out_idx, in_idx]] - expected.lower_a[[out_idx, in_idx]]).abs()
                    < 1e-6,
                "lower_a[{out_idx},{in_idx}] mismatch: {} vs {}",
                result.lower_a[[out_idx, in_idx]],
                expected.lower_a[[out_idx, in_idx]],
            );
            assert!(
                (result.upper_a[[out_idx, in_idx]] - expected.upper_a[[out_idx, in_idx]]).abs()
                    < 1e-6,
                "upper_a[{out_idx},{in_idx}] mismatch: {} vs {}",
                result.upper_a[[out_idx, in_idx]],
                expected.upper_a[[out_idx, in_idx]],
            );
        }
        assert!(
            (result.lower_b[out_idx] - expected.lower_b[out_idx]).abs() < 1e-6,
            "lower_b[{out_idx}] mismatch: {} vs {}",
            result.lower_b[out_idx],
            expected.lower_b[out_idx],
        );
        assert!(
            (result.upper_b[out_idx] - expected.upper_b[out_idx]).abs() < 1e-6,
            "upper_b[{out_idx}] mismatch: {} vs {}",
            result.upper_b[out_idx],
            expected.upper_b[out_idx],
        );
    }
}

fn assemble_rowwise_scalar_heuristic_result(
    layer: &CausalSoftmaxLayer,
    bounds: &LinearBounds,
    pre: &BoundedTensor,
    seq_q: usize,
    seq_k: usize,
) -> LinearBounds {
    let num_outputs = bounds.num_outputs();
    let total_size = seq_q * seq_k;
    let pre_lower = pre
        .lower()
        .view()
        .into_shape_with_order((seq_q, seq_k))
        .unwrap();
    let pre_upper = pre
        .upper()
        .view()
        .into_shape_with_order((seq_q, seq_k))
        .unwrap();

    let mut lower_a = Array2::<f32>::zeros((num_outputs, total_size));
    let mut upper_a = Array2::<f32>::zeros((num_outputs, total_size));
    let mut lower_b = bounds.lower_b.clone();
    let mut upper_b = bounds.upper_b.clone();

    for row_idx in 0..seq_q {
        let row_start = row_idx * seq_k;
        let row_end = row_start + seq_k;
        let row_bounds = LinearBounds::new(
            bounds
                .lower_a
                .slice(ndarray::s![.., row_start..row_end])
                .to_owned(),
            Array1::zeros(num_outputs),
            bounds
                .upper_a
                .slice(ndarray::s![.., row_start..row_end])
                .to_owned(),
            Array1::zeros(num_outputs),
        )
        .unwrap();
        let row_result = layer
            .propagate_linear_row_with_bounds_heuristic(
                &row_bounds,
                &pre_lower.row(row_idx).to_owned(),
                &pre_upper.row(row_idx).to_owned(),
                row_idx,
            )
            .unwrap();

        lower_a
            .slice_mut(ndarray::s![.., row_start..row_end])
            .assign(&row_result.lower_a);
        upper_a
            .slice_mut(ndarray::s![.., row_start..row_end])
            .assign(&row_result.upper_a);

        for out_idx in 0..num_outputs {
            lower_b[out_idx] = next_down_f32(lower_b[out_idx] + row_result.lower_b[out_idx]);
            upper_b[out_idx] = next_up_f32(upper_b[out_idx] + row_result.upper_b[out_idx]);
        }
    }

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b).unwrap()
}
