// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Explicit real-CUDA qualification of declared model-error propagation.
//! Missing hardware or any failed enclosure is an error, never a skipped test.
//! This checks the numeric contract, not ONNX loader provenance or performance.

use ny_core::{
    CertifiedWeightError, GpuCrownBackward, GpuCrownLayer, GpuCrownSeed, GpuResnetSegment, Result,
};
use ny_cuda::CudaGemmEngine;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn seed(a: Vec<f32>, width: usize) -> GpuCrownSeed {
    GpuCrownSeed {
        lower_a: Arc::from(a.clone()),
        upper_a: Arc::from(a),
        lower_b: Arc::from([0.0, 0.0]),
        upper_b: Arc::from([0.0, 0.0]),
        current_dim: width,
        num_specs: 2,
    }
}

fn enclose(got: &ny_core::GpuCrownResult, exact: &[(f64, f64); 2], tight: bool) {
    assert_eq!(got.lower_bounds.len(), 2);
    assert_eq!(got.upper_bounds.len(), 2);
    for row in 0..2 {
        let lo = f64::from(got.lower_bounds[row]);
        let hi = f64::from(got.upper_bounds[row]);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(
            lo <= exact[row].0 && hi >= exact[row].1,
            "row {row}: [{lo},{hi}] does not enclose {:?}",
            exact[row]
        );
        if tight {
            assert!(
                exact[row].0 - lo < 0.02 && hi - exact[row].1 < 0.02,
                "unexpectedly vacuous linear enclosure"
            );
        }
    }
}

fn linear(device: &CudaGemmEngine, stored_bias: bool, error: CertifiedWeightError) -> Result<()> {
    // Two real objective rows cross the unchanged 2^24-MAC throughput gate.
    // Each stored row has one dyadic nonzero. No padded/fake objective rows.
    let (outputs, inputs) = (8192, 1024);
    let mut weights = vec![0.0; outputs * inputs];
    let mut bias = vec![0.0; outputs];
    let mut a = vec![0.0; 2 * outputs];
    let mut centers = vec![vec![0.0f64; inputs]; 2];
    let mut weight_radii = vec![vec![0.0f64; inputs]; 2];
    let mut bias_centers = [0.0f64; 2];
    let mut bias_radii = [0.0f64; 2];
    for o in 0..outputs {
        let sign: f32 = if o / inputs % 2 == 0 { 1.0 } else { -1.0 };
        let w = sign * 0.125;
        weights[o * inputs + o % inputs] = w;
        bias[o] = if stored_bias { sign * 0.25 } else { 0.0 };
        for row in 0..2 {
            let coefficient = if row == 0 { sign } else { 1.0 };
            a[row * outputs + o] = coefficient;
            centers[row][o % inputs] += f64::from(coefficient * w);
            weight_radii[row][o % inputs] +=
                f64::from(error.weight_rel_err) * f64::from((coefficient * w).abs());
            bias_centers[row] += f64::from(coefficient * bias[o]);
            bias_radii[row] += f64::from(error.bias_abs_err) * f64::from(coefficient.abs());
        }
    }
    // All reference operations are exact dyadic binary64 sums (<2^53 units).
    // Independent coefficient/bias perturbations attain both extrema: choose
    // input signs from the nominal coefficient, then each perturbation's sign
    // to align with that objective and input. This is the entire declared ball.
    let exact = std::array::from_fn(|row| {
        let radius = centers[row]
            .iter()
            .zip(&weight_radii[row])
            .map(|(center, radius)| center.abs() + radius)
            .sum::<f64>()
            + bias_radii[row];
        (bias_centers[row] - radius, bias_centers[row] + radius)
    });
    let segments = [GpuResnetSegment::Chain(vec![GpuCrownLayer::Linear {
        weight: weights.into(),
        bias: stored_bias.then(|| Arc::from(bias)),
        out_features: outputs,
        in_features: inputs,
        cert_err: error,
    }])];
    let seed = seed(a, outputs);
    let got = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &segments,
        &seed,
        &vec![-1.0; inputs],
        &vec![1.0; inputs],
        &[],
        &[],
        Instant::now() + Duration::from_secs(60),
    )?;
    enclose(&got, &exact, true);
    println!("case=linear stored_bias={stored_bias} rho={} delta={} lower={:?} upper={:?} exact={exact:?}",
        error.weight_rel_err, error.bias_abs_err, got.lower_bounds, got.upper_bounds);
    Ok(())
}

fn convolution(device: &CudaGemmEngine) -> Result<()> {
    let (channels, side, output_side) = (32, 33, 31);
    let (input_width, output_width) =
        (channels * side * side, channels * output_side * output_side);
    let mut weight = vec![0.0f32; channels * channels * 9];
    for c in 0..channels {
        weight[(c * channels + c) * 9] = 0.125;
        weight[(c * channels + c) * 9 + 8] = -0.0625;
    }
    let mut a = vec![0.0f32; 2 * output_width];
    for row in 0..2 {
        for o in 0..output_width {
            a[row * output_width + o] = if row == 0 || o % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    let error = CertifiedWeightError {
        weight_rel_err: 0.125,
        bias_abs_err: 0.0625,
    };
    // Independently compose four admissible shared-kernel/bias endpoint models
    // using sparse spatial loops, not the implementation's im2col indexing.
    let mut exact = Vec::new();
    for ws in [-1.0, 1.0] {
        for bs in [-1.0, 1.0] {
            let mut rows = [(0.0, 0.0); 2];
            for row in 0..2 {
                let mut coefficients = vec![0.0f64; input_width];
                let mut bias = 0.0f64;
                for c in 0..channels {
                    for y in 0..output_side {
                        for x in 0..output_side {
                            let av = f64::from(
                                a[row * output_width + (c * output_side + y) * output_side + x],
                            );
                            coefficients[(c * side + y) * side + x] += av * (0.125 + ws * 0.015625);
                            coefficients[(c * side + y + 2) * side + x + 2] +=
                                av * (-0.0625 + ws * 0.0078125);
                            bias += av * bs * 0.0625;
                        }
                    }
                }
                let radius = coefficients.iter().map(|v| v.abs()).sum::<f64>();
                rows[row] = (bias - radius, bias + radius);
            }
            exact.push(rows);
        }
    }
    let segments = [GpuResnetSegment::Chain(vec![GpuCrownLayer::Conv2d {
        weight_col: weight.into(),
        bias_expanded: None,
        out_channels: channels,
        in_channels: channels,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        out_h: output_side,
        out_w: output_side,
        in_h: side,
        in_w: side,
        cert_err: error,
    }])];
    let seed = seed(a, output_width);
    let got = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &segments,
        &seed,
        &vec![-1.0; input_width],
        &vec![1.0; input_width],
        &[],
        &[],
        Instant::now() + Duration::from_secs(60),
    )?;
    for endpoints in &exact {
        enclose(&got, endpoints, false);
    }
    let late = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &segments,
        &seed,
        &vec![-1.0; input_width],
        &vec![1.0; input_width],
        &[],
        &[],
        Instant::now(),
    );
    assert!(matches!(late, Err(ny_core::NyError::DeadlineExceeded(_))));
    println!(
        "case=overlapping-conv endpoint_models={} lower={:?} upper={:?} expired_entry=refused",
        exact.len(),
        got.lower_bounds,
        got.upper_bounds
    );
    Ok(())
}

fn main() -> Result<()> {
    let device = CudaGemmEngine::new()?;
    println!(
        "device={} transport={}",
        device.device_name(),
        device.deadline_f64_transport_name()
    );
    linear(&device, true, CertifiedWeightError::EXACT)?;
    linear(
        &device,
        true,
        CertifiedWeightError {
            weight_rel_err: 0.125,
            bias_abs_err: 0.0625,
        },
    )?;
    linear(
        &device,
        false,
        CertifiedWeightError {
            weight_rel_err: 0.125,
            bias_abs_err: 0.0625,
        },
    )?;
    convolution(&device)?;
    println!("completed=4 actual_cuda_required=true benchmark=false original_loader_proof=false");
    Ok(())
}
