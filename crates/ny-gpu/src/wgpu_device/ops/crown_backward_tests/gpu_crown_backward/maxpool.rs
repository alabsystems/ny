// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward tests for MaxPool2d routing.

use super::*;

pub(super) fn cpu_maxpool2d_backward(
    a_l: &mut Vec<f32>,
    a_u: &mut Vec<f32>,
    b_l: &mut [f32],
    b_u: &mut [f32],
    routing: &[u32],
    ibp_lower: &[f32],
    ibp_upper: &[f32],
    num_specs: usize,
    input_dim: usize,
    output_dim: usize,
) {
    let mut new_l = vec![0.0f32; num_specs * input_dim];
    let mut new_u = vec![0.0f32; num_specs * input_dim];

    for s in 0..num_specs {
        for out_idx in 0..output_dim {
            let lower_coeff = a_l[s * output_dim + out_idx];
            let upper_coeff = a_u[s * output_dim + out_idx];
            let route = routing[out_idx];

            if route != u32::MAX {
                let dst = s * input_dim + route as usize;
                new_l[dst] += lower_coeff;
                new_u[dst] += upper_coeff;
            } else {
                if lower_coeff > 0.0 {
                    b_l[s] += lower_coeff * ibp_lower[out_idx];
                } else if lower_coeff < 0.0 {
                    b_l[s] += lower_coeff * ibp_upper[out_idx];
                }
                if upper_coeff > 0.0 {
                    b_u[s] += upper_coeff * ibp_upper[out_idx];
                } else if upper_coeff < 0.0 {
                    b_u[s] += upper_coeff * ibp_lower[out_idx];
                }
            }
        }
    }

    *a_l = new_l;
    *a_u = new_u;
}

#[test]
fn test_crown_backward_gpu_maxpool2d_routing_and_fallback() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let layers = vec![GpuCrownLayer::MaxPool2d {
        routing: vec![1, u32::MAX],
        ibp_lower: vec![0.25, -0.2],
        ibp_upper: vec![0.75, 0.9],
        input_dim: 4,
        output_dim: 2,
    }];
    let spec = vec![1.0f32, 0.0, -0.5, 1.0];
    let input_lower = vec![-1.0f32, 0.25, -0.4, 0.2];
    let input_upper = vec![1.0f32, 0.75, 0.6, 1.2];

    let gpu = device
        .crown_backward_gpu(&layers, &spec, 2, &input_lower, &input_upper)
        .expect("GPU MaxPool2d backward should succeed");
    let (cpu_lower, cpu_upper) = cpu_crown_backward(&layers, &spec, 2, &input_lower, &input_upper);

    for idx in 0..2 {
        let dl = (gpu.lower_bounds[idx] - cpu_lower[idx]).abs();
        let du = (gpu.upper_bounds[idx] - cpu_upper[idx]).abs();
        assert!(
            dl <= 1e-5,
            "lower[{idx}] GPU={} CPU={} diff={dl}",
            gpu.lower_bounds[idx],
            cpu_lower[idx]
        );
        assert!(
            du <= 1e-5,
            "upper[{idx}] GPU={} CPU={} diff={du}",
            gpu.upper_bounds[idx],
            cpu_upper[idx]
        );
    }
}
