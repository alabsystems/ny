// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv2d propagation tests.

use super::*;
use ndarray::{arr1, Array1, ArrayD};

mod batched;
mod crown;
mod engine;
mod ibp;
mod network;
mod stride_padding;
mod transpose;
mod validation;

/// Helper to create 4D kernel from nested slices.
fn kernel_4d(data: &[[[[f32; 2]; 2]; 1]; 1]) -> ArrayD<f32> {
    // Shape: (out_channels=1, in_channels=1, kh=2, kw=2)
    let mut arr = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2, 2]));
    for oc in 0..1 {
        for ic in 0..1 {
            for kh in 0..2 {
                for kw in 0..2 {
                    arr[[oc, ic, kh, kw]] = data[oc][ic][kh][kw];
                }
            }
        }
    }
    arr
}

/// Helper to create 3D input from nested slices.
fn input_3d(data: &[[[f32; 3]; 3]; 1]) -> ArrayD<f32> {
    // Shape: (channels=1, height=3, width=3)
    let mut arr = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for c in 0..1 {
        for h in 0..3 {
            for w in 0..3 {
                arr[[c, h, w]] = data[c][h][w];
            }
        }
    }
    arr
}
