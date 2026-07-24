// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Convolution layer tests: Conv1D, Conv2D, ConvTranspose, and pooling layers.
//!
//! These tests validate the IBP and CROWN propagation through convolutional
//! layers, including stride, padding, transpose operations, and pooling.

// Provide shared layer/bounds types to conv test submodules via super::*.
use crate::*;

mod average_pool;
mod conv1d;
mod conv2d;
mod conv_transpose;
mod max_pool;
