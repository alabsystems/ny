// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision (f64) layer implementations for soundness-critical propagation.
//!
//! These modules provide IBP and CROWN backward for the minimal layer set needed
//! by VNN-COMP soundnessbench and sat_relu: Linear (FC), Conv2D, and ReLU.
//! All arithmetic is f64 throughout — no intermediate f32 casts.
//!
//! Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).

pub(crate) mod conv2d;
pub(crate) mod linear;
pub(crate) mod relu;

#[cfg(test)]
mod tests;
