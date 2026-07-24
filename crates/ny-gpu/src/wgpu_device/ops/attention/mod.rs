// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention-specific wgpu ops for interval bound propagation.
//! Splits basic attention paths from fully fused GPU attention.

mod basic;
mod fused;
mod helpers;
