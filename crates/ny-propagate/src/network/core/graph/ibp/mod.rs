// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP, detailed IBP, and zonotope propagation for `GraphNetwork`.

mod base;
mod block_wise;
mod block_wise_helpers;
mod detailed;
pub(crate) mod dfl_envelope;
pub(crate) mod dispatch;
pub(crate) mod gpu_plan;
mod zonotope;
mod zonotope_matmul;

pub use zonotope::{ZonotopePropagationOptions, ZonotopeSoftmaxMode};
