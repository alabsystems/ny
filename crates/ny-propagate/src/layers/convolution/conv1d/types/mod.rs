// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv1d and ConvTranspose1d layer types with batched CROWN backward propagation.

mod common;
mod forward;
mod transpose;

pub use forward::Conv1dLayer;
pub use transpose::ConvTranspose1dLayer;

#[cfg(test)]
mod common_tests;
