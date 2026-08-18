// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BatchNorm layer for bound propagation.

mod crown_batched;
mod crown_scalar;
mod ibp;
mod math;
mod patches;
#[cfg(test)]
mod tests;
mod types;

pub use types::{BatchNormChannelAxisHint, BatchNormLayer};
