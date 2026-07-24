// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SiLU (Swish) activation relaxation functions.

pub(crate) mod math;
mod relaxation;

pub use math::silu_eval;
pub use relaxation::silu_sound_linear_relaxation;
