// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic layers for bound propagation.
//!
//! This module contains constant arithmetic operations used in neural network
//! verification:
//!
//! - **Affine constant ops**: [`AddConstantLayer`], [`MulConstantLayer`],
//!   [`DivConstantLayer`], [`SubConstantLayer`]
//! - **Nonlinear ops**: [`AbsLayer`], [`SqrtLayer`], [`PowConstantLayer`]
//!
//! Each layer implements IBP and CROWN bound propagation via the
//! [`BoundPropagation`](super::common::BoundPropagation) trait.

mod common;

mod add_constant;
mod div_constant;
mod mul_constant;
mod sub_constant;
mod validate;

mod abs;
mod pow;
mod pow_relaxation;
mod sqrt;

#[cfg(test)]
mod tests;

// Re-export all public types and functions to preserve the existing API.
pub use add_constant::AddConstantLayer;
pub use div_constant::DivConstantLayer;
pub use mul_constant::MulConstantLayer;
pub use sub_constant::SubConstantLayer;

pub use abs::AbsLayer;
pub use pow::PowConstantLayer;
pub use pow_relaxation::pow2_linear_relaxation;
pub use sqrt::SqrtLayer;

// Re-export relaxation functions for decomposed normalization CROWN (#318)
// and tests (#3240).
pub use abs::abs_linear_relaxation;
pub use sqrt::sqrt_linear_relaxation;
