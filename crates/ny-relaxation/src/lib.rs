// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]
// The scalar mirrors keep `(l + u) / 2.0` written exactly as the production
// relaxations write it (which allow this lint per-site): `f64::midpoint`
// rounds differently, and these copies exist to mirror that arithmetic.
#![allow(clippy::manual_midpoint)]

//! Scalar relaxation functions for neural network verification proofs.
//!
//! This crate contains the pure-scalar (f32) mathematical functions used by
//! Kani proof harnesses in `proofs/kani/`. It depends only on `ny-core`
//! and `libm`, avoiding the heavy dependencies (faer, ndarray, rayon, etc.)
//! that ny-propagate requires.
//!
//! **Status:** This is the proof-support scalar surface used by
//! `proofs/kani/`. `ny-propagate` does not yet depend on this crate — the
//! functions are duplicated across the two crates — but the copies are now
//! BOUND to production by the differential tests in `tests/drift.rs`
//! (ny-propagate is a dev-dependency), which evaluate every mirrored pair
//! over an adversarial grid:
//!
//! - Bit-exact pairs: `abs`, `pow2`, `exp`, `gelu_eval`, `silu_eval`,
//!   `gelu_tanh_inflection_point`, the softmax/logsoftmax/logsumexp IBP
//!   helpers, and the `safe_*`/`interval_mul` helpers.
//! - Proof-bound pairs (`log`, `sqrt`, and the sound `gelu`/`silu`
//!   relaxations) are bit-exact with production. Their intercepts include the
//!   full f32 affine-evaluation allowance (slope conversion, multiplication,
//!   addition, and FTZ floors), and the GELU mirror includes production's
//!   sound interval-minimum floor tightening.
//! - `relu` remains a standalone reference implementation; the production
//!   `relu_linear_relaxation`/`relu_crossing_upper_chord` path is pub(crate)
//!   and is exercised through the public `ReLULayer` CROWN backward pass in
//!   the drift tests.
//! - The centered-normalization helpers proved in
//!   `proofs/kani/src/bin/normalization.rs` have NO function counterpart in
//!   either crate (production inlines the formulas in the layer_norm /
//!   group_norm / instance_norm IBP paths) and remain unbound.

pub mod abs;
pub mod exp;
pub mod gelu;
pub mod log;
pub mod pow;
pub mod relu;
pub mod rounding;
pub mod safe_math;
pub mod silu;
pub mod softmax;
pub mod sqrt;
pub mod types;

// Convenient re-exports matching the proofs crate import surface.
pub use abs::abs_linear_relaxation;
pub use exp::exp_linear_relaxation;
pub use gelu::{
    gelu_eval, gelu_sound_linear_relaxation, gelu_tanh_inflection_point,
    gelu_tanh_sound_linear_relaxation,
};
pub use log::log_linear_relaxation;
pub use pow::pow2_linear_relaxation;
pub use relu::relu_crown_relaxation;
pub use rounding::{next_down_f32, next_up_f32};
#[cfg(any(test, feature = "kani-proofs"))]
pub use safe_math::safe_add_for_bounds;
pub use safe_math::{
    interval_mul_for_bounds, safe_add_for_bounds_with_polarity, safe_add_lower_for_bounds,
    safe_add_upper_for_bounds, safe_mul_for_bounds, safe_mul_pair_for_bounds,
};
pub use silu::{silu_eval, silu_sound_linear_relaxation};
pub use softmax::{
    exp_interval_bounds, logsoftmax_ibp_bounds, logsumexp_slice, softmax_ibp_element_bounds,
};
pub use sqrt::sqrt_linear_relaxation;
pub use types::{GeluApproximation, LinearRelaxation};
