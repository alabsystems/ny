// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU activation linear relaxation for neural network verification proofs.
//!
//! Provides both Erf and Tanh approximation variants with precomputed tangent tables.
//! Reference: α,β-CROWN Team, auto_LiRPA @ 9d100ec070868440b48d34e2f1dd21b97aab9172

pub(crate) mod eval;
pub(crate) mod sound_relax;
pub(crate) mod tables;

pub use eval::{gelu_eval, gelu_tanh_inflection_point};
pub use sound_relax::{
    gelu_sound_linear_relaxation, gelu_sound_linear_relaxation_with_alpha,
    gelu_tanh_sound_linear_relaxation,
};
