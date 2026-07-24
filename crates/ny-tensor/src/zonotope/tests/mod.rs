// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! Zonotope unit tests organized by feature area to avoid monolithic modules.

mod accessors;
mod basic;
mod dot;
mod from_bounded_per_position;
mod from_input_per_position;
mod layer_norm;
mod linear;
mod mul;
// SiLU affine tests split by scope to keep modules small.
mod silu_affine_basic;
mod silu_affine_mutation_1d;
mod silu_affine_mutation_2d;
// GELU affine tests (#2470).
mod gelu_affine;
mod softmax;
mod softmax_causal_validation_2519;
mod softmax_containment_2744;
mod softmax_nan;
mod softmax_overflow_3012;
mod softmax_vacuous_2479;
// Star-set affine transformer IBP-parity soundness gate (S1-2).
mod star_parity;
// Unwired block-generator convolution: scalar differential and resource model.
mod star_blocked_conv2d;
mod transpose;
