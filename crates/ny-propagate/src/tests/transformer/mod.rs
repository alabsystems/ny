// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transformer component tests (attention, softmax, causal softmax, MLP, LayerNorm, GELU,
//! zonotope, batched bounds, batched CROWN, transformer blocks).

mod prelude {
    pub(super) use crate::*;
    pub(super) use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};
}

mod attention;
mod attention_bilinear;
mod attention_full_composition;
mod batched_bounds;
mod batched_crown;
mod causal_softmax;
mod gelu;
mod layernorm;
mod mlp;
mod softmax;
mod softmax_soundness;
mod transformer_block;
mod zonotope;
