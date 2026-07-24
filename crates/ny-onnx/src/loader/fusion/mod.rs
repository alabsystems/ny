// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod batch_norm_fold;
mod causal_softmax;
mod gelu;
mod gelu_tanh;
mod helpers;
mod instance_norm;
mod layer_norm;
mod logsumexp;
mod merge_linear;

pub(super) use batch_norm_fold::fold_batch_norm_into_conv_linear_with_context;
#[cfg(test)]
pub(super) use batch_norm_fold::{fold_batch_norm_into_conv_linear, gemm_has_exact_default_affine};
pub(super) use causal_softmax::try_fuse_causal_softmax;
pub(super) use gelu::try_fuse_gelu;
pub(super) use gelu_tanh::try_fuse_gelu_tanh;
pub(super) use instance_norm::try_discriminate_instance_norm;
pub(super) use layer_norm::try_fuse_layer_norm;
pub(super) use logsumexp::try_fuse_logsumexp;
pub(super) use merge_linear::try_fuse_merge_linear;

#[cfg(test)]
mod tests;
