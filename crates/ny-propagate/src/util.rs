// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod contiguous;

pub(crate) use contiguous::{contiguous_flat_slice, contiguous_flat_slice_mut};

/// Env knob for the conv-patches collection feature (`#conv-patches-collect`).
const CONV_PATCHES_COLLECT_ENV: &str = "NY_CONV_PATCHES_COLLECT";

/// Whether the conv-patches collection feature is enabled — **on by default**,
/// disabled only by an explicit `NY_CONV_PATCHES_COLLECT=0` (#conv-crown-residual).
///
/// This one switch gates four cooperating parts of the same capability:
///
/// - the EXACT padded-conv patches composition (intermediate-tap masking,
///   `conv2d/bound_patches.rs`), without which a patches walk dies at the
///   SECOND padded conv;
/// - patches-mode alpha-CROWN (`target_backward_patches.rs`), without which any
///   active `alpha_state` forces the dense path;
/// - the raised aggregate patches budget (`budget_policy.rs`);
/// - the beta-CROWN graph init counterpart (`beta_crown/.../init.rs`).
///
/// It shipped default-OFF while the padded composition was being proven. That
/// proof exists: `crown_patches_composition.rs::padded_compose_parity` pins the
/// masked composition against the dense CROWN backward at coefficient level
/// (tol 2e-4) and calls itself "the soundness contract for lifting the
/// `nonzero_incoming_padding` guard".
///
/// Leaving it off made the whole capability unreachable in production, which in
/// turn made the residual-`Add` patches route inert on exactly the ResNet
/// benchmarks it was built for: with the gate off a CIFAR100/TinyImageNet walk
/// stays in patches for a single conv and then densifies, because every 3x3 conv
/// in those models has `pads=[1,1,1,1]`.
///
/// Default-on with an explicit disable also matches this repo's stated
/// convention for feature flags.
pub(crate) fn conv_patches_collect_enabled() -> bool {
    match std::env::var_os(CONV_PATCHES_COLLECT_ENV) {
        Some(value) => value != "0" && !value.is_empty(),
        None => true,
    }
}
