// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Stable user-facing status for the dormant cGAN depth-two replay.
///
/// The authenticated production leaf-row call deliberately supplies no
/// depth-two context, so both row receipts are `NotRequested`. This module is
/// compiled with and without `mip`, and the protected-cover example includes
/// the same source, keeping every CLI/JSON surface on one token.
pub(crate) const CGAN_DEPTH_TWO_PRODUCTION_MODE: &str = "disabled_not_requested";
