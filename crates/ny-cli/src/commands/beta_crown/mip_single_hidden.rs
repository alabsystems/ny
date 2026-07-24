// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;

/// Returns true when the preprocessed MIP network is exactly
/// `Linear -> ReLU -> Linear`.
pub(super) fn is_single_hidden_linear_relu_linear(network: &Network) -> bool {
    matches!(
        network.layers(),
        [Layer::Linear(_), Layer::ReLU(_), Layer::Linear(_)]
    )
}

/// Collect exact intermediate bounds for the single-hidden affine MIP fast path.
///
/// For `Linear -> ReLU -> Linear` over an axis-aligned input box, the only
/// Big-M pre-activation vector is the first affine output. Interval propagation
/// is exact for that linear map on a box, so spending CROWN/LP budget there
/// cannot tighten the MIP encoding. See
/// `designs/2026-03-14-issue-3864-safenlp-exact-affine-mip-fast-path.md`.
pub(super) fn collect_exact_single_hidden_intermediate_bounds(
    network: &Network,
    input: &BoundedTensor,
) -> Result<Vec<BoundedTensor>> {
    network
        .collect_ibp_bounds(input)
        .map_err(|e| anyhow::anyhow!("IBP failed: {}", e))
}
