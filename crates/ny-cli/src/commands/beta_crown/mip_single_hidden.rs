// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;

use super::mip_sound_intermediate_bounds::SoundIntermediateBounds;

/// Returns true when the preprocessed MIP network is exactly
/// `Linear -> ReLU -> Linear`.
pub(super) fn is_single_hidden_linear_relu_linear(network: &Network) -> bool {
    matches!(
        network.layers(),
        [Layer::Linear(_), Layer::ReLU(_), Layer::Linear(_)]
    )
}

/// Collect certified intermediate bounds for the single-hidden affine MIP fast
/// path.
///
/// For `Linear -> ReLU -> Linear` over an axis-aligned input box, the only
/// Big-M pre-activation vector is the first affine output. In REAL arithmetic
/// interval propagation is exact for that linear map on a box, so spending
/// CROWN/LP budget there cannot tighten the MIP encoding.
/// That exactness claim does NOT survive f32: the first affine output is a
/// K-term dot product evaluated in round-to-nearest f32, and for the Conv2d
/// arm of the plain collector (`Conv -> ReLU -> Flatten -> Gemm` nets such as
/// malbeware's `malware_malimg_family_scaled_16-25.onnx`, which match this
/// fast path AFTER `unfold_conv2d_to_linear`) there is no directed rounding
/// and no Higham widening at all — under cancellation the plain box excludes
/// the true pre-activation, and this path adds NO CROWN intersection and NO
/// LP tightening on top of it. The box goes into `encode_relu` verbatim and
/// the certificate is checked against the encoded model, so an inward box is
/// a minted `Unsat` (#mip-conv-sound-enclosure).
///
/// Therefore the fast path collects with [`Network::collect_ibp_bounds_sound`]:
/// Linear via `propagate_ibp_sound` (Higham/abssum widening plus directed
/// casts) and Conv2d via `propagate_ibp_sound_with_engine` (certified
/// `γ_{K+2}·S + 2u·|y|` term, rounded outward). "Exact" here means "the
/// certified enclosure of the first affine output" — strictly wider-or-equal
/// than the plain box, so it can only cost a proof, never mint one.
///
/// Returns the TYPE-ENFORCED [`SoundIntermediateBounds`]
/// (`mip_sound_intermediate_bounds.rs`): its only constructor is the certified
/// collector, and `convert_intermediate_bounds` accepts nothing else, so the
/// plain collector cannot be substituted back here without a compile error.
pub(super) fn collect_exact_single_hidden_intermediate_bounds(
    network: &Network,
    input: &BoundedTensor,
) -> Result<SoundIntermediateBounds> {
    let bounds = SoundIntermediateBounds::collect_sound_ibp(network, input)?;
    tracing::info!(
        "[mip-sound-ibp] single-hidden MIP fast path: certified IBP enclosure over {} layers (no CROWN/LP pass)",
        bounds.as_slice().len()
    );
    Ok(bounds)
}
