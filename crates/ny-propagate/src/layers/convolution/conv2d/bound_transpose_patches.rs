// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-native CROWN backward for `ConvTranspose2dLayer` (LEVER 2 stage 2a).
//!
//! The dense ConvTranspose2d CROWN backward
//! (`bound_transpose.rs::propagate_linear_with_engine`) materializes a full
//! `[target_dim x conv_in]` dense coefficient pair. On the 28,800-dim cGAN
//! `BatchNormalization_11` target this measured 52.5s, starving alpha-opt and
//! BaB. This module keeps the coefficient in patches form (O(spec_rows x window
//! x kernel), ~ms) for the **stride-1** case, falling back to the exact dense
//! path for every other configuration.
//!
//! ## The reduction (stage 2a scope)
//!
//! A `ConvTranspose2d` forward `y = conv_transpose(x, W)` (ONNX kernel layout
//! `W[in_c, out_c, kh, kw]`, stride 1, dilation 1, output_padding 0, padding
//! `(ph, pw)`) has a CROWN backward that is *exactly* the CROWN backward of a
//! plain `Conv2d` with
//!   - kernel `Kc[oc, ic, ki, kj] = W[ic, oc, kh-1-ki, kw-1-kj]`
//!     (in/out channels swapped **and** the kernel spatially flipped), and
//!   - padding `Cp = (kh-1-ph, kw-1-pw)`, and
//!   - the SAME bias (per output channel, broadcast over the output grid).
//!
//! This is the standard "the adjoint of a transposed convolution is a
//! convolution" identity specialized to stride 1; it is verified numerically
//! against the dense `conv2d_forward_batched_gemm` backward (the exact operator
//! the dense path evaluates) in the proptest harness and unit tests. Reducing to
//! `Conv2dLayer::propagate_patches_engine` means this stride-1 ConvTranspose
//! backward introduces **zero new bound math**: the identity build, non-identity
//! composition, certified `coeff_err`, the outward-rounded bias, and the
//! `should_fallback_to_dense` memory guard are all the already-proven Conv2d
//! patches path (`bound_patches.rs`), which is pinned bit-equivalent-to-dense by
//! `crown_patches.rs`.
//!
//! ## Corners routed to the sound dense fallback
//!
//! Every configuration outside the stride-1 reduction returns
//! `UnsupportedConfiguration`, which the patches dispatcher
//! (`patches_step.rs`) turns into `ensure_dense()` +
//! `propagate_linear_with_engine` — the exact dense CROWN backward. Never a
//! silently-wrong bound. These are:
//!   - `stride != (1, 1)` — see the STAGE 2b note below;
//!   - `dilation != (1, 1)`;
//!   - `output_padding != (0, 0)` (unreachable for stride 1, guarded anyway);
//!   - `padding.0 > kh-1` or `padding.1 > kw-1` — the equivalent Conv2d padding
//!     `kh-1-ph` would be negative (not representable without cropping);
//!   - `input_shape` unset;
//!   - a NaN kernel (→ `NumericalInstability`);
//!   - and, inside the delegated Conv2d path, non-identity incoming patches that
//!     carry nonzero composed padding (the Conv2d composition soundness guard).
//!
//! ## STAGE 2b (stride>1): stays on the sound dense fallback — why
//!
//! A forward stride-s ConvTranspose upsamples, so its CROWN backward
//! *downsamples* the coefficient grid (output pixels -> input pixels, /s). The
//! phase-partition (split the output grid into the s^2 residue classes
//! `(oh mod s, ow mod s)`; each class is a stride-1 ConvTranspose backward on the
//! decimated sub-grid with the per-phase kernel slice `W[:, :, (ph+a)%s :: s,
//! (pw+b)%s :: s]`, reducing to the stage-2a Conv2d patches path) COMPUTES the
//! backward correctly — validated bit-exact to the dense path for stride-2 and
//! stride-3 in `crown_patches_convtranspose.rs`
//! (`proptest_convtranspose2d_phase_partition_{identity,nonidentity}`).
//!
//! But the reassembled coefficient cannot be held in the memory-light
//! `PatchesData`: that representation positions each spec row's receptive field
//! at `spec_pos * stride + tap - pad` with an INTEGER `stride >= 1` (an
//! *upsampling* map), whereas the ConvTranspose backward needs
//! `⌊(oh + ph)/s⌋ + tap` (a *downsampling*/floor map, a step function of the spec
//! index) — not expressible for any integer stride. A single `PatchesData` (6D or
//! 7D) therefore cannot encode the result; keeping stride>1 in patches would
//! require a new multi-phase geometry threaded soundly through EVERY
//! geometry-aware patches consumer (`to_dense`, every element-wise activation
//! backward, pooling, the conv compose, the coeff_err scatter, merge) — a
//! missed consumer would be silently unsound. (Additionally the per-phase
//! equiv-Conv2d reduction only size-matches for `padding == 0`.) Per the
//! soundness mandate, stride>1 returns `UnsupportedConfiguration` here so the
//! caller takes the exact dense CROWN backward.

use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};

use super::{Conv2dLayer, ConvTranspose2dLayer};
use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;

impl PatchesPropagation for ConvTranspose2dLayer {
    /// CROWN backward with Patches coefficients for a **stride-1**
    /// ConvTranspose2d. Delegates to the engine-aware variant with no engine.
    ///
    /// Supports identity incoming patches (first layer in the backward chain)
    /// and non-identity composition, both handled by the reduction to the
    /// Conv2d patches path. Any non-stride-1 / unsupported corner returns
    /// `UnsupportedConfiguration` so the caller falls back to the sound dense
    /// CROWN backward.
    fn propagate_patches(&self, bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        self.propagate_patches_engine(bounds, None)
    }
}

impl ConvTranspose2dLayer {
    /// Build the flipped-and-channel-swapped Conv2d kernel `Kc` whose CROWN
    /// backward equals this ConvTranspose2d's CROWN backward (stride 1):
    /// `Kc[oc, ic, ki, kj] = W[ic, oc, kh-1-ki, kw-1-kj]`.
    ///
    /// `W` is the ONNX ConvTranspose layout `(in_c, out_c, kh, kw)`; `Kc` is the
    /// Conv2d layout `(out_c, in_c, kh, kw)`. Pure data movement (no bound math).
    fn crown_backward_equivalent_conv2d_kernel(&self) -> ArrayD<f32> {
        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (kh, kw) = self.kernel_size();
        let mut kc = ArrayD::<f32>::zeros(IxDyn(&[out_c, in_c, kh, kw]));
        for oc in 0..out_c {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        kc[[oc, ic, ki, kj]] = self.kernel[[ic, oc, kh - 1 - ki, kw - 1 - kj]];
                    }
                }
            }
        }
        kc
    }

    /// Engine-aware patches ConvTranspose2d CROWN backward (stride-1 only).
    ///
    /// Mirrors `Conv2dLayer::propagate_patches_engine` by *being* it: this method
    /// rejects every non-stride-1 / unsupported corner up front (routing them to
    /// the sound dense fallback via `UnsupportedConfiguration`), then reduces the
    /// stride-1 case to the equivalent `Conv2d` (flip+swap kernel, adjusted
    /// padding, same bias) and delegates to the proven Conv2d patches path. The
    /// returned `CrownBounds` (Patches, or Dense if the Conv2d path decided the
    /// composed patches no longer save memory) is over this layer's INPUT space,
    /// exactly like the dense `propagate_linear_with_engine`.
    pub(crate) fn propagate_patches_engine(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBounds> {
        // Guard: reject NaN weights (mirrors the Conv2d patches entry guard).
        if self.kernel.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "ConvTranspose2d Patches backward: kernel contains NaN".into(),
            ));
        }

        // STRIDE-1 only. stride>1 is the phase-partition (stage 2b): the reduction
        // is correct (validated bit-exact to dense in
        // crown_patches_convtranspose.rs) but its downsampling (floor) coefficient
        // map is not representable in the memory-light PatchesData (integer
        // upsampling stride only) — see the STAGE 2b module note. Route it (and
        // every other unsupported corner) to the exact dense CROWN path so the
        // caller never gets a silently-wrong bound.
        if self.stride != (1, 1) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN supports only stride 1; stride>1 (stage 2b) \
                 is not representable in patches form; got stride {:?}; use dense CROWN",
                self.stride
            )));
        }
        if self.dilation != (1, 1) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN does not support dilation {:?}; use dense CROWN",
                self.dilation
            )));
        }
        // For stride 1 the layer constructor already forces output_padding == 0
        // (output_padding < stride); guard defensively regardless.
        if self.output_padding != (0, 0) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN does not support output_padding {:?}; use dense CROWN",
                self.output_padding
            )));
        }

        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d Patches CROWN requires input_shape to be set. \
                 Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let (kh, kw) = self.kernel_size();
        let (ph, pw) = self.padding;
        // The equivalent Conv2d padding is Cp = (kh-1-ph, kw-1-pw). When the
        // ConvTranspose padding exceeds kernel-1 in a dimension, Cp would be
        // negative — not representable as a Conv2d padding without cropping the
        // kernel/patch. Route that corner to the sound dense path.
        if ph > kh - 1 || pw > kw - 1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN cannot represent padding ({ph},{pw}) > kernel-1 \
                 ({},{}) as an equivalent Conv2d padding (stage 2a); use dense CROWN",
                kh - 1,
                kw - 1
            )));
        }
        let cph = kh - 1 - ph;
        let cpw = kw - 1 - pw;

        // Reduce to the equivalent Conv2d and reuse its proven patches path
        // (identity build, non-identity composition, certified coeff_err, and the
        // outward-rounded bias). The Conv2d INPUT space equals this
        // ConvTranspose's INPUT space ((in_c, in_h, in_w)), and the Conv2d OUTPUT
        // space equals this ConvTranspose's OUTPUT space ((out_c, out_h, out_w)),
        // so the incoming patches (over the ConvTranspose output space) and the
        // result (over the ConvTranspose input space) map through unchanged.
        let kc = self.crown_backward_equivalent_conv2d_kernel();
        let equiv_conv =
            Conv2dLayer::with_input_shape(kc, self.bias.clone(), (1, 1), (cph, cpw), in_h, in_w)?;
        equiv_conv.propagate_patches_engine(bounds, engine)
    }
}
