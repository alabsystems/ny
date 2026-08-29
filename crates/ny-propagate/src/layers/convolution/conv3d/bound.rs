// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound propagation for [`Conv3dLayer`]: certified IBP, fail-closed CROWN.

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use std::time::Instant;
use tracing::debug;

use super::ops_ibp_cert::conv3d_ibp_certified_forward;
use super::Conv3dLayer;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

impl BoundPropagation for Conv3dLayer {
    /// IBP for `y = conv3d(x, W) + b`.
    ///
    /// # Why this is the certified kernel and not the plain f32 W+/W- forward
    ///
    /// Conv1d and Conv2d keep a fast round-to-nearest f32 `W+`/`W-` interval
    /// forward as their DEFAULT `propagate_ibp` and expose the certified
    /// enclosure separately (`propagate_ibp_sound_with_engine*`), because their
    /// f32 GEMM path predates the certificate and a whole dispatch layer
    /// (`Network::propagate_ibp_sound`, `GraphNetwork::propagate_ibp_sound`,
    /// `collect_ibp_bounds_sound`) exists to route soundness-critical callers to
    /// the certified arm. A conv variant that misses ONE of those match arms
    /// silently falls back to `propagate_ibp` plus the driver's 1-ULP widening —
    /// and 1 ULP does not cover a K-term f32 dot product under cancellation
    /// (`W = [2^24, -1, -2^24]` returns exactly `0` where the true value is
    /// `-1`), so the "sound" box EXCLUDES the true output. That is a
    /// false-proof class, and `conv_family_sound_ibp_dispatch.rs` exists
    /// precisely because it has been hit before.
    ///
    /// Conv3d refuses to depend on that wiring. Its ONLY interval forward is
    /// the directed-binary64 certificate in
    /// [`super::ops_ibp_cert::conv3d_ibp_certified_forward`], so every caller —
    /// routed or not, deadline or not, engine or not — gets an enclosure. The
    /// cost is a scalar contraction instead of an im2col GEMM; the benefit is
    /// that no future dispatch site can make this layer unsound by omission.
    /// Whatever widening a generic driver adds on top of an enclosure keeps it
    /// an enclosure.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_with_deadline(input, None)
    }

    /// CROWN backward through Conv3d: FAIL CLOSED.
    ///
    /// No certified transposed-convolution backward exists for spatial rank 3
    /// yet. The alternative to refusing would be publishing a relation this
    /// layer cannot certify — either a silently vacuous plane (`A = 0`,
    /// `b = ±inf`), which reads downstream as a real relation and wastes the
    /// pass, or an uncertified f32 transpose-conv coefficient, which is the
    /// false-proof risk `#vnncomp-aw-soundness` fixed for Conv1d/Conv2d.
    ///
    /// Returning [`NyError::UnsupportedOp`] is the same contract every other
    /// IBP-only layer uses (`AveragePool::propagate_linear`,
    /// `NonZero::propagate_linear`): CROWN drivers treat it as "fall back to
    /// IBP for this node", which is always sound. Conv3d is correspondingly
    /// absent from `Layer::propagates_coeff_err` and
    /// `Layer::supports_batched_crown`, so no dispatcher can route a certified
    /// coefficient error through a backward that would drop it.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Conv3d has no CROWN backward: no certified rank-3-spatial transposed convolution \
             exists, so the layer refuses rather than publish an uncertified (or vacuous) \
             relation. Use IBP for this node."
                .to_string(),
        ))
    }
}

impl Conv3dLayer {
    /// Certified interval forward with an optional deadline.
    ///
    /// `deadline: None` is the ordinary path. With a finite deadline the kernel
    /// polls every bounded work quantum and returns a typed
    /// [`NyError::DeadlineExceeded`]; it is engine-free by construction, so a
    /// finite authority never disappears into an unpollable GEMM.
    ///
    /// Accepts `(C, D, H, W)` and `(N, C, D, H, W)` inputs. The channel count
    /// must equal `kernel.shape()[1] * groups`.
    pub fn propagate_ibp_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (in_c, _) = self.validate_geometry()?;
        let rank = input.lower().ndim();
        if !matches!(rank, 4 | 5) {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0, 0],
                got: input.lower().shape().to_vec(),
            });
        }
        let channel_axis = if rank == 4 { 0 } else { 1 };
        if input.lower().shape()[channel_axis] != in_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c],
                got: vec![input.lower().shape()[channel_axis]],
            });
        }

        debug!(
            "Conv3d certified IBP forward: kernel {:?}, stride {:?}, padding {:?}, dilation {:?}, \
             groups {}",
            self.kernel.shape(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        );
        let forward = conv3d_ibp_certified_forward(
            input.lower().view(),
            input.upper().view(),
            &self.kernel,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            deadline,
        )?;
        // Repair non-finite outputs conservatively, matching every other IBP
        // arm (#3030). `Conservative` only ever widens (NaN -> ±inf, inverted
        // pairs swapped), so it cannot erode the enclosure above.
        BoundedTensor::new_repaired(forward.lower, forward.upper, RepairStrategy::Conservative)
    }

    /// SOUND IBP forward — the Conv3d member of the conv-family certified
    /// forward set (`Conv1dLayer::propagate_ibp_sound_with_engine`,
    /// `Conv2dLayer::propagate_ibp_sound_with_engine`).
    ///
    /// For Conv1d/Conv2d this method is a DIFFERENT, tighter-certified path
    /// than their default `propagate_ibp`. For Conv3d it is the same kernel:
    /// the default already IS the certificate (see
    /// [`BoundPropagation::propagate_ibp`]), so a sound dispatcher that reaches
    /// this entry point and a generic driver that does not both get an
    /// enclosure. It exists so the conv family presents one signature.
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_sound_with_engine_and_deadline(input, engine, None)
    }

    /// Deadline-authoritative certified Conv3d interval forward.
    ///
    /// # The engine parameter
    ///
    /// No accelerator kernel exists for rank-3-spatial convolution, so the
    /// contraction always runs on the pollable CPU certificate and `engine` is
    /// deliberately unused for arithmetic. That is sound (the certificate does
    /// not depend on where a GEMM would have run), but it is NOT free: a
    /// bounded host executor — one whose
    /// [`GemmEngine::forbids_unbounded_cpu_fallback`] is set — is asserting that
    /// no unbounded CPU work may run under its authority. With no deadline to
    /// poll, this kernel is exactly such unbounded work, so it FAILS CLOSED
    /// with [`NyError::UnsupportedOp`] instead of quietly running to
    /// completion. Supply a deadline and the same executor is served by the
    /// pollable arm.
    pub fn propagate_ibp_sound_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        if deadline.is_none() && engine.is_some_and(GemmEngine::forbids_unbounded_cpu_fallback) {
            return Err(NyError::UnsupportedOp(
                "bounded Conv3d IBP requires the explicit deadline-aware entry: the certified \
                 rank-3-spatial forward is a CPU contraction and can only honour a bounded \
                 executor's authority through a deadline it can poll"
                    .to_string(),
            ));
        }
        self.propagate_ibp_with_deadline(input, deadline)
    }
}
