// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP propagation entry points for sequential networks.
//!
//! Moved out of the sequential core module to break bidirectional dependency
//! (#2380).

use crate::network::ibp::NetworkIbpExt;
use crate::network::Network;
use crate::types::CrownIbpBoundsResult;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;

impl Network {
    /// Propagate bounds through the entire network using IBP.
    ///
    /// # REQUIRES
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    ///
    /// # ENSURES
    /// - Output bounds contain all possible network outputs for inputs in `input`
    /// - If network has N layers, output shape matches final layer's output shape
    /// - Soundness: for any `x` where `input.contains(x)`, `output.contains(network(x))`
    #[inline]
    pub fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_ibp_impl(self, input)
    }

    /// Propagate bounds through the entire network using IBP with an optional GEMM engine.
    ///
    /// Linear layers use the supplied engine when present; all other layers keep
    /// their existing CPU IBP implementation.
    #[inline]
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_ibp_with_engine_impl(self, input, engine)
    }

    /// Propagate bounds while preserving a prepended leading axis.
    ///
    /// Batched PGD prepends a restart axis to concrete inputs. Sequential
    /// networks still store Flatten/Reshape layers in the unbatched convention,
    /// so this mode restores the intended per-sample shape contract for those
    /// layers without changing the stored network definition.
    #[inline]
    pub fn propagate_ibp_with_engine_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_ibp_with_engine_preserve_leading_axis_impl(self, input, engine)
    }

    /// True concrete (point) forward at `input`, returning a degenerate
    /// (lower == upper) tensor whose value is a faithful f32 evaluation of the
    /// network — matching ONNX Runtime to ~1e-6 even on deep generators.
    ///
    /// Unlike `propagate_ibp_with_engine` (which propagates a box and, for a point
    /// input, returns a non-trivially-wide box because per-layer soundness widening
    /// is amplified by the deep stack), this collapses to the interval center after
    /// every layer. NON-soundness-critical: for sat-finding / witness evaluation.
    #[inline]
    pub fn propagate_concrete_point(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_concrete_point_impl(self, input, engine)
    }

    /// Concrete point forward (see [`Network::propagate_concrete_point`]) preserving
    /// a prepended leading restart axis, mirroring
    /// `propagate_ibp_with_engine_preserve_leading_axis`.
    #[inline]
    pub fn propagate_concrete_point_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_concrete_point_preserve_leading_axis_impl(self, input, engine)
    }

    /// Propagate bounds with strict soundness guarantees.
    ///
    /// Each layer's output is widened OUTWARD (lower toward -∞, upper toward +∞)
    /// by a certified bound on that layer's OWN floating-point error, so the
    /// returned box encloses the true range rather than its f32 approximation.
    ///
    /// # REQUIRES
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    ///
    /// # ENSURES
    /// - Output bounds contain all possible network outputs for inputs in `input`
    /// - Soundness: for any `x` where `input.contains(x)`, `output.contains(network(x))`
    ///
    /// # Per-layer widening
    /// How much a layer widens by depends on how it accumulates:
    /// - Linear (CERTIFIED): `in_features + 2` ULPs (Higham Thm 3.1 over the two
    ///   matmuls, the combine and the bias).
    /// - Conv1d / Conv2d / ConvTranspose1d / ConvTranspose2d (CERTIFIED): the
    ///   certified `up(γ_{K+2}·S + 2u·|y|)` Higham term for the layer's
    ///   `K`-product window sum, which costs one extra forward pass over
    ///   `|kernel|`. A generic 1-ULP widening is NOT sufficient for these: under
    ///   cancellation an f32 window sum can lose far more than 1 ULP of the RESULT.
    /// - AveragePool (CERTIFIED): the certified `γ⁶⁴_{k+1}·S/d` Higham term for
    ///   the layer's `k`-term f64 window sum (uniform `+1/k` weights), costing one
    ///   extra pooling pass over `max(|l|,|u|)`. The plain forward's outward 1-ULP
    ///   store covers only its f64→f32 cast, not the f64 accumulation residual —
    ///   see `AveragePoolLayer::propagate_ibp_sound`.
    /// - Every other layer (ASSUMPTION, not a certificate): 1 ULP. Valid for ops
    ///   exact in f32 (ReLU/clip/abs, reshapes, MaxPool's exact max), for single-
    ///   rounding arithmetic (one nearest-rounding is ≤ half-ULP), for layers that
    ///   already round their own endpoints outward (e.g. BatchNorm), and for
    ///   pointwise transcendentals
    ///   (Exp, Tanh, Sigmoid, GELU, ...) ONLY IF the platform libm is faithfully
    ///   rounded (≤ 1 ULP). Layers on this arm that ACCUMULATE across many terms
    ///   (Softmax/LogSumExp denominators, LayerNorm/RMSNorm statistics,
    ///   ReduceSum/Mean, CumSum) are NOT certified — their residual can exceed
    ///   1 ULP under the same cancellation (#sound-ibp-generic-arm, open item).
    ///
    /// The resulting extra width is negligible compared to the relaxation
    /// approximation error it is carried alongside.
    ///
    /// # When to Use
    /// - When strict mathematical soundness is required
    /// - For formal verification applications
    /// - When comparing bounds against reference implementations
    #[inline]
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_ibp_sound_impl(self, input)
    }

    /// Strict-soundness IBP forward with an optional GPU engine
    /// (`docs/SOUND_GPU_IBP_PLAN.md` §6.3, T1.1).
    ///
    /// Identical bounds to [`propagate_ibp_sound`](Self::propagate_ibp_sound) but,
    /// when the soundness gate is engaged and `engine` advertises a sound GPU IBP
    /// forward, a SEQUENTIAL dense chain is decided on the certified GPU sound path
    /// (a SUPERSET of both the true range and the CPU sound bound). Any failure falls
    /// back to the proven-sound CPU loop, so a verdict is never decided by a failed
    /// GPU op. `engine == None` is exactly the CPU loop.
    #[inline]
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkIbpExt::propagate_ibp_sound_with_engine_impl(self, input, engine)
    }

    /// Run IBP forward pass and collect bounds at each layer.
    ///
    /// Returns a vector of bounds, where `bounds\[i\]` is the output of layer i.
    /// The input bounds are NOT included in the returned vector.
    pub fn collect_ibp_bounds(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        NetworkIbpExt::collect_ibp_bounds_impl(self, input)
    }

    /// Run IBP forward pass and collect bounds at each layer with directed rounding.
    ///
    /// Like `collect_ibp_bounds`, but every returned `bounds[i]` carries the same
    /// certified per-layer widening as [`propagate_ibp_sound`](Self::propagate_ibp_sound)
    /// — see that method for the per-layer rule — so each intermediate box encloses
    /// the true range of that layer and can be trusted by a stability classifier.
    pub fn collect_ibp_bounds_sound(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        NetworkIbpExt::collect_ibp_bounds_sound_impl(self, input)
    }

    /// Run CROWN-IBP to collect tighter intermediate bounds.
    ///
    /// This method computes tighter bounds than pure IBP by using CROWN backward
    /// propagation for each intermediate layer. For each layer k, it:
    /// 1. Runs CROWN backward from layer k to the input
    /// 2. Takes the intersection (tighter) of IBP and CROWN bounds
    ///
    /// This is more expensive than pure IBP (O(n) CROWN passes) but produces
    /// significantly tighter intermediate bounds, which leads to tighter ReLU
    /// relaxations and ultimately tighter final bounds.
    ///
    /// Returns a vector of bounds, where `bounds[i]` is the (tightened) output of layer i.
    pub fn collect_crown_ibp_bounds(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        NetworkIbpExt::collect_crown_ibp_bounds_impl(self, input)
    }

    /// Run CROWN-IBP and return per-layer fallback diagnostics.
    ///
    /// This method reports when CROWN could not tighten a layer and the implementation
    /// had to keep forward bounds (for example due to CROWN errors, shape mismatch,
    /// or empty forward/CROWN intersections).
    pub fn collect_crown_ibp_bounds_with_status(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownIbpBoundsResult> {
        NetworkIbpExt::collect_crown_ibp_bounds_with_status_impl(self, input)
    }

    pub fn collect_crown_ibp_bounds_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>> {
        NetworkIbpExt::collect_crown_ibp_bounds_with_engine_impl(self, input, engine)
    }

    /// Run CROWN-IBP with custom GEMM engine and return fallback diagnostics.
    pub fn collect_crown_ibp_bounds_with_engine_and_status(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownIbpBoundsResult> {
        NetworkIbpExt::collect_crown_ibp_bounds_with_engine_and_status_impl(self, input, engine)
    }

    /// Run CROWN-IBP with deadline enforcement (#3328).
    ///
    /// A local per-layer budget may select an IBP bound while the caller's
    /// deadline remains live. Exhausting the caller deadline itself is a typed
    /// refusal: the collector never clones remaining bounds after expiry.
    pub fn collect_crown_ibp_bounds_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Vec<BoundedTensor>> {
        Ok(
            NetworkIbpExt::collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
                self, input, engine, deadline,
            )?
            .bounds,
        )
    }

    /// Run CROWN-IBP with pre-computed per-layer IBP bounds (#3397).
    ///
    /// Skips the internal IBP forward pass, using `precomputed_ibp` directly.
    /// Saves ~59s for soundnessbench-scale models where IBP is already available.
    pub fn collect_crown_ibp_bounds_with_precomputed_ibp(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Vec<BoundedTensor>> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            if precomputed_ibp.len() != self.layers().len() {
                return Err(NyError::InvalidSpec(format!(
                    "pre-computed IBP bounds have {} entries, expected {} (one per layer)",
                    precomputed_ibp.len(),
                    self.layers().len()
                )));
            }
            // This bounds-only API owns the complete certified vector. Moving
            // it to the caller is O(1) and requires no post-expiry scan, clone,
            // or fallback computation.
            return Ok(precomputed_ibp);
        }
        Ok(
            NetworkIbpExt::collect_crown_ibp_bounds_with_precomputed_ibp_impl(
                self,
                input,
                precomputed_ibp,
                engine,
                deadline,
            )?
            .bounds,
        )
    }

    /// Run IBP forward pass and collect bounds at each layer with deadline (#3328).
    pub fn collect_ibp_bounds_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<Vec<BoundedTensor>> {
        NetworkIbpExt::collect_ibp_bounds_with_deadline_impl(self, input, deadline)
    }
}
