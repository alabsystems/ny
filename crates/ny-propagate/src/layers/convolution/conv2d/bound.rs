// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use std::time::Instant;
use tracing::{debug, warn};

use super::super::crown_helpers::{
    compute_conv_bias_f64, compute_conv_bias_f64_with_poll, detect_and_fix_nonfinite_rows,
    detect_and_fix_nonfinite_rows_with_poll, guard_nan_weights,
};
use super::ops_ibp_fwd::{
    conv2d_ibp_forward_grouped, conv2d_ibp_forward_grouped_certified_f64_with_deadline,
    conv2d_ibp_forward_grouped_with_deadline,
};
use super::ops_ibp_gemm::propagate_ibp_via_gemm;
use super::{
    conv2d_transpose_backward_coeff_f64_with_deadline,
    conv2d_transpose_backward_coeff_f64_with_engine_and_deadline,
    conv2d_transpose_pair_batched_gemm_grouped_with_deadline, Conv2dLayer,
};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

const DEADLINE_IBP_POLL_ELEMENTS: usize = 4_096;

fn check_ibp_deadline(deadline: Option<Instant>, stage: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "Conv2d IBP forward: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

#[inline]
fn poll_crown_backward_authority(
    bounded_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<()> {
    if let Some(engine) = bounded_engine {
        engine.poll_crown_backward_deadline()?;
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    Ok(())
}

impl BoundPropagation for Conv2dLayer {
    /// IBP for Conv2d layer: y = conv(x, W) + b
    ///
    /// For x in [l, u], compute y bounds:
    /// - W+ = max(W, 0), W- = min(W, 0)
    /// - lower_y = conv(l, W+) + conv(u, W-) + b
    /// - upper_y = conv(u, W+) + conv(l, W-) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // No engine: pure CPU faer path (engine-agnostic; see propagate_ibp_inner).
        self.propagate_ibp_inner(input, None)
    }

    /// CROWN backward propagation through Conv2d layer (CPU path).
    ///
    /// Delegates to `propagate_linear_with_engine` with `engine: None`.
    /// For GPU-accelerated path, use `Conv2dLayer::propagate_linear_with_engine`.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine(bounds, None)
    }
}

impl Conv2dLayer {
    /// IBP forward for Conv2d, optionally routing the four W+/W- im2col GEMMs
    /// through an injected [`GemmEngine`] (GPU/accelerator).
    ///
    /// `engine: None` is the pure CPU faer path. `engine: Some(_)` dispatches
    /// each per-group matmul through `engine.gemm_f32(...)` inside
    /// `conv2d_ibp_forward_grouped`, falling back to CPU faer per-matmul on
    /// engine error. The W+/W- interval decomposition and bias add are identical
    /// in both cases, so the engine path yields the same sound bounds (#hot-conv-ibp).
    fn propagate_ibp_inner(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_inner_with_deadline(input, engine, None)
    }

    fn propagate_ibp_inner_with_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        check_ibp_deadline(deadline, "before entry")?;
        let (in_c, _) = self.validate_geometry()?;

        match input.lower().ndim() {
            3 => {
                // Input shape: (in_channels, height, width)
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }

                // im2col + GEMM interval forward (W+/W- splitting) replaces the
                // four naive nested-loop conv2d_single_grouped calls (#hot-conv-ibp).
                let fwd = if let Some(limit) = deadline {
                    conv2d_ibp_forward_grouped_with_deadline(
                        input.lower().view(),
                        input.upper().view(),
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                        limit,
                    )?
                } else {
                    conv2d_ibp_forward_grouped(
                        input.lower(),
                        input.upper(),
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                        engine,
                    )?
                };
                let mut lower_y = fwd.lower;
                let mut upper_y = fwd.upper;

                // Add bias if present (broadcast over spatial dimensions)
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_h = fwd.out_h;
                    let out_w = fwd.out_w;

                    let mut bias_elements = 0usize;
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                bias_elements += 1;
                                if bias_elements == DEADLINE_IBP_POLL_ELEMENTS {
                                    check_ibp_deadline(deadline, "while adding bias")?;
                                    bias_elements = 0;
                                }
                                lower_y[[oc, oh, ow]] += b[oc];
                                upper_y[[oc, oh, ow]] += b[oc];
                            }
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                let result =
                    BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)?;
                check_ibp_deadline(deadline, "before publishing the result")?;
                Ok(result)
            }
            4 => {
                // Input shape: (batch, in_channels, height, width)
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }

                let batch = input.lower().shape()[0];
                let input_h = input.lower().shape()[2];
                let input_w = input.lower().shape()[3];
                let (out_h, out_w) = self.output_size(input_h, input_w)?;
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));

                for b in 0..batch {
                    check_ibp_deadline(deadline, "before a batch item")?;
                    let lower_b = input.lower().index_axis(ndarray::Axis(0), b);
                    let upper_b = input.upper().index_axis(ndarray::Axis(0), b);
                    let fwd = if let Some(limit) = deadline {
                        conv2d_ibp_forward_grouped_with_deadline(
                            lower_b,
                            upper_b,
                            &self.kernel,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups,
                            limit,
                        )?
                    } else {
                        conv2d_ibp_forward_grouped(
                            &lower_b.to_owned().into_dyn(),
                            &upper_b.to_owned().into_dyn(),
                            &self.kernel,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups,
                            engine,
                        )?
                    };
                    let lower_batch = fwd.lower;
                    let upper_batch = fwd.upper;

                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                lower_y[[b, oc, oh, ow]] = lower_batch[[oc, oh, ow]];
                                upper_y[[b, oc, oh, ow]] = upper_batch[[oc, oh, ow]];
                            }
                        }
                    }
                }

                // Add bias if present (broadcast over batch/spatial dimensions)
                if let Some(ref bias) = self.bias {
                    let mut bias_elements = 0usize;
                    for b in 0..batch {
                        for oc in 0..out_c {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    bias_elements += 1;
                                    if bias_elements == DEADLINE_IBP_POLL_ELEMENTS {
                                        check_ibp_deadline(deadline, "while adding batched bias")?;
                                        bias_elements = 0;
                                    }
                                    lower_y[[b, oc, oh, ow]] += bias[oc];
                                    upper_y[[b, oc, oh, ow]] += bias[oc];
                                }
                            }
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                let result =
                    BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)?;
                check_ibp_deadline(deadline, "before publishing the batched result")?;
                Ok(result)
            }
            _ => Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower().shape().to_vec(),
            }),
        }
    }

    /// IBP propagation with optional GEMM-engine acceleration.
    ///
    /// For `groups == 1` this uses the GPU-resident `propagate_ibp_via_gemm`
    /// helper (batched-GEMM forward), which is the most efficient single-group
    /// path. For `groups > 1` — which `propagate_ibp_via_gemm` does not support
    /// — the engine is threaded through the grouped im2col+GEMM forward
    /// (`conv2d_ibp_forward_grouped`) so the dominant conv GEMMs of grouped /
    /// depthwise convs are still offloaded to the engine instead of silently
    /// falling back to CPU. Both engine paths degrade to CPU faer on failure.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let Some(engine) = engine else {
            return self.propagate_ibp(input);
        };
        if engine.forbids_unbounded_cpu_fallback() {
            return Err(NyError::UnsupportedOp(
                "bounded Conv2d IBP requires the explicit deadline-aware entry".into(),
            ));
        }
        if self.groups != 1 {
            // Grouped/depthwise: route the per-group GEMMs through the engine via
            // the im2col+GEMM grouped forward instead of dropping to CPU (#hot-conv-ibp).
            // `conv2d_ibp_forward_grouped` already degrades per-matmul to CPU faer
            // on engine error, so this stays sound.
            return self.propagate_ibp_inner(input, Some(engine));
        }

        match propagate_ibp_via_gemm(self, input, engine) {
            Ok(bounds) => Ok(bounds),
            Err(err) => {
                debug!("Conv2d IBP GemmEngine path failed, falling back to CPU: {err}");
                self.propagate_ibp(input)
            }
        }
    }

    /// Deadline-authoritative Conv2d interval forward.
    ///
    /// `deadline: None` preserves [`Self::propagate_ibp_with_engine`] exactly.
    /// With a finite deadline, neither the caller engine nor faer's unpollable
    /// GEMM is entered; grouped/depthwise and dense convolutions use a direct
    /// CPU contraction with bounded polling quanta.
    pub fn propagate_ibp_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        if deadline.is_none() {
            return self.propagate_ibp_with_engine(input, engine);
        }
        self.propagate_ibp_inner_with_deadline(input, None, deadline)
    }

    /// SOUND IBP forward: [`Self::propagate_ibp_with_engine`] accumulates each output
    /// over `K = (in_c/groups)*kh*kw` products in round-to-nearest f32 (`gemm_f32`,
    /// no f64, no directed rounding). The generic IBP driver then widens non-Linear
    /// layers by only **1 ULP** (`round_for_soundness`), but a K-term f32 dot product
    /// can deviate from the true value by up to the Higham bound
    /// `γ_K · Σ_k |W_ok|·|x_k|` — which, under cancellation, vastly exceeds 1 ULP
    /// (e.g. `W=[2^24,1,-2^24,4]` loses ~1.0 of magnitude). The under-widened box can
    /// then exclude the true value → a wrong VERIFIED on the verdict / intermediate-
    /// bound path. This is the conv IBP-**forward** analogue of the conv CROWN-backward
    /// f32/error mismatch fixed in `becc501` (#vnncomp-aw-soundness).
    ///
    /// Fix: add the certified Higham error term `err_o = γ_{K+2} · S_o + 2u·|y_o|`
    /// (with `S_o = Σ_k |W_ok|·max(|x_l_k|,|x_u_k|)` computed by the SAME interval
    /// forward run on `|kernel|` and a degenerate `max(|l|,|u|)` input, so it handles
    /// 3D/4D/grouped uniformly), rounded outward. SOUND: the returned box strictly
    /// encloses the true conv output; looser only → Timeout, never a false proof.
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_sound_with_engine_and_deadline(input, engine, None)
    }

    /// Deadline-authoritative certified Conv2d interval forward.
    ///
    /// # Finite-deadline arm (REPLACED 3-pass structure)
    ///
    /// With `deadline: Some(_)` this returns the f64 dual-accumulator kernel
    /// [`conv2d_ibp_forward_grouped_certified_f64_with_deadline`] directly,
    /// SKIPPING the y-then-S-then-widen 3-pass structure below entirely. The
    /// old f32 3-pass deadline arm charged
    /// `err_o = γ_{K+2}^{f32}·S_safe + 2u·|y_o|` with
    /// `S_o = Σ_k |W_ok|·max(|l_k|,|u_k|)`; since `S` scales with activation
    /// MAGNITUDE, not box width, that widening is a floor input splitting
    /// cannot erode. Bisect + config-discriminator evidence pinned commit
    /// `6f49a660` (which routed every finite-deadline conv IBP through that
    /// arm) as the regression that stopped cgan BaB trees from closing (row
    /// `cGAN_imgSz32_nCh_3_prop_2`: unsat@640s before, timeout@1800s after;
    /// starvation knobs and `ibp_enhancement` A/Bs ruled out the
    /// alternatives). The fix is MORE PRECISION, not reversion: the audit gap
    /// `6f49a660` closed ("no finite deadline may enter an opaque engine/faer
    /// kernel without a cancellation contract") stays closed — the new arm is
    /// pollable, engine-free, scalar-CPU (#cgan-conv-ibp-magnitude-floor).
    ///
    /// # `deadline: None` arm
    ///
    /// Keeps the historical engine/faer route byte-identical: the plain
    /// interval forward plus the `|W|·max(|l|,|u|)` abssum pass and the
    /// `γ_{K+2}^{f32}·S_safe + 2u·|y|` outward widening, exactly as before.
    pub fn propagate_ibp_sound_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        if let Some(limit) = deadline {
            return self.propagate_ibp_sound_certified_f64_with_deadline(input, limit);
        }
        let y = self.propagate_ibp_with_engine(input, engine)?;

        // S_o = Σ_k |W_ok| · max(|x_l_k|, |x_u_k|): run the interval forward with the
        // absolute kernel (so W+ = |W|, W- = 0) on the degenerate `xmax` box.
        let lo = input.lower();
        let up = input.upper();
        let xmax = {
            let mut result = lo.mapv(f32::abs);
            ndarray::Zip::from(&mut result)
                .and(up)
                .for_each(|m, &u| *m = m.max(u.abs()));
            result
        };
        let abs_kernel = self.kernel.mapv(f32::abs);
        let abs_layer = Conv2dLayer::new_dilated(
            abs_kernel,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        let xmax_bounds = BoundedTensor::concrete(xmax)?;
        let s_bt = abs_layer.propagate_ibp_with_engine(&xmax_bounds, engine)?;
        let s = s_bt.lower(); // |kernel|, xmax ≥ 0 ⇒ lower == upper == S

        // Higham growth factor γ_{K+2} (u = 2^-24), +2 covers the W+/W- combine and
        // bias add on top of the K window MACs. Saturate to +inf if K·u ≥ 1.
        let (kh, kw) = self.kernel_size();
        let in_per_group = self.in_channels() / self.groups;
        let macs = in_per_group.saturating_mul(kh).saturating_mul(kw);
        let k = (macs.saturating_add(2)) as f64;
        const U: f64 = 1.0 / (1u64 << 24) as f64; // f32 unit roundoff 2^-24
        let gamma = if k * U < 1.0 {
            (k * U) / (1.0 - k * U)
        } else {
            f64::INFINITY
        };
        // S_o is computed by a round-to-NEAREST f32 abs-conv, so S_f32 can fall SHORT
        // of the true abssum S by up to its own Higham accumulation error γ_macs·S.
        // Inflate it OUTWARD by 1/(1−γ_macs) ≥ that deficit so S_safe ≥ S_true —
        // otherwise γ·S_f32 can UNDER-bound the true Higham error for very large convs
        // (macs > ~5793, e.g. 1024-ch 3×3). (#vnncomp-aw-soundness self-audit.)
        let gamma_macs = {
            let m = macs as f64;
            if m * U < 1.0 {
                (m * U) / (1.0 - m * U)
            } else {
                f64::INFINITY
            }
        };
        let s_inflate = if gamma_macs < 1.0 {
            1.0 / (1.0 - gamma_macs)
        } else {
            f64::INFINITY
        };

        let (mut lower, mut upper) = (y.lower().to_owned(), y.upper().to_owned());
        let widen = |lo_o: &mut f32, up_o: &mut f32, s_o: f32| {
            let mag = (lo_o.abs()).max(up_o.abs()) as f64;
            // err_o = up( γ·S_safe + 2u·|y_o| ): covers the K MACs (γ·S) plus the
            // bias add / W+/W- combine roundings (2u·|y|). S_safe = S_f32·s_inflate
            // ≥ S_true (S computed in round-to-nearest f32 above).
            let s_safe = s_o as f64 * s_inflate;
            let err = next_up_f32((gamma * s_safe + 2.0 * U * mag) as f32);
            if err.is_finite() {
                *lo_o = next_down_f32(*lo_o - err);
                *up_o = next_up_f32(*up_o + err);
            } else {
                *lo_o = f32::NEG_INFINITY;
                *up_o = f32::INFINITY;
            }
        };
        ndarray::Zip::from(&mut lower)
            .and(&mut upper)
            .and(s)
            .for_each(|lo_o, up_o, &s_o| widen(lo_o, up_o, s_o));
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    /// Finite-deadline certified arm: the f64 dual-accumulator kernel
    /// [`conv2d_ibp_forward_grouped_certified_f64_with_deadline`] (which holds
    /// the full error-bound derivation), applied per batch item for 4D inputs.
    /// Bias is folded into the kernel's f64 accumulation (and its abs-sum), so
    /// no separate bias pass runs. Engine-free by construction; the output is
    /// repaired conservatively for consistency with the other IBP arms
    /// (#3030) and the deadline is re-checked before publication.
    fn propagate_ibp_sound_certified_f64_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Instant,
    ) -> Result<BoundedTensor> {
        let limit = deadline;
        let deadline = Some(limit);
        check_ibp_deadline(deadline, "before certified propagation")?;
        let (in_c, _) = self.validate_geometry()?;

        match input.lower().ndim() {
            3 => {
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }
                let fwd = conv2d_ibp_forward_grouped_certified_f64_with_deadline(
                    input.lower().view(),
                    input.upper().view(),
                    &self.kernel,
                    self.bias.as_ref(),
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    limit,
                )?;
                let result = BoundedTensor::new_repaired(
                    fwd.lower,
                    fwd.upper,
                    RepairStrategy::Conservative,
                )?;
                check_ibp_deadline(deadline, "before publishing certified bounds")?;
                Ok(result)
            }
            4 => {
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }
                let batch = input.lower().shape()[0];
                let input_h = input.lower().shape()[2];
                let input_w = input.lower().shape()[3];
                let (out_h, out_w) = self.output_size(input_h, input_w)?;
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));
                let mut elements = 0usize;
                for b in 0..batch {
                    check_ibp_deadline(deadline, "before a certified batch item")?;
                    let lower_b = input.lower().index_axis(ndarray::Axis(0), b);
                    let upper_b = input.upper().index_axis(ndarray::Axis(0), b);
                    let fwd = conv2d_ibp_forward_grouped_certified_f64_with_deadline(
                        lower_b,
                        upper_b,
                        &self.kernel,
                        self.bias.as_ref(),
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                        limit,
                    )?;
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                elements += 1;
                                if elements == DEADLINE_IBP_POLL_ELEMENTS {
                                    check_ibp_deadline(
                                        deadline,
                                        "while scattering certified batch bounds",
                                    )?;
                                    elements = 0;
                                }
                                lower_y[[b, oc, oh, ow]] = fwd.lower[[oc, oh, ow]];
                                upper_y[[b, oc, oh, ow]] = fwd.upper[[oc, oh, ow]];
                            }
                        }
                    }
                }
                let result =
                    BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)?;
                check_ibp_deadline(deadline, "before publishing certified bounds")?;
                Ok(result)
            }
            _ => Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower().shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through Conv2d layer with optional GemmEngine.
    ///
    /// When `engine` is `Some`, dispatches the GEMM to the provided engine (GPU).
    /// Falls back to CPU faer GEMM on engine failure or when engine is `None`.
    /// Requires `input_shape` to be set for proper shape computation.
    ///
    /// Reference: linear/crown_single.rs `propagate_linear_with_engine` for the
    /// same pattern applied to Linear layers (#3399).
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine_and_deadline(bounds, engine, None)
    }

    /// CROWN backward propagation with optional deadline enforcement.
    ///
    /// When `deadline` is present, large GEMM workloads are chunked so the
    /// backward pass can abort inside a single Conv2d node instead of running
    /// past the verifier timeout. Part of #3795.
    pub fn propagate_linear_with_engine_and_deadline<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Cow<'a, LinearBounds>> {
        let input_shape = self.input_shape.ok_or_else(|| NyError::UnsupportedConfiguration(
            "Conv2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape().".to_string()
        ))?;
        self.propagate_linear_with_engine_and_deadline_for_input_shape(
            bounds,
            engine,
            deadline,
            input_shape,
        )
    }

    /// Borrowing variant for callers that have already authenticated the
    /// current pre-activation shape and therefore need not deep-clone this
    /// layer just to update `input_shape`.
    pub(crate) fn propagate_linear_with_engine_and_deadline_for_input_shape<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        (in_h, in_w): (usize, usize),
    ) -> Result<Cow<'a, LinearBounds>> {
        debug!("Conv2d layer CROWN backward propagation");
        let (in_c, out_c) = self.validate_geometry()?;

        // Guard: reject NaN weights at CROWN backward entry. The IBP path uses
        // nan_propagating_max_zero/min_zero which handle NaN in kernel splitting.
        // The CROWN path passes self.kernel directly to conv2d_transpose, which
        // would silently produce NaN coefficient matrices. (#2747)
        guard_nan_weights(&self.kernel, self.bias.as_ref(), "Conv2d")?;

        let (out_h, out_w) = self.output_size(in_h, in_w)?;

        // Verify that bounds dimensions match expected conv output
        let expected_conv_out = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        if bounds.num_inputs() != expected_conv_out {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;

        // Batched GEMM replaces per-row conv2d_transpose with per-group GEMM + col2im.
        // For groups=1, this is a single GEMM. For groups>1, each group gets its own GEMM.
        // Design: designs/2026-03-06-conv-crown-backward-gemm.md (#3382).
        // #3399: engine parameter enables GPU acceleration via GemmEngine.
        // #3770: groups parameter enables depthwise/grouped convolutions.
        // Fused lower+upper conv_transpose. Both A-matrices share the same
        // Conv2d kernel, so the GPU engine keeps the per-group weight column
        // GPU-resident across the two and stacks them into one dispatch (pure
        // perf, bit-identical). On the CPU/dilated/chunked-deadline paths this
        // delegates to two independent per-matrix calls — identical to the
        // prior two-call behavior, including the inter-GEMM deadline check
        // (#3795). See conv2d_transpose_pair_batched_gemm_grouped_with_deadline.
        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug).
        // The f32 GEMM+col2im pair accumulates `Σ a·kernel` in f32; its real
        // error is `γ_n^f32·S` (~2^29× the f64 factor), NOT `γ_n^f64·S`. Using the
        // f64 factor (as the prior code did) UNDER-counts the f32 coefficient
        // error → FALSE PROOF on wide-contraction convs. Fix exactly like Linear's
        // `aw_f64_with_abssum`: on wide contractions re-accumulate the SAME
        // contraction in f64 (exact f32→f64 widening; only the f64 sum rounds),
        // store the directed-to-nearest f32 of that f64 result as the point
        // coefficient, and certify a per-coefficient `cast_err = |f64 − stored_f32|`
        // PLUS `γ_n^f64·S`. On SMALL contractions skip the f64 recompute and certify
        // the f32 GEMM coefficient with `γ_n^f32·S` (also sound, and tight for small
        // n) — see crown_helpers::conv_should_f64_recompute. A WANTED-but-failed
        // recompute degrades the row to ±inf bias (sound).
        let (kh, kw) = self.kernel_size();
        let n_contraction = out_c.saturating_mul(kh).saturating_mul(kw);
        let want_recompute = super::super::crown_helpers::conv_should_f64_recompute(n_contraction);
        let bounded_deadline_engine =
            super::ops_transpose_gemm::bounded_pollable_host_engine(engine)?;
        let finite_authority = bounded_deadline_engine.is_some() || deadline.is_some();
        let mut poll_authority =
            || poll_crown_backward_authority(bounded_deadline_engine, deadline);
        // #wall-deadwork (default-on; `NY_CONV_SKIP_DEAD_F32=0` is the kill-switch):
        // under `want_recompute` the pair's A-values are discarded on BOTH paths
        // below (success → overwritten with the rounded f64 recompute; failure →
        // row degraded to ±inf bias), so the pair contributes only the buffers and
        // the per-node deadline check. Skip it, allocating directly and keeping an
        // explicit deadline check in its place; the recompute-failure degrade path
        // is unchanged (the zeroed buffers it writes are what the allocation holds).
        // A bounded host facade may never enter the generic f32 pair: its
        // fallback and post-GEMM host work are not governed by that authority.
        // The diagnostic kill-switch remains effective only on legacy routes.
        let skip_dead_f32 = want_recompute
            && (bounded_deadline_engine.is_some()
                || super::super::crown_helpers::conv_skip_dead_f32_enabled());
        let (mut new_lower_a, mut new_upper_a) = if skip_dead_f32 {
            poll_authority()?;
            // Same memory-cap refusal the pair enforces (CpuMemoryExceeded →
            // sound IBP fallback at the collector) — the skip must not turn a
            // capped refusal into an attempted allocation.
            super::ops_transpose_gemm::guard_conv_crown_backward_buffer(
                bounds.lower_a().nrows().max(bounds.upper_a().nrows()),
                conv_in_size,
                2,
            )?;
            let result = (
                Array2::<f32>::zeros((bounds.lower_a().nrows(), conv_in_size)),
                Array2::<f32>::zeros((bounds.upper_a().nrows(), conv_in_size)),
            );
            poll_authority()?;
            result
        } else {
            conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
                bounds.lower_a(),
                bounds.upper_a(),
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (in_h, in_w),
                (out_h, out_w),
                out_c,
                self.groups,
                engine,
                deadline,
            )?
        };

        let recompute_side = |a: &Array2<f32>| -> Result<Option<Array2<f64>>> {
            let recompute = if let Some(bounded_engine) = bounded_deadline_engine {
                conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
                    a,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // lower/upper f64 results coexist, always serial here
                    Some(bounded_engine),
                    deadline,
                )
            } else {
                conv2d_transpose_backward_coeff_f64_with_deadline(
                    a,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // lower/upper f64 results coexist, serially or via rayon::join
                    deadline,
                )
            };
            match recompute {
                Ok(coefficients) => Ok(Some(coefficients)),
                Err(error @ NyError::DeadlineExceeded(_)) => Err(error),
                Err(error) if bounded_deadline_engine.is_some() => Err(error),
                Err(_) => Ok(None),
            }
        };
        // Under the gate the two independent f64 recomputes run concurrently —
        // each is internally deterministic and the certified error channel is
        // summation-order independent, so the join is bit-safe. Gate-off keeps
        // the shipped serial order untouched.
        let (coeff_f64, coeff_f64_u) = if !want_recompute {
            (None, None)
        } else if skip_dead_f32 && bounded_deadline_engine.is_none() {
            let (lower, upper) = rayon::join(
                || recompute_side(bounds.lower_a()),
                || recompute_side(bounds.upper_a()),
            );
            (lower?, upper?)
        } else {
            (
                recompute_side(bounds.lower_a())?,
                recompute_side(bounds.upper_a())?,
            )
        };
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_lower_a.raw_dim());
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_upper_a.raw_dim());
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        // #recompute-failure-keeps-f32 WAS TRIED HERE AND WITHDRAWN (2026-07-29).
        //
        // The idea was: a failed f64 recompute need not vacate the relation, because
        // `conv_coeff_err_matrix` will certify a plain f32-GEMM coefficient with
        // `gamma = gamma_n^f32` and `cast_err = 0` when `coeff_f64` is absent. That
        // certificate is NOT sufficient, and adversarial review produced an
        // exact-rational counterexample:
        //
        //   * `gamma_n_f32` cites Higham Thm 3.1, which assumes NO UNDERFLOW. In the
        //     subnormal regime each rounding contributes an ABSOLUTE eta <= 2^-150
        //     that no relative `gamma*S` term covers. Constructed case (out_c=4,
        //     3x3 kernel, subnormal weights): certified `gamma*S = 1.69e-46` vs a
        //     measured true error of `4.25e-45` -- a 25x violation, i.e. the
        //     certified interval does NOT contain the true coefficient.
        //   * This repo's own f32-sound primitive already knows that.
        //     `crown_single.rs`'s `aw_f32_sound_bound` charges
        //     `gamma*s + ftz + daz` plus a containment margin; the conv branch at
        //     `crown_helpers.rs:556` charges only `gamma * row_in_max * kl1`.
        //   * `conv_should_f64_recompute` returns `true` unconditionally, so that
        //     branch is DEAD CODE on this path today. Enabling it would make an
        //     under-charged certificate live in production for the first time.
        //
        // Landing the objective-chunking half alone removes the memory refusal that
        // motivated this in the first place (measured: cap-trigger warnings 4x/6x/2x
        // -> 0x on cifar100 and traffic_signs), so the fallback buys nothing we need.
        // If it is ever revisited, it must first charge the `ftz`/`daz` terms and
        // must NOT forward `engine` (the pair only nulls it under a finite deadline,
        // so a DAZ/GPU backend could otherwise produce the coefficient).
        // Overwrite the f32-GEMM point coefficient with the directed-rounded f32
        // of the f64 recompute when available (tighter and matched to cast_err).
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                if finite_authority {
                    poll_authority()?;
                    for (index, (dst, &src)) in new_lower_a.iter_mut().zip(c64.iter()).enumerate() {
                        if index.is_multiple_of(DEADLINE_IBP_POLL_ELEMENTS) {
                            poll_authority()?;
                        }
                        *dst = src as f32;
                    }
                    poll_authority()?;
                } else {
                    for i in 0..new_lower_a.nrows() {
                        for p in 0..new_lower_a.ncols() {
                            new_lower_a[[i, p]] = c64[[i, p]] as f32;
                        }
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                if finite_authority {
                    poll_authority()?;
                    for (index, (dst, &src)) in new_upper_a.iter_mut().zip(c64.iter()).enumerate() {
                        if index.is_multiple_of(DEADLINE_IBP_POLL_ELEMENTS) {
                            poll_authority()?;
                        }
                        *dst = src as f32;
                    }
                    poll_authority()?;
                } else {
                    for i in 0..new_upper_a.nrows() {
                        for p in 0..new_upper_a.ncols() {
                            new_upper_a[[i, p]] = c64[[i, p]] as f32;
                        }
                    }
                }
            }
        }

        let (mut new_lower_b, mut new_upper_b) = if finite_authority {
            compute_conv_bias_f64_with_poll(
                bounds,
                self.bias.as_ref(),
                out_c,
                out_h * out_w,
                &mut poll_authority,
            )?
        } else {
            compute_conv_bias_f64(bounds, self.bias.as_ref(), out_c, out_h * out_w)?
        };

        // Certified coefficient error `cast + γ·S + prop` (shared helper). `S` is
        // over-bounded per row by `row_max(a,i)·‖kernel‖_1` (SOUND; `γ·S` is sub-ULP
        // at concretize) avoiding a second transpose-conv pass. γ is `γ_n^f64`
        // (with cast) on the recompute path and `γ_n^f32` (cast=0) on the small-n
        // fast path — both sound.
        //
        // #cgan-conv-err-compose: the INCOMING error, by contrast, is composed
        // EXACTLY through the same transpose-conv column transform with |kernel|
        // (`prop[i,p] = Σ_j err_in[i,j]·|K_{j→p}|`) instead of the row-constant
        // `row_max(err_in)·‖kernel‖_1` over-bound — see bound_transpose.rs for
        // the full rationale (the row bound amplified carried error by
        // ~‖kernel‖_1/column-L1 per conv layer, the dominant cGAN looseness).
        // Sound: exact first-order enclosure, f32 rounding of the non-negative
        // composition covered by the (1+γ) inflation in `conv_coeff_err_matrix`.
        let prop_pair = match (bounds.lower_a_err(), bounds.upper_a_err()) {
            (None, None) => None,
            (le, ue) => {
                let abs_kernel = if finite_authority {
                    poll_authority()?;
                    let mut absolute = ArrayD::<f32>::zeros(self.kernel.raw_dim());
                    for (index, (dst, &src)) in
                        absolute.iter_mut().zip(self.kernel.iter()).enumerate()
                    {
                        if index.is_multiple_of(DEADLINE_IBP_POLL_ELEMENTS) {
                            poll_authority()?;
                        }
                        *dst = src.abs();
                    }
                    poll_authority()?;
                    absolute
                } else {
                    self.kernel.mapv(f32::abs)
                };
                let zeros;
                let (el, eu) = match (le, ue) {
                    (Some(el), Some(eu)) => (el, eu),
                    (Some(el), None) => {
                        zeros = Array2::<f32>::zeros(el.raw_dim());
                        (el, &zeros)
                    }
                    (None, Some(eu)) => {
                        zeros = Array2::<f32>::zeros(eu.raw_dim());
                        (&zeros, eu)
                    }
                    (None, None) => unreachable!("outer match handles (None, None)"),
                };
                if bounded_deadline_engine.is_some() || deadline.is_some() {
                    // A finite verifier authority or bounded-executor authority
                    // must not enter the fused/generic f32 GEMM (or its
                    // unpollable faer fallback). Compose the non-negative
                    // incoming error through the same pollable f64
                    // transpose-conv used for certified coefficients, then cast
                    // upward to f32. `conv_coeff_err_matrix` additionally
                    // inflates this non-negative value by (1 + gamma_f32),
                    // covering both the much smaller f64 sum error and the cast.
                    let compose_side = |error: &Array2<f32>| -> Result<Option<Array2<f32>>> {
                        let recompute = if let Some(bounded_engine) = bounded_deadline_engine {
                            conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
                                error,
                                &abs_kernel,
                                self.stride,
                                self.padding,
                                self.dilation,
                                (in_h, in_w),
                                (out_h, out_w),
                                out_c,
                                self.groups,
                                2,
                                Some(bounded_engine),
                                deadline,
                            )
                        } else {
                            conv2d_transpose_backward_coeff_f64_with_deadline(
                                error,
                                &abs_kernel,
                                self.stride,
                                self.padding,
                                self.dilation,
                                (in_h, in_w),
                                (out_h, out_w),
                                out_c,
                                self.groups,
                                2,
                                deadline,
                            )
                        };
                        let composed = match recompute {
                            Ok(composed) => composed,
                            Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                            Err(error) if bounded_deadline_engine.is_some() => return Err(error),
                            Err(_) => return Ok(None),
                        };
                        if let Some(engine) = bounded_deadline_engine {
                            engine.poll_crown_backward_deadline()?;
                        }
                        if deadline.is_some_and(|limit| Instant::now() >= limit) {
                            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
                        }
                        let mut outward = Array2::<f32>::zeros(composed.raw_dim());
                        for (index, (dst, &value)) in
                            outward.iter_mut().zip(composed.iter()).enumerate()
                        {
                            if index.is_multiple_of(4_096) {
                                if let Some(engine) = bounded_deadline_engine {
                                    engine.poll_crown_backward_deadline()?;
                                }
                                if deadline.is_some_and(|limit| Instant::now() >= limit) {
                                    return Err(
                                        super::ops_transpose_gemm::per_node_deadline_exceeded(),
                                    );
                                }
                            }
                            *dst = if value.is_finite() {
                                next_up_f32(value.max(0.0) as f32)
                            } else {
                                f32::INFINITY
                            };
                        }
                        if let Some(engine) = bounded_deadline_engine {
                            engine.poll_crown_backward_deadline()?;
                        }
                        if deadline.is_some_and(|limit| Instant::now() >= limit) {
                            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
                        }
                        Ok(Some(outward))
                    };
                    match (compose_side(el)?, compose_side(eu)?) {
                        (Some(lower), Some(upper)) => Some((lower, upper)),
                        _ => None,
                    }
                } else {
                    // Preserve the historical fused/GEMM behavior exactly when
                    // neither a verifier deadline nor bounded-executor
                    // authority exists.
                    conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
                        el,
                        eu,
                        &abs_kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                        (in_h, in_w),
                        (out_h, out_w),
                        out_c,
                        self.groups,
                        engine,
                        None,
                    )
                    .ok()
                }
            }
        };
        let (prop_l, prop_u) = match &prop_pair {
            Some((l, u)) => (
                bounds.lower_a_err().is_some().then_some(l),
                bounds.upper_a_err().is_some().then_some(u),
            ),
            None => (None, None),
        };
        let kernel_l1: f64 = if finite_authority {
            poll_authority()?;
            let mut total = 0.0;
            for (index, &value) in self.kernel.iter().enumerate() {
                if index.is_multiple_of(DEADLINE_IBP_POLL_ELEMENTS) {
                    poll_authority()?;
                }
                total += (value as f64).abs();
            }
            poll_authority()?;
            total
        } else {
            self.kernel.iter().map(|&v| (v as f64).abs()).sum()
        };
        // #patches-perchannel-l1: column `p` of the output block is one
        // (in_channel, y, x), and only that channel's kernel slice (within its
        // group) can reach it — so charging the whole-kernel L1 over-counts by
        // ~in_c_per_group. Build the per-column norm; `None` on any shape we do
        // not recognise keeps the scalar norm.
        let kernel_l1_cols: Option<Vec<f64>> = {
            let ks = self.kernel.shape();
            let ncols = new_lower_a.ncols();
            if ks.len() == 4 && self.groups > 0 && ks[0].is_multiple_of(self.groups) {
                let (out_c, in_c_per_group, kh_k, kw_k) = (ks[0], ks[1], ks[2], ks[3]);
                let in_c_total = in_c_per_group * self.groups;
                let out_c_per_group = out_c / self.groups;
                if in_c_total > 0 && ncols.is_multiple_of(in_c_total) {
                    let spatial = ncols / in_c_total;
                    let mut per_ic = vec![0.0f64; in_c_total];
                    let mut kernel_work = 0usize;
                    for g in 0..self.groups {
                        for ic_local in 0..in_c_per_group {
                            let mut acc = 0.0f64;
                            for oc_local in 0..out_c_per_group {
                                let oc = g * out_c_per_group + oc_local;
                                for ki in 0..kh_k {
                                    for kj in 0..kw_k {
                                        acc += (self.kernel[[oc, ic_local, ki, kj]] as f64).abs();
                                        if finite_authority {
                                            kernel_work += 1;
                                            if kernel_work >= DEADLINE_IBP_POLL_ELEMENTS {
                                                kernel_work = 0;
                                                poll_authority()?;
                                            }
                                        }
                                    }
                                }
                            }
                            per_ic[g * in_c_per_group + ic_local] = acc;
                        }
                    }
                    if finite_authority {
                        poll_authority()?;
                        let mut per_column = Vec::with_capacity(ncols);
                        for p in 0..ncols {
                            if p.is_multiple_of(DEADLINE_IBP_POLL_ELEMENTS) {
                                poll_authority()?;
                            }
                            per_column.push(per_ic[p / spatial]);
                        }
                        poll_authority()?;
                        Some(per_column)
                    } else {
                        Some((0..ncols).map(|p| per_ic[p / spatial]).collect())
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };
        let kernel_l1_cols_ref = kernel_l1_cols.as_deref();
        let mut lower_err = if finite_authority {
            super::super::crown_helpers::conv_coeff_err_matrix_with_poll(
                bounds.lower_a(),
                bounds.lower_a_err(),
                &new_lower_a,
                coeff_f64.as_ref().filter(|_| lower_recompute_ok),
                kernel_l1,
                n_contraction,
                prop_l,
                kernel_l1_cols_ref,
                &mut poll_authority,
            )?
        } else {
            super::super::crown_helpers::conv_coeff_err_matrix(
                bounds.lower_a(),
                bounds.lower_a_err(),
                &new_lower_a,
                coeff_f64.as_ref().filter(|_| lower_recompute_ok),
                kernel_l1,
                n_contraction,
                prop_l,
                kernel_l1_cols_ref,
            )
        };
        let mut upper_err = if finite_authority {
            super::super::crown_helpers::conv_coeff_err_matrix_with_poll(
                bounds.upper_a(),
                bounds.upper_a_err(),
                &new_upper_a,
                coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
                kernel_l1,
                n_contraction,
                prop_u,
                kernel_l1_cols_ref,
                &mut poll_authority,
            )?
        } else {
            super::super::crown_helpers::conv_coeff_err_matrix(
                bounds.upper_a(),
                bounds.upper_a_err(),
                &new_upper_a,
                coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
                kernel_l1,
                n_contraction,
                prop_u,
                kernel_l1_cols_ref,
            )
        };
        // A WANTED-but-failed recompute degrades the row to ±inf bias (the f32
        // coefficient cannot be soundly certified with the f64 gamma in that case).
        //
        // SAY SO. `recompute_side` maps every non-deadline error (including the
        // memory-cap `CpuMemoryExceeded` refusals, which log only the refusal
        // itself) to `Ok(None)`, so this degrade — which vacates the ENTIRE
        // relation and makes the target concretize to `[-inf, inf]` — was
        // previously invisible in the log. On yolo_2023 that hid the Conv_12
        // `[-inf, inf]` behind what looked like a clean, NaN-free backward.
        let nrows = new_lower_a.nrows();
        if lower_recompute_failed || upper_recompute_failed {
            warn!(
                "Conv2d CROWN backward: the f64 coefficient recompute FAILED \
                 (lower={lower_recompute_failed}, upper={upper_recompute_failed}) for a \
                 [{nrows} x {}] relation; every affected row is degraded to ±inf bias, so this \
                 target concretizes to [-inf, inf] and its CROWN pass buys nothing. Look for a \
                 Conv2d CROWN backward memory-cap refusal just above -- if the shape is \
                 objective-scaled, #f64-recompute-objective-chunk should have rescued it, so a \
                 refusal here means even ONE objective exceeded the envelope cap.",
                new_lower_a.ncols()
            );
        }
        if lower_recompute_failed {
            let mut work = 0usize;
            for i in 0..nrows {
                for p in 0..new_lower_a.ncols() {
                    new_lower_a[[i, p]] = 0.0;
                    lower_err[[i, p]] = 0.0;
                    if finite_authority {
                        work += 1;
                        if work >= DEADLINE_IBP_POLL_ELEMENTS {
                            work = 0;
                            poll_authority()?;
                        }
                    }
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
            if finite_authority {
                poll_authority()?;
            }
        }
        if upper_recompute_failed {
            let mut work = 0usize;
            for i in 0..nrows {
                for p in 0..new_upper_a.ncols() {
                    new_upper_a[[i, p]] = 0.0;
                    upper_err[[i, p]] = 0.0;
                    if finite_authority {
                        work += 1;
                        if work >= DEADLINE_IBP_POLL_ELEMENTS {
                            work = 0;
                            poll_authority()?;
                        }
                    }
                }
                new_upper_b[i] = f32::INFINITY;
            }
            if finite_authority {
                poll_authority()?;
            }
        }

        if finite_authority {
            detect_and_fix_nonfinite_rows_with_poll(
                &mut new_lower_a,
                &mut new_upper_a,
                &mut new_lower_b,
                &mut new_upper_b,
                conv_in_size,
                "Conv2d",
                &mut poll_authority,
            )?;
        } else {
            detect_and_fix_nonfinite_rows(
                &mut new_lower_a,
                &mut new_upper_a,
                &mut new_lower_b,
                &mut new_upper_b,
                conv_in_size,
                "Conv2d",
            );
        }
        // Zero error on any row that detect_and_fix degraded to ±inf bias (the
        // row is already maximally loose); also covers shape consistency.
        let mut zero_work = 0usize;
        for i in 0..new_lower_a.nrows() {
            if !new_lower_b[i].is_finite() {
                for p in 0..lower_err.ncols() {
                    lower_err[[i, p]] = 0.0;
                    if finite_authority {
                        zero_work += 1;
                        if zero_work >= DEADLINE_IBP_POLL_ELEMENTS {
                            zero_work = 0;
                            poll_authority()?;
                        }
                    }
                }
            }
            if !new_upper_b[i].is_finite() {
                for p in 0..upper_err.ncols() {
                    upper_err[[i, p]] = 0.0;
                    if finite_authority {
                        zero_work += 1;
                        if zero_work >= DEADLINE_IBP_POLL_ELEMENTS {
                            zero_work = 0;
                            poll_authority()?;
                        }
                    }
                }
            }
        }

        poll_authority()?;
        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        // SOUND transpose-conv coefficient interval carried via the error matrices.
        let published = if finite_authority {
            // Every coefficient row has just passed the pollable
            // `detect_and_fix_nonfinite_rows_with_poll` firewall and every
            // certificate entry was produced by the outward publisher above.
            // Construct directly so a finite authority does not disappear into
            // the ordinary constructor's duplicate, unpollable full-array scans
            // and error-map copies.
            LinearBounds {
                lower_a: new_lower_a,
                lower_b: new_lower_b,
                upper_a: new_upper_a,
                upper_b: new_upper_b,
                lower_a_err: Some(lower_err),
                upper_a_err: Some(upper_err),
            }
        } else {
            LinearBounds::new_or_conservative_with_err(
                new_lower_a,
                new_lower_b,
                new_upper_a,
                new_upper_b,
                lower_err,
                upper_err,
            )?
        };
        poll_authority()?;
        Ok(Cow::Owned(published))
    }
}
