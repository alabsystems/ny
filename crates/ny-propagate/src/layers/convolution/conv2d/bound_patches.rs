// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use rayon::prelude::*;

use super::ops::{conv2d_transpose_grouped_batched_fast, conv2d_transpose_grouped_into};
use super::{conv2d_transpose_grouped, Conv2dLayer};
use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;

/// Route the patches-mode Conv2d transpose-conv composition through a single
/// engine GEMM (GPU-capable) instead of the per-position CPU scatter — the
/// keystone for getting the conv-CROWN warmup onto the GPU.
///
/// DEFAULT: on whenever a fast-f32 GEMM accelerator (cuBLAS/CUDA) is installed,
/// off otherwise. The GPU compose carries the same certified outward-rounded
/// coefficient error as the CPU scatter (verified by the patches +
/// `crown_linear_aw` soundness oracles under this gate), so its bounds are
/// sound enclosures — enabling it can only shift a borderline verdict between
/// `verified` and `unknown`, never toward an unsound `unsat`. It is NOT
/// byte-identical to the CPU scatter (GPU GEMM accumulation order differs), so
/// CPU-only builds keep the byte-identical scatter as their default.
/// `NY_PATCHES_GPU=1` forces it on (e.g. a CPU build with a wgpu engine);
/// `NY_PATCHES_GPU=0` forces the byte-identical CPU scatter even on a CUDA box.
fn patches_gpu_enabled() -> bool {
    match std::env::var("NY_PATCHES_GPU") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => crate::fast_f32_gemm::is_installed(),
    }
}

/// #conv-patches-collect: default-OFF gate for the EXACT padded-conv patches
/// composition (intermediate-tap masking). When set, `propagate_patches_engine`
/// masks the out-of-range intermediate taps of a non-identity incoming patch so
/// composing THROUGH a padded conv stays in the memory-light patches
/// representation instead of falling back to the OOM-prone dense CROWN path.
/// Env-UNSET is byte-identical: the guard below returns `UnsupportedConfiguration`
/// exactly as before, so the caller takes the pre-existing dense fallback.
fn conv_patches_padded_compose_enabled() -> bool {
    std::env::var_os("NY_CONV_PATCHES_COLLECT").is_some_and(|v| v != "0" && !v.is_empty())
}

/// Zero the taps of a **6D dense non-identity** patch that reference OUT-OF-RANGE
/// intermediate positions, in place, before a transpose-compose step
/// (#conv-patches-collect).
///
/// The incoming patch maps its input space Y (this composition's intermediate,
/// == the downstream conv's input = this conv's output) to the spec. A tap at
/// spec position `(soh, sow)`, kernel offset `(ki, kj)` references
/// `yh = soh·prev_sh − prev_pt + ki`, `yw = sow·prev_sw − prev_pl + kj`. When
/// that `(yh, yw)` lies outside `[0, y_h) × [0, y_w)` it addresses the
/// zero-padding the downstream conv added around Y — a HARD zero whose true
/// contribution is 0. The dense operator drops it (`to_dense`'s unfold clips
/// out-of-bounds), but the transpose-compose would otherwise smear it through
/// this conv's kernel onto REAL input cells (the boundary leak the guard below
/// rejects). Zeroing those taps up front makes the compose EXACT — verified
/// bit-close to the dense CROWN backward by
/// `patches_padded_compose_matches_dense` and the padded proptest.
///
/// The row/column in-range predicates are separable, so this is O(spec · out_c ·
/// prev_kh · prev_kw) with tiny precomputed masks. No-op unless the patch is 6D,
/// dense (no `unstable_idx`), non-identity, and carries nonzero padding.
fn mask_out_of_range_intermediate_taps(pd: &mut PatchesData) {
    use rayon::prelude::*;
    if pd.identity || pd.unstable_idx.is_some() || pd.padding == (0, 0, 0, 0) {
        return;
    }
    let Some(patches) = pd.patches.as_mut() else {
        return;
    };
    let shape = patches.shape().to_vec();
    if shape.len() != 6 {
        return;
    }
    let (_spec_oc, spec_oh, spec_ow) = (shape[0], shape[1], shape[2]);
    let (out_c, prev_kh, prev_kw) = (shape[3], shape[4], shape[5]);
    let block = out_c * prev_kh * prev_kw;
    if block == 0 || spec_oh == 0 || spec_ow == 0 {
        return;
    }
    let (prev_sh, prev_sw) = pd.stride;
    let (prev_pl, _prev_pr, prev_pt, _prev_pb) = pd.padding;
    let (_y_c, y_h, y_w) = pd.input_shape;

    // Separable in-range predicates: yh depends only on (soh, ki); yw on (sow, kj).
    let mut row_ok = vec![false; spec_oh * prev_kh];
    for soh in 0..spec_oh {
        for ki in 0..prev_kh {
            let yh = soh as isize * prev_sh as isize - prev_pt as isize + ki as isize;
            row_ok[soh * prev_kh + ki] = yh >= 0 && yh < y_h as isize;
        }
    }
    let mut col_ok = vec![false; spec_ow * prev_kw];
    for sow in 0..spec_ow {
        for kj in 0..prev_kw {
            let yw = sow as isize * prev_sw as isize - prev_pl as isize + kj as isize;
            col_ok[sow * prev_kw + kj] = yw >= 0 && yw < y_w as isize;
        }
    }

    // A spec position owns the contiguous [c, ki, kj] block; positions are
    // row-major over (soc, soh, sow), so `pos` decodes soh/sow directly.
    if let Some(flat) = patches.as_slice_mut() {
        flat.par_chunks_mut(block)
            .enumerate()
            .for_each(|(pos, chunk)| {
                let sow = pos % spec_ow;
                let soh = (pos / spec_ow) % spec_oh;
                for c in 0..out_c {
                    for ki in 0..prev_kh {
                        let rok = row_ok[soh * prev_kh + ki];
                        for kj in 0..prev_kw {
                            if !(rok && col_ok[sow * prev_kw + kj]) {
                                chunk[(c * prev_kh + ki) * prev_kw + kj] = 0.0;
                            }
                        }
                    }
                }
            });
    }
}

impl PatchesPropagation for Conv2dLayer {
    /// CROWN backward with Patches coefficients for Conv2d.
    ///
    /// Supports both identity incoming patches (first Conv2d in backward chain)
    /// and non-identity patches (chained Conv2d with composition).
    ///
    /// For identity patches: creates initial patches from the conv kernel.
    /// For non-identity patches: composes by applying conv2d_transpose to each
    /// patch, producing a larger receptive field with composed stride/padding.
    ///
    /// Composition math (reference: alpha-beta-CROWN auto_LiRPA/patches.py):
    /// - new_kh = (prev_kh - 1) * stride_h + kh
    /// - new_stride = prev_stride * conv_stride
    /// - new_padding = prev_padding * conv_stride + conv_padding
    ///
    /// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    fn propagate_patches(&self, bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        self.propagate_patches_engine(bounds, None)
    }
}

impl Conv2dLayer {
    /// Engine-aware patches Conv2d CROWN backward. With `engine` present AND
    /// `NY_PATCHES_GPU` set, the per-position transpose-conv composition runs as
    /// a single GEMM (GPU-capable) instead of the rayon per-position CPU scatter
    /// — the keystone for getting the conv-CROWN warmup onto the GPU while
    /// staying in the memory-light patches representation. With `engine` None or
    /// the flag unset the result is byte-identical to the CPU path.
    pub(crate) fn propagate_patches_engine(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<CrownBounds> {
        // Guard: reject NaN weights
        if self.kernel.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "Conv2d Patches backward: kernel contains NaN".into(),
            ));
        }

        // The Patches composition math (new_kh = (prev_kh-1)*stride + kh) does not
        // yet account for dilation. Reject dilated convolutions here so the caller
        // falls back to the dilation-aware dense CROWN path; never silently
        // produce wrong bounds.
        if self.dilation != (1, 1) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Conv2d Patches CROWN does not support dilation {:?}; use dense CROWN",
                self.dilation
            )));
        }

        // SOUNDNESS GUARD (#hotpath): composing a non-identity patch through this
        // Conv2d is only equivalent to the dense operator when the INCOMING patches
        // carry zero padding. When the already-composed patches have nonzero
        // padding, the boundary-truncation of the intermediate conv (its kernel is
        // clipped against the input edge) is not reconstructible from the composed
        // `conv2d_transpose` (which runs with padding=0) plus the additive padding
        // metadata: edge output positions get coefficients that disagree with dense
        // (verified: interior rows match, boundary rows diverge). Single-layer
        // padding is sound because `to_dense`'s unfold clips correctly; the issue
        // is ONLY padding accumulated ACROSS a composition step. Reject here so the
        // caller falls back to the exact dense CROWN path. Identity incoming patches
        // and zero-padding chains (the common stride-1/pad-0 and pad-after-pad-0
        // cases) remain in patches mode.
        // #conv-patches-collect: the leak is EXACTLY the out-of-range intermediate
        // taps (the downstream conv's zero-padding around this conv's output). When
        // the padded-compose feature is enabled AND both incoming sides are 6D
        // dense maskable patches, `mask_out_of_range_intermediate_taps` zeros those
        // taps up front, which makes the transpose-compose bit-equivalent to the
        // dense operator (parity-tested) — so the composition stays SOUND and in
        // patches. Otherwise (feature off, or a 7D explicit-rows / sparse incoming
        // the mask does not cover) keep the exact pre-existing dense fallback.
        let nonzero_incoming_padding = (!bounds.lower_a.identity
            && bounds.lower_a.padding != (0, 0, 0, 0))
            || (!bounds.upper_a.identity && bounds.upper_a.padding != (0, 0, 0, 0));
        let masked_bounds_storage;
        let bounds: &PatchesLinearBounds = if nonzero_incoming_padding {
            let side_maskable = |pd: &PatchesData| -> bool {
                pd.identity
                    || (pd.unstable_idx.is_none()
                        && pd.patches.as_ref().map(|p| p.ndim() == 6).unwrap_or(false))
            };
            let can_mask = conv_patches_padded_compose_enabled()
                && side_maskable(&bounds.lower_a)
                && side_maskable(&bounds.upper_a);
            if !can_mask {
                return Err(NyError::UnsupportedConfiguration(
                    "Conv2d Patches CROWN cannot soundly compose through incoming patches \
                     with nonzero padding; use dense CROWN"
                        .to_string(),
                ));
            }
            if std::env::var_os("NY_CONV_PATCHES_DEBUG").is_some_and(|v| v != "0" && !v.is_empty())
            {
                eprintln!(
                    "[conv-patches-dbg] MASK compose: in_c={} out_c={} incoming_pad={:?} spec_rows={}",
                    self.in_channels(),
                    self.out_channels(),
                    bounds.lower_a.padding,
                    bounds.row_count,
                );
            }
            let mut masked = bounds.clone();
            mask_out_of_range_intermediate_taps(&mut masked.lower_a);
            mask_out_of_range_intermediate_taps(&mut masked.upper_a);
            masked_bounds_storage = masked;
            &masked_bounds_storage
        } else {
            bounds
        };

        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Conv2d Patches CROWN requires input_shape".to_string(),
            )
        })?;

        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let params = Conv2dPatchesParams {
            kernel: &self.kernel,
            in_c,
            out_c,
            groups: self.groups,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            in_h,
            in_w,
            out_h,
            out_w,
        };

        let lower_result = Self::conv2d_patches_backward(&bounds.lower_a, &params, engine)?;
        let upper_result = Self::conv2d_patches_backward(&bounds.upper_a, &params, engine)?;

        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            Self::compute_patches_bias(bounds, bias, out_c, out_h, out_w)?
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };

        if lower_result.should_fallback_to_dense() || upper_result.should_fallback_to_dense() {
            let plb = PatchesLinearBounds {
                row_count: bounds.row_count,
                lower_a: lower_result,
                lower_b: new_lower_b,
                upper_a: upper_result,
                upper_b: new_upper_b,
            };
            return Ok(CrownBounds::Dense(plb.to_dense()?));
        }

        Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: lower_result,
            lower_b: new_lower_b,
            upper_a: upper_result,
            upper_b: new_upper_b,
        })))
    }
}

/// Convolution parameters for Patches backward propagation.
struct Conv2dPatchesParams<'a> {
    kernel: &'a ArrayD<f32>,
    in_c: usize,
    out_c: usize,
    groups: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
}

impl Conv2dLayer {
    /// Compute patches backward for a single PatchesData (lower_a or upper_a).
    ///
    /// For identity patches: creates initial patches from the conv kernel.
    /// For existing patches: composes by applying conv2d_transpose to each
    /// patch position, producing a larger receptive field.
    ///
    /// Composition math (reference: alpha-beta-CROWN auto_LiRPA/patches.py):
    /// - Each patch of shape (prev_in_c, prev_kh, prev_kw) is convolved backward
    ///   through this Conv2d via conv2d_transpose(patch, kernel, stride, padding=0)
    /// - Output patch shape: (in_c, new_kh, new_kw) where
    ///   new_kh = (prev_kh - 1) * stride_h + kh
    /// - Composed stride: prev_stride * conv_stride
    /// - Composed padding: prev_padding * conv_stride + conv_padding
    fn conv2d_patches_backward(
        patches_data: &PatchesData,
        p: &Conv2dPatchesParams<'_>,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<PatchesData> {
        if patches_data.identity {
            // First Conv2d in backward chain: create initial patches from kernel.
            // Patches shape: (spec_oc, spec_oh, spec_ow, in_c, kH, kW)
            // For identity, spec output shape = this conv's output shape.
            //
            // Crash guard (#hotpath robustness): the build loop below derives
            // `out_c_per_group = out_c / groups` and `ic_start = group_idx *
            // in_c_per_group` and then indexes `kernel[[oc, ic_local, ..]]` and
            // `patches[[.., ic, ..]]` without rechecking that the grouped layout is
            // self-consistent. If the layer's `groups` metadata is inconsistent with
            // its channel counts (`groups == 0`, `out_c` not divisible by `groups`,
            // or `in_c_per_group * groups != in_c`), those indices run out of bounds
            // and panic. Reject such a malformed convolution with a clean
            // ShapeMismatch before the loop. No bound math changes.
            let in_c_per_group = p.kernel.shape()[1];
            if p.groups == 0
                || !p.out_c.is_multiple_of(p.groups)
                || in_c_per_group.checked_mul(p.groups) != Some(p.in_c)
            {
                return Err(NyError::ShapeMismatch {
                    expected: vec![p.in_c, p.out_c, p.groups],
                    got: vec![in_c_per_group.saturating_mul(p.groups), p.out_c, p.groups],
                });
            }
            let mut patches =
                ArrayD::<f32>::zeros(IxDyn(&[p.out_c, p.out_h, p.out_w, p.in_c, p.kh, p.kw]));
            let out_c_per_group = p.out_c / p.groups;
            for oc in 0..p.out_c {
                let group_idx = oc / out_c_per_group;
                let ic_start = group_idx * in_c_per_group;
                for oh in 0..p.out_h {
                    for ow in 0..p.out_w {
                        for ic_local in 0..in_c_per_group {
                            let ic = ic_start + ic_local;
                            for ki in 0..p.kh {
                                for kj in 0..p.kw {
                                    patches[[oc, oh, ow, ic, ki, kj]] =
                                        p.kernel[[oc, ic_local, ki, kj]];
                                }
                            }
                        }
                    }
                }
            }
            Ok(PatchesData {
                coeff_err: None,
                patches: Some(patches),
                stride: (p.sh, p.sw),
                padding: (p.pw, p.pw, p.ph, p.ph),
                identity: false,
                output_shape: (p.out_c, p.out_h, p.out_w),
                input_shape: (p.in_c, p.in_h, p.in_w),
                unstable_idx: None,
            })
        } else {
            // Non-identity: compose existing patches through this Conv2d.
            // Each patch represents coefficients in this Conv2d's OUTPUT space.
            // Apply conv2d_transpose to map them to this Conv2d's INPUT space.
            let incoming = patches_data.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;
            let shape = incoming.shape();
            let (spec_oc, spec_oh, spec_ow) = patches_data.output_shape;
            let explicit_rows = match shape.len() {
                6 => false,
                7 => true,
                _ => {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![6, 7],
                        got: vec![shape.len()],
                    });
                }
            };
            let row_count = if explicit_rows { shape[0] } else { 1 };
            let (channel_axis, kh_axis, kw_axis) =
                if explicit_rows { (4, 5, 6) } else { (3, 4, 5) };
            let prev_kh = shape[kh_axis];
            let prev_kw = shape[kw_axis];

            if shape[channel_axis] != p.out_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![p.out_c],
                    got: vec![shape[channel_axis]],
                });
            }

            let new_kh = (prev_kh - 1) * p.sh + p.kh;
            let new_kw = (prev_kw - 1) * p.sw + p.kw;

            let (prev_sh, prev_sw) = patches_data.stride;
            let new_stride = (prev_sh * p.sh, prev_sw * p.sw);

            let (prev_pl, prev_pr, prev_pt, prev_pb) = patches_data.padding;
            let new_padding = (
                prev_pl * p.sw + p.pw,
                prev_pr * p.sw + p.pw,
                prev_pt * p.sh + p.ph,
                prev_pb * p.sh + p.ph,
            );

            let num_positions = row_count * spec_oc * spec_oh * spec_ow;
            let patch_volume = p.in_c * new_kh * new_kw;
            let decode = |idx: usize| {
                let row = idx / (spec_oc * spec_oh * spec_ow);
                let position_idx = idx % (spec_oc * spec_oh * spec_ow);
                let soc = position_idx / (spec_oh * spec_ow);
                let rem = position_idx % (spec_oh * spec_ow);
                (row, soc, rem / spec_ow, rem % spec_ow)
            };

            // Seam gate (#patches-coeff-err-soundness): route the per-position
            // transpose-conv composition to ONE GEMM. Now enabled for BOTH the
            // 6D dense and 7D explicit-rows layouts — the 7D closure
            // (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5, §14 C5) made 7D carry a
            // certified coeff_err channel, and the γ_K bound is summation-order
            // independent, so it certifies the engine GEMM's arbitrary
            // accumulation order for either layout. The certified err is computed
            // from the INCOMING coefficients (below), never from this GEMM's
            // output, so the seam only substitutes a value-equivalent (up to the
            // certified rounding) faster compose. Runs when `NY_PATCHES_GPU` is
            // set AND either a GEMM engine is threaded here OR the process-global
            // fast f32 accelerator (cuBLAS) is installed — the latter reaches the
            // CPU-routed scored workload, whose `engine` is `None`.
            let seam_enabled =
                patches_gpu_enabled() && (engine.is_some() || crate::fast_f32_gemm::is_installed());
            let engine_batched: Option<Vec<f32>> = if seam_enabled {
                // Build the [num_positions × out_c·prev_kh·prev_kw] input matrix in
                // the same (oc, ki, kj) flattening the operator matrix expects.
                // Each position owns the disjoint row [idx·k_dim, (idx+1)·k_dim),
                // so the gather parallelizes per position with per-element values
                // unchanged. k_dim == 0 gathers nothing (par_chunks_mut requires a
                // nonzero chunk length).
                let k_dim = p.out_c * prev_kh * prev_kw;
                let mut pmat = vec![0.0f32; num_positions * k_dim];
                if k_dim > 0 {
                    pmat.par_chunks_mut(k_dim)
                        .enumerate()
                        .for_each(|(idx, prow)| {
                            let (row, soc, soh, sow) = decode(idx);
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        prow[(c * prev_kh + ki) * prev_kw + kj] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }
                        });
                }
                // Route large products to cuBLAS (unified memory), else the passed
                // engine; `None` ⇒ below the MACs gate with no passed engine ⇒
                // fall through to the per-position CPU scatter.
                match conv2d_transpose_grouped_batched_fast(
                    &pmat,
                    num_positions,
                    p.kernel,
                    (p.sh, p.sw),
                    (prev_kh, prev_kw),
                    (new_kh, new_kw),
                    p.in_c,
                    p.groups,
                    engine,
                ) {
                    Some(batched) => {
                        let batched = batched?;
                        debug_assert_eq!(batched.len(), num_positions * patch_volume);
                        Some(batched)
                    }
                    None => None,
                }
            } else {
                None
            };

            let mut new_patches = if explicit_rows {
                ArrayD::<f32>::zeros(IxDyn(&[
                    row_count, spec_oc, spec_oh, spec_ow, p.in_c, new_kh, new_kw,
                ]))
            } else {
                ArrayD::<f32>::zeros(IxDyn(&[spec_oc, spec_oh, spec_ow, p.in_c, new_kh, new_kw]))
            };

            // Position idx (row-major over (row, soc, soh, sow)) owns exactly the
            // flat range [idx·patch_volume, (idx+1)·patch_volume) of the output in
            // both the 6D and 7D layouts, so positions fill disjoint chunks in
            // parallel with a per-thread reused gather buffer instead of a fresh
            // ArrayD + Vec per position. Per-position math is unchanged; only
            // allocation and scheduling differ. patch_volume == 0 falls through
            // (par_chunks_mut requires a nonzero chunk length).
            let filled_direct = if patch_volume == 0 {
                false
            } else if let Some(flat_out) = new_patches.as_slice_mut() {
                if let Some(ref batched) = engine_batched {
                    debug_assert_eq!(batched.len(), flat_out.len());
                    flat_out.copy_from_slice(batched);
                } else {
                    // Per-position scatter into the caller-owned output `chunk` (already
                    // `patch_volume` long, disjoint per position). `patch_3d` is a reused
                    // gather buffer — every (c, ki, kj) is overwritten, so no re-zero
                    // needed. `_into` re-zeros `chunk`, so the result is byte-identical
                    // to the owned form. Positions are DISJOINT, so serial and parallel
                    // give bit-identical output.
                    let make_buf = || ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
                    let scatter =
                        |patch_3d: &mut ArrayD<f32>, idx: usize, chunk: &mut [f32]| -> Result<()> {
                            let (row, soc, soh, sow) = decode(idx);
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        patch_3d[[c, ki, kj]] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }
                            conv2d_transpose_grouped_into(
                                chunk,
                                patch_3d,
                                p.kernel,
                                (p.sh, p.sw),
                                (0, 0),
                                (1, 1),
                                (new_kh, new_kw),
                                p.groups,
                            )?;
                            Ok(())
                        };
                    // Inside a parallel IMB region loop, run the scatter SERIALLY — this
                    // backward is already one region-worker's single-core slice, so a
                    // nested `par_chunks_mut` would fan out on the region pool and starve
                    // the N-way region parallelism (`crate::imb::region_seq_inner`).
                    if crate::imb::region_seq_inner() {
                        let mut buf = make_buf();
                        for (idx, chunk) in flat_out.chunks_mut(patch_volume).enumerate() {
                            scatter(&mut buf, idx, chunk)?;
                        }
                    } else {
                        flat_out
                            .par_chunks_mut(patch_volume)
                            .enumerate()
                            .try_for_each_init(make_buf, |patch_3d, (idx, chunk)| {
                                scatter(patch_3d, idx, chunk)
                            })?;
                    }
                }
                true
            } else {
                false
            };

            if !filled_direct {
                // Non-standard output layout or zero-volume patches (unreachable for
                // the freshly zeroed tensor above; kept so degenerate shapes retain
                // the original behavior): per-position collect + serial writeback.
                let composed_patches: Vec<Vec<f32>> = match engine_batched {
                    Some(batched) => batched.chunks(patch_volume).map(<[f32]>::to_vec).collect(),
                    None => (0..num_positions)
                        .into_par_iter()
                        .map(|idx| {
                            let (row, soc, soh, sow) = decode(idx);

                            let mut patch_3d =
                                ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        patch_3d[[c, ki, kj]] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }

                            let composed = conv2d_transpose_grouped(
                                &patch_3d,
                                p.kernel,
                                (p.sh, p.sw),
                                (0, 0),
                                (1, 1),
                                (new_kh, new_kw),
                                p.groups,
                            )?;

                            let flat: Vec<f32> = composed.iter().copied().collect();
                            debug_assert_eq!(flat.len(), patch_volume);
                            Ok(flat)
                        })
                        .collect::<Result<Vec<_>>>()?,
                };
                for (idx, flat) in composed_patches.iter().enumerate() {
                    let row = idx / (spec_oc * spec_oh * spec_ow);
                    let position_idx = idx % (spec_oc * spec_oh * spec_ow);
                    let soc = position_idx / (spec_oh * spec_ow);
                    let rem = position_idx % (spec_oh * spec_ow);
                    let soh = rem / spec_ow;
                    let sow = rem % spec_ow;
                    let mut fi = 0;
                    for ic in 0..p.in_c {
                        for ni in 0..new_kh {
                            for nj in 0..new_kw {
                                if explicit_rows {
                                    new_patches[[row, soc, soh, sow, ic, ni, nj]] = flat[fi];
                                } else {
                                    new_patches[[soc, soh, sow, ic, ni, nj]] = flat[fi];
                                }
                                fi += 1;
                            }
                        }
                    }
                }
            }

            // Certified coefficient error (#patches-coeff-err-soundness,
            // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5). The conv-transpose
            // composition is position-preserving (output logical row `i` composes
            // incoming row `i` only), so with K = out_c·prev_kh·prev_kw
            // (over-bounds the per-cell contraction) and gain ‖kernel‖₁ = Σ|kernel|:
            //   6D (logical row = output position):
            //     new_err[i] = next_up(γ_K^f32·RowMaxAbs(incoming@i)·‖k‖₁ + ‖k‖₁·old_err[i]).
            //   7D explicit rows (err index = SPEC row = axis 0, length row_count,
            //   spec I1): one scalar must cover every coefficient of the row, so
            //   the magnitude max runs over the WHOLE spec-row slab (all
            //   positions — max-lift; old_err[row] is row-constant so the lift is
            //   exact):
            //     new_err[row] = next_up(γ_K^f32·RowMaxAbs7D(incoming@row)·‖k‖₁
            //                            + ‖k‖₁·old_err[row]).
            //   Emitted Some even for old_err None: the intrinsic f32 contraction
            //   rounding is real on both layouts.
            // The hoisted ingredients below are computed in the exact order the
            // 6D arm always used, so the 6D arm stays bit-identical (pinned by
            // `dense_6d_compose_and_bias_err_bit_identical`).
            let kernel_l1: f64 = p.kernel.iter().map(|v| f64::from(*v).abs()).sum();
            let k_contraction = p.out_c.saturating_mul(prev_kh).saturating_mul(prev_kw);
            let gamma = crate::layers::linear::crown_single_gamma_n_f32(k_contraction);
            let old = patches_data.coeff_err.as_ref();
            let coeff_err = if explicit_rows {
                // Hard length check (spec I6): a carried err that does not index
                // by spec row is a construction bug; error out so the caller
                // falls back to the sound dense path — never a silent
                // `.get(i).unwrap_or(0.0)` under-count.
                if let Some(e) = old {
                    if e.len() != row_count {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![row_count],
                            got: vec![e.len()],
                        });
                    }
                }
                // Each ne[row] depends only on incoming spec-row slab `row` and
                // old_err[row] (a per-row max plus one fused expression — no
                // accumulation across rows), so rows compute in parallel with no
                // summation-order change (spec I8: rows only; max is
                // order-independent).
                let row_err = |row: usize| -> f32 {
                    let mut rowmax = 0.0f64;
                    for &v in incoming.index_axis(Axis(0), row).iter() {
                        let a = f64::from(v).abs();
                        if a > rowmax {
                            rowmax = a;
                        }
                    }
                    // Sanitize the carried err (spec I5): non-finite or negative
                    // maps to +INF (poisons outward; the row degrades at
                    // consumption) — NEVER NaN -> 0.
                    let oe = match old {
                        None => 0.0f64,
                        Some(e) => {
                            let v = e[row]; // length validated above (I6)
                            if v.is_finite() && v >= 0.0 {
                                f64::from(v)
                            } else {
                                f64::INFINITY
                            }
                        }
                    };
                    // Exact-zero short-circuits BEFORE multiplying possibly
                    // infinite factors (spec I5 + §14 C2 clamp): rowmax == 0 ⇒
                    // pure carry term (γ_K may be +INF at pathological K, and
                    // INF·0 = NaN must never be emitted); kernel_l1 == 0 ⇒ every
                    // composed product is exactly ±0 and the carried deviation
                    // is scaled by Σ|w| = 0, so both terms are exactly 0.
                    let intrinsic = if rowmax == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        gamma * rowmax * kernel_l1
                    };
                    let carry = if oe == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        kernel_l1 * oe
                    };
                    // Both addends are finite >= 0 or +INF — never NaN. f64
                    // evaluation, one outward next_up at the f32 cast (spec I4).
                    ny_tensor::next_up_f32((intrinsic + carry) as f32)
                };
                let mut ne = Array1::<f32>::zeros(row_count);
                if let Some(ne_slice) = ne.as_slice_mut() {
                    ne_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(row, out)| *out = row_err(row));
                } else {
                    for row in 0..row_count {
                        ne[row] = row_err(row);
                    }
                }
                Some(ne)
            } else {
                // Each ne[idx] depends only on incoming row idx and old_err[idx]
                // (a per-row max plus one fused expression — no accumulation across
                // rows), so rows compute in parallel with no summation-order change.
                let row_err = |idx: usize| -> f32 {
                    let (_row, soc, soh, sow) = decode(idx);
                    let mut rowmax = 0.0f64;
                    for c in 0..p.out_c {
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                let a = f64::from(incoming[[soc, soh, sow, c, ki, kj]]).abs();
                                if a > rowmax {
                                    rowmax = a;
                                }
                            }
                        }
                    }
                    let oe = old.map_or(0.0, |e| f64::from(e.get(idx).copied().unwrap_or(0.0)));
                    ny_tensor::next_up_f32((gamma * rowmax * kernel_l1 + kernel_l1 * oe) as f32)
                };
                let mut ne = Array1::<f32>::zeros(num_positions);
                if let Some(ne_slice) = ne.as_slice_mut() {
                    ne_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(idx, out)| *out = row_err(idx));
                } else {
                    for idx in 0..num_positions {
                        ne[idx] = row_err(idx);
                    }
                }
                Some(ne)
            };

            Ok(PatchesData {
                coeff_err,
                patches: Some(new_patches),
                stride: new_stride,
                padding: new_padding,
                identity: false,
                output_shape: patches_data.output_shape,
                input_shape: (p.in_c, p.in_h, p.in_w),
                unstable_idx: None,
            })
        }
    }

    /// Compute bias contribution for Patches backward.
    ///
    /// For conv bias b of shape [out_c], broadcast to [out_c, out_h, out_w]:
    /// new_b = old_b + sum over spatial positions of patches coefficients * bias
    fn compute_patches_bias(
        bounds: &PatchesLinearBounds,
        bias: &Array1<f32>,
        out_c: usize,
        out_h: usize,
        out_w: usize,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        use ny_tensor::{next_down_f32, next_up_f32};

        let lower_patches = &bounds.lower_a;
        let upper_patches = &bounds.upper_a;
        let old_lower_b = &bounds.lower_b;
        let old_upper_b = &bounds.upper_b;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d bias patches: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        if lower_patches.identity && upper_patches.identity {
            // Crash guard (mirrors BatchNorm patches fix 9a6bc1a): the per-output-neuron
            // bias indexing below (idx = oc*out_h*out_w + oh*out_w + ow) reads
            // old_lower_b[idx] for idx in 0..out_dim. Under disjunctive multi-clause
            // input-split the incoming bias is spec-row-shaped, not out_dim — a shorter
            // vector would index out of bounds (SIGABRT under panic=abort). Require the
            // exact per-neuron layout; otherwise return ShapeMismatch so the caller's
            // try_patches_or_dense_fallback drops to the sound dense Conv2d backward.
            if old_lower_b.len() != out_dim || old_upper_b.len() != out_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_dim],
                    got: vec![old_lower_b.len().min(old_upper_b.len())],
                });
            }
            let mut new_lower_b = Array1::<f32>::zeros(out_dim);
            let mut new_upper_b = Array1::<f32>::zeros(out_dim);
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx = oc * out_h * out_w + oh * out_w + ow;
                        let lb_f64 = old_lower_b[idx] as f64 + bias[oc] as f64;
                        let ub_f64 = old_upper_b[idx] as f64 + bias[oc] as f64;
                        new_lower_b[idx] = next_down_f32(lb_f64 as f32);
                        new_upper_b[idx] = next_up_f32(ub_f64 as f32);
                    }
                }
            }
            return Ok((new_lower_b, new_upper_b));
        }

        let (spec_oc, spec_oh, spec_ow) = lower_patches.output_shape;
        let mut new_lower_b = old_lower_b.clone();
        let mut new_upper_b = old_upper_b.clone();

        let lower_p = lower_patches.patches.as_ref();
        let upper_p = upper_patches.patches.as_ref();
        let explicit_rows = lower_p
            .map(|p| p.ndim() == 7)
            .or_else(|| upper_p.map(|p| p.ndim() == 7))
            .unwrap_or(false);

        if explicit_rows {
            // Crash guard: this branch indexes new_bias[row] for row in 0..row_count
            // (new_*_b are clones of the incoming spec-row-shaped bias). A bias shorter
            // than row_count would index out of bounds; fall back to dense on mismatch.
            if new_lower_b.len() != bounds.row_count || new_upper_b.len() != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![new_lower_b.len().min(new_upper_b.len())],
                });
            }
            // Hardening (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §14 C3/C4): the
            // explicit-rows fold has no analog of the 6D identity-side
            // `else if c == soc` contribution, so an identity side here would
            // silently DROP its affine term — the wrong-affine verdict-bug
            // class. Hard error instead (believed unreachable:
            // from_dense_spatial_rows materializes both sides and the merge
            // family gate bars mixed pairs). Likewise a non-7D tensor on
            // either side would panic on the 7-index reads below (the
            // pre-existing mixed 6D/7D pair panic) — clean ShapeMismatch
            // instead. Both errors route the caller to the sound dense
            // fallback.
            let lp = lower_p.ok_or_else(|| {
                NyError::UnsupportedConfiguration(
                    "Conv2d Patches bias: identity lower side in the explicit-rows fold \
                     (its affine contribution has no 7D analog); use dense CROWN"
                        .to_string(),
                )
            })?;
            let up = upper_p.ok_or_else(|| {
                NyError::UnsupportedConfiguration(
                    "Conv2d Patches bias: identity upper side in the explicit-rows fold \
                     (its affine contribution has no 7D analog); use dense CROWN"
                        .to_string(),
                )
            })?;
            if lp.ndim() != 7 || up.ndim() != 7 {
                return Err(NyError::ShapeMismatch {
                    expected: vec![7, 7],
                    got: vec![lp.ndim(), up.ndim()],
                });
            }
            let prev_kh_l = lp.shape()[5];
            let prev_kw_l = lp.shape()[6];
            let prev_kh_u = up.shape()[5];
            let prev_kw_u = up.shape()[6];

            // Certified coefficient-error discharge into the bias
            // (#patches-coeff-err-soundness, HOLE2 — spec §5.1 2B with the §14
            // A1-adopted f64-summation discharge). The fold below sums, per
            // SPEC row, Σ_{pos,c,taps} stored_coeff·bias[c] over ALL
            // spec-output positions into the ONE spec-row bias slot, so:
            //   • carried-err widen: every one of the positions·out_c·kh·kw
            //     stored coefficients of the row deviates by ≤ old_err[row]
            //     ⇒ |fold_stored − fold_true| ≤
            //       old_err[row]·positions·(kh·kw)·Σ_c|bias[c]|  (SUM-lift —
            //     the `positions` factor is exactly what the 6D per-position
            //     formula does not have);
            //   • γ̄ fold-rounding discharge (§14 A1, closes C1's f64
            //     catastrophic-cancellation corner): the fold's own f64
            //     accumulation error is ≤ γ̄·ABS[row] with
            //     ABS[row] = |b_old[row]| + Σ|coeff·bias[c]| and
            //     γ̄ = γ_n^f64(8·row_volume + 16) — ≥ 4x headroom over the
            //     ≤ 2·row_volume+4 roundings on any addend's path, absorbing
            //     the read-only ABS accumulator's own f64 deficit and the
            //     final product/cast roundings (same argument as the
            //     activation-site discharge, spec §6.2). The (1+γ̄) factor on
            //     the widen absorbs the widen product's / ‖bias‖₁ sum's f64
            //     under-reads.
            // Both discharges land in the f64 accumulator BEFORE the directed
            // cast (spec I4) and are emitted even for err-free inputs (the
            // fold rounding is intrinsic). Per side independently.
            let bias_abs_l1: f64 = bias.iter().map(|b| f64::from(*b).abs()).sum();
            let lower_err = lower_patches.coeff_err.as_ref();
            let upper_err = upper_patches.coeff_err.as_ref();
            // Hard length checks (spec I6): direct [row] indexing below, never
            // a silent `.get(row).unwrap_or(0.0)` under-count.
            if let Some(e) = lower_err {
                if e.len() != bounds.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![bounds.row_count],
                        got: vec![e.len()],
                    });
                }
            }
            if let Some(e) = upper_err {
                if e.len() != bounds.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![bounds.row_count],
                        got: vec![e.len()],
                    });
                }
            }
            let positions_usize = spec_oc.saturating_mul(spec_oh).saturating_mul(spec_ow);
            let positions = positions_usize as f64;
            let lower_taps = (prev_kh_l as f64) * (prev_kw_l as f64);
            let upper_taps = (prev_kh_u as f64) * (prev_kw_u as f64);
            let gbar_of = |taps_kh: usize, taps_kw: usize| -> f64 {
                let row_volume = positions_usize
                    .saturating_mul(out_c)
                    .saturating_mul(taps_kh)
                    .saturating_mul(taps_kw);
                crate::layers::linear::crown_single_gamma_n_f64(
                    row_volume.saturating_mul(8).saturating_add(16),
                )
            };
            let gbar_l = gbar_of(prev_kh_l, prev_kw_l);
            let gbar_u = gbar_of(prev_kh_u, prev_kw_u);
            for row in 0..bounds.row_count {
                let mut lower_sum = 0.0f64;
                let mut upper_sum = 0.0f64;
                // Read-only |·| mirrors of the fold, seeded with |b_old[row]|
                // (the final b_old + sum addition is part of the certified
                // chain). The VALUE accumulation statements and order below
                // are unchanged (spec I3).
                let mut lower_abs = f64::from(new_lower_b[row]).abs();
                let mut upper_abs = f64::from(new_upper_b[row]).abs();

                for soc in 0..spec_oc {
                    for soh in 0..spec_oh {
                        for sow in 0..spec_ow {
                            for c in 0..out_c {
                                let mut lc_sum = 0.0f64;
                                let mut uc_sum = 0.0f64;
                                let mut lc_abs = 0.0f64;
                                let mut uc_abs = 0.0f64;

                                for ki in 0..prev_kh_l {
                                    for kj in 0..prev_kw_l {
                                        let a = lp[[row, soc, soh, sow, c, ki, kj]] as f64;
                                        lc_sum += a;
                                        lc_abs += a.abs();
                                    }
                                }

                                for ki in 0..prev_kh_u {
                                    for kj in 0..prev_kw_u {
                                        let a = up[[row, soc, soh, sow, c, ki, kj]] as f64;
                                        uc_sum += a;
                                        uc_abs += a.abs();
                                    }
                                }

                                lower_sum += lc_sum * (bias[c] as f64);
                                upper_sum += uc_sum * (bias[c] as f64);
                                let bc_abs = (bias[c] as f64).abs();
                                lower_abs += lc_abs * bc_abs;
                                upper_abs += uc_abs * bc_abs;
                            }
                        }
                    }
                }

                // Per-side outward discharge
                //   D = γ̄·ABS[row] + old_err[row]·positions·taps·‖bias‖₁·(1+γ̄),
                // with the spec-I5 sanitize (non-finite/negative err ⇒ +INF
                // poison) and exact-zero short-circuits BEFORE multiplying
                // possibly infinite factors (0·INF = NaN must never appear):
                // ABS == 0 ⇒ every folded product is exactly 0 ⇒ the fold is
                // exact; widen_base == 0 ⇒ no folded tap can deviate (zero
                // taps or an all-zero bias) ⇒ zero widen is exact.
                let side_discharge =
                    |err: Option<&Array1<f32>>, taps: f64, gbar: f64, abs_sum: f64| -> f64 {
                        let oe = match err {
                            None => 0.0f64,
                            Some(e) => {
                                let v = e[row]; // length validated above (I6)
                                if v.is_finite() && v >= 0.0 {
                                    f64::from(v)
                                } else {
                                    f64::INFINITY
                                }
                            }
                        };
                        let fold_disc = if abs_sum == 0.0 { 0.0 } else { gbar * abs_sum };
                        let widen_base = positions * taps * bias_abs_l1;
                        let widen = if oe == 0.0 || widen_base == 0.0 {
                            0.0
                        } else {
                            (oe * widen_base) * (1.0 + gbar)
                        };
                        fold_disc + widen
                    };
                let dl = side_discharge(lower_err, lower_taps, gbar_l, lower_abs);
                let du = side_discharge(upper_err, upper_taps, gbar_u, upper_abs);

                // Discharge lands in the f64 accumulator BEFORE the directed
                // cast (spec I4); a non-finite discharge poisons the row
                // outward to ∓INF (vacuous certificate, never NaN — spec I5).
                new_lower_b[row] = if dl.is_finite() {
                    next_down_f32((new_lower_b[row] as f64 + lower_sum - dl) as f32)
                } else {
                    f32::NEG_INFINITY
                };
                new_upper_b[row] = if du.is_finite() {
                    next_up_f32((new_upper_b[row] as f64 + upper_sum + du) as f32)
                } else {
                    f32::INFINITY
                };
            }
        } else {
            let spec_dim = spec_oc * spec_oh * spec_ow;
            // Crash guard: this branch indexes new_bias[j] for j in 0..spec_dim. A bias
            // vector shorter than spec_dim would index out of bounds; fall back to dense.
            if new_lower_b.len() != spec_dim || new_upper_b.len() != spec_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![spec_dim],
                    got: vec![new_lower_b.len().min(new_upper_b.len())],
                });
            }
            // Certified coefficient-error discharge into the bias
            // (#patches-coeff-err-soundness, HOLE2). The fold below sums, per
            // logical row j, Σ_c (Σ_taps stored_coeff)·bias[c]. Every stored
            // coefficient in row j deviates from the true value by at most
            // old_err[j] (the incoming PatchesData coeff_err), and each of the
            // out_c·prev_kh·prev_kw taps is scaled by bias[c(tap)], so the fold
            // error is bounded by old_err[j]·Σ_taps|bias[c(tap)]| =
            // old_err[j]·(prev_kh·prev_kw)·Σ_c|bias[c]|. Widen the bias OUTWARD
            // (lower down, upper up) by that amount, per side independently.
            // 6D dense layout arm (byte-identical, pinned): an identity side
            // (patches None → 0 taps) or a sparse side (coeff_err None) yields a
            // zero widen, leaving that side's bias unchanged (no regression).
            // The 7D explicit-rows layout takes its own row-indexed branch above
            // (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5.1 2B).
            let bias_abs_l1: f64 = bias.iter().map(|b| f64::from(*b).abs()).sum();
            let lower_err = lower_patches.coeff_err.as_ref();
            let upper_err = upper_patches.coeff_err.as_ref();
            let lower_taps = lower_p
                .map(|p| p.shape()[4].saturating_mul(p.shape()[5]))
                .unwrap_or(0) as f64;
            let upper_taps = upper_p
                .map(|p| p.shape()[4].saturating_mul(p.shape()[5]))
                .unwrap_or(0) as f64;
            for j in 0..spec_dim {
                let soc = j / (spec_oh * spec_ow);
                let rem = j % (spec_oh * spec_ow);
                let soh = rem / spec_ow;
                let sow = rem % spec_ow;

                let mut lower_sum = 0.0f64;
                let mut upper_sum = 0.0f64;

                for c in 0..out_c {
                    let mut lc_sum = 0.0f64;
                    let mut uc_sum = 0.0f64;

                    if let Some(lp) = lower_p {
                        let prev_kh = lp.shape()[4];
                        let prev_kw = lp.shape()[5];
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                lc_sum += lp[[soc, soh, sow, c, ki, kj]] as f64;
                            }
                        }
                    } else if c == soc {
                        lc_sum = 1.0;
                    }

                    if let Some(up) = upper_p {
                        let prev_kh = up.shape()[4];
                        let prev_kw = up.shape()[5];
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                uc_sum += up[[soc, soh, sow, c, ki, kj]] as f64;
                            }
                        }
                    } else if c == soc {
                        uc_sum = 1.0;
                    }

                    lower_sum += lc_sum * (bias[c] as f64);
                    upper_sum += uc_sum * (bias[c] as f64);
                }

                let lower_widen = lower_err
                    .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)))
                    * lower_taps
                    * bias_abs_l1;
                let upper_widen = upper_err
                    .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)))
                    * upper_taps
                    * bias_abs_l1;

                new_lower_b[j] =
                    next_down_f32((new_lower_b[j] as f64 + lower_sum - lower_widen) as f32);
                new_upper_b[j] =
                    next_up_f32((new_upper_b[j] as f64 + upper_sum + upper_widen) as f32);
            }
        }

        Ok((new_lower_b, new_upper_b))
    }
}

// =====================================================================
// Byte-identity pin tests for the certified conv coeff_err channel
// (#patches-coeff-err-soundness; 7D explicit-rows closure spec §5.4 T4,
// docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
//
// Committed against the UNMODIFIED tree: these pin the CURRENT 6D
// compose-err formula and the CURRENT 6D bias fold + coeff_err widen
// bit-for-bit against in-test verbatim formula replicas. The 7D closure
// restructures the surrounding code (hoisting kernel_l1/gamma/old before
// the layout dispatch) but is required to keep every 6D arm byte
// identical — these tests must pass unmodified after it lands.
// =====================================================================
#[cfg(test)]
mod coeff_err_tests {
    use super::*;

    /// Deterministic non-dyadic mixed-sign fill with exact zeros sprinkled in.
    fn det_fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                if k.is_multiple_of(11) {
                    0.0
                } else {
                    (((k >> 8) % 2000) as f32 - 1000.0) * 0.001_37
                }
            })
            .collect()
    }

    fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: actual {a:?} (bits {:#010x}) != replica {e:?} (bits {:#010x})",
                a.to_bits(),
                e.to_bits()
            );
        }
    }

    /// In-test verbatim replica of the CURRENT 6D compose coeff_err rule
    /// (`conv2d_patches_backward`, non-identity 6D arm):
    ///   ne[idx] = next_up(γ_K^f32 · RowMaxAbs(incoming@idx) · ‖k‖₁ + ‖k‖₁ · old_err[idx])
    fn replica_compose_err_6d(
        incoming: &ArrayD<f32>,
        old: Option<&Array1<f32>>,
        kernel: &ArrayD<f32>,
        out_c: usize,
        prev_kh: usize,
        prev_kw: usize,
        spec: (usize, usize, usize),
    ) -> Array1<f32> {
        let (spec_oc, spec_oh, spec_ow) = spec;
        let num_positions = spec_oc * spec_oh * spec_ow;
        let kernel_l1: f64 = kernel.iter().map(|v| f64::from(*v).abs()).sum();
        let k_contraction = out_c.saturating_mul(prev_kh).saturating_mul(prev_kw);
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(k_contraction);
        let mut ne = Array1::<f32>::zeros(num_positions);
        for idx in 0..num_positions {
            let position_idx = idx % (spec_oc * spec_oh * spec_ow);
            let soc = position_idx / (spec_oh * spec_ow);
            let rem = position_idx % (spec_oh * spec_ow);
            let soh = rem / spec_ow;
            let sow = rem % spec_ow;
            let mut rowmax = 0.0f64;
            for c in 0..out_c {
                for ki in 0..prev_kh {
                    for kj in 0..prev_kw {
                        let a = f64::from(incoming[[soc, soh, sow, c, ki, kj]]).abs();
                        if a > rowmax {
                            rowmax = a;
                        }
                    }
                }
            }
            let oe = old.map_or(0.0, |e| f64::from(e.get(idx).copied().unwrap_or(0.0)));
            ne[idx] = ny_tensor::next_up_f32((gamma * rowmax * kernel_l1 + kernel_l1 * oe) as f32);
        }
        ne
    }

    /// In-test verbatim replica of the CURRENT 6D (non-identity, dense-layout)
    /// branch of `compute_patches_bias`: per-row bias fold + coeff_err widen
    /// (HOLE2 discharge) + directed casts.
    fn replica_bias_6d(
        bounds: &PatchesLinearBounds,
        bias: &Array1<f32>,
        out_c: usize,
    ) -> (Array1<f32>, Array1<f32>) {
        use ny_tensor::{next_down_f32, next_up_f32};

        let lower_patches = &bounds.lower_a;
        let upper_patches = &bounds.upper_a;
        let (spec_oc, spec_oh, spec_ow) = lower_patches.output_shape;
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_b = bounds.upper_b.clone();
        let lower_p = lower_patches.patches.as_ref();
        let upper_p = upper_patches.patches.as_ref();
        let spec_dim = spec_oc * spec_oh * spec_ow;

        let bias_abs_l1: f64 = bias.iter().map(|b| f64::from(*b).abs()).sum();
        let lower_err = lower_patches.coeff_err.as_ref();
        let upper_err = upper_patches.coeff_err.as_ref();
        let lower_taps = lower_p
            .map(|p| p.shape()[4].saturating_mul(p.shape()[5]))
            .unwrap_or(0) as f64;
        let upper_taps = upper_p
            .map(|p| p.shape()[4].saturating_mul(p.shape()[5]))
            .unwrap_or(0) as f64;
        for j in 0..spec_dim {
            let soc = j / (spec_oh * spec_ow);
            let rem = j % (spec_oh * spec_ow);
            let soh = rem / spec_ow;
            let sow = rem % spec_ow;

            let mut lower_sum = 0.0f64;
            let mut upper_sum = 0.0f64;

            for c in 0..out_c {
                let mut lc_sum = 0.0f64;
                let mut uc_sum = 0.0f64;

                if let Some(lp) = lower_p {
                    let prev_kh = lp.shape()[4];
                    let prev_kw = lp.shape()[5];
                    for ki in 0..prev_kh {
                        for kj in 0..prev_kw {
                            lc_sum += lp[[soc, soh, sow, c, ki, kj]] as f64;
                        }
                    }
                } else if c == soc {
                    lc_sum = 1.0;
                }

                if let Some(up) = upper_p {
                    let prev_kh = up.shape()[4];
                    let prev_kw = up.shape()[5];
                    for ki in 0..prev_kh {
                        for kj in 0..prev_kw {
                            uc_sum += up[[soc, soh, sow, c, ki, kj]] as f64;
                        }
                    }
                } else if c == soc {
                    uc_sum = 1.0;
                }

                lower_sum += lc_sum * (bias[c] as f64);
                upper_sum += uc_sum * (bias[c] as f64);
            }

            let lower_widen = lower_err
                .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)))
                * lower_taps
                * bias_abs_l1;
            let upper_widen = upper_err
                .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)))
                * upper_taps
                * bias_abs_l1;

            new_lower_b[j] =
                next_down_f32((new_lower_b[j] as f64 + lower_sum - lower_widen) as f32);
            new_upper_b[j] = next_up_f32((new_upper_b[j] as f64 + upper_sum + upper_widen) as f32);
        }
        (new_lower_b, new_upper_b)
    }

    fn make_patches_data(
        shape: &[usize],
        seed: u32,
        coeff_err: Option<Array1<f32>>,
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
    ) -> PatchesData {
        let n: usize = shape.iter().product();
        PatchesData {
            coeff_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(shape), det_fill(n, seed)).unwrap()),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        }
    }

    /// Spec §5.4 T4 pin: 6D compose err and 6D bias fold+widen are bit-identical
    /// to in-test verbatim formula replicas, with err-carrying AND err-free
    /// inputs (the latter pins that the err channel never perturbs values).
    #[test]
    fn dense_6d_compose_and_bias_err_bit_identical() {
        // ---- fixture geometry ----
        // conv: kernel [out_c=3, in_c=2, kh=2, kw=2], stride 1, pad 0, in 3x3 -> out 2x2
        // incoming 6D patches: [spec_oc=2, spec_oh=2, spec_ow=2, out_c=3, prev_kh=2, prev_kw=2]
        let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 2]), det_fill(24, 77)).unwrap();
        let params = Conv2dPatchesParams {
            kernel: &kernel,
            in_c: 2,
            out_c: 3,
            groups: 1,
            kh: 2,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            in_h: 3,
            in_w: 3,
            out_h: 2,
            out_w: 2,
        };
        let spec = (2usize, 2usize, 2usize);
        let spec_dim = spec.0 * spec.1 * spec.2;
        let lower_err = Array1::from_vec(vec![
            1.0e-3_f32, 0.0, 5.0e-4, 2.0e-6, 3.0e-3, 0.0, 7.0e-5, 1.0e-4,
        ]);
        let upper_err = Array1::from_vec(vec![
            2.0e-3_f32, 1.0e-4, 0.0, 5.0e-5, 4.0e-4, 0.0, 6.0e-4, 8.0e-4,
        ]);

        // ---- 2A: compose err, err-carrying input ----
        let pd = make_patches_data(
            &[2, 2, 2, 3, 2, 2],
            1,
            Some(lower_err.clone()),
            spec,
            (3, 2, 2),
        );
        let incoming = pd.patches.as_ref().unwrap().clone();
        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out
            .coeff_err
            .as_ref()
            .expect("6D compose must emit Some coeff_err");
        let expected = replica_compose_err_6d(&incoming, Some(&lower_err), &kernel, 3, 2, 2, spec);
        assert_bits_eq(
            "compose ne (err-carrying)",
            ne.as_slice().unwrap(),
            expected.as_slice().unwrap(),
        );
        // Sanity: the carried term is live (row 0 has nonzero old err).
        assert!(ne[0] > 0.0 && ne[0].is_finite());

        // ---- 2A: compose err, err-free input (oe = 0, intrinsic term only),
        // and the VALUE tensor must be bit-identical to the err-carrying run ----
        let pd_none = make_patches_data(&[2, 2, 2, 3, 2, 2], 1, None, spec, (3, 2, 2));
        let out_none = Conv2dLayer::conv2d_patches_backward(&pd_none, &params, None).unwrap();
        let ne_none = out_none
            .coeff_err
            .as_ref()
            .expect("6D compose must emit Some coeff_err for err-free input");
        let expected_none = replica_compose_err_6d(&incoming, None, &kernel, 3, 2, 2, spec);
        assert_bits_eq(
            "compose ne (err-free)",
            ne_none.as_slice().unwrap(),
            expected_none.as_slice().unwrap(),
        );
        assert_bits_eq(
            "composed value tensor unchanged by err channel",
            out.patches.as_ref().unwrap().as_slice().unwrap(),
            out_none.patches.as_ref().unwrap().as_slice().unwrap(),
        );

        // ---- 2B: bias fold + widen, err-carrying on both sides ----
        let bias = Array1::from_vec(vec![0.3_f32, -0.7, 0.11]);
        let plb = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 1, Some(lower_err), spec, (3, 2, 2)),
            lower_b: Array1::from_vec(det_fill(spec_dim, 900)),
            upper_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 2, Some(upper_err), spec, (3, 2, 2)),
            upper_b: Array1::from_vec(det_fill(spec_dim, 901)),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 2, 2).unwrap();
        let (rlb, rub) = replica_bias_6d(&plb, &bias, 3);
        assert_bits_eq(
            "bias lower (err-carrying)",
            nlb.as_slice().unwrap(),
            rlb.as_slice().unwrap(),
        );
        assert_bits_eq(
            "bias upper (err-carrying)",
            nub.as_slice().unwrap(),
            rub.as_slice().unwrap(),
        );

        // ---- 2B: bias fold, err-free (widen exactly 0 — the plain fold) ----
        let plb_none = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 1, None, spec, (3, 2, 2)),
            lower_b: plb.lower_b.clone(),
            upper_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 2, None, spec, (3, 2, 2)),
            upper_b: plb.upper_b,
        };
        let (nlb_none, nub_none) =
            Conv2dLayer::compute_patches_bias(&plb_none, &bias, 3, 2, 2).unwrap();
        let (rlb_none, rub_none) = replica_bias_6d(&plb_none, &bias, 3);
        assert_bits_eq(
            "bias lower (err-free)",
            nlb_none.as_slice().unwrap(),
            rlb_none.as_slice().unwrap(),
        );
        assert_bits_eq(
            "bias upper (err-free)",
            nub_none.as_slice().unwrap(),
            rub_none.as_slice().unwrap(),
        );
    }

    // =================================================================
    // 7D explicit-rows tests (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5.4
    // T1/T2/T3 + the I5 poison guard).
    //
    // Shared 7D fixture geometry:
    //   conv: kernel [out_c=3, in_c=2, kh=2, kw=2], stride 1, pad 0,
    //         input 4x4 -> out 3x3;
    //   incoming: [row_count=2, spec=(2,2,2), out_c=3, prev_kh=2, prev_kw=2]
    //   composed: [2, 2, 2, 2, in_c=2, new_kh=3, new_kw=3], <= 12 taps/cell.
    // =================================================================

    /// Deterministic dyadic fill: multiples of 2^-6 in [-2, 2]. Exact in f32
    /// and through every small dyadic product/sum the oracles below build, so
    /// coverage is asserted with NO tolerance.
    fn dyadic_fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                (((k >> 7) % 257) as f32 - 128.0) / 64.0
            })
            .collect()
    }

    fn fixture_params(kernel: &ArrayD<f32>) -> Conv2dPatchesParams<'_> {
        Conv2dPatchesParams {
            kernel,
            in_c: 2,
            out_c: 3,
            groups: 1,
            kh: 2,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            in_h: 4,
            in_w: 4,
            out_h: 3,
            out_w: 3,
        }
    }

    /// All-positive dyadic kernel [3, 2, 2, 2] (multiples of 2^-4).
    fn fixture_kernel() -> ArrayD<f32> {
        let vals: Vec<f32> = (0..24).map(|i| ((i % 15) + 1) as f32 / 16.0).collect();
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 2]), vals).unwrap()
    }

    fn make_patches_7d(vals: Vec<f32>, coeff_err: Option<Array1<f32>>) -> PatchesData {
        PatchesData {
            coeff_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), vals).unwrap()),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (2, 2, 2),
            input_shape: (3, 3, 3),
            unstable_idx: None,
        }
    }

    /// Exhaustive f64 transpose-conv oracle for the fixture geometry
    /// (stride 1, padding 0): true composed coefficient of output cell
    /// (ic, ni, nj) from incoming slab (row, soc, soh, sow, ·).
    fn transpose_conv_oracle(
        a: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        cell: (usize, usize, usize, usize, usize, usize, usize),
    ) -> f64 {
        let (row, soc, soh, sow, ic, ni, nj) = cell;
        let mut acc = 0.0f64;
        for c in 0..3 {
            for gy in 0..2 {
                for gx in 0..2 {
                    let ki = ni as isize - gy as isize;
                    let kj = nj as isize - gx as isize;
                    if (0..2).contains(&ki) && (0..2).contains(&kj) {
                        acc += f64::from(a[[row, soc, soh, sow, c, gy, gx]])
                            * f64::from(kernel[[c, ic, ki as usize, kj as usize]]);
                    }
                }
            }
        }
        acc
    }

    /// Spec §5.4 T1: the 7D compose emits a SPEC-ROW-indexed err
    /// (len row_count, not num_positions) that covers |stored − true| for
    /// EVERY composed coefficient. Dyadic fixture: kernel and truths dyadic,
    /// stored = true + old_err[row] tap-wise (all exact in f32, composed
    /// values exact, <= 12 dyadic terms per cell), so the deviation is
    /// EXACTLY old_err[row]·Σ_taps w and coverage is asserted tolerance-free.
    #[test]
    fn explicit_rows_compose_err_covers_true_deviation() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);

        let old_err = [2f32.powi(-12), 2f32.powi(-8)];
        let n = 2 * 8 * 12;
        let per_row = n / 2;
        let a_true_vals = dyadic_fill(n, 9);
        let a_stored_vals: Vec<f32> = a_true_vals
            .iter()
            .enumerate()
            .map(|(i, &v)| v + old_err[i / per_row])
            .collect();
        let a_true = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), a_true_vals).unwrap();
        let pd = make_patches_7d(a_stored_vals, Some(Array1::from_vec(old_err.to_vec())));

        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out
            .coeff_err
            .as_ref()
            .expect("7D compose must emit Some coeff_err");
        assert_eq!(ne.len(), 2, "err must be spec-row indexed (len row_count)");
        assert!(ne.iter().all(|e| e.is_finite() && *e > 0.0));

        let stored = out.patches.as_ref().unwrap();
        assert_eq!(stored.shape(), &[2, 2, 2, 2, 2, 3, 3]);
        let mut max_dev = [0.0f64; 2];
        for row in 0..2 {
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for ic in 0..2 {
                            for ni in 0..3 {
                                for nj in 0..3 {
                                    let tru = transpose_conv_oracle(
                                        &a_true,
                                        &kernel,
                                        (row, soc, soh, sow, ic, ni, nj),
                                    );
                                    let dev = (f64::from(stored[[row, soc, soh, sow, ic, ni, nj]])
                                        - tru)
                                        .abs();
                                    assert!(
                                        dev <= f64::from(ne[row]),
                                        "row {row} cell ({soc},{soh},{sow},{ic},{ni},{nj}): \
                                         deviation {dev:e} not covered by err {:e}",
                                        ne[row]
                                    );
                                    max_dev[row] = max_dev[row].max(dev);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sharpness: the ‖k‖₁·old_err[row] carry term is load-bearing — an
        // implementation without it (or misindexing the err by position
        // instead of spec row) could at most emit the intrinsic γ_K term,
        // which the actual deviation exceeds on BOTH rows.
        let kernel_l1: f64 = kernel.iter().map(|v| f64::from(*v).abs()).sum();
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(3 * 2 * 2);
        for row in 0..2 {
            let mut rowmax = 0.0f64;
            for &v in pd.patches.as_ref().unwrap().index_axis(Axis(0), row).iter() {
                rowmax = rowmax.max(f64::from(v).abs());
            }
            let intrinsic_only =
                f64::from(ny_tensor::next_up_f32((gamma * rowmax * kernel_l1) as f32));
            assert!(
                max_dev[row] > intrinsic_only,
                "row {row}: fixture too weak to pin the carried term \
                 ({:e} <= {intrinsic_only:e})",
                max_dev[row]
            );
        }

        // Err-free input: Some is still emitted (the intrinsic f32
        // contraction rounding is real) and covers an f64 oracle. Non-dyadic
        // fills so the composition genuinely rounds; the f64 oracle's own
        // noise (~2^-52 relative) is orders below the γ_K-based emission.
        let pd_none = make_patches_7d(det_fill(n, 5), None);
        let out_none = Conv2dLayer::conv2d_patches_backward(&pd_none, &params, None).unwrap();
        let ne_none = out_none
            .coeff_err
            .as_ref()
            .expect("7D compose must emit Some coeff_err for err-free input");
        assert_eq!(ne_none.len(), 2);
        let a_none = pd_none.patches.as_ref().unwrap();
        let stored_none = out_none.patches.as_ref().unwrap();
        for row in 0..2 {
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for ic in 0..2 {
                            for ni in 0..3 {
                                for nj in 0..3 {
                                    let tru = transpose_conv_oracle(
                                        a_none,
                                        &kernel,
                                        (row, soc, soh, sow, ic, ni, nj),
                                    );
                                    let dev =
                                        (f64::from(stored_none[[row, soc, soh, sow, ic, ni, nj]])
                                            - tru)
                                            .abs();
                                    assert!(
                                        dev <= f64::from(ne_none[row]),
                                        "err-free row {row}: intrinsic rounding not covered"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Spec §5.4 T2: a carried 7D err whose length is not the spec row count
    /// is a hard ShapeMismatch (never a silent `.get().unwrap_or(0.0)`
    /// under-count), for both the compose and the bias fold; plus the §14 C4
    /// hardening (mixed 6D/7D pair: clean error where the old code panicked).
    #[test]
    fn explicit_rows_compose_err_length_mismatch_rejected() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);
        let n = 2 * 8 * 12;

        // 2A compose: Some(len 3) against row_count (= shape[0]) 2.
        let pd = make_patches_7d(
            dyadic_fill(n, 3),
            Some(Array1::from_vec(vec![1e-4_f32, 2e-4, 3e-4])),
        );
        let r = Conv2dLayer::conv2d_patches_backward(&pd, &params, None);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "compose must reject a wrong-length 7D err, got {r:?}"
        );

        // 2B bias fold: wrong-length err on the lower side, then the upper.
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let good = || {
            make_patches_7d(
                dyadic_fill(n, 3),
                Some(Array1::from_vec(vec![1e-4_f32, 2e-4])),
            )
        };
        let bad = || make_patches_7d(dyadic_fill(n, 4), Some(Array1::from_vec(vec![1e-4_f32; 3])));
        let lower_b = Array1::from_vec(vec![0.5_f32, -0.25]);
        let upper_b = Array1::from_vec(vec![1.0_f32, 0.125]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: bad(),
            lower_b: lower_b.clone(),
            upper_a: good(),
            upper_b: upper_b.clone(),
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a wrong-length lower err, got {r:?}"
        );
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: good(),
            lower_b: lower_b.clone(),
            upper_a: bad(),
            upper_b: upper_b.clone(),
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a wrong-length upper err, got {r:?}"
        );

        // §14 C4: a mixed 7D-lower/6D-upper pair is a clean ShapeMismatch
        // (the pre-change code panicked on the 7-index read of the 6D side).
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: good(),
            lower_b,
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 3, 2, 2]), dyadic_fill(96, 8)).unwrap(),
                ),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape: (2, 2, 2),
                input_shape: (3, 3, 3),
                unstable_idx: None,
            },
            upper_b,
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a mixed 6D/7D side pair, got {r:?}"
        );
    }

    /// Spec §5.4 T3 — THE PRE-FIX HOLE2 UNSOUNDNESS REPRODUCER: with carried
    /// coefficient errs, the adversarial admissible truths
    /// `a_true = a_stored ∓ sign(bias[c])·old_err[row]` (per side) push the
    /// true folded bias strictly past what the no-widen fold brackets; the
    /// widened fold must still enclose them. This test FAILS on the
    /// pre-closure explicit-rows branch (which applied no widen at all). All
    /// fixture values are dyadic, so the f64 oracle folds are EXACT and the
    /// asserts carry no tolerance.
    #[test]
    fn explicit_rows_bias_widen_covers_adversarial_truth() {
        let n = 2 * 8 * 12;
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let le = [2f32.powi(-6), 2f32.powi(-9)];
        let ue = [2f32.powi(-7), 2f32.powi(-10)];
        let old_lb = Array1::from_vec(vec![0.5_f32, -0.25]);
        let old_ub = Array1::from_vec(vec![1.25_f32, 0.125]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(dyadic_fill(n, 31), Some(Array1::from_vec(le.to_vec()))),
            lower_b: old_lb.clone(),
            upper_a: make_patches_7d(dyadic_fill(n, 57), Some(Array1::from_vec(ue.to_vec()))),
            upper_b: old_ub.clone(),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3).unwrap();

        let lt = plb.lower_a.patches.as_ref().unwrap();
        let ut = plb.upper_a.patches.as_ref().unwrap();
        // Σ over all folded taps of |bias[c(tap)]| =
        // positions·(kh·kw)·Σ_c|bias_c| = 8·4·1.5 = 48 (exact).
        let widen_factor = 48.0f64;
        for row in 0..2 {
            let mut s_l = 0.0f64;
            let mut s_u = 0.0f64;
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for c in 0..3 {
                            for ki in 0..2 {
                                for kj in 0..2 {
                                    s_l += f64::from(lt[[row, soc, soh, sow, c, ki, kj]])
                                        * f64::from(bias[c]);
                                    s_u += f64::from(ut[[row, soc, soh, sow, c, ki, kj]])
                                        * f64::from(bias[c]);
                                }
                            }
                        }
                    }
                }
            }
            // Extremal admissible truths (|Δ per tap| = old_err[row], each
            // tap's sign chosen against the bias): T_min = S − e·48 (lower),
            // T_max = S + e·48 (upper), exactly.
            let t_min = s_l - f64::from(le[row]) * widen_factor;
            let t_max = s_u + f64::from(ue[row]) * widen_factor;
            assert!(
                f64::from(nlb[row]) <= f64::from(old_lb[row]) + t_min,
                "row {row}: lower bias {} does not enclose the adversarial true \
                 fold {} (pre-fix HOLE2)",
                nlb[row],
                f64::from(old_lb[row]) + t_min
            );
            assert!(
                f64::from(nub[row]) >= f64::from(old_ub[row]) + t_max,
                "row {row}: upper bias {} does not enclose the adversarial true \
                 fold {} (pre-fix HOLE2)",
                nub[row],
                f64::from(old_ub[row]) + t_max
            );
            // Liveness: the widen is real — strictly outside a no-widen fold.
            let rep_l = ny_tensor::next_down_f32((f64::from(old_lb[row]) + s_l) as f32);
            let rep_u = ny_tensor::next_up_f32((f64::from(old_ub[row]) + s_u) as f32);
            assert!(nlb[row] < rep_l, "row {row}: lower widen not live");
            assert!(nub[row] > rep_u, "row {row}: upper widen not live");
        }

        // §14 C3 hardening: an identity side in the explicit-rows fold is a
        // hard error (its affine contribution has no 7D analog in the fold),
        // never a silent drop.
        let plb_ident = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(dyadic_fill(n, 31), Some(Array1::from_vec(le.to_vec()))),
            lower_b: old_lb,
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (2, 2, 2),
                input_shape: (3, 3, 3),
                unstable_idx: None,
            },
            upper_b: old_ub,
        };
        assert!(
            Conv2dLayer::compute_patches_bias(&plb_ident, &bias, 3, 3, 3).is_err(),
            "identity side must be a hard error on the explicit-rows bias fold"
        );
    }

    /// Spec I5 pins for both 7D arms: non-finite or negative carried err
    /// poisons OUTWARD (+INF err; −INF lower / +INF upper bias) on the
    /// affected rows only — NEVER NaN and never a silent 0 — and the
    /// 0·INF hazards are short-circuited.
    #[test]
    fn explicit_rows_nonfinite_err_poisons_outward_never_nan() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);
        let n = 2 * 8 * 12;

        // 2A compose: NaN err row poisons to +INF; the other row stays finite.
        let pd = make_patches_7d(
            dyadic_fill(n, 11),
            Some(Array1::from_vec(vec![f32::NAN, 1e-4])),
        );
        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out.coeff_err.as_ref().unwrap();
        assert_eq!(
            ne[0],
            f32::INFINITY,
            "NaN err must poison to +INF, never 0/NaN"
        );
        assert!(ne[1].is_finite() && ne[1] > 0.0);

        // Negative err likewise poisons (it violates the channel contract).
        let pd_neg = make_patches_7d(
            dyadic_fill(n, 11),
            Some(Array1::from_vec(vec![-1.0_f32, 1e-4])),
        );
        let out_neg = Conv2dLayer::conv2d_patches_backward(&pd_neg, &params, None).unwrap();
        assert_eq!(out_neg.coeff_err.as_ref().unwrap()[0], f32::INFINITY);

        // 2B bias fold: an +INF err row poisons THAT side's bias outward on
        // ITS rows only; the other rows and the other side stay finite; no
        // NaN anywhere.
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(
                dyadic_fill(n, 31),
                Some(Array1::from_vec(vec![f32::INFINITY, 2f32.powi(-9)])),
            ),
            lower_b: Array1::from_vec(vec![0.5_f32, -0.25]),
            upper_a: make_patches_7d(
                dyadic_fill(n, 57),
                Some(Array1::from_vec(vec![2f32.powi(-9), f32::INFINITY])),
            ),
            upper_b: Array1::from_vec(vec![1.25_f32, 0.125]),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3).unwrap();
        assert_eq!(nlb[0], f32::NEG_INFINITY);
        assert!(nlb[1].is_finite());
        assert!(nub[0].is_finite());
        assert_eq!(nub[1], f32::INFINITY);
        assert!(nlb.iter().chain(nub.iter()).all(|v| !v.is_nan()));

        // 0·INF short-circuit: with an all-zero conv bias no folded tap can
        // deviate the fold (the true widen is exactly 0), so even an +INF
        // carried err must yield finite, NaN-free biases.
        let zero_bias = Array1::from_vec(vec![0.0_f32, 0.0, 0.0]);
        let (zlb, zub) = Conv2dLayer::compute_patches_bias(&plb, &zero_bias, 3, 3, 3).unwrap();
        assert!(
            zlb.iter().chain(zub.iter()).all(|v| v.is_finite()),
            "all-zero bias with +INF err must not poison or NaN: {zlb:?} {zub:?}"
        );
    }
}
