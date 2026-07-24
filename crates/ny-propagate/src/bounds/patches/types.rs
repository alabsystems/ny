// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::mem::size_of;

/// Indices of unstable output neurons for sparse patches mode.
///
/// When set on `PatchesData`, the patches tensor has shape
/// `(unstable_size, in_c, kH, kW)` (4D) instead of the full
/// `(out_c, out_h, out_w, in_c, kH, kW)` (6D). Each entry `(c, h, w)`
/// maps sparse index `i` to output position `(c, h, w)` in the full grid.
///
/// Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` `Patches.unstable_idx`
/// Part of #2613 Phase 4 step 19
#[derive(Debug, Clone)]
pub(crate) struct UnstableIdx {
    /// Channel indices of unstable output neurons.
    pub(crate) channels: Vec<usize>,
    /// Height indices of unstable output neurons.
    pub(crate) heights: Vec<usize>,
    /// Width indices of unstable output neurons.
    pub(crate) widths: Vec<usize>,
}

impl UnstableIdx {
    /// Number of unstable neurons tracked.
    pub(crate) fn len(&self) -> usize {
        self.channels.len()
    }

    /// Convert sparse index `i` to the flat output index within
    /// the full `(out_c, out_h, out_w)` grid.
    ///
    /// Callers must first call [`UnstableIdx::validate`] for the same
    /// `(out_c, out_h, out_w)` grid; that guarantees `i < len()` and that the
    /// `(channel, height, width)` triple is in-bounds, so the indexing here and
    /// the flat result are guaranteed valid. Validating once up front keeps this
    /// hot inner helper branch-free.
    pub(crate) fn flat_index(&self, i: usize, out_h: usize, out_w: usize) -> usize {
        self.channels[i] * out_h * out_w + self.heights[i] * out_w + self.widths[i]
    }

    /// Validate the sparse-index layout against the output grid it indexes into.
    ///
    /// Sparse patches store, per unstable neuron `i`, a `(channel, height, width)`
    /// triple that downstream code turns into a flat row index via
    /// [`UnstableIdx::flat_index`] (then used to index dense rows, the dense
    /// `(flat, flat)` diagonal, and the expanded bias vectors). Those paths index
    /// without bounds checks, so a layout mismatch (parallel vectors of different
    /// lengths, a channel/height/width past the output extent, or a sparse bias
    /// whose length disagrees with the index count) is an out-of-bounds panic.
    ///
    /// This one-time check rejects such a mismatch with a clean [`NyError`] so the
    /// caller can fall back to the sound dense CROWN path instead of panicking.
    /// No bound math changes — it only refuses to index a malformed layout.
    ///
    /// `sparse_bias_len` is the length the sparse lower/upper bias vectors are
    /// expected to have when biases are consumed per-sparse-row (`Some(n)`); pass
    /// `None` when the caller carries explicit (full row-count) biases that are
    /// not indexed by sparse `i`.
    pub(crate) fn validate(
        &self,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        sparse_bias_len: Option<usize>,
    ) -> Result<()> {
        let n = self.channels.len();
        if self.heights.len() != n || self.widths.len() != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![self.heights.len(), self.widths.len()],
            });
        }
        if let Some(bias_len) = sparse_bias_len {
            if bias_len != n {
                return Err(NyError::ShapeMismatch {
                    expected: vec![n],
                    got: vec![bias_len],
                });
            }
        }
        for i in 0..n {
            if self.channels[i] >= out_c || self.heights[i] >= out_h || self.widths[i] >= out_w {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_c, out_h, out_w],
                    got: vec![self.channels[i], self.heights[i], self.widths[i]],
                });
            }
        }
        Ok(())
    }

    /// Create from IBP output bounds by identifying unstable neurons.
    ///
    /// A neuron is unstable if `lower < 0 < upper` (could be either sign).
    /// Returns `None` if the unstable fraction is above `(1.0 - min_sparsity)`,
    /// meaning there aren't enough stable neurons to justify sparse mode.
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py` `get_unstable_locations()`
    pub(crate) fn from_ibp_bounds(
        lower: &[f32],
        upper: &[f32],
        spatial: (usize, usize, usize),
        min_sparsity: f32,
    ) -> Option<Self> {
        let (out_c, out_h, out_w) = spatial;
        let total = checked_shape_product(&[out_c, out_h, out_w])?;
        debug_assert_eq!(lower.len(), total);
        debug_assert_eq!(upper.len(), total);

        let mut channels = Vec::new();
        let mut heights = Vec::new();
        let mut widths = Vec::new();

        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    let flat = c * out_h * out_w + h * out_w + w;
                    if lower[flat] < 0.0 && upper[flat] > 0.0 {
                        channels.push(c);
                        heights.push(h);
                        widths.push(w);
                    }
                }
            }
        }

        let unstable_count = channels.len();
        let stable_frac = 1.0 - (unstable_count as f32 / total as f32);
        if stable_frac < min_sparsity {
            return None;
        }

        Some(UnstableIdx {
            channels,
            heights,
            widths,
        })
    }
}

/// Sparse convolutional patches for CNN-optimized CROWN backward.
///
/// Stores the receptive field coefficients per output position instead of
/// a full dense [out_dim x in_dim] matrix. Memory: O(out_c * out_h * out_w *
/// in_c * kH * kW) instead of O((out_c * out_h * out_w)^2).
///
/// Reference: alpha-beta-CROWN auto_LiRPA/patches.py, Patches class
#[derive(Debug, Clone)]
pub(crate) struct PatchesData {
    /// Patches tensor.
    ///
    /// **Dense mode** (`unstable_idx` is `None`):
    ///   Shape: (out_c, out_h, out_w, in_c, kH, kW) — 6D
    ///
    /// **Sparse mode** (`unstable_idx` is `Some`):
    ///   Shape: (unstable_size, in_c, kH, kW) — 4D
    ///   Only patches for unstable output neurons are stored.
    ///
    /// `None` when identity=true (virtual identity, not materialized).
    pub(crate) patches: Option<ArrayD<f32>>,

    /// Composed stride across chained convolutions
    pub(crate) stride: (usize, usize),

    /// Composed padding: (left, right, top, bottom)
    pub(crate) padding: (usize, usize, usize, usize),

    /// True when this is the identity transform (no actual tensor)
    pub(crate) identity: bool,

    /// Output spatial shape: (out_c, out_h, out_w)
    pub(crate) output_shape: (usize, usize, usize),

    /// Input spatial shape: (in_c, in_h, in_w) — set at creation
    pub(crate) input_shape: (usize, usize, usize),

    /// Indices of unstable output neurons for sparse patches mode.
    ///
    /// When `None`, all output neurons are tracked (dense patches, 6D).
    /// When `Some`, only unstable neurons are tracked (sparse patches, 4D).
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` `Patches.unstable_idx`
    /// Part of #2613 Phase 4 step 19
    pub(crate) unstable_idx: Option<UnstableIdx>,

    /// Certified per-logical-ROW coefficient error (#patches-coeff-err-soundness).
    ///
    /// `Some(err)`: for every stored coefficient in logical row `i`,
    /// `|stored − true| ≤ err[i]` (all entries ≥ 0, finite or `+INF`).
    /// `None` ⇒ exact (0). `+INF` is the degrade poison: the row carries no
    /// certified bound and degrades to `[-inf, +inf]` at concretize. NaN is
    /// NEVER a legal stored err; consumers sanitize non-finite/negative
    /// entries to `+INF` (outward), never to 0
    /// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md I5/X3).
    /// Indexed identically to the row's bias (`PatchesLinearBounds::{lower_b,upper_b}`):
    /// 6D dense ⇒ `out_c*out_h*out_w`; 7D explicit-rows ⇒ `row_count`; 4D/5D sparse ⇒
    /// `unstable_size`. Carried per side (lower_a / upper_a each have their own), and
    /// materialized into an overlap-aware per-column error at `to_dense`.
    ///
    /// Closes the false-VERIFIED gap: the f32 patches composition + to_dense scatter
    /// rounding previously reached the verdict uncertified. Mirrors the dense conv
    /// `conv_coeff_err_matrix` certified error, reduced to a per-row over-bound.
    pub(crate) coeff_err: Option<ndarray::Array1<f32>>,
}

impl PatchesData {
    /// Effective kernel size of the composed patches.
    pub(crate) fn effective_kernel_size(&self) -> (usize, usize) {
        match &self.patches {
            Some(p) => {
                let shape = p.shape();
                match shape.len() {
                    4 => (shape[2], shape[3]),
                    5 => (shape[3], shape[4]),
                    6 => (shape[4], shape[5]),
                    7 => (shape[5], shape[6]),
                    _ => (1, 1),
                }
            }
            None => (1, 1),
        }
    }

    /// Check if patches should fall back to dense.
    ///
    /// Returns true once the (already-materialized) patches tensor no longer saves
    /// memory over the equivalent dense A-matrix. The patches element count is
    /// `out_c * out_h * out_w * in_c * kh * kw` vs dense
    /// `out_c * out_h * out_w * in_c * in_h * in_w`; the common factor cancels, so
    /// patches are cheaper exactly while `kh * kw < in_h * in_w` (the area
    /// crossover). The effective kernel is clamped to the input extent since the
    /// patch can never index beyond it. This matches the area-based pre-check in
    /// `would_conv_compose_cover_input` so the pre-check and post-check agree.
    pub(crate) fn should_fallback_to_dense(&self) -> bool {
        let (_, in_h, in_w) = self.input_shape;
        let (kh, kw) = self.effective_kernel_size();
        let eff_kh = kh.min(in_h);
        let eff_kw = kw.min(in_w);
        eff_kh.saturating_mul(eff_kw) >= in_h.saturating_mul(in_w)
    }

    /// Predict whether composing through a Conv2d backward would produce patches
    /// that no longer save memory over the equivalent dense A-matrix.
    ///
    /// The composed kernel size is `(prev_kh - 1) * stride + conv_kh`. The patches
    /// tensor that would result has element count
    /// `out_c * out_h * out_w * in_c * new_kh * new_kw`, while the equivalent dense
    /// A-matrix has `out_c * out_h * out_w * in_c * in_h * in_w` elements. The
    /// `out_c * out_h * out_w * in_c` factor is common, so patches use *less*
    /// memory than dense exactly while `new_kh * new_kw < in_h * in_w`.
    ///
    /// This pre-check therefore bails to dense only when patches reach or exceed
    /// the dense element count (the true memory/compute crossover), keeping patches
    /// mode active for deep conv trunks where the receptive field is still smaller
    /// than the input — previously a fixed 75%-per-dimension heuristic (#3813)
    /// bailed at ~56% of the dense area (0.75² per dim), abandoning patches while
    /// it was still nearly 2x cheaper. Switching to the area crossover keeps patches
    /// active far deeper without ever letting the patches tensor exceed dense. (#hotpath)
    ///
    /// Note `new_kh`/`new_kw` are clamped to the input extent: the composed
    /// receptive field can never reference input pixels beyond `in_h`/`in_w`
    /// (`conv2d_transpose` output is bounded by the input), so the materialized
    /// patches are at most `in_h * in_w` wide. Clamping makes the crossover compare
    /// the *effective* (useful) patch area, matching what `to_dense` materializes.
    pub(crate) fn would_conv_compose_cover_input(
        &self,
        conv_stride: (usize, usize),
        conv_kernel: (usize, usize),
        conv_input_h: usize,
        conv_input_w: usize,
    ) -> bool {
        let (prev_kh, prev_kw) = self.effective_kernel_size();
        let new_kh = if self.identity {
            conv_kernel.0
        } else {
            (prev_kh - 1) * conv_stride.0 + conv_kernel.0
        };
        let new_kw = if self.identity {
            conv_kernel.1
        } else {
            (prev_kw - 1) * conv_stride.1 + conv_kernel.1
        };
        // Effective patch extent never exceeds the input it indexes into.
        let eff_kh = new_kh.min(conv_input_h);
        let eff_kw = new_kw.min(conv_input_w);
        // Memory crossover: bail only once the patch area reaches the dense area.
        // `usize` products are safe here (already-validated layer spatial dims).
        eff_kh.saturating_mul(eff_kw) >= conv_input_h.saturating_mul(conv_input_w)
    }

    /// Materialize identity patches into an explicit tensor.
    ///
    /// Converts the virtual identity (patches=None, identity=true) into a
    /// concrete 6D tensor of shape (out_c, out_h, out_w, in_c, 1, 1) where
    /// only the diagonal channel entries are 1.0. This is needed before
    /// element-wise activation backward, which must scale individual coefficients
    /// by per-neuron slopes.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
    pub(crate) fn materialize_identity(&self) -> PatchesData {
        debug_assert!(
            self.identity,
            "materialize_identity called on non-identity PatchesData"
        );
        let (out_c, out_h, out_w) = self.output_shape;
        let (in_c, _in_h, _in_w) = self.input_shape;

        if let Some(idx) = &self.unstable_idx {
            let n = idx.len();
            let mut patches = ArrayD::<f32>::zeros(IxDyn(&[n, in_c, 1, 1]));
            for (i, &c) in idx.channels.iter().enumerate() {
                if c < in_c {
                    patches[[i, c, 0, 0]] = 1.0;
                }
            }
            return PatchesData {
                patches: Some(patches),
                stride: self.stride,
                padding: self.padding,
                identity: false,
                output_shape: self.output_shape,
                input_shape: self.input_shape,
                unstable_idx: self.unstable_idx.clone(),
                coeff_err: None, // identity diagonal store is exact (0/1)
            };
        }

        let mut patches = ArrayD::<f32>::zeros(IxDyn(&[out_c, out_h, out_w, in_c, 1, 1]));
        for c in 0..out_c.min(in_c) {
            for h in 0..out_h {
                for w in 0..out_w {
                    patches[[c, h, w, c, 0, 0]] = 1.0;
                }
            }
        }

        PatchesData {
            patches: Some(patches),
            stride: self.stride,
            padding: self.padding,
            identity: false,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            unstable_idx: None,
            coeff_err: None, // identity diagonal store is exact (0/1)
        }
    }

    /// Heap memory used by this patches tensor, in bytes.
    pub(crate) fn memory_bytes(&self) -> usize {
        match &self.patches {
            Some(p) => p.len() * size_of::<f32>(),
            None => 0,
        }
        .saturating_add(
            self.coeff_err
                .as_ref()
                .map_or(0, |e| e.len() * size_of::<f32>()),
        )
    }
}
