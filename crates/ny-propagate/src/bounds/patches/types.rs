// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::{collections::HashSet, mem::size_of, sync::Arc};

use super::PatchesMaterializationDeadline;

/// Indices of unstable output neurons for sparse patches mode.
///
/// When set on `PatchesData`, the patches tensor has shape
/// `(unstable_size, in_c, kH, kW)` (4D) instead of the full
/// `(out_c, out_h, out_w, in_c, kH, kW)` (6D). Each entry `(c, h, w)`
/// maps sparse index `i` to output position `(c, h, w)` in the full grid.
///
/// Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` `Patches.unstable_idx`
/// Part of #2613 Phase 4 step 19
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnstableIdx {
    /// Channel indices of unstable output neurons.
    pub(crate) channels: Vec<usize>,
    /// Height indices of unstable output neurons.
    pub(crate) heights: Vec<usize>,
    /// Width indices of unstable output neurons.
    pub(crate) widths: Vec<usize>,
}

/// Conservative heap preflight for the temporary `HashSet` used to reject
/// duplicate sparse rows.
///
/// `HashSet::try_reserve` does not expose the allocation layout it will choose.
/// Budget at least three copies of every tuple payload to cover load-factor and
/// power-of-two capacity rounding, then separately account for control bytes
/// and one control group. Saturation turns arithmetic overflow into a refusal
/// at every configured budget below `usize::MAX`.
fn sparse_index_set_required_bytes(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }

    // hashbrown uses a minimum four-bucket table even for one requested entry.
    // Apply the conservative payload/control formula to at least that floor.
    let budgeted_entries = entries.max(4);
    let tuple_bytes = budgeted_entries.saturating_mul(size_of::<(usize, usize, usize)>());
    let tuple_capacity_bytes = tuple_bytes.saturating_mul(3);
    let control_bytes = budgeted_entries.saturating_mul(2).saturating_add(16);
    tuple_capacity_bytes.saturating_add(control_bytes)
}

fn enforce_sparse_index_set_budget(entries: usize, budget_bytes: usize) -> Result<usize> {
    let required_bytes = sparse_index_set_required_bytes(entries);
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "sparse patch index validation",
        });
    }
    Ok(required_bytes)
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
    /// lengths, a channel/height/width past the output extent, a duplicate
    /// output position, or a sparse bias whose length disagrees with the index
    /// count) is either an out-of-bounds panic or an ambiguous sparse relation.
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
        let mut deadline = PatchesMaterializationDeadline::new(None);
        self.validate_with_poll(out_c, out_h, out_w, sparse_bias_len, &mut deadline)
    }

    /// Cooperative form of [`Self::validate`] used by finite-deadline Patches
    /// materialization. Validation/error order is identical; only allocation
    /// boundaries and the sparse-index scan gain checks.
    pub(crate) fn validate_with_poll(
        &self,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        sparse_bias_len: Option<usize>,
        deadline: &mut PatchesMaterializationDeadline,
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
        // Duplicate sparse rows do not have one consistent meaning across
        // consumers: coefficient scatter adds them, while bias expansion and
        // native concretization overwrite the earlier row.  Reject them once
        // at the authenticated index boundary instead of letting those paths
        // silently compute different relations.  `try_reserve` keeps a bogus
        // attacker-controlled index count from turning validation into an OOM
        // abort.
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let required_bytes = enforce_sparse_index_set_budget(n, budget_bytes)?;
        let mut seen = HashSet::new();
        deadline.checkpoint("before sparse patch index validation allocation")?;
        seen.try_reserve(n)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "sparse patch index validation allocation",
            })?;
        deadline.checkpoint("after sparse patch index validation allocation")?;
        for i in 0..n {
            if self.channels[i] >= out_c || self.heights[i] >= out_h || self.widths[i] >= out_w {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_c, out_h, out_w],
                    got: vec![self.channels[i], self.heights[i], self.widths[i]],
                });
            }
            let position = (self.channels[i], self.heights[i], self.widths[i]);
            if !seen.insert(position) {
                return Err(NyError::InvalidSpec(format!(
                    "duplicate sparse patch index at position {position:?}"
                )));
            }
            deadline.work(1, "during sparse patch index validation")?;
        }
        deadline.checkpoint("after sparse patch index validation")?;
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

/// Geometry used to map a logical patch tap back to an input image position.
///
/// `Affine` is the historical regular sliding-window grid. `Anchored` carries
/// one exact signed origin per output row and per output column.  The latter is
/// separable by construction, so a `H x W` output needs `H + W` origins rather
/// than `H x W` coordinate pairs.  Signed `i128` origins represent padding
/// directly and keep every integer exact, including coordinates above f32's
/// `2^24` exact-integer limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PatchGeometry {
    Affine(AffinePatchGeometry),
    /// Exact per-position origins produced by the admitted ConvTranspose
    /// virtual-identity composition (finite stride 1 or stride>1). Consumers
    /// that have not implemented this mapping refuse it through
    /// [`Self::require_affine`].
    Anchored(AnchoredPatchGeometry),
}

/// A regular sliding-window patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AffinePatchGeometry {
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
}

/// An irregular but separable patch grid.
///
/// Tap `(ki, kj)` of output position `(oh, ow)` addresses
/// `(row_origins[oh] + ki, column_origins[ow] + kj)`.  The vectors are
/// Arc-backed because lower/upper carriers and cloned bounds normally share one
/// geometry. Keeping the allocation as `Arc<Vec<_>>` is intentional: a planner
/// can reserve/fill the large vector fallibly and `Arc::new` then moves that
/// allocation instead of performing the hidden O(axis length) allocation/copy
/// required by `Vec<_> -> Arc<[_]>`.
#[derive(Clone, PartialEq, Eq)]
// `Arc<Vec<_>>` is a deliberate allocation-safety boundary: unlike
// `Vec<_> -> Arc<[_]>`, `Arc::new` moves the already fallibly reserved buffer
// and cannot trigger another O(axis length) allocation/copy.
#[allow(clippy::rc_buffer)]
pub(crate) struct AnchoredPatchGeometry {
    row_origins: Arc<Vec<i128>>,
    column_origins: Arc<Vec<i128>>,
}

impl std::fmt::Debug for AnchoredPatchGeometry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Diagnostics are deliberately bounded: proof metadata can contain
        // millions of origins, and formatting the full Arc slices on a refusal
        // must never become an accidental OOM path.
        formatter
            .debug_struct("AnchoredPatchGeometry")
            .field("row_count", &self.row_origins.len())
            .field("row_first", &self.row_origins.first())
            .field("row_last", &self.row_origins.last())
            .field("column_count", &self.column_origins.len())
            .field("column_first", &self.column_origins.first())
            .field("column_last", &self.column_origins.last())
            .finish()
    }
}

impl AffinePatchGeometry {
    /// Construct a checked affine geometry.
    pub(crate) fn new(
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Result<Self> {
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "patch geometry requires non-zero stride, got {stride:?}"
            )));
        }
        Ok(Self { stride, padding })
    }

    pub(crate) const fn stride(self) -> (usize, usize) {
        self.stride
    }

    pub(crate) const fn padding(self) -> (usize, usize, usize, usize) {
        self.padding
    }

    /// Compute the exact affine output extent using checked arithmetic.
    pub(crate) fn output_size(
        self,
        input: (usize, usize),
        kernel: (usize, usize),
    ) -> Result<(usize, usize)> {
        let (in_h, in_w) = input;
        let (kh, kw) = kernel;
        let (sh, sw) = self.stride;
        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;

        if kh == 0 || kw == 0 {
            return Err(NyError::InvalidSpec(format!(
                "patch geometry requires a non-empty kernel, got {kernel:?}"
            )));
        }
        let padded_h = in_h
            .checked_add(pad_top)
            .and_then(|value| value.checked_add(pad_bottom))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry vertical padding overflows usize".into())
            })?;
        let padded_w = in_w
            .checked_add(pad_left)
            .and_then(|value| value.checked_add(pad_right))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry horizontal padding overflows usize".into())
            })?;
        if padded_h < kh || padded_w < kw {
            return Err(NyError::ShapeMismatch {
                expected: vec![kh, kw],
                got: vec![padded_h, padded_w],
            });
        }

        let out_h = (padded_h - kh)
            .checked_div(sh)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry output height overflows usize".into())
            })?;
        let out_w = (padded_w - kw)
            .checked_div(sw)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry output width overflows usize".into())
            })?;
        Ok((out_h, out_w))
    }

    /// Map one patch tap to its exact zero-based input-flat index.
    ///
    /// `None` denotes a tap in the zero-padding region. The typed enum owns the
    /// one mapping implementation used by both geometry variants.
    #[cfg(test)]
    pub(crate) fn input_flat_index(
        self,
        output: (usize, usize),
        channel: usize,
        tap: (usize, usize),
        input_shape: (usize, usize, usize),
    ) -> Result<Option<usize>> {
        PatchGeometry::Affine(self).input_flat_index(output, channel, tap, input_shape)
    }
}

impl PatchGeometry {
    fn equals_with_poll(
        &self,
        other: &Self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<bool> {
        match (self, other) {
            (Self::Affine(left), Self::Affine(right)) => Ok(left == right),
            (Self::Anchored(left), Self::Anchored(right)) => {
                if left.row_origins.len() != right.row_origins.len() {
                    return Ok(false);
                }
                for (&left, &right) in left.row_origins.iter().zip(right.row_origins.iter()) {
                    if left != right {
                        return Ok(false);
                    }
                    deadline.work(1, "during anchored row-origin pair comparison")?;
                }
                if left.column_origins.len() != right.column_origins.len() {
                    return Ok(false);
                }
                for (&left, &right) in left.column_origins.iter().zip(right.column_origins.iter()) {
                    if left != right {
                        return Ok(false);
                    }
                    deadline.work(1, "during anchored column-origin pair comparison")?;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Construct an affine descriptor ergonomically inside `PatchesData`
    /// literals.  Validation is deliberately performed at the consumer
    /// boundary by [`PatchGeometry::validate_for`], so malformed test/external
    /// metadata is returned as a typed error rather than made unrepresentable.
    pub(crate) const fn affine(
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Self {
        Self::Affine(AffinePatchGeometry { stride, padding })
    }

    /// Construct checked separable origins for an exact anchored carrier.
    /// The ConvTranspose identity planner reserves and fills both vectors
    /// fallibly before moving them into this descriptor.
    pub(crate) fn anchored(row_origins: Vec<i128>, column_origins: Vec<i128>) -> Result<Self> {
        if row_origins.is_empty() || column_origins.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "anchored patch geometry requires non-empty axes, got {} rows and {} columns",
                row_origins.len(),
                column_origins.len()
            )));
        }
        Ok(Self::Anchored(AnchoredPatchGeometry {
            row_origins: Arc::new(row_origins),
            column_origins: Arc::new(column_origins),
        }))
    }

    /// Require the historical affine form at an operation which has not yet
    /// been generalized to per-position origins.
    pub(crate) fn require_affine(&self, consumer: &str) -> Result<AffinePatchGeometry> {
        match self {
            Self::Affine(geometry) => Ok(*geometry),
            Self::Anchored(_) => Err(NyError::UnsupportedConfiguration(format!(
                "{consumer}: anchored patch geometry is not supported"
            ))),
        }
    }

    /// Validate this mapping against the carrier and kernel it will index.
    pub(crate) fn validate_for(
        &self,
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        kernel: (usize, usize),
    ) -> Result<()> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        self.validate_for_with_poll(output_shape, input_shape, kernel, &mut deadline)
    }

    /// Cooperative form of [`Self::validate_for`] for deadline-bearing
    /// materializers. It preserves the same validation and error order.
    pub(crate) fn validate_for_with_poll(
        &self,
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        kernel: (usize, usize),
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<()> {
        let (_, out_h, out_w) = output_shape;
        let (_, in_h, in_w) = input_shape;
        let (kh, kw) = kernel;
        if kh == 0 || kw == 0 {
            return Err(NyError::InvalidSpec(format!(
                "patch geometry requires a non-empty kernel, got {kernel:?}"
            )));
        }

        match self {
            Self::Affine(geometry) => {
                let checked = AffinePatchGeometry::new(geometry.stride, geometry.padding)?;
                let actual = checked.output_size((in_h, in_w), kernel)?;
                if actual != (out_h, out_w) {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_h, out_w],
                        got: vec![actual.0, actual.1],
                    });
                }
            }
            Self::Anchored(geometry) => {
                if geometry.row_origins.len() != out_h || geometry.column_origins.len() != out_w {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_h, out_w],
                        got: vec![geometry.row_origins.len(), geometry.column_origins.len()],
                    });
                }
                let last_ki = i128::try_from(kh - 1).map_err(|_| {
                    NyError::InvalidSpec("anchored patch kernel height exceeds i128".into())
                })?;
                let last_kj = i128::try_from(kw - 1).map_err(|_| {
                    NyError::InvalidSpec("anchored patch kernel width exceeds i128".into())
                })?;
                for &origin in geometry.row_origins.iter() {
                    origin.checked_add(last_ki).ok_or_else(|| {
                        NyError::InvalidSpec(
                            "anchored patch row origin plus kernel overflows i128".into(),
                        )
                    })?;
                    deadline.work(1, "during anchored row-origin validation")?;
                }
                for &origin in geometry.column_origins.iter() {
                    origin.checked_add(last_kj).ok_or_else(|| {
                        NyError::InvalidSpec(
                            "anchored patch column origin plus kernel overflows i128".into(),
                        )
                    })?;
                    deadline.work(1, "during anchored column-origin validation")?;
                }
            }
        }
        deadline.checkpoint("after patch geometry validation")?;
        Ok(())
    }

    /// Exact signed input origin for output position `(oh, ow)`.
    pub(crate) fn origin(&self, output: (usize, usize)) -> Result<(i128, i128)> {
        let (oh, ow) = output;
        match self {
            Self::Affine(geometry) => {
                let oh = i128::try_from(oh)
                    .map_err(|_| NyError::InvalidSpec("patch output row exceeds i128".into()))?;
                let ow = i128::try_from(ow)
                    .map_err(|_| NyError::InvalidSpec("patch output column exceeds i128".into()))?;
                let sh = i128::try_from(geometry.stride.0)
                    .map_err(|_| NyError::InvalidSpec("patch row stride exceeds i128".into()))?;
                let sw = i128::try_from(geometry.stride.1)
                    .map_err(|_| NyError::InvalidSpec("patch column stride exceeds i128".into()))?;
                let pad_top = i128::try_from(geometry.padding.2)
                    .map_err(|_| NyError::InvalidSpec("patch top padding exceeds i128".into()))?;
                let pad_left = i128::try_from(geometry.padding.0)
                    .map_err(|_| NyError::InvalidSpec("patch left padding exceeds i128".into()))?;
                let row = oh
                    .checked_mul(sh)
                    .and_then(|value| value.checked_sub(pad_top))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("affine patch row origin overflows i128".into())
                    })?;
                let column = ow
                    .checked_mul(sw)
                    .and_then(|value| value.checked_sub(pad_left))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("affine patch column origin overflows i128".into())
                    })?;
                Ok((row, column))
            }
            Self::Anchored(geometry) => {
                let row = geometry.row_origins.get(oh).copied().ok_or_else(|| {
                    NyError::ShapeMismatch {
                        expected: vec![geometry.row_origins.len()],
                        got: vec![oh.saturating_add(1)],
                    }
                })?;
                let column = geometry.column_origins.get(ow).copied().ok_or_else(|| {
                    NyError::ShapeMismatch {
                        expected: vec![geometry.column_origins.len()],
                        got: vec![ow.saturating_add(1)],
                    }
                })?;
                Ok((row, column))
            }
        }
    }

    /// Map one patch tap to its exact zero-based input-flat index.
    /// `None` denotes a tap outside the input (zero padding).
    pub(crate) fn input_flat_index(
        &self,
        output: (usize, usize),
        channel: usize,
        tap: (usize, usize),
        input_shape: (usize, usize, usize),
    ) -> Result<Option<usize>> {
        let (in_c, in_h, in_w) = input_shape;
        if channel >= in_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c],
                got: vec![channel],
            });
        }
        let (origin_h, origin_w) = self.origin(output)?;
        let ki = i128::try_from(tap.0)
            .map_err(|_| NyError::InvalidSpec("patch row tap exceeds i128".into()))?;
        let kj = i128::try_from(tap.1)
            .map_err(|_| NyError::InvalidSpec("patch column tap exceeds i128".into()))?;
        let ih = origin_h.checked_add(ki).ok_or_else(|| {
            NyError::InvalidSpec("patch input row coordinate overflows i128".into())
        })?;
        let iw = origin_w.checked_add(kj).ok_or_else(|| {
            NyError::InvalidSpec("patch input column coordinate overflows i128".into())
        })?;
        let in_h_i128 = i128::try_from(in_h)
            .map_err(|_| NyError::InvalidSpec("patch input height exceeds i128".into()))?;
        let in_w_i128 = i128::try_from(in_w)
            .map_err(|_| NyError::InvalidSpec("patch input width exceeds i128".into()))?;
        if ih < 0 || iw < 0 || ih >= in_h_i128 || iw >= in_w_i128 {
            return Ok(None);
        }
        let ih = usize::try_from(ih)
            .map_err(|_| NyError::InvalidSpec("patch input row cannot fit usize".into()))?;
        let iw = usize::try_from(iw)
            .map_err(|_| NyError::InvalidSpec("patch input column cannot fit usize".into()))?;
        let channel_base = channel
            .checked_mul(in_h)
            .and_then(|value| value.checked_mul(in_w))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry channel offset overflows usize".into())
            })?;
        let row_offset = ih.checked_mul(in_w).ok_or_else(|| {
            NyError::InvalidSpec("patch geometry row offset overflows usize".into())
        })?;
        let flat = channel_base
            .checked_add(row_offset)
            .and_then(|value| value.checked_add(iw))
            .ok_or_else(|| {
                NyError::InvalidSpec("patch geometry input-flat index overflows usize".into())
            })?;
        Ok(Some(flat))
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Affine(_) => 0,
            Self::Anchored(geometry) => geometry
                .row_origins
                .capacity()
                .saturating_add(geometry.column_origins.capacity())
                .saturating_mul(size_of::<i128>()),
        }
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

    /// Exact receptive-field mapping. This is the single source of geometry;
    /// consumers must not carry parallel stride/padding metadata.
    pub(crate) geometry: PatchGeometry,

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
    /// Validate and return the typed mapping used by this carrier.
    // Stage-4 compatibility seam: keep the no-deadline validator as an exact
    // delegate while callers migrate to the request-local polling form.
    #[allow(dead_code)]
    pub(crate) fn validated_geometry(&self) -> Result<&PatchGeometry> {
        self.validated_geometry_for(self.effective_kernel_size())
    }

    /// Validate the typed mapping for an explicitly authenticated kernel.
    /// Scatter builders use this instead of inferring a virtual identity's
    /// 1x1 kernel when they are deliberately constructing another exact plan.
    // Explicit-kernel sibling retained for the same staged deadline migration.
    #[allow(dead_code)]
    pub(crate) fn validated_geometry_for(&self, kernel: (usize, usize)) -> Result<&PatchGeometry> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        self.validated_geometry_for_with_poll(kernel, &mut deadline)
    }

    pub(crate) fn validated_geometry_for_with_poll(
        &self,
        kernel: (usize, usize),
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<&PatchGeometry> {
        self.geometry.validate_for_with_poll(
            self.output_shape,
            self.input_shape,
            kernel,
            deadline,
        )?;
        Ok(&self.geometry)
    }

    /// Validate the geometry descriptor shared by a lower/upper pair.
    ///
    /// A virtual identity is a seed awaiting its first spatial operator, so its
    /// `input_shape` may already name that operator's input rather than the
    /// seed's output. It still must carry canonical unit affine metadata and no
    /// stored coefficients/error. Strict `A = I` emission performs the
    /// additional output/input equality check in `validate_identity_geometry`.
    fn validated_common_geometry_side_with_poll(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<&PatchGeometry> {
        if self.identity {
            if self.patches.is_some() {
                return Err(NyError::InvalidSpec(
                    "virtual identity must not carry a materialized patches tensor".into(),
                ));
            }
            if self.coeff_err.is_some() {
                return Err(NyError::InternalError(
                    "virtual identity cannot carry coefficient error".into(),
                ));
            }
            let affine = self
                .geometry
                .require_affine("virtual identity common geometry")?;
            if affine.stride() != (1, 1) || affine.padding() != (0, 0, 0, 0) {
                return Err(NyError::InvalidSpec(format!(
                    "virtual identity common geometry requires unit affine metadata, got stride {:?} padding {:?}",
                    affine.stride(),
                    affine.padding()
                )));
            }
            Ok(&self.geometry)
        } else {
            self.validated_geometry_for_with_poll(self.effective_kernel_size(), deadline)
        }
    }

    /// Require two coefficient carriers to describe exactly the same mapping.
    ///
    /// Several dual-side consumers reuse the lower-side map for upper
    /// coefficients. This method turns their former debug-only assumption into
    /// a production precondition over the single typed geometry source.
    pub(crate) fn validate_common_geometry(&self, other: &Self) -> Result<()> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        self.validate_common_geometry_with_poll(other, &mut deadline)
    }

    pub(crate) fn validate_common_geometry_with_poll(
        &self,
        other: &Self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<()> {
        let left = self.validated_common_geometry_side_with_poll(deadline)?;
        let right = other.validated_common_geometry_side_with_poll(deadline)?;
        if !left.equals_with_poll(right, deadline)?
            || self.identity != other.identity
            || self.input_shape != other.input_shape
            || self.output_shape != other.output_shape
        {
            return Err(NyError::InvalidSpec(format!(
                "patch geometry mismatch: left={left:?} input={:?} output={:?}, right={right:?} input={:?} output={:?}",
                self.input_shape,
                self.output_shape,
                other.input_shape,
                other.output_shape
            )));
        }
        deadline.checkpoint("after common patch geometry validation")?;
        Ok(())
    }

    /// Authenticate metadata for a virtual identity before any consumer emits
    /// `A = I` without consulting an unfold plan.
    ///
    /// An anchored or non-unit affine descriptor cannot be silently ignored:
    /// its taps denote different input columns. Closed-route callers receive a
    /// typed refusal and can fall back to Dense.
    pub(crate) fn validate_identity_geometry(&self) -> Result<()> {
        if !self.identity || self.patches.is_some() {
            return Err(NyError::InvalidSpec(
                "virtual identity validation requires identity=true and no patches tensor".into(),
            ));
        }
        let affine = self
            .geometry
            .require_affine("virtual identity patch emission")?;
        if affine.stride() != (1, 1) || affine.padding() != (0, 0, 0, 0) {
            return Err(NyError::InvalidSpec(format!(
                "virtual identity requires unit affine geometry, got stride {:?} padding {:?}",
                affine.stride(),
                affine.padding()
            )));
        }
        if self.output_shape != self.input_shape {
            return Err(NyError::ShapeMismatch {
                expected: vec![
                    self.output_shape.0,
                    self.output_shape.1,
                    self.output_shape.2,
                ],
                got: vec![self.input_shape.0, self.input_shape.1, self.input_shape.2],
            });
        }
        self.geometry
            .validate_for(self.output_shape, self.input_shape, (1, 1))
    }

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
    #[cfg(test)]
    pub(crate) fn materialize_identity(&self) -> PatchesData {
        self.try_materialize_identity()
            .expect("test identity fixture must satisfy checked geometry and memory contracts")
    }

    /// Fallible identity materialization for proof-boundary callers.
    ///
    /// Anchored identity is intentionally not inferred: unless every anchor is
    /// authenticated as the ordinary identity grid, writing a diagonal 1 into
    /// an anchored carrier changes which input column it denotes.  The closed
    /// route therefore returns `UnsupportedConfiguration`. Allocation uses a
    /// checked shape, the CROWN memory budget, and `try_reserve_exact` before
    /// constructing ndarray storage.
    pub(crate) fn try_materialize_identity(&self) -> Result<PatchesData> {
        self.validate_identity_geometry()?;
        if self.coeff_err.is_some() {
            return Err(NyError::InternalError(
                "identity materialization cannot carry coefficient error".into(),
            ));
        }
        let (out_c, out_h, out_w) = self.output_shape;
        let (in_c, _in_h, _in_w) = self.input_shape;

        if let Some(idx) = &self.unstable_idx {
            idx.validate(out_c, out_h, out_w, None)?;
            let n = idx.len();
            let shape = [n, in_c, 1, 1];
            let len = checked_shape_product(&shape).ok_or_else(|| {
                NyError::InvalidSpec("sparse identity patch shape overflows usize".into())
            })?;
            let bytes = len.saturating_mul(size_of::<f32>());
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            if bytes > budget {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes: bytes,
                    budget_bytes: budget,
                    site: "sparse patches identity materialization",
                });
            }
            let mut values = Vec::new();
            values
                .try_reserve_exact(len)
                .map_err(|_| NyError::CpuMemoryExceeded {
                    required_bytes: bytes,
                    budget_bytes: budget,
                    site: "sparse patches identity allocation",
                })?;
            values.resize(len, 0.0f32);
            let mut patches = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
                NyError::InternalError(format!(
                    "sparse identity patch shape construction failed: {error}"
                ))
            })?;
            for (i, &c) in idx.channels.iter().enumerate() {
                patches[[i, c, 0, 0]] = 1.0;
            }
            return Ok(PatchesData {
                patches: Some(patches),
                geometry: self.geometry.clone(),
                identity: false,
                output_shape: self.output_shape,
                input_shape: self.input_shape,
                unstable_idx: self.unstable_idx.clone(),
                coeff_err: None, // identity diagonal store is exact (0/1)
            });
        }

        let shape = [out_c, out_h, out_w, in_c, 1, 1];
        let len = checked_shape_product(&shape).ok_or_else(|| {
            NyError::InvalidSpec("dense identity patch shape overflows usize".into())
        })?;
        let bytes = len.saturating_mul(size_of::<f32>());
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if bytes > budget {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: budget,
                site: "patches identity materialization",
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: budget,
                site: "patches identity allocation",
            })?;
        values.resize(len, 0.0f32);
        let mut patches = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
            NyError::InternalError(format!(
                "dense identity patch shape construction failed: {error}"
            ))
        })?;
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    patches[[c, h, w, c, 0, 0]] = 1.0;
                }
            }
        }

        Ok(PatchesData {
            patches: Some(patches),
            geometry: self.geometry.clone(),
            identity: false,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            unstable_idx: None,
            coeff_err: None, // identity diagonal store is exact (0/1)
        })
    }

    /// Logical heap payload used by this coefficient carrier, including
    /// anchored axes. ndarray backing capacities are not observable through a
    /// shared array reference; materialization admission adds exact capacities
    /// only for Vecs allocated inside the current request and relies on the
    /// process-envelope headroom for pre-existing allocator slack.
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
        .saturating_add(self.geometry.heap_bytes())
    }
}

#[cfg(test)]
mod sparse_index_budget_tests {
    use super::{enforce_sparse_index_set_budget, sparse_index_set_required_bytes, UnstableIdx};
    use crate::bounds::patches::PatchesMaterializationDeadline;
    use ny_core::NyError;
    use std::mem::size_of;

    #[test]
    fn preflight_counts_three_tuple_payloads_and_control_storage() {
        assert_eq!(sparse_index_set_required_bytes(0), 0);
        assert_eq!(
            sparse_index_set_required_bytes(1),
            4 * size_of::<(usize, usize, usize)>() * 3 + 4 * 2 + 16
        );

        let entries = 7usize;
        let tuple_payload = entries * size_of::<(usize, usize, usize)>();
        let control_storage = entries * 2 + 16;
        assert_eq!(
            sparse_index_set_required_bytes(entries),
            tuple_payload * 3 + control_storage
        );
    }

    #[test]
    fn preflight_accepts_exact_budget_and_refuses_one_byte_less() {
        let entries = 7usize;
        let required_bytes = sparse_index_set_required_bytes(entries);
        assert_eq!(
            enforce_sparse_index_set_budget(entries, required_bytes).unwrap(),
            required_bytes
        );

        let budget_bytes = required_bytes - 1;
        match enforce_sparse_index_set_budget(entries, budget_bytes).unwrap_err() {
            NyError::CpuMemoryExceeded {
                required_bytes: got_required,
                budget_bytes: got_budget,
                site,
            } => {
                assert_eq!(got_required, required_bytes);
                assert_eq!(got_budget, budget_bytes);
                assert_eq!(site, "sparse patch index validation");
            }
            error => panic!("expected CpuMemoryExceeded, got {error:?}"),
        }
    }

    #[test]
    fn preflight_overflow_saturates_and_preserves_memory_error_taxonomy() {
        assert_eq!(sparse_index_set_required_bytes(usize::MAX), usize::MAX);

        let budget_bytes = usize::MAX - 1;
        assert!(matches!(
            enforce_sparse_index_set_budget(usize::MAX, budget_bytes),
            Err(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes: got_budget,
                site: "sparse patch index validation",
            }) if got_budget == budget_bytes
        ));
    }

    #[test]
    fn sparse_index_validation_polls_inside_the_authenticated_scan() {
        let idx = UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, 1],
            widths: vec![0, 0],
        };
        let mut deadline =
            PatchesMaterializationDeadline::forced_at("during sparse patch index validation");
        assert!(matches!(
            idx.validate_with_poll(1, 2, 1, None, &mut deadline),
            Err(NyError::DeadlineExceeded(_))
        ));
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::{AffinePatchGeometry, PatchGeometry, PatchesData};
    use crate::bounds::patches::PatchesMaterializationDeadline;
    use ny_core::NyError;

    fn data_with_geometry(
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> PatchesData {
        PatchesData {
            patches: None,
            geometry: PatchGeometry::affine(stride, padding),
            identity: true,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 1),
            unstable_idx: None,
            coeff_err: None,
        }
    }

    #[test]
    fn affine_geometry_checks_stride_padding_and_output_extent() {
        assert!(matches!(
            AffinePatchGeometry::new((0, 1), (0, 0, 0, 0)),
            Err(NyError::InvalidSpec(_))
        ));

        let geometry = AffinePatchGeometry::new((2, 3), (1, 2, 3, 4)).unwrap();
        assert_eq!(geometry.output_size((8, 9), (3, 4)).unwrap(), (7, 3));
        assert!(matches!(
            geometry.output_size((usize::MAX, 1), (1, 1)),
            Err(NyError::InvalidSpec(_))
        ));
    }

    #[test]
    fn affine_geometry_keeps_adjacent_indices_exact_above_f32_limit() {
        let geometry = AffinePatchGeometry::new((1, 1), (0, 0, 0, 0)).unwrap();
        let input_shape = (2, 4096, 4096);
        let first = geometry
            .input_flat_index((0, 0), 1, (0, 0), input_shape)
            .unwrap()
            .unwrap();
        let second = geometry
            .input_flat_index((0, 1), 1, (0, 0), input_shape)
            .unwrap()
            .unwrap();

        assert_eq!(first, 1usize << f32::MANTISSA_DIGITS);
        assert_eq!(second, first + 1);
        assert_eq!(first as f32, second as f32, "legacy f32 indices alias here");
    }

    #[test]
    fn common_geometry_is_a_checked_dual_side_precondition() {
        let left = data_with_geometry((1, 1), (0, 0, 0, 0));
        let same = left.clone();
        assert!(left.validate_common_geometry(&same).is_ok());

        let different_stride = data_with_geometry((2, 1), (0, 0, 0, 0));
        assert!(matches!(
            left.validate_common_geometry(&different_stride),
            Err(NyError::InvalidSpec(_))
        ));

        let mut different_input = left.clone();
        different_input.input_shape = (1, 2, 1);
        assert!(left.validate_common_geometry(&different_input).is_err());

        let mut different_anchors = left.clone();
        different_anchors.geometry = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        assert!(matches!(
            left.validate_common_geometry(&different_anchors),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }

    #[test]
    fn virtual_identity_may_name_next_layer_input_but_cannot_emit_that_as_i() {
        let mut lower = data_with_geometry((1, 1), (0, 0, 0, 0));
        lower.input_shape = (1, 2, 1);
        let upper = lower.clone();

        assert!(lower.validate_common_geometry(&upper).is_ok());
        assert!(matches!(
            lower.validate_identity_geometry(),
            Err(NyError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn anchored_geometry_checks_axes_overflow_and_affine_refusal() {
        assert!(matches!(
            PatchGeometry::anchored(Vec::new(), vec![0]),
            Err(NyError::InvalidSpec(_))
        ));

        let malformed = PatchGeometry::anchored(vec![-1, 2], vec![0]).unwrap();
        assert!(matches!(
            malformed.validate_for((1, 3, 1), (1, 4, 4), (2, 2)),
            Err(NyError::ShapeMismatch { .. })
        ));

        let overflow = PatchGeometry::anchored(vec![i128::MAX], vec![0]).unwrap();
        assert!(matches!(
            overflow.validate_for((1, 1, 1), (1, 4, 4), (2, 1)),
            Err(NyError::InvalidSpec(_))
        ));
        assert!(matches!(
            malformed.require_affine("sparse test"),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }

    #[test]
    fn anchored_geometry_validation_polls_inside_axis_scans() {
        let geometry = PatchGeometry::anchored(vec![0, 1], vec![0, 1]).unwrap();
        let mut deadline =
            PatchesMaterializationDeadline::forced_at("during anchored row-origin validation");
        assert!(matches!(
            geometry.validate_for_with_poll((1, 2, 2), (1, 3, 3), (2, 2), &mut deadline),
            Err(NyError::DeadlineExceeded(_))
        ));
    }

    #[test]
    fn anchored_geometry_maps_negative_and_large_origins_exactly() {
        let padded = PatchGeometry::anchored(vec![-1, 2], vec![1, -2]).unwrap();
        padded.validate_for((1, 2, 2), (1, 4, 4), (2, 3)).unwrap();
        assert_eq!(padded.origin((1, 0)).unwrap(), (2, 1));
        assert_eq!(
            padded
                .input_flat_index((0, 0), 0, (0, 0), (1, 4, 4))
                .unwrap(),
            None
        );
        assert_eq!(
            padded
                .input_flat_index((0, 0), 0, (1, 2), (1, 4, 4))
                .unwrap(),
            Some(3)
        );

        let first = 1usize << f32::MANTISSA_DIGITS;
        let large =
            PatchGeometry::anchored(vec![0], vec![first as i128, first as i128 + 1]).unwrap();
        large
            .validate_for((1, 1, 2), (1, 1, first + 2), (1, 1))
            .unwrap();
        let left = large
            .input_flat_index((0, 0), 0, (0, 0), (1, 1, first + 2))
            .unwrap()
            .unwrap();
        let right = large
            .input_flat_index((0, 1), 0, (0, 0), (1, 1, first + 2))
            .unwrap()
            .unwrap();
        assert_eq!(right, left + 1);
        assert_eq!(left as f32, right as f32, "legacy f32 indices alias here");
    }

    #[test]
    fn anchored_geometry_memory_is_accounted() {
        let mut data = data_with_geometry((1, 1), (0, 0, 0, 0));
        data.output_shape = (1, 2, 3);
        data.input_shape = (1, 3, 4);
        data.geometry = PatchGeometry::anchored(vec![-1, 1], vec![0, 1, 2]).unwrap();
        assert_eq!(data.memory_bytes(), 5 * size_of::<i128>());
    }

    #[test]
    fn identity_materialization_refuses_anchored_geometry_without_mutation() {
        let mut data = data_with_geometry((1, 1), (0, 0, 0, 0));
        data.geometry = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        assert!(matches!(
            data.try_materialize_identity(),
            Err(NyError::UnsupportedConfiguration(_))
        ));
        assert!(data.identity);
        assert!(data.patches.is_none());

        let mut nonunit = data_with_geometry((2, 1), (0, 0, 0, 0));
        assert!(matches!(
            nonunit.validate_identity_geometry(),
            Err(NyError::InvalidSpec(_))
        ));
        nonunit.geometry = PatchGeometry::affine((1, 1), (0, 0, 0, 0));
        nonunit.input_shape = (1, 1, 2);
        assert!(matches!(
            nonunit.validate_identity_geometry(),
            Err(NyError::ShapeMismatch { .. })
        ));
    }
}
