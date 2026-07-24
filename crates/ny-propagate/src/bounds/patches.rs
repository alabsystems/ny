// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode linear bounds for CNN-optimized CROWN backward propagation.
//!
//! Instead of materializing a full dense [out_dim x in_dim] A-matrix,
//! Patches stores receptive field coefficients per output position:
//! O(out_c * out_h * out_w * in_c * kH * kW) instead of
//! O((out_c * out_h * out_w)^2).
//!
//! Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` (Patches class)
//! Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::mem::size_of;

use super::LinearBounds;

mod crown_bounds;
mod merge;
mod scatter;
mod sparse_concretize;
mod to_dense;
mod types;

pub(crate) use crown_bounds::CrownBounds;
pub(crate) use types::{PatchesData, UnstableIdx};

#[cfg(test)]
mod tests;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    /// Per-thread patches→dense call-site recorder.
    ///
    /// This was previously a process-global `Mutex<Vec<String>>`, but cargo runs
    /// the test binary multi-threaded and `to_dense()` records on *every* call in
    /// `#[cfg(test)]` builds. Any other test that triggered a patches→dense
    /// conversion concurrently appended to the shared buffer and corrupted the
    /// reset→propagate→read window of a test that observes the recorder (#4138:
    /// passed in isolation, failed in the full parallel suite).
    ///
    /// Each test runs on its own thread and the CROWN propagation it exercises is
    /// synchronous on that thread, so a thread-local buffer captures exactly the
    /// observing test's own conversions and is immune to concurrent tests.
    static PATCHES_TO_DENSE_CALL_SITES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn reset_patches_to_dense_call_count() {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn patches_to_dense_call_sites() -> Vec<String> {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow().clone())
}

#[cfg(test)]
pub(crate) fn record_patches_to_dense_call_site(site: String) {
    PATCHES_TO_DENSE_CALL_SITES.with(|sites| sites.borrow_mut().push(site));
}

/// Patches-mode linear bounds for CROWN backward propagation.
///
/// Analogous to LinearBounds but with structured sparse A-matrices.
/// The bias vectors remain dense Array1<f32> since they're per-output-neuron.
#[derive(Debug, Clone)]
pub(crate) struct PatchesLinearBounds {
    /// Logical number of Dense rows represented by this Patches object.
    ///
    /// Legacy spatial-output Patches use one logical row per output position
    /// (`out_c * out_h * out_w`). Dense->Patches re-entry uses arbitrary spec
    /// rows over the same spatial output grid, so the row count must be tracked
    /// explicitly instead of being inferred from `output_shape`.
    pub(crate) row_count: usize,
    pub(crate) lower_a: PatchesData,
    pub(crate) lower_b: Array1<f32>,
    pub(crate) upper_a: PatchesData,
    pub(crate) upper_b: Array1<f32>,
}

impl PatchesLinearBounds {
    /// Create identity Patches bounds for starting CROWN backward.
    ///
    /// The A-matrices are identity (each output position maps to itself).
    /// This is the Patches equivalent of `LinearBounds::identity(out_dim)`.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    pub(crate) fn identity(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
    ) -> Self {
        let out_dim = output_shape.0 * output_shape.1 * output_shape.2;
        PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            lower_b: Array1::zeros(out_dim),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: None,
            },
            upper_b: Array1::zeros(out_dim),
        }
    }

    /// Create sparse identity Patches bounds tracking only unstable neurons.
    ///
    /// Like `identity()`, but only creates patches for the specified unstable
    /// output positions. The patches tensor is 4D `(unstable_size, in_c, 1, 1)`
    /// instead of 6D. Bias vectors have length `unstable_size`.
    ///
    /// This is the starting point for sparse CROWN backward: only computing
    /// bounds for neurons that are actually unstable (lower < 0 < upper).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py` `get_sparse_C` + `Patches(identity=1, unstable_idx=...)`
    /// Part of #2613 Phase 4 step 19
    pub(crate) fn sparse_identity(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        unstable_idx: UnstableIdx,
    ) -> Self {
        let n = unstable_idx.len();
        let idx = Some(unstable_idx);
        PatchesLinearBounds {
            row_count: n,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: idx.clone(),
            },
            lower_b: Array1::zeros(n),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape,
                input_shape,
                unstable_idx: idx,
            },
            upper_b: Array1::zeros(n),
        }
    }

    /// Convert Dense rows over a known spatial output tensor into row-aware
    /// Patches coefficients with 1x1 receptive fields.
    pub(crate) fn from_dense_spatial_rows(
        bounds: &LinearBounds,
        output_shape: (usize, usize, usize),
    ) -> Result<Self> {
        let (out_c, out_h, out_w) = output_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
        })?;
        if bounds.num_inputs() != out_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![bounds.num_inputs()],
            });
        }

        let row_count = bounds.num_outputs();
        // #patches-row-range: the 7D re-entry PAIR materializes
        // `2 x row_count x out_c^2 x out_h x out_w` f32 cells — a factor of
        // `out_c` MORE than the dense pair it replaces (250 rows over VGG16's
        // 64x224x224 conv1 grid is a 411 GB pair, which aborted the process on
        // allocation). Refuse an over-budget (or overflowing) re-entry with
        // the structured `CpuMemoryExceeded`: the only production caller
        // (`try_dense_spatial_patches_reentry`) treats any Err as "skip
        // re-entry, stay Dense" — sound AND more precise (per-cell err carried
        // natively there; see the err-carry notes below).
        let required = checked_shape_product(&[row_count, out_c, out_h, out_w, out_c])
            .and_then(|cells| cells.checked_mul(2 * size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required > budget {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: budget,
                site: "patches from_dense_spatial_rows 7D re-entry",
            });
        }
        let mut lower_patches =
            ArrayD::<f32>::zeros(IxDyn(&[row_count, out_c, out_h, out_w, out_c, 1, 1]));
        let mut upper_patches =
            ArrayD::<f32>::zeros(IxDyn(&[row_count, out_c, out_h, out_w, out_c, 1, 1]));

        for row in 0..row_count {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let flat = oc * out_h * out_w + oh * out_w + ow;
                        lower_patches[[row, oc, oh, ow, oc, 0, 0]] = bounds.lower_a()[[row, flat]];
                        upper_patches[[row, oc, oh, ow, oc, 0, 0]] = bounds.upper_a()[[row, flat]];
                    }
                }
            }
        }

        // Carry the source dense per-cell coefficient error into the 7D
        // re-entry as a per-spec-row bound (#patches-coeff-err-soundness,
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §10). The copy loop above writes
        // ONLY diagonal entries `P[r,oc,oh,ow,oc,0,0] = s_a[[r, flat]]` (bitwise;
        // every other 7D entry is a structural zero). So for each spec row `r`,
        // the per-row max of the source row's per-cell err over-bounds every
        // stored coefficient's true deviation: a copied diagonal entry deviates
        // by `<= E_s[r, flat] <= rowmax`; a structural zero has deviation 0 <=
        // any nonnegative row err. `sanitize` maps non-finite/negative to `+INF`
        // (an outward degrade — `set_coeff_err` does NOT sanitize despite its
        // doc); a wrong-shaped `E_s` returns `Err(ShapeMismatch)` so the caller
        // skips re-entry and stays Dense (per-cell err carried natively there:
        // sound AND more precise; never silently zero-filled).
        let row_max_err = |err: Option<&Array2<f32>>| -> Result<Option<Array1<f32>>> {
            let Some(e) = err else {
                return Ok(None); // exact source (unchanged)
            };
            if e.shape() != [row_count, out_dim] {
                return Err(NyError::ShapeMismatch {
                    expected: vec![row_count, out_dim],
                    got: e.shape().to_vec(),
                });
            }
            let mut out = Array1::<f32>::zeros(row_count);
            for r in 0..row_count {
                let mut rowmax = 0.0f32;
                for j in 0..out_dim {
                    let v = e[[r, j]];
                    // sanitize: non-finite or negative => +INF (poison outward).
                    // `-0.0` is finite and `>= 0.0`, so it does NOT poison.
                    let s = if v.is_finite() && v >= 0.0 {
                        v
                    } else {
                        f32::INFINITY
                    };
                    if s > rowmax {
                        rowmax = s;
                    }
                }
                // H1 (spec §14): an all-zero sanitized row max stays exactly 0.0
                // (tighter than `next_up(0)`); a nonzero row takes one outward
                // ULP of doc-conformance slack. `+INF` is a `next_up` fixed point.
                out[r] = if rowmax == 0.0 {
                    0.0
                } else {
                    ny_tensor::next_up_f32(rowmax)
                };
            }
            Ok(Some(out))
        };
        let lower_coeff_err = row_max_err(bounds.lower_a_err())?;
        let upper_coeff_err = row_max_err(bounds.upper_a_err())?;

        Ok(PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: lower_coeff_err,
                patches: Some(lower_patches),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape,
                input_shape: output_shape,
                unstable_idx: None,
            },
            lower_b: bounds.lower_b().to_owned(),
            upper_a: PatchesData {
                coeff_err: upper_coeff_err,
                patches: Some(upper_patches),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape,
                input_shape: output_shape,
                unstable_idx: None,
            },
            upper_b: bounds.upper_b().to_owned(),
        })
    }

    fn validate_row_count(&self) -> Result<()> {
        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            return Ok(());
        }
        if self.lower_b.len() != self.row_count || self.upper_b.len() != self.row_count {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.row_count],
                got: vec![self.lower_b.len().max(self.upper_b.len())],
            });
        }
        Ok(())
    }

    /// Dense matrix shape that `to_dense()` would materialize.
    pub(crate) fn dense_pair_shape(&self) -> Result<(usize, usize)> {
        self.validate_row_count()?;
        if self.lower_a.output_shape != self.upper_a.output_shape
            || self.lower_a.input_shape != self.upper_a.input_shape
        {
            return Err(NyError::InternalError(
                "PatchesLinearBounds: lower/upper spatial shapes differ".into(),
            ));
        }
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec("PatchesLinearBounds input shape overflow".into())
        })?;
        let legacy_sparse_rows =
            self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some();
        // Identity bounds still require the historical one-row-per-output-position
        // invariant. Explicit spec rows are represented with materialized patches.
        if self.lower_a.identity || self.upper_a.identity {
            let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
                NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
            })?;
            if self.row_count != out_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_dim],
                    got: vec![self.row_count],
                });
            }
        }
        if legacy_sparse_rows {
            let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
                NyError::InvalidSpec("PatchesLinearBounds output shape overflow".into())
            })?;
            Ok((out_dim, in_dim))
        } else {
            Ok((self.row_count, in_dim))
        }
    }

    /// Heap bytes needed for the Dense lower/upper coefficient pair.
    pub(crate) fn dense_pair_bytes(&self) -> Result<usize> {
        let (rows, cols) = self.dense_pair_shape()?;
        Ok(crate::network::crown_memory::dense_pair_bytes(rows, cols).unwrap_or(usize::MAX))
    }

    /// Filter dense patches to only keep unstable output neurons (sparse mode).
    ///
    /// Given a boolean mask of unstable neurons over the full `(out_c, out_h, out_w)`
    /// grid, extracts only the patches and biases for unstable positions.
    ///
    /// Returns `None` if all neurons are unstable (no benefit from sparse mode).
    /// Returns `None` if fewer than `(1.0 - min_sparsity) * total` neurons are unstable
    /// (the default `min_sparsity` of 0.9 means at least 10% must be stable).
    ///
    /// **Precondition:** `self` must be in dense mode (no existing `unstable_idx`).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py` `get_sparse_C`, `minimum_sparsity=0.9`
    /// Part of #2613 Phase 4 step 19 — currently test-only; remove #[cfg(test)] when
    /// wiring to CROWN backward engine.
    #[cfg(test)]
    pub(crate) fn filter_to_unstable(
        &self,
        unstable_mask: &ndarray::Array3<bool>,
        min_sparsity: f32,
    ) -> Option<PatchesLinearBounds> {
        debug_assert!(
            self.lower_a.unstable_idx.is_none(),
            "filter_to_unstable called on already-sparse patches"
        );
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let total = checked_shape_product(&[out_c, out_h, out_w])?;
        if self.row_count != total {
            return None;
        }

        // Collect unstable positions
        let mut channels = Vec::new();
        let mut heights = Vec::new();
        let mut widths = Vec::new();
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    if unstable_mask[[c, h, w]] {
                        channels.push(c);
                        heights.push(h);
                        widths.push(w);
                    }
                }
            }
        }
        let unstable_size = channels.len();

        // Check sparsity threshold
        if unstable_size >= total || (unstable_size as f32) > min_sparsity * (total as f32) {
            return None;
        }
        if unstable_size == 0 {
            // All stable — no backward needed. Return empty sparse patches.
            let idx = UnstableIdx {
                channels: vec![],
                heights: vec![],
                widths: vec![],
            };
            return Some(PatchesLinearBounds::sparse_identity(
                self.lower_a.output_shape,
                self.lower_a.input_shape,
                idx,
            ));
        }

        let idx = UnstableIdx {
            channels,
            heights,
            widths,
        };

        // Extract sparse patches from dense 6D tensor
        let lower_a = Self::extract_sparse_patches(&self.lower_a, &idx)?;
        let upper_a = Self::extract_sparse_patches(&self.upper_a, &idx)?;

        // Extract sparse bias vectors
        let mut lower_b = Array1::zeros(unstable_size);
        let mut upper_b = Array1::zeros(unstable_size);
        for (i, flat) in idx
            .channels
            .iter()
            .zip(idx.heights.iter())
            .zip(idx.widths.iter())
            .map(|((c, h), w)| c * out_h * out_w + h * out_w + w)
            .enumerate()
        {
            lower_b[i] = self.lower_b[flat];
            upper_b[i] = self.upper_b[flat];
        }

        Some(PatchesLinearBounds {
            row_count: unstable_size,
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        })
    }

    /// Extract sparse patches from a single PatchesData, keeping only unstable positions.
    #[cfg(test)]
    fn extract_sparse_patches(data: &PatchesData, idx: &UnstableIdx) -> Option<PatchesData> {
        let unstable_size = idx.len();
        let patches = match &data.patches {
            None => {
                // Identity: no tensor to extract from
                return Some(PatchesData {
                    coeff_err: None,
                    patches: None,
                    stride: data.stride,
                    padding: data.padding,
                    identity: true,
                    output_shape: data.output_shape,
                    input_shape: data.input_shape,
                    unstable_idx: Some(idx.clone()),
                });
            }
            Some(p) => p,
        };
        let shape = patches.shape();
        let in_c = shape[3];
        let kh = shape[4];
        let kw = shape[5];

        // Sparse patches: (unstable_size, in_c, kH, kW) — 4D
        let mut sparse = ArrayD::zeros(IxDyn(&[unstable_size, in_c, kh, kw]));
        for (i, ((&c, &h), &w)) in idx
            .channels
            .iter()
            .zip(idx.heights.iter())
            .zip(idx.widths.iter())
            .enumerate()
        {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        sparse[[i, ic, ki, kj]] = patches[[c, h, w, ic, ki, kj]];
                    }
                }
            }
        }

        Some(PatchesData {
            coeff_err: None,
            patches: Some(sparse),
            stride: data.stride,
            padding: data.padding,
            identity: false,
            output_shape: data.output_shape,
            input_shape: data.input_shape,
            unstable_idx: Some(idx.clone()),
        })
    }

    /// Total heap memory used by this Patches bounds struct, in bytes.
    ///
    /// Includes both A-matrices and bias vectors. Returns 0 for identity
    /// A-matrices (virtual, no allocation) plus bias bytes.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.lower_a.memory_bytes()
            + self.lower_b.len() * size_of::<f32>()
            + self.upper_a.memory_bytes()
            + self.upper_b.len() * size_of::<f32>()
    }
}
