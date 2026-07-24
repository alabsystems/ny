// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Operations on Patches-mode tensors for CNN-optimized CROWN backward.
//!
//! Shared helpers for AvgPool and MaxPool PatchesPropagation.
//!
//! Reference: designs/2026-03-01-patches-phase3-pooling-termination.md
//! Part of #2613

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};

/// Nearest-neighbor upsample the last two dims of a dense patches array.
///
/// 6D dense layout `(out_c, out_h, out_w, in_c, kH, kW)` or 7D explicit-rows
/// layout `(rows, out_c, out_h, out_w, in_c, kH, kW)`; the output keeps every
/// leading axis and scales only the trailing kernel axes to
/// `(kH * scale_h, kW * scale_w)`.
///
/// Each element `patches[.., ic, ki, kj]` is replicated to fill a
/// `scale_h × scale_w` block at position `(ki*scale_h + di, kj*scale_w + dj)`.
/// Pure replication — no arithmetic, so it is EXACT (a copied coefficient
/// inherits any carried per-row deviation verbatim;
/// docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9.2).
///
/// Shared between AvgPool and MaxPool PatchesPropagation. Does NOT divide
/// by pool_size — caller handles that (AvgPool divides, MaxPool does not).
///
/// Returns [`NyError::ShapeMismatch`] when `patches` is not 6D or 7D. The
/// dispatch is deliberately NOT trailing-axes-agnostic (spec §14 F2): a 4D/5D
/// sparse layout must keep failing HERE, early, so pooling
/// PatchesPropagation rejects it cleanly instead of panicking (caller falls
/// back to the sound dense path). No bound math changes.
///
/// Reference: designs/2026-03-01-patches-phase3-pooling-termination.md Section 1
pub(crate) fn nearest_neighbor_upsample_last2(
    patches: &ArrayD<f32>,
    scale_h: usize,
    scale_w: usize,
) -> Result<ArrayD<f32>> {
    let shape = patches.shape();
    match shape.len() {
        6 => {
            let (oc, oh, ow, ic, kh, kw) =
                (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
            let new_kh = kh * scale_h;
            let new_kw = kw * scale_w;
            let mut result = ArrayD::zeros(IxDyn(&[oc, oh, ow, ic, new_kh, new_kw]));

            for o_c in 0..oc {
                for o_h in 0..oh {
                    for o_w in 0..ow {
                        for i_c in 0..ic {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let val = patches[[o_c, o_h, o_w, i_c, ki, kj]];
                                    for di in 0..scale_h {
                                        for dj in 0..scale_w {
                                            result[[
                                                o_c,
                                                o_h,
                                                o_w,
                                                i_c,
                                                ki * scale_h + di,
                                                kj * scale_w + dj,
                                            ]] = val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(result)
        }
        7 => {
            // 7D explicit-rows: identical replication with the spec-row axis
            // (axis 0) carried through untouched.
            let (rows, oc, oh, ow, ic, kh, kw) = (
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
            );
            let new_kh = kh * scale_h;
            let new_kw = kw * scale_w;
            let mut result = ArrayD::zeros(IxDyn(&[rows, oc, oh, ow, ic, new_kh, new_kw]));

            for row in 0..rows {
                for o_c in 0..oc {
                    for o_h in 0..oh {
                        for o_w in 0..ow {
                            for i_c in 0..ic {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let val = patches[[row, o_c, o_h, o_w, i_c, ki, kj]];
                                        for di in 0..scale_h {
                                            for dj in 0..scale_w {
                                                result[[
                                                    row,
                                                    o_c,
                                                    o_h,
                                                    o_w,
                                                    i_c,
                                                    ki * scale_h + di,
                                                    kj * scale_w + dj,
                                                ]] = val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(result)
        }
        _ => Err(NyError::ShapeMismatch {
            expected: vec![6, 7],
            got: vec![shape.len()],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Crash guard: a non-{6,7}D input must return a clean ShapeMismatch instead
    /// of panicking. The ndim invariant was previously only a `debug_assert`,
    /// which is compiled out in release and would let the unrolled shape access /
    /// fixed-arity indexing run out of bounds and panic. Pooling
    /// PatchesPropagation propagates this error and falls back to the sound
    /// dense path. The scope is deliberately {6,7} only (spec §14 F2): 4D/5D
    /// sparse layouts must keep failing here, early.
    #[test]
    fn nearest_neighbor_upsample_rejects_non_6d_7d_input() {
        // 4D tensor (e.g. a sparse patches layout) where a dense tensor is required.
        let patches = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 1, 1]));
        let result = nearest_neighbor_upsample_last2(&patches, 2, 2);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for 4D input, got {result:?}"
        );
        // 5D tensor (e.g. a sparse explicit-rows layout) must also fail early.
        let patches = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 1, 1, 1]));
        let result = nearest_neighbor_upsample_last2(&patches, 2, 2);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for 5D input, got {result:?}"
        );
    }

    /// Soundness/regression: a valid 6D input still upsamples correctly and
    /// replicates each element into its `scale_h x scale_w` block.
    #[test]
    fn nearest_neighbor_upsample_6d_valid_input_replicates_block() {
        // (oc, oh, ow, ic, kh, kw) = (1, 1, 1, 1, 1, 1) with a single value.
        let mut patches = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1, 1, 1]));
        patches[[0, 0, 0, 0, 0, 0]] = 2.5;
        let result =
            nearest_neighbor_upsample_last2(&patches, 2, 3).expect("valid 6D input must succeed");
        assert_eq!(result.shape(), &[1, 1, 1, 1, 2, 3]);
        for ki in 0..2 {
            for kj in 0..3 {
                assert_eq!(result[[0, 0, 0, 0, ki, kj]], 2.5);
            }
        }
    }

    /// 7D explicit-rows: each element replicates into its block with the
    /// spec-row axis carried through (per-row values stay per-row).
    #[test]
    fn nearest_neighbor_upsample_7d_replicates_block_per_row() {
        // (rows, oc, oh, ow, ic, kh, kw) = (2, 1, 1, 2, 1, 1, 1); distinct
        // values per (row, ow) so a row/position mixup would be caught.
        let mut patches = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 1, 2, 1, 1, 1]));
        patches[[0, 0, 0, 0, 0, 0, 0]] = 1.5;
        patches[[0, 0, 0, 1, 0, 0, 0]] = -2.25;
        patches[[1, 0, 0, 0, 0, 0, 0]] = 3.75;
        patches[[1, 0, 0, 1, 0, 0, 0]] = -0.5;
        let result =
            nearest_neighbor_upsample_last2(&patches, 2, 3).expect("valid 7D input must succeed");
        assert_eq!(result.shape(), &[2, 1, 1, 2, 1, 2, 3]);
        for row in 0..2 {
            for ow in 0..2 {
                let expected = patches[[row, 0, 0, ow, 0, 0, 0]];
                for ki in 0..2 {
                    for kj in 0..3 {
                        assert_eq!(
                            result[[row, 0, 0, ow, 0, ki, kj]],
                            expected,
                            "row {row} ow {ow} tap ({ki},{kj})"
                        );
                    }
                }
            }
        }
    }
}
