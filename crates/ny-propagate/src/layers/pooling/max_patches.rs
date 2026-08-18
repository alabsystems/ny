// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward for MaxPool2d.
//!
//! MaxPool is nonlinear: y = max(x_1, ..., x_k). The CROWN relaxation uses:
//! - Definite winner (l_i >= max_{j!=i}(u_j)): gradient flows through winner
//! - No winner: constant IBP-style bounds (max_lower / max_upper), no gradient
//!
//! The Patches backward:
//! 1. Computes per-position definite-winner slope and per-output constant bounds
//! 2. Accumulates bias from non-linear positions (constant bounds)
//! 3. Upsamples patches by pool kernel (nearest-neighbor)
//! 4. Applies winner slopes with positive/negative coefficient dispatch
//!
//! Reference: alpha-beta-CROWN auto_LiRPA/operators/pooling.py:78-337
//! Reference: designs/2026-03-01-patches-phase3-pooling-termination.md Section 2
//! Part of #2613

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::nan_propagating_max;
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::bounds::patches_ops::nearest_neighbor_upsample_last2;
use crate::layers::common::PatchesPropagation;

use super::max::MaxPool2dLayer;

impl PatchesPropagation for MaxPool2dLayer {
    fn propagate_patches(&self, _bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        Err(NyError::UnsupportedOp(
            "MaxPool2d Patches requires pre-activation bounds - use propagate_patches_with_bounds"
                .to_string(),
        ))
    }

    fn propagate_patches_with_bounds(
        &self,
        bounds: &PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<CrownBounds> {
        // Guard: stride == kernel_size required
        // Reference: alpha-beta-CROWN pooling.py:83-84
        if self.kernel_size != self.stride {
            return Err(NyError::UnsupportedOp(
                "MaxPool Patches requires kernel_size == stride".into(),
            ));
        }
        // This legacy pool kernel is affine-only. Reject Anchored in O(1)
        // before common validation scans its origin axes.
        let affine_geometry = bounds
            .lower_a
            .geometry
            .require_affine("MaxPool Patches backward")?;
        bounds
            .upper_a
            .geometry
            .require_affine("MaxPool Patches backward")?;
        bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;

        let (pool_kh, pool_kw) = self.kernel_size;
        let (pool_sh, pool_sw) = self.stride;
        let (pool_ph, pool_pw) = self.padding;
        let pool_size = pool_kh * pool_kw;

        // Extract MaxPool input shape from pre_activation
        let input_shape_arr = pre_activation.shape();
        let ndim = input_shape_arr.len();
        let (channels, in_h, in_w) = if ndim == 3 {
            (input_shape_arr[0], input_shape_arr[1], input_shape_arr[2])
        } else if ndim == 4 {
            (input_shape_arr[1], input_shape_arr[2], input_shape_arr[3])
        } else {
            return Err(NyError::InvalidSpec(format!(
                "MaxPool2d Patches requires 3D or 4D input, got {}D",
                ndim
            )));
        };

        let (out_h, out_w) = self.output_size(in_h, in_w)?;

        // Step 1: Compute per-position relaxation.
        // winner_d: (channels, in_h, in_w) -- 1 at definite-winner position, 0 elsewhere.
        //   Used for BOTH lower and upper bounds (matching Dense path behavior).
        // For non-linear positions (no definite winner), use constant IBP bounds:
        //   lower_b_per_pos: max(l_j)  -- used for lower bound positive coefficients
        //   upper_b_per_pos: max(u_j)  -- used for upper bound positive coefficients
        let mut winner_d = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
        let mut lower_b_per_pos = ArrayD::<f32>::zeros(IxDyn(&[channels, out_h, out_w]));
        let mut upper_b_per_pos = ArrayD::<f32>::zeros(IxDyn(&[channels, out_h, out_w]));

        let pre_lower = pre_activation.lower();
        let pre_upper = pre_activation.upper();

        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih_start = oh * pool_sh;
                    let iw_start = ow * pool_sw;

                    // Collect valid inputs in this pooling window
                    let mut window: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(pool_size);

                    for kh_off in 0..pool_kh {
                        for kw_off in 0..pool_kw {
                            let ih = (ih_start + kh_off) as isize - pool_ph as isize;
                            let iw = (iw_start + kw_off) as isize - pool_pw as isize;
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;
                                let l = if ndim == 3 {
                                    pre_lower[[c, ih, iw]]
                                } else {
                                    pre_lower[[0, c, ih, iw]]
                                };
                                let u = if ndim == 3 {
                                    pre_upper[[c, ih, iw]]
                                } else {
                                    pre_upper[[0, c, ih, iw]]
                                };
                                window.push((ih, iw, l, u));
                            }
                        }
                    }

                    if window.is_empty() {
                        // All positions are padding: the output is max over an empty
                        // set (-inf), and the (0,0) init below would masquerade as
                        // the definite-winner sentinel, folding no constant at all.
                        // output_size() already rejects the padding >= kernel
                        // geometry that creates such windows; refuse here too.
                        return Err(NyError::InvalidSpec(format!(
                            "MaxPool2d Patches: pooling window at output ({oh},{ow}) \
                             contains no input positions: kernel=({pool_kh},{pool_kw}), \
                             padding=({pool_ph},{pool_pw})"
                        )));
                    }

                    // NaN-propagating max of lower and upper bounds
                    let max_lower = window
                        .iter()
                        .map(|&(_, _, l, _)| l)
                        .fold(f32::NEG_INFINITY, nan_propagating_max);
                    let max_upper = window
                        .iter()
                        .map(|&(_, _, _, u)| u)
                        .fold(f32::NEG_INFINITY, nan_propagating_max);

                    // Definite winner check: l_i >= all other u_j
                    let definite_winner = window.iter().enumerate().find(|&(i, &(_, _, l, _))| {
                        window
                            .iter()
                            .enumerate()
                            .all(|(j, &(_, _, _, u))| i == j || l >= u)
                    });

                    if let Some((_, &(dw_ih, dw_iw, _, _))) = definite_winner {
                        // Gradient flows through winner for BOTH lower and upper bounds
                        winner_d[[c, dw_ih, dw_iw]] = 1.0;
                    } else {
                        // Non-linear: use constant IBP-style bounds (no gradient)
                        lower_b_per_pos[[c, oh, ow]] = max_lower;
                        upper_b_per_pos[[c, oh, ow]] = max_upper;
                    }
                }
            }
        }

        // Spec-row count for the 7D explicit-rows layout (axis 0 must match it;
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9).
        let spec_rows = bounds.row_count;

        // Steps 2-5: Process lower_a and upper_a separately.
        // Both use winner_d as the slope (only definite-winner positions get gradient).
        // For non-linear positions, bias absorbs the constant bound:
        //   Lower bound: pos * max_lower + neg * max_upper
        //   Upper bound: pos * max_upper + neg * max_lower
        let process_patches = |patches_data: &PatchesData,
                               bias_vec: &ndarray::Array1<f32>,
                               is_lower: bool|
         -> Result<(PatchesData, ndarray::Array1<f32>)> {
            let (in_sh, in_sw) = affine_geometry.stride();
            let (in_pad_left, in_pad_right, in_pad_top, in_pad_bottom) = affine_geometry.padding();
            let materialized = if patches_data.identity {
                patches_data.try_materialize_identity()?
            } else {
                patches_data.clone()
            };
            let patches_tensor = materialized.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;

            let shape = patches_tensor.shape();
            // 6D dense vs 7D explicit-rows dispatch
            // (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9). Also fixes two LIVE
            // panics (spec R7): a 7D tensor was previously destructured as 6D
            // and panicked on the 6-index tap read below; a sparse 4D tensor
            // panicked on the `shape[4]` access. Both now return a clean error
            // so the caller falls back to the sound dense MaxPool backward.
            let explicit_rows = match shape.len() {
                6 => false,
                7 => {
                    if shape[0] != spec_rows {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![spec_rows],
                            got: vec![shape[0]],
                        });
                    }
                    true
                }
                _ => {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![6, 7],
                        got: vec![shape.len()],
                    });
                }
            };
            let (oc, oh_p, ow_p, ic, kh, kw) = if explicit_rows {
                (shape[1], shape[2], shape[3], shape[4], shape[5], shape[6])
            } else {
                (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5])
            };

            // Step 2: Compute bias from constant-bound (non-linear) positions.
            // For each output spec position, sum A-coefficient contributions
            // weighted by the constant MaxPool bound OF THE SPECIFIC OUTPUT TAP
            // that each coefficient multiplies.
            //
            // SOUNDNESS (#maxpool-patches-lumped-bias): the patches coefficient
            // `patches_tensor[[o_c, o_h, o_w, i_c, ki, kj]]` multiplies the MaxPool
            // OUTPUT element at channel `i_c`, spatial
            //   (ih = o_h*sh + ki - pad_top,  iw = o_w*sw + kj - pad_left)
            // — the SAME geometry the winner loop (Steps 4-5) uses pre-upsample.
            // For a non-linear (no-definite-winner) MaxPool output that element is a
            // CONSTANT bounded by [lower_b_per_pos[i_c,ih,iw], upper_b_per_pos[..]].
            // The OLD code lumped every tap's coefficient into a single pos/neg sum
            // and multiplied by the bound at the SPEC index [o_c,o_h,o_w] — wrong
            // dimensions entirely (spec position vs. MaxPool output position) and a
            // single bound for taps that map to DIFFERENT MaxPool outputs with
            // DIFFERENT bounds. That under/over-counts the constant contribution →
            // a bias that can leave the concretized bound on the wrong side of the
            // true value = FALSE PROOF. Fix: per-tap lookup of the tap's own MaxPool
            // output bound, with the sign-correct interval-arithmetic rule.
            let mut new_bias = bias_vec.mapv(|x| x as f64);

            // Certified coefficient error (#patches-coeff-err-soundness,
            // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9 M1). MaxPool's coefficient
            // transform is EXACT: winner routing multiplies each surviving coeff by
            // winner_d ∈ {0,1} (exact in f32) and the upsample is a copy, so the A
            // coefficients carry the incoming per-row error UNCHANGED (gain 1):
            // new_err[i] = old_err[i], None→None — on BOTH the 6D dense layout
            // (logical row = output position) and the 7D explicit-rows layout
            // (logical row = SPEC row, axis 0). Non-winner taps fold their coeff into
            // the bias and are discharged in the loop below. Sparse (unstable_idx
            // Some) stays None (out of scope, spec I2); its 4D/5D tensor already
            // hard-errored at the ndim dispatch above.
            let coeff_err_ok = patches_data.unstable_idx.is_none();
            let old_err = if coeff_err_ok {
                patches_data.coeff_err.as_ref()
            } else {
                None
            };
            // Hard guard (spec I6, 7D arm ONLY — the 6D arm keeps its silent
            // `.get().unwrap_or(0.0)` read for byte-identity, spec §14 B5): a
            // carried err whose length is not the spec-row count returns Err ⇒
            // the caller's sound dense fallback, never a silent under-count
            // (the false-proof direction).
            if explicit_rows {
                if let Some(e) = old_err {
                    if e.len() != spec_rows {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![spec_rows],
                            got: vec![e.len()],
                        });
                    }
                }
            }

            // Crash guard (mirrors BatchNorm/Conv2d patches-bias fix): the loops below
            // index new_bias by 0..oc*oh_p*ow_p (6D: one slot per output position) or
            // by 0..spec_rows (7D explicit-rows: one slot per SPEC row). Under
            // disjunctive multi-clause input-split the incoming bias is spec-row-shaped,
            // not oc*oh_p*ow_p — a mismatched vector would index out of bounds (SIGABRT
            // under panic=abort). Require the exact per-layout length; else return
            // ShapeMismatch so the caller falls back to the sound dense MaxPool backward.
            if explicit_rows {
                if new_bias.len() != spec_rows {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![spec_rows],
                        got: vec![new_bias.len()],
                    });
                }
            } else {
                let out_positions = oc
                    .checked_mul(oh_p)
                    .and_then(|x| x.checked_mul(ow_p))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(
                            "MaxPool patches: output position count overflow".into(),
                        )
                    })?;
                if new_bias.len() != out_positions {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_positions],
                        got: vec![new_bias.len()],
                    });
                }
            }

            if explicit_rows {
                // 7D explicit-rows bias fold (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md
                // §9 M2/M3): every position (oc,oh,ow) of spec row `row` folds into
                // the ONE bias slot new_bias[row] (SUM-lift, spec I1). The tap
                // geometry is row/oc-independent — identical to the 6D arm. Loop
                // nest mirrors the BatchNorm 7D fold; fully serial in the fixed
                // tap order row -> oc -> oh -> ow -> (ic, ki, kj) (spec I8).
                //
                // spec §14 E3/F3: the f64 fold rounding is dominated by the
                // directed casts only while the per-row addend count stays well
                // under 2^28 (cifar-scale rows are ~4e6, 60x under the bound).
                debug_assert!(
                    (oc as u128)
                        * (oh_p as u128)
                        * (ow_p as u128)
                        * (ic as u128)
                        * (kh as u128)
                        * (kw as u128)
                        < (1u128 << 28),
                    "MaxPool 7D bias fold: row addend count breaches the documented \
                     n < 2^28 rounding-dominance bound"
                );
                for row in 0..spec_rows {
                    // Sanitize the carried err once per row (spec I5): non-finite
                    // or negative maps to +INF (degrade poison), NEVER to 0.
                    // Direct index — the length was hard-validated above (I6).
                    let row_oe = old_err.map_or(0.0_f64, |e| {
                        let v = e[row];
                        if v.is_finite() && v >= 0.0 {
                            f64::from(v)
                        } else {
                            f64::INFINITY
                        }
                    });
                    for o_c in 0..oc {
                        for o_h in 0..oh_p {
                            for o_w in 0..ow_p {
                                let mut row_acc = 0.0_f64;
                                let mut row_discharge = 0.0_f64;
                                for i_c in 0..ic {
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let coeff = patches_tensor
                                                [[row, o_c, o_h, o_w, i_c, ki, kj]]
                                                as f64;
                                            // Fast path: with no incoming error a
                                            // stored-zero tap contributes nothing to
                                            // bias or discharge. With row_oe > 0 a
                                            // stored-zero NON-WINNER tap still needs
                                            // its error discharged (the true coeff
                                            // may deviate by up to row_oe) — every
                                            // plan tap counts (spec I7).
                                            if coeff == 0.0 && row_oe == 0.0 {
                                                continue;
                                            }
                                            // Map this tap to the MaxPool OUTPUT
                                            // element it multiplies (row- and
                                            // oc-independent geometry, same as the
                                            // 6D arm). Out of range → padded
                                            // (non-existent) output, no constant.
                                            let ih =
                                                (o_h * in_sh + ki) as isize - in_pad_top as isize;
                                            let iw =
                                                (o_w * in_sw + kj) as isize - in_pad_left as isize;
                                            if ih < 0
                                                || ih as usize >= out_h
                                                || iw < 0
                                                || iw as usize >= out_w
                                                || i_c >= channels
                                            {
                                                continue;
                                            }
                                            let ih = ih as usize;
                                            let iw = iw as usize;
                                            let lb = lower_b_per_pos[[i_c, ih, iw]] as f64;
                                            let ub = upper_b_per_pos[[i_c, ih, iw]] as f64;
                                            // Definite-winner outputs carry (0,0) and
                                            // flow through the winner slope in Steps
                                            // 4-5 instead — no constant here (they
                                            // are covered by the M1 err carry).
                                            if lb == 0.0 && ub == 0.0 {
                                                continue;
                                            }
                                            // Discharge (M2): min/max(c·lb, c·ub) is
                                            // Lipschitz in c with constant
                                            // max(|lb|,|ub|) across the sign change,
                                            // so this tap's fold moves by at most
                                            // row_oe·max(|lb|,|ub|). Short-circuit
                                            // row_oe == 0 BEFORE the multiply:
                                            // max(|lb|,|ub|) may be +INF and
                                            // 0·INF = NaN (spec I5).
                                            if row_oe > 0.0 {
                                                row_discharge += row_oe * lb.abs().max(ub.abs());
                                            }
                                            // Interval-arithmetic constant rule per
                                            // tap (identical to the 6D arm).
                                            if is_lower {
                                                row_acc += if coeff > 0.0 {
                                                    coeff * lb
                                                } else {
                                                    coeff * ub
                                                };
                                            } else {
                                                row_acc += if coeff > 0.0 {
                                                    coeff * ub
                                                } else {
                                                    coeff * lb
                                                };
                                            }
                                        }
                                    }
                                }
                                new_bias[row] += row_acc;
                                // Apply the discharge OUTWARD (row_discharge ≥ 0,
                                // possibly +INF ⇒ vacuous row, NaN-free): lower
                                // bias down, upper bias up.
                                if is_lower {
                                    new_bias[row] -= row_discharge;
                                } else {
                                    new_bias[row] += row_discharge;
                                }
                            }
                        }
                    }
                }
            } else {
                // (The 6D nest below is deliberately kept textually verbatim —
                // including its original indentation — for the byte-identity pin,
                // spec I2 / `pooling_6d_bitwise_regression_max`.)
                for o_c in 0..oc {
                    for o_h in 0..oh_p {
                        for o_w in 0..ow_p {
                            let spec_idx = o_c * oh_p * ow_p + o_h * ow_p + o_w;
                            // Accumulate this spec row's constant contribution in f64.
                            let mut row_acc = 0.0_f64;
                            // Coefficient-error discharge for this row's non-winner taps
                            // (#patches-coeff-err-soundness). row_oe bounds |stored-true| for
                            // every incoming coeff in this spec row; each non-winner tap folds
                            // a coeff into the bias as coeff·const (const∈[lb,ub]), so its
                            // error perturbs the bias by ≤ row_oe·max(|lb|,|ub|). Summed and
                            // applied OUTWARD after the tap loops.
                            let row_oe = old_err.map_or(0.0_f64, |e| {
                                f64::from(e.get(spec_idx).copied().unwrap_or(0.0))
                            });
                            let mut row_discharge = 0.0_f64;
                            for i_c in 0..ic {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let coeff =
                                            patches_tensor[[o_c, o_h, o_w, i_c, ki, kj]] as f64;
                                        // Fast path: with no incoming error a stored-zero tap
                                        // contributes nothing to bias or discharge. With
                                        // row_oe>0 a stored-zero NON-WINNER tap still needs its
                                        // error discharged (true coeff may be up to row_oe), so
                                        // fall through to the lb/ub lookup below.
                                        if coeff == 0.0 && row_oe == 0.0 {
                                            continue;
                                        }
                                        // Map this tap to the MaxPool OUTPUT element it
                                        // multiplies. Out of range → the coefficient
                                        // multiplies a padded (non-existent) output and
                                        // contributes nothing here (the winner loop also
                                        // drops it; it carries no constant bound).
                                        let ih = (o_h * in_sh + ki) as isize - in_pad_top as isize;
                                        let iw = (o_w * in_sw + kj) as isize - in_pad_left as isize;
                                        if ih < 0
                                            || ih as usize >= out_h
                                            || iw < 0
                                            || iw as usize >= out_w
                                            || i_c >= channels
                                        {
                                            continue;
                                        }
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let lb = lower_b_per_pos[[i_c, ih, iw]] as f64;
                                        let ub = upper_b_per_pos[[i_c, ih, iw]] as f64;
                                        // Linear (definite-winner) MaxPool outputs carry
                                        // (lb,ub)=(0,0) and flow through the winner slope
                                        // in Steps 4-5 instead — no constant here.
                                        if lb == 0.0 && ub == 0.0 {
                                            continue;
                                        }
                                        // Non-winner tap: its coeff folds into the bias as
                                        // coeff·const. The incoming coefficient error on this
                                        // tap shifts that constant by ≤ row_oe·max(|lb|,|ub|);
                                        // since |min/max Σ − Σ min/max| ≤ Σ per-tap, sum the
                                        // per-tap bounds (counted even when the stored coeff is
                                        // 0) and apply outward after the loops.
                                        row_discharge += row_oe * lb.abs().max(ub.abs());
                                        // Interval-arithmetic constant rule per tap:
                                        //   Lower: pos*lb + neg*ub
                                        //   Upper: pos*ub + neg*lb
                                        // (pos minimized at lb / maximized at ub; neg vice versa.)
                                        if is_lower {
                                            row_acc +=
                                                if coeff > 0.0 { coeff * lb } else { coeff * ub };
                                        } else {
                                            row_acc +=
                                                if coeff > 0.0 { coeff * ub } else { coeff * lb };
                                        }
                                    }
                                }
                            }
                            new_bias[spec_idx] += row_acc;
                            // Apply the coefficient-error discharge OUTWARD (row_discharge ≥
                            // 0): lower bias down, upper bias up. The final directed f32 cast
                            // (next_down/next_up) then rounds further outward.
                            if is_lower {
                                new_bias[spec_idx] -= row_discharge;
                            } else {
                                new_bias[spec_idx] += row_discharge;
                            }
                        }
                    }
                }
            }

            // Step 3: Upsample patches (no division -- MaxPool not averaging)
            let upsampled = nearest_neighbor_upsample_last2(patches_tensor, pool_kh, pool_kw)?;

            // Steps 4-5: Apply winner_d slope via unfolding and element-wise multiply.
            let compose = |value: usize, scale: usize, add: usize, label: &str| {
                value
                    .checked_mul(scale)
                    .and_then(|scaled| scaled.checked_add(add))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "MaxPool Patches composed {label} overflows usize"
                        ))
                    })
            };
            let new_sh = compose(in_sh, pool_sh, 0, "height stride")?;
            let new_sw = compose(in_sw, pool_sw, 0, "width stride")?;
            let new_pad_left = compose(in_pad_left, pool_sw, pool_pw, "left padding")?;
            let new_pad_right = compose(in_pad_right, pool_sw, pool_pw, "right padding")?;
            let new_pad_top = compose(in_pad_top, pool_sh, pool_ph, "top padding")?;
            let new_pad_bottom = compose(in_pad_bottom, pool_sh, pool_ph, "bottom padding")?;

            let new_kh = kh.checked_mul(pool_kh).ok_or_else(|| {
                NyError::InvalidSpec(
                    "MaxPool Patches composed kernel height overflows usize".into(),
                )
            })?;
            let new_kw = kw.checked_mul(pool_kw).ok_or_else(|| {
                NyError::InvalidSpec("MaxPool Patches composed kernel width overflows usize".into())
            })?;
            let mut result_patches = if explicit_rows {
                ArrayD::<f32>::zeros(IxDyn(&[spec_rows, oc, oh_p, ow_p, ic, new_kh, new_kw]))
            } else {
                ArrayD::<f32>::zeros(IxDyn(&[oc, oh_p, ow_p, ic, new_kh, new_kw]))
            };

            if explicit_rows {
                // 7D winner application: identical val==0.0 skip / geometry /
                // val·winner_d body with the spec-row axis carried through
                // (row-independent geometry). The multiply by winner_d ∈ {0,1}
                // is exact in f32, so the M1 err carry over-bounds every
                // resulting coefficient (spec §9.2).
                for row in 0..spec_rows {
                    for o_c in 0..oc {
                        for o_h in 0..oh_p {
                            for o_w in 0..ow_p {
                                for i_c in 0..ic {
                                    for ki in 0..new_kh {
                                        for kj in 0..new_kw {
                                            let val = upsampled[[row, o_c, o_h, o_w, i_c, ki, kj]];
                                            if val == 0.0 {
                                                continue;
                                            }

                                            // Map to input spatial position
                                            let ih_raw =
                                                (o_h * new_sh + ki) as isize - new_pad_top as isize;
                                            let iw_raw = (o_w * new_sw + kj) as isize
                                                - new_pad_left as isize;

                                            if ih_raw < 0
                                                || ih_raw as usize >= in_h
                                                || iw_raw < 0
                                                || iw_raw as usize >= in_w
                                            {
                                                continue;
                                            }
                                            let ih = ih_raw as usize;
                                            let iw = iw_raw as usize;

                                            // winner_d is the same for both lower
                                            // and upper bounds
                                            let d = winner_d[[i_c, ih, iw]];
                                            result_patches[[row, o_c, o_h, o_w, i_c, ki, kj]] =
                                                val * d;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // (6D nest kept textually verbatim for the byte-identity pin.)
                for o_c in 0..oc {
                    for o_h in 0..oh_p {
                        for o_w in 0..ow_p {
                            for i_c in 0..ic {
                                for ki in 0..new_kh {
                                    for kj in 0..new_kw {
                                        let val = upsampled[[o_c, o_h, o_w, i_c, ki, kj]];
                                        if val == 0.0 {
                                            continue;
                                        }

                                        // Map to input spatial position
                                        let ih_raw =
                                            (o_h * new_sh + ki) as isize - new_pad_top as isize;
                                        let iw_raw =
                                            (o_w * new_sw + kj) as isize - new_pad_left as isize;

                                        if ih_raw < 0
                                            || ih_raw as usize >= in_h
                                            || iw_raw < 0
                                            || iw_raw as usize >= in_w
                                        {
                                            continue;
                                        }
                                        let ih = ih_raw as usize;
                                        let iw = iw_raw as usize;

                                        // winner_d is the same for both lower and upper bounds
                                        let d = winner_d[[i_c, ih, iw]];
                                        result_patches[[o_c, o_h, o_w, i_c, ki, kj]] = val * d;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let new_data = PatchesData {
                // Exact transform (gain 1): carry the incoming per-row error unchanged
                // (None→None) — legitimately Some on both the 6D dense and the 7D
                // explicit-rows layouts now (spec §9 M1; length hard-validated on 7D).
                // Sparse stays None via coeff_err_ok above.
                coeff_err: old_err.cloned(),
                patches: Some(result_patches),
                geometry: PatchGeometry::affine(
                    (new_sh, new_sw),
                    (new_pad_left, new_pad_right, new_pad_top, new_pad_bottom),
                ),
                identity: false,
                output_shape: patches_data.output_shape,
                input_shape: (channels, in_h, in_w),
                unstable_idx: None,
            };

            // Directed rounding (#1745)
            let new_bias_f32 = if is_lower {
                new_bias.mapv(|x| next_down_f32(x as f32))
            } else {
                new_bias.mapv(|x| next_up_f32(x as f32))
            };
            Ok((new_data, new_bias_f32))
        };

        // Both bounds use same winner_d slope; only bias differs
        let (new_lower_a, new_lower_b) = process_patches(&bounds.lower_a, &bounds.lower_b, true)?;
        let (new_upper_a, new_upper_b) = process_patches(&bounds.upper_a, &bounds.upper_b, false)?;

        let result = PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        };

        if result.lower_a.should_fallback_to_dense() {
            Ok(CrownBounds::Dense(result.to_dense()?))
        } else {
            Ok(CrownBounds::Patches(Box::new(result)))
        }
    }
}

// --- 7D explicit-rows coeff_err closure: oracle + guard tests ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9.4 T2/T4/T5)
#[cfg(test)]
mod coeff_err_7d_tests {
    use super::*;
    use crate::bounds::patches::UnstableIdx;
    use ndarray::Array1;

    // Oracle-noise note (spec §9.4): the f64 oracle computations below carry
    // ~2^-53 relative rounding, while the implementation's biases take a
    // directed outward f32 cast (>= 2^-25 relative slack) after an f64
    // discharge — so the outwardness comparisons cannot flip on oracle noise.

    /// maxpool (2,2)/(2,2) over input (1,2,8) -> output (1,1,4).
    /// Window 0: definite winner at input (0,0,0) (l=5 >= all other u);
    /// windows 1..3: non-winner with DISTINCT constant bounds
    /// (lb,ub) = (0.3,1.7), (-0.9,0.4), (-2.1,-0.5).
    fn maxpool_fixture() -> (MaxPool2dLayer, BoundedTensor) {
        let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
        let pre_lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 8]),
            vec![
                5.0_f32, -1.0, 0.3, 0.1, -0.9, -1.5, -2.1, -3.0, //
                0.1, 0.2, -0.4, 0.25, -1.2, -1.3, -2.5, -2.8,
            ],
        )
        .unwrap();
        let pre_upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 8]),
            vec![
                6.0_f32, 1.0, 1.7, 1.2, 0.4, 0.2, -0.5, -1.0, //
                2.0, 0.5, 0.9, 1.1, 0.35, 0.1, -0.8, -0.6,
            ],
        )
        .unwrap();
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();
        (maxpool, pre)
    }

    /// Per-output-window constant bounds of the fixture, in the exact f64
    /// values the implementation reads (f32 widened). Window 0 is the definite
    /// winner and carries the (0,0) sentinel.
    fn fixture_window_bounds() -> [(f64, f64); 4] {
        [
            (0.0, 0.0),
            (f64::from(0.3f32), f64::from(1.7f32)),
            (f64::from(-0.9f32), f64::from(0.4f32)),
            (f64::from(-2.1f32), f64::from(-0.5f32)),
        ]
    }

    /// Incoming 7D explicit-rows side [rows=2, oc=1, oh=1, ow=4, ic=1, kh=1,
    /// kw=1]: 1x1 taps over the maxpool OUTPUT grid (1,1,4); 2 spec rows of 4
    /// positions each (position ow taps output window ow).
    fn make_side_7d(vals: Vec<f32>, err: Option<Vec<f32>>) -> PatchesData {
        PatchesData {
            coeff_err: err.map(Array1::from_vec),
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 4, 1, 1, 1]), vals).unwrap()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 4),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        }
    }

    #[test]
    fn maxpool_anchored_geometry_refuses_before_relaxation_arithmetic() {
        let (maxpool, pre) = maxpool_fixture();
        let mut bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vec![0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0], None),
            lower_b: Array1::from_vec(vec![0.125, -0.25]),
            upper_a: make_side_7d(vec![-0.75, 0.5, -0.25, 1.0, -1.25, 1.5, -1.75, 2.0], None),
            upper_b: Array1::from_vec(vec![-0.375, 0.5]),
        };
        let anchored = PatchGeometry::anchored(vec![0], vec![0, 1, 2, 3]).unwrap();
        bounds.lower_a.geometry = anchored.clone();
        bounds.upper_a.geometry = anchored;

        let lower_patches_before = bounds.lower_a.patches.clone();
        let upper_patches_before = bounds.upper_a.patches.clone();
        let lower_bias_before = bounds.lower_b.clone();
        let upper_bias_before = bounds.upper_b.clone();

        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(matches!(
            result,
            Err(NyError::UnsupportedConfiguration(message))
                if message.contains("MaxPool Patches backward")
        ));
        assert_eq!(bounds.lower_a.patches, lower_patches_before);
        assert_eq!(bounds.upper_a.patches, upper_patches_before);
        assert_eq!(bounds.lower_b, lower_bias_before);
        assert_eq!(bounds.upper_b, upper_bias_before);
    }

    /// Independent bitwise replica of the 7D bias fold + discharge for one spec
    /// row of the fixture (4 positions, 1 tap each), mirroring the
    /// implementation's per-position f64 apply order and skips exactly.
    fn expected_bias_7d(b0: f32, stored_row: &[f32], row_oe: f64, is_lower: bool) -> f32 {
        let win = fixture_window_bounds();
        let mut b = f64::from(b0);
        for ow in 0..4usize {
            let (lb, ub) = win[ow];
            let c = f64::from(stored_row[ow]);
            let mut acc = 0.0_f64;
            let mut disch = 0.0_f64;
            if !(c == 0.0 && row_oe == 0.0 || lb == 0.0 && ub == 0.0) {
                if row_oe > 0.0 {
                    disch += row_oe * lb.abs().max(ub.abs());
                }
                acc += if is_lower {
                    if c > 0.0 {
                        c * lb
                    } else {
                        c * ub
                    }
                } else if c > 0.0 {
                    c * ub
                } else {
                    c * lb
                };
            }
            b += acc;
            if is_lower {
                b -= disch;
            } else {
                b += disch;
            }
        }
        if is_lower {
            next_down_f32(b as f32)
        } else {
            next_up_f32(b as f32)
        }
    }

    /// §9.4 T2: M1 carry bitwise unchanged; every result coefficient within
    /// `e_old[r]` of an exact f64 truth; bias outward vs the f64 true folds;
    /// bitwise formula pin for the discharge `D_r`.
    ///
    /// Adversarial corner (pins the fast-path port): the row-0 tap at position
    /// ow=1 maps to a NON-WINNER output, has stored coefficient exactly 0.0
    /// and true coefficient +e_old[0] — a mis-ported `coeff == 0.0` skip that
    /// ignores `row_oe` would drop its discharge and emit an unsound bias.
    #[test]
    fn maxpool_7d_carry_and_discharge_cover_oracle() {
        let (maxpool, pre) = maxpool_fixture();

        let stored_l: Vec<f32> = vec![1.25, 0.0, -0.75, 2.5, -0.5, 1.5, 0.25, -1.0];
        let e_l: Vec<f32> = vec![1.0e-3, 5.0e-4];
        let e0 = f64::from(e_l[0]);
        let e1 = f64::from(e_l[1]);
        // True coefficients: stored + delta, |delta| <= e_old[row]; the ow=1
        // corner uses delta = +e0 exactly on a stored 0.0.
        let deltas_l: [f64; 8] = [
            -0.4 * e0,
            e0,
            0.6 * e0,
            -e0,
            0.6 * e1,
            -0.98 * e1,
            0.2 * e1,
            -0.4 * e1,
        ];
        let true_l: Vec<f64> = stored_l
            .iter()
            .zip(deltas_l.iter())
            .map(|(&s, &d)| f64::from(s) + d)
            .collect();

        let stored_u: Vec<f32> = vec![-0.35, 0.85, 1.15, -0.65, 0.6, -1.2, 0.0, 0.45];
        let e_u: Vec<f32> = vec![2.0e-3, 0.0];
        let eu0 = f64::from(e_u[0]);
        // Row 1 carries err 0.0 => its truth must equal the stored values.
        let deltas_u: [f64; 8] = [0.5 * eu0, -eu0, 0.3 * eu0, -0.7 * eu0, 0.0, 0.0, 0.0, 0.0];
        let true_u: Vec<f64> = stored_u
            .iter()
            .zip(deltas_u.iter())
            .map(|(&s, &d)| f64::from(s) + d)
            .collect();

        let b0_l: [f32; 2] = [0.1, -0.2];
        let b0_u: [f32; 2] = [0.5, 0.6];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(stored_l.clone(), Some(e_l.clone())),
            lower_b: Array1::from_vec(b0_l.to_vec()),
            upper_a: make_side_7d(stored_u.clone(), Some(e_u.clone())),
            upper_b: Array1::from_vec(b0_u.to_vec()),
        };

        let result = maxpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("7D maxpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => {
                panic!("expected Patches mode (upsampled kernel area 4 < input area 16)")
            }
        };
        assert_eq!(pb.row_count, 2);
        assert_eq!(
            pb.lower_a.patches.as_ref().unwrap().shape(),
            &[2, 1, 1, 4, 1, 2, 2]
        );

        // (1) M1 carry: coeff_err cloned UNCHANGED per spec row (gain 1).
        for (side, e_in) in [(&pb.lower_a, &e_l), (&pb.upper_a, &e_u)] {
            let carried = side
                .coeff_err
                .as_ref()
                .expect("7D maxpool backward must carry coeff_err");
            assert_eq!(carried.len(), 2);
            for r in 0..2 {
                assert_eq!(carried[r].to_bits(), e_in[r].to_bits(), "carry row {r}");
            }
        }

        // (2) Coefficient coverage: the only surviving tap per row is the
        // winner tap (position ow=0, kernel (0,0) -> input (0,0,0), d=1); its
        // stored value passes through EXACTLY (val * 1.0), so the carried err
        // covers it against the truth. Every other cell must be exactly 0.0
        // (true composed linear coefficient 0: constant relaxation windows).
        for (side, stored, truth, e_in) in [
            (&pb.lower_a, &stored_l, &true_l, &e_l),
            (&pb.upper_a, &stored_u, &true_u, &e_u),
        ] {
            let pt = side.patches.as_ref().unwrap();
            for row in 0..2usize {
                let winner = pt[[row, 0, 0, 0, 0, 0, 0]];
                assert_eq!(
                    winner.to_bits(),
                    stored[4 * row].to_bits(),
                    "row {row}: winner coefficient must pass through exactly"
                );
                assert!(
                    (f64::from(winner) - truth[4 * row]).abs() <= f64::from(e_in[row]),
                    "row {row}: winner coefficient not covered by carried err"
                );
                for ow in 0..4usize {
                    for ki in 0..2usize {
                        for kj in 0..2usize {
                            if ow == 0 && ki == 0 && kj == 0 {
                                continue;
                            }
                            assert_eq!(
                                pt[[row, 0, 0, ow, 0, ki, kj]],
                                0.0,
                                "row {row} pos {ow} tap ({ki},{kj}) must be zero"
                            );
                        }
                    }
                }
            }
        }

        // (3) Bias outwardness vs the exact f64 true folds (winner windows
        // contribute no constant; each non-winner tap contributes
        // min/max(c_true·lb, c_true·ub)).
        let win = fixture_window_bounds();
        for row in 0..2usize {
            let mut true_low = f64::from(b0_l[row]);
            let mut true_up = f64::from(b0_u[row]);
            for ow in 1..4usize {
                let (lb, ub) = win[ow];
                let tl = true_l[4 * row + ow];
                let tu = true_u[4 * row + ow];
                true_low += (tl * lb).min(tl * ub);
                true_up += (tu * lb).max(tu * ub);
            }
            assert!(
                f64::from(pb.lower_b[row]) <= true_low,
                "row {row}: lower bias {} not outward of true fold {true_low}",
                pb.lower_b[row]
            );
            assert!(
                f64::from(pb.upper_b[row]) >= true_up,
                "row {row}: upper bias {} not outward of true fold {true_up}",
                pb.upper_b[row]
            );
        }

        // (4) Bitwise formula pin: bias == independent replica of the fold +
        // discharge D_r = Σ row_oe·max(|lb|,|ub|) over valid non-winner taps
        // (counted even at stored coeff 0.0 whenever row_oe > 0 — the corner).
        for row in 0..2usize {
            let exp_l = expected_bias_7d(
                b0_l[row],
                &stored_l[4 * row..4 * row + 4],
                f64::from(e_l[row]),
                true,
            );
            let exp_u = expected_bias_7d(
                b0_u[row],
                &stored_u[4 * row..4 * row + 4],
                f64::from(e_u[row]),
                false,
            );
            assert_eq!(
                pb.lower_b[row].to_bits(),
                exp_l.to_bits(),
                "row {row} lower"
            );
            assert_eq!(
                pb.upper_b[row].to_bits(),
                exp_u.to_bits(),
                "row {row} upper"
            );
        }

        // Liveness of the corner: dropping the stored-zero non-winner tap from
        // the discharge (a mis-ported fast path) would shift the row-0 lower
        // bias UP by e0·max(|0.3|,|1.7|) — assert the emitted bias sits at
        // least that far below the no-corner-discharge replica.
        let mut no_corner = f64::from(b0_l[0]);
        for ow in 1..4usize {
            let (lb, ub) = win[ow];
            let c = f64::from(stored_l[ow]);
            no_corner += if c > 0.0 { c * lb } else { c * ub };
            if ow != 1 {
                no_corner -= e0 * lb.abs().max(ub.abs());
            }
        }
        assert!(
            f64::from(pb.lower_b[0]) < no_corner,
            "row 0: the stored-zero non-winner tap's discharge is missing"
        );
    }

    /// padding >= kernel must fail closed before any window is examined: an
    /// all-padding window's (0,0) init would otherwise masquerade as the
    /// definite-winner sentinel and emit a ~[0,0] row for a -inf output.
    #[test]
    fn maxpool_patches_rejects_padding_ge_kernel() {
        let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (3, 3));
        let pre_lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 5.0_f32);
        let pre_upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 6.0_f32);
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::zeros(2),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::zeros(2),
        };
        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::InvalidSpec(_))),
            "expected InvalidSpec for padding >= kernel, got {result:?}"
        );
    }

    /// §9.4 T4: degenerate 7D shape [3,1,1,2,1,2,2] — pre-change this
    /// destructured the shape as 6D and PANICKED on the 6-index tap read
    /// (spec R7); post-change it must process with row semantics (or return a
    /// clean Err), never panic.
    #[test]
    fn maxpool_7d_degenerate_shape_no_panic() {
        let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
        // maxpool input (1,4,6) -> output (1,2,3); incoming taps the output
        // grid with kernel (2,2)/stride (1,1): ih ∈ {0,1}, iw ∈ {0,1,2}.
        let pre_lower = ArrayD::from_elem(IxDyn(&[1, 4, 6]), -1.0f32);
        let pre_upper = ArrayD::from_elem(IxDyn(&[1, 4, 6]), 1.0f32);
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();
        let vals: Vec<f32> = (0..24).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let make_side = |err: Option<Vec<f32>>| PatchesData {
            coeff_err: err.map(Array1::from_vec),
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 2, 1, 2, 2]), vals.clone()).unwrap(),
            ),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 2, 3),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 3,
            lower_a: make_side(Some(vec![1.0e-3, 0.0, 5.0e-4])),
            lower_b: Array1::zeros(3),
            upper_a: make_side(None),
            upper_b: Array1::zeros(3),
        };
        let result = maxpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("degenerate 7D shape must process with row semantics, not panic");
        match result {
            CrownBounds::Patches(pb) => {
                assert_eq!(
                    pb.lower_a.patches.as_ref().unwrap().shape(),
                    &[3, 1, 1, 2, 1, 4, 4]
                );
                // M1 carry: Some in => Some out (bitwise), None in => None out.
                assert!(pb.lower_a.coeff_err.is_some());
                assert!(pb.upper_a.coeff_err.is_none());
                for b in pb.lower_b.iter().chain(pb.upper_b.iter()) {
                    assert!(!b.is_nan(), "bias must stay NaN-free");
                }
            }
            CrownBounds::Dense(_) => panic!("expected Patches (kernel area 16 < input area 24)"),
        }
    }

    /// §9.4 T4: a sparse 4D layout must fail with a clean error (pre-change:
    /// out-of-bounds `shape[4]` panic at the destructure).
    #[test]
    fn maxpool_sparse_4d_clean_error() {
        let (maxpool, pre) = maxpool_fixture();
        let make_sparse = || PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[2, 1, 1, 1]), 1.0f32)),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 4),
            input_shape: (1, 1, 4),
            unstable_idx: Some(UnstableIdx {
                channels: vec![0, 0],
                heights: vec![0, 0],
                widths: vec![0, 1],
            }),
        };
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_sparse(),
            lower_b: Array1::zeros(2),
            upper_a: make_sparse(),
            upper_b: Array1::zeros(2),
        };
        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "sparse 4D must fail cleanly (was a shape[4] panic pre-change), got {result:?}"
        );
    }

    /// §9.4 T5: carried err whose length is not row_count => hard Err
    /// (spec I6), never a silent under-count.
    #[test]
    fn max_7d_coeff_err_length_mismatch_errors() {
        let (maxpool, pre) = maxpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), Some(vec![1e-3, 2e-3, 3e-3])),
            lower_b: Array1::zeros(2),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::zeros(2),
        };
        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for err len 3 vs row_count 2, got {result:?}"
        );
    }

    /// §9.4 T5: bias length disagreeing with the spec-row count => hard Err.
    #[test]
    fn max_7d_bias_length_mismatch_errors() {
        let (maxpool, pre) = maxpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::zeros(3),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::zeros(3),
        };
        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for bias len 3 vs 2 spec rows, got {result:?}"
        );
    }

    /// §9.4 T5: 7D tensor whose axis 0 disagrees with row_count => hard Err.
    #[test]
    fn max_7d_row_count_mismatch_errors() {
        let (maxpool, pre) = maxpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 3,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::zeros(3),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::zeros(3),
        };
        let result = maxpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for tensor rows 2 vs row_count 3, got {result:?}"
        );
    }
}

// --- 7D explicit-rows coeff_err closure: 6D byte-identity pin ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9.4 T3; validation gate §13 item 1)
#[cfg(test)]
mod bitwise_regression_pins {
    use super::*;
    use ndarray::Array1;

    /// Compare f32 slices against pinned bit literals. On ANY mismatch, dump
    /// ALL actual arrays in copy-pastable `const` form (one-shot capture from
    /// the pre-change tree), then panic.
    fn check_bit_pins(pins: &[(&str, &[f32], &[u32])]) {
        let mut mismatch = false;
        for (label, actual, expected) in pins {
            let bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
            if bits.as_slice() != *expected {
                mismatch = true;
                eprintln!("PIN MISMATCH: {label}");
            }
        }
        if mismatch {
            for (label, actual, _) in pins {
                let dump: Vec<String> = actual
                    .iter()
                    .map(|v| format!("{:#010X}", v.to_bits()))
                    .collect();
                eprintln!(
                    "const {label}: [u32; {}] = [{}];",
                    actual.len(),
                    dump.join(", ")
                );
            }
            panic!("byte-identity pin mismatch — actual bit arrays dumped above");
        }
    }

    /// BYTE-IDENTITY PIN: the 6D dense MaxPool patches backward WITH incoming
    /// `coeff_err: Some` on both sides must stay bit-for-bit unchanged by the
    /// 7D explicit-rows coeff_err closure (spec §9.3 step 5 keeps the 6D arms
    /// of the ndim dispatch, the step-2 bias/discharge nest, and the winner
    /// upsample VERBATIM; only new 7D arms and guards are added).
    ///
    /// Committed and verified green against the UNMODIFIED (pre-closure) tree;
    /// bit literals captured from that tree. Must pass unmodified after the
    /// closure lands.
    ///
    /// Fixture: pool (2,2)/(2,2) over input (1,2,8) -> output (1,1,4); windows
    /// 0 and 2 have definite winners (carry path, coeff·winner_d), windows 1
    /// and 3 are non-winner with DISTINCT (lb,ub) constant bounds (bias fold +
    /// per-row coeff_err discharge); errs include exact-0.0 rows.
    #[test]
    fn pooling_6d_bitwise_regression_max() {
        let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

        // Pre-activation (1, 2, 8). Window ow covers {(0,2ow),(0,2ow+1),
        // (1,2ow),(1,2ow+1)}:
        //   window 0: definite winner at (0,0) (l=5.0 >= all other u).
        //   window 1: no winner; (max_lower, max_upper) = (0.3, 1.7).
        //   window 2: definite winner at (1,4) (l=0.9 >= all other u).
        //   window 3: no winner; (max_lower, max_upper) = (-2.1, -0.5).
        let pre_lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 8]),
            vec![
                5.0_f32, -1.0, 0.3, 0.1, -0.9, -1.5, -2.1, -3.0, //
                0.0, 0.1, -0.4, 0.25, 0.9, -1.2, -2.5, -2.8,
            ],
        )
        .unwrap();
        let pre_upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 8]),
            vec![
                6.0_f32, 1.0, 1.7, 1.2, 0.4, 0.2, -0.5, -1.0, //
                2.0, 0.5, 0.9, 1.1, 1.4, 0.1, -0.8, -0.6,
            ],
        )
        .unwrap();
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        // Incoming 6D patches [oc=1, oh=1, ow=4, ic=1, kh=1, kw=1] indexing the
        // maxpool OUTPUT grid (1, 1, 4); 4 spec rows.
        let make_side = |vals: Vec<f32>, err: Vec<f32>| PatchesData {
            coeff_err: Some(Array1::from_vec(err)),
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 1, 1, 1]), vals).unwrap()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 4),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 4,
            lower_a: make_side(
                vec![0.7_f32, -1.3, 0.55, 2.4],
                vec![1.0e-3, 0.0, 5.0e-4, 2.0e-3],
            ),
            lower_b: Array1::from_vec(vec![0.1_f32, -0.2, 0.3, -0.4]),
            upper_a: make_side(
                vec![-0.35_f32, 0.85, 1.15, -0.65],
                vec![2.0e-3, 7.0e-4, 0.0, 1.0e-4],
            ),
            upper_b: Array1::from_vec(vec![0.5_f32, 0.6, -0.7, 0.8]),
        };

        let result = maxpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("maxpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => {
                panic!("expected Patches mode (upsampled kernel area 4 < input area 16)")
            }
        };

        // Metadata composition (must not move either).
        assert_eq!(pb.row_count, 4);
        assert_eq!(
            pb.lower_a.geometry,
            PatchGeometry::affine((2, 2), (0, 0, 0, 0))
        );
        assert_eq!(pb.lower_a.output_shape, (1, 1, 4));
        assert_eq!(pb.lower_a.input_shape, (1, 2, 8));
        assert_eq!(
            pb.lower_a.patches.as_ref().unwrap().shape(),
            &[1, 1, 4, 1, 2, 2]
        );
        assert_eq!(
            pb.upper_a.patches.as_ref().unwrap().shape(),
            &[1, 1, 4, 1, 2, 2]
        );

        // M1 carry: coeff_err cloned UNCHANGED per spec row (gain 1) — pinned
        // both against the incoming arrays and, below, against bit literals.
        let lower_err: Vec<f32> = pb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("6D maxpool backward must carry lower coeff_err")
            .to_vec();
        let upper_err: Vec<f32> = pb
            .upper_a
            .coeff_err
            .as_ref()
            .expect("6D maxpool backward must carry upper coeff_err")
            .to_vec();

        // Bit literals captured from pre-change HEAD (see doc comment).
        const EXP_LOWER_PATCHES: [u32; 16] = [
            0x3F333333, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000,
            0x80000000, 0x00000000, 0x00000000, 0x3F0CCCCD, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000,
        ];
        const EXP_UPPER_PATCHES: [u32; 16] = [
            0xBEB33333, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x3F933333, 0x00000000, 0x80000000, 0x80000000,
            0x80000000, 0x80000000,
        ];
        const EXP_LOWER_B: [u32; 4] = [0x3DCCCCCC, 0xC01A3D72, 0x3E999999, 0xC0AE36E4];
        const EXP_UPPER_B: [u32; 4] = [0x3F000001, 0x4002F4C8, 0xBF333332, 0x400A92CE];
        const EXP_LOWER_ERR: [u32; 4] = [0x3A83126F, 0x00000000, 0x3A03126F, 0x3B03126F];
        const EXP_UPPER_ERR: [u32; 4] = [0x3B03126F, 0x3A378034, 0x00000000, 0x38D1B717];

        let lower_patches: Vec<f32> = pb
            .lower_a
            .patches
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();
        let upper_patches: Vec<f32> = pb
            .upper_a
            .patches
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();

        check_bit_pins(&[
            ("EXP_LOWER_PATCHES", &lower_patches, &EXP_LOWER_PATCHES),
            ("EXP_UPPER_PATCHES", &upper_patches, &EXP_UPPER_PATCHES),
            ("EXP_LOWER_B", pb.lower_b.as_slice().unwrap(), &EXP_LOWER_B),
            ("EXP_UPPER_B", pb.upper_b.as_slice().unwrap(), &EXP_UPPER_B),
            ("EXP_LOWER_ERR", &lower_err, &EXP_LOWER_ERR),
            ("EXP_UPPER_ERR", &upper_err, &EXP_UPPER_ERR),
        ]);
    }
}
