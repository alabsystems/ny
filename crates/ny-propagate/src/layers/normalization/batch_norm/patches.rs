// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward propagation for BatchNorm.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use super::types::BatchNormLayer;

/// Patches-mode CROWN backward through BatchNorm.
///
/// BatchNorm is a per-channel linear operation: y[c,h,w] = scale[c] * x[c,h,w] + bias[c].
/// In the Patches representation [oc, oh, ow, ic, ki, kj], the `ic` dimension directly
/// corresponds to the BatchNorm channel, so backward is simple:
///   - Scale each coefficient by scale[ic]
///   - Add bias contribution to the output bias vectors
///
/// No upper/lower swap needed: CROWN backward composes by substitution (exact linear
/// layer), not IBP. Negative scale just flips coefficient sign.
///
/// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
/// Part of #2613
impl crate::layers::common::PatchesPropagation for BatchNormLayer {
    fn propagate_patches(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};

        let row_count = bounds.row_count;

        let process_patches = |patches_data: &PatchesData,
                               bias_vec: &Array1<f32>|
         -> Result<(PatchesData, Array1<f64>, Array1<f64>)> {
            let (out_c, out_h, out_w) = patches_data.output_shape;
            let (in_c, _in_h, _in_w) = patches_data.input_shape;

            // Channel count must match BatchNorm num_channels
            if in_c != self.num_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![self.num_channels],
                    got: vec![in_c],
                });
            }

            let mut new_bias = bias_vec.mapv(|x| x as f64);

            // Determine the patches layout for the non-identity case so the bias-length
            // guard matches the destination index used below.
            //   - rank-6 [oc, oh, ow, ic, ki, kj]   : bias is per patches output neuron
            //     (`n = oc*out_h*out_w + oh*out_w + ow`), so bias len must equal the
            //     output-neuron count.
            //   - rank-7 [row, oc, oh, ow, ic, ki, kj] (EXPLICIT-ROWS, e.g. the 1x1 conv
            //     re-entry on a rank-3 spatial spec): bias is per spec-row, so bias len
            //     must equal `row_count`.
            // Identity patches always use the per-neuron layout below, so treat them as
            // rank-6 for the guard.
            // For some specs (e.g. cgan's disjunctive multi-clause input split, where
            // the spec/bias vector is per spec-row, not per layer neuron) the counts
            // differ: a shorter bias panics with an ndarray index-out-of-bounds, and a
            // longer bias would leave trailing rows without their BatchNorm bias
            // contribution (silently unsound). In either mismatch, return an error so
            // `try_patches_or_dense_fallback` drops to the dense BatchNorm backward,
            // which handles arbitrary spec layouts exactly. SOUND: dense is exact;
            // patches and dense agree when the layout matches (the common case, so no
            // perf regression there).
            let explicit_rows = if patches_data.identity {
                false
            } else {
                let shape = patches_data
                    .patches
                    .as_ref()
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "PatchesData: not identity but patches tensor is None".into(),
                        )
                    })?
                    .shape()
                    .to_vec();
                match shape.len() {
                    6 => false,
                    7 => {
                        if shape[0] != row_count {
                            return Err(NyError::ShapeMismatch {
                                expected: vec![row_count],
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
                }
            };

            if explicit_rows {
                // Explicit-rows: bias is per spec-row.
                if row_count != new_bias.len() {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![row_count],
                        got: vec![new_bias.len()],
                    });
                }
            } else {
                let out_neuron_count = out_c
                    .checked_mul(out_h)
                    .and_then(|x| x.checked_mul(out_w))
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "BatchNorm patches: output-neuron count overflow".into(),
                        )
                    })?;
                if out_neuron_count != new_bias.len() {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_neuron_count],
                        got: vec![new_bias.len()],
                    });
                }
            }

            if patches_data.identity {
                // Identity patches: each output (oc, oh, ow) maps to itself with coeff 1.0.
                // In identity mode, the coefficient for output neuron (oc, oh, ow)
                // references input neuron at the same position with channel oc.
                // But oc corresponds to the output channel of the previous conv,
                // which IS the BatchNorm's input channel. So:
                //   new_coeff = 1.0 * scale[oc]  (the identity coeff times BN scale)
                //   delta_bias = 1.0 * bias[oc]   (one coefficient per output neuron)
                //
                // After scaling, this is no longer identity — materialize.
                // Actually, we can represent this as a materialized 6D tensor
                // where patches[oc,oh,ow,oc,0,0] = scale[oc] (only diagonal ic=oc).
                let mut patches_arr =
                    ArrayD::<f32>::zeros(IxDyn(&[out_c, out_h, out_w, in_c, 1, 1]));
                for oc in 0..out_c.min(in_c) {
                    let s = self.scale[[oc]];
                    let b = self.bias[[oc]] as f64;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            patches_arr[[oc, oh, ow, oc, 0, 0]] = s;
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            // Each identity position contributes 1.0 * bias[oc]
                            new_bias[n] += b;
                        }
                    }
                }
                // Certified error channel (#patches-coeff-err-soundness). The
                // materialized diagonal coefficient is scale[oc] (a direct f32
                // assignment), whose gap to the real BN scale is scale_err[oc].
                // HOLE1: the folded diagonal bias is bias[oc], gap bias_err[oc].
                // Identity input carries no coefficient error (old_err defaults to
                // 0). This is the per-neuron 6D dense layout; a sparse
                // (unstable_idx Some) identity is not wired for the dense to_dense
                // err scatter, so keep the channel None + zero discharge there
                // (prior behavior, no regression).
                let (coeff_err, widen): (Option<Array1<f32>>, Array1<f64>) =
                    if patches_data.unstable_idx.is_some() {
                        (None, Array1::<f64>::zeros(new_bias.len()))
                    } else {
                        let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                        let old = patches_data.coeff_err.as_ref();
                        let mut ne = Array1::<f32>::zeros(new_bias.len());
                        let mut wd = Array1::<f64>::zeros(new_bias.len());
                        for oc in 0..out_c.min(in_c) {
                            let s = f64::from(self.scale[[oc]]).abs();
                            let se = f64::from(self.scale_err[[oc]]);
                            let bb = f64::from(self.bias[[oc]]).abs();
                            let be = f64::from(self.bias_err[[oc]]);
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let n = oc * out_h * out_w + oh * out_w + ow;
                                    let oe = old.map_or(0.0, |e| {
                                        f64::from(e.get(n).copied().unwrap_or(0.0))
                                    });
                                    ne[n] = next_up_f32((rnd * s + se + (s + se) * oe) as f32);
                                    wd[n] = oe * bb + be;
                                }
                            }
                        }
                        (Some(ne), wd)
                    };
                let new_data = PatchesData {
                    coeff_err,
                    patches: Some(patches_arr),
                    stride: patches_data.stride,
                    padding: patches_data.padding,
                    identity: false,
                    output_shape: patches_data.output_shape,
                    input_shape: patches_data.input_shape,
                    unstable_idx: None,
                };
                return Ok((new_data, new_bias, widen));
            }

            // Non-identity: scale existing patches coefficients by scale[ic]
            // and accumulate bias contributions.
            let patches = patches_data.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;
            let shape = patches.shape();
            // kh/kw are the trailing kernel axes. For the EXPLICIT-ROWS (rank-7)
            // layout [row, oc, oh, ow, ic, ki, kj] they are axes 5/6; for the rank-6
            // layout [oc, oh, ow, ic, ki, kj] they are axes 4/5.
            let (kh, kw) = if explicit_rows {
                (shape[5], shape[6])
            } else {
                (shape[4], shape[5])
            };

            let mut new_patches = patches.clone();

            // Extract stride/padding for bounds checking.
            // Padding-zone positions map to virtual zero-input — their coefficients
            // are correctly dropped by to_dense(), but we must also exclude them
            // from the bias sum. Reference: PatchesData::scatter_patches_to_dense
            // (bounds/patches.rs:273-278).
            let (sh, sw) = patches_data.stride;
            let (pad_left, _pad_right, pad_top, _pad_bottom) = patches_data.padding;
            let (_in_c, in_h, in_w) = patches_data.input_shape;

            // Scale coefficients by per-channel scale and accumulate bias.
            // Coefficient scaling applies to ALL positions (padding-zone coefficients
            // stay dead through to_dense()). Bias accumulation only includes valid
            // (non-padding) positions. The coefficient transform (`new = coeff * scale[ic]`)
            // and the padding-zone-excluded bias contribution (`Σ_valid(coeff) * bias[ic]`)
            // are identical in both arms; only the index arity and the bias destination
            // differ (per spec-row for rank-7 vs per output neuron for rank-6), matching
            // the audited ReLU/elementwise and conv compute_patches_bias handlers.
            if explicit_rows {
                for row in 0..row_count {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut bias_accum = 0.0_f64;
                                for ic in 0..in_c {
                                    let s = self.scale[[ic]];
                                    let b = self.bias[[ic]] as f64;
                                    let mut channel_sum = 0.0_f64;
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let coeff = patches[[row, oc, oh, ow, ic, ki, kj]];
                                            new_patches[[row, oc, oh, ow, ic, ki, kj]] = coeff * s;
                                            // Only include valid (non-padding) positions in bias sum
                                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                            let iw_raw =
                                                (ow * sw + kj) as isize - pad_left as isize;
                                            if ih_raw >= 0
                                                && (ih_raw as usize) < in_h
                                                && iw_raw >= 0
                                                && (iw_raw as usize) < in_w
                                            {
                                                channel_sum += coeff as f64;
                                            }
                                        }
                                    }
                                    bias_accum += channel_sum * b;
                                }
                                new_bias[row] += bias_accum;
                            }
                        }
                    }
                }
            } else {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            let mut bias_accum = 0.0_f64;
                            for ic in 0..in_c {
                                let s = self.scale[[ic]];
                                let b = self.bias[[ic]] as f64;
                                let mut channel_sum = 0.0_f64;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let coeff = patches[[oc, oh, ow, ic, ki, kj]];
                                        new_patches[[oc, oh, ow, ic, ki, kj]] = coeff * s;
                                        // Only include valid (non-padding) positions in bias sum
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw >= 0
                                            && (ih_raw as usize) < in_h
                                            && iw_raw >= 0
                                            && (iw_raw as usize) < in_w
                                        {
                                            channel_sum += coeff as f64;
                                        }
                                    }
                                }
                                bias_accum += channel_sum * b;
                            }
                            new_bias[n] += bias_accum;
                        }
                    }
                }
            }

            // Certified error channel (#patches-coeff-err-soundness). BN substitutes
            // z = scale·x + bn_bias; the coefficient on x scales by scale[ic]. With
            // rnd = γ_1 = u/(1-u) >= 2^-24 (sound single-f32-multiply rounding
            // factor), gain = max_c(|scale[c]|+scale_err[c]) and c(j) = ic:
            //   new_err[i] = next_up( max_j(rnd·|new_coeff[i,j]| + |coeff[i,j]|·scale_err[c(j)])
            //                         + gain·old_err[i] ).
            // HOLE1: the folded bias Σ_valid coeff·bn_bias picks up
            //   widen[i] = Σ_valid|coeff[i,tap]|·bias_err[c] + old_err[i]·Σ_valid|bn_bias[c]|
            // over the SAME valid (non-padding) taps as the bias loop; discharged
            // outward by the caller (lower_b down, upper_b up).
            //
            // Layout dispatch (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §8): ONLY the
            // sparse layout (unstable_idx Some) stays None + zero discharge — the
            // sparse to_dense err scatter is out of scope and hard-guarded
            // downstream, and checking sparseness FIRST keeps that true even for a
            // hypothetical sparse rank-7 tensor.
            //
            // 7D explicit-rows arm: the err index is the SPEC row (axis 0, length
            // row_count == bias length, invariant I1): MAX-lift of the per-tap
            // coefficient bracket terms over the whole row (one scalar must cover
            // every coefficient of the row) and SUM-lift of the bias-widen terms
            // (every position folds into the ONE spec-row bias slot). Two deliberate
            // strengthenings vs the literal 6D mirror:
            //   (a) the widen carries the `oe·(bb+be)` CROSS term required by exact
            //       algebra (gap = -(a·β + α·bias + α·β), |α|<=oe, |β|<=be; spec R5)
            //       — the 6D arm omits `oe·be` (6D follow-up, byte-identity);
            //   (b) per lead adjudication A1 the f64 bias fold's own accumulation
            //       rounding is discharged outright: gbar·ABS with
            //       gbar = γ^f64(8·row_volume+16),
            //       ABS = |b[row]| + Σ_valid |a|·|bn_bias[ic]|, and the carried widen
            //       sum is inflated by (1+gbar) to cover its own f64
            //       nearest-summation under-estimate (gbar has >= 4x headroom over
            //       the actual γ of both folds; saturates to +INF -> outward poison).
            // Hard guards (I5/I6): a Some old err whose len != row_count is
            // Err(ShapeMismatch) => the caller's sound dense-BN fallback; a
            // non-finite or negative old_err[row] poisons the row (+INF err, +INF
            // widen, so the caller's discharge yields a -INF/+INF vacuous bias),
            // NEVER NaN; every 0·INF product is short-circuited.
            //
            // 6D dense arm: textually byte-identical to the certified design
            // (pinned by test_bn_patches_6d_coeff_err_byte_identical_regression).
            let (coeff_err, widen): (Option<Array1<f32>>, Array1<f64>) = if patches_data
                .unstable_idx
                .is_some()
            {
                (None, Array1::<f64>::zeros(new_bias.len()))
            } else if explicit_rows {
                let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                let mut gain = 0.0f64;
                for c in 0..in_c {
                    let g = f64::from(self.scale[[c]]).abs() + f64::from(self.scale_err[[c]]);
                    if g > gain {
                        gain = g;
                    }
                }
                let old = patches_data.coeff_err.as_ref();
                if let Some(e) = old {
                    if e.len() != row_count {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![row_count],
                            got: vec![e.len()],
                        });
                    }
                }
                // A1 fold-discharge factor. `row_volume` = per-row tap count;
                // the value bias fold performs <= 4·row_volume + 4 f64 nearest
                // roundings per row, each |θ| <= gbar := γ^f64(8·row_volume+16)
                // (>= 4x headroom, which also absorbs the f64 under-estimates
                // of ABS and of the carried widen sum, plus the final
                // combination roundings). Accepted regime (lead E3/F3): row
                // addend count n < 2^28 — cifar-scale rows are ~4e6, 60x under
                // it; beyond, gbar merely grows (saturating to +INF => outward
                // poison), still sound.
                let row_volume = out_c
                    .checked_mul(out_h)
                    .and_then(|x| x.checked_mul(out_w))
                    .and_then(|x| x.checked_mul(in_c))
                    .and_then(|x| x.checked_mul(kh))
                    .and_then(|x| x.checked_mul(kw))
                    .unwrap_or(usize::MAX);
                debug_assert!(
                    row_volume < (1usize << 28),
                    "BN 7D err pass: row addend count {row_volume} exceeds the \
                         documented n < 2^28 regime (still sound: gbar only grows)"
                );
                let gbar = crate::layers::linear::crown_single_gamma_n_f64(
                    row_volume.saturating_mul(8).saturating_add(16),
                );
                let mut ne = Array1::<f32>::zeros(new_bias.len());
                let mut wd = Array1::<f64>::zeros(new_bias.len());
                for row in 0..row_count {
                    // Direct index — length validated above (never the 6D-style
                    // silent `.get(i).unwrap_or(0.0)`, spec I6/R6).
                    let oe = old.map_or(0.0, |e| f64::from(e[row]));
                    if !oe.is_finite() || oe < 0.0 {
                        // Poison the row outward (I5): +INF err (degrades at
                        // consumption), +INF widen (the caller's discharge gives
                        // -INF lower / +INF upper — vacuous, NaN-free since the
                        // folded bias is finite-or-INF, never matched against a
                        // 0 factor). Skip the accumulation entirely so INF never
                        // meets a 0 multiplicand (e.g. bb + be == 0).
                        ne[row] = f32::INFINITY;
                        wd[row] = f64::INFINITY;
                        continue;
                    }
                    let mut cast = 0.0f64;
                    let mut wsum = 0.0f64;
                    // ABS is initialized with the incoming |bias|: the value
                    // fold's f64 `+=` chain starts from it, so Higham's
                    // Σ|addends| must include it.
                    let mut abs_sum = f64::from(bias_vec[row]).abs();
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                for ic in 0..in_c {
                                    let se = f64::from(self.scale_err[[ic]]);
                                    let be = f64::from(self.bias_err[[ic]]);
                                    let bb = f64::from(self.bias[[ic]]).abs();
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let coeff =
                                                f64::from(patches[[row, oc, oh, ow, ic, ki, kj]])
                                                    .abs();
                                            let ncoeff = f64::from(
                                                new_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            )
                                            .abs();
                                            // Coefficient bracket term: MAX over
                                            // ALL taps of the row, padding-zone
                                            // taps INCLUDED (they are scaled too
                                            // and only die at to_dense()), same
                                            // as the 6D arm.
                                            let t = rnd * ncoeff + coeff * se;
                                            if t > cast {
                                                cast = t;
                                            }
                                            // HOLE1: only valid (non-padding)
                                            // taps fold into the bias, mirroring
                                            // the bias loop above.
                                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                            let iw_raw =
                                                (ow * sw + kj) as isize - pad_left as isize;
                                            if ih_raw >= 0
                                                && (ih_raw as usize) < in_h
                                                && iw_raw >= 0
                                                && (iw_raw as usize) < in_w
                                            {
                                                // Per-tap exact algebra incl. the
                                                // oe·be cross term (spec R5):
                                                // |a·b - a_true·b_real|
                                                //   <= |a|·be + oe·(bb + be).
                                                // oe == 0 short-circuits so a
                                                // degenerate +INF channel bias
                                                // cannot make 0·INF = NaN (I5).
                                                let cross =
                                                    if oe == 0.0 { 0.0 } else { oe * (bb + be) };
                                                wsum += coeff * be + cross;
                                                // A zero stored coefficient
                                                // contributes exactly 0 to the
                                                // value fold; skip it so 0·INF
                                                // (degenerate bb) cannot poison
                                                // ABS with NaN (I5).
                                                if coeff != 0.0 {
                                                    abs_sum += coeff * bb;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // oe == 0 short-circuit: gain can be +INF (degenerate scale
                    // channel) and 0·INF = NaN (I5).
                    let ne_val = if oe == 0.0 { cast } else { cast + gain * oe };
                    let ne_f32 = next_up_f32(ne_val as f32);
                    // Defensive: the err channel is "finite or +INF", never NaN.
                    // (A NaN stored coefficient — the pre-existing value-path
                    // 0·INF — is skipped by the max above since NaN comparisons
                    // are false; keep the emission NaN-free regardless.)
                    ne[row] = if ne_f32.is_nan() {
                        f32::INFINITY
                    } else {
                        ne_f32
                    };
                    // Widen: carried terms inflated by (1+gbar) (covers wsum's
                    // own f64 summation under-estimate) + the A1 fold discharge
                    // gbar·ABS. Zero operands are short-circuited before the
                    // possibly-saturated (+INF) gbar so the correct limit of a
                    // zero sum stays 0 — pure carry, adjudication C2 analog.
                    let carried = if wsum == 0.0 {
                        0.0
                    } else {
                        wsum * (1.0 + gbar)
                    };
                    let fold = if abs_sum == 0.0 { 0.0 } else { gbar * abs_sum };
                    let w = carried + fold;
                    // Residual non-finite/negative (NaN compares false) maps to
                    // +INF: outward poison, never NaN (I5).
                    wd[row] = if w >= 0.0 { w } else { f64::INFINITY };
                }
                (Some(ne), wd)
            } else {
                let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                let mut gain = 0.0f64;
                for c in 0..in_c {
                    let g = f64::from(self.scale[[c]]).abs() + f64::from(self.scale_err[[c]]);
                    if g > gain {
                        gain = g;
                    }
                }
                let old = patches_data.coeff_err.as_ref();
                let mut ne = Array1::<f32>::zeros(new_bias.len());
                let mut wd = Array1::<f64>::zeros(new_bias.len());
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            let oe =
                                old.map_or(0.0, |e| f64::from(e.get(n).copied().unwrap_or(0.0)));
                            let mut cast = 0.0f64;
                            let mut wsum = 0.0f64;
                            for ic in 0..in_c {
                                let se = f64::from(self.scale_err[[ic]]);
                                let be = f64::from(self.bias_err[[ic]]);
                                let bb = f64::from(self.bias[[ic]]).abs();
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let coeff =
                                            f64::from(patches[[oc, oh, ow, ic, ki, kj]]).abs();
                                        let ncoeff =
                                            f64::from(new_patches[[oc, oh, ow, ic, ki, kj]]).abs();
                                        let t = rnd * ncoeff + coeff * se;
                                        if t > cast {
                                            cast = t;
                                        }
                                        // HOLE1: only valid (non-padding) taps
                                        // fold into the bias, mirroring the bias
                                        // loop above.
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw >= 0
                                            && (ih_raw as usize) < in_h
                                            && iw_raw >= 0
                                            && (iw_raw as usize) < in_w
                                        {
                                            wsum += coeff * be + oe * bb;
                                        }
                                    }
                                }
                            }
                            ne[n] = next_up_f32((cast + gain * oe) as f32);
                            wd[n] = wsum;
                        }
                    }
                }
                (Some(ne), wd)
            };
            let new_data = PatchesData {
                coeff_err,
                patches: Some(new_patches),
                stride: patches_data.stride,
                padding: patches_data.padding,
                identity: false,
                output_shape: patches_data.output_shape,
                input_shape: patches_data.input_shape,
                unstable_idx: None,
            };
            // HOLE1 bias discharge (`widen`) + directed rounding (#1745) are applied
            // by the caller (lower_b -= widen then round down, upper_b += widen up).
            Ok((new_data, new_bias, widen))
        };

        let (new_lower_a, new_lower_b, widen_lower) =
            process_patches(&bounds.lower_a, &bounds.lower_b)?;
        let (new_upper_a, new_upper_b, widen_upper) =
            process_patches(&bounds.upper_a, &bounds.upper_b)?;

        // Discharge the HOLE1 bias widening OUTWARD, then directed rounding (#1745):
        // lower bounds subtract widen and round down, upper bounds add widen and
        // round up. Each branch uses its own path's widen (lower_a's vs upper_a's
        // coeffs/old_err). widen >= 0, so this only ever loosens the bound; the
        // single f64->f32 cast per cell preserves the #1745 soundness.
        let new_lower_b = ndarray::Zip::from(&new_lower_b)
            .and(&widen_lower)
            .map_collect(|&b, &w| next_down_f32((b - w) as f32));
        let new_upper_b = ndarray::Zip::from(&new_upper_b)
            .and(&widen_upper)
            .map_collect(|&b, &w| next_up_f32((b + w) as f32));

        Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })))
    }
}
