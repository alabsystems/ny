// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward for AveragePool.
//!
//! AvgPool is linear: y = (1/k) * sum x_i over the pooling window.
//! The Patches backward upsamples patch coefficients by the pool kernel size
//! (nearest-neighbor), then divides by pool_size. This is equivalent to
//! replicating each coefficient across the pooling window positions.
//!
//! Reference: alpha-beta-CROWN auto_LiRPA/operators/pooling.py:584-601
//! Reference: designs/2026-03-01-patches-phase3-pooling-termination.md Section 1
//! Part of #2613

use ndarray::{Array1, Axis};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
use crate::bounds::patches_ops::nearest_neighbor_upsample_last2;
use crate::layers::common::PatchesPropagation;

use super::average::AveragePoolLayer;

impl PatchesPropagation for AveragePoolLayer {
    fn propagate_patches(&self, _bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        // AvgPool needs pre-activation shape for input_shape field.
        // Delegate to propagate_patches_with_bounds which has access to pre_activation.
        Err(NyError::UnsupportedOp(
            "AveragePool Patches requires pre-activation shape - use propagate_patches_with_bounds"
                .to_string(),
        ))
    }

    fn propagate_patches_with_bounds(
        &self,
        bounds: &PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<CrownBounds> {
        // Guard: only support kernel_size == stride (non-overlapping windows)
        // Reference: alpha-beta-CROWN pooling.py:583 (stride != kernel_size -> no patches support)
        if self.kernel_size != self.stride {
            return Err(NyError::UnsupportedOp(
                "AvgPool Patches requires kernel_size == stride".into(),
            ));
        }
        // Guard: count_include_pad must be true (alpha-beta-CROWN default)
        if !self.count_include_pad {
            return Err(NyError::UnsupportedOp(
                "AvgPool Patches requires count_include_pad=true".into(),
            ));
        }
        // Guard: global pooling not supported in Patches mode
        if self.is_global() {
            return Err(NyError::UnsupportedOp(
                "Global AvgPool not supported in Patches mode".into(),
            ));
        }

        let (pool_kh, pool_kw) = self.kernel_size;
        let (pool_sh, pool_sw) = self.stride;
        let (pool_ph, pool_pw) = self.padding;
        let pool_size = (pool_kh * pool_kw) as f32;

        // Extract AvgPool input shape from pre_activation
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let (channels, in_h, in_w) = if ndim == 3 {
            (input_shape[0], input_shape[1], input_shape[2])
        } else if ndim == 4 {
            (input_shape[1], input_shape[2], input_shape[3])
        } else {
            return Err(NyError::InvalidSpec(format!(
                "AvgPool Patches requires 3D or 4D input, got {}D",
                ndim
            )));
        };

        // Spec-row count for the 7D explicit-rows layout (axis 0 must match it;
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9).
        let spec_rows = bounds.row_count;

        let upsample_patches = |data: &PatchesData| -> Result<PatchesData> {
            let materialized = if data.identity {
                data.materialize_identity()
            } else {
                data.clone()
            };
            let patches_tensor = materialized.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;

            let upsampled = nearest_neighbor_upsample_last2(patches_tensor, pool_kh, pool_kw)?;
            // Divide by pool_size for averaging
            let patches = upsampled / pool_size;

            // Certified coefficient error (#patches-coeff-err-soundness,
            // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9 A1). AvgPool divides each
            // coefficient by pool_size in round-to-nearest f32 (accumulation depth 1;
            // the nearest-neighbor upsample only replicates, no arithmetic) and the
            // backward is row-preserving, hence with gain 1/pool_size:
            //   new_err[r] = next_up(γ_1^f32·RowMaxAbs(divided@r) + old_err[r]/pool_size).
            // RowMaxAbs uses the STORED (already-divided) coeffs and γ_1 = u/(1-u), so
            // γ_1·|stored| over-bounds |fl(v/pool)-v/pool| for every coeff in the row.
            // 6D dense: logical row = output position (oc,oh,ow), err length = the
            // output grid. 7D explicit-rows: logical row = SPEC row (axis 0), err
            // length = row_count; the row max reduces over ALL positions of the spec
            // row (MAX-lift, spec I1), old_err is spec-row indexed after a hard
            // length check (I6), and non-finite/negative carried err poisons to +INF
            // (I5). Sparse (unstable_idx Some) stays None (out of scope, spec I2).
            // This is orthogonal to the outward division-error bias fold below, which
            // is retained as-is (its behavior is pinned by an existing test);
            // together they double-count the same rounding, which only widens bounds
            // and is therefore sound (spec F6).
            let coeff_err = if data.unstable_idx.is_some() {
                None
            } else {
                match patches.shape().len() {
                    6 => {
                        let shape = patches.shape();
                        let (oc_e, oh_e, ow_e, in_c_e, kh_e, kw_e) =
                            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
                        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
                        let pool_size_f64 = pool_size as f64;
                        let old = data.coeff_err.as_ref();
                        let mut ne = Array1::<f32>::zeros(oc_e * oh_e * ow_e);
                        for o_c in 0..oc_e {
                            for o_h in 0..oh_e {
                                for o_w in 0..ow_e {
                                    let mut rowmax = 0.0f64;
                                    for i_c in 0..in_c_e {
                                        for ki in 0..kh_e {
                                            for kj in 0..kw_e {
                                                let a = f64::from(
                                                    patches[[o_c, o_h, o_w, i_c, ki, kj]],
                                                )
                                                .abs();
                                                if a > rowmax {
                                                    rowmax = a;
                                                }
                                            }
                                        }
                                    }
                                    let flat = o_c * oh_e * ow_e + o_h * ow_e + o_w;
                                    let oe = old.map_or(0.0, |e| {
                                        f64::from(e.get(flat).copied().unwrap_or(0.0))
                                    });
                                    ne[flat] =
                                        next_up_f32((gamma1 * rowmax + oe / pool_size_f64) as f32);
                                }
                            }
                        }
                        Some(ne)
                    }
                    7 => {
                        let rows = patches.shape()[0];
                        // Hard guards (spec I6/§14 B5): construction-bug-class
                        // mismatches return Err ⇒ the caller's sound dense
                        // fallback — NEVER a silent `.get().unwrap_or(0.0)`
                        // under-count (the false-proof direction).
                        if rows != spec_rows {
                            return Err(NyError::ShapeMismatch {
                                expected: vec![spec_rows],
                                got: vec![rows],
                            });
                        }
                        if let Some(e) = data.coeff_err.as_ref() {
                            if e.len() != rows {
                                return Err(NyError::ShapeMismatch {
                                    expected: vec![rows],
                                    got: vec![e.len()],
                                });
                            }
                        }
                        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
                        let pool_size_f64 = pool_size as f64;
                        let old = data.coeff_err.as_ref();
                        let mut ne = Array1::<f32>::zeros(rows);
                        for (row, slab) in patches.axis_iter(Axis(0)).enumerate() {
                            // MAX-lift (spec I1): the single spec-row scalar must
                            // cover every coefficient of the row, so the max spans
                            // ALL positions (order-independent reduction, spec I8).
                            let mut rowmax = 0.0f64;
                            for &v in slab.iter() {
                                let a = f64::from(v).abs();
                                if a > rowmax {
                                    rowmax = a;
                                }
                            }
                            // Sanitize the carried err (spec I5): non-finite or
                            // negative maps to +INF (degrade poison), never to 0.
                            let oe = match old {
                                None => 0.0f64,
                                Some(e) => {
                                    let v = e[row];
                                    if v.is_finite() && v >= 0.0 {
                                        f64::from(v)
                                    } else {
                                        f64::INFINITY
                                    }
                                }
                            };
                            // γ_1 is finite and both addends are ≥ 0 (possibly
                            // +INF), so the sum is never NaN; +INF survives the
                            // outward cast (next_up_f32(+INF) = +INF) and degrades
                            // the row at consumption. Emitted even for old == None:
                            // the division rounding is intrinsic (spec R2 analog).
                            ne[row] = next_up_f32((gamma1 * rowmax + oe / pool_size_f64) as f32);
                        }
                        Some(ne)
                    }
                    // Unreachable: the upsample above already rejected non-{6,7}
                    // layouts with a hard error (spec §14 F2).
                    _ => None,
                }
            };

            Ok(PatchesData {
                coeff_err,
                patches: Some(patches),
                // Stride composition: new_stride = patches.stride * pool_stride
                stride: (data.stride.0 * pool_sh, data.stride.1 * pool_sw),
                // Padding composition (simplified, no inserted_zeros):
                // new_padding = patches.padding * pool_stride + pool_padding
                // Reference: alpha-beta-CROWN patches.py:354
                padding: (
                    data.padding.0 * pool_sw + pool_pw, // left
                    data.padding.1 * pool_sw + pool_pw, // right
                    data.padding.2 * pool_sh + pool_ph, // top
                    data.padding.3 * pool_sh + pool_ph, // bottom
                ),
                identity: false,
                output_shape: data.output_shape,
                input_shape: (channels, in_h, in_w),
                unstable_idx: None,
            })
        };

        let lower_a = upsample_patches(&bounds.lower_a)?;
        let upper_a = upsample_patches(&bounds.upper_a)?;

        // SOUND division-rounding error (#avgpool-patches / #vnncomp-aw-soundness).
        // The patch coefficients are divided by `pool_size` in round-to-nearest f32
        // (`upsampled / pool_size` — one f32 op per coefficient, accumulation depth 1;
        // the nearest-neighbor upsample only replicates, no arithmetic), so each
        // stored coefficient carries a fresh rounding error of at most
        // gamma_1*|coeff| (gamma_1 = gamma_1^f32 = 2^-24/(1-2^-24)). Per logical row
        // r (6D output position / 7D spec row) fold
        //   err_r = next_up_f32(gamma_1 * X * L1_r),  X = max pre-activation |.|,
        //   L1_r = sum over the row of |divided coeff| (row L1),
        // OUTWARD: lower_b[r] -= err_r, upper_b[r] += err_r. By the triangle inequality
        // |sum_j d_coeff_j * h_in_j| <= gamma_1*X*L1_r over the full receptive field
        // under EVERY sign pattern (cancellation), and the bias travels additively
        // (never rescaled) through the remaining backward layers, so this is sound at
        // the final concretize. The per-row `coeff_err` channel above now ALSO
        // certifies this rounding; the double-count is retained deliberately (its
        // behavior is pinned by an existing test) and only widens (spec F6).
        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        let mut x_mag = 0.0f64;
        for (&l, &u) in pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
        {
            let m = (l as f64).abs().max((u as f64).abs());
            if m > x_mag {
                x_mag = m;
            }
        }

        let mut lower_b = bounds.lower_b.clone();
        let mut upper_b = bounds.upper_b.clone();
        fold_division_err_into_bias(&lower_a, gamma1, x_mag, &mut lower_b, true)?;
        fold_division_err_into_bias(&upper_a, gamma1, x_mag, &mut upper_b, false)?;

        let result = PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        };

        // If patches now cover entire input, fall back to Dense
        if result.lower_a.should_fallback_to_dense() {
            Ok(CrownBounds::Dense(result.to_dense()?))
        } else {
            Ok(CrownBounds::Patches(Box::new(result)))
        }
    }
}

/// Fold the f32 division-rounding error of the divided AvgPool patches OUTWARD into
/// the bias. (Historically the only sound home; the per-row `coeff_err` channel now
/// ALSO certifies the same division rounding — the deliberate double-count only
/// widens and the bias fold's behavior is pinned by an existing test, spec F6.)
///
/// `data.patches` is the divided 6D tensor `(oc, oh, ow, in_c, kh, kw)` or 7D
/// explicit-rows tensor `(rows, oc, oh, ow, in_c, kh, kw)`. Per logical row this
/// computes the row L1 norm `L1_r = Σ |coeff|`, the certified outward error
/// `err_r = next_up_f32(gamma1 * x_mag * L1_r)`, and applies it to the row's bias
/// slot: subtract for the lower bias (rounded DOWN), add for the upper bias
/// (rounded UP). 6D: logical row = output position `(oc, oh, ow)`, slot
/// `flat = oc*(oh*ow) + oh*ow_idx + ow_idx`. 7D: logical row = SPEC row (axis 0);
/// every position of the row folds into the ONE row slot, so the L1 sums the WHOLE
/// row slab (SUM-lift, spec I1; padding taps multiply structurally-zero inputs, so
/// including them only widens, spec §9.2).
/// Returns an error (→ sound dense fallback in the caller) if the divided patches
/// are not a dense {6,7}D layout or the bias length disagrees with the row count,
/// so a malformed layout can never index out of bounds or silently drop error.
fn fold_division_err_into_bias(
    data: &PatchesData,
    gamma1: f64,
    x_mag: f64,
    bias: &mut Array1<f32>,
    subtract: bool,
) -> Result<()> {
    let patches = data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("AvgPool patches: divided patches tensor is None".into())
    })?;
    let shape = patches.shape();
    match shape.len() {
        6 => {
            let (oc, oh, ow, in_c, kh, kw) =
                (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
            let n_rows = oc
                .checked_mul(oh)
                .and_then(|v| v.checked_mul(ow))
                .ok_or_else(|| {
                    NyError::InvalidSpec("AvgPool patches: output grid overflow".into())
                })?;
            if bias.len() != n_rows {
                return Err(NyError::ShapeMismatch {
                    expected: vec![n_rows],
                    got: vec![bias.len()],
                });
            }
            for o_c in 0..oc {
                for o_h in 0..oh {
                    for o_w in 0..ow {
                        let mut l1 = 0.0f64;
                        for i_c in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    l1 += (patches[[o_c, o_h, o_w, i_c, ki, kj]] as f64).abs();
                                }
                            }
                        }
                        let err = next_up_f32((gamma1 * x_mag * l1) as f32);
                        let flat = o_c * oh * ow + o_h * ow + o_w;
                        let b = bias[flat] as f64;
                        bias[flat] = if subtract {
                            next_down_f32((b - err as f64) as f32)
                        } else {
                            next_up_f32((b + err as f64) as f32)
                        };
                    }
                }
            }
        }
        7 => {
            let n_rows = shape[0];
            if bias.len() != n_rows {
                return Err(NyError::ShapeMismatch {
                    expected: vec![n_rows],
                    got: vec![bias.len()],
                });
            }
            // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §14 E3/F3: the f64 L1 fold
            // rounding is dominated by the outward casts only while the per-row
            // addend count stays well under 2^28 (cifar-scale rows are ~4e6,
            // 60x under the bound).
            debug_assert!(
                n_rows == 0 || patches.len() / n_rows < (1usize << 28),
                "AvgPool 7D division-err fold: row addend count {} breaches the \
                 documented n < 2^28 rounding-dominance bound",
                patches.len().checked_div(n_rows).unwrap_or(0)
            );
            for (row, slab) in patches.axis_iter(Axis(0)).enumerate() {
                // Serial row-major slab walk = the fixed tap order
                // oc -> oh -> ow -> (ic, ki, kj) (spec I8); SUM over ALL
                // positions of the spec row (SUM-lift, spec I1).
                let mut l1 = 0.0f64;
                for &v in slab.iter() {
                    l1 += f64::from(v).abs();
                }
                let err = next_up_f32((gamma1 * x_mag * l1) as f32);
                let b = bias[row] as f64;
                bias[row] = if subtract {
                    next_down_f32((b - err as f64) as f32)
                } else {
                    next_up_f32((b + err as f64) as f32)
                };
            }
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![6, 7],
                got: vec![shape.len()],
            });
        }
    }
    Ok(())
}

// --- 7D explicit-rows coeff_err closure: 6D byte-identity pin ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9.4 T3; validation gate §13 item 1)
#[cfg(test)]
mod bitwise_regression_pins {
    use super::*;
    use ndarray::{Array1, ArrayD, IxDyn};

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

    /// BYTE-IDENTITY PIN: the 6D dense AvgPool patches backward WITH incoming
    /// `coeff_err: Some` on both sides must stay bit-for-bit unchanged by the
    /// 7D explicit-rows coeff_err closure (spec §9.3 keeps the 6D arms of the
    /// upsample dispatch, the per-row err rule, and `fold_division_err_into_bias`
    /// textually VERBATIM; only new 7D arms are added).
    ///
    /// Committed and verified green against the UNMODIFIED (pre-closure) tree;
    /// bit literals captured from that tree. Must pass unmodified after the
    /// closure lands. The existing pinned
    /// `avgpool_patches_division_error_folded_into_bias` stays untouched.
    ///
    /// Fixture: pool (1,3)/(1,3) so the division by pool_size=3 ROUNDS (a 2x2
    /// pool divides by 4 exactly and would pin nothing); non-dyadic mixed-sign
    /// coefficients; asymmetric per-side errs including exact-0.0 rows.
    #[test]
    fn pooling_6d_bitwise_regression_avg() {
        let avgpool = AveragePoolLayer::new((1, 3), (1, 3), (0, 0), true);

        // Pre-activation (1, 2, 6) -> avgpool output (1, 2, 2). x_mag = 3.1.
        let pre_lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 6]),
            vec![
                -2.5_f32, -1.75, 0.3, -0.9, -1.2, -3.1, -0.6, -2.2, -1.4, 0.15, -0.85, -1.9,
            ],
        )
        .unwrap();
        let pre_upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 6]),
            vec![
                2.75_f32, 0.4, 1.9, 0.7, 2.1, -0.2, 1.3, 0.95, 2.6, 1.75, 0.5, 0.35,
            ],
        )
        .unwrap();
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        // Incoming 6D patches [oc=1, oh=2, ow=2, ic=1, kh=1, kw=1] indexing the
        // avgpool OUTPUT grid (1, 2, 2); 4 spec rows.
        let make_side = |vals: Vec<f32>, err: Vec<f32>| PatchesData {
            coeff_err: Some(Array1::from_vec(err)),
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1, 1, 1]), vals).unwrap()),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 2, 2),
            input_shape: (1, 2, 2),
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

        let result = avgpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("avgpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => {
                panic!("expected Patches mode (upsampled kernel area 3 < input area 12)")
            }
        };

        // Metadata composition (layout-independent, must not move either).
        assert_eq!(pb.row_count, 4);
        assert_eq!(pb.lower_a.stride, (1, 3));
        assert_eq!(pb.lower_a.padding, (0, 0, 0, 0));
        assert_eq!(pb.lower_a.output_shape, (1, 2, 2));
        assert_eq!(pb.lower_a.input_shape, (1, 2, 6));
        assert_eq!(
            pb.lower_a.patches.as_ref().unwrap().shape(),
            &[1, 2, 2, 1, 1, 3]
        );
        assert_eq!(
            pb.upper_a.patches.as_ref().unwrap().shape(),
            &[1, 2, 2, 1, 1, 3]
        );

        // Bit literals captured from pre-change HEAD (see doc comment).
        const EXP_LOWER_PATCHES: [u32; 12] = [
            0x3E6EEEEF, 0x3E6EEEEF, 0x3E6EEEEF, 0xBEDDDDDD, 0xBEDDDDDD, 0xBEDDDDDD, 0x3E3BBBBC,
            0x3E3BBBBC, 0x3E3BBBBC, 0x3F4CCCCD, 0x3F4CCCCD, 0x3F4CCCCD,
        ];
        const EXP_UPPER_PATCHES: [u32; 12] = [
            0xBDEEEEEF, 0xBDEEEEEF, 0xBDEEEEEF, 0x3E911111, 0x3E911111, 0x3E911111, 0x3EC44444,
            0x3EC44444, 0x3EC44444, 0xBE5DDDDD, 0xBE5DDDDD, 0xBE5DDDDD,
        ];
        const EXP_LOWER_B: [u32; 4] = [0x3DCCCCBB, 0xBE4CCCDE, 0x3E999996, 0xBECCCCDD];
        const EXP_UPPER_B: [u32; 4] = [0x3F000002, 0x3F19999E, 0xBF33332E, 0x3F4CCCD0];
        const EXP_LOWER_ERR: [u32; 4] = [0x39AEC51E, 0x32DDDDDF, 0x392EC62F, 0x3A2EC673];
        const EXP_UPPER_ERR: [u32; 4] = [0x3A2EC3B7, 0x3974AF7A, 0x32C44446, 0x380BDD44];

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
        let lower_err: Vec<f32> = pb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("6D avgpool backward must emit lower coeff_err")
            .to_vec();
        let upper_err: Vec<f32> = pb
            .upper_a
            .coeff_err
            .as_ref()
            .expect("6D avgpool backward must emit upper coeff_err")
            .to_vec();

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

// --- 7D explicit-rows coeff_err closure: oracle + guard tests ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §9.4 T1/T5)
#[cfg(test)]
mod coeff_err_7d_tests {
    use super::*;
    use ndarray::{Array1, ArrayD, IxDyn};

    // Oracle-noise note (spec §9.4): the f64 oracle computations below carry
    // ~2^-53 relative rounding, while every emitted err/bias includes >= 2^-25
    // relative outward slack from next_up/next_down at the f32 cast — so the
    // strict <=/>= comparisons cannot flip on oracle noise.

    /// avgpool (1,3)/(1,3) over input (1,2,6) -> output (1,2,2); pool_size = 3,
    /// so the coefficient division ROUNDS (a 2x2 pool divides by 4 exactly and
    /// would test nothing). x_mag = 3.0 exactly.
    fn avgpool_fixture() -> (AveragePoolLayer, BoundedTensor) {
        let avgpool = AveragePoolLayer::new((1, 3), (1, 3), (0, 0), true);
        let pre_lower = ArrayD::from_elem(IxDyn(&[1, 2, 6]), -3.0f32);
        let pre_upper = ArrayD::from_elem(IxDyn(&[1, 2, 6]), 3.0f32);
        let pre = BoundedTensor::new(pre_lower, pre_upper).unwrap();
        (avgpool, pre)
    }

    /// Incoming 7D explicit-rows side [rows=2, oc=1, oh=2, ow=2, ic=1, kh=1,
    /// kw=1]: 1x1 taps over the avgpool OUTPUT grid (1,2,2); 2 spec rows of 4
    /// positions each.
    fn make_side_7d(vals: Vec<f32>, err: Option<Vec<f32>>) -> PatchesData {
        PatchesData {
            coeff_err: err.map(Array1::from_vec),
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2, 1, 1, 1]), vals).unwrap()),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 2, 2),
            input_shape: (1, 2, 2),
            unstable_idx: None,
        }
    }

    /// Per-row carried err from an explicit truth model:
    /// `e_r = next_up(max_row |f64(stored) - true|)` — the tightest legal
    /// carried err for `stored = true as f32`.
    fn row_err(stored: &[f32], truth: &[f64], row: usize) -> f32 {
        let mut m = 0.0f64;
        for i in 4 * row..4 * row + 4 {
            m = m.max((f64::from(stored[i]) - truth[i]).abs());
        }
        next_up_f32(m as f32)
    }

    /// §9.4 T1: (a) every expanded (upsampled+divided) coefficient is covered
    /// by the emitted spec-row err against an exact f64 truth; (b) the err
    /// formula pins bitwise against an independently recomputed row max M_r;
    /// (c) the division-error bias fold pins bitwise against an independently
    /// recomputed whole-row L1_r (SUM-lift over ALL positions of the row).
    #[test]
    fn avgpool_7d_coeff_err_covers_true_deviation() {
        let (avgpool, pre) = avgpool_fixture();

        // Exact f64 truths, all inexact in f32 so quantization gives genuine
        // nonzero per-row deviation.
        let true_l: Vec<f64> = vec![0.1, -0.7, 1.3, 2.9, -1.7, 0.23, -0.61, 3.7];
        let true_u: Vec<f64> = vec![-0.35, 0.85, 1.15, -0.65, 0.9, -2.3, 0.55, 1.05];
        let stored_l: Vec<f32> = true_l.iter().map(|&v| v as f32).collect();
        let stored_u: Vec<f32> = true_u.iter().map(|&v| v as f32).collect();
        let e_l: Vec<f32> = (0..2).map(|r| row_err(&stored_l, &true_l, r)).collect();
        let e_u: Vec<f32> = (0..2).map(|r| row_err(&stored_u, &true_u, r)).collect();
        assert!(
            e_l.iter().chain(e_u.iter()).any(|&e| e > 0.0),
            "fixture must exercise nonzero carried err"
        );

        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(stored_l, Some(e_l.clone())),
            lower_b: Array1::from_vec(vec![0.1_f32, -0.2]),
            upper_a: make_side_7d(stored_u, Some(e_u.clone())),
            upper_b: Array1::from_vec(vec![0.5_f32, 0.6]),
        };

        let result = avgpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("7D avgpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => {
                panic!("expected Patches mode (upsampled kernel area 3 < input area 12)")
            }
        };
        assert_eq!(pb.row_count, 2);
        assert_eq!(
            pb.lower_a.patches.as_ref().unwrap().shape(),
            &[2, 1, 2, 2, 1, 1, 3]
        );

        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        let x_mag = 3.0f64;
        let pool_size_f64 = f64::from(3.0f32);

        let check_side = |data: &PatchesData,
                          bias: &Array1<f32>,
                          b0: &[f32],
                          truth: &[f64],
                          e_old: &[f32],
                          is_lower: bool| {
            let pt = data.patches.as_ref().unwrap();
            let err = data
                .coeff_err
                .as_ref()
                .expect("7D avgpool backward must emit coeff_err");
            assert_eq!(err.len(), 2, "err length must be row_count");
            for &e in err.iter() {
                assert!(e.is_finite() && e >= 0.0, "err must be finite >= 0: {e}");
            }
            for row in 0..2usize {
                let er = f64::from(err[row]);
                // (a) coverage: every expanded cell (oh, ow, kj) replicates the
                // single source tap (row, 0, oh, ow, 0, 0, 0); the true divided
                // coefficient is truth/3 exactly (3 is exact in f64 up to one
                // ~2^-53 division rounding — oracle noise, see mod note).
                let mut m_r = 0.0f64;
                for oh in 0..2usize {
                    for ow in 0..2usize {
                        let flat = row * 4 + oh * 2 + ow;
                        let true_div = truth[flat] / 3.0f64;
                        for kj in 0..3usize {
                            let stored_div = f64::from(pt[[row, 0, oh, ow, 0, 0, kj]]);
                            assert!(
                                (stored_div - true_div).abs() <= er,
                                "row {row} cell ({oh},{ow},{kj}): |{stored_div} - {true_div}| \
                                 > err {er}"
                            );
                            m_r = m_r.max(stored_div.abs());
                        }
                    }
                }
                // (b) formula pin: err[row] == next_up((γ1·M_r + e_old/pool) as f32).
                let expected =
                    next_up_f32((gamma1 * m_r + f64::from(e_old[row]) / pool_size_f64) as f32);
                assert_eq!(
                    err[row].to_bits(),
                    expected.to_bits(),
                    "row {row}: err formula pin"
                );
                // (c) bias pin: independently recompute the WHOLE-ROW L1 (sum
                // over all positions, row-major order) and the outward apply.
                let slab = pt.index_axis(Axis(0), row);
                let mut l1 = 0.0f64;
                for &v in slab.iter() {
                    l1 += f64::from(v).abs();
                }
                let e_div = next_up_f32((gamma1 * x_mag * l1) as f32);
                let b = f64::from(b0[row]);
                let expected_b = if is_lower {
                    next_down_f32((b - e_div as f64) as f32)
                } else {
                    next_up_f32((b + e_div as f64) as f32)
                };
                assert_eq!(
                    bias[row].to_bits(),
                    expected_b.to_bits(),
                    "row {row}: bias fold pin (is_lower={is_lower})"
                );
            }
        };
        check_side(&pb.lower_a, &pb.lower_b, &[0.1, -0.2], &true_l, &e_l, true);
        check_side(&pb.upper_a, &pb.upper_b, &[0.5, 0.6], &true_u, &e_u, false);
    }

    /// A `None` incoming err still emits `Some` on 7D: the division rounding is
    /// intrinsic (γ1·M_r term with oe = 0).
    #[test]
    fn avgpool_7d_none_err_still_emits_division_err() {
        let (avgpool, pre) = avgpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::from_vec(vec![0.0_f32, 0.0]),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::from_vec(vec![0.0_f32, 0.0]),
        };
        let result = avgpool
            .propagate_patches_with_bounds(&bounds, &pre)
            .expect("7D avgpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => panic!("expected Patches mode"),
        };
        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        for data in [&pb.lower_a, &pb.upper_a] {
            let pt = data.patches.as_ref().unwrap();
            let err = data
                .coeff_err
                .as_ref()
                .expect("None-in must still emit Some on 7D (intrinsic division rounding)");
            for row in 0..2usize {
                let slab = pt.index_axis(Axis(0), row);
                let mut m_r = 0.0f64;
                for &v in slab.iter() {
                    m_r = m_r.max(f64::from(v).abs());
                }
                let expected = next_up_f32((gamma1 * m_r + 0.0f64 / f64::from(3.0f32)) as f32);
                assert_eq!(err[row].to_bits(), expected.to_bits(), "row {row}");
                assert!(err[row] > 0.0, "nonzero row must carry division err");
            }
        }
    }

    /// §9.4 T5: carried err whose length is not row_count => hard Err
    /// (spec I6), never a silent under-count.
    #[test]
    fn avg_7d_coeff_err_length_mismatch_errors() {
        let (avgpool, pre) = avgpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), Some(vec![1e-3, 2e-3, 3e-3])),
            lower_b: Array1::from_vec(vec![0.0_f32, 0.0]),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::from_vec(vec![0.0_f32, 0.0]),
        };
        let result = avgpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for err len 3 vs row_count 2, got {result:?}"
        );
    }

    /// §9.4 T5: bias length disagreeing with the spec-row count => hard Err
    /// (the division-err fold's 7D guard).
    #[test]
    fn avg_7d_bias_length_mismatch_errors() {
        let (avgpool, pre) = avgpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::from_vec(vec![0.0_f32, 0.0, 0.0]),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::from_vec(vec![0.0_f32, 0.0, 0.0]),
        };
        let result = avgpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for bias len 3 vs 2 spec rows, got {result:?}"
        );
    }

    /// §9.4 T5: 7D tensor whose axis 0 disagrees with row_count => hard Err.
    #[test]
    fn avg_7d_row_count_mismatch_errors() {
        let (avgpool, pre) = avgpool_fixture();
        let vals: Vec<f32> = vec![0.7, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65];
        let bounds = PatchesLinearBounds {
            row_count: 3,
            lower_a: make_side_7d(vals.clone(), None),
            lower_b: Array1::from_vec(vec![0.0_f32, 0.0, 0.0]),
            upper_a: make_side_7d(vals, None),
            upper_b: Array1::from_vec(vec![0.0_f32, 0.0, 0.0]),
        };
        let result = avgpool.propagate_patches_with_bounds(&bounds, &pre);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "expected ShapeMismatch for tensor rows 2 vs row_count 3, got {result:?}"
        );
    }
}

#[cfg(test)]
mod division_err_soundness_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// #avgpool-patches / #vnncomp-aw-soundness: the f32 division `upsampled /
    /// pool_size` in the Patches AvgPool backward must fold its rounding error OUTWARD
    /// into the bias (PatchesData has no per-coefficient error field). Pre-fix the
    /// divided coefficients were stored with NO certified error and the bias passed
    /// through unchanged — a false-proof. This pins the certified term, which bounds
    /// the worst-case division rounding (gamma_1*|coeff|*X) whether or not a
    /// particular division happens to round.
    #[test]
    fn avgpool_patches_division_error_folded_into_bias() {
        let avgpool = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), true);
        let identity = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1000.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1000.0f32);
        let pre = BoundedTensor::new(lower, upper).unwrap();

        let result = avgpool
            .propagate_patches_with_bounds(&identity, &pre)
            .expect("avgpool patches backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => panic!("expected Patches mode (kernel area < input area)"),
        };

        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        let expected_err = next_up_f32((gamma1 * 1000.0_f64 * 1.0_f64) as f32);
        let expected_lower = next_down_f32((0.0_f64 - expected_err as f64) as f32);
        let expected_upper = next_up_f32((0.0_f64 + expected_err as f64) as f32);
        assert!(
            expected_err > 0.0,
            "sanity: certified error must be positive"
        );

        for i in 0..pb.lower_b.len() {
            assert!(
                pb.lower_b[i] < 0.0,
                "row {i}: lower bias not pushed DOWN by division error: {}",
                pb.lower_b[i]
            );
            assert!(
                pb.upper_b[i] > 0.0,
                "row {i}: upper bias not pushed UP by division error: {}",
                pb.upper_b[i]
            );
            assert_eq!(pb.lower_b[i], expected_lower, "row {i}: lower bias");
            assert_eq!(pb.upper_b[i], expected_upper, "row {i}: upper bias");
        }
    }
}
