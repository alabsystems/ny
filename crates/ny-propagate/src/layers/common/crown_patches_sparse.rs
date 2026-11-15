// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use super::compose;
use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};

/// Sparse 4D/5D activation composition is not authoritative until it emits a
/// coefficient-error receipt for its own rounded products.
#[inline]
fn require_sparse_coeff_error_transport() -> Result<()> {
    Err(NyError::UnsupportedConfiguration(
        "sparse Patches activation requires the unimplemented 4D/5D coefficient-error channel"
            .into(),
    ))
}

/// The EXACT real product `compose_lower` intends to store, in binary64.
///
/// `f32 * f32` is exact in binary64 (24 + 24 <= 53 significand bits), so this is
/// the true value the stored binary32 coefficient approximates — not an estimate
/// of it. That is what lets the channel emit a MEASURED deviation rather than a
/// `k * ulp` bound, and it mirrors `compose_lower_with_policy`'s branch exactly:
/// positive coefficients take the lower slope, negative take the upper, zero is
/// exactly zero.
#[inline]
fn exact_lower_product(coeff: f32, relax: &crate::layers::activations::LinearRelaxation) -> f64 {
    if coeff > 0.0 {
        coeff as f64 * relax.lower_slope as f64
    } else if coeff < 0.0 {
        coeff as f64 * relax.upper_slope as f64
    } else {
        0.0
    }
}

/// The EXACT real product `compose_upper` intends to store. Mirrors
/// `compose_upper_with_policy`: positive takes the upper slope, negative the
/// lower.
#[inline]
fn exact_upper_product(coeff: f32, relax: &crate::layers::activations::LinearRelaxation) -> f64 {
    if coeff > 0.0 {
        coeff as f64 * relax.upper_slope as f64
    } else if coeff < 0.0 {
        coeff as f64 * relax.lower_slope as f64
    } else {
        0.0
    }
}

/// The slope `compose_lower` will multiply by, for the incoming-error term.
#[inline]
fn lower_slope_used(coeff: f32, relax: &crate::layers::activations::LinearRelaxation) -> f32 {
    if coeff > 0.0 {
        relax.lower_slope
    } else if coeff < 0.0 {
        relax.upper_slope
    } else {
        0.0
    }
}

/// The slope `compose_upper` will multiply by.
#[inline]
fn upper_slope_used(coeff: f32, relax: &crate::layers::activations::LinearRelaxation) -> f32 {
    if coeff > 0.0 {
        relax.upper_slope
    } else if coeff < 0.0 {
        relax.lower_slope
    } else {
        0.0
    }
}

/// Accumulate one stored coefficient's certified deviation into a row.
///
/// `stored` is what the composition actually wrote; `exact` is the true real
/// product. A non-finite deviation poisons the row with `+INF` (invariant I5:
/// sanitize outward, NEVER to zero).
#[inline]
fn accumulate_dev(row_err: &mut f64, stored: f32, exact: f64) {
    let dev = (stored as f64 - exact).abs();
    if !dev.is_finite() {
        *row_err = f64::INFINITY;
    } else if dev > *row_err {
        *row_err = dev;
    }
}

/// Sparse patches backward for element-wise activations.
///
/// Iterates only over unstable output positions (from `unstable_idx`)
/// instead of the full (out_c, out_h, out_w) grid. Patches tensor is 4D
/// (unstable_size, in_c, kH, kW). Bias vectors have length unstable_size.
///
/// Reference: alpha-beta-CROWN auto_LiRPA/patches.py sparse Patches.matmul
pub(super) fn backward_patches_sparse(
    lower_a_data: &PatchesData,
    upper_a_data: &PatchesData,
    lower_patches: &ArrayD<f32>,
    upper_patches: &ArrayD<f32>,
    bounds: &PatchesLinearBounds,
    relaxations: &[crate::layers::activations::LinearRelaxation],
    input_shape: (usize, usize, usize),
) -> Result<CrownBounds> {
    // The channel below is implemented and unit-tested but has NO declared
    // lever, so the shipped entry point refuses exactly as it did before it
    // existed — byte-identical. Arming it needs a TYPED PRESET KEY with a real
    // `Provenance::Measured`, and no such measurement exists yet: sparse
    // Patches requires `unstable_idx`, which only POOLING creates, and no
    // officially scored 2025 category has a pooled model. Tests reach the
    // implementation through `backward_patches_sparse_with_policy`.
    //
    // Deliberately NOT an `env::var` gate: that is the debt
    // `ny-levers/tests/ratchet.rs` exists to stop (it tripped this exact file
    // at +1 over its 706 baseline), and a raw `NY_*` read is unreachable in
    // competition anyway (`run_instance.sh` exports none).
    backward_patches_sparse_with_policy(
        lower_a_data,
        upper_a_data,
        lower_patches,
        upper_patches,
        bounds,
        relaxations,
        input_shape,
        false,
    )
}

/// Policy-explicit body of [`backward_patches_sparse`]; `coeff_err_armed` is the
/// resolved `#sparse-coeff-err` gate, threaded so tests exercise both policies
/// without mutating process env (the same idiom as
/// `compose::compose_lower_with_policy`).
#[allow(clippy::too_many_arguments)]
pub(super) fn backward_patches_sparse_with_policy(
    lower_a_data: &PatchesData,
    upper_a_data: &PatchesData,
    lower_patches: &ArrayD<f32>,
    upper_patches: &ArrayD<f32>,
    bounds: &PatchesLinearBounds,
    relaxations: &[crate::layers::activations::LinearRelaxation],
    input_shape: (usize, usize, usize),
    coeff_err_armed: bool,
) -> Result<CrownBounds> {
    let idx = lower_a_data
        .unstable_idx
        .as_ref()
        .ok_or_else(|| NyError::InternalError("backward_patches_sparse: no unstable_idx".into()))?;
    let upper_idx = upper_a_data.unstable_idx.as_ref().ok_or_else(|| {
        NyError::InvalidSpec("sparse Patches requires an upper unstable-index map".into())
    })?;
    lower_a_data.validate_common_geometry(upper_a_data)?;
    let affine_geometry = lower_a_data
        .geometry
        .require_affine("sparse activation Patches backward")?;
    if idx.channels != upper_idx.channels
        || idx.heights != upper_idx.heights
        || idx.widths != upper_idx.widths
    {
        return Err(NyError::InvalidSpec(
            "sparse Patches requires one authenticated lower/upper unstable-index map".into(),
        ));
    }
    let (_, in_h, in_w) = input_shape;
    let n = idx.len();
    let shape = lower_patches.shape();
    if upper_patches.shape() != shape {
        return Err(NyError::ShapeMismatch {
            expected: shape.to_vec(),
            got: upper_patches.shape().to_vec(),
        });
    }
    // `compose_lower`/`compose_upper` below round every stored product. Even an
    // exact incoming carrier therefore needs a new intrinsic error receipt;
    // emitting `coeff_err: None` would be a false-proof gap. Keep the reviewed
    // implementation body available for the eventual error-channel work, but
    // refuse before arithmetic today.
    if !coeff_err_armed {
        require_sparse_coeff_error_transport()?;
    }
    let explicit_rows = match shape.len() {
        4 => {
            if shape[0] != n {
                return Err(NyError::ShapeMismatch {
                    expected: vec![n],
                    got: vec![shape[0]],
                });
            }
            false
        }
        5 => {
            if shape[0] != bounds.row_count || shape[1] != n {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count, n],
                    got: vec![shape[0], shape[1]],
                });
            }
            true
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![4, 5],
                got: vec![shape.len()],
            });
        }
    };
    let (in_c, kh, kw) = if explicit_rows {
        (shape[2], shape[3], shape[4])
    } else {
        (shape[1], shape[2], shape[3])
    };

    let (sh, sw) = affine_geometry.stride();
    let (pad_left, _pad_right, pad_top, _pad_bottom) = affine_geometry.padding();

    let mut new_lower_patches = ArrayD::<f32>::zeros(lower_patches.raw_dim());
    let mut new_upper_patches = ArrayD::<f32>::zeros(upper_patches.raw_dim());
    let mut new_lower_b_f64 = bounds.lower_b.mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b.mapv(|x| x as f64);
    let logical_rows = if explicit_rows { bounds.row_count } else { n };
    let mut lower_nonfinite = vec![false; logical_rows];
    let mut upper_nonfinite = vec![false; logical_rows];
    // #sparse-coeff-err. Per-LOGICAL-ROW maxima, matching the `coeff_err`
    // contract ("for every stored coefficient in logical row i,
    // |stored - true| <= err[i]"): a MAX over the row, not a sum.
    // `*_slope_abs_max` carries the incoming carrier's error forward —
    // |stored_new - true_new| <= rounding_dev + |slope| * err_in.
    let mut lower_dev = vec![0.0f64; logical_rows];
    let mut upper_dev = vec![0.0f64; logical_rows];
    let mut lower_slope_abs_max = vec![0.0f64; logical_rows];
    let mut upper_slope_abs_max = vec![0.0f64; logical_rows];

    if explicit_rows {
        for row in 0..bounds.row_count {
            for i in 0..n {
                let oh = idx.heights[i];
                let ow = idx.widths[i];
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                            let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                            if ih_raw < 0
                                || (ih_raw as usize) >= in_h
                                || iw_raw < 0
                                || (iw_raw as usize) >= in_w
                            {
                                continue;
                            }
                            let ih = ih_raw as usize;
                            let iw = iw_raw as usize;
                            let input_flat = ic * in_h * in_w + ih * in_w + iw;
                            let relax = &relaxations[input_flat];

                            let la = lower_patches[[row, i, ic, ki, kj]];
                            let lr = compose::compose_lower(la, relax);
                            new_lower_patches[[row, i, ic, ki, kj]] = lr.new_coeff;
                            new_lower_b_f64[row] += lr.intercept_contrib;
                            lower_nonfinite[row] |= lr.nonfinite;
                            accumulate_dev(
                                &mut lower_dev[row],
                                lr.new_coeff,
                                exact_lower_product(la, relax),
                            );
                            let ls = lower_slope_used(la, relax).abs() as f64;
                            if ls > lower_slope_abs_max[row] {
                                lower_slope_abs_max[row] = ls;
                            }

                            let ua = upper_patches[[row, i, ic, ki, kj]];
                            let ur = compose::compose_upper(ua, relax);
                            new_upper_patches[[row, i, ic, ki, kj]] = ur.new_coeff;
                            new_upper_b_f64[row] += ur.intercept_contrib;
                            upper_nonfinite[row] |= ur.nonfinite;
                            accumulate_dev(
                                &mut upper_dev[row],
                                ur.new_coeff,
                                exact_upper_product(ua, relax),
                            );
                            let us = upper_slope_used(ua, relax).abs() as f64;
                            if us > upper_slope_abs_max[row] {
                                upper_slope_abs_max[row] = us;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for i in 0..n {
            let oh = idx.heights[i];
            let ow = idx.widths[i];
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                        if ih_raw < 0
                            || (ih_raw as usize) >= in_h
                            || iw_raw < 0
                            || (iw_raw as usize) >= in_w
                        {
                            continue;
                        }
                        let ih = ih_raw as usize;
                        let iw = iw_raw as usize;
                        let input_flat = ic * in_h * in_w + ih * in_w + iw;
                        let relax = &relaxations[input_flat];

                        let la = lower_patches[[i, ic, ki, kj]];
                        let lr = compose::compose_lower(la, relax);
                        new_lower_patches[[i, ic, ki, kj]] = lr.new_coeff;
                        new_lower_b_f64[i] += lr.intercept_contrib;
                        lower_nonfinite[i] |= lr.nonfinite;
                        accumulate_dev(
                            &mut lower_dev[i],
                            lr.new_coeff,
                            exact_lower_product(la, relax),
                        );
                        let ls = lower_slope_used(la, relax).abs() as f64;
                        if ls > lower_slope_abs_max[i] {
                            lower_slope_abs_max[i] = ls;
                        }

                        let ua = upper_patches[[i, ic, ki, kj]];
                        let ur = compose::compose_upper(ua, relax);
                        new_upper_patches[[i, ic, ki, kj]] = ur.new_coeff;
                        new_upper_b_f64[i] += ur.intercept_contrib;
                        upper_nonfinite[i] |= ur.nonfinite;
                        accumulate_dev(
                            &mut upper_dev[i],
                            ur.new_coeff,
                            exact_upper_product(ua, relax),
                        );
                        let us = upper_slope_used(ua, relax).abs() as f64;
                        if us > upper_slope_abs_max[i] {
                            upper_slope_abs_max[i] = us;
                        }
                    }
                }
            }
        }
    }

    compose::log_nonfinite_fallback(
        "Patches activation (sparse)",
        lower_nonfinite.iter().filter(|&&r| r).count(),
        upper_nonfinite.iter().filter(|&&r| r).count(),
        logical_rows,
    );

    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

    if explicit_rows {
        for row in 0..bounds.row_count {
            if lower_nonfinite[row] {
                for i in 0..n {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                new_lower_patches[[row, i, ic, ki, kj]] = 0.0;
                            }
                        }
                    }
                }
                new_lower_b[row] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[row] {
                for i in 0..n {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                new_upper_patches[[row, i, ic, ki, kj]] = 0.0;
                            }
                        }
                    }
                }
                new_upper_b[row] = f32::INFINITY;
            }
        }
    } else {
        for i in 0..n {
            if lower_nonfinite[i] {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            new_lower_patches[[i, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[i] {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            new_upper_patches[[i, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_upper_b[i] = f32::INFINITY;
            }
        }
    }

    // #sparse-coeff-err: publish the receipt.
    //
    // SOUNDNESS. For a stored coefficient in logical row `i`,
    //   |stored - true|
    //     <= |stored - a_stored*slope|            (this composition's rounding)
    //      + |a_stored*slope - a_true*slope|      (the INCOMING carrier's error)
    //     <= dev[i] + max_j|slope_j| * err_in[i]
    // The first term is MEASURED exactly (f32*f32 is exact in f64); the second
    // is the incoming per-row error scaled by the largest slope this row
    // multiplied by. Both are row maxima, so their sum over-bounds every
    // coefficient in the row.
    //
    // A row whose composition went non-finite is already poisoned downstream
    // (its coefficients are zeroed and its bias set to +/-INF above), but we
    // still emit +INF here rather than a finite value: invariant I5 sanitizes
    // outward, never to zero.
    let carry_in = |incoming: Option<&ndarray::Array1<f32>>, row: usize| -> f64 {
        match incoming {
            None => 0.0,
            Some(e) => {
                if row >= e.len() {
                    // A length mismatch is not something to guess through.
                    f64::INFINITY
                } else {
                    let v = e[row] as f64;
                    if v.is_finite() && v >= 0.0 {
                        v
                    } else {
                        f64::INFINITY
                    }
                }
            }
        }
    };
    let finish = |dev: &[f64],
                  slope_max: &[f64],
                  incoming: Option<&ndarray::Array1<f32>>,
                  nonfinite: &[bool]|
     -> ndarray::Array1<f32> {
        ndarray::Array1::from_iter((0..logical_rows).map(|row| {
            if nonfinite[row] {
                return f32::INFINITY;
            }
            let total = dev[row] + slope_max[row] * carry_in(incoming, row);
            if !total.is_finite() {
                return f32::INFINITY;
            }
            // Round the certified bound OUTWARD (up) into binary32.
            let as_f32 = next_up_f32(total as f32);
            if as_f32.is_finite() && as_f32 >= 0.0 {
                as_f32
            } else {
                f32::INFINITY
            }
        }))
    };
    let lower_coeff_err = finish(
        &lower_dev,
        &lower_slope_abs_max,
        lower_a_data.coeff_err.as_ref(),
        &lower_nonfinite,
    );
    let upper_coeff_err = finish(
        &upper_dev,
        &upper_slope_abs_max,
        upper_a_data.coeff_err.as_ref(),
        &upper_nonfinite,
    );

    Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: bounds.row_count,
        lower_a: PatchesData {
            coeff_err: Some(lower_coeff_err),
            patches: Some(new_lower_patches),
            geometry: lower_a_data.geometry.clone(),
            identity: false,
            output_shape: lower_a_data.output_shape,
            input_shape: lower_a_data.input_shape,
            unstable_idx: lower_a_data.unstable_idx.clone(),
        },
        lower_b: new_lower_b,
        upper_a: PatchesData {
            coeff_err: Some(upper_coeff_err),
            patches: Some(new_upper_patches),
            geometry: upper_a_data.geometry.clone(),
            identity: false,
            output_shape: upper_a_data.output_shape,
            input_shape: upper_a_data.input_shape,
            unstable_idx: upper_a_data.unstable_idx.clone(),
        },
        upper_b: new_upper_b,
    })))
}

#[cfg(test)]
mod sparse_coeff_err_tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, UnstableIdx};
    use crate::layers::activations::LinearRelaxation;
    use ndarray::{array, Array1, IxDyn};

    /// Two unstable output positions over a 1x1x2 input, 1x1 kernel, no padding,
    /// so every stored coefficient maps to exactly one relaxation and the
    /// expected arithmetic is checkable by hand.
    fn fixture(
        lower_vals: Vec<f32>,
        upper_vals: Vec<f32>,
        incoming_err: Option<Array1<f32>>,
    ) -> (
        PatchesData,
        PatchesData,
        ArrayD<f32>,
        ArrayD<f32>,
        PatchesLinearBounds,
    ) {
        let idx = UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, 0],
            widths: vec![0, 1],
        };
        let shape = IxDyn(&[2, 1, 1, 1]);
        let lp = ArrayD::from_shape_vec(shape.clone(), lower_vals).unwrap();
        let up = ArrayD::from_shape_vec(shape, upper_vals).unwrap();
        let side = |patches: &ArrayD<f32>, err: Option<Array1<f32>>| PatchesData {
            coeff_err: err,
            patches: Some(patches.clone()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 2),
            unstable_idx: Some(idx.clone()),
        };
        let lo = side(&lp, incoming_err.clone());
        let hi = side(&up, incoming_err);
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: lo.clone(),
            lower_b: array![0.0_f32, 0.0],
            upper_a: hi.clone(),
            upper_b: array![0.0_f32, 0.0],
        };
        (lo, hi, lp, up, bounds)
    }

    /// Slopes chosen so `coeff * slope` is NOT representable in binary32 and the
    /// stored value therefore genuinely deviates from the real product.
    fn relaxations() -> Vec<LinearRelaxation> {
        vec![
            LinearRelaxation {
                lower_slope: 0.1,
                lower_intercept: 0.0,
                upper_slope: 0.3,
                upper_intercept: 0.25,
            },
            LinearRelaxation {
                lower_slope: 0.7,
                lower_intercept: -0.125,
                upper_slope: 0.9,
                upper_intercept: 0.5,
            },
        ]
    }

    fn run(armed: bool, incoming: Option<Array1<f32>>) -> Result<CrownBounds> {
        let (lo, hi, lp, up, bounds) = fixture(vec![1.1, -2.3], vec![3.7, -4.9], incoming);
        backward_patches_sparse_with_policy(
            &lo,
            &hi,
            &lp,
            &up,
            &bounds,
            &relaxations(),
            (1, 1, 2),
            armed,
        )
    }

    /// The DEFAULT path must be byte-identical to before this channel existed.
    #[test]
    fn unarmed_still_refuses_exactly_as_before() {
        let err = run(false, None).expect_err("unarmed must refuse");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(_)),
            "{err:?}"
        );
    }

    /// THE SOUNDNESS PROPERTY. For every stored coefficient in logical row `i`,
    /// `|stored - true| <= err[i]`. Checked against the EXACT real products,
    /// recomputed here independently of the implementation's accumulator.
    #[test]
    fn the_receipt_bounds_every_stored_coefficient() {
        let lower_in = vec![1.1_f32, -2.3];
        let upper_in = vec![3.7_f32, -4.9];
        let (lo, hi, lp, up, bounds) = fixture(lower_in.clone(), upper_in.clone(), None);
        let relax = relaxations();
        let out = backward_patches_sparse_with_policy(
            &lo,
            &hi,
            &lp,
            &up,
            &bounds,
            &relax,
            (1, 1, 2),
            true,
        )
        .expect("armed must compose");
        let CrownBounds::Patches(pb) = out else {
            panic!("expected Patches");
        };
        let le = pb
            .lower_a
            .coeff_err
            .as_ref()
            .expect("lower receipt emitted");
        let ue = pb
            .upper_a
            .coeff_err
            .as_ref()
            .expect("upper receipt emitted");
        assert_eq!(le.len(), 2, "one entry per logical row");
        assert_eq!(ue.len(), 2);
        let lpat = pb.lower_a.patches.as_ref().unwrap();
        let upat = pb.upper_a.patches.as_ref().unwrap();
        for i in 0..2 {
            let exact_lo = exact_lower_product(lower_in[i], &relax[i]);
            let dev_lo = (lpat[[i, 0, 0, 0]] as f64 - exact_lo).abs();
            assert!(
                dev_lo <= le[i] as f64,
                "row {i}: lower deviation {dev_lo:e} exceeds its receipt {:e}",
                le[i]
            );
            let exact_up = exact_upper_product(upper_in[i], &relax[i]);
            let dev_up = (upat[[i, 0, 0, 0]] as f64 - exact_up).abs();
            assert!(
                dev_up <= ue[i] as f64,
                "row {i}: upper deviation {dev_up:e} exceeds its receipt {:e}",
                ue[i]
            );
        }
    }

    /// The receipt must be NON-ZERO where rounding actually occurred — a channel
    /// that always emitted 0 would pass the bound check above vacuously on an
    /// exact fixture and be a false-proof gap on a real one.
    #[test]
    fn the_receipt_is_not_vacuously_zero() {
        let out = run(true, None).expect("armed must compose");
        let CrownBounds::Patches(pb) = out else {
            panic!("expected Patches");
        };
        let le = pb.lower_a.coeff_err.as_ref().unwrap();
        assert!(
            le.iter().any(|&e| e > 0.0),
            "every entry is 0.0 — the fixture's products round, so this channel is not measuring"
        );
    }

    /// An incoming carrier error must be CARRIED FORWARD, scaled by the slope
    /// this row multiplied by. Dropping it would understate the certified error.
    #[test]
    fn an_incoming_error_is_carried_forward() {
        let clean = run(true, None).expect("armed");
        let dirty = run(true, Some(array![0.5_f32, 0.5])).expect("armed");
        let (CrownBounds::Patches(c), CrownBounds::Patches(d)) = (clean, dirty) else {
            panic!("expected Patches");
        };
        let ce = c.lower_a.coeff_err.as_ref().unwrap();
        let de = d.lower_a.coeff_err.as_ref().unwrap();
        for i in 0..2 {
            assert!(
                de[i] > ce[i],
                "row {i}: incoming error 0.5 did not widen the receipt ({} -> {})",
                ce[i],
                de[i]
            );
        }
    }

    /// I5: a non-finite deviation sanitizes to +INF, never to zero.
    #[test]
    fn a_nonfinite_deviation_poisons_the_row_outward() {
        let mut err = 0.0f64;
        accumulate_dev(&mut err, f32::INFINITY, 1.0);
        assert!(
            err.is_infinite(),
            "non-finite deviation must poison outward"
        );
        let mut err2 = 0.0f64;
        accumulate_dev(&mut err2, 2.0, 1.0);
        assert_eq!(err2, 1.0, "a finite deviation is recorded exactly");
        let mut err3 = 5.0f64;
        accumulate_dev(&mut err3, 2.0, 1.0);
        assert_eq!(err3, 5.0, "the row keeps its MAXIMUM, not the latest");
    }

    /// `f32 * f32` is exact in binary64 — the premise the whole channel rests on.
    #[test]
    fn the_exact_product_helper_mirrors_compose_and_is_exact() {
        let r = LinearRelaxation {
            lower_slope: 0.1,
            lower_intercept: 0.0,
            upper_slope: 0.3,
            upper_intercept: 0.0,
        };
        // positive coefficient takes the LOWER slope on the lower side
        assert_eq!(exact_lower_product(2.0, &r), 2.0_f64 * 0.1_f32 as f64);
        // negative takes the UPPER slope
        assert_eq!(exact_lower_product(-2.0, &r), -2.0_f64 * 0.3_f32 as f64);
        // and the upper side is the mirror image
        assert_eq!(exact_upper_product(2.0, &r), 2.0_f64 * 0.3_f32 as f64);
        assert_eq!(exact_upper_product(-2.0, &r), -2.0_f64 * 0.1_f32 as f64);
        // zero is exactly zero on both sides
        assert_eq!(exact_lower_product(0.0, &r), 0.0);
        assert_eq!(exact_upper_product(0.0, &r), 0.0);
    }
}
