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

/// Sparse patches backward for element-wise activations.
///
/// Iterates only over unstable output positions (from `unstable_idx`)
/// instead of the full (out_c, out_h, out_w) grid. Patches tensor is 4D
/// (unstable_size, in_c, kH, kW). Bias vectors have length unstable_size.
///
/// Reference: alpha-beta-CROWN auto_LiRPA/patches.py sparse Patches.matmul
/// Part of #2613 Phase 4 step 19
pub(super) fn backward_patches_sparse(
    lower_a_data: &PatchesData,
    upper_a_data: &PatchesData,
    lower_patches: &ArrayD<f32>,
    upper_patches: &ArrayD<f32>,
    bounds: &PatchesLinearBounds,
    relaxations: &[crate::layers::activations::LinearRelaxation],
    input_shape: (usize, usize, usize),
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
    require_sparse_coeff_error_transport()?;
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

                            let ua = upper_patches[[row, i, ic, ki, kj]];
                            let ur = compose::compose_upper(ua, relax);
                            new_upper_patches[[row, i, ic, ki, kj]] = ur.new_coeff;
                            new_upper_b_f64[row] += ur.intercept_contrib;
                            upper_nonfinite[row] |= ur.nonfinite;
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

                        let ua = upper_patches[[i, ic, ki, kj]];
                        let ur = compose::compose_upper(ua, relax);
                        new_upper_patches[[i, ic, ki, kj]] = ur.new_coeff;
                        new_upper_b_f64[i] += ur.intercept_contrib;
                        upper_nonfinite[i] |= ur.nonfinite;
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

    Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: bounds.row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(new_lower_patches),
            geometry: lower_a_data.geometry.clone(),
            identity: false,
            output_shape: lower_a_data.output_shape,
            input_shape: lower_a_data.input_shape,
            unstable_idx: lower_a_data.unstable_idx.clone(),
        },
        lower_b: new_lower_b,
        upper_a: PatchesData {
            coeff_err: None,
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
