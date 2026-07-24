// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Convolution input shape inference for β-CROWN backward propagation.
//!
//! These helpers infer spatial dimensions (H×W for 2D, length for 1D) from
//! flattened or partially-shaped pre-activation bounds so that Conv/ConvTranspose
//! layers can reconstruct the input layout for CROWN coefficient propagation.

use ny_core::{NyError, Result};

use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn infer_conv2d_input_hw(
        shape: &[usize],
        in_channels: usize,
        layer_label: &str,
    ) -> Result<(usize, usize)> {
        if in_channels == 0 {
            return Err(NyError::InvalidSpec(format!(
                "β-CROWN {} backward: in_channels must be > 0 (shape={:?})",
                layer_label, shape
            )));
        }
        if shape.len() >= 3 {
            let n = shape.len();
            let last3 = &shape[n - 3..];
            if let Some(chan_pos) = last3.iter().position(|&d| d == in_channels) {
                let mut spatial = Vec::with_capacity(2);
                for (idx, &d) in last3.iter().enumerate() {
                    if idx != chan_pos {
                        spatial.push(d);
                    }
                }
                if spatial.len() == 2 {
                    return Ok((spatial[0], spatial[1]));
                }
            }
        }

        if shape.len() <= 2 {
            let mut last_candidate: Option<(usize, usize)> = None;
            let mut other_candidate: Option<(usize, usize)> = None;
            let mut other_area = 0;
            let mut consider = |feature_dim: usize, is_last: bool| {
                if feature_dim > 0 && feature_dim.is_multiple_of(in_channels) {
                    let spatial = feature_dim / in_channels;
                    if spatial == 0 {
                        return;
                    }
                    // Use integer sqrt to avoid float-to-usize cast (Part of #2983).
                    // Manual implementation for MSRV 1.75 compat (isqrt stabilized in 1.84).
                    let side = {
                        let mut x = (spatial as f64).sqrt() as usize;
                        // Newton correction: f64 sqrt is exact for ≤2^52, but
                        // correct off-by-one for safety on all platforms.
                        if x.checked_mul(x).is_some_and(|sq| sq > spatial) && x > 0 {
                            x -= 1;
                        }
                        x
                    };
                    let square = side.checked_mul(side);
                    let is_square = square == Some(spatial)
                        || side.checked_add(1).and_then(|next| next.checked_mul(next))
                            == Some(spatial);
                    if is_square {
                        let actual_side = if square == Some(spatial) {
                            side
                        } else {
                            side + 1
                        };
                        let hw = (actual_side, actual_side);
                        if is_last {
                            last_candidate = Some(hw);
                        } else {
                            let area = spatial;
                            if area > other_area {
                                other_area = area;
                                other_candidate = Some(hw);
                            }
                        }
                    }
                }
            };
            if shape.len() == 2 {
                consider(shape[1], true);
                consider(shape[0], false);
                if let Some(hw) = last_candidate {
                    if shape[1] != in_channels || other_candidate.is_none() {
                        return Ok(hw);
                    }
                }
                if let Some(hw) = other_candidate {
                    return Ok(hw);
                }
                if let Some(hw) = last_candidate {
                    return Ok(hw);
                }
            } else if shape.len() == 1 {
                consider(shape[0], true);
                if let Some(hw) = last_candidate {
                    return Ok(hw);
                }
            }
        }

        Err(NyError::InvalidSpec(format!(
            "β-CROWN {} backward requires inferrable input H/W for in_channels={} (shape={:?})",
            layer_label, in_channels, shape
        )))
    }

    pub(in crate::beta_crown::engine) fn infer_conv1d_input_len(
        shape: &[usize],
        in_channels: usize,
        layer_label: &str,
    ) -> Result<usize> {
        if in_channels == 0 {
            return Err(NyError::InvalidSpec(format!(
                "β-CROWN {} backward: in_channels must be > 0 (shape={:?})",
                layer_label, shape
            )));
        }
        if shape.len() >= 2 {
            let n = shape.len();
            let last2 = [shape[n - 2], shape[n - 1]];
            if last2[0] == in_channels {
                return Ok(last2[1]);
            }
            if last2[1] == in_channels {
                return Ok(last2[0]);
            }
        }

        if shape.len() <= 2 {
            let mut last_len = None;
            let mut other_len = None;
            let mut consider = |feature_dim: usize, is_last: bool| {
                if in_channels > 0 && feature_dim.is_multiple_of(in_channels) {
                    let len = feature_dim / in_channels;
                    if is_last {
                        last_len = Some(len);
                    } else if other_len.map_or(true, |best| len > best) {
                        other_len = Some(len);
                    }
                }
            };
            if shape.len() == 2 {
                consider(shape[1], true);
                consider(shape[0], false);
                if let Some(len) = last_len {
                    if shape[1] != in_channels || other_len.is_none() {
                        return Ok(len);
                    }
                }
                if let Some(len) = other_len {
                    return Ok(len);
                }
                if let Some(len) = last_len {
                    return Ok(len);
                }
            } else if shape.len() == 1 {
                consider(shape[0], true);
                if let Some(len) = last_len {
                    return Ok(len);
                }
            }
        }

        Err(NyError::InvalidSpec(format!(
            "β-CROWN {} backward requires inferrable input length for in_channels={} (shape={:?})",
            layer_label, in_channels, shape
        )))
    }
}
