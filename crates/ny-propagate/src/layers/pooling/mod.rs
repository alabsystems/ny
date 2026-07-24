// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pooling layers for bound propagation.

pub mod average;
mod average_patches;
pub mod max;
mod max_patches;

pub use average::AveragePoolLayer;
pub use max::MaxPool2dLayer;

use ny_core::{NyError, Result};

#[inline]
fn checked_pool_output_size(
    op_name: &str,
    input_h: usize,
    input_w: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(usize, usize)> {
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{op_name} stride must be non-zero, got stride=({sh},{sw})"
        )));
    }
    let padded_h = input_h
        .checked_add(
            ph.checked_mul(2)
                .ok_or_else(|| NyError::InvalidSpec(format!("{op_name} padding overflow")))?,
        )
        .ok_or_else(|| NyError::InvalidSpec(format!("{op_name} padded height overflow")))?;
    let padded_w = input_w
        .checked_add(
            pw.checked_mul(2)
                .ok_or_else(|| NyError::InvalidSpec(format!("{op_name} padding overflow")))?,
        )
        .ok_or_else(|| NyError::InvalidSpec(format!("{op_name} padded width overflow")))?;
    if padded_h < kh || padded_w < kw {
        return Err(NyError::InvalidSpec(format!(
            "{op_name} kernel larger than padded input: input=({input_h},{input_w}), padding=({ph},{pw}), kernel=({kh},{kw})"
        )));
    }
    Ok(((padded_h - kh) / sh + 1, (padded_w - kw) / sw + 1))
}

#[cfg(test)]
mod tests;
