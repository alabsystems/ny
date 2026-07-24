// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for arithmetic layer constructor parameters.

use ndarray::ArrayD;
use ny_core::{NyError, Result};

#[inline]
pub(crate) fn validate_finite_array(array: &ArrayD<f32>, layer: &str, param: &str) -> Result<()> {
    if let Some((index, value)) = array
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(NyError::InvalidSpec(format!(
            "{layer} {param} contains non-finite value at flat index {index}: {value}"
        )));
    }
    Ok(())
}
