// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Errors arising from MIP encoding or solving.
#[derive(Debug, thiserror::Error)]
pub enum MipError {
    /// Network encoding error (dimension mismatch, invalid bounds, etc.)
    #[error("encoding error: {0}")]
    Encoding(String),

    /// Invalid bounds (NaN, inverted, etc.)
    #[error("invalid bounds: {0}")]
    InvalidBounds(String),

    /// HiGHS solver returned an error status.
    #[error("solver error: {0}")]
    Solver(String),
}

impl From<MipError> for ny_core::NyError {
    fn from(e: MipError) -> Self {
        match e {
            MipError::Encoding(s) => ny_core::NyError::InternalError(format!("mip: encoding: {s}")),
            MipError::InvalidBounds(s) => {
                ny_core::NyError::InvalidSpec(format!("mip: invalid bounds: {s}"))
            }
            MipError::Solver(s) => ny_core::NyError::InternalError(format!("mip: solver: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_mip_op() -> ny_core::Result<()> {
        Err(MipError::Encoding("test".into()))?
    }

    #[test]
    fn mip_error_converts_to_ny_error() {
        let err = try_mip_op().unwrap_err();
        assert!(
            matches!(err, ny_core::NyError::InternalError(ref s) if s.contains("mip: encoding:")),
            "expected InternalError with mip prefix, got: {err:?}"
        );
    }
}
