// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared shape parsing and batch-index helpers for MatMul propagation.

use ny_core::checked_shape_product;

use super::{NyError, Result};

/// Parsed dimensions for a MatMul operation (A @ B or A @ B^T).
#[derive(Debug)]
pub(in crate::layers::binary_ops) struct MatMulDims {
    pub batch_dims: Vec<usize>,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// b_shape[-2] * b_shape[-1] (raw storage size per batch for B).
    pub b_size_per_batch: usize,
}

impl MatMulDims {
    /// Total number of batch elements (product of batch_dims, min 1).
    pub fn batch_size(&self) -> Result<usize> {
        Ok(checked_shape_product(&self.batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MatMul: batch dimensions {:?} overflow usize",
                    self.batch_dims,
                ))
            })?
            .max(1))
    }

    /// Flattened output size per batch (m * n).
    ///
    /// Uses checked multiplication to prevent silent overflow from adversarial
    /// shapes (#3012).
    pub fn c_size_per_batch(&self) -> Result<usize> {
        self.m.checked_mul(self.n).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: c_size_per_batch overflow: m={} * n={}",
                self.m, self.n,
            ))
        })
    }

    /// Flattened A size per batch (m * k).
    ///
    /// Uses checked multiplication to prevent silent overflow from adversarial
    /// shapes (#3012).
    pub fn a_size_per_batch(&self) -> Result<usize> {
        self.m.checked_mul(self.k).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: a_size_per_batch overflow: m={} * k={}",
                self.m, self.k,
            ))
        })
    }
}

/// Parse and validate MatMul dimensions from input shapes.
///
/// Both inputs must be at least 2D and have matching batch and contraction
/// dimensions. Returns the parsed dimensions or an error.
pub(in crate::layers::binary_ops) fn parse_matmul_dims(
    transpose_b: bool,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<MatMulDims> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "MatMul requires at least 2D inputs".to_string(),
        ));
    }

    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    let m = a_shape[a_ndim - 2];
    let k_a = a_shape[a_ndim - 1];

    let (k_b, n) = if transpose_b {
        (b_shape[b_ndim - 1], b_shape[b_ndim - 2])
    } else {
        (b_shape[b_ndim - 2], b_shape[b_ndim - 1])
    };

    if k_a != k_b {
        return Err(NyError::ShapeMismatch {
            expected: vec![k_a],
            got: vec![k_b],
        });
    }

    let a_batch: Vec<usize> = a_shape[..a_ndim - 2].to_vec();
    let b_batch: Vec<usize> = b_shape[..b_ndim - 2].to_vec();

    if a_batch != b_batch {
        return Err(NyError::ShapeMismatch {
            expected: a_batch,
            got: b_batch,
        });
    }

    let b_size_per_batch = b_shape[b_ndim - 2]
        .checked_mul(b_shape[b_ndim - 1])
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: b_size_per_batch overflow: {} * {}",
                b_shape[b_ndim - 2],
                b_shape[b_ndim - 1],
            ))
        })?;

    Ok(MatMulDims {
        batch_dims: a_batch,
        m,
        k: k_a,
        n,
        b_size_per_batch,
    })
}

/// Decode a flat batch index into multi-dimensional batch indices.
///
/// For batch_dims = [2, 3], flat index 4 -> [1, 1].
///
/// Returns an error if any batch dimension is zero (prevents division-by-zero
/// panic from degenerate ONNX models with zero-valued tensor dimensions). (#2806)
pub(in crate::layers::binary_ops) fn decode_batch_index_into(
    batch_idx: usize,
    batch_dims: &[usize],
    scratch: &mut Vec<usize>,
) -> Result<()> {
    if batch_dims.contains(&0) {
        return Err(NyError::InvalidSpec(format!(
            "decode_batch_index: zero-valued batch dimension in {:?}",
            batch_dims
        )));
    }

    scratch.resize(batch_dims.len(), 0);
    let mut remaining = batch_idx;
    for d in (0..batch_dims.len()).rev() {
        scratch[d] = remaining % batch_dims[d];
        remaining /= batch_dims[d];
    }

    Ok(())
}

/// Stack-friendly variant of [`decode_batch_index_into`] that writes into a
/// pre-sized `&mut [usize]` buffer instead of a `Vec`.
///
/// The caller must ensure `buf.len() >= batch_dims.len()`. Only the first
/// `batch_dims.len()` elements are written; remaining elements are untouched.
///
/// Part of #2237 Finding 4 — eliminates per-call Vec heap allocations in
/// bilinear CROWN hot paths.
pub(in crate::layers::binary_ops) fn decode_batch_index_into_buf(
    batch_idx: usize,
    batch_dims: &[usize],
    buf: &mut [usize],
) -> Result<()> {
    if batch_dims.contains(&0) {
        return Err(NyError::InvalidSpec(format!(
            "decode_batch_index: zero-valued batch dimension in {:?}",
            batch_dims
        )));
    }
    debug_assert!(
        buf.len() >= batch_dims.len(),
        "decode_batch_index_into_buf: buffer too small ({} < {})",
        buf.len(),
        batch_dims.len(),
    );

    let mut remaining = batch_idx;
    for d in (0..batch_dims.len()).rev() {
        buf[d] = remaining % batch_dims[d];
        remaining /= batch_dims[d];
    }

    Ok(())
}

#[cfg(test)]
pub(in crate::layers::binary_ops) fn decode_batch_index(
    batch_idx: usize,
    batch_dims: &[usize],
) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(batch_dims.len());
    decode_batch_index_into(batch_idx, batch_dims, &mut indices)?;
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a `MatMulDims` directly for unit-testing overflow guards.
    fn dims(m: usize, k: usize, n: usize) -> MatMulDims {
        MatMulDims {
            batch_dims: vec![],
            m,
            k,
            n,
            b_size_per_batch: k * n, // might wrap, but tests below exercise the methods
        }
    }

    #[test]
    fn test_c_size_per_batch_normal_3012() {
        let d = dims(4, 3, 5);
        assert_eq!(d.c_size_per_batch().unwrap(), 20);
    }

    #[test]
    fn test_c_size_per_batch_overflow_3012() {
        let d = MatMulDims {
            batch_dims: vec![],
            m: usize::MAX,
            k: 1,
            n: 2,
            b_size_per_batch: 2,
        };
        let err = d.c_size_per_batch().unwrap_err();
        assert!(
            matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("c_size_per_batch overflow")),
            "expected overflow error, got: {err:?}"
        );
    }

    #[test]
    fn test_a_size_per_batch_normal_3012() {
        let d = dims(4, 3, 5);
        assert_eq!(d.a_size_per_batch().unwrap(), 12);
    }

    #[test]
    fn test_a_size_per_batch_overflow_3012() {
        let d = MatMulDims {
            batch_dims: vec![],
            m: usize::MAX,
            k: 2,
            n: 1,
            b_size_per_batch: 2,
        };
        let err = d.a_size_per_batch().unwrap_err();
        assert!(
            matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("a_size_per_batch overflow")),
            "expected overflow error, got: {err:?}"
        );
    }

    #[test]
    fn test_parse_matmul_dims_b_size_overflow_3012() {
        // Use shapes where k matches but b storage (k*n) overflows.
        // A = [1, big], B = [big, 2] where big * 2 overflows usize.
        let big = (usize::MAX / 2) + 1;
        let a_shape = &[1usize, big];
        let b_shape = &[big, 2usize];
        let err = parse_matmul_dims(false, a_shape, b_shape).unwrap_err();
        assert!(
            matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("b_size_per_batch overflow")),
            "expected b_size overflow error, got: {err:?}"
        );
    }
}
