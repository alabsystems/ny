// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array4, ArrayD, ArrayView4, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::borrow::Cow;

use super::ZonotopeTensor;

struct DisjointMatmulDims {
    batch_shape: Vec<usize>,
    batch_size: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
    n_self: usize,
    n_other: usize,
    result_n_errors: usize,
    has_cross_terms: bool,
}

fn checked_batch_size(shape: &[usize]) -> Result<usize> {
    Ok(checked_shape_product(shape)
        .ok_or_else(|| NyError::InvalidSpec(format!("batch shape product overflows: {:?}", shape)))?
        .max(1))
}

fn disjoint_matmul_dims(lhs: &ZonotopeTensor, rhs: &ZonotopeTensor) -> Result<DisjointMatmulDims> {
    if lhs.element_shape.len() < 2 || rhs.element_shape.len() < 2 {
        return Err(NyError::InvalidSpec(format!(
            "matmul_transposed_disjoint() requires inputs with at least 2 dims, got {:?} and {:?}",
            lhs.element_shape, rhs.element_shape
        )));
    }

    let lhs_rank = lhs.element_shape.len();
    let rhs_rank = rhs.element_shape.len();
    let lhs_batch_shape = &lhs.element_shape[..lhs_rank - 2];
    let rhs_batch_shape = &rhs.element_shape[..rhs_rank - 2];
    if lhs_batch_shape != rhs_batch_shape {
        return Err(NyError::InvalidSpec(format!(
            "matmul_transposed_disjoint batch dims must match, got {:?} and {:?}",
            lhs_batch_shape, rhs_batch_shape
        )));
    }

    let seq_q = lhs.element_shape[lhs_rank - 2];
    let dim_q = lhs.element_shape[lhs_rank - 1];
    let seq_k = rhs.element_shape[rhs_rank - 2];
    let dim_k = rhs.element_shape[rhs_rank - 1];
    if dim_q != dim_k {
        return Err(NyError::shape_mismatch(vec![dim_k], vec![dim_q]));
    }

    let n_self = lhs.n_error_terms;
    let n_other = rhs.n_error_terms;
    let has_cross_terms = n_self > 0 && n_other > 0;
    Ok(DisjointMatmulDims {
        batch_shape: lhs_batch_shape.to_vec(),
        batch_size: checked_batch_size(lhs_batch_shape)?,
        seq_q,
        seq_k,
        dim: dim_q,
        n_self,
        n_other,
        result_n_errors: n_self + n_other + usize::from(has_cross_terms),
        has_cross_terms,
    })
}

fn as_standard_coeffs(z: &ZonotopeTensor) -> Cow<'_, ArrayD<f32>> {
    if z.coeffs.is_standard_layout() {
        Cow::Borrowed(&z.coeffs)
    } else {
        Cow::Owned(z.coeffs.as_standard_layout().to_owned())
    }
}

fn reshape_coeffs_4d<'a>(
    coeffs: &'a ArrayD<f32>,
    n_errors: usize,
    batch_size: usize,
    seq: usize,
    dim: usize,
    label: &str,
) -> Result<ArrayView4<'a, f32>> {
    coeffs
        .view()
        .into_shape_with_order(IxDyn(&[1 + n_errors, batch_size, seq, dim]))
        .map_err(|_| {
            NyError::InvalidSpec(format!("Cannot reshape {label} coeffs for disjoint matmul"))
        })?
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| {
            NyError::InvalidSpec(format!(
                "Cannot view {label} coeffs as 4D for disjoint matmul"
            ))
        })
}

fn fill_disjoint_entry(
    result: &mut Array4<f32>,
    lhs: &ArrayView4<'_, f32>,
    rhs: &ArrayView4<'_, f32>,
    dims: &DisjointMatmulDims,
    batch_idx: usize,
    q_idx: usize,
    k_idx: usize,
) {
    let mut center = 0.0_f32;
    for d in 0..dims.dim {
        center += lhs[[0, batch_idx, q_idx, d]] * rhs[[0, batch_idx, k_idx, d]];
    }
    result[[0, batch_idx, q_idx, k_idx]] = center;

    for i in 1..=dims.n_self {
        let mut linear_coeff = 0.0_f32;
        for d in 0..dims.dim {
            linear_coeff += lhs[[i, batch_idx, q_idx, d]] * rhs[[0, batch_idx, k_idx, d]];
        }
        result[[i, batch_idx, q_idx, k_idx]] = linear_coeff;
    }

    for j in 1..=dims.n_other {
        let mut linear_coeff = 0.0_f32;
        for d in 0..dims.dim {
            linear_coeff += lhs[[0, batch_idx, q_idx, d]] * rhs[[j, batch_idx, k_idx, d]];
        }
        result[[dims.n_self + j, batch_idx, q_idx, k_idx]] = linear_coeff;
    }

    if dims.has_cross_terms {
        let mut mixed_radius = 0.0_f32;
        for i in 1..=dims.n_self {
            for j in 1..=dims.n_other {
                let mut mixed = 0.0_f32;
                for d in 0..dims.dim {
                    mixed += lhs[[i, batch_idx, q_idx, d]] * rhs[[j, batch_idx, k_idx, d]];
                }
                mixed_radius += mixed.abs();
            }
        }
        result[[dims.result_n_errors, batch_idx, q_idx, k_idx]] = mixed_radius;
    }
}

fn disjoint_result(result_4d: Array4<f32>, dims: &DisjointMatmulDims) -> Result<ZonotopeTensor> {
    let mut element_shape = dims.batch_shape.clone();
    element_shape.push(dims.seq_q);
    element_shape.push(dims.seq_k);

    let mut coeffs_shape = vec![1 + dims.result_n_errors];
    coeffs_shape.extend_from_slice(&element_shape);
    let coeffs = result_4d
        .into_dyn()
        .into_shape_with_order(IxDyn(&coeffs_shape))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape disjoint matmul output".to_string()))?;

    Ok(ZonotopeTensor {
        coeffs,
        n_error_terms: dims.result_n_errors,
        element_shape,
    })
}

impl ZonotopeTensor {
    /// Matrix multiplication: Z₁ @ Z₂^T where Z₁ and Z₂ use disjoint error symbols.
    ///
    /// This is the conservative path for bilinear products after an interval fallback
    /// has broken the original shared-symbol provenance, such as Packet C's
    /// `softmax @ V` seam in Whisper attention (#318).
    ///
    /// The result preserves each operand's linear error terms separately and
    /// collapses all mixed bilinear products into one new radius term.
    pub fn matmul_transposed_disjoint(&self, other: &Self) -> Result<Self> {
        let dims = disjoint_matmul_dims(self, other)?;
        let self_coeffs = as_standard_coeffs(self);
        let other_coeffs = as_standard_coeffs(other);
        let self_4d = reshape_coeffs_4d(
            self_coeffs.as_ref(),
            dims.n_self,
            dims.batch_size,
            dims.seq_q,
            dims.dim,
            "self",
        )?;
        let other_4d = reshape_coeffs_4d(
            other_coeffs.as_ref(),
            dims.n_other,
            dims.batch_size,
            dims.seq_k,
            dims.dim,
            "other",
        )?;

        let mut result_4d = Array4::<f32>::zeros((
            1 + dims.result_n_errors,
            dims.batch_size,
            dims.seq_q,
            dims.seq_k,
        ));
        for batch_idx in 0..dims.batch_size {
            for q_idx in 0..dims.seq_q {
                for k_idx in 0..dims.seq_k {
                    fill_disjoint_entry(
                        &mut result_4d,
                        &self_4d,
                        &other_4d,
                        &dims,
                        batch_idx,
                        q_idx,
                        k_idx,
                    );
                }
            }
        }

        disjoint_result(result_4d, &dims)
    }

    /// Matrix multiplication: Z₁ @ Z₂ where Z₁ and Z₂ use disjoint error symbols.
    pub fn matmul_disjoint(&self, other: &Self) -> Result<Self> {
        let other_t = other.transpose_last_two()?;
        self.matmul_transposed_disjoint(&other_t)
    }
}
