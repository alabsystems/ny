// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, Array4, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::borrow::Cow;

use super::ZonotopeTensor;

/// Checked shape product with overflow error, clamped to at least 1.
fn checked_batch_size(shape: &[usize]) -> Result<usize> {
    Ok(checked_shape_product(shape)
        .ok_or_else(|| NyError::InvalidSpec(format!("batch shape product overflows: {:?}", shape)))?
        .max(1))
}

/// Bilinear operations on zonotopes.
impl ZonotopeTensor {
    /// Dot product of two 1D zonotopes: z₁ · z₂
    ///
    /// For z₁ = a₀ + Σᵢ aᵢeᵢ and z₂ = b₀ + Σᵢ bᵢeᵢ:
    ///
    /// z₁·z₂ = (a₀·b₀) + Σᵢ(a₀bᵢ + aᵢb₀)eᵢ + Σᵢ(aᵢbᵢ)eᵢ² + Σᵢ≠ⱼ(aᵢbⱼ)eᵢeⱼ
    ///
    /// Key insight: `eᵢ² ∈ [0,1]`, so we compute:
    /// - Center shift: +0.5 · Σᵢ(aᵢbᵢ)
    /// - New error: 0.5 · Σᵢ|aᵢbᵢ| + Σᵢ<ⱼ|aᵢbⱼ + aⱼbᵢ|
    ///
    /// # Returns
    /// A scalar zonotope (`element_shape = [1]`) with result and new error term.
    pub fn dot(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(NyError::InvalidSpec(format!(
                "Cannot compute dot product of zonotopes with different error counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        if self.element_shape != other.element_shape {
            return Err(NyError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        // For now, only support 1D zonotopes
        if self.element_shape.len() != 1 {
            return Err(NyError::InvalidSpec(
                "dot() currently only supports 1D zonotopes".to_string(),
            ));
        }

        let n = self.n_error_terms;

        // Extract center and error coefficients
        let a0 = self.coeffs.index_axis(Axis(0), 0);
        let b0 = other.coeffs.index_axis(Axis(0), 0);

        // 1. Center term: a₀ · b₀
        let mut center: f32 = a0.iter().zip(b0.iter()).map(|(&a, &b)| a * b).sum();

        // 2. Linear error terms: Σᵢ(a₀bᵢ + aᵢb₀)eᵢ
        // These are preserved in output
        let mut linear_coeffs = Vec::with_capacity(n);
        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            // Coefficient for eᵢ in output
            let coeff: f32 = a0.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                + ai.iter().zip(b0.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            linear_coeffs.push(coeff);
        }

        // 3. Quadratic terms eᵢ² and cross terms eᵢeⱼ

        // 3a. Same-symbol products: aᵢbᵢ·eᵢ² where eᵢ² = 0.5 ± 0.5
        let mut center_shift: f32 = 0.0;
        let mut half_term: f32 = 0.0;

        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            // aᵢ · bᵢ (dot product of coefficient vectors for error i)
            let ai_dot_bi: f32 = ai.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum();

            // eᵢ² = 0.5 + 0.5·e_new, so contribution is:
            // center += 0.5 * ai_dot_bi
            // new_error += 0.5 * |ai_dot_bi|
            center_shift += 0.5 * ai_dot_bi;
            half_term += 0.5 * ai_dot_bi.abs();
        }

        // 3b. Cross terms: aᵢbⱼ·eᵢeⱼ where i≠j
        // These become independent new errors, but we collapse them
        let mut big_term: f32 = 0.0;

        for i in 1..=n {
            let ai = self.coeffs.index_axis(Axis(0), i);
            let bi = other.coeffs.index_axis(Axis(0), i);

            for j in (i + 1)..=n {
                let aj = self.coeffs.index_axis(Axis(0), j);
                let bj = other.coeffs.index_axis(Axis(0), j);

                // Mixed term: aᵢ·bⱼ + aⱼ·bᵢ
                let ai_dot_bj: f32 = ai.iter().zip(bj.iter()).map(|(&a, &b)| a * b).sum();
                let aj_dot_bi: f32 = aj.iter().zip(bi.iter()).map(|(&a, &b)| a * b).sum();

                // Collapse cross terms into single error bound
                big_term += (ai_dot_bj + aj_dot_bi).abs();
            }
        }

        // Final center
        center += center_shift;

        // New error term coefficient (collapses all cross-products)
        let new_error_coeff = half_term + big_term;

        // Build result: scalar zonotope with original + 1 new error terms
        // coeffs shape: (1 + n + 1, 1)
        let mut result_coeffs = Array2::<f32>::zeros((1 + n + 1, 1));

        result_coeffs[[0, 0]] = center;
        for (i, &c) in linear_coeffs.iter().enumerate() {
            result_coeffs[[1 + i, 0]] = c;
        }
        result_coeffs[[1 + n, 0]] = new_error_coeff;

        Ok(Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: n + 1,
            element_shape: vec![1],
        })
    }

    /// Matrix multiplication: Z₁ @ Z₂^T where Z₁ and Z₂ share error symbols.
    ///
    /// This is the key operation for Q@K^T in attention.
    ///
    /// Following DeepT's `dot_product_precise` algorithm:
    /// 1. Center: `Q[0] @ K[0]^T`
    /// 2. Linear terms: `Q[0] @ K[i]^T + Q[i] @ K[0]^T` for each error `i`
    /// 3. Quadratic `e_i²`: center shift `0.5·Σ(Q[i]·K[i])` + radius `0.5·|Σ(Q[i]·K[i])|`
    /// 4. Cross terms `e_i×e_j`: collapse to single radius term
    ///
    /// # Arguments
    /// * `other` - The K zonotope (will be transposed for the multiplication)
    ///
    /// # Shapes
    /// * self: (seq_q, dim) zonotope with n error terms
    /// * other: (seq_k, dim) zonotope with n error terms (same symbols!)
    /// * result: (seq_q, seq_k) zonotope with n+1 error terms
    pub fn matmul_transposed(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(NyError::InvalidSpec(format!(
                "Cannot matmul zonotopes with different error counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        // Support N-D zonotopes by treating all leading dimensions as batch dimensions.
        if self.element_shape.len() < 2 || other.element_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "matmul_transposed() requires inputs with at least 2 dims, got {:?} and {:?}",
                self.element_shape, other.element_shape
            )));
        }

        let self_rank = self.element_shape.len();
        let other_rank = other.element_shape.len();
        let self_batch_shape = &self.element_shape[..self_rank - 2];
        let other_batch_shape = &other.element_shape[..other_rank - 2];
        if self_batch_shape != other_batch_shape {
            return Err(NyError::InvalidSpec(format!(
                "matmul_transposed batch dims must match, got {:?} and {:?}",
                self_batch_shape, other_batch_shape
            )));
        }

        let seq_q = self.element_shape[self_rank - 2];
        let dim_q = self.element_shape[self_rank - 1];
        let seq_k = other.element_shape[other_rank - 2];
        let dim_k = other.element_shape[other_rank - 1];

        if dim_q != dim_k {
            return Err(NyError::shape_mismatch(vec![dim_k], vec![dim_q]));
        }

        let dim = dim_q;
        let n = self.n_error_terms;

        let result_n_errors = n + 1;
        let batch_size = checked_batch_size(self_batch_shape)?;

        let self_coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let other_coeffs: Cow<'_, ArrayD<f32>> = if other.coeffs.is_standard_layout() {
            Cow::Borrowed(&other.coeffs)
        } else {
            Cow::Owned(other.coeffs.as_standard_layout().to_owned())
        };

        let self_4d = self_coeffs
            .as_ref()
            .view()
            .into_shape_with_order(IxDyn(&[1 + n, batch_size, seq_q, dim]))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape self coeffs for matmul".to_string()))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|_| NyError::InvalidSpec("Cannot view self coeffs as 4D".to_string()))?;
        let other_4d = other_coeffs
            .as_ref()
            .view()
            .into_shape_with_order(IxDyn(&[1 + n, batch_size, seq_k, dim]))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape other coeffs for matmul".to_string())
            })?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|_| NyError::InvalidSpec("Cannot view other coeffs as 4D".to_string()))?;

        let mut result_4d = Array4::<f32>::zeros((1 + result_n_errors, batch_size, seq_q, seq_k));

        for b in 0..batch_size {
            for q in 0..seq_q {
                for k in 0..seq_k {
                    // ===== Section 1: Center =====
                    let mut center: f32 = 0.0;
                    for d in 0..dim {
                        center += self_4d[[0, b, q, d]] * other_4d[[0, b, k, d]];
                    }

                    // ===== Section 2: Handle e_i² terms =====
                    let mut center_shift: f32 = 0.0;
                    let mut half_term: f32 = 0.0;
                    for i in 1..=n {
                        let mut q_dot_k: f32 = 0.0;
                        for d in 0..dim {
                            q_dot_k += self_4d[[i, b, q, d]] * other_4d[[i, b, k, d]];
                        }
                        center_shift += 0.5 * q_dot_k;
                        half_term += 0.5 * q_dot_k.abs();
                    }

                    // ===== Section 3: Preserve linear error terms =====
                    for i in 1..=n {
                        let mut linear_coeff: f32 = 0.0;
                        for d in 0..dim {
                            linear_coeff += self_4d[[0, b, q, d]] * other_4d[[i, b, k, d]];
                            linear_coeff += self_4d[[i, b, q, d]] * other_4d[[0, b, k, d]];
                        }
                        result_4d[[i, b, q, k]] = linear_coeff;
                    }

                    // ===== Section 4: Cross terms =====
                    let mut big_term: f32 = 0.0;
                    for i in 1..=n {
                        for j in (i + 1)..=n {
                            let mut mixed: f32 = 0.0;
                            for d in 0..dim {
                                mixed += self_4d[[i, b, q, d]] * other_4d[[j, b, k, d]];
                                mixed += self_4d[[j, b, q, d]] * other_4d[[i, b, k, d]];
                            }
                            big_term += mixed.abs();
                        }
                    }

                    result_4d[[0, b, q, k]] = center + center_shift;
                    result_4d[[n + 1, b, q, k]] = half_term + big_term;
                }
            }
        }

        let mut out_element_shape = self_batch_shape.to_vec();
        out_element_shape.push(seq_q);
        out_element_shape.push(seq_k);

        let mut out_coeffs_shape = vec![1 + result_n_errors];
        out_coeffs_shape.extend_from_slice(&out_element_shape);
        let out_coeffs = result_4d
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_coeffs_shape))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape matmul output".to_string()))?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: result_n_errors,
            element_shape: out_element_shape,
        })
    }

    /// Element-wise multiplication of two zonotopes: z1 ⊙ z2
    ///
    /// For SwiGLU: silu(gate) ⊙ up, where both have shared error symbols.
    ///
    /// # Mathematical Details
    ///
    /// For z1 = c1 + Σᵢ a1ᵢeᵢ and z2 = c2 + Σᵢ a2ᵢeᵢ:
    ///
    /// z1 * z2 = c1*c2 + c1*(Σᵢ a2ᵢeᵢ) + c2*(Σᵢ a1ᵢeᵢ) + (Σᵢ a1ᵢeᵢ)(Σⱼ a2ⱼeⱼ)
    ///
    /// The quadratic terms split into:
    /// - Same-symbol: `eᵢ² ∈ [0,1]` → shift center by `0.5*Σᵢ(a1ᵢ*a2ᵢ)`, add error `0.5*Σᵢ|a1ᵢ*a2ᵢ|`
    /// - Cross-symbol: `eᵢeⱼ ∈ [-1,1]` → add new error term `Σᵢ<ⱼ|a1ᵢ*a2ⱼ + a1ⱼ*a2ᵢ|`
    ///
    /// This preserves correlations between z1 and z2, giving tighter bounds than IBP
    /// which treats them as independent intervals.
    ///
    /// # Key Insight for SwiGLU
    ///
    /// When silu(gate) and up share error symbols (they do, since both come from the
    /// same input through FFN projections), this method exploits that correlation.
    /// IBP gives 36x growth; zonotope multiplication should be much tighter.
    pub fn mul_elementwise(&self, other: &Self) -> Result<Self> {
        // Expand to match error term counts
        let (z1, z2) = self.expand_to_match(other)?;

        if z1.element_shape != z2.element_shape {
            return Err(NyError::shape_mismatch(z1.element_shape, z2.element_shape));
        }

        // Generalize to N-D by flattening elements; the multiplication is per-element
        // and does not mix coordinates, so flattening preserves semantics.
        let element_shape = z1.element_shape.clone();
        let n_elements = checked_shape_product(&element_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "bilinear_elementwise_mul: element shape product overflows: {:?}",
                element_shape
            ))
        })?;

        let z1_flat = z1.reshape(&[n_elements])?;
        let z2_flat = z2.reshape(&[n_elements])?;

        let dim = n_elements;
        let n_errors = z1_flat.n_error_terms;

        let n_rows = 1 + n_errors + 1;
        let mut result_coeffs = Array2::<f32>::zeros((n_rows, dim));

        let c1 = z1_flat.coeffs.index_axis(Axis(0), 0);
        let c2 = z2_flat.coeffs.index_axis(Axis(0), 0);

        let mut a1: Vec<f32> = Vec::with_capacity(n_errors);
        let mut a2: Vec<f32> = Vec::with_capacity(n_errors);
        for d in 0..dim {
            let c1_d = c1[d];
            let c2_d = c2[d];

            a1.clear();
            a2.clear();
            for i in 1..=n_errors {
                a1.push(z1_flat.coeffs[[i, d]]);
                a2.push(z2_flat.coeffs[[i, d]]);
            }

            let same_symbol_sum: f32 = a1.iter().zip(a2.iter()).map(|(&x, &y)| x * y).sum();
            result_coeffs[[0, d]] = c1_d * c2_d + 0.5 * same_symbol_sum;

            for i in 0..n_errors {
                result_coeffs[[i + 1, d]] = c1_d * a2[i] + c2_d * a1[i];
            }

            let same_symbol_error: f32 = 0.5
                * a1.iter()
                    .zip(a2.iter())
                    .map(|(&x, &y)| (x * y).abs())
                    .sum::<f32>();
            let mut cross_error: f32 = 0.0;
            for i in 0..n_errors {
                for j in (i + 1)..n_errors {
                    cross_error += (a1[i] * a2[j] + a1[j] * a2[i]).abs();
                }
            }
            result_coeffs[[n_errors + 1, d]] = same_symbol_error + cross_error;
        }

        let flat = Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: n_errors + 1,
            element_shape: vec![n_elements],
        };

        flat.reshape(&element_shape)
    }
}
