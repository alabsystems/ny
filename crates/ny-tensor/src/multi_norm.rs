// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-norm linear bounds for Lp-bounded perturbations (DeepT-style).
//!
//! This mirrors DeepT's bound representation where each output is expressed as
//! a linear function of perturbed inputs, then concretized using a dual-norm
//! radius. It is a building block for transformer-specific tightening.
//! Reference: `DeepT/Verifiers/Bounds.py`.
//!
//! Split into submodules by concern:
//! - `concretize`: Interval concretization and Lq dual norm computation
//! - `ops`: Matrix multiplication, dot product, and elementwise operations

mod concretize;
mod ops;

use ndarray::{Array3, Array4};
use ny_core::{NyError, Result, FALLBACK_BOUND};

#[cfg(test)]
mod tests;

/// Linear bounds for an Lp-bounded perturbation domain.
///
/// Shapes follow DeepT's convention:
/// - lw/uw: (batch, length, dim_in, dim_out)
/// - lb/ub: (batch, length, dim_out)
#[derive(Debug, Clone)]
pub struct MultiNormBounds {
    pub(crate) p: f32,
    pub(crate) q: f32,
    pub(crate) eps: f32,
    pub(crate) perturbed_words: usize,
    pub(crate) lw: Array4<f32>,
    pub(crate) lb: Array3<f32>,
    pub(crate) uw: Array4<f32>,
    pub(crate) ub: Array3<f32>,
}

impl MultiNormBounds {
    /// Get the Lp norm exponent.
    pub fn p(&self) -> f32 {
        self.p
    }

    /// Get the dual norm exponent.
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Get the perturbation radius.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get the number of perturbed words.
    pub fn perturbed_words(&self) -> usize {
        self.perturbed_words
    }

    /// Get lower weight coefficients (batch, length, dim_in, dim_out).
    pub fn lw(&self) -> &Array4<f32> {
        &self.lw
    }

    /// Get lower bias (batch, length, dim_out).
    pub fn lb(&self) -> &Array3<f32> {
        &self.lb
    }

    /// Get upper weight coefficients (batch, length, dim_in, dim_out).
    pub fn uw(&self) -> &Array4<f32> {
        &self.uw
    }

    /// Get upper bias (batch, length, dim_out).
    pub fn ub(&self) -> &Array3<f32> {
        &self.ub
    }

    /// Create new bounds with explicit tensors.
    pub fn new(
        p: f32,
        eps: f32,
        perturbed_words: usize,
        lw: Array4<f32>,
        lb: Array3<f32>,
        uw: Array4<f32>,
        ub: Array3<f32>,
    ) -> Result<Self> {
        if perturbed_words == 0 {
            return Err(NyError::InvalidSpec(
                "perturbed_words must be >= 1".to_string(),
            ));
        }
        if lw.shape() != uw.shape() {
            return Err(NyError::shape_mismatch(
                lw.shape().to_vec(),
                uw.shape().to_vec(),
            ));
        }
        if lb.shape() != ub.shape() {
            return Err(NyError::shape_mismatch(
                lb.shape().to_vec(),
                ub.shape().to_vec(),
            ));
        }
        let (batch, length, dim_in, dim_out) = lw.dim();
        let (lb_batch, lb_length, lb_dim_out) = lb.dim();
        if batch != lb_batch || length != lb_length || dim_out != lb_dim_out {
            return Err(NyError::shape_mismatch(
                vec![batch, length, dim_out],
                vec![lb_batch, lb_length, lb_dim_out],
            ));
        }
        if dim_in % perturbed_words != 0 {
            return Err(NyError::InvalidSpec(format!(
                "dim_in {} not divisible by perturbed_words {}",
                dim_in, perturbed_words
            )));
        }
        let q = Self::dual_norm(p);
        Ok(Self {
            p,
            q,
            eps,
            perturbed_words,
            lw,
            lb,
            uw,
            ub,
        })
    }

    /// Construct input bounds for perturbed word embeddings (DeepT-style).
    ///
    /// This mirrors DeepT's `_bound_input` initialization:
    /// - `embeddings` provides the unperturbed center values.
    /// - `indices` selects which token positions are perturbed.
    /// - Each perturbed word gets an identity block in the coefficient matrix.
    pub fn from_input_embeddings(
        embeddings: &Array3<f32>,
        p: f32,
        eps: f32,
        perturbed_words: usize,
        indices: &[usize],
    ) -> Result<Self> {
        let (batch, length, dim) = embeddings.dim();
        if perturbed_words == 0 {
            return Err(NyError::InvalidSpec(
                "perturbed_words must be >= 1".to_string(),
            ));
        }
        if indices.len() != perturbed_words {
            return Err(NyError::InvalidSpec(format!(
                "indices length {} does not match perturbed_words {}",
                indices.len(),
                perturbed_words
            )));
        }
        for &idx in indices {
            if idx >= length {
                return Err(NyError::InvalidSpec(format!(
                    "perturbed word index {} out of range (length {})",
                    idx, length
                )));
            }
        }

        let mut lw = Array4::<f32>::zeros((batch, length, dim * perturbed_words, dim));
        for b in 0..batch {
            for (word_i, &token_idx) in indices.iter().enumerate() {
                let start = word_i * dim;
                for d in 0..dim {
                    lw[[b, token_idx, start + d, d]] = 1.0;
                }
            }
        }
        let uw = lw.clone();
        let lb = embeddings.clone();
        let ub = embeddings.clone();

        Self::new(p, eps, perturbed_words, lw, lb, uw, ub)
    }

    /// Transpose length and output dimensions (DeepT-style t()).
    pub fn transpose_len_out(&self) -> Result<Self> {
        let lw = self.lw.view().permuted_axes([0, 3, 2, 1]).to_owned();
        let uw = self.uw.view().permuted_axes([0, 3, 2, 1]).to_owned();
        let lb = self.lb.view().permuted_axes([0, 2, 1]).to_owned();
        let ub = self.ub.view().permuted_axes([0, 2, 1]).to_owned();
        Self::new(self.p, self.eps, self.perturbed_words, lw, lb, uw, ub)
    }

    /// Add a constant scalar to bounds (shifts bias terms by `delta`).
    ///
    /// Matches `BoundedTensor::shift` and `ZonotopeTensor::shift` naming.
    pub fn shift(&self, scalar: f32) -> Self {
        Self {
            p: self.p,
            q: self.q,
            eps: self.eps,
            perturbed_words: self.perturbed_words,
            lw: self.lw.clone(),
            lb: &self.lb + scalar,
            uw: self.uw.clone(),
            ub: &self.ub + scalar,
        }
    }

    /// Multiply bounds by a scalar (swaps lower/upper for negative values).
    ///
    /// Matches `BoundedTensor::scale` and `ZonotopeTensor::scale` naming.
    pub fn scale(&self, scalar: f32) -> Self {
        // NaN scalar: cannot determine sign or magnitude. Return conservative
        // fallback bounds matching BoundedTensor::scale() repair behavior.
        if scalar.is_nan() {
            return Self {
                p: self.p,
                q: self.q,
                eps: self.eps,
                perturbed_words: self.perturbed_words,
                lw: Array4::zeros(self.lw.raw_dim()),
                lb: Array3::from_elem(self.lb.raw_dim(), -FALLBACK_BOUND),
                uw: Array4::zeros(self.uw.raw_dim()),
                ub: Array3::from_elem(self.ub.raw_dim(), FALLBACK_BOUND),
            };
        }
        // Avoid IEEE-754 indeterminate products (0 * inf -> NaN). Scaling by zero
        // should collapse both affine bounds to exactly zero.
        if matches!(scalar.classify(), std::num::FpCategory::Zero) {
            return Self {
                p: self.p,
                q: self.q,
                eps: self.eps,
                perturbed_words: self.perturbed_words,
                lw: Array4::zeros(self.lw.raw_dim()),
                lb: Array3::zeros(self.lb.raw_dim()),
                uw: Array4::zeros(self.uw.raw_dim()),
                ub: Array3::zeros(self.ub.raw_dim()),
            };
        }
        if scalar >= 0.0 {
            Self {
                p: self.p,
                q: self.q,
                eps: self.eps,
                perturbed_words: self.perturbed_words,
                lw: &self.lw * scalar,
                lb: &self.lb * scalar,
                uw: &self.uw * scalar,
                ub: &self.ub * scalar,
            }
        } else {
            Self {
                p: self.p,
                q: self.q,
                eps: self.eps,
                perturbed_words: self.perturbed_words,
                lw: &self.uw * scalar,
                lb: &self.ub * scalar,
                uw: &self.lw * scalar,
                ub: &self.lb * scalar,
            }
        }
    }

    /// Add another bounds object with the same parameters.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_compatible(other)?;
        Ok(Self {
            p: self.p,
            q: self.q,
            eps: self.eps,
            perturbed_words: self.perturbed_words,
            lw: &self.lw + &other.lw,
            lb: &self.lb + &other.lb,
            uw: &self.uw + &other.uw,
            ub: &self.ub + &other.ub,
        })
    }

    /// Add a bias term to the bounds.
    pub fn add_bias(&self, bias: &Array3<f32>) -> Result<Self> {
        if self.lb.shape() != bias.shape() {
            return Err(NyError::shape_mismatch(
                self.lb.shape().to_vec(),
                bias.shape().to_vec(),
            ));
        }
        Ok(Self {
            p: self.p,
            q: self.q,
            eps: self.eps,
            perturbed_words: self.perturbed_words,
            lw: self.lw.clone(),
            lb: &self.lb + bias,
            uw: self.uw.clone(),
            ub: &self.ub + bias,
        })
    }

    pub(crate) fn ensure_compatible(&self, other: &Self) -> Result<()> {
        if self.p != other.p
            || self.eps != other.eps
            || self.perturbed_words != other.perturbed_words
        {
            return Err(NyError::InvalidSpec(
                "multi-norm bounds parameters must match".to_string(),
            ));
        }
        if self.lw.shape() != other.lw.shape() || self.lb.shape() != other.lb.shape() {
            return Err(NyError::shape_mismatch(
                self.lw.shape().to_vec(),
                other.lw.shape().to_vec(),
            ));
        }
        Ok(())
    }

    pub(crate) fn dual_norm(p: f32) -> f32 {
        if p.is_infinite() {
            1.0
        } else if (p - 1.0).abs() < 1e-6 {
            f32::INFINITY
        } else {
            1.0 / (1.0 - 1.0 / p)
        }
    }
}
