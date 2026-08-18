// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Linear layers for bound propagation.
//!
//! Organized as a module directory:
//! - `mod.rs`: `LinearLayer` struct, constructors, accessors, and thin delegators
//! - `ibp.rs`: IBP forward propagation and `BoundPropagation` trait impl
//! - `crown_single.rs`: Single-domain CROWN backward (CPU faer + GEMM engine)
//! - `crown_batched.rs`: Batched CROWN backward for N-D batch dims
//! - `crown_batched_multi_domain.rs`: multi-domain GEMM batching
//! - `layout.rs`: Shared backward layout resolution helper
//! - `bias.rs`: Shared f64 bias accumulation + directed-rounding helper
//! - `spectral.rs`: Spectral norm computation

#[allow(dead_code)]
pub(crate) mod allocation_provenance;
pub(crate) mod bias;
pub(crate) mod crown_batched;
mod crown_batched_multi_domain;
pub(crate) mod crown_batched_soa;
pub(crate) mod crown_single;
mod ibp;
mod layout;
mod spectral;

/// Re-export of the f64 dot-product growth factor `γ_n` used by the conv2d
/// CROWN-backward error path (#vnncomp-aw-soundness).
pub(crate) use crown_single::gamma_n_f64 as crown_single_gamma_n_f64;

/// Re-export of the **f32** dot-product growth factor `γ_n` (`u = 2^-24`). Bounds
/// the error of an f32-ACCUMULATED dot of width `n` (`γ_n^f32·S`). Used by the
/// conv CROWN-backward small-contraction fast path, which keeps the f32 GEMM
/// coefficient and certifies it with this (sound) factor instead of paying for a
/// full f64 recompute — the looseness is negligible for small `n`
/// (#vnncomp-aw-soundness).
pub(crate) use crown_single::gamma_n_f32 as crown_single_gamma_n_f32;

use faer::Mat;
use ndarray::{Array1, Array2, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::next_up_f32;
use std::borrow::Cow;

use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};
use crate::{BatchedLinearBounds, LinearBounds};

use self::allocation_provenance::TrackedLinearArray;

// These imports are used by the test submodule (which accesses the parent's namespace).
// They are not used by production code in this file.
#[cfg(test)]
use super::common::BoundPropagation;
#[cfg(test)]
use crate::BoundedTensor;

/// A fully-connected linear layer: y = Wx + b
///
/// Stores weight matrix W and optional bias b for bound propagation.
/// Precomputes W+ = max(W,0) and W- = min(W,0) as faer matrices for fast IBP.
pub struct LinearLayer {
    /// Weight matrix of shape (out_features, in_features)
    ///
    /// Parameters are immutable after construction because every accelerated
    /// propagation path relies on construction-time positive/negative,
    /// transpose, norm, and faer caches. Expose read-only access through
    /// [`Self::weight`].
    pub(crate) weight: TrackedLinearArray<ndarray::Ix2>,
    /// Optional bias of shape (out_features,)
    ///
    /// Exposed read-only through [`Self::bias`].
    pub(crate) bias: Option<TrackedLinearArray<ndarray::Ix1>>,
    /// Cached positive part of weight for IBP (ndarray): max(W, 0)
    w_pos: TrackedLinearArray<ndarray::Ix2>,
    /// Cached negative part of weight for IBP (ndarray): min(W, 0)
    w_neg: TrackedLinearArray<ndarray::Ix2>,
    /// Cached transpose of w_pos as faer Mat: [in_features, out_features] for fast matmul
    w_pos_t_faer: Mat<f32>,
    /// Cached transpose of w_neg as faer Mat: [in_features, out_features] for fast matmul
    w_neg_t_faer: Mat<f32>,
    /// Cached weight as faer Mat: [out_features, in_features] for fast CROWN backward matmul
    weight_faer: Mat<f32>,
    /// Sound upper bound on spectral norm (largest singular value) of weight matrix.
    /// Used for zonotope scaling and SDP-CROWN radius propagation.
    spectral_norm: f32,
    /// Cached per-output-row L2 norm ‖W[o,:]‖₂ (rounded outward), computed once.
    /// Reused by the L2/Cauchy-Schwarz tightening so it is not recomputed per IBP
    /// call (the per-call recompute was an O(out·in) hot loop on deep models).
    row_l2_norms: TrackedLinearArray<ndarray::Ix1>,
    /// Lazily cached ROW-MAJOR transposes (in×out) of weight / W+ / W− for the
    /// engine-GEMM paths (#cora-transpose-cache): `propagate_concrete_via_gemm`
    /// / `propagate_ibp_via_gemm` previously rebuilt these on EVERY call —
    /// measured 62% of ALL samples on a cora PGD run (thousands of concrete
    /// point evaluations per second, each re-transposing the same immutable
    /// weight). `Arc<OnceLock<..>>` preserves the historical clone/cache-sharing
    /// semantics: the manual `Clone` shares filled caches while recapturing all
    /// deep owner allocations. This is consistent with the existing invariant
    /// that `weight` is immutable after construction (every other cached field
    /// above already relies on it). Lazy so layers never touched by an
    /// engine-GEMM path pay no memory.
    weight_t_rm: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
    w_pos_t_rm: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
    w_neg_t_rm: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
}

impl Clone for LinearLayer {
    fn clone(&self) -> Self {
        Self {
            // Each tracked ndarray clone captures the allocation actually
            // produced by ndarray. Provenance facts are never copied from the
            // source owner.
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            w_pos: self.w_pos.clone(),
            w_neg: self.w_neg.clone(),
            w_pos_t_faer: self.w_pos_t_faer.clone(),
            w_neg_t_faer: self.w_neg_t_faer.clone(),
            weight_faer: self.weight_faer.clone(),
            spectral_norm: self.spectral_norm,
            row_l2_norms: self.row_l2_norms.clone(),
            // Preserve the historical cache-sharing behavior. A retained-v1
            // observation is issued only after all three cells are filled and
            // their immutable Vec owners have been validated.
            weight_t_rm: self.weight_t_rm.clone(),
            w_pos_t_rm: self.w_pos_t_rm.clone(),
            w_neg_t_rm: self.w_neg_t_rm.clone(),
        }
    }
}

/// Transpose an (r×c) row-major ndarray into a (c×r) row-major flat Vec.
/// Shared by the lazy transpose caches above and value-identical to the
/// historical per-call loop in `ibp.rs`.
fn transpose_to_row_major_vec_of(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();
    let mut transposed = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            transposed.push(array[[row, col]]);
        }
    }
    transposed
}

impl std::fmt::Debug for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearLayer")
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("in_features", &self.in_features())
            .field("out_features", &self.out_features())
            .field("spectral_norm", &self.spectral_norm)
            .finish_non_exhaustive()
    }
}

impl LinearLayer {
    /// Create a new linear layer from weight matrix and optional bias.
    /// Precomputes W+, W-, their transposes as faer matrices, and spectral norm for fast IBP.
    pub fn new(weight: Array2<f32>, bias: Option<Array1<f32>>) -> Result<Self> {
        // Ensure weight is C-contiguous (standard layout) for GEMM engine paths
        // that require as_slice(). ndarray 0.17 clone() preserves F-layout, so
        // weights from const-fold or from_dynamic may arrive non-contiguous.
        // Ref: #4221, design doc 2026-03-20-issue-4221-linear-weight-contiguity-guard.md
        let weight = if weight.is_standard_layout() {
            weight
        } else {
            weight.as_standard_layout().into_owned()
        };
        if let Some(ref b) = bias {
            if b.len() != weight.nrows() {
                return Err(NyError::ShapeMismatch {
                    expected: vec![weight.nrows()],
                    got: vec![b.len()],
                });
            }
        }
        // Precompute W+ and W- for IBP (big speedup - avoids recomputing every call)
        // Use NaN-propagating variants so NaN weights poison bounds (#2432).
        let w_pos = weight.mapv(nan_propagating_max_zero);
        let w_neg = weight.mapv(nan_propagating_min_zero);

        // Precompute transposed faer matrices: w_pos_t is [in_features, out_features]
        // For IBP: X @ W_pos_t where X is [batch, in] and W_pos_t is [in, out]
        let (out_features, in_features) = (weight.nrows(), weight.ncols());
        let w_pos_t_faer = Mat::<f32>::from_fn(in_features, out_features, |i, j| {
            nan_propagating_max_zero(w_pos[[j, i]])
        });
        let w_neg_t_faer = Mat::<f32>::from_fn(in_features, out_features, |i, j| {
            nan_propagating_min_zero(w_neg[[j, i]])
        });

        // Precompute weight as faer Mat for fast CROWN backward matmul
        // Shape: [out_features, in_features] - same as weight
        let weight_faer = Mat::<f32>::from_fn(out_features, in_features, |i, j| weight[[i, j]]);

        // Compute a sound upper bound on the spectral norm.
        // Used for zonotope scaling and SDP-CROWN radius propagation.
        let spectral_norm = spectral::compute_spectral_norm(&weight);

        // Cache per-output-row ‖W[o,:]‖₂ once (constant; the L2/Cauchy-Schwarz
        // tightening reads it per IBP call). f64 accumulation, rounded UP to f32
        // (it scales the L2 radius → round outward for soundness).
        let mut row_l2_norms = Array1::<f32>::zeros(out_features);
        for o in 0..out_features {
            let mut sumsq = 0.0_f64;
            for j in 0..in_features {
                let w = weight[[o, j]] as f64;
                sumsq += w * w;
            }
            row_l2_norms[o] = next_up_f32(sumsq.sqrt() as f32);
        }

        Ok(Self {
            weight: TrackedLinearArray::new(weight),
            bias: bias.map(TrackedLinearArray::new),
            w_pos: TrackedLinearArray::new(w_pos),
            w_neg: TrackedLinearArray::new(w_neg),
            w_pos_t_faer,
            w_neg_t_faer,
            weight_faer,
            spectral_norm,
            row_l2_norms: TrackedLinearArray::new(row_l2_norms),
            weight_t_rm: std::sync::Arc::new(std::sync::OnceLock::new()),
            w_pos_t_rm: std::sync::Arc::new(std::sync::OnceLock::new()),
            w_neg_t_rm: std::sync::Arc::new(std::sync::OnceLock::new()),
        })
    }

    /// Row-major (in×out) transpose of `weight`, computed once and shared
    /// across clones (#cora-transpose-cache). Value-identical to transposing
    /// per call.
    pub(super) fn weight_t_row_major(&self) -> &[f32] {
        self.weight_t_rm
            .get_or_init(|| transpose_to_row_major_vec_of(&self.weight))
    }

    /// Row-major transpose of W+ (see [`Self::weight_t_row_major`]).
    pub(super) fn w_pos_t_row_major(&self) -> &[f32] {
        self.w_pos_t_rm
            .get_or_init(|| transpose_to_row_major_vec_of(&self.w_pos))
    }

    /// Row-major transpose of W− (see [`Self::weight_t_row_major`]).
    pub(super) fn w_neg_t_row_major(&self) -> &[f32] {
        self.w_neg_t_rm
            .get_or_init(|| transpose_to_row_major_vec_of(&self.w_neg))
    }

    /// Create from a flat row-major weight buffer and optional flat bias.
    ///
    /// `weight` is the `out_features * in_features` matrix in row-major (C) order;
    /// `bias`, if present, has length `out_features`. This is the ndarray-free
    /// constructor used by facade crates (e.g. `ny-api`) that build affine layers
    /// from plain `Vec<f32>` data without depending on `ndarray` directly.
    ///
    /// # Errors
    /// - [`NyError::ShapeMismatch`] if `weight.len() != out_features * in_features`,
    ///   or if `bias.len() != out_features`.
    pub fn from_flat(
        weight: Vec<f32>,
        out_features: usize,
        in_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        let expected = out_features
            .checked_mul(in_features)
            .ok_or_else(|| NyError::InvalidSpec("LinearLayer::from_flat shape overflow".into()))?;
        if weight.len() != expected {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_features, in_features],
                got: vec![weight.len()],
            });
        }
        let weight = Array2::from_shape_vec((out_features, in_features), weight)
            .map_err(|e| NyError::InvalidSpec(format!("LinearLayer::from_flat weight: {e}")))?;
        let bias = match bias {
            Some(b) => {
                if b.len() != out_features {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_features],
                        got: vec![b.len()],
                    });
                }
                Some(Array1::from_vec(b))
            }
            None => None,
        };
        Self::new(weight, bias)
    }

    /// Create from ArrayD (dynamic arrays), converting to appropriate shapes.
    pub fn from_dynamic(weight: &ArrayD<f32>, bias: Option<&ArrayD<f32>>) -> Result<Self> {
        // Weight should be 2D: (out_features, in_features)
        if weight.ndim() != 2 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0], // 2D expected
                got: weight.shape().to_vec(),
            });
        }

        let weight_2d = weight
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![weight.shape()[0], weight.shape()[1]],
                got: weight.shape().to_vec(),
            })?;

        let bias_1d = if let Some(b) = bias {
            if b.ndim() != 1 {
                return Err(NyError::ShapeMismatch {
                    expected: vec![weight.shape()[0]],
                    got: b.shape().to_vec(),
                });
            }
            Some(
                b.clone()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: vec![b.len()],
                        got: b.shape().to_vec(),
                    })?,
            )
        } else {
            None
        };

        Self::new(weight_2d, bias_1d)
    }

    /// Input dimension.
    pub fn in_features(&self) -> usize {
        self.weight.ncols()
    }

    /// Immutable weight matrix of shape `(out_features, in_features)`.
    pub fn weight(&self) -> &Array2<f32> {
        self.weight.as_array()
    }

    /// Immutable optional bias of shape `(out_features,)`.
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.bias.as_ref().map(TrackedLinearArray::as_array)
    }

    /// Replace both parameter tensors and rebuild every derived propagation cache.
    ///
    /// This is the supported mutation boundary for a constructed layer. Direct
    /// field mutation cannot keep the positive/negative, transpose, spectral,
    /// and row-norm caches coherent, so parameters are exposed read-only and
    /// updates are committed atomically through this method.
    ///
    /// If validation fails, `self` is left unchanged.
    pub fn replace_parameters(
        &mut self,
        weight: Array2<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<()> {
        let replacement = Self::new(weight, bias)?;
        *self = replacement;
        Ok(())
    }

    /// Replace the weight matrix and rebuild every derived propagation cache.
    ///
    /// The existing bias is retained. If the new output dimension is
    /// incompatible with that bias, `self` is left unchanged and a shape error
    /// is returned.
    pub fn set_weight(&mut self, weight: Array2<f32>) -> Result<()> {
        self.replace_parameters(weight, self.bias().cloned())
    }

    /// Replace the optional bias and rebuild the layer atomically.
    ///
    /// If the bias shape is invalid, `self` is left unchanged.
    pub fn set_bias(&mut self, bias: Option<Array1<f32>>) -> Result<()> {
        self.replace_parameters(self.weight().clone(), bias)
    }

    /// Output dimension.
    pub fn out_features(&self) -> usize {
        self.weight.nrows()
    }

    /// Spectral norm (largest singular value) of the weight matrix.
    /// Precomputed during construction for zonotope scaling.
    pub fn spectral_norm(&self) -> f32 {
        self.spectral_norm
    }

    /// Cached per-output-row L2 norm ‖W[o,:]‖₂ (rounded outward), for the
    /// L2/Cauchy-Schwarz IBP tightening. Computed once at construction.
    pub(crate) fn row_l2_norms(&self) -> &Array1<f32> {
        self.row_l2_norms.as_array()
    }

    // --- Internal accessors for submodules ---

    /// Cached positive part of weight (ndarray): max(W, 0).
    pub(super) fn w_pos(&self) -> &Array2<f32> {
        self.w_pos.as_array()
    }

    /// Cached negative part of weight (ndarray): min(W, 0).
    pub(super) fn w_neg(&self) -> &Array2<f32> {
        self.w_neg.as_array()
    }

    /// Cached transpose of w_pos as faer Mat.
    pub(super) fn w_pos_t_faer(&self) -> &Mat<f32> {
        &self.w_pos_t_faer
    }

    /// Cached transpose of w_neg as faer Mat.
    pub(super) fn w_neg_t_faer(&self) -> &Mat<f32> {
        &self.w_neg_t_faer
    }

    /// Cached weight as faer Mat.
    pub(super) fn weight_faer(&self) -> &Mat<f32> {
        &self.weight_faer
    }

    // --- Delegating CROWN entrypoints ---

    /// CROWN backward propagation using an optional GEMM engine for acceleration.
    ///
    /// Falls back to CPU propagation if the engine is `None` or if the GEMM call fails.
    #[inline]
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        crown_single::propagate_linear_with_engine(self, bounds, engine)
    }

    /// Deadline-aware CROWN backward propagation (#4321).
    ///
    /// When a deadline is present, the dense `A @ W` work uses a pollable CPU
    /// implementation and never enters a generic or process-global GEMM engine,
    /// whose API has no cancellation contract. A wide classifier-head GEMM is
    /// otherwise the single longest uninterrupted op on the spec-matrix root
    /// output-bound path and can overrun the verifier timeout.
    /// Returns [`ny_core::NyError::DeadlineExceeded`] once the deadline passes,
    /// which the graph-CROWN dispatch degrades to a sound per-node IBP fallback.
    #[inline]
    pub fn propagate_linear_with_engine_and_deadline<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<Cow<'a, LinearBounds>> {
        crown_single::propagate_linear_with_engine_and_deadline(self, bounds, engine, deadline)
    }

    /// Batched CROWN backward propagation through linear layer (N-D batch dims).
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        crown_batched::propagate_linear_batched(self, bounds)
    }

    pub(crate) fn propagate_linear_batched_maybe_engine(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        crown_batched::propagate_linear_batched_maybe_engine(self, bounds, engine)
    }

    /// Multi-domain GPU-batched CROWN backward propagation.
    pub fn propagate_linear_batched_with_engine(
        &self,
        bounds_batch: &[&LinearBounds],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<LinearBounds>> {
        crown_batched_multi_domain::propagate_linear_batched_with_engine(self, bounds_batch, engine)
    }
}

/// Fuse two consecutive linear layers into a single equivalent linear layer.
///
/// Given `layer1: x -> W1 x + b1` and `layer2: y -> W2 y + b2`, returns the
/// fused layer `x -> W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)`.
///
/// Panics when the layers are dimensionally incompatible, which indicates the
/// caller attempted to fuse a non-consecutive affine pair.
///
/// Needed by `ny_propagate::elimination::eliminate_dead_neurons`.
pub fn merge_linear(layer1: &LinearLayer, layer2: &LinearLayer) -> LinearLayer {
    assert_eq!(
        layer1.out_features(),
        layer2.in_features(),
        "merge_linear requires consecutive affine layers: {} -> {} is invalid",
        layer1.out_features(),
        layer2.in_features()
    );

    let weight = layer2.weight().dot(layer1.weight());
    let bias = match (layer1.bias(), layer2.bias()) {
        (Some(b1), Some(b2)) => Some(layer2.weight().dot(b1) + b2),
        (Some(b1), None) => Some(layer2.weight().dot(b1)),
        (None, Some(b2)) => Some(b2.clone()),
        (None, None) => None,
    };

    LinearLayer::new(weight, bias).expect("fused linear layer should remain valid")
}

#[cfg(test)]
mod tests;
