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
#[allow(dead_code)]
pub(crate) mod original_affine;
mod point;
mod spectral;

pub use point::{LinearPointArm, LinearPointReceipt};

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
    /// Lazily cached f64 widening of `weight_faer` — `W` and `|W|`, in both the
    /// owned-`Mat` and packed column-major layouts the certified `A·W` +
    /// `|A|·|W|` kernels consume (#linear-widened-weight-cache). Same defect and
    /// same remedy as the three transpose caches above: `crown_single`'s
    /// chunked, unbounded and rebound-scratch arms each rebuilt TWO full f64
    /// copies of the entire weight per call, and that call is reached once per
    /// Linear layer per domain per backward pass. `None` inside the cell means
    /// the weight is too large to retain (the ceiling the faer path already
    /// declines at), so those calls keep widening per call. `Arc<OnceLock<..>>`
    /// keeps the clone/cache-sharing semantics of the fields above, and rests on
    /// the same invariant: `weight` is immutable after construction. Lazy, so a
    /// layer no certified f64 path ever touches pays no memory.
    weight_f64: std::sync::Arc<std::sync::OnceLock<Option<crown_single::WidenedWeightF64>>>,
    /// Lazily cached `|W|` as an owned faer `Mat<f32>`
    /// (#linear-abs-weight-cache) — the operand every certified incoming-error
    /// composition `P[i,j] = Σ_k err_in[i,k]·|W[k,j]|` contracts against. Six
    /// production backward sites (two in `crown_single`, three in
    /// `crown_batched`, one in `crown_batched_multi_domain`) each rebuilt this
    /// whole `out_features × in_features` matrix on EVERY call, hoisted out of
    /// their inner position/batch loop but not out of the call, and each of
    /// those calls is reached once per Linear layer per domain per backward
    /// pass. Same invariant as the four cells above: `weight` — hence
    /// `weight_faer` — is immutable after construction, and the one supported
    /// mutation boundary ([`Self::replace_parameters`], which `set_weight` and
    /// `set_bias` both delegate to) commits through `*self = Self::new(..)`, so
    /// this cell is replaced with the parameters rather than surviving them.
    ///
    /// No retention ceiling, unlike `weight_f64`, and deliberately so: this is
    /// ONE more f32 copy of the weight beside the six the constructor already
    /// retains unconditionally (`weight`, `w_pos`, `w_neg`, `w_pos_t_faer`,
    /// `w_neg_t_faer`, `weight_faer`), so it cannot introduce a resident-memory
    /// class the layer does not already have. `weight_f64` retains four f64
    /// copies — eight times the weight, and a class of its own — which is why
    /// it needs the per-entry and process-total gates and this does not.
    ///
    /// Lazy, and read ONLY from inside the `Some(err_in)` arms at all six
    /// sites: a call whose incoming bounds carry no certified-error plane
    /// allocated and filled the whole matrix and dropped it unused, and now
    /// does not build it at all.
    weight_abs_faer: std::sync::Arc<std::sync::OnceLock<Mat<f32>>>,
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
            weight_f64: self.weight_f64.clone(),
            weight_abs_faer: self.weight_abs_faer.clone(),
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

fn linear_bytes_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| NyError::InvalidSpec("Linear allocation size overflow".into()))
}
fn linear_bytes_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| NyError::InvalidSpec("Linear allocation size overflow".into()))
}
fn faer_f32_bytes_bound(rows: usize, columns: usize) -> Result<usize> {
    // Qualified to faer 0.24's 64-byte row alignment for owned f32 matrices.
    let padded = rows
        .checked_next_multiple_of(16)
        .ok_or_else(|| NyError::InvalidSpec("Linear faer padding overflow".into()))?;
    linear_bytes_mul(linear_bytes_mul(padded, columns)?, 4)
}
fn faer_f32_payload_bytes(matrix: &Mat<f32>) -> Result<usize> {
    // These private matrices are built from_fn/clone, never reserve extra cols.
    let stride = usize::try_from(matrix.col_stride())
        .map_err(|_| NyError::InvalidSpec("Linear faer negative stride".into()))?;
    linear_bytes_mul(linear_bytes_mul(stride, matrix.ncols())?, 4)
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
    /// Retained requested bytes, including padded faer storage and every filled
    /// lazy cache. Shared Arc cache owners are attributed to each layer here;
    /// callers counting cloned layers may conservatively count them twice.
    /// This census excludes allocator bookkeeping/slack and is not process RSS.
    pub fn retained_bytes(&self) -> Result<usize> {
        let mut bytes = Self::allocation_control_bytes()?;
        for n in [
            self.weight.retained_payload_bytes()?,
            self.w_pos.retained_payload_bytes()?,
            self.w_neg.retained_payload_bytes()?,
            self.row_l2_norms.retained_payload_bytes()?,
            self.bias
                .as_ref()
                .map_or(Ok(0), |a| a.retained_payload_bytes())?,
        ] {
            bytes = linear_bytes_add(bytes, n)?;
        }
        for m in [&self.w_pos_t_faer, &self.w_neg_t_faer, &self.weight_faer] {
            bytes = linear_bytes_add(bytes, faer_f32_payload_bytes(m)?)?;
        }
        for cell in [&self.weight_t_rm, &self.w_pos_t_rm, &self.w_neg_t_rm] {
            if let Some(values) = cell.get() {
                bytes = linear_bytes_add(bytes, linear_bytes_mul(values.capacity(), 4)?)?;
            }
        }
        if let Some(Some(values)) = self.weight_f64.get() {
            bytes = linear_bytes_add(
                bytes,
                values.retained_payload_bytes().ok_or_else(|| {
                    NyError::InvalidSpec("Linear widened-cache census overflow".into())
                })?,
            )?;
        }
        if let Some(m) = self.weight_abs_faer.get() {
            bytes = linear_bytes_add(bytes, faer_f32_payload_bytes(m)?)?;
        }
        Ok(bytes)
    }

    fn allocation_control_bytes() -> Result<usize> {
        // Include the inline owner plus every Arc control/OnceLock allocation.
        // Rust's control prefix has two usize counters; round up for alignment.
        let arc = |payload: usize| linear_bytes_add(payload, 3 * size_of::<usize>());
        let vec_cell = arc(size_of::<std::sync::OnceLock<Vec<f32>>>())?;
        let wide_cell = arc(size_of::<
            std::sync::OnceLock<Option<crown_single::WidenedWeightF64>>,
        >())?;
        let abs_cell = arc(size_of::<std::sync::OnceLock<Mat<f32>>>())?;
        linear_bytes_add(
            size_of::<Self>(),
            linear_bytes_add(
                linear_bytes_mul(vec_cell, 3)?,
                linear_bytes_add(wide_cell, abs_cell)?,
            )?,
        )
    }

    /// Conservative retained capacity BEFORE constructing a fresh layer and
    /// running point/IBP forwards: all six constructor copies, row norms/bias,
    /// padded faer rows, three f32 transpose caches and the f32 absolute cache.
    /// Certified backward's optional f64 caches require separate admission.
    pub fn point_retained_bytes_bound(out: usize, inner: usize, bias: bool) -> Result<usize> {
        let weights = linear_bytes_mul(linear_bytes_mul(out, inner)?, 4)?;
        let transpose = faer_f32_bytes_bound(inner, out)?;
        let original = faer_f32_bytes_bound(out, inner)?;
        let arrays = linear_bytes_add(
            linear_bytes_mul(weights, 6)?,
            linear_bytes_mul(out, if bias { 8 } else { 4 })?,
        )?;
        linear_bytes_add(
            Self::allocation_control_bytes()?,
            linear_bytes_add(
                arrays,
                linear_bytes_add(
                    linear_bytes_mul(transpose, 2)?,
                    linear_bytes_mul(original, 2)?,
                )?,
            )?,
        )
    }

    /// Construction overlap: the graph builder deep-clones its temporary layer,
    /// conversion can copy parameters twice, and spectral norm retains one f64
    /// column sum per input. Full retained graph capacity is charged separately.
    pub fn construction_scratch_bytes_bound(out: usize, inner: usize, bias: bool) -> Result<usize> {
        linear_bytes_add(
            Self::point_retained_bytes_bound(out, inner, bias)?,
            linear_bytes_add(
                linear_bytes_mul(linear_bytes_mul(out, inner)?, 8)?,
                linear_bytes_mul(inner, 8)?,
            )?,
        )
    }

    /// Constructor maps/copies, spectral/Frobenius/induced sums and row norms.
    /// There is no SVD: spectral scratch is O(inner), work is O(out*inner).
    pub fn construction_work_bound(out: usize, inner: usize) -> Result<u64> {
        let work = linear_bytes_add(
            linear_bytes_mul(linear_bytes_mul(out, inner)?, 32)?,
            linear_bytes_mul(linear_bytes_add(out, inner)?, 8)?,
        )?;
        u64::try_from(work)
            .map_err(|_| NyError::InvalidSpec("Linear construction work overflow".into()))
    }

    /// Temporary padded faer products and ndarray copies simultaneously live in
    /// the N-D IBP path used by the concrete graph route. Graph cache endpoints
    /// and the retained lazy caches are charged by their separate owners.
    pub fn point_scratch_bytes_bound(out: usize, inner: usize, rows: usize) -> Result<usize> {
        let copies = linear_bytes_mul(linear_bytes_mul(rows, linear_bytes_add(out, inner)?)?, 32)?;
        let inputs = linear_bytes_mul(faer_f32_bytes_bound(rows, inner)?, 4)?;
        let outputs = linear_bytes_mul(faer_f32_bytes_bound(rows, out)?, 8)?;
        let absolute_products = linear_bytes_add(
            faer_f32_bytes_bound(inner, rows)?,
            faer_f32_bytes_bound(out, rows)?,
        )?;
        linear_bytes_add(
            copies,
            linear_bytes_add(inputs, linear_bytes_add(outputs, absolute_products)?)?,
        )
    }

    /// Create a new linear layer from weight matrix and optional bias.
    /// Precomputes W+, W-, their transposes as faer matrices, and spectral norm for fast IBP.
    pub fn new(weight: Array2<f32>, bias: Option<Array1<f32>>) -> Result<Self> {
        // Ensure weight is C-contiguous (standard layout) for GEMM engine paths
        // that require as_slice(). ndarray 0.17 clone() preserves F-layout, so
        // weights from const-fold or from_dynamic may arrive non-contiguous.
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
        // Use NaN-propagating variants so NaN weights poison bounds.
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
            weight_f64: std::sync::Arc::new(std::sync::OnceLock::new()),
            weight_abs_faer: std::sync::Arc::new(std::sync::OnceLock::new()),
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

    /// The f64 widening of [`Self::weight_faer`], built once on first use and
    /// shared across clones (#linear-widened-weight-cache). `None` when the
    /// weight exceeds the retention ceiling, in which case the certified `A·W`
    /// kernels widen per call exactly as they always did.
    ///
    /// Value-identical to widening in the kernel: `f32→f64` is EXACT and f64
    /// `abs` clears the sign bit, so the cached operands are the same bits in
    /// the same layout. See [`crown_single::WidenedWeightF64`].
    ///
    /// DEADLINE COST OF FIRST USE. Callers evaluate this as an ARGUMENT, so the
    /// one-time build runs before the callee's first `poll()`. Every subsequent
    /// call for the layer gets strictly MORE deadline headroom (it no longer
    /// widens at all), but the first pays the build in one unpolled stretch.
    /// That stretch is bounded by `WIDENED_WEIGHT_CACHE_MAX_ENTRY_BYTES` (8
    /// MiB, `crown_single`) — a weight above it is refused and never built —
    /// which is a fraction of the work in one `DEADLINE_AW_FAER_CHUNK_MACS`
    /// quantum, the granularity the chunked arm already treats as an acceptable
    /// gap between polls. So it is accounted for rather than machinery-fixed:
    /// tightening the ceiling tightens this bound with it.
    pub(crate) fn weight_widened_f64(&self) -> Option<&crown_single::WidenedWeightF64> {
        self.weight_f64
            .get_or_init(|| crown_single::WidenedWeightF64::new(&self.weight_faer))
            .as_ref()
    }

    /// `|W|` as an owned faer `Mat<f32>`, built once on first use and shared
    /// across clones (#linear-abs-weight-cache). Shape is
    /// `[out_features, in_features]`, i.e. [`Self::weight_faer`]'s own shape.
    ///
    /// BIT-IDENTICAL to the per-call
    /// `Mat::<f32>::from_fn(rows, cols, |k, j| weight_faer[(k, j)].abs())` it
    /// replaced. `f32::abs` clears the sign bit and rounds nothing, the operand
    /// is the same immutable `weight_faer`, and the constructor, dimensions and
    /// fill order are the ones each call site used: every site's `(rows, cols)`
    /// resolves to `(weight_faer.nrows(), weight_faer.ncols())` — the scalar
    /// and multi-domain sites through `resolve_backward_layout`, which yields
    /// `out_features == weight_rows` on BOTH of its branches, and the batched
    /// sites through the entry check that refuses any
    /// `mid_dim != layer.out_features()`.
    ///
    /// LAYOUT is preserved (an owned `Mat` from the identical constructor), and
    /// independently cannot matter here: the only consumers are
    /// `crown_single::incoming_error_product`'s two arms, which read
    /// `nrows()`/`ncols()` and individual `w_abs[(k, j)]` elements and copy
    /// them into a freshly built `Vec<f64>` before any `matmul`. No faer kernel
    /// ever observes this matrix's own storage, so faer's layout-driven kernel
    /// selection has nothing to key on.
    pub(super) fn weight_abs_faer(&self) -> &Mat<f32> {
        self.weight_abs_faer.get_or_init(|| {
            Mat::<f32>::from_fn(
                self.weight_faer.nrows(),
                self.weight_faer.ncols(),
                |k, j| self.weight_faer[(k, j)].abs(),
            )
        })
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

    /// Deadline-aware CROWN backward propagation.
    ///
    /// ROUTING (#pollable-faer). When a deadline is
    /// present, the dense `A @ W` + `|A|·|W|` pair runs on the pollable
    /// row-block-chunked faer host path
    /// (`crown_single::aw_f64_with_abssum_cpu_deadline`, restored by
    /// a79a42d1 after the 6f49a660 scalar-per-MAC regression), polled between
    /// blocks; sufficiently large products may enter the process-global sound
    /// f64 engine ONLY through its explicit bounded
    /// [`ny_core::GemmEngine::gemm_f64_with_deadline`] contract. The ordinary
    /// (cancellation-free) engine methods are never entered under finite
    /// authority, and the historical scalar per-MAC loop survives only as the
    /// over-quantum / scratch-ceiling / `NY_NAIVE_F64_AW` fallback. A wide
    /// classifier-head GEMM is otherwise the single longest uninterrupted op
    /// on the spec-matrix root output-bound path and can overrun the verifier
    /// timeout.
    ///
    /// The batched (N-D) sibling's bounded-host lane is `crown_batched.rs`'s
    /// #pollable-faer arm (`NY_POLLABLE_FAER=1`, default OFF), which converts
    /// the historical "no pollable host implementation" refusal into the same
    /// chunked-faer discipline — bit-identical bounds, polls between blocks.
    ///
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

    /// Deadline-PRESENCE-independent CROWN backward (`#deadline-parity`).
    ///
    /// Publishes the SAME certified f64 bound whether or not a deadline is
    /// supplied, polling only when one is present. Use this wherever a bound
    /// must not depend on the mere existence of a deadline — see
    /// [`crown_single::propagate_linear_pollable_cpu`] for why
    /// [`Self::propagate_linear_with_engine_and_deadline`] is not that.
    #[inline]
    pub(crate) fn propagate_linear_pollable_cpu<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<Cow<'a, LinearBounds>> {
        crown_single::propagate_linear_pollable_cpu(self, bounds, engine, deadline)
    }

    /// Batched CROWN backward propagation through linear layer (N-D batch dims).
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        crown_batched::propagate_linear_batched(self, bounds)
    }

    /// Batched CROWN backward with an optional GEMM engine.
    ///
    /// A bounded host facade (an engine with
    /// [`ny_core::GemmEngine::forbids_unbounded_cpu_fallback`], i.e. finite
    /// deadline authority) historically got an unconditional refusal here.
    /// With `NY_POLLABLE_FAER=1` it instead runs the #pollable-faer host
    /// implementation: the identical certified f64 computation, row-block
    /// chunked with the facade's deadline polled between blocks
    /// (see `crown_batched.rs`).
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
