// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Star-set affine form for correlation-exact reachability (S1-2 foundation).
//!
//! A *star set* upgrades a [`ZonotopeTensor`] with a generator-constraint polytope:
//!
//! ```text
//!     X = { c + G·α : A·α ≤ b },   α ∈ [-1, 1]^m
//! ```
//!
//! where `c` is the center, `G` are the generators (the zonotope error-term
//! coefficients, one column per error symbol `αᵢ`), and `A·α ≤ b` is an optional
//! *predicate* polytope over the error symbols. With **empty** `A, b` the predicate
//! is vacuous and the star reduces exactly to the plain zonotope `{ c + G·α : α ∈ box }`.
//!
//! # Why a star (vs a plain zonotope)
//!
//! The predicate `A·α ≤ b` is what lets reachability track ReLU-split correlations
//! *exactly*: a split `x ≥ 0` / `x < 0` becomes a linear constraint on the error
//! symbols instead of a fresh over-approximating relaxation. It is therefore a
//! promising sound-method hypothesis for rows where CROWN branch-and-bound times
//! out. It is **not** inferred from NNV's 2025 CIFAR results: that submission used
//! probabilistic `cp-star` (a sampled conformal surrogate), not deterministic
//! STAR-set reachability. Populating the predicate is the ReLU transformer (S3-4)
//! and is **out of scope here** — S1-2 is the affine skeleton plus the box-α bound
//! and the IBP-parity soundness gate.
//!
//! # Domain convention
//!
//! `α ∈ [-1, 1]^m` is the *base* domain, inherited from the zonotope error-symbol
//! convention (each `αᵢ ∈ [-1, 1]`). The rows of `A·α ≤ b` are *additional* predicate
//! constraints layered on top of the box; they can only ever shrink the set. Because
//! of that, [`Star::interval_bounds`] — which uses the box domain and ignores the
//! predicate — is always a **sound over-approximation** of the true (constrained) set.
//! A tighter, predicate-aware bound needs a per-coordinate LP; see [`Star::bounds_lp`].
//!
//! # Status
//!
//! New, default-off, and **unwired** into any verdict path. Nothing outside tests
//! constructs or calls a `Star`. Soundness of the affine transformers is pinned by
//! IBP-parity proptests (see `tests/star_parity.rs`).

use ndarray::{Array1, Array2, Array4, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};

use crate::unfold::{inplace_unfold, unfold_output_size};
use crate::BoundedTensor;

use super::ZonotopeTensor;

mod blocked_conv2d;

pub use blocked_conv2d::{StarConv2dBlockLimits, StarConv2dBlockPlan};

/// A star set `X = { c + G·α : A·α ≤ b }`, `α ∈ [-1, 1]^m`.
///
/// The center `c` and generators `G` are stored in the wrapped [`ZonotopeTensor`]
/// (`coeffs[0]` = center, `coeffs[1..]` = the `m` generator columns). The predicate
/// polytope is `a` (shape `(k, m)`) and `b` (shape `(k,)`); empty `a, b` (`k = 0`)
/// means the star is a plain zonotope over the α-box.
///
/// Every affine transformer maps `c, G` linearly and leaves `A, b` **unchanged**
/// (an affine map acts on the value space, not the error-symbol space).
#[derive(Debug, Clone)]
pub struct Star {
    /// Center + generators, reusing the zonotope layout (`coeffs[0]` = center).
    zono: ZonotopeTensor,
    /// Predicate constraint matrix, shape `(k, m)` with `m = zono.n_error_terms()`.
    /// Empty (`k = 0`) ⇒ pure zonotope over the α-box.
    a: Array2<f32>,
    /// Predicate constraint right-hand side, shape `(k,)`.
    b: Array1<f32>,
}

impl Star {
    /// Wrap a zonotope as a star with an **empty** predicate (`α ∈ box`).
    ///
    /// The result is mathematically identical to the input zonotope.
    pub fn from_zonotope(zono: ZonotopeTensor) -> Self {
        let m = zono.n_error_terms();
        Self {
            zono,
            a: Array2::<f32>::zeros((0, m)),
            b: Array1::<f32>::zeros(0),
        }
    }

    /// Build a star from a box input, one error symbol per element (empty predicate).
    ///
    /// Convenience wrapper over [`ZonotopeTensor::from_input_elementwise`]; the
    /// resulting star is a pure zonotope over `α ∈ [-1, 1]^{n_elements}`.
    pub fn from_input_box(values: &ArrayD<f32>, epsilon: f32) -> Self {
        Self::from_zonotope(ZonotopeTensor::from_input_elementwise(values, epsilon))
    }

    /// Construct a star from explicit center/generators and a predicate polytope.
    ///
    /// # Errors
    /// Returns [`NyError::InvalidSpec`] if `a`/`b` are shape-inconsistent: `a` must be
    /// `(k, m)` with `m = zono.n_error_terms()` and `b` must be `(k,)`.
    pub fn new(zono: ZonotopeTensor, a: Array2<f32>, b: Array1<f32>) -> Result<Self> {
        let m = zono.n_error_terms();
        if a.nrows() != b.len() {
            return Err(NyError::InvalidSpec(format!(
                "Star::new: constraint rows {} != rhs len {}",
                a.nrows(),
                b.len()
            )));
        }
        if a.ncols() != m {
            return Err(NyError::InvalidSpec(format!(
                "Star::new: constraint cols {} != alpha dim (n_error_terms) {}",
                a.ncols(),
                m
            )));
        }
        Ok(Self { zono, a, b })
    }

    /// Read-only view of the wrapped center+generators zonotope.
    pub fn zonotope(&self) -> &ZonotopeTensor {
        &self.zono
    }

    /// Consume the star and return its center+generators zonotope, dropping the predicate.
    pub fn into_zonotope(self) -> ZonotopeTensor {
        self.zono
    }

    /// Predicate polytope `(A, b)` of `A·α ≤ b` (both empty ⇒ pure zonotope).
    pub fn constraints(&self) -> (&Array2<f32>, &Array1<f32>) {
        (&self.a, &self.b)
    }

    /// Center tensor (point estimate), shape [`Star::shape`].
    pub fn center(&self) -> ArrayD<f32> {
        self.zono.center()
    }

    /// Element shape of the value space (excludes the error-symbol axis).
    pub fn shape(&self) -> &[usize] {
        self.zono.shape()
    }

    /// Number of error symbols `m` (the dimension of `α`).
    pub fn alpha_dim(&self) -> usize {
        self.zono.n_error_terms()
    }

    /// Number of predicate constraints `k` (rows of `A·α ≤ b`).
    pub fn num_constraints(&self) -> usize {
        self.b.len()
    }

    /// Whether the star is a plain zonotope (empty predicate ⇒ `α` ranges the full box).
    pub fn is_zonotope(&self) -> bool {
        self.b.is_empty()
    }

    // ---------------------------------------------------------------------
    // Affine transformers.  Each maps c, G linearly and leaves A, b unchanged.
    // ---------------------------------------------------------------------

    /// `Gemm`/`Linear`: `c' = W·c + bias`, `G' = W·G`, predicate unchanged.
    ///
    /// `weight` is `(out_features, in_features)`, applied to the last axis of the
    /// value space. Delegates to [`ZonotopeTensor::linear`] for the center and every
    /// generator column, so the resulting bounds are bit-identical to the zonotope
    /// affine path.
    ///
    /// # Errors
    /// Propagates shape errors from [`ZonotopeTensor::linear`].
    pub fn gemm(&self, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Result<Self> {
        let zono = self.zono.linear(weight, bias)?;
        // Affine map does not touch the error-symbol space ⇒ (A, b) carry over unchanged.
        Ok(Self {
            zono,
            a: self.a.clone(),
            b: self.b.clone(),
        })
    }

    /// `Conv2d` (cross-correlation, BN assumed folded into the weights upstream).
    ///
    /// Applies the same conv-as-matmul (im2col via [`crate::unfold::inplace_unfold`])
    /// used by the CROWN conv path to the center and to **each generator column**;
    /// `bias` is added to the center only. The predicate `A·α ≤ b` is unchanged.
    ///
    /// * `weight` — `(out_channels, in_channels, kH, kW)`.
    /// * `bias` — optional `(out_channels,)`, added to the center.
    /// * `stride` — `(sH, sW)`.
    /// * `padding` — `(padH, padW)`, symmetric zero-padding.
    ///
    /// Input value shape must be `(C, H, W)`; output value shape is `(out_C, out_H, out_W)`.
    ///
    /// # Errors
    /// Returns [`NyError::InvalidSpec`]/[`NyError::ShapeMismatch`] on a non-`(C,H,W)`
    /// input, an `in_channels` mismatch, or an infeasible kernel/stride/padding.
    pub fn conv2d(
        &self,
        weight: &Array4<f32>,
        bias: Option<&Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        let es = self.zono.shape();
        if es.len() != 3 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d expects (C, H, W) value shape, got {:?}",
                es
            )));
        }
        let (c_in, h, w) = (es[0], es[1], es[2]);

        let wshape = weight.shape();
        let (out_c, w_cin, kh, kw) = (wshape[0], wshape[1], wshape[2], wshape[3]);
        if w_cin != c_in {
            return Err(NyError::shape_mismatch(
                vec![out_c, c_in, kh, kw],
                wshape.to_vec(),
            ));
        }

        let (sh, sw) = stride;
        let (pad_h, pad_w) = padding;
        if sh == 0 || sw == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d requires non-zero stride, got {stride:?}"
            )));
        }
        if kh == 0 || kw == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d requires a non-empty kernel, got ({kh}, {kw})"
            )));
        }
        let padded_h = h
            .checked_add(pad_h)
            .and_then(|value| value.checked_add(pad_h))
            .ok_or_else(|| {
                NyError::InvalidSpec("Star::conv2d vertical padding overflows usize".to_string())
            })?;
        let padded_w = w
            .checked_add(pad_w)
            .and_then(|value| value.checked_add(pad_w))
            .ok_or_else(|| {
                NyError::InvalidSpec("Star::conv2d horizontal padding overflows usize".to_string())
            })?;
        if padded_h < kh || padded_w < kw {
            return Err(NyError::ShapeMismatch {
                expected: vec![kh, kw],
                got: vec![padded_h, padded_w],
            });
        }
        // inplace_unfold padding order is (left, right, top, bottom).
        let pad = (pad_w, pad_w, pad_h, pad_h);
        let (out_h, out_w) = unfold_output_size(h, w, (kh, kw), (sh, sw), pad);

        // Weight as a (out_C, C·kH·kW) matrix; the inner (c, ki, kj) ordering matches
        // the unfolded patch layout (out_h, out_w, c, ki, kj).
        let patch = c_in * kh * kw;
        let w_flat: Array2<f32> = weight
            .as_standard_layout()
            .to_owned()
            .into_shape_with_order((out_c, patch))
            .map_err(|e| NyError::InvalidSpec(format!("Star::conv2d weight reshape: {e}")))?;

        let n_rows = 1 + self.zono.n_error_terms();
        let mut result = ArrayD::<f32>::zeros(IxDyn(&[n_rows, out_c, out_h, out_w]));

        for row in 0..n_rows {
            // Each row (center or one generator) is a (C, H, W) image.
            let img = self.zono.coeffs().index_axis(Axis(0), row).to_owned();
            let unfolded = inplace_unfold(&img, (kh, kw), (sh, sw), pad)?; // (out_h,out_w,C,kH,kW)
            let unfolded2 = unfolded
                .into_shape_with_order((out_h * out_w, patch))
                .map_err(|e| NyError::InvalidSpec(format!("Star::conv2d patch reshape: {e}")))?;
            // (out_C, patch) · (patch, out_h·out_w) = (out_C, out_h·out_w)
            let out2 = w_flat.dot(&unfolded2.t());
            let out3 = out2
                .into_shape_with_order((out_c, out_h, out_w))
                .map_err(|e| NyError::InvalidSpec(format!("Star::conv2d output reshape: {e}")))?;
            result.index_axis_mut(Axis(0), row).assign(&out3.into_dyn());
        }

        // Bias applies to the center row only (generators carry no constant term).
        if let Some(bvec) = bias {
            if bvec.len() != out_c {
                return Err(NyError::shape_mismatch(vec![out_c], vec![bvec.len()]));
            }
            let mut center = result.index_axis_mut(Axis(0), 0); // (out_C, out_H, out_W)
            for co in 0..out_c {
                let bc = bvec[co];
                center.index_axis_mut(Axis(0), co).mapv_inplace(|v| v + bc);
            }
        }

        let zono = ZonotopeTensor::new(result)?;
        Ok(Self {
            zono,
            a: self.a.clone(),
            b: self.b.clone(),
        })
    }

    /// Residual `Add` of two stars: `c' = c1 + c2`, `G' = G1 + G2`.
    ///
    /// The two branches are assumed to derive from a **shared** input `α` (the usual
    /// residual `y = f(x) + x` case), so equal error symbols are added positionally.
    /// Differing generator counts are reconciled by [`ZonotopeTensor::expand_to_match`]
    /// (shared-prefix alignment, padding the shorter branch's generators with zeros).
    /// The two predicates are combined into `[[A1'];[A2']]·α ≤ [b1; b2]`, each padded
    /// with zero columns for the symbols it does not own — sound because it requires
    /// `α` to satisfy both branches' predicates over the shared symbol space.
    ///
    /// # Errors
    /// Returns [`NyError::ShapeMismatch`] if the value shapes differ.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.zono.shape() != other.zono.shape() {
            return Err(NyError::shape_mismatch(
                self.zono.shape().to_vec(),
                other.zono.shape().to_vec(),
            ));
        }

        // Align error symbols (shared prefix), then add centers+generators.
        let (z1, z2) = self.zono.expand_to_match(&other.zono)?;
        let sum = z1.add(&z2)?;
        let m = sum.n_error_terms();

        // Reconcile predicates over the (possibly widened) symbol space.
        let (a, b) = if self.b.is_empty() && other.b.is_empty() {
            (Array2::<f32>::zeros((0, m)), Array1::<f32>::zeros(0))
        } else {
            let a1 = pad_constraint_cols(&self.a, m);
            let a2 = pad_constraint_cols(&other.a, m);
            let a = if a1.nrows() == 0 {
                a2
            } else if a2.nrows() == 0 {
                a1
            } else {
                ndarray::concatenate(Axis(0), &[a1.view(), a2.view()]).map_err(|e| {
                    NyError::InvalidSpec(format!("Star::add constraint stack failed: {e}"))
                })?
            };
            let b = ndarray::concatenate(Axis(0), &[self.b.view(), other.b.view()])
                .map_err(|e| NyError::InvalidSpec(format!("Star::add rhs stack failed: {e}")))?;
            (a, b)
        };

        Ok(Self { zono: sum, a, b })
    }

    /// `Flatten`/reshape: identity on the affine data, only value-shape metadata changes.
    ///
    /// Delegates to [`ZonotopeTensor::reshape`]; the error-symbol space (and thus the
    /// predicate `A·α ≤ b`) is untouched. `flatten()` is `reshape(&[num_elements])`.
    ///
    /// # Errors
    /// Returns [`NyError::ShapeMismatch`] if the element count changes.
    pub fn reshape(&self, target_shape: &[usize]) -> Result<Self> {
        let zono = self.zono.reshape(target_shape)?;
        Ok(Self {
            zono,
            a: self.a.clone(),
            b: self.b.clone(),
        })
    }

    /// Flatten the value space to a 1-D vector (rank-1 [`Star::reshape`]).
    ///
    /// # Errors
    /// Propagates shape errors from [`ZonotopeTensor::reshape`].
    pub fn flatten(&self) -> Result<Self> {
        let n = self.zono.len();
        self.reshape(&[n])
    }

    // ---------------------------------------------------------------------
    // Bounds extraction.
    // ---------------------------------------------------------------------

    /// Per-coordinate interval bounds `[lo, hi]` via the **box-α** rule.
    ///
    /// For the pure-zonotope case (empty predicate) this is exactly `c_i ± Σⱼ |G_ij|`
    /// (the existing zonotope bound). When a predicate `A·α ≤ b` is present this bound
    /// **ignores** it and ranges `α` over the full box — a sound over-approximation,
    /// since intersecting with the predicate can only shrink the set. A tighter,
    /// predicate-aware bound requires a per-coordinate LP; see [`Star::bounds_lp`].
    ///
    /// # Errors
    /// Returns [`NyError`] if the underlying bounds contain NaN (via
    /// [`ZonotopeTensor::to_bounded_tensor`]).
    pub fn interval_bounds(&self) -> Result<BoundedTensor> {
        self.zono.to_bounded_tensor()
    }

    /// Predicate-aware per-coordinate bounds via LP over `{ α ∈ box : A·α ≤ b }`.
    ///
    /// **Not implemented in `ny-tensor` (stub).** Tightening past [`Star::interval_bounds`]
    /// requires solving `min/max cᵢ + Gᵢ·α  s.t.  A·α ≤ b, α ∈ box` per output coordinate,
    /// i.e. an LP solver. The only in-tree LP/MIP backend lives in `ny-mip`, and the
    /// dependency edge is `ny-mip → ny-tensor` (see `crates/ny-mip/Cargo.toml`): letting
    /// `ny-tensor` reach the solver would invert that edge and create a cycle. So the
    /// output-LP must live *above* `ny-tensor` — in `ny-mip`/`ny-propagate`, consuming the
    /// star's `(center, generators, A, b)` — and cannot be implemented here.
    ///
    // TODO(S3-4): once the ReLU transformer populates A·α ≤ b, implement the per-coordinate
    // output-LP in ny-mip/ny-propagate (where the solver is reachable) and call it from there.
    ///
    /// # Errors
    /// Always returns [`NyError::InvalidSpec`] describing the dependency-direction reason.
    pub fn bounds_lp(&self) -> Result<BoundedTensor> {
        Err(NyError::InvalidSpec(
            "Star::bounds_lp is a stub: the per-coordinate output-LP requires an LP solver, \
             but ny-tensor cannot depend on ny-mip (edge is ny-mip -> ny-tensor; reversing it \
             would cycle). Implement it in ny-mip/ny-propagate over the star's (c, G, A, b). \
             Use interval_bounds() for the sound box-alpha bound."
                .to_string(),
        ))
    }
}

/// Pad a `(k, m0)` constraint matrix to `(k, m)` by appending zero columns (`m ≥ m0`).
///
/// Widening the α space with symbols a branch does not constrain is a no-op on that
/// branch's predicate: the new symbols get zero coefficients.
fn pad_constraint_cols(a: &Array2<f32>, m: usize) -> Array2<f32> {
    let (k, m0) = (a.nrows(), a.ncols());
    if m0 == m || k == 0 {
        // Reindex an empty matrix onto the wider space so shapes stay consistent.
        if k == 0 {
            return Array2::<f32>::zeros((0, m));
        }
        return a.clone();
    }
    let mut padded = Array2::<f32>::zeros((k, m));
    padded.slice_mut(ndarray::s![.., 0..m0]).assign(a);
    padded
}
