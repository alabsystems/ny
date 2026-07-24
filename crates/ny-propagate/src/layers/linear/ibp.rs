// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (Interval Bound Propagation) for linear layers.
//!
//! For x in [l, u] and y = Wx + b:
//! - W+ = max(W, 0), W- = min(W, 0)
//! - lower_y = W+ @ l + W- @ u + b
//! - upper_y = W+ @ u + W- @ l + b

use std::borrow::Cow;

use faer::Mat;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_dim_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, L2Constraint, RepairStrategy};
use tracing::debug;

use super::LinearLayer;
use crate::faer_parallelism::mat_mul;
use crate::layers::common::BoundPropagation;

impl LinearLayer {
    /// IBP propagation with optional GEMM-engine acceleration.
    ///
    /// When `engine` is provided, both 1-D and N-D inputs are flattened to a
    /// `[batch, in_features]` matrix and evaluated via `GemmEngine::gemm_f32`.
    /// This mirrors the existing CROWN engine path and is intended for
    /// performance-sensitive callers such as PGD.
    ///
    /// The engine path may differ slightly from `propagate_ibp()` on 1-D inputs
    /// because it accumulates in f32 instead of the scalar path's f64 dot
    /// products. `propagate_ibp_sound()` remains the soundness-first API.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let Some(engine) = engine else {
            return self.propagate_ibp(input);
        };

        match propagate_ibp_via_gemm(self, input, engine) {
            // The GEMM path produces the same box interval as the CPU path, so
            // apply the same sound Cauchy–Schwarz intersection when the input
            // carries an L2 constraint (only tightens; no-op otherwise).
            Ok(bounds) => Ok(intersect_with_l2_cauchy_schwarz(self, input, bounds)),
            Err(err) => {
                debug!("GEMM engine failed for Linear IBP, falling back to CPU: {err}");
                self.propagate_ibp(input)
            }
        }
    }

    /// Sound IBP propagation with directed rounding proportional to dot product size.
    ///
    /// For a linear layer y = W+·l + W-·u + b, each output element involves:
    /// - `in_features` multiply-accumulate operations (MACs) in W+·l
    /// - `in_features` MACs in W-·u
    /// - 1 addition combining the two matmul results
    /// - 1 bias addition (if present)
    ///
    /// Total rounding error bound: `in_features + 2` ULPs per output element.
    ///
    /// Reference: Higham, "Accuracy and Stability of Numerical Algorithms",
    /// Theorem 3.1. For a sum of n non-negative terms, the accumulated
    /// rounding error is at most (n-1) ULPs. Each matmul produces a sum of
    /// `in_features` non-negative terms (since w_pos >= 0 and x_lower >= any).
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut result = self.propagate_ibp(input)?;
        // in_features MACs + 1 for combining two matmul results + 1 for bias
        let rounding_ulps = u32::try_from(self.in_features())
            .unwrap_or(u32::MAX)
            .saturating_add(2);
        result.round_for_soundness_n_ulps_inplace(rounding_ulps);
        Ok(result)
    }
}

impl BoundPropagation for LinearLayer {
    /// IBP for linear layer: y = Wx + b
    ///
    /// Supports N-D batched inputs where the last dimension must match in_features().
    /// For input shape [...batch_dims..., in_features], output is [...batch_dims..., out_features].
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "Linear IBP: rank-0 (scalar) input not supported".to_string(),
            ));
        }
        let box_result = if ndim == 1 {
            propagate_ibp_1d(self, input)?
        } else {
            propagate_ibp_nd(self, input)?
        };
        // THE LEVER: if the input carries a per-slice L2 (Euclidean-ball)
        // constraint from an upstream normalization, intersect the decorrelated
        // box interval with the EXACT Cauchy–Schwarz row bound. Intersection
        // only tightens; a non-applicable / malformed constraint is ignored.
        Ok(intersect_with_l2_cauchy_schwarz(self, input, box_result))
    }

    /// CROWN backward propagation through linear layer.
    ///
    /// Delegates to the crown_single module.
    #[inline]
    fn propagate_linear<'a>(
        &self,
        bounds: &'a crate::LinearBounds,
    ) -> Result<Cow<'a, crate::LinearBounds>> {
        super::crown_single::propagate_linear_cpu(self, bounds)
    }
}

/// IBP for 1D (unbatched) input.
fn propagate_ibp_1d(layer: &LinearLayer, input: &BoundedTensor) -> Result<BoundedTensor> {
    let shape = input.shape();
    let in_len = shape[0];
    if in_len != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_len],
        });
    }

    let x_lower = input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![in_len],
            got: input.lower().shape().to_vec(),
        })?;

    let x_upper = input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![in_len],
            got: input.upper().shape().to_vec(),
        })?;

    // f64 accumulation for IBP dot products to prevent precision loss for large
    // in_features. Each dot product accumulates in_features multiply-add terms;
    // f32 sum loses ~log2(n) bits of precision.
    // Reference: Higham, "Accuracy and Stability of Numerical Algorithms", Theorem 3.1.
    // Part of #2423.
    let w_pos_f64 = layer.w_pos().mapv(|x| x as f64);
    let w_neg_f64 = layer.w_neg().mapv(|x| x as f64);
    let xl_f64 = x_lower.mapv(|x| x as f64);
    let xu_f64 = x_upper.mapv(|x| x as f64);

    let lower_y_f64 = w_pos_f64.dot(&xl_f64) + w_neg_f64.dot(&xu_f64);
    let upper_y_f64 = w_pos_f64.dot(&xu_f64) + w_neg_f64.dot(&xl_f64);

    let (lower_y_f64, upper_y_f64) = if let Some(ref b) = layer.bias {
        let b_f64 = b.mapv(|x| x as f64);
        (lower_y_f64 + &b_f64, upper_y_f64 + b_f64)
    } else {
        (lower_y_f64, upper_y_f64)
    };

    // Directed rounding on f64→f32 cast: lower → next_down, upper → next_up.
    // This ensures the f32 result is a sound enclosure of the f64 result.
    let lower_y: Array1<f32> = lower_y_f64.mapv(|x| next_down_f32(x as f32));
    let upper_y: Array1<f32> = upper_y_f64.mapv(|x| next_up_f32(x as f32));

    // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #2549).
    BoundedTensor::new_repaired(
        lower_y.into_dyn(),
        upper_y.into_dyn(),
        RepairStrategy::Conservative,
    )
}

/// Compute batch_size from batch dimensions with overflow check.
fn checked_batch_size(batch_dims: &[usize]) -> Result<usize> {
    checked_dim_product(batch_dims, "Linear IBP batch dimensions")
}

fn contiguous_2d_slice(array: &Array2<f32>) -> Cow<'_, [f32]> {
    match array.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.iter().copied().collect()),
    }
}

fn reshape_gemm_output(
    output: Vec<f32>,
    batch_size: usize,
    out_features: usize,
) -> Result<Array2<f32>> {
    let got_len = output.len();
    Array2::from_shape_vec((batch_size, out_features), output).map_err(|_| NyError::ShapeMismatch {
        expected: vec![batch_size, out_features],
        got: vec![got_len],
    })
}

fn broadcast_bias_2d(
    bias: &Array1<f32>,
    batch_size: usize,
    out_features: usize,
) -> Result<Array2<f32>> {
    bias.broadcast((batch_size, out_features))
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: vec![batch_size, out_features],
            got: bias.shape().to_vec(),
        })
        .map(|view| view.to_owned())
}

fn propagate_concrete_via_gemm(
    layer: &LinearLayer,
    input_2d: &Array2<f32>,
    batch_size: usize,
    out_shape: &[usize],
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let output = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            layer.in_features(),
            layer.out_features(),
            contiguous_2d_slice(input_2d).as_ref(),
            // #cora-transpose-cache: cached once on the layer — the per-call
            // rebuild was 62% of ALL samples on a cora PGD profile.
            layer.weight_t_row_major(),
        )?,
        batch_size,
        layer.out_features(),
    )?;

    let output = if let Some(ref bias) = layer.bias {
        output + broadcast_bias_2d(bias, batch_size, layer.out_features())?
    } else {
        output
    };

    BoundedTensor::concrete(reshape_2d_to_nd(output, out_shape)?)
}

fn propagate_ibp_via_gemm(
    layer: &LinearLayer,
    input: &BoundedTensor,
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(NyError::InvalidSpec(
            "Linear IBP: rank-0 (scalar) input not supported".to_string(),
        ));
    }

    let in_features = shape[ndim - 1];
    if in_features != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_features],
        });
    }

    let batch_size = checked_batch_size(&shape[..ndim - 1])?;
    let out_features = layer.out_features();

    let mut out_shape = shape[..ndim - 1].to_vec();
    out_shape.push(out_features);

    // Normalize to standard (row-major) layout before the reshape — a
    // non-leading-axis Concat yields a non-contiguous owned array that
    // `into_shape_with_order` would reject (mislabeled as ShapeMismatch).
    // `as_standard_layout` copies to C-order only when needed; sound.
    let x_lower_2d = input
        .lower()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.lower().shape().to_vec(),
        })?;
    if input.lower() == input.upper() {
        return propagate_concrete_via_gemm(layer, &x_lower_2d, batch_size, &out_shape, engine);
    }
    let x_upper_2d = input
        .upper()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.upper().shape().to_vec(),
        })?;

    let x_lower = contiguous_2d_slice(&x_lower_2d);
    let x_upper = contiguous_2d_slice(&x_upper_2d);
    // #cora-transpose-cache: cached once on the layer (see mod.rs).
    let w_pos_t = layer.w_pos_t_row_major();
    let w_neg_t = layer.w_neg_t_row_major();

    let lower_pos = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_lower.as_ref(),
            w_pos_t,
        )?,
        batch_size,
        out_features,
    )?;
    let lower_neg = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_upper.as_ref(),
            w_neg_t,
        )?,
        batch_size,
        out_features,
    )?;
    let upper_pos = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_upper.as_ref(),
            w_pos_t,
        )?,
        batch_size,
        out_features,
    )?;
    let upper_neg = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_lower.as_ref(),
            w_neg_t,
        )?,
        batch_size,
        out_features,
    )?;

    let (lower_y_2d, upper_y_2d) = if let Some(ref bias) = layer.bias {
        let bias_broadcast = broadcast_bias_2d(bias, batch_size, out_features)?;
        (
            lower_pos + lower_neg + &bias_broadcast,
            upper_pos + upper_neg + bias_broadcast,
        )
    } else {
        (lower_pos + lower_neg, upper_pos + upper_neg)
    };

    let out_lower = reshape_2d_to_nd(lower_y_2d, &out_shape)?;
    let out_upper = reshape_2d_to_nd(upper_y_2d, &out_shape)?;
    // SOUNDNESS (#vnncomp-aw-soundness): the GEMM-engine IBP path also accumulates
    // each matmul in round-to-nearest f32. Same Higham `in_features + 2` ULP
    // widening as the N-D faer path so engine-accelerated IBP can never feed an
    // unsoundly-tight pre-activation interval into the CROWN verdict path.
    let mut result =
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)?;
    let rounding_ulps = u32::try_from(layer.in_features())
        .unwrap_or(u32::MAX)
        .saturating_add(2);
    result.round_for_soundness_n_ulps_inplace(rounding_ulps);
    Ok(result)
}

/// IBP for N-D batched input (last dimension is in_features).
fn propagate_ibp_nd(layer: &LinearLayer, input: &BoundedTensor) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();
    let in_features = shape[ndim - 1];
    if in_features != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_features],
        });
    }

    // Output shape: [...batch_dims..., out_features]
    let mut out_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    out_shape.push(layer.out_features());
    let batch_size = checked_batch_size(&shape[..ndim - 1])?;

    // Reshape input to [batch_size, in_features]. Normalize to standard
    // (row-major) layout first: a non-leading-axis Concat (e.g. channel concat)
    // produces an OWNED but non-contiguous (column-major) array, and
    // `into_shape_with_order` errors on a non-contiguous buffer even for an
    // identity reshape — which was mislabeled as a ShapeMismatch (expected ==
    // got). `as_standard_layout` copies to C-order only when needed; values and
    // bounds are unchanged, so this is sound.
    let x_lower_2d = input
        .lower()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.lower().shape().to_vec(),
        })?;
    let x_upper_2d = input
        .upper()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.upper().shape().to_vec(),
        })?;

    // Convert ndarray to faer matrices (copy into column-major format for optimal matmul)
    // Note: Attempted zero-copy MatRef::from_row_major_slice but it causes 3-5x regression
    // due to strided memory access in matmul. The copy cost is amortized by faster matmul.
    let x_lower_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_lower_2d[[i, j]]);
    let x_upper_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_upper_2d[[i, j]]);

    // IBP matmul: [batch, in] @ [in, out] = [batch, out]
    let lower_pos = mat_mul(&x_lower_faer, layer.w_pos_t_faer());
    let lower_neg = mat_mul(&x_upper_faer, layer.w_neg_t_faer());
    let upper_pos = mat_mul(&x_upper_faer, layer.w_pos_t_faer());
    let upper_neg = mat_mul(&x_lower_faer, layer.w_neg_t_faer());
    let lower_y_faer = &lower_pos + &lower_neg;
    let upper_y_faer = &upper_pos + &upper_neg;

    // Convert faer results back to ndarray
    let out_features = layer.out_features();
    let lower_y_2d =
        Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| lower_y_faer[(i, j)]);
    let upper_y_2d =
        Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| upper_y_faer[(i, j)]);

    // Add bias if present (broadcast across batch)
    let (lower_y_2d, upper_y_2d) = if let Some(ref b) = layer.bias {
        let b_broadcast = b
            .broadcast((batch_size, layer.out_features()))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: vec![batch_size, layer.out_features()],
                got: b.shape().to_vec(),
            })?
            .to_owned();
        (lower_y_2d + &b_broadcast, upper_y_2d + b_broadcast)
    } else {
        (lower_y_2d, upper_y_2d)
    };

    // Reshape back to original batch dimensions + out_features
    let out_lower = reshape_2d_to_nd(lower_y_2d, &out_shape)?;
    let out_upper = reshape_2d_to_nd(upper_y_2d, &out_shape)?;

    // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #2549).
    // SOUNDNESS (#vnncomp-aw-soundness): the N-D faer path accumulates each matmul
    // in round-to-nearest f32 with no directed rounding. Each output element is a
    // sum of `in_features` same-sign terms per matmul plus the combine and bias,
    // so by Higham Thm 3.1 the accumulated error is bounded by `in_features + 2`
    // ULPs of the result. Widen by that many ULPs (lower down / upper up) so this
    // path matches `propagate_ibp_sound` and never under-reports the interval —
    // this matters because N-D IBP feeds pre-activation bounds into the CROWN
    // relaxations on the verdict path.
    let mut result =
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)?;
    let rounding_ulps = u32::try_from(layer.in_features())
        .unwrap_or(u32::MAX)
        .saturating_add(2);
    result.round_for_soundness_n_ulps_inplace(rounding_ulps);
    Ok(result)
}

/// Reshape a 2D array back to N-D target shape.
fn reshape_2d_to_nd(arr: Array2<f32>, target_shape: &[usize]) -> Result<ArrayD<f32>> {
    let src_shape: Vec<usize> = arr.shape().to_vec();
    arr.into_shape_with_order(IxDyn(target_shape))
        .map_err(|_| NyError::ShapeMismatch {
            expected: target_shape.to_vec(),
            got: src_shape,
        })
}

/// Intersect the box IBP result with the exact Cauchy–Schwarz row bound implied
/// by an input L2 (Euclidean-ball) constraint. **Only ever tightens.**
///
/// For `y = W·x + b` and a per-slice constraint `‖x_slice − c_slice‖₂ ≤ r_slice`
/// (the ball axis being the contracted `in_features` axis), each output obeys
///   y[s,o] = W[o,:]·c[s,:] + b[o] + W[o,:]·(x[s,:] − c[s,:]),
///   |W[o,:]·(x[s,:] − c[s,:])| ≤ ‖W[o,:]‖₂ · r_slice    (Cauchy–Schwarz, EXACT),
/// so y[s,o] ∈ nominal[s,o] ± ‖W_o‖₂·r_slice. This interval and the box interval
/// are BOTH sound enclosures of the same true value, so their per-element
/// intersection (max of lowers, min of uppers) is sound and non-empty.
///
/// SOUNDNESS / DIRECTED ROUNDING. `nominal` and `‖W_o‖₂` accumulate in f64; the
/// radius term is rounded UP (`next_up_f32`) and the interval endpoints rounded
/// OUTWARD (`next_down_f32` low / `next_up_f32` high) before intersecting, so the
/// CS interval can never under-cover. The radius supplied by the producer is a
/// proven OUTWARD bound on the true distance. If the constraint is absent, its
/// ball axis is not the contracted axis, or any shape/feature check fails, the
/// box result is returned unchanged (sound: no tightening).
fn intersect_with_l2_cauchy_schwarz(
    layer: &LinearLayer,
    input: &BoundedTensor,
    box_result: BoundedTensor,
) -> BoundedTensor {
    // THE GATE: the lever is active ONLY in a top-level plain IBP pass. Inside
    // any iterative CROWN bound recomputation (alpha-/beta-CROWN) the gate is OFF
    // and this is byte-identical to the pre-lever box result. Skipping is sound
    // (the box already encloses the true value). See `crate::l2_lever_gate`.
    if !crate::l2_lever_gate::l2_lever_active() {
        return box_result;
    }
    let Some(constraint) = input.l2_constraint() else {
        return box_result;
    };
    match try_cauchy_schwarz_tighten(layer, input, constraint, &box_result) {
        Some(tightened) => tightened,
        None => box_result,
    }
}

/// Core Cauchy–Schwarz tightening; returns `None` to fall back to the box.
///
/// ## Cheap O(out + in) nominal — the midpoint identity
///
/// The exact CS bound is `y[s,o] ∈ (W_o·center[s] + b_o) ± ‖W_o‖₂·r[s]`. Computing
/// the nominal `W_o·center[s]` directly is O(out·in) per slice (the old hot loop
/// that hung deep-CROWN). We avoid it with an identity on the box-IBP result.
///
/// For the box IBP of `y = Wz + b` over the Linear-input box `z ∈ [zl, zu]`,
/// `W = W⁺ + W⁻` and (in real arithmetic)
///   box_lo[o] + box_hi[o]
///     = (W⁺·zl + W⁻·zu + b) + (W⁺·zu + W⁻·zl + b)
///     = (W⁺ + W⁻)·(zl + zu) + 2b = W·(zl + zu) + 2b,
/// so the box-result midpoint `mid_o := (box_lo[o]+box_hi[o])/2 = W_o·z_mid + b_o`
/// EXACTLY, where `z_mid = (zl+zu)/2` is the input-box midpoint — O(1) per output,
/// no inner loop. This is the nominal centred at `z_mid` instead of `center`.
///
/// ## Recentring margin `d` (sound for ANY centre — not just symmetric boxes)
///
/// `z_mid` need not equal the ball centre. Writing `y_o = W_o·z_mid + W_o·(z−z_mid)`
/// and using `‖z − z_mid‖₂ ≤ ‖z − center‖₂ + ‖center − z_mid‖₂ ≤ r + d` (triangle
/// inequality, with `d := ‖z_mid − center‖₂` computed ONCE per slice — O(in), not
/// O(out·in)), Cauchy–Schwarz gives
///   |W_o·(z − z_mid)| ≤ ‖W_o‖₂·(r + d),
/// hence `y_o ∈ mid_o ± ‖W_o‖₂·(r + d)`. EXACT in real arithmetic. For the
/// Standard-LayerNorm IBP box `d ≈ 0` (the centred-(x−mean) interval is symmetric
/// about `beta`), so this is byte-for-byte as tight as the old `W·center` nominal;
/// for RMSNorm (origin-centred, asymmetric box) `d > 0` recovers soundness at a
/// modest widening. Either way the result is ⊆ the box (intersection) and never
/// looser than before.
///
/// ## f32 rounding of the midpoint (extra outward margin `μ`)
///
/// `mid_o` is computed from the *directed-rounded* f32 box endpoints, which were
/// widened OUTWARD by `in_features + 2` ULPs (Higham) from the real box endpoints
/// `true_lo, true_hi` (whose midpoint is the real `W_o·z_mid + b_o`). Since
/// `box_lo ≤ true_lo` and `box_hi ≥ true_hi`,
///   |fl(mid_o) − (W_o·z_mid + b_o)|
///     ≤ ((box_hi − true_hi) + (true_lo − box_lo))/2 + ½·ulp(mid_o)
///     ≤ (in_features + 2.5)·ulp(max(|box_lo|,|box_hi|)),
/// so we fold a generous `μ_o = (in_features + 4)·ulp(max(|box_lo[o]|,|box_hi[o]|))`
/// into the per-output delta (rounded up). Combined with the radius/`d` term rounded
/// outward, the CS interval provably encloses the true `y_o`, so intersecting it
/// with the box only removes infeasible mass.
fn try_cauchy_schwarz_tighten(
    layer: &LinearLayer,
    input: &BoundedTensor,
    constraint: &L2Constraint,
    box_result: &BoundedTensor,
) -> Option<BoundedTensor> {
    let in_shape = input.shape();
    let ndim = in_shape.len();
    if ndim == 0 {
        return None;
    }
    // The ball must be taken over the axis the Linear contracts (last axis).
    if constraint.axis() != ndim - 1 {
        return None;
    }
    let in_features = in_shape[ndim - 1];
    if in_features != layer.in_features() {
        return None;
    }
    let out_features = layer.out_features();

    // The nominal is now O(out) via the box-midpoint identity (see doc above) plus
    // an O(in) per-slice recentring norm `d`, so there is no O(out·in) per-call cost
    // to guard against — the old `CS_MAX_WEIGHT_ELEMS` cap is gone. This makes the
    // lever affordable even inside CROWN's intermediate-bound IBP passes.

    // Flatten batch dims; the constraint center matches the input shape and its
    // radius is per-batch-slice (rank ndim-1).
    let batch_size = checked_batch_size(&in_shape[..ndim - 1]).ok()?;
    if constraint.center().shape() != in_shape {
        return None;
    }
    let radius = constraint.radius();
    let center_2d = constraint
        .center()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;
    let radius_flat = radius
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(batch_size)
        .ok()?;

    // Per-output-row ‖W[o,:]‖₂, precomputed once on the layer (this was an O(out·in)
    // hot loop recomputed every IBP call — the table-transformer deep-CROWN hang).
    // Already rounded outward at construction; reject the whole tightening if any
    // row norm is non-finite.
    let w_row_l2 = layer.row_l2_norms();
    if w_row_l2.iter().any(|n| !n.is_finite()) {
        return None;
    }

    // The Linear INPUT box `z` (flattened) — its per-slice midpoint `z_mid` is the
    // implicit centre of the box-result midpoint identity, and the per-slice
    // distance `d = ‖z_mid − center‖₂` is the recentring margin (computed O(in)
    // per slice below, never O(out·in)).
    let (in_lower, in_upper) = input.lower_upper();
    let in_lower_2d = in_lower
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;
    let in_upper_2d = in_upper
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;

    let (box_lower, box_upper) = box_result.lower_upper();
    // Operate on flat [batch_size, out_features] views of the box result.
    let mut out_lower = box_lower
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, out_features))
        .ok()?;
    let mut out_upper = box_upper
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, out_features))
        .ok()?;

    for s in 0..batch_size {
        let r = radius_flat[s];
        if !(r.is_finite() && r >= 0.0) {
            // Malformed slice radius: leave this slice's box bound untouched.
            continue;
        }

        // d = ‖z_mid − center‖₂ for THIS slice, in f64, rounded UP to f32 (it adds
        // to the radius → round outward). O(in) per slice — paid once, not per
        // output. z_mid = (in_lo + in_hi)/2 (the Linear-input box midpoint).
        let mut d_sumsq = 0.0_f64;
        let mut center_finite = true;
        for j in 0..in_features {
            let zl = in_lower_2d[[s, j]] as f64;
            let zu = in_upper_2d[[s, j]] as f64;
            let cj = center_2d[[s, j]] as f64;
            if !(zl.is_finite() && zu.is_finite() && cj.is_finite()) {
                center_finite = false;
                break;
            }
            // Bit-identical to `0.5 * (zl + zu)`: finite f32-cast operands stay on
            // f64::midpoint's non-overflow `(a + b) * 0.5` path.
            let z_mid = f64::midpoint(zl, zu);
            let diff = z_mid - cj;
            d_sumsq += diff * diff;
        }
        if !center_finite {
            // Non-finite input/center on this slice: leave its box bound untouched.
            continue;
        }
        let d = next_up_f32(d_sumsq.sqrt() as f32);
        // r_eff = r + d, rounded UP (the total CS radius from z_mid).
        let r_eff = next_up_f32(r + d);
        if !r_eff.is_finite() {
            continue;
        }

        for o in 0..out_features {
            let bl = out_lower[[s, o]];
            let bu = out_upper[[s, o]];
            if !(bl.is_finite() && bu.is_finite()) {
                continue;
            }
            // nominal `mid_o = (box_lo + box_hi)/2 = W_o·z_mid + b_o` EXACTLY in
            // real arithmetic (the box-midpoint identity). f32 add then ×0.5.
            // Kept verbatim: `bl + bu` overflowing to ±inf must reach the guard
            // below (f32::midpoint would fabricate a finite center instead).
            #[allow(clippy::manual_midpoint)]
            let mid = 0.5 * (bl + bu);
            if !mid.is_finite() {
                // bl + bu overflowed to ±inf: skip (leave box untouched — sound).
                continue;
            }

            // delta = ‖W_o‖₂·(r + d) + μ_o, rounded UP (outward). μ_o accounts for
            // the f32 rounding of `mid` vs the real `W_o·z_mid + b_o`: the box
            // endpoints were widened OUTWARD by `in_features + 2` ULPs, so the
            // midpoint is off by ≤ (in_features + 2.5)·ulp(max|endpoint|); we use a
            // generous (in_features + 4)·ulp margin.
            let radius_term = next_up_f32(w_row_l2[o] * r_eff);
            let max_abs = bl.abs().max(bu.abs());
            let ulp_max = next_up_f32(max_abs) - max_abs; // 1 ULP at the larger endpoint
            let mu = next_up_f32(((in_features as f32) + 4.0) * ulp_max);
            let delta = next_up_f32(radius_term + mu);
            if !delta.is_finite() {
                continue;
            }
            let cs_lower = next_down_f32(next_down_f32(mid) - delta);
            let cs_upper = next_up_f32(next_up_f32(mid) + delta);

            // Intersect: keep the tighter (higher lower / lower upper). Both the
            // box and CS endpoints are sound enclosures of the true value, so the
            // intersection still contains it (no inversion against the box).
            let new_l = if cs_lower > bl { cs_lower } else { bl };
            let new_u = if cs_upper < bu { cs_upper } else { bu };
            // Final guard: never emit an inverted interval (rounding paranoia).
            if new_l <= new_u {
                out_lower[[s, o]] = new_l;
                out_upper[[s, o]] = new_u;
            }
        }
    }

    let mut out_shape: Vec<usize> = in_shape[..ndim - 1].to_vec();
    out_shape.push(out_features);
    let lower_nd = reshape_2d_to_nd(out_lower, &out_shape).ok()?;
    let upper_nd = reshape_2d_to_nd(out_upper, &out_shape).ok()?;
    // Tightened box only; the output's own L2 sphere is not propagated here.
    BoundedTensor::new_repaired(lower_nd, upper_nd, RepairStrategy::Conservative).ok()
}
