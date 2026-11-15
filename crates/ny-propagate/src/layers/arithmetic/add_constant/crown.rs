// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense and batched CROWN backward for AddConstant.

use ndarray::{Array1, Array2, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::debug;

use super::super::common::dot_bias_f64;
use super::AddConstantLayer;
use crate::shape::broadcast_shapes;
use crate::{BatchedLinearBounds, LinearBounds};

impl AddConstantLayer {
    /// Shared CROWN backward body for `propagate_linear` (no operand shape) and
    /// `propagate_crown_backward` (operand shape from the pre-activation).
    ///
    /// For y = x + c, the linear relationship is preserved:
    /// `A @ y + b = A @ (x + c) + b = A @ x + (A @ c + b)`, so the coefficient
    /// matrix passes through and only the bias moves by `A @ c`. `A` has shape
    /// (num_outputs, num_inputs) where `num_inputs` counts THIS LAYER'S OUTPUT
    /// positions, so `c` must be the constant broadcast onto those positions.
    pub(super) fn crown_backward_with_operand_shape(
        &self,
        bounds: &LinearBounds,
        operand_shape: Option<&[usize]>,
    ) -> Result<LinearBounds> {
        let num_inputs = bounds.num_inputs();
        let c_flat = self.backward_constant_columns(num_inputs, operand_shape)?;

        // Compute A @ c with f64 accumulation + directed rounding.
        let (lower_c_f64, upper_c_f64) = dot_bias_f64(bounds.lower_a(), bounds.upper_a(), &c_flat);
        let mut lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_c_f64;
        let mut ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_c_f64;

        // Certified coefficient-error transfer through the `A→bias` fold
        // (#vnncomp-aw-soundness). The fold above used the STORED coefficients;
        // with an incoming certificate `|A_stored − A_true| ≤ e` the folded bias
        // can be off by up to the MAGNITUDE charge `Σ_j e_j·|c_j|` per row —
        // magnitudes, because cancellation in the VALUE does not cancel the
        // ERROR. Charge it OUTWARD (Pad's certified constant-fold idiom,
        // `layers/transform/pad.rs`), and pass the err planes through UNCHANGED
        // below: `A` is untouched by this backward, so the incoming certificate
        // holds verbatim on the result. This is what lets `AddConstant` sit in
        // `propagates_coeff_err()` (layer_enum/query.rs) instead of taking the
        // sequential lane's `discharge_coeff_err_to_conservative` vacuous-row
        // degrade when the constant is mixed-sign: the magnitude charge DOMINATES
        // the signed carrier fold `|Σ_j e_j·c_j|` on every constant and equals it
        // exactly when the constant is sign-uniform.
        if bounds.has_coeff_err() {
            let n_rows = lb_f64.len();
            let charge_rows = |err: Option<&Array2<f32>>| -> Vec<f64> {
                let mut rows = vec![0.0f64; n_rows];
                if let Some(e) = err {
                    for (row, slot) in rows.iter_mut().enumerate() {
                        let mut acc = 0.0f64;
                        for (j, &c) in c_flat.iter().enumerate() {
                            // f32×f32 in f64 is exact; the directed f32 casts
                            // below absorb the f64 summation rounding (Pad's
                            // idiom). A non-finite (∞-sanitized) err entry
                            // poisons the row toward ±inf/NaN, which the
                            // constructor firewall degrades to conservative.
                            acc += (e[[row, j]] as f64).abs() * (c as f64).abs();
                        }
                        *slot = acc;
                    }
                }
                rows
            };
            for (row, charge) in charge_rows(bounds.lower_a_err()).into_iter().enumerate() {
                lb_f64[row] -= charge;
            }
            for (row, charge) in charge_rows(bounds.upper_a_err()).into_iter().enumerate() {
                ub_f64[row] += charge;
            }
        }

        let new_lower_b = lb_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = ub_f64.mapv(|x| next_up_f32(x as f32));

        if bounds.has_coeff_err() {
            LinearBounds::new_or_conservative_with_err(
                bounds.lower_a().clone(),
                new_lower_b,
                bounds.upper_a().clone(),
                new_upper_b,
                bounds
                    .lower_a_err()
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros(bounds.lower_a().raw_dim())),
                bounds
                    .upper_a_err()
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros(bounds.upper_a().raw_dim())),
            )
        } else {
            LinearBounds::new_or_conservative(
                bounds.lower_a().clone(),
                new_lower_b,
                bounds.upper_a().clone(),
                new_upper_b,
            )
        }
    }

    /// The constant, laid out onto this layer's OUTPUT positions, as the column
    /// vector the CROWN backward substitution needs.
    ///
    /// THIS IS THE SOUNDNESS SITE. The previous implementation tiled a short
    /// constant as `c[i % const_len]`, which silently ASSUMES the constant
    /// varies along the FASTEST (last) axis. `propagate_ibp` — which is what
    /// `propagate_concrete_point` uses to evaluate the network, i.e. the
    /// reference for what the graph computes — broadcasts by ONNX rules plus
    /// the rank-1 CNN-bias override, under which a per-channel constant varies
    /// along the SLOWEST axis (`c[i / (H*W)]`). For every constant that
    /// broadcasts along a non-last axis (`[C,1,1]`, `[1,C,1,1]`, and the
    /// CNN-bias `[C]`) the two therefore certified DIFFERENT FUNCTIONS. That is
    /// not a loose bound: nothing makes the mis-placed bias conservative, and
    /// on any output where it exceeds the true one the certified interval is
    /// TIGHTER than the truth — a false `unsat`.
    ///
    /// The fix resolves the layout from the operand shape whenever one is
    /// available, using the same `effective_constant_shape` + ONNX broadcast
    /// that `propagate_ibp` uses, and REFUSES (`UnsupportedOp` /
    /// `ShapeMismatch`) whenever it cannot be proven. Both refusals are
    /// fail-closed at every production caller: eligible contexts degrade to
    /// IBP/partial CROWN; contexts with pending paths abort that CROWN lane so a
    /// higher-level fallback can run. Neither path can admit a tighter bound.
    fn backward_constant_columns(
        &self,
        layer_dim: usize,
        operand_shape: Option<&[usize]>,
    ) -> Result<Array1<f32>> {
        // Guard: empty constant would divide by zero in the layout checks.
        if self.constant.is_empty() {
            return Err(NyError::InvalidSpec(
                "AddConstant CROWN backward: constant tensor is empty".to_string(),
            ));
        }
        match operand_shape {
            // A present shape is evidence. If it is inconsistent with the
            // incoming coefficient dimension, propagate the refusal instead of
            // discarding that evidence and retrying an unshaped heuristic.
            Some(shape) => self.backward_constant_columns_from_shape(layer_dim, shape),
            None => self.backward_constant_columns_without_shape(layer_dim),
        }
    }

    /// Layout resolved from the operand's true shape — mirrors `propagate_ibp`.
    fn backward_constant_columns_from_shape(
        &self,
        layer_dim: usize,
        operand_shape: &[usize],
    ) -> Result<Array1<f32>> {
        let const_shape = self.constant.shape();
        let effective_shape = self.effective_broadcast_shape(operand_shape)?;
        let out_shape = broadcast_shapes(operand_shape, &effective_shape).ok_or_else(|| {
            NyError::ShapeMismatch {
                expected: operand_shape.to_vec(),
                got: const_shape.to_vec(),
            }
        })?;
        let out_len = checked_shape_product(&out_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AddConstant CROWN backward: output shape {out_shape:?} overflows usize"
            ))
        })?;
        if out_len != layer_dim {
            // The incoming coefficients do not index this node's outputs. Refuse
            // rather than reinterpret them.
            return Err(NyError::ShapeMismatch {
                expected: vec![layer_dim],
                got: vec![out_len],
            });
        }
        let operand_len = checked_shape_product(operand_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AddConstant CROWN backward: operand shape {operand_shape:?} overflows usize"
            ))
        })?;
        if operand_len != out_len {
            // The CONSTANT expands the bounded operand (ONNX Add broadcasts both
            // ways). The substitution below passes `A` through unchanged, which
            // is only valid when the operand and the output have the SAME
            // positions; the expanding case needs the column reduction
            // `SubConstantLayer::propagate_linear_broadcast_backward` implements
            // and this layer does not. Fail closed.
            //
            // Testing element COUNTS, not shape equality, is deliberate:
            // broadcasting only ever REPLICATES, so `operand_len == out_len`
            // means no operand axis was replicated, and a purely rank-raising
            // broadcast (`[10]` + `[1,10]` -> `[1,10]`, the ordinary ONNX bias
            // authoring) leaves the row-major flat order untouched. Requiring
            // literal shape equality would refuse that everyday case.
            return Err(NyError::UnsupportedOp(format!(
                "AddConstant CROWN backward: constant {const_shape:?} EXPANDS the bounded \
                 operand {operand_shape:?} ({operand_len} positions) to {out_shape:?} \
                 ({out_len} positions); the broadcast column reduction is not implemented \
                 for AddConstant — refusing so the caller degrades to IBP"
            )));
        }
        let reshaped = if effective_shape.as_slice() == const_shape {
            self.constant.clone()
        } else {
            self.constant
                .clone()
                .into_shape_with_order(IxDyn(&effective_shape))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: effective_shape.clone(),
                    got: const_shape.to_vec(),
                })?
        };
        let broadcast =
            reshaped
                .broadcast(IxDyn(&out_shape))
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: out_shape.clone(),
                    got: effective_shape.clone(),
                })?;
        // `iter()` walks a broadcast view in LOGICAL (row-major) order, which is
        // the order the coefficient columns are in. Same pattern as
        // `SubConstantLayer::propagate_linear_broadcast_backward`.
        let c_flat: Array1<f32> = broadcast.iter().copied().collect();
        Ok(c_flat)
    }

    /// Layout that can be proven without the operand shape.
    ///
    /// Only a scalar is sufficient evidence by itself. Even when the constant
    /// has `layer_dim` elements or has a trailing-only shape, an unseen operand
    /// may have been expanded by broadcasting (for example `[1,3] + [2,3]`).
    /// Passing the coefficient columns through unchanged would then target the
    /// wrong predecessor dimension. Non-scalars therefore require the operand
    /// shape through `BoundPropagation::propagate_crown_backward`.
    fn backward_constant_columns_without_shape(&self, layer_dim: usize) -> Result<Array1<f32>> {
        let const_shape = self.constant.shape();
        let const_len = self.constant.len();

        if const_len == 1 {
            let value = self.constant.iter().copied().next().ok_or_else(|| {
                NyError::InvalidSpec(
                    "AddConstant CROWN backward: scalar constant unexpectedly empty".to_string(),
                )
            })?;
            return Ok(Array1::from_elem(layer_dim, value));
        }

        Err(NyError::UnsupportedOp(format!(
            "AddConstant CROWN backward: constant {const_shape:?} ({const_len} elements) against \
             {layer_dim} output positions cannot prove that the bounded operand was not expanded \
             by broadcasting; refusing without the operand shape"
        )))
    }
}

impl AddConstantLayer {
    /// Batched CROWN backward propagation through AddConstant.
    ///
    /// For y = x + c, the linear relationship is:
    /// A @ y + b = A @ (x + c) + b = A @ x + (A @ c + b)
    /// So coefficient matrices stay the same, bias gets A @ c added.
    ///
    /// This compatibility entry point has no operand shape, so it accepts only
    /// scalars. Production enum dispatch calls the shaped method below, which
    /// can also admit proven last-axis vectors.
    ///
    /// Reference: auto_LiRPA Bound.get_bias (einsum over A and bias):
    /// alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/base.py:468-480
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        if self.constant.len() != 1 {
            return Err(NyError::UnsupportedOp(
                "AddConstant batched CROWN: a non-scalar constant requires the operand shape; \
                 use Layer::propagate_crown_backward_batched with pre-activation evidence"
                    .to_string(),
            ));
        }
        self.propagate_linear_batched_with_operand_shape(bounds, bounds.input_shape())
    }

    /// Batched backward with the actual Add operand shape authenticated by the
    /// caller's pre-activation bounds.
    pub(crate) fn propagate_linear_batched_with_operand_shape(
        &self,
        bounds: &BatchedLinearBounds,
        operand_shape: &[usize],
    ) -> Result<BatchedLinearBounds> {
        let const_len = self.constant.len();
        if const_len == 0 {
            return Err(NyError::InvalidSpec(
                "AddConstant batched CROWN: constant tensor is empty".to_string(),
            ));
        }

        let ndim = bounds.lower_a.ndim();
        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "AddConstant batched CROWN: bounds must have at least 2 dimensions".to_string(),
            ));
        }
        let in_dim = bounds.lower_a.shape()[ndim - 1];

        let effective_shape = self.effective_broadcast_shape(operand_shape)?;
        let add_output_shape =
            broadcast_shapes(operand_shape, &effective_shape).ok_or_else(|| {
                NyError::ShapeMismatch {
                    expected: operand_shape.to_vec(),
                    got: self.constant.shape().to_vec(),
                }
            })?;
        if add_output_shape != bounds.input_shape() {
            return Err(NyError::ShapeMismatch {
                expected: bounds.input_shape().to_vec(),
                got: add_output_shape,
            });
        }
        if operand_shape != bounds.input_shape() {
            return Err(NyError::UnsupportedOp(format!(
                "AddConstant batched CROWN: operand {operand_shape:?} is broadcast to layer \
                 output {:?}; the batched substitution cannot reduce replicated predecessor \
                 columns — refusing so the caller falls back",
                bounds.input_shape()
            )));
        }

        // Compute bias contribution A @ c in f64 for soundness.
        let (lower_contrib_f64, upper_contrib_f64) = if const_len == 1 {
            // Scalar fast path: A @ c_broadcast = c * sum(A, axis=-1).
            let c_f64 = self
                .constant
                .iter()
                .next()
                .copied()
                .expect("invariant: const_len == 1 checked above") as f64;
            debug!("AddConstant batched CROWN: scalar c = {}", c_f64);
            let lower_sum = bounds.lower_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));
            let upper_sum = bounds.upper_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));
            (lower_sum.mapv(|v| v * c_f64), upper_sum.mapv(|v| v * c_f64))
        } else {
            // Vector constant path: compute A @ c directly.
            // The layout is resolved against the operand shape and REFUSED when
            // it cannot be proven — see `batched_constant_along_last_axis`.
            let c_flat = self.batched_constant_along_last_axis(in_dim, operand_shape)?;
            debug!("AddConstant batched CROWN: vector c len={}", c_flat.len());

            // Flatten A from [...batch, out_dim, in_dim] to [N, in_dim] for dot product,
            // then reshape result back to [...batch, out_dim].
            let a_shape = bounds.lower_a.shape().to_vec();
            let leading_size: usize = a_shape[..ndim - 1].iter().product();

            let flat_lower_a = bounds
                .lower_a
                .view()
                .into_shape_with_order((leading_size, in_dim))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "AddConstant batched CROWN: reshape lower_a to 2D: {}",
                        e
                    ))
                })?;
            let flat_upper_a = bounds
                .upper_a
                .view()
                .into_shape_with_order((leading_size, in_dim))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "AddConstant batched CROWN: reshape upper_a to 2D: {}",
                        e
                    ))
                })?;

            // A @ c with f64 accumulation.
            let (lower_dot, upper_dot) =
                dot_bias_f64(&flat_lower_a.to_owned(), &flat_upper_a.to_owned(), &c_flat);

            // Reshape back to bias shape [...batch, out_dim].
            let bias_shape: Vec<usize> = a_shape[..ndim - 1].to_vec();
            let lower_contrib = lower_dot
                .into_dyn()
                .into_shape_with_order(IxDyn(&bias_shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "AddConstant batched CROWN: reshape lower contrib: {}",
                        e
                    ))
                })?;
            let upper_contrib = upper_dot
                .into_dyn()
                .into_shape_with_order(IxDyn(&bias_shape))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "AddConstant batched CROWN: reshape upper contrib: {}",
                        e
                    ))
                })?;
            (lower_contrib, upper_contrib)
        };

        // Add to existing bias in f64, then directed rounding to f32.
        // Handles inf + (-inf) = NaN → conservative bound (same as safe_array_add).
        let lb_f64 = bounds.lower_b.mapv(|v| v as f64) + &lower_contrib_f64;
        let ub_f64 = bounds.upper_b.mapv(|v| v as f64) + &upper_contrib_f64;

        let new_lower_b = lb_f64.mapv(|v| {
            let f = v as f32;
            if f.is_nan() {
                f32::NEG_INFINITY // conservative for lower bound
            } else {
                next_down_f32(f)
            }
        });
        let new_upper_b = ub_f64.mapv(|v| {
            let f = v as f32;
            if f.is_nan() {
                f32::INFINITY // conservative for upper bound
            } else {
                next_up_f32(f)
            }
        });

        BatchedLinearBounds::new_or_conservative(
            bounds.lower_a.clone(),
            new_lower_b,
            bounds.upper_a.clone(),
            new_upper_b,
            operand_shape.to_vec(),
            bounds.output_shape.clone(),
        )
    }

    /// The constant as the ONE column vector the BATCHED substitution can use.
    ///
    /// The batched bias contribution is `Σ_j A[..b, o, j] · c[..b, j]` computed
    /// with a SINGLE `c_flat` of length `in_dim` reused at EVERY batch position.
    /// That is correct only when the broadcast constant is invariant along every
    /// axis but the last. The old `flatten_constant_to_in_dim` instead tiled the
    /// constant to `in_dim` as `c[i % const_len]` with no shape check at all, so
    /// a `[C,1,1]` / `[1,C,1,1]` / CNN-bias `[C]` constant — which varies along a
    /// LEADING axis — was folded into the bias at the wrong positions, and the
    /// resulting certificate described a different function than
    /// `propagate_ibp` computes. Nothing made that error conservative.
    ///
    /// `operand_shape` comes from the pre-activation bounds at enum dispatch;
    /// `bounds.input_shape()` is instead this Add node's output carrier shape.
    /// Every failed check is a refusal. Eligible contexts take partial-CROWN or
    /// IBP fallback; a context with other pending paths aborts the CROWN lane.
    fn batched_constant_along_last_axis(
        &self,
        in_dim: usize,
        operand_shape: &[usize],
    ) -> Result<Array1<f32>> {
        let const_shape = self.constant.shape();
        if operand_shape.last().copied() != Some(in_dim) {
            return Err(NyError::UnsupportedOp(format!(
                "AddConstant batched CROWN: coefficient last axis {in_dim} does not match the \
                 operand shape {operand_shape:?}; refusing rather than guessing the layout"
            )));
        }
        let effective_shape = self.effective_broadcast_shape(operand_shape)?;
        let out_shape = broadcast_shapes(operand_shape, &effective_shape).ok_or_else(|| {
            NyError::ShapeMismatch {
                expected: operand_shape.to_vec(),
                got: const_shape.to_vec(),
            }
        })?;
        let operand_len = checked_shape_product(operand_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AddConstant batched CROWN: operand shape {operand_shape:?} overflows usize"
            ))
        })?;
        let out_len = checked_shape_product(&out_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AddConstant batched CROWN: output shape {out_shape:?} overflows usize"
            ))
        })?;
        if operand_len != out_len {
            // Element COUNTS, not shape equality — see the scalar path: a purely
            // rank-raising broadcast preserves row-major flat order, a genuine
            // expansion does not and is refused.
            return Err(NyError::UnsupportedOp(format!(
                "AddConstant batched CROWN: constant {const_shape:?} EXPANDS the operand \
                 {operand_shape:?} ({operand_len} positions) to {out_shape:?} ({out_len} \
                 positions); not supported"
            )));
        }
        // Right-align the constant into the OUTPUT rank; every axis but the last
        // must be an extent-1 (replicated) axis, else the constant differs
        // between batch positions and no single shared vector represents it.
        let rank = out_shape.len();
        let pad = rank - effective_shape.len();
        let leading_replicated =
            (0..rank.saturating_sub(1)).all(|axis| axis < pad || effective_shape[axis - pad] == 1);
        if !leading_replicated || effective_shape.last().copied() != Some(in_dim) {
            return Err(NyError::UnsupportedOp(format!(
                "AddConstant batched CROWN: constant {const_shape:?} varies along a non-last \
                 axis of the operand {operand_shape:?}; the batched substitution applies one \
                 shared column vector at every position — refusing so the caller degrades to IBP"
            )));
        }
        // Every leading axis is 1 and the last is `in_dim`, so the flat constant
        // IS the per-position column vector.
        self.constant
            .clone()
            .into_shape_with_order((self.constant.len(),))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: const_shape.to_vec(),
            })
    }
}
