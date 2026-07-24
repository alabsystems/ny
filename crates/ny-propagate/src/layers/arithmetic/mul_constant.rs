// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multiply by constant layer: y = x * c (element-wise).

use ndarray::{Array1, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Multiply by constant layer: y = x * c (element-wise).
///
/// Used in attention for scaling by 1/sqrt(head_dim).
#[derive(Debug, Clone)]
pub struct MulConstantLayer {
    /// The constant tensor to multiply by.
    pub(crate) constant: ArrayD<f32>,
    /// Original input shape from conversion, used to reconstruct broadcasted
    /// per-element scales for CROWN backward.
    input_shape: Option<Vec<usize>>,
}

impl MulConstantLayer {
    /// Validate and create a new multiply constant layer.
    pub fn try_new(constant: ArrayD<f32>) -> Result<Self> {
        validate_finite_array(&constant, "MulConstantLayer", "constant")?;
        Ok(Self {
            constant,
            input_shape: None,
        })
    }

    /// Create a new multiply constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self::try_new(constant).expect("invariant: MulConstantLayer::new requires finite constant")
    }

    /// Validate and create a multiply constant layer with the original input shape.
    pub fn try_with_input_shape(constant: ArrayD<f32>, input_shape: Vec<usize>) -> Result<Self> {
        let mut layer = Self::try_new(constant)?;
        layer.input_shape = Some(input_shape);
        Ok(layer)
    }

    /// Create a multiply constant layer with the original input shape.
    pub fn with_input_shape(constant: ArrayD<f32>, input_shape: Vec<usize>) -> Self {
        Self::try_with_input_shape(constant, input_shape)
            .expect("invariant: MulConstantLayer::with_input_shape requires finite constant")
    }

    /// Validate and create a scalar multiply layer.
    pub fn try_scalar(value: f32) -> Result<Self> {
        Self::try_new(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Create a scalar multiply layer.
    pub fn scalar(value: f32) -> Self {
        Self::try_scalar(value)
            .expect("invariant: MulConstantLayer::scalar requires finite constant")
    }

    /// Return the constant tensor.
    pub fn constant(&self) -> &ArrayD<f32> {
        &self.constant
    }

    /// Return the original input shape if conversion recorded it.
    pub fn input_shape(&self) -> Option<&[usize]> {
        self.input_shape.as_deref()
    }
}

impl BoundPropagation for MulConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x * c: bounds depend on sign of c
        // If c >= 0: y ∈ [l*c, u*c]
        // If c < 0: y ∈ [u*c, l*c]
        //
        // ONNX Mul broadcasts both inputs to a common shape. When the constant
        // is larger than the input (e.g., constant [4] * input [1] → output [4]),
        // we broadcast both to the output shape before element-wise computation.

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Compute broadcast output shape (ONNX semantics).
        let output_shape =
            crate::shape::broadcast_shapes(input_shape, const_shape).ok_or_else(|| {
                NyError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?;

        // Broadcast constant and input bounds to output shape.
        let c = self
            .constant
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: const_shape.to_vec(),
            })?;
        let lower_in = input
            .lower()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;
        let upper_in = input
            .upper()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;

        // Compute bounds element-wise, handling sign
        let mut out_lower = ArrayD::zeros(IxDyn(&output_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&output_shape));

        for (idx, &c_val) in c.indexed_iter() {
            let l = lower_in[idx.clone()];
            let u = upper_in[idx.clone()];

            if c_val == 0.0 {
                // Zero constant: x * 0 = 0 for all x.
                // Avoids Inf * 0.0 = NaN for upstream Inf bounds (#3273, #3034).
                out_lower[idx.clone()] = 0.0;
                out_upper[idx] = 0.0;
            } else if c_val > 0.0 {
                out_lower[idx.clone()] = l * c_val;
                out_upper[idx] = u * c_val;
            } else {
                out_lower[idx.clone()] = u * c_val;
                out_upper[idx] = l * c_val;
            }
        }

        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3273/#2549).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_runtime_shape(bounds, None)
    }

    /// CROWN backward with the actual input bounds available.
    ///
    /// Same exact affine substitution as `propagate_linear`, but the runtime
    /// input shape (taken from the pre-activation `BoundedTensor` that every
    /// backward dispatch site already supplies) lets the per-channel broadcast
    /// path work even when conversion could not record `input_shape` (e.g.
    /// shape inference failed through a Tile/Reshape chain, as in the talker
    /// attention RoPE cos/sin multiplies).
    fn propagate_crown_backward(
        &self,
        bounds: &LinearBounds,
        pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_runtime_shape(bounds, pre_activation.map(|t| t.shape()))
            .map(Cow::into_owned)
    }
}

impl MulConstantLayer {
    /// Shared CROWN backward body for `propagate_linear` (no runtime shape)
    /// and `propagate_crown_backward` (runtime shape from pre-activation).
    fn propagate_linear_with_runtime_shape<'a>(
        &self,
        bounds: &'a LinearBounds,
        runtime_input_shape: Option<&[usize]>,
    ) -> Result<Cow<'a, LinearBounds>> {
        // For y = x * c, the linear relationship scales:
        // If we have A @ y + b for bounds on y,
        // then for x where y = x * c:
        // A @ (x * c) + b = (A * c) @ x + b
        //
        // For scalar c, this is simple scaling of A matrices.
        // For broadcasted c, we need to scale each column of A by corresponding c value.

        let num_inputs = bounds.num_inputs();
        let scale =
            self.scale_for_linear_bounds_with_runtime_shape(num_inputs, runtime_input_shape)?;

        // Scale each column by corresponding c value (affine substitution).
        // No swap for c < 0: CROWN backward composes by substitution, not IBP.
        // Negative c just flips coefficient sign; downstream nonlinear relaxations
        // already branch on coefficient sign (see crown_elementwise_backward).
        // Reference: designs/2026-01-29-crown-affine-negative-scale.md
        let mut lower_a = bounds.lower_a().clone();
        let mut upper_a = bounds.upper_a().clone();

        // SOUND MulConstant coefficient error (#vnncomp-aw-soundness). The scaled
        // coefficient `A_new[i,j] = A[i,j]·c_j` is a single round-to-nearest f32
        // product, so it carries a relative rounding error of at most one unit
        // roundoff `u = 2^-24`: `|A_new - exact| ≤ |A_new|·u`. Without certifying it
        // the fresh error leaks uncertified into concretize (a bound a few ULP from
        // the threshold could read Verified when it is not). MulConstant now lives in
        // `propagates_coeff_err` (query.rs), so incoming coeff error arrives here and
        // MUST be propagated: the op scales column `j` by `c_j`, so an incoming error
        // `e_in[i,j]` becomes `|c_j|·e_in[i,j]`. Combine both, rounded OUTWARD.
        const F32_U: f32 = 1.0 / (1u32 << 24) as f32; // 2^-24 round-to-nearest unit roundoff
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let mut lower_err = ndarray::Array2::<f32>::zeros(lower_a.raw_dim());
        let mut upper_err = ndarray::Array2::<f32>::zeros(upper_a.raw_dim());
        for j in 0..num_inputs {
            let c_j = scale[j];
            let abs_c = c_j.abs();
            if c_j == 0.0 {
                // Zero column: set coefficients to zero directly. Both the fresh
                // product error (0) and the propagated error (|c_j|·e_in = 0) vanish.
                // Avoids Inf * 0.0 = NaN for upstream Inf coefficients (#3034).
                for i in 0..bounds.num_outputs() {
                    lower_a[[i, j]] = 0.0;
                    upper_a[[i, j]] = 0.0;
                }
            } else {
                for i in 0..bounds.num_outputs() {
                    lower_a[[i, j]] *= c_j;
                    upper_a[[i, j]] *= c_j;
                    let prop_l = in_lower_err.map_or(0.0, |e| abs_c * e[[i, j]]);
                    let prop_u = in_upper_err.map_or(0.0, |e| abs_c * e[[i, j]]);
                    lower_err[[i, j]] =
                        ny_tensor::next_up_f32(lower_a[[i, j]].abs() * F32_U + prop_l);
                    upper_err[[i, j]] =
                        ny_tensor::next_up_f32(upper_a[[i, j]].abs() * F32_U + prop_u);
                }
            }
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
            lower_a,
            bounds.lower_b().clone(),
            upper_a,
            bounds.upper_b().clone(),
            lower_err,
            upper_err,
        )?))
    }
}

impl MulConstantLayer {
    fn flatten_constant(&self) -> Result<Array1<f32>> {
        self.constant
            .clone()
            .into_shape_with_order(self.constant.len())
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })
    }

    fn broadcast_constant_to_shape(&self, target_shape: &[usize]) -> Result<ArrayD<f32>> {
        // Preserve ONNX multi-dimensional broadcast semantics instead of flat
        // tiling, matching auto_LiRPA's constant backward broadcast-reduction.
        self.constant
            .broadcast(IxDyn(target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.to_vec(),
                got: self.constant.shape().to_vec(),
            })
            .map(|view| view.into_owned())
    }

    fn recorded_input_shape(&self, num_inputs: usize) -> Result<&[usize]> {
        let input_shape = self.input_shape.as_deref().ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "MulConstant CROWN backward: per-channel broadcast \
                 (constant {} elements, input {} elements) requires input_shape",
                self.constant.len(),
                num_inputs
            ))
        })?;
        let expected_elems = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulConstant input_shape overflows usize: {input_shape:?}"
            ))
        })?;
        if expected_elems != num_inputs {
            return Err(NyError::InvalidSpec(format!(
                "MulConstant input_shape {:?} has {} elements, but incoming bounds expect {}",
                input_shape, expected_elems, num_inputs
            )));
        }
        Ok(input_shape)
    }

    /// Per-column scale for the CROWN backward substitution, optionally
    /// recovering the broadcast layout from the runtime input shape.
    ///
    /// SOUNDNESS of the runtime-shape recovery: `runtime_input_shape` is the
    /// shape of the node's actual pre-activation `BoundedTensor` — the true
    /// shape of `x` at propagation time, not a guess. When (a) the constant
    /// broadcasts INTO that shape (every right-aligned constant dim is 1 or
    /// equal, ndim(c) <= ndim(x)), the ONNX output shape broadcast(x, c)
    /// equals the input shape exactly, so `y_flat[j] = x_flat[j] *
    /// scale_flat[j]` element-for-element; and (b) `prod(shape) ==
    /// num_inputs` ties that layout to the incoming coefficient columns.
    /// Under (a)+(b) the column scaling below is the same EXACT affine
    /// substitution as the recorded-`input_shape` path — no relaxation, no
    /// new error term beyond the per-product ulp already certified by the
    /// caller. Anything else (broadcast expansion of `x`, layout mismatch)
    /// falls through to the conservative `UnsupportedOp` → IBP fallback.
    fn scale_for_linear_bounds_with_runtime_shape(
        &self,
        num_inputs: usize,
        runtime_input_shape: Option<&[usize]>,
    ) -> Result<Array1<f32>> {
        let flat = self.flatten_constant()?;
        if flat.len() == 1 {
            return Ok(Array1::from_elem(num_inputs, flat[0]));
        }
        if self.input_shape.is_some() {
            let input_shape = self.recorded_input_shape(num_inputs)?;
            return self
                .broadcast_constant_to_shape(input_shape)?
                .into_shape_with_order(num_inputs)
                .map_err(|_| NyError::ShapeMismatch {
                    expected: vec![num_inputs],
                    got: input_shape.to_vec(),
                });
        }
        if flat.len() == num_inputs {
            return Ok(flat);
        }
        if flat.len() > num_inputs {
            return Err(NyError::UnsupportedOp(format!(
                "MulConstant CROWN backward: broadcast expansion \
                 (constant {} elements > input {} elements) not supported",
                flat.len(),
                num_inputs
            )));
        }
        // Conversion did not record input_shape (shape inference can fail
        // through Tile/Reshape chains); recover the broadcast layout from the
        // runtime pre-activation shape when it is consistent (see soundness
        // note above).
        if let Some(shape) = runtime_input_shape {
            if checked_shape_product(shape) == Some(num_inputs) {
                if let Ok(scale) = self.broadcast_constant_to_shape(shape) {
                    return scale.into_shape_with_order(num_inputs).map_err(|_| {
                        NyError::ShapeMismatch {
                            expected: vec![num_inputs],
                            got: shape.to_vec(),
                        }
                    });
                }
            }
        }
        Err(NyError::UnsupportedOp(format!(
            "MulConstant CROWN backward: per-channel broadcast \
             (constant {} elements, input {} elements) requires input_shape",
            flat.len(),
            num_inputs
        )))
    }

    fn ensure_batched_input_shape_matches_output(&self, output_shape: &[usize]) -> Result<()> {
        if let Some(input_shape) = self.input_shape.as_deref() {
            if input_shape != output_shape {
                return Err(NyError::UnsupportedOp(format!(
                    "MulConstant batched CROWN: broadcast expansion/reduction from input shape {:?} to output shape {:?} is not supported",
                    input_shape, output_shape
                )));
            }
        }
        Ok(())
    }

    /// Batched CROWN backward propagation through MulConstant.
    ///
    /// For y = x * c, CROWN backward substitution scales coefficients:
    /// If we have A @ y + b for bounds on y, where y = x * c:
    /// A @ (x * c) + b = (A * c) @ x + b (scaling coefficients)
    ///
    /// No swap for c < 0: CROWN composes by substitution (not IBP).
    /// Reference: designs/2026-01-29-crown-affine-negative-scale.md
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // 2^-24 round-to-nearest f32 unit roundoff for the fresh per-coeff product
        // error (#vnncomp-aw-soundness). See `propagate_linear` for the derivation.
        const F32_U: f32 = 1.0 / (1u32 << 24) as f32;
        let flat = self.flatten_constant()?;
        if flat.len() == 1 {
            let c_val = flat[0];
            debug!("MulConstant batched CROWN: scalar c = {}", c_val);

            // Short-circuit when c == 0: all coefficients become zero, bias unchanged.
            // Avoids Inf * 0.0 = NaN for upstream Inf coefficients (#3034).
            if c_val == 0.0 {
                return BatchedLinearBounds::new_or_conservative(
                    ArrayD::zeros(bounds.lower_a.raw_dim()),
                    bounds.lower_b.clone(),
                    ArrayD::zeros(bounds.upper_a.raw_dim()),
                    bounds.upper_b.clone(),
                    bounds.input_shape.clone(),
                    bounds.output_shape.clone(),
                );
            }

            // Scale the coefficient matrices by c (affine substitution).
            // No swap for c < 0: CROWN backward composes by substitution, not IBP.
            // Negative c just flips coefficient sign; downstream nonlinear relaxations
            // already branch on coefficient sign.
            let new_lower_a = bounds.lower_a.mapv(|v| v * c_val);
            let new_upper_a = bounds.upper_a.mapv(|v| v * c_val);
            // SOUND fresh per-coeff f32 product error (#vnncomp-aw-soundness):
            // `|A·c - exact| ≤ |A·c|·u`, u = 2^-24. In the BATCHED pipeline MulConstant
            // is a coeff-err CARRIER (dispatch.rs): the dispatcher carries any INCOMING
            // err via a separate carrier re-run and ADDS it to this fresh err, so this
            // run sees err-free bounds and emits the fresh err ONLY (no prop term).
            let lower_err = new_lower_a.mapv(|v| ny_tensor::next_up_f32(v.abs() * F32_U));
            let upper_err = new_upper_a.mapv(|v| ny_tensor::next_up_f32(v.abs() * F32_U));
            let mut out = BatchedLinearBounds::new_or_conservative(
                new_lower_a,
                bounds.lower_b.clone(),
                new_upper_a,
                bounds.upper_b.clone(),
                bounds.input_shape.clone(),
                bounds.output_shape.clone(),
            )?;
            out.set_coeff_err(lower_err, upper_err);
            return Ok(out);
        }

        let input_elems = checked_shape_product(bounds.input_shape()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulConstant batched CROWN: input shape overflows usize: {:?}",
                bounds.input_shape()
            ))
        })?;
        if flat.len() > input_elems {
            return Err(NyError::UnsupportedOp(format!(
                "MulConstant batched CROWN: broadcast expansion \
                 (constant {} elements > input {} elements) not supported",
                flat.len(),
                input_elems
            )));
        }

        self.ensure_batched_input_shape_matches_output(bounds.input_shape())?;
        let scale = self.broadcast_constant_to_shape(bounds.input_shape())?;
        let scale_shape = scale.shape().to_vec();
        let scale_with_output_axis = scale
            .view()
            .insert_axis(Axis(scale.ndim().saturating_sub(1)));
        let expanded_scale = scale_with_output_axis
            .broadcast(bounds.lower_a.raw_dim())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MulConstant batched CROWN: scale shape {:?} cannot broadcast to coefficient shape {:?}",
                    scale_shape,
                    bounds.lower_a.shape()
                ))
            })?;

        let mut lower_a = bounds.lower_a.clone();
        ndarray::Zip::from(&mut lower_a)
            .and(expanded_scale)
            .for_each(|coeff, &s| {
                *coeff = if s == 0.0 { 0.0 } else { *coeff * s };
            });

        let scale_with_output_axis = scale
            .view()
            .insert_axis(Axis(scale.ndim().saturating_sub(1)));
        let expanded_scale = scale_with_output_axis
            .broadcast(bounds.upper_a.raw_dim())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MulConstant batched CROWN: scale shape {:?} cannot broadcast to coefficient shape {:?}",
                    scale_shape,
                    bounds.upper_a.shape()
                ))
            })?;
        let mut upper_a = bounds.upper_a.clone();
        ndarray::Zip::from(&mut upper_a)
            .and(expanded_scale)
            .for_each(|coeff, &s| {
                *coeff = if s == 0.0 { 0.0 } else { *coeff * s };
            });

        // SOUND fresh per-coeff f32 product error (#vnncomp-aw-soundness):
        // `|A·s - exact| ≤ |A·s|·u`, u = 2^-24. Zero-scale columns are set to
        // exactly 0 above, so their err is 0 too. In the BATCHED pipeline MulConstant
        // is a coeff-err CARRIER (dispatch.rs): incoming err is carried by a separate
        // carrier re-run and ADDED to this fresh err, so this run sees err-free bounds
        // and emits the fresh err ONLY (no prop term).
        let lower_err = lower_a.mapv(|v| ny_tensor::next_up_f32(v.abs() * F32_U));
        let upper_err = upper_a.mapv(|v| ny_tensor::next_up_f32(v.abs() * F32_U));
        let mut out = BatchedLinearBounds::new_or_conservative(
            lower_a,
            bounds.lower_b.clone(),
            upper_a,
            bounds.upper_b.clone(),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        out.set_coeff_err(lower_err, upper_err);
        Ok(out)
    }
}
