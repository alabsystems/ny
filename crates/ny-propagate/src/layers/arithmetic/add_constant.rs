// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Add constant layer: y = x + c (element-wise).

use ndarray::{s, Array1, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{add_down_f32, add_up_f32, next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::common::dot_bias_f64;
use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Add constant layer: adds a constant tensor to input (e.g., bias addition).
///
/// This is used for ONNX Add operations where one input is a constant (weight/bias).
/// For y = x + c where c is constant and x is bounded:
/// y ∈ [l + c, u + c]
#[derive(Debug, Clone)]
pub struct AddConstantLayer {
    /// The constant tensor to add.
    pub(crate) constant: ArrayD<f32>,
}

impl AddConstantLayer {
    /// Validate and create a new add constant layer.
    pub fn try_new(constant: ArrayD<f32>) -> Result<Self> {
        validate_finite_array(&constant, "AddConstantLayer", "constant")?;
        Ok(Self { constant })
    }

    /// Create a new add constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self::try_new(constant).expect("invariant: AddConstantLayer::new requires finite constant")
    }

    /// Return the constant tensor.
    pub fn constant(&self) -> &ArrayD<f32> {
        &self.constant
    }
}

impl BoundPropagation for AddConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x + c: y ∈ [l + c, u + c]
        //
        // ONNX Add broadcasts both inputs to a common shape. When the constant
        // has higher rank than the input (e.g., constant [1,256,1] + input [256,80]
        // → output [1,256,80]), we broadcast both to the output shape.

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // If shapes match exactly, simple addition
        if input_shape == const_shape {
            // DIRECTED: a plain f32 `+` rounds to nearest and can move an
            // endpoint INWARD by up to half an ULP. See `AddLayer`.
            let out_lower = ndarray::Zip::from(input.lower())
                .and(&self.constant)
                .map_collect(|&v, &c| add_down_f32(v, c));
            let out_upper = ndarray::Zip::from(input.upper())
                .and(&self.constant)
                .map_collect(|&v, &c| add_up_f32(v, c));
            // `new_allow_infinite`, not the strict `new`: an upstream node that
            // failed closed to an OpaqueSkip legitimately hands this layer
            // `[-inf, +inf]` (`OpaqueSkipLayer::unbounded_like` builds exactly
            // that), and ±inf + finite constant is still a sound enclosure. The
            // strict constructor rejected it as NumericalInstability, which is
            // NOT in `is_degradable_error`, so the WHOLE IBP pass aborted at an
            // already-tainted node instead of degrading — measured on lsnc_relu
            // via `ny verify` ("BoundedTensor::new: lower bounds contain NaN or
            // Inf"). NaN and inverted bounds are still rejected, so the NaN
            // firewall is unchanged.
            return BoundedTensor::new_allow_infinite(out_lower, out_upper);
        }

        // Handle CNN bias case: 1D bias [channels] added to 3D input [channels, height, width]
        // Need to reshape [C] to [C, 1, 1] for broadcasting along channel dimension.
        // This must happen before computing the broadcast output shape, since standard
        // ONNX broadcasting would left-pad [C] to [1, 1, C] instead of [C, 1, 1].
        let const_for_broadcast =
            if const_shape.len() == 1 && input_shape.len() == 3 && const_shape[0] == input_shape[0]
            {
                // CNN bias: reshape [C] to [C, 1, 1]
                self.constant
                    .clone()
                    .into_shape_with_order(IxDyn(&[const_shape[0], 1, 1]))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: vec![const_shape[0], 1, 1],
                        got: const_shape.to_vec(),
                    })?
            } else {
                self.constant.clone()
            };

        // Compute ONNX bidirectional broadcast output shape using the
        // (potentially reshaped) constant.
        let broadcast_const_shape = const_for_broadcast.shape();
        let output_shape = crate::shape::broadcast_shapes(input_shape, broadcast_const_shape)
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: const_shape.to_vec(),
            })?;

        // Broadcast both constant and input bounds to the output shape.
        let broadcast_const = const_for_broadcast
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

        let out_lower = ndarray::Zip::from(&lower_in)
            .and(&broadcast_const)
            .map_collect(|&v, &c| add_down_f32(v, c));
        let out_upper = ndarray::Zip::from(&upper_in)
            .and(&broadcast_const)
            .map_collect(|&v, &c| add_up_f32(v, c));

        // See the exact-shape branch above: OpaqueSkip-tainted inputs are
        // legitimately infinite, and rejecting them here aborts the whole IBP
        // pass. NaN and inverted bounds are still rejected.
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x + c, the linear relationship is preserved:
        // If we have A @ y + b, then A @ (x + c) + b = A @ x + (A @ c + b)
        // Since c is constant, A @ c becomes a constant vector added to bias.

        // For simplicity, we add the contribution of c to the bias terms.
        // A @ c where A has shape (num_outputs, num_inputs) and c has shape (num_inputs,)
        // gives a vector of shape (num_outputs,)

        let num_inputs = bounds.num_inputs();
        let const_len = self.constant.len();

        // Guard: empty constant causes `% 0` panic in broadcast check below. (#2818)
        if const_len == 0 {
            return Err(NyError::InvalidSpec(
                "AddConstant CROWN backward: constant tensor is empty".to_string(),
            ));
        }

        // Handle broadcasting: if constant is smaller than num_inputs, it was broadcast
        // in the forward pass. We need to tile/broadcast it to match num_inputs.
        let c_flat = if const_len == num_inputs {
            // Exact match - no broadcasting needed
            self.constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })?
        } else if num_inputs.is_multiple_of(const_len) {
            // Constant was broadcast (tiled) along some axis
            // Tile the constant to match num_inputs
            let repeat_count = num_inputs / const_len;
            let c_1d = self
                .constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })?;
            // Tile by repeating the constant
            let mut tiled = Array1::<f32>::zeros(num_inputs);
            for i in 0..repeat_count {
                let start = i * const_len;
                tiled.slice_mut(s![start..start + const_len]).assign(&c_1d);
            }
            tiled
        } else {
            // Incompatible sizes - this shouldn't happen in well-formed networks
            return Err(NyError::ShapeMismatch {
                expected: vec![num_inputs],
                got: vec![const_len],
            });
        };

        // Compute A @ c with f64 accumulation + directed rounding (#3157, #1863).
        let (lower_c_f64, upper_c_f64) = dot_bias_f64(bounds.lower_a(), bounds.upper_a(), &c_flat);
        let lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_c_f64;
        let ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_c_f64;
        let new_lower_b = lb_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = ub_f64.mapv(|x| next_up_f32(x as f32));

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            bounds.lower_a().clone(),
            new_lower_b,
            bounds.upper_a().clone(),
            new_upper_b,
        )?))
    }
}

impl AddConstantLayer {
    /// Batched CROWN backward propagation through AddConstant.
    ///
    /// For y = x + c, the linear relationship is:
    /// A @ y + b = A @ (x + c) + b = A @ x + (A @ c + b)
    /// So coefficient matrices stay the same, bias gets A @ c added.
    ///
    /// Supports both scalar and vector constants. For scalar c, uses the fast
    /// path c * sum(A, axis=-1). For vector c, computes the full A @ c product.
    ///
    /// Reference: auto_LiRPA Bound.get_bias (einsum over A and bias):
    /// alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/base.py:468-480
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
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

        // Compute bias contribution A @ c in f64 for soundness (#2423, #3157).
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
            // Flatten constant to 1D and handle broadcasting (tile if needed).
            let c_flat = self.flatten_constant_to_in_dim(in_dim)?;
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
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }

    /// Flatten the constant tensor to a 1D array of size `in_dim`, handling
    /// broadcasting (tiling) when the constant is smaller.
    fn flatten_constant_to_in_dim(&self, in_dim: usize) -> Result<Array1<f32>> {
        let const_len = self.constant.len();
        if const_len == in_dim {
            self.constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })
        } else if const_len > 0 && in_dim.is_multiple_of(const_len) {
            // Constant was broadcast (tiled) along some axis.
            let repeat_count = in_dim / const_len;
            let c_1d = self
                .constant
                .clone()
                .into_shape_with_order((const_len,))
                .map_err(|_| NyError::ShapeMismatch {
                    expected: vec![const_len],
                    got: self.constant.shape().to_vec(),
                })?;
            let mut tiled = Array1::<f32>::zeros(in_dim);
            for i in 0..repeat_count {
                let start = i * const_len;
                tiled.slice_mut(s![start..start + const_len]).assign(&c_1d);
            }
            Ok(tiled)
        } else {
            Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![const_len],
            })
        }
    }
}

#[cfg(test)]
mod opaque_skip_taint_tests {
    use super::AddConstantLayer;
    use crate::layers::common::BoundPropagation;
    use ndarray::array;
    use ny_tensor::BoundedTensor;

    /// An upstream node that failed closed to an OpaqueSkip hands its consumers
    /// `[-inf, +inf]` (`OpaqueSkipLayer::unbounded_like`). Adding a finite
    /// constant to that is still a sound enclosure, so `AddConstant` must
    /// PROPAGATE it — not abort. The strict constructor rejected it as
    /// `NumericalInstability`, which is not in `is_degradable_error`, so one
    /// tainted element killed the entire IBP pass (measured on lsnc_relu via
    /// `ny verify`).
    #[test]
    fn infinite_input_from_an_opaque_skip_propagates_instead_of_aborting() {
        let layer = AddConstantLayer::new(array![1.5_f32, -2.0].into_dyn());
        let input = BoundedTensor::new_allow_infinite(
            array![f32::NEG_INFINITY, -1.0].into_dyn(),
            array![f32::INFINITY, 1.0].into_dyn(),
        )
        .expect("[-inf, +inf] is a valid conservative enclosure");

        let out = layer
            .propagate_ibp(&input)
            .expect("a tainted element must widen, not abort the pass");

        // The tainted element stays unbounded; the finite element stays exact.
        assert_eq!(out.lower()[[0]], f32::NEG_INFINITY);
        assert_eq!(out.upper()[[0]], f32::INFINITY);
        assert_eq!(out.lower()[[1]], -1.0 + -2.0);
        assert_eq!(out.upper()[[1]], 1.0 + -2.0);
    }

    /// Same guarantee on the broadcast path (constant shape != input shape).
    #[test]
    fn infinite_input_propagates_through_the_broadcast_path_too() {
        // 1D bias [C] against 3D input [C, H, W]: the CNN-bias reshape branch.
        let layer = AddConstantLayer::new(array![1.0_f32, 2.0].into_dyn());
        let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 1]), f32::NEG_INFINITY);
        let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 1]), f32::INFINITY);
        let input = BoundedTensor::new_allow_infinite(lower, upper)
            .expect("[-inf, +inf] is a valid conservative enclosure");

        let out = layer
            .propagate_ibp(&input)
            .expect("broadcast path must also widen instead of aborting");
        assert!(out.lower().iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(out.upper().iter().all(|&v| v == f32::INFINITY));
    }

    /// The relaxation must NOT relax the NaN firewall: a NaN input is still a
    /// hard error, exactly as before.
    #[test]
    fn nan_input_is_still_rejected() {
        let layer = AddConstantLayer::new(array![1.0_f32].into_dyn());
        let input =
            BoundedTensor::new_unchecked(array![f32::NAN].into_dyn(), array![1.0_f32].into_dyn())
                .expect("shape-only constructor should accept NaN");

        assert!(
            layer.propagate_ibp(&input).is_err(),
            "NaN must not be absorbed into AddConstant IBP bounds"
        );
    }
}
