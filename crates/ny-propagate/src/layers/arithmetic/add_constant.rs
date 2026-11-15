// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Add constant layer: y = x + c (element-wise).

mod crown;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{add_down_f32, add_up_f32, BoundedTensor};
use std::borrow::Cow;

use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::shape::broadcast_shapes;
use crate::LinearBounds;

/// The constant's EFFECTIVE broadcast shape against an operand of `input_shape`.
///
/// This is the SINGLE definition of "which axes does the constant vary along"
/// for `AddConstant`, and every path in this file reads it. It is not merely
/// tidiness: IBP and CROWN backward previously disagreed here, and that is a
/// wrong-FUNCTION defect rather than a loose bound (see the note on
/// `AddConstantLayer::backward_constant_columns`).
///
/// Two rules, in order:
///
/// 1. **Rank-1 CNN bias.** A rank-1 `[C]` constant against a rank-3 `[C,H,W]`
///    operand is read as `[C,1,1]` — a per-CHANNEL bias varying along the
///    SLOWEST axis. This is a DELIBERATE DEPARTURE from strict ONNX
///    right-alignment, which would left-pad `[C]` to `[1,1,C]` and vary it
///    along the FASTEST axis. It exists for ny-synthesized channel biases
///    (BatchNorm folding, the DSP graphs in `network/dsp/`), and the rest of
///    the tree already mirrors it — see `one_axis_phase.rs`
///    (`add_constant_tensor`), whose comment records the same hazard.
///    When `W == C`, both this convention and strict ONNX right-alignment are
///    shape-valid but compute different functions. With no source-provenance
///    bit to distinguish them, every path refuses that ambiguous case.
/// 2. Otherwise the constant keeps its authored shape and ordinary ONNX
///    (numpy) right-aligned broadcasting applies.
fn effective_constant_shape(const_shape: &[usize], input_shape: &[usize]) -> Result<Vec<usize>> {
    let channel_bias =
        const_shape.len() == 1 && input_shape.len() == 3 && const_shape[0] == input_shape[0];
    let right_aligned = const_shape.len() == 1 && input_shape.last() == const_shape.first();
    if channel_bias && right_aligned {
        return Err(NyError::UnsupportedOp(format!(
            "AddConstant rank-1 constant {const_shape:?} is ambiguous against input \
             {input_shape:?}: both the internal channel-bias convention and ONNX \
             right-aligned broadcasting apply; explicit layout provenance is required"
        )));
    }
    if channel_bias {
        Ok(vec![const_shape[0], 1, 1])
    } else {
        Ok(const_shape.to_vec())
    }
}

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

    /// Resolve the constant's internal broadcast semantics against an operand.
    ///
    /// This is crate-visible so certificate construction and structural
    /// preflight can use the same layout decision as IBP and CROWN. It carries
    /// no proof or admission authority: callers must still validate the
    /// resulting output shape and, where applicable, replay their certificate.
    pub(crate) fn effective_broadcast_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        effective_constant_shape(self.constant.shape(), input_shape)
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

        // Broadcast layout comes from `effective_constant_shape` — the shared
        // definition the CROWN backward paths below read as well. Handles the
        // CNN-bias case (rank-1 [C] on a rank-3 [C,H,W] operand is reshaped to
        // [C,1,1] so it varies along the CHANNEL axis, not the last one) and
        // otherwise leaves the authored shape to ordinary ONNX right-aligned
        // broadcasting. Behaviour here is unchanged; the point of routing it
        // through the helper is that CROWN can no longer read a DIFFERENT rule.
        let effective_shape = self.effective_broadcast_shape(input_shape)?;
        let const_for_broadcast = if effective_shape.as_slice() == const_shape {
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

        // Compute ONNX bidirectional broadcast output shape using the
        // (potentially reshaped) constant.
        let broadcast_const_shape = const_for_broadcast.shape();
        let output_shape =
            broadcast_shapes(input_shape, broadcast_const_shape).ok_or_else(|| {
                NyError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
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

    /// CROWN backward WITHOUT the operand shape.
    ///
    /// Only the layouts that can be PROVEN from the constant's own shape are
    /// accepted here; everything else refuses. See
    /// `AddConstantLayer::backward_constant_columns_without_shape`.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.crown_backward_with_operand_shape(bounds, None)
            .map(Cow::Owned)
    }

    /// CROWN backward WITH the operand's runtime shape.
    ///
    /// Every production backward dispatch already carries the node's
    /// pre-activation `BoundedTensor` (`beta_crown_ibp_fallback`,
    /// `graph_crown/backward_node_dispatch.rs`, `alpha_crown/backward.rs`,
    /// `graph_alpha/propagate_sequential/backward.rs`, `backward_step.rs`),
    /// so the true operand shape is available at the moment the constant's
    /// broadcast layout has to be resolved. Overriding this hook is the same
    /// design `MulConstantLayer` and `SubConstantLayer` already use; taking the
    /// linear-layer default (traits.rs) is what left this layer guessing.
    fn propagate_crown_backward(
        &self,
        bounds: &LinearBounds,
        pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        self.crown_backward_with_operand_shape(bounds, pre_activation.map(|pre| pre.shape()))
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
