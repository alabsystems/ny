// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BoundPropagation trait implementation and batched CROWN dispatch for the Layer enum.

use ndarray::ArrayD;
use ny_core::{checked_dim_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use super::{for_each_elementwise_activation, Layer};
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

impl Layer {
    /// Forward propagation for a concrete (point) input.
    ///
    /// Runs point IBP and only publishes the result when every lower/upper
    /// endpoint is bitwise identical and the operation is known to copy,
    /// select, reshape, or produce exactly representable values.  A collapsed
    /// point interval alone is insufficient: ordinary f32 arithmetic can round
    /// an exact-real result to one shared endpoint.
    ///
    /// Used by graph-builder constant pre-evaluation. Nontrivial frozen
    /// arithmetic remains in the runtime graph unless it has a dedicated exact
    /// evaluator.
    pub fn propagate_concrete(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        let concrete = BoundedTensor::concrete(input)?;
        let output = self.propagate_ibp(&concrete)?;
        // An IBP image of a point is not necessarily a point. Many sound
        // layers deliberately round their endpoints outward, so returning
        // `lower` would publish one side of an enclosure as an exact constant.
        // Only materialize when the layer itself proves point preservation by
        // returning bitwise identical endpoints. Signed zero is
        // conservatively treated as distinct as well.
        let is_point = output
            .lower()
            .iter()
            .zip(output.upper().iter())
            .all(|(&lower, &upper)| lower.to_bits() == upper.to_bits());
        // Keep this an explicit allowlist.  Linear, convolution, arithmetic,
        // normalization, reduction-by-sum, and transcendental operations need
        // a separately certified exact evaluator even if their current point
        // IBP implementation happens to collapse.  In particular, f32
        // 0.1*0.1 collapses in Conv1d IBP but is not the exact-real product.
        let exact_operation = matches!(
            self,
            Layer::ReLU(_)
                | Layer::Clip(_)
                | Layer::ThresholdedRelu(_)
                | Layer::Abs(_)
                | Layer::MaxPool2d(_)
                | Layer::ReduceMax(_)
                | Layer::ReduceMin(_)
                | Layer::Transpose(_)
                | Layer::Reshape(_)
                | Layer::Flatten(_)
                | Layer::Pad(_)
                | Layer::Tile(_)
                | Layer::Gather(_)
                | Layer::Slice(_)
                | Layer::Squeeze(_)
                | Layer::Unsqueeze(_)
                | Layer::Resize(_)
                | Layer::Compare(_)
                | Layer::Floor(_)
                | Layer::Ceil(_)
                | Layer::Round(_)
                | Layer::Trunc(_)
                | Layer::Sign(_)
        );
        if is_point && exact_operation {
            Ok(output.lower().clone())
        } else {
            Err(NyError::UnsupportedOp(format!(
                "{} concrete evaluation lacks a certified exact singleton result; refusing to materialize a rounded enclosure as an exact constant",
                self.layer_type(),
            )))
        }
    }

    /// Propagate batched CROWN bounds through this layer.
    ///
    /// Layers without batched CROWN support return `UnsupportedOp`.
    pub fn propagate_crown_backward_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: Option<&BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        // SOUNDNESS / TIGHTNESS for the certified coefficient error
        // (#vnncomp-aw-soundness). The error must be CARRIED all the way to
        // concretize for the affine subgraph (so the penalty is applied against the
        // NETWORK INPUT box, matching the scalar path). Layers split into three
        // disjoint classes:
        //   - DIRECT err-propagators (element-wise activations + Linear): their batched
        //     backward propagates the INCOMING error itself — the activation relaxation
        //     scales each `err_in` by `|lower_slope|+|upper_slope|`, and the Linear
        //     backward composes `Σ_k err_in·|W|` against the ABSOLUTE weight (`w_abs`),
        //     exactly like the scalar `propagate_linear` path (mirroring the scalar
        //     dispatcher's `propagates_coeff_err()` gate). Pass the err-carrying bounds
        //     straight through (no carrier, no discharge) so the batched (β-CROWN/BaB)
        //     verdict path carries the SAME certified margin as scalar. NOTE: the
        //     carrier re-run (below) instead composes `|err_in @ W|` against the SIGNED
        //     weight, which CANCELS and UNDER-estimates the penalty (too tight) for an
        //     `A·W` contraction; direct propagation avoids that.
        //   - error CARRIERS (the remaining EXACT-linear ops: Add/Sub/Concat/Slice/…,
        //     Conv1d, Conv2d, ConvTranspose): their backward is a fixed permute/select/
        //     scale (no sign cancellation in the carrier transform), so the error is
        //     carried by re-running the SAME backward on the non-negative error CARRIER
        //     and ADDing the result. (Conv stays here: its batched backward does not
        //     itself propagate the incoming error field — the carrier re-run supplies
        //     the penalty.)
        //   - NONLINEAR ops that CANNOT carry it (softmax, norms, …): discharge any
        //     INCOMING error precisely into the bias (or to conservative rows) before
        //     the op (sound; those are relaxation surfaces anyway).
        let batched_propagates_err_directly = self.is_elementwise_activation()
            || matches!(self, Layer::Linear(_) | Layer::BatchNorm(_));
        // BatchNorm (#cgan-conv-err-compose): its batched backward is an EXACT
        // per-column scaling that now PROPAGATES incoming coeff error as `e·|scale|`
        // (mirroring crown_scalar.rs — see crown_batched.rs). It must therefore route
        // to the propagate backend and NO LONGER take the output-box discharge below;
        // it stays OUT of `batched_carries_err` (not in the carrier match), so the
        // carrier re-run — which would OVERWRITE the fresh per-coeff ULP err — never
        // fires. The propagated err then SURVIVES on the result (BatchNorm is not an
        // elementwise activation, so `fold_result_err` leaves it alone) and is
        // discharged only at the terminal network-input concretize — the whole point
        // of the scalar fix, now holding in the batched (β-CROWN/BaB) path too.
        // NOTE: the BATCHED carrier set is intentionally LARGER than the scalar
        // `is_exact_linear_coeff_err_carrier()` set. Conv1d/Conv2d/ConvTranspose*,
        // MulConstant/DivConstant, and Tile were MOVED to `propagates_coeff_err()`
        // in query.rs so the SCALAR dispatcher routes incoming err THROUGH their
        // backends' `prop` term (the scalar carrier path REPLACES fresh err, which
        // would be unsound). The BATCHED carrier path instead ADDS the carried error
        // to any fresh err the backend emitted (see `attach_err_from_carried`), so
        // here they stay CARRIERS: the carrier re-run carries incoming err precisely
        // (preserving batched↔scalar parity), and the fresh `γ_n·S` / per-coeff ULP
        // error their batched backends emit is ADDED on top — sound and tight.
        let batched_carries_err = !batched_propagates_err_directly
            && (self.is_exact_linear_coeff_err_carrier()
                || matches!(
                    self,
                    Layer::Conv1d(_)
                        | Layer::Conv2d(_)
                        | Layer::ConvTranspose1d(_)
                        | Layer::ConvTranspose2d(_)
                        | Layer::MulConstant(_)
                        | Layer::DivConstant(_)
                        | Layer::Tile(_)
                        // Resize (nearest) duplicate-fan-in scatter-add: same as Tile —
                        // its batched backend emits the fresh γ_k·S, the carrier adds any
                        // incoming err on top. (#vnncomp-aw-soundness)
                        | Layer::Resize(_)
                        // Pad (Reflect duplicate scatter-add + Constant directed bias fold):
                        // batched backend emits fresh err, carrier adds incoming.
                        | Layer::Pad(_)
                ));
        if bounds.has_coeff_err() && batched_carries_err {
            if let Some(carrier) = bounds.coeff_err_carrier() {
                let mut plain = bounds.clone();
                plain.lower_a_err = None;
                plain.upper_a_err = None;
                let mut real =
                    self.propagate_crown_backward_batched(&plain, pre_activation, engine)?;
                let carried =
                    self.propagate_crown_backward_batched(&carrier, pre_activation, engine)?;
                real.attach_err_from_carried(&carried);
                return Ok(real);
            }
        }
        let discharged;
        let bounds: &BatchedLinearBounds =
            if bounds.has_coeff_err() && !batched_carries_err && !batched_propagates_err_directly {
                // Nonlinear op cannot carry the error through its relaxation here. Fold
                // the incoming error PRECISELY into the bias instead of degrading the
                // whole row: the error coefficients multiply THIS op's OUTPUT, whose box
                // is the IBP-forward of `pre_activation`. `penalty = Σ max(|y|)·err` is a
                // sound, tight discharge that the op then carries as a constant bias.
                let mut tmp = bounds.clone();
                let mut folded = false;
                if let Some(pre) = pre_activation {
                    if let Ok(out_box) = self.propagate_ibp(pre) {
                        let n = *tmp.lower_a.shape().last().unwrap_or(&0);
                        let flat = out_box.flatten();
                        if flat.lower().len() == n {
                            let l: Vec<f32> = flat.lower().iter().copied().collect();
                            let u: Vec<f32> = flat.upper().iter().copied().collect();
                            tmp.fold_coeff_err_into_bias(&l, &u);
                            folded = true;
                        }
                    }
                }
                if !folded {
                    tmp.discharge_coeff_err_to_conservative();
                }
                discharged = tmp;
                &discharged
            } else {
                bounds
            };
        // Carrier/Linear/Conv with NO incoming error: their fresh error (Linear/Conv)
        // must SURVIVE on the result to be carried by the next op (or consumed at
        // concretize). Nonlinear ops produce no error.
        //
        // EXCEPTION (#cgan-conv-err-compose): elementwise ACTIVATIONS eagerly
        // discharge the (fresh + propagated) error over their pre-activation cut —
        // the same policy as the scalar graph/sequential backward drivers (see
        // LinearBounds::fold_coeff_err_over_box_eager for the enclosure and
        // tightness argument; keeps batched↔scalar parity within test tolerance).
        // Rows with a non-finite penalty keep carrying.
        let fold_result_err = |mut result: BatchedLinearBounds| -> BatchedLinearBounds {
            if self.is_elementwise_activation() && result.has_coeff_err() {
                if let Some(pre) = pre_activation {
                    let flat = pre.flatten();
                    if let (Some(l), Some(u)) = (flat.lower().as_slice(), flat.upper().as_slice()) {
                        result.fold_coeff_err_over_box_eager(l, u);
                    }
                }
            }
            result
        };
        let require_pre_activation = || {
            pre_activation.ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "{} batched CROWN backward requires pre-activation bounds",
                    self.layer_type()
                ))
            })
        };

        let reshape_pre_activation = |pre_activation: &BoundedTensor| -> Result<BoundedTensor> {
            let a_shape = bounds.lower_a.shape();
            if a_shape.len() < 2 {
                return Err(NyError::InvalidSpec(
                    "BatchedLinearBounds must have at least 2 dimensions".to_string(),
                ));
            }
            let in_dim = a_shape[a_shape.len() - 1];
            let mut target_shape = a_shape[..a_shape.len() - 2].to_vec();
            target_shape.push(in_dim);
            if pre_activation.shape() == target_shape.as_slice() {
                Ok(pre_activation.clone())
            } else {
                pre_activation.reshape(&target_shape)
            }
        };

        let validate_flatten_like = |pre_activation: &BoundedTensor, op: &str| -> Result<()> {
            if bounds.input_shape.is_empty() {
                return Ok(());
            }
            let feature_dim = *bounds
                .input_shape
                .last()
                .ok_or_else(|| NyError::InvalidSpec("Missing input shape".to_string()))?;
            let batch_dims = &bounds.input_shape[..bounds.input_shape.len() - 1];
            let pre_shape = pre_activation.shape();
            if pre_shape.len() < batch_dims.len() {
                return Err(NyError::InvalidSpec(format!(
                    "Batched CROWN {} requires at least {} dims, got {:?}",
                    op,
                    batch_dims.len(),
                    pre_shape
                )));
            }
            if !batch_dims.is_empty() && pre_shape[..batch_dims.len()] != *batch_dims {
                return Err(NyError::InvalidSpec(format!(
                    "Batched CROWN {} batch dims mismatch: expected {:?}, got {:?}",
                    op,
                    batch_dims,
                    &pre_shape[..batch_dims.len()]
                )));
            }
            let remaining: usize = checked_dim_product(
                &pre_shape[batch_dims.len()..],
                &format!("Batched CROWN {op} feature dimensions"),
            )?;
            if remaining != feature_dim {
                return Err(NyError::InvalidSpec(format!(
                    "Batched CROWN {} feature dim mismatch: expected {}, got {} (shape {:?})",
                    op, feature_dim, remaining, pre_shape
                )));
            }
            Ok(())
        };

        // Elementwise activations: macro-generated match arms (#1708, #1753).
        // All share identical calling convention: reshape pre-activation, then delegate.
        macro_rules! elementwise_batched_arms {
            ($($Variant:ident),*) => {
                match self {
                    // Elementwise activations (generated from for_each_elementwise_activation)
                    $(Layer::$Variant(inner) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        inner.propagate_linear_batched_with_bounds(bounds, &pre)
                    })*
                    // Softmax: extra soundness_mode parameter.
                    // When bounds are flat (2D) but the pre-activation still carries
                    // grouped rows, pass the original tensor so Softmax can recover
                    // per-row group structure from A-matrix dimensions.
                    Layer::Softmax(s) => {
                        let raw_pre = require_pre_activation()?;
                        let a_shape = bounds.lower_a.shape();
                        let pre_shape = raw_pre.shape();
                        let pre_softmax_size = *pre_shape.last().unwrap_or(&0);
                        let a_in_dim = *a_shape.last().unwrap_or(&0);
                        let is_flat_with_groups = a_shape.len() == 2
                            && pre_shape.len() >= 2
                            && pre_softmax_size > 0
                            && a_in_dim != pre_softmax_size
                            && a_in_dim % pre_softmax_size == 0;
                        let pre = if is_flat_with_groups {
                            raw_pre.clone()
                        } else {
                            reshape_pre_activation(raw_pre)?
                        };
                        s.propagate_linear_batched_with_bounds(bounds, &pre, s.soundness_mode())
                    }
                    Layer::CausalSoftmax(s) => {
                        let pre = require_pre_activation()?;
                        s.propagate_linear_batched_with_bounds(bounds, pre, s.soundness_mode())
                    }
                    Layer::LogSoftmax(s) => {
                        let pre = require_pre_activation()?;
                        s.propagate_linear_batched_with_bounds(bounds, pre, s.soundness_mode())
                    }
                    Layer::LogSumExp(s) => {
                        let pre = require_pre_activation()?;
                        s.propagate_linear_batched_with_bounds(bounds, pre)
                    }
                    // LayerNorm: complex non-elementwise batched CROWN
                    Layer::LayerNorm(ln) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        ln.propagate_linear_batched_with_bounds(bounds, &pre)
                    }
                    // RMSNorm: complex non-elementwise batched CROWN (same pattern as LayerNorm)
                    Layer::RmsNorm(rn) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        rn.propagate_linear_batched_with_bounds(bounds, &pre)
                    }
                    // InstanceNorm1d: per-channel normalization batched CROWN
                    Layer::InstanceNorm1d(inn) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        inn.propagate_linear_batched_with_bounds(bounds, &pre)
                    }
                    // GroupNorm: per-group normalization batched CROWN
                    Layer::GroupNorm(gn) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        gn.propagate_linear_batched_with_bounds(bounds, &pre)
                    }
                    // AdaIN1d: style-conditioned instance normalization batched CROWN
                    Layer::AdaIN1d(adain) => {
                        let pre = reshape_pre_activation(require_pre_activation()?)?;
                        adain.propagate_linear_batched_with_bounds(bounds, &pre)
                    }
                    // BatchNorm: element-wise affine (inference mode), needs original
                    // pre-activation shape for channel-axis heuristic — no reshape.
                    Layer::BatchNorm(bn) => {
                        let pre = require_pre_activation()?;
                        bn.propagate_linear_batched_with_bounds(bounds, pre)
                    }
                    // Linear layers (no pre-activation needed)
                    Layer::Linear(l) => l.propagate_linear_batched_maybe_engine(bounds, engine),
                    // RoPE: linear (block-diagonal rotation), no pre-activation needed
                    Layer::RoPE(rope) => rope.propagate_linear_batched(bounds),
                    Layer::Conv1d(c) => {
                        let pre = require_pre_activation()?;
                        let in_len = *pre.shape().last().ok_or_else(|| {
                            NyError::InvalidSpec(
                                "Conv1d batched CROWN requires non-empty pre-activation shape".to_string(),
                            )
                        })?;
                        let mut conv_with_shape = c.clone();
                        conv_with_shape.set_input_length(in_len);
                        conv_with_shape.propagate_linear_batched_maybe_engine(bounds, engine)
                    }
                    Layer::ConvTranspose1d(c) => {
                        let pre = require_pre_activation()?;
                        let in_len = *pre.shape().last().ok_or_else(|| {
                            NyError::InvalidSpec(
                                "ConvTranspose1d batched CROWN requires non-empty pre-activation shape"
                                    .to_string(),
                            )
                        })?;
                        let mut conv_with_shape = c.clone();
                        conv_with_shape.set_input_length(in_len);
                        conv_with_shape.propagate_linear_batched_maybe_engine(bounds, engine)
                    }
                    Layer::Conv2d(c) => {
                        let pre = require_pre_activation()?;
                        let input_shape = pre.shape();
                        if input_shape.len() < 3 {
                            return Err(NyError::InvalidSpec(
                                "Conv2d batched CROWN requires at least 3D input".to_string(),
                            ));
                        }
                        let in_h = input_shape[input_shape.len() - 2];
                        let in_w = input_shape[input_shape.len() - 1];
                        let mut conv_with_shape = c.clone();
                        conv_with_shape.set_input_shape(in_h, in_w);
                        conv_with_shape.propagate_linear_batched(bounds, engine)
                    }
                    Layer::ConvTranspose2d(c) => {
                        let pre = require_pre_activation()?;
                        let input_shape = pre.shape();
                        if input_shape.len() < 3 {
                            return Err(NyError::InvalidSpec(
                                "ConvTranspose2d batched CROWN requires at least 3D input".to_string(),
                            ));
                        }
                        let in_h = input_shape[input_shape.len() - 2];
                        let in_w = input_shape[input_shape.len() - 1];
                        let mut conv_with_shape = c.clone();
                        conv_with_shape.set_input_shape(in_h, in_w);
                        conv_with_shape.propagate_linear_batched_maybe_engine(bounds, engine)
                    }
                    // Shape transforms (passthrough)
                    Layer::Flatten(_) => {
                        validate_flatten_like(require_pre_activation()?, "Flatten")?;
                        Ok(bounds.clone())
                    }
                    Layer::Reshape(_) => {
                        validate_flatten_like(require_pre_activation()?, "Reshape")?;
                        Ok(bounds.clone())
                    }
                    Layer::Pad(p) => {
                        let pre = require_pre_activation()?;
                        p.propagate_linear_batched(bounds, pre)
                    }
                    Layer::Squeeze(_) => {
                        validate_flatten_like(require_pre_activation()?, "Squeeze")?;
                        Ok(bounds.clone())
                    }
                    Layer::Unsqueeze(_) => {
                        validate_flatten_like(require_pre_activation()?, "Unsqueeze")?;
                        Ok(bounds.clone())
                    }
                    Layer::Resize(r) => {
                        let pre = require_pre_activation()?;
                        r.propagate_linear_batched(bounds, pre)
                    }
                    Layer::Transpose(t) => {
                        let pre = require_pre_activation()?;
                        let mut transpose_with_shape = t.clone();
                        transpose_with_shape.set_input_shape(pre.shape().to_vec());
                        transpose_with_shape.propagate_linear_batched(bounds)
                    }
                    // Slice: expand coefficients back to original size (#3188)
                    Layer::Slice(s) => {
                        let pre = require_pre_activation()?;
                        s.propagate_linear_batched(bounds, pre)
                    }
                    // Tile: sum coefficients from replicated output positions (#287)
                    Layer::Tile(t) => {
                        let pre = require_pre_activation()?;
                        t.propagate_linear_batched(bounds, pre)
                    }
                    // Constant arithmetic (linear, no pre-activation needed)
                    Layer::AddConstant(ac) => ac.propagate_linear_batched(bounds),
                    Layer::SubConstant(sc) => sc.propagate_linear_batched(bounds),
                    Layer::MulConstant(mc) => mc.propagate_linear_batched(bounds),
                    Layer::DivConstant(dc) => dc.propagate_linear_batched(bounds),
                    // Reduction operations (linear, need pre-activation for shape)
                    Layer::ReduceMean(rm) => {
                        let pre = require_pre_activation()?;
                        rm.propagate_linear_batched(bounds, pre)
                    }
                    Layer::ReduceSum(rs) => {
                        let pre = require_pre_activation()?;
                        rs.propagate_linear_batched(bounds, pre)
                    }
                    Layer::QdqPerturbation(qdq) => {
                        let pre = require_pre_activation()?;
                        qdq.propagate_linear_batched_with_bounds(bounds, pre)
                    }
                    Layer::CumSum(c) => {
                        let pre = require_pre_activation()?;
                        c.propagate_linear_batched(bounds, pre)
                    }
                    // Extremum reductions (nonlinear, need pre-activation for argext)
                    Layer::ReduceMax(rm) => {
                        let pre = require_pre_activation()?;
                        rm.propagate_linear_batched(bounds, pre)
                    }
                    Layer::ReduceMin(rm) => {
                        let pre = require_pre_activation()?;
                        rm.propagate_linear_batched(bounds, pre)
                    }
                    // === Unsupported in batched CROWN (#3424) ===
                    // Every variant listed explicitly — no catch-all.
                    // Adding a new Layer variant is a compile error until
                    // it's added to a supported arm above or here.
                    //
                    // Pooling (spatial ops, not batched-compatible):
                    Layer::AveragePool(_) | Layer::MaxPool2d(_)
                    // Gather (index-based, not batched-compatible):
                    | Layer::Gather(_) | Layer::ScatterAdd(_) | Layer::IndexAdd(_) | Layer::ScatterNd(_)
                    | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
                    // Skip/passthrough:
                    | Layer::SkipMerge(_) | Layer::OpaqueSkip(_)
                    // Data-dependent:
                    | Layer::NonZero(_)
                    // Binary ops (need separate multi-input handling):
                    | Layer::MatMul(_) | Layer::MulBinary(_) | Layer::Add(_)
                    | Layer::Concat(_) | Layer::Sub(_) | Layer::Div(_) | Layer::Atan2(_)
                    | Layer::BilinearCrown(_) | Layer::MinBinary(_) | Layer::MaxBinary(_)
                    | Layer::ExpandLikeLastAxis(_) | Layer::CompareTensor(_)
                    // Ternary ops:
                    | Layer::Where(_) | Layer::SelfAttention(_) => {
                        Err(NyError::UnsupportedOp(format!(
                            "{} batched CROWN backward not implemented",
                            self.layer_type()
                        )))
                    }
                }
            };
        }
        let dispatch_result: Result<BatchedLinearBounds> =
            for_each_elementwise_activation!(elementwise_batched_arms);
        // Fold any certified coefficient error the op attached to its result into
        // the bias (#vnncomp-aw-soundness), using this layer's input box — what the
        // result coefficients multiply — so the next backward op sees error-free
        // (yet sound, tightly-penalized) bounds.
        dispatch_result.map(fold_result_err)
    }
}

/// Generate `BoundPropagation` dispatch for the `Layer` enum.
///
/// Two groups: `unary` variants delegate to their inner type's `BoundPropagation` impl;
/// `binary` variants (which don't implement the trait) return `UnsupportedOp`.
///
/// Adding a new layer: add it to the appropriate group below. If it implements
/// `BoundPropagation`, add it to `unary`. If it's a multi-input op without the trait,
/// add it to `binary`.
macro_rules! impl_layer_bound_propagation {
    (
        unary { $( $Variant:ident($binding:ident) ),* $(,)? }
        binary { $( $BinVariant:ident ),* $(,)? }
    ) => {
        impl BoundPropagation for Layer {
            #[inline]
            fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
                match self {
                    $( Layer::$Variant($binding) => $binding.propagate_ibp(input), )*
                    $( Layer::$BinVariant(_) => Err(NyError::UnsupportedOp(
                        concat!(stringify!($BinVariant), " is a multi-input op — use propagate_ibp_binary/ternary").to_string(),
                    )), )*
                }
            }

            #[inline]
            fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
                match self {
                    $( Layer::$Variant($binding) => $binding.propagate_linear(bounds), )*
                    $( Layer::$BinVariant(_) => Err(NyError::UnsupportedOp(
                        concat!(stringify!($BinVariant), " CROWN propagation not supported — multi-input op").to_string(),
                    )), )*
                }
            }

            #[inline]
            fn requires_pre_activation_bounds(&self) -> bool {
                match self {
                    $( Layer::$Variant($binding) => $binding.requires_pre_activation_bounds(), )*
                    $( Layer::$BinVariant(_) => false, )*
                }
            }

            #[inline]
            fn propagate_crown_backward(
                &self,
                bounds: &LinearBounds,
                pre_activation: Option<&BoundedTensor>,
            ) -> Result<LinearBounds> {
                match self {
                    $( Layer::$Variant($binding) => $binding.propagate_crown_backward(bounds, pre_activation), )*
                    $( Layer::$BinVariant(_) => Err(NyError::UnsupportedOp(
                        concat!(stringify!($BinVariant), " CROWN backward not supported — multi-input op").to_string(),
                    )), )*
                }
            }
        }
    };
}

impl_layer_bound_propagation! {
    unary {
        // Linear / convolution
        Linear(l), Conv1d(c), Conv2d(c), ConvTranspose1d(c), ConvTranspose2d(c),
        // Pooling
        AveragePool(ap), MaxPool2d(mp),
        // Activations
        ReLU(r), LeakyReLU(lr), Clip(c), Elu(e), Selu(s), PRelu(p),
        HardSigmoid(hs), HardSwish(hw), SiLU(silu), Exp(exp), Log(log),
        Celu(c), Mish(m), ThresholdedRelu(tr), Shrink(sh), Softsign(ss), Snake(sn),
        // Softmax family
        Softmax(s), CausalSoftmax(cs), LogSoftmax(ls), LogSumExp(ls), GELU(g),
        // Normalization
        LayerNorm(ln), RmsNorm(rn), InstanceNorm1d(inn), GroupNorm(gn), AdaIN1d(adain), BatchNorm(bn),
        // Arithmetic (constant operand)
        AddConstant(ac), MulConstant(m), DivConstant(d), SubConstant(s), PowConstant(p),
        Abs(a), Sqrt(s), Reciprocal(r),
        // Reductions
        ReduceMean(rm), ReduceSum(rs), ReduceMax(rm), ReduceMin(rm),
        Topk(tk), ArgMax(am), ArgMin(am), ArgSort(asort), CumSum(cs),
        // Trigonometric / S-shaped
        Tanh(t), Sigmoid(s), Erf(e), Softplus(sp), Sin(s), Cos(c), Tan(t), Arctan(a),
        // Positional encoding
        RoPE(rope),
        // Tensor transforms
        Transpose(t), Reshape(r), Flatten(f), Pad(p), Tile(t), Gather(g),
        ScatterAdd(sa), IndexAdd(ia), ScatterNd(sn), Slice(s),
        Resize(r),
        Squeeze(sq), Unsqueeze(usq),
        // Misc
        SelfAttention(sa), Where(w), NonZero(nz),
        Floor(f), Ceil(c), Round(r), Trunc(tc), Sign(s),
        Compare(cmp),
        SkipMerge(sm), OpaqueSkip(os), QdqPerturbation(qdq),
    }
    binary {
        // Binary ops (two bounded inputs, no BoundPropagation trait impl)
        MatMul, MulBinary, Add, Concat, Sub, Div, Atan2, BilinearCrown, MinBinary, MaxBinary,
        ExpandLikeLastAxis, CompareTensor,
    }
}

impl Layer {
    /// Propagate IBP bounds for binary operations (MatMul, MulBinary, Add, Concat, Sub, Div).
    ///
    /// Returns an error for unary layers.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            Layer::MatMul(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::MulBinary(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::Add(a) => a.propagate_ibp_binary(input_a, input_b),
            Layer::Concat(c) => c.propagate_ibp_binary(input_a, input_b),
            Layer::Sub(s) => s.propagate_ibp_binary(input_a, input_b),
            Layer::Div(d) => d.propagate_ibp_binary(input_a, input_b),
            Layer::Atan2(a) => a.propagate_ibp_binary(input_a, input_b),
            Layer::BilinearCrown(bc) => bc.propagate_ibp_binary(input_a, input_b),
            Layer::MinBinary(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::MaxBinary(m) => m.propagate_ibp_binary(input_a, input_b),
            Layer::ExpandLikeLastAxis(expand) => expand.propagate_ibp_binary(input_a, input_b),
            Layer::CompareTensor(ct) => ct.propagate_ibp_binary(input_a, input_b),
            Layer::ScatterAdd(scatter) => scatter.propagate_ibp_binary(input_a, input_b),
            Layer::IndexAdd(index) => index.propagate_ibp_binary(input_a, input_b),
            Layer::ScatterNd(scatter) => scatter.propagate_ibp_binary(input_a, input_b),
            _ => Err(NyError::UnsupportedOp(format!(
                "{} is not a binary operation",
                self.layer_type()
            ))),
        }
    }

    /// Propagate IBP bounds for ternary operations (Where, SelfAttention, variable-style AdaIN).
    pub fn propagate_ibp_ternary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        input_c: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            Layer::Where(w) => w.propagate_ibp_ternary(input_a, input_b, input_c),
            Layer::SelfAttention(attn) => attn.propagate_ibp_ternary(input_a, input_b, input_c),
            Layer::ScatterAdd(scatter) => scatter.propagate_ibp_ternary(input_a, input_b, input_c),
            Layer::IndexAdd(index) => index.propagate_ibp_ternary(input_a, input_b, input_c),
            Layer::ScatterNd(scatter) => scatter.propagate_ibp_ternary(input_a, input_b, input_c),
            // Variable-style AdaIN: (x, style_gamma, style_beta).
            Layer::AdaIN1d(adain) => adain.propagate_ibp_ternary(input_a, input_b, input_c),
            _ => Err(NyError::UnsupportedOp(format!(
                "{} is not a ternary operation",
                self.layer_type()
            ))),
        }
    }
}
