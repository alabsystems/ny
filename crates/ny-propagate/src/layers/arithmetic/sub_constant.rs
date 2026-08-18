// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Subtract constant layer: y = x - c or y = c - x (element-wise).

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, sub_down_f32, sub_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::common::{dot_bias_f64, extract_scalar_constant_for_batched, scalar_row_sum_f64};
use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::shape::{broadcast_flat_index_map, broadcast_shapes};
use crate::{BatchedLinearBounds, LinearBounds};

/// Subtract constant layer: y = x - c or y = c - x (element-wise).
///
/// Used in LayerNorm for mean subtraction.
#[derive(Debug, Clone)]
pub struct SubConstantLayer {
    /// The constant tensor.
    pub(crate) constant: ArrayD<f32>,
    /// If true: y = constant - x, if false: y = x - constant
    pub reverse: bool,
}

impl SubConstantLayer {
    /// Validate and create a new layer for y = x - constant.
    pub fn try_new(constant: ArrayD<f32>) -> Result<Self> {
        validate_finite_array(&constant, "SubConstantLayer", "constant")?;
        Ok(Self {
            constant,
            reverse: false,
        })
    }

    /// Create a new layer for y = x - constant.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self::try_new(constant).expect("invariant: SubConstantLayer::new requires finite constant")
    }

    /// Validate and create a new layer for y = constant - x.
    pub fn try_new_reverse(constant: ArrayD<f32>) -> Result<Self> {
        let mut layer = Self::try_new(constant)?;
        layer.reverse = true;
        Ok(layer)
    }

    /// Create a new layer for y = constant - x.
    pub fn new_reverse(constant: ArrayD<f32>) -> Self {
        Self::try_new_reverse(constant)
            .expect("invariant: SubConstantLayer::new_reverse requires finite constant")
    }

    /// Create a scalar subtraction layer (y = x - scalar).
    pub fn scalar(value: f32) -> Self {
        Self::new(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Return the constant tensor.
    pub fn constant(&self) -> &ArrayD<f32> {
        &self.constant
    }

    /// CROWN backward through a BROADCASTING SubConstant (#ml4acopf-genbab).
    ///
    /// Handles `y = broadcast(x) - broadcast(c)` (or `c - x` when `reverse`)
    /// where the output shape is the ONNX broadcast of the input and constant
    /// shapes. The ml4acopf trigonometric threshold banks are the motivating
    /// pattern: `Sub(x[1,P,1], thresholds[K]) -> [1,P,K]` — the input expands
    /// along the last axis AND the constant broadcasts across the leading axes.
    ///
    /// Backward substitution: with `A` over the output (columns = flat output),
    /// `A @ y = A @ (B_x x - B_c c)` where `B_x`/`B_c` are the implicit
    /// broadcast operators, so:
    /// - new `A' = A @ B_x` — column `i` of the input receives the SUM of the
    ///   `A` columns of every output position that reads `x[i]`;
    /// - new bias `b' = b - A @ (B_c c)` (`+` when `reverse`, which also
    ///   negates `A'`; no lower/upper swap — CROWN composes by substitution).
    ///
    /// Soundness (#vnncomp-aw-soundness): the column reduction sums `fan_in`
    /// f32 coefficients per input cell in round-to-nearest — the same
    /// scatter-add class as `ExpandLikeLastAxisLayer::propagate_linear_binary`
    /// — so it carries the certified `gamma_{fan_in} * S + prop` coefficient
    /// error outward via `new_or_conservative_with_err`. The bias dot product
    /// accumulates in f64 with directed rounding on the f32 cast (#3157).
    fn propagate_linear_broadcast_backward(
        &self,
        bounds: &LinearBounds,
        input_shape: &[usize],
    ) -> Result<LinearBounds> {
        let layer_dim = bounds.lower_a().ncols();
        let const_shape = self.constant.shape();
        let out_shape =
            broadcast_shapes(input_shape, const_shape).ok_or_else(|| NyError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: const_shape.to_vec(),
            })?;
        let out_len = checked_shape_product(&out_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "SubConstant broadcast: output shape {:?} overflows usize",
                out_shape
            ))
        })?;
        if out_len != layer_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![layer_dim],
                got: vec![out_len],
            });
        }
        let input_len = checked_shape_product(input_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "SubConstant broadcast: input shape {:?} overflows usize",
                    input_shape
                ))
            })?
            .max(1);

        // Bias contribution: A @ broadcast(c), f64 accumulation (#3157).
        let c_broadcast: Array1<f32> = self
            .constant
            .broadcast(IxDyn(&out_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: out_shape.clone(),
                got: const_shape.to_vec(),
            })?
            .iter()
            .copied()
            .collect();
        let (lower_c_f64, upper_c_f64) =
            dot_bias_f64(bounds.lower_a(), bounds.upper_a(), &c_broadcast);

        // Column reduction through the implicit input broadcast: every output
        // flat index maps to the input flat index it reads (row-major).
        let index_map = broadcast_flat_index_map(&out_shape, input_shape);
        let num_outputs = bounds.num_outputs();
        // Uniform fan-in: broadcasting replicates each input cell the same
        // number of times (product of the broadcast axis widths).
        let fan_in = layer_dim / input_len;
        let mut lower_a = Array2::<f32>::zeros((num_outputs, input_len));
        let mut upper_a = Array2::<f32>::zeros((num_outputs, input_len));
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let mut s_lower = Array2::<f64>::zeros((num_outputs, input_len));
        let mut s_upper = Array2::<f64>::zeros((num_outputs, input_len));
        let mut p_lower = Array2::<f64>::zeros((num_outputs, input_len));
        let mut p_upper = Array2::<f64>::zeros((num_outputs, input_len));
        for row in 0..num_outputs {
            for (out_idx, &in_idx) in index_map.iter().enumerate() {
                let wl = bounds.lower_a()[[row, out_idx]];
                let wu = bounds.upper_a()[[row, out_idx]];
                lower_a[[row, in_idx]] += wl;
                upper_a[[row, in_idx]] += wu;
                s_lower[[row, in_idx]] += (wl as f64).abs();
                s_upper[[row, in_idx]] += (wu as f64).abs();
                if let Some(e) = in_lower_err {
                    p_lower[[row, in_idx]] += (e[[row, out_idx]] as f64).abs();
                }
                if let Some(e) = in_upper_err {
                    p_upper[[row, in_idx]] += (e[[row, out_idx]] as f64).abs();
                }
            }
        }

        // Bias with directed rounding; A negated for reverse (substitution,
        // no lower/upper swap — same convention as `propagate_linear`).
        let (new_lower_b, new_upper_b) = if self.reverse {
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_c_f64;
            (
                lb_f64.mapv(|x| next_down_f32(x as f32)),
                ub_f64.mapv(|x| next_up_f32(x as f32)),
            )
        } else {
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) - &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) - &upper_c_f64;
            (
                lb_f64.mapv(|x| next_down_f32(x as f32)),
                ub_f64.mapv(|x| next_up_f32(x as f32)),
            )
        };
        if self.reverse {
            lower_a.mapv_inplace(|v| -v);
            upper_a.mapv_inplace(|v| -v);
        }

        if fan_in >= 2 || in_lower_err.is_some() || in_upper_err.is_some() {
            let gamma = if fan_in >= 2 {
                crate::layers::linear::crown_single_gamma_n_f32(fan_in)
            } else {
                0.0
            };
            let lower_err = ndarray::Zip::from(&s_lower)
                .and(&p_lower)
                .map_collect(|&s, &p| next_up_f32((gamma * s + p) as f32));
            let upper_err = ndarray::Zip::from(&s_upper)
                .and(&p_upper)
                .map_collect(|&s, &p| next_up_f32((gamma * s + p) as f32));
            LinearBounds::new_or_conservative_with_err(
                lower_a,
                new_lower_b,
                upper_a,
                new_upper_b,
                lower_err,
                upper_err,
            )
        } else {
            LinearBounds::new_or_conservative(lower_a, new_lower_b, upper_a, new_upper_b)
        }
    }
}

impl BoundPropagation for SubConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Compute broadcast output shape (ONNX semantics).
        // Handles both constant→input and input→constant broadcast.
        let output_shape =
            broadcast_shapes(input_shape, const_shape).ok_or_else(|| NyError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: const_shape.to_vec(),
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

        // Directed rounding (#vnncomp-softmax-complex): plain `propagate_ibp`
        // is a production node-bound source, so the subtraction must round
        // OUTWARD. f64 subtraction of two f32 values is exact; only the
        // f64→f32 cast rounds, so next_down/next_up on the cast guarantees
        // enclosure of the real-arithmetic interval.
        let (out_lower, out_upper) = if self.reverse {
            // y = c - x: when x is large, y is small and vice versa
            // y_lower = c - x_upper
            // y_upper = c - x_lower
            let lower = ndarray::Zip::from(&c)
                .and(&upper_in)
                .map_collect(|&ci, &ui| sub_down_f32(ci, ui));
            let upper = ndarray::Zip::from(&c)
                .and(&lower_in)
                .map_collect(|&ci, &li| sub_up_f32(ci, li));
            (lower, upper)
        } else {
            // y = x - c: subtraction preserves order
            // y_lower = x_lower - c
            // y_upper = x_upper - c
            let lower = ndarray::Zip::from(&lower_in)
                .and(&c)
                .map_collect(|&li, &ci| sub_down_f32(li, ci));
            let upper = ndarray::Zip::from(&upper_in)
                .and(&c)
                .map_collect(|&ui, &ci| sub_up_f32(ui, ci));
            (lower, upper)
        };

        // OpaqueSkip taint (#opaque-skip-six-sites): an upstream OpaqueSkip
        // legitimately emits ±Inf endpoints. The constant is validated finite
        // at construction (`validate_finite_array`), so `±inf ∓ c` and
        // `c - (±inf)` stay clean ±Inf and the only NaN-producing pattern for
        // subtraction (inf - inf) is unreachable. A NaN here therefore implies
        // a NaN INPUT — a real bug — which `new_allow_infinite` still rejects
        // as a hard error. (Contrast binary Sub, where both operands can be
        // ±Inf and inf - inf forces `new_repaired(Conservative)`.)
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    /// Unified CROWN backward. Non-broadcast cases (constant matches the layer
    /// dim, or scalar constant, with an un-broadcast input) delegate to
    /// `propagate_linear` — byte-identical to the previous behavior. Broadcast
    /// cases (ml4acopf threshold banks: `Sub(x[1,P,1], c[K]) -> [1,P,K]`)
    /// route to `propagate_linear_broadcast_backward`, which previously
    /// hard-errored with `ShapeMismatch` and killed GenBaB child propagation
    /// (#ml4acopf-genbab).
    fn propagate_crown_backward(
        &self,
        bounds: &LinearBounds,
        pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        let layer_dim = bounds.lower_a().ncols();
        let c_len = self.constant.len();
        let elementwise_const = c_len == layer_dim || c_len == 1;
        let input_matches = pre_activation.is_none_or(|p| p.len() == layer_dim);
        if elementwise_const && input_matches {
            return self.propagate_linear(bounds).map(Cow::into_owned);
        }
        match pre_activation {
            Some(pre) => self.propagate_linear_broadcast_backward(bounds, pre.shape()),
            // Without the input shape the broadcast column reduction is
            // ill-posed; keep the legacy behavior (errors preserved).
            None => self.propagate_linear(bounds).map(Cow::into_owned),
        }
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x - c: just shift the bias by -c
        // For y = c - x: negate coefficients and adjust bias

        // Flatten constant to 1D and convert to Array1 for compatibility with LinearBounds
        // bounds.lower_b is Array1<f32>, so we need c as Array1<f32> for subtraction
        let c_flat: Array1<f32> = self
            .constant
            .clone()
            .into_shape_with_order((self.constant.len(),))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![self.constant.len()],
                got: self.constant.shape().to_vec(),
            })?;

        // Debug shapes
        debug!(
            "SubConstant propagate_linear: constant shape {:?} (len {}), lower_b len {}, c_flat len {}",
            self.constant.shape(),
            self.constant.len(),
            bounds.lower_b().len(),
            c_flat.len()
        );

        // CROWN backward propagation for SubConstant.
        //
        // For y = x - c (where y is output, x is input):
        // - Before: lA @ y + lb <= output (where y is this layer's output)
        // - Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
        // - After: new_lA = lA, new_lb = lb - lA @ c
        //
        // The bias adjustment requires lA @ c, NOT lb - c!
        // lA has shape (num_outputs, layer_dim), c has shape (layer_dim,)
        // lA @ c gives shape (num_outputs,) which matches lb
        //
        // Special case: scalar constant (broadcasts to input shape)
        // If c is scalar, c_flat has len=1 but layer_dim > 1
        // In this case, lA @ c_broadcast = c * sum(lA, axis=1)
        // Compute A @ c with f64 accumulation + directed rounding (#3157, #1863).
        let layer_dim = bounds.lower_a().ncols();
        let (lower_c_f64, upper_c_f64) = if c_flat.len() == 1 && layer_dim > 1 {
            scalar_row_sum_f64(bounds.lower_a(), bounds.upper_a(), c_flat[0])
        } else if c_flat.len() == layer_dim {
            dot_bias_f64(bounds.lower_a(), bounds.upper_a(), &c_flat)
        } else if c_flat.len() > layer_dim {
            // Broadcast expansion: constant is larger than input. CROWN backward
            // through broadcast expansion requires column aggregation, not yet
            // implemented — fall back to IBP at this layer boundary.
            return Err(NyError::UnsupportedOp(format!(
                "SubConstant CROWN backward: broadcast expansion \
                 (constant {} elements > layer dim {}) not supported",
                c_flat.len(),
                layer_dim
            )));
        } else {
            return Err(NyError::ShapeMismatch {
                expected: vec![layer_dim],
                got: vec![c_flat.len()],
            });
        };

        if self.reverse {
            // y = c - x: CROWN backward substitution (no swap).
            // Before: lA @ y + lb, uA @ y + ub
            // Substitute y = c - x: lA @ (c - x) + lb = -lA @ x + (lb + lA @ c)
            // No swap: CROWN composes by substitution, not IBP.
            // Reference: designs/2026-01-29-crown-affine-negative-scale.md
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) + &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) + &upper_c_f64;
            let new_lower_b = lb_f64.mapv(|x| next_down_f32(x as f32));
            let new_upper_b = ub_f64.mapv(|x| next_up_f32(x as f32));
            Ok(Cow::Owned(LinearBounds::new_or_conservative(
                -bounds.lower_a(),
                new_lower_b,
                -bounds.upper_a(),
                new_upper_b,
            )?))
        } else {
            // y = x - c: new_lb = old_lb - lA @ c, new_ub = old_ub - uA @ c
            let lb_f64 = bounds.lower_b().mapv(|x| x as f64) - &lower_c_f64;
            let ub_f64 = bounds.upper_b().mapv(|x| x as f64) - &upper_c_f64;
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
}

impl SubConstantLayer {
    /// Batched CROWN backward propagation through SubConstant.
    ///
    /// For y = x - c: coefficient matrices unchanged, bias shifts by -A @ c
    /// For y = c - x: coefficient matrices negated, bias shifts by +A @ c
    ///
    /// Currently only supports scalar constants in batched mode.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // Compute the bias contribution: A @ c
        // For scalar c: A @ c = c * sum(A, axis=-1)
        // For non-scalar c: Not yet supported in batched mode
        let c_val = extract_scalar_constant_for_batched(&self.constant, "SubConstant")?;
        debug!("SubConstant batched CROWN: scalar c = {}", c_val);

        // f64 accumulation + directed rounding to match non-batched path (#2423).
        // Previous f32 sum_axis lost precision in large reductions.
        let ndim = bounds.lower_a.ndim();
        let c_f64 = c_val as f64;

        // Row sums in f64 for precision.
        let lower_sum_f64 = bounds.lower_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));
        let upper_sum_f64 = bounds.upper_a.mapv(|v| v as f64).sum_axis(Axis(ndim - 1));

        // Bias contributions: c * sum(A_row) in f64.
        let lower_contrib_f64 = lower_sum_f64.mapv(|v| v * c_f64);
        let upper_contrib_f64 = upper_sum_f64.mapv(|v| v * c_f64);

        if self.reverse {
            // y = c - x: CROWN backward substitution (no swap).
            // Substitute y = c - x: lA @ (c - x) + lb = -lA @ x + (lb + lA @ c)
            // Reference: designs/2026-01-29-crown-affine-negative-scale.md
            debug!("SubConstant batched CROWN: reverse mode (c - x)");
            // Add bias + contribution in f64, directed rounding, NaN → conservative.
            let lb_f64 = bounds.lower_b.mapv(|v| v as f64) + &lower_contrib_f64;
            let ub_f64 = bounds.upper_b.mapv(|v| v as f64) + &upper_contrib_f64;

            let new_lower_b = lb_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(f)
                }
            });
            let new_upper_b = ub_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::INFINITY
                } else {
                    next_up_f32(f)
                }
            });

            BatchedLinearBounds::new_or_conservative(
                bounds.lower_a.mapv(|v| -v),
                new_lower_b,
                bounds.upper_a.mapv(|v| -v),
                new_upper_b,
                bounds.input_shape.clone(),
                bounds.output_shape.clone(),
            )
        } else {
            // y = x - c: CROWN backward substitution.
            // Substitute y = x - c: lA @ (x - c) + lb = lA @ x + (lb - lA @ c)
            debug!("SubConstant batched CROWN: standard mode (x - c)");
            // Subtract bias contribution in f64, directed rounding, NaN → conservative.
            let lb_f64 = bounds.lower_b.mapv(|v| v as f64) - &lower_contrib_f64;
            let ub_f64 = bounds.upper_b.mapv(|v| v as f64) - &upper_contrib_f64;

            let new_lower_b = lb_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(f)
                }
            });
            let new_upper_b = ub_f64.mapv(|v| {
                let f = v as f32;
                if f.is_nan() {
                    f32::INFINITY
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
    }
}

/// OpaqueSkip taint probes (#opaque-skip-six-sites): the IBP output
/// constructor must let the legitimate ±Inf an upstream OpaqueSkip emits flow
/// through as widened bounds, while NaN inputs remain a hard error.
#[cfg(test)]
mod opaque_skip_taint_tests {
    use super::*;

    fn opaque_input() -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
            ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
        )
        .unwrap()
    }

    /// [-inf, +inf] - c (and c - [-inf, +inf]) must propagate as [-inf, +inf],
    /// not abort with NumericalInstability. The constant is finite, so no
    /// inf - inf NaN pattern exists.
    #[test]
    fn test_ibp_opaque_skip_inf_input_flows() {
        let forward = SubConstantLayer::new(ArrayD::from_elem(IxDyn(&[2]), 1.5_f32));
        let out = forward
            .propagate_ibp(&opaque_input())
            .expect("±inf input must propagate through x - c");
        assert_eq!(out.lower()[[0]], f32::NEG_INFINITY);
        assert_eq!(out.upper()[[1]], f32::INFINITY);

        let reverse = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[2]), 1.5_f32));
        let out = reverse
            .propagate_ibp(&opaque_input())
            .expect("±inf input must propagate through c - x");
        assert_eq!(out.lower()[[0]], f32::NEG_INFINITY);
        assert_eq!(out.upper()[[1]], f32::INFINITY);
    }

    /// NaN input (a real bug, not OpaqueSkip taint) must still hard-error:
    /// `new_allow_infinite` rejects NaN.
    #[test]
    fn test_ibp_nan_input_still_errors() {
        let layer = SubConstantLayer::new(ArrayD::from_elem(IxDyn(&[1]), 1.0_f32));
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
            ArrayD::from_elem(IxDyn(&[1]), 1.0_f32),
        )
        .unwrap();
        assert!(
            layer.propagate_ibp(&input).is_err(),
            "NaN input must remain a hard error"
        );
    }
}
