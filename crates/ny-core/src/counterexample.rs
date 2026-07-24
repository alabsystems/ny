// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{
    checked_shape_product, nan_propagating_max, nan_propagating_min, Bound, LayerOutput, NyError,
    Result, ViolatedConstraint,
};

/// Detailed counterexample with layer-by-layer trace and explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformativeCounterexample {
    /// Concrete input values that violate the property.
    input: Vec<f32>,
    /// Output values at the counterexample input.
    output: Vec<f32>,
    /// Layer-by-layer trace showing values at each layer.
    trace: Vec<LayerOutput>,
    /// Which constraint was violated.
    violated_constraint: Option<ViolatedConstraint>,
    /// Human-readable explanation of the counterexample.
    explanation: String,
}

impl InformativeCounterexample {
    /// Create an informative counterexample from basic inputs/outputs.
    ///
    /// If bounds are provided, detects which constraint was violated.
    /// If bounds length doesn't match output length, the explanation notes the
    /// mismatch and `violated_constraint` is set to `None`.
    ///
    /// # REQUIRES
    /// - None
    ///
    /// # ENSURES
    /// - Returns `InformativeCounterexample` with `input` and `output` set
    /// - `trace` is empty on the returned value
    /// - `violated_constraint` matches `ViolatedConstraint::detect` when bounds provided and lengths match
    /// - `explanation` contains mismatch note when bounds lengths don't match
    pub fn new(input: Vec<f32>, output: Vec<f32>, output_bounds: Option<&[Bound]>) -> Self {
        let bounds_mismatch = output_bounds
            .map(|bounds| bounds.len() != output.len())
            .unwrap_or(false);
        let violated_constraint = output_bounds.and_then(|bounds| {
            // Gracefully handle length mismatch - bounds_mismatch flag ensures
            // the explanation will note this issue
            if output.len() != bounds.len() {
                return None;
            }
            ViolatedConstraint::detect(&output, bounds)
        });

        let explanation = match &violated_constraint {
            Some(vc) => format!(
                "Property violated: {}. Input: {:?}, Output: {:?}",
                vc.explain(),
                &input[..input.len().min(5)],
                &output[..output.len().min(5)]
            ),
            None => {
                let mut message = format!(
                    "Counterexample found. Input: {:?}, Output: {:?}",
                    &input[..input.len().min(5)],
                    &output[..output.len().min(5)]
                );
                if bounds_mismatch {
                    message.push_str(&format!(
                        " (output_bounds length mismatch: {} vs {})",
                        output.len(),
                        output_bounds.map_or(0, <[Bound]>::len)
                    ));
                }
                message
            }
        };

        Self {
            input,
            output,
            trace: Vec::new(),
            violated_constraint,
            explanation,
        }
    }

    /// Read-only access to the counterexample input values.
    #[inline]
    pub fn input(&self) -> &[f32] {
        &self.input
    }

    /// Read-only access to the counterexample output values.
    #[inline]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Read-only access to the layer-by-layer trace.
    #[inline]
    pub fn trace(&self) -> &[LayerOutput] {
        &self.trace
    }

    /// Read-only access to the violated constraint, if any.
    #[inline]
    pub fn violated_constraint(&self) -> Option<&ViolatedConstraint> {
        self.violated_constraint.as_ref()
    }

    /// Read-only access to the human-readable explanation.
    #[inline]
    pub fn explanation(&self) -> &str {
        &self.explanation
    }

    /// Add a layer output to the trace.
    ///
    /// # REQUIRES
    /// - None.
    ///
    /// # ENSURES
    /// - Appends `layer_output` to the end of `trace`
    pub fn add_layer_output(&mut self, layer_output: LayerOutput) {
        self.trace.push(layer_output);
    }

    /// Set the layer trace.
    ///
    /// # REQUIRES
    /// - None.
    ///
    /// # ENSURES
    /// - Returns the same value with `trace` replaced by `trace`
    pub fn with_trace(mut self, trace: Vec<LayerOutput>) -> Self {
        self.trace = trace;
        self
    }

    /// Get a formatted string showing the layer-by-layer trace.
    ///
    /// # REQUIRES
    /// - None.
    ///
    /// # ENSURES
    /// - Returns a human-readable summary of the trace
    /// - Returns `"No layer trace available."` when `trace` is empty
    pub fn format_trace(&self) -> String {
        if self.trace.is_empty() {
            return "No layer trace available.".to_string();
        }

        let mut result = String::from("Layer-by-layer trace:\n");
        for layer in &self.trace {
            let name_part = layer
                .layer_name
                .as_ref()
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            result.push_str(&format!(
                "  Layer {:3}{}: {:10} | min={:10.4} max={:10.4} | {} values\n",
                layer.layer_idx,
                name_part,
                layer.layer_type,
                layer.min_value,
                layer.max_value,
                layer.values.len()
            ));
        }
        result
    }
}

/// A concrete counterexample: a single input and its resulting output.
///
/// This is the lightweight, framework-facing counterexample surface (P6). It
/// carries the raw input/output vectors and optional shapes, without the
/// layer-by-layer trace of [`InformativeCounterexample`]. It is additive and
/// does not replace any existing type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Counterexample {
    /// Concrete input values (row-major) that exhibit the counterexample.
    pub input: Vec<f32>,
    /// Output values produced by the network at `input` (row-major).
    pub output: Vec<f32>,
    /// Optional shape of `input`; its product should equal `input.len()`.
    pub input_shape: Option<Vec<usize>>,
    /// Optional shape of `output`; its product should equal `output.len()`.
    pub output_shape: Option<Vec<usize>>,
}

impl Counterexample {
    /// Create a counterexample from concrete input and output values.
    ///
    /// Shapes are left unset; use [`with_input_shape`](Self::with_input_shape)
    /// and [`with_output_shape`](Self::with_output_shape) to attach them.
    #[must_use]
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self {
            input,
            output,
            input_shape: None,
            output_shape: None,
        }
    }

    /// Attach the input shape, consuming and returning `self` (builder style).
    #[must_use]
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = Some(shape);
        self
    }

    /// Attach the output shape, consuming and returning `self` (builder style).
    #[must_use]
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = Some(shape);
        self
    }

    /// Set the input shape in place.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Set the output shape in place.
    pub fn set_output_shape(&mut self, shape: Vec<usize>) {
        self.output_shape = Some(shape);
    }
}

/// Per-element interval bounds with an explicit tensor shape.
///
/// Unlike a scalar `[min, max]` collapse, this preserves the full elementwise
/// `[lower, upper]` interval for every tensor element, alongside the shape it
/// describes. The historical scalar summary is still available via
/// [`scalar_summary`](Self::scalar_summary) but is now opt-in, not forced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerElementBounds {
    /// Lower endpoints, one per element (row-major).
    pub lower: Vec<f32>,
    /// Upper endpoints, one per element (row-major).
    pub upper: Vec<f32>,
    /// Shape these bounds describe; its product equals `lower.len()`.
    pub shape: Vec<usize>,
}

impl PerElementBounds {
    /// Construct validated per-element bounds.
    ///
    /// # REQUIRES
    /// - `lower.len() == upper.len() == prod(shape)`
    /// - For every index, `lower[i] <= upper[i]` OR either endpoint is NaN
    ///
    /// # ENSURES
    /// - Returns `Ok(Self)` when all invariants hold
    /// - Returns `Err(NyError::InvalidSpec)` on any length/shape mismatch or
    ///   strictly inverted (non-NaN) interval
    pub fn try_new(lower: Vec<f32>, upper: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(NyError::InvalidSpec(format!(
                "PerElementBounds: lower length {} != upper length {}",
                lower.len(),
                upper.len()
            )));
        }
        let product = checked_shape_product(&shape).ok_or_else(|| {
            NyError::InvalidSpec("PerElementBounds: shape product overflows usize".to_string())
        })?;
        if product != lower.len() {
            return Err(NyError::InvalidSpec(format!(
                "PerElementBounds: shape product {} does not match bounds length {}",
                product,
                lower.len()
            )));
        }
        for (i, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
            // NaN endpoints are permitted (they signal unsound/degraded values
            // that downstream guards must handle); only finite inversions fail.
            if !lo.is_nan() && !hi.is_nan() && lo > hi {
                return Err(NyError::InvalidSpec(format!(
                    "PerElementBounds: lower[{i}] = {lo} > upper[{i}] = {hi}"
                )));
            }
        }
        Ok(Self {
            lower,
            upper,
            shape,
        })
    }

    /// Number of elements (length of either endpoint vector).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Whether there are no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Mask of elements whose `lower` and `upper` are both finite.
    ///
    /// `true` where both endpoints are finite (neither NaN nor infinite).
    #[must_use]
    pub fn finite_mask(&self) -> Vec<bool> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(&lo, &hi)| lo.is_finite() && hi.is_finite())
            .collect()
    }

    /// Elementwise interval width `upper - lower`.
    ///
    /// Saturates at `f32::INFINITY`: any element whose subtraction is non-finite
    /// (e.g. `INF - finite`, or an `INF - INF` / NaN producing case) widens to
    /// `f32::INFINITY` rather than leaking NaN, keeping the width conservative.
    #[must_use]
    pub fn width(&self) -> Vec<f32> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(&lo, &hi)| {
                let w = hi - lo;
                if w.is_finite() {
                    w
                } else {
                    f32::INFINITY
                }
            })
            .collect()
    }

    /// Collapse to a scalar `(min lower, max upper)` over FINITE elements.
    ///
    /// This is the historical back-compat scalar summary, now opt-in. Non-finite
    /// endpoints (NaN or infinite) are skipped on each side independently. Uses
    /// NaN-propagating folds for soundness; with non-finite values filtered out
    /// the result is finite. Returns `(f32::INFINITY, f32::NEG_INFINITY)` when no
    /// finite endpoints exist on the respective side (empty-fold identity).
    #[must_use]
    pub fn scalar_summary(&self) -> (f32, f32) {
        let min_lower = self
            .lower
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, nan_propagating_min);
        let max_upper = self
            .upper
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        (min_lower, max_upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_rejects_length_mismatch() {
        let err = PerElementBounds::try_new(vec![0.0, 1.0], vec![1.0], vec![2]);
        assert!(matches!(err, Err(NyError::InvalidSpec(_))));
    }

    #[test]
    fn try_new_rejects_shape_mismatch() {
        // lengths agree (3) but shape product is 4.
        let err = PerElementBounds::try_new(vec![0.0; 3], vec![1.0; 3], vec![2, 2]);
        assert!(matches!(err, Err(NyError::InvalidSpec(_))));
    }

    #[test]
    fn try_new_rejects_inverted_finite_interval() {
        let err = PerElementBounds::try_new(vec![1.0], vec![0.0], vec![1]);
        assert!(matches!(err, Err(NyError::InvalidSpec(_))));
    }

    #[test]
    fn try_new_allows_nan_endpoints() {
        // A NaN endpoint must not be treated as an inversion.
        let ok = PerElementBounds::try_new(vec![f32::NAN], vec![0.0], vec![1]);
        assert!(ok.is_ok());
    }

    #[test]
    fn finite_mask_flags_infinities() {
        let b = PerElementBounds::try_new(
            vec![0.0, f32::NEG_INFINITY, 1.0],
            vec![1.0, 2.0, f32::INFINITY],
            vec![3],
        )
        .unwrap();
        assert_eq!(b.finite_mask(), vec![true, false, false]);
    }

    #[test]
    fn width_handles_infinities() {
        let b = PerElementBounds::try_new(
            vec![0.0, f32::NEG_INFINITY, -1.0],
            vec![2.0, 5.0, f32::INFINITY],
            vec![3],
        )
        .unwrap();
        let w = b.width();
        assert_eq!(w[0], 2.0);
        assert_eq!(w[1], f32::INFINITY);
        assert_eq!(w[2], f32::INFINITY);
    }

    #[test]
    fn scalar_summary_ignores_non_finite() {
        let b = PerElementBounds::try_new(
            vec![f32::NEG_INFINITY, -2.0, 3.0],
            vec![1.0, f32::INFINITY, 5.0],
            vec![3],
        )
        .unwrap();
        let (min_lower, max_upper) = b.scalar_summary();
        assert_eq!(min_lower, -2.0);
        assert_eq!(max_upper, 5.0);
    }

    #[test]
    fn len_and_is_empty() {
        let empty = PerElementBounds::try_new(vec![], vec![], vec![0]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        let b = PerElementBounds::try_new(vec![0.0, 1.0], vec![1.0, 2.0], vec![2]).unwrap();
        assert!(!b.is_empty());
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn counterexample_builders_set_shapes() {
        let cx = Counterexample::new(vec![1.0, 2.0], vec![3.0])
            .with_input_shape(vec![2])
            .with_output_shape(vec![1]);
        assert_eq!(cx.input_shape, Some(vec![2]));
        assert_eq!(cx.output_shape, Some(vec![1]));
    }

    #[test]
    fn counterexample_round_trips_through_serde_json() {
        let cx = Counterexample::new(vec![0.5, -1.0], vec![2.0])
            .with_input_shape(vec![2])
            .with_output_shape(vec![1]);
        let json = serde_json::to_string(&cx).expect("serialize");
        let back: Counterexample = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cx, back);
    }
}
