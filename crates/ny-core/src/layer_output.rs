// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{LayerType, NyError, Result};

/// Output values at a specific layer in the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    /// Layer index (0-based).
    pub(crate) layer_idx: usize,
    /// Layer name (if available).
    pub(crate) layer_name: Option<String>,
    /// Layer type.
    pub(crate) layer_type: LayerType,
    /// Output values at this layer (flattened).
    pub(crate) values: Vec<f32>,
    /// Minimum output value (computed from `values`).
    pub(crate) min_value: f32,
    /// Maximum output value (computed from `values`).
    pub(crate) max_value: f32,
}

impl LayerOutput {
    /// Read-only access to the layer index.
    #[inline]
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Read-only access to the layer name.
    #[inline]
    pub fn layer_name(&self) -> Option<&str> {
        self.layer_name.as_deref()
    }

    /// Read-only access to the layer type.
    #[inline]
    pub fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }

    /// Read-only access to the output values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Read-only access to the minimum output value.
    #[inline]
    pub fn min_value(&self) -> f32 {
        self.min_value
    }

    /// Read-only access to the maximum output value.
    #[inline]
    pub fn max_value(&self) -> f32 {
        self.max_value
    }

    /// Create a new LayerOutput from values.
    ///
    /// # Panics
    /// Panics if any value is non-finite (NaN or Inf). Use `try_new` for
    /// fallible construction with a structured error.
    ///
    /// # Notes
    /// If `values` is empty, `min_value` and `max_value` are set to NaN.
    pub fn new(
        layer_idx: usize,
        layer_name: Option<String>,
        layer_type: LayerType,
        values: Vec<f32>,
    ) -> Self {
        Self::try_new(layer_idx, layer_name, layer_type, values)
            .expect("LayerOutput::new: values must be finite")
    }

    /// Create a new LayerOutput from values with validation.
    pub fn try_new(
        layer_idx: usize,
        layer_name: Option<String>,
        layer_type: LayerType,
        values: Vec<f32>,
    ) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self {
                layer_idx,
                layer_name,
                layer_type,
                values,
                min_value: f32::NAN,
                max_value: f32::NAN,
            });
        }
        let layer_name_label = layer_name.as_deref().unwrap_or("<unnamed>");
        let layer_type_label = layer_type.to_string();
        let (min_value, max_value) = Self::min_max_checked(
            "LayerOutput::try_new",
            layer_idx,
            layer_name_label,
            &layer_type_label,
            &values,
        )?;
        Ok(Self {
            layer_idx,
            layer_name,
            layer_type,
            values,
            min_value,
            max_value,
        })
    }

    fn min_max_checked(
        caller_label: &str,
        layer_idx: usize,
        layer_name_label: &str,
        layer_type_label: &str,
        values: &[f32],
    ) -> Result<(f32, f32)> {
        // Defense in depth (Trust full-verifier obligation): callers guard
        // emptiness, but make the index self-contained-safe so this helper cannot
        // panic on an empty slice regardless of caller.
        let Some(&first_value) = values.first() else {
            return Err(NyError::NumericalInstability(format!(
                "{caller_label}: empty value slice for layer {layer_idx} (name={layer_name_label}, type={layer_type_label})"
            )));
        };
        if !first_value.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "{caller_label}: non-finite value at index 0 for layer {layer_idx} (name={layer_name_label}, type={layer_type_label}): {first_value}"
            )));
        }
        let mut min_value = first_value;
        let mut max_value = first_value;
        for (idx, value) in values.iter().copied().enumerate().skip(1) {
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "{caller_label}: non-finite value at index {idx} for layer {layer_idx} (name={layer_name_label}, type={layer_type_label}): {value}"
                )));
            }
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
        }
        Ok((min_value, max_value))
    }
}
