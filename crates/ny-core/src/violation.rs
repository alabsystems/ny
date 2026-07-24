// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::Bound;

/// Information about which constraint was violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolatedConstraint {
    /// Index of the output dimension that was violated.
    output_idx: usize,
    /// The output value that violated the constraint.
    actual_value: f32,
    /// The required bound that was violated.
    required_bound: Bound,
    /// Direction of violation.
    violation_type: ViolationType,
    /// Amount by which the bound was violated.
    violation_amount: f32,
}

/// Type of constraint violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Output value below lower bound.
    BelowLower,
    /// Output value above upper bound.
    AboveUpper,
}

impl ViolatedConstraint {
    /// Create a violated constraint record.
    pub fn new(
        output_idx: usize,
        actual_value: f32,
        required_bound: Bound,
        violation_type: ViolationType,
        violation_amount: f32,
    ) -> Self {
        Self {
            output_idx,
            actual_value,
            required_bound,
            violation_type,
            violation_amount,
        }
    }

    /// Get the violated output index.
    pub fn output_idx(&self) -> usize {
        self.output_idx
    }

    /// Get the observed violating value.
    pub fn actual_value(&self) -> f32 {
        self.actual_value
    }

    /// Get the required bound for this output.
    pub fn required_bound(&self) -> Bound {
        self.required_bound
    }

    /// Get the violation type.
    pub fn violation_type(&self) -> ViolationType {
        self.violation_type
    }

    /// Get the violation magnitude.
    pub fn violation_amount(&self) -> f32 {
        self.violation_amount
    }

    /// Detect which constraint was violated given output values and bounds.
    ///
    /// # REQUIRES
    /// - `output.len() == bounds.len()` for meaningful matching
    ///
    /// # ENSURES
    /// - Returns `Some` for the first violated bound in index order
    /// - Returns `None` iff no output violates its corresponding bound
    pub fn detect(output: &[f32], bounds: &[Bound]) -> Option<Self> {
        debug_assert_eq!(
            output.len(),
            bounds.len(),
            "ViolatedConstraint::detect: output length must match bounds length"
        );
        if output.len() != bounds.len() {
            return None;
        }
        for (idx, (&value, bound)) in output.iter().zip(bounds.iter()).enumerate() {
            // NaN output is always a violation — it cannot satisfy any finite bound.
            // IEEE 754: NaN < x and NaN > x are both false, so without this guard
            // a NaN counterexample would silently pass as non-violating. (#3291 F1)
            if value.is_nan() {
                return Some(Self::new(
                    idx,
                    value,
                    *bound,
                    ViolationType::BelowLower,
                    f32::INFINITY,
                ));
            }
            if value < bound.lower {
                return Some(Self::new(
                    idx,
                    value,
                    *bound,
                    ViolationType::BelowLower,
                    bound.lower - value,
                ));
            }
            if value > bound.upper {
                return Some(Self::new(
                    idx,
                    value,
                    *bound,
                    ViolationType::AboveUpper,
                    value - bound.upper,
                ));
            }
        }
        None
    }

    /// Human-readable description of the violation.
    ///
    /// # REQUIRES
    /// - None.
    ///
    /// # ENSURES
    /// - Returns a formatted description matching the violation type
    pub fn explain(&self) -> String {
        match self.violation_type {
            ViolationType::BelowLower => format!(
                "Output[{}] = {:.6} < lower bound {:.6} (by {:.6})",
                self.output_idx,
                self.actual_value,
                self.required_bound.lower,
                self.violation_amount
            ),
            ViolationType::AboveUpper => format!(
                "Output[{}] = {:.6} > upper bound {:.6} (by {:.6})",
                self.output_idx,
                self.actual_value,
                self.required_bound.upper,
                self.violation_amount
            ),
        }
    }
}
