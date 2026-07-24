// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use crate::{Bound, OutputConstraint};

/// Specification of a property to verify.
///
/// # Invariants (enforced by constructors and deserialization)
/// - `input_bounds` and `output_bounds` are non-empty
/// - All bounds are well-formed (no NaN, lower <= upper — enforced by `Bound`)
/// - If `input_shape` is present, its product equals `input_bounds.len()`
/// - All `output_constraints` are structurally well-formed (see [`OutputConstraint::validate`])
#[derive(Debug, Clone, Serialize)]
pub struct VerificationSpec {
    /// Input bounds (per element, in row-major order).
    input_bounds: Vec<Bound>,
    /// Required output bounds (property holds if outputs within these).
    output_bounds: Vec<Bound>,
    /// Optional timeout in milliseconds.
    timeout_ms: Option<u64>,
    /// Optional input shape (if None, input is treated as 1D).
    /// This allows proper reshaping for Conv1d/Conv2d inputs.
    #[serde(default)]
    input_shape: Option<Vec<usize>>,
    /// Additional output-constraint properties (P7). Defaults to empty so legacy
    /// callers (which rely solely on `output_bounds`) are unaffected.
    #[serde(default)]
    output_constraints: Vec<OutputConstraint>,
}

/// Custom deserialization for `VerificationSpec` that validates invariants (#2367, #2778).
///
/// Enforces the same bound contract as `new()` and `from_parts()`:
/// - Input bounds must be finite, non-NaN, and ordered.
/// - Output bounds may be infinite, but must be non-NaN and ordered.
///
/// `Bound`'s custom `Deserialize` already rejects NaN and inverted bounds.
/// This impl additionally validates finiteness of input bounds, non-empty
/// bounds, and `input_shape` consistency.
impl<'de> Deserialize<'de> for VerificationSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct VerificationSpecRaw {
            input_bounds: Vec<Bound>,
            output_bounds: Vec<Bound>,
            timeout_ms: Option<u64>,
            #[serde(default)]
            input_shape: Option<Vec<usize>>,
            #[serde(default)]
            output_constraints: Vec<OutputConstraint>,
        }
        let raw = VerificationSpecRaw::deserialize(deserializer)?;
        // Use the same shared validation as constructors (#2778).
        validate_input_bounds(&raw.input_bounds).map_err(serde::de::Error::custom)?;
        validate_output_bounds(&raw.output_bounds).map_err(serde::de::Error::custom)?;
        validate_output_constraints(&raw.output_constraints).map_err(serde::de::Error::custom)?;
        if let Some(ref shape) = raw.input_shape {
            let shape_product = checked_shape_product(shape).ok_or_else(|| {
                serde::de::Error::custom("VerificationSpec: input_shape product overflows usize")
            })?;
            if shape_product != raw.input_bounds.len() {
                return Err(serde::de::Error::custom(format!(
                    "VerificationSpec: input_shape product {} does not match input_bounds length {}",
                    shape_product,
                    raw.input_bounds.len()
                )));
            }
        }
        Ok(VerificationSpec {
            input_bounds: raw.input_bounds,
            output_bounds: raw.output_bounds,
            timeout_ms: raw.timeout_ms,
            input_shape: raw.input_shape,
            output_constraints: raw.output_constraints,
        })
    }
}

/// Compute the product of shape dimensions with overflow checking.
///
/// Returns `None` on overflow. Matches the pattern used by
/// `SpecBuilder::build` in ny-api.
pub fn checked_shape_product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

/// Compute a dimension product with overflow checking and `InvalidSpec` routing.
///
/// This is the ergonomic companion to [`checked_shape_product`] for call sites
/// that want a `NyError` instead of manual `Option` handling.
pub fn checked_dim_product(dims: &[usize], context: &str) -> crate::Result<usize> {
    checked_shape_product(dims).ok_or_else(|| {
        crate::NyError::InvalidSpec(format!("{context}: dimension product overflows usize"))
    })
}

/// Validate input bounds: must be non-empty, finite, non-NaN, and ordered.
///
/// Input bounds define the verification input domain and must be numerically
/// concrete — infinite input domains are not meaningful for bound propagation.
fn validate_input_bounds(bounds: &[Bound]) -> crate::Result<()> {
    if bounds.is_empty() {
        return Err(crate::NyError::InvalidSpec(
            "input_bounds cannot be empty".to_string(),
        ));
    }
    for (i, bound) in bounds.iter().enumerate() {
        if bound.lower().is_nan() || bound.upper().is_nan() {
            return Err(crate::NyError::InvalidSpec(format!(
                "input_bounds[{i}] contains NaN: [{}, {}]",
                bound.lower(),
                bound.upper()
            )));
        }
        if !bound.lower().is_finite() || !bound.upper().is_finite() {
            return Err(crate::NyError::InvalidSpec(format!(
                "input_bounds[{i}] contains non-finite value: [{}, {}]",
                bound.lower(),
                bound.upper()
            )));
        }
        if bound.lower() > bound.upper() {
            return Err(crate::NyError::InvalidSpec(format!(
                "input_bounds[{i}] is malformed: lower {} > upper {}",
                bound.lower(),
                bound.upper()
            )));
        }
    }
    Ok(())
}

/// Validate output bounds: must be non-empty, non-NaN, and ordered.
///
/// Output bounds may include infinite endpoints (e.g., `[-inf, +inf]` for
/// unconstrained output properties). Only NaN and inverted intervals are
/// rejected.
fn validate_output_bounds(bounds: &[Bound]) -> crate::Result<()> {
    if bounds.is_empty() {
        return Err(crate::NyError::InvalidSpec(
            "output_bounds cannot be empty".to_string(),
        ));
    }
    for (i, bound) in bounds.iter().enumerate() {
        if bound.lower().is_nan() || bound.upper().is_nan() {
            return Err(crate::NyError::InvalidSpec(format!(
                "output_bounds[{i}] contains NaN: [{}, {}]",
                bound.lower(),
                bound.upper()
            )));
        }
        if bound.lower() > bound.upper() {
            return Err(crate::NyError::InvalidSpec(format!(
                "output_bounds[{i}] is malformed: lower {} > upper {}",
                bound.lower(),
                bound.upper()
            )));
        }
    }
    Ok(())
}

/// Validate output constraints: each must be structurally well-formed.
///
/// An empty constraint list is valid (the legacy / default case). Each present
/// constraint is checked via [`OutputConstraint::validate`] (cheap shape checks
/// only — e.g. `Linear.coeffs` non-empty).
fn validate_output_constraints(constraints: &[OutputConstraint]) -> crate::Result<()> {
    for (i, c) in constraints.iter().enumerate() {
        c.validate()
            .map_err(|e| crate::NyError::InvalidSpec(format!("output_constraints[{i}]: {e}")))?;
    }
    Ok(())
}

impl VerificationSpec {
    /// Read-only access to the input bounds.
    #[inline]
    pub fn input_bounds(&self) -> &[Bound] {
        &self.input_bounds
    }

    /// Read-only access to the output bounds.
    #[inline]
    pub fn output_bounds(&self) -> &[Bound] {
        &self.output_bounds
    }

    /// Read-only access to the timeout in milliseconds.
    #[inline]
    pub fn timeout_ms(&self) -> Option<u64> {
        self.timeout_ms
    }

    /// Read-only access to the input shape.
    #[inline]
    pub fn input_shape(&self) -> Option<&[usize]> {
        self.input_shape.as_deref()
    }

    /// Read-only access to the additional output constraints (P7).
    ///
    /// Empty for legacy specs that rely solely on `output_bounds`.
    #[inline]
    pub fn output_constraints(&self) -> &[OutputConstraint] {
        &self.output_constraints
    }

    /// Construct a `VerificationSpec` from all parts.
    ///
    /// Enforces the same bound contract as `new()`:
    /// - Input bounds must be finite, non-NaN, and ordered.
    /// - Output bounds may be infinite, but must be non-NaN and ordered.
    ///
    /// # Errors
    /// - `NyError::InvalidSpec` if `input_bounds` or `output_bounds` is empty
    /// - `NyError::InvalidSpec` if any input bound is non-finite or NaN
    /// - `NyError::InvalidSpec` if any output bound is NaN
    /// - `NyError::InvalidSpec` if any bound has `lower > upper`
    /// - `NyError::InvalidSpec` if `input_shape` product overflows `usize`
    pub fn from_parts(
        input_bounds: Vec<Bound>,
        output_bounds: Vec<Bound>,
        timeout_ms: Option<u64>,
        input_shape: Option<Vec<usize>>,
    ) -> crate::Result<Self> {
        validate_input_bounds(&input_bounds)?;
        validate_output_bounds(&output_bounds)?;
        if let Some(ref shape) = input_shape {
            let shape_product = checked_shape_product(shape).ok_or_else(|| {
                crate::NyError::InvalidSpec("input_shape product overflows usize".to_string())
            })?;
            if shape_product != input_bounds.len() {
                return Err(crate::NyError::InvalidSpec(format!(
                    "input_shape product {} does not match input_bounds length {}",
                    shape_product,
                    input_bounds.len()
                )));
            }
        }
        Ok(Self {
            input_bounds,
            output_bounds,
            timeout_ms,
            input_shape,
            output_constraints: Vec::new(),
        })
    }

    /// Construct a `VerificationSpec` from all parts, including output constraints (P7).
    ///
    /// This is the constraint-aware companion to [`from_parts`](Self::from_parts):
    /// it enforces the identical bound contract *and* validates each
    /// [`OutputConstraint`] via [`OutputConstraint::validate`].
    ///
    /// # Errors
    /// - Everything [`from_parts`](Self::from_parts) can error on, plus
    /// - `NyError::InvalidSpec` if any `output_constraint` is structurally invalid
    ///   (e.g. a `Linear` constraint with empty `coeffs`).
    pub fn from_parts_with_constraints(
        input_bounds: Vec<Bound>,
        output_bounds: Vec<Bound>,
        timeout_ms: Option<u64>,
        input_shape: Option<Vec<usize>>,
        output_constraints: Vec<OutputConstraint>,
    ) -> crate::Result<Self> {
        validate_input_bounds(&input_bounds)?;
        validate_output_bounds(&output_bounds)?;
        validate_output_constraints(&output_constraints)?;
        if let Some(ref shape) = input_shape {
            let shape_product = checked_shape_product(shape).ok_or_else(|| {
                crate::NyError::InvalidSpec("input_shape product overflows usize".to_string())
            })?;
            if shape_product != input_bounds.len() {
                return Err(crate::NyError::InvalidSpec(format!(
                    "input_shape product {} does not match input_bounds length {}",
                    shape_product,
                    input_bounds.len()
                )));
            }
        }
        Ok(Self {
            input_bounds,
            output_bounds,
            timeout_ms,
            input_shape,
            output_constraints,
        })
    }

    /// Create a new verification specification with required fields.
    ///
    /// # Bound Contract
    /// - Input bounds must be finite, non-NaN, and ordered (`lower <= upper`).
    /// - Output bounds may include infinite endpoints (`[-inf, +inf]` for
    ///   unconstrained properties) but must be non-NaN and ordered.
    ///
    /// This is the same contract enforced by `from_parts()` and `Deserialize`.
    ///
    /// # Arguments
    /// * `input_bounds` - Bounds for each input element (row-major order)
    /// * `output_bounds` - Required output bounds for the property
    ///
    /// # Returns
    /// * `Ok(VerificationSpec)` if inputs are valid
    /// * `Err(NyError::InvalidSpec)` if:
    ///   - `input_bounds` is empty
    ///   - `output_bounds` is empty
    ///   - Any input bound is non-finite or contains NaN
    ///   - Any output bound contains NaN
    ///   - Any bound has `lower > upper`
    ///
    /// # Example
    /// ```
    /// use ny_core::{VerificationSpec, Bound};
    /// let spec = VerificationSpec::new(
    ///     vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
    ///     vec![Bound::new(0.0, 1.0)],
    /// ).unwrap();
    /// ```
    pub fn new(input_bounds: Vec<Bound>, output_bounds: Vec<Bound>) -> crate::Result<Self> {
        validate_input_bounds(&input_bounds)?;
        validate_output_bounds(&output_bounds)?;
        Ok(Self {
            input_bounds,
            output_bounds,
            timeout_ms: None,
            input_shape: None,
            output_constraints: Vec::new(),
        })
    }

    /// Set the timeout in milliseconds.
    ///
    /// # Example
    /// ```
    /// use ny_core::{VerificationSpec, Bound};
    /// let spec = VerificationSpec::new(
    ///     vec![Bound::new(-1.0, 1.0)],
    ///     vec![Bound::new(0.0, 1.0)],
    /// ).unwrap().with_timeout_ms(5000);
    /// assert_eq!(spec.timeout_ms(), Some(5000));
    /// ```
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set the input shape for multi-dimensional inputs (e.g., images).
    ///
    /// # Arguments
    /// * `input_shape` - Shape dimensions (e.g., `[1, 28, 28]` for MNIST)
    ///
    /// # Returns
    /// * `Ok(Self)` if shape product equals `input_bounds.len()`
    /// * `Err(NyError::InvalidSpec)` if shape product overflows `usize`
    /// * `Err(NyError::InvalidSpec)` if shape product mismatches input_bounds length
    ///
    /// # Example
    /// ```
    /// use ny_core::{VerificationSpec, Bound};
    /// let spec = VerificationSpec::new(
    ///     vec![Bound::new(0.0, 1.0); 6],  // 6 elements
    ///     vec![Bound::new(0.0, 1.0)],
    /// ).unwrap().with_input_shape(vec![2, 3]).unwrap();  // 2*3 = 6
    /// ```
    pub fn with_input_shape(mut self, input_shape: Vec<usize>) -> crate::Result<Self> {
        let shape_product = checked_shape_product(&input_shape).ok_or_else(|| {
            crate::NyError::InvalidSpec("input_shape product overflows usize".to_string())
        })?;
        if shape_product != self.input_bounds.len() {
            return Err(crate::NyError::InvalidSpec(format!(
                "input_shape product {} does not match input_bounds length {}",
                shape_product,
                self.input_bounds.len()
            )));
        }
        self.input_shape = Some(input_shape);
        Ok(self)
    }

    /// Set the additional output constraints (P7).
    ///
    /// Each constraint is validated via [`OutputConstraint::validate`] (cheap
    /// shape checks). This is additive: a spec without output constraints keeps
    /// the legacy `output_bounds`-only behavior.
    ///
    /// # Errors
    /// * `Err(NyError::InvalidSpec)` if any constraint is structurally invalid
    ///   (e.g. a `Linear` constraint with empty `coeffs`).
    ///
    /// # Example
    /// ```
    /// use ny_core::{VerificationSpec, Bound, OutputConstraint};
    /// let spec = VerificationSpec::new(
    ///     vec![Bound::new(-1.0, 1.0)],
    ///     vec![Bound::new(0.0, 1.0)],
    /// )
    /// .unwrap()
    /// .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
    /// .unwrap();
    /// assert_eq!(spec.output_constraints().len(), 1);
    /// ```
    pub fn with_output_constraints(
        mut self,
        output_constraints: Vec<OutputConstraint>,
    ) -> crate::Result<Self> {
        validate_output_constraints(&output_constraints)?;
        self.output_constraints = output_constraints;
        Ok(self)
    }
}
