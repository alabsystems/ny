// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification annotation helpers for external integrations.
//!
//! This module provides a thin wrapper around `ny_core::VerificationSpec`
//! to attach metadata (name/tags) without altering verifier behavior.
//! Only per-output bounds are enforced today; linear constraints are a
//! future extension.

use crate::materialize::VerificationBoundsSource;
use ny_core::{Bound, ConstraintKind, NyError, OutputConstraint, Result, VerificationSpec};

/// Output constraints for verification annotations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationConstraint {
    /// Per-output interval bounds (current verification behavior).
    Bounds(Vec<Bound>),
    /// Halfspace constraint: `sum_i coeffs[i]*y[i] {<=|>=} bias`.
    ///
    /// Recorded as an [`OutputConstraint::Linear`] carried into the built
    /// [`VerificationSpec`]; it does not overwrite `output_bounds`.
    Linear {
        /// Per-output coefficients of the linear form.
        coeffs: Vec<f32>,
        /// Right-hand-side constant of the halfspace.
        bias: f32,
        /// Whether the halfspace is `<=` (`Le`) or `>=` (`Ge`).
        kind: ConstraintKind,
    },
    /// Robustness: `argmax(y)` must equal `class`.
    ///
    /// Recorded as an [`OutputConstraint::ArgmaxMargin`] carried into the built
    /// [`VerificationSpec`]; it does not overwrite `output_bounds`.
    ArgmaxMargin {
        /// The output index that must remain the strict argmax.
        class: usize,
    },
}

/// Annotated verification specification with optional metadata.
#[derive(Debug, Clone)]
pub struct AnnotatedSpec {
    /// Optional property name.
    pub name: Option<String>,
    /// Tags for grouping/analytics.
    pub tags: Vec<String>,
    /// Underlying verification spec consumed by the verifier.
    pub spec: VerificationSpec,
}

impl AnnotatedSpec {
    /// Construct an annotated spec with no metadata.
    pub fn new(spec: VerificationSpec) -> Self {
        Self {
            name: None,
            tags: Vec::new(),
            spec,
        }
    }

    /// Drop metadata and return the underlying spec.
    pub fn into_spec(self) -> VerificationSpec {
        self.spec
    }
}

/// Builder for `VerificationSpec` that validates bounds and shapes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecBuilder {
    input_bounds: Vec<Bound>,
    input_shape: Option<Vec<usize>>,
    output_bounds: Vec<Bound>,
    timeout_ms: Option<u64>,
    output_constraints: Vec<OutputConstraint>,
}

impl SpecBuilder {
    /// Set input bounds and optional input shape.
    pub fn input_bounds(mut self, bounds: Vec<Bound>, shape: Option<Vec<usize>>) -> Self {
        self.input_bounds = bounds;
        self.input_shape = shape;
        self
    }

    /// Materialize input bounds directly from a public bounds source.
    pub fn try_input_source<T: VerificationBoundsSource + ?Sized>(
        mut self,
        source: &T,
    ) -> Result<Self> {
        let tensor = source.materialize_bounds()?;
        let shape = tensor.shape().to_vec();
        let bounds = tensor
            .lower()
            .iter()
            .copied()
            .zip(tensor.upper().iter().copied())
            .map(|(lower, upper)| Bound::try_new(lower, upper))
            .collect::<Result<Vec<_>>>()?;
        self.input_bounds = bounds;
        self.input_shape = Some(shape);
        Ok(self)
    }

    /// Set output bounds.
    pub fn output_bounds(mut self, bounds: Vec<Bound>) -> Self {
        self.output_bounds = bounds;
        self
    }

    /// Set output bounds from an output constraint.
    ///
    /// `Bounds` keeps the legacy behavior of setting `output_bounds`. `Linear`
    /// and `ArgmaxMargin` are recorded as [`OutputConstraint`]s carried into the
    /// built [`VerificationSpec`] via [`output_constraints`](Self::output_constraints)
    /// without overwriting `output_bounds`.
    pub fn output_constraint(mut self, constraint: AnnotationConstraint) -> Self {
        match constraint {
            AnnotationConstraint::Bounds(bounds) => {
                self.output_bounds = bounds;
            }
            AnnotationConstraint::Linear { coeffs, bias, kind } => {
                self.output_constraints
                    .push(OutputConstraint::Linear { coeffs, bias, kind });
            }
            AnnotationConstraint::ArgmaxMargin { class } => {
                self.output_constraints
                    .push(OutputConstraint::ArgmaxMargin { class });
            }
        }
        self
    }

    /// Append additional [`OutputConstraint`]s to be carried into the built spec.
    ///
    /// Additive: this does not modify `output_bounds`. Constraints are validated
    /// at [`build`](Self::build) time via
    /// [`VerificationSpec::from_parts_with_constraints`].
    pub fn output_constraints(mut self, constraints: Vec<OutputConstraint>) -> Self {
        self.output_constraints.extend(constraints);
        self
    }

    /// Set timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout_ms: Option<u64>) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Validate and build a `VerificationSpec`.
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if input/output bounds are empty or the
    /// provided input shape does not match the input bounds length.
    pub fn build(self) -> Result<VerificationSpec> {
        if self.input_bounds.is_empty() {
            return Err(NyError::InvalidSpec(
                "SpecBuilder: input bounds must be non-empty".to_string(),
            ));
        }
        if self.output_bounds.is_empty() {
            return Err(NyError::InvalidSpec(
                "SpecBuilder: output bounds must be non-empty".to_string(),
            ));
        }
        if let Some(shape) = &self.input_shape {
            let total = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    NyError::InvalidSpec(
                        "SpecBuilder: input shape product overflowed usize".to_string(),
                    )
                })?;
            if total != self.input_bounds.len() {
                return Err(NyError::InvalidSpec(format!(
                    "SpecBuilder: input shape {:?} has {} elements but bounds has {}",
                    shape,
                    total,
                    self.input_bounds.len()
                )));
            }
        }

        VerificationSpec::from_parts_with_constraints(
            self.input_bounds,
            self.output_bounds,
            self.timeout_ms,
            self.input_shape,
            self.output_constraints,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{AnnotatedSpec, AnnotationConstraint, SpecBuilder};
    use ndarray::arr1;
    use ny_core::{Bound, ConstraintKind, NyError, OutputConstraint, VerificationSpec};
    use ny_tensor::BoundedTensor;

    fn simple_bounds(count: usize) -> Vec<Bound> {
        (0..count)
            .map(|idx| Bound::new(idx as f32, (idx + 1) as f32))
            .collect()
    }

    #[test]
    fn annotated_spec_round_trip() {
        let spec = VerificationSpec::from_parts(
            simple_bounds(2),
            simple_bounds(1),
            Some(250),
            Some(vec![2]),
        )
        .expect("valid test spec");
        let annotated = AnnotatedSpec::new(spec.clone());
        assert!(annotated.name.is_none());
        assert!(annotated.tags.is_empty());
        assert_eq!(annotated.spec.input_bounds(), spec.input_bounds());
        assert_eq!(annotated.spec.output_bounds(), spec.output_bounds());
        assert_eq!(annotated.spec.timeout_ms(), spec.timeout_ms());
        assert_eq!(annotated.spec.input_shape(), spec.input_shape());

        let round_trip = annotated.into_spec();
        assert_eq!(round_trip.input_bounds(), spec.input_bounds());
        assert_eq!(round_trip.output_bounds(), spec.output_bounds());
        assert_eq!(round_trip.timeout_ms(), spec.timeout_ms());
        assert_eq!(round_trip.input_shape(), spec.input_shape());
    }

    #[test]
    fn builder_rejects_empty_input_bounds() {
        let err = SpecBuilder::default()
            .output_bounds(simple_bounds(1))
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("input bounds")));
    }

    #[test]
    fn builder_rejects_empty_output_bounds() {
        let err = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("output bounds")));
    }

    #[test]
    fn builder_rejects_mismatched_input_shape() {
        let err = SpecBuilder::default()
            .input_bounds(simple_bounds(2), Some(vec![3]))
            .output_bounds(simple_bounds(1))
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("input shape")));
    }

    #[test]
    fn builder_rejects_shape_overflow() {
        let err = SpecBuilder::default()
            .input_bounds(simple_bounds(1), Some(vec![usize::MAX, 2]))
            .output_bounds(simple_bounds(1))
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("overflowed")));
    }

    #[test]
    fn builder_accepts_valid_shape_and_timeout() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(4), Some(vec![2, 2]))
            .output_bounds(simple_bounds(1))
            .timeout_ms(Some(900))
            .build()
            .expect("valid spec");

        assert_eq!(spec.input_bounds().len(), 4);
        assert_eq!(spec.input_shape(), Some(&[2, 2][..]));
        assert_eq!(spec.timeout_ms(), Some(900));
    }

    #[test]
    fn builder_output_constraint_sets_output_bounds() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .output_constraint(AnnotationConstraint::Bounds(simple_bounds(2)))
            .build()
            .expect("valid spec");
        assert_eq!(spec.output_bounds().len(), 2);
    }

    #[test]
    fn builder_linear_constraint_records_output_constraint() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .output_bounds(simple_bounds(1))
            .output_constraint(AnnotationConstraint::Linear {
                coeffs: vec![1.0, -1.0],
                bias: 0.5,
                kind: ConstraintKind::Le,
            })
            .build()
            .expect("valid spec");
        // output_bounds is preserved (not overwritten by the linear constraint).
        assert_eq!(spec.output_bounds().len(), 1);
        assert_eq!(
            spec.output_constraints(),
            &[OutputConstraint::Linear {
                coeffs: vec![1.0, -1.0],
                bias: 0.5,
                kind: ConstraintKind::Le,
            }]
        );
    }

    #[test]
    fn builder_argmax_margin_constraint_records_output_constraint() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(2), None)
            .output_bounds(simple_bounds(3))
            .output_constraint(AnnotationConstraint::ArgmaxMargin { class: 2 })
            .build()
            .expect("valid spec");
        assert_eq!(spec.output_bounds().len(), 3);
        assert_eq!(
            spec.output_constraints(),
            &[OutputConstraint::ArgmaxMargin { class: 2 }]
        );
    }

    #[test]
    fn builder_accumulates_multiple_constraints() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .output_bounds(simple_bounds(2))
            .output_constraint(AnnotationConstraint::ArgmaxMargin { class: 0 })
            .output_constraint(AnnotationConstraint::Linear {
                coeffs: vec![1.0, 1.0],
                bias: 2.0,
                kind: ConstraintKind::Ge,
            })
            .build()
            .expect("valid spec");
        assert_eq!(spec.output_constraints().len(), 2);
    }

    #[test]
    fn builder_output_constraints_method_appends() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .output_bounds(simple_bounds(2))
            .output_constraints(vec![
                OutputConstraint::ArgmaxMargin { class: 1 },
                OutputConstraint::ArgmaxMargin { class: 0 },
            ])
            .build()
            .expect("valid spec");
        assert_eq!(spec.output_constraints().len(), 2);
    }

    #[test]
    fn builder_rejects_structurally_invalid_constraint() {
        // Linear with empty coeffs is rejected by from_parts_with_constraints.
        let err = SpecBuilder::default()
            .input_bounds(simple_bounds(1), None)
            .output_bounds(simple_bounds(1))
            .output_constraint(AnnotationConstraint::Linear {
                coeffs: vec![],
                bias: 0.0,
                kind: ConstraintKind::Le,
            })
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn builder_build_accepts_valid_spec() {
        let spec = SpecBuilder::default()
            .input_bounds(simple_bounds(3), Some(vec![3]))
            .output_bounds(simple_bounds(1))
            .build()
            .expect("valid spec");
        assert_eq!(spec.input_bounds().len(), 3);
        assert_eq!(spec.output_bounds().len(), 1);
    }

    #[test]
    fn builder_rejects_invalid_inputs_via_build() {
        let err = SpecBuilder::default()
            .output_bounds(simple_bounds(1))
            .build()
            .unwrap_err();
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("input bounds")));
    }

    #[test]
    fn builder_accepts_input_source_and_copies_shape() {
        let source = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.5_f32]).into_dyn(),
            arr1(&[2.0_f32, 1.5_f32]).into_dyn(),
        )
        .expect("valid input source bounds");

        let spec = SpecBuilder::default()
            .try_input_source(&source)
            .expect("bounded tensor should materialize into input bounds")
            .output_bounds(simple_bounds(1))
            .build()
            .expect("source-backed spec should build");

        assert_eq!(spec.input_shape(), Some(&[2][..]));
        assert_eq!(
            spec.input_bounds(),
            &[Bound::new(-1.0, 2.0), Bound::new(0.5, 1.5)]
        );
    }

    #[test]
    fn builder_rejects_non_finite_input_source() {
        let source = BoundedTensor::new_allow_infinite(
            arr1(&[0.0_f32]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
        )
        .expect("infinite bounds are allowed at the tensor boundary");

        let err = SpecBuilder::default()
            .try_input_source(&source)
            .expect_err("spec input builder should fail closed on non-finite bounds");

        assert!(
            matches!(err, NyError::NumericalInstability(message) if message.contains("not finite"))
        );
    }
}
