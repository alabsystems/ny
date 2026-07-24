// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Output-constraint types (P7).
//!
//! These types describe the property to prove about a network's *outputs*. They
//! generalize the legacy per-output interval bounds (`VerificationSpec::output_bounds`)
//! to richer halfspace and robustness (argmax-margin) constraints, while remaining
//! fully additive: a `VerificationSpec` defaults to an empty constraint list, so
//! existing callers are unaffected.

use serde::{Deserialize, Serialize};

use crate::Bound;

/// Direction of a linear (halfspace) output constraint.
///
/// `Le` encodes `a·y <= b`; `Ge` encodes `a·y >= b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// `sum_i coeffs[i]*y[i] <= bias`.
    Le,
    /// `sum_i coeffs[i]*y[i] >= bias`.
    Ge,
}

/// A constraint on a network's output vector `y`.
///
/// Variants are serde-tagged in `snake_case`. `Bounds` is exactly equivalent to
/// the legacy [`VerificationSpec::output_bounds`](crate::VerificationSpec::output_bounds)
/// behavior and is provided so that the legacy property can be expressed uniformly
/// alongside richer constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputConstraint {
    /// Per-output interval bounds (legacy behavior; equivalent to `VerificationSpec.output_bounds`).
    Bounds(Vec<Bound>),
    /// Halfspace constraint: `sum_i coeffs[i]*y[i]  {<=|>=}  bias`.
    Linear {
        /// Per-output coefficients of the linear form.
        coeffs: Vec<f32>,
        /// Right-hand-side constant of the halfspace.
        bias: f32,
        /// Whether the halfspace is `<=` (`Le`) or `>=` (`Ge`).
        kind: ConstraintKind,
    },
    /// Robustness: `argmax(y)` must equal `class`
    /// (i.e. `y[class] - y[j] > 0` for all `j != class`).
    ArgmaxMargin {
        /// The output index that must remain the strict argmax.
        class: usize,
    },
}

impl OutputConstraint {
    /// Validate the structural shape of this constraint.
    ///
    /// Cheap, value-independent checks only:
    /// - `Bounds` must be non-empty.
    /// - `Linear` must have non-empty `coeffs` and a non-NaN, finite `bias`.
    /// - `ArgmaxMargin` is always structurally valid (any `class` index is allowed;
    ///   range validity against an output dimension is checked at use sites).
    ///
    /// # Errors
    /// Returns [`NyError::InvalidSpec`](crate::NyError::InvalidSpec) if the shape is invalid.
    pub fn validate(&self) -> crate::Result<()> {
        match self {
            OutputConstraint::Bounds(bounds) => {
                if bounds.is_empty() {
                    return Err(crate::NyError::InvalidSpec(
                        "OutputConstraint::Bounds cannot be empty".to_string(),
                    ));
                }
                Ok(())
            }
            OutputConstraint::Linear { coeffs, bias, .. } => {
                if coeffs.is_empty() {
                    return Err(crate::NyError::InvalidSpec(
                        "OutputConstraint::Linear coeffs cannot be empty".to_string(),
                    ));
                }
                if bias.is_nan() {
                    return Err(crate::NyError::InvalidSpec(
                        "OutputConstraint::Linear bias is NaN".to_string(),
                    ));
                }
                if !bias.is_finite() {
                    return Err(crate::NyError::InvalidSpec(format!(
                        "OutputConstraint::Linear bias is non-finite: {bias}"
                    )));
                }
                if coeffs.iter().any(|c| c.is_nan()) {
                    return Err(crate::NyError::InvalidSpec(
                        "OutputConstraint::Linear coeffs contain NaN".to_string(),
                    ));
                }
                Ok(())
            }
            OutputConstraint::ArgmaxMargin { .. } => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bound, VerificationSpec};

    fn sample_bounds() -> Vec<Bound> {
        vec![Bound::new(0.0, 1.0), Bound::new(-1.0, 1.0)]
    }

    #[test]
    fn construct_each_variant() {
        let b = OutputConstraint::Bounds(sample_bounds());
        let l = OutputConstraint::Linear {
            coeffs: vec![1.0, -1.0],
            bias: 0.5,
            kind: ConstraintKind::Le,
        };
        let a = OutputConstraint::ArgmaxMargin { class: 3 };

        // Pattern-match to confirm the shapes are what we expect.
        match b {
            OutputConstraint::Bounds(ref v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Bounds"),
        }
        match l {
            OutputConstraint::Linear {
                ref coeffs,
                bias,
                kind,
            } => {
                assert_eq!(coeffs, &[1.0, -1.0]);
                assert_eq!(bias, 0.5);
                assert_eq!(kind, ConstraintKind::Le);
            }
            _ => panic!("expected Linear"),
        }
        match a {
            OutputConstraint::ArgmaxMargin { class } => assert_eq!(class, 3),
            _ => panic!("expected ArgmaxMargin"),
        }
    }

    #[test]
    fn constraint_kind_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&ConstraintKind::Le).unwrap(),
            "\"le\""
        );
        assert_eq!(
            serde_json::to_string(&ConstraintKind::Ge).unwrap(),
            "\"ge\""
        );
        let k: ConstraintKind = serde_json::from_str("\"ge\"").unwrap();
        assert_eq!(k, ConstraintKind::Ge);
    }

    #[test]
    fn output_constraint_serde_round_trip_snake_case() {
        let cases = vec![
            OutputConstraint::Bounds(sample_bounds()),
            OutputConstraint::Linear {
                coeffs: vec![2.0, 0.0, -3.5],
                bias: -1.25,
                kind: ConstraintKind::Ge,
            },
            OutputConstraint::ArgmaxMargin { class: 7 },
        ];
        for c in cases {
            let json = serde_json::to_string(&c).unwrap();
            let back: OutputConstraint = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back, "round trip mismatch for {json}");
        }
    }

    #[test]
    fn output_constraint_serde_uses_snake_case_tags() {
        let json = serde_json::to_string(&OutputConstraint::ArgmaxMargin { class: 1 }).unwrap();
        assert!(json.contains("argmax_margin"), "tag not snake_case: {json}");

        let json = serde_json::to_string(&OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 0.0,
            kind: ConstraintKind::Le,
        })
        .unwrap();
        assert!(json.contains("linear"), "tag not snake_case: {json}");
        assert!(json.contains("\"le\""), "kind not snake_case: {json}");
    }

    #[test]
    fn validate_accepts_well_formed() {
        OutputConstraint::Bounds(sample_bounds())
            .validate()
            .unwrap();
        OutputConstraint::Linear {
            coeffs: vec![1.0, 2.0],
            bias: 3.0,
            kind: ConstraintKind::Ge,
        }
        .validate()
        .unwrap();
        OutputConstraint::ArgmaxMargin { class: 0 }
            .validate()
            .unwrap();
    }

    #[test]
    fn validate_rejects_empty_linear_coeffs() {
        let c = OutputConstraint::Linear {
            coeffs: vec![],
            bias: 0.0,
            kind: ConstraintKind::Le,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_bounds() {
        let c = OutputConstraint::Bounds(vec![]);
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_linear_bias() {
        let c = OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: f32::INFINITY,
            kind: ConstraintKind::Le,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn verification_spec_defaults_to_empty_constraints() {
        // Back-compat: from_parts and new yield empty constraints.
        let spec = VerificationSpec::from_parts(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new(0.0, 1.0)],
            None,
            None,
        )
        .unwrap();
        assert!(spec.output_constraints().is_empty());

        let spec2 =
            VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(0.0, 1.0)]).unwrap();
        assert!(spec2.output_constraints().is_empty());
    }

    #[test]
    fn verification_spec_carries_constraints() {
        let constraints = vec![
            OutputConstraint::ArgmaxMargin { class: 2 },
            OutputConstraint::Linear {
                coeffs: vec![1.0, -1.0],
                bias: 0.0,
                kind: ConstraintKind::Ge,
            },
        ];
        let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(0.0, 1.0)])
            .unwrap()
            .with_output_constraints(constraints.clone())
            .unwrap();
        assert_eq!(spec.output_constraints(), constraints.as_slice());
    }

    #[test]
    fn verification_spec_from_parts_with_constraints() {
        let constraints = vec![OutputConstraint::ArgmaxMargin { class: 0 }];
        let spec = VerificationSpec::from_parts_with_constraints(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new(0.0, 1.0)],
            None,
            None,
            constraints.clone(),
        )
        .unwrap();
        assert_eq!(spec.output_constraints(), constraints.as_slice());
    }

    #[test]
    fn with_output_constraints_validates_shapes() {
        let bad = vec![OutputConstraint::Linear {
            coeffs: vec![],
            bias: 0.0,
            kind: ConstraintKind::Le,
        }];
        let res = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(0.0, 1.0)])
            .unwrap()
            .with_output_constraints(bad);
        assert!(res.is_err());
    }
}
