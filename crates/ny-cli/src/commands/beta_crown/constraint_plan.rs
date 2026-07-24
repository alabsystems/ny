// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint planning helpers for VNN-LIB specifications.
//!
//! Pure data transforms from [`VnnLibSpec`] to verification plan structures.
//! Used by both `beta_crown verify` and `bench_acasxu` to eliminate semantic
//! drift in constraint classification and objective construction.
//!
//! Part of #1881: CLI verification semantics unification.

use ny_core::NyError;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

/// Whether the unsafe region is a conjunction (AND) or disjunction (OR) of constraints.
///
/// This determines the aggregation strategy for per-constraint verification results:
/// - **Conjunctive**: unsafe if ALL constraints hold. SAFE if ANY single constraint is
///   provably violated (early-exit on first verified constraint).
/// - **Disjunctive**: unsafe if ANY clause holds. SAFE if ALL clauses are provably
///   violated (early-exit when any clause fails verification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggregationMode {
    /// Unsafe region is AND of constraints. SAFE if ANY is violated.
    Conjunctive,
    /// Unsafe region is OR of clauses. SAFE if ALL are violated.
    Disjunctive,
}

/// Classification of a VNN-LIB property's constraint types.
#[derive(Debug, Clone)]
pub(crate) struct ConstraintClassification {
    /// Whether any relational constraints (Y_i op Y_j) are present.
    pub(crate) has_relational: bool,
    /// Aggregation semantics for the property.
    pub(crate) aggregation: AggregationMode,
}

/// Extracted constant constraint parameters for verification.
///
/// Produced by [`extract_constant_params`] from a VNN-LIB spec's constant
/// constraints. Contains all information needed to construct a specification
/// layer or objective vector for constant-threshold verification.
#[derive(Debug, Clone)]
pub(crate) struct ConstantConstraintParams {
    /// Threshold value from the constraint (cast to f32).
    pub(crate) threshold: f32,
    /// Whether to verify upper bound (true) or lower bound (false).
    ///
    /// - `true`: unsafe if Y >= c, so verify upper(Y) < c.
    /// - `false`: unsafe if Y <= c, so verify lower(Y) > c.
    pub(crate) verify_upper: bool,
    /// Output index that the constraint refers to.
    pub(crate) output_idx: usize,
}

/// A single relational objective: the specification coefficient vector encoding
/// a relational constraint as a linear combination over outputs.
///
/// For a constraint like Y_i <= Y_j, the objective encodes the difference
/// Y_i - Y_j and verification checks whether this difference can be bounded
/// away from satisfying the constraint.
#[derive(Debug, Clone)]
pub(crate) struct RelationalObjective {
    /// Specification coefficient vector. Length = num_outputs.
    /// For LessEq(i,j): coeffs[i] = 1.0, coeffs[j] = -1.0 (compute Y_i - Y_j).
    /// For GreaterEq(i,j): coeffs[j] = 1.0, coeffs[i] = -1.0 (compute Y_j - Y_i).
    pub(crate) spec_coeffs: Vec<f32>,
    /// Human-readable constraint description (e.g., "Y_0 <= Y_1").
    pub(crate) constraint_desc: String,
    /// Human-readable difference description (e.g., "Y_0 - Y_1").
    pub(crate) diff_desc: String,
}

/// Unified objective for any constraint type (relational or constant).
///
/// The per-constraint verification loops need to handle both relational and
/// constant constraints uniformly. This enum wraps the objective data for
/// both types so that mixed specs (relational + constant) are fully evaluated
/// without silently dropping constant constraints.
///
/// Fixes #1888: constant constraints were previously dropped when any
/// relational constraint was present in the spec.
#[derive(Debug, Clone)]
pub(crate) enum ConstraintObjective {
    /// Relational constraint: Y_i op Y_j encoded as spec_coeffs, threshold = 0.
    Relational(RelationalObjective),
    /// Constant constraint: Y_i op c encoded as one-hot spec_coeffs + threshold.
    Constant {
        /// Extracted constant constraint parameters.
        params: ConstantConstraintParams,
        /// Specification coefficient vector (one-hot at output_idx, sign-adjusted).
        /// For GreaterEqConst(i, c): coeffs[i] = 1.0 (verify upper < c).
        /// For LessEqConst(i, c): coeffs[i] = -1.0 (verify -lower > -c).
        spec_coeffs: Vec<f32>,
        /// Effective threshold after sign adjustment.
        threshold: f32,
        /// Human-readable constraint description.
        constraint_desc: String,
    },
}

impl ConstraintObjective {
    /// Get the specification coefficient vector for this objective.
    pub(crate) fn spec_coeffs(&self) -> &[f32] {
        match self {
            ConstraintObjective::Relational(obj) => &obj.spec_coeffs,
            ConstraintObjective::Constant { spec_coeffs, .. } => spec_coeffs,
        }
    }

    /// Get the verification threshold for this objective.
    /// Relational constraints use 0.0; constant constraints use the adjusted threshold.
    pub(crate) fn threshold(&self) -> f32 {
        match self {
            ConstraintObjective::Relational(_) => 0.0,
            ConstraintObjective::Constant { threshold, .. } => *threshold,
        }
    }

    /// Get a human-readable description of this constraint.
    pub(crate) fn constraint_desc(&self) -> &str {
        match self {
            ConstraintObjective::Relational(obj) => &obj.constraint_desc,
            ConstraintObjective::Constant {
                constraint_desc, ..
            } => constraint_desc,
        }
    }

    /// Get a human-readable description of what is being verified.
    ///
    /// For constant constraints, the sign matches the violation-semantics objective:
    /// - GreaterEqConst (verify_upper): proving -Y > -c, so "-Y_{idx}"
    /// - LessEqConst (!verify_upper): proving Y > c, so "Y_{idx}"
    pub(crate) fn diff_desc(&self) -> String {
        match self {
            ConstraintObjective::Relational(obj) => obj.diff_desc.clone(),
            ConstraintObjective::Constant { params, .. } => {
                if params.verify_upper {
                    format!("-Y_{}", params.output_idx)
                } else {
                    format!("Y_{}", params.output_idx)
                }
            }
        }
    }
}

/// Classify the constraints in a VNN-LIB spec.
///
/// Returns a [`ConstraintClassification`] describing the constraint types
/// and aggregation mode. This replaces the duplicated `has_relational` match
/// patterns found in `verify.rs`, `inputs.rs`, and `bench_acasxu.rs`.
pub(crate) fn classify_constraints(vnnlib: &VnnLibSpec) -> ConstraintClassification {
    let has_relational = vnnlib.output_constraints.iter().any(|c| c.is_relational());

    let aggregation = if vnnlib.is_disjunction {
        AggregationMode::Disjunctive
    } else {
        AggregationMode::Conjunctive
    };

    ConstraintClassification {
        has_relational,
        aggregation,
    }
}

/// Extract constant constraint parameters from a VNN-LIB spec.
///
/// Scans all constant constraints and returns parameters for the first one found.
/// Returns `None` if no constant constraints exist.
///
/// Unlike the previous `find_map` + `unwrap_or` pattern, this makes the
/// "no constant found" case explicit via `Option`, preventing silent fallback
/// to `(0.0, false, 0)`.
pub(crate) fn extract_constant_params(vnnlib: &VnnLibSpec) -> Option<ConstantConstraintParams> {
    use ny_tensor::{next_down_f32, next_up_f32};
    vnnlib.output_constraints.iter().find_map(|c| match c {
        OutputConstraint::GreaterEqConst(i, val) | OutputConstraint::GreaterThanConst(i, val) => {
            // Property: Y_i >= c (unsafe). Standard verification proves upper(Y_i) < threshold.
            // Round DOWN so threshold <= c → verified upper(Y_i) is still < c. (#3462)
            Some(ConstantConstraintParams {
                threshold: next_down_f32(*val as f32),
                verify_upper: true,
                output_idx: *i,
            })
        }
        OutputConstraint::LessEqConst(i, val) | OutputConstraint::LessThanConst(i, val) => {
            // Property: Y_i <= c (unsafe). Standard verification proves lower(Y_i) > threshold.
            // Round UP so threshold >= c → verified lower(Y_i) is still > c. (#3462)
            Some(ConstantConstraintParams {
                threshold: next_up_f32(*val as f32),
                verify_upper: false,
                output_idx: *i,
            })
        }
        _ => None,
    })
}

/// Extract ALL constant constraint parameters from a VNN-LIB spec.
///
/// Unlike [`extract_constant_params`] which returns only the first constant,
/// this returns parameters for every constant constraint in the spec.
///
/// Fixes #1889: multi-constant specs previously had all but the first constant
/// silently dropped. Production code now routes multi-constant specs through
/// the per-constraint loop via `build_constraint_objective`, but this function
/// is retained for test validation and potential future callers.
#[cfg(test)]
pub(crate) fn extract_all_constant_params(vnnlib: &VnnLibSpec) -> Vec<ConstantConstraintParams> {
    vnnlib
        .output_constraints
        .iter()
        .filter_map(constant_params_from_constraint)
        .collect()
}

/// Convert a single output constraint into its `ConstantConstraintParams`.
///
/// Returns `None` for relational constraints.
///
/// Uses EXACT f64→f32 cast for thresholds (no ULP shift). This function is
/// used by `build_constraint_objective` for per-constraint and multi-objective
/// violation proofs, where the spec coefficients are NEGATED. In the negated
/// context, `next_up_f32` would make verification EASIER (proving Y_0 < c+ε
/// instead of Y_0 < c), which is unsound for violation proofs.
///
/// The standard path (`extract_constant_params`) uses directed rounding aligned
/// with the proof direction: `next_down_f32` when proving `upper(Y) < c`, and
/// `next_up_f32` when proving `lower(Y) > c`.
fn constant_params_from_constraint(
    constraint: &OutputConstraint,
) -> Option<ConstantConstraintParams> {
    match constraint {
        OutputConstraint::GreaterEqConst(i, val) | OutputConstraint::GreaterThanConst(i, val) => {
            Some(ConstantConstraintParams {
                threshold: *val as f32,
                verify_upper: true,
                output_idx: *i,
            })
        }
        OutputConstraint::LessEqConst(i, val) | OutputConstraint::LessThanConst(i, val) => {
            Some(ConstantConstraintParams {
                threshold: *val as f32,
                verify_upper: false,
                output_idx: *i,
            })
        }
        _ => None,
    }
}

/// Build a unified constraint objective for any constraint type.
///
/// Unlike [`build_relational_objective`] which returns `None` for constants,
/// this always returns `Some(ConstraintObjective)` for valid constraints.
/// This ensures constant constraints are never silently skipped in
/// per-constraint verification loops.
///
/// Fixes #1888: mixed relational+constant specs now have all constraints
/// represented in the verification loop.
///
/// # Errors
///
/// Returns `NyError::InvalidSpec` if any output index >= `num_outputs`.
pub(crate) fn build_constraint_objective(
    constraint: &OutputConstraint,
    num_outputs: usize,
) -> ny_core::Result<ConstraintObjective> {
    // Try relational first
    if let Some(obj) = build_relational_objective(constraint, num_outputs)? {
        return Ok(ConstraintObjective::Relational(obj));
    }

    // Must be a constant constraint
    let params = constant_params_from_constraint(constraint).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Constraint is neither relational nor constant: {:?}",
            constraint
        ))
    })?;

    if params.output_idx >= num_outputs {
        return Err(NyError::InvalidSpec(format!(
            "Constant constraint references Y_{} but only {} outputs declared",
            params.output_idx, num_outputs,
        )));
    }

    // Build spec coeffs to prove the constraint is VIOLATED (safety).
    // Convention: verifier proves `spec · Y > threshold` → constraint violated → SAFE.
    //
    // GreaterEqConst(i, c): unsafe if Y_i >= c. Prove Y_i < c → -Y_i > -c.
    //   spec[i] = -1.0, threshold = -c.
    // LessEqConst(i, c): unsafe if Y_i <= c. Prove Y_i > c → Y_i > c.
    //   spec[i] = 1.0, threshold = c.
    //
    // This is the OPPOSITE sign from the standard path (verify_standard), which
    // proves the constraint HOLDS rather than is violated. The per-constraint loop
    // interprets Verified as "constraint violated" for safety aggregation.
    //
    // Reference: relational LessEq(i,j) uses spec[i]=1, spec[j]=-1, threshold=0
    // to prove Y_i - Y_j > 0 → Y_i > Y_j → constraint Y_i <= Y_j violated.
    let mut spec_coeffs = vec![0.0f32; num_outputs];
    spec_coeffs[params.output_idx] = if params.verify_upper { -1.0 } else { 1.0 };

    let threshold = if params.verify_upper {
        -params.threshold
    } else {
        params.threshold
    };

    let constraint_desc = if params.verify_upper {
        format!("Y_{} >= {}", params.output_idx, params.threshold)
    } else {
        format!("Y_{} <= {}", params.output_idx, params.threshold)
    };

    Ok(ConstraintObjective::Constant {
        params,
        spec_coeffs,
        threshold,
        constraint_desc,
    })
}

/// Build a specification coefficient vector for a single relational constraint.
///
/// Returns `Ok(None)` for non-relational constraints (constant comparisons).
/// Returns `Err` if a relational constraint references an output index that
/// is >= `num_outputs`. This is a defense-in-depth check; malformed specs
/// should be caught at parse time by `VnnLibSpec::validate_output_indices`.
///
/// The coefficient vector encodes the constraint as a linear combination of
/// outputs suitable for verification:
/// - `LessEq(i,j)` / `LessThan(i,j)` → coeffs[i]=1.0, coeffs[j]=-1.0, verify upper
/// - `GreaterEq(i,j)` / `GreaterThan(i,j)` → coeffs[j]=1.0, coeffs[i]=-1.0, verify lower
///
/// Reference: alpha-beta-CROWN constructs identical objective vectors for
/// per-property BaB verification in `bab_verification.py`.
///
/// # Errors
///
/// Returns `NyError::InvalidSpec` if any output index >= `num_outputs`. (#1886)
pub(crate) fn build_relational_objective(
    constraint: &OutputConstraint,
    num_outputs: usize,
) -> ny_core::Result<Option<RelationalObjective>> {
    match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            if *i >= num_outputs || *j >= num_outputs {
                return Err(NyError::InvalidSpec(format!(
                    "Relational constraint references Y_{} or Y_{} but only {} outputs declared",
                    i, j, num_outputs,
                )));
            }
            let mut coeffs = vec![0.0f32; num_outputs];
            coeffs[*i] = 1.0;
            coeffs[*j] = -1.0;
            Ok(Some(RelationalObjective {
                spec_coeffs: coeffs,
                constraint_desc: format!("Y_{} <= Y_{}", i, j),
                diff_desc: format!("Y_{} - Y_{}", i, j),
            }))
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            if *i >= num_outputs || *j >= num_outputs {
                return Err(NyError::InvalidSpec(format!(
                    "Relational constraint references Y_{} or Y_{} but only {} outputs declared",
                    i, j, num_outputs,
                )));
            }
            let mut coeffs = vec![0.0f32; num_outputs];
            coeffs[*j] = 1.0;
            coeffs[*i] = -1.0;
            Ok(Some(RelationalObjective {
                spec_coeffs: coeffs,
                constraint_desc: format!("Y_{} >= Y_{}", i, j),
                diff_desc: format!("Y_{} - Y_{}", j, i),
            }))
        }
        _ => Ok(None),
    }
}

/// Build multi-objective coefficient vectors for disjunctive verification.
///
/// Returns a `(objectives, thresholds)` pair suitable for multi-objective BaB.
/// Includes BOTH relational and constant constraints. Relational thresholds
/// are 0.0; constant thresholds are adjusted per `ConstraintObjective::threshold()`.
///
/// This replaces the inline `objectives: Vec<Vec<f32>>` construction in
/// `verify_graph_relational` (verify.rs lines 317-335).
///
/// Fixes #1888: previously filtered out constant constraints via
/// `build_relational_objective` returning None.
///
/// # Errors
///
/// Propagates `NyError::InvalidSpec` from `build_constraint_objective` if
/// any constraint references out-of-range output indices. (#1886)
pub(crate) fn build_multi_objectives(
    vnnlib: &VnnLibSpec,
) -> ny_core::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let mut objectives: Vec<Vec<f32>> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    for c in &vnnlib.output_constraints {
        let obj = build_constraint_objective(c, vnnlib.num_outputs)?;
        objectives.push(obj.spec_coeffs().to_vec());
        thresholds.push(obj.threshold());
    }
    Ok((objectives, thresholds))
}

/// Grouped objective plan for disjunctive (OR-of-AND) multi-clause properties.
///
/// Flattens all clause rows into a single objectives/thresholds surface while
/// preserving clause boundaries in `clause_sizes`. Matches the reference's
/// `or_spec_size` contract (`specifications.py:298-339`).
///
/// Part of #3740 Packet B.
pub(crate) struct GroupedObjectivePlan {
    pub(crate) objectives: Vec<Vec<f32>>,
    pub(crate) thresholds: Vec<f32>,
    pub(crate) clause_sizes: Vec<usize>,
}

/// Build grouped disjunctive objectives preserving clause boundaries.
///
/// Flattens rows from `output_constraint_clauses` clause-by-clause, recording
/// each clause's row count in `clause_sizes`. When `output_constraint_clauses`
/// is empty but `is_disjunction` is true, treats `output_constraints` as a
/// single clause (backward-compatible fallback).
///
/// Uses `build_constraint_objective(...)` for every row so constant-vs-relational
/// semantics stay aligned with the existing CLI planner.
///
/// Part of #3740 Packet B.
pub(crate) fn build_grouped_disjunctive_objectives(
    vnnlib: &VnnLibSpec,
) -> ny_core::Result<GroupedObjectivePlan> {
    let mut objectives: Vec<Vec<f32>> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    let mut clause_sizes: Vec<usize> = Vec::new();

    // Use output_constraint_clauses when available, fallback to a single clause
    // over output_constraints for backward compatibility.
    let fallback;
    let clauses: &[Vec<OutputConstraint>] = if vnnlib.output_constraint_clauses.is_empty() {
        fallback = vec![vnnlib.output_constraints.clone()];
        &fallback
    } else {
        &vnnlib.output_constraint_clauses
    };

    for clause in clauses {
        let clause_start = objectives.len();
        for c in clause {
            let obj = build_constraint_objective(c, vnnlib.num_outputs)?;
            objectives.push(obj.spec_coeffs().to_vec());
            thresholds.push(obj.threshold());
        }
        clause_sizes.push(objectives.len() - clause_start);
    }

    Ok(GroupedObjectivePlan {
        objectives,
        thresholds,
        clause_sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_onnx::vnnlib::VnnLibSpec;

    fn make_spec(constraints: Vec<OutputConstraint>, is_disjunction: bool) -> VnnLibSpec {
        VnnLibSpec {
            num_inputs: 5,
            num_outputs: 5,
            input_bounds: vec![(0.0, 1.0); 5],
            output_constraints: constraints.clone(),
            output_constraint_clauses: if is_disjunction {
                vec![constraints]
            } else {
                Vec::new()
            },
            is_disjunction,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        }
    }

    #[test]
    fn test_classify_relational_conjunctive() {
        let spec = make_spec(
            vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::LessEq(0, 2),
            ],
            false,
        );
        let cls = classify_constraints(&spec);
        assert!(cls.has_relational);
        assert_eq!(cls.aggregation, AggregationMode::Conjunctive);
    }

    #[test]
    fn test_classify_constant_only() {
        let spec = make_spec(vec![OutputConstraint::GreaterEqConst(0, 3.99)], false);
        let cls = classify_constraints(&spec);
        assert!(!cls.has_relational);
        assert_eq!(cls.aggregation, AggregationMode::Conjunctive);
    }

    #[test]
    fn test_classify_disjunctive() {
        let spec = make_spec(vec![OutputConstraint::LessEq(0, 1)], true);
        let cls = classify_constraints(&spec);
        assert_eq!(cls.aggregation, AggregationMode::Disjunctive);
    }

    #[test]
    fn test_extract_constant_params_greater_eq() {
        let spec = make_spec(vec![OutputConstraint::GreaterEqConst(2, 3.99)], false);
        let params = extract_constant_params(&spec).expect("should find constant");
        // Proving upper(Y_2) < c requires rounding the threshold DOWN so the
        // proof target stays at or below the original constant.
        let raw = 3.99f32;
        assert!(
            params.threshold < raw,
            "next_down_f32 must round DOWN for soundness: got {}, expected < {}",
            params.threshold,
            raw
        );
        assert!(
            (params.threshold - raw).abs() < 1e-6,
            "ULP shift should be tiny: delta = {}",
            (params.threshold - raw).abs()
        );
        assert_ne!(
            params.threshold, raw,
            "threshold must differ from raw cast — next_down_f32 must be applied"
        );
        assert!(params.verify_upper);
        assert_eq!(params.output_idx, 2);
    }

    #[test]
    fn test_extract_constant_params_less_eq() {
        let spec = make_spec(vec![OutputConstraint::LessEqConst(1, 5.0)], false);
        let params = extract_constant_params(&spec).expect("should find constant");
        // Proving lower(Y_1) > c requires rounding the threshold UP so the
        // proof target stays at or above the original constant.
        let raw = 5.0f32;
        assert!(
            params.threshold > raw,
            "next_up_f32 must round UP for soundness: got {}, expected > {}",
            params.threshold,
            raw
        );
        assert!(
            (params.threshold - raw).abs() < 1e-6,
            "ULP shift should be tiny: delta = {}",
            (params.threshold - raw).abs()
        );
        assert_ne!(
            params.threshold, raw,
            "threshold must differ from raw cast — next_up_f32 must be applied"
        );
        assert!(!params.verify_upper);
        assert_eq!(params.output_idx, 1);
    }

    #[test]
    fn test_extract_constant_params_greater_eq_rejects_one_ulp_false_safe_gap() {
        use ny_tensor::next_up_f32;

        let spec = make_spec(vec![OutputConstraint::GreaterEqConst(0, 5.0)], false);
        let params = extract_constant_params(&spec).expect("should find constant");
        let boundary_upper = 5.0f32;

        assert!(
            boundary_upper < next_up_f32(5.0),
            "old next_up threshold would have falsely verified upper=5.0"
        );
        assert!(
            boundary_upper >= params.threshold,
            "fixed next_down threshold must keep upper=5.0 unverified"
        );
        assert!(params.verify_upper);
    }

    #[test]
    fn test_extract_constant_params_less_eq_rejects_one_ulp_false_safe_gap() {
        use ny_tensor::next_down_f32;

        let spec = make_spec(vec![OutputConstraint::LessEqConst(0, 5.0)], false);
        let params = extract_constant_params(&spec).expect("should find constant");
        let boundary_lower = 5.0f32;

        assert!(
            boundary_lower > next_down_f32(5.0),
            "old next_down threshold would have falsely verified lower=5.0"
        );
        assert!(
            boundary_lower <= params.threshold,
            "fixed next_up threshold must keep lower=5.0 unverified"
        );
        assert!(!params.verify_upper);
    }

    #[test]
    fn test_extract_constant_params_none_for_relational() {
        let spec = make_spec(vec![OutputConstraint::LessEq(0, 1)], false);
        assert!(extract_constant_params(&spec).is_none());
    }

    #[test]
    fn test_build_relational_objective_less_eq() {
        let constraint = OutputConstraint::LessEq(1, 3);
        let obj = build_relational_objective(&constraint, 5)
            .expect("should not error")
            .expect("should produce objective");
        assert_eq!(obj.spec_coeffs, vec![0.0, 1.0, 0.0, -1.0, 0.0]);
        assert_eq!(obj.constraint_desc, "Y_1 <= Y_3");
    }

    #[test]
    fn test_build_relational_objective_greater_eq() {
        let constraint = OutputConstraint::GreaterEq(2, 4);
        let obj = build_relational_objective(&constraint, 5)
            .expect("should not error")
            .expect("should produce objective");
        assert_eq!(obj.spec_coeffs, vec![0.0, 0.0, -1.0, 0.0, 1.0]);
        assert_eq!(obj.constraint_desc, "Y_2 >= Y_4");
    }

    #[test]
    fn test_build_relational_objective_constant_returns_none() {
        let constraint = OutputConstraint::GreaterEqConst(0, 3.99);
        assert!(build_relational_objective(&constraint, 5)
            .expect("should not error")
            .is_none());
    }

    /// #1886: Out-of-range index in LessEq must return error, not silently mask.
    #[test]
    fn test_build_relational_objective_out_of_range_less_eq_1886() {
        let constraint = OutputConstraint::LessEq(1, 5); // Y_5 with only 5 outputs (0-4)
        let result = build_relational_objective(&constraint, 5);
        assert!(result.is_err(), "out-of-range index should produce error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Y_5") || err_msg.contains("5 outputs"),
            "error should mention the invalid index: {}",
            err_msg
        );
    }

    /// #1886: Out-of-range index in GreaterEq must return error, not silently mask.
    #[test]
    fn test_build_relational_objective_out_of_range_greater_eq_1886() {
        let constraint = OutputConstraint::GreaterEq(10, 0); // Y_10 with only 3 outputs
        let result = build_relational_objective(&constraint, 3);
        assert!(result.is_err(), "out-of-range index should produce error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Y_10")
                || err_msg.contains("3 outputs")
                || err_msg.contains("out of range"),
            "error should mention the invalid index or output count: {}",
            err_msg
        );
    }

    /// #1886: Both indices out of range produces error.
    #[test]
    fn test_build_relational_objective_both_indices_out_of_range_1886() {
        let constraint = OutputConstraint::LessThan(7, 8); // both out of range for 5 outputs
        let result = build_relational_objective(&constraint, 5);
        assert!(
            result.is_err(),
            "both out-of-range indices should produce error"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Y_7") || err_msg.contains("Y_8") || err_msg.contains("5 outputs"),
            "error should mention the invalid indices: {}",
            err_msg
        );
    }

    /// #1888: build_multi_objectives now includes constant constraints.
    #[test]
    fn test_build_multi_objectives_includes_constants_1888() {
        let spec = make_spec(
            vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::GreaterEq(2, 3),
                OutputConstraint::GreaterEqConst(4, 1.0), // now included, not filtered
            ],
            true,
        );
        let (objectives, thresholds) = build_multi_objectives(&spec).expect("should not error");
        assert_eq!(
            objectives.len(),
            3,
            "constant constraint should be included"
        );
        assert_eq!(thresholds.len(), 3);
        assert_eq!(objectives[0], vec![1.0, -1.0, 0.0, 0.0, 0.0]);
        assert_eq!(objectives[1], vec![0.0, 0.0, -1.0, 1.0, 0.0]);
        // GreaterEqConst(4, 1.0): prove violation -Y_4 > -1.0 → coeffs[4]=-1.0, threshold=-1.0
        assert_eq!(objectives[2], vec![0.0, 0.0, 0.0, 0.0, -1.0]);
        assert!((thresholds[0]).abs() < 1e-6, "relational threshold = 0");
        assert!((thresholds[1]).abs() < 1e-6, "relational threshold = 0");
        assert!(
            (thresholds[2] - (-1.0)).abs() < 1e-6,
            "constant threshold = -1.0 (negated for violation semantics)"
        );
    }

    /// #1886: build_multi_objectives propagates error from invalid indices.
    #[test]
    fn test_build_multi_objectives_invalid_index_1886() {
        let spec = make_spec(
            vec![
                OutputConstraint::LessEq(0, 1),  // valid
                OutputConstraint::LessEq(0, 10), // invalid: Y_10 >= num_outputs=5
            ],
            true,
        );
        let result = build_multi_objectives(&spec);
        assert!(result.is_err(), "should propagate error from invalid index");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Y_10")
                || err_msg.contains("5 outputs")
                || err_msg.contains("out of range"),
            "error should mention the invalid index: {}",
            err_msg
        );
    }

    /// #1889: extract_all_constant_params returns all constants, not just first.
    #[test]
    fn test_extract_all_constant_params_multi_1889() {
        let spec = make_spec(
            vec![
                OutputConstraint::GreaterEqConst(0, 3.99),
                OutputConstraint::LessEqConst(1, 5.0),
                OutputConstraint::LessEq(2, 3), // relational, should be skipped
            ],
            false,
        );
        let params = extract_all_constant_params(&spec);
        assert_eq!(params.len(), 2, "should find both constant constraints");
        assert_eq!(params[0].output_idx, 0);
        assert!(params[0].verify_upper);
        assert_eq!(params[1].output_idx, 1);
        assert!(!params[1].verify_upper);
    }

    /// #1889: extract_all_constant_params returns empty for relational-only.
    #[test]
    fn test_extract_all_constant_params_relational_only_1889() {
        let spec = make_spec(
            vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::LessEq(0, 2),
            ],
            false,
        );
        let params = extract_all_constant_params(&spec);
        assert!(params.is_empty());
    }

    /// #1888: build_constraint_objective handles relational.
    #[test]
    fn test_build_constraint_objective_relational_1888() {
        let constraint = OutputConstraint::LessEq(1, 3);
        let obj = build_constraint_objective(&constraint, 5).expect("should not error");
        assert!(matches!(obj, ConstraintObjective::Relational(_)));
        assert_eq!(obj.spec_coeffs(), &[0.0, 1.0, 0.0, -1.0, 0.0]);
        assert!((obj.threshold()).abs() < 1e-6);
    }

    /// #1888: build_constraint_objective handles constant (upper bound).
    /// GreaterEqConst(2, 3.99): unsafe if Y_2 >= 3.99.
    /// Prove violation: -Y_2 > -3.99 → spec[2]=-1.0, threshold=-3.99.
    #[test]
    fn test_build_constraint_objective_constant_upper_1888() {
        let constraint = OutputConstraint::GreaterEqConst(2, 3.99);
        let obj = build_constraint_objective(&constraint, 5).expect("should not error");
        assert!(matches!(obj, ConstraintObjective::Constant { .. }));
        assert_eq!(obj.spec_coeffs(), &[0.0, 0.0, -1.0, 0.0, 0.0]);
        assert!((obj.threshold() - (-3.99)).abs() < 1e-4);
    }

    /// #1888: build_constraint_objective handles constant (lower bound).
    /// LessEqConst(1, 5.0): unsafe if Y_1 <= 5.0.
    /// Prove violation: Y_1 > 5.0 → spec[1]=1.0, threshold=5.0.
    #[test]
    fn test_build_constraint_objective_constant_lower_1888() {
        let constraint = OutputConstraint::LessEqConst(1, 5.0);
        let obj = build_constraint_objective(&constraint, 5).expect("should not error");
        assert!(matches!(obj, ConstraintObjective::Constant { .. }));
        assert_eq!(obj.spec_coeffs(), &[0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!((obj.threshold() - 5.0).abs() < 1e-4);
    }

    /// #1888: build_constraint_objective rejects out-of-range constant index.
    #[test]
    fn test_build_constraint_objective_constant_out_of_range_1888() {
        let constraint = OutputConstraint::GreaterEqConst(5, 1.0);
        let result = build_constraint_objective(&constraint, 5);
        assert!(
            result.is_err(),
            "out-of-range constant index should produce error"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Y_5")
                || err_msg.contains("5 outputs")
                || err_msg.contains("out of range"),
            "error should mention the invalid index: {}",
            err_msg
        );
    }

    /// Verify that `build_multi_objectives` produces EXACT thresholds for constant
    /// constraints (no ULP shift). The multi-objective path proves constraints are
    /// VIOLATED (negated sense), so any ULP shift in the "easier" direction is
    /// unsound — it could cause false Verified results at boundary values.
    ///
    /// The standard path (`extract_constant_params`) shifts thresholds toward the
    /// proof obligation: `next_down_f32` for `upper(Y) < c`, `next_up_f32` for
    /// `lower(Y) > c`.
    ///
    /// Part of #3334: caught by sat_relu SAT instances returning false Verified.
    #[test]
    fn test_build_multi_objectives_exact_thresholds_no_ulp_shift() {
        // Mimics sat_relu property: Y_0 >= 1.0 AND Y_1 <= 0.0
        let spec = make_spec(
            vec![
                OutputConstraint::GreaterEqConst(0, 1.0),
                OutputConstraint::LessEqConst(1, 0.0),
            ],
            false,
        );
        let (objectives, thresholds) =
            build_multi_objectives(&spec).expect("should build objectives");

        assert_eq!(objectives.len(), 2);
        assert_eq!(thresholds.len(), 2);

        // GreaterEqConst(0, 1.0): prove violation -Y_0 > -1.0
        // Threshold must be EXACTLY -1.0, not -1.0000001 (next_up_f32 would be unsound)
        assert_eq!(
            thresholds[0], -1.0f32,
            "threshold for GreaterEqConst must be exact (no ULP shift)"
        );

        // LessEqConst(1, 0.0): prove violation Y_1 > 0.0
        // Threshold must be EXACTLY 0.0, not -0.0 or any shifted value
        assert_eq!(
            thresholds[1], 0.0f32,
            "threshold for LessEqConst must be exact (no ULP shift)"
        );

        // Verify spec coefficients are correct
        // GreaterEqConst(0, 1.0): prove -Y_0 > -1.0 → coeffs[0] = -1.0
        assert_eq!(objectives[0], vec![-1.0, 0.0, 0.0, 0.0, 0.0]);
        // LessEqConst(1, 0.0): prove Y_1 > 0.0 → coeffs[1] = 1.0
        assert_eq!(objectives[1], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    /// Grouped planner preserves clause boundaries as clause_sizes.
    /// Two clauses with different widths (2, 1): clause_sizes = [2, 1].
    /// Part of #3740 Packet B.
    #[test]
    fn test_build_grouped_disjunctive_objectives_preserves_clause_sizes_3740() {
        let clause0 = vec![
            OutputConstraint::LessEq(0, 1),
            OutputConstraint::LessEq(0, 2),
        ];
        let clause1 = vec![OutputConstraint::LessEq(1, 3)];

        let spec = VnnLibSpec {
            num_inputs: 5,
            num_outputs: 5,
            input_bounds: vec![(0.0, 1.0); 5],
            output_constraints: Vec::new(),
            output_constraint_clauses: vec![clause0, clause1],
            is_disjunction: true,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        };

        let plan = build_grouped_disjunctive_objectives(&spec).expect("valid plan");
        assert_eq!(plan.clause_sizes, vec![2, 1], "clause boundaries preserved");
        assert_eq!(plan.objectives.len(), 3, "3 total rows");
        assert_eq!(plan.thresholds.len(), 3, "3 total thresholds");
    }

    /// Grouped planner flattens rows in clause order: clause 0 rows first,
    /// then clause 1 rows, matching the reference's or_spec_size contract.
    /// Part of #3740 Packet B.
    #[test]
    fn test_build_grouped_disjunctive_objectives_flattens_clause_order_3740() {
        // Clause 0: Y_0 <= Y_1 (prove Y_1 - Y_0 > 0)
        // Clause 1: Y_2 <= Y_3 (prove Y_3 - Y_2 > 0)
        let clause0 = vec![OutputConstraint::LessEq(0, 1)];
        let clause1 = vec![OutputConstraint::LessEq(2, 3)];

        let spec = VnnLibSpec {
            num_inputs: 5,
            num_outputs: 5,
            input_bounds: vec![(0.0, 1.0); 5],
            output_constraints: Vec::new(),
            output_constraint_clauses: vec![clause0, clause1],
            is_disjunction: true,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        };

        let plan = build_grouped_disjunctive_objectives(&spec).expect("valid plan");

        // First objective: LessEq(0, 1) = prove Y_0 > Y_1 = prove Y_0 - Y_1 > 0
        // → coeffs[0] = 1, coeffs[1] = -1
        assert_eq!(plan.objectives[0][0], 1.0, "clause 0 row coeffs[0]");
        assert_eq!(plan.objectives[0][1], -1.0, "clause 0 row coeffs[1]");

        // Second objective: LessEq(2, 3) = prove Y_2 > Y_3 = prove Y_2 - Y_3 > 0
        // → coeffs[2] = 1, coeffs[3] = -1
        assert_eq!(plan.objectives[1][2], 1.0, "clause 1 row coeffs[2]");
        assert_eq!(plan.objectives[1][3], -1.0, "clause 1 row coeffs[3]");

        assert_eq!(plan.clause_sizes, vec![1, 1]);
    }

    /// Fallback: when output_constraint_clauses is empty, treat output_constraints
    /// as a single clause for backward compatibility.
    /// Part of #3740 Packet B.
    #[test]
    fn test_build_grouped_disjunctive_objectives_fallback_single_clause_3740() {
        let spec = VnnLibSpec {
            num_inputs: 5,
            num_outputs: 5,
            input_bounds: vec![(0.0, 1.0); 5],
            output_constraints: vec![
                OutputConstraint::LessEq(0, 1),
                OutputConstraint::LessEq(0, 2),
            ],
            output_constraint_clauses: Vec::new(), // empty → fallback
            is_disjunction: true,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        };

        let plan = build_grouped_disjunctive_objectives(&spec).expect("valid plan");
        assert_eq!(
            plan.clause_sizes,
            vec![2],
            "fallback should produce one clause"
        );
        assert_eq!(plan.objectives.len(), 2);
    }
}
