// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP fast-path check for VNNLIB constraint verification.

/// Check if IBP bounds verify all VNNLIB constraints (IBP fast-path). #3218
///
/// For classification properties (disjunctive with relational constraints), this
/// checks if the IBP lower bound of the true label exceeds the IBP upper bound
/// of every other class. Returns true if all constraints are verified by IBP.
pub(super) fn ibp_check_vnnlib_safe(
    ibp_lower: &[f32],
    ibp_upper: &[f32],
    vnnlib: &ny_onnx::vnnlib::VnnLibSpec,
) -> bool {
    // Check per-clause constraints.
    // For disjunctive unsafe region (OR of clauses): SAFE iff ALL clauses are violated.
    // For conjunctive unsafe region (AND of clauses): SAFE iff ANY clause is violated.
    let clauses = &vnnlib.output_constraint_clauses;
    if clauses.is_empty() {
        return false;
    }

    let clause_results: Vec<bool> = clauses
        .iter()
        .map(|clause| {
            // A clause is violated (safe) if ANY constraint in the clause is impossible.
            clause.iter().any(|c| {
                use ny_onnx::vnnlib::OutputConstraint;
                match c {
                    // UNSAFE: Y_i <= Y_j → violated when lower[i] > upper[j]
                    OutputConstraint::LessEq(i, j) => {
                        *i < ibp_lower.len()
                            && *j < ibp_upper.len()
                            && ibp_lower[*i] > ibp_upper[*j]
                    }
                    // UNSAFE: Y_i < Y_j → violated when lower[i] >= upper[j]
                    OutputConstraint::LessThan(i, j) => {
                        *i < ibp_lower.len()
                            && *j < ibp_upper.len()
                            && ibp_lower[*i] >= ibp_upper[*j]
                    }
                    // UNSAFE: Y_i >= Y_j → violated when upper[i] < lower[j]
                    OutputConstraint::GreaterEq(i, j) => {
                        *i < ibp_upper.len()
                            && *j < ibp_lower.len()
                            && ibp_upper[*i] < ibp_lower[*j]
                    }
                    // UNSAFE: Y_i > Y_j → violated when upper[i] <= lower[j]
                    OutputConstraint::GreaterThan(i, j) => {
                        *i < ibp_upper.len()
                            && *j < ibp_lower.len()
                            && ibp_upper[*i] <= ibp_lower[*j]
                    }
                    // UNSAFE: Y_i <= c → violated when lower[i] > c
                    // Directed rounding: round c UP so refutation is conservative.
                    // Matches disjunctive_precheck.rs directed rounding convention.
                    OutputConstraint::LessEqConst(i, c) => {
                        let c_f32 = ny_tensor::next_up_f32(*c as f32);
                        *i < ibp_lower.len() && ibp_lower[*i] > c_f32
                    }
                    // UNSAFE: Y_i < c → violated when lower[i] >= c
                    OutputConstraint::LessThanConst(i, c) => {
                        let c_f32 = ny_tensor::next_up_f32(*c as f32);
                        *i < ibp_lower.len() && ibp_lower[*i] >= c_f32
                    }
                    // UNSAFE: Y_i >= c → violated when upper[i] < c
                    // Directed rounding: round c DOWN so refutation is conservative.
                    OutputConstraint::GreaterEqConst(i, c) => {
                        let c_f32 = ny_tensor::next_down_f32(*c as f32);
                        *i < ibp_upper.len() && ibp_upper[*i] < c_f32
                    }
                    // UNSAFE: Y_i > c → violated when upper[i] <= c
                    OutputConstraint::GreaterThanConst(i, c) => {
                        let c_f32 = ny_tensor::next_down_f32(*c as f32);
                        *i < ibp_upper.len() && ibp_upper[*i] <= c_f32
                    }
                    _ => false, // unknown constraint variants cannot be proven violated
                }
            })
        })
        .collect();

    if vnnlib.is_disjunction {
        // Disjunctive unsafe: SAFE iff ALL clauses violated
        clause_results.iter().all(|&violated| violated)
    } else {
        // Conjunctive unsafe: SAFE iff ANY clause violated
        clause_results.iter().any(|&violated| violated)
    }
}
