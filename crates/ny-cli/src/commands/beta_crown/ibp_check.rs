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
    // This helper is verdict-authoritative: malformed enclosures must never be
    // interpreted as refuting an unsafe clause. In particular, IEEE infinities
    // and inverted intervals can make the raw comparisons below true.
    if ibp_lower.is_empty()
        || ibp_lower.len() != ibp_upper.len()
        || ibp_lower.len() != vnnlib.num_outputs
        || ibp_lower
            .iter()
            .zip(ibp_upper)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return false;
    }

    // Check per-clause constraints.
    // For disjunctive unsafe region (OR of clauses): SAFE iff ALL clauses are violated.
    // For conjunctive unsafe region (AND of clauses): SAFE iff ANY clause is violated.
    let clauses = &vnnlib.output_constraint_clauses;
    if clauses.is_empty() {
        return false;
    }

    use ny_onnx::vnnlib::OutputConstraint;
    if clauses.iter().flatten().any(|constraint| match constraint {
        OutputConstraint::LessEqConst(_, c)
        | OutputConstraint::LessThanConst(_, c)
        | OutputConstraint::GreaterEqConst(_, c)
        | OutputConstraint::GreaterThanConst(_, c) => !c.is_finite(),
        _ => false,
    }) {
        return false;
    }

    let clause_results: Vec<bool> = clauses
        .iter()
        .map(|clause| {
            // A clause is violated (safe) if ANY constraint in the clause is impossible.
            clause.iter().any(|c| {
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
                    // f32 endpoints embed exactly in f64, so compare against
                    // the exact VNNLIB threshold without a lossy f32 cast.
                    OutputConstraint::LessEqConst(i, c) => {
                        *i < ibp_lower.len() && f64::from(ibp_lower[*i]) > *c
                    }
                    // UNSAFE: Y_i < c → violated when lower[i] >= c
                    OutputConstraint::LessThanConst(i, c) => {
                        *i < ibp_lower.len() && f64::from(ibp_lower[*i]) >= *c
                    }
                    // UNSAFE: Y_i >= c → violated when upper[i] < c
                    OutputConstraint::GreaterEqConst(i, c) => {
                        *i < ibp_upper.len() && f64::from(ibp_upper[*i]) < *c
                    }
                    // UNSAFE: Y_i > c → violated when upper[i] <= c
                    OutputConstraint::GreaterThanConst(i, c) => {
                        *i < ibp_upper.len() && f64::from(ibp_upper[*i]) <= *c
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
