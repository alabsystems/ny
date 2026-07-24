// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Normal form conversion (DNF/CNF) and output constraint generation.

use super::parse::{parse_bool_expr, parse_declarations};
use super::types::{
    BoolExpr, LinearConstraint, NormalizeOptions, NormalizedConstraints, Relation, VarKind,
    COEFF_EPS,
};
use crate::vnnlib::syntax::{parse_expressions, strip_vnnlib_comments, tokenize, Expr};
use ny_core::{NyError, Result};
use std::collections::BTreeMap;

fn combine_clauses<T: Clone>(
    left: &[Vec<T>],
    right: &[Vec<T>],
    options: &NormalizeOptions,
) -> Result<Vec<Vec<T>>> {
    let mut combined = Vec::new();
    for l in left {
        for r in right {
            let mut clause = Vec::with_capacity(l.len() + r.len());
            clause.extend_from_slice(l);
            clause.extend_from_slice(r);
            if clause.len() > options.max_clause_len {
                return Err(NyError::InvalidSpec(format!(
                    "Clause length {} exceeds max_clause_len {}",
                    clause.len(),
                    options.max_clause_len
                )));
            }
            combined.push(clause);
            if combined.len() > options.max_clauses {
                return Err(NyError::InvalidSpec(format!(
                    "DNF/CNF clause count {} exceeds max_clauses {}",
                    combined.len(),
                    options.max_clauses
                )));
            }
        }
    }
    Ok(combined)
}

pub fn to_dnf<T: Clone>(expr: &BoolExpr<T>, options: &NormalizeOptions) -> Result<Vec<Vec<T>>> {
    match expr {
        BoolExpr::Atom(atom) => Ok(vec![vec![atom.clone()]]),
        BoolExpr::Or(children) => {
            let mut clauses = Vec::new();
            for child in children {
                let child_clauses = to_dnf(child, options)?;
                clauses.extend(child_clauses);
                if clauses.len() > options.max_clauses {
                    return Err(NyError::InvalidSpec(format!(
                        "DNF clause count {} exceeds max_clauses {}",
                        clauses.len(),
                        options.max_clauses
                    )));
                }
            }
            Ok(clauses)
        }
        BoolExpr::And(children) => {
            let mut clauses = vec![Vec::new()];
            for child in children {
                // Atom fast path (#vnnlib-parse-dnf): appending the literal to
                // every accumulated clause IN PLACE is value-identical to
                // `combine_clauses(&clauses, &[vec![atom]])` — same clause
                // count, same literal order (accumulated prefix ++ atom), same
                // `max_clause_len` check — but skips re-cloning the whole
                // accumulated prefix per literal. The nn4sys mscn properties
                // are thousands of `(and <hundreds of atoms>)` disjuncts, and
                // the quadratic re-clone made parsing a 34MB property ~18s
                // (profiled: ~all malloc/free of cloned LinearExpr BTreeMaps);
                // this path parses it in ~1s. Bit-identity vs the general
                // path is asserted by `dnf_atom_fast_path_matches_reference`.
                if let BoolExpr::Atom(atom) = child {
                    for clause in clauses.iter_mut() {
                        clause.push(atom.clone());
                        if clause.len() > options.max_clause_len {
                            return Err(NyError::InvalidSpec(format!(
                                "Clause length {} exceeds max_clause_len {}",
                                clause.len(),
                                options.max_clause_len
                            )));
                        }
                    }
                    continue;
                }
                let child_clauses = to_dnf(child, options)?;
                clauses = combine_clauses(&clauses, &child_clauses, options)?;
            }
            Ok(clauses)
        }
    }
}

#[cfg(test)]
pub fn to_cnf<T: Clone>(expr: &BoolExpr<T>, options: &NormalizeOptions) -> Result<Vec<Vec<T>>> {
    match expr {
        BoolExpr::Atom(atom) => Ok(vec![vec![atom.clone()]]),
        BoolExpr::And(children) => {
            let mut clauses = Vec::new();
            for child in children {
                let child_clauses = to_cnf(child, options)?;
                clauses.extend(child_clauses);
                if clauses.len() > options.max_clauses {
                    return Err(NyError::InvalidSpec(format!(
                        "CNF clause count {} exceeds max_clauses {}",
                        clauses.len(),
                        options.max_clauses
                    )));
                }
            }
            Ok(clauses)
        }
        BoolExpr::Or(children) => {
            let mut clauses = vec![Vec::new()];
            for child in children {
                let child_clauses = to_cnf(child, options)?;
                clauses = combine_clauses(&clauses, &child_clauses, options)?;
            }
            Ok(clauses)
        }
    }
}

/// Check if a linear constraint references only input variables.
fn is_input_only(constraint: &LinearConstraint) -> bool {
    !constraint.expr.terms().any(|(_, _)| false)
        && constraint
            .expr
            .terms()
            .all(|(var, _)| var.kind == VarKind::Input)
}

/// Extract input bounds from a single input-only LinearConstraint.
/// Returns `Some((var_index, is_lower, bound_value))` for single-variable constraints.
fn extract_single_input_bound(constraint: &LinearConstraint) -> Option<(usize, bool, f64)> {
    if !is_input_only(constraint) {
        return None;
    }
    let terms: Vec<_> = constraint.expr.terms().collect();
    if terms.len() != 1 {
        return None; // Multi-variable input constraint
    }
    let (var, coeff) = terms[0];
    if coeff.abs() <= COEFF_EPS {
        return None;
    }
    let bound = -constraint.expr.constant_term() / coeff;
    match constraint.relation {
        Relation::LessEq => {
            if *coeff > 0.0 {
                Some((var.index, false, bound)) // X_i <= bound (upper)
            } else {
                Some((var.index, true, bound)) // X_i >= bound (lower)
            }
        }
        Relation::GreaterEq => {
            if *coeff > 0.0 {
                Some((var.index, true, bound)) // X_i >= bound (lower)
            } else {
                Some((var.index, false, bound)) // X_i <= bound (upper)
            }
        }
        Relation::Equal => None,
    }
}

/// Extract per-clause input bounds from DNF clauses containing mixed constraints.
fn extract_per_clause_input_bounds(
    clauses: &[Vec<LinearConstraint>],
) -> Vec<BTreeMap<usize, (f64, f64)>> {
    clauses
        .iter()
        .map(|clause| {
            let mut bounds: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
            for constraint in clause {
                if let Some((idx, is_lower, val)) = extract_single_input_bound(constraint) {
                    let entry = bounds
                        .entry(idx)
                        .or_insert((f64::NEG_INFINITY, f64::INFINITY));
                    if is_lower {
                        entry.0 = entry.0.max(val);
                    } else {
                        entry.1 = entry.1.min(val);
                    }
                }
            }
            bounds
        })
        .collect()
}

/// Remove input-only constraints from DNF clauses, keeping only output constraints.
fn strip_input_constraints(clauses: Vec<Vec<LinearConstraint>>) -> Vec<Vec<LinearConstraint>> {
    clauses
        .into_iter()
        .map(|clause| clause.into_iter().filter(|c| !is_input_only(c)).collect())
        .collect()
}

pub fn normalize_output_constraints(
    content: &str,
    options: NormalizeOptions,
) -> Result<NormalizedConstraints> {
    let content = strip_vnnlib_comments(content);
    let tokens = tokenize(&content)?;
    let exprs = parse_expressions(&tokens)?;
    let decls = parse_declarations(&exprs)?;

    let mut output_exprs = Vec::new();
    let mut input_box_disjunctions = Vec::new();
    for expr in &exprs {
        let Expr::List(items) = expr else {
            continue;
        };
        if items.is_empty() {
            continue;
        }
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        if op == "assert" {
            if let Some(assert_expr) = items.get(1) {
                if let Some(expr) =
                    parse_bool_expr(assert_expr, &decls.input_declared, &decls.output_declared)?
                {
                    output_exprs.push(expr);
                } else if let Some(input_or) = super::parse::parse_input_box_disjunction(
                    assert_expr,
                    &decls.input_declared,
                    &decls.output_declared,
                )? {
                    // A pure input-box disjunction assert (ACAS Xu prop_6:
                    // `(or (and <input box 1>) (and <input box 2>))`). Kept as
                    // a boolean factor so the DNF below distributes the output
                    // clauses over the disjunct boxes — each resulting clause
                    // carries its OWN input box (`per_clause_input_bounds`),
                    // exactly the nn4sys boxed-clause shape the disjunctive
                    // lane verifies per clause over that clause's box.
                    input_box_disjunctions.push(input_or);
                }
            }
        }
    }

    // No output constraints → nothing to normalize. Input-box disjunctions
    // are deliberately NOT emitted alone: a clause with no output constraint
    // has empty-conjunction (trivially satisfiable) semantics that downstream
    // lanes do not implement.
    if output_exprs.is_empty() {
        return Ok(NormalizedConstraints {
            clauses: Vec::new(),
            per_clause_input_bounds: Vec::new(),
        });
    }

    output_exprs.extend(input_box_disjunctions);
    let combined = if output_exprs.len() == 1 {
        output_exprs.remove(0)
    } else {
        BoolExpr::And(output_exprs)
    };

    let clauses = to_dnf(&combined, &options)?;

    // Extract per-clause input bounds from mixed and clauses, then strip
    // input-only constraints so they don't reach the output constraint converter.
    let per_clause_input_bounds = extract_per_clause_input_bounds(&clauses);
    let clauses = strip_input_constraints(clauses);

    // Defense-in-depth (unreachable by construction: every output_expr DNF
    // clause carries at least one output atom, and input-box factors only ever
    // ADD atoms to those clauses): a clause left EMPTY after input stripping
    // would mean a disjunct with no output constraint — its empty-conjunction
    // semantics ("everything in the box violates") are not what the downstream
    // clause loops implement, so fail closed to a sound parse error instead.
    if clauses.iter().any(|clause| clause.is_empty()) {
        return Err(NyError::InvalidSpec(
            "Disjunct contains no output constraint after input-bound extraction".to_string(),
        ));
    }

    Ok(NormalizedConstraints {
        clauses,
        per_clause_input_bounds,
    })
}

pub fn to_output_constraint_clauses(
    normalized: &NormalizedConstraints,
) -> Result<(Vec<Vec<crate::vnnlib::OutputConstraint>>, bool)> {
    if normalized.clauses.is_empty() {
        return Ok((Vec::new(), false));
    }

    let is_disjunction = normalized.clauses.len() > 1;
    let mut output_clauses = Vec::new();

    for clause in &normalized.clauses {
        let mut clause_constraints = Vec::new();
        for constraint in clause {
            clause_constraints.extend(linear_constraint_to_output_constraints(constraint)?);
        }
        output_clauses.push(clause_constraints);
    }

    Ok((output_clauses, is_disjunction))
}

#[cfg(test)]
pub fn to_output_constraints(
    normalized: &NormalizedConstraints,
) -> Result<(Vec<crate::vnnlib::OutputConstraint>, bool)> {
    let (clauses, is_disjunction) = to_output_constraint_clauses(normalized)?;
    let mut output_constraints = Vec::new();
    for clause in clauses {
        output_constraints.extend(clause);
    }
    Ok((output_constraints, is_disjunction))
}

fn linear_constraint_to_output_constraints(
    constraint: &LinearConstraint,
) -> Result<Vec<crate::vnnlib::OutputConstraint>> {
    if matches!(constraint.relation, Relation::Equal) {
        if constraint.is_strict {
            return Err(NyError::InvalidSpec(
                "Strict equality constraints are not supported".to_string(),
            ));
        }
        let mut constraints = convert_linear_relation(&constraint.expr, Relation::LessEq, false)?;
        constraints.extend(convert_linear_relation(
            &constraint.expr,
            Relation::GreaterEq,
            false,
        )?);
        return Ok(constraints);
    }

    convert_linear_relation(&constraint.expr, constraint.relation, constraint.is_strict)
}

fn convert_linear_relation(
    expr: &super::types::LinearExpr,
    relation: Relation,
    is_strict: bool,
) -> Result<Vec<crate::vnnlib::OutputConstraint>> {
    let mut terms: Vec<(usize, f64)> = Vec::new();
    for (var, coeff) in &expr.coeffs {
        match var.kind {
            VarKind::Output => terms.push((var.index, *coeff)),
            VarKind::Input => {
                return Err(NyError::InvalidSpec(
                    "Output constraint references input variable".to_string(),
                ))
            }
        }
    }

    if terms.is_empty() {
        return Err(NyError::InvalidSpec(
            "Output constraint must reference at least one output variable".to_string(),
        ));
    }

    let constant = expr.constant;
    if terms.len() == 1 {
        let (idx, coeff) = terms[0];
        if coeff.abs() <= COEFF_EPS {
            return Err(NyError::InvalidSpec(
                "Output constraint has zero coefficient".to_string(),
            ));
        }
        let rhs = -constant / coeff;
        let (direction, strict) = match relation {
            Relation::LessEq => (coeff > 0.0, is_strict),
            Relation::GreaterEq => (coeff < 0.0, is_strict),
            Relation::Equal => {
                return Err(NyError::InternalError(
                    "Relation::Equal should have been handled before single-term conversion"
                        .to_string(),
                ))
            }
        };

        let constraint = match (direction, strict) {
            (true, false) => crate::vnnlib::OutputConstraint::LessEqConst(idx, rhs),
            (true, true) => crate::vnnlib::OutputConstraint::LessThanConst(idx, rhs),
            (false, false) => crate::vnnlib::OutputConstraint::GreaterEqConst(idx, rhs),
            (false, true) => crate::vnnlib::OutputConstraint::GreaterThanConst(idx, rhs),
        };
        return Ok(vec![constraint]);
    }

    if terms.len() == 2 {
        let (i, a) = terms[0];
        let (j, b) = terms[1];
        if (a + b).abs() > COEFF_EPS {
            return Err(NyError::InvalidSpec(
                "Relational constraint must have opposite coefficients".to_string(),
            ));
        }
        if constant.abs() > COEFF_EPS {
            return Err(NyError::InvalidSpec(
                "Relational constraint must have zero constant".to_string(),
            ));
        }

        let (lhs, rhs) = if a > 0.0 { (i, j) } else { (j, i) };
        let constraint = match relation {
            Relation::LessEq => {
                if is_strict {
                    crate::vnnlib::OutputConstraint::LessThan(lhs, rhs)
                } else {
                    crate::vnnlib::OutputConstraint::LessEq(lhs, rhs)
                }
            }
            Relation::GreaterEq => {
                if is_strict {
                    crate::vnnlib::OutputConstraint::GreaterThan(lhs, rhs)
                } else {
                    crate::vnnlib::OutputConstraint::GreaterEq(lhs, rhs)
                }
            }
            Relation::Equal => {
                return Err(NyError::InternalError(
                    "Relation::Equal should have been handled before two-term conversion"
                        .to_string(),
                ))
            }
        };
        return Ok(vec![constraint]);
    }

    Err(NyError::InvalidSpec(
        "Output constraint uses more than two outputs".to_string(),
    ))
}
