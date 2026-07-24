// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNNLib declaration parsing, expression linearization, and boolean expression parsing.

use super::types::{
    BoolExpr, LinearConstraint, LinearExpr, LinearVar, Relation, VarKind, COEFF_EPS,
};
use crate::vnnlib::syntax::{
    apply_tensor_decl, parse_select_indices, parse_tensor_indices, parse_var_index,
    resolve_var_info, Expr, TensorDecl,
};
use ny_core::{NyError, Result};
use std::collections::HashMap;
use tracing::warn;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConstraintScope {
    InputOnly,
    OutputOnly,
    Mixed,
    None,
}

pub(super) struct DeclState {
    pub(super) input_declared: HashMap<String, TensorDecl>,
    pub(super) output_declared: HashMap<String, TensorDecl>,
}

pub(super) fn parse_declarations(exprs: &[Expr]) -> Result<DeclState> {
    let mut input_declared: HashMap<String, TensorDecl> = HashMap::new();
    let mut output_declared: HashMap<String, TensorDecl> = HashMap::new();
    let mut max_input_idx = 0usize;
    let mut max_output_idx = 0usize;

    for expr in exprs {
        let Expr::List(items) = expr else {
            continue;
        };
        if items.is_empty() {
            continue;
        }
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        match op.as_str() {
            "declare-const" => {
                if let Some(Expr::Symbol(var_name)) = items.get(1) {
                    if let Some(idx) = parse_var_index(var_name, "X_") {
                        let dimension = idx.checked_add(1).ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "input declaration index overflows the platform dimension: {var_name}"
                            ))
                        })?;
                        max_input_idx = max_input_idx.max(dimension);
                    } else if let Some(idx) = parse_var_index(var_name, "Y_") {
                        let dimension = idx.checked_add(1).ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "output declaration index overflows the platform dimension: {var_name}"
                            ))
                        })?;
                        max_output_idx = max_output_idx.max(dimension);
                    }
                }
            }
            "declare-input" | "declare-output" | "declare-hidden" => {
                apply_tensor_decl(
                    op,
                    items,
                    &mut input_declared,
                    &mut output_declared,
                    &mut max_input_idx,
                    &mut max_output_idx,
                )?;
            }
            "declare-network" => {
                if items.len() < 2 {
                    return Err(NyError::InvalidSpec(
                        "declare-network missing network name".to_string(),
                    ));
                }
                for nested in items.iter().skip(2) {
                    let Expr::List(nested_items) = nested else {
                        return Err(NyError::InvalidSpec(
                            "declare-network entries must be lists".to_string(),
                        ));
                    };
                    let Some(Expr::Symbol(nested_op)) = nested_items.first() else {
                        return Err(NyError::InvalidSpec(
                            "declare-network contains invalid entry".to_string(),
                        ));
                    };
                    if nested_op == "declare-input"
                        || nested_op == "declare-output"
                        || nested_op == "declare-hidden"
                    {
                        apply_tensor_decl(
                            nested_op,
                            nested_items,
                            &mut input_declared,
                            &mut output_declared,
                            &mut max_input_idx,
                            &mut max_output_idx,
                        )?;
                    } else {
                        warn!("Ignoring unsupported declare-network entry '{}'", nested_op);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(DeclState {
        input_declared,
        output_declared,
    })
}

pub(super) fn linearize_expr(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<LinearExpr> {
    if let Some((idx, is_input)) = resolve_var_info(expr, input_declared, output_declared)? {
        let kind = if is_input {
            VarKind::Input
        } else {
            VarKind::Output
        };
        return Ok(LinearExpr::var(LinearVar { kind, index: idx }));
    }
    match expr {
        Expr::Number(n) => Ok(LinearExpr::constant(*n)),
        Expr::Symbol(name) => Err(NyError::InvalidSpec(format!(
            "Unknown symbol '{}' in linear expression",
            name
        ))),
        Expr::List(items) => {
            if items.is_empty() {
                return Err(NyError::InvalidSpec(
                    "Empty list in linear expression".to_string(),
                ));
            }
            let Expr::Symbol(op) = &items[0] else {
                return Err(NyError::InvalidSpec(
                    "Invalid list head in linear expression".to_string(),
                ));
            };
            match op.as_str() {
                "+" => {
                    let mut acc = LinearExpr::constant(0.0);
                    for item in items.iter().skip(1) {
                        let term = linearize_expr(item, input_declared, output_declared)?;
                        acc = acc.add(&term);
                    }
                    Ok(acc)
                }
                "-" => {
                    if items.len() == 2 {
                        let term = linearize_expr(&items[1], input_declared, output_declared)?;
                        Ok(term.scale(-1.0))
                    } else if items.len() >= 3 {
                        let mut acc = linearize_expr(&items[1], input_declared, output_declared)?;
                        for item in items.iter().skip(2) {
                            let term = linearize_expr(item, input_declared, output_declared)?;
                            acc = acc.sub(&term);
                        }
                        Ok(acc)
                    } else {
                        Err(NyError::InvalidSpec(
                            "Subtraction expects at least one operand".to_string(),
                        ))
                    }
                }
                "*" => {
                    if items.len() != 3 {
                        return Err(NyError::InvalidSpec(
                            "Multiplication supports exactly two operands".to_string(),
                        ));
                    }
                    let left = linearize_expr(&items[1], input_declared, output_declared)?;
                    let right = linearize_expr(&items[2], input_declared, output_declared)?;
                    if let Some(scale) = left.as_constant() {
                        Ok(right.scale(scale))
                    } else if let Some(scale) = right.as_constant() {
                        Ok(left.scale(scale))
                    } else {
                        Err(NyError::InvalidSpec(
                            "Non-linear multiplication in constraint".to_string(),
                        ))
                    }
                }
                "/" => {
                    if items.len() != 3 {
                        return Err(NyError::InvalidSpec(
                            "Division supports exactly two operands".to_string(),
                        ));
                    }
                    let numerator = linearize_expr(&items[1], input_declared, output_declared)?;
                    let denominator = linearize_expr(&items[2], input_declared, output_declared)?;
                    if let Some(scale) = denominator.as_constant() {
                        if scale.abs() <= COEFF_EPS {
                            return Err(NyError::InvalidSpec(
                                "Division by zero in linear expression".to_string(),
                            ));
                        }
                        Ok(numerator.scale(1.0 / scale))
                    } else {
                        Err(NyError::InvalidSpec(
                            "Non-linear division in constraint".to_string(),
                        ))
                    }
                }
                _ => Err(NyError::InvalidSpec(format!(
                    "Unsupported operator '{}' in linear expression",
                    op
                ))),
            }
        }
    }
}

pub fn parse_linear_constraint(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<LinearConstraint> {
    let Expr::List(items) = expr else {
        return Err(NyError::InvalidSpec(
            "Expected comparison expression".to_string(),
        ));
    };
    if items.len() != 3 {
        return Err(NyError::InvalidSpec(
            "Comparison expression requires exactly two operands".to_string(),
        ));
    }
    let Expr::Symbol(op) = &items[0] else {
        return Err(NyError::InvalidSpec(
            "Invalid comparison operator".to_string(),
        ));
    };
    let lhs = linearize_expr(&items[1], input_declared, output_declared)?;
    let rhs = linearize_expr(&items[2], input_declared, output_declared)?;
    let diff = lhs.sub(&rhs);
    if diff.is_empty() {
        return Err(NyError::InvalidSpec(
            "Constraint must reference at least one variable".to_string(),
        ));
    }
    let (relation, is_strict) = match op.as_str() {
        "<=" => (Relation::LessEq, false),
        ">=" => (Relation::GreaterEq, false),
        "<" => (Relation::LessEq, true),
        ">" => (Relation::GreaterEq, true),
        // `==` is VNN-LIB 2.0's spelling of equality (the 2026 relational
        // ACAS benchmarks use it for the f/g input couplings); identical
        // semantics to `=`.
        "=" | "==" => (Relation::Equal, false),
        _ => {
            return Err(NyError::InvalidSpec(format!(
                "Unsupported comparison operator '{}'",
                op
            )))
        }
    };
    Ok(LinearConstraint {
        expr: diff,
        relation,
        is_strict,
    })
}

pub(super) fn constraint_scope(constraint: &LinearConstraint) -> ConstraintScope {
    let mut has_input = false;
    let mut has_output = false;
    for var in constraint.expr.coeffs.keys() {
        match var.kind {
            VarKind::Input => has_input = true,
            VarKind::Output => has_output = true,
        }
    }
    match (has_input, has_output) {
        (true, true) => ConstraintScope::Mixed,
        (true, false) => ConstraintScope::InputOnly,
        (false, true) => ConstraintScope::OutputOnly,
        (false, false) => ConstraintScope::None,
    }
}

pub(super) fn parse_bool_expr(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<Option<BoolExpr<LinearConstraint>>> {
    let Expr::List(items) = expr else {
        return Err(NyError::InvalidSpec(
            "Unsupported assert expression".to_string(),
        ));
    };
    if items.is_empty() {
        return Err(NyError::InvalidSpec("Empty assert expression".to_string()));
    }
    let Expr::Symbol(op) = &items[0] else {
        return Err(NyError::InvalidSpec("Invalid assert operator".to_string()));
    };
    match op.as_str() {
        "and" | "or" => {
            let mut children = Vec::new();
            let mut saw_output = false;
            let mut saw_input = false;
            for child in items.iter().skip(1) {
                match parse_bool_expr(child, input_declared, output_declared)? {
                    Some(expr) => {
                        saw_output = true;
                        children.push(expr);
                    }
                    None => {
                        saw_input = true;
                    }
                }
            }
            if saw_input && saw_output {
                if op == "or" {
                    return Err(NyError::InvalidSpec(
                        "Mixed input-only and output constraints in 'or' expression".to_string(),
                    ));
                }
                // Mixed 'and': valid VNN-LIB pattern (e.g., lindex benchmarks).
                // Re-parse all children, keeping input constraints as atoms
                // alongside output constraints. Per-clause input bounds are
                // extracted later in normalize_output_constraints.
                let mut all_children = Vec::new();
                for child in items.iter().skip(1) {
                    let Expr::List(child_items) = child else {
                        return Err(NyError::InvalidSpec(
                            "Non-list expression in mixed and clause".to_string(),
                        ));
                    };
                    if child_items.is_empty() {
                        continue;
                    }
                    let Some(Expr::Symbol(child_op)) = child_items.first() else {
                        continue;
                    };
                    if matches!(child_op.as_str(), "<=" | ">=" | "<" | ">" | "=") {
                        let lc = parse_linear_constraint(child, input_declared, output_declared)?;
                        all_children.push(BoolExpr::Atom(lc));
                    } else {
                        // Nested boolean expression
                        if let Some(expr) = parse_bool_expr(child, input_declared, output_declared)?
                        {
                            all_children.push(expr);
                        }
                    }
                }
                let flattened = flatten_bool(op, all_children);
                return Ok(Some(BoolExpr::And(flattened)));
            }
            if !saw_output {
                return Ok(None);
            }
            let flattened = flatten_bool(op, children);
            Ok(Some(match op.as_str() {
                "and" => BoolExpr::And(flattened),
                _ => BoolExpr::Or(flattened),
            }))
        }
        "<=" | ">=" | "<" | ">" | "=" => {
            if !expr_contains_output(expr, output_declared) {
                return Ok(None);
            }
            let constraint = parse_linear_constraint(expr, input_declared, output_declared)?;
            match constraint_scope(&constraint) {
                ConstraintScope::InputOnly => Ok(None),
                ConstraintScope::OutputOnly => Ok(Some(BoolExpr::Atom(constraint))),
                ConstraintScope::Mixed => Err(NyError::InvalidSpec(
                    "Constraint mixes input and output variables".to_string(),
                )),
                ConstraintScope::None => Err(NyError::InvalidSpec(
                    "Constraint must reference at least one variable".to_string(),
                )),
            }
        }
        _ => Err(NyError::InvalidSpec(format!(
            "Unsupported boolean operator '{}' in assert",
            op
        ))),
    }
}

/// Parse a PURE input-box disjunction assert: `(or <disjunct> ...)` where
/// every leaf of every disjunct is a single-variable input-only comparison
/// (disjuncts are typically `(and <input atoms>)`, e.g. ACAS Xu prop_6's
/// per-disjunct input boxes).
///
/// `parse_bool_expr` returns `None` for this shape (no output atom anywhere),
/// which used to DROP the disjunct boxes entirely — the main parser then
/// intersected the boxes' atoms into one (often empty) global box and the
/// whole property degraded to a sound-unknown parse error. Capturing the
/// disjunction here lets `normalize_output_constraints` distribute the output
/// clauses over the disjunct boxes via DNF, yielding the same per-clause
/// input-box shape as the nn4sys mscn/lindex benchmarks (which the
/// boxed-clause disjunctive lane verifies clause-by-clause over each clause's
/// OWN box).
///
/// Returns `Ok(None)` (decline — caller keeps the legacy ignore-the-assert
/// behavior, which only ever WIDENS the verified domain and is therefore
/// sound) for any other shape: a non-`or` head, any output mention, any
/// multi-variable/non-linear leaf.
pub(super) fn parse_input_box_disjunction(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<Option<BoolExpr<LinearConstraint>>> {
    let Expr::List(items) = expr else {
        return Ok(None);
    };
    if !matches!(items.first(), Some(Expr::Symbol(op)) if op == "or") {
        return Ok(None);
    }
    if expr_contains_output(expr, output_declared) {
        return Ok(None);
    }
    parse_input_only_bool_tree(expr, input_declared, output_declared)
}

/// Recursive core of [`parse_input_box_disjunction`]: an `and`/`or` tree whose
/// leaves are all single-variable input-only comparisons. Declines (Ok(None))
/// on the first leaf that is anything else, so a partially-supported assert is
/// ignored as a whole rather than half-captured.
fn parse_input_only_bool_tree(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<Option<BoolExpr<LinearConstraint>>> {
    let Expr::List(items) = expr else {
        return Ok(None);
    };
    let Some(Expr::Symbol(op)) = items.first() else {
        return Ok(None);
    };
    match op.as_str() {
        "and" | "or" => {
            let mut children = Vec::with_capacity(items.len().saturating_sub(1));
            for child in items.iter().skip(1) {
                match parse_input_only_bool_tree(child, input_declared, output_declared)? {
                    Some(child_expr) => children.push(child_expr),
                    None => return Ok(None),
                }
            }
            if children.is_empty() {
                return Ok(None);
            }
            let flattened = flatten_bool(op, children);
            Ok(Some(if op == "and" {
                BoolExpr::And(flattened)
            } else {
                BoolExpr::Or(flattened)
            }))
        }
        "<=" | ">=" | "<" | ">" | "=" => {
            // A leaf that fails to linearize (unknown symbol, non-linear term)
            // declines the whole assert instead of erroring: ignoring an
            // input-only assert widens the domain (sound), while erroring
            // would reject specs the legacy path accepted.
            let Ok(constraint) = parse_linear_constraint(expr, input_declared, output_declared)
            else {
                return Ok(None);
            };
            // Only single-variable, non-equality atoms:
            // `extract_single_input_bound` is what ultimately consumes these,
            // and it can only turn one-variable `<=`/`>=` constraints into
            // per-clause boxes (it returns None for `=` and multi-variable
            // atoms). Such an atom would be silently dropped there — a clause
            // box MISSING one of its constraints (an over-approximation that
            // is sound for unsat but wider than the written spec, which the
            // witness gate must not be) — so decline the whole assert.
            let is_single_var_input =
                matches!(constraint_scope(&constraint), ConstraintScope::InputOnly)
                    && constraint.expr.terms().count() == 1
                    && !matches!(constraint.relation, Relation::Equal);
            if !is_single_var_input {
                return Ok(None);
            }
            Ok(Some(BoolExpr::Atom(constraint)))
        }
        _ => Ok(None),
    }
}

fn expr_contains_output(expr: &Expr, output_declared: &HashMap<String, TensorDecl>) -> bool {
    match expr {
        Expr::Symbol(name) => {
            if parse_var_index(name, "Y_").is_some() {
                return true;
            }
            if let Ok(Some((base, _))) = parse_tensor_indices(name) {
                return output_declared.contains_key(&base);
            }
            false
        }
        Expr::List(items) => {
            if let Ok(Some((base, _))) = parse_select_indices(expr) {
                if output_declared.contains_key(&base) {
                    return true;
                }
            }
            items
                .iter()
                .any(|item| expr_contains_output(item, output_declared))
        }
        _ => false,
    }
}

fn flatten_bool<T: Clone>(op: &str, nodes: Vec<BoolExpr<T>>) -> Vec<BoolExpr<T>> {
    let mut flattened = Vec::new();
    for node in nodes {
        match node {
            BoolExpr::And(children) if op == "and" => flattened.extend(children),
            BoolExpr::Or(children) if op == "or" => flattened.extend(children),
            other => flattened.push(other),
        }
    }
    flattened
}
