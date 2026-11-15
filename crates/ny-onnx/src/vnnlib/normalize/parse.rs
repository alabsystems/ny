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

/// What to do with a PURE-INPUT leaf that the affine input box cannot represent.
///
/// `parse_bool_expr` returns `Ok(None)` for pure-input atoms because the input
/// BOX collectors own them: `parser.rs` `parse_assert` matches
/// `<bare var> op <number>` (and the reversed form), and — in v2 files only —
/// `try_apply_linear_input_constraint` additionally reduces a single-variable
/// affine atom. Anything OUTSIDE those shapes is captured by nobody: it
/// vanishes from `input_bounds`, from `declared_input_bounds`, from
/// `per_clause_input_bounds`, and therefore from every witness gate.
///
/// # Direction (why this is a refusal and not a widening)
///
/// A dropped input atom only ever REMOVES a restriction — bounds merge with
/// `min`/`max` and per-clause boxes intersect, so nothing narrows. The verified
/// region is then a strict SUPERSET of the region the property specifies.
/// `unsat` over a superset is still correct. `sat` is NOT: the witness may
/// violate the dropped atom, so it lies outside the property's input region,
/// and the organiser's checker — which evaluates the full assert — scores it
/// INCORRECT. Refusing costs at most one verdict; dropping can cost a wrong
/// one, so the atom must fail closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputLeafPolicy {
    /// Refuse an unrepresentable pure-input atom: nothing downstream would
    /// capture it, so declining would erase it from the spec.
    FailClosed,
    /// Decline (`Ok(None)`) and let the caller speak. Used while descending
    /// into an `and`/`or` whose own re-parse carries a STRONGER certificate for
    /// the same atom (the mixed-`and` per-clause-box check below, and the
    /// mixed-`or` rejection), so the caller reports the precise reason instead
    /// of this generic one.
    DeferToCaller,
}

pub(super) fn parse_bool_expr(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<Option<BoolExpr<LinearConstraint>>> {
    parse_bool_expr_with_policy(
        expr,
        input_declared,
        output_declared,
        InputLeafPolicy::FailClosed,
    )
}

fn parse_bool_expr_with_policy(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
    input_leaf_policy: InputLeafPolicy,
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
                // Descend under `DeferToCaller`: if this turns out to be a
                // MIXED `and`, the re-parse below (or, for `or`, the hard
                // rejection) carries a stronger, better-worded certificate for
                // the very same atom. The `!saw_output` arm re-runs the
                // children under this node's own policy once we know no such
                // caller exists.
                match parse_bool_expr_with_policy(
                    child,
                    input_declared,
                    output_declared,
                    InputLeafPolicy::DeferToCaller,
                )? {
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
                        return Err(NyError::InvalidSpec(
                            "Empty expression in mixed input/output 'and' clause".to_string(),
                        ));
                    }
                    let Some(Expr::Symbol(child_op)) = child_items.first() else {
                        return Err(NyError::InvalidSpec(
                            "Invalid expression in mixed input/output 'and' clause".to_string(),
                        ));
                    };
                    if matches!(child_op.as_str(), "<=" | ">=" | "<" | ">" | "=" | "==") {
                        let lc = parse_linear_constraint(child, input_declared, output_declared)?;
                        // FAIL CLOSED on an input atom the per-clause box cannot
                        // represent (#mixed-and-drop). `extract_single_input_bound`
                        // returns None for multi-variable atoms and for `=`, and
                        // `strip_input_constraints` then removes EVERY input-only
                        // constraint regardless of whether a bound was captured — so
                        // the atom vanishes from the spec entirely. That is wrong in
                        // BOTH directions: the clause's per-clause map is left empty
                        // so `property_violated_f64` applies no input gate to it
                        // (false SAT, the nn4sys-71 class), and the union widening
                        // contributes nothing for that clause so its true region can
                        // fall outside the searched domain (false UNSAT).
                        //
                        // This assertion also contains an output condition, so
                        // declining with `Ok(None)` would drop the WHOLE unsafe
                        // clause. Fail closed with a hard parse error instead.
                        let is_input_only =
                            matches!(constraint_scope(&lc), ConstraintScope::InputOnly);
                        if is_input_only
                            && (lc.is_strict
                                || lc.expr.terms().count() != 1
                                || matches!(lc.relation, Relation::Equal))
                        {
                            return Err(NyError::InvalidSpec(if lc.is_strict {
                                "Strict input constraints are unsupported: affine input boxes \
                                     cannot represent exclusive endpoints"
                                    .to_string()
                            } else {
                                "Input constraint in mixed input/output 'and' clause cannot be \
                                     represented as one per-clause bound"
                                    .to_string()
                            }));
                        }
                        all_children.push(BoolExpr::Atom(lc));
                    } else {
                        // A nested output/mixed boolean tree is parsed normally.
                        // A nested pure-input tree must be retained too; silently
                        // skipping it widens the unsafe clause and can certify a
                        // false SAT witness.
                        if let Some(expr) = parse_bool_expr_with_policy(
                            child,
                            input_declared,
                            output_declared,
                            InputLeafPolicy::DeferToCaller,
                        )? {
                            all_children.push(expr);
                        } else if let Some(input_expr) =
                            parse_input_only_bool_tree(child, input_declared, output_declared)?
                        {
                            all_children.push(input_expr);
                        } else {
                            return Err(NyError::InvalidSpec(
                                "Unsupported input expression in mixed input/output 'and' clause"
                                    .to_string(),
                            ));
                        }
                    }
                }
                let flattened = flatten_bool(op, all_children);
                return Ok(Some(BoolExpr::And(flattened)));
            }
            if !saw_output {
                // A PURE-INPUT boolean tree. No caller downstream carries a
                // certificate for it: `parse_input_box_disjunction` only accepts
                // an `or` head whose every leaf is a single-variable comparison,
                // and `parse_assert` skips disjunct-scoped input atoms entirely
                // (`in_disjunction => return Ok(())`). So an unrepresentable
                // leaf in here is dropped by EVERYONE. Re-run the children under
                // this node's own policy — which is `FailClosed` at the top of an
                // assert and `DeferToCaller` only while a mixed-`and` re-parse is
                // still going to speak for them.
                //
                // Skipped entirely when this node is itself deferring: the pass
                // would find nothing (every leaf would defer too) and, since each
                // level would repeat it, re-running unconditionally would cost
                // 2^depth traversals instead of depth*size.
                if input_leaf_policy == InputLeafPolicy::FailClosed {
                    for child in items.iter().skip(1) {
                        // Result discarded on purpose: the first pass above
                        // already collected what parses; this pass exists ONLY so
                        // an unrepresentable input leaf raises its refusal.
                        let _ = parse_bool_expr_with_policy(
                            child,
                            input_declared,
                            output_declared,
                            InputLeafPolicy::FailClosed,
                        )?;
                    }
                }
                return Ok(None);
            }
            let flattened = flatten_bool(op, children);
            Ok(Some(match op.as_str() {
                "and" => BoolExpr::And(flattened),
                _ => BoolExpr::Or(flattened),
            }))
        }
        "<=" | ">=" | "<" | ">" | "=" | "==" => {
            if !expr_contains_output(expr, output_declared) {
                // Pure input atoms are normally omitted from the output
                // expression. Inspect strict affine leaves before doing so:
                // omitting their openness and later retaining an inclusive
                // endpoint would make boundary witnesses falsely valid.
                if matches!(op.as_str(), "<" | ">") {
                    if let Ok(constraint) =
                        parse_linear_constraint(expr, input_declared, output_declared)
                    {
                        if matches!(constraint_scope(&constraint), ConstraintScope::InputOnly) {
                            return Err(NyError::InvalidSpec(
                                "Strict input constraints are unsupported: affine input boxes \
                                 cannot represent exclusive endpoints"
                                    .to_string(),
                            ));
                        }
                    }
                }
                // The NON-strict half of the same certificate. The strict check
                // above rejects an atom whose OPENNESS the box cannot hold; this
                // one rejects an atom whose SHAPE it cannot hold. Both halves are
                // needed: a `<=` over two input variables has no exclusive
                // endpoint to lose, yet the box represents it no better, and the
                // v2 twin `try_apply_linear_input_constraint`
                // (`parser.rs`, "Input constraints must reference a single
                // variable") has always refused exactly this shape. VNN-LIB 1.0
                // never reaches that twin — `parse_assert` guards it with
                // `v2_mode &&` — so the v1 atom was silently DROPPED while its
                // 2.0 spelling was refused.
                //
                // Only `terms().count() != 1` is refused, which is the twin's own
                // condition: a single-variable atom is exactly what an affine box
                // CAN hold, and refusing those would reject `(assert (<= X_0 1))`
                // — the most common assert in the corpus.
                //
                // A leaf that fails to linearize is left alone (`if let Ok`), as
                // the strict check above leaves it: non-linear input atoms belong
                // to `nonlinear::NonLinearFormula`, not to this affine path.
                if input_leaf_policy == InputLeafPolicy::FailClosed {
                    if let Ok(constraint) =
                        parse_linear_constraint(expr, input_declared, output_declared)
                    {
                        if matches!(constraint_scope(&constraint), ConstraintScope::InputOnly)
                            && !constraint.is_strict
                            && constraint.expr.terms().count() != 1
                        {
                            return Err(NyError::InvalidSpec(
                                "Input constraints must reference a single variable: a \
                                 multi-variable input-only constraint cannot be represented as \
                                 an affine input box, and dropping it would verify a LARGER \
                                 input region than the property specifies"
                                    .to_string(),
                            ));
                        }
                    }
                }
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
/// Returns `Ok(None)` when this assert is not an input-box disjunction at all
/// (a non-`or` head, or any output mention) — those shapes are `parse_bool_expr`'s
/// and `parse_assert`'s business, not this function's.
///
/// An `or` head with NO output mention IS this function's business, and it fails
/// closed rather than declining. The old contract said declining "only ever
/// WIDENS the verified domain and is therefore sound"; the first half is true
/// and the second does not follow. `parse_assert` deliberately applies NOTHING
/// from a disjunct-scoped input atom (`parser.rs`, `in_disjunction => return
/// Ok(())`, because a disjunct bounds one region and not the global domain), so
/// declining here drops the WHOLE disjunction from every consumer — from
/// `input_bounds`, from `per_clause_input_bounds`, and from the witness gates.
/// A wider domain is sound for `unsat` and UNSOUND for `sat`: a witness found in
/// the widened region need not satisfy the dropped disjunction, and the
/// organiser's checker evaluates it and scores the row INCORRECT. The sibling
/// comment on `is_single_var_input` in [`parse_input_only_bool_tree`] already
/// states this ("wider than the written spec, which the witness gate must not
/// be"); this one contradicted it.
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
    // `parse_input_only_bool_tree` still DECLINES on an unsupported leaf, which
    // is what its other caller (the mixed-`and` nested-tree fallback) needs so
    // that caller can report its own message. Here the decline has no second
    // reader, so convert it into the refusal.
    match parse_input_only_bool_tree(expr, input_declared, output_declared)? {
        Some(tree) => Ok(Some(tree)),
        None => Err(NyError::InvalidSpec(
            "Unsupported pure-input disjunction: every leaf must be a single-variable, \
             non-equality input comparison for the per-clause boxes to represent it. \
             Dropping the disjunction would verify a LARGER input region than the property \
             specifies"
                .to_string(),
        )),
    }
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
        "<=" | ">=" | "<" | ">" | "=" | "==" => {
            // A leaf that fails to linearize (unknown symbol, non-linear term)
            // declines the whole assert instead of erroring: ignoring an
            // input-only assert widens the domain (sound), while erroring
            // would reject specs the legacy path accepted.
            let Ok(constraint) = parse_linear_constraint(expr, input_declared, output_declared)
            else {
                return Ok(None);
            };
            if constraint.is_strict
                && matches!(constraint_scope(&constraint), ConstraintScope::InputOnly)
            {
                return Err(NyError::InvalidSpec(
                    "Strict input constraints are unsupported: affine input boxes cannot \
                     represent exclusive endpoints"
                        .to_string(),
                ));
            }
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

#[cfg(test)]
mod fail_closed_input_atom_tests {
    use super::super::{normalize_output_constraints, NormalizeOptions};
    use crate::vnnlib::parse_vnnlib;

    fn normalize_err(content: &str) -> String {
        normalize_output_constraints(content, NormalizeOptions::default())
            .expect_err("expected a refusal")
            .to_string()
    }

    /// WIRING, through the real `parse_vnnlib` entry point: a VNN-LIB 1.0
    /// relational input atom must not vanish.
    ///
    /// On the pre-fix tree this file parsed cleanly and the resulting spec had
    /// **no trace of `X_0 <= X_1`** — not in `input_bounds`, not in
    /// `declared_input_bounds`, not in `per_clause_input_bounds`. The box was
    /// `[0,1]^2`, so the point `(1.0, 0.0)` is inside the searched region and
    /// violates the assert: a witness there is confirmed locally and rejected
    /// by the organiser's checker. The VNN-LIB 2.0 spelling of the same atom
    /// has always been refused (`try_apply_linear_input_constraint`,
    /// "Input constraints must reference a single variable"); only the 1.0 path
    /// dropped it, because `parse_assert` guards that twin with `v2_mode &&`.
    #[test]
    fn v1_relational_input_atom_is_refused_not_dropped() {
        let with_relation = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= X_0 X_1))
(assert (>= Y_0 0.0))
";
        let error = parse_vnnlib(with_relation)
            .expect_err("a dropped input relation must fail closed, not widen the region");
        assert!(
            error.to_string().contains("single variable"),
            "refusal should name the representability limit: {error}"
        );

        // Positive control that the atom really was contributing NOTHING before:
        // deleting it leaves an identical spec, which is exactly why dropping it
        // was invisible.
        let without_relation = with_relation.replace("(assert (<= X_0 X_1))", "");
        let spec = parse_vnnlib(&without_relation).expect("the box-only spec still parses");
        assert_eq!(spec.input_bounds, vec![(0.0, 1.0), (0.0, 1.0)]);
    }

    /// Same defect, affine spelling: `(<= (+ X_0 X_1) 5.0)`.
    #[test]
    fn v1_affine_multi_variable_input_atom_is_refused() {
        let content = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= (+ X_0 X_1) 1.0))
(assert (>= Y_0 0.0))
";
        assert!(
            normalize_err(content).contains("single variable"),
            "an affine two-variable input atom must fail closed"
        );
    }

    /// The pure-input DISJUNCTION half of the same defect.
    ///
    /// `parse_assert` deliberately applies nothing from a disjunct-scoped input
    /// atom, so when `parse_input_box_disjunction` declined, the whole `or`
    /// disappeared. On the pre-fix tree this spec parsed with
    /// `input_bounds[1] == (-inf, +inf)` — the verified region for `X_1` was the
    /// entire real line while the property confines it to two small intervals.
    #[test]
    fn pure_input_disjunction_with_unrepresentable_leaf_is_refused_not_dropped() {
        let content = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (or
    (and (<= X_1 0.5) (>= X_1 0.1))
    (and (= X_1 -0.3))
))
(assert (>= Y_0 0.0))
";
        let error = parse_vnnlib(content)
            .expect_err("a dropped input-box disjunction must fail closed, not widen X_1 to R");
        assert!(
            error
                .to_string()
                .contains("Unsupported pure-input disjunction"),
            "refusal should name the disjunction: {error}"
        );
    }

    /// OVER-CORRECTION GUARD: everything the affine box CAN represent must keep
    /// parsing. These are the shapes that make up the entire scored corpus, so a
    /// refusal here would cost verdicts on every benchmark.
    #[test]
    fn representable_input_shapes_still_parse() {
        // Plain single-variable box, direct and reversed, plus an equality.
        let plain = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (>= X_0 -1.0))
(assert (>= 1.0 X_0))
(assert (= X_1 0.25))
(assert (>= Y_0 0.0))
";
        let spec = parse_vnnlib(plain).expect("single-variable input atoms must still parse");
        // direct lower bound + REVERSED upper bound (`1.0 >= X_0`), and an
        // equality collapsed to a point box — all three are shapes the affine
        // box represents, so all three must survive.
        assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
        assert_eq!(spec.input_bounds[1], (0.25, 0.25));

        // ACAS Xu prop_6 shape: a pure-input `or` of single-variable boxes still
        // distributes into per-clause boxes.
        let prop6 = "
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (or
    (and (<= X_0 0.5) (>= X_0 0.1))
    (and (<= X_0 -0.1) (>= X_0 -0.5))
))
(assert (<= Y_0 Y_1))
";
        let spec = parse_vnnlib(prop6).expect("prop_6-shaped input-box disjunction must parse");
        assert_eq!(spec.per_clause_input_bounds.len(), 2);
        assert_eq!(spec.input_bounds[0], (-0.5, 0.5));
    }

    /// The refusal must DEFER wherever a caller already carries a stronger
    /// certificate, so the mixed-`and` diagnostics keep naming the real reason.
    /// This pins the `InputLeafPolicy::DeferToCaller` plumbing: if the leaf
    /// refusal ever fires eagerly inside a boolean tree, these messages change.
    #[test]
    fn mixed_and_certificates_still_speak_for_their_own_atoms() {
        let direct = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (and (<= (+ X_0 X_1) 1.0) (>= Y_0 0.0)))
";
        assert!(
            normalize_err(direct).contains("cannot be represented as one per-clause bound"),
            "the mixed-`and` per-clause certificate must still be the one reported"
        );

        let nested = "
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (and
  (or (<= (+ X_0 X_1) 1.0) (>= X_0 0.0))
  (>= Y_0 0.0)
))
";
        assert!(
            normalize_err(nested).contains("Unsupported input expression in mixed input/output"),
            "the nested-tree certificate must still be the one reported"
        );

        let strict = "
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (and (< X_0 1.0) (> Y_0 0.0)))
";
        assert!(
            normalize_err(strict).contains("Strict input constraints"),
            "the strict-endpoint certificate must still be the one reported"
        );
    }
}
