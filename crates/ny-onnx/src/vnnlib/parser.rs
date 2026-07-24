// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::normalize;
use super::syntax::{
    apply_tensor_decl, get_number, parse_expressions, parse_tensor_decl, parse_tensor_indices,
    parse_var_index, resolve_var_info, strip_vnnlib_comments, tokenize, Expr, TensorDecl,
};
use super::{
    DeclaredNetwork, DualNetworkProperty, DualNetworkSpec, DualNetworkValidation,
    IsomorphicAtomRelation, IsomorphicOutputAtom, NetworkRelation, OutputConstraint, VnnLibSpec,
};
use super::{TensorDeclaration, TensorDeclarationKind};
use ny_core::{nan_propagating_max_f64, nan_propagating_min_f64, NyError, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn};

/// Single-entry parse memo for [`load_vnnlib`] (#vnnlib-parse-once).
///
/// One `ny vnncomp` run loads the SAME property file several times (the
/// upfront translate/attack pass, the verification driver, and the f64
/// witness-recheck spec each call `load_vnnlib`); on the 20-34MB nn4sys mscn
/// properties every parse costs seconds, so the redundant parses dominated
/// the official budgets. The memo keys on (canonical path, file size, mtime)
/// and returns a CLONE of the cached spec — value-identical to a re-parse
/// (`parse_vnnlib` is a pure function of the file content) and invalidated
/// whenever the file changes. Holds at most one spec (the last file parsed).
///
/// Kill-switch: `NY_VNNLIB_CACHE=0` restores a full parse per call.
type VnnlibCacheKey = (std::path::PathBuf, u64, Option<std::time::SystemTime>);

static VNNLIB_PARSE_MEMO: std::sync::Mutex<Option<(VnnlibCacheKey, std::sync::Arc<VnnLibSpec>)>> =
    std::sync::Mutex::new(None);

fn vnnlib_cache_enabled() -> bool {
    !std::env::var("NY_VNNLIB_CACHE").is_ok_and(|v| v == "0")
}

/// Cache key for `path`, or `None` (skip the memo) when the metadata is
/// unavailable. The canonical path unifies different spellings; size + mtime
/// invalidate on any content change the filesystem can observe.
fn vnnlib_cache_key(path: &Path) -> Option<VnnlibCacheKey> {
    let meta = std::fs::metadata(path).ok()?;
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Some((canon, meta.len(), meta.modified().ok()))
}

/// Load a VNN-LIB property specification from a file.
///
/// Repeated loads of an unchanged file are served from a parse memo
/// (#vnnlib-parse-once) — value-identical to a fresh parse; disable with
/// `NY_VNNLIB_CACHE=0`.
///
/// # Arguments
///
/// * `path` - Path to the .vnnlib file
///
/// # Returns
///
/// A parsed `VnnLibSpec` containing input bounds and output constraints.
pub fn load_vnnlib<P: AsRef<Path>>(path: P) -> Result<VnnLibSpec> {
    let path = path.as_ref();
    info!("Loading VNN-LIB from: {}", path.display());

    let key = if vnnlib_cache_enabled() {
        vnnlib_cache_key(path)
    } else {
        None
    };
    if let Some(key_ref) = key.as_ref() {
        if let Ok(memo) = VNNLIB_PARSE_MEMO.lock() {
            if let Some((cached_key, spec)) = memo.as_ref() {
                if cached_key == key_ref {
                    info!("Reusing memoized VNN-LIB parse (file unchanged)");
                    return Ok((**spec).clone());
                }
            }
        }
    }

    let content = crate::io::read_string_maybe_gzip(path)?;

    let spec = parse_vnnlib(&content)?;
    if let Some(key) = key {
        let shared = std::sync::Arc::new(spec);
        if let Ok(mut memo) = VNNLIB_PARSE_MEMO.lock() {
            *memo = Some((key, std::sync::Arc::clone(&shared)));
        }
        return Ok((*shared).clone());
    }
    Ok(spec)
}

/// Load the tensor definitions that a VNN-LIB 2.0 counterexample assignment
/// must contain. VNN-LIB 1.0 properties return an empty vector.
pub fn load_vnnlib_assignment_declarations<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<TensorDeclaration>> {
    let content = crate::io::read_string_maybe_gzip(path.as_ref())?;
    parse_vnnlib_assignment_declarations(&content)
}

/// Parse tensor definitions in the exact order required by VNN-LIB 2.0
/// section 5.3 and the official VNN-COMP checker: networks in source order,
/// with inputs, hidden tensors, then outputs within each network.
pub fn parse_vnnlib_assignment_declarations(content: &str) -> Result<Vec<TensorDeclaration>> {
    let cleaned_content = strip_vnnlib_comments(content);
    let tokens = tokenize(&cleaned_content)?;
    let exprs = parse_expressions(&tokens)?;
    let mut network_declarations = Vec::new();
    let mut top_level = [Vec::new(), Vec::new(), Vec::new()];

    for expr in &exprs {
        let Expr::List(items) = expr else { continue };
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        if op == "declare-network" {
            let Some(Expr::Symbol(network_name)) = items.get(1) else {
                return Err(NyError::InvalidSpec(
                    "declare-network missing network name".to_string(),
                ));
            };
            let mut grouped = [Vec::new(), Vec::new(), Vec::new()];
            for nested in items.iter().skip(2) {
                let Expr::List(nested_items) = nested else {
                    continue;
                };
                let Some(Expr::Symbol(nested_op)) = nested_items.first() else {
                    continue;
                };
                if let Some((kind, group)) = declaration_kind(nested_op) {
                    grouped[group].push(parse_assignment_declaration(
                        nested_items,
                        nested_op,
                        Some(network_name.clone()),
                        kind,
                    )?);
                }
            }
            network_declarations.extend(grouped.into_iter().flatten());
        } else if let Some((kind, group)) = declaration_kind(op) {
            top_level[group].push(parse_assignment_declaration(items, op, None, kind)?);
        }
    }

    if !network_declarations.is_empty() && top_level.iter().any(|group| !group.is_empty()) {
        return Err(NyError::InvalidSpec(
            "VNN-LIB 2.0 mixes top-level and declare-network tensor definitions".to_string(),
        ));
    }
    let declarations = if network_declarations.is_empty() {
        top_level.into_iter().flatten().collect()
    } else {
        network_declarations
    };

    let mut names = std::collections::HashSet::new();
    for declaration in &declarations {
        if !names.insert(declaration.name.clone()) {
            return Err(NyError::InvalidSpec(format!(
                "Duplicate tensor declaration '{}'",
                declaration.name
            )));
        }
    }
    Ok(declarations)
}

fn declaration_kind(op: &str) -> Option<(TensorDeclarationKind, usize)> {
    match op {
        "declare-input" => Some((TensorDeclarationKind::Input, 0)),
        "declare-hidden" => Some((TensorDeclarationKind::Hidden, 1)),
        "declare-output" => Some((TensorDeclarationKind::Output, 2)),
        _ => None,
    }
}

fn parse_assignment_declaration(
    items: &[Expr],
    declaration_op: &str,
    network: Option<String>,
    kind: TensorDeclarationKind,
) -> Result<TensorDeclaration> {
    let (name, shape) = parse_tensor_decl(items, declaration_op)?;
    let name =
        name.ok_or_else(|| NyError::InvalidSpec(format!("{declaration_op} missing tensor name")))?;
    let element_type = match items.get(2) {
        Some(Expr::Symbol(element_type)) => element_type.clone(),
        _ => {
            return Err(NyError::InvalidSpec(format!(
                "{declaration_op} for '{name}' missing element type"
            )))
        }
    };
    let shape = shape.ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{declaration_op} for '{name}' missing tensor shape"
        ))
    })?;
    tensor_shape_product(&shape)?;
    Ok(TensorDeclaration {
        network,
        name,
        element_type,
        shape,
        kind,
    })
}

/// Parse VNN-LIB format from string content.
pub fn parse_vnnlib(content: &str) -> Result<VnnLibSpec> {
    let cleaned_content = strip_vnnlib_comments(content);
    let raw_content = cleaned_content.as_str();
    let mut spec = VnnLibSpec::new();
    let mut input_lower: HashMap<usize, f64> = HashMap::new();
    let mut input_upper: HashMap<usize, f64> = HashMap::new();
    let mut input_declared: HashMap<String, TensorDecl> = HashMap::new();
    let mut output_declared: HashMap<String, TensorDecl> = HashMap::new();
    let mut max_input_idx = 0;
    let mut max_output_idx = 0;
    let mut v2_mode = false;
    let collect_output_constraints = false;

    // Parse S-expressions
    let tokens = tokenize(raw_content)?;
    let exprs = parse_expressions(&tokens)?;
    let dual_network = parse_dual_network_spec(&exprs)?;

    for expr in &exprs {
        if let Expr::List(items) = expr {
            if items.is_empty() {
                continue;
            }

            match items.first() {
                Some(Expr::Symbol(s)) if s == "vnnlib-version" => {
                    // (vnnlib-version 2.0)
                    if items.len() >= 2 {
                        let version_str = match &items[1] {
                            Expr::Number(n) => {
                                // Preserve decimal point for version numbers like 1.0, 2.0
                                let formatted = format!("{}", n);
                                if formatted.contains('.') {
                                    formatted
                                } else {
                                    format!("{}.0", formatted)
                                }
                            }
                            Expr::Symbol(s) => s.clone(),
                            _ => "unknown".to_string(),
                        };
                        spec.version = Some(version_str.clone());

                        // Warn if version is 2.0 or higher (partial support)
                        if let Ok(major_minor) = version_str.parse::<f64>() {
                            if major_minor >= 2.0 {
                                v2_mode = true;
                                warn!(
                                    "VNN-LIB version {} detected. This parser partially supports VNN-LIB 2.0 \
                                     (declare-input/declare-output and tensor indexing). declare-network \
                                     wrappers are parsed but network metadata is ignored; non-linear arithmetic \
                                     expressions remain unsupported. Unsupported constructs may fail.",
                                    version_str
                                );
                            }
                        }
                    }
                }
                Some(Expr::Symbol(s)) if s == "declare-const" => {
                    // (declare-const X_0 Real)
                    if items.len() >= 3 {
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
                }
                Some(Expr::Symbol(s))
                    if s == "declare-input" || s == "declare-output" || s == "declare-hidden" =>
                {
                    v2_mode = true;
                    apply_tensor_decl(
                        s,
                        items,
                        &mut input_declared,
                        &mut output_declared,
                        &mut max_input_idx,
                        &mut max_output_idx,
                    )?;
                }
                Some(Expr::Symbol(s)) if s == "declare-network" => {
                    v2_mode = true;
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
                        let Some(Expr::Symbol(op)) = nested_items.first() else {
                            return Err(NyError::InvalidSpec(
                                "declare-network contains invalid entry".to_string(),
                            ));
                        };
                        if op == "declare-input" || op == "declare-output" || op == "declare-hidden"
                        {
                            apply_tensor_decl(
                                op,
                                nested_items,
                                &mut input_declared,
                                &mut output_declared,
                                &mut max_input_idx,
                                &mut max_output_idx,
                            )?;
                        } else if op != "isomorphic-to" && op != "equal-to" && op != "ground-truth"
                        {
                            warn!("Ignoring unsupported declare-network entry '{}'", op);
                        }
                    }
                }
                Some(Expr::Symbol(s)) if s == "assert" => {
                    // (assert (<= X_0 0.5))
                    if items.len() >= 2 {
                        parse_assert(
                            &items[1],
                            &mut input_lower,
                            &mut input_upper,
                            &mut spec,
                            &input_declared,
                            &output_declared,
                            true,
                            false,
                            v2_mode,
                            collect_output_constraints,
                            dual_network.as_ref(),
                        )?;
                    }
                }
                _ => {
                    // Skip unknown expressions
                    tracing::debug!("Skipping unknown expression: {:?}", items.first());
                }
            }
        }
    }

    // Build input bounds from collected constraints
    spec.num_inputs = max_input_idx;
    spec.num_outputs = max_output_idx;
    spec.input_bounds = Vec::new();
    spec.input_bounds
        .try_reserve_exact(max_input_idx)
        .map_err(|_| {
            NyError::InvalidSpec(format!(
                "unable to reserve input bounds for {max_input_idx} declared scalars"
            ))
        })?;

    for i in 0..max_input_idx {
        let lower = input_lower.get(&i).copied().unwrap_or(f64::NEG_INFINITY);
        let upper = input_upper.get(&i).copied().unwrap_or(f64::INFINITY);
        spec.input_bounds.push((lower, upper));
    }
    // Retain the UN-WIDENED top-level bounds (disjunct-scoped atoms never reach
    // `input_lower`/`input_upper` — see `parse_assert`). The per-clause union
    // widening below overwrites `input_bounds` for the verification domain, but
    // a declared global assert constrains EVERY clause, so witness-membership
    // gates must still be able to enforce these exact values.
    spec.declared_input_bounds = Vec::new();
    spec.declared_input_bounds
        .try_reserve_exact(spec.input_bounds.len())
        .map_err(|_| {
            NyError::InvalidSpec(format!(
                "unable to reserve declared input bounds for {} scalars",
                spec.input_bounds.len()
            ))
        })?;
    spec.declared_input_bounds
        .extend_from_slice(&spec.input_bounds);

    spec.dual_network = dual_network;

    if spec.dual_network.is_none() {
        apply_normalized_output_constraints(&mut spec, raw_content)?;
    }

    // Validate that all output constraint indices are within [0, num_outputs).
    // This catches malformed VNN-LIB specs at the system boundary, preventing
    // downstream panics or silently weakened objectives. (#1886)
    spec.validate_output_indices()?;

    // Reject NaN / inverted input bounds at the VNN-LIB boundary so the error
    // names the offending `X_i` variable instead of a downstream numeric index
    // (#2800). NaN literals in `assert` propagate into the stored bounds here.
    spec.validate_input_bounds()?;

    info!(
        "Parsed VNN-LIB: {} inputs, {} outputs, {} constraints",
        spec.num_inputs,
        spec.num_outputs,
        spec.output_constraints.len()
    );

    Ok(spec)
}

fn apply_normalized_output_constraints(spec: &mut VnnLibSpec, content: &str) -> Result<()> {
    let normalized =
        normalize::normalize_output_constraints(content, normalize::NormalizeOptions::default())?;
    if normalized.clauses.is_empty() {
        return Ok(());
    }
    let (clauses, is_disjunction) = normalize::to_output_constraint_clauses(&normalized)?;
    spec.output_constraints = clauses.iter().flatten().cloned().collect();
    spec.output_constraint_clauses = clauses;
    spec.is_disjunction = is_disjunction;

    // Apply per-clause input bounds from mixed and clauses (e.g., nn4sys lindex).
    // Also tighten global input_bounds to the union of all per-clause bounds so
    // the verification domain covers every clause's input region.
    let has_per_clause = normalized
        .per_clause_input_bounds
        .iter()
        .any(|b| !b.is_empty());
    if has_per_clause {
        spec.per_clause_input_bounds = normalized.per_clause_input_bounds;

        // Compute union of all per-clause bounds to set/tighten global bounds.
        for clause_bounds in &spec.per_clause_input_bounds {
            for (&idx, &(lower, upper)) in clause_bounds {
                if idx < spec.input_bounds.len() {
                    let global = &mut spec.input_bounds[idx];
                    // Union: widen global bounds to include this clause's range
                    if global.0 == f64::NEG_INFINITY && global.1 == f64::INFINITY {
                        // Global bounds are unbounded — set to first clause's bounds
                        *global = (lower, upper);
                    } else {
                        global.0 = global.0.min(lower);
                        global.1 = global.1.max(upper);
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn parse_dual_network_spec(exprs: &[Expr]) -> Result<Option<DualNetworkSpec>> {
    let mut networks = Vec::new();
    for expr in exprs {
        let Expr::List(items) = expr else { continue };
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        if op != "declare-network" || items.len() < 2 {
            continue;
        }
        let Some(Expr::Symbol(network_name)) = items.get(1) else {
            continue;
        };

        let mut input = None;
        let mut output = None;
        let mut input_dim = None;
        let mut output_dim = None;
        let mut input_type = None;
        let mut output_type = None;
        let mut input_shape = None;
        let mut output_shape = None;
        let mut relation_to = None;

        for nested in items.iter().skip(2) {
            let Expr::List(nested_items) = nested else {
                continue;
            };
            let Some(Expr::Symbol(nested_op)) = nested_items.first() else {
                continue;
            };
            match nested_op.as_str() {
                "declare-input" => {
                    let (name, shape) = parse_tensor_decl(nested_items, "declare-input")?;
                    if let Some(name) = name {
                        input_type = match nested_items.get(2) {
                            Some(Expr::Symbol(value)) => Some(value.clone()),
                            _ => None,
                        };
                        input_shape = shape.clone();
                        input = Some(name);
                        input_dim = Some(tensor_shape_product(shape.as_deref().unwrap_or(&[]))?);
                    }
                }
                "declare-output" => {
                    let (name, shape) = parse_tensor_decl(nested_items, "declare-output")?;
                    if let Some(name) = name {
                        output_type = match nested_items.get(2) {
                            Some(Expr::Symbol(value)) => Some(value.clone()),
                            _ => None,
                        };
                        output_shape = shape.clone();
                        output = Some(name);
                        output_dim = Some(tensor_shape_product(shape.as_deref().unwrap_or(&[]))?);
                    }
                }
                "isomorphic-to" | "equal-to" => {
                    let Some(Expr::Symbol(target)) = nested_items.get(1) else {
                        return Err(NyError::InvalidSpec(format!(
                            "{} requires a target network name",
                            nested_op
                        )));
                    };
                    let relation = if nested_op == "isomorphic-to" {
                        NetworkRelation::IsomorphicTo
                    } else {
                        NetworkRelation::EqualTo
                    };
                    relation_to = Some((relation, target.clone()));
                }
                "ground-truth" => {
                    // (ground-truth "cyl.gt.json") — the sidecar path/ref, as a
                    // quoted string (quotes stripped) or bare symbol. The
                    // relation target is the counterpart network, filled in
                    // once both declarations have been collected.
                    let Some(Expr::Symbol(path)) = nested_items.get(1) else {
                        return Err(NyError::InvalidSpec(
                            "ground-truth requires a .gt.json path or reference".to_string(),
                        ));
                    };
                    let path = path
                        .strip_prefix('"')
                        .and_then(|p| p.strip_suffix('"'))
                        .unwrap_or(path)
                        .to_string();
                    if path.is_empty() {
                        return Err(NyError::InvalidSpec(
                            "ground-truth path must be non-empty".to_string(),
                        ));
                    }
                    relation_to = Some((NetworkRelation::GroundTruth(path), String::new()));
                }
                _ => {}
            }
        }

        if let (
            Some(input),
            Some(output),
            Some(input_type),
            Some(output_type),
            Some(input_shape),
            Some(output_shape),
            Some(input_dim),
            Some(output_dim),
        ) = (
            input,
            output,
            input_type,
            output_type,
            input_shape,
            output_shape,
            input_dim,
            output_dim,
        ) {
            networks.push(DeclaredNetwork {
                name: network_name.clone(),
                input,
                output,
                input_type,
                output_type,
                input_shape,
                output_shape,
                input_dim,
                output_dim,
                relation_to,
            });
        }
    }

    if networks.len() != 2 {
        return Ok(None);
    }
    // A ground-truth declaration names its sidecar, not a target network; the
    // implicit relation target is the only other declared network.
    let names = [networks[0].name.clone(), networks[1].name.clone()];
    for (idx, network) in networks.iter_mut().enumerate() {
        if let Some((NetworkRelation::GroundTruth(_), target)) = &mut network.relation_to {
            if target.is_empty() {
                target.clone_from(&names[1 - idx]);
            }
        }
    }
    let f = networks
        .iter()
        .find(|n| n.name == "f")
        .unwrap_or(&networks[0]);
    let g = networks
        .iter()
        .find(|n| n.name == "g")
        .unwrap_or(&networks[1]);
    if f.input_dim != g.input_dim || f.output_dim != g.output_dim {
        return Err(NyError::InvalidSpec(
            "dual-network inputs/outputs must have matching dimensions".to_string(),
        ));
    }

    let mut f_bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); f.input_dim];
    let mut g_bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); g.input_dim];
    let mut epsilon = None;
    let mut monotonic_output = None;
    let mut strict_unsafe = false;
    let mut dominance_unsafe: Option<bool> = None;
    let mut input_equalities = vec![false; f.input_dim];
    let mut f_input_ge_g_input = vec![false; f.input_dim];
    let mut g_input_ge_f_input = vec![false; f.input_dim];
    let mut isomorphic_outputs = IsomorphicOutputValidation::new(f.output_dim);
    let mut monotonic_output_relation_count = 0usize;
    let mut unsupported_output_relation = false;

    let relation = g
        .relation_to
        .as_ref()
        .map(|(relation, _)| relation)
        .or_else(|| f.relation_to.as_ref().map(|(relation, _)| relation));

    for expr in exprs {
        let Expr::List(items) = expr else { continue };
        if !matches!(items.first(), Some(Expr::Symbol(op)) if op == "assert") {
            continue;
        }
        let Some(assert_expr) = items.get(1) else {
            continue;
        };
        collect_dual_input_bounds(assert_expr, &f.input, &mut f_bounds)?;
        collect_dual_input_bounds(assert_expr, &g.input, &mut g_bounds)?;
        collect_dual_input_equalities(assert_expr, &f.input, &g.input, &mut input_equalities)?;
        collect_dual_input_orderings(
            assert_expr,
            &f.input,
            &g.input,
            &mut f_input_ge_g_input,
            &mut g_input_ge_f_input,
        )?;
        match relation {
            Some(NetworkRelation::IsomorphicTo) => {
                if epsilon.is_none() {
                    epsilon =
                        extract_same_index_epsilon_candidate(assert_expr, &f.output, &g.output)?;
                }
                collect_isomorphic_output_relations(
                    assert_expr,
                    &f.output,
                    &g.output,
                    &mut isomorphic_outputs,
                )?;
            }
            Some(NetworkRelation::EqualTo) => {
                if monotonic_output.is_none() {
                    if let Some((idx, strict)) =
                        extract_monotonic_unsafe(assert_expr, &f.output, &g.output)?
                    {
                        monotonic_output = Some(idx);
                        strict_unsafe = strict;
                    }
                }
                if let Some((_, relation_count, unsupported)) =
                    classify_output_comparisons(assert_expr, &f.output, &g.output)?
                {
                    monotonic_output_relation_count += relation_count;
                    unsupported_output_relation |= unsupported;
                }
            }
            Some(NetworkRelation::GroundTruth(_)) => {
                // Dominance shares the monotonic relation's unsafe shape: one
                // same-index `Y_f ⋖ Y_g` comparison (`<` strict / `<=`
                // non-strict), whose safe complement is `f − g ≥ 0`.
                if dominance_unsafe.is_none() {
                    if let Some((_, strict)) =
                        extract_monotonic_unsafe(assert_expr, &f.output, &g.output)?
                    {
                        dominance_unsafe = Some(strict);
                    }
                }
                if let Some((_, relation_count, unsupported)) =
                    classify_output_comparisons(assert_expr, &f.output, &g.output)?
                {
                    monotonic_output_relation_count += relation_count;
                    unsupported_output_relation |= unsupported;
                }
            }
            None => {}
        }
    }

    if epsilon.is_none() {
        epsilon = isomorphic_outputs.epsilon;
    }

    let property = match relation {
        Some(NetworkRelation::IsomorphicTo) => DualNetworkProperty::EpsilonEquivalence {
            epsilon: match epsilon {
                Some(epsilon) => epsilon,
                None => return Ok(None),
            },
        },
        Some(NetworkRelation::EqualTo) => DualNetworkProperty::MonotonicGreaterEq {
            output: match monotonic_output {
                Some(output) => output,
                None => return Ok(None),
            },
            varying_input: 0,
            strict_unsafe,
        },
        Some(NetworkRelation::GroundTruth(_)) => DualNetworkProperty::DominatesSecond {
            strict_unsafe: match dominance_unsafe {
                Some(strict) => strict,
                None => return Ok(None),
            },
        },
        None => return Ok(None),
    };

    // When EVERY input is coupled by equality (`X_g[i] == X_f[i]`), both nets
    // read ONE shared input point constrained by BOTH declared boxes — the
    // joint box is the elementwise INTERSECTION. The real 2026 isomorphic
    // instances bound only `X_f` and couple `X_g` purely through the
    // equalities, so `g`'s explicit box is the parser default `[-inf, inf]`
    // and the intersection recovers the finite shared box (the previous
    // `f_bounds == g_bounds` condition never fired there, leaving `g`
    // unbounded and the whole shortcut fail-closed at validate_dual_bounds).
    // Fail-closed: an empty or non-finite intersection keeps the old bounds
    // (and the downstream validation rejects exactly as before).
    let all_inputs_coupled = matches!(
        property,
        DualNetworkProperty::EpsilonEquivalence { .. }
            | DualNetworkProperty::DominatesSecond { .. }
    ) && !input_equalities.is_empty()
        && input_equalities.iter().all(|coupled| *coupled)
        && f_bounds.len() == g_bounds.len();
    let shared_input_coupling = all_inputs_coupled && {
        let inter: Vec<(f64, f64)> = f_bounds
            .iter()
            .zip(g_bounds.iter())
            .map(|(&(fl, fu), &(gl, gu))| (fl.max(gl), fu.min(gu)))
            .collect();
        let ok = inter
            .iter()
            .all(|&(l, u)| l.is_finite() && u.is_finite() && l <= u);
        if ok {
            f_bounds = inter.clone();
            g_bounds = inter;
        }
        ok
    };

    let validation = DualNetworkValidation {
        input_equalities,
        f_input_ge_g_input,
        g_input_ge_f_input,
        // NOTE: the flag deliberately does NOT require a conjunctive structure.
        // The canonical unsafe complement `∃i |Y_g[i]-Y_f[i]| > eps` IS a
        // disjunction (the real 2026 files spell it as nested or-of-ors), and
        // any and/or combination of validated same-index strict deviation
        // atoms is a SUBSET of their union, which the difference-network band
        // proof refutes atom-by-atom. The one consumer that additionally needs
        // a conjunction (the Farkas emptiness shortcut) checks
        // `isomorphic_output_is_conjunction` separately.
        isomorphic_output_safe_complement: isomorphic_outputs.is_safe_complement()
            && !isomorphic_outputs.unsupported,
        monotonic_output_relation_count,
        unsupported_output_relation: unsupported_output_relation || isomorphic_outputs.unsupported,
        isomorphic_output_atoms: isomorphic_outputs.atoms.clone(),
        isomorphic_output_is_conjunction: isomorphic_outputs.is_conjunction,
    };

    // FULL-COVERAGE formula DNF (gate-flip hardening): independent of the
    // canonical-shape flags above; `None` (any inexpressible construct) keeps
    // the relational `unsat` gate down for this instance.
    let formula_dnf = crate::vnnlib::dual_formula::extract_dual_formula_dnf(exprs);

    Ok(Some(DualNetworkSpec {
        networks,
        property,
        shared_input_coupling,
        f_input_bounds: f_bounds,
        g_input_bounds: g_bounds,
        validation,
        formula_dnf,
    }))
}

fn tensor_shape_product(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| NyError::InvalidSpec("Tensor shape size overflow".to_string()))
    })
}

fn comparison_items(expr: &Expr) -> Option<(&str, &Expr, &Expr)> {
    let Expr::List(items) = expr else { return None };
    if items.len() != 3 {
        return None;
    }
    if let Expr::Symbol(op) = &items[0] {
        if matches!(op.as_str(), "<=" | ">=" | "<" | ">" | "=" | "==") {
            return Some((op.as_str(), &items[1], &items[2]));
        }
    }
    if let Expr::Symbol(op) = &items[1] {
        if matches!(op.as_str(), "<=" | ">=" | "<" | ">" | "=" | "==") {
            return Some((op.as_str(), &items[0], &items[2]));
        }
    }
    None
}

fn tensor_ref(expr: &Expr, tensor: &str) -> Result<Option<usize>> {
    let Expr::Symbol(name) = expr else {
        return Ok(None);
    };
    let Some((base, indices)) = parse_tensor_indices(name)? else {
        return Ok(None);
    };
    if base != tensor || indices.is_empty() {
        return Ok(None);
    }
    // Flat element index. Single-index refs (`Y[3]`) are the index itself.
    // MULTI-index refs (the 2026 relational ACAS files: `X_f[0,0,0,4]`,
    // `Y_g[0,3]`) flatten row-major; when every index except the LAST is
    // zero, the flat index equals the last index REGARDLESS of the (unknown
    // here) tensor shape, because the leading terms of `Σ idxᵢ·strideᵢ`
    // vanish and the trailing stride is 1. Any nonzero leading index would
    // need the real shape — decline (fail-closed) as before.
    if indices[..indices.len() - 1].iter().all(|&i| i == 0) {
        Ok(Some(indices[indices.len() - 1]))
    } else {
        Ok(None)
    }
}

fn collect_dual_input_bounds(expr: &Expr, tensor: &str, bounds: &mut [(f64, f64)]) -> Result<()> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if let Some(idx) = tensor_ref(lhs, tensor)? {
            if let Some(value) = get_number(rhs) {
                apply_scalar_bound(bounds, idx, op, value, false)?;
            }
        } else if let Some(idx) = tensor_ref(rhs, tensor)? {
            if let Some(value) = get_number(lhs) {
                apply_scalar_bound(bounds, idx, op, value, true)?;
            }
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            collect_dual_input_bounds(child, tensor, bounds)?;
        }
    }
    Ok(())
}

fn collect_dual_input_equalities(
    expr: &Expr,
    f_input: &str,
    g_input: &str,
    equalities: &mut [bool],
) -> Result<()> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if matches!(op, "=" | "==") {
            let lhs_f = tensor_ref(lhs, f_input)?;
            let rhs_g = tensor_ref(rhs, g_input)?;
            let lhs_g = tensor_ref(lhs, g_input)?;
            let rhs_f = tensor_ref(rhs, f_input)?;
            let matching = match ((lhs_f, rhs_g), (lhs_g, rhs_f)) {
                ((Some(f_idx), Some(g_idx)), _) | (_, (Some(g_idx), Some(f_idx)))
                    if f_idx == g_idx =>
                {
                    Some(f_idx)
                }
                _ => None,
            };
            if let Some(idx) = matching {
                let Some(coupled) = equalities.get_mut(idx) else {
                    return Err(NyError::InvalidSpec(format!(
                        "Input tensor index {idx} out of bounds in dual-network coupling"
                    )));
                };
                *coupled = true;
            }
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            collect_dual_input_equalities(child, f_input, g_input, equalities)?;
        }
    }
    Ok(())
}

fn collect_dual_input_orderings(
    expr: &Expr,
    f_input: &str,
    g_input: &str,
    f_ge_g: &mut [bool],
    g_ge_f: &mut [bool],
) -> Result<()> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        let lhs_f = tensor_ref(lhs, f_input)?;
        let rhs_g = tensor_ref(rhs, g_input)?;
        let lhs_g = tensor_ref(lhs, g_input)?;
        let rhs_f = tensor_ref(rhs, f_input)?;

        if let (Some(f_idx), Some(g_idx)) = (lhs_f, rhs_g) {
            if f_idx == g_idx {
                match op {
                    ">=" | ">" => mark_input_ordering(f_ge_g, f_idx)?,
                    "<=" | "<" => mark_input_ordering(g_ge_f, f_idx)?,
                    _ => {}
                }
            }
        }
        if let (Some(g_idx), Some(f_idx)) = (lhs_g, rhs_f) {
            if f_idx == g_idx {
                match op {
                    ">=" | ">" => mark_input_ordering(g_ge_f, f_idx)?,
                    "<=" | "<" => mark_input_ordering(f_ge_g, f_idx)?,
                    _ => {}
                }
            }
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            collect_dual_input_orderings(child, f_input, g_input, f_ge_g, g_ge_f)?;
        }
    }
    Ok(())
}

fn mark_input_ordering(orderings: &mut [bool], idx: usize) -> Result<()> {
    let Some(ordering) = orderings.get_mut(idx) else {
        return Err(NyError::InvalidSpec(format!(
            "Input tensor index {idx} out of bounds in dual-network ordering"
        )));
    };
    *ordering = true;
    Ok(())
}

fn apply_scalar_bound(
    bounds: &mut [(f64, f64)],
    idx: usize,
    op: &str,
    value: f64,
    reversed: bool,
) -> Result<()> {
    let Some(bound) = bounds.get_mut(idx) else {
        return Err(NyError::InvalidSpec(format!(
            "Input tensor index {idx} out of bounds in dual-network property"
        )));
    };
    let op = if reversed {
        match op {
            "<=" => ">=",
            ">=" => "<=",
            "<" => ">",
            ">" => "<",
            other => other,
        }
    } else {
        op
    };
    match op {
        "<=" | "<" => bound.1 = bound.1.min(value),
        ">=" | ">" => bound.0 = bound.0.max(value),
        "=" | "==" => {
            bound.0 = bound.0.max(value);
            bound.1 = bound.1.min(value);
        }
        _ => {}
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct IsomorphicOutputValidation {
    epsilon: Option<f64>,
    positive: Vec<bool>,
    negative: Vec<bool>,
    unsupported: bool,
    /// Every REAL parsed deviation atom, with its true signed constant. The
    /// Farkas emptiness gate builds its certificate from these (not a template).
    atoms: Vec<IsomorphicOutputAtom>,
    /// Cleared the moment a deviation atom is found inside an `or`: the unsafe
    /// region is then a disjunction and the emptiness shortcut must decline.
    is_conjunction: bool,
}

impl IsomorphicOutputValidation {
    fn new(output_dim: usize) -> Self {
        Self {
            epsilon: None,
            positive: vec![false; output_dim],
            negative: vec![false; output_dim],
            unsupported: false,
            atoms: Vec::new(),
            is_conjunction: true,
        }
    }

    /// Record a canonical difference-form atom `t = Y_g[i] - Y_f[i] ⋈ c`.
    ///
    /// `under_disjunction` is true when the atom syntactically occurs inside an
    /// `or`; such an atom clears `is_conjunction` so the emptiness shortcut
    /// declines (the region `|t| > eps` is feasible, NOT empty).
    ///
    /// The shared-epsilon / positive+negative bookkeeping is derived from the
    /// atom's REAL signed constant (the magnitude is the candidate epsilon, the
    /// sign of the constant the deviation side). Storing the raw signed atom is
    /// what severs the downstream certificate from a `±eps` template.
    fn record(&mut self, atom: IsomorphicOutputAtom, under_disjunction: bool) {
        self.atoms.push(atom);
        if under_disjunction {
            self.is_conjunction = false;
        }
        let idx = atom.index;
        if idx >= self.positive.len() {
            self.unsupported = true;
            return;
        }
        if !atom.constant.is_finite() {
            self.unsupported = true;
            return;
        }
        // The candidate shared epsilon is the magnitude of the signed constant;
        // the deviation side (positive `t > +eps` vs negative `t < -eps`) is the
        // sign. A strict-strict safe complement requires a matching pair per
        // index under ONE shared eps magnitude.
        let epsilon = atom.constant.abs();
        match self.epsilon {
            Some(existing) if existing != epsilon => {
                self.unsupported = true;
                return;
            }
            None => self.epsilon = Some(epsilon),
            _ => {}
        }
        // Map the canonical relation + signed constant to a deviation side:
        //   `t > +eps`  (Gt, c > 0) is the Positive side,
        //   `t < -eps`  (Lt, c < 0) is the Negative side.
        // Any other strict shape (e.g. `t > -eps` / `t < +eps`) is NOT part of
        // the infeasible complement and marks the structure unsupported, so the
        // shortcut declines rather than mislabeling a feasible region.
        match (atom.relation, atom.constant) {
            (IsomorphicAtomRelation::Gt, c) if c > 0.0 => self.positive[idx] = true,
            (IsomorphicAtomRelation::Lt, c) if c < 0.0 => self.negative[idx] = true,
            _ => self.unsupported = true,
        }
    }

    fn is_safe_complement(&self) -> bool {
        self.epsilon.is_some()
            && self.positive.iter().all(|seen| *seen)
            && self.negative.iter().all(|seen| *seen)
    }
}

fn collect_isomorphic_output_relations(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
    validation: &mut IsomorphicOutputValidation,
) -> Result<()> {
    collect_isomorphic_output_relations_inner(expr, f_output, g_output, validation, false)
}

/// Recurse over the assertion tree, recording every deviation atom together
/// with whether it appears under an `or` (`under_disjunction`).
///
/// SOUNDNESS FIX (BUG 2): unlike the previous flat recursion, this distinguishes
/// `and` from `or`. Any output atom reached through an `or` node sets
/// `under_disjunction`, which clears `is_conjunction` in the validation. The
/// emptiness shortcut requires a conjunction (`|t| > eps` disjunctions are
/// feasible and must stay `unknown`).
fn collect_isomorphic_output_relations_inner(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
    validation: &mut IsomorphicOutputValidation,
    under_disjunction: bool,
) -> Result<()> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if let Some(atom) = classify_isomorphic_deviation(op, lhs, rhs, f_output, g_output)? {
            validation.record(atom, under_disjunction);
        } else if output_comparison_mentions_dual_outputs(lhs, rhs, f_output, g_output)? {
            validation.unsupported = true;
        }
    }
    if let Expr::List(items) = expr {
        // A list whose head is `or` introduces a disjunction for all descendant
        // output atoms; `not` similarly inverts the polarity and is treated as a
        // disjunctive (non-conjunctive) context to stay conservative.
        let child_disjunction = under_disjunction
            || matches!(items.first(), Some(Expr::Symbol(head)) if head == "or" || head == "not");
        for child in items {
            collect_isomorphic_output_relations_inner(
                child,
                f_output,
                g_output,
                validation,
                child_disjunction,
            )?;
        }
    }
    Ok(())
}

fn extract_same_index_epsilon_candidate(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<Option<f64>> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if matches!(op, ">" | "<" | ">=" | "<=") {
            if let Some(g_idx) = tensor_ref(lhs, g_output)? {
                if let Some((f_idx, _, eps)) = output_plus_const(rhs, f_output)? {
                    if f_idx == g_idx {
                        return Ok(Some(eps.abs()));
                    }
                }
            }
            if let Some(f_idx) = tensor_ref(lhs, f_output)? {
                if let Some((g_idx, _, eps)) = output_plus_const(rhs, g_output)? {
                    if f_idx == g_idx {
                        return Ok(Some(eps.abs()));
                    }
                }
            }
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            if let Some(eps) = extract_same_index_epsilon_candidate(child, f_output, g_output)? {
                return Ok(Some(eps));
            }
        }
    }
    Ok(None)
}

/// Flip a strict relation `op` under multiplication by `-1` (the algebraic
/// step used when the difference `t = Y_g - Y_f` is expressed with `Y_f` on the
/// left). Returns `None` for relations this parser does not classify.
fn flip_relation(op: &str) -> Option<IsomorphicAtomRelation> {
    match op {
        ">" => Some(IsomorphicAtomRelation::Lt),
        "<" => Some(IsomorphicAtomRelation::Gt),
        _ => None,
    }
}

/// The relation parsed directly from a `>`/`<` operator (no algebraic flip).
fn direct_relation(op: &str) -> Option<IsomorphicAtomRelation> {
    match op {
        ">" => Some(IsomorphicAtomRelation::Gt),
        "<" => Some(IsomorphicAtomRelation::Lt),
        _ => None,
    }
}

/// The signed value of `(arith e)`: `+e` for `+`, `-e` for `-`.
fn signed_const(arith: &str, e: f64) -> Option<f64> {
    match arith {
        "+" => Some(e),
        "-" => Some(-e),
        _ => None,
    }
}

/// Classify a single output comparison into its canonical difference-form atom
/// `t = Y_g[i] - Y_f[i]  ⋈  c`, preserving the REAL signed constant `c`.
///
/// CRITICAL SOUNDNESS FIX: the constant is NOT `.abs()`-normalized. A crafted
/// atom such as `(> Y_g[i] (+ Y_f[i] -0.05))` is `t > -0.05` (feasible at
/// `t = 0`) and is reported with `constant = -0.05`, so the downstream Farkas
/// gate operates on the real region and cannot be tricked into a wrong `unsat`.
fn classify_isomorphic_deviation(
    op: &str,
    lhs: &Expr,
    rhs: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<Option<IsomorphicOutputAtom>> {
    // The difference-network certificate proves the closed safe complement
    // `-eps <= g_i - f_i <= eps`. That is sound only for strict unsafe
    // deviations, so non-strict `>=`/`<=` epsilon violations are declined here.
    if !matches!(op, ">" | "<") {
        return Ok(None);
    }

    // Case A: `Y_g[i] op (Y_f[i] arith e)`  =>  `t op signed(arith,e)`.
    if let Some(g_idx) = tensor_ref(lhs, g_output)? {
        if let Some((f_idx, arith, epsilon)) = output_plus_const(rhs, f_output)? {
            if f_idx != g_idx {
                return Ok(None);
            }
            let (Some(relation), Some(constant)) =
                (direct_relation(op), signed_const(arith, epsilon))
            else {
                return Ok(None);
            };
            return Ok(Some(IsomorphicOutputAtom {
                index: g_idx,
                relation,
                constant,
            }));
        }
    }
    // Case B: `Y_f[i] op (Y_g[i] arith e)`  =>  `-t op signed(arith,e)`
    //   =>  `t flip(op) -signed(arith,e)`.
    if let Some(f_idx) = tensor_ref(lhs, f_output)? {
        if let Some((g_idx, arith, epsilon)) = output_plus_const(rhs, g_output)? {
            if f_idx != g_idx {
                return Ok(None);
            }
            let (Some(relation), Some(c)) = (flip_relation(op), signed_const(arith, epsilon))
            else {
                return Ok(None);
            };
            return Ok(Some(IsomorphicOutputAtom {
                index: f_idx,
                relation,
                constant: -c,
            }));
        }
    }
    // Case C: `(Y_f[i] arith e) op Y_g[i]`  =>  `signed(arith,e) op t`
    //   =>  `t flip(op) signed(arith,e)`.
    if let Some((f_idx, arith, epsilon)) = output_plus_const(lhs, f_output)? {
        if let Some(g_idx) = tensor_ref(rhs, g_output)? {
            if f_idx != g_idx {
                return Ok(None);
            }
            let (Some(relation), Some(constant)) =
                (flip_relation(op), signed_const(arith, epsilon))
            else {
                return Ok(None);
            };
            return Ok(Some(IsomorphicOutputAtom {
                index: g_idx,
                relation,
                constant,
            }));
        }
    }
    // Case D: `(Y_g[i] arith e) op Y_f[i]`  =>  `t op -signed(arith,e)`.
    if let Some((g_idx, arith, epsilon)) = output_plus_const(lhs, g_output)? {
        if let Some(f_idx) = tensor_ref(rhs, f_output)? {
            if f_idx != g_idx {
                return Ok(None);
            }
            let (Some(relation), Some(c)) = (direct_relation(op), signed_const(arith, epsilon))
            else {
                return Ok(None);
            };
            return Ok(Some(IsomorphicOutputAtom {
                index: f_idx,
                relation,
                constant: -c,
            }));
        }
    }

    Ok(None)
}

fn output_plus_const<'a>(expr: &'a Expr, tensor: &str) -> Result<Option<(usize, &'a str, f64)>> {
    let Expr::List(items) = expr else {
        return Ok(None);
    };
    if items.len() != 3 {
        return Ok(None);
    }
    let Some(Expr::Symbol(arith)) = items.first() else {
        return Ok(None);
    };
    if !matches!(arith.as_str(), "+" | "-") {
        return Ok(None);
    }
    let Some(idx) = tensor_ref(&items[1], tensor)? else {
        return Ok(None);
    };
    let Some(epsilon) = get_number(&items[2]) else {
        return Ok(None);
    };
    Ok(Some((idx, arith.as_str(), epsilon)))
}

fn output_comparison_mentions_dual_outputs(
    lhs: &Expr,
    rhs: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<bool> {
    Ok(expr_mentions_tensor(lhs, f_output)?
        || expr_mentions_tensor(lhs, g_output)?
        || expr_mentions_tensor(rhs, f_output)?
        || expr_mentions_tensor(rhs, g_output)?)
}

fn expr_mentions_tensor(expr: &Expr, tensor: &str) -> Result<bool> {
    if tensor_ref(expr, tensor)?.is_some() {
        return Ok(true);
    }
    if let Expr::List(items) = expr {
        for child in items {
            if expr_mentions_tensor(child, tensor)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn classify_output_comparisons(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<Option<(Option<usize>, usize, bool)>> {
    let mut output = None;
    let mut relation_count = 0usize;
    let mut unsupported = false;
    collect_output_comparison_summary(
        expr,
        f_output,
        g_output,
        &mut output,
        &mut relation_count,
        &mut unsupported,
    )?;
    if relation_count == 0 && !unsupported {
        Ok(None)
    } else {
        Ok(Some((output, relation_count, unsupported)))
    }
}

fn collect_output_comparison_summary(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
    output: &mut Option<usize>,
    relation_count: &mut usize,
    unsupported: &mut bool,
) -> Result<()> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if output_comparison_mentions_dual_outputs(lhs, rhs, f_output, g_output)? {
            if let Some((idx, _strict)) =
                monotonic_unsafe_at_current_node(op, lhs, rhs, f_output, g_output)?
            {
                *relation_count += 1;
                match *output {
                    Some(existing) if existing != idx => *unsupported = true,
                    None => *output = Some(idx),
                    _ => {}
                }
            } else if classify_isomorphic_deviation(op, lhs, rhs, f_output, g_output)?.is_none() {
                *unsupported = true;
            }
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            collect_output_comparison_summary(
                child,
                f_output,
                g_output,
                output,
                relation_count,
                unsupported,
            )?;
        }
    }
    Ok(())
}

fn extract_monotonic_unsafe(
    expr: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<Option<(usize, bool)>> {
    if let Some((op, lhs, rhs)) = comparison_items(expr) {
        if let Some(found) = monotonic_unsafe_at_current_node(op, lhs, rhs, f_output, g_output)? {
            return Ok(Some(found));
        }
    }
    if let Expr::List(items) = expr {
        for child in items {
            if let Some(found) = extract_monotonic_unsafe(child, f_output, g_output)? {
                return Ok(Some(found));
            }
        }
    }
    Ok(None)
}

fn monotonic_unsafe_at_current_node(
    op: &str,
    lhs: &Expr,
    rhs: &Expr,
    f_output: &str,
    g_output: &str,
) -> Result<Option<(usize, bool)>> {
    let lhs_f = tensor_ref(lhs, f_output)?;
    let rhs_g = tensor_ref(rhs, g_output)?;
    if matches!(op, "<" | "<=") {
        if let (Some(f_idx), Some(g_idx)) = (lhs_f, rhs_g) {
            if f_idx == g_idx {
                return Ok(Some((f_idx, op == "<")));
            }
        }
    }

    let lhs_g = tensor_ref(lhs, g_output)?;
    let rhs_f = tensor_ref(rhs, f_output)?;
    if matches!(op, ">" | ">=") {
        if let (Some(g_idx), Some(f_idx)) = (lhs_g, rhs_f) {
            if f_idx == g_idx {
                return Ok(Some((f_idx, op == ">")));
            }
        }
    }
    Ok(None)
}

fn unsupported_constraint_expr_reason(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Option<String> {
    match resolve_var_info(expr, input_declared, output_declared) {
        Ok(Some(_)) => return None,
        Ok(None) => {}
        Err(err) => return Some(err.to_string()),
    }
    let Expr::List(items) = expr else {
        return None;
    };
    if items.is_empty() {
        return Some("Unsupported empty list expression in constraint".to_string());
    }
    let head = match &items[0] {
        Expr::Symbol(s) => s.as_str(),
        _ => return Some("Unsupported list expression in constraint".to_string()),
    };
    if matches!(head, "+" | "-" | "*" | "/") {
        return Some(
            "Non-linear arithmetic expressions are not supported in constraints".to_string(),
        );
    }
    if input_declared.contains_key(head) || output_declared.contains_key(head) {
        return Some(format!(
            "Function-style tensor access '{}' is not supported; use {}[idx] syntax",
            head, head
        ));
    }
    Some(format!(
        "Unsupported list expression '{}' in constraint",
        head
    ))
}

fn dual_network_consumes_input_comparison(
    op: &str,
    lhs: &Expr,
    rhs: &Expr,
    dual: Option<&DualNetworkSpec>,
) -> Result<bool> {
    let Some(dual) = dual else {
        return Ok(false);
    };
    let Some((f_input, g_input)) = dual_network_input_pair(dual) else {
        return Ok(false);
    };

    if matches!(op, "=" | "==") && same_index_dual_input_refs(lhs, rhs, f_input, g_input)?.is_some()
    {
        return Ok(true);
    }

    if let DualNetworkProperty::MonotonicGreaterEq { varying_input, .. } = dual.property {
        if let Some(idx) = same_index_f_ge_g_input_relation(op, lhs, rhs, f_input, g_input)? {
            return Ok(idx == varying_input);
        }
    }

    Ok(false)
}

fn dual_network_input_pair(dual: &DualNetworkSpec) -> Option<(&str, &str)> {
    let f = dual
        .networks
        .iter()
        .find(|network| network.name == "f")
        .or_else(|| dual.networks.first())?;
    let g = dual
        .networks
        .iter()
        .find(|network| network.name == "g")
        .or_else(|| dual.networks.iter().find(|network| network.name != f.name))?;
    Some((f.input.as_str(), g.input.as_str()))
}

fn same_index_dual_input_refs(
    lhs: &Expr,
    rhs: &Expr,
    f_input: &str,
    g_input: &str,
) -> Result<Option<usize>> {
    let lhs_f = tensor_ref(lhs, f_input)?;
    let rhs_g = tensor_ref(rhs, g_input)?;
    if let (Some(f_idx), Some(g_idx)) = (lhs_f, rhs_g) {
        if f_idx == g_idx {
            return Ok(Some(f_idx));
        }
    }

    let lhs_g = tensor_ref(lhs, g_input)?;
    let rhs_f = tensor_ref(rhs, f_input)?;
    if let (Some(g_idx), Some(f_idx)) = (lhs_g, rhs_f) {
        if f_idx == g_idx {
            return Ok(Some(f_idx));
        }
    }

    Ok(None)
}

fn same_index_f_ge_g_input_relation(
    op: &str,
    lhs: &Expr,
    rhs: &Expr,
    f_input: &str,
    g_input: &str,
) -> Result<Option<usize>> {
    let lhs_f = tensor_ref(lhs, f_input)?;
    let rhs_g = tensor_ref(rhs, g_input)?;
    if let (Some(f_idx), Some(g_idx)) = (lhs_f, rhs_g) {
        if f_idx == g_idx && matches!(op, ">=" | ">") {
            return Ok(Some(f_idx));
        }
    }

    let lhs_g = tensor_ref(lhs, g_input)?;
    let rhs_f = tensor_ref(rhs, f_input)?;
    if let (Some(g_idx), Some(f_idx)) = (lhs_g, rhs_f) {
        if f_idx == g_idx && matches!(op, "<=" | "<") {
            return Ok(Some(f_idx));
        }
    }

    Ok(None)
}

/// Check if an expression contains output variable constraints (Y_i comparisons).
pub(crate) fn contains_output_constraint(expr: &Expr) -> bool {
    match expr {
        Expr::Symbol(s) => s.starts_with('Y') || s.starts_with("Y_"),
        Expr::List(items) => items.iter().any(contains_output_constraint),
        _ => false,
    }
}

/// Parse an assert expression and update bounds/constraints.
/// `is_top_level` indicates whether this is the first level of output constraints
/// (used to detect disjunctive vs conjunctive property structure).
///
/// `in_disjunction` is true when `expr` sits (at any depth) under an `or`.
/// Input atoms under a disjunction are NOT global constraints — each disjunct
/// bounds ITS OWN region and the property's input domain is their UNION.
/// Intersecting them here (the historical behavior) at best produced an empty
/// interval (ACAS Xu prop_6: `lower 0.111 > upper -0.111` → a parse error →
/// instant sound unknown), and for OVERLAPPING disjunct boxes it silently
/// shrank the verified domain below the union — a false-unsat hazard. The
/// normalizer captures such boxes as `per_clause_input_bounds`, and
/// `apply_normalized_output_constraints` widens the global box to their hull.
#[allow(clippy::too_many_arguments)] // Parser state requires all these parameters
fn parse_assert(
    expr: &Expr,
    input_lower: &mut HashMap<usize, f64>,
    input_upper: &mut HashMap<usize, f64>,
    spec: &mut VnnLibSpec,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
    is_top_level: bool,
    in_disjunction: bool,
    v2_mode: bool,
    collect_output_constraints: bool,
    dual_network: Option<&DualNetworkSpec>,
) -> Result<()> {
    if let Expr::List(items) = expr {
        if items.is_empty() {
            if v2_mode {
                return Err(NyError::InvalidSpec(
                    "Empty constraint expression in VNN-LIB 2.0 assert".to_string(),
                ));
            }
            return Ok(());
        }

        let mut infix_comparison: Option<(&str, &Expr, &Expr)> = None;
        if items.len() == 3 {
            if let Some(Expr::Symbol(s)) = items.get(1) {
                if matches!(s.as_str(), "<=" | ">=" | "<" | ">" | "=" | "==") {
                    infix_comparison = Some((s.as_str(), &items[0], &items[2]));
                }
            }
        }
        let op = match items.first() {
            _ if infix_comparison.is_some() => infix_comparison
                .as_ref()
                .map(|(op, _, _)| *op)
                .unwrap_or(""),
            Some(Expr::Symbol(s)) => s.as_str(),
            _ => {
                return Ok(());
            }
        };

        // Handle OR expressions: (or C1 C2 ... Cn)
        // For OR semantics, unsafe if ANY holds, so SAFE requires ALL violated.
        if op == "or" {
            // Check if this OR contains output constraints (Y_i comparisons)
            // If at top level and contains output constraints, mark as disjunction
            if collect_output_constraints && is_top_level && contains_output_constraint(expr) {
                spec.is_disjunction = true;
            }
            for child in items.iter().skip(1) {
                parse_assert(
                    child,
                    input_lower,
                    input_upper,
                    spec,
                    input_declared,
                    output_declared,
                    false,
                    true, // children of an `or` are disjunct-scoped
                    v2_mode,
                    collect_output_constraints,
                    dual_network,
                )?;
            }
            return Ok(());
        }

        // Handle AND expressions: (and C1 C2 ... Cn)
        // For conjunctive properties, all constraints must hold for unsafe.
        if op == "and" {
            for child in items.iter().skip(1) {
                parse_assert(
                    child,
                    input_lower,
                    input_upper,
                    spec,
                    input_declared,
                    output_declared,
                    false,
                    in_disjunction, // an `and` inherits its enclosing scope
                    v2_mode,
                    collect_output_constraints,
                    dual_network,
                )?;
            }
            return Ok(());
        }

        let is_comparison = matches!(op, "<=" | ">=" | "<" | ">" | "=" | "==");
        if !is_comparison {
            if v2_mode {
                let reason =
                    unsupported_constraint_expr_reason(expr, input_declared, output_declared)
                        .unwrap_or_else(|| {
                            format!(
                                "Unsupported constraint operator '{}' in VNN-LIB 2.0 assert",
                                op
                            )
                        });
                return Err(NyError::InvalidSpec(reason));
            }
            return Ok(());
        }

        // For comparison operators, we need exactly 2 arguments
        if items.len() != 3 {
            if v2_mode {
                return Err(NyError::InvalidSpec(format!(
                    "Comparison constraint '{}' requires exactly 2 operands in VNN-LIB 2.0",
                    op
                )));
            }
            return Ok(());
        }

        // Get the arguments
        let (op, arg1, arg2) = infix_comparison.unwrap_or((op, &items[1], &items[2]));
        let op = if op == "==" { "=" } else { op };

        if v2_mode {
            let lhs = resolve_var_info(arg1, input_declared, output_declared)?;
            let rhs = resolve_var_info(arg2, input_declared, output_declared)?;
            if matches!((lhs, rhs), (Some((_, true)), Some((_, true)))) {
                if dual_network_consumes_input_comparison(op, arg1, arg2, dual_network)? {
                    // The single-network box spec cannot encode dual-network
                    // input relations. Only skip relations that the parsed
                    // dual-network property validates explicitly.
                    return Ok(());
                }
            }
        }

        // Try to parse as input bound
        if let Some((var_idx, is_input)) = resolve_var_info(arg1, input_declared, output_declared)?
        {
            if is_input {
                // Input constraint: X_i op constant
                if let Some(val) = get_number(arg2) {
                    if in_disjunction {
                        // Disjunct-scoped input atom: bounds ONE disjunct's
                        // region, not the global domain (whose true extent is
                        // the disjuncts' UNION — see the doc comment above).
                        // Consumed by the normalizer's per-clause extraction.
                        return Ok(());
                    }
                    match op {
                        "<=" => {
                            // X_i <= val means upper bound
                            // NaN-propagating: IEEE 754 f64::min absorbs NaN (#2813)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        ">=" => {
                            // X_i >= val means lower bound
                            // NaN-propagating: IEEE 754 f64::max absorbs NaN (#2813)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                        }
                        "<" => {
                            // X_i < val means upper bound (exclusive)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        ">" => {
                            // X_i > val means lower bound (exclusive)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                        }
                        "=" => {
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "Unsupported operator '{}' in input constraint (X_{}). Supported: <=, >=, <, >, =",
                                op, var_idx
                            )));
                        }
                    }
                    return Ok(());
                }
            } else {
                // Output constraint involving Y_i
                if let Some((other_idx, other_is_input)) =
                    resolve_var_info(arg2, input_declared, output_declared)?
                {
                    if !other_is_input {
                        if !collect_output_constraints {
                            return Ok(());
                        }
                        // Y_i op Y_j
                        let constraint = match op {
                            "<=" => OutputConstraint::LessEq(var_idx, other_idx),
                            ">=" => OutputConstraint::GreaterEq(var_idx, other_idx),
                            "<" => OutputConstraint::LessThan(var_idx, other_idx),
                            ">" => OutputConstraint::GreaterThan(var_idx, other_idx),
                            "=" => {
                                spec.output_constraints
                                    .push(OutputConstraint::LessEq(var_idx, other_idx));
                                spec.output_constraints
                                    .push(OutputConstraint::GreaterEq(var_idx, other_idx));
                                return Ok(());
                            }
                            _ => {
                                return Err(NyError::InvalidSpec(format!(
                                    "Unsupported operator '{}' in output constraint (Y_{} vs Y_{}). Supported: <=, >=, <, >, =",
                                    op, var_idx, other_idx
                                )));
                            }
                        };
                        spec.output_constraints.push(constraint);
                        return Ok(());
                    }
                } else if let Some(val) = get_number(arg2) {
                    if !collect_output_constraints {
                        return Ok(());
                    }
                    // Y_i op constant
                    let constraint = match op {
                        "<=" => OutputConstraint::LessEqConst(var_idx, val),
                        ">=" => OutputConstraint::GreaterEqConst(var_idx, val),
                        "<" => OutputConstraint::LessThanConst(var_idx, val),
                        ">" => OutputConstraint::GreaterThanConst(var_idx, val),
                        "=" => {
                            spec.output_constraints
                                .push(OutputConstraint::LessEqConst(var_idx, val));
                            spec.output_constraints
                                .push(OutputConstraint::GreaterEqConst(var_idx, val));
                            return Ok(());
                        }
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "Unsupported operator '{}' in output constraint (Y_{} vs const). Supported: <=, >=, <, >, =",
                                op, var_idx
                            )));
                        }
                    };
                    spec.output_constraints.push(constraint);
                    return Ok(());
                }
            }
        }

        // Try reversed form: constant op var
        if let Some((var_idx, is_input)) = resolve_var_info(arg2, input_declared, output_declared)?
        {
            if let Some(val) = get_number(arg1) {
                if is_input {
                    if in_disjunction {
                        // Disjunct-scoped input atom (reversed form) — see above.
                        return Ok(());
                    }
                    // constant op X_i
                    // NaN-propagating min/max: IEEE 754 absorbs NaN (#2813)
                    match op {
                        "<=" => {
                            // val <= X_i means X_i >= val (lower bound)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                        }
                        ">=" => {
                            // val >= X_i means X_i <= val (upper bound)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        "<" => {
                            // val < X_i means X_i > val (lower bound exclusive)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                        }
                        ">" => {
                            // val > X_i means X_i < val (upper bound exclusive)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        "=" => {
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = nan_propagating_max_f64(*l, val))
                                .or_insert(val);
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = nan_propagating_min_f64(*u, val))
                                .or_insert(val);
                        }
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "Unsupported operator '{}' in input constraint (const vs X_{}). Supported: <=, >=, <, >, =",
                                op, var_idx
                            )));
                        }
                    }
                } else {
                    // constant op Y_i
                    if !collect_output_constraints {
                        return Ok(());
                    }
                    let constraint = match op {
                        "<=" => OutputConstraint::GreaterEqConst(var_idx, val),
                        ">=" => OutputConstraint::LessEqConst(var_idx, val),
                        "<" => OutputConstraint::GreaterThanConst(var_idx, val),
                        ">" => OutputConstraint::LessThanConst(var_idx, val),
                        "=" => {
                            spec.output_constraints
                                .push(OutputConstraint::GreaterEqConst(var_idx, val));
                            spec.output_constraints
                                .push(OutputConstraint::LessEqConst(var_idx, val));
                            return Ok(());
                        }
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "Unsupported operator '{}' in reversed output constraint (const vs Y_{}). Supported: <=, >=, <, >, =",
                                op, var_idx
                            )));
                        }
                    };
                    spec.output_constraints.push(constraint);
                }
            }
        }

        if v2_mode && in_disjunction && !contains_output_constraint(expr) {
            // Disjunct-scoped v2 input constraint: not a global bound (see the
            // doc comment above); the normalizer's per-clause extraction owns
            // it. Unsupported output shapes still error via the normalizer.
            return Ok(());
        }
        if v2_mode
            && try_apply_linear_input_constraint(
                expr,
                input_lower,
                input_upper,
                input_declared,
                output_declared,
                dual_network.is_some(),
            )?
        {
            return Ok(());
        }

        if v2_mode {
            if !collect_output_constraints && contains_output_constraint(expr) {
                return Ok(());
            }
            if dual_network.is_some() {
                // Dual-network file: relational atoms (input couplings,
                // cross-network output relations) are not single-network
                // constraints. The dual machinery — the validation flags and
                // the fail-closed full-formula DNF extractor — owns their
                // semantics; the single-network collector tolerates them.
                return Ok(());
            }
            if let Some(reason) =
                unsupported_constraint_expr_reason(arg1, input_declared, output_declared).or_else(
                    || unsupported_constraint_expr_reason(arg2, input_declared, output_declared),
                )
            {
                return Err(NyError::InvalidSpec(reason));
            }
            return Err(NyError::InvalidSpec(
                "Unsupported comparison constraint expression in VNN-LIB 2.0 assert".to_string(),
            ));
        }
    }

    if v2_mode {
        if !collect_output_constraints && contains_output_constraint(expr) {
            return Ok(());
        }
        if dual_network.is_some() {
            return Ok(()); // dual-network relational shape: tolerated (see above)
        }
        return Err(NyError::InvalidSpec(
            "Unsupported constraint expression in VNN-LIB 2.0 assert".to_string(),
        ));
    }

    Ok(())
}

fn try_apply_linear_input_constraint(
    expr: &Expr,
    input_lower: &mut HashMap<usize, f64>,
    input_upper: &mut HashMap<usize, f64>,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
    dual_mode: bool,
) -> Result<bool> {
    let constraint = match normalize::parse_linear_constraint(expr, input_declared, output_declared)
    {
        Ok(constraint) => constraint,
        Err(err) => {
            if let Expr::List(items) = expr {
                if items.len() == 3 {
                    if let Expr::Symbol(op) = &items[0] {
                        if matches!(op.as_str(), "<=" | ">=" | "<" | ">" | "=" | "==") {
                            if let Some(reason) = unsupported_constraint_expr_reason(
                                &items[1],
                                input_declared,
                                output_declared,
                            )
                            .or_else(|| {
                                unsupported_constraint_expr_reason(
                                    &items[2],
                                    input_declared,
                                    output_declared,
                                )
                            }) {
                                return Err(NyError::InvalidSpec(reason));
                            }
                            return Err(NyError::InvalidSpec(
                                "Unsupported comparison constraint expression in VNN-LIB 2.0 assert"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
            return Err(err);
        }
    };
    let mut input_var: Option<(usize, f64)> = None;
    let mut saw_input = false;
    let mut saw_output = false;
    for (var, coeff) in constraint.expr.terms() {
        match var.kind {
            normalize::VarKind::Input => {
                saw_input = true;
                if input_var.is_some() {
                    if dual_mode {
                        // Dual-network relational coupling (`X_f[i] == X_g[i]`,
                        // `X_f[0] >= X_g[0]`, …): NOT a box bound. The dual
                        // machinery (`parse_dual_network_spec` couplings + the
                        // full-formula DNF extractor) owns these semantics;
                        // the standard box collector just skips them.
                        return Ok(false);
                    }
                    return Err(NyError::InvalidSpec(
                        "Input constraints must reference a single variable".to_string(),
                    ));
                }
                input_var = Some((var.index, *coeff));
            }
            normalize::VarKind::Output => {
                saw_output = true;
            }
        }
    }

    if saw_input && saw_output {
        if dual_mode {
            return Ok(false); // relational dual-network atom; not a box bound
        }
        return Err(NyError::InvalidSpec(
            "Constraint mixes input and output variables".to_string(),
        ));
    }
    if saw_output && !saw_input {
        return Ok(false);
    }

    let Some((var_idx, coeff)) = input_var else {
        return Ok(false);
    };
    if coeff.abs() <= f64::EPSILON {
        return Err(NyError::InvalidSpec(
            "Input constraint has zero coefficient".to_string(),
        ));
    }

    // NaN-propagating min/max: IEEE 754 absorbs NaN in f64::min/max (#2813)
    let bound = -constraint.expr.constant_term() / coeff;
    match constraint.relation {
        normalize::Relation::LessEq => {
            if coeff > 0.0 {
                input_upper
                    .entry(var_idx)
                    .and_modify(|u| *u = nan_propagating_min_f64(*u, bound))
                    .or_insert(bound);
            } else {
                input_lower
                    .entry(var_idx)
                    .and_modify(|l| *l = nan_propagating_max_f64(*l, bound))
                    .or_insert(bound);
            }
        }
        normalize::Relation::GreaterEq => {
            if coeff > 0.0 {
                input_lower
                    .entry(var_idx)
                    .and_modify(|l| *l = nan_propagating_max_f64(*l, bound))
                    .or_insert(bound);
            } else {
                input_upper
                    .entry(var_idx)
                    .and_modify(|u| *u = nan_propagating_min_f64(*u, bound))
                    .or_insert(bound);
            }
        }
        normalize::Relation::Equal => {
            input_lower
                .entry(var_idx)
                .and_modify(|l| *l = nan_propagating_max_f64(*l, bound))
                .or_insert(bound);
            input_upper
                .entry(var_idx)
                .and_modify(|u| *u = nan_propagating_min_f64(*u, bound))
                .or_insert(bound);
        }
    }
    Ok(true)
}
