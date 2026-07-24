// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct TensorDecl {
    base: usize,
    shape: Vec<usize>,
}

/// Simple S-expression representation.
#[derive(Debug, Clone)]
pub(crate) enum Expr {
    Symbol(String),
    Number(f64),
    List(Vec<Expr>),
}

/// Strip VNN-LIB comments (semicolon to end-of-line) while preserving strings.
pub(crate) fn strip_vnnlib_comments(content: &str) -> String {
    let mut cleaned = String::with_capacity(content.len());
    let mut in_string = false;

    for line in content.lines() {
        for c in line.chars() {
            if c == '"' {
                in_string = !in_string;
                cleaned.push(c);
                continue;
            }
            if c == ';' && !in_string {
                break;
            }
            cleaned.push(c);
        }
        cleaned.push('\n');
    }

    cleaned
}

/// Tokenize VNN-LIB content.
pub(crate) fn tokenize(content: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_string = false;

    for c in content.chars() {
        if in_string {
            current.push(c);
            if c == '"' {
                tokens.push(current.clone());
                current.clear();
                in_string = false;
            }
        } else {
            match c {
                '(' | ')' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(c.to_string());
                }
                ' ' | '\t' | '\n' | '\r' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                '"' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    current.push(c);
                    in_string = true;
                }
                _ => {
                    current.push(c);
                }
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

/// Parse tokens into S-expressions.
pub(crate) fn parse_expressions(tokens: &[String]) -> Result<Vec<Expr>> {
    let mut exprs = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        let (expr, new_pos) = parse_expr(tokens, pos)?;
        exprs.push(expr);
        pos = new_pos;
    }

    Ok(exprs)
}

/// Parse a single S-expression starting at position.
pub(crate) fn parse_expr(tokens: &[String], pos: usize) -> Result<(Expr, usize)> {
    if pos >= tokens.len() {
        return Err(NyError::ModelLoad("Unexpected end of input".to_string()));
    }

    let token = &tokens[pos];

    if token == "(" {
        // Parse list
        let mut items = Vec::new();
        let mut i = pos + 1;

        while i < tokens.len() && tokens[i] != ")" {
            let (expr, new_pos) = parse_expr(tokens, i)?;
            items.push(expr);
            i = new_pos;
        }

        if i >= tokens.len() {
            return Err(NyError::ModelLoad(
                "Unmatched opening parenthesis".to_string(),
            ));
        }

        Ok((Expr::List(items), i + 1))
    } else if token == ")" {
        Err(NyError::ModelLoad(
            "Unexpected closing parenthesis".to_string(),
        ))
    } else if let Ok(num) = token.parse::<f64>() {
        Ok((Expr::Number(num), pos + 1))
    } else {
        Ok((Expr::Symbol(token.clone()), pos + 1))
    }
}

/// Parse variable index from name like "X_0" or "Y_1".
pub(crate) fn parse_var_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix).and_then(|s| s.parse().ok())
}

pub(crate) fn parse_shape_from_items(
    items: &[Expr],
    start_idx: usize,
) -> Result<Option<Vec<usize>>> {
    let mut started = false;
    let mut tokens: Vec<String> = Vec::new();

    for item in items.iter().skip(start_idx) {
        match item {
            Expr::Symbol(s) => {
                if !started && s.contains('[') {
                    started = true;
                }
                if started {
                    tokens.push(s.clone());
                    if s.contains(']') {
                        break;
                    }
                }
            }
            Expr::Number(n) if started => {
                tokens.push(format!("{}", n));
            }
            _ => {}
        }
    }

    if !started {
        return Ok(None);
    }

    let joined = tokens.join(" ");
    Ok(Some(parse_shape_string(&joined)?))
}

pub(crate) fn parse_shape_string(shape_text: &str) -> Result<Vec<usize>> {
    let start = shape_text
        .find('[')
        .ok_or_else(|| NyError::InvalidSpec("Missing '[' in tensor shape".to_string()))?;
    let end = shape_text
        .rfind(']')
        .ok_or_else(|| NyError::InvalidSpec("Missing ']' in tensor shape".to_string()))?;
    // Fail closed on a malformed `]...[` ordering: without this guard the slice
    // below would be a reversed range and panic on an untrusted property file.
    // `start + 1 > end` is exact — it still admits the valid empty shape "[]"
    // (start=0, end=1 → 1 > 1 is false → empty slice).
    if start + 1 > end {
        return Err(NyError::InvalidSpec(format!(
            "Malformed tensor shape (']' before '['): {shape_text}"
        )));
    }
    let inner = shape_text[start + 1..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    let normalized = inner.replace(',', " ");
    let mut shape = Vec::new();
    for part in normalized.split_whitespace() {
        let dim: usize = part
            .parse()
            .map_err(|_| NyError::InvalidSpec(format!("Invalid tensor dimension '{}'", part)))?;
        shape.push(dim);
    }
    Ok(shape)
}

pub(crate) fn tensor_shape_len(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut total = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Err(NyError::InvalidSpec(
                "Tensor dimension must be non-zero".to_string(),
            ));
        }
        total = total
            .checked_mul(dim)
            .ok_or_else(|| NyError::InvalidSpec("Tensor shape size overflow".to_string()))?;
    }
    Ok(total)
}

pub(crate) fn parse_tensor_decl(
    items: &[Expr],
    kind: &str,
) -> Result<(Option<String>, Option<Vec<usize>>)> {
    let name = match items.get(1) {
        Some(Expr::Symbol(s)) => Some(s.clone()),
        Some(_) => {
            return Err(NyError::InvalidSpec(format!(
                "Invalid {} name in declaration",
                kind
            )))
        }
        None => return Ok((None, None)),
    };
    let shape = parse_shape_from_items(items, 2)?;
    Ok((name, shape))
}

pub(crate) fn apply_tensor_decl(
    kind: &str,
    items: &[Expr],
    input_declared: &mut HashMap<String, TensorDecl>,
    output_declared: &mut HashMap<String, TensorDecl>,
    max_input_idx: &mut usize,
    max_output_idx: &mut usize,
) -> Result<()> {
    let (decl_name, decl_shape) = parse_tensor_decl(items, kind)?;
    if let Some(name) = decl_name {
        let shape = decl_shape.unwrap_or_default();
        let size = tensor_shape_len(&shape)?;
        let is_input = kind == "declare-input";
        let is_output = kind == "declare-output";
        if is_input {
            if input_declared.contains_key(&name) {
                return Err(NyError::InvalidSpec(format!(
                    "Duplicate declare-input for tensor '{}'",
                    name
                )));
            }
            input_declared.insert(
                name,
                TensorDecl {
                    base: *max_input_idx,
                    shape,
                },
            );
            *max_input_idx = (*max_input_idx).saturating_add(size);
        } else if is_output {
            if output_declared.contains_key(&name) {
                return Err(NyError::InvalidSpec(format!(
                    "Duplicate declare-output for tensor '{}'",
                    name
                )));
            }
            output_declared.insert(
                name,
                TensorDecl {
                    base: *max_output_idx,
                    shape,
                },
            );
            *max_output_idx = (*max_output_idx).saturating_add(size);
        } else {
            tracing::debug!("Skipping declare-hidden tensor '{}'", name);
        }
    }
    Ok(())
}

pub(crate) fn parse_tensor_indices(name: &str) -> Result<Option<(String, Vec<usize>)>> {
    let Some(open) = name.find('[') else {
        return Ok(None);
    };
    let base = name[..open].to_string();
    if base.is_empty() {
        return Err(NyError::InvalidSpec(format!(
            "Invalid tensor reference '{}'",
            name
        )));
    }
    let normalized = name[open..].replace("][", ",").replace(['[', ']'], "");
    let mut indices = Vec::new();
    for part in normalized.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let idx: usize = trimmed.parse().map_err(|_| {
            NyError::InvalidSpec(format!("Invalid tensor index '{}' in '{}'", trimmed, name))
        })?;
        indices.push(idx);
    }
    if indices.is_empty() {
        return Err(NyError::InvalidSpec(format!(
            "Missing tensor indices in '{}'",
            name
        )));
    }
    Ok(Some((base, indices)))
}

pub(crate) fn parse_select_indices(expr: &Expr) -> Result<Option<(String, Vec<usize>)>> {
    let Expr::List(items) = expr else {
        return Ok(None);
    };
    if items.is_empty() {
        return Ok(None);
    }
    let Expr::Symbol(op) = &items[0] else {
        return Ok(None);
    };
    if op != "select" {
        return Ok(None);
    }
    let Some(Expr::Symbol(base)) = items.get(1) else {
        return Err(NyError::InvalidSpec(
            "select requires a tensor name".to_string(),
        ));
    };
    if items.len() < 3 {
        return Err(NyError::InvalidSpec(
            "select requires at least one index".to_string(),
        ));
    }
    let mut indices = Vec::new();
    for item in items.iter().skip(2) {
        let idx = match item {
            Expr::Number(n) => {
                if n.fract() != 0.0 {
                    return Err(NyError::InvalidSpec(
                        "select index must be an integer".to_string(),
                    ));
                }
                *n as isize
            }
            Expr::Symbol(s) => s
                .parse::<isize>()
                .map_err(|_| NyError::InvalidSpec(format!("Invalid select index '{}'", s)))?,
            _ => {
                return Err(NyError::InvalidSpec(
                    "select index must be a number".to_string(),
                ))
            }
        };
        if idx < 0 {
            return Err(NyError::InvalidSpec(
                "select index must be non-negative".to_string(),
            ));
        }
        indices.push(idx as usize);
    }
    Ok(Some((base.clone(), indices)))
}

pub(crate) fn flatten_tensor_index(
    name: &str,
    indices: &[usize],
    shape: &[usize],
) -> Result<usize> {
    if shape.is_empty() {
        if indices.len() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "Scalar tensor '{}' expects 1 index, got {}",
                name,
                indices.len()
            )));
        }
        if indices[0] != 0 {
            return Err(NyError::InvalidSpec(format!(
                "Scalar tensor '{}' index must be 0, got {}",
                name, indices[0]
            )));
        }
        return Ok(0);
    }
    if indices.len() != shape.len() {
        return Err(NyError::InvalidSpec(format!(
            "Tensor '{}' expects {} indices, got {}",
            name,
            shape.len(),
            indices.len()
        )));
    }

    let mut stride = 1usize;
    let mut offset = 0usize;
    for (dim, &idx) in shape.iter().rev().zip(indices.iter().rev()) {
        if idx >= *dim {
            return Err(NyError::InvalidSpec(format!(
                "Tensor '{}' index {} out of bounds for dimension {}",
                name, idx, dim
            )));
        }
        offset = offset.saturating_add(idx.saturating_mul(stride));
        stride = stride.saturating_mul(*dim);
    }
    Ok(offset)
}

pub(crate) fn resolve_var_info(
    expr: &Expr,
    input_declared: &HashMap<String, TensorDecl>,
    output_declared: &HashMap<String, TensorDecl>,
) -> Result<Option<(usize, bool)>> {
    match expr {
        Expr::Symbol(name) => {
            if let Some(idx) = parse_var_index(name, "X_") {
                return Ok(Some((idx, true)));
            }
            if let Some(idx) = parse_var_index(name, "Y_") {
                return Ok(Some((idx, false)));
            }

            if let Some((base, indices)) = parse_tensor_indices(name)? {
                if let Some(decl) = input_declared.get(&base) {
                    let offset = flatten_tensor_index(name, &indices, &decl.shape)?;
                    return Ok(Some((decl.base + offset, true)));
                }
                if let Some(decl) = output_declared.get(&base) {
                    let offset = flatten_tensor_index(name, &indices, &decl.shape)?;
                    return Ok(Some((decl.base + offset, false)));
                }
                return Err(NyError::InvalidSpec(format!(
                    "Tensor reference '{}' does not match any declared input/output",
                    name
                )));
            }
        }
        Expr::List(_) => {
            if let Some((base, indices)) = parse_select_indices(expr)? {
                if let Some(decl) = input_declared.get(&base) {
                    let offset = flatten_tensor_index(&base, &indices, &decl.shape)?;
                    return Ok(Some((decl.base + offset, true)));
                }
                if let Some(decl) = output_declared.get(&base) {
                    let offset = flatten_tensor_index(&base, &indices, &decl.shape)?;
                    return Ok(Some((decl.base + offset, false)));
                }
                return Err(NyError::InvalidSpec(format!(
                    "Tensor reference '{}' does not match any declared input/output",
                    base
                )));
            }
        }
        _ => {}
    }

    Ok(None)
}

/// Get number from expression.
pub(crate) fn get_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Number(n) => Some(*n),
        Expr::Symbol(s) => s.parse().ok(),
        _ => None,
    }
}
