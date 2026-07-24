// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Source-text parsing helpers for dispatch coverage tests.
//!
//! These functions parse Rust source (via `include_str!`) to extract
//! `match &node.layer` / `match ctx.layer` / `match layer` dispatch
//! signatures and `Layer::*` variant references without requiring
//! the full Rust AST.

use std::collections::BTreeSet;

#[derive(Debug)]
pub(super) struct DispatchSignature {
    pub(super) explicit_layers: BTreeSet<String>,
}

pub(super) fn extract_dispatch_signature(
    source: &str,
    fn_marker: &str,
    match_index: usize,
) -> DispatchSignature {
    let fn_start = find_function_impl_start(source, fn_marker).unwrap_or_else(|| {
        panic!("dispatch coverage parser: function marker not found: {fn_marker}")
    });
    let fn_open_brace = source[fn_start..]
        .find('{')
        .map(|offset| fn_start + offset)
        .expect("dispatch coverage parser: function opening brace not found");
    let fn_close_brace = find_matching_brace(source, fn_open_brace)
        .expect("dispatch coverage parser: function closing brace not found");
    let fn_body = &source[(fn_open_brace + 1)..fn_close_brace];

    // Support local and shared dispatch variants:
    // - match &node.layer
    // - match ctx.layer
    // - match layer / match layer.propagate_crown_backward(...)
    let match_start = find_dispatch_match_start(fn_body, match_index, fn_marker);
    let match_open_brace = fn_body[match_start..]
        .find('{')
        .map(|offset| match_start + offset)
        .expect("dispatch coverage parser: dispatch match opening brace not found");
    let match_close_brace = find_matching_brace(fn_body, match_open_brace)
        .expect("dispatch coverage parser: dispatch match closing brace not found");
    let match_body = &fn_body[(match_open_brace + 1)..match_close_brace];

    parse_dispatch_match(match_body)
}

fn find_dispatch_match_start(fn_body: &str, match_index: usize, fn_marker: &str) -> usize {
    let mut candidates = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel_idx) = fn_body[search_from..].find("match ") {
        let candidate = search_from + rel_idx;
        let tail = &fn_body[candidate..];
        if tail.starts_with("match &node.layer")
            || tail.starts_with("match ctx.layer")
            || tail.starts_with("match layer")
        {
            candidates.push(candidate);
        }
        search_from = candidate.saturating_add(1);
    }

    let candidate = candidates.get(match_index).unwrap_or_else(|| {
        let body_preview: String = fn_body.chars().take(200).collect();
        panic!(
            "dispatch coverage parser: dispatch match index {match_index} missing \
             (found {found} candidates in function body of {len} bytes). \
             fn_marker: {fn_marker}\n\
             Patterns searched: match &node.layer, match ctx.layer, match layer\n\
             Body preview (first 200 chars):\n{body_preview}",
            found = candidates.len(),
            len = fn_body.len(),
        )
    });
    *candidate
}

pub(super) fn find_function_impl_start(source: &str, fn_marker: &str) -> Option<usize> {
    let mut search_from = 0usize;

    while let Some(rel_idx) = source[search_from..].find(fn_marker) {
        let candidate = search_from + rel_idx;
        let signature_tail = &source[candidate..];
        let open_brace = signature_tail.find('{');
        let semicolon = signature_tail.find(';');

        if let Some(open_idx) = open_brace {
            let brace_before_semicolon = match semicolon {
                Some(semi_idx) => open_idx < semi_idx,
                None => true,
            };
            if brace_before_semicolon {
                return Some(candidate);
            }
        }

        search_from = candidate + fn_marker.len();
    }

    None
}

fn parse_dispatch_match(match_body: &str) -> DispatchSignature {
    let sanitized = strip_comments_and_literals(match_body);
    DispatchSignature {
        explicit_layers: scan_layer_variants(&sanitized),
    }
}

pub(super) fn extract_layer_references(fn_body: &str) -> BTreeSet<String> {
    let sanitized = strip_comments_and_literals(fn_body);
    scan_layer_variants(&sanitized)
}

fn scan_layer_variants(sanitized: &str) -> BTreeSet<String> {
    let mut layers = BTreeSet::new();
    let mut scan_from = 0usize;

    while let Some(rel_idx) = sanitized[scan_from..].find("Layer::") {
        let variant_start = scan_from + rel_idx + "Layer::".len();
        let variant: String = sanitized[variant_start..]
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        if !variant.is_empty() {
            layers.insert(variant);
        }
        scan_from = variant_start.saturating_add(1);
    }

    layers
}

fn strip_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut cleaned = String::with_capacity(source.len());
    let mut i = 0usize;

    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut in_string = false;
    let mut in_char = false;
    let mut escaped = false;

    while i < bytes.len() {
        let ch = bytes[i] as char;
        let next = if i + 1 < bytes.len() {
            bytes[i + 1] as char
        } else {
            '\0'
        };

        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
                cleaned.push('\n');
            } else {
                cleaned.push(' ');
            }
            i += 1;
            continue;
        }

        if block_comment_depth > 0 {
            if ch == '/' && next == '*' {
                block_comment_depth += 1;
                cleaned.push_str("  ");
                i += 2;
                continue;
            }
            if ch == '*' && next == '/' {
                block_comment_depth -= 1;
                cleaned.push_str("  ");
                i += 2;
                continue;
            }
            cleaned.push(if ch == '\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            cleaned.push(if ch == '\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }

        if in_char {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '\'' {
                in_char = false;
            }
            cleaned.push(if ch == '\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }

        if ch == '/' && next == '/' {
            in_line_comment = true;
            cleaned.push_str("  ");
            i += 2;
            continue;
        }
        if ch == '/' && next == '*' {
            block_comment_depth = 1;
            cleaned.push_str("  ");
            i += 2;
            continue;
        }
        if ch == '"' {
            in_string = true;
            cleaned.push(' ');
            i += 1;
            continue;
        }
        if ch == '\'' {
            in_char = true;
            cleaned.push(' ');
            i += 1;
            continue;
        }

        cleaned.push(ch);
        i += 1;
    }

    cleaned
}

pub(super) fn find_matching_brace(source: &str, open_brace_idx: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = open_brace_idx;
    let mut brace_depth = 0usize;

    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut in_string = false;
    let mut in_char = false;
    let mut escaped = false;

    while i < bytes.len() {
        let ch = bytes[i] as char;
        let next = if i + 1 < bytes.len() {
            bytes[i + 1] as char
        } else {
            '\0'
        };

        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }

        if block_comment_depth > 0 {
            if ch == '/' && next == '*' {
                block_comment_depth += 1;
                i += 2;
                continue;
            }
            if ch == '*' && next == '/' {
                block_comment_depth -= 1;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }

        if in_char {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '\'' {
                in_char = false;
            }
            i += 1;
            continue;
        }

        if ch == '/' && next == '/' {
            in_line_comment = true;
            i += 2;
            continue;
        }
        if ch == '/' && next == '*' {
            block_comment_depth = 1;
            i += 2;
            continue;
        }
        if ch == '"' {
            in_string = true;
            i += 1;
            continue;
        }
        if ch == '\'' {
            in_char = true;
            i += 1;
            continue;
        }

        if ch == '{' {
            brace_depth += 1;
        } else if ch == '}' {
            brace_depth = brace_depth.saturating_sub(1);
            if brace_depth == 0 {
                return Some(i);
            }
        }

        i += 1;
    }

    None
}
