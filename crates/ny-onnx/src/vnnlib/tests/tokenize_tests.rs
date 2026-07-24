// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    contains_output_constraint, get_number, parse_expr, parse_expressions, parse_var_index,
    resolve_var_info, tokenize, Expr,
};
use std::collections::HashMap;

#[ntest::timeout(10000)]
#[test]
fn test_tokenize_basic() {
    let tokens = tokenize("(assert (<= X_0 1))").unwrap();
    assert_eq!(tokens, vec!["(", "assert", "(", "<=", "X_0", "1", ")", ")"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_tokenize_with_whitespace() {
    let tokens = tokenize("  (  assert   X_0  )  ").unwrap();
    assert_eq!(tokens, vec!["(", "assert", "X_0", ")"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_tokenize_with_newlines() {
    let tokens = tokenize("(assert\n  X_0\n)").unwrap();
    assert_eq!(tokens, vec!["(", "assert", "X_0", ")"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_tokenize_with_string() {
    let tokens = tokenize("(set-info \"test string\")").unwrap();
    assert_eq!(tokens, vec!["(", "set-info", "\"test string\"", ")"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_tokenize_empty() {
    let tokens = tokenize("").unwrap();
    assert!(tokens.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_expr_unexpected_end() {
    let tokens: Vec<String> = vec![];
    let result = parse_expr(&tokens, 0);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Unexpected end"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_expr_unmatched_open_paren() {
    let tokens = vec!["(".to_string(), "assert".to_string()];
    let result = parse_expr(&tokens, 0);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Unmatched opening"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_expr_unexpected_close_paren() {
    let tokens = vec![")".to_string()];
    let result = parse_expr(&tokens, 0);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Unexpected closing"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_var_index_valid() {
    assert_eq!(parse_var_index("X_0", "X_"), Some(0));
    assert_eq!(parse_var_index("X_42", "X_"), Some(42));
    assert_eq!(parse_var_index("Y_0", "Y_"), Some(0));
    assert_eq!(parse_var_index("Y_123", "Y_"), Some(123));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_var_index_invalid() {
    assert_eq!(parse_var_index("Z_0", "X_"), None);
    assert_eq!(parse_var_index("X_abc", "X_"), None);
    assert_eq!(parse_var_index("X_", "X_"), None);
    assert_eq!(parse_var_index("", "X_"), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_contains_output_constraint_positive() {
    let tokens = tokenize("(<= Y_0 Y_1)").unwrap();
    let exprs = parse_expressions(&tokens).unwrap();
    assert!(contains_output_constraint(&exprs[0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_contains_output_constraint_negative() {
    let tokens = tokenize("(<= X_0 1.0)").unwrap();
    let exprs = parse_expressions(&tokens).unwrap();
    assert!(!contains_output_constraint(&exprs[0]));
}

#[ntest::timeout(10000)]
#[test]
fn test_get_number_from_symbol() {
    // Test that get_number can parse number strings
    let expr = Expr::Symbol("3.125".to_string());
    assert!((get_number(&expr).unwrap() - 3.125).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_number_from_number() {
    let expr = Expr::Number(2.75);
    assert!((get_number(&expr).unwrap() - 2.75).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_number_from_list() {
    let expr = Expr::List(vec![]);
    assert!(get_number(&expr).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_get_var_info_x() {
    let expr = Expr::Symbol("X_5".to_string());
    let input_declared = HashMap::new();
    let output_declared = HashMap::new();
    assert_eq!(
        resolve_var_info(&expr, &input_declared, &output_declared).unwrap(),
        Some((5, true))
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_get_var_info_y() {
    let expr = Expr::Symbol("Y_10".to_string());
    let input_declared = HashMap::new();
    let output_declared = HashMap::new();
    assert_eq!(
        resolve_var_info(&expr, &input_declared, &output_declared).unwrap(),
        Some((10, false))
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_get_var_info_invalid() {
    let expr = Expr::Symbol("Z_0".to_string());
    let input_declared = HashMap::new();
    let output_declared = HashMap::new();
    assert!(resolve_var_info(&expr, &input_declared, &output_declared)
        .unwrap()
        .is_none());

    let expr2 = Expr::Number(1.0);
    assert!(resolve_var_info(&expr2, &input_declared, &output_declared)
        .unwrap()
        .is_none());
}
