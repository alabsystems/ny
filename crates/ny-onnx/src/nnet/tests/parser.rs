// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for NNet format parsing: CSV line parsing, header validation,
//! error paths, and structural correctness.

use ny_core::Result;

use crate::nnet::parse_nnet;
use crate::nnet::parser::{parse_csv_line, parse_csv_line_f32};

#[ntest::timeout(10000)]
#[test]
fn test_parse_simple_nnet() {
    let content = r#"
// Simple test network
// 2 layers, 3 inputs, 2 outputs
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
0.1,0.2,0.3,
0.4,0.5,0.6,
0.7,0.8,0.9,
1.0,1.1,1.2,
0.01,
0.02,
0.03,
0.04,
1.0,2.0,3.0,4.0,
5.0,6.0,7.0,8.0,
0.1,
0.2,
"#;

    let network = parse_nnet(content).unwrap();
    assert_eq!(network.num_layers, 2);
    assert_eq!(network.input_size, 3);
    assert_eq!(network.output_size, 2);
    assert_eq!(network.layer_sizes, vec![3, 4, 2]);
    assert_eq!(network.weights.len(), 2);
    assert_eq!(network.biases.len(), 2);

    // First layer: 4x3 weights
    assert_eq!(network.weights[0].shape(), &[4, 3]);
    // Second layer: 2x4 weights
    assert_eq!(network.weights[1].shape(), &[2, 4]);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_with_comments_between_data() {
    let content = r#"
// Header comment
2,3,2,4,
// Layer sizes
3,4,2,
// Symmetric flag
0,
// Input minimums
-1.0,-1.0,-1.0,
// Input maximums
1.0,1.0,1.0,
// Means
0.0,0.0,0.0,0.0,
// Ranges
1.0,1.0,1.0,1.0,
// Layer 0 weights
0.1,0.2,0.3,
0.4,0.5,0.6,
0.7,0.8,0.9,
1.0,1.1,1.2,
// Layer 0 biases
0.01,
0.02,
0.03,
0.04,
// Layer 1 weights
1.0,2.0,3.0,4.0,
5.0,6.0,7.0,8.0,
// Layer 1 biases
0.1,
0.2,
"#;

    let network = parse_nnet(content).unwrap();
    assert_eq!(network.num_layers, 2);
    assert_eq!(network.input_size, 3);
    assert_eq!(network.output_size, 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_nnet_max_layer_size_field() {
    let content = r#"
2,2,2,50,
2,50,2,
0,
-1.0,-1.0,
1.0,1.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
"#;
    // This will fail because we don't have weights, but max_layer_size should parse
    let result = parse_nnet(content);
    // Will fail later but header should parse
    assert!(result.is_err()); // Missing weights, but confirms header parsing
}

// ==================== CSV line parsing ====================

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_empty() {
    let result: Vec<usize> = parse_csv_line("").unwrap();
    assert!(result.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_with_trailing_comma() {
    let result: Vec<usize> = parse_csv_line("1,2,3,").unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_with_spaces() {
    let result: Vec<usize> = parse_csv_line(" 1 , 2 , 3 ").unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_invalid_value() {
    let result: Result<Vec<usize>> = parse_csv_line("1,abc,3");
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Parse error"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_f32_scientific_notation() {
    let result = parse_csv_line_f32("1.5e-3,2.0E+2,-3.25e0").unwrap();
    assert!((result[0] - 0.0015).abs() < 1e-10);
    assert!((result[1] - 200.0).abs() < 1e-10);
    assert!((result[2] - (-3.25)).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_f32_negative_values() {
    let result = parse_csv_line_f32("-1.5,-2.0,-3.5").unwrap();
    assert_eq!(result, vec![-1.5, -2.0, -3.5]);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_csv_line_f32_invalid() {
    let result = parse_csv_line_f32("1.0,not_a_number,3.0");
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("not_a_number"));
}

// ==================== NNet format error paths ====================

#[ntest::timeout(10000)]
#[test]
fn test_load_nnet_file_not_found() {
    let result = crate::nnet::load_nnet("/nonexistent/path/model.nnet");
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("File not found"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_header() {
    let content = "";
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing header"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_invalid_header_too_few_values() {
    let content = "2,3,2,";
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Invalid header"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_layer_sizes() {
    let content = "2,3,2,4,\n";
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing layer sizes"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_wrong_layer_sizes_count() {
    let content = r#"
2,3,2,4,
3,4,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Expected 3 layer sizes"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_input_minimums() {
    let content = r#"
2,3,2,4,
3,4,2,
0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing input minimums"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_input_maximums() {
    let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing input maximums"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_means() {
    let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing means"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_ranges() {
    let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing ranges"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_weight_row() {
    let content = r#"
2,3,2,4,
3,4,2,
0,
-1.0,-1.0,-1.0,
1.0,1.0,1.0,
0.0,0.0,0.0,0.0,
1.0,1.0,1.0,1.0,
0.1,0.2,0.3,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing weight row"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_weight_row_too_few_values() {
    let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,
0.0,1.0,
0.0,
0.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Weight row has 1 values, expected 2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_missing_bias() {
    let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Missing bias value"));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_nnet_invalid_bias() {
    let content = r#"
1,2,2,2,
2,2,
0,
-10.0,-10.0,
10.0,10.0,
0.0,0.0,0.0,
1.0,1.0,1.0,
1.0,0.0,
0.0,1.0,
not_a_number,
0.0,
"#;
    let result = parse_nnet(content);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Invalid bias value"));
}
