// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::shape_infer::DEFAULT_OPSET_VERSION;
use super::super::metadata::collect_opset_imports;
use crate::onnx_proto::{ModelProto, OperatorSetIdProto};

#[test]
fn test_collect_opset_imports_aliases_default_domain() {
    let model = ModelProto {
        opset_import: vec![OperatorSetIdProto {
            version: 17,
            domain: "ai.onnx".to_string(),
        }],
        ..Default::default()
    };
    let opsets = collect_opset_imports(&model);
    assert_eq!(opsets.get("ai.onnx").copied(), Some(17));
    assert_eq!(opsets.get("").copied(), Some(17));
}

#[test]
fn test_collect_opset_imports_fills_missing_default() {
    let model = ModelProto {
        opset_import: Vec::new(),
        ..Default::default()
    };
    let opsets = collect_opset_imports(&model);
    assert_eq!(opsets.get("").copied(), Some(DEFAULT_OPSET_VERSION));
    assert_eq!(opsets.get("ai.onnx").copied(), Some(DEFAULT_OPSET_VERSION));
}

#[test]
fn test_collect_opset_imports_normalizes_zero_version() {
    let model = ModelProto {
        opset_import: vec![OperatorSetIdProto {
            version: 0,
            domain: String::new(),
        }],
        ..Default::default()
    };
    let opsets = collect_opset_imports(&model);
    assert_eq!(opsets.get("").copied(), Some(DEFAULT_OPSET_VERSION));
    assert_eq!(opsets.get("ai.onnx").copied(), Some(DEFAULT_OPSET_VERSION));
}

#[test]
fn test_collect_opset_imports_skips_invalid_custom_domain() {
    let model = ModelProto {
        opset_import: vec![OperatorSetIdProto {
            version: 0,
            domain: "custom.domain".to_string(),
        }],
        ..Default::default()
    };
    let opsets = collect_opset_imports(&model);
    assert!(!opsets.contains_key("custom.domain"));
    assert_eq!(opsets.get("").copied(), Some(DEFAULT_OPSET_VERSION));
    assert_eq!(opsets.get("ai.onnx").copied(), Some(DEFAULT_OPSET_VERSION));
}
