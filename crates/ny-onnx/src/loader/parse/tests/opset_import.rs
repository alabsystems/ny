// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::metadata::collect_opset_imports;
use crate::onnx_proto::{GraphProto, ModelProto, NodeProto, OperatorSetIdProto};

fn core_node() -> NodeProto {
    NodeProto {
        input: vec!["x".to_string(), "y".to_string()],
        output: vec!["z".to_string()],
        op_type: "Add".to_string(),
        ..Default::default()
    }
}

fn model_with_imports(opset_import: Vec<OperatorSetIdProto>) -> ModelProto {
    ModelProto {
        opset_import,
        graph: Some(GraphProto {
            node: vec![core_node()],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn collect_opset_imports_aliases_one_explicit_core_authority() {
    let model = model_with_imports(vec![OperatorSetIdProto {
        version: 17,
        domain: "ai.onnx".to_string(),
    }]);
    let opsets = collect_opset_imports(&model).expect("one positive core import is valid");
    assert_eq!(opsets.get("ai.onnx").copied(), Some(17));
    assert_eq!(opsets.get("").copied(), Some(17));
}

#[test]
fn collect_opset_imports_rejects_missing_core_authority_instead_of_guessing() {
    let error = collect_opset_imports(&model_with_imports(vec![]))
        .expect_err("a used core operator set must be explicitly imported");
    assert!(error.to_string().contains("refusing to guess"), "{error}");
}

#[test]
fn collect_opset_imports_rejects_nonpositive_core_versions() {
    for version in [0, -1] {
        let error = collect_opset_imports(&model_with_imports(vec![OperatorSetIdProto {
            version,
            domain: String::new(),
        }]))
        .expect_err("a core opset version must be positive");
        assert!(error.to_string().contains("positive version"), "{error}");
    }
}

/// The empty domain and `ai.onnx` are the SAME domain per the ONNX spec, so a
/// model may legally declare both. At the SAME version that is redundant but
/// unambiguous — every operator resolves to one core opset either way — and it
/// must be ACCEPTED. Rejecting it blocked the whole dist_shift_2023 benchmark
/// (72/72 rows, "versions 11 and 11"), a guaranteed 0 on models ny handles fine.
///
/// Disagreeing versions are a different matter and still fail closed; see
/// `collect_opset_imports_rejects_conflicting_core_alias_versions` below.
#[test]
fn collect_opset_imports_accepts_core_aliases_at_the_same_version() {
    let imports = collect_opset_imports(&model_with_imports(vec![
        OperatorSetIdProto {
            version: 17,
            domain: String::new(),
        },
        OperatorSetIdProto {
            version: 17,
            domain: "ai.onnx".to_string(),
        },
    ]))
    .expect("core aliases at one version are redundant but unambiguous");
    // Both spellings resolve to the one core operator set, and `collect_opset_imports`
    // publishes the resolved version under each alias.
    assert_eq!(imports.get(""), Some(&17), "{imports:?}");
    assert_eq!(imports.get("ai.onnx"), Some(&17), "{imports:?}");
}

#[test]
fn collect_opset_imports_rejects_conflicting_core_alias_versions() {
    let error = collect_opset_imports(&model_with_imports(vec![
        OperatorSetIdProto {
            version: 17,
            domain: String::new(),
        },
        OperatorSetIdProto {
            version: 18,
            domain: "ai.onnx".to_string(),
        },
    ]))
    .expect_err("core aliases cannot select different schemas");
    assert!(error.to_string().contains("conflicting"), "{error}");
}

#[test]
fn collect_opset_imports_rejects_duplicate_custom_domain_authority() {
    for (versions, expected) in [([3, 3], "duplicate"), ([3, 4], "conflicting")] {
        let error = collect_opset_imports(&model_with_imports(vec![
            OperatorSetIdProto {
                version: 17,
                domain: String::new(),
            },
            OperatorSetIdProto {
                version: versions[0],
                domain: "vendor.example".to_string(),
            },
            OperatorSetIdProto {
                version: versions[1],
                domain: "vendor.example".to_string(),
            },
        ]))
        .expect_err("one custom domain cannot declare two operator-set authorities");
        assert!(
            error.to_string().contains(expected) && error.to_string().contains("vendor.example"),
            "{error}"
        );
    }
}

#[test]
fn collect_opset_imports_still_skips_invalid_unused_custom_domain() {
    let model = ModelProto {
        opset_import: vec![OperatorSetIdProto {
            version: 0,
            domain: "custom.domain".to_string(),
        }],
        ..Default::default()
    };
    let opsets = collect_opset_imports(&model).expect("unused custom imports are not core");
    assert!(!opsets.contains_key("custom.domain"));
    assert!(!opsets.contains_key(""));
    assert!(!opsets.contains_key("ai.onnx"));
}
