// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the typed consumer-facing Python surface added in #3942.
//!
//! Keeps `ny.pyi` and the runtime module aligned for nested typed fields that
//! top-level export parity alone cannot catch.

use crate::diff::{ModelInfo, TensorSpec};
use crate::weights::{TensorComparison, TensorComparisonStatus};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn read_stub_source() -> String {
    let stub_path = format!("{}/ny.pyi", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&stub_path)
        .unwrap_or_else(|e| panic!("Failed to read {stub_path}: {e}"))
}

fn stub_class_members(py: Python<'_>, class_name: &str) -> PyResult<Vec<String>> {
    let stub_source = read_stub_source();
    let locals = PyDict::new(py);
    locals.set_item("source", stub_source)?;
    locals.set_item("target_class", class_name)?;
    py.run(
        c"
import ast
tree = ast.parse(source)
members = []
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == target_class:
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                members.append(item.target.id)
            elif isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                members.append(item.name)
        break
",
        None,
        Some(&locals),
    )?;

    locals.get_item("members")?.unwrap().extract()
}

fn stub_class_field_annotation(
    py: Python<'_>,
    class_name: &str,
    field_name: &str,
) -> PyResult<Option<String>> {
    let stub_source = read_stub_source();

    let parser = PyModule::from_code(
        py,
        c"
import ast

def parse_class_field_annotation(source, class_name, field_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == field_name
                ):
                    return ast.unparse(item.annotation)
            return None
    raise KeyError(f'Class not found in ny.pyi: {class_name}')
",
        c"typed_consumer_stub_field_parser.py",
        c"typed_consumer_stub_field_parser",
    )?;

    parser
        .call_method1(
            "parse_class_field_annotation",
            (stub_source, class_name, field_name),
        )?
        .extract()
}

fn assert_runtime_instance_members(
    instance: &Bound<'_, PyAny>,
    type_name: &str,
    required_fields: &[&str],
) {
    for field in required_fields {
        instance.getattr(*field).unwrap_or_else(|e| {
            panic!("Failed to access runtime {type_name}.{field}: {e}");
        });
    }
}

fn assert_runtime_string_attribute(
    instance: &Bound<'_, PyAny>,
    type_name: &str,
    field: &str,
    expected: &str,
) {
    assert_eq!(
        instance
            .getattr(field)
            .unwrap_or_else(|e| panic!("Failed to access runtime {type_name}.{field}: {e}"))
            .extract::<String>()
            .unwrap_or_else(|e| {
                panic!("Failed to extract runtime {type_name}.{field} as str: {e}")
            }),
        expected,
        "Runtime {type_name}.{field} should remain part of the typed consumer contract"
    );
}

fn assert_runtime_attribute_type(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    type_name: &str,
    field: &str,
    expected_type_name: &str,
) {
    let locals = PyDict::new(py);
    locals
        .set_item("instance", instance)
        .unwrap_or_else(|e| panic!("Failed to bind runtime {type_name} instance for {field}: {e}"));
    locals
        .set_item("field", field)
        .unwrap_or_else(|e| panic!("Failed to bind runtime field name {type_name}.{field}: {e}"));
    let actual_type_name: String = py
        .eval(
            c"type(getattr(instance, field)).__name__",
            None,
            Some(&locals),
        )
        .unwrap_or_else(|e| {
            panic!("Failed to evaluate runtime {type_name}.{field} Python type: {e}")
        })
        .extract()
        .unwrap_or_else(|e| {
            panic!("Failed to extract runtime {type_name}.{field} Python type name: {e}")
        });

    assert_eq!(
        actual_type_name, expected_type_name,
        "Runtime {type_name}.{field} should expose Python type {expected_type_name}, got {actual_type_name}"
    );
}

fn assert_runtime_first_list_item_type(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    type_name: &str,
    field: &str,
    expected_item_type: &str,
) {
    let locals = PyDict::new(py);
    locals
        .set_item("instance", instance)
        .unwrap_or_else(|e| panic!("Failed to bind runtime {type_name} instance for {field}: {e}"));
    locals
        .set_item("field", field)
        .unwrap_or_else(|e| panic!("Failed to bind runtime field name {type_name}.{field}: {e}"));

    let item_count: usize = py
        .eval(c"len(getattr(instance, field))", None, Some(&locals))
        .unwrap_or_else(|e| panic!("Failed to count runtime {type_name}.{field}: {e}"))
        .extract()
        .unwrap_or_else(|e| panic!("Failed to extract runtime {type_name}.{field} length: {e}"));
    assert!(
        item_count > 0,
        "Runtime {type_name}.{field} should contain at least one typed element"
    );

    let actual_item_type: String = py
        .eval(
            c"type(getattr(instance, field)[0]).__name__",
            None,
            Some(&locals),
        )
        .unwrap_or_else(|e| {
            panic!("Failed to inspect runtime {type_name}.{field}[0] Python type: {e}")
        })
        .extract()
        .unwrap_or_else(|e| {
            panic!("Failed to extract runtime {type_name}.{field}[0] Python type name: {e}")
        });

    assert_eq!(
        actual_item_type, expected_item_type,
        "Runtime {type_name}.{field}[0] should expose Python type {expected_item_type}, got {actual_item_type}"
    );
}

fn sample_tensor_spec(name: &str, shape: &[i64], dtype: &str) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
    }
}

fn sample_model_info() -> ModelInfo {
    ModelInfo {
        inputs: vec![sample_tensor_spec("input", &[1, 4], "float32")],
        outputs: vec![sample_tensor_spec("output", &[1, 2], "float32")],
        layer_count: 2,
        layer_names: vec!["linear".to_string(), "relu".to_string()],
    }
}

fn sample_tensor_comparison() -> TensorComparison {
    TensorComparison {
        name: "weight".to_string(),
        status: TensorComparisonStatus::Match,
        max_diff: Some(0.0),
        shape_a: Some(vec![4, 8]),
        shape_b: Some(vec![4, 8]),
    }
}

fn assert_stub_typed_member_list(py: Python<'_>, class_name: &str, fields: &[&str]) {
    let members = stub_class_members(py, class_name)
        .unwrap_or_else(|e| panic!("Failed to parse {class_name} from ny.pyi: {e}"));
    for field in fields {
        assert!(
            members.contains(&(*field).to_string()),
            "ny.pyi class {class_name} missing required field '{field}' for the #3942 typed consumer surface"
        );
    }
}

fn assert_stub_field_annotation(
    py: Python<'_>,
    class_name: &str,
    field: &str,
    expected_annotation: &str,
) {
    let annotation = stub_class_field_annotation(py, class_name, field).unwrap_or_else(|e| {
        panic!("Failed to parse annotation for ny.pyi {class_name}.{field}: {e}")
    });
    assert_eq!(
        annotation.as_deref(),
        Some(expected_annotation),
        "ny.pyi field {class_name}.{field} should stay annotated as {expected_annotation:?}, got {annotation:?}"
    );
}

fn assert_stub_typed_consumer_members(py: Python<'_>) {
    assert_stub_typed_member_list(py, "TensorSpec", &["name", "shape", "dtype"]);
    assert_stub_typed_member_list(
        py,
        "ModelInfo",
        &["inputs", "outputs", "layer_count", "layer_names"],
    );
    assert_stub_typed_member_list(
        py,
        "TensorComparison",
        &["name", "status", "max_diff", "shape_a", "shape_b"],
    );

    assert_stub_field_annotation(py, "TensorSpec", "name", "str");
    assert_stub_field_annotation(py, "TensorSpec", "shape", "List[int]");
    assert_stub_field_annotation(py, "TensorSpec", "dtype", "str");
    assert_stub_field_annotation(py, "ModelInfo", "inputs", "List[TensorSpec]");
    assert_stub_field_annotation(py, "ModelInfo", "outputs", "List[TensorSpec]");
    assert_stub_field_annotation(py, "ModelInfo", "layer_count", "int");
    assert_stub_field_annotation(py, "ModelInfo", "layer_names", "List[str]");
    assert_stub_field_annotation(py, "TensorComparison", "status", "TensorComparisonStatus");
}

fn assert_runtime_tensor_spec_surface(py: Python<'_>) {
    let tensor_spec = Bound::new(py, sample_tensor_spec("input", &[1, 4], "float32"))
        .unwrap_or_else(|e| panic!("Failed to build runtime TensorSpec sample: {e}"));
    assert_runtime_instance_members(
        tensor_spec.as_any(),
        "TensorSpec",
        &["name", "shape", "dtype"],
    );
    assert_runtime_string_attribute(tensor_spec.as_any(), "TensorSpec", "name", "input");
    assert_runtime_attribute_type(py, tensor_spec.as_any(), "TensorSpec", "shape", "list");
    assert_runtime_string_attribute(tensor_spec.as_any(), "TensorSpec", "dtype", "float32");
}

fn assert_runtime_model_info_surface(py: Python<'_>) {
    let model_info = Bound::new(py, sample_model_info())
        .unwrap_or_else(|e| panic!("Failed to build runtime ModelInfo sample: {e}"));
    assert_runtime_instance_members(
        model_info.as_any(),
        "ModelInfo",
        &["inputs", "outputs", "layer_count", "layer_names"],
    );
    assert_runtime_attribute_type(py, model_info.as_any(), "ModelInfo", "inputs", "list");
    assert_runtime_first_list_item_type(
        py,
        model_info.as_any(),
        "ModelInfo",
        "inputs",
        "TensorSpec",
    );
    assert_runtime_attribute_type(py, model_info.as_any(), "ModelInfo", "outputs", "list");
    assert_runtime_first_list_item_type(
        py,
        model_info.as_any(),
        "ModelInfo",
        "outputs",
        "TensorSpec",
    );
}

fn assert_runtime_tensor_comparison_surface(py: Python<'_>) {
    let comparison = Bound::new(py, sample_tensor_comparison())
        .unwrap_or_else(|e| panic!("Failed to build runtime TensorComparison sample: {e}"));
    assert_runtime_instance_members(
        comparison.as_any(),
        "TensorComparison",
        &["name", "status", "max_diff", "shape_a", "shape_b"],
    );
    assert_runtime_attribute_type(
        py,
        comparison.as_any(),
        "TensorComparison",
        "status",
        "TensorComparisonStatus",
    );
    assert_eq!(
        comparison
            .getattr("status")
            .unwrap_or_else(|e| panic!("Failed to access runtime TensorComparison.status: {e}"))
            .str()
            .unwrap_or_else(|e| panic!("Failed to stringify runtime TensorComparison.status: {e}"))
            .to_string_lossy()
            .as_ref(),
        "match",
        "Runtime TensorComparison.status should stringify as the typed enum value"
    );
}

#[test]
fn test_typed_consumer_surface_contract() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ny").expect("create module");
        crate::ny(&module).expect("init ny");

        for type_name in [
            "TensorSpec",
            "ModelInfo",
            "TensorComparisonStatus",
            "TensorComparison",
        ] {
            module.getattr(type_name).unwrap_or_else(|_| {
                panic!("Runtime module missing typed consumer type: {type_name}")
            });
        }

        assert_runtime_tensor_spec_surface(py);
        assert_runtime_model_info_surface(py);
        assert_runtime_tensor_comparison_surface(py);
        assert_stub_typed_consumer_members(py);
    });
}
