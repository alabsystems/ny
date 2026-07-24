// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Contract tests: runtime ny module exports must match ny.pyi stub.
//!
//! Prevents stub/runtime drift that caused API breakage in #1828.
//! Design: designs/2026-03-10-issue-1829-ny-python-stub-contract.md

use crate::verify::{HeuristicUsed, SoundnessProvenance, VerifyResult, VerifyStatus};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::BTreeSet;

/// Read the ny.pyi stub source from the crate directory.
fn read_stub_source() -> String {
    let stub_path = format!("{}/ny.pyi", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&stub_path)
        .unwrap_or_else(|e| panic!("Failed to read {stub_path}: {e}"))
}

/// Collect public top-level names from the runtime ny module.
fn runtime_public_names(py: Python<'_>) -> PyResult<BTreeSet<String>> {
    let module = PyModule::new(py, "ny")?;
    crate::ny(&module)?;

    let dir_names: Vec<String> = module.dir()?.extract()?;
    let mut names = BTreeSet::new();
    for name in dir_names {
        // Keep __version__ but skip other dunders
        if name.starts_with("__") && name.ends_with("__") && name != "__version__" {
            continue;
        }
        names.insert(name);
    }
    Ok(names)
}

/// Parse ny.pyi and collect public top-level names using Python's ast module.
fn stub_public_names(py: Python<'_>) -> PyResult<BTreeSet<String>> {
    let stub_source = read_stub_source();

    let parser = PyModule::from_code(
        py,
        c"
import ast

def parse_top_level(source):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            names.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith('_'):
                names.append(node.name)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if name == '__version__':
                    names.append(name)
    return names
",
        c"stub_parser.py",
        c"stub_parser",
    )?;

    let result: Vec<String> = parser
        .call_method1("parse_top_level", (stub_source,))?
        .extract()?;

    Ok(result.into_iter().collect())
}

/// Extract member names (attributes and public methods) from a class in ny.pyi.
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

/// Extract a selected function parameter/default pair from ny.pyi.
fn stub_function_parameter_contract(
    py: Python<'_>,
    function_name: &str,
    parameter_name: &str,
) -> PyResult<(bool, bool, Option<String>)> {
    let stub_source = read_stub_source();

    let parser = PyModule::from_code(
        py,
        c"
import ast

def parse_function_parameter_contract(source, function_name, parameter_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            arg_names = [arg.arg for arg in node.args.args]
            if parameter_name not in arg_names:
                return (False, False, None)
            defaults = [None] * (len(arg_names) - len(node.args.defaults)) + list(node.args.defaults)
            default_node = defaults[arg_names.index(parameter_name)]
            if default_node is None:
                return (True, False, None)
            if isinstance(default_node, ast.Constant):
                if default_node.value is None:
                    return (True, True, None)
                if isinstance(default_node.value, str):
                    return (True, True, default_node.value)
                return (True, True, repr(default_node.value))
            raise AssertionError(
                f'Unsupported default expression for {function_name}.{parameter_name}: '
                f'{ast.dump(default_node)}'
            )
    raise KeyError(f'Function not found in ny.pyi: {function_name}')
",
        c"stub_signature_parser.py",
        c"stub_signature_parser",
    )?;

    parser
        .call_method1(
            "parse_function_parameter_contract",
            (stub_source, function_name, parameter_name),
        )?
        .extract()
}

/// Extract a selected runtime function parameter/default pair via inspect.signature.
fn runtime_function_parameter_contract(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    function_name: &str,
    parameter_name: &str,
) -> PyResult<(bool, bool, Option<String>)> {
    let inspector = PyModule::from_code(
        py,
        c"
import inspect

def runtime_function_parameter_contract(module, function_name, parameter_name):
    function = getattr(module, function_name)
    parameters = inspect.signature(function).parameters
    if parameter_name not in parameters:
        return (False, False, None)
    default = parameters[parameter_name].default
    if default is inspect._empty:
        return (True, False, None)
    if default is None:
        return (True, True, None)
    if isinstance(default, str):
        return (True, True, default)
    return (True, True, repr(default))
",
        c"runtime_signature_parser.py",
        c"runtime_signature_parser",
    )?;

    inspector
        .call_method1(
            "runtime_function_parameter_contract",
            (module, function_name, parameter_name),
        )?
        .extract()
}

fn assert_parameter_contract_matches(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    function_name: &str,
    parameter_name: &str,
    expected_default: Option<&str>,
) {
    let (runtime_has_parameter, runtime_has_default, runtime_default) =
        runtime_function_parameter_contract(py, module, function_name, parameter_name)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to inspect runtime signature for {function_name}.{parameter_name}: {e}"
                )
            });
    assert!(
        runtime_has_parameter,
        "Runtime function {function_name} is missing the {parameter_name} kwarg"
    );
    assert!(
        runtime_has_default,
        "Runtime function {function_name} should expose a default for {parameter_name}"
    );
    assert_eq!(
        runtime_default.as_deref(),
        expected_default,
        "Runtime function {function_name} should expose {parameter_name}={expected_default:?}, got {runtime_default:?}"
    );

    let (stub_has_parameter, stub_has_default, stub_default) =
        stub_function_parameter_contract(py, function_name, parameter_name).unwrap_or_else(|e| {
            panic!("Failed to parse ny.pyi signature for {function_name}.{parameter_name}: {e}")
        });
    assert!(
        stub_has_parameter,
        "ny.pyi function {function_name} is missing the {parameter_name} kwarg"
    );
    assert!(
        stub_has_default,
        "ny.pyi function {function_name} should expose a default for {parameter_name}"
    );
    assert_eq!(
        stub_default.as_deref(),
        expected_default,
        "ny.pyi function {function_name} should expose {parameter_name}={expected_default:?}, got {stub_default:?}"
    );
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
        "Runtime {type_name}.{field} should remain part of the Python contract"
    );
}

fn assert_runtime_attribute_absent(instance: &Bound<'_, PyAny>, type_name: &str, field: &str) {
    assert!(
        !instance
            .hasattr(field)
            .unwrap_or_else(|e| panic!("Failed to inspect runtime {type_name}.{field}: {e}")),
        "Runtime {type_name} should not expose attribute '{field}'"
    );
}

fn sample_heuristic_used() -> HeuristicUsed {
    HeuristicUsed {
        type_: "instancenorm_forward_mode".to_string(),
        num_nodes: Some(7),
    }
}

fn sample_soundness_provenance() -> SoundnessProvenance {
    SoundnessProvenance {
        mode: "heuristic".to_string(),
        heuristics_used: vec![sample_heuristic_used()],
    }
}

fn sample_verify_result() -> VerifyResult {
    VerifyResult {
        status: VerifyStatus::Unknown,
        soundness: sample_soundness_provenance(),
        output_bounds: None,
        counterexample: None,
        counterexample_output: None,
        reason: None,
        method: "alpha-crown".to_string(),
        actual_method: None,
        epsilon: 1e-3,
    }
}

fn assert_runtime_heuristic_used_surface(py: Python<'_>) {
    let heuristic = Bound::new(py, sample_heuristic_used())
        .unwrap_or_else(|e| panic!("Failed to build runtime HeuristicUsed sample: {e}"));
    assert_runtime_instance_members(heuristic.as_any(), "HeuristicUsed", &["type", "num_nodes"]);
    assert_runtime_string_attribute(
        heuristic.as_any(),
        "HeuristicUsed",
        "type",
        "instancenorm_forward_mode",
    );
    assert_runtime_attribute_absent(heuristic.as_any(), "HeuristicUsed", "type_");
}

fn assert_runtime_soundness_provenance_surface(py: Python<'_>) {
    let soundness = Bound::new(py, sample_soundness_provenance())
        .unwrap_or_else(|e| panic!("Failed to build runtime SoundnessProvenance sample: {e}"));
    assert_runtime_instance_members(
        soundness.as_any(),
        "SoundnessProvenance",
        &["mode", "heuristics_used"],
    );
    assert_runtime_string_attribute(
        soundness.as_any(),
        "SoundnessProvenance",
        "mode",
        "heuristic",
    );
}

fn assert_runtime_verify_result_surface(py: Python<'_>) {
    let verify_result = Bound::new(py, sample_verify_result())
        .unwrap_or_else(|e| panic!("Failed to build runtime VerifyResult sample: {e}"));
    assert_runtime_instance_members(verify_result.as_any(), "VerifyResult", &["soundness"]);
    let runtime_soundness = verify_result
        .getattr("soundness")
        .unwrap_or_else(|e| panic!("Failed to access runtime VerifyResult.soundness: {e}"));
    assert_runtime_instance_members(
        &runtime_soundness,
        "VerifyResult.soundness",
        &["mode", "heuristics_used"],
    );
}

fn assert_stub_soundness_members(py: Python<'_>) {
    let expected_members: &[(&str, &[&str])] = &[
        ("HeuristicUsed", &["type", "num_nodes"]),
        ("SoundnessProvenance", &["mode", "heuristics_used"]),
        ("VerifyResult", &["soundness"]),
    ];

    for (class_name, required_fields) in expected_members {
        let members = stub_class_members(py, class_name)
            .unwrap_or_else(|e| panic!("Failed to parse {class_name} from ny.pyi: {e}"));

        for field in *required_fields {
            assert!(
                members.contains(&(*field).to_string()),
                "ny.pyi class {class_name} missing required field '{field}' \
                 (found: {members:?}). This field is part of the soundness \
                 metadata surface from #1828."
            );
        }
    }
}

/// Top-level parity: every public name in the runtime must be in ny.pyi and vice versa.
///
/// This is the primary contract guard. It catches:
/// - New exports added to lib.rs but not ny.pyi
/// - Stale entries in ny.pyi for removed exports
/// - PyO3 rename mismatches (e.g. bench vs run_benchmark)
#[test]
fn test_stub_runtime_top_level_parity() {
    Python::initialize();
    Python::attach(|py| {
        let runtime = runtime_public_names(py).expect("failed to collect runtime exports");
        let stub = stub_public_names(py).expect("failed to parse ny.pyi");

        let missing_from_stub: Vec<_> = runtime.difference(&stub).collect();
        let extra_in_stub: Vec<_> = stub.difference(&runtime).collect();

        assert!(
            missing_from_stub.is_empty() && extra_in_stub.is_empty(),
            "ny.pyi / runtime parity violation:\n  \
             In runtime but missing from ny.pyi: {missing_from_stub:?}\n  \
             In ny.pyi but missing from runtime: {extra_in_stub:?}"
        );
    });
}

/// Explicit soundness metadata surface guard for the #1828 regression area.
///
/// These types and fields are the public contract users consume for
/// verification soundness metadata. They must exist in both the runtime
/// module and ny.pyi.
#[test]
fn test_soundness_metadata_surface() {
    Python::initialize();
    Python::attach(|py| {
        // Build runtime module
        let module = PyModule::new(py, "ny").expect("create module");
        crate::ny(&module).expect("init ny");

        // Verify runtime exposes the three soundness types
        for type_name in ["HeuristicUsed", "SoundnessProvenance", "VerifyResult"] {
            module
                .getattr(type_name)
                .unwrap_or_else(|_| panic!("Runtime module missing soundness type: {type_name}"));
        }

        assert_runtime_heuristic_used_surface(py);
        assert_runtime_soundness_provenance_surface(py);
        assert_runtime_verify_result_surface(py);
        assert_stub_soundness_members(py);
    });
}

/// Explicit backend kwarg/default parity guard for the #3674 drift class.
///
/// This stays intentionally narrow: selected public functions only, and only the
/// `backend` kwarg/default contract between the runtime module and ny.pyi.
#[test]
fn test_backend_signature_surface() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ny").expect("create module");
        crate::ny(&module).expect("init ny");

        let expected_backend_contracts = [
            ("verify", "auto"),
            ("verify_bytes", "auto"),
            ("verify_torch", "auto"),
            ("compare", "cpu"),
            ("compare_bytes", "cpu"),
            ("compare_torch", "cpu"),
        ];

        for (function_name, expected_default) in expected_backend_contracts {
            assert_parameter_contract_matches(
                py,
                &module,
                function_name,
                "backend",
                Some(expected_default),
            );
        }
    });
}

/// Explicit batch_size kwarg/default parity guard for the #2689 regression area.
#[test]
fn test_verify_batch_size_signature_surface() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ny").expect("create module");
        crate::ny(&module).expect("init ny");

        for function_name in ["verify", "verify_bytes", "verify_torch"] {
            assert_parameter_contract_matches(py, &module, function_name, "batch_size", None);
        }
    });
}

/// Explicit output_bounds kwarg/default parity guard.
///
/// The verify functions must expose the output specification parameter —
/// without it no property can be expressed and no property verdict is
/// reachable.
#[test]
fn test_verify_output_bounds_signature_surface() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ny").expect("create module");
        crate::ny(&module).expect("init ny");

        for function_name in ["verify", "verify_bytes", "verify_torch"] {
            assert_parameter_contract_matches(py, &module, function_name, "output_bounds", None);
        }
    });
}
