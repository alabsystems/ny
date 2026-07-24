// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use approx::assert_relative_eq;
use ny_onnx::vnnlib::{load_vnnlib, OutputConstraint, VnnLibSpec};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

const PYTHON_HARNESS: &str = r#"
import json
import re
import sys
import vnnlib

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    content = handle.read()

match = re.search(r'\(vnnlib-version\s+([^)]+)\)', content)
version = match.group(1).strip() if match else None

query = vnnlib.parse_query_file(path)
if len(query.networks) != 1:
    raise SystemExit(f"expected 1 network, found {len(query.networks)}")

network = query.networks[0]

def shape_size(shape):
    size = 1
    for dim in shape:
        size *= int(dim)
    return size

input_map = {}
base = 0
for inp in network.inputs:
    shape = list(inp.shape)
    size = shape_size(shape) if shape else 1
    input_map[inp.name] = (base, shape)
    base += size
num_inputs = base

output_map = {}
base = 0
for out in network.outputs:
    shape = list(out.shape)
    size = shape_size(shape) if shape else 1
    output_map[out.name] = (base, shape)
    base += size
num_outputs = base

input_bounds = [[None, None] for _ in range(num_inputs)]
output_constraints = []
is_disjunction = False


def flatten_index(indices, shape):
    if not shape:
        if len(indices) != 1:
            raise ValueError("scalar tensor expects exactly one index")
        return int(indices[0])
    if len(indices) != len(shape):
        raise ValueError("tensor index arity mismatch")
    offset = 0
    stride = 1
    for idx, dim in zip(reversed(indices), reversed(shape)):
        if idx >= dim:
            raise ValueError("tensor index out of bounds")
        offset += idx * stride
        stride *= dim
    return int(offset)


def var_index(var):
    name = var.name
    indices = list(var.indices)
    if var.kind == vnnlib.SymbolKind.Input:
        base_idx, shape = input_map[name]
    elif var.kind == vnnlib.SymbolKind.Output:
        base_idx, shape = output_map[name]
    else:
        raise ValueError("unknown variable kind")
    return base_idx + flatten_index(indices, shape)


def to_number(expr):
    if isinstance(expr, vnnlib.Float):
        return float(expr.value)
    if isinstance(expr, vnnlib.Int):
        return float(expr.value)
    if isinstance(expr, vnnlib.Literal):
        return float(expr.lexeme)
    if isinstance(expr, vnnlib.Negate):
        value = to_number(expr.expr)
        return None if value is None else -value
    return None


def op_kind(expr):
    if isinstance(expr, vnnlib.LessEqual):
        return "le"
    if isinstance(expr, vnnlib.GreaterEqual):
        return "ge"
    if isinstance(expr, vnnlib.LessThan):
        return "lt"
    if isinstance(expr, vnnlib.GreaterThan):
        return "gt"
    return None


def apply_bound(index, kind, value):
    if kind in ("le", "lt"):
        current = input_bounds[index][1]
        input_bounds[index][1] = value if current is None else min(current, value)
    elif kind in ("ge", "gt"):
        current = input_bounds[index][0]
        input_bounds[index][0] = value if current is None else max(current, value)
    else:
        raise ValueError("unknown bound kind")


def handle_comparison(comp):
    kind = op_kind(comp)
    if kind is None:
        raise ValueError(f"unsupported comparison {comp}")

    lhs = comp.lhs
    rhs = comp.rhs
    lhs_var = isinstance(lhs, vnnlib.Var)
    rhs_var = isinstance(rhs, vnnlib.Var)
    lhs_num = to_number(lhs)
    rhs_num = to_number(rhs)

    if lhs_var and rhs_var:
        lhs_idx = var_index(lhs)
        rhs_idx = var_index(rhs)
        if lhs.kind == vnnlib.SymbolKind.Output and rhs.kind == vnnlib.SymbolKind.Output:
            output_constraints.append({"kind": kind, "lhs": lhs_idx, "rhs": rhs_idx})
            return
        raise ValueError("unsupported relational constraint")

    if lhs_var and rhs_num is not None:
        lhs_idx = var_index(lhs)
        if lhs.kind == vnnlib.SymbolKind.Input:
            apply_bound(lhs_idx, kind, rhs_num)
            return
        if lhs.kind == vnnlib.SymbolKind.Output:
            output_constraints.append({"kind": f"{kind}_const", "lhs": lhs_idx, "rhs_const": rhs_num})
            return
        raise ValueError("unsupported lhs var kind")

    if rhs_var and lhs_num is not None:
        inverse = {"le": "ge", "ge": "le", "lt": "gt", "gt": "lt"}[kind]
        rhs_idx = var_index(rhs)
        if rhs.kind == vnnlib.SymbolKind.Input:
            apply_bound(rhs_idx, inverse, lhs_num)
            return
        if rhs.kind == vnnlib.SymbolKind.Output:
            output_constraints.append({"kind": f"{inverse}_const", "lhs": rhs_idx, "rhs_const": lhs_num})
            return
        raise ValueError("unsupported rhs var kind")

    raise ValueError("unsupported comparison form")


for assertion in query.assertions:
    dnf = assertion.expr.to_dnf()
    if len(dnf) > 1:
        is_disjunction = True
    for clause in dnf:
        if len(clause) != 1:
            raise ValueError("only single-constraint clauses are supported")
        handle_comparison(clause[0])

for idx, (lower, upper) in enumerate(input_bounds):
    if lower is None or upper is None:
        raise ValueError(f"missing bounds for input index {idx}")

payload = {
    "parser_version": vnnlib.__version__,
    "version": version,
    "num_inputs": num_inputs,
    "num_outputs": num_outputs,
    "input_bounds": input_bounds,
    "output_constraints": output_constraints,
    "is_disjunction": is_disjunction,
}
print(json.dumps(payload))
"#;

const PYTHON_PARSE_ONLY: &str = r#"
import sys
import vnnlib

path = sys.argv[1]
vnnlib.parse_query_file(path)
"#;

#[derive(Debug, Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    path: String,
    expect: String,
    error_contains: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PythonSpec {
    parser_version: String,
    version: Option<String>,
    num_inputs: usize,
    num_outputs: usize,
    input_bounds: Vec<[f64; 2]>,
    output_constraints: Vec<PythonConstraint>,
    is_disjunction: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct PythonConstraint {
    kind: String,
    lhs: usize,
    rhs: Option<usize>,
    rhs_const: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExpectedSpecs {
    cases: HashMap<String, ExpectedSpec>,
}

#[derive(Debug, Deserialize, Clone)]
struct ExpectedSpec {
    version: Option<String>,
    num_inputs: usize,
    num_outputs: usize,
    input_bounds: Vec<[f64; 2]>,
    output_constraints: Vec<PythonConstraint>,
    is_disjunction: bool,
}

#[test]
fn vnnlib_conformance_cases() {
    let data_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/vnnlib2");
    let manifest_path = data_root.join("manifest.json");
    let manifest = load_manifest(&manifest_path);
    let expected_path = data_root.join("expected_specs.json");
    let expected_specs = load_expected_specs(&expected_path);
    let mut python_available = python_vnnlib_available();

    for case in manifest.cases {
        let case_path = data_root.join(&case.path);
        match case.expect.as_str() {
            "supported" => {
                let baseline = if python_available {
                    match load_python_spec(&case_path, &case.id) {
                        Ok(spec) => BaselineSpec::Python(spec),
                        Err(err) if python_missing_error(&err) => {
                            python_available = false;
                            BaselineSpec::Expected(load_expected_spec(
                                &expected_specs,
                                &case.id,
                                &expected_path,
                            ))
                        }
                        Err(err) => panic!("case '{}' python parser failed: {}", case.id, err),
                    }
                } else {
                    BaselineSpec::Expected(load_expected_spec(
                        &expected_specs,
                        &case.id,
                        &expected_path,
                    ))
                };
                let spec = load_vnnlib(&case_path)
                    .unwrap_or_else(|err| panic!("case '{}' failed to parse: {}", case.id, err));
                assert_spec_matches(&case.id, &spec, &baseline);
            }
            "unsupported" => {
                if python_available {
                    if let Err(err) = assert_python_parses(&case_path, &case.id) {
                        if python_missing_error(&err) {
                            python_available = false;
                        } else {
                            panic!("case '{}' python parser failed: {}", case.id, err);
                        }
                    }
                }
                let err = load_vnnlib(&case_path).unwrap_err();
                let expected = case.error_contains.unwrap_or_default();
                let err_string = err.to_string();
                assert!(
                    err_string.contains(&expected),
                    "case '{}' expected error containing '{}' but got '{}'",
                    case.id,
                    expected,
                    err_string
                );
            }
            other => panic!("case '{}' has unknown expect '{}'", case.id, other),
        }
    }
}

enum BaselineSpec {
    Python(PythonSpec),
    Expected(ExpectedSpec),
}

fn python_vnnlib_available() -> bool {
    let output = Command::new("python3")
        .arg("-c")
        .arg("import vnnlib")
        .output();
    matches!(output, Ok(result) if result.status.success())
}

fn load_expected_specs(path: &Path) -> ExpectedSpecs {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read expected specs {}: {}", path.display(), err));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse expected specs {}: {}", path.display(), err))
}

fn load_expected_spec(expected: &ExpectedSpecs, case_id: &str, path: &Path) -> ExpectedSpec {
    expected
        .cases
        .get(case_id)
        .unwrap_or_else(|| {
            panic!(
                "missing expected spec for case '{}' in {}",
                case_id,
                path.display()
            )
        })
        .clone()
}

fn load_manifest(path: &Path) -> Manifest {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read manifest {}: {}", path.display(), err));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse manifest {}: {}", path.display(), err))
}

fn load_python_spec(path: &Path, case_id: &str) -> Result<PythonSpec, String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(PYTHON_HARNESS)
        .arg(path)
        .output()
        .map_err(|err| format!("case '{}' failed to start python: {}", case_id, err))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("{}{}", stderr, stdout));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|err| format!("case '{}' invalid python output: {}", case_id, err))
}

fn assert_python_parses(path: &Path, case_id: &str) -> Result<(), String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(PYTHON_PARSE_ONLY)
        .arg(path)
        .output()
        .map_err(|err| format!("case '{}' failed to start python: {}", case_id, err))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("{}{}", stderr, stdout));
    }
    Ok(())
}

fn python_missing_error(err: &str) -> bool {
    let lowered = err.to_lowercase();
    (lowered.contains("no module named") && lowered.contains("vnnlib"))
        || lowered.contains("modulenotfounderror")
        || lowered.contains("failed to start python")
        || lowered.contains("no such file or directory")
}

fn assert_spec_matches(case_id: &str, spec: &VnnLibSpec, baseline: &BaselineSpec) {
    let (version, num_inputs, num_outputs, input_bounds, output_constraints, is_disjunction) =
        match baseline {
            BaselineSpec::Python(python) => {
                assert!(
                    python.parser_version.starts_with("1.0"),
                    "case '{}' expected vnnlib parser version 1.0.x, got '{}'",
                    case_id,
                    python.parser_version
                );
                (
                    python.version.as_deref(),
                    python.num_inputs,
                    python.num_outputs,
                    &python.input_bounds,
                    &python.output_constraints,
                    python.is_disjunction,
                )
            }
            BaselineSpec::Expected(expected) => (
                expected.version.as_deref(),
                expected.num_inputs,
                expected.num_outputs,
                &expected.input_bounds,
                &expected.output_constraints,
                expected.is_disjunction,
            ),
        };

    assert_eq!(
        spec.version.as_deref(),
        version,
        "case '{}' version mismatch",
        case_id
    );
    assert_eq!(
        spec.num_inputs, num_inputs,
        "case '{}' num_inputs mismatch",
        case_id
    );
    assert_eq!(
        spec.num_outputs, num_outputs,
        "case '{}' num_outputs mismatch",
        case_id
    );
    assert_eq!(
        spec.input_bounds.len(),
        input_bounds.len(),
        "case '{}' input_bounds length mismatch",
        case_id
    );

    for (spec_bounds, py_bounds) in spec.input_bounds.iter().zip(input_bounds.iter()) {
        assert_relative_eq!(spec_bounds.0, py_bounds[0], epsilon = 1e-6);
        assert_relative_eq!(spec_bounds.1, py_bounds[1], epsilon = 1e-6);
    }

    assert_eq!(
        spec.is_disjunction, is_disjunction,
        "case '{}' disjunction mismatch",
        case_id
    );
    assert_eq!(
        spec.output_constraints.len(),
        output_constraints.len(),
        "case '{}' output_constraints length mismatch",
        case_id
    );

    for (idx, (spec_constraint, py_constraint)) in spec
        .output_constraints
        .iter()
        .zip(output_constraints.iter())
        .enumerate()
    {
        assert_output_constraint(case_id, idx, spec_constraint, py_constraint);
    }
}

fn assert_output_constraint(
    case_id: &str,
    index: usize,
    spec: &OutputConstraint,
    py: &PythonConstraint,
) {
    let expected = match py.kind.as_str() {
        "le" => OutputConstraint::LessEq(py.lhs, py.rhs.expect("missing rhs")),
        "ge" => OutputConstraint::GreaterEq(py.lhs, py.rhs.expect("missing rhs")),
        "lt" => OutputConstraint::LessThan(py.lhs, py.rhs.expect("missing rhs")),
        "gt" => OutputConstraint::GreaterThan(py.lhs, py.rhs.expect("missing rhs")),
        "le_const" => {
            OutputConstraint::LessEqConst(py.lhs, py.rhs_const.expect("missing rhs_const"))
        }
        "ge_const" => {
            OutputConstraint::GreaterEqConst(py.lhs, py.rhs_const.expect("missing rhs_const"))
        }
        "lt_const" => {
            OutputConstraint::LessThanConst(py.lhs, py.rhs_const.expect("missing rhs_const"))
        }
        "gt_const" => {
            OutputConstraint::GreaterThanConst(py.lhs, py.rhs_const.expect("missing rhs_const"))
        }
        other => panic!("case '{}' unsupported constraint kind '{}'", case_id, other),
    };

    match (&expected, spec) {
        (
            OutputConstraint::LessEqConst(_, expected_val),
            OutputConstraint::LessEqConst(_, actual),
        ) => {
            assert_relative_eq!(expected_val, actual, epsilon = 1e-6);
        }
        (
            OutputConstraint::GreaterEqConst(_, expected_val),
            OutputConstraint::GreaterEqConst(_, actual),
        ) => {
            assert_relative_eq!(expected_val, actual, epsilon = 1e-6);
        }
        (
            OutputConstraint::LessThanConst(_, expected_val),
            OutputConstraint::LessThanConst(_, actual),
        ) => {
            assert_relative_eq!(expected_val, actual, epsilon = 1e-6);
        }
        (
            OutputConstraint::GreaterThanConst(_, expected_val),
            OutputConstraint::GreaterThanConst(_, actual),
        ) => {
            assert_relative_eq!(expected_val, actual, epsilon = 1e-6);
        }
        _ => {
            assert_eq!(
                &expected, spec,
                "case '{}' constraint {} mismatch",
                case_id, index
            );
        }
    }
}
