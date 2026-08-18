// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    assert_invalid_spec_contains, parse_vnnlib, parse_vnnlib_assignment_declarations,
    OutputConstraint, TensorDeclarationKind,
};

#[ntest::timeout(10000)]
#[test]
fn test_parse_tensor_decl_vnnlib_2() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Real [2 2])
(declare-output Y Real [2])

(assert (>= X[0,0] -1))
(assert (<= X[0,0] 1))
(assert (<= Y[1] 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 4);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.input_bounds.len(), 4);
    assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(1, c) if (c - 0.5).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_assignment_declarations_preserve_reference_checker_order_and_metadata() {
    let content = r#"
(vnnlib-version <2.0>)
(declare-network f
  (declare-output Y_f real [1, 5])
  (declare-input X_f real [1, 1, 1, 5]))
(declare-network g
  (equal-to f)
  (declare-output Y_g float32 [1, 5])
  (declare-input X_g float32 [1, 1, 1, 5]))
"#;
    let declarations = parse_vnnlib_assignment_declarations(content).unwrap();
    let summary: Vec<_> = declarations
        .iter()
        .map(|declaration| {
            (
                declaration.network.as_deref(),
                declaration.name.as_str(),
                declaration.element_type.as_str(),
                declaration.shape.as_slice(),
                declaration.kind,
            )
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            (
                Some("f"),
                "X_f",
                "real",
                &[1, 1, 1, 5][..],
                TensorDeclarationKind::Input,
            ),
            (
                Some("f"),
                "Y_f",
                "real",
                &[1, 5][..],
                TensorDeclarationKind::Output,
            ),
            (
                Some("g"),
                "X_g",
                "float32",
                &[1, 1, 1, 5][..],
                TensorDeclarationKind::Input,
            ),
            (
                Some("g"),
                "Y_g",
                "float32",
                &[1, 5][..],
                TensorDeclarationKind::Output,
            ),
        ]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_v2_linear_input_constraints() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (>= (+ (* 2 X[0]) 1) 3))
(assert (<= (+ X[0] 1) 2))
(assert (<= (* -1 X[1]) 1))
(assert (<= (* 2 X[1]) 4))
(assert (<= Y[0] 1))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.input_bounds[0], (1.0, 1.0));
    assert_eq!(spec.input_bounds[1], (-1.0, 2.0));
    assert_eq!(spec.output_constraints.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_v2_rejects_strict_scaled_input_constraint() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert (< (+ (* 2 X[0]) 1) 3))
(assert (> Y[0] 0))
"#;

    let err =
        parse_vnnlib(content).expect_err("a scaled affine strict input endpoint must fail closed");
    assert_invalid_spec_contains(err, "Strict input constraints");
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_v2_rejects_multi_var_input_constraint() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= (+ X[0] X[1]) 1))
"#;

    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "single variable");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_tensor_index_bounds() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= X[0] 1.0))
(assert (>= X[1] -1.0))
(assert (<= Y[0] 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 2);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    assert!((spec.input_bounds[1].0 - (-1.0)).abs() < 1e-10);
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(0, c) if (c - 0.5).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_select_index_bounds() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= (select X 0) 1.0))
(assert (>= (select X 1) -1.0))
(assert (<= (select Y 0) 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 2);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    assert!((spec.input_bounds[1].0 - (-1.0)).abs() < 1e-10);
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(0, c) if (c - 0.5).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_multidim_tensor_indices_bounds() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2 3])
(declare-output Y Float32 [1 2])

(assert (>= X[0,1] -1.0))
(assert (<= X[1,2] 2.0))
(assert (<= Y[0,1] 0.25))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 6);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.input_bounds.len(), 6);
    assert!((spec.input_bounds[1].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[5].1 - 2.0).abs() < 1e-10);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(1, c) if (c - 0.25).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_select_multidim_indices_bounds() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2 3])
(declare-output Y Float32 [1 2])

(assert (>= (select X 0 1) -1.0))
(assert (<= (select X 1 2) 2.0))
(assert (<= (select Y 0 1) 0.25))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 6);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.input_bounds.len(), 6);
    assert!((spec.input_bounds[1].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[5].1 - 2.0).abs() < 1e-10);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(1, c) if (c - 0.25).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_declare_network_multidim_indices() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network N
(declare-input X Float32 [2 3])
(declare-output Y Float32 [1 2])
)
(assert (>= X[0,1] -1.0))
(assert (<= X[1,2] 2.0))
(assert (<= Y[0,1] 0.25))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 6);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.input_bounds.len(), 6);
    assert!((spec.input_bounds[1].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[5].1 - 2.0).abs() < 1e-10);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(1, c) if (c - 0.25).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_function_style_indexing_unsupported() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= (X 0) 1.0))
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Function-style tensor access");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_arithmetic_expression_supported() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= (+ X[0] 1.0) 2.0))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    assert!(spec.output_constraints.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_arithmetic_expression_supported_without_version() {
    let content = r#"
(declare-input X Float32 [2])
(declare-output Y Float32 [1])

(assert (<= (+ X[0] 1.0) 2.0))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    assert!(spec.output_constraints.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_declare_network_wrapper_parses_decls() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network N
(declare-input X Float32 [2])
(declare-output Y Float32 [1])
)
(assert (>= X[0] -1.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] -2.0))
(assert (<= X[1] 2.0))
(assert (<= Y[0] 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.version, Some("2.0".to_string()));
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 2);
    assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    assert!((spec.input_bounds[1].0 - (-2.0)).abs() < 1e-10);
    assert!((spec.input_bounds[1].1 - 2.0).abs() < 1e-10);
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(0, c) if (c - 0.5).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_declare_network_ignores_unknown_entries() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network N
(set-info :inputs 2)
(declare-input X Float32 [2])
(set-info :outputs 1)
(declare-output Y Float32 [1])
)
(assert (<= X[0] 1.0))
(assert (>= X[1] -1.0))
(assert (<= Y[0] 0.5))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 2);
    assert_eq!(spec.output_constraints.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_unknown_constraint_symbol_rejected() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert (<= Z_0 1.0))
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Unsupported comparison constraint expression");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_unsupported_boolean_operator_rejected() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert (not (<= X[0] 1.0)))
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Unsupported list expression 'not'");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_comparison_requires_two_operands() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert (<= X[0]))
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "requires exactly 2 operands");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_comparison_rejects_extra_operands() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert (<= X[0] 1.0 2.0))
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "requires exactly 2 operands");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_empty_constraint_rejected() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert ())
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Empty constraint expression");
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_non_list_constraint_rejected() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [1])
(declare-output Y Float32 [1])

(assert X[0])
"#;
    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Unsupported constraint expression");
}
