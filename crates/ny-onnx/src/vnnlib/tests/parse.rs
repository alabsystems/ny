// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{assert_invalid_spec_contains, load_vnnlib, parse_vnnlib, OutputConstraint};
use crate::vnnlib::{DualNetworkProperty, NetworkRelation};

#[ntest::timeout(10000)]
#[test]
fn test_parse_simple_vnnlib() {
    let content = r#"
; Simple test
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))

(assert (<= Y_0 -1))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 1);
    assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEqConst(0, c) if (c - (-1.0)).abs() < 1e-10
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_ignores_inline_comments() {
    let content = r#"
(declare-const X_0 Real) ; input variable
(declare-const Y_0 Real) ; output variable

(assert (>= X_0 -1)) ; lower bound
(assert (<= X_0 1)) ; upper bound
(assert (<= Y_0 0.25)) ; output constraint
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.input_bounds.len(), 1);
    assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
    assert_eq!(spec.output_constraints.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_vnnlib_linearized_output_constraint() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= (+ Y_0 (* -1 Y_1)) 0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEq(0, 1)
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_acasxu_property() {
    let content = r#"
; ACAS Xu property 2
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const X_4 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)

(assert (<= X_0 0.679857769))
(assert (>= X_0 0.6))

(assert (<= X_1 0.5))
(assert (>= X_1 -0.5))

(assert (<= X_2 0.5))
(assert (>= X_2 -0.5))

(assert (<= X_3 0.5))
(assert (>= X_3 0.45))

(assert (<= X_4 -0.45))
(assert (>= X_4 -0.5))

; Unsafe if COC is maximal
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 5);
    assert_eq!(spec.num_outputs, 5);
    assert_eq!(spec.input_bounds.len(), 5);

    // Check input bounds
    assert!((spec.input_bounds[0].0 - 0.6).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 0.679857769).abs() < 1e-10);
    assert!((spec.input_bounds[1].0 - (-0.5)).abs() < 1e-10);
    assert!((spec.input_bounds[1].1 - 0.5).abs() < 1e-10);

    // Check output constraints (Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0)
    assert_eq!(spec.output_constraints.len(), 4);
    assert!(matches!(
        spec.output_constraints[0],
        OutputConstraint::LessEq(1, 0)
    ));
    assert!(matches!(
        spec.output_constraints[1],
        OutputConstraint::LessEq(2, 0)
    ));
    assert!(matches!(
        spec.output_constraints[2],
        OutputConstraint::LessEq(3, 0)
    ));
    assert!(matches!(
        spec.output_constraints[3],
        OutputConstraint::LessEq(4, 0)
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_load_real_vnnlib() {
    // Try to load an actual VNN-LIB file if available
    let test_paths = [
        "../../research/repos/nnenum/examples/test/test_prop.vnnlib",
        "../../research/repos/nnenum/examples/acasxu/data/prop_2.vnnlib",
    ];

    for path in test_paths {
        let full_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        if full_path.exists() {
            let spec = load_vnnlib(&full_path).unwrap();
            assert!(spec.num_inputs > 0);
            assert!(spec.num_outputs > 0);
            assert!(spec.has_valid_bounds());
            println!("Loaded {}: {}", path, spec.describe());
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_isomorphic_requires_complete_input_coupling() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [2])
  (declare-output Y_f Float32 [2])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Float32 [2])
  (declare-output Y_g Float32 [2])
)
(assert (>= X_f[0] -1))
(assert (<= X_f[0] 1))
(assert (>= X_g[0] -1))
(assert (<= X_g[0] 1))
(assert (>= X_f[1] 0))
(assert (<= X_f[1] 2))
(assert (>= X_g[1] 0))
(assert (<= X_g[1] 2))
(assert (== X_f[0] X_g[0]))
(assert (<= Y_g[0] (+ Y_f[0] 0.01)))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(matches!(
        dual.property,
        DualNetworkProperty::EpsilonEquivalence { epsilon } if (epsilon - 0.01).abs() < 1e-12
    ));
    assert!(
        !dual.shared_input_coupling,
        "one missing input equality must not authorize shared-input unsat"
    );
    assert_eq!(dual.f_input_bounds, dual.g_input_bounds);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_isomorphic_complete_input_coupling_collapses_bounds() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [2])
  (declare-output Y_f Float32 [2])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Float32 [2])
  (declare-output Y_g Float32 [2])
)
(assert (>= X_f[0] -1))
(assert (<= X_f[0] 1))
(assert (>= X_g[0] -1))
(assert (<= X_g[0] 1))
(assert (>= X_f[1] 0))
(assert (<= X_f[1] 2))
(assert (>= X_g[1] 0))
(assert (<= X_g[1] 2))
(assert (== X_f[0] X_g[0]))
(assert (== X_g[1] X_f[1]))
(assert (<= Y_g[0] (+ Y_f[0] 0.01)))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(
        dual.shared_input_coupling,
        "all exact equalities plus matching bounds authorize shared-input verification"
    );
    assert_eq!(dual.f_input_bounds, dual.g_input_bounds);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_monotonic_preserves_non_strict_unsafe() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network f
(declare-input X_f Float32 [2])
(declare-output Y_f Float32 [2])
)
(declare-network g
(equal-to f)
(declare-input X_g Float32 [2])
(declare-output Y_g Float32 [2])
)
(assert (<= Y_f[1] Y_g[1]))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(matches!(
        dual.property,
        DualNetworkProperty::MonotonicGreaterEq {
            output: 1,
            varying_input: 0,
            strict_unsafe: false
        }
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_monotonic_preserves_reversed_strict_unsafe() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network f
(declare-input X_f Float32 [2])
(declare-output Y_f Float32 [2])
)
(declare-network g
(equal-to f)
(declare-input X_g Float32 [2])
(declare-output Y_g Float32 [2])
)
(assert (> Y_g[0] Y_f[0]))
"#;

    let spec = parse_vnnlib(content).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(matches!(
        dual.property,
        DualNetworkProperty::MonotonicGreaterEq {
            output: 0,
            varying_input: 0,
            strict_unsafe: true
        }
    ));
}

/// A canonical ground-truth dominance property: f is the real network, g a
/// symbolic ground truth resolved from a `.gt.json` sidecar, unsafe clause
/// `Y_f < Y_g` (f dips below the ground truth), safe complement `f − g ≥ 0`.
fn ground_truth_dominance_vnnlib(unsafe_op: &str) -> String {
    format!(
        r#"
(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [3])
  (declare-output Y_f Float32 [1])
)
(declare-network g
  (ground-truth "cyl.gt.json")
  (declare-input X_g Float32 [3])
  (declare-output Y_g Float32 [1])
)
(assert (>= X_f[0] -1)) (assert (<= X_f[0] 1))
(assert (>= X_g[0] -1)) (assert (<= X_g[0] 1))
(assert (>= X_f[1] -1)) (assert (<= X_f[1] 1))
(assert (>= X_g[1] -1)) (assert (<= X_g[1] 1))
(assert (>= X_f[2] -1)) (assert (<= X_f[2] 1))
(assert (>= X_g[2] -1)) (assert (<= X_g[2] 1))
(assert (== X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (== X_f[2] X_g[2]))
(assert ({unsafe_op} Y_f[0] Y_g[0]))
"#
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_ground_truth_dominance_strict() {
    let spec = parse_vnnlib(&ground_truth_dominance_vnnlib("<")).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(matches!(
        dual.property,
        DualNetworkProperty::DominatesSecond {
            strict_unsafe: true
        }
    ));
    // The sidecar reference is preserved verbatim (quotes stripped) and the
    // implicit relation target is the counterpart network.
    let g = dual
        .networks
        .iter()
        .find(|n| n.name == "g")
        .expect("g declared");
    match &g.relation_to {
        Some((NetworkRelation::GroundTruth(path), target)) => {
            assert_eq!(path, "cyl.gt.json");
            assert_eq!(target, "f");
        }
        other => panic!("expected ground-truth relation, got {other:?}"),
    }
    // Full explicit equality coupling with matching boxes authorizes the
    // shared-input difference-network path (like epsilon-equivalence).
    assert!(dual.shared_input_coupling, "explicit coupling recognized");
    assert_eq!(dual.f_input_bounds, vec![(-1.0, 1.0); 3]);
    assert_eq!(dual.f_input_bounds, dual.g_input_bounds);
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_ground_truth_dominance_non_strict_unsafe() {
    // `<=` unsafe: equality remains unsafe, so proving h >= 0 is NOT enough —
    // the parser must preserve strict_unsafe = false for the verify layer.
    let spec = parse_vnnlib(&ground_truth_dominance_vnnlib("<=")).unwrap();
    let dual = spec.dual_network.expect("dual-network property");
    assert!(matches!(
        dual.property,
        DualNetworkProperty::DominatesSecond {
            strict_unsafe: false
        }
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_ground_truth_without_unsafe_clause_is_not_dual() {
    // No f/g output comparison: no property can be derived; the parser must
    // decline the dual-network interpretation (sound) rather than invent a
    // dominance property. (Without a recognized dual property, the coupling
    // equalities have no consumer either, so they are dropped here too.)
    let content = ground_truth_dominance_vnnlib("<")
        .replace("(assert (< Y_f[0] Y_g[0]))", "")
        .replace("(assert (== X_f[0] X_g[0]))", "")
        .replace("(assert (== X_f[1] X_g[1]))", "")
        .replace("(assert (== X_f[2] X_g[2]))", "");
    let spec = parse_vnnlib(&content).unwrap();
    assert!(spec.dual_network.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_dual_ground_truth_requires_path() {
    let content = ground_truth_dominance_vnnlib("<").replace("\"cyl.gt.json\"", "\"\"");
    assert!(parse_vnnlib(&content).is_err(), "empty path rejected");
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_v20_rejects_single_network_input_relation() {
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [1])
(assert (>= X[0] 0))
(assert (<= X[0] 1))
(assert (>= X[1] 0))
(assert (<= X[1] 1))
(assert (<= X[0] X[1]))
(assert (<= Y[0] 0))
"#;

    let err = parse_vnnlib(content).unwrap_err();
    assert_invalid_spec_contains(err, "Input constraints must reference a single variable");
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_v20_rejects_isomorphic_input_ordering() {
    let content = r#"
(vnnlib-version 2.0)
(declare-network f
  (declare-input X_f Float32 [1])
  (declare-output Y_f Float32 [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Float32 [1])
  (declare-output Y_g Float32 [1])
)
(assert (>= X_f[0] 0))
(assert (<= X_f[0] 1))
(assert (>= X_g[0] 0))
(assert (<= X_g[0] 1))
(assert (>= X_f[0] X_g[0]))
(assert (> Y_g[0] (+ Y_f[0] 0.1)))
"#;

    // Dual-network files now PARSE with relational input atoms tolerated (the
    // 2026 relational benchmarks require it); the ISOMORPHIC-with-ordering
    // shape is rejected SEMANTICALLY instead: no equality coupling is
    // recorded, so `shared_input_coupling` stays false — the difference-net
    // soundness gate and the formula-implication check both decline, and no
    // verdict can be emitted for this shape (same safety, better layer).
    let spec = parse_vnnlib(content).expect("dual-network file must parse");
    let dual = spec.dual_network.expect("dual spec present");
    assert!(
        !dual.shared_input_coupling,
        "ordering is not an equality coupling"
    );
    assert_eq!(dual.validation.input_equalities, vec![false]);
    assert!(dual.validation.f_input_ge_g_input[0]);
    assert!(
        !dual.validation.isomorphic_output_safe_complement,
        "one-sided deviation is not the safe complement"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_unknown_expression_ignored() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(set-info :status unknown)
(set-logic QF_LRA)

(assert (>= X_0 0))
(assert (<= X_0 1))
"#;

    // Unknown top-level expressions should be ignored
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert!((spec.input_bounds[0].0 - 0.0).abs() < 1e-10);
    assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
}

#[ntest::timeout(10000)]
#[test]
fn test_unknown_output_operator_errors() {
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (!= Y_0 Y_1))
"#;

    assert!(parse_vnnlib(content).is_err());
}

// --- Output index validation at parse time (#1886) ---

#[ntest::timeout(10000)]
#[test]
fn test_parse_rejects_undeclared_output_index_relational() {
    // Declares Y_0 only (num_outputs=1), but constraint references Y_5.
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_5))
"#;

    let err = parse_vnnlib(content).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Y_5"),
        "Error should reference Y_5, got: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_rejects_undeclared_output_index_const() {
    // Declares Y_0 only (num_outputs=1), but constraint references Y_3.
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_3 0.5))
"#;

    let err = parse_vnnlib(content).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Y_3"),
        "Error should reference Y_3, got: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_parse_accepts_all_declared_output_indices() {
    // All referenced indices are declared — should parse successfully.
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_1))
(assert (>= Y_2 0.5))
"#;

    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_outputs, 3);
    assert!(spec.validate_output_indices().is_ok());
}
