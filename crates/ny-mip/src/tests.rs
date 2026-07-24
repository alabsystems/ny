// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::config::{MipBackend, MipConfig};
use crate::encoder::encode_feedforward;
use crate::solver::{MipResult, MipSolver};
use ny_core::Bound;

/// Every backend available to this build+environment. Solve-based tests run
/// against each so all backends are held to identical verdict contracts.
///
/// ay (the production backend, docs/SOLVER_POLICY.md) is in-process and
/// always available. AyProc (the frozen P0 subprocess lane) is included when
/// the external binary is reachable ($NY_AY/$PATH) and skipped loudly
/// otherwise. (HiGHS was deleted at LG3.)
fn all_backends() -> Vec<MipBackend> {
    let mut backends = vec![MipBackend::Ay];
    if ay_available() {
        backends.push(MipBackend::AyProc);
    } else {
        eprintln!("SKIP ay-proc backend: no ay binary on $NY_AY/$PATH");
    }
    backends
}

/// Is the external ay binary reachable?
pub(crate) fn ay_available() -> bool {
    let mut cmd = match std::env::var_os("NY_AY") {
        Some(path) => std::process::Command::new(path),
        None => std::process::Command::new("ay"),
    };
    cmd.arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Default config pinned to a specific backend.
fn config_for(backend: MipBackend) -> MipConfig {
    MipConfig {
        backend,
        ..MipConfig::default()
    }
}

/// Small FC+ReLU network: 2 inputs -> 3 hidden (ReLU) -> 2 outputs.
///
/// Weights/biases chosen so the network is simple to verify by hand.
/// Layer 0: W = [[1, 0], [0, 1], [1, 1]], b = [0, 0, -1]
/// Layer 1: W = [[1, -1, 0], [0, 1, 1]], b = [0, 0]
fn small_network() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
    let weights = vec![
        // Layer 0: 3x2 (3 outputs, 2 inputs), row-major
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        // Layer 1: 2x3 (2 outputs, 3 inputs), row-major
        vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
    ];
    let biases = vec![
        vec![0.0, 0.0, -1.0], // Layer 0
        vec![0.0, 0.0],       // Layer 1
    ];
    let layer_dims = vec![2, 3, 2]; // input=2, hidden=3, output=2
    (weights, biases, layer_dims)
}

#[test]
#[ntest::timeout(30_000)]
fn test_encode_small_network() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    // Pre-activation bounds for hidden layer (after linear, before ReLU).
    // Layer 0 output before ReLU: x1 in [0,1], x2 in [0,1], x1+x2-1 in [-1,1]
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),  // x1: always active
        Bound::new(0.0, 1.0),  // x2: always active
        Bound::new(-1.0, 1.0), // x1+x2-1: unstable
    ]];

    let encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("encoding should succeed");

    assert_eq!(encoder.output_vars().len(), 2);
    // One unstable neuron -> one binary variable
    assert_eq!(encoder.num_binary_vars(), 1);
}

#[test]
#[ntest::timeout(30_000)]
fn test_feasibility_sat() {
    // Network with input [0,1]^2 should be feasible (non-empty input region).
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    for backend in all_backends() {
        let encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Sat {
                output_values,
                input_values,
                ..
            } => {
                assert_eq!(output_values.len(), 2, "{backend:?}");
                assert_eq!(input_values.len(), 2, "{backend:?}");
                // Inputs should be within bounds
                for &v in &input_values {
                    assert!(
                        (-1e-8..=1.0 + 1e-8).contains(&v),
                        "{backend:?}: input {v} out of [0,1]"
                    );
                }
            }
            other => panic!("{backend:?}: expected SAT, got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_feasibility_linear_network() {
    // Trivial 1-layer network: y = 2*x + 3, input [0,1]
    let weights = vec![vec![2.0]];
    let biases = vec![vec![3.0]];
    let layer_dims = vec![1, 1];
    let input_bounds = vec![Bound::new(0.0, 1.0)];
    let intermediate_bounds: Vec<Vec<Bound>> = vec![];

    for backend in all_backends() {
        let encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Sat {
                output_values,
                input_values,
                ..
            } => {
                // Input x in [0,1], output y = 2x+3 in [3,5]
                assert_eq!(output_values.len(), 1, "{backend:?}");
                let y = output_values[0];
                let x = input_values[0];
                assert!(
                    (y - (2.0 * x + 3.0)).abs() < 1e-8,
                    "{backend:?}: output {y} != 2*{x}+3 = {}",
                    2.0 * x + 3.0
                );
            }
            other => panic!("{backend:?}: expected SAT for non-empty input, got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_all_active_relu() {
    // When all pre-activation bounds are positive, no binary variables needed.
    let weights = vec![
        vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
        vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
    ];
    let biases = vec![
        vec![1.0, 1.0], // Shift positive so ReLU is always active
        vec![0.0, 0.0],
    ];
    let layer_dims = vec![2, 2, 2];
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    // Pre-activation: x+1 in [1,2] -- always positive
    let intermediate_bounds = vec![vec![Bound::new(1.0, 2.0), Bound::new(1.0, 2.0)]];

    let encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("encoding should succeed");

    // No unstable neurons -> no binary variables
    assert_eq!(encoder.num_binary_vars(), 0);
}

#[test]
#[ntest::timeout(30_000)]
fn test_all_inactive_relu() {
    // When all pre-activation bounds are negative, ReLU output is zero.
    let weights = vec![vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0]];
    let biases = vec![
        vec![-5.0, -5.0], // Shift very negative
        vec![0.0, 0.0],
    ];
    let layer_dims = vec![2, 2, 2];
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    // Pre-activation: x-5 in [-5,-4] -- always negative
    let intermediate_bounds = vec![vec![Bound::new(-5.0, -4.0), Bound::new(-5.0, -4.0)]];

    for backend in all_backends() {
        let encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        assert_eq!(encoder.num_binary_vars(), 0);

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Sat { output_values, .. } => {
                // Both outputs should be 0 since all ReLUs are inactive:
                // hidden = ReLU(x + [-5,-5]) = [0, 0], output = I * [0,0] = [0, 0]
                for (i, &val) in output_values.iter().enumerate() {
                    assert!(
                        val.abs() < 1e-8,
                        "{backend:?}: expected output[{i}] ~0.0, got {val}"
                    );
                }
            }
            other => panic!("{backend:?}: expected SAT, got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_output_constraint_leq() {
    // Network: 2 inputs -> 3 hidden (ReLU) -> 2 outputs.
    // Add constraint: output[0] <= output[1] (unsafe region).
    // At input (0,0): hidden = ReLU([0, 0, -1]) = [0, 0, 0], output = [0, 0].
    //   => output[0] <= output[1] (0 <= 0) is satisfied → SAT.
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    for backend in all_backends() {
        let mut encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        encoder
            .constrain_output_leq(0, 1)
            .expect("constraint should succeed");

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Sat { .. } => {} // Expected: unsafe region is reachable
            other => panic!("{backend:?}: expected SAT (unsafe region reachable), got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_output_constraint_geq_const_unsat() {
    // Trivial network: y = 2*x + 3, input x in [0, 1].
    // Output y in [3, 5].
    // Unsafe constraint: y >= 10.0. This is infeasible → UNSAT → verified safe.
    let weights = vec![vec![2.0]];
    let biases = vec![vec![3.0]];
    let layer_dims = vec![1, 1];
    let input_bounds = vec![Bound::new(0.0, 1.0)];
    let intermediate_bounds: Vec<Vec<Bound>> = vec![];

    for backend in all_backends() {
        let mut encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        encoder
            .constrain_output_geq_const(0, 10.0)
            .expect("constraint should succeed");

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Unsat { .. } => {} // Expected: y can never reach 10
            other => panic!("{backend:?}: expected UNSAT (property verified), got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_output_constraint_leq_const_sat() {
    // Trivial network: y = 2*x + 3, input x in [0, 1].
    // Output y in [3, 5].
    // Unsafe constraint: y <= 4.0. Satisfiable when x in [0, 0.5] → SAT.
    let weights = vec![vec![2.0]];
    let biases = vec![vec![3.0]];
    let layer_dims = vec![1, 1];
    let input_bounds = vec![Bound::new(0.0, 1.0)];
    let intermediate_bounds: Vec<Vec<Bound>> = vec![];

    for backend in all_backends() {
        let mut encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");

        encoder
            .constrain_output_leq_const(0, 4.0)
            .expect("constraint should succeed");

        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));
        let result = solver.check_feasibility().expect("solve should succeed");

        match result {
            MipResult::Sat {
                output_values,
                input_values,
                ..
            } => {
                assert!(
                    output_values[0] <= 4.0 + 1e-8,
                    "{backend:?}: output should be <= 4.0"
                );
                assert!(
                    input_values[0] <= 0.5 + 1e-8,
                    "{backend:?}: input should be <= 0.5 for y <= 4.0"
                );
            }
            other => panic!("{backend:?}: expected SAT, got {other:?}"),
        }
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_dimension_mismatch_errors() {
    // Bias-layer cardinality must match before indexing. This is a recoverable
    // encoding error, never a panic or an opportunity to ignore extra layers.
    let result = encode_feedforward(
        &[vec![1.0], vec![1.0]],
        &[vec![0.0]],
        &[1, 1, 1],
        &[Bound::new(0.0, 1.0)],
        &[vec![Bound::new(0.0, 1.0)]],
    );
    assert!(result.is_err());

    // Wrong weight dimensions
    let result = encode_feedforward(
        &[vec![1.0, 2.0, 3.0]], // 3 elements, but need 2x1=2
        &[vec![0.0, 0.0]],
        &[1, 2],
        &[Bound::new(0.0, 1.0)],
        &[],
    );
    assert!(result.is_err());

    // Wrong bias dimensions
    let result = encode_feedforward(
        &[vec![1.0]],
        &[vec![0.0, 0.0]], // 2 biases, but only 1 output
        &[1, 1],
        &[Bound::new(0.0, 1.0)],
        &[],
    );
    assert!(result.is_err());

    // Empty network
    let result = encode_feedforward(&[], &[], &[1], &[Bound::new(0.0, 1.0)], &[]);
    assert!(result.is_err());

    // The encoder has exactly one implicit ReLU after every non-final Linear.
    // Extra or missing bound vectors signal a topology mismatch and must not be
    // silently ignored.
    let result = encode_feedforward(
        &[vec![1.0]],
        &[vec![0.0]],
        &[1, 1],
        &[Bound::new(0.0, 1.0)],
        &[vec![Bound::new(0.0, 1.0)]],
    );
    assert!(result.is_err());

    let result = encode_feedforward(
        &[vec![1.0], vec![1.0]],
        &[vec![0.0], vec![0.0]],
        &[1, 1, 1],
        &[Bound::new(0.0, 1.0)],
        &[],
    );
    assert!(result.is_err());
}

/// Build a warm-start vector for `small_network()` by forward-propagating [x0, x1].
///
/// Replays column order: input(2) → linear0 pre-act(3) → relu0 unstable cols(2) → linear1 pre-act(2) = 9.
fn build_small_network_warm_start(x0: f64, x1: f64, num_cols: usize) -> Vec<f64> {
    let mut ws = vec![0.0f64; num_cols];
    let mut ci = 0;
    // Input columns
    ws[ci] = x0;
    ci += 1;
    ws[ci] = x1;
    ci += 1;
    // Linear0 pre-activation: W=[[1,0],[0,1],[1,1]], b=[0,0,-1]
    let z0 = x0;
    let z1 = x1;
    let z2 = x0 + x1 - 1.0;
    ws[ci] = z0;
    ci += 1;
    ws[ci] = z1;
    ci += 1;
    ws[ci] = z2;
    ci += 1;
    // ReLU0: neurons 0,1 active (lb>=0) → no cols. Neuron 2 unstable → y_var + z_var.
    let y2 = z2.max(0.0);
    let ind2 = if z2 >= 0.0 { 1.0 } else { 0.0 };
    ws[ci] = y2;
    ci += 1;
    ws[ci] = ind2;
    ci += 1;
    // Linear1 pre-activation: W=[[1,-1,0],[0,1,1]], b=[0,0]
    let (a0, a1, a2) = (z0, z1, y2);
    ws[ci] = a0 - a1;
    ci += 1; // out0 = 1*a0 - 1*a1
    ws[ci] = a1 + a2;
    ci += 1; // out1 = 1*a1 + 1*a2
    assert_eq!(ci, num_cols, "warm-start vector should fill all columns");
    ws
}

/// #3865: Warm-started and cold solves return the same verdict on a small network.
#[test]
#[ntest::timeout(30_000)]
fn warm_start_and_cold_solve_agree_on_verdict_3865() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    for backend in all_backends() {
        // Cold solve
        let encoder_cold = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");
        let parts_cold = encoder_cold.into_parts();
        let num_cols = parts_cold.num_cols;
        let solver_cold = MipSolver::new(parts_cold, config_for(backend));
        let cold_result = solver_cold
            .check_feasibility()
            .expect("cold solve should succeed");

        // Warm-started solve with candidate [0.5, 0.5]
        let encoder_warm = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");
        let parts_warm = encoder_warm.into_parts();
        let solver_warm = MipSolver::new(parts_warm, config_for(backend));
        let warm_start = build_small_network_warm_start(0.5, 0.5, num_cols);
        let warm_result = solver_warm
            .check_feasibility_with_warm_start(Some(&warm_start))
            .expect("warm solve should succeed");

        assert!(
            matches!(cold_result, MipResult::Sat { .. }),
            "{backend:?} cold: {:?}",
            cold_result
        );
        assert!(
            matches!(warm_result, MipResult::Sat { .. }),
            "{backend:?} warm: {:?}",
            warm_result
        );
    }
}

/// #3865: Warm-start with wrong-length vector falls back to cold solve gracefully.
#[test]
#[ntest::timeout(30_000)]
fn warm_start_wrong_length_falls_back_to_cold_solve_3865() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    for backend in all_backends() {
        let encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("encoding should succeed");
        let parts = encoder.into_parts();
        let solver = MipSolver::new(parts, config_for(backend));

        // Provide a wrong-length warm-start vector (should fall back to cold)
        let wrong_length = vec![0.0f64; 3]; // too short
        let result = solver
            .check_feasibility_with_warm_start(Some(&wrong_length))
            .expect("solve should succeed even with rejected warm-start");

        assert!(
            matches!(result, MipResult::Sat { .. }),
            "{backend:?}: should still find SAT via cold-solve fallback, got {:?}",
            result
        );
    }
}

/// #3865: num_cols in MipParts matches the encoder's column count.
#[test]
fn num_cols_matches_encoder_column_count_3865() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    let mut encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("encoding should succeed");
    let pre_finalize_cols = encoder.num_cols();
    encoder.finalize();
    let parts = encoder.into_parts();

    // num_cols should match what the encoder reported before into_parts
    assert_eq!(parts.num_cols, pre_finalize_cols);
    // Expected: 2 (input) + 3 (linear0) + 2 (unstable relu: y_var + z_var) + 2 (linear1) = 9
    assert_eq!(parts.num_cols, 9);
}

/// binary_widths is aligned with binary_vars and carries pre-activation u-l.
/// Drives the phase-split branching selection (designs/scip.md Phase C).
#[test]
fn binary_widths_align_with_binary_vars() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),  // stable active: no binary
        Bound::new(0.0, 1.0),  // stable active: no binary
        Bound::new(-1.0, 1.0), // unstable: one binary, width 2.0
    ]];

    let encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("encoding should succeed");
    let parts = encoder.into_parts();

    assert_eq!(parts.binary_vars.len(), 1);
    assert_eq!(parts.binary_widths.len(), parts.binary_vars.len());
    assert!((parts.binary_widths[0] - 2.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Backend verdict-equality property test (designs/scip.md validation plan):
// random tiny FC+ReLU nets must get the exact same Sat/Unsat verdict from
// every compiled backend. Any disagreement is a soundness red flag in one of
// the lowerings and fails the suite.
// ---------------------------------------------------------------------------

/// Sound interval (IBP) pre-activation bounds for the tiny test nets, computed
/// in f64 and widened outward before the f32 cast so the Big-M encoding is
/// valid for every backend.
fn ibp_intermediate_bounds(
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
) -> Vec<Vec<Bound>> {
    const WIDEN: f64 = 1e-6;
    let mut lower: Vec<f64> = input_bounds.iter().map(|b| b.lower() as f64).collect();
    let mut upper: Vec<f64> = input_bounds.iter().map(|b| b.upper() as f64).collect();
    let mut result = Vec::new();

    for layer in 0..weights.len().saturating_sub(1) {
        let in_dim = layer_dims[layer];
        let out_dim = layer_dims[layer + 1];
        let mut new_lower = Vec::with_capacity(out_dim);
        let mut new_upper = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            let mut lo = biases[layer][i];
            let mut hi = biases[layer][i];
            for j in 0..in_dim {
                let w = weights[layer][i * in_dim + j];
                if w >= 0.0 {
                    lo += w * lower[j];
                    hi += w * upper[j];
                } else {
                    lo += w * upper[j];
                    hi += w * lower[j];
                }
            }
            new_lower.push(lo - WIDEN);
            new_upper.push(hi + WIDEN);
        }
        result.push(
            new_lower
                .iter()
                .zip(new_upper.iter())
                .map(|(&l, &u)| Bound::new(l as f32, u as f32))
                .collect(),
        );
        // Post-ReLU bounds feed the next layer.
        lower = new_lower.iter().map(|&l| l.max(0.0)).collect();
        upper = new_upper.iter().map(|&u| u.max(0.0)).collect();
    }
    result
}

/// Coarse verdict label for equality comparison (witnesses may differ).
fn verdict_label(result: &MipResult) -> &'static str {
    match result {
        MipResult::Sat { .. } => "sat",
        MipResult::Unsat { .. } => "unsat",
        MipResult::Timeout => "timeout",
        MipResult::Error(_) => "error",
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 24, // each case is 1 MIP solve per backend; keep the suite fast
        ..proptest::prelude::ProptestConfig::default()
    })]

    /// HiGHS and SCIP (when compiled) must agree exactly on Sat/Unsat for
    /// random tiny 2-layer FC+ReLU nets with a `output[0] >= t` property.
    #[test]
    fn backends_agree_on_random_tiny_nets(
        in_dim in 1usize..=3,
        hidden_dim in 2usize..=5,
        weight_seed in proptest::collection::vec(-2.0f64..2.0, 40),
        bias_seed in proptest::collection::vec(-1.0f64..1.0, 12),
        threshold in -4.0f64..4.0,
    ) {
        // Assemble a [in_dim, hidden_dim, 1] net from the seeds.
        let out_dim = 1usize;
        let layer_dims = vec![in_dim, hidden_dim, out_dim];
        let w0: Vec<f64> = (0..hidden_dim * in_dim)
            .map(|k| weight_seed[k % weight_seed.len()])
            .collect();
        let w1: Vec<f64> = (0..out_dim * hidden_dim)
            .map(|k| weight_seed[(k + 7) % weight_seed.len()])
            .collect();
        let b0: Vec<f64> = (0..hidden_dim).map(|k| bias_seed[k % bias_seed.len()]).collect();
        let b1: Vec<f64> = (0..out_dim).map(|k| bias_seed[(k + 3) % bias_seed.len()]).collect();
        let weights = vec![w0, w1];
        let biases = vec![b0, b1];
        let input_bounds = vec![Bound::new(0.0, 1.0); in_dim];
        let intermediate_bounds =
            ibp_intermediate_bounds(&weights, &biases, &layer_dims, &input_bounds);

        let mut verdicts = Vec::new();
        for backend in all_backends() {
            let mut encoder = encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &intermediate_bounds,
            )
            .expect("encoding should succeed");
            encoder
                .constrain_output_geq_const(0, threshold)
                .expect("constraint should succeed");
            let solver = MipSolver::new(encoder.into_parts(), config_for(backend));
            let result = solver.check_feasibility().expect("solve should succeed");
            proptest::prop_assert!(
                !matches!(result, MipResult::Timeout | MipResult::Error(_)),
                "{backend:?}: tiny net must decide, got {result:?}"
            );
            verdicts.push((backend, verdict_label(&result)));
        }

        // All compiled backends agree exactly.
        for window in verdicts.windows(2) {
            proptest::prop_assert_eq!(
                window[0].1,
                window[1].1,
                "backend verdict disagreement: {:?}",
                &verdicts
            );
        }
    }
}

/// Phase-split racing agrees with the serial solve on both fixtures and every
/// compiled backend (designs/scip.md Phase C validation): sat stays sat with a
/// valid witness, unsat stays unsat. parallel_split=16 forces the maximum
/// split (capped by the fixture's binary count); parallel_split=1 is the
/// serial disable path.
#[test]
#[ntest::timeout(60_000)]
fn split_and_serial_solves_agree_on_fixtures() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0), // one unstable binary -> k=1 -> 2 subproblems
    ]];

    for backend in all_backends() {
        for &(unsafe_threshold, expect_sat) in &[(1.5f64, true), (10.0f64, false)] {
            let mut verdicts = Vec::new();
            for parallel_split in [1usize, 16usize] {
                let mut encoder = encode_feedforward(
                    &weights,
                    &biases,
                    &layer_dims,
                    &input_bounds,
                    &intermediate_bounds,
                )
                .expect("encoding should succeed");
                // output[1] = x2 + relu(x1+x2-1) has range [0, 2].
                encoder
                    .constrain_output_geq_const(1, unsafe_threshold)
                    .expect("constraint should succeed");
                let config = MipConfig {
                    parallel_split,
                    ..config_for(backend)
                };
                let solver = MipSolver::new(encoder.into_parts(), config);
                let result = solver.check_feasibility().expect("solve should succeed");
                if expect_sat {
                    match &result {
                        MipResult::Sat { input_values, .. } => {
                            for &v in input_values {
                                assert!(
                                    (-1e-8..=1.0 + 1e-8).contains(&v),
                                    "{backend:?} split={parallel_split}: witness {v} out of box"
                                );
                            }
                        }
                        other => panic!(
                            "{backend:?} split={parallel_split}: expected SAT, got {other:?}"
                        ),
                    }
                } else {
                    assert!(
                        matches!(result, MipResult::Unsat { .. }),
                        "{backend:?} split={parallel_split}: expected UNSAT, got {result:?}"
                    );
                }
                verdicts.push(verdict_label(&result));
            }
            assert_eq!(
                verdicts[0], verdicts[1],
                "{backend:?}: split/serial verdict divergence at threshold {unsafe_threshold}"
            );
        }
    }
}
