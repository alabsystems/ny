// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Offline end-to-end verification regression tests.
//!
//! The repository checks in NO real ONNX models (they are gitignored and
//! fetched at runtime), so the full verify pipeline could not previously be
//! exercised offline. These tests close that gap using the committed `.nnet`
//! fixtures: they load a tiny ReLU MLP, convert it to the verification
//! `Network`, and run the real IBP + CROWN bound-propagation engines against a
//! VNN-LIB-style property (an output linear constraint over an input box).
//!
//! The point is a *known verdict* regression that runs with zero downloads, so
//! a soundness flip (verified <-> violated) is caught offline / in CI.

use ndarray::Array1;
use ny_onnx::nnet::load_nnet;
use ny_propagate::prelude::Network;
use ny_tensor::BoundedTensor;
use ny_test_utils::{require_model, test_models_dir, workspace_root};

/// A safety property over a single output: `Y_0 <= threshold`. The matching
/// `.vnnlib` fixtures encode it in standard negated VNN-LIB form — they assert
/// the VIOLATION `(assert (>= Y_0 t))`, which the verifier must prove UNSAT.
/// The verdict is:
///   - VERIFIED if the network's CERTIFIED upper bound on Y_0 is <= threshold
///     (sound: no input in the box can violate it).
///   - VIOLATED if some concrete input produces Y_0 > threshold.
fn box_input(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("well-formed input box")
}

/// Concrete forward evaluation via a degenerate (point) box: for a ReLU MLP,
/// IBP on a zero-width interval is exact.
fn forward(net: &Network, x: &[f32]) -> Vec<f32> {
    let pt = box_input(x, x);
    let out = net.propagate_ibp(&pt).expect("forward eval");
    out.lower().iter().copied().collect()
}

fn load_prop_network(name: &str) -> Network {
    let path = test_models_dir().join(name);
    require_model(&path);
    let nnet = load_nnet(&path).expect("parse nnet fixture");
    nnet.to_prop_network().expect("convert to prop network")
}

/// `crossing_relu.nnet`: 1 input, 1 output, single ReLU hidden layer.
///
/// Layers: Linear([[1],[-1]], b=0) -> ReLU -> Linear([[1,1]], b=0).
/// So f(x) = relu(x) + relu(-x) = |x|.
#[test]
fn crossing_relu_forward_is_abs() {
    let net = load_prop_network("crossing_relu.nnet");
    // Spot-check the |x| behavior the fixture encodes.
    for &x in &[-2.0f32, -0.5, 0.0, 0.5, 2.0] {
        let y = forward(&net, &[x]);
        assert_eq!(y.len(), 1, "crossing_relu has one output");
        assert!(
            (y[0] - x.abs()).abs() < 1e-4,
            "crossing_relu(x={x}) = {} expected |x| = {}",
            y[0],
            x.abs()
        );
    }
}

/// VERIFIED case: a sound property that no input in the box can violate.
///
/// On the box x in [-1, 1], f(x) = |x| in [0, 1], so the property `Y_0 <= 2`
/// holds for every input. CROWN must certify an upper bound <= 2.
#[test]
fn crossing_relu_safe_property_is_verified() {
    let net = load_prop_network("crossing_relu.nnet");
    let input = box_input(&[-1.0], &[1.0]);

    let crown = net.propagate_crown(&input).expect("crown");
    let upper = crown.upper()[0];

    const THRESHOLD: f32 = 2.0;
    assert!(
        upper <= THRESHOLD,
        "VERIFIED expected: certified upper bound {upper} should be <= {THRESHOLD} \
         (property Y_0 <= {THRESHOLD} on x in [-1,1] where f=|x| in [0,1]). \
         A regression here means CROWN got LOOSER or unsound."
    );
    // Sanity: the bound must also be sound (>= true max of 1.0).
    assert!(
        upper >= 1.0 - 1e-3,
        "certified upper bound {upper} is BELOW the true maximum 1.0 — UNSOUND"
    );
}

/// VIOLATED case: an unsafe property with a concrete counterexample.
///
/// On the box x in [-1, 1], f(x) = |x| reaches 1.0 at x = ±1, so the property
/// `Y_0 <= 0.5` is violated. We exhibit a concrete counterexample (the forward
/// pass) and confirm CROWN's certified upper bound cannot prove the property.
#[test]
fn crossing_relu_unsafe_property_is_violated() {
    let net = load_prop_network("crossing_relu.nnet");

    const THRESHOLD: f32 = 0.5;

    // Concrete counterexample at x = 1.0: f(1.0) = 1.0 > 0.5.
    let cex = forward(&net, &[1.0]);
    assert!(
        cex[0] > THRESHOLD,
        "expected a real violation: f(1.0) = {} should exceed {THRESHOLD}",
        cex[0]
    );

    // The verifier must NOT be able to certify Y_0 <= 0.5: its sound upper
    // bound must be > the threshold (a verifier returning "verified" here would
    // be unsound).
    let input = box_input(&[-1.0], &[1.0]);
    let crown = net.propagate_crown(&input).expect("crown");
    let upper = crown.upper()[0];
    assert!(
        upper > THRESHOLD,
        "VIOLATED expected: certified upper bound {upper} must exceed {THRESHOLD}; \
         certifying the property would be UNSOUND (real counterexample exists)"
    );
}

/// Cross-check IBP vs CROWN soundness on the multi-input fixture.
///
/// `minimal_relu.nnet`: 2 inputs, 1 output, one ReLU hidden layer of width 3.
/// We assert both engines produce well-formed bounds that CONTAIN concrete
/// forward outputs sampled across the input box — the core soundness invariant.
#[test]
fn minimal_relu_ibp_and_crown_contain_forward() {
    let net = load_prop_network("minimal_relu.nnet");
    let lower = [-1.0f32, -1.0];
    let upper = [1.0f32, 1.0];
    let input = box_input(&lower, &upper);

    let ibp = net.propagate_ibp(&input).expect("ibp");
    let crown = net.propagate_crown(&input).expect("crown");

    let ibp_lo = ibp.lower()[0];
    let ibp_hi = ibp.upper()[0];
    let crown_lo = crown.lower()[0];
    let crown_hi = crown.upper()[0];

    assert!(ibp_lo <= ibp_hi, "IBP bounds malformed");
    assert!(crown_lo <= crown_hi, "CROWN bounds malformed");

    // Sample the box (corners + center) and assert both engines contain f(x).
    let corners = [
        [lower[0], lower[1]],
        [lower[0], upper[1]],
        [upper[0], lower[1]],
        [upper[0], upper[1]],
        [0.0, 0.0],
    ];
    for c in &corners {
        let y = forward(&net, c)[0];
        let tol = 1e-3 * y.abs().max(1.0) + 1e-3;
        assert!(
            y >= ibp_lo - tol && y <= ibp_hi + tol,
            "IBP UNSOUND: f({c:?}) = {y} not in [{ibp_lo}, {ibp_hi}]"
        );
        assert!(
            y >= crown_lo - tol && y <= crown_hi + tol,
            "CROWN UNSOUND: f({c:?}) = {y} not in [{crown_lo}, {crown_hi}]"
        );
    }

    // CROWN should be no looser than IBP for this tiny net (it may tie).
    assert!(
        crown_hi <= ibp_hi + 1e-3 && crown_lo >= ibp_lo - 1e-3,
        "CROWN bounds [{crown_lo}, {crown_hi}] should be at least as tight as \
         IBP [{ibp_lo}, {ibp_hi}] for this network"
    );
}

// =============================================================================
// Offline scorecard / regression baseline (task 4)
// =============================================================================

/// A scored property: an input box and an output upper-bound constraint
/// `Y[output_idx] <= threshold`, plus the verdict we expect to (re)derive.
struct ScoredCase {
    /// nnet model file (key, with extension, matching the reference CSV).
    model: &'static str,
    /// vnnlib property file basename without extension (the reference key).
    property: &'static str,
    input_lower: &'static [f32],
    input_upper: &'static [f32],
    output_idx: usize,
    threshold: f32,
}

/// Recompute the verdict for one property using the real engine.
///
/// VERIFIED iff CROWN's certified upper bound on the output is <= threshold
/// (no input in the box can violate the constraint). Otherwise we look for a
/// concrete counterexample by sampling the box; if found, VIOLATED. If neither,
/// UNKNOWN. For these tiny deterministic fixtures the outcome is well-defined.
fn recompute_verdict(case: &ScoredCase) -> &'static str {
    let net = load_prop_network(case.model);
    let input = box_input(case.input_lower, case.input_upper);
    let crown = net.propagate_crown(&input).expect("crown");
    let certified_upper = crown.upper()[case.output_idx];

    if certified_upper <= case.threshold {
        return "verified";
    }

    // Search the box (corners + center) for a concrete counterexample.
    let n = case.input_lower.len();
    let mut found_violation = false;
    // Enumerate up to 2^n corners (n is tiny for these fixtures) plus center.
    let corners = 1usize << n;
    for mask in 0..corners {
        let pt: Vec<f32> = (0..n)
            .map(|i| {
                if mask & (1 << i) != 0 {
                    case.input_upper[i]
                } else {
                    case.input_lower[i]
                }
            })
            .collect();
        if forward(&net, &pt)[case.output_idx] > case.threshold {
            found_violation = true;
            break;
        }
    }
    let center: Vec<f32> = (0..n)
        .map(|i| f32::midpoint(case.input_lower[i], case.input_upper[i]))
        .collect();
    if forward(&net, &center)[case.output_idx] > case.threshold {
        found_violation = true;
    }

    if found_violation {
        "violated"
    } else {
        "unknown"
    }
}

/// Parse the committed `model,property,result` reference scorecard into rows.
fn load_reference() -> Vec<(String, String, String)> {
    let path = workspace_root().join("tests/fixtures/offline_scorecard_reference.csv");
    require_model(&path);
    let text = std::fs::read_to_string(&path).expect("read reference scorecard");
    text.lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split(',').map(|c| c.trim()).collect();
            assert_eq!(
                cols.len(),
                3,
                "reference row must be model,property,result: {l}"
            );
            (
                cols[0].to_string(),
                cols[1].to_string(),
                cols[2].to_string(),
            )
        })
        .collect()
}

/// Offline regression: recompute verdicts for every in-repo scorecard fixture
/// and compare against the committed reference.
///
/// This is the offline analogue of `scripts/validate_vnncomp_results.sh`: a
/// soundness regression (verified <-> violated flip) trips this test without
/// any model download. The committed reference lives at
/// `tests/fixtures/offline_scorecard_reference.csv`.
#[test]
fn offline_scorecard_matches_reference() {
    // The scored fixtures must correspond 1:1 to the committed reference rows.
    let cases = [
        ScoredCase {
            model: "crossing_relu.nnet",
            property: "crossing_relu_safe",
            input_lower: &[-1.0],
            input_upper: &[1.0],
            output_idx: 0,
            threshold: 2.0,
        },
        ScoredCase {
            model: "crossing_relu.nnet",
            property: "crossing_relu_unsafe",
            input_lower: &[-1.0],
            input_upper: &[1.0],
            output_idx: 0,
            threshold: 0.5,
        },
    ];

    // Sanity: the matching .vnnlib property files exist in-repo.
    for case in &cases {
        require_model(&test_models_dir().join(format!("{}.vnnlib", case.property)));
    }

    let reference = load_reference();
    assert_eq!(
        reference.len(),
        cases.len(),
        "scorecard reference row count must match the scored fixtures"
    );

    for case in &cases {
        let expected = reference
            .iter()
            .find(|(m, p, _)| m == case.model && p == case.property)
            .map(|(_, _, r)| r.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "no reference row for {}/{} — update the scorecard CSV",
                    case.model, case.property
                )
            });

        let actual = recompute_verdict(case);

        // A verified<->violated flip is the catastrophic VNN-COMP -1 case.
        let is_flip = matches!(
            (expected, actual),
            ("verified", "violated") | ("violated", "verified")
        );
        assert!(
            !is_flip,
            "SOUNDNESS REGRESSION for {}/{}: reference={expected} but recomputed={actual} \
             (verified<->violated flip)",
            case.model, case.property
        );
        assert_eq!(
            actual, expected,
            "scorecard mismatch for {}/{}: expected {expected}, got {actual}",
            case.model, case.property
        );
    }
}
