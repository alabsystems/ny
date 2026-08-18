// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Acceptance tests for the fail-closed deployed-precision verification entry
//! point (`verify_with_sound_precision`).
//!
//! Non-F32 policies remain experimental and must be rejected until every
//! deployed store and reduction is modeled rigorously. The all-F32 policy is a
//! strict no-op equal to the ordinary sound verifier.

use ndarray::{arr2, Array2};
use ny_api::graph::{GraphNetwork, GraphNode};
use ny_api::layers::{Layer, LinearLayer};
use ny_api::precision::{verify_with_sound_precision, FloatPrecision, MixedPrecisionPolicy};
use ny_api::verify::{PropagationConfig, Verifier};
use ny_api::{Bound, NyError, VerificationResult, VerificationSoundnessMode, VerificationSpec};

/// Build a single-`Linear` graph network computing `y = W x` (no bias).
/// `weight` has shape `[out_features, in_features]`.
fn single_linear_graph(weight: Array2<f32>) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    let lin = LinearLayer::new(weight, None).expect("valid linear layer");
    g.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
    g.set_output("lin");
    g
}

/// A spec with the given per-element input interval and a wide-open output
/// requirement (so the verdict is governed only by containment, not the bound).
fn spec_with_input(input: Vec<Bound>, n_out: usize) -> VerificationSpec {
    let shape = vec![input.len()];
    let out = vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); n_out];
    VerificationSpec::from_parts(input, out, None, Some(shape)).expect("valid spec")
}

/// Extract the computed/effective output bounds from any verdict variant.
fn effective_bounds(result: &VerificationResult) -> Vec<Bound> {
    match result {
        VerificationResult::Verified { output_bounds, .. } => output_bounds.clone(),
        VerificationResult::Unknown { bounds, .. } => bounds.clone(),
        VerificationResult::Timeout { partial_bounds, .. } => {
            partial_bounds.clone().unwrap_or_default()
        }
        VerificationResult::Violated { .. } => vec![],
    }
}

fn assert_non_f32_policy_rejected(
    graph: &GraphNetwork,
    spec: &VerificationSpec,
    policy: &MixedPrecisionPolicy,
) {
    let error = verify_with_sound_precision(graph, spec, policy)
        .expect_err("non-F32 verification must fail closed");
    match error {
        NyError::UnsupportedConfiguration(message) => {
            assert!(
                message.contains("not yet implemented"),
                "rejection should explain the unsupported proof obligation: {message}"
            );
        }
        other => panic!("expected UnsupportedConfiguration, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Counterexample 1 formerly exercised an experimental f16 widening path.
// The public proof API must reject it until that path is fully sound.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_1_f16_5000_ones_sum_is_rejected() {
    let n = 5000usize;
    // A 1x5000 all-ones weight makes y = sum_i x_i.
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    // x_i = exactly 1.0 (representable in f16), point input.
    let input = vec![Bound::new(1.0, 1.0); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    assert_non_f32_policy_rejected(&g, &spec, &policy);
}

// ---------------------------------------------------------------------------
// Counterexample 2: a long f16 dot product with substantial reduction drift.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_2_f16_4096_dot_drift_is_rejected() {
    let n = 4096usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let x = 0.333_251_95_f32; // exactly representable in f16
    let input = vec![Bound::new(x, x); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    assert_non_f32_policy_rejected(&g, &spec, &policy);
}

// ---------------------------------------------------------------------------
// Counterexample 3: a bf16 interval dot product whose deployed corners differ
// from the idealized f32 computation.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_3_bf16_512_dot_corner_is_rejected() {
    let n = 512usize;
    let weight = Array2::from_elem((1, n), 0.5_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(0.9, 1.1); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::Bf16);

    assert_non_f32_policy_rejected(&g, &spec, &policy);
}

// ---------------------------------------------------------------------------
// F32 policy is a strict no-op: identical to the normal graph verdict.
// ---------------------------------------------------------------------------
#[test]
fn f32_policy_equals_normal_verdict() {
    let weight = arr2(&[[2.0_f32, -1.0], [0.5, 0.5]]);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
    // Requirement loose enough to be Verified.
    let req = vec![Bound::new(-10.0, 10.0), Bound::new(-10.0, 10.0)];
    let spec = VerificationSpec::from_parts(input, req, None, Some(vec![2])).expect("valid spec");
    let policy = MixedPrecisionPolicy::default(); // all-F32

    let sound = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    let normal = Verifier::new(PropagationConfig::default())
        .verify_graph(&g, &spec)
        .expect("normal verify");

    // Same variant and same bounds, byte-for-byte.
    let sb = effective_bounds(&sound);
    let nb = effective_bounds(&normal);
    assert_eq!(sb.len(), nb.len());
    for (a, b) in sb.iter().zip(nb.iter()) {
        assert_eq!(a.lower().to_bits(), b.lower().to_bits());
        assert_eq!(a.upper().to_bits(), b.upper().to_bits());
    }
    assert_eq!(sound.is_verified(), normal.is_verified());
    // F32 no-op preserves the normal (Sound) provenance.
    if let VerificationResult::Verified { provenance, .. } = &sound {
        assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
    } else {
        panic!("expected Verified under loose requirement");
    }
}

// ---------------------------------------------------------------------------
// A property that the deployed computation violates must be rejected before a
// verification verdict is produced.
// ---------------------------------------------------------------------------
#[test]
fn non_f32_path_rejects_a_deployed_violation_query() {
    let n = 5000usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(1.0, 1.0); n];
    // Property: output in [4999, 5001]. The f32-idealized output (5000) satisfies
    // this, but the deployed f16 value (~2048) does NOT.
    let req = vec![Bound::new(4999.0, 5001.0)];
    let spec = VerificationSpec::from_parts(input, req, None, Some(vec![n])).expect("valid spec");
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    assert_non_f32_policy_rejected(&g, &spec, &policy);

    // Sanity: the f32-idealized verdict WOULD verify the same property.
    let normal = Verifier::new(PropagationConfig::default())
        .verify_graph(&g, &spec)
        .expect("normal verify");
    assert!(
        normal.is_verified(),
        "the f32-idealized verdict should verify [4999, 5001] for a sum of 5000 ones"
    );
}

// ---------------------------------------------------------------------------
// Even an open output requirement cannot turn an unsupported non-F32 execution
// policy into a proof.
// ---------------------------------------------------------------------------
#[test]
fn non_f32_path_rejects_even_an_open_output_requirement() {
    let n = 5000usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(1.0, 1.0); n];
    let req = vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)];
    let spec = VerificationSpec::from_parts(input, req, None, Some(vec![n])).expect("valid spec");
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    assert_non_f32_policy_rejected(&g, &spec, &policy);
}
