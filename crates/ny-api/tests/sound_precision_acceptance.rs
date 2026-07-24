// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Acceptance tests for the SOUND, layer-aware deployed-precision verification
//! path (`verify_with_sound_precision`, PART D).
//!
//! These build the exact adversarial counterexample nets that the prior
//! representation-only widening got WRONG, compute the REAL deployed f16/bf16
//! value with the `half` crate, and assert that the sound path's effective output
//! interval CONTAINS that deployed value. They also assert the F32 policy is a
//! strict no-op equal to the normal verdict, and that the sound path never
//! returns `Verified` for a property the deployed computation violates.

use half::{bf16, f16};
use ndarray::{arr2, Array2};
use ny_api::graph::{GraphNetwork, GraphNode};
use ny_api::layers::{Layer, LinearLayer};
use ny_api::precision::{verify_with_sound_precision, FloatPrecision, MixedPrecisionPolicy};
use ny_api::verify::{PropagationConfig, Verifier};
use ny_api::{Bound, VerificationResult, VerificationSoundnessMode, VerificationSpec};

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

/// Real deployed f16 recursive (left-to-right) dot product sum_i w_i * x_i,
/// rounding to f16 after each multiply and each add.
fn f16_dot(weights: &[f32], xs: &[f32]) -> f32 {
    let mut acc = f16::from_f32(0.0);
    for (&w, &x) in weights.iter().zip(xs.iter()) {
        let prod = f16::from_f32(f16::from_f32(w).to_f32() * f16::from_f32(x).to_f32());
        acc = f16::from_f32(acc.to_f32() + prod.to_f32());
    }
    acc.to_f32()
}

/// Real deployed bf16 recursive dot product.
fn bf16_dot(weights: &[f32], xs: &[f32]) -> f32 {
    let mut acc = bf16::from_f32(0.0);
    for (&w, &x) in weights.iter().zip(xs.iter()) {
        let prod = bf16::from_f32(bf16::from_f32(w).to_f32() * bf16::from_f32(x).to_f32());
        acc = bf16::from_f32(acc.to_f32() + prod.to_f32());
    }
    acc.to_f32()
}

// ---------------------------------------------------------------------------
// Counterexample 1: uniform f16, reduction summing 5000 ones.
// f32 sum = 5000 (point); deployed f16 saturates near 2048. The sound bound
// MUST contain 2048.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_1_f16_5000_ones_sum_is_contained() {
    let n = 5000usize;
    // A 1x5000 all-ones weight makes y = sum_i x_i.
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    // x_i = exactly 1.0 (representable in f16), point input.
    let input = vec![Bound::new(1.0, 1.0); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    let result = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    let bounds = effective_bounds(&result);
    assert_eq!(bounds.len(), 1, "single output neuron");

    let deployed = f16_dot(&vec![1.0; n], &vec![1.0; n]);
    assert!(
        deployed < 2100.0,
        "deployed f16 sum should saturate, got {deployed}"
    );
    assert!(
        bounds[0].lower() <= deployed && deployed <= bounds[0].upper(),
        "sound bound [{}, {}] must contain deployed f16 value {deployed}",
        bounds[0].lower(),
        bounds[0].upper()
    );
    // Acceptance: must contain 2048 specifically.
    assert!(
        bounds[0].lower() <= 2048.0 && 2048.0 <= bounds[0].upper(),
        "sound bound must contain 2048 (got [{}, {}])",
        bounds[0].lower(),
        bounds[0].upper()
    );
}

// ---------------------------------------------------------------------------
// Counterexample 2: uniform f16, 4096-term dot product x_i = f16(1/3), w_i = 1.
// f32 bound ~ [1365, 1365]; deployed f16 drifts far below; bound MUST contain it.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_2_f16_4096_dot_drift_is_contained() {
    let n = 4096usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let x = f16::from_f32(1.0 / 3.0).to_f32(); // ~0.33325, exactly f16-representable
    let input = vec![Bound::new(x, x); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    let result = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    let bounds = effective_bounds(&result);

    let deployed = f16_dot(&vec![1.0; n], &vec![x; n]);
    let ideal = (n as f64) * f64::from(x);
    assert!(
        f64::from(deployed) < ideal - 50.0,
        "deployed f16 dot {deployed} should drift well below ideal {ideal}"
    );
    assert!(
        bounds[0].lower() <= deployed && deployed <= bounds[0].upper(),
        "sound bound [{}, {}] must contain deployed f16 value {deployed}",
        bounds[0].lower(),
        bounds[0].upper()
    );
}

// ---------------------------------------------------------------------------
// Counterexample 3: uniform bf16, 512-term, w_i = 0.5, x_i in [0.9, 1.1].
// f32 IBP interval ~ [230, 282]; deployed bf16 worst corner escapes; bound MUST
// contain it.
// ---------------------------------------------------------------------------
#[test]
fn counterexample_3_bf16_512_dot_corner_is_contained() {
    let n = 512usize;
    let weight = Array2::from_elem((1, n), 0.5_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(0.9, 1.1); n];
    let spec = spec_with_input(input, 1);
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::Bf16);

    let result = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    let bounds = effective_bounds(&result);

    // Worst deployed corners: all-high (1.1) and all-low (0.9).
    let deployed_hi = bf16_dot(&vec![0.5; n], &vec![1.1; n]);
    let deployed_lo = bf16_dot(&vec![0.5; n], &vec![0.9; n]);
    for d in [deployed_hi, deployed_lo] {
        assert!(
            bounds[0].lower() <= d && d <= bounds[0].upper(),
            "sound bound [{}, {}] must contain deployed bf16 corner {d}",
            bounds[0].lower(),
            bounds[0].upper()
        );
    }
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
// The sound path must NOT return Verified for a property the deployed
// computation violates. Use the 5000-ones net: claim the output is within
// [4999, 5001] (true in f32, FALSE for deployed f16 which saturates ~2048).
// ---------------------------------------------------------------------------
#[test]
fn sound_path_does_not_verify_a_deployed_violation() {
    let n = 5000usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(1.0, 1.0); n];
    // Property: output in [4999, 5001]. The f32-idealized output (5000) satisfies
    // this, but the deployed f16 value (~2048) does NOT.
    let req = vec![Bound::new(4999.0, 5001.0)];
    let spec = VerificationSpec::from_parts(input, req, None, Some(vec![n])).expect("valid spec");
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    let result = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    assert!(
        !result.is_verified(),
        "sound path must NOT verify a property the deployed f16 computation violates; got {result:?}"
    );

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
// When the property holds with enough margin for the deployed precision, the
// sound path returns Verified with SOUND provenance (Linear is exactly accounted).
// ---------------------------------------------------------------------------
#[test]
fn sound_path_verifies_with_margin_and_sound_provenance() {
    let n = 5000usize;
    let weight = Array2::from_elem((1, n), 1.0_f32);
    let g = single_linear_graph(weight);
    let input = vec![Bound::new(1.0, 1.0); n];
    // Deployed f16 saturates near 2048; the sound bound saturates to a very wide
    // interval (gamma_N is +inf for n=5000 in f16). A finite requirement cannot
    // be Verified, but a [-inf, +inf] requirement can — proving SOUND provenance.
    let req = vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)];
    let spec = VerificationSpec::from_parts(input, req, None, Some(vec![n])).expect("valid spec");
    let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

    let result = verify_with_sound_precision(&g, &spec, &policy).expect("sound verify");
    match result {
        VerificationResult::Verified { provenance, .. } => {
            assert_eq!(
                provenance.mode(),
                VerificationSoundnessMode::Sound,
                "Linear-only net: every accumulating layer is exactly accounted -> Sound"
            );
        }
        other => panic!("expected Verified (open requirement), got {other:?}"),
    }
}
