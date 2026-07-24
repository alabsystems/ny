// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for probabilistic bound techniques against real ONNX networks.
//!
//! Exercises the 7 probabilistic techniques from `ny_propagate::probabilistic`
//! against ONNX models loaded via the ny-onnx pipeline:
//!
//! 1. Monte Carlo sampling + CROWN consistency
//! 2. Hoeffding concentration bounds
//! 3. McDiarmid concentration bounds (with Lipschitz estimate)
//! 4. Gaussian distributional propagation
//! 5. Cornish-Fisher moment propagation
//! 6. CGF/Chernoff propagation
//! 7. Branch-and-bound distributional refinement
//!
//! Part of #4265.

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::bounds::LinearBounds;
use ny_propagate::probabilistic::branch_and_bound::{
    refine_distributional_bounds, BranchAndBoundConfig,
};
use ny_propagate::probabilistic::cgf::propagate_cgf;
use ny_propagate::probabilistic::concentration::{
    estimate_lipschitz_from_network, ConcentrationCertificate,
};
use ny_propagate::probabilistic::distributional::{propagate_distribution, AnalyticDistribution};
use ny_propagate::probabilistic::distributional_alpha::propagate_distribution_tight;
use ny_propagate::probabilistic::moments::propagate_moments;
use ny_propagate::probabilistic::monte_carlo::MonteCarloVerifier;
use ny_propagate::Network as PropNetwork;
use ny_tensor::BoundedTensor;

/// Load simple_mlp.onnx (Linear(2->4) + ReLU + Linear(4->2)) and return
/// the propagation network, input bounds, CROWN output bounds, and linear bounds.
fn load_simple_mlp_with_crown() -> (PropNetwork, BoundedTensor, BoundedTensor, LinearBounds) {
    let path = require_test_model("simple_mlp.onnx");
    let model = load_onnx(&path).expect("Failed to load simple_mlp.onnx");
    let network = model
        .to_propagate_network()
        .expect("Failed to convert simple_mlp to propagation network");

    // Input bounds: [0, 1]^2 — small perturbation region
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let (crown_bounds, linear_bounds) = network
        .propagate_crown_with_linear(&input)
        .expect("CROWN propagation failed on simple_mlp");

    (network, input, crown_bounds, linear_bounds)
}

/// Load linear_relu.onnx (Linear(2->3) + ReLU) and return
/// the propagation network, input bounds, CROWN output bounds, and linear bounds.
fn load_linear_relu_with_crown() -> (PropNetwork, BoundedTensor, BoundedTensor, LinearBounds) {
    let path = require_test_model("linear_relu.onnx");
    let model = load_onnx(&path).expect("Failed to load linear_relu.onnx");
    let network = model
        .to_propagate_network()
        .expect("Failed to convert linear_relu to propagation network");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let (crown_bounds, linear_bounds) = network
        .propagate_crown_with_linear(&input)
        .expect("CROWN propagation failed on linear_relu");

    (network, input, crown_bounds, linear_bounds)
}

// ---------------------------------------------------------------------------
// Test 1: Monte Carlo sampling + CROWN consistency
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_monte_carlo_crown_consistency_simple_mlp() {
    let (network, input, crown_bounds, _linear_bounds) = load_simple_mlp_with_crown();

    let verifier = MonteCarloVerifier::new(500).with_seed(42);
    let samples = verifier
        .sample_inputs(&input)
        .expect("Monte Carlo sampling failed");

    // Evaluate network on each sample
    let outputs: Vec<ArrayD<f32>> = samples
        .iter()
        .map(|sample| {
            let concrete = BoundedTensor::new(sample.clone(), sample.clone()).unwrap();
            let result = network.propagate_ibp(&concrete).unwrap();
            result.lower().clone()
        })
        .collect();

    let prob_bound = verifier
        .compute_bounds(&outputs, Some(&crown_bounds))
        .expect("compute_bounds failed");

    // Soundness: empirical bounds must be consistent with CROWN bounds
    assert!(
        prob_bound.is_consistent(1e-5),
        "Monte Carlo empirical bounds exceed CROWN bounds: \
         emp_lower={:?}, emp_upper={:?}, crown_lower={:?}, crown_upper={:?}",
        prob_bound.empirical_lower,
        prob_bound.empirical_upper,
        prob_bound.crown_lower,
        prob_bound.crown_upper,
    );

    // Basic sanity: empirical lower <= empirical upper
    for i in 0..prob_bound.empirical_lower.len() {
        assert!(
            prob_bound.empirical_lower[[i]] <= prob_bound.empirical_upper[[i]],
            "empirical_lower[{i}]={} > empirical_upper[{i}]={}",
            prob_bound.empirical_lower[[i]],
            prob_bound.empirical_upper[[i]],
        );
    }

    // Mean should be between empirical lower and upper
    for i in 0..prob_bound.empirical_mean.len() {
        assert!(
            prob_bound.empirical_mean[[i]] >= prob_bound.empirical_lower[[i]],
            "mean[{i}] below empirical_lower"
        );
        assert!(
            prob_bound.empirical_mean[[i]] <= prob_bound.empirical_upper[[i]],
            "mean[{i}] above empirical_upper"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Hoeffding concentration bounds
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_hoeffding_bounds_simple_mlp() {
    let (network, input, crown_bounds, _linear_bounds) = load_simple_mlp_with_crown();

    // Run Monte Carlo to get empirical mean
    let verifier = MonteCarloVerifier::new(500).with_seed(42);
    let samples = verifier.sample_inputs(&input).unwrap();
    let outputs: Vec<ArrayD<f32>> = samples
        .iter()
        .map(|s| {
            let c = BoundedTensor::new(s.clone(), s.clone()).unwrap();
            network.propagate_ibp(&c).unwrap().lower().clone()
        })
        .collect();
    let prob_bound = verifier
        .compute_bounds(&outputs, Some(&crown_bounds))
        .unwrap();

    // Compute Hoeffding certificate
    let certificate = ConcentrationCertificate::compute(
        &prob_bound.empirical_mean,
        &crown_bounds,
        500,
        0.95,
        true, // Bonferroni correction
    )
    .expect("Hoeffding certificate computation failed");

    assert!(
        certificate.is_sound,
        "Hoeffding certificate should be sound"
    );

    // Each Hoeffding bound should have finite epsilon and valid failure probability
    for hb in &certificate.hoeffding_bounds {
        assert!(
            hb.epsilon.is_finite(),
            "Hoeffding epsilon[{}] is not finite: {}",
            hb.dimension,
            hb.epsilon
        );
        assert!(
            hb.epsilon >= 0.0,
            "Hoeffding epsilon[{}] is negative: {}",
            hb.dimension,
            hb.epsilon
        );
        assert!(
            hb.failure_probability >= 0.0 && hb.failure_probability <= 1.0,
            "Hoeffding failure_probability[{}] out of [0,1]: {}",
            hb.dimension,
            hb.failure_probability
        );
        assert_eq!(hb.num_samples, 500);
    }
}

// ---------------------------------------------------------------------------
// Test 3: McDiarmid concentration bounds (with Lipschitz estimate)
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_mcdiarmid_bounds_linear_relu() {
    let (network, input, crown_bounds, _linear_bounds) = load_linear_relu_with_crown();

    // Estimate Lipschitz constant
    let lipschitz = estimate_lipschitz_from_network(&network)
        .expect("Lipschitz estimation failed on linear_relu");

    assert!(
        lipschitz.value.is_finite() && lipschitz.value >= 0.0,
        "Lipschitz constant should be finite and non-negative, got {}",
        lipschitz.value
    );

    // linear_relu has only Linear + ReLU — both handled, so estimate should be sound
    assert!(
        lipschitz.is_sound,
        "Lipschitz estimate for Linear+ReLU should be sound, unhandled: {:?}",
        lipschitz.unhandled_layers
    );

    // Run Monte Carlo
    let verifier = MonteCarloVerifier::new(500).with_seed(42);
    let samples = verifier.sample_inputs(&input).unwrap();
    let outputs: Vec<ArrayD<f32>> = samples
        .iter()
        .map(|s| {
            let c = BoundedTensor::new(s.clone(), s.clone()).unwrap();
            network.propagate_ibp(&c).unwrap().lower().clone()
        })
        .collect();
    let prob_bound = verifier
        .compute_bounds(&outputs, Some(&crown_bounds))
        .unwrap();

    // Full certificate with McDiarmid
    let certificate = ConcentrationCertificate::compute_with_mcdiarmid(
        &prob_bound.empirical_mean,
        &crown_bounds,
        &prob_bound.empirical_mean, // Use mean as representative output
        &input,
        &lipschitz,
        500,
        0.95,
        true,
    )
    .expect("McDiarmid certificate computation failed");

    assert!(
        certificate.is_sound,
        "Certificate with sound Lipschitz should be sound"
    );
    let mcdiarmid = certificate
        .mcdiarmid_bounds
        .as_ref()
        .expect("McDiarmid bounds should be present");
    for mb in mcdiarmid {
        assert!(
            mb.epsilon.is_finite() && mb.epsilon >= 0.0,
            "McDiarmid epsilon[{}] invalid: {}",
            mb.dimension,
            mb.epsilon
        );
        assert!(mb.is_sound, "McDiarmid bound should be sound");
    }
}

// ---------------------------------------------------------------------------
// Test 4: Gaussian distributional propagation
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_gaussian_distributional_propagation_simple_mlp() {
    let (_network, input, crown_bounds, linear_bounds) = load_simple_mlp_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&linear_bounds, &dist, &input, 0.99)
        .expect("Distributional propagation failed on simple_mlp");

    let num_outputs = result.num_outputs();
    assert!(num_outputs > 0, "Should have at least one output dimension");

    for i in 0..num_outputs {
        // Mean bounds should be between CROWN lower and upper
        assert!(
            result.mean_lower[[i]] <= result.mean_upper[[i]],
            "mean_lower[{i}]={} > mean_upper[{i}]={}",
            result.mean_lower[[i]],
            result.mean_upper[[i]]
        );

        // Variance should be non-negative
        assert!(
            result.variance_upper[[i]] >= 0.0,
            "variance_upper[{i}] is negative: {}",
            result.variance_upper[[i]]
        );

        // Probabilistic bounds should contain the mean bounds
        assert!(
            result.prob_lower[[i]] <= result.mean_lower[[i]],
            "prob_lower[{i}]={} > mean_lower[{i}]={} — probabilistic bound should be wider",
            result.prob_lower[[i]],
            result.mean_lower[[i]]
        );
        assert!(
            result.prob_upper[[i]] >= result.mean_upper[[i]],
            "prob_upper[{i}]={} < mean_upper[{i}]={} — probabilistic bound should be wider",
            result.prob_upper[[i]],
            result.mean_upper[[i]]
        );

        assert!(
            result.prob_lower[[i]].is_finite(),
            "prob_lower[{i}] is not finite"
        );
        assert!(
            result.prob_upper[[i]].is_finite(),
            "prob_upper[{i}] is not finite"
        );
        assert!(
            result.prob_lower[[i]] <= result.prob_upper[[i]],
            "prob_lower[{i}]={} > prob_upper[{i}]={}",
            result.prob_lower[[i]],
            result.prob_upper[[i]]
        );

        // The mean relaxation bounds should remain inside the deterministic
        // CROWN interval. The Gaussian quantile interval is an approximation
        // and may be wider unless the API explicitly intersects it with CROWN.
        assert!(
            result.mean_lower[[i]] >= crown_bounds.lower()[[i]] - 1e-3,
            "mean_lower[{i}]={} below CROWN lower={}",
            result.mean_lower[[i]],
            crown_bounds.lower()[[i]]
        );
        assert!(
            result.mean_upper[[i]] <= crown_bounds.upper()[[i]] + 1e-3,
            "mean_upper[{i}]={} above CROWN upper={}",
            result.mean_upper[[i]],
            crown_bounds.upper()[[i]]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: Cornish-Fisher moment propagation
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_cornish_fisher_tighter_than_gaussian_simple_mlp() {
    let (_network, input, _crown_bounds, linear_bounds) = load_simple_mlp_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let moment_bound = propagate_moments(&linear_bounds, &dist, &input, 0.99)
        .expect("Moment propagation failed on simple_mlp");

    let num_outputs = moment_bound.mean_lower.len();
    assert!(num_outputs > 0);

    for i in 0..num_outputs {
        // CF bounds should be at least as tight as (or equal to) Gaussian bounds
        // CF lower >= Gaussian lower (tighter from below)
        assert!(
            moment_bound.prob_lower[[i]] >= moment_bound.prob_lower_gaussian[[i]] - 1e-6,
            "CF lower[{i}]={} < Gaussian lower[{i}]={} — CF should be at least as tight",
            moment_bound.prob_lower[[i]],
            moment_bound.prob_lower_gaussian[[i]]
        );
        // CF upper <= Gaussian upper (tighter from above)
        assert!(
            moment_bound.prob_upper[[i]] <= moment_bound.prob_upper_gaussian[[i]] + 1e-6,
            "CF upper[{i}]={} > Gaussian upper[{i}]={} — CF should be at least as tight",
            moment_bound.prob_upper[[i]],
            moment_bound.prob_upper_gaussian[[i]]
        );

        // Uniform distribution has excess kurtosis = -1.2 (platykurtic)
        // After linear transformation, kurtosis should be <= 0 for uniform inputs
        assert!(
            moment_bound.excess_kurtosis[[i]] <= 0.0 + 1e-6,
            "excess_kurtosis[{i}]={} should be <= 0 for uniform inputs",
            moment_bound.excess_kurtosis[[i]]
        );

        // Variance should be non-negative and finite
        assert!(
            moment_bound.variance_upper[[i]] >= 0.0 && moment_bound.variance_upper[[i]].is_finite(),
            "variance[{i}]={} should be finite and non-negative",
            moment_bound.variance_upper[[i]]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: CGF/Chernoff propagation
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_cgf_chernoff_bounds_finite_simple_mlp() {
    let (_network, input, _crown_bounds, linear_bounds) = load_simple_mlp_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let confidence = 0.99;

    let cgf_bound = propagate_cgf(&linear_bounds, &dist, &input, confidence)
        .expect("CGF propagation failed on simple_mlp");

    let moment_bound = propagate_moments(&linear_bounds, &dist, &input, confidence)
        .expect("Moment propagation failed on simple_mlp");

    let num_outputs = cgf_bound.prob_lower.len();
    assert!(num_outputs > 0);

    for i in 0..num_outputs {
        // CGF bounds should be finite
        assert!(
            cgf_bound.prob_lower[[i]].is_finite(),
            "CGF prob_lower[{i}] is not finite"
        );
        assert!(
            cgf_bound.prob_upper[[i]].is_finite(),
            "CGF prob_upper[{i}] is not finite"
        );

        // Lower <= upper
        assert!(
            cgf_bound.prob_lower[[i]] <= cgf_bound.prob_upper[[i]],
            "CGF prob_lower[{i}]={} > prob_upper[{i}]={}",
            cgf_bound.prob_lower[[i]],
            cgf_bound.prob_upper[[i]]
        );

        let cgf_width = cgf_bound.prob_upper[[i]] - cgf_bound.prob_lower[[i]];
        let cf_width = moment_bound.prob_upper[[i]] - moment_bound.prob_lower[[i]];
        assert!(
            cgf_width.is_finite() && cgf_width >= 0.0,
            "CGF width[{i}] invalid: {cgf_width}"
        );
        assert!(
            cf_width.is_finite() && cf_width >= 0.0,
            "CF width[{i}] invalid: {cf_width}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Branch-and-bound distributional refinement
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_branch_and_bound_tighter_than_single_region_simple_mlp() {
    let (_network, input, _crown_bounds, linear_bounds) = load_simple_mlp_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let confidence = 0.99;

    // Single-region baseline
    let single_region = propagate_distribution(&linear_bounds, &dist, &input, confidence).unwrap();

    // B&B refinement: provide a closure that re-computes CROWN for sub-regions.
    // For this integration test, we re-use the original linear bounds (since the
    // closure receives sub-region bounds and should re-compute CROWN). For a real
    // network this would call network.propagate_crown_with_linear on the sub-region.
    // Here we use the original linear bounds as a conservative approximation.
    let config = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence,
        use_probability_weighting: false,
        use_exact_conditional_moments: false,
        ..Default::default()
    };

    let crown_fn = |_sub_bounds: &BoundedTensor| -> Result<LinearBounds> {
        // Return the original linear bounds (conservative approximation).
        // In a real scenario this would re-run CROWN on the sub-region.
        Ok(linear_bounds.clone())
    };

    let bab_result = refine_distributional_bounds(&crown_fn, &input, &dist, &config)
        .expect("B&B refinement failed on simple_mlp");

    assert!(
        bab_result.iterations > 0,
        "B&B should run at least one iteration"
    );
    assert!(
        bab_result.num_regions >= 2,
        "B&B should produce at least 2 regions, got {}",
        bab_result.num_regions
    );

    let num_outputs = single_region.num_outputs();
    for i in 0..num_outputs {
        // B&B variance should be <= single-region variance
        // (splitting reduces variance via law of total variance)
        assert!(
            bab_result.bound.variance_upper[[i]] <= single_region.variance_upper[[i]] + 1e-4,
            "B&B variance[{i}]={} > single-region variance[{i}]={} — B&B should reduce variance",
            bab_result.bound.variance_upper[[i]],
            single_region.variance_upper[[i]]
        );

        // Mean bounds should be valid
        assert!(
            bab_result.bound.mean_lower[[i]] <= bab_result.bound.mean_upper[[i]],
            "B&B mean_lower[{i}] > mean_upper[{i}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: DA-CROWN path-specific variance tightening
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_da_crown_path_specific_at_least_as_tight_simple_mlp() {
    let (_network, input, _crown_bounds, linear_bounds) = load_simple_mlp_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let confidence = 0.99;

    let standard = propagate_distribution(&linear_bounds, &dist, &input, confidence).unwrap();
    let tight = propagate_distribution_tight(&linear_bounds, &dist, &input, confidence).unwrap();

    let num_outputs = standard.num_outputs();
    for i in 0..num_outputs {
        // DA-CROWN probabilistic interval should be at most as wide as standard
        let standard_width = standard.prob_upper[[i]] - standard.prob_lower[[i]];
        let tight_width = tight.prob_upper[[i]] - tight.prob_lower[[i]];
        assert!(
            tight_width <= standard_width + 1e-5,
            "DA-CROWN width[{i}]={tight_width} > standard width[{i}]={standard_width}"
        );

        // Both should have valid ordering
        assert!(
            tight.prob_lower[[i]] <= tight.prob_upper[[i]],
            "DA-CROWN prob_lower[{i}] > prob_upper[{i}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: Probabilistic techniques on linear_relu
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn test_probabilistic_techniques_finite_linear_relu() {
    let (_network, input, crown_bounds, linear_bounds) = load_linear_relu_with_crown();

    let dist = AnalyticDistribution::UniformFromBounds;
    let confidence = 0.99;

    // 1. CROWN interval (widest)
    let crown_width: Vec<f32> = crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect();

    // 2. Gaussian distributional bounds
    let gaussian = propagate_distribution(&linear_bounds, &dist, &input, confidence).unwrap();

    // 3. Cornish-Fisher moment bounds
    let cf = propagate_moments(&linear_bounds, &dist, &input, confidence).unwrap();

    // 4. CGF/Chernoff bounds (tightest analytical)
    let cgf = propagate_cgf(&linear_bounds, &dist, &input, confidence).unwrap();

    let num_outputs = gaussian.num_outputs();
    for i in 0..num_outputs {
        let gauss_width = gaussian.prob_upper[[i]] - gaussian.prob_lower[[i]];
        let cf_width = cf.prob_upper[[i]] - cf.prob_lower[[i]];
        let cgf_width = cgf.prob_upper[[i]] - cgf.prob_lower[[i]];

        assert!(
            crown_width[i].is_finite() && crown_width[i] >= 0.0,
            "dim {i}: invalid CROWN width {}",
            crown_width[i]
        );
        assert!(
            gauss_width.is_finite() && gauss_width >= 0.0,
            "dim {i}: invalid Gaussian width {gauss_width}"
        );
        assert!(
            cf_width.is_finite() && cf_width >= 0.0,
            "dim {i}: invalid CF width {cf_width}"
        );
        assert!(
            cgf_width.is_finite() && cgf_width >= 0.0,
            "dim {i}: invalid CGF width {cgf_width}"
        );

        // The Gaussian quantile interval is not intersected with deterministic
        // CROWN, so it can be wider than CROWN. Its mean relaxation bounds
        // should still sit inside the deterministic interval.
        assert!(
            gaussian.mean_lower[[i]] >= crown_bounds.lower()[[i]] - 1e-3,
            "dim {i}: Gaussian mean lower {} below CROWN lower {}",
            gaussian.mean_lower[[i]],
            crown_bounds.lower()[[i]]
        );
        assert!(
            gaussian.mean_upper[[i]] <= crown_bounds.upper()[[i]] + 1e-3,
            "dim {i}: Gaussian mean upper {} above CROWN upper {}",
            gaussian.mean_upper[[i]],
            crown_bounds.upper()[[i]]
        );

        // Cornish-Fisher is explicitly clamped to be no worse than the
        // Gaussian approximation in `propagate_moments`.
        assert!(
            gauss_width >= cf_width - 1e-3,
            "dim {i}: Gaussian width {gauss_width} < CF width {cf_width}"
        );
    }
}
