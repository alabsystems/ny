// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::super::*;
use crate::types::PropagationMethod;
use ny_core::{Bound, VerificationSpec};

#[ntest::timeout(5000)]
#[test]
fn test_verifier_new_default_config() {
    let config = PropagationConfig::default();
    let verifier = Verifier::new(config);
    // Default method is AlphaCrown
    assert!(matches!(
        verifier.config.method,
        PropagationMethod::AlphaCrown
    ));
}

#[ntest::timeout(5000)]
#[test]
fn test_verifier_new_crown_config() {
    let config = PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    assert!(matches!(verifier.config.method, PropagationMethod::Crown));
}

#[ntest::timeout(5000)]
#[test]
fn test_verifier_new_alpha_crown_config() {
    let config = PropagationConfig {
        method: PropagationMethod::AlphaCrown,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    assert!(matches!(
        verifier.config.method,
        PropagationMethod::AlphaCrown
    ));
}

#[ntest::timeout(5000)]
#[test]
fn test_verifier_new_sdp_crown_config() {
    let config = PropagationConfig {
        method: PropagationMethod::SdpCrown,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    assert!(matches!(
        verifier.config.method,
        PropagationMethod::SdpCrown
    ));
}

#[ntest::timeout(5000)]
#[test]
fn test_verifier_new_beta_crown_config() {
    let config = PropagationConfig {
        method: PropagationMethod::BetaCrown,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    assert!(matches!(
        verifier.config.method,
        PropagationMethod::BetaCrown
    ));
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_config_uses_propagation_config() {
    let config = PropagationConfig {
        method: PropagationMethod::AlphaCrown,
        max_iterations: 7,
        tolerance: 1e-3,
        use_gpu: false,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    let alpha_config = verifier.alpha_crown_config(None);
    assert_eq!(alpha_config.iterations, 7);
    assert!((alpha_config.tolerance - 1e-3).abs() < 1e-8);
    assert!(alpha_config.deadline.is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_crown_config_uses_propagation_config() {
    let config = PropagationConfig {
        method: PropagationMethod::BetaCrown,
        max_iterations: 9,
        tolerance: 5e-4,
        use_gpu: false,
        ..Default::default()
    };
    let verifier = Verifier::new(config);
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0)],
        vec![Bound::new(-1.0, 1.0)],
        Some(1234),
        None,
    )
    .expect("valid test spec");
    let beta_config = verifier.beta_crown_config(&spec);
    assert_eq!(beta_config.alpha_config.iterations, 9);
    assert!((beta_config.alpha_config.tolerance - 5e-4).abs() < 1e-8);
    assert_eq!(beta_config.beta_iterations, 9);
    assert!((beta_config.beta_tolerance - 5e-4).abs() < 1e-8);
    assert!(beta_config.root_beta_iterations <= 9);
    assert_eq!(beta_config.timeout, std::time::Duration::from_millis(1234));
}
