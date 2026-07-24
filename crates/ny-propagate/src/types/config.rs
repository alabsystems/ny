// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for bound propagation.

use ny_core::MethodUsed;
use serde::{Deserialize, Serialize};

/// Configuration for bound propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationConfig {
    /// Which propagation method to use.
    pub method: PropagationMethod,
    /// Maximum iterations for optimization-based methods.
    pub max_iterations: usize,
    /// Convergence threshold.
    pub tolerance: f32,
    /// Whether to use GPU acceleration (future).
    pub use_gpu: bool,
    /// Relaxation mode for MulBinary CROWN propagation.
    /// Defaults to McCormick for backward compatibility.
    #[serde(default)]
    pub mul_binary_relaxation: MulBinaryRelaxationMode,
    /// Use f64 (double precision) for all bound propagation.
    /// Required for soundnessbench/sat_relu. Only supports sequential Linear+Conv2D+ReLU.
    /// Reference: alpha-beta-CROWN `double_fp: true` (`abcrown.py:81-82`).
    #[serde(default)]
    pub double_fp: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            method: PropagationMethod::AlphaCrown,
            max_iterations: 100,
            tolerance: 1e-4,
            use_gpu: false,
            mul_binary_relaxation: MulBinaryRelaxationMode::default(),
            double_fp: false,
        }
    }
}

/// Available bound propagation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationMethod {
    /// Interval Bound Propagation: fastest, loosest.
    Ibp,
    /// CROWN: linear relaxation.
    Crown,
    /// α-CROWN: optimized linear relaxation.
    AlphaCrown,
    /// SDP-CROWN: tighter LiRPA for ℓ2 input sets (Linear/ReLU only for now).
    SdpCrown,
    /// β-CROWN: branch and bound.
    BetaCrown,
}

impl PropagationMethod {
    pub(crate) fn method_used(self) -> MethodUsed {
        match self {
            PropagationMethod::Ibp => MethodUsed::Ibp,
            PropagationMethod::Crown => MethodUsed::Crown,
            PropagationMethod::AlphaCrown => MethodUsed::AlphaCrown,
            PropagationMethod::SdpCrown => MethodUsed::SdpCrown,
            PropagationMethod::BetaCrown => MethodUsed::BetaCrown,
        }
    }
}

/// Relaxation mode for MulBinary CROWN propagation.
///
/// Controls which linear relaxation strategy is used for element-wise
/// multiplication of two bounded inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MulBinaryRelaxationMode {
    /// McCormick envelope: selects among four facets based on weight sign
    /// and bound direction. Default behavior, matches classic CROWN.
    #[default]
    McCormick,
    /// Middle interpolation: uses fixed coefficients with interpolation
    /// parameter 0.5, matching auto_LiRPA's `mul.middle` option.
    Middle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_default() {
        let config = PropagationConfig::default();
        assert_eq!(config.method, PropagationMethod::AlphaCrown);
        assert_eq!(config.max_iterations, 100);
        assert!((config.tolerance - 1e-4).abs() < 1e-8);
        assert!(!config.use_gpu);
        assert_eq!(
            config.mul_binary_relaxation,
            MulBinaryRelaxationMode::McCormick
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_custom() {
        let config = PropagationConfig {
            method: PropagationMethod::Ibp,
            max_iterations: 50,
            tolerance: 1e-6,
            use_gpu: true,
            ..Default::default()
        };
        assert_eq!(config.method, PropagationMethod::Ibp);
        assert_eq!(config.max_iterations, 50);
        assert!(config.use_gpu);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_clone() {
        let config = PropagationConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.method, config.method);
        assert_eq!(cloned.max_iterations, config.max_iterations);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_serialization() {
        let config = PropagationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PropagationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.method, config.method);
        assert_eq!(deserialized.max_iterations, config.max_iterations);
        assert_eq!(
            deserialized.mul_binary_relaxation,
            config.mul_binary_relaxation
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_serde_default_mul_binary_relaxation() {
        // Test that omitting mul_binary_relaxation from JSON uses default (McCormick)
        let json =
            r#"{"method":"AlphaCrown","max_iterations":100,"tolerance":0.0001,"use_gpu":false}"#;
        let deserialized: PropagationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            deserialized.mul_binary_relaxation,
            MulBinaryRelaxationMode::McCormick
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_config_with_middle_relaxation() {
        let config = PropagationConfig {
            mul_binary_relaxation: MulBinaryRelaxationMode::Middle,
            ..Default::default()
        };
        assert_eq!(
            config.mul_binary_relaxation,
            MulBinaryRelaxationMode::Middle
        );

        // Verify round-trip serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PropagationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.mul_binary_relaxation,
            MulBinaryRelaxationMode::Middle
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_mul_binary_relaxation_mode_default() {
        assert_eq!(
            MulBinaryRelaxationMode::default(),
            MulBinaryRelaxationMode::McCormick
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_mul_binary_relaxation_mode_equality() {
        assert_eq!(
            MulBinaryRelaxationMode::McCormick,
            MulBinaryRelaxationMode::McCormick
        );
        assert_eq!(
            MulBinaryRelaxationMode::Middle,
            MulBinaryRelaxationMode::Middle
        );
        assert_ne!(
            MulBinaryRelaxationMode::McCormick,
            MulBinaryRelaxationMode::Middle
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_mul_binary_relaxation_mode_serialization() {
        let mode = MulBinaryRelaxationMode::Middle;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, r#""Middle""#);

        let deserialized: MulBinaryRelaxationMode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, mode);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_method_equality() {
        assert_eq!(PropagationMethod::Ibp, PropagationMethod::Ibp);
        assert_ne!(PropagationMethod::Ibp, PropagationMethod::Crown);
        assert_ne!(PropagationMethod::AlphaCrown, PropagationMethod::BetaCrown);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_method_all_variants() {
        let methods = [
            PropagationMethod::Ibp,
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown,
            PropagationMethod::SdpCrown,
            PropagationMethod::BetaCrown,
        ];
        // All methods should be distinct
        for (i, m1) in methods.iter().enumerate() {
            for (j, m2) in methods.iter().enumerate() {
                if i == j {
                    assert_eq!(m1, m2);
                } else {
                    assert_ne!(m1, m2);
                }
            }
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_method_method_used_mapping() {
        assert_eq!(PropagationMethod::Ibp.method_used(), MethodUsed::Ibp);
        assert_eq!(PropagationMethod::Crown.method_used(), MethodUsed::Crown);
        assert_eq!(
            PropagationMethod::AlphaCrown.method_used(),
            MethodUsed::AlphaCrown
        );
        assert_eq!(
            PropagationMethod::SdpCrown.method_used(),
            MethodUsed::SdpCrown
        );
        assert_eq!(
            PropagationMethod::BetaCrown.method_used(),
            MethodUsed::BetaCrown
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagation_method_serialization() {
        let method = PropagationMethod::SdpCrown;
        let json = serde_json::to_string(&method).unwrap();
        let deserialized: PropagationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, method);
    }
}
