// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::VerificationSoundnessMode;

use crate::bounds::nan_propagating_max;

/// Softmax layer: y = softmax(x, dim)
///
/// Softmax normalizes inputs along a dimension so outputs sum to 1.
/// Uses Auto-LiRPA interval propagation algorithm for tight bounds.
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    /// Dimension along which to apply softmax (default: -1)
    pub axis: i32,
    /// Use sound (no sampling) relaxation for CROWN.
    ///
    /// When true, uses LSE-based affine bounds (sound) instead of heuristic sampling.
    pub sound: bool,
}

impl SoftmaxLayer {
    /// Create a new Softmax layer.
    pub fn new(axis: i32) -> Self {
        Self { axis, sound: true }
    }

    /// Enable or disable sound (no sampling) CROWN mode.
    pub fn with_sound_mode(mut self, enabled: bool) -> Self {
        self.sound = enabled;
        self
    }

    /// Enable heuristic sampling-based CROWN relaxation (not provably sound).
    pub fn with_heuristic_sampling(mut self, enabled: bool) -> Self {
        self.sound = !enabled;
        self
    }

    /// Returns the current verification soundness mode (Sound or Heuristic).
    pub fn soundness_mode(&self) -> VerificationSoundnessMode {
        if self.sound {
            VerificationSoundnessMode::Sound
        } else {
            VerificationSoundnessMode::Heuristic
        }
    }

    /// Evaluate softmax at a concrete point (1D).
    ///
    /// Returns softmax(x) = exp(x_i) / sum_j(exp(x_j))
    pub fn eval(&self, x: &Array1<f32>) -> Array1<f32> {
        // For numerical stability, subtract max
        let max_x = x.fold(f32::NEG_INFINITY, |a, &b| nan_propagating_max(a, b));
        let exp_x: Array1<f32> = x.mapv(|xi| (xi - max_x).exp());
        let sum_exp = exp_x.sum();
        exp_x.mapv(|ei| ei / sum_exp)
    }

    /// Compute the Jacobian of softmax at a point.
    ///
    /// For softmax: s_i = exp(x_i) / sum_j(exp(x_j))
    /// The Jacobian entry `J[i,j]` = ∂s_i/∂x_j:
    ///   `J[i,j]` = s_i * (δ_ij - s_j)
    /// where δ_ij = 1 if i=j, 0 otherwise.
    ///
    /// Diagonal:     `J[i,i]` = s_i * (1 - s_i)
    /// Off-diagonal: `J[i,j]` = -s_i * s_j  for i ≠ j
    pub fn jacobian(&self, x: &Array1<f32>) -> Array2<f32> {
        let s = self.eval(x);
        let n = s.len();
        let mut jacobian = Array2::<f32>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal: s_i * (1 - s_i)
                    jacobian[[i, j]] = s[i] * (1.0 - s[i]);
                } else {
                    // Off-diagonal: -s_i * s_j
                    jacobian[[i, j]] = -s[i] * s[j];
                }
            }
        }

        jacobian
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ========== eval() tests ==========

    #[test]
    fn softmax_eval_outputs_sum_to_one() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![1.0, 2.0, 3.0];
        let s = layer.eval(&x);
        assert!(
            (s.sum() - 1.0).abs() < 1e-6,
            "softmax outputs should sum to 1, got {}",
            s.sum()
        );
    }

    #[test]
    fn softmax_eval_outputs_in_unit_interval() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![-10.0, 0.0, 5.0, 100.0];
        let s = layer.eval(&x);
        for (i, &si) in s.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&si),
                "softmax[{}] = {} not in [0, 1]",
                i,
                si
            );
        }
    }

    #[test]
    fn softmax_eval_uniform_input() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![1.0, 1.0, 1.0, 1.0];
        let s = layer.eval(&x);
        for (i, &si) in s.iter().enumerate() {
            assert!(
                (si - 0.25).abs() < 1e-6,
                "softmax[{}] = {} should be 0.25 for uniform input",
                i,
                si
            );
        }
    }

    #[test]
    fn softmax_eval_large_input_stability() {
        // Numerical stability: large inputs should not overflow or produce NaN
        let layer = SoftmaxLayer::new(-1);
        let x = array![1000.0, 1001.0, 999.0];
        let s = layer.eval(&x);
        assert!(
            (s.sum() - 1.0).abs() < 1e-5,
            "softmax sum should be 1 for large inputs, got {}",
            s.sum()
        );
        for &si in s.iter() {
            assert!(
                !si.is_nan(),
                "softmax should not produce NaN for large inputs"
            );
        }
    }

    #[test]
    fn softmax_eval_single_dominant_element() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![100.0, 0.0, 0.0];
        let s = layer.eval(&x);
        // First element should be nearly 1, others nearly 0
        assert!(s[0] > 0.99, "dominant element softmax should be near 1");
        assert!(s[1] < 0.01, "non-dominant element should be near 0");
        assert!(s[2] < 0.01, "non-dominant element should be near 0");
    }

    #[test]
    fn softmax_eval_single_element() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![42.0];
        let s = layer.eval(&x);
        assert!(
            (s[0] - 1.0).abs() < 1e-6,
            "single element softmax should be 1.0"
        );
    }

    // ========== jacobian() tests ==========

    #[test]
    fn softmax_jacobian_rows_sum_to_zero() {
        // Each row of the softmax Jacobian sums to 0 because softmax outputs
        // are constrained to sum to 1 (constant sum → derivative of constraint = 0).
        let layer = SoftmaxLayer::new(-1);
        let x = array![1.0, 2.0, 3.0];
        let j = layer.jacobian(&x);
        for i in 0..j.nrows() {
            let row_sum: f32 = j.row(i).sum();
            assert!(
                row_sum.abs() < 1e-6,
                "Jacobian row {} sums to {}, expected 0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn softmax_jacobian_diagonal_is_si_times_one_minus_si() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![1.0, 2.0, 3.0];
        let s = layer.eval(&x);
        let j = layer.jacobian(&x);
        for i in 0..s.len() {
            let expected = s[i] * (1.0 - s[i]);
            assert!(
                (j[[i, i]] - expected).abs() < 1e-6,
                "Jacobian[{0},{0}] = {1}, expected s[{0}]*(1-s[{0}]) = {2}",
                i,
                j[[i, i]],
                expected
            );
        }
    }

    #[test]
    fn softmax_jacobian_off_diagonal_is_neg_si_sj() {
        let layer = SoftmaxLayer::new(-1);
        let x = array![1.0, 2.0, 3.0];
        let s = layer.eval(&x);
        let j = layer.jacobian(&x);
        for i in 0..s.len() {
            for k in 0..s.len() {
                if i != k {
                    let expected = -s[i] * s[k];
                    assert!(
                        (j[[i, k]] - expected).abs() < 1e-6,
                        "Jacobian[{},{}] = {}, expected -s[{}]*s[{}] = {}",
                        i,
                        k,
                        j[[i, k]],
                        i,
                        k,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn softmax_jacobian_symmetric() {
        // Jacobian is not symmetric in general, but J is symmetric iff s has specific structure.
        // However, J[i,j] = -s_i*s_j for i≠j, which IS symmetric because s_i*s_j = s_j*s_i.
        let layer = SoftmaxLayer::new(-1);
        let x = array![0.5, 1.5, -0.5, 2.0];
        let j = layer.jacobian(&x);
        for i in 0..j.nrows() {
            for k in 0..j.ncols() {
                assert!(
                    (j[[i, k]] - j[[k, i]]).abs() < 1e-6,
                    "Jacobian not symmetric: J[{},{}]={} vs J[{},{}]={}",
                    i,
                    k,
                    j[[i, k]],
                    k,
                    i,
                    j[[k, i]]
                );
            }
        }
    }

    // ========== soundness_mode tests ==========

    #[test]
    fn softmax_soundness_mode_defaults_to_sound() {
        let layer = SoftmaxLayer::new(-1);
        assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Sound);
    }

    #[test]
    fn softmax_soundness_mode_heuristic_toggle() {
        let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
        assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Heuristic);
        let layer = layer.with_sound_mode(true);
        assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Sound);
    }
}
