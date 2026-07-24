// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GAMA guidance-loss helpers (#1449), shared by the sequential disjunctive
//! attacker (this crate) and ny-cli's graph PGD lane.
//!
//! GAMA (Guided Adversarial Margin Attack, Sriramanan et al., NeurIPS 2020)
//! ascends `softmax_margin + λ·Σ_c (P_c − q_c)²` where `P` is the softmax at
//! the restart's initial point and `λ` anneals linearly from λ₀ to 0 over the
//! early steps. The guidance term pushes the output distribution away from
//! its initial state, escaping the masked-gradient plateaus of
//! adversarially-trained networks; once λ reaches 0 the objective reduces to
//! the plain margin. Attack-only: every candidate is re-validated before any
//! `sat`, so nothing here can affect a sound verdict.

use ndarray::ArrayD;

/// Numerically-stable softmax (f64 accumulation), returned flat in `out`'s order.
pub fn gama_softmax(out: &ArrayD<f32>) -> Vec<f32> {
    // Uniform fallback on ANY non-finite logit: `f32::max` skips NaN, so a
    // max-based guard alone lets a NaN slip through and poison every
    // component (attack-only, but a NaN objective wastes the restart).
    let m = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() || out.iter().any(|v| !v.is_finite()) {
        let n = out.len().max(1);
        return vec![1.0 / n as f32; out.len()];
    }
    let exps: Vec<f64> = out.iter().map(|&z| f64::from(z - m).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Number of steps over which λ anneals linearly from λ₀ to 0: the first
/// `NY_PGD_GAMA_LIN_FRAC` fraction of the step budget (default 0.25).
pub fn gama_lin_steps(num_steps: usize) -> usize {
    let frac: f32 = std::env::var("NY_PGD_GAMA_LIN_FRAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v: &f32| v.is_finite() && *v > 0.0 && *v <= 1.0)
        .unwrap_or(0.25);
    ((frac * num_steps as f32).round() as usize).max(1)
}

/// Annealed guidance weight at `step`: `λ₀·max(0, 1 − step/lin_steps)`.
pub fn gama_lambda_at(lambda0: f32, step: usize, lin_steps: usize) -> f32 {
    lambda0 * (1.0 - (step as f32) / (lin_steps as f32)).max(0.0)
}

/// GAMA guidance term `Σ_c (P_c − q_c)²`.
pub fn gama_guidance(q: &[f32], p_ref: &[f32]) -> f32 {
    q.iter()
        .zip(p_ref.iter())
        .map(|(&qi, &p)| (p - qi).powi(2))
        .sum()
}

/// Exact output-space cotangent of [`gama_guidance`] through softmax.
///
/// For `q = softmax(z)`, `P = p_ref`, and
/// `G(z) = sum_k (P_k - q_k)^2`, this returns `dG/dz`. Invalid or
/// non-finite probability vectors are rejected so callers can fail open to
/// their raw attack objective without poisoning a restart with NaNs.
pub fn gama_guidance_cotangent(q: &[f32], p_ref: &[f32]) -> Option<Vec<f32>> {
    if q.is_empty() || q.len() != p_ref.len() || q.iter().chain(p_ref).any(|v| !v.is_finite()) {
        return None;
    }
    let s: f32 = q
        .iter()
        .zip(p_ref)
        .map(|(&qk, &pk)| qk * qk - pk * qk)
        .sum();
    Some(
        q.iter()
            .zip(p_ref)
            .map(|(&qk, &pk)| 2.0 * qk * ((qk - pk) - s))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn gama_softmax_normalizes_and_orders() {
        let z = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0f32, 3.0, -2.0, 0.5]).unwrap();
        let q = gama_softmax(&z);
        let sum: f32 = q.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must normalize: {sum}");
        assert!(
            q[1] > q[0] && q[0] > q[3] && q[3] > q[2],
            "order-preserving"
        );
    }

    #[test]
    fn gama_softmax_non_finite_input_returns_uniform() {
        let z = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();
        let q = gama_softmax(&z);
        assert_eq!(q, vec![0.5, 0.5]);
    }

    #[test]
    fn gama_lambda_at_anneals_linearly_to_zero() {
        let lin = gama_lin_steps(100);
        assert_eq!(lin, 25, "default schedule is the first 25% of steps");
        assert_eq!(gama_lambda_at(50.0, 0, lin), 50.0);
        assert!((gama_lambda_at(50.0, lin / 5, lin) - 40.0).abs() < 1e-4);
        assert_eq!(gama_lambda_at(50.0, lin, lin), 0.0);
        assert_eq!(gama_lambda_at(50.0, 99, lin), 0.0, "clamped past lin_steps");
    }

    #[test]
    fn gama_guidance_zero_at_reference_positive_away() {
        let p = vec![0.7f32, 0.2, 0.1];
        assert_eq!(gama_guidance(&p, &p), 0.0);
        let q = vec![0.1f32, 0.2, 0.7];
        assert!(gama_guidance(&q, &p) > 0.0);
    }

    #[test]
    fn gama_guidance_cotangent_rejects_invalid_vectors() {
        assert!(gama_guidance_cotangent(&[], &[]).is_none());
        assert!(gama_guidance_cotangent(&[0.5, 0.5], &[1.0]).is_none());
        assert!(gama_guidance_cotangent(&[f32::NAN, 0.5], &[0.5, 0.5]).is_none());
    }
}
