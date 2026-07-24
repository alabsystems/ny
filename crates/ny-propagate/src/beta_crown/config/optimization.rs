// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for β-CROWN branch-and-bound search.

use ny_core::{NyError, Result};
use serde::{Deserialize, Serialize};

/// Learning rate scheduler for adaptive optimization.
///
/// Controls how the base learning rate changes over iterations, enabling
/// exploration-exploitation tradeoffs during optimization.
///
/// **Reference parity note for β-CROWN:**
///
/// alpha-beta-CROWN uses `ExponentialLR(ny=0.98)` during bounds optimization
/// (`optimized_bounds.py:74,498`). `AdaptiveOptConfig::default()` follows that
/// reference choice, while callers can still opt into `Constant` or other
/// schedules for workload-specific tuning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LRScheduler {
    /// Constant learning rate (no decay).
    #[default]
    Constant,

    /// Step decay: multiply LR by `ny` every `step_size` iterations.
    /// LR(t) = base_lr * ny^(floor(t / step_size))
    StepDecay {
        /// Multiplicative factor for decay (typically 0.1-0.5).
        ny: f32,
        /// Number of iterations between decay steps.
        step_size: usize,
    },

    /// Exponential decay: LR(t) = base_lr * ny^t
    ExponentialDecay {
        /// Per-iteration decay factor (typically 0.9-0.99).
        ny: f32,
    },

    /// Cosine annealing: smooth decay from base_lr to min_lr following cosine curve.
    /// LR(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t / T_max))
    CosineAnnealing {
        /// Minimum learning rate at the end of annealing.
        min_lr: f32,
        /// Total number of iterations for full cosine cycle.
        t_max: usize,
    },

    /// Linear warmup followed by cosine annealing.
    /// For t < warmup_steps: LR(t) = base_lr * t / warmup_steps
    /// For t >= warmup_steps: cosine annealing to min_lr
    WarmupCosine {
        /// Number of warmup iterations.
        warmup_steps: usize,
        /// Minimum learning rate after warmup.
        min_lr: f32,
        /// Total iterations including warmup.
        t_max: usize,
    },
}

impl LRScheduler {
    /// Compute the learning rate multiplier for the given iteration.
    ///
    /// Returns a factor in [0, 1] that should be multiplied with the base learning rate.
    /// Iteration `t` is 0-indexed.
    pub fn lr_factor(&self, t: usize, base_lr: f32) -> f32 {
        match self {
            LRScheduler::Constant => 1.0,

            LRScheduler::StepDecay { ny, step_size } => {
                if *step_size == 0 {
                    return 1.0;
                }
                let num_decays = (t / step_size).min(i32::MAX as usize);
                ny.powi(num_decays as i32)
            }

            LRScheduler::ExponentialDecay { ny } => ny.powi(t.min(i32::MAX as usize) as i32),

            LRScheduler::CosineAnnealing { min_lr, t_max } => {
                if *t_max == 0 {
                    return 1.0;
                }
                let progress = (t as f32 / *t_max as f32).min(1.0);
                let cosine = (std::f32::consts::PI * progress).cos();
                // Returns factor such that: min_lr + factor * (base_lr - min_lr) gives desired LR
                // Desired: min_lr + 0.5 * (base_lr - min_lr) * (1 + cos)
                // = base_lr * [min_lr/base_lr + 0.5 * (1 - min_lr/base_lr) * (1 + cos)]
                let min_ratio = min_lr / base_lr.max(1e-10);
                min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + cosine)
            }

            LRScheduler::WarmupCosine {
                warmup_steps,
                min_lr,
                t_max,
            } => {
                if t < *warmup_steps {
                    // Linear warmup: 0 -> 1 over warmup_steps
                    (t + 1) as f32 / (*warmup_steps).max(1) as f32
                } else {
                    // Cosine annealing after warmup
                    let t_after_warmup = t - warmup_steps;
                    let t_max_after_warmup = t_max.saturating_sub(*warmup_steps);
                    if t_max_after_warmup == 0 {
                        return 1.0;
                    }
                    let progress = (t_after_warmup as f32 / t_max_after_warmup as f32).min(1.0);
                    let cosine = (std::f32::consts::PI * progress).cos();
                    let min_ratio = min_lr / base_lr.max(1e-10);
                    min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + cosine)
                }
            }
        }
    }

    /// The actual learning rate for iteration `t`.
    pub fn lr(&self, t: usize, base_lr: f32) -> f32 {
        base_lr * self.lr_factor(t, base_lr)
    }
}

/// Strategy for scaling learning rates across different layers.
///
/// Different layers in a neural network may benefit from different learning rates.
/// Early layers (close to input) typically need smaller learning rates because
/// their changes propagate through more downstream computations. Later layers
/// (close to output) can often use larger learning rates.
///
/// Reference: Layer-wise adaptive learning rates are used in various deep learning
/// optimizers including LARS and LAMB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum PerLayerLR {
    /// All layers use the same learning rate (current behavior).
    #[default]
    Uniform,

    /// Learning rate scales inversely with depth.
    /// LR(layer) = base_lr / (1 + layer_idx * scale_factor)
    ///
    /// For scale_factor=0.1 and base_lr=0.1:
    /// - Layer 0: 0.1
    /// - Layer 5: 0.067
    /// - Layer 10: 0.05
    DepthScaling {
        /// Scale factor for depth-based reduction.
        /// Typical values: 0.05-0.2
        scale_factor: f32,
    },

    /// Learning rate decays exponentially with depth.
    /// LR(layer) = base_lr * decay^layer_idx
    ///
    /// For decay=0.9 and base_lr=0.1:
    /// - Layer 0: 0.1
    /// - Layer 5: 0.059
    /// - Layer 10: 0.035
    ExponentialDepth {
        /// Per-layer decay factor (typically 0.8-0.95).
        decay: f32,
    },

    /// Learning rate scales with square root of inverse depth.
    /// LR(layer) = base_lr / sqrt(1 + layer_idx * scale)
    ///
    /// Provides gentler decay than linear for deep networks.
    SqrtDepthScaling {
        /// Scale factor for sqrt-based reduction.
        scale: f32,
    },

    /// Linear warmup of LR from early layers to later layers.
    /// LR(layer) = base_lr * (start_factor + (1 - start_factor) * layer_idx / total_layers)
    ///
    /// Later layers get higher LR, useful when output layers need more updates.
    LinearWarmup {
        /// Initial factor for layer 0 (e.g., 0.5 means half the base LR).
        start_factor: f32,
    },

    /// Custom per-layer multipliers.
    /// LR(layer) = base_lr * factors\[layer_idx\] (or 1.0 if out of bounds)
    Custom {
        /// Learning rate multiplier for each layer index.
        factors: Vec<f32>,
    },
}

impl PerLayerLR {
    /// Compute the learning rate multiplier for a given layer.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer (0-indexed)
    /// * `total_layers` - Total number of layers in the network (used by some strategies)
    ///
    /// # Returns
    /// A factor in (0, inf) to multiply with the base learning rate.
    pub fn factor(&self, layer_idx: usize, total_layers: usize) -> f32 {
        match self {
            PerLayerLR::Uniform => 1.0,

            PerLayerLR::DepthScaling { scale_factor } => {
                1.0 / (1.0 + layer_idx as f32 * scale_factor)
            }

            PerLayerLR::ExponentialDepth { decay } => {
                decay.powi(layer_idx.min(i32::MAX as usize) as i32)
            }

            PerLayerLR::SqrtDepthScaling { scale } => 1.0 / (1.0 + layer_idx as f32 * scale).sqrt(),

            PerLayerLR::LinearWarmup { start_factor } => {
                if total_layers <= 1 {
                    1.0
                } else {
                    let progress = layer_idx as f32 / (total_layers - 1).max(1) as f32;
                    start_factor + (1.0 - start_factor) * progress
                }
            }

            PerLayerLR::Custom { factors } => *factors.get(layer_idx).unwrap_or(&1.0),
        }
    }
}

/// Configuration for the Lookahead optimizer wrapper.
///
/// Lookahead maintains two sets of weights: "fast weights" (θ) updated by the inner
/// optimizer (e.g., Adam), and "slow weights" (φ) that are a moving average of
/// the fast weights.
///
/// Algorithm:
/// 1. Run inner optimizer for k steps, updating θ
/// 2. After k steps: φ = φ + α * (θ - φ)  [interpolate slow toward fast]
/// 3. Reset: θ = φ  [sync fast weights back to slow]
///
/// This provides a stabilizing effect and can improve convergence, especially
/// when the inner optimizer produces noisy updates.
///
/// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LookaheadConfig {
    /// Whether lookahead is enabled.
    /// When false, no slow weights are maintained and the inner optimizer runs normally.
    pub enabled: bool,
    /// Synchronization period (k).
    /// Slow weights are updated every k inner optimizer steps.
    /// Typical values: 5-10
    pub sync_period: usize,
    /// Interpolation coefficient (α).
    /// How far slow weights move toward fast weights: φ = φ + α * (θ - φ)
    /// Typical values: 0.5-0.8
    pub alpha: f32,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for backward compatibility
            sync_period: 5, // Sync every 5 steps (paper default)
            alpha: 0.5,     // Move halfway toward fast weights (paper default)
        }
    }
}

impl LookaheadConfig {
    /// Create a new enabled lookahead config with the given parameters.
    pub fn new(sync_period: usize, alpha: f32) -> Self {
        let projected_alpha = alpha.clamp(0.0, 1.0);
        Self {
            enabled: true,
            sync_period: sync_period.max(1),
            alpha: if projected_alpha.is_nan() {
                0.5
            } else {
                projected_alpha
            },
        }
    }

    /// Check if slow weights should be synchronized at the given iteration.
    ///
    /// # Arguments
    /// * `iteration` - Current iteration (1-indexed)
    ///
    /// # Returns
    /// true if this iteration is a sync point (iteration % sync_period == 0)
    #[inline]
    pub fn should_sync(&self, iteration: usize) -> bool {
        self.enabled && iteration > 0 && iteration.is_multiple_of(self.sync_period)
    }
}

/// Configuration for Adam-style adaptive optimizer.
///
/// Adam (Adaptive Moment Estimation) maintains per-parameter first and second
/// moment estimates of gradients, providing automatic learning rate adaptation.
/// This is particularly effective for β-CROWN where gradient magnitudes vary
/// significantly across different neurons and optimization stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOptConfig {
    /// Base learning rate for β parameters.
    pub beta_lr: f32,
    /// Base learning rate for α parameters.
    pub alpha_lr: f32,
    /// Exponential decay rate for first moment estimate (β₁ in Adam paper).
    /// Typical value: 0.9
    pub beta1: f32,
    /// Exponential decay rate for second moment estimate (β₂ in Adam paper).
    /// Typical value: 0.999
    pub beta2: f32,
    /// Small constant for numerical stability (ε in Adam paper).
    /// Typical value: 1e-8
    pub epsilon: f32,
    /// Maximum gradient magnitude (gradient clipping). 0.0 = no clipping.
    ///
    /// Gradient clipping bounds the magnitude of gradient updates to prevent
    /// exploding gradients that can destabilize optimization. Both α and β
    /// gradients are clipped to the range `[-grad_clip, grad_clip]` before
    /// being used in the Adam update.
    ///
    /// **When to adjust:**
    /// - If optimization diverges (NaN values), try reducing grad_clip (e.g., 1.0)
    /// - If convergence is too slow, try increasing grad_clip or disabling (0.0)
    ///
    /// Default: 10.0 (moderate clipping)
    pub grad_clip: f32,
    /// Enable bias correction for moment estimates.
    ///
    /// Adam's moment estimates (m, v) are initialized to zero and biased toward zero
    /// during early iterations. Bias correction divides by `(1 - beta^t)` to correct
    /// this bias.
    ///
    /// Recommended: `true` for small iteration counts (β-CROWN typically uses 10-20).
    /// Default: true
    pub bias_correction: bool,
    /// Learning rate scheduler for controlling LR over iterations.
    pub scheduler: LRScheduler,
    /// Enable AMSGrad variant.
    /// AMSGrad maintains a maximum of past squared gradients (v_max) to prevent
    /// the effective learning rate from increasing when v decreases. This provides
    /// more stable convergence guarantees than standard Adam.
    /// Reference: Reddi et al., "On the Convergence of Adam and Beyond" (ICLR 2018)
    pub amsgrad: bool,
    /// Enable RAdam (Rectified Adam) variant.
    ///
    /// RAdam rectifies the variance of the adaptive learning rate during early
    /// iterations to avoid excessively large/unstable steps when the second-moment
    /// estimate is unreliable.
    ///
    /// When enabled:
    /// - For early iterations (ρ_t ≤ 4): uses an SGD-with-momentum style step (no variance term)
    /// - For later iterations (ρ_t > 4): uses a rectified Adam step with factor r_t
    ///
    /// **Performance Warning for β-CROWN:**
    ///
    /// RAdam is **NOT recommended** for β-CROWN optimization. Benchmark testing shows
    /// RAdam significantly underperforms compared to standard Adam:
    /// - Adam/AMSGrad: converge in ~15 domains
    /// - RAdam: fails to converge even after 100+ domains
    ///
    /// The cause is RAdam's warmup behavior. With `beta2=0.999`, rectification activates
    /// at t=5, meaning the first 4 iterations per domain use SGD-style updates without
    /// the adaptive variance term. Since β-CROWN typically uses only 10-20 iterations
    /// per domain (`beta_iterations`), RAdam spends 20-40% of each domain in warmup mode.
    /// For constraint optimization in β-CROWN, the full Adam adaptive update is more
    /// effective from iteration 1.
    ///
    /// If you must use RAdam, consider increasing `beta_iterations` to 50+ to allow
    /// the warmup to complete. However, even then, standard Adam is generally preferred.
    ///
    /// **Recommended alternatives:**
    /// - `Adam` (default): Best general performance
    /// - `AMSGrad`: Same performance as Adam with better convergence guarantees
    /// - `Lookahead + Adam`: Same performance with added stability
    ///
    /// Reference: Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond"
    /// (RAdam, 2019)
    pub radam: bool,
    /// Weight decay coefficient for AdamW (decoupled weight decay regularization).
    /// Unlike L2 regularization in standard Adam, AdamW applies weight decay directly
    /// to the parameters after the Adam update step, preventing interaction with
    /// the adaptive learning rate.
    ///
    /// **Performance Note for β-CROWN:**
    ///
    /// Weight decay is generally **not recommended** for β-CROWN constraint optimization.
    /// Benchmark testing shows AdamW (weight_decay=0.01) increases domains needed from
    /// 15 to 24 (~60% slower). The regularization adds unnecessary overhead for the
    /// Lagrangian constraint optimization in β-CROWN, where parameter magnitudes are
    /// naturally bounded by the problem structure.
    ///
    /// Typical values: 0.0 (disabled, recommended), 0.01-0.1 if regularization is needed.
    /// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019)
    pub weight_decay: f32,
    /// Per-layer learning rate strategy for β parameters.
    /// Allows different layers to use different learning rates based on their depth.
    /// Default: Uniform (all layers use the same base_lr).
    pub per_layer_lr_beta: PerLayerLR,
    /// Per-layer learning rate strategy for α parameters.
    /// Allows different layers to use different learning rates based on their depth.
    /// Default: Uniform (all layers use the same base_lr).
    pub per_layer_lr_alpha: PerLayerLR,
    /// Total number of layers in the network (used by some PerLayerLR strategies).
    /// This should be set when using LinearWarmup or similar strategies that need
    /// to know the total depth. If not set (0), defaults to layer_idx+1.
    pub total_layers: usize,
    /// Configuration for Lookahead optimizer wrapper.
    /// When enabled, maintains slow weights that stabilize optimization.
    /// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019)
    pub lookahead: LookaheadConfig,
    /// Learning rate for λ parameters (cutting plane Lagrangian multipliers).
    /// If None, uses beta_lr as default.
    /// GCP-CROWN typically uses a separate (often lower) learning rate for cuts.
    #[serde(default)]
    pub lr_lambda: Option<f32>,
}

impl Default for AdaptiveOptConfig {
    fn default() -> Self {
        Self {
            beta_lr: 0.05,  // α,β-CROWN default: 0.05
            alpha_lr: 0.01, // α,β-CROWN default: 0.01 (much lower than init!)
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            grad_clip: 10.0, // Clip gradients > 10 to prevent explosion
            bias_correction: true,
            scheduler: LRScheduler::ExponentialDecay { ny: 0.98 },
            amsgrad: false,    // Disabled by default for backward compatibility
            radam: false,      // Disabled by default for backward compatibility
            weight_decay: 0.0, // Disabled by default for backward compatibility
            per_layer_lr_beta: PerLayerLR::Uniform, // Default: same LR for all layers
            per_layer_lr_alpha: PerLayerLR::Uniform, // Default: same LR for all layers
            total_layers: 0,   // Will use layer_idx+1 as fallback
            lookahead: LookaheadConfig::default(), // Disabled by default
            lr_lambda: None,   // Uses beta_lr by default
        }
    }
}

impl AdaptiveOptConfig {
    /// Validate all numeric fields for soundness-critical constraints.
    ///
    /// Catches configuration errors that would silently corrupt optimization:
    /// - Negative learning rates invert gradient direction (unsound bounds)
    /// - NaN parameters disable optimization silently
    /// - Invalid Adam hyperparameters (beta1/beta2 outside [0,1], epsilon <= 0)
    ///
    /// Call after CLI flag application, YAML deserialization, or preset application.
    pub fn validate(&self) -> Result<()> {
        if !self.beta_lr.is_finite() || self.beta_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "beta_lr must be finite and >= 0, got {}",
                self.beta_lr
            )));
        }
        if !self.alpha_lr.is_finite() || self.alpha_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "alpha_lr must be finite and >= 0, got {}",
                self.alpha_lr
            )));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "epsilon must be finite and > 0, got {}",
                self.epsilon
            )));
        }
        if !self.beta1.is_finite() || !(0.0..=1.0).contains(&self.beta1) {
            return Err(NyError::InvalidConfig(format!(
                "beta1 must be in [0, 1], got {}",
                self.beta1
            )));
        }
        if !self.beta2.is_finite() || !(0.0..=1.0).contains(&self.beta2) {
            return Err(NyError::InvalidConfig(format!(
                "beta2 must be in [0, 1], got {}",
                self.beta2
            )));
        }
        if !self.grad_clip.is_finite() || self.grad_clip < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "grad_clip must be finite and >= 0, got {}",
                self.grad_clip
            )));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "weight_decay must be finite and >= 0, got {}",
                self.weight_decay
            )));
        }
        if let Some(lr) = self.lr_lambda {
            if !lr.is_finite() || lr < 0.0 {
                return Err(NyError::InvalidConfig(format!(
                    "lr_lambda must be finite and >= 0, got {lr}"
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn radam_rectification_factor(beta2: f32, t: f32) -> Option<f32> {
    if !(0.0..1.0).contains(&beta2) {
        return None;
    }
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
    if rho_inf <= 4.0 {
        return None;
    }

    let beta2_t = beta2.powf(t.max(1.0));
    let one_minus_beta2_t = 1.0 - beta2_t;
    if one_minus_beta2_t <= 0.0 {
        return None;
    }

    // ρ_t from the RAdam paper (t is 1-indexed).
    let rho_t = rho_inf - 2.0 * t * beta2_t / one_minus_beta2_t;
    if rho_t <= 4.0 {
        return None;
    }

    let numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
    let denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
    if numerator <= 0.0 || denominator <= 0.0 {
        return None;
    }

    Some((numerator / denominator).sqrt())
}
