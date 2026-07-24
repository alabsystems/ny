// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain Clipping: Tighten intermediate bounds using activation statistics.
//!
//! Domain clipping is a bound tightening technique that uses empirical
//! activation statistics to clip intermediate bounds to realistic ranges.
//! By observing what values activations actually take on concrete inputs,
//! we can tighten bounds that would otherwise explode through deep networks.
//! Soundness is CONDITIONAL on the statistics being representative; see the
//! Soundness Guarantee section below.
//!
//! ## Algorithm
//!
//! 1. **Statistics Collection**: Run concrete forward passes on representative inputs
//!    to collect per-layer activation statistics (mean μ, std σ, min, max).
//!
//! 2. **Bound Clipping**: During abstract propagation, clip each layer's output bounds:
//!    - `clip_lower = max(original_lower, observed_min - margin)`
//!    - `clip_upper = min(original_upper, observed_max + margin)`
//!
//!    The margin ensures soundness by extending beyond observed values.
//!
//! ## Soundness Guarantee
//!
//! Clipping is sound (never excludes reachable values) when:
//! - The margin is large enough to cover unobserved but reachable values
//! - The clipping range contains all values the network can actually produce
//!
//! We provide two margin strategies:
//! - **Statistical**: μ ± k*σ (k=6 gives 99.99% coverage for Gaussian distributions)
//! - **Empirical**: observed_min/max ± empirical_margin (based on sample extremes)
//!
//! The empirical strategy is more robust for non-Gaussian activations.
//!
//! ## References
//!
//! - arxiv:2512.11087 - Domain-specific bound tightening for neural network verification

use ndarray::{ArrayD, IxDyn, Zip};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, trace};

/// Statistics for a single layer's activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStatistics {
    /// Layer identifier (name or index).
    pub layer_id: String,
    /// Per-element mean across samples.
    pub mean: ArrayD<f32>,
    /// Per-element standard deviation across samples.
    pub std: ArrayD<f32>,
    /// Per-element minimum observed value.
    pub min_observed: ArrayD<f32>,
    /// Per-element maximum observed value.
    pub max_observed: ArrayD<f32>,
    /// Number of samples used to compute statistics.
    pub num_samples: usize,
    /// Shape of the activation tensor.
    pub shape: Vec<usize>,
}

impl LayerStatistics {
    /// Create new statistics initialized to empty state.
    pub fn new(layer_id: impl Into<String>, shape: Vec<usize>) -> Self {
        let dim = IxDyn(&shape);
        Self {
            layer_id: layer_id.into(),
            mean: ArrayD::zeros(dim.clone()),
            std: ArrayD::zeros(dim.clone()),
            min_observed: ArrayD::from_elem(dim.clone(), f32::INFINITY),
            max_observed: ArrayD::from_elem(dim, f32::NEG_INFINITY),
            num_samples: 0,
            shape,
        }
    }

    /// Update statistics with a new sample (Welford's online algorithm).
    pub fn update(&mut self, sample: &ArrayD<f32>) -> Result<()> {
        if sample.shape() != self.shape.as_slice() {
            return Err(NyError::shape_mismatch(
                self.shape.clone(),
                sample.shape().to_vec(),
            ));
        }

        self.num_samples += 1;
        let n = self.num_samples as f32;

        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = sample.len() >= PARALLEL_THRESHOLD;

        // Update min/max
        if use_parallel {
            Zip::from(&mut self.min_observed)
                .and(sample)
                .par_for_each(|min_val, &s| {
                    *min_val = min_val.min(s);
                });
            Zip::from(&mut self.max_observed)
                .and(sample)
                .par_for_each(|max_val, &s| {
                    *max_val = max_val.max(s);
                });
        } else {
            Zip::from(&mut self.min_observed)
                .and(sample)
                .for_each(|min_val, &s| {
                    *min_val = min_val.min(s);
                });
            Zip::from(&mut self.max_observed)
                .and(sample)
                .for_each(|max_val, &s| {
                    *max_val = max_val.max(s);
                });
        }

        // Welford's online mean/variance update
        let delta = sample - &self.mean;
        self.mean = &self.mean + &(&delta / n);
        let delta2 = sample - &self.mean;

        // For variance: M2 = M2 + delta * delta2
        // We store sqrt(M2/(n-1)) as std after sufficient samples
        if self.num_samples > 1 {
            // Update variance estimate (using Bessel's correction)
            let variance_update = &delta * &delta2;
            // Running variance: new_var = old_var * (n-2)/(n-1) + delta*delta2/(n-1)
            let old_var = &self.std * &self.std;
            let new_var = &old_var * ((n - 2.0) / (n - 1.0)) + &variance_update / (n - 1.0);
            self.std = new_var.mapv(|v| v.max(0.0).sqrt());
        }

        Ok(())
    }

    /// Get the clipping bounds using statistical margin (μ ± k*σ).
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: clip_factor >= 0.0 (non-negative multiplier)
    /// REQUIRES: num_samples > 0 (statistics have been collected)
    ///
    /// ENSURES: For Gaussian-distributed activations, coverage probability
    ///          is approximately erf(clip_factor / √2):
    ///          - clip_factor = 3.0 → 99.73% coverage
    ///          - clip_factor = 6.0 → 99.9999998% coverage
    ///
    /// ENSURES: lower = mean - clip_factor * std (per-element)
    /// ENSURES: upper = mean + clip_factor * std (per-element)
    ///
    /// WARNING: Assumes Gaussian distribution. For heavy-tailed activations
    ///          (e.g., ReLU outputs), use empirical_bounds() instead.
    pub fn statistical_bounds(&self, clip_factor: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let margin = &self.std * clip_factor;
        let lower = &self.mean - &margin;
        let upper = &self.mean + &margin;
        (lower, upper)
    }

    /// Get the clipping bounds using empirical margin (observed ± margin).
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: margin_factor >= 0.0 (non-negative margin fraction)
    /// REQUIRES: num_samples > 0 (min/max have been observed)
    ///
    /// ENSURES: lower = min_observed - margin_factor * (max_observed - min_observed)
    /// ENSURES: upper = max_observed + margin_factor * (max_observed - min_observed)
    /// ENSURES: lower <= min_observed (margin extends below observed minimum)
    /// ENSURES: upper >= max_observed (margin extends above observed maximum)
    ///
    /// INVARIANT: min_observed <= max_observed (maintained by update() which uses min/max)
    ///
    /// SOUNDNESS: This method is sound when margin_factor is large enough that
    ///            [lower, upper] contains all reachable activation values.
    ///            With insufficient margin, clipping may exclude reachable values
    ///            causing UNSOUND verification results.
    ///
    /// RECOMMENDATION: margin_factor >= 0.1 (10%) for safety margin.
    pub fn empirical_bounds(&self, margin_factor: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let range = &self.max_observed - &self.min_observed;
        let margin = &range * margin_factor;
        let lower = &self.min_observed - &margin;
        let upper = &self.max_observed + &margin;
        (lower, upper)
    }

    /// Get the tighter of statistical and empirical bounds.
    ///
    /// Uses the intersection of both bounds, which is sound if both are sound.
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: statistical_factor >= 0.0
    /// REQUIRES: empirical_factor >= 0.0
    /// REQUIRES: num_samples > 0
    ///
    /// ENSURES: result.lower() = max(statistical_lower, empirical_lower) (per-element)
    /// ENSURES: result.upper() = min(statistical_upper, empirical_upper) (per-element)
    /// ENSURES: result bounds are tighter than or equal to either method alone
    ///
    /// SOUNDNESS: The intersection is sound if BOTH statistical_bounds() and
    ///            empirical_bounds() are sound with their respective factors.
    ///            If either bound is unsound (excludes reachable values), the
    ///            combined result inherits that unsoundness.
    ///
    /// INVARIANT: If both methods are sound, the tighter combined bounds are
    ///            also sound (intersection of two overapproximations).
    pub fn combined_bounds(
        &self,
        statistical_factor: f32,
        empirical_factor: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let (stat_lower, stat_upper) = self.statistical_bounds(statistical_factor);
        let (emp_lower, emp_upper) = self.empirical_bounds(empirical_factor);

        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = stat_lower.len() >= PARALLEL_THRESHOLD;

        // Take the tighter (inner) bounds
        let (lower, upper) = if use_parallel {
            let mut lower_out = ArrayD::zeros(stat_lower.raw_dim());
            let mut upper_out = ArrayD::zeros(stat_upper.raw_dim());
            Zip::from(&mut lower_out)
                .and(&stat_lower)
                .and(&emp_lower)
                .par_for_each(|out, &s, &e| *out = s.max(e));
            Zip::from(&mut upper_out)
                .and(&stat_upper)
                .and(&emp_upper)
                .par_for_each(|out, &s, &e| *out = s.min(e));
            (lower_out, upper_out)
        } else {
            let lower = Zip::from(&stat_lower)
                .and(&emp_lower)
                .map_collect(|&s, &e| s.max(e));
            let upper = Zip::from(&stat_upper)
                .and(&emp_upper)
                .map_collect(|&s, &e| s.min(e));
            (lower, upper)
        };

        (lower, upper)
    }
}

/// Strategy for computing clipping margins.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClipStrategy {
    /// Use statistical bounds: μ ± k*σ
    /// Good for Gaussian-like distributions (most activations).
    Statistical {
        /// Number of standard deviations (default: 6.0 for 99.99% coverage)
        k: f32,
    },

    /// Use empirical bounds: observed_min/max ± margin*range
    /// More robust for heavy-tailed or bounded distributions.
    Empirical {
        /// Fraction of observed range to add as margin (default: 0.1 = 10%)
        margin_factor: f32,
    },

    /// Use the tighter of statistical and empirical bounds.
    /// Best overall tightness while maintaining soundness.
    Combined {
        /// Statistical factor (k in μ ± k*σ)
        statistical_k: f32,
        /// Empirical margin factor
        empirical_margin: f32,
    },
}

impl Default for ClipStrategy {
    fn default() -> Self {
        // Combined strategy provides best tightness
        Self::Combined {
            statistical_k: 6.0,
            empirical_margin: 0.1,
        }
    }
}

/// Configuration for domain clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainClipConfig {
    /// Clipping strategy to use.
    pub strategy: ClipStrategy,
    /// Minimum number of samples before clipping is applied.
    /// Too few samples may give unreliable statistics.
    pub min_samples: usize,
    /// Whether to apply clipping (can be disabled for soundness testing).
    pub enabled: bool,
    /// Layer name patterns to exclude from clipping (e.g., output layers).
    pub exclude_patterns: Vec<String>,
    /// Maximum tightening factor: if clipping would reduce width by more than
    /// this factor, limit the tightening. Prevents over-aggressive clipping.
    pub max_tightening_factor: f32,
}

impl Default for DomainClipConfig {
    fn default() -> Self {
        Self {
            strategy: ClipStrategy::default(),
            min_samples: 10,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 100.0, // Allow up to 100x tightening
        }
    }
}

impl DomainClipConfig {
    /// Create a conservative configuration with wide margins.
    pub fn conservative() -> Self {
        Self {
            strategy: ClipStrategy::Statistical { k: 10.0 },
            min_samples: 100,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 10.0,
        }
    }

    /// Create an aggressive configuration for tighter bounds.
    /// Use only when soundness has been verified via sampling.
    pub fn aggressive() -> Self {
        Self {
            strategy: ClipStrategy::Combined {
                statistical_k: 4.0,
                empirical_margin: 0.05,
            },
            min_samples: 10,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 1000.0,
        }
    }
}

/// Domain clipper that stores and applies activation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainClipper {
    /// Configuration for clipping behavior.
    pub config: DomainClipConfig,
    /// Per-layer statistics, keyed by layer identifier.
    pub statistics: HashMap<String, LayerStatistics>,
    /// Number of times clipping has been applied.
    pub clip_count: usize,
    /// Total amount of bound width reduction from clipping.
    pub total_width_reduction: f64,
}

impl DomainClipper {
    /// Create a new domain clipper with the given configuration.
    pub fn new(config: DomainClipConfig) -> Self {
        Self {
            config,
            statistics: HashMap::new(),
            clip_count: 0,
            total_width_reduction: 0.0,
        }
    }

    /// Create a new domain clipper with default configuration.
    pub fn default_config() -> Self {
        Self::new(DomainClipConfig::default())
    }

    /// Check if a layer should be excluded from clipping.
    fn is_excluded(&self, layer_id: &str) -> bool {
        self.config
            .exclude_patterns
            .iter()
            .any(|pattern| layer_id.contains(pattern))
    }

    /// Update statistics for a layer with a concrete sample.
    pub fn observe(&mut self, layer_id: &str, sample: &ArrayD<f32>) -> Result<()> {
        let stats = self
            .statistics
            .entry(layer_id.to_string())
            .or_insert_with(|| LayerStatistics::new(layer_id, sample.shape().to_vec()));

        stats.update(sample)
    }

    /// Observe multiple samples at once for a layer.
    pub fn observe_batch(&mut self, layer_id: &str, samples: &[ArrayD<f32>]) -> Result<()> {
        for sample in samples {
            self.observe(layer_id, sample)?;
        }
        Ok(())
    }

    /// Statistics for a layer, if available.
    pub fn statistics(&self, layer_id: &str) -> Option<&LayerStatistics> {
        self.statistics.get(layer_id)
    }

    /// The computed clipping bounds for a layer based on its statistics.
    pub fn computed_clip_bounds(&self, layer_id: &str) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
        let stats = self.statistics.get(layer_id)?;

        if stats.num_samples < self.config.min_samples {
            trace!(
                "Layer {} has insufficient samples ({} < {}), skipping clip",
                layer_id,
                stats.num_samples,
                self.config.min_samples
            );
            return None;
        }

        let bounds = match self.config.strategy {
            ClipStrategy::Statistical { k } => stats.statistical_bounds(k),
            ClipStrategy::Empirical { margin_factor } => stats.empirical_bounds(margin_factor),
            ClipStrategy::Combined {
                statistical_k,
                empirical_margin,
            } => stats.combined_bounds(statistical_k, empirical_margin),
        };

        Some(bounds)
    }

    /// Apply clipping to a bounded tensor for a specific layer.
    ///
    /// Returns the clipped tensor and the amount of width reduction achieved.
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: bounds.lower() <= bounds.upper (well-formed input bounds)
    /// REQUIRES: bounds.shape() matches statistics shape for this layer
    ///
    /// ENSURES: result.lower() >= bounds.lower (clipping only tightens, never loosens)
    /// ENSURES: result.upper() <= bounds.upper (clipping only tightens, never loosens)
    /// ENSURES: result.lower() <= result.upper (output bounds are well-formed)
    /// ENSURES: width_reduction >= 0.0 (non-negative reduction)
    ///
    /// SOUNDNESS GUARANTEE: Clipping is sound (output contains all reachable values)
    ///                      when the underlying statistics capture all reachable
    ///                      activation values within the configured margin.
    ///
    /// CONDITIONAL SOUNDNESS:
    /// - If observed samples are representative of the true activation distribution
    /// - If margin factors (statistical_k, empirical_margin) are sufficient
    /// - Then output bounds contain all reachable values
    ///
    /// INVARIANT: If clipping would invert bounds (lower > upper), the original
    ///            bounds are preserved at those positions (fail-safe behavior).
    ///
    /// INVARIANT: If tightening would exceed max_tightening_factor, the clip is
    ///            relaxed back toward the original bounds until the factor is
    ///            respected (prevents over-aggressive clipping). The relaxation
    ///            only ever widens the clipped intervals, elementwise, and never
    ///            past the original bounds.
    ///
    /// INVARIANT: If config.enabled is false, or layer is excluded, or statistics
    ///            are insufficient, returns (bounds.clone(), 0.0) unchanged.
    pub fn clip_bounds(
        &mut self,
        layer_id: &str,
        bounds: &BoundedTensor,
    ) -> Result<(BoundedTensor, f32)> {
        if !self.config.enabled {
            return Ok((bounds.clone(), 0.0));
        }

        if self.is_excluded(layer_id) {
            trace!("Layer {} is excluded from clipping", layer_id);
            return Ok((bounds.clone(), 0.0));
        }

        let Some((clip_lower, clip_upper)) = self.computed_clip_bounds(layer_id) else {
            return Ok((bounds.clone(), 0.0));
        };

        // Verify shapes match
        if clip_lower.shape() != bounds.shape() {
            return Err(NyError::shape_mismatch(
                clip_lower.shape().to_vec(),
                bounds.shape().to_vec(),
            ));
        }

        let original_width = bounds.max_width();

        // Apply clipping: intersection of original bounds and clip bounds
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = bounds.lower().len() >= PARALLEL_THRESHOLD;

        let (clipped_lower, clipped_upper) = if use_parallel {
            let mut lower_out = ArrayD::zeros(bounds.lower().raw_dim());
            let mut upper_out = ArrayD::zeros(bounds.upper().raw_dim());
            Zip::from(&mut lower_out)
                .and(bounds.lower())
                .and(&clip_lower)
                .par_for_each(|out, &orig, &clip| *out = orig.max(clip));
            Zip::from(&mut upper_out)
                .and(bounds.upper())
                .and(&clip_upper)
                .par_for_each(|out, &orig, &clip| *out = orig.min(clip));
            (lower_out, upper_out)
        } else {
            let clipped_lower = Zip::from(bounds.lower())
                .and(&clip_lower)
                .map_collect(|&orig, &clip| orig.max(clip));
            let clipped_upper = Zip::from(bounds.upper())
                .and(&clip_upper)
                .map_collect(|&orig, &clip| orig.min(clip));
            (clipped_lower, clipped_upper)
        };

        // Ensure bounds remain valid (lower <= upper)
        // If clipping inverts bounds, keep original (indicates our statistics are off)
        let (final_lower, final_upper) =
            Self::ensure_valid_bounds(bounds.lower(), bounds.upper(), clipped_lower, clipped_upper);

        let clipped = BoundedTensor::new(final_lower, final_upper)?;
        let clipped_width = clipped.max_width();

        // Check tightening factor limit
        let tightening_factor = if clipped_width > 1e-10 {
            original_width / clipped_width
        } else {
            self.config.max_tightening_factor + 1.0
        };

        if tightening_factor > self.config.max_tightening_factor {
            debug!(
                "Layer {} clipping would exceed max tightening factor ({:.1}x > {:.1}x), limiting",
                layer_id, tightening_factor, self.config.max_tightening_factor
            );
            // Limit the tightening by relaxing the clip back toward the original
            // bounds: each element widens from its clipped interval toward
            // [center - target_width/2, center + target_width/2], clamped to its
            // original interval. The limiter can only undo clipping — it never
            // tightens past the clipped bounds nor loosens past the originals —
            // so the widest element regains at least target_width and the
            // reported tightening factor stays within the configured cap.
            let target_width = original_width / self.config.max_tightening_factor;
            let half_width = target_width / 2.0;
            // `(l+u)/2` kept verbatim: `f32::midpoint` differs once |bound| > f32::MAX/2
            // (sum overflow), and this center feeds the emitted bounds bit-for-bit.
            #[allow(clippy::manual_midpoint)]
            let limited_lower = Zip::from(bounds.lower())
                .and(bounds.upper())
                .and(clipped.lower())
                .map_collect(|&orig_l, &orig_u, &clip_l| {
                    let center = (orig_l + orig_u) / 2.0;
                    orig_l.max(clip_l.min(center - half_width))
                });
            // Same overflow-behavior preservation as `limited_lower` above.
            #[allow(clippy::manual_midpoint)]
            let limited_upper = Zip::from(bounds.lower())
                .and(bounds.upper())
                .and(clipped.upper())
                .map_collect(|&orig_l, &orig_u, &clip_u| {
                    let center = (orig_l + orig_u) / 2.0;
                    orig_u.min(clip_u.max(center + half_width))
                });
            let limited = BoundedTensor::new(limited_lower, limited_upper)?;
            let width_reduction = original_width - limited.max_width();
            self.clip_count += 1;
            self.total_width_reduction += width_reduction as f64;
            return Ok((limited, width_reduction));
        }

        let width_reduction = original_width - clipped_width;
        if width_reduction > 0.0 {
            self.clip_count += 1;
            self.total_width_reduction += width_reduction as f64;
            debug!(
                "Layer {} clipped: width {:.4} -> {:.4} ({:.1}x tighter)",
                layer_id, original_width, clipped_width, tightening_factor
            );
        }

        Ok((clipped, width_reduction))
    }

    /// Ensure clipped bounds are valid (lower <= upper).
    /// If clipping inverts bounds at any position, keep original bounds there.
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: orig_lower <= orig_upper (original bounds are well-formed)
    /// REQUIRES: All arrays have the same shape
    ///
    /// ENSURES: final_lower <= final_upper (output bounds are well-formed)
    /// ENSURES: For all i where clipped_lower[i] <= clipped_upper[i]:
    ///          final_lower[i] == clipped_lower[i] AND final_upper[i] == clipped_upper[i]
    /// ENSURES: For all i where clipped_lower[i] > clipped_upper[i]:
    ///          final_lower[i] == orig_lower[i] AND final_upper[i] == orig_upper[i]
    ///
    /// ATOMICITY: Each position uses EITHER the clipped bounds (both lower and upper)
    ///            OR the original bounds (both lower and upper), never a mix.
    ///
    /// SOUNDNESS: This is a fail-safe that preserves original bounds when clipping
    ///            would produce invalid (inverted) bounds. Invalid clipped bounds
    ///            indicate the statistics don't match the actual value range.
    fn ensure_valid_bounds(
        orig_lower: &ArrayD<f32>,
        orig_upper: &ArrayD<f32>,
        clipped_lower: ArrayD<f32>,
        clipped_upper: ArrayD<f32>,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = clipped_lower.len() >= PARALLEL_THRESHOLD;

        let mut final_lower = ArrayD::zeros(clipped_lower.raw_dim());
        let mut final_upper = ArrayD::zeros(clipped_upper.raw_dim());

        if use_parallel {
            Zip::from(&mut final_lower)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_lower)
                .par_for_each(|out, &clip_l, &clip_u, &orig_l| {
                    *out = if clip_l <= clip_u { clip_l } else { orig_l };
                });
            Zip::from(&mut final_upper)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_upper)
                .par_for_each(|out, &clip_l, &clip_u, &orig_u| {
                    *out = if clip_l <= clip_u { clip_u } else { orig_u };
                });
        } else {
            Zip::from(&mut final_lower)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_lower)
                .for_each(|out, &clip_l, &clip_u, &orig_l| {
                    *out = if clip_l <= clip_u { clip_l } else { orig_l };
                });
            Zip::from(&mut final_upper)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_upper)
                .for_each(|out, &clip_l, &clip_u, &orig_u| {
                    *out = if clip_l <= clip_u { clip_u } else { orig_u };
                });
        }

        (final_lower, final_upper)
    }

    /// Get a summary of clipping statistics.
    pub fn summary(&self) -> ClipperSummary {
        let total_layers = self.statistics.len();
        let layers_with_sufficient_samples = self
            .statistics
            .values()
            .filter(|s| s.num_samples >= self.config.min_samples)
            .count();

        ClipperSummary {
            total_layers,
            layers_with_sufficient_samples,
            total_samples: self.statistics.values().map(|s| s.num_samples).sum(),
            clip_count: self.clip_count,
            total_width_reduction: self.total_width_reduction,
            config: self.config.clone(),
        }
    }

    /// Reset clipping counters (but keep statistics).
    pub fn reset_counters(&mut self) {
        self.clip_count = 0;
        self.total_width_reduction = 0.0;
    }

    /// Clear all statistics.
    pub fn clear(&mut self) {
        self.statistics.clear();
        self.clip_count = 0;
        self.total_width_reduction = 0.0;
    }

    /// Merge statistics from another clipper.
    pub fn merge(&mut self, other: &DomainClipper) -> Result<()> {
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;

        for (layer_id, other_stats) in &other.statistics {
            if let Some(self_stats) = self.statistics.get_mut(layer_id) {
                // Merge by weighted combination
                let total_n = (self_stats.num_samples + other_stats.num_samples).max(1) as f32;
                let self_weight = self_stats.num_samples as f32 / total_n;
                let other_weight = other_stats.num_samples as f32 / total_n;

                self_stats.mean =
                    &(&self_stats.mean * self_weight) + &(&other_stats.mean * other_weight);
                self_stats.std =
                    &(&self_stats.std * self_weight) + &(&other_stats.std * other_weight);

                let use_parallel = self_stats.min_observed.len() >= PARALLEL_THRESHOLD;
                if use_parallel {
                    let mut new_min = ArrayD::zeros(self_stats.min_observed.raw_dim());
                    let mut new_max = ArrayD::zeros(self_stats.max_observed.raw_dim());
                    Zip::from(&mut new_min)
                        .and(&self_stats.min_observed)
                        .and(&other_stats.min_observed)
                        .par_for_each(|out, &a, &b| *out = a.min(b));
                    Zip::from(&mut new_max)
                        .and(&self_stats.max_observed)
                        .and(&other_stats.max_observed)
                        .par_for_each(|out, &a, &b| *out = a.max(b));
                    self_stats.min_observed = new_min;
                    self_stats.max_observed = new_max;
                } else {
                    self_stats.min_observed = Zip::from(&self_stats.min_observed)
                        .and(&other_stats.min_observed)
                        .map_collect(|&a, &b| a.min(b));
                    self_stats.max_observed = Zip::from(&self_stats.max_observed)
                        .and(&other_stats.max_observed)
                        .map_collect(|&a, &b| a.max(b));
                }

                self_stats.num_samples += other_stats.num_samples;
            } else {
                self.statistics
                    .insert(layer_id.clone(), other_stats.clone());
            }
        }
        Ok(())
    }
}

impl Default for DomainClipper {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Summary of domain clipping statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipperSummary {
    /// Total number of layers with statistics.
    pub total_layers: usize,
    /// Layers with enough samples for clipping.
    pub layers_with_sufficient_samples: usize,
    /// Total samples collected across all layers.
    pub total_samples: usize,
    /// Number of times clipping was applied.
    pub clip_count: usize,
    /// Total bound width reduction from clipping.
    pub total_width_reduction: f64,
    /// Configuration used.
    pub config: DomainClipConfig,
}

#[cfg(test)]
mod tests;
