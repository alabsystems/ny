// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for domain clipping (domain refinement with intermediate bounds).

use super::*;
use ndarray::array;

#[ntest::timeout(10000)]
#[test]
fn test_layer_statistics_update() {
    let mut stats = LayerStatistics::new("test", vec![3]);

    // Add some samples
    let samples = vec![
        array![1.0, 2.0, 3.0].into_dyn(),
        array![2.0, 3.0, 4.0].into_dyn(),
        array![3.0, 4.0, 5.0].into_dyn(),
    ];

    for sample in &samples {
        stats.update(sample).unwrap();
    }

    assert_eq!(stats.num_samples, 3);

    // Check mean (should be [2, 3, 4])
    let mean = stats.mean.as_slice().unwrap();
    assert!((mean[0] - 2.0).abs() < 1e-5);
    assert!((mean[1] - 3.0).abs() < 1e-5);
    assert!((mean[2] - 4.0).abs() < 1e-5);

    // Check min/max
    let min = stats.min_observed.as_slice().unwrap();
    let max = stats.max_observed.as_slice().unwrap();
    assert!((min[0] - 1.0).abs() < 1e-5);
    assert!((max[0] - 3.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_statistical_bounds() {
    let mut stats = LayerStatistics::new("test", vec![2]);

    // Add samples with known mean and std
    for i in 0..100 {
        let val = (i as f32 - 50.0) / 10.0; // Range [-5, 4.9]
        stats.update(&array![val, val * 2.0].into_dyn()).unwrap();
    }

    let (lower, upper) = stats.statistical_bounds(3.0);

    // Bounds should be approximately μ ± 3σ
    // For uniform distribution, σ ≈ range/√12 ≈ 2.87
    assert!(lower[[0]] < -5.0);
    assert!(upper[[0]] > 5.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_empirical_bounds() {
    let mut stats = LayerStatistics::new("test", vec![2]);

    stats.update(&array![0.0, 10.0].into_dyn()).unwrap();
    stats.update(&array![10.0, 20.0].into_dyn()).unwrap();

    let (lower, upper) = stats.empirical_bounds(0.1); // 10% margin

    // Range is [0, 10] with margin of 10 * 0.1 = 1
    let lower_slice = lower.as_slice().unwrap();
    let upper_slice = upper.as_slice().unwrap();
    assert!((lower_slice[0] - (-1.0)).abs() < 1e-5);
    assert!((upper_slice[0] - 11.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_clipper_clip_bounds() {
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Empirical { margin_factor: 0.1 },
        min_samples: 1,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 100.0,
    });

    // Observe some concrete values
    for _ in 0..10 {
        clipper
            .observe("layer1", &array![5.0, 5.0, 5.0].into_dyn())
            .unwrap();
    }

    // Create bounds that are wider than observed
    let wide_bounds = BoundedTensor::new(
        array![-100.0, -100.0, -100.0].into_dyn(),
        array![100.0, 100.0, 100.0].into_dyn(),
    )
    .unwrap();

    let (clipped, reduction) = clipper.clip_bounds("layer1", &wide_bounds).unwrap();

    // Bounds should be clipped to approximately [5-0.1*0, 5+0.1*0] = [5, 5]
    // since range is 0 (all same values), empirical bounds are [5, 5]
    assert!(clipped.max_width() < wide_bounds.max_width());
    assert!(reduction > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_excluded_layers() {
    let clipper = DomainClipper::new(DomainClipConfig {
        exclude_patterns: vec!["output".to_string(), "final".to_string()],
        ..Default::default()
    });

    assert!(clipper.is_excluded("output_layer"));
    assert!(clipper.is_excluded("model/final_norm"));
    assert!(!clipper.is_excluded("hidden_layer"));
}

#[ntest::timeout(10000)]
#[test]
fn test_insufficient_samples() {
    let mut clipper = DomainClipper::new(DomainClipConfig {
        min_samples: 100,
        ..Default::default()
    });

    // Only add 10 samples
    for _ in 0..10 {
        clipper.observe("layer1", &array![1.0].into_dyn()).unwrap();
    }

    // Should return None (insufficient samples)
    assert!(clipper.computed_clip_bounds("layer1").is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_max_tightening_factor() {
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Empirical { margin_factor: 0.0 },
        min_samples: 1,
        max_tightening_factor: 2.0, // Only allow 2x tightening
        enabled: true,
        exclude_patterns: vec![],
    });

    // Observe a narrow range
    clipper.observe("layer1", &array![0.0].into_dyn()).unwrap();

    // Try to clip very wide bounds
    let wide_bounds =
        BoundedTensor::new(array![-100.0].into_dyn(), array![100.0].into_dyn()).unwrap();

    let (clipped, _) = clipper.clip_bounds("layer1", &wide_bounds).unwrap();

    // Should be limited to 2x tightening (width 200 -> 100)
    assert!(clipped.max_width() >= 100.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_tightening_limiter_stays_within_original_bounds() {
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Empirical { margin_factor: 0.0 },
        min_samples: 1,
        max_tightening_factor: 100.0,
        enabled: true,
        exclude_patterns: vec![],
    });

    // One wide element (observed near the upper edge) and one degenerate element.
    clipper
        .observe("layer1", &array![99.0, 2.0].into_dyn())
        .unwrap();

    let bounds = BoundedTensor::new(
        array![-100.0, 2.0].into_dyn(),
        array![100.0, 2.0].into_dyn(),
    )
    .unwrap();

    let (clipped, reduction) = clipper.clip_bounds("layer1", &bounds).unwrap();

    // The limiter may only relax the clip back toward the original bounds:
    // every element must stay inside its original interval.
    for i in 0..2 {
        assert!(
            clipped.lower()[[i]] >= bounds.lower()[[i]],
            "element {}: limited lower {} loosened past original {}",
            i,
            clipped.lower()[[i]],
            bounds.lower()[[i]]
        );
        assert!(
            clipped.upper()[[i]] <= bounds.upper()[[i]],
            "element {}: limited upper {} loosened past original {}",
            i,
            clipped.upper()[[i]],
            bounds.upper()[[i]]
        );
    }

    // The observed (reachable) value 99.0 must remain within the result.
    assert!(
        clipped.lower()[[0]] <= 99.0 && 99.0 <= clipped.upper()[[0]],
        "observed value 99.0 excluded from [{}, {}]",
        clipped.lower()[[0]],
        clipped.upper()[[0]]
    );

    // The degenerate original interval must be preserved exactly.
    assert_eq!(clipped.lower()[[1]], 2.0);
    assert_eq!(clipped.upper()[[1]], 2.0);

    assert!(reduction >= 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_inverted_bounds_protection() {
    // Test that clipping doesn't invert bounds
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Statistical { k: 0.1 }, // Very tight
        min_samples: 1,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 1000.0,
    });

    // Observe values around 0
    for _ in 0..10 {
        clipper
            .observe("layer1", &array![0.0, 0.0].into_dyn())
            .unwrap();
    }

    // Bounds that are already very tight
    let tight_bounds = BoundedTensor::new(
        array![10.0, 10.0].into_dyn(), // Far from observed values!
        array![11.0, 11.0].into_dyn(),
    )
    .unwrap();

    let (clipped, _) = clipper.clip_bounds("layer1", &tight_bounds).unwrap();

    // Should preserve original bounds since clipping would invert them
    assert!(clipped.lower()[[0]] <= clipped.upper()[[0]]);
}

// ========== Mutation-killing tests ==========

/// Test that Welford's mean update uses subtraction (delta - not delta +)
/// Kills: domain_clip.rs:122:29: replace - with + in LayerStatistics::update
#[ntest::timeout(10000)]
#[test]
fn test_welford_mean_uses_subtraction() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add samples with known mean
    stats.update(&array![10.0].into_dyn()).unwrap();
    stats.update(&array![20.0].into_dyn()).unwrap();
    stats.update(&array![30.0].into_dyn()).unwrap();

    // Mean should be 20.0, not some other value
    let mean = stats.mean[[0]];
    assert!(
        (mean - 20.0).abs() < 1e-5,
        "Mean should be 20.0 but got {}",
        mean
    );
}

/// Test variance update only happens after first sample
/// Kills: domain_clip.rs:126:29: replace > with >= in LayerStatistics::update
#[ntest::timeout(10000)]
#[test]
fn test_variance_update_after_first_sample() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // After first sample, std should still be 0
    stats.update(&array![10.0].into_dyn()).unwrap();
    assert_eq!(stats.std[[0]], 0.0, "Std should be 0 after first sample");

    // After second sample, std should be non-zero for different values
    stats.update(&array![20.0].into_dyn()).unwrap();
    assert!(
        stats.std[[0]] > 0.0,
        "Std should be positive after two different samples"
    );
}

/// Test variance update arithmetic operations
/// Kills multiple mutations in line 128-131
#[ntest::timeout(10000)]
#[test]
fn test_welford_variance_formula() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add samples with known variance
    // Values: 0, 10, 20 have mean=10 and population std dev ~= 8.165
    stats.update(&array![0.0].into_dyn()).unwrap();
    stats.update(&array![10.0].into_dyn()).unwrap();
    stats.update(&array![20.0].into_dyn()).unwrap();

    let std = stats.std[[0]];
    // Sample std dev = sqrt((100+0+100)/2) = 10.0 (exact for f32)
    assert!(
        (std - 10.0).abs() < 1e-4,
        "Std should be approximately 10.0 but got {}",
        std
    );
}

/// Test statistical bounds margin calculation uses multiplication
/// Kills: domain_clip.rs:140:32: replace * with + in LayerStatistics::statistical_bounds
#[ntest::timeout(10000)]
#[test]
fn test_statistical_bounds_multiplication() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Create stats with known std
    stats.update(&array![0.0].into_dyn()).unwrap();
    stats.update(&array![10.0].into_dyn()).unwrap();

    // Get bounds with clip_factor = 2.0
    let (lower, upper) = stats.statistical_bounds(2.0);

    // Mean is 5.0, std should be ~7.07 (sample std for 0,10)
    // With multiplication: margin = std * 2.0 = ~14.14
    // With addition: margin = std + 2.0 = ~9.07
    // Bounds should be significantly different

    let width = upper[[0]] - lower[[0]];
    // Width should be 2 * margin = 2 * std * 2.0 = 4 * std
    // std ~= 7.07, so width ~= 28.28
    assert!(
        width > 20.0,
        "Width should be > 20 (using multiplication) but got {}",
        width
    );
}

/// Test empirical bounds range calculation uses subtraction
/// Kills: domain_clip.rs:148:40: replace - with + in LayerStatistics::empirical_bounds
#[ntest::timeout(10000)]
#[test]
fn test_empirical_bounds_subtraction() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    stats.update(&array![10.0].into_dyn()).unwrap();
    stats.update(&array![20.0].into_dyn()).unwrap();

    let (lower, upper) = stats.empirical_bounds(0.1);

    // Range is max - min = 20 - 10 = 10
    // Margin = range * 0.1 = 1.0
    // With subtraction: range = 10, margin = 1, lower = 10-1=9, upper = 20+1=21
    // With addition: range = 30, margin = 3, lower = 10-3=7, upper = 20+3=23

    let lower_val = lower[[0]];
    let upper_val = upper[[0]];

    // With subtraction (correct): lower should be 9.0
    assert!(
        (lower_val - 9.0).abs() < 1e-5,
        "Lower should be 9.0 but got {}",
        lower_val
    );
    assert!(
        (upper_val - 21.0).abs() < 1e-5,
        "Upper should be 21.0 but got {}",
        upper_val
    );
}

/// Test conservative config differs from default
/// Kills: domain_clip.rs:268:9: replace DomainClipConfig::conservative -> Self with Default::default()
#[ntest::timeout(10000)]
#[test]
fn test_conservative_config_differs_from_default() {
    let conservative = DomainClipConfig::conservative();
    let default = DomainClipConfig::default();

    // Conservative should have higher min_samples
    assert!(
        conservative.min_samples > default.min_samples,
        "Conservative should require more samples: {} vs {}",
        conservative.min_samples,
        default.min_samples
    );

    // Conservative should have lower max_tightening_factor
    assert!(
        conservative.max_tightening_factor < default.max_tightening_factor,
        "Conservative should have lower max tightening: {} vs {}",
        conservative.max_tightening_factor,
        default.max_tightening_factor
    );
}

/// Test aggressive config differs from default
/// Kills: domain_clip.rs:280:9: replace DomainClipConfig::aggressive -> Self with Default::default()
#[ntest::timeout(10000)]
#[test]
fn test_aggressive_config_differs_from_default() {
    let aggressive = DomainClipConfig::aggressive();
    let default = DomainClipConfig::default();

    // Aggressive should have higher max_tightening_factor
    assert!(
        aggressive.max_tightening_factor > default.max_tightening_factor,
        "Aggressive should have higher max tightening: {} vs {}",
        aggressive.max_tightening_factor,
        default.max_tightening_factor
    );
}

/// Test observe_batch actually processes samples
/// Kills: domain_clip.rs:342:9: replace DomainClipper::observe_batch -> Result<()> with Ok(())
#[ntest::timeout(10000)]
#[test]
fn test_observe_batch_processes_samples() {
    let mut clipper = DomainClipper::default();

    let samples = vec![
        array![1.0, 2.0].into_dyn(),
        array![3.0, 4.0].into_dyn(),
        array![5.0, 6.0].into_dyn(),
    ];

    clipper.observe_batch("layer1", &samples).unwrap();

    let stats = clipper.statistics("layer1").unwrap();
    assert_eq!(
        stats.num_samples, 3,
        "observe_batch should have processed 3 samples, got {}",
        stats.num_samples
    );
}

/// Test get_statistics returns actual statistics
/// Kills: domain_clip.rs:350:9: replace DomainClipper::get_statistics -> Option<&LayerStatistics> with None
#[ntest::timeout(10000)]
#[test]
fn test_get_statistics_returns_data() {
    let mut clipper = DomainClipper::default();

    clipper
        .observe("layer1", &array![1.0, 2.0].into_dyn())
        .unwrap();

    let stats = clipper.statistics("layer1");
    assert!(
        stats.is_some(),
        "get_statistics should return Some after observe"
    );
    assert_eq!(stats.unwrap().num_samples, 1);
}

/// Test min_samples boundary condition (< vs <=)
/// Kills: domain_clip.rs:357:30: replace < with <= in DomainClipper::get_clip_bounds
#[ntest::timeout(10000)]
#[test]
fn test_min_samples_boundary() {
    let mut clipper = DomainClipper::new(DomainClipConfig {
        min_samples: 5,
        ..Default::default()
    });

    // Add exactly 5 samples
    for _ in 0..5 {
        clipper.observe("layer1", &array![1.0].into_dyn()).unwrap();
    }

    // With < (correct): 5 < 5 is false, so clipping should work
    // With <= (wrong): 5 <= 5 is true, so clipping would be skipped
    let bounds = clipper.computed_clip_bounds("layer1");
    assert!(
        bounds.is_some(),
        "Should get clip bounds when num_samples == min_samples"
    );
}

/// Test variance update delta multiplication (not addition)
/// Kills: domain_clip.rs:128:42: replace * with + in LayerStatistics::update
#[ntest::timeout(10000)]
#[test]
fn test_variance_delta_multiplication() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // With very different values, variance should be large
    stats.update(&array![0.0].into_dyn()).unwrap();
    stats.update(&array![100.0].into_dyn()).unwrap();

    // With multiplication: variance_update = delta * delta2 (large)
    // With addition: variance_update = delta + delta2 (smaller)
    let std = stats.std[[0]];
    assert!(
        std > 30.0,
        "Std for [0, 100] should be large (>30) but got {}",
        std
    );
}

/// Test variance formula division operations
/// Kills: domain_clip.rs:131:49: replace / with % or *
#[ntest::timeout(10000)]
#[test]
fn test_variance_division_not_modulo() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add samples to test variance calculation
    stats.update(&array![1.0].into_dyn()).unwrap();
    stats.update(&array![2.0].into_dyn()).unwrap();
    stats.update(&array![3.0].into_dyn()).unwrap();
    stats.update(&array![4.0].into_dyn()).unwrap();
    stats.update(&array![5.0].into_dyn()).unwrap();

    // Mean should be 3.0
    assert!((stats.mean[[0]] - 3.0).abs() < 1e-5);

    // Sample variance for [1,2,3,4,5] = 2.5, so std = sqrt(2.5) ≈ 1.58
    let std = stats.std[[0]];
    assert!(
        std > 1.0 && std < 2.0,
        "Std should be ~1.58 but got {}",
        std
    );
}

/// Test old_var multiplication in variance update
/// Kills: domain_clip.rs:130:37 and 131:36: replace * with + in old_var calculations
#[ntest::timeout(10000)]
#[test]
fn test_variance_old_var_multiplication() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add many samples to test running variance
    for i in 0..20 {
        stats.update(&array![i as f32].into_dyn()).unwrap();
    }

    // For 0..19, mean=9.5, variance should be consistent
    // If * were replaced with +, the variance would explode
    let std = stats.std[[0]];
    assert!(
        std > 5.0 && std < 10.0,
        "Std for 0..19 should be ~5.92 but got {}",
        std
    );
}

/// Test variance subtraction operations
/// Kills: domain_clip.rs:131:42, 131:54, 131:86: replace - with + or /
#[ntest::timeout(10000)]
#[test]
fn test_variance_subtraction_operations() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add samples with known variance
    stats.update(&array![10.0].into_dyn()).unwrap();
    stats.update(&array![12.0].into_dyn()).unwrap();
    stats.update(&array![14.0].into_dyn()).unwrap();

    // Mean should be 12.0
    assert!((stats.mean[[0]] - 12.0).abs() < 1e-4);

    // Sample std for [10, 12, 14] = sqrt((4+0+4)/2) = 2.0 (exact for f32)
    let std = stats.std[[0]];
    assert!(
        (std - 2.0).abs() < 1e-4,
        "Std for [10,12,14] should be ~2.0 but got {}",
        std
    );
}

/// Test variance formula with (n-2)/(n-1) ratio
/// Kills: domain_clip.rs:131:81: replace / with % or *
#[ntest::timeout(10000)]
#[test]
fn test_variance_n_ratio_division() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Need at least 3 samples to get (n-2)/(n-1) = 1/2
    stats.update(&array![0.0].into_dyn()).unwrap();
    stats.update(&array![10.0].into_dyn()).unwrap();
    stats.update(&array![20.0].into_dyn()).unwrap();
    stats.update(&array![30.0].into_dyn()).unwrap();

    // If division were replaced with modulo or multiplication,
    // the variance calculation would be completely wrong
    let std = stats.std[[0]];
    // For [0,10,20,30]: mean=15, sample var = 166.67, std ≈ 12.91
    assert!(
        std > 10.0 && std < 16.0,
        "Std for [0,10,20,30] should be ~12.91 but got {}",
        std
    );
}

/// Test that variance update check uses > (not >=)
/// Kills: domain_clip.rs:126:29: replace > with >= in LayerStatistics::update
///
/// With >=: variance would be updated on first sample where n=1
/// This would cause (n-2)/(n-1) = -1/0 = -inf or NaN
#[ntest::timeout(10000)]
#[test]
fn test_variance_check_strictly_greater_than_one() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add exactly one sample
    stats.update(&array![42.0].into_dyn()).unwrap();

    // With > (correct): variance not updated, std stays at 0
    // With >= (wrong): variance updated with n=1, causing (n-2)/(n-1) = -1/0
    // This would result in NaN or infinity

    let std = stats.std[[0]];
    assert!(!std.is_nan(), "Std should not be NaN after one sample");
    assert!(
        !std.is_infinite(),
        "Std should not be infinite after one sample"
    );
    assert_eq!(
        std, 0.0,
        "Std should be exactly 0.0 after one sample, got {}",
        std
    );
}

/// Test variance formula subtraction vs division
/// Kills: domain_clip.rs:131:54: replace - with / in LayerStatistics::update
///
/// The formula has: (n - 1.0), if replaced with (n / 1.0) = n, the Bessel correction breaks
#[ntest::timeout(10000)]
#[test]
fn test_variance_bessel_correction_subtraction() {
    let mut stats = LayerStatistics::new("test", vec![1]);

    // Add exactly 2 samples: 0 and 10
    stats.update(&array![0.0].into_dyn()).unwrap();
    stats.update(&array![10.0].into_dyn()).unwrap();

    // With n=2:
    // Correct: (n-1) = 1, variance_update / 1 = variance_update
    // Wrong: (n/1) = 2, variance_update / 2 = half the variance

    // For [0, 10]: mean=5, sample variance = (25+25)/1 = 50, std = 7.07
    // With division bug: sample variance = (25+25)/2 = 25, std = 5.0

    let std = stats.std[[0]];
    assert!(
        std > 6.0,
        "Std for [0, 10] should be >6 (around 7.07), got {}. If ~5.0, the Bessel correction is broken",
        std
    );
}
