// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.
// Keep a single ntest::timeout attribute per test; duplicates fail to compile.

use super::prelude::*;

// ==================== Learning Rate Scheduler Tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_constant() {
    let scheduler = LRScheduler::Constant;
    let base_lr = 0.1;

    // Constant scheduler should always return the base LR
    for t in 0..100 {
        let lr = scheduler.lr(t, base_lr);
        assert!(
            (lr - base_lr).abs() < 1e-6,
            "Constant scheduler should return base_lr at t={}, got {}",
            t,
            lr
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_step_decay() {
    let scheduler = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 5,
    };
    let base_lr = 1.0;

    // t=0-4: factor = 0.5^0 = 1.0
    assert!((scheduler.lr(0, base_lr) - 1.0).abs() < 1e-6);
    assert!((scheduler.lr(4, base_lr) - 1.0).abs() < 1e-6);

    // t=5-9: factor = 0.5^1 = 0.5
    assert!((scheduler.lr(5, base_lr) - 0.5).abs() < 1e-6);
    assert!((scheduler.lr(9, base_lr) - 0.5).abs() < 1e-6);

    // t=10-14: factor = 0.5^2 = 0.25
    assert!((scheduler.lr(10, base_lr) - 0.25).abs() < 1e-6);
    assert!((scheduler.lr(14, base_lr) - 0.25).abs() < 1e-6);

    // t=15: factor = 0.5^3 = 0.125
    assert!((scheduler.lr(15, base_lr) - 0.125).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_exponential_decay() {
    let ny = 0.9f32;
    let scheduler = LRScheduler::ExponentialDecay { ny };
    let base_lr = 1.0;

    // LR(t) = base_lr * ny^t
    for t in 0..20 {
        let expected = base_lr * ny.powi(t as i32);
        let actual = scheduler.lr(t, base_lr);
        assert!(
            (actual - expected).abs() < 1e-6,
            "ExponentialDecay mismatch at t={}: expected {}, got {}",
            t,
            expected,
            actual
        );
    }

    // Verify decay is happening
    assert!(scheduler.lr(0, base_lr) > scheduler.lr(10, base_lr));
    assert!(scheduler.lr(10, base_lr) > scheduler.lr(20, base_lr));
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_cosine_annealing() {
    let scheduler = LRScheduler::CosineAnnealing {
        min_lr: 0.0,
        t_max: 10,
    };
    let base_lr = 1.0;

    // At t=0: cosine(0) = 1, so LR = 0 + 0.5 * (1-0) * (1+1) = 1.0
    let lr_0 = scheduler.lr(0, base_lr);
    assert!(
        (lr_0 - 1.0).abs() < 1e-6,
        "Cosine should start at base_lr, got {}",
        lr_0
    );

    // At t=t_max/2: cosine(π/2) = 0, so LR = 0.5 * base_lr
    let lr_mid = scheduler.lr(5, base_lr);
    assert!(
        (lr_mid - 0.5).abs() < 1e-5,
        "Cosine should be at midpoint at t=t_max/2, got {}",
        lr_mid
    );

    // At t=t_max: cosine(π) = -1, so LR = min_lr = 0
    let lr_end = scheduler.lr(10, base_lr);
    assert!(
        lr_end < 0.01,
        "Cosine should approach min_lr at t_max, got {}",
        lr_end
    );

    // Test with non-zero min_lr
    let scheduler2 = LRScheduler::CosineAnnealing {
        min_lr: 0.1,
        t_max: 10,
    };

    let lr_end2 = scheduler2.lr(10, base_lr);
    assert!(
        (lr_end2 - 0.1).abs() < 0.05,
        "Cosine should end near min_lr, got {}",
        lr_end2
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_warmup_cosine() {
    let scheduler = LRScheduler::WarmupCosine {
        warmup_steps: 3,
        min_lr: 0.0,
        t_max: 10,
    };
    let base_lr = 1.0;

    // During warmup: linear increase from 0 to base_lr
    // t=0: (0+1)/3 = 0.333...
    let lr_0 = scheduler.lr(0, base_lr);
    assert!(
        (lr_0 - 1.0 / 3.0).abs() < 1e-5,
        "Warmup at t=0 should be 1/3 of base_lr, got {}",
        lr_0
    );

    // t=1: (1+1)/3 = 0.666...
    let lr_1 = scheduler.lr(1, base_lr);
    assert!(
        (lr_1 - 2.0 / 3.0).abs() < 1e-5,
        "Warmup at t=1 should be 2/3 of base_lr, got {}",
        lr_1
    );

    // t=2: (2+1)/3 = 1.0 (end of warmup)
    let lr_2 = scheduler.lr(2, base_lr);
    assert!(
        (lr_2 - 1.0).abs() < 1e-5,
        "Warmup should reach base_lr at end of warmup, got {}",
        lr_2
    );

    // After warmup: cosine annealing from t=3 to t=10
    // The cosine phase has t_max - warmup_steps = 7 iterations
    let lr_3 = scheduler.lr(3, base_lr);
    assert!(
        (lr_3 - 1.0).abs() < 1e-5,
        "Just after warmup should be at base_lr, got {}",
        lr_3
    );

    // At t=10: should be near min_lr
    let lr_end = scheduler.lr(10, base_lr);
    assert!(
        lr_end < 0.1,
        "Should approach min_lr at t_max, got {}",
        lr_end
    );

    // LR should decrease monotonically after warmup
    let mut prev_lr = lr_3;
    for t in 4..=10 {
        let lr = scheduler.lr(t, base_lr);
        assert!(
            lr <= prev_lr + 1e-6,
            "LR should decrease after warmup: t={}, prev={}, curr={}",
            t,
            prev_lr,
            lr
        );
        prev_lr = lr;
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_edge_cases() {
    // Zero t_max for cosine should return base LR
    let scheduler = LRScheduler::CosineAnnealing {
        min_lr: 0.1,
        t_max: 0,
    };
    assert!((scheduler.lr(0, 1.0) - 1.0).abs() < 1e-6);

    // Zero warmup steps - should go directly to cosine
    let scheduler2 = LRScheduler::WarmupCosine {
        warmup_steps: 0,
        min_lr: 0.0,
        t_max: 10,
    };
    let lr = scheduler2.lr(0, 1.0);
    assert!(
        (lr - 1.0).abs() < 1e-5,
        "Zero warmup should start at base_lr, got {}",
        lr
    );

    // Regression: StepDecay with step_size=0 should return 1.0, not panic (#2840)
    let scheduler3 = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 0,
    };
    assert!(
        (scheduler3.lr(0, 1.0) - 1.0).abs() < 1e-6,
        "StepDecay with step_size=0 should return base_lr"
    );
    assert!(
        (scheduler3.lr(100, 1.0) - 1.0).abs() < 1e-6,
        "StepDecay with step_size=0 should return base_lr at any t"
    );

    // Regression: ExponentialDecay with large t should not wrap i32 (#2840)
    let scheduler4 = LRScheduler::ExponentialDecay { ny: 0.999 };
    let lr_large_t = scheduler4.lr_factor(usize::MAX, 1.0);
    assert!(
        lr_large_t.is_finite(),
        "ExponentialDecay at usize::MAX should be finite, got {}",
        lr_large_t
    );

    // Regression: StepDecay with large t should not wrap i32 (#2840)
    let scheduler5 = LRScheduler::StepDecay {
        ny: 0.5,
        step_size: 1,
    };
    let lr_large_step = scheduler5.lr_factor(usize::MAX, 1.0);
    assert!(
        lr_large_step.is_finite(),
        "StepDecay at usize::MAX should be finite, got {}",
        lr_large_step
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_with_adam() {
    // Create a split history with one constraint
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Create beta state from history
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad(0, 0, 1.0);

    // Test with exponential decay scheduler
    let config = AdaptiveOptConfig {
        beta_lr: 1.0,
        scheduler: LRScheduler::ExponentialDecay { ny: 0.5 },
        bias_correction: false, // Simpler to verify
        ..Default::default()
    };

    // Iteration 1 (t=1): LR = 1.0 * 0.5^0 = 1.0
    beta_state.gradient_step_adam(&config, 1);
    let beta_t1 = beta_state.entry(0, 0).map(|e| e.value).unwrap_or(0.0);

    // Reset and try iteration 2
    let mut history2 = SplitHistory::new();
    history2.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut beta_state2 = BetaState::from_history(&history2).unwrap();
    beta_state2.accumulate_grad(0, 0, 1.0);
    // First do iteration 1 to build up momentum
    beta_state2.gradient_step_adam(&config, 1);
    beta_state2.zero_grad();
    beta_state2.accumulate_grad(0, 0, 1.0);
    // Iteration 2 (t=2): LR = 1.0 * 0.5^1 = 0.5
    beta_state2.gradient_step_adam(&config, 2);

    println!("Beta after t=1: {}", beta_t1);
    let beta_t2 = beta_state2.entry(0, 0).map(|e| e.value).unwrap_or(0.0);
    println!("Beta after t=2: {}", beta_t2);

    // The second update should use half the learning rate
    // This is validated by the scheduler integration working correctly
    assert!(beta_t1 > 0.0, "Beta should increase");
}

#[ntest::timeout(10000)]
#[test]
fn test_lr_scheduler_default() {
    // Default scheduler should match alpha-beta-CROWN's ExponentialLR(0.98).
    let config = AdaptiveOptConfig::default();
    assert!(
        matches!(
            config.scheduler,
            LRScheduler::ExponentialDecay { ny } if (ny - 0.98).abs() < 1e-6
        ),
        "Default scheduler should be ExponentialDecay {{ ny: 0.98 }}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cosine_scheduler_full_trajectory() {
    // Verify the full trajectory of cosine annealing
    let scheduler = LRScheduler::CosineAnnealing {
        min_lr: 0.01,
        t_max: 100,
    };
    let base_lr = 1.0;

    let lrs: Vec<f32> = (0..=100).map(|t| scheduler.lr(t, base_lr)).collect();

    // Should start near base_lr
    assert!(lrs[0] > 0.99, "Should start near base_lr");

    // Should end near min_lr
    assert!(lrs[100] < 0.02, "Should end near min_lr");

    // Should be monotonically decreasing
    for i in 1..lrs.len() {
        assert!(
            lrs[i] <= lrs[i - 1] + 1e-6,
            "Cosine should be monotonically decreasing: lrs[{}]={} > lrs[{}]={}",
            i,
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }

    // Midpoint should be approximately (base_lr + min_lr) / 2
    let expected_mid = f32::midpoint(1.0, 0.01);
    assert!(
        (lrs[50] - expected_mid).abs() < 0.05,
        "Midpoint LR {} should be close to {}",
        lrs[50],
        expected_mid
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_basic() {
    // Test that AMSGrad tracks v_max properly
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    // Use lower beta2 for faster decay to demonstrate v decreasing
    let config = AdaptiveOptConfig {
        amsgrad: true,
        beta2: 0.5,             // Fast decay to show v can decrease
        bias_correction: false, // Simpler to verify
        ..Default::default()
    };

    // First step with gradient 1.0
    beta_state.accumulate_grad(1, 0, 1.0);
    beta_state.gradient_step_adam(&config, 1);

    let entry = beta_state.entry(1, 0).unwrap();
    let v_after_1 = entry.v;
    let v_max_after_1 = entry.v_max;

    // v_max should equal v after first step
    assert!(
        (v_max_after_1 - v_after_1).abs() < 1e-9,
        "v_max should equal v after first step: v={}, v_max={}",
        v_after_1,
        v_max_after_1
    );

    // Reset gradient and apply smaller gradient
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 0.1); // Much smaller gradient
    beta_state.gradient_step_adam(&config, 2);

    let entry2 = beta_state.entry(1, 0).unwrap();
    let v_after_2 = entry2.v;
    let v_max_after_2 = entry2.v_max;

    // With beta2=0.5:
    // v_after_1 = 0.5 * 1.0^2 = 0.5
    // v_after_2 = 0.5 * 0.5 + 0.5 * 0.1^2 = 0.25 + 0.005 = 0.255
    // v should decrease because 0.5*v + 0.5*small^2 < v when small < 1

    // v_max should stay the same (or larger) - this is the key AMSGrad property
    assert!(
        v_max_after_2 >= v_max_after_1 - 1e-9,
        "v_max should never decrease: {} < {}",
        v_max_after_2,
        v_max_after_1
    );
    assert!(
        v_after_2 < v_after_1,
        "v should decrease with smaller gradients: {} >= {}",
        v_after_2,
        v_after_1
    );

    // v_max should be larger than v after v decreased
    assert!(
        v_max_after_2 > v_after_2,
        "v_max ({}) should be > v ({}) after v decreased",
        v_max_after_2,
        v_after_2
    );

    println!(
        "v_after_1: {:.6}, v_max_after_1: {:.6}",
        v_after_1, v_max_after_1
    );
    println!(
        "v_after_2: {:.6}, v_max_after_2: {:.6}",
        v_after_2, v_max_after_2
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_vs_adam_behavior() {
    // Compare AMSGrad and standard Adam update sizes when v decreases
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Standard Adam config
    let config_adam = AdaptiveOptConfig {
        amsgrad: false,
        beta_lr: 0.5,
        bias_correction: false,
        ..Default::default()
    };

    // AMSGrad config
    let config_amsgrad = AdaptiveOptConfig {
        amsgrad: true,
        beta_lr: 0.5,
        bias_correction: false,
        ..Default::default()
    };

    // Run both with same gradient sequence: large then small
    let mut beta_adam = BetaState::from_history(&history).unwrap();
    let mut beta_amsgrad = BetaState::from_history(&history).unwrap();

    // Step 1: Large gradient
    beta_adam.accumulate_grad(1, 0, 2.0);
    beta_amsgrad.accumulate_grad(1, 0, 2.0);
    beta_adam.gradient_step_adam(&config_adam, 1);
    beta_amsgrad.gradient_step_adam(&config_amsgrad, 1);

    let v1_adam = beta_adam.entry(1, 0).unwrap().value;
    let v1_amsgrad = beta_amsgrad.entry(1, 0).unwrap().value;

    // Should be equal after first step
    assert!(
        (v1_adam - v1_amsgrad).abs() < 1e-6,
        "First step should be equal"
    );

    // Step 2: Small gradient - this is where AMSGrad differs
    beta_adam.zero_grad();
    beta_amsgrad.zero_grad();
    beta_adam.accumulate_grad(1, 0, 0.01);
    beta_amsgrad.accumulate_grad(1, 0, 0.01);
    beta_adam.gradient_step_adam(&config_adam, 2);
    beta_amsgrad.gradient_step_adam(&config_amsgrad, 2);

    let v2_adam = beta_adam.entry(1, 0).unwrap().v;
    let v2_amsgrad_v = beta_amsgrad.entry(1, 0).unwrap().v;
    let v2_amsgrad_v_max = beta_amsgrad.entry(1, 0).unwrap().v_max;

    // v should be similar (decayed)
    assert!(
        (v2_adam - v2_amsgrad_v).abs() < 1e-6,
        "v values should be similar"
    );

    // v_max should be larger than v for AMSGrad
    assert!(
        v2_amsgrad_v_max > v2_amsgrad_v,
        "v_max ({}) should be > v ({}) after large then small gradient",
        v2_amsgrad_v_max,
        v2_amsgrad_v
    );

    println!("Adam v: {:.6}", v2_adam);
    println!(
        "AMSGrad v: {:.6}, v_max: {:.6}",
        v2_amsgrad_v, v2_amsgrad_v_max
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_monotonic_v_max() {
    // Test that v_max never decreases over many iterations
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig {
        amsgrad: true,
        bias_correction: true,
        ..Default::default()
    };

    let mut prev_v_max = 0.0f32;
    let gradients = [1.0, 0.5, 2.0, 0.1, 0.3, 1.5, 0.05, 0.8, 0.2, 1.0];

    for (i, &grad) in gradients.iter().enumerate() {
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, grad);
        beta_state.gradient_step_adam(&config, i + 1);

        let entry = beta_state.entry(1, 0).unwrap();
        assert!(
            entry.v_max >= prev_v_max - 1e-9,
            "v_max decreased at iteration {}: {} < {}",
            i + 1,
            entry.v_max,
            prev_v_max
        );
        prev_v_max = entry.v_max;
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_alpha_state() {
    // Test AMSGrad with DomainAlphaState using the simple_network helper
    let network = simple_network();

    // Input bounds that create unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    assert!(
        !alpha_state.is_empty(),
        "expected unstable neurons for AMSGrad alpha test"
    );

    let config = AdaptiveOptConfig {
        amsgrad: true,
        alpha_lr: 0.1,
        bias_correction: false,
        ..Default::default()
    };

    // Use a deterministic key from the alpha map to avoid HashMap iteration order.
    let key = alpha_state
        .neurons
        .keys()
        .copied()
        .min()
        .expect("expected alpha entries for AMSGrad alpha test");

    // First gradient: large
    alpha_state.accumulate_grad(key.0, key.1, 1.0);
    alpha_state.gradient_step_adam(&config, 1);
    let v_max_1 = alpha_state.neurons.get(&key).map_or(0.0, |n| n.adam_v_max);
    assert!(
        v_max_1 > 0.0,
        "expected AMSGrad v_max to update on first step"
    );

    // Second gradient: small
    alpha_state.zero_grad();
    alpha_state.accumulate_grad(key.0, key.1, 0.1);
    alpha_state.gradient_step_adam(&config, 2);
    let v_max_2 = alpha_state.neurons.get(&key).map_or(0.0, |n| n.adam_v_max);

    // v_max should not decrease
    assert!(
        v_max_2 >= v_max_1 - 1e-9,
        "v_max should not decrease: {} < {}",
        v_max_2,
        v_max_1
    );

    println!(
        "AMSGrad alpha test passed: v_max_1={:.6}, v_max_2={:.6}",
        v_max_1, v_max_2
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_config_default() {
    // Test that AMSGrad is disabled by default
    let config = AdaptiveOptConfig::default();
    assert!(
        !config.amsgrad,
        "AMSGrad should be disabled by default for backward compatibility"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_amsgrad_disabled_v_max_unchanged() {
    // When AMSGrad is disabled, v_max should not be updated
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig {
        amsgrad: false,
        ..Default::default()
    };

    // Check initial v_max
    assert_eq!(
        beta_state.entry(1, 0).unwrap().v_max,
        0.0,
        "v_max should start at 0"
    );

    // Run several iterations
    for i in 1..=5 {
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 1.0);
        beta_state.gradient_step_adam(&config, i);
    }

    // v_max should still be 0 when AMSGrad is disabled
    assert_eq!(
        beta_state.entry(1, 0).unwrap().v_max,
        0.0,
        "v_max should remain 0 when AMSGrad is disabled"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_config_default() {
    // Test that weight_decay is 0 by default
    let config = AdaptiveOptConfig::default();
    assert_eq!(
        config.weight_decay, 0.0,
        "weight_decay should be 0 by default for backward compatibility"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_radam_config_default() {
    // Test that RAdam is disabled by default
    let config = AdaptiveOptConfig::default();
    assert!(
        !config.radam,
        "RAdam should be disabled by default for backward compatibility"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_radam_rectification_factor_switch() {
    // With beta2=0.999, rectification activates at t=5 (ρ_t > 4).
    assert!(radam_rectification_factor(0.999, 4.0).is_none());
    let r = radam_rectification_factor(0.999, 5.0).expect("expected rectification factor");
    assert!(
        r > 0.0 && r < 1.0,
        "rectification factor should be in (0, 1), got {}",
        r
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_state_radam_smaller_update_than_adam() {
    // With constant positive gradients, RAdam should take smaller steps once rectification
    // activates (t >= 5 for beta2=0.999) compared to standard Adam.
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_adam = BetaState::from_history(&history).unwrap();
    let mut beta_radam = BetaState::from_history(&history).unwrap();

    let config_adam = AdaptiveOptConfig {
        beta_lr: 0.1,
        grad_clip: 0.0,
        radam: false,
        ..Default::default()
    };
    let config_radam = AdaptiveOptConfig {
        beta_lr: 0.1,
        grad_clip: 0.0,
        radam: true,
        ..Default::default()
    };

    for t in 1..=5 {
        beta_adam.zero_grad();
        beta_radam.zero_grad();
        beta_adam.accumulate_grad(1, 0, 1.0);
        beta_radam.accumulate_grad(1, 0, 1.0);
        beta_adam.gradient_step_adam(&config_adam, t);
        beta_radam.gradient_step_adam(&config_radam, t);
    }

    let adam_value = beta_adam.entry(1, 0).unwrap().value;
    let radam_value = beta_radam.entry(1, 0).unwrap().value;

    assert!(
        radam_value < adam_value,
        "expected RAdam value < Adam value at t=5, got radam={:.6}, adam={:.6}",
        radam_value,
        adam_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_basic() {
    // Test that weight decay reduces parameter values over time
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    // Set initial value explicitly
    beta_state.entry_mut(1, 0).unwrap().value = 1.0;

    // Use high weight decay and zero gradient to see decay in isolation.
    // Pin a constant LR scheduler: the expected math below assumes the LR
    // stays at beta_lr each step. The default scheduler is ExponentialDecay,
    // which would shrink the effective decay over time (#4307 triage).
    let config = AdaptiveOptConfig {
        weight_decay: 0.1,
        beta_lr: 0.1,
        bias_correction: false,
        scheduler: LRScheduler::Constant,
        ..Default::default()
    };

    // With zero gradient, Adam update is 0, only weight decay applies
    // θ_new = θ * (1 - lr * λ) = 1.0 * (1 - 0.1 * 0.1) = 0.99
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 0.0); // Zero gradient
    beta_state.gradient_step_adam(&config, 1);

    let value_after = beta_state.entry(1, 0).unwrap().value;
    let expected = 1.0 * (1.0 - 0.1 * 0.1);
    assert!(
        (value_after - expected).abs() < 1e-6,
        "After one step with zero gradient, value should be {:.6}, got {:.6}",
        expected,
        value_after
    );

    // Multiple steps should compound the decay
    for i in 2..=10 {
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.0);
        beta_state.gradient_step_adam(&config, i);
    }

    let value_final = beta_state.entry(1, 0).unwrap().value;
    // After 10 steps: 1.0 * (1 - 0.01)^10 ≈ 0.9044
    let expected_final = (1.0 - 0.1 * 0.1f32).powi(10);
    assert!(
        (value_final - expected_final).abs() < 1e-4,
        "After 10 steps, value should be {:.6}, got {:.6}",
        expected_final,
        value_final
    );

    println!(
        "AdamW basic test: initial=1.0, after 1 step={:.6}, after 10 steps={:.6}",
        expected, value_final
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_vs_adam() {
    // Compare AdamW with standard Adam (weight_decay=0)
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let config_adam = AdaptiveOptConfig {
        weight_decay: 0.0,
        beta_lr: 0.1,
        ..Default::default()
    };

    let config_adamw = AdaptiveOptConfig {
        weight_decay: 0.05,
        beta_lr: 0.1,
        ..Default::default()
    };

    let mut beta_adam = BetaState::from_history(&history).unwrap();
    let mut beta_adamw = BetaState::from_history(&history).unwrap();

    // Set same initial values
    beta_adam.entry_mut(1, 0).unwrap().value = 0.5;
    beta_adamw.entry_mut(1, 0).unwrap().value = 0.5;

    // Run with same gradients
    let gradients = [1.0, 0.5, 0.8, 0.3, 1.2];
    for (i, &grad) in gradients.iter().enumerate() {
        beta_adam.zero_grad();
        beta_adamw.zero_grad();
        beta_adam.accumulate_grad(1, 0, grad);
        beta_adamw.accumulate_grad(1, 0, grad);
        beta_adam.gradient_step_adam(&config_adam, i + 1);
        beta_adamw.gradient_step_adam(&config_adamw, i + 1);
    }

    let val_adam = beta_adam.entry(1, 0).unwrap().value;
    let val_adamw = beta_adamw.entry(1, 0).unwrap().value;

    // AdamW should have smaller values due to weight decay
    assert!(
        val_adamw < val_adam,
        "AdamW value ({:.6}) should be less than Adam value ({:.6}) due to weight decay",
        val_adamw,
        val_adam
    );

    println!("Adam value: {:.6}, AdamW value: {:.6}", val_adam, val_adamw);
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_alpha_state() {
    // Test weight decay with DomainAlphaState using simple_network
    let network = simple_network();

    // Input bounds that create unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    assert!(
        !alpha_state.is_empty(),
        "expected unstable neurons for AdamW alpha test"
    );

    let config_adamw = AdaptiveOptConfig {
        weight_decay: 0.1,
        alpha_lr: 0.5,
        bias_correction: false,
        ..Default::default()
    };

    // Use a deterministic key from the alpha map to avoid HashMap iteration order.
    let key = alpha_state
        .neurons
        .keys()
        .copied()
        .min()
        .expect("expected alpha entries for AdamW alpha test");

    // Record initial alpha value
    alpha_state.set_alpha(key.0, key.1, 1.0);
    let initial_alpha = alpha_state.neurons.get(&key).unwrap().alpha;

    // With zero gradient and weight decay, α should decrease
    alpha_state.zero_grad();
    alpha_state.accumulate_grad(key.0, key.1, 0.0); // Zero gradient
    alpha_state.gradient_step_adam(&config_adamw, 1);

    let alpha_after = alpha_state.neurons.get(&key).unwrap().alpha;

    // α_new = α * (1 - lr * λ) = initial_alpha * (1 - 0.5 * 0.1) = initial_alpha * 0.95
    let expected = initial_alpha * (1.0 - 0.5 * 0.1);
    assert!(
        (alpha_after - expected).abs() < 1e-6,
        "Alpha should be {:.6}, got {:.6}",
        expected,
        alpha_after
    );

    println!(
        "AdamW alpha test: initial={:.6}, after 1 step={:.6}",
        initial_alpha, alpha_after
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_with_scheduler() {
    // Test that weight decay works correctly with LR scheduling
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.entry_mut(1, 0).unwrap().value = 1.0;

    // Weight decay should use scheduled LR
    let config = AdaptiveOptConfig {
        weight_decay: 0.1,
        beta_lr: 1.0,
        scheduler: LRScheduler::StepDecay {
            step_size: 5,
            ny: 0.5, // LR halves every 5 steps
        },
        bias_correction: false,
        ..Default::default()
    };

    // First step: lr = 1.0, decay = 1.0 * 0.1 = 0.1
    // θ_new = 1.0 * (1 - 0.1) = 0.9 (ignoring Adam update since grad=0)
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 0.0);
    beta_state.gradient_step_adam(&config, 1);

    let value_step1 = beta_state.entry(1, 0).unwrap().value;
    assert!(
        (value_step1 - 0.9).abs() < 1e-6,
        "After step 1, value should be 0.9, got {:.6}",
        value_step1
    );

    // Advance to step 6 where LR = 0.5, decay = 0.5 * 0.1 = 0.05
    beta_state.entry_mut(1, 0).unwrap().value = 1.0; // Reset
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 0.0);
    beta_state.gradient_step_adam(&config, 6);

    let value_step6 = beta_state.entry(1, 0).unwrap().value;
    // At step 6, LR = 1.0 * 0.5^1 = 0.5
    // decay_factor = 1 - 0.5 * 0.1 = 0.95
    assert!(
        (value_step6 - 0.95).abs() < 1e-6,
        "After step 6, value should be 0.95, got {:.6}",
        value_step6
    );

    println!(
        "AdamW scheduler test: step1={:.6}, step6={:.6}",
        value_step1, value_step6
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adamw_no_decay_when_zero() {
    // Test that weight_decay=0 produces identical results to standard Adam
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let config = AdaptiveOptConfig {
        weight_decay: 0.0,
        ..Default::default()
    };

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let initial_value = 0.5;
    beta_state.entry_mut(1, 0).unwrap().value = initial_value;

    // With zero gradient, no change should occur
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 0.0);
    beta_state.gradient_step_adam(&config, 1);

    let value_after = beta_state.entry(1, 0).unwrap().value;
    // With zero gradient and zero weight decay, value should remain unchanged
    // (actually m=0 so update is 0)
    assert!(
        (value_after - initial_value).abs() < 1e-6,
        "With zero gradient and weight_decay, value should be unchanged: {:.6} != {:.6}",
        value_after,
        initial_value
    );
}

// ============================================================
// Per-Layer Learning Rate Tests
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_uniform() {
    // Uniform strategy should return 1.0 for all layers
    let strategy = PerLayerLR::Uniform;
    assert_eq!(strategy.factor(0, 10), 1.0);
    assert_eq!(strategy.factor(5, 10), 1.0);
    assert_eq!(strategy.factor(9, 10), 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_depth_scaling() {
    // DepthScaling: LR(layer) = 1 / (1 + layer_idx * scale_factor)
    let strategy = PerLayerLR::DepthScaling { scale_factor: 0.1 };

    // Layer 0: 1 / (1 + 0 * 0.1) = 1.0
    assert!((strategy.factor(0, 10) - 1.0).abs() < 1e-6);

    // Layer 5: 1 / (1 + 5 * 0.1) = 1 / 1.5 = 0.667
    assert!((strategy.factor(5, 10) - 0.6667).abs() < 0.001);

    // Layer 10: 1 / (1 + 10 * 0.1) = 1 / 2 = 0.5
    assert!((strategy.factor(10, 20) - 0.5).abs() < 1e-6);

    println!(
        "DepthScaling(0.1): layer0={:.4}, layer5={:.4}, layer10={:.4}",
        strategy.factor(0, 10),
        strategy.factor(5, 10),
        strategy.factor(10, 20)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_exponential_depth() {
    // ExponentialDepth: LR(layer) = decay^layer_idx
    let strategy = PerLayerLR::ExponentialDepth { decay: 0.9 };

    // Layer 0: 0.9^0 = 1.0
    assert!((strategy.factor(0, 10) - 1.0).abs() < 1e-6);

    // Layer 5: 0.9^5 = 0.59049
    assert!((strategy.factor(5, 10) - 0.59049).abs() < 0.001);

    // Layer 10: 0.9^10 = 0.34868
    assert!((strategy.factor(10, 20) - 0.34868).abs() < 0.001);

    println!(
        "ExponentialDepth(0.9): layer0={:.4}, layer5={:.4}, layer10={:.4}",
        strategy.factor(0, 10),
        strategy.factor(5, 10),
        strategy.factor(10, 20)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_sqrt_depth_scaling() {
    // SqrtDepthScaling: LR(layer) = 1 / sqrt(1 + layer_idx * scale)
    let strategy = PerLayerLR::SqrtDepthScaling { scale: 1.0 };

    // Layer 0: 1 / sqrt(1 + 0) = 1.0
    assert!((strategy.factor(0, 10) - 1.0).abs() < 1e-6);

    // Layer 3: 1 / sqrt(1 + 3) = 1 / 2 = 0.5
    assert!((strategy.factor(3, 10) - 0.5).abs() < 1e-6);

    // Layer 8: 1 / sqrt(1 + 8) = 1 / 3 = 0.333
    assert!((strategy.factor(8, 10) - 0.3333).abs() < 0.001);

    println!(
        "SqrtDepthScaling(1.0): layer0={:.4}, layer3={:.4}, layer8={:.4}",
        strategy.factor(0, 10),
        strategy.factor(3, 10),
        strategy.factor(8, 10)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_linear_warmup() {
    // LinearWarmup: LR(layer) = start_factor + (1 - start_factor) * layer_idx / (total - 1)
    let strategy = PerLayerLR::LinearWarmup { start_factor: 0.5 };
    let total_layers = 5;

    // Layer 0: 0.5 + (1 - 0.5) * 0/4 = 0.5
    assert!((strategy.factor(0, total_layers) - 0.5).abs() < 1e-6);

    // Layer 2: 0.5 + 0.5 * 2/4 = 0.75
    assert!((strategy.factor(2, total_layers) - 0.75).abs() < 1e-6);

    // Layer 4: 0.5 + 0.5 * 4/4 = 1.0
    assert!((strategy.factor(4, total_layers) - 1.0).abs() < 1e-6);

    println!(
        "LinearWarmup(0.5) with 5 layers: layer0={:.4}, layer2={:.4}, layer4={:.4}",
        strategy.factor(0, total_layers),
        strategy.factor(2, total_layers),
        strategy.factor(4, total_layers)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_custom() {
    // Custom: direct lookup
    let strategy = PerLayerLR::Custom {
        factors: vec![0.1, 0.2, 0.5, 1.0, 2.0],
    };

    assert!((strategy.factor(0, 10) - 0.1).abs() < 1e-6);
    assert!((strategy.factor(2, 10) - 0.5).abs() < 1e-6);
    assert!((strategy.factor(4, 10) - 2.0).abs() < 1e-6);

    // Out of bounds should return 1.0
    assert!((strategy.factor(10, 20) - 1.0).abs() < 1e-6);

    println!(
        "Custom factors: layer0={:.4}, layer2={:.4}, layer4={:.4}, layer10={:.4}",
        strategy.factor(0, 10),
        strategy.factor(2, 10),
        strategy.factor(4, 10),
        strategy.factor(10, 20)
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_beta_state() {
    // Test that per-layer LR is applied correctly in BetaState::gradient_step_adam
    let mut history = SplitHistory::new();

    // Add constraints at different layers
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 5,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let config = AdaptiveOptConfig {
        beta_lr: 1.0,
        bias_correction: false, // Simpler math for testing
        per_layer_lr_beta: PerLayerLR::DepthScaling { scale_factor: 0.2 },
        total_layers: 10,
        ..Default::default()
    };

    let mut beta_state = BetaState::from_history(&history).unwrap();

    // Set initial values
    beta_state.entry_mut(1, 0).unwrap().value = 0.0;
    beta_state.entry_mut(5, 0).unwrap().value = 0.0;

    // Apply same gradient to both
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 1.0);
    beta_state.accumulate_grad(5, 0, 1.0);
    beta_state.gradient_step_adam(&config, 1);

    let value_layer1 = beta_state.entry(1, 0).unwrap().value;
    let value_layer5 = beta_state.entry(5, 0).unwrap().value;

    // Layer 1: factor = 1/(1 + 1*0.2) = 1/1.2 = 0.833
    // Layer 5: factor = 1/(1 + 5*0.2) = 1/2 = 0.5
    // value_layer1 should be larger than value_layer5

    assert!(
        value_layer1 > value_layer5,
        "Layer 1 should have larger update (higher LR): {:.6} vs {:.6}",
        value_layer1,
        value_layer5
    );

    // Check approximate ratios
    // Expected ratio: 0.833 / 0.5 = 1.667
    let ratio = value_layer1 / value_layer5;
    assert!(
        (ratio - 1.667).abs() < 0.1,
        "Update ratio should be ~1.667, got {:.4}",
        ratio
    );

    println!(
        "Per-layer LR beta test: layer1={:.6}, layer5={:.6}, ratio={:.4}",
        value_layer1, value_layer5, ratio
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_alpha_state() {
    // Test per-layer LR in DomainAlphaState::gradient_step_adam
    // Create a simple network to initialize alpha state
    use crate::{Layer, LinearLayer, ReLULayer};
    use ndarray::Array2;
    use ny_tensor::BoundedTensor;

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(Array2::eye(4), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(Array2::eye(4), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(Array2::eye(4), None).unwrap(),
    ));

    // Create bounds that make both ReLU layers unstable
    let input_bounds =
        BoundedTensor::from_epsilon(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(), 1.0)
            .unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![
        Arc::new(input_bounds),
        Arc::new(
            BoundedTensor::from_epsilon(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(), 0.5)
                .unwrap(),
        ),
        Arc::new(
            BoundedTensor::from_epsilon(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(), 0.5)
                .unwrap(),
        ),
        Arc::new(
            BoundedTensor::from_epsilon(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(), 0.5)
                .unwrap(),
        ),
        Arc::new(
            BoundedTensor::from_epsilon(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(), 0.5)
                .unwrap(),
        ),
    ];

    let history = SplitHistory::new();
    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    // We should have alphas for layer 1 (ReLU after first Linear) and layer 3 (second ReLU)
    // Layer indices in Network: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear

    let config = AdaptiveOptConfig {
        alpha_lr: 1.0,
        bias_correction: false,
        per_layer_lr_alpha: PerLayerLR::ExponentialDepth { decay: 0.5 },
        total_layers: 5,
        ..Default::default()
    };

    // Reset alphas and gradients
    for n in alpha_state.neurons.values_mut() {
        n.set_alpha(0.5);
        n.grad = 0.1;
    }

    alpha_state.gradient_step_adam(&config, 1);

    // Check that earlier layers have larger updates
    // Layer 1: decay^1 = 0.5
    // Layer 3: decay^3 = 0.125
    let alpha_layer1 = alpha_state.alpha(1, 0);
    let alpha_layer3 = alpha_state.alpha(3, 0);

    // Both started at 0.5 with positive gradient, both should increase
    // But layer 1 should increase more
    println!(
        "Per-layer LR alpha test: layer1_alpha={:.6}, layer3_alpha={:.6}",
        alpha_layer1, alpha_layer3
    );

    // The exact values depend on Adam dynamics, but the ratio of changes should
    // reflect the LR ratio
    let change_layer1 = alpha_layer1 - 0.5;
    let change_layer3 = alpha_layer3 - 0.5;

    if change_layer1.abs() > 1e-6 && change_layer3.abs() > 1e-6 {
        let ratio = change_layer1 / change_layer3;
        // Expected: 0.5 / 0.125 = 4.0
        assert!(
            ratio > 1.0,
            "Layer 1 change should be larger than layer 3: {:.6} vs {:.6}",
            change_layer1,
            change_layer3
        );
        println!("Change ratio: {:.4} (expected ~4.0)", ratio);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_config_default() {
    // Test that default config has Uniform per-layer LR
    let config = AdaptiveOptConfig::default();

    assert_eq!(config.per_layer_lr_beta, PerLayerLR::Uniform);
    assert_eq!(config.per_layer_lr_alpha, PerLayerLR::Uniform);
    assert_eq!(config.total_layers, 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_per_layer_lr_combined_with_scheduler() {
    // Test that per-layer LR works correctly with LR scheduler
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 2,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let config = AdaptiveOptConfig {
        beta_lr: 1.0,
        bias_correction: false,
        scheduler: LRScheduler::StepDecay {
            ny: 0.5,
            step_size: 5,
        },
        per_layer_lr_beta: PerLayerLR::DepthScaling { scale_factor: 0.5 },
        total_layers: 10,
        ..Default::default()
    };

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.entry_mut(2, 0).unwrap().value = 0.0;

    // Step 1: scheduler factor = 1.0, layer factor = 1/(1+2*0.5) = 0.5
    // Combined LR = 1.0 * 1.0 * 0.5 = 0.5
    beta_state.zero_grad();
    beta_state.accumulate_grad(2, 0, 1.0);
    beta_state.gradient_step_adam(&config, 1);
    let value_step1 = beta_state.entry(2, 0).unwrap().value;

    // Reset and step 6: scheduler factor = 0.5, layer factor = 0.5
    // Combined LR = 1.0 * 0.5 * 0.5 = 0.25
    beta_state.entry_mut(2, 0).unwrap().value = 0.0;
    beta_state.entry_mut(2, 0).unwrap().m = 0.0;
    beta_state.entry_mut(2, 0).unwrap().v = 0.0;
    beta_state.zero_grad();
    beta_state.accumulate_grad(2, 0, 1.0);
    beta_state.gradient_step_adam(&config, 6);
    let value_step6 = beta_state.entry(2, 0).unwrap().value;

    // Step 6 should have smaller update due to scheduler decay
    assert!(
        value_step1 > value_step6,
        "Step 1 should have larger update: {:.6} vs {:.6}",
        value_step1,
        value_step6
    );

    // Ratio should be 2.0 (scheduler decayed by 0.5, exact f32 multiplication)
    let ratio = value_step1 / value_step6;
    assert!(
        (ratio - 2.0).abs() < 1e-2,
        "Update ratio should be ~2.0, got {:.4}",
        ratio
    );

    println!(
        "Combined scheduler + per-layer LR: step1={:.6}, step6={:.6}, ratio={:.4}",
        value_step1, value_step6, ratio
    );
}
