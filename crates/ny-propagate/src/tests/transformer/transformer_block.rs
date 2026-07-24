// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transformer block and multi-block bound tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_transformer_block_ibp_with_residual() {
    // Test a simplified transformer block with residual connection:
    // output = input + MLP(LayerNorm(input))
    // where MLP = Linear -> GELU -> Linear
    //
    // This tests Phase 4 of transformer verification: full block with residuals.

    let hidden = 4;
    let expansion = 2; // MLP expands to 2x hidden for speed

    // Create LayerNorm
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();

    // Create MLP: up projection -> GELU -> down projection
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.3
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.3
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    // Create input bounds: [batch=2, seq=3, hidden]
    let batch = 2;
    let seq = 3;
    let epsilon = 0.1; // Small perturbation

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }

    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    // Track bound widths through the block
    let mut width_log: Vec<(&str, f32, f32, f32)> = Vec::new();

    let compute_width_stats = |bounds: &BoundedTensor| -> (f32, f32, f32) {
        let widths: Vec<f32> = bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(l, u)| u - l)
            .collect();
        let min_w = widths.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_w = widths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg_w = widths.iter().sum::<f32>() / widths.len() as f32;
        (min_w, avg_w, max_w)
    };

    // Input
    let (min_w, avg_w, max_w) = compute_width_stats(&input_bounds);
    width_log.push(("Input", min_w, avg_w, max_w));

    // Step 1: LayerNorm (per position)
    // Flatten to [batch*seq, hidden], apply layernorm, reshape back
    let flat_shape = vec![batch * seq, hidden];
    let flat_input = input_bounds.reshape(&flat_shape).unwrap();

    // LayerNorm IBP
    let ln_out = ln.propagate_ibp(&flat_input).unwrap();
    let ln_out = ln_out.reshape(&[batch, seq, hidden]).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&ln_out);
    width_log.push(("LayerNorm", min_w, avg_w, max_w));

    // Step 2: MLP up projection
    let flat_ln_out = ln_out.reshape(&flat_shape).unwrap();
    let mlp_up = linear_up.propagate_ibp(&flat_ln_out).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&mlp_up);
    width_log.push(("MLP Up", min_w, avg_w, max_w));

    // Step 3: GELU
    let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&gelu_out);
    width_log.push(("GELU", min_w, avg_w, max_w));

    // Step 4: MLP down projection
    let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
    let mlp_down = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&mlp_down);
    width_log.push(("MLP Down", min_w, avg_w, max_w));

    // Step 5: Residual Add
    let add_layer = AddLayer;
    let output_bounds = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down)
        .unwrap();
    let (min_w, avg_w, max_w) = compute_width_stats(&output_bounds);
    width_log.push(("Residual Add", min_w, avg_w, max_w));

    // Print bound width progression
    println!(
        "
=== Transformer Block IBP Bound Width Progression ==="
    );
    println!("{:<15} {:>10} {:>10} {:>10}", "Layer", "Min", "Avg", "Max");
    println!("{}", "-".repeat(48));
    for (name, min_w, avg_w, max_w) in &width_log {
        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4}",
            name, min_w, avg_w, max_w
        );
    }

    // Verify soundness by sampling
    let mut violations = 0;
    for sample_idx in 0..50 {
        // Sample random input within bounds
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower()[[b, s, h]]
                        + (input_bounds.upper()[[b, s, h]] - input_bounds.lower()[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate the block manually: output = input + MLP(LayerNorm(input))
        // Flatten for per-position operations
        let x_flat = x
            .clone()
            .into_shape_with_order((batch * seq, hidden))
            .unwrap();

        // LayerNorm per position
        let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
        for pos in 0..(batch * seq) {
            let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
            let y_pos = ln.eval(&x_pos).unwrap();
            for h in 0..hidden {
                ln_y[[pos, h]] = y_pos[h];
            }
        }

        // MLP up
        let mlp_up_y = ln_y.dot(&linear_up.weight.t());

        // GELU
        let gelu_y = mlp_up_y.mapv(|v| {
            0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
        });

        // MLP down
        let mlp_down_y = gelu_y.dot(&linear_down.weight.t());

        // Reshape back
        let mlp_down_y = mlp_down_y
            .into_shape_with_order((batch, seq, hidden))
            .unwrap();

        // Residual add: output = input + mlp_down
        let output = &x + &mlp_down_y;

        // Check if output is within bounds
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = output[[b, s, h]];
                    let l = output_bounds.lower()[[b, s, h]];
                    let u = output_bounds.upper()[[b, s, h]];
                    if y < l - 1e-5 || y > u + 1e-5 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "
Soundness check: {} violations out of {} samples",
        violations,
        50 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "IBP bounds should be sound");

    // Check that bounds don't explode
    let final_avg_width = width_log.last().unwrap().2;
    assert!(
        final_avg_width < 10.0,
        "Final bound width {} should be reasonable (< 10)",
        final_avg_width
    );

    println!("Transformer block IBP test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_add_layer_batched_crown_backward() {
    // Test AddLayer::propagate_linear_batched_binary
    // Verifies that batched CROWN backward for residual connections works correctly.

    let shape = vec![2, 3, 4]; // batch=2, seq=3, hidden=4

    // Create identity bounds at output
    let output_bounds = BatchedLinearBounds::identity(&shape).unwrap();

    // Propagate backward through Add
    let add_layer = AddLayer;
    let (bounds_a, bounds_b) = add_layer
        .propagate_linear_batched_binary(&output_bounds)
        .unwrap();

    // Both branches should have the same coefficient matrices
    assert_eq!(bounds_a.lower_a.shape(), output_bounds.lower_a.shape());
    assert_eq!(bounds_b.lower_a.shape(), output_bounds.lower_a.shape());

    // Coefficient matrices should be identical (identity passes through)
    for (a, b) in bounds_a.lower_a.iter().zip(bounds_b.lower_a.iter()) {
        assert!((a - b).abs() < 1e-6, "Coefficients should match");
    }

    // Biases should be halved (to avoid double-counting)
    // Output has zero bias initially, so both should have zero bias
    let total_bias_a: f32 = bounds_a.lower_b.iter().sum();
    let total_bias_b: f32 = bounds_b.lower_b.iter().sum();
    assert!((total_bias_a).abs() < 1e-6, "Bias should be near zero");
    assert!((total_bias_b).abs() < 1e-6, "Bias should be near zero");

    // Test with non-zero bias
    let mut bounds_with_bias = output_bounds;
    bounds_with_bias.lower_b.fill(2.0);
    bounds_with_bias.upper_b.fill(2.0);

    let (bounds_a, bounds_b) = add_layer
        .propagate_linear_batched_binary(&bounds_with_bias)
        .unwrap();

    // Biases should be halved
    for v in bounds_a.lower_b.iter() {
        assert!((v - 1.0).abs() < 1e-6, "Lower bias should be halved: {}", v);
    }
    for v in bounds_b.lower_b.iter() {
        assert!((v - 1.0).abs() < 1e-6, "Lower bias should be halved: {}", v);
    }

    println!("AddLayer batched CROWN backward test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_transformer_block_bound_explosion_analysis() {
    // Phase 4: Identify bound explosion source
    // Run the block with varying epsilon and track which operation causes explosion.

    let hidden = 8;
    let expansion = 4;
    let batch = 1;
    let seq = 4;

    // Create layers
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();

    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.1 * phase.sin()
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        0.1 * phase.cos()
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    let add_layer = AddLayer;

    println!(
        "
=== Bound Explosion Analysis ==="
    );
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Epsilon", "Input", "LayerNorm", "MLPUp", "GELU", "MLPDown", "Output"
    );
    println!("{}", "-".repeat(82));

    let assert_width = |label: &str, width: f32| {
        assert!(width.is_finite(), "{} width is not finite", label);
        assert!(width >= 0.0, "{} width is negative: {}", label, width);
    };

    for epsilon in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5] {
        // Create input with given epsilon
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.5; // base in [0, 0.5]
                    lower[[b, s, h]] = base - epsilon;
                    upper[[b, s, h]] = base + epsilon;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower()
                .iter()
                .zip(bt.upper().iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        let input_width = avg_width(&input);
        assert_width("input", input_width);

        // LayerNorm
        let flat_input = input.reshape(&[batch * seq, hidden]).unwrap();
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();
        let ln_width = avg_width(&ln_out);
        assert_width("layernorm", ln_width);

        // MLP Up
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();
        let up_width = avg_width(&mlp_up);
        assert_width("mlp_up", up_width);

        // GELU
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
        let gelu_width = avg_width(&gelu_out);
        assert_width("gelu", gelu_width);

        // MLP Down
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let down_width = avg_width(&mlp_down);
        assert_width("mlp_down", down_width);

        // Reshape and Add
        let mlp_down = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
        let output = add_layer.propagate_ibp_binary(&input, &mlp_down).unwrap();
        let output_width = avg_width(&output);
        assert_width("output", output_width);

        println!(
            "{:<10.3} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            epsilon, input_width, ln_width, up_width, gelu_width, down_width, output_width
        );
    }

    // Key insight: With small weights (0.1), bound growth is manageable.
    // The test documents the growth pattern for analysis.

    println!(
        "
Analysis: LayerNorm slightly expands bounds, Linear layers grow proportionally"
    );
    println!("to weight magnitudes, GELU can amplify unstable regions.");
}

#[ntest::timeout(10000)]
#[test]
fn test_transformer_block_crown_with_residual() {
    // Test CROWN backward through residual connection:
    // output = input + MLP(LayerNorm(input))
    //
    // For CROWN backward through y = x + F(x):
    // - Start with identity bounds at output
    // - Split through Add: bounds go to both branches
    // - Identity branch: bounds_x = bounds
    // - F(x) branch: propagate bounds backward through F
    // - Final bounds: sum coefficients from both branches
    //
    // This tests Phase 4 task 4: implement block CROWN.

    let hidden = 4;
    let expansion = 2;
    let batch = 2;
    let seq = 2;
    let epsilon = 0.05; // Smaller epsilon to avoid bound explosion

    // Create layers
    use crate::layers::LayerNormCrownMode;
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.2
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.2
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    // Create input bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.5;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }

    let input_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // === IBP Forward Pass to collect intermediate bounds ===
    let flat_shape = vec![batch * seq, hidden];
    let flat_input = input_bounds.reshape(&flat_shape).unwrap();

    // LayerNorm
    let ln_out = ln.propagate_ibp(&flat_input).unwrap();

    // MLP Up
    let mlp_up_out = linear_up.propagate_ibp(&ln_out).unwrap();

    // GELU
    let gelu_out = gelu.propagate_ibp(&mlp_up_out).unwrap();

    // MLP Down
    let mlp_down_out = linear_down.propagate_ibp(&gelu_out).unwrap();
    let mlp_down_3d = mlp_down_out.reshape(&[batch, seq, hidden]).unwrap();

    // IBP final output
    let add_layer = AddLayer;
    let ibp_output = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down_3d)
        .unwrap();

    // === CROWN Backward Pass ===
    // Initialize identity bounds at output
    let output_shape = vec![batch, seq, hidden];
    let crown_bounds = BatchedLinearBounds::identity(&output_shape).unwrap();

    // Step 1: Split through Add -> (bounds_input, bounds_mlp)
    let (bounds_input_branch, bounds_mlp_branch) = add_layer
        .propagate_linear_batched_binary(&crown_bounds)
        .unwrap();

    // Step 2: Propagate bounds_mlp_branch backward through MLP + LayerNorm
    // Reshape to flat for operations
    let flat_mlp_bounds = BatchedLinearBounds::from_parts_unchecked(
        bounds_mlp_branch
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_branch
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        bounds_mlp_branch
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_branch
            .upper_b
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        vec![batch * seq, hidden],
        vec![batch * seq, hidden],
    );

    // MLP Down backward
    let after_down = linear_down
        .propagate_linear_batched(&flat_mlp_bounds)
        .unwrap();

    // GELU backward
    let after_gelu = gelu
        .propagate_linear_batched_with_bounds(&after_down, &mlp_up_out)
        .unwrap();

    // MLP Up backward
    let after_up = linear_up.propagate_linear_batched(&after_gelu).unwrap();

    // LayerNorm backward
    let after_ln = ln
        .propagate_linear_batched_with_bounds(&after_up, &flat_input)
        .unwrap();

    // Reshape back to 3D
    let mlp_branch_final = BatchedLinearBounds::from_parts_unchecked(
        after_ln
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        after_ln
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Step 3: Combine input branch and MLP branch
    // For y = x + F(x), the combined coefficients are: A_combined = A_input + A_mlp
    let combined_lower_a = &bounds_input_branch.lower_a + &mlp_branch_final.lower_a;
    let combined_upper_a = &bounds_input_branch.upper_a + &mlp_branch_final.upper_a;
    let combined_lower_b = &bounds_input_branch.lower_b + &mlp_branch_final.lower_b;
    let combined_upper_b = &bounds_input_branch.upper_b + &mlp_branch_final.upper_b;

    let combined_bounds = BatchedLinearBounds::from_parts_unchecked(
        combined_lower_a,
        combined_lower_b,
        combined_upper_a,
        combined_upper_b,
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Step 4: Concretize with input bounds
    let crown_output = combined_bounds.concretize(&input_bounds).unwrap();

    // === Compare IBP vs CROWN ===
    println!(
        "
=== CROWN vs IBP for Transformer Block with Residual ==="
    );

    let ibp_widths: Vec<f32> = ibp_output
        .lower()
        .iter()
        .zip(ibp_output.upper().iter())
        .map(|(l, u)| u - l)
        .collect();
    let crown_widths: Vec<f32> = crown_output
        .lower()
        .iter()
        .zip(crown_output.upper().iter())
        .map(|(l, u)| u - l)
        .collect();

    let ibp_avg_width: f32 = ibp_widths.iter().sum::<f32>() / ibp_widths.len() as f32;
    let crown_avg_width: f32 = crown_widths.iter().sum::<f32>() / crown_widths.len() as f32;

    println!("IBP average bound width:   {:.6}", ibp_avg_width);
    println!("CROWN average bound width: {:.6}", crown_avg_width);
    println!(
        "CROWN tightness ratio:     {:.2}x",
        ibp_avg_width / crown_avg_width.max(1e-10)
    );

    // Verify CROWN bounds are valid (lower <= upper)
    let mut valid_count = 0;
    for (l, u) in crown_output.lower().iter().zip(crown_output.upper().iter()) {
        if *l <= *u + 1e-5 {
            valid_count += 1;
        }
    }
    assert_eq!(
        valid_count,
        crown_output.len(),
        "All CROWN bounds should be valid"
    );

    // Verify CROWN is at least as tight as IBP (or close)
    // Note: Due to numerical issues, CROWN might be slightly looser in some cases
    let tightness_ratio = crown_avg_width / ibp_avg_width.max(1e-10);
    println!("Tightness check: CROWN/IBP = {:.4}", tightness_ratio);

    // CROWN should generally be tighter, but allow some slack for numerical issues
    assert!(
        tightness_ratio < 2.0,
        "CROWN bounds ({}) should not be much looser than IBP ({})",
        crown_avg_width,
        ibp_avg_width
    );

    // Verify soundness by sampling
    let mut violations = 0;
    for sample_idx in 0..50 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower()[[b, s, h]]
                        + (input_bounds.upper()[[b, s, h]] - input_bounds.lower()[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate the block
        let x_flat = x
            .clone()
            .into_shape_with_order((batch * seq, hidden))
            .unwrap();

        let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
        for pos in 0..(batch * seq) {
            let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
            let y_pos = ln.eval(&x_pos).unwrap();
            for h in 0..hidden {
                ln_y[[pos, h]] = y_pos[h];
            }
        }

        let mlp_up_y = ln_y.dot(&linear_up.weight.t());
        let gelu_y = mlp_up_y.mapv(|v| {
            0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
        });
        let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
        let mlp_down_y = mlp_down_y
            .into_shape_with_order((batch, seq, hidden))
            .unwrap();
        let output = &x + &mlp_down_y;

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = output[[b, s, h]];
                    let l = crown_output.lower()[[b, s, h]];
                    let u = crown_output.upper()[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "CROWN soundness check: {} violations out of {} samples",
        violations,
        50 * batch * seq * hidden
    );
    assert!(
        violations == 0,
        "CROWN bounds should be sound ({} violations)",
        violations
    );

    println!("Transformer block CROWN with residual test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_block_bound_growth() {
    // Phase 5: Test bound growth through sequential transformer blocks.
    // Each block: output = input + MLP(LayerNorm(input))
    //
    // Measure how bounds grow through 1, 2, 3 blocks with both IBP and CROWN.

    let hidden = 4;
    let expansion = 2;
    let batch = 1;
    let seq = 2;

    // Create shared MLP weights (same weights for each block for simplicity)
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.15 // Smaller weights to reduce explosion
    });

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.15
    });

    // Helper function to run one block IBP
    let run_block_ibp = |input: &BoundedTensor,
                         ln: &LayerNormLayer,
                         linear_up: &LinearLayer,
                         gelu: &GELULayer,
                         linear_down: &LinearLayer|
     -> BoundedTensor {
        let flat_shape = vec![batch * seq, hidden];
        let flat_input = input.reshape(&flat_shape).unwrap();

        // LayerNorm
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();

        // MLP Up
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();

        // GELU
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();

        // MLP Down
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let mlp_down_3d = mlp_down.reshape(&[batch, seq, hidden]).unwrap();

        // Residual Add
        let add_layer = AddLayer;
        add_layer.propagate_ibp_binary(input, &mlp_down_3d).unwrap()
    };

    println!(
        "
=== Phase 5: Multi-Block Bound Growth Analysis ==="
    );
    println!(
        "{:<8} {:>10} {:>14} {:>14} {:>14} {:>14}",
        "Epsilon", "Input", "Block 1", "Block 2", "Block 3", "Growth Rate"
    );
    println!("{}", "-".repeat(76));

    for epsilon in [0.001, 0.005, 0.01, 0.02, 0.05] {
        // Create layers for this run
        let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
        let linear_up = LinearLayer::new(weight_up.clone(), None).unwrap();
        let gelu = GELULayer::default();
        let linear_down = LinearLayer::new(weight_down.clone(), None).unwrap();

        // Create input bounds
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.3;
                    lower[[b, s, h]] = base - epsilon as f32;
                    upper[[b, s, h]] = base + epsilon as f32;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower()
                .iter()
                .zip(bt.upper().iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        let input_width = avg_width(&input);

        // Block 1
        let after_block1 = run_block_ibp(&input, &ln, &linear_up, &gelu, &linear_down);
        let width1 = avg_width(&after_block1);

        // Block 2
        let after_block2 = run_block_ibp(&after_block1, &ln, &linear_up, &gelu, &linear_down);
        let width2 = avg_width(&after_block2);

        // Block 3
        let after_block3 = run_block_ibp(&after_block2, &ln, &linear_up, &gelu, &linear_down);
        let width3 = avg_width(&after_block3);

        // Compute average per-block growth rate
        let growth_rate = (width3 / input_width).powf(1.0 / 3.0);

        println!(
            "{:<8.3} {:>10.4} {:>14.4} {:>14.4} {:>14.4} {:>14.2}x",
            epsilon, input_width, width1, width2, width3, growth_rate
        );
    }

    // Test that small epsilon keeps bounds manageable after 3 blocks
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    let linear_up = LinearLayer::new(weight_up, None).unwrap();
    let gelu = GELULayer::default();
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    let epsilon = 0.001f32;
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.3;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }
    let input = BoundedTensor::new(lower, upper).unwrap();

    let after_block1 = run_block_ibp(&input, &ln, &linear_up, &gelu, &linear_down);
    let after_block2 = run_block_ibp(&after_block1, &ln, &linear_up, &gelu, &linear_down);
    let after_block3 = run_block_ibp(&after_block2, &ln, &linear_up, &gelu, &linear_down);

    let avg_width = |bt: &BoundedTensor| -> f32 {
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(l, u)| u - l)
            .sum::<f32>()
            / bt.len() as f32
    };

    let final_width = avg_width(&after_block3);
    println!(
        "
With ε=0.001, final bound width after 3 blocks: {:.6}",
        final_width
    );

    // Verify soundness on final output
    let mut violations = 0;
    for sample_idx in 0..30 {
        // Sample input
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input.lower()[[b, s, h]]
                        + (input.upper()[[b, s, h]] - input.lower()[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate 3 blocks
        let eval_block = |x: &ArrayD<f32>| -> ArrayD<f32> {
            let x_flat = x
                .clone()
                .into_shape_with_order((batch * seq, hidden))
                .unwrap();

            // LayerNorm
            let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
            for pos in 0..(batch * seq) {
                let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
                let y_pos = ln.eval(&x_pos).unwrap();
                for h in 0..hidden {
                    ln_y[[pos, h]] = y_pos[h];
                }
            }

            // MLP
            let mlp_up_y = ln_y.dot(&linear_up.weight.t());
            let gelu_y = mlp_up_y.mapv(|v| {
                0.5 * v
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
            });
            let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
            let mlp_down_y = mlp_down_y
                .into_shape_with_order((batch, seq, hidden))
                .unwrap();

            // Residual
            x + &mlp_down_y
        };

        let y1 = eval_block(&x);
        let y2 = eval_block(&y1);
        let y3 = eval_block(&y2);

        // Check bounds
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = y3[[b, s, h]];
                    let l = after_block3.lower()[[b, s, h]];
                    let u = after_block3.upper()[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "Soundness check: {} violations out of {} samples",
        violations,
        30 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "Multi-block IBP bounds should be sound");

    // Verify bounds don't explode too much
    assert!(
        final_width < 1.0,
        "With ε=0.001, 3-block bounds should stay under 1.0, got {}",
        final_width
    );

    println!("Multi-block bound growth test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_block_crown_vs_ibp() {
    // Phase 5: Compare CROWN vs IBP for multi-block verification.
    // CROWN should provide tighter bounds than IBP across multiple blocks.

    let hidden = 4;
    let expansion = 2;
    let batch = 1;
    let seq = 2;
    let epsilon = 0.01f32;

    // Create layers
    use crate::layers::LayerNormCrownMode;
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * phase.sin() * 0.1
    });
    let linear_up = LinearLayer::new(weight_up, None).unwrap();

    let gelu = GELULayer::default();

    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * phase.cos() * 0.1
    });
    let linear_down = LinearLayer::new(weight_down, None).unwrap();

    let add_layer = AddLayer;

    // Create input bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 0.3;
                lower[[b, s, h]] = base - epsilon;
                upper[[b, s, h]] = base + epsilon;
            }
        }
    }
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let avg_width = |bt: &BoundedTensor| -> f32 {
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(l, u)| u - l)
            .sum::<f32>()
            / bt.len() as f32
    };

    // === IBP through 2 blocks ===
    let run_block_ibp = |input: &BoundedTensor| -> BoundedTensor {
        let flat_shape = vec![batch * seq, hidden];
        let flat_input = input.reshape(&flat_shape).unwrap();
        let ln_out = ln.propagate_ibp(&flat_input).unwrap();
        let mlp_up = linear_up.propagate_ibp(&ln_out).unwrap();
        let gelu_out = gelu.propagate_ibp(&mlp_up).unwrap();
        let mlp_down = linear_down.propagate_ibp(&gelu_out).unwrap();
        let mlp_down_3d = mlp_down.reshape(&[batch, seq, hidden]).unwrap();
        add_layer.propagate_ibp_binary(input, &mlp_down_3d).unwrap()
    };

    let ibp_block1 = run_block_ibp(&input_bounds);
    let ibp_block2 = run_block_ibp(&ibp_block1);

    // === CROWN through 2 blocks (using IBP intermediate bounds) ===
    // For CROWN through multiple blocks, we run IBP first to get intermediate bounds,
    // then run CROWN backward from the final output.

    // Collect all intermediate bounds via IBP
    let flat_input = input_bounds.reshape(&[batch * seq, hidden]).unwrap();

    // Block 1 intermediates
    let ln_out_1 = ln.propagate_ibp(&flat_input).unwrap();
    let mlp_up_out_1 = linear_up.propagate_ibp(&ln_out_1).unwrap();
    let gelu_out_1 = gelu.propagate_ibp(&mlp_up_out_1).unwrap();
    let mlp_down_out_1 = linear_down.propagate_ibp(&gelu_out_1).unwrap();
    let mlp_down_3d_1 = mlp_down_out_1.reshape(&[batch, seq, hidden]).unwrap();
    let block1_out = add_layer
        .propagate_ibp_binary(&input_bounds, &mlp_down_3d_1)
        .unwrap();

    // Block 2 intermediates
    let flat_block1 = block1_out.reshape(&[batch * seq, hidden]).unwrap();
    let ln_out_2 = ln.propagate_ibp(&flat_block1).unwrap();
    let mlp_up_out_2 = linear_up.propagate_ibp(&ln_out_2).unwrap();
    let gelu_out_2 = gelu.propagate_ibp(&mlp_up_out_2).unwrap();
    let _mlp_down_out_2 = linear_down.propagate_ibp(&gelu_out_2).unwrap();

    // CROWN backward from output of block 2
    let output_shape = vec![batch, seq, hidden];
    let crown_bounds = BatchedLinearBounds::identity(&output_shape).unwrap();

    // Split through Add of block 2
    let (bounds_input_2, bounds_mlp_2) = add_layer
        .propagate_linear_batched_binary(&crown_bounds)
        .unwrap();

    // Propagate MLP branch backward through block 2
    let flat_mlp_bounds_2 = BatchedLinearBounds::from_parts_unchecked(
        bounds_mlp_2
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_2
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        bounds_mlp_2
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_2
            .upper_b
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        vec![batch * seq, hidden],
        vec![batch * seq, hidden],
    );

    let after_down_2 = linear_down
        .propagate_linear_batched(&flat_mlp_bounds_2)
        .unwrap();
    let after_gelu_2 = gelu
        .propagate_linear_batched_with_bounds(&after_down_2, &mlp_up_out_2)
        .unwrap();
    let after_up_2 = linear_up.propagate_linear_batched(&after_gelu_2).unwrap();
    let after_ln_2 = ln
        .propagate_linear_batched_with_bounds(&after_up_2, &flat_block1)
        .unwrap();

    // Reshape and combine
    let mlp_branch_2 = BatchedLinearBounds::from_parts_unchecked(
        after_ln_2
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln_2
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        after_ln_2
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln_2
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Combined bounds at block 1 output
    let combined_at_block1_lower_a = &bounds_input_2.lower_a + &mlp_branch_2.lower_a;
    let combined_at_block1_upper_a = &bounds_input_2.upper_a + &mlp_branch_2.upper_a;
    let combined_at_block1_lower_b = &bounds_input_2.lower_b + &mlp_branch_2.lower_b;
    let combined_at_block1_upper_b = &bounds_input_2.upper_b + &mlp_branch_2.upper_b;

    let bounds_at_block1 = BatchedLinearBounds::from_parts_unchecked(
        combined_at_block1_lower_a,
        combined_at_block1_lower_b,
        combined_at_block1_upper_a,
        combined_at_block1_upper_b,
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Now propagate these bounds backward through block 1
    let (bounds_input_1, bounds_mlp_1) = add_layer
        .propagate_linear_batched_binary(&bounds_at_block1)
        .unwrap();

    let flat_mlp_bounds_1 = BatchedLinearBounds::from_parts_unchecked(
        bounds_mlp_1
            .lower_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_1
            .lower_b
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        bounds_mlp_1
            .upper_a
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, hidden, hidden]))
            .unwrap(),
        bounds_mlp_1
            .upper_b
            .into_shape_with_order(IxDyn(&[batch * seq, hidden]))
            .unwrap(),
        vec![batch * seq, hidden],
        vec![batch * seq, hidden],
    );

    let after_down_1 = linear_down
        .propagate_linear_batched(&flat_mlp_bounds_1)
        .unwrap();
    let after_gelu_1 = gelu
        .propagate_linear_batched_with_bounds(&after_down_1, &mlp_up_out_1)
        .unwrap();
    let after_up_1 = linear_up.propagate_linear_batched(&after_gelu_1).unwrap();
    let after_ln_1 = ln
        .propagate_linear_batched_with_bounds(&after_up_1, &flat_input)
        .unwrap();

    let mlp_branch_1 = BatchedLinearBounds::from_parts_unchecked(
        after_ln_1
            .lower_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln_1
            .lower_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        after_ln_1
            .upper_a
            .into_shape_with_order(IxDyn(&[batch, seq, hidden, hidden]))
            .unwrap(),
        after_ln_1
            .upper_b
            .into_shape_with_order(IxDyn(&[batch, seq, hidden]))
            .unwrap(),
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Final combined bounds at input
    let final_lower_a = &bounds_input_1.lower_a + &mlp_branch_1.lower_a;
    let final_upper_a = &bounds_input_1.upper_a + &mlp_branch_1.upper_a;
    let final_lower_b = &bounds_input_1.lower_b + &mlp_branch_1.lower_b;
    let final_upper_b = &bounds_input_1.upper_b + &mlp_branch_1.upper_b;

    let final_bounds = BatchedLinearBounds::from_parts_unchecked(
        final_lower_a,
        final_lower_b,
        final_upper_a,
        final_upper_b,
        vec![batch, seq, hidden],
        vec![batch, seq, hidden],
    );

    // Concretize CROWN bounds
    let crown_output = final_bounds.concretize(&input_bounds).unwrap();

    // Compare
    let ibp_width = avg_width(&ibp_block2);
    let crown_width = avg_width(&crown_output);

    println!(
        "
=== Phase 5: Multi-Block CROWN vs IBP ==="
    );
    println!("IBP bound width after 2 blocks:   {:.6}", ibp_width);
    println!("CROWN bound width after 2 blocks: {:.6}", crown_width);
    println!(
        "CROWN improvement: {:.2}x tighter",
        ibp_width / crown_width.max(1e-10)
    );

    // Verify CROWN bounds are sound
    let mut violations = 0;
    for sample_idx in 0..30 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[batch, seq, hidden]));
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((sample_idx * 10000 + b * 100 + s * 10 + h) as u32)
                        .wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x[[b, s, h]] = input_bounds.lower()[[b, s, h]]
                        + (input_bounds.upper()[[b, s, h]] - input_bounds.lower()[[b, s, h]]) * t;
                }
            }
        }

        // Evaluate 2 blocks
        let eval_block = |x: &ArrayD<f32>| -> ArrayD<f32> {
            let x_flat = x
                .clone()
                .into_shape_with_order((batch * seq, hidden))
                .unwrap();

            let mut ln_y = Array2::<f32>::zeros((batch * seq, hidden));
            for pos in 0..(batch * seq) {
                let x_pos: Array1<f32> = (0..hidden).map(|h| x_flat[[pos, h]]).collect();
                let y_pos = ln.eval(&x_pos).unwrap();
                for h in 0..hidden {
                    ln_y[[pos, h]] = y_pos[h];
                }
            }

            let mlp_up_y = ln_y.dot(&linear_up.weight.t());
            let gelu_y = mlp_up_y.mapv(|v| {
                0.5 * v
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
            });
            let mlp_down_y = gelu_y.dot(&linear_down.weight.t());
            let mlp_down_y = mlp_down_y
                .into_shape_with_order((batch, seq, hidden))
                .unwrap();

            x + &mlp_down_y
        };

        let y1 = eval_block(&x);
        let y2 = eval_block(&y1);

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let y = y2[[b, s, h]];
                    let l = crown_output.lower()[[b, s, h]];
                    let u = crown_output.upper()[[b, s, h]];
                    if y < l - 1e-4 || y > u + 1e-4 {
                        violations += 1;
                    }
                }
            }
        }
    }

    println!(
        "CROWN soundness: {} violations out of {} samples",
        violations,
        30 * batch * seq * hidden
    );
    assert_eq!(violations, 0, "Multi-block CROWN bounds should be sound");

    // CROWN uses linear relaxation which should produce tighter bounds than IBP.
    // Observed ratio is ~0.55x (1.81x tighter); assert strict improvement.
    assert!(
        crown_width <= ibp_width,
        "CROWN width {crown_width} should not exceed IBP width {ibp_width}"
    );

    println!("Multi-block CROWN vs IBP test passed!");
}
