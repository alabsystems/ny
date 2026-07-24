// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NNetNetwork runtime methods: evaluation, normalization, parameter counting.

use ny_core::{NyError, Result};

use super::NNetNetwork;

impl NNetNetwork {
    /// Evaluate the network on an input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of size `input_size`
    /// * `normalize` - If true, apply input normalization and output denormalization
    ///
    /// # Returns
    ///
    /// Output vector of size `output_size`, or error if input length mismatches.
    pub fn evaluate(&self, input: &[f32], normalize: bool) -> Result<Vec<f32>> {
        if input.len() != self.input_size {
            return Err(NyError::shape_mismatch(
                vec![self.input_size],
                vec![input.len()],
            ));
        }

        let mut x: Vec<f32> = if normalize {
            // Normalize inputs
            input
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let clamped = v.clamp(self.input_minimums[i], self.input_maximums[i]);
                    (clamped - self.input_means[i]) / self.input_ranges[i]
                })
                .collect()
        } else {
            input.to_vec()
        };

        // Forward pass through all layers
        for (layer_idx, (weights, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            // Linear: y = Wx + b
            let mut y = vec![0.0f32; weights.nrows()];
            for (i, row) in weights.rows().into_iter().enumerate() {
                y[i] = row.iter().zip(&x).map(|(&w, &xi)| w * xi).sum::<f32>() + bias[i];
            }

            // ReLU for hidden layers (not output)
            if layer_idx < self.num_layers - 1 {
                for v in &mut y {
                    *v = v.max(0.0);
                }
            }

            x = y;
        }

        // Denormalize output if requested
        if normalize {
            for v in &mut x {
                *v = *v * self.output_range + self.output_mean;
            }
        }

        Ok(x)
    }

    /// Normalized input bounds (after applying input normalization).
    pub fn normalized_input_bounds(&self) -> (Vec<f32>, Vec<f32>) {
        let lower: Vec<f32> = self
            .input_minimums
            .iter()
            .zip(&self.input_means)
            .zip(&self.input_ranges)
            .map(|((&min, &mean), &range)| (min - mean) / range)
            .collect();

        let upper: Vec<f32> = self
            .input_maximums
            .iter()
            .zip(&self.input_means)
            .zip(&self.input_ranges)
            .map(|((&max, &mean), &range)| (max - mean) / range)
            .collect();

        (lower, upper)
    }

    /// Deprecated compatibility alias for [`normalized_input_bounds`](Self::normalized_input_bounds).
    #[deprecated(note = "use normalized_input_bounds")]
    pub fn get_normalized_input_bounds(&self) -> (Vec<f32>, Vec<f32>) {
        self.normalized_input_bounds()
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }
}
