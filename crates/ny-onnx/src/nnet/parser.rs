// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NNet text format parsing.
//!
//! Parses the NNet format header, metadata, normalization parameters,
//! and weight/bias matrices.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use tracing::{debug, info};

use super::NNetNetwork;

/// Parse NNet format from string content.
pub fn parse_nnet(content: &str) -> Result<NNetNetwork> {
    let mut lines = content
        .lines()
        .filter(|line| !line.starts_with("//") && !line.trim().is_empty());

    // Parse header: numLayers, inputSize, outputSize, maxLayerSize
    let header_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing header line".to_string()))?;
    let header: Vec<usize> = parse_csv_line(header_line)?;
    if header.len() < 4 {
        return Err(NyError::ModelLoad(format!(
            "Invalid header: expected 4 values, got {}",
            header.len()
        )));
    }
    let (num_layers, input_size, output_size, max_layer_size) =
        (header[0], header[1], header[2], header[3]);

    debug!(
        "NNet: {} layers, {} inputs, {} outputs",
        num_layers, input_size, output_size
    );

    // Parse layer sizes
    let sizes_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing layer sizes line".to_string()))?;
    let layer_sizes: Vec<usize> = parse_csv_line(sizes_line)?;
    if layer_sizes.len() != num_layers + 1 {
        return Err(NyError::ModelLoad(format!(
            "Expected {} layer sizes, got {}",
            num_layers + 1,
            layer_sizes.len()
        )));
    }

    // Parse symmetric flag (unused)
    let _symmetric_line = lines.next();

    // Parse input minimums
    let min_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing input minimums".to_string()))?;
    let input_minimums: Vec<f32> = parse_csv_line_f32(min_line)?;

    // Parse input maximums
    let max_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing input maximums".to_string()))?;
    let input_maximums: Vec<f32> = parse_csv_line_f32(max_line)?;

    // Parse means (inputs + output)
    let means_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing means".to_string()))?;
    let means: Vec<f32> = parse_csv_line_f32(means_line)?;
    let (input_means, output_mean) = if means.len() > input_size {
        (means[..input_size].to_vec(), means[input_size])
    } else {
        (means, 0.0)
    };

    // Parse ranges (inputs + output)
    let ranges_line = lines
        .next()
        .ok_or_else(|| NyError::ModelLoad("Missing ranges".to_string()))?;
    let ranges: Vec<f32> = parse_csv_line_f32(ranges_line)?;
    let (input_ranges, output_range) = if ranges.len() > input_size {
        (ranges[..input_size].to_vec(), ranges[input_size])
    } else {
        (ranges, 1.0)
    };

    // Parse weights and biases for each layer
    let mut weights = Vec::with_capacity(num_layers);
    let mut biases = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let prev_size = layer_sizes[layer_idx];
        let curr_size = layer_sizes[layer_idx + 1];

        let weight_count = curr_size
            .checked_mul(prev_size)
            .ok_or_else(|| NyError::ModelLoad(format!("Layer {layer_idx} weight size overflow")))?;
        debug!("Layer {layer_idx}: {prev_size} -> {curr_size} ({weight_count} weights, {curr_size} biases)");
        let mut weight_data = Vec::with_capacity(weight_count);
        for _row in 0..curr_size {
            let row_line = lines
                .next()
                .ok_or_else(|| NyError::ModelLoad("Missing weight row".to_string()))?;
            let row_values: Vec<f32> = parse_csv_line_f32(row_line)?;
            if row_values.len() < prev_size {
                return Err(NyError::ModelLoad(format!(
                    "Weight row has {} values, expected {}",
                    row_values.len(),
                    prev_size
                )));
            }
            weight_data.extend_from_slice(&row_values[..prev_size]);
        }
        let weight = Array2::from_shape_vec((curr_size, prev_size), weight_data)
            .map_err(|e| NyError::ModelLoad(format!("Failed to create weight matrix: {}", e)))?;
        weights.push(weight);

        // Read bias vector (curr_size values)
        let mut bias_data = Vec::with_capacity(curr_size);
        for _i in 0..curr_size {
            let bias_line = lines
                .next()
                .ok_or_else(|| NyError::ModelLoad("Missing bias value".to_string()))?;
            let bias_value: f32 = bias_line
                .trim()
                .trim_end_matches(',')
                .parse()
                .map_err(|e| NyError::ModelLoad(format!("Invalid bias value: {}", e)))?;
            bias_data.push(bias_value);
        }
        let bias = Array1::from_vec(bias_data);
        biases.push(bias);
    }

    let network = NNetNetwork {
        num_layers,
        input_size,
        output_size,
        max_layer_size,
        layer_sizes,
        input_minimums,
        input_maximums,
        input_means,
        input_ranges,
        output_mean,
        output_range,
        weights,
        biases,
    };

    info!(
        "Loaded NNet: {} layers, {} params",
        num_layers,
        network.param_count()
    );

    Ok(network)
}

pub(crate) fn parse_csv_line<T: std::str::FromStr>(line: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    line.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|e| NyError::ModelLoad(format!("Parse error: {}", e)))
        })
        .collect()
}

pub(crate) fn parse_csv_line_f32(line: &str) -> Result<Vec<f32>> {
    line.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let trimmed = s.trim();
            // Handle scientific notation
            trimmed
                .parse()
                .map_err(|e| NyError::ModelLoad(format!("Parse error '{}': {}", trimmed, e)))
        })
        .collect()
}
