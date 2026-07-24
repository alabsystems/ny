// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Weight extraction utilities for decoder layers.

use ny_core::{NyError, Result};
use tracing::warn;

use super::DecoderModel;

impl DecoderModel {
    /// Extract LayerNorm weights for decoder layers.
    pub(super) fn get_decoder_layer_norm_weights(
        &self,
        prefix: &str,
    ) -> Result<(ndarray::Array1<f32>, ndarray::Array1<f32>, f32)> {
        // Look for ny/beta in ONNX weight names
        // Common patterns: "{prefix}/ny", "{prefix}/weight", "{prefix}.weight"

        let ny = self.find_decoder_weight_1d(prefix, &["ny", "weight"])?;
        let beta = self.find_decoder_weight_1d(prefix, &["beta", "bias"])?;

        // Default epsilon for LayerNorm
        let eps = 1e-5;

        Ok((ny, beta, eps))
    }

    /// Extract Linear weights for decoder layers.
    pub(super) fn get_decoder_linear_weights(
        &self,
        prefix: &str,
    ) -> Result<(ndarray::Array2<f32>, Option<ndarray::Array1<f32>>)> {
        // Look for weight matrix
        let weight = self.find_decoder_weight_2d(prefix, &["MatMul"])?;

        // Look for bias (optional — not all layers have bias)
        let bias = self
            .find_decoder_weight_1d(prefix, &["Add", "bias"])
            .map_err(|e| {
                warn!("decoder bias lookup ({prefix}): {e}");
                e
            })
            .ok();

        Ok((weight, bias))
    }

    /// Helper to find 1D weights by pattern matching.
    pub(super) fn find_decoder_weight_1d(
        &self,
        prefix: &str,
        suffixes: &[&str],
    ) -> Result<ndarray::Array1<f32>> {
        for suffix in suffixes {
            // Try direct weight name patterns
            let patterns = [
                format!("{}/{}", prefix, suffix),
                format!(
                    "{}.{}",
                    prefix.replace("/", ".").trim_start_matches('.'),
                    suffix
                ),
            ];

            for pattern in &patterns {
                if let Some(w) = self.model.weights.get(pattern) {
                    return w
                        .clone()
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|_| {
                            NyError::InvalidSpec(format!("Weight {} must be 1D", pattern))
                        });
                }
            }
        }

        // Try finding by substring match
        let search_key = prefix.replace("/", ".");
        for (key, value) in self.model.weights.iter() {
            if key.contains(&search_key) {
                for suffix in suffixes {
                    if key.contains(suffix) {
                        return value
                            .clone()
                            .into_dimensionality::<ndarray::Ix1>()
                            .map_err(|_| {
                                NyError::InvalidSpec(format!("Weight {} must be 1D", key))
                            });
                    }
                }
            }
        }

        Err(NyError::InvalidSpec(format!(
            "Could not find 1D weight for {} with suffixes {:?}",
            prefix, suffixes
        )))
    }

    /// Helper to find 2D weights by pattern matching.
    pub(super) fn find_decoder_weight_2d(
        &self,
        prefix: &str,
        suffixes: &[&str],
    ) -> Result<ndarray::Array2<f32>> {
        for suffix in suffixes {
            let patterns = [
                format!("{}/{}", prefix, suffix),
                format!(
                    "{}.{}",
                    prefix.replace("/", ".").trim_start_matches('.'),
                    suffix
                ),
            ];

            for pattern in &patterns {
                if let Some(w) = self.model.weights.get(pattern) {
                    return w
                        .clone()
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| {
                            NyError::InvalidSpec(format!("Weight {} must be 2D", pattern))
                        });
                }
            }
        }

        // Try finding by substring match
        let search_key = prefix.replace("/", ".");
        for (key, value) in self.model.weights.iter() {
            if key.contains(&search_key) {
                for suffix in suffixes {
                    if key.contains(suffix) {
                        return value
                            .clone()
                            .into_dimensionality::<ndarray::Ix2>()
                            .map_err(|_| {
                                NyError::InvalidSpec(format!("Weight {} must be 2D", key))
                            });
                    }
                }
            }
        }

        Err(NyError::InvalidSpec(format!(
            "Could not find 2D weight for {} with suffixes {:?}",
            prefix, suffixes
        )))
    }
}
