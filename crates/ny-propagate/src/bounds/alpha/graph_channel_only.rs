// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Channel-only alpha helpers for `GraphAlphaState` (#4404).
//!
//! When `full_conv_alpha: False`, conv-output ReLU alpha is reduced to
//! per-channel (length C instead of C*H*W). These helpers handle:
//! - Expanding channel-only alpha to full spatial for backward pass
//! - Reducing per-neuron gradients to per-channel for optimizer updates
//! - Expanding channel-only unstable mask for GPU layer extraction
//!
//! Reference: `backward_bound.py:868-938`, `relu.py:reconstruct_full_alpha()`.

use ndarray::Array1;

use super::graph::GraphAlphaState;

impl GraphAlphaState {
    /// Spatial shape for a channel-only alpha node, or None if full alpha.
    #[must_use]
    pub(crate) fn spatial_shape(&self, node_name: &str) -> Option<&[usize]> {
        self.spatial_shapes.get(node_name).map(|v| v.as_slice())
    }

    /// Expand channel-only alpha [C] to full spatial [C*H*W] by broadcasting.
    ///
    /// If the node uses full alpha (no spatial_shape entry), returns a clone.
    /// Reference: `relu.py:reconstruct_full_alpha()`.
    #[must_use]
    pub(crate) fn expand_alpha(&self, node_name: &str, alpha: &Array1<f32>) -> Array1<f32> {
        let Some(shape) = self.spatial_shapes.get(node_name) else {
            return alpha.clone();
        };
        let channels = shape[0];
        let spatial: usize = shape[1..].iter().product();
        let mut expanded = Array1::<f32>::zeros(channels * spatial);
        for c in 0..channels {
            for s in 0..spatial {
                expanded[c * spatial + s] = alpha[c];
            }
        }
        expanded
    }

    /// Reduce per-neuron gradient [C*H*W] to per-channel [C] by summing spatial dims.
    ///
    /// If the node uses full alpha (no spatial_shape entry), returns a clone.
    /// If the gradient is already channel-sized [C], returns it unchanged.
    /// If the gradient length doesn't match either C or C*H*W (e.g., because
    /// chain-rule intermediate storage was missing), returns zeros with length C.
    /// This is the reverse of `expand_alpha` for gradient aggregation.
    #[must_use]
    pub(crate) fn reduce_gradient(&self, node_name: &str, gradient: &Array1<f32>) -> Array1<f32> {
        let Some(shape) = self.spatial_shapes.get(node_name) else {
            return gradient.clone();
        };
        let channels = shape[0];
        let spatial: usize = shape[1..].iter().product();
        let expected = channels * spatial;
        if gradient.len() == channels {
            return gradient.clone();
        }
        if gradient.len() != expected {
            return Array1::<f32>::zeros(channels);
        }
        let mut reduced = Array1::<f32>::zeros(channels);
        for c in 0..channels {
            let mut sum = 0.0f32;
            for s in 0..spatial {
                sum += gradient[c * spatial + s];
            }
            reduced[c] = sum;
        }
        reduced
    }

    /// (total, interior) counts of the raw ReLU lower alphas — interior means
    /// strictly inside (0, 1), i.e. moved off the adaptive lattice
    /// (#w4-root-alpha diagnostics).
    #[must_use]
    pub(crate) fn relu_lower_alpha_interior_count(&self) -> (usize, usize) {
        self.alphas.values().fold((0usize, 0usize), |(n, i), a| {
            (
                n + a.len(),
                i + a.iter().filter(|v| **v != 0.0 && **v != 1.0).count(),
            )
        })
    }

    /// Expand channel-only unstable mask [C] to full spatial [C*H*W] by broadcasting.
    ///
    /// If the node uses full alpha, returns a clone.
    #[must_use]
    pub(crate) fn expand_mask(&self, node_name: &str, mask: &Array1<bool>) -> Array1<bool> {
        let Some(shape) = self.spatial_shapes.get(node_name) else {
            return mask.clone();
        };
        let channels = shape[0];
        let spatial: usize = shape[1..].iter().product();
        let mut expanded = Array1::<bool>::from_elem(channels * spatial, false);
        for c in 0..channels {
            for s in 0..spatial {
                expanded[c * spatial + s] = mask[c];
            }
        }
        expanded
    }
}
