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

    /// Checked channel-to-spatial expansion for proof-adjacent optional lanes.
    ///
    /// The historical [`Self::expand_alpha`] assumes metadata minted by
    /// `add_relu_node` and indexes `shape[0]`/`alpha[c]` directly.  A dark
    /// experiment must not turn malformed imported metadata into a panic, an
    /// overflowed allocation, or a partially transported state.  This face
    /// therefore validates the complete shape/parameter identity and declines
    /// atomically.
    pub(crate) fn try_expand_alpha(
        &self,
        node_name: &str,
        alpha: &Array1<f32>,
        expected_shape: &[usize],
    ) -> Option<Array1<f32>> {
        let expected_len = expected_shape
            .iter()
            .try_fold(1usize, |product, &dimension| {
                (dimension != 0)
                    .then_some(())
                    .and_then(|()| product.checked_mul(dimension))
            })?;
        let Some(shape) = self.spatial_shapes.get(node_name) else {
            return (alpha.len() == expected_len && alpha.iter().all(|value| value.is_finite()))
                .then(|| alpha.clone());
        };
        if shape.as_slice() != expected_shape {
            return None;
        }
        let (&channels, spatial_shape) = shape.split_first()?;
        if channels == 0
            || spatial_shape.contains(&0)
            || alpha.len() != channels
            || alpha.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let spatial = spatial_shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))?;
        let total = channels.checked_mul(spatial)?;
        if total != expected_len {
            return None;
        }
        let mut values = Vec::new();
        values.try_reserve_exact(total).ok()?;
        for &value in alpha {
            values.extend(std::iter::repeat_n(value, spatial));
        }
        (values.len() == total).then(|| Array1::from_vec(values))
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

    /// #channel-alpha-grad: MEASURED-shape key for the spatial→channel
    /// gradient reduction `dL/dα_c = Σ_{h,w} dL/dα_{c,h,w}` (the exact chain
    /// rule through the channel-shared α broadcast: moving α_c moves EVERY
    /// spatial position of channel c, so the true derivative is the spatial
    /// sum).
    ///
    /// Returns `Some((channels, spatial))` exactly when every measured shape
    /// reconciles:
    /// - this node has recorded conv geometry (`spatial_shapes[node]`),
    /// - the STORED α vector has length `channels == spatial_shapes[node][0]`,
    /// - `alpha_len == channels` (the caller's target layout IS the stored α),
    /// - `grad_len == channels · Π spatial_shapes[node][1..]` (checked mul),
    /// - the reduction is not the identity (`grad_len != alpha_len`).
    ///
    /// Never keyed on a config flag. `None` ⇒ the layouts do not reconcile
    /// and the caller must keep its existing typed refusal.
    #[must_use]
    pub(crate) fn channel_reduction_geometry(
        &self,
        node_name: &str,
        alpha_len: usize,
        grad_len: usize,
    ) -> Option<(usize, usize)> {
        let shape = self.spatial_shapes.get(node_name)?;
        let (&channels, spatial_shape) = shape.split_first()?;
        if channels == 0 || spatial_shape.is_empty() || spatial_shape.contains(&0) {
            return None;
        }
        let spatial = spatial_shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))?;
        let total = channels.checked_mul(spatial)?;
        if alpha_len != channels
            || self.alphas.get(node_name).map(Array1::len) != Some(channels)
            || grad_len != total
            || grad_len == alpha_len
        {
            return None;
        }
        Some((channels, spatial))
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
