// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph/DAG alpha-CROWN state (`GraphAlphaState`).
//!
//! Stores learnable alpha parameters for unstable ReLU neurons in DAG-structured
//! graph models. Uses node names as keys (BTreeMap for deterministic iteration
//! order). Also holds monotone S-shaped and Sqrt tangent-point alpha bundles.

use crate::layers::trigonometric::{
    sigmoid_crossing_default_tangents, tanh_crossing_default_tangents,
};
use ndarray::{Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::contiguous_flat_slice;

use super::super::alpha_reciprocal::ReciprocalAlpha;
use super::super::alpha_s_shaped::MonotoneSShapedAlpha;
use super::super::alpha_sqrt::SqrtAlpha;
use super::shared::{
    extract_contiguous_bounds, init_alpha_from_bounds, update_alphas_adam, update_alphas_sgd,
};
use super::AdamParams;

/// Alpha state for DAG/GraphNetwork models.
///
/// Unlike `AlphaState` which uses indices, `GraphAlphaState` uses node names
/// as keys, since DAG models have named nodes rather than sequential layer indices.
#[derive(Debug, Clone)]
pub struct GraphAlphaState {
    /// Alpha values per ReLU node for the **lower bound path** (alpha[0]).
    /// Key is the node name.
    /// Each Array1 has length equal to the number of neurons in that ReLU node.
    /// BTreeMap ensures deterministic iteration order, which is required for
    /// reproducible SPSA gradient estimation (RNG consumption order must be
    /// consistent across runs). See #1976.
    pub(crate) alphas: std::collections::BTreeMap<String, Array1<f32>>,
    /// Alpha values per ReLU node for the **upper bound path** (alpha[1]). (#3393)
    pub(crate) alphas_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Mask for unstable neurons (l < 0 < u). Only these neurons have optimizable alpha.
    pub(crate) unstable_mask: std::collections::BTreeMap<String, Array1<bool>>,
    /// Momentum for gradient updates (velocity) - used by SGD with momentum.
    pub(crate) velocity: std::collections::BTreeMap<String, Array1<f32>>,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub(crate) adam_m: std::collections::BTreeMap<String, Array1<f32>>,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub(crate) adam_v: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path optimizer state (#3393).
    pub(crate) velocity_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path first moment estimate (#3393).
    pub(crate) adam_m_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path second moment estimate (#3393).
    pub(crate) adam_v_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Tangent-point alpha bundles for monotone Sigmoid/Tanh DAG nodes.
    pub(crate) monotone_s_shaped_alphas: std::collections::BTreeMap<String, MonotoneSShapedAlpha>,
    /// Tangent-point alpha bundles for positive-domain Sqrt DAG nodes.
    pub(crate) sqrt_alphas: std::collections::BTreeMap<String, SqrtAlpha>,
    /// Tangent-point alpha bundles for non-zero-domain Reciprocal DAG nodes.
    pub(crate) reciprocal_alphas: std::collections::BTreeMap<String, ReciprocalAlpha>,
    /// Original spatial shape for channel-only alpha nodes.
    /// When `full_conv_alpha: False`, conv-output ReLU alpha has length C instead
    /// of C*H*W. This map stores [C, H, W] for nodes using channel-only alpha,
    /// enabling expansion before backward pass and reduction after gradient
    /// computation. Absent key = full alpha (no expansion needed).
    /// Reference: `backward_bound.py:868-938`, `relu.py:658-664`.
    pub(crate) spatial_shapes: std::collections::BTreeMap<String, Vec<usize>>,
    /// Per-node negative cache for GPU-suffix offload attempts (perf only, no
    /// bound impact). Suffix extractability is a property of the GRAPH
    /// STRUCTURE from a node to the input, which never changes across alpha
    /// iterations — yet on suffix-ineligible graphs (vit attention: MatMul/
    /// Softmax/Transpose never decompose) every backward pass re-attempted the
    /// full extraction walk on every node (measured: 102 wasted walks per pass
    /// per iteration). A node lands here after BOTH the unary-chain extraction
    /// AND the resnet decomposition declined it; seed-dependent rejections
    /// (non-finite coefficients) are NOT cached. `Arc` so cheap state clones
    /// share the cache for the same graph.
    pub(crate) gpu_suffix_ineligible:
        std::sync::Arc<std::sync::RwLock<std::collections::BTreeSet<String>>>,
}

impl GraphAlphaState {
    /// Create empty state.
    pub fn new() -> Self {
        Self {
            alphas: std::collections::BTreeMap::new(),
            alphas_upper: std::collections::BTreeMap::new(),
            unstable_mask: std::collections::BTreeMap::new(),
            velocity: std::collections::BTreeMap::new(),
            adam_m: std::collections::BTreeMap::new(),
            adam_v: std::collections::BTreeMap::new(),
            velocity_upper: std::collections::BTreeMap::new(),
            adam_m_upper: std::collections::BTreeMap::new(),
            adam_v_upper: std::collections::BTreeMap::new(),
            monotone_s_shaped_alphas: std::collections::BTreeMap::new(),
            sqrt_alphas: std::collections::BTreeMap::new(),
            reciprocal_alphas: std::collections::BTreeMap::new(),
            spatial_shapes: std::collections::BTreeMap::new(),
            gpu_suffix_ineligible: std::sync::Arc::new(std::sync::RwLock::new(
                std::collections::BTreeSet::new(),
            )),
        }
    }

    /// Clone only the state a CROWN **backward pass** reads, leaving the six
    /// optimizer-state maps (`velocity`/`adam_m`/`adam_v` and their `_upper`
    /// variants) EMPTY.
    ///
    /// SPSA gradient estimation evaluates the objective at perturbed alpha
    /// values: it builds `2 * num_samples` perturbed copies of the alpha state
    /// and runs a backward pass on each. The backward pass
    /// (`run_target_backward_pass` and everything it calls) reads ONLY the seven
    /// alpha/shape fields — `alphas`, `alphas_upper`, `unstable_mask`,
    /// `monotone_s_shaped_alphas`, `sqrt_alphas`, `reciprocal_alphas`,
    /// `spatial_shapes`. The six optimizer-state maps are touched only by the
    /// Adam/SGD `update*`/`add_relu_node` paths, which a perturbation copy never
    /// reaches. A full `clone()` would deep-copy those six maps `2 * num_samples`
    /// times per optimization iteration for nothing.
    ///
    /// NUMERICALLY IDENTICAL: every field the backward pass reads is cloned
    /// bit-for-bit; the omitted maps are provably never read on that path, so the
    /// computed bounds (and their exact f32 bits) are unchanged. The result is
    /// suitable ONLY for a read-only backward pass — never route it into an
    /// optimizer `update*` step, which expects the optimizer maps populated.
    #[must_use]
    pub(crate) fn clone_for_backward(&self) -> Self {
        Self {
            alphas: self.alphas.clone(),
            alphas_upper: self.alphas_upper.clone(),
            unstable_mask: self.unstable_mask.clone(),
            // Optimizer state is not read by a backward pass — leave empty.
            velocity: std::collections::BTreeMap::new(),
            adam_m: std::collections::BTreeMap::new(),
            adam_v: std::collections::BTreeMap::new(),
            velocity_upper: std::collections::BTreeMap::new(),
            adam_m_upper: std::collections::BTreeMap::new(),
            adam_v_upper: std::collections::BTreeMap::new(),
            monotone_s_shaped_alphas: self.monotone_s_shaped_alphas.clone(),
            sqrt_alphas: self.sqrt_alphas.clone(),
            reciprocal_alphas: self.reciprocal_alphas.clone(),
            spatial_shapes: self.spatial_shapes.clone(),
            // Share the negative cache: suffix eligibility is a graph-structure
            // property, identical for every perturbation copy.
            gpu_suffix_ineligible: std::sync::Arc::clone(&self.gpu_suffix_ineligible),
        }
    }

    /// Consume a state returned by a warm-bound call and retain only data read
    /// by a later DAG warm-start.
    ///
    /// `collect_alpha_crown_bounds_dag_warm_with_engine` creates a fresh child
    /// state and therefore resets the six ReLU optimizer maps before copying
    /// the parent's lower/upper alpha values. Keeping those maps on every queued
    /// input-split domain can multiply queue memory without affecting the next
    /// bound. All warm-read parameters, masks, tangent bundles, spatial shapes,
    /// and the shared GPU-suffix cache are preserved exactly.
    #[must_use]
    pub(crate) fn into_warm_start_seed(mut self) -> Self {
        self.velocity.clear();
        self.adam_m.clear();
        self.adam_v.clear();
        self.velocity_upper.clear();
        self.adam_m_upper.clear();
        self.adam_v_upper.clear();
        self
    }

    /// Initialize alpha state from pre-activation bounds for a single ReLU node.
    ///
    /// For unstable neurons (l < 0 < u), initializes alpha using the adaptive heuristic:
    /// alpha = 1 if u > -l, else 0.
    ///
    /// When `channel_only_alpha` is true and the pre-activation has spatial dimensions
    /// (ndim >= 3, i.e., [C, H, W]), alpha is reduced to per-channel (length C) by
    /// taking per-channel worst-case bounds (min lower, max upper). This is the
    /// `full_conv_alpha: False` mode from the reference (`backward_bound.py:868-938`).
    pub fn add_relu_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
        channel_only_alpha: bool,
    ) -> Result<()> {
        let shape = pre_activation.shape();
        let use_channel_only = channel_only_alpha && shape.len() >= 3;
        tracing::debug!(
            node = node_name,
            shape = ?shape,
            channel_only = use_channel_only,
            "alpha init"
        );

        let (alpha, mask, num_alpha) = if use_channel_only {
            // Channel-only: reduce [C, H, W] bounds to [C] by taking worst-case per channel.
            // Reference: get_unstable_locations(..., channel_only=True, conv=True)
            let channels = shape[0];
            let spatial: usize = shape[1..].iter().product();

            let (lower_std, upper_std) = extract_contiguous_bounds(&pre_activation.flatten())?;
            let lower_arr = contiguous_flat_slice(&lower_std);
            let upper_arr = contiguous_flat_slice(&upper_std);

            // Reshape to [C, spatial] and reduce per channel
            let lower_2d = Array2::from_shape_vec((channels, spatial), lower_arr.to_vec())
                .map_err(|_e| ny_core::NyError::ShapeMismatch {
                    expected: vec![channels, spatial],
                    got: vec![lower_arr.len()],
                })?;
            let upper_2d = Array2::from_shape_vec((channels, spatial), upper_arr.to_vec())
                .map_err(|_e| ny_core::NyError::ShapeMismatch {
                    expected: vec![channels, spatial],
                    got: vec![upper_arr.len()],
                })?;

            let channel_lower: Array1<f32> = lower_2d.map_axis(ndarray::Axis(1), |row| {
                row.iter().copied().fold(f32::INFINITY, f32::min)
            });
            let channel_upper: Array1<f32> = upper_2d.map_axis(ndarray::Axis(1), |row| {
                row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            });

            let (alpha, mask) = init_alpha_from_bounds(
                channel_lower.as_slice().expect("contiguous"),
                channel_upper.as_slice().expect("contiguous"),
            );
            self.spatial_shapes
                .insert(node_name.to_string(), shape.to_vec());
            (alpha, mask, channels)
        } else {
            let pre_flat = pre_activation.flatten();
            let num_neurons = pre_flat.len();
            let (lower_std, upper_std) = extract_contiguous_bounds(&pre_flat)?;
            let lower_arr = contiguous_flat_slice(&lower_std);
            let upper_arr = contiguous_flat_slice(&upper_std);
            let (alpha, mask) = init_alpha_from_bounds(lower_arr.as_ref(), upper_arr.as_ref());
            (alpha, mask, num_neurons)
        };

        // Dual alpha (#3393): upper path initialized identically to lower path.
        self.alphas_upper
            .insert(node_name.to_string(), alpha.clone());
        self.alphas.insert(node_name.to_string(), alpha);
        self.unstable_mask.insert(node_name.to_string(), mask);
        self.velocity
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_m
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_v
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.velocity_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_m_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_v_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        Ok(())
    }

    /// Lower-path alpha values for a specific ReLU node.
    pub fn alpha(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.alphas.get(node_name)
    }

    /// Upper-path alpha values for a specific ReLU node (#3393).
    pub fn alpha_upper(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.alphas_upper.get(node_name)
    }

    /// Length of the alpha vector for one ReLU node.
    #[must_use]
    pub(crate) fn relu_len(&self, node_name: &str) -> Option<usize> {
        self.alphas.get(node_name).map(Array1::len)
    }

    /// Unstable mask for one ReLU node.
    #[must_use]
    pub(crate) fn relu_unstable_mask(&self, node_name: &str) -> Option<&Array1<bool>> {
        self.unstable_mask.get(node_name)
    }

    /// Lower/upper alpha pair for one ReLU node.
    #[must_use]
    pub(crate) fn relu_alpha_pair(&self, node_name: &str) -> Option<(&Array1<f32>, &Array1<f32>)> {
        Some((
            self.alphas.get(node_name)?,
            self.alphas_upper.get(node_name)?,
        ))
    }

    /// Mutable lower/upper alpha pair for one ReLU node.
    pub(crate) fn relu_alpha_pair_mut(
        &mut self,
        node_name: &str,
    ) -> Option<(&mut Array1<f32>, &mut Array1<f32>)> {
        let lower = self.alphas.get_mut(node_name)?;
        let upper = self.alphas_upper.get_mut(node_name)?;
        Some((lower, upper))
    }

    /// Register monotone Sigmoid tangent-point alpha state for one DAG node.
    pub fn add_sigmoid_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha =
            MonotoneSShapedAlpha::from_bounds(pre_activation, sigmoid_crossing_default_tangents)?;
        self.monotone_s_shaped_alphas
            .insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Register monotone Tanh tangent-point alpha state for one DAG node.
    pub fn add_tanh_node(&mut self, node_name: &str, pre_activation: &BoundedTensor) -> Result<()> {
        let alpha =
            MonotoneSShapedAlpha::from_bounds(pre_activation, tanh_crossing_default_tangents)?;
        self.monotone_s_shaped_alphas
            .insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Register positive-domain Sqrt tangent-point alpha state for one DAG node.
    ///
    /// Kept `pub(crate)` per #3773 design: all observed call sites are crate-internal
    /// DAG wiring helpers. The existing `pub` on `add_sigmoid_node`/`add_tanh_node`
    /// is acknowledged visibility debt (#2611).
    pub(crate) fn add_sqrt_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha = SqrtAlpha::from_bounds(pre_activation)?;
        self.sqrt_alphas.insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Tangent-point alpha bundle for a monotone Sigmoid/Tanh DAG node.
    #[must_use]
    pub(crate) fn monotone_s_shaped_alpha(&self, node_name: &str) -> Option<&MonotoneSShapedAlpha> {
        self.monotone_s_shaped_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG node.
    pub(crate) fn monotone_s_shaped_alpha_mut(
        &mut self,
        node_name: &str,
    ) -> Option<&mut MonotoneSShapedAlpha> {
        self.monotone_s_shaped_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG monotone alpha node names.
    pub(crate) fn monotone_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.monotone_s_shaped_alphas.keys()
    }

    /// Tangent-point alpha bundle for one DAG Sqrt node.
    #[must_use]
    pub(crate) fn sqrt_alpha(&self, node_name: &str) -> Option<&SqrtAlpha> {
        self.sqrt_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG Sqrt node.
    pub(crate) fn sqrt_alpha_mut(&mut self, node_name: &str) -> Option<&mut SqrtAlpha> {
        self.sqrt_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG Sqrt alpha node names.
    pub(crate) fn sqrt_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.sqrt_alphas.keys()
    }

    /// Register non-zero-domain Reciprocal tangent-point alpha state for one DAG node.
    pub(crate) fn add_reciprocal_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha = ReciprocalAlpha::from_bounds(pre_activation)?;
        self.reciprocal_alphas.insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Tangent-point alpha bundle for one DAG Reciprocal node.
    #[must_use]
    pub(crate) fn reciprocal_alpha(&self, node_name: &str) -> Option<&ReciprocalAlpha> {
        self.reciprocal_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG Reciprocal node.
    pub(crate) fn reciprocal_alpha_mut(&mut self, node_name: &str) -> Option<&mut ReciprocalAlpha> {
        self.reciprocal_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG Reciprocal alpha node names.
    pub(crate) fn reciprocal_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.reciprocal_alphas.keys()
    }

    /// Absolute velocity buffer for one ReLU node.
    #[must_use]
    pub(crate) fn relu_velocity(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.velocity.get(node_name)
    }

    /// Update alpha values using gradient descent with optional momentum.
    ///
    /// Delegates to `update_alphas_sgd` for the core optimization loop.
    pub fn update(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(vel) = self.velocity.get_mut(node_name) else {
            return;
        };

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update: gradient length {} != alpha length {} for '{}' (#1937), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Count total number of unstable neurons.
    pub fn num_unstable(&self) -> usize {
        self.unstable_mask
            .values()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum()
    }

    /// Get all ReLU node names.
    pub fn relu_nodes(&self) -> impl Iterator<Item = &str> {
        self.alphas.keys().map(|s| s.as_str())
    }

    /// Update alpha values using Adam optimizer.
    ///
    /// Delegates to `update_alphas_adam` for the core optimization loop.
    pub fn update_adam(&mut self, node_name: &str, gradient: &Array1<f32>, params: &AdamParams) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(m) = self.adam_m.get_mut(node_name) else {
            return;
        };
        let Some(v) = self.adam_v.get_mut(node_name) else {
            return;
        };

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_adam: gradient length {} != alpha length {} for '{}' (#1937), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }

    /// Update upper-path alpha values using gradient descent with optional momentum. (#3393)
    pub fn update_upper(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        let Some(alpha) = self.alphas_upper.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(vel) = self.velocity_upper.get_mut(node_name) else {
            return;
        };

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_upper: gradient length {} != alpha length {} for '{}' (#3393), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Update upper-path alpha values using Adam optimizer. (#3393)
    pub fn update_adam_upper(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        params: &AdamParams,
    ) {
        let Some(alpha) = self.alphas_upper.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(m) = self.adam_m_upper.get_mut(node_name) else {
            return;
        };
        let Some(v) = self.adam_v_upper.get_mut(node_name) else {
            return;
        };

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_adam_upper: gradient length {} != alpha length {} for '{}' (#3393), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }
}

impl Default for GraphAlphaState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod warm_start_seed_tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn consuming_warm_start_seed_drops_only_reset_optimizer_maps() {
        let pre_activation = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.5_f32]).into_dyn(),
            arr1(&[1.0_f32, 1.5_f32]).into_dyn(),
        )
        .expect("valid ReLU bounds");
        let mut state = GraphAlphaState::new();
        state
            .add_relu_node("relu", &pre_activation, false)
            .expect("ReLU state should initialize");
        state
            .spatial_shapes
            .insert("relu".to_string(), vec![1, 1, 2]);

        let expected_alphas = state.alphas.clone();
        let expected_alphas_upper = state.alphas_upper.clone();
        let expected_unstable_mask = state.unstable_mask.clone();
        let expected_spatial_shapes = state.spatial_shapes.clone();
        let expected_cache = std::sync::Arc::clone(&state.gpu_suffix_ineligible);
        assert!(
            [
                &state.velocity,
                &state.adam_m,
                &state.adam_v,
                &state.velocity_upper,
                &state.adam_m_upper,
                &state.adam_v_upper,
            ]
            .iter()
            .all(|map| !map.is_empty()),
            "fixture must populate every reset optimizer map"
        );

        let seed = state.into_warm_start_seed();

        assert_eq!(seed.alphas, expected_alphas);
        assert_eq!(seed.alphas_upper, expected_alphas_upper);
        assert_eq!(seed.unstable_mask, expected_unstable_mask);
        assert_eq!(seed.spatial_shapes, expected_spatial_shapes);
        assert!(std::sync::Arc::ptr_eq(
            &seed.gpu_suffix_ineligible,
            &expected_cache
        ));
        assert!(seed.velocity.is_empty());
        assert!(seed.adam_m.is_empty());
        assert!(seed.adam_v.is_empty());
        assert!(seed.velocity_upper.is_empty());
        assert!(seed.adam_m_upper.is_empty());
        assert!(seed.adam_v_upper.is_empty());
    }
}
