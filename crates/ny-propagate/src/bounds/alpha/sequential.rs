// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential alpha-CROWN state (`AlphaState`).
//!
//! Stores learnable alpha parameters for unstable ReLU neurons across all layers
//! in a sequential network. Also holds bilinear alpha parameters for BilinearCrown
//! layers (attention Q@K^T) and INVPROP ny state.

use crate::invprop::{InvpropState, OutputConstraints};
use ndarray::{Array1, Array4};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

use crate::contiguous_flat_slice;

use super::shared::{
    extract_contiguous_bounds, init_alpha_from_bounds, update_alphas_adam, update_alphas_sgd,
};
use super::AdamParams;

/// State for alpha-CROWN optimization.
///
/// Stores the learnable alpha parameters for unstable ReLU neurons across all layers.
/// `alpha[layer_idx][neuron_idx]` in [0, 1] is the lower bound slope for crossing ReLUs.
///
/// Also stores bilinear alpha parameters for BilinearCrown layers (attention Q@K^T).
/// Bilinear alphas have shape [4, m, n, k] with direction-dependent McCormick face selection.
///
/// The 4-parameter layout matches auto_LiRPA's `_init_opt_parameters_impl`
/// (bivariate.py:128-135). Each pair is optimized independently based on the sign of
/// downstream A:
/// - `[0]`: r_l for positive downstream A (lower bound, positive face)
/// - `[1]`: r_l for negative downstream A (lower bound, negative face)
/// - `[2]`: r_u for positive downstream A (upper bound, positive face)
/// - `[3]`: r_u for negative downstream A (upper bound, negative face)
///
/// Reference: Xu et al., "Automatic Perturbation Analysis", Appendix C (McCormick
/// interpolation), https://openreview.net/pdf?id=BJxwPJHFwS
#[derive(Debug, Clone)]
pub struct AlphaState {
    /// Alpha values per ReLU layer for the **lower bound path** (alpha[0] in reference).
    /// Index is the layer index in the network.
    /// Each Array1 has length equal to the number of neurons in that ReLU layer.
    /// For stable neurons (always positive or negative), alpha is unused but stored.
    pub(crate) alphas: Vec<Array1<f32>>,
    /// Alpha values per ReLU layer for the **upper bound path** (alpha[1] in reference).
    /// Used when `ua < 0` in the CROWN backward pass.
    /// Separate from `alphas` to allow independent optimization of lower and upper
    /// bound relaxation slopes. (#3393)
    /// Reference: auto_LiRPA/operators/relu.py:647-652 `selected_alpha[0]`/`[1]`
    pub(crate) alphas_upper: Vec<Array1<f32>>,
    /// Mask for unstable neurons (l < 0 < u). Only these neurons have optimizable alpha.
    pub(crate) unstable_mask: Vec<Array1<bool>>,
    /// Momentum for gradient updates (velocity) - used by SGD with momentum.
    pub(crate) velocity: Vec<Array1<f32>>,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub(crate) adam_m: Vec<Array1<f32>>,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub(crate) adam_v: Vec<Array1<f32>>,
    /// Upper-path optimizer state for dual alpha (#3393).
    pub(crate) velocity_upper: Vec<Array1<f32>>,
    /// Upper-path first moment estimate for Adam optimizer (#3393).
    pub(crate) adam_m_upper: Vec<Array1<f32>>,
    /// Upper-path second moment estimate for Adam optimizer (#3393).
    pub(crate) adam_v_upper: Vec<Array1<f32>>,
    // Bilinear alpha parameters for BilinearCrown layers (attention Q@K^T)
    /// Bilinear alpha values per BilinearCrown layer. Key is layer index in network.
    /// Each Array4 has shape [4, m, n, k] for direction-dependent McCormick interpolation.
    ///
    /// Reference: auto_LiRPA/operators/bivariate.py:128-135 `_init_opt_parameters_impl`
    pub(crate) bilinear_alphas: HashMap<usize, Array4<f32>>,
    /// First moment estimate for bilinear Adam optimizer.
    pub(crate) bilinear_adam_m: HashMap<usize, Array4<f32>>,
    /// Second moment estimate for bilinear Adam optimizer.
    pub(crate) bilinear_adam_v: HashMap<usize, Array4<f32>>,

    /// State for INVPROP output constraint ny optimization.
    ///
    /// When Some, ny dual variables are optimized alongside alphas.
    pub(crate) invprop_state: Option<InvpropState>,
}

impl AlphaState {
    /// Initialize alpha state from pre-activation bounds.
    ///
    /// For each ReLU layer, identifies unstable neurons and initializes alpha using
    /// the adaptive heuristic: alpha = 1 if u > -l, else 0.
    pub fn from_preactivation_bounds(
        layer_bounds: &[BoundedTensor],
        relu_layer_indices: &[usize],
    ) -> Result<Self> {
        let mut alphas = Vec::with_capacity(relu_layer_indices.len());
        let mut unstable_mask = Vec::with_capacity(relu_layer_indices.len());
        let mut velocity = Vec::with_capacity(relu_layer_indices.len());

        for &layer_idx in relu_layer_indices {
            let pre_bounds = &layer_bounds[layer_idx];
            let pre_flat = pre_bounds.flatten();
            let num_neurons = pre_flat.len();

            let (lower_std, upper_std) = extract_contiguous_bounds(&pre_flat)?;
            let lower_arr = contiguous_flat_slice(&lower_std);
            let upper_arr = contiguous_flat_slice(&upper_std);

            let (alpha, mask) = init_alpha_from_bounds(lower_arr.as_ref(), upper_arr.as_ref());

            alphas.push(alpha);
            unstable_mask.push(mask);
            velocity.push(Array1::<f32>::zeros(num_neurons));
        }

        // Initialize Adam moment estimates
        let adam_m = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();
        let adam_v = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();

        // Dual alpha (#3393): upper path alphas initialized identically to lower path.
        // Both paths start with the same heuristic (u > -l ? 1.0 : 0.0) and diverge
        // during optimization as they receive independent gradients.
        let alphas_upper = alphas.clone();
        let velocity_upper = velocity.clone();
        let adam_m_upper: Vec<Array1<f32>> = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();
        let adam_v_upper: Vec<Array1<f32>> = alphas
            .iter()
            .map(|a| Array1::<f32>::zeros(a.len()))
            .collect();

        Ok(Self {
            alphas,
            alphas_upper,
            unstable_mask,
            velocity,
            adam_m,
            adam_v,
            velocity_upper,
            adam_m_upper,
            adam_v_upper,
            bilinear_alphas: HashMap::new(),
            bilinear_adam_m: HashMap::new(),
            bilinear_adam_v: HashMap::new(),
            invprop_state: None,
        })
    }

    /// Lower-path alpha values for a specific ReLU layer (by index in relu_layer_indices).
    /// This is alpha[0] in the reference (used when lA > 0).
    pub fn alpha(&self, relu_idx: usize) -> Option<&Array1<f32>> {
        self.alphas.get(relu_idx)
    }

    /// Upper-path alpha values for a specific ReLU layer (by index in relu_layer_indices).
    /// This is alpha[1] in the reference (used when uA < 0). (#3393)
    pub fn alpha_upper(&self, relu_idx: usize) -> Option<&Array1<f32>> {
        self.alphas_upper.get(relu_idx)
    }

    /// Initialize bilinear alpha parameters for a BilinearCrown layer.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the BilinearCrown layer in the network
    /// * `m` - Number of rows in output (query sequence length)
    /// * `n` - Number of columns in output (key sequence length)
    /// * `k` - Inner dimension (head dimension)
    ///
    /// Alpha array has shape [4, m, n, k] initialized to 1.0 (auto_LiRPA default).
    /// The 4 slices correspond to (r_l_pos, r_l_neg, r_u_pos, r_u_neg) —
    /// direction-dependent face selection matching auto_LiRPA's 4-alpha layout.
    ///
    /// Reference: auto_LiRPA/operators/bivariate.py:134 `alpha = torch.ones(4, ...)`
    pub fn init_bilinear_alpha(&mut self, layer_idx: usize, m: usize, n: usize, k: usize) {
        let alpha = Array4::ones((4, m, n, k));
        let adam_m = Array4::zeros((4, m, n, k));
        let adam_v = Array4::zeros((4, m, n, k));
        self.bilinear_alphas.insert(layer_idx, alpha);
        self.bilinear_adam_m.insert(layer_idx, adam_m);
        self.bilinear_adam_v.insert(layer_idx, adam_v);
    }

    /// Bilinear alpha parameters for a layer.
    pub fn bilinear_alpha(&self, layer_idx: usize) -> Option<&Array4<f32>> {
        self.bilinear_alphas.get(&layer_idx)
    }

    /// Mutable bilinear alpha parameters for a layer.
    pub fn bilinear_alpha_mut(&mut self, layer_idx: usize) -> Option<&mut Array4<f32>> {
        self.bilinear_alphas.get_mut(&layer_idx)
    }

    /// Get all bilinear alpha layer indices.
    pub fn bilinear_layer_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.bilinear_alphas.keys().copied().collect();
        indices.sort_unstable();
        indices
    }

    /// Count total number of bilinear alpha parameters.
    pub fn num_bilinear_params(&self) -> usize {
        self.bilinear_alphas.values().map(|a| a.len()).sum()
    }

    /// Update bilinear alpha values using Adam optimizer.
    ///
    /// All elements are optimizable (no unstable mask — McCormick interpolation always
    /// benefits from face selection optimization).
    ///
    /// # Arguments
    /// * `layer_idx` - BilinearCrown layer index
    /// * `gradient` - Gradient array of same shape as bilinear alpha [4, m, n, k]
    /// * `params` - Adam hyperparameters (lr, beta1, beta2, epsilon, t)
    pub fn update_bilinear_adam(
        &mut self,
        layer_idx: usize,
        gradient: &Array4<f32>,
        params: &AdamParams,
    ) {
        let Some(alpha) = self.bilinear_alphas.get_mut(&layer_idx) else {
            return;
        };
        let Some(m) = self.bilinear_adam_m.get_mut(&layer_idx) else {
            return;
        };
        let Some(v) = self.bilinear_adam_v.get_mut(&layer_idx) else {
            return;
        };

        if gradient.shape() != alpha.shape() {
            tracing::warn!(
                "AlphaState::update_bilinear_adam: gradient shape {:?} != alpha shape {:?} for layer {}, skipping",
                gradient.shape(), alpha.shape(), layer_idx
            );
            return;
        }

        let t_f = params.t.max(1) as f32;
        let bias_correction1 = (1.0 - params.beta1.powf(t_f)).max(f32::EPSILON);
        let bias_correction2 = (1.0 - params.beta2.powf(t_f)).max(f32::EPSILON);

        // Element-wise Adam update over the [4, m, n, k] array.
        ndarray::Zip::from(alpha.view_mut())
            .and(gradient.view())
            .and(m.view_mut())
            .and(v.view_mut())
            .for_each(|a, &g, m_val, v_val| {
                *m_val = params.beta1 * *m_val + (1.0 - params.beta1) * g;
                *v_val = params.beta2 * *v_val + (1.0 - params.beta2) * g * g;

                let m_hat = *m_val / bias_correction1;
                let v_hat = *v_val / bias_correction2;

                *a -= params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);
                *a = a.clamp(0.0, 1.0);
                if a.is_nan() {
                    *a = 0.5;
                    *m_val = 0.0;
                    *v_val = 0.0;
                }
            });
    }

    // ==================== INVPROP NY METHODS ====================

    /// Initialize INVPROP state for output constraint optimization.
    ///
    /// # Errors
    /// Returns `NyError::InternalError` if INVPROP state is already initialized.
    pub fn init_invprop_state(
        &mut self,
        constraints: OutputConstraints,
        batch_size: usize,
    ) -> Result<()> {
        if self.invprop_state.is_some() {
            return Err(NyError::InternalError(
                "AlphaState::init_invprop_state called on already-initialized state".to_string(),
            ));
        }
        self.invprop_state = Some(InvpropState::new(constraints, batch_size));
        Ok(())
    }

    /// Check if INVPROP is active.
    #[must_use]
    pub fn has_invprop(&self) -> bool {
        self.invprop_state.is_some()
    }

    /// Get immutable reference to INVPROP state.
    #[must_use]
    pub fn invprop(&self) -> Option<&InvpropState> {
        self.invprop_state.as_ref()
    }

    /// Get mutable reference to INVPROP state.
    pub fn invprop_mut(&mut self) -> Option<&mut InvpropState> {
        self.invprop_state.as_mut()
    }

    /// Clip all ny parameters to enforce non-negativity.
    pub fn clip_gammas(&mut self) {
        if let Some(ref mut state) = self.invprop_state {
            state.clip_all_gammas();
        }
    }

    /// Get total number of ny parameters.
    #[must_use]
    pub fn num_ny_params(&self) -> usize {
        self.invprop_state
            .as_ref()
            .map(|s| s.all_ny_params().len())
            .unwrap_or(0)
    }

    /// Get all ny parameters as a flat vector.
    #[must_use]
    pub fn ny_params(&self) -> Vec<f32> {
        self.invprop_state
            .as_ref()
            .map(|s| s.all_ny_params())
            .unwrap_or_default()
    }

    /// Update ny parameters from a flat vector.
    ///
    /// Returns `Err` if params length doesn't match expected ny count.
    /// No-op (returns `Ok`) if INVPROP state is not initialized. (#2712)
    pub fn update_ny_params(&mut self, params: &[f32]) -> Result<()> {
        if let Some(ref mut state) = self.invprop_state {
            state.update_ny_params(params)?;
        }
        Ok(())
    }

    // ==================== END INVPROP NY METHODS ====================

    /// Update alpha values using gradient descent with optional momentum.
    ///
    /// gradient: d(loss)/d(alpha) (where loss = -lower_bound, so minimize loss = maximize lower)
    /// learning_rate: step size
    /// momentum: momentum coefficient (0 = no momentum)
    pub fn update(
        &mut self,
        relu_idx: usize,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        if relu_idx >= self.alphas.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas[relu_idx];
        let vel = &mut self.velocity[relu_idx];

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "AlphaState::update: gradient length {} != alpha length {} for relu_idx {} (#1937), skipping",
                gradient.len(), alpha.len(), relu_idx
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Count total number of unstable neurons.
    pub fn num_unstable(&self) -> usize {
        self.unstable_mask
            .iter()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum()
    }

    /// Update alpha values using Adam optimizer.
    ///
    /// Delegates to `update_alphas_adam` for the core optimization loop.
    pub fn update_adam(&mut self, relu_idx: usize, gradient: &Array1<f32>, params: &AdamParams) {
        if relu_idx >= self.alphas.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas[relu_idx];
        let m = &mut self.adam_m[relu_idx];
        let v = &mut self.adam_v[relu_idx];

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "AlphaState::update_adam: gradient length {} != alpha length {} for relu_idx {} (#1937), skipping",
                gradient.len(), alpha.len(), relu_idx
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }

    /// Update upper-path alpha values using gradient descent with momentum (#3393).
    pub fn update_upper(
        &mut self,
        relu_idx: usize,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        if relu_idx >= self.alphas_upper.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas_upper[relu_idx];
        let vel = &mut self.velocity_upper[relu_idx];

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "AlphaState::update_upper: gradient length {} != alpha length {} for relu_idx {} (#3393), skipping",
                gradient.len(), alpha.len(), relu_idx
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Update upper-path alpha values using Adam optimizer (#3393).
    pub fn update_adam_upper(
        &mut self,
        relu_idx: usize,
        gradient: &Array1<f32>,
        params: &AdamParams,
    ) {
        if relu_idx >= self.alphas_upper.len() {
            return;
        }

        let mask = &self.unstable_mask[relu_idx];
        let alpha = &mut self.alphas_upper[relu_idx];
        let m = &mut self.adam_m_upper[relu_idx];
        let v = &mut self.adam_v_upper[relu_idx];

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "AlphaState::update_adam_upper: gradient length {} != alpha length {} for relu_idx {} (#3393), skipping",
                gradient.len(), alpha.len(), relu_idx
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }
}
