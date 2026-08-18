// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for GroupNorm.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::super::layer_norm::types::LayerNormCrownMode;
use super::super::trait_norm::NormLayer;
use super::super::validate::validate_norm_eps;

/// GroupNorm layer: y[c, t] = ny[c] * (x[c, t] - mean_g) / sqrt(var_g + eps) + beta[c]
///
/// Group normalization normalizes each group of channels independently across
/// all channels in the group and all spatial/time positions. For input shape
/// `[C, T]` with `G` groups, each group has `C/G` channels and normalizes
/// over `(C/G) * T` elements.
///
/// Reference: Wu & He, "Group Normalization," ECCV 2018.
///
/// Used in Demucs DConv sub-layers (dilated Conv1d + GroupNorm + GELU).
/// Part of #3205.
#[derive(Debug, Clone)]
pub struct GroupNormLayer {
    /// Scale parameter per channel (ny), shape [C]
    pub ny: Array1<f32>,
    /// Shift parameter per channel (beta), shape [C]
    pub beta: Array1<f32>,
    /// Number of groups. Must divide num_channels evenly.
    pub num_groups: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Use forward mode for IBP: compute mean/std from the center point.
    ///
    /// This is a heuristic analysis mode and is not admitted as proof
    /// authority; use conservative IBP for certified verification.
    /// Default: false (use conservative IBP)
    pub forward_mode: bool,
    /// CROWN linearization mode.
    /// Default: IbpValidated (sound Jacobian-based linearization with IBP validation).
    /// Reuses LayerNormCrownMode since the mode semantics are identical.
    pub crown_mode: LayerNormCrownMode,
}

impl GroupNormLayer {
    /// Create a new GroupNorm layer.
    ///
    /// `ny` and `beta` have shape `[num_channels]`.
    /// `num_groups` must divide `num_channels` evenly.
    /// Returns an error if eps is invalid or num_groups doesn't divide num_channels.
    pub fn new(ny: Array1<f32>, beta: Array1<f32>, num_groups: usize, eps: f32) -> Result<Self> {
        if ny.len() != beta.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![ny.len()],
                got: vec![beta.len()],
            });
        }
        if num_groups == 0 {
            return Err(NyError::InvalidSpec(
                "GroupNorm: num_groups must be > 0".to_string(),
            ));
        }
        if !ny.len().is_multiple_of(num_groups) {
            return Err(NyError::InvalidSpec(format!(
                "GroupNorm: num_channels ({}) must be divisible by num_groups ({})",
                ny.len(),
                num_groups
            )));
        }
        Ok(Self {
            ny,
            beta,
            num_groups,
            eps: validate_norm_eps(eps, "GroupNorm")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
        })
    }

    /// Create a GroupNorm layer with default ny=1 and beta=0.
    pub fn new_default(num_channels: usize, num_groups: usize, eps: f32) -> Result<Self> {
        Self::new(
            Array1::ones(num_channels),
            Array1::zeros(num_channels),
            num_groups,
            eps,
        )
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.ny.len()
    }

    /// Channels per group.
    pub fn channels_per_group(&self) -> usize {
        self.ny.len() / self.num_groups
    }

    /// Enable or disable forward mode.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.forward_mode = enabled;
        self
    }

    /// Set the CROWN linearization mode.
    pub fn with_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.crown_mode = mode;
        self
    }
}

impl NormLayer for GroupNormLayer {
    fn layer_name(&self) -> &'static str {
        "GroupNorm"
    }

    fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        // Flat eval: split into groups, eval each group, concatenate.
        // Input is flat [C*T]. Groups partition the C channels.
        let num_channels = self.num_channels();
        let cpg = self.channels_per_group();
        let total = x.len();

        // Explicit zero-channel guard: `%`/`is_multiple_of` on divisor 0 would
        // panic/accept-empty; a 0-channel layer is malformed, reject cleanly.
        if num_channels == 0 || !total.is_multiple_of(num_channels) {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![total],
            });
        }
        let time_len = total / num_channels;

        let mut y = Array1::<f32>::zeros(total);
        for g in 0..self.num_groups {
            let group_start_ch = g * cpg;
            // Collect all elements in this group: cpg channels × time_len
            let group_size = cpg * time_len;
            let mut group_vals = Vec::with_capacity(group_size);
            for c_offset in 0..cpg {
                let c = group_start_ch + c_offset;
                let start = c * time_len;
                for t in 0..time_len {
                    group_vals.push(x[start + t]);
                }
            }
            let y_group = self.eval_group(&group_vals, g, cpg, time_len)?;
            // Write back
            for c_offset in 0..cpg {
                let c = group_start_ch + c_offset;
                let start = c * time_len;
                for t in 0..time_len {
                    y[start + t] = y_group[c_offset * time_len + t];
                }
            }
        }
        Ok(y)
    }

    fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>> {
        // Flat Jacobian: block-diagonal, one (cpg*T)×(cpg*T) block per group.
        let num_channels = self.num_channels();
        let cpg = self.channels_per_group();
        let total = x.len();

        // Explicit zero-channel guard (see `eval`): reject rather than panic on `% 0`.
        if num_channels == 0 || !total.is_multiple_of(num_channels) {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![total],
            });
        }
        let time_len = total / num_channels;
        let group_size = cpg * time_len;

        let mut jacobian = Array2::<f32>::zeros((total, total));
        for g in 0..self.num_groups {
            let group_start_ch = g * cpg;
            // Collect group elements
            let mut group_vals = Vec::with_capacity(group_size);
            for c_offset in 0..cpg {
                let c = group_start_ch + c_offset;
                let start = c * time_len;
                for t in 0..time_len {
                    group_vals.push(x[start + t]);
                }
            }
            let j_group = self.jacobian_group(&group_vals, g, cpg, time_len)?;

            // Write block into full Jacobian.
            // Mapping: group local index i -> global index
            for i_local in 0..group_size {
                let i_c_offset = i_local / time_len;
                let i_t = i_local % time_len;
                let i_global = (group_start_ch + i_c_offset) * time_len + i_t;
                for j_local in 0..group_size {
                    let j_c_offset = j_local / time_len;
                    let j_t = j_local % time_len;
                    let j_global = (group_start_ch + j_c_offset) * time_len + j_t;
                    jacobian[[i_global, j_global]] = j_group[[i_local, j_local]];
                }
            }
        }
        Ok(jacobian)
    }

    fn crown_mode(&self) -> LayerNormCrownMode {
        self.crown_mode
    }
}
