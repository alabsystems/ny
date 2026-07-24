// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared BatchNorm layout decoding and flattened scale/bias expansion.

use ndarray::Array1;
use ny_core::{checked_shape_product, NyError, Result};

use super::types::BatchNormLayer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BatchNormInputLayout {
    pub(super) channel_idx: usize,
    pub(super) num_channels: usize,
    pub(super) elements_per_channel: usize,
    pub(super) batch_size: usize,
    pub(super) total_elements: usize,
    pub(super) chw: usize,
}

impl BatchNormInputLayout {
    fn channel_for_flat_index(self, flat_idx: usize) -> usize {
        debug_assert_eq!(self.total_elements, self.batch_size * self.chw);
        let channel_idx = (flat_idx % self.chw) / self.elements_per_channel;
        debug_assert!(channel_idx < self.num_channels);
        channel_idx
    }
}

pub(super) fn detect_input_layout(
    shape: &[usize],
    expected_channels: usize,
    expected_total_elements: Option<usize>,
) -> Result<BatchNormInputLayout> {
    if shape.is_empty() {
        return Err(NyError::InvalidSpec(
            "BatchNorm: rank-0 input not supported".to_string(),
        ));
    }

    let channel_idx = if shape.len() >= 4 {
        // The squeezed, unbatched conversion emits feature tensors of rank <= 3
        // ([C], [C, L], [C, H, W]), so a rank-4+ input necessarily carries a
        // leading batch axis ([N, C, ...]) where the channel axis is 1 by NCHW
        // layout. Resolving by value here would mistake a leading batch axis
        // for the channel axis whenever its extent collides with
        // `expected_channels`; the `num_channels` check below rejects
        // (fail-closed) any rank-4+ input whose axis 1 does not match.
        1
    } else if shape.len() == 3 {
        // Rank 3 is either a squeezed [C, H, W] feature map (channel axis 0)
        // or a batched [N, C, L] tensor (channel axis 1). [C, C', W] with
        // C' == C is indistinguishable from [N == C, C, L] by shape alone, so
        // axis 0 wins: the squeezed convention is the only rank-3 producer on
        // the unbatched path, and batch-stacking callers must not route
        // BatchNorm graphs here (enforced by `is_input_split_batch_stack_safe`).
        if shape[0] == expected_channels {
            0
        } else {
            1
        }
    } else if shape.len() == 2 {
        if shape[1] == expected_channels {
            1
        } else if shape[0] == expected_channels {
            0
        } else {
            1
        }
    } else {
        0
    };

    let num_channels = shape[channel_idx];
    if num_channels != expected_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_channels],
            got: vec![num_channels],
        });
    }

    let elements_per_channel = if channel_idx + 1 >= shape.len() {
        1
    } else {
        checked_shape_product(&shape[channel_idx + 1..]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchNorm: spatial dimensions {:?} overflow usize",
                &shape[channel_idx + 1..],
            ))
        })?
    };

    if elements_per_channel == 0 {
        return Err(NyError::InvalidSpec(
            "BatchNorm: zero-valued spatial dimension in input shape".to_string(),
        ));
    }

    let batch_size = if channel_idx == 0 { 1 } else { shape[0] };
    let chw = num_channels
        .checked_mul(elements_per_channel)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchNorm: channel/spatial dimensions {:?} overflow usize",
                &shape[channel_idx..],
            ))
        })?;
    let total_elements = batch_size.checked_mul(chw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "BatchNorm: input dimensions {:?} overflow usize",
            shape
        ))
    })?;

    if let Some(expected_total_elements) = expected_total_elements {
        if total_elements != expected_total_elements {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_elements],
                got: vec![expected_total_elements],
            });
        }
    }

    Ok(BatchNormInputLayout {
        channel_idx,
        num_channels,
        elements_per_channel,
        batch_size,
        total_elements,
        chw,
    })
}

impl BatchNormLayer {
    /// Expand the stored per-channel affine parameters and their certified
    /// precompute errors to one entry per flattened input position.
    ///
    /// This is the shape-aware source of truth shared by BatchNorm's IBP/CROWN
    /// implementations and forward-linear image composition.  Keeping the
    /// rank/layout heuristic here is important: duplicating it at a caller can
    /// silently select the leading batch axis instead of the channel axis for
    /// squeezed `[C, H, W]` tensors.
    pub(crate) fn expanded_affine_parameters(
        &self,
        input_shape: &[usize],
        expected_total_elements: usize,
    ) -> Result<(Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>)> {
        let layout = detect_input_layout(
            input_shape,
            self.num_channels,
            Some(expected_total_elements),
        )?;
        let (scale, bias) = self.expand_scale_bias(&layout);
        let (scale_err, bias_err) = self.expand_errs(&layout);
        Ok((scale, bias, scale_err, bias_err))
    }

    /// #cgan-bn-gpu-extract: pub(crate) face of `detect_input_layout` for the GPU
    /// CROWN 1x1-conv BatchNorm extraction: `(num_channels, elements_per_channel)`
    /// for an UNBATCHED input shape. Errs on batched/mismatched shapes (fail-closed).
    pub(crate) fn gpu_extraction_layout(
        &self,
        input_shape: &[usize],
        expected_total: Option<usize>,
    ) -> Result<(usize, usize)> {
        let layout = detect_input_layout(input_shape, self.num_channels, expected_total)?;
        if layout.batch_size != 1 {
            return Err(NyError::InvalidSpec(
                "BatchNorm GPU extraction: batched input unsupported".to_string(),
            ));
        }
        Ok((layout.num_channels, layout.elements_per_channel))
    }

    pub(super) fn expand_scale_bias(
        &self,
        layout: &BatchNormInputLayout,
    ) -> (Array1<f32>, Array1<f32>) {
        let expanded_scale = Array1::from_shape_fn(layout.total_elements, |flat_idx| {
            let channel_idx = layout.channel_for_flat_index(flat_idx);
            self.scale[[channel_idx]]
        });
        let expanded_bias = Array1::from_shape_fn(layout.total_elements, |flat_idx| {
            let channel_idx = layout.channel_for_flat_index(flat_idx);
            self.bias[[channel_idx]]
        });
        (expanded_scale, expanded_bias)
    }

    /// Per-flat-position expansion of the precompute error bounds
    /// (`scale_err`, `bias_err`), mirroring [`expand_scale_bias`]. Used by the
    /// CROWN paths to fold the f32 precompute error of `scale`/`bias` outward into
    /// the bias over the input box (#batchnorm-ibp-directed-rounding).
    pub(super) fn expand_errs(&self, layout: &BatchNormInputLayout) -> (Array1<f32>, Array1<f32>) {
        let expanded_scale_err = Array1::from_shape_fn(layout.total_elements, |flat_idx| {
            let channel_idx = layout.channel_for_flat_index(flat_idx);
            self.scale_err[[channel_idx]]
        });
        let expanded_bias_err = Array1::from_shape_fn(layout.total_elements, |flat_idx| {
            let channel_idx = layout.channel_for_flat_index(flat_idx);
            self.bias_err[[channel_idx]]
        });
        (expanded_scale_err, expanded_bias_err)
    }
}
