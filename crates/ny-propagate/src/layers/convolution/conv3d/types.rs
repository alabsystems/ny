// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `Conv3dLayer` — volumetric convolution geometry and validation.
//!
//! Mirrors [`crate::layers::Conv2dLayer`]'s structure: publicly mutable
//! geometry fields with constructor-established invariants, a
//! [`Conv3dLayer::validate_geometry`] that EVERY propagation entry point
//! re-runs before indexing or arithmetic (callers can mutate the public fields
//! after construction), and fallible `try_*` accessors beside the legacy
//! infallible ones.
//!
//! Layout follows PyTorch `torch.nn.Conv3d` / ONNX `Conv` at spatial rank 3:
//! kernel `(out_channels, in_channels/groups, kD, kH, kW)`, input
//! `(C, D, H, W)` or `(N, C, D, H, W)`, spatial parameters ordered
//! `[depth, height, width]` — the same `[usize; 3]` ordering `nn`'s
//! `Conv3dConfig` and the trace schema's `TraceOp::Conv3d` use, so a bridge arm
//! can hand its attributes straight across.

use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};

use super::ops_ibp_cert::checked_conv3d_output_extent;

/// 3-D convolution with interval bound propagation.
///
/// # Supported parameters
///
/// `stride`, `padding`, `dilation` (all per-axis `[d, h, w]`) and `groups` —
/// exactly the set [`crate::layers::Conv2dLayer`] supports, generalised to the
/// extra spatial axis. Everything else is refused BY NAME at construction /
/// validation: zero stride, zero dilation, zero groups, a kernel that is not
/// rank-5, a zero kernel extent, an `out_channels` that is not a multiple of
/// `groups`, a bias whose length is not `out_channels`, and (in the ONNX
/// converter) `auto_pad` and asymmetric `pads`. There is no `output_padding`
/// here — that belongs to the transposed convolution, which this layer is not.
///
/// # Bound propagation
///
/// IBP only. [`crate::layers::BoundPropagation::propagate_ibp`] runs the
/// certified directed-binary64 interval forward in
/// [`super::ops_ibp_cert`]; the CROWN backward FAILS CLOSED with
/// [`NyError::UnsupportedOp`] (see `super::bound`) rather than publishing a
/// relation it cannot certify.
#[derive(Debug, Clone)]
pub struct Conv3dLayer {
    /// Convolution kernel of shape `(out_channels, in_channels/groups, kD, kH, kW)`.
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape `(out_channels,)`.
    pub bias: Option<Array1<f32>>,
    /// Stride `[depth, height, width]`. Every entry must be >= 1.
    pub stride: [usize; 3],
    /// Zero padding `[depth, height, width]`, applied symmetrically.
    pub padding: [usize; 3],
    /// Dilation `[depth, height, width]`, spacing between kernel taps. Default `[1, 1, 1]`.
    pub dilation: [usize; 3],
    /// Groups (blocked connections from input to output channels). Default 1.
    pub groups: usize,
    /// Input spatial dimensions `[depth, height, width]`, when known.
    ///
    /// Recorded for consumers that need the volumetric geometry of a flattened
    /// activation. The IBP forward reads the geometry from the input tensor
    /// itself and never depends on this field.
    pub input_shape: Option<[usize; 3]>,
}

impl Conv3dLayer {
    /// Create a Conv3d layer with `dilation = [1, 1, 1]` and `groups = 1`.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, 1)
    }

    /// Create a Conv3d layer with explicit groups and `dilation = [1, 1, 1]`.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: [usize; 3],
        padding: [usize; 3],
        groups: usize,
    ) -> Result<Self> {
        Self::new_dilated(kernel, bias, stride, padding, [1, 1, 1], groups)
    }

    /// Create a Conv3d layer with explicit dilation and groups.
    ///
    /// Kernel shape: `(out_channels, in_channels/groups, kD, kH, kW)`.
    /// Reference: PyTorch `torch.nn.Conv3d`, ONNX `Conv` `dilations`/`group`.
    pub fn new_dilated(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> Result<Self> {
        let layer = Self {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_shape: None,
        };
        layer.validate_geometry()?;
        Ok(layer)
    }

    /// Create a Conv3d layer with known input spatial dimensions.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: [usize; 3],
        padding: [usize; 3],
        input_shape: [usize; 3],
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some(input_shape);
        Ok(layer)
    }

    /// Create a Conv3d layer with dilation, groups, and known input dimensions.
    pub fn with_input_shape_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
        input_shape: [usize; 3],
    ) -> Result<Self> {
        let mut layer = Self::new_dilated(kernel, bias, stride, padding, dilation, groups)?;
        layer.input_shape = Some(input_shape);
        Ok(layer)
    }

    /// Record the input spatial dimensions `[depth, height, width]`.
    pub fn set_input_shape(&mut self, input_shape: [usize; 3]) {
        self.input_shape = Some(input_shape);
    }

    /// Revalidate geometry stored in the publicly mutable fields, returning
    /// `(total_in_channels, out_channels)`.
    ///
    /// Constructors establish these invariants, but callers can mutate the
    /// public fields afterwards, so every propagation / output-size entry point
    /// calls this BEFORE indexing shapes or doing arithmetic. Refusals name the
    /// offending parameter.
    ///
    /// # Errors
    ///
    /// Rank != 5 kernel, a zero kernel extent, zero stride/dilation/groups, an
    /// `out_channels` that is not a multiple of `groups`, a grouped
    /// input-channel product that overflows, or a bias whose length is not
    /// `out_channels`.
    pub fn validate_geometry(&self) -> Result<(usize, usize)> {
        if self.kernel.ndim() != 5 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0, 0],
                got: self.kernel.shape().to_vec(),
            });
        }
        if self.stride.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "Conv3d stride must be >= 1 on every axis, got stride={:?}",
                self.stride
            )));
        }
        if self.dilation.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "Conv3d dilation must be >= 1 on every axis, got dilation={:?}",
                self.dilation
            )));
        }
        if self.groups == 0 {
            return Err(NyError::InvalidSpec(
                "Conv3d groups must be >= 1, got 0".to_string(),
            ));
        }
        let shape = self.kernel.shape();
        let out_channels = shape[0];
        let input_channels_per_group = shape[1];
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "Conv3d kernel dimensions must be nonzero, got {shape:?}"
            )));
        }
        let in_channels = input_channels_per_group
            .checked_mul(self.groups)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Conv3d total input channels overflow: {input_channels_per_group} * {}",
                    self.groups
                ))
            })?;
        if !out_channels.is_multiple_of(self.groups) {
            return Err(NyError::InvalidSpec(format!(
                "Conv3d out_channels ({out_channels}) must be divisible by groups ({})",
                self.groups
            )));
        }
        if let Some(bias) = &self.bias {
            if bias.len() != out_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![bias.len()],
                });
            }
        }
        Ok((in_channels, out_channels))
    }

    /// Output channels.
    ///
    /// This legacy infallible accessor returns `0` if a caller has replaced the
    /// public kernel with a value that is not five-dimensional. New fallible
    /// code should use [`Self::try_out_channels`] so malformed public geometry
    /// is reported rather than represented by the impossible zero sentinel.
    pub fn out_channels(&self) -> usize {
        if self.kernel.ndim() == 5 {
            self.kernel.shape()[0]
        } else {
            0
        }
    }

    /// Validated output-channel count.
    ///
    /// # Errors
    ///
    /// Any violation reported by [`Self::validate_geometry`].
    pub fn try_out_channels(&self) -> Result<usize> {
        self.validate_geometry()
            .map(|(_, out_channels)| out_channels)
    }

    /// Total input channels (`kernel.shape()[1] * groups`).
    ///
    /// Legacy infallible accessor: returns `0` for malformed or overflowing
    /// public geometry. New fallible code should use [`Self::try_in_channels`].
    pub fn in_channels(&self) -> usize {
        if self.kernel.ndim() == 5 {
            self.kernel.shape()[1].checked_mul(self.groups).unwrap_or(0)
        } else {
            0
        }
    }

    /// Validated total input-channel count.
    ///
    /// # Errors
    ///
    /// Any violation reported by [`Self::validate_geometry`].
    pub fn try_in_channels(&self) -> Result<usize> {
        self.validate_geometry().map(|(in_channels, _)| in_channels)
    }

    /// Kernel extents `[kD, kH, kW]`.
    ///
    /// Legacy infallible accessor: returns `[0, 0, 0]` for a malformed public
    /// kernel. New fallible code should use [`Self::try_kernel_size`].
    pub fn kernel_size(&self) -> [usize; 3] {
        if self.kernel.ndim() == 5 {
            let shape = self.kernel.shape();
            [shape[2], shape[3], shape[4]]
        } else {
            [0, 0, 0]
        }
    }

    /// Validated kernel extents `[kD, kH, kW]`.
    ///
    /// # Errors
    ///
    /// Any violation reported by [`Self::validate_geometry`].
    pub fn try_kernel_size(&self) -> Result<[usize; 3]> {
        self.validate_geometry()?;
        Ok(self.kernel_size())
    }

    /// Output spatial dimensions `[D_out, H_out, W_out]` for a given input volume.
    ///
    /// `floor((in + 2*pad - (dilation*(k - 1) + 1)) / stride) + 1` per axis, with
    /// every intermediate checked. Shares
    /// [`checked_conv3d_output_extent`] with the certified kernel so the layer
    /// geometry and the forward can never disagree.
    ///
    /// # Errors
    ///
    /// Invalid geometry, an arithmetic overflow, or a padded input smaller than
    /// the effective (dilated) kernel on any axis.
    pub fn output_size(&self, input_shape: [usize; 3]) -> Result<[usize; 3]> {
        self.validate_geometry()?;
        let kernel = self.kernel_size();
        let axes = ["depth", "height", "width"];
        let mut output = [0usize; 3];
        for axis in 0..3 {
            output[axis] = checked_conv3d_output_extent(
                input_shape[axis],
                kernel[axis],
                self.stride[axis],
                self.padding[axis],
                self.dilation[axis],
                axes[axis],
            )?;
        }
        Ok(output)
    }
}
