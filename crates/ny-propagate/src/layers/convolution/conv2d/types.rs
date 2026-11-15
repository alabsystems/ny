// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{s, Array1, Array2, ArrayD, ArrayView2, IxDyn};
use ny_core::{checked_shape_product, is_crown_coeff_safe, GemmEngine, NyError, Result};
use tracing::debug;

use super::super::crown_helpers::{
    batched_conv_coeff_err, batched_conv_coeff_err_with_poll, compute_conv_bias_rows_f64,
    compute_conv_bias_rows_f64_with_poll,
};
use super::{
    conv2d_forward_backward_coeff_f64, conv2d_forward_batched_gemm, conv2d_single,
    conv2d_transpose_backward_coeff_f64,
    conv2d_transpose_backward_coeff_f64_with_engine_and_deadline,
    conv2d_transpose_batched_gemm_grouped,
};
use crate::{contiguous_flat_slice_mut, BatchedLinearBounds};

const BOUNDED_CROWN_POLL_ELEMENTS: usize = 4_096;

fn copy_array2_with_poll<F>(source: ArrayView2<'_, f32>, poll: &mut F) -> Result<Array2<f32>>
where
    F: FnMut() -> Result<()>,
{
    poll()?;
    let mut owned = Array2::<f32>::zeros(source.raw_dim());
    poll()?;
    for (index, (dst, &src)) in owned.iter_mut().zip(source.iter()).enumerate() {
        if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
            poll()?;
        }
        *dst = src;
    }
    poll()?;
    Ok(owned)
}

fn copy_arrayd_with_poll<F>(source: &ArrayD<f32>, poll: &mut F) -> Result<ArrayD<f32>>
where
    F: FnMut() -> Result<()>,
{
    poll()?;
    let mut owned = ArrayD::<f32>::zeros(source.raw_dim());
    poll()?;
    for (index, (dst, &src)) in owned.iter_mut().zip(source.iter()).enumerate() {
        if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
            poll()?;
        }
        *dst = src;
    }
    poll()?;
    Ok(owned)
}

fn flatten_incoming_coeff_err(
    name: &str,
    err: Option<&ArrayD<f32>>,
    coefficient_shape: &[usize],
    total_rows: usize,
    mid_dim: usize,
) -> Result<Option<Array2<f32>>> {
    let Some(err) = err else {
        return Ok(None);
    };
    if err.shape() != coefficient_shape {
        return Err(NyError::InvalidSpec(format!(
            "Conv2d batched CROWN: {name} shape {:?} must match coefficient shape \
             {coefficient_shape:?}",
            err.shape()
        )));
    }
    err.view()
        .into_shape_with_order((total_rows, mid_dim))
        .map(|view| Some(view.to_owned()))
        .map_err(|_| {
            NyError::InvalidSpec(format!(
                "Conv2d batched CROWN: cannot flatten {name} with shape {:?}",
                err.shape()
            ))
        })
}

fn flatten_incoming_coeff_err_with_poll<F>(
    name: &str,
    err: Option<&ArrayD<f32>>,
    coefficient_shape: &[usize],
    total_rows: usize,
    mid_dim: usize,
    poll: &mut F,
) -> Result<Option<Array2<f32>>>
where
    F: FnMut() -> Result<()>,
{
    let Some(err) = err else {
        poll()?;
        return Ok(None);
    };
    if err.shape() != coefficient_shape {
        return Err(NyError::InvalidSpec(format!(
            "Conv2d batched CROWN: {name} shape {:?} must match coefficient shape \
             {coefficient_shape:?}",
            err.shape()
        )));
    }
    let view = err
        .view()
        .into_shape_with_order((total_rows, mid_dim))
        .map_err(|_| {
            NyError::InvalidSpec(format!(
                "Conv2d batched CROWN: cannot flatten {name} with shape {:?}",
                err.shape()
            ))
        })?;
    copy_array2_with_poll(view, poll).map(Some)
}

/// A 2D convolution layer: `y = conv2d(x, kernel) + bias`.
///
/// # Parameter-mutation contract (#conv2d-parameter-boundary)
///
/// [`Self::replace_parameters`] is the supported way to change `kernel` or
/// `bias` on a constructed layer. It commits through the constructor
/// (`*self = Self::new_dilated(..)`), so anything the constructor derives is
/// rebuilt *with* the parameters instead of surviving them. That is the same
/// boundary [`crate::layers::LinearLayer::replace_parameters`] provides, and it
/// is the reason `LinearLayer` may hold weight-derived caches soundly.
///
/// The contract here is WEAKER than `LinearLayer`'s in one respect that is
/// load-bearing, so do not read across from it. `LinearLayer::weight` is not
/// publicly assignable; `kernel` and `bias` below still are. Any holder of a
/// `&mut Conv2dLayer`, in this crate or another, can replace or edit them in
/// place with no revalidation and no rebuild, and writers of exactly that shape
/// exist in the tree today: a subnormal-weight sanitiser that walks
/// `kernel.iter_mut()` (`ny-cli/tests/flush_charged_metaroom_wallclock.rs`) and
/// several tests that assign a deliberately malformed kernel to prove the
/// consuming code refuses it instead of panicking (this module's `tests.rs`,
/// `ny-cli`'s `cz_metaroom_qualification.rs`). That exposure is why the
/// propagation and output-size entry points revalidate through
/// `validate_geometry` rather than trusting construction.
///
/// The consequence, and the reason this block exists:
///
/// **`Conv2dLayer` must not gain a field derived from `kernel` or `bias`.** A
/// derived field would be built once, would not be reached by a `pub` write,
/// and would then publish a bound computed from parameters the layer no longer
/// holds — a wrong bound, not merely a stale one. Adding one requires FIRST
/// making `kernel`/`bias` non-`pub` and migrating every reader onto
/// [`Self::kernel()`] / [`Self::bias()`], which is what those accessors exist
/// for. `parameter_boundary_tests::conv_structs_carry_no_parameter_derived_field`
/// destructures this struct exhaustively with no `..`, so adding any field
/// stops that module compiling until an author lists the field there — which
/// is where the choice above has to be made deliberately.
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    /// Convolution kernel of shape (out_channels, in_channels/groups, kernel_h, kernel_w)
    ///
    /// Publicly assignable — see the parameter-mutation contract on the struct.
    /// Prefer [`Self::kernel()`] to read and [`Self::replace_parameters`] to
    /// write; a direct write skips every constructor check.
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    ///
    /// Publicly assignable on the same terms as `kernel`. Prefer
    /// [`Self::bias()`] and [`Self::replace_parameters`]; a direct write can
    /// leave `bias.len() != out_channels`, which only the entry-point
    /// `validate_geometry` call then catches.
    pub bias: Option<Array1<f32>>,
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width), spacing between kernel elements. Default (1, 1).
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `dilations` attribute.
    pub dilation: (usize, usize),
    /// Groups (number of blocked connections from input to output channels). Default 1.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `group` attribute.
    pub groups: usize,
    /// Input spatial dimensions (height, width) - required for CROWN backward pass
    pub input_shape: Option<(usize, usize)>,
}

impl Conv2dLayer {
    /// Create a new Conv2d layer with groups=1.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, 1)
    }

    /// Create a new Conv2d layer with explicit groups and dilation=(1, 1).
    ///
    /// Kernel shape: `(out_channels, in_channels/groups, kernel_h, kernel_w)`.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `group` attribute.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        Self::new_dilated(kernel, bias, stride, padding, (1, 1), groups)
    }

    /// Create a new Conv2d layer with explicit dilation and groups.
    ///
    /// Kernel shape: `(out_channels, in_channels/groups, kernel_h, kernel_w)`.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `dilations`/`group` attributes.
    pub fn new_dilated(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d stride must be >= 1 in both dimensions, got stride=({},{})",
                stride.0, stride.1
            )));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d dilation must be >= 1 in both dimensions, got dilation=({},{})",
                dilation.0, dilation.1
            )));
        }
        if groups == 0 {
            return Err(NyError::InvalidSpec(
                "Conv2d groups must be >= 1, got 0".to_string(),
            ));
        }
        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0], // 4D expected
                got: kernel.shape().to_vec(),
            });
        }
        let out_channels = kernel.shape()[0];
        let input_channels_per_group = kernel.shape()[1];
        let kernel_h = kernel.shape()[2];
        let kernel_w = kernel.shape()[3];
        if out_channels == 0 || input_channels_per_group == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d kernel dimensions must be nonzero, got {:?}",
                kernel.shape()
            )));
        }
        input_channels_per_group
            .checked_mul(groups)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Conv2d total input channels overflow: {input_channels_per_group} * {groups}"
                ))
            })?;
        if !out_channels.is_multiple_of(groups) {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d out_channels ({out_channels}) must be divisible by groups ({groups})"
            )));
        }
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_shape: None,
        })
    }

    /// Revalidate geometry that is stored in publicly mutable fields.
    ///
    /// Constructors establish these invariants initially, but callers can mutate
    /// the public compatibility fields afterwards. Every propagation/output-size
    /// entry point calls this before indexing shapes or performing arithmetic.
    pub(crate) fn validate_geometry(&self) -> Result<(usize, usize)> {
        if self.kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: self.kernel.shape().to_vec(),
            });
        }
        if self.stride.0 == 0
            || self.stride.1 == 0
            || self.dilation.0 == 0
            || self.dilation.1 == 0
            || self.groups == 0
        {
            return Err(NyError::InvalidSpec(
                "Conv2d geometry requires nonzero stride, dilation, and groups".to_string(),
            ));
        }
        let out_channels = self.kernel.shape()[0];
        let input_channels_per_group = self.kernel.shape()[1];
        let kernel_h = self.kernel.shape()[2];
        let kernel_w = self.kernel.shape()[3];
        if out_channels == 0 || input_channels_per_group == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d kernel dimensions must be nonzero, got {:?}",
                self.kernel.shape()
            )));
        }
        let in_channels = input_channels_per_group
            .checked_mul(self.groups)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Conv2d total input channels overflow: {input_channels_per_group} * {}",
                    self.groups
                ))
            })?;
        if !out_channels.is_multiple_of(self.groups) {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d out_channels ({out_channels}) must be divisible by groups ({})",
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

    /// Create a new Conv2d layer with known input spatial dimensions.
    /// Required for CROWN backward propagation.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Create a new Conv2d layer with groups and known input spatial dimensions.
    pub fn with_input_shape_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new_full(kernel, bias, stride, padding, groups)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Set the input spatial dimensions. Required for CROWN backward propagation.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_shape = Some((height, width));
    }

    /// Immutable convolution kernel, shape
    /// `(out_channels, in_channels/groups, kernel_h, kernel_w)`.
    ///
    /// The read half of the parameter-mutation contract on [`Conv2dLayer`].
    /// Prefer this over the `kernel` field: it is the migration target if the
    /// field is ever made non-`pub`, so a reader written against it does not
    /// have to change.
    pub fn kernel(&self) -> &ArrayD<f32> {
        &self.kernel
    }

    /// Immutable optional bias, shape `(out_channels,)`.
    ///
    /// Same role as [`Self::kernel()`]; prefer it over the `bias` field.
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.bias.as_ref()
    }

    /// Replace both parameter tensors, rebuilding the layer through the
    /// constructor.
    ///
    /// This is the supported mutation boundary for a constructed layer, and the
    /// counterpart of `LinearLayer::replace_parameters`. It commits via
    /// `*self = Self::new_dilated(..)`, so the replacement is a value the
    /// constructor produced: whatever a future revision of that constructor
    /// derives from the parameters is rebuilt here, and cannot be left over
    /// from the previous ones. `input_shape` is not a parameter and is carried
    /// across (`new_dilated` leaves it `None`); `stride`, `padding`,
    /// `dilation`, and `groups` are likewise reused from `self`.
    ///
    /// Because it goes through the constructor it revalidates the WHOLE
    /// geometry, not just the new tensors — so it also refuses on a layer whose
    /// public geometry fields were already corrupted by a direct write. That is
    /// deliberate: refusing is the safe direction.
    ///
    /// # Errors
    ///
    /// Returns the constructor's error if the kernel is not 4-D, has a zero
    /// dimension, if `bias.len() != out_channels`, or if `self`'s existing
    /// `stride`/`dilation`/`groups` no longer satisfy the constructor's
    /// invariants. On any error `self` is left **unchanged**.
    pub fn replace_parameters(
        &mut self,
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<()> {
        let mut replacement = Self::new_dilated(
            kernel,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        replacement.input_shape = self.input_shape;
        *self = replacement;
        Ok(())
    }

    /// Replace the kernel, retaining the existing bias, via
    /// [`Self::replace_parameters`].
    ///
    /// # Errors
    ///
    /// Returns a shape error if the retained bias does not match the new
    /// `out_channels`, or on any other constructor violation. `self` is left
    /// unchanged.
    pub fn set_kernel(&mut self, kernel: ArrayD<f32>) -> Result<()> {
        self.replace_parameters(kernel, self.bias.clone())
    }

    /// Replace the bias, retaining the existing kernel, via
    /// [`Self::replace_parameters`].
    ///
    /// # Errors
    ///
    /// Returns a shape error if the new bias does not match `out_channels`, or
    /// on any other constructor violation. `self` is left unchanged.
    pub fn set_bias(&mut self, bias: Option<Array1<f32>>) -> Result<()> {
        self.replace_parameters(self.kernel.clone(), bias)
    }

    /// Output channels.
    ///
    /// This legacy infallible accessor returns `0` if callers have replaced the
    /// public kernel with a value that is not four-dimensional. New fallible
    /// code should use [`Self::try_out_channels`] so malformed public geometry is
    /// reported rather than represented by the impossible zero-channel sentinel.
    pub fn out_channels(&self) -> usize {
        if self.kernel.ndim() == 4 {
            self.kernel.shape()[0]
        } else {
            0
        }
    }

    /// Validated output-channel count.
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field violates the
    /// invariants established by the constructors.
    pub fn try_out_channels(&self) -> Result<usize> {
        self.validate_geometry()
            .map(|(_, out_channels)| out_channels)
    }

    /// Input channels (total, accounting for groups).
    ///
    /// Kernel shape is `(out_c, in_c/groups, kh, kw)`, so total input channels
    /// is `kernel.shape()[1] * groups`.
    ///
    /// This legacy infallible accessor returns `0` for malformed/overflowing
    /// public geometry. New fallible code should use [`Self::try_in_channels`].
    pub fn in_channels(&self) -> usize {
        if self.kernel.ndim() == 4 {
            self.kernel.shape()[1].checked_mul(self.groups).unwrap_or(0)
        } else {
            0
        }
    }

    /// Validated total input-channel count.
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field is invalid or
    /// if the grouped input-channel count overflows.
    pub fn try_in_channels(&self) -> Result<usize> {
        self.validate_geometry().map(|(in_channels, _)| in_channels)
    }

    /// Kernel size (height, width).
    ///
    /// This legacy infallible accessor returns `(0, 0)` for a malformed public
    /// kernel. New fallible code should use [`Self::try_kernel_size`].
    pub fn kernel_size(&self) -> (usize, usize) {
        if self.kernel.ndim() == 4 {
            (self.kernel.shape()[2], self.kernel.shape()[3])
        } else {
            (0, 0)
        }
    }

    /// Validated kernel size (height, width).
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field violates the
    /// invariants established by the constructors.
    pub fn try_kernel_size(&self) -> Result<(usize, usize)> {
        self.validate_geometry()?;
        Ok((self.kernel.shape()[2], self.kernel.shape()[3]))
    }

    /// Compute output spatial dimensions.
    ///
    /// Returns an error if the padded input is smaller than the kernel.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        self.validate_geometry()?;
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
        let eff_kh = kh
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(dh))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("Conv2d effective kernel height overflow".to_string())
            })?;
        let eff_kw = kw
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(dw))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("Conv2d effective kernel width overflow".to_string())
            })?;
        let padded_h = ph
            .checked_mul(2)
            .and_then(|padding| input_h.checked_add(padding))
            .ok_or_else(|| NyError::InvalidSpec("Conv2d padded height overflow".to_string()))?;
        let padded_w = pw
            .checked_mul(2)
            .and_then(|padding| input_w.checked_add(padding))
            .ok_or_else(|| NyError::InvalidSpec("Conv2d padded width overflow".to_string()))?;
        if padded_h < eff_kh || padded_w < eff_kw {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d effective kernel larger than padded input: input=({input_h},{input_w}), \
                 padding=({ph},{pw}), kernel=({kh},{kw}), dilation=({dh},{dw})"
            )));
        }
        Ok(((padded_h - eff_kh) / sh + 1, (padded_w - eff_kw) / sw + 1))
    }

    /// Batched CROWN backward propagation through Conv2d layer.
    ///
    /// For a conv layer y = conv2d(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// BatchedLinearBounds:
    /// - lower_a shape: [...batch, out_dim, out_c * out_h * out_w]
    /// - Reshapes to [...batch, out_dim, out_c, out_h, out_w] for conv_transpose
    /// - Output: [...batch, out_dim, in_c * in_h * in_w]
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        debug!("Conv2d layer batched CROWN backward propagation");
        let (in_c, out_c) = self.validate_geometry()?;

        // Get input spatial dimensions (required for CROWN)
        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Conv2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        let conv_out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: batch dims product overflows: {:?}",
                batch_dims
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_h * in_w]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Flatten (total_batch, out_dim, mid_dim) → (total_batch * out_dim, mid_dim)
        // for batched GEMM. Same approach as conv2d/bound.rs propagate_linear.
        let total_rows = total_batch.checked_mul(out_dim).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: total_batch * out_dim overflows: {total_batch} * {out_dim}"
            ))
        })?;
        let lower_a_flat = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for GEMM".to_string()))?;
        let upper_a_flat = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for GEMM".to_string()))?;
        let bounded_engine = super::ops_transpose_gemm::bounded_pollable_host_engine(engine)?;
        let mut poll_bounded = || -> Result<()> {
            if let Some(engine) = bounded_engine {
                engine.poll_crown_backward_deadline()?;
            }
            Ok(())
        };
        let (kh_e, kw_e) = self.kernel_size();
        let n_contraction = out_c.saturating_mul(kh_e).saturating_mul(kw_e);
        let want_recompute =
            crate::layers::convolution::crown_helpers::conv_should_f64_recompute(n_contraction);

        // Per-group batched GEMM replaces N per-row conv2d_transpose calls.
        // Engine parameter enables GPU acceleration via GemmEngine.
        // Groups parameter enables depthwise/grouped convolutions.
        let lower_a_2d = if bounded_engine.is_some() {
            copy_array2_with_poll(lower_a_flat, &mut poll_bounded)?
        } else {
            lower_a_flat.to_owned()
        };
        let upper_a_2d = if bounded_engine.is_some() {
            copy_array2_with_poll(upper_a_flat, &mut poll_bounded)?
        } else {
            upper_a_flat.to_owned()
        };
        let (mut new_lower_a, mut new_upper_a) = if bounded_engine.is_some() {
            if !want_recompute {
                return Err(NyError::UnsupportedOp(
                    "bounded Conv2d batched CROWN requires the certified f64 coefficient route"
                        .into(),
                ));
            }
            // The legacy f32 pair is dead whenever the certified f64 recompute
            // is active. More importantly, its generic engine/fallback path is
            // not governed by the bounded host authority.
            poll_bounded()?;
            super::ops_transpose_gemm::guard_conv_crown_backward_buffer(
                lower_a_2d.nrows().max(upper_a_2d.nrows()),
                conv_in_size,
                2,
            )?;
            let pair = (
                Array2::<f32>::zeros((lower_a_2d.nrows(), conv_in_size)),
                Array2::<f32>::zeros((upper_a_2d.nrows(), conv_in_size)),
            );
            poll_bounded()?;
            pair
        } else {
            (
                conv2d_transpose_batched_gemm_grouped(
                    &lower_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // new_lower_a remains live while new_upper_a is built
                    engine,
                )?,
                conv2d_transpose_batched_gemm_grouped(
                    &upper_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // both f32 result buffers are retained by the caller
                    engine,
                )?,
            )
        };

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug on
        // the BATCHED β-CROWN/BaB verdict path). This path historically used the
        // f32 GEMM coefficient with NO certified error, yet Conv2d is declared
        // `propagates_coeff_err = true` (the dispatcher trusts it to carry the
        // error) — an UNSOUND false-proof risk identical to the scalar path. Fix
        // it the same way: f64-recompute the SAME contraction, store the directed
        // f32, and attach `cast_err + γ_n^f64·S + prop` (built below). The flattened
        // (total_rows, mid_dim) layout matches `lower_a_2d`/`upper_a_2d`.
        // Small-contraction fast path: skip the f64 recompute and certify the f32
        // GEMM coefficient with `γ_n^f32·S` (sound; tight for small n) — see
        // crown_helpers::conv_should_f64_recompute.
        let recompute = |coefficients: &Array2<f32>| -> Result<Option<Array2<f64>>> {
            if !want_recompute {
                return Ok(None);
            }
            let result = if let Some(bounded_engine) = bounded_engine {
                conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
                    coefficients,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2,
                    Some(bounded_engine),
                    None,
                )
            } else {
                conv2d_transpose_backward_coeff_f64(
                    coefficients,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2,
                )
            };
            match result {
                Ok(coefficients) => Ok(Some(coefficients)),
                Err(error) if bounded_engine.is_some() => Err(error),
                Err(_) => Ok(None),
            }
        };
        let coeff_f64 = recompute(&lower_a_2d)?;
        let coeff_f64_u = recompute(&upper_a_2d)?;
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_lower_a.raw_dim());
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_upper_a.raw_dim());
        // Degrade ONLY when a recompute was WANTED (wide conv) but failed — the f32
        // coefficient cannot then be soundly certified with the f64 gamma. On the
        // small-n fast path (`!want_recompute`) the f32 coefficient is kept and
        // certified with `γ_n^f32` (sound), so no degrade.
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                if bounded_engine.is_some() {
                    poll_bounded()?;
                    for (index, (dst, &src)) in new_lower_a.iter_mut().zip(c64.iter()).enumerate() {
                        if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                            poll_bounded()?;
                        }
                        *dst = src as f32;
                    }
                    poll_bounded()?;
                } else {
                    for i in 0..new_lower_a.nrows() {
                        for p in 0..new_lower_a.ncols() {
                            new_lower_a[[i, p]] = c64[[i, p]] as f32;
                        }
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                if bounded_engine.is_some() {
                    poll_bounded()?;
                    for (index, (dst, &src)) in new_upper_a.iter_mut().zip(c64.iter()).enumerate() {
                        if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                            poll_bounded()?;
                        }
                        *dst = src as f32;
                    }
                    poll_bounded()?;
                } else {
                    for i in 0..new_upper_a.nrows() {
                        for p in 0..new_upper_a.ncols() {
                            new_upper_a[[i, p]] = c64[[i, p]] as f32;
                        }
                    }
                }
            }
        }
        // Certified error matrices (`cast + γ·S + prop`, flattened), reshaped to
        // out_a_shape below.
        let kernel_l1: f64 = if bounded_engine.is_some() {
            poll_bounded()?;
            let mut total = 0.0;
            for (index, &value) in self.kernel.iter().enumerate() {
                if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                    poll_bounded()?;
                }
                total += (value as f64).abs();
            }
            poll_bounded()?;
            total
        } else {
            self.kernel.iter().map(|&v| (v as f64).abs()).sum()
        };
        // Flatten any incoming error to (total_rows, mid_dim) for propagation.
        let in_lower_err_2d = if bounded_engine.is_some() {
            flatten_incoming_coeff_err_with_poll(
                "lower_a_err",
                bounds.lower_a_err.as_ref(),
                bounds.lower_a.shape(),
                total_rows,
                mid_dim,
                &mut poll_bounded,
            )?
        } else {
            flatten_incoming_coeff_err(
                "lower_a_err",
                bounds.lower_a_err.as_ref(),
                bounds.lower_a.shape(),
                total_rows,
                mid_dim,
            )?
        };
        let in_upper_err_2d = if bounded_engine.is_some() {
            flatten_incoming_coeff_err_with_poll(
                "upper_a_err",
                bounds.upper_a_err.as_ref(),
                bounds.upper_a.shape(),
                total_rows,
                mid_dim,
                &mut poll_bounded,
            )?
        } else {
            flatten_incoming_coeff_err(
                "upper_a_err",
                bounds.upper_a_err.as_ref(),
                bounds.upper_a.shape(),
                total_rows,
                mid_dim,
            )?
        };
        let mut lower_err_2d = if bounded_engine.is_some() {
            batched_conv_coeff_err_with_poll(
                &lower_a_2d,
                in_lower_err_2d.as_ref(),
                &new_lower_a,
                coeff_f64.as_ref().filter(|_| lower_recompute_ok),
                kernel_l1,
                n_contraction,
                None,
                None,
                &mut poll_bounded,
            )?
        } else {
            batched_conv_coeff_err(
                &lower_a_2d,
                in_lower_err_2d.as_ref(),
                &new_lower_a,
                coeff_f64.as_ref().filter(|_| lower_recompute_ok),
                kernel_l1,
                n_contraction,
                None,
                None,
            )
        };
        let mut upper_err_2d = if bounded_engine.is_some() {
            batched_conv_coeff_err_with_poll(
                &upper_a_2d,
                in_upper_err_2d.as_ref(),
                &new_upper_a,
                coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
                kernel_l1,
                n_contraction,
                None,
                None,
                &mut poll_bounded,
            )?
        } else {
            batched_conv_coeff_err(
                &upper_a_2d,
                in_upper_err_2d.as_ref(),
                &new_upper_a,
                coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
                kernel_l1,
                n_contraction,
                None,
                None,
            )
        };
        // A WANTED-but-failed recompute degrades to ±inf bias (the f32 coefficient
        // cannot be soundly certified with the f64 gamma in that case).
        if lower_recompute_failed {
            if bounded_engine.is_some() {
                for (index, (coefficient, error)) in new_lower_a
                    .iter_mut()
                    .zip(lower_err_2d.iter_mut())
                    .enumerate()
                {
                    if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                        poll_bounded()?;
                    }
                    *coefficient = 0.0;
                    *error = 0.0;
                }
                poll_bounded()?;
            } else {
                new_lower_a.fill(0.0);
                lower_err_2d.fill(0.0);
            }
        }
        if upper_recompute_failed {
            if bounded_engine.is_some() {
                for (index, (coefficient, error)) in new_upper_a
                    .iter_mut()
                    .zip(upper_err_2d.iter_mut())
                    .enumerate()
                {
                    if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                        poll_bounded()?;
                    }
                    *coefficient = 0.0;
                    *error = 0.0;
                }
                poll_bounded()?;
            } else {
                new_upper_a.fill(0.0);
                upper_err_2d.fill(0.0);
            }
        }

        // Track which output rows have unsafe coefficients.
        // Scan GEMM output for non-finite or near-overflow values (same pattern as bound.rs).
        // Rows whose f64 recompute failed are forced unsafe so they degrade to the
        // maximally-loose ±inf bias (the f32-GEMM coefficient cannot be soundly
        // certified with the f64 gamma).
        let mut lower_nonfinite_rows = vec![lower_recompute_failed; total_rows];
        let mut upper_nonfinite_rows = vec![upper_recompute_failed; total_rows];
        if bounded_engine.is_some() {
            let mut scan_work = 0usize;
            poll_bounded()?;
            for row_idx in 0..total_rows {
                for &value in new_lower_a.row(row_idx) {
                    if !is_crown_coeff_safe(value) {
                        lower_nonfinite_rows[row_idx] = true;
                        break;
                    }
                    scan_work += 1;
                    if scan_work >= BOUNDED_CROWN_POLL_ELEMENTS {
                        scan_work = 0;
                        poll_bounded()?;
                    }
                }
                for &value in new_upper_a.row(row_idx) {
                    if !is_crown_coeff_safe(value) {
                        upper_nonfinite_rows[row_idx] = true;
                        break;
                    }
                    scan_work += 1;
                    if scan_work >= BOUNDED_CROWN_POLL_ELEMENTS {
                        scan_work = 0;
                        poll_bounded()?;
                    }
                }
            }
            poll_bounded()?;
        } else {
            for row_idx in 0..total_rows {
                if new_lower_a
                    .row(row_idx)
                    .iter()
                    .any(|v| !is_crown_coeff_safe(*v))
                {
                    lower_nonfinite_rows[row_idx] = true;
                }
                if new_upper_a
                    .row(row_idx)
                    .iter()
                    .any(|v| !is_crown_coeff_safe(*v))
                {
                    upper_nonfinite_rows[row_idx] = true;
                }
            }
        }

        // For rows with non-finite A-matrix coefficients, zero the entire row.
        // Bias override happens after bias computation below.
        // Reference: conv2d/bound.rs:328-355 (scalar path).
        let (lower_affected, upper_affected) = if bounded_engine.is_some() {
            let mut lower = 0usize;
            let mut upper = 0usize;
            for (index, (&lower_bad, &upper_bad)) in lower_nonfinite_rows
                .iter()
                .zip(upper_nonfinite_rows.iter())
                .enumerate()
            {
                if index.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                    poll_bounded()?;
                }
                lower += usize::from(lower_bad);
                upper += usize::from(upper_bad);
            }
            poll_bounded()?;
            (lower, upper)
        } else {
            (
                lower_nonfinite_rows.iter().filter(|&&r| r).count(),
                upper_nonfinite_rows.iter().filter(|&&r| r).count(),
            )
        };
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "Conv2d batched CROWN backward: non-finite A-matrix overflow in {}/{} lower rows, \
                 {}/{} upper rows — falling back to ±inf bias for affected rows",
                lower_affected, total_rows, upper_affected, total_rows
            );
            let mut zero_work = 0usize;
            for i in 0..total_rows {
                if lower_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_lower_a[[i, j]] = 0.0;
                        lower_err_2d[[i, j]] = 0.0;
                        if bounded_engine.is_some() {
                            zero_work += 1;
                            if zero_work >= BOUNDED_CROWN_POLL_ELEMENTS {
                                zero_work = 0;
                                poll_bounded()?;
                            }
                        }
                    }
                }
                if upper_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_upper_a[[i, j]] = 0.0;
                        upper_err_2d[[i, j]] = 0.0;
                        if bounded_engine.is_some() {
                            zero_work += 1;
                            if zero_work >= BOUNDED_CROWN_POLL_ELEMENTS {
                                zero_work = 0;
                                poll_bounded()?;
                            }
                        }
                    }
                }
            }
            if bounded_engine.is_some() {
                poll_bounded()?;
            }
        }

        // Reshape back to [...batch, out_dim, in_c * in_h * in_w]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
        // Reshape the certified error to the same out_a_shape for attachment.
        let (lower_err_vec, _) = lower_err_2d.into_raw_vec_and_offset();
        let (upper_err_vec, _) = upper_err_2d.into_raw_vec_and_offset();
        let new_lower_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), lower_err_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_err".to_string()))?;
        let new_upper_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), upper_err_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_err".to_string()))?;

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            let lower_b_1d = bounds
                .lower_b
                .view()
                .into_shape_with_order(total_rows)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_1d = bounds
                .upper_b
                .view()
                .into_shape_with_order(total_rows)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let spatial_size = out_h
                .checked_mul(out_w)
                .ok_or_else(|| NyError::InvalidSpec("Conv2d spatial size overflows".into()))?;
            let (mut new_lower_b, mut new_upper_b) = if bounded_engine.is_some() {
                compute_conv_bias_rows_f64_with_poll(
                    lower_a_2d.view(),
                    in_lower_err_2d.as_ref().map(|err| err.view()),
                    lower_b_1d,
                    upper_a_2d.view(),
                    in_upper_err_2d.as_ref().map(|err| err.view()),
                    upper_b_1d,
                    bias,
                    out_c,
                    spatial_size,
                    &mut poll_bounded,
                )?
            } else {
                compute_conv_bias_rows_f64(
                    lower_a_2d.view(),
                    in_lower_err_2d.as_ref().map(|err| err.view()),
                    lower_b_1d,
                    upper_a_2d.view(),
                    in_upper_err_2d.as_ref().map(|err| err.view()),
                    upper_b_1d,
                    bias,
                    out_c,
                    spatial_size,
                )?
            };

            // Override bias for non-finite A-matrix rows.
            // Setting bias to ±inf with zeroed A-row produces sound [-inf,+inf] bounds.
            for row_idx in 0..total_rows {
                if bounded_engine.is_some() && row_idx.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                    poll_bounded()?;
                }
                if lower_nonfinite_rows[row_idx] {
                    new_lower_b[row_idx] = f32::NEG_INFINITY;
                }
                if upper_nonfinite_rows[row_idx] {
                    new_upper_b[row_idx] = f32::INFINITY;
                }
            }
            if bounded_engine.is_some() {
                poll_bounded()?;
            }

            let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            )
        } else {
            // Even without conv bias, override for non-finite A-matrix rows.
            let mut lb = if bounded_engine.is_some() {
                copy_arrayd_with_poll(&bounds.lower_b, &mut poll_bounded)?
            } else {
                bounds.lower_b.clone()
            };
            let mut ub = if bounded_engine.is_some() {
                copy_arrayd_with_poll(&bounds.upper_b, &mut poll_bounded)?
            } else {
                bounds.upper_b.clone()
            };
            if lower_affected > 0 || upper_affected > 0 {
                let lb_flat = contiguous_flat_slice_mut(&mut lb)?;
                let ub_flat = contiguous_flat_slice_mut(&mut ub)?;
                for i in 0..total_rows {
                    if bounded_engine.is_some() && i.is_multiple_of(BOUNDED_CROWN_POLL_ELEMENTS) {
                        poll_bounded()?;
                    }
                    if lower_nonfinite_rows[i] {
                        lb_flat[i] = f32::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[i] {
                        ub_flat[i] = f32::INFINITY;
                    }
                }
            }
            if bounded_engine.is_some() {
                poll_bounded()?;
            }
            (lb, ub)
        };

        // Update input shape to reflect the conv layer's input dimensions
        let new_input_shape = if bounds.input_shape.is_empty() {
            vec![conv_in_size]
        } else if bounds.input_shape.len() >= 3 {
            // Update last three dims from [out_c, out_h, out_w] to [in_c, in_h, in_w].
            let mut shape = bounds.input_shape.clone();
            let len = shape.len();
            shape[len - 3] = in_c;
            shape[len - 2] = in_h;
            shape[len - 1] = in_w;
            shape
        } else {
            // Preserve batch dims while updating flattened feature size.
            let mut shape = bounds.input_shape[..bounds.input_shape.len() - 1].to_vec();
            shape.push(conv_in_size);
            shape
        };

        // CROWN backward NaN firewall: conservative fallback instead of hard error.
        poll_bounded()?;
        let out = if bounded_engine.is_some() {
            // The pollable scans above already applied the per-row conservative
            // firewall and established every shape. Avoid the generic
            // constructor's duplicate unpollable full-array validation scan.
            BatchedLinearBounds {
                lower_a: new_lower_a,
                lower_b: new_lower_b,
                upper_a: new_upper_a,
                upper_b: new_upper_b,
                input_shape: new_input_shape,
                output_shape: bounds.output_shape.clone(),
                lower_a_err: Some(new_lower_a_err),
                upper_a_err: Some(new_upper_a_err),
            }
        } else {
            let mut out = BatchedLinearBounds::new_or_conservative(
                new_lower_a,
                new_lower_b,
                new_upper_a,
                new_upper_b,
                new_input_shape,
                bounds.output_shape.clone(),
            )?;
            // Attach the certified coefficient error (#vnncomp-aw-soundness). Conv2d is
            // `propagates_coeff_err = true`, so concretize_sound MUST see this penalty.
            out.set_coeff_err(new_lower_a_err, new_upper_a_err);
            out
        };
        // Refuse a stale relation immediately before publication.
        poll_bounded()?;
        Ok(out)
    }
}

/// A 2D transposed convolution layer: `y = conv_transpose2d(x, kernel) + bias`.
///
/// # Parameter-mutation contract (#conv2d-parameter-boundary)
///
/// Identical in every respect to the contract documented on [`Conv2dLayer`],
/// including the part that is weaker than `LinearLayer`'s: `kernel` and `bias`
/// are `pub` and directly assignable, so **this struct must not gain a field
/// derived from them** either. [`Self::replace_parameters`] is the supported
/// mutation boundary and commits through `Self::new_full`. Read the block on
/// [`Conv2dLayer`] before adding any field here.
#[derive(Debug, Clone)]
pub struct ConvTranspose2dLayer {
    /// Transposed convolution kernel of shape (in_channels, out_channels, kernel_h, kernel_w)
    ///
    /// Publicly assignable — see the parameter-mutation contract on the struct.
    /// Prefer [`Self::kernel()`] to read and [`Self::replace_parameters`] to
    /// write; a direct write skips every constructor check.
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    ///
    /// Publicly assignable on the same terms as `kernel`. Prefer
    /// [`Self::bias()`] and [`Self::replace_parameters`].
    pub bias: Option<Array1<f32>>,
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width), spacing between kernel elements. Default (1, 1).
    /// Reference: PyTorch `torch.nn.ConvTranspose2d`, ONNX ConvTranspose `dilations`.
    pub dilation: (usize, usize),
    /// Output padding (height, width): extra size added to the high side of each
    /// output spatial dimension. Default (0, 0). This disambiguates output
    /// shape; it does not itself add zeros. A newly exposed high-edge position
    /// can still receive a kernel contribution when ordinary padding makes an
    /// input/kernel pair land there. Reference: ONNX ConvTranspose
    /// `output_padding`, PyTorch `torch.nn.ConvTranspose2d`.
    pub output_padding: (usize, usize),
    /// Input spatial dimensions (height, width) - required for CROWN backward pass
    pub input_shape: Option<(usize, usize)>,
}

impl ConvTranspose2dLayer {
    /// Create a new ConvTranspose2d layer with dilation=(1, 1), output_padding=(0, 0).
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, (1, 1), (0, 0))
    }

    /// Create a new ConvTranspose2d layer with explicit dilation and output_padding.
    ///
    /// `output_padding` must be strictly less than `stride` in each dimension.
    /// This matches the PyTorch/ONNX uniqueness constraint and, critically,
    /// guarantees that the CROWN backward pass (a regular strided convolution
    /// over the padded output grid) recovers the exact input size:
    /// `floor(output_padding/stride) == 0` keeps the recovered input dimension
    /// equal to the true input dimension. It does not imply that every exposed
    /// high-edge cell is bias-only. Reference: ONNX ConvTranspose.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Self> {
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d stride must be >= 1 in both dimensions, got stride=({},{})",
                stride.0, stride.1
            )));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d dilation must be >= 1 in both dimensions, got dilation=({},{})",
                dilation.0, dilation.1
            )));
        }
        // Require output_padding < stride per dimension (see doc comment): this
        // is what keeps the CROWN backward conv's recovered input size exact.
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d output_padding ({},{}) must be < stride ({},{}) per dimension \
                 for sound bound propagation",
                output_padding.0, output_padding.1, stride.0, stride.1
            )));
        }
        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }
        let in_channels = kernel.shape()[0];
        let out_channels = kernel.shape()[1];
        let kernel_h = kernel.shape()[2];
        let kernel_w = kernel.shape()[3];
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d kernel dimensions must be nonzero, got {:?}",
                kernel.shape()
            )));
        }
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            input_shape: None,
        })
    }

    /// Revalidate geometry stored in publicly mutable fields.
    pub(crate) fn validate_geometry(&self) -> Result<(usize, usize)> {
        if self.kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: self.kernel.shape().to_vec(),
            });
        }
        if self.stride.0 == 0 || self.stride.1 == 0 || self.dilation.0 == 0 || self.dilation.1 == 0
        {
            return Err(NyError::InvalidSpec(
                "ConvTranspose2d geometry requires nonzero stride and dilation".to_string(),
            ));
        }
        if self.output_padding.0 >= self.stride.0 || self.output_padding.1 >= self.stride.1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d output_padding ({},{}) must be < stride ({},{}) per dimension \
                 for sound bound propagation",
                self.output_padding.0, self.output_padding.1, self.stride.0, self.stride.1
            )));
        }
        let in_channels = self.kernel.shape()[0];
        let out_channels = self.kernel.shape()[1];
        let kernel_h = self.kernel.shape()[2];
        let kernel_w = self.kernel.shape()[3];
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d kernel dimensions must be nonzero, got {:?}",
                self.kernel.shape()
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

    /// Create a new ConvTranspose2d layer with known input spatial dimensions.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Set the input spatial dimensions. Required for CROWN backward propagation.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_shape = Some((height, width));
    }

    /// Immutable transposed-convolution kernel, shape
    /// `(in_channels, out_channels, kernel_h, kernel_w)`.
    ///
    /// The read half of the parameter-mutation contract; see [`Conv2dLayer`].
    pub fn kernel(&self) -> &ArrayD<f32> {
        &self.kernel
    }

    /// Immutable optional bias, shape `(out_channels,)`.
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.bias.as_ref()
    }

    /// Replace both parameter tensors, rebuilding the layer through the
    /// constructor.
    ///
    /// The [`Conv2dLayer::replace_parameters`] contract, applied to
    /// `Self::new_full`: `stride`, `padding`, `dilation`, and `output_padding`
    /// are reused from `self`, `input_shape` is carried across, and the whole
    /// geometry is revalidated.
    ///
    /// # Errors
    ///
    /// Returns the constructor's error if the kernel is not 4-D, has a zero
    /// dimension, if `bias.len() != out_channels`, or if `self`'s existing
    /// `stride`/`dilation`/`output_padding` no longer satisfy the constructor's
    /// invariants. On any error `self` is left **unchanged**.
    pub fn replace_parameters(
        &mut self,
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<()> {
        let mut replacement = Self::new_full(
            kernel,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        replacement.input_shape = self.input_shape;
        *self = replacement;
        Ok(())
    }

    /// Replace the kernel, retaining the existing bias, via
    /// [`Self::replace_parameters`].
    ///
    /// # Errors
    ///
    /// Returns a shape error if the retained bias does not match the new
    /// `out_channels`, or on any other constructor violation. `self` is left
    /// unchanged.
    pub fn set_kernel(&mut self, kernel: ArrayD<f32>) -> Result<()> {
        self.replace_parameters(kernel, self.bias.clone())
    }

    /// Replace the bias, retaining the existing kernel, via
    /// [`Self::replace_parameters`].
    ///
    /// # Errors
    ///
    /// Returns a shape error if the new bias does not match `out_channels`, or
    /// on any other constructor violation. `self` is left unchanged.
    pub fn set_bias(&mut self, bias: Option<Array1<f32>>) -> Result<()> {
        self.replace_parameters(self.kernel.clone(), bias)
    }

    /// Output channels.
    ///
    /// This legacy infallible accessor returns `0` if callers have replaced the
    /// public kernel with a value that is not four-dimensional. New fallible
    /// code should use [`Self::try_out_channels`].
    pub fn out_channels(&self) -> usize {
        if self.kernel.ndim() == 4 {
            self.kernel.shape()[1]
        } else {
            0
        }
    }

    /// Validated output-channel count.
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field violates the
    /// invariants established by the constructors.
    pub fn try_out_channels(&self) -> Result<usize> {
        self.validate_geometry()
            .map(|(_, out_channels)| out_channels)
    }

    /// Input channels.
    ///
    /// This legacy infallible accessor returns `0` for a malformed public
    /// kernel. New fallible code should use [`Self::try_in_channels`].
    pub fn in_channels(&self) -> usize {
        if self.kernel.ndim() == 4 {
            self.kernel.shape()[0]
        } else {
            0
        }
    }

    /// Validated input-channel count.
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field violates the
    /// invariants established by the constructors.
    pub fn try_in_channels(&self) -> Result<usize> {
        self.validate_geometry().map(|(in_channels, _)| in_channels)
    }

    /// Kernel size (height, width).
    ///
    /// This legacy infallible accessor returns `(0, 0)` for a malformed public
    /// kernel. New fallible code should use [`Self::try_kernel_size`].
    pub fn kernel_size(&self) -> (usize, usize) {
        if self.kernel.ndim() == 4 {
            (self.kernel.shape()[2], self.kernel.shape()[3])
        } else {
            (0, 0)
        }
    }

    /// Validated kernel size (height, width).
    ///
    /// # Errors
    ///
    /// Returns an error if any publicly mutable geometry field violates the
    /// invariants established by the constructors.
    pub fn try_kernel_size(&self) -> Result<(usize, usize)> {
        self.validate_geometry()?;
        Ok((self.kernel.shape()[2], self.kernel.shape()[3]))
    }

    /// Compute output spatial dimensions.
    ///
    /// Returns an error if the arithmetic would underflow.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        self.validate_geometry()?;
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        let (oph, opw) = self.output_padding;
        // ConvTranspose output:
        //   stride*(in-1) + dilation*(kernel-1) + 1 - 2*pad + output_padding
        let eff_kh = kh
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(dh))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d effective kernel height overflow".to_string())
            })?;
        let eff_kw = kw
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(dw))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d effective kernel width overflow".to_string())
            })?;
        let expanded_h = input_h
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(sh))
            .and_then(|v| v.checked_add(eff_kh))
            .and_then(|v| v.checked_add(oph))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d output height overflow".to_string())
            })?;
        let expanded_w = input_w
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(sw))
            .and_then(|v| v.checked_add(eff_kw))
            .and_then(|v| v.checked_add(opw))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d output width overflow".to_string())
            })?;
        let double_ph = ph.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec("ConvTranspose2d padding height overflow".to_string())
        })?;
        let double_pw = pw.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec("ConvTranspose2d padding width overflow".to_string())
        })?;
        if expanded_h < double_ph || expanded_w < double_pw {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d output size underflow: \
                 input=({input_h},{input_w}), stride=({sh},{sw}), kernel=({kh},{kw}), \
                 dilation=({dh},{dw}), padding=({ph},{pw}), output_padding=({oph},{opw})"
            )));
        }
        Ok((expanded_h - double_ph, expanded_w - double_pw))
    }

    /// Batched CROWN backward propagation through ConvTranspose2d layer.
    ///
    /// For a transposed conv layer y = conv_transpose(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv_transpose is a regular convolution
    /// - new_A = conv(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        self.propagate_linear_batched_maybe_engine(bounds, None)
    }

    /// Batched CROWN backward propagation through ConvTranspose2d with optional GemmEngine.
    pub(crate) fn propagate_linear_batched_maybe_engine(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        debug!("ConvTranspose2d layer batched CROWN backward propagation");
        let (in_c, out_c) = self.validate_geometry()?;

        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        let conv_out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: batch dims product overflows: {:?}",
                batch_dims
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_h * in_w]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [total_batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        let total_rows = total_batch.checked_mul(out_dim).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: total_batch * out_dim overflows: {total_batch} * {out_dim}"
            ))
        })?;
        let (kh_e, kw_e) = self.kernel_size();
        let in_c_per_group = self.kernel.shape()[1];
        let n_contraction = in_c_per_group.saturating_mul(kh_e).saturating_mul(kw_e);
        let want_recompute =
            crate::layers::convolution::crown_helpers::conv_should_f64_recompute(n_contraction);
        // #wall-deadwork ConvTranspose batched port: whenever the certified f64
        // route is authoritative, both legacy f32 coefficient results are dead.
        // Successful recomputes overwrite them; failed recomputes zero the rows
        // and publish ±inf bias below. Allocate the destination buffers directly
        // and retain `NY_CONV_SKIP_DEAD_F32=0` as the diagnostic kill-switch,
        // matching the scalar ConvTranspose path.
        let skip_dead_f32 = want_recompute
            && crate::layers::convolution::crown_helpers::conv_skip_dead_f32_enabled();
        let mut new_lower_a = Array2::zeros((total_rows, conv_in_size));
        let mut new_upper_a = Array2::zeros((total_rows, conv_in_size));
        let mut lower_nonfinite_rows = vec![false; total_rows];
        let mut upper_nonfinite_rows = vec![false; total_rows];

        // When skipped, the f64-authoritative path below fills these buffers
        // before publication; on failure it conservatively degrades every row.
        if !skip_dead_f32 && engine.is_some() {
            let lower_a_flat = bounds
                .lower_a
                .view()
                .into_shape_with_order((total_rows, mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for GEMM".to_string()))?
                .to_owned();
            let upper_a_flat = bounds
                .upper_a
                .view()
                .into_shape_with_order((total_rows, mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for GEMM".to_string()))?
                .to_owned();

            new_lower_a = conv2d_forward_batched_gemm(
                &lower_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                engine,
            )?;
            new_upper_a = conv2d_forward_batched_gemm(
                &upper_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                engine,
            )?;
        } else if !skip_dead_f32 {
            for b in 0..total_batch {
                for d in 0..out_dim {
                    let lower_row = lower_a_3d.slice(s![b, d, ..]);
                    let upper_row = upper_a_3d.slice(s![b, d, ..]);

                    let lower_3d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), lower_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![lower_row.len()],
                        })?;

                    let upper_3d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), upper_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![upper_row.len()],
                        })?;

                    let lower_conv = conv2d_single(
                        &lower_3d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                    )?;
                    let upper_conv = conv2d_single(
                        &upper_3d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                    )?;

                    let row_idx = b * out_dim + d;
                    for (i, &val) in lower_conv.iter().enumerate() {
                        new_lower_a[[row_idx, i]] = val;
                    }
                    for (i, &val) in upper_conv.iter().enumerate() {
                        new_upper_a[[row_idx, i]] = val;
                    }
                }
            }
        }

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug on
        // the BATCHED β-CROWN/BaB verdict path). ConvTranspose2d is
        // `propagates_coeff_err = true` but stored the f32 forward-conv coefficient
        // with NO certified error → unsound. f64-recompute the SAME forward-conv
        // contraction (conv input = ConvTranspose2d output domain), store the
        // directed f32, attach `cast_err + γ_n^f64·S + prop`.
        let lower_a_2d = lower_a_3d
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for f64".to_string()))?
            .to_owned();
        let upper_a_2d = upper_a_3d
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for f64".to_string()))?
            .to_owned();
        let coeff_f64 = want_recompute
            .then(|| {
                conv2d_forward_backward_coeff_f64(
                    &lower_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                )
                .ok()
            })
            .flatten();
        let coeff_f64_u = want_recompute
            .then(|| {
                conv2d_forward_backward_coeff_f64(
                    &upper_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                )
                .ok()
            })
            .flatten();
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.dim() == (total_rows, conv_in_size));
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.dim() == (total_rows, conv_in_size));
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                for i in 0..total_rows {
                    for p in 0..conv_in_size {
                        new_lower_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                for i in 0..total_rows {
                    for p in 0..conv_in_size {
                        new_upper_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        // Scan the coefficient matrices exactly once, after the authoritative
        // f64 recompute has replaced the legacy f32 results. This keeps the
        // default skip and diagnostic legacy routes identical while retaining
        // the CROWN_COEFF_MAX/non-finite firewall on the values that will
        // actually be published. A wanted-but-failed recompute is marked below
        // and conservatively degrades every row regardless of this scan.
        for row_idx in 0..total_rows {
            lower_nonfinite_rows[row_idx] = (0..conv_in_size)
                .any(|col_idx| !is_crown_coeff_safe(new_lower_a[[row_idx, col_idx]]));
            upper_nonfinite_rows[row_idx] = (0..conv_in_size)
                .any(|col_idx| !is_crown_coeff_safe(new_upper_a[[row_idx, col_idx]]));
        }
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let in_lower_err_2d = flatten_incoming_coeff_err(
            "lower_a_err",
            bounds.lower_a_err.as_ref(),
            bounds.lower_a.shape(),
            total_rows,
            mid_dim,
        )?;
        let in_upper_err_2d = flatten_incoming_coeff_err(
            "upper_a_err",
            bounds.upper_a_err.as_ref(),
            bounds.upper_a.shape(),
            total_rows,
            mid_dim,
        )?;
        let mut lower_err_2d = batched_conv_coeff_err(
            &lower_a_2d,
            in_lower_err_2d.as_ref(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
            None,
        );
        let mut upper_err_2d = batched_conv_coeff_err(
            &upper_a_2d,
            in_upper_err_2d.as_ref(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
            None,
        );
        if lower_recompute_failed {
            new_lower_a.fill(0.0);
            lower_err_2d.fill(0.0);
            for r in lower_nonfinite_rows.iter_mut() {
                *r = true;
            }
        }
        if upper_recompute_failed {
            new_upper_a.fill(0.0);
            upper_err_2d.fill(0.0);
            for r in upper_nonfinite_rows.iter_mut() {
                *r = true;
            }
        }

        // Zero rows with unsafe A-matrix coefficients.
        let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "ConvTranspose2d batched CROWN backward: non-finite A-matrix in {}/{} lower, \
                 {}/{} upper rows — ±inf bias fallback",
                lower_affected, total_rows, upper_affected, total_rows
            );
            for i in 0..total_rows {
                if lower_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_lower_a[[i, j]] = 0.0;
                        lower_err_2d[[i, j]] = 0.0;
                    }
                }
                if upper_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_upper_a[[i, j]] = 0.0;
                        upper_err_2d[[i, j]] = 0.0;
                    }
                }
            }
        }

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
        let (lower_err_vec, _) = lower_err_2d.into_raw_vec_and_offset();
        let (upper_err_vec, _) = upper_err_2d.into_raw_vec_and_offset();
        let new_lower_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), lower_err_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_err".to_string()))?;
        let new_upper_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), upper_err_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_err".to_string()))?;

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            let lower_b_1d = bounds
                .lower_b
                .view()
                .into_shape_with_order(total_rows)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_1d = bounds
                .upper_b
                .view()
                .into_shape_with_order(total_rows)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let (mut new_lower_b, mut new_upper_b) = compute_conv_bias_rows_f64(
                lower_a_2d.view(),
                in_lower_err_2d.as_ref().map(|err| err.view()),
                lower_b_1d,
                upper_a_2d.view(),
                in_upper_err_2d.as_ref().map(|err| err.view()),
                upper_b_1d,
                bias,
                out_c,
                out_h.checked_mul(out_w).ok_or_else(|| {
                    NyError::InvalidSpec("ConvTranspose2d spatial size overflows".into())
                })?,
            )?;

            // Override bias for non-finite A-matrix rows.
            for row_idx in 0..total_rows {
                if lower_nonfinite_rows[row_idx] {
                    new_lower_b[row_idx] = f32::NEG_INFINITY;
                }
                if upper_nonfinite_rows[row_idx] {
                    new_upper_b[row_idx] = f32::INFINITY;
                }
            }

            let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            )
        } else {
            // Even without conv bias, override for non-finite A-matrix rows.
            let mut lb = bounds.lower_b.clone();
            let mut ub = bounds.upper_b.clone();
            if lower_affected > 0 || upper_affected > 0 {
                let lb_flat = contiguous_flat_slice_mut(&mut lb)?;
                let ub_flat = contiguous_flat_slice_mut(&mut ub)?;
                for i in 0..total_rows {
                    if lower_nonfinite_rows[i] {
                        lb_flat[i] = f32::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[i] {
                        ub_flat[i] = f32::INFINITY;
                    }
                }
            }
            (lb, ub)
        };

        let new_input_shape = if bounds.input_shape.is_empty() {
            vec![conv_in_size]
        } else if bounds.input_shape.len() >= 3 {
            let mut shape = bounds.input_shape.clone();
            let len = shape.len();
            shape[len - 3] = in_c;
            shape[len - 2] = in_h;
            shape[len - 1] = in_w;
            shape
        } else {
            let mut shape = bounds.input_shape[..bounds.input_shape.len() - 1].to_vec();
            shape.push(conv_in_size);
            shape
        };

        // CROWN backward NaN firewall: conservative fallback instead of hard error.
        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            new_input_shape,
            bounds.output_shape.clone(),
        )?;
        // Attach the certified coefficient error (#vnncomp-aw-soundness).
        out.set_coeff_err(new_lower_a_err, new_upper_a_err);
        Ok(out)
    }
}

/// Tests for the parameter-mutation boundary documented on [`Conv2dLayer`]
/// (#conv2d-parameter-boundary).
///
/// These live beside the structs rather than in this module's `tests.rs`
/// because the first of them is a compile-time ratchet on the struct
/// definitions themselves, and must break in the same file the definitions
/// change in.
#[cfg(test)]
mod parameter_boundary_tests {
    use super::{Conv2dLayer, ConvTranspose2dLayer};
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_core::NyError;

    /// out_channels = 2, in_channels/groups = 1, 1x1 kernel, bias of length 2,
    /// groups = 1, input_shape = (3, 4).
    fn conv2d_fixture() -> Conv2dLayer {
        Conv2dLayer::with_input_shape(
            ArrayD::<f32>::from_elem(IxDyn(&[2, 1, 1, 1]), 1.0),
            Some(Array1::from_vec(vec![0.5_f32, -0.25])),
            (1, 1),
            (0, 0),
            3,
            4,
        )
        .expect("the Conv2d fixture must construct")
    }

    /// in_channels = 1, out_channels = 2, 1x1 kernel, bias of length 2,
    /// input_shape = (3, 4).
    fn convtranspose2d_fixture() -> ConvTranspose2dLayer {
        ConvTranspose2dLayer::with_input_shape(
            ArrayD::<f32>::from_elem(IxDyn(&[1, 2, 1, 1]), 1.0),
            Some(Array1::from_vec(vec![0.5_f32, -0.25])),
            (1, 1),
            (0, 0),
            3,
            4,
        )
        .expect("the ConvTranspose2d fixture must construct")
    }

    /// COMPILE-TIME RATCHET, not a runtime assertion.
    ///
    /// Both patterns are exhaustive with no `..`, so adding ANY field to either
    /// struct stops this module compiling (E0027). That is the intended
    /// failure: a new field is either derived from `kernel`/`bias` — which
    /// these structs must not hold while those fields are `pub`, because a
    /// direct write would leave it stale and publish a wrong bound — or it is
    /// not, and the author says which by listing it here.
    #[ntest::timeout(10000)]
    #[test]
    fn conv_structs_carry_no_parameter_derived_field() {
        let Conv2dLayer {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_shape,
        } = conv2d_fixture();
        assert_eq!(kernel.shape(), &[2, 1, 1, 1]);
        assert_eq!(bias.map(|b| b.len()), Some(2));
        assert_eq!(stride, (1, 1));
        assert_eq!(padding, (0, 0));
        assert_eq!(dilation, (1, 1));
        assert_eq!(groups, 1);
        assert_eq!(input_shape, Some((3, 4)));

        let ConvTranspose2dLayer {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            input_shape,
        } = convtranspose2d_fixture();
        assert_eq!(kernel.shape(), &[1, 2, 1, 1]);
        assert_eq!(bias.map(|b| b.len()), Some(2));
        assert_eq!(stride, (1, 1));
        assert_eq!(padding, (0, 0));
        assert_eq!(dilation, (1, 1));
        assert_eq!(output_padding, (0, 0));
        assert_eq!(input_shape, Some((3, 4)));
    }

    /// A malformed kernel is refused and `self` is left untouched.
    ///
    /// Fails closed against the defect this method exists to prevent: an
    /// implementation that assigned the fields directly would return `Ok` here
    /// and leave a rank-1 kernel in place.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_replace_parameters_refuses_malformed_kernel_without_mutating() {
        let mut layer = conv2d_fixture();
        let kernel_before = layer.kernel().clone();
        let error = layer
            .replace_parameters(ArrayD::<f32>::zeros(IxDyn(&[1])), None)
            .expect_err("a rank-1 kernel must be refused");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(layer.kernel(), &kernel_before);
        assert_eq!(layer.bias().map(|b| b.to_vec()), Some(vec![0.5, -0.25]));
        assert_eq!(layer.out_channels(), 2);
        assert_eq!(layer.input_shape, Some((3, 4)));
    }

    /// A bias whose length does not match the NEW `out_channels` is refused,
    /// and `self` is left untouched.
    ///
    /// This is the exact incoherent state a direct `layer.kernel = ..` write
    /// produces, pinned here as something the boundary rejects.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_replace_parameters_refuses_bias_arity_mismatch_without_mutating() {
        let mut layer = conv2d_fixture();
        let error = layer
            .replace_parameters(
                ArrayD::<f32>::from_elem(IxDyn(&[3, 1, 1, 1]), 1.0),
                Some(Array1::from_vec(vec![0.0_f32, 0.0])),
            )
            .expect_err("a length-2 bias cannot match out_channels 3");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(layer.kernel().shape(), &[2, 1, 1, 1]);
        assert_eq!(layer.out_channels(), 2);
        assert_eq!(layer.bias().map(|b| b.to_vec()), Some(vec![0.5, -0.25]));
    }

    /// The success path commits a value the constructor produced: the derived
    /// geometry accessors follow the new kernel, and the non-parameter geometry
    /// — `input_shape` in particular, which `new_dilated` leaves `None` —
    /// survives.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_replace_parameters_commits_through_the_constructor() {
        let mut layer = conv2d_fixture();
        layer
            .replace_parameters(
                ArrayD::<f32>::from_elem(IxDyn(&[3, 1, 2, 2]), 0.25),
                Some(Array1::from_vec(vec![1.0_f32, 2.0, 3.0])),
            )
            .expect("a well-formed replacement must be accepted");

        assert_eq!(layer.kernel().shape(), &[3, 1, 2, 2]);
        assert_eq!(layer.bias().map(|b| b.to_vec()), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(layer.out_channels(), 3);
        assert_eq!(layer.in_channels(), 1);
        assert_eq!(layer.kernel_size(), (2, 2));
        assert_eq!(layer.stride, (1, 1));
        assert_eq!(layer.padding, (0, 0));
        assert_eq!(layer.dilation, (1, 1));
        assert_eq!(layer.groups, 1);
        assert_eq!(
            layer.input_shape,
            Some((3, 4)),
            "input_shape is not a parameter and must be carried across"
        );
        assert_eq!(
            layer.output_size(3, 4).expect("geometry must revalidate"),
            (2, 3)
        );
    }

    /// `set_kernel` / `set_bias` inherit the boundary's validation rather than
    /// writing their field.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_set_kernel_and_set_bias_route_through_replace_parameters() {
        let mut layer = conv2d_fixture();

        let error = layer
            .set_kernel(ArrayD::<f32>::from_elem(IxDyn(&[3, 1, 1, 1]), 1.0))
            .expect_err("the retained length-2 bias cannot match out_channels 3");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(layer.kernel().shape(), &[2, 1, 1, 1]);

        let replacement = ArrayD::<f32>::from_elem(IxDyn(&[2, 1, 1, 1]), 7.0);
        layer
            .set_kernel(replacement.clone())
            .expect("a shape-preserving kernel must be accepted");
        assert_eq!(layer.kernel(), &replacement);
        assert_eq!(layer.bias().map(|b| b.to_vec()), Some(vec![0.5, -0.25]));
        assert_eq!(layer.input_shape, Some((3, 4)));

        let error = layer
            .set_bias(Some(Array1::from_vec(vec![0.0_f32])))
            .expect_err("a length-1 bias cannot match out_channels 2");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(layer.bias().map(|b| b.to_vec()), Some(vec![0.5, -0.25]));

        layer.set_bias(None).expect("dropping the bias is valid");
        assert!(layer.bias().is_none());
        assert_eq!(layer.kernel(), &replacement);
    }

    /// Pins the exposure the struct doc describes, so that doc cannot go stale.
    ///
    /// The parameter fields are still `pub`, so a direct write lands with no
    /// revalidation and leaves the layer incoherent; only the entry-point
    /// `validate_geometry` call catches it afterwards. The supported boundary
    /// refuses the identical transition up front. If `kernel` is ever made
    /// non-`pub`, this test stops compiling — the correct failure, because the
    /// struct doc must be rewritten in the same change.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_direct_field_assignment_still_bypasses_validation() {
        let mut unguarded = conv2d_fixture();
        unguarded.kernel = ArrayD::<f32>::from_elem(IxDyn(&[3, 1, 1, 1]), 1.0);
        assert_eq!(
            unguarded.out_channels(),
            3,
            "the direct write landed with no revalidation"
        );
        assert_eq!(
            unguarded.bias().map(|b| b.len()),
            Some(2),
            "and left a bias the new out_channels cannot match"
        );
        assert!(
            matches!(
                unguarded.validate_geometry(),
                Err(NyError::ShapeMismatch { .. })
            ),
            "the entry-point revalidation is the only thing that still catches this"
        );

        let mut guarded = conv2d_fixture();
        let error = guarded
            .set_kernel(ArrayD::<f32>::from_elem(IxDyn(&[3, 1, 1, 1]), 1.0))
            .expect_err("the boundary must refuse what the field write allows");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(guarded.out_channels(), 2);
        assert!(guarded.validate_geometry().is_ok());
    }

    /// The ConvTranspose2d boundary behaves identically: refuse-and-leave on a
    /// malformed kernel, commit-through-the-constructor on a valid one.
    #[ntest::timeout(10000)]
    #[test]
    fn convtranspose2d_replace_parameters_refuses_then_commits() {
        let mut layer = convtranspose2d_fixture();
        let kernel_before = layer.kernel().clone();
        let error = layer
            .replace_parameters(ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])), None)
            .expect_err("a rank-3 kernel must be refused");
        assert!(matches!(&error, NyError::ShapeMismatch { .. }), "{error:?}");
        assert_eq!(layer.kernel(), &kernel_before);
        assert_eq!(layer.bias().map(|b| b.len()), Some(2));

        layer
            .replace_parameters(
                ArrayD::<f32>::from_elem(IxDyn(&[1, 3, 2, 2]), 0.5),
                Some(Array1::from_vec(vec![1.0_f32, 2.0, 3.0])),
            )
            .expect("a well-formed replacement must be accepted");
        assert_eq!(layer.in_channels(), 1);
        assert_eq!(layer.out_channels(), 3);
        assert_eq!(layer.kernel_size(), (2, 2));
        assert_eq!(layer.output_padding, (0, 0));
        assert_eq!(
            layer.input_shape,
            Some((3, 4)),
            "input_shape is not a parameter and must be carried across"
        );
    }
}
