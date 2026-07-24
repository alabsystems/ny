// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};

/// Transposed 1D convolution for CROWN backward pass (gradient scatter).
///
/// Given gradient at conv output, compute gradient at conv input.
/// Supports dilation and groups.
///
/// Kernel shape: `(out_channels, in_channels/groups, kernel_size)`.
/// Input shape: `(out_channels, out_len)` — gradient from above.
/// Output shape: `(in_channels, in_len)`.
///
/// Reference: PyTorch `torch.nn.Conv1d` backward.
pub(crate) fn conv1d_transpose(
    input: &ArrayD<f32>,  // (out_channels, out_len) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kernel_size)
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_length: usize, // expected input length (in_len)
) -> Result<ArrayD<f32>> {
    // Guard: dilation=0 silently produces wrong indices; groups=0 panics on division.
    if dilation == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_transpose: dilation must be >= 1".to_string(),
        ));
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_transpose: groups must be >= 1".to_string(),
        ));
    }
    // Guard: ndim checks prevent panic on shape indexing (#2920 WP-B).
    if input.ndim() < 2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![2],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = input.shape()[0];
    let grad_len = input.shape()[1];

    let ker_out_c = kernel.shape()[0];
    let ker_in_c_per_group = kernel.shape()[1]; // in_channels / groups
    let k = kernel.shape()[2];

    if out_c != ker_out_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![ker_out_c],
            got: vec![out_c],
        });
    }

    let in_c = ker_in_c_per_group * groups;
    let in_len = output_length;
    let out_c_per_group = out_c / groups;

    let mut output = ArrayD::zeros(ndarray::IxDyn(&[in_c, in_len]));

    // Transposed convolution with groups: scatter gradient to input positions.
    // For each output position ol in the forward conv,
    // il = ol*stride + ki*dilation - padding.
    // In backward: input_grad[ic, il] += output_grad[oc, ol] * kernel[oc, ic_local, ki]
    for g in 0..groups {
        let ic_start = g * ker_in_c_per_group;
        let oc_start = g * out_c_per_group;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            for grad_l in 0..grad_len {
                let grad_val = input[[oc, grad_l]];
                if grad_val == 0.0 {
                    continue;
                }
                for ic_local in 0..ker_in_c_per_group {
                    let ic = ic_start + ic_local;
                    for ki in 0..k {
                        let il = (grad_l * stride + ki * dilation) as isize - padding as isize;
                        // SAFETY(as usize): il is isize, guard ensures >= 0 and < in_len.
                        if il >= 0 && il < in_len as isize {
                            output[[ic, il as usize]] += grad_val * kernel[[oc, ic_local, ki]];
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Perform 1D transposed convolution (forward op).
///
/// Input shape: (in_channels, in_len)
/// Kernel shape: (in_channels, out_channels/groups, kernel_size) (ONNX ConvTranspose layout)
/// Output shape: (out_channels, out_len)
pub(crate) fn conv1d_transpose_forward(
    input: &ArrayD<f32>,  // (in_channels, in_len)
    kernel: &ArrayD<f32>, // (in_channels, out_channels/groups, kernel_size)
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ArrayD<f32>> {
    if dilation == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_transpose_forward: dilation must be >= 1".to_string(),
        ));
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_transpose_forward: groups must be >= 1".to_string(),
        ));
    }
    // Guard: ndim checks prevent panic on shape indexing (#2920 WP-B).
    if input.ndim() < 2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![2],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let in_c = input.shape()[0];
    let in_len = input.shape()[1];

    let ker_in_c = kernel.shape()[0];
    let out_c_per_group = kernel.shape()[1];
    let k = kernel.shape()[2];

    if in_c != ker_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![ker_in_c],
            got: vec![in_c],
        });
    }
    if !in_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "conv1d_transpose_forward: in_channels ({in_c}) must be divisible by groups ({groups})"
        )));
    }
    let out_c = out_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv1d_transpose_forward: output channels overflow: {out_c_per_group} * {groups}"
        ))
    })?;
    let in_c_per_group = in_c / groups;

    // Checked arithmetic: (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * padding
    // Guard against underflow when in_len=0 or 2*padding > (in_len-1)*stride + k.
    let effective_k = dilation * (k - 1) + 1;
    let expanded = in_len.checked_sub(1)
        .and_then(|v| v.checked_mul(stride))
        .and_then(|v| v.checked_add(effective_k))
        .ok_or_else(|| NyError::InvalidSpec(
            format!(
                "conv1d_transpose_forward: dimension overflow: in_len={in_len}, stride={stride}, effective_k={effective_k}"
            )
        ))?;
    let double_pad = 2 * padding;
    if expanded < double_pad {
        return Err(NyError::InvalidSpec(format!(
            "conv1d_transpose_forward: output length underflow: \
             (in_len={in_len}-1)*stride={stride}+effective_k={effective_k}={expanded} < 2*padding={double_pad}"
        )));
    }
    let out_len = expanded - double_pad;

    let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_len]));

    for g in 0..groups {
        let ic_start = g * in_c_per_group;
        let oc_start = g * out_c_per_group;
        for ic_local in 0..in_c_per_group {
            let ic = ic_start + ic_local;
            for i in 0..in_len {
                let input_val = input[[ic, i]];
                if input_val == 0.0 {
                    continue;
                }
                for oc_local in 0..out_c_per_group {
                    let oc = oc_start + oc_local;
                    for k_idx in 0..k {
                        let out_idx = (i * stride + k_idx * dilation) as isize - padding as isize;
                        // SAFETY(as usize): out_idx is isize, guard ensures >= 0 and < out_len.
                        if out_idx >= 0 && out_idx < out_len as isize {
                            output[[oc, out_idx as usize]] +=
                                input_val * kernel[[ic, oc_local, k_idx]];
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Perform 1D convolution on a single (channels, length) input.
///
/// Supports dilation and groups.
///
/// Kernel shape: `(out_channels, in_channels/groups, kernel_size)`.
/// Input shape: `(in_channels, length)`.
/// Output shape: `(out_channels, output_length)`.
///
/// Reference: PyTorch `torch.nn.Conv1d`.
pub(crate) fn conv1d_single(
    input: &ArrayD<f32>,  // (in_channels, length)
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kernel_size)
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<ArrayD<f32>> {
    // Guard: dilation=0 silently produces wrong indices; groups=0 panics on division.
    if dilation == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_single: dilation must be >= 1".to_string(),
        ));
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_single: groups must be >= 1".to_string(),
        ));
    }
    // Guard: ndim checks prevent panic on shape indexing (#2920 WP-B).
    if input.ndim() < 2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![2],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let in_c = input.shape()[0];
    let in_len = input.shape()[1];

    let out_c = kernel.shape()[0];
    let ker_in_c_per_group = kernel.shape()[1]; // in_channels / groups
    let k = kernel.shape()[2];

    // With groups, kernel has in_c/groups channels per group.
    if in_c != ker_in_c_per_group * groups {
        return Err(NyError::ShapeMismatch {
            expected: vec![ker_in_c_per_group * groups],
            got: vec![in_c],
        });
    }

    // Checked arithmetic: (in_len + 2*padding - dilation*(k-1) - 1) / stride + 1
    let effective_k = dilation * (k - 1) + 1;
    let padded = in_len.checked_add(2 * padding).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv1d_single: padded length overflow: in_len={in_len}, padding={padding}"
        ))
    })?;
    if padded < effective_k {
        return Err(NyError::InvalidSpec(format!(
            "conv1d_single: effective kernel size ({effective_k}, dilation={dilation}) \
             larger than padded input ({padded}): in_len={in_len}, padding={padding}"
        )));
    }
    if stride == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_single: stride must be >= 1".to_string(),
        ));
    }
    let out_len = (padded - effective_k) / stride + 1;
    let out_c_per_group = out_c / groups;
    let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_len]));

    for g in 0..groups {
        let ic_start = g * ker_in_c_per_group;
        let oc_start = g * out_c_per_group;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            for ol in 0..out_len {
                let mut sum = 0.0f32;
                for ic_local in 0..ker_in_c_per_group {
                    let ic = ic_start + ic_local;
                    for ki in 0..k {
                        // Dilation: space kernel elements by dilation factor.
                        let il = (ol * stride + ki * dilation) as isize - padding as isize;
                        // SAFETY(as usize): il is isize, guard ensures >= 0 and < in_len.
                        if il >= 0 && il < in_len as isize {
                            sum += input[[ic, il as usize]] * kernel[[oc, ic_local, ki]];
                        }
                        // Padding: out-of-bounds treated as 0
                    }
                }
                output[[oc, ol]] = sum;
            }
        }
    }

    Ok(output)
}
