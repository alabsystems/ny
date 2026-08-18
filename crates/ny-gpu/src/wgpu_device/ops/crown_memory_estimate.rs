// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Upfront memory estimation for GPU CROWN backward (#3515).
//!
//! Computes a conservative memory footprint estimate before allocating any
//! GPU buffers, enabling the memory budget gate in `crown_backward_gpu()` to
//! reject oversized workloads and fall back to CPU CROWN backward.
//!
//! Reference: designs/2026-03-10-issue-3515-gpu-memory-bounded-crown-backward.md

use ny_core::GpuCrownLayer;

use super::crown_backward_types::{conv2d_buffer_sizes, layer_input_dim, layer_output_dim};

/// Default GPU memory budget: 8 GiB.
///
/// On macOS/Metal, GPU allocations come from unified system memory, so the
/// effective budget is `min(system_ram / 2, 8 GiB)` unless overridden.
const DEFAULT_GPU_MEMORY_BUDGET_MB: usize = 8192;

/// Upfront memory estimate for GPU CROWN backward (#3515).
///
/// Aggregates A-matrix ping-pong buffers, conv intermediates, weight/bias
/// uploads, and miscellaneous staging buffers. Includes the 1.2× growth factor
/// applied by `BufferPool::get_or_create_storage_buffer`.
#[derive(Debug, Clone)]
pub(crate) struct CrownMemoryEstimate {
    /// A-matrix ping-pong buffers: 4 × num_specs × max_dim × sizeof(f32)
    pub(crate) a_matrix_bytes: usize,
    /// Conv intermediate buffers: 2 × (max_reshaped + max_gemm_out) × sizeof(f32)
    pub(crate) conv_bytes: usize,
    /// Weight, bias, slopes, staging, and other small buffers
    pub(crate) misc_bytes: usize,
    /// Total estimated bytes including 1.2× growth factor
    pub(crate) total_bytes: usize,
}

/// Estimate total GPU CROWN backward working-set bytes for this model/spec set.
pub fn estimate_crown_backward_peak_bytes(layers: &[GpuCrownLayer], num_specs: usize) -> usize {
    estimate_crown_backward_memory(layers, num_specs).total_bytes
}

/// Estimate GPU CROWN backward memory before allocating any buffers (#3515).
///
/// This is the Tier 1 estimate from the design doc. It computes the total
/// memory footprint based on layer topology and num_specs, then applies the
/// BufferPool 1.2× growth factor.
pub(crate) fn estimate_crown_backward_memory(
    layers: &[GpuCrownLayer],
    num_specs: usize,
) -> CrownMemoryEstimate {
    let f32_size = size_of::<f32>();

    // Max dimension across all layer inputs/outputs for A-matrix sizing
    let max_dim = layers
        .iter()
        .filter_map(|l| layer_input_dim(l).ok())
        .chain(layers.iter().filter_map(|l| layer_output_dim(l).ok()))
        .max()
        .unwrap_or(0);

    // A-matrix ping-pong: 4 buffers × num_specs × max_dim × sizeof(f32)
    let a_matrix_bytes = 4 * num_specs * max_dim * f32_size;

    // Conv intermediates (reshaped + GEMM output, lower + upper = 2×)
    let (max_conv_reshaped, max_conv_gemm_out) = conv2d_buffer_sizes(layers, num_specs);
    let conv_bytes = 2 * (max_conv_reshaped + max_conv_gemm_out) * f32_size;

    // Misc: weight uploads, bias, slopes, input bounds, output, staging
    let max_weight_elems = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Linear {
                in_features,
                out_features,
                ..
            } => Some(in_features * out_features),
            GpuCrownLayer::MaxPool2d { output_dim, .. } => Some(*output_dim),
            GpuCrownLayer::Conv2d { weight_col, .. } => Some(weight_col.len()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let max_bias_elems = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Linear { out_features, .. } => Some(*out_features),
            GpuCrownLayer::Conv2d {
                out_channels,
                out_h,
                out_w,
                ..
            } => Some(out_channels * out_h * out_w),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let max_activation_elems = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Activation { num_neurons, .. }
            | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => Some(num_neurons * 4),
            GpuCrownLayer::MaxPool2d { output_dim, .. } => Some(output_dim * 2),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    // Bias accumulators (2), input bounds (2), output (2), staging (2) — all num_specs
    let fixed_num_specs_bufs = 8 * num_specs * f32_size;
    let misc_bytes = (max_weight_elems + max_bias_elems + max_activation_elems) * f32_size
        + fixed_num_specs_bufs;

    // Apply 1.2× growth factor (matching BufferPool allocation)
    let total_bytes = ((a_matrix_bytes + conv_bytes + misc_bytes) as f64 * 1.2) as usize;

    CrownMemoryEstimate {
        a_matrix_bytes,
        conv_bytes,
        misc_bytes,
        total_bytes,
    }
}

/// Get the GPU CROWN memory budget in bytes (#3515).
///
/// Priority: `NY_GPU_MEMORY_BUDGET_MB` env var > `min(system_ram / 2, 8 GiB)`.
pub fn gpu_memory_budget_bytes() -> usize {
    if let Ok(mb) = std::env::var("NY_GPU_MEMORY_BUDGET_MB") {
        if let Ok(n) = mb.parse::<usize>() {
            return n * 1024 * 1024;
        }
    }
    let sys_ram = system_memory_bytes();
    if sys_ram > 0 {
        (sys_ram / 2).min(DEFAULT_GPU_MEMORY_BUDGET_MB * 1024 * 1024)
    } else {
        DEFAULT_GPU_MEMORY_BUDGET_MB * 1024 * 1024
    }
}

/// Query total system RAM in bytes. Returns 0 on failure.
fn system_memory_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .parse::<usize>()
                    .ok()
            })
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|kb| kb.parse::<usize>().ok())
                            .map(|kb| kb * 1024)
                    })
            })
            .unwrap_or(0)
    }
}

/// Find the largest spec batch that fits inside the provided GPU memory budget.
///
/// Returns `0` when even a single-spec batch exceeds the budget.
pub(crate) fn max_specs_per_budget(
    layers: &[GpuCrownLayer],
    num_specs: usize,
    budget_bytes: usize,
) -> usize {
    if num_specs == 0 {
        return 0;
    }
    if estimate_crown_backward_peak_bytes(layers, 1) > budget_bytes {
        return 0;
    }
    if estimate_crown_backward_peak_bytes(layers, num_specs) <= budget_bytes {
        return num_specs;
    }

    let mut low = 1usize;
    let mut high = num_specs;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if estimate_crown_backward_peak_bytes(layers, mid) <= budget_bytes {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    // Blessed env-mutation choke point (clippy env wall): the previous local
    // ScopedEnvVar + lock duplicate is replaced by the shared serialized one.
    use ny_test_utils::env::ScopedEnvVar;

    /// Soundnessbench Conv2-like: 384 specs, in_channels=24, 24->24 3x3, 64x64.
    /// GEMM output = 384 x 64 x 64 x 24 x 3 x 3 = 339M f32 ~ 1.3 GB x 2 = 2.6 GB.
    /// With A-matrices + misc, total should exceed 4 GB.
    #[test]
    fn test_memory_estimate_soundnessbench_exceeds_4gb() {
        let layers = vec![
            GpuCrownLayer::Conv2d {
                weight_col: vec![0.0; 24 * 24 * 3 * 3].into(),
                bias_expanded: Some(vec![0.0; 24 * 64 * 64].into()),
                out_channels: 24,
                in_channels: 24,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                out_h: 64,
                out_w: 64,
                in_h: 64,
                in_w: 64,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.5; 24 * 64 * 64],
                upper_slope: vec![1.0; 24 * 64 * 64],
                lower_intercept: vec![0.0; 24 * 64 * 64],
                upper_intercept: vec![0.0; 24 * 64 * 64],
                num_neurons: 24 * 64 * 64,
            },
        ];
        let est = estimate_crown_backward_memory(&layers, 384);
        let gb = est.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            gb > 4.0,
            "soundnessbench-like estimate should exceed 4 GB, got {gb:.2} GB"
        );
        assert!(
            est.a_matrix_bytes > 0,
            "a_matrix_bytes should be positive for soundnessbench-like"
        );
        assert!(
            est.conv_bytes > 0,
            "conv_bytes should be positive for conv network"
        );
        assert!(est.misc_bytes > 0, "misc_bytes should be positive");
        assert!(
            est.total_bytes > est.a_matrix_bytes + est.conv_bytes,
            "total_bytes should exceed sum of a_matrix + conv (misc contributes too)"
        );
    }

    /// ACAS-Xu: 5 specs, small MLP (50->50->50->50->50->5). Should be < 1 MB.
    #[test]
    fn test_memory_estimate_acasxu_small() {
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: vec![0.0; 5 * 50].into(),
                bias: Some(vec![0.0; 5].into()),
                out_features: 5,
                in_features: 50,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.5; 50],
                upper_slope: vec![1.0; 50],
                lower_intercept: vec![0.0; 50],
                upper_intercept: vec![0.0; 50],
                num_neurons: 50,
            },
            GpuCrownLayer::Linear {
                weight: vec![0.0; 50 * 50].into(),
                bias: Some(vec![0.0; 50].into()),
                out_features: 50,
                in_features: 50,
                cert_err: Default::default(),
            },
        ];
        let est = estimate_crown_backward_memory(&layers, 5);
        let mb = est.total_bytes as f64 / (1024.0 * 1024.0);
        assert!(
            mb < 1.0,
            "ACAS-Xu estimate should be < 1 MB, got {mb:.4} MB"
        );
        assert_eq!(est.conv_bytes, 0);
    }

    /// Empty layer list should estimate ~0 bytes.
    #[test]
    fn test_memory_estimate_empty_layers() {
        let est = estimate_crown_backward_memory(&[], 10);
        assert_eq!(est.a_matrix_bytes, 0);
        assert_eq!(est.conv_bytes, 0);
        assert!(est.total_bytes > 0);
    }

    #[test]
    fn test_gpu_memory_budget_env_override() {
        let _guard = ny_test_utils::env::lock_env();
        let _env = ScopedEnvVar::set("NY_GPU_MEMORY_BUDGET_MB", "123");
        assert_eq!(gpu_memory_budget_bytes(), 123 * 1024 * 1024);
    }

    #[test]
    fn test_max_specs_per_budget_finds_largest_fitting_batch() {
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: vec![0.0; 16 * 16].into(),
                bias: Some(vec![0.0; 16].into()),
                out_features: 16,
                in_features: 16,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.5; 16],
                upper_slope: vec![1.0; 16],
                lower_intercept: vec![0.0; 16],
                upper_intercept: vec![0.0; 16],
                num_neurons: 16,
            },
        ];
        let one_spec = estimate_crown_backward_peak_bytes(&layers, 1);
        let two_specs = estimate_crown_backward_peak_bytes(&layers, 2);
        let three_specs = estimate_crown_backward_peak_bytes(&layers, 3);

        assert_eq!(max_specs_per_budget(&layers, 3, one_spec - 1), 0);
        assert_eq!(max_specs_per_budget(&layers, 3, one_spec), 1);
        assert_eq!(max_specs_per_budget(&layers, 3, two_specs), 2);
        assert_eq!(max_specs_per_budget(&layers, 3, three_specs), 3);
    }
}
