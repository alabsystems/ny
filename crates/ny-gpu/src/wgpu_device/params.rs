// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Parameters for the linear IBP shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LinearIbpParams {
    pub(super) batch_size: u32,
    pub(super) in_features: u32,
    pub(super) out_features: u32,
    pub(super) _padding: u32,
}

/// Parameters for the SOUND linear IBP shader (`docs/SOUND_GPU_IBP_PLAN.md`
/// §3.1), passed via uniform buffer.
///
/// 32 bytes, std140-clean. Layout MUST match the WGSL `struct Params` in
/// `LINEAR_IBP_SOUND_BODY` exactly: `batch_size, in_features, out_features,
/// n_ulps` (u32) then `gamma_k, slack, additive` (f32) and a trailing `_pad`.
///
/// - `n_ulps = 2·(in_features + 2)` — the CPU N-D sound ULP count the dense-chain
///   GPU stands in for. `linear.propagate_ibp_sound` widens by `in_features + 2`
///   ULPs and `propagate_ibp` (N-D) already applied the same widen, so the CPU
///   verdict path widens TWICE ⇒ `2·(in_features + 2)`.
/// - `gamma_k = gamma_k_f32(k)`, `slack = combine_slack_f32(k)`,
///   `additive = ftz_safe_underflow_floor(k)` with `k = in_features + 3`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LinearIbpSoundParams {
    pub(super) batch_size: u32,
    pub(super) in_features: u32,
    pub(super) out_features: u32,
    pub(super) n_ulps: u32,
    pub(super) gamma_k: f32,
    pub(super) slack: f32,
    pub(super) additive: f32,
    pub(super) _pad: u32,
}

/// Parameters for the SOUND Conv2d IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.2).
///
/// 80 bytes, std140-clean (5×16). Layout MUST match the WGSL `struct Params` in
/// `CONV2D_IBP_SOUND_BODY` exactly: 15 `u32` (the 14 conv dims + `n_ulps`), then
/// `gamma_k, slack, additive` (f32) and two trailing `_pad` u32.
///
/// - `k = (in_channels/groups)·kernel_h·kernel_w + 3` (full window; padding taps
///   only shrink the true count ⇒ the fixed larger `k` over-bounds `γ`).
/// - `n_ulps = 2·((in_channels/groups)·kernel_h·kernel_w + 2)` (the strict-CPU term).
/// - `gamma_k = gamma_k_f32(k)`, `slack = combine_slack_f32(k)`,
///   `additive = ftz_safe_underflow_floor(k)`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Conv2dIbpSoundParams {
    pub(super) batch_size: u32,
    pub(super) in_channels: u32,
    pub(super) out_channels: u32,
    pub(super) input_h: u32,
    pub(super) input_w: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride_h: u32,
    pub(super) stride_w: u32,
    pub(super) pad_h: u32,
    pub(super) pad_w: u32,
    pub(super) groups: u32,
    pub(super) n_ulps: u32,
    pub(super) gamma_k: f32,
    pub(super) slack: f32,
    pub(super) additive: f32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

/// Parameters for the SOUND MatMul IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.3).
///
/// 32 bytes, std140-clean. Layout MUST match the WGSL `struct Params` in
/// `MATMUL_IBP_SOUND_BODY`: `batch_size, m, k, n` (u32) then `gamma_k, slack,
/// additive` (f32) and a trailing `_pad`. `k = contraction + 3`; there is NO
/// `n_ulps` term (CORE radius `γ·S·slack + flush`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MatmulIbpSoundParams {
    pub(super) batch_size: u32,
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
    pub(super) gamma_k: f32,
    pub(super) slack: f32,
    pub(super) additive: f32,
    pub(super) _pad: u32,
}

/// Parameters for the SOUND AvgPool IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.4).
///
/// 64 bytes (identical layout to the FAST [`AvgPoolIbpParams`] but the 3 trailing
/// `_padding` u32 are repurposed as `gamma_k, slack, additive` f32). `k =
/// kernel_h·kernel_w + 3`; coefficient `1/D ≤ 1` ⇒ no §0 amplifier.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AvgPoolIbpSoundParams {
    pub(super) num_elements: u32,
    pub(super) channels: u32,
    pub(super) input_h: u32,
    pub(super) input_w: u32,
    pub(super) output_h: u32,
    pub(super) output_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride_h: u32,
    pub(super) stride_w: u32,
    pub(super) pad_h: u32,
    pub(super) pad_w: u32,
    pub(super) count_include_pad: u32,
    pub(super) gamma_k: f32,
    pub(super) slack: f32,
    pub(super) additive: f32,
}

/// Parameters for the SOUND Scale IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.8).
///
/// 16 bytes (same shape as the FAST [`ScaleIbpParams`] but the 2 trailing padding
/// u32 are repurposed as the `|s|`-amplified `scale_floor` and the base
/// `zero_ulp_floor`). `scale_floor = up(|s|·FLOOR + FLOOR) ≥ |s|·FLT_MIN`,
/// `FLOOR = ftz_safe_underflow_floor(1)`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScaleIbpSoundParams {
    pub(super) total_elements: u32,
    pub(super) scale: f32,
    pub(super) scale_floor: f32,
    pub(super) zero_ulp_floor: f32,
}

/// Parameters for the matmul IBP shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MatmulIbpParams {
    pub(super) batch_size: u32,
    pub(super) m: u32, // rows of A
    pub(super) k: u32, // cols of A = rows of B
    pub(super) n: u32, // cols of B
}

/// Parameters for the GEMM shader, passed via uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GemmParams {
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
    pub(super) _padding: u32,
}

/// Parameters for the softmax IBP shader, passed via uniform buffer.
/// Softmax is computed along the last dimension (row-wise for 2D).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxIbpParams {
    pub(super) num_rows: u32, // Total number of rows (batch * leading dims)
    pub(super) row_size: u32, // Size of softmax dimension (last axis)
    pub(super) _padding: [u32; 2],
}

/// Parameters for the transpose IBP shader, passed via uniform buffer.
/// Transposes the last two dimensions of a tensor.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TransposeIbpParams {
    pub(super) batch_size: u32, // Product of all dims except last two
    pub(super) rows: u32,       // Second-to-last dimension (before transpose)
    pub(super) cols: u32,       // Last dimension (before transpose)
    pub(super) _padding: u32,
}

/// Parameters for the scale IBP shader, passed via uniform buffer.
/// Element-wise multiplication by a scalar.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScaleIbpParams {
    pub(super) total_elements: u32,
    pub(super) scale: f32,
    pub(super) _padding: [u32; 2],
}

/// Parameters for the CROWN activation backward shader.
/// One workgroup per spec row; each thread handles neurons in a strided loop.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CrownActivationParams {
    pub(super) num_specs: u32,
    pub(super) num_neurons: u32,
    pub(super) _padding: [u32; 2],
}

/// Parameters for the CROWN linear bias accumulation shader.
/// Computes b[i] += sum_j(A[i,j] * layer_bias[j]) per spec row.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CrownBiasAccumParams {
    pub(super) num_specs: u32,
    pub(super) num_features: u32,
    pub(super) _padding: [u32; 2],
}

/// Parameters for the CROWN MaxPool2d backward shader.
///
/// One workgroup handles one spec row. Threads zero the destination input-space
/// A-matrix, scatter routed winner coefficients, and reduce IBP fallback bias.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CrownMaxPool2dParams {
    pub(super) num_specs: u32,
    pub(super) input_dim: u32,
    pub(super) output_dim: u32,
    pub(super) _padding: u32,
}

/// Parameters for the CROWN concretization shader.
/// Final bound computation from A-matrices and input bounds.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CrownConcretizeParams {
    pub(super) num_specs: u32,
    pub(super) input_dim: u32,
    pub(super) _padding: [u32; 2],
}

/// Parameters for the Add IBP shader (#4319).
/// Element-wise addition of two interval bound pairs (residual connections).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AddIbpParams {
    pub(super) num_elements: u32,
    pub(super) _padding: [u32; 3],
}

/// Parameters for the ReLU IBP in-place shader (#4081).
/// Elementwise max(x, 0) on lower and upper bound buffers.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReluIbpParams {
    pub(super) num_elements: u32,
    pub(super) _padding: [u32; 3],
}

/// Parameters for the resident Conv2d IBP shader (#4275).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Conv2dIbpParams {
    pub(super) batch_size: u32,
    pub(super) in_channels: u32,
    pub(super) out_channels: u32,
    pub(super) input_h: u32,
    pub(super) input_w: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride_h: u32,
    pub(super) stride_w: u32,
    pub(super) pad_h: u32,
    pub(super) pad_w: u32,
    pub(super) groups: u32,
    pub(super) _padding: [u32; 2],
}

/// Parameters for the AveragePool IBP shader (#4320).
/// Windowed or global average pooling on interval bound pairs.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AvgPoolIbpParams {
    pub(super) num_elements: u32,
    pub(super) channels: u32,
    pub(super) input_h: u32,
    pub(super) input_w: u32,
    pub(super) output_h: u32,
    pub(super) output_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride_h: u32,
    pub(super) stride_w: u32,
    pub(super) pad_h: u32,
    pub(super) pad_w: u32,
    pub(super) count_include_pad: u32,
    pub(super) _padding: [u32; 3],
}

/// Parameters for the Conv2d A-matrix reshape shader (#3397).
/// Transforms A from (S, OC*OH*OW) to (S*OH*OW, OC).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ConvReshapeParams {
    pub(super) num_specs: u32,
    pub(super) out_channels: u32,
    pub(super) spatial: u32, // oh * ow
    pub(super) _padding: u32,
}

/// Parameters for the Conv2d col2im gather shader (#3397).
/// Gathers GEMM output into scattered input-space A-matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ConvCol2imParams {
    pub(super) num_specs: u32,
    pub(super) flat_input_dim: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) in_channels: u32,
    pub(super) in_h: u32,
    pub(super) in_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride_h: u32,
    pub(super) stride_w: u32,
    pub(super) pad_h: u32,
    pub(super) pad_w: u32,
    pub(super) kernel_cols: u32,
    pub(super) _padding2: [u32; 2],
}

/// Params for the SOUND MaxPool2d CROWN-backward coefficient shader (T1.2). 80 bytes
/// (std140-clean). `total = num_outputs·input_size` (the offset into `err_comb` for
/// the upper-row errors). `gamma_k`/`slack`/`additive` drive the per-coefficient
/// certified error; `additive` is the NORMAL-range FTZ-safe floor (coefficient-1
/// accumulation ⇒ no §0 amplifier).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaxpoolCrownSoundParams {
    pub(super) num_outputs: u32,
    pub(super) input_size: u32,
    pub(super) output_size: u32,
    pub(super) channels: u32,
    pub(super) in_h: u32,
    pub(super) in_w: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) kh: u32,
    pub(super) kw: u32,
    pub(super) sh: u32,
    pub(super) sw: u32,
    pub(super) ph: u32,
    pub(super) pw: u32,
    pub(super) gamma_k: f32,
    pub(super) slack: f32,
    pub(super) additive: f32,
    pub(super) total: u32,
    pub(super) _p0: u32,
    pub(super) _p1: u32,
}
