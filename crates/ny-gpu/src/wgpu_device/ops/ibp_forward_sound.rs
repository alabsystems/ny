// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUND (verdict-legal) GPU-resident IBP forward pass (`docs/SOUND_GPU_IBP_PLAN.md`
//! §3.1 keystone + §6.3 sequential dense-chain entry, T1.1).
//!
//! Every emitted endpoint is a CERTIFIED enclosure: the reduction rounding is
//! over-bounded by a directed `3·γ_k·S + 4·N·U·|endpoint|` widening, the underflow
//! floor is NORMAL-range (Metal FTZ-safe), the §0 weight-amplified subnormal-flush
//! loss is covered on-device via `flushacc`, and the outward store uses
//! `center ∓ positive radius`. So the returned interval is a SUPERSET of BOTH the
//! true forward range (S1) AND the CPU `propagate_ibp_sound` bound (S2).
//!
//! # Scope of this landing (T1.1)
//! Handles sequential dense chains of [`GpuIbpLayer::Linear`], groups=1
//! [`GpuIbpLayer::Conv2d`] (§3.2), [`GpuIbpLayer::ReLU`], and metadata-only
//! [`GpuIbpLayer::View`] (Flatten/Reshape) — the keystone, the sound conv sibling,
//! the activation, and the shape reinterpretation, which cover flat MLP + plain
//! conv dense chains INCLUDING the ubiquitous flatten-before-FC-head. A grouped
//! Conv2d still returns `Err(UnsupportedOp)` so the caller falls back to the
//! proven-sound CPU loop (verdict-safe). `View` is a pure element-preserving
//! pass-through (row-major flat bounds unchanged, NO dispatch, NO widening) —
//! sound and EXACT, matching the CPU dense-chain fast path which likewise applies
//! no widening to Flatten/Reshape (network/ibp/forward.rs:236); the earlier
//! "1-ULP post-reshape widen" concern applied only to the general CPU loop, not
//! this widening-free dense-chain set. The graph-only sound siblings
//! (MatMul/Add/AvgPool/Transpose/Scale, §3.3–§3.8) live as standalone dispatch
//! helpers in `ibp_ops_sound.rs`; with T1.0 landed, the Add and AvgPool sound
//! shaders are additionally compiled into the graph-DAG sound plan
//! (`ibp_graph_forward_plan_sound.rs`) and are verdict-legal on the default-on
//! sound gate, while MatMul/Transpose/Scale remain oracle-tested only (no
//! `GpuDagIbpOp` kind yet).
//!
//! The FAST (unsound) speed shaders and their pipelines are never touched; the sound
//! pipelines are separate siblings, lazily compiled on first verdict use.

use ny_core::{
    ftz_safe_underflow_floor, nan_propagating_max_zero, nan_propagating_min_zero, GpuIbpLayer,
    GpuIbpResult, NyError, Result,
};

use crate::wgpu_device::sound_consts::{combine_slack_f32, gamma_k_f32};

use super::super::params::{Conv2dIbpSoundParams, LinearIbpSoundParams, ReluIbpParams};
use super::super::{IbpSoundPipelines, WgpuDevice};
use super::ibp_forward::create_buffer;
use super::{gpu_checked_u32, sanitize_readback};

/// One encoded sound-IBP step: a compute-pass dispatch against a prebuilt bind
/// group. `Linear` reads from one ping-pong bank and writes the other; `ReLU`
/// rewrites the current bank in place.
enum SoundStep {
    Linear {
        bind_group: wgpu::BindGroup,
        workgroups: u32,
    },
    Conv2d {
        bind_group: wgpu::BindGroup,
        workgroups: u32,
    },
    ReLU {
        bind_group: wgpu::BindGroup,
        workgroups: u32,
    },
}

impl WgpuDevice {
    /// Lazily-built, reused-forever SOUND IBP forward pipelines (§3.1/§3.7). Built
    /// under the `gpu_serialize` lock held by the enclosing `run_gpu_checked`, so the
    /// one-time compilation is single-threaded and any shader-validation error is
    /// captured (→ CPU fallback) rather than aborting.
    pub(super) fn ibp_sound_pipelines(&self) -> &IbpSoundPipelines {
        self.ibp_sound_pipelines.get_or_init(|| IbpSoundPipelines {
            // bindings 1..7: in_lower(RO), in_upper(RO), wp(RO), wn(RO), bias(RO),
            // out_lower(RW), out_upper(RW) — exactly at Metal's 8-buffer limit.
            linear: self.create_simple_pipeline(
                &super::super::shaders::linear_ibp_sound_source(),
                "linear_ibp_sound",
                &[false, false, false, false, false, true, true],
            ),
            // bindings 1..2: lower(RW), upper(RW) — in-place elementwise.
            relu: self.create_simple_pipeline(
                &super::super::shaders::relu_ibp_sound_source(),
                "relu_ibp_sound",
                &[true, true],
            ),
            // §3.2: same 7-storage binding shape as sound Linear (in_l, in_u, wp,
            // wn, bias RO; out_l, out_u RW).
            conv2d: self.create_simple_pipeline(
                &super::super::shaders::conv2d_ibp_sound_source(),
                "conv2d_ibp_sound",
                &[false, false, false, false, false, true, true],
            ),
            // §3.3: a_l, a_u, b_l, b_u RO; out_l, out_u RW.
            matmul: self.create_simple_pipeline(
                &super::super::shaders::matmul_ibp_sound_source(),
                "matmul_ibp_sound",
                &[false, false, false, false, true, true],
            ),
            // §3.4: in_l, in_u RO; out_l, out_u RW.
            avgpool: self.create_simple_pipeline(
                &super::super::shaders::avgpool_ibp_sound_source(),
                "avgpool_ibp_sound",
                &[false, false, true, true],
            ),
            // §3.5: a_l, a_u, b_l, b_u RO; out_l, out_u RW.
            add: self.create_simple_pipeline(
                &super::super::shaders::add_ibp_sound_source(),
                "add_ibp_sound",
                &[false, false, false, false, true, true],
            ),
            // §3.6: in_l, in_u RO; out_l, out_u RW.
            transpose: self.create_simple_pipeline(
                &super::super::shaders::transpose_ibp_sound_source(),
                "transpose_ibp_sound",
                &[false, false, true, true],
            ),
            // §3.8: in_l, in_u RO; out_l, out_u RW.
            scale: self.create_simple_pipeline(
                &super::super::shaders::scale_ibp_sound_source(),
                "scale_ibp_sound",
                &[false, false, true, true],
            ),
            // T1.2: lower_a, upper_a, window_meta RO; new_lower_a, new_upper_a,
            // err_comb RW (6 storage + params = 7 buffers, Metal-safe).
            maxpool_crown: self.create_simple_pipeline(
                &super::super::shaders::maxpool_crown_sound_source(),
                "maxpool_crown_sound",
                &[false, false, false, true, true, true],
            ),
        })
    }

    /// SOUND GPU IBP forward for a Linear/ReLU dense chain (§6.3). Wraps the encode
    /// in `run_gpu_checked` so any wgpu error → `Err` → CPU sound fallback (never an
    /// abort, never a value from a failed op).
    pub(super) fn ibp_forward_gpu_sound_dispatch(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        // Identity: an empty chain returns the input bounds verbatim (no widening
        // needed; nothing was computed).
        if layers.is_empty() {
            return Ok(GpuIbpResult {
                lower_bounds: input_lower.to_vec(),
                upper_bounds: input_upper.to_vec(),
                output_shape: input_shape.to_vec(),
            });
        }

        // Scope guard: only Linear/ReLU are certified in this landing. Reject
        // everything else UP FRONT so the caller falls back to the CPU sound loop
        // without any GPU work. (View/Conv2d are verdict-safe on CPU.)
        for layer in layers {
            match layer {
                GpuIbpLayer::Linear { .. } | GpuIbpLayer::ReLU { .. } => {}
                GpuIbpLayer::Conv2d { groups, .. } => {
                    if *groups != 1 {
                        return Err(NyError::UnsupportedOp(
                            "sound GPU IBP forward: grouped Conv2d not certified (shader emits a \
                             maximal FALLBACK superset, but the host sizing assumes groups=1); \
                             CPU sound fallback"
                                .into(),
                        ));
                    }
                }
                // View/Flatten/Reshape is metadata-only and element-preserving —
                // certified below (handled as a pure pass-through, matching the CPU
                // dense-chain fast path which applies NO widening to it).
                GpuIbpLayer::View { .. } => {}
            }
        }

        let input_elements = input_shape.iter().product::<usize>();
        if input_lower.len() != input_elements || input_upper.len() != input_elements {
            return Err(NyError::InvalidSpec(format!(
                "sound GPU IBP forward: input length mismatch, shape {input_shape:?} implies \
                 {input_elements} elements, got lower={} upper={}",
                input_lower.len(),
                input_upper.len()
            )));
        }

        self.run_gpu_checked("ibp_forward_gpu_sound", || {
            self.ibp_forward_gpu_sound_inner(layers, input_lower, input_upper, input_shape)
        })
    }

    fn ibp_forward_gpu_sound_inner(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        let pipes = self.ibp_sound_pipelines();
        let f32_size = size_of::<f32>() as u64;

        // Size the ping-pong banks to the widest intermediate width in the chain.
        let input_elements = input_shape.iter().product::<usize>();
        let mut cur_dim = input_elements;
        let mut max_dim = input_elements;
        for layer in layers {
            cur_dim = sound_next_dim(cur_dim, layer)?;
            max_dim = max_dim.max(cur_dim);
        }
        let max_buf_bytes = (max_dim as u64) * f32_size;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let buf_lower_a = create_buffer(&self.device, "ibp_snd_lower_a", max_buf_bytes, usage_rw);
        let buf_upper_a = create_buffer(&self.device, "ibp_snd_upper_a", max_buf_bytes, usage_rw);
        let buf_lower_b = create_buffer(&self.device, "ibp_snd_lower_b", max_buf_bytes, usage_rw);
        let buf_upper_b = create_buffer(&self.device, "ibp_snd_upper_b", max_buf_bytes, usage_rw);

        // Seed bank A with the input bounds.
        self.queue
            .write_buffer(&buf_lower_a, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&buf_upper_a, 0, bytemuck::cast_slice(input_upper));

        // Build per-layer bind groups. The bind groups hold Arc references to the
        // per-layer weight/param buffers, keeping them alive until submit completes;
        // `_keepalive` additionally pins them for clarity.
        let mut steps: Vec<SoundStep> = Vec::with_capacity(layers.len());
        let mut keepalive: Vec<wgpu::Buffer> = Vec::new();
        let mut use_b = false;
        cur_dim = input_elements;
        let mut cur_shape = input_shape.to_vec();

        for layer in layers {
            match layer {
                GpuIbpLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                } => {
                    let in_features = *in_features;
                    let out_features = *out_features;
                    let expected_wlen = in_features.checked_mul(out_features).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "sound ibp: linear weight overflow in={in_features} out={out_features}"
                        ))
                    })?;
                    if weight.len() != expected_wlen {
                        return Err(NyError::shape_mismatch(
                            vec![expected_wlen],
                            vec![weight.len()],
                        ));
                    }
                    if let Some(bias) = bias {
                        if bias.len() != out_features {
                            return Err(NyError::shape_mismatch(
                                vec![out_features],
                                vec![bias.len()],
                            ));
                        }
                    }
                    if in_features == 0 || cur_dim % in_features != 0 {
                        return Err(NyError::shape_mismatch(vec![in_features], vec![cur_dim]));
                    }
                    let batch_size = cur_dim / in_features;
                    let out_elems = batch_size.checked_mul(out_features).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "sound ibp: linear output overflow batch={batch_size} out={out_features}"
                        ))
                    })?;

                    // Weight split: wp = max(W,0) >= 0, wn = min(W,0) <= 0, so
                    // |W| = wp - wn EXACTLY (one is 0). NaN-propagating so a NaN
                    // weight makes both NaN → the shader `is_non_finite` guard fires
                    // → FALLBACK. Matches the fast path's split.
                    let weight_pos: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_max_zero(w))
                        .collect();
                    let weight_neg: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_min_zero(w))
                        .collect();
                    let bias_data: Vec<f32> = match bias {
                        Some(b) => b.to_vec(),
                        None => vec![0.0; out_features],
                    };

                    // Host-side sound error sizing (§3.1). k = reduction length + 3
                    // (dot + {combine, bias-add, final widen-op}).
                    let k = in_features.checked_add(3).ok_or_else(|| {
                        NyError::InvalidSpec("sound ibp linear reduction length overflow".into())
                    })?;
                    let k_u32 = gpu_checked_u32(k, "sound ibp linear k")?;
                    let params = LinearIbpSoundParams {
                        batch_size: gpu_checked_u32(batch_size, "sound ibp linear batch")?,
                        in_features: gpu_checked_u32(in_features, "sound ibp linear in")?,
                        out_features: gpu_checked_u32(out_features, "sound ibp linear out")?,
                        // N = 2·(in_features + 2): the CPU N-D sound path widens by
                        // (in+2) ULPs inside propagate_ibp AND again in
                        // propagate_ibp_sound ⇒ the GPU stands in for a DOUBLE widen.
                        n_ulps: gpu_checked_u32(
                            2usize.saturating_mul(in_features.saturating_add(2)),
                            "sound ibp linear n_ulps",
                        )?,
                        gamma_k: gamma_k_f32(k)?,
                        slack: combine_slack_f32(k)?,
                        additive: ftz_safe_underflow_floor(k_u32),
                        _pad: 0,
                    };

                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_snd_lin_params",
                        size_of::<LinearIbpSoundParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let wbytes = expected_wlen as u64 * f32_size;
                    let wp_buf = create_buffer(
                        &self.device,
                        "ibp_snd_wp",
                        wbytes,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));
                    let wn_buf = create_buffer(
                        &self.device,
                        "ibp_snd_wn",
                        wbytes,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));
                    let bias_buf = create_buffer(
                        &self.device,
                        "ibp_snd_bias",
                        (out_features as u64) * f32_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

                    let (src_lower, src_upper, dst_lower, dst_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b, &buf_lower_a, &buf_upper_a)
                    } else {
                        (&buf_lower_a, &buf_upper_a, &buf_lower_b, &buf_upper_b)
                    };

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_snd_linear_bg"),
                        layout: &pipes.linear.1,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: src_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: src_upper.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wp_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wn_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: bias_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: dst_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: dst_upper.as_entire_binding(),
                            },
                        ],
                    });

                    keepalive.push(params_buf);
                    keepalive.push(wp_buf);
                    keepalive.push(wn_buf);
                    keepalive.push(bias_buf);
                    steps.push(SoundStep::Linear {
                        bind_group,
                        workgroups: gpu_checked_u32(out_elems, "sound ibp linear dispatch")?
                            .div_ceil(64),
                    });

                    use_b = !use_b;
                    cur_dim = out_elems;
                    let Some(last) = cur_shape.last_mut() else {
                        return Err(NyError::InvalidSpec(
                            "sound ibp: linear requires >= 1D shape".into(),
                        ));
                    };
                    if *last != in_features {
                        return Err(NyError::shape_mismatch(vec![in_features], vec![*last]));
                    }
                    *last = out_features;
                }
                GpuIbpLayer::ReLU { num_elements } => {
                    if *num_elements != cur_dim {
                        return Err(NyError::shape_mismatch(vec![cur_dim], vec![*num_elements]));
                    }
                    let params = ReluIbpParams {
                        num_elements: gpu_checked_u32(*num_elements, "sound ibp relu elems")?,
                        _padding: [0; 3],
                    };
                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_snd_relu_params",
                        size_of::<ReluIbpParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let (cur_lower, cur_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b)
                    } else {
                        (&buf_lower_a, &buf_upper_a)
                    };

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_snd_relu_bg"),
                        layout: &pipes.relu.1,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: cur_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cur_upper.as_entire_binding(),
                            },
                        ],
                    });

                    keepalive.push(params_buf);
                    steps.push(SoundStep::ReLU {
                        bind_group,
                        workgroups: gpu_checked_u32(*num_elements, "sound ibp relu dispatch")?
                            .div_ceil(64),
                    });
                    // ReLU is in-place: bank parity and shape unchanged.
                }
                GpuIbpLayer::Conv2d {
                    weight,
                    bias,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    groups,
                    input_h,
                    input_w,
                } => {
                    // groups!=1 was rejected in the scope guard (host sizing assumes
                    // groups=1; the shader's FALLBACK early-out is not exercised here).
                    if *groups != 1 {
                        return Err(NyError::UnsupportedOp(
                            "sound GPU IBP forward: grouped Conv2d reached encode".into(),
                        ));
                    }
                    let (out_channels, in_channels) = (*out_channels, *in_channels);
                    let (kernel_h, kernel_w) = (*kernel_h, *kernel_w);
                    let (input_h, input_w) = (*input_h, *input_w);
                    let macs = in_channels
                        .checked_mul(kernel_h)
                        .and_then(|v| v.checked_mul(kernel_w))
                        .filter(|v| *v != 0)
                        .ok_or_else(|| {
                            NyError::InvalidSpec("sound ibp: conv2d MAC count overflow/zero".into())
                        })?;
                    let expected_wlen = out_channels.checked_mul(macs).ok_or_else(|| {
                        NyError::InvalidSpec("sound ibp: conv2d weight overflow".into())
                    })?;
                    if weight.len() != expected_wlen {
                        return Err(NyError::shape_mismatch(
                            vec![expected_wlen],
                            vec![weight.len()],
                        ));
                    }
                    if let Some(bias) = bias {
                        if bias.len() != out_channels {
                            return Err(NyError::shape_mismatch(
                                vec![out_channels],
                                vec![bias.len()],
                            ));
                        }
                    }
                    let per_batch = in_channels
                        .checked_mul(input_h)
                        .and_then(|v| v.checked_mul(input_w))
                        .filter(|v| *v != 0)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "sound ibp: conv2d input size overflow/zero".into(),
                            )
                        })?;
                    if !cur_dim.is_multiple_of(per_batch) {
                        return Err(NyError::shape_mismatch(vec![per_batch], vec![cur_dim]));
                    }
                    let batch_size = cur_dim / per_batch;
                    let (out_h, out_w) = conv_out_hw(
                        input_h, input_w, kernel_h, kernel_w, *stride_h, *stride_w, *pad_h, *pad_w,
                    )?;
                    let out_elems = batch_size
                        .checked_mul(out_channels)
                        .and_then(|v| v.checked_mul(out_h))
                        .and_then(|v| v.checked_mul(out_w))
                        .ok_or_else(|| {
                            NyError::InvalidSpec("sound ibp: conv2d output size overflow".into())
                        })?;

                    // Weight split (same soundness as Linear: |W| = wp - wn exactly).
                    let weight_pos: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_max_zero(w))
                        .collect();
                    let weight_neg: Vec<f32> = weight
                        .iter()
                        .map(|&w| nan_propagating_min_zero(w))
                        .collect();
                    let bias_data: Vec<f32> = match bias {
                        Some(b) => b.to_vec(),
                        None => vec![0.0; out_channels],
                    };

                    // k = macs + 3; n_ulps = 2·(macs + 2) — full window (padding taps
                    // over-counted ⇒ sound but looser at the border, §3.2).
                    let k = macs.checked_add(3).ok_or_else(|| {
                        NyError::InvalidSpec("sound ibp conv reduction length overflow".into())
                    })?;
                    let k_u32 = gpu_checked_u32(k, "sound ibp conv k")?;
                    let params = Conv2dIbpSoundParams {
                        batch_size: gpu_checked_u32(batch_size, "sound ibp conv batch")?,
                        in_channels: gpu_checked_u32(in_channels, "sound ibp conv in_c")?,
                        out_channels: gpu_checked_u32(out_channels, "sound ibp conv out_c")?,
                        input_h: gpu_checked_u32(input_h, "sound ibp conv in_h")?,
                        input_w: gpu_checked_u32(input_w, "sound ibp conv in_w")?,
                        out_h: gpu_checked_u32(out_h, "sound ibp conv out_h")?,
                        out_w: gpu_checked_u32(out_w, "sound ibp conv out_w")?,
                        kernel_h: gpu_checked_u32(kernel_h, "sound ibp conv k_h")?,
                        kernel_w: gpu_checked_u32(kernel_w, "sound ibp conv k_w")?,
                        stride_h: gpu_checked_u32(*stride_h, "sound ibp conv s_h")?,
                        stride_w: gpu_checked_u32(*stride_w, "sound ibp conv s_w")?,
                        pad_h: gpu_checked_u32(*pad_h, "sound ibp conv p_h")?,
                        pad_w: gpu_checked_u32(*pad_w, "sound ibp conv p_w")?,
                        groups: 1,
                        n_ulps: gpu_checked_u32(
                            2usize.saturating_mul(macs.saturating_add(2)),
                            "sound ibp conv n_ulps",
                        )?,
                        gamma_k: gamma_k_f32(k)?,
                        slack: combine_slack_f32(k)?,
                        additive: ftz_safe_underflow_floor(k_u32),
                        _pad0: 0,
                        _pad1: 0,
                    };

                    let params_buf = create_buffer(
                        &self.device,
                        "ibp_snd_conv_params",
                        size_of::<Conv2dIbpSoundParams>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

                    let wbytes = expected_wlen as u64 * f32_size;
                    let wp_buf = create_buffer(
                        &self.device,
                        "ibp_snd_conv_wp",
                        wbytes,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wp_buf, 0, bytemuck::cast_slice(&weight_pos));
                    let wn_buf = create_buffer(
                        &self.device,
                        "ibp_snd_conv_wn",
                        wbytes,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&wn_buf, 0, bytemuck::cast_slice(&weight_neg));
                    let bias_buf = create_buffer(
                        &self.device,
                        "ibp_snd_conv_bias",
                        (out_channels as u64) * f32_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    self.queue
                        .write_buffer(&bias_buf, 0, bytemuck::cast_slice(&bias_data));

                    let (src_lower, src_upper, dst_lower, dst_upper) = if use_b {
                        (&buf_lower_b, &buf_upper_b, &buf_lower_a, &buf_upper_a)
                    } else {
                        (&buf_lower_a, &buf_upper_a, &buf_lower_b, &buf_upper_b)
                    };

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ibp_snd_conv_bg"),
                        layout: &pipes.conv2d.1,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: src_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: src_upper.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wp_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wn_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: bias_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: dst_lower.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: dst_upper.as_entire_binding(),
                            },
                        ],
                    });

                    keepalive.push(params_buf);
                    keepalive.push(wp_buf);
                    keepalive.push(wn_buf);
                    keepalive.push(bias_buf);
                    steps.push(SoundStep::Conv2d {
                        bind_group,
                        workgroups: gpu_checked_u32(out_elems, "sound ibp conv dispatch")?
                            .div_ceil(64),
                    });

                    use_b = !use_b;
                    cur_dim = out_elems;
                    // Replace the trailing [C, H, W] with [out_channels, out_h, out_w].
                    let ndim = cur_shape.len();
                    if ndim < 3 {
                        return Err(NyError::InvalidSpec(
                            "sound ibp: conv2d requires >= 3D (…, C, H, W) shape".into(),
                        ));
                    }
                    if cur_shape[ndim - 3] != in_channels
                        || cur_shape[ndim - 2] != input_h
                        || cur_shape[ndim - 1] != input_w
                    {
                        return Err(NyError::shape_mismatch(
                            vec![in_channels, input_h, input_w],
                            vec![
                                cur_shape[ndim - 3],
                                cur_shape[ndim - 2],
                                cur_shape[ndim - 1],
                            ],
                        ));
                    }
                    cur_shape[ndim - 3] = out_channels;
                    cur_shape[ndim - 2] = out_h;
                    cur_shape[ndim - 1] = out_w;
                }
                GpuIbpLayer::View { output_shape } => {
                    // Metadata-only reshape: row-major flat bounds are unchanged, so
                    // there is NO dispatch, NO ping-pong toggle, and NO widening — only
                    // the tracked output shape advances. This exactly matches the CPU
                    // dense-chain fast path, where a pure Linear/ReLU/Flatten/Reshape
                    // chain applies no soundness widening (network/ibp/forward.rs:236),
                    // so `GPU == CPU == exact` for the reshape step (still ⊇, sound).
                    let out_elems = output_shape.iter().product::<usize>();
                    if out_elems != cur_dim {
                        return Err(NyError::shape_mismatch(vec![cur_dim], vec![out_elems]));
                    }
                    cur_shape = output_shape.to_vec();
                }
            }
        }

        // Encode all passes into one command buffer (each dispatch in its own pass
        // → an execution barrier between neighbors).
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ibp_snd_encoder"),
            });
        for step in &steps {
            match step {
                SoundStep::Linear {
                    bind_group,
                    workgroups,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_snd_linear_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.linear.0);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroups, 1, 1);
                }
                SoundStep::Conv2d {
                    bind_group,
                    workgroups,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_snd_conv_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.conv2d.0);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroups, 1, 1);
                }
                SoundStep::ReLU {
                    bind_group,
                    workgroups,
                } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ibp_snd_relu_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.relu.0);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(*workgroups, 1, 1);
                }
            }
        }

        let (final_lower, final_upper) = if use_b {
            (&buf_lower_b, &buf_upper_b)
        } else {
            (&buf_lower_a, &buf_upper_a)
        };
        let out_bytes = (cur_dim as u64) * f32_size;
        let staging_lower = create_buffer(
            &self.device,
            "ibp_snd_staging_lower",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = create_buffer(
            &self.device,
            "ibp_snd_staging_upper",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(final_lower, 0, &staging_lower, 0, out_bytes);
        encoder.copy_buffer_to_buffer(final_upper, 0, &staging_upper, 0, out_bytes);

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut batched = WgpuDevice::read_buffers_batched(
            &self.device,
            &[(&staging_lower, cur_dim), (&staging_upper, cur_dim)],
        )?;
        let mut result_upper = batched.pop().expect("2 readbacks");
        let mut result_lower = batched.pop().expect("2 readbacks");

        // Defense-in-depth: NaN/Inf → ±FALLBACK, inverted → widen (matches the fast
        // path). The shader already emits FALLBACK for non-finite / inverted, so this
        // only ever widens, never tightens.
        sanitize_readback(&mut result_lower, &mut result_upper);

        // `keepalive` (and `steps`) held every per-layer buffer alive across submit.
        drop(keepalive);

        Ok(GpuIbpResult {
            lower_bounds: result_lower,
            upper_bounds: result_upper,
            output_shape: cur_shape,
        })
    }
}

/// `(out_h, out_w)` of a groups=1 Conv2d from its (padded, strided) window, with
/// checked arithmetic. Returns an error on an ill-formed (kernel larger than the
/// padded input) or overflowing configuration so the caller falls back to CPU.
fn conv_out_hw(
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(usize, usize)> {
    if stride_h == 0 || stride_w == 0 {
        return Err(NyError::InvalidSpec("sound ibp: conv2d zero stride".into()));
    }
    let padded_h = input_h.checked_add(2 * pad_h);
    let padded_w = input_w.checked_add(2 * pad_w);
    let out_h = padded_h
        .and_then(|p| p.checked_sub(kernel_h))
        .map(|d| d / stride_h + 1)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "sound ibp: conv2d window overflow h in={input_h} k={kernel_h} pad={pad_h}"
            ))
        })?;
    let out_w = padded_w
        .and_then(|p| p.checked_sub(kernel_w))
        .map(|d| d / stride_w + 1)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "sound ibp: conv2d window overflow w in={input_w} k={kernel_w} pad={pad_w}"
            ))
        })?;
    Ok((out_h, out_w))
}

/// Next flat element count after `layer`, for the certified set (Linear/ReLU/Conv2d).
fn sound_next_dim(cur_dim: usize, layer: &GpuIbpLayer) -> Result<usize> {
    match layer {
        GpuIbpLayer::Linear {
            in_features,
            out_features,
            ..
        } => {
            if *in_features == 0 || !cur_dim.is_multiple_of(*in_features) {
                return Err(NyError::shape_mismatch(vec![*in_features], vec![cur_dim]));
            }
            (cur_dim / *in_features)
                .checked_mul(*out_features)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "sound ibp: linear output overflow cur={cur_dim} in={in_features} out={out_features}"
                    ))
                })
        }
        GpuIbpLayer::ReLU { num_elements } => {
            if *num_elements != cur_dim {
                return Err(NyError::shape_mismatch(vec![cur_dim], vec![*num_elements]));
            }
            Ok(cur_dim)
        }
        GpuIbpLayer::Conv2d {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            groups,
            input_h,
            input_w,
            ..
        } => {
            if *groups != 1 {
                return Err(NyError::UnsupportedOp(
                    "sound GPU IBP forward: grouped Conv2d not certified; CPU sound fallback"
                        .into(),
                ));
            }
            let per_batch = in_channels
                .checked_mul(*input_h)
                .and_then(|v| v.checked_mul(*input_w))
                .filter(|v| *v != 0)
                .ok_or_else(|| {
                    NyError::InvalidSpec("sound ibp: conv2d input size overflow/zero".into())
                })?;
            if !cur_dim.is_multiple_of(per_batch) {
                return Err(NyError::shape_mismatch(vec![per_batch], vec![cur_dim]));
            }
            let batch = cur_dim / per_batch;
            let (out_h, out_w) = conv_out_hw(
                *input_h, *input_w, *kernel_h, *kernel_w, *stride_h, *stride_w, *pad_h, *pad_w,
            )?;
            batch
                .checked_mul(*out_channels)
                .and_then(|v| v.checked_mul(out_h))
                .and_then(|v| v.checked_mul(out_w))
                .ok_or_else(|| {
                    NyError::InvalidSpec("sound ibp: conv2d output size overflow".into())
                })
        }
        GpuIbpLayer::View { output_shape } => {
            // Element-preserving reshape: the flat element count is unchanged.
            let out_elems = output_shape.iter().product::<usize>();
            if out_elems != cur_dim {
                return Err(NyError::shape_mismatch(vec![cur_dim], vec![out_elems]));
            }
            Ok(cur_dim)
        }
    }
}
