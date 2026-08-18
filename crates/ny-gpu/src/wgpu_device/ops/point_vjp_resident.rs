// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vjp-resident (#sb-rebank lever 3): device-resident wide-template cache for
//! the batched point-VJP attack fold.
//!
//! ## Why
//!
//! The batched attack (`crown_point_vjp_batched`, #batched-vjp) pays a ~72 ms
//! FIXED cost per step on the soundnessbench chain (measured K=8/16/32/64/128
//! → 88/108/136/194/318 ms/step ≈ 72 ms + 2 ms/lane): every step re-runs
//! `stack_point_vjp_wide_layers` (fresh K-wide slope/intercept `Vec`s, ~250 MB
//! of allocations at K=64), re-creates ~30 wgpu buffers, re-uploads ~300 MB of
//! constant data (zero intercepts, duplicate upper slopes, the all-ones DAZ
//! vector, zero error tails), recompiles the conv reshape/col2im pipelines,
//! and pays ~16 per-layer `queue.submit`s. All of that is TEMPLATE data —
//! only the K ReLU mask slabs and the K spec rows change between steps.
//!
//! This cache keeps the wide template DEVICE-RESIDENT per (template identity,
//! K): buffers created once, static activation slabs uploaded once, per-layer
//! uniforms pre-written once, weights bound from the existing
//! [`resident_weight_buf`](super::super::WgpuDevice::resident_weight_buf)
//! cache. A step uploads ONLY the K mask slabs + the K spec rows and issues
//! ONE submit.
//!
//! ## Bit-identity (the oracle contract)
//!
//! The fold reproduces the EXACT lower-coefficient dispatch sequence of the
//! audited resident backward (`crown_backward_sound_resident_coeff_seeded_err_gather`)
//! restricted to what the VJP reads. The returned gradient is the folded
//! input-level LOWER coefficient; by data flow that stream is produced only by
//!   * the main GEMM `A@W` (same pipeline, same `GemmParams`, same
//!     `select_gemm_dispatch` workgroups),
//!   * the conv reshape → GEMM → col2im chain (same pipelines/params), and
//!   * the resident activation coefficient pass (same shader; `a_out` depends
//!     only on `a_in`, the slope slabs and `beta`, per the WGSL),
//!
//! with identical buffer contents at every pass: the seed rows are the same
//! bytes, mask slabs are the same domain-blocked concatenation the stacker
//! builds (`lower_slope == upper_slope` ⇒ binding ONE slab buffer to both
//! slots reads identical values), `beta` is the same all-zero vector, and the
//! error-stream inputs never feed `a_out`. The upper/error/bias dispatches of
//! the original fold write only buffers the VJP never reads, so skipping them
//! cannot change the returned bits. The oracle test below asserts cached vs
//! uncached gradients are BIT-IDENTICAL on fixed seeds, across repeated calls.
//!
//! ## Keying + keep-alive (soundness of the cache itself)
//!
//! Entries are keyed by the per-layer structural signature: weight `Arc`
//! identity (data pointer + length — the `resident_weights.rs` precedent),
//! all conv/linear dims, mask slot positions/widths, a CONTENT hash of any
//! static (non-mask) activation slabs (they are plain `Vec<f32>` with no
//! identity to key), plus (K, output_dim, input_dim). Every entry retains
//! KEEP-ALIVE clones of the weight `Arc`s so a freed allocation's address can
//! never be recycled into a false pointer hit. The cache is cleared between
//! models via `clear_crown_working_set` alongside the other plan caches.
//!
//! ATTACK-ONLY: gradients steer PGD restarts; no verdict path reads them.
//! `NY_VJP_RESIDENT=0` kills the cache (the un-cached stacking path remains
//! the fallback for any error, resnet templates, and the A/B oracle).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use ny_core::{GpuCrownLayer, NyError, Result};

use super::super::WgpuDevice;
use super::gemm::select_gemm_dispatch;
use super::resident_weights::WeightForm;
use crate::wgpu_device::params::{ConvCol2imParams, ConvReshapeParams, GemmParams};

/// Byte-identical mirror of the resident activation shader's uniform block
/// (`crown_backward_sound_resident::ActParams` / the WGSL `Params` struct).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActParams {
    num_specs: u32,
    num_neurons: u32,
    is_upper: u32,
    additive: f32,
    num_specs_per_dom: u32,
    _p: [u32; 3],
}

/// Whether the resident point-VJP cache is enabled (default ON;
/// `NY_VJP_RESIDENT=0` opts out for A/B differential runs).
pub(crate) fn resident_vjp_enabled() -> bool {
    std::env::var("NY_VJP_RESIDENT").ok().as_deref() != Some("0")
}

/// Cache key: structural template signature + wave width. See module docs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PointVjpPlanKey {
    layers: Vec<VjpLayerKey>,
    mask_positions: Vec<usize>,
    num_specs: usize,
    output_dim: usize,
    input_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// `Conv.dims` ([usize; 12]) makes that variant ~112 B; this key is rebuilt every
// attack step for the cache lookup, so boxing it would add a per-conv-layer heap
// alloc + pointer-chase to that hot path — worse for a Vec that is only ~14 long.
#[allow(clippy::large_enum_variant)]
enum VjpLayerKey {
    Linear {
        w: (usize, usize),
        of: usize,
        if_: usize,
    },
    #[allow(clippy::too_many_arguments)]
    Conv {
        w: (usize, usize),
        dims: [usize; 12],
    },
    /// Mask slot: content replaced per call — width is the whole signature.
    MaskAct { nn: usize },
    /// Static activation: content participates via a hash (no Arc identity).
    StaticAct { nn: usize, content: u64 },
}

fn arc_id(a: &Arc<[f32]>) -> (usize, usize) {
    (Arc::as_ptr(a).cast::<f32>() as usize, a.len())
}

fn hash_f32s(hasher: &mut DefaultHasher, vs: &[f32]) {
    for v in vs {
        hasher.write_u32(v.to_bits());
    }
}

/// One prepared layer of the resident template.
enum VjpLayerPlan {
    Linear {
        w_buf: Arc<wgpu::Buffer>,
        gp: wgpu::Buffer,
        use_small_k: bool,
        wg: (u32, u32),
        // Layer dims are folded into `gp`/`wg` at build; retained for
        // introspection and parity with the cache key.
        #[allow(dead_code)]
        of: usize,
        #[allow(dead_code)]
        if_: usize,
    },
    Conv {
        w_buf: Arc<wgpu::Buffer>,
        crp: wgpu::Buffer,
        gp: wgpu::Buffer,
        ccp: wgpu::Buffer,
        use_small_k: bool,
        wg: (u32, u32),
        reshape_wg: u32,
        col2im_wg: u32,
        // `ic*ih*iw`; folded into `ccp`/`col2im_wg` at build; kept for key parity.
        #[allow(dead_code)]
        out_d: usize,
    },
    /// Per-restart ReLU mask slot: `slab` rewritten each call (domain-blocked
    /// `K × nn`), bound as BOTH lower and upper slopes (identical bytes).
    MaskAct {
        slot: usize,
        slab: wgpu::Buffer,
        actp: wgpu::Buffer,
        nn: usize,
        elem_wg: u32,
    },
    /// Static affine activation: K-tiled slabs uploaded once at build.
    StaticAct {
        ls: wgpu::Buffer,
        us: wgpu::Buffer,
        actp: wgpu::Buffer,
        // Neuron count folded into `actp`/`elem_wg`; kept for parity with `MaskAct`.
        #[allow(dead_code)]
        nn: usize,
        elem_wg: u32,
    },
}

/// A device-resident wide point-VJP template (one per (template, K)).
pub(crate) struct PointVjpResidentEntry {
    layers: Vec<VjpLayerPlan>,
    /// Coefficient ping-pong (`num_specs * max_dim` each).
    la: [wgpu::Buffer; 2],
    /// Error-stream scratch for the activation pass's mandatory err bindings
    /// (written, never read back — the coefficient output is independent).
    err: [wgpu::Buffer; 2],
    conv_reshaped: Option<wgpu::Buffer>,
    conv_gemm: Option<wgpu::Buffer>,
    /// All-zero β (max `K*nn` over activation layers), uploaded once.
    beta_zero: wgpu::Buffer,
    /// MAP_READ staging for the final `K × input_dim` gradient rows.
    stage: wgpu::Buffer,
    /// Reused CPU assembly buffer for the per-step mask slabs.
    slab_staging: Mutex<Vec<f32>>,
    /// KEEP-ALIVE weight `Arc`s: pins every pointer-keyed allocation in the
    /// entry's key for the entry's lifetime (mirrors `resident_weights.rs`).
    #[allow(dead_code)]
    keep_alive: Vec<Arc<[f32]>>,
    /// Conv reshape/col2im pipelines (compiled once per entry).
    conv_pipes: Option<(
        (wgpu::ComputePipeline, wgpu::BindGroupLayout),
        (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    )>,
    num_specs: usize,
    /// Template output dim; retained for parity with the key (the fold reads
    /// only `num_specs`/`input_dim`).
    #[allow(dead_code)]
    output_dim: usize,
    input_dim: usize,
    mask_widths: Vec<usize>,
}

impl PointVjpResidentEntry {
    /// Conservative logical bytes retained by this entry.
    ///
    /// Each layer's `w_buf` is an `Arc` from the resident-weight cache. We count
    /// every reference here even though multiple layers/entries may share the
    /// same allocation; intentional double-counting makes cap admission safely
    /// conservative without relying on raw buffer identity.
    fn retained_device_bytes(&self) -> Result<usize> {
        fn add(total: &mut usize, buffer: &wgpu::Buffer, label: &str) -> Result<()> {
            let bytes = usize::try_from(buffer.size()).map_err(|_| {
                NyError::InternalError(format!("point-VJP buffer `{label}` does not fit in usize"))
            })?;
            *total = total.checked_add(bytes).ok_or_else(|| {
                NyError::InternalError("point-VJP retained byte count overflow".into())
            })?;
            Ok(())
        }

        let mut total = 0usize;
        for buffer in &self.la {
            add(&mut total, buffer, "coefficient")?;
        }
        for buffer in &self.err {
            add(&mut total, buffer, "error")?;
        }
        if let Some(buffer) = self.conv_reshaped.as_ref() {
            add(&mut total, buffer, "conv_reshaped")?;
        }
        if let Some(buffer) = self.conv_gemm.as_ref() {
            add(&mut total, buffer, "conv_gemm")?;
        }
        add(&mut total, &self.beta_zero, "beta_zero")?;
        add(&mut total, &self.stage, "staging")?;

        for layer in &self.layers {
            match layer {
                VjpLayerPlan::Linear { w_buf, gp, .. } => {
                    add(&mut total, w_buf, "linear_weight_shared")?;
                    add(&mut total, gp, "linear_params")?;
                }
                VjpLayerPlan::Conv {
                    w_buf,
                    crp,
                    gp,
                    ccp,
                    ..
                } => {
                    add(&mut total, w_buf, "conv_weight_shared")?;
                    add(&mut total, crp, "conv_reshape_params")?;
                    add(&mut total, gp, "conv_gemm_params")?;
                    add(&mut total, ccp, "conv_col2im_params")?;
                }
                VjpLayerPlan::MaskAct { slab, actp, .. } => {
                    add(&mut total, slab, "mask_slab")?;
                    add(&mut total, actp, "mask_activation_params")?;
                }
                VjpLayerPlan::StaticAct { ls, us, actp, .. } => {
                    add(&mut total, ls, "static_lower_slopes")?;
                    add(&mut total, us, "static_upper_slopes")?;
                    add(&mut total, actp, "static_activation_params")?;
                }
            }
        }
        Ok(total)
    }
}

impl WgpuDevice {
    /// Checked conservative bytes across cached resident point-VJP templates.
    pub(crate) fn point_vjp_resident_cache_bytes(&self) -> Result<usize> {
        let cache = self.point_vjp_resident_plans.lock().map_err(|err| {
            NyError::InternalError(format!("point-vjp plan cache lock poisoned: {err}"))
        })?;
        cache.values().try_fold(0usize, |total, entry| {
            total
                .checked_add(entry.retained_device_bytes()?)
                .ok_or_else(|| NyError::InternalError("point-VJP cache byte count overflow".into()))
        })
    }

    /// Build the cache key, validating the template is a supported pure chain
    /// (Linear / Conv2d / Activation only). `None` ⇒ caller uses the un-cached
    /// stacking path.
    fn point_vjp_plan_key(
        layers_backward: &[GpuCrownLayer],
        mask_positions: &[usize],
        num_specs: usize,
        output_dim: usize,
        input_dim: usize,
    ) -> Option<PointVjpPlanKey> {
        let is_mask = |idx: usize| mask_positions.contains(&idx);
        let mut keys = Vec::with_capacity(layers_backward.len());
        for (idx, layer) in layers_backward.iter().enumerate() {
            keys.push(match layer {
                GpuCrownLayer::Linear {
                    weight,
                    out_features,
                    in_features,
                    ..
                } => VjpLayerKey::Linear {
                    w: arc_id(weight),
                    of: *out_features,
                    if_: *in_features,
                },
                GpuCrownLayer::Conv2d {
                    weight_col,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    ..
                } => VjpLayerKey::Conv {
                    w: arc_id(weight_col),
                    dims: [
                        *out_channels,
                        *in_channels,
                        *kernel_h,
                        *kernel_w,
                        *stride_h,
                        *stride_w,
                        *pad_h,
                        *pad_w,
                        *out_h,
                        *out_w,
                        *in_h,
                        *in_w,
                    ],
                },
                GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_neurons,
                } => {
                    if is_mask(idx) {
                        VjpLayerKey::MaskAct { nn: *num_neurons }
                    } else {
                        // The lean fold folds ONLY the coefficient stream, which
                        // is exact for a static activation IFF its intercepts are
                        // zero (a nonzero intercept feeds the BIAS stream, which
                        // the gradient is independent of — but keep the key
                        // content-complete anyway).
                        let mut h = DefaultHasher::new();
                        hash_f32s(&mut h, lower_slope);
                        hash_f32s(&mut h, upper_slope);
                        hash_f32s(&mut h, lower_intercept);
                        hash_f32s(&mut h, upper_intercept);
                        VjpLayerKey::StaticAct {
                            nn: *num_neurons,
                            content: h.finish(),
                        }
                    }
                }
                // DualAlpha / MaxPool2d: not wide-batchable (matches the
                // stacker's refusal) — and anything unknown fails closed.
                _ => return None,
            });
        }
        Some(PointVjpPlanKey {
            layers: keys,
            mask_positions: mask_positions.to_vec(),
            num_specs,
            output_dim,
            input_dim,
        })
    }

    /// Look up (or build once) the resident entry for this template + K.
    fn point_vjp_resident_entry(
        &self,
        key: PointVjpPlanKey,
        layers_backward: &[GpuCrownLayer],
        mask_positions: &[usize],
        num_specs: usize,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Arc<PointVjpResidentEntry>> {
        {
            let cache = self.point_vjp_resident_plans.lock().map_err(|err| {
                NyError::InternalError(format!("point-vjp plan cache lock poisoned: {err}"))
            })?;
            if let Some(entry) = cache.get(&key) {
                return Ok(Arc::clone(entry));
            }
        }
        let entry = Arc::new(self.build_point_vjp_resident_entry(
            layers_backward,
            mask_positions,
            num_specs,
            output_dim,
            input_dim,
        )?);
        let mut cache = self.point_vjp_resident_plans.lock().map_err(|err| {
            NyError::InternalError(format!("point-vjp plan cache lock poisoned: {err}"))
        })?;
        let canonical = cache.entry(key).or_insert_with(|| Arc::clone(&entry));
        Ok(Arc::clone(canonical))
    }

    /// Validate the chain, create every buffer, pre-write every uniform, and
    /// upload all template-constant data (static slabs, zero β). Runs once per
    /// (template, K); mirrors the dim walk of the audited resident fold.
    fn build_point_vjp_resident_entry(
        &self,
        layers_backward: &[GpuCrownLayer],
        mask_positions: &[usize],
        num_specs: usize,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<PointVjpResidentEntry> {
        // Same walk as the resident fold: dims chain output → input.
        let mut cur = output_dim;
        let mut max_dim = output_dim;
        let mut max_gemm_out = 1usize;
        let mut max_act = 0usize;
        let mut has_conv = false;
        for (idx, layer) in layers_backward.iter().enumerate() {
            match layer {
                GpuCrownLayer::Linear {
                    out_features,
                    in_features,
                    ..
                } => {
                    if *out_features != cur {
                        return Err(NyError::shape_mismatch(vec![cur], vec![*out_features]));
                    }
                    max_dim = max_dim.max(*in_features);
                    cur = *in_features;
                }
                GpuCrownLayer::Activation { num_neurons, .. } => {
                    if *num_neurons != cur {
                        return Err(NyError::shape_mismatch(vec![cur], vec![*num_neurons]));
                    }
                    max_act = max_act.max(*num_neurons);
                }
                GpuCrownLayer::Conv2d {
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    ..
                } => {
                    let in_d = out_channels * out_h * out_w;
                    let out_d = in_channels * in_h * in_w;
                    if in_d != cur {
                        return Err(NyError::shape_mismatch(vec![cur], vec![in_d]));
                    }
                    max_dim = max_dim.max(out_d);
                    max_gemm_out = max_gemm_out
                        .max(num_specs * out_h * out_w * in_channels * kernel_h * kernel_w);
                    has_conv = true;
                    cur = out_d;
                }
                _ => {
                    return Err(NyError::UnsupportedOp(
                        "point-vjp resident: Linear/Activation/Conv2d only".into(),
                    ));
                }
            }
            if mask_positions.contains(&idx) && !matches!(layer, GpuCrownLayer::Activation { .. }) {
                return Err(NyError::InvalidSpec(format!(
                    "point-vjp resident: mask position {idx} is not an Activation layer"
                )));
            }
        }
        if cur != input_dim {
            return Err(NyError::shape_mismatch(vec![input_dim], vec![cur]));
        }
        if let Some(&bad) = mask_positions.iter().find(|&&p| p >= layers_backward.len()) {
            return Err(NyError::InvalidSpec(format!(
                "point-vjp resident: mask position {bad} beyond layer count"
            )));
        }

        let a_elems = num_specs * max_dim;
        let storage = |label: &str, n: usize| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n.max(1) * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        };

        let la = [storage("vjp_la0", a_elems), storage("vjp_la1", a_elems)];
        let err = [storage("vjp_err0", a_elems), storage("vjp_err1", a_elems)];
        let conv_reshaped = has_conv.then(|| storage("vjp_conv_reshaped", a_elems));
        let conv_gemm = has_conv.then(|| storage("vjp_conv_gemm", max_gemm_out));
        // All-zero β for every activation pass (uploaded once; wgpu buffers are
        // zero-initialized on creation, so no explicit fill is needed).
        let beta_zero = storage("vjp_beta_zero", num_specs * max_act.max(1));
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vjp_stage"),
            size: ((num_specs * input_dim).max(1) * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let conv_pipes = has_conv.then(|| {
            (
                self.create_simple_pipeline(
                    super::super::shaders::CONV_RESHAPE_SHADER,
                    "vjp_conv_reshape",
                    &[false, true],
                ),
                self.create_simple_pipeline(
                    super::super::shaders::CONV_COL2IM_SHADER,
                    "vjp_conv_col2im",
                    &[false, true],
                ),
            )
        });

        // Identical FTZ-safe elementwise floor to the audited activation pass.
        let add_e = ny_core::ftz_safe_underflow_floor(1);
        let mk_act_lo = |nn: usize| ActParams {
            num_specs: num_specs as u32,
            num_neurons: nn as u32,
            is_upper: 0,
            additive: add_e,
            // The VJP fold stacks one restart per wide row (num_specs_per_dom=1).
            num_specs_per_dom: 1,
            _p: [0; 3],
        };

        let mut cur = output_dim;
        let mut keep_alive: Vec<Arc<[f32]>> = Vec::new();
        let mut mask_widths = vec![0usize; mask_positions.len()];
        let mut layers = Vec::with_capacity(layers_backward.len());
        for (idx, layer) in layers_backward.iter().enumerate() {
            match layer {
                GpuCrownLayer::Linear {
                    weight,
                    out_features,
                    in_features,
                    ..
                } => {
                    let (of, if_) = (*out_features, *in_features);
                    let w_buf = self.resident_weight_buf(weight, WeightForm::Raw)?;
                    keep_alive.push(Arc::clone(weight));
                    let disp = select_gemm_dispatch(num_specs as u32, of as u32, if_ as u32);
                    let gp = uniform(
                        "vjp_gp",
                        bytemuck::bytes_of(&GemmParams {
                            m: num_specs as u32,
                            k: of as u32,
                            n: if_ as u32,
                            _padding: 0,
                        }),
                    );
                    layers.push(VjpLayerPlan::Linear {
                        w_buf,
                        gp,
                        use_small_k: disp.use_small_k,
                        wg: (disp.wg_x, disp.wg_y),
                        of,
                        if_,
                    });
                    cur = if_;
                }
                GpuCrownLayer::Conv2d {
                    weight_col,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    ..
                } => {
                    let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
                    let (oh, ow, ih, iw) = (*out_h, *out_w, *in_h, *in_w);
                    let out_d = ic * ih * iw;
                    let spatial = oh * ow;
                    let kernel_cols = ic * kh * kw;
                    let (m, k, n) = (num_specs * spatial, oc, kernel_cols);
                    let w_buf = self.resident_weight_buf(weight_col, WeightForm::Raw)?;
                    keep_alive.push(Arc::clone(weight_col));
                    let disp = select_gemm_dispatch(m as u32, k as u32, n as u32);
                    let crp = uniform(
                        "vjp_crp",
                        bytemuck::bytes_of(&ConvReshapeParams {
                            num_specs: num_specs as u32,
                            out_channels: oc as u32,
                            spatial: spatial as u32,
                            _padding: 0,
                        }),
                    );
                    let gp = uniform(
                        "vjp_conv_gp",
                        bytemuck::bytes_of(&GemmParams {
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            _padding: 0,
                        }),
                    );
                    let ccp = uniform(
                        "vjp_ccp",
                        bytemuck::bytes_of(&ConvCol2imParams {
                            num_specs: num_specs as u32,
                            flat_input_dim: out_d as u32,
                            out_h: oh as u32,
                            out_w: ow as u32,
                            in_channels: ic as u32,
                            in_h: ih as u32,
                            in_w: iw as u32,
                            kernel_h: kh as u32,
                            kernel_w: kw as u32,
                            stride_h: *stride_h as u32,
                            stride_w: *stride_w as u32,
                            pad_h: *pad_h as u32,
                            pad_w: *pad_w as u32,
                            kernel_cols: kernel_cols as u32,
                            _padding2: [0; 2],
                        }),
                    );
                    layers.push(VjpLayerPlan::Conv {
                        w_buf,
                        crp,
                        gp,
                        ccp,
                        use_small_k: disp.use_small_k,
                        wg: (disp.wg_x, disp.wg_y),
                        reshape_wg: ((num_specs * spatial * oc) as u32).div_ceil(256),
                        col2im_wg: ((num_specs * out_d) as u32).div_ceil(256),
                        out_d,
                    });
                    cur = out_d;
                }
                GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_neurons,
                } => {
                    let nn = *num_neurons;
                    let elem_wg = ((num_specs * nn) as u32).div_ceil(256);
                    let actp = uniform("vjp_actp", bytemuck::bytes_of(&mk_act_lo(nn)));
                    if let Some(slot) = mask_positions.iter().position(|&p| p == idx) {
                        mask_widths[slot] = nn;
                        layers.push(VjpLayerPlan::MaskAct {
                            slot,
                            slab: storage("vjp_mask_slab", num_specs * nn),
                            actp,
                            nn,
                            elem_wg,
                        });
                    } else {
                        // Static affine activation: K-tiled slabs, uploaded once.
                        if lower_slope.len() != nn
                            || upper_slope.len() != nn
                            || lower_intercept.len() != nn
                            || upper_intercept.len() != nn
                        {
                            return Err(NyError::shape_mismatch(vec![nn], vec![lower_slope.len()]));
                        }
                        let tile = |v: &[f32]| {
                            let mut w = Vec::with_capacity(num_specs * nn);
                            for _ in 0..num_specs {
                                w.extend_from_slice(v);
                            }
                            w
                        };
                        let ls = storage("vjp_static_ls", num_specs * nn);
                        let us = storage("vjp_static_us", num_specs * nn);
                        self.queue
                            .write_buffer(&ls, 0, bytemuck::cast_slice(&tile(lower_slope)));
                        self.queue
                            .write_buffer(&us, 0, bytemuck::cast_slice(&tile(upper_slope)));
                        layers.push(VjpLayerPlan::StaticAct {
                            ls,
                            us,
                            actp,
                            nn,
                            elem_wg,
                        });
                    }
                    // dim unchanged
                    let _ = cur;
                }
                _ => unreachable!("validated by the dim walk above"),
            }
        }
        debug_assert_eq!(cur, input_dim);

        self.point_vjp_resident_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(PointVjpResidentEntry {
            layers,
            la,
            err,
            conv_reshaped,
            conv_gemm,
            beta_zero,
            stage,
            slab_staging: Mutex::new(Vec::new()),
            keep_alive,
            conv_pipes,
            num_specs,
            output_dim,
            input_dim,
            mask_widths,
        })
    }

    /// The resident wide point-VJP fold: upload the K spec rows + K mask
    /// slabs, encode the whole lower-coefficient chain in ONE submit, read
    /// back the `K × input_dim` gradient rows.
    ///
    /// Returns the same flat layout as
    /// [`crown_backward_gpu_point_vjp_wide_inner`](WgpuDevice::crown_backward_gpu_point_vjp_wide_inner)'s
    /// `input_coeff` channel. Any `Err` ⇒ the caller falls back to the
    /// un-cached stacking path (never the sequential cliff directly).
    pub(crate) fn crown_point_vjp_batched_resident(
        &self,
        layers_backward: &[GpuCrownLayer],
        mask_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let k = masks.len();
        if layers_backward.is_empty() || k == 0 || input_dim == 0 || output_dim == 0 {
            return Err(NyError::InvalidSpec(
                "point-vjp resident: empty layers/masks or zero dims".into(),
            ));
        }
        if spec_rows.len() != k * output_dim {
            return Err(NyError::shape_mismatch(
                vec![k, output_dim],
                vec![spec_rows.len()],
            ));
        }
        for mk in masks {
            if mk.len() != mask_positions.len() {
                return Err(NyError::shape_mismatch(
                    vec![mask_positions.len()],
                    vec![mk.len()],
                ));
            }
        }
        let key =
            Self::point_vjp_plan_key(layers_backward, mask_positions, k, output_dim, input_dim)
                .ok_or_else(|| {
                    NyError::UnsupportedOp("point-vjp resident: unsupported template".into())
                })?;

        self.run_gpu_checked("crown_point_vjp_batched_resident", || {
            let entry = self.point_vjp_resident_entry(
                key.clone(),
                layers_backward,
                mask_positions,
                k,
                output_dim,
                input_dim,
            )?;

            // Per-step uploads: seed rows + the K mask slabs (domain-blocked
            // concatenation, identical bytes to `stack_point_vjp_wide_layers`).
            self.queue
                .write_buffer(&entry.la[0], 0, bytemuck::cast_slice(spec_rows));
            {
                let mut staging = entry.slab_staging.lock().map_err(|err| {
                    NyError::InternalError(format!("point-vjp slab staging poisoned: {err}"))
                })?;
                for layer in &entry.layers {
                    let VjpLayerPlan::MaskAct { slot, slab, nn, .. } = layer else {
                        continue;
                    };
                    staging.clear();
                    staging.reserve(k * nn);
                    for mk in masks {
                        let m = &mk[*slot];
                        if m.len() != *nn {
                            return Err(NyError::shape_mismatch(vec![*nn], vec![m.len()]));
                        }
                        staging.extend_from_slice(m);
                    }
                    self.queue
                        .write_buffer(slab, 0, bytemuck::cast_slice(&staging[..]));
                }
            }

            // Sanity: the entry's widths must match this call's masks (the key
            // encodes them, so a mismatch is a logic error — fail closed).
            for (slot, w) in entry.mask_widths.iter().enumerate() {
                if masks.first().map(|mk| mk[slot].len()) != Some(*w) {
                    return Err(NyError::InvalidSpec(
                        "point-vjp resident: mask width mismatch vs cached template".into(),
                    ));
                }
            }

            // ONE encoder for the whole lower-coefficient chain.
            let act_pipe = &self.resident_backward_pipelines().act;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vjp_resident"),
                });
            let mut ping = 0usize;
            for layer in &entry.layers {
                match layer {
                    VjpLayerPlan::Linear {
                        w_buf,
                        gp,
                        use_small_k,
                        wg,
                        ..
                    } => {
                        let pipe = if *use_small_k {
                            &self.gemm_f32_small_k_pipeline
                        } else {
                            &self.gemm_f32_pipeline
                        };
                        self.pass_gemm(
                            &mut encoder,
                            pipe,
                            gp,
                            &entry.la[ping],
                            w_buf,
                            &entry.la[1 - ping],
                            wg.0,
                            wg.1,
                        );
                        ping = 1 - ping;
                    }
                    VjpLayerPlan::Conv {
                        w_buf,
                        crp,
                        gp,
                        ccp,
                        use_small_k,
                        wg,
                        reshape_wg,
                        col2im_wg,
                        ..
                    } => {
                        let (rp, cp) = entry.conv_pipes.as_ref().ok_or_else(|| {
                            NyError::InternalError("point-vjp resident: conv pipes missing".into())
                        })?;
                        let pipe = if *use_small_k {
                            &self.gemm_f32_small_k_pipeline
                        } else {
                            &self.gemm_f32_pipeline
                        };
                        let reshaped = entry.conv_reshaped.as_ref().expect("conv scratch");
                        let gemm_out = entry.conv_gemm.as_ref().expect("conv scratch");
                        self.pass_simple(
                            &mut encoder,
                            rp,
                            crp,
                            &[&entry.la[ping], reshaped],
                            *reshape_wg,
                        );
                        self.pass_gemm(
                            &mut encoder,
                            pipe,
                            gp,
                            reshaped,
                            w_buf,
                            gemm_out,
                            wg.0,
                            wg.1,
                        );
                        self.pass_simple(
                            &mut encoder,
                            cp,
                            ccp,
                            &[gemm_out, &entry.la[1 - ping]],
                            *col2im_wg,
                        );
                        ping = 1 - ping;
                    }
                    VjpLayerPlan::MaskAct {
                        slab,
                        actp,
                        elem_wg,
                        ..
                    } => {
                        // lower==upper slopes: ONE slab bound to both slots
                        // (identical reads to two identical buffers).
                        self.pass_simple(
                            &mut encoder,
                            act_pipe,
                            actp,
                            &[
                                &entry.la[ping],
                                &entry.err[ping],
                                slab,
                                slab,
                                &entry.la[1 - ping],
                                &entry.err[1 - ping],
                                &entry.beta_zero,
                            ],
                            *elem_wg,
                        );
                        ping = 1 - ping;
                    }
                    VjpLayerPlan::StaticAct {
                        ls,
                        us,
                        actp,
                        elem_wg,
                        ..
                    } => {
                        self.pass_simple(
                            &mut encoder,
                            act_pipe,
                            actp,
                            &[
                                &entry.la[ping],
                                &entry.err[ping],
                                ls,
                                us,
                                &entry.la[1 - ping],
                                &entry.err[1 - ping],
                                &entry.beta_zero,
                            ],
                            *elem_wg,
                        );
                        ping = 1 - ping;
                    }
                }
            }
            let out_elems = entry.num_specs * entry.input_dim;
            encoder.copy_buffer_to_buffer(
                &entry.la[ping],
                0,
                &entry.stage,
                0,
                (out_elems * size_of::<f32>()) as u64,
            );
            self.queue.submit(Some(encoder.finish()));
            let coeff = Self::read_buffer(&self.device, &entry.stage, out_elems)?;
            Ok((0..entry.num_specs)
                .map(|d| coeff[d * entry.input_dim..(d + 1) * entry.input_dim].to_vec())
                .collect())
        })
    }

    /// Clear the resident point-VJP plan cache (buffers + keep-alive `Arc`s).
    /// Called from `clear_crown_working_set` between models.
    pub(crate) fn clear_point_vjp_resident_plans(&self) -> Result<()> {
        let mut cache = self.point_vjp_resident_plans.lock().map_err(|err| {
            NyError::InternalError(format!("point-vjp plan cache lock poisoned: {err}"))
        })?;
        cache.clear();
        Ok(())
    }

    /// Number of cached resident VJP templates (test-only introspection).
    // Consumed by the in-file gpu_tests oracle; unused in a lib build that
    // enables `gpu-tests` without `test`, hence the allow.
    #[allow(dead_code)]
    #[cfg(any(test, feature = "gpu-tests"))]
    pub(crate) fn point_vjp_resident_cache_len(&self) -> usize {
        self.point_vjp_resident_plans
            .lock()
            .map(|c| c.len())
            .unwrap_or(0)
    }

    /// Total resident VJP template builds (cache misses; test-only).
    // Consumed by the in-file gpu_tests oracle; unused in a lib build that
    // enables `gpu-tests` without `test`, hence the allow.
    #[allow(dead_code)]
    #[cfg(any(test, feature = "gpu-tests"))]
    pub(crate) fn point_vjp_resident_build_count(&self) -> usize {
        self.point_vjp_resident_builds
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The shared cache-map type stored on [`WgpuDevice`].
pub(crate) type PointVjpResidentPlans = Mutex<HashMap<PointVjpPlanKey, Arc<PointVjpResidentEntry>>>;

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};
    use ny_test_utils::env::ScopedEnvVar;

    /// Deterministic xorshift (no dev-dep).
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn linear(rng: &mut Rng, of: usize, if_: usize) -> GpuCrownLayer {
        let w: Vec<f32> = (0..of * if_).map(|_| rng.next_f32()).collect();
        GpuCrownLayer::Linear {
            weight: Arc::from(w),
            bias: Some(Arc::from(
                (0..of).map(|_| rng.next_f32()).collect::<Vec<f32>>(),
            )),
            out_features: of,
            in_features: if_,
            cert_err: Default::default(),
        }
    }

    fn relu_placeholder(nn: usize) -> GpuCrownLayer {
        GpuCrownLayer::Activation {
            lower_slope: vec![0.0; nn],
            upper_slope: vec![0.0; nn],
            lower_intercept: vec![0.0; nn],
            upper_intercept: vec![0.0; nn],
            num_neurons: nn,
        }
    }

    fn conv(
        rng: &mut Rng,
        oc: usize,
        ic: usize,
        khw: usize,
        stride: usize,
        pad: usize,
        in_hw: usize,
    ) -> (GpuCrownLayer, usize) {
        let out_hw = (in_hw + 2 * pad - khw) / stride + 1;
        let w: Vec<f32> = (0..oc * ic * khw * khw).map(|_| rng.next_f32()).collect();
        (
            GpuCrownLayer::Conv2d {
                weight_col: Arc::from(w),
                bias_expanded: None,
                out_channels: oc,
                in_channels: ic,
                kernel_h: khw,
                kernel_w: khw,
                stride_h: stride,
                stride_w: stride,
                pad_h: pad,
                pad_w: pad,
                out_h: out_hw,
                out_w: out_hw,
                in_h: in_hw,
                in_w: in_hw,
                cert_err: Default::default(),
            },
            oc * out_hw * out_hw,
        )
    }

    /// A soundnessbench-SHAPED chain in BACKWARD order (output → input):
    /// Gemm(384←384), [ReLU, Conv]×3, ReLU, Gemm(12288… scaled down ←input).
    /// Returns (layers_backward, mask_positions, output_dim, input_dim,
    /// per-mask-slot widths).
    fn sb_shaped_template(
        rng: &mut Rng,
    ) -> (Vec<GpuCrownLayer>, Vec<usize>, usize, usize, Vec<usize>) {
        // Forward: in 32 → Linear → 3*8*8 → reshape → conv(3→8,1x1) → ReLU
        // → conv(8→8,3x3 pad1) → ReLU → conv(8→8,1x1 stride2) → ReLU
        // → flatten → Linear(8*4*4=128 → 24).
        let in_dim = 32usize;
        let (c1, c1_out) = conv(rng, 8, 3, 1, 1, 0, 8); // 8x8 → 512
        let (c2, c2_out) = conv(rng, 8, 8, 3, 1, 1, 8); // 512
        let (c3, c3_out) = conv(rng, 8, 8, 1, 2, 0, 8); // 4x4 → 128
        let l_first = linear(rng, 3 * 8 * 8, in_dim); // 192
        let l_last = linear(rng, 24, c3_out);
        // Backward order with masks after (= backward-before) each conv/linear.
        let layers = vec![
            l_last,
            relu_placeholder(c3_out),
            c3,
            relu_placeholder(c2_out),
            c2,
            relu_placeholder(c1_out),
            c1,
            relu_placeholder(3 * 8 * 8),
            l_first,
        ];
        let mask_positions = vec![1, 3, 5, 7];
        let widths = vec![c3_out, c2_out, c1_out, 3 * 8 * 8];
        (layers, mask_positions, 24, in_dim, widths)
    }

    fn random_masks(rng: &mut Rng, k: usize, widths: &[usize]) -> Vec<Vec<Vec<f32>>> {
        (0..k)
            .map(|_| {
                widths
                    .iter()
                    .map(|&nn| {
                        (0..nn)
                            .map(|_| if rng.next_f32() > 0.0 { 1.0 } else { 0.0 })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Run the legacy stacked VJP as a raw coefficient fold.  The production
    /// point-VJP helper also concretizes bounds that its caller discards; armed
    /// C1 correctly refuses that unworded verdict boundary, whereas this test
    /// needs only the pre-concretization coefficients it compares below.
    fn uncached_point_vjp_coeff(
        device: &WgpuDevice,
        wide: &[GpuCrownLayer],
        seed: &ny_core::GpuCrownSeed,
    ) -> Vec<f32> {
        let zero_a_err = vec![0.0f32; seed.lower_a.len()];
        let zero_b_err = vec![0.0f32; seed.num_specs];
        device
            .crown_backward_sound_resident_coeff_seeded_err_gather(
                wide,
                &seed.lower_a,
                &seed.upper_a,
                &zero_a_err,
                &zero_a_err,
                &seed.lower_b,
                &seed.upper_b,
                &zero_b_err,
                &zero_b_err,
                seed.num_specs,
                1,
                seed.current_dim,
                &[],
                &[],
                &[],
            )
            .expect("uncached coefficient fold")
            .lower_a
    }

    /// THE ORACLE (#vjp-resident): cached vs un-cached gradients are
    /// BIT-IDENTICAL on fixed seeds, across repeated calls with fresh masks
    /// (cache-hit path), for a soundnessbench-shaped conv chain and a
    /// static-activation chain. Also proves the cache builds ONCE.
    #[test]
    fn resident_vjp_gradients_bit_identical_to_uncached() {
        let _g = gpu_test_serial_guard();
        // This diagnostic compares raw coefficient values and cache behavior,
        // not verdict authority. Disable words to isolate the value-path cache
        // from the independently tested Conv receipt channel.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        device.clear_crown_working_set().expect("clear");

        let mut rng = Rng(0x5bba_2026 ^ 0xdead_beef);
        let (layers, mask_positions, output_dim, input_dim, widths) = sb_shaped_template(&mut rng);
        let k = 16usize;

        let builds0 = device.point_vjp_resident_build_count();
        let mut timings = (0.0f64, 0.0f64);
        let mut warm_rounds = 0usize;
        for round in 0..3 {
            let masks = random_masks(&mut rng, k, &widths);
            let spec_rows: Vec<f32> = (0..k * output_dim).map(|_| rng.next_f32()).collect();

            // Un-cached reference: the audited stacking path (exactly what the
            // trait fold runs with NY_VJP_RESIDENT=0 / on any resident error).
            // Use its raw coefficient entry: this VJP oracle never consumes
            // bounds, and armed C1 rightly refuses an unworded concretization.
            let t0 = std::time::Instant::now();
            let wide = super::super::crown_backward::stack_point_vjp_wide_layers(
                &layers,
                &mask_positions,
                &masks,
            )
            .expect("stack");
            let seed = ny_core::GpuCrownSeed {
                lower_a: spec_rows.clone().into(),
                upper_a: spec_rows.clone().into(),
                lower_b: vec![0.0; k].into(),
                upper_b: vec![0.0; k].into(),
                num_specs: k,
                current_dim: output_dim,
            };
            let coeff = uncached_point_vjp_coeff(&device, &wide, &seed);
            let reference: Vec<Vec<f32>> = (0..k)
                .map(|d| coeff[d * input_dim..(d + 1) * input_dim].to_vec())
                .collect();
            let t_uncached = t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let cached = device
                .crown_point_vjp_batched_resident(
                    &layers,
                    &mask_positions,
                    &masks,
                    &spec_rows,
                    output_dim,
                    input_dim,
                )
                .expect("resident fold");
            let t_cached = t1.elapsed().as_secs_f64();
            if round > 0 {
                timings.0 += t_uncached;
                timings.1 += t_cached;
                warm_rounds += 1;
            }

            assert_eq!(cached.len(), reference.len(), "round {round}: K");
            for (lane, (c, r)) in cached.iter().zip(reference.iter()).enumerate() {
                let cb: Vec<u32> = c.iter().map(|v| v.to_bits()).collect();
                let rb: Vec<u32> = r.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    cb, rb,
                    "round {round} lane {lane}: cached gradient bits diverge from un-cached"
                );
            }
        }
        assert_eq!(
            device.point_vjp_resident_build_count(),
            builds0 + 1,
            "template must build exactly once across repeated steps"
        );
        eprintln!(
            "vjp-resident oracle: warm rounds avg uncached {:.1}ms vs cached {:.1}ms",
            timings.0 * 1000.0 / warm_rounds.max(1) as f64,
            timings.1 * 1000.0 / warm_rounds.max(1) as f64
        );

        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.point_vjp_resident_cache_len(), 0);
    }

    /// Throughput probe at the EXACT soundnessbench template dims (synthetic
    /// weights, K=64): the fixed-overhead claim behind #sb-rebank lever 3.
    /// Bit-identity is asserted here too (same oracle, production shape).
    /// Target: cached step ≤ uncached step − ~25 ms (the stack/upload/submit
    /// overhead); the assert only requires "strictly faster" so adapter noise
    /// can't flake CI — the printed numbers feed the measurement report.
    #[test]
    fn resident_vjp_soundnessbench_dims_throughput_probe() {
        let _g = gpu_test_serial_guard();
        // This is a raw throughput/bit-identity probe, not an authority oracle.
        // Disable words so the measurement isolates VJP caching overhead.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        device.clear_crown_working_set().expect("clear");

        let mut rng = Rng(0x5b_d105);
        // Backward order: Gemm(384<-384), Conv(model.10) ... Conv(model.0),
        // ReLU masks at the 6 forward ReLUs, Gemm(12288<-128).
        let (c10, c10_out) = conv(&mut rng, 24, 24, 1, 2, 0, 8); // 4x4->? in 8x8: out_hw=(8-1)/2+1=4 ✓ enters 384
        let (c8, c8_out) = conv(&mut rng, 24, 24, 1, 2, 0, 16); // enters 1536
        let (c6, c6_out) = conv(&mut rng, 24, 24, 1, 2, 0, 32); // enters 6144
        let (c4, c4_out) = conv(&mut rng, 24, 24, 1, 2, 0, 64); // enters 24576
        let (c2, c2_out) = conv(&mut rng, 24, 24, 3, 1, 1, 64); // 98304
        let (c0, _c0_out) = conv(&mut rng, 24, 3, 1, 1, 0, 64); // enters 98304, exits 12288
        assert_eq!(
            (c10_out, c8_out, c6_out, c4_out, c2_out),
            (384, 1536, 6144, 24576, 98304)
        );
        let layers = vec![
            linear(&mut rng, 384, 384),
            c10,
            relu_placeholder(1536),
            c8,
            relu_placeholder(6144),
            c6,
            relu_placeholder(24576),
            c4,
            relu_placeholder(98304),
            c2,
            relu_placeholder(98304),
            c0,
            relu_placeholder(12288),
            linear(&mut rng, 12288, 128),
        ];
        let mask_positions = vec![2, 4, 6, 8, 10, 12];
        let widths = vec![1536usize, 6144, 24576, 98304, 98304, 12288];
        let (output_dim, input_dim, k) = (384usize, 128usize, 64usize);

        let mut t_uncached = 0.0f64;
        let mut t_cached = 0.0f64;
        let mut warm = 0usize;
        for round in 0..3 {
            let masks = random_masks(&mut rng, k, &widths);
            let spec_rows: Vec<f32> = (0..k * output_dim).map(|_| rng.next_f32()).collect();

            let t0 = std::time::Instant::now();
            let wide = super::super::crown_backward::stack_point_vjp_wide_layers(
                &layers,
                &mask_positions,
                &masks,
            )
            .expect("stack");
            let seed = ny_core::GpuCrownSeed {
                lower_a: spec_rows.clone().into(),
                upper_a: spec_rows.clone().into(),
                lower_b: vec![0.0; k].into(),
                upper_b: vec![0.0; k].into(),
                num_specs: k,
                current_dim: output_dim,
            };
            let coeff = uncached_point_vjp_coeff(&device, &wide, &seed);
            let reference: Vec<Vec<f32>> = (0..k)
                .map(|d| coeff[d * input_dim..(d + 1) * input_dim].to_vec())
                .collect();
            let d_uncached = t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let cached = device
                .crown_point_vjp_batched_resident(
                    &layers,
                    &mask_positions,
                    &masks,
                    &spec_rows,
                    output_dim,
                    input_dim,
                )
                .expect("resident fold");
            let d_cached = t1.elapsed().as_secs_f64();

            for (lane, (c, r)) in cached.iter().zip(reference.iter()).enumerate() {
                let cb: Vec<u32> = c.iter().map(|v| v.to_bits()).collect();
                let rb: Vec<u32> = r.iter().map(|v| v.to_bits()).collect();
                assert_eq!(cb, rb, "round {round} lane {lane}: bits diverge at sb dims");
            }
            if round > 0 {
                t_uncached += d_uncached;
                t_cached += d_cached;
                warm += 1;
            }
        }
        let (u_ms, c_ms) = (
            t_uncached * 1000.0 / warm as f64,
            t_cached * 1000.0 / warm as f64,
        );
        eprintln!(
            "vjp-resident sb-dims probe (K={k}): uncached {u_ms:.1}ms/step vs cached \
             {c_ms:.1}ms/step (gpu_vjp phase only; production step adds the CPU forward)"
        );
        assert!(
            c_ms < u_ms,
            "resident fold must beat the per-step stacking path ({c_ms:.1} !< {u_ms:.1})"
        );

        device.clear_crown_working_set().expect("clear");
    }

    /// Static (non-mask) activations: tiled-once slabs must still produce
    /// bit-identical gradients, and a CONTENT change in the static slab must
    /// key a NEW entry (no stale-slab reuse).
    #[test]
    fn resident_vjp_static_activation_and_content_keying() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        device.clear_crown_working_set().expect("clear");

        let mut rng = Rng(0x0dd_ba11);
        let l1 = linear(&mut rng, 12, 10);
        let l2 = linear(&mut rng, 8, 12);
        let static_act = |scale: f32| GpuCrownLayer::Activation {
            lower_slope: (0..12).map(|i| scale * (i as f32 * 0.1 - 0.5)).collect(),
            upper_slope: (0..12).map(|i| scale * (i as f32 * 0.1 - 0.5)).collect(),
            lower_intercept: vec![0.0; 12],
            upper_intercept: vec![0.0; 12],
            num_neurons: 12,
        };
        let k = 4usize;
        let masks: Vec<Vec<Vec<f32>>> = (0..k).map(|_| Vec::new()).collect(); // no mask slots
        let spec_rows: Vec<f32> = (0..k * 8).map(|_| rng.next_f32()).collect();

        let grads = |layers: &[GpuCrownLayer]| -> Vec<Vec<f32>> {
            device
                .crown_point_vjp_batched_resident(layers, &[], &masks, &spec_rows, 8, 10)
                .expect("resident fold")
        };
        let uncached = |layers: &[GpuCrownLayer]| -> Vec<Vec<f32>> {
            let wide =
                super::super::crown_backward::stack_point_vjp_wide_layers(layers, &[], &masks)
                    .expect("stack");
            let seed = ny_core::GpuCrownSeed {
                lower_a: spec_rows.clone().into(),
                upper_a: spec_rows.clone().into(),
                lower_b: vec![0.0; k].into(),
                upper_b: vec![0.0; k].into(),
                num_specs: k,
                current_dim: 8,
            };
            let dummy = vec![0.0f32; k * 10];
            let coeff = device
                .crown_backward_gpu_point_vjp_wide_inner(
                    &[ny_core::GpuResnetSegment::Chain(wide)],
                    &seed,
                    1,
                    &dummy,
                    &dummy,
                )
                .expect("uncached fold");
            (0..k)
                .map(|d| coeff[d * 10..(d + 1) * 10].to_vec())
                .collect()
        };

        let a_layers = vec![l2.clone(), static_act(1.0), l1.clone()];
        let b_layers = vec![l2, static_act(2.0), l1];

        let bits = |g: &Vec<Vec<f32>>| -> Vec<Vec<u32>> {
            g.iter()
                .map(|v| v.iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        let (ga, ua) = (grads(&a_layers), uncached(&a_layers));
        assert_eq!(bits(&ga), bits(&ua), "static-act A: cached != uncached");
        // Content change with SAME shapes/weights: must NOT hit A's entry.
        let (gb, ub) = (grads(&b_layers), uncached(&b_layers));
        assert_eq!(bits(&gb), bits(&ub), "static-act B: cached != uncached");
        assert_ne!(bits(&ga), bits(&gb), "A/B slabs differ, gradients must too");
        assert_eq!(device.point_vjp_resident_cache_len(), 2);

        device.clear_crown_working_set().expect("clear");
    }
}
