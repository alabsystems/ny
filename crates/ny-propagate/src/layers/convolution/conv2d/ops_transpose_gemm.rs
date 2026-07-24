// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use faer::Mat;
use ndarray::{Array2, ArrayD};
use ny_core::{checked_shape_product, ConvTranspose2dParams, GemmEngine, NyError, Result};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

use crate::faer_parallelism::mat_mul;

const DEADLINE_GEMM_ROW_CHUNK: usize = 256;

/// Hard backstop site identifier reported when the Conv2d CROWN backward
/// coefficient buffer would exceed the per-buffer memory cap.
const CONV_CROWN_BACKWARD_SITE: &str = "conv2d::ops_transpose_gemm::backward_result";
const CONV_CROWN_BACKWARD_SCRATCH_SITE: &str = "conv2d::ops_transpose_gemm::backward_scratch";

/// Floor for the host-RAM-adaptive per-buffer Conv2d CROWN-backward policy cap
/// in MiB (#conv-crown-oom). This is also the fallback policy when total system
/// RAM cannot be determined (e.g. `/proc/meminfo` unreadable/unparseable).
///
/// Each dense backward coefficient buffer is `num_objectives * conv_in_size`
/// `f32` entries. On deep ResNet conv DAGs (e.g. CIFAR100 ResNet) the mid-size
/// intermediate conv targets (target_dim 4096-16384) slip under the
/// `[target_dim x target_dim]` identity gate yet materialize a multi-GiB
/// `[target_dim x conv_in_size]` matrix here, OOMing the process. This cap is a
/// hard, sound backstop on that single allocation: when the buffer would exceed
/// it we return `CpuMemoryExceeded` so the graph collector degrades to IBP for
/// that node instead of allocating (and OOMing).
const DEFAULT_CROWN_MEM_CAP_MB: usize = 512;

/// Ceiling for the RAM-adaptive default cap in MiB. Even a very large host
/// never defaults past 16 GiB per buffer — the cap remains an OOM backstop,
/// not an unlimited grant.
const MAX_ADAPTIVE_CROWN_MEM_CAP_MB: usize = 16 * 1024;

/// The RAM-adaptive default grants `total_system_ram / 16` per buffer. The
/// fraction is deliberately conservative: the buffer really allocates (and the
/// fused lower/upper path allocates a pair), other resident state (weights,
/// bound maps, GPU staging) shares the same RAM, and several targets can be in
/// flight across a run. 1/16 keeps the worst-case pair under 1/8 of RAM.
const ADAPTIVE_CROWN_MEM_CAP_RAM_DIVISOR: u64 = 16;

/// Env override for the host-RAM-adaptive policy cap. A value of `0` disables
/// this policy ceiling; an explicitly configured CPU Dense budget and
/// observable kernel process/container envelopes remain mandatory. Any
/// positive integer sets the per-buffer policy cap in MiB.
const CROWN_MEM_CAP_ENV: &str = "NY_CROWN_MEM_CAP_MB";

/// RAM-adaptive default cap in MiB, from an explicit total-RAM value
/// (testable without touching `/proc`).
///
/// `clamp(total_ram / 16, 512 MiB, 16 GiB)`; `None` (RAM unknown) keeps the
/// conservative 512 MiB fixed default. The 512 MiB floor was sized for 24 GB
/// competition boxes; on large-RAM hosts the fixed value forced sound-but-loose
/// IBP fallbacks on buffers the machine holds trivially (VGG16's
/// `[1000 x 401408]` pair is ~1.6 GiB each on a 121 GB box), costing verdicts.
/// On such hosts this policy ceiling no longer forces that fallback; the
/// independently enforced Dense-result and process/container bounds may still
/// do so. Raising the policy cap is soundness-neutral: it only chooses between
/// computing the tight CROWN bound and the looser-but-sound IBP fallback.
fn adaptive_default_crown_mem_cap_mb(total_ram_bytes: Option<u64>) -> usize {
    match total_ram_bytes {
        Some(total) => {
            let cap_mb = total / ADAPTIVE_CROWN_MEM_CAP_RAM_DIVISOR / (1024 * 1024);
            cap_mb.clamp(
                DEFAULT_CROWN_MEM_CAP_MB as u64,
                MAX_ADAPTIVE_CROWN_MEM_CAP_MB as u64,
            ) as usize
        }
        None => DEFAULT_CROWN_MEM_CAP_MB,
    }
}

/// Parse a named kB-valued field out of `/proc/meminfo` contents.
fn parse_meminfo_field_bytes(contents: &str, field: &str) -> Option<u64> {
    let rest = contents.lines().find_map(|line| line.strip_prefix(field))?;
    let mut values = rest.split_whitespace();
    let kb: u64 = values.next()?.parse().ok()?;
    if values.next()? != "kB" {
        return None;
    }
    kb.checked_mul(1024)
}

fn parse_meminfo_total_bytes(contents: &str) -> Option<u64> {
    parse_meminfo_field_bytes(contents, "MemTotal:")
}

fn parse_meminfo_available_bytes(contents: &str) -> Option<u64> {
    parse_meminfo_field_bytes(contents, "MemAvailable:")
}

/// Total system RAM in bytes via `/proc/meminfo`, read once per process.
/// `None` when unavailable (non-Linux, unreadable, or unparseable).
fn total_system_ram_bytes() -> Option<u64> {
    static TOTAL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TOTAL.get_or_init(|| {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .as_deref()
            .and_then(parse_meminfo_total_bytes)
    })
}

/// Currently available host RAM. Unlike total capacity this is intentionally
/// not cached: sibling jobs can consume or release memory during a long run.
fn available_system_ram_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(parse_meminfo_available_bytes)
}

/// Effective default cap in MiB: RAM-adaptive on hosts whose total RAM is
/// readable, the fixed 512 MiB otherwise.
fn default_crown_mem_cap_mb() -> usize {
    adaptive_default_crown_mem_cap_mb(total_system_ram_bytes())
}

/// Convert remaining process/container headroom to the conservative
/// per-buffer share used by the existing adaptive policy. There is no 512 MiB
/// floor here: a small enforced envelope must be allowed to tighten the cap
/// below the host-oriented fallback rather than being silently overruled.
fn envelope_crown_mem_cap_bytes(headroom_bytes: u64) -> usize {
    let cap = headroom_bytes / ADAPTIVE_CROWN_MEM_CAP_RAM_DIVISOR;
    usize::try_from(cap.min((MAX_ADAPTIVE_CROWN_MEM_CAP_MB as u64) * 1024 * 1024))
        .unwrap_or(usize::MAX)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConvCrownMemCapSource {
    Policy,
    HostAvailable,
    ProcessEnvelope,
    ExplicitDenseResultBudget,
    FallbackDenseResultBudget,
}

impl ConvCrownMemCapSource {
    const fn label(self) -> &'static str {
        match self {
            Self::Policy => "NY_CROWN_MEM_CAP_MB/adaptive policy",
            Self::HostAvailable => "/proc/meminfo MemAvailable headroom",
            Self::ProcessEnvelope => "cgroup/RLIMIT_AS headroom",
            Self::ExplicitDenseResultBudget => "explicit NY_DENSE_BUDGET_MB result budget",
            Self::FallbackDenseResultBudget => "default Dense-result fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConvCrownMemCap {
    bytes: usize,
    source: ConvCrownMemCapSource,
}

/// Narrow the per-buffer policy cap by every independently meaningful bound:
/// dynamic host availability, kernel-enforced cgroup/RLIMIT headroom, and the
/// explicitly configured `NY_DENSE_BUDGET_MB` CROWN Dense-materialization
/// budget.
///
/// Dividing the Dense budget by the retained result-buffer count constrains
/// that result set as a unit. The general 2 GiB Dense default is a last resort
/// only when no policy/envelope can be observed; applying it unconditionally
/// would silently defeat the RAM-adaptive policy's documented multi-GiB case.
/// This is not a whole-process peak-memory estimate; the conservative
/// host/process headroom shares cover unrelated resident state and scratch.
fn narrowest_conv_crown_mem_cap(
    policy_cap: Option<usize>,
    host_available_bytes: Option<u64>,
    process_headroom_bytes: Option<u64>,
    explicit_dense_budget_bytes: Option<usize>,
    n_buffers: usize,
) -> ConvCrownMemCap {
    let fallback = ConvCrownMemCap {
        bytes: crate::network::crown_memory::DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024
            / n_buffers.max(1),
        source: ConvCrownMemCapSource::FallbackDenseResultBudget,
    };
    [
        policy_cap.map(|bytes| ConvCrownMemCap {
            bytes,
            source: ConvCrownMemCapSource::Policy,
        }),
        host_available_bytes.map(|available| ConvCrownMemCap {
            bytes: envelope_crown_mem_cap_bytes(available),
            source: ConvCrownMemCapSource::HostAvailable,
        }),
        process_headroom_bytes.map(|headroom| ConvCrownMemCap {
            bytes: envelope_crown_mem_cap_bytes(headroom),
            source: ConvCrownMemCapSource::ProcessEnvelope,
        }),
        explicit_dense_budget_bytes.map(|budget| ConvCrownMemCap {
            bytes: budget / n_buffers.max(1),
            source: ConvCrownMemCapSource::ExplicitDenseResultBudget,
        }),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|cap| cap.bytes)
    .unwrap_or(fallback)
}

/// Per-buffer Conv2d CROWN-backward memory cap in bytes.
///
/// Priority: `NY_CROWN_MEM_CAP_MB` env var > host-RAM-adaptive default
/// ([`adaptive_default_crown_mem_cap_mb`]). A configured value of `0` returns
/// `None`, disabling this policy ceiling. The allocation guard separately
/// enforces an explicit Dense budget and observable process/container envelope,
/// with the Dense default retained only when no other cap is observable.
fn conv_crown_mem_cap_bytes() -> Option<usize> {
    let mb = match std::env::var(CROWN_MEM_CAP_ENV) {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(parsed) => parsed,
            Err(_) => default_crown_mem_cap_mb(),
        },
        Err(_) => default_crown_mem_cap_mb(),
    };
    if mb == 0 {
        None
    } else {
        Some(mb.saturating_mul(1024 * 1024))
    }
}

fn effective_conv_crown_mem_cap(n_buffers: usize) -> ConvCrownMemCap {
    narrowest_conv_crown_mem_cap(
        conv_crown_mem_cap_bytes(),
        available_system_ram_bytes(),
        crate::network::crown_memory::process_memory_headroom_bytes(),
        crate::network::crown_memory::explicit_cpu_crown_dense_budget_bytes(),
        n_buffers,
    )
}

fn effective_conv_crown_envelope_cap() -> Option<ConvCrownMemCap> {
    [
        available_system_ram_bytes().map(|available| ConvCrownMemCap {
            bytes: envelope_crown_mem_cap_bytes(available),
            source: ConvCrownMemCapSource::HostAvailable,
        }),
        crate::network::crown_memory::process_memory_headroom_bytes().map(|headroom| {
            ConvCrownMemCap {
                bytes: envelope_crown_mem_cap_bytes(headroom),
                source: ConvCrownMemCapSource::ProcessEnvelope,
            }
        }),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|cap| cap.bytes)
}

/// Bytes required for one dense backward coefficient buffer of shape
/// `[num_objectives x conv_in_size]` (`f32`), saturating to `usize::MAX` on
/// overflow so an overflowing shape always trips the cap.
#[cfg(test)]
fn backward_buffer_bytes(num_objectives: usize, conv_in_size: usize) -> usize {
    backward_buffer_bytes_for(num_objectives, conv_in_size, size_of::<f32>())
}

fn backward_buffer_bytes_for(
    num_objectives: usize,
    conv_in_size: usize,
    element_bytes: usize,
) -> usize {
    num_objectives
        .checked_mul(conv_in_size)
        .and_then(|n| n.checked_mul(element_bytes))
        .unwrap_or(usize::MAX)
}

/// Largest known single transient allocation in one grouped transpose-conv
/// call: extracted A, extracted weights, or the GEMM/col2im matrix. The
/// returned byte count saturates so overflow is refused before allocation.
fn largest_conv_crown_scratch_bytes(
    matrix_shapes: &[(usize, usize)],
    element_bytes: usize,
) -> usize {
    matrix_shapes
        .iter()
        .map(|&(rows, cols)| backward_buffer_bytes_for(rows, cols, element_bytes))
        .max()
        .unwrap_or(0)
}

/// Apply the dynamic host/process envelope to the largest visible transient.
///
/// The 1/16 envelope share is deliberately per allocation: the f32 fallback's
/// worst visible pair path has fewer than 16 simultaneously live allocations,
/// including duplicated A/W flattening and GEMM output conversion. Opaque GEMM
/// engine workspace is not claimed as exact here; it remains inside the large
/// reserve left by this gate. This is deliberately not presented as an exact
/// whole-process peak estimator.
fn guard_conv_crown_scratch_bytes(required_bytes: usize) -> Result<()> {
    guard_conv_crown_scratch_bytes_with_cap(required_bytes, effective_conv_crown_envelope_cap())
}

fn guard_conv_crown_scratch_bytes_with_cap(
    required_bytes: usize,
    cap: Option<ConvCrownMemCap>,
) -> Result<()> {
    if required_bytes == usize::MAX {
        warn!(
            "Conv2d CROWN backward scratch size overflowed usize; refusing the allocation so the \
             caller can take its sound fallback"
        );
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: cap.map_or(usize::MAX, |cap| cap.bytes),
            site: CONV_CROWN_BACKWARD_SCRATCH_SITE,
        });
    }
    let Some(cap) = cap else {
        return Ok(());
    };
    if required_bytes <= cap.bytes {
        return Ok(());
    }
    warn!(
        "Conv2d CROWN backward scratch cap triggered: largest transient requires \
         {required_bytes} bytes (> effective cap {} bytes from {}); refusing the allocation so \
         the caller can take its sound fallback",
        cap.bytes,
        cap.source.label()
    );
    Err(NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes: cap.bytes,
        site: CONV_CROWN_BACKWARD_SCRATCH_SITE,
    })
}

/// Hard, sound backstop: refuse to materialize a Conv2d CROWN backward
/// coefficient buffer larger than the configured per-buffer cap.
///
/// Returns `Err(CpuMemoryExceeded)` when a `[num_objectives x conv_in_size]`
/// `f32` buffer would exceed [`conv_crown_mem_cap_bytes`]. The graph CROWN-IBP
/// collector classifies this error as a sound IBP fallback for the target node
/// (`crown_tighten.rs`), so the result is looser-but-sound bounds instead of an
/// OOM. `NY_CROWN_MEM_CAP_MB=0` disables only that optional policy ceiling;
/// an explicitly configured Dense result budget and observable
/// host/process/container envelope are still enforced; the 2 GiB Dense default
/// is retained as a last resort when no other cap can be observed.
///
/// `n_buffers` is the number of buffers of this size resident on the path (1
/// for a genuinely standalone result, 2 for a retained lower/upper pair). It
/// divides NY's Dense-materialization budget across that retained result set,
/// so a pair cannot claim the whole result budget independently per member.
pub(crate) fn guard_conv_crown_backward_buffer(
    num_objectives: usize,
    conv_in_size: usize,
    n_buffers: usize,
) -> Result<()> {
    guard_conv_crown_backward_buffer_with_cap(
        num_objectives,
        conv_in_size,
        n_buffers,
        size_of::<f32>(),
        effective_conv_crown_mem_cap(n_buffers),
    )
}

fn guard_conv_crown_backward_f64_buffer(
    num_objectives: usize,
    conv_in_size: usize,
    n_buffers: usize,
) -> Result<()> {
    guard_conv_crown_backward_buffer_with_cap(
        num_objectives,
        conv_in_size,
        n_buffers,
        size_of::<f64>(),
        effective_conv_crown_mem_cap(n_buffers),
    )
}

fn guard_conv_crown_backward_buffer_with_cap(
    num_objectives: usize,
    conv_in_size: usize,
    n_buffers: usize,
    element_bytes: usize,
    cap: ConvCrownMemCap,
) -> Result<()> {
    let required = backward_buffer_bytes_for(num_objectives, conv_in_size, element_bytes);
    if required == usize::MAX {
        warn!(
            "Conv2d CROWN backward result size overflowed usize for {n_buffers} buffer(s) of \
             [{num_objectives} x {conv_in_size}] with {element_bytes}-byte elements; refusing the \
             allocation so the caller can take its sound fallback"
        );
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: cap.bytes,
            site: CONV_CROWN_BACKWARD_SITE,
        });
    }
    if required > cap.bytes {
        warn!(
            "Conv2d CROWN backward memory cap triggered: {n_buffers} buffer(s) of \
             [{num_objectives} x {conv_in_size}] with {element_bytes}-byte elements require \
             {required} bytes each \
             (> effective cap {} bytes from {}); refusing the allocation so the caller can \
             take its sound fallback",
            cap.bytes,
            cap.source.label()
        );
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: cap.bytes,
            site: CONV_CROWN_BACKWARD_SITE,
        });
    }
    Ok(())
}

/// Fused `(lower_a, upper_a)` Conv2d CROWN backward sharing one resident weight.
///
/// Both A-matrices propagate through the *same* Conv2d kernel, so this builds
/// each group's weight column matrix as a single `Arc<[f32]>` and hands it to
/// [`GemmEngine::conv_transpose_2d_pair_cached`]. A GPU engine keeps that weight
/// matrix GPU-resident (keyed by `Arc` identity), reuses its buffers, and stacks
/// both inputs into one `2*S`-row dispatch — halving dispatch/readback overhead
/// versus two independent calls, with a bit-identical result.
///
/// The fused pair path is used only when it is provably equivalent to the
/// existing per-matrix path: an engine is present, dilation is 1 (the fused GPU
/// kernel has no dilation support), and the non-chunked (no per-row deadline
/// chunking) branch applies. In every other case — and on any per-group fused
/// error — this delegates to two independent
/// [`conv2d_transpose_batched_gemm_grouped_with_deadline`] calls, so behavior on
/// those paths is exactly unchanged.
///
/// Returns `(new_lower_a, new_upper_a)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    grad_size: (usize, usize),
    out_channels: usize,
    groups: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let (dh, dw) = dilation;
    let (grad_h, grad_w) = grad_size;
    let num_objectives = lower_a.nrows();
    let use_chunked_deadline = deadline.is_some()
        && num_objectives.saturating_mul(grad_h).saturating_mul(grad_w) > DEADLINE_GEMM_ROW_CHUNK;

    // Conditions under which the fused, weight-resident pair path is exactly
    // equivalent to the per-matrix path. Anything else delegates below.
    let fused_pair_ok = engine.is_some() && dh == 1 && dw == 1 && !use_chunked_deadline;

    if fused_pair_ok {
        if let Some(result) = try_fused_pair(
            lower_a,
            upper_a,
            kernel,
            stride,
            padding,
            grad_size,
            output_size,
            out_channels,
            groups,
            engine.expect("fused_pair_ok implies engine present"),
        )? {
            return Ok(result);
        }
        // try_fused_pair returned None: a group could not use the fused engine
        // call (unsupported/failed). Fall through to the per-matrix path, which
        // itself retries the (non-paired) fused path and the CPU fallback.
    }

    // The lower_a and upper_a backward GEMMs are fully independent (they share
    // only immutable inputs). On the CPU path (no engine) the inner grouped GEMM
    // is single-threaded (faer degrades to Par::Seq inside Rayon / faer's global
    // default is Seq), and the per-layer driver leaves cores idle — so run the
    // two matrices concurrently via `rayon::join` to use two cores instead of
    // one. This is soundness-neutral: each call performs exactly the same
    // per-matrix f64-accumulating computation as before, only scheduled
    // concurrently. We restrict this to `engine.is_none()` because `&dyn
    // GemmEngine` is not guaranteed `Sync` (and the engine path has its own
    // batched pair fast-path above).
    if engine.is_none() {
        let (lower_res, upper_res) = rayon::join(
            || {
                conv2d_transpose_batched_gemm_grouped_with_deadline(
                    lower_a,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    output_size,
                    grad_size,
                    out_channels,
                    groups,
                    2,
                    None,
                    deadline,
                )
            },
            || {
                conv2d_transpose_batched_gemm_grouped_with_deadline(
                    upper_a,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    output_size,
                    grad_size,
                    out_channels,
                    groups,
                    2,
                    None,
                    deadline,
                )
            },
        );
        return Ok((lower_res?, upper_res?));
    }

    let new_lower_a = conv2d_transpose_batched_gemm_grouped_with_deadline(
        lower_a,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        groups,
        2,
        engine,
        deadline,
    )?;
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(per_node_deadline_exceeded());
        }
    }
    let new_upper_a = conv2d_transpose_batched_gemm_grouped_with_deadline(
        upper_a,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        groups,
        2,
        engine,
        deadline,
    )?;
    Ok((new_lower_a, new_upper_a))
}

/// Attempt the fully-fused per-group pair path. Returns `Ok(None)` if any group
/// cannot use the fused engine call, so the caller falls back. The non-fused
/// portions (matrix extraction, col2im scatter) mirror the single-matrix path
/// in [`conv2d_transpose_batched_gemm_grouped_with_deadline`] exactly.
#[allow(clippy::too_many_arguments)]
fn try_fused_pair(
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    grad_size: (usize, usize),
    output_size: (usize, usize),
    out_channels: usize,
    groups: usize,
    engine: &dyn GemmEngine,
) -> Result<Option<(Array2<f32>, Array2<f32>)>> {
    let num_objectives = lower_a.nrows();
    let (grad_h, grad_w) = grad_size;
    let (in_h, in_w) = output_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if out_c != out_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_channels],
            got: vec![out_c],
        });
    }

    let expected_cols = checked_shape_product(&[out_c, grad_h, grad_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: output dims overflow: {out_c} * {grad_h} * {grad_w}"
        ))
    })?;
    if lower_a.ncols() != expected_cols || upper_a.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![lower_a.ncols().max(upper_a.ncols())],
        });
    }
    if upper_a.nrows() != num_objectives {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_objectives],
            got: vec![upper_a.nrows()],
        });
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_transpose_pair: groups must be nonzero".to_string(),
        ));
    }

    let total_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: total_in_c overflow: {in_c_per_group} * {groups}"
        ))
    })?;
    let out_c_per_group = out_c / groups;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: kernel spatial overflow: {kh} * {kw}"
        ))
    })?;
    let kernel_cols_per_group = in_c_per_group.checked_mul(kernel_spatial).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: kernel columns overflow: {in_c_per_group} * {kernel_spatial}"
        ))
    })?;
    let spatial_per_obj = grad_h.checked_mul(grad_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: spatial overflow: {grad_h} * {grad_w}"
        ))
    })?;
    let total_spatial = num_objectives.checked_mul(spatial_per_obj).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: total spatial overflow: {num_objectives} * {spatial_per_obj}"
        ))
    })?;
    let input_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: input spatial overflow: {in_h} * {in_w}"
        ))
    })?;
    let conv_in_size = total_in_c.checked_mul(input_spatial).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: input size overflow: {total_in_c} * {input_spatial}"
        ))
    })?;
    let group_input_dim = in_c_per_group.checked_mul(input_spatial).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_pair: group input overflow: {in_c_per_group} * {input_spatial}"
        ))
    })?;

    guard_conv_crown_scratch_bytes(largest_conv_crown_scratch_bytes(
        &[
            (total_spatial, out_c_per_group),
            (out_c_per_group, kernel_cols_per_group),
        ],
        size_of::<f32>(),
    ))?;

    // Hard memory backstop before the dense pair allocation (#conv-crown-oom):
    // the fused pair materializes BOTH lower_flat and upper_flat full-size, so a
    // single mid-size conv target can pin multiple GiB here. Cap each buffer and
    // fall back to sound IBP when exceeded instead of OOMing.
    guard_conv_crown_backward_buffer(num_objectives, conv_in_size, 2)?;

    let mut lower_flat = vec![0.0f32; num_objectives * conv_in_size];
    let mut upper_flat = vec![0.0f32; num_objectives * conv_in_size];

    for g in 0..groups {
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;

        // Per-group weight column matrix, shared by both bounds as ONE Arc so
        // the GPU keeps it resident across the lower+upper dispatch.
        let mut w_vec = Vec::with_capacity(out_c_per_group * kernel_cols_per_group);
        for oc_local in 0..out_c_per_group {
            for col in 0..kernel_cols_per_group {
                let ic = col / kernel_spatial;
                let rem = col % kernel_spatial;
                let ki = rem / kw;
                let kj = rem % kw;
                w_vec.push(kernel[[oc_start + oc_local, ic, ki, kj]]);
            }
        }
        let w_arc: Arc<[f32]> = Arc::from(w_vec);

        // Per-group A slices for lower and upper, row-major (rows = S*OH*OW).
        let build_a = |a: &Array2<f32>| -> Vec<f32> {
            let mut flat = Vec::with_capacity(total_spatial * out_c_per_group);
            for pos in 0..total_spatial {
                let obj = pos / spatial_per_obj;
                let spatial = pos % spatial_per_obj;
                for oc_local in 0..out_c_per_group {
                    let oc = oc_start + oc_local;
                    flat.push(a[[obj, oc * spatial_per_obj + spatial]]);
                }
            }
            flat
        };
        let a_lower_flat = build_a(lower_a);
        let a_upper_flat = build_a(upper_a);

        let ct_params = ConvTranspose2dParams {
            num_specs: num_objectives,
            out_channels: out_c_per_group,
            in_channels: in_c_per_group,
            out_h: grad_h,
            out_w: grad_w,
            in_h,
            in_w,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
        };

        match engine.conv_transpose_2d_pair_cached(&a_lower_flat, &a_upper_flat, &w_arc, &ct_params)
        {
            Ok((lower_res, upper_res)) => {
                debug!(
                    "Conv2d CROWN backward group {g}: fused conv_transpose pair succeeded ({}x{}x{})",
                    total_spatial, out_c_per_group, kernel_cols_per_group,
                );
                for obj in 0..num_objectives {
                    for ic_local in 0..in_c_per_group {
                        let ic = ic_start + ic_local;
                        let src_base = obj * group_input_dim + ic_local * input_spatial;
                        let dst_base = obj * conv_in_size + ic * input_spatial;
                        for pixel in 0..input_spatial {
                            lower_flat[dst_base + pixel] += lower_res[src_base + pixel];
                            upper_flat[dst_base + pixel] += upper_res[src_base + pixel];
                        }
                    }
                }
            }
            Err(error) => {
                debug!(
                    "Conv2d CROWN backward group {g}: fused conv_transpose pair fallback: {error}"
                );
                // Any failure: abandon the fused pair path entirely and let the
                // caller delegate to the proven per-matrix path.
                return Ok(None);
            }
        }
    }

    let lower = Array2::from_shape_vec((num_objectives, conv_in_size), lower_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d_transpose_pair lower reshape: {e}")))?;
    let upper = Array2::from_shape_vec((num_objectives, conv_in_size), upper_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d_transpose_pair upper reshape: {e}")))?;
    Ok(Some((lower, upper)))
}

pub(crate) fn per_node_deadline_exceeded() -> NyError {
    NyError::DeadlineExceeded("Conv2d CROWN backward: per-node deadline exceeded".to_string())
}

/// Convert faer::Mat (column-major) to flat row-major Vec<f32>.
/// Used in conv2d_transpose_batched_gemm to produce cache-friendly layout
/// for the col2im scatter loop.
fn faer_mat_to_row_major(mat: &Mat<f32>, rows: usize, cols: usize) -> Vec<f32> {
    // Build the row-major buffer with `with_capacity` + `push` rather than
    // `vec![0.0; rows*cols]` + overwrite. Every slot is written exactly once,
    // so the zero-init memset (`_platform_memset`) was pure overhead; pushing
    // produces bit-identical values without it. (#3795 alloc/copy reduction)
    let mut flat = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            flat.push(mat[(row, col)]);
        }
    }
    flat
}

/// Deadline-aware batched conv2d transpose via GEMM with explicit groups.
///
/// When `deadline` is present and the workload is large enough to need chunking,
/// the GEMM rows are processed in bounded batches so graph-CROWN can abort inside
/// a single expensive Conv2d node instead of overshooting the global verifier
/// timeout. Returns `DeadlineExceeded("...per-node deadline exceeded")`
/// so higher layers can degrade to sound IBP fallback. Part of #3795.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_batched_gemm_grouped_with_deadline(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    grad_size: (usize, usize),
    out_channels: usize,
    groups: usize,
    resident_result_buffers: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Array2<f32>> {
    let num_objectives = a_coefficients.nrows();
    let (grad_h, grad_w) = grad_size;
    let (in_h, in_w) = output_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    // The fused GPU conv_transpose_2d path does not yet support dilation>1
    // (ConvTranspose2dParams has no dilation field). Disable it for dilated
    // convolutions so we fall through to the dilation-aware CPU col2im path
    // and never silently produce wrong bounds. Part of dilated-conv support.
    let engine = if dh != 1 || dw != 1 { None } else { engine };

    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    if out_c != out_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_channels],
            got: vec![out_c],
        });
    }

    let expected_cols = checked_shape_product(&[out_c, grad_h, grad_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: output dims overflow: {out_c} * {grad_h} * {grad_w}"
        ))
    })?;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_transpose_batched_gemm: groups must be nonzero".to_string(),
        ));
    }

    let total_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: total_in_c overflow: {in_c_per_group} * {groups}"
        ))
    })?;
    let out_c_per_group = out_c / groups;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: kernel spatial overflow: {kh} * {kw}"
        ))
    })?;
    let kernel_cols_per_group =
        checked_shape_product(&[in_c_per_group, kh, kw]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: kernel cols overflow: {in_c_per_group} * {kh} * {kw}"
        ))
        })?;
    let spatial_per_obj = grad_h.checked_mul(grad_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: spatial overflow: {grad_h} * {grad_w}"
        ))
    })?;
    let total_spatial = num_objectives.checked_mul(spatial_per_obj).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: total_spatial overflow: {num_objectives} * {spatial_per_obj}"
        ))
    })?;
    let input_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: input spatial overflow: {in_h} * {in_w}"
        ))
    })?;
    let conv_in_size = checked_shape_product(&[total_in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: conv_in_size overflow: {total_in_c} * {in_h} * {in_w}"
        ))
    })?;
    let group_input_dim = in_c_per_group.checked_mul(input_spatial).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: group_input_dim overflow: {in_c_per_group} * {input_spatial}"
        ))
    })?;

    let use_chunked_deadline = deadline.is_some() && total_spatial > DEADLINE_GEMM_ROW_CHUNK;
    let initial_a_rows = if use_chunked_deadline && engine.is_none() {
        total_spatial.min(DEADLINE_GEMM_ROW_CHUNK)
    } else {
        total_spatial
    };
    guard_conv_crown_scratch_bytes(largest_conv_crown_scratch_bytes(
        &[
            (initial_a_rows, out_c_per_group),
            (out_c_per_group, kernel_cols_per_group),
        ],
        size_of::<f32>(),
    ))?;

    // Hard memory backstop before the dense result allocation (#conv-crown-oom):
    // result_flat is num_objectives * conv_in_size f32 and is allocated full-size
    // before any GEMM/col2im work (the deadline chunking only bounds GEMM input
    // rows, not this buffer). `resident_result_buffers` comes from the caller's
    // actual result lifetime: pair propagation passes 2 because the first result
    // remains live while the second is built; a genuinely standalone result
    // passes 1. Cap each and take the caller's sound fallback when exceeded.
    guard_conv_crown_backward_buffer(num_objectives, conv_in_size, resident_result_buffers)?;

    let result_len = num_objectives.checked_mul(conv_in_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_batched_gemm: result alloc overflow: {num_objectives} * {conv_in_size}"
        ))
    })?;
    let mut result_flat = vec![0.0f32; result_len];

    if !use_chunked_deadline {
        for g in 0..groups {
            let oc_start = g * out_c_per_group;
            let ic_start = g * in_c_per_group;

            // Step 1: Extract A-coefficients for this group's output channels.
            // From (N, out_c * spatial) slice to (N * spatial, oc_per_group).
            let a_group = Mat::<f32>::from_fn(total_spatial, out_c_per_group, |pos, oc_local| {
                let obj = pos / spatial_per_obj;
                let spatial = pos % spatial_per_obj;
                let oc = oc_start + oc_local;
                a_coefficients[[obj, oc * spatial_per_obj + spatial]]
            });

            // Step 2: Extract kernel for this group: (oc_per_group, in_c_per_group * kh * kw).
            let w_group =
                Mat::<f32>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
                    let ic = col / kernel_spatial;
                    let rem = col % kernel_spatial;
                    let ki = rem / kw;
                    let kj = rem % kw;
                    kernel[[oc_start + oc_local, ic, ki, kj]]
                });

            // Flatten matrices for engine calls (needed by both fused and GEMM paths).
            let (a_flat, w_flat) = if engine.is_some() {
                let a_flat_len = total_spatial.checked_mul(out_c_per_group).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "conv2d_transpose_batched_gemm: a_flat alloc overflow: {total_spatial} * {out_c_per_group}"
                    ))
                })?;
                // `with_capacity` + `push`: every slot is written once below,
                // so the zero-init memset is wasted work. (#3795)
                let mut af = Vec::with_capacity(a_flat_len);
                for row in 0..total_spatial {
                    for col in 0..out_c_per_group {
                        af.push(a_group[(row, col)]);
                    }
                }
                let w_flat_len =
                    out_c_per_group
                        .checked_mul(kernel_cols_per_group)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "conv2d_transpose_batched_gemm: w_flat alloc overflow: {out_c_per_group} * {kernel_cols_per_group}"
                            ))
                        })?;
                let mut wf = Vec::with_capacity(w_flat_len);
                for row in 0..out_c_per_group {
                    for col in 0..kernel_cols_per_group {
                        wf.push(w_group[(row, col)]);
                    }
                }
                (Some(af), Some(wf))
            } else {
                (None, None)
            };

            // Try fused GPU conv_transpose_2d (GEMM + col2im in one GPU dispatch).
            // This eliminates the CPU col2im bottleneck. Part of #3813.
            if let Some(eng) = engine {
                let ct_params = ConvTranspose2dParams {
                    num_specs: num_objectives,
                    out_channels: out_c_per_group,
                    in_channels: in_c_per_group,
                    out_h: grad_h,
                    out_w: grad_w,
                    in_h,
                    in_w,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h: sh,
                    stride_w: sw,
                    pad_h: ph,
                    pad_w: pw,
                };
                match eng.conv_transpose_2d(
                    a_flat.as_ref().expect("a_flat built when engine present"),
                    w_flat.as_ref().expect("w_flat built when engine present"),
                    &ct_params,
                ) {
                    Ok(fused_result) => {
                        debug!(
                            "Conv2d CROWN backward group {g}: fused conv_transpose_2d succeeded ({}x{}x{})",
                            total_spatial, out_c_per_group, kernel_cols_per_group,
                        );
                        // Scatter fused result (S, ic_per_group * IH * IW) into result_flat.
                        // Disjoint per objective (each owns one conv_in_size chunk) and
                        // each element is touched once per group, so the FP order is
                        // unchanged. conv_in_size == 0 scatters nothing (par_chunks_mut
                        // requires a nonzero chunk length).
                        if conv_in_size > 0 {
                            result_flat
                                .par_chunks_mut(conv_in_size)
                                .enumerate()
                                .for_each(|(obj, out_row)| {
                                    for ic_local in 0..in_c_per_group {
                                        let ic = ic_start + ic_local;
                                        let src_base =
                                            obj * group_input_dim + ic_local * input_spatial;
                                        let dst_base = ic * input_spatial;
                                        for pixel in 0..input_spatial {
                                            out_row[dst_base + pixel] +=
                                                fused_result[src_base + pixel];
                                        }
                                    }
                                });
                        }
                        continue;
                    }
                    Err(error) => {
                        debug!(
                            "Conv2d CROWN backward group {g}: fused conv_transpose_2d fallback to GEMM + CPU col2im: {error}"
                        );
                    }
                }
                // Fused path not supported or failed — fall through to GEMM + CPU col2im.
            }

            guard_conv_crown_scratch_bytes(backward_buffer_bytes_for(
                total_spatial,
                kernel_cols_per_group,
                size_of::<f32>(),
            ))?;

            // Step 3: Per-group GEMM — (total_spatial, oc_per_group) × (oc_per_group, kernel_cols_per_group)
            let col_flat: Vec<f32> = if let Some(eng) = engine {
                match eng.gemm_f32(
                    total_spatial,
                    out_c_per_group,
                    kernel_cols_per_group,
                    a_flat.as_ref().expect("a_flat built when engine present"),
                    w_flat.as_ref().expect("w_flat built when engine present"),
                ) {
                    Ok(result) => {
                        debug!(
                            "Conv2d CROWN backward group {g}: GemmEngine GEMM {}x{}x{} succeeded",
                            total_spatial, out_c_per_group, kernel_cols_per_group
                        );
                        result
                    }
                    Err(e) => {
                        debug!(
                            "Conv2d CROWN backward group {g}: GemmEngine failed, CPU fallback: {e}"
                        );
                        let result = mat_mul(&a_group, &w_group);
                        faer_mat_to_row_major(&result, total_spatial, kernel_cols_per_group)
                    }
                }
            } else {
                let result = mat_mul(&a_group, &w_group);
                faer_mat_to_row_major(&result, total_spatial, kernel_cols_per_group)
            };

            // Step 4: col2im scatter for this group into the correct input channel range.
            // Disjoint per objective: `obj` was the innermost serial axis, so hoisting
            // it to the parallel outer axis keeps each element's accumulation in the
            // identical (gy, gx, ic_local, ki, kj) FP order. conv_in_size == 0 scatters
            // nothing (par_chunks_mut requires a nonzero chunk length).
            if conv_in_size > 0 {
                result_flat
                    .par_chunks_mut(conv_in_size)
                    .enumerate()
                    .for_each(|(obj, out_row)| {
                        for gy in 0..grad_h {
                            for gx in 0..grad_w {
                                let spatial_offset = gy * grad_w + gx;
                                let col_row = obj * spatial_per_obj + spatial_offset;
                                for ic_local in 0..in_c_per_group {
                                    let ic = ic_start + ic_local;
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let ih = (gy * sh + ki * dh) as isize - ph as isize;
                                            let iw = (gx * sw + kj * dw) as isize - pw as isize;
                                            if ih >= 0
                                                && ih < in_h as isize
                                                && iw >= 0
                                                && iw < in_w as isize
                                            {
                                                let col_idx =
                                                    ic_local * kernel_spatial + ki * kw + kj;
                                                let out_idx = ic * input_spatial
                                                    + ih as usize * in_w
                                                    + iw as usize;
                                                out_row[out_idx] += col_flat
                                                    [col_row * kernel_cols_per_group + col_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
            }
        }

        return Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
            .map_err(|e| NyError::InternalError(format!("col2im reshape: {e}")));
    }

    for g in 0..groups {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(per_node_deadline_exceeded());
        }
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;

        let w_group =
            Mat::<f32>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
                let ic = col / kernel_spatial;
                let rem = col % kernel_spatial;
                let ki = rem / kw;
                let kj = rem % kw;
                kernel[[oc_start + oc_local, ic, ki, kj]]
            });
        let w_flat = if engine.is_some() {
            let w_flat_len = out_c_per_group
                .checked_mul(kernel_cols_per_group)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "conv2d_transpose_batched_gemm: w_flat alloc overflow: {out_c_per_group} * {kernel_cols_per_group}"
                    ))
                })?;
            let mut flat = Vec::with_capacity(w_flat_len);
            for row in 0..out_c_per_group {
                for col in 0..kernel_cols_per_group {
                    flat.push(w_group[(row, col)]);
                }
            }
            Some(flat)
        } else {
            None
        };

        // Try fused GPU path: GEMM + col2im in one dispatch, skipping the entire
        // chunked CPU col2im loop. GPU dispatch is fast enough that per-group
        // deadline granularity suffices. Part of #3813.
        if let Some(eng) = engine {
            let a_flat_len = total_spatial.checked_mul(out_c_per_group).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "conv2d_transpose_batched_gemm: a_flat alloc overflow: {total_spatial} * {out_c_per_group}"
                ))
            })?;
            let mut a_flat = Vec::with_capacity(a_flat_len);
            for pos in 0..total_spatial {
                let obj = pos / spatial_per_obj;
                let spatial = pos % spatial_per_obj;
                for oc_local in 0..out_c_per_group {
                    let oc = oc_start + oc_local;
                    a_flat.push(a_coefficients[[obj, oc * spatial_per_obj + spatial]]);
                }
            }
            let ct_params = ConvTranspose2dParams {
                num_specs: num_objectives,
                out_channels: out_c_per_group,
                in_channels: in_c_per_group,
                out_h: grad_h,
                out_w: grad_w,
                in_h,
                in_w,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                pad_h: ph,
                pad_w: pw,
            };
            match eng.conv_transpose_2d(
                &a_flat,
                w_flat.as_ref().expect("w_flat built when engine present"),
                &ct_params,
            ) {
                Ok(fused_result) => {
                    debug!(
                        "Conv2d CROWN backward group {g}: fused conv_transpose_2d succeeded \
                         (deadline-aware, {}x{}x{})",
                        total_spatial, out_c_per_group, kernel_cols_per_group,
                    );
                    // Disjoint per objective; each element is touched once per group,
                    // so the FP order is unchanged. conv_in_size == 0 scatters nothing
                    // (par_chunks_mut requires a nonzero chunk length).
                    if conv_in_size > 0 {
                        result_flat
                            .par_chunks_mut(conv_in_size)
                            .enumerate()
                            .for_each(|(obj, out_row)| {
                                for ic_local in 0..in_c_per_group {
                                    let ic = ic_start + ic_local;
                                    let src_base = obj * group_input_dim + ic_local * input_spatial;
                                    let dst_base = ic * input_spatial;
                                    for pixel in 0..input_spatial {
                                        out_row[dst_base + pixel] += fused_result[src_base + pixel];
                                    }
                                }
                            });
                    }
                    continue;
                }
                Err(error) => {
                    debug!(
                        "Conv2d CROWN backward group {g}: fused conv_transpose_2d deadline-aware fallback to GEMM + CPU col2im: {error}"
                    );
                }
            }
            // Fused path unsupported or failed — fall through to chunked GEMM + CPU col2im.
        }

        let max_chunk_rows = total_spatial.min(DEADLINE_GEMM_ROW_CHUNK);
        guard_conv_crown_scratch_bytes(largest_conv_crown_scratch_bytes(
            &[
                (max_chunk_rows, out_c_per_group),
                (max_chunk_rows, kernel_cols_per_group),
            ],
            size_of::<f32>(),
        ))?;

        for row_start in (0..total_spatial).step_by(DEADLINE_GEMM_ROW_CHUNK) {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(per_node_deadline_exceeded());
            }
            let chunk_rows = (total_spatial - row_start).min(DEADLINE_GEMM_ROW_CHUNK);
            let a_group =
                Mat::<f32>::from_fn(chunk_rows, out_c_per_group, |local_row, oc_local| {
                    let pos = row_start + local_row;
                    let obj = pos / spatial_per_obj;
                    let spatial = pos % spatial_per_obj;
                    let oc = oc_start + oc_local;
                    a_coefficients[[obj, oc * spatial_per_obj + spatial]]
                });

            let col_flat: Vec<f32> = if let Some(eng) = engine {
                let a_flat_len = chunk_rows.checked_mul(out_c_per_group).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "conv2d_transpose_batched_gemm: chunk a_flat alloc overflow: {chunk_rows} * {out_c_per_group}"
                    ))
                })?;
                let mut a_flat = Vec::with_capacity(a_flat_len);
                for row in 0..chunk_rows {
                    for col in 0..out_c_per_group {
                        a_flat.push(a_group[(row, col)]);
                    }
                }
                match eng.gemm_f32(
                    chunk_rows,
                    out_c_per_group,
                    kernel_cols_per_group,
                    &a_flat,
                    w_flat
                        .as_ref()
                        .expect("w_flat is built whenever a GemmEngine is present"),
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        debug!("Conv2d CROWN backward group {g}: GemmEngine failed on deadline-aware chunk, CPU fallback: {e}");
                        let result = mat_mul(&a_group, &w_group);
                        faer_mat_to_row_major(&result, chunk_rows, kernel_cols_per_group)
                    }
                }
            } else {
                let result = mat_mul(&a_group, &w_group);
                faer_mat_to_row_major(&result, chunk_rows, kernel_cols_per_group)
            };

            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(per_node_deadline_exceeded());
            }

            // Disjoint per objective: each objective owns one conv_in_size chunk of
            // result_flat, and its GEMM-chunk rows are visited in the same ascending
            // order as the serial loop, so every element accumulates in the identical
            // FP order. Deadline cadence: every 32 chunk rows (as before) plus once at
            // each objective's first row. conv_in_size == 0 scatters nothing but keeps
            // the serial per-row deadline checks (par_chunks_mut requires a nonzero
            // chunk length).
            if conv_in_size == 0 {
                for local_row in 0..chunk_rows {
                    if local_row % 32 == 0 && deadline.is_some_and(|d| Instant::now() >= d) {
                        return Err(per_node_deadline_exceeded());
                    }
                }
            } else {
                let obj_lo = row_start / spatial_per_obj;
                let obj_hi = (row_start + chunk_rows - 1) / spatial_per_obj;
                result_flat[obj_lo * conv_in_size..(obj_hi + 1) * conv_in_size]
                    .par_chunks_mut(conv_in_size)
                    .enumerate()
                    .try_for_each(|(obj_off, out_row)| {
                        let obj = obj_lo + obj_off;
                        let pos_lo = (obj * spatial_per_obj).max(row_start);
                        let pos_hi = ((obj + 1) * spatial_per_obj).min(row_start + chunk_rows);
                        for pos in pos_lo..pos_hi {
                            let local_row = pos - row_start;
                            if (pos == pos_lo || local_row % 32 == 0)
                                && deadline.is_some_and(|d| Instant::now() >= d)
                            {
                                return Err(per_node_deadline_exceeded());
                            }
                            let spatial_offset = pos % spatial_per_obj;
                            let gy = spatial_offset / grad_w;
                            let gx = spatial_offset % grad_w;

                            for ic_local in 0..in_c_per_group {
                                let ic = ic_start + ic_local;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = (gy * sh + ki * dh) as isize - ph as isize;
                                        let iw = (gx * sw + kj * dw) as isize - pw as isize;
                                        if ih >= 0
                                            && ih < in_h as isize
                                            && iw >= 0
                                            && iw < in_w as isize
                                        {
                                            let col_idx = ic_local * kernel_spatial + ki * kw + kj;
                                            let out_idx = ic * input_spatial
                                                + ih as usize * in_w
                                                + iw as usize;
                                            out_row[out_idx] += col_flat
                                                [local_row * kernel_cols_per_group + col_idx];
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })?;
            }
        }
    }

    Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("col2im reshape: {e}")))
}

/// f64 transpose-conv backward coefficient recomputation (#vnncomp-aw-soundness).
///
/// Computes the SAME Conv2d CROWN-backward contraction as
/// [`conv2d_transpose_batched_gemm_grouped_with_deadline`] but accumulates the
/// per-group GEMM AND the col2im scatter in **f64** (the f32 operands widen
/// exactly to f64; only the f64 sum rounds). The result is the exact real
/// coefficient up to a sound `γ_n^f64·S` error, so the caller can:
///   - store the directed (round-to-nearest) f32 of this value as the point
///     coefficient, and
///   - certify a per-coefficient `cast_err = |f64 − stored_f32|` PLUS `γ_n^f64·S`
///     (now valid: the sum is f64-accumulated, not f32),
///     mirroring the Linear `aw_f64_with_abssum` fix. This is CPU-only (single group
///     path supports any `groups`); it is on the certified-error path, not the hot
///     point-coefficient path, so f64-on-CPU is acceptable.
///
/// SOUNDNESS: the f32→f64 widening of `a` and `kernel` is exact, and `f32*f32`
/// is exact in f64 (48 < 53 significand bits), so the only rounding is the f64
/// running sum — bounded by `γ_n^f64·S` with `n ≤ out_c·kh·kw` (the col2im
/// summation order is irrelevant: Higham's `γ_n·S` bound is order-independent).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_backward_coeff_f64(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    grad_size: (usize, usize),
    out_channels: usize,
    groups: usize,
    resident_result_buffers: usize,
) -> Result<Array2<f64>> {
    let num_objectives = a_coefficients.nrows();
    let (grad_h, grad_w) = grad_size;
    let (in_h, in_w) = output_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if out_c != out_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_channels],
            got: vec![out_c],
        });
    }

    let expected_cols = checked_shape_product(&[out_c, grad_h, grad_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: output dims overflow: {out_c} * {grad_h} * {grad_w}"
        ))
    })?;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_transpose_backward_coeff_f64: groups must be nonzero".to_string(),
        ));
    }

    let total_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: total_in_c overflow: {in_c_per_group} * {groups}"
        ))
    })?;
    let out_c_per_group = out_c / groups;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: kernel spatial overflow: {kh} * {kw}"
        ))
    })?;
    let kernel_cols_per_group = in_c_per_group
        .checked_mul(kernel_spatial)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_transpose_backward_coeff_f64: kernel columns overflow: {in_c_per_group} * {kernel_spatial}"
            ))
        })?;
    let spatial_per_obj = grad_h.checked_mul(grad_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: spatial overflow: {grad_h} * {grad_w}"
        ))
    })?;
    let total_spatial = num_objectives.checked_mul(spatial_per_obj).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: total spatial overflow: {num_objectives} * {spatial_per_obj}"
        ))
    })?;
    let input_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: input spatial overflow: {in_h} * {in_w}"
        ))
    })?;
    let conv_in_size = checked_shape_product(&[total_in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: conv_in_size overflow: {total_in_c} * {in_h} * {in_w}"
        ))
    })?;

    guard_conv_crown_scratch_bytes(largest_conv_crown_scratch_bytes(
        &[
            (total_spatial, out_c_per_group),
            (out_c_per_group, kernel_cols_per_group),
            (total_spatial, kernel_cols_per_group),
        ],
        size_of::<f64>(),
    ))?;

    // Reuse the same byte cap as the f32 path while charging the actual 8-byte
    // element width. A transient f64 allocation still consumes real process /
    // cgroup headroom and must not receive an implicit 2x exemption. Lower/upper
    // callers pass 2 because the first f64 result remains live while the second
    // is computed (including the rayon::join path); advisory single-result
    // propagation passes 1.
    guard_conv_crown_backward_f64_buffer(num_objectives, conv_in_size, resident_result_buffers)?;

    let result_len = num_objectives.checked_mul(conv_in_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: result alloc overflow: {num_objectives} * {conv_in_size}"
        ))
    })?;
    let mut result_flat = vec![0.0f64; result_len];

    for g in 0..groups {
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;

        // f64 GEMM: (total_spatial, oc_per_group) x (oc_per_group, kernel_cols).
        // col_flat[row, col] = Σ_oc a[row, oc] · w[oc, col]  (f64 accumulation).
        // For large products this routes the single f64 GEMM to cuBLAS Dgemm (the
        // conv twin of the Linear `aw_via_engine` seam); the col2im scatter below
        // stays f64 on CPU. SOUND: the conv certified error (`cast + γ_n^f64·S +
        // prop`, `S` over-bounded by `row_max(a)·‖kernel‖₁`) is summation-order
        // independent, so cuBLAS's reduction order is safe.
        let col_flat = conv_group_col_flat_f64(
            a_coefficients,
            kernel,
            oc_start,
            out_c_per_group,
            kernel_cols_per_group,
            total_spatial,
            spatial_per_obj,
            kernel_spatial,
            kw,
        );

        // col2im scatter (f64 accumulation into the input pixels), parallelized
        // over objectives. Objective `obj` owns the disjoint output chunk
        // `result_flat[obj*conv_in_size .. (obj+1)*conv_in_size]`, and — since
        // each `out_idx` belongs to exactly one group (its `ic` is in this
        // group's channel range) — the per-`(obj, out_idx)` `+=` order stays the
        // fixed `gy -> gx -> ic_local -> ki -> kj` order it had when `obj` was
        // the innermost loop, so the result is bit-for-bit unchanged; only the
        // cross-objective parallelism is new. `conv_in_size == 0` scatters
        // nothing (par_chunks_mut requires a nonzero chunk length).
        if conv_in_size > 0 {
            result_flat
                .par_chunks_mut(conv_in_size)
                .enumerate()
                .for_each(|(obj, obj_out)| {
                    for gy in 0..grad_h {
                        for gx in 0..grad_w {
                            let spatial_offset = gy * grad_w + gx;
                            let col_row = obj * spatial_per_obj + spatial_offset;
                            for ic_local in 0..in_c_per_group {
                                let ic = ic_start + ic_local;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = (gy * sh + ki * dh) as isize - ph as isize;
                                        let iw = (gx * sw + kj * dw) as isize - pw as isize;
                                        if ih >= 0
                                            && ih < in_h as isize
                                            && iw >= 0
                                            && iw < in_w as isize
                                        {
                                            let col_idx = ic_local * kernel_spatial + ki * kw + kj;
                                            let out_idx = ic * input_spatial
                                                + ih as usize * in_w
                                                + iw as usize;
                                            obj_out[out_idx] +=
                                                col_flat[col_row * kernel_cols_per_group + col_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }

    Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d f64 col2im reshape: {e}")))
}

/// Largest single sound-conv f64 GEMM MAC count below which the CPU path wins
/// (GPU launch/transfer bound) — the same crossover the Linear seam uses.
///
/// Shared with the ConvTranspose forward-conv recompute in `ops_gemm`.
pub(super) const CONV_SOUND_F64_GEMM_MIN_MACS: usize = 1 << 24;

/// Per-group conv-transpose `col_flat = A·W` in f64, routing the single f64 GEMM
/// to the process-global sound engine (cuBLAS Dgemm) for large products, else the
/// CPU triple-loop. The conv twin of [`crate::layers::linear::crown_single`]'s
/// `aw_via_engine`. SOUND: the f32→f64 widening is exact and the conv certified
/// error (`cast + γ_n^f64·S + prop`) is summation-order independent, so cuBLAS's
/// reduction order cannot defeat it. Returns `(total_spatial × kernel_cols_per_group)`
/// row-major. Any engine error falls back to the CPU loop (bit-for-bit as before).
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
) -> Vec<f64> {
    let macs = total_spatial
        .saturating_mul(out_c_per_group)
        .saturating_mul(kernel_cols_per_group);
    if macs >= CONV_SOUND_F64_GEMM_MIN_MACS {
        if let Some(cf) = conv_group_col_flat_via_engine(
            a_coefficients,
            kernel,
            oc_start,
            out_c_per_group,
            kernel_cols_per_group,
            total_spatial,
            spatial_per_obj,
            kernel_spatial,
            kw,
        ) {
            return cf;
        }
    }
    // CPU fallback: faer f64 GEMM (#cgan-conv-f64-gemm). Replaces the prior
    // single-threaded sparsity-skipping scalar triple-loop, which dominated
    // Conv2d CROWN backward wall time on conv stacks with no CUDA engine
    // (cGAN-class: one intermediate-node backward took ~16s, starving
    // α-CROWN and BaB). SOUND: same operands as the engine seam above, and
    // the certified error `cast + γ_n^f64·S + prop` is summation-order
    // independent, so faer's blocked f64 accumulation is covered by the same
    // certificate as the scalar loop (identical argument to the cuBLAS route).
    let a64 = Mat::<f64>::from_fn(total_spatial, out_c_per_group, |row, oc_local| {
        let obj = row / spatial_per_obj;
        let spatial = row % spatial_per_obj;
        let oc = oc_start + oc_local;
        f64::from(a_coefficients[[obj, oc * spatial_per_obj + spatial]])
    });
    let w64 = Mat::<f64>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
        let oc = oc_start + oc_local;
        let ic = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        f64::from(kernel[[oc, ic, ki, kj]])
    });
    let prod = crate::faer_parallelism::mat_mul_f64(&a64, &w64);
    let mut col_flat = vec![0.0f64; total_spatial * kernel_cols_per_group];
    for row in 0..total_spatial {
        let col_base = row * kernel_cols_per_group;
        for col in 0..kernel_cols_per_group {
            col_flat[col_base + col] = prod[(row, col)];
        }
    }
    col_flat
}

/// Route one conv group's f64 GEMM to the process-global sound engine (cuBLAS).
/// `None` ⇒ no engine / engine error ⇒ caller uses the CPU loop.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_via_engine(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
) -> Option<Vec<f64>> {
    crate::sound_f64_gemm::with_engine(|eng| {
        conv_group_col_flat_with_engine(
            eng,
            a_coefficients,
            kernel,
            oc_start,
            out_c_per_group,
            kernel_cols_per_group,
            total_spatial,
            spatial_per_obj,
            kernel_spatial,
            kw,
        )
    })
    .flatten()
}

/// Build the dense row-major f64 operands for one conv group and run the f64
/// GEMM on `eng`. The engine is a parameter (not the process-global slot) so
/// the operand construction is unit-testable regardless of global-slot state.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_with_engine(
    eng: &dyn GemmEngine,
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
) -> Option<Vec<f64>> {
    // A: (total_spatial × out_c_per_group) row-major; exact f32→f64 widening.
    let mut a64 = vec![0.0f64; total_spatial * out_c_per_group];
    for row in 0..total_spatial {
        let obj = row / spatial_per_obj;
        let spatial = row % spatial_per_obj;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            a64[row * out_c_per_group + oc_local] =
                f64::from(a_coefficients[[obj, oc * spatial_per_obj + spatial]]);
        }
    }
    // W: (out_c_per_group × kernel_cols_per_group) row-major.
    let mut w64 = vec![0.0f64; out_c_per_group * kernel_cols_per_group];
    for oc_local in 0..out_c_per_group {
        let oc = oc_start + oc_local;
        for col in 0..kernel_cols_per_group {
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            w64[oc_local * kernel_cols_per_group + col] = f64::from(kernel[[oc, ic, ki, kj]]);
        }
    }
    match eng.gemm_f64(
        total_spatial,
        out_c_per_group,
        kernel_cols_per_group,
        &a64,
        &w64,
    ) {
        Ok(v) if v.len() == total_spatial * kernel_cols_per_group => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adaptive_default_crown_mem_cap_mb, backward_buffer_bytes, backward_buffer_bytes_for,
        conv_crown_mem_cap_bytes, default_crown_mem_cap_mb, envelope_crown_mem_cap_bytes,
        faer_mat_to_row_major, guard_conv_crown_backward_buffer_with_cap,
        guard_conv_crown_scratch_bytes_with_cap, largest_conv_crown_scratch_bytes,
        narrowest_conv_crown_mem_cap, parse_meminfo_available_bytes, parse_meminfo_total_bytes,
        ConvCrownMemCap, ConvCrownMemCapSource, DEFAULT_CROWN_MEM_CAP_MB,
        MAX_ADAPTIVE_CROWN_MEM_CAP_MB,
    };
    use crate::tests::with_crown_mem_cap_mb;
    use faer::Mat;
    use ny_core::NyError;

    const MB: usize = 1024 * 1024;

    fn injected_cap(bytes: usize) -> ConvCrownMemCap {
        ConvCrownMemCap {
            bytes,
            source: ConvCrownMemCapSource::Policy,
        }
    }

    #[test]
    fn backward_buffer_bytes_is_rows_times_cols_times_f32_conv_crown_oom() {
        // 8192 rows (layer3.0.conv1 target_dim) x 32768 cols (conv input) f32.
        assert_eq!(backward_buffer_bytes(8192, 32768), 8192 * 32768 * 4);
        // Overflow saturates to usize::MAX so an overflowing shape always trips.
        assert_eq!(backward_buffer_bytes(usize::MAX, 2), usize::MAX);
    }

    /// The cuBLAS-seam operand construction (`conv_group_col_flat_with_engine`)
    /// must compute the SAME `col_flat = A·W` as the CPU triple-loop. Drives it
    /// with an explicit CPU f64 engine (deterministic, no GPU, no process-global
    /// slot — the `sound_f64_gemm` OnceLock materializes on first use anywhere
    /// in the test process, so a test must not depend on setting it) — catches
    /// any a64/w64 row/col indexing bug in the GEMM operands.
    #[test]
    fn conv_cublas_seam_operands_match_cpu_reference() {
        use super::conv_group_col_flat_with_engine;
        use ndarray::{Array2, ArrayD, IxDyn};
        use ny_core::NaiveCpuGemmEngine;

        let (num_obj, oc, icpg, kh, kw) = (2usize, 4usize, 2usize, 2usize, 2usize);
        let spatial_per_obj = 4usize; // grad_h*grad_w = 2*2
        let total_spatial = num_obj * spatial_per_obj;
        let kernel_spatial = kh * kw;
        let kcpg = icpg * kernel_spatial; // 8
        let a = Array2::from_shape_fn((num_obj, oc * spatial_per_obj), |(i, j)| {
            ((i * 7 + j * 3) % 11) as f32 * 0.1 - 0.4
        });
        let kernel = ArrayD::from_shape_fn(IxDyn(&[oc, icpg, kh, kw]), |d| {
            ((d[0] + d[1] * 2 + d[2] * 3 + d[3]) % 7) as f32 * 0.2 - 0.5
        });

        let eng = conv_group_col_flat_with_engine(
            &NaiveCpuGemmEngine,
            &a,
            &kernel,
            0,
            oc,
            kcpg,
            total_spatial,
            spatial_per_obj,
            kernel_spatial,
            kw,
        )
        .expect("engine path (explicit NaiveCpuGemmEngine)");

        // CPU reference: the exact triple-loop the seam replaces (oc_start = 0).
        let mut cpu = vec![0.0f64; total_spatial * kcpg];
        for row in 0..total_spatial {
            let obj = row / spatial_per_obj;
            let spatial = row % spatial_per_obj;
            let col_base = row * kcpg;
            for oc_local in 0..oc {
                let av = f64::from(a[[obj, oc_local * spatial_per_obj + spatial]]);
                if av == 0.0 {
                    continue;
                }
                for col in 0..kcpg {
                    let ic = col / kernel_spatial;
                    let rem = col % kernel_spatial;
                    let ki = rem / kw;
                    let kj = rem % kw;
                    cpu[col_base + col] += av * f64::from(kernel[[oc_local, ic, ki, kj]]);
                }
            }
        }
        assert_eq!(eng.len(), cpu.len());
        for i in 0..eng.len() {
            assert!(
                (eng[i] - cpu[i]).abs() <= 1e-9 + 1e-9 * cpu[i].abs(),
                "conv seam operand mismatch at {i}: engine={} cpu={}",
                eng[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn conv_crown_mem_cap_defaults_and_env_override_conv_crown_oom() {
        // Default cap when the env var is unset: the RAM-adaptive default.
        with_crown_mem_cap_mb_unset(|| {
            assert_eq!(
                conv_crown_mem_cap_bytes(),
                Some(default_crown_mem_cap_mb() * MB)
            );
        });
        // Env override sets an explicit cap.
        with_crown_mem_cap_mb("256", || {
            assert_eq!(conv_crown_mem_cap_bytes(), Some(256 * MB));
        });
        // A value of 0 disables this policy ceiling. The allocation guard still
        // applies an explicit Dense budget and process/container envelopes.
        with_crown_mem_cap_mb("0", || {
            assert_eq!(conv_crown_mem_cap_bytes(), None);
        });
        // A non-numeric value falls back to the RAM-adaptive default.
        with_crown_mem_cap_mb("not-a-number", || {
            assert_eq!(
                conv_crown_mem_cap_bytes(),
                Some(default_crown_mem_cap_mb() * MB)
            );
        });
    }

    /// #conv-crown-oom RAM-adaptive default: `clamp(total_ram/16, 512MiB, 16GiB)`,
    /// with unknown RAM keeping the fixed 512 MiB default. Driven by an explicit
    /// total-RAM parameter — never by reading /proc in the test.
    #[test]
    fn adaptive_default_cap_clamps_conv_crown_oom() {
        const GB: u64 = 1024 * 1024 * 1024;
        // Unknown RAM (meminfo unreadable/unparseable): fixed default.
        assert_eq!(
            adaptive_default_crown_mem_cap_mb(None),
            DEFAULT_CROWN_MEM_CAP_MB
        );
        // Small boxes clamp UP to the 512 MiB floor: 4 GB / 16 = 256 MiB.
        assert_eq!(
            adaptive_default_crown_mem_cap_mb(Some(4 * GB)),
            DEFAULT_CROWN_MEM_CAP_MB
        );
        // Exactly at the floor: 8 GB / 16 = 512 MiB.
        assert_eq!(
            adaptive_default_crown_mem_cap_mb(Some(8 * GB)),
            DEFAULT_CROWN_MEM_CAP_MB
        );
        // 24 GB competition box: 24 GB / 16 = 1536 MiB (past the old fixed cap,
        // still far from anything the box cannot hold as a pair).
        assert_eq!(adaptive_default_crown_mem_cap_mb(Some(24 * GB)), 1536);
        // 121 GB dev box: 121 GB / 16 = 7744 MiB > the 1.6 GiB VGG16 buffer.
        assert_eq!(adaptive_default_crown_mem_cap_mb(Some(121 * GB)), 7744);
        // Huge hosts clamp DOWN to the 16 GiB ceiling: 1 TB / 16 = 64 GiB.
        assert_eq!(
            adaptive_default_crown_mem_cap_mb(Some(1024 * GB)),
            MAX_ADAPTIVE_CROWN_MEM_CAP_MB
        );
        // Degenerate zero never yields a "disabled" (0) default: floor applies.
        assert_eq!(
            adaptive_default_crown_mem_cap_mb(Some(0)),
            DEFAULT_CROWN_MEM_CAP_MB
        );
    }

    /// The tiny std-only /proc/meminfo parser extracts total and currently
    /// available kB as bytes; malformed fields yield `None`.
    #[test]
    fn parse_meminfo_total_bytes_conv_crown_oom() {
        let real_shape = concat!(
            "MemTotal:       126619648 kB\n",
            "MemFree:         90000000 kB\n",
            "MemAvailable:    33554432 kB\n",
        );
        assert_eq!(
            parse_meminfo_total_bytes(real_shape),
            Some(126_619_648 * 1024)
        );
        assert_eq!(
            parse_meminfo_available_bytes(real_shape),
            Some(33_554_432 * 1024)
        );
        // MemTotal not on the first line is still found.
        assert_eq!(
            parse_meminfo_total_bytes("MemFree: 1 kB\nMemTotal: 2048 kB\n"),
            Some(2048 * 1024)
        );
        assert_eq!(parse_meminfo_total_bytes(""), None);
        assert_eq!(parse_meminfo_total_bytes("MemFree: 12345 kB\n"), None);
        assert_eq!(parse_meminfo_total_bytes("MemTotal: garbage kB\n"), None);
        assert_eq!(parse_meminfo_total_bytes("MemTotal:\n"), None);
        assert_eq!(parse_meminfo_total_bytes("MemTotal: 2048 bytes\n"), None);
        assert_eq!(parse_meminfo_available_bytes("MemTotal: 2048 kB\n"), None);
        // kB -> bytes overflow returns None instead of wrapping.
        assert_eq!(
            parse_meminfo_total_bytes(&format!("MemTotal: {} kB\n", u64::MAX)),
            None
        );
    }

    #[test]
    fn narrowest_envelope_and_shared_budget_win_conv_crown_oom() {
        const GB: u64 = 1024 * 1024 * 1024;
        // Policy=2 GiB, host has 32 GiB available => 2 GiB share, process has
        // only 8 GiB headroom => 512 MiB share, Dense pair budget => 1.5 GiB
        // per member. The enforced process envelope is correctly narrowest.
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(2 * GB as usize),
                Some(32 * GB),
                Some(8 * GB),
                Some(3 * GB as usize),
                2,
            )
            .bytes,
            512 * MB
        );
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(2 * GB as usize),
                Some(32 * GB),
                Some(8 * GB),
                Some(3 * GB as usize),
                2,
            )
            .source,
            ConvCrownMemCapSource::ProcessEnvelope
        );
        // An explicit smaller policy remains authoritative.
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(256 * MB),
                Some(32 * GB),
                Some(8 * GB),
                Some(3 * GB as usize),
                2,
            )
            .bytes,
            256 * MB
        );
        // Disabling only the policy does not disable the Dense result-set
        // budget: two retained buffers split 2 GiB into 1 GiB each.
        assert_eq!(
            narrowest_conv_crown_mem_cap(None, None, None, Some(2 * GB as usize), 2).bytes,
            GB as usize
        );
        // An exhausted kernel envelope fails closed at a zero-byte cap.
        assert_eq!(
            narrowest_conv_crown_mem_cap(None, None, Some(0), Some(usize::MAX), 1).bytes,
            0
        );
        assert_eq!(
            narrowest_conv_crown_mem_cap(None, None, Some(0), Some(usize::MAX), 1).source,
            ConvCrownMemCapSource::ProcessEnvelope
        );
        assert_eq!(envelope_crown_mem_cap_bytes(8 * GB), 512 * MB);
    }

    #[test]
    fn unset_dense_budget_preserves_large_host_adaptive_case() {
        const GB: u64 = 1024 * 1024 * 1024;
        let vgg_buffer_bytes = 1000 * 401_408 * size_of::<f32>();

        let adaptive =
            narrowest_conv_crown_mem_cap(Some(2 * GB as usize), Some(32 * GB), None, None, 2);
        assert_eq!(adaptive.bytes, 2 * GB as usize);
        assert_eq!(adaptive.source, ConvCrownMemCapSource::Policy);
        assert!(vgg_buffer_bytes < adaptive.bytes);

        let explicit_default = narrowest_conv_crown_mem_cap(
            Some(2 * GB as usize),
            Some(32 * GB),
            None,
            Some(2 * GB as usize),
            2,
        );
        assert_eq!(explicit_default.bytes, GB as usize);
        assert_eq!(
            explicit_default.source,
            ConvCrownMemCapSource::ExplicitDenseResultBudget
        );
        assert!(vgg_buffer_bytes > explicit_default.bytes);

        let fallback = narrowest_conv_crown_mem_cap(None, None, None, None, 2);
        assert_eq!(fallback.bytes, GB as usize);
        assert_eq!(
            fallback.source,
            ConvCrownMemCapSource::FallbackDenseResultBudget
        );
    }

    #[test]
    fn guard_trips_above_injected_cap_and_passes_below_conv_crown_oom() {
        // 8192 x 32768 f32 = 1 GiB per buffer > 512 MiB cap -> trips.
        let over =
            guard_conv_crown_backward_buffer_with_cap(8192, 32768, 2, 4, injected_cap(512 * MB));
        match over {
            Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                ..
            }) => {
                assert_eq!(required_bytes, 8192 * 32768 * 4);
                assert_eq!(budget_bytes, 512 * MB);
            }
            other => panic!("expected CpuMemoryExceeded over the cap, got {other:?}"),
        }

        // A small acasxu/sat_relu-class buffer (4096 x 4096 f32 = 64 MiB) is
        // far under the cap and must pass unchanged (no regression).
        assert!(guard_conv_crown_backward_buffer_with_cap(
            4096,
            4096,
            1,
            4,
            injected_cap(512 * MB),
        )
        .is_ok());
        // Exactly at the cap is allowed (strict `>` comparison).
        let at_cap_cols = (512 * MB) / (4 * 1024);
        assert!(guard_conv_crown_backward_buffer_with_cap(
            1024,
            at_cap_cols,
            1,
            4,
            injected_cap(512 * MB),
        )
        .is_ok());
    }

    #[test]
    fn allocation_charge_uses_actual_element_width_and_overflow_fails_closed() {
        assert_eq!(backward_buffer_bytes_for(1024, 1024, 4), 4 * MB);
        assert_eq!(backward_buffer_bytes_for(1024, 1024, 8), 8 * MB);
        assert!(
            guard_conv_crown_backward_buffer_with_cap(1024, 1024, 1, 4, injected_cap(4 * MB),)
                .is_ok()
        );
        assert!(matches!(
            guard_conv_crown_backward_buffer_with_cap(1024, 1024, 1, 8, injected_cap(4 * MB),),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        assert!(matches!(
            guard_conv_crown_backward_buffer_with_cap(
                usize::MAX,
                2,
                1,
                8,
                injected_cap(usize::MAX),
            ),
            Err(NyError::CpuMemoryExceeded { .. })
        ));

        // A=10x3, W=3x5, and GEMM/col2im=10x5: the 50-element GEMM result is
        // the largest visible transient and f64 charges twice the f32 bytes.
        let scratch_shapes = [(10, 3), (3, 5), (10, 5)];
        assert_eq!(largest_conv_crown_scratch_bytes(&scratch_shapes, 4), 200);
        assert_eq!(largest_conv_crown_scratch_bytes(&scratch_shapes, 8), 400);
        assert_eq!(
            largest_conv_crown_scratch_bytes(&[(usize::MAX, 2)], 8),
            usize::MAX
        );
        assert!(matches!(
            guard_conv_crown_scratch_bytes_with_cap(
                513,
                Some(ConvCrownMemCap {
                    bytes: 512,
                    source: ConvCrownMemCapSource::ProcessEnvelope,
                }),
            ),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        assert!(guard_conv_crown_scratch_bytes_with_cap(usize::MAX, None).is_err());
    }

    /// Run `f` with `NY_CROWN_MEM_CAP_MB` removed (serialized against other env
    /// mutators) so the default-cap branch is exercised deterministically.
    fn with_crown_mem_cap_mb_unset<T>(f: impl FnOnce() -> T) -> T {
        crate::tests::with_serialized_env_vars_removed(&["NY_CROWN_MEM_CAP_MB"], f)
    }

    /// `faer_mat_to_row_major` was changed from `vec![0.0; n]` + indexed
    /// overwrite to `Vec::with_capacity` + `push` to drop the wasted zero-init
    /// memset (#3795). This asserts the row-major layout and values are
    /// bit-identical to the explicit reference, i.e. the alloc change did not
    /// alter any numerics. (col-major faer Mat -> row-major flat)
    #[test]
    fn faer_mat_to_row_major_matches_reference_3795() {
        let rows = 5;
        let cols = 3;
        // Distinct value per cell so any transpose/index bug is caught.
        let mat = Mat::<f32>::from_fn(rows, cols, |r, c| (r * cols + c) as f32 + 0.5);

        let got = faer_mat_to_row_major(&mat, rows, cols);

        // Reference: original zero-init-then-overwrite implementation.
        let mut expected = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                expected[row * cols + col] = mat[(row, col)];
            }
        }

        assert_eq!(got.len(), rows * cols);
        assert_eq!(got, expected, "row-major flatten must be bit-identical");
        // Spot-check explicit layout: flat[row*cols + col] == mat[(row,col)].
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(got[row * cols + col], mat[(row, col)]);
            }
        }
    }
}
