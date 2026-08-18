// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use faer::Mat;
use ndarray::{s, Array2, ArrayD};
use ny_core::{checked_shape_product, ConvTranspose2dParams, GemmEngine, NyError, Result};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

use crate::faer_parallelism::mat_mul;

const DEADLINE_GEMM_ROW_CHUNK: usize = 256;
/// Maximum scalar multiply/add or scatter operations admitted between literal
/// deadline polls on the finite-deadline certified-f64 CPU path.
const DEADLINE_F64_CPU_POLL_OPS: usize = 4_096;
/// Per-buffer host-allocation ceiling for the finite bounded-executor Conv2d
/// seam. This matches the local facade's hard allocation cap.
pub(crate) const DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES: usize = 256 * 1024 * 1024;
const DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE: &str =
    "conv2d::ops_transpose_gemm::bounded_f64_buffer";

/// Launch one blocking f64 GEMM only while its finite authority is still live.
///
/// Operand construction can consume the remaining node budget. Injecting the
/// clock and launch as closures makes that exact boundary deterministic to
/// test: expiry must refuse before entering the unpollable faer call.
#[inline]
fn run_f64_gemm_before_deadline_with_clock<N, G>(
    deadline: Instant,
    now: N,
    gemm: G,
) -> Result<Mat<f64>>
where
    N: FnOnce() -> Instant,
    G: FnOnce() -> Mat<f64>,
{
    if now() >= deadline {
        return Err(per_node_deadline_exceeded());
    }
    Ok(gemm())
}

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

/// Floor for the per-buffer envelope share divisor
/// (#patches-envelope-nbuffers). Even a single live buffer never claims more
/// than a quarter of an observed envelope.
const MIN_ENVELOPE_SHARE_DIVISOR: u64 = 4;

/// Live cgroup/RLIMIT headroom divisor for Conv CROWN admissions.
///
/// Unlike the host-wide heuristic, a kernel envelope is a hard process peak.
/// The paired CPU path can retain roughly fourteen named allocations across
/// result, A/W/G scratch, row-major copies, and recursive objective chunks;
/// opaque GEMM workspace and allocator overhead are additional.  Restore the
/// conservative sixteenth share here until aggregate operation reservations
/// can replace independent per-buffer admission.
const PROCESS_ENVELOPE_SHARE_DIVISOR: u64 = 16;

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

/// Total system RAM in bytes, read once per process. `None` when unavailable
/// (unreadable, unparseable, or an unsupported platform), which keeps the
/// conservative fixed default.
fn total_system_ram_bytes() -> Option<u64> {
    static TOTAL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TOTAL.get_or_init(|| {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .as_deref()
            .and_then(parse_meminfo_total_bytes)
            .or_else(sysctl_total_ram_bytes)
    })
}

/// Total RAM on hosts without `/proc/meminfo` (#portable-host-ram).
///
/// WHY THIS EXISTS. Every memory policy in this file is derived from observed
/// host capacity, and the only probe was `/proc/meminfo` — Linux-only. On any
/// other host `total_system_ram_bytes()` returned `None`,
/// `adaptive_default_crown_mem_cap_mb` fell to the fixed
/// [`DEFAULT_CROWN_MEM_CAP_MB`] (512 MiB), and NY sized a memory policy for a
/// machine it had never measured. Observed on a 24 GB macOS host: the effective
/// per-buffer cap was 512 MiB against a genuine 936 MiB conv-backward
/// requirement, so CROWN targets took the sound-but-loose IBP fallback on a
/// machine with ~20 GB free. The verifier could not see its own environment.
///
/// `sysctl -n hw.memsize` is the Apple-host query for physical memory.
/// It is invoked at most once per process (the caller caches in a `OnceLock`),
/// costs a few milliseconds, and adds no dependency. Any failure — missing
/// binary, non-zero exit, unparseable output — yields `None` and preserves the
/// previous conservative fallback exactly.
///
/// Soundness-neutral: this only informs a policy CEILING that chooses between
/// computing the tight CROWN bound and taking the looser-but-sound IBP
/// fallback. A wrong or absent reading can cost tightness, never validity.
fn sysctl_total_ram_bytes() -> Option<u64> {
    if !cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_vendor = "apple"
    )) {
        return None;
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_sysctl_memsize(&String::from_utf8_lossy(&out.stdout))
}

/// Pure parser for `sysctl -n hw.memsize` output (a single decimal byte count).
fn parse_sysctl_memsize(raw: &str) -> Option<u64> {
    let value = raw.trim().parse::<u64>().ok()?;
    (value > 0).then_some(value)
}

/// Currently available host RAM, tagged with the probe that produced it.
///
/// Unlike total capacity this is intentionally not cached: sibling jobs can
/// consume or release memory during a long run. The tag matters for diagnosis:
/// a macOS host has no `/proc/meminfo`, so reporting the estimate under that
/// name sent an operator looking for a file that does not exist.
fn available_system_ram_observation() -> Option<(u64, ConvCrownMemCapSource)> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(parse_meminfo_available_bytes)
        .map(|bytes| (bytes, ConvCrownMemCapSource::HostAvailable))
        // #portable-host-ram: without a per-moment "available" probe on this
        // platform, fall back to a conservative FRACTION of measured physical
        // memory rather than to no observation at all. Half of physical RAM is
        // deliberately pessimistic (it assumes the rest of the machine may be
        // busy) but is still a measurement, which is strictly better than the
        // fixed floor this path previously produced.
        .or_else(|| {
            total_system_ram_bytes()
                .map(|total| (total / 2, ConvCrownMemCapSource::HostPhysicalHalf))
        })
}

/// Effective default cap in MiB: RAM-adaptive on hosts whose total RAM is
/// readable, the fixed 512 MiB otherwise.
fn default_crown_mem_cap_mb() -> usize {
    adaptive_default_crown_mem_cap_mb(total_system_ram_bytes())
}

/// Test-only accessor for the effective RAM-adaptive policy cap in MiB, so the
/// macOS `sysctl hw.memsize` probe can be asserted end to end.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn effective_crown_mem_cap_mb_for_test() -> usize {
    default_crown_mem_cap_mb()
}

/// Per-buffer share of a host-wide memory observation
/// (#patches-envelope-nbuffers).
///
/// The divisor is derived from how many buffers this call site ACTUALLY holds
/// live, not from a blanket constant. `n_buffers` is already threaded to every
/// caller (1 for a standalone result, 2 for a retained lower/upper pair); the
/// allowance is `2 * n_buffers`, i.e. the named buffers plus an equal budget for
/// transient scratch, floored at [`MIN_ENVELOPE_SHARE_DIVISOR`] so a single
/// buffer can still never claim more than a quarter of the envelope.
///
/// This larger share applies only to a reclaimable host observation, where a
/// refusal is a performance choice rather than a kernel kill.  The separate
/// cgroup/RLIMIT helper below retains a `/16` boundary because the complete
/// paired/chunked operation holds far more than the result buffers named at an
/// individual admission.  Applying the old 16-way split to both sources left
/// 7/8 of otherwise available host RAM unusable and forced sound-but-loose IBP
/// fallbacks on buffers the host holds comfortably. Measured on TinyYOLO
/// (yolo_2023) on a 24 GB host: the effective cap was 512 MiB against a genuine
/// 936 MiB requirement — a 1.7x shortfall — and three of eight demanded CROWN
/// targets reverted to IBP for that reason alone once the kernel-selection fix
/// removed the competing time pressure.
///
/// Soundness-neutral: this ceiling only chooses between computing the tight
/// CROWN bound and taking the looser-but-sound IBP fallback. It never affects a
/// bound's validity, only whether the tighter one is attempted. The `MAX_...`
/// clamp is retained so this stays an OOM backstop rather than an open grant.
fn envelope_crown_mem_cap_bytes(headroom_bytes: u64, n_buffers: usize) -> usize {
    let divisor = (n_buffers as u64)
        .saturating_mul(2)
        .max(MIN_ENVELOPE_SHARE_DIVISOR);
    let cap = headroom_bytes / divisor;
    usize::try_from(cap.min((MAX_ADAPTIVE_CROWN_MEM_CAP_MB as u64) * 1024 * 1024))
        .unwrap_or(usize::MAX)
}

/// Per-buffer share of a kernel-enforced process envelope.  This is separate
/// from [`envelope_crown_mem_cap_bytes`] so recovering useful host RAM through
/// the measured live-buffer heuristic can never weaken the cgroup/RLIMIT OOM
/// boundary.
fn process_envelope_crown_mem_cap_bytes(headroom_bytes: u64, n_buffers: usize) -> usize {
    let divisor = (n_buffers as u64)
        .saturating_mul(2)
        .max(PROCESS_ENVELOPE_SHARE_DIVISOR);
    let cap = headroom_bytes / divisor;
    usize::try_from(cap.min((MAX_ADAPTIVE_CROWN_MEM_CAP_MB as u64) * 1024 * 1024))
        .unwrap_or(usize::MAX)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConvCrownMemCapSource {
    Policy,
    HostAvailable,
    HostPhysicalHalf,
    ProcessEnvelope,
    ExplicitDenseResultBudget,
    FallbackDenseResultBudget,
}

impl ConvCrownMemCapSource {
    const fn label(self) -> &'static str {
        match self {
            Self::Policy => "NY_CROWN_MEM_CAP_MB/adaptive policy",
            Self::HostAvailable => "/proc/meminfo MemAvailable headroom",
            Self::HostPhysicalHalf => "half of measured physical RAM (no MemAvailable probe)",
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
    host_available: Option<(u64, ConvCrownMemCapSource)>,
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
        host_available.map(|(available, source)| ConvCrownMemCap {
            bytes: envelope_crown_mem_cap_bytes(available, n_buffers),
            source,
        }),
        process_headroom_bytes.map(|headroom| ConvCrownMemCap {
            bytes: process_envelope_crown_mem_cap_bytes(headroom, n_buffers),
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

/// Was `NY_CROWN_MEM_CAP_MB` set to a parseable value by the operator?
///
/// #explicit-cap-overrides-heuristic. An explicitly configured cap is an
/// ASSERTION about the machine, and the docstring on `conv_crown_mem_cap_bytes`
/// has always promised `NY_CROWN_MEM_CAP_MB env var > host-RAM-adaptive
/// default`. It did not deliver: `narrowest_conv_crown_mem_cap` takes the MIN
/// over every source, so the explicit value could only ever LOWER the cap --
/// useless for the one case an operator reaches for it.
///
/// Measured on yolo_2023 (2026-07-29): `NY_CROWN_MEM_CAP_MB=16000` left the
/// deciding bound BIT-IDENTICAL, because the host-RAM heuristic
/// (half of physical RAM, then divided again by `MIN_ENVELOPE_SHARE_DIVISOR`,
/// i.e. RAM/8 == 3 GiB here) won the min and refused transients of 3.7, 4.9,
/// 6.3 and 8.4 GB on a 24 GB machine.
fn conv_crown_mem_cap_is_explicit() -> bool {
    std::env::var(CROWN_MEM_CAP_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .is_some()
}

fn effective_conv_crown_mem_cap(n_buffers: usize) -> ConvCrownMemCap {
    narrowest_conv_crown_mem_cap(
        conv_crown_mem_cap_bytes(),
        // #explicit-cap-overrides-heuristic: suppress ONLY the host-RAM guess.
        // `process_memory_headroom_bytes` (cgroup / RLIMIT) and the explicit
        // Dense budget still bind below -- those are kernel-enforced or
        // operator-stated, not heuristics, and exceeding them is a real OOM.
        if conv_crown_mem_cap_is_explicit() {
            None
        } else {
            available_system_ram_observation()
        },
        crate::network::crown_memory::process_memory_headroom_bytes(),
        crate::network::crown_memory::explicit_cpu_crown_dense_budget_bytes(),
        n_buffers,
    )
}

fn effective_conv_crown_envelope_cap() -> Option<ConvCrownMemCap> {
    [
        // A standalone transient: one live buffer.
        // #explicit-cap-overrides-heuristic: same rule as
        // `effective_conv_crown_mem_cap` -- an operator-stated cap displaces the
        // host-RAM guess here too, or the envelope path would silently reimpose
        // the very ceiling the operator just raised.
        if conv_crown_mem_cap_is_explicit() {
            conv_crown_mem_cap_bytes().map(|bytes| ConvCrownMemCap {
                bytes,
                source: ConvCrownMemCapSource::Policy,
            })
        } else {
            available_system_ram_observation().map(|(available, source)| ConvCrownMemCap {
                bytes: envelope_crown_mem_cap_bytes(available, 1),
                source,
            })
        },
        crate::network::crown_memory::process_memory_headroom_bytes().map(|headroom| {
            ConvCrownMemCap {
                bytes: process_envelope_crown_mem_cap_bytes(headroom, 1),
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

fn guard_deadline_bounded_conv_f64_buffer(elements: usize) -> Result<()> {
    let required_bytes = elements.saturating_mul(size_of::<f64>());
    if required_bytes > DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES,
            site: DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE,
        });
    }
    Ok(())
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

/// Objective-chunk size for the f64 coefficient recompute, or `None` to keep
/// the single-shot path exactly as it is (#f64-recompute-objective-chunk).
///
/// Returns `Some(chunk)` ONLY when all of the following hold, so the
/// unchunked path is untouched wherever it already works and a genuinely
/// infeasible shape still receives the honest refusal:
///   - an envelope cap is actually observable (no cap => no refusal => no need),
///   - the single-shot transient EXCEEDS that cap,
///   - a strictly smaller objective chunk brings every objective-scaled
///     transient (and the objective-independent weight block) under it.
///
/// The returned chunk is always `>= 1` and `< num_objectives`.
fn f64_recompute_objective_chunk(
    num_objectives: usize,
    spatial_per_obj: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
) -> Option<usize> {
    f64_recompute_objective_chunk_with_cap(
        effective_conv_crown_envelope_cap().map(|cap| cap.bytes),
        num_objectives,
        spatial_per_obj,
        out_c_per_group,
        kernel_cols_per_group,
    )
}

fn f64_recompute_objective_chunk_with_cap(
    cap: Option<usize>,
    num_objectives: usize,
    spatial_per_obj: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
) -> Option<usize> {
    if num_objectives <= 1 || spatial_per_obj == 0 {
        return None;
    }
    let cap = cap?;
    let total_spatial = num_objectives.checked_mul(spatial_per_obj)?;
    let full = largest_conv_crown_scratch_bytes(
        &[
            (total_spatial, out_c_per_group),
            (out_c_per_group, kernel_cols_per_group),
            (total_spatial, kernel_cols_per_group),
        ],
        size_of::<f64>(),
    );
    if full <= cap {
        return None; // already fits: unchanged single-shot behavior
    }
    // The `(out_c_per_group, kernel_cols_per_group)` weight block does not
    // shrink with the objective range, so chunking cannot rescue it.
    if backward_buffer_bytes_for(out_c_per_group, kernel_cols_per_group, size_of::<f64>()) > cap {
        return None;
    }
    let per_objective = largest_conv_crown_scratch_bytes(
        &[
            (spatial_per_obj, out_c_per_group),
            (spatial_per_obj, kernel_cols_per_group),
        ],
        size_of::<f64>(),
    );
    if per_objective == 0 || per_objective > cap {
        return None; // even one objective is over the cap: keep the refusal
    }
    let chunk = (cap / per_objective).max(1);
    if chunk >= num_objectives {
        None
    } else {
        Some(chunk)
    }
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
        // State the CONSEQUENCE, not just the refusal. "Sound fallback" reads as
        // benign; what actually happens is that this target keeps its IBP bound,
        // and on a deep conv graph IBP width compounds multiplicatively per
        // layer, so a single refusal here can cost orders of magnitude at the
        // output. Name the shortfall and the exact knob so the reader can act.
        warn!(
            "Conv2d CROWN backward memory cap triggered: {n_buffers} buffer(s) of \
             [{num_objectives} x {conv_in_size}] with {element_bytes}-byte elements require \
             {required} bytes each \
             (> effective cap {} bytes from {}, short by {} bytes / {:.1}x); this target KEEPS ITS \
             IBP BOUND — on a deep conv graph that loss compounds multiplicatively at every \
             later layer. Raise {CROWN_MEM_CAP_ENV} (MiB) if the host can afford it, or accept \
             the looser bound.",
            cap.bytes,
            cap.source.label(),
            required.saturating_sub(cap.bytes),
            required as f64 / (cap.bytes.max(1) as f64)
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
/// chunking) branch applies. In every other case — and on any non-authoritative
/// per-group fused error — this delegates to two independent
/// [`conv2d_transpose_batched_gemm_grouped_with_deadline`] calls, so behavior on
/// those paths is exactly unchanged. An authoritative deadline reported by the
/// engine is terminal instead of being retried on the CPU.
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
    // #deadline-gemm: a finite deadline must NOT cost the GEMM.
    //
    // This used to return the certified scalar f64 contraction for EVERY
    // finite-deadline pair. Since every scored run sets a deadline, that made
    // the scalar route the only conv CROWN backward production ever executes —
    // while tests and benchmarks (no deadline) took the fast GEMM. It also made
    // the entire chunked-deadline GEMM path below dead code: the early `return`
    // meant `use_chunked_deadline = deadline.is_some() && ...` was evaluated
    // only when `deadline` was `None`, so it was always false.
    //
    // The deadline contract is preserved without paying that cost. A generic
    // `GemmEngine` still has no launch-before-deadline or cooperative
    // cancellation contract, so it stays refused under a finite deadline
    // (`admitted_engine` below). What we take instead is the CHUNKED CPU GEMM,
    // which polls between groups and between row chunks of
    // `DEADLINE_GEMM_ROW_CHUNK`, so overshoot is bounded by one chunk rather
    // than by one opaque full-size GEMM.
    //
    // Soundness is unchanged: this restores the path the caller's certified
    // error channel was sized for in the first place. The scalar f64
    // contraction is strictly MORE accurate than the f32 GEMM, and the comment
    // it carried said so — "the caller's f32 γ_n·S channel remains conservative
    // for this more accurate contraction". Returning to the GEMM returns to the
    // error the `γ_n^f32·S` channel already charges.
    //
    // Dilated kernels keep the scalar route: the batched GEMM kernel has no
    // dilation support (see the doc comment on this function).
    let dilated = dh != 1 || dw != 1;
    if let (Some(limit), true) = (deadline, dilated) {
        let finite_side = |coefficients: &Array2<f32>| -> Result<Array2<f32>> {
            let widened = conv2d_transpose_backward_coeff_f64_with_deadline(
                coefficients,
                kernel,
                stride,
                padding,
                dilation,
                output_size,
                grad_size,
                out_channels,
                groups,
                2,
                Some(limit),
            )?;
            if Instant::now() >= limit {
                return Err(per_node_deadline_exceeded());
            }
            let mut point = Array2::<f32>::zeros(widened.raw_dim());
            for (index, (dst, &value)) in point.iter_mut().zip(widened.iter()).enumerate() {
                if index.is_multiple_of(DEADLINE_F64_CPU_POLL_OPS) && Instant::now() >= limit {
                    return Err(per_node_deadline_exceeded());
                }
                *dst = value as f32;
            }
            if Instant::now() >= limit {
                return Err(per_node_deadline_exceeded());
            }
            Ok(point)
        };
        return Ok((finite_side(lower_a)?, finite_side(upper_a)?));
    }

    // Under a finite deadline the external engine is refused (no
    // launch-before-deadline contract); the chunked CPU GEMM below carries the
    // work with bounded, pollable overshoot.
    let engine = if deadline.is_some() { None } else { engine };

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
            deadline,
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
/// cannot use the fused engine call, so the caller falls back; an authoritative
/// deadline remains terminal. The non-fused portions (matrix extraction,
/// col2im scatter) mirror the single-matrix path in
/// [`conv2d_transpose_batched_gemm_grouped_with_deadline`] exactly.
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
    deadline: Option<Instant>,
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
            Err(error) if engine_error_is_terminal(&error, engine, deadline) => {
                return Err(error);
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

/// A typed engine error changes the historical CPU-fallback contract only when
/// this helper owns an explicit deadline, the engine proves that its call-local
/// CROWN deadline has expired, or a bounded host facade refuses the allocation
/// that local fallback would recreate. Ordinary engines retain legacy fallback.
#[inline]
fn engine_error_is_terminal(
    error: &NyError,
    engine: &dyn GemmEngine,
    deadline: Option<Instant>,
) -> bool {
    (error.is_cpu_memory_exceeded() && engine.forbids_unbounded_cpu_fallback())
        || (error.is_deadline_exceeded()
            && (deadline.is_some()
                || matches!(
                    engine.poll_crown_backward_deadline(),
                    Err(error) if error.is_deadline_exceeded()
                )))
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
                    Err(error) if engine_error_is_terminal(&error, eng, deadline) => {
                        return Err(error);
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
                    Err(e) if engine_error_is_terminal(&e, eng, deadline) => {
                        return Err(e);
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
                Err(error) if engine_error_is_terminal(&error, eng, deadline) => {
                    return Err(error);
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
                    Err(e) if engine_error_is_terminal(&e, eng, deadline) => {
                        return Err(e);
                    }
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
///     mirroring the Linear `aw_f64_with_abssum` fix. The col2im scatter stays on
///     CPU, while large per-group contractions may use the process-global sound
///     f64 engine (the single-group path supports any outer `groups` layout).
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
    conv2d_transpose_backward_coeff_f64_with_deadline(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        groups,
        resident_result_buffers,
        None,
    )
}

/// Deadline-authoritative variant of
/// [`conv2d_transpose_backward_coeff_f64`].
///
/// `deadline: None` retains the historical process-global sound-f64 GEMM/faer
/// implementation byte-for-byte. Under a finite deadline, large contractions
/// may enter only [`GemmEngine::gemm_f64_with_deadline`], whose output-axis
/// tiling retains the complete contraction and bounds each accelerator
/// dispatch. Missing/unsupported engines and small products use the pollable
/// scalar f64 CPU contraction. The col2im scatter remains pollable CPU work,
/// and every path polls before publishing the completed result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_backward_coeff_f64_with_deadline(
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
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        grad_size,
        out_channels,
        groups,
        resident_result_buffers,
        None,
        deadline,
    )
}

/// Finite-deadline f64 recompute with an optional bounded host fallback.
///
/// Ordinary callers pass no engine and retain the historical global
/// accelerator/faer route. The bounded shared executor passes its local
/// deadline-polling facade, which is used exclusively and is never replaced by
/// a process-global accelerator or opaque faer work on refusal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
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
    bounded_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Array2<f64>> {
    let bounded_engine = bounded_pollable_host_engine(bounded_engine)?;
    let poll_authority = || -> Result<()> {
        if let Some(engine) = bounded_engine {
            engine.poll_crown_backward_deadline()?;
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(per_node_deadline_exceeded());
        }
        Ok(())
    };
    poll_authority()?;
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
    if bounded_engine.is_some() {
        let bounded_a = total_spatial
            .checked_mul(out_c_per_group)
            .ok_or_else(|| NyError::InvalidSpec("bounded Conv2d f64 A size overflow".into()))?;
        let bounded_w = out_c_per_group
            .checked_mul(kernel_cols_per_group)
            .ok_or_else(|| {
                NyError::InvalidSpec("bounded Conv2d f64 weight size overflow".into())
            })?;
        let bounded_col = total_spatial
            .checked_mul(kernel_cols_per_group)
            .ok_or_else(|| {
                NyError::InvalidSpec("bounded Conv2d f64 column size overflow".into())
            })?;
        let bounded_result = num_objectives.checked_mul(conv_in_size).ok_or_else(|| {
            NyError::InvalidSpec("bounded Conv2d f64 result size overflow".into())
        })?;
        for elements in [bounded_a, bounded_w, bounded_col, bounded_result] {
            guard_deadline_bounded_conv_f64_buffer(elements)?;
        }
    }

    // #f64-recompute-objective-chunk: the dominant transient here is the
    // `(total_spatial x kernel_cols_per_group)` f64 column matrix, and
    // `total_spatial = num_objectives * spatial_per_obj` — so it grows with the
    // OBJECTIVE COUNT, not with anything intrinsic to the convolution. On
    // yolo_2023 the Conv_12 intermediate target (10,816 objectives, 26x26 grad,
    // 3x3x16 kernel) asks for 8,422,981,632 bytes in one shot against a 3 GiB
    // envelope cap and is refused.
    //
    // That refusal is NOT harmless. `Conv2d::propagate_linear_with_engine_and_deadline`
    // maps any non-deadline error from this function to `Ok(None)` (a "failed
    // recompute"), which degrades EVERY row of the relation to `+/-inf` bias —
    // the target then concretizes to `[-inf, inf]` and the whole CROWN pass for
    // it is discarded (measured: Conv_12 returned `[-inf, inf]` with no NaN and
    // no reported failure anywhere else).
    //
    // Objectives are INDEPENDENT in both halves of this routine: the GEMM row
    // index is `obj * spatial_per_obj + spatial_offset`, and the col2im scatter
    // writes only into `result_flat[obj*conv_in_size .. (obj+1)*conv_in_size]`.
    // So splitting the objective range computes each row from the IDENTICAL
    // operand set with the IDENTICAL term count, keeping the transient under the
    // cap. Note the results are BOUND-EQUIVALENT, not necessarily bit-equal:
    // faer selects its blocking from the matrix dimensions, so a narrower chunk
    // may reduce in a different order. That freedom is already granted by the
    // published certificate -- `cast_err = |c64 - stored|` is MEASURED per element
    // and the Higham parameter `n` is unchanged by reordering -- and it is the
    // same argument this file already relies on for the cuBLAS seam and the
    // `rayon::join` pair. Only entered
    // when the single-shot transient would be REFUSED (otherwise this returns
    // `None` and the path below is byte-for-byte unchanged), and only when a
    // chunk actually fits — a genuinely infeasible shape still takes the honest
    // refusal below.
    let chunk_objectives = bounded_engine.is_none().then(|| {
        f64_recompute_objective_chunk(
            num_objectives,
            spatial_per_obj,
            out_c_per_group,
            kernel_cols_per_group,
        )
    });
    if let Some(chunk_objectives) = chunk_objectives.flatten() {
        // The FULL result buffer is still charged in full, exactly as the
        // single-shot path charges it (each chunk call re-charges only its own
        // smaller slab, which is additionally live).
        guard_conv_crown_backward_f64_buffer(
            num_objectives,
            conv_in_size,
            resident_result_buffers,
        )?;
        let mut result = Array2::<f64>::zeros((num_objectives, conv_in_size));
        let mut start = 0usize;
        while start < num_objectives {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(per_node_deadline_exceeded());
            }
            let end = start.saturating_add(chunk_objectives).min(num_objectives);
            let a_chunk = a_coefficients.slice(s![start..end, ..]).to_owned();
            let part = conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
                &a_chunk,
                kernel,
                stride,
                padding,
                dilation,
                output_size,
                grad_size,
                out_channels,
                groups,
                resident_result_buffers,
                bounded_engine,
                deadline,
            )?;
            result.slice_mut(s![start..end, ..]).assign(&part);
            start = end;
        }
        return Ok(result);
    }

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
    poll_authority()?;

    let result_len = num_objectives.checked_mul(conv_in_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_transpose_backward_coeff_f64: result alloc overflow: {num_objectives} * {conv_in_size}"
        ))
    })?;
    if bounded_engine.is_some() {
        guard_deadline_bounded_conv_f64_buffer(result_len)?;
    }
    let mut result_flat = if bounded_engine.is_some() {
        let mut values = Vec::new();
        values
            .try_reserve_exact(result_len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: result_len.saturating_mul(size_of::<f64>()),
                budget_bytes: DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES,
                site: DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE,
            })?;
        while values.len() < result_len {
            poll_authority()?;
            values.resize(
                values
                    .len()
                    .saturating_add(DEADLINE_F64_CPU_POLL_OPS)
                    .min(result_len),
                0.0,
            );
        }
        values
    } else {
        vec![0.0f64; result_len]
    };
    poll_authority()?;

    for g in 0..groups {
        poll_authority()?;
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;

        // f64 GEMM: (total_spatial, oc_per_group) x (oc_per_group, kernel_cols).
        // col_flat[row, col] = Σ_oc a[row, oc] · w[oc, col]  (f64 accumulation).
        // For large products this routes the single f64 GEMM to cuBLAS Dgemm (the
        // conv twin of the Linear `aw_via_engine` seam); the col2im scatter below
        // stays f64 on CPU. SOUND: the conv certified error (`cast + γ_n^f64·S +
        // prop`, `S` over-bounded by `row_max(a)·‖kernel‖₁`) is summation-order
        // independent, so cuBLAS's reduction order is safe.
        let col_flat = if let Some(engine) = bounded_engine {
            conv_group_col_flat_f64_bounded_engine(
                engine,
                a_coefficients,
                kernel,
                oc_start,
                out_c_per_group,
                kernel_cols_per_group,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
            )?
        } else {
            match deadline {
                Some(limit) => conv_group_col_flat_f64_with_deadline(
                    a_coefficients,
                    kernel,
                    oc_start,
                    out_c_per_group,
                    kernel_cols_per_group,
                    total_spatial,
                    spatial_per_obj,
                    kernel_spatial,
                    kw,
                    limit,
                )?,
                None => conv_group_col_flat_f64(
                    a_coefficients,
                    kernel,
                    oc_start,
                    out_c_per_group,
                    kernel_cols_per_group,
                    total_spatial,
                    spatial_per_obj,
                    kernel_spatial,
                    kw,
                ),
            }
        };

        // col2im scatter (f64 accumulation into the input pixels), parallelized
        // over objectives. Objective `obj` owns the disjoint output chunk
        // `result_flat[obj*conv_in_size .. (obj+1)*conv_in_size]`, and — since
        // each `out_idx` belongs to exactly one group (its `ic` is in this
        // group's channel range) — the per-`(obj, out_idx)` `+=` order stays the
        // fixed `gy -> gx -> ic_local -> ki -> kj` order it had when `obj` was
        // the innermost loop, so the result is bit-for-bit unchanged; only the
        // cross-objective parallelism is new. `conv_in_size == 0` scatters
        // nothing (par_chunks_mut requires a nonzero chunk length).
        //
        // #interm-scatter-deadline: this branch used to additionally require
        // `deadline.is_none()`, which silently excluded the single heaviest
        // caller — the root intermediate tightener threads `Some(pass_deadline)`
        // into every conv, so it always fell to the SERIAL scatter below on a
        // 20-core box. The GEMM half of this same routine already got the
        // deadline-aware treatment (`#cifar100-deadline-gemm` row-chunks faer so
        // it can be polled); the col2im scatter was simply never given the twin.
        //
        // A deadline changes nothing about the arithmetic, so the bit-identity
        // argument above carries over verbatim — the only new obligation is
        // polling, which is satisfied per objective chunk (the serial path polls
        // once per chunk too, at the same boundary; only its extra every-4096-MAC
        // check is dropped, and one chunk is ~10^5 MACs ≈ tens of microseconds).
        // Expiry short-circuits `try_for_each` and returns the same
        // `per_node_deadline_exceeded()` error, so the caller still fails closed
        // onto its pre-pass sound box; the partially-written `result_flat` is
        // discarded with the `Err` and never observed.
        //
        // `bounded_engine.is_some()` stays serial: that seam must poll the ENGINE
        // (a `&dyn GemmEngine`, not required to be `Sync`), and it is the bounded
        // executor's authority path rather than a throughput path.
        if conv_in_size > 0 && bounded_engine.is_none() {
            result_flat
                .par_chunks_mut(conv_in_size)
                .enumerate()
                .try_for_each(|(obj, obj_out)| -> Result<()> {
                    // Deadline-only poll: `bounded_engine` is `None` on this
                    // branch, so `poll_authority`'s engine arm cannot fire and
                    // the check needs nothing that is not `Copy + Sync`.
                    if deadline.is_some_and(|limit| Instant::now() >= limit) {
                        return Err(per_node_deadline_exceeded());
                    }
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
                    Ok(())
                })?;
        } else if conv_in_size > 0 {
            let mut scatter_ops = 0usize;
            for (obj, obj_out) in result_flat.chunks_mut(conv_in_size).enumerate() {
                poll_authority()?;
                for gy in 0..grad_h {
                    for gx in 0..grad_w {
                        let spatial_offset = gy * grad_w + gx;
                        let col_row = obj * spatial_per_obj + spatial_offset;
                        for ic_local in 0..in_c_per_group {
                            let ic = ic_start + ic_local;
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    if scatter_ops.is_multiple_of(DEADLINE_F64_CPU_POLL_OPS) {
                                        poll_authority()?;
                                    }
                                    scatter_ops = scatter_ops.wrapping_add(1);
                                    let ih = (gy * sh + ki * dh) as isize - ph as isize;
                                    let iw = (gx * sw + kj * dw) as isize - pw as isize;
                                    if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let col_idx = ic_local * kernel_spatial + ki * kw + kj;
                                        let out_idx =
                                            ic * input_spatial + ih as usize * in_w + iw as usize;
                                        obj_out[out_idx] +=
                                            col_flat[col_row * kernel_cols_per_group + col_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    poll_authority()?;
    let result = Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d f64 col2im reshape: {e}")))?;
    poll_authority()?;
    Ok(result)
}

/// Resolve the bounded-executor host GEMM capability without permitting a
/// fail-open downgrade to the process-global or opaque CPU implementation.
///
/// Ordinary engines are not authorities for this local bounded seam and are
/// therefore reported as `None`. Once an engine marks unbounded CPU fallback
/// as forbidden, however, it must also advertise the pollable host-GEMM
/// contract; silently treating a partial implementation as an ordinary engine
/// would violate the bounded executor's admission guarantee.
pub(crate) fn bounded_pollable_host_engine(
    engine: Option<&dyn GemmEngine>,
) -> Result<Option<&dyn GemmEngine>> {
    let Some(engine) = engine else {
        return Ok(None);
    };
    if !engine.forbids_unbounded_cpu_fallback() {
        return Ok(None);
    }
    if !engine.provides_deadline_pollable_host_gemm() {
        return Err(NyError::UnsupportedOp(
            "Conv2d bounded f64 requires a pollable capped host engine".into(),
        ));
    }
    Ok(Some(engine))
}

/// One Conv2d f64 group through the local pollable host facade.
///
/// This is deliberately separate from the accelerator helper below: every
/// buffer is capped and fallibly reserved, ordinary GEMM is permitted only by
/// the dedicated pollable-host capability, and no error can enter global/faer
/// fallback.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64_bounded_engine(
    engine: &dyn GemmEngine,
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
) -> Result<Vec<f64>> {
    const PREP_POLL_ELEMENTS: usize = 1 << 12;

    if !engine.forbids_unbounded_cpu_fallback() || !engine.provides_deadline_pollable_host_gemm() {
        return Err(NyError::UnsupportedOp(
            "Conv2d bounded f64 requires a pollable capped host engine".into(),
        ));
    }
    let poll = || engine.poll_crown_backward_deadline();
    poll()?;
    if spatial_per_obj == 0 || kernel_spatial == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(
            "Conv2d bounded f64 received zero spatial metadata".into(),
        ));
    }

    let a_len = total_spatial
        .checked_mul(out_c_per_group)
        .ok_or_else(|| NyError::InvalidSpec("Conv2d bounded f64 A length overflow".into()))?;
    let w_len = out_c_per_group
        .checked_mul(kernel_cols_per_group)
        .ok_or_else(|| NyError::InvalidSpec("Conv2d bounded f64 W length overflow".into()))?;
    let output_len = total_spatial
        .checked_mul(kernel_cols_per_group)
        .ok_or_else(|| NyError::InvalidSpec("Conv2d bounded f64 output length overflow".into()))?;
    for elements in [a_len, w_len, output_len] {
        guard_deadline_bounded_conv_f64_buffer(elements)?;
    }

    let reserve = |elements: usize| -> Result<Vec<f64>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: elements.saturating_mul(size_of::<f64>()),
                budget_bytes: DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES,
                site: DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE,
            })?;
        Ok(values)
    };

    let mut a64 = reserve(a_len)?;
    for row in 0..total_spatial {
        let obj = row / spatial_per_obj;
        let spatial = row % spatial_per_obj;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            let value = a_coefficients[[obj, oc * spatial_per_obj + spatial]];
            if !value.is_finite() {
                return Err(NyError::InvalidSpec(
                    "Conv2d bounded f64 A contains non-finite data".into(),
                ));
            }
            a64.push(f64::from(value));
            if a64.len().is_multiple_of(PREP_POLL_ELEMENTS) {
                poll()?;
            }
        }
    }

    let mut w64 = reserve(w_len)?;
    for oc_local in 0..out_c_per_group {
        let oc = oc_start + oc_local;
        for col in 0..kernel_cols_per_group {
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            let value = kernel[[oc, ic, ki, kj]];
            if !value.is_finite() {
                return Err(NyError::InvalidSpec(
                    "Conv2d bounded f64 W contains non-finite data".into(),
                ));
            }
            w64.push(f64::from(value));
            if w64.len().is_multiple_of(PREP_POLL_ELEMENTS) {
                poll()?;
            }
        }
    }
    poll()?;

    let result = engine.gemm_f64(
        total_spatial,
        out_c_per_group,
        kernel_cols_per_group,
        &a64,
        &w64,
    )?;
    poll()?;
    if result.len() != output_len {
        return Err(NyError::InvalidSpec(
            "Conv2d bounded f64 engine returned malformed or non-finite output".into(),
        ));
    }
    for chunk in result.chunks(PREP_POLL_ELEMENTS) {
        if chunk.iter().any(|value| !value.is_finite()) {
            return Err(NyError::InvalidSpec(
                "Conv2d bounded f64 engine returned malformed or non-finite output".into(),
            ));
        }
        poll()?;
    }
    poll()?;
    Ok(result)
}

/// Finite-deadline f64 `A·W` for one convolution group.
///
/// Large products may use the process-global sound f64 engine, but only through
/// [`GemmEngine::gemm_f64_with_deadline`]. That contract retains the complete
/// contraction in every output tile, so the caller's order-independent
/// `gamma_n * S` certificate remains valid. Missing, unsupported, malformed, or
/// non-finite engine results fall back to the scalar pollable CPU contraction;
/// a deadline error or any post-deadline completion is terminal. Small products
/// stay on the CPU side of the measured accelerator crossover.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64_with_deadline(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
    deadline: Instant,
) -> Result<Vec<f64>> {
    if Instant::now() >= deadline {
        return Err(per_node_deadline_exceeded());
    }
    // The private comprehensive-intermediate worker pool is explicitly
    // CPU-only. Bypass process-global sound-f64 admission before even querying
    // the installed CUDA/WGPU slot and retain this routine's existing
    // deadline-aware faer implementation. Ordinary callers remain unchanged.
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return conv_group_col_flat_f64_pollable(
            a_coefficients,
            kernel,
            oc_start,
            out_c_per_group,
            kernel_cols_per_group,
            total_spatial,
            spatial_per_obj,
            kernel_spatial,
            kw,
            deadline,
        );
    }
    let macs = total_spatial
        .saturating_mul(out_c_per_group)
        .saturating_mul(kernel_cols_per_group);
    if macs >= CONV_SOUND_F64_GEMM_MIN_MACS {
        let attempt = crate::sound_f64_gemm::with_engine_deadline(deadline, |engine| {
            conv_group_col_flat_f64_deadline_via_engine_or_cpu(
                engine,
                a_coefficients,
                kernel,
                oc_start,
                out_c_per_group,
                kernel_cols_per_group,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
                deadline,
            )
        })?;
        if let Some(result) = attempt {
            return result;
        }
    }
    conv_group_col_flat_f64_pollable(
        a_coefficients,
        kernel,
        oc_start,
        out_c_per_group,
        kernel_cols_per_group,
        total_spatial,
        spatial_per_obj,
        kernel_spatial,
        kw,
        deadline,
    )
}

/// Try one explicit bounded-engine contraction, retaining the pollable CPU
/// contraction as the only fallback.
///
/// Kept separate from process-global admission so tests can inject engines that
/// panic on ordinary GEMM and prove finite authority never enters those methods.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64_deadline_via_engine_or_cpu(
    engine: &dyn GemmEngine,
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
    deadline: Instant,
) -> Result<Vec<f64>> {
    if let Some(result) = conv_group_col_flat_f64_try_deadline_engine(
        engine,
        a_coefficients,
        kernel,
        oc_start,
        out_c_per_group,
        kernel_cols_per_group,
        total_spatial,
        spatial_per_obj,
        kernel_spatial,
        kw,
        deadline,
    )? {
        return Ok(result);
    }
    conv_group_col_flat_f64_pollable(
        a_coefficients,
        kernel,
        oc_start,
        out_c_per_group,
        kernel_cols_per_group,
        total_spatial,
        spatial_per_obj,
        kernel_spatial,
        kw,
        deadline,
    )
}

/// Build exact f32→f64 row-major operands and run the complete contraction
/// through an engine's explicit bounded method.
///
/// `Ok(None)` is a safe CPU-fallback signal. No partial/malformed/non-finite
/// engine image is observable. Deadline expiry is always terminal.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64_try_deadline_engine(
    engine: &dyn GemmEngine,
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
    deadline: Instant,
) -> Result<Option<Vec<f64>>> {
    const PREP_POLL_ELEMENTS: usize = 1 << 12;

    let check_deadline = || {
        if Instant::now() >= deadline {
            Err(per_node_deadline_exceeded())
        } else {
            Ok(())
        }
    };
    check_deadline()?;

    if spatial_per_obj == 0 || kernel_spatial == 0 || kw == 0 {
        return Ok(None);
    }
    let a_len = total_spatial
        .checked_mul(out_c_per_group)
        .ok_or_else(|| NyError::InvalidSpec("conv2d bounded f64 A length overflow".into()))?;
    let w_len = out_c_per_group
        .checked_mul(kernel_cols_per_group)
        .ok_or_else(|| NyError::InvalidSpec("conv2d bounded f64 W length overflow".into()))?;
    let output_len = total_spatial
        .checked_mul(kernel_cols_per_group)
        .ok_or_else(|| NyError::InvalidSpec("conv2d bounded f64 output length overflow".into()))?;
    let bounded_host_engine = engine.forbids_unbounded_cpu_fallback();
    if bounded_host_engine {
        guard_deadline_bounded_conv_f64_buffer(a_len)?;
        guard_deadline_bounded_conv_f64_buffer(w_len)?;
        guard_deadline_bounded_conv_f64_buffer(output_len)?;
    }

    // A: (total_spatial × out_c_per_group), exactly widened from f32.
    let mut a64 = if bounded_host_engine {
        let mut values = Vec::new();
        values
            .try_reserve_exact(a_len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: a_len.saturating_mul(size_of::<f64>()),
                budget_bytes: DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES,
                site: DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE,
            })?;
        values
    } else {
        Vec::with_capacity(a_len)
    };
    for row in 0..total_spatial {
        let obj = row / spatial_per_obj;
        let spatial = row % spatial_per_obj;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            let value = a_coefficients[[obj, oc * spatial_per_obj + spatial]];
            if !value.is_finite() {
                check_deadline()?;
                return Ok(None);
            }
            a64.push(f64::from(value));
            if a64.len().is_multiple_of(PREP_POLL_ELEMENTS) {
                check_deadline()?;
            }
        }
    }

    // W: (out_c_per_group × kernel_cols_per_group), exactly widened from f32.
    let mut w64 = if bounded_host_engine {
        let mut values = Vec::new();
        values
            .try_reserve_exact(w_len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: w_len.saturating_mul(size_of::<f64>()),
                budget_bytes: DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES,
                site: DEADLINE_BOUNDED_CONV_HOST_BUFFER_SITE,
            })?;
        values
    } else {
        Vec::with_capacity(w_len)
    };
    for oc_local in 0..out_c_per_group {
        let oc = oc_start + oc_local;
        for col in 0..kernel_cols_per_group {
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            let value = kernel[[oc, ic, ki, kj]];
            if !value.is_finite() {
                check_deadline()?;
                return Ok(None);
            }
            w64.push(f64::from(value));
            if w64.len().is_multiple_of(PREP_POLL_ELEMENTS) {
                check_deadline()?;
            }
        }
    }
    check_deadline()?;

    let minimum_dispatches = total_spatial
        .saturating_mul(out_c_per_group)
        .saturating_mul(kernel_cols_per_group)
        .div_ceil(CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS);
    debug!(
        target: "ny::propagate::conv2d",
        m = total_spatial,
        k = out_c_per_group,
        n = kernel_cols_per_group,
        max_dispatch_macs = CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
        minimum_dispatches,
        "admitting deadline-bounded sound-f64 Conv2d contraction"
    );
    match engine.gemm_f64_with_deadline(
        total_spatial,
        out_c_per_group,
        kernel_cols_per_group,
        &a64,
        &w64,
        deadline,
        CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
    ) {
        Ok(result) if result.len() == output_len => {
            for chunk in result.chunks(PREP_POLL_ELEMENTS) {
                if chunk.iter().any(|value| !value.is_finite()) {
                    check_deadline()?;
                    return if bounded_host_engine {
                        Err(NyError::InvalidSpec(
                            "bounded Conv2d f64 engine returned non-finite output".into(),
                        ))
                    } else {
                        Ok(None)
                    };
                }
                check_deadline()?;
            }
            check_deadline()?;
            Ok(Some(result))
        }
        Ok(_) => {
            check_deadline()?;
            if bounded_host_engine {
                Err(NyError::InvalidSpec(
                    "bounded Conv2d f64 engine returned a malformed output length".into(),
                ))
            } else {
                Ok(None)
            }
        }
        Err(error) if error.is_deadline_exceeded() => Err(error),
        Err(error) if bounded_host_engine => Err(error),
        Err(_) => {
            check_deadline()?;
            Ok(None)
        }
    }
}

/// Scalar CPU fallback for finite-deadline f64 `A·W`.
///
/// Products of two widened f32 values are exact in f64; the caller's
/// order-independent `gamma_n * S` certificate covers the running-sum order.
#[allow(clippy::too_many_arguments)]
fn conv_group_col_flat_f64_pollable(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    oc_start: usize,
    out_c_per_group: usize,
    kernel_cols_per_group: usize,
    total_spatial: usize,
    spatial_per_obj: usize,
    kernel_spatial: usize,
    kw: usize,
    deadline: Instant,
) -> Result<Vec<f64>> {
    if Instant::now() >= deadline {
        return Err(per_node_deadline_exceeded());
    }
    let len = total_spatial
        .checked_mul(kernel_cols_per_group)
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d f64 deadline CPU result length overflow".to_string())
        })?;
    let mut col_flat = vec![0.0f64; len];
    if Instant::now() >= deadline {
        return Err(per_node_deadline_exceeded());
    }

    // ROW-CHUNKED faer f64 GEMM (#cifar100-deadline-gemm).
    //
    // This path previously ran a single-threaded scalar triple loop with
    // dynamic-rank `ArrayD` indexing and a poll test per MAC, while its
    // no-deadline twin `conv_group_col_flat_f64` (below) already used faer f64
    // GEMM — whose own comment records that it replaced this very loop because
    // it "dominated Conv2d CROWN backward wall time on conv stacks with no CUDA
    // engine". Since a competition run ALWAYS carries a deadline, ny always took
    // the slow branch: measured on CIFAR100_resnet_medium, ONE root CROWN
    // backward over the 99 margin objectives cost ~80-89s against a 100s total
    // budget, so alpha-CROWN got 0.0s and BaB explored 0 domains.
    //
    // faer's `mat_mul` is one blocking call and cannot be polled internally, so
    // the rows are processed in chunks and the deadline is checked BETWEEN
    // chunks. That keeps this path deadline-responsive (the reason the scalar
    // loop existed) while getting blocked/parallel GEMM throughput.
    //
    // SOUND: identical operands to both twins, and the caller's certified error
    // `cast + γ_n^f64·S + prop` is summation-ORDER INDEPENDENT (Higham γ_n·S),
    // so faer's blocked accumulation is covered by exactly the same certificate
    // as the scalar running sum — the identical argument the no-deadline twin
    // and the cuBLAS route already rely on.
    let w64 = Mat::<f64>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
        let oc = oc_start + oc_local;
        let ic = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        f64::from(kernel[[oc, ic, ki, kj]])
    });

    // ONE GEMM over the full row extent — deliberately NOT chunked.
    //
    // Chunking the rows would let the deadline be polled mid-multiply, but faer
    // selects its blocking from the operand shape, so a chunked call reduces the
    // k-dimension in a different order and the result differs in the low bits
    // from the no-deadline twin. `deadline_conv_randomized_geometry_and_group_
    // mapping_matches_cpu` pins BIT-IDENTITY between the deadline path and
    // `conv2d_transpose_backward_coeff_f64`, and that parity is worth keeping:
    // it is what lets the deadline and no-deadline routes be treated as the same
    // computation everywhere else. Matching the twin's single call preserves it
    // exactly.
    //
    // Losing mid-multiply polling is acceptable BECAUSE of the speedup: the
    // per-MAC poll existed only to keep a slow scalar loop interruptible. The
    // checks before and after operand construction refuse to START after the
    // absolute deadline; they make no prediction about GEMM duration. Any
    // over-run is therefore bounded by one GEMM rather than by the whole loop.
    let macs = total_spatial
        .saturating_mul(out_c_per_group)
        .saturating_mul(kernel_cols_per_group);
    if macs > DEADLINE_F64_CPU_POLL_OPS {
        // Only for multiplies large enough to matter, repeat the authority
        // checkpoint before constructing A. This is not a duration estimate;
        // the exact launch-boundary check below runs after construction too.
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(per_node_deadline_exceeded());
        }
    }

    let a64 = Mat::<f64>::from_fn(total_spatial, out_c_per_group, |row, oc_local| {
        let obj = row / spatial_per_obj;
        let spatial = row % spatial_per_obj;
        let oc = oc_start + oc_local;
        f64::from(a_coefficients[[obj, oc * spatial_per_obj + spatial]])
    });
    // Operand materialization itself can consume the remaining node budget.
    // Poll at the exact launch boundary so expired authority never enters the
    // unpollable faer contraction.
    let prod = run_f64_gemm_before_deadline_with_clock(deadline, Instant::now, || {
        crate::faer_parallelism::mat_mul_f64(&a64, &w64)
    })?;
    for row in 0..total_spatial {
        let row_base = row * kernel_cols_per_group;
        for col in 0..kernel_cols_per_group {
            col_flat[row_base + col] = prod[(row, col)];
        }
    }

    if Instant::now() >= deadline {
        return Err(per_node_deadline_exceeded());
    }
    Ok(col_flat)
}

/// Largest single sound-conv f64 GEMM MAC count below which the CPU path wins
/// (GPU launch/transfer bound) — the same crossover the Linear seam uses.
///
/// Shared with the ConvTranspose forward-conv recompute in `ops_gemm`.
pub(super) const CONV_SOUND_F64_GEMM_MIN_MACS: usize = 1 << 24;

/// Maximum MAC count in one non-interruptible bounded-engine dispatch. The
/// engine may impose a smaller cap, but must retain the complete contraction.
const CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS: usize = 1 << 24;

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
        adaptive_default_crown_mem_cap_mb, available_system_ram_observation, backward_buffer_bytes,
        backward_buffer_bytes_for, conv2d_transpose_backward_coeff_f64,
        conv2d_transpose_backward_coeff_f64_with_deadline,
        conv2d_transpose_backward_coeff_f64_with_engine_and_deadline,
        conv2d_transpose_batched_gemm_grouped_with_deadline,
        conv2d_transpose_pair_batched_gemm_grouped_with_deadline, conv_crown_mem_cap_bytes,
        conv_group_col_flat_f64_deadline_via_engine_or_cpu, conv_group_col_flat_f64_pollable,
        conv_group_col_flat_f64_try_deadline_engine, conv_group_col_flat_f64_with_deadline,
        default_crown_mem_cap_mb, envelope_crown_mem_cap_bytes,
        f64_recompute_objective_chunk_with_cap, faer_mat_to_row_major,
        guard_conv_crown_backward_buffer_with_cap, guard_conv_crown_scratch_bytes_with_cap,
        largest_conv_crown_scratch_bytes, narrowest_conv_crown_mem_cap,
        parse_meminfo_available_bytes, parse_meminfo_total_bytes,
        process_envelope_crown_mem_cap_bytes, run_f64_gemm_before_deadline_with_clock,
        ConvCrownMemCap, ConvCrownMemCapSource, CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
        DEADLINE_GEMM_ROW_CHUNK, DEFAULT_CROWN_MEM_CAP_MB, MAX_ADAPTIVE_CROWN_MEM_CAP_MB,
    };
    use crate::tests::with_crown_mem_cap_mb;
    use faer::Mat;
    use ndarray::{Array2, ArrayD, IxDyn};
    use ny_core::{ConvTranspose2dParams, GemmEngine, NyError, Result};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    const MB: usize = 1024 * 1024;

    fn injected_cap(bytes: usize) -> ConvCrownMemCap {
        ConvCrownMemCap {
            bytes,
            source: ConvCrownMemCapSource::Policy,
        }
    }

    /// Deterministic property-test oracle without adding a `rand` dev-dependency.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn bounded(&mut self, upper_exclusive: usize) -> usize {
            self.next_u64() as usize % upper_exclusive
        }

        fn finite_f32(&mut self) -> f32 {
            (self.bounded(33) as f32 - 16.0) / 7.0
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct BoundedConvCall {
        m: usize,
        k: usize,
        n: usize,
        max_dispatch_macs: usize,
        a: Vec<f64>,
        w: Vec<f64>,
    }

    fn exact_engine_product(m: usize, k: usize, n: usize, a: &[f64], w: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for kk in 0..k {
                for col in 0..n {
                    output[row * n + col] += a[row * k + kk] * w[kk * n + col];
                }
            }
        }
        output
    }

    #[derive(Default)]
    struct RecordingDeadlineConvEngine {
        calls: Mutex<Vec<BoundedConvCall>>,
    }

    #[derive(Default)]
    struct PollableBoundedHostConvEngine {
        ordinary_calls: AtomicUsize,
        polls: AtomicUsize,
    }

    struct IncompleteBoundedHostConvEngine;

    impl GemmEngine for IncompleteBoundedHostConvEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("incomplete bounded engine must be rejected before GEMM")
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }
    }

    impl GemmEngine for PollableBoundedHostConvEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("bounded f64 test entered f32 GEMM")
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], w: &[f64]) -> Result<Vec<f64>> {
            self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
            Ok(exact_engine_product(m, k, n, a, w))
        }

        fn gemm_f64_with_deadline(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            panic!("bounded host f64 seam entered accelerator deadline GEMM")
        }

        fn poll_crown_backward_deadline(&self) -> Result<()> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }

        fn provides_deadline_pollable_host_gemm(&self) -> bool {
            true
        }
    }

    #[test]
    fn bounded_f64_recompute_uses_pollable_host_engine_without_outer_deadline() {
        let coefficients =
            Array2::from_shape_vec((2, 4), vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -0.25])
                .expect("coefficient shape");
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.75]).expect("kernel shape");
        let expected = conv2d_transpose_backward_coeff_f64(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (2, 2),
            (2, 2),
            1,
            1,
            2,
        )
        .expect("legacy reference");
        let engine = PollableBoundedHostConvEngine::default();
        let actual = conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (2, 2),
            (2, 2),
            1,
            1,
            2,
            Some(&engine),
            None,
        )
        .expect("bounded host recompute");

        assert_eq!(actual, expected);
        assert_eq!(engine.ordinary_calls.load(Ordering::SeqCst), 1);
        assert!(engine.polls.load(Ordering::SeqCst) >= 4);
    }

    #[test]
    fn bounded_f64_recompute_rejects_missing_pollable_host_capability() {
        let coefficients = Array2::ones((1, 1));
        let kernel = ArrayD::ones(IxDyn(&[1, 1, 1, 1]));
        let error = conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            1,
            1,
            Some(&IncompleteBoundedHostConvEngine),
            None,
        )
        .expect_err("partial bounded capability must fail closed");
        assert!(matches!(error, NyError::UnsupportedOp(_)));
    }

    impl GemmEngine for RecordingDeadlineConvEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("finite-deadline Conv2d CROWN entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("finite-deadline Conv2d CROWN entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f64],
            w: &[f64],
            _deadline: Instant,
            max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls
                .lock()
                .expect("recording bounded-conv call lock")
                .push(BoundedConvCall {
                    m,
                    k,
                    n,
                    max_dispatch_macs,
                    a: a.to_vec(),
                    w: w.to_vec(),
                });
            Ok(exact_engine_product(m, k, n, a, w))
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ScriptedConvOutcome {
        Unsupported,
        OrdinaryFailure,
        Malformed,
        Nonfinite,
        MidTilingDeadline,
    }

    struct ScriptedDeadlineConvEngine {
        outcome: ScriptedConvOutcome,
        calls: AtomicUsize,
        completed_tiles: AtomicUsize,
    }

    impl ScriptedDeadlineConvEngine {
        fn new(outcome: ScriptedConvOutcome) -> Self {
            Self {
                outcome,
                calls: AtomicUsize::new(0),
                completed_tiles: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for ScriptedDeadlineConvEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("deadline fallback entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("deadline fallback entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _w: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.outcome {
                ScriptedConvOutcome::Unsupported => Err(NyError::UnsupportedOp(
                    "injected bounded-conv unsupported".into(),
                )),
                ScriptedConvOutcome::OrdinaryFailure => Err(NyError::NumericalInstability(
                    "injected bounded-conv numerical failure".into(),
                )),
                ScriptedConvOutcome::Malformed => {
                    Ok(vec![0.0; m.saturating_mul(n).saturating_sub(1)])
                }
                ScriptedConvOutcome::Nonfinite => Ok(vec![f64::NAN; m * n]),
                ScriptedConvOutcome::MidTilingDeadline => {
                    self.completed_tiles.store(2, Ordering::SeqCst);
                    Err(NyError::DeadlineExceeded(
                        "injected bounded-conv deadline after output tile 2".into(),
                    ))
                }
            }
        }
    }

    struct SleepPastDeadlineConvEngine {
        calls: AtomicUsize,
    }

    impl GemmEngine for SleepPastDeadlineConvEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("post-deadline Conv2d test entered ordinary f32 GEMM")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            panic!("post-deadline Conv2d test entered ordinary f64 GEMM")
        }

        fn gemm_f64_with_deadline(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _b: &[f64],
            _deadline: Instant,
            _max_dispatch_macs: usize,
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(30));
            Ok(vec![0.0; m * n])
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum F32CrownEngineOutcome {
        FusedDeadline,
        GemmDeadline,
        FusedMemoryRefusal,
        GemmMemoryRefusal,
        OrdinaryFailure,
    }

    struct ScriptedF32CrownEngine {
        outcome: F32CrownEngineOutcome,
        deadline_expired: bool,
        fused_calls: AtomicUsize,
        gemm_calls: AtomicUsize,
    }

    impl ScriptedF32CrownEngine {
        fn expired(outcome: F32CrownEngineOutcome) -> Self {
            Self {
                outcome,
                deadline_expired: true,
                fused_calls: AtomicUsize::new(0),
                gemm_calls: AtomicUsize::new(0),
            }
        }

        fn unscoped(outcome: F32CrownEngineOutcome) -> Self {
            Self {
                outcome,
                deadline_expired: false,
                fused_calls: AtomicUsize::new(0),
                gemm_calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for ScriptedF32CrownEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            self.gemm_calls.fetch_add(1, Ordering::SeqCst);
            match self.outcome {
                F32CrownEngineOutcome::GemmDeadline => Err(NyError::DeadlineExceeded(
                    "injected f32 Conv2d CROWN GEMM deadline".into(),
                )),
                F32CrownEngineOutcome::GemmMemoryRefusal => Err(NyError::CpuMemoryExceeded {
                    required_bytes: 2,
                    budget_bytes: 1,
                    site: "test bounded Conv2d CROWN GEMM",
                }),
                F32CrownEngineOutcome::FusedDeadline
                | F32CrownEngineOutcome::FusedMemoryRefusal
                | F32CrownEngineOutcome::OrdinaryFailure => Err(NyError::NumericalInstability(
                    "injected ordinary f32 Conv2d CROWN GEMM failure".into(),
                )),
            }
        }

        fn conv_transpose_2d(
            &self,
            _a_reshaped: &[f32],
            _weight_col: &[f32],
            _params: &ConvTranspose2dParams,
        ) -> Result<Vec<f32>> {
            self.fused_calls.fetch_add(1, Ordering::SeqCst);
            match self.outcome {
                F32CrownEngineOutcome::FusedDeadline => Err(NyError::DeadlineExceeded(
                    "injected fused f32 Conv2d CROWN deadline".into(),
                )),
                F32CrownEngineOutcome::FusedMemoryRefusal => Err(NyError::CpuMemoryExceeded {
                    required_bytes: 2,
                    budget_bytes: 1,
                    site: "test bounded fused Conv2d CROWN",
                }),
                F32CrownEngineOutcome::GemmDeadline
                | F32CrownEngineOutcome::GemmMemoryRefusal
                | F32CrownEngineOutcome::OrdinaryFailure => Err(NyError::UnsupportedOp(
                    "injected unsupported fused f32 Conv2d CROWN".into(),
                )),
            }
        }

        fn poll_crown_backward_deadline(&self) -> Result<()> {
            if self.deadline_expired {
                Err(NyError::DeadlineExceeded(
                    "injected expired f32 Conv2d CROWN proxy".into(),
                ))
            } else {
                Ok(())
            }
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            matches!(
                self.outcome,
                F32CrownEngineOutcome::FusedMemoryRefusal
                    | F32CrownEngineOutcome::GemmMemoryRefusal
            )
        }
    }

    fn f32_crown_fixture(num_objectives: usize) -> (Array2<f32>, ArrayD<f32>) {
        (
            Array2::from_elem((num_objectives, 1), 2.0),
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![3.0])
                .expect("valid f32 Conv2d CROWN kernel"),
        )
    }

    fn run_f32_crown_fixture(
        coefficients: &Array2<f32>,
        kernel: &ArrayD<f32>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Array2<f32>> {
        conv2d_transpose_batched_gemm_grouped_with_deadline(
            coefficients,
            kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            1,
            1,
            engine,
            deadline,
        )
    }

    #[derive(Clone)]
    struct DeadlineConvFixture {
        coefficients: Array2<f32>,
        kernel: ArrayD<f32>,
        oc_start: usize,
        out_c_per_group: usize,
        kernel_cols_per_group: usize,
        total_spatial: usize,
        spatial_per_obj: usize,
        kernel_spatial: usize,
        kw: usize,
    }

    fn deadline_conv_fixture() -> DeadlineConvFixture {
        let (num_obj, out_c, in_c_per_group, kh, kw) = (2usize, 4usize, 2usize, 2usize, 2usize);
        let spatial_per_obj = 4usize;
        DeadlineConvFixture {
            coefficients: Array2::from_shape_fn(
                (num_obj, out_c * spatial_per_obj),
                |(row, col)| ((row * 19 + col * 7) % 23) as f32 / 6.0 - 1.5,
            ),
            kernel: ArrayD::from_shape_fn(IxDyn(&[out_c, in_c_per_group, kh, kw]), |index| {
                ((index[0] * 17 + index[1] * 5 + index[2] * 3 + index[3]) % 19) as f32 / 8.0 - 1.0
            }),
            // Exercise the second of two groups, not the easy zero-offset case.
            oc_start: 2,
            out_c_per_group: 2,
            kernel_cols_per_group: in_c_per_group * kh * kw,
            total_spatial: num_obj * spatial_per_obj,
            spatial_per_obj,
            kernel_spatial: kh * kw,
            kw,
        }
    }

    fn bounded_fixture(
        engine: &dyn GemmEngine,
        fixture: &DeadlineConvFixture,
        deadline: Instant,
    ) -> Result<Vec<f64>> {
        conv_group_col_flat_f64_deadline_via_engine_or_cpu(
            engine,
            &fixture.coefficients,
            &fixture.kernel,
            fixture.oc_start,
            fixture.out_c_per_group,
            fixture.kernel_cols_per_group,
            fixture.total_spatial,
            fixture.spatial_per_obj,
            fixture.kernel_spatial,
            fixture.kw,
            deadline,
        )
    }

    fn cpu_fixture(fixture: &DeadlineConvFixture, deadline: Instant) -> Result<Vec<f64>> {
        conv_group_col_flat_f64_pollable(
            &fixture.coefficients,
            &fixture.kernel,
            fixture.oc_start,
            fixture.out_c_per_group,
            fixture.kernel_cols_per_group,
            fixture.total_spatial,
            fixture.spatial_per_obj,
            fixture.kernel_spatial,
            fixture.kw,
            deadline,
        )
    }

    #[test]
    fn cpu_only_scope_bypasses_conv_process_global_f64_accessor() {
        let _counter_lock = crate::sound_f64_gemm::accessor_counter_test_lock();
        crate::sound_f64_gemm::reset_cpu_only_global_accessor_attempts();

        // Exactly the global Conv-f64 admission floor, while keeping the
        // operands and result small enough for a focused unit test:
        // 512 x 256 x 128 = 2^24 MACs.
        let total_spatial = 512usize;
        let out_channels = 256usize;
        let kernel_cols = 128usize;
        let coefficients = Array2::from_shape_fn((total_spatial, out_channels), |(row, col)| {
            ((row * 5 + col * 3) % 17) as f32 / 16.0
        });
        let kernel = ArrayD::from_shape_fn(IxDyn(&[out_channels, kernel_cols, 1, 1]), |index| {
            ((index[0] * 7 + index[1] * 11) % 19) as f32 / 18.0
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        let before_all = crate::sound_f64_gemm::global_accessor_attempts();
        let guarded = {
            let _cpu_only = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
            conv_group_col_flat_f64_with_deadline(
                &coefficients,
                &kernel,
                0,
                out_channels,
                kernel_cols,
                total_spatial,
                1,
                1,
                1,
                deadline,
            )
            .expect("guarded pollable faer Conv contraction")
        };
        assert_eq!(
            crate::sound_f64_gemm::cpu_only_global_accessor_attempts(),
            0
        );

        // The identical ordinary call remains sensitive to the historical
        // process-global admission seam.
        let ordinary = conv_group_col_flat_f64_with_deadline(
            &coefficients,
            &kernel,
            0,
            out_channels,
            kernel_cols,
            total_spatial,
            1,
            1,
            1,
            deadline,
        )
        .expect("ordinary Conv contraction");
        assert_eq!(guarded.len(), ordinary.len());
        assert!(ordinary.iter().all(|value| value.is_finite()));
        assert!(crate::sound_f64_gemm::global_accessor_attempts() > before_all);
    }

    /// Expiry after operand construction is a launch-boundary refusal: the
    /// unpollable faer f64 GEMM must not start. An injected clock pins the
    /// ordering without relying on a large allocation outrunning a tiny real
    /// deadline.
    #[test]
    fn convtranspose_f64_post_operand_expiry_refuses_gemm_launch() {
        let deadline = Instant::now();
        let gemm_entered = Cell::new(false);
        let result = run_f64_gemm_before_deadline_with_clock(
            deadline,
            || deadline,
            || {
                gemm_entered.set(true);
                Mat::<f64>::zeros(1, 1)
            },
        );

        assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
        assert!(!gemm_entered.get(), "expired authority entered f64 GEMM");
    }

    fn assert_same_f64_image(got: &[f64], want: &[f64], context: &str) {
        assert_eq!(got.len(), want.len(), "{context}: result length");
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            if want.is_nan() {
                assert!(
                    got.is_nan(),
                    "{context}: expected NaN at {index}, got {got}"
                );
            } else if got == 0.0 && want == 0.0 {
                // SIGNED ZERO ONLY (#cifar100-deadline-gemm). `-0.0 == 0.0` in
                // IEEE-754, and a certified bound cannot distinguish them: every
                // downstream use is an add/compare against the enclosure, never a
                // reciprocal or copysign. faer's blocked accumulation can emit
                // `+0.0` where a scalar running sum emits `-0.0` (e.g. adding
                // `-0.0 + -0.0` vs a zero-initialised accumulator).
                //
                // This is deliberately the ONLY relaxation: every non-zero value
                // still requires exact `to_bits()` equality, so the parity gate
                // between the deadline and no-deadline routes keeps all of its
                // teeth for anything that could change a bound.
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{context}: bit mismatch at {index}: got={got:e}, want={want:e}"
                );
            }
        }
    }

    #[test]
    fn f32_conv_crown_pair_preserves_expired_proxy_deadline() {
        let (coefficients, kernel) = f32_crown_fixture(1);
        let engine = ScriptedF32CrownEngine::expired(F32CrownEngineOutcome::FusedDeadline);

        let error = conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
            &coefficients,
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            1,
            Some(&engine),
            None,
        )
        .expect_err("typed fused-pair engine deadline must be terminal");

        assert!(error.is_deadline_exceeded(), "wrong error: {error}");
        assert_eq!(engine.fused_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            engine.gemm_calls.load(Ordering::SeqCst),
            0,
            "pair deadline must not enter a secondary GEMM fallback"
        );
    }

    #[test]
    fn f32_conv_crown_single_preserves_expired_proxy_deadlines_in_both_branches() {
        for chunked in [false, true] {
            let num_objectives = if chunked {
                DEADLINE_GEMM_ROW_CHUNK + 1
            } else {
                1
            };
            let deadline = chunked.then(|| Instant::now() + Duration::from_secs(30));
            let (coefficients, kernel) = f32_crown_fixture(num_objectives);

            for outcome in [
                F32CrownEngineOutcome::FusedDeadline,
                F32CrownEngineOutcome::GemmDeadline,
            ] {
                let engine = ScriptedF32CrownEngine::expired(outcome);
                let error = run_f32_crown_fixture(&coefficients, &kernel, Some(&engine), deadline)
                    .expect_err("typed Conv2d CROWN engine deadline must be terminal");
                assert!(
                    error.is_deadline_exceeded(),
                    "chunked={chunked}, outcome={outcome:?}: wrong error: {error}"
                );
                assert_eq!(
                    engine.fused_calls.load(Ordering::SeqCst),
                    1,
                    "chunked={chunked}, outcome={outcome:?}: fused call count"
                );
                let expected_gemm_calls =
                    usize::from(matches!(outcome, F32CrownEngineOutcome::GemmDeadline));
                assert_eq!(
                    engine.gemm_calls.load(Ordering::SeqCst),
                    expected_gemm_calls,
                    "chunked={chunked}, outcome={outcome:?}: GEMM call count"
                );
            }
        }
    }

    #[test]
    fn f32_conv_crown_preserves_bounded_engine_memory_refusal() {
        let (coefficients, kernel) = f32_crown_fixture(1);
        for outcome in [
            F32CrownEngineOutcome::FusedMemoryRefusal,
            F32CrownEngineOutcome::GemmMemoryRefusal,
        ] {
            let engine = ScriptedF32CrownEngine::unscoped(outcome);
            let error = run_f32_crown_fixture(&coefficients, &kernel, Some(&engine), None)
                .expect_err("bounded memory refusal must not enter local CPU fallback");
            assert!(
                error.is_cpu_memory_exceeded(),
                "outcome={outcome:?}: wrong error: {error}"
            );
            assert_eq!(engine.fused_calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                engine.gemm_calls.load(Ordering::SeqCst),
                usize::from(matches!(outcome, F32CrownEngineOutcome::GemmMemoryRefusal))
            );
        }
    }

    #[test]
    fn f32_conv_crown_none_authority_keeps_typed_and_ordinary_error_cpu_fallback() {
        let (coefficients, kernel) = f32_crown_fixture(2);
        let expected = run_f32_crown_fixture(&coefficients, &kernel, None, None)
            .expect("legacy CPU Conv2d CROWN path");
        let pair_expected = conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
            &coefficients,
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            1,
            None,
            None,
        )
        .expect("legacy CPU Conv2d CROWN pair path");

        for outcome in [
            F32CrownEngineOutcome::FusedDeadline,
            F32CrownEngineOutcome::GemmDeadline,
            F32CrownEngineOutcome::OrdinaryFailure,
        ] {
            let engine = ScriptedF32CrownEngine::unscoped(outcome);
            let actual = run_f32_crown_fixture(&coefficients, &kernel, Some(&engine), None)
                .expect("unscoped engine errors must retain the CPU fallback");
            assert_eq!(actual, expected, "single fallback mismatch: {outcome:?}");
            assert_eq!(engine.fused_calls.load(Ordering::SeqCst), 1);
            assert_eq!(engine.gemm_calls.load(Ordering::SeqCst), 1);

            let pair_engine = ScriptedF32CrownEngine::unscoped(outcome);
            let pair_actual = conv2d_transpose_pair_batched_gemm_grouped_with_deadline(
                &coefficients,
                &coefficients,
                &kernel,
                (1, 1),
                (0, 0),
                (1, 1),
                (1, 1),
                (1, 1),
                1,
                1,
                Some(&pair_engine),
                None,
            )
            .expect("unscoped fused-pair and GEMM errors must retain CPU fallback");
            assert_eq!(
                pair_actual, pair_expected,
                "pair fallback mismatch: {outcome:?}"
            );
            assert_eq!(pair_engine.fused_calls.load(Ordering::SeqCst), 3);
            assert_eq!(pair_engine.gemm_calls.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn certified_f64_finite_deadline_is_pollable_and_matches_legacy() {
        let coefficients = Array2::from_shape_fn((2, 2 * 2 * 2), |(row, col)| {
            (((row * 17 + col * 7) % 13) as f32 - 6.0) / 5.0
        });
        let kernel = ArrayD::from_shape_fn(IxDyn(&[2, 1, 2, 2]), |index| {
            (((index[0] * 11 + index[2] * 5 + index[3] * 3) % 9) as f32 - 4.0) / 7.0
        });
        let legacy = conv2d_transpose_backward_coeff_f64(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (3, 3),
            (2, 2),
            2,
            1,
            1,
        )
        .expect("legacy certified f64 recompute");
        let finite = conv2d_transpose_backward_coeff_f64_with_deadline(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (3, 3),
            (2, 2),
            2,
            1,
            1,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect("pollable certified f64 recompute");
        assert_eq!(finite.raw_dim(), legacy.raw_dim());
        for (index, (&got, &want)) in finite.iter().zip(legacy.iter()).enumerate() {
            let tolerance = 2.0e-13 * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tolerance,
                "finite/legacy transpose-f64 parity mismatch at {index}: got={got:e} want={want:e}"
            );
        }

        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("Instant epoch is at least one millisecond old");
        let error = conv2d_transpose_backward_coeff_f64_with_deadline(
            &coefficients,
            &kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (3, 3),
            (2, 2),
            2,
            1,
            1,
            Some(expired),
        )
        .expect_err("expired certified f64 recompute must refuse before work");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
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
    fn deadline_conv_engine_uses_one_full_k_bounded_product_for_nonzero_group() {
        let fixture = deadline_conv_fixture();
        let engine = RecordingDeadlineConvEngine::default();
        let deadline = Instant::now() + Duration::from_secs(30);
        let actual =
            bounded_fixture(&engine, &fixture, deadline).expect("live bounded Conv2d contraction");
        let expected = cpu_fixture(&fixture, deadline).expect("pollable CPU Conv2d contraction");
        assert_same_f64_image(&actual, &expected, "bounded second-group contraction");

        let calls = engine.calls.lock().expect("recording bounded-conv lock");
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(
            (call.m, call.k, call.n, call.max_dispatch_macs),
            (
                fixture.total_spatial,
                fixture.out_c_per_group,
                fixture.kernel_cols_per_group,
                CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
            )
        );
        // The bounded operand image starts at group 1's output channels. This
        // catches accidentally accelerating group 0 for every outer group.
        assert_eq!(
            call.a[0],
            f64::from(fixture.coefficients[[0, fixture.oc_start * fixture.spatial_per_obj]])
        );
        assert_eq!(
            call.a[1],
            f64::from(fixture.coefficients[[0, (fixture.oc_start + 1) * fixture.spatial_per_obj]])
        );
        assert_eq!(
            call.w[0],
            f64::from(fixture.kernel[[fixture.oc_start, 0, 0, 0]])
        );
    }

    #[test]
    fn deadline_conv_randomized_geometry_and_group_mapping_matches_cpu() {
        let mut rng = SplitMix64(0x0C0D_EC01_25EE_DF64);
        let mut bounded_call_count = 0usize;

        for case in 0..32 {
            let groups = [1usize, 2, 4][rng.bounded(3)];
            let out_c_per_group = 1 + rng.bounded(3);
            let out_c = groups * out_c_per_group;
            let in_c_per_group = 1 + rng.bounded(2);
            let num_objectives = 1 + rng.bounded(3);
            let grad_h = 1 + rng.bounded(3);
            let grad_w = 1 + rng.bounded(3);
            let kh = 1 + rng.bounded(3);
            let kw = 1 + rng.bounded(3);
            let stride = (1 + rng.bounded(2), 1 + rng.bounded(2));
            let dilation = (1 + rng.bounded(2), 1 + rng.bounded(2));

            // Construct a valid ConvTranspose output geometry and independently
            // vary padding on each axis.
            let unpadded_h = (grad_h - 1) * stride.0 + (kh - 1) * dilation.0 + 1;
            let unpadded_w = (grad_w - 1) * stride.1 + (kw - 1) * dilation.1 + 1;
            let max_pad_h = ((unpadded_h - 1) / 2).min(2);
            let max_pad_w = ((unpadded_w - 1) / 2).min(2);
            let padding = (rng.bounded(max_pad_h + 1), rng.bounded(max_pad_w + 1));
            let output_size = (unpadded_h - 2 * padding.0, unpadded_w - 2 * padding.1);
            let spatial_per_obj = grad_h * grad_w;
            let total_spatial = num_objectives * spatial_per_obj;
            let kernel_spatial = kh * kw;
            let kernel_cols_per_group = in_c_per_group * kernel_spatial;

            let coefficients =
                Array2::from_shape_fn((num_objectives, out_c * spatial_per_obj), |_| {
                    rng.finite_f32()
                });
            let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c_per_group, kh, kw]), |_| {
                rng.finite_f32()
            });
            let deadline = Instant::now() + Duration::from_secs(30);

            // End-to-end parity covers the unchanged col2im mapping: stride,
            // padding, dilation, every group, and every input channel.
            let legacy = conv2d_transpose_backward_coeff_f64(
                &coefficients,
                &kernel,
                stride,
                padding,
                dilation,
                output_size,
                (grad_h, grad_w),
                out_c,
                groups,
                1,
            )
            .unwrap_or_else(|error| panic!("case {case}: legacy Conv2d path: {error}"));
            let finite = conv2d_transpose_backward_coeff_f64_with_deadline(
                &coefficients,
                &kernel,
                stride,
                padding,
                dilation,
                output_size,
                (grad_h, grad_w),
                out_c,
                groups,
                1,
                Some(deadline),
            )
            .unwrap_or_else(|error| panic!("case {case}: finite Conv2d path: {error}"));
            assert_eq!(finite.raw_dim(), legacy.raw_dim(), "case {case}");
            for (index, (&got, &want)) in finite.iter().zip(legacy.iter()).enumerate() {
                let tolerance = 4.0e-13 * (1.0 + want.abs());
                assert!(
                    (got - want).abs() <= tolerance,
                    "case {case}, output {index}: finite={got:e}, legacy={want:e}, \
                     groups={groups}, stride={stride:?}, padding={padding:?}, \
                     dilation={dilation:?}"
                );
            }

            // Independently inject the bounded engine into every group. The
            // ordinary GEMM methods panic, proving the finite seam can use only
            // the explicit deadline contract. Equality with the scalar oracle
            // validates A/W flattening and nonzero group offsets.
            for group in 0..groups {
                let engine = RecordingDeadlineConvEngine::default();
                let oc_start = group * out_c_per_group;
                let actual = conv_group_col_flat_f64_deadline_via_engine_or_cpu(
                    &engine,
                    &coefficients,
                    &kernel,
                    oc_start,
                    out_c_per_group,
                    kernel_cols_per_group,
                    total_spatial,
                    spatial_per_obj,
                    kernel_spatial,
                    kw,
                    deadline,
                )
                .unwrap_or_else(|error| {
                    panic!("case {case}, group {group}: bounded contraction: {error}")
                });
                let expected = conv_group_col_flat_f64_pollable(
                    &coefficients,
                    &kernel,
                    oc_start,
                    out_c_per_group,
                    kernel_cols_per_group,
                    total_spatial,
                    spatial_per_obj,
                    kernel_spatial,
                    kw,
                    deadline,
                )
                .unwrap_or_else(|error| {
                    panic!("case {case}, group {group}: CPU contraction: {error}")
                });
                assert_same_f64_image(&actual, &expected, &format!("case {case}, group {group}"));

                let calls = engine.calls.lock().expect("recording randomized call");
                assert_eq!(calls.len(), 1, "case {case}, group {group}");
                assert_eq!(
                    (
                        calls[0].m,
                        calls[0].k,
                        calls[0].n,
                        calls[0].max_dispatch_macs,
                    ),
                    (
                        total_spatial,
                        out_c_per_group,
                        kernel_cols_per_group,
                        CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS,
                    ),
                    "case {case}, group {group}"
                );
                bounded_call_count += 1;
            }
        }

        // Ensure the deterministic oracle actually covered many grouped seams,
        // rather than degenerating to only groups=1.
        assert!(bounded_call_count >= 64, "only {bounded_call_count} calls");
    }

    #[test]
    fn deadline_conv_engine_failures_fallback_but_deadline_is_terminal() {
        let fixture = deadline_conv_fixture();
        let deadline = Instant::now() + Duration::from_secs(30);
        let expected = cpu_fixture(&fixture, deadline).expect("CPU fallback oracle");

        for outcome in [
            ScriptedConvOutcome::Unsupported,
            ScriptedConvOutcome::OrdinaryFailure,
            ScriptedConvOutcome::Malformed,
            ScriptedConvOutcome::Nonfinite,
        ] {
            let engine = ScriptedDeadlineConvEngine::new(outcome);
            let actual = bounded_fixture(&engine, &fixture, deadline)
                .unwrap_or_else(|error| panic!("{outcome:?} should fall back: {error}"));
            assert_same_f64_image(&actual, &expected, &format!("{outcome:?} fallback"));
            assert_eq!(engine.calls.load(Ordering::SeqCst), 1, "{outcome:?}");
        }

        // Simulate an engine that completed some output tiles before observing
        // its deadline. DeadlineExceeded is terminal: the caller must not mask
        // it by recomputing the entire contraction on CPU.
        let mid_tiling = ScriptedDeadlineConvEngine::new(ScriptedConvOutcome::MidTilingDeadline);
        let error = bounded_fixture(&mid_tiling, &fixture, deadline)
            .expect_err("mid-tiling deadline must be terminal");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(mid_tiling.calls.load(Ordering::SeqCst), 1);
        assert_eq!(mid_tiling.completed_tiles.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deadline_conv_nonfinite_inputs_bypass_engine_and_timeout_fails_closed() {
        let deadline = Instant::now() + Duration::from_secs(30);

        for poison_kernel in [false, true] {
            let mut fixture = deadline_conv_fixture();
            if poison_kernel {
                fixture.kernel[[fixture.oc_start, 0, 0, 0]] = f32::INFINITY;
            } else {
                fixture.coefficients[[0, fixture.oc_start * fixture.spatial_per_obj]] = f32::NAN;
            }
            let engine = RecordingDeadlineConvEngine::default();
            let actual = bounded_fixture(&engine, &fixture, deadline)
                .expect("nonfinite input uses historical CPU fallback");
            let expected =
                cpu_fixture(&fixture, deadline).expect("nonfinite CPU fallback oracle completes");
            assert_same_f64_image(&actual, &expected, "nonfinite input fallback");
            assert!(
                engine
                    .calls
                    .lock()
                    .expect("recording nonfinite call lock")
                    .is_empty(),
                "nonfinite operands must not reach the bounded engine"
            );
        }

        let fixture = deadline_conv_fixture();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("Instant epoch exceeds one millisecond");
        let no_launch = RecordingDeadlineConvEngine::default();
        let error = conv_group_col_flat_f64_try_deadline_engine(
            &no_launch,
            &fixture.coefficients,
            &fixture.kernel,
            fixture.oc_start,
            fixture.out_c_per_group,
            fixture.kernel_cols_per_group,
            fixture.total_spatial,
            fixture.spatial_per_obj,
            fixture.kernel_spatial,
            fixture.kw,
            expired,
        )
        .expect_err("expired deadline must refuse before engine launch");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert!(no_launch
            .calls
            .lock()
            .expect("recording expired call lock")
            .is_empty());

        // A buggy engine might return success after its deadline. The caller
        // independently re-checks and refuses to publish that late result.
        let late_engine = SleepPastDeadlineConvEngine {
            calls: AtomicUsize::new(0),
        };
        let error = bounded_fixture(
            &late_engine,
            &fixture,
            Instant::now() + Duration::from_millis(5),
        )
        .expect_err("post-deadline engine success must be rejected");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(late_engine.calls.load(Ordering::SeqCst), 1);
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

    /// The reported cap SOURCE must name the probe that actually produced the
    /// number. The estimate path was previously reported as "/proc/meminfo
    /// MemAvailable headroom" on every host, including hosts with no `/proc` at
    /// all, which points operators at a file that cannot exist there.
    #[test]
    fn cap_source_names_the_probe_that_actually_ran() {
        const GB: u64 = 1024 * 1024 * 1024;

        assert!(ConvCrownMemCapSource::HostAvailable
            .label()
            .contains("/proc/meminfo"));
        let estimate = ConvCrownMemCapSource::HostPhysicalHalf.label();
        assert!(
            !estimate.contains("/proc"),
            "the no-/proc estimate must not cite /proc: {estimate}"
        );
        assert!(estimate.contains("physical RAM"), "{estimate}");

        // The tag survives cap narrowing, so the warning an operator reads
        // reflects the probe rather than a fixed string.
        let narrowed = narrowest_conv_crown_mem_cap(
            None,
            Some((32 * GB, ConvCrownMemCapSource::HostPhysicalHalf)),
            None,
            None,
            2,
        );
        assert_eq!(narrowed.source, ConvCrownMemCapSource::HostPhysicalHalf);

        // On a host with no /proc/meminfo the live probe must take the estimate
        // path; on Linux it may legitimately read MemAvailable.
        if let Some((_, source)) = available_system_ram_observation() {
            if !std::path::Path::new("/proc/meminfo").exists() {
                assert_eq!(source, ConvCrownMemCapSource::HostPhysicalHalf);
            }
        }
    }

    #[test]
    fn narrowest_envelope_and_shared_budget_win_conv_crown_oom() {
        const GB: u64 = 1024 * 1024 * 1024;
        // Policy=2 GiB, host has 32 GiB available => 2 GiB share, process has
        // only 8 GiB headroom => 512 MiB process share, Dense pair budget =>
        // 1.5 GiB per member. The enforced process envelope is narrowest.
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(2 * GB as usize),
                Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
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
                Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
                Some(8 * GB),
                Some(3 * GB as usize),
                2,
            )
            .source,
            ConvCrownMemCapSource::ProcessEnvelope
        );
        // The process envelope still wins when it is genuinely the tightest:
        // a 2 GiB headroom gives each admission 128 MiB.
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(2 * GB as usize),
                Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
                Some(2 * GB),
                Some(3 * GB as usize),
                2,
            )
            .bytes,
            128 * MB
        );
        assert_eq!(
            narrowest_conv_crown_mem_cap(
                Some(2 * GB as usize),
                Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
                Some(2 * GB),
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
                Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
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
        // #patches-envelope-nbuffers: the share is derived from the live-buffer
        // count (2 * n_buffers, floored at MIN_ENVELOPE_SHARE_DIVISOR = 4), not
        // from a fixed 16-way split. Each member of a retained lower/upper PAIR
        // therefore gets a quarter of the envelope (half total), as does a
        // standalone transient, while a hypothetical 8-buffer site still
        // narrows to the old 1/16.
        // divisor = max(2 * n_buffers, MIN_ENVELOPE_SHARE_DIVISOR = 4)
        assert_eq!(envelope_crown_mem_cap_bytes(8 * GB, 2), 2 * GB as usize);
        // n_buffers = 1 clamps to the same floor rather than claiming a half.
        assert_eq!(envelope_crown_mem_cap_bytes(8 * GB, 1), 2 * GB as usize);
        // A wide site still narrows to (and past) the old fixed 1/16 share.
        assert_eq!(envelope_crown_mem_cap_bytes(8 * GB, 8), 512 * MB);
        assert_eq!(envelope_crown_mem_cap_bytes(8 * GB, 16), 256 * MB);
        // The MAX clamp keeps this an OOM backstop, not an open grant.
        assert_eq!(
            envelope_crown_mem_cap_bytes(1024 * GB, 1),
            MAX_ADAPTIVE_CROWN_MEM_CAP_MB * 1024 * 1024
        );
        // A kernel process envelope keeps the restored fixed /16 containment
        // for the actual paired/chunked allocation graph, independent of the
        // result-buffer count at this individual call site.
        assert_eq!(process_envelope_crown_mem_cap_bytes(8 * GB, 1), 512 * MB);
        assert_eq!(process_envelope_crown_mem_cap_bytes(8 * GB, 2), 512 * MB);
        assert_eq!(process_envelope_crown_mem_cap_bytes(8 * GB, 16), 256 * MB);
    }

    #[test]
    fn unset_dense_budget_preserves_large_host_adaptive_case() {
        const GB: u64 = 1024 * 1024 * 1024;
        let vgg_buffer_bytes = 1000 * 401_408 * size_of::<f32>();

        let adaptive = narrowest_conv_crown_mem_cap(
            Some(2 * GB as usize),
            Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
            None,
            None,
            2,
        );
        assert_eq!(adaptive.bytes, 2 * GB as usize);
        assert_eq!(adaptive.source, ConvCrownMemCapSource::Policy);
        assert!(vgg_buffer_bytes < adaptive.bytes);

        let explicit_default = narrowest_conv_crown_mem_cap(
            Some(2 * GB as usize),
            Some((32 * GB, ConvCrownMemCapSource::HostAvailable)),
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

    /// #f64-recompute-objective-chunk policy: chunk ONLY where the single-shot
    /// transient would be refused, never otherwise, and never when a chunk
    /// cannot help.
    #[test]
    fn f64_recompute_objective_chunk_policy() {
        // yolo_2023 Conv_12: 10_816 objectives, 26x26 grad, 16 out-channels,
        // 16*3*3 kernel columns. Single-shot transient = 8_422_981_632 bytes
        // against the observed 3 GiB envelope cap -> refused, and refusal is
        // what degraded the whole relation to +/-inf bias.
        let cap = 3 * 1024 * 1024 * 1024;
        assert_eq!(
            largest_conv_crown_scratch_bytes(
                &[(10_816 * 676, 16), (16, 144), (10_816 * 676, 144)],
                size_of::<f64>(),
            ),
            8_422_981_632
        );
        let chunk = f64_recompute_objective_chunk_with_cap(Some(cap), 10_816, 676, 16, 144)
            .expect("over-cap single shot must chunk");
        assert!((1..10_816).contains(&chunk));
        // Every chunked transient is under the cap.
        assert!(
            largest_conv_crown_scratch_bytes(
                &[(chunk * 676, 16), (16, 144), (chunk * 676, 144)],
                size_of::<f64>(),
            ) <= cap
        );

        // Already fits -> unchanged single-shot path.
        assert_eq!(
            f64_recompute_objective_chunk_with_cap(Some(cap), 8, 676, 16, 144),
            None
        );
        // No observable cap -> no refusal is possible, so no chunking.
        assert_eq!(
            f64_recompute_objective_chunk_with_cap(None, 10_816, 676, 16, 144),
            None
        );
        // A single objective already over the cap keeps the honest refusal.
        assert_eq!(
            f64_recompute_objective_chunk_with_cap(Some(1_024), 10_816, 676, 16, 144),
            None
        );
        // Nothing to split.
        assert_eq!(
            f64_recompute_objective_chunk_with_cap(Some(1_024), 1, 676, 16, 144),
            None
        );
    }

    /// The soundness core of #f64-recompute-objective-chunk: result row `i` of
    /// the f64 coefficient recompute depends ONLY on row `i` of the incoming
    /// coefficients, so evaluating an objective sub-range and evaluating the
    /// full range agree on the shared rows. (The chunked branch in
    /// `conv2d_transpose_backward_coeff_f64_with_deadline` is exactly this
    /// decomposition; it only ever fires when the single-shot transient is
    /// refused, which no in-process cap can be forced to reproduce here.)
    ///
    /// Compared with an f64 tolerance rather than bit-for-bit: the inner GEMM
    /// is faer's blocked `mat_mul_f64`, whose reduction blocking may depend on
    /// the row count. That is exactly the freedom the certificate already
    /// grants (`cast + γ_n^f64·S` is summation-order independent — the same
    /// argument the cuBLAS seam and the `rayon::join` pair rely on). A genuine
    /// cross-objective dependency would change the value outright, not by an
    /// f64 ulp.
    #[test]
    fn f64_recompute_rows_are_objective_independent() {
        let (out_c, in_c, kh, kw) = (3usize, 2usize, 3usize, 3usize);
        let (grad_h, grad_w) = (5usize, 4usize);
        let (in_h, in_w) = (5usize, 4usize);
        let num_objectives = 7usize;
        let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, kh, kw]), |ix| {
            ((ix[0] * 37 + ix[1] * 17 + ix[2] * 5 + ix[3]) as f32).mul_add(0.031, -0.4)
        });
        let a =
            Array2::<f32>::from_shape_fn((num_objectives, out_c * grad_h * grad_w), |(r, c)| {
                ((r * 101 + c * 7) % 23) as f32 * 0.017 - 0.2
            });
        let call = |a: &Array2<f32>| {
            conv2d_transpose_backward_coeff_f64_with_deadline(
                a,
                &kernel,
                (1, 1),
                (1, 1),
                (1, 1),
                (in_h, in_w),
                (grad_h, grad_w),
                out_c,
                1,
                1,
                None,
            )
            .expect("f64 recompute")
        };
        let full = call(&a);
        for (start, end) in [(0usize, 3usize), (3, 5), (5, 7)] {
            let part = call(&a.slice(ndarray::s![start..end, ..]).to_owned());
            for r in start..end {
                for p in 0..full.ncols() {
                    let (got, want) = (part[[r - start, p]], full[[r, p]]);
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "row {r} col {p} differs between chunked ({got}) and full ({want}) recompute"
                    );
                }
            }
        }
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
