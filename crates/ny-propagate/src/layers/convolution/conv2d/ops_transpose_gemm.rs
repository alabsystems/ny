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

/// Total wall time inside the flat f64 conv-group column kernel, reported on
/// the `[conv-share]` line in `network/ibp/crown_ibp.rs`.
///
/// The three sub-probes that used to sit beside this one -- FILL, GEMM and
/// COPY, splitting the pollable CPU body -- are gone because they ANSWERED.
/// 227560c01 measured them and recorded "Sub-probes in the pollable CPU body
/// recorded ~nothing", which is what moved suspicion to the sound_f64_gemm
/// engine-attempt path and produced [`CONV_GROUP_ENGINE_NANOS`]. Their writers
/// were removed with that result; only the declarations were left behind, so
/// they had become three statics nothing wrote and nothing read.
pub(crate) static CONV_GROUP_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
fn run_f64_gemm_before_deadline_with_clock<N, G, T>(deadline: Instant, now: N, gemm: G) -> Result<T>
where
    N: FnOnce() -> Instant,
    G: FnOnce() -> T,
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
    // produces bit-identical values without it.
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
/// so higher layers can degrade to sound IBP fallback.
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
                // so the zero-init memset is wasted work.
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
            // This eliminates the CPU col2im bottleneck.
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
        // deadline granularity suffices.
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
            note_conv_f64_scatter_lane(CONV_F64_SCATTER_LANE_PARALLEL);
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
                    // #f64-scatter-hoist: `ih` depends only on `(gy, ki)` and the
                    // two flat bases only on `(gy, gx, ic_local, ki)`, so the
                    // original nest re-derived all of them once per MAC —
                    // `in_c_per_group * kw` times more often than they change.
                    // Hoisting them leaves the (destination index, addend) pair
                    // sequence and its order untouched: no float is read, written,
                    // or rounded differently, so the scatter is bit-identical by
                    // construction. Skipping the `kj` loop when `ih` is out of
                    // range drops only iterations whose `if` was already false.
                    for gy in 0..grad_h {
                        let anchor_h = gy * sh;
                        for gx in 0..grad_w {
                            let anchor_w = gx * sw;
                            let spatial_offset = gy * grad_w + gx;
                            let col_row = obj * spatial_per_obj + spatial_offset;
                            let col_base = col_row * kernel_cols_per_group;
                            for ic_local in 0..in_c_per_group {
                                let ic = ic_start + ic_local;
                                let out_channel_base = ic * input_spatial;
                                let col_channel_base = col_base + ic_local * kernel_spatial;
                                for ki in 0..kh {
                                    let ih = (anchor_h + ki * dh) as isize - ph as isize;
                                    if ih < 0 || ih >= in_h as isize {
                                        continue;
                                    }
                                    let out_row_base = out_channel_base + ih as usize * in_w;
                                    let col_row_base = col_channel_base + ki * kw;
                                    for kj in 0..kw {
                                        let iw = (anchor_w + kj * dw) as isize - pw as isize;
                                        if iw >= 0 && iw < in_w as isize {
                                            obj_out[out_row_base + iw as usize] +=
                                                col_flat[col_row_base + kj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                })?;
        } else if conv_in_size > 0 {
            note_conv_f64_scatter_lane(CONV_F64_SCATTER_LANE_SERIAL);
            let mut scatter_ops = 0usize;
            for (obj, obj_out) in result_flat.chunks_mut(conv_in_size).enumerate() {
                poll_authority()?;
                // Same hoist as the parallel arm above. This arm keeps the
                // per-MAC `scatter_ops` bookkeeping EXACTLY as it was — the
                // counter still advances once per `(gy, gx, ic_local, ki, kj)`
                // tuple and `poll_authority` is still called at the same tuple
                // boundaries — so the bounded-executor poll cadence is unchanged
                // as well as the arithmetic. `out_row_base` is computed only when
                // `ih` is in range, so an out-of-range row cannot overflow the
                // `usize` address arithmetic it never uses.
                for gy in 0..grad_h {
                    let anchor_h = gy * sh;
                    for gx in 0..grad_w {
                        let anchor_w = gx * sw;
                        let spatial_offset = gy * grad_w + gx;
                        let col_row = obj * spatial_per_obj + spatial_offset;
                        let col_base = col_row * kernel_cols_per_group;
                        for ic_local in 0..in_c_per_group {
                            let ic = ic_start + ic_local;
                            let out_channel_base = ic * input_spatial;
                            let col_channel_base = col_base + ic_local * kernel_spatial;
                            for ki in 0..kh {
                                let ih = (anchor_h + ki * dh) as isize - ph as isize;
                                let ih_in_range = ih >= 0 && ih < in_h as isize;
                                let out_row_base = if ih_in_range {
                                    out_channel_base + ih as usize * in_w
                                } else {
                                    0
                                };
                                let col_row_base = col_channel_base + ki * kw;
                                for kj in 0..kw {
                                    if scatter_ops.is_multiple_of(DEADLINE_F64_CPU_POLL_OPS) {
                                        poll_authority()?;
                                    }
                                    scatter_ops = scatter_ops.wrapping_add(1);
                                    let iw = (anchor_w + kj * dw) as isize - pw as isize;
                                    if ih_in_range && iw >= 0 && iw < in_w as isize {
                                        obj_out[out_row_base + iw as usize] +=
                                            col_flat[col_row_base + kj];
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
    let _share_timer =
        crate::layers::convolution::crown_helpers::ShareTimer::new(&CONV_GROUP_NANOS);
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
        let engine_t = Instant::now();
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
        CONV_GROUP_ENGINE_NANOS.fetch_add(
            u64::try_from(engine_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
            std::sync::atomic::Ordering::Relaxed,
        );
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
        CONV_GROUP_ENGINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(result);
    }
    // Engine declined (dispatch cap, refusal, or non-finite image): the
    // pollable body below runs *inside* the engine-attempt timer, so
    // [group-split] engine_attempt covers it. The miss counter makes that
    // attribution visible instead of inferred.
    CONV_GROUP_ENGINE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    let fill_t = Instant::now();
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
    // Fast lane (#conv-fill-hoist): identical scan order, identical values,
    // identical first-non-finite early return — with the per-element 2D
    // bounds-checked index hoisted to one contiguous row slice per `obj` and
    // the poll moved to once per row (>= the legacy 4096-element cadence in
    // responsiveness; ~20 ns per row). Probe-measured motivation: these fills
    // carried ~16 s of the group arm's engine share on `cifar_bias_field_46`
    // while the tile math itself cost 0.055 s (REGRESSION doc, 2026-08-20).
    let a_flat = if a_coefficients.is_standard_layout() {
        a_coefficients.as_slice()
    } else {
        None
    };
    if let (Some(a_flat), true, 0) = (
        a_flat,
        spatial_per_obj > 0,
        total_spatial % spatial_per_obj.max(1),
    ) {
        // Cache-blocked transpose (#conv-fill-hoist leg 2). The row-order fill
        // read one element per cache line (source stride `spatial_per_obj`,
        // measured 2.9 s of the engine attempt). Per `obj`, the source block
        // is (out_c x spatial) row-major and the destination wants its
        // transpose; tiling `spatial` so a destination tile of
        // `s_tile * out_c_per_group` f64s stays L2-resident makes every source
        // read sequential and every destination line written to completion
        // while resident. Values, the any-non-finite -> Ok(None) outcome, and
        // the deadline-checked returns are identical; only the DETECTION order
        // of the first non-finite element changes, which no caller can
        // observe (the fallback signal carries no location).
        let a_cols = a_coefficients.ncols();
        let num_objs = total_spatial / spatial_per_obj;
        let s_tile = (32_768 / out_c_per_group.max(1)).max(64);
        a64.resize(a_len, 0.0);
        // Objects are independent (each writes its own disjoint destination
        // block), so the blocked transpose parallelizes over them with
        // identical values. The early Ok(None)/deadline returns become flags:
        // any non-finite anywhere still yields the same Ok(None), any expiry
        // observed at the same per-oc-pass cadence still yields the same
        // typed error, and expiry keeps precedence over the fallback signal
        // exactly as the serial `check_deadline()?; return Ok(None)` did.
        let fill_bad = std::sync::atomic::AtomicBool::new(false);
        let fill_expired = std::sync::atomic::AtomicBool::new(false);
        a64.par_chunks_mut(spatial_per_obj * out_c_per_group)
            .enumerate()
            .for_each(|(obj, obj_out)| {
                use std::sync::atomic::Ordering::Relaxed;
                if obj >= num_objs {
                    return;
                }
                let row_slice = &a_flat[obj * a_cols..(obj + 1) * a_cols];
                let mut s0 = 0usize;
                while s0 < spatial_per_obj {
                    if fill_bad.load(Relaxed) || fill_expired.load(Relaxed) {
                        return;
                    }
                    let s1 = (s0 + s_tile).min(spatial_per_obj);
                    for oc_local in 0..out_c_per_group {
                        let src_base = (oc_start + oc_local) * spatial_per_obj;
                        let src = &row_slice[src_base + s0..src_base + s1];
                        for (ds, &value) in src.iter().enumerate() {
                            if !value.is_finite() {
                                fill_bad.store(true, Relaxed);
                                return;
                            }
                            obj_out[(s0 + ds) * out_c_per_group + oc_local] = f64::from(value);
                        }
                        if Instant::now() >= deadline {
                            fill_expired.store(true, Relaxed);
                            return;
                        }
                    }
                    s0 = s1;
                }
            });
        if fill_expired.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(per_node_deadline_exceeded());
        }
        if fill_bad.load(std::sync::atomic::Ordering::Relaxed) {
            check_deadline()?;
            return Ok(None);
        }
    } else if let Some(a_flat) = a_flat {
        let a_cols = a_coefficients.ncols();
        for row in 0..total_spatial {
            let obj = row / spatial_per_obj;
            let spatial = row % spatial_per_obj;
            let row_slice = &a_flat[obj * a_cols..(obj + 1) * a_cols];
            let base = oc_start * spatial_per_obj + spatial;
            for oc_local in 0..out_c_per_group {
                let value = row_slice[base + oc_local * spatial_per_obj];
                if !value.is_finite() {
                    check_deadline()?;
                    return Ok(None);
                }
                a64.push(f64::from(value));
            }
            check_deadline()?;
        }
    } else {
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
    // Fast lane (#conv-fill-hoist, W side): `col = ic·kernel_spatial + ki·kw
    // + kj` reconstructs exactly the flat offset inside kernel row `oc` when
    // the kernel is standard-layout [OC, ic_g, kh, kw] with matching group
    // geometry — so each W row is one contiguous slice and the whole
    // div/mod-per-element chain disappears. Guarded structurally; any
    // mismatch keeps the legacy loop.
    let kernel_flat = if kernel.ndim() == 4
        && kernel.is_standard_layout()
        && kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3] == kernel_cols_per_group
        && kernel.shape()[2] * kernel.shape()[3] == kernel_spatial
        && kernel.shape()[3] == kw
    {
        kernel.as_slice_memory_order()
    } else {
        None
    };
    if let Some(kernel_flat) = kernel_flat {
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            let krow = &kernel_flat[oc * kernel_cols_per_group..(oc + 1) * kernel_cols_per_group];
            for &value in krow {
                if !value.is_finite() {
                    check_deadline()?;
                    return Ok(None);
                }
                w64.push(f64::from(value));
            }
            check_deadline()?;
        }
    } else {
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
    CONV_GROUP_ENGINE_FILL_NANOS.fetch_add(
        u64::try_from(fill_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    CONV_GROUP_ENGINE_MACS.fetch_add(
        u64::try_from(
            total_spatial
                .saturating_mul(out_c_per_group)
                .saturating_mul(kernel_cols_per_group),
        )
        .unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    let engine_gemm_t = Instant::now();
    // The engine states its own cancellation granularity; the CPU floor engine
    // widens the accelerator-sized cap to tiles it can block and thread
    // (#cpu-sound-f64-deadline). Accelerator engines return the cap unchanged.
    let dispatch_macs =
        engine.deadline_dispatch_macs_hint(CONV_DEADLINE_F64_ACCELERATOR_MAX_DISPATCH_MACS);
    let gemm_result = engine.gemm_f64_with_deadline(
        total_spatial,
        out_c_per_group,
        kernel_cols_per_group,
        &a64,
        &w64,
        deadline,
        dispatch_macs,
    );
    CONV_GROUP_ENGINE_GEMM_NANOS.fetch_add(
        u64::try_from(engine_gemm_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    match gemm_result {
        Ok(result) if result.len() == output_len => {
            let scan_t = Instant::now();
            // Parallel finite-scan: chunks are independent reads; the serial
            // loop's outcomes map onto flags with the same precedence (expiry
            // beats the non-finite signal, which beats Ok). The badness fold
            // per chunk uses `|=` so the reduction vectorizes.
            let scan_bad = std::sync::atomic::AtomicBool::new(false);
            let scan_expired = std::sync::atomic::AtomicBool::new(false);
            result.par_chunks(PREP_POLL_ELEMENTS).for_each(|chunk| {
                use std::sync::atomic::Ordering::Relaxed;
                if scan_bad.load(Relaxed) || scan_expired.load(Relaxed) {
                    return;
                }
                let mut bad = false;
                for value in chunk {
                    bad |= !value.is_finite();
                }
                if bad {
                    scan_bad.store(true, Relaxed);
                }
                if Instant::now() >= deadline {
                    scan_expired.store(true, Relaxed);
                }
            });
            if scan_expired.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(per_node_deadline_exceeded());
            }
            if scan_bad.load(std::sync::atomic::Ordering::Relaxed) {
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
            CONV_GROUP_ENGINE_SCAN_NANOS.fetch_add(
                u64::try_from(scan_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
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
    let fill_t = Instant::now();
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
    CONV_GROUP_FILL_NANOS.fetch_add(
        u64::try_from(fill_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    // Operand materialization itself can consume the remaining node budget.
    // Poll at the exact launch boundary so expired authority never enters the
    // unpollable faer contraction.
    let gemm_t = Instant::now();
    // Direct-write (#conv-fill-hoist): matmul straight into `col_flat` through
    // a row-major MatMut view — the former per-element `prod[(row, col)]`
    // copy paid a bounds-checked column-major read per output cell (the same
    // defect the deadline-GEMM tile loop had; both probe-measured). Same faer
    // kernel, same single-call blocking, certificate order-independent.
    run_f64_gemm_before_deadline_with_clock(deadline, Instant::now, || {
        let out_view = faer::MatMut::from_row_major_slice_mut(
            &mut col_flat,
            total_spatial,
            kernel_cols_per_group,
        );
        crate::faer_parallelism::mat_mul_f64_into(out_view, &a64, &w64)
    })?;
    CONV_GROUP_GEMM_NANOS.fetch_add(
        u64::try_from(gemm_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );

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

/// Fill-lane identifiers for [`note_conv_fill_lane`]. Live in every build: the
/// lane selections in [`conv_group_col_flat_f64`] and
/// [`conv_group_col_flat_with_engine`] name them unconditionally.
const CONV_FILL_LANE_A_HOISTED: usize = 0;
const CONV_FILL_LANE_A_LEGACY: usize = 1;
const CONV_FILL_LANE_W_HOISTED: usize = 2;
const CONV_FILL_LANE_W_LEGACY: usize = 3;

// Test-only record of which `#conv-fill-hoist` fill lane each of the two
// no-deadline seams took, so `conv_fill_hoist_lanes_match_legacy_loops_bitwise`
// can prove it compared the HOISTED lane against the legacy loops rather than
// the legacy loops against themselves.
//
// The lanes are selected by five conditions that the parity test satisfies only
// BY CONSTRUCTION: `Array2::as_slice` returning `Some`, `total_spatial %
// spatial_per_obj == 0`, and the three kernel shape equalities. A later edit
// that tightened any of them would send BOTH arms of that test down the
// fallback, where it would still pass while comparing the fallback with itself
// — an absence read as evidence. Counting the lane makes that a FAILURE,
// because the test asserts the exact lane vector for every call.
//
// Thread-local, and every counter is bumped on the CALLING thread (never inside
// the `par_chunks_mut` closure), so tests running concurrently in one process
// cannot pollute each other's counts.
#[cfg(test)]
thread_local! {
    /// Per-lane entry counts for this thread, indexed by the
    /// `CONV_FILL_LANE_*` constants above.
    static CONV_FILL_LANE_HITS: std::cell::Cell<[u32; 4]> =
        const { std::cell::Cell::new([0u32; 4]) };
}

/// Record that `lane` was entered on this thread. Outside `cfg(test)` this is
/// an empty function with no counter behind it, so the shipped fills pay
/// nothing for it.
#[cfg(test)]
#[inline]
fn note_conv_fill_lane(lane: usize) {
    CONV_FILL_LANE_HITS.with(|hits| {
        let mut taken = hits.get();
        taken[lane] += 1;
        hits.set(taken);
    });
}

#[cfg(not(test))]
#[inline(always)]
fn note_conv_fill_lane(_lane: usize) {}

/// Read and clear this thread's fill-lane counters, indexed by the
/// `CONV_FILL_LANE_*` constants.
#[cfg(test)]
fn take_conv_fill_lane_hits() -> [u32; 4] {
    CONV_FILL_LANE_HITS.with(|hits| hits.replace([0u32; 4]))
}

/// Scatter-lane identifiers for [`note_conv_f64_scatter_lane`]: which of the two
/// col2im arms of `conv2d_transpose_backward_coeff_f64_with_engine_and_deadline`
/// a group actually ran. Live in every build so the call sites keep naming them.
const CONV_F64_SCATTER_LANE_PARALLEL: usize = 0;
const CONV_F64_SCATTER_LANE_SERIAL: usize = 1;

// Deliberately a SEPARATE counter from `CONV_FILL_LANE_HITS`: that one pins the
// operand fills and is asserted as an exact four-element vector, so widening it
// would rewrite unrelated pins. Bumped on the CALLING thread — on the parallel
// arm before `par_chunks_mut`, never inside the closure — so a test reads its
// own counts even though the scatter body runs on rayon workers. (A plain
// comment: a `///` on a macro invocation documents nothing.)
#[cfg(test)]
thread_local! {
    /// Per-lane entry counts for this thread, indexed by the
    /// `CONV_F64_SCATTER_LANE_*` constants above.
    static CONV_F64_SCATTER_LANE_HITS: std::cell::Cell<[u32; 2]> =
        const { std::cell::Cell::new([0u32; 2]) };
}

/// Record that `lane` was entered on this thread. Outside `cfg(test)` this is an
/// empty function with no counter behind it, so the shipped scatter pays nothing.
#[cfg(test)]
#[inline]
fn note_conv_f64_scatter_lane(lane: usize) {
    CONV_F64_SCATTER_LANE_HITS.with(|hits| {
        let mut taken = hits.get();
        taken[lane] += 1;
        hits.set(taken);
    });
}

#[cfg(not(test))]
#[inline(always)]
fn note_conv_f64_scatter_lane(_lane: usize) {}

/// Read and clear this thread's f64 scatter-lane counters, indexed by the
/// `CONV_F64_SCATTER_LANE_*` constants.
#[cfg(test)]
fn take_conv_f64_scatter_lane_hits() -> [u32; 2] {
    CONV_F64_SCATTER_LANE_HITS.with(|hits| hits.replace([0u32; 2]))
}

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
    let fill_t = Instant::now();
    // Fast lane (#conv-fill-hoist), ported from the deadline twin
    // `conv_group_col_flat_f64_try_deadline_engine` — but the two sides do NOT
    // have the twin's shape here, because this arm's destination is a faer
    // `Mat` (column-major, `row_capacity`-padded) rather than the twin's
    // row-major `Vec`. `Mat::from_fn` fills column-by-column (faer 0.24.0
    // `mat/matown.rs::init_with`: outer loop over `Cols::indices`, inner over
    // rows), so for a fixed `oc_local` the closure already walks the source in
    // ascending `obj`, ascending `spatial` — runs of `spatial_per_obj`
    // consecutive f32 — while writing a contiguous destination column. There
    // is no transpose left to block on this side; what the closure pays per
    // element is the `row / spatial_per_obj` div/rem pair and the
    // bounds-checked 2-D `Array2` index, and that is what the run-copy below
    // removes. It is not free: `Mat::zeros` writes the whole buffer once
    // before the fill overwrites it, a pass `from_fn` does not pay. NOT
    // measured — there is no benchmark data on the host this was written on.
    //
    // BIT-IDENTICAL: `Mat::zeros` IS `Mat::from_fn(|_, _| 0.0)` on the same
    // `RawMat::try_with_capacity(nrows, ncols)`, so the operand handed to faer
    // has the same dims, the same `row_capacity` and the same strides — which
    // matters because faer selects kernels by operand layout — and every
    // in-bounds cell is then overwritten with the f64 the closure would have
    // produced for it. Only the fill order and the index arithmetic change.
    // The source is re-sliced per objective ROW before being indexed, so a
    // read that would have left row `obj` still panics exactly where the 2-D
    // index did instead of silently reading a neighbouring row.
    let a_flat = if a_coefficients.is_standard_layout() {
        a_coefficients.as_slice()
    } else {
        None
    };
    // `.max(1)` is load-bearing, not defensive noise: every element of a tuple
    // expression is evaluated before the pattern is matched, so the remainder
    // is computed even on the calls where the `spatial_per_obj > 0` guard
    // sitting beside it is false.
    let a64 = if let (Some(a_flat), true, 0) = (
        a_flat,
        spatial_per_obj > 0 && out_c_per_group > 0 && total_spatial > 0,
        total_spatial % spatial_per_obj.max(1),
    ) {
        note_conv_fill_lane(CONV_FILL_LANE_A_HOISTED);
        let a_cols = a_coefficients.ncols();
        let mut a64 = Mat::<f64>::zeros(total_spatial, out_c_per_group);
        for oc_local in 0..out_c_per_group {
            let src_base = (oc_start + oc_local) * spatial_per_obj;
            // Destination column `oc_local` is contiguous (row stride 1) and
            // the divisibility guard makes `total_spatial` an exact multiple
            // of `spatial_per_obj`, so its chunks are whole objectives in
            // ascending `obj` order — the same (obj, spatial) -> row map the
            // closure computed with div/rem.
            let column = a64.col_as_slice_mut(oc_local);
            for (obj, dst_run) in column.chunks_mut(spatial_per_obj).enumerate() {
                let row_slice = &a_flat[obj * a_cols..(obj + 1) * a_cols];
                let src = &row_slice[src_base..src_base + spatial_per_obj];
                for (dst, &value) in dst_run.iter_mut().zip(src) {
                    *dst = f64::from(value);
                }
            }
        }
        a64
    } else {
        note_conv_fill_lane(CONV_FILL_LANE_A_LEGACY);
        Mat::<f64>::from_fn(total_spatial, out_c_per_group, |row, oc_local| {
            let obj = row / spatial_per_obj;
            let spatial = row % spatial_per_obj;
            let oc = oc_start + oc_local;
            f64::from(a_coefficients[[obj, oc * spatial_per_obj + spatial]])
        })
    };
    // W side (#conv-fill-hoist, same structural kernel guards as the twin).
    // For a standard-layout `[OC, ic_g, kh, kw]` kernel the flat offset of
    // `[oc, ic, ki, kj]` is `oc*(s1*s2*s3) + ic*(s2*s3) + ki*s3 + kj`; under
    // the three shape equalities below that is exactly
    // `oc * kernel_cols_per_group + col`, so both div/rem pairs and the 4-axis
    // bounds-checked `ArrayD` index collapse to one flat read. The traversal
    // stays `from_fn`'s, so the operand is filled cell for cell in the
    // identical order and only the address computation changes.
    //
    // What is NOT fixed here, deliberately: because the destination is
    // column-major, `from_fn` runs `oc_local` innermost, so this flat read
    // still strides by `kernel_cols_per_group`. Blocking it would need several
    // destination columns live at once, which faer's owned `Mat` exposes only
    // one `&mut` column slice at a time; and this operand is the kernel group
    // itself (`out_c_per_group x kernel_cols_per_group`) — A is the operand
    // that grows with `total_spatial`. The twin gets a straight contiguous row
    // copy on this side only because its destination is row-major.
    let kernel_flat = if kernel.ndim() == 4
        && kernel.is_standard_layout()
        && kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3] == kernel_cols_per_group
        && kernel.shape()[2] * kernel.shape()[3] == kernel_spatial
        && kernel.shape()[3] == kw
    {
        kernel.as_slice_memory_order()
    } else {
        None
    };
    let w64 = if let Some(kernel_flat) = kernel_flat {
        note_conv_fill_lane(CONV_FILL_LANE_W_HOISTED);
        Mat::<f64>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
            let oc = oc_start + oc_local;
            f64::from(kernel_flat[oc * kernel_cols_per_group + col])
        })
    } else {
        note_conv_fill_lane(CONV_FILL_LANE_W_LEGACY);
        Mat::<f64>::from_fn(out_c_per_group, kernel_cols_per_group, |oc_local, col| {
            let oc = oc_start + oc_local;
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            f64::from(kernel[[oc, ic, ki, kj]])
        })
    };
    CONV_GROUP_FILL_NANOS.fetch_add(
        u64::try_from(fill_t.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    let prod = crate::faer_parallelism::mat_mul_f64(&a64, &w64);
    let mut col_flat = vec![0.0f64; total_spatial * kernel_cols_per_group];
    // Copy-out (#conv-fill-hoist, transpose side). `prod` is a column-major
    // faer `Mat` and `col_flat` is row-major, so the former row-major sweep
    // read `prod[(row, col)]` down a column stride, holding one open cache
    // line per column, and paid a bounds check per cell. That is the same
    // defect `faer_parallelism.rs` records against the deadline-GEMM tile loop
    // ("the per-element bounds-checked 2D `dst[(i, j)]` copy"). Only the
    // SOURCE stream becomes sequential: per column the tile reads `r1 - r0`
    // consecutive f64s out of `prod`. The destination is NOT written in runs —
    // each store lands at `dst_row[col]`, `kernel_cols_per_group` apart, one
    // element per column pass. What the tiling buys on that side is residency,
    // not sequentiality: a 64x64 f64 tile touches 512 B in each of 64
    // destination rows (32 KiB), so every destination line it opens is filled
    // to completion across the tile's 64 column passes before it is evicted.
    //
    // WHICH SIDE THE WIN COMES FROM — and it is workload-dependent and NOT
    // MEASURED. The pre-existing sweep already had a perfectly sequential
    // destination; its defect was the SOURCE, `prod[(row, col)]` strided by
    // `prod`'s column stride (>= `total_spatial`), so the inner sweep over
    // `col` held `kernel_cols_per_group` distinct cache lines open at once.
    // That only hurts once they stop fitting: at `kernel_cols_per_group = 72`
    // (parity case 1 in this file's own tests) it is 72 lines = 4.5 KiB,
    // trivially L1-resident and reused by 8 consecutive rows, and the tiling is
    // then pure loop overhead. At cGAN-class widths (`ic_g * kh * kw` in the
    // thousands) it is hundreds of KiB and each load pulls a line used once.
    // What the rewrite buys unconditionally is the bounds check: the source is
    // now read through a slice iterator instead of `prod[(row, col)]`.
    //
    // The closest measured evidence in the tree is not this loop but its
    // cousin: `faer_parallelism.rs`'s 2026-08-21 "Receipts for this choice"
    // acceptance row timed the same column-major-to-row-major copy over the
    // deadline GEMM's tiles at 21.5 s per-element against 16.7 s L1-blocked
    // (tile_mm 7.8 + copy 7.2). Different loop, different shapes — a
    // direction, not a number for this seam.
    //
    // This moves NO float through an arithmetic operation — every value is the
    // same `prod` cell stored at the same `col_flat` offset, a pure reordering
    // of stores — so it is bit-identical by construction. Deliberately NOT
    // taken: making `mat_mul_f64` write `col_flat` directly through a
    // row-major `MatMut`. That eliminates the copy entirely, but
    // `faer_parallelism.rs` records in its own words that faer's dot order
    // tracks destination layout and that exactly this rewrite cost 1 ULP, so
    // it is a measured change, not a free one.
    const COPY_OUT_TILE: usize = 64;
    let mut r0 = 0usize;
    while r0 < total_spatial {
        let r1 = (r0 + COPY_OUT_TILE).min(total_spatial);
        let mut c0 = 0usize;
        while c0 < kernel_cols_per_group {
            let c1 = (c0 + COPY_OUT_TILE).min(kernel_cols_per_group);
            // `kernel_cols_per_group > 0` inside this loop, so the row chunking
            // below is well defined and every chunk is one whole output row.
            let dst_tile = &mut col_flat[r0 * kernel_cols_per_group..r1 * kernel_cols_per_group];
            for col in c0..c1 {
                let src = &prod.col_as_slice(col)[r0..r1];
                for (dst_row, &value) in dst_tile.chunks_mut(kernel_cols_per_group).zip(src) {
                    dst_row[col] = value;
                }
            }
            c0 = c1;
        }
        r0 = r1;
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
    // Fast lane (#conv-fill-hoist), ported from the deadline twin
    // `conv_group_col_flat_f64_try_deadline_engine`. Both arms had the same
    // two pre-hoist fills; only the deadline one was repaired, and this is the
    // arm the BATCHED beta-CROWN/BaB verdict path reaches (no deadline is in
    // scope there), via `conv_group_col_flat_via_engine`. Same operand values,
    // same row-major destination layout, same buffer: only the fill's scan
    // order and its index arithmetic change.
    let a_flat = if a_coefficients.is_standard_layout() {
        a_coefficients.as_slice()
    } else {
        None
    };
    // The guard is written to read identically to the one in
    // `conv_group_col_flat_f64`, `total_spatial > 0` included. That term is
    // redundant on both sides — `par_chunks_mut` on an empty slice yields no
    // chunks, and `Mat::zeros(0, _)` has no cells — but two ports of one
    // optimisation whose guards differ invite the next reader to hunt for a
    // reason that does not exist.
    //
    // `.max(1)` is load-bearing, not defensive noise: every element of a tuple
    // expression is evaluated before the pattern is matched, so the remainder
    // is computed even on the calls where the `spatial_per_obj > 0` guard
    // sitting beside it is false.
    if let (Some(a_flat), true, 0) = (
        a_flat,
        spatial_per_obj > 0 && out_c_per_group > 0 && total_spatial > 0,
        total_spatial % spatial_per_obj.max(1),
    ) {
        // Cache-blocked per-objective transpose. The legacy loop below walks
        // `oc_local` innermost against a source that strides by
        // `spatial_per_obj` f32, so once that stride covers a line it reads one
        // element per cache line — the shape the twin repaired. Per `obj`
        // the source block is (out_c × spatial) row-major and the destination
        // wants its transpose, so tiling `spatial` to keep a destination tile
        // of `s_tile * out_c_per_group` f64s L2-resident makes every source
        // read sequential and finishes each destination line while it is still
        // resident.
        //
        // Objectives own disjoint destination chunks, and the divisibility
        // guard makes the chunk count exactly `total_spatial / spatial_per_obj`
        // (every chunk one whole objective), so the `par_chunks_mut` split
        // writes the same f64 to the same offset as the serial loop — there is
        // no cross-chunk read and no accumulation. Unlike the twin this lane
        // has no deadline and no non-finite early return to reproduce, so the
        // closure carries no flags. `out_c_per_group > 0` joins the guard
        // because `par_chunks_mut` panics on a zero chunk size; the twin
        // reaches its copy only past an early return this arm has no place for.
        //
        // Nesting is safe here even though this seam is normally reached from
        // inside an outer BaB `par_iter`: a `par_chunks_mut` scope is
        // work-stealing, so it finds no idle worker when the batch already
        // saturates the pool and fans out when it does not (the self-balancing
        // argument `faer_parallelism.rs`'s `NestedFaerParGuard` records). It
        // spawns no threads, and it holds no lock: `with_engine`
        // (`sound_f64_gemm.rs`) invokes this closure AFTER its
        // `OnceLock::get_or_init` has returned, not while the slot is being
        // initialized.
        //
        // Reads are taken through a per-objective ROW slice, so an index that
        // would have left row `obj` panics exactly where the 2-D `Array2`
        // index did rather than silently reading the next row.
        note_conv_fill_lane(CONV_FILL_LANE_A_HOISTED);
        let a_cols = a_coefficients.ncols();
        let s_tile = (32_768 / out_c_per_group.max(1)).max(64);
        a64.par_chunks_mut(spatial_per_obj * out_c_per_group)
            .enumerate()
            .for_each(|(obj, obj_out)| {
                let row_slice = &a_flat[obj * a_cols..(obj + 1) * a_cols];
                let mut s0 = 0usize;
                while s0 < spatial_per_obj {
                    let s1 = (s0 + s_tile).min(spatial_per_obj);
                    for oc_local in 0..out_c_per_group {
                        let src_base = (oc_start + oc_local) * spatial_per_obj;
                        let src = &row_slice[src_base + s0..src_base + s1];
                        for (ds, &value) in src.iter().enumerate() {
                            obj_out[(s0 + ds) * out_c_per_group + oc_local] = f64::from(value);
                        }
                    }
                    s0 = s1;
                }
            });
    } else {
        note_conv_fill_lane(CONV_FILL_LANE_A_LEGACY);
        for row in 0..total_spatial {
            let obj = row / spatial_per_obj;
            let spatial = row % spatial_per_obj;
            for oc_local in 0..out_c_per_group {
                let oc = oc_start + oc_local;
                a64[row * out_c_per_group + oc_local] =
                    f64::from(a_coefficients[[obj, oc * spatial_per_obj + spatial]]);
            }
        }
    }
    // W: (out_c_per_group × kernel_cols_per_group) row-major.
    let mut w64 = vec![0.0f64; out_c_per_group * kernel_cols_per_group];
    // Fast lane (#conv-fill-hoist, W side), same structural guards as the
    // twin: for a standard-layout `[OC, ic_g, kh, kw]` kernel the flat offset
    // of `[oc, ic, ki, kj]` is `oc*(s1*s2*s3) + ic*(s2*s3) + ki*s3 + kj`, and
    // under the three shape equalities below that is exactly
    // `oc * kernel_cols_per_group + col`. Because the destination row here is
    // also contiguous, each `oc` becomes one contiguous-to-contiguous widening
    // copy and the whole `/kernel_spatial; %; /kw; %kw` chain plus the 4-axis
    // bounds-checked `ArrayD` index disappear. Any shape mismatch keeps the
    // legacy loop.
    let kernel_flat = if kernel.ndim() == 4
        && kernel.is_standard_layout()
        && kernel.shape()[1] * kernel.shape()[2] * kernel.shape()[3] == kernel_cols_per_group
        && kernel.shape()[2] * kernel.shape()[3] == kernel_spatial
        && kernel.shape()[3] == kw
    {
        kernel.as_slice_memory_order()
    } else {
        None
    };
    if let Some(kernel_flat) = kernel_flat {
        note_conv_fill_lane(CONV_FILL_LANE_W_HOISTED);
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            let src = &kernel_flat[oc * kernel_cols_per_group..(oc + 1) * kernel_cols_per_group];
            let dst_base = oc_local * kernel_cols_per_group;
            let dst_row = &mut w64[dst_base..dst_base + kernel_cols_per_group];
            for (dst, &value) in dst_row.iter_mut().zip(src) {
                *dst = f64::from(value);
            }
        }
    } else {
        note_conv_fill_lane(CONV_FILL_LANE_W_LEGACY);
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

/// Wall time spent in the sound_f64_gemm ENGINE-ATTEMPT path of the conv-group
/// arm, reported alongside the group total on `[conv-share]`.
///
/// This is the probe 61c628a76 landed to test its own claim that "the group
/// arm's time sits in the sound_f64_gemm engine-attempt path". Reading it is
/// the point: written-but-never-read, it measures into a void and the claim
/// stays untested, which is the state it was in when restored.
pub(crate) static CONV_GROUP_FILL_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_GEMM_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_ENGINE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Engine-attempt outcome counters: HITS = the explicit bounded engine method
/// produced the contraction; MISSES = it declined (`Ok(None)`) and the
/// pollable CPU body ran *inside* the engine-attempt timer. A nonzero miss
/// count means `engine_attempt` nanos are mostly pollable-fallback cost, not
/// engine cost — the attribution 61c628a76's probe could not distinguish.
pub(crate) static CONV_GROUP_ENGINE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Phase split INSIDE the engine attempt (fill → engine GEMM → finite scan),
/// plus total contraction MACs, so `[group-split]` can attribute the attempt
/// wall time to preparation, arithmetic, or validation instead of guessing.
pub(crate) static CONV_GROUP_ENGINE_FILL_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_ENGINE_GEMM_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_ENGINE_SCAN_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_ENGINE_MACS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_GROUP_ENGINE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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

    /// #conv-fill-hoist parity for BOTH no-deadline arms: the hoisted fills
    /// must reproduce the legacy loops BIT-FOR-BIT, not to a tolerance. The
    /// hoists are a pure reordering plus cheaper address arithmetic — same f32
    /// widened to the same f64, written to the same offset of the same operand
    /// layout — so any drift at all means an index or a layout changed, and a
    /// changed operand layout is exactly what makes faer pick a different
    /// kernel (`faer_parallelism.rs` records that costing 1 ULP).
    ///
    /// LAYOUT is the only thing varied. A C-order `Array2` and a C-order
    /// `[OC, ic_g, kh, kw]` kernel satisfy the structural guards and take the
    /// hoisted lanes; the same values in an F-order `Array2` and an
    /// axis-permuted kernel fail them and take the legacy loops. The two cases
    /// are sized so both new tile loops cross a boundary: case 1 makes the
    /// copy-out tile partial in BOTH dimensions (75 rows x 72 cols against a
    /// 64x64 tile), and case 2 makes `out_c_per_group` large enough that the A
    /// lane's `s_tile` is 64 against `spatial_per_obj = 100`, so it runs one
    /// full tile and one partial one. No kernel axis is length 1 in either
    /// case: `is_standard_layout` SKIPS unit axes, so a permuted `[N,1,1,1]`
    /// kernel still reports C-order and would silently take the fast lane the
    /// case exists to avoid.
    ///
    /// IT FAILS CLOSED, which is the whole point of the lane counters. Two
    /// calls agreeing is evidence only while the first of them actually ENTERS
    /// the hoisted lane. Those lanes are gated by five conditions this test
    /// satisfies by construction and never observed before — `as_slice` being
    /// `Some`, the `total_spatial % spatial_per_obj` divisibility, and three
    /// kernel shape equalities — so an edit tightening any one of them would
    /// have sent both arms down the fallback and left a green test comparing
    /// the fallback with itself. Every call below is therefore bracketed by
    /// `take_conv_fill_lane_hits` and asserted to have entered its two lanes
    /// exactly once and the other two not at all, and every guard condition is
    /// asserted on the fixture as well. The lane vector also pins the routing:
    /// both cases keep MACs under `CONV_SOUND_F64_GEMM_MIN_MACS` (asserted),
    /// so `conv_group_col_flat_f64` stays on its faer CPU path instead of
    /// re-entering `conv_group_col_flat_with_engine` — which would double the
    /// counts — and the test never touches the process-global engine slot.
    #[test]
    fn conv_fill_hoist_lanes_match_legacy_loops_bitwise() {
        use super::{
            conv_group_col_flat_f64, conv_group_col_flat_with_engine, take_conv_fill_lane_hits,
            CONV_FILL_LANE_A_HOISTED, CONV_FILL_LANE_A_LEGACY, CONV_FILL_LANE_W_HOISTED,
            CONV_FILL_LANE_W_LEGACY, CONV_SOUND_F64_GEMM_MIN_MACS,
        };
        use ndarray::{Array2, ArrayD, IxDyn};
        use ny_core::NaiveCpuGemmEngine;

        // The lane vectors below are written positionally, so pin the indices:
        // renumbering the constants must break this test, not silently rotate
        // what it asserts.
        assert_eq!(
            [
                CONV_FILL_LANE_A_HOISTED,
                CONV_FILL_LANE_A_LEGACY,
                CONV_FILL_LANE_W_HOISTED,
                CONV_FILL_LANE_W_LEGACY,
            ],
            [0usize, 1, 2, 3],
            "lane vectors are [A hoisted, A legacy, W hoisted, W legacy]"
        );
        const BOTH_HOISTED: [u32; 4] = [1, 0, 1, 0];
        const BOTH_LEGACY: [u32; 4] = [0, 1, 0, 1];

        // (num_obj, spatial_per_obj, oc, icpg, kh, kw)
        const CASES: [(usize, usize, usize, usize, usize, usize); 2] =
            [(3, 25, 4, 8, 3, 3), (2, 100, 512, 2, 2, 2)];

        for &(num_obj, spatial_per_obj, oc, icpg, kh, kw) in &CASES {
            let total_spatial = num_obj * spatial_per_obj;
            let kernel_spatial = kh * kw;
            let kcpg = icpg * kernel_spatial;
            let a_value = |obj: usize, j: usize| ((obj * 31 + j * 17) % 97) as f32 * 0.03 - 1.4;
            let k_value = |o: usize, i: usize, r: usize, c: usize| {
                ((o * 13 + i * 7 + r * 3 + c) % 89) as f32 * 0.05 - 2.1
            };

            let a_c =
                Array2::from_shape_fn((num_obj, oc * spatial_per_obj), |(obj, j)| a_value(obj, j));
            let k_c = ArrayD::from_shape_fn(IxDyn(&[oc, icpg, kh, kw]), |d| {
                k_value(d[0], d[1], d[2], d[3])
            });

            // Every condition the hoisted lanes gate on, observed on the
            // fixture rather than assumed. `is_standard_layout` is one of five.
            assert!(a_c.is_standard_layout(), "C-order A must report C order");
            assert!(
                a_c.as_slice().is_some(),
                "the A hoist reads through `as_slice`, which must be `Some`"
            );
            assert_eq!(
                a_c.ncols(),
                oc * spatial_per_obj,
                "the A hoist slices each objective row by `ncols()`"
            );
            assert!(spatial_per_obj > 0 && oc > 0 && total_spatial > 0);
            assert_eq!(
                total_spatial % spatial_per_obj,
                0,
                "the A hoist needs exact divisibility"
            );
            assert_eq!(k_c.ndim(), 4, "the W hoist requires a 4-axis kernel");
            assert!(
                k_c.is_standard_layout(),
                "C-order kernel must report C order"
            );
            assert!(
                k_c.as_slice_memory_order().is_some(),
                "the W hoist reads through `as_slice_memory_order`"
            );
            assert_eq!(k_c.shape()[1] * k_c.shape()[2] * k_c.shape()[3], kcpg);
            assert_eq!(k_c.shape()[2] * k_c.shape()[3], kernel_spatial);
            assert_eq!(k_c.shape()[3], kw);
            assert!(
                total_spatial * oc * kcpg < CONV_SOUND_F64_GEMM_MIN_MACS,
                "the case must stay under the engine-routing threshold, or \
                 `conv_group_col_flat_f64` re-enters the engine seam and every \
                 lane count below doubles"
            );

            // Same values, layouts the guards reject.
            let a_f =
                Array2::from_shape_fn((oc * spatial_per_obj, num_obj), |(j, obj)| a_value(obj, j))
                    .reversed_axes();
            let k_p = ArrayD::from_shape_fn(IxDyn(&[kw, kh, icpg, oc]), |d| {
                k_value(d[3], d[2], d[1], d[0])
            })
            .permuted_axes(IxDyn(&[3, 2, 1, 0]));
            assert!(
                !a_f.is_standard_layout(),
                "F-order A must take the fallback"
            );
            assert!(
                a_f.as_slice().is_none(),
                "F-order A must not expose a C-order flat slice"
            );
            assert!(
                !k_p.is_standard_layout(),
                "permuted kernel must take the fallback"
            );
            assert!(a_f == a_c, "the two A layouts must hold the same values");
            assert!(
                k_p == k_c,
                "the two kernel layouts must hold the same values"
            );

            // Discard any residue so each call is measured in isolation.
            let _ = take_conv_fill_lane_hits();

            let hoisted = conv_group_col_flat_with_engine(
                &NaiveCpuGemmEngine,
                &a_c,
                &k_c,
                0,
                oc,
                kcpg,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
            )
            .expect("engine seam, hoisted fills");
            assert_eq!(
                take_conv_fill_lane_hits(),
                BOTH_HOISTED,
                "engine seam on C-order operands must enter BOTH hoisted lanes exactly once"
            );
            let legacy = conv_group_col_flat_with_engine(
                &NaiveCpuGemmEngine,
                &a_f,
                &k_p,
                0,
                oc,
                kcpg,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
            )
            .expect("engine seam, legacy fills");
            assert_eq!(
                take_conv_fill_lane_hits(),
                BOTH_LEGACY,
                "engine seam on rejected layouts must fall to BOTH legacy loops"
            );
            assert_eq!(hoisted.len(), total_spatial * kcpg);
            assert_eq!(hoisted.len(), legacy.len());
            for (i, (fast, slow)) in hoisted.iter().zip(&legacy).enumerate() {
                assert_eq!(
                    fast.to_bits(),
                    slow.to_bits(),
                    "engine-seam bit drift at {i}: {fast:e} vs {slow:e}"
                );
            }

            let hoisted_cpu = conv_group_col_flat_f64(
                &a_c,
                &k_c,
                0,
                oc,
                kcpg,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
            );
            assert_eq!(
                take_conv_fill_lane_hits(),
                BOTH_HOISTED,
                "faer CPU lane on C-order operands must enter BOTH hoisted lanes exactly once"
            );
            let legacy_cpu = conv_group_col_flat_f64(
                &a_f,
                &k_p,
                0,
                oc,
                kcpg,
                total_spatial,
                spatial_per_obj,
                kernel_spatial,
                kw,
            );
            assert_eq!(
                take_conv_fill_lane_hits(),
                BOTH_LEGACY,
                "faer CPU lane on rejected layouts must fall to BOTH legacy loops"
            );
            assert_eq!(hoisted_cpu.len(), total_spatial * kcpg);
            assert_eq!(hoisted_cpu.len(), legacy_cpu.len());
            for (i, (fast, slow)) in hoisted_cpu.iter().zip(&legacy_cpu).enumerate() {
                assert_eq!(
                    fast.to_bits(),
                    slow.to_bits(),
                    "faer CPU lane bit drift at {i}: {fast:e} vs {slow:e}"
                );
            }
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
    /// memset. This asserts the row-major layout and values are
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

    /// Independent naive model of the f64 conv-transpose col2im scatter, walked
    /// in the mathematical order `(obj, oc, gy, gx, ic_local, ki, kj)` — a
    /// DIFFERENT nesting from the shipped `GEMM over oc, then scatter` split.
    /// The caller feeds it only small integers, so every product and every
    /// partial sum is an exact f64 and no accumulation order can separate the
    /// two; that is what licenses asserting equality on BITS instead of a
    /// tolerance.
    #[allow(clippy::too_many_arguments)]
    fn naive_f64_transpose_backward(
        a_coefficients: &Array2<f32>,
        kernel: &ArrayD<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        output_size: (usize, usize),
        grad_size: (usize, usize),
        groups: usize,
    ) -> Array2<f64> {
        let (in_h, in_w) = output_size;
        let (grad_h, grad_w) = grad_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let out_c = kernel.shape()[0];
        let in_c_per_group = kernel.shape()[1];
        let kh = kernel.shape()[2];
        let kw = kernel.shape()[3];
        let out_c_per_group = out_c / groups;
        let spatial_per_obj = grad_h * grad_w;
        let input_spatial = in_h * in_w;
        let num_objectives = a_coefficients.nrows();
        let mut out =
            Array2::<f64>::zeros((num_objectives, in_c_per_group * groups * input_spatial));
        for obj in 0..num_objectives {
            for oc in 0..out_c {
                let ic_start = (oc / out_c_per_group) * in_c_per_group;
                for gy in 0..grad_h {
                    for gx in 0..grad_w {
                        let av = f64::from(
                            a_coefficients[[obj, oc * spatial_per_obj + gy * grad_w + gx]],
                        );
                        for ic_local in 0..in_c_per_group {
                            for ki in 0..kh {
                                let ih = (gy * sh + ki * dh) as isize - ph as isize;
                                for kj in 0..kw {
                                    let iw = (gx * sw + kj * dw) as isize - pw as isize;
                                    if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let col = (ic_start + ic_local) * input_spatial
                                            + ih as usize * in_w
                                            + iw as usize;
                                        out[[obj, col]] +=
                                            av * f64::from(kernel[[oc, ic_local, ki, kj]]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// #f64-scatter-hoist pin. The hoist lifted `ih`, both flat bases and an
    /// out-of-range row skip out of the innermost loop; it must not have moved a
    /// single float. The geometry exercises every part that moved: `ph = 1` with
    /// `sh = dh = 2` drives `ih` NEGATIVE at `gy = ki = 0` and PAST `in_h` at
    /// `gy = 2, ki = 1`, so the new `continue` fires in both directions (only
    /// input rows 1 and 3 can receive anything at all, asserted below);
    /// `pw = 2` drives `iw` negative; `groups = 2` makes `ic_start` nonzero; and
    /// no two of `sh/sw`, `dh/dw`, `kh/kw`, `grad_h/grad_w`, `in_h/in_w` are
    /// equal, so a transposed index cannot pass by symmetry.
    ///
    /// IT FAILS CLOSED ON LANE COVERAGE. Two calls agreeing is evidence only
    /// while each one enters the arm it is supposed to, so both arms are
    /// counted and each is asserted to have run once PER GROUP with the other at
    /// zero: a reroute that quietly sends both calls down one arm breaks this
    /// test instead of comparing an arm with itself. The bounded call
    /// additionally pins `ordinary_calls`, which only the serial bounded-host
    /// seam can bump.
    #[test]
    fn f64_col2im_scatter_hoist_matches_naive_reference_on_both_arms() {
        use super::{
            take_conv_f64_scatter_lane_hits, CONV_F64_SCATTER_LANE_PARALLEL,
            CONV_F64_SCATTER_LANE_SERIAL,
        };

        // The lane vectors below are written positionally, so pin the indices:
        // renumbering the constants must break this test, not rotate what it
        // asserts.
        assert_eq!(
            [CONV_F64_SCATTER_LANE_PARALLEL, CONV_F64_SCATTER_LANE_SERIAL],
            [0usize, 1],
            "lane vectors are [parallel, serial]"
        );

        let out_c = 4usize;
        let groups = 2usize;
        let in_c_per_group = 3usize;
        let in_c = in_c_per_group * groups;
        let (kh, kw) = (2usize, 3usize);
        let stride = (2usize, 1usize);
        let padding = (1usize, 2usize);
        let dilation = (2usize, 1usize);
        let grad_size = (3usize, 4usize);
        let output_size = (5usize, 4usize);
        let num_objectives = 3usize;
        let spatial_per_obj = grad_size.0 * grad_size.1;
        let input_spatial = output_size.0 * output_size.1;

        let a = Array2::from_shape_fn((num_objectives, out_c * spatial_per_obj), |(i, j)| {
            ((i * 7 + j * 5) % 9) as f32 - 4.0
        });
        let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c_per_group, kh, kw]), |d| {
            ((d[0] * 3 + d[1] * 5 + d[2] * 7 + d[3]) % 7) as f32 - 3.0
        });
        assert!(
            a.iter().chain(kernel.iter()).all(|v| v.fract() == 0.0),
            "fixture must be integer-valued: that is what licenses the bit assert"
        );

        let expected = naive_f64_transpose_backward(
            &a,
            &kernel,
            stride,
            padding,
            dilation,
            output_size,
            grad_size,
            groups,
        );
        // An all-zero or fully dense expectation would prove nothing about the
        // guard: demand both a written cell and a skipped one.
        assert!(
            expected.iter().any(|v| *v != 0.0),
            "fixture must actually write something"
        );
        assert!(
            expected.iter().any(|v| *v == 0.0),
            "fixture must leave out-of-range cells untouched"
        );

        // Input rows 0, 2 and 4 are unreachable for this geometry, so they pin
        // the skip directly: a mis-hoisted `ih` would deposit into them.
        let assert_unreachable_rows_empty = |label: &str, got: &Array2<f64>| {
            for obj in 0..num_objectives {
                for ic in 0..in_c {
                    for ih in [0usize, 2, 4] {
                        for iw in 0..output_size.1 {
                            let col = ic * input_spatial + ih * output_size.1 + iw;
                            assert_eq!(
                                got[[obj, col]],
                                0.0,
                                "{label}: row {ih} is out of reach yet obj {obj} ic {ic} \
                                 iw {iw} received a contribution"
                            );
                        }
                    }
                }
            }
        };

        let _ = take_conv_f64_scatter_lane_hits();
        let parallel = conv2d_transpose_backward_coeff_f64(
            &a,
            &kernel,
            stride,
            padding,
            dilation,
            output_size,
            grad_size,
            out_c,
            groups,
            1,
        )
        .expect("engine-free f64 backward");
        assert_eq!(
            take_conv_f64_scatter_lane_hits(),
            [2, 0],
            "engine-free call must take the parallel scatter once per group"
        );
        assert_eq!(parallel.dim(), expected.dim());
        for (got, want) in parallel.iter().zip(expected.iter()) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "parallel scatter drifted: {got:?} vs {want:?}"
            );
        }
        assert_unreachable_rows_empty("parallel", &parallel);

        let engine = PollableBoundedHostConvEngine::default();
        let serial = conv2d_transpose_backward_coeff_f64_with_engine_and_deadline(
            &a,
            &kernel,
            stride,
            padding,
            dilation,
            output_size,
            grad_size,
            out_c,
            groups,
            1,
            Some(&engine),
            None,
        )
        .expect("bounded-host f64 backward");
        assert_eq!(
            take_conv_f64_scatter_lane_hits(),
            [0, 2],
            "bounded-host call must take the serial scatter once per group"
        );
        assert!(
            engine.ordinary_calls.load(Ordering::SeqCst) >= 1,
            "bounded-host seam must be what produced col_flat"
        );
        assert_eq!(serial.dim(), expected.dim());
        for (got, want) in serial.iter().zip(expected.iter()) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "serial scatter drifted: {got:?} vs {want:?}"
            );
        }
        assert_unreachable_rows_empty("serial", &serial);
    }
}
