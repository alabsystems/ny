// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Native CUDA / cuBLAS GEMM engine for ny (NVIDIA datacenter GPUs, e.g. GB10).
//!
//! # Why a CUDA backend (the f64 soundness unlock)
//!
//! ny's wgpu/Vulkan GPU CROWN fast-path is **unsound for verdicts** because WGSL
//! has no `f64`: the GPU concretizes in round-to-nearest f32 with no certified
//! `γ_n·S` rounding-error term, so a GPU bound can be tighter than the true range
//! and flip a violated instance to `Verified` (see `ny_propagate::sound_gpu_gate`).
//! The verdict-deciding CROWN is therefore forced onto the slow CPU f64 path.
//!
//! cuBLAS provides **exact IEEE-`f64`** GEMM (`cublasDgemm`). The sound CPU CROWN
//! backward (`aw_f64_with_abssum`) computes `A·W` with f64 accumulation plus an
//! abs-sum `S` whose certified error `γ_n·S` bounds the f64 rounding for **any**
//! summation order. cuBLAS's blocked/pairwise f64 summation has error ≤ `γ_n·S`,
//! so routing those two f64 GEMMs (`A·W` and `|A|·|W|`) through cuBLAS is a
//! **sound** acceleration — the f64 GPU work WGSL cannot do. (Validated against
//! the exact-rational A·W soundness proptest when wired into ny-propagate.)
//!
//! # Coherent unified memory (the GB10 lever)
//!
//! The CROWN backward issues MANY small GEMMs (k≈64) which are transfer-bound, not
//! compute-bound. On Grace-Blackwell the host and GPU share **coherent** memory,
//! so we allocate the operands as **managed/unified** buffers and write/read them
//! host-side with no explicit H2D/D2H copy — ~3.5× faster per call than the
//! copy-based path (measured: 481→137 µs at 512×64×512 f64). `cuMemAllocManaged`
//! is expensive, so the buffers are **cached and reused** (grown to the max size
//! seen) behind the handle's lock; per-call allocation would be slower than copy.
//!
//! This crate is the only `unsafe` FFI surface in the GPU stack (cuBLAS is an
//! `unsafe` C API); `ny-core`/`ny-gpu` keep `#![forbid(unsafe_code)]`.

use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, UnifiedSlice, ValidAsZeroBits};

use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed,
    GpuCrownTrajectoryResult, GpuResnetBatchedDomainRef, NyError, Result,
};

mod ieee_selfcheck;
mod sound_crown;

// Intentionally bypass tracing: sealed VNN-COMP canaries run at WARN and a
// hard-killed process must retain the unmatched start line that proves which
// concrete CUDA wrapper it entered. NY's production routing reaches these
// wrappers only through the explicitly enabled CUDA-wide route.
const CUDA_WIDE_ENGAGEMENT_MARKER: &str = "NY_CUDA_WIDE_ENGAGEMENT_V1";
const CUDA_WIDE_ERROR_MARKER: &str = "NY_CUDA_WIDE_ERROR_V1";
const CUDA_WIDE_ERROR_DETAIL_MAX_BYTES: usize = 256;
static CUDA_WIDE_ENGAGEMENT_CALL_ID: AtomicU64 = AtomicU64::new(1);

/// Print-only telemetry for the default-dark ATS DGEMM transaction. Counters
/// are updated after a transaction result has been fixed and emitted at four
/// process-wide milestones at most; they never feed routing, bounds, deadlines,
/// or memory planning.
const CUDA_DGEMM_TRIPLET_MARKER: &str = "NY_CUDA_DGEMM_TRIPLET_V1";
const CUDA_DGEMM_TRIPLET_REPORT_AT: [u64; 4] = [1, 64, 4_096, 262_144];
const CUDA_DGEMM_DRAIN_BIND_ATTEMPTS: usize = 3;
static CUDA_DGEMM_TRIPLET_TRANSACTIONS: AtomicU64 = AtomicU64::new(0);
static CUDA_DGEMM_TRIPLET_CALLS: AtomicU64 = AtomicU64::new(0);
static CUDA_DGEMM_TRIPLET_SYNCS: AtomicU64 = AtomicU64::new(0);
static CUDA_DGEMM_TRIPLET_ERRORS: AtomicU64 = AtomicU64::new(0);
static CUDA_DGEMM_TRIPLET_WALL_US: AtomicU64 = AtomicU64::new(0);
static CUDA_DGEMM_TRIPLET_REPORTS: AtomicU64 = AtomicU64::new(0);

fn cuda_wide_engagement_line(
    phase: &'static str,
    call_id: u64,
    op: &'static str,
    domains: usize,
    specs_per_domain: usize,
    status: &'static str,
) -> String {
    let specs_total = domains.saturating_mul(specs_per_domain);
    format!(
        "{CUDA_WIDE_ENGAGEMENT_MARKER} phase={phase} call_id={call_id} op={op} \
         domains={domains} specs_per_domain={specs_per_domain} specs_total={specs_total} \
         status={status}"
    )
}

fn cuda_wide_error_reason_code(error: &NyError) -> &'static str {
    match error {
        NyError::UnsupportedOp(reason)
            if reason.starts_with("cuda wide resnet: retained/static estimate") =>
        {
            "cap_below_fixed"
        }
        NyError::UnsupportedOp(reason)
            if reason.starts_with("cuda wide resnet: one domain exceeds") =>
        {
            "cap_below_one_domain"
        }
        NyError::InvalidSpec(reason) if reason.starts_with("NY_CUDA_WIDE_MAX_BYTES") => {
            "invalid_cap"
        }
        NyError::UnsupportedOp(_) => "unsupported_op",
        NyError::InvalidSpec(_) => "invalid_spec",
        NyError::GpuMemoryExceeded { .. } => "gpu_memory_exceeded",
        NyError::CpuMemoryExceeded { .. } => "cpu_memory_exceeded",
        NyError::NumericalInstability(_) => "numerical_instability",
        NyError::DeadlineExceeded(_) => "deadline_exceeded",
        NyError::InternalError(_) => "internal_error",
        _ => "other",
    }
}

/// Encode a bounded prefix of an error as lowercase hex. The marker therefore
/// remains one physical ASCII line even when a backend error contains control
/// characters, whitespace, quotes, or arbitrary UTF-8.
fn cuda_wide_error_detail_hex(detail: &str) -> (String, bool) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = detail.as_bytes();
    let used = bytes.len().min(CUDA_WIDE_ERROR_DETAIL_MAX_BYTES);
    let mut encoded = String::with_capacity(used * 2);
    for &byte in &bytes[..used] {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    (encoded, bytes.len() > used)
}

fn cuda_wide_error_line(call_id: u64, op: &'static str, error: &NyError) -> String {
    let detail = error.to_string();
    let (detail_hex, truncated) = cuda_wide_error_detail_hex(&detail);
    let reason_code = cuda_wide_error_reason_code(error);
    format!(
        "{CUDA_WIDE_ERROR_MARKER} call_id={call_id} op={op} reason_code={reason_code} \
         detail_hex={detail_hex} detail_truncated={}",
        u8::from(truncated)
    )
}

fn cuda_wide_engagement_start(op: &'static str, domains: usize, specs_per_domain: usize) -> u64 {
    let call_id = CUDA_WIDE_ENGAGEMENT_CALL_ID.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "{}",
        cuda_wide_engagement_line("start", call_id, op, domains, specs_per_domain, "started")
    );
    call_id
}

fn cuda_wide_engagement_finish<T>(
    call_id: u64,
    op: &'static str,
    domains: usize,
    specs_per_domain: usize,
    result: &Result<T>,
) {
    let status = if result.is_ok() { "ok" } else { "err" };
    eprintln!(
        "{}",
        cuda_wide_engagement_line("finish", call_id, op, domains, specs_per_domain, status)
    );
    if let Err(error) = result {
        eprintln!("{}", cuda_wide_error_line(call_id, op, error));
    }
}

fn cuda_err<E: std::fmt::Debug>(e: E) -> NyError {
    NyError::InternalError(format!("cuda/cublas: {e:?}"))
}

/// cuBLAS environment overrides that can silently replace IEEE `f64`/`f32` GEMM
/// arithmetic with reduced-precision emulation (Ozaki-scheme fixed-point DGEMM,
/// BF16x9 SGEMM). libcublasLt honors these regardless of the per-handle math
/// mode, and the sound seams' certified `γ_n·S` error terms assume IEEE
/// round-to-nearest semantics (order-independence is the ONLY liberty the
/// certificate grants). An environment requesting emulation therefore must not
/// get this engine: construction fails and callers fall back to the
/// proven-sound CPU f64 path.
const CUBLAS_EMULATION_ENV_VARS: [&str; 4] = [
    "CUBLAS_EMULATE_DOUBLE_PRECISION",
    "CUBLAS_EMULATE_SINGLE_PRECISION",
    "CUBLAS_EMULATION_STRATEGY",
    "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
];

/// First emulation-requesting env override, if any. Empty and `"0"` values are
/// treated as "explicitly disabled" and allowed; anything else (including
/// strategy names like `eager`) conservatively blocks engine construction.
/// Env access is injected so the policy is unit-testable without mutating
/// process state.
fn blocked_emulation_override(
    get_env: impl Fn(&str) -> Option<String>,
) -> Option<(&'static str, String)> {
    CUBLAS_EMULATION_ENV_VARS.iter().find_map(|&key| {
        get_env(key).and_then(|value| {
            let explicitly_disabled = value.is_empty() || value == "0";
            (!explicitly_disabled).then_some((key, value))
        })
    })
}

/// Pin the handle to `CUBLAS_DEFAULT_MATH` and read it back. cudarc never sets
/// a math mode, and cuBLAS defaults may differ by version/environment (TF32,
/// BF16x9, FP64 fixed-point emulation on Blackwell); the sound seams require
/// plain IEEE arithmetic, so we assert rather than assume.
fn pin_default_math_mode(blas: &CudaBlas) -> Result<()> {
    use cudarc::cublas::sys;
    // SAFETY: the handle is valid for `blas`'s lifetime and exclusively ours
    // during construction; set/get math mode are plain attribute accessors.
    let status =
        unsafe { sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_DEFAULT_MATH) };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(NyError::InternalError(format!(
            "cuda: cublasSetMathMode(CUBLAS_DEFAULT_MATH) failed: {status:?}"
        )));
    }
    // Read back through a raw u32, not the Rust enum: cuBLAS math modes are
    // OR-able flag bits, and writing a combination into a `#[repr(u32)]` enum
    // out-pointer would be an invalid discriminant. CUBLAS_DEFAULT_MATH == 0.
    let mut mode_bits: u32 = u32::MAX;
    // SAFETY: as above; the pointer is a valid u32 out-slot for the call and
    // cublasMath_t is #[repr(u32)], so the layouts match.
    let status = unsafe { sys::cublasGetMathMode(*blas.handle(), (&raw mut mode_bits).cast()) };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || mode_bits != 0 {
        return Err(NyError::InternalError(format!(
            "cuda: math mode not pinned to CUBLAS_DEFAULT_MATH (status {status:?}, mode bits {mode_bits:#x})"
        )));
    }
    Ok(())
}

/// cuBLAS handle + stream + cached unified scratch buffers (one set per element
/// type). Guarded by a single `Mutex` on the engine; GEMM dispatches serialize on
/// the host while the GPU runs each one ~3.5× faster than the copy path.
struct Inner {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    fa64: Option<UnifiedSlice<f64>>,
    fb64: Option<UnifiedSlice<f64>>,
    fc64: Option<UnifiedSlice<f64>>,
    fa32: Option<UnifiedSlice<f32>>,
    fb32: Option<UnifiedSlice<f32>>,
    fc32: Option<UnifiedSlice<f32>>,
}

/// A [`GemmEngine`] backed by CUDA + cuBLAS on a single NVIDIA device, using
/// cached coherent unified-memory buffers (no H2D/D2H copy on the GB10).
/// Construct once; share via `&dyn GemmEngine`.
pub struct CudaGemmEngine {
    ctx: Arc<CudaContext>,
    inner: Mutex<Inner>,
    device_name: String,
    /// ATS / pageable-memory-access capability (`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`).
    /// When `true` (Grace-class coherent systems, e.g. GB10), cuBLAS can read/write
    /// plain host pointers directly, so `gemm_f32`/`gemm_f64` skip the managed-buffer
    /// copy + readback (measured ~2.2×/call). When `false`, host pointers to cuBLAS
    /// are UNDEFINED, so the unified-buffer path is mandatory — this flag is the
    /// soundness linchpin of the zero-copy fast path.
    host_ptr_ok: bool,
}

/// Grow `buf` to at least `len` managed elements (reallocating only when too
/// small), then return a mutable handle. Managed memory is coherent on the GB10.
fn ensure_unified<T: DeviceRepr + ValidAsZeroBits + Unpin>(
    ctx: &Arc<CudaContext>,
    buf: &mut Option<UnifiedSlice<T>>,
    len: usize,
) -> Result<()> {
    let grow = buf.as_ref().map_or(true, |b| b.len() < len);
    if grow {
        // SAFETY: T is f32/f64 (any bit pattern valid); written before use.
        *buf = Some(unsafe { ctx.alloc_unified::<T>(len, true) }.map_err(cuda_err)?);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedDgemmShape {
    m: i32,
    k: i32,
    n: i32,
    output_len: usize,
    output_bytes: usize,
}

/// Validate every integer that reaches cuBLAS and every slice/product used by
/// the ATS triplet before allocating an output or launching work. In particular,
/// `usize -> i32` is never a truncating cast on this path.
fn validate_dgemm_triplet(
    m: usize,
    k: usize,
    n: usize,
    a: [&[f64]; 3],
    b: [&[f64]; 3],
) -> Result<CheckedDgemmShape> {
    let m_i32 = i32::try_from(m)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm triplet: m exceeds i32".into()))?;
    let k_i32 = i32::try_from(k)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm triplet: k exceeds i32".into()))?;
    let n_i32 = i32::try_from(n)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm triplet: n exceeds i32".into()))?;
    let lhs_len = m
        .checked_mul(k)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm triplet: m*k overflow".into()))?;
    let rhs_len = k
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm triplet: k*n overflow".into()))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm triplet: m*n overflow".into()))?;
    let output_bytes = output_len
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm triplet: output-byte overflow".into()))?;
    // Wide-CROWN already charges all three simultaneous output arrays in
    // CUDA_WIDE_BYTES_PER_ROW_CELL, so its 512 MiB default plan continues to
    // bind wide callers. Non-wide callers retain the legacy sequence's existing
    // three-output peak. This is only a representability check, not a universal
    // 512 MiB cap for arbitrary GemmEngine callers.
    output_bytes.checked_mul(3).ok_or_else(|| {
        NyError::InvalidSpec("cuda dgemm triplet: output-triplet overflow".into())
    })?;

    for input in a {
        if input.len() != lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![lhs_len],
                got: vec![input.len()],
            });
        }
    }
    for input in b {
        if input.len() != rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![rhs_len],
                got: vec![input.len()],
            });
        }
    }

    Ok(CheckedDgemmShape {
        m: m_i32,
        k: k_i32,
        n: n_i32,
        output_len,
        output_bytes,
    })
}

/// Validate the two left operands and shared RHS used by the ATS pair
/// transaction before allocating output or handing any host pointer to CUDA.
fn validate_dgemm_pair_shared_rhs(
    m: usize,
    k: usize,
    n: usize,
    a: [&[f64]; 2],
    b: &[f64],
) -> Result<CheckedDgemmShape> {
    let m_i32 = i32::try_from(m)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm pair: m exceeds i32".into()))?;
    let k_i32 = i32::try_from(k)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm pair: k exceeds i32".into()))?;
    let n_i32 = i32::try_from(n)
        .map_err(|_| NyError::InvalidSpec("cuda dgemm pair: n exceeds i32".into()))?;
    let lhs_len = m
        .checked_mul(k)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: m*k overflow".into()))?;
    let rhs_len = k
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: k*n overflow".into()))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: m*n overflow".into()))?;
    let output_bytes = output_len
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: output-byte overflow".into()))?;
    output_bytes
        .checked_mul(2)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: output-pair overflow".into()))?;

    for input in a {
        if input.len() != lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![lhs_len],
                got: vec![input.len()],
            });
        }
    }
    if b.len() != rhs_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![rhs_len],
            got: vec![b.len()],
        });
    }

    Ok(CheckedDgemmShape {
        m: m_i32,
        k: k_i32,
        n: n_i32,
        output_len,
        output_bytes,
    })
}

/// Only exact, valid Unicode `1` engages the transaction. Unset, exact `0`,
/// padded/truthy/malformed Unicode, and present non-Unicode values all preserve
/// the three-call legacy path.
fn cuda_dgemm_triplet_enabled(raw: Option<&std::ffi::OsStr>) -> bool {
    raw.and_then(std::ffi::OsStr::to_str) == Some("1")
}

fn cuda_dgemm_triplet_line(
    transactions: u64,
    calls: u64,
    syncs: u64,
    errors: u64,
    wall_us: u64,
) -> String {
    format!(
        "{CUDA_DGEMM_TRIPLET_MARKER} transactions={transactions} calls={calls} \
         syncs={syncs} errors={errors} wall_us={wall_us}"
    )
}

fn cuda_dgemm_triplet_should_report(transactions: u64) -> bool {
    CUDA_DGEMM_TRIPLET_REPORT_AT.contains(&transactions)
}

fn record_cuda_dgemm_triplet(calls: usize, wall_us: u128, failed: bool) {
    let transactions = CUDA_DGEMM_TRIPLET_TRANSACTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    let calls = CUDA_DGEMM_TRIPLET_CALLS.fetch_add(calls as u64, Ordering::Relaxed) + calls as u64;
    let syncs = CUDA_DGEMM_TRIPLET_SYNCS.fetch_add(1, Ordering::Relaxed) + 1;
    let errors = CUDA_DGEMM_TRIPLET_ERRORS.fetch_add(u64::from(failed), Ordering::Relaxed)
        + u64::from(failed);
    let wall_us = u64::try_from(wall_us).unwrap_or(u64::MAX);
    let wall_us = CUDA_DGEMM_TRIPLET_WALL_US.fetch_add(wall_us, Ordering::Relaxed) + wall_us;

    if cuda_dgemm_triplet_should_report(transactions)
        && CUDA_DGEMM_TRIPLET_REPORTS
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |reports| {
                (reports < CUDA_DGEMM_TRIPLET_REPORT_AT.len() as u64).then_some(reports + 1)
            })
            .is_ok()
    {
        use std::io::Write as _;

        let line = cuda_dgemm_triplet_line(transactions, calls, syncs, errors, wall_us);
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProvenCudaDrain {
    bind_attempts: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnprovenCudaDrain {
    Bind { attempts: usize },
    Synchronize { bind_attempts: usize },
}

/// Bind the stream context and then invoke the raw CUDA synchronization call.
/// `CudaStream::synchronize` combines these steps, which makes an early bind
/// failure indistinguishable from a `cuStreamSynchronize` failure. The first
/// bind attempt can consume a deferred cudarc drop error, so retry a bounded
/// number of times before declaring quiescence unproven. Synchronization itself
/// is never retried: any CUDA error from that call leaves pointer lifetime safety
/// unproven.
fn prove_cuda_stream_quiescent(
    mut bind: impl FnMut() -> bool,
    mut synchronize: impl FnMut() -> bool,
) -> std::result::Result<ProvenCudaDrain, UnprovenCudaDrain> {
    for attempt in 1..=CUDA_DGEMM_DRAIN_BIND_ATTEMPTS {
        if !bind() {
            continue;
        }
        return if synchronize() {
            Ok(ProvenCudaDrain {
                bind_attempts: attempt,
            })
        } else {
            Err(UnprovenCudaDrain::Synchronize {
                bind_attempts: attempt,
            })
        };
    }
    Err(UnprovenCudaDrain::Bind {
        attempts: CUDA_DGEMM_DRAIN_BIND_ATTEMPTS,
    })
}

struct QueuedDgemms {
    launch_result: Result<()>,
    calls: usize,
    drain: std::result::Result<ProvenCudaDrain, UnprovenCudaDrain>,
}

/// Queue `count` launches and establish stream quiescence afterwards, including
/// when a later launch fails. The launch result may be returned only when
/// `drain` is `Ok`; the call count includes the failing launch attempt.
fn queue_dgemms_and_drain(
    count: usize,
    mut launch: impl FnMut(usize) -> Result<()>,
    bind: impl FnMut() -> bool,
    synchronize: impl FnMut() -> bool,
) -> QueuedDgemms {
    let mut calls = 0usize;
    let mut launch_result = Ok(());
    for index in 0..count {
        calls += 1;
        if let Err(error) = launch(index) {
            launch_result = Err(error);
            break;
        }
    }
    QueuedDgemms {
        launch_result,
        calls,
        drain: prove_cuda_stream_quiescent(bind, synchronize),
    }
}

fn queue_triplet_and_drain(
    launch: impl FnMut(usize) -> Result<()>,
    bind: impl FnMut() -> bool,
    synchronize: impl FnMut() -> bool,
) -> QueuedDgemms {
    queue_dgemms_and_drain(3, launch, bind, synchronize)
}

fn queue_pair_and_drain(
    launch: impl FnMut(usize) -> Result<()>,
    bind: impl FnMut() -> bool,
    synchronize: impl FnMut() -> bool,
) -> QueuedDgemms {
    queue_dgemms_and_drain(2, launch, bind, synchronize)
}

/// There is no safe Rust value to return when CUDA might still hold borrowed
/// host pointers. Aborting is deliberate: `process::abort` cannot unwind and
/// therefore cannot run destructors for the caller's inputs or our outputs.
/// Keep this function free of formatting, allocation, logging, and panicking
/// operations before the abort.
#[cold]
fn abort_unproven_cuda_quiescence(_failure: UnprovenCudaDrain) -> ! {
    std::process::abort()
}

fn allocate_dgemm_output(shape: CheckedDgemmShape, site: &'static str) -> Result<Vec<f64>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.output_len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: shape.output_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    output.resize(shape.output_len, 0.0);
    Ok(output)
}

impl CudaGemmEngine {
    /// Initialize CUDA device 0 and a cuBLAS handle on its default stream.
    /// Returns an error if no CUDA device/driver is available.
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Initialize a specific CUDA device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        // cudarc's dynamic loading PANICS (not Err) when the shared library is
        // absent, so probe for libcuda/libcublas first — on CUDA-less hosts the
        // engine must decline with Err so callers fall back to the sound CPU
        // path instead of dying mid-verification.
        // SAFETY: is_culib_present only attempts a dlopen and reports success.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(NyError::InternalError(
                "cuda: libcuda not present on this host; engine unavailable".to_string(),
            ));
        }
        // SAFETY: as above, for libcublas.
        if !unsafe { cudarc::cublas::sys::is_culib_present() } {
            return Err(NyError::InternalError(
                "cuda: libcublas not present on this host; engine unavailable".to_string(),
            ));
        }
        if let Some((key, value)) = blocked_emulation_override(|k| std::env::var(k).ok()) {
            tracing::warn!(
                "cuda: {key}={value} requests cuBLAS precision emulation, which would \
                 invalidate the certified IEEE rounding-error bounds; refusing the CUDA \
                 engine (the sound CPU f64 path is used instead). Unset {key} to re-enable."
            );
            return Err(NyError::InternalError(format!(
                "cuda: emulation env override {key}={value} present; engine refused"
            )));
        }
        let ctx = CudaContext::new(ordinal).map_err(cuda_err)?;
        let device_name = ctx.name().unwrap_or_else(|_| "unknown-cuda".to_string());
        // ATS / pageable-memory-access probe (#p2-ats-zero-copy). On a device that
        // reports CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS != 0 (Grace-class
        // coherent, e.g. GB10) cuBLAS may read/write plain host pointers, so the
        // GEMM can skip the managed-buffer copy + readback. On any device that
        // reports 0 — or if the query fails — host pointers to cuBLAS are UNDEFINED,
        // so we fail closed to `false` and keep the unified-buffer path. Without
        // this guard a non-faulting garbage read on a non-ATS GPU could silently
        // corrupt A·W (a false-VERIFIED), so the probe is soundness-critical.
        let host_ptr_ok = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            )
            .map(|v| v != 0)
            .unwrap_or(false);
        // NOTE: CU_CTX_SCHED_BLOCKING_SYNC (via ctx.set_blocking_synchronize) was
        // measured and REJECTED — it is ~3% SLOWER on the single-threaded
        // α/β-CROWN path (mnist_concat --method alpha, 103% CPU): freeing the
        // synchronize()-spinning core gives no benefit when there is no
        // concurrent CPU work to overlap, while the semaphore wakeup latency adds
        // per-GEMM overhead. The default CU_CTX_SCHED_AUTO spin wins there; the
        // one multi-threaded case (cifar BaB) is BaB-bound and times out
        // regardless, so there is no demonstrated win. Keep the default sync.
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err)?;
        pin_default_math_mode(&blas)?;
        let engine = Self {
            ctx,
            inner: Mutex::new(Inner {
                blas,
                stream,
                fa64: None,
                fb64: None,
                fc64: None,
                fa32: None,
                fb32: None,
                fc32: None,
            }),
            device_name,
            host_ptr_ok,
        };
        // Known-answer IEEE bit-exactness probes (docs/F32_ABSSUM_SEAM.md §5,
        // `ieee_selfcheck`): the env blocklist + math-mode pin above only assert
        // what cuBLAS was ASKED to do; this measures what it DID, on the real
        // dispatch path this engine will use. If TF32/BF16x9/fixed-point
        // emulation ever leaks into Sgemm/Dgemm, every certified IEEE
        // rounding-error term is unsound, so ANY bit deviation refuses the
        // engine (callers fall back to the proven-sound CPU f64 path).
        if let Err(e) = engine.assert_ieee_bit_exact() {
            tracing::warn!(
                "cuda: device {ordinal} ({}) failed the IEEE known-answer GEMM probe: {e}; \
                 refusing the CUDA engine (the sound CPU f64 path is used instead)",
                engine.device_name
            );
            return Err(e);
        }
        tracing::info!(
            "CudaGemmEngine: initialized on device {ordinal} ({}), \
             {} memory, CUBLAS_DEFAULT_MATH pinned, IEEE known-answer probes bit-exact",
            engine.device_name,
            if host_ptr_ok {
                "ATS host-pointer zero-copy"
            } else {
                "unified"
            }
        );
        Ok(engine)
    }

    /// The CUDA device name (e.g. "NVIDIA GB10").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether this engine uses the ATS host-pointer zero-copy GEMM fast path
    /// (`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS != 0`). Diagnostic accessor;
    /// `false` means the (bit-identical) unified-buffer path is used instead.
    #[must_use]
    pub fn host_ptr_zero_copy(&self) -> bool {
        self.host_ptr_ok
    }

    fn gemm_f64_triplet_legacy(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        Ok([
            self.gemm_f64(m, k, n, a[0], b[0])?,
            self.gemm_f64(m, k, n, a[1], b[1])?,
            self.gemm_f64(m, k, n, a[2], b[2])?,
        ])
    }

    fn gemm_f64_pair_shared_rhs_legacy(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        Ok([
            self.gemm_f64(m, k, n, a[0], b)?,
            self.gemm_f64(m, k, n, a[1], b)?,
        ])
    }

    /// ATS-only shared-RHS pair. Both products preserve their individual GEMM
    /// shapes and reduction axes; only the host scheduling changes to one lock
    /// and one mandatory stream drain.
    fn gemm_f64_pair_shared_rhs_ats(
        &self,
        shape: CheckedDgemmShape,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        if !self.host_ptr_ok {
            return Err(NyError::UnsupportedOp(
                "cuda dgemm pair: ATS host-pointer access unavailable".into(),
            ));
        }
        if shape.m == 0 || shape.k == 0 || shape.n == 0 {
            return Ok([
                allocate_dgemm_output(shape, "cuda::gemm_f64_pair_shared_rhs/output")?,
                allocate_dgemm_output(shape, "cuda::gemm_f64_pair_shared_rhs/output")?,
            ]);
        }

        let mut output = [
            allocate_dgemm_output(shape, "cuda::gemm_f64_pair_shared_rhs/output")?,
            allocate_dgemm_output(shape, "cuda::gemm_f64_pair_shared_rhs/output")?,
        ];
        let alpha = 1.0f64;
        let beta = 0.0f64;
        let guard = self.inner.lock().expect("cublas mutex poisoned");
        // A bind failure before the first launch is safe to return. Once a
        // borrowed host pointer reaches CUDA, all exits pass through the same
        // explicit quiescence proof as the triplet transaction.
        guard.stream.context().bind_to_thread().map_err(cuda_err)?;
        let transaction = queue_pair_and_drain(
            |index| {
                // SAFETY: `validate_dgemm_pair_shared_rhs` checked both A slices,
                // the shared B slice, every product, and lossless i32 dimensions.
                // ATS admission permits pageable host pointers. The ordered
                // stream, inputs, and outputs remain live through the mandatory
                // drain; row-major C=A*B uses the standard C^T=B^T*A^T swap.
                unsafe {
                    cudarc::cublas::result::dgemm(
                        *guard.blas.handle(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        shape.n,
                        shape.m,
                        shape.k,
                        &raw const alpha,
                        b.as_ptr(),
                        shape.n,
                        a[index].as_ptr(),
                        shape.k,
                        &raw const beta,
                        output[index].as_mut_ptr(),
                        shape.n,
                    )
                    .map_err(cuda_err)
                }
            },
            || guard.stream.context().bind_to_thread().is_ok(),
            || {
                // SAFETY: the live guarded stream is bound immediately before
                // this raw synchronization attempt.
                unsafe {
                    cudarc::driver::result::stream::synchronize(guard.stream.cu_stream()).is_ok()
                }
            },
        );
        let result = match transaction.drain {
            Ok(_proof) => transaction.launch_result,
            Err(failure) => abort_unproven_cuda_quiescence(failure),
        };
        drop(guard);
        result?;
        Ok(output)
    }

    /// ATS-only implementation: the caller's six already-live input slices are
    /// passed straight to cuBLAS (no packing), and the three output vectors are
    /// the same vectors that the legacy sound-CROWN sequence retains
    /// simultaneously after its third call. The 512 MiB wide plan already
    /// accounts for this peak and continues to bind wide callers; non-wide
    /// callers preserve their preexisting peak and do not acquire a universal
    /// 512 MiB cap here.
    fn gemm_f64_triplet_ats(
        &self,
        shape: CheckedDgemmShape,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        if !self.host_ptr_ok {
            return Err(NyError::UnsupportedOp(
                "cuda dgemm triplet: ATS host-pointer access unavailable".into(),
            ));
        }
        let started = Instant::now();
        if shape.m == 0 || shape.k == 0 || shape.n == 0 {
            return Ok([
                allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
                allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
                allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
            ]);
        }

        let mut output = [
            allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
            allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
            allocate_dgemm_output(shape, "cuda::gemm_f64_triplet/output")?,
        ];
        let alpha = 1.0f64;
        let beta = 0.0f64;
        let guard = self.inner.lock().expect("cublas mutex poisoned");
        // A bind failure here is safe to return because no borrowed pointer has
        // reached CUDA yet. After the first launch attempt, every exit must pass
        // through the explicit quiescence proof below.
        guard.stream.context().bind_to_thread().map_err(cuda_err)?;
        let transaction = queue_triplet_and_drain(
            |index| {
                // SAFETY: `validate_dgemm_triplet` checked all six slice lengths,
                // every product, and lossless i32 dimensions. `host_ptr_ok` is
                // checked by the public override before entering this method, so
                // cuBLAS may access these pageable host pointers. The engine owns
                // one ordered stream/handle under `guard`; inputs and outputs stay
                // live through the mandatory drain below. Row-major C=A*B is the
                // same column-major C^T=B^T*A^T swap used by `gemm_f64`.
                unsafe {
                    cudarc::cublas::result::dgemm(
                        *guard.blas.handle(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        shape.n,
                        shape.m,
                        shape.k,
                        &raw const alpha,
                        b[index].as_ptr(),
                        shape.n,
                        a[index].as_ptr(),
                        shape.k,
                        &raw const beta,
                        output[index].as_mut_ptr(),
                        shape.n,
                    )
                    .map_err(cuda_err)
                }
            },
            || guard.stream.context().bind_to_thread().is_ok(),
            || {
                // SAFETY: `guard.stream` is live and owns this CUstream. Binding
                // was established immediately before this call; separating the
                // raw call is what lets us prove it was actually attempted.
                unsafe {
                    cudarc::driver::result::stream::synchronize(guard.stream.cu_stream()).is_ok()
                }
            },
        );
        let result = match transaction.drain {
            Ok(_proof) => transaction.launch_result,
            Err(failure) => abort_unproven_cuda_quiescence(failure),
        };
        drop(guard);
        let wall_us = started.elapsed().as_micros();
        record_cuda_dgemm_triplet(transaction.calls, wall_us, result.is_err());
        result?;
        Ok(output)
    }
}

/// cuBLAS column-major config for a row-major `C(m×n) = A(m×k)·B(k×n)` computed
/// as the column-major `Cᵀ = Bᵀ·Aᵀ` (pass B,A swapped, op_N, ld = n,k,n).
fn gemm_cfg<T: Copy>(m: usize, k: usize, n: usize, one: T, zero: T) -> GemmConfig<T> {
    GemmConfig::<T> {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: one,
        lda: n as i32,
        ldb: k as i32,
        beta: zero,
        ldc: n as i32,
    }
}

// Generate `gemm_f32` / `gemm_f64` over the type-specific cached buffer fields.
macro_rules! impl_cached_gemm {
    ($name:ident, $T:ty, $fa:ident, $fb:ident, $fc:ident, $one:expr, $zero:expr, $raw:ident) => {
        fn $name(&self, m: usize, k: usize, n: usize, a: &[$T], b: &[$T]) -> Result<Vec<$T>> {
            if a.len() != m * k {
                return Err(NyError::ShapeMismatch {
                    expected: vec![m * k],
                    got: vec![a.len()],
                });
            }
            if b.len() != k * n {
                return Err(NyError::ShapeMismatch {
                    expected: vec![k * n],
                    got: vec![b.len()],
                });
            }
            if m == 0 || k == 0 || n == 0 {
                return Ok(vec![$zero; m * n]);
            }
            // ATS zero-copy fast path (#p2-ats-zero-copy): on a pageable-memory-access
            // device (host_ptr_ok) cuBLAS reads/writes the host slices directly — no
            // managed buffer, no H2D/D2H copy, no readback (measured ~2.2×/call).
            // BIT-IDENTICAL to the unified path below: same `cublas?gemm`, same
            // column-major swap (row-major `C = A·B` as `Cᵀ = Bᵀ·Aᵀ`, so pass B,A
            // swapped with op_N, ld = n,k,n), same pinned CUBLAS_DEFAULT_MATH mode.
            // cuBLAS selects its blocked/pairwise reduction by shape + math mode, never
            // by pointer residence, so the f64/f32 result is unchanged — and even if it
            // differed, the sound A·W bound is Higham order-independent.
            if self.host_ptr_ok {
                let alpha: $T = $one;
                let beta: $T = $zero;
                let mut c = vec![$zero; m * n];
                let g = self.inner.lock().expect("cublas mutex poisoned");
                // SAFETY: a/b/c are host slices of exactly m*k, k*n, m*n elements; the
                // device reports CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS != 0
                // (host_ptr_ok), so cuBLAS may access this pageable host memory; leading
                // dims match the column-major swap; the slices + guard keep the pointers
                // valid for the whole call; a/b are read-only, c write-only.
                unsafe {
                    cudarc::cublas::result::$raw(
                        *g.blas.handle(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        &raw const alpha,
                        b.as_ptr(),
                        n as i32,
                        a.as_ptr(),
                        k as i32,
                        &raw const beta,
                        c.as_mut_ptr(),
                        n as i32,
                    )
                    .map_err(cuda_err)?;
                }
                g.stream.synchronize().map_err(cuda_err)?;
                return Ok(c);
            }
            let cfg = gemm_cfg::<$T>(m, k, n, $one, $zero);
            let mut g = self.inner.lock().expect("cublas mutex poisoned");
            ensure_unified(&self.ctx, &mut g.$fa, m * k)?;
            ensure_unified(&self.ctx, &mut g.$fb, k * n)?;
            ensure_unified(&self.ctx, &mut g.$fc, m * n)?;
            let Inner {
                blas,
                stream,
                $fa,
                $fb,
                $fc,
                ..
            } = &mut *g;
            let ua = $fa.as_mut().expect("ensured");
            ua.as_mut_slice().map_err(cuda_err)?[..m * k].copy_from_slice(a);
            let ub = $fb.as_mut().expect("ensured");
            ub.as_mut_slice().map_err(cuda_err)?[..k * n].copy_from_slice(b);
            let uc = $fc.as_mut().expect("ensured");
            // SAFETY: shapes/leading-dims validated; views are exactly m*k, k*n,
            // m*n; cuBLAS reads/writes strictly within them.
            unsafe {
                blas.gemm(
                    cfg,
                    &$fb.as_ref().unwrap().slice(..k * n),
                    &$fa.as_ref().unwrap().slice(..m * k),
                    &mut uc.slice_mut(..m * n),
                )
                .map_err(cuda_err)?;
            }
            stream.synchronize().map_err(cuda_err)?;
            Ok($fc.as_ref().unwrap().as_slice().map_err(cuda_err)?[..m * n].to_vec())
        }
    };
}

impl GemmEngine for CudaGemmEngine {
    impl_cached_gemm!(gemm_f32, f32, fa32, fb32, fc32, 1.0f32, 0.0f32, sgemm);
    impl_cached_gemm!(gemm_f64, f64, fa64, fb64, fc64, 1.0f64, 0.0f64, dgemm);

    fn gemm_f64_pair_shared_rhs(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        let shape = validate_dgemm_pair_shared_rhs(m, k, n, a, b)?;
        if !self.host_ptr_ok {
            return self.gemm_f64_pair_shared_rhs_legacy(m, k, n, a, b);
        }
        self.gemm_f64_pair_shared_rhs_ats(shape, a, b)
    }

    fn gemm_f64_triplet(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        let shape = validate_dgemm_triplet(m, k, n, a, b)?;
        let enabled =
            cuda_dgemm_triplet_enabled(std::env::var_os("NY_CUDA_DGEMM_TRIPLET").as_deref());
        if !enabled || !self.host_ptr_ok {
            return self.gemm_f64_triplet_legacy(m, k, n, a, b);
        }
        self.gemm_f64_triplet_ats(shape, a, b)
    }

    /// Tensor-core f32 GEMM via `cublasGemmEx` with
    /// `CUBLAS_COMPUTE_32F_FAST_16BF` (BF16 inputs on tensor cores, f32
    /// accumulate). The compute type is per-CALL: the handle's pinned
    /// `CUBLAS_DEFAULT_MATH` and every `gemm_f32`/`gemm_f64` caller keep exact
    /// IEEE semantics. Per the trait contract this is ONLY for soundness-free
    /// consumers (attack / counterexample search — candidates are re-checked
    /// concretely); reduced precision here can never decide a verdict.
    fn gemm_f32_fast(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        use cudarc::cublas::sys;
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        if a.len() != m * k {
            return Err(NyError::ShapeMismatch {
                expected: vec![m * k],
                got: vec![a.len()],
            });
        }
        if b.len() != k * n {
            return Err(NyError::ShapeMismatch {
                expected: vec![k * n],
                got: vec![b.len()],
            });
        }
        if m == 0 || k == 0 || n == 0 {
            return Ok(vec![0.0f32; m * n]);
        }
        // Row-major C = A·B as column-major Cᵀ = Bᵀ·Aᵀ (same swap as gemm_cfg).
        let mut g = self.inner.lock().expect("cublas mutex poisoned");
        ensure_unified(&self.ctx, &mut g.fa32, m * k)?;
        ensure_unified(&self.ctx, &mut g.fb32, k * n)?;
        ensure_unified(&self.ctx, &mut g.fc32, m * n)?;
        let Inner {
            blas,
            stream,
            fa32,
            fb32,
            fc32,
            ..
        } = &mut *g;
        let ua = fa32.as_mut().expect("ensured");
        ua.as_mut_slice().map_err(cuda_err)?[..m * k].copy_from_slice(a);
        let ub = fb32.as_mut().expect("ensured");
        ub.as_mut_slice().map_err(cuda_err)?[..k * n].copy_from_slice(b);
        let uc = fc32.as_mut().expect("ensured");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        {
            let bview = fb32.as_ref().expect("ensured").slice(..k * n);
            let aview = fa32.as_ref().expect("ensured").slice(..m * k);
            let mut cview = uc.slice_mut(..m * n);
            let (b_ptr, _rec_b) = bview.device_ptr(stream);
            let (a_ptr, _rec_a) = aview.device_ptr(stream);
            let (c_ptr, _rec_c) = cview.device_ptr_mut(stream);
            // SAFETY: operand views are exactly (k·n), (m·k), (m·n) elements of
            // live unified allocations; leading dims match the column-major
            // swap; pointers stay valid for the duration of the call (guards
            // held). Reduced-precision compute is the documented contract of
            // this method only.
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *blas.handle(),
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    (&raw const alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n as i32,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k as i32,
                    (&raw const beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(cuda_err)?;
            }
        }
        stream.synchronize().map_err(cuda_err)?;
        Ok(uc.as_slice().map_err(cuda_err)?[..m * n].to_vec())
    }

    /// Expose the f64-exact sound GPU-resident CROWN backward for verdict routing.
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for CudaGemmEngine {
    /// Non-sound contract: return the SOUND f64 bounds (a valid — and tighter than
    /// f32 — enclosure also satisfies the non-sound contract).
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        sound_crown::crown_backward_gpu_sound_impl(
            self,
            layers,
            spec,
            num_specs,
            input_lower,
            input_upper,
        )
    }

    /// SOUND f64-exact GPU-resident CROWN backward (Linear/Activation chains);
    /// `UnsupportedOp` on conv/pool/dual-alpha layers ⇒ caller falls back to the
    /// proven CPU sound path (verified safe at all dispatch sites).
    fn crown_backward_gpu_sound(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        sound_crown::crown_backward_gpu_sound_impl(
            self,
            layers,
            spec,
            num_specs,
            input_lower,
            input_upper,
        )
    }

    /// SOUND f64-exact GPU-resident SEEDED CROWN backward (the alpha-CROWN suffix
    /// counterpart of `crown_backward_gpu_sound`): starts from the alpha-suffix
    /// `seed` frontier instead of a spec. `UnsupportedOp` on conv/pool/dual-alpha
    /// layers or below the size-gate ⇒ caller falls back to the proven CPU sound
    /// suffix. Previously CUDA left this to the trait default (always CPU), so a
    /// cuBLAS engine lost the f64-resident graph-alpha suffix that wgpu already had.
    fn crown_backward_gpu_seeded_sound(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_seeded_sound: empty layer list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_seeded_sound_impl(
            self,
            layers,
            seed,
            input_lower,
            input_upper,
        )
    }

    /// SOUND f64-exact GPU-resident RESNET-decomposed seeded CROWN backward (T1.3):
    /// the cifar100/tinyimagenet ResNet counterpart of `crown_backward_gpu_seeded_sound`.
    /// Propagates the seed frontier across plain chains + identity/projection residual
    /// blocks (`A_in = backward_F(A) [+ backward_P(A) | + A]`), carrying certified
    /// error across block boundaries. `frontier_abs`/`node_abs` (the exploding-net
    /// error-concretization tightening) are accepted for parity but not required for
    /// soundness — the base path is a valid enclosure. `UnsupportedOp` below the
    /// size-gate / on an unsupported layer ⇒ caller keeps the CPU sound suffix.
    /// Previously CUDA left this to the trait default (always CPU), so a cuBLAS engine
    /// bailed to the slow CPU dense path on every residual `Add`.
    fn crown_backward_gpu_resnet_sound(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound: empty segment list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_resnet_sound_impl(
            self,
            segments,
            seed,
            input_lower,
            input_upper,
        )
    }

    /// SOUND f64-exact GPU-resident β-CROWN RESNET seeded backward (T1.3, BaB
    /// per-domain bound): `crown_backward_gpu_resnet_sound` with the per-domain split
    /// dual `beta_signed` folded into each POST-slope coefficient. Sound for ANY β≥0
    /// (valid Lagrangian dual); the fold add is certified outward. Previously CUDA
    /// left this to the trait default (the ~60 s/domain CPU dense backward).
    fn crown_backward_gpu_resnet_sound_beta(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta: empty segment list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_resnet_sound_beta_impl(
            self,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
        )
    }

    /// Independent serial oracle for the wide proof-forest re-fold guard.
    /// Bypass only CUDA's performance size-gate; the implementation runs the
    /// same sound f64 core and all of its structural/numeric validation.
    fn crown_backward_gpu_resnet_sound_beta_refold_oracle(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_refold_oracle: empty segment list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_resnet_sound_beta_refold_oracle_impl(
            self,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
        )
    }

    /// Wide CUDA proof-forest fold used by the multi-objective BaB lane.  Unlike
    /// the trait default, this stacks every child/spec row into the same cuBLAS
    /// matrices while preserving domain-local relaxations, β, and input boxes.
    fn crown_backward_gpu_resnet_sound_beta_batched(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Result<Vec<GpuCrownResult>> {
        const OP: &str = "beta_batched";
        let call_id = cuda_wide_engagement_start(OP, domains.len(), seed.num_specs);
        let result =
            sound_crown::crown_backward_gpu_resnet_sound_beta_batched_impl(self, domains, seed);
        cuda_wide_engagement_finish(call_id, OP, domains.len(), seed.num_specs, &result);
        result
    }

    /// Wide β/α-gradient capture from the same stacked coefficient stream.  These
    /// captures steer dual variables only; the returned bounds use the identical
    /// sound f64 fold as the bound-only entry.
    fn crown_backward_gpu_resnet_sound_beta_batched_grad(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        const OP: &str = "beta_batched_grad";
        let call_id = cuda_wide_engagement_start(OP, domains.len(), seed.num_specs);
        let result = sound_crown::crown_backward_gpu_resnet_sound_beta_batched_grad_impl(
            self,
            domains,
            seed,
            union_gather_idx,
            relu_pre_lower,
        );
        cuda_wide_engagement_finish(call_id, OP, domains.len(), seed.num_specs, &result);
        result
    }

    /// Combined proof-forest trajectory capture.  CUDA widens the already-folded
    /// f64 frontier into f32 center/error intervals, charging every cast delta,
    /// so coefficients do not require a second backward.
    fn crown_backward_gpu_resnet_sound_beta_batched_trajectory(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<GpuCrownTrajectoryResult> {
        const OP: &str = "beta_batched_trajectory";
        let call_id = cuda_wide_engagement_start(OP, domains.len(), seed.num_specs);
        let result = sound_crown::crown_backward_gpu_resnet_sound_beta_batched_trajectory_impl(
            self,
            domains,
            seed,
            union_gather_idx,
            relu_pre_lower,
        );
        cuda_wide_engagement_finish(call_id, OP, domains.len(), seed.num_specs, &result);
        result
    }

    /// GRADIENT-capturing resnet backward (T1.3, warmup): same sound bounds as
    /// `crown_backward_gpu_resnet_sound` + each ReLU's analytic alpha gradient
    /// (fold order). Gradients are non-soundness-critical (they only steer α), so a
    /// gather error can never affect a verdict. Was CPU-only on CUDA.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_grad(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        relu_pre_lower: &[Vec<f32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<ny_core::GpuCrownGradResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_grad: empty segment list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_resnet_sound_grad_impl(
            self,
            segments,
            seed,
            input_lower,
            input_upper,
            relu_pre_lower,
        )
    }

    /// β-GRADIENT resnet backward (T1.3, per-domain β optimization): same sound
    /// β-folded bounds as `crown_backward_gpu_resnet_sound_beta` + the requested
    /// split columns' pre-transform lower A-coefficients gathered (fold order). The
    /// gather is non-soundness-critical. Was CPU-only on CUDA.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_grad(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        beta_gather_idx: &[Vec<u32>],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<ny_core::GpuCrownBetaGradResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_grad: empty segment list".into(),
            ));
        }
        sound_crown::crown_backward_gpu_resnet_sound_beta_grad_impl(
            self,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            beta_gather_idx,
        )
    }

    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_engagement_marker_schema_is_stable() {
        assert_eq!(
            cuda_wide_engagement_line("start", 17, "beta_batched", 4, 3, "started"),
            "NY_CUDA_WIDE_ENGAGEMENT_V1 phase=start call_id=17 op=beta_batched domains=4 \
             specs_per_domain=3 specs_total=12 status=started"
        );
        assert_eq!(
            cuda_wide_engagement_line("finish", 17, "beta_batched", 4, 3, "err"),
            "NY_CUDA_WIDE_ENGAGEMENT_V1 phase=finish call_id=17 op=beta_batched domains=4 \
             specs_per_domain=3 specs_total=12 status=err"
        );
        assert_eq!(
            cuda_wide_engagement_line("finish", 18, "beta_batched_grad", 2, 5, "ok"),
            "NY_CUDA_WIDE_ENGAGEMENT_V1 phase=finish call_id=18 op=beta_batched_grad domains=2 \
             specs_per_domain=5 specs_total=10 status=ok"
        );
    }

    #[test]
    fn wide_error_marker_is_stable_ascii_and_bounded() {
        let error = NyError::UnsupportedOp(
            "cuda wide resnet: one domain exceeds cap\nretry \"quoted\" Ω".into(),
        );
        let line = cuda_wide_error_line(19, "beta_batched_grad", &error);
        assert_eq!(
            line.split_whitespace().take(4).collect::<Vec<_>>(),
            [
                "NY_CUDA_WIDE_ERROR_V1",
                "call_id=19",
                "op=beta_batched_grad",
                "reason_code=cap_below_one_domain",
            ]
        );
        assert_eq!(line.lines().count(), 1);
        assert!(line.is_ascii());
        let detail_hex = line
            .split_whitespace()
            .find_map(|field| field.strip_prefix("detail_hex="))
            .expect("detail field");
        assert!(detail_hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(detail_hex.len() <= CUDA_WIDE_ERROR_DETAIL_MAX_BYTES * 2);
        assert!(line.ends_with("detail_truncated=0"));
        assert_eq!(
            cuda_wide_error_detail_hex("A\nΩ"),
            ("410acea9".into(), false)
        );

        assert_eq!(
            cuda_wide_error_reason_code(&NyError::InvalidSpec(
                "NY_CUDA_WIDE_MAX_BYTES must be positive".into()
            )),
            "invalid_cap"
        );
        assert_eq!(
            cuda_wide_error_reason_code(&NyError::UnsupportedOp(
                "cuda wide resnet: retained/static estimate 9 exceeds 8-byte cap".into()
            )),
            "cap_below_fixed"
        );

        let oversized = "x".repeat(CUDA_WIDE_ERROR_DETAIL_MAX_BYTES + 1);
        let (encoded, truncated) = cuda_wide_error_detail_hex(&oversized);
        assert_eq!(encoded.len(), CUDA_WIDE_ERROR_DETAIL_MAX_BYTES * 2);
        assert!(truncated);
    }

    // ---- Emulation env guard (soundness): must block anything that could
    // switch cuBLAS off IEEE arithmetic, and allow explicit disables. ----

    #[test]
    fn emulation_guard_blocks_enabled_overrides() {
        for key in CUBLAS_EMULATION_ENV_VARS {
            for value in ["1", "eager", "performant", "default", "38"] {
                let hit = blocked_emulation_override(|k| (k == key).then(|| value.to_string()));
                assert_eq!(
                    hit,
                    Some((key, value.to_string())),
                    "{key}={value} must block engine construction"
                );
            }
        }
    }

    #[test]
    fn emulation_guard_allows_clean_and_disabled_env() {
        assert_eq!(blocked_emulation_override(|_| None), None);
        for disabled in ["", "0"] {
            let hit = blocked_emulation_override(|_| Some(disabled.to_string()));
            assert_eq!(hit, None, "value {disabled:?} means explicitly disabled");
        }
    }

    #[test]
    fn dgemm_triplet_gate_is_exact_raw_and_default_dark() {
        use std::ffi::OsStr;

        assert!(!cuda_dgemm_triplet_enabled(None));
        assert!(cuda_dgemm_triplet_enabled(Some(OsStr::new("1"))));
        for legacy in ["0", "", " 1", "1 ", "+1", "true", "01", "１"] {
            assert!(
                !cuda_dgemm_triplet_enabled(Some(OsStr::new(legacy))),
                "malformed value {legacy:?} must preserve legacy scheduling"
            );
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
            assert!(!cuda_dgemm_triplet_enabled(Some(&non_unicode)));
        }
    }

    #[test]
    fn dgemm_triplet_validation_refuses_i32_and_allocation_overflow() {
        let empty: &[f64] = &[];
        let too_wide = usize::try_from(i32::MAX).expect("usize holds i32") + 1;
        let error =
            validate_dgemm_triplet(too_wide, 1, 1, [empty, empty, empty], [empty, empty, empty])
                .expect_err("truncating cuBLAS dimension must be refused");
        assert!(error.to_string().contains("m exceeds i32"));

        let huge = usize::try_from(i32::MAX).expect("usize holds i32");
        let error =
            validate_dgemm_triplet(huge, 1, huge, [empty, empty, empty], [empty, empty, empty])
                .expect_err("unrepresentable output allocation must be refused");
        assert!(error.to_string().contains("output-byte overflow"));

        let one = [0.0f64];
        let error = validate_dgemm_triplet(1, 1, 1, [&one, empty, &one], [&one, &one, &one])
            .expect_err("every triplet input length must be checked");
        assert!(matches!(error, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn dgemm_pair_validation_checks_both_lhs_and_shared_rhs() {
        let empty: &[f64] = &[];
        let too_wide = usize::try_from(i32::MAX).expect("usize holds i32") + 1;
        let error = validate_dgemm_pair_shared_rhs(too_wide, 1, 1, [empty, empty], empty)
            .expect_err("truncating cuBLAS dimension must be refused");
        assert!(error.to_string().contains("m exceeds i32"));

        let huge = usize::try_from(i32::MAX).expect("usize holds i32");
        let error = validate_dgemm_pair_shared_rhs(huge, 1, huge, [empty, empty], empty)
            .expect_err("unrepresentable output allocation must be refused");
        assert!(error.to_string().contains("output-byte overflow"));

        let one = [0.0f64];
        let error = validate_dgemm_pair_shared_rhs(1, 1, 1, [&one, empty], &one)
            .expect_err("both pair LHS lengths must be checked");
        assert!(matches!(error, NyError::ShapeMismatch { .. }));
        let error = validate_dgemm_pair_shared_rhs(1, 1, 1, [&one, &one], empty)
            .expect_err("shared RHS length must be checked");
        assert!(matches!(error, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn dgemm_pair_later_launch_failure_still_drains_once() {
        use std::cell::{Cell, RefCell};

        let attempts = RefCell::new(Vec::new());
        let drains = Cell::new(0usize);
        let transaction = queue_pair_and_drain(
            |index| {
                attempts.borrow_mut().push(index);
                if index == 1 {
                    Err(NyError::InternalError(
                        "injected pair second-launch failure".into(),
                    ))
                } else {
                    Ok(())
                }
            },
            || true,
            || {
                drains.set(drains.get() + 1);
                true
            },
        );

        assert_eq!(transaction.drain, Ok(ProvenCudaDrain { bind_attempts: 1 }));
        let error = transaction
            .launch_result
            .expect_err("injected pair second launch must fail");
        assert!(error
            .to_string()
            .contains("injected pair second-launch failure"));
        assert_eq!(transaction.calls, 2);
        assert_eq!(*attempts.borrow(), vec![0, 1]);
        assert_eq!(drains.get(), 1, "queued pair work must be drained once");
    }

    #[test]
    fn dgemm_triplet_later_launch_failure_still_drains_once() {
        use std::cell::{Cell, RefCell};

        let attempts = RefCell::new(Vec::new());
        let drains = Cell::new(0usize);
        let transaction = queue_triplet_and_drain(
            |index| {
                attempts.borrow_mut().push(index);
                if index == 1 {
                    Err(NyError::InternalError(
                        "injected second-launch failure".into(),
                    ))
                } else {
                    Ok(())
                }
            },
            || true,
            || {
                drains.set(drains.get() + 1);
                true
            },
        );

        assert_eq!(transaction.drain, Ok(ProvenCudaDrain { bind_attempts: 1 }));
        let error = transaction
            .launch_result
            .expect_err("injected second launch must fail");
        assert!(error.to_string().contains("injected second-launch failure"));
        assert_eq!(transaction.calls, 2);
        assert_eq!(*attempts.borrow(), vec![0, 1]);
        assert_eq!(drains.get(), 1, "queued work must be drained before Err");
        assert_eq!(
            cuda_dgemm_triplet_line(7, 20, 7, 2, 17),
            "NY_CUDA_DGEMM_TRIPLET_V1 transactions=7 calls=20 syncs=7 errors=2 wall_us=17"
        );
        assert!(cuda_dgemm_triplet_should_report(1));
        assert!(cuda_dgemm_triplet_should_report(64));
        assert!(!cuda_dgemm_triplet_should_report(65));
        assert!(cuda_dgemm_triplet_should_report(262_144));
        assert!(!cuda_dgemm_triplet_should_report(262_145));
    }

    #[test]
    fn dgemm_triplet_pre_sync_bind_failure_is_retried_before_one_drain() {
        use std::cell::Cell;

        let binds = Cell::new(0usize);
        let syncs = Cell::new(0usize);
        let transaction = queue_triplet_and_drain(
            |_| Ok(()),
            || {
                binds.set(binds.get() + 1);
                binds.get() != 1
            },
            || {
                syncs.set(syncs.get() + 1);
                true
            },
        );

        assert_eq!(transaction.calls, 3);
        assert!(transaction.launch_result.is_ok());
        assert_eq!(transaction.drain, Ok(ProvenCudaDrain { bind_attempts: 2 }));
        assert_eq!(binds.get(), 2);
        assert_eq!(syncs.get(), 1);
    }

    #[test]
    fn dgemm_triplet_persistent_pre_sync_bind_failure_is_unproven() {
        use std::cell::Cell;

        let binds = Cell::new(0usize);
        let syncs = Cell::new(0usize);
        let transaction = queue_triplet_and_drain(
            |_| Ok(()),
            || {
                binds.set(binds.get() + 1);
                false
            },
            || {
                syncs.set(syncs.get() + 1);
                true
            },
        );

        assert_eq!(
            transaction.drain,
            Err(UnprovenCudaDrain::Bind {
                attempts: CUDA_DGEMM_DRAIN_BIND_ATTEMPTS,
            })
        );
        assert_eq!(binds.get(), CUDA_DGEMM_DRAIN_BIND_ATTEMPTS);
        assert_eq!(syncs.get(), 0, "raw synchronize was never reached");
        let _fail_hard: fn(UnprovenCudaDrain) -> ! = abort_unproven_cuda_quiescence;
    }

    #[test]
    fn dgemm_triplet_raw_drain_failure_is_unproven_and_not_retried() {
        use std::cell::Cell;

        let syncs = Cell::new(0usize);
        let transaction = queue_triplet_and_drain(
            |_| Ok(()),
            || true,
            || {
                syncs.set(syncs.get() + 1);
                false
            },
        );

        assert_eq!(
            transaction.drain,
            Err(UnprovenCudaDrain::Synchronize { bind_attempts: 1 })
        );
        assert_eq!(syncs.get(), 1);
    }

    /// On ATS hardware, the one-sync transaction must be bit-identical to three
    /// serial pinned-IEEE Dgemm calls for the same center/magnitude/error inputs.
    #[test]
    fn cuda_dgemm_triplet_matches_serial_bits_on_ats() {
        const CHILD_MARKER: &str = "NY_CUDA_DGEMM_TRIPLET_PARITY_CHILD";
        const TEST_NAME: &str = "tests::cuda_dgemm_triplet_matches_serial_bits_on_ats";

        if std::env::var_os(CHILD_MARKER).as_deref() != Some(std::ffi::OsStr::new("1")) {
            // Environment mutation is process-global. Run the gated half in a
            // single-test child so parallel tests cannot observe a transient
            // gate value and the call still exercises the public env dispatch.
            let output = std::process::Command::new(
                std::env::current_exe().expect("locate ny-cuda unit-test executable"),
            )
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_MARKER, "1")
            .env("NY_CUDA_DGEMM_TRIPLET", "1")
            .output()
            .expect("spawn isolated CUDA DGEMM triplet parity child");
            assert!(
                output.status.success(),
                "CUDA DGEMM triplet parity child failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }
        assert_eq!(
            std::env::var_os("NY_CUDA_DGEMM_TRIPLET").as_deref(),
            Some(std::ffi::OsStr::new("1")),
            "parity child must enter through the exact public gate"
        );

        let engine = match CudaGemmEngine::new() {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("skipping CUDA DGEMM triplet test (no device): {error}");
                return;
            }
        };
        if !engine.host_ptr_zero_copy() {
            eprintln!(
                "skipping CUDA DGEMM triplet transaction test: device {:?} has no ATS host-pointer access",
                engine.device_name()
            );
            return;
        }

        let (m, k, n) = (37usize, 65usize, 29usize);
        let center_a: Vec<f64> = (0..m * k)
            .map(|i| ((i % 73) as f64) * 0.03125 - 1.0625)
            .collect();
        let center_b: Vec<f64> = (0..k * n)
            .map(|i| ((i % 67) as f64) * -0.0234375 + 0.78125)
            .collect();
        let magnitude_a: Vec<f64> = center_a.iter().map(|x| x.abs()).collect();
        let magnitude_b: Vec<f64> = center_b.iter().map(|x| x.abs()).collect();
        let error_a: Vec<f64> = (0..m * k)
            .map(|i| ((i % 19) as f64) * f64::EPSILON)
            .collect();
        let inputs_a = [&center_a[..], &magnitude_a[..], &error_a[..]];
        let inputs_b = [&center_b[..], &magnitude_b[..], &magnitude_b[..]];

        let serial = [
            engine
                .gemm_f64(m, k, n, inputs_a[0], inputs_b[0])
                .expect("serial center DGEMM"),
            engine
                .gemm_f64(m, k, n, inputs_a[1], inputs_b[1])
                .expect("serial magnitude DGEMM"),
            engine
                .gemm_f64(m, k, n, inputs_a[2], inputs_b[2])
                .expect("serial error DGEMM"),
        ];
        let transaction = engine
            .gemm_f64_triplet(m, k, n, inputs_a, inputs_b)
            .expect("public environment-gated one-sync ATS triplet");

        for member in 0..3 {
            let serial_bits = serial[member].iter().map(|x| x.to_bits());
            let transaction_bits = transaction[member].iter().map(|x| x.to_bits());
            assert!(
                serial_bits.eq(transaction_bits),
                "triplet member {member} changed DGEMM result bits"
            );
        }
    }

    /// On ATS hardware, the shared-RHS pair transaction must be bit-identical
    /// to two serial pinned-IEEE Dgemm calls at a non-square shape.
    #[test]
    fn cuda_dgemm_pair_shared_rhs_matches_two_serial_bits_on_ats() {
        let engine = match CudaGemmEngine::new() {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("skipping CUDA DGEMM pair test (no device): {error}");
                return;
            }
        };
        if !engine.host_ptr_zero_copy() {
            eprintln!(
                "skipping CUDA DGEMM pair transaction test: device {:?} has no ATS host-pointer access",
                engine.device_name()
            );
            return;
        }

        let (m, k, n) = (37usize, 65usize, 29usize);
        let lower: Vec<f64> = (0..m * k)
            .map(|i| ((i % 73) as f64) * 0.03125 - 1.0625)
            .collect();
        let upper: Vec<f64> = (0..m * k)
            .map(|i| ((i % 61) as f64) * -0.046875 + 0.71875)
            .collect();
        let shared_rhs: Vec<f64> = (0..k * n)
            .map(|i| ((i % 67) as f64) * -0.0234375 + 0.78125)
            .collect();

        let serial = [
            engine
                .gemm_f64(m, k, n, &lower, &shared_rhs)
                .expect("serial lower DGEMM"),
            engine
                .gemm_f64(m, k, n, &upper, &shared_rhs)
                .expect("serial upper DGEMM"),
        ];
        let transaction = engine
            .gemm_f64_pair_shared_rhs(m, k, n, [&lower, &upper], &shared_rhs)
            .expect("one-sync ATS shared-RHS pair");

        for member in 0..2 {
            let serial_bits = serial[member].iter().map(|x| x.to_bits());
            let transaction_bits = transaction[member].iter().map(|x| x.to_bits());
            assert!(
                serial_bits.eq(transaction_bits),
                "shared-RHS pair member {member} changed DGEMM result bits"
            );
        }
    }

    /// gemm_f32_fast (tensor-core BF16-split) must stay close to exact f32 —
    /// loose tolerance by design (reduced precision is its contract), but a
    /// wildly wrong result would mean a broken pointer/layout in the raw
    /// cublasGemmEx call, not acceptable even for attack use.
    #[test]
    fn cuda_gemm_f32_fast_approximates_exact() {
        let engine = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping CUDA fast GEMM test (no device): {e}");
                return;
            }
        };
        let (m, k, n) = (33usize, 129usize, 65usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 61) as f32) * 0.043 - 1.2)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 53) as f32) * -0.031 + 0.8)
            .collect();
        let exact = engine.gemm_f32(m, k, n, &a, &b).expect("exact f32 gemm");
        let fast = engine
            .gemm_f32_fast(m, k, n, &a, &b)
            .expect("fast f32 gemm");
        assert_eq!(fast.len(), m * n);
        let mut max_rel = 0.0f32;
        for (f, e) in fast.iter().zip(&exact) {
            let rel = (f - e).abs() / e.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 1e-2,
            "fast gemm diverged from exact beyond attack tolerance: max_rel={max_rel}"
        );
    }

    /// ATS host-pointer zero-copy path (#p2-ats-zero-copy) must give results that
    /// match a CPU f64 GEMM to tight tolerance. When the device advertises
    /// pageable-memory-access (host_ptr_ok, e.g. GB10) `gemm_f64` routes through the
    /// raw-host-pointer `dgemm`; this test asserts that path is exercised AND
    /// numerically sound. On a non-ATS device it validates the unified fallback and
    /// notes the zero-copy path was not exercised (no false confidence).
    #[test]
    fn cuda_gemm_f64_ats_zero_copy_matches_cpu() {
        let engine = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping CUDA ATS zero-copy test (no device): {e}");
                return;
            }
        };
        eprintln!(
            "CUDA device {:?}: host_ptr_zero_copy = {} (ATS path {})",
            engine.device_name(),
            engine.host_ptr_zero_copy(),
            if engine.host_ptr_zero_copy() {
                "EXERCISED"
            } else {
                "NOT exercised (unified fallback)"
            }
        );
        let (m, k, n) = (48usize, 96usize, 40usize);
        let a: Vec<f64> = (0..m * k)
            .map(|i| ((i % 71) as f64) * 0.037 - 1.3)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| ((i % 59) as f64) * -0.029 + 0.9)
            .collect();
        let gpu = engine.gemm_f64(m, k, n, &a, &b).expect("f64 gemm");
        let cpu = cpu_gemm_f64(m, k, n, &a, &b);
        assert_eq!(gpu.len(), m * n);
        let mut max_abs = 0.0f64;
        for (g, c) in gpu.iter().zip(&cpu) {
            max_abs = max_abs.max((g - c).abs());
        }
        // f64 GEMM vs the naive CPU triple loop: both IEEE-f64, different reduction
        // order — a few ULPs at most on these magnitudes.
        assert!(
            max_abs < 1e-9,
            "ATS/unified f64 gemm diverged from CPU: max_abs={max_abs}"
        );
    }

    fn cpu_gemm_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn cuda_gemm_matches_cpu_f32_and_f64() {
        let engine = match CudaGemmEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping CUDA GEMM test (no device): {e}");
                return;
            }
        };
        eprintln!("CUDA device: {}", engine.device_name());

        // Run twice at different sizes to exercise the cached-buffer grow path.
        for &(m, k, n) in &[(5usize, 7usize, 3usize), (9, 4, 6)] {
            let a64: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.3 - 1.1).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64) * -0.2 + 0.7).collect();
            let want = cpu_gemm_f64(m, k, n, &a64, &b64);

            let got64 = engine.gemm_f64(m, k, n, &a64, &b64).expect("cuda f64 gemm");
            assert_eq!(got64.len(), m * n);
            for (g, w) in got64.iter().zip(&want) {
                assert!(
                    (g - w).abs() < 1e-9,
                    "f64 mismatch {m}x{k}x{n}: got {g} want {w}"
                );
            }

            let a32: Vec<f32> = a64.iter().map(|&x| x as f32).collect();
            let b32: Vec<f32> = b64.iter().map(|&x| x as f32).collect();
            let got32 = engine.gemm_f32(m, k, n, &a32, &b32).expect("cuda f32 gemm");
            for (g, w) in got32.iter().zip(&want) {
                assert!(
                    (f64::from(*g) - w).abs() < 1e-3,
                    "f32 mismatch: got {g} want {w}"
                );
            }
        }
    }
}
