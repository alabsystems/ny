// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Native CUDA / cuBLAS GEMM engine for ny (coherent and discrete NVIDIA GPUs).
//!
//! # Why a CUDA backend (a native f64 sound lane)
//!
//! The historical wgpu/Vulkan CROWN **fast** lane is unsound for verdicts: its
//! round-to-nearest f32 backward and concretization omit a certified rounding-error
//! term. That is not a consequence of WGSL lacking `f64`. The separate WGPU sound
//! lane carries f32 Higham `γ_n·S` corrections and an EFT/double-single residual;
//! its U1/U3/U4/U5/U6 obligations and B0 source review are discharged, and the
//! qualified WGPU CROWN gate is open. The public seam is deliberately narrow:
//! an explicit typed constructor runs five live qualification rungs on one
//! device and exposes only that device's CROWN capability. Its reviewed
//! resident Conv route is admitted; host Conv and segment-resident streams
//! still refuse. The CLI consumes that seam conditionally and falls back to CPU
//! on refusal; ordinary WGPU devices and non-CROWN operations remain
//! quarantined (see
//! `ny_gpu::wgpu_device::ops::sound_authority` and
//! `ny_propagate::sound_gpu_gate`).
//!
//! cuBLAS provides IEEE-`f64` GEMM (`cublasDgemm`). The sound CPU CROWN
//! backward (`aw_f64_with_abssum`) computes `A·W` with f64 accumulation plus an
//! abs-sum `S` whose certified error `γ_n·S` bounds the f64 rounding for **any**
//! summation order. cuBLAS's blocked/pairwise f64 summation has error ≤ `γ_n·S`,
//! so routing those two f64 GEMMs (`A·W` and `|A|·|W|`) through cuBLAS is a
//! **sound** acceleration — a native f64 GPU route WGSL cannot express. (Validated
//! against the exact-rational A·W soundness proptest when wired into ny-propagate.)
//!
//! # Memory transport
//!
//! The CROWN backward issues MANY small GEMMs (k≈64) which are transfer-bound, not
//! compute-bound. On Grace-Blackwell the host and GPU share **coherent** memory,
//! so we allocate the operands as **managed/unified** buffers and write/read them
//! host-side with no explicit H2D/D2H copy — ~3.5× faster per call than the
//! copy-based path (measured: 481→137 µs at 512×64×512 f64). `cuMemAllocManaged`
//! is expensive, so the buffers are **cached and reused** (grown to the max size
//! seen) behind the handle's lock; per-call allocation would be slower than copy.
//!
//! Discrete GPUs have the opposite transport tradeoff: CPU access to managed
//! allocations can trigger HMM page migration. NY therefore selects the memory
//! transport from live CUDA capabilities, never from a device name: host-page-
//! table devices use direct access, integrated devices use managed memory, and
//! discrete or unknown-topology devices use cached device allocations with
//! ordered H2D/GEMM/D2H work. `NY_CUDA_GEMM_TRANSPORT` supplies exact A/B
//! overrides; legacy `NY_CUDA_DISCRETE_MODE=1` still forces explicit copies.
//! Constructor known-answer tests qualify the exact selected IEEE f32/f64 path
//! before it can carry a sound result.
//!
//! This crate is the only `unsafe` FFI surface in the GPU stack (cuBLAS is an
//! `unsafe` C API); `ny-core`/`ny-gpu` keep `#![forbid(unsafe_code)]`.

use std::mem::size_of;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, UnifiedSlice,
    ValidAsZeroBits,
};
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};

use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed,
    GpuCrownTrajectoryResult, GpuResidentPatchesRootObservation, GpuResidentPatchesRootPlan,
    GpuResnetBatchedDomainRef, NyError, ResidentCutShadowObservation, ResidentCutShadowOutcome,
    ResidentCutShadowPolicy, ResidentLowerCutCarrier, Result,
    DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};

mod ieee_selfcheck;
mod joint_alpha;
mod sound_authority;
mod sound_crown;

/// Classification used by hardware-adaptive CUDA tests.
///
/// A test is allowed to exercise CUDA only when both runtime libraries are
/// present and the driver reports at least one visible device.  Keeping this
/// decision as a pure seam means ordinary CI can prove the unavailable cases
/// instead of accumulating permanently ignored tests.  Conversely, a host
/// that advertises a device is required to construct and qualify the engine;
/// an initialization failure there is a real test failure, not a skip.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaTestAdmission {
    Capable,
    MissingRuntimeLibraries,
    DeviceProbeUnavailable,
    NoVisibleDevice,
}

#[cfg(test)]
const fn cuda_test_admission(
    runtime_libraries_present: bool,
    device_count: Option<i32>,
) -> CudaTestAdmission {
    if !runtime_libraries_present {
        return CudaTestAdmission::MissingRuntimeLibraries;
    }
    match device_count {
        Some(count) if count > 0 => CudaTestAdmission::Capable,
        Some(_) => CudaTestAdmission::NoVisibleDevice,
        None => CudaTestAdmission::DeviceProbeUnavailable,
    }
}

/// Serialize live CUDA unit tests so parallel `cargo test` execution cannot
/// create many contexts and large scratch caches at once.  This is especially
/// important on memory-constrained development hosts.
#[cfg(test)]
static CUDA_HARDWARE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Run `test` on a qualified CUDA device when this host advertises one.
///
/// On a host without CUDA, this still tests something concrete: the live facts
/// must select one of the unavailable states covered exhaustively by
/// `cuda_test_admission_is_fail_closed`.  There is deliberately no ignored-test
/// or early-return path.  Once a device is advertised, every construction or
/// qualification error fails the test.
#[cfg(test)]
fn with_capable_cuda(test: impl FnOnce(&mut CudaGemmEngine)) {
    let _serial = CUDA_HARDWARE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let libraries = CudaGemmEngine::runtime_libraries_present();
    let count = CudaGemmEngine::runtime_device_count();
    match cuda_test_admission(libraries, count) {
        CudaTestAdmission::Capable => {
            let mut engine = CudaGemmEngine::new().unwrap_or_else(|error| {
                panic!(
                    "CUDA libraries and a visible device were advertised, but engine \
                     construction/qualification failed: {error}"
                )
            });
            test(&mut engine);
        }
        unavailable => {
            assert_ne!(unavailable, CudaTestAdmission::Capable);
            eprintln!(
                "CUDA hardware integration unavailable by tested admission seam: \
                 {unavailable:?} (libraries={libraries}, device_count={count:?})"
            );
        }
    }
}

/// CUDA capability wrapper for paths that specifically require direct access
/// to host page tables.  A capable non-ATS device must select the explicit-copy
/// transport; that fallback assertion replaces the former silent runtime skip.
#[cfg(test)]
fn with_capable_ats_cuda(test: impl FnOnce(&mut CudaGemmEngine)) {
    with_capable_cuda(|engine| {
        if engine.host_ptr_zero_copy() {
            test(engine);
        } else {
            assert_eq!(
                engine.deadline_f64_transport_name(),
                "explicit-device-copy",
                "non-ATS hardware must select the tested explicit-copy fallback"
            );
            eprintln!(
                "CUDA device {} has no direct host-page-table capability; \
                 explicit-device-copy fallback selected",
                engine.device_name()
            );
        }
    });
}

pub use sound_authority::{
    published_bound_is_degraded, sentinel_taint_lanes, sentinel_taint_sticky, CudaAuthorityLadder,
    LaneOutcome,
};

/// Machine-readable schema emitted by `ny --cuda-runtime-info`.
pub const CUDA_RUNTIME_INFO_SCHEMA: &str = "ny_cuda_runtime_info_v3";

/// Exact dynamic-library name candidates compiled into cudarc.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeLibraryCandidates {
    pub driver: Vec<String>,
    pub cublas: Vec<String>,
    pub cublas_lt: Vec<String>,
    pub nvrtc: Vec<String>,
}

/// One file-backed CUDA object mapped into this exact process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeObjectIdentity {
    pub role: String,
    pub provider_symbol: String,
    pub mapped_path: String,
    pub resolved_path: String,
    pub mapped_device_major: u64,
    pub mapped_device_minor: u64,
    pub mapped_inode: u64,
    pub size_bytes: u64,
    pub sha256: String,
    pub fingerprint: CudaRuntimeFileFingerprint,
}

/// Stable metadata sampled from the same open file descriptor used for hashing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeFileFingerprint {
    pub device: u64,
    pub inode: u64,
    pub size_bytes: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

/// Exact CUDA runtime objects selected in this process after engine qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntimeIdentity {
    pub candidates: CudaRuntimeLibraryCandidates,
    pub objects: Vec<CudaRuntimeObjectIdentity>,
    pub nvrtc_status: String,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CudaRuntimeRole {
    Driver,
    Cublas,
    CublasLt,
}

#[cfg(target_os = "linux")]
impl CudaRuntimeRole {
    const fn name(self) -> &'static str {
        match self {
            Self::Driver => "driver",
            Self::Cublas => "cublas",
            Self::CublasLt => "cublas_lt",
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ProcMapProvider {
    mapped_path: PathBuf,
    device_major: u64,
    device_minor: u64,
    inode: u64,
}

/// Pin cudarc's process-global dynamic-library handles exactly once.
///
/// Calling cudarc's `is_culib_present` is not a harmless filesystem query: it
/// performs a separate `dlopen`/`dlclose`, including ELF constructors, before
/// `culib()` later opens the object that cudarc actually retains. Besides doing
/// duplicate work, that creates a check/load pathname race and leaves the first
/// object outside runtime provenance. Calling the real `culib()` initializer
/// under `catch_unwind` converts a loader panic to an error and ensures every
/// later cudarc lookup uses these exact retained handles. The panic hook may
/// still print before unwinding, so callers use [`CudaGemmEngine::runtime_libraries_present`]
/// to cleanly reject the absence of any transiently dlopen-able candidate names
/// before installing a lazy factory. That probe does not identify or retain the
/// objects this initializer will actually select.
fn initialize_cudarc_libraries() -> Result<()> {
    fn load_one(role: &'static str, load: impl FnOnce() + std::panic::UnwindSafe) -> Result<()> {
        match std::panic::catch_unwind(load) {
            Ok(()) => Ok(()),
            Err(payload) => {
                let detail = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("<non-string loader panic>");
                Err(NyError::InternalError(format!(
                    "cuda: cannot initialize retained {role} dynamic-library handle: {detail}"
                )))
            }
        }
    }

    load_one("driver", || {
        // SAFETY: this initializes and borrows cudarc's process-global
        // `OnceLock<Library>`; the returned handle remains owned by cudarc.
        let _ = unsafe { cudarc::driver::sys::culib() };
    })?;
    load_one("cuBLAS", || {
        // SAFETY: as above, for cudarc's retained cuBLAS handle.
        let _ = unsafe { cudarc::cublas::sys::culib() };
    })
}

#[cfg(target_os = "linux")]
const DRIVER_PROVIDER_SYMBOLS: &[&str] = &[
    "cuInit",
    "cuDeviceGet",
    "cuDeviceGetAttribute",
    "cuDeviceGetName",
    "cuDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxRelease_v2",
    "cuCtxGetCurrent",
    "cuCtxSetCurrent",
    "cuMemAllocManaged",
    // CudaStream::alloc selects one of these from the admitted device
    // MEMORY_POOLS_SUPPORTED attribute. Validate both so the invariant is
    // stable across supported CUDA 13 devices.
    "cuMemAllocAsync",
    "cuMemAlloc_v2",
    "cuMemFreeAsync",
    "cuMemFree_v2",
    "cuMemcpyHtoD_v2",
    "cuMemcpyDtoH_v2",
    "cuStreamSynchronize",
    "cuEventCreate",
    "cuEventRecord",
    "cuEventSynchronize",
    "cuEventDestroy_v2",
    "cuStreamWaitEvent",
];

#[cfg(target_os = "linux")]
const CUBLAS_PROVIDER_SYMBOLS: &[&str] = &[
    "cublasCreate_v2",
    "cublasDestroy_v2",
    "cublasSetStream_v2",
    "cublasSetMathMode",
    "cublasGetMathMode",
    "cublasSgemm_v2",
    "cublasDgemm_v2",
    // This path is attack-only rather than verdict-authoritative, but it is a
    // current operation of the shared engine and must not escape provenance.
    "cublasGemmEx",
];

#[cfg(target_os = "linux")]
const CUBLAS_LT_PROVIDER_SYMBOLS: &[&str] = &[
    "cublasLtGetVersion",
    // Public CUDA-13 cuBLASLt entry points present across toolkit minors. The
    // numerical entry point and its algorithm selector are both bound so a
    // wrapper cannot forward only the diagnostic version query.
    "cublasLtMatmul",
    "cublasLtMatmulAlgoGetHeuristic",
];

#[cfg(target_os = "linux")]
const CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS: &[&str; 2] = &[
    // Some CUDA-13 libcublas builds (including the qualified 13.2 runtime)
    // delegate verdict-relevant f64/f32 GEMMs through these undocumented
    // DT_NEEDED libcublasLt exports. They are not a cross-minor API contract:
    // bind them when both exist, accept the public-symbol path when neither
    // exists, and reject a half-present pair.
    "cublasLt_for_cublas_DDD",
    "cublasLt_for_cublas_SSS",
];

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct CudaRuntimeProviderMappings {
    driver: ProcMapProvider,
    cublas: ProcMapProvider,
    cublas_lt: ProcMapProvider,
}

/// Return the exact candidate arrays used by this cudarc build.
#[must_use]
pub fn cuda_runtime_library_candidates() -> CudaRuntimeLibraryCandidates {
    let candidates = |names: &[&str]| {
        names
            .iter()
            .flat_map(|name| cudarc::get_lib_name_candidates(name))
            .collect()
    };
    CudaRuntimeLibraryCandidates {
        driver: candidates(&["cuda", "nvcuda"]),
        cublas: candidates(&["cublas"]),
        cublas_lt: candidates(&["cublasLt"]),
        nvrtc: candidates(&["nvrtc"]),
    }
}

#[cfg(target_os = "linux")]
fn decode_proc_maps_path(raw: &str) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;

    let raw = raw.as_bytes();
    let mut decoded = Vec::with_capacity(raw.len());
    let mut index = 0;
    while index < raw.len() {
        if raw[index] != b'\\' {
            decoded.push(raw[index]);
            index += 1;
            continue;
        }
        if index + 3 >= raw.len()
            || !raw[index + 1..=index + 3]
                .iter()
                .all(|byte| matches!(byte, b'0'..=b'7'))
        {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: malformed /proc/self/maps escape in {raw:?}"
            )));
        }
        let value = u16::from(raw[index + 1] - b'0') * 64
            + u16::from(raw[index + 2] - b'0') * 8
            + u16::from(raw[index + 3] - b'0');
        let value = u8::try_from(value).map_err(|_| {
            NyError::InternalError(format!(
                "cuda runtime identity: out-of-range /proc/self/maps escape in {raw:?}"
            ))
        })?;
        decoded.push(value);
        index += 4;
    }
    Ok(PathBuf::from(std::ffi::OsString::from_vec(decoded)))
}

#[cfg(target_os = "linux")]
fn parse_proc_maps_provider(maps: &str, address: usize) -> Result<ProcMapProvider> {
    let mut provider = None;
    for line in maps.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(range) = fields.next() else {
            continue;
        };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end, 16) else {
            continue;
        };
        if !(start <= address && address < end) {
            continue;
        }
        if provider.is_some() {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: provider address {address:#x} has overlapping mappings"
            )));
        }
        let permissions = fields.next().ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: provider mapping has no permissions: {line}"
            ))
        })?;
        if permissions.as_bytes().get(2) != Some(&b'x') {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: provider symbol is not in an executable mapping: {line}"
            )));
        }
        let _offset = fields.next().ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: provider mapping has no file offset: {line}"
            ))
        })?;
        let device = fields.next().ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: provider mapping has no device: {line}"
            ))
        })?;
        let inode = fields.next().ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: provider mapping has no inode: {line}"
            ))
        })?;
        let raw_path = fields.next().ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: provider mapping has no file path: {line}"
            ))
        })?;
        if fields.next().is_some() {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: provider object is deleted or malformed: {line}"
            )));
        }
        let path = decode_proc_maps_path(raw_path)?;
        if !path.is_absolute() {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: provider path is not absolute: {}",
                path.display()
            )));
        }
        let (major, minor) = device.split_once(':').ok_or_else(|| {
            NyError::InternalError(format!(
                "cuda runtime identity: malformed provider device: {device}"
            ))
        })?;
        let device_major = u64::from_str_radix(major, 16).map_err(|error| {
            NyError::InternalError(format!(
                "cuda runtime identity: malformed provider device major {major}: {error}"
            ))
        })?;
        let device_minor = u64::from_str_radix(minor, 16).map_err(|error| {
            NyError::InternalError(format!(
                "cuda runtime identity: malformed provider device minor {minor}: {error}"
            ))
        })?;
        let inode = inode.parse::<u64>().map_err(|error| {
            NyError::InternalError(format!(
                "cuda runtime identity: malformed provider inode {inode}: {error}"
            ))
        })?;
        if inode == 0 {
            return Err(NyError::InternalError(format!(
                "cuda runtime identity: provider object has no file inode: {}",
                path.display(),
            )));
        }
        provider = Some(ProcMapProvider {
            mapped_path: path,
            device_major,
            device_minor,
            inode,
        });
    }
    provider.ok_or_else(|| {
        NyError::InternalError(format!(
            "cuda runtime identity: no file-backed mapping contains provider address {address:#x}"
        ))
    })
}

#[cfg(target_os = "linux")]
const fn linux_device_numbers(device: u64) -> (u64, u64) {
    let major = ((device & 0x0000_0000_000f_ff00) >> 8) | ((device & 0xffff_f000_0000_0000) >> 32);
    let minor = (device & 0x0000_0000_0000_00ff) | ((device & 0x0000_0fff_fff0_0000) >> 12);
    (major, minor)
}

#[cfg(target_os = "linux")]
fn file_fingerprint(metadata: &std::fs::Metadata) -> CudaRuntimeFileFingerprint {
    use std::os::unix::fs::MetadataExt;

    const NANOS_PER_SECOND: i128 = 1_000_000_000;
    CudaRuntimeFileFingerprint {
        device: metadata.dev(),
        inode: metadata.ino(),
        size_bytes: metadata.size(),
        mtime_ns: i128::from(metadata.mtime()) * NANOS_PER_SECOND
            + i128::from(metadata.mtime_nsec()),
        ctime_ns: i128::from(metadata.ctime()) * NANOS_PER_SECOND
            + i128::from(metadata.ctime_nsec()),
    }
}

#[cfg(target_os = "linux")]
fn hash_provider_file(
    role: CudaRuntimeRole,
    provider_symbol: &'static str,
    mapping: ProcMapProvider,
) -> Result<CudaRuntimeObjectIdentity> {
    use std::io::Read as _;

    let resolved = mapping.mapped_path.canonicalize().map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot resolve {} provider {}: {error}",
            role.name(),
            mapping.mapped_path.display()
        ))
    })?;
    let mut file = std::fs::File::open(&resolved).map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot open {} provider {}: {error}",
            role.name(),
            resolved.display()
        ))
    })?;
    let before_metadata = file.metadata().map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot fstat {} provider {}: {error}",
            role.name(),
            resolved.display()
        ))
    })?;
    if !before_metadata.file_type().is_file() {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: {} provider is not a regular file: {}",
            role.name(),
            resolved.display()
        )));
    }
    let before = file_fingerprint(&before_metadata);
    let (device_major, device_minor) = linux_device_numbers(before.device);
    if before.inode != mapping.inode
        || device_major != mapping.device_major
        || device_minor != mapping.device_minor
    {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: opened {} provider does not match its symbol mapping: {}",
            role.name(),
            mapping.mapped_path.display()
        )));
    }

    let mut hasher = Sha256::new();
    let mut bytes_read = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024].into_boxed_slice();
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            NyError::InternalError(format!(
                "cuda runtime identity: cannot hash {} provider {}: {error}",
                role.name(),
                resolved.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        bytes_read = bytes_read
            .checked_add(u64::try_from(count).expect("read count fits u64"))
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "cuda runtime identity: {} provider length overflow",
                    role.name()
                ))
            })?;
        hasher.update(&buffer[..count]);
    }
    let after = file_fingerprint(&file.metadata().map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot re-fstat {} provider {}: {error}",
            role.name(),
            resolved.display()
        ))
    })?);
    if after != before || bytes_read != before.size_bytes {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: {} provider changed while its open descriptor was hashed: {}",
            role.name(),
            resolved.display()
        )));
    }
    let resolved_after = mapping.mapped_path.canonicalize().map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot re-resolve {} provider {}: {error}",
            role.name(),
            mapping.mapped_path.display()
        ))
    })?;
    let path_after = file_fingerprint(&resolved_after.metadata().map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot re-stat {} provider {}: {error}",
            role.name(),
            resolved_after.display()
        ))
    })?);
    if resolved_after != resolved || path_after != after {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: {} provider path changed during capture: {}",
            role.name(),
            mapping.mapped_path.display()
        )));
    }
    let mapped_path = mapping.mapped_path.to_str().ok_or_else(|| {
        NyError::InternalError(format!(
            "cuda runtime identity: mapped {} provider path is not UTF-8",
            role.name()
        ))
    })?;
    let resolved_path = resolved.to_str().ok_or_else(|| {
        NyError::InternalError(format!(
            "cuda runtime identity: resolved {} provider path is not UTF-8",
            role.name()
        ))
    })?;
    Ok(CudaRuntimeObjectIdentity {
        role: role.name().to_string(),
        provider_symbol: provider_symbol.to_string(),
        mapped_path: mapped_path.to_string(),
        resolved_path: resolved_path.to_string(),
        mapped_device_major: mapping.device_major,
        mapped_device_minor: mapping.device_minor,
        mapped_inode: mapping.inode,
        size_bytes: before.size_bytes,
        sha256: format!("{:x}", hasher.finalize()),
        fingerprint: before,
    })
}

#[cfg(target_os = "linux")]
fn confirm_dladdr_provider(
    role: CudaRuntimeRole,
    provider_symbol: &'static str,
    address: usize,
) -> Result<()> {
    use std::ffi::CStr;
    use std::mem::MaybeUninit;

    let mut info = MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: `address` came from libloading's typed lookup of
    // `provider_symbol`; `info` is a valid out-slot for dladdr.
    if unsafe { libc::dladdr(address as *const libc::c_void, info.as_mut_ptr()) } == 0 {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: dladdr could not bind {} symbol {provider_symbol}",
            role.name()
        )));
    }
    // SAFETY: dladdr returned success and initialized the complete Dl_info.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() || info.dli_fbase.is_null() {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: dladdr returned an incomplete {} provider for \
             {provider_symbol}",
            role.name()
        )));
    }
    // SAFETY: a successful dladdr returns a NUL-terminated loader-owned name.
    let provider_name = unsafe { CStr::from_ptr(info.dli_fname) };
    if provider_name.to_bytes().is_empty() {
        return Err(NyError::InternalError(format!(
            "cuda runtime identity: dladdr returned an empty {} provider name for \
             {provider_symbol}",
            role.name()
        )));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
type OpaqueCudaFunction = unsafe extern "C" fn();

#[cfg(target_os = "linux")]
fn resolve_driver_symbol_address(symbol_name: &'static str) -> Result<usize> {
    // SAFETY: `initialize_cudarc_libraries` has pinned this exact handle.
    // We never call through this deliberately opaque function type; dlsym's
    // address is used only to identify its executable file mapping.
    let symbol = unsafe {
        cudarc::driver::sys::culib()
            .get::<OpaqueCudaFunction>(symbol_name.as_bytes())
            .map_err(|error| {
                NyError::InternalError(format!(
                    "cuda runtime identity: cannot resolve driver provider symbol \
                     {symbol_name}: {error}"
                ))
            })?
    };
    Ok((*symbol as *const ()) as usize)
}

#[cfg(target_os = "linux")]
fn resolve_cublas_scope_symbol_address(
    role: CudaRuntimeRole,
    symbol_name: &'static str,
) -> Result<usize> {
    // SAFETY: as above, for cudarc's retained cuBLAS handle. dlsym on that
    // handle searches libcublas and its DT_NEEDED dependency scope, which is
    // how the CUDA-13.2 libcublasLt entry points are bound without performing a
    // separate candidate-name dlopen.
    let symbol = unsafe {
        cudarc::cublas::sys::culib()
            .get::<OpaqueCudaFunction>(symbol_name.as_bytes())
            .map_err(|error| {
                NyError::InternalError(format!(
                    "cuda runtime identity: cannot resolve {} provider symbol \
                     {symbol_name} through the retained cuBLAS scope: {error}",
                    role.name()
                ))
            })?
    };
    Ok((*symbol as *const ()) as usize)
}

#[cfg(target_os = "linux")]
fn try_resolve_cublas_scope_symbol_address(symbol_name: &'static str) -> Option<usize> {
    // SAFETY: as in `resolve_cublas_scope_symbol_address`, this address is
    // inspected but never called. Missing undocumented delegation exports are
    // a supported CUDA-13-minor state and therefore return `None`.
    unsafe {
        cudarc::cublas::sys::culib()
            .get::<OpaqueCudaFunction>(symbol_name.as_bytes())
            .ok()
            .map(|symbol| (*symbol as *const ()) as usize)
    }
}

#[cfg(target_os = "linux")]
fn select_cublas_lt_delegation_addresses(
    addresses: [Option<usize>; 2],
) -> Result<Vec<(&'static str, usize)>> {
    match addresses {
        [None, None] => Ok(Vec::new()),
        [Some(ddd), Some(sss)] => Ok(vec![
            (CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS[0], ddd),
            (CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS[1], sss),
        ]),
        _ => Err(NyError::InternalError(
            "cuda runtime identity: inconsistent cuBLASLt delegation exports; \
             cublasLt_for_cublas_DDD and cublasLt_for_cublas_SSS must be both present or both absent"
                .to_string(),
        )),
    }
}

#[cfg(target_os = "linux")]
fn resolve_cublas_lt_provider_symbol_addresses() -> Result<Vec<(&'static str, usize)>> {
    let mut addresses =
        resolve_role_symbol_addresses(CudaRuntimeRole::CublasLt, CUBLAS_LT_PROVIDER_SYMBOLS)?;
    addresses.extend(select_cublas_lt_delegation_addresses([
        try_resolve_cublas_scope_symbol_address(CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS[0]),
        try_resolve_cublas_scope_symbol_address(CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS[1]),
    ])?);
    Ok(addresses)
}

#[cfg(target_os = "linux")]
fn resolve_role_symbol_addresses(
    role: CudaRuntimeRole,
    symbols: &'static [&'static str],
) -> Result<Vec<(&'static str, usize)>> {
    symbols
        .iter()
        .copied()
        .map(|symbol| {
            let address = match role {
                CudaRuntimeRole::Driver => resolve_driver_symbol_address(symbol),
                CudaRuntimeRole::Cublas | CudaRuntimeRole::CublasLt => {
                    resolve_cublas_scope_symbol_address(role, symbol)
                }
            }?;
            Ok((symbol, address))
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn validate_role_symbol_addresses(
    maps: &str,
    role: CudaRuntimeRole,
    symbol_addresses: &[(&'static str, usize)],
    mut confirm: impl FnMut(CudaRuntimeRole, &'static str, usize) -> Result<()>,
) -> Result<ProcMapProvider> {
    let mut representative: Option<(&'static str, ProcMapProvider)> = None;
    for &(symbol, address) in symbol_addresses {
        confirm(role, symbol, address)?;
        let mapping = parse_proc_maps_provider(maps, address)?;
        if let Some((representative_symbol, representative_mapping)) = representative.as_ref() {
            if mapping.device_major != representative_mapping.device_major
                || mapping.device_minor != representative_mapping.device_minor
                || mapping.inode != representative_mapping.inode
            {
                return Err(NyError::InternalError(format!(
                    "cuda runtime identity: mixed {} providers: {representative_symbol} maps to \
                     {} ({:x}:{:x}, inode {}), but {symbol} maps to {} ({:x}:{:x}, inode {})",
                    role.name(),
                    representative_mapping.mapped_path.display(),
                    representative_mapping.device_major,
                    representative_mapping.device_minor,
                    representative_mapping.inode,
                    mapping.mapped_path.display(),
                    mapping.device_major,
                    mapping.device_minor,
                    mapping.inode,
                )));
            }
        } else {
            representative = Some((symbol, mapping));
        }
    }
    representative.map(|(_, mapping)| mapping).ok_or_else(|| {
        NyError::InternalError(format!(
            "cuda runtime identity: {} provider symbol set is empty",
            role.name()
        ))
    })
}

#[cfg(target_os = "linux")]
fn validate_live_cuda_provider_consistency() -> Result<CudaRuntimeProviderMappings> {
    // Resolve the complete sets before sampling maps. A dependency lookup may
    // perform lazy loader work; the subsequent single maps snapshot must include
    // every address that this admission decision validates.
    let driver_addresses =
        resolve_role_symbol_addresses(CudaRuntimeRole::Driver, DRIVER_PROVIDER_SYMBOLS)?;
    let cublas_addresses =
        resolve_role_symbol_addresses(CudaRuntimeRole::Cublas, CUBLAS_PROVIDER_SYMBOLS)?;
    let cublas_lt_addresses = resolve_cublas_lt_provider_symbol_addresses()?;
    let maps = std::fs::read_to_string("/proc/self/maps").map_err(|error| {
        NyError::InternalError(format!(
            "cuda runtime identity: cannot read /proc/self/maps: {error}"
        ))
    })?;
    Ok(CudaRuntimeProviderMappings {
        driver: validate_role_symbol_addresses(
            &maps,
            CudaRuntimeRole::Driver,
            &driver_addresses,
            confirm_dladdr_provider,
        )?,
        cublas: validate_role_symbol_addresses(
            &maps,
            CudaRuntimeRole::Cublas,
            &cublas_addresses,
            confirm_dladdr_provider,
        )?,
        cublas_lt: validate_role_symbol_addresses(
            &maps,
            CudaRuntimeRole::CublasLt,
            &cublas_lt_addresses,
            confirm_dladdr_provider,
        )?,
    })
}

#[cfg(all(target_os = "linux", test))]
fn capture_symbol_provider(
    maps: &str,
    role: CudaRuntimeRole,
    provider_symbol: &'static str,
    address: usize,
) -> Result<CudaRuntimeObjectIdentity> {
    confirm_dladdr_provider(role, provider_symbol, address)?;
    let mapping = parse_proc_maps_provider(maps, address)?;
    hash_provider_file(role, provider_symbol, mapping)
}

/// Capture exact file-backed CUDA objects mapped into this process.
///
/// Call this only after [`CudaGemmEngine::new`] succeeds, so the driver,
/// cuBLAS, and cuBLASLt dependency have all been loaded and exercised.
pub fn cuda_runtime_identity() -> Result<CudaRuntimeIdentity> {
    #[cfg(target_os = "linux")]
    {
        initialize_cudarc_libraries()?;
        let providers = validate_live_cuda_provider_consistency()?;
        let mut objects = vec![
            hash_provider_file(
                CudaRuntimeRole::Driver,
                DRIVER_PROVIDER_SYMBOLS[0],
                providers.driver,
            )?,
            hash_provider_file(
                CudaRuntimeRole::Cublas,
                CUBLAS_PROVIDER_SYMBOLS[0],
                providers.cublas,
            )?,
            hash_provider_file(
                CudaRuntimeRole::CublasLt,
                CUBLAS_LT_PROVIDER_SYMBOLS[0],
                providers.cublas_lt,
            )?,
        ];
        for left in 0..objects.len() {
            for right in left + 1..objects.len() {
                if objects[left].fingerprint.device == objects[right].fingerprint.device
                    && objects[left].fingerprint.inode == objects[right].fingerprint.inode
                {
                    return Err(NyError::InternalError(format!(
                        "cuda runtime identity: {} and {} unexpectedly resolve to the same \
                         provider object",
                        objects[left].role, objects[right].role
                    )));
                }
            }
        }
        objects.sort_by(|left, right| left.role.cmp(&right.role));
        Ok(CudaRuntimeIdentity {
            candidates: cuda_runtime_library_candidates(),
            objects,
            // ny-cuda deliberately enables only cudarc's driver+cublas
            // features. Do not infer optional NVRTC identity from unrelated
            // basename-like mappings.
            nvrtc_status: "not_loaded_feature_disabled".to_string(),
        })
    }
    #[cfg(not(target_os = "linux"))]
    Err(NyError::InternalError(
        "cuda runtime identity is supported only on Linux".to_string(),
    ))
}

fn contain_device_count<E>(
    probe: impl FnOnce() -> std::result::Result<i32, E> + std::panic::UnwindSafe,
) -> Option<i32> {
    std::panic::catch_unwind(probe)
        .ok()
        .and_then(std::result::Result::ok)
}

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

/// Transport for a finite-deadline f64 CUDA tile.
///
/// `PAGEABLE_MEMORY_ACCESS` alone is a correctness capability, not a performance
/// guarantee. On a discrete GPU CUDA HMM may service a plain host pointer by
/// faulting and migrating every touched page. Direct access is therefore
/// reserved for devices that also report that pageable access uses the host page
/// tables; all other devices use explicit copies to cached device allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeadlineF64Transport {
    DirectHostPageTables,
    ExplicitDeviceCopy,
}

/// Transport used by ordinary (non-deadline) CUDA GEMMs.
///
/// Automatic selection is based only on live memory-topology capabilities:
/// host page tables beat all other routes, integrated GPUs retain managed
/// memory, and discrete or unknown-topology devices use explicit copies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrdinaryGemmTransport {
    DirectHostPageTables,
    UnifiedMemory,
    ExplicitDeviceCopy,
}

const CUDA_DISCRETE_MODE_ENV: &str = "NY_CUDA_DISCRETE_MODE";
const CUDA_GEMM_TRANSPORT_ENV: &str = "NY_CUDA_GEMM_TRANSPORT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrdinaryGemmTransportPolicy {
    Auto,
    ForceDirectHostPageTables,
    ForceUnifiedMemory,
    ForceExplicitDeviceCopy,
    LegacyForceExplicitDeviceCopy,
}

impl OrdinaryGemmTransportPolicy {
    const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceDirectHostPageTables => "override-direct-host-page-tables",
            Self::ForceUnifiedMemory => "override-unified-memory",
            Self::ForceExplicitDeviceCopy => "override-explicit-device-copy",
            Self::LegacyForceExplicitDeviceCopy => "legacy-discrete-mode-override",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrdinaryGemmTransportReason {
    HostPageTables,
    IntegratedDevice,
    DiscreteDevice,
    UnknownTopology,
    ExplicitOverride,
    LegacyDiscreteOverride,
}

impl OrdinaryGemmTransportReason {
    const fn name(self) -> &'static str {
        match self {
            Self::HostPageTables => "pageable-access-uses-host-page-tables",
            Self::IntegratedDevice => "integrated-device",
            Self::DiscreteDevice => "discrete-device",
            Self::UnknownTopology => "topology-query-failed-explicit-copy",
            Self::ExplicitOverride => "explicit-transport-override",
            Self::LegacyDiscreteOverride => "legacy-discrete-mode-override",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OrdinaryGemmTransportSelection {
    transport: OrdinaryGemmTransport,
    policy: OrdinaryGemmTransportPolicy,
    reason: OrdinaryGemmTransportReason,
}

/// Parse the public discrete-GPU opt-in exactly. An invalid value fails engine
/// construction instead of silently selecting a transport the user did not
/// request.
fn cuda_discrete_mode_requested(raw: Option<&std::ffi::OsStr>) -> Result<bool> {
    match raw.and_then(std::ffi::OsStr::to_str) {
        None => {
            if raw.is_some() {
                Err(NyError::InvalidSpec(format!(
                    "{CUDA_DISCRETE_MODE_ENV} must be exactly 0 or 1"
                )))
            } else {
                Ok(false)
            }
        }
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(NyError::InvalidSpec(format!(
            "{CUDA_DISCRETE_MODE_ENV} must be exactly 0 or 1"
        ))),
    }
}

fn parse_ordinary_gemm_transport_policy(
    raw: Option<&std::ffi::OsStr>,
    legacy_discrete_raw: Option<&std::ffi::OsStr>,
) -> Result<OrdinaryGemmTransportPolicy> {
    let legacy_discrete = cuda_discrete_mode_requested(legacy_discrete_raw)?;
    let requested = match raw.and_then(std::ffi::OsStr::to_str) {
        None if raw.is_some() => {
            return Err(NyError::InvalidSpec(format!(
                "{CUDA_GEMM_TRANSPORT_ENV} must be exactly auto, direct-host-page-tables, \
                 unified-memory, or explicit-device-copy"
            )))
        }
        None | Some("auto") => OrdinaryGemmTransportPolicy::Auto,
        Some("direct-host-page-tables") => OrdinaryGemmTransportPolicy::ForceDirectHostPageTables,
        Some("unified-memory") => OrdinaryGemmTransportPolicy::ForceUnifiedMemory,
        Some("explicit-device-copy") => OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy,
        Some(_) => {
            return Err(NyError::InvalidSpec(format!(
                "{CUDA_GEMM_TRANSPORT_ENV} must be exactly auto, direct-host-page-tables, \
                 unified-memory, or explicit-device-copy"
            )))
        }
    };

    if !legacy_discrete {
        return Ok(requested);
    }
    match requested {
        OrdinaryGemmTransportPolicy::Auto => {
            Ok(OrdinaryGemmTransportPolicy::LegacyForceExplicitDeviceCopy)
        }
        OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy => Ok(requested),
        OrdinaryGemmTransportPolicy::ForceDirectHostPageTables
        | OrdinaryGemmTransportPolicy::ForceUnifiedMemory => Err(NyError::InvalidSpec(format!(
            "{CUDA_DISCRETE_MODE_ENV}=1 conflicts with {CUDA_GEMM_TRANSPORT_ENV}; \
             unset the legacy option or select explicit-device-copy"
        ))),
        OrdinaryGemmTransportPolicy::LegacyForceExplicitDeviceCopy => unreachable!(),
    }
}

fn select_ordinary_gemm_transport(
    policy: OrdinaryGemmTransportPolicy,
    pageable_memory_access: bool,
    pageable_access_uses_host_page_tables: bool,
    integrated_device: Option<bool>,
) -> Result<OrdinaryGemmTransportSelection> {
    let selected = match policy {
        OrdinaryGemmTransportPolicy::Auto => {
            if pageable_memory_access && pageable_access_uses_host_page_tables {
                OrdinaryGemmTransportSelection {
                    transport: OrdinaryGemmTransport::DirectHostPageTables,
                    policy,
                    reason: OrdinaryGemmTransportReason::HostPageTables,
                }
            } else {
                match integrated_device {
                    Some(true) => OrdinaryGemmTransportSelection {
                        transport: OrdinaryGemmTransport::UnifiedMemory,
                        policy,
                        reason: OrdinaryGemmTransportReason::IntegratedDevice,
                    },
                    Some(false) => OrdinaryGemmTransportSelection {
                        transport: OrdinaryGemmTransport::ExplicitDeviceCopy,
                        policy,
                        reason: OrdinaryGemmTransportReason::DiscreteDevice,
                    },
                    None => OrdinaryGemmTransportSelection {
                        transport: OrdinaryGemmTransport::ExplicitDeviceCopy,
                        policy,
                        reason: OrdinaryGemmTransportReason::UnknownTopology,
                    },
                }
            }
        }
        OrdinaryGemmTransportPolicy::ForceDirectHostPageTables => {
            if !(pageable_memory_access && pageable_access_uses_host_page_tables) {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{CUDA_GEMM_TRANSPORT_ENV}=direct-host-page-tables requires CUDA pageable \
                     memory access backed by host page tables"
                )));
            }
            OrdinaryGemmTransportSelection {
                transport: OrdinaryGemmTransport::DirectHostPageTables,
                policy,
                reason: OrdinaryGemmTransportReason::ExplicitOverride,
            }
        }
        OrdinaryGemmTransportPolicy::ForceUnifiedMemory => OrdinaryGemmTransportSelection {
            transport: OrdinaryGemmTransport::UnifiedMemory,
            policy,
            reason: OrdinaryGemmTransportReason::ExplicitOverride,
        },
        OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy => OrdinaryGemmTransportSelection {
            transport: OrdinaryGemmTransport::ExplicitDeviceCopy,
            policy,
            reason: OrdinaryGemmTransportReason::ExplicitOverride,
        },
        OrdinaryGemmTransportPolicy::LegacyForceExplicitDeviceCopy => {
            OrdinaryGemmTransportSelection {
                transport: OrdinaryGemmTransport::ExplicitDeviceCopy,
                policy,
                reason: OrdinaryGemmTransportReason::LegacyDiscreteOverride,
            }
        }
    };
    Ok(selected)
}

const fn optional_bool_name(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

fn select_deadline_f64_transport(
    pageable_memory_access: bool,
    pageable_access_uses_host_page_tables: bool,
) -> DeadlineF64Transport {
    if pageable_memory_access && pageable_access_uses_host_page_tables {
        DeadlineF64Transport::DirectHostPageTables
    } else {
        DeadlineF64Transport::ExplicitDeviceCopy
    }
}

/// cuBLAS handle + stream + cached scratch buffers. Unified buffers serve the
/// ordinary coherent-memory path; ordinary device buffers serve the automatic
/// or overridden explicit-copy mode; the final f64 triplet is dedicated to the
/// finite-deadline copy transport on discrete/HMM devices.
struct Inner {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    fa64: Option<UnifiedSlice<f64>>,
    fb64: Option<UnifiedSlice<f64>>,
    fc64: Option<UnifiedSlice<f64>>,
    fa32: Option<UnifiedSlice<f32>>,
    fb32: Option<UnifiedSlice<f32>>,
    fc32: Option<UnifiedSlice<f32>>,
    ordinary_device_f64: DeadlineDeviceScratch<CudaSlice<f64>>,
    ordinary_device_f32: DeadlineDeviceScratch<CudaSlice<f32>>,
    deadline_device_f64: DeadlineDeviceScratch<CudaSlice<f64>>,
}

/// A [`GemmEngine`] backed by CUDA + cuBLAS on a single NVIDIA device. It uses
/// direct host-page-table access where qualified, cached coherent unified
/// memory on integrated devices, and cached device allocations on discrete or
/// unknown-topology devices. Construct once; share via `&dyn GemmEngine`.
pub struct CudaGemmEngine {
    ctx: Arc<CudaContext>,
    inner: Mutex<Inner>,
    device_name: String,
    /// Direct plain-host-pointer access is performance-authorized only when CUDA
    /// reports both pageable access and host-page-table use. CUDA HMM without
    /// host page tables is safe but can fault/migrate a page per GPU touch, so it
    /// uses the explicit device-copy transport instead.
    deadline_f64_transport: DeadlineF64Transport,
    ordinary_gemm_transport: OrdinaryGemmTransport,
    ordinary_gemm_transport_policy: OrdinaryGemmTransportPolicy,
    ordinary_gemm_transport_reason: OrdinaryGemmTransportReason,
    pageable_memory_access: bool,
    pageable_access_uses_host_page_tables: bool,
    integrated_device: Option<bool>,
    /// Sound-GPU authority rung caches (`sound_authority`). Each is evaluated at
    /// most once per engine, and only when `NY_CUDA_AUTHORITY_LADDER=1`; an
    /// uninitialized cache is never read as a grant.
    ieee_gemm_model_rung: OnceLock<bool>,
    gradual_underflow_rung: OnceLock<bool>,
    triplet_composition_rung: OnceLock<bool>,
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

/// Grow cached device-only buffers used by the explicit-copy ordinary path.
///
/// `CudaStream::alloc` may enqueue `cuMemAllocAsync`, so replacements are not
/// published to the reusable cache until one mandatory raw stream drain proves
/// them ready. Every caller overwrites the complete live A/B views before GEMM
/// and GEMM overwrites the complete C view (`beta = 0`), so uninitialized device
/// storage is never exposed to Rust or read numerically.
fn ensure_ordinary_device_scratch<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    scratch: &mut DeadlineDeviceScratch<CudaSlice<T>>,
    required: [usize; 3],
) -> Result<()> {
    grow_deadline_device_scratch_transactionally(
        scratch,
        required,
        CudaSlice::len,
        || Ok(()),
        |len| {
            // SAFETY: the exact requested A/B ranges are populated by H2D copy
            // before use; C is fully written by a beta=0 GEMM before D2H copy.
            unsafe { stream.alloc::<T>(len) }.map_err(cuda_err)
        },
        || {
            let quiescence = prove_cuda_stream_quiescent(
                || stream.context().bind_to_thread().is_ok(),
                || {
                    // SAFETY: the live stream is bound immediately before this
                    // raw synchronization attempt.
                    unsafe {
                        cudarc::driver::result::stream::synchronize(stream.cu_stream()).is_ok()
                    }
                },
            );
            queued_gemm_numeric_result(Ok(()), quiescence)
        },
    )
}

struct DeadlineDeviceScratch<T> {
    a: Option<T>,
    b: Option<T>,
    c: Option<T>,
}

impl<T> DeadlineDeviceScratch<T> {
    const fn empty() -> Self {
        Self {
            a: None,
            b: None,
            c: None,
        }
    }
}

/// Transactionally grow cached device scratch.
///
/// cudarc may implement `CudaStream::alloc` with `cuMemAllocAsync`. A returned
/// `CudaSlice` is therefore not reusable until the allocating stream has been
/// proven quiescent. Replacements remain call-local until every requested
/// allocation is complete, a mandatory drain succeeds, and the deadline is
/// still live. Any allocation error or deadline leaves the prior, known-ready
/// cache untouched; a later call cannot mistake an unready allocation for a
/// sufficiently-sized reusable buffer.
///
/// Allocation, polling, and draining are injected so the state-reuse invariant
/// is deterministic and device-independent in unit tests.
fn grow_deadline_device_scratch_transactionally<T>(
    scratch: &mut DeadlineDeviceScratch<T>,
    required: [usize; 3],
    len_of: impl Fn(&T) -> usize,
    mut poll: impl FnMut() -> Result<()>,
    mut allocate: impl FnMut(usize) -> Result<T>,
    mut drain: impl FnMut() -> Result<()>,
) -> Result<()> {
    let needs = [
        scratch
            .a
            .as_ref()
            .is_none_or(|value| len_of(value) < required[0]),
        scratch
            .b
            .as_ref()
            .is_none_or(|value| len_of(value) < required[1]),
        scratch
            .c
            .as_ref()
            .is_none_or(|value| len_of(value) < required[2]),
    ];
    if !needs.into_iter().any(|need| need) {
        return Ok(());
    }

    let mut replacements = DeadlineDeviceScratch::empty();
    let mut allocation_attempted = false;
    let mut preparation = Ok(());
    for index in 0..3 {
        if !needs[index] || preparation.is_err() {
            continue;
        }
        if let Err(error) = poll() {
            preparation = Err(error);
            continue;
        }
        allocation_attempted = true;
        match allocate(required[index]) {
            Ok(replacement) => match index {
                0 => replacements.a = Some(replacement),
                1 => replacements.b = Some(replacement),
                2 => replacements.c = Some(replacement),
                _ => unreachable!("three device scratch slots"),
            },
            Err(error) => preparation = Err(error),
        }
    }
    if !allocation_attempted {
        return preparation;
    }

    // Even an allocation API error may have queued or surfaced asynchronous
    // work. Deadline expiry is terminal but cannot waive this mandatory drain.
    let deadline_before_drain = poll();
    let drain_result = drain();
    let deadline_after_drain = poll();
    deadline_after_drain?;
    deadline_before_drain?;
    preparation?;
    drain_result?;

    if let Some(replacement) = replacements.a {
        scratch.a = Some(replacement);
    }
    if let Some(replacement) = replacements.b {
        scratch.b = Some(replacement);
    }
    if let Some(replacement) = replacements.c {
        scratch.c = Some(replacement);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedGemmShape {
    m: i32,
    k: i32,
    n: i32,
    lhs_len: usize,
    rhs_len: usize,
    output_len: usize,
    output_bytes: usize,
}

/// Validate every dimension/product used by a public CUDA GEMM before any
/// allocation or `usize -> i32` conversion reaches cuBLAS.
fn validate_gemm_shape(
    m: usize,
    k: usize,
    n: usize,
    element_size: usize,
    site: &'static str,
) -> Result<CheckedGemmShape> {
    let m_i32 =
        i32::try_from(m).map_err(|_| NyError::InvalidSpec(format!("{site}: m exceeds i32")))?;
    let k_i32 =
        i32::try_from(k).map_err(|_| NyError::InvalidSpec(format!("{site}: k exceeds i32")))?;
    let n_i32 =
        i32::try_from(n).map_err(|_| NyError::InvalidSpec(format!("{site}: n exceeds i32")))?;
    let lhs_len = m
        .checked_mul(k)
        .ok_or_else(|| NyError::InvalidSpec(format!("{site}: m*k overflow")))?;
    let rhs_len = k
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec(format!("{site}: k*n overflow")))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| NyError::InvalidSpec(format!("{site}: m*n overflow")))?;
    let output_bytes = output_len
        .checked_mul(element_size)
        .ok_or_else(|| NyError::InvalidSpec(format!("{site}: output-byte overflow")))?;
    Ok(CheckedGemmShape {
        m: m_i32,
        k: k_i32,
        n: n_i32,
        lhs_len,
        rhs_len,
        output_len,
        output_bytes,
    })
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
) -> Result<CheckedGemmShape> {
    let shape = validate_gemm_shape(m, k, n, size_of::<f64>(), "cuda dgemm triplet")?;
    // Wide-CROWN already charges all three simultaneous output arrays in
    // CUDA_WIDE_BYTES_PER_ROW_CELL, so its 512 MiB default plan continues to
    // bind wide callers. Non-wide callers retain the legacy sequence's existing
    // three-output peak. This is only a representability check, not a universal
    // 512 MiB cap for arbitrary GemmEngine callers.
    shape.output_bytes.checked_mul(3).ok_or_else(|| {
        NyError::InvalidSpec("cuda dgemm triplet: output-triplet overflow".into())
    })?;

    for input in a {
        if input.len() != shape.lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.lhs_len],
                got: vec![input.len()],
            });
        }
    }
    for input in b {
        if input.len() != shape.rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.rhs_len],
                got: vec![input.len()],
            });
        }
    }

    Ok(shape)
}

/// Validate the two left operands and shared RHS used by the ATS pair
/// transaction before allocating output or handing any host pointer to CUDA.
fn validate_dgemm_pair_shared_rhs(
    m: usize,
    k: usize,
    n: usize,
    a: [&[f64]; 2],
    b: &[f64],
) -> Result<CheckedGemmShape> {
    let shape = validate_gemm_shape(m, k, n, size_of::<f64>(), "cuda dgemm pair")?;
    shape
        .output_bytes
        .checked_mul(2)
        .ok_or_else(|| NyError::InvalidSpec("cuda dgemm pair: output-pair overflow".into()))?;

    for input in a {
        if input.len() != shape.lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.lhs_len],
                got: vec![input.len()],
            });
        }
    }
    if b.len() != shape.rhs_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![shape.rhs_len],
            got: vec![b.len()],
        });
    }

    Ok(shape)
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
/// bind attempt can consume a deferred CUDA error, so retry a bounded number of
/// times solely to establish pointer quiescence. The numeric-publication policy
/// separately rejects every proof that needed a retry. Synchronization itself is
/// never retried: any CUDA error from that call leaves pointer lifetime safety
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
/// when a later launch fails. The call count includes the failing launch
/// attempt. Numeric output is publishable only through
/// `queued_gemm_numeric_result`, whose policy also rejects a drain that needed
/// a context-bind retry.
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

/// Queue one explicit-copy GEMM transaction as four ordered stages:
/// H2D(A), H2D(B), GEMM, D2H(C), followed by one mandatory raw stream drain.
///
/// The H2D sources and D2H destination may be ordinary pageable Rust slices.
/// Therefore no stage error may escape before the drain has proven CUDA no
/// longer retains their pointers. The caller publishes the transaction only via
/// [`queued_gemm_numeric_result`], whose unproven-drain branch aborts without
/// unwinding those borrowed values.
fn queue_explicit_copy_gemm_and_drain(
    stage: impl FnMut(usize) -> Result<()>,
    bind: impl FnMut() -> bool,
    synchronize: impl FnMut() -> bool,
) -> QueuedDgemms {
    queue_dgemms_and_drain(4, stage, bind, synchronize)
}

/// Apply the numeric-publication policy after a borrowed-pointer transaction.
///
/// An unproven drain must abort because CUDA may still retain borrowed host
/// pointers. A proven drain permits the Rust values to be dropped, but a bind
/// retry is still a numeric failure: CUDA context APIs may surface deferred
/// asynchronous launch errors, so the first failed bind can have consumed an
/// error from this transaction. Only a first-attempt bind plus successful
/// synchronization authorizes the launch result.
fn queued_gemm_numeric_result(
    launch_result: Result<()>,
    drain: std::result::Result<ProvenCudaDrain, UnprovenCudaDrain>,
) -> Result<()> {
    match drain {
        Ok(ProvenCudaDrain { bind_attempts: 1 }) => launch_result,
        Ok(proof) => Err(NyError::InternalError(format!(
            "cuda borrowed-pointer GEMM drain required {} context-bind attempts; \
             numeric output is unusable",
            proof.bind_attempts
        ))),
        Err(failure) => abort_unproven_cuda_quiescence(failure),
    }
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

fn allocate_gemm_output<T: Copy>(
    shape: CheckedGemmShape,
    zero: T,
    site: &'static str,
) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.output_len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: shape.output_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    output.resize(shape.output_len, zero);
    Ok(output)
}

fn allocate_dgemm_output(shape: CheckedGemmShape, site: &'static str) -> Result<Vec<f64>> {
    allocate_gemm_output(shape, 0.0f64, site)
}

fn copy_gemm_output<T: Copy>(
    source: &[T],
    shape: CheckedGemmShape,
    site: &'static str,
) -> Result<Vec<T>> {
    debug_assert_eq!(source.len(), shape.output_len);
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.output_len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: shape.output_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    output.extend_from_slice(source);
    Ok(output)
}

impl CudaGemmEngine {
    /// Initialize CUDA device 0 and a cuBLAS handle on its default stream.
    /// Returns an error if no CUDA device/driver is available.
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Whether `libcuda` is loadable on this host, without initializing the
    /// driver or a context (#backend-detect).
    ///
    /// This isolates NVIDIA-driver evidence from cuBLAS availability: backend
    /// detection must avoid a redundant Vulkan context even when the GEMM
    /// engine itself cannot be built.
    pub fn runtime_driver_library_present() -> bool {
        // SAFETY: is_culib_present only attempts a dlopen and reports success.
        unsafe { cudarc::driver::sys::is_culib_present() }
    }

    /// Whether at least one cudarc candidate name for each of libcuda and
    /// libcublas can be transiently dlopen-ed, without initializing a context.
    ///
    /// This is a transient startup-admission probe, not retained-library or
    /// exact-object provenance, and it does not establish symbol compatibility.
    /// `with_ordinal` deliberately initializes the retained cudarc handles it
    /// actually selects; device/context initialization, mapped-object validation,
    /// and the IEEE self-check remain that engine's authority.
    pub fn runtime_libraries_present() -> bool {
        Self::runtime_driver_library_present() && {
            // SAFETY: is_culib_present only attempts a dlopen and reports success.
            unsafe { cudarc::cublas::sys::is_culib_present() }
        }
    }

    /// CUDA's process-visible device count, without creating a context.
    ///
    /// Library presence alone is not device authority: container mounts and
    /// stale driver installations can expose `libcuda` while hiding every
    /// device. `CudaContext::device_count` performs `cuInit` followed by
    /// `cuDeviceGetCount`; neither retains a primary context. Callers should
    /// still initialize the real engine lazily because a later context or
    /// cuBLAS allocation can fail independently.
    ///
    /// A partial/version-mismatched dynamic library can make cudarc panic while
    /// resolving a symbol. Release builds unwind, so contain that failure and
    /// report an indeterminate count instead of aborting backend detection.
    pub fn runtime_device_count() -> Option<i32> {
        if !Self::runtime_driver_library_present() {
            return None;
        }
        contain_device_count(CudaContext::device_count)
    }

    /// Initialize a specific CUDA device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let transport_policy = parse_ordinary_gemm_transport_policy(
            std::env::var_os(CUDA_GEMM_TRANSPORT_ENV).as_deref(),
            std::env::var_os(CUDA_DISCRETE_MODE_ENV).as_deref(),
        )?;
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
        initialize_cudarc_libraries()?;
        // Linux scorecard admission is intentionally before cuInit, context
        // creation, allocation, copies, or cuBLAS handle creation. Every
        // correctness/lifecycle symbol must come from one file per role; this
        // prevents a wrapper from forwarding only the old representative symbol
        // while replacing the numerical or transport path. Non-Linux retains
        // the loader pin and existing CPU-fallback behavior, but runtime mapped-
        // object provenance remains Linux-only.
        #[cfg(target_os = "linux")]
        let _providers = validate_live_cuda_provider_consistency()?;

        let ctx = CudaContext::new(ordinal).map_err(cuda_err)?;
        let device_name = ctx.name().unwrap_or_else(|_| "unknown-cuda".to_string());
        // Pageable-memory probes (#p2-ats-zero-copy). PAGEABLE_MEMORY_ACCESS
        // authorizes plain host pointers for correctness, but CUDA HMM on a
        // discrete GPU may implement that access by faulting/migrating every
        // touched page. The direct route is therefore performance-authorized
        // only when USES_HOST_PAGE_TABLES is also true. Every other device uses
        // cached device allocations plus explicit copies for deadline f64 work.
        // Ordinary GEMMs select from these live capabilities and the INTEGRATED
        // topology bit: unknown topology is treated like a discrete GPU, because
        // explicit copies avoid catastrophic HMM migration and remain correct on
        // both classes. Requiring both pageable attributes also fails closed when
        // either query fails.
        let pageable_memory_access = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            )
            .map(|v| v != 0)
            .unwrap_or(false);
        let pageable_access_uses_host_page_tables = pageable_memory_access
            && ctx
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                )
                .map(|v| v != 0)
                .unwrap_or(false);
        let integrated_device = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .map(|v| v != 0)
            .ok();
        let deadline_f64_transport = select_deadline_f64_transport(
            pageable_memory_access,
            pageable_access_uses_host_page_tables,
        );
        let ordinary_selection = select_ordinary_gemm_transport(
            transport_policy,
            pageable_memory_access,
            pageable_access_uses_host_page_tables,
            integrated_device,
        )?;
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
                ordinary_device_f64: DeadlineDeviceScratch::empty(),
                ordinary_device_f32: DeadlineDeviceScratch::empty(),
                deadline_device_f64: DeadlineDeviceScratch::empty(),
            }),
            device_name,
            deadline_f64_transport,
            ordinary_gemm_transport: ordinary_selection.transport,
            ordinary_gemm_transport_policy: ordinary_selection.policy,
            ordinary_gemm_transport_reason: ordinary_selection.reason,
            pageable_memory_access,
            pageable_access_uses_host_page_tables,
            integrated_device,
            ieee_gemm_model_rung: OnceLock::new(),
            gradual_underflow_rung: OnceLock::new(),
            triplet_composition_rung: OnceLock::new(),
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
        if let Err(e) = engine.assert_deadline_f64_transport_bit_exact() {
            tracing::warn!(
                "cuda: device {ordinal} ({}) failed the deadline f64 transport probe: {e}; \
                 refusing the CUDA engine (the sound CPU f64 path is used instead)",
                engine.device_name
            );
            return Err(e);
        }
        // Sound-GPU authority ladder (`sound_authority`). No-op unless
        // `NY_CUDA_AUTHORITY_LADDER=1`; when engaged it evaluates every rung
        // here — construction holds no cuBLAS lock, so the rungs' own guarded
        // dispatches cannot self-deadlock — and logs the rung-by-rung verdict,
        // so later `provides_sound_gpu_crown()` reads hit warm caches.
        engine.prime_sound_gpu_authority();
        tracing::info!(
            "CudaGemmEngine: initialized on device {ordinal} ({}), \
             {} memory, ordinary GEMM transport {} (policy {}, reason {}, integrated {}), \
             deadline f64 transport {}, \
             CUBLAS_DEFAULT_MATH pinned, \
             IEEE and transport known-answer probes bit-exact",
            engine.device_name,
            if pageable_memory_access && pageable_access_uses_host_page_tables {
                "pageable host-pointer access (host page tables)"
            } else if pageable_memory_access {
                "pageable host-pointer access (CUDA HMM/software coherence)"
            } else {
                "unified"
            },
            engine.ordinary_gemm_transport_name(),
            engine.ordinary_gemm_transport_policy_name(),
            engine.ordinary_gemm_transport_reason(),
            engine.integrated_device_state(),
            engine.deadline_f64_transport_name(),
        );
        Ok(engine)
    }

    /// The CUDA device name (e.g. "NVIDIA GB10").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether this engine authorizes the direct pageable-host-pointer fast path.
    ///
    /// This requires both pageable-memory access and host-page-table use. CUDA
    /// HMM/software coherence without host page tables deliberately returns
    /// `false` and selects explicit device copies for finite-deadline f64 work.
    #[must_use]
    pub fn host_ptr_zero_copy(&self) -> bool {
        self.deadline_f64_transport == DeadlineF64Transport::DirectHostPageTables
    }

    /// Whether ordinary GEMMs may borrow pageable host pointers directly.
    /// Unlike [`Self::host_ptr_zero_copy`], this honors the explicit discrete
    /// transport override and therefore also governs pair/triplet scheduling.
    fn ordinary_host_ptr_zero_copy(&self) -> bool {
        self.ordinary_gemm_transport == OrdinaryGemmTransport::DirectHostPageTables
    }

    /// Stable diagnostic name for the finite-deadline f64 memory transport.
    #[must_use]
    pub fn deadline_f64_transport_name(&self) -> &'static str {
        match self.deadline_f64_transport {
            DeadlineF64Transport::DirectHostPageTables => "direct-host-page-tables",
            DeadlineF64Transport::ExplicitDeviceCopy => "explicit-device-copy",
        }
    }

    /// Stable diagnostic name for ordinary f32/f64 and proposal GEMM transport.
    #[must_use]
    pub fn ordinary_gemm_transport_name(&self) -> &'static str {
        match self.ordinary_gemm_transport {
            OrdinaryGemmTransport::DirectHostPageTables => "direct-host-page-tables",
            OrdinaryGemmTransport::UnifiedMemory => "unified-memory",
            OrdinaryGemmTransport::ExplicitDeviceCopy => "explicit-device-copy",
        }
    }

    /// Stable diagnostic name for how the ordinary transport was selected.
    #[must_use]
    pub fn ordinary_gemm_transport_policy_name(&self) -> &'static str {
        self.ordinary_gemm_transport_policy.name()
    }

    /// Stable diagnostic reason for the selected ordinary transport.
    #[must_use]
    pub fn ordinary_gemm_transport_reason(&self) -> &'static str {
        self.ordinary_gemm_transport_reason.name()
    }

    /// Whether the CUDA topology probe identified this as an integrated GPU.
    /// `None` means the driver query failed; auto mode then fails toward the
    /// universally correct explicit-copy route.
    #[must_use]
    pub fn integrated_device(&self) -> Option<bool> {
        self.integrated_device
    }

    /// Stable text form of [`Self::integrated_device`] for line diagnostics.
    #[must_use]
    pub fn integrated_device_state(&self) -> &'static str {
        optional_bool_name(self.integrated_device)
    }

    /// Live CUDA pageable-memory capability used by transport selection.
    #[must_use]
    pub fn pageable_memory_access(&self) -> bool {
        self.pageable_memory_access
    }

    /// Live CUDA host-page-table capability used by transport selection.
    #[must_use]
    pub fn pageable_access_uses_host_page_tables(&self) -> bool {
        self.pageable_access_uses_host_page_tables
    }

    /// Whether cached device allocations and explicit copies are active.
    ///
    /// Retained for compatibility with the original discrete-mode diagnostic;
    /// this can now be selected automatically as well as by an override.
    #[must_use]
    pub fn discrete_mode_enabled(&self) -> bool {
        self.ordinary_gemm_transport == OrdinaryGemmTransport::ExplicitDeviceCopy
    }

    /// Exercise the finite-deadline f64 transport with an exact known answer.
    ///
    /// The constructor's IEEE probes cover ordinary DGEMM. This additional
    /// probe covers the synchronous H2D/D2H and cached-device-scratch path used
    /// on pageable-memory devices without host page tables. Direct-host-pointer
    /// engines have no distinct transport to qualify, so this is a no-op there.
    pub fn assert_deadline_f64_transport_bit_exact(&self) -> Result<()> {
        if self.deadline_f64_transport != DeadlineF64Transport::ExplicitDeviceCopy {
            return Ok(());
        }

        // (1 + 2^-26)^2 = 1 + 2^-25 + 2^-52 exactly. Its lowest bit catches
        // any reduced-precision DGEMM while the 1x1 shape keeps diagnostics
        // cheap and exercises both copies plus the deadline-bounded launch.
        const OPERAND_BITS: u64 = 0x3FF0_0000_0400_0000;
        const EXPECTED_BITS: u64 = 0x3FF0_0000_0800_0001;
        let operand = f64::from_bits(OPERAND_BITS);
        let got = self.gemm_f64_deadline_device_tile(
            1,
            1,
            1,
            &[operand],
            &[operand],
            Instant::now() + std::time::Duration::from_secs(5),
        )?;
        if got.len() != 1 {
            return Err(NyError::InternalError(format!(
                "cuda: deadline f64 explicit-device-copy probe returned {} elements, want 1",
                got.len()
            )));
        }
        if got[0].to_bits() != EXPECTED_BITS {
            return Err(NyError::InternalError(format!(
                "cuda: deadline f64 explicit-device-copy probe NOT bit-exact: got {:#018x}, \
                 want {EXPECTED_BITS:#018x}",
                got[0].to_bits()
            )));
        }
        Ok(())
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

    /// Direct-host-page-table shared-RHS pair. Both products preserve their
    /// individual GEMM shapes and reduction axes; only the host scheduling
    /// changes to one lock and one mandatory stream drain.
    fn gemm_f64_pair_shared_rhs_ats(
        &self,
        shape: CheckedGemmShape,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        if !self.host_ptr_zero_copy() {
            return Err(NyError::UnsupportedOp(
                "cuda dgemm pair: direct host-page-table access unavailable".into(),
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
        let result = queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
        drop(guard);
        result?;
        Ok(output)
    }

    /// Direct-host-page-table implementation: the caller's six already-live
    /// input slices are passed straight to cuBLAS (no packing), and the three output vectors are
    /// the same vectors that the legacy sound-CROWN sequence retains
    /// simultaneously after its third call. The 512 MiB wide plan already
    /// accounts for this peak and continues to bind wide callers; non-wide
    /// callers preserve their preexisting peak and do not acquire a universal
    /// 512 MiB cap here.
    fn gemm_f64_triplet_ats(
        &self,
        shape: CheckedGemmShape,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        if !self.host_ptr_zero_copy() {
            return Err(NyError::UnsupportedOp(
                "cuda dgemm triplet: direct host-page-table access unavailable".into(),
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
                // every product, and lossless i32 dimensions. Direct
                // host-page-table access is checked before entering this method,
                // so cuBLAS may access these pageable host pointers. The engine owns
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
        let calls = transaction.calls;
        let result = queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
        drop(guard);
        let wall_us = started.elapsed().as_micros();
        record_cuda_dgemm_triplet(calls, wall_us, result.is_err());
        result?;
        Ok(output)
    }

    /// Acquire the cuBLAS serialization lock without ever entering the
    /// unbounded `Mutex::lock` wait used by ordinary GEMMs.
    fn deadline_inner_lock(&self, deadline: Instant) -> Result<std::sync::MutexGuard<'_, Inner>> {
        loop {
            cuda_deadline_gemm_check(deadline)?;
            match self.inner.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(NyError::InternalError(
                        "cuda deadline-bounded f64 GEMM: cublas mutex poisoned".into(),
                    ));
                }
            }
        }
    }

    /// One direct-host-page-table DGEMM tile. The caller proves the full
    /// contraction and MAC cap. After any launch attempt we MUST drain the
    /// stream before borrowed input/output vectors can be dropped, even on
    /// launch failure or deadline expiry.
    fn gemm_f64_deadline_ats_tile(
        &self,
        rows: usize,
        k: usize,
        cols: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
    ) -> Result<Vec<f64>> {
        if !self.host_ptr_zero_copy() {
            return Err(NyError::UnsupportedOp(
                "cuda deadline-bounded direct host-pointer GEMM requires host page tables".into(),
            ));
        }
        cuda_deadline_gemm_check(deadline)?;
        let shape = validate_gemm_shape(
            rows,
            k,
            cols,
            size_of::<f64>(),
            "cuda deadline-bounded f64 GEMM",
        )?;
        if a.len() != shape.lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.lhs_len],
                got: vec![a.len()],
            });
        }
        if b.len() != shape.rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.rhs_len],
                got: vec![b.len()],
            });
        }
        let mut output = allocate_dgemm_output_with_deadline(
            shape.output_len,
            shape.output_bytes,
            "cuda::gemm_f64_with_deadline/tile_output",
            deadline,
        )?;

        let alpha = 1.0f64;
        let beta = 0.0f64;
        let guard = self.deadline_inner_lock(deadline)?;
        cuda_deadline_gemm_check(deadline)?;
        // Binding precedes the first borrowed-pointer launch, so an error here
        // is safe to return without a drain.
        guard.stream.context().bind_to_thread().map_err(cuda_err)?;
        cuda_deadline_gemm_check(deadline)?;
        let transaction = queue_dgemms_and_drain(
            1,
            |_| {
                // SAFETY: checked row/column/contraction dimensions fit i32;
                // a/b/output lengths exactly match them; ATS admission proves
                // pageable host-pointer access; the guard and mandatory drain
                // keep every borrowed pointer live until stream quiescence.
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
                        a.as_ptr(),
                        shape.k,
                        &raw const beta,
                        output.as_mut_ptr(),
                        shape.n,
                    )
                    .map_err(cuda_err)
                }
            },
            || guard.stream.context().bind_to_thread().is_ok(),
            || {
                // SAFETY: the guarded stream remains live and bound; a failed
                // synchronization leaves pointer lifetime unproven and aborts.
                unsafe {
                    cudarc::driver::result::stream::synchronize(guard.stream.cu_stream()).is_ok()
                }
            },
        );
        let result = queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
        drop(guard);
        result?;
        cuda_deadline_gemm_check(deadline)?;
        Ok(output)
    }

    /// One explicit-copy DGEMM tile for discrete GPUs and CUDA HMM devices that
    /// do not use the host page tables.
    ///
    /// Host/device copies use CUDA's synchronous APIs. For pageable H2D input,
    /// CUDA has consumed/staged the borrowed source before the call returns;
    /// device DMA may remain queued, but is ordered before DGEMM on the same
    /// default stream. The three device allocations stay cached behind the
    /// engine lock. Mandatory drains cover all queued allocation, copy, and
    /// cuBLAS work before cached storage can be reused.
    fn gemm_f64_deadline_device_tile(
        &self,
        rows: usize,
        k: usize,
        cols: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
    ) -> Result<Vec<f64>> {
        cuda_deadline_gemm_check(deadline)?;
        let shape = validate_gemm_shape(
            rows,
            k,
            cols,
            size_of::<f64>(),
            "cuda deadline-bounded device-copy f64 GEMM",
        )?;
        if a.len() != shape.lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.lhs_len],
                got: vec![a.len()],
            });
        }
        if b.len() != shape.rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.rhs_len],
                got: vec![b.len()],
            });
        }
        let mut output = allocate_dgemm_output_with_deadline(
            shape.output_len,
            shape.output_bytes,
            "cuda::gemm_f64_with_deadline/device_tile_output",
            deadline,
        )?;

        let mut guard = self.deadline_inner_lock(deadline)?;
        cuda_deadline_gemm_check(deadline)?;
        guard.stream.context().bind_to_thread().map_err(cuda_err)?;
        let stream = guard.stream.clone();

        grow_deadline_device_scratch_transactionally(
            &mut guard.deadline_device_f64,
            [shape.lhs_len, shape.rhs_len, shape.output_len],
            CudaSlice::len,
            || cuda_deadline_gemm_check(deadline),
            |len| {
                // SAFETY: A/B are completely initialized by the synchronous H2D
                // copies below. C is completely written by beta=0 DGEMM before
                // any D2H copy; no uninitialized device value reaches Rust.
                unsafe { stream.alloc::<f64>(len) }.map_err(cuda_err)
            },
            || {
                let quiescence = prove_cuda_stream_quiescent(
                    || stream.context().bind_to_thread().is_ok(),
                    || {
                        // SAFETY: the live default stream is bound immediately
                        // before this raw synchronization attempt.
                        unsafe {
                            cudarc::driver::result::stream::synchronize(stream.cu_stream()).is_ok()
                        }
                    },
                );
                queued_gemm_numeric_result(Ok(()), quiescence)
            },
        )?;

        cuda_deadline_gemm_check(deadline)?;
        let copy_a = {
            let device = guard
                .deadline_device_f64
                .a
                .as_mut()
                .expect("device A ensured");
            let (device_ptr, record) = device.device_ptr_mut(&stream);
            // SAFETY: the cached allocation has at least lhs_len f64 elements,
            // `a` has exactly lhs_len elements, and `cuMemcpyHtoD` completes its
            // read of pageable source memory before this call returns.
            let result = unsafe { cudarc::driver::result::memcpy_htod_sync(device_ptr, a) }
                .map_err(cuda_err);
            drop(record);
            result
        };
        cuda_deadline_gemm_check(deadline)?;
        copy_a?;

        cuda_deadline_gemm_check(deadline)?;
        let copy_b = {
            let device = guard
                .deadline_device_f64
                .b
                .as_mut()
                .expect("device B ensured");
            let (device_ptr, record) = device.device_ptr_mut(&stream);
            // SAFETY: same argument as A, for the exact rhs_len slice.
            let result = unsafe { cudarc::driver::result::memcpy_htod_sync(device_ptr, b) }
                .map_err(cuda_err);
            drop(record);
            result
        };
        cuda_deadline_gemm_check(deadline)?;
        copy_b?;

        cuda_deadline_gemm_check(deadline)?;
        let cfg = gemm_cfg::<f64>(shape, 1.0, 0.0);
        let launch_result = {
            let Inner {
                blas,
                deadline_device_f64,
                ..
            } = &mut *guard;
            let a_view = deadline_device_f64
                .a
                .as_ref()
                .expect("device A ensured")
                .slice(..shape.lhs_len);
            let b_view = deadline_device_f64
                .b
                .as_ref()
                .expect("device B ensured")
                .slice(..shape.rhs_len);
            let mut c_view = deadline_device_f64
                .c
                .as_mut()
                .expect("device C ensured")
                .slice_mut(..shape.output_len);
            // SAFETY: the exact-sized device views and leading dimensions were
            // validated above. Row-major C=A*B uses C^T=B^T*A^T, so the operands
            // are intentionally passed B,A as in every other CUDA GEMM path.
            unsafe { blas.gemm(cfg, &b_view, &a_view, &mut c_view) }.map_err(cuda_err)
        };

        // A deadline can expire during the non-interruptible kernel. It remains
        // terminal, but cannot waive the mandatory drain of cached device
        // storage. An unproven drain aborts through the shared fail-hard policy.
        let deadline_before_drain = cuda_deadline_gemm_check(deadline);
        let drain = prove_cuda_stream_quiescent(
            || stream.context().bind_to_thread().is_ok(),
            || {
                // SAFETY: the guarded default stream remains live and was bound
                // immediately before this raw synchronization attempt.
                unsafe { cudarc::driver::result::stream::synchronize(stream.cu_stream()).is_ok() }
            },
        );
        let numeric_result = queued_gemm_numeric_result(launch_result, drain);
        deadline_before_drain?;
        numeric_result?;
        cuda_deadline_gemm_check(deadline)?;

        let copy_c = {
            let device = guard
                .deadline_device_f64
                .c
                .as_ref()
                .expect("device C ensured");
            let (device_ptr, record) = device.device_ptr(&stream);
            // SAFETY: the successful drain completed the full shape.output_len
            // device write. The destination has exactly that many f64 elements,
            // and this copy is synchronous.
            let result =
                unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut output, device_ptr) }
                    .map_err(cuda_err);
            drop(record);
            let event_record = stream.context().check_err().map_err(cuda_err);
            (result, event_record)
        };
        cuda_deadline_gemm_check(deadline)?;
        copy_c.0?;
        copy_c.1?;
        drop(guard);
        cuda_deadline_gemm_check(deadline)?;
        Ok(output)
    }
}

/// cuBLAS column-major config for a row-major `C(m×n) = A(m×k)·B(k×n)` computed
/// as the column-major `Cᵀ = Bᵀ·Aᵀ` (pass B,A swapped, op_N, ld = n,k,n).
fn gemm_cfg<T: Copy>(shape: CheckedGemmShape, one: T, zero: T) -> GemmConfig<T> {
    GemmConfig::<T> {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: shape.n,
        n: shape.m,
        k: shape.k,
        alpha: one,
        lda: shape.n,
        ldb: shape.k,
        beta: zero,
        ldc: shape.n,
    }
}

/// Absolute implementation cap for one non-interruptible deadline-scoped
/// DGEMM dispatch. Callers may request a smaller cap, never a larger one.
const CUDA_DEADLINE_F64_HARD_MAX_MACS: usize = 1 << 24;
static CUDA_DEADLINE_F64_CALLS: AtomicU64 = AtomicU64::new(0);
static CUDA_DEADLINE_F64_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static CUDA_DEADLINE_F64_WALL_US: AtomicU64 = AtomicU64::new(0);

/// Process-wide, lock-free accounting for completed deadline-scoped CUDA f64
/// GEMM computations. Reading a snapshot never changes routing or bounds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaDeadlineF64GemmStats {
    pub calls: u64,
    pub dispatches: u64,
    pub wall_us: u64,
}

/// Read aggregate deadline-scoped CUDA f64 GEMM telemetry without performing
/// I/O in the authoritative deadline path.
pub fn cuda_deadline_f64_gemm_stats() -> CudaDeadlineF64GemmStats {
    CudaDeadlineF64GemmStats {
        calls: CUDA_DEADLINE_F64_CALLS.load(Ordering::Relaxed),
        dispatches: CUDA_DEADLINE_F64_DISPATCHES.load(Ordering::Relaxed),
        wall_us: CUDA_DEADLINE_F64_WALL_US.load(Ordering::Relaxed),
    }
}

fn record_cuda_deadline_f64_gemm(dispatches: usize, wall_us: u128) {
    CUDA_DEADLINE_F64_CALLS.fetch_add(1, Ordering::Relaxed);
    CUDA_DEADLINE_F64_DISPATCHES.fetch_add(
        u64::try_from(dispatches).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    CUDA_DEADLINE_F64_WALL_US.fetch_add(
        u64::try_from(wall_us).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

#[inline]
fn cuda_deadline_gemm_check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(
            "cuda deadline-bounded f64 GEMM exceeded its deadline".into(),
        ))
    } else {
        Ok(())
    }
}

/// Allocate and initialize a deadline-scoped f64 buffer in bounded host units.
///
/// `try_reserve_exact` is bracketed as a single allocator resource wait; page
/// initialization is split so large output geometry never creates an
/// unpolled zero-fill tail.
fn allocate_dgemm_output_with_deadline(
    output_len: usize,
    output_bytes: usize,
    site: &'static str,
    deadline: Instant,
) -> Result<Vec<f64>> {
    const HOST_INIT_CHUNK: usize = 4096;

    cuda_deadline_gemm_check(deadline)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: output_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    cuda_deadline_gemm_check(deadline)?;
    while output.len() < output_len {
        cuda_deadline_gemm_check(deadline)?;
        let end = output.len().saturating_add(HOST_INIT_CHUNK).min(output_len);
        output.resize(end, 0.0);
    }
    cuda_deadline_gemm_check(deadline)?;
    Ok(output)
}

/// Fill the bounded dispatch first across columns and then rows, while keeping
/// the complete contraction in every tile. Unlike the faer CPU deadline path,
/// there is deliberately no 64-row cache cap: it would turn the first CIFAR
/// convolution into roughly 16k synchronous CUDA launches.
fn cuda_deadline_f64_tile_shape(
    remaining_rows: usize,
    k: usize,
    remaining_cols: usize,
    max_dispatch_macs: usize,
) -> Option<(usize, usize)> {
    if remaining_rows == 0 || remaining_cols == 0 || max_dispatch_macs == 0 || k > max_dispatch_macs
    {
        return None;
    }
    let max_outputs = (max_dispatch_macs / k.max(1)).max(1);
    let cols = remaining_cols.min(max_outputs).max(1);
    let rows = remaining_rows.min((max_outputs / cols).max(1)).max(1);
    Some((rows, cols))
}

// Generate `gemm_f32` / `gemm_f64` over the type-specific cached buffer fields.
macro_rules! impl_cached_gemm {
    (
        $name:ident,
        $T:ty,
        $fa:ident,
        $fb:ident,
        $fc:ident,
        $device:ident,
        $one:expr,
        $zero:expr,
        $raw:ident
    ) => {
        fn $name(&self, m: usize, k: usize, n: usize, a: &[$T], b: &[$T]) -> Result<Vec<$T>> {
            const SITE: &str = concat!("cuda::", stringify!($name));
            let shape = validate_gemm_shape(m, k, n, size_of::<$T>(), SITE)?;
            if a.len() != shape.lhs_len {
                return Err(NyError::ShapeMismatch {
                    expected: vec![shape.lhs_len],
                    got: vec![a.len()],
                });
            }
            if b.len() != shape.rhs_len {
                return Err(NyError::ShapeMismatch {
                    expected: vec![shape.rhs_len],
                    got: vec![b.len()],
                });
            }
            if m == 0 || k == 0 || n == 0 {
                return allocate_gemm_output(shape, $zero, SITE);
            }
            // Explicit discrete-GPU mode keeps operands in cached device-only
            // allocations. The H2D copies, GEMM, and D2H copy share one ordered
            // stream and one final synchronization, avoiding HMM page migration
            // while preserving this method's exact arithmetic contract.
            if self.ordinary_gemm_transport == OrdinaryGemmTransport::ExplicitDeviceCopy {
                let cfg = gemm_cfg::<$T>(shape, $one, $zero);
                let mut output = allocate_gemm_output(shape, $zero, SITE)?;
                let mut g = self.inner.lock().expect("cublas mutex poisoned");
                let stream = g.stream.clone();
                ensure_ordinary_device_scratch(
                    &stream,
                    &mut g.$device,
                    [shape.lhs_len, shape.rhs_len, shape.output_len],
                )?;
                // A failure here precedes every borrowed-pointer enqueue, so
                // returning is safe. Once stage 0 begins, every exit passes
                // through the mandatory raw drain below.
                stream.context().bind_to_thread().map_err(cuda_err)?;
                let Inner { blas, $device, .. } = &mut *g;
                let transaction = queue_explicit_copy_gemm_and_drain(
                    |stage| match stage {
                        0 => stream
                            .memcpy_htod(a, $device.a.as_mut().expect("device A ensured"))
                            .map_err(cuda_err),
                        1 => stream
                            .memcpy_htod(b, $device.b.as_mut().expect("device B ensured"))
                            .map_err(cuda_err),
                        2 => {
                            let a_view = $device
                                .a
                                .as_ref()
                                .expect("device A ensured")
                                .slice(..shape.lhs_len);
                            let b_view = $device
                                .b
                                .as_ref()
                                .expect("device B ensured")
                                .slice(..shape.rhs_len);
                            let mut c_view = $device
                                .c
                                .as_mut()
                                .expect("device C ensured")
                                .slice_mut(..shape.output_len);
                            // SAFETY: validation fixed every view length/leading
                            // dimension; all operands remain live through the
                            // ordered D2H stage and mandatory drain below.
                            unsafe { blas.gemm(cfg, &b_view, &a_view, &mut c_view) }
                                .map_err(cuda_err)
                        }
                        3 => {
                            let c_view = $device
                                .c
                                .as_ref()
                                .expect("device C ensured")
                                .slice(..shape.output_len);
                            stream.memcpy_dtoh(&c_view, &mut output).map_err(cuda_err)
                        }
                        _ => unreachable!("four-stage explicit-copy GEMM"),
                    },
                    || stream.context().bind_to_thread().is_ok(),
                    || {
                        // SAFETY: the live guarded stream is bound immediately
                        // before this raw synchronization attempt.
                        unsafe {
                            cudarc::driver::result::stream::synchronize(stream.cu_stream()).is_ok()
                        }
                    },
                );
                let result =
                    queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
                drop(g);
                result?;
                return Ok(output);
            }
            // Host-page-table fast path (#p2-ats-zero-copy): on an authorized
            // device cuBLAS reads/writes the host slices directly — no
            // managed buffer, no H2D/D2H copy, no readback (measured ~2.2×/call).
            // BIT-IDENTICAL to the unified path below: same `cublas?gemm`, same
            // column-major swap (row-major `C = A·B` as `Cᵀ = Bᵀ·Aᵀ`, so pass B,A
            // swapped with op_N, ld = n,k,n), same pinned CUBLAS_DEFAULT_MATH mode.
            // cuBLAS selects its blocked/pairwise reduction by shape + math mode, never
            // by pointer residence, so the f64/f32 result is unchanged — and even if it
            // differed, the sound A·W bound is Higham order-independent.
            if self.ordinary_gemm_transport == OrdinaryGemmTransport::DirectHostPageTables {
                let alpha: $T = $one;
                let beta: $T = $zero;
                let mut c = allocate_gemm_output(shape, $zero, SITE)?;
                let g = self.inner.lock().expect("cublas mutex poisoned");
                // Safe to return before the raw launch has received any
                // borrowed pointer. The transaction below owns all later exits.
                g.stream.context().bind_to_thread().map_err(cuda_err)?;
                // Even a launch error can leave CUDA retaining the borrowed
                // host pointers. Queue the launch and unconditionally prove
                // stream quiescence before any Rust slice may be dropped.
                let transaction = queue_dgemms_and_drain(
                    1,
                    |_| {
                        // SAFETY: a/b/c are host slices of exactly m*k, k*n,
                        // m*n elements; ATS admission authorizes direct access;
                        // leading dimensions match the column-major swap. The
                        // transaction keeps all pointers live through its drain.
                        unsafe {
                            cudarc::cublas::result::$raw(
                                *g.blas.handle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                shape.n,
                                shape.m,
                                shape.k,
                                &raw const alpha,
                                b.as_ptr(),
                                shape.n,
                                a.as_ptr(),
                                shape.k,
                                &raw const beta,
                                c.as_mut_ptr(),
                                shape.n,
                            )
                            .map_err(cuda_err)
                        }
                    },
                    || g.stream.context().bind_to_thread().is_ok(),
                    || {
                        // SAFETY: the guarded stream was bound immediately
                        // before this raw synchronization attempt.
                        unsafe {
                            cudarc::driver::result::stream::synchronize(g.stream.cu_stream())
                                .is_ok()
                        }
                    },
                );
                let result =
                    queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
                drop(g);
                result?;
                return Ok(c);
            }
            let cfg = gemm_cfg::<$T>(shape, $one, $zero);
            let mut g = self.inner.lock().expect("cublas mutex poisoned");
            ensure_unified(&self.ctx, &mut g.$fa, shape.lhs_len)?;
            ensure_unified(&self.ctx, &mut g.$fb, shape.rhs_len)?;
            ensure_unified(&self.ctx, &mut g.$fc, shape.output_len)?;
            let Inner {
                blas,
                stream,
                $fa,
                $fb,
                $fc,
                ..
            } = &mut *g;
            let ua = $fa.as_mut().expect("ensured");
            ua.as_mut_slice().map_err(cuda_err)?[..shape.lhs_len].copy_from_slice(a);
            let ub = $fb.as_mut().expect("ensured");
            ub.as_mut_slice().map_err(cuda_err)?[..shape.rhs_len].copy_from_slice(b);
            let uc = $fc.as_mut().expect("ensured");
            // SAFETY: shapes/leading-dims validated; views are exactly m*k, k*n,
            // m*n; cuBLAS reads/writes strictly within them.
            unsafe {
                blas.gemm(
                    cfg,
                    &$fb.as_ref().unwrap().slice(..shape.rhs_len),
                    &$fa.as_ref().unwrap().slice(..shape.lhs_len),
                    &mut uc.slice_mut(..shape.output_len),
                )
                .map_err(cuda_err)?;
            }
            stream.synchronize().map_err(cuda_err)?;
            copy_gemm_output(
                &$fc.as_ref().unwrap().as_slice().map_err(cuda_err)?[..shape.output_len],
                shape,
                SITE,
            )
        }
    };
}

impl GemmEngine for CudaGemmEngine {
    fn backend_provenance(&self) -> &'static str {
        "cuda-cublas-rn"
    }

    impl_cached_gemm!(
        gemm_f32,
        f32,
        fa32,
        fb32,
        fc32,
        ordinary_device_f32,
        1.0f32,
        0.0f32,
        sgemm
    );
    impl_cached_gemm!(
        gemm_f64,
        f64,
        fa64,
        fb64,
        fc64,
        ordinary_device_f64,
        1.0f64,
        0.0f64,
        dgemm
    );

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
        max_dispatch_macs: usize,
    ) -> Result<Vec<f64>> {
        let started = Instant::now();
        cuda_deadline_gemm_check(deadline)?;
        let lhs_len = m.checked_mul(k).ok_or_else(|| {
            NyError::InvalidSpec("cuda deadline-bounded f64 GEMM: m*k overflow".into())
        })?;
        let rhs_len = k.checked_mul(n).ok_or_else(|| {
            NyError::InvalidSpec("cuda deadline-bounded f64 GEMM: k*n overflow".into())
        })?;
        let output_len = m.checked_mul(n).ok_or_else(|| {
            NyError::InvalidSpec("cuda deadline-bounded f64 GEMM: m*n overflow".into())
        })?;
        if a.len() != lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![lhs_len],
                got: vec![a.len()],
            });
        }
        if b.len() != rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![rhs_len],
                got: vec![b.len()],
            });
        }

        let output_bytes = output_len.checked_mul(size_of::<f64>()).ok_or_else(|| {
            NyError::InvalidSpec("cuda deadline-bounded f64 GEMM: output bytes overflow".into())
        })?;
        let dispatch_cap = max_dispatch_macs.min(CUDA_DEADLINE_F64_HARD_MAX_MACS);
        // Preserve the mathematical zero-contraction result. For every
        // nonempty GEMM, decline unsupported geometry before reserving the
        // potentially large output that the pollable CPU fallback would
        // allocate again. Every constructed engine has either the direct
        // host-page-table or explicit-device-copy transport.
        if m != 0 && k != 0 && n != 0 {
            if k > dispatch_cap || dispatch_cap == 0 {
                return Err(NyError::UnsupportedOp(format!(
                    "cuda deadline-bounded f64 GEMM: contraction {k} exceeds dispatch cap \
                     {dispatch_cap}"
                )));
            }
        }

        let mut output = allocate_dgemm_output_with_deadline(
            output_len,
            output_bytes,
            "cuda::gemm_f64_with_deadline/output",
            deadline,
        )?;
        if m == 0 || k == 0 || n == 0 {
            return Ok(output);
        }

        let mut i0 = 0usize;
        let mut dispatches = 0usize;
        while i0 < m {
            cuda_deadline_gemm_check(deadline)?;
            let (rows, max_cols) = cuda_deadline_f64_tile_shape(m - i0, k, n, dispatch_cap)
                .ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "cuda deadline-bounded f64 GEMM cannot form a bounded tile".into(),
                    )
                })?;
            let a_start = i0.checked_mul(k).expect("validated lhs geometry");
            let a_end = (i0 + rows).checked_mul(k).expect("validated lhs geometry");
            let a_tile = &a[a_start..a_end];

            let mut j0 = 0usize;
            while j0 < n {
                cuda_deadline_gemm_check(deadline)?;
                let cols = max_cols.min(n - j0);
                let tile_macs = rows
                    .checked_mul(k)
                    .and_then(|value| value.checked_mul(cols))
                    .expect("tile geometry is bounded by dispatch cap");
                debug_assert!(tile_macs <= dispatch_cap);

                let mut gathered_b = Vec::new();
                let b_tile: &[f64] = if j0 == 0 && cols == n {
                    b
                } else {
                    let gathered_len = k.checked_mul(cols).expect("bounded RHS tile");
                    let gathered_bytes =
                        gathered_len.checked_mul(size_of::<f64>()).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "cuda deadline-bounded f64 GEMM: RHS tile bytes overflow".into(),
                            )
                        })?;
                    gathered_b.try_reserve_exact(gathered_len).map_err(|_| {
                        NyError::CpuMemoryExceeded {
                            required_bytes: gathered_bytes,
                            budget_bytes: usize::MAX,
                            site: "cuda::gemm_f64_with_deadline/rhs_tile",
                        }
                    })?;
                    cuda_deadline_gemm_check(deadline)?;
                    for kk in 0..k {
                        let start = kk
                            .checked_mul(n)
                            .and_then(|base| base.checked_add(j0))
                            .expect("validated RHS geometry");
                        let row = &b[start..start + cols];
                        for chunk in row.chunks(4096) {
                            cuda_deadline_gemm_check(deadline)?;
                            gathered_b.extend_from_slice(chunk);
                        }
                    }
                    cuda_deadline_gemm_check(deadline)?;
                    &gathered_b
                };

                let tile = match self.deadline_f64_transport {
                    DeadlineF64Transport::DirectHostPageTables => {
                        self.gemm_f64_deadline_ats_tile(rows, k, cols, a_tile, b_tile, deadline)?
                    }
                    DeadlineF64Transport::ExplicitDeviceCopy => {
                        self.gemm_f64_deadline_device_tile(rows, k, cols, a_tile, b_tile, deadline)?
                    }
                };
                dispatches += 1;
                cuda_deadline_gemm_check(deadline)?;
                for row in 0..rows {
                    let src = &tile[row * cols..(row + 1) * cols];
                    let dst_start = (i0 + row)
                        .checked_mul(n)
                        .and_then(|base| base.checked_add(j0))
                        .expect("validated output geometry");
                    for (offset, chunk) in src.chunks(4096).enumerate() {
                        cuda_deadline_gemm_check(deadline)?;
                        let chunk_start = dst_start + offset * 4096;
                        output[chunk_start..chunk_start + chunk.len()].copy_from_slice(chunk);
                    }
                }
                j0 += cols;
            }
            i0 += rows;
        }
        cuda_deadline_gemm_check(deadline)?;
        record_cuda_deadline_f64_gemm(dispatches, started.elapsed().as_micros());
        // Even lock-free accounting is inside the return-boundary contract.
        cuda_deadline_gemm_check(deadline)?;
        Ok(output)
    }

    fn gemm_f64_pair_shared_rhs(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 2],
        b: &[f64],
    ) -> Result<[Vec<f64>; 2]> {
        let shape = validate_dgemm_pair_shared_rhs(m, k, n, a, b)?;
        if !self.ordinary_host_ptr_zero_copy() {
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
        if !enabled || !self.ordinary_host_ptr_zero_copy() {
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

        const SITE: &str = "cuda::gemm_f32_fast";
        let shape = validate_gemm_shape(m, k, n, size_of::<f32>(), SITE)?;
        if a.len() != shape.lhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.lhs_len],
                got: vec![a.len()],
            });
        }
        if b.len() != shape.rhs_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![shape.rhs_len],
                got: vec![b.len()],
            });
        }
        if m == 0 || k == 0 || n == 0 {
            return allocate_gemm_output(shape, 0.0f32, SITE);
        }
        if self.ordinary_gemm_transport == OrdinaryGemmTransport::ExplicitDeviceCopy {
            let mut output = allocate_gemm_output(shape, 0.0f32, SITE)?;
            let mut g = self.inner.lock().expect("cublas mutex poisoned");
            let stream = g.stream.clone();
            ensure_ordinary_device_scratch(
                &stream,
                &mut g.ordinary_device_f32,
                [shape.lhs_len, shape.rhs_len, shape.output_len],
            )?;
            // Safe to return before any borrowed pageable pointer is queued.
            // After stage 0, the transaction's raw drain is mandatory.
            stream.context().bind_to_thread().map_err(cuda_err)?;
            let Inner {
                blas,
                ordinary_device_f32,
                ..
            } = &mut *g;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let transaction = queue_explicit_copy_gemm_and_drain(
                |stage| match stage {
                    0 => stream
                        .memcpy_htod(a, ordinary_device_f32.a.as_mut().expect("device A ensured"))
                        .map_err(cuda_err),
                    1 => stream
                        .memcpy_htod(b, ordinary_device_f32.b.as_mut().expect("device B ensured"))
                        .map_err(cuda_err),
                    2 => {
                        let b_view = ordinary_device_f32
                            .b
                            .as_ref()
                            .expect("device B ensured")
                            .slice(..shape.rhs_len);
                        let a_view = ordinary_device_f32
                            .a
                            .as_ref()
                            .expect("device A ensured")
                            .slice(..shape.lhs_len);
                        let mut c_view = ordinary_device_f32
                            .c
                            .as_mut()
                            .expect("device C ensured")
                            .slice_mut(..shape.output_len);
                        let (b_ptr, _rec_b) = b_view.device_ptr(&stream);
                        let (a_ptr, _rec_a) = a_view.device_ptr(&stream);
                        let (c_ptr, _rec_c) = c_view.device_ptr_mut(&stream);
                        // SAFETY: operand views are the validated row-major
                        // shapes in cached device storage. The column-major
                        // swap and reduced compute type are identical to the
                        // established proposal path.
                        unsafe {
                            cudarc::cublas::result::gemm_ex(
                                *blas.handle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                shape.n,
                                shape.m,
                                shape.k,
                                (&raw const alpha).cast(),
                                b_ptr as *const _,
                                sys::cudaDataType_t::CUDA_R_32F,
                                shape.n,
                                a_ptr as *const _,
                                sys::cudaDataType_t::CUDA_R_32F,
                                shape.k,
                                (&raw const beta).cast(),
                                c_ptr as *mut _,
                                sys::cudaDataType_t::CUDA_R_32F,
                                shape.n,
                                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
                                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                            )
                            .map_err(cuda_err)
                        }
                    }
                    3 => {
                        let c_view = ordinary_device_f32
                            .c
                            .as_ref()
                            .expect("device C ensured")
                            .slice(..shape.output_len);
                        stream.memcpy_dtoh(&c_view, &mut output).map_err(cuda_err)
                    }
                    _ => unreachable!("four-stage explicit-copy GEMM"),
                },
                || stream.context().bind_to_thread().is_ok(),
                || {
                    // SAFETY: the live guarded stream is bound immediately
                    // before this raw synchronization attempt.
                    unsafe {
                        cudarc::driver::result::stream::synchronize(stream.cu_stream()).is_ok()
                    }
                },
            );
            let result = queued_gemm_numeric_result(transaction.launch_result, transaction.drain);
            drop(g);
            result?;
            return Ok(output);
        }
        // Row-major C = A·B as column-major Cᵀ = Bᵀ·Aᵀ (same swap as gemm_cfg).
        let mut g = self.inner.lock().expect("cublas mutex poisoned");
        ensure_unified(&self.ctx, &mut g.fa32, shape.lhs_len)?;
        ensure_unified(&self.ctx, &mut g.fb32, shape.rhs_len)?;
        ensure_unified(&self.ctx, &mut g.fc32, shape.output_len)?;
        let Inner {
            blas,
            stream,
            fa32,
            fb32,
            fc32,
            ..
        } = &mut *g;
        let ua = fa32.as_mut().expect("ensured");
        ua.as_mut_slice().map_err(cuda_err)?[..shape.lhs_len].copy_from_slice(a);
        let ub = fb32.as_mut().expect("ensured");
        ub.as_mut_slice().map_err(cuda_err)?[..shape.rhs_len].copy_from_slice(b);
        let uc = fc32.as_mut().expect("ensured");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        {
            let bview = fb32.as_ref().expect("ensured").slice(..shape.rhs_len);
            let aview = fa32.as_ref().expect("ensured").slice(..shape.lhs_len);
            let mut cview = uc.slice_mut(..shape.output_len);
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
                    shape.n,
                    shape.m,
                    shape.k,
                    (&raw const alpha).cast(),
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    shape.n,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    shape.k,
                    (&raw const beta).cast(),
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    shape.n,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(cuda_err)?;
            }
        }
        stream.synchronize().map_err(cuda_err)?;
        copy_gemm_output(
            &uc.as_slice().map_err(cuda_err)?[..shape.output_len],
            shape,
            SITE,
        )
    }

    /// Expose the f64-exact sound GPU-resident CROWN backward for verdict routing.
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

/// Call-local GEMM view for deadline-bounded small-row CROWN.
///
/// It carries no mutable backend-global state: ordinary CUDA callers continue
/// to use `CudaGemmEngine` directly, while this adapter alone redirects the
/// sound f64 triplet to the already-certified bounded ATS primitive.
struct DeadlineCrownGemm<'a> {
    inner: &'a CudaGemmEngine,
    deadline: Instant,
}

impl GemmEngine for DeadlineCrownGemm<'_> {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let _ = (m, k, n, a, b);
        Err(NyError::UnsupportedOp(
            "deadline-bounded small-row CROWN adapter refuses unbounded f32 GEMM".into(),
        ))
    }

    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        self.inner.gemm_f64_with_deadline(
            m,
            k,
            n,
            a,
            b,
            self.deadline,
            CUDA_DEADLINE_F64_HARD_MAX_MACS,
        )
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
        max_dispatch_macs: usize,
    ) -> Result<Vec<f64>> {
        self.inner.gemm_f64_with_deadline(
            m,
            k,
            n,
            a,
            b,
            deadline.min(self.deadline),
            max_dispatch_macs,
        )
    }

    fn poll_crown_backward_deadline(&self) -> Result<()> {
        cuda_deadline_gemm_check(self.deadline)
    }

    fn gemm_f64_triplet(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: [&[f64]; 3],
        b: [&[f64]; 3],
    ) -> Result<[Vec<f64>; 3]> {
        validate_dgemm_triplet(m, k, n, a, b)?;
        let bounded = |lhs: &[f64], rhs: &[f64]| {
            self.inner.gemm_f64_with_deadline(
                m,
                k,
                n,
                lhs,
                rhs,
                self.deadline,
                CUDA_DEADLINE_F64_HARD_MAX_MACS,
            )
        };
        Ok([
            bounded(a[0], b[0])?,
            bounded(a[1], b[1])?,
            bounded(a[2], b[2])?,
        ])
    }
}

impl GpuCrownBackward for CudaGemmEngine {
    fn provides_resident_patches_root_observer(&self) -> bool {
        true
    }

    fn observe_resident_patches_root_plan(
        &self,
        plan: &GpuResidentPatchesRootPlan,
    ) -> Result<GpuResidentPatchesRootObservation> {
        // M0 is deliberately metadata-only. Do not lock `inner`, touch the CUDA
        // context/stream, allocate, synchronize, or run an IEEE self-check here.
        // The engine was created by the ordinary caller before this capability
        // can be reached; merely acknowledging a plan must never become lazy
        // CUDA initialization.
        plan.validate(Instant::now())?;
        if !self.host_ptr_zero_copy() {
            return Ok(GpuResidentPatchesRootObservation {
                backend_ready: false,
                ..GpuResidentPatchesRootObservation::default()
            });
        }
        let observation = GpuResidentPatchesRootObservation {
            backend_ready: true,
            accepted_targets: plan.targets.len(),
            accepted_rows: plan.total_rows(),
            device_allocations: 0,
            cuda_dispatches: 0,
            bound_values_published: 0,
            verdict_mutations: 0,
        };
        if !observation.is_zero_authority() {
            return Err(NyError::InternalError(
                "resident Patches root M0 observer attempted authoritative work".into(),
            ));
        }
        // Refuse a plan that expired during host-only validation rather than
        // publishing a misleading engagement acknowledgement.
        plan.validate(Instant::now())?;
        Ok(observation)
    }

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

    fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
        true
    }

    fn crown_backward_gpu_resnet_sound_single_row_with_deadline(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        cuda_deadline_gemm_check(deadline)?;
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "deadline-bounded single-row resnet CROWN: empty segment list".into(),
            ));
        }
        if seed.num_specs != 1 {
            return Err(NyError::UnsupportedOp(
                "cuda deadline-bounded resnet sound CROWN requires exactly one specification row"
                    .into(),
            ));
        }
        let adapter = DeadlineCrownGemm {
            inner: self,
            deadline,
        };
        let result = sound_crown::crown_backward_gpu_resnet_sound_single_row_with_deadline_impl(
            &adapter,
            segments,
            seed,
            input_lower,
            input_upper,
        )?;
        cuda_deadline_gemm_check(deadline)?;
        Ok(result)
    }

    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
    }

    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        if seed.num_specs == 1 {
            return self.crown_backward_gpu_resnet_sound_single_row_with_deadline(
                segments,
                seed,
                input_lower,
                input_upper,
                frontier_abs,
                node_abs,
                deadline,
            );
        }
        cuda_deadline_gemm_check(deadline)?;
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "deadline-bounded rows resnet CROWN: empty segment list".into(),
            ));
        }
        if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&seed.num_specs) {
            return Err(NyError::InvalidSpec(format!(
                "cuda deadline-bounded rows resnet sound CROWN requires 1..={} specification rows; got {}",
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS, seed.num_specs
            )));
        }
        let adapter = DeadlineCrownGemm {
            inner: self,
            deadline,
        };
        let result = sound_crown::crown_backward_gpu_resnet_sound_bounded_rows_with_deadline_impl(
            &adapter,
            segments,
            seed,
            input_lower,
            input_upper,
            frontier_abs,
            node_abs,
        )?;
        cuda_deadline_gemm_check(deadline)?;
        Ok(result)
    }

    fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
        true
    }

    fn crown_joint_alpha_gradient_resident_with_deadline(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        deadline: Instant,
    ) -> Result<Vec<Vec<f32>>> {
        cuda_deadline_gemm_check(deadline)?;
        let adapter = DeadlineCrownGemm {
            inner: self,
            deadline,
        };
        let result = joint_alpha::crown_joint_alpha_gradient_with_deadline_impl(
            &adapter,
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
        )?;
        cuda_deadline_gemm_check(deadline)?;
        Ok(result)
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

    fn deadline_bounded_resnet_sound_beta_max_rows(&self) -> usize {
        DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
    }

    fn crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        cuda_deadline_gemm_check(deadline)?;
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "deadline-bounded beta resnet CROWN: empty segment list".into(),
            ));
        }
        if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&seed.num_specs) {
            return Err(NyError::InvalidSpec(format!(
                "cuda deadline-bounded beta resnet sound CROWN requires 2..={} \
                 specification rows; got {}",
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS, seed.num_specs
            )));
        }
        let adapter = DeadlineCrownGemm {
            inner: self,
            deadline,
        };
        let result =
            sound_crown::crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline_impl(
                &adapter,
                segments,
                seed,
                input_lower,
                input_upper,
                beta_signed,
                frontier_abs,
                node_abs,
            )?;
        cuda_deadline_gemm_check(deadline)?;
        Ok(result)
    }

    /// Observation-only, deadline-bounded Cut-CROWN fold on the actual CUDA f64
    /// resident path. The ordinary beta result remains the sole consumable
    /// baseline; a complete cut fold can only attach telemetry.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_cut_shadow(
        &self,
        policy: ResidentCutShadowPolicy,
        segments: &[ny_core::GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        carrier: Option<&ResidentLowerCutCarrier>,
        binding_row: usize,
        deadline: Instant,
    ) -> Result<ResidentCutShadowOutcome> {
        // Load-bearing off parity: exactly the historical call, before reading
        // carrier, binding row, explicit shadow deadline, or ATS capability.
        if policy == ResidentCutShadowPolicy::Disabled {
            let baseline = self.crown_backward_gpu_resnet_sound_beta(
                segments,
                seed,
                input_lower,
                input_upper,
                beta_signed,
                frontier_abs,
                node_abs,
            )?;
            return Ok(ResidentCutShadowOutcome::disabled(baseline));
        }

        // Production already retained its historical verdict result before
        // calling this observation. Keep this trait outcome self-contained by
        // replaying the same beta fold, but do so through the call-local
        // deadline adapter: a late or failed replay returns Err and the caller
        // keeps its historical result unchanged.
        cuda_deadline_gemm_check(deadline)?;
        let adapter = DeadlineCrownGemm {
            inner: self,
            deadline,
        };
        let baseline = sound_crown::crown_backward_gpu_resnet_sound_beta_impl(
            &adapter,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
        )?;
        cuda_deadline_gemm_check(deadline)?;
        let Some(carrier) = carrier else {
            return Ok(ResidentCutShadowOutcome::rejected(baseline));
        };
        if binding_row >= seed.num_specs
            || binding_row >= baseline.lower_bounds.len()
            || baseline.lower_bounds.len() != seed.num_specs
            || baseline.upper_bounds.len() != seed.num_specs
            || !carrier.has_nonzero_multiplier()
            || carrier.deadline() != deadline
            || Instant::now() >= deadline
        {
            return Ok(ResidentCutShadowOutcome::rejected(baseline));
        }
        if cuda_deadline_gemm_check(deadline).is_err() {
            return Ok(ResidentCutShadowOutcome::rejected(baseline));
        }

        let shadow = sound_crown::crown_backward_gpu_resnet_sound_beta_cut_shadow_impl(
            &adapter,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            carrier,
            deadline,
        );
        let shadow = match shadow {
            Ok(shadow) => shadow,
            Err(NyError::UnsupportedOp(_)) => {
                return Ok(ResidentCutShadowOutcome::backend_unavailable(baseline));
            }
            Err(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
        };
        if cuda_deadline_gemm_check(deadline).is_err()
            || shadow.lower_bounds.len() != seed.num_specs
            || shadow.upper_bounds.len() != seed.num_specs
            || shadow
                .lower_bounds
                .iter()
                .chain(&shadow.upper_bounds)
                .any(|value| !value.is_finite())
            // A lower-only carrier has no authority to perturb the upper
            // channel. Require exact replay identity before observing it.
            || shadow
                .upper_bounds
                .iter()
                .zip(&baseline.upper_bounds)
                .any(|(shadow, baseline)| shadow.to_bits() != baseline.to_bits())
        {
            return Ok(ResidentCutShadowOutcome::rejected(baseline));
        }

        let observation = match ResidentCutShadowObservation::try_new(
            binding_row,
            baseline.lower_bounds[binding_row],
            shadow.lower_bounds[binding_row],
        ) {
            Ok(observation) => observation,
            Err(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
        };
        match ResidentCutShadowOutcome::try_observed(baseline.clone(), observation) {
            Ok(outcome) => Ok(outcome),
            Err(_) => Ok(ResidentCutShadowOutcome::rejected(baseline)),
        }
    }

    fn provides_resident_cut_shadow(&self) -> bool {
        true
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

    /// CUDA coefficient authority is quarantined. The experimental differential
    /// replay retains a full primary carrier and can change GEMM shape at a
    /// final partial chunk, so it is not exposed until its peak-memory accounting
    /// and chunk-local parity proof are complete. Bound/gradient paths remain
    /// available.
    fn crown_backward_gpu_resnet_sound_beta_batched_coeff(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Result<(Vec<GpuCrownResult>, ny_core::GpuResidentCoeffBatched)> {
        let _ = (domains, seed);
        Err(NyError::SoundnessRefusal(
            "CUDA Complete Clip coefficient capture is quarantined pending chunk-local replay \
             and peak-memory certification"
                .into(),
        ))
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

    /// THE verdict-authority seam for CUDA. `sound_gpu_gate.rs` filters every
    /// verdict-carrying GPU CROWN route by this predicate.
    ///
    /// This used to be a hardwired `true` — asserted, never measured, and not
    /// connected to the construction-time IEEE probes it implicitly relied on.
    /// It now delegates to the per-device rung ladder in `sound_authority`,
    /// which is DARK (`NY_CUDA_AUTHORITY_LADDER=1`, default OFF). With the
    /// ladder disengaged this returns `true` without dispatching anything, so
    /// the default build is byte-identical to the old literal; engaging it can
    /// only ever REFUSE. See `sound_authority`'s module docs for why the CUDA
    /// gate's polarity is the mirror of wgpu's.
    fn provides_sound_gpu_crown(&self) -> bool {
        self.sound_gpu_authority()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_test_admission_is_fail_closed() {
        assert_eq!(
            cuda_test_admission(false, None),
            CudaTestAdmission::MissingRuntimeLibraries
        );
        assert_eq!(
            cuda_test_admission(false, Some(4)),
            CudaTestAdmission::MissingRuntimeLibraries,
            "a device count cannot compensate for a missing cuBLAS runtime"
        );
        assert_eq!(
            cuda_test_admission(true, None),
            CudaTestAdmission::DeviceProbeUnavailable
        );
        assert_eq!(
            cuda_test_admission(true, Some(-1)),
            CudaTestAdmission::NoVisibleDevice
        );
        assert_eq!(
            cuda_test_admission(true, Some(0)),
            CudaTestAdmission::NoVisibleDevice
        );
        assert_eq!(
            cuda_test_admission(true, Some(1)),
            CudaTestAdmission::Capable
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_candidates_are_the_exact_cudarc_arrays() {
        let observed = cuda_runtime_library_candidates();
        let expected_driver: Vec<_> = ["cuda", "nvcuda"]
            .iter()
            .flat_map(|name| cudarc::get_lib_name_candidates(name))
            .collect();
        assert_eq!(observed.driver, expected_driver);
        assert_eq!(observed.cublas, cudarc::get_lib_name_candidates("cublas"));
        assert_eq!(
            observed.cublas_lt,
            cudarc::get_lib_name_candidates("cublasLt")
        );
        assert_eq!(observed.nvrtc, cudarc::get_lib_name_candidates("nvrtc"));
        assert!(
            observed.cublas.iter().any(|name| name == "libcublas.so.13"),
            "cuda-13000 Linux candidate list must include the CUDA 13 SONAME"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_mapping_uses_the_symbol_address_not_a_decoy_name() {
        let maps = "\
7f000000-7f001000 r-xp 00000000 103:02 101 /attacker/libcuda.so.evil
7f100000-7f101000 r--p 00000000 103:02 202 /opt/provider/odd-name.so
7f101000-7f102000 r-xp 00001000 103:02 202 /opt/provider/odd-name.so
";
        let provider = parse_proc_maps_provider(maps, 0x7f101123).expect("symbol provider mapping");
        assert_eq!(
            provider.mapped_path,
            PathBuf::from("/opt/provider/odd-name.so")
        );
        assert_eq!(provider.device_major, 0x103);
        assert_eq!(provider.device_minor, 0x02);
        assert_eq!(provider.inode, 202);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_mapping_decodes_paths_and_rejects_deleted_objects() {
        let escaped = "\
7f000000-7f001000 r-xp 00000000 103:02 101 /opt/CUDA\\040Runtime/provider.bin
";
        let provider = parse_proc_maps_provider(escaped, 0x7f000123).expect("escaped path");
        assert_eq!(
            provider.mapped_path,
            PathBuf::from("/opt/CUDA Runtime/provider.bin")
        );

        let deleted = "\
7f000000-7f001000 r-xp 00000000 103:02 101 /opt/cuda/provider.bin (deleted)
";
        let error = parse_proc_maps_provider(deleted, 0x7f000123)
            .expect_err("deleted symbol provider must fail closed");
        assert!(error.to_string().contains("deleted or malformed"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_admission_rejects_injected_mixed_symbol_mappings() {
        use std::cell::RefCell;

        let maps = "\
7f100000-7f101000 r-xp 00000000 103:02 101 /sealed/libcuda-clean.so
7f200000-7f201000 r-xp 00000000 103:02 202 /attacker/libcuda-wrapper.so
";
        let symbol_addresses = [
            ("cuInit", 0x7f100123usize),
            ("cuMemcpyHtoD_v2", 0x7f200456usize),
        ];
        let confirmed = RefCell::new(Vec::new());
        let error = validate_role_symbol_addresses(
            maps,
            CudaRuntimeRole::Driver,
            &symbol_addresses,
            |role, symbol, address| {
                confirmed.borrow_mut().push((role, symbol, address));
                Ok(())
            },
        )
        .expect_err("forwarded representative plus wrapper copy symbol must be rejected");
        let detail = error.to_string();
        assert!(detail.contains("mixed driver providers"), "{detail}");
        assert!(detail.contains("cuInit"), "{detail}");
        assert!(detail.contains("cuMemcpyHtoD_v2"), "{detail}");
        assert!(detail.contains("/sealed/libcuda-clean.so"), "{detail}");
        assert!(detail.contains("/attacker/libcuda-wrapper.so"), "{detail}");
        assert_eq!(
            *confirmed.borrow(),
            vec![
                (CudaRuntimeRole::Driver, "cuInit", 0x7f100123),
                (
                    CudaRuntimeRole::Driver,
                    "cuMemcpyHtoD_v2",
                    0x7f200456
                ),
            ],
            "every injected address must pass the dladdr-confirmation seam before mapping admission"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_admission_accepts_one_inode_across_text_mappings() {
        use std::cell::Cell;

        let maps = "\
7f100000-7f101000 r-xp 00000000 103:02 101 /sealed/libcublas.so.13
7f102000-7f103000 r-xp 00002000 103:02 101 /sealed/libcublas.so.13
";
        let symbol_addresses = [
            ("cublasCreate_v2", 0x7f100123usize),
            ("cublasDgemm_v2", 0x7f102456usize),
        ];
        let confirmations = Cell::new(0usize);
        let provider = validate_role_symbol_addresses(
            maps,
            CudaRuntimeRole::Cublas,
            &symbol_addresses,
            |role, _, _| {
                assert_eq!(role, CudaRuntimeRole::Cublas);
                confirmations.set(confirmations.get() + 1);
                Ok(())
            },
        )
        .expect("multiple executable mappings of one inode are one provider");
        assert_eq!(confirmations.get(), symbol_addresses.len());
        assert_eq!(provider.device_major, 0x103);
        assert_eq!(provider.device_minor, 0x02);
        assert_eq!(provider.inode, 101);
        assert_eq!(
            provider.mapped_path,
            PathBuf::from("/sealed/libcublas.so.13")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_symbol_sets_have_stable_representatives_and_no_duplicates() {
        use std::collections::BTreeSet;

        for (role, symbols, representative) in [
            ("driver", DRIVER_PROVIDER_SYMBOLS, "cuInit"),
            ("cublas", CUBLAS_PROVIDER_SYMBOLS, "cublasCreate_v2"),
            (
                "cublas_lt",
                CUBLAS_LT_PROVIDER_SYMBOLS,
                "cublasLtGetVersion",
            ),
        ] {
            assert_eq!(symbols.first().copied(), Some(representative), "{role}");
            assert_eq!(
                symbols.iter().copied().collect::<BTreeSet<_>>().len(),
                symbols.len(),
                "{role} provider admission symbols must be unique"
            );
        }
        let all_cublas_lt = CUBLAS_LT_PROVIDER_SYMBOLS
            .iter()
            .chain(CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS)
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            all_cublas_lt.len(),
            CUBLAS_LT_PROVIDER_SYMBOLS.len() + CUBLAS_LT_DELEGATION_PROVIDER_SYMBOLS.len(),
            "stable and optional cuBLASLt provider symbols must be disjoint"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_admits_absent_or_complete_delegation_pair_only() {
        assert!(select_cublas_lt_delegation_addresses([None, None])
            .expect("older CUDA-13 minors may omit private delegation exports")
            .is_empty());
        assert_eq!(
            select_cublas_lt_delegation_addresses([Some(0x1000), Some(0x2000)])
                .expect("a complete delegation pair is provenance-bound"),
            vec![
                ("cublasLt_for_cublas_DDD", 0x1000),
                ("cublasLt_for_cublas_SSS", 0x2000),
            ]
        );
        for partial in [[Some(0x1000), None], [None, Some(0x2000)]] {
            let error = select_cublas_lt_delegation_addresses(partial)
                .expect_err("a half-present delegation surface must fail closed");
            assert!(
                error.to_string().contains("both present or both absent"),
                "{error}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_hash_is_fd_bound_and_complete() {
        use std::os::unix::fs::MetadataExt as _;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("Cargo.toml")
            .canonicalize()
            .expect("canonical manifest");
        let metadata = path.metadata().expect("manifest metadata");
        let (device_major, device_minor) = linux_device_numbers(metadata.dev());
        let identity = hash_provider_file(
            CudaRuntimeRole::Driver,
            "fixture_symbol",
            ProcMapProvider {
                mapped_path: path.clone(),
                device_major,
                device_minor,
                inode: metadata.ino(),
            },
        )
        .expect("FD-bound provider hash");
        let bytes = std::fs::read(path).expect("read manifest independently");
        assert_eq!(
            identity.sha256,
            format!("{:x}", Sha256::digest(bytes)),
            "identity must hash the complete provider bytes"
        );
        assert_eq!(identity.provider_symbol, "fixture_symbol");
        assert_eq!(identity.size_bytes, metadata.size());
        assert_eq!(identity.fingerprint.device, metadata.dev());
        assert_eq!(identity.fingerprint.inode, metadata.ino());
        assert_eq!(identity.fingerprint.size_bytes, metadata.size());

        let mismatch = hash_provider_file(
            CudaRuntimeRole::Driver,
            "fixture_symbol",
            ProcMapProvider {
                mapped_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
                device_major,
                device_minor,
                inode: metadata.ino() + 1,
            },
        )
        .expect_err("a decoy inode must not satisfy the provider mapping");
        assert!(mismatch.to_string().contains("does not match"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_runtime_provider_binding_follows_a_real_symbol_to_its_open_file() {
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read process maps");
        let address = (libc::malloc as *const ()) as usize;
        let identity = capture_symbol_provider(
            &maps,
            CudaRuntimeRole::Driver,
            "malloc_test_fixture",
            address,
        )
        .expect("bind libc malloc to its exact provider");
        assert_eq!(identity.provider_symbol, "malloc_test_fixture");
        assert_eq!(identity.mapped_inode, identity.fingerprint.inode);
        assert_eq!(identity.size_bytes, identity.fingerprint.size_bytes);
        assert_eq!(identity.sha256.len(), 64);
        assert!(identity.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    fn force_explicit_device_cuda_transport(engine: &mut CudaGemmEngine) {
        // Force path coverage even when this test happens to run on coherent
        // host-page-table hardware.
        engine.deadline_f64_transport = DeadlineF64Transport::ExplicitDeviceCopy;
        assert_eq!(engine.deadline_f64_transport_name(), "explicit-device-copy");
    }

    #[test]
    fn deadline_f64_transport_requires_both_pageable_capabilities_for_direct_access() {
        assert_eq!(
            select_deadline_f64_transport(true, true),
            DeadlineF64Transport::DirectHostPageTables
        );
        for (pageable, host_page_tables) in [(true, false), (false, true), (false, false)] {
            assert_eq!(
                select_deadline_f64_transport(pageable, host_page_tables),
                DeadlineF64Transport::ExplicitDeviceCopy,
                "capabilities pageable={pageable} host_page_tables={host_page_tables}"
            );
        }
    }

    #[test]
    fn cuda_transport_policy_is_exact_capability_driven_and_fail_closed() {
        use std::ffi::OsStr;

        assert!(!cuda_discrete_mode_requested(None).expect("unset is default-off"));
        assert!(!cuda_discrete_mode_requested(Some(OsStr::new("0"))).expect("exact zero"));
        assert!(cuda_discrete_mode_requested(Some(OsStr::new("1"))).expect("exact one"));
        for invalid in ["", " 1", "1 ", "+1", "true", "01", "１"] {
            let error = cuda_discrete_mode_requested(Some(OsStr::new(invalid)))
                .expect_err("malformed opt-in must fail closed");
            assert!(error.to_string().contains(CUDA_DISCRETE_MODE_ENV));
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
            assert!(cuda_discrete_mode_requested(Some(&non_unicode)).is_err());
            assert!(parse_ordinary_gemm_transport_policy(Some(&non_unicode), None).is_err());
        }

        assert_eq!(
            parse_ordinary_gemm_transport_policy(None, None).expect("unset selects auto"),
            OrdinaryGemmTransportPolicy::Auto
        );
        assert_eq!(
            parse_ordinary_gemm_transport_policy(Some(OsStr::new("auto")), None)
                .expect("exact auto"),
            OrdinaryGemmTransportPolicy::Auto
        );
        assert_eq!(
            parse_ordinary_gemm_transport_policy(
                Some(OsStr::new("direct-host-page-tables")),
                None,
            )
            .expect("exact direct override"),
            OrdinaryGemmTransportPolicy::ForceDirectHostPageTables
        );
        assert_eq!(
            parse_ordinary_gemm_transport_policy(Some(OsStr::new("unified-memory")), None)
                .expect("exact unified override"),
            OrdinaryGemmTransportPolicy::ForceUnifiedMemory
        );
        assert_eq!(
            parse_ordinary_gemm_transport_policy(Some(OsStr::new("explicit-device-copy")), None,)
                .expect("exact explicit override"),
            OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy
        );
        for invalid in ["", "explicit", "discrete", "AUTO", " auto", "auto "] {
            let error = parse_ordinary_gemm_transport_policy(Some(OsStr::new(invalid)), None)
                .expect_err("malformed transport policy must fail closed");
            assert!(error.to_string().contains(CUDA_GEMM_TRANSPORT_ENV));
        }

        assert_eq!(
            parse_ordinary_gemm_transport_policy(None, Some(OsStr::new("1")))
                .expect("legacy opt-in remains supported"),
            OrdinaryGemmTransportPolicy::LegacyForceExplicitDeviceCopy
        );
        assert_eq!(
            parse_ordinary_gemm_transport_policy(
                Some(OsStr::new("explicit-device-copy")),
                Some(OsStr::new("1")),
            )
            .expect("matching overrides are allowed"),
            OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy
        );
        for conflict in ["direct-host-page-tables", "unified-memory"] {
            let error = parse_ordinary_gemm_transport_policy(
                Some(OsStr::new(conflict)),
                Some(OsStr::new("1")),
            )
            .expect_err("conflicting legacy and current overrides must refuse");
            assert!(error.to_string().contains("conflicts"));
        }

        assert_eq!(
            select_ordinary_gemm_transport(
                OrdinaryGemmTransportPolicy::Auto,
                true,
                true,
                Some(false),
            )
            .expect("ATS auto selection")
            .transport,
            OrdinaryGemmTransport::DirectHostPageTables
        );
        let integrated = select_ordinary_gemm_transport(
            OrdinaryGemmTransportPolicy::Auto,
            true,
            false,
            Some(true),
        )
        .expect("integrated auto selection");
        assert_eq!(integrated.transport, OrdinaryGemmTransport::UnifiedMemory);
        assert_eq!(
            integrated.reason,
            OrdinaryGemmTransportReason::IntegratedDevice
        );

        let discrete = select_ordinary_gemm_transport(
            OrdinaryGemmTransportPolicy::Auto,
            true,
            false,
            Some(false),
        )
        .expect("discrete auto selection");
        assert_eq!(
            discrete.transport,
            OrdinaryGemmTransport::ExplicitDeviceCopy
        );
        assert_eq!(discrete.reason, OrdinaryGemmTransportReason::DiscreteDevice);

        let unknown =
            select_ordinary_gemm_transport(OrdinaryGemmTransportPolicy::Auto, false, false, None)
                .expect("unknown topology uses safe explicit-copy fallback");
        assert_eq!(unknown.transport, OrdinaryGemmTransport::ExplicitDeviceCopy);
        assert_eq!(unknown.reason, OrdinaryGemmTransportReason::UnknownTopology);

        assert!(select_ordinary_gemm_transport(
            OrdinaryGemmTransportPolicy::ForceDirectHostPageTables,
            true,
            false,
            Some(false),
        )
        .is_err());
        assert_eq!(
            select_ordinary_gemm_transport(
                OrdinaryGemmTransportPolicy::ForceUnifiedMemory,
                false,
                false,
                Some(false),
            )
            .expect("unified A/B override")
            .transport,
            OrdinaryGemmTransport::UnifiedMemory
        );
        for policy in [
            OrdinaryGemmTransportPolicy::ForceExplicitDeviceCopy,
            OrdinaryGemmTransportPolicy::LegacyForceExplicitDeviceCopy,
        ] {
            assert_eq!(
                select_ordinary_gemm_transport(policy, true, true, Some(true))
                    .expect("explicit override takes precedence")
                    .transport,
                OrdinaryGemmTransport::ExplicitDeviceCopy
            );
        }
    }

    /// Exercise the public environment option in an isolated process. Engine
    /// construction runs the IEEE f32/f64 KATs through the selected transport;
    /// this additionally covers the tensor-core proposal method on that cache.
    #[test]
    fn cuda_discrete_mode_qualifies_selected_transport_when_hardware_is_capable() {
        const CHILD_MARKER: &str = "NY_CUDA_DISCRETE_MODE_TEST_CHILD";
        const TEST_NAME: &str =
            "tests::cuda_discrete_mode_qualifies_selected_transport_when_hardware_is_capable";

        if std::env::var_os(CHILD_MARKER).as_deref() != Some(std::ffi::OsStr::new("1")) {
            let output = std::process::Command::new(
                std::env::current_exe().expect("locate ny-cuda unit-test executable"),
            )
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_MARKER, "1")
            .env(CUDA_DISCRETE_MODE_ENV, "1")
            .env("NY_CUDA_DGEMM_TRIPLET", "1")
            .output()
            .expect("spawn isolated CUDA discrete-mode child");
            assert!(
                output.status.success(),
                "CUDA discrete-mode child failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        assert_eq!(
            std::env::var_os(CUDA_DISCRETE_MODE_ENV).as_deref(),
            Some(std::ffi::OsStr::new("1")),
            "child must enter through the exact public option"
        );
        with_capable_cuda(|engine| {
            assert!(engine.discrete_mode_enabled());
            assert_eq!(
                engine.ordinary_gemm_transport_name(),
                "explicit-device-copy"
            );
            assert_eq!(
                engine.ordinary_gemm_transport_policy_name(),
                "legacy-discrete-mode-override"
            );
            assert_eq!(
                engine.ordinary_gemm_transport_reason(),
                "legacy-discrete-mode-override"
            );
            assert!(
                !engine.ordinary_host_ptr_zero_copy(),
                "discrete mode must suppress ordinary ATS pair/triplet dispatch"
            );
            engine
                .assert_ieee_bit_exact()
                .expect("selected explicit-copy f32/f64 path must remain bit-exact");

            let got = engine
                .gemm_f32_fast(2, 2, 2, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
                .expect("explicit-copy tensor-core proposal GEMM");
            assert_eq!(got.len(), 4);
            assert!(got.iter().all(|value| value.is_finite()));
            for (value, expected) in got.iter().zip([19.0f32, 22.0, 43.0, 50.0]) {
                assert!(
                    (value - expected).abs() <= 0.5,
                    "proposal GEMM layout/copy mismatch: got {value}, want about {expected}"
                );
            }

            let lhs = [[2.0f64], [3.0], [5.0]];
            let rhs = [[7.0f64], [11.0], [13.0]];
            let pair = engine
                .gemm_f64_pair_shared_rhs(1, 1, 1, [&lhs[0], &lhs[1]], &rhs[0])
                .expect("discrete-mode shared-RHS pair");
            assert_eq!(pair, [vec![14.0], vec![21.0]]);
            let triplet = engine
                .gemm_f64_triplet(
                    1,
                    1,
                    1,
                    [&lhs[0], &lhs[1], &lhs[2]],
                    [&rhs[0], &rhs[1], &rhs[2]],
                )
                .expect("discrete-mode triplet must use scalar explicit-copy transactions");
            assert_eq!(triplet, [vec![14.0], vec![33.0], vec![65.0]]);
        });
    }

    #[derive(Debug, Eq, PartialEq)]
    struct MockDeadlineDeviceAllocation {
        id: usize,
        len: usize,
    }

    #[test]
    fn deadline_device_scratch_deadline_does_not_publish_unready_growth() {
        use std::cell::{Cell, RefCell};

        let mut scratch = DeadlineDeviceScratch {
            a: Some(MockDeadlineDeviceAllocation { id: 1, len: 4 }),
            b: None,
            c: Some(MockDeadlineDeviceAllocation { id: 3, len: 4 }),
        };
        let polls = Cell::new(0usize);
        let next_id = Cell::new(10usize);
        let allocation_lengths = RefCell::new(Vec::new());
        let drains = Cell::new(0usize);

        let error = grow_deadline_device_scratch_transactionally(
            &mut scratch,
            [8, 8, 4],
            |allocation| allocation.len,
            || {
                let poll = polls.get();
                polls.set(poll + 1);
                if poll == 0 {
                    Ok(())
                } else {
                    Err(NyError::DeadlineExceeded(
                        "injected deadline during scratch growth".into(),
                    ))
                }
            },
            |len| {
                allocation_lengths.borrow_mut().push(len);
                let id = next_id.get();
                next_id.set(id + 1);
                Ok(MockDeadlineDeviceAllocation { id, len })
            },
            || {
                drains.set(drains.get() + 1);
                Ok(())
            },
        )
        .expect_err("deadline after the first async allocation must abort growth");
        assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        assert_eq!(drains.get(), 1, "attempted allocation must be drained");
        assert_eq!(*allocation_lengths.borrow(), vec![8]);
        assert_eq!(scratch.a.as_ref().map(|value| value.id), Some(1));
        assert!(scratch.b.is_none());
        assert_eq!(scratch.c.as_ref().map(|value| value.id), Some(3));

        grow_deadline_device_scratch_transactionally(
            &mut scratch,
            [8, 8, 4],
            |allocation| allocation.len,
            || Ok(()),
            |len| {
                allocation_lengths.borrow_mut().push(len);
                let id = next_id.get();
                next_id.set(id + 1);
                Ok(MockDeadlineDeviceAllocation { id, len })
            },
            || {
                drains.set(drains.get() + 1);
                Ok(())
            },
        )
        .expect("a later call must retry and publish only drained replacements");

        assert_eq!(
            *allocation_lengths.borrow(),
            vec![8, 8, 8],
            "the aborted A allocation must not satisfy the next call"
        );
        assert_eq!(drains.get(), 2);
        assert_eq!(
            scratch.a.as_ref(),
            Some(&MockDeadlineDeviceAllocation { id: 11, len: 8 })
        );
        assert_eq!(
            scratch.b.as_ref(),
            Some(&MockDeadlineDeviceAllocation { id: 12, len: 8 })
        );
        assert_eq!(
            scratch.c.as_ref(),
            Some(&MockDeadlineDeviceAllocation { id: 3, len: 4 })
        );
    }

    #[test]
    fn deadline_device_scratch_allocation_fault_preserves_ready_cache() {
        use std::cell::Cell;

        let mut scratch = DeadlineDeviceScratch {
            a: Some(MockDeadlineDeviceAllocation { id: 1, len: 4 }),
            b: Some(MockDeadlineDeviceAllocation { id: 2, len: 4 }),
            c: Some(MockDeadlineDeviceAllocation { id: 3, len: 4 }),
        };
        let attempts = Cell::new(0usize);
        let drains = Cell::new(0usize);

        let error = grow_deadline_device_scratch_transactionally(
            &mut scratch,
            [8, 4, 4],
            |allocation| allocation.len,
            || Ok(()),
            |len| {
                attempts.set(attempts.get() + 1);
                Err(NyError::InternalError(format!(
                    "injected allocation fault for {len} elements"
                )))
            },
            || {
                drains.set(drains.get() + 1);
                Ok(())
            },
        )
        .expect_err("allocation fault must abort growth");
        assert!(error.to_string().contains("injected allocation fault"));
        assert_eq!(attempts.get(), 1);
        assert_eq!(drains.get(), 1, "allocation API errors must still drain");
        assert_eq!(
            scratch.a.as_ref(),
            Some(&MockDeadlineDeviceAllocation { id: 1, len: 4 })
        );

        grow_deadline_device_scratch_transactionally(
            &mut scratch,
            [8, 4, 4],
            |allocation| allocation.len,
            || Ok(()),
            |len| {
                attempts.set(attempts.get() + 1);
                Ok(MockDeadlineDeviceAllocation { id: 10, len })
            },
            || {
                drains.set(drains.get() + 1);
                Ok(())
            },
        )
        .expect("a later call must retry after an allocation fault");
        assert_eq!(attempts.get(), 2);
        assert_eq!(drains.get(), 2);
        assert_eq!(
            scratch.a.as_ref(),
            Some(&MockDeadlineDeviceAllocation { id: 10, len: 8 })
        );
    }

    #[test]
    fn device_count_probe_contains_errors_and_panics() {
        assert_eq!(contain_device_count(|| Ok::<i32, ()>(2)), Some(2));
        assert_eq!(contain_device_count(|| Err::<i32, ()>(())), None);
        assert_eq!(
            contain_device_count(|| -> std::result::Result<i32, ()> {
                panic!("synthetic dynamic-loader panic")
            }),
            None
        );
    }

    #[test]
    fn deadline_f64_tile_shape_covers_without_overlap_and_respects_cap() {
        let (m, k, n, cap) = (7usize, 3usize, 11usize, 50usize);
        let mut covered = vec![false; m * n];
        let mut i0 = 0usize;
        while i0 < m {
            let (rows, max_cols) =
                cuda_deadline_f64_tile_shape(m - i0, k, n, cap).expect("bounded row tile");
            let mut j0 = 0usize;
            while j0 < n {
                let cols = max_cols.min(n - j0);
                assert!(rows * k * cols <= cap);
                for i in i0..i0 + rows {
                    for j in j0..j0 + cols {
                        let cell = &mut covered[i * n + j];
                        assert!(!*cell, "output tile overlap at ({i},{j})");
                        *cell = true;
                    }
                }
                j0 += cols;
            }
            i0 += rows;
        }
        assert!(covered.into_iter().all(|cell| cell));
        assert!(cuda_deadline_f64_tile_shape(1, cap + 1, 1, cap).is_none());
    }

    #[test]
    fn deadline_f64_cuda_tiles_fill_cap_instead_of_using_cpu_row_cap() {
        let (m, k, n, cap) = (
            1_048_576usize,
            27usize,
            16usize,
            CUDA_DEADLINE_F64_HARD_MAX_MACS,
        );
        let (rows, cols) =
            cuda_deadline_f64_tile_shape(m, k, n, cap).expect("CIFAR convolution tile");
        assert_eq!(cols, n);
        assert!(rows > 64, "CUDA tile must not inherit the CPU 64-row cap");
        assert!(rows * k * cols <= cap);
        assert!((rows + 1) * k * cols > cap);
        assert!(m.div_ceil(rows) < 64, "too many synchronous CUDA launches");
    }

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
    fn single_gemm_validation_refuses_product_overflow_and_i32_truncation() {
        let huge = 1usize << (usize::BITS - 1);
        assert!(
            validate_gemm_shape(huge, huge, huge, size_of::<f32>(), "cuda::gemm_f32").is_err(),
            "wrapped shape products must be rejected before CUDA access"
        );

        let too_wide = usize::try_from(i32::MAX).expect("usize holds i32") + 1;
        let error = validate_gemm_shape(too_wide, 0, 0, size_of::<f64>(), "cuda::gemm_f64")
            .expect_err("truncating even an empty cuBLAS dimension must be refused");
        assert!(error.to_string().contains("m exceeds i32"));
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
    fn explicit_copy_transaction_drains_every_stage_failure_before_return() {
        use std::cell::{Cell, RefCell};

        for fail_at in 0..4usize {
            let attempts = RefCell::new(Vec::new());
            let drains = Cell::new(0usize);
            let transaction = queue_explicit_copy_gemm_and_drain(
                |stage| {
                    attempts.borrow_mut().push(stage);
                    if stage == fail_at {
                        Err(NyError::InternalError(format!(
                            "injected explicit-copy stage {stage} failure"
                        )))
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

            assert_eq!(
                transaction.calls,
                fail_at + 1,
                "no stage after the injected failure may be queued"
            );
            assert_eq!(
                *attempts.borrow(),
                (0..=fail_at).collect::<Vec<_>>(),
                "the transaction must preserve H2D/H2D/GEMM/D2H order"
            );
            assert_eq!(drains.get(), 1, "every failure must drain exactly once");
            let error = queued_gemm_numeric_result(transaction.launch_result, transaction.drain)
                .expect_err("a drained stage failure must still reject numeric output");
            assert!(
                error
                    .to_string()
                    .contains(&format!("explicit-copy stage {fail_at} failure")),
                "unexpected error: {error}"
            );
        }

        let drains = Cell::new(0usize);
        let success = queue_explicit_copy_gemm_and_drain(
            |_| Ok(()),
            || true,
            || {
                drains.set(drains.get() + 1);
                true
            },
        );
        assert_eq!(success.calls, 4);
        assert_eq!(drains.get(), 1, "the happy path has one final drain");
        queued_gemm_numeric_result(success.launch_result, success.drain)
            .expect("four queued stages plus a first-attempt drain are publishable");
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
    fn dgemm_bind_retry_proves_quiescence_but_rejects_numeric_result() {
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
        let error = queued_gemm_numeric_result(transaction.launch_result, transaction.drain)
            .expect_err("a failed post-launch bind must taint numeric output");
        assert!(
            matches!(error, NyError::InternalError(_)),
            "unexpected error: {error}"
        );
        assert!(error.to_string().contains("numeric output is unusable"));
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
    /// Non-ATS hosts prove selection of the explicit-copy fallback through the
    /// shared capability seam.
    #[test]
    fn cuda_dgemm_triplet_matches_serial_bits_when_ats_is_capable() {
        const CHILD_MARKER: &str = "NY_CUDA_DGEMM_TRIPLET_PARITY_CHILD";
        const TEST_NAME: &str = "tests::cuda_dgemm_triplet_matches_serial_bits_when_ats_is_capable";

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
        } else {
            assert_eq!(
                std::env::var_os("NY_CUDA_DGEMM_TRIPLET").as_deref(),
                Some(std::ffi::OsStr::new("1")),
                "parity child must enter through the exact public gate"
            );

            with_capable_ats_cuda(|engine| {
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
            });
        }
    }

    /// On ATS hardware, the shared-RHS pair transaction must be bit-identical
    /// to two serial pinned-IEEE Dgemm calls at a non-square shape.
    #[test]
    fn cuda_dgemm_pair_shared_rhs_matches_serial_when_ats_is_capable() {
        with_capable_ats_cuda(|engine| {
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
        });
    }

    /// gemm_f32_fast (tensor-core BF16-split) must stay close to exact f32 —
    /// loose tolerance by design (reduced precision is its contract), but a
    /// wildly wrong result would mean a broken pointer/layout in the raw
    /// cublasGemmEx call, not acceptable even for attack use.
    #[test]
    fn cuda_gemm_f32_fast_approximates_exact_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
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
        });
    }

    /// Pageable host-pointer zero-copy path (#p2-ats-zero-copy) must give results
    /// that match a CPU f64 GEMM to tight tolerance. When direct host-page-table
    /// access is selected,
    /// `gemm_f64` routes through raw-host-pointer `dgemm`; this test requires that
    /// capability so a unified fallback cannot masquerade as path coverage.
    #[test]
    fn cuda_gemm_f64_ats_zero_copy_matches_cpu_when_capable() {
        with_capable_ats_cuda(|engine| {
            eprintln!(
                "CUDA device {:?}: host_ptr_zero_copy = {} (pageable path EXERCISED)",
                engine.device_name(),
                engine.host_ptr_zero_copy(),
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
            // f64 GEMM vs the naive CPU triple loop: both IEEE-f64, different
            // reduction order — a few ULPs at most on these magnitudes.
            assert!(
                max_abs < 1e-9,
                "ATS/unified f64 gemm diverged from CPU: max_abs={max_abs}"
            );
        });
    }

    #[test]
    fn cuda_deadline_bounded_f64_tiles_match_cpu_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            force_explicit_device_cuda_transport(engine);
            let (m, k, n) = (9usize, 4usize, 6usize);
            let a: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.125 - 1.0).collect();
            let b: Vec<f64> = (0..k * n).map(|i| i as f64 * -0.25 + 0.5).collect();
            // cap=64 forces five full-k output tiles for this 216-MAC product.
            let got = engine
                .gemm_f64_with_deadline(
                    m,
                    k,
                    n,
                    &a,
                    &b,
                    Instant::now() + std::time::Duration::from_secs(5),
                    64,
                )
                .expect("bounded tiled DGEMM");
            let expected = cpu_gemm_f64(m, k, n, &a, &b);
            for (got, expected) in got.iter().zip(&expected) {
                assert!((got - expected).abs() < 1e-9);
            }

            let error = engine
                .gemm_f64_with_deadline(
                    1,
                    k,
                    1,
                    &a[..k],
                    &b[..k],
                    Instant::now() + std::time::Duration::from_secs(1),
                    k - 1,
                )
                .expect_err("one full contraction cannot fit below the cap");
            assert!(matches!(error, NyError::UnsupportedOp(_)));
        });
    }

    #[test]
    fn cuda_deadline_explicit_copy_matches_ordinary_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            let (m, k, n) = (37usize, 65usize, 29usize);
            let a: Vec<f64> = (0..m * k)
                .map(|i| ((i % 73) as f64) * 0.03125 - 1.0625)
                .collect();
            let b: Vec<f64> = (0..k * n)
                .map(|i| ((i % 67) as f64) * -0.0234375 + 0.78125)
                .collect();
            let ordinary = engine
                .gemm_f64(m, k, n, &a, &b)
                .expect("ordinary pinned-IEEE DGEMM");
            force_explicit_device_cuda_transport(engine);
            let explicit = engine
                .gemm_f64_with_deadline(
                    m,
                    k,
                    n,
                    &a,
                    &b,
                    Instant::now() + std::time::Duration::from_secs(5),
                    m * k * n,
                )
                .expect("explicit-device-copy DGEMM");
            assert!(
                ordinary
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(explicit.iter().map(|value| value.to_bits())),
                "pointer residence changed the pinned-IEEE DGEMM result"
            );
            eprintln!(
                "CUDA device {:?}: deadline_f64_transport=explicit-device-copy \
                 (device-copy path EXERCISED)",
                engine.device_name()
            );
        });
    }

    #[test]
    fn cuda_deadline_mutex_contention_expires_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            force_explicit_device_cuda_transport(engine);
            let guard = engine.inner.lock().expect("test owns CUDA mutex");
            let error = engine
                .gemm_f64_with_deadline(
                    1,
                    1,
                    1,
                    &[2.0],
                    &[3.0],
                    Instant::now() + std::time::Duration::from_millis(10),
                    16,
                )
                .expect_err("contended deadline lock must expire");
            drop(guard);
            assert!(error.is_deadline_exceeded(), "unexpected error: {error}");
        });
    }

    #[test]
    fn cuda_deadline_joint_alpha_dispatches_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            force_explicit_device_cuda_transport(engine);
            assert!(
                engine.provides_deadline_bounded_joint_alpha_gradient_resident(),
                "explicit-device CUDA must advertise the method-specific joint capability"
            );
            assert!(
                !engine.honors_crown_backward_deadline(),
                "the narrow joint capability must not broaden CUDA's global claim"
            );

            let segments = vec![ny_core::GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(vec![0.7, -0.2, 0.4, 0.9, -0.1, 0.8, 0.3, -0.6]),
                    bias: Some(Arc::from(vec![0.1, -0.3])),
                    out_features: 2,
                    in_features: 4,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Activation {
                    lower_slope: vec![0.25, 0.8, 0.4, 0.65],
                    upper_slope: vec![0.6, 0.7, 0.55, 0.75],
                    lower_intercept: vec![0.0; 4],
                    upper_intercept: vec![0.3, 0.2, 0.1, 0.4],
                    num_neurons: 4,
                },
                GpuCrownLayer::Conv2d {
                    weight_col: Arc::from(vec![1.25]),
                    bias_expanded: Some(Arc::from(vec![0.05, -0.1, 0.2, 0.15])),
                    out_channels: 1,
                    in_channels: 1,
                    kernel_h: 1,
                    kernel_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                    out_h: 2,
                    out_w: 2,
                    in_h: 2,
                    in_w: 2,
                    cert_err: Default::default(),
                },
            ])];
            let seed = [1.0f32, -0.75];
            let input_lower = [-1.0f32, -0.5, -0.25, -0.8];
            let input_upper = [0.8f32, 1.2, 0.9, 0.6];
            let before = cuda_deadline_f64_gemm_stats();
            let got = engine
                .crown_joint_alpha_gradient_resident_with_deadline(
                    &segments,
                    &seed,
                    1,
                    2,
                    &input_lower,
                    &input_upper,
                    Instant::now() + std::time::Duration::from_secs(10),
                )
                .expect("deadline-transport CUDA joint adjoint");
            let after = cuda_deadline_f64_gemm_stats();
            assert!(
                after.calls >= before.calls.saturating_add(4)
                    && after.dispatches >= before.dispatches.saturating_add(4),
                "linear+conv forward and conv+linear adjoint must reach four bounded cuBLAS \
             dispatches: before={before:?}, after={after:?}"
            );
            eprintln!(
                "CUDA joint-alpha hardware dispatch: device={} transport={} calls_delta={} \
             dispatches_delta={}",
                engine.device_name(),
                engine.deadline_f64_transport_name(),
                after.calls - before.calls,
                after.dispatches - before.dispatches
            );
            let expected = ny_core::joint_alpha_grad::joint_alpha_gradient(
                &segments,
                &seed,
                &[0.0],
                1,
                2,
                &input_lower,
                &input_upper,
                ny_core::joint_alpha_grad::JointGradConfig::default(),
            )
            .expect("CPU oracle");
            assert_eq!(got.len(), expected.len(), "ReLU row count");
            for (row, (got, expected)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(got.len(), expected.len(), "ReLU row {row} width");
                for (neuron, (got, expected)) in got.iter().zip(expected).enumerate() {
                    assert!(
                        (got - expected).abs() < 1e-5,
                        "row={row} neuron={neuron}: CUDA={got} CPU oracle={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn cuda_deadline_resnet_routes_are_reachable_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            force_explicit_device_cuda_transport(engine);
            assert!(
                !engine.honors_crown_backward_deadline(),
                "the dedicated API must not globally advertise deadline behavior"
            );
            assert!(engine.provides_deadline_bounded_single_row_resnet_sound());
            assert_eq!(
                engine.deadline_bounded_resnet_sound_max_rows(),
                8,
                "explicit-device CUDA must advertise the full bounded-row capacity"
            );
            assert_eq!(
                engine.deadline_bounded_resnet_sound_beta_max_rows(),
                8,
                "explicit-device CUDA must separately advertise the bounded beta-row capacity"
            );

            let segments = vec![ny_core::GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(vec![2.0_f32]),
                    bias: None,
                    out_features: 1,
                    in_features: 1,
                    cert_err: Default::default(),
                },
            ])];
            let one_row = GpuCrownSeed {
                lower_a: Arc::from(vec![1.0_f32]),
                upper_a: Arc::from(vec![1.0_f32]),
                lower_b: Arc::from(vec![0.0_f32]),
                upper_b: Arc::from(vec![0.0_f32]),
                num_specs: 1,
                current_dim: 1,
            };
            let ordinary = engine
                .crown_backward_gpu_resnet_sound(&segments, &one_row, &[-1.0], &[1.0], &[], &[])
                .expect_err("ordinary tiny route must retain its resident-work gate");
            assert!(matches!(ordinary, NyError::UnsupportedOp(_)));

            let before = cuda_deadline_f64_gemm_stats();
            let bounded = engine
                .crown_backward_gpu_resnet_sound_single_row_with_deadline(
                    &segments,
                    &one_row,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    Instant::now() + std::time::Duration::from_secs(5),
                )
                .expect("dedicated one-row route must bypass only the resident-work gate");
            assert_eq!(bounded.lower_bounds.len(), 1);
            assert_eq!(bounded.upper_bounds.len(), 1);
            assert!(bounded.lower_bounds[0] <= -2.0);
            assert!(bounded.upper_bounds[0] >= 2.0);
            let after = cuda_deadline_f64_gemm_stats();
            assert!(
            after.calls >= before.calls + 3 && after.dispatches >= before.dispatches + 3,
            "one affine sound step must use three bounded deadline DGEMMs: before={before:?}, after={after:?}"
        );

            let compatible_one_row = engine
                .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                    &segments,
                    &one_row,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    Instant::now() + std::time::Duration::from_secs(5),
                )
                .expect("bounded-row K=1 must delegate to the legacy dedicated route");
            assert_eq!(
                compatible_one_row, bounded,
                "bounded-row K=1 must preserve the legacy result exactly"
            );

            let row_scales = [1.0_f32, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0];
            let eight_rows = GpuCrownSeed {
                lower_a: Arc::from(row_scales.to_vec()),
                upper_a: Arc::from(row_scales.to_vec()),
                lower_b: Arc::from(vec![0.0_f32; DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS]),
                upper_b: Arc::from(vec![0.0_f32; DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS]),
                num_specs: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                current_dim: 1,
            };
            let before_eight = cuda_deadline_f64_gemm_stats();
            let bounded_eight = engine
                .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                    &segments,
                    &eight_rows,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    Instant::now() + std::time::Duration::from_secs(5),
                )
                .expect("deadline-transport CUDA must execute the full bounded K=8 route");
            assert_eq!(
                bounded_eight.lower_bounds.len(),
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
            );
            assert_eq!(
                bounded_eight.upper_bounds.len(),
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
            );
            for (row, scale) in row_scales.into_iter().enumerate() {
                let lower = bounded_eight.lower_bounds[row];
                let upper = bounded_eight.upper_bounds[row];
                assert!(
                    lower.is_finite() && upper.is_finite() && lower <= upper,
                    "row {row} must return one finite ordered enclosure: [{lower}, {upper}]"
                );

                let expected_radius = 2.0 * scale.abs();
                let expected_lower = -expected_radius;
                let expected_upper = expected_radius;
                let tolerance = 32.0 * f32::EPSILON * expected_radius.max(1.0);
                assert!(
                lower <= expected_lower
                    && expected_lower - lower <= tolerance
                    && upper >= expected_upper
                    && upper - expected_upper <= tolerance,
                "row {row} scale {scale} must tightly enclose [{expected_lower}, {expected_upper}], got [{lower}, {upper}]"
            );
            }
            let after_eight = cuda_deadline_f64_gemm_stats();
            assert!(
            after_eight.calls >= before_eight.calls + 3
                && after_eight.dispatches >= before_eight.dispatches + 3,
            "one K=8 affine sound step must reach the bounded CUDA DGEMMs: before={before_eight:?}, after={after_eight:?}"
        );

            let bounded_beta_eight = engine
                .crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
                    &segments,
                    &eight_rows,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    &[],
                    Instant::now() + std::time::Duration::from_secs(5),
                )
                .expect("deadline-transport CUDA must execute the full bounded K=8 beta route");
            assert_eq!(
                bounded_beta_eight, bounded_eight,
                "an activation-free beta fold must equal the plain bounded fold"
            );

            let two_rows = GpuCrownSeed {
                lower_a: Arc::from(vec![1.0_f32, -1.0]),
                upper_a: Arc::from(vec![1.0_f32, -1.0]),
                lower_b: Arc::from(vec![0.0_f32, 0.0]),
                upper_b: Arc::from(vec![0.0_f32, 0.0]),
                num_specs: 2,
                current_dim: 1,
            };
            let wider = engine
                .crown_backward_gpu_resnet_sound_single_row_with_deadline(
                    &segments,
                    &two_rows,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    Instant::now() + std::time::Duration::from_secs(1),
                )
                .expect_err("dedicated route must refuse every wider seed");
            assert!(matches!(wider, NyError::UnsupportedOp(_)));

            let expired = engine
                .crown_backward_gpu_resnet_sound_single_row_with_deadline(
                    &segments,
                    &one_row,
                    &[-1.0],
                    &[1.0],
                    &[],
                    &[],
                    Instant::now()
                        .checked_sub(std::time::Duration::from_nanos(1))
                        .expect("one nanosecond must be representable"),
                )
                .expect_err("expired calls must refuse before any fold or dispatch");
            assert!(
                expired.is_deadline_exceeded(),
                "unexpected expired-call error: {expired}"
            );
        });
    }

    #[test]
    fn cuda_resident_cut_shadow_dispatches_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
            force_explicit_device_cuda_transport(engine);
            assert!(engine.provides_resident_cut_shadow());
            assert!(
                !engine.honors_crown_backward_deadline(),
                "the cut-specific explicit deadline must not broaden CUDA's general claim"
            );

            const ROWS: usize = 64;
            const WIDTH: usize = 512;
            let mut weight = vec![0.0_f32; WIDTH * WIDTH];
            for i in 0..WIDTH {
                weight[i * WIDTH + i] = 1.0;
            }
            // z0=x0+x1, z1=x0-x1: the exact diamond used by the host oracle.
            weight[1] = 1.0;
            weight[WIDTH] = 1.0;
            weight[WIDTH + 1] = -1.0;
            let segments = vec![ny_core::GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Activation {
                    lower_slope: vec![0.0; WIDTH],
                    upper_slope: vec![0.5; WIDTH],
                    lower_intercept: vec![0.0; WIDTH],
                    upper_intercept: vec![1.0; WIDTH],
                    num_neurons: WIDTH,
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(weight),
                    bias: None,
                    out_features: WIDTH,
                    in_features: WIDTH,
                    cert_err: Default::default(),
                },
            ])];
            let mut seed_rows = vec![0.0_f32; ROWS * WIDTH];
            for row in 0..ROWS {
                seed_rows[row * WIDTH] = -1.0;
                seed_rows[row * WIDTH + 1] = -1.0;
            }
            let seed = GpuCrownSeed {
                lower_a: Arc::from(seed_rows.clone()),
                upper_a: Arc::from(seed_rows),
                lower_b: Arc::from(vec![0.0_f32; ROWS]),
                upper_b: Arc::from(vec![0.0_f32; ROWS]),
                num_specs: ROWS,
                current_dim: WIDTH,
            };
            let input_lower = vec![-1.0_f32; WIDTH];
            let input_upper = vec![1.0_f32; WIDTH];
            let beta_signed = vec![vec![0.0_f32; WIDTH]];
            let baseline = engine
                .crown_backward_gpu_resnet_sound_beta(
                    &segments,
                    &seed,
                    &input_lower,
                    &input_upper,
                    &beta_signed,
                    &[],
                    &[],
                )
                .expect("ordinary CUDA beta baseline");

            let expired_deadline = Instant::now()
                .checked_sub(std::time::Duration::from_secs(1))
                .expect("one second of monotonic history");
            let disabled = engine
                .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                    ResidentCutShadowPolicy::Disabled,
                    &segments,
                    &seed,
                    &input_lower,
                    &input_upper,
                    &beta_signed,
                    &[],
                    &[],
                    None,
                    usize::MAX,
                    expired_deadline,
                )
                .expect("disabled CUDA cut route");
            assert_eq!(
                disabled.disposition(),
                ny_core::ResidentCutShadowDisposition::Disabled
            );
            for (got, expected) in disabled
                .baseline()
                .lower_bounds
                .iter()
                .chain(&disabled.baseline().upper_bounds)
                .zip(baseline.lower_bounds.iter().chain(&baseline.upper_bounds))
            {
                assert_eq!(got.to_bits(), expected.to_bits());
            }

            let channel = |value| {
                ny_core::ResidentLowerCutChannel::try_new(value, 0.0).expect("exact test channel")
            };
            let carrier_at = |deadline| {
                ResidentLowerCutCarrier::try_new(
                    0,
                    WIDTH,
                    [0, 1],
                    (0..ROWS)
                        .map(|_| {
                            ny_core::ResidentLowerCutRow::try_new(
                                vec![1.0],
                                [channel(-0.5), channel(-0.5)],
                                [channel(1.0), channel(1.0)],
                                channel(-1.0),
                            )
                            .expect("complete diamond cut row")
                        })
                        .collect(),
                    deadline,
                )
                .expect("complete diamond cut carrier")
            };
            let deadline = Instant::now() + std::time::Duration::from_secs(30);
            let carrier = carrier_at(deadline);
            let before = cuda_deadline_f64_gemm_stats();
            let observed = engine
                .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                    ResidentCutShadowPolicy::Shadow,
                    &segments,
                    &seed,
                    &input_lower,
                    &input_upper,
                    &beta_signed,
                    &[],
                    &[],
                    Some(&carrier),
                    0,
                    deadline,
                )
                .expect("complete resident cut observation");
            let after = cuda_deadline_f64_gemm_stats();
            assert_eq!(
                observed.disposition(),
                ny_core::ResidentCutShadowDisposition::Observed
            );
            let observation = observed.observation().expect("complete telemetry");
            assert!(
                observation.delta() > 0.5,
                "diamond cut must materially tighten the observed lower row: {observation:?}"
            );
            assert!(
                after.calls >= before.calls.saturating_add(6)
                    && after.dispatches >= before.dispatches.saturating_add(6),
                "bounded baseline + cut folds must each reach center/magnitude/error DGEMMs: \
             before={before:?}, after={after:?}"
            );
            for (got, expected) in observed
                .baseline()
                .lower_bounds
                .iter()
                .chain(&observed.baseline().upper_bounds)
                .zip(baseline.lower_bounds.iter().chain(&baseline.upper_bounds))
            {
                assert_eq!(got.to_bits(), expected.to_bits());
            }

            let expired_carrier = carrier_at(expired_deadline);
            let bounded_before_expired = cuda_deadline_f64_gemm_stats();
            let rejected = engine
                .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                    ResidentCutShadowPolicy::Shadow,
                    &segments,
                    &seed,
                    &input_lower,
                    &input_upper,
                    &beta_signed,
                    &[],
                    &[],
                    Some(&expired_carrier),
                    0,
                    expired_deadline,
                )
                .expect_err("late shadow must return before any replay or mutation");
            let bounded_after_expired = cuda_deadline_f64_gemm_stats();
            assert!(rejected.is_deadline_exceeded());
            assert_eq!(
                bounded_after_expired.calls, bounded_before_expired.calls,
                "expired cut must start no bounded shadow call"
            );
            assert_eq!(
                bounded_after_expired.dispatches, bounded_before_expired.dispatches,
                "expired cut must start no bounded shadow dispatch"
            );
        });
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
    fn cuda_gemm_matches_cpu_f32_and_f64_when_hardware_is_capable() {
        with_capable_cuda(|engine| {
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
        });
    }
}
