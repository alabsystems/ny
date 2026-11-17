// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Native ordered paired-f32 root contraction in an isolated CUDA context.
//! An engine exists only after its actual module and transport pass literal-bit
//! known-answer checks. CPU root error/bias/fold authority remains at the caller.

use cudarc::driver::{
    result as drv, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use ny_core::packed_root_conv::{check_deadline, Engine, Layout, Limits, Output, Request};
use ny_core::{NyError, Result};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;

mod qualification;
mod telemetry;
#[cfg(test)]
mod tests;

const MANIFEST: &str = include_str!("../kernels/packed_root_kernels.manifest");
const SOURCE: &[u8] = include_bytes!("../kernels/packed_root_conv.cu");
const FIXTURES: &[u8] = include_bytes!("../kernels/packed_root_kat.bin");
const GUARD_WORDS: usize = 32;
const GUARD: u32 = 0x7fc0_0123;
const POISON: u32 = 0x7fc1_2345;
const GUARD_BYTES: usize = 7 * 2 * GUARD_WORDS * 4;
const COPY_WORDS: usize = 1 << 18;
const MAX_STEPS: usize = 1 << 28;
static COMPLETED: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static WALL_US: AtomicU64 = AtomicU64::new(0);

/// Completed production arithmetic and all production dispatch attempts.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Calls which published both output lanes.
    pub completed: u64,
    /// Actual kernel dispatch attempts, excluding startup qualification.
    pub dispatches: u64,
    /// Total wall time for production calls which published outputs.
    pub wall_us: u64,
}
/// Observe counters without initializing an engine or performing GPU work.
pub fn stats() -> Stats {
    Stats {
        completed: COMPLETED.load(Ordering::Relaxed),
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        wall_us: WALL_US.load(Ordering::Relaxed),
    }
}

fn error(message: impl Into<String>) -> NyError {
    NyError::InternalError(message.into())
}
fn manifest(key: &str) -> Result<&'static str> {
    MANIFEST
        .lines()
        .find_map(|line| {
            line.split_once('=')
                .filter(|(k, _)| *k == key)
                .map(|(_, v)| v)
        })
        .ok_or_else(|| error(format!("packed CUDA manifest missing {key}")))
}
fn authenticate(bytes: &[u8], key: &str) -> Result<()> {
    if format!("{:x}", Sha256::digest(bytes)) != manifest(key)? {
        return Err(error(format!("packed CUDA artifact mismatch: {key}")));
    }
    Ok(())
}
fn authenticate_contract() -> Result<()> {
    if manifest("format")? != "1"
        || manifest("abi")? != "seven-pointers-six-u32-two-u64"
        || manifest("entry")? != "packed_root_conv_ordered"
        || manifest("flags")?
            != "--std=c++17 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true"
        || manifest("runtime_compilation")? != "false"
        || manifest("ptx_fallback")? != "false"
    {
        return Err(error("packed CUDA arithmetic/ABI manifest mismatch"));
    }
    Ok(())
}
fn image(sm: i32) -> Option<&'static [u8]> {
    match sm {
        86 => Some(include_bytes!("../kernels/packed_root_conv.sm86.cubin")),
        120 => Some(include_bytes!("../kernels/packed_root_conv.sm120.cubin")),
        121 => Some(include_bytes!("../kernels/packed_root_conv.sm121.cubin")),
        _ => None,
    }
}

// Only the retained driver is needed. A cuBLAS KAT cannot authorize this module.
fn driver_admission() -> Result<()> {
    std::panic::catch_unwind(|| {
        // SAFETY: initialize and retain cudarc's process-global driver handle.
        let _ = unsafe { cudarc::driver::sys::culib() };
    })
    .map_err(|_| error("packed CUDA driver unavailable"))?;
    #[cfg(target_os = "linux")]
    {
        const SYMBOLS: &[&str] = &[
            "cuInit",
            "cuDeviceGet",
            "cuDeviceGetAttribute",
            "cuDeviceGetName",
            "cuCtxCreate_v4",
            "cuCtxDestroy_v2",
            "cuCtxGetCurrent",
            "cuCtxSetCurrent",
            "cuMemAlloc_v2",
            "cuMemFree_v2",
            "cuMemGetInfo_v2",
            "cuMemcpyHtoD_v2",
            "cuMemcpyDtoH_v2",
            "cuStreamSynchronize",
            "cuModuleLoadData",
            "cuModuleGetFunction",
            "cuModuleUnload",
            "cuLaunchKernel",
        ];
        let addresses =
            super::resolve_role_symbol_addresses(super::CudaRuntimeRole::Driver, SYMBOLS)?;
        let maps = std::fs::read_to_string("/proc/self/maps").map_err(super::cuda_err)?;
        super::validate_role_symbol_addresses(
            &maps,
            super::CudaRuntimeRole::Driver,
            &addresses,
            super::confirm_dladdr_provider,
        )?;
    }
    drv::init().map_err(super::cuda_err)
}

struct RestoreContext(cudarc::driver::sys::CUcontext);
impl RestoreContext {
    fn save() -> Result<Self> {
        Ok(Self(
            drv::ctx::get_current()
                .map_err(super::cuda_err)?
                .unwrap_or(std::ptr::null_mut()),
        ))
    }
}
impl Drop for RestoreContext {
    fn drop(&mut self) {
        // SAFETY: this caller-owned context was current before our serialized
        // operation. Our independently owned context never destroys it.
        if unsafe { drv::ctx::set_current(self.0) }.is_err() {
            // A context restoration failure cannot masquerade as a successful
            // transaction or unwind into unrelated CUDA users on this thread.
            std::process::abort();
        }
    }
}

struct State {
    // Drop module, then stream, then final context reference, while the owning
    // engine's context-restoration guard is still alive.
    function: CudaFunction,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    faulted: AtomicBool,
}

/// Qualified capability. Initialization is explicit and must precede deadlines.
pub struct PackedRootEngine {
    state: Mutex<Option<State>>,
}
impl PackedRootEngine {
    /// Qualify the selected native image and actual copy/launch path. Unsupported
    /// architectures return None and keep the compact CPU implementation.
    pub fn new(ordinal: usize) -> Result<Option<Self>> {
        driver_admission()?;
        let _restore = RestoreContext::save()?;
        authenticate(SOURCE, "source_sha256")?;
        authenticate(FIXTURES, "fixtures_sha256")?;
        authenticate_contract()?;
        let ctx = CudaContext::new_non_primary(ordinal, 0).map_err(super::cuda_err)?;
        let (major, minor) = ctx.compute_capability().map_err(super::cuda_err)?;
        let sm = major * 10 + minor;
        let Some(image) = image(sm) else {
            return Ok(None);
        };
        authenticate(image, &format!("sm_{sm}_sha256"))?;
        let binary = image.to_vec();
        if binary.as_ptr().align_offset(8) != 0 || !binary.starts_with(b"\x7fELF") {
            return Err(error("packed CUDA native image alignment/format mismatch"));
        }
        let module = ctx
            .load_module(Ptx::from_binary(binary))
            .map_err(super::cuda_err)?;
        let function = module
            .load_function("packed_root_conv_ordered")
            .map_err(super::cuda_err)?;
        let stream = ctx.default_stream();
        let state = State {
            function,
            stream,
            ctx,
            faulted: AtomicBool::new(false),
        };
        qualification::qualify(&state)?;
        tracing::info!(sm, "packed root CUDA module and transport qualified");
        Ok(Some(Self {
            state: Mutex::new(Some(state)),
        }))
    }
}
impl Drop for PackedRootEngine {
    fn drop(&mut self) {
        let _restore = RestoreContext::save().unwrap_or_else(|_| std::process::abort());
        // Explicitly destroy CUDA fields before restoring the caller context.
        // Automatic field teardown after a Drop body would rebind our context.
        drop(
            self.state
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take(),
        );
    }
}
impl Engine for PackedRootEngine {
    fn contract(&self, request: &Request<'_>, limits: Limits) -> Result<Option<Output>> {
        check_deadline(limits.deadline)?;
        let started = Instant::now();
        let guard = loop {
            check_deadline(limits.deadline)?;
            match self.state.try_lock() {
                Ok(g) => break g,
                Err(TryLockError::WouldBlock) => std::thread::yield_now(),
                Err(TryLockError::Poisoned(_)) => return Err(error("packed CUDA mutex poisoned")),
            }
        };
        let _restore = RestoreContext::save()?;
        let state = guard
            .as_ref()
            .ok_or_else(|| error("packed CUDA engine closed"))?;
        let result = state.execute(request, limits, Audit::production())?;
        if result.is_some() {
            COMPLETED.fetch_add(1, Ordering::Relaxed);
            WALL_US.fetch_add(
                started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct Audit<'a> {
    expected: Option<[&'a [u32]; 2]>,
    chunked: bool,
    #[cfg(test)]
    invalid_launch: bool,
    #[cfg(test)]
    expire_after_launch: bool,
}
impl Audit<'_> {
    fn production() -> Self {
        Self {
            expected: None,
            chunked: false,
            #[cfg(test)]
            invalid_launch: false,
            #[cfg(test)]
            expire_after_launch: false,
        }
    }
}

// Every possibly queued transaction gets an unwind guard. An error does not
// waive drain; unproven quiescence uses the crate's fail-hard policy.
struct Pending<'a> {
    state: &'a State,
    armed: bool,
}
impl Pending<'_> {
    fn finish(mut self, result: Result<()>) -> Result<()> {
        let drained = self.state.drain(result);
        self.armed = false;
        drained
    }
}
impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.state.faulted.store(true, Ordering::Release);
            let _ = self.state.drain(Ok(()));
        }
    }
}

struct Buffer<'a> {
    state: &'a State,
    base: u64,
    words: usize,
}
impl Buffer<'_> {
    fn pointer(&self) -> u64 {
        if self.words == 0 {
            0
        } else {
            self.base + (GUARD_WORDS * 4) as u64
        }
    }
    fn close(&mut self) -> Result<()> {
        let pointer = std::mem::take(&mut self.base);
        if pointer == 0 {
            return Ok(());
        }
        self.state.device(self.state.ctx.bind_to_thread())?;
        // SAFETY: all submissions have drained; this handle is retired before
        // its sole free attempt, including on an error or later Drop.
        self.state.device(unsafe { drv::free_sync(pointer) })
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn allocate<T: Copy>(len: usize, fill: T, deadline: Option<Instant>) -> Result<Vec<T>> {
    check_deadline(deadline)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: len.saturating_mul(size_of::<T>()),
            budget_bytes: 0,
            site: "packed-root-host-payload",
        })?;
    while output.len() < len {
        check_deadline(deadline)?;
        output.resize(len.min(output.len() + COPY_WORDS), fill);
    }
    check_deadline(deadline)?;
    Ok(output)
}

impl State {
    fn device<T>(&self, result: std::result::Result<T, cudarc::driver::DriverError>) -> Result<T> {
        result.map_err(|e| {
            self.faulted.store(true, Ordering::Release);
            super::cuda_err(e)
        })
    }
    fn drain(&self, result: Result<()>) -> Result<()> {
        let drain = super::prove_cuda_stream_quiescent(
            || self.ctx.bind_to_thread().is_ok(),
            || {
                // SAFETY: the live stream's context was bound by the preceding callback.
                unsafe { drv::stream::synchronize(self.stream.cu_stream()).is_ok() }
            },
        );
        let result = super::queued_gemm_numeric_result(result, drain);
        if result.is_err() {
            self.faulted.store(true, Ordering::Release);
        }
        result
    }
    fn upload<T: Copy>(&self, pointer: u64, data: &[T], deadline: Option<Instant>) -> Result<()> {
        for (i, chunk) in data.chunks(COPY_WORDS).enumerate() {
            check_deadline(deadline)?;
            let pending = Pending {
                state: self,
                armed: true,
            };
            // SAFETY: the call's checked allocation contains the complete slice.
            // The borrowed host slice stays alive until mandatory drain finishes.
            let result = self.device(unsafe {
                drv::memcpy_htod_sync(pointer + (i * COPY_WORDS * size_of::<T>()) as u64, chunk)
            });
            pending.finish(result)?;
            check_deadline(deadline)?;
        }
        Ok(())
    }
    fn download<T: Copy>(
        &self,
        pointer: u64,
        data: &mut [T],
        deadline: Option<Instant>,
    ) -> Result<()> {
        for (i, chunk) in data.chunks_mut(COPY_WORDS).enumerate() {
            check_deadline(deadline)?;
            let pending = Pending {
                state: self,
                armed: true,
            };
            // SAFETY: exact-sized host slice and device allocation; drain precedes
            // return/unwind so CUDA cannot retain the borrowed destination.
            let result = self.device(unsafe {
                drv::memcpy_dtoh_sync(chunk, pointer + (i * COPY_WORDS * size_of::<T>()) as u64)
            });
            pending.finish(result)?;
            check_deadline(deadline)?;
        }
        Ok(())
    }
    fn buffer(&self, words: usize, deadline: Option<Instant>) -> Result<Buffer<'_>> {
        check_deadline(deadline)?;
        if words == 0 {
            return Ok(Buffer {
                state: self,
                base: 0,
                words,
            });
        }
        let bytes = words
            .checked_add(2 * GUARD_WORDS)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("packed CUDA guarded allocation overflow"))?;
        // SAFETY: raw allocation size is checked; there is no queued async alloc.
        let base = self.device(unsafe { drv::malloc_sync(bytes) })?;
        let buffer = Buffer {
            state: self,
            base,
            words,
        };
        self.upload(base, &[GUARD; GUARD_WORDS], deadline)?;
        self.upload(
            buffer.pointer() + (words * 4) as u64,
            &[GUARD; GUARD_WORDS],
            deadline,
        )?;
        Ok(buffer)
    }
    fn guards(&self, buffers: &[Buffer<'_>], deadline: Option<Instant>) -> Result<()> {
        for buffer in buffers {
            if buffer.words == 0 {
                continue;
            }
            for ptr in [buffer.base, buffer.pointer() + (buffer.words * 4) as u64] {
                let mut guard = [0u32; GUARD_WORDS];
                self.download(ptr, &mut guard, deadline)?;
                if guard != [GUARD; GUARD_WORDS] {
                    self.faulted.store(true, Ordering::Release);
                    return Err(error("packed CUDA allocation guard changed"));
                }
            }
        }
        Ok(())
    }
    fn launch(
        &self,
        request: &Request<'_>,
        buffers: &[Buffer<'_>],
        first: usize,
        count: usize,
        audit: Audit<'_>,
    ) -> Result<()> {
        let pointers: [u64; 7] = std::array::from_fn(|i| buffers[i].pointer());
        let dims = [
            request.source_stride,
            request.output_stride,
            request.output_spatial,
            request.output_channels,
            request.spatial_taps,
            request.taps,
        ]
        .map(|n| n as u32);
        let (first, count) = (first as u64, count as u64);
        let mut launch = self.stream.launch_builder(&self.function);
        for pointer in &pointers {
            launch.arg(pointer);
        }
        for dim in &dims {
            launch.arg(dim);
        }
        launch.arg(&first).arg(&count);
        let cfg = LaunchConfig {
            grid_dim: ((count.max(1).div_ceil(128)) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        #[cfg(test)]
        let cfg = if audit.invalid_launch {
            LaunchConfig {
                block_dim: (0, 1, 1),
                ..cfg
            }
        } else {
            cfg
        };
        if audit.expected.is_none() {
            DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }
        let pending = Pending {
            state: self,
            armed: true,
        };
        // SAFETY: validated dimensions/maps, seven distinct guarded allocations,
        // exact fifteen-argument ABI, disjoint in-range owner partitions. Scalar
        // argument storage and the loaded module remain live through the drain.
        let result = self.device(unsafe { launch.launch(cfg) }).map(|_| ());
        pending.finish(result)
    }
    fn execute(&self, r: &Request<'_>, limits: Limits, audit: Audit<'_>) -> Result<Option<Output>> {
        let mut trace = telemetry::Trace::new(r, limits, audit);
        let result = self.execute_inner(r, limits, audit, &trace);
        trace.finish(&result);
        result
    }
    fn execute_inner(
        &self,
        r: &Request<'_>,
        mut limits: Limits,
        audit: Audit<'_>,
        trace: &telemetry::Trace<'_>,
    ) -> Result<Option<Output>> {
        let admission = trace.phase(0);
        check_deadline(limits.deadline)?;
        if self.faulted.load(Ordering::Acquire) {
            return Err(error("packed CUDA capability faulted"));
        }
        limits.max_dispatch_steps = limits.max_dispatch_steps.min(MAX_STEPS);
        let Some(payload_limit) = limits.max_device_bytes.checked_sub(GUARD_BYTES) else {
            return Ok(None);
        };
        limits.max_device_bytes = payload_limit;
        let Some(layout) = r.layout(limits)? else {
            return Ok(None);
        };
        self.device(self.ctx.bind_to_thread())?;
        let (free, _) = self.device(drv::mem_get_info())?;
        if layout.device_bytes + GUARD_BYTES > free / 2 {
            return Ok(None);
        }
        check_deadline(limits.deadline)?;
        drop(admission);
        let (gather, inverse) = trace.measure(1, || maps(r, layout, limits.deadline))?;
        let (mut output, mut buffers) = trace.measure(2, || -> Result<_> {
            let output = Output {
                m: allocate(layout.output_len, f32::from_bits(POISON), limits.deadline)?,
                d: allocate(layout.output_len, f32::from_bits(POISON), limits.deadline)?,
            };
            let mut buffers = Vec::with_capacity(7);
            for words in [
                layout.source_len,
                layout.source_len,
                r.weights.len(),
                layout.gather_len,
                layout.inverse_len,
                layout.output_len,
                layout.output_len,
            ] {
                buffers.push(self.buffer(words, limits.deadline)?);
            }
            self.upload(buffers[0].pointer(), r.m, limits.deadline)?;
            self.upload(buffers[1].pointer(), r.d, limits.deadline)?;
            self.upload(buffers[2].pointer(), r.weights, limits.deadline)?;
            self.upload(buffers[3].pointer(), &gather, limits.deadline)?;
            self.upload(buffers[4].pointer(), &inverse, limits.deadline)?;
            Ok((output, buffers))
        })?;
        if audit.expected.is_some() {
            self.upload(buffers[5].pointer(), &output.m, limits.deadline)?;
            self.upload(buffers[6].pointer(), &output.d, limits.deadline)?;
            self.launch(r, &buffers, 0, 0, audit)?;
            qualification::check_chunk(self, r, &buffers, &mut output, audit, 0)?;
        }
        let mut first = 0;
        let mut dispatch = 0;
        while first < layout.output_len {
            check_deadline(limits.deadline)?;
            // The hard paired-work cap implies <=2^27 owners, well inside the
            // CUDA x-grid limit. Never split the tap sequence of an owner.
            let mut count = (layout.output_len - first).min(layout.owners_per_dispatch);
            if audit.chunked && dispatch < 5 {
                count = count.min([1, 7, 13, 2, 7][dispatch]);
            } else if audit.chunked && r.output_spatial > 1 {
                // Force a later launch to START in the second spatial site,
                // not merely to cross it during one large final dispatch.
                let boundary = r.output_channels * r.output_stride + 1;
                if first < boundary {
                    count = count.min(boundary - first);
                }
            }
            trace.dispatch(count, || self.launch(r, &buffers, first, count, audit))?;
            check_deadline(limits.deadline)?;
            #[cfg(test)]
            if audit.expire_after_launch {
                return Err(NyError::DeadlineExceeded(
                    "injected after drained packed launch".into(),
                ));
            }
            first += count;
            dispatch += 1;
            if audit.expected.is_some() {
                qualification::check_chunk(self, r, &buffers, &mut output, audit, first)?;
            }
        }
        trace.measure(4, || -> Result<()> {
            self.download(buffers[5].pointer(), &mut output.m, limits.deadline)?;
            self.download(buffers[6].pointer(), &mut output.d, limits.deadline)
        })?;
        let checks = trace.phase(5);
        self.guards(&buffers, limits.deadline)?;
        if audit.expected.is_some() {
            qualification::check_inputs(self, &buffers, r, &gather, &inverse)?;
        }
        for chunk in output.m.chunks(16384).chain(output.d.chunks(16384)) {
            check_deadline(limits.deadline)?;
            if chunk
                .iter()
                .any(|v| v.to_bits() & 0x7f80_0000 == 0x7f80_0000)
            {
                self.faulted.store(true, Ordering::Release);
                return Err(NyError::NumericalInstability(
                    "packed CUDA nonfinite output".into(),
                ));
            }
        }
        drop(checks);
        let cleanup = trace.phase(6);
        let mut close_error = None;
        for buffer in &mut buffers {
            if let Err(e) = buffer.close() {
                if close_error.is_none() {
                    close_error = Some(e);
                }
            }
        }
        if let Some(e) = close_error {
            return Err(e);
        }
        drop(cleanup);
        check_deadline(limits.deadline)?;
        Ok(Some(output))
    }
}

fn maps(
    r: &Request<'_>,
    layout: Layout,
    deadline: Option<Instant>,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut gather = allocate(layout.gather_len, u32::MAX, deadline)?;
    for (i, (&src, dst)) in r.gather.iter().zip(&mut gather).enumerate() {
        if i % 16384 == 0 {
            check_deadline(deadline)?;
        }
        *dst = if src == usize::MAX {
            u32::MAX
        } else {
            src as u32
        };
    }
    let mut inverse = allocate(layout.inverse_len, u32::MAX, deadline)?;
    let slots = r.output_stride - 1;
    for bucket in 0..r.offsets.len() - 1 {
        check_deadline(deadline)?;
        let mut source = 0;
        for run in &r.runs[r.offsets[bucket]..r.offsets[bucket + 1]] {
            for i in 0..run.len {
                if i % 16384 == 0 {
                    check_deadline(deadline)?;
                }
                inverse[bucket * slots + run.dest + i] = (source + i) as u32;
            }
            source += run.len;
        }
    }
    Ok((gather, inverse))
}
